# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""SENTINEL-Orbit teacher interface for M7 SENTINEL-Lite (orbit_protocol.md §8; docs/model_cards/orbit.md).

Imagery only — no in-situ labels are read. Given S2 L2A chips from a chip store (e.g.
processed/s2_chips/naip_teacher), returns per-chip predictive distributions from the SatWQ-Net seed
ensemble, for all bands and for the RGB+NIR subset (B02/B03/B04/B08), with and without context, and
optionally per-pixel maps (every valid SCL-water pixel, each pixel as a 1-pixel set).

    from sentinel.models.orbit.teacher import OrbitTeacher
    t = OrbitTeacher.from_manifest()                      # frozen teacher (docs/model_cards/orbit.md)
    df = t.predict_store(paths.processed("s2_chips") / "naip_teacher")    # one row per chip
    maps = t.predict_pixels(chip, meta_row)               # dict target -> [64, 64] log10 median (NaN off-water)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ...core import paths
from ...data import chips as C
from ...data.sources import stac as S
from . import data as D
from . import pipeline as PL

TEACHER_MANIFEST = paths.REPO_ROOT / "docs" / "model_cards" / "orbit_teacher.json"
QS = (0.16, 0.5, 0.84)


class OrbitTeacher:
    def __init__(self, ckpts: list[str | Path], device: str | None = None):
        import torch

        from .satwq import SatWQNet

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.models, self.norms = [], []
        for c in ckpts:
            ck = torch.load(c, map_location="cpu", weights_only=False)
            m = SatWQNet(**ck["arch"]).to(self.device)
            m.load_state_dict(ck["state_dict"])
            m.eval()
            self.models.append(m)
            self.norms.append(PL.Norm.from_dict(ck["norm"]))
        self.ckpts = [str(c) for c in ckpts]

    @classmethod
    def from_manifest(cls, path: Path = TEACHER_MANIFEST, verify: bool = True, **kw):
        man = json.loads(Path(path).read_text())
        if verify:
            from .guard import artifact_sha256

            for c, sha in zip(man["checkpoints"], man["sha256"]):
                if artifact_sha256(c) != sha:
                    raise RuntimeError(f"teacher checkpoint {c} sha256 mismatch")
        return cls(man["checkpoints"], **kw)

    # -------------------------------------------------------------------------------- chip level
    def _mixture(self, ds_fn, *, ctx_off: bool, band_subset=None):
        from .satwq import predict_mixture

        ws, mus, sgs = [], [], []
        for m, nm in zip(self.models, self.norms):
            w, mu, sg = predict_mixture(m, ds_fn(nm), self.device, nm.y_mean, nm.y_std, ctx_off=ctx_off,
                                        band_subset=band_subset)
            ws.append(w / len(self.models))
            mus.append(mu)
            sgs.append(sg)
        return np.concatenate(ws, -1), np.concatenate(mus, -1), np.concatenate(sgs, -1)

    @staticmethod
    def summarise(w, mu, sg, prefix: str) -> dict[str, np.ndarray]:
        from .satwq import mixture_quantiles

        out = {}
        wn = w / w.sum(-1, keepdims=True)
        for j, t in enumerate(D.TARGETS):
            q = mixture_quantiles(wn[:, j], mu[:, j], sg[:, j], QS)
            mean = (wn[:, j] * mu[:, j]).sum(-1)
            var = (wn[:, j] * (sg[:, j] ** 2 + mu[:, j] ** 2)).sum(-1) - mean ** 2
            out[f"{prefix}{t}_log10_q16"], out[f"{prefix}{t}_log10_q50"], out[f"{prefix}{t}_log10_q84"] = q.T
            out[f"{prefix}{t}_log10_mean"] = mean
            out[f"{prefix}{t}_log10_var"] = var
        return out

    def predict_table(self, meta: pd.DataFrame, pix: dict) -> pd.DataFrame:
        """``meta``: chip-index rows (needs platform, sensing_time_utc, lat/lon, angles, aot_c3, jrc_dist_land_m,
        scl_c3_n_water); ``pix``: :func:`data.chip_pixels` arrays. Returns one row per chip."""
        meta = meta.copy()
        t = pd.to_datetime(meta.sensing_time_utc, utc=True, errors="coerce")
        meta["doy"] = t.dt.dayofyear.to_numpy()
        n = len(meta)
        dummy = pd.DataFrame({k: np.full(n, np.nan) for k in D.TARGETS}, index=meta.index)
        dfx = pd.concat([meta, dummy], axis=1)

        def ds(nm):
            return PL.make_set(dfx, pix, nm)
        out = {}
        for tag, kw in (("", {"ctx_off": False}), ("imgonly_", {"ctx_off": True}),
                        ("rgbn_imgonly_", {"ctx_off": True, "band_subset": D.RGBN})):
            out.update(self.summarise(*self._mixture(ds, **kw), tag))
        res = pd.DataFrame(out, index=meta.index)
        res.insert(0, "n_water_pixels", pix["n_pix"])
        return res

    def predict_store(self, store_root: Path, chip_ids=None, workers: int = 6) -> pd.DataFrame:
        idx = C.ChipStore(store_root).index
        idx = idx[idx.status == "ok"].copy()
        if chip_ids is not None:
            idx = idx[idx.chip_id.isin(set(chip_ids))]
        idx["chip_store"] = Path(store_root).name
        need = ["sun_zenith", "sun_azimuth", "view_zenith", "view_azimuth", "aot_c3", "jrc_dist_land_m", "scl_c3_n_water"]
        missing = [c for c in need + ["lat", "lon", "platform", "sensing_time_utc", "shard"] if c not in idx.columns]
        if missing:
            raise KeyError(f"chip index lacks {missing}")
        pix = _extract_any_store(idx, workers)
        res = self.predict_table(idx.reset_index(drop=True), pix)
        res.insert(0, "chip_id", idx.chip_id.to_numpy())
        res.loc[pix["n_pix"] == 0, [c for c in res.columns if "_log10_" in c]] = np.nan
        return res

    # -------------------------------------------------------------------------------- pixel level
    def predict_pixels(self, chip: np.ndarray, meta_row: pd.Series, *, ctx_off: bool = True,
                       band_subset=None) -> dict[str, np.ndarray]:
        """Per-pixel log10 median maps [64, 64] for every valid SCL-water pixel (each pixel predicted as a
        1-pixel set — in distribution because training samples random pixel subsets down to 1 pixel)."""
        L = {n: i for i, n in enumerate(C.S2_STORE_LAYERS)}
        dn = np.stack([chip[L[b]] for b in D.BANDS])
        ok = (chip[L["SCL"]] == 6) & np.all(dn != S.NODATA_I16, 0)
        r, c = np.nonzero(ok)
        n = r.size
        maps = {t: np.full(chip.shape[1:], np.nan, np.float32) for t in D.TARGETS}
        if not n:
            return maps
        refl = (dn[:, r, c].T.astype(np.float32) * 1e-4)[:, None, :]
        pix = {"refl": refl.astype(np.float16), "offsets": np.zeros((n, 1, 2), np.int8), "n_pix": np.ones(n, np.int32),
               "n_water_win": np.full(n, int(ok[25:40, 25:40].sum()), np.int32)}
        meta = pd.DataFrame([meta_row] * n).reset_index(drop=True)
        meta["doy"] = pd.to_datetime(meta.sensing_time_utc, utc=True).dt.dayofyear
        for t in D.TARGETS:
            meta[t] = np.nan

        def ds(nm):
            return PL.make_set(meta, pix, nm)
        w, mu, sg = self._mixture(ds, ctx_off=ctx_off, band_subset=band_subset)
        s = self.summarise(w, mu, sg, "")
        for t in D.TARGETS:
            maps[t][r, c] = s[f"{t}_log10_q50"]
        return maps


def _extract_any_store(idx: pd.DataFrame, workers: int) -> dict:
    from concurrent.futures import ProcessPoolExecutor

    jobs = [(st, sh, g.chip_id.tolist()) for (st, sh), g in idx.groupby(["chip_store", "shard"])]
    res = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for part in ex.map(D._extract_shard, jobs):
            for cid, refl, off, n, nw in part:
                res[cid] = (refl, off, n, nw)
    ids = idx.chip_id.tolist()
    return {"refl": np.stack([res[c][0] for c in ids]).astype(np.float16),
            "offsets": np.stack([res[c][1] for c in ids]).astype(np.int8),
            "n_pix": np.array([res[c][2] for c in ids], np.int32), "n_water_win": np.array([res[c][3] for c in ids], np.int32)}
