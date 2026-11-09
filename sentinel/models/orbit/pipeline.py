# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Shared plumbing for scripts/train/orbit_*.py: development data with role masks, model arrays with
fit-only normalisation, GPU lock (shared helper, prefer GPU 1), TrainingRun wrapping, configuration log, prediction I/O."""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ...core import paths
from ...core.registry import config_hash
from . import data as D
from . import features as F

WORKSTREAM = "M2-orbit"
PLAN_REF = "RESEARCH_PLAN.md §5.2 (M2 SENTINEL-Orbit); docs/preregistration/orbit_protocol.md"
GPU_PREFER = 1
# Development evaluation cells (orbit_protocol.md §6). Model selection cell first; in-lake rows mirror the
# headline (test role x test years, in-lake). Buffer rows never scored. The seen-lake cell excludes inner
# blocks (no model was fitted on them; review W3). TSS/turbidity are also summarised on the pooled
# unseen-lake cell (val role x non-test years) because the val-year cell holds < 50 rows for them.
EVAL_CELLS = ("valxval_lake", "valxany_lake", "valxval_nonlake", "trainxval_lake")
PRIMARY_CELL = "valxval_lake"
GEO_COLS = ("cos_sza", "cos_vza", "cos_raa", "aot_c3", "log_dist_land", "n_water_win", "scl_c3_n_water")
CTX_COLS = ("doy_sin", "doy_cos", "ux", "uy", "uz")


def runs_root() -> Path:
    return paths.ensure(paths.RUNS_ROOT / "orbit")


def setup_gpu(name: str, poll_s: float = 30.0, log=print, min_free_mb: int = 3000) -> int:
    """Lock one GPU through the shared lock helper (``sentinel.core.gpu.acquire_gpu``; hard cap
    ``.gpu_locks/ALLOWED`` = GPUs 0,1 since 2026-11-08 16:00). Prefers GPU 1, waits for a free allowed GPU
    and lets the helper set CUDA_VISIBLE_DEVICES. Each Orbit GPU job is one process <= 2 h (one seed, one
    baseline run, one compare), so chains release the lock between jobs (CONTRIBUTING_V2)."""
    from ...core import gpu as gpu_mod

    if gpu_mod._held:
        raise RuntimeError("a GPU is already held by this process")
    log(f"[gpu] requesting a GPU (allowed {gpu_mod.ALLOWED}, prefer {GPU_PREFER}): {gpu_mod.status()}")
    g = gpu_mod.acquire_gpu(name, poll_s=poll_s, prefer=GPU_PREFER, min_free_mb=min_free_mb)
    log(f"[gpu] locked GPU {g}")
    return g


def load_dev(version: str = D.CACHE_VERSION, fit_scheme: str = "base"):
    df, pix, info = D.load_dev(version, training=True)
    fit, inner = F.fit_and_inner(df, fit_scheme)
    masks = {"fit": fit, "inner": inner}
    ok = ~df.buffer_excluded.to_numpy(bool)
    lake = df.in_lake.to_numpy(bool)
    role, tb = df.role.to_numpy(), df.tblock.to_numpy()
    masks["valxval_lake"] = (role == "val") & (tb == "val") & lake & ok
    masks["valxany_lake"] = (role == "val") & np.isin(tb, ["train", "val"]) & lake & ok
    masks["valxval_nonlake"] = (role == "val") & (tb == "val") & ~lake & ok
    masks["trainxval_lake"] = (role == "train") & (tb == "val") & lake & ok & ~F.inner_mask(df) & ~fit
    if not masks["trainxval_lake"].any():
        masks.pop("trainxval_lake")       # consumed by the 'recent' fit scheme
    return df, pix, info, masks


def geo_ctx(df: pd.DataFrame, pix: dict) -> tuple[np.ndarray, np.ndarray]:
    s, c = F.doy_harmonics(df.doy)
    u = F.unit_vector(df.lat, df.lon)
    geo = np.stack([np.cos(np.radians(df.sun_zenith.to_numpy(float))), np.cos(np.radians(df.view_zenith.to_numpy(float))),
                    np.cos(np.radians(df.sun_azimuth.to_numpy(float) - df.view_azimuth.to_numpy(float))),
                    df.aot_c3.to_numpy(float), np.log1p(df.jrc_dist_land_m.to_numpy(float)),
                    pix["n_water_win"] / 225.0, df.scl_c3_n_water.to_numpy(float) / 9.0], 1)
    ctx = np.stack([s, c, u[:, 0], u[:, 1], u[:, 2]], 1)
    return geo, ctx


class Norm:
    """Fit-only normalisation of geometry and standardisation of log10 targets."""

    def __init__(self, geo_fit: np.ndarray, y_fit_log: np.ndarray):
        self.g_mean = np.nanmean(geo_fit, 0)
        self.g_std = np.nanstd(geo_fit, 0) + 1e-6
        self.y_mean = np.nanmean(y_fit_log, 0)
        self.y_std = np.nanstd(y_fit_log, 0) + 1e-6

    def geo(self, g):
        return np.nan_to_num((g - self.g_mean) / self.g_std, nan=0.0).astype(np.float32)

    def y(self, ylog):
        return ((ylog - self.y_mean) / self.y_std).astype(np.float32)

    def to_dict(self):
        return {k: np.asarray(v).tolist() for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d):
        o = cls.__new__(cls)
        for k, v in d.items():
            setattr(o, k, np.asarray(v, float))
        return o


def ylog_matrix(df: pd.DataFrame) -> np.ndarray:
    return np.stack([np.log10(df[t].to_numpy(float)) for t in D.TARGETS], 1)


def lam_matrix(platforms) -> np.ndarray:
    wl = D.band_wavelengths()
    tab = {p: np.array([wl[p][b] for b in D.BANDS], np.float32) for p in wl}
    return np.stack([tab.get(p, tab["S2A"]) for p in platforms])


def make_set(df, pix, norm: Norm, mask=None):
    from .satwq import Batchable

    m = np.ones(len(df), bool) if mask is None else mask
    geo, ctx = geo_ctx(df, pix)
    y = ylog_matrix(df)
    return Batchable(pix["refl"][m], pix["n_pix"][m], pix["offsets"][m], lam_matrix(df.platform.to_numpy()[m]),
                     norm.geo(geo[m]), ctx[m].astype(np.float32), norm.y(y[m]))


def gloria_set(g: pd.DataFrame, norm: Norm):
    """GLORIA rows as single-pixel chips with 9 bands (B09/B11/B12 missing), mean geometry, no context."""
    from .satwq import Batchable

    n = len(g)
    refl = np.full((n, 1, len(D.BANDS)), np.nan, np.float32)
    for j, b in enumerate(D.BANDS):
        if b in D.GLORIA_BANDS:
            refl[:, 0, j] = g[b].to_numpy(float)
    y = np.stack([np.log10(g[t].to_numpy(float)) for t in D.TARGETS], 1)
    return Batchable(refl, np.ones(n, np.int32), np.zeros((n, 1, 2), np.float32), lam_matrix(["S2A"] * n),
                     np.zeros((n, len(GEO_COLS)), np.float32), np.zeros((n, len(CTX_COLS)), np.float32), norm.y(y))


def log_config_tried(model: str, cfg: dict, run_dir: Path, status: str, extra: dict | None = None) -> None:
    p = runs_root() / "configs_tried.jsonl"
    rec = {"time": _dt.datetime.now().isoformat(timespec="seconds"), "model": model, "config_hash": config_hash(cfg),
           "config": cfg, "run_dir": str(run_dir), "status": status, **(extra or {})}
    with open(p, "a") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")


def save_predictions(path: Path, df: pd.DataFrame, mask: np.ndarray, mixtures: dict | None = None,
                     point: dict | None = None) -> None:
    """Predictions file: row_id + per-target log10 point estimates (``point``) and/or mixtures
    (``mixtures`` = {'w','mu','sg'} arrays [N, T, K])."""
    out = {"row_id": df.row_id.to_numpy()[mask].astype(str)}
    if mixtures:
        out.update({k: v.astype(np.float32) for k, v in mixtures.items()})
    if point:
        for t, v in point.items():
            out[f"point_{t}"] = np.asarray(v, np.float32)
    np.savez_compressed(path, **out)


def write_json(path: Path, obj) -> None:
    Path(path).write_text(json.dumps(obj, indent=1, default=str))
