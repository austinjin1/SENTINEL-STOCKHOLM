# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Chip stores: sharded compressed NPZ arrays + a Parquet index, resume-safe.

Layout of a store directory::

    <root>/layers.json            layer names, dtype and value semantics
    <root>/shards/shard-00012.npz one zip member per chip (key = chip_id), deflate-compressed
    <root>/index/part-00012.parquet  one row per *request*: status 'ok' rows point to a chip,
                                  other statuses (no_scene, edge, error:...) record why not
    <root>/index.parquet          consolidated index (``ChipWriter.consolidate``)

A shard and its index part are written to temporary names and renamed (shard first, then the part),
so a crash never leaves a part that points to a missing chip. On restart, ``ChipWriter`` reads all
parts and skips every request key already present (failures included unless ``retry_failed``).
Shards stay far below 2 GB (``max_shard_bytes``). Per-chip random access: ``np.load`` of an NPZ
decompresses only the requested member.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .sources import stac as S

# ------------------------------------------------------------------------------------ writing


class ChipWriter:
    def __init__(self, root: str | os.PathLike, *, layers: Sequence[str], dtype: str,
                 semantics: dict[str, Any], max_shard_chips: int = 2048,
                 max_shard_bytes: int = 1_500_000_000, flush_seconds: float = 900.0,
                 retry_failed: bool = False):
        self.root = Path(root)
        (self.root / "shards").mkdir(parents=True, exist_ok=True)
        (self.root / "index").mkdir(parents=True, exist_ok=True)
        self.layers = list(layers)
        self.dtype = np.dtype(dtype)
        meta = {"layers": self.layers, "dtype": str(self.dtype), "semantics": semantics}
        lj = self.root / "layers.json"
        if lj.exists():
            old = json.loads(lj.read_text())
            if old.get("layers") != self.layers or old.get("dtype") != str(self.dtype):
                raise ValueError(f"store {self.root} has layers {old.get('layers')} / {old.get('dtype')}")
        else:
            lj.write_text(json.dumps(meta, indent=1, default=str))
        self.max_chips = max_shard_chips
        self.max_bytes = max_shard_bytes
        self.flush_seconds = flush_seconds
        self._lock = threading.Lock()
        self._arrays: dict[str, np.ndarray] = {}
        self._rows: list[dict] = []
        self._bytes = 0
        self._t_last = time.time()
        idx = read_index_parts(self.root)
        self.done: set[str] = set()
        if len(idx):
            keep = idx[idx["status"] == "ok"] if retry_failed else idx
            self.done = set(keep["request_key"].astype(str))
        self._clean_orphans()
        nums = [int(p.stem.split("-")[1]) for p in (self.root / "index").glob("part-*.parquet")]
        self.shard_no = max(nums) + 1 if nums else 0
        self.n_written = 0

    def _clean_orphans(self) -> None:
        parts = {p.stem.split("-")[1] for p in (self.root / "index").glob("part-*.parquet")}
        for sh in (self.root / "shards").glob("shard-*.npz"):
            if sh.stem.split("-")[1] not in parts:
                sh.unlink()  # shard whose index part was never written (crash): its chips are redone
        for tmp in list((self.root / "shards").glob("*.tmp*")) + list((self.root / "index").glob("*.tmp*")):
            tmp.unlink()

    def is_done(self, key: str) -> bool:
        return key in self.done

    def add(self, key: str, array: np.ndarray | None, meta: dict[str, Any], status: str = "ok") -> None:
        """Queue one request result. ``array`` is required when ``status == 'ok'``."""
        with self._lock:
            row = dict(meta)
            row["request_key"] = key
            row["status"] = status
            if status == "ok":
                if array is None:
                    raise ValueError("ok row without array")
                a = np.ascontiguousarray(array, dtype=self.dtype)
                if a.shape[0] != len(self.layers):
                    raise ValueError(f"chip has {a.shape[0]} layers, store expects {len(self.layers)}")
                chip_id = f"{self.shard_no:05d}_{len(self._arrays):05d}"
                self._arrays[chip_id] = a
                self._bytes += a.nbytes
                row["chip_id"] = chip_id
                row["shard"] = f"shard-{self.shard_no:05d}.npz"
            else:
                row["chip_id"] = None
                row["shard"] = None
            self._rows.append(row)
            self.done.add(key)
            # max_bytes bounds the *uncompressed* shard size, so the compressed file is smaller still
            if (len(self._arrays) >= self.max_chips or self._bytes >= self.max_bytes
                    or time.time() - self._t_last > self.flush_seconds):
                self._flush_locked()

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        self._t_last = time.time()
        if not self._rows:
            return
        n = self.shard_no
        if self._arrays:
            dest = self.root / "shards" / f"shard-{n:05d}.npz"
            tmp = dest.with_name(dest.name + ".tmp.npz")
            np.savez_compressed(tmp, **self._arrays)
            os.replace(tmp, dest)
        part = self.root / "index" / f"part-{n:05d}.parquet"
        tmp = part.with_name(part.name + ".tmp")
        df = normalise_columns(pd.DataFrame(self._rows))
        df.to_parquet(tmp, index=False, compression="zstd")
        os.replace(tmp, part)
        self.n_written += len(self._arrays)
        self._arrays, self._rows, self._bytes = {}, [], 0
        self.shard_no += 1

    def close(self) -> None:
        self.flush()


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Make object columns Parquet-safe without changing meaning: all-bool -> nullable boolean,
    all-numeric -> float64, dict/list -> JSON text, anything else -> str (None stays null)."""
    out = df.copy()
    for c in out.columns:
        if out[c].dtype != object:
            continue
        vals = [v for v in out[c].tolist() if v is not None and not (isinstance(v, float) and np.isnan(v))]
        if vals and all(isinstance(v, (bool, np.bool_)) for v in vals):
            out[c] = pd.array([None if (v is None or (isinstance(v, float) and np.isnan(v))) else bool(v)
                               for v in out[c].tolist()], dtype="boolean")
        elif vals and all(isinstance(v, (int, float, np.integer, np.floating)) and
                          not isinstance(v, (bool, np.bool_)) for v in vals):
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
        else:
            out[c] = out[c].map(lambda v: v if v is None or isinstance(v, (str, bytes)) else
                                (None if isinstance(v, float) and np.isnan(v) else
                                 json.dumps(v, default=str) if isinstance(v, (dict, list, tuple)) else str(v)))
    return out


def read_index_parts(root: str | os.PathLike) -> pd.DataFrame:
    parts = sorted((Path(root) / "index").glob("part-*.parquet"))
    if not parts:
        return pd.DataFrame(columns=["request_key", "status", "chip_id", "shard"])
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)


def consolidate(root: str | os.PathLike) -> pd.DataFrame:
    """Write <root>/index.parquet from all parts (latest row per request key wins)."""
    root = Path(root)
    idx = read_index_parts(root)
    if len(idx):
        idx = idx.drop_duplicates("request_key", keep="last").reset_index(drop=True)
    idx.to_parquet(root / "index.parquet", index=False, compression="zstd")
    return idx


# ------------------------------------------------------------------------------------ reading


class ChipStore:
    """Read access to a chip store (consolidated index if present, else the parts)."""

    def __init__(self, root: str | os.PathLike):
        self.root = Path(root)
        meta = json.loads((self.root / "layers.json").read_text())
        self.layers: list[str] = meta["layers"]
        self.dtype = np.dtype(meta["dtype"])
        self.semantics = meta.get("semantics", {})
        ip = self.root / "index.parquet"
        self.index = pd.read_parquet(ip) if ip.exists() else read_index_parts(self.root)
        self._ok = self.index[self.index["status"] == "ok"].set_index("chip_id")
        self._open: dict[str, Any] = {}
        self._lock = threading.Lock()

    def _shard(self, name: str):
        with self._lock:
            z = self._open.get(name)
            if z is None:
                z = np.load(self.root / "shards" / name)
                self._open[name] = z
            return z

    def load(self, chip_id: str) -> np.ndarray:
        shard = self._ok.loc[chip_id, "shard"]
        return self._shard(shard)[chip_id]

    def layer(self, chip: np.ndarray, name: str) -> np.ndarray:
        return chip[self.layers.index(name)]

    def close(self) -> None:
        for z in self._open.values():
            z.close()
        self._open.clear()


# ---------------------------------------------------------------------------- S2 chip features

S2_STORE_LAYERS = list(S.S2_LAYER_NAMES) + ["JRC"]
S2_SEMANTICS = {
    "grid": "10 m UTM grid of the MGRS tile; chip 64x64 = 640 m; point in pixel (32, 32) (0-based)",
    "reflectance_layers": list(S.S2_REFLECTANCE_LAYERS),
    "reflectance": "int16 harmonised DN = raw DN + offset_dn (offset_dn = BOA_ADD_OFFSET, 0 or -1000); "
                   "reflectance = value * 1e-4; nodata = -32768",
    "AOT": "raw DN, aerosol optical thickness = value * 1e-3; nodata = -32768",
    "SCL": "Sen2Cor scene classification code 0..11 (0 = no data)",
    "JRC": "JRC GSW v1.3 occurrence 0..100 % (1984-2020), nearest-resampled onto the chip grid; 255 = no data",
    "resampling": "nearest-neighbour replication on nested MGRS grids (20 m -> 2x2, 60 m -> 6x6); "
                  "identical footprint for all layers",
}


def _median_over(a: np.ndarray, m: np.ndarray) -> float:
    v = a[m]
    return float(np.median(v)) if v.size else float("nan")


def s2_chip_features(chip: np.ndarray, layers: Sequence[str] = S2_STORE_LAYERS,
                     windows: Iterable[int] = (3, 7)) -> dict[str, Any]:
    """Summary statistics of one S2 chip (int16, layers as in ``S2_STORE_LAYERS``).

    For each centre window k (k x k at 10 m, centred on the point's pixel): SCL class fractions,
    water/valid counts, and per-band median reflectance over SCL-water pixels (``rw{k}_<band>``)
    and over SCL-valid pixels (``rv{k}_<band>``). Chip-level: SCL fractions, nodata fraction, JRC
    water fraction, and offset-sanity statistics (``qc_refl_p01_min``: minimum over bands of the 1st
    percentile reflectance over valid pixels — about -0.1 after a double offset correction and
    >= +0.08 when a required correction is missing; ``qc_water_b11_median``)."""
    L = {n: i for i, n in enumerate(layers)}
    size = chip.shape[-1]
    scl = chip[L["SCL"]]
    refl = {b: S.to_reflectance(chip[L[b]]) for b in S.S2_REFLECTANCE_LAYERS if b in L}
    out: dict[str, Any] = {}
    out.update(S.scl_summary(scl, "scl_chip"))
    nodata = chip[L["B02"]] == S.NODATA_I16
    out["chip_nodata_frac"] = float(nodata.mean())
    valid = ~np.isin(scl, S.SCL_INVALID) & ~nodata
    water = (scl == 6) & ~nodata
    if "JRC" in L:
        jrc = chip[L["JRC"]]
        jv = jrc != S.JRC_NODATA
        out["jrc_chip_water_frac"] = float(((jrc >= 50) & jv).sum() / max(1, jv.sum()))
    p01 = [np.nanpercentile(r[valid], 1) for r in refl.values() if valid.any()]
    out["qc_refl_p01_min"] = float(np.nanmin(p01)) if p01 else float("nan")
    out["qc_water_b11_median"] = _median_over(refl["B11"], water) if "B11" in refl else float("nan")
    for k in windows:
        sl = S.centre_slice(size, k)
        c_scl = scl[sl, sl]
        out.update(S.scl_summary(c_scl, f"scl_c{k}"))
        w = water[sl, sl]
        v = valid[sl, sl]
        for b, r in refl.items():
            out[f"rw{k}_{b}"] = _median_over(r[sl, sl], w)
            out[f"rv{k}_{b}"] = _median_over(r[sl, sl], v)
        if "AOT" in L:
            aot = chip[L["AOT"]][sl, sl].astype(float)
            aot = aot[aot != S.NODATA_I16]
            out[f"aot_c{k}"] = float(np.median(aot) * S.AOT_SCALE) if aot.size else float("nan")
        if "JRC" in L:
            j = chip[L["JRC"]][sl, sl]
            j = j[j != S.JRC_NODATA]
            out[f"jrc_c{k}_median"] = float(np.median(j)) if j.size else float("nan")
    return out


def s2_offset_suspect(features: dict[str, Any]) -> str:
    """Chip-level radiometric sanity flag.

    'double'       — the darkest valid reflectance is < -0.05 (no natural surface; the signature of an
                     offset subtracted twice);
    'bright_water' — clear-sky chip (SCL cloud/cirrus <= 5 %) whose SCL-water pixels have a median B11
                     reflectance > 0.08 (sun glint, haze or a missing offset — indistinguishable per chip,
                     and unusable for water-quality retrieval either way);
    ''             — otherwise. A *systematic* missing offset is tested per (source, baseline) group on
                     clear water (datasheet s2_chips.md), not per chip."""
    p01 = features.get("qc_refl_p01_min", np.nan)
    if np.isfinite(p01) and p01 < -0.05:
        return "double"
    wb11 = features.get("qc_water_b11_median", np.nan)
    cloud = sum(features.get(f"scl_chip_{k}", 0.0) or 0.0 for k in ("cloud_medium", "cloud_high", "thin_cirrus"))
    if np.isfinite(wb11) and wb11 > 0.08 and cloud <= 0.05:
        return "bright_water"
    return ""


V1_BANDS = ("B02", "B03", "B04", "B05", "B08", "B11")


def v1_definition_features(chip: np.ndarray, layers: Sequence[str] = S2_STORE_LAYERS,
                           water_only: bool = False) -> dict[str, float]:
    """The 26 v1 microcystin features (``analyze_microcystin_fusion.spectral_features``: per band of
    B02/B03/B04/B05/B08/B11 the chip mean, centre-20x20 mean and std; NDVI/NDCI/NDWI/G:R mean and
    centre mean) recomputed on a v2 chip: offset-harmonised reflectance, every band on the same 640 m
    10 m grid (v1 read 20 m bands over 1.28 km), nodata excluded; optionally SCL-water pixels only."""
    L = {n: i for i, n in enumerate(layers)}
    b = {n: S.to_reflectance(chip[L[n]]) for n in V1_BANDS}
    if water_only:
        w = chip[L["SCL"]] == 6
        b = {n: np.where(w, v, np.nan) for n, v in b.items()}
    c = slice(22, 42)
    out: dict[str, float] = {}
    with np.errstate(all="ignore"):
        for n, arr in b.items():
            out[f"{n}_mean"] = float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan")
            cc = arr[c, c]
            out[f"{n}_c20_mean"] = float(np.nanmean(cc)) if np.isfinite(cc).any() else float("nan")
            out[f"{n}_std"] = float(np.nanstd(arr)) if np.isfinite(arr).any() else float("nan")
        eps = 1e-6
        idx = {"ndvi": (b["B08"] - b["B04"]) / (b["B08"] + b["B04"] + eps),
               "ndci": (b["B05"] - b["B04"]) / (b["B05"] + b["B04"] + eps),
               "ndwi": (b["B03"] - b["B08"]) / (b["B03"] + b["B08"] + eps),
               "g_r": b["B03"] / (b["B04"] + eps)}
        for n, arr in idx.items():
            out[f"{n}_mean"] = float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan")
            cc = arr[c, c]
            out[f"{n}_c20_mean"] = float(np.nanmean(cc)) if np.isfinite(cc).any() else float("nan")
    return out


def recompute_offset_suspect(idx: pd.DataFrame) -> pd.Series:
    """``s2_offset_suspect`` re-evaluated from stored index features (rows with status 'ok')."""
    cols = ["qc_refl_p01_min", "qc_water_b11_median", "scl_chip_cloud_medium", "scl_chip_cloud_high",
            "scl_chip_thin_cirrus"]
    out = pd.Series(None, index=idx.index, dtype=object)
    ok = idx["status"] == "ok"
    if ok.any():
        sub = idx.loc[ok, [c for c in cols if c in idx.columns]]
        out[ok] = [s2_offset_suspect(r) for r in sub.to_dict("records")]
    return out


def offset_group_check(idx: pd.DataFrame, min_water_px: int = 20, max_cloud: float = 0.05) -> pd.DataFrame:
    """Systematic BOA-offset check: per (stac_source, processing_baseline, offset_dn) group, the median
    B11 reflectance over SCL-water pixels of clear chips (cloud/cirrus <= ``max_cloud``, >= ``min_water_px``
    water pixels). Clean water is ~0-0.02 in B11; a group shifted by ~+0.1 (missing offset) or ~-0.1
    (double offset) would stand out."""
    ok = idx[idx["status"] == "ok"].copy()
    need = ["scl_chip_cloud_medium", "scl_chip_cloud_high", "scl_chip_thin_cirrus", "scl_chip_n_water",
            "qc_water_b11_median", "stac_source", "processing_baseline", "offset_dn"]
    if ok.empty or any(c not in ok.columns for c in need):
        return pd.DataFrame(columns=["stac_source", "processing_baseline", "offset_dn", "n", "median", "p10", "p90"])
    cloud = ok[["scl_chip_cloud_medium", "scl_chip_cloud_high", "scl_chip_thin_cirrus"]].sum(axis=1)
    clear = ok[(cloud <= max_cloud) & (ok["scl_chip_n_water"] >= min_water_px)]
    g = clear.groupby(["stac_source", "processing_baseline", "offset_dn"])["qc_water_b11_median"]
    return g.agg(n="size", median="median", p10=lambda x: x.quantile(0.1), p90=lambda x: x.quantile(0.9)).reset_index()
