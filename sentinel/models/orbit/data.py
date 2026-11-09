# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""SENTINEL-Orbit data: S2 L2A matchup table (AquaCorr-Global + WQP), image QC, de-duplication, split
assignment, guard, per-chip water-pixel cache, and GLORIA pretraining rows (orbit_protocol.md §2–§4).

Row = one (location, in-situ date, S2 acquisition) with up to four targets (chl-a µg/L, TSS mg/L,
turbidity NTU/FNU, Secchi m). Same-day matchups only (primary window). Only rows passing OUR image QC
(:func:`image_qc`) are used; AquaCorr's own ``passed_qc`` is not trusted (datasheet aquacorr_global.md).
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ...core import paths
from ...data import chips as C
from ...data.sources import stac as S
from . import guard
from . import split as SP

TARGETS = ("chla", "tss", "turbidity", "secchi")
UNITS = {"chla": "ug/L", "tss": "mg/L", "turbidity": "NTU or FNU (AquaCorr: unit unstated)", "secchi": "m"}
VALID = {"chla": (0.05, 3000.0), "tss": (0.1, 5000.0), "turbidity": (0.05, 5000.0), "secchi": (0.05, 40.0)}
BANDS = tuple(S.S2_REFLECTANCE_LAYERS)          # B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B11 B12
GLORIA_BANDS = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A")
RGBN = ("B02", "B03", "B04", "B08")
P_MAX = 64            # water pixels per chip kept in the cache (nearest to the point)
WIN = 15              # centre window (pixels at 10 m) searched for water pixels
CENTER = 32
QC_RULES = {"scl_c3_n_water_min": 5, "jrc_water_ok": True, "offset_suspect": "", "scl_c3_cloud_shadow_cirrus_max": 0.0,
            "rw3_B02_max": 0.15, "rw3_B11_max": 0.05, "aquacorr_same_acquisition": True, "window": "same UTC/local day"}
STORES = {"aquacorr": "aquacorr", "wqp": "wqp"}
CACHE_VERSION = "dev_v3"


# ------------------------------------------------------------------------------------- wavelengths

def band_wavelengths() -> dict[str, dict[str, float]]:
    """Centre wavelengths (nm, SRF centroid) per platform from processed('gloria')/srf_info.parquet.
    S2C has no SRF table locally: S2A values are used (documented approximation)."""
    info = pd.read_parquet(paths.processed("gloria") / "srf_info.parquet")
    out = {}
    for plat in ("S2A", "S2B"):
        d = info[info.sensor == plat].set_index("band")["centroid_nm"]
        out[plat] = {b: float(d[b.replace("B0", "B")]) for b in BANDS}
    out["S2C"] = dict(out["S2A"])
    return out


# --------------------------------------------------------------------------------- source tables

def _index(store: str) -> pd.DataFrame:
    idx = C.read_index_parts(paths.processed("s2_chips") / store).drop_duplicates("request_key", keep="last")
    idx = idx[idx.status == "ok"].copy()
    idx["offset_suspect"] = C.recompute_offset_suspect(idx)
    return idx


IMG_COLS = (["chip_id", "shard", "item_id", "platform", "sensing_time_utc", "processing_baseline", "stac_source",
             "sun_zenith", "sun_azimuth", "view_zenith", "view_azimuth", "aot_c3", "aot_c7", "jrc_occ_point",
             "jrc_dist_land_m", "jrc_water_ok", "jrc_chip_water_frac", "scl_c3_n_water", "scl_c7_n_water",
             "scl_c3_cloud_medium", "scl_c3_cloud_high", "scl_c3_thin_cirrus", "scl_c3_cloud_shadow",
             "offset_suspect", "scene_cloud_cover"]
            + [f"rw3_{b}" for b in BANDS] + [f"rw7_{b}" for b in BANDS])


def aquacorr_rows(log=print) -> pd.DataFrame:
    proc = paths.processed("aquacorr_global")
    tcols = ["chla_value", "tss_value", "sdd_value", "turbidity_value", "chla_method_class", "chla_recommended",
             "chla_outlier", "tss_outlier", "sdd_outlier", "turbidity_outlier"]
    sel = pd.read_parquet(proc / "s2_chip_requests.parquet",
                          columns=["matchup_id", "site_id", "source", "lat", "lon", "date_insitu", "sensor",
                                   "chip_request_key"] + tcols)
    plan = pd.read_parquet(proc / "s2_chip_plan.parquet", columns=["chip_request_key", "match_kind"])
    idx = _index("aquacorr").rename(columns={"request_key": "chip_request_key"})
    df = sel.merge(plan, on="chip_request_key", how="left").merge(
        idx[["chip_request_key"] + [c for c in IMG_COLS if c in idx.columns]], on="chip_request_key", how="inner")
    st = pd.to_datetime(df.sensing_time_utc, utc=True, errors="coerce")
    same_acq = (df.platform == df.sensor) & (st.dt.strftime("%Y-%m-%d") == pd.to_datetime(df.date_insitu).dt.strftime(
        "%Y-%m-%d")) & df.match_kind.isin(["designated", "alternate_tile"])
    out = pd.DataFrame({"source": "aquacorr", "row_src_id": df.matchup_id.astype(str), "site_id": df.site_id.astype(str),
                        "provider": df.source.astype(str), "lat": df.lat.astype(float), "lon": df.lon.astype(float),
                        "date": pd.to_datetime(df.date_insitu).dt.normalize(), "chip_store": "aquacorr",
                        "same_acquisition": same_acq.to_numpy()})
    for c in IMG_COLS:
        if c in df:
            out[c] = df[c].to_numpy()
    lab = df.chla_method_class.astype(str)
    chla = df.chla_value.where(~lab.eq("in_vivo_sensor") & ~df.chla_outlier.astype("boolean").fillna(False))
    out["chla"] = chla.astype(float).to_numpy()
    out["chla_lab"] = lab.eq("extraction_lab").to_numpy()
    for t, c in (("tss", "tss"), ("turbidity", "turbidity"), ("secchi", "sdd")):
        v = df[f"{c}_value"].where(~df[f"{c}_outlier"].astype("boolean").fillna(False))
        out[t] = v.astype(float).to_numpy()
    out["turbidity_unit"] = "NTU_or_FNU_unspecified"
    log(f"[data] aquacorr: {len(sel)} requests -> {len(out)} rows with an extracted chip")
    return out


WQP_UNIT_FACTORS = {"chla": {"ug/L": 1.0, "mg/m3": 1.0, "ppb": 1.0},
                    "tss": {"mg/L": 1.0},
                    "secchi": {"m": 1.0, "cm": 0.01, "ft": 0.3048},
                    "turbidity": {"NTU": 1.0, "FNU": 1.0, "NTRU": 1.0}}


# WQP x S2 matchup tables (processed/<name>): the national 2024-2025 build and the 2015-2023 build
# (``build_s2_wqp_matchups.py --out-name s2_wqp_matchups_hist``). Date ranges are disjoint; missing sets
# are skipped (dev_v4 was built before the historical set existed).
WQP_MATCHUP_SETS = ("s2_wqp_matchups", "s2_wqp_matchups_hist", "s2_wqp_matchups_2026")


def _wqp_tables(log=print):
    smp, mt = [], []
    for name in WQP_MATCHUP_SETS:
        d = paths.PROCESSED / name
        if not ((d / "samples.parquet").exists() and (d / "matchups.parquet").exists()):
            log(f"[data] wqp: matchup set {name} not present, skipped")
            continue
        smp.append(pd.read_parquet(d / "samples.parquet").assign(_set=name))
        mt.append(pd.read_parquet(d / "matchups.parquet").assign(_set=name))
        log(f"[data] wqp: matchup set {name}: {len(smp[-1])} samples, {len(mt[-1])} sample x acquisition rows")
    if not smp:
        raise FileNotFoundError("no WQP matchup set found")
    smp = pd.concat(smp, ignore_index=True)
    # a sample present in two sets (overlapping date ranges) is kept once (first set)
    smp = smp.drop_duplicates("sample_key", keep="first") if "sample_key" in smp else smp
    return smp, pd.concat(mt, ignore_index=True)


def wqp_rows(log=print) -> pd.DataFrame:
    smp, mt_all = _wqp_tables(log)
    smp = smp[smp.target.isin(TARGETS) & smp.jrc_water_ok.astype("boolean").fillna(False)].copy()
    fac = smp.apply(lambda r: WQP_UNIT_FACTORS.get(r.target, {}).get(r.unit, np.nan), axis=1)
    n_bad_unit = int(fac.isna().sum())
    smp["v"] = smp.value * fac
    smp = smp[np.isfinite(smp.v)]
    smp["date"] = pd.to_datetime(smp.date).dt.normalize()
    piv = smp.pivot_table(index=["point_key", "date"], columns="target", values="v", aggfunc="median").reset_index()
    tu = smp[smp.target == "turbidity"].groupby(["point_key", "date"]).unit.agg(lambda u: "|".join(sorted(set(u))))
    meta = smp.drop_duplicates(["point_key", "date"])[["point_key", "date", "site_id", "lat", "lon", "provider",
                                                        "site_class", "organization"]]
    mt = mt_all[mt_all.win_same_day & mt_all.chip_ok].copy()
    mt["date"] = pd.to_datetime(mt.date).dt.normalize()
    ren = {f"{c}_chip": c for c in ("scl_c3_n_water", "scl_c7_n_water", "scl_c3_cloud_medium", "scl_c3_cloud_high",
                                     "scl_c3_thin_cirrus", "scl_c3_cloud_shadow")}
    mt = mt.drop(columns=[v for v in ren.values() if v in mt.columns]).rename(columns=ren)
    mt = mt.sort_values(["point_key", "date", "scl_c3_n_water", "request_key"], ascending=[True, True, False, True])
    mt = mt.drop_duplicates(["point_key", "date"])
    idx = _index("wqp")[["request_key", "chip_id", "shard", "platform", "sensing_time_utc", "scene_cloud_cover",
                         "item_id"]]
    keep = [c for c in IMG_COLS if c in mt.columns and c not in idx.columns]
    mt = mt[["point_key", "date", "request_key"] + keep].merge(idx, on="request_key", how="inner")
    df = meta.merge(piv, on=["point_key", "date"]).merge(mt, on=["point_key", "date"], how="inner")
    out = pd.DataFrame({"source": "wqp", "row_src_id": (df.point_key + "|" + df.date.dt.strftime("%Y-%m-%d")).to_numpy(),
                        "site_id": df.site_id.astype(str).to_numpy(), "provider": df.provider.astype(str).to_numpy(),
                        "lat": df.lat.astype(float).to_numpy(), "lon": df.lon.astype(float).to_numpy(),
                        "date": df.date.to_numpy(), "chip_store": "wqp", "same_acquisition": True})
    for c in IMG_COLS:
        if c in df:
            out[c] = df[c].to_numpy()
    for t in TARGETS:
        out[t] = df[t].astype(float).to_numpy() if t in df else np.nan
    out["chla_lab"] = True   # WQP lab characteristic names only (probe names are never requested upstream)
    tu_map = tu.to_dict()
    out["turbidity_unit"] = [tu_map.get((p, dd), "") for p, dd in zip(df.point_key, df.date)]
    log(f"[data] wqp: {len(smp)} usable samples ({n_bad_unit} dropped for unit), {len(out)} same-day chip rows")
    return out


# ------------------------------------------------------------------------------------ QC / dedupe

def image_qc(df: pd.DataFrame) -> pd.Series:
    """Image-only QC (never uses labels): see :data:`QC_RULES`."""
    cl = sum(df[c].fillna(0) for c in ("scl_c3_cloud_medium", "scl_c3_cloud_high", "scl_c3_thin_cirrus",
                                        "scl_c3_cloud_shadow"))
    ok = (df.scl_c3_n_water.fillna(0) >= QC_RULES["scl_c3_n_water_min"]) \
        & df.jrc_water_ok.astype("boolean").fillna(False).astype(bool) \
        & (df.offset_suspect.fillna("") == "") & (cl <= QC_RULES["scl_c3_cloud_shadow_cirrus_max"]) \
        & (df.rw3_B02 <= QC_RULES["rw3_B02_max"]) & (df.rw3_B11 <= QC_RULES["rw3_B11_max"]) \
        & df.same_acquisition.astype(bool)
    return ok


def _components_within(lat, lon, keys, radius_m: float) -> np.ndarray:
    """Connected components of rows that share ``keys`` (e.g. date) and lie within ``radius_m``."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from sklearn.neighbors import BallTree

    n = len(lat)
    rows, cols = [], []
    keys = np.asarray(keys)
    for k in np.unique(keys):
        ix = np.flatnonzero(keys == k)
        if ix.size < 2:
            continue
        tree = BallTree(np.radians(np.c_[lat[ix], lon[ix]]), metric="haversine")
        nb = tree.query_radius(np.radians(np.c_[lat[ix], lon[ix]]), r=radius_m / 6371008.8)
        for i, js in enumerate(nb):
            rows.extend([ix[i]] * len(js))
            cols.extend(ix[js])
    g = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)) if rows else coo_matrix((n, n))
    return connected_components(g, directed=False)[1]


def dedupe(df: pd.DataFrame, log=print, radius_aq_m: float = 20.0, radius_wqp_m: float = 100.0) -> pd.DataFrame:
    """Distance-based de-duplication (review W5): (1) AquaCorr rows on the same date and platform within
    ``radius_aq_m`` are one sample (cross-source repeats) -> one row, targets = median; (2) an AquaCorr row
    within ``radius_wqp_m`` of a WQP row on the same date is dropped (WQP preferred)."""
    n0 = len(df)
    a = df[df.source == "aquacorr"].reset_index(drop=True)
    w = df[df.source == "wqp"].reset_index(drop=True)
    if len(a):
        comp = _components_within(a.lat.to_numpy(), a.lon.to_numpy(),
                                  (a.date.astype(str) + "|" + a.platform.astype(str)).to_numpy(), radius_aq_m)
        a["_k"] = comp
        a = a.sort_values(["_k", "scl_c3_n_water", "row_src_id"], ascending=[True, False, True])
        med = a.groupby("_k")[list(TARGETS)].median()
        nsrc = a.groupby("_k").size()
        a = a.drop_duplicates("_k").set_index("_k")
        a[list(TARGETS)] = med.loc[a.index]
        a["n_merged"] = nsrc.loc[a.index].to_numpy()
        a = a.reset_index(drop=True)
    if len(w) and len(a):
        w["n_merged"] = 1
        both = pd.concat([a[["lat", "lon", "date"]].assign(_s="a"), w[["lat", "lon", "date"]].assign(_s="w")],
                         ignore_index=True)
        comp = _components_within(both.lat.to_numpy(), both.lon.to_numpy(), both.date.astype(str).to_numpy(),
                                  radius_wqp_m)
        wcomp = set(comp[len(a):])
        drop = np.isin(comp[:len(a)], list(wcomp))
        log(f"[data] dedupe: {int(drop.sum())} AquaCorr rows duplicate a WQP row within {radius_wqp_m:.0f} m (kept WQP)")
        a = a[~drop]
    elif len(w):
        w["n_merged"] = 1
    out = pd.concat([a, w], ignore_index=True)
    log(f"[data] dedupe: {n0} -> {len(out)} rows")
    return out


TURBIDITY_FIT_CLASSES = ("NTU", "unspecified")


def turbidity_unit_class(units: pd.Series) -> np.ndarray:
    """NTU family (EPA 180.1 white-light NTU and ratio NTRU) vs FNU (ISO 7027, 860 nm) vs mixed (both on one
    sample-date) vs unspecified (AquaCorr states no unit)."""
    def one(u):
        parts = {x for x in str(u or "").split("|") if x}
        if not parts or parts == {"NTU_or_FNU_unspecified"}:
            return "unspecified"
        fam = {"NTU" if x in ("NTU", "NTRU") else "FNU" if x == "FNU" else "other" for x in parts}
        return fam.pop() if len(fam) == 1 else "mixed"
    return np.array([one(u) for u in units.fillna("")], dtype=object)


def apply_turbidity_unit_policy(df: pd.DataFrame) -> dict:
    """Unit rule (CONTRIBUTING_V2 "never mix FNU and NTU"; protocol v0.3 §2.1): the modelled/scored target
    ``turbidity`` holds NTU-family and AquaCorr-unspecified values only. FNU and mixed-unit values are moved
    to ``turbidity_fnu`` (kept for the stratified report, never fitted). ``turbidity_all_units`` keeps the
    pooled value for the pre-registered pooled sensitivity. Idempotent."""
    if "turbidity_unit_class" in df.columns:
        return {}
    cls = turbidity_unit_class(df.get("turbidity_unit", pd.Series([""] * len(df), index=df.index)))
    df["turbidity_unit_class"] = cls
    df["turbidity_all_units"] = df["turbidity"].to_numpy(float)
    off = ~np.isin(cls, TURBIDITY_FIT_CLASSES)
    df["turbidity_fnu"] = np.where(off & (cls == "FNU"), df["turbidity"].to_numpy(float), np.nan)
    df.loc[off, "turbidity"] = np.nan
    has = df["turbidity_all_units"].notna().to_numpy()
    return {str(k): int(v) for k, v in pd.Series(cls[has]).value_counts().items()}


def apply_valid_ranges(df: pd.DataFrame) -> dict:
    counts = {}
    for t in TARGETS:
        lo, hi = VALID[t]
        v = df[t]
        bad = v.notna() & ((v < lo) | (v > hi))
        counts[t] = int(bad.sum())
        df.loc[bad, t] = np.nan
    return counts


# ------------------------------------------------------------------------------------ build table

def protected_cell_counts(df: pd.DataFrame) -> dict:
    """COUNTS ONLY for protected cells (for the W4 power analysis): per (role x tblock x in-lake) and
    target, the number of rows with a value inside the valid range, water bodies and role blocks, by
    source and turbidity unit. No label value, statistic or prediction leaves this function."""
    out = {}
    sub = df[df.protected.to_numpy(bool) & ~df.buffer_excluded.to_numpy(bool)]
    for (r, t, il), g in sub.groupby(["role", "tblock", "in_lake"]):
        key = f"{r}x{t}_{'lake' if il else 'nonlake'}"
        out[key] = {}
        for tg in TARGETS:
            lo, hi = VALID[tg]
            ok = g[tg].between(lo, hi).to_numpy()
            gg = g[ok]
            rec = {"n": int(ok.sum()), "n_waterbodies": int(gg.waterbody.nunique()), "n_blocks": int(gg.block.nunique()),
                   "by_source": {k: int(v) for k, v in gg.source.value_counts().items()}}
            if tg == "turbidity":
                rec["by_unit"] = {str(k): int(v) for k, v in gg.turbidity_unit.fillna("").value_counts().items()}
            out[key][tg] = rec
    return out


def build_table(*, allow_test: bool = False, freeze_file=None, log=print) -> tuple[pd.DataFrame, dict]:
    """Assemble the benchmark table. Without authorised access, protected rows (test role / test years)
    are dropped right after split assignment; only their row/water-body counts are reported."""
    raw = pd.concat([aquacorr_rows(log), wqp_rows(log)], ignore_index=True)
    info = {"n_rows_with_chip": int(len(raw)), "by_source": raw.source.value_counts().to_dict()}
    qc = image_qc(raw)
    info["image_qc_pass"] = int(qc.sum())
    df = raw[qc.to_numpy()].reset_index(drop=True)
    df = dedupe(df, log)
    sp = SP.assign(df.lon.to_numpy(), df.lat.to_numpy(), df.date.to_numpy(), log=log)
    df = pd.concat([df.reset_index(drop=True), sp], axis=1)
    cells = df.groupby(["role", "tblock"]).agg(n_rows=("lat", "size"), n_waterbodies=("waterbody", "nunique"))
    info["cells_all_rows"] = {f"{r}x{t}": v for (r, t), v in cells.to_dict("index").items()}
    info["split_distance_km"] = SP.distance_report(df.drop_duplicates(["lat", "lon"]))
    info["protected_cell_counts"] = protected_cell_counts(df)
    df, acc = guard.drop_protected(df, allow_test=allow_test, freeze_file=freeze_file, log=log)
    info["access"] = acc
    info["invalid_range_dropped"] = apply_valid_ranges(df)
    info["turbidity_unit_classes"] = apply_turbidity_unit_policy(df)
    df = df[df[list(TARGETS) + ["turbidity_fnu"]].notna().any(axis=1)].reset_index(drop=True)
    df["doy"] = pd.to_datetime(df.date).dt.dayofyear
    df["year"] = pd.to_datetime(df.date).dt.year
    df["row_id"] = [hashlib.sha1(f"{s}|{i}|{c}".encode()).hexdigest()[:16]
                    for s, i, c in zip(df.source, df.row_src_id, df.chip_id)]
    if df.row_id.duplicated().any():
        raise RuntimeError("row_id collision")
    info["n_rows_final"] = int(len(df))
    info["targets_by_cell"] = {f"{r}x{t}": {k: int(g[k].notna().sum()) for k in TARGETS}
                               for (r, t), g in df.groupby(["role", "tblock"])}
    return df, info


# ------------------------------------------------------------------------------------ pixel cache

def chip_pixels(chip: np.ndarray, layers=C.S2_STORE_LAYERS, p_max: int = P_MAX, win: int = WIN):
    """Nearest ``p_max`` valid SCL-water pixels (all 12 bands present) within the centre ``win``x``win``
    window: (refl [p_max, 12] float32 NaN-padded, offsets [p_max, 2] in pixels, n_valid, n_water_win)."""
    L = {n: i for i, n in enumerate(layers)}
    h = win // 2
    sl = slice(CENTER - h, CENTER + h + 1)
    scl = chip[L["SCL"], sl, sl]
    dn = np.stack([chip[L[b], sl, sl] for b in BANDS])
    ok = (scl == 6) & np.all(dn != S.NODATA_I16, axis=0)
    yy, xx = np.mgrid[-h:h + 1, -h:h + 1]
    d2 = (yy ** 2 + xx ** 2).astype(float) + (yy * win + xx) * 1e-6     # deterministic tie-break
    order = np.argsort(np.where(ok, d2, np.inf), axis=None)[: int(ok.sum())][:p_max]
    refl = np.full((p_max, len(BANDS)), np.nan, np.float32)
    off = np.zeros((p_max, 2), np.float32)
    n = order.size
    if n:
        r, c = np.unravel_index(order, ok.shape)
        refl[:n] = (dn[:, r, c].T.astype(np.float32) * 1e-4)
        off[:n, 0] = yy[r, c]
        off[:n, 1] = xx[r, c]
    return refl, off, n, int(ok.sum())


def _extract_shard(args):
    store, shard, chip_ids = args
    z = np.load(paths.processed("s2_chips") / store / "shards" / shard)
    out = []
    try:
        for cid in chip_ids:
            out.append((cid,) + chip_pixels(z[cid]))
    finally:
        z.close()
    return out


def extract_pixels(df: pd.DataFrame, workers: int = 6, log=print) -> dict[str, np.ndarray]:
    jobs = [(st, sh, g.chip_id.tolist()) for (st, sh), g in df.groupby(["chip_store", "shard"])]
    res = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, part in enumerate(ex.map(_extract_shard, jobs)):
            for cid, refl, off, n, nw in part:
                res[cid] = (refl, off, n, nw)
            if (i + 1) % 20 == 0:
                log(f"[data] pixels: {i + 1}/{len(jobs)} shards")
    refl = np.stack([res[c][0] for c in df.chip_id])
    off = np.stack([res[c][1] for c in df.chip_id])
    n = np.array([res[c][2] for c in df.chip_id], np.int32)
    nw = np.array([res[c][3] for c in df.chip_id], np.int32)
    return {"refl": refl.astype(np.float16), "offsets": off.astype(np.int8), "n_pix": n, "n_water_win": nw}


def cache_dir(version: str = CACHE_VERSION) -> Path:
    return paths.ensure(paths.RUNS_ROOT / "orbit" / "cache" / version)


def write_cache(df: pd.DataFrame, pix: dict, info: dict, version: str = CACHE_VERSION) -> Path:
    if df.protected.any():
        raise guard.ProtectedDataError("development cache must not contain protected rows")
    d = cache_dir(version)
    df.to_parquet(d / "table.parquet", index=False)
    np.savez(d / "pixels.npz", **pix)
    (d / "info.json").write_text(json.dumps(info, indent=1, default=str))
    return d


def load_dev(version: str = CACHE_VERSION, *, training: bool = True):
    """Development table + pixels. The guard re-checks that no protected row is present."""
    d = cache_dir(version)
    df = pd.read_parquet(d / "table.parquet")
    guard.check_rows(df.protected.to_numpy(bool), training=training, what="cached rows")
    # review C3: roles are recomputed from (lon, lat, date) with the frozen rule and must equal the cache
    SP.verify_split_file()
    re_ = SP.assign(df.lon.to_numpy(), df.lat.to_numpy(), df.date.to_numpy(), log=lambda *a: None)
    for c in ("role", "block", "tblock", "waterbody", "buffer_excluded", "protected"):
        if not np.array_equal(re_[c].to_numpy(), df[c].to_numpy()):
            raise RuntimeError(f"cached split column {c!r} differs from the frozen rule: rebuild the cache")
    apply_turbidity_unit_policy(df)       # caches built before protocol v0.3 (no-op if already applied)
    z = np.load(d / "pixels.npz")
    pix = {k: z[k] for k in z.files}
    return df, pix, json.loads((d / "info.json").read_text())


# ------------------------------------------------------------------------------------ GLORIA

GLORIA_TARGETS = {"chla": "Chla", "tss": "TSS", "turbidity": "Turbidity", "secchi": "Secchi_depth"}


def gloria_rows(log=print) -> pd.DataFrame:
    """GLORIA unflagged samples for pretraining: S2A-simulated band *reflectance* rho = pi * Rrs (to match
    L2A surface reflectance units), targets as above. Samples in val/test-role water bodies or dated
    >= 2022 are excluded (orbit_protocol.md §4.1) so pretraining never sees evaluation lakes or years."""
    g = pd.read_parquet(paths.processed("gloria") / "gloria_samples.parquet")
    g = g[~g.any_qc_flag.astype(bool)].copy()
    g = g[np.isfinite(g.Latitude.astype(float)) & np.isfinite(g.Longitude.astype(float))].copy()   # review W8
    t = pd.to_datetime(g.time_utc, utc=True, errors="coerce").dt.tz_localize(None)
    sp = SP.assign(g.Longitude.to_numpy(float), g.Latitude.to_numpy(float), t.fillna(pd.Timestamp("2000-01-01")).to_numpy(),
                   log=log)
    from .features import inner_block

    keep = (sp.role.to_numpy() == "train") & (t.dt.year.to_numpy() <= 2021) & t.notna().to_numpy() \
        & ~sp.buffer_excluded.to_numpy() & ~inner_block(sp.block)      # inner blocks: early-stopping only
    out = pd.DataFrame({"gloria_id": g.GLORIA_ID.astype(str).to_numpy(), "dataset_id": g.Dataset_ID.astype(str).to_numpy(),
                        "lat": g.Latitude.to_numpy(float), "lon": g.Longitude.to_numpy(float), "date": t.to_numpy(),
                        "waterbody": sp.waterbody.to_numpy(), "role": sp.role.to_numpy(),
                        "water_body_type": g.water_body_type_name.astype(str).to_numpy()})
    for b in GLORIA_BANDS:
        out[b] = np.pi * g[f"rrs_S2A_{b.replace('B0', 'B')}"].to_numpy(float)
    for k, c in GLORIA_TARGETS.items():
        out[k] = g[c].to_numpy(float)
    apply_valid_ranges(out)
    n0 = len(out)
    out = out[keep & out[list(TARGETS)].notna().any(axis=1).to_numpy()].reset_index(drop=True)
    log(f"[data] gloria: {n0} unflagged -> {len(out)} usable (train role, <= 2021, >= 1 target)")
    return out
