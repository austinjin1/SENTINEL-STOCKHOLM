# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Input assembly for the EW-v2 live trial (no model code here).

The live cube spans 2007-10-01 .. d (d = last complete day before the issue) and contains ONLY
  * the training block (WY2008–2021) from the frozen daily snapshot (climatology, training-year
    climatology deviations — exactly what the frozen EW-v2 score phase uses), and
  * the live window (from ``window_start``, normally 2026-10-01 = WY2027) from the live USGS pull
    (or, for the dry run / replay, from the snapshot).
Everything in between (WY2022–WY2026, i.e. the val block and the spent/reserved Test-A/Test-B blocks) is NaN:
site files are read whole and cut in memory, so those values never enter any array. The longest feature window
is 30 days (+3 days staleness, +7-day lags), so issues from 2026-12-01 depend on WY2027 inputs only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from live_common import REPO  # noqa: F401  (sets sys.path)
from sentinel.core import paths
from sentinel.data.sonde_daily import clean_long, splice_site
from sentinel.data.sources.usgs_legacy import ALL_PARAMS, STATS
from sentinel.eval.chloride import derived_series
from sentinel.models.sonde.data import SITE_DIR, Cube, climatology

TRAIN_END = pd.Timestamp("2021-09-30")
CUBE_START = pd.Timestamp("2007-10-01")
CHLORIDE_VARS = ("cl_mgl_mean", "cl_mgl_4dmean", "cl_mgl_max")
FORCINGS = paths.PROCESSED / "forcings_daily" / "forcings_daily.parquet"
WEATHER_COLS = ("precip_mm", "tmax_c", "tmin_c", "srad_wm2", "wind_ms", "vpd_kpa")
WEATHER_LAG_DAYS = 1
# OGC Water Data API qualifier handling, DERIVED from agreement with the legacy training tier on val-block data
# (scripts/live/derive_qualifier_map.py; review W-D1; exact names, no substring matching):
#   MASK — the legacy tier dropped these values (or they are censored / estimated / incomplete):
#          ESTIMATED, LESSTHAN, GREATERTHAN, EQUIP (71 % missing in legacy), MAINT (95 %); and, not observed on val
#          data, INCOMPLETE ([4]), MAINTENANCE, BACKWATER, FLOOD, PARTIAL (masked by the pre-specified rule);
#   KEEP — the legacy tier kept these values unchanged: FORCEINTERPOLATION (100 % equal), REVISED (90 %; 0 % missing on
#          hazard variables), ICE, DRY, SEASONAL, UNAVAIL, DISCONTINUED;
#   any other (unknown) name: MASKED and counted in the manifest.
MASK_OGC = {"ESTIMATED": "e", "LESSTHAN": "<", "GREATERTHAN": ">", "INCOMPLETE": "[4]", "EQUIP": "e", "MAINT": "e",
            "MAINTENANCE": "e", "BACKWATER": "e", "FLOOD": "e", "PARTIAL": "e"}
KEEP_OGC = frozenset({"FORCEINTERPOLATION", "REVISED", "ICE", "DRY", "SEASONAL", "UNAVAIL", "DISCONTINUED"})


# --------------------------------------------------------------------------- OGC daily -> frozen format
def ogc_long(records: list[dict]) -> tuple[pd.DataFrame, dict]:
    """OGC ``daily`` feature properties -> long table in the legacy DV layout used by ``clean_long``.
    The legacy qualifier string is synthesised as ``"P"``/``"A"`` + masked codes, so the frozen masking rule
    (``sonde_daily.MASK_QUALIFIER``) is applied unchanged. Unknown qualifier names are counted (reported)."""
    rows, unknown = [], {}
    for p in records:
        pc, st = p.get("parameter_code"), p.get("statistic_id")
        if pc not in ALL_PARAMS or st not in STATS:
            continue
        q = p.get("qualifier") or []
        if isinstance(q, str):   # stored as the string form of a list, e.g. "['EQUIP']"
            q = [t.strip(" '\"") for t in q.strip("[]").split(",") if t.strip(" '\"") and t.strip() != "None"]
        q = q if isinstance(q, list) else [q]
        codes = []
        for name in q:
            n = str(name).upper()
            if n in MASK_OGC:
                codes.append(MASK_OGC[n])
            elif n not in KEEP_OGC:                # pre-specified rule (review W-L6): unknown -> masked + reported
                unknown[n] = unknown.get(n, 0) + 1
                codes.append("e")
        appr = str(p.get("approval_status") or "").lower()
        qual = ("A" if appr == "approved" else "P") + ("" if not codes else " " + " ".join(codes))
        v = p.get("value")
        try:
            v = float(v) if v is not None else np.nan
        except (TypeError, ValueError):
            v = np.nan
        rows.append({"site_no": str(p.get("monitoring_location_id", "")).replace("USGS-", ""),
                     "date": pd.Timestamp(str(p.get("time"))[:10]), "parm_cd": pc, "stat_cd": st,
                     "ts_id": str(p.get("time_series_id")), "value": v, "qualifier": qual})
    return pd.DataFrame(rows, columns=["site_no", "date", "parm_cd", "stat_cd", "ts_id", "value", "qualifier"]), unknown


def apply_series_map(long: pd.DataFrame, smap: dict | None, window_end=None,
                     stale_days: int = 7) -> tuple[pd.DataFrame, dict]:
    """Frozen series map with fallback (review C-L3). For every (site, parameter, statistic) in the map, keep only
    the mapped series — UNLESS the mapped series has no record in the last ``stale_days`` days of the fetch window
    (sensor replaced / relocated / discontinued): then every series of that key is kept and the frozen
    ``splice_site`` rule decides. Fallback keys are listed in the returned info (issue manifest)."""
    if not smap or long.empty:
        return long, {"n_dropped_rows": 0, "unmapped_keys": 0, "fallback_keys": []}
    k = long["site_no"] + "|" + long["parm_cd"] + "|" + long["stat_cd"]
    mapped = k.map(smap)
    end = pd.Timestamp(window_end) if window_end is not None else long["date"].max()
    recent = long["date"] >= end - pd.Timedelta(days=stale_days - 1)
    alive = set(k[(mapped == long["ts_id"]) & recent & long["value"].notna()])
    fallback = sorted(set(k[mapped.notna()]) - alive)
    keep = mapped.isna() | (mapped == long["ts_id"]) | k.isin(fallback)
    return long[keep], {"n_dropped_rows": int((~keep).sum()), "unmapped_keys": int(k[mapped.isna()].nunique()),
                        "fallback_keys": fallback}


def wide_by_site(long: pd.DataFrame) -> tuple[dict, dict]:
    """Frozen QC (``clean_long``) + frozen series selection (``splice_site``) per site."""
    if long.empty:
        return {}, {}
    clean, counts = clean_long(long)
    out = {}
    for s, g in clean.groupby("site_no"):
        w, _ = splice_site(g)
        if not w.empty:
            out[str(s)] = w
    return out, counts


# --------------------------------------------------------------------------- cube
SNAPSHOT_DIR = {"dir": SITE_DIR.parent, "sha": None}   # set by issue.py from the frozen config


def _snapshot_site(s: str) -> pd.DataFrame | None:
    p = SNAPSHOT_DIR["dir"] / "sites" / f"{s}.parquet"
    pins = SNAPSHOT_DIR["sha"]
    if not p.exists():
        if pins is not None and s in pins:
            raise PermissionError(f"pinned snapshot site file {p} is missing")   # review W-L1
        return None
    if pins is not None:
        from live_common import sha256_file
        if pins.get(s) != sha256_file(p):
            raise PermissionError(f"snapshot site file {p} differs from the frozen pin")
    df = pd.read_parquet(p)
    df.index = pd.to_datetime(df.index)
    return df


def build_cube(sites: list[str], variables: list[str], end, *, window_start, window_frames: dict | None,
               window_from_snapshot: bool = False):
    """Cube (train block + live window) and, per site, the SC parts needed for derived chloride.
    ``window_frames``: site -> wide frame (live pull). ``window_from_snapshot``: take the window from the
    frozen snapshot instead (dry run on dates already in the snapshot; replay on the val block).
    Returns (cube, sc_parts) with sc_parts[site] = list of row frames (train part, window part) holding the
    float32 columns ``spcond_uscm_mean`` / ``spcond_uscm_max`` exactly as stored (rows = days with any record)."""
    dates = pd.date_range(CUBE_START, pd.Timestamp(end), freq="D")
    ws = pd.Timestamp(window_start)
    keep = (dates <= TRAIN_END) | (dates >= ws)
    S, T, V = len(sites), len(dates), len(variables)
    y = np.full((S, T, V), np.nan, np.float32)
    prov = np.zeros((S, T, V), bool)
    sc_parts = {}
    for i, s in enumerate(sites):
        snap = _snapshot_site(s)
        parts = []
        if snap is not None:
            parts.append(snap[snap.index <= TRAIN_END])
            if window_from_snapshot:
                parts.append(snap[(snap.index >= ws) & (snap.index <= pd.Timestamp(end))])
        if not window_from_snapshot and window_frames and s in window_frames:
            w = window_frames[s]
            parts.append(w[(w.index >= ws) & (w.index <= pd.Timestamp(end))])
        parts = [p_ for p_ in parts if len(p_)]
        if not parts:
            continue
        sc_parts[s] = [p_[[c for c in ("spcond_uscm_mean", "spcond_uscm_max") if c in p_.columns]] for p_ in parts]
        df = pd.concat(parts, axis=0, sort=False)
        df = df[~df.index.duplicated(keep="last")].reindex(dates)
        for j, v in enumerate(variables):
            if v in df.columns:
                col = df[v].to_numpy(dtype="float32")
                col[~keep] = np.nan          # never let WY2022–26 values into the cube
                y[i, :, j] = col
                if f"{v}_prov" in df.columns:
                    prov[i, :, j] = df[f"{v}_prov"].astype("boolean").fillna(False).to_numpy(dtype=bool) & keep
    assert not np.isfinite(y[:, ~keep, :]).any(), "WY2022–26 values leaked into the live cube"
    return Cube(list(sites), dates, list(variables), y, prov), sc_parts


def chloride_cube(cube: Cube, sc_parts: dict, regs: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Derived chloride exactly as ``scripts/data/build_chloride_derived.py`` (frozen per-site regression,
    ``chloride.derived_series`` on the stored float32 SC rows; the 4-day mean rolls over stored rows; no
    ``cl_mgl_max`` without an SC-max column), computed separately per data part, then cast to float32 as
    ``common.attach_chloride``. Provisional flags as ``attach_chloride``."""
    S, T = len(cube.sites), len(cube.dates)
    y = np.full((S, T, 3), np.nan, np.float32)
    for i, s in enumerate(cube.sites):
        if s not in regs.index or s not in sc_parts:
            continue
        fit = {"a": float(regs.loc[s, "a"]), "b": float(regs.loc[s, "b"])}
        for part in sc_parts[s]:
            if "spcond_uscm_mean" not in part:
                continue
            d = derived_series(part["spcond_uscm_mean"], part.get("spcond_uscm_max"), fit)
            d.index = pd.to_datetime(d.index)
            d = d.reindex(cube.dates)
            for k, v in enumerate(CHLORIDE_VARS):
                if v in d.columns:
                    val = d[v].to_numpy(np.float32)
                    m = np.isfinite(val)
                    y[i, m, k] = val[m]
    jsc = cube.var_index("spcond_uscm_mean")
    psc = cube.prov[:, :, jsc]
    p4 = psc.copy()
    for k in range(1, 4):
        p4[:, k:] |= psc[:, :-k]
    return y, np.stack([psc, p4, psc], axis=-1)


def append_chloride(cube: Cube, cm, cs, y_cl, prov_cl):
    cm_n, cs_n = climatology(Cube(cube.sites, cube.dates, list(CHLORIDE_VARS), y_cl, prov_cl))
    newcube = Cube(cube.sites, cube.dates, cube.variables + list(CHLORIDE_VARS),
                   np.concatenate([cube.y, y_cl], axis=-1), np.concatenate([cube.prov, prov_cl], axis=-1))
    return newcube, np.concatenate([cm, cm_n], axis=-1), np.concatenate([cs, cs_n], axis=-1)


# --------------------------------------------------------------------------- weather
def weather_array(cube: Cube, frames: list[pd.DataFrame], lag_days: int = WEATHER_LAG_DAYS) -> np.ndarray:
    """(S, T, 6) lagged observed weather as ``common.load_weather``: index t holds weather of day t − lag.
    ``frames``: long tables [site_key, date, *WEATHER_COLS]; later frames override earlier ones."""
    out = np.full((len(cube.sites), len(cube.dates), len(WEATHER_COLS)), np.nan, np.float32)
    if not frames:
        return out
    tb = pd.concat(frames, ignore_index=True)
    tb["date"] = pd.to_datetime(tb["date"])
    tb = tb.drop_duplicates(["site_key", "date"], keep="last")
    pos = {f"USGS-{s}": i for i, s in enumerate(cube.sites)}
    tb = tb[tb["site_key"].isin(pos)]
    tb["date"] = tb["date"] + pd.Timedelta(days=lag_days)
    tb = tb[tb["date"] <= cube.dates[-1]]
    si = tb["site_key"].map(pos).to_numpy()
    ti = cube.dates.get_indexer(pd.DatetimeIndex(tb["date"]))
    ok = ti >= 0
    for j, v in enumerate(WEATHER_COLS):
        out[si[ok], ti[ok], j] = tb[v].to_numpy(np.float32)[ok]
    return out


def forcings_window(sites: list[str], start, end) -> pd.DataFrame:
    import pyarrow.dataset as pds

    keys = [f"USGS-{s}" for s in sites]
    d0, d1 = pd.Timestamp(start).date(), pd.Timestamp(end).date()
    return pds.dataset(FORCINGS).to_table(
        filter=pds.field("site_key").isin(keys) & (pds.field("date") >= d0) & (pds.field("date") <= d1),
        columns=["site_key", "date", *WEATHER_COLS]).to_pandas()
