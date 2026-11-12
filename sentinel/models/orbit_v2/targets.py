# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Track N and auxiliary targets: label rules of docs/preregistration/orbit_arch_v2_amendment.md §2.1 and §2.4 as
ruled in docs/preregistration/orbit_arch_v2_lead_rulings.md (R-3, R-8, R-21, C-9; where the two differ, the rulings
win).

Sources: the harmonised, QC-passed tables ``processed/wqp_discrete/variable=<v>/data.parquet`` only (the sidecar
files ``_qc_rejects.parquet`` and ``_unmapped.parquet`` are never read; lead ruling 2026-11-10 13:21 EST; R-3).

Label rule, identical for every model and every role:

* fraction: TP and TN ``fraction_class == "total"``; other targets: every row of the table;
* surface result: the predicate of the unchanged ``sentinel.data.matchups.wqp_surface_samples`` on the harmonised
  ``depth_m`` (rejected results dropped; depth <= 1.0 m or unknown; a result flagged "bottom" never). The shared
  function cannot be called for these variables (it keeps its own target list only), so the predicate is written
  once here (:func:`surface_mask`) and ``tests/test_orbit_v2_targets.py`` checks that it selects the same results
  and gives the same medians as the shared function on a target both know;
* replicates (R-3): median of the quantified results per (site, local date, target, characteristic, unit), then
  the median across those groups per site-date (as ``orbit.data.wqp_rows`` does for the base targets). A table
  holds one unit (checked; another unit is refused), so the unit key cannot merge or split a group;
* censoring (C-9): a site-date with a quantified result uses the quantified median. A site-date whose surface
  results are ALL left-censored ("<") is "censored at the largest detection limit of that site-date" when that
  limit lies inside the a-priori range; otherwise it carries no value. A site-date without a quantified result
  whose results are not all left-censored (a right-censored result, a censored result of unknown type, an
  unreadable value) carries no value. Each outcome is counted per target (:func:`site_date_labels`);
* valid range (a priori, :data:`SPECS`; never revised): a quantified site-date value outside the range is set
  missing and is NOT replaced by a censored label;
* 100 m join (R-8, C-9; :data:`RULE_R8`): a row without a WQP site (AquaCorr) is joined to one WQP station, the
  nearest station within 100 m whose frozen-rule water body and split role equal the row's (ties: the lower
  station id), chosen from keys before any value is loaded; without such a station the row carries no value.
  This is a constant: there is no other setting (C-2).

Leakage rules (stricter than the amendment text; arch-v2 review finding F-LOADER-V2):

* split roles come from KEYS first. :func:`read_results` reads the value columns only for sites handed to it with a
  development role (train or val, not buffer-excluded) and only for dates of a development year block. It refuses
  any other request; there is no flag that opens protected rows here. The Test-S look reads its rows in its own
  module inside the look claim;
* :func:`read_keys` never reads ``value`` or ``detection_limit``. Protected rows are counted from it
  (:func:`key_counts`): site-dates with a quantified surface result. That count is an upper bound of the scored
  rows (the valid-range rule needs the value and is applied only inside the look);
* lake assignment for new points never writes the shared cache (:func:`lakes_readonly`).

No function here produces a validation number.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ...core import paths
from ...data import matchups as M
from ..orbit import split as SP


@dataclass(frozen=True)
class Spec:
    variable: str            # processed/wqp_discrete/variable=<variable>
    unit: str                # the one unit of the harmonised table
    lo: float                # valid range (a priori; amendment §2.1)
    hi: float
    fractions: tuple | None  # allowed fraction_class values (None: every row)
    family: str              # "track_n" | "aux"


SPECS = {
    "tp": Spec("tp", "mg/L as P", 0.001, 10.0, ("total",), "track_n"),
    "tn": Spec("tn", "mg/L as N", 0.01, 50.0, ("total",), "track_n"),
    "doc": Spec("doc", "mg/L as C", 0.1, 200.0, None, "track_n"),
    "turbidity_fnu": Spec("turbidity_fnu", "FNU", 0.05, 5000.0, None, "aux"),
    "ssc": Spec("ssc", "mg/L", 0.1, 5000.0, None, "aux"),
    "acdom440": Spec("acdom440", "1/m", 0.01, 50.0, None, "aux"),
}
TRACK_N = tuple(t for t, s in SPECS.items() if s.family == "track_n")
AUX = tuple(t for t, s in SPECS.items() if s.family == "aux")

KEY_COLS = ("site_id", "activity_start_date", "depth_m", "activity_relative_depth", "status_class",
            "fraction_class", "censored", "censor_type", "characteristic", "unit")
VALUE_COLS = ("value", "detection_limit")
DEV_ROLES = ("train", "val")
DEV_TBLOCKS = ("train", "val")
JOIN_RADIUS_M = 100.0        # amendment §2.4 (= protocol §2.1 de-duplication radius)
RULE_R8 = ("R-8 / C-9: the 100 m join uses one WQP station per row, the nearest station within 100 m whose "
           "frozen-rule water body and split role equal the row's (ties: the lower station id), chosen from keys; "
           "otherwise the row carries no value")
REPLICATE_KEYS = ("site_id", "date", "characteristic", "unit")     # R-3 (the target is the table)
LABEL_COLS = ("value", "cens", "dl", "n_results", "n_quant", "n_left", "n_char")


class TargetRuleError(RuntimeError):
    """A new-target label was requested in a way the rules above do not allow, or a source table does not have the
    schema the rules assume."""


def dev_date_max() -> _dt.date:
    """Last day of the development year blocks of the frozen split rule (``split.YEAR_BLOCKS``)."""
    his = [hi for name, _lo, hi in SP.YEAR_BLOCKS if name in DEV_TBLOCKS]
    if not his or any(h is None for h in his):
        raise TargetRuleError("split.YEAR_BLOCKS has no closed development block")
    return _dt.date(int(max(his)), 12, 31)


def source_file(target: str) -> Path:
    return paths.PROCESSED / "wqp_discrete" / f"variable={SPECS[target].variable}" / "data.parquet"


# ------------------------------------------------------------------------------------ rule (pure functions)

def _s(df: pd.DataFrame, col: str, default="") -> pd.Series:
    return df[col] if col in df.columns else pd.Series(default, index=df.index)


def surface_mask(df: pd.DataFrame) -> np.ndarray:
    """The row filter of the unchanged ``matchups.wqp_surface_samples`` for a depth-dependent target, on the
    harmonised ``depth_m`` [m]: QC activities and rejected results dropped; depth <= SURFACE_MAX_M or unknown; a
    result flagged "bottom" never."""
    if "depth_m" not in df.columns:
        raise TargetRuleError("result rows carry no depth_m column")
    qc = _s(df, "qc_class").fillna("").astype(str)
    st = _s(df, "status_class", "unknown").fillna("unknown").astype(str)
    d = pd.to_numeric(df["depth_m"], errors="coerce")
    rel = _s(df, "activity_relative_depth").fillna("").astype(str).str.lower()
    return ((qc == "") & (st != "rejected") & (d.isna() | (d <= M.SURFACE_MAX_M))
            & ~rel.str.contains("bottom")).to_numpy(bool)


def fraction_mask(df: pd.DataFrame, target: str) -> np.ndarray:
    fr = SPECS[target].fractions
    if fr is None:
        return np.ones(len(df), bool)
    return _s(df, "fraction_class").fillna("").astype(str).isin(fr).to_numpy(bool)


def _prepared(res: pd.DataFrame, target: str, need_values: bool) -> pd.DataFrame:
    sp = SPECS[target]
    df = res[fraction_mask(res, target) & surface_mask(res)].copy()
    df["date"] = pd.to_datetime(_s(df, "activity_start_date", pd.NaT), errors="coerce").dt.normalize()
    df = df[df["date"].notna()]
    units = sorted(set(_s(df, "unit").fillna("").astype(str)) - {sp.unit})
    if units:
        raise TargetRuleError(f"{target}: unit(s) {units} in the harmonised table; the rule assumes {sp.unit!r} only")
    df["site_id"] = df["site_id"].astype(str)
    df["characteristic"] = _s(df, "characteristic").fillna("").astype(str)
    df["unit"] = _s(df, "unit").fillna("").astype(str)
    cens = _s(df, "censored", False).astype("boolean").fillna(False).astype(bool)
    ctype = _s(df, "censor_type").fillna("").astype(str).str.strip()
    df["_cens"] = cens.to_numpy()
    df["_left"] = (cens & (ctype == "<")).to_numpy()
    if need_values:
        df["_v"] = pd.to_numeric(df["value"], errors="coerce")
        df["_dl"] = pd.to_numeric(df["detection_limit"], errors="coerce")
    return df


def per_characteristic(res: pd.DataFrame, target: str) -> pd.DataFrame:
    """One row per (site_id, date, characteristic, unit) (:data:`REPLICATE_KEYS`) with a surface result of the
    allowed fraction: ``q_med`` (median of the quantified results; NaN when none), ``n_results``, ``n_quant``,
    ``n_left_any`` (results flagged "<"), ``n_left`` (of those, with a positive finite DL) and ``dl`` (largest such
    DL)."""
    df = _prepared(res, target, need_values=True)
    quant = ~df["_cens"] & np.isfinite(df["_v"])
    left = df["_left"] & np.isfinite(df["_dl"]) & (df["_dl"] > 0)
    df["_q"] = df["_v"].where(quant)
    df["_l"] = df["_dl"].where(left)
    df["_nq"] = quant.astype(int)
    df["_nl"] = left.astype(int)
    df["_na"] = df["_left"].astype(int)
    return df.groupby(list(REPLICATE_KEYS), sort=True).agg(
        q_med=("_q", "median"), n_results=("_q", "size"), n_quant=("_nq", "sum"), n_left=("_nl", "sum"),
        n_left_any=("_na", "sum"), dl=("_l", "max")).reset_index()


def site_date_labels(res: pd.DataFrame, target: str) -> tuple[pd.DataFrame, dict]:
    """(labels, counts). ``labels``: one row per (site_id, date) that carries a label of ``target``:
    ``value`` (quantified, in range; NaN for a censored row), ``cens`` (left-censored at ``dl``), ``dl``,
    ``n_results``, ``n_quant``, ``n_left``, ``n_char``. ``counts``: site-dates by outcome (no value).

    C-9: a site-date without a quantified result is censored only when ALL its surface results are left-censored
    and the largest detection limit of the site-date lies inside the a-priori range."""
    sp = SPECS[target]
    pc = per_characteristic(res, target)
    g = pc.groupby(["site_id", "date"], sort=True).agg(
        value=("q_med", "median"), n_results=("n_results", "sum"), n_quant=("n_quant", "sum"),
        n_left=("n_left", "sum"), n_left_any=("n_left_any", "sum"), dl=("dl", "max"),
        n_char=("characteristic", "nunique")).reset_index()
    quant = np.isfinite(g["value"].to_numpy(float))
    v = g["value"].to_numpy(float)
    out_q = quant & ((v < sp.lo) | (v > sp.hi))
    dl = g["dl"].to_numpy(float)
    all_left = g["n_left_any"].to_numpy() == g["n_results"].to_numpy()
    mixed = ~quant & ~all_left & (g["n_left_any"].to_numpy() > 0)
    cens = ~quant & all_left & (g["n_left"].to_numpy() > 0)
    out_c = cens & ((dl < sp.lo) | (dl > sp.hi))
    g["value"] = np.where(quant & ~out_q, v, np.nan)
    g["cens"] = cens & ~out_c
    g["dl"] = np.where(g["cens"], dl, np.nan)
    keep = np.isfinite(g["value"].to_numpy(float)) | g["cens"].to_numpy(bool)
    counts = {"n_site_dates_with_surface_result": int(len(g)),
              "n_quantified_in_range": int((quant & ~out_q).sum()),
              "n_quantified_out_of_range_set_missing": int(out_q.sum()),
              "n_left_censored_dl_in_range": int((cens & ~out_c).sum()),
              "n_left_censored_dl_out_of_range_dropped": int(out_c.sum()),
              "n_not_all_results_left_censored_no_value": int(mixed.sum()),
              "n_no_usable_result_dropped": int((~quant & ~cens & ~mixed).sum())}
    return g.loc[keep, ["site_id", "date", *LABEL_COLS]].reset_index(drop=True), counts


def key_counts(keys: pd.DataFrame, target: str) -> pd.DataFrame:
    """Value-free count table from :func:`read_keys` rows: one row per (site_id, date) with a surface result of the
    allowed fraction; ``n_quant`` (results not flagged censored) and ``n_left`` (flagged "<"). No value and no
    detection limit is used, so the valid-range rule is not applied: an upper bound of the labelled site-dates."""
    for c in VALUE_COLS:
        if c in keys.columns:
            raise TargetRuleError(f"key_counts was handed a value column ({c})")
    df = _prepared(keys, target, need_values=False)
    df["_nq"] = (~df["_cens"]).astype(int)
    df["_nl"] = df["_left"].astype(int)
    return df.groupby(["site_id", "date"], sort=True).agg(n_quant=("_nq", "sum"), n_left=("_nl", "sum")).reset_index()


# ------------------------------------------------------------------------------------ readers

def _site_filter(site_ids) -> list:
    ids = sorted({str(s) for s in site_ids})
    if not ids:
        raise TargetRuleError("empty site list")
    return ids


def _read(target: str, columns, site_ids, date_min, date_max, path=None) -> pd.DataFrame:
    import pyarrow.dataset as ds
    import pyarrow as pa

    f = Path(path) if path else source_file(target)
    if f.name in ("_qc_rejects.parquet", "_unmapped.parquet"):
        raise TargetRuleError(f"{f.name} is not a harmonised, QC-passed result table")
    d = ds.dataset(f, format="parquet")
    missing = [c for c in columns if c not in d.schema.names]
    if missing:
        raise TargetRuleError(f"{f}: missing columns {missing}")
    flt = None
    if site_ids is not None:
        flt = ds.field("site_id").isin(pa.array(_site_filter(site_ids), type=pa.string()))
    for op, val in ((">=", date_min), ("<=", date_max)):
        if val is None:
            continue
        val = pd.Timestamp(val).date()
        e = ds.field("activity_start_date") >= val if op == ">=" else ds.field("activity_start_date") <= val
        flt = e if flt is None else flt & e
    return d.to_table(columns=list(columns), filter=flt).to_pandas()


def read_keys(target: str, site_ids=None, *, date_min=None, date_max=None, path=None) -> pd.DataFrame:
    """Key and rule columns of the harmonised table (:data:`KEY_COLS`). Never reads a value or a detection limit, so
    it may be used for any role (counts only)."""
    return _read(target, KEY_COLS, site_ids, date_min, date_max, path)


def check_dev_sites(sites: pd.DataFrame) -> list:
    """Site ids of ``sites`` (columns ``site_id``, ``role``, ``buffer_excluded``) after checking that every one has a
    development role and is not buffer-excluded. Raises on any other row: the caller filters, this function proves."""
    need = {"site_id", "role", "buffer_excluded"}
    if not need <= set(sites.columns):
        raise TargetRuleError(f"site table needs the columns {sorted(need)}")
    role = sites["role"].astype(str)
    bad = ~role.isin(DEV_ROLES) | sites["buffer_excluded"].astype(bool)
    if bad.any():
        raise TargetRuleError(f"{int(bad.sum())} of {len(sites)} sites are test-role, unassigned or buffer-excluded; "
                              "new-target values are read for development sites only")
    ids = sites["site_id"].astype(str)
    if ids[~pd.DataFrame({"s": ids, "r": role}).duplicated().to_numpy()].duplicated().any():
        raise TargetRuleError("a site id appears with two roles")
    return _site_filter(ids)


def read_results(target: str, dev_sites: pd.DataFrame, *, date_max=None, path=None) -> pd.DataFrame:
    """Result rows WITH values, for development sites and development dates only.

    ``dev_sites``: one row per site with ``site_id``, ``role`` and ``buffer_excluded`` from the frozen split rule
    (assigned from keys before this call). ``date_max`` may only lower the limit :func:`dev_date_max`."""
    ids = check_dev_sites(dev_sites)
    lim = dev_date_max()
    dmax = lim if date_max is None else pd.Timestamp(date_max).date()
    if dmax > lim:
        raise TargetRuleError(f"date_max {dmax} is after the last development day {lim}")
    df = _read(target, (*KEY_COLS, *VALUE_COLS), ids, None, dmax, path)
    if len(df):
        tb = SP.time_block(pd.to_datetime(df["activity_start_date"]))
        if not np.isin(tb, DEV_TBLOCKS).all() or not df["site_id"].astype(str).isin(ids).all():
            raise TargetRuleError("the filtered read returned a row outside the development sites / years")
    return df


def dev_labels(target: str, dev_sites: pd.DataFrame, *, date_max=None, path=None) -> tuple[pd.DataFrame, dict]:
    """:func:`site_date_labels` of the development rows of ``target`` (see :func:`read_results`)."""
    return site_date_labels(read_results(target, dev_sites, date_max=date_max, path=path), target)


# ------------------------------------------------------------------------------------ joins

def join_site_date(rows: pd.DataFrame, labels: pd.DataFrame, target: str, *, site_col: str = "site_id") -> pd.DataFrame:
    """Columns ``<target>``, ``<target>_cens``, ``<target>_dl`` for ``rows`` (same site and same local date).
    Rows without a label get NaN / False / NaN. The row order and index of ``rows`` are kept."""
    k = pd.MultiIndex.from_arrays([rows[site_col].astype(str).to_numpy(),
                                   pd.to_datetime(rows["date"]).dt.normalize().to_numpy()])
    lab = labels.set_index([labels["site_id"].astype(str), pd.to_datetime(labels["date"]).dt.normalize()])
    if lab.index.duplicated().any():
        raise TargetRuleError("labels are not unique per (site_id, date)")
    r = lab.reindex(k)
    return pd.DataFrame({target: r["value"].to_numpy(float),
                         f"{target}_cens": r["cens"].astype("boolean").fillna(False).to_numpy(bool),
                         f"{target}_dl": r["dl"].to_numpy(float)}, index=rows.index)


def _unit_xyz(lon, lat) -> np.ndarray:
    lo, la = np.radians(np.asarray(lon, float)), np.radians(np.asarray(lat, float))
    return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def nearest_station(rows: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """For rows without a WQP site (AquaCorr): :data:`RULE_R8`, from keys only.

    ``rows``: ``lon``, ``lat``, ``waterbody``, ``role``. ``stations``: ``site_id``, ``lon``, ``lat``, ``waterbody``,
    ``role`` (frozen-rule split columns of the station's own coordinates, :func:`site_roles`); the caller passes
    development stations only (:func:`check_dev_sites`). A station qualifies when it lies within
    :data:`JOIN_RADIUS_M` and its water body and split role equal the row's; the nearest qualifying station is
    taken, ties by the lower station id. There is no other setting (C-2).
    Returns ``join_site_id`` (None when there is no qualifying station) and ``join_dist_m``, aligned with ``rows``."""
    from scipy.spatial import cKDTree

    for name, df, cols in (("rows", rows, ("lon", "lat", "waterbody", "role")),
                           ("stations", stations, ("site_id", "lon", "lat", "waterbody", "role"))):
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise TargetRuleError(f"nearest_station: {name} lack the key columns {missing}")
    for c in VALUE_COLS:
        if c in rows.columns or c in stations.columns:
            raise TargetRuleError(f"nearest_station was handed a value column ({c}); the join is made from keys")
    radius_m = JOIN_RADIUS_M
    earth = 6_371_008.8
    out = pd.DataFrame({"join_site_id": pd.Series([None] * len(rows), dtype=object, index=rows.index),
                        "join_dist_m": np.full(len(rows), np.nan)}, index=rows.index)
    st = stations[np.isfinite(stations["lon"].to_numpy(float)) & np.isfinite(stations["lat"].to_numpy(float))]
    ok = np.isfinite(rows["lon"].to_numpy(float)) & np.isfinite(rows["lat"].to_numpy(float))
    if not len(st) or not ok.any():
        return out
    tree = cKDTree(_unit_xyz(st["lon"], st["lat"]))
    chord = 2.0 * np.sin(radius_m / earth / 2.0)
    q = _unit_xyz(rows["lon"].to_numpy(float)[ok], rows["lat"].to_numpy(float)[ok])
    sid = st["site_id"].astype(str).to_numpy()
    swb, srole = st["waterbody"].astype(str).to_numpy(), st["role"].astype(str).to_numpy()
    rwb, rrole = rows["waterbody"].astype(str).to_numpy()[ok], rows["role"].astype(str).to_numpy()[ok]
    pos = np.flatnonzero(ok)
    for i, cand in enumerate(tree.query_ball_point(q, chord)):
        cand = [c for c in cand if swb[c] == rwb[i] and srole[c] == rrole[i]]      # qualifying stations (R-8)
        if not cand:
            continue
        d = 2.0 * np.arcsin(np.clip(np.linalg.norm(tree.data[cand] - q[i], axis=1) / 2.0, 0, 1)) * earth
        d = np.round(d, 3)                             # distances equal to the millimetre are a tie
        j = int(np.lexsort((sid[cand], d))[0])         # nearest; ties by the lower station id (C-9)
        out.iloc[pos[i], 0] = sid[cand[j]]
        out.iloc[pos[i], 1] = float(d[j])
    return out


# ------------------------------------------------------------------------------------ split helpers

def lakes_readonly(lon, lat, log=print) -> pd.DataFrame:
    """``split.assign_lakes`` without writing the shared cache ``orbit/cache/lake_assign_v3_*.parquet`` (the dev
    build chain owns it): cached points are read, new points are assigned in memory only."""
    from ...data.sources import hydrolakes as H

    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    keys = SP.point_key(lon, lat)
    cp = SP._lake_cache_path()
    cache = pd.read_parquet(cp) if cp.exists() else pd.DataFrame(columns=["point_key"])
    uk, first = np.unique(keys, return_index=True)
    todo = np.setdiff1d(uk, cache["point_key"].to_numpy())
    if todo.size:
        sel = first[np.searchsorted(uk, todo)]
        lakes = SP._load_lakes()
        log(f"[targets] HydroLAKES assignment for {todo.size} points (in memory only)")
        a = H.assign_points(lon[sel], lat[sel], lakes, max_dist_m=SP.LAKE_MAX_DIST_M)
        b = H.assign_points(lon[sel], lat[sel], lakes, max_dist_m=SP.BUFFER_M)
        a["near_hylak_id"] = b["hylak_id"].to_numpy()
        a["near_dist_m"] = b["lake_dist_m"].to_numpy()
        a["near_pour_lon"] = b["lake_pour_lon"].to_numpy()
        a["near_pour_lat"] = b["lake_pour_lat"].to_numpy()
        a.insert(0, "point_key", todo)
        cache = pd.concat([cache, a], ignore_index=True) if len(cache) else a
    return cache.set_index("point_key").loc[keys].reset_index()


def site_roles(stations: pd.DataFrame, log=print) -> pd.DataFrame:
    """Split columns per station from its coordinates (keys only; ``lat_std`` / ``lon_std`` preferred, as the
    matchup builds do): ``site_id``, ``lon``, ``lat``, ``waterbody``, ``in_lake``, ``block``, ``role``,
    ``buffer_excluded``. Stations without coordinates get role "" and are never development sites."""
    st = stations.drop_duplicates("site_id").reset_index(drop=True)
    lat = st["lat_std"].fillna(st["lat"]) if "lat_std" in st.columns else st["lat"]
    lon = st["lon_std"].fillna(st["lon"]) if "lon_std" in st.columns else st["lon"]
    lon, lat = lon.to_numpy(float), lat.to_numpy(float)
    ok = np.isfinite(lon) & np.isfinite(lat)
    out = pd.DataFrame({"site_id": st["site_id"].astype(str), "lon": lon, "lat": lat, "waterbody": "",
                        "in_lake": False, "block": "", "role": "", "buffer_excluded": True})
    if ok.any():
        sp = SP.assign(lon[ok], lat[ok], np.full(int(ok.sum()), np.datetime64("2020-01-01")),
                       lakes=lakes_readonly(lon[ok], lat[ok], log=log), log=log)
        for c in ("waterbody", "in_lake", "block", "role", "buffer_excluded"):
            out.loc[ok, c] = sp[c].to_numpy()
    return out
