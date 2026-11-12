# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Instantaneous grab-sample rows for the M6 architecture-v2 label pool (amendment §2.2, §2.5; source L-grab, Tier 1).

Lock-at-extraction: station roles come from station metadata (the lever engineer's pinned ``lever_sites.parquet``);
sample **values are read only for stations with rule ``fit`` or ``inner_group``** (the ``site_id`` filter is pushed
into the parquet scan). Every other station appears as a count. Filters are the track-N filters
(``hydromet_protocol.md`` §12) exactly as the lever build applies them.

Row rules (fixed here, pinned with the output table):

* a sample time of exactly 00:00:00 local, or a missing time, is "no time" (placeholder; uniform 08–17 h mixture);
* per (station, date, variable): samples within a 2 h span collapse to one row (mean value, mean hour); a wider span
  with < ``LOGGER_MIN_HOURS`` distinct hours keeps the first timed sample (the rest are counted and dropped);
* **L-logger rule**: a station-day with ≥ ``LOGGER_MIN_HOURS`` distinct local solar hours is a logger day: it gives
  daily mean/min/max rows and instantaneous rows at the fixed hours {1, 5, 9, 13, 17, 21} (as the sonde hourly rows).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ...core import paths
from .. import reserved as R
from .. import sites as LS
from .solar import local_solar_hour

WQP = paths.PROCESSED / "wqp_discrete"
ND = paths.PROCESSED / "hydromet_wqp_national"
LEVER_SITES = paths.PROCESSED / "hydromet_sota" / "lever_sites.parquet"
VARS = {"temp": ("T", "deg C"), "do": ("DO", "mg/L")}
MAX_DEPTH_M = 2.0
NON_STATES = {"AK", "HI", "PR", "GU", "VI", "AS", "MP"}
VALUE_RULES = ("fit", "inner_group")          # the only station rules whose sample values may be read
PROVIDER_CLASSES = ("usgs", "state", "tribal", "local", "volunteer", "other")
LOGGER_MIN_HOURS = 20
FIXED_HOURS = (1, 5, 9, 13, 17, 21)
SAME_VISIT_SPAN_H = 2.0

_TRIBAL = re.compile(r"\b(tribe|tribes|tribal|nation|pueblo|rancheria|band of|indian|indians|reservation)\b", re.I)
_VOLUNT = re.compile(r"(volunteer|citizen|riverkeeper|waterkeeper|baykeeper|river ?watch|stream ?watch|stream team|"
                     r"friends of|adopt[- ]a|watershed (association|alliance|council|coalition|group|partnership)|"
                     r"conservancy|trout unlimited|izaak walton|save our|school)", re.I)
_STATE = re.compile(r"\b(state|department of|dept\.? of|dnr|deq|dep|dec|dem|dhec|tceq|commission|commonwealth|"
                    r"pollution control|environmental (quality|protection|conservation|management|services)|"
                    r"natural resources|water resources (board|control))\b", re.I)
_LOCAL = re.compile(r"\b(county|city|town|township|village|borough|parish|municipal|metropolitan|district|authority|"
                    r"utility|utilities|sanitation|sewer|regional|council of governments)\b", re.I)


def provider_class(provider, organization, organization_name, tribal_land) -> str:
    """Fixed mapping of (WQP provider, organisation id, organisation name, tribal-land flag) to
    :data:`PROVIDER_CLASSES`; first match wins: USGS → tribal → volunteer → state → local → other."""
    prov, org = str(provider or "").upper(), str(organization or "").upper()
    name = str(organization_name or "")
    if prov == "NWIS" or org.startswith("USGS"):
        return "usgs"
    if str(tribal_land or "").strip().lower() in ("yes", "y", "true") or _TRIBAL.search(name):
        return "tribal"
    if _VOLUNT.search(name):
        return "volunteer"
    if org.startswith("21") or _STATE.search(name):   # legacy STORET state organisation ids start with "21"
        return "state"
    if _LOCAL.search(name):
        return "local"
    return "other"


def tier1_stations() -> pd.DataFrame:
    """Track-N-set WQP stations of the lever pool with their rule; adds lat/lon and the provider class. Metadata
    only. Stations with a rule outside :data:`VALUE_RULES` are kept in the frame for counts."""
    pool = pd.read_parquet(LEVER_SITES)
    pool = pool[pool["source"] == "wqp_grab"].copy()
    st = pd.read_parquet(ND / "stations.parquet").set_index("station_key")
    meta = pd.read_parquet(WQP / "_stations.parquet", columns=["site_id", "organization", "organization_name",
                                                                 "provider", "tribal_land"])
    meta = meta.drop_duplicates("site_id").set_index("site_id")
    pool["lat"] = st["lat"].reindex(pool["key"]).to_numpy()
    pool["lon"] = st["lon"].reindex(pool["key"]).to_numpy()
    m = meta.reindex(pool["site_no"])
    prov = st["provider"].reindex(pool["key"]).to_numpy()
    pool["provider_class"] = [provider_class(p, o, n, t) for p, o, n, t in
                              zip(prov, m["organization"].to_numpy(), m["organization_name"].to_numpy(),
                                  m["tribal_land"].to_numpy())]
    trackn = LS.load_trackn_roles()
    role = pool["key"].map(trackn["role_of"])
    bad = pool["rule"].isin(VALUE_RULES) & (role != "train")
    if bad.any():
        raise RuntimeError(f"{int(bad.sum())} fit/inner grab stations without a track-N train role")
    return pool.reset_index(drop=True)


def read_samples(stations: pd.DataFrame, *, lo=LS.TRAIN_START, hi=LS.TRAIN_END, log=print) -> pd.DataFrame:
    """Accepted grab samples (track-N filters) in [lo, hi] at the stations whose rule is in :data:`VALUE_RULES`.
    Returns one row per sample: site_id, date, var (T | DO), value, hour (local solar; NaN = no time)."""
    import pyarrow.dataset as ds

    ok = stations[stations["rule"].isin(VALUE_RULES)]
    ids = sorted(ok["site_no"].astype(str))
    R.assert_no_reserved(ids)
    from ...data.reserved import assert_no_reserved as shared_assert_no_reserved

    shared_assert_no_reserved(ids, where="M6 arch-v2 grab-sample extraction")
    res = R.reserved_usgs_sites()
    if any(LS.contains_reserved_id(k, res) for k in ok["key"]):
        raise R.ReservedSiteError("reserved number inside a grab station key")
    if pd.Timestamp(hi) > pd.Timestamp(LS.TRAIN_END) or pd.Timestamp(lo) < pd.Timestamp(LS.TRAIN_START):
        raise ValueError("grab rows are read from the train block only")
    lon = ok.set_index("site_no")["lon"]
    out = []
    lo_d, hi_d = pd.Timestamp(lo).date(), pd.Timestamp(hi).date()
    for v, (name, unit) in VARS.items():
        d = ds.dataset(WQP / f"variable={v}" / "data.parquet")
        flt = ((ds.field("site_class") == "stream") & (ds.field("activity_start_date") >= lo_d)
               & (ds.field("activity_start_date") <= hi_d) & ds.field("site_id").isin(ids))
        t = d.to_table(columns=["site_id", "state", "activity_start_date", "activity_start_time", "time_utc", "value",
                                "unit", "censored", "depth_m", "activity_type", "status_class"], filter=flt).to_pandas()
        n0 = len(t)
        t = t[(t["unit"] == unit) & (t["status_class"] == "accepted") & ~t["censored"].fillna(False).astype(bool)
              & np.isfinite(t["value"]) & ~t["activity_type"].fillna("").str.contains("Time Series|Quality Control")
              & (t["depth_m"].isna() | (t["depth_m"] <= MAX_DEPTH_M)) & ~t["state"].isin(NON_STATES)]
        t["date"] = pd.to_datetime(t["activity_start_date"])
        tu = pd.to_datetime(t["time_utc"], utc=True)
        utc_h = tu.dt.hour + tu.dt.minute / 60.0 + tu.dt.second / 3600.0
        hour = local_solar_hour(utc_h.to_numpy(dtype="float64"), lon.reindex(t["site_id"]).to_numpy(dtype="float64"))
        no_time = tu.isna().to_numpy() | t["activity_start_time"].fillna("").astype(str).str.strip().isin(
            ["", "00:00:00", "00:00"]).to_numpy()
        hour[no_time] = np.nan
        out.append(pd.DataFrame({"site_id": t["site_id"].to_numpy(), "date": t["date"].to_numpy(), "var": name,
                                 "value": t["value"].to_numpy(dtype="float64"), "hour": hour}))
        log(f"[v2.grab] {v}: {n0} rows at value-readable stations -> {len(t)} after filters; "
            f"{int(no_time.sum())} without a time")
    return pd.concat(out, ignore_index=True)


def collapse(samples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Apply the row rules of the module docstring. Returns ``(inst, daily, counts)``: ``inst`` has one row per
    (site_id, date, var[, fixed hour on logger days]) with value and hour; ``daily`` holds logger-day mean/min/max."""
    s = samples.sort_values(["site_id", "var", "date", "hour"], kind="mergesort", na_position="last")
    g = s.groupby(["site_id", "var", "date"], sort=False)
    agg = g.agg(n=("value", "size"), vmean=("value", "mean"), vmin=("value", "min"), vmax=("value", "max"),
                hmin=("hour", "min"), hmax=("hour", "max"), hmean=("hour", "mean"), n_timed=("hour", "count"),
                first_val=("value", "first"), first_hour=("hour", "first")).reset_index()
    nh = s.assign(hb=np.floor(s["hour"])).dropna(subset=["hb"]).groupby(["site_id", "var", "date"])["hb"].nunique()
    agg["n_hours"] = nh.reindex(pd.MultiIndex.from_frame(agg[["site_id", "var", "date"]])).fillna(0).to_numpy()
    span = (agg["hmax"] - agg["hmin"]).fillna(0.0)
    logger = agg["n_hours"] >= LOGGER_MIN_HOURS
    visit = ~logger & (span <= SAME_VISIT_SPAN_H)
    wide = ~logger & ~visit
    inst = [pd.DataFrame({"site_id": agg.loc[visit, "site_id"], "date": agg.loc[visit, "date"],
                          "var": agg.loc[visit, "var"], "value": agg.loc[visit, "vmean"],
                          "hour": agg.loc[visit, "hmean"], "kind": "grab"}),
            pd.DataFrame({"site_id": agg.loc[wide, "site_id"], "date": agg.loc[wide, "date"],
                          "var": agg.loc[wide, "var"], "value": agg.loc[wide, "first_val"],
                          "hour": agg.loc[wide, "first_hour"], "kind": "grab"})]
    daily = agg.loc[logger, ["site_id", "date", "var", "vmean", "vmin", "vmax"]].rename(
        columns={"vmean": "mean", "vmin": "min", "vmax": "max"})
    if logger.any():
        key = pd.MultiIndex.from_frame(agg.loc[logger, ["site_id", "var", "date"]])
        sl = s[pd.MultiIndex.from_frame(s[["site_id", "var", "date"]]).isin(key) & s["hour"].notna()].copy()
        sl["hb"] = np.floor(sl["hour"]).astype(int)
        sl = sl[sl["hb"].isin(FIXED_HOURS)]
        fx = sl.groupby(["site_id", "var", "date", "hb"], as_index=False).agg(value=("value", "mean"), hour=("hour", "mean"))
        inst.append(fx[["site_id", "date", "var", "value", "hour"]].assign(kind="logger"))
    inst = pd.concat(inst, ignore_index=True)
    counts = {"sample_rows": int(len(s)), "station_days": int(len(agg)), "visit_days": int(visit.sum()),
              "wide_span_days": int(wide.sum()), "rows_dropped_in_wide_span_days": int((agg.loc[wide, "n"] - 1).sum()),
              "logger_days": int(logger.sum()), "inst_rows": int(len(inst)),
              "inst_rows_without_time": int(inst["hour"].isna().sum())}
    return inst, daily.reset_index(drop=True), counts
