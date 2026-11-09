# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""D1 SafeSwim — Chicago beach fecal-indicator exceedance (docs/preregistration/downstream_safeswim.md).

Sub-tasks (units are never pooled, RESEARCH_PLAN §8 / datasheet §7.1):

* **D1a** culture *E. coli* > 235 CFU/100 mL, 2006–2016, **morning forecast** for beach-day *d*:
  culture results from samples taken on day ≤ d−1 are allowed (18–24 h lab time), same-day culture
  never.
* **D1b** qPCR *Enterococci* > 1,000 CCE/100 mL, 2015–2026 (2020 excluded):
  - ``nowcast`` (afternoon, same information time as the City's official "DNA Model"): same-day qPCR at
    the 5 model-input beaches is allowed; labels are scored at the non-input beaches only;
  - ``forecast`` (morning): no same-day qPCR anywhere.

Decision-time rules are enforced here (:func:`network_features` refuses same-day culture and any
same-day value in forecast mode; persistence/network "previous" values come from strictly earlier
dates). Weather is observed gridMET with windows ending d−1 for both modes (the gridMET day ends at
07 UTC the next day, so day *d* is not complete at decision time). Phase 1 uses observed covariates
only (no upstream-model features; :func:`sentinel.models.downstream.crossfit.phase1_no_upstream`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...core import paths
from ...data.sources import downstream_chicago_beaches as CB
from . import common as C
from . import weather as W

APP = "safeswim"
DATA_MANIFESTS = ("labels_chicago_beaches", "forcings_daily", "safeswim_drivers")
RESERVE_YEARS = {"d1b": ("2026",)}   # test-role reserve season: not loaded by the frozen look (protocol §11)
INPUT_BEACHES = tuple(CB.MODEL_INPUT_BEACHES)          # calumet, 63rdstreet, rainbow, montrose, southshore
EXCLUDED_BEACHES = ("humboldt",)                        # inland lagoon (datasheet §7.5), not Lake Michigan
THRESHOLDS = {"d1a": CB.CULTURE_BAV_CFU, "d1b": CB.QPCR_ADVISORY_CCE}
KIND = {"d1a": "culture", "d1b": "qpcr"}
PERSIST_MAX_GAP_D = 3                                   # research/07 §b.2 eligibility rule
# Forward-chaining year roles (protocol §3). 2020 (COVID, 47 qPCR tests) is in no role for D1b.
D1A_YEARS = {**{y: "train" for y in range(2006, 2014)}, 2014: "val", 2015: "val", 2016: "test"}
D1B_YEARS = {**{y: "train" for y in (2015, 2016, 2017, 2018, 2019, 2021)}, 2022: "val", 2023: "val",
             2024: "test", 2025: "test", 2026: "test"}
LAT_BANDS = ((0, "south", -90.0, 41.85), (1, "central", 41.85, 41.95), (2, "north", 41.95, 90.0))
CLUSTER_UNIT = "calendar ISO week (all beaches of that week together: beaches share weather and lake state)"
LOG_FLOOR = 1.0
SPLIT_SHA = "faabfff9bfbe6d5d72cca8c30f26c0ff79faeb8d1771626bbfd8d3dc854b25b9"   # protocol §1


# --------------------------------------------------------------------------- labels
def _P():
    return paths.PROCESSED / "labels_chicago_beaches"


def load_tests(kind: str) -> pd.DataFrame:
    f = {"culture": "culture_ecoli.parquet", "qpcr": "qpcr_enterococci.parquet"}[kind]
    return pd.read_parquet(_P() / f)


def beach_days(tests: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Beach-day label table: geometric mean of the day's tests (values floored at 1 before logging,
    as in ``daily_qpcr``); ``y`` = gmean > threshold. Repeat tests on one day are one unit (no
    weighting by sample)."""
    d = tests[tests["date"].notna() & tests["result"].notna() & tests["beach_key"].notna()].copy()
    d = d[~d["beach_key"].isin(EXCLUDED_BEACHES)]
    d["logv"] = np.log10(d["result"].clip(lower=LOG_FLOOR))
    g = d.groupby(["beach_key", "date"], as_index=False).agg(log_gmean=("logv", "mean"), n_tests=("logv", "size"))
    g["gmean"] = 10 ** g["log_gmean"]
    g["y"] = (g["gmean"] > threshold).astype(int)
    g["date"] = pd.to_datetime(g["date"])
    g["beach_key"] = g["beach_key"].astype(str)
    return g.sort_values(["beach_key", "date"]).reset_index(drop=True)


def beach_table() -> pd.DataFrame:
    b = pd.read_parquet(_P() / "beaches.parquet")
    b = b[b["latitude"].notna() & ~b["beach_key"].isin(EXCLUDED_BEACHES)].copy()
    b["lat_band"] = [next(i for i, _, lo, hi in LAT_BANDS if lo <= la < hi) for la in b["latitude"]]
    return b


# --------------------------------------------------------------------------- split
def split_assignments() -> tuple[dict, dict]:
    b = beach_table()
    assignments = {
        "d1a_year_role": {str(y): r for y, r in D1A_YEARS.items()},
        "d1b_year_role": {str(y): r for y, r in D1B_YEARS.items()},
        "beach_lat_band": {k: int(v) for k, v in zip(b["beach_key"], b["lat_band"])},
    }
    pred_beaches = sorted(b.loc[b["is_predicted_beach"], "beach_key"])
    man = (paths.MANIFESTS / "labels_chicago_beaches.json")
    import json

    files = {f["path"].split("/")[-1]: f.get("sha256") for f in json.loads(man.read_text())["files"]
             if f["path"].endswith(".parquet")}
    meta = {
        "tasks": {"d1a": "culture E. coli beach-day gmean > 235 CFU/100 mL; morning forecast",
                  "d1b": "qPCR Enterococci beach-day gmean > 1000 CCE/100 mL; nowcast (same-day input-beach "
                         "qPCR allowed) and morning forecast"},
        "unit": "beach-day (geometric mean of the day's tests); role = year role of the date",
        "excluded_years": {"d1b": [2020]}, "excluded_beaches": list(EXCLUDED_BEACHES),
        "d1b_input_beaches": list(INPUT_BEACHES),
        "d1b_eval_beaches": [x for x in pred_beaches if x not in INPUT_BEACHES],
        "lat_bands": {str(i): [name, lo, hi] for i, name, lo, hi in LAT_BANDS},
        "persistence_max_gap_days": PERSIST_MAX_GAP_D, "bootstrap_cluster": CLUSTER_UNIT,
        "bootstrap_cluster_secondary": "beach (claim requires CI > 0 under both week and beach clusters)",
        "primary_comparisons": {"d1b_nowcast": ["official_city_model", "feature_beach_gm"],
                                "d1b_forecast": ["persistence"], "d1a_forecast": ["persistence"]},
        "label_vintage_sha256": files,
    }
    return assignments, meta


def load_frozen_split() -> tuple[dict, str]:
    """The protocol's split (sha verified) after checking the label-file sha256 values it recorded."""
    from ...core.registry import file_sha256

    split, sha = C.load_split(APP, expect_sha=SPLIT_SHA)
    for name, want in split["meta"]["label_vintage_sha256"].items():
        got = file_sha256(_P() / name)
        if got != want:
            raise C.ProtectedDataError(f"label file {name} sha256 {got[:12]} != frozen {str(want)[:12]}")
    return split, sha


def fast_auroc(y, s) -> float:
    """Rank AUROC (ties averaged); NaN if one class is absent. Equivalent to sklearn roc_auc_score."""
    from scipy.stats import rankdata

    y = np.asarray(y, float)
    s = np.asarray(s, float)
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def freeze() -> str:
    a, m = split_assignments()
    return C.freeze_split(APP, a, m)


def assign_roles(bd: pd.DataFrame, task: str, split: dict) -> pd.DataFrame:
    yr = split["assignments"][f"{task}_year_role"]
    out = bd.copy()
    out["role"] = out["date"].dt.year.astype(str).map(yr)          # NaN for excluded years
    out["lat_band"] = out["beach_key"].map(split["assignments"]["beach_lat_band"])
    out = out[out["lat_band"].notna()]          # beaches without coordinates (Columbia) are not label units
    if task == "d1b":
        out = out[out["beach_key"].isin(split["meta"]["d1b_eval_beaches"])]
    return out


# --------------------------------------------------------------------------- features
def persistence_features(bd: pd.DataFrame, max_gap: int = PERSIST_MAX_GAP_D) -> pd.DataFrame:
    """Previous beach-day at the same beach (strictly earlier date)."""
    s = bd.sort_values(["beach_key", "date"])
    g = s.groupby("beach_key")
    prev_date = g["date"].shift(1)
    out = pd.DataFrame(index=s.index)
    out["prev_log"] = g["log_gmean"].shift(1)
    out["prev_gap_d"] = (s["date"] - prev_date).dt.days
    out["prev_exceed"] = g["y"].shift(1)
    out["persist_eligible"] = (out["prev_gap_d"] <= max_gap).astype(int)
    if (prev_date >= s["date"]).any():
        raise AssertionError("persistence used a same-day or later value")
    return out.reindex(bd.index)


def network_features(query: pd.DataFrame, source_bd: pd.DataFrame, beaches, *, kind: str, same_day: bool,
                     prefix: str, exclude_own: bool = False) -> pd.DataFrame:
    """Network signal from ``beaches`` for each query (beach_key, date).

    ``same_day=True`` → mean log10 gmean over ``beaches`` on date d (only legal for qPCR nowcasts).
    ``same_day=False`` → the most recent network day strictly before d (``<prefix>gap_d`` = days back).
    """
    if same_day and kind != "qpcr":
        raise C.ProtectedDataError("decision-time rule: culture results are never same-day inputs")
    src = source_bd[source_bd["beach_key"].isin(list(beaches))]
    out = pd.DataFrame(index=query.index)
    if exclude_own:
        # per-query exclusion of the query's own beach
        rows = []
        for bk, qs in query.groupby("beach_key"):
            sub = network_features(qs, src[src["beach_key"] != bk], beaches, kind=kind, same_day=same_day,
                                   prefix=prefix, exclude_own=False)
            rows.append(sub)
        return pd.concat(rows).reindex(query.index) if rows else out
    daily = src.groupby("date").agg(m=("log_gmean", "mean"), mx=("log_gmean", "max"), n=("log_gmean", "size"),
                                    anyx=("y", "max"))
    if same_day:
        j = daily.reindex(query["date"].to_numpy())
        out[f"{prefix}mean_log"] = j["m"].to_numpy()
        out[f"{prefix}max_log"] = j["mx"].to_numpy()
        out[f"{prefix}n"] = j["n"].fillna(0).to_numpy()
        out[f"{prefix}any_exceed"] = j["anyx"].to_numpy()
        return out
    days = daily.index.to_numpy(dtype="datetime64[ns]")
    qd = query["date"].to_numpy(dtype="datetime64[ns]")
    pos = np.searchsorted(days, qd, side="left") - 1          # strictly earlier
    ok = pos >= 0
    take = np.where(ok, pos, 0)
    for col, name in (("m", "mean_log"), ("mx", "max_log"), ("n", "n"), ("anyx", "any_exceed")):
        v = daily[col].to_numpy(float)[take]
        out[f"{prefix}{name}"] = np.where(ok, v, np.nan)
    gap = (qd - days[take]).astype("timedelta64[D]").astype(float)
    out[f"{prefix}gap_d"] = np.where(ok, gap, np.nan)
    if (np.where(ok, gap, 1) <= 0).any():
        raise AssertionError("network 'previous' feature used a same-day or later value")
    return out


def input_beach_each(query: pd.DataFrame, inp_bd: pd.DataFrame, beaches: pd.DataFrame) -> pd.DataFrame:
    """Iteration feature (City "DNA Model" design, Lucius et al. 2019): the same-day log10 qPCR gmean at
    each of the 5 input beaches individually (NaN if not sampled), plus the value at the input beach
    nearest the label beach and that distance. Same-day qPCR only — nowcast mode only."""
    from ...eval.splits import haversine_km

    piv = inp_bd.pivot_table(index="date", columns="beach_key", values="log_gmean", aggfunc="mean")
    piv = piv.reindex(columns=list(INPUT_BEACHES))
    j = piv.reindex(query["date"].to_numpy())
    out = pd.DataFrame(j.to_numpy(), index=query.index, columns=[f"inpx_{b}" for b in INPUT_BEACHES])
    b = beaches.set_index("beach_key")
    ib = b.loc[list(INPUT_BEACHES)]
    near, dist = {}, {}
    for bk in query["beach_key"].unique():
        d = haversine_km(b.loc[bk, "latitude"], b.loc[bk, "longitude"], ib["latitude"].to_numpy(), ib["longitude"].to_numpy())
        near[bk], dist[bk] = INPUT_BEACHES[int(np.argmin(d))], float(np.min(d))
    nb = query["beach_key"].map(near)
    out["inpx_nearest_log"] = [out.at[i, f"inpx_{n}"] for i, n in zip(query.index, nb)]
    out["inpx_nearest_km"] = query["beach_key"].map(dist).to_numpy()
    return out


def best_corr_input(X: pd.DataFrame) -> pd.DataFrame:
    """Iteration feature (Lucius et al. 2019: beaches are *selectively* correlated): for each label beach,
    the input beach whose same-day log10 value correlates best with the label beach's log10 value over
    **training-role rows only**; returns that input beach's same-day value (``inpc_bestcorr_log``) and
    the training correlation (``inpc_bestcorr_r``)."""
    tr = X[X["role"] == "train"]
    pick, rr = {}, {}
    for bk, g in tr.groupby("beach_key"):
        cors = {b: g["log_gmean"].corr(g[f"inpx_{b}"]) for b in INPUT_BEACHES}
        cors = {b: c for b, c in cors.items() if np.isfinite(c)}
        if cors:
            pick[bk] = max(cors, key=cors.get)
            rr[bk] = cors[pick[bk]]
    out = pd.DataFrame(index=X.index)
    nb = X["beach_key"].map(pick)
    out["inpc_bestcorr_log"] = [X.at[i, f"inpx_{b}"] if isinstance(b, str) else np.nan for i, b in zip(X.index, nb)]
    out["inpc_bestcorr_r"] = X["beach_key"].map(rr).astype(float)
    return out


DRIVER_GAUGES = ("05536000", "05536290", "05536890")    # 05536105 ends 2018-04 (not in validation years)
DRIVER_GROUPS = ("q", "lake", "waves", "stn", "nbr")


def _drv():
    return paths.PROCESSED / "safeswim_drivers"


def _lag_join(dates: pd.Series, daily: pd.DataFrame, lag_days: int = 1) -> pd.DataFrame:
    """Values of a date-indexed daily table on d − ``lag_days`` (strictly before the decision day)."""
    j = daily.reindex((dates - pd.Timedelta(days=lag_days)).to_numpy())
    j.index = dates.index
    return j


def driver_features(bd: pd.DataFrame, bd_all: pd.DataFrame, beaches: pd.DataFrame) -> pd.DataFrame:
    """Iteration-2 hydrometeorological drivers (datasheet `safeswim_drivers`); every value ends d−1.

    * ``drv_q_*``: USGS daily-mean discharge (log10 ft³/s) on d−1, max over d−3…d−1, and the d−1 value
      relative to the median of d−31…d−2 (storm-flow flag) for 3 gauges (NB Chicago R at Niles, Little
      Calumet R at South Holland, Chicago S&S Canal at Lemont — large canal flow = storm diversion).
    * ``drv_lake_*``: Calumet Harbor daily mean water level (m IGLD85) on d−1 and its 7-day change.
    * ``drv_wave_*``: NDBC 45007 (offshore) and 45174 (Wilmette nearshore, 2015+) d−1 mean/max wave
      height, mean wind speed, onshore (easterly) wind component, water temperature.
    * ``drv_stn_*``: Chicago beach weather stations (63rd St, Oak St, Foster), hourly ``interval_rain``:
      d−1 lake-front max of daily totals and of the max hourly rain (intensity); 2015+ only.
    * ``drv_nbr_*``: the nearest other beach's result on exactly d−1 (NaN if not sampled) and distance.
    """
    out = pd.DataFrame(index=bd.index)
    dates = bd["date"]
    q = pd.read_parquet(_drv() / "usgs_discharge_daily.parquet")
    for site in DRIVER_GAUGES:
        s = q[q["site_no"] == site].set_index("date")["discharge_cfs"].sort_index()
        s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D"))
        lg = np.log10(s.clip(lower=0.1))
        tab = pd.DataFrame({f"drv_q_{site}_log": lg, f"drv_q_{site}_max3": lg.rolling(3, min_periods=2).max(),
                            f"drv_q_{site}_rel": lg - lg.shift(1).rolling(30, min_periods=20).median()})
        out = out.join(_lag_join(dates, tab))
    wl = pd.read_parquet(_drv() / "coops_water_level_daily.parquet").set_index("date")["water_level_m_igld"].sort_index()
    wl = wl.reindex(pd.date_range(wl.index.min(), wl.index.max(), freq="D"))
    out = out.join(_lag_join(dates, pd.DataFrame({"drv_lake_level_m": wl, "drv_lake_level_d7": wl - wl.shift(7)})))
    nd = pd.read_parquet(_drv() / "ndbc_daily.parquet")
    for st in ("45007", "45174"):
        t = nd[nd["station"] == st].set_index("date")[["wvht_mean_m", "wvht_max_m", "wspd_mean_ms", "onshore_wind_ms",
                                                       "wtmp_mean_c"]]
        out = out.join(_lag_join(dates, t.add_prefix(f"drv_wave_{st}_")))
    w = pd.read_parquet(_P() / "weather_sensors_long.parquet", columns=["sensor_name", "date", "variable", "value"])
    r = w[w["variable"] == "interval_rain"].copy()
    r["date"] = pd.to_datetime(r["date"])
    per = r.groupby(["sensor_name", "date"])["value"].agg(["sum", "max", "size"])
    per = per[per["size"] >= 18]                         # ≥ 18 of 24 hourly records for a valid day
    lake = per.groupby(level="date").agg(drv_stn_rain_total_mm=("sum", "max"), drv_stn_rain_maxhr_mm=("max", "max"))
    out = out.join(_lag_join(dates, lake))
    from ...eval.splits import haversine_km

    b = beaches.set_index("beach_key")
    have = [k for k in bd_all["beach_key"].unique() if k in b.index]
    nbr, nkm = {}, {}
    for bk in bd["beach_key"].unique():
        others = [k for k in have if k != bk]
        d = haversine_km(b.loc[bk, "latitude"], b.loc[bk, "longitude"], b.loc[others, "latitude"].to_numpy(),
                         b.loc[others, "longitude"].to_numpy())
        # beaches sharing coordinates (datasheet §7.5) are not informative neighbours: require > 0.2 km
        d = np.where(d > 0.2, d, np.inf)
        nbr[bk], nkm[bk] = others[int(np.argmin(d))], float(np.min(d))
    val = bd_all.set_index(["beach_key", "date"])["log_gmean"]
    keys = list(zip(bd["beach_key"].map(nbr), dates - pd.Timedelta(days=1)))
    out["drv_nbr_prev_d1_log"] = val.reindex(keys).to_numpy()
    out["drv_nbr_km"] = bd["beach_key"].map(nkm).to_numpy()
    return out


def calendar_features(bd: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=bd.index)
    out["doy"] = bd["date"].dt.dayofyear
    out["dow"] = bd["date"].dt.dayofweek
    return out


def weather_features(bd: pd.DataFrame, beaches: pd.DataFrame, clim_years) -> pd.DataFrame:
    """gridMET antecedent windows ending d−1 at the nearest available cell (anomalies vs a DOY
    climatology fitted on ``clim_years`` = the task's training years only)."""
    b = beaches.set_index("beach_key")
    nc = W.nearest_cells(b["latitude"], b["longitude"])
    nc.index = b.index
    q = bd[["beach_key", "date"]].join(nc, on="beach_key")
    series = W.load_series(q["wx_site_key"].dropna().unique(), start="2005-01-01")
    clim = W.doy_climatology(series, clim_years)
    f = W.antecedent_features(q, series, clim=clim, windows=(1, 2, 3, 7, 14), lag0=False)
    f["wx_cell_km"] = q["wx_cell_km"].to_numpy()
    return f


def lake_sensor_daily(long: pd.DataFrame | None = None, flat_run: int = 12) -> pd.DataFrame:
    """Lake-wide daily means of water temperature, turbidity and wave height (all water sensors), after
    masking flatline runs (≥ ``flat_run`` identical consecutive hourly values per sensor × variable, e.g.
    the constant 0.187 m Ohio St wave height in 2017) and zero turbidity. Index: local date."""
    if long is None:
        long = pd.read_parquet(_P() / "water_sensors_long.parquet")
    d = long[long["variable"].isin(["water_temperature", "turbidity", "wave_height"]) & long["value"].notna()].copy()
    d = d.sort_values(["sensor_name", "variable", "time_local"])
    run_id = (d["value"] != d.groupby(["sensor_name", "variable"])["value"].shift()).cumsum()
    run_len = d.groupby(run_id)["value"].transform("size")
    d = d[(run_len < flat_run) & ~((d["variable"] == "turbidity") & (d["value"] <= 0))]
    d["date"] = pd.to_datetime(d["date"])
    w = d.groupby(["date", "variable"])["value"].mean().unstack()
    return w.rename(columns=lambda c: f"lake_{c}")


def sensor_features(bd: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Previous-day (d−1) lake-wide sensor means (hourly sensors are near-real-time; d−1 is conservative)."""
    j = daily.reindex((bd["date"] - pd.Timedelta(days=1)).to_numpy())
    j.index = bd.index
    return j.add_suffix("_d1")


def build(task: str, mode: str, split: dict, *, with_sensors: bool = False, include_test: bool = False,
          split_sha: str | None = None) -> pd.DataFrame:
    """Feature table for ``task`` ∈ {d1a, d1b} and ``mode`` ∈ {forecast, nowcast}; test-role rows are
    dropped before any feature fitted on data (the climatology) is computed.

    ``include_test=True`` is for the single authorised test look only (``testlook.require_authorised``
    refuses without a valid FREEZE file and a claimed look); fitted feature parts stay train-only."""
    if include_test:
        from .testlook import require_authorised

        require_authorised(APP, split_sha)
    if task == "d1a" and mode != "forecast":
        raise ValueError("D1a (culture) is a morning forecast only")
    kind = KIND[task]
    bd_all = beach_days(load_tests(kind), THRESHOLDS[task])
    yr = split["assignments"][f"{task}_year_role"]
    dev_years = {y for y, r in yr.items() if r in (("train", "val", "test") if include_test else ("train", "val"))}
    dev_years -= set(RESERVE_YEARS.get(task, ()))    # reserve season never loaded (pre-test gate C-D1-1)
    # history (persistence / network) is restricted to development years: test-year rows never load
    bd_all = bd_all[bd_all["date"].dt.year.astype(str).isin(dev_years)]
    labels = assign_roles(bd_all, task, split)
    labels = labels[labels["role"].notna()] if include_test else C.drop_protected(labels)
    beaches = beach_table()
    feats = [calendar_features(labels)]
    pers = persistence_features(bd_all)            # history may include any earlier beach-day
    feats.append(pers.loc[labels.index])
    if task == "d1b":
        inp = bd_all[bd_all["beach_key"].isin(INPUT_BEACHES)]
        if mode == "nowcast":
            feats.append(network_features(labels, inp, INPUT_BEACHES, kind=kind, same_day=True, prefix="inp_sd_"))
            feats.append(input_beach_each(labels, inp, beaches))
        feats.append(network_features(labels, inp, INPUT_BEACHES, kind=kind, same_day=False, prefix="inp_prev_"))
        allb = bd_all["beach_key"].unique()
        feats.append(network_features(labels, bd_all, allb, kind=kind, same_day=False, prefix="net_prev_",
                                      exclude_own=True))
    else:
        allb = bd_all["beach_key"].unique()
        feats.append(network_features(labels, bd_all, allb, kind=kind, same_day=False, prefix="net_prev_",
                                      exclude_own=True))
    train_years = sorted(int(y) for y, r in split["assignments"][f"{task}_year_role"].items() if r == "train")
    feats.append(weather_features(labels, beaches, train_years))
    if with_sensors:
        feats.append(sensor_features(labels, lake_sensor_daily()))
    if (_drv() / "usgs_discharge_daily.parquet").exists():
        feats.append(driver_features(labels, bd_all, beaches))
    X = pd.concat([labels] + feats, axis=1)
    X = X.join(beaches.set_index("beach_key")[["latitude", "longitude"]], on="beach_key")
    if task == "d1b" and mode == "nowcast":
        X = pd.concat([X, best_corr_input(X)], axis=1)
    if task == "d1b":
        X = X.merge(official_predictions(), on=["beach_key", "date"], how="left")
    if not include_test:
        C.assert_no_test(X)
    return X.reset_index(drop=True)


def official_predictions() -> pd.DataFrame:
    p = pd.read_parquet(_P() / "predictions_official.parquet")
    p = p[p["date"].notna() & p["predicted_level"].notna()]
    p = p.groupby(["beach_key", "date"], as_index=False)["predicted_level"].mean()
    p["date"] = pd.to_datetime(p["date"])
    p["beach_key"] = p["beach_key"].astype(str)
    return p.rename(columns={"predicted_level": "official_pred"})


# --------------------------------------------------------------------------- baselines and models
FEATURE_SETS = ("base", "inp_each", "compact", "compact_each", "base_corr", "compact_corr")
COMPACT = ["doy", "latitude", "prev_log", "prev_gap_d", "wx_precip_sum_1d", "wx_precip_sum_3d", "wx_tmax_mean_3d"]
COMPACT_NOWCAST = ["inp_sd_mean_log", "inp_sd_max_log", "inp_sd_n"]
COMPACT_FORECAST = ["inp_prev_mean_log", "net_prev_mean_log", "net_prev_gap_d"]


def feature_columns(X: pd.DataFrame, task: str, mode: str, *, with_sensors: bool = False,
                    feature_set: str = "base") -> list[str]:
    """``feature_set`` = one of FEATURE_SETS, optionally followed by ``+<driver group>`` terms
    (DRIVER_GROUPS), e.g. ``base+q`` or ``compact_corr+q+waves``."""
    parts = feature_set.split("+")
    feature_set, groups = parts[0], parts[1:]
    if feature_set not in FEATURE_SETS or any(g not in DRIVER_GROUPS for g in groups):
        raise KeyError(f"unknown feature set {'+'.join(parts)}")
    if feature_set.startswith("compact"):
        cols = COMPACT + (COMPACT_NOWCAST if mode == "nowcast" else [c for c in COMPACT_FORECAST if c in X.columns])
    else:
        cols = ["doy", "dow", "latitude", "prev_log", "prev_gap_d", "prev_exceed"]
        cols += [c for c in X.columns if c.startswith(("net_prev_", "inp_prev_"))]
        if mode == "nowcast":
            cols += [c for c in X.columns if c.startswith("inp_sd_")]
        cols += [c for c in X.columns if c.startswith("wx_") and c != "wx_cell_km"]
    if feature_set.endswith("corr"):
        if mode != "nowcast":
            raise ValueError("best-correlated input-beach features exist only for the nowcast")
        cols += ["inpc_bestcorr_log", "inpc_bestcorr_r"]
    if feature_set.endswith("each"):
        if mode != "nowcast":
            raise ValueError("individual same-day input-beach features exist only for the nowcast")
        cols += [c for c in X.columns if c.startswith("inpx_")]
    if with_sensors:
        cols += [c for c in X.columns if c.startswith("lake_")]
    for g in groups:
        add = [c for c in X.columns if c.startswith(f"drv_{g}_")]
        if "role" in X:   # a driver column missing on > 50% of TRAINING rows is not usable (train-only rule)
            trm = X.loc[X["role"] == "train", add].isna().mean()
            add = [c for c in add if trm[c] <= 0.5]
        cols += add
    from .crossfit import phase1_no_upstream

    phase1_no_upstream(cols)
    if "official_pred" in cols or "y" in cols or "log_gmean" in cols or "gmean" in cols:
        raise AssertionError("label or incumbent leaked into features")
    return cols


def baseline_scores(train: pd.DataFrame, ev: pd.DataFrame, task: str, mode: str) -> dict[str, np.ndarray]:
    """Untrained / train-fitted reference scores on ``ev`` (NaN where a baseline is undefined)."""
    p0 = train["y"].mean()
    out = {"prevalence": np.full(len(ev), p0)}
    rate = train.groupby("beach_key")["y"].agg(["sum", "size"])
    k = 20.0
    beach_rate = (rate["sum"] + k * p0) / (rate["size"] + k)
    out["location_only"] = ev["beach_key"].map(beach_rate).fillna(p0).to_numpy(float)
    tm = train.assign(m=train["date"].dt.month).groupby(["beach_key", "m"])["y"].agg(["sum", "size"])
    prior = tm.index.get_level_values(0).map(beach_rate).to_numpy()
    clim = pd.Series((tm["sum"].to_numpy() + k * prior) / (tm["size"].to_numpy() + k), index=tm.index)
    key = list(zip(ev["beach_key"], ev["date"].dt.month))
    out["climatology_beach_month"] = np.array([clim.get(kk, beach_rate.get(kk[0], p0)) for kk in key], float)
    out["persistence"] = np.where(ev["persist_eligible"] == 1, ev["prev_log"], np.nan)
    out["rain_rule_1d"] = ev["wx_precip_sum_1d"].to_numpy(float)
    if task == "d1b":
        out["official_city_model"] = ev["official_pred"].to_numpy(float)
        if mode == "nowcast":
            out["feature_beach_gm"] = ev["inp_sd_mean_log"].to_numpy(float)
    return out


def sens_at_spec(y, s, spec: float = 0.85) -> float:
    """ROC-point sensitivity at the threshold giving specificity ≥ ``spec`` on these rows."""
    y = np.asarray(y, float)
    s = np.asarray(s, float)
    m = np.isfinite(y) & np.isfinite(s)
    y, s = y[m], s[m]
    if y.sum() == 0 or (1 - y).sum() == 0:
        return np.nan
    thr = np.quantile(s[y == 0], spec, method="higher")
    return float(np.mean(s[y == 1] > thr))


def cluster_ids(X: pd.DataFrame) -> np.ndarray:
    iso = X["date"].dt.isocalendar()
    return (iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)).to_numpy()


MODEL_CONFIGS = {
    "logistic_l2": {"C": 1.0, "max_iter": 2000},
    "gbm_hgb": {"max_iter": 300, "learning_rate": 0.05, "max_leaf_nodes": 15, "min_samples_leaf": 40,
                "l2_regularization": 1.0, "early_stopping": False, "random_state": 0},
}


def fit_predict(model: str, Xtr: pd.DataFrame, ytr: np.ndarray, Xev: pd.DataFrame, seed: int = 0) -> np.ndarray:
    """Pre-specified learners (protocol §5). Missing values: HGB native; logistic = train-median
    imputation + missingness indicators + standardisation fitted on train only."""
    cfg = dict(MODEL_CONFIGS[model])
    if model == "gbm_hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier

        cfg["random_state"] = seed
        m = HistGradientBoostingClassifier(**cfg)
        m.fit(Xtr.to_numpy(float), ytr)
        return m.predict_proba(Xev.to_numpy(float))[:, 1]
    if model == "logistic_l2":
        from sklearn.linear_model import LogisticRegression

        med = Xtr.median()
        miss_cols = [c for c in Xtr.columns if Xtr[c].isna().any()]

        def prep(X):
            Z = X.fillna(med)
            for c in miss_cols:
                Z[f"{c}__na"] = X[c].isna().astype(float)
            return Z

        Ztr, Zev = prep(Xtr), prep(Xev)
        mu, sd = Ztr.mean(), Ztr.std().replace(0, 1.0)
        m = LogisticRegression(**cfg)
        m.fit(((Ztr - mu) / sd).to_numpy(float), ytr)
        return m.predict_proba(((Zev - mu) / sd).to_numpy(float))[:, 1]
    raise KeyError(model)
