# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""D3 KillAlert — Iowa fish-kill risk from observed antecedent weather / flow (case-crossover).

Protocol: docs/preregistration/downstream_killalert.md. Design (docs/research/07 §d.3):

* **Cases** = Iowa DNR fish-kill events 2000+ with coordinates (`labels_iowa_fishkill`). Primary
  positive class ``is_env_wq`` (Environmental + Cyanobacteria Bloom); secondary all-cause.
* **Referents (time-stratified, same location)** = the event's calendar date ±``HALF_WINDOW`` days in
  every *other* year of the event's own year block (so referent weather never crosses a split boundary),
  excluding any referent day within ``EXCL_KM`` km and ±``EXCL_DAYS`` days of any recorded kill (any
  cause, any year). Each event + its referents is one matched set (``stratum``).
* **Features** = observed gridMET windows ending **d − 1** (sentinel.models.downstream.weather) with
  anomalies vs a day-of-year climatology fitted on train-block years only, and the nearest USGS gauge's
  log-discharge / water-temperature anomalies (also ending d − 1; climatology on train years).
* **Year blocks** (forward-chaining): train 2000–2009, val 2010–2017, test 2018–2026.

Also: the natural-vs-anthropogenic cause classifier (events with a known origin) and the
conditional-logistic / matched-set-concordance utilities.
"""

from __future__ import annotations

from ...data.reserved import assert_no_reserved, exclude_reserved

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

from ...core import paths
from ...eval.splits import haversine_km, spatial_kfold
from . import weather as W
from .common import N_THREADS as C_N_THREADS

APP = "killalert"
SPLIT_SHA = "4c7301c3144af503ab1ca72841ddfbbdda53e5ab30014150c28b092b72d0cf1e"
SEEDS = (0, 1, 2)
LATE_ENTRY_DAYS = 365
# v2 (2026-10-25, pre-registered before any v2 fit): Iowa + Delaware pooled; own frozen split file
SPLIT_V2_PATH_NAME = "downstream_killalert_v2_split.json"
SPLIT_SHA_V2 = "0edd117dd15ad02ca2203d8f36c079d7c00ff11856ed954b15ad8fb8337cc7ef"
BLOCKS = {"train": (2000, 2009), "val": (2010, 2017), "test": (2018, 2026)}
HALF_WINDOW = 3
EXCL_KM = 5.0
EXCL_DAYS = 30
GAUGE_MAX_KM = 30.0
CLIM_YEARS = range(BLOCKS["train"][0], BLOCKS["train"][1] + 1)
DATA_MANIFESTS = ["labels_iowa_fishkill", "forcings_daily", "sonde_daily", "sites"]

WX_FEATURES = [  # used by the full-weather models (names after weather.antecedent_features, prefix wx_)
    "wx_tmax_anom_1d", "wx_tmax_anom_3d", "wx_tmax_anom_7d", "wx_tmax_anom_14d", "wx_tmax_anom_30d",
    "wx_tmin_anom_7d", "wx_tmin_anom_30d", "wx_precip_anom_7d", "wx_precip_anom_30d",
    "wx_precip_sum_1d", "wx_precip_sum_3d", "wx_precip_sum_7d", "wx_precip_max1d_7d", "wx_dry_days_14d",
    "wx_tmax_mean_3d", "wx_tmax_mean_7d", "wx_srad_mean_3d", "wx_vpd_mean_3d", "wx_wind_mean_3d", "wx_wind_mean_7d",
]
GAUGE_FEATURES = ["g_logq_anom_7d", "g_logq_change_3d"]  # g_wtemp_anom_7d is built but ~99% missing in Iowa -> not a model input
TEMP_ANOM_FEATURES = ["ta7_summer", "ta30_summer", "ta7_winter", "ta30_winter"]  # Till/Tye-style baseline


def role_of_year(year: int) -> str | None:
    for r, (a, b) in BLOCKS.items():
        if a <= int(year) <= b:
            return r
    return None


# --------------------------------------------------------------------------- events
def _delaware_events() -> pd.DataFrame:
    d = pd.read_parquet(paths.PROCESSED / "labels_delaware_fishkill" / "events.parquet")
    d["page_created"] = pd.NaT
    return d


def load_events(min_year: int = 2000, states=("IA",)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(eligible events 2000+ with coordinates and valid date, all geocoded events of any year).
    ``states`` = ("IA",) is the v1 analysis; ("IA", "DE") the pooled v2 analysis."""
    e = pd.read_parquet(paths.PROCESSED / "labels_iowa_fishkill" / "events.parquet")
    e["state"] = "IA"
    if "DE" in states:
        e = pd.concat([e, _delaware_events()], ignore_index=True)
    e = e[e["state"].isin(list(states))]
    e["date"] = pd.to_datetime(e["date"])
    geo = e[e["latitude"].notna() & e["longitude"].notna() & e["date"].notna()].copy()
    elig = geo[geo["date"].dt.year >= min_year].copy()
    elig["event_id"] = elig["event_id"].astype(str)
    elig["role"] = elig["date"].dt.year.map(role_of_year)
    elig["is_env_wq"] = elig["is_env_wq"].astype(bool)
    elig["is_env_or_natsusp"] = elig["is_env_wq"] | elig["is_natural_suspected"].astype(bool)
    # review W3: placeholder dates (1 Jan, e.g. ice-out winterkills recorded at the year start) are
    # excluded from the primary analysis; very late database entry is a sensitivity filter
    elig["date_suspect"] = (elig["date"].dt.month == 1) & (elig["date"].dt.day == 1)
    pc = pd.to_datetime(elig["page_created"], errors="coerce")
    elig["late_entry"] = ((pc - elig["date"]).dt.days > LATE_ENTRY_DAYS).fillna(False)
    return elig.reset_index(drop=True), geo.reset_index(drop=True)


def _replace_year(d: pd.Timestamp, y: int) -> pd.Timestamp:
    if d.month == 2 and d.day == 29:
        return pd.Timestamp(year=y, month=2, day=28)
    return d.replace(year=y)


def referent_dates(event_date, years, half_window: int = HALF_WINDOW) -> list[pd.Timestamp]:
    """Same calendar date ±half_window days in each other year in ``years``."""
    d = pd.Timestamp(event_date)
    out = []
    for y in years:
        if y == d.year:
            continue
        c = _replace_year(d, int(y))
        out += [c + pd.Timedelta(days=k) for k in range(-half_window, half_window + 1)]
    return out


def near_any_event(dates, lat: float, lon: float, all_events: pd.DataFrame, *, km: float = EXCL_KM,
                   days: int = EXCL_DAYS) -> np.ndarray:
    """True where a candidate date at (lat, lon) is within ``km`` and ±``days`` of any recorded kill."""
    dates = pd.to_datetime(pd.Series(dates)).to_numpy()
    d = haversine_km(lat, lon, all_events["latitude"].to_numpy(float), all_events["longitude"].to_numpy(float))
    close = all_events.loc[d <= km, "date"].to_numpy()
    if not len(close):
        return np.zeros(len(dates), bool)
    gap = np.abs((dates[:, None] - close[None, :]).astype("timedelta64[D]").astype(int))
    return (gap <= days).any(axis=1)


def build_case_crossover(events: pd.DataFrame, all_events: pd.DataFrame, *, last_date: pd.Timestamp,
                         half_window: int = HALF_WINDOW) -> pd.DataFrame:
    """One row per case/referent day. Referent years = other years of the event's own block."""
    rows = []
    for ev in events.itertuples(index=False):
        a, b = BLOCKS[ev.role]
        # review W7: referent windows are clipped to the block's calendar years (a ±3-d window at a year
        # edge can otherwise spill into the neighbouring block)
        cands = [c for c in referent_dates(ev.date, range(a, b + 1), half_window)
                 if c <= last_date and a <= c.year <= b]
        cands = pd.DatetimeIndex(cands)
        if len(cands):
            bad = near_any_event(cands, ev.latitude, ev.longitude, all_events)
            cands = cands[~bad]
        base = {"stratum": ev.event_id, "event_id": ev.event_id, "role": ev.role, "lat": ev.latitude,
                "lon": ev.longitude, "event_date": ev.date, "is_env_wq": ev.is_env_wq,
                "is_env_or_natsusp": ev.is_env_or_natsusp, "cause_origin": ev.cause_origin,
                "waterbody_type": ev.waterbody_type, "est_location": bool(ev.est_location) if pd.notna(ev.est_location) else False,
                "date_suspect": bool(getattr(ev, "date_suspect", False)), "late_entry": bool(getattr(ev, "late_entry", False)),
                "state": getattr(ev, "state", "IA")}
        rows.append({**base, "date": ev.date, "is_case": 1})
        rows += [{**base, "date": c, "is_case": 0} for c in cands]
    cc = pd.DataFrame(rows)
    cc["n_referents"] = cc.groupby("stratum")["is_case"].transform(lambda s: int((s == 0).sum()))
    return cc


# --------------------------------------------------------------------------- features
LONG_WINDOWS = (1, 3, 7, 14, 30, 60, 90)
# Iteration it1 (2026-10-24, validation/CV only): winter-severity and long heat/drought windows
# (Till et al. 2019 / Tye et al. 2022: summerkill ~ heat extremes; winterkill ~ long cold/ice periods)
LONG_FEATURES = ["wx_tmax_anom_60d", "wx_tmax_anom_90d", "wx_tmin_anom_60d", "wx_tmin_anom_90d",
                 "wx_tmin_mean_60d", "wx_tmin_mean_90d", "wx_precip_anom_60d", "wx_precip_anom_90d"]


def add_weather(cc: pd.DataFrame, *, clim_years=CLIM_YEARS, lag_end_days: int = 1,
                windows=(1, 3, 7, 14, 30)) -> pd.DataFrame:
    """gridMET windows ending d - lag_end_days (default d-1) + anomalies (climatology on train years)."""
    cc = cc.copy()
    nc = W.nearest_cells(cc["lat"], cc["lon"])
    nc.index = cc.index
    cc[["wx_cell", "wx_site_key", "wx_cell_km"]] = nc
    series = W.load_series(cc["wx_site_key"].unique())
    clim = W.doy_climatology(series, clim_years)
    q = cc[["wx_site_key", "date"]].copy()
    q["date"] = q["date"] - pd.Timedelta(days=lag_end_days - 1)  # antecedent_features ends at q.date - 1
    f = W.antecedent_features(q, series, clim=clim, windows=windows)
    cc = pd.concat([cc, f], axis=1)
    m = cc["date"].dt.month
    summer, winter = m.isin([6, 7, 8, 9]).astype(float), m.isin([12, 1, 2, 3]).astype(float)
    cc["ta7_summer"], cc["ta30_summer"] = cc["wx_tmax_anom_7d"] * summer, cc["wx_tmax_anom_30d"] * summer
    cc["ta7_winter"], cc["ta30_winter"] = cc["wx_tmax_anom_7d"] * winter, cc["wx_tmax_anom_30d"] * winter
    doy = cc["date"].dt.dayofyear
    cc["doy_sin"], cc["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    cc["year_c"] = (cc["date"].dt.year - 2005) / 10.0          # review W1: secular-trend (reporting) term
    return cc


def iowa_gauges(var: str, min_days: int = 2000) -> pd.DataFrame:
    idx = pd.read_parquet(paths.PROCESSED / "sonde_daily" / "index.parquet")
    sites = pd.read_parquet(paths.PROCESSED / "sites" / "sites.parquet")
    ia = sites[sites["state"].isin(["IA", "MN", "WI", "IL", "MO", "NE", "SD", "DE", "MD", "PA", "NJ"])
               & sites["network"].eq("USGS")]
    q = idx[(idx["var"] == var) & (idx["n_days"] >= min_days)]
    g = ia[ia["site_id"].isin(set(q["site_no"]))][["site_id", "lat", "lon"]]
    g = g[g["site_id"].isin(exclude_reserved(g["site_id"].astype(str)))]   # reserved new-site block (index-driven)
    return g.reset_index(drop=True)


def gauge_anoms_from_series(x: pd.Series, clim_years, log: bool, lag_end_days: int = 1) -> pd.DataFrame:
    """Daily anomaly features from one gauge series (DatetimeIndex): 7-d mean anomaly vs a DOY
    climatology fitted on ``clim_years`` only, and the 3-d change — both ending d-1 (shift 1)."""
    x = x.asfreq("D")
    v = np.log10(x.clip(lower=0.01)) if log else x
    doy = np.minimum(v.index.dayofyear, 365)
    in_clim = v.index.year.isin(list(clim_years))
    clim = v[in_clim].groupby(doy[in_clim]).mean().reindex(range(1, 366))
    ext = pd.concat([clim.iloc[-15:], clim, clim.iloc[:15]]).rolling(31, center=True, min_periods=10).mean().iloc[15:-15]
    anom = v - ext.reindex(doy).to_numpy()
    out = pd.DataFrame(index=v.index)
    out["anom_7d"] = anom.rolling(7, min_periods=5).mean().shift(lag_end_days)            # ends d-lag
    out["change_3d"] = (v - v.shift(3)).shift(lag_end_days)
    return out


def _gauge_anoms(site_no: str, var: str, clim_years, log: bool, lag_end_days: int = 1) -> pd.DataFrame:
    assert_no_reserved([site_no], "killalert gauges")
    x = pd.read_parquet(paths.PROCESSED / "sonde_daily" / "sites" / f"{site_no}.parquet", columns=[var])[var]
    return gauge_anoms_from_series(x, clim_years, log, lag_end_days)


def add_gauges(cc: pd.DataFrame, *, clim_years=CLIM_YEARS, max_km: float = GAUGE_MAX_KM,
               lag_end_days: int = 1) -> pd.DataFrame:
    """Nearest-gauge (≤ max_km, straight line; not necessarily the same stream) flow / temperature anomalies."""
    cc = cc.copy()
    for var, name, log in (("discharge_cfs_mean", "q", True), ("temp_c_mean", "wtemp", False)):
        g = iowa_gauges(var, min_days=2000 if name == "q" else 600)
        d = haversine_km(cc["lat"].to_numpy()[:, None], cc["lon"].to_numpy()[:, None],
                         g["lat"].to_numpy()[None], g["lon"].to_numpy()[None])
        j = d.argmin(1)
        dist = d.min(1)
        cc[f"g_{name}_site"] = np.where(dist <= max_km, g["site_id"].to_numpy()[j], None)
        cc[f"g_{name}_km"] = dist
        vals_a, vals_c = np.full(len(cc), np.nan), np.full(len(cc), np.nan)
        for site, ix in cc.groupby(f"g_{name}_site").groups.items():
            an = _gauge_anoms(site, var, clim_years, log, lag_end_days)
            dates = cc.loc[ix, "date"].to_numpy()
            pos = cc.index.get_indexer(ix)
            vals_a[pos] = an["anom_7d"].reindex(dates).to_numpy()
            vals_c[pos] = an["change_3d"].reindex(dates).to_numpy()
        if name == "q":
            cc["g_logq_anom_7d"], cc["g_logq_change_3d"] = vals_a, vals_c
        else:
            cc["g_wtemp_anom_7d"] = vals_a
    return cc


def spatial_folds(events: pd.DataFrame, n_folds: int = 5, block_km: float = 50.0, seed: int = 0) -> dict:
    """50-km block folds over development (train+val) events."""
    dev = events[events["role"].isin(["train", "val"])]
    return spatial_kfold(list(dev["event_id"]), dev["latitude"].to_numpy(), dev["longitude"].to_numpy(),
                         n_folds=n_folds, block_km=block_km, seed=seed)


# --------------------------------------------------------------------------- models / metrics
def matched_concordance(y, s, strata) -> float:
    """Mean over matched sets (one case each) of P(score_case > score_referent) (+0.5 ties)."""
    df = pd.DataFrame({"y": np.asarray(y, float), "s": np.asarray(s, float), "g": np.asarray(strata)})
    df = df[np.isfinite(df["s"])]
    vals = []
    for _, sub in df.groupby("g"):
        cs, ct = sub.loc[sub["y"] == 1, "s"].to_numpy(), sub.loc[sub["y"] == 0, "s"].to_numpy()
        if len(cs) != 1 or not len(ct):
            continue
        vals.append(np.mean(ct < cs[0]) + 0.5 * np.mean(ct == cs[0]))
    return float(np.mean(vals)) if vals else float("nan")


class ConditionalLogit:
    """L2-penalised conditional logistic regression (one case per stratum), standardised inputs."""

    def __init__(self, l2: float = 1.0):
        self.l2 = l2

    def fit(self, X: np.ndarray, y: np.ndarray, strata: np.ndarray):
        X = np.asarray(X, float)
        self.mu_ = np.nanmean(X, 0)
        self.sd_ = np.nanstd(X, 0) + 1e-9
        Z = np.nan_to_num((X - self.mu_) / self.sd_)
        y = np.asarray(y, int)
        _, inv = np.unique(np.asarray(strata), return_inverse=True)
        groups = [np.flatnonzero(inv == k) for k in range(inv.max() + 1)]
        groups = [g for g in groups if y[g].sum() == 1 and len(g) > 1]

        def nll(b):
            eta = Z @ b
            ll = sum(eta[g][y[g] == 1][0] - logsumexp(eta[g]) for g in groups)
            return -ll + 0.5 * self.l2 * b @ b

        def grad(b):
            eta = Z @ b
            gr = np.zeros_like(b)
            for g in groups:
                w = np.exp(eta[g] - logsumexp(eta[g]))
                gr += Z[g][y[g] == 1][0] - w @ Z[g]
            return -gr + self.l2 * b

        res = minimize(nll, np.zeros(Z.shape[1]), jac=grad, method="L-BFGS-B")
        self.coef_ = res.x
        self.n_strata_ = len(groups)
        return self

    def decision_function(self, X):
        Z = np.nan_to_num((np.asarray(X, float) - self.mu_) / self.sd_)
        return Z @ self.coef_


def cause_frame(events: pd.DataFrame) -> pd.DataFrame:
    """Events with a known origin (Natural / Anthropogenic) → y_natural."""
    k = events[events["cause_origin"].isin(["Natural", "Anthropogenic"])].copy()
    k["y_natural"] = (k["cause_origin"] == "Natural").astype(int)
    return k


def cause_features(frame: pd.DataFrame, streamcat: pd.DataFrame | None = None) -> pd.DataFrame:
    """Report-time features: season, waterbody type, weather, gauge, optional StreamCat by COMID."""
    f = frame.copy()
    doy = f["date"].dt.dayofyear
    f["doy_sin"], f["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    for t in ("River/Stream", "Lake", "Pond"):
        f[f"wb_{t.split('/')[0].lower()}"] = (f["waterbody_type"] == t).astype(float)
    if streamcat is not None and "comid" in f:
        f = f.merge(streamcat, on="comid", how="left")
    return f


STREAMCAT_COLS = ["pcturbmd2019ws", "pcturbhi2019ws", "pctcrop2019ws", "pcthay2019ws", "manurews", "fertws",
                  "pctdecid2019ws", "pctwdwet2019ws", "wsareasqkm", "pctcrop2019cat", "pcturbmd2019cat"]
CAUSE_BASE_FEATURES = ["doy_sin", "doy_cos", "wb_river", "wb_lake", "wb_pond"]


# --------------------------------------------------------------------------- phase-1 modelling
def stratum_concordance(y, s, strata) -> pd.Series:
    """Per-stratum P(case > referent) (+0.5 ties); index = stratum."""
    df = pd.DataFrame({"y": np.asarray(y, float), "s": np.asarray(s, float), "g": np.asarray(strata)})
    df = df[np.isfinite(df["s"])]
    out = {}
    for g, sub in df.groupby("g"):
        cs, ct = sub.loc[sub["y"] == 1, "s"].to_numpy(), sub.loc[sub["y"] == 0, "s"].to_numpy()
        if len(cs) == 1 and len(ct):
            out[g] = np.mean(ct < cs[0]) + 0.5 * np.mean(ct == cs[0])
    return pd.Series(out, dtype=float)


def bootstrap_mean(values: pd.Series, clusters: pd.Series | None = None, *, n_boot: int = 2000, seed: int = 0,
                   other: pd.Series | None = None) -> dict:
    """Cluster bootstrap of mean(values) (or of mean(values) - mean(other), paired on the same strata).
    ``clusters`` maps stratum -> cluster (default: each stratum its own cluster)."""
    v = values if other is None else (values - other.reindex(values.index))
    v = v.dropna()
    cl = pd.Series(v.index, index=v.index) if clusters is None else clusters.reindex(v.index)
    groups = [v[cl == c].to_numpy() for c in pd.unique(cl)]
    rs = np.random.default_rng(seed)
    bs = np.empty(n_boot)
    for b in range(n_boot):
        pick = rs.integers(0, len(groups), len(groups))
        bs[b] = np.concatenate([groups[i] for i in pick]).mean()
    out = {"point": float(v.mean()), "ci_low": float(np.quantile(bs, 0.025)), "ci_high": float(np.quantile(bs, 0.975)),
           "n_strata": int(len(v)), "n_clusters": len(groups)}
    if other is not None:
        out["p_value"] = float(min(1.0, 2 * min(np.mean(bs <= 0), np.mean(bs >= 0))))
    return out


def design(cc: pd.DataFrame, cols) -> pd.DataFrame:
    """Feature matrix. Review W2: gauge series start 2000-10-01 and have gaps, so *value* missingness
    depends on the date and could separate cases from referents. Gauge anomalies are therefore imputed
    to 0 (= climatological) for every row, and the only missingness signal kept is the location-level
    ``no gauge within 30 km`` flag, which is constant within a matched set."""
    X = cc[list(cols)].astype(float).copy()
    if any(c in X for c in GAUGE_FEATURES):
        for c in GAUGE_FEATURES:
            if c in X:
                X[c] = X[c].fillna(0.0)
        X["g_no_gauge"] = cc["g_q_site"].isna().astype(float).to_numpy()
    return X


MODEL_SPECS = {
    # name: (kind, feature list, hyperparameter grid) — every learned model gets the same inner-CV budget
    "year_only_clogit": ("clogit", ["year_c"], [{"l2": v} for v in (0.01, 0.1, 1.0, 3.0, 10.0, 30.0)]),  # placebo
    "temp_anom_clogit": ("clogit", TEMP_ANOM_FEATURES + ["year_c"], [{"l2": v} for v in (0.01, 0.1, 1.0, 3.0, 10.0, 30.0)]),
    "wx_clogit": ("clogit", WX_FEATURES + GAUGE_FEATURES + ["year_c"], [{"l2": v} for v in (0.01, 0.1, 1.0, 3.0, 10.0, 30.0)]),
    "wx_gbm": ("gbm", WX_FEATURES + GAUGE_FEATURES + ["doy_sin", "doy_cos", "year_c"],
               [{"num_leaves": nl, "min_child_samples": mc, "learning_rate": 0.03, "n_estimators": 300}
                for nl in (7, 15, 31) for mc in (50, 200)]),
}


def iteration_specs(iteration: str) -> dict:
    """Model specs for an improvement iteration (one change at a time; baselines unchanged)."""
    if iteration == "base":
        return MODEL_SPECS
    if iteration == "it1_longwin":
        specs = dict(MODEL_SPECS)
        for name in ("wx_clogit", "wx_gbm"):
            kind, cols, grid = MODEL_SPECS[name]
            specs[name] = (kind, cols + LONG_FEATURES, grid)
        return specs
    raise ValueError(iteration)


def fit_model(kind: str, params: dict, X, y, strata, seed: int = 0):
    if kind == "clogit":
        return ConditionalLogit(**params).fit(np.asarray(X, float), y, strata)
    import lightgbm as lgb

    m = lgb.LGBMClassifier(objective="binary", subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                           reg_lambda=1.0, n_jobs=C_N_THREADS, random_state=seed, verbose=-1, **params)
    return m.fit(X, y)


def score_model(model, X) -> np.ndarray:
    if isinstance(model, ConditionalLogit):
        return model.decision_function(X)
    return model.predict_proba(X)[:, 1]


def predict_seeds(kind: str, params: dict, Xtr, ytr, str_, Xte, seeds=SEEDS) -> tuple[np.ndarray, list]:
    """Seed-averaged scores (GBM: one fit per seed; clogit is deterministic -> one fit)."""
    seeds = seeds if kind == "gbm" else (0,)
    per = [score_model(fit_model(kind, params, Xtr, ytr, str_, sd), Xte) for sd in seeds]
    return np.mean(per, axis=0), per


def inner_cv_select(kind: str, grid: list, X: pd.DataFrame, y, strata, *, event_year=None, n_folds: int = 5,
                    seed: int = 0) -> tuple[dict, list]:
    """Pick hyperparameters by matched concordance in inner CV on TRAIN rows only. Folds are grouped by
    **event year** when ``event_year`` is given (review W4), else by stratum."""
    strata = np.asarray(strata)
    if event_year is not None:
        ey = np.asarray(event_year)
        years = np.unique(ey)
        fold_year = dict(zip(years, np.arange(len(years)) % n_folds))
        f = np.array([fold_year[v] for v in ey])
    else:
        uniq = np.unique(strata)
        fold_of = dict(zip(uniq, np.random.default_rng(seed).permutation(len(uniq)) % n_folds))
        f = np.array([fold_of[s] for s in strata])
    log = []
    for p in grid:
        vals = []
        for k in range(n_folds):
            tr, te = f != k, f == k
            sc, _ = predict_seeds(kind, p, X[tr], np.asarray(y)[tr], strata[tr], X[te])
            vals.append(stratum_concordance(np.asarray(y)[te], sc, strata[te]))
        log.append({"params": p, "inner_cv_concordance": float(pd.concat(vals).mean())})
    best = max(log, key=lambda r: r["inner_cv_concordance"])
    return best["params"], log
