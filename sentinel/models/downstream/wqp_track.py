# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Shared machinery for the national WQP site-day exceedance tracks (RESEARCH_PLAN §15.5 items 5–7).

Two enlarged tracks are built on the already-harmonised Water Quality Portal table
(`processed/wqp_discrete`, manifest `wqp_discrete`, datasheet `docs/datasheets/wqp_discrete.md`):

* ``d1n_beaches``  — D1 national: freshwater beach *E. coli* > 235 CFU/MPN per 100 mL, morning forecast
  (protocol `docs/preregistration/downstream_d1n_beaches.md`);
* ``d2n_microcystin`` — D2 national: lake microcystin > 8 µg/L (protocol
  `docs/preregistration/downstream_d2n_microcystin.md`).

Everything here is label/cluster/feature definition (part of each track's scoring-code hash once the lead
adds this file to ``common.APP_SCORING_FILES``). Rules fixed before any model fit:

* site-day label from accepted, non-QC results; censoring resolved only where the bound decides;
* clusters = union-find of sites within ``cluster_km``; spatial group = HUC4 of the nearest WBD HUC12
  centroid; HUC8 (same way) is the bootstrap unit;
* group roles dealt from site-day counts only (no outcome used), largest group first, to the role
  furthest below its target share; groups holding > 10 % of all site-days are train;
* history / network features use cluster-days dated ≤ d − ``lag_days`` only (lab turnaround);
  weather windows end d − 1;
* rows of test-role groups and of test years are dropped before any feature is built (phase 1).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ...core import paths
from . import common as C
from . import weather as W

WQP = paths.PROCESSED / "wqp_discrete"
QC_ACTIVITY = ("Quality Control", "Blank", "Spike", "Duplicate", "Replicate")


@dataclass(frozen=True)
class TrackConfig:
    app: str
    variable: str
    threshold: float
    agg: str                      # "gmean" (FIB) or "max" (toxin)
    lag_days: int                 # prior results usable if dated <= d - lag_days
    net_window: int               # network window length (days) ending d - lag_days
    net_km: float
    cluster_km: float
    train_years: tuple
    val_years: tuple
    test_years: tuple
    excluded_years: tuple
    site_filter: str              # "beach" | "lake"
    exclude_states: tuple = ()
    exclude_boxes: tuple = ()     # (lat_lo, lat_hi, lon_lo, lon_hi)
    role_targets: dict = field(default_factory=lambda: {"test": 0.20, "val": 0.15})
    giant_frac: float = 0.10
    persist_gap: int = 3
    max_wx_km: float = 30.0
    seed: int = 20261108


# --------------------------------------------------------------------------- labels
def load_sitedays(cfg: TrackConfig) -> pd.DataFrame:
    cols = ["site_id", "state", "lat", "lon", "site_class", "activity_start_date", "value", "censored",
            "censor_type", "detection_limit", "activity_type", "status_class"]
    d = pd.read_parquet(WQP / f"variable={cfg.variable}" / "data.parquet", columns=cols)
    d = d[(d["status_class"] == "accepted")
          & ~d["activity_type"].fillna("").str.contains("|".join(QC_ACTIVITY), case=False)]
    st = pd.read_parquet(WQP / "_stations.parquet", columns=["site_id", "name"]).drop_duplicates("site_id")
    d = d.merge(st, on="site_id", how="left")
    if cfg.site_filter == "beach":
        d = d[(d["site_class"] == "lake") & d["name"].fillna("").str.upper().str.contains("BEACH")]
    elif cfg.site_filter == "lake":
        d = d[d["site_class"] == "lake"]
    else:
        raise ValueError(cfg.site_filter)
    if cfg.exclude_states:
        d = d[~d["state"].isin(cfg.exclude_states)]
    for la0, la1, lo0, lo1 in cfg.exclude_boxes:
        d = d[~(d["lat"].between(la0, la1) & d["lon"].between(lo0, lo1))]
    d = d[d["lat"].notna() & d["lon"].notna()]
    # censoring: '<DL' is non-exceeding iff DL <= threshold; '>DL' exceeding iff DL >= threshold
    v = d["value"].to_numpy(float)
    dl = d["detection_limit"].to_numpy(float)
    ct = d["censor_type"].fillna("").to_numpy()
    cens = d["censored"].fillna(False).to_numpy(bool)
    exc = np.where(~cens, v > cfg.threshold,
                   np.where(ct == ">", dl >= cfg.threshold, False)).astype(float)
    det = np.where(~cens, np.isfinite(v),
                   np.where(ct == "<", dl <= cfg.threshold, np.where(ct == ">", dl >= cfg.threshold, False)))
    # value used for log features: measured value, else the bound (half DL for '<')
    val = np.where(~cens, v, np.where(ct == "<", 0.5 * dl, dl))
    d = d.assign(exc=exc, determined=det, val=val, date=pd.to_datetime(d["activity_start_date"]))
    d = d[np.isfinite(d["val"].to_numpy(float)) | d["determined"]]
    floor = 1.0 if cfg.agg == "gmean" else 0.01
    d["logv"] = np.log10(np.maximum(d["val"].to_numpy(float), floor))
    g = d.groupby(["site_id", "date"], sort=False)
    if cfg.agg == "gmean":
        sd = g.agg(logv=("logv", "mean"), lat=("lat", "first"), lon=("lon", "first"), state=("state", "first"),
                   n_res=("logv", "size"), any_cens=("censored", "any"), all_det=("determined", "all"),
                   any_exc=("exc", "max")).reset_index()
        # the day's geometric mean decides; a day with censored results is kept only if the bound decides
        sd["y"] = (sd["logv"] > np.log10(cfg.threshold)).astype(float)
        sd = sd[~sd["any_cens"] | sd["all_det"]]
    else:
        sd = g.agg(logv=("logv", "max"), lat=("lat", "first"), lon=("lon", "first"), state=("state", "first"),
                   n_res=("logv", "size"), any_exc=("exc", "max"), all_det=("determined", "all")).reset_index()
        sd["y"] = sd["any_exc"].astype(float)
        sd = sd[(sd["y"] == 1) | sd["all_det"]]          # any result > T decides; else all must be determined
    sd["year"] = sd["date"].dt.year
    return sd.drop(columns=[c for c in ("any_cens", "all_det", "any_exc") if c in sd]).reset_index(drop=True)


# --------------------------------------------------------------------------- clusters and groups
def _balltree(lat, lon):
    from sklearn.neighbors import BallTree

    return BallTree(np.radians(np.column_stack([lat, lon])), metric="haversine")


def site_clusters(sites: pd.DataFrame, km: float) -> pd.Series:
    """Union-find of sites within ``km`` (great circle); cluster id = smallest site_id of the component."""
    sites = sites.sort_values("site_id").reset_index(drop=True)
    tree = _balltree(sites["lat"].to_numpy(float), sites["lon"].to_numpy(float))
    nb = tree.query_radius(np.radians(sites[["lat", "lon"]].to_numpy(float)), r=km / 6371.0088)
    parent = np.arange(len(sites))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, js in enumerate(nb):
        for j in js:
            a, b = find(i), find(int(j))
            if a != b:
                parent[max(a, b)] = min(a, b)
    root = np.array([find(i) for i in range(len(sites))])
    ids = sites["site_id"].to_numpy()
    return pd.Series(ids[root], index=ids, name="cluster")


def huc_of_points(lat, lon) -> pd.DataFrame:
    u = pd.read_parquet(paths.PROCESSED / "wbd_huc12" / "huc12.parquet", columns=["huc12", "lat", "lon", "huc4", "huc8"])
    u = u[u["lat"].notna() & u["lon"].notna()].reset_index(drop=True)
    tree = _balltree(u["lat"].to_numpy(float), u["lon"].to_numpy(float))
    dist, idx = tree.query(np.radians(np.column_stack([lat, lon])), k=1)
    i = idx[:, 0]
    return pd.DataFrame({"huc4": u["huc4"].to_numpy()[i].astype(str), "huc8": u["huc8"].to_numpy()[i].astype(str),
                         "huc12_km": dist[:, 0] * 6371.0088})


def cluster_table(sd: pd.DataFrame, cfg: TrackConfig) -> pd.DataFrame:
    sites = sd.groupby("site_id").agg(lat=("lat", "first"), lon=("lon", "first"), state=("state", "first"),
                                      n=("y", "size")).reset_index()
    cl = site_clusters(sites, cfg.cluster_km)
    sites["cluster"] = sites["site_id"].map(cl)
    ct = sites.groupby("cluster").apply(
        lambda g: pd.Series({"lat": np.average(g["lat"], weights=g["n"]), "lon": np.average(g["lon"], weights=g["n"]),
                             "state": g["state"].mode().iloc[0], "n_sites": len(g)}), include_groups=False).reset_index()
    h = huc_of_points(ct["lat"].to_numpy(float), ct["lon"].to_numpy(float))
    ct = pd.concat([ct, h], axis=1)
    return sites[["site_id", "cluster"]], ct


def _sha_key(seed, s):
    return hashlib.sha1(f"{seed}:{s}".encode()).hexdigest()


def group_roles(counts: pd.Series, cfg: TrackConfig) -> dict:
    """HUC4 group -> role from site-day COUNTS only (no labels). Largest first to the role furthest below
    its target share; ties broken by sha1(seed, group)."""
    total = float(counts.sum())
    order = sorted(counts.index, key=lambda g: (-counts[g], _sha_key(cfg.seed, g)))
    roles, got = {}, {"train": 0.0, "val": 0.0, "test": 0.0}
    target = {"test": cfg.role_targets["test"], "val": cfg.role_targets["val"]}
    target["train"] = 1.0 - target["test"] - target["val"]
    for g in order:
        if counts[g] / total > cfg.giant_frac:
            r = "train"
        else:
            r = max(target, key=lambda k: (target[k] - got[k] / total, k == "train"))
        roles[str(g)] = r
        got[r] += counts[g]
    return roles


def year_role(year: int, cfg: TrackConfig) -> str | None:
    if year in cfg.excluded_years:
        return None
    if cfg.train_years[0] <= year <= cfg.train_years[1]:
        return "train"
    if cfg.val_years[0] <= year <= cfg.val_years[1]:
        return "val"
    if cfg.test_years[0] <= year <= cfg.test_years[1]:
        return "test"
    return None


def _file_sha(p):
    from ...core.registry import file_sha256

    return file_sha256(p)


def freeze(cfg: TrackConfig) -> tuple[str, dict]:
    sd = load_sitedays(cfg)
    sites, ct = cluster_table(sd, cfg)
    sd = sd.merge(sites, on="site_id")
    sd = sd.merge(ct[["cluster", "huc4"]], on="cluster")
    sd = sd[sd["year"].map(lambda y: year_role(y, cfg) is not None)]
    counts = sd.groupby("huc4").size()
    roles = group_roles(counts, cfg)
    meta = {"track": cfg.app, "wqp_variable": cfg.variable, "threshold": cfg.threshold, "agg": cfg.agg,
            "lag_days": cfg.lag_days, "years": {"train": cfg.train_years, "val": cfg.val_years,
                                                "test": cfg.test_years, "excluded": cfg.excluded_years},
            "site_filter": cfg.site_filter, "exclude_states": cfg.exclude_states, "exclude_boxes": cfg.exclude_boxes,
            "cluster_km": cfg.cluster_km, "role_targets": cfg.role_targets, "giant_frac": cfg.giant_frac,
            "wqp_data_sha256": _file_sha(WQP / f"variable={cfg.variable}" / "data.parquet"),
            "wqp_stations_sha256": _file_sha(WQP / "_stations.parquet"),
            "row_rule": "train = train-role HUC4 x train years; val_known = train-role HUC4 x val years; "
                        "val_new = val-role HUC4 x val years; test_known = train/val-role HUC4 x test years; "
                        "test_new = test-role HUC4 x test years; val-role HUC4 rows in train years and all "
                        "excluded years are dropped",
            "freeze_date": "2026-11-08"}
    sha = C.freeze_split(cfg.app, {"huc4_role": roles,
                                   "site_cluster": dict(zip(sites["site_id"].astype(str), sites["cluster"].astype(str)))},
                         meta)
    return sha, roles


def assign(sd: pd.DataFrame, split: dict, cfg: TrackConfig, ct: pd.DataFrame) -> pd.DataFrame:
    a = split["assignments"]
    sd = sd.copy()
    sd["cluster"] = sd["site_id"].astype(str).map(a["site_cluster"])
    sd = sd[sd["cluster"].notna()]
    sd = sd.merge(ct[["cluster", "huc4", "huc8"]], on="cluster", how="left")
    grole = sd["huc4"].map(a["huc4_role"])
    yrole = sd["year"].map(lambda y: year_role(y, cfg))
    role = pd.Series(None, index=sd.index, dtype=object)
    evalset = pd.Series(None, index=sd.index, dtype=object)
    tr = (grole == "train")
    role[tr & (yrole == "train")] = "train"
    role[tr & (yrole == "val")] = "val"
    evalset[tr & (yrole == "val")] = "val_known"
    role[(grole == "val") & (yrole == "val")] = "val"
    evalset[(grole == "val") & (yrole == "val")] = "val_new"
    te_known = grole.isin(["train", "val"]) & (yrole == "test")
    te_new = (grole == "test") & (yrole == "test")
    role[te_known | te_new] = "test"
    evalset[te_known] = "test_known"
    evalset[te_new] = "test_new"
    # test-role HUC4 rows in train/val years stay unassigned (dropped): test_new clusters start cold,
    # exactly as val_new clusters do
    sd["role"], sd["evalset"] = role, evalset
    return sd[sd["role"].notna()].reset_index(drop=True)


def cluster_days(sd: pd.DataFrame) -> pd.DataFrame:
    """One row per cluster-day (several stations of one beach/lake): mean log value, any exceedance."""
    g = sd.groupby(["cluster", "date"], sort=True)
    return g.agg(logv=("logv", "mean"), y=("y", "max")).reset_index()


# --------------------------------------------------------------------------- features
def _window_sums(dates_hist, cs, q_lo, q_hi):
    """Sum over history entries with q_lo <= date <= q_hi using cumulative sums ``cs`` (len n+1)."""
    lo = np.searchsorted(dates_hist, q_lo, side="left")
    hi = np.searchsorted(dates_hist, q_hi, side="right")
    return cs[hi] - cs[lo], hi - lo


def history_features(rows: pd.DataFrame, cd: pd.DataFrame, cfg: TrackConfig, p0: float, a_eb: float = 5.0) -> pd.DataFrame:
    """Own-cluster history from cluster-days dated <= d - lag (``cd`` = cluster_days of non-test rows)."""
    L = cfg.lag_days
    out = {k: np.full(len(rows), np.nan) for k in
           ("tw_n_prior", "tw_n_exc_prior", "tw_prior_rate_s", "tw_days_since_last", "tw_last_log_value",
            "tw_last_exceed", "tw_mean_log_365", "tw_n_365", "tw_exc_365", "tw_prior_max_log_365",
            "tw_mean_log_30", "tw_n_30", "tw_same_year_exc")}
    by = {c: g for c, g in cd.groupby("cluster", sort=False)}
    for c, idx in rows.groupby("cluster").indices.items():
        h = by.get(c)
        q = rows["date"].to_numpy("datetime64[D]")[idx]
        cut = q - np.timedelta64(L, "D")
        if h is None or h.empty:
            out["tw_n_prior"][idx] = 0
            out["tw_n_exc_prior"][idx] = 0
            out["tw_prior_rate_s"][idx] = p0
            continue
        hd = h["date"].to_numpy("datetime64[D]")
        lv, yy = h["logv"].to_numpy(float), h["y"].to_numpy(float)
        cs_n = np.arange(len(hd) + 1, dtype=float)
        cs_y = np.concatenate([[0], np.cumsum(yy)])
        cs_l = np.concatenate([[0], np.cumsum(lv)])
        k = np.searchsorted(hd, cut, side="right")          # entries with date <= cut
        n = k.astype(float)
        ne = cs_y[k]
        out["tw_n_prior"][idx] = n
        out["tw_n_exc_prior"][idx] = ne
        out["tw_prior_rate_s"][idx] = (ne + a_eb * p0) / (n + a_eb)
        has = k > 0
        last = np.where(has, k - 1, 0)
        out["tw_days_since_last"][idx] = np.where(has, (q - hd[last]).astype(float), np.nan)
        out["tw_last_log_value"][idx] = np.where(has, lv[last], np.nan)
        out["tw_last_exceed"][idx] = np.where(has, yy[last], np.nan)
        for w, nm in ((365, "365"), (30, "30")):
            lo = cut - np.timedelta64(w - 1, "D")
            s_l, cnt = _window_sums(hd, cs_l, lo, cut)
            out[f"tw_mean_log_{nm}"][idx] = np.where(cnt > 0, s_l / np.maximum(cnt, 1), np.nan)
            out[f"tw_n_{nm}"][idx] = cnt
            if nm == "365":
                s_y, _ = _window_sums(hd, cs_y, lo, cut)
                out["tw_exc_365"][idx] = s_y
        # 365-d max log value (loop-free: sparse table would be overkill; histories are short)
        mx = np.full(len(idx), np.nan)
        lo = np.searchsorted(hd, cut - np.timedelta64(364, "D"), side="left")
        for j in range(len(idx)):
            if k[j] > lo[j]:
                mx[j] = lv[lo[j]:k[j]].max()
        out["tw_prior_max_log_365"][idx] = mx
        ys = np.array([np.datetime64(f"{str(x)[:4]}-01-01") for x in q], dtype="datetime64[D]")
        s_y, _ = _window_sums(hd, cs_y, ys, cut)
        out["tw_same_year_exc"][idx] = s_y
    return pd.DataFrame(out, index=rows.index)


def network_features(rows: pd.DataFrame, cd: pd.DataFrame, ct: pd.DataFrame, cfg: TrackConfig) -> pd.DataFrame:
    """Other clusters within ``net_km``: results dated in [d - lag - window + 1, d - lag] (count, mean log,
    max log, any exceedance) and distance to the nearest other monitored cluster."""
    L, Wd = cfg.lag_days, cfg.net_window
    ctc = ct.set_index("cluster")
    cl = ctc.index.to_numpy()
    tree = _balltree(ctc["lat"].to_numpy(float), ctc["lon"].to_numpy(float))
    nb, nd = tree.query_radius(np.radians(ctc[["lat", "lon"]].to_numpy(float)), r=cfg.net_km / 6371.0088,
                               return_distance=True)
    pos = {c: i for i, c in enumerate(cl)}
    by = {c: g for c, g in cd.groupby("cluster", sort=False)}
    out = {k: np.full(len(rows), np.nan) for k in ("net_n", "net_mean_log", "net_max_log", "net_any_exc", "net_nearest_km",
                                                   "net_n_clusters")}
    for c, idx in rows.groupby("cluster").indices.items():
        i = pos.get(c)
        if i is None:
            continue
        others = [(cl[j], d * 6371.0088) for j, d in zip(nb[i], nd[i]) if cl[j] != c]
        out["net_n_clusters"][idx] = len(others)
        out["net_nearest_km"][idx] = min([d for _, d in others], default=np.nan)
        hs = [by[o] for o, _ in others if o in by]
        q = rows["date"].to_numpy("datetime64[D]")[idx]
        if not hs:
            out["net_n"][idx] = 0
            continue
        h = pd.concat(hs).sort_values("date")
        hd = h["date"].to_numpy("datetime64[D]")
        lv, yy = h["logv"].to_numpy(float), h["y"].to_numpy(float)
        hi_d = q - np.timedelta64(L, "D")
        lo_d = hi_d - np.timedelta64(Wd - 1, "D")
        s_l, cnt = _window_sums(hd, np.concatenate([[0], np.cumsum(lv)]), lo_d, hi_d)
        s_y, _ = _window_sums(hd, np.concatenate([[0], np.cumsum(yy)]), lo_d, hi_d)
        out["net_n"][idx] = cnt
        out["net_mean_log"][idx] = np.where(cnt > 0, s_l / np.maximum(cnt, 1), np.nan)
        out["net_any_exc"][idx] = np.where(cnt > 0, (s_y > 0).astype(float), np.nan)
        lo = np.searchsorted(hd, lo_d, side="left")
        hi = np.searchsorted(hd, hi_d, side="right")
        out["net_max_log"][idx] = [lv[a:b].max() if b > a else np.nan for a, b in zip(lo, hi)]
    return pd.DataFrame(out, index=rows.index)


WX_WINDOWS = (1, 2, 3, 7, 14, 30)


def weather_features(rows: pd.DataFrame, ct: pd.DataFrame, cfg: TrackConfig) -> pd.DataFrame:
    cells = W.nearest_cells(ct["lat"].to_numpy(float), ct["lon"].to_numpy(float))
    cells.index = ct["cluster"].to_numpy()
    cells.loc[cells["wx_cell_km"] > cfg.max_wx_km, "wx_site_key"] = None
    q = pd.DataFrame({"wx_site_key": rows["cluster"].map(cells["wx_site_key"]).to_numpy(), "date": rows["date"].to_numpy()},
                     index=rows.index)
    keys = q["wx_site_key"].dropna().unique()
    start = f"{cfg.train_years[0] - 1}-10-01"
    ser = W.load_series(keys, start=start, variables=("precip_mm", "tmax_c", "tmin_c", "srad_wm2", "wind_ms", "vpd_kpa"))
    clim = W.doy_climatology(ser, range(cfg.train_years[0], cfg.train_years[1] + 1))
    f = W.antecedent_features(q.dropna(subset=["wx_site_key"]), ser, clim=clim, windows=WX_WINDOWS)
    f = f.reindex(rows.index)
    f["wx_cell_km"] = rows["cluster"].map(cells["wx_cell_km"]).to_numpy()
    return f


def build_frame(cfg: TrackConfig, split: dict, *, include_test: bool = False,
                split_sha: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Phase-1 frame: train + val rows only (test rows dropped before features). ``include_test=True``
    only inside the single authorised test look (``testlook.require_authorised``)."""
    if include_test:
        from .testlook import require_authorised

        require_authorised(cfg.app, split_sha)          # before any label is read
    sd = load_sitedays(cfg)
    sites, ct = cluster_table(sd, cfg)
    sd = assign(sd, split, cfg, ct)
    if include_test:
        sd = sd[sd["role"].notna()].reset_index(drop=True)
    else:
        C.check_roles(cfg.app, set(sd["role"]) - {"test"})
        sd = C.drop_protected(sd)
        C.assert_no_test(sd)
    cd = cluster_days(sd)
    rows = sd.groupby(["cluster", "date"], sort=True).agg(
        y=("y", "max"), logv=("logv", "mean"), role=("role", "first"), evalset=("evalset", "first"),
        huc4=("huc4", "first"), huc8=("huc8", "first"), state=("state", "first"), n_sites_day=("site_id", "nunique")).reset_index()
    rows["year"] = rows["date"].dt.year
    p0 = float(rows.loc[rows["role"] == "train", "y"].mean())
    X = pd.concat([history_features(rows, cd, cfg, p0), network_features(rows, cd, ct, cfg),
                   weather_features(rows, ct, cfg)], axis=1)
    geo = ct.set_index("cluster")[["lat", "lon"]]
    X["lat"] = rows["cluster"].map(geo["lat"]).to_numpy()
    X["lon"] = rows["cluster"].map(geo["lon"]).to_numpy()
    doy = rows["date"].dt.dayofyear.to_numpy(float)
    X["doy_sin"], X["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    X["doy"] = doy
    X["dow"] = rows["date"].dt.dayofweek.to_numpy(float)
    X["year_c"] = (rows["year"].to_numpy(float) - 2010) / 10
    return rows, X.reset_index(drop=True), ct


# --------------------------------------------------------------------------- power check (before any fit)
def power_check(rows: pd.DataFrame, auc: float = 0.75, rho: float = 0.6) -> dict:
    """Hanley–McNeil SE of AUROC per evaluation set; paired-difference MDE (80 % power, two-sided 0.05)
    assuming correlation ``rho`` between the two models' AUROCs; NOT inflated for clustering (stated)."""
    out = {}
    for es, g in rows[rows["evalset"].notna()].groupby("evalset"):
        npos, nneg = int(g["y"].sum()), int((1 - g["y"]).sum())
        q1, q2 = auc / (2 - auc), 2 * auc ** 2 / (1 + auc)
        se = np.sqrt((auc * (1 - auc) + (npos - 1) * (q1 - auc ** 2) + (nneg - 1) * (q2 - auc ** 2)) / max(npos * nneg, 1))
        mde = 2.8 * se * np.sqrt(2 * (1 - rho))
        out[es] = {"n": int(len(g)), "n_pos": npos, "prevalence": float(g["y"].mean()),
                   "clusters": int(g["cluster"].nunique()), "huc8": int(g["huc8"].nunique()),
                   "huc4": int(g["huc4"].nunique()), "auroc_se_at_0.75": float(se), "paired_mde_auroc": float(mde)}
    tr = rows[rows["role"] == "train"]
    out["train"] = {"n": int(len(tr)), "n_pos": int(tr["y"].sum()), "clusters": int(tr["cluster"].nunique()),
                    "huc4": int(tr["huc4"].nunique())}
    return out


# =========================================================================== models (phase 1: train -> val)
NON_FEATURES = ("year_c", "doy")
INNER_YEARS = 3          # inner selection: fit on train years except the last 3, score the last 3
LOGIT_GRID = [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)]
GBM_GRID = [{"num_leaves": nl, "min_child_samples": mc} for nl in (7, 15) for mc in (20, 100)]
RIDGE_GRID = [{"alpha": a} for a in (0.1, 1.0, 10.0, 100.0)]
GLMM_GRID = [{"C_reg": c} for c in (0.01, 0.1, 1.0, 10.0)]
SEEDS = (0, 1, 2)
VB_PREDICTORS = ("wx_precip_sum_1d", "wx_precip_sum_2d", "wx_precip_sum_3d", "wx_precip_sum_7d",
                 "wx_tmax_mean_3d", "wx_wind_mean_1d", "tw_last_log_value", "tw_days_since_last",
                 "tw_mean_log_365", "doy_sin", "doy_cos")
VB_MIN_DAYS = 50


def feature_cols(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if c not in NON_FEATURES]


def _auroc(y, s):
    return C.fast_auroc(np.asarray(y, float), np.asarray(s, float))


class _Imp:
    """Train-median imputation + missingness indicators + standardisation (fitted on train)."""

    def fit(self, X):
        from sklearn.preprocessing import StandardScaler

        A = X.to_numpy(float)
        self.med_ = np.nanmedian(A, axis=0)
        self.med_ = np.where(np.isfinite(self.med_), self.med_, 0.0)
        self.miss_ = np.isnan(A).any(axis=0)
        self.sc_ = StandardScaler().fit(self._raw(A))
        return self

    def _raw(self, A):
        M = np.isnan(A)[:, self.miss_].astype(float)
        return np.hstack([np.where(np.isnan(A), self.med_, A), M])

    def transform(self, X):
        return self.sc_.transform(self._raw(X.to_numpy(float)))


def fit_logistic(X, y, C_):
    from sklearn.linear_model import LogisticRegression

    imp = _Imp().fit(X)
    m = LogisticRegression(C=C_, max_iter=3000).fit(imp.transform(X), y)
    return lambda Z: m.predict_proba(imp.transform(Z))[:, 1]


def fit_gbm(X, y, params, seed, n_estimators=None, Xv=None, yv=None):
    import lightgbm as lgb

    p = {"objective": "binary", "learning_rate": 0.03, "colsample_bytree": 0.7, "subsample": 0.8,
         "subsample_freq": 1, "reg_lambda": 1.0, "n_estimators": n_estimators or 2000, "verbose": -1,
         "n_jobs": C.N_THREADS, "random_state": seed, **params}
    m = lgb.LGBMClassifier(**p)
    if Xv is not None:
        m.fit(X, y, eval_set=[(Xv, yv)], eval_metric="auc", callbacks=[lgb.early_stopping(100, verbose=False)])
    else:
        m.fit(X, y)
    return m


class VBMLR:
    """USGS nowcast / Virtual-Beach-style multiple linear regression of log10 FIB (re-implemented
    incumbent): one ridge MLR per beach cluster with >= VB_MIN_DAYS training beach-days, a pooled MLR (same
    predictors) for the others; predictors = rainfall windows, temperature, wind, previous result and its
    age, seasonal harmonics (turbidity/wave height are not nationally available — disclosed)."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def _A(self, X):
        A = X[list(VB_PREDICTORS)].to_numpy(float).copy()
        j = VB_PREDICTORS.index("tw_days_since_last")
        A[:, j] = np.log1p(A[:, j])
        return A

    def fit(self, rows, X, logv):
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        A = self._A(X)
        self.med_ = np.nanmedian(A, axis=0)
        A = np.where(np.isnan(A), self.med_, A)
        self.sc_ = StandardScaler().fit(A)
        Z = self.sc_.transform(A)
        self.pool_ = Ridge(alpha=self.alpha).fit(Z, logv)
        self.site_ = {}
        for c, idx in rows.groupby("cluster").indices.items():
            if len(idx) >= VB_MIN_DAYS:
                self.site_[c] = Ridge(alpha=self.alpha).fit(Z[idx], np.asarray(logv)[idx])
        return self

    def predict(self, rows, X):
        A = np.where(np.isnan(self._A(X)), self.med_, self._A(X))
        Z = self.sc_.transform(A)
        out = self.pool_.predict(Z)
        for c, idx in rows.groupby("cluster").indices.items():
            m = self.site_.get(c)
            if m is not None:
                out[idx] = m.predict(Z[idx])
        return out


class SparseStGlmm:
    """INLA-class incumbent of D2 (``toxinwatch.StGlmm`` v1/v2 design, identical terms) with a SPARSE
    lake one-hot so that thousands of national lake clusters fit in memory."""

    def __init__(self, C_reg=1.0, wx_cols=(), variant="v1"):
        self.C_reg, self.wx_cols, self.variant = C_reg, list(wx_cols), variant

    def _design(self, rows, X, loc_rate, fit=False):
        import scipy.sparse as sp
        from sklearn.preprocessing import SplineTransformer, StandardScaler

        from .toxinwatch import st_glmm_design

        core = st_glmm_design(rows, X, loc_rate)
        if self.variant == "v2":
            age = np.nan_to_num(X["tw_days_since_last"].to_numpy(float), nan=3650.0)
            lv = np.nan_to_num(X["tw_last_log_value"].to_numpy(float), nan=-1.0)
            le = np.nan_to_num(X["tw_last_exceed"].to_numpy(float), nan=0.0)
            pm = np.nan_to_num(X["tw_prior_max_log_365"].to_numpy(float), nan=-1.0)
            ar = [v * np.exp(-age / tau) for tau in (14.0, 60.0) for v in (lv, le)]
            ll = X[["lat", "lon"]].to_numpy(float)
            if fit:
                self.spl_ = SplineTransformer(n_knots=6, degree=3, extrapolation="constant").fit(ll)
            core = np.hstack([core, np.column_stack(ar + [pm]), self.spl_.transform(ll)])
        wx = X[self.wx_cols].to_numpy(float) if self.wx_cols else np.empty((len(X), 0))
        if fit:
            self.med_ = np.nanmedian(wx, axis=0) if wx.shape[1] else np.empty(0)
            self.med_ = np.where(np.isfinite(self.med_), self.med_, 0.0)
        wx = np.where(np.isfinite(wx), wx, self.med_)
        cont = np.hstack([core, wx])
        if fit:
            self.scaler_ = StandardScaler().fit(cont)
            self.lakes_ = {c: i for i, c in enumerate(sorted(rows["cluster"].unique()))}
        cont = self.scaler_.transform(cont)
        idx = rows["cluster"].map(self.lakes_)
        ok = idx.notna().to_numpy()
        oh = sp.csr_matrix((np.ones(ok.sum()), (np.flatnonzero(ok), idx[ok].astype(int).to_numpy())),
                           shape=(len(rows), len(self.lakes_)))
        return sp.hstack([sp.csr_matrix(cont), oh]).tocsr()

    def fit(self, rows, X, loc_rate, y):
        from sklearn.linear_model import LogisticRegression

        self.m_ = LogisticRegression(C=self.C_reg, max_iter=5000, solver="lbfgs").fit(
            self._design(rows, X, loc_rate, fit=True), np.asarray(y, float))
        return self

    def predict_proba(self, rows, X, loc_rate):
        return self.m_.predict_proba(self._design(rows, X, loc_rate))[:, 1]


# --------------------------------------------------------------------------- simple baselines
def knn_rate(train_rows: pd.DataFrame, ct: pd.DataFrame, query_clusters, k: int = 5, exclude_self: bool = True):
    """Inverse-distance kNN of training-cluster exceedance rates (EB-shrunk, k_eb = 20)."""
    p0 = train_rows["y"].mean()
    g = train_rows.groupby("cluster")["y"].agg(["sum", "size"])
    rate = (g["sum"] + 20 * p0) / (g["size"] + 20)
    geo = ct.set_index("cluster").loc[rate.index]
    tree = _balltree(geo["lat"].to_numpy(float), geo["lon"].to_numpy(float))
    qc = pd.Index(query_clusters)
    q = ct.set_index("cluster").loc[qc]
    d, i = tree.query(np.radians(q[["lat", "lon"]].to_numpy(float)), k=min(k + 1, len(rate)))
    names = rate.index.to_numpy()
    out = np.empty(len(qc))
    for r in range(len(qc)):
        sel = [(dd, ii) for dd, ii in zip(d[r], i[r]) if not (exclude_self and names[ii] == qc[r])][:k]
        w = np.array([1.0 / (dd * 6371.0088 + 1.0) for dd, _ in sel])
        out[r] = float(np.sum(w * rate.to_numpy()[[ii for _, ii in sel]]) / w.sum())
    return pd.Series(out, index=qc)


def baseline_scores(tr: pd.DataFrame, Xtr: pd.DataFrame, ev: pd.DataFrame, Xev: pd.DataFrame, ct, cfg) -> dict:
    p0 = tr["y"].mean()
    m_rate = tr.groupby(tr["date"].dt.month)["y"].agg(["sum", "size"])
    m_rate = (m_rate["sum"] + 20 * p0) / (m_rate["size"] + 20)
    cm = tr.groupby(["cluster", tr["date"].dt.month])["y"].agg(["sum", "size"])
    mon = ev["date"].dt.month
    prior = mon.map(m_rate).fillna(p0).to_numpy()
    key = list(zip(ev["cluster"], mon))
    s = cm["sum"].reindex(key).to_numpy()
    n = cm["size"].reindex(key).to_numpy()
    clim = np.where(np.isfinite(n), (np.nan_to_num(s) + 20 * prior) / (np.nan_to_num(n) + 20), prior)
    knn = knn_rate(tr, ct, ev["cluster"].unique()).reindex(ev["cluster"]).to_numpy()
    g = tr.groupby("cluster")["y"].agg(["sum", "size"])
    gs, gn = g["sum"].reindex(ev["cluster"]).to_numpy(), g["size"].reindex(ev["cluster"]).to_numpy()
    loc = np.where(np.isfinite(gn), (np.nan_to_num(gs) + 20 * knn) / (np.nan_to_num(gn) + 20), knn)
    last = Xev["tw_last_log_value"].to_numpy(float)
    age = Xev["tw_days_since_last"].to_numpy(float)
    pers = np.where(np.isfinite(last) & (age <= cfg.persist_gap), 10 * Xev["tw_last_exceed"].to_numpy(float) + last, np.nan)
    return {"climatology": clim, "location_only": loc, "history_rate": Xev["tw_prior_rate_s"].to_numpy(float),
            "persistence": pers}


# --------------------------------------------------------------------------- selection + fitting
def _inner(tr: pd.DataFrame, cfg):
    cut = cfg.train_years[1] - INNER_YEARS
    return (tr["year"] <= cut).to_numpy(), (tr["year"] > cut).to_numpy()


def fit_all(tr, Xtr, evs: dict, ct, cfg, incumbent: str, log_config) -> tuple[dict, dict]:
    """Returns ({model: {evalset: scores}}, selection log). Same 4-config budget for every learned model."""
    cols = feature_cols(Xtr)
    wx_cols = [c for c in cols if c.startswith("wx_") and c != "wx_cell_km" and any(
        c.endswith(f"_{w}d") for w in (3, 7, 30))]
    a, b = _inner(tr, cfg)
    ytr = tr["y"].to_numpy(float)
    sel, scores = {}, {}

    def choose(name, grid, inner_score):
        log = []
        for p in grid:
            try:
                v = inner_score(p)
            except Exception as e:  # noqa: BLE001 — a failed configuration is logged, never silently dropped
                v = float("nan")
                log_config(name, p, f"failed: {e!r}"[:200])
            log.append({"params": p, "inner_auroc": v})
            log_config(name, p, f"inner_auroc={v:.4f}")
        best = max(log, key=lambda r: (np.nan_to_num(r["inner_auroc"], nan=-1)))
        sel[name] = {"selected": best["params"], "log": log}
        return best["params"]

    # logistic
    p = choose("logistic", LOGIT_GRID, lambda p: _auroc(ytr[b], fit_logistic(Xtr.loc[a, cols], ytr[a], p["C"])(Xtr.loc[b, cols])))
    f = fit_logistic(Xtr[cols], ytr, p["C"])
    scores["logistic"] = {k: f(Xe[cols]) for k, (ev, Xe) in evs.items()}
    # GBM challenger (3 seeds; n_estimators from inner early stopping)
    best_iter = {}

    def gbm_inner(p):
        m = fit_gbm(Xtr.loc[a, cols], ytr[a], p, 0, Xv=Xtr.loc[b, cols], yv=ytr[b])
        best_iter[str(p)] = max(int(m.best_iteration_ or 200), 50)
        return _auroc(ytr[b], m.predict_proba(Xtr.loc[b, cols])[:, 1])

    p = choose("gbm_challenger", GBM_GRID, gbm_inner)
    ms = [fit_gbm(Xtr[cols], ytr, p, sd, n_estimators=best_iter[str(p)]) for sd in SEEDS]
    scores["gbm_challenger"] = {k: np.mean([m.predict_proba(Xe[cols])[:, 1] for m in ms], axis=0) for k, (ev, Xe) in evs.items()}
    scores["gbm_challenger_per_seed"] = {k: [m.predict_proba(Xe[cols])[:, 1] for m in ms] for k, (ev, Xe) in evs.items()}
    sel["gbm_challenger"]["n_estimators"] = best_iter[str(p)]
    # incumbent(s)
    if incumbent == "vb_mlr":
        lv = tr["logv"].to_numpy(float)
        tra, trb = tr[a].reset_index(drop=True), tr[b].reset_index(drop=True)
        p = choose("vb_mlr_incumbent", RIDGE_GRID, lambda p: _auroc(
            ytr[b], VBMLR(p["alpha"]).fit(tra, Xtr[a].reset_index(drop=True), lv[a]).predict(trb, Xtr[b].reset_index(drop=True))))
        m = VBMLR(p["alpha"]).fit(tr, Xtr, lv)
        scores["vb_mlr_incumbent"] = {k: m.predict(ev.reset_index(drop=True), Xe.reset_index(drop=True)) for k, (ev, Xe) in evs.items()}
    elif incumbent == "st_glmm":
        loc_tr = knn_rate(tr, ct, tr["cluster"].unique()).reindex(tr["cluster"]).to_numpy()
        for variant in ("v1", "v2"):
            name = "st_glmm_incumbent" + ("" if variant == "v1" else "_v2")
            tra, trb = tr[a].reset_index(drop=True), tr[b].reset_index(drop=True)
            Xa, Xb = Xtr[a].reset_index(drop=True), Xtr[b].reset_index(drop=True)
            p = choose(name, GLMM_GRID, lambda p: _auroc(ytr[b], SparseStGlmm(p["C_reg"], wx_cols, variant).fit(
                tra, Xa, loc_tr[a], ytr[a]).predict_proba(trb, Xb, loc_tr[b])))
            m = SparseStGlmm(p["C_reg"], wx_cols, variant).fit(tr, Xtr, loc_tr, ytr)
            scores[name] = {}
            for k, (ev, Xe) in evs.items():
                loc_ev = knn_rate(tr, ct, ev["cluster"].unique(), exclude_self=False).reindex(ev["cluster"]).to_numpy()
                scores[name][k] = m.predict_proba(ev.reset_index(drop=True), Xe.reset_index(drop=True), loc_ev)
    else:
        raise ValueError(incumbent)
    return scores, sel


# --------------------------------------------------------------------------- evaluation
def evaluate(app: str, rows: pd.DataFrame, X: pd.DataFrame, ct: pd.DataFrame, cfg: TrackConfig, *,
             incumbent: str, strata: dict, register: bool, out_dir, manifests, tag: str = "") -> dict:
    """Fit on train rows, score val_known (clusters with >= 1 training row) and val_new; paired AUROC
    differences (HUC8 cluster bootstrap) of each challenger vs every baseline / incumbent, Holm per
    challenger x set. ``strata`` = {name: callable(rows, X, Xtrain) -> bool mask} defined a priori."""
    from sentinel.eval.bootstrap import holm_bonferroni

    tr_m = (rows["role"] == "train").to_numpy()
    tr, Xtr = rows[tr_m].reset_index(drop=True), X[tr_m].reset_index(drop=True)
    C.assert_no_test(tr, rows)
    train_clusters = set(tr["cluster"])
    evs = {}
    for es in ("val_known", "val_new"):
        m = (rows["evalset"] == es).to_numpy()
        if es == "val_known":
            m &= rows["cluster"].isin(train_clusters).to_numpy()
        evs[es] = (rows[m].reset_index(drop=True), X[m].reset_index(drop=True))

    def log_cfg(name, p, status):
        C.log_config_tried(app, f"{name}{tag}", p, status=status)

    scores, sel = fit_all(tr, Xtr, evs, ct, cfg, incumbent, log_cfg)
    res = {"selection": sel, "sets": {}}
    challengers = ("gbm_challenger", "logistic")
    for es, (ev, Xe) in evs.items():
        base = baseline_scores(tr, Xtr, ev, Xe, ct, cfg)
        allsc = {**base, **{k: v[es] for k, v in scores.items() if k != "gbm_challenger_per_seed"}}
        y, grp = ev["y"].to_numpy(float), ev["huc8"].to_numpy()
        r = {"n": int(len(ev)), "n_pos": int(y.sum()), "scores": {}, "paired": {}, "strata": {}}
        for name, s in allsc.items():
            bs = C.binary_scores(y, s, grp, prob=name in ("logistic", "gbm_challenger", "climatology", "location_only",
                                                          "history_rate") or name.startswith("st_glmm"))
            r["scores"][name] = bs
            if register:
                for met in ("auroc", "auprc", "brier"):
                    if met in bs and np.isfinite(bs[met]):
                        C.log_metric(app=app, experiment=f"{es}{tag}", model=name, metric=met, value=bs[met],
                                     split=f"{es}_{cfg.val_years[0]}_{cfg.val_years[1]}", n=bs["n"],
                                     ci={"ci_low": bs[f"{met}_ci"][0], "ci_high": bs[f"{met}_ci"][1]},
                                     ci_method="cluster bootstrap over HUC8, B=2000",
                                     is_baseline=name not in challengers, data_manifests=list(manifests),
                                     extra={"prevalence": bs["prevalence"], "n_pos": bs["n_pos"],
                                            "per_seed_auroc": [C.fast_auroc(y, p) for p in scores["gbm_challenger_per_seed"][es]]
                                            if name == "gbm_challenger" else None})
        for ch in challengers:
            pv, pr = {}, {}
            for b in allsc:
                if b in challengers:
                    continue
                d = C.paired(C.fast_auroc, y, allsc[ch], allsc[b], grp)
                pr[b], pv[b] = d, d["p_value"]
                if register:
                    C.log_metric(app=app, experiment=f"{es}{tag}", model=ch, metric=f"auroc_diff_vs_{b}",
                                 value=d["diff"], split=f"{es}_{cfg.val_years[0]}_{cfg.val_years[1]}", n=int(np.isfinite(allsc[b]).sum()),
                                 ci={"ci_low": d["ci_low"], "ci_high": d["ci_high"]},
                                 ci_method="paired cluster bootstrap over HUC8, B=2000", data_manifests=list(manifests),
                                 extra={"p_value": d["p_value"]})
            h = holm_bonferroni(pv)
            for b in pr:
                pr[b]["holm_reject"] = bool(h[b])
            r["paired"][ch] = pr
        for sn, fn in strata.items():
            m = np.asarray(fn(ev, Xe, Xtr), bool)
            r["strata"][sn] = {"n": int(m.sum()), "n_pos": int(y[m].sum()),
                               "auroc": {k: (C.fast_auroc(y[m], np.asarray(v)[m]) if y[m].sum() > 0 and (1 - y[m]).sum() > 0 else None)
                                         for k, v in allsc.items()}}
        best_simple = max(("climatology", "location_only", "history_rate"), key=lambda k: r["scores"][k].get("auroc", -1))
        r["strongest_simple_baseline"] = best_simple
        res["sets"][es] = r
        pred = ev[["cluster", "date", "y", "huc8"]].copy()
        for k, v in allsc.items():
            pred[f"s_{k}"] = v
        pred.to_parquet(out_dir / f"val_predictions_{es}.parquet", index=False)
    return res


# --------------------------------------------------------------------------- CLI shared by both drivers
def run_cli(mod, argv=None) -> int:
    import argparse
    import json

    from sentinel.core.runlog import TrainingRun

    ap = argparse.ArgumentParser(description=f"{mod.APP} national WQP track")
    ap.add_argument("--freeze-split", action="store_true")
    ap.add_argument("--build-only", action="store_true", help="labels, counts and power check; no model fitted")
    ap.add_argument("--phase1", action="store_true", help="fit train, score val (validation only)")
    ap.add_argument("--no-register", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)
    cfg = mod.CFG
    if a.freeze_split:
        sha, roles = freeze(cfg)
        from collections import Counter
        print(json.dumps({"split_sha256": sha, "huc4_roles": Counter(roles.values())}, default=str))
        return 0
    split, sha = C.load_split(cfg.app, expect_sha=mod.SPLIT_SHA)
    with C.cpu_slot(f"{cfg.app}_{'build' if a.build_only else 'phase1'}"):
        rows, X, ct = build_frame(cfg, split)
        rd = C.run_dir(cfg.app, "build" if a.build_only else f"phase1{a.tag.replace('/', '_')}")
        summary = {"split_sha256": sha, "power": power_check(rows),
                   "wx_missing_frac": float(X["wx_precip_sum_2d"].isna().mean()),
                   "by_state_eval": {f"{a}|{b}": int(n) for (a, b), n in
                                     rows[rows["evalset"].notna()].groupby(["evalset", "state"]).size().items()}}
        C.write_json(rd / "build_summary.json", summary)
        if a.build_only:
            print(json.dumps(summary["power"], indent=1, default=str))
            return 0
        with TrainingRun(C.WORKSTREAM, f"{cfg.app}/phase1{a.tag}", config={"split_sha256": sha, "cfg": str(cfg)},
                         data=list(mod.DATA_MANIFESTS), plan_ref=C.PLAN_REF, artifacts_dir=str(rd)) as run:
            res = evaluate(cfg.app, rows, X, ct, cfg, incumbent=mod.INCUMBENT, strata=mod.strata(),
                           register=not a.no_register, out_dir=rd, manifests=mod.DATA_MANIFESTS, tag=a.tag)
            res["split_sha256"] = sha
            C.write_json(rd / "phase1_results.json", res)
            for es, r in res["sets"].items():
                for k in ("gbm_challenger", "logistic"):
                    run.metric(f"{es}/{k}/auroc", r["scores"][k]["auroc"])
    return 0
