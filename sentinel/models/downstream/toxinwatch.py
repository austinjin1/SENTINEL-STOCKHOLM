# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""D2 ToxinWatch — microcystin > 8 µg/L at Washington lake sites (protocol:
docs/preregistration/downstream_toxinwatch.md).

Unit: site-day with a microcystin lab result in the WA Toxic Algae database (label
``exceed_epa_8ugL_any``; undetermined censored days dropped). Site IDs of one waterbody are merged
into a **lake cluster** (:func:`lake_clusters`) — the unit of the spatial split and of every bootstrap.

Decision time: the morning of the sampling day ``d``. Lab toxin results take days, so every
prior-result feature uses only cluster-days dated ``<= d - EMBARGO_DAYS`` (7; sensitivity 3).
Weather windows end ``d - 1``. ``scum_info`` (an "additional sample information" form exists for the
sample) is a property of the sample, not known before sampling, so it is **not** a feature; results
are reported stratified by it. No upstream-model features in phase 1 (``crossfit.phase1_no_upstream``).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ...core import paths
from ...eval import splits as ES
from . import common as C
from . import crossfit
from . import weather as W

APP = "toxinwatch"
EMBARGO_DAYS = 7
CLUSTER_KM = 2.0
TRAIN_YEARS = (2007, 2018)
VAL_YEARS = (2019, 2021)
TEST_FROM_YEAR = 2022
LAKE_FRACS = {"test": 0.2, "val": 0.2}
GIANT_FRAC = 0.10  # v2 rule: clusters holding > 10 % of all site-days take the train lake role
SPLIT_VERSION = "v2"
SPLIT_FILE = "downstream_toxinwatch_split_v2.json"
SPLIT_SHA = "13852e4a97490da97ddafeefeb757074c2b832c7cb361ab9bf85c077e11bef85"  # v2, frozen 2026-10-24 (protocol §3)
SUPERSEDED_SPLIT_SHA = "04e1482072045c6c641b8d5974aac2b012811ef6ff80db68b21dcb503ccdb383"
MAX_WX_KM = 25.0
ROUTINE_MIN_SITEDAYS = 200  # lake clusters with >= this many site-days form their own split stratum
LAKE_SEED = 20261024
DATA_MANIFESTS = ("labels_wa_toxic_algae", "forcings_daily")
# Curated waterbody aliases for named swim beaches / access points whose site name does not contain the
# waterbody (King County routine beach programme). Keys are lower-cased, whitespace-normalised site names.
# Sources: King County swim-beach locations (shoreline of the named lake); 'Sammamish Landing Beach' has
# implausible published coordinates (-122.688) and is aliased by name only.
WATERBODY_ALIASES = {
    **{n: "washington" for n in (
        "juanita beach", "juanita creek", "houghton beach", "waverly park beach", "marina park beach",
        "matthews beach", "thornton creek", "magnuson beach", "magnuson beach off leash area", "madison park beach",
        "madrona beach", "mount baker beach", "andrews bay - seward park", "groveland park beach",
        "pritchard island beach", "gene coulon", "john's creek", "kennydale beach", "newcastle beach",
        "enatai beach", "luther burbank beach", "meydenbauer bay beach", "medina park north pond",
        "fairweather creek", "lake washington arrowhead pt")},
    **{n: "sammamish" for n in ("idylwood beach", "idylwood creek", "lake sammamish beach",
                                "sammamish landing beach")},
    "moses lake south": "moses",  # review C2 (2026-10-24): southern basin of Moses Lake, > 2 km from other sites
}
_WB_WORD = re.compile(r"\b(lake|river|reservoir|pond|potholes|creek|slough|lagoon)\b", re.I)


# --------------------------------------------------------------------------- lake clusters
def waterbody_key(name: str) -> str:
    """Normalised waterbody name: parenthetical waterbody > text before ',' / ' - ' > full name."""
    n = " ".join((name or "").split())
    if n.lower() in WATERBODY_ALIASES:
        return WATERBODY_ALIASES[n.lower()]
    m = re.search(r"\(([^)]*)\)", n)
    if m and _WB_WORD.search(m.group(1)):
        n = m.group(1)
    else:
        n = re.sub(r"\([^)]*\)", " ", n)
        for sep in (",", " - "):
            if sep in n:
                head = n.split(sep)[0]
                if _WB_WORD.search(head):
                    n = head
                    break
    n = n.lower()
    n = re.sub(r"\blake\b", " ", n)
    n = re.sub(r"[^a-z0-9]+", " ", n)
    return " ".join(n.split())


def lake_clusters(sites: pd.DataFrame, max_km: float = CLUSTER_KM) -> pd.Series:
    """site_id -> cluster id. Union of (a) same normalised waterbody name in the same county (rivers:
    same name regardless of county) and (b) coordinates within ``max_km``."""
    s = sites.reset_index(drop=True)
    n = len(s)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    key = s["site_name"].fillna("").map(waterbody_key)
    is_river = key.str.contains(r"\briver\b")
    county = s["county"].astype("string").fillna("?")
    gkey = np.where(is_river, "R|" + key, "L|" + key + "|" + county)
    first = {}
    for i, k in enumerate(gkey):
        if not key.iloc[i]:
            continue
        if k in first:
            union(i, first[k])
        else:
            first[k] = i
    lat = pd.to_numeric(s["latitude"], errors="coerce").to_numpy()
    lon = pd.to_numeric(s["longitude"], errors="coerce").to_numpy()
    ok = np.flatnonzero(np.isfinite(lat) & np.isfinite(lon))
    if len(ok) > 1:
        d = ES.haversine_km(lat[ok, None], lon[ok, None], lat[None, ok], lon[None, ok])
        ii, jj = np.nonzero(np.triu(d <= max_km, 1))
        for a, b in zip(ii, jj):
            union(ok[a], ok[b])
    root = np.array([find(i) for i in range(n)])
    cid = pd.Series(root).map({r: f"wa_lc{k:04d}" for k, r in enumerate(sorted(set(root)))})
    return pd.Series(cid.to_numpy(), index=s["site_id"].astype(str).to_numpy(), name="lake_cluster")


# --------------------------------------------------------------------------- data
def load_sitedays() -> tuple[pd.DataFrame, pd.DataFrame]:
    P = paths.PROCESSED / "labels_wa_toxic_algae"
    sites = pd.read_parquet(P / "wa_sites.parquet")
    sites["site_id"] = sites["site_id"].astype(str)
    sd = pd.read_parquet(P / "wa_toxin_siteday.parquet")
    sd = sd[sd["toxin"] == "microcystin"].copy()
    sd["site_id"] = sd["site_id"].astype(str)
    sd["date"] = pd.to_datetime(sd["date"])
    lab = sd["exceed_epa_8ugL_any"]
    sd["label_determined"] = lab.notna()
    sd["y"] = lab.map({True: 1.0, False: 0.0}).astype(float)
    cl = lake_clusters(sites)
    sites["lake_cluster"] = sites["site_id"].map(cl)
    sd["lake_cluster"] = sd["site_id"].map(cl)
    sd = sd.merge(sites[["site_id", "latitude", "longitude", "name_has_river_or_creek"]], on="site_id", how="left")
    # cluster centroid fills missing site coordinates (weather lookup only)
    cent = sites.groupby("lake_cluster")[["latitude", "longitude"]].mean()
    for c in ("latitude", "longitude"):
        sd[c] = sd[c].fillna(sd["lake_cluster"].map(cent[c]))
    return sd.reset_index(drop=True), sites


def lake_role_assignment(sd: pd.DataFrame) -> dict:
    """v2 lake-cluster roles, chosen from row counts only (pre-training review C1, 2026-10-24).

    1. Clusters with > GIANT_FRAC of all site-days -> train lake role (their later years still feed the
       "known lakes, future" val/test blocks through the year rule).
    2. Remaining clusters, separately per stratum (>= ROUTINE_MIN_SITEDAYS site-days / other), are dealt
       largest-first to the role whose site-day share is furthest below its target (test 0.2, val 0.2,
       train 0.6); ties broken by sha1(seed|cluster). Whole clusters only.
    """
    import hashlib

    n = sd.groupby("lake_cluster").size()
    total = float(n.sum())
    role = {c: "train" for c in n.index[n > GIANT_FRAC * total]}
    rest = n.drop(list(role))
    target = {"test": LAKE_FRACS["test"], "val": LAKE_FRACS["val"],
              "train": 1.0 - LAKE_FRACS["test"] - LAKE_FRACS["val"]}
    for k, stratum in enumerate((rest[rest >= ROUTINE_MIN_SITEDAYS], rest[rest < ROUTINE_MIN_SITEDAYS])):
        if stratum.empty:
            continue
        tie = {c: hashlib.sha1(f"{LAKE_SEED + k}|{c}".encode()).hexdigest() for c in stratum.index}
        order = sorted(stratum.index, key=lambda c: (-int(stratum[c]), tie[c]))
        load = {r: 0.0 for r in target}
        for c in order:
            tot = sum(load.values()) + float(stratum[c])
            r = max(target, key=lambda r: (target[r] * tot - load[r], r == "train"))
            role[c] = r
            load[r] += float(stratum[c])
    return role


def split_meta() -> dict:
    return {"unit": "site-day (microcystin)", "cluster_km": CLUSTER_KM, "train_years": TRAIN_YEARS,
            "val_years": VAL_YEARS, "test_from_year": TEST_FROM_YEAR, "lake_fracs": LAKE_FRACS,
            "lake_seed": LAKE_SEED, "routine_min_sitedays": ROUTINE_MIN_SITEDAYS,
            "split_version": SPLIT_VERSION, "supersedes_sha256": SUPERSEDED_SPLIT_SHA, "giant_frac": GIANT_FRAC,
            "lake_split": "v2: clusters > 10% of site-days -> train; others per stratum (routine >= 200 site-days / "
                          "other) dealt largest-first to the role furthest below its site-day target "
                          "(test .2 / val .2 / train .6), ties by sha1(seed+k|cluster)",
            "primary_new_lake_window": "val new lakes 2019-2021; test new lakes 2022+; pre-2019 new-lake rows exploratory",
            "embargo_days": EMBARGO_DAYS,
            "row_rule": "test if lake_role==test or year>=2022; val_new if lake_role==val and 2019<=year<=2021; "
                        "val_new_pre2019 (exploratory) if lake_role==val and year<=2018; train if lake_role==train "
                        "and year<=2018; val_known if lake_role==train and 2019<=year<=2021",
            "vintage": "labels_wa_toxic_algae 2026-10-22"}


def assign_rows(sd: pd.DataFrame, lake_role: dict) -> pd.DataFrame:
    """Row role (train/val/test) and evaluation subset (train/val_known/val_new/test)."""
    out = sd.copy()
    lr = out["lake_cluster"].map(lake_role).fillna("test")  # unseen cluster ids are protected
    yr = out["date"].dt.year
    subset = np.select(
        [(lr == "test") | (yr >= TEST_FROM_YEAR),
         (lr == "val") & (yr >= VAL_YEARS[0]),
         lr == "val",
         (lr == "train") & (yr <= TRAIN_YEARS[1]),
         (lr == "train") & (yr >= VAL_YEARS[0]) & (yr <= VAL_YEARS[1])],
        ["test", "val_new", "val_new_pre2019", "train", "val_known"], default="test")
    out["subset"] = subset
    out["role"] = np.where(subset == "train", "train", np.where(subset == "test", "test", "val"))
    return out


# --------------------------------------------------------------------------- prior-result features
def cluster_days(sd: pd.DataFrame) -> pd.DataFrame:
    """Cluster × date history table: any exceedance, max quantified value, n results."""
    d = sd[sd["label_determined"]]
    g = d.groupby(["lake_cluster", "date"]).agg(y=("y", "max"), max_value=("max_value", "max"),
                                                n=("n_results", "sum")).reset_index()
    return g.sort_values(["lake_cluster", "date"]).reset_index(drop=True)


def prior_features(rows: pd.DataFrame, hist: pd.DataFrame, *, embargo_days: int = EMBARGO_DAYS,
                   p0: float = 0.1, a: float = 2.0) -> pd.DataFrame:
    """History features using only cluster-days dated <= date - embargo_days.

    ``p0``: training prevalence for the smoothed rate (fit on train rows by the caller)."""
    out = pd.DataFrame(index=rows.index, dtype=float,
                       columns=["tw_prior_n", "tw_prior_n_exceed", "tw_prior_rate_s", "tw_days_since_last",
                                "tw_last_exceed", "tw_last_log_value", "tw_prior_max_log_365",
                                "tw_prior_exceed_365", "tw_prior_n_365", "tw_prior_exceed_same_year"])
    emb = np.timedelta64(int(embargo_days), "D")
    for cl, rs in rows.groupby("lake_cluster"):
        h = hist[hist["lake_cluster"] == cl]
        hd = h["date"].to_numpy()
        hy = h["y"].to_numpy(float)
        hv = np.log10(np.clip(h["max_value"].to_numpy(float), 1e-3, None))
        cum_y = np.concatenate([[0.0], np.cumsum(hy)])
        for idx, d in zip(rs.index, rs["date"].to_numpy()):
            cut = d - emb
            k = int(np.searchsorted(hd, cut, side="right"))  # history rows [0, k) have date <= d - emb
            n, ne = float(k), float(cum_y[k])
            rec = {"tw_prior_n": n, "tw_prior_n_exceed": ne, "tw_prior_rate_s": (ne + a * p0) / (n + a)}
            if k:
                rec["tw_days_since_last"] = float((d - hd[k - 1]) / np.timedelta64(1, "D"))
                rec["tw_last_exceed"] = hy[k - 1]
                rec["tw_last_log_value"] = hv[k - 1] if np.isfinite(hv[k - 1]) else np.nan
                lo = int(np.searchsorted(hd, cut - np.timedelta64(365, "D"), side="right"))
                win_v = hv[lo:k]
                rec["tw_prior_max_log_365"] = float(np.nanmax(win_v)) if np.isfinite(win_v).any() else np.nan
                rec["tw_prior_exceed_365"] = float(hy[lo:k].sum())
                rec["tw_prior_n_365"] = float(k - lo)
                y0 = np.datetime64(f"{pd.Timestamp(d).year}-01-01")
                lo_y = int(np.searchsorted(hd, y0, side="left"))
                rec["tw_prior_exceed_same_year"] = float(hy[lo_y:k].sum()) if k > lo_y else 0.0
            else:
                rec.update({"tw_prior_exceed_365": 0.0, "tw_prior_n_365": 0.0, "tw_prior_exceed_same_year": 0.0})
            for kk, vv in rec.items():
                out.at[idx, kk] = vv
    return out


def site_and_intensity_features(rows: pd.DataFrame, sitehist: pd.DataFrame, hist: pd.DataFrame, *,
                                embargo_days: int = EMBARGO_DAYS, p0: float = 0.1, a: float = 2.0) -> pd.DataFrame:
    """Iteration 1 (2026-10-24): sampling-SITE history (beach-level, <= d - embargo) and lake sampling
    intensity (cluster-days in the 30 / 90 days ending d - embargo; reactive sampling signals blooms)."""
    out = pd.DataFrame(index=rows.index, dtype=float,
                       columns=["tw_site_prior_n", "tw_site_rate_s", "tw_site_last_exceed", "tw_site_days_since_last",
                                "tw_lake_n_30d", "tw_lake_n_90d", "tw_lake_exceed_90d"])
    emb = np.timedelta64(int(embargo_days), "D")
    for sid, rs in rows.groupby("site_id"):
        h = sitehist[sitehist["site_id"] == sid]
        hd, hy = h["date"].to_numpy(), h["y"].to_numpy(float)
        cum = np.concatenate([[0.0], np.cumsum(hy)])
        for idx, d in zip(rs.index, rs["date"].to_numpy()):
            k = int(np.searchsorted(hd, d - emb, side="right"))
            out.at[idx, "tw_site_prior_n"] = k
            out.at[idx, "tw_site_rate_s"] = (cum[k] + a * p0) / (k + a)
            if k:
                out.at[idx, "tw_site_last_exceed"] = hy[k - 1]
                out.at[idx, "tw_site_days_since_last"] = (d - hd[k - 1]) / np.timedelta64(1, "D")
    for cl, rs in rows.groupby("lake_cluster"):
        h = hist[hist["lake_cluster"] == cl]
        hd, hy = h["date"].to_numpy(), h["y"].to_numpy(float)
        for idx, d in zip(rs.index, rs["date"].to_numpy()):
            k = int(np.searchsorted(hd, d - emb, side="right"))
            k30 = int(np.searchsorted(hd, d - emb - np.timedelta64(30, "D"), side="right"))
            k90 = int(np.searchsorted(hd, d - emb - np.timedelta64(90, "D"), side="right"))
            out.at[idx, "tw_lake_n_30d"] = k - k30
            out.at[idx, "tw_lake_n_90d"] = k - k90
            out.at[idx, "tw_lake_exceed_90d"] = hy[k90:k].sum()
    return out


def site_days(sd: pd.DataFrame) -> pd.DataFrame:
    d = sd[sd["label_determined"]].groupby(["site_id", "date"]).agg(y=("y", "max")).reset_index()
    return d.sort_values(["site_id", "date"]).reset_index(drop=True)


# --------------------------------------------------------------------------- feature table
WX_KEEP = ("precip_sum_3d", "precip_sum_7d", "precip_sum_30d", "tmax_mean_3d", "tmax_mean_7d", "tmax_mean_30d",
           "tmin_mean_7d", "tmax_anom_7d", "tmax_anom_30d", "srad_mean_3d", "srad_mean_7d", "wind_mean_3d",
           "wind_mean_7d", "vpd_mean_7d", "dry_days_14d")


def build_features(rows: pd.DataFrame, hist: pd.DataFrame, *, p0: float, embargo_days: int = EMBARGO_DAYS,
                   clim_years=range(TRAIN_YEARS[0], TRAIN_YEARS[1] + 1), iteration: str = "base",
                   sitehist: pd.DataFrame | None = None) -> tuple[pd.DataFrame, list[str]]:
    if "role" in rows and (rows["role"] == "test").any():   # only inside the single authorised test look
        from .testlook import require_authorised

        require_authorised(APP, SPLIT_SHA)
    X = pd.DataFrame(index=rows.index)
    doy = rows["date"].dt.dayofyear.to_numpy(float)
    X["doy_sin"], X["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    X["lat"], X["lon"] = rows["latitude"].astype(float), rows["longitude"].astype(float)
    X["is_river"] = rows["name_has_river_or_creek"].fillna(False).astype(float)
    X = X.join(prior_features(rows, hist, embargo_days=embargo_days, p0=p0))
    if iteration in ("it1", "it2"):
        X = X.join(site_and_intensity_features(rows, sitehist, hist, embargo_days=embargo_days, p0=p0))
    ok = rows["latitude"].notna() & rows["longitude"].notna()
    nc = W.nearest_cells(rows.loc[ok, "latitude"], rows.loc[ok, "longitude"])
    nc.index = rows.index[ok]
    q = pd.DataFrame({"wx_site_key": nc["wx_site_key"], "date": rows.loc[ok, "date"]})
    series = W.load_series(q["wx_site_key"].unique(), start=f"{TRAIN_YEARS[0] - 1}-01-01")
    clim = W.doy_climatology(series, clim_years)
    wx = W.antecedent_features(q, series, clim=clim)
    wx = wx[[f"wx_{c}" for c in WX_KEEP if f"wx_{c}" in wx]]
    X = X.join(wx)
    X["wx_cell_km"] = nc["wx_cell_km"]
    far = X["wx_cell_km"].to_numpy() > MAX_WX_KM
    X.loc[far, [c for c in X.columns if c.startswith("wx_") and c != "wx_cell_km"]] = np.nan
    cols = [c for c in X.columns if c != "wx_cell_km"]
    crossfit.phase1_no_upstream(cols)
    return X, cols


# --------------------------------------------------------------------------- baselines (fit on train rows only)
def baseline_scores(train: pd.DataFrame, Xtr: pd.DataFrame, ev: pd.DataFrame, Xev: pd.DataFrame,
                    k_nn: int = 5) -> dict[str, np.ndarray]:
    p0 = float(train["y"].mean())
    out = {"prevalence": np.full(len(ev), p0)}
    # season climatology: month exceedance rate (train)
    mrate = train.groupby(train["date"].dt.month)["y"].mean()
    out["climatology_month"] = ev["date"].dt.month.map(mrate).fillna(p0).to_numpy(float)
    # location-only: smoothed exceedance rate of the k nearest TRAIN clusters (own cluster included when known)
    out["location_knn"] = knn_lake_rate(train, ev, k=k_nn, exclude_self=False)
    # persistence: last available (<= d-7) exceedance at the lake, tie-broken by its value; no history -> p0
    le = Xev["tw_last_exceed"].to_numpy(float)
    lv = Xev["tw_last_log_value"].to_numpy(float)
    pers = np.where(np.isfinite(le), le + 0.01 * np.nan_to_num(lv, nan=-3.0), p0)
    out["persistence"] = pers
    out["lake_history_rate"] = Xev["tw_prior_rate_s"].to_numpy(float)
    return out


def knn_lake_rate(train: pd.DataFrame, ev: pd.DataFrame, *, k: int = 5, exclude_self: bool = False) -> np.ndarray:
    """Inverse-distance-weighted smoothed exceedance rate of the k nearest TRAIN lake clusters.

    ``exclude_self=True`` drops the row's own cluster (spatial-neighbourhood term of the incumbent; used for
    training rows so the term is never in-sample)."""
    p0 = float(train["y"].mean())
    cr = train.groupby("lake_cluster").agg(n=("y", "size"), ne=("y", "sum"), lat=("latitude", "mean"),
                                           lon=("longitude", "mean")).dropna()
    rate = ((cr["ne"] + 2 * p0) / (cr["n"] + 2)).to_numpy()
    names = cr.index.to_numpy()
    ec = ev.groupby("lake_cluster")[["latitude", "longitude"]].mean()
    res = {}
    for cl, r in ec.iterrows():
        if not np.isfinite(r["latitude"]):
            res[cl] = p0
            continue
        d = ES.haversine_km(r["latitude"], r["longitude"], cr["lat"].to_numpy(), cr["lon"].to_numpy())
        order = [i for i in np.argsort(d) if not (exclude_self and names[i] == cl)][:k]
        wts = 1.0 / (d[order] + 1.0)
        res[cl] = float(np.sum(wts * rate[order]) / wts.sum())
    return ev["lake_cluster"].map(res).fillna(p0).to_numpy(float)


def fit_logistic(Xtr: pd.DataFrame, ytr, cols, seed: int = 0, C_reg: float = 1.0):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    m = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                      LogisticRegression(C=C_reg, max_iter=2000, random_state=seed))
    m.fit(Xtr[cols].to_numpy(float), np.asarray(ytr, float))
    return m


GBM_PARAMS = {"objective": "binary", "learning_rate": 0.03, "num_leaves": 15, "min_child_samples": 40,
              "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8, "reg_lambda": 1.0,
              "n_estimators": 2000, "num_threads": C.N_THREADS, "verbose": -1}


def fit_gbm(Xtr: pd.DataFrame, train: pd.DataFrame, cols, seed: int = 0, params: dict | None = None):
    """LightGBM; n_estimators chosen by early stopping on an INNER time split of the training rows
    (fit <= 2015, stop on 2016–2018), then refit on all training rows. Validation data never used."""
    import lightgbm as lgb

    yr = train["date"].dt.year.to_numpy()
    inner = yr >= INNER_FROM_YEAR
    p = dict(GBM_PARAMS, **(params or {}), random_state=seed)
    m0 = lgb.LGBMClassifier(**p)
    m0.fit(Xtr.loc[~inner, cols], train.loc[~inner, "y"], eval_set=[(Xtr.loc[inner, cols], train.loc[inner, "y"])],
           eval_metric="binary_logloss", callbacks=[lgb.early_stopping(100, verbose=False)])
    best = int(m0.best_iteration_ or GBM_PARAMS["n_estimators"])
    m = lgb.LGBMClassifier(**dict(p, n_estimators=max(best, 50)))
    m.fit(Xtr[cols], train["y"])
    m.best_inner_iteration_ = best
    m.inner_logloss_ = float(m0.best_score_["valid_0"]["binary_logloss"])
    return m


# Tuning parity (protocol §6): the same number of configurations (4) per learned model, selected by
# log-loss on the inner time split of TRAINING rows (fit <= 2015, score 2016–2018).
INNER_FROM_YEAR = 2016
GRIDS = {"st_glmm_incumbent": [{"C_reg": c} for c in (0.01, 0.1, 1.0, 10.0)],
         "st_glmm_incumbent_v2": [{"C_reg": c, "variant": "v2"} for c in (0.01, 0.1, 1.0, 10.0)],
         "logistic": [{"C_reg": c} for c in (0.01, 0.1, 1.0, 10.0)],
         "gbm_challenger": [{"num_leaves": nl, "min_child_samples": mc} for nl in (7, 15) for mc in (20, 60)]}


# --------------------------------------------------------------------------- incumbent re-implementation
def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def st_glmm_design(rows: pd.DataFrame, X: pd.DataFrame, loc_rate: np.ndarray) -> np.ndarray:
    """Design of the INLA-class spatiotemporal incumbent (Schaeffer et al. 2024 model class, re-implemented
    on toxin labels): logit of the partially pooled lake effect (empirical-Bayes shrunken history rate,
    <= d-7), logit of the spatial neighbourhood effect (k-NN train-lake rate, a discrete stand-in for the
    SPDE field), seasonal harmonics, and an AR(1)-type state term (last available exceedance, its age)."""
    last = X["tw_last_exceed"].to_numpy(float)
    has = np.isfinite(last)
    age = np.log1p(np.nan_to_num(X["tw_days_since_last"].to_numpy(float), nan=3650.0))
    doy = rows["date"].dt.dayofyear.to_numpy(float)
    return np.column_stack([
        _logit(X["tw_prior_rate_s"].to_numpy(float)), _logit(loc_rate),
        np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25),
        np.sin(4 * np.pi * doy / 365.25), np.cos(4 * np.pi * doy / 365.25),
        np.where(has, last, 0.0), has.astype(float), np.where(has, last, 0.0) * age, age,
    ])


class StGlmm:
    """INLA-class incumbent, strengthened after the pre-training review (W1, 2026-10-24): the design of
    :func:`st_glmm_design` + the same antecedent-weather windows the GBM sees (train-median imputed,
    standardised) + an L2-penalised lake one-hot (a ridge-shrunk lake random intercept; lakes unseen in
    training get 0 and rely on the pooled/spatial terms). ``C_reg`` is the inverse L2 strength."""

    def __init__(self, C_reg: float = 1.0, wx_cols=None, variant: str = "v1"):
        self.C_reg, self.wx_cols, self.variant = C_reg, list(wx_cols or []), variant

    def _design(self, rows, X, loc_rate, fit=False):
        from sklearn.preprocessing import SplineTransformer, StandardScaler

        core = st_glmm_design(rows, X, loc_rate)
        if self.variant == "v2":
            # closer INLA analogue (lead guidance 2026-10-25): AR(1)-type latent decay of the last known
            # result (respecting d-7) at two time constants, and an additive spatial smooth (cubic B-splines
            # in lat and lon fitted on training coordinates) in place of an SPDE field.
            age = np.nan_to_num(X["tw_days_since_last"].to_numpy(float), nan=3650.0)
            lv = np.nan_to_num(X["tw_last_log_value"].to_numpy(float), nan=-1.0)
            le = np.nan_to_num(X["tw_last_exceed"].to_numpy(float), nan=0.0)
            pm = np.nan_to_num(X["tw_prior_max_log_365"].to_numpy(float), nan=-1.0)
            ar = [v * np.exp(-age / tau) for tau in (14.0, 60.0) for v in (lv, le)]
            ll = np.column_stack([np.nan_to_num(X["lat"].to_numpy(float), nan=47.4),
                                  np.nan_to_num(X["lon"].to_numpy(float), nan=-121.0)])
            if fit:
                self.spl_ = SplineTransformer(n_knots=6, degree=3, extrapolation="constant").fit(ll)
            core = np.hstack([core, np.column_stack(ar + [pm]), self.spl_.transform(ll)])
        wx = X[self.wx_cols].to_numpy(float) if self.wx_cols else np.empty((len(X), 0))
        if fit:
            self.med_ = np.nanmedian(wx, axis=0) if wx.shape[1] else np.empty(0)
        wx = np.where(np.isfinite(wx), wx, self.med_)
        cont = np.hstack([core, wx])
        if fit:
            self.scaler_ = StandardScaler().fit(cont)
            self.lakes_ = sorted(rows["lake_cluster"].unique())
        cont = self.scaler_.transform(cont)
        lk = {c: i for i, c in enumerate(self.lakes_)}
        oh = np.zeros((len(rows), len(self.lakes_)))
        idx = rows["lake_cluster"].map(lk)
        ok = idx.notna().to_numpy()
        oh[np.flatnonzero(ok), idx[ok].astype(int).to_numpy()] = 1.0
        return np.hstack([cont, oh])

    def fit(self, rows, X, loc_rate, y):
        from sklearn.linear_model import LogisticRegression

        D = self._design(rows, X, loc_rate, fit=True)
        self.m_ = LogisticRegression(C=self.C_reg, max_iter=5000).fit(D, np.asarray(y, float))
        return self

    def predict_proba(self, rows, X, loc_rate):
        return self.m_.predict_proba(self._design(rows, X, loc_rate))[:, 1]
