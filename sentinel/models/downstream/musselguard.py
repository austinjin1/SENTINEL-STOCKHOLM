# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""D6 MusselGuard — rolling-origin forecast of new dreissenid (zebra/quagga) HUC12 detections.

Protocol: docs/preregistration/downstream_musselguard.md.

* **Universe** = CONUS WBD HUC12s (`processed/wbd_huc12`, 2026-10-24 pull of WBD MapServer layer 6).
* **Infested by origin T** = HUC12 whose earliest USGS NAS dreissenid record (any status, group
  ``dreissena_any`` = zebra + quagga + *Dreissena* sp.) has ``first_year_any <= T``.
* **Candidates at T** = universe HUC12 not infested by T. **Positive** = first record in (T, T + H],
  H = 3. Everything else is a *presumed* negative (presence-only data; undetected infestations are
  label noise).
* **Origins** (forward-chaining): train T = 2006, 2009, 2012 (windows end ≤ 2015 = val origin); val
  T = 2015 (2016–2018); test T = 2018 (2019–2021) and 2021 (2022–2024, ends 2 y before the 2026 NAS
  vintage). Spatial secondary: leave-one-HUC2-out over the train origins.
* **Features at T only**: distance to / counts of NAS record locations dated ≤ T (all countries, so
  Canadian Great-Lakes records count as sources), recent spread (records in (T−3, T]), WBD flow
  connectivity (``tohuc`` hops to an infested HUC12 upstream/downstream), location (centroid, area), and
  optional observed WQP specific-conductance / pH medians per HUC8 over [T−9, T] (calcium proxies;
  WQP harmonised data has no calcium variable).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...core import paths
from ...eval.splits import haversine_km
from .common import N_THREADS as C_N_THREADS

APP = "musselguard"
SPLIT_SHA = "04daee8a36e240443a8f711d8b8286c4db2f0e9cd2b9377f0b5d963eeb66dd46"
SEEDS = (0, 1, 2)
ORIGINS = {2006: "train", 2009: "train", 2012: "train", 2015: "val", 2018: "test", 2021: "test"}
HORIZON = 3
GROUP = "dreissena_any"
RADII_KM = (25, 50, 100, 200)
FRONT_KM = 100.0
MAX_HOPS = 50
CONUS_HUC2 = [f"{i:02d}" for i in range(1, 19)]
DATA_MANIFESTS = ["labels_nas_dreissena", "wbd_huc12", "census_county"]

LOCATION_FEATURES = ["lat", "lon", "log_area"]
PRESSURE_FEATURES = ["log_dist_nearest_km", "log_dist_recent_km"] + [f"n_src_{r}km" for r in RADII_KM] + \
                    ["n_recent_100km", "hops_from_infested_upstream", "hops_to_infested_downstream",
                     "infested_upstream_within5", "infested_downstream_within5"]
WQP_FEATURES = ["huc8_spcond_median", "huc8_ph_median", "huc8_n_spcond"]


# --------------------------------------------------------------------------- inputs
def load_universe() -> pd.DataFrame:
    u = pd.read_parquet(paths.PROCESSED / "wbd_huc12" / "huc12.parquet")
    u = u[u["huc2"].isin(CONUS_HUC2) & u["lat"].notna() & has_us_state(u["states"])].copy()
    u["log_area"] = np.log10(u["areasqkm"].clip(lower=0.1))
    return u.reset_index(drop=True)


def load_first_detection() -> pd.DataFrame:
    f = pd.read_parquet(paths.PROCESSED / "labels_nas_dreissena" / "nas_dreissena_first_detection_huc12.parquet")
    f = f[f["species_group"] == GROUP][["huc12", "first_year_any", "first_year_actual", "first_year_established"]]
    return f.astype({"huc12": str})


def load_records() -> pd.DataFrame:
    """NAS dreissenid record points (all countries) with record year — record level, not aggregated.
    Aggregation to location bins happens per origin in :func:`sources_at` so that nothing dated after
    T can move a bin (review C1, 2026-10-24)."""
    o = pd.read_parquet(paths.PROCESSED / "labels_nas_dreissena" / "nas_dreissena_occurrences.parquet",
                        columns=["species_id", "year", "latitude", "longitude", "coord_flag"])
    o = o[o["year"].notna() & o["latitude"].notna() & (o["coord_flag"].fillna("") == "")].copy()
    o["year"] = o["year"].astype(int)
    o["loc"] = o["latitude"].round(2).astype(str) + "_" + o["longitude"].round(2).astype(str)
    return o[["loc", "latitude", "longitude", "year"]].reset_index(drop=True)


def sources_at(records: pd.DataFrame, T: int) -> pd.DataFrame:
    """0.01° location bins built ONLY from records dated <= T (bin position = mean of those records)."""
    r = records[records["year"] <= T]
    s = r.groupby("loc").agg(lat=("latitude", "mean"), lon=("longitude", "mean"), first_year=("year", "min"))
    return s.reset_index()


US_STATES = set("AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ "
                "NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY".split())


def has_us_state(states: pd.Series) -> pd.Series:
    """True when the WBD ``states`` list contains at least one US state (drops CN/MX-only units)."""
    return states.fillna("").astype(str).str.upper().str.split(r"[,\s]+").map(lambda t: bool(US_STATES & set(t)))


# --------------------------------------------------------------------------- features
GRAVITY_ALPHAS = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
GRAVITY_FLOOR_KM = 5.0
SMOOTH_BANDWIDTHS_KM = (10, 20, 40, 80, 160, 320)


def _nearest_and_counts(lat, lon, slat, slon, radii, alphas=()) -> tuple[np.ndarray, dict, dict]:
    """Nearest-source distance, source counts within ``radii`` and gravity sums Σ max(d, 5 km)^-α."""
    n = len(lat)
    dmin = np.full(n, np.inf)
    counts = {r: np.zeros(n, int) for r in radii}
    grav = {a: np.zeros(n) for a in alphas}
    if len(slat) == 0:
        return dmin, counts, grav
    for s in range(0, n, 2048):
        d = haversine_km(lat[s:s + 2048, None], lon[s:s + 2048, None], slat[None], slon[None])
        dmin[s:s + 2048] = d.min(1)
        for r in radii:
            counts[r][s:s + 2048] = (d <= r).sum(1)
        df = np.maximum(d, GRAVITY_FLOOR_KM)
        for a in alphas:
            grav[a][s:s + 2048] = (df ** -a).sum(1)
    return dmin, counts, grav


def kernel_smooth_status(universe: pd.DataFrame, infested: set, bandwidths=SMOOTH_BANDWIDTHS_KM,
                         cell_km: float = 5.0) -> dict:
    """Spatial smoother of infestation status at T (an indicator-kriging / co-kriging analogue):
    Gaussian-kernel share of infested HUC12 centroids among all HUC12 centroids, on a 5-km grid in an
    equirectangular projection (lat0 = 39°). Returns {bandwidth: array aligned with universe rows}."""
    from scipy.ndimage import gaussian_filter

    kx = 111.32 * np.cos(np.radians(39.0))
    x = (universe["lon"].to_numpy(float) + 125.0) * kx / cell_km
    y = (universe["lat"].to_numpy(float) - 24.0) * 110.57 / cell_km
    ix, iy = np.floor(x).astype(int), np.floor(y).astype(int)
    shape = (iy.max() + 1, ix.max() + 1)
    allg = np.zeros(shape)
    np.add.at(allg, (iy, ix), 1.0)
    inf = universe["huc12"].isin(infested).to_numpy()
    infg = np.zeros(shape)
    np.add.at(infg, (iy[inf], ix[inf]), 1.0)
    out = {}
    for h in bandwidths:
        sig = h / cell_km
        num = gaussian_filter(infg, sig, mode="constant", truncate=3.0)
        den = gaussian_filter(allg, sig, mode="constant", truncate=3.0)
        out[h] = np.where(den[iy, ix] > 0, num[iy, ix] / np.maximum(den[iy, ix], 1e-12), 0.0)
    return out


def network_hops(universe: pd.DataFrame, infested: set, max_hops: int = MAX_HOPS) -> tuple[pd.Series, pd.Series]:
    """(hops downstream from the nearest infested HUC12 to this unit, hops from this unit down to an
    infested HUC12) along WBD ``tohuc``; NaN if none within ``max_hops``."""
    to = dict(zip(universe["huc12"], universe["tohuc"]))
    from_inf = {}
    for h in infested:
        cur, k = to.get(h), 1
        while cur is not None and k <= max_hops and cur in to:
            if cur not in from_inf or from_inf[cur] > k:
                from_inf[cur] = k
            cur, k = to.get(cur), k + 1
    to_inf = {}
    for h in universe["huc12"]:
        cur, k = to.get(h), 1
        while cur is not None and k <= max_hops and cur in to:
            if cur in infested:
                to_inf[h] = k
                break
            cur, k = to.get(cur), k + 1
    idx = universe["huc12"]
    return idx.map(from_inf).astype(float), idx.map(to_inf).astype(float)


def wqp_huc8_proxies(origin: int, years_back: int = 9) -> pd.DataFrame:
    """Observed WQP specific conductance / pH medians per HUC8 over sample dates in
    [T − years_back, T] (calendar years, i.e. nothing after 31 Dec T). Censored rows (value NaN) drop out."""
    out = []
    for var, col in (("spcond", "huc8_spcond_median"), ("ph", "huc8_ph_median")):
        d = pd.read_parquet(paths.PROCESSED / "wqp_discrete", filters=[("variable", "=", var)],
                            columns=["huc8", "activity_start_date", "value"])
        yr = pd.to_datetime(d["activity_start_date"], errors="coerce").dt.year
        d = d[(yr >= origin - years_back) & (yr <= origin) & d["huc8"].notna() & d["value"].notna()]
        g = d.groupby(d["huc8"].astype(str))["value"].agg(["median", "size"])
        g.columns = [col, "huc8_n_" + var]
        out.append(g)
    res = pd.concat(out, axis=1).drop(columns=["huc8_n_ph"], errors="ignore")
    res.index.name = "huc8"
    return res.reset_index()


def build_origin(T: int, universe: pd.DataFrame, first: pd.DataFrame, records: pd.DataFrame, *,
                 horizon: int = HORIZON, wqp: pd.DataFrame | None = None) -> pd.DataFrame:
    """Candidate rows for origin T. Only information dated ≤ T enters the features: ``records`` are
    record-level NAS points; bins are built from records dated ≤ T inside this function."""
    if ORIGINS.get(int(T)) == "test":     # only inside the single authorised test look
        from .testlook import require_authorised

        require_authorised(APP, SPLIT_SHA)
    sources = sources_at(records, T)
    fy = dict(zip(first["huc12"], first["first_year_any"].astype("Int64")))
    u = universe.copy()
    u["first_year"] = u["huc12"].map(fy)
    infested = set(u.loc[u["first_year"].notna() & (u["first_year"] <= T), "huc12"])
    cand = u[~u["huc12"].isin(infested)].copy()
    fyc = cand["first_year"]
    cand["y"] = (fyc.notna() & (fyc > T) & (fyc <= T + horizon)).astype(int)
    src = sources[sources["first_year"] <= T]
    rec = sources[(sources["first_year"] > T - 3) & (sources["first_year"] <= T)]
    lat, lon = cand["lat"].to_numpy(float), cand["lon"].to_numpy(float)
    dmin, counts, grav = _nearest_and_counts(lat, lon, src["lat"].to_numpy(float), src["lon"].to_numpy(float),
                                             RADII_KM, GRAVITY_ALPHAS)
    drec, crec, _ = _nearest_and_counts(lat, lon, rec["lat"].to_numpy(float), rec["lon"].to_numpy(float), (100,))
    for a in GRAVITY_ALPHAS:
        cand[f"gravity_a{a:g}"] = np.log10(grav[a] + 1e-12)
    if "log_pop_50km" in u:      # Kanankege-2018 re-implementation (incumbent; see protocol §11)
        pos0 = u.index.get_indexer(cand.index)
        if USE_EXT_GRID:
            for k_ in sorted({g[1] for g in KRIGING_GRID_EXT}):
                for n_ in sorted({g[2] for g in KRIGING_GRID_EXT}):
                    rg = [g[0] for g in KRIGING_GRID_EXT if g[1] == k_ and g[2] == n_]
                    kk = kanankege_scores(u, infested, u["log_pop_50km"].to_numpy(float), ranges=rg, nugget=n_,
                                          k=k_, chunk=8000 if k_ <= 32 else 3000)
                    for a_ in rg:
                        cand[kanankege_col(a_, k_, n_)] = kk[a_][pos0]
        else:
            kk = kanankege_scores(u, infested, u["log_pop_50km"].to_numpy(float))
            for a_ in KRIGING_RANGES_KM:
                cand[f"kanankege_r{a_}"] = kk[a_][pos0]
    sm = kernel_smooth_status(u, infested)
    pos = u.index.get_indexer(cand.index)
    for h in SMOOTH_BANDWIDTHS_KM:
        cand[f"smooth_h{h}"] = sm[h][pos]
    cand["dist_nearest_km"] = dmin
    cand["log_dist_nearest_km"] = np.log10(np.minimum(dmin, 5000) + 1)
    cand["log_dist_recent_km"] = np.log10(np.minimum(drec, 5000) + 1)
    for r in RADII_KM:
        cand[f"n_src_{r}km"] = counts[r]
    cand["n_recent_100km"] = crec[100]
    up, dn = network_hops(u, infested)
    cand["hops_from_infested_upstream"] = up.reindex(cand.index).to_numpy()
    cand["hops_to_infested_downstream"] = dn.reindex(cand.index).to_numpy()
    cand["infested_upstream_within5"] = (cand["hops_from_infested_upstream"] <= 5).astype(float)
    cand["infested_downstream_within5"] = (cand["hops_to_infested_downstream"] <= 5).astype(float)
    cand["front"] = cand["dist_nearest_km"] <= FRONT_KM
    cand["mn"] = cand["states"].fillna("").str.contains("MN")
    cand["origin"] = T
    cand["n_infested_at_T"] = len(infested)
    if wqp is not None:
        cand = cand.merge(wqp, on="huc8", how="left")
    return cand.drop(columns=["first_year"]).reset_index(drop=True)


def sc_rule_score(df: pd.DataFrame) -> np.ndarray:
    """Rule baseline (Whittier-2008 analogue on specific conductance, since Ca is unavailable):
    risk class by HUC8 median SC (<100, 100–200, 200–300, >300 µS/cm → 0..3) — a coarse habitat
    suitability rule, ties broken by nothing (it is a rule)."""
    sc = df.get("huc8_spcond_median")
    if sc is None:
        return np.full(len(df), np.nan)
    return np.digitize(sc.to_numpy(float), [100, 200, 300]).astype(float) + np.where(np.isfinite(sc), 0, np.nan)


# --------------------------------------------------------------------------- phase-1 modelling
def design(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Hop features: NaN (no infested HUC12 within MAX_HOPS) -> MAX_HOPS + 1 for linear models."""
    X = df[list(cols)].astype(float).copy()
    for c in ("hops_from_infested_upstream", "hops_to_infested_downstream"):
        if c in X:
            X[c] = X[c].fillna(MAX_HOPS + 1)
    return X


RULE_FEATURES = [f"gravity_a{a:g}" for a in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0)] + \
                [f"smooth_h{h}" for h in (10, 20, 40, 80, 160, 320)]
MODEL_SPECS = {
    # name: (kind, features, grid) — same inner-CV budget (6 configurations) for every learned model
    "location_only_gbm": ("gbm", LOCATION_FEATURES, None),
    "pressure_logit": ("logit", PRESSURE_FEATURES + RULE_FEATURES + ["log_area"],
                       [{"C": c} for c in (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)]),
    "pressure_gbm": ("gbm", PRESSURE_FEATURES + RULE_FEATURES + LOCATION_FEATURES, None),
}
GBM_GRID = [{"num_leaves": nl, "min_child_samples": mc, "learning_rate": 0.03, "n_estimators": 400}
            for nl in (7, 15, 31) for mc in (50, 300)]


def fit_model(kind: str, params: dict, X, y, seed: int = 0):
    if kind == "logit":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, **params)).fit(X, y)
    import lightgbm as lgb

    return lgb.LGBMClassifier(objective="binary", subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                              n_jobs=C_N_THREADS, random_state=seed, verbose=-1, **params).fit(X, y)


def front_auprc(y, s, front) -> float:
    """The primary endpoint: AUPRC among invasion-front candidates."""
    from ...eval.metrics import auprc

    f = np.asarray(front, bool)
    yy = np.asarray(y)[f]
    return float(auprc(yy, np.asarray(s)[f])) if 0 < yy.sum() < len(yy) else float("nan")


def inner_cv_select(kind: str, grid: list, X: pd.DataFrame, y, groups, front, *, seeds=(0,)) -> tuple[dict, list]:
    """Hyperparameters by 5-fold HUC2-grouped CV on TRAIN origins only; criterion = front-stratum AUPRC
    (= the primary endpoint, review W9)."""
    from sklearn.model_selection import GroupKFold

    log = []
    for p in grid:
        vals = []
        for a, b in GroupKFold(5).split(X, groups=groups):
            pr = np.mean([fit_model(kind, p, X.iloc[a], np.asarray(y)[a], seed=sd).predict_proba(X.iloc[b])[:, 1]
                          for sd in (seeds if kind == "gbm" else (0,))], axis=0)
            vals.append(front_auprc(np.asarray(y)[b], pr, np.asarray(front)[b]))
        log.append({"params": p, "inner_cv_front_auprc": float(np.nanmean(vals))})
    return max(log, key=lambda r: r["inner_cv_front_auprc"])["params"], log


def select_rule(df: pd.DataFrame, columns: list[str]) -> tuple[str, list]:
    """One-parameter rule baselines (gravity α, smoother bandwidth): pick the column with the best
    front-stratum AUPRC on the TRAIN origins (same 6-configuration budget as the learned models)."""
    log = [{"params": {"column": c}, "train_front_auprc": front_auprc(df["y"], df[c], df["front"])} for c in columns]
    return max(log, key=lambda r: r["train_front_auprc"])["params"]["column"], log


# --------------------------------------------------------------------------- Kanankege-2018 re-implementation
POP_RADII_KM = (25, 50, 100)
POP_FEATURES = [f"log_pop_{r}km" for r in POP_RADII_KM]
KRIGING_RANGES_KM = (10, 20, 40, 80, 160, 320)
KRIGING_NUGGET = 0.1      # nugget as a fraction of the residual sill (fixed; the range is tuned)
KRIGING_K = 32            # local neighbourhood size
INCUMBENT_FEATURES = [f"kanankege_r{a}" for a in KRIGING_RANGES_KM]
# it2b (protocol §11b): incumbent-tuning sensitivity — the it2 selection hit the top of the range grid, so
# the incumbent gets a wider grid (23 configurations, ~4x the challenger's budget; k=128 dropped as
# computationally infeasible, protocol §11b deviation). (a, k, nugget); the it2 configurations keep their column names.
KRIGING_GRID_EXT = ([(a, 32, n) for a in (10, 20, 40, 80, 160, 320, 640, 1280, 2560) for n in (0.1, 0.5)]
                    + [(a, 64, 0.1) for a in (160, 320, 640, 1280, 2560)])   # 23 configs (amended 2026-11-08)
USE_EXT_GRID = False


def kanankege_col(a, k=KRIGING_K, nugget=KRIGING_NUGGET) -> str:
    return f"kanankege_r{a}" if (k == KRIGING_K and nugget == KRIGING_NUGGET) else f"kanankege_r{a}_k{k}_n{nugget:g}"


def incumbent_features() -> list[str]:
    return [kanankege_col(*g) for g in KRIGING_GRID_EXT] if USE_EXT_GRID else list(INCUMBENT_FEATURES)


def population_pressure(universe: pd.DataFrame, counties: pd.DataFrame | None = None) -> pd.DataFrame:
    """log10(1 + 2010 population of counties whose internal point lies within r km of the HUC12 centroid)
    — the public proxy for Kanankege's boater-traffic and road-access covariates (static, 2010 census)."""
    if counties is None:
        counties = pd.read_parquet(paths.PROCESSED / "census_county" / "counties_2010.parquet")
    lat, lon = universe["lat"].to_numpy(float), universe["lon"].to_numpy(float)
    clat, clon, pop = (counties[c].to_numpy(float) for c in ("lat", "lon", "pop2010"))
    out = {f"log_pop_{r}km": np.zeros(len(lat)) for r in POP_RADII_KM}
    for s in range(0, len(lat), 2048):
        d = haversine_km(lat[s:s + 2048, None], lon[s:s + 2048, None], clat[None], clon[None])
        for r in POP_RADII_KM:
            out[f"log_pop_{r}km"][s:s + 2048] = np.log10(1 + (pop[None] * (d <= r)).sum(1))
    return pd.DataFrame(out, index=universe.index)


_KNN_CACHE: dict = {}


def knn_index(universe: pd.DataFrame, k: int = KRIGING_K) -> tuple[np.ndarray, np.ndarray]:
    """k nearest OTHER HUC12 centroids (haversine BallTree); cached per universe size."""
    key = (len(universe), k, float(universe["lat"].sum()))
    if key not in _KNN_CACHE:
        from sklearn.neighbors import BallTree

        X = np.radians(universe[["lat", "lon"]].to_numpy(float))
        dist, idx = BallTree(X, metric="haversine").query(X, k=k + 1)
        _KNN_CACHE[key] = (idx[:, 1:], dist[:, 1:] * 6371.0088)   # drop self -> leave-self-out kriging
    return _KNN_CACHE[key]


def _pairwise_km(lat, lon):
    """(n, k, k) haversine distances among each row's neighbours."""
    la, lo = np.radians(lat), np.radians(lon)
    a = np.sin((la[:, :, None] - la[:, None, :]) / 2) ** 2 + \
        np.cos(la[:, :, None]) * np.cos(la[:, None, :]) * np.sin((lo[:, :, None] - lo[:, None, :]) / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def kanankege_scores(universe: pd.DataFrame, infested: set, drift: np.ndarray, ranges=KRIGING_RANGES_KM,
                     nugget: float = KRIGING_NUGGET, k: int = KRIGING_K, chunk: int = 8000) -> dict:
    """Regression-kriging analogue of Kanankege et al. 2018 probability co-kriging.

    Primary variable: invasion status at T (1 = HUC12 infested by T). Secondary variable (co-kriging's
    cross-correlated covariate, here an external drift): population pressure ``drift`` (their boater
    traffic / road access). Stage 1: logistic regression of status on the drift → p. Stage 2: local
    ordinary kriging (exponential covariance, range ``a``, nugget fraction ``nugget``, k neighbours,
    leave-self-out) of the residual status − p. Score = p + kriged residual. Returns {a: scores}."""
    from sklearn.linear_model import LogisticRegression

    z = universe["huc12"].isin(infested).to_numpy(float)
    D = np.asarray(drift, float).reshape(-1, 1)
    D = (D - D.mean()) / (D.std() + 1e-9)
    p = LogisticRegression(max_iter=1000).fit(D, z).predict_proba(D)[:, 1] if 0 < z.sum() < len(z) else np.full(len(z), z.mean())
    r = z - p
    sill = float(np.var(r)) or 1e-9
    idx, dnb = knn_index(universe, k)
    lat, lon = universe["lat"].to_numpy(float), universe["lon"].to_numpy(float)
    out = {}
    for a in ranges:
        pred = np.empty(len(z))
        for s in range(0, len(z), chunk):
            ii = idx[s:s + chunk]
            Dnn = _pairwise_km(lat[ii], lon[ii])
            n = len(ii)
            A = np.zeros((n, k + 1, k + 1))
            A[:, :k, :k] = sill * np.exp(-Dnn / a) + nugget * sill * np.eye(k)[None]
            A[:, :k, k] = 1.0
            A[:, k, :k] = 1.0
            b = np.ones((n, k + 1))
            b[:, :k] = sill * np.exp(-dnb[s:s + chunk] / a)
            w = np.linalg.solve(A, b[..., None])[..., 0][:, :k]
            pred[s:s + chunk] = (w * r[ii]).sum(1)
        out[a] = p + pred
    return out
