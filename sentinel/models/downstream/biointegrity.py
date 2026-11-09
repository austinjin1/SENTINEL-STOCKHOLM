# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""D5 BioIntegrity — NRSA benthic-macroinvertebrate condition (Good vs Poor) at reaches without a
biological sample (RESEARCH_PLAN §8 D5; docs/research/07 §d.5; protocol
docs/preregistration/downstream_biointegrity.md).

Units are NRSA first visits (``visit_no == '1'``) with a benthic MMI class. Location clusters (1-km
connected components over all four NRSA cycles, from ``sentinel.models.bioassess.data``) define
"new reach" vs "revisited reach" and are never split across folds in the development CV.

Feature sets (observed covariates only; no upstream-model outputs in phase 1):
``geo`` lat/lon/elevation/ecoregion · ``streamcat`` curated StreamCat cat/ws/riparian attributes +
NHDPlus VAA (via ``processed/downstream_catchment``) · ``wx`` antecedent gridMET weather before the
visit (nearest SENTINEL cell) · ``chem`` / ``habitat`` co-measured NRSA chemistry / physical habitat
(**oracle rows**: measured at the same visit; upper bounds, never the deployable model).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...core import paths
from . import common as C
from . import weather as W

APP = "biointegrity"
TARGET = "bent_mmi_cond"
TRAIN_SURVEYS = ("nrsa0809", "nrsa1314")
VAL_SURVEYS = ("nrsa1819",)
TEST_SURVEYS = ("nrsa2324",)
SURVEY_ROLE = {**{s: "train" for s in TRAIN_SURVEYS}, **{s: "val" for s in VAL_SURVEYS},
               **{s: "test" for s in TEST_SURVEYS}}
CHEM_COLS = ["tp_ug_l", "tn_mg_l", "nh4_n_mg_l", "no3_n_mg_l", "chloride_mg_l", "sulfate_mg_l", "cond_us_cm",
             "ph_lab_su", "turbidity_ntu", "doc_mg_l", "tss_mg_l", "anc_ueq_l"]
CHEM_LOG = {"tp_ug_l": 1.0, "tn_mg_l": 0.01, "nh4_n_mg_l": 0.001, "no3_n_mg_l": 0.001, "chloride_mg_l": 0.01,
            "sulfate_mg_l": 0.01, "cond_us_cm": 1.0, "turbidity_ntu": 0.01, "doc_mg_l": 0.01, "tss_mg_l": 0.01}
HABITAT_COLS = ["LRBS_USE", "PCT_FN", "PCT_SAFN", "PCT_FAST", "PCT_POOL", "RP100", "SINU", "W1_HALL", "XCDENBK",
                "XCMGW", "XDEPTH_CM", "XFC_NAT", "XSLOPE", "XWIDTH"]
ECO9 = ["CPL", "NAP", "NPL", "SAP", "SPL", "TPL", "UMW", "WMT", "XER"]
WX_MAX_KM = 30.0
SPLIT_SHA = "0d9de79a17b33e06e57703e15b2f4309bcabd0dbc55e8e21f3eb2a9558f1601c"
# Model-derived StreamCat columns excluded (review 2026-10-24 C1): Doyle et al. 2023 predicted channel
# dimensions (RF trained on field data that may include NRSA sites) and the IWI/ICI composite indices
# (weights/calibration possibly involving NRSA condition) — upstream predictions that may have seen
# evaluation sites (RESEARCH_PLAN §8 cross-fitting, R4).
EXCLUDED_STREAMCAT = ("iwi", "ici", "bankfulldepth", "bankfullwidth", "wettedwidth", "thalwegdepth")


# --------------------------------------------------------------------------- units and labels
def load_units() -> pd.DataFrame:
    """First visits with a Good/Fair/Poor benthic class, all four cycles, with location clusters."""
    from ..bioassess.data import all_site_locations  # coordinates-only clustering (read-only reuse)

    P = paths.PROCESSED / "nars_streams"
    v = pd.read_parquet(P / "visits.parquet")
    cond = pd.read_parquet(P / "condition_wide.parquet", columns=["visit_id", "bent_mmi_cond", "fish_mmi_cond",
                                                                  "bent_mmi"])
    loc = all_site_locations(v)
    v = v.merge(loc[["survey", "site_id", "location_id"]], on=["survey", "site_id"], how="left")
    v = v.merge(cond, on="visit_id", how="left")
    u = v[(v["visit_no"].astype(str) == "1") & v[TARGET].isin(["Good", "Fair", "Poor"])].copy()
    u["role"] = u["survey"].map(SURVEY_ROLE)
    u["y_poor"] = np.where(u[TARGET] == "Poor", 1.0, np.where(u[TARGET] == "Good", 0.0, np.nan))
    u["y_ord"] = u[TARGET].map({"Good": 0, "Fair": 1, "Poor": 2}).astype(float)
    u["unit_id"] = u["survey"].astype(str) + ":" + u["site_id"].astype(str)
    u["date"] = pd.to_datetime(u["date"])
    train_locs = set(u.loc[u["role"] == "train", "location_id"])
    u["new_location"] = ~u["location_id"].isin(train_locs)
    u["huc8"] = u["huc8"].astype("string")
    u["huc4"] = u["huc8"].str[:4]
    if u["visit_id"].duplicated().any():
        raise AssertionError("duplicate visit ids")
    return u.reset_index(drop=True)


def split_assignments(units: pd.DataFrame) -> tuple[dict, dict]:
    """Frozen split content: visit → role, visit → location cluster, visit → new-location flag."""
    assignments = {
        "visit_role": dict(zip(units["visit_id"], units["role"])),
        "visit_location": dict(zip(units["visit_id"], units["location_id"])),
        "val_test_new_location": {k: bool(f) for k, f, r in zip(units["visit_id"], units["new_location"], units["role"])
                                  if r in ("val", "test")},
    }
    meta = {"target": TARGET, "unit": "NRSA first visit (visit_no == '1') with Good/Fair/Poor benthic MMI class",
            "survey_role": SURVEY_ROLE, "location_cluster": "1.0-km connected components over all NRSA cycles "
            "(sentinel.models.bioassess.data.all_site_locations)",
            "new_location": "location cluster has no train-cycle (2008-09/2013-14) first visit",
            "primary_binary": "Poor (1) vs Good (0); Fair excluded from the binary task"}
    return assignments, meta


# --------------------------------------------------------------------------- features
def geo_features(u: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=u.index)
    f["lat"] = u["lat"].astype(float)
    f["lon"] = u["lon"].astype(float)
    f["elev_m"] = pd.to_numeric(u["elevation_m"], errors="coerce")
    for e in ECO9:
        f[f"eco_{e}"] = (u["ag_eco9"] == e).astype(float)
    return f


def streamcat_features(u: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    D = paths.PROCESSED / "downstream_catchment"
    links = pd.read_parquet(D / "unit_links.parquet")
    links = links[links["unit_source"] == "nrsa"][["unit_id", "comid", "comid_method"]]
    sc = pd.read_parquet(D / "streamcat_comid.parquet")
    drop = [c for c in sc.columns if c in ("streamcat_found",)]
    m = u[["unit_id"]].merge(links, on="unit_id", how="left").merge(sc.drop(columns=drop), on="comid", how="left")
    m.index = u.index
    cols = [c for c in sc.columns if c not in ("comid", "streamcat_found")
            and not any(c.startswith(x) for x in EXCLUDED_STREAMCAT)]
    f = m[cols].apply(pd.to_numeric, errors="coerce").astype(float)
    # National Nutrient Inventory totals -> intensities (datasheet: kg/yr in the AOI)
    for c in list(f.columns):
        if (c.startswith(("n_", "p_")) or c.startswith(("nani", "nsurp"))) and c.endswith(("cat", "ws")):
            area = f["catareasqkm"] if c.endswith("cat") else f["wsareasqkm"]
            f[c] = f[c] / area.replace(0, np.nan)
    for c in ("wsareasqkm", "catareasqkm", "nhd_totdasqkm"):
        if c in f:
            f[f"log_{c}"] = np.log10(f[c].clip(lower=1e-3))
    info = {"n_cols": int(f.shape[1]), "frac_with_comid": float(m["comid"].notna().mean()),
            "frac_found": float(f.notna().any(axis=1).mean()),
            "comid_method": m["comid_method"].value_counts(dropna=False).to_dict()}
    return f, info


def wx_features(u: pd.DataFrame, clim_years) -> tuple[pd.DataFrame, dict]:
    nc = W.nearest_cells(u["lat"], u["lon"])
    nc.index = u.index
    q = pd.DataFrame({"wx_site_key": nc["wx_site_key"], "date": u["date"]}, index=u.index)
    s = W.load_series(nc["wx_site_key"].unique(), start="2000-01-01")
    clim = W.doy_climatology(s, clim_years)
    f = W.antecedent_features(q, s, clim=clim, windows=(7, 30, 90))
    keep = [c for c in f.columns if any(k in c for k in ("precip_sum", "tmax_mean", "tmax_anom", "precip_anom"))]
    f = f[keep]
    far = nc["wx_cell_km"] > WX_MAX_KM
    f.loc[far, :] = np.nan     # distance itself is not a feature (proxy for "near a USGS site")
    return f, {"median_cell_km": float(nc["wx_cell_km"].median()), "frac_far": float(far.mean())}


def chem_features(u: pd.DataFrame) -> pd.DataFrame:
    c = pd.read_parquet(paths.PROCESSED / "nars_streams" / "chemistry_wide.parquet", columns=["visit_id", *CHEM_COLS])
    m = u[["visit_id"]].merge(c, on="visit_id", how="left")
    m.index = u.index
    f = pd.DataFrame(index=u.index)
    for col in CHEM_COLS:
        x = pd.to_numeric(m[col], errors="coerce")
        f[f"chem_{col}"] = np.log10(x.clip(lower=CHEM_LOG[col])) if col in CHEM_LOG else x
    return f


def habitat_features(u: pd.DataFrame) -> pd.DataFrame:
    h = pd.read_parquet(paths.PROCESSED / "nars_streams" / "phab_key_wide.parquet")
    m = u[["visit_id"]].merge(h, on="visit_id", how="left")
    m.index = u.index
    return m[[c for c in HABITAT_COLS if c in m]].apply(pd.to_numeric, errors="coerce").add_prefix("hab_")


FEATURE_SETS = {
    "geo": ("geo",),
    "streamcat": ("streamcat",),
    "geo_streamcat": ("geo", "streamcat"),
    "geo_streamcat_wx": ("geo", "streamcat", "wx"),
    "geo_streamcat_nbr": ("geo", "streamcat", "nbr"),
    "geo_streamcat_wx_nbr": ("geo", "streamcat", "wx", "nbr"),
    "oracle_chem": ("geo", "streamcat", "chem"),
    "oracle_chem_habitat": ("geo", "streamcat", "chem", "habitat"),
}


def build_features(u: pd.DataFrame, clim_years) -> tuple[dict, dict]:
    if "role" in u and (u["role"] == "test").any():   # only inside the single authorised test look
        from .testlook import require_authorised

        require_authorised(APP, SPLIT_SHA)
    blocks, info = {"geo": geo_features(u)}, {}
    blocks["streamcat"], info["streamcat"] = streamcat_features(u)
    blocks["wx"], info["wx"] = wx_features(u, clim_years)
    blocks["chem"] = chem_features(u)
    blocks["habitat"] = habitat_features(u)
    return blocks, info


def neighbour_block(u: pd.DataFrame, train: pd.DataFrame, ks=(5, 15, 50)) -> pd.DataFrame:
    """Spatial-context features: Poor share among the k nearest *training-cycle* reaches (other location
    clusters only), for every row of ``u`` (training rows see only other training locations)."""
    f = pd.DataFrame(index=u.index)
    for k in ks:
        f[f"nbr_poor_k{k}"] = knn_rate(train, u, k=k)
    return f


def assemble(blocks: dict, names) -> pd.DataFrame:
    return pd.concat([blocks[n] for n in names], axis=1)


# --------------------------------------------------------------------------- baselines
def ecoregion_rate(train: pd.DataFrame, other: pd.DataFrame, y: str = "y_poor") -> np.ndarray:
    tr = train[train[y].notna()]
    rate = tr.groupby("ag_eco9")[y].mean()
    return other["ag_eco9"].map(rate).fillna(tr[y].mean()).to_numpy(float)


def knn_rate(train: pd.DataFrame, other: pd.DataFrame, k: int = 15, y: str = "y_poor",
             exclude_same_location: bool = True) -> np.ndarray:
    """Location-only baseline: Poor share among the k nearest training reaches (great-circle)."""
    from sklearn.neighbors import BallTree

    tr = train[train[y].notna()]
    tree = BallTree(np.radians(tr[["lat", "lon"]].to_numpy(float)), metric="haversine")
    kk = min(k + 5, len(tr))
    dist, idx = tree.query(np.radians(other[["lat", "lon"]].to_numpy(float)), k=kk)
    ytr = tr[y].to_numpy(float)
    loc_tr = tr["location_id"].to_numpy()
    out = np.empty(len(other))
    for i, (row, loc) in enumerate(zip(idx, other["location_id"].to_numpy())):
        cand = [j for j in row if not (exclude_same_location and loc_tr[j] == loc)][:k]
        out[i] = ytr[cand].mean() if cand else np.nan
    return out


def persistence(train: pd.DataFrame, other: pd.DataFrame, y: str = "y_ord") -> np.ndarray:
    """Most recent train-cycle class at the same location cluster (ordinal 0/1/2; NaN if none)."""
    tr = train[train[y].notna()].sort_values("date")
    last = tr.groupby("location_id")[y].last()
    return other["location_id"].map(last).to_numpy(float)


# --------------------------------------------------------------------------- models
def make_model(kind: str, seed: int = 0, n_jobs: int = C.N_THREADS, params: dict | None = None):
    if kind == "logistic":
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=1.0, max_iter=2000))
    if kind == "rf":  # Hill et al. 2017 style: random forest, 500 trees, default mtry (sqrt)
        from sklearn.ensemble import RandomForestClassifier

        p = dict(n_estimators=500, n_jobs=n_jobs, random_state=seed, oob_score=True)
        p.update(params or {})
        return RandomForestClassifier(**p)
    if kind == "gbm":
        import lightgbm as lgb

        p = dict(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=20, subsample=0.8,
                 subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0, n_jobs=n_jobs, random_state=seed, verbose=-1)
        p.update(params or {})
        return lgb.LGBMClassifier(**p)
    if kind == "rf_reg":  # same RF family, trained on the continuous benthic MMI score
        from sklearn.ensemble import RandomForestRegressor

        p = dict(n_estimators=500, max_features="sqrt", n_jobs=n_jobs, random_state=seed)
        p.update(params or {})
        return RandomForestRegressor(**p)
    if kind == "gbm_reg":
        import lightgbm as lgb

        p = dict(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=20, subsample=0.8,
                 subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0, n_jobs=n_jobs, random_state=seed, verbose=-1)
        p.update(params or {})
        return lgb.LGBMRegressor(**p)
    raise ValueError(kind)


REGRESSION_KINDS = ("rf_reg", "gbm_reg")


def fit_predict(kind: str, Xtr: pd.DataFrame, ytr: np.ndarray, Xev: pd.DataFrame, seed: int = 0,
                params: dict | None = None):
    """Fit and return (model, Poor score). Classification kinds: P(Poor). Regression kinds are trained
    on the benthic MMI score (higher = better condition), so the Poor score is ``-prediction``."""
    model = make_model(kind, seed, params=params)
    Xtr = Xtr.astype(float)
    Xev = Xev[Xtr.columns].astype(float)
    if kind in ("rf", "rf_reg"):  # impute with train medians (RF parity with the incumbent)
        med = Xtr.median()
        Xtr, Xev = Xtr.fillna(med).fillna(0.0), Xev.fillna(med).fillna(0.0)
    model.fit(Xtr, ytr)
    if kind in REGRESSION_KINDS:
        return model, -model.predict(Xev)
    return model, model.predict_proba(Xev)[:, 1]
