# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""D8 RiverPulse — daily GPP / ER at reaches without sensors (protocol:
docs/preregistration/downstream_riverpulse.md).

Labels: Powell Center daily metabolism (Appling et al. 2018), filtered to model confidence H/M and
day-level plausibility (GPP >= -0.5, ER <= 0.5); H-only is a sensitivity analysis.

Circularity control: GPP/ER were inverted from each site's own DO, water temperature, depth/discharge
and modelled light, so **none** of the label table's daily predictor columns may enter features
(:data:`FORBIDDEN`, enforced by :func:`assert_no_forbidden`). Allowed inputs: observed gridMET weather
at the gauge (same-day allowed — this is an estimation task at the label day, not a forecast),
astronomical day length, DOY, location/elevation and StreamCat catchment attributes. No upstream-model
features in phase 1.

Split: site roles from the FINAL sonde split (docs/preregistration/sonde_split.json; HUC4 basin groups,
flow-connected sites together) so later M1/M6 features at val/test sites can be cross-fit.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from ...core import paths
from ...eval import splits as ES
from . import common as C
from . import crossfit
from . import weather as W

APP = "riverpulse"
SONDE_SPLIT = paths.REPO_ROOT / "docs" / "preregistration" / "sonde_split.json"
SONDE_SPLIT_SHA = "fd4bb14fe7304d05310e286e9aced22675fd6d07aa4dcac6415ed039a1e06a80"
FALLBACK_SEED = 20261024
MAX_WX_KM = 25.0  # weather from a gridMET cell farther than this is treated as missing
FALLBACK_FRACS = {"test": 0.2, "val": 0.15}
DATA_MANIFESTS = ("labels_powell_metabolism", "forcings_daily", "streamcat", "sites", "downstream_catchment")
SPLIT_VERSION = "v2"
SPLIT_FILE = "downstream_riverpulse_split_v2.json"
SPLIT_SHA = "20c550f9ab42ac343fdbdda1bd2eb845ad7e2081a06904037bbd19881b66b0d8"  # v2, frozen 2026-10-24 (protocol §3)
SUPERSEDED_SPLIT_SHA = "4e4d064e639f4f746835586d2e1464604ae54692ff473b1cef8d84f9a52cd486"
# review W6 (2026-10-24, before any fit): HUC4 1307 is directly downstream of HUC4 1306 (sonde train sites);
# Powell sites in these HUC4s are kept with the upstream group in the train role.
FLOW_OVERRIDE_TRAIN_HUC4 = ("1307",)
# Model-derived StreamCat columns (predicted channel dimensions, Doyle et al. 2023; integrity indices built
# from other models) are excluded (review C4; cross-fitting).
STREAMCAT_EXCLUDE = ("iwi", "ici", "bankfulldepth", "bankfullwidth", "wettedwidth", "thalwegdepth")
FORBIDDEN = ("do_obs_mgl", "do_sat_mgl", "do_amp_mgl", "do_psat_pct", "depth_m", "temp_water_c", "day_length_h",
             "discharge_m3s", "shortwave_wm2", "velocity_ms", "do_tdist80_m", "k600", "k600_lower", "k600_upper",
             "gpp_lower", "gpp_upper", "er_lower", "er_upper", "gpp_n_eff", "er_n_eff", "gpp_rhat", "er_rhat",
             "hydgeom_c", "hydgeom_f", "hydgeom_a", "hydgeom_b", "hydgeom_k", "hydgeom_m")
TARGETS = ("gpp", "er")


def assert_no_forbidden(cols) -> None:
    bad = [c for c in cols if c in FORBIDDEN or c.startswith(("hydgeom_", "do_", "k600"))]
    if bad:
        raise AssertionError(f"RiverPulse circularity: label-model input columns used as features: {bad}")


# --------------------------------------------------------------------------- data
def load_labels(confidence=("H", "M"), plausible: bool = True) -> pd.DataFrame:
    P = paths.PROCESSED / "labels_powell_metabolism"
    d = pd.read_parquet(P / "daily_metabolism.parquet",
                        columns=["site_id", "nwis_id", "date", "gpp", "er", "model_confidence",
                                 "flag_gpp_negative", "flag_er_positive"])
    d = d[d["model_confidence"].astype(str).isin(confidence)]
    if plausible:
        d = d[(d["gpp"] >= -0.5) & (d["er"] <= 0.5)]
    d["date"] = pd.to_datetime(d["date"])
    return d.drop(columns=["flag_gpp_negative", "flag_er_positive"]).reset_index(drop=True)


def load_sites() -> pd.DataFrame:
    s = pd.read_parquet(paths.PROCESSED / "labels_powell_metabolism" / "sites.parquet")
    s = s[s["modeled"].astype(bool)].copy()
    s["nwis_id"] = s["nwis_id"].astype(str)
    s["site_key"] = "USGS-" + s["nwis_id"]
    return s.reset_index(drop=True)


def _huc4_lookup(sites: pd.DataFrame) -> pd.Series:
    """HUC4 per Powell site: registry `huc4`, else first 4 digits of the NHDPlus V2 reachcode of the
    published COMID (VAA)."""
    reg = pd.read_parquet(paths.PROCESSED / "sites" / "sites.parquet", columns=["site_key", "huc4"])
    h = sites["site_key"].map(reg.set_index("site_key")["huc4"])
    miss = h.isna() & sites["comid"].notna()
    if miss.any():
        vaa = pd.read_parquet(paths.RAW / "nhdplus_vaa" / "nhdplus_vaa_subset.parquet", columns=["comid", "reachcode"])
        vaa["comid"] = vaa["comid"].astype("int64").astype(str)
        rc = sites.loc[miss, "comid"].astype(str).str.replace(r"\.0$", "", regex=True).map(vaa.set_index("comid")["reachcode"])
        h.loc[miss] = rc.str[:4]
    return h.astype("string")


def site_roles(sites: pd.DataFrame) -> tuple[dict, dict]:
    """Powell site_id -> role, and provenance per site (sonde split / basin group / fallback)."""
    text = SONDE_SPLIT.read_text()
    if hashlib.sha256(SONDE_SPLIT.read_bytes()).hexdigest() != SONDE_SPLIT_SHA:
        raise C.ProtectedDataError("sonde_split.json sha256 changed since this protocol was written")
    sp = json.loads(text)
    role, bg = sp["site_role"], sp["basin_group"]
    reg = pd.read_parquet(paths.PROCESSED / "sites" / "sites.parquet", columns=["site_id", "huc4"])
    reg = reg[reg["site_id"].isin(role)]
    reg["bg"] = reg["site_id"].map(bg)
    reg["role"] = reg["site_id"].map(role)
    h2r = reg.groupby("huc4")["role"].agg(lambda x: x.iloc[0] if x.nunique() == 1 else None).dropna().to_dict()
    huc4 = _huc4_lookup(sites)
    out, prov, pending = {}, {}, []
    for sid, nwis, h in zip(sites["site_id"], sites["nwis_id"], huc4):
        if nwis in role:
            out[sid], prov[sid] = role[nwis], "sonde_split_site"
        elif h is not pd.NA and h is not None and h in h2r:
            out[sid], prov[sid] = h2r[h], f"sonde_split_huc4:{h}"
        else:
            pending.append((sid, h if isinstance(h, str) else f"nohuc:{sid}"))
    for sid, g in pending:
        # deterministic per-HUC4 fallback (works for any number of groups): sha1(seed|huc4) mod 100
        u = int(hashlib.sha1(f"{FALLBACK_SEED}|{g}".encode()).hexdigest(), 16) % 100
        r = "test" if u < 100 * FALLBACK_FRACS["test"] else (
            "val" if u < 100 * (FALLBACK_FRACS["test"] + FALLBACK_FRACS["val"]) else "train")
        out[sid], prov[sid] = r, f"fallback_huc4_hash:{g}"
    for sid, h in zip(sites["site_id"], huc4):
        if isinstance(h, str) and h in FLOW_OVERRIDE_TRAIN_HUC4:
            out[sid], prov[sid] = "train", f"flow_override_train_huc4:{h}"
    return out, prov


def split_meta(prov: dict) -> dict:
    from collections import Counter
    return {"unit": "site (all days of a site share its role)", "source_split": "docs/preregistration/sonde_split.json",
            "source_split_sha256": SONDE_SPLIT_SHA, "fallback_rule": "role by sha1(f\"{seed}|{huc4}\") mod 100: <20 test, <35 val, else train", "fallback_seed": FALLBACK_SEED, "fallback_fracs": FALLBACK_FRACS,
            "split_version": SPLIT_VERSION, "supersedes_sha256": SUPERSEDED_SPLIT_SHA,
            "flow_override_train_huc4": FLOW_OVERRIDE_TRAIN_HUC4,
            "provenance_counts": dict(Counter(v.split(":")[0] for v in prov.values())), "provenance": prov,
            "label_filter": "model_confidence in {H,M}, gpp >= -0.5, er <= 0.5",
            "vintage": "labels_powell_metabolism 2026-10-22 (doi:10.5066/F70864KX)"}


# --------------------------------------------------------------------------- features
WX_KEEP = ("srad_mean_1d", "srad_mean_3d", "srad_mean_7d", "tmax_mean_1d", "tmax_mean_7d", "tmax_mean_30d",
           "tmin_mean_1d", "tmin_mean_7d", "precip_sum_1d", "precip_sum_3d", "precip_sum_7d", "precip_sum_14d",
           "precip_sum_30d", "vpd_mean_3d", "wind_mean_3d", "dry_days_14d")


def day_length_h(lat_deg, doy) -> np.ndarray:
    """Astronomical day length (Forsythe et al. 1995 CBM model, p = 0.833°)."""
    lat = np.radians(np.asarray(lat_deg, float))
    theta = 0.2163108 + 2 * np.arctan(0.9671396 * np.tan(0.00860 * (np.asarray(doy, float) - 186)))
    phi = np.arcsin(0.39795 * np.cos(theta))
    p = np.radians(0.8333)
    x = (np.sin(p) + np.sin(lat) * np.sin(phi)) / (np.cos(lat) * np.cos(phi))
    return 24 - (24 / np.pi) * np.arccos(np.clip(x, -1, 1))


def catchment_ready() -> bool:
    return (paths.MANIFESTS / "downstream_catchment.json").exists() and \
        (paths.PROCESSED / "downstream_catchment" / "streamcat_comid.parquet").exists()


def streamcat_for_sites(sites: pd.DataFrame) -> pd.DataFrame:
    """StreamCat attributes per Powell site_id: `processed/downstream_catchment` (Powell published COMID or
    NLDI snap; primary for every site), else the registry table (gauge's NLDI COMID). Model-derived columns
    (:data:`STREAMCAT_EXCLUDE`) are dropped from both."""
    sc = pd.read_parquet(paths.PROCESSED / "catchment" / "streamcat_sites.parquet")
    drop = {"comid", "comid_method", "streamcat_found", "site_key", *STREAMCAT_EXCLUDE}
    num = [c for c in sc.columns if c not in drop and pd.api.types.is_numeric_dtype(sc[c])]
    a = sites[["site_id", "site_key"]].copy()
    a["sc_source"] = None
    for c in num:
        a[c] = np.nan
    if catchment_ready():
        links = pd.read_parquet(paths.PROCESSED / "downstream_catchment" / "unit_links.parquet")
        links = links[links["unit_source"] == "powell"][["unit_id", "comid"]].rename(columns={"unit_id": "site_id"})
        dc = pd.read_parquet(paths.PROCESSED / "downstream_catchment" / "streamcat_comid.parquet")
        dc["comid"] = dc["comid"].astype(str)
        links["comid"] = links["comid"].astype("string").astype(str)
        f = a[["site_id"]].merge(links, on="site_id", how="left").merge(dc, on="comid", how="left")
        f.index = a.index
        use = [c for c in num if c in f]
        ok = f[use].notna().any(axis=1)
        a.loc[ok, use] = f.loc[ok, use]
        a.loc[ok, "sc_source"] = "downstream_catchment"
    reg = a[["site_key"]].merge(sc[["site_key", *num]], on="site_key", how="left")
    reg.index = a.index
    miss = a["sc_source"].isna() & reg[num].notna().any(axis=1)
    a.loc[miss, num] = reg.loc[miss, num]
    a.loc[miss, "sc_source"] = "registry"
    return a.drop(columns=["site_key"])


def weather_for_rows(rows: pd.DataFrame, sites: pd.DataFrame) -> pd.DataFrame:
    """Same-day-inclusive observed weather (lag0) at the gauge cell (registry site_key) or nearest cell."""
    cov = pd.read_parquet(paths.PROCESSED / "forcings_daily" / "coverage.parquet", columns=["site_key", "source"])
    have = set(cov.loc[cov["source"] == "gridmet", "site_key"])
    st = sites.set_index("site_id")
    key = st["site_key"].where(st["site_key"].isin(have))
    miss = key.isna()
    km = pd.Series(0.0, index=st.index)
    if miss.any():
        nc = W.nearest_cells(st.loc[miss, "latitude"], st.loc[miss, "longitude"])
        key.loc[miss] = nc["wx_site_key"].to_numpy()
        km.loc[miss] = nc["wx_cell_km"].to_numpy()
    q = pd.DataFrame({"wx_site_key": rows["site_id"].map(key), "date": rows["date"]}, index=rows.index)
    series = W.load_series(q["wx_site_key"].dropna().unique(), start="2007-01-01", end="2017-01-31")
    wx = W.antecedent_features(q, series, lag0=True)
    wx = wx[[f"wx_{c}" for c in WX_KEEP if f"wx_{c}" in wx]]
    wx["wx_cell_km"] = rows["site_id"].map(km).to_numpy()
    far = wx["wx_cell_km"].to_numpy() > MAX_WX_KM
    wx.loc[far, [c for c in wx.columns if c != "wx_cell_km"]] = np.nan  # no representative gridMET cell
    return wx


def build_features(rows: pd.DataFrame, sites: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    st = sites.set_index("site_id")
    X = pd.DataFrame(index=rows.index)
    doy = rows["date"].dt.dayofyear.to_numpy(float)
    X["doy_sin"], X["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    lat = rows["site_id"].map(st["latitude"]).to_numpy(float)
    X["lat"], X["lon"] = lat, rows["site_id"].map(st["longitude"]).to_numpy(float)
    X["elev_m"] = rows["site_id"].map(st["altitude_ft"].astype(float) * 0.3048).to_numpy(float)
    X["daylen_h"] = day_length_h(lat, doy)
    X = X.join(weather_for_rows(rows, sites))
    sc = streamcat_for_sites(sites).set_index("site_id")
    scn = [c for c in sc.columns if c != "sc_source"]
    scf = sc.reindex(rows["site_id"].to_numpy())[scn]
    scf.index = rows.index
    X = X.join(scf.add_prefix("sc_"))
    cols = [c for c in X.columns if c != "wx_cell_km"]
    assert_no_forbidden(cols)
    crossfit.phase1_no_upstream(cols)
    return X, cols


# --------------------------------------------------------------------------- baselines (fit on train only)
def doy_climatology_fit(train: pd.DataFrame, target: str, window: int = 15) -> pd.Series:
    doy = np.minimum(train["date"].dt.dayofyear, 365)
    m = train.groupby(doy)[target].mean().reindex(range(1, 366))
    ext = pd.concat([m.iloc[-window:], m, m.iloc[:window]])
    return ext.rolling(2 * window + 1, center=True, min_periods=window).mean().iloc[window:-window]


def predict_doy(clim: pd.Series, rows: pd.DataFrame) -> np.ndarray:
    return clim.reindex(np.minimum(rows["date"].dt.dayofyear, 365)).to_numpy(float)


def location_knn(train: pd.DataFrame, sites: pd.DataFrame, rows: pd.DataFrame, target: str, k: int = 5) -> np.ndarray:
    """Mean DOY climatology (±15 d smoothing) of the k nearest TRAIN sites (location-only baseline)."""
    st = sites.set_index("site_id")
    tr_sites = sorted(train["site_id"].unique())
    clims = {s: doy_climatology_fit(train[train["site_id"] == s], target) for s in tr_sites}
    glob = doy_climatology_fit(train, target)
    tl, tn = st.loc[tr_sites, "latitude"].to_numpy(float), st.loc[tr_sites, "longitude"].to_numpy(float)
    out = np.full(len(rows), np.nan)
    doy = np.minimum(rows["date"].dt.dayofyear.to_numpy(), 365)
    for s, idx in rows.groupby("site_id").indices.items():
        d = ES.haversine_km(st.at[s, "latitude"], st.at[s, "longitude"], tl, tn)
        nn = [tr_sites[i] for i in np.argsort(d) if tr_sites[i] != s][:k]
        mat = np.vstack([clims[n].to_numpy() for n in nn])
        mean = np.nanmean(mat, axis=0)
        mean = np.where(np.isfinite(mean), mean, glob.to_numpy())
        out[idx] = mean[doy[idx] - 1]
    return out


LIGHT_COLS = ("wx_srad_mean_1d", "wx_srad_mean_7d", "daylen_h", "doy_sin", "doy_cos")


def fit_light_season(Xtr: pd.DataFrame, ytr, alpha: float = 1.0):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), PolynomialFeatures(2, include_bias=False),
                      Ridge(alpha=alpha))
    m.fit(Xtr[list(LIGHT_COLS)].to_numpy(float), np.asarray(ytr, float))
    return m


GBM_PARAMS = {"objective": "regression", "learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 200,
              "subsample": 0.7, "subsample_freq": 1, "colsample_bytree": 0.7, "reg_lambda": 1.0,
              "n_estimators": 3000, "num_threads": C.N_THREADS, "verbose": -1}


INNER_SEED = 0
GRIDS = {"light_season": [{"alpha": a} for a in (0.1, 1.0, 10.0, 100.0)],
         "rf_segatto_incumbent": [{"min_samples_leaf": l, "max_features": f} for l in (5, 20) for f in (0.33, 0.6)],
         "gbm_challenger": [{"num_leaves": nl, "min_child_samples": mc} for nl in (15, 31) for mc in (50, 200)]}
GRIDS["rf_maavara_incumbent"] = [{"min_samples_leaf": l, "max_features": f} for l in (5, 20) for f in (0.33, 0.6)]
# Maavara et al. 2025 (Science 390:622-627, doi:10.1126/science.adu9843): scikit-learn RandomForestRegressor
# on Appling et al. 2018 metabolism with 25 reach/catchment features (candidate classes stated on the Dryad
# page: width, slope, upstream area, temperature, precipitation, land-cover proportions). Their exact list
# and their modelled-discharge/width inputs are not available here (Dryad files need a login; width and
# discharge are excluded by our circularity / model-derived rules), so this is the closest public analogue:
# 25 features of the same classes.
MAAVARA_FEATURES = tuple(f"sc_{c}" for c in (
    "pcturbop2019ws", "pcturblo2019ws", "pcturbmd2019ws", "pcturbhi2019ws", "pctbl2019ws", "pctdecid2019ws",
    "pctconif2019ws", "pctmxfst2019ws", "pctshrb2019ws", "pctgrs2019ws", "pcthay2019ws", "pctcrop2019ws",
    "pctwdwet2019ws", "pcthbwet2019ws", "pctimp2019ws", "nhd_slope", "nhd_totdasqkm", "nhd_streamorde",
    "elevws", "precip9120ws", "tmean9120ws")) + ("wx_tmax_mean_1d", "wx_precip_sum_7d", "doy_sin", "doy_cos")
RF_SELECT_TREES = 100  # trees per RF during inner selection (final fit: 300)


def inner_groups(groups_tr) -> set:
    """The single inner hold-out (20 % of TRAIN basin groups, seed INNER_SEED) shared by every model's
    selection (tuning parity)."""
    ug = np.array(sorted(set(map(str, groups_tr))))
    rs = np.random.default_rng(INNER_SEED)
    return set(rs.choice(ug, max(1, int(0.2 * len(ug))), replace=False))


def fit_gbm(Xtr: pd.DataFrame, ytr, cols, groups_tr, seed: int = 0, params: dict | None = None):
    """LightGBM; n_estimators by early stopping on an inner GROUP split of TRAIN sites (20% of train
    basin groups), then refit on all train rows. Validation sites are never used for selection."""
    import lightgbm as lgb

    inner = np.isin(groups_tr, list(inner_groups(groups_tr)))
    p = dict(GBM_PARAMS, **(params or {}), random_state=seed)
    m0 = lgb.LGBMRegressor(**p)
    m0.fit(Xtr.loc[~inner, cols], np.asarray(ytr)[~inner], eval_set=[(Xtr.loc[inner, cols], np.asarray(ytr)[inner])],
           callbacks=[lgb.early_stopping(100, verbose=False)])
    best = int(m0.best_iteration_ or GBM_PARAMS["n_estimators"])
    m = lgb.LGBMRegressor(**dict(p, n_estimators=max(best, 50)))
    m.fit(Xtr[cols], ytr)
    m.best_inner_iteration_ = best
    m.inner_mse_ = float(m0.best_score_["valid_0"]["l2"])
    return m


def fit_rf(Xtr: pd.DataFrame, ytr, cols, seed: int = 0, params: dict | None = None):
    """Segatto-style random forest on ALL training rows with native missing-value support (review W4)."""
    from sklearn.ensemble import RandomForestRegressor

    p = {"n_estimators": 300, "min_samples_leaf": 20, "max_features": 0.33, "max_samples": 0.5, **(params or {})}
    m = RandomForestRegressor(n_jobs=C.N_THREADS, random_state=seed, **p)
    m.fit(Xtr[cols].to_numpy(np.float32), np.asarray(ytr, float))
    return m


def site_year_means(rows: pd.DataFrame, pred: np.ndarray, target: str, min_days: int = 180) -> pd.DataFrame:
    d = rows[["site_id", "date", target]].copy()
    d["pred"] = pred
    d["year"] = d["date"].dt.year
    g = d.groupby(["site_id", "year"]).agg(n=(target, "size"), obs=(target, "mean"), pred=("pred", "mean")).reset_index()
    return g[g["n"] >= min_days]
