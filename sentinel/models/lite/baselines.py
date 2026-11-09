# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""SENTINEL-Lite baselines (protocol §5): location/season-only, RGBN band-ratio regressions, GBM on RGBN
statistics (± context). Each ``fit_predict_*`` fits on the given train frame only and predicts log10 targets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...eval.splits import haversine_km

GBM_PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=20, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.8, n_jobs=6, verbose=-1)

# Pre-registered band-ratio forms (feature names from features.chip_features; 'c9s_*' = centre 9x9 median DN / DOQQ
# band median; 'ws_*' = median over dark-NIR water pixels in the ~30 m window / DOQQ band median). Several forms per
# target; the strongest on val is the H1 comparator (selection in the baseline's favour; review W3).
def _lg(x):
    return np.log10(np.asarray(x, float))


BAND_RATIO_FORMS = {
    "secchi": {
        "kloiber_linear": lambda F: np.column_stack([F["c9s_B"] / F["c9s_R"], F["c9s_B"]]),        # Kloiber 2002
        "kloiber_log": lambda F: np.column_stack([_lg(F["c9s_B"] / F["c9s_R"]), _lg(F["c9s_B"])]),
        "kloiber_log_water": lambda F: np.column_stack([_lg(F["ws_B"] / F["ws_R"]), _lg(F["ws_B"])]),
    },
    "turbidity": {
        "red_nir_log": lambda F: np.column_stack([_lg(F["c9s_R"]), _lg(F["c9s_NIR"])]),             # Nechad/Dogliotti
        "red_nir_log_water": lambda F: np.column_stack([_lg(F["ws_R"]), _lg(F["ws_NIR"])]),
        "red_green_log_water": lambda F: np.column_stack([_lg(F["ws_R"]), _lg(F["ws_R"] / F["ws_G"])]),
    },
    "chla": {
        "oc_ndvi": lambda F: np.column_stack([_lg(F["c9s_B"] / F["c9s_G"]),                         # OC-type + NDVI
                                              (F["c9s_NIR"] - F["c9s_R"]) / (F["c9s_NIR"] + F["c9s_R"])]),
        "oc_ndvi_water": lambda F: np.column_stack([_lg(F["ws_B"] / F["ws_G"]),
                                                    (F["ws_NIR"] - F["ws_R"]) / (F["ws_NIR"] + F["ws_R"])]),
    },
}
BAND_RATIO_FORMS["tss"] = dict(BAND_RATIO_FORMS["turbidity"])
GBM_SEEDS = (0, 1, 2)


def _ols_fit_predict(Xtr, ytr, Xte):
    A = np.column_stack([np.ones(len(Xtr)), Xtr])
    coef, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    return np.column_stack([np.ones(len(Xte)), Xte]) @ coef, coef


def band_ratio(Ftr: pd.DataFrame, ytr, Fte: pd.DataFrame, target: str, form: str | None = None):
    form = form or next(iter(BAND_RATIO_FORMS[target]))
    fn = BAND_RATIO_FORMS[target][form]
    Xtr, Xte = fn(Ftr), fn(Fte)
    ok = np.isfinite(Xtr).all(1) & np.isfinite(ytr)
    med = np.median(Xtr[ok], axis=0)
    n_imputed = int((~np.isfinite(Xte)).any(1).sum())
    Xte = np.where(np.isfinite(Xte), Xte, med[None, :])          # train-median imputation (counted)
    pred, coef = _ols_fit_predict(Xtr[ok], ytr[ok], Xte)
    return pred, {"form": form, "coef": coef.tolist(), "n_fit": int(ok.sum()), "n_imputed": n_imputed}


def gbm(Ftr: pd.DataFrame, ytr, Fte: pd.DataFrame, seeds=GBM_SEEDS, cols=None):
    """LightGBM averaged over ``seeds`` (same seed count as the student ensemble; review observation)."""
    import lightgbm as lgb

    cols = list(cols or Ftr.columns)
    ok = np.isfinite(ytr)
    preds = []
    for sd in seeds:
        m = lgb.LGBMRegressor(random_state=sd, **GBM_PARAMS)
        m.fit(Ftr.loc[ok, cols].to_numpy(float), ytr[ok])
        preds.append(m.predict(Fte[cols].to_numpy(float)))
    return np.mean(preds, 0), {"n_fit": int(ok.sum()), "n_features": len(cols), "seeds": list(seeds)}


def location_knn(lat_tr, lon_tr, site_tr, ytr, lat_te, lon_te, k: int = 10):
    """Median of the labels of the k nearest train *sites* (site-level medians first)."""
    d = pd.DataFrame({"site": site_tr, "lat": lat_tr, "lon": lon_tr, "y": ytr}).dropna(subset=["y"])
    s = d.groupby("site").agg(lat=("lat", "first"), lon=("lon", "first"), y=("y", "median"))
    slat, slon, sy = s["lat"].to_numpy(), s["lon"].to_numpy(), s["y"].to_numpy()
    out = np.empty(len(lat_te))
    for i, (a, b) in enumerate(zip(lat_te, lon_te)):
        dist = haversine_km(a, b, slat, slon)
        out[i] = np.median(sy[np.argsort(dist)[:k]])
    return out, {"k": k, "n_sites": int(len(s))}


def location_season_gbm(Ctr: pd.DataFrame, ytr, Cte: pd.DataFrame):
    return gbm(Ctr[["lat", "lon", "doy"]], ytr, Cte[["lat", "lon", "doy"]])
