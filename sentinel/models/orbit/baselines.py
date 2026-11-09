# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""SENTINEL-Orbit baseline ladder (orbit_protocol.md §4.2): global median, location kNN, season+location
GBM, calibrated index algorithms, GBM on band statistics ± context. All fitted on fit rows only;
hyperparameters selected on inner rows (train-role inner water bodies, train years) by MdSA.

Every baseline returns log10 predictions per target. Each ``fit_*`` returns (predict_fn, info) where
``predict_fn(df, X) -> {target: log10 pred}``."""

from __future__ import annotations

import itertools
import os

import numpy as np
import pandas as pd

from .data import TARGETS
from .evaluate import mdsa_log10
from .features import unit_vector

N_THREADS = int(os.environ.get("ORBIT_N_THREADS", "6"))   # shared-host rule: <= 6


def _ok(y):
    return np.isfinite(y)


# ------------------------------------------------------------------------------- trivial context

def fit_global_median(df_fit, y_fit: dict):
    med = {t: float(np.nanmedian(y_fit[t])) for t in TARGETS}

    def predict(df, X=None):
        return {t: np.full(len(df), med[t]) for t in TARGETS}
    return predict, {"median_log10": med}


def fit_location_knn(df_fit, y_fit: dict, df_inner, y_inner: dict, ks=(4, 8, 16, 32)):
    """Inverse-distance kNN over train POINTS (point-level mean of log10 values) on the sphere."""
    from sklearn.neighbors import BallTree

    models, info = {}, {}
    for t in TARGETS:
        m = _ok(y_fit[t])
        pts = pd.DataFrame({"lat": df_fit.lat.to_numpy()[m], "lon": df_fit.lon.to_numpy()[m], "y": y_fit[t][m]})
        pts = pts.groupby(["lat", "lon"], as_index=False).y.mean()
        tree = BallTree(np.radians(pts[["lat", "lon"]].to_numpy()), metric="haversine")
        yv = pts.y.to_numpy()

        def pred_k(df, k, tree=tree, yv=yv):
            d, i = tree.query(np.radians(df[["lat", "lon"]].to_numpy()), k=min(k, len(yv)))
            w = 1.0 / (d * 6371.0 + 0.1)
            return (w * yv[i]).sum(1) / w.sum(1)
        mi = _ok(y_inner[t])
        scores = {k: mdsa_log10(pred_k(df_inner[mi], k) - y_inner[t][mi]) for k in ks}
        kbest = min(scores, key=scores.get)
        models[t] = (pred_k, kbest)
        info[t] = {"k": kbest, "inner_mdsa": scores, "n_points": int(len(pts))}

    def predict(df, X=None, k=None):
        """k=None: inner-selected k per target; else the given k (variant predictions)."""
        return {t: models[t][0](df, models[t][1] if k is None else k) for t in TARGETS}
    predict.variants = tuple(ks)
    return predict, info


# ------------------------------------------------------------------------------------- GBM family

GBM_GRID = [dict(num_leaves=nl, min_child_samples=mc, objective=obj, learning_rate=lr, feature_fraction=0.8,
                 bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0)
            for nl, mc, obj, lr in itertools.product((15, 63), (20, 60), ("l2", "l1"), (0.03, 0.1))]
GBM_FINAL_SEEDS = (0, 1, 2, 3, 4)


def fit_gbm(X_fit: pd.DataFrame, y_fit: dict, X_inner: pd.DataFrame, y_inner: dict, *, grid=GBM_GRID, seed=0,
            max_rounds=3000, final_seeds=GBM_FINAL_SEEDS):
    """Grid on inner rows (early stopping on inner, selection by inner MdSA); the representative model per
    target is the inner-best config refitted with ``final_seeds`` (bagging/feature-fraction seeds) and
    averaged in log space — seed-ensembled like SatWQ-Net (review W2). Every grid config's single-seed
    model is kept for variant predictions (``predict(df, X, variant=j)``)."""
    import lightgbm as lgb

    models, grid_models, info = {}, {}, {}
    for t in TARGETS:
        mf, mi = _ok(y_fit[t]), _ok(y_inner[t])
        if mf.sum() < 50 or mi.sum() < 10:
            info[t] = {"skipped": f"n_fit={int(mf.sum())} n_inner={int(mi.sum())}"}
            continue
        scores, grid_models[t] = [], []
        dtr = lgb.Dataset(X_fit[mf], y_fit[t][mf], free_raw_data=False)
        for j, params in enumerate(grid):
            p = dict(params, verbose=-1, num_threads=N_THREADS, seed=seed)
            dva = lgb.Dataset(X_inner[mi], y_inner[t][mi], reference=dtr)
            bst = lgb.train(p, dtr, max_rounds, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
            scores.append(mdsa_log10(bst.predict(X_inner[mi], num_iteration=bst.best_iteration) - y_inner[t][mi]))
            grid_models[t].append(bst)
        jb = int(np.argmin(scores))
        n_rounds = max(1, int(grid_models[t][jb].best_iteration))
        ens = [lgb.train(dict(grid[jb], verbose=-1, num_threads=N_THREADS, seed=s_), dtr, n_rounds) for s_ in final_seeds]
        models[t] = ens
        ens_inner = np.mean([b.predict(X_inner[mi]) for b in ens], 0)
        info[t] = {"inner_mdsa": scores[jb], "inner_mdsa_seed_ensemble": mdsa_log10(ens_inner - y_inner[t][mi]),
                   "config": grid[jb], "config_index": jb, "best_iteration": n_rounds, "n_configs": len(grid),
                   "inner_mdsa_grid": scores, "final_seeds": list(final_seeds)}
    cols = list(X_fit.columns)

    def predict(df, X, variant=None):
        out = {}
        for t in TARGETS:
            if t not in models:
                out[t] = np.full(len(X), np.nan)
            elif variant is None:
                out[t] = np.mean([b.predict(X[cols]) for b in models[t]], 0)
            else:
                b = grid_models[t][variant]
                out[t] = b.predict(X[cols], num_iteration=b.best_iteration)
        return out
    predict.variants = tuple(range(len(grid)))
    return predict, info


def season_location_X(df):
    from .features import context_features

    return context_features(df)


def location_season_matrix(df):
    c = season_location_X(df)
    u = unit_vector(df.lat, df.lon)
    return pd.concat([c, pd.DataFrame(u, columns=["ux", "uy", "uz"], index=df.index)], axis=1)


# ------------------------------------------------------------------------------- index algorithms

def _lg(x):
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.log10(np.asarray(x, float))
    v[~np.isfinite(v)] = np.nan
    return v


INDEX_DEFS = {
    "chla": {"OC3": lambda X: _lg(np.fmax(X.rw3_B01, X.rw3_B02) / X.rw3_B03),
             "NDCI": lambda X: X.ndci.to_numpy(float),
             "B5_B4": lambda X: _lg(X.rw3_B05 / X.rw3_B04),
             # MCI (Gower et al. 2005; recommended for US lakes on S2 by Salls et al. 2024, Remote Sens. 16:1977)
             "MCI": lambda X: (X.rw3_B05 - X.rw3_B04 - (X.rw3_B06 - X.rw3_B04) * (705 - 665) / (740 - 665)).to_numpy(float)},
    "tss": {"red_B4": lambda X: _lg(X.rw3_B04), "nir_B8A": lambda X: _lg(X.rw3_B8A),
            "B4_B3": lambda X: _lg(X.rw3_B04 / X.rw3_B03)},
    "turbidity": {"red_B4": lambda X: _lg(X.rw3_B04), "nir_B8A": lambda X: _lg(X.rw3_B8A),
                  "B4_B3": lambda X: _lg(X.rw3_B04 / X.rw3_B03)},
    "secchi": {"B2_B4": lambda X: _lg(X.rw3_B02 / X.rw3_B04), "B3_B4": lambda X: _lg(X.rw3_B03 / X.rw3_B04),
               "red_B4": lambda X: _lg(X.rw3_B04)},
}


def fit_index(X_fit, y_fit, X_inner, y_inner, degrees=(1, 2, 3)):
    """'-opt' index algorithms: polynomial of the index (log10 target) fitted on fit rows; index and
    degree chosen on inner rows. Undefined index values (non-positive reflectance) fall back to the fit
    median (documented; keeps the baseline defined on every row)."""
    models, info = {}, {}
    for t in TARGETS:
        med = float(np.nanmedian(y_fit[t]))
        best = None
        for name, f in INDEX_DEFS[t].items():
            xf, xi = f(X_fit), f(X_inner)
            mf = _ok(y_fit[t]) & np.isfinite(xf)
            mi = _ok(y_inner[t])
            if mf.sum() < 20 or mi.sum() < 5:
                continue
            lo, hi = np.nanpercentile(xf[mf], [0.5, 99.5])
            for d in degrees:
                coef = np.polyfit(np.clip(xf[mf], lo, hi), y_fit[t][mf], d)
                pi = np.polyval(coef, np.clip(xi[mi], lo, hi))
                pi = np.where(np.isfinite(pi), pi, med)
                s = mdsa_log10(pi - y_inner[t][mi])
                if best is None or s < best[0]:
                    best = (s, name, d, coef, lo, hi)
        if best is None:
            continue
        models[t] = best[1:] + (med,)
        info[t] = {"index": best[1], "degree": best[2], "inner_mdsa": best[0]}

    def predict(df, X):
        out = {}
        for t in TARGETS:
            if t not in models:
                out[t] = np.full(len(X), np.nan)
                continue
            name, d, coef, lo, hi, med = models[t]
            x = INDEX_DEFS[t][name](X)
            p = np.polyval(coef, np.clip(x, lo, hi))
            predict.fallback_rate[t] = float(np.mean(~np.isfinite(p)))   # undefined index -> fit median
            out[t] = np.where(np.isfinite(p), p, med)
        return out
    predict.fallback_rate = {}
    return predict, info
