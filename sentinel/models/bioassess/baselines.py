# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Baselines for community -> chemistry, all with one interface (protocol §6).

``fit_predict(name, design, train_ids, test_ids, seed, params, targets)`` returns a DataFrame
(index = test_ids, columns = targets) of predictions. Every model is fit on ``train_ids`` only;
features come from :mod:`sentinel.models.bioassess.features` (fit on the training fold) and never
include chemistry. Targets with a missing value are dropped row-wise for single-target models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sentinel.models.bioassess.data import TARGET_NAMES
from sentinel.models.bioassess.features import CommunityFeaturizer, geography_matrix

BASELINES = ("rf_community", "rf_taxa_only", "rf_geography", "rf_community_geography", "gbm_community",
             "ecoregion_mean", "nearest_reference", "knn_community", "wapls2", "tabpfn_community",
             "rf_context", "rf_community_context")
REGISTRY_MODEL_NAME = {
    "rf_community": "RandomForest[community]", "rf_taxa_only": "RandomForest[community-taxa-only]",
    "rf_geography": "RandomForest[geography]", "rf_community_geography": "RandomForest[community+geography]",
    "gbm_community": "LightGBM[community]", "ecoregion_mean": "EcoregionMean",
    "nearest_reference": "NearestReferenceSite", "knn_community": "kNN-MAT[community]",
    "wapls2": "WA-PLS2[community]", "tabpfn_community": "TabPFNv2[community]",
    "rf_context": "RandomForest[context]", "rf_community_context": "RandomForest[community+context]",
}


def _yv(design, ids, t):
    return design.Y.loc[pd.Index(ids), t].to_numpy(float)


def _per_target(fit_one, Xtr, Xte, design, train_ids, targets):
    out = {}
    for t in targets:
        y = _yv(design, train_ids, t)
        ok = np.isfinite(y)
        out[t] = fit_one(Xtr[ok], y[ok], Xte, t)
    return out


def _rf(params, seed, n_jobs):
    from sklearn.ensemble import RandomForestRegressor

    def f(X, y, Xte, _t):
        m = RandomForestRegressor(**params, n_jobs=n_jobs, random_state=seed)
        return m.fit(X, y).predict(Xte)
    return f


def _gbm(params, seed, n_jobs):
    import lightgbm as lgb

    def f(X, y, Xte, _t):
        m = lgb.LGBMRegressor(**params, random_state=seed, n_jobs=n_jobs)
        return m.fit(X, y).predict(Xte)
    return f


def wapls_fit_predict(A_tr: np.ndarray, y: np.ndarray, A_te: np.ndarray, n_components: int = 2) -> np.ndarray:
    """WA-PLS (ter Braak & Juggins 1993), component ``n_components`` prediction.

    ``A`` = relative abundances (rows sum to 1 over the retained taxa). Implementation follows the
    rioja/C2 algorithm: weighted-averaging PLS with sample weights = row sums, species weights =
    column sums; prediction uses the regression of y on site scores.
    """
    Y = np.asarray(A_tr, float)
    keep = Y.sum(0) > 0
    Y = Y[:, keep]
    Xn = np.asarray(A_te, float)[:, keep]
    R = Y.sum(1)
    R = np.where(R > 0, R, 1e-12)
    C = Y.sum(0)
    tot = R.sum()
    x = np.asarray(y, float)
    xbar = np.sum(R * x) / tot
    xc = x - xbar
    T = np.zeros((len(x), n_components))
    U = np.zeros((Y.shape[1], n_components))
    B = np.zeros(n_components)
    resid = xc.copy()
    for k in range(n_components):
        u = (Y.T @ resid) / C                       # species optima (WA of residuals)
        r = (Y @ u) / R                              # site scores
        for j in range(k):                           # orthogonalise (weighted by R)
            tj = T[:, j]
            r = r - tj * (np.sum(R * r * tj) / np.sum(R * tj * tj))
        r = r - np.sum(R * r) / tot
        T[:, k] = r
        b = np.sum(R * r * resid) / np.sum(R * r * r)
        B[k] = b
        resid = resid - b * r
        U[:, k] = u
    # Prediction for new samples: replicate the score recursion with the stored optima
    Rn = Xn.sum(1)
    Rn = np.where(Rn > 0, Rn, 1e-12)
    Tn = np.zeros((len(Xn), n_components))
    for k in range(n_components):
        r = (Xn @ U[:, k]) / Rn
        rtr = (Y @ U[:, k]) / R
        for j in range(k):
            tj = T[:, j]
            c = np.sum(R * rtr * tj) / np.sum(R * tj * tj)
            rtr = rtr - tj * c
            r = r - Tn[:, j] * c
        mu = np.sum(R * rtr) / tot
        Tn[:, k] = r - mu
    return xbar + Tn @ B


def fit_predict(name: str, design, train_ids, test_ids, *, seed: int = 0, params: dict | None = None,
                targets=TARGET_NAMES, n_jobs: int = 8, min_samples: int = 10) -> pd.DataFrame:
    params = dict(params or {})
    train_ids, test_ids = pd.Index(train_ids), pd.Index(test_ids)
    if set(train_ids) & set(test_ids):
        raise AssertionError("train and test ids overlap")
    s_tr, s_te = design.samples.loc[train_ids], design.samples.loc[test_ids]
    preds: dict[str, np.ndarray] = {}

    if name in ("rf_community", "rf_taxa_only", "rf_community_geography", "gbm_community", "rf_community_context"):
        fz = CommunityFeaturizer(min_samples=min_samples, descriptors=(name != "rf_taxa_only"),
                                 geography=name in ("rf_community_geography", "rf_community_context"),
                                 season=(name == "rf_community_context")).fit(design, train_ids)
        Xtr, Xte = fz.transform(design, train_ids), fz.transform(design, test_ids)
        fit_one = _gbm(params, seed, n_jobs) if name == "gbm_community" else _rf(params, seed, n_jobs)
        preds = _per_target(fit_one, Xtr, Xte, design, train_ids, targets)
    elif name in ("rf_geography", "rf_context"):
        sea = name == "rf_context"
        Xtr = geography_matrix(design, train_ids, train_ids, sea)
        Xte = geography_matrix(design, train_ids, test_ids, sea)
        preds = _per_target(_rf(params, seed, n_jobs), Xtr, Xte, design, train_ids, targets)
    elif name == "ecoregion_mean":
        for t in targets:
            y = pd.Series(_yv(design, train_ids, t), index=train_ids)
            means = y.groupby(s_tr["ag_eco9"]).mean()
            p = s_te["ag_eco9"].map(means).to_numpy(float)
            preds[t] = np.where(np.isfinite(p), p, np.nanmean(y))   # unseen ecoregion -> global mean
    elif name == "nearest_reference":
        from sklearn.neighbors import BallTree

        k = int(params.get("k", 1))
        for t in targets:
            y = _yv(design, train_ids, t)
            ref = (s_tr["reference_class"] == "R").to_numpy() & np.isfinite(y)
            tree = BallTree(np.radians(s_tr.loc[ref, ["lat", "lon"]].to_numpy(float)), metric="haversine")
            _, idx = tree.query(np.radians(s_te[["lat", "lon"]].to_numpy(float)), k=k)
            preds[t] = y[ref][idx].mean(1)
    elif name == "knn_community":
        from sklearn.neighbors import KNeighborsRegressor

        fz = CommunityFeaturizer(min_samples=min_samples, descriptors=False).fit(design, train_ids)
        Xtr, Xte = fz.transform(design, train_ids), fz.transform(design, test_ids)

        def f(X, y, Xq, _t):
            m = KNeighborsRegressor(n_neighbors=int(params.get("k", 10)), weights=params.get("weights", "distance"))
            return m.fit(X, y).predict(Xq)
        preds = _per_target(f, Xtr, Xte, design, train_ids, targets)
    elif name == "wapls2":
        fz = CommunityFeaturizer(min_samples=min_samples, descriptors=False).fit(design, train_ids)
        Atr, Ate = fz.transform(design, train_ids) ** 2, fz.transform(design, test_ids) ** 2   # back to rel. abund.

        def f(A, y, Aq, _t):
            return wapls_fit_predict(A, y, Aq, int(params.get("n_components", 2)))
        preds = _per_target(f, Atr, Ate, design, train_ids, targets)
    elif name == "tabpfn_community":
        from sentinel.models.chemtox.tabpfn_model import load_tabpfn

        TabPFNRegressor, _ = load_tabpfn()
        fz = CommunityFeaturizer(min_samples=min_samples, descriptors=True,
                                 max_taxa=int(params.get("max_taxa", 300))).fit(design, train_ids)
        Xtr = fz.transform(design, train_ids).astype(np.float32)
        Xte = fz.transform(design, test_ids).astype(np.float32)
        max_rows = int(params.get("max_train_rows", 6000))
        chunk = int(params.get("predict_chunk", 1000))

        def f(X, y, Xq, _t):
            if len(X) > max_rows:
                pick = np.random.default_rng(seed).choice(len(X), max_rows, replace=False)
                X, y = X[pick], y[pick]
            m = TabPFNRegressor(n_estimators=int(params.get("n_estimators", 8)),
                                device=params.get("device", "cuda"), random_state=seed,
                                ignore_pretraining_limits=True)
            m.fit(X, y)
            return np.concatenate([m.predict(Xq[i:i + chunk]) for i in range(0, len(Xq), chunk)])
        preds = _per_target(f, Xtr, Xte, design, train_ids, targets)
    else:
        raise ValueError(f"unknown baseline {name}")
    return pd.DataFrame(preds, index=test_ids)
