# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Stage-0 comparators for SENTINEL-Fuse (first-class competitors, RESEARCH_PLAN §6).

* ``gbm_gauss`` — LightGBM mean model + LightGBM gamma model of squared out-of-fold residuals →
  Gaussian predictive distribution (μ, σ). Native NaN handling (absent modalities stay NaN).
* ``gbm_binary`` — LightGBM classifier → probability.
* Configurations (per target and fold): ``hub`` (forcings + catchment + time only), ``single:<m>``
  (modality m + hub), ``concat`` (all modalities + hub), ``ume`` (uni-modal ensemble: Gaussian
  mixture of the single-modality predictions of the modalities present in the row, hub model if
  none), ``stack`` (Super-Learner-style GBM over inner-fold OOF single-modality predictions and
  presence flags), ``pattern`` (P4: a model per availability pattern, trained on all training rows
  whose availability ⊇ the pattern, using only that pattern's features), ``persistence`` (last valid
  value of the same sensor variable; σ = training RMSE by target).

Inner validation (early stopping, OOF residuals) always splits *training* rows by group (site), so
no evaluation row informs any fitted quantity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LGB_PARAMS = dict(learning_rate=0.05, num_leaves=31, min_child_samples=40, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1)
MAX_ROUNDS = 2000
EARLY_STOP = 100


def _group_kfold(groups: np.ndarray, k: int, seed: int) -> list[np.ndarray]:
    u = np.unique(groups)
    rs = np.random.default_rng(seed)
    rs.shuffle(u)
    fold_of = {g: i % k for i, g in enumerate(u)}
    f = np.array([fold_of[g] for g in groups])
    return [np.flatnonzero(f == i) for i in range(k)]


def _lgb():
    import lightgbm as lgb
    return lgb


def _fit_l2(X, y, groups, seed, threads, objective="regression", init_rounds=None, weight=None):
    lgb = _lgb()
    params = dict(LGB_PARAMS, objective=objective, seed=seed, num_threads=threads)
    if init_rounds is not None:
        return lgb.train(params, lgb.Dataset(X, y, weight=weight), num_boost_round=init_rounds)
    folds = _group_kfold(groups, 5, seed)
    va = folds[0]
    tr = np.setdiff1d(np.arange(len(y)), va)
    if len(va) < 20 or len(tr) < 50:
        return lgb.train(params, lgb.Dataset(X, y, weight=weight), num_boost_round=200)
    dtr = lgb.Dataset(X.iloc[tr] if hasattr(X, "iloc") else X[tr], y[tr], weight=None if weight is None else weight[tr])
    dva = lgb.Dataset(X.iloc[va] if hasattr(X, "iloc") else X[va], y[va], reference=dtr)
    b = lgb.train(params, dtr, num_boost_round=MAX_ROUNDS, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
    n = max(int(b.best_iteration or 100), 20)
    return lgb.train(params, lgb.Dataset(X, y, weight=weight), num_boost_round=int(n * 1.1))


def gbm_gauss(Xtr: pd.DataFrame, ytr: np.ndarray, gtr: np.ndarray, Xev: pd.DataFrame, *, seed: int = 0,
              threads: int = 6, n_resid_folds: int = 3) -> tuple[np.ndarray, np.ndarray, dict]:
    """Gaussian predictive distribution from a mean GBM and a gamma GBM on OOF squared residuals."""
    keep = np.isfinite(ytr)
    Xtr, ytr, gtr = Xtr.loc[keep], ytr[keep], gtr[keep]
    mean_model = _fit_l2(Xtr, ytr, gtr, seed, threads)
    n_rounds = mean_model.current_iteration()
    oof = np.full(len(ytr), np.nan)
    for va in _group_kfold(gtr, n_resid_folds, seed + 1):
        tr = np.setdiff1d(np.arange(len(ytr)), va)
        m = _fit_l2(Xtr.iloc[tr], ytr[tr], gtr[tr], seed, threads, init_rounds=n_rounds)
        oof[va] = m.predict(Xtr.iloc[va])
    r2 = (ytr - oof) ** 2 + 1e-6
    var_model = _fit_l2(Xtr, r2, gtr, seed + 2, threads, objective="gamma")
    mu = mean_model.predict(Xev)
    var = var_model.predict(Xev)
    sigma = np.sqrt(np.clip(var, 1e-8, None))
    return mu, sigma, {"rounds_mean": int(n_rounds), "rounds_var": int(var_model.current_iteration()),
                       "n_train": int(len(ytr))}


def gbm_gauss_anchored(Xtr: pd.DataFrame, ytr: np.ndarray, gtr: np.ndarray, Xev: pd.DataFrame,
                       atr: np.ndarray, aev: np.ndarray, *, seed: int = 0, threads: int = 6,
                       min_level_rows: int = 200) -> tuple[np.ndarray, np.ndarray, dict]:
    """Persistence-anchored Gaussian GBM: on rows with a finite anchor (last valid value of the
    target's own variable, available at issue time) the mean/variance models are fit to y − anchor
    and μ = anchor + Δ̂; rows without an anchor use the level model (:func:`gbm_gauss`).

    Trees cannot extrapolate levels to unseen sites; the anchored parametrisation removes the site
    level (review of Stage-0 v2: persistence beat level GBMs at h = 7 for DO and nitrate). The anchor
    belongs to its modality, so only configurations containing that modality may use it.
    """
    ok_tr = np.isfinite(atr)
    ok_ev = np.isfinite(aev)
    mu = np.full(len(Xev), np.nan)
    sg = np.full(len(Xev), np.nan)
    info = {}
    if ok_tr.sum() >= 100 and ok_ev.any():
        m, s_, i = gbm_gauss(Xtr.loc[ok_tr], ytr[ok_tr] - atr[ok_tr], gtr[ok_tr], Xev.loc[ok_ev], seed=seed,
                             threads=threads)
        mu[ok_ev], sg[ok_ev] = aev[ok_ev] + m, s_
        info["anchored"] = i
    rest = ~ok_ev if ok_tr.sum() >= 100 else np.ones(len(Xev), bool)
    if rest.any():
        lvl = (~ok_tr) if (~ok_tr & np.isfinite(ytr)).sum() >= min_level_rows else np.ones(len(ytr), bool)
        m, s_, i = gbm_gauss(Xtr.loc[lvl], ytr[lvl], gtr[lvl], Xev.loc[rest], seed=seed, threads=threads)
        mu[rest], sg[rest] = m, s_
        info["level"] = i
    return mu, sg, info


def gbm_mean(Xtr, ytr, gtr, Xev, *, seed=0, threads=6):
    keep = np.isfinite(ytr)
    m = _fit_l2(Xtr.loc[keep], ytr[keep], gtr[keep], seed, threads)
    return m.predict(Xev)


def gbm_binary(Xtr, ytr, gtr, Xev, *, seed=0, threads=6):
    keep = np.isfinite(ytr)
    m = _fit_l2(Xtr.loc[keep], ytr[keep].astype(float), gtr[keep], seed, threads, objective="binary")
    return m.predict(Xev)


def mixture_moments(mus: np.ndarray, sigmas: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Moment-matched Gaussian of an equal-weight mixture (rows × components, w = availability)."""
    w = w.astype(float)
    ws = w.sum(1, keepdims=True)
    ws[ws == 0] = 1.0
    w = w / ws
    mu = np.nansum(w * mus, 1)
    m2 = np.nansum(w * (sigmas ** 2 + mus ** 2), 1)
    return mu, np.sqrt(np.clip(m2 - mu ** 2, 1e-12, None))


def crps_gauss_rows(y, mu, sigma) -> np.ndarray:
    from scipy.stats import norm
    sigma = np.maximum(sigma, 1e-9)
    z = (y - mu) / sigma
    return sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def pattern_key(pres: pd.DataFrame) -> pd.Series:
    mods = list(pres.columns)
    return pres.apply(lambda r: "+".join(m for m in mods if r[m]) or "hub", axis=1)
