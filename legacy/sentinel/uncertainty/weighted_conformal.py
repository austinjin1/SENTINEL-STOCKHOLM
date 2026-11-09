"""Weighted split-conformal prediction under covariate shift.

Reference
---------
Tibshirani, Barber, Candes & Ramdas (2019), "Conformal Prediction Under
Covariate Shift", NeurIPS 2019.  Under a covariate shift where the test
covariate density p_test(x) differs from the calibration/training density
p_cal(x) but the conditional P(y|x) is unchanged, ordinary split-conformal
loses its finite-sample coverage guarantee.  Tibshirani et al. restore
validity by *weighting* each calibration nonconformity score by the
likelihood ratio

        w(x) = p_test(x) / p_cal(x)

and taking a **weighted** quantile of the calibration scores, where the
weight budget also includes the (unknown-score) test point via a normalising
constant.  Concretely, for a new test point x_{n+1} the prediction set uses
the (1 - alpha) quantile of the distribution

        sum_i  p_i(x_{n+1}) * delta_{V_i}  +  p_{n+1}(x_{n+1}) * delta_{+inf}

with p_i(x) = w(x_i) / (sum_{j=1..n} w(x_j) + w(x)).  Because w(x_{n+1})
enters the normaliser, the *effective* target quantile level is

        alpha_eff = (1 - alpha) * (W + w_test) / W  clipped to [0, 1]

which for a single scalar test weight equals the weighted quantile of the
calibration scores at level (1-alpha) on the weight vector [w_cal, w_test].
This module implements that construction; when all weights are equal it
reduces exactly to ordinary split-conformal.

The likelihood ratio w(x) is estimated with a *domain classifier* d(x) =
P(test | x) trained to separate calibration covariates (label 0) from test
covariates (label 1); then

        w(x) = P(test|x) / P(cal|x) = d(x) / (1 - d(x))

up to the constant prior ratio (which cancels in the normalised weights).
This is a standard density-ratio-via-classification estimator.

All routines are model-agnostic: they consume nonconformity SCORES, not
models, so they can wrap any classifier / regressor.

REAL data only.  MIT License -- Bryan Cheng, 2026.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Likelihood-ratio weights via a domain classifier
# ---------------------------------------------------------------------------

def domain_classifier_weights(
    X_cal: np.ndarray,
    X_test: np.ndarray,
    *,
    method: str = "logreg",
    clip: float = 1e4,
    standardize: bool = True,
    n_splits: int = 5,
    seed: int = 0,
    return_auc: bool = False,
):
    """Estimate covariate-shift likelihood ratios w(x)=p_test(x)/p_cal(x).

    A binary domain classifier is trained to separate calibration covariates
    (label 0) from test covariates (label 1).  To avoid the classifier simply
    memorising points, out-of-fold probabilities are used for BOTH pools via
    stratified K-fold cross-fitting on the pooled (cal+test) covariates.  The
    likelihood ratio is w(x) = d(x)/(1-d(x)), corrected for the sampling prior
    so that w estimates p_test/p_cal (not the posterior odds).

    Parameters
    ----------
    X_cal, X_test:
        Calibration and test covariate matrices (n_cal, d) and (n_test, d).
    method:
        'logreg' (L2 logistic regression) or 'gb' (HistGradientBoosting).
    clip:
        Symmetric clip on the ratio to [1/clip, clip] for numerical stability.
    standardize:
        Standardise features on the pooled data before fitting (logreg only).
    n_splits:
        Cross-fitting folds.
    seed:
        RNG seed.
    return_auc:
        If True, also return the cross-fit domain-classifier AUC (a diagnostic
        of how strong the covariate shift is: 0.5 = no detectable shift).

    Returns
    -------
    w_cal, w_test  (and optionally auc)
        Likelihood-ratio weights for calibration and test points.
    """
    from sklearn.model_selection import StratifiedKFold

    X_cal = np.asarray(X_cal, dtype=np.float64)
    X_test = np.asarray(X_test, dtype=np.float64)
    n_cal, n_test = len(X_cal), len(X_test)
    X = np.vstack([X_cal, X_test])
    y = np.concatenate([np.zeros(n_cal), np.ones(n_test)]).astype(int)

    # impute non-finite before fitting
    X = np.where(np.isfinite(X), X, np.nan)
    col_med = np.nanmedian(X, axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0)
    inds = np.where(~np.isfinite(X))
    X[inds] = np.take(col_med, inds[1])

    if standardize:
        mu = X.mean(0)
        sd = X.std(0)
        sd[sd == 0] = 1.0
        Xs = (X - mu) / sd
    else:
        Xs = X

    prob = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in skf.split(Xs, y):
        if method == "logreg":
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
        elif method == "gb":
            from sklearn.ensemble import HistGradientBoostingClassifier
            clf = HistGradientBoostingClassifier(
                max_depth=3, learning_rate=0.05, max_iter=300,
                l2_regularization=1.0, random_state=seed)
        else:
            raise ValueError(method)
        clf.fit(Xs[tr], y[tr])
        prob[te] = clf.predict_proba(Xs[te])[:, 1]

    # AUC diagnostic
    auc = float("nan")
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y, prob))
    except Exception:
        pass

    # posterior odds d/(1-d) -> density ratio by dividing out the sampling prior
    # odds ratio n_test/n_cal:  p_test/p_cal = [d/(1-d)] * [P(cal)/P(test)]
    eps = 1e-6
    prob = np.clip(prob, eps, 1 - eps)
    odds = prob / (1.0 - prob)
    prior_correction = n_cal / max(n_test, 1)  # P(cal)/P(test)
    w = odds * prior_correction
    w = np.clip(w, 1.0 / clip, clip)

    w_cal, w_test = w[:n_cal], w[n_cal:]
    if return_auc:
        return w_cal, w_test, auc
    return w_cal, w_test


# ---------------------------------------------------------------------------
# Weighted quantile
# ---------------------------------------------------------------------------

def weighted_quantile(scores: np.ndarray, weights: np.ndarray, level: float) -> float:
    """(level)-quantile of a discrete distribution sum_i p_i delta_{scores_i}.

    p_i = weights_i / sum(weights).  Returns the smallest score s such that the
    cumulative weight of {scores <= s} is >= level.  With equal weights this is
    the usual empirical quantile with the (n+1) correction supplied by the
    caller (by appending a +inf atom for the test point).
    """
    scores = np.asarray(scores, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    order = np.argsort(scores, kind="mergesort")
    s = scores[order]
    w = weights[order]
    tot = w.sum()
    if tot <= 0:
        return float(np.inf)
    cum = np.cumsum(w) / tot
    idx = np.searchsorted(cum, level, side="left")
    if idx >= len(s):
        return float(np.inf)
    return float(s[idx])


def _weighted_threshold(cal_scores: np.ndarray, w_cal: np.ndarray,
                        w_test_scalar: float, alpha: float) -> float:
    """Tibshirani-2019 weighted-conformal threshold for one test weight.

    Appends a +inf atom carrying the test-point weight (its score is unknown /
    unbounded) and returns the (1-alpha) weighted quantile of the augmented
    distribution.  Equal weights -> ordinary split-conformal (n+1)-quantile.
    """
    scores_aug = np.concatenate([cal_scores, [np.inf]])
    weights_aug = np.concatenate([w_cal, [w_test_scalar]])
    return weighted_quantile(scores_aug, weights_aug, 1.0 - alpha)


# ---------------------------------------------------------------------------
# Classification: conformal prediction SETS
# ---------------------------------------------------------------------------

def conformal_classification_sets(
    cal_scores: np.ndarray,
    test_proba: np.ndarray,
    alpha: float = 0.1,
    w_cal: Optional[np.ndarray] = None,
    w_test: Optional[np.ndarray] = None,
):
    """Build conformal prediction SETS for a classifier.

    Nonconformity = 1 - p_model(true class).  A label k is INCLUDED in the set
    for test point i iff its score  s_ik = 1 - test_proba[i, k]  <=  qhat_i,
    where qhat_i is the (weighted) (1-alpha) quantile of the calibration scores.

    Parameters
    ----------
    cal_scores:
        (n_cal,) nonconformity scores 1 - p(true class) on the calibration set.
    test_proba:
        (n_test, n_classes) model class probabilities on the test set.
    alpha:
        Miscoverage level (target coverage = 1 - alpha).
    w_cal, w_test:
        Optional likelihood-ratio weights.  If None -> ordinary split-conformal
        (equal weights).

    Returns
    -------
    dict with:
      sets        : list of length n_test, each a sorted list of class indices
      qhat        : (n_test,) per-point conformal threshold
      set_sizes   : (n_test,) |set_i|
    """
    cal_scores = np.asarray(cal_scores, dtype=np.float64)
    P = np.asarray(test_proba, dtype=np.float64)
    n_test, n_classes = P.shape
    test_scores = 1.0 - P  # (n_test, n_classes) score if that class were true

    if w_cal is None:
        w_cal = np.ones_like(cal_scores)
    if w_test is None:
        w_test = np.ones(n_test)
    w_cal = np.asarray(w_cal, dtype=np.float64)
    w_test = np.asarray(w_test, dtype=np.float64)

    qhat = np.empty(n_test)
    sets = []
    for i in range(n_test):
        q = _weighted_threshold(cal_scores, w_cal, float(w_test[i]), alpha)
        qhat[i] = q
        members = np.where(test_scores[i] <= q)[0]
        sets.append(sorted(int(k) for k in members))
    set_sizes = np.array([len(s) for s in sets], dtype=int)
    return {"sets": sets, "qhat": qhat, "set_sizes": set_sizes}


def coverage_and_size_classification(sets, y_true):
    """Empirical coverage (y_true in set) and mean set size."""
    y_true = np.asarray(y_true)
    covered = np.array([int(int(y_true[i]) in set(sets[i])) for i in range(len(sets))])
    sizes = np.array([len(s) for s in sets], dtype=float)
    return {
        "coverage": float(covered.mean()),
        "mean_set_size": float(sizes.mean()),
        "median_set_size": float(np.median(sizes)),
        "n": int(len(sets)),
        "n_empty_sets": int((sizes == 0).sum()),
    }


# ---------------------------------------------------------------------------
# Regression: conformal prediction INTERVALS
# ---------------------------------------------------------------------------

def conformal_regression_intervals(
    cal_residuals: np.ndarray,
    test_pred: np.ndarray,
    alpha: float = 0.1,
    cal_sigma: Optional[np.ndarray] = None,
    test_sigma: Optional[np.ndarray] = None,
    w_cal: Optional[np.ndarray] = None,
    w_test: Optional[np.ndarray] = None,
):
    """Build conformal prediction INTERVALS for a regressor.

    Absolute-residual nonconformity:  s_i = |y_i - yhat_i|  (cal_sigma=None).
    Normalized (locally-adaptive):     s_i = |y_i - yhat_i| / sigma_i.

    The (weighted) (1-alpha) quantile qhat_i gives the half-width; the interval
    for test point i is  [yhat_i - qhat_i * sig_i,  yhat_i + qhat_i * sig_i]
    (sig_i = 1 in the absolute case).

    Returns dict with lo, hi, qhat, half_width.
    """
    cal_residuals = np.abs(np.asarray(cal_residuals, dtype=np.float64))
    test_pred = np.asarray(test_pred, dtype=np.float64)
    n_test = len(test_pred)

    if cal_sigma is not None:
        cal_sigma = np.asarray(cal_sigma, dtype=np.float64)
        cal_scores = cal_residuals / np.maximum(cal_sigma, 1e-8)
    else:
        cal_scores = cal_residuals
    if test_sigma is None:
        test_sigma = np.ones(n_test)
    test_sigma = np.asarray(test_sigma, dtype=np.float64)

    if w_cal is None:
        w_cal = np.ones_like(cal_scores)
    if w_test is None:
        w_test = np.ones(n_test)
    w_cal = np.asarray(w_cal, dtype=np.float64)
    w_test = np.asarray(w_test, dtype=np.float64)

    qhat = np.empty(n_test)
    lo = np.empty(n_test)
    hi = np.empty(n_test)
    for i in range(n_test):
        q = _weighted_threshold(cal_scores, w_cal, float(w_test[i]), alpha)
        qhat[i] = q
        hw = q * test_sigma[i]
        lo[i] = test_pred[i] - hw
        hi[i] = test_pred[i] + hw
    return {"lo": lo, "hi": hi, "qhat": qhat, "half_width": (hi - lo) / 2.0}


def coverage_and_width_regression(lo, hi, y_true):
    """Empirical coverage (y_true in [lo,hi]) and mean interval width."""
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    covered = ((y_true >= lo) & (y_true <= hi)).astype(float)
    width = hi - lo
    finite = np.isfinite(width)
    return {
        "coverage": float(covered.mean()),
        "mean_width": float(width[finite].mean()) if finite.any() else float("inf"),
        "median_width": float(np.median(width[finite])) if finite.any() else float("inf"),
        "n": int(len(y_true)),
        "n_infinite_width": int((~finite).sum()),
    }
