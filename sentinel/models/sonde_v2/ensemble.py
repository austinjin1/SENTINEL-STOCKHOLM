# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""E2 / E2-B weighting rule (``sonde_arch_v2_amendment.md`` §2a, §2b, §2h). ONE function serves both the SENTINEL
ensemble E2 and the baseline-only strict control E2-B, so the weighting gain is never credited to SENTINEL.

The rule: quantile averaging q_ens = Σ_m w_m q_m with w on the simplex, minimising the mean pinball loss over the 19
levels (a CRPS approximation) on OUT-OF-FOLD predictions. The problem is convex in w; it is solved by exponentiated
(mirror) subgradient descent with a fixed schedule (deterministic, no tuning). Variants:
W1 = one weight vector per variable × horizon block {h ≤ 2, 3 ≤ h ≤ 7, h ≥ 10}; W2 = one per variable.
:func:`cross_fitted_score` gives the honest selection score: weights fitted on K−1 folds, scored on the K-th.
"""

from __future__ import annotations

import numpy as np

TAUS = np.round(np.arange(0.05, 0.951, 0.05), 2)
H_BLOCKS = {"W1": ((1, 2), (3, 5, 7), (10, 14)), "W2": ((1, 2, 3, 5, 7, 10, 14),)}


def pinball_mean(q: np.ndarray, y: np.ndarray) -> float:
    """q (N, 19), y (N,) → mean pinball over rows and levels (rows with non-finite y or q dropped)."""
    ok = np.isfinite(y) & np.isfinite(q).all(axis=1)
    d = y[ok, None] - q[ok]
    return float(np.mean(np.maximum(TAUS * d, (TAUS - 1) * d))) if ok.any() else np.nan


def fit_weights(member_q: np.ndarray, y: np.ndarray, *, iters: int = 400, eta0: float = 0.5) -> np.ndarray:
    """member_q (M, N, 19), y (N,) → simplex weights (M,)."""
    M = member_q.shape[0]
    ok = np.isfinite(y) & np.isfinite(member_q).all(axis=(0, 2))
    Q, yy = member_q[:, ok], y[ok]
    if yy.size == 0:
        return np.full(M, 1.0 / M)
    scale = np.nanstd(yy) + 1e-9
    w = np.full(M, 1.0 / M)
    best_w, best = w.copy(), np.inf
    for it in range(iters):
        q = np.tensordot(w, Q, axes=1)                       # (N, 19)
        d = yy[:, None] - q
        loss = float(np.mean(np.maximum(TAUS * d, (TAUS - 1) * d)))
        if loss < best:
            best, best_w = loss, w.copy()
        g_q = np.where(d > 0, -TAUS, 1 - TAUS) / d.size      # d loss / d q
        grad = np.tensordot(Q, g_q, axes=([1, 2], [0, 1])) / scale
        w = w * np.exp(-eta0 / np.sqrt(it + 1) * grad / (np.abs(grad).max() + 1e-12))
        w = w / w.sum()
    return best_w


def combine(member_q: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.sort(np.tensordot(w, member_q, axes=1), axis=-1)


def fit_rule(oof: dict, y: dict, variant: str = "W1") -> dict:
    """oof[(var, h)] = (M, N, 19) out-of-fold member quantiles (members in a FIXED order), y[(var, h)] = (N,).
    Returns weights[(var, block_index)] = (M,) — same structure for E2 and E2-B."""
    blocks = H_BLOCKS[variant]
    out = {}
    for var in sorted({v for v, _ in oof}):
        for bi, hs in enumerate(blocks):
            keys = [(var, h) for h in hs if (var, h) in oof]
            if not keys:
                continue
            Qcat = np.concatenate([oof[k] for k in keys], axis=1)
            ycat = np.concatenate([y[k] for k in keys])
            out[(var, bi)] = fit_weights(Qcat, ycat)
    return out


def weight_for(weights: dict, var: str, h: int, variant: str = "W1") -> np.ndarray:
    for bi, hs in enumerate(H_BLOCKS[variant]):
        if h in hs:
            return weights[(var, bi)]
    raise KeyError(h)


def cross_fitted_score(oof: dict, y: dict, fold: dict, variant: str = "W1") -> dict:
    """fold[(var, h)] = (N,) fold id of each out-of-fold row. For each fold k: fit the rule on rows of the other
    folds, score the combined forecast on fold k. Returns {(var, h): mean pinball} pooled over folds."""
    folds = sorted({int(f) for a in fold.values() for f in np.unique(a)})
    num, den = {}, {}
    for k in folds:
        tr_o = {key: oof[key][:, fold[key] != k] for key in oof}
        tr_y = {key: y[key][fold[key] != k] for key in oof}
        w = fit_rule(tr_o, tr_y, variant)
        for key in oof:
            m = fold[key] == k
            if not m.any():
                continue
            var, h = key
            q = combine(oof[key][:, m], weight_for(w, var, h, variant))
            s = pinball_mean(q, y[key][m])
            if np.isfinite(s):
                n = int(np.isfinite(y[key][m]).sum())
                num[key] = num.get(key, 0.0) + s * n
                den[key] = den.get(key, 0) + n
    return {key: num[key] / den[key] for key in num}
