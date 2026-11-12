# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Convex stacking weights (amendment §6.3): per endpoint ``min_w Σ_i r_i (y_i − P_i·w)²`` with ``w ≥ 0`` and
``Σ w = 1``, and the cross-fitted criterion (leave one inner basin group out).

The solver is exact for the small member counts of the round (≤ 5): it enumerates the active sets. For every
non-empty subset of the members the equality-constrained least-squares problem has a closed form (KKT system); a
subset is feasible if all its weights are non-negative; the feasible subset with the lowest squared error is the
solution (the problem is convex, so this is the global minimum). Ties go to the subset listed first (fewer members
first, then member order). No iteration count, step size or tolerance is tuned."""

from __future__ import annotations

import itertools

import numpy as np

FEAS_TOL = 1e-10
TIE_REL = 1e-12


def simplex_lstsq(P: np.ndarray, y: np.ndarray, r: np.ndarray | None = None, ridge: float = 0.0) -> np.ndarray:
    """Weights (M,) on the simplex. ``P`` (n, M) member predictions, ``y`` (n,), ``r`` (n,) row weights ≥ 0."""
    P, y = np.asarray(P, "float64"), np.asarray(y, "float64")
    n, M = P.shape
    if n == 0 or not (np.isfinite(P).all() and np.isfinite(y).all()):
        raise ValueError("simplex_lstsq needs at least one row and finite inputs")
    r = np.ones(n) if r is None else np.asarray(r, "float64")
    if (r < 0).any() or r.sum() <= 0:
        raise ValueError("row weights must be non-negative with a positive sum")
    Pr = P * r[:, None]
    A, b, yy = P.T @ Pr + ridge * np.eye(M), Pr.T @ y, float((r * y * y).sum())
    scale = max(yy, 1e-300)
    best_w, best_sse = None, np.inf
    for k in range(1, M + 1):
        for s in itertools.combinations(range(M), k):
            ix = np.asarray(s)
            K = np.zeros((k + 1, k + 1))
            K[:k, :k], K[:k, k], K[k, :k] = 2.0 * A[np.ix_(ix, ix)], 1.0, 1.0
            sol = np.linalg.lstsq(K, np.r_[2.0 * b[ix], 1.0], rcond=None)[0]
            ws = sol[:k]
            if (ws < -FEAS_TOL).any() or abs(ws.sum() - 1.0) > 1e-8:
                continue
            ws = np.clip(ws, 0.0, None)
            ws = ws / ws.sum()
            w = np.zeros(M)
            w[ix] = ws
            sse = float(w @ A @ w - 2.0 * b @ w + yy)
            if sse < best_sse - TIE_REL * scale:
                best_w, best_sse = w, sse
    if best_w is None:                                        # cannot happen: every singleton is feasible
        raise RuntimeError("no feasible active set")
    return best_w


def cross_fit(P: np.ndarray, y: np.ndarray, groups, r: np.ndarray | None = None, ridge: float = 0.0):
    """Leave-one-group-out stack predictions: for each group, weights from all other groups. Returns
    ``(pred (n,), weights_by_fold {group: (M,)})``. One group only → the full-data weights (reported as such)."""
    P, y = np.asarray(P, "float64"), np.asarray(y, "float64")
    g = np.asarray(groups)
    r = np.ones(len(y)) if r is None else np.asarray(r, "float64")
    pred, folds = np.full(len(y), np.nan), {}
    ug = sorted(set(g.tolist()), key=str)
    if len(ug) < 2:
        w = simplex_lstsq(P, y, r, ridge)
        return P @ w, {str(ug[0]) if ug else "all": w}
    for u in ug:
        te = g == u
        w = simplex_lstsq(P[~te], y[~te], r[~te], ridge)
        pred[te] = P[te] @ w
        folds[str(u)] = w
    return pred, folds


def grab_row_weights(ok: np.ndarray, cap: int) -> np.ndarray:
    """(n, T) row weights of J_grab: min(n_s, cap) / n_s for a station with n_s scored rows (0 where not scored)."""
    n = ok.sum(axis=1, keepdims=True).astype("float64")
    return np.where(ok, np.minimum(n, cap) / np.maximum(n, 1.0), 0.0)
