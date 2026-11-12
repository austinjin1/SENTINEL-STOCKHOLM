# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Split-conformal recalibration of the predictive intervals (orbit_protocol.md §5b; secondary, pre-declared
2026-11-10 before any conformal result).

Conformalised quantile regression (Romano et al. 2019) on the raw central intervals [q_lo, q_hi] (log10) of a
distributional model (SatWQ-Net mixture; MDN_B):
    score_i = max(q_lo_i - y_i, y_i - q_hi_i);   interval = [q_lo - Q, q_hi + Q]
with Q the **water-body-weighted** (1 - alpha) quantile of the calibration scores: every water body carries total
weight 1 (rows within a lake are not exchangeable; lakes are the exchangeable unit), with the finite-sample level
min(1, (1 - alpha) (1 + 1/n_waterbodies)). One Q per (model, target, level), fitted on the val calibration cell and
frozen. Point predictions and model selection are unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LEVELS = {"68": (0.16, 0.84), "95": (0.025, 0.975)}
N_FOLDS = 5


def scores(y, lo, hi) -> np.ndarray:
    return np.maximum(np.asarray(lo, float) - y, np.asarray(y, float) - np.asarray(hi, float))


def wb_quantile(s, groups, alpha: float) -> float:
    """Water-body-weighted conformal quantile of scores ``s`` (finite rows only)."""
    s, g = np.asarray(s, float), np.asarray(groups)
    ok = np.isfinite(s)
    s, g = s[ok], g[ok]
    if s.size == 0:
        return np.nan
    cnt = pd.Series(g).map(pd.Series(g).value_counts()).to_numpy(float)
    w = 1.0 / cnt
    n_g = len(np.unique(g))
    level = min(1.0, (1 - alpha) * (1 + 1 / n_g))
    o = np.argsort(s)
    cw = np.cumsum(w[o]) / w.sum()
    return float(s[o][min(np.searchsorted(cw, level, side="left"), len(s) - 1)])


def fit(y, Q: dict, groups) -> dict:
    """Q = {quantile level: array}; returns {"68": offset, "95": offset, "n": .., "n_waterbodies": ..}."""
    out = {}
    for lvl, (lo, hi) in LEVELS.items():
        alpha = 1 - (hi - lo)
        out[lvl] = wb_quantile(scores(y, Q[lo], Q[hi]), groups, alpha)
    ok = np.isfinite(np.asarray(y, float))
    out["n"] = int(ok.sum())
    out["n_waterbodies"] = int(len(np.unique(np.asarray(groups)[ok])))
    return out


def apply(Q: dict, offsets: dict) -> dict:
    """Conformal intervals {lvl: (lo, hi)} from raw quantiles and frozen offsets."""
    return {lvl: (np.asarray(Q[lo], float) - offsets[lvl], np.asarray(Q[hi], float) + offsets[lvl])
            for lvl, (lo, hi) in LEVELS.items()}


def crossfit_intervals(y, Q: dict, groups, n_folds: int = N_FOLDS, seed: int = 0) -> dict:
    """Out-of-fold conformal intervals on the calibration cell (folds = water bodies) for honest val metrics."""
    y, g = np.asarray(y, float), np.asarray(groups)
    ug = np.unique(g)
    rs = np.random.default_rng(seed)
    fold_of = dict(zip(ug, rs.permutation(len(ug)) % n_folds))
    f = np.array([fold_of[x] for x in g])
    out = {lvl: (np.full(len(y), np.nan), np.full(len(y), np.nan)) for lvl in LEVELS}
    for k in range(n_folds):
        tr, te = f != k, f == k
        if not te.any() or not tr.any():
            continue
        off = fit(y[tr], {q: v[tr] for q, v in Q.items()}, g[tr])
        iv = apply({q: v[te] for q, v in Q.items()}, off)
        for lvl in LEVELS:
            out[lvl][0][te], out[lvl][1][te] = iv[lvl]
    return out


def coverage(y, lo, hi) -> float:
    ok = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    return float(np.mean((y[ok] >= lo[ok]) & (y[ok] <= hi[ok]))) if ok.any() else np.nan
