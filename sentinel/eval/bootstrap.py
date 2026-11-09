# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Cluster (block) bootstrap confidence intervals and paired model comparisons.

Observations from one site, lake or study are not independent, so resampling is done
over *groups* by default. v1 reported i.i.d. bootstrap intervals over pseudo-replicated
windows; v2 always states the resampling unit.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def _group_index(groups, n: int):
    if groups is None:
        return [np.array([i]) for i in range(n)]
    groups = np.asarray(groups)
    _, inv = np.unique(groups, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    bounds = np.flatnonzero(np.diff(inv[order])) + 1
    return np.split(order, bounds)


def bootstrap_ci(metric: Callable, *arrays, groups=None, n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0, return_samples: bool = False):
    """Percentile CI of ``metric(*arrays)`` resampling groups with replacement.

    Returns dict(point, ci_low, ci_high, n_boot_valid, unit[, samples]).
    """
    arrays = [np.asarray(a) for a in arrays]
    n = len(arrays[0])
    idx_groups = _group_index(groups, n)
    rs = np.random.default_rng(seed)
    point = float(metric(*arrays))
    samples = np.empty(n_boot)
    for b in range(n_boot):
        pick = rs.integers(0, len(idx_groups), len(idx_groups))
        idx = np.concatenate([idx_groups[g] for g in pick])
        samples[b] = metric(*(a[idx] for a in arrays))
    ok = samples[np.isfinite(samples)]
    out = {
        "point": point,
        "ci_low": float(np.quantile(ok, alpha / 2)) if ok.size else np.nan,
        "ci_high": float(np.quantile(ok, 1 - alpha / 2)) if ok.size else np.nan,
        "n_boot_valid": int(ok.size),
        "unit": "group" if groups is not None else "row",
        "n_groups": len(idx_groups),
    }
    if return_samples:
        out["samples"] = samples
    return out


def paired_bootstrap(metric: Callable, y, pred_a, pred_b, *, groups=None, n_boot: int = 2000,
                     alpha: float = 0.05, seed: int = 0, higher_is_better: bool = True):
    """CI and two-sided p-value for metric(y, pred_a) - metric(y, pred_b) on the SAME rows.

    The p-value is the bootstrap estimate of P(difference has the opposite sign), doubled
    and capped at 1 — adequate for reporting, not a substitute for pre-registration.
    """
    y, pa, pb = (np.asarray(a) for a in (y, pred_a, pred_b))
    idx_groups = _group_index(groups, len(y))
    rs = np.random.default_rng(seed)
    point = float(metric(y, pa) - metric(y, pb))
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        pick = rs.integers(0, len(idx_groups), len(idx_groups))
        idx = np.concatenate([idx_groups[g] for g in pick])
        diffs[b] = metric(y[idx], pa[idx]) - metric(y[idx], pb[idx])
    ok = diffs[np.isfinite(diffs)]
    if not ok.size:
        return {"diff": point, "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan}
    p_le = np.mean(ok <= 0)
    p_ge = np.mean(ok >= 0)
    better = point > 0 if higher_is_better else point < 0
    return {
        "diff": point,
        "ci_low": float(np.quantile(ok, alpha / 2)),
        "ci_high": float(np.quantile(ok, 1 - alpha / 2)),
        "p_value": float(min(1.0, 2 * min(p_le, p_ge))),
        "a_better": bool(better),
        "unit": "group" if groups is not None else "row",
        "n_groups": len(idx_groups),
        "n_boot_valid": int(ok.size),
    }


def holm_bonferroni(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Holm step-down correction; returns {name: rejected?}."""
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    rejected = {}
    still = True
    for i, (k, p) in enumerate(items):
        thr = alpha / (m - i)
        still = still and (p <= thr)
        rejected[k] = bool(still)
    return rejected
