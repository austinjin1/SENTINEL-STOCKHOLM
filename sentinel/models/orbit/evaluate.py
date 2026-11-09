# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""SENTINEL-Orbit scoring (orbit_protocol.md §5, §7): bio-optical metrics with lake-cluster bootstrap CIs,
paired MdSA differences, interval coverage, within-lake skill, bloom-class detection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...eval import metrics as M
from ...eval.bootstrap import _group_index

N_BOOT = 2000
BLOOM_THRESHOLDS = (10.0, 50.0)   # µg/L chl-a (WHO 2003 recreational guidance)


def mdsa_log10(err_log10: np.ndarray) -> float:
    """MdSA from log10 errors: 100*(exp(median|ln Q|)-1). Raises on non-finite errors (pre-training review C2:
    a silent NaN once disabled a comparator's model selection) — callers mask rows explicitly."""
    e = np.abs(np.asarray(err_log10, float)) * np.log(10)
    if not np.all(np.isfinite(e)):
        raise ValueError(f"mdsa_log10: {int((~np.isfinite(e)).sum())} non-finite errors; mask rows first")
    return float(100.0 * (np.exp(np.median(e)) - 1.0)) if e.size else np.nan


def point_metrics(y_log: np.ndarray, p_log: np.ndarray) -> dict:
    y, p = 10 ** y_log, 10 ** p_log
    return {"mdsa": M.mdsa(y, p), "sspb": M.sspb(y, p), "slope": M.log_slope(y, p), "rmsle": M.log_rmse(y, p),
            "log_mae": M.log_mae(y, p), "log_r2": M.log_r2(y, p), "n": int(len(y))}


def _boot_indices(groups, n_boot, seed):
    idx_groups = _group_index(groups, len(groups))
    rs = np.random.default_rng(seed)
    for _ in range(n_boot):
        pick = rs.integers(0, len(idx_groups), len(idx_groups))
        yield np.concatenate([idx_groups[g] for g in pick])


def cluster_ci(stat, arrays, groups, n_boot=N_BOOT, seed=0, alpha=0.05):
    vals = np.array([stat(*(a[ix] for a in arrays)) for ix in _boot_indices(groups, n_boot, seed)])
    ok = vals[np.isfinite(vals)]
    return (float(np.quantile(ok, alpha / 2)), float(np.quantile(ok, 1 - alpha / 2))) if ok.size else (np.nan, np.nan)


def paired_mdsa(y_log, pa_log, pb_log, groups, n_boot=N_BOOT, seed=0):
    """MdSA(a) - MdSA(b) on identical rows; negative = a better. Cluster bootstrap CI + p (two-sided)."""
    ea, eb = pa_log - y_log, pb_log - y_log
    point = mdsa_log10(ea) - mdsa_log10(eb)
    d = np.array([mdsa_log10(ea[ix]) - mdsa_log10(eb[ix]) for ix in _boot_indices(groups, n_boot, seed)])
    lo, hi = np.quantile(d, [0.025, 0.975])
    p = min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0)))
    return {"diff": float(point), "ci_low": float(lo), "ci_high": float(hi), "p_value": float(p),
            "n_clusters": int(len(np.unique(groups))), "n": int(len(y_log))}


def score_models(y_log: np.ndarray, preds: dict[str, np.ndarray], groups: np.ndarray, *, n_boot=N_BOOT,
                 seed=0) -> pd.DataFrame:
    """Per-model metrics on the common finite rows, MdSA CI by cluster bootstrap."""
    ok = np.isfinite(y_log)
    for p in preds.values():
        ok &= np.isfinite(p)
    rows = []
    for name, p in preds.items():
        y, q, g = y_log[ok], p[ok], groups[ok]
        r = {"model": name, **point_metrics(y, q), "n_clusters": int(len(np.unique(g))),
             "n_dropped_nonfinite": int((np.isfinite(y_log) & ~np.isfinite(p)).sum())}
        r["mdsa_ci_low"], r["mdsa_ci_high"] = cluster_ci(lambda a, b: mdsa_log10(b - a), (y, q), g, n_boot, seed)
        rows.append(r)
    return pd.DataFrame(rows), ok


def interval_coverage_mixture(y_log, q_lo, q_hi) -> float:
    ok = np.isfinite(y_log) & np.isfinite(q_lo) & np.isfinite(q_hi)
    return float(np.mean((y_log[ok] >= q_lo[ok]) & (y_log[ok] <= q_hi[ok]))) if ok.any() else np.nan


def within_lake_spearman(y_log, p_log, lakes, min_rows: int = 8) -> dict:
    df = pd.DataFrame({"y": y_log, "p": p_log, "g": lakes}).dropna()
    cnt = df.g.value_counts()
    df = df[df.g.isin(cnt[cnt >= min_rows].index)]
    if df.empty:
        return {"rho": np.nan, "n": 0, "n_lakes": 0}
    df["ya"] = df.y - df.groupby("g").y.transform("mean")
    df["pa"] = df.p - df.groupby("g").p.transform("mean")
    return {"rho": M.spearman(df.ya.to_numpy(), df.pa.to_numpy()), "n": int(len(df)), "n_lakes": int(df.g.nunique())}


def bloom_scores(y_chla: np.ndarray, prob: np.ndarray, thr: float, fpr: float = 0.10, frozen_cut=None) -> dict:
    ok = np.isfinite(y_chla) & np.isfinite(prob)
    yb = (y_chla[ok] >= thr).astype(int)
    s = prob[ok]
    out = {"threshold_ugL": thr, "n": int(ok.sum()), "prevalence": float(yb.mean()) if ok.any() else np.nan}
    if yb.min() == yb.max():
        return out
    out["auroc"] = M.auroc(yb, s)
    out["auprc"] = M.auprc(yb, s)
    cut = frozen_cut if frozen_cut is not None else float(np.quantile(s[yb == 0], 1 - fpr))
    out["cut"] = cut
    out["recall_at_fpr"] = float(np.mean(s[yb == 1] > cut))
    out["fpr_realised"] = float(np.mean(s[yb == 0] > cut))
    return out
