# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Pooled and within-region skill for out-of-fold predictions (protocol §5).

* pooled R² — ``sentinel.eval.metrics.r2`` over all rows;
* per-region R² — R² inside each region with that region's observed mean as reference;
* anomaly R² — R² of (y − ȳ_region) vs (p − p̄_region): within-region skill net of region bias;
* between-region R² — R² of the region means (diagnostic).

CIs: ``sentinel.eval.bootstrap`` with resampling unit = location cluster (primary) and = ecoregion
(coarse); paired comparisons with ``paired_bootstrap`` on the same rows.
"""

from __future__ import annotations

import numpy as np
from sentinel.eval import bootstrap as EB
from sentinel.eval import metrics as EM


def _clean(y, p, region):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    region = np.asarray(region)
    ok = np.isfinite(y) & np.isfinite(p)
    return y[ok], p[ok], region[ok]


def _codes(region):
    _, inv = np.unique(np.asarray(region).astype(str), return_inverse=True)
    return inv


def anomaly_r2(y, p, region) -> float:
    y, p, region = _clean(y, p, region)
    if y.size < 3:
        return np.nan
    c = _codes(region)
    n = np.bincount(c)
    ya = y - (np.bincount(c, y) / n)[c]
    pa = p - (np.bincount(c, p) / n)[c]
    den = np.sum(ya ** 2)
    return float(1.0 - np.sum((ya - pa) ** 2) / den) if den > 0 else np.nan


def _region_r2_values(y, p, region, min_n: int = 10) -> tuple[np.ndarray, np.ndarray]:
    labels, c = np.unique(np.asarray(region).astype(str), return_inverse=True)
    n = np.bincount(c, minlength=len(labels))
    ym = np.bincount(c, y, minlength=len(labels)) / np.maximum(n, 1)
    sse = np.bincount(c, (y - p) ** 2, minlength=len(labels))
    sst = np.bincount(c, (y - ym[c]) ** 2, minlength=len(labels))
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = np.where((n >= min_n) & (sst > 0), 1.0 - sse / sst, np.nan)
    return labels, r2


def per_region_r2(y, p, region, min_n: int = 10) -> dict:
    y, p, region = _clean(y, p, region)
    labels, r2 = _region_r2_values(y, p, region, min_n)
    return {str(g): float(v) for g, v in zip(labels, r2)}


def between_region_r2(y, p, region) -> float:
    y, p, region = _clean(y, p, region)
    c = _codes(region)
    n = np.bincount(c)
    if len(n) < 3:
        return np.nan
    return EM.r2(np.bincount(c, y) / n, np.bincount(c, p) / n)


def summary(y, p, region) -> dict:
    pr = per_region_r2(y, p, region)
    vals = np.array([v for v in pr.values() if np.isfinite(v)])
    yc, pc, _ = _clean(y, p, region)
    return {
        "n": int(yc.size), "r2": EM.r2(yc, pc), "rmse": EM.rmse(yc, pc), "spearman": EM.spearman(yc, pc),
        "r2_anomaly": anomaly_r2(y, p, region),
        "r2_region_median": float(np.median(vals)) if vals.size else np.nan,
        "r2_region_mean": float(np.mean(vals)) if vals.size else np.nan,
        "r2_region_min": float(np.min(vals)) if vals.size else np.nan,
        "r2_region_max": float(np.max(vals)) if vals.size else np.nan,
        "r2_between_region": between_region_r2(y, p, region),
        "r2_by_region": pr,
    }


def _median_region_r2(y, p, region):
    _, v = _region_r2_values(np.asarray(y, float), np.asarray(p, float), region)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else np.nan


METRIC_FUNCS = {
    "r2": lambda y, p, r: EM.r2(y, p),
    "r2_anomaly": anomaly_r2,
    "r2_region_median": _median_region_r2,
}


def cis(y, p, region, cluster, *, n_boot: int = 2000, seed: int = 0, metrics=("r2", "r2_anomaly",
                                                                                "r2_region_median")) -> dict:
    """Cluster-bootstrap CIs for each metric: unit = ``cluster`` (location) and unit = region."""
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    region = np.asarray(region)
    cluster = np.asarray(cluster)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p, region, cluster = y[ok], p[ok], region[ok], cluster[ok]
    out = {}
    for m in metrics:
        f = METRIC_FUNCS[m]
        a = EB.bootstrap_ci(f, y, p, region, groups=cluster, n_boot=n_boot, seed=seed)
        out[m] = {"ci_low": a["ci_low"], "ci_high": a["ci_high"], "unit": "location_cluster",
                  "n_groups": a["n_groups"]}
        if m != "r2_region_median":  # resampling whole regions makes the median of regions degenerate
            b = EB.bootstrap_ci(f, y, p, region, groups=region, n_boot=n_boot, seed=seed)
            out[m]["ci_ecoregion"] = [b["ci_low"], b["ci_high"]]
    return out


def paired_vs(y, p_model, p_ref, region, cluster, *, metric: str = "r2", n_boot: int = 2000, seed: int = 0) -> dict:
    """Paired cluster bootstrap of metric(model) − metric(reference) on the same rows."""
    y = np.asarray(y, float)
    a = np.asarray(p_model, float)
    b = np.asarray(p_ref, float)
    ok = np.isfinite(y) & np.isfinite(a) & np.isfinite(b)
    region = np.asarray(region)[ok]
    f = METRIC_FUNCS[metric]
    # paired_bootstrap passes (y, pred); close over the region labels via index arrays
    idx = np.arange(ok.sum())

    def mfun(yy, pp):
        return f(yy[:, 0], pp, region[yy[:, 1].astype(int)])

    Y = np.column_stack([y[ok], idx])
    return EB.paired_bootstrap(mfun, Y, a[ok], b[ok], groups=np.asarray(cluster)[ok], n_boot=n_boot, seed=seed)
