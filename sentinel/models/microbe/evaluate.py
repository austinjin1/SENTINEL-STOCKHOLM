# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Skill metrics, CIs, permutation null and site-identity probe for SENTINEL-Microbe (protocol §5).

* pooled R² over all out-of-fold rows (``sentinel.eval.metrics.r2``), RMSE, Spearman;
* per-domain R² (each held-out domain against its own observed mean; domains with < 10 rows or
  zero variance are NaN) and their median;
* anomaly R² — within-domain skill net of domain-level bias (a constant-per-domain predictor = 0);
* MSE skill vs a reference model on the same rows: 1 − MSE(model)/MSE(ref).

CIs: percentile cluster bootstrap with **site** as the resampling unit (primary; 30-odd sites), and
a coarse domain-unit CI in ``extra``. Paired comparisons: ``paired_bootstrap`` on the same rows.
"""

from __future__ import annotations

import numpy as np

from sentinel.eval import bootstrap as EB
from sentinel.eval import metrics as EM


def _codes(region):
    _, inv = np.unique(np.asarray(region).astype(str), return_inverse=True)
    return inv


def anomaly_r2(y, p, region) -> float:
    y, p, region = np.asarray(y, float), np.asarray(p, float), np.asarray(region)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p, region = y[ok], p[ok], region[ok]
    if y.size < 3:
        return np.nan
    c = _codes(region)
    n = np.bincount(c)
    ya = y - (np.bincount(c, y) / n)[c]
    pa = p - (np.bincount(c, p) / n)[c]
    den = np.sum(ya ** 2)
    return float(1.0 - np.sum((ya - pa) ** 2) / den) if den > 0 else np.nan


def per_region_r2(y, p, region, min_n: int = 10) -> dict:
    y, p, region = np.asarray(y, float), np.asarray(p, float), np.asarray(region).astype(str)
    ok = np.isfinite(y) & np.isfinite(p)
    out = {}
    for g in np.unique(region[ok]):
        m = ok & (region == g)
        out[str(g)] = EM.r2(y[m], p[m]) if m.sum() >= min_n else np.nan
    return out


def median_region_r2(y, p, region) -> float:
    v = np.array([x for x in per_region_r2(y, p, region).values() if np.isfinite(x)])
    return float(np.median(v)) if v.size else np.nan


def mse_skill(y, p, p_ref) -> float:
    y, p, q = (np.asarray(a, float) for a in (y, p, p_ref))
    ok = np.isfinite(y) & np.isfinite(p) & np.isfinite(q)
    if ok.sum() < 3:
        return np.nan
    den = np.mean((q[ok] - y[ok]) ** 2)
    return float(1.0 - np.mean((p[ok] - y[ok]) ** 2) / den) if den > 0 else np.nan


def summary(y, p, region) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    ok = np.isfinite(y) & np.isfinite(p)
    pr = per_region_r2(y, p, region)
    vals = np.array([v for v in pr.values() if np.isfinite(v)])
    return {"n": int(ok.sum()), "r2": EM.r2(y[ok], p[ok]), "rmse": EM.rmse(y[ok], p[ok]),
            "spearman": EM.spearman(y[ok], p[ok]), "r2_anomaly": anomaly_r2(y, p, region),
            "r2_region_median": float(np.median(vals)) if vals.size else np.nan,
            "r2_by_region": pr}


METRIC_FUNCS = {"r2": lambda y, p, r: EM.r2(y, p), "r2_anomaly": anomaly_r2,
                "r2_region_median": median_region_r2}


def cis(y, p, region, site, *, n_boot: int = 2000, seed: int = 0, metrics=("r2", "r2_anomaly")) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p, region, site = y[ok], p[ok], np.asarray(region)[ok], np.asarray(site)[ok]
    out = {}
    for m in metrics:
        f = METRIC_FUNCS[m]
        a = EB.bootstrap_ci(f, y, p, region, groups=site, n_boot=n_boot, seed=seed)
        b = EB.bootstrap_ci(f, y, p, region, groups=region, n_boot=n_boot, seed=seed)
        out[m] = {"ci_low": a["ci_low"], "ci_high": a["ci_high"], "unit": "site", "n_groups": a["n_groups"],
                  "ci_domain": [b["ci_low"], b["ci_high"]]}
    return out


def paired_vs(y, p_model, p_ref, region, site, *, metric: str = "r2", n_boot: int = 2000, seed: int = 0) -> dict:
    """Paired site-cluster bootstrap of metric(model) − metric(ref) on rows where both predict."""
    y, a, b = (np.asarray(v, float) for v in (y, p_model, p_ref))
    ok = np.isfinite(y) & np.isfinite(a) & np.isfinite(b)
    region = np.asarray(region)[ok]
    f = METRIC_FUNCS[metric]
    idx = np.arange(ok.sum())

    def mfun(yy, pp):
        return f(yy[:, 0], pp, region[yy[:, 1].astype(int)])

    Y = np.column_stack([y[ok], idx])
    res = EB.paired_bootstrap(mfun, Y, a[ok], b[ok], groups=np.asarray(site)[ok], n_boot=n_boot, seed=seed)
    res["n"] = int(ok.sum())
    return res


# ------------------------------------------------------------------------------ site-identity probe

def site_identity_probe(X: np.ndarray, site: np.ndarray, *, seed: int = 0, n_folds: int = 5, n_jobs: int = 8,
                        model: str = "logreg") -> dict:
    """Can the community recover *which site* a sample came from? Stratified k-fold over rows
    (samples of one site appear on both sides by design — the probe asks whether site identity is
    encoded, which is what makes random splits invalid). Returns accuracy, balanced accuracy and
    chance (majority-class rate and 1/n_sites)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    site = np.asarray(site).astype(str)
    keep = np.isin(site, [s for s, c in zip(*np.unique(site, return_counts=True)) if c >= n_folds])
    X, site = X[keep], site[keep]
    pred = np.empty(len(site), dtype=object)
    for tr, te in StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(X, site):
        if model == "logreg":
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
        else:
            from sklearn.ensemble import RandomForestClassifier

            clf = RandomForestClassifier(500, min_samples_leaf=1, random_state=seed, n_jobs=n_jobs)
        pred[te] = clf.fit(X[tr], site[tr]).predict(X[te])
    acc = float(np.mean(pred == site))
    _, cnt = np.unique(site, return_counts=True)
    return {"accuracy": acc, "balanced_accuracy": float(balanced_accuracy_score(site, pred.astype(str))),
            "chance_majority": float(cnt.max() / cnt.sum()), "chance_uniform": float(1 / len(cnt)),
            "n": len(site), "n_sites": len(cnt), "model": model}
