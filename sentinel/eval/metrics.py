# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Evaluation metrics used across SENTINEL v2.

All functions take array-likes, drop pairs where either side is non-finite, and return
``nan`` (never raise) when a metric is undefined (e.g. AUROC with a single class), so
undefined results are visible instead of silently replaced.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def _pair(y, p):
    y = np.asarray(y, dtype=float).ravel()
    p = np.asarray(p, dtype=float).ravel()
    if y.shape != p.shape:
        raise ValueError(f"shape mismatch {y.shape} vs {p.shape}")
    m = np.isfinite(y) & np.isfinite(p)
    return y[m], p[m]


# --------------------------------------------------------------------------- regression

def rmse(y, p) -> float:
    y, p = _pair(y, p)
    return float(np.sqrt(np.mean((p - y) ** 2))) if y.size else np.nan


def mae(y, p) -> float:
    y, p = _pair(y, p)
    return float(np.mean(np.abs(p - y))) if y.size else np.nan


def bias(y, p) -> float:
    """Mean error (prediction minus observation)."""
    y, p = _pair(y, p)
    return float(np.mean(p - y)) if y.size else np.nan


def nse(y, p) -> float:
    """Nash–Sutcliffe efficiency (identical to the coefficient of determination R²)."""
    y, p = _pair(y, p)
    if y.size < 2:
        return np.nan
    denom = np.sum((y - y.mean()) ** 2)
    return float(1.0 - np.sum((p - y) ** 2) / denom) if denom > 0 else np.nan


r2 = nse


def pearson(y, p) -> float:
    y, p = _pair(y, p)
    if y.size < 3 or y.std() == 0 or p.std() == 0:
        return np.nan
    return float(np.corrcoef(y, p)[0, 1])


def spearman(y, p) -> float:
    from scipy.stats import spearmanr

    y, p = _pair(y, p)
    if y.size < 3:
        return np.nan
    return float(spearmanr(y, p).statistic)


def kge(y, p) -> float:
    """Kling–Gupta efficiency (Gupta et al., 2009)."""
    y, p = _pair(y, p)
    if y.size < 3 or y.std() == 0 or abs(y.mean()) < _EPS:
        return np.nan
    r = np.corrcoef(y, p)[0, 1] if p.std() > 0 else 0.0
    alpha = p.std() / y.std()
    beta = p.mean() / y.mean()
    return float(1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


# ------------------------------------------------------- log-space (bio-optical) metrics

def _pos_pair(y, p):
    y, p = _pair(y, p)
    m = (y > 0) & (p > 0)
    return y[m], p[m]


def mdsa(y, p) -> float:
    """Median symmetric accuracy, % (Morley et al., 2018), standard in ocean-colour work."""
    y, p = _pos_pair(y, p)
    if not y.size:
        return np.nan
    return float(100.0 * (np.exp(np.median(np.abs(np.log(p / y)))) - 1.0))


def sspb(y, p) -> float:
    """Symmetric signed percentage bias, % (Morley et al., 2018)."""
    y, p = _pos_pair(y, p)
    if not y.size:
        return np.nan
    m = np.median(np.log(p / y))
    return float(100.0 * np.sign(m) * (np.exp(np.abs(m)) - 1.0))


def log_rmse(y, p) -> float:
    y, p = _pos_pair(y, p)
    return float(np.sqrt(np.mean((np.log10(p) - np.log10(y)) ** 2))) if y.size else np.nan


def log_mae(y, p) -> float:
    y, p = _pos_pair(y, p)
    return float(np.mean(np.abs(np.log10(p) - np.log10(y)))) if y.size else np.nan


def log_slope(y, p) -> float:
    """OLS slope of log10(pred) on log10(obs); 1 is ideal."""
    y, p = _pos_pair(y, p)
    if y.size < 3:
        return np.nan
    ly, lp = np.log10(y), np.log10(p)
    vy = np.var(ly)
    return float(np.cov(ly, lp, bias=True)[0, 1] / vy) if vy > 0 else np.nan


def log_r2(y, p) -> float:
    y, p = _pos_pair(y, p)
    return nse(np.log10(y), np.log10(p)) if y.size else np.nan


# --------------------------------------------------------------------- probabilistic

def crps_gaussian(y, mu, sigma) -> float:
    """Mean CRPS of Gaussian predictive distributions (closed form)."""
    from scipy.stats import norm

    y = np.asarray(y, float).ravel()
    mu = np.asarray(mu, float).ravel()
    sigma = np.maximum(np.asarray(sigma, float).ravel(), _EPS)
    m = np.isfinite(y) & np.isfinite(mu) & np.isfinite(sigma)
    if not m.any():
        return np.nan
    z = (y[m] - mu[m]) / sigma[m]
    crps = sigma[m] * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
    return float(np.mean(crps))


def crps_ensemble(y, samples) -> float:
    """Mean CRPS from predictive samples, ``samples`` shape (n, m); exact for the ensemble."""
    y = np.asarray(y, float).ravel()
    s = np.sort(np.asarray(samples, float), axis=1)
    if s.shape[0] != y.size:
        raise ValueError("samples must have one row per observation")
    m = np.isfinite(y) & np.isfinite(s).all(axis=1)
    y, s = y[m], s[m]
    if not y.size:
        return np.nan
    k = s.shape[1]
    term1 = np.mean(np.abs(s - y[:, None]), axis=1)
    # E|X - X'| for sorted samples: (2 / k^2) * sum_i (2i - k - 1) x_(i)
    i = np.arange(1, k + 1)
    term2 = (2.0 / k**2) * np.sum((2 * i - k - 1) * s, axis=1)
    return float(np.mean(term1 - 0.5 * term2))


def pinball_loss(y, q_pred, quantiles) -> float:
    """Mean quantile (pinball) loss over observations and quantile levels."""
    y = np.asarray(y, float).ravel()
    q_pred = np.asarray(q_pred, float)
    qs = np.asarray(quantiles, float)
    m = np.isfinite(y) & np.isfinite(q_pred).all(axis=1)
    diff = y[m, None] - q_pred[m]
    loss = np.maximum(qs * diff, (qs - 1) * diff)
    return float(np.mean(loss)) if m.any() else np.nan


def crps_from_quantiles(y, q_pred, quantiles) -> float:
    """CRPS approximated as twice the mean pinball loss over evenly spaced quantiles."""
    return 2.0 * pinball_loss(y, q_pred, quantiles)


def interval_coverage(y, lo, hi) -> float:
    y, lo, hi = (np.asarray(a, float).ravel() for a in (y, lo, hi))
    m = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    return float(np.mean((y[m] >= lo[m]) & (y[m] <= hi[m]))) if m.any() else np.nan


def interval_width(lo, hi) -> float:
    lo, hi = np.asarray(lo, float).ravel(), np.asarray(hi, float).ravel()
    m = np.isfinite(lo) & np.isfinite(hi)
    return float(np.mean(hi[m] - lo[m])) if m.any() else np.nan


def interval_score(y, lo, hi, alpha: float) -> float:
    """Gneiting & Raftery (2007) interval score for a central (1-alpha) interval; lower is better."""
    y, lo, hi = (np.asarray(a, float).ravel() for a in (y, lo, hi))
    m = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    y, lo, hi = y[m], lo[m], hi[m]
    if not y.size:
        return np.nan
    s = (hi - lo) + (2 / alpha) * (lo - y) * (y < lo) + (2 / alpha) * (y - hi) * (y > hi)
    return float(np.mean(s))


# --------------------------------------------------------------------- classification

def _bin_pair(y, s):
    y = np.asarray(y, float).ravel()
    s = np.asarray(s, float).ravel()
    m = np.isfinite(y) & np.isfinite(s)
    return y[m].astype(int), s[m]


def auroc(y, s) -> float:
    from sklearn.metrics import roc_auc_score

    y, s = _bin_pair(y, s)
    if y.size == 0 or y.min() == y.max():
        return np.nan
    return float(roc_auc_score(y, s))


def auprc(y, s) -> float:
    """Average precision (area under precision–recall); chance level = prevalence."""
    from sklearn.metrics import average_precision_score

    y, s = _bin_pair(y, s)
    if y.size == 0 or y.max() == 0:
        return np.nan
    return float(average_precision_score(y, s))


def brier(y, p) -> float:
    y, p = _bin_pair(y, p)
    return float(np.mean((p - y) ** 2)) if y.size else np.nan


def brier_skill_score(y, p, p_ref) -> float:
    """BSS = 1 - BS/BS_ref. ``p_ref`` is a scalar climatology or an array (e.g. seasonal)."""
    y_arr = np.asarray(y, float).ravel()
    p_arr = np.asarray(p, float).ravel()
    ref = np.broadcast_to(np.asarray(p_ref, float), y_arr.shape)
    m = np.isfinite(y_arr) & np.isfinite(p_arr) & np.isfinite(ref)
    if not m.any():
        return np.nan
    bs = np.mean((p_arr[m] - y_arr[m]) ** 2)
    bs_ref = np.mean((ref[m] - y_arr[m]) ** 2)
    return float(1 - bs / bs_ref) if bs_ref > 0 else np.nan


def log_loss(y, p) -> float:
    y, p = _bin_pair(y, p)
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))) if y.size else np.nan


def reliability_curve(y, p, n_bins: int = 10, strategy: str = "quantile"):
    """Returns (mean predicted, observed frequency, count) per bin."""
    y, p = _bin_pair(y, p)
    if not y.size:
        return np.array([]), np.array([]), np.array([])
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    mp, fo, cnt = [], [], []
    for b in range(len(edges) - 1):
        sel = idx == b
        if sel.any():
            mp.append(p[sel].mean())
            fo.append(y[sel].mean())
            cnt.append(sel.sum())
    return np.array(mp), np.array(fo), np.array(cnt)


def ece(y, p, n_bins: int = 10, strategy: str = "quantile") -> float:
    """Expected calibration error (count-weighted |observed - predicted| over bins)."""
    mp, fo, cnt = reliability_curve(y, p, n_bins, strategy)
    if not cnt.size:
        return np.nan
    return float(np.sum(cnt * np.abs(fo - mp)) / cnt.sum())


# ------------------------------------------------------------ contingency (yes/no) scores

def contingency(y_true, y_alarm) -> dict[str, int]:
    t = np.asarray(y_true).astype(bool).ravel()
    a = np.asarray(y_alarm).astype(bool).ravel()
    return {
        "tp": int(np.sum(t & a)),
        "fp": int(np.sum(~t & a)),
        "fn": int(np.sum(t & ~a)),
        "tn": int(np.sum(~t & ~a)),
    }


def contingency_scores(y_true, y_alarm) -> dict[str, float]:
    """POD (hit rate), FAR (false-alarm RATIO), POFD (false-alarm RATE), CSI, ETS, PSS, bias."""
    c = contingency(y_true, y_alarm)
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    n = tp + fp + fn + tn

    def div(a, b):
        return float(a / b) if b > 0 else np.nan

    pod = div(tp, tp + fn)
    pofd = div(fp, fp + tn)
    hits_random = div((tp + fp) * (tp + fn), n)
    out = {
        **{k: float(v) for k, v in c.items()},
        "n": float(n),
        "pod": pod,
        "far": div(fp, tp + fp),
        "pofd": pofd,
        "csi": div(tp, tp + fp + fn),
        "ets": div(tp - hits_random, tp + fp + fn - hits_random) if np.isfinite(hits_random) else np.nan,
        "pss": pod - pofd if np.isfinite(pod) and np.isfinite(pofd) else np.nan,
        "freq_bias": div(tp + fp, tp + fn),
    }
    return out


REGRESSION_METRICS = {
    "rmse": rmse, "mae": mae, "bias": bias, "nse": nse, "kge": kge, "pearson": pearson,
}
LOG_METRICS = {
    "mdsa": mdsa, "sspb": sspb, "log_rmse": log_rmse, "log_mae": log_mae, "log_slope": log_slope,
    "log_r2": log_r2,
}
BINARY_METRICS = {"auroc": auroc, "auprc": auprc, "brier": brier, "log_loss": log_loss, "ece": ece}


def regression_report(y, p, *, log_space: bool = False) -> dict[str, float]:
    out = {k: f(y, p) for k, f in REGRESSION_METRICS.items()}
    if log_space:
        out.update({k: f(y, p) for k, f in LOG_METRICS.items()})
    out["n"] = float(_pair(y, p)[0].size)
    return out


def binary_report(y, p, *, p_ref=None) -> dict[str, float]:
    out = {k: f(y, p) for k, f in BINARY_METRICS.items()}
    yy, _ = _bin_pair(y, p)
    out["n"] = float(yy.size)
    out["prevalence"] = float(yy.mean()) if yy.size else np.nan
    if p_ref is not None:
        out["bss"] = brier_skill_score(y, p, p_ref)
    return out
