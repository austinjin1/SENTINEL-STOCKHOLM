# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Scoring for SENTINEL-Sonde quantile forecasts (identical rows for every model).

Per-sample scores are stored so that any two models can later be compared with a paired,
site-clustered bootstrap (sentinel.eval.bootstrap.paired_bootstrap).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ...eval import metrics as M
from .data import LOG_VARS, from_model_space, to_model_space

QUANTILES = np.round(np.arange(0.05, 0.951, 0.05), 2)


def pinball_crps(y: np.ndarray, q: np.ndarray, levels=QUANTILES) -> np.ndarray:
    """Per-sample CRPS approximation (2 × mean pinball loss); y (...,), q (..., K)."""
    diff = y[..., None] - q
    loss = np.maximum(levels * diff, (levels - 1) * diff)
    return 2.0 * loss.mean(axis=-1)


def score_samples(var: str, q_model: np.ndarray, obs_model: np.ndarray, mask: np.ndarray,
                  sites: list[str], dates: pd.DatetimeIndex, t_idx: np.ndarray, h: int,
                  model: str) -> pd.DataFrame:
    """Score quantile forecasts in ORIGINAL units on the masked rows; returns a tidy frame."""
    q_sorted = np.sort(q_model, axis=-1)  # guard against quantile crossing
    q_orig = from_model_space(var, q_sorted)
    y_orig = from_model_space(var, obs_model)
    si, ti = np.nonzero(mask)
    q = q_orig[si, ti]
    y = y_orig[si, ti]
    med = q[:, QUANTILES.tolist().index(0.5)]
    out = pd.DataFrame({
        "model": model, "var": var, "h": h,
        "site_no": np.asarray(sites)[si],
        "issue_date": dates[t_idx[ti]],
        "obs": y.astype("float32"), "median": med.astype("float32"),
        "q05": q[:, 0].astype("float32"), "q95": q[:, -1].astype("float32"),
        "crps": pinball_crps(y, q).astype("float32"),
        "ae": np.abs(med - y).astype("float32"),
        "se": ((med - y) ** 2).astype("float32"),
        "in90": ((y >= q[:, 0]) & (y <= q[:, -1])),
    })
    return out


def site_metrics(df: pd.DataFrame, min_n: int = 30) -> pd.DataFrame:
    """Per (model, var, h, site): mean CRPS, MAE, RMSE, NSE, KGE, 90% coverage."""
    rows = []
    for key, g in df.groupby(["model", "var", "h", "site_no"], observed=True):
        if len(g) < min_n:
            continue
        rec = {"model": key[0], "var": key[1], "h": key[2], "site_no": key[3], "n": len(g),
               "crps": float(g["crps"].mean()), "mae": float(g["ae"].mean()),
               "rmse": float(np.sqrt(g["se"].mean())), "nse": M.nse(g["obs"], g["median"]),
               "kge": M.kge(g["obs"], g["median"]), "cov90": float(g["in90"].mean())}
        if key[1] in LOG_VARS:  # skill on the modelling (log) scale for skewed variables
            rec["nse_log"] = M.nse(to_model_space(key[1], g["obs"]), to_model_space(key[1], g["median"]))
        rows.append(rec)
    return pd.DataFrame(rows)


def skill_vs(site_df: pd.DataFrame, model: str, ref: str, metric: str = "crps") -> pd.DataFrame:
    """Per-site skill 1 - metric_model/metric_ref on sites scored by both."""
    a = site_df[site_df["model"] == model].set_index(["var", "h", "site_no"])[metric]
    b = site_df[site_df["model"] == ref].set_index(["var", "h", "site_no"])[metric]
    j = pd.concat([a.rename("m"), b.rename("r")], axis=1).dropna()
    j["skill"] = 1 - j["m"] / j["r"].where(j["r"] > 0)
    return j.reset_index()


def save_samples(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression="zstd")


# --------------------------------------------------------------------------- additions (M1 phase 1)

def plausible_bounds(var: str) -> tuple[float, float]:
    """Physical plausibility range (original units) of a daily-tier variable, from the ingest QC
    table ``sentinel.data.sonde_daily.PLAUSIBLE`` (lower bound 0 where ingest clips negatives)."""
    from ...data.sonde_daily import CLIP_ZERO, PLAUSIBLE
    from ...data.sources.usgs_legacy import ALL_PARAMS

    base = var.rsplit("_", 1)[0]
    codes = [c for c, name in ALL_PARAMS.items() if name == base]
    if not codes or codes[0] not in PLAUSIBLE:
        return -np.inf, np.inf
    lo, hi = PLAUSIBLE[codes[0]]
    if codes[0] in CLIP_ZERO:
        lo = max(lo, 0.0)
    return float(lo), float(hi)


def clip_plausible(var: str, q_model: np.ndarray) -> np.ndarray:
    """Clip model-space quantiles to the plausible range (protocol §6; identical for all models)."""
    lo, hi = plausible_bounds(var)
    zlo = to_model_space(var, np.array([lo]))[0] if np.isfinite(lo) else -np.inf
    zhi = to_model_space(var, np.array([hi]))[0] if np.isfinite(hi) else np.inf
    return np.clip(np.asarray(q_model, dtype="float64"), zlo, zhi)


def score_rows(var: str, q_model: np.ndarray, obs_model: np.ndarray, site_no, issue_date, h: int,
               model: str, extra: dict | None = None, clip: bool = True) -> pd.DataFrame:
    """Row-wise twin of :func:`score_samples` for sparse forecast tables: q_model (n, 19) and
    obs_model (n,) in model space for the rows to score. Identical columns and maths (checked by
    tests/test_sonde_models.py) except that quantiles are first clipped to the plausible range
    (``clip``, default on for every model); ``extra`` adds columns such as the site role."""
    q_in = clip_plausible(var, q_model) if clip else np.asarray(q_model, dtype="float64")
    q_sorted = np.sort(q_in, axis=-1)
    q = from_model_space(var, q_sorted)
    y = from_model_space(var, np.asarray(obs_model, dtype="float64"))
    med = q[:, QUANTILES.tolist().index(0.5)]
    out = pd.DataFrame({
        "model": model, "var": var, "h": h,
        "site_no": np.asarray(site_no),
        "issue_date": pd.to_datetime(np.asarray(issue_date)),
        "obs": y.astype("float32"), "median": med.astype("float32"),
        "q05": q[:, 0].astype("float32"), "q95": q[:, -1].astype("float32"),
        "crps": pinball_crps(y, q).astype("float32"),
        "ae": np.abs(med - y).astype("float32"),
        "se": ((med - y) ** 2).astype("float32"),
        "in90": ((y >= q[:, 0]) & (y <= q[:, -1])),
    })
    for k, v in (extra or {}).items():
        out[k] = v
    return out


def score_forecast(var: str, q: np.ndarray, score_mask: dict, obs: dict, site_no, issue_date, role,
                   horizons, model: str) -> pd.DataFrame:
    """Score a (n, H, 19) forecast on the common scorable rows of every horizon."""
    site_no, issue_date, role = np.asarray(site_no), np.asarray(issue_date), np.asarray(role)
    frames = []
    for k, h in enumerate(horizons):
        m = np.asarray(score_mask[h], bool)
        if not m.any():
            continue
        qk = q[m, k]
        if not np.isfinite(qk).all():
            raise ValueError(f"{model}/{var}/h={h}: {int((~np.isfinite(qk).all(axis=1)).sum())} "
                             "non-finite forecasts on common scorable rows")
        frames.append(score_rows(var, qk, obs[h][m], site_no[m], issue_date[m], h, model,
                                 extra={"role": role[m]}))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def skill_summary(site_df: pd.DataFrame, model: str, refs=("damped_anomaly", "climatology", "persistence"),
                  metric: str = "crps") -> pd.DataFrame:
    """Per (var, h): number of sites, median across sites of the per-site skill vs each reference,
    and the median 90% coverage of ``model``. Descriptive only (no CIs)."""
    rows = []
    sub = site_df[site_df["model"] == model]
    for (v, h), g in sub.groupby(["var", "h"]):
        rec = {"model": model, "var": v, "h": int(h), "n_sites": int(g["site_no"].nunique()),
               f"median_{metric}": float(g[metric].median()), "median_cov90": float(g["cov90"].median())}
        for r in refs:
            s = skill_vs(site_df[(site_df["var"] == v) & (site_df["h"] == h)], model, r, metric)
            rec[f"median_skill_vs_{r}"] = float(s["skill"].median()) if len(s) else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def diebold_mariano(d: np.ndarray, max_lag: int) -> dict:
    """Diebold–Mariano test on a loss-differential series d_t (model − ref; negative = model better)
    with a Newey–West (Bartlett) HAC variance, lag ``max_lag`` (≥ h − 1 for h-step forecasts),
    Harvey–Leybourne–Newbold small-sample correction, two-sided normal/t p-value."""
    from scipy.stats import t as tdist

    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    n = d.size
    if n < 10:
        return {"dm": np.nan, "p_value": np.nan, "n": n}
    m = d.mean()
    e = d - m
    var = e @ e / n
    for k in range(1, max_lag + 1):
        w = 1 - k / (max_lag + 1)
        var += 2 * w * (e[k:] @ e[:-k]) / n
    if var <= 0:
        return {"dm": np.nan, "p_value": np.nan, "n": n}
    dm = m / np.sqrt(var / n)
    h = max_lag + 1
    corr = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = dm * corr
    return {"dm": float(stat), "p_value": float(2 * tdist.sf(abs(stat), df=n - 1)), "n": int(n), "mean_d": float(m)}
