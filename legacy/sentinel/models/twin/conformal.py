"""Post-hoc split-conformal calibration for the Digital Twin uncertainty head.

The Digital Twin's :class:`~sentinel.models.twin.twin_engine.ForecastHead`
emits a symmetric 90% interval ``pred +/- z * scale`` with ``z = 1.6449``.
In evaluation this interval is badly *under-covered* (empirical coverage ~14%
vs the nominal 90%), i.e. the learned ``scale`` is far too small / mis-scaled.

This module applies **split conformal prediction** as a *post-hoc* fix that
requires no retraining.  It computes nonconformity scores on a held-out
*calibration* split and uses their empirical quantile to rescale (or widen)
the intervals so that empirical coverage on a disjoint *evaluation* split
reaches the target ~90% with a finite-sample guarantee.

Two score types are supported:

* ``"normalized"`` -- scaled residual ``r_i = |y_i - yhat_i| / s_i`` where
  ``s_i`` is the model's own predicted half-width-implied scale.  The
  calibrated interval keeps the model's *adaptive* width shape but is
  globally rescaled by the conformal quantile ``q``:
  ``yhat +/- q * s_i``.  This preserves heteroscedastic structure.
* ``"absolute"`` -- raw residual ``r_i = |y_i - yhat_i|``, giving a constant
  half-width ``q`` (classic absolute-residual conformal).  A useful sanity
  baseline.

The split-conformal coverage guarantee (Vovk et al.; Lei et al. 2018) is

    P(y in C(x)) >= 1 - alpha

for exchangeable data, using the quantile level
``ceil((n_cal + 1) * (1 - alpha)) / n_cal``.

This module operates purely on cached arrays (predictions, intervals,
targets, masks) so it is decoupled from the model forward pass and trivially
testable.  Nothing here fabricates data: every number is computed from the
real cached test-set predictions/targets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

#: z-score used by ForecastHead to build its symmetric 90% interval.
FORECAST_HEAD_Z: float = 1.6449

#: Floor on the per-sample scale to avoid division blow-up / degenerate
#: normalized scores when the model emits a near-zero half-width.
_SCALE_FLOOR: float = 1e-6


def implied_scale(
    preds: np.ndarray,
    upper: np.ndarray,
    z: float = FORECAST_HEAD_Z,
) -> np.ndarray:
    """Recover the model's per-sample uncertainty scale from its interval.

    The ForecastHead builds ``upper = preds + z * scale``, so
    ``scale = (upper - preds) / z``.

    Args:
        preds: Point predictions, any shape.
        upper: Upper 90% interval bound, same shape as ``preds``.
        z: z-score the head used (default 1.6449 for nominal 90%).

    Returns:
        Per-element scale ``>= _SCALE_FLOOR`` (floored for stability).
    """
    scale = (upper - preds) / z
    return np.maximum(scale, _SCALE_FLOOR)


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample split-conformal quantile of nonconformity scores.

    Uses the conservative level ``ceil((n + 1) * (1 - alpha)) / n`` which
    yields the standard ``>= 1 - alpha`` marginal coverage guarantee.

    Args:
        scores: 1-D array of nonconformity scores from the calibration set.
        alpha: Miscoverage level (e.g. 0.10 for a 90% interval).

    Returns:
        The conformal quantile.  ``+inf`` if the level exceeds 1 (too few
        calibration points to certify the target coverage).
    """
    scores = np.asarray(scores, dtype=np.float64)
    scores = scores[np.isfinite(scores)]
    n = scores.size
    if n == 0:
        return float("inf")
    level = np.ceil((n + 1.0) * (1.0 - alpha)) / n
    if level > 1.0:
        return float("inf")
    return float(np.quantile(scores, level, method="higher"))


@dataclass
class ConformalResult:
    """Outcome of one conformal calibration + evaluation pass.

    Attributes:
        score_type: ``"normalized"`` or ``"absolute"``.
        alpha: Miscoverage level used.
        n_cal: Number of calibration points.
        n_eval: Number of evaluation points.
        coverage_before: Empirical coverage of the ORIGINAL head interval on
            the evaluation split.
        coverage_after: Empirical coverage of the CONFORMAL interval on the
            evaluation split.
        mean_width_before: Mean original interval width on the eval split.
        mean_width_after: Mean conformal interval width on the eval split.
        median_width_before: Median original interval width on eval split.
        median_width_after: Median conformal interval width on eval split.
        q: The conformal quantile applied.
        per_horizon: Optional per-horizon breakdown.
        per_var_horizon: Optional per-(variable, horizon) breakdown.
    """

    score_type: str
    alpha: float
    n_cal: int
    n_eval: int
    coverage_before: float
    coverage_after: float
    mean_width_before: float
    mean_width_after: float
    median_width_before: float
    median_width_after: float
    q: float
    per_horizon: Dict = field(default_factory=dict)
    per_var_horizon: Dict = field(default_factory=dict)


def _coverage(lower: np.ndarray, upper: np.ndarray, y: np.ndarray) -> float:
    if y.size == 0:
        return float("nan")
    return float(np.mean((y >= lower) & (y <= upper)))


def _widths(lower: np.ndarray, upper: np.ndarray) -> Tuple[float, float]:
    if lower.size == 0:
        return float("nan"), float("nan")
    w = upper - lower
    return float(np.mean(w)), float(np.median(w))


def calibrate_split_conformal(
    preds: np.ndarray,
    upper: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
    *,
    alpha: float = 0.10,
    score_type: str = "normalized",
    cal_frac: float = 0.5,
    seed: int = 42,
    z: float = FORECAST_HEAD_Z,
    per_cell: bool = True,
    min_cell_cal: int = 200,
) -> ConformalResult:
    """Run split-conformal calibration on cached twin predictions.

    The valid (observed, finite) entries are flattened across
    ``[horizon, sample, variable]`` and split into disjoint calibration and
    evaluation sets.  The conformal quantile is fit on calibration only and
    coverage/width are reported on the held-out evaluation set.

    When ``per_cell`` is True, a separate quantile is fit for each
    ``(horizon, variable)`` cell that has at least ``min_cell_cal``
    calibration points; cells with too little data fall back to the
    per-horizon quantile, and then to the global quantile.  This is the
    honest, adaptive choice because uncertainty structure differs strongly
    across both horizon and variable.

    Args:
        preds: ``[H, N, D]`` point predictions.
        upper: ``[H, N, D]`` upper 90% bound from the head.
        targets: ``[H, N, D]`` ground-truth future states.
        masks: ``[H, N, D]`` observed mask (``> 0.5`` means observed).
        alpha: Miscoverage level (0.10 -> 90% interval).
        score_type: ``"normalized"`` (scaled residual) or ``"absolute"``.
        cal_frac: Fraction of valid entries used for calibration.
        seed: RNG seed for the calibration/eval split.
        z: z-score the head used to build its interval.
        per_cell: If True, fit per-(horizon, variable) quantiles with
            hierarchical fallback.
        min_cell_cal: Minimum calibration count to trust a per-cell quantile.

    Returns:
        A :class:`ConformalResult`.
    """
    if score_type not in ("normalized", "absolute"):
        raise ValueError(f"unknown score_type {score_type!r}")

    H, N, D = preds.shape
    scale = implied_scale(preds, upper, z=z)

    valid = (
        (masks > 0.5)
        & np.isfinite(preds)
        & np.isfinite(targets)
        & np.isfinite(upper)
        & np.isfinite(scale)
    )

    # Index grids so we can recover (horizon, variable) for each valid entry.
    h_grid = np.broadcast_to(np.arange(H)[:, None, None], preds.shape)
    d_grid = np.broadcast_to(np.arange(D)[None, None, :], preds.shape)

    yhat = preds[valid].astype(np.float64)
    y = targets[valid].astype(np.float64)
    s = scale[valid].astype(np.float64)
    h_idx = h_grid[valid]
    d_idx = d_grid[valid]
    orig_lower = (preds - z * scale)[valid].astype(np.float64)
    orig_upper = upper[valid].astype(np.float64)

    n_total = yhat.size
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_total)
    n_cal = int(round(cal_frac * n_total))
    cal_idx = perm[:n_cal]
    eval_idx = perm[n_cal:]

    abs_res = np.abs(y - yhat)
    if score_type == "normalized":
        scores = abs_res / s
    else:
        scores = abs_res

    # ---- Global quantile (fit on calibration only) -----------------------
    q_global = conformal_quantile(scores[cal_idx], alpha)

    # ---- Per-horizon quantiles -------------------------------------------
    q_horizon: Dict[int, float] = {}
    for h in range(H):
        sel = cal_idx[h_idx[cal_idx] == h]
        if sel.size > 0:
            q_horizon[h] = conformal_quantile(scores[sel], alpha)

    # ---- Per-(horizon, variable) quantiles -------------------------------
    q_cell: Dict[Tuple[int, int], float] = {}
    if per_cell:
        for h in range(H):
            for d in range(D):
                sel = cal_idx[(h_idx[cal_idx] == h) & (d_idx[cal_idx] == d)]
                if sel.size >= min_cell_cal:
                    q_cell[(h, d)] = conformal_quantile(scores[sel], alpha)

    def _q_for(h: int, d: int) -> float:
        if per_cell and (h, d) in q_cell:
            return q_cell[(h, d)]
        if h in q_horizon and np.isfinite(q_horizon[h]):
            return q_horizon[h]
        return q_global

    # ---- Apply to evaluation split ---------------------------------------
    he = h_idx[eval_idx]
    de = d_idx[eval_idx]
    q_eval = np.array([_q_for(int(h), int(d)) for h, d in zip(he, de)])

    yhat_e = yhat[eval_idx]
    s_e = s[eval_idx]
    y_e = y[eval_idx]

    if score_type == "normalized":
        half = q_eval * s_e
    else:
        half = q_eval
    new_lower = yhat_e - half
    new_upper = yhat_e + half

    cov_before = _coverage(orig_lower[eval_idx], orig_upper[eval_idx], y_e)
    cov_after = _coverage(new_lower, new_upper, y_e)
    mw_before, mdw_before = _widths(orig_lower[eval_idx], orig_upper[eval_idx])
    mw_after, mdw_after = _widths(new_lower, new_upper)

    # ---- Per-horizon eval breakdown --------------------------------------
    per_horizon_out: Dict[str, Dict] = {}
    horizon_days = [1, 7, 14, 30, 90, 365]
    for h in range(H):
        m = he == h
        if m.sum() == 0:
            continue
        per_horizon_out[f"{horizon_days[h]}d"] = {
            "n_eval": int(m.sum()),
            "coverage_before": _coverage(
                orig_lower[eval_idx][m], orig_upper[eval_idx][m], y_e[m]
            ),
            "coverage_after": _coverage(new_lower[m], new_upper[m], y_e[m]),
            "mean_width_before": _widths(
                orig_lower[eval_idx][m], orig_upper[eval_idx][m]
            )[0],
            "mean_width_after": _widths(new_lower[m], new_upper[m])[0],
            "q": float(q_horizon.get(h, q_global)),
        }

    # ---- Per-(variable, horizon) eval breakdown --------------------------
    per_vh_out: Dict[str, Dict] = {}
    state_vars = [
        "dissolved_oxygen", "bod", "total_nitrogen", "total_phosphorus",
        "chlorophyll_a", "temperature", "ph", "turbidity", "doc", "sediment",
    ]
    for d in range(D):
        for h in range(H):
            m = (he == h) & (de == d)
            if m.sum() == 0:
                continue
            key = f"{state_vars[d]}@{horizon_days[h]}d"
            per_vh_out[key] = {
                "n_eval": int(m.sum()),
                "coverage_before": _coverage(
                    orig_lower[eval_idx][m], orig_upper[eval_idx][m], y_e[m]
                ),
                "coverage_after": _coverage(new_lower[m], new_upper[m], y_e[m]),
                "q": float(_q_for(h, d)),
            }

    return ConformalResult(
        score_type=score_type,
        alpha=alpha,
        n_cal=int(n_cal),
        n_eval=int(eval_idx.size),
        coverage_before=cov_before,
        coverage_after=cov_after,
        mean_width_before=mw_before,
        mean_width_after=mw_after,
        median_width_before=mdw_before,
        median_width_after=mdw_after,
        q=q_global,
        per_horizon=per_horizon_out,
        per_var_horizon=per_vh_out,
    )
