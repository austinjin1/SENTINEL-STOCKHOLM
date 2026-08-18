"""Causal rolling-baseline multivariate CUSUM scoring on cached residuals.

The specificity limitation of the baseline detector is that it compares each
window's reconstruction error to a *fixed global* (mean, std, tau) fitted on
interleaved clean runs. On non-stationary sites the clean residual baseline
drifts over the record (seasonal regime, sensor recalibration, biofouling), so
a fixed threshold fires persistently on drifted-but-clean held-out windows.

Genuinely different idea (not deseasonalization, not adaptive-K):
  1. STRICTLY CAUSAL ROLLING LOCAL BASELINE. For each param, standardize the
     current residual against a trailing rolling median + MAD computed only from
     the detector's own PAST windows (a look-back buffer), with a short guard gap
     so the incipient shift itself does not contaminate its own baseline. This
     makes the operating point self-adapt to each site's local residual dynamics,
     with NO future leakage and NO labels.
  2. MULTIVARIATE CUSUM. Accumulate the positive standardized deviation (summed
     across params, minus a slack k) into a running sum S; alarm when S exceeds
     h, then reset. CUSUM detects a *sustained level SHIFT* while ignoring a
     stationary-but-noisy baseline -- exactly the failure mode of a fixed
     threshold. A single global h is calibrated on control sites, so the operating
     point transfers instead of over-fitting a per-site calibration slice.

This module is pure NumPy and runs on the cached residuals (CPU only).
"""
from __future__ import annotations

import contextlib
import warnings

import numpy as np


@contextlib.contextmanager
def _suppress_nan_warnings():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"All-NaN slice encountered")
        warnings.filterwarnings("ignore", r"Degrees of freedom <= 0")
        warnings.filterwarnings("ignore", r"Mean of empty slice")
        yield


def rolling_causal_z(err_col: np.ndarray, valid: np.ndarray,
                     w_base: int, guard: int, min_base: int) -> np.ndarray:
    """Strictly causal robust z for one param over the ordered window sequence.

    For index i, baseline = windows [i - guard - w_base, i - guard) that are
    valid (clean-covered). z = (err_i - median_base) / (1.4826 * MAD_base).
    Returns NaN where fewer than min_base valid baseline windows exist.

    `valid` marks windows usable as baseline (QC-covered AND not hypoxia-near),
    so the rolling baseline is built only from clean context -- but note the
    z of EVERY window (including near/onset ones) is still computed against that
    clean trailing baseline. No future information is ever used.

    Vectorized via a strided sliding view over a NaN-masked, guard-shifted copy
    so the per-window robust median/MAD are batched (nanmedian across the window
    axis). Invalid (non-clean or non-finite) baseline members are set to NaN and
    ignored by nanmedian.
    """
    n = len(err_col)
    z = np.full(n, np.nan, np.float64)
    if n <= guard:
        return z
    # masked baseline source: value where valid & finite, else NaN
    src = np.where(valid & np.isfinite(err_col), err_col.astype(np.float64), np.nan)
    # For index i the baseline window is src[i-guard-w_base : i-guard].
    # Equivalently, shift src right by `guard` then take the trailing w_base slice
    # ending at i. Build a padded array so index arithmetic is uniform.
    pad = w_base + guard
    padded = np.concatenate([np.full(pad, np.nan), src])          # len n+pad
    # sliding windows of length w_base whose RIGHT edge sits `guard` before i.
    # window for i is padded[i : i + w_base]  (padded index i == src index i-pad)
    # check: padded[i : i+w_base] maps to src[i-pad : i-pad+w_base]
    #        = src[i-guard-w_base : i-guard]  -> exactly the causal baseline.
    from numpy.lib.stride_tricks import sliding_window_view
    win = sliding_window_view(padded, w_base)[:n]                 # [n, w_base]
    cnt = np.sum(np.isfinite(win), axis=1)
    with np.errstate(invalid="ignore"), \
            _suppress_nan_warnings():
        med = np.nanmedian(win, axis=1)
        mad = np.nanmedian(np.abs(win - med[:, None]), axis=1)
        std = np.nanstd(win, axis=1)
    scale = 1.4826 * mad
    # fall back to std where MAD degenerate
    scale = np.where(scale > 1e-6, scale, np.where(std > 1e-6, std, 1.0))
    ok = cnt >= min_base
    zz = (err_col - med) / scale
    z[ok] = zz[ok]
    return z


def cusum_alarms(dev: np.ndarray, slack_k: float, h: float):
    """One-sided multivariate CUSUM over an aggregate positive-deviation signal.

    dev: per-window aggregate deviation (>=0 typical when residuals shift up).
    S_i = max(0, S_{i-1} + dev_i - slack_k); alarm where S_i >= h; reset S=0 on
    alarm. NaN dev breaks/resets the accumulator (missing context). Returns a
    boolean alarm array.
    """
    n = len(dev)
    alarm = np.zeros(n, bool)
    S = 0.0
    for i in range(n):
        d = dev[i]
        if not np.isfinite(d):
            S = 0.0
            continue
        S = max(0.0, S + d - slack_k)
        if S >= h:
            alarm[i] = True
            S = 0.0
    return alarm


def aggregate_deviation(z: np.ndarray, floor: float = 0.0) -> np.ndarray:
    """Aggregate per-param causal z into a single positive-shift signal.

    Sum of positive parts across params: sum_p max(0, z_p - floor). Rewards a
    coordinated upward shift in reconstruction error across parameters (the
    multivariate decoupling signature of an incipient regime change) while a
    single noisy param contributes little. NaN if any needed z is NaN in a way
    that leaves no valid params.
    """
    pos = np.clip(z - floor, 0.0, None)
    # a window is usable if at least one param has a finite z
    any_valid = np.isfinite(z).any(axis=1)
    agg = np.nansum(pos, axis=1)
    agg[~any_valid] = np.nan
    return agg


def score_site(err: np.ndarray, valid: np.ndarray, *,
               w_base: int, guard: int, min_base: int,
               floor: float, slack_k: float, h: float):
    """Full pipeline for one site: causal z per param -> agg dev -> CUSUM alarms.

    err: [N,4] residuals; valid: [N] baseline-usable mask. Returns dict with
    z [N,4], dev [N], alarm [N].
    """
    n, p = err.shape
    z = np.full((n, p), np.nan, np.float64)
    for j in range(p):
        z[:, j] = rolling_causal_z(err[:, j], valid, w_base, guard, min_base)
    dev = aggregate_deviation(z, floor=floor)
    alarm = cusum_alarms(dev, slack_k, h)
    return {"z": z, "dev": dev, "alarm": alarm}


def first_alarm_time(alarm: np.ndarray, end_ts: np.ndarray, keep: np.ndarray):
    """UTC seconds of the first alarm among windows where keep=True, else None."""
    idx = np.where(alarm & keep)[0]
    return int(end_ts[idx[0]]) if idx.size else None
