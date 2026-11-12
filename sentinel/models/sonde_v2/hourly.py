# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""M1-H targets and naive references (``sonde_arch_v2_amendment.md`` §3; lead rulings R4, L2).

Targets are the HOURLY IV values themselves (hourly means of USGS IV readings, Approved only, local standard time),
never DV-equivalent statistics. Issue time = 00:00 local standard time after issue day d (hour index (d + 1)·24 on
the cube's hourly axis). Lead ℓ hours → the hourly value of hour [issue + ℓ − 1, issue + ℓ).
* Primary: hourly-IV DO at ℓ ∈ {6, 12, 24, 48, 72}; secondary: hourly-IV water temperature at the same leads,
  daily hypoxia hours (< 3, < 5 mg/L; a day needs ≥ 20 observed hours) for days d+1..d+7, daily minimum of hourly-IV DO.
* Reference-window climatology (design B): hour-of-day × month mean and sd from the first 730 days of the site's
  hourly record (≥ 365 days with data), the M1-N reference-window rule applied at hourly resolution.
* Naive references (quantiles from the reference window's standardised residuals per lead, pooled over training
  sites): hourly persistence (last value ≤ 6 h before issue), diel persistence (the same hour of day on issue day d,
  shifted by the climatological change to the target hour), hourly climatology (the hour × month climatology).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

LEADS_H = (6, 12, 24, 48, 72)
TAUS = np.round(np.arange(0.05, 0.951, 0.05), 2)


def hourly_series(df: pd.DataFrame, col: str, prov: str, utc_offset_h: float, d0: pd.Timestamp, T: int) -> np.ndarray:
    """(T·24,) hourly values on the local-standard-time axis starting at d0 00:00; provisional values dropped."""
    out = np.full(T * 24, np.nan, np.float32)
    if col not in df:
        return out
    t = pd.DatetimeIndex(df.index).tz_convert("UTC").tz_localize(None) + pd.Timedelta(hours=float(utc_offset_h))
    hi = ((t - d0) / pd.Timedelta(hours=1)).to_numpy().astype(np.int64)
    v = df[col].to_numpy(float)
    good = (hi >= 0) & (hi < T * 24) & np.isfinite(v)
    if prov in df:
        good &= ~df[prov].fillna(False).to_numpy(bool)
    out[hi[good]] = v[good]
    return out


def ref_window_hourly_clim(x: np.ndarray, d0: pd.Timestamp, *, days: int = 730, min_days: int = 365):
    """x (T·24,) → (mean, sd) arrays (24, 12) from the reference window; None if too little data."""
    fin = np.flatnonzero(np.isfinite(x))
    if fin.size == 0:
        return None
    a = fin[0]
    w = slice(a, min(a + days * 24, x.size))
    xw = x[w]
    hrs = np.arange(w.start, w.stop)
    days_obs = np.unique(hrs[np.isfinite(xw)] // 24).size
    if days_obs < min_days:
        return None
    hod = hrs % 24
    month = (d0 + pd.to_timedelta(hrs // 24, unit="D")).month.to_numpy() - 1
    mean = np.full((24, 12), np.nan)
    sd = np.full((24, 12), np.nan)
    ok = np.isfinite(xw)
    allm, alls = float(np.nanmean(xw)), float(np.nanstd(xw) + 1e-6)
    for h in range(24):
        for m in range(12):
            sel = ok & (hod == h) & (month == m)
            if sel.sum() >= 10:
                mean[h, m], sd[h, m] = xw[sel].mean(), xw[sel].std() + 1e-6
    # months without support: hour-of-day mean over all months, then overall
    for h in range(24):
        sel = ok & (hod == h)
        mh = xw[sel].mean() if sel.any() else allm
        sh = xw[sel].std() + 1e-6 if sel.sum() > 1 else alls
        miss = ~np.isfinite(mean[h])
        mean[h, miss], sd[h, miss] = mh, sh
    return mean, sd


def targets_at(x: np.ndarray, issue_day: np.ndarray, leads=LEADS_H) -> np.ndarray:
    """(n, len(leads)) hourly targets for issue days (index into days)."""
    base = (issue_day + 1) * 24
    idx = base[:, None] + np.asarray(leads)[None, :] - 1
    ok = idx < x.size
    out = np.full(idx.shape, np.nan, np.float32)
    out[ok] = x[idx[ok]]
    return out


def hypoxia_hours(x: np.ndarray, issue_day: np.ndarray, *, thr: float, days: int = 7, min_hours: int = 20) -> np.ndarray:
    """(n, days) number of hourly values < thr in local days d+1..d+days (NaN if < min_hours observed)."""
    out = np.full((issue_day.size, days), np.nan, np.float32)
    for k in range(days):
        a = (issue_day + 1 + k) * 24
        ok = a + 24 <= x.size
        blk = np.full((issue_day.size, 24), np.nan)
        blk[ok] = x[a[ok, None] + np.arange(24)[None, :]]
        n = np.isfinite(blk).sum(1)
        cnt = (blk < thr).sum(1).astype(np.float32)
        out[:, k] = np.where(n >= min_hours, cnt * 24.0 / np.maximum(n, 1), np.nan)
    return out


def clim_at(clim, d0: pd.Timestamp, hour_idx: np.ndarray):
    mean, sd = clim
    hod = hour_idx % 24
    month = (d0 + pd.to_timedelta(hour_idx // 24, unit="D")).month.to_numpy() - 1
    return mean[hod, month], sd[hod, month]


def naive_point(kind: str, x: np.ndarray, clim, d0: pd.Timestamp, issue_day: np.ndarray, leads=LEADS_H):
    """Point forecast of a naive reference (n, L); NaN where undefined."""
    base = (issue_day + 1) * 24
    tgt = base[:, None] + np.asarray(leads)[None, :] - 1
    cm_t, _ = clim_at(clim, d0, tgt.ravel())
    cm_t = cm_t.reshape(tgt.shape)
    if kind == "climatology":
        return cm_t
    if kind == "persistence":
        last = np.full(issue_day.size, np.nan)
        for back in range(1, 7):                                   # last value within 6 h before issue
            v = x[np.clip(base - back, 0, None)]
            last = np.where(np.isfinite(last), last, v)
        return np.repeat(last[:, None], len(leads), 1)
    if kind == "diel":
        # the same hour of day on issue day d, shifted by the climatological change to the target hour
        same_idx = issue_day[:, None] * 24 + (np.asarray(leads)[None, :] - 1) % 24
        same = x[np.clip(same_idx, 0, None)]
        csame, _ = clim_at(clim, d0, same_idx.ravel())
        return same + (cm_t - csame.reshape(tgt.shape))
    raise ValueError(kind)


def residual_quantiles(point: np.ndarray, y: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Per-lead standardised-residual quantiles (L, 19) from TRAINING rows (pooled over sites)."""
    r = (y - point) / sd
    out = np.zeros((point.shape[1], len(TAUS)))
    for k in range(point.shape[1]):
        v = r[:, k][np.isfinite(r[:, k])]
        out[k] = np.quantile(v, TAUS) if v.size >= 100 else norm.ppf(TAUS)
    return out
