# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Uncalibrated physics features for the PF-GBM reference (``sonde_arch_v2_amendment.md`` §2b). All constants are
fixed here, before any result (no fitting, no tuning on any block):

* **air2stream-style water-temperature forecast (regional default).** Starting from the last observed water
  temperature, relax towards an equilibrium of the forecast air temperature, one exact exponential step per day:
  Tw_k = Te_k + (Tw_{k−1} − Te_k)·exp(−κ), Te_k = A0 + A1·Tair_k, with A0 = 3.0 °C, A1 = 0.8, κ = 0.25 d⁻¹
  (a Mohseni/air2stream-type regional default; Te is floored at 0 °C).
* **DO saturation** of that water-temperature forecast: Benson–Krause (M6 ``hydromet.physics.do_sat_mgl``) at the
  barometric pressure of the site elevation.
* **Raymond (2012) K600 prior** from the last observed discharge and the channel slope (Sim ``k600_prior``); NaN
  where discharge or slope is missing.
* **Two-end-member SC dilution:** SC_mix = SC_last · Q_b / (Q_b + Q_r), with Q_b = last discharge (baseflow end
  member at the last observed SC) and Q_r = runoff from the forecast precipitation sum over leads 1..h,
  Q_r = C_R · P_sum[mm] · 1e-3 · A[km²] · 1e6 / 86400 / h  (m³/s, mean over the h days), runoff SC taken as 0,
  C_R = 0.2. NaN without discharge or drainage area.
"""

from __future__ import annotations

import numpy as np

A0, A1, KAPPA = 3.0, 0.8, 0.25
C_R = 0.2
CFS_TO_M3S = 0.028316846592


def air2stream_forecast(tw_last: np.ndarray, tair_fc: np.ndarray) -> np.ndarray:
    """tw_last (N,), tair_fc (N, L) daily mean air temperature forecasts → (N, L) water-temperature forecast."""
    n, L = tair_fc.shape
    out = np.full((n, L), np.nan, dtype=np.float64)
    tw = tw_last.astype(np.float64)
    decay = np.exp(-KAPPA)
    for k in range(L):
        te = np.maximum(A0 + A1 * tair_fc[:, k], 0.0)
        tw = np.where(np.isfinite(te), te + (tw - te) * decay, np.nan)
        out[:, k] = tw
    return out


def dosat_forecast(tw_fc: np.ndarray, elev_m: np.ndarray) -> np.ndarray:
    from sentinel.models.hydromet.physics import do_sat_mgl, pressure_atm_from_elevation

    p = pressure_atm_from_elevation(np.nan_to_num(np.asarray(elev_m, float), nan=0.0))
    p = np.broadcast_to(np.asarray(p, float).reshape(-1, *([1] * (tw_fc.ndim - 1))), tw_fc.shape)
    return np.asarray(do_sat_mgl(tw_fc, p), dtype=np.float64)


def k600_raymond(q_cfs_last: np.ndarray, slope: np.ndarray) -> np.ndarray:
    from sentinel.models.sim.physics import k600_prior

    q, s = np.broadcast_arrays(np.asarray(q_cfs_last, float) * CFS_TO_M3S, np.asarray(slope, float))
    ok = np.isfinite(q) & np.isfinite(s) & (q > 0) & (s > 0)
    out = np.full(q.shape, np.nan)
    if ok.any():
        out[ok] = np.asarray(k600_prior(q[ok], s[ok]), dtype=float)
    return out


def sc_dilution(sc_last: np.ndarray, q_cfs_last: np.ndarray, p_sum_mm: np.ndarray, area_km2: np.ndarray,
                h: int) -> np.ndarray:
    qb = np.asarray(q_cfs_last, float) * CFS_TO_M3S
    qr = C_R * np.asarray(p_sum_mm, float) * 1e-3 * np.asarray(area_km2, float) * 1e6 / 86400.0 / max(h, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.asarray(sc_last, float) * qb / (qb + np.maximum(qr, 0.0))
    return np.where(np.isfinite(qb) & (qb > 0) & np.isfinite(qr), out, np.nan)
