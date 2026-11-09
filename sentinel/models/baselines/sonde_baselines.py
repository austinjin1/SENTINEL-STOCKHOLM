# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Naive probabilistic baselines for SENTINEL-Sonde daily forecasting (RESEARCH_PLAN §5.1).

All parameters (climatology, damping coefficients, error quantiles) are fitted on the
training block only (dates <= TIME_BLOCKS['train'][1]); forecasts are 19 quantiles in model
space (see sentinel.models.sonde.data.to_model_space).

* climatology       — site × day-of-year mean/std, Gaussian quantiles
* persistence       — last value (≤ 3 days old) + training quantiles of h-day changes
* damped_anomaly    — clim(d+h) + φ_h·(z(d) − clim(d)) + training residual quantiles
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..sonde.data import CLIM_END, Cube, to_model_space

QUANTILES = np.round(np.arange(0.05, 0.951, 0.05), 2)  # 19 levels
MAX_STALENESS = 3


def last_value(z: np.ndarray, max_stale: int = MAX_STALENESS) -> tuple[np.ndarray, np.ndarray]:
    """Forward-fill along time with a staleness limit. z: (S, T). Returns (value, age_days)."""
    S, T = z.shape
    val = np.full((S, T), np.nan)
    age = np.full((S, T), np.inf)
    cur = np.full(S, np.nan)
    cur_age = np.full(S, np.inf)
    for t in range(T):
        obs = np.isfinite(z[:, t])
        cur = np.where(obs, z[:, t], cur)
        cur_age = np.where(obs, 0, cur_age + 1)
        ok = cur_age <= max_stale
        val[:, t] = np.where(ok, cur, np.nan)
        age[:, t] = np.where(ok, cur_age, np.inf)
    return val, age


class SondeBaselines:
    """Fit once per variable on the training block; produce quantile forecasts for any issue t."""

    def __init__(self, cube: Cube, clim_mean: np.ndarray, clim_std: np.ndarray, var: str,
                 horizons=(1, 2, 3, 5, 7, 10, 14)):
        self.cube, self.var, self.h = cube, var, tuple(horizons)
        j = cube.var_index(var)
        self.z = to_model_space(var, cube.y[:, :, j])
        self.doy = cube.dates.dayofyear.to_numpy() - 1
        self.cm = clim_mean[:, :, j]
        self.cs = clim_std[:, :, j]
        self.train_t = np.flatnonzero(cube.dates <= pd.Timestamp(CLIM_END))
        self.last, _ = last_value(self.z)
        self.anom_last = self.last - self.cm[:, self.doy]
        self._fit()

    def _fit(self):
        S = self.z.shape[0]
        T_tr = self.train_t.max() + 1
        self.phi = {}
        self.res_q = {}
        self.chg_q = {}
        a_obs = self.z - self.cm[:, self.doy]
        for h in self.h:
            a0 = self.anom_last[:, : T_tr - h]
            a1 = a_obs[:, h:T_tr]
            ok = np.isfinite(a0) & np.isfinite(a1)
            num = np.nansum(np.where(ok, a0 * a1, 0), axis=1)
            den = np.nansum(np.where(ok, a0 * a0, 0), axis=1)
            phi = np.clip(np.where(den > 0, num / np.maximum(den, 1e-12), 0.0), 0.0, 1.0)
            self.phi[h] = phi
            res = np.where(ok, a1 - phi[:, None] * a0, np.nan)
            chg = np.where(np.isfinite(self.last[:, : T_tr - h]) & np.isfinite(self.z[:, h:T_tr]),
                           self.z[:, h:T_tr] - self.last[:, : T_tr - h], np.nan)
            rq = np.full((S, len(QUANTILES)), np.nan)
            cq = np.full((S, len(QUANTILES)), np.nan)
            enough = ok.sum(axis=1) >= 60
            for i in np.flatnonzero(enough):
                rq[i] = np.nanquantile(res[i], QUANTILES)
                cq[i] = np.nanquantile(chg[i], QUANTILES)
            self.res_q[h] = rq
            self.chg_q[h] = cq

    def forecast(self, model: str, t_idx: np.ndarray, h: int) -> np.ndarray:
        """Quantiles in model space, shape (S, len(t_idx), 19)."""
        tgt = t_idx + h
        d_tgt = self.doy[np.minimum(tgt, len(self.doy) - 1)]
        if model == "climatology":
            m = self.cm[:, d_tgt][..., None]
            s = self.cs[:, d_tgt][..., None]
            return m + s * norm.ppf(QUANTILES)[None, None, :]
        if model == "persistence":
            return self.last[:, t_idx][..., None] + self.chg_q[h][:, None, :]
        if model == "damped_anomaly":
            mu = self.cm[:, d_tgt] + self.phi[h][:, None] * self.anom_last[:, t_idx]
            return mu[..., None] + self.res_q[h][:, None, :]
        raise ValueError(model)

    def observed(self, t_idx: np.ndarray, h: int) -> np.ndarray:
        tgt = t_idx + h
        out = np.full((self.z.shape[0], len(t_idx)), np.nan)
        ok = tgt < self.z.shape[1]
        out[:, ok] = self.z[:, tgt[ok]]
        return out

    def scorable(self, t_idx: np.ndarray, h: int) -> np.ndarray:
        """Common scoring mask: recent observation (≤ 3 d) at issue and observation at target."""
        return np.isfinite(self.last[:, t_idx]) & np.isfinite(self.observed(t_idx, h)) \
            & np.isfinite(self.cm[:, self.doy[t_idx]])
