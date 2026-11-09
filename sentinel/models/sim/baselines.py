# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Hindcast comparators for SENTINEL-Sim (protocol §4.2). All statistics are fitted per site on that
site's TRAIN-BLOCK data only (the same information the M1 naive baselines use); quantile forecasts
in model space (log for SC/turbidity) at the 19 levels of :data:`QUANTILES`.

* persistence, climatology, damped_anomaly — re-used unchanged from
  :class:`sentinel.models.baselines.sonde_baselines.SondeBaselines` (M1).
* physics_only (oracle inputs at the target day, labelled as such):
  - water temperature (mean/min/max): per-site linear air–water regression on the 7-day trailing
    mean gridMET air temperature;
  - DO (mean/min/max): DO_sat(T_obs,mean(t+h), P_site) × the site's train-block median ratio
    DO_stat / DO_sat for that calendar month (docs/research/06 §a.8);
  - SC, turbidity: per-site log C – log Q rating curve with observed discharge at t+h.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..baselines.sonde_baselines import QUANTILES, SondeBaselines
from ..sonde.data import Cube, climatology
from . import data as D
from . import physics

MIN_FIT = 60


def make_cube(d: D.SimData) -> Cube:
    """Comparator cube; observations before the train block (WY2006–07 spin-up period) are removed so
    every fitted statistic uses train-block data only, like the Sim (review W8)."""
    y = d.y.copy()
    y[:, d.dates < pd.Timestamp(D.TIME_BLOCKS["train"][0])] = np.nan
    return Cube(list(d.sites), d.dates, list(D.TARGETS), y, d.prov)


def naive_baselines(d: D.SimData, horizons) -> dict[str, SondeBaselines]:
    cube = make_cube(d)
    cm, cs = climatology(cube)
    return {v: SondeBaselines(cube, cm, cs, v, horizons=tuple(horizons)) for v in D.TARGETS}


def _per_site_residual_q(res: np.ndarray, train_t: np.ndarray) -> np.ndarray:
    S = res.shape[0]
    q = np.full((S, len(QUANTILES)), np.nan)
    for i in range(S):
        r = res[i, train_t]
        r = r[np.isfinite(r)]
        if r.size >= MIN_FIT:
            q[i] = np.quantile(r, QUANTILES)
    return q


class PhysicsOnly:
    """Per-site physics/process reference point predictions + train-block residual quantiles."""

    def __init__(self, d: D.SimData, fit_t: np.ndarray | None = None):
        self.d = d
        self.train_t = np.flatnonzero(D.block_mask(d.dates, "train")) if fit_t is None else np.asarray(fit_t, int)
        tmean = 0.5 * (d.forcing[..., D.FORCING_RAW.index("tmax_c")] + d.forcing[..., D.FORCING_RAW.index("tmin_c")])
        self.ta7 = pd.DataFrame(tmean.T).rolling(7, min_periods=5).mean().to_numpy().T
        self.logq = np.log(np.maximum(d.q_m3s, 0) + D.Q_FLOOR)
        self.month = d.dates.month.to_numpy()
        self.point = {}
        self.res_q = {}
        for v in D.TARGETS:
            p = self._fit_predict(v)
            z = D.to_model(v, d.y[:, :, D.TARGETS.index(v)])
            self.point[v] = p
            self.res_q[v] = _per_site_residual_q(z - p, self.train_t)

    def _linreg(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        S, T = z.shape
        out = np.full((S, T), np.nan)
        tt = self.train_t
        for i in range(S):
            ok = np.isfinite(x[i, tt]) & np.isfinite(z[i, tt])
            if ok.sum() < MIN_FIT:
                continue
            b, a = np.polyfit(x[i, tt][ok], z[i, tt][ok], 1)
            out[i] = a + b * x[i]
        return out

    def _fit_predict(self, v: str) -> np.ndarray:
        d = self.d
        z = D.to_model(v, d.y[:, :, D.TARGETS.index(v)])
        if v.startswith("temp_c"):
            return self._linreg(self.ta7, z)
        if v in ("spcond_uscm_mean", "turb_fnu_mean"):
            return self._linreg(self.logq, z)
        # DO: saturation at the observed daily-mean water temperature × monthly median ratio
        tobs = d.y[:, :, D.TARGETS.index("temp_c_mean")]
        sat = physics.do_sat_mgl(np.where(np.isfinite(tobs), tobs, np.nan), d.pressure[:, None])
        ratio = z / sat
        out = np.full_like(z, np.nan)
        tt = self.train_t
        for i in range(z.shape[0]):
            for mth in range(1, 13):
                sel_tr = tt[self.month[tt] == mth]
                r = ratio[i, sel_tr]
                r = r[np.isfinite(r)]
                if r.size >= 20:
                    msk = self.month == mth
                    out[i, msk] = sat[i, msk] * np.median(r)
        return out

    def forecast(self, v: str, t_idx: np.ndarray, h: int) -> np.ndarray:
        tgt = np.minimum(t_idx + h, self.point[v].shape[1] - 1)
        return self.point[v][:, tgt][..., None] + self.res_q[v][:, None, :]
