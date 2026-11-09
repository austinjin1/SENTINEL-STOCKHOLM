# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Derived ecological features exported by SENTINEL-Sim (RESEARCH_PLAN §7.1) for the platform/downstream.

Daily per site (from a Sim simulation dict, aligned with ``SimData``):
    gpp, er, k600, nep = gpp + er (g O2 m⁻² d⁻¹ / d⁻¹), do_deficit = DO_mean − DO_sat (mg/L),
    hypoxia_h_lt2 / lowdo_h_lt5 (simulated hours per day below 2 / 5 mg/L),
    t7dadmax (7-day running mean of the simulated daily maximum water temperature; defined only when all 7
    days exist), salinity_pulse (simulated SC above the site's train-block 95th percentile of OBSERVED SC).
Per site: thermal_sensitivity (train-block weekly air–water slope of observations).
Not yet implemented (reported as such): turbidity–discharge hysteresis class, forecastability score.
These are MODEL-DERIVED quantities: every export carries ``source="SENTINEL-Sim simulation"``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as D
from .checks import _weekly_slope


def daily_features(d: D.SimData, sim: dict, idx) -> pd.DataFrame:
    idx = np.asarray(idx)
    tr = D.block_mask(d.dates, "train")
    jS = D.TARGETS.index("spcond_uscm_mean")
    frames = []
    for i in idx:
        sc_sim = D.from_model("spcond_uscm_mean", sim["spcond_uscm_mean"][i])
        obs_sc = d.y[i, tr, jS]
        p95 = np.nanquantile(obs_sc, 0.95) if np.isfinite(obs_sc).sum() >= 365 else np.nan
        tmax = pd.Series(sim["temp_c_max"][i])
        frames.append(pd.DataFrame({
            "site": d.sites[i], "date": d.dates, "gpp": sim["gpp"][i], "er": sim["er"][i], "k600": sim["k600"][i],
            "nep": sim["gpp"][i] + sim["er"][i], "do_deficit": sim["do_mgl_mean"][i] - sim["do_sat_mean"][i],
            "hypoxia_h_lt2": sim["hyp2_h"][i], "lowdo_h_lt5": sim["hyp5_h"][i],
            "t7dadmax": tmax.rolling(7, min_periods=7).mean().to_numpy(),
            "salinity_pulse": (sc_sim > p95) if np.isfinite(p95) else np.zeros(len(d.dates), bool),
            "source": "SENTINEL-Sim simulation"}))
    out = pd.concat(frames, ignore_index=True)
    return out[np.isfinite(out["gpp"])]


def site_features(d: D.SimData, idx) -> pd.DataFrame:
    tr = D.block_mask(d.dates, "train")
    jT = D.TARGETS.index("temp_c_mean")
    rows = []
    for i in np.asarray(idx):
        ta = 0.5 * (d.forcing[i, tr, 0] + d.forcing[i, tr, 1])
        rows.append({"site": d.sites[i], "thermal_sensitivity": _weekly_slope(d.y[i, tr, jT], ta, d.dates[tr]),
                     "hysteresis_class": None, "forecastability": None})
    return pd.DataFrame(rows)
