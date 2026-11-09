# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Per-site matrices for the causal ladder (protocol §7 "Common setup").

Daily variables (anomalies vs the site's train-block DOY climatology, ±15 d):
    tair   gridMET daily-mean air temperature (°C)
    sw     gridMET shortwave (W/m²)
    precip log1p(precipitation, mm)
    logq   log observed discharge (m³/s) — gap-filled values are NOT used (masked)
    tw     water temperature daily mean (°C)
    dmax   DO_max − DO_sat(T_mean, P)  (mg/L; daytime-maximum deficit)
    dmin   DO_min − DO_sat(T_mean, P)  (mg/L; night-minimum deficit)
    logsc  log(SC + 1)
    logturb log(turbidity + 0.1)
Ice-affected days (T_w < 1 °C) mask every river variable; nothing is interpolated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...models.sim import data as D
from ...models.sim import physics
from ...models.sim.dsst import fit_climatology

VARS = ["tair", "sw", "precip", "logq", "tw", "dmax", "dmin", "logsc", "logturb"]
WEATHER = ("tair", "sw", "precip")
RIVER = tuple(v for v in VARS if v not in WEATHER)
MIN_VALID = 730


def raw_site(d: D.SimData, i: int) -> pd.DataFrame:
    f = {n: d.forcing[i, :, k].astype("float64") for k, n in enumerate(D.FORCING_RAW)}
    y = {v: d.y[i, :, k].astype("float64") for k, v in enumerate(D.TARGETS)}
    tw = y["temp_c_mean"]
    sat = physics.do_sat_mgl(tw, float(d.pressure[i]))
    q = np.where(d.q_missing[i], np.nan, d.q_m3s[i].astype("float64"))
    df = pd.DataFrame({
        "tair": 0.5 * (f["tmax_c"] + f["tmin_c"]), "sw": f["srad_wm2"], "precip": np.log1p(np.maximum(f["precip_mm"], 0)),
        "logq": np.log(np.maximum(q, 0) + D.Q_FLOOR), "tw": tw, "dmax": y["do_mgl_max"] - sat,
        "dmin": y["do_mgl_min"] - sat, "logsc": np.log(np.maximum(y["spcond_uscm_mean"], 0) + 1.0),
        "logturb": np.log(np.maximum(y["turb_fnu_mean"], 0) + 0.1)}, index=d.dates)
    ice = ~(tw >= 1.0)
    for v in RIVER:
        if v != "logq":
            df.loc[ice, v] = np.nan
    df.loc[ice & np.isfinite(tw), "logq"] = np.nan
    return df


def anomalies(df: pd.DataFrame, fit_days: np.ndarray) -> pd.DataFrame:
    z = df.to_numpy().T
    cm, _ = fit_climatology(z, df.index, np.asarray(fit_days, bool))
    doy = df.index.dayofyear.to_numpy() - 1
    return pd.DataFrame((z - cm[:, doy]).T, index=df.index, columns=df.columns)


def site_matrix(d: D.SimData, i: int, days: np.ndarray, *, fit_days: np.ndarray | None = None,
                variables=VARS, min_valid: int = MIN_VALID) -> pd.DataFrame | None:
    """Anomaly matrix restricted to ``days`` (other rows NaN, time axis kept contiguous); variables with
    < ``min_valid`` valid days in ``days`` are dropped. Requires tw and logq."""
    fit = D.block_mask(d.dates, "train") if fit_days is None else fit_days
    a = anomalies(raw_site(d, i)[list(variables)], fit)
    a.loc[~np.asarray(days, bool)] = np.nan
    keep = [v for v in variables if np.isfinite(a[v]).sum() >= min_valid]
    if "tw" not in keep or "logq" not in keep:
        return None
    a = a[keep]
    rows = np.flatnonzero(np.asarray(days, bool))
    return a.iloc[rows[0]: rows[-1] + 1]


def cq_slope(d: D.SimData, i: int) -> tuple[float, float, float]:
    """Train-block log SC – log Q slope with an AR(1)-adjusted 95% CI (dilution-site definition)."""
    from ...models.sim.checks import _cq_slope_ci

    tr = D.block_mask(d.dates, "train")
    q = np.where(d.q_missing[i], np.nan, d.q_m3s[i])
    return _cq_slope_ci(d.y[i, tr, D.TARGETS.index("spcond_uscm_mean")], q[tr])
