# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Diel-resolution controls for causal ladder step 2 (protocol §7, "Diel resolution").

Per site: one JJA summer of a train-block water year (the one with the most hours where both hourly DO and
water temperature are valid; ≥ 1,000 such hours), from ``processed/sonde_hourly``. Variables (hourly):

    light  solar-geometry light fraction of the local-solar hour × daily gridMET shortwave (W/m²)
    tw     hourly water temperature minus its daily mean (°C)
    dh     DO_h − DO_sat(T_h, P), minus its daily mean (mg/L)

Within-day deviations remove day-to-day weather/flow trends (declared in protocol §12, 2026-11-08); the
light series keeps its diel shape (it is the driver). Missing hours are masked, never interpolated.
"""

from __future__ import annotations

from ...data.reserved import is_reserved

import numpy as np
import pandas as pd

from ...core import paths
from ...models.sim import data as D
from ...models.sim import physics
from ...models.sim.dsst import water_year

HOURLY = paths.PROCESSED / "sonde_hourly" / "sites"
VARS = ["light", "tw", "dh"]
TAU_MAX_H = 26
MIN_HOURS = 1000
# (cause, effect, sign, (lag_lo, lag_hi) hours)
POSITIVE = [("light", "dh", +1, (0, 3)), ("light", "tw", +1, (0, 6))]
NEGATIVE = [("dh", "light"), ("tw", "light")]


def has_hourly(site: str) -> bool:
    return (HOURLY / f"{site}.parquet").exists() and not is_reserved(site)   # reserved new-site block


def site_matrix(d: D.SimData, i: int) -> pd.DataFrame | None:
    site = d.sites[i]
    f = HOURLY / f"{site}.parquet"
    if not f.exists():
        return None
    h = pd.read_parquet(f, columns=["temp_c_mean", "do_mgl_mean"])
    h.index = pd.DatetimeIndex(h.index)
    if h.index.tz is None:
        h.index = h.index.tz_localize("UTC")
    # local solar time
    solar = h.index.tz_convert(None) + pd.to_timedelta(float(d.lon[i]) / 15.0, unit="h")
    sdate = solar.normalize()
    wy = water_year(pd.DatetimeIndex(sdate))
    train_wy = set(np.unique(water_year(d.dates[D.block_mask(d.dates, "train")])))
    jja = np.isin(solar.month, (6, 7, 8)) & np.isin(wy, list(train_wy))
    ok = jja & h["temp_c_mean"].notna().to_numpy() & h["do_mgl_mean"].notna().to_numpy()
    if ok.sum() < MIN_HOURS:
        return None
    counts = pd.Series(wy[ok]).value_counts()
    best = int(counts.idxmax())
    if counts.max() < MIN_HOURS:
        return None
    sel = jja & (wy == best)
    hh = h[sel].copy()
    sol = solar[sel]
    # regular hourly grid in solar time (missing hours stay NaN)
    hh.index = sol.floor("h")
    hh = hh[~hh.index.duplicated()]
    grid = pd.date_range(hh.index.min().normalize(), hh.index.max().normalize() + pd.Timedelta(hours=23), freq="h")
    hh = hh.reindex(grid)
    tw = hh["temp_c_mean"].to_numpy(float)
    sat = physics.do_sat_mgl(tw, float(d.pressure[i]))
    dh = hh["do_mgl_mean"].to_numpy(float) - sat
    # daily shortwave (gridMET, local date) × light fraction of the local-solar hour
    day = grid.normalize()
    pos = pd.Series(np.arange(len(d.dates)), index=d.dates)
    k = pos.reindex(day).to_numpy()
    sw = np.full(len(grid), np.nan)
    good = np.isfinite(k)
    sw[good] = d.forcing[i, k[good].astype(int), D.FORCING_RAW.index("srad_wm2")]
    lw = physics.hourly_light_weights(np.array([float(d.lat[i])]), day.dayofyear.to_numpy())[0]   # (T, 24)
    frac = lw[np.arange(len(grid)), grid.hour]
    light = frac * sw * 24.0          # mean-preserving: daily mean of light = sw
    df = pd.DataFrame({"light": light, "tw": tw, "dh": dh}, index=grid)
    for v in ("tw", "dh"):
        df[v] = df[v] - df[v].groupby(day).transform("mean")
        # days with < 18 valid hours give a biased daily mean → mask the day
        n = df[v].notna().groupby(day).transform("sum")
        df.loc[n < 18, v] = np.nan
    df.attrs["water_year"] = best
    return df


def score(graphs, vals, names) -> list[dict]:
    from .controls import DETECT, ROBUST, freq

    ix = {n: k for k, n in enumerate(names)}
    rows = []
    for c, e, sgn, lags in POSITIVE:
        f = freq(graphs, vals, ix[c], ix[e], lags=lags, sign=sgn)
        rows.append({"kind": "positive", "control": f"{c}->{e}", "freq": f, "detected": f >= DETECT,
                     "robust": f >= ROBUST})
    return rows


def score_negative(graphs, vals, names) -> list[dict]:
    from .controls import DETECT, detect_rep

    ix = {n: k for k, n in enumerate(names)}
    rows = []
    for c, e in NEGATIVE:
        i, j = ix[c], ix[e]
        lag = detect_rep(graphs, vals, i, j, (1, TAU_MAX_H))
        con = graphs[:, i, j, 0] == 1          # contemporaneous edge oriented INTO light
        f = float((lag | con).mean())
        rows.append({"kind": "negative", "control": f"{c}->{e}", "freq": f, "detected": f >= DETECT})
    return rows


def link_assumptions_diel(names, tau_max=TAU_MAX_H) -> dict:
    from .discovery import link_assumptions

    return link_assumptions(names, tau_max, exogenous=("light",))
