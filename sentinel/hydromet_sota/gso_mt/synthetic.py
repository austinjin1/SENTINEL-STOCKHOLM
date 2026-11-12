# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Synthetic :class:`~.data.V2Train` for tests and timing smoke runs (no real observation; code paths only).

Water temperature follows smoothed air temperature with a diel cosine peaking at 15 h local solar; DO follows
saturation with a diel cycle peaking at 14 h. Grab stations hold sparse instantaneous rows at daytime hours (a share
without a time); dense sites hold daily statistics and hourly tables."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...data.sonde_splits import SondeSplit
from ...models.hydromet import data as D
from .data import AUX_WQ, DAILY, VAA_COLS, V2Train
from .solar import solar_channels

STATIC_EXTRA = ["wsareasqkm", "bfiws", "tmean9120ws", "precip9120ws", "elevws", "pctimp2019ws", "pcturbop2019ws",
                "runoffws", "elevcat"]


def synth_v2(n_fit: int = 6, n_inner: int = 2, n_grab: int = 4, n_grab_inner: int = 2, *, seed: int = 0,
             n_forc: int | None = None, n_static_extra: int = 0, end: str = "2021-09-30") -> V2Train:
    rs = np.random.default_rng(seed)
    S = n_fit + n_inner + n_grab + n_grab_inner
    sites = [f"{i:08d}" for i in range(n_fit + n_inner)] + [f"wqp:SYN-{i:05d}" for i in range(n_grab + n_grab_inner)]
    dates = pd.date_range(D.DATA_START, end, freq="D")
    T = len(dates)
    F = n_forc or len(D.FORCING_VARS)
    doy = dates.dayofyear.to_numpy()
    season = 12 * np.sin(2 * np.pi * (doy - 110) / 365.25)
    forc = rs.uniform(0.5, 2.0, (S, T, F)).astype("float32")
    fi = D.FORCING_VARS.index
    tmean = (10 + season[None, :] + rs.normal(0, 3, (S, T)) + rs.uniform(-4, 4, (S, 1))).astype("float32")
    forc[:, :, fi("tmean_c")], forc[:, :, fi("tmax_c")], forc[:, :, fi("tmin_c")] = tmean, tmean + 5, tmean - 5
    forc[:, :, fi("srad_wm2")] = np.clip(180 + 120 * np.sin(2 * np.pi * (doy - 80) / 365.25)[None] + rs.normal(0, 40, (S, T)), 5, None)
    forc[:, :, fi("precip_mm")] = rs.exponential(2, (S, T)) * (rs.random((S, T)) < 0.3)
    tw = np.maximum(pd.DataFrame(tmean.T).rolling(7, min_periods=1).mean().to_numpy().T * 0.8 + 2, 0).astype("float32")
    amp_t = rs.uniform(0.5, 3.0, (S, 1)).astype("float32")
    dos = (14.6 - 0.39 * tw + 0.0045 * tw ** 2).astype("float32")
    amp_d = rs.uniform(0.2, 1.5, (S, 1)).astype("float32")
    hrs = np.arange(24, dtype="float32")
    static = pd.DataFrame({"lat": 35 + rs.uniform(0, 10, S), "lon": -100 + rs.uniform(0, 20, S),
                           "elev_m": rs.uniform(0, 1500, S), **{c: rs.uniform(1, 100, S) for c in STATIC_EXTRA},
                           **{f"x{i:03d}": rs.normal(0, 1, S) for i in range(n_static_extra)}}, index=sites)
    n_dense = n_fit + n_inner
    hour_frac = ((0.5 + static["lon"].to_numpy()[:n_dense] / 15.0) % 1.0).astype("float32")
    hgrid = hrs[None, :] + hour_frac[:, None]                                           # (Sd, 24)
    cT = np.cos(2 * np.pi * (hgrid - 15.0) / 24.0)[:, None, :]
    cD = np.cos(2 * np.pi * (hgrid - 14.0) / 24.0)[:, None, :]
    hourly = np.stack([tw[:n_dense, :, None] + amp_t[:n_dense, :, None] * cT,
                       0.9 * dos[:n_dense, :, None] + amp_d[:n_dense, :, None] * cD], axis=-1).astype("float32")
    hourly += rs.normal(0, 0.1, hourly.shape).astype("float32")
    daily = np.full((S, T, len(DAILY)), np.nan, dtype="float32")
    dj = DAILY.index
    daily[:n_dense, :, dj("T_mean")] = hourly[..., 0].mean(-1)
    daily[:n_dense, :, dj("T_max")] = hourly[..., 0].max(-1)
    daily[:n_dense, :, dj("T_min")] = hourly[..., 0].min(-1)
    daily[:n_dense, :, dj("T_7dadm")] = D.trailing_mean_strict(daily[:n_dense, :, dj("T_max")], 7)
    daily[:n_dense, :, dj("DO_mean")] = hourly[..., 1].mean(-1)
    daily[:n_dense, :, dj("DO_min")] = hourly[..., 1].min(-1)
    daily[:n_dense, :, dj("DO_max")] = hourly[..., 1].max(-1)
    daily[:n_dense, :, dj("logq")] = rs.normal(0, 1, (n_dense, T))
    for k in AUX_WQ:
        daily[:n_dense, :, dj(k)] = rs.normal(1, 0.5, (n_dense, T))
    daily[:, :400] = np.nan
    hourly[:, :400] = np.nan
    daily[rs.random(daily.shape) < 0.1] = np.nan
    hourly[rs.random(hourly.shape[:2]) < 0.5] = np.nan                                   # half the site-days hourly
    y = np.full((S, T, len(D.TARGETS)), np.nan, dtype="float32")
    for v, k in (("temp_c_mean", "T_mean"), ("temp_c_7dadm", "T_7dadm"), ("do_mgl_mean", "DO_mean"),
                 ("do_mgl_min", "DO_min")):
        y[:, :, D.TARGETS.index(v)] = daily[:, :, dj(k)]
    for k, v in AUX_WQ.items():
        y[:, :, D.TARGETS.index(v)] = D.from_model(v, daily[:, :, dj(k)].astype("float64"))
    inst_val = np.full((S, T, 2), np.nan, dtype="float32")
    inst_hour = np.full((S, T, 2), np.nan, dtype="float32")
    for i in range(n_dense, S):
        days = np.sort(rs.choice(np.arange(400, T), size=int(rs.integers(40, 160)), replace=False))
        h = np.clip(rs.normal(11, 2.5, len(days)), 5, 20).astype("float32")
        inst_val[i, days, 0] = tw[i, days] + amp_t[i, 0] * np.cos(2 * np.pi * (h - 15) / 24) + rs.normal(0, 0.3, len(days))
        inst_val[i, days, 1] = 0.9 * dos[i, days] + amp_d[i, 0] * np.cos(2 * np.pi * (h - 14) / 24) + rs.normal(0, 0.3, len(days))
        inst_hour[i, days, :] = h[:, None]
        untimed = rs.random(len(days)) < 0.1
        inst_hour[i, days[untimed], :] = np.nan
        inst_val[i, days[rs.random(len(days)) < 0.2], 1] = np.nan
    groups = [f"g{i // 2}" for i in range(n_fit)] + [f"in{i // 2}" for i in range(n_inner)] + \
             [f"w{i}" for i in range(n_grab)] + [f"in{i % max(n_inner // 2, 1)}" for i in range(n_grab_inner)]
    split = SondeSplit(dict(zip(sites, ["train"] * S)), dict(zip(sites, groups)))
    data = D.HydroMetData(sites, np.array(["train"] * S), np.array(groups), dates, forc, y,
                          np.zeros_like(y, dtype=bool), static, split, {"final": False, "sha256": split.sha256()}, {},
                          ("train",), {"inner_sites": sites[n_fit:n_dense], "synthetic": True})
    hourly_row = np.r_[np.arange(n_dense), np.full(S - n_dense, -1)].astype(int)
    prov_class = np.r_[np.full(n_dense, -1), rs.integers(0, 6, S - n_dense)].astype(int)
    solar = solar_channels(static["lat"].to_numpy(), static["elev_m"].to_numpy(), dates.dayofyear,
                           forc[:, :, fi("srad_wm2")])
    vaa = pd.DataFrame({VAA_COLS[0]: rs.uniform(0.01, 30, S), VAA_COLS[1]: rs.integers(1, 8, S).astype(float),
                        VAA_COLS[2]: rs.uniform(0.1, 9, S)}, index=sites)
    vaa.iloc[::5, 0] = np.nan
    a = np.arange
    return V2Train(data, a(n_fit), a(n_fit, n_dense), a(n_fit), a(n_dense, n_dense + n_grab), a(n_dense + n_grab, S),
                   daily, inst_val, inst_hour, hourly, hourly_row, hour_frac, prov_class, solar, vaa,
                   {"synthetic": True, "n_locations": S})
