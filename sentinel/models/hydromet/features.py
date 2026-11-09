# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Causal forcing features for the tabular HydroMet models (GBM baselines).

Every dynamic feature for day t is a function of forcings on days ≤ t − ``lag`` only (trailing
windows, causal recursions; ``lag = 0`` = nowcast with same-day observed weather). Implemented by
shifting the forcing array ``lag`` days to the right before computing trailing statistics, so no
feature can see a later day (tested by perturbing future forcings).

Feature names start with ``f_`` (forcing), static columns keep their StreamCat/site names, calendar
columns are :data:`.data.CALENDAR_COLS`; :func:`design_matrix` checks every column with
:func:`.data.assert_allowed_inputs`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import CALENDAR_COLS, FORCING_VARS, HydroMetData, assert_allowed_inputs, doy_harmonics
from .physics import do_sat_mgl, pressure_atm_from_elevation

MEAN_WINDOWS = {"tmean_c": (1, 3, 7, 14, 30, 60, 90, 365), "tmax_c": (1, 7), "tmin_c": (1,),
                "srad_wm2": (1, 7, 30), "wind_ms": (1, 7), "vpd_kpa": (1, 7), "rhmin_pct": (1,),
                "rhmax_pct": (1,), "sph_kgkg": (1,)}
SUM_WINDOWS = {"precip_mm": (1, 3, 7, 14, 30, 90, 180), "pet_mm": (7, 30)}
WATER_BALANCE_WINDOWS = (30, 90, 365)
API_K = 0.9          # antecedent precipitation index decay per day
DDF_MM = 3.0         # degree-day snowmelt factor (mm °C⁻¹ d⁻¹), threshold 0 °C


def shift_right(x: np.ndarray, lag: int) -> np.ndarray:
    """x[:, t] -> x[:, t - lag] along axis 1 (NaN-padded)."""
    if lag <= 0:
        return x
    out = np.full_like(x, np.nan)
    out[:, lag:] = x[:, :-lag]
    return out


def trailing(x: np.ndarray, w: int, how: str = "mean", min_frac: float = 0.5) -> np.ndarray:
    """Trailing w-day mean/sum along axis 1 over finite values (days t−w+1..t); NaN if fewer than
    ``min_frac``·w finite values (sums are rescaled to w days)."""
    S, T = x.shape
    ok = np.isfinite(x)
    v = np.where(ok, x, 0.0).astype("float64")
    cs = np.concatenate([np.zeros((S, 1)), np.cumsum(v, axis=1)], axis=1)
    cn = np.concatenate([np.zeros((S, 1)), np.cumsum(ok, axis=1)], axis=1)
    lo = np.maximum(np.arange(1, T + 1) - w, 0)
    s = cs[:, 1:] - cs[:, lo]
    n = cn[:, 1:] - cn[:, lo]
    full = (np.arange(1, T + 1) >= w)[None, :]
    good = (n >= max(1, min_frac * w)) & full
    with np.errstate(invalid="ignore", divide="ignore"):
        m = s / n
    out = m if how == "mean" else m * w
    return np.where(good, out, np.nan).astype("float32")


def api(precip: np.ndarray, k: float = API_K) -> np.ndarray:
    S, T = precip.shape
    out = np.empty((S, T), dtype="float32")
    prev = np.zeros(S)
    for t in range(T):
        p = np.nan_to_num(precip[:, t], nan=0.0)
        prev = k * prev + p
        out[:, t] = prev
    return out


def snow_bucket(precip: np.ndarray, tmean: np.ndarray, ddf: float = DDF_MM) -> tuple[np.ndarray, np.ndarray]:
    """Degree-day snow storage (mm SWE proxy) and daily melt (mm); causal recursion."""
    S, T = precip.shape
    swe = np.empty((S, T), dtype="float32")
    melt = np.empty((S, T), dtype="float32")
    store = np.zeros(S)
    for t in range(T):
        p = np.nan_to_num(precip[:, t], nan=0.0)
        ta = np.nan_to_num(tmean[:, t], nan=0.0)
        store = store + np.where(ta <= 0, p, 0.0)
        m = np.minimum(store, ddf * np.maximum(ta, 0.0))
        store = store - m
        swe[:, t] = store
        melt[:, t] = m
    return swe, melt


def forcing_features(forc: np.ndarray, elev_m: np.ndarray, lag: int = 0) -> tuple[np.ndarray, list[str]]:
    """(S, T, K) float32 causal features and their names from raw forcings (S, T, F). With upstream-basin
    channels appended (protocol v0.10, F = 2 × len(FORCING_VARS)) the same feature set is computed on them
    too, named ``f_ws_*`` (GBM parity with the EA-LSTM inputs)."""
    nf = len(FORCING_VARS)
    if forc.shape[-1] == nf:
        return _forcing_block(forc, elev_m, lag, "f_")
    if forc.shape[-1] != 2 * nf:
        raise ValueError(f"unexpected number of forcing channels {forc.shape[-1]}")
    a, na = _forcing_block(forc[:, :, :nf], elev_m, lag, "f_")
    b, nb = _forcing_block(forc[:, :, nf:], elev_m, lag, "f_ws_")
    return np.concatenate([a, b], axis=-1), na + nb


def _forcing_block(forc: np.ndarray, elev_m: np.ndarray, lag: int, px: str) -> tuple[np.ndarray, list[str]]:
    fv = {v: shift_right(forc[:, :, j].astype("float64"), lag) for j, v in enumerate(FORCING_VARS)}
    feats, names = [], []
    for v, ws in MEAN_WINDOWS.items():
        for w in ws:
            feats.append(trailing(fv[v], w, "mean"))
            names.append(f"{px}{v}_m{w}")
    for v, ws in SUM_WINDOWS.items():
        for w in ws:
            feats.append(trailing(fv[v], w, "sum"))
            names.append(f"{px}{v}_s{w}")
    wb = fv["precip_mm"] - fv["pet_mm"]
    for w in WATER_BALANCE_WINDOWS:
        feats.append(trailing(wb, w, "sum"))
        names.append(f"{px}p_minus_pet_s{w}")
    feats.append(api(fv["precip_mm"]))
    names.append(f"{px}api_k090")
    swe, melt = snow_bucket(fv["precip_mm"], fv["tmean_c"])
    feats += [swe, trailing(melt, 7, "sum")]
    names += [f"{px}swe_proxy_mm", f"{px}melt_s7"]
    t7 = feats[names.index(f"{px}tmean_c_m7")]
    feats.append(trailing(fv["tmean_c"], 30, "mean") - feats[names.index(f"{px}tmean_c_m365")])
    names.append(f"{px}tmean_anom_m30_m365")
    p_atm = pressure_atm_from_elevation(elev_m)[:, None]
    feats.append(np.where(np.isfinite(t7), do_sat_mgl(np.nan_to_num(np.maximum(t7, 0.0)), p_atm), np.nan)
                 .astype("float32"))
    names.append(f"{px}dosat_air_m7")
    return np.stack(feats, axis=-1).astype("float32"), names


@dataclass
class FeatureSet:
    dyn: np.ndarray            # (S, T, K)
    dyn_names: list
    static: pd.DataFrame       # (S, n_static)
    cal: pd.DataFrame          # (T, 4)
    lag: int

    @classmethod
    def build(cls, data: HydroMetData, lag: int = 0, static_cols=None) -> "FeatureSet":
        dyn, names = forcing_features(data.forc, data.static["elev_m"].to_numpy(), lag)
        st = data.static if static_cols is None else data.static[list(static_cols)]
        return cls(dyn, names, st, doy_harmonics(data.dates), lag)

    def columns(self, static_cols=None, *, use_dyn=True, use_cal=True) -> list[str]:
        sc = list(self.static.columns) if static_cols is None else list(static_cols)
        return (list(self.dyn_names) if use_dyn else []) + sc + (list(CALENDAR_COLS) if use_cal else [])

    def design_matrix(self, si: np.ndarray, ti: np.ndarray, columns: list[str]) -> np.ndarray:
        assert_allowed_inputs(columns, static_cols=list(self.static.columns), forcing_feature_cols=self.dyn_names)
        out = np.empty((len(si), len(columns)), dtype="float32")
        st = self.static.to_numpy(dtype="float32")
        cal = self.cal.to_numpy(dtype="float32")
        dpos = {n: k for k, n in enumerate(self.dyn_names)}
        spos = {n: k for k, n in enumerate(self.static.columns)}
        cpos = {n: k for k, n in enumerate(CALENDAR_COLS)}
        for c, name in enumerate(columns):
            if name in dpos:
                out[:, c] = self.dyn[si, ti, dpos[name]]
            elif name in spos:
                out[:, c] = st[si, spos[name]]
            else:
                out[:, c] = cal[ti, cpos[name]]
        return out
