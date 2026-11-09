# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Causal (issue-date) features for SENTINEL-Sonde tabular models.

Every feature for issue date d is a function of observations at dates <= d only (trailing
windows ``(d - w, d]``), of the training-years climatology evaluated at d or at the target day
d + h (deterministic given the date), of the calendar, or of static site attributes. This is
asserted by ``tests/test_sonde_models.py::test_features_causal`` (perturbing every value after d
must leave the features at d unchanged).

Static location features (lat, lon, HUC2, elevation, drainage area) are allowed: forecasting at a
site uses that site's own history anyway, and unseen-site evaluation keeps whole basin groups
out of training, so location cannot identify a training target.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..baselines.sonde_baselines import MAX_STALENESS, last_value
from .data import to_model_space

DISCHARGE = "discharge_cfs_mean"


def _csum(x: np.ndarray) -> np.ndarray:
    """Cumulative sum along time with a leading zero column: (S, T) -> (S, T + 1)."""
    return np.concatenate([np.zeros((x.shape[0], 1)), np.cumsum(x, axis=1)], axis=1)


def rolling_sum(x: np.ndarray, w: int) -> np.ndarray:
    """Sum over the trailing window (t - w, t] (truncated at the series start); x has no NaN."""
    c = _csum(x)
    T = x.shape[1]
    hi = np.arange(1, T + 1)
    lo = np.maximum(hi - w, 0)
    return c[:, hi] - c[:, lo]


def rolling_mean(z: np.ndarray, w: int, min_count: int = 1) -> np.ndarray:
    ok = np.isfinite(z)
    n = rolling_sum(ok.astype(float), w)
    s = rolling_sum(np.where(ok, z, 0.0), w)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n >= min_count, s / np.maximum(n, 1), np.nan)


def rolling_std(z: np.ndarray, w: int, min_count: int = 3) -> np.ndarray:
    ok = np.isfinite(z)
    n = rolling_sum(ok.astype(float), w)
    s = rolling_sum(np.where(ok, z, 0.0), w)
    s2 = rolling_sum(np.where(ok, z * z, 0.0), w)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = s / np.maximum(n, 1)
        v = np.maximum(s2 / np.maximum(n, 1) - m * m, 0.0)
        return np.where(n >= min_count, np.sqrt(v * n / np.maximum(n - 1, 1)), np.nan)


def rolling_slope(z: np.ndarray, w: int, min_count: int = 3) -> np.ndarray:
    """OLS slope (per day) over the trailing window, NaN-aware."""
    ok = np.isfinite(z)
    T = z.shape[1]
    t = np.broadcast_to(np.arange(T, dtype=float), z.shape)
    t = t - t[:, :1]  # numerically small
    zz = np.where(ok, z, 0.0)
    tt = np.where(ok, t, 0.0)
    n = rolling_sum(ok.astype(float), w)
    st, sz = rolling_sum(tt, w), rolling_sum(zz, w)
    stz, stt = rolling_sum(tt * zz, w), rolling_sum(tt * tt, w)
    den = n * stt - st * st
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where((n >= min_count) & (den > 1e-9), (n * stz - st * sz) / den, np.nan)


def rolling_extreme(z: np.ndarray, w: int, kind: str = "min") -> np.ndarray:
    """Trailing-window nanmin/nanmax (NaN if the window is empty)."""
    S, T = z.shape
    fill = np.inf if kind == "min" else -np.inf
    zz = np.where(np.isfinite(z), z, fill)
    out = np.full((S, T), fill)
    f = np.minimum if kind == "min" else np.maximum
    for k in range(w):
        sh = np.full((S, T), fill)
        sh[:, k:] = zz[:, : T - k]
        out = f(out, sh)
    return np.where(np.isfinite(out), out, np.nan)


def lag(z: np.ndarray, k: int) -> np.ndarray:
    out = np.full_like(z, np.nan, dtype=float)
    if k == 0:
        return z.astype(float)
    out[:, k:] = z[:, :-k]
    return out


class FeatureBuilder:
    """Dense (S, T) causal feature arrays for one cube; rows are gathered with ``table``."""

    def __init__(self, cube, clim_mean, clim_std, static: pd.DataFrame, variables=None):
        self.cube = cube
        self.cm, self.cs = clim_mean, clim_std
        self.static = static
        self.variables = list(variables or cube.variables)
        self.doy = cube.dates.dayofyear.to_numpy() - 1
        self._z, self._last, self._age = {}, {}, {}
        huc = static["huc2"].astype(str).where(static["huc2"].astype(str).str.fullmatch(r"\d{2}"), "")
        self.huc2 = np.array([int(h) if h else -1 for h in huc], dtype=float)

    # --- per-variable primitives (cached)
    def z(self, v: str) -> np.ndarray:
        if v not in self._z:
            self._z[v] = to_model_space(v, self.cube.y[:, :, self.cube.var_index(v)])
        return self._z[v]

    def last(self, v: str):
        if v not in self._last:
            self._last[v], age = last_value(self.z(v), MAX_STALENESS)
            self._age[v] = np.where(np.isfinite(age), age, np.nan)
        return self._last[v], self._age[v]

    def clim_at(self, v: str, t_idx: np.ndarray, si: np.ndarray, shift: int = 0) -> tuple[np.ndarray, np.ndarray]:
        j = self.cube.var_index(v)
        T = len(self.doy)
        tt = np.minimum(t_idx + shift, T - 1)
        d = self.doy[tt]
        # for targets beyond the loaded range use the calendar DOY directly
        beyond = t_idx + shift >= T
        if beyond.any():
            d = d.copy()
            extra = (self.cube.dates[-1] + pd.to_timedelta(t_idx[beyond] + shift - (T - 1), "D")).dayofyear - 1
            d[beyond] = extra
        return self.cm[si, d, j], self.cs[si, d, j]

    def target_doy(self, t_idx: np.ndarray, h: int) -> np.ndarray:
        dates = self.cube.dates[np.minimum(t_idx, len(self.doy) - 1)] + pd.to_timedelta(h, "D")
        return dates.dayofyear.to_numpy() - 1

    # --- dense blocks
    def target_block(self, v: str) -> dict[str, np.ndarray]:
        """Dense features of the target variable itself, all causal, (S, T) each."""
        z = self.z(v)
        last, age = self.last(v)
        j = self.cube.var_index(v)
        cmd = self.cm[:, self.doy, j]
        csd = np.maximum(self.cs[:, self.doy, j], 1e-6)
        a = z - cmd  # anomaly of the actual observation (NaN if missing)
        out = {
            "last": last, "age": age,
            "anom0": last - cmd, "sanom0": (last - cmd) / csd,
            "anom_lag1": lag(a, 1), "anom_lag2": lag(a, 2),
            "anom_mean3": rolling_mean(a, 3), "anom_mean7": rolling_mean(a, 7, 2),
            "anom_mean30": rolling_mean(a, 30, 5),
            "slope7": rolling_slope(z, 7), "std7": rolling_std(z, 7),
            "min7": rolling_extreme(z, 7, "min"), "max7": rolling_extreme(z, 7, "max"),
            "miss0": (~np.isfinite(z)).astype(float),
            "frac_obs7": rolling_mean(np.isfinite(z).astype(float), 7),
            "frac_obs30": rolling_mean(np.isfinite(z).astype(float), 30),
        }
        return out

    def context_block(self, exclude: str | None = None) -> dict[str, np.ndarray]:
        """Anomalies (at d) of the other variables + discharge level/change."""
        out = {}
        for u in self.variables:
            if u == exclude:
                continue
            last, _ = self.last(u)
            j = self.cube.var_index(u)
            out[f"anom0_{u}"] = last - self.cm[:, self.doy, j]
        if DISCHARGE in self.cube.variables:
            q, _ = self.last(DISCHARGE)
            out["logq"] = q
            out["dlogq1"] = q - lag(q, 1)
            out["dlogq7"] = q - lag(q, 7)
        return out

    def weather_block(self, weather: np.ndarray | None) -> dict[str, np.ndarray]:
        """Trailing summaries of LAGGED observed weather (common.load_weather): index t already
        holds only values published by the end of day t."""
        if weather is None:
            return {}
        w = {n: weather[:, :, k].astype(float) for k, n in enumerate(
            ("precip", "tmax", "tmin", "srad", "wind", "vpd"))}
        out = {f"precip_sum{k}": rolling_mean(w["precip"], k) * k for k in (1, 3, 7, 30)}
        out.update({"tmax_mean3": rolling_mean(w["tmax"], 3), "tmax_mean7": rolling_mean(w["tmax"], 7),
                    "tmin_mean3": rolling_mean(w["tmin"], 3), "srad_mean3": rolling_mean(w["srad"], 3),
                    "wind_mean3": rolling_mean(w["wind"], 3), "vpd_mean3": rolling_mean(w["vpd"], 3),
                    "tmax_change7": rolling_mean(w["tmax"], 3) - lag(rolling_mean(w["tmax"], 3), 7)})
        return out

    def snow_block(self, weather: np.ndarray | None) -> dict[str, np.ndarray]:
        """Snow / freeze-thaw proxies from LAGGED observed weather (road-salt wash-off, snowmelt):
        precipitation on sub-freezing days and freezing degree-days over 7 days, warm days in 3."""
        if weather is None:
            return {}
        precip, tmax, tmin = (weather[:, :, k].astype(float) for k in (0, 1, 2))
        tmean = (tmax + tmin) / 2
        return {"snow_precip7": rolling_mean(np.where(tmax < 1.0, precip, 0.0), 7) * 7,
                "fdd7": rolling_mean(np.maximum(-tmean, 0.0), 7) * 7,
                "thaw_days3": rolling_mean((tmax > 2.0).astype(float), 3) * 3}

    def table(self, v: str, si: np.ndarray, ti: np.ndarray, h: int, *, threshold_model: float | None = None,
              blocks: dict | None = None) -> pd.DataFrame:
        """Feature table for rows (si, ti) and horizon h (target variable ``v``)."""
        if blocks is None:
            blocks = {**self.target_block(v), **self.context_block(exclude=v)}
        X = {k: arr[si, ti].astype("float32") for k, arr in blocks.items()}
        cm_t, cs_t = self.clim_at(v, ti, si, shift=h)
        X["clim_tgt"] = cm_t.astype("float32")
        X["clim_std_tgt"] = cs_t.astype("float32")
        d = self.target_doy(ti, h)
        X["doy_sin"] = np.sin(2 * np.pi * d / 365.25).astype("float32")
        X["doy_cos"] = np.cos(2 * np.pi * d / 365.25).astype("float32")
        if threshold_model is not None:
            X["margin_last"] = (X["last"] - threshold_model).astype("float32")
            X["margin_min7"] = (X["min7"] - threshold_model).astype("float32")
            X["margin_max7"] = (X["max7"] - threshold_model).astype("float32")
        for c in ("lat", "lon", "elev_m", "log_drain_area_km2") + tuple(
                c for c in self.static.columns if str(c).startswith("sc_")):
            X[c] = self.static[c].to_numpy(dtype=float)[si].astype("float32")
        X["huc2"] = self.huc2[si]
        df = pd.DataFrame(X)
        df["huc2"] = pd.Categorical(df["huc2"].astype(int), categories=list(range(-1, 23)))
        return df
