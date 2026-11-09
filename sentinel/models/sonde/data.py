# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Daily-tier data access for SENTINEL-Sonde: dense site × date × variable cubes, training-only
climatologies, and leakage-safe issue-date sampling.

Transform convention: variables in ``LOG_VARS`` are modelled as log(x + offset); metrics are
reported in original units unless a metric explicitly says "log".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ...core import paths
from ...data.sonde_splits import TIME_BLOCKS, SondeSplit

# Daily statistics are part of a variable's identity: *_median (USGS stat 00008; e.g. pH and many
# turbidity sites publish only min/max/median) is a separate variable from *_mean (00003).
TARGETS = ["temp_c_mean", "do_mgl_mean", "do_mgl_min", "spcond_uscm_mean", "ph_mean", "ph_median",
           "turb_fnu_mean", "turb_fnu_median", "no3no2_mgl_mean", "chl_ugl_mean", "fdom_qse_mean",
           "discharge_cfs_mean"]
LOG_VARS = {"spcond_uscm_mean": 1.0, "turb_fnu_mean": 0.1, "turb_fnu_median": 0.1, "no3no2_mgl_mean": 0.01,
            "chl_ugl_mean": 0.1, "fdom_qse_mean": 0.1, "discharge_cfs_mean": 0.01}
SITE_DIR = paths.PROCESSED / "sonde_daily" / "sites"
CLIM_END = TIME_BLOCKS["train"][1]  # climatologies & per-site statistics: training years only


def to_model_space(var: str, x):
    x = np.asarray(x, dtype="float64")
    if var in LOG_VARS:
        return np.log(np.maximum(x, 0) + LOG_VARS[var])
    return x


def from_model_space(var: str, z):
    z = np.asarray(z, dtype="float64")
    if var in LOG_VARS:
        return np.maximum(np.exp(z) - LOG_VARS[var], 0.0)
    return z


@dataclass
class Cube:
    sites: list[str]
    dates: pd.DatetimeIndex
    variables: list[str]
    y: np.ndarray        # (S, T, V) float32, original units, NaN = missing
    prov: np.ndarray     # (S, T, V) bool, provisional flag

    def var_index(self, v: str) -> int:
        return self.variables.index(v)


def _read_site(p, retries: int, wait_s: float):
    """Read one site table; with retries > 0, wait for a missing/partially written file (the
    builder rewrites the directory in place) instead of silently skipping it."""
    import time

    for k in range(retries + 1):
        try:
            if p.exists():
                return pd.read_parquet(p)
        except Exception:  # noqa: BLE001 - partially written file during a rebuild
            if k == retries:
                raise
        if k < retries:
            time.sleep(wait_s)
    return None


def load_cube(sites, variables=TARGETS, start="2007-10-01", end="2026-09-30", *, retries: int = 0,
              wait_s: float = 15.0) -> Cube:
    """Dense cube; sites without a table stay all-NaN (default) — with ``retries`` > 0 a missing
    table is waited for ``retries`` × ``wait_s`` seconds first."""
    dates = pd.date_range(start, end, freq="D")
    S, T, V = len(sites), len(dates), len(variables)
    y = np.full((S, T, V), np.nan, dtype="float32")
    prov = np.zeros((S, T, V), dtype=bool)
    for i, s in enumerate(sites):
        p = SITE_DIR / f"{s}.parquet"
        df = _read_site(p, retries, wait_s)
        if df is None:
            continue
        df.index = pd.to_datetime(df.index)
        df = df.reindex(dates)
        for j, v in enumerate(variables):
            if v in df.columns:
                y[i, :, j] = df[v].to_numpy(dtype="float32")
                if f"{v}_prov" in df.columns:
                    prov[i, :, j] = df[f"{v}_prov"].astype("boolean").fillna(False).to_numpy(dtype=bool)
    return Cube(list(sites), dates, list(variables), y, prov)


def climatology(cube: Cube, *, window: int = 15, min_years: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Per site/variable day-of-year mean and std in MODEL space from training years only.

    Circular ±window smoothing over day-of-year; DOYs whose smoothed window has fewer than
    ``min_years`` distinct years of data fall back to the site's overall training mean/std.
    Returns arrays of shape (S, 366, V); NaN where a site has < 365 training days.
    """
    from scipy.ndimage import maximum_filter1d

    S, T, V = cube.y.shape
    train = cube.dates <= pd.Timestamp(CLIM_END)
    doy = cube.dates.dayofyear.to_numpy() - 1
    yidx = cube.dates.year.to_numpy() - cube.dates.year.min()
    n_years = int(yidx.max()) + 1
    k = np.ones(2 * window + 1)

    def circ(a):
        return np.convolve(np.concatenate([a[-window:], a, a[:window]]), k, "valid")

    mean = np.full((S, 366, V), np.nan)
    std = np.full((S, 366, V), np.nan)
    for j, v in enumerate(cube.variables):
        z = to_model_space(v, cube.y[:, :, j])
        z[:, ~train] = np.nan
        for i in range(S):
            zi = z[i]
            ok = np.isfinite(zi)
            if ok.sum() < 365:
                continue
            s_sum = circ(np.bincount(doy[ok], weights=zi[ok], minlength=366))
            s_sq = circ(np.bincount(doy[ok], weights=zi[ok] ** 2, minlength=366))
            s_cnt = circ(np.bincount(doy[ok], minlength=366).astype(float))
            pres = np.zeros((366, n_years), dtype=bool)
            pres[doy[ok], yidx[ok]] = True
            support = maximum_filter1d(pres.astype(np.uint8), size=2 * window + 1, axis=0,
                                       mode="wrap").sum(axis=1)
            m = s_sum / np.maximum(s_cnt, 1)
            var = np.maximum(s_sq / np.maximum(s_cnt, 1) - m ** 2, 1e-8)
            fb = support < min_years
            m[fb] = np.mean(zi[ok])
            var[fb] = np.var(zi[ok]) + 1e-8
            mean[i, :, j] = m
            std[i, :, j] = np.sqrt(var)
    return mean, std


def issue_dates(split: SondeSplit, block: str, *, horizon: int, context: int,
                dates: pd.DatetimeIndex) -> np.ndarray:
    """Indices into ``dates`` of valid issue dates for a block (embargo + window containment)."""
    return np.flatnonzero(split.time_mask(dates, block, horizon_days=horizon, context_days=context))
