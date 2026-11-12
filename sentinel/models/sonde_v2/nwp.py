# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Archived NWP forecasts as known-future inputs for SENTINEL-Sonde v2 (``sonde_arch_v2_amendment.md`` §2c).

Layout. An :class:`NWPTensor` holds, per product, ``x[s, t, k, f]`` = the forecast of field f for VALID day
``t + k + 1`` available at the end of issue day t (lead k + 1, k = 0..L−1), and ``m[s, t, k]`` = availability.
Fields (``FIELDS``): tmax_c, tmin_c, tmean_c, precip_mm, srad_wm2. For Open-Meteo Previous Runs the forecast for valid
day d at lead day ℓ is the row (date_local = d, lead_day = ℓ), whose contributing runs were initialised no later than
``max_init_utc`` (≈ 00 UTC of d − ℓ + 1, available about 4 h later, i.e. by the end of local issue day d − ℓ in
CONUS); so issue t uses lead ℓ = k + 1 for valid t + k + 1.

Bias correction (amendment §2c): per site × lead × field × month of the valid day, a linear map forecast → gridMET
fitted on the pairs selected by ``fit_mask`` (issue dates) and ``valid_end`` (valid days) only. Order of the rules per
cell: (1) site-month map where the site has ≥ ``min_days`` pairs in that month; else (3) the pooled map of the site's
group (HUC2) for that month where the group has ≥ ``POOL_MIN_PAIRS`` pairs; else (2) the site's all-months map where
it has ≥ ``min_days`` pairs; else (0) identity (the forecast is passed through). Slopes are clipped to
``SLOPE_CLIP``. Precipitation is corrected by the ratio of means (no intercept, ratio NOT clipped), floored at 0.

Perfect-prog training input (stage 1): observed gridMET of the target days plus a lead-, field- and season-dependent
Gaussian error with AR(1) correlation across leads, fitted on the same train-block overlap
(:class:`ErrorModel`). Validation and test always use the archived issued forecasts.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

FIELDS = ("tmax_c", "tmin_c", "tmean_c", "precip_mm", "srad_wm2")
F_IDX = {f: i for i, f in enumerate(FIELDS)}
MAX_LEAD = 14
POOL_MIN_PAIRS = 3            # minimum finite pairs for any fitted map (pooled group-month maps included)
SLOPE_CLIP = (0.5, 1.5)       # slope clip of the linear maps (not applied to the precipitation ratio)


@dataclass
class NWPTensor:
    """One product. ``x`` (S, T, L, F) float32 (NaN = missing), ``m`` (S, T, L) bool (all fields of a lead present
    or the lead is unusable for temperature; precipitation/radiation availability is ``np.isfinite`` per field)."""

    name: str
    x: np.ndarray
    sites: list
    dates: pd.DatetimeIndex
    meta: dict = field(default_factory=dict)

    @property
    def m(self) -> np.ndarray:
        return np.isfinite(self.x[..., F_IDX["tmean_c"]])

    @property
    def L(self) -> int:
        return int(self.x.shape[2])

    def field_avail(self, f: str) -> np.ndarray:
        return np.isfinite(self.x[..., F_IDX[f]])


def from_daily_table(df: pd.DataFrame, sites: list, dates: pd.DatetimeIndex, *, name: str,
                     site_col: str = "site_key", site_prefix: str = "USGS-", max_lead: int = 7) -> NWPTensor:
    """Open-Meteo-style daily table (date_local, lead_day, fields) → NWPTensor aligned to issue dates."""
    S, T = len(sites), len(dates)
    x = np.full((S, T, max_lead, len(FIELDS)), np.nan, dtype=np.float32)
    pos = {f"{site_prefix}{s}": i for i, s in enumerate(sites)}
    d = df[df[site_col].isin(pos) & (df["lead_day"] >= 1) & (df["lead_day"] <= max_lead)]
    if d.empty:
        return NWPTensor(name, x, list(sites), dates)
    si = d[site_col].map(pos).to_numpy()
    lead = d["lead_day"].to_numpy(int)
    issue = pd.DatetimeIndex(pd.to_datetime(d["date_local"])) - pd.to_timedelta(lead, unit="D")
    ti = dates.get_indexer(issue)
    ok = ti >= 0
    for j, f in enumerate(FIELDS):
        if f in d:
            x[si[ok], ti[ok], lead[ok] - 1, j] = d[f].to_numpy(np.float32)[ok]
    return NWPTensor(name, x, list(sites), dates, {"source_rows": int(ok.sum())})


def observed_future(weather_daily: np.ndarray, L: int) -> np.ndarray:
    """Perfect-prog 'forecast' from OBSERVED daily weather w[s, t, f] (unlagged, valid day t): out[s, t, k, f] =
    w[s, t + k + 1, f]. TRAINING ONLY (amendment §2c)."""
    S, T, F = weather_daily.shape
    out = np.full((S, T, L, F), np.nan, dtype=np.float32)
    for k in range(L):
        out[:, : T - k - 1, k] = weather_daily[:, k + 1:]
    return out


# ----------------------------------------------------------------------------- bias correction

@dataclass
class BiasMap:
    """a + b·x per (site, lead, field[, month]); precipitation: b·x (ratio of means)."""

    a: np.ndarray            # (S, L, F, 12)
    b: np.ndarray            # (S, L, F, 12)
    source: np.ndarray       # (S, L, F, 12) int8: 0 identity, 1 site-month, 2 site-all-months, 3 group pooled

    def apply(self, x: np.ndarray, dates: pd.DatetimeIndex) -> np.ndarray:
        """x (S, T, L, F) → corrected copy; month = month of the VALID day."""
        S, T, L, F = x.shape
        out = np.empty_like(x)
        valid_month = np.stack([(dates + pd.Timedelta(days=k + 1)).month.to_numpy() - 1 for k in range(L)], axis=1)
        for k in range(L):
            mo = valid_month[:, k]                            # (T,)
            a = self.a[:, k][:, :, mo].transpose(0, 2, 1)     # (S, T, F)
            b = self.b[:, k][:, :, mo].transpose(0, 2, 1)
            out[:, :, k] = a + b * x[:, :, k]
        p = F_IDX["precip_mm"]
        out[..., p] = np.maximum(out[..., p], 0.0)
        return out


def _fit_lin(xf: np.ndarray, yo: np.ndarray, ratio: bool) -> tuple[float, float] | None:
    ok = np.isfinite(xf) & np.isfinite(yo)
    if ok.sum() < POOL_MIN_PAIRS:
        return None
    xf, yo = xf[ok], yo[ok]
    if ratio:
        mx = xf.mean()
        return (0.0, float(yo.mean() / mx)) if mx > 1e-6 else (0.0, 1.0)
    vx = xf.var()
    if vx < 1e-9:
        return float(yo.mean() - xf.mean()), 1.0
    b = float(np.cov(xf, yo, bias=True)[0, 1] / vx)
    b = float(np.clip(b, *SLOPE_CLIP))                      # guard against degenerate short overlaps
    return float(yo.mean() - b * xf.mean()), b


def fit_bias_map(fc: np.ndarray, obs_future: np.ndarray, dates: pd.DatetimeIndex, fit_mask: np.ndarray,
                 groups: np.ndarray, *, min_days: int = 60, valid_end=None) -> BiasMap:
    """Fit on issue dates where ``fit_mask`` (T,) is True ONLY; with ``valid_end`` a pair of lead k + 1 is used only
    if its VALID day (issue + k + 1) is ≤ ``valid_end`` (so no pair reaches past the fit window through the lead).
    ``obs_future`` = observed weather aligned like ``fc`` (see :func:`observed_future`); ``groups`` (S,) e.g. HUC2
    for the pooled fallback."""
    S, T, L, F = fc.shape
    a = np.zeros((S, L, F, 12))
    b = np.ones((S, L, F, 12))
    src = np.zeros((S, L, F, 12), dtype=np.int8)
    fm0 = np.asarray(fit_mask, bool)
    groups = np.asarray(groups)
    for k in range(L):
        vday = dates + pd.Timedelta(days=k + 1)
        fm = fm0 if valid_end is None else fm0 & np.asarray(vday <= pd.Timestamp(valid_end))
        mo = vday.month.to_numpy()[fm] - 1
        for f in range(F):
            ratio = FIELDS[f] == "precip_mm"
            X, Y = fc[:, fm, k, f], obs_future[:, fm, k, f]           # (S, Tf)
            pooled = {}
            for g in np.unique(groups):
                gi = groups == g
                for mm in range(12):
                    sel = mo == mm
                    r = _fit_lin(X[gi][:, sel].ravel(), Y[gi][:, sel].ravel(), ratio)
                    if r is not None:
                        pooled[(g, mm)] = r
            for s in range(S):
                fit_all = _fit_lin(X[s], Y[s], ratio)
                n_all = int((np.isfinite(X[s]) & np.isfinite(Y[s])).sum())
                for mm in range(12):
                    sel = mo == mm
                    n_m = int((np.isfinite(X[s, sel]) & np.isfinite(Y[s, sel])).sum())
                    if n_m >= min_days:
                        a[s, k, f, mm], b[s, k, f, mm] = _fit_lin(X[s, sel], Y[s, sel], ratio)
                        src[s, k, f, mm] = 1
                    elif (groups[s], mm) in pooled:
                        a[s, k, f, mm], b[s, k, f, mm] = pooled[(groups[s], mm)]
                        src[s, k, f, mm] = 3
                    elif fit_all is not None and n_all >= min_days:
                        a[s, k, f, mm], b[s, k, f, mm] = fit_all
                        src[s, k, f, mm] = 2
    return BiasMap(a, b, src)


def bias_source_counts(bm: BiasMap) -> dict:
    """Cells (site × lead × month) per field and source code 0–3, plus the range of the fitted precipitation
    ratios (weather only; for the preregistration record)."""
    out = {}
    for f, name in enumerate(FIELDS):
        c = np.bincount(bm.source[:, :, f].ravel(), minlength=4)
        out[name] = {str(i): int(c[i]) for i in range(4)}
    pi = F_IDX["precip_mm"]
    pb = bm.b[:, :, pi][bm.source[:, :, pi] > 0]
    out["precip_ratio"] = ({"n": int(pb.size), "min": float(pb.min()), "p01": float(np.quantile(pb, 0.01)),
                            "median": float(np.median(pb)), "p99": float(np.quantile(pb, 0.99)),
                            "max": float(pb.max())} if pb.size else {"n": 0})
    return out


# ----------------------------------------------------------------------------- error model (perfect-prog)

@dataclass
class ErrorModel:
    """Residual (forecast − observed) per lead × field × season: mean, sd; AR(1) correlation across leads."""

    mean: np.ndarray     # (L, F, 4)
    sd: np.ndarray       # (L, F, 4)
    rho: np.ndarray      # (F,)

    @staticmethod
    def season(month0: np.ndarray) -> np.ndarray:
        return ((month0 + 1) % 12) // 3              # DJF 0, MAM 1, JJA 2, SON 3

    def sample(self, obs_future: np.ndarray, dates: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
        S, T, L, F = obs_future.shape
        sea = np.stack([self.season((dates + pd.Timedelta(days=k + 1)).month.to_numpy() - 1)
                        for k in range(L)], axis=1)                                   # (T, L)
        e = np.empty((S, T, L, F), dtype=np.float32)
        z = rng.standard_normal((S, T, F))
        for k in range(L):
            if k:
                z = self.rho[None, None, :] * z + np.sqrt(1 - self.rho[None, None, :] ** 2) * \
                    rng.standard_normal((S, T, F))
            mu = self.mean[k][:, sea[:, k]].T          # (T, F)
            sd = self.sd[k][:, sea[:, k]].T
            e[:, :, k] = mu[None] + sd[None] * z
        out = obs_future + e
        p = F_IDX["precip_mm"]
        out[..., p] = np.maximum(out[..., p], 0.0)
        return out.astype(np.float32)


def fit_error_model(fc: np.ndarray, obs_future: np.ndarray, dates: pd.DatetimeIndex, fit_mask: np.ndarray,
                    L: int | None = None) -> ErrorModel:
    """Fit on the issue dates of ``fit_mask`` only (the caller restricts them so that every valid day lies inside
    the fit window). Leads beyond the product's horizon reuse the last fitted lead, inflated by sqrt(k/k_last)
    (random-walk growth)."""
    S, T, Lp, F = fc.shape
    L = L or Lp
    r = fc[:, fit_mask] - obs_future[:, fit_mask]                       # (S, Tf, Lp, F)
    sea = np.stack([ErrorModel.season((dates[fit_mask] + pd.Timedelta(days=k + 1)).month.to_numpy() - 1)
                    for k in range(Lp)], axis=1)                         # (Tf, Lp)
    mean = np.zeros((L, F, 4))
    sd = np.ones((L, F, 4))
    for k in range(L):
        kk = min(k, Lp - 1)
        infl = np.sqrt((k + 1) / (kk + 1))
        for f in range(F):
            for q in range(4):
                v = r[:, sea[:, kk] == q, kk, f].ravel()
                v = v[np.isfinite(v)]
                if v.size >= 30:
                    mean[k, f, q], sd[k, f, q] = float(v.mean()), float(v.std()) * infl
                else:
                    allv = r[:, :, kk, f].ravel()
                    allv = allv[np.isfinite(allv)]
                    if allv.size >= 30:
                        mean[k, f, q], sd[k, f, q] = float(allv.mean()), float(allv.std()) * infl
    rho = np.zeros(F)
    for f in range(F):
        a, b = r[:, :, :-1, f].ravel(), r[:, :, 1:, f].ravel()
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() >= 30 and a[ok].std() > 0 and b[ok].std() > 0:
            rho[f] = float(np.clip(np.corrcoef(a[ok], b[ok])[0, 1], 0.0, 0.99))
    return ErrorModel(mean, sd, rho)


# ----------------------------------------------------------------------------- multi-product combination

def masked_mean_products(products: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Field-wise mean over the products available at each (s, t, k, f), no zero-fill. Utility only: in this round
    every arm reads exactly ONE declared product (GEFS c00), so no driver calls it. Returns (x, n)."""
    st = np.stack(products, axis=0)
    n = np.isfinite(st).sum(axis=0)
    with np.errstate(invalid="ignore"):
        x = np.nansum(st, axis=0) / np.where(n > 0, n, np.nan)
    return x.astype(np.float32), n.astype(np.int8)


# ----------------------------------------------------------------------------- GBM-NWP features (per horizon)

def gbm_nwp_blocks(x: np.ndarray, h: int, *, base_c: float = 0.0, decay: float = 0.7) -> dict[str, np.ndarray]:
    """(S, T) feature arrays for horizon h from the ONE declared product x (S, T, L, F) (bias-corrected):
    per-lead Tmax/Tmin/Tmean for leads 1..hh, hh = min(h, L); trajectory summaries over leads 1..hh: mean Tmean,
    cumulative degree-days above ``base_c`` (``nwp_dd_cum``), exponentially weighted degree-days (``nwp_dd_ew``,
    weight decay^(hh − lead), most weight on the last lead), P sum and max, SW mean; the forecast of the last
    available lead hh (``nwp_tmean_target``; the target day itself only when h ≤ L); availability flags.
    Every summary is NaN (never 0) where no lead of the product is finite."""
    S, T, L, F = x.shape
    hh = min(h, L)
    tm, tx, tn = x[..., F_IDX["tmean_c"]], x[..., F_IDX["tmax_c"]], x[..., F_IDX["tmin_c"]]
    p, sw = x[..., F_IDX["precip_mm"]], x[..., F_IDX["srad_wm2"]]
    out: dict[str, np.ndarray] = {}
    for k in range(hh):
        out[f"nwp_tmean_l{k + 1}"] = tm[:, :, k]
        out[f"nwp_tmax_l{k + 1}"] = tx[:, :, k]
        out[f"nwp_tmin_l{k + 1}"] = tn[:, :, k]
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        any_t = np.isfinite(tm[:, :, :hh]).any(axis=2)
        any_p = np.isfinite(p[:, :, :hh]).any(axis=2)
        out["nwp_tmean_mean"] = np.nanmean(tm[:, :, :hh], axis=2)
        dd = np.maximum(tm[:, :, :hh] - base_c, 0.0)                       # NaN where the lead is missing
        w = decay ** (hh - 1 - np.arange(hh))
        out["nwp_dd_cum"] = np.where(any_t, np.nansum(dd, axis=2), np.nan)
        out["nwp_dd_ew"] = np.where(any_t, np.nansum(dd * w, axis=2), np.nan)
        out["nwp_p_sum"] = np.where(any_p, np.nansum(p[:, :, :hh], axis=2), np.nan)
        out["nwp_p_max"] = np.where(any_p, np.nanmax(np.where(np.isfinite(p[:, :, :hh]), p[:, :, :hh], -np.inf),
                                                     axis=2), np.nan)
        out["nwp_sw_mean"] = np.nanmean(sw[:, :, :hh], axis=2)
    out["nwp_tmean_target"] = tm[:, :, hh - 1]
    out["nwp_avail_t"] = np.isfinite(tm[:, :, :hh]).mean(axis=2).astype(np.float32)
    out["nwp_avail_p"] = np.isfinite(p[:, :, :hh]).mean(axis=2).astype(np.float32)
    # all-NaN slices from nanmean/nanmax are expected (missing product)
    return {k: v.astype(np.float32) for k, v in out.items()}
