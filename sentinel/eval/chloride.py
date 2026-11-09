# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""H2 chloride hazard: site-specific chloride ~ specific-conductance regressions (EW-v2 §3).

Chloride is not measured continuously, so the H2 series is derived from the sonde specific
conductance (SC) with a per-site ordinary least-squares line ``Cl = a + b * SC`` fitted on
paired lab samples (WQP chloride on a day with a USGS daily-mean SC at the same station).

Pre-specified eligibility (docs/preregistration/ew_v2_protocol.md §3):
* pairs from the training block only (date <= ``FIT_END``), so the label definition is fixed
  before any validation/test data exist;
* >= ``MIN_PAIRS`` pairs and R^2 >= ``MIN_R2``; slope > 0;
* limited extrapolation: the SC at which the fitted line reaches the 4-day criterion must be
  <= ``MAX_EXTRAP`` x the largest paired SC (sites whose samples never come near the criterion
  range cannot define an excursion reliably).

Event series: ``cl_mgl_4dmean`` (4-day trailing mean of derived daily-mean chloride, all four
days observed) against 230 mg/L, and ``cl_mgl_max`` (derived from daily-max SC) against the
acute 860 mg/L criterion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FIT_END = pd.Timestamp("2021-09-30")   # end of the training block (WY2021)
MIN_PAIRS = 20
MIN_R2 = 0.9
MAX_EXTRAP = 1.5
CHRONIC_MGL = 230.0
ACUTE_MGL = 860.0


def pair_samples(cl: pd.DataFrame, sc_daily: pd.Series) -> pd.DataFrame:
    """Pair lab chloride samples (columns ``date``, ``value`` mg/L) with daily-mean SC (uS/cm).

    Censored/missing chloride values are dropped; same-day replicates are reduced to the median.
    """
    c = cl.dropna(subset=["value"]).copy()
    c["date"] = pd.to_datetime(c["date"]).dt.normalize()
    c = c.groupby("date", as_index=False)["value"].median()
    sc = sc_daily.dropna()
    sc.index = pd.to_datetime(sc.index).normalize()
    out = c.join(sc.rename("sc"), on="date", how="inner").rename(columns={"value": "cl"})
    return out[["date", "sc", "cl"]].reset_index(drop=True)


def fit_site(pairs: pd.DataFrame) -> dict:
    """OLS ``cl = a + b * sc`` on training-block pairs; returns the fit and eligibility."""
    p = pairs[pairs["date"] <= FIT_END]
    n = len(p)
    res = {"n_pairs": n, "a": np.nan, "b": np.nan, "r2": np.nan, "sc_max": np.nan,
           "sc_at_chronic": np.nan, "eligible": False, "reason": ""}
    if n < MIN_PAIRS:
        res["reason"] = f"n_pairs<{MIN_PAIRS}"
        return res
    x, y = p["sc"].to_numpy(float), p["cl"].to_numpy(float)
    if np.ptp(x) <= 0:
        res["reason"] = "no_sc_range"
        return res
    b, a = np.polyfit(x, y, 1)
    r2 = 1.0 - np.sum((y - (a + b * x)) ** 2) / np.sum((y - y.mean()) ** 2)
    res.update(a=float(a), b=float(b), r2=float(r2), sc_max=float(x.max()))
    if b <= 0:
        res["reason"] = "slope<=0"
        return res
    res["sc_at_chronic"] = float((CHRONIC_MGL - a) / b)
    if r2 < MIN_R2:
        res["reason"] = f"r2<{MIN_R2}"
    elif res["sc_at_chronic"] > MAX_EXTRAP * res["sc_max"]:
        res["reason"] = "extrapolation"
    else:
        res["eligible"] = True
    return res


def derived_series(sc_mean: pd.Series, sc_max: pd.Series | None, fit: dict) -> pd.DataFrame:
    """Daily derived chloride series for an eligible site (NaN where SC is missing)."""
    a, b = fit["a"], fit["b"]
    out = pd.DataFrame(index=sc_mean.index)
    out["cl_mgl_mean"] = (a + b * sc_mean).clip(lower=0)
    # 4-day trailing mean requires all four days observed (rolling skips NaN otherwise).
    out["cl_mgl_4dmean"] = out["cl_mgl_mean"].rolling(4, min_periods=4).mean()
    if sc_max is not None:
        out["cl_mgl_max"] = (a + b * sc_max.reindex(sc_mean.index)).clip(lower=0)
    return out
