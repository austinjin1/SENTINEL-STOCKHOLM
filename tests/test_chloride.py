# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""H2 chloride regression and derived-series tests (synthetic arrays, tests only)."""

import numpy as np
import pandas as pd

from sentinel.eval import chloride as C
from sentinel.eval.hazards import HAZARDS, daily_onsets


def _pairs(n, start="2015-01-01", noise=5.0, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.date_range(start, periods=n, freq="15D")
    sc = rng.uniform(200, 1500, n)
    return pd.DataFrame({"date": d, "sc": sc, "cl": 0.25 * sc - 20 + rng.normal(0, noise, n)})


def test_pair_samples_median_and_inner_join():
    cl = pd.DataFrame({"date": ["2015-01-01", "2015-01-01", "2015-01-02", "2015-01-05"],
                       "value": [10.0, 20.0, np.nan, 7.0]})
    sc = pd.Series([100.0, 110.0, 120.0], index=pd.to_datetime(["2015-01-01", "2015-01-02", "2015-01-03"]))
    p = C.pair_samples(cl, sc)
    assert len(p) == 1 and p.loc[0, "cl"] == 15.0 and p.loc[0, "sc"] == 100.0


def test_fit_eligible_and_training_block_only():
    p = _pairs(40)
    late = _pairs(40, start="2022-01-01", seed=1)
    late["cl"] = 0.0  # post-training pairs must not influence the fit
    f = C.fit_site(pd.concat([p, late], ignore_index=True))
    assert f["eligible"] and f["n_pairs"] == 40 and abs(f["b"] - 0.25) < 0.01 and f["r2"] > 0.99


def test_fit_rejections():
    assert C.fit_site(_pairs(10))["reason"] == "n_pairs<20"
    assert C.fit_site(_pairs(40, noise=150.0))["reason"] == "r2<0.9"
    p = _pairs(40)
    p["sc"] = p["sc"] / 10  # samples only at low SC: criterion SC far beyond the data
    p["cl"] = 0.25 * p["sc"] - 20 + np.random.default_rng(2).normal(0, 0.5, len(p))
    assert C.fit_site(p)["reason"] == "extrapolation"


def test_derived_series_4day_requires_all_days_and_onset():
    fit = {"a": 0.0, "b": 0.25}
    idx = pd.date_range("2020-01-01", periods=20)
    sc = pd.Series(400.0, index=idx)          # 100 mg/L (safe)
    sc.iloc[12:] = 1200.0                     # 300 mg/L
    sc.iloc[0] = np.nan
    d = C.derived_series(sc, sc * 1.1, fit)
    assert np.isnan(d["cl_mgl_4dmean"].iloc[0:4]).all()
    assert d["cl_mgl_4dmean"].iloc[15] == 300.0
    ev, on = daily_onsets(d["cl_mgl_4dmean"].to_numpy(), HAZARDS["H2_chloride"])
    # 4-day mean: day 12 = 150, day 13 = 200 (ramp), day 14 = 250 -> first event day. With
    # pre_gap = 3 the clean window is days 4..10 (all 100 mg/L, observed) -> onset on day 14.
    assert np.flatnonzero(on).tolist() == [14] and ev[14]
    # without the gap the ramp day 13 (200 > 184) would block the onset
    import dataclasses
    nogap = dataclasses.replace(HAZARDS["H2_chloride"], pre_gap=0)
    assert not daily_onsets(d["cl_mgl_4dmean"].to_numpy(), nogap)[1].any()
