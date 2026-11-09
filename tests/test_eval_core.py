# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Unit tests for the shared evaluation core.

Synthetic arrays here only exercise code paths; no test output is a research result.
"""

import json

import numpy as np
import pandas as pd
import pytest

from sentinel.core import registry
from sentinel.eval import bootstrap as B
from sentinel.eval import early_warning as EW
from sentinel.eval import metrics as M
from sentinel.eval import splits as S


# ----------------------------------------------------------------------------- metrics

def test_regression_perfect_and_known_values():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert M.rmse(y, y) == 0
    assert M.nse(y, y) == pytest.approx(1.0)
    assert M.kge(y, y) == pytest.approx(1.0)
    assert M.nse(y, np.full_like(y, y.mean())) == pytest.approx(0.0)
    assert M.bias(y, y + 1) == pytest.approx(1.0)


def test_log_metrics_factor_two():
    y = np.array([0.5, 1.0, 10.0, 100.0])
    assert M.mdsa(y, 2 * y) == pytest.approx(100.0)
    assert M.sspb(y, 2 * y) == pytest.approx(100.0)
    assert M.sspb(y, 0.5 * y) == pytest.approx(-100.0)
    assert M.log_slope(y, 3 * y) == pytest.approx(1.0)


def test_nan_pairs_dropped_not_imputed():
    y = np.array([1.0, np.nan, 3.0])
    p = np.array([1.0, 2.0, np.nan])
    assert M.rmse(y, p) == 0.0
    assert np.isnan(M.auroc([1, 1, 1], [0.1, 0.2, 0.3]))


def test_crps_gaussian_matches_ensemble():
    rs = np.random.default_rng(0)
    y = rs.normal(size=200)
    mu, sd = np.zeros(200), np.ones(200)
    ens = rs.normal(size=(200, 4000))
    assert M.crps_gaussian(y, mu, sd) == pytest.approx(M.crps_ensemble(y, ens), rel=0.03)


def test_brier_skill_and_contingency():
    y = np.array([0, 0, 1, 1])
    assert M.brier_skill_score(y, y.astype(float), 0.5) == pytest.approx(1.0)
    assert M.brier_skill_score(y, np.full(4, 0.5), 0.5) == pytest.approx(0.0)
    c = M.contingency_scores([1, 1, 0, 0], [1, 0, 1, 0])
    assert c["pod"] == 0.5 and c["far"] == 0.5 and c["pofd"] == 0.5 and c["csi"] == pytest.approx(1 / 3)


# ------------------------------------------------------------------------------ splits

def _grid_sites(n=400, seed=0):
    rs = np.random.default_rng(seed)
    lat = rs.uniform(30, 45, n)
    lon = rs.uniform(-110, -75, n)
    return [f"s{i}" for i in range(n)], lat, lon


def test_spatial_block_split_disjoint_and_buffer():
    sites, lat, lon = _grid_sites()
    sp = S.spatial_block_split(sites, lat, lon, block_km=200, test_frac=0.2, val_frac=0.1,
                               seed=1, buffer_km=50)
    S.assert_disjoint(train=sp.train, val=sp.val, test=sp.test)
    lat_of, lon_of = dict(zip(sites, lat)), dict(zip(sites, lon))
    rep = S.split_report(sp, lat_of, lon_of)
    assert rep["test_to_train_km_min"] > 50
    assert rep["n_test"] > 0 and rep["n_train"] > rep["n_test"]


def test_assert_disjoint_raises():
    with pytest.raises(AssertionError):
        S.assert_disjoint(a=[1, 2], b=[2, 3])


def test_time_split_embargo():
    ts = S.time_split("2020-01-01", "2021-01-01", embargo="7D")
    times = pd.date_range("2019-01-01", "2022-01-01", freq="D")
    m = ts.masks(times)
    assert not (m["train"] & m["test"]).any()
    assert times[m["train"]].max() < pd.Timestamp("2019-12-25")
    assert times[m["test"]].min() >= pd.Timestamp("2021-01-01")


def test_spatial_kfold_covers_all_sites():
    sites, lat, lon = _grid_sites()
    folds = S.spatial_kfold(sites, lat, lon, n_folds=5, block_km=300, seed=0)
    assert set(folds) == set(sites) and set(folds.values()) == set(range(5))


# --------------------------------------------------------------------------- bootstrap

def test_bootstrap_ci_contains_point_and_paired_detects_difference():
    rs = np.random.default_rng(0)
    y = rs.normal(size=500)
    good = y + rs.normal(scale=0.3, size=500)
    bad = y + rs.normal(scale=1.0, size=500)
    groups = np.repeat(np.arange(50), 10)
    ci = B.bootstrap_ci(M.rmse, y, good, groups=groups, n_boot=300)
    assert ci["ci_low"] <= ci["point"] <= ci["ci_high"]
    res = B.paired_bootstrap(M.rmse, y, good, bad, groups=groups, n_boot=300, higher_is_better=False)
    assert res["diff"] < 0 and res["ci_high"] < 0 and res["p_value"] < 0.05


# ---------------------------------------------------------------------- early warning

def _synthetic_do(seed=0):
    """Hourly DO with three hypoxic dips; code-path test only."""
    rs = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", "2021-12-31 23:00", freq="h")
    do = 8 + np.sin(np.arange(len(idx)) * 2 * np.pi / 24) + rs.normal(0, 0.2, len(idx))
    onsets = [pd.Timestamp("2020-07-10 03:00"), pd.Timestamp("2021-06-01 05:00"),
              pd.Timestamp("2021-08-20 02:00")]
    for on in onsets:
        sel = (idx >= on) & (idx < on + pd.Timedelta("10h"))
        do[sel] = 1.0
    s = pd.Series(do, index=idx)
    s[(idx >= "2021-03-01") & (idx < "2021-03-05")] = np.nan  # a data gap
    return s, onsets


def test_extract_events_and_labels():
    s, onsets = _synthetic_do()
    ev = EW.extract_events(s, 2.0, direction="below", min_duration="2h", site="x")
    assert [e.onset for e in ev] == onsets
    assert not any(e.onset_censored for e in ev)
    issue = EW.make_issue_times(s.index.min(), s.index.max(), "1D")
    lab = EW.onset_labels(issue, ev, "3D", series=s, obs_freq="1h")
    assert lab.loc[lab["y"] == 1].shape[0] == 9  # 3 issue days before each onset
    assert not lab.loc[pd.Timestamp("2021-03-02"), "valid"]  # inside data gap -> unknown


def test_calibrated_threshold_respects_pofd():
    rs = np.random.default_rng(0)
    scores = rs.random(10000)
    y = np.zeros(10000, int)
    thr = EW.calibrate_threshold(scores, y, np.ones(10000, bool), 0.05)
    assert np.mean(scores >= thr) <= 0.05 + 1e-9
    assert np.mean(scores >= thr) > 0.045


def _frames_for(s, ev, scores, horizon="3D"):
    issue = EW.make_issue_times(s.index.min(), s.index.max(), "1D")
    lab = EW.onset_labels(issue, ev, horizon, series=s, obs_freq="1h")
    f = lab.assign(score=scores(issue, lab), usable=True)
    return {"x": f}


def test_placebo_random_scores_not_significant_oracle_significant():
    s, _ = _synthetic_do()
    ev = EW.extract_events(s, 2.0, direction="below", site="x")
    rs = np.random.default_rng(1)
    rand = _frames_for(s, ev, lambda issue, lab: rs.random(len(issue)))
    out_r = EW.evaluate_site_set(rand, {"x": ev}, horizon="3D", threshold=0.9,
                                 n_placebo_per_site=200)
    assert out_r["placebo_fisher_p"] > 0.05
    oracle = _frames_for(s, ev, lambda issue, lab: lab["y"].to_numpy(float))
    out_o = EW.evaluate_site_set(oracle, {"x": ev}, horizon="3D", threshold=0.5,
                                 n_placebo_per_site=200)
    assert out_o["event_detection_rate"] == 1.0 and out_o["placebo_detection_rate"] == 0.0
    assert out_o["median_lead_first_h"] > 48


# ---------------------------------------------------------------------------- registry

def test_registry_roundtrip(tmp_path):
    p = tmp_path / "reg.jsonl"
    e = registry.log_result(workstream="t", experiment="e", model="m", metric="rmse", value=1.5,
                            split="test", n=10, registry_path=p)
    got = registry.lookup("e", "m", "rmse", "test", registry_path=p)
    assert got["value"] == 1.5 and got["id"] == e["id"]
    registry.log_result(workstream="t", experiment="e", model="m", metric="rmse", value=1.2,
                        split="test", supersedes=e["id"], registry_path=p)
    assert registry.lookup("e", "m", "rmse", "test", registry_path=p)["value"] == 1.2
    with pytest.raises(ValueError):
        registry.log_result(workstream="t", experiment="e", model="m", metric="x",
                            value=float("nan"), split="test", registry_path=p)
    assert all(json.loads(l)["code"]["package_hash"] for l in p.read_text().splitlines())
