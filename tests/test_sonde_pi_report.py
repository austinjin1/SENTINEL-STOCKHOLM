# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Arithmetic of the numeric-forecast report (median, interval coverage/width, MAE/RMSE, skill vs a reference)."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from sentinel.models.baselines.sonde_baselines import QUANTILES

spec = importlib.util.spec_from_file_location(
    "sonde_pi_report", Path(__file__).resolve().parents[1] / "scripts" / "eval" / "sonde_pi_report.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)


def _q(center, half):
    """19 quantiles linear in the level: q(level) = center + (level-0.5)/0.45*half (q05..q95 = center ± half)."""
    lv = np.asarray(QUANTILES)
    return center[:, None] + (lv - 0.5)[None, :] / 0.45 * half


def test_report_one_metrics():
    n = 4
    obs = np.array([0.0, 1.0, 2.0, 10.0])
    rows = SimpleNamespace(score_mask={1: np.ones(n, bool)}, obs={1: obs}, site_no=np.array(["a"] * n),
                           issue_date=np.array(["2022-01-01"] * n, dtype="datetime64[ns]"))
    data = SimpleNamespace(horizons=(1,))
    med = np.array([0.0, 1.5, 2.0, 7.0])
    q = _q(med, 1.0)[:, None, :]                              # (n, H=1, 19); 90 % interval = med ± 1
    ref = _q(np.zeros(n), 1.0)[:, None, :]                  # persistence-like reference with median 0
    out, tab, sites = R.report_one(data, "temp_c_mean", "val", "m", q, rows, {"persistence": ref}, table=True,
                                   min_site_n=1)
    r = out[0]
    assert r["n"] == 4
    assert np.isclose(r["mae"], np.mean(np.abs(med - obs)))
    assert np.isclose(r["rmse"], np.sqrt(np.mean((med - obs) ** 2)))
    assert np.isclose(r["bias"], np.mean(med - obs))
    assert np.isclose(r["pi90_coverage"], 0.75)               # 10.0 is outside 7 ± 1
    assert np.isclose(r["pi90_mean_width"], 2.0)
    ref_mae = np.mean(np.abs(0 - obs))
    assert np.isclose(r["mae_skill_vs_persistence"], 1 - r["mae"] / ref_mae)
    assert len(tab) == 4 and np.allclose(tab["q50"], med)
    assert len(sites) == 1 and np.isclose(r["site_median_pi90_coverage"], 0.75)


def test_crossing_quantiles_are_sorted():
    n = 2
    rows = SimpleNamespace(score_mask={1: np.ones(n, bool)}, obs={1: np.array([0.0, 0.0])}, site_no=np.array(["a", "b"]),
                           issue_date=np.array(["2022-01-01"] * n, dtype="datetime64[ns]"))
    q = _q(np.zeros(n), 1.0)[:, None, ::-1]                   # fully reversed (crossed) quantiles
    out, _, _ = R.report_one(SimpleNamespace(horizons=(1,)), "temp_c_mean", "val", "m", q, rows, {}, table=False)
    assert np.isclose(out[0]["pi90_mean_width"], 2.0) and out[0]["pi90_coverage"] == 1.0


def test_cluster_bootstrap_matches_frozen_bootstrap_ci():
    """Per-group-sum bootstrap == sentinel.eval.bootstrap.bootstrap_ci on per-site totals (same draws)."""
    from sentinel.eval.bootstrap import bootstrap_ci

    rs = np.random.default_rng(1)
    n_sites = 40
    groups = rs.integers(0, 9, n_sites).astype(str)
    m, r = rs.uniform(1, 2, n_sites), rs.uniform(1.5, 3, n_sites)
    ref = bootstrap_ci(lambda a, b: 1 - a.sum() / b.sum(), m, r, groups=groups, n_boot=300, seed=7, return_samples=True)
    inv, counts = R.group_bootstrap_counts(groups, 300, 7)
    gm = np.bincount(inv, weights=m)
    gr = np.bincount(inv, weights=r)
    mine = 1 - (counts @ gm) / (counts @ gr)
    assert np.allclose(np.sort(mine), np.sort(ref["samples"]))
    assert np.allclose(mine, ref["samples"])


def test_report_one_bootstrap_ci_brackets_point():
    rs = np.random.default_rng(3)
    n = 600
    site = np.repeat(np.array([f"s{i}" for i in range(30)]), 20)
    obs = rs.normal(10, 2, n)
    med = obs + rs.normal(0, 1, n)
    rows = SimpleNamespace(score_mask={1: np.ones(n, bool)}, obs={1: obs}, site_no=site,
                           issue_date=np.array(["2022-01-01"] * n, dtype="datetime64[ns]"))
    q = _q(med, 1.645)[:, None, :]
    ref = _q(obs + rs.normal(0, 2, n), 1.0)[:, None, :]
    basin = {f"s{i}": f"g{i % 6}" for i in range(30)}
    out, _, _ = R.report_one(SimpleNamespace(horizons=(1,)), "temp_c_mean", "val", "m", q, rows,
                             {"persistence": ref}, table=False, basin=basin, n_boot=200, seed=0)
    r = out[0]
    assert r["n_basin_groups"] == 6
    for c in ("mae", "pi90_coverage", "mae_skill_vs_persistence"):
        assert r[f"{c}_ci_low"] <= r[c] <= r[f"{c}_ci_high"]
