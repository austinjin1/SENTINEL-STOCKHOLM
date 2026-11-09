# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Unit tests for sentinel/analysis (causal ladder, thresholds, E1). Synthetic arrays only (R9)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sentinel.analysis import thresholds as TH
from sentinel.analysis.causal import controls as C
from sentinel.analysis.causal import discovery as DI


def test_hc05_interpolation_rule():
    x = np.array([100.0, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700,
                  1800, 1900])   # N = 19 → P(R=1) = 0.05 exactly
    assert TH.hc05(x) == pytest.approx(100.0, rel=1e-9)


def test_xc95_uniform_occurrence():
    rs = np.random.default_rng(0)
    sc = 10 ** rs.uniform(1, 3, 20000)
    present = np.log10(sc) < 2.0           # genus present only below 100 µS/cm
    x = TH.xc95(sc, present)
    assert 85 < x < 100                     # 95th centile of a uniform log-occurrence on [1, 2]


def test_titan_finds_planted_change_point():
    rs = np.random.default_rng(1)
    n = 400
    sc = 10 ** rs.uniform(1.5, 3.5, n)
    ab = pd.DataFrame({f"dec{k}": np.where(sc < 300, rs.poisson(20, n), rs.poisson(0.2, n)) for k in range(4)}
                      | {f"noise{k}": rs.poisson(3, n) for k in range(3)})
    meta = pd.DataFrame({"sc": sc, "unit": np.arange(n).astype(str)})
    r = TH.titan(meta, ab, n_perm=50, n_boot=20, n_cuts=60)
    assert r["n_decreasers"] >= 3
    assert 200 < r["community_cp_minus"] < 450


def test_link_assumptions_forbid_river_to_weather():
    names = ["tair", "sw", "precip", "logq", "tw"]
    la = DI.link_assumptions(names, 3)
    # no lagged links into weather from river variables
    for j in (0, 1, 2):
        for (i, lag), mark in la[j].items():
            if i >= 3:
                assert lag == 0 and mark == "<?-"
    # contemporaneous weather→river oriented, consistent on both sides
    assert la[4][(0, 0)] == "-?>" and la[0][(4, 0)] == "<?-"
    assert la[4][(3, 0)] == "o?o" and la[3][(4, 0)] == "o?o"


def test_detection_and_recovery_scoring():
    B, N, T = 10, 3, 4
    g = np.zeros((B, N, N, T), np.uint8)
    v = np.zeros((B, N, N, T), np.float32)
    g[:, 0, 1, 1] = 1
    v[:, 0, 1, 1] = 0.3                 # 0 → 1 at lag 1, positive, all replicates
    g[:6, 1, 2, 0] = 3
    g[:6, 2, 1, 0] = 3                  # contemporaneous 1 – 2 in 60% of replicates
    v[:6, 1, 2, 0] = -0.2
    assert C.freq(g, v, 0, 1, lags=(0, 2), sign=+1) == 1.0
    assert C.freq(g, v, 0, 1, lags=(0, 2), sign=-1) == 0.0
    assert C.freq(g, v, 1, 2, lags=(0, 0)) == pytest.approx(0.6)
    truth = {("a", "b"), ("b", "c")}
    s = C.recovery_scores(g, v, ["a", "b", "c"], truth, [("a", "b"), ("b", "c"), ("c", "b"), ("a", "c")])
    assert s["tp"] == 2 and s["fn"] == 0
    assert s["fp"] == 0          # unoriented b–c with true b→c is not an FP for c→b
    assert s["tn"] == 2


def test_xcorr_baseline_detects_lagged_driver():
    rs = np.random.default_rng(0)
    n = 1500
    x = rs.normal(size=n)
    y = np.r_[0, 0.8 * x[:-1]] + 0.3 * rs.normal(size=n)
    z = rs.normal(size=n)
    g, r = DI.xcorr_baseline(np.column_stack([x, y, z]), ["x", "y", "z"], tau_max=3)
    assert g[0, 1, 1] == 1 and g[0, 2].sum() == 0


def test_pcmciplus_recovers_simple_chain():
    pytest.importorskip("tigramite")
    rs = np.random.default_rng(0)
    n = 800
    a = rs.normal(size=n)
    b = np.zeros(n)
    c = np.zeros(n)
    for t in range(1, n):
        b[t] = 0.5 * b[t - 1] + 0.7 * a[t - 1] + 0.5 * rs.normal()
        c[t] = 0.4 * c[t - 1] + 0.6 * b[t - 1] + 0.5 * rs.normal()
    x = np.column_stack([a, b, c])
    x[rs.random(n) < 0.05, 2] = np.nan          # missing values handled by the missing flag
    g, v, p = DI.run_pcmciplus(x, ["tair", "b", "c"], tau_max=2, constrained=True)
    assert g[0, 1, 1] == 1 and g[1, 2, 1] == 1
    assert g[0, 2, 1] == 0 and g[0, 2, 2] == 0   # indirect link removed


def test_e1_slope_recovers_solubility_at_constant_saturation():
    from sentinel.analysis.insights import e1
    from sentinel.models.sim import physics

    rs = np.random.default_rng(0)
    dates = pd.date_range("2008-06-01", "2012-08-31")
    dates = dates[np.isin(dates.month, (6, 7, 8))]
    n = len(dates)
    tw = 20 + 3 * rs.normal(size=n)
    sat = physics.do_sat_mgl(tw, 1.0)
    df = pd.DataFrame({"tw": tw, "do_mean": 0.95 * sat, "do_min": 0.8 * sat, "lq": rs.normal(size=n),
                       "sw": rs.normal(size=n), "ta": tw, "wy": dates.year}, index=dates)
    for k in ("tw", "lq", "sw"):
        df[f"a_{k}"] = df[k] - df[k].mean()
    df["a_mean"] = df["do_mean"] - df["do_mean"].mean()
    df["a_min"] = df["do_min"] - df["do_min"].mean()
    s = e1.site_slopes(df, 1.0)
    assert s["beta_bio_mean"] == pytest.approx(0.0, abs=0.01)
    assert s["beta_bio_min"] == pytest.approx(0.0, abs=0.01)
    assert s["beta_min"] > s["beta_mean"]     # smaller saturation fraction → smaller |slope|


# ----------------------------------------------------------------------------- review addendum fixes (2026-11-08)

def test_bh_adjust_matches_definition():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    q = DI.bh_adjust(p)
    assert q[0] == pytest.approx(0.01)
    assert np.all(np.diff(q[np.argsort(p)]) >= -1e-12) and q.max() <= 1


def test_fdr_prune_removes_links():
    N, T = 3, 3
    g = np.full((N, N, T), "", dtype="<U3")
    p = np.ones((N, N, T))
    g[0, 1, 1] = "-->"
    p[0, 1, 1] = 1e-6            # survives BH
    g[1, 2, 2] = "-->"
    p[1, 2, 2] = 0.009           # significant uncorrected at 0.01, not after BH over the family
    g[0, 2, 0], g[2, 0, 0] = "o-o", "o-o"
    p[0, 2, 0] = p[2, 0, 0] = 0.008
    mask = DI.tested_mask(N, T - 1, None)
    out, q = DI.fdr_prune(g, p, 0.01, mask)
    assert out[0, 1, 1] == "-->"
    assert out[1, 2, 2] == "" and out[0, 2, 0] == "" and out[2, 0, 0] == ""
    assert np.isfinite(q[2, 0, 0]) and q[2, 0, 0] == q[0, 2, 0]


def test_tested_mask_counts_contemporaneous_pairs_once():
    names = ["tair", "sw", "precip", "logq", "tw"]
    la = DI.link_assumptions(names, 2)
    m = DI.tested_mask(5, 2, la)
    assert m[:, :, 0].sum() == 5 * 4 // 2            # every unordered contemporaneous pair, once
    assert not m[3, 0, 1]                           # lagged river → weather is not tested (forbidden)


def test_reversal_asymmetry_sign():
    names = ["tair", "sw", "precip", "tw"]
    B = 10
    gf = np.zeros((B, 4, 4, 3), np.uint8)
    gr = np.zeros_like(gf)
    v = np.ones((B, 4, 4, 3), np.float32)
    gf[:, 0, 3, 1] = 1            # forward: tair → tw at lag 1 in every replicate
    gr[:5, 0, 3, 2] = 1           # reversed: half the replicates
    rows = {r["control"]: r for r in C.reversal_asymmetry(gf, v, gr, v, names, tau_max=2)}
    assert rows["N2asym:tair->tw"]["rev_minus_fwd"] == pytest.approx(-0.5)
    assert rows["N2asym:tair->tw"]["fwd_detected"]


def test_site_clusters_link_revisits_by_coordinates_and_unique_id():
    v = pd.DataFrame({"lat": [40.0, 40.001, 41.0, 42.0, 43.0], "lon": [-80.0, -80.001, -81.0, -82.0, -83.0],
                      "unique_id": [None, None, "U1", "U1", None]})
    c = TH.site_clusters(v)
    assert c.iloc[0] == c.iloc[1]          # ~140 m apart → same site
    assert c.iloc[2] == c.iloc[3]          # same unique_id, 140 km apart
    assert c.iloc[0] != c.iloc[2] and c.iloc[4] not in set(c.iloc[:4])


def test_titan_bootstrap_resamples_units():
    rs = np.random.default_rng(3)
    n = 120
    sc = 10 ** rs.uniform(1.5, 3.5, n)
    ab = pd.DataFrame({f"dec{k}": np.where(sc < 300, rs.poisson(20, n), rs.poisson(0.2, n)) for k in range(3)})
    meta = pd.DataFrame({"sc": np.repeat(sc[:60], 2), "unit": np.repeat(np.arange(60).astype(str), 2)})
    ab2 = pd.concat([ab.iloc[:60]] * 2).sort_index().reset_index(drop=True)
    r = TH.titan(meta, ab2, n_perm=20, n_boot=5, n_cuts=30)
    assert r["n_units"] == 60 and r["bootstrap_unit"] == "site cluster"


def test_e1_hac_calendar_gaps_not_adjacent():
    from sentinel.analysis.insights import e1

    rs = np.random.default_rng(0)
    x = rs.normal(size=(200, 1))
    y = 0.5 * x[:, 0] + rs.normal(size=200)
    t_contig = np.arange(200)
    t_far = np.arange(200) * 100            # no two observations within 7 days → plain White SE
    b1, se_contig, n = e1._ols_hac(y, x, t_contig)
    b2, se_far, _ = e1._ols_hac(y, x, t_far)
    assert b1 == pytest.approx(b2) and n == 200
    import statsmodels.api as sm

    w = sm.OLS(y, sm.add_constant(x)).fit(cov_type="HC0")
    assert se_far == pytest.approx(float(w.bse[1]), rel=1e-6)


def test_all_sim_scripts_parse():
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    files = sorted((root / "scripts" / "eval").glob("sim_*.py")) + sorted((root / "scripts" / "train").glob("sim_*.py"))
    assert files
    for f in files:
        ast.parse(f.read_text(), str(f))


def test_step1_selection_rule(tmp_path):
    import json

    from sentinel.analysis.causal.ladder import step1_selection

    def put(tag, rec, fpr):
        p = tmp_path / tag / "dev"
        p.mkdir(parents=True)
        (p / "step1_results.json").write_text(json.dumps({"summary": {
            "bagged_pcmciplus": {"recall": rec, "fpr": fpr, "n_sites": 20}, "config": {"boot": 50}}}))

    put("a", 0.70, 0.04)
    put("b", 0.85, 0.06)
    put("c", 0.75, 0.05)
    assert step1_selection(tmp_path, ("a", "b", "c"))["winner"] == "c"      # best recall with FPR <= 0.05
    assert step1_selection(tmp_path, ("b",))["winner"] == "b"              # none admissible → lowest FPR


class _FakeSim:
    def __init__(self, tw, ta, dates):
        from sentinel.models.sim import data as D

        self.dates = dates
        self.sites = ["s0"]
        self.basin_group = np.array(["g0"])
        self.y = np.full((1, len(dates), len(D.TARGETS)), np.nan, "float32")
        self.y[0, :, D.TARGETS.index("temp_c_mean")] = tw
        self.forcing = np.zeros((1, len(dates), 8), "float32")
        self.forcing[0, :, 0] = ta
        self.forcing[0, :, 1] = ta


def _sine_site(lag_days, amp_w=8.0, gap=None):
    dates = pd.date_range("2007-10-01", "2011-09-30", freq="D")
    t = np.arange(len(dates))
    w = 2 * np.pi / 365.25
    ta = 12 + 10 * np.sin(w * t)
    tw = 12 + amp_w * np.sin(w * (t - lag_days))
    if gap is not None:
        tw[gap[0]: gap[1]] = np.nan
    return _FakeSim(tw, ta, dates)


def test_e3_hare_rules_lag_clip_and_classes():
    from sentinel.analysis.insights import e3

    r = e3.site_metrics(_sine_site(-5.0), 0, wy_range=(2008, 2011))
    assert r is not None and r["phase_lag_d"] == 0.0 and r["hare_class"] == "atmospheric"
    assert e3.site_metrics(_sine_site(-20.0), 0, wy_range=(2008, 2011))["hare_class"] == "negative_lag"
    assert e3.site_metrics(_sine_site(15.0), 0, wy_range=(2008, 2011))["hare_class"] == "shallow_gw"
    assert e3.site_metrics(_sine_site(2.0, amp_w=4.0), 0, wy_range=(2008, 2011))["hare_class"] == "deep_gw"


def test_e3_gap_rule_drops_year():
    from sentinel.analysis.insights import e3

    full = e3.site_metrics(_sine_site(0.0), 0, wy_range=(2008, 2011))
    gapped = e3.site_metrics(_sine_site(0.0, gap=(400, 431)), 0, wy_range=(2008, 2011))   # 31-day gap in WY2009
    assert full["n_years"] == 4 and gapped["n_years"] == 3
