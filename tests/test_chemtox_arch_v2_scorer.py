"""Tests for the M4 arch-v2 scorer: rulings F1, F5, F6, F7, F8.

Synthetic arrays only. Nothing here reads a file, a row or the guard, so the whole statistical
contract is verifiable while the guard is closed.
"""

from __future__ import annotations

import numpy as np
import pytest

from sentinel.models.chemtox import arch_v2_scorer as S

BLOCK = "T1_test"


def _res(rng, n, scale):
    return rng.normal(0.0, scale, n)


def _seeds(rng, n, scale):
    return {s: _res(rng, n, scale) for s in S.SEEDS}


# ------------------------------------------------------------------------------- the metric, F6

def test_micro_rmse_is_the_root_mean_square():
    assert S.micro_rmse([3.0, 4.0]) == pytest.approx((25 / 2) ** 0.5)
    assert S.micro_rmse([1.0, -1.0]) == pytest.approx(1.0)
    assert S.micro_rmse([2.0, 2.0, 2.0]) == pytest.approx(2.0)


def test_micro_rmse_rejects_empty_and_nonfinite():
    with pytest.raises(S.ScorerError):
        S.micro_rmse([])
    with pytest.raises(S.ScorerError):
        S.micro_rmse([1.0, np.nan])


def test_seed_mean_difference_is_negative_when_the_candidate_is_better():
    rng = np.random.default_rng(0)
    cand = _seeds(rng, 400, 1.0)
    comp = _seeds(rng, 400, 1.3)
    d = S.seed_mean_difference(cand, comp)
    assert d < 0, "lower micro-RMSE is better, so a better candidate gives a negative difference"


def test_seed_mean_difference_requires_all_three_seeds():
    rng = np.random.default_rng(1)
    cand = {0: _res(rng, 10, 1.0), 1: _res(rng, 10, 1.0)}
    comp = _seeds(rng, 10, 1.0)
    with pytest.raises(S.ScorerError, match="missing seeds"):
        S.seed_mean_difference(cand, comp)


def test_seed_mean_difference_rejects_unexpected_seeds():
    rng = np.random.default_rng(2)
    cand = {0: _res(rng, 5, 1.0), 1: _res(rng, 5, 1.0), 2: _res(rng, 5, 1.0), 7: _res(rng, 5, 1.0)}
    with pytest.raises(S.ScorerError, match="unexpected seeds"):
        S.seed_mean_difference(cand, _seeds(rng, 5, 1.0))


def test_seed_mean_difference_requires_identical_rows():
    rng = np.random.default_rng(3)
    with pytest.raises(S.ScorerError, match="identical rows"):
        S.seed_mean_difference(_seeds(rng, 10, 1.0), _seeds(rng, 11, 1.0))


def test_seed_pairing_is_by_index():
    """Seed s of the candidate pairs with seed s of the comparator (F6)."""
    cand = {0: [0.0], 1: [0.0], 2: [3.0]}
    comp = {0: [3.0], 1: [0.0], 2: [0.0]}
    # the seed means are equal, so the difference is 0 even though no seed is a tie
    assert S.seed_mean_difference(cand, comp) == pytest.approx(0.0)
    # and all_seeds_better is False, because seed 2 goes the wrong way
    assert S.all_seeds_better(cand, {"x": comp}) is False


def test_all_seeds_better_requires_every_seed_and_every_comparator():
    rng = np.random.default_rng(4)
    good = {s: np.full(50, 0.1) for s in S.SEEDS}
    weak = {s: np.full(50, 1.0) for s in S.SEEDS}
    mixed = {0: np.full(50, 1.0), 1: np.full(50, 1.0), 2: np.full(50, 0.01)}
    assert S.all_seeds_better(good, {"a_arm": weak, "b_arm": weak}) is True
    assert S.all_seeds_better(good, {"a_arm": weak, "b_arm": mixed}) is False
    with pytest.raises(S.ScorerError):
        S.all_seeds_better(good, {})


# ------------------------------------------------------------------- F6  bootstrap and Holm p

def test_bootstrap_refuses_B_below_the_pre_registered_floor():
    rng = np.random.default_rng(5)
    cand, comp = _seeds(rng, 60, 1.0), _seeds(rng, 60, 1.0)
    with pytest.raises(S.ScorerError, match="forbids lowering"):
        S.paired_cluster_bootstrap(cand, comp, np.arange(60), B=2000)


def test_bootstrap_default_B_is_ten_thousand():
    assert S.B_HOLM_FAMILY == 10_000
    rng = np.random.default_rng(6)
    cand, comp = _seeds(rng, 40, 1.0), _seeds(rng, 40, 1.0)
    out = S.paired_cluster_bootstrap(cand, comp, np.repeat(np.arange(8), 5))
    assert out["B"] == 10_000 and out["n_clusters"] == 8 and out["n_rows"] == 40


def test_bootstrap_detects_a_real_difference_and_its_sign():
    rng = np.random.default_rng(7)
    n = 600
    cand = {s: _res(rng, n, 0.6) for s in S.SEEDS}
    comp = {s: _res(rng, n, 1.4) for s in S.SEEDS}
    out = S.paired_cluster_bootstrap(cand, comp, np.repeat(np.arange(120), 5), seed=1)
    assert out["favours_candidate"] is True
    assert out["excludes_zero"] is True
    assert out["ci_high"] < 0
    assert out["p_value"] < 0.05


def test_bootstrap_finds_no_difference_when_there_is_none():
    rng = np.random.default_rng(8)
    n = 400
    base = {s: _res(rng, n, 1.0) for s in S.SEEDS}
    out = S.paired_cluster_bootstrap(base, base, np.repeat(np.arange(80), 5), seed=2)
    assert out["point"] == pytest.approx(0.0)
    assert out["excludes_zero"] is False
    assert out["p_value"] > 0.05


def test_bootstrap_rejects_a_cluster_vector_of_the_wrong_length():
    rng = np.random.default_rng(9)
    cand, comp = _seeds(rng, 20, 1.0), _seeds(rng, 20, 1.0)
    with pytest.raises(S.ScorerError, match="cluster vector"):
        S.paired_cluster_bootstrap(cand, comp, np.arange(19))


def test_holm_p_value_formula_is_the_ruling_one():
    """2 * min(share <= 0, share >= 0), each share as (count + 1)/(B + 1)."""
    d = np.array([-1.0] * 90 + [1.0] * 10)
    # share >= 0 is (10 + 1)/(100 + 1) = 11/101; doubled = 22/101
    assert S.holm_p_value(d, B=100) == pytest.approx(2 * 11 / 101)
    d2 = np.array([-1.0] * 50 + [1.0] * 50)
    assert S.holm_p_value(d2, B=100) == pytest.approx(min(1.0, 2 * 51 / 101))


def test_holm_p_value_is_never_zero_and_never_above_one():
    allneg = np.full(10_000, -1.0)
    p = S.holm_p_value(allneg, B=10_000)
    assert 0.0 < p <= 1.0
    assert p == pytest.approx(2 * 1 / 10_001), "the +1 keeps a p-value off exactly zero"
    straddle = np.zeros(10)
    assert S.holm_p_value(straddle, B=10) == 1.0


def test_holm_adjust_is_step_down_and_monotone():
    p = {"a_leg": 0.001, "b_leg": 0.02, "c_leg": 0.3}
    out = S.holm_adjust(p, m=3)
    assert out["a_leg"] == pytest.approx(0.003)
    assert out["b_leg"] == pytest.approx(0.04)
    assert out["c_leg"] == pytest.approx(0.3)
    vals = [out[k] for k in sorted(out, key=lambda k: p[k])]
    assert vals == sorted(vals), "adjusted p-values must be non-decreasing"


def test_holm_adjust_uses_the_pre_registered_family_size():
    """T1's family is m = 14, larger than the legs one candidate contributes (§8.3)."""
    p = {"a_leg": 0.004, "b_leg": 0.006}
    small = S.holm_adjust(p, m=2)
    big = S.holm_adjust(p, m=14)
    assert big["a_leg"] > small["a_leg"], "a larger declared family is stricter"
    assert big["a_leg"] == pytest.approx(0.056)
    with pytest.raises(S.ScorerError, match="smaller"):
        S.holm_adjust(p, m=1)


# --------------------------------------------------------------------------------- F5  both legs

def test_evaluate_leg_runs_both_clusterings():
    rng = np.random.default_rng(10)
    n = 300
    cand = {s: _res(rng, n, 0.7) for s in S.SEEDS}
    comp = {s: _res(rng, n, 1.3) for s in S.SEEDS}
    clusters = {"holdout_unit": np.repeat(np.arange(60), 5),
                "reference_number": np.tile(np.arange(50), 6)}
    leg = S.evaluate_leg("a", "mmoe", cand, comp, clusters, B=10_000, seed=3)
    assert set(leg.by_clustering) == set(S.CLUSTERINGS)
    assert leg.favours_candidate is True
    assert leg.excludes_zero_under_both is True
    d = leg.as_dict()
    assert d["excludes_zero_under_both_clusterings"] is True
    assert set(d["p_values"]) == set(S.CLUSTERINGS)


def test_evaluate_leg_requires_both_clusterings():
    rng = np.random.default_rng(11)
    cand, comp = _seeds(rng, 30, 1.0), _seeds(rng, 30, 1.0)
    with pytest.raises(S.ScorerError, match="requires both"):
        S.evaluate_leg("a", "mmoe", cand, comp, {"holdout_unit": np.arange(30)})


def test_excludes_zero_under_both_is_false_if_one_clustering_fails():
    leg = S.LegResult(leg="a", comparator="mmoe", point=-0.2, by_clustering={
        "holdout_unit": {"excludes_zero": True},
        "reference_number": {"excludes_zero": False}})
    assert leg.excludes_zero_under_both is False


def test_leg_rejects_an_unknown_leg_name():
    rng = np.random.default_rng(12)
    with pytest.raises(S.ScorerError, match="leg must be"):
        S.evaluate_leg("c", "x", _seeds(rng, 10, 1.0), _seeds(rng, 10, 1.0),
                       {c: np.arange(10) for c in S.CLUSTERINGS})


# ------------------------------------------------------------------------------- F7  the MDE gate

def test_mde_gate_uses_the_analytic_value_and_says_so():
    out = S.mde_gate(BLOCK, -0.20)
    assert out["governing_mde"] == pytest.approx(0.0959)
    assert out["magnitude_exceeds_mde"] is True
    assert out["favours_candidate"] is True
    assert "not a gate" in out["source"]


def test_mde_gate_fails_a_realistic_difference():
    """The point of F1: a 0.03-0.07 differential is below every block's gate."""
    for block, mde in S.GOVERNING_MDE_ICC_0_5.items():
        for diff in (-0.029, -0.070):
            assert S.mde_gate(block, diff)["magnitude_exceeds_mde"] is False, (block, diff, mde)


def test_mde_gate_has_a_separate_no_shared_reference_table():
    a = S.mde_gate(BLOCK, -0.12, stratum="all")["governing_mde"]
    b = S.mde_gate(BLOCK, -0.12, stratum="no_shared_reference")["governing_mde"]
    assert b > a, "the stratum is smaller, so its MDE is larger"


def test_mde_gate_rejects_an_unknown_block():
    with pytest.raises(S.ScorerError, match="no governing MDE"):
        S.mde_gate("T9_test", -0.1)


def test_describe_mde_states_both_figures_and_the_caveat():
    s = S.describe_mde(BLOCK, bootstrap=0.0891)
    assert "0.0959" in s and "0.0891" in s
    assert "has not been verified" in s
    assert "not quoted as a gate" in S.describe_mde("T6a_test")


# ------------------------------------------------------------------------- F8 / F1  the wording

def _all_met():
    return {k: True for k in "abcdefghij"}


def test_outcome_all_conditions_met_is_not_a_sota_label():
    out = S.outcome_label(BLOCK, conditions=_all_met(), point=-0.2, comparator="mmoe")
    assert out["wording"] == ("exceeds the pre-registered MDE on an under-powered block; "
                              "no SOTA label")
    assert out["sota_label"] is False and out["label_available"] is False
    assert "descriptive only" in out["why_no_label"]


def test_outcome_ahead_with_an_unmet_condition():
    c = _all_met()
    c["f"] = False
    out = S.outcome_label(BLOCK, conditions=c, point=-0.05, comparator="mmoe")
    assert out["wording"] == "not resolvable on this block"
    assert out["conditions_unmet"] == ["f"]


def test_outcome_shared_reference_leg_is_its_own_wording():
    """F5: passing everything but (h) is reported, not labelled 'qualified'."""
    c = _all_met()
    c["h"] = False
    out = S.outcome_label(BLOCK, conditions=c, point=-0.2, comparator="mmoe")
    assert out["wording"] == ("ahead on the full block, not separable from publication structure")
    assert out["kind"] == "shared_reference"


def test_outcome_behind_is_no_gain_and_equally_prominent():
    out = S.outcome_label(BLOCK, conditions=_all_met(), point=0.04, comparator="mmoe")
    assert out["wording"] == "no gain"
    assert out["equal_prominence_required"] is True


def test_outcome_behind_with_a_ci_excluding_zero_names_the_comparator():
    out = S.outcome_label(BLOCK, conditions=_all_met(), point=0.3, comparator="gpboost_published",
                          behind_ci_excludes_zero=True)
    assert out["wording"] == "worse than gpboost_published"


def test_outcome_refuses_a_block_that_is_not_descriptive_only():
    with pytest.raises(S.ScorerError, match="its own stamped track"):
        S.outcome_label("some_new_block", conditions=_all_met(), point=-0.3, comparator="x")


def test_none_ahead_wording_points_at_research_plan_15_5():
    assert "15.5" in S.OUTCOME["none_ahead"]
    assert "harder or larger" in S.OUTCOME["none_ahead"]


def test_descriptive_only_blocks_are_exactly_the_three():
    assert S.DESCRIPTIVE_ONLY_BLOCKS == ("T1_test", "T6a_test", "T6b_test")


def test_constants_match_the_rulings():
    assert S.SEEDS == (0, 1, 2)
    assert S.CLUSTERINGS == ("holdout_unit", "reference_number")
    assert S.GOVERNING_MDE_ICC_0_5 == {"T1_test": 0.0959, "T6a_test": 0.1027, "T6b_test": 0.1033}
