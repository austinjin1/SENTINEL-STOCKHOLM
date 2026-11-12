"""Tests for the arch-v2 ensembles, their matched baseline controls, and the build dispatcher."""

from __future__ import annotations

import numpy as np
import pytest

from sentinel.models.chemtox import arch_v2_arms as ARMS
from sentinel.models.chemtox import arch_v2_build as B
from sentinel.models.chemtox import arch_v2_ensemble as EN


def _res(n=200, scale=1.0, seed=0, shift=0.0):
    rng = np.random.default_rng(seed)
    return rng.normal(shift, scale, n)


# --------------------------------------------------------------------------------- the basics

def test_rmse_and_its_guards():
    assert EN.rmse([3.0, 4.0]) == pytest.approx((25 / 2) ** 0.5)
    with pytest.raises(EN.EnsembleError):
        EN.rmse([])
    with pytest.raises(EN.EnsembleError):
        EN.rmse([1.0, np.nan])


def test_mean_ensemble_averages_predictions():
    out = EN.mean_ensemble({"a": [1.0, 3.0], "b": [3.0, 5.0]})
    assert out.tolist() == [2.0, 4.0]


def test_members_must_agree_on_row_count():
    with pytest.raises(EN.EnsembleError, match="row count"):
        EN.mean_ensemble({"a": [1.0, 2.0], "b": [1.0]})
    with pytest.raises(EN.EnsembleError, match="at least two"):
        EN.mean_ensemble({"a": [1.0]})


def test_residual_correlation_is_symmetric_with_unit_diagonal():
    r = {"a": _res(seed=1), "b": _res(seed=2), "c": _res(seed=3)}
    c = EN.residual_correlation(r)
    assert np.allclose(np.diag(c), 1.0)
    assert np.allclose(c, c.T)


def test_zero_variance_member_gets_correlation_zero():
    c = EN.residual_correlation({"a": np.zeros(10), "b": _res(10, seed=4)})
    assert c[0, 1] == 0.0


# ------------------------------------------------------------------- decorrelation selection

def test_margin_excludes_a_materially_worse_member():
    r = {"best": _res(scale=1.0, seed=1),
         "close": _res(scale=1.0, seed=2),
         "far": _res(scale=3.0, seed=3)}
    out = EN.select_decorrelated(r, margin=EN.MEMBER_MARGIN)
    assert "far" in out["excluded_by_margin"]
    assert "far" not in out["members"]
    assert out["margin"] == 0.05


def test_selection_prefers_the_decorrelated_member():
    """Two near-identical models and one independent one: the independent one must get in."""
    base = _res(seed=10)
    r = {"m1": base,
         "m2": base + 1e-6 * _res(seed=11),      # a near-duplicate of m1
         "m3": _res(seed=12)}                     # independent
    out = EN.select_decorrelated(r, margin=10.0, max_members=2)
    assert out["n_members"] == 2
    assert "m3" in out["members"], out
    assert out["mean_pairwise_decorrelation"] > 0.5


def test_selection_is_deterministic():
    r = {f"m{i}": _res(seed=i) for i in range(5)}
    a = EN.select_decorrelated(r, margin=10.0)
    b = EN.select_decorrelated(r, margin=10.0)
    assert a["members"] == b["members"], "the member set must be reproducible before the look"


def test_selection_reports_the_margin_disclosure():
    r = {"a": _res(seed=1), "b": _res(seed=2)}
    out = EN.select_decorrelated(r)
    assert "five times" in out["disclosure"] and "0.05" in out["disclosure"]
    assert out["best_single"] in ("a", "b")


def test_selection_rejects_a_negative_margin():
    with pytest.raises(EN.EnsembleError, match="non-negative"):
        EN.select_decorrelated({"a": _res(), "b": _res(seed=2)}, margin=-1.0)


# --------------------------------------------------------- the matched baseline control (parity)

def test_matched_control_is_required():
    with pytest.raises(EN.EnsembleError, match="C-M4P-1"):
        EN.require_matched_control("T1", candidate_members=["a", "b"], baseline_members=None)
    with pytest.raises(EN.EnsembleError, match="C-M4P-1"):
        EN.require_matched_control("T1", candidate_members=["a", "b"], baseline_members=[])


def test_matched_control_must_have_the_same_member_count():
    with pytest.raises(EN.EnsembleError, match="member counts differ"):
        EN.require_matched_control("T1", candidate_members=["a", "b", "c"],
                                   baseline_members=["x", "y"])


def test_matched_control_accepts_a_proper_pair_but_grants_no_label():
    out = EN.require_matched_control("T1", candidate_members=["a", "b"],
                                     baseline_members=["x", "y"])
    assert out["n_members"] == 2
    assert out["label_available_for_ensemble"] is False
    assert "descriptive only" in out["why"]


def test_no_ensemble_claim_exists_on_t6():
    with pytest.raises(EN.EnsembleError, match="no ensemble entry is declared"):
        EN.require_matched_control("T6", candidate_members=["a"], baseline_members=["x"])
    assert EN.NO_ENSEMBLE_TRACKS == ("T6",)


def test_candidate_members_cannot_be_empty():
    with pytest.raises(EN.EnsembleError, match="no members"):
        EN.require_matched_control("T1", candidate_members=[], baseline_members=["x"])


# ------------------------------------------------------------------------- the dispatcher

def test_coverage_accounts_for_every_declared_entry():
    cov = B.coverage()
    assert cov["unaccounted"] == [], cov["unaccounted"]
    assert cov["declared"] == 30
    assert "ted" in cov["learned"] and "r1_monotone_gbm" in cov["delegated"]


def test_dispatcher_builds_the_learned_arms():
    cols = ["logp", "mw", "emb_0", "emb_1"]
    assert B.build("ted", feature_columns=cols, latent_dim=8, hidden=8, depth=1) is not None
    assert B.build("timehead", feature_columns=cols, hidden=8, depth=1) is not None
    assert B.build("mmoe", feature_columns=cols, n_experts=2, expert_hidden=8,
                   expert_out=4, tower_hidden=4) is not None
    assert B.build("rx_chemprop")["multitask"] is False
    assert B.build("rx_lightgbm", feature_columns=cols, n_estimators=10) is not None
    assert B.build("lightgbm_noemb", feature_columns=["logp", "mw"], n_estimators=10) is not None


def test_dispatcher_refuses_time_in_the_ted_trunk():
    from sentinel.models.chemtox import arch_v2_ted as TED
    with pytest.raises(TED.TedError, match="ONLY through the composition layer"):
        B.build("ted", feature_columns=["logp", "duration_h", "emb_0"], latent_dim=8,
                hidden=8, depth=1)


def test_dispatcher_applies_f10_for_the_dur_variants():
    cols = ["logp", "emb_0"]
    parent = B.build("rf_emb", feature_columns=cols, n_estimators=5)
    child = B.build("rf_emb_dur", feature_columns=cols, n_estimators=5)
    assert parent._sentinel_feature_columns == cols
    assert child._sentinel_feature_columns == cols + ["duration_h"]
    assert set(child._sentinel_feature_columns) - set(cols) == {"duration_h"}


def test_dispatcher_refuses_a_parent_that_already_has_duration():
    with pytest.raises(B.BuildError, match="F10"):
        B.build("rf_emb", feature_columns=["logp", "emb_0", "duration_h"], n_estimators=5)


def test_parity_arms_need_the_embedding():
    with pytest.raises(B.BuildError, match="R4 parity"):
        B.build("xgboost_emb", feature_columns=["logp", "mw"], n_estimators=5)


def test_dispatcher_names_the_reason_for_everything_it_will_not_build():
    for name, reason in B.NOT_BUILDABLE.items():
        with pytest.raises(B.BuildError, match="not built by this dispatcher"):
            B.build(name)
        assert len(reason) > 20, name


def test_dispatcher_delegates_to_the_pinned_arms_module():
    assert B.build("trivial_global_mean") is not None
    assert B.build("haber_rule") is not None
    assert B.build("r1_monotone_gbm", feature_columns=["x", "duration_h"]) is not None


def test_dispatcher_rejects_an_unknown_arm():
    with pytest.raises(ARMS.ArmError, match="unknown arm"):
        B.build("not_an_arm")


def test_the_pinned_arms_module_was_not_edited():
    """arch_v2_arms.build_arm still refuses the learned arms; the dispatcher is the entry point."""
    with pytest.raises(ARMS.ArmNotImplemented):
        ARMS.build_arm("ted")
