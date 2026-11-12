"""Tests for the M4 arch-v2 entry registry and the non-neural comparators.

Synthetic frames only. The only file read is ``configs/chemtox/arch_v2.yaml``; no project row, no
pool row and no sealed row is touched, and the guard stays closed throughout.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sentinel.models.chemtox import arch_v2_arms as M

CFG = M.load_config()


def _frame(n=60, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "test_cas": rng.choice(["100", "200", "300"], n),
        "latin_name": rng.choice(["Danio rerio", "Daphnia magna"], n),
        "duration_h": rng.choice([24.0, 48.0, 96.0], n),
        "duration_band": "w24_96",
        "x1": rng.normal(size=n),
        "x2": rng.normal(size=n),
        "endpoint_kind": rng.choice([0, 1], n),
    })


# ------------------------------------------------------------------- the inventory and the caps

def test_config_loads_and_declares_both_tracks():
    assert set(CFG["entries"]) == {"T1", "T6"}
    assert CFG["meta"]["blocks"]["T1_test"]["status"] == "descriptive_only"


def test_validate_caps_reproduces_every_declared_figure():
    got = M.validate_caps(CFG)
    assert got["T1"]["searched_total"] == 384
    assert got["T6"]["searched_total"] == 128
    assert got["T1"]["fixed_configurations"] == 13
    assert got["T6"]["fixed_configurations"] == 13
    assert got["ablations_fixed"] == 12
    assert got["grand_total"] == 550
    assert got["entries_total"] == 44


def test_validate_caps_catches_a_per_entry_cap_breach():
    import copy
    bad = copy.deepcopy(CFG)
    bad["entries"]["T1"]["searched"]["ted"]["configs"] = 64
    with pytest.raises(M.ArmError, match="per-entry cap"):
        M.validate_caps(bad)


def test_validate_caps_catches_a_drifted_total():
    import copy
    bad = copy.deepcopy(CFG)
    bad["caps"]["grand_total"] = 544
    with pytest.raises(M.ArmError, match="grand total"):
        M.validate_caps(bad)


def test_validate_caps_catches_an_extra_entry():
    import copy
    bad = copy.deepcopy(CFG)
    bad["entries"]["T6"]["fixed"]["sneaked_in"] = {"role": "comparator", "configs": 1}
    with pytest.raises(M.ArmError):
        M.validate_caps(bad)


def test_every_searched_entry_sits_at_its_track_cap():
    for track, cap in (("T1", 32), ("T6", 16)):
        for name, spec in CFG["entries"][track]["searched"].items():
            assert spec["configs"] == cap, (track, name)


def test_an_entry_cannot_be_both_searched_and_fixed():
    import copy
    bad = copy.deepcopy(CFG)
    bad["entries"]["T1"]["fixed"]["ted"] = {"role": "candidate", "configs": 1}
    with pytest.raises(M.ArmError, match="both searched and fixed"):
        M.entries(bad, "T1")


def test_entries_rejects_an_unknown_track():
    with pytest.raises(M.ArmError, match="track must be"):
        M.entries(CFG, "T9")


# ------------------------------------------------------------------------- F3  the closed list

def test_condition_b_list_is_closed_and_consistent():
    lst = M.condition_b_list(CFG)
    assert "rf_emb_dur" in lst and "haber_rule" in lst and "readacross" in lst
    assert "ted" not in lst and "mmoe" not in lst
    assert len(lst) == len(set(lst))


def test_condition_b_excludes_candidates_and_published_arms():
    import copy
    bad = copy.deepcopy(CFG)
    bad["condition_b"]["closed_list"].append("ted")
    with pytest.raises(M.ArmError, match="must exclude candidates"):
        M.condition_b_list(bad)
    bad2 = copy.deepcopy(CFG)
    bad2["condition_b"]["closed_list"].append("gpboost_published")
    with pytest.raises(M.ArmError, match="must exclude candidates"):
        M.condition_b_list(bad2)


def test_condition_b_rejects_an_undeclared_entry():
    import copy
    bad = copy.deepcopy(CFG)
    bad["condition_b"]["closed_list"].append("nonexistent_arm")
    with pytest.raises(M.ArmError, match="not declared"):
        M.condition_b_list(bad)


# ---------------------------------------------------------------------------- F10  *_dur variants

def test_duration_variant_adds_duration_and_nothing_else():
    parent = ["maccs_1", "logp", "mw", "emb_0"]
    child = M.duration_variant(parent)
    assert child == parent + ["duration_h"]
    assert set(child) - set(parent) == {"duration_h"}


def test_duration_variant_refuses_a_parent_that_already_has_duration():
    with pytest.raises(M.ArmError, match="duplicate configuration"):
        M.duration_variant(["logp", "duration_h"])


def test_config_records_the_f10_rule_for_rf_emb_dur():
    spec = CFG["rf_emb_dur"]
    assert spec["derives_from"] == "rf_emb"
    assert spec["adds_features"] == ["duration_h"]
    assert spec["changes_nothing_else"] is True
    assert spec["configs"] == 1 and spec["track"] == "T6"
    # F10 / T6-BASELINE-LADDER: it is FIXED, so the 128 searched cap is untouched
    assert "rf_emb_dur" in CFG["entries"]["T6"]["fixed"]
    assert "rf_emb_dur" not in CFG["entries"]["T6"]["searched"]


# ----------------------------------------------------------------------------- parity (§4.1)

def test_feature_fingerprint_is_order_independent():
    a = M.feature_fingerprint(["b", "a", "c"])
    b = M.feature_fingerprint(["c", "b", "a"])
    assert a == b and a["n_features"] == 3


def test_assert_parity_accepts_matching_entries():
    rec = {n: {"n_rows": 100, "features": ["x1", "x2"], "aux_targets": ["H3"]}
           for n in ("ted", "r1_monotone_gbm", "xgboost_emb")}
    out = M.assert_parity(rec)
    assert out["n_rows"] == 100 and out["aux_targets"] == ["H3"]
    assert out["entries_checked"] == ["r1_monotone_gbm", "ted", "xgboost_emb"]


def test_assert_parity_catches_a_row_count_difference():
    rec = {"ted": {"n_rows": 100, "features": ["x1"]},
           "r1_monotone_gbm": {"n_rows": 99, "features": ["x1"]}}
    with pytest.raises(M.ArmError, match="row counts differ"):
        M.assert_parity(rec)


def test_assert_parity_catches_an_input_a_candidate_alone_received():
    """The C-M4P-1 failure mode: the candidate alone gets the embedding."""
    rec = {"ted": {"n_rows": 100, "features": ["x1", "emb_0"]},
           "xgboost_emb": {"n_rows": 100, "features": ["x1"]}}
    with pytest.raises(M.ArmError, match="feature sets differ"):
        M.assert_parity(rec)


def test_assert_parity_catches_an_auxiliary_target_difference():
    rec = {"ted": {"n_rows": 10, "features": ["x1"], "aux_targets": ["H2", "H3"]},
           "r3_multitask_stack_gbm": {"n_rows": 10, "features": ["x1"], "aux_targets": ["H3"]}}
    with pytest.raises(M.ArmError, match="auxiliary-target sets differ"):
        M.assert_parity(rec)


def test_assert_parity_exempts_published_form_arms():
    """§4.3: the published arms stay verbatim WITHOUT the embedding and are not parity-bound."""
    rec = {"ted": {"n_rows": 100, "features": ["x1", "emb_0"]},
           "xgboost_emb": {"n_rows": 100, "features": ["x1", "emb_0"]},
           "rf_published_adore": {"n_rows": 100, "features": ["maccs_1"], "published_form": True}}
    out = M.assert_parity(rec)
    assert out["published_form_exempt"] == ["rf_published_adore"]


def test_assert_parity_needs_two_parity_bound_entries():
    with pytest.raises(M.ArmError):
        M.assert_parity({"ted": {"n_rows": 1, "features": ["x"]}})


# ----------------------------------------------------------------------- the trivial baselines

def test_global_mean_predicts_the_training_mean():
    X = _frame(20)
    y = np.arange(20, dtype=float)
    m = M.GlobalMean().fit(X, y)
    p = m.predict(_frame(5, seed=1))
    assert np.allclose(p, y.mean()) and p.size == 5


def test_global_mean_refuses_before_fit_and_on_no_rows():
    with pytest.raises(M.ArmError):
        M.GlobalMean().predict(_frame(3))
    with pytest.raises(M.ArmError):
        M.GlobalMean().fit(_frame(0), [])


def test_group_mean_per_species_and_the_unseen_group_fallback():
    X = pd.DataFrame({"latin_name": ["a", "a", "b", "b"], "test_cas": ["1"] * 4})
    y = [1.0, 3.0, 10.0, 12.0]
    m = M.GroupMean(by=("latin_name",)).fit(X, y)
    assert m.predict(pd.DataFrame({"latin_name": ["a"], "test_cas": ["1"]}))[0] == pytest.approx(2.0)
    assert m.predict(pd.DataFrame({"latin_name": ["b"], "test_cas": ["1"]}))[0] == pytest.approx(11.0)
    # an unseen species falls back to the global training mean (§4.5's declared fallback)
    got = m.predict(pd.DataFrame({"latin_name": ["zzz"], "test_cas": ["1"]}))[0]
    assert got == pytest.approx(np.mean(y))


def test_group_mean_per_species_x_band():
    X = pd.DataFrame({"latin_name": ["a", "a"], "duration_band": ["lt24", "gt96_168"]})
    m = M.GroupMean(by=("latin_name", "duration_band")).fit(X, [1.0, 5.0])
    out = m.predict(X)
    assert out[0] == pytest.approx(1.0) and out[1] == pytest.approx(5.0)


def test_group_mean_requires_its_columns():
    with pytest.raises(M.ArmError, match="needs columns"):
        M.GroupMean(by=("not_a_column",)).fit(_frame(5), np.zeros(5))


# ------------------------------------------------------------------------------- Haber's rule

def test_haber_rule_is_exact_on_constant_c_times_t():
    """If C x t is constant then log10 C = a - log10 t exactly, so the fit is exact."""
    t = np.array([1.0, 2.0, 4.0, 24.0, 96.0])
    a = -3.0
    y = a - np.log10(t)
    X = pd.DataFrame({"duration_h": t})
    m = M.HaberRule().fit(X, y)
    assert m.intercept_ == pytest.approx(a)
    assert np.allclose(m.predict(X), y)


def test_haber_rule_extrapolates_in_duration():
    t = np.array([24.0, 48.0, 96.0])
    y = -3.0 - np.log10(t)
    m = M.HaberRule().fit(pd.DataFrame({"duration_h": t}), y)
    far = m.predict(pd.DataFrame({"duration_h": [26298.0]}))[0]
    assert far == pytest.approx(-3.0 - np.log10(26298.0))
    assert far < y.min(), "a longer exposure must give a lower lethal concentration"


def test_haber_rule_handles_nonpositive_duration_without_nan():
    m = M.HaberRule().fit(pd.DataFrame({"duration_h": [24.0, 48.0]}), [-4.0, -4.3])
    out = m.predict(pd.DataFrame({"duration_h": [0.0, np.nan, -5.0]}))
    assert np.isfinite(out).all()


def test_haber_rule_needs_a_duration_column():
    with pytest.raises(M.ArmError, match="duration_h"):
        M.HaberRule().fit(pd.DataFrame({"x": [1.0]}), [1.0])


# -------------------------------------------------------------- nearest-duration carry-forward

def test_carry_forward_returns_the_nearest_training_duration():
    X = pd.DataFrame({"test_cas": ["1", "1"], "latin_name": ["a", "a"],
                      "duration_h": [24.0, 96.0]})
    m = M.NearestDurationCarryForward().fit(X, [-4.0, -5.0])
    out = m.predict(pd.DataFrame({"test_cas": ["1", "1"], "latin_name": ["a", "a"],
                                  "duration_h": [30.0, 720.0]}))
    assert out[0] == pytest.approx(-4.0), "30 h is nearest to 24 h"
    assert out[1] == pytest.approx(-5.0), "720 h is nearest to 96 h -> a CONSTANT past the last"


def test_carry_forward_is_constant_past_the_last_observed_duration():
    """This is the structural handicap T6 exists to expose."""
    X = pd.DataFrame({"test_cas": ["1"] * 2, "latin_name": ["a"] * 2,
                      "duration_h": [24.0, 96.0]})
    m = M.NearestDurationCarryForward().fit(X, [-4.0, -5.0])
    q = pd.DataFrame({"test_cas": ["1"] * 3, "latin_name": ["a"] * 3,
                      "duration_h": [168.0, 720.0, 26298.0]})
    out = m.predict(q)
    assert len(set(np.round(out, 12))) == 1 and out[0] == pytest.approx(-5.0)


def test_carry_forward_falls_back_for_unseen_pairs():
    X = pd.DataFrame({"test_cas": ["1"], "latin_name": ["a"], "duration_h": [24.0]})
    m = M.NearestDurationCarryForward().fit(X, [-4.0])
    out = m.predict(pd.DataFrame({"test_cas": ["9"], "latin_name": ["z"], "duration_h": [24.0]}))
    assert np.isfinite(out).all()


# --------------------------------------------------------------------------- R1, R2, R3

def test_r1_puts_the_monotone_constraint_on_duration_only():
    cols = ["x1", "duration_h", "x2"]
    m = M.monotone_duration_gbm(cols)
    assert m._sentinel_monotone == {"x1": 0, "duration_h": -1, "x2": 0}
    assert list(m.get_params()["monotone_constraints"]) == [0, -1, 0]


def test_r1_enforces_monotonicity_on_synthetic_data():
    """The claim ruling §1.4 corrected: a tree CAN be monotone in duration."""
    rng = np.random.default_rng(0)
    n = 600
    t = rng.choice([24.0, 48.0, 72.0, 96.0], n)
    x1 = rng.normal(size=n)
    # a deliberately NON-monotone relationship in t; the constraint must suppress it
    y = 0.5 * x1 + 0.4 * np.sin(t / 12.0) + rng.normal(0, 0.05, n)
    X = pd.DataFrame({"x1": x1, "duration_h": t})
    m = M.monotone_duration_gbm(["x1", "duration_h"], n_estimators=60, num_leaves=15)
    m.fit(X, y)
    grid = pd.DataFrame({"x1": np.zeros(5), "duration_h": [24.0, 36.0, 48.0, 72.0, 96.0]})
    p = m.predict(grid)
    assert np.all(np.diff(p) <= 1e-9), f"must be non-increasing in duration, got {p}"


def test_r1_needs_duration_among_its_features():
    with pytest.raises(M.ArmError, match="needs"):
        M.monotone_duration_gbm(["x1", "x2"])


def test_r1_rejects_a_bad_direction():
    with pytest.raises(M.ArmError, match="direction"):
        M.monotone_duration_gbm(["duration_h"], direction=0)


def test_r2_returns_aft_parameters_that_xgboost_accepts():
    params, rounds = M.aft_gbm()
    assert params["objective"] == "survival:aft"
    assert params["aft_loss_distribution"] == "normal"
    assert rounds == 300


def test_r2_fits_interval_censored_targets():
    """The claim ruling §1.4 corrected: a GBM need NOT drop censored rows."""
    import xgboost as xgb

    rng = np.random.default_rng(1)
    n = 200
    x = rng.normal(size=(n, 2))
    lower = np.exp(rng.normal(0, 0.3, n))
    upper = lower * rng.choice([1.0, np.inf], n)       # half right-censored
    d = xgb.DMatrix(x, nthread=2)
    d.set_float_info("label_lower_bound", lower)
    d.set_float_info("label_upper_bound", upper)
    params, _ = M.aft_gbm()
    params["nthread"] = 2
    bst = xgb.train(params, d, num_boost_round=5)
    pred = bst.predict(d)
    assert pred.shape == (n,) and np.isfinite(pred).all()


def test_r2_rejects_an_unsupported_distribution():
    with pytest.raises(M.ArmError, match="distribution"):
        M.aft_gbm(distribution="cauchy")


def test_r3_requires_the_endpoint_indicator():
    with pytest.raises(M.ArmError, match="endpoint indicator"):
        M.multitask_stack_gbm(["x1", "x2"])
    m = M.multitask_stack_gbm(["x1", "endpoint_kind"])
    assert m._sentinel_endpoint_column == "endpoint_kind"


def test_r3_fits_over_two_endpoint_families():
    rng = np.random.default_rng(2)
    n = 400
    kind = rng.choice([0, 1], n)
    x1 = rng.normal(size=n)
    y = 0.8 * x1 - 1.5 * kind + rng.normal(0, 0.1, n)
    X = pd.DataFrame({"x1": x1, "endpoint_kind": kind})
    m = M.multitask_stack_gbm(["x1", "endpoint_kind"], n_estimators=40, num_leaves=7)
    m.fit(X, y)
    acute = m.predict(pd.DataFrame({"x1": [0.0], "endpoint_kind": [0]}))[0]
    chronic = m.predict(pd.DataFrame({"x1": [0.0], "endpoint_kind": [1]}))[0]
    assert chronic < acute, "the per-endpoint offset must be learned"


# ------------------------------------------------------------------------------- build_arm

@pytest.mark.parametrize("name", list(M._FIXED_BUILDERS))
def test_build_arm_builds_every_fixed_baseline(name):
    arm = M.build_arm(name)
    assert hasattr(arm, "fit") and hasattr(arm, "predict")


def test_build_arm_refuses_a_declared_but_unimplemented_arm():
    """Nothing may be reported as having run when its code does not exist."""
    for name in ("ted", "timehead", "mmoe", "rx_chemprop"):
        with pytest.raises(M.ArmNotImplemented, match="does not exist yet"):
            M.build_arm(name)


def test_build_arm_rejects_an_unknown_name():
    with pytest.raises(M.ArmError, match="unknown arm"):
        M.build_arm("not_an_arm")


def test_build_arm_wires_the_references():
    assert M.build_arm("r1_monotone_gbm", feature_columns=["x", "duration_h"]) is not None
    assert M.build_arm("r2_aft_gbm")[0]["objective"] == "survival:aft"
    assert M.build_arm("r3_multitask_stack_gbm",
                       feature_columns=["x", "endpoint_kind"]) is not None
    with pytest.raises(M.ArmError, match="needs feature_columns"):
        M.build_arm("r1_monotone_gbm")


def test_every_declared_entry_is_either_implemented_or_explicitly_not():
    """No entry may be silently missing from both the builders and the not-implemented list."""
    declared = set()
    for track in ("T1", "T6"):
        declared |= set(M.entries(CFG, track))
    known = set(M._FIXED_BUILDERS) | set(M.NEURAL_ARMS) | {
        "r1_monotone_gbm", "r2_aft_gbm", "r3_multitask_stack_gbm"}
    assert not declared - known, f"declared but unaccounted for: {sorted(declared - known)}"


def test_module_reads_only_the_config():
    from pathlib import Path
    src = Path(M.__file__).read_text()
    for banned in ("read_parquet", "ecotox_aquatic", "load_test_block"):
        assert banned not in src, banned
