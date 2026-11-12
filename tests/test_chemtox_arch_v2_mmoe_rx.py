"""Tests for the MMoE comparator and the RX arms. Synthetic data only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from sentinel.models.chemtox import arch_v2_mmoe as M
from sentinel.models.chemtox import arch_v2_rx as R

F = 12


def _mmoe(**kw):
    torch.manual_seed(0)
    return M.MMoE(M.MMoEConfig(n_features=F, n_experts=3, expert_hidden=16, expert_out=8,
                               tower_hidden=8, **kw))


# ------------------------------------------------------------------------------------ MMoE

def test_mmoe_shapes_and_gate_weights():
    model = _mmoe()
    out = model(torch.randn(10, F))
    assert out["acute"].shape == (10,) and out["chronic"].shape == (10,)
    for task in M.TASKS:
        w = out["gate_weights"][task]
        assert w.shape == (10, 3)
        assert torch.allclose(w.sum(dim=1), torch.ones(10), atol=1e-5)
        assert f"log_sigma_{task}" in out


def test_mmoe_ordering_holds_in_output_space():
    """L3 imposes EC50 >= EC10 in output space; TED derives it from one latent."""
    for seed in range(10):
        torch.manual_seed(seed)
        model = _mmoe()
        with torch.no_grad():
            for p in model.parameters():
                p.mul_(5.0)
        out = model(torch.randn(32, F) * 3)
        assert model.ordering_holds(out)


def test_mmoe_ordering_can_be_switched_off_for_a_diagnostic():
    model = _mmoe(enforce_ordering=False)
    out = model(torch.randn(16, F))
    assert "acute" in out and "chronic" in out


def test_ordered_outputs_is_a_softplus_gap():
    acute = torch.tensor([0.0, 1.0])
    raw = torch.tensor([10.0, -10.0])
    a, c = M.ordered_outputs(acute, raw)
    assert torch.all(c <= a)
    assert torch.allclose(a, acute)


def test_mmoe_config_validation():
    with pytest.raises(M.MMoEError, match="positive"):
        M.MMoEConfig(n_features=0)
    with pytest.raises(M.MMoEError, match="at least one task"):
        M.MMoEConfig(n_features=4, tasks=())
    with pytest.raises(M.MMoEError, match="ordering constraint assumes"):
        M.MMoEConfig(n_features=4, tasks=("chronic", "acute"))


def test_mmoe_rejects_the_wrong_feature_width():
    model = _mmoe()
    with pytest.raises(M.MMoEError, match="must be"):
        model(torch.randn(4, F + 1))


def test_mmoe_gradients_flow_to_every_expert():
    model = _mmoe()
    out = model(torch.randn(16, F))
    (out["acute"].sum() + out["chronic"].sum()).backward()
    for expert in model.experts:
        gs = [p.grad for p in expert.parameters() if p.grad is not None]
        assert gs and any(g.abs().sum() > 0 for g in gs)


# -------------------------------------------------------------------------------------- RX

def test_chemprop_must_be_single_task():
    with pytest.raises(R.RxError, match="inadmissible"):
        R.assert_single_task({"multitask": True})
    R.assert_single_task({"multitask": False})
    with pytest.raises(R.RxError, match="inadmissible"):
        R.RxChempropConfig(multitask=True)


def test_chemprop_params_force_single_task():
    p = R.rx_chemprop_params(R.RxChempropConfig())
    assert p["multitask"] is False


def test_species_dim_above_zero_is_disclosed_as_information_free():
    p = R.rx_chemprop_params(R.RxChempropConfig(species_dim=16))
    assert "INFORMATION-FREE" in p["_disclosure_species_dim"]
    assert "_disclosure_species_dim" not in R.rx_chemprop_params(R.RxChempropConfig(species_dim=0))


def test_placeholder_species_frame_does_not_touch_the_original():
    frame = pd.DataFrame({"tax_gs": ["Danio rerio", "Daphnia magna"], "x": [1, 2]})
    out = R.placeholder_species_frame(frame)
    assert list(out["tax_gs"]) == [R.SPECIES_PLACEHOLDER] * 2
    assert list(frame["tax_gs"]) == ["Danio rerio", "Daphnia magna"], "the pool is never modified"
    R.assert_no_species_key(out)


def test_a_real_species_key_is_refused():
    frame = pd.DataFrame({"tax_gs": ["Danio rerio", "Daphnia magna"]})
    with pytest.raises(R.RxError, match="distinct values"):
        R.assert_no_species_key(frame)
    wrong = pd.DataFrame({"tax_gs": ["something_else", "something_else"]})
    with pytest.raises(R.RxError, match="constant placeholder"):
        R.assert_no_species_key(wrong)
    R.assert_no_species_key(pd.DataFrame({"x": [1]}))      # no column at all is fine


def test_rx_gbm_requires_the_embedding_when_declared():
    cfg = R.RxGbmConfig(with_embedding=True)
    with pytest.raises(R.RxError, match="CONCATENATED"):
        R.build_rx_gbm(cfg, ["logp", "mw"])
    model = R.build_rx_gbm(cfg, ["logp", "mw", "emb_0", "emb_1"])
    assert model._sentinel_rx["n_embedding_columns"] == 2


def test_lightgbm_noemb_refuses_embedding_columns():
    cfg = R.RxGbmConfig(with_embedding=False)
    with pytest.raises(R.RxError, match="information control"):
        R.build_rx_gbm(cfg, ["logp", "emb_0"])
    assert R.build_rx_gbm(cfg, ["logp", "mw"]) is not None


def test_rx_gbm_duration_flag_requires_the_column():
    cfg = R.RxGbmConfig(with_embedding=True, with_duration=True)
    with pytest.raises(R.RxError, match="F10"):
        R.build_rx_gbm(cfg, ["logp", "emb_0"])
    assert R.build_rx_gbm(cfg, ["logp", "emb_0", "duration_h"]) is not None


def test_rx_gbm_catboost_family_and_bad_family():
    assert R.build_rx_gbm(R.RxGbmConfig(family="catboost"), ["emb_0", "x"]) is not None
    with pytest.raises(R.RxError, match="unsupported family"):
        R.RxGbmConfig(family="sklearn")


def test_rx_gbm_actually_fits_on_synthetic_data():
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame({"logp": rng.normal(size=n), "emb_0": rng.normal(size=n)})
    y = 0.7 * X["logp"] + 0.3 * X["emb_0"] + rng.normal(0, 0.1, n)
    model = R.build_rx_gbm(R.RxGbmConfig(n_estimators=30, num_leaves=7), list(X.columns))
    model.fit(X, y)
    pred = model.predict(X)
    assert np.isfinite(pred).all() and np.corrcoef(pred, y)[0, 1] > 0.8
