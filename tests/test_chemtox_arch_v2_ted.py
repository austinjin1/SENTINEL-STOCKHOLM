"""Tests for ChemTox-TED and its ablations A1-A6. Synthetic tensors only."""

from __future__ import annotations

import pytest
import torch

from sentinel.models.chemtox import arch_v2_ted as T

N, F = 24, 10
DUR = torch.tensor([24.0, 48.0, 96.0, 168.0, 720.0, 26298.0])


def _batch(n=N, f=F, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(n, f, generator=g),
            "t": torch.full((n,), 48.0),
            "log_kow": torch.randn(n, generator=g),
            "log10_mw": torch.randn(n, generator=g) * 0.2 + 2.3}


def _ted(**kw):
    cfg = T.TedConfig(n_features=F, latent_dim=16, hidden=16, depth=1, **kw)
    torch.manual_seed(0)
    return T.TED(cfg), cfg


# ------------------------------------------------------------- the design claim: no time in trunk

def test_trunk_refuses_duration_and_endpoint_columns():
    for bad in ("duration_h", "log10_duration", "endpoint_kind", "duration_band", "band_lt24"):
        with pytest.raises(T.TedError, match="ONLY through the composition layer"):
            T.assert_no_time_in_trunk(["logp", "mw", bad])


def test_trunk_accepts_a_clean_feature_list():
    T.assert_no_time_in_trunk(["logp", "mw", "maccs_1", "emb_0"])


def test_a3_requires_a_time_column_to_be_present():
    """A3 is a leak test, so requesting it with no time column would measure nothing."""
    T.assert_no_time_in_trunk(["logp", "duration_h"], allow=True)
    with pytest.raises(T.TedError, match="would measure nothing"):
        T.assert_no_time_in_trunk(["logp", "mw"], allow=True)


# ----------------------------------------------------------------------------- the full model

def test_forward_shapes_and_keys():
    model, _ = _ted()
    out = model(**_batch())
    for k in ("acute", "chronic", "log10_acr", "log_sigma", "z", "k", "k_chronic"):
        assert k in out, k
    assert out["acute"].shape == (N,) and out["z"].shape == (N, 16)


def test_ordering_holds_by_construction():
    for seed in range(10):
        torch.manual_seed(seed)
        model, _ = _ted()
        with torch.no_grad():
            for p in model.parameters():
                p.mul_(5.0)
        out = model(**_batch(seed=seed))
        assert model.ordering_holds(out), (out["chronic"] - out["acute"]).max()


def test_prediction_is_monotone_in_duration():
    model, _ = _ted()
    b = _batch(n=DUR.shape[0])
    b["t"] = DUR
    b["x"] = b["x"][:1].expand(DUR.shape[0], F)
    out = model(**b)
    assert torch.all(torch.diff(out["acute"]) <= 1e-5), out["acute"]
    assert torch.all(torch.diff(out["chronic"]) <= 1e-5)


def test_log10_acr_is_non_negative():
    model, _ = _ted()
    out = model(**_batch())
    assert torch.all(out["log10_acr"] >= 0.0)


def test_gradients_reach_the_trunk_from_the_chronic_head():
    """A chronic row's gradient must move the shared latent: that is the multitask claim."""
    model, _ = _ted()
    b = _batch()
    out = model(**b)
    out["chronic"].sum().backward()
    grads = [p.grad for p in model.trunk.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)


def test_forward_validates_shapes():
    model, _ = _ted()
    with pytest.raises(T.TedError, match="n_features"):
        model(x=torch.randn(N), t=torch.ones(N), log_kow=torch.zeros(N), log10_mw=torch.zeros(N))
    with pytest.raises(T.TedError, match="one entry per row"):
        model(x=torch.randn(N, F), t=torch.ones(N + 1), log_kow=torch.zeros(N),
              log10_mw=torch.zeros(N))


# -------------------------------------------------------------------------------- ablations

def test_ablation_names_are_distinct_and_complete():
    assert T.ABLATIONS == ("A1", "A2", "A3", "A4", "A5", "A6")
    assert T.TedConfig(n_features=F).ablation_name() is None
    assert T.TedConfig(n_features=F, mechanism=False).ablation_name() == "A1"
    assert T.TedConfig(n_features=F, b_decades=0.0).ablation_name() == "A2"
    assert T.TedConfig(n_features=F, leak_time_into_trunk=True).ablation_name() == "A3"
    assert T.TedConfig(n_features=F, multitask=False).ablation_name() == "A4"
    assert T.TedConfig(n_features=F, bcf_head=True).ablation_name() == "A5"
    assert T.TedConfig(n_features=F, q10=True).ablation_name() == "A6"


def test_a1_has_no_composition_layer_and_need_not_be_monotone():
    """A1 is the TRIDENT-shaped control: duration is conditioned on, not composed."""
    model, _ = _ted(mechanism=False)
    assert model.head is None and hasattr(model, "direct")
    b = _batch(n=DUR.shape[0])
    b["t"] = DUR
    b["x"] = b["x"][:1].expand(DUR.shape[0], F)
    out = model(**b)
    assert out["acute"].shape == (DUR.shape[0],)
    assert torch.isfinite(out["acute"]).all()


def test_a2_is_the_pure_anchored_law():
    model, cfg = _ted(b_decades=0.0, anchor=(-2.0, 0.3, -0.1))
    assert cfg.b_decades == 0.0
    b = _batch()
    out = model(**b)
    want_logk = -2.0 + 0.3 * b["log_kow"] - 0.1 * b["log10_mw"]
    assert torch.allclose(torch.log10(out["k"]), want_logk, atol=1e-4)


def test_a4_drops_the_chronic_head():
    model, _ = _ted(multitask=False)
    out = model(**_batch())
    assert "chronic" not in out
    assert model.ordering_holds(out) is True


def test_a5_adds_a_bcf_head():
    model, _ = _ted(bcf_head=True)
    out = model(**_batch())
    assert "log10_bcf" in out and out["log10_bcf"].shape == (N,)
    model_off, _ = _ted()
    assert "log10_bcf" not in model_off(**_batch())


def test_a6_q10_is_learned_with_a_free_prior_and_needs_temperature():
    model, _ = _ted(q10=True)
    b = _batch()
    with pytest.raises(T.TedError, match="temperature_c"):
        model(**b)
    out = model(**b, temperature_c=torch.full((N,), 25.0))
    assert out["q10"].item() > 1.0, "Q10 must be above 1 by construction"
    # no literature constant is assumed: at initialisation it is NOT log 2
    assert abs(out["q10"].item() - 2.0) > 1e-3


def test_a6_shifts_with_temperature_in_one_direction():
    model, _ = _ted(q10=True)
    b = _batch()
    warm = model(**b, temperature_c=torch.full((N,), 30.0))["acute"]
    cold = model(**b, temperature_c=torch.full((N,), 10.0))["acute"]
    assert torch.all(warm < cold), "a higher temperature must not raise the lethal concentration"


# ----------------------------------------------------------------------------------- losses

def test_heteroscedastic_nll_prefers_the_right_sigma():
    pred = torch.zeros(500)
    target = torch.randn(500)
    good = T.heteroscedastic_nll(pred, target, torch.zeros(500))         # sigma = 1
    bad = T.heteroscedastic_nll(pred, target, torch.full((500,), 2.0))   # sigma = e^2
    assert good < bad


def test_censored_nll_handles_right_censoring_and_exact_rows():
    pred = torch.zeros(4)
    ls = torch.zeros(4)
    lower = torch.tensor([0.0, 1.0, -1.0, 2.0])
    upper = torch.tensor([0.0, float("inf"), 1.0, float("inf")])
    loss = T.censored_nll(pred, lower, upper, ls)
    assert torch.isfinite(loss) and loss.item() > 0


def test_censored_nll_is_lower_when_the_interval_contains_the_prediction():
    pred = torch.zeros(1)
    ls = torch.zeros(1)
    inside = T.censored_nll(pred, torch.tensor([-1.0]), torch.tensor([1.0]), ls)
    outside = T.censored_nll(pred, torch.tensor([5.0]), torch.tensor([6.0]), ls)
    assert inside < outside


def test_loss_terms_assembles_only_what_is_present():
    model, _ = _ted()
    out = model(**_batch())
    terms = T.loss_terms(out, {"acute": torch.randn(N)})
    assert set(terms) == {"acute", "total"}
    terms2 = T.loss_terms(out, {"acute": torch.randn(N), "chronic": torch.randn(N)})
    assert set(terms2) == {"acute", "chronic", "total"}
    assert torch.isfinite(terms2["total"])


def test_loss_terms_refuses_an_empty_target_set():
    model, _ = _ted()
    with pytest.raises(T.TedError, match="no loss term"):
        T.loss_terms(model(**_batch()), {})


def test_loss_includes_the_censored_term_when_bounds_are_given():
    model, _ = _ted()
    out = model(**_batch())
    terms = T.loss_terms(out, {"acute": torch.randn(N),
                               "censored_lower": torch.zeros(N),
                               "censored_upper": torch.full((N,), float("inf"))})
    assert "censored" in terms


def test_a_training_step_reduces_the_loss():
    """Not a performance claim: just proof the thing is trainable end to end."""
    model, _ = _ted()
    b = _batch()
    target = {"acute": torch.randn(N) * 0.1 - 4.0}
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = None
    for _ in range(30):
        opt.zero_grad()
        loss = T.loss_terms(model(**b), target)["total"]
        loss.backward()
        opt.step()
        first = loss.item() if first is None else first
    assert loss.item() < first
