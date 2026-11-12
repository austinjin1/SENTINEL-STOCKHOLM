"""Tests for the arch-v2 time law: the properties the pre-registration asserts, enforced.

Synthetic tensors only; no project row, no file, no GPU required.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from sentinel.models.chemtox import arch_v2_timelaw as L

DURATIONS = torch.tensor([1.0, 6.0, 24.0, 48.0, 96.0, 168.0, 720.0, 2160.0, 8035.5, 26298.0])


# ------------------------------------------------------------------------ the saturation term

def test_saturation_term_is_non_increasing_in_t_over_a_random_sweep():
    """The core claim: d/dt <= 0 analytically, over random rates and the full duration range."""
    g = torch.Generator().manual_seed(0)
    for _ in range(200):
        k = torch.pow(10.0, torch.empty(1).uniform_(-6.0, 1.0, generator=g))
        vals = L.saturation_term(k.expand(DURATIONS.shape), DURATIONS)
        d = torch.diff(vals)
        assert torch.all(d <= 1e-6), (k.item(), vals)


def test_saturation_term_is_non_negative_and_tends_to_zero():
    k = torch.full((5,), 0.5)
    t = torch.tensor([1.0, 10.0, 50.0, 200.0, 1000.0])
    v = L.saturation_term(k, t)
    assert torch.all(v >= 0)
    assert v[-1] < 1e-6, "y_inf must be the asymptote"
    assert v[0] > v[-1]


def test_saturation_term_is_decreasing_in_k():
    """Needed for the EC10 <= EC50 proof: a larger k gives a smaller term."""
    t = torch.full((4,), 48.0)
    ks = torch.tensor([0.001, 0.01, 0.1, 1.0])
    v = L.saturation_term(ks, t)
    assert torch.all(torch.diff(v) < 0), v


def test_saturation_term_matches_an_fp64_reference():
    """fp16 destroys expm1 near zero, so the layer is fp32: check it against fp64."""
    t = DURATIONS.to(torch.float64)
    k = torch.full_like(t, 1e-4)
    kt = torch.clamp(k * t, min=L.KT_MIN, max=L.KT_MAX)
    ref = -(1.0 / math.log(10.0)) * torch.log(-torch.expm1(-kt))
    got = L.saturation_term(k.to(torch.float32), t.to(torch.float32))
    assert torch.allclose(got.to(torch.float64), ref, atol=2e-5), (got, ref)
    # and the naive formulation is demonstrably worse, which is why expm1 is mandatory
    naive = -(1.0 / math.log(10.0)) * torch.log1p(-torch.exp(-kt.to(torch.float32)))
    assert (naive.to(torch.float64) - ref).abs().max() > (got.to(torch.float64)
                                                          - ref).abs().max()


def test_saturation_term_survives_fp16_inputs():
    """An fp16 caller must still get a finite, correctly ordered answer (the fp32 block)."""
    t = DURATIONS.to(torch.float16)
    k = torch.full(t.shape, 1e-3, dtype=torch.float16)
    v = L.saturation_term(k, t)
    assert torch.isfinite(v).all()
    assert torch.all(torch.diff(v) <= 1e-5)


def test_saturation_term_is_finite_at_both_extremes():
    k = torch.tensor([1e-9, 1e-9, 1e3, 1e3])
    t = torch.tensor([1e-3, 1e5, 1e-3, 1e5])
    v = L.saturation_term(k, t)
    assert torch.isfinite(v).all(), v
    # the short-duration tail sits at the KT_MIN clamp: -log10(1e-9) = 9
    assert v[0] == pytest.approx(9.0, abs=1e-3)


def test_saturation_term_rejects_nonpositive_inputs():
    with pytest.raises(L.TimeLawError, match="k must be"):
        L.saturation_term(torch.tensor([0.0]), torch.tensor([1.0]))
    with pytest.raises(L.TimeLawError, match="t must be"):
        L.saturation_term(torch.tensor([1.0]), torch.tensor([0.0]))


# -------------------------------------------------------------------------------- the anchor

def test_anchor_least_squares_recovers_a_planted_relationship():
    rng = np.random.default_rng(0)
    kow = rng.normal(3.0, 1.5, 200)
    mw = rng.normal(2.3, 0.3, 200)
    y = -2.0 + 0.4 * kow - 0.7 * mw
    a0, a1, a2 = L.anchor_least_squares(kow, mw, y)
    assert a0 == pytest.approx(-2.0, abs=1e-6)
    assert a1 == pytest.approx(0.4, abs=1e-6)
    assert a2 == pytest.approx(-0.7, abs=1e-6)


def test_anchor_is_degenerate_safe():
    """One distinct chemical must not produce a wild anchor."""
    a0, a1, a2 = L.anchor_least_squares(np.full(5, 3.0), np.full(5, 2.0), np.full(5, -4.0))
    assert (a1, a2) == (0.0, 0.0) and a0 == pytest.approx(-4.0)


def test_anchor_rejects_mismatched_arrays():
    with pytest.raises(L.TimeLawError):
        L.anchor_least_squares(np.zeros(3), np.zeros(2), np.zeros(3))
    with pytest.raises(L.TimeLawError):
        L.anchor_least_squares(np.zeros(0), np.zeros(0), np.zeros(0))


# ---------------------------------------------------------------------------------- the rate

def test_rate_head_with_b_zero_is_the_pure_anchored_law():
    """Ablation A2: B = 0 leaves no learned residual at all."""
    head = L.RateHead(8, b_decades=0.0, anchor=(-1.0, 0.25, -0.5))
    z = torch.randn(4, 8)
    kow = torch.tensor([1.0, 2.0, 3.0, 4.0])
    mw = torch.tensor([2.0, 2.0, 2.0, 2.0])
    got = head(z, kow, mw)
    want = -1.0 + 0.25 * kow - 0.5 * mw
    assert torch.allclose(got, want, atol=1e-6)
    # and it is independent of the latent, which is what "pure anchored" means
    assert torch.allclose(head(torch.randn(4, 8) * 100, kow, mw), want, atol=1e-6)


def test_rate_head_residual_is_bounded_by_b_decades():
    b = 1.5
    head = L.RateHead(8, b_decades=b, anchor=(0.0, 0.0, 0.0))
    with torch.no_grad():
        for p in head.u.parameters():
            p.mul_(50.0)          # push the residual to saturation
    z = torch.randn(256, 8) * 10
    out = head(z, torch.zeros(256), torch.zeros(256))
    assert out.abs().max().item() <= b + 1e-5, out.abs().max()


def test_rate_head_rate_is_strictly_positive():
    head = L.RateHead(8, b_decades=1.0, anchor=(-3.0, 0.0, 0.0))
    k = head.rate(torch.randn(16, 8), torch.zeros(16), torch.zeros(16))
    assert torch.all(k > 0) and torch.isfinite(k).all()


def test_rate_head_set_anchor_and_bad_b():
    head = L.RateHead(4, b_decades=0.0)
    head.set_anchor(1.0, 2.0, 3.0)
    assert head.anchor.tolist() == [1.0, 2.0, 3.0]
    with pytest.raises(L.TimeLawError, match="non-negative"):
        L.RateHead(4, b_decades=-1.0)


# ----------------------------------------------------------------------------- the acute head

def test_acute_head_is_monotone_in_duration():
    torch.manual_seed(1)
    head = L.AcuteHead(8, b_decades=1.0, anchor=(-2.0, 0.1, -0.2))
    z = torch.randn(1, 8).expand(DURATIONS.shape[0], 8)
    out = head(z, DURATIONS, torch.full(DURATIONS.shape, 3.0),
               torch.full(DURATIONS.shape, 2.3))
    assert torch.all(torch.diff(out) <= 1e-5), out


def test_acute_head_extrapolates_as_a_shape_not_a_constant():
    """The one testable advantage over R1: past the last observed duration it keeps moving."""
    torch.manual_seed(2)
    head = L.AcuteHead(8, b_decades=0.0, anchor=(-3.0, 0.0, 0.0))
    z = torch.randn(1, 8).expand(4, 8)
    t = torch.tensor([24.0, 96.0, 720.0, 26298.0])
    out = head(z, t, torch.zeros(4), torch.zeros(4))
    assert out[2] < out[1] and out[3] <= out[2]
    assert not torch.allclose(out[1], out[3], atol=1e-4), "a constant past 96 h would be R1"


# ------------------------------------------------------- the ordering, true by construction

def test_ec10_never_exceeds_ec50_over_a_random_parameter_sweep():
    """EC10 <= EC50 for EVERY parameter value and EVERY t, not merely in expectation."""
    for seed in range(40):
        torch.manual_seed(seed)
        head = L.AcuteChronicHead(6, b_decades=1.5, anchor=(-2.0, 0.3, -0.4))
        with torch.no_grad():
            for p in head.parameters():
                p.mul_(5.0)       # random, large weights
        z = torch.randn(DURATIONS.shape[0], 6) * 3
        out = head(z, DURATIONS, torch.randn(DURATIONS.shape[0]) * 2,
                   torch.randn(DURATIONS.shape[0]) * 2)
        assert torch.all(out["chronic"] <= out["acute"] + 1e-5), (
            seed, out["chronic"] - out["acute"])


def test_chronic_rate_is_at_least_the_acute_rate():
    """The mechanism that makes the ordering analytic."""
    torch.manual_seed(3)
    head = L.AcuteChronicHead(6)
    out = head(torch.randn(32, 6), torch.full((32,), 48.0), torch.randn(32), torch.randn(32))
    assert torch.all(out["k_chronic"] >= out["k"] - 1e-9)


def test_log10_acr_is_non_negative_and_is_a_reported_output():
    torch.manual_seed(4)
    head = L.AcuteChronicHead(6)
    with torch.no_grad():
        for p in head.u_acr.parameters():
            p.mul_(-100.0)        # try to force it negative
    acr = head.log10_acr(torch.randn(64, 6) * 10)
    assert torch.all(acr >= 0.0)
    out = head(torch.randn(8, 6), torch.full((8,), 48.0), torch.zeros(8), torch.zeros(8))
    assert "log10_acr" in out and out["log10_acr"].shape == (8,)


def test_acute_chronic_head_is_monotone_in_duration_on_both_endpoints():
    torch.manual_seed(5)
    head = L.AcuteChronicHead(6, b_decades=1.0)
    z = torch.randn(1, 6).expand(DURATIONS.shape[0], 6)
    out = head(z, DURATIONS, torch.zeros(DURATIONS.shape[0]), torch.zeros(DURATIONS.shape[0]))
    assert torch.all(torch.diff(out["acute"]) <= 1e-5)
    assert torch.all(torch.diff(out["chronic"]) <= 1e-5)


# --------------------------------------------------------------------------- the chronic chain

def test_chronic_chain_orders_noec_matc_loec():
    torch.manual_seed(6)
    base = torch.randn(64)
    deltas = torch.randn(64, 2) * 10
    out = L.chronic_chain(base, deltas)
    assert torch.all(out["noec"] <= out["matc"] + 1e-6)
    assert torch.all(out["matc"] <= out["loec"] + 1e-6)
    assert torch.allclose(out["noec"], base, atol=1e-5)


def test_chronic_chain_rejects_a_bad_delta_shape():
    with pytest.raises(L.TimeLawError, match=r"\(n, 2\)"):
        L.chronic_chain(torch.zeros(4), torch.zeros(4, 3))


# ------------------------------------------------------------------------------- numerics

def test_kt_clamp_is_recorded_and_bounds_the_extrapolation():
    assert L.KT_MIN == 1e-9 and L.KT_MAX == 40.0
    # at the upper clamp the term is already negligible, so clamping changes no prediction
    k = torch.tensor([1.0, 1.0])
    t = torch.tensor([40.0, 1e6])
    v = L.saturation_term(k, t)
    assert v[0] == pytest.approx(v[1].item(), abs=1e-9)
    assert v[1].item() < 1e-9


def test_fp32_block_is_usable_on_cpu():
    with L.fp32_block():
        out = L.saturation_term(torch.tensor([0.01]), torch.tensor([48.0]))
    assert torch.isfinite(out).all()


def test_gradients_flow_through_the_composition_layer():
    torch.manual_seed(7)
    head = L.AcuteChronicHead(6, b_decades=1.0)
    z = torch.randn(16, 6, requires_grad=True)
    out = head(z, torch.full((16,), 48.0), torch.zeros(16), torch.zeros(16))
    (out["acute"].sum() + out["chronic"].sum()).backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()
    assert z.grad.abs().sum() > 0, "a chronic row's gradient must reach the shared latent"
