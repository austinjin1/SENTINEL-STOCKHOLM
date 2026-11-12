"""Tests for ChemTox-TimeHead. Synthetic tensors only."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sentinel.models.chemtox import arch_v2_timehead as H

F = 8
DUR = torch.tensor([24.0, 48.0, 96.0, 168.0, 720.0, 2160.0, 26298.0])


def _th(**kw):
    torch.manual_seed(0)
    return H.TimeHead(H.TimeHeadConfig(n_features=F, hidden=16, depth=1, **kw))


def test_rate_head_is_under_the_fifty_thousand_parameter_budget():
    model = H.TimeHead(H.TimeHeadConfig(n_features=200, hidden=64, depth=2))
    assert model.n_parameters() < model.cfg.n_parameters_budget, model.n_parameters()


def test_prediction_is_monotone_in_duration():
    model = _th()
    n = DUR.shape[0]
    out = model(torch.randn(1, F).expand(n, F), torch.full((n,), -4.0), DUR,
                torch.zeros(n), torch.zeros(n))
    assert torch.all(torch.diff(out["pred"]) <= 1e-5), out["pred"]


def test_prediction_approaches_the_frozen_asymptote():
    model = _th(b_decades=0.0, anchor=(-1.0, 0.0, 0.0))
    y_inf = torch.full((2,), -5.0)
    out = model(torch.zeros(2, F), y_inf, torch.tensor([1e5, 1e6]), torch.zeros(2), torch.zeros(2))
    assert torch.allclose(out["pred"], y_inf, atol=1e-3)


def test_the_tabular_model_must_be_frozen():
    """y_inf comes from a FROZEN model: a gradient-carrying asymptote is refused."""
    model = _th()
    y = torch.full((4,), -4.0, requires_grad=True)
    with pytest.raises(H.TimeHeadError, match="FROZEN"):
        model(torch.randn(4, F), y, torch.full((4,), 48.0), torch.zeros(4), torch.zeros(4))


def test_only_the_rate_is_learned():
    model = _th()
    x = torch.randn(6, F)
    y = torch.full((6,), -4.0)
    t = torch.full((6,), 48.0)
    out = model(x, y, t, torch.zeros(6), torch.zeros(6))
    out["pred"].sum().backward()
    assert all(p.grad is not None for p in model.head.parameters() if p.requires_grad)
    assert torch.equal(out["y_inf"], y), "y_inf must pass through untouched"


def test_rate_is_positive_and_finite():
    model = _th()
    out = model(torch.randn(32, F) * 10, torch.full((32,), -4.0), torch.full((32,), 48.0),
                torch.randn(32), torch.randn(32))
    assert torch.all(out["k"] > 0) and torch.isfinite(out["k"]).all()


def test_shape_validation():
    model = _th()
    with pytest.raises(H.TimeHeadError, match="n_features"):
        model(torch.randn(4), torch.zeros(4), torch.ones(4), torch.zeros(4), torch.zeros(4))
    with pytest.raises(H.TimeHeadError, match="row count"):
        model(torch.randn(4, F), torch.zeros(5), torch.ones(4), torch.zeros(4), torch.zeros(4))
    with pytest.raises(H.TimeHeadError, match="positive"):
        H.TimeHead(H.TimeHeadConfig(n_features=0))


def test_incipient_target_selects_the_long_durations():
    y = [1.0, 2.0, 3.0, 4.0]
    t = [24.0, 48.0, 96.0, 96.0]
    mask = H.incipient_target(y, t, long_duration_h=96.0)
    assert mask.tolist() == [False, False, True, True]


def test_incipient_target_raises_when_no_long_row_exists():
    with pytest.raises(H.TimeHeadError, match="undefined"):
        H.incipient_target([1.0], [24.0], long_duration_h=96.0)
    with pytest.raises(H.TimeHeadError, match="same length"):
        H.incipient_target([1.0, 2.0], [24.0])


def test_a_training_step_recovers_a_planted_rate():
    """Trainability, not performance: the head must be able to learn a known k."""
    torch.manual_seed(1)
    from sentinel.models.chemtox import arch_v2_timelaw as L
    model = _th(b_decades=2.0, anchor=(-2.0, 0.0, 0.0))
    n = 256
    x = torch.randn(n, F)
    t = torch.from_numpy(np.random.default_rng(0).choice([24.0, 48.0, 96.0], n)).float()
    y_inf = torch.full((n,), -5.0)
    k_true = torch.full((n,), 10.0 ** -1.5)
    target = y_inf + L.saturation_term(k_true, t)
    opt = torch.optim.Adam(model.parameters(), lr=5e-2)
    first = None
    for _ in range(80):
        opt.zero_grad()
        pred = model(x, y_inf, t, torch.zeros(n), torch.zeros(n))["pred"]
        loss = torch.mean((pred - target) ** 2)
        loss.backward()
        opt.step()
        first = loss.item() if first is None else first
    assert loss.item() < first * 0.5, (first, loss.item())
