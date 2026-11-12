"""Tests for the corrected M4 arch-v2 power helper (lead ruling L5).

The required pair of limit tests is ``test_closed_form_is_row_level_at_icc0`` and
``test_closed_form_is_unit_level_at_icc1``; the Monte-Carlo tests check the formula itself rather
than restating it, so a sign or exponent error in the derivation cannot pass.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from sentinel.models.chemtox import arch_v2_power as P


def _mc_sd(n_units: int, unit_size: int, *, icc: float, rho: float, s: float = 1.2,
           reps: int = 4000, seed: int = 0) -> float:
    """Monte-Carlo SD of the paired RMSE difference: independent realisations, no bootstrap."""
    rng = np.random.default_rng(seed)
    wu, wr = math.sqrt(icc), math.sqrt(max(0.0, 1.0 - icc))
    c = math.sqrt(1.0 - rho * rho)
    out = np.empty(reps)
    for k in range(reps):
        g = rng.normal(size=n_units)
        h = rng.normal(size=n_units)
        z = rng.normal(size=(n_units, unit_size))
        w = rng.normal(size=(n_units, unit_size))
        a = wu * g[:, None] + wr * z
        b = wu * (rho * g + c * h)[:, None] + wr * (rho * z + c * w)
        out[k] = s * (np.sqrt(np.mean(a * a)) - np.sqrt(np.mean(b * b)))
    return float(out.std(ddof=1))


def test_z_two_sided():
    assert P.z_two_sided(0.05) == pytest.approx(1.959963985, abs=1e-6)
    assert P.z_two_sided(0.01) == pytest.approx(2.575829304, abs=1e-6)


def test_closed_form_is_row_level_at_icc0():
    """icc = 0 must reproduce the ROW-level value, i.e. what unseen_taxa.mde estimates."""
    sizes = [7] * 300                     # N = 2100 rows, U = 300 units
    n = sum(sizes)
    want = P.z_two_sided(0.05) * 1.2 * math.sqrt(1 - 0.9 ** 2) / math.sqrt(n)
    got = P.mde_closed_form(sizes, rmse_level=1.2, rho=0.9, icc=0.0)
    assert got == pytest.approx(want, rel=1e-12)
    assert P.design_effect(sizes, 0.0) == pytest.approx(1.0)
    assert P.effective_units(sizes, 0.0) == pytest.approx(n)


def test_closed_form_is_unit_level_at_icc1():
    """icc = 1 must reproduce the UNIT-level value z*s*sqrt(1-rho^2)/sqrt(U)."""
    sizes = [7] * 300
    u = len(sizes)
    want = P.z_two_sided(0.05) * 1.2 * math.sqrt(1 - 0.9 ** 2) / math.sqrt(u)
    got = P.mde_closed_form(sizes, rmse_level=1.2, rho=0.9, icc=1.0)
    assert got == pytest.approx(want, rel=1e-12)
    assert P.effective_units(sizes, 1.0) == pytest.approx(u)


def test_unit_level_reproduces_the_verifier_figures():
    """The verifier's gate values for the v1 blocks, from counts alone."""
    for n_rows, n_units, want in ((10548, 468, 0.047), (13059, 1463, 0.027), (2410, 372, 0.053)):
        sizes = [1] * n_units
        sizes[0] += n_rows - n_units          # same U, same N, unit-level limit ignores shape
        got = P.z_two_sided(0.05) * 1.2 * math.sqrt(1 - 0.81) / math.sqrt(n_units)
        assert got == pytest.approx(want, abs=0.0006)


@pytest.mark.parametrize("icc", [0.0, 0.5, 1.0])
def test_closed_form_matches_monte_carlo(icc):
    """The derivation, checked against simulation at every tabulated icc."""
    n_units, unit_size, rho = 120, 6, 0.9
    sd_mc = _mc_sd(n_units, unit_size, icc=icc, rho=rho, reps=3000, seed=7)
    sd_formula = P.mde_closed_form([unit_size] * n_units, rmse_level=1.2, rho=rho,
                                   icc=icc) / P.z_two_sided(0.05)
    assert sd_formula == pytest.approx(sd_mc, rel=0.08), (icc, sd_formula, sd_mc)


def test_design_effect_uses_icc_squared():
    """Squared errors correlate at icc^2, so the design effect is 1 + icc^2 (nbar - 1)."""
    sizes = [11] * 50
    assert P.design_effect(sizes, 0.5) == pytest.approx(1 + 0.25 * 10)
    assert P.design_effect(sizes, 1.0) == pytest.approx(11.0)


def test_bootstrap_agrees_with_closed_form_on_equal_units():
    sizes = [6] * 250
    for icc in (0.0, 0.5, 1.0):
        r = P.mde_bootstrap(sizes, icc=icc, n_boot=8000, seed=3)
        assert r["mde_bootstrap"] == pytest.approx(r["mde_closed_form"], rel=0.12), (icc, r)


def test_mde_is_monotone_increasing_in_icc():
    sizes = [9] * 180
    vals = [P.mde_bootstrap(sizes, icc=i, n_boot=4000, seed=11)["mde_bootstrap"]
            for i in (0.0, 0.5, 1.0)]
    assert vals[0] < vals[1] < vals[2], vals


def test_rho_zero_is_more_pessimistic_than_rho_high():
    sizes = [5] * 200
    hi = P.mde_closed_form(sizes, rho=0.9, icc=1.0)
    lo = P.mde_closed_form(sizes, rho=0.0, icc=1.0)
    assert lo > hi


def test_unequal_units_are_handled_and_worse_than_equal():
    """A block whose rows concentrate in a few units resolves less than an even one."""
    even = [10] * 100
    skew = [1] * 99 + [901]
    assert sum(even) == sum(skew)
    r_even = P.mde_bootstrap(even, icc=1.0, n_boot=6000, seed=5)["mde_bootstrap"]
    r_skew = P.mde_bootstrap(skew, icc=1.0, n_boot=6000, seed=5)["mde_bootstrap"]
    assert r_skew > r_even


def test_signature_is_counts_only():
    """Structural guarantee for lead ruling L6: no frame, target or prediction can be passed."""
    import inspect
    for fn in (P.mde_closed_form, P.mde_bootstrap, P.mde_table, P.design_effect,
               P.effective_units):
        names = set(inspect.signature(fn).parameters)
        assert not names & {"block", "df", "frame", "y", "target", "pred", "predictions"}, fn
        assert "unit_sizes" in names, fn


def test_label_free_flag_and_shape_invariance():
    """Identical unit-size vectors give identical answers whatever the labels would have been."""
    sizes = [3, 4, 5, 6, 7] * 20
    a = P.mde_bootstrap(sizes, icc=0.5, n_boot=2000, seed=0)
    b = P.mde_bootstrap(list(sizes), icc=0.5, n_boot=2000, seed=0)
    assert a == b and a["label_free"] is True


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        P.mde_closed_form([])
    with pytest.raises(ValueError):
        P.mde_closed_form([1, 0, 2])
    with pytest.raises(ValueError):
        P.mde_bootstrap([3, 3], icc=1.5)
    with pytest.raises(ValueError):
        P.mde_bootstrap([3, 3], rho=1.0)
    with pytest.raises(ValueError):
        P.mde_bootstrap([3, 3], n_boot=10)


def test_closed_form_is_exact_for_unequal_units():
    """The exact sum form, hand-computed, and strictly worse than the mean-size approximation."""
    sizes = [1] * 99 + [901]
    n = sum(sizes)
    var_sum = sum(x * (1 + 1.0 ** 2 * (x - 1)) for x in sizes)   # icc = 1 -> sum n_u^2
    want = P.z_two_sided(0.05) * 1.2 * math.sqrt(1 - 0.81) * math.sqrt(var_sum) / n
    got = P.mde_closed_form(sizes, rmse_level=1.2, rho=0.9, icc=1.0)
    assert got == pytest.approx(want, rel=1e-12)
    # the mean-size approximation would have been far more optimistic
    approx = (P.z_two_sided(0.05) * 1.2 * math.sqrt(1 - 0.81)
              * math.sqrt(P.design_effect(sizes, 1.0) / n))
    assert got > 2 * approx


def test_effective_sample_size_limits():
    sizes = [8] * 60
    assert P.effective_sample_size(sizes, 0.0) == pytest.approx(sum(sizes))
    assert P.effective_sample_size(sizes, 1.0) == pytest.approx(len(sizes))


def test_bootstrap_averaging_stabilises_high_icc():
    """n_sim averaging must cut the spread of the icc = 1 estimate on skewed units."""
    sizes = [1] * 200 + [50] * 10 + [500]
    one = [P.mde_bootstrap(sizes, icc=1.0, n_boot=1500, seed=s, n_sim=1)["mde_bootstrap"]
           for s in range(6)]
    many = [P.mde_bootstrap(sizes, icc=1.0, n_boot=1500, seed=s, n_sim=8)["mde_bootstrap"]
            for s in range(6)]
    assert np.std(many) < np.std(one)
