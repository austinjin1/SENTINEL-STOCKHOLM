# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Physical constants used by the M6 architecture-v2 arms (lead note X7, 2026-11-10): the arms call the shared routines
and never hold a private copy of a formula. Each test states the cited source and how the value was checked.

* DO saturation: ``sentinel.models.hydromet.physics.do_sat_mgl`` (Benson & Krause 1984), frozen; table test in
  ``tests/test_hydromet.py::test_do_sat``; re-checked here at the values the arms use.
* Gas exchange (only if gate G-DO opens the D2 DO stage): ``sentinel.models.sim.physics`` — the single implementation in
  the repository (``sonde_v2/physics_features.k600_raymond`` wraps it; no other file holds the formula).
  - hydraulic geometry D = 0.409 Q^0.294 m, V = 0.194 Q^0.285 m/s: checked on 2026-11-10 against the streamMetabolizer
    source (DOI-USGS/streamMetabolizer, R/calc_depth.R ``c=u(0.409,"m"), f=u(0.294,"")`` and R/calc_velocity.R
    ``k=u(0.194,"m s^-1"), m=u(0.285,"")``, both citing Raymond et al. 2012);
  - Raymond et al. 2012 Table 2 eq. 1, k600 = (V S)^0.89 D^0.54 · 5037 m/d: transcribed from memory of the paper; the
    publisher page was not readable from this host on 2026-11-10 (HTTP 403), so the transcription check against the
    paper itself is still OPEN (needed before any DO-stage fit);
  - Schmidt number of O2: the repository uses Wanninkhof (1992) fresh water, 1800.6 − 120.10 t + 3.7818 t² −
    0.047608 t³. The Powell Center K600 estimates were produced by streamMetabolizer, which converts with
    LakeMetabolizer::getSchmidt, O2 = 1568 − 86.04 t + 2.142 t² − 0.0216 t³ (Raymond et al. 2012 Table 1; checked
    against GLEON/LakeMetabolizer R/getSchmidt.R on 2026-11-10). The two differ; the test below pins the size of the
    difference so that a K600 target from Powell rows is never combined with the other polynomial unnoticed.
"""

import numpy as np
import pytest

from sentinel.models.hydromet.physics import do_sat_mgl, pressure_atm_from_elevation
from sentinel.models.sim import physics as SP


def schmidt_o2_raymond2012(t):
    """LakeMetabolizer::getSchmidt, gas O2 (the polynomial behind the Powell Center K600 values). Test-only."""
    t = np.asarray(t, float)
    return 1568.0 - 86.04 * t + 2.142 * t ** 2 - 0.0216 * t ** 3


def test_do_saturation_values_used_by_the_arms():
    t = np.array([0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0])
    np.testing.assert_allclose(do_sat_mgl(t), [14.62, 12.77, 11.29, 10.08, 9.09, 8.26, 7.56], atol=0.01)   # B&K 1984
    assert do_sat_mgl(20.0, pressure_atm_from_elevation(1000.0)) == pytest.approx(9.09 * 0.887, rel=0.01)


def test_hydraulic_geometry_matches_streammetabolizer_defaults():
    assert (SP.DEPTH_C, SP.DEPTH_F, SP.VEL_C, SP.VEL_F) == (0.409, 0.294, 0.194, 0.285)
    q = np.array([0.1, 1.0, 25.0])
    np.testing.assert_allclose(SP.depth_m(q), 0.409 * q ** 0.294, rtol=1e-12)
    np.testing.assert_allclose(SP.velocity_ms(q), 0.194 * q ** 0.285, rtol=1e-12)
    assert SP.CFS_TO_M3S == pytest.approx(0.3048 ** 3, rel=1e-12)                 # 1 ft = 0.3048 m exactly


def test_k600_prior_is_raymond_eq1_divided_by_depth():
    q, s = np.array([0.5, 1.0, 10.0]), np.array([0.002, 0.01, 0.0005])
    d, v = 0.409 * q ** 0.294, 0.194 * q ** 0.285
    k600_m_per_day = (v * s) ** 0.89 * d ** 0.54 * 5037.0                         # Raymond et al. 2012, Table 2, eq. 1
    np.testing.assert_allclose(SP.k600_prior(q, s), k600_m_per_day / d, rtol=1e-12)
    assert SP.k600_prior(1.0, 0.01) == pytest.approx(29.3, abs=0.1)               # hand calculation, 1/d
    assert SP.k600_prior(1e3, 1e-7) >= 0.05 and SP.k600_prior(0.01, 0.5) <= 150.0  # the routine's clip range


def test_schmidt_number_and_its_two_published_polynomials():
    assert SP.schmidt_o2(20.0) == pytest.approx(530.5, abs=0.1)                   # Wanninkhof 1992, fresh water
    t600 = np.linspace(15.0, 20.0, 5001)
    assert t600[np.argmin(np.abs(SP.schmidt_o2(t600) - 600.0))] == pytest.approx(17.6, abs=0.1)
    np.testing.assert_allclose(SP.k600_to_ko2(10.0, 20.0), 10.0 * (530.456 / 600.0) ** -0.5, rtol=1e-4)
    # Powell-pipeline polynomial (Raymond 2012 Table 1) against the repository's (Wanninkhof 1992): K_O2 for one K600
    t = np.array([5.0, 10.0, 20.0, 30.0])
    ratio = (schmidt_o2_raymond2012(t) / SP.schmidt_o2(t)) ** -0.5
    np.testing.assert_allclose(ratio, [1.041, 1.017, 0.999, 0.976], atol=0.001)
    assert np.abs(ratio - 1.0).max() < 0.05          # up to about 4 % in K_O2 at 5 °C; documented, not silently mixed


# ------------------------------------------------------------------ surface energy balance (thermox.physics; D2, G2)
from sentinel.hydromet_sota.thermox import physics as TP  # noqa: E402


def test_fao56_worked_examples():
    # FAO-56 (Allen et al. 1998): Example 3 (e°), Annex 2 Table 2.4 (slope), Example 2 (P, γ at 1800 m), Example 14 (u2)
    assert TP.sat_vp_kpa(np.array([24.5, 15.0])) == pytest.approx([3.075, 1.705], abs=0.001)
    assert TP.sat_vp_slope(np.array([20.0, 25.0])) == pytest.approx([0.1447, 0.1888], abs=0.0005)
    p = TP.pressure_kpa(np.array([1800.0]))[0]
    # the shared routine is the ISA at 15 °C; FAO-56 eq. 7 uses 20 °C and gives 81.8 kPa: 0.4 % apart, one formula kept
    assert p == pytest.approx(81.49, abs=0.01) and p == pytest.approx(81.8, rel=0.005)
    assert TP.psychrometric(81.8) == pytest.approx(0.054, abs=0.0005)
    assert TP.WIND_10M_TO_2M * 3.2 == pytest.approx(2.4, abs=0.01)
    assert TP.vp_from_sph(0.010, 101.325) == pytest.approx(0.010 * 101.325 / 0.62578, rel=1e-9)


def test_radiation_and_wind_terms():
    assert TP.wind_function(0.0) == pytest.approx(74.42, abs=0.01)             # 6.43 MJ m-2 d-1 kPa-1 in W m-2 kPa-1
    assert TP.wind_function(2.0) / TP.wind_function(0.0) == pytest.approx(2.072)
    assert TP.emissivity_clear(1.5, 14.85) == pytest.approx(1.24 * (15.0 / 288.0) ** (1 / 7), rel=1e-9)
    assert 0.80 < TP.emissivity_clear(1.5, 14.85) < 0.83
    assert TP.emissivity_all_sky(1.5, 14.85, 1.0) == pytest.approx(TP.emissivity_clear(1.5, 14.85))   # clear sky
    assert TP.emissivity_all_sky(1.5, 14.85, 0.0) == pytest.approx(1.0)                               # overcast
    assert TP.SIGMA * 288.15 ** 4 == pytest.approx(390.9, abs=0.2)


def test_equilibrium_temperature_is_the_root_of_the_heat_budget_in_numpy_and_torch():
    torch = pytest.importorskip("torch")
    rs = np.random.default_rng(0)
    n = 4000
    tm, sr = rs.uniform(-10, 35, n), rs.uniform(5, 380, n)
    q = 0.622 * rs.uniform(0.2, 0.95, n) * TP.sat_vp_kpa(tm) / 95.0
    kw = TP.surface_drivers(tmean=tm, srad=sr, wind10=rs.uniform(0.3, 9, n), sph=q, p_kpa=rs.uniform(70, 101.3, n),
                            cloud_ratio=rs.uniform(0, 1.2, n), shade=rs.uniform(0, 0.9, n), wind_shelter=rs.uniform(0.2, 1, n))
    te, k = TP.equilibrium_temperature(**kw)
    h, dh = TP.heat_flux(te, **kw)
    inside = (te > TP.TE_RANGE[0]) & (te < TP.TE_RANGE[1])
    assert inside.mean() > 0.8 and np.abs(h[inside]).max() < 0.05              # W m-2 after 4 Newton steps
    assert (k > 5).all() and (k < 200).all() and np.allclose(k, -dh)           # exchange coefficient, W m-2 K-1
    eps = 1e-3
    assert np.allclose(dh, (TP.heat_flux(te + eps, **kw)[0] - TP.heat_flux(te - eps, **kw)[0]) / (2 * eps), rtol=1e-4)
    # more sun or warmer air -> warmer equilibrium; more shade -> cooler
    kw2 = dict(kw, sw_abs=kw["sw_abs"] + 50.0)
    assert (TP.equilibrium_temperature(**kw2)[0] >= te).all()
    tt = {a: torch.tensor(b, dtype=torch.float32, requires_grad=(a == "sw_abs")) for a, b in kw.items()}
    te_t, k_t = TP.equilibrium_temperature(**tt)
    assert np.allclose(te_t.detach().numpy(), te, atol=2e-3) and np.allclose(k_t.detach().numpy(), k, rtol=1e-3)
    te_t.sum().backward()
    g = tt["sw_abs"].grad.numpy()
    assert np.isfinite(g).all() and (g[inside] > 0).all()
    assert np.allclose(g[inside], 1.0 / k[inside], rtol=0.02)                  # dTe/dS = 1/K (implicit function)


def test_prior_equilibrium_temperature_from_the_forcing_cube():
    from sentinel.hydromet_sota.gso_mt.synthetic import synth_v2
    from sentinel.hydromet_sota.gso_mt.solar import SOLAR_CHANNELS
    from sentinel.models.hydromet import data as D

    v2 = synth_v2(3, 1, 2, 1, seed=3)
    v2.data.forc[:, :, D.FORCING_VARS.index("sph_kgkg")] = 0.006
    v2.data.forc[:, :, D.FORCING_VARS.index("wind_ms")] = 3.0
    cr = v2.solar[:, :, SOLAR_CHANNELS.index("cloud_ratio")]
    te = TP.te_prior(v2.data.forc, v2.data.static["elev_m"].to_numpy(), cr)
    assert te.shape == v2.data.forc.shape[:2] and np.isfinite(te).all()
    tm = v2.data.forc[:, :, D.FORCING_VARS.index("tmean_c")]
    assert np.corrcoef(te.ravel(), tm.ravel())[0, 1] > 0.9 and te.min() >= -5 and te.max() <= 45
    f2 = v2.data.forc.copy()
    f2[0, 10, D.FORCING_VARS.index("srad_wm2")] = np.nan
    assert np.isnan(TP.te_prior(f2, v2.data.static["elev_m"].to_numpy(), cr)[0, 10])
    shady = TP.te_prior(v2.data.forc, v2.data.static["elev_m"].to_numpy(), cr, prior={"shade": 0.9})
    assert (shady <= te + 1e-5).all() and (shady < te - 0.1).mean() > 0.5
