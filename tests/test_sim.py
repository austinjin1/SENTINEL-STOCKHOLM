# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Unit tests for SENTINEL-Sim physics, integrator, scenario rules, hindcast scoring and guard.

Synthetic arrays are used only inside these tests (RESEARCH_PLAN R9)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from sentinel.models.sim import guard, physics
from sentinel.models.sim import data as D
from sentinel.models.sim.hindcast import SimEC, row_crps
from sentinel.models.sim.model import N_RAW, SimModel
from sentinel.models.sim.scenario import LIMITS, Scenario, ScenarioRangeError
from sentinel.eval.metrics import crps_from_quantiles
from sentinel.models.baselines.sonde_baselines import QUANTILES


# ----------------------------------------------------------------------------- physics checks (protocol §5.1)

def test_do_sat_table_benson_krause():
    # docs/research/06 §a.6 table (APHA Standard Methods form, 1 atm)
    t = np.array([0, 5, 10, 15, 20, 25, 30.0])
    ref = np.array([14.62, 12.77, 11.29, 10.08, 9.09, 8.26, 7.56])
    assert np.allclose(physics.do_sat_mgl(t), ref, atol=0.01)


def test_plus2C_lowers_do_sat_035_at_20C():
    d20 = physics.do_sat_mgl(22.0) - physics.do_sat_mgl(20.0)
    d10 = physics.do_sat_mgl(12.0) - physics.do_sat_mgl(10.0)
    assert d20 == pytest.approx(-0.35, abs=0.01)
    assert d10 == pytest.approx(-0.51, abs=0.01)


def test_pressure_correction_elevation():
    p = physics.pressure_atm_from_elevation(1500.0)
    assert p == pytest.approx(0.835, abs=0.005)
    ratio = physics.do_sat_mgl(20.0, p) / physics.do_sat_mgl(20.0)
    assert 0.82 < ratio < 0.84    # ≈ 16–17% lower DO_sat at 1,500 m


def test_schmidt_scaling():
    assert physics.schmidt_o2(20.0) == pytest.approx(530.5, abs=0.1)
    assert physics.k600_to_ko2(1.0, 20.0) == pytest.approx(1.064, abs=0.002)
    assert physics.k600_to_ko2(1.0, 22.0) / physics.k600_to_ko2(1.0, 20.0) == pytest.approx(1.049, abs=0.003)


def test_arrhenius_ratio():
    assert physics.arrhenius_ratio(20.0, 2.0) == pytest.approx(1.190, abs=0.002)


def test_torch_numpy_do_sat_agree():
    t = np.linspace(0, 35, 50)
    a = physics.do_sat_mgl(t, 0.9)
    b = physics.do_sat_mgl(torch.tensor(t), torch.tensor(0.9)).numpy()
    assert np.allclose(a, b, atol=1e-9)


def test_light_weights():
    w = physics.hourly_light_weights(np.array([40.0, 25.0]), np.arange(1, 366))
    assert w.shape == (2, 365, 24)
    assert np.allclose(w.sum(-1), 1.0, atol=1e-5)
    assert (w[:, :, :3] < 1e-6).all() and (w[:, :, 21:] < 1e-6).all()   # night
    assert np.all(w[:, :, 11] >= w[:, :, 8])                            # noon ≥ morning
    dl = physics.day_length_h(np.array([40.0]), np.array([172, 355]))
    assert dl[0, 0] == pytest.approx(15.0, abs=0.3) and dl[0, 1] == pytest.approx(9.3, abs=0.3)


def test_k600_prior_and_depth():
    assert physics.depth_m(1.0) == pytest.approx(0.409, abs=1e-6)
    assert physics.k600_prior(10.0, 1e-3) == pytest.approx(4.96, abs=0.05)
    # steeper channel → faster gas exchange
    assert physics.k600_prior(10.0, 1e-2) > physics.k600_prior(10.0, 1e-3)


# ----------------------------------------------------------------------------- integrator

def _rand_case(B=3, Dd=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    gpp = torch.rand(B, Dd, generator=g, dtype=torch.float64) * 8
    er = -torch.rand(B, Dd, generator=g, dtype=torch.float64) * 8
    k = 0.5 + torch.rand(B, Dd, generator=g, dtype=torch.float64) * 30
    tm = 5 + 20 * torch.rand(B, Dd, generator=g, dtype=torch.float64)
    th = physics.diel_temperature(tm, torch.full_like(tm, 1.5))
    lw = torch.as_tensor(physics.hourly_light_weights(np.full(B, 40.0), np.arange(150, 150 + Dd)), dtype=torch.float64)
    z = 0.3 + torch.rand(B, Dd, generator=g, dtype=torch.float64)
    return gpp, er, k, th, lw, z


def test_exact_integrator_matches_fine_euler():
    gpp, er, k, th, lw, z = _rand_case()
    a, _ = physics.integrate_do(gpp, er, k, th, lw, z, 1.0)
    b, _ = physics.integrate_do_euler(gpp, er, k, th, lw, z, 1.0, substeps=400)
    assert torch.max(torch.abs(a - b)) < 2e-3


def test_affine_scan_equals_loop():
    g = torch.Generator().manual_seed(1)
    a = torch.rand(2, 1000, generator=g, dtype=torch.float64)
    b = torch.randn(2, 1000, generator=g, dtype=torch.float64)
    c0 = torch.randn(2, dtype=torch.float64)
    out = physics.affine_scan(a, b, c0)
    c = c0.clone()
    ref = []
    for n in range(1000):
        c = a[:, n] * c + b[:, n]
        ref.append(c)
    assert torch.allclose(out, torch.stack(ref, 1), atol=1e-10)


def test_equilibrium_and_respiration_deficit():
    B, Dd = 1, 10
    tm = torch.full((B, Dd), 20.0, dtype=torch.float64)
    th = physics.diel_temperature(tm, torch.zeros_like(tm))
    lw = torch.full((B, Dd, 24), 1 / 24, dtype=torch.float64)
    z = torch.ones(B, Dd, dtype=torch.float64)
    k = torch.full((B, Dd), 10.0, dtype=torch.float64)
    zero = torch.zeros(B, Dd, dtype=torch.float64)
    do, sat = physics.integrate_do(zero, zero, k, th, lw, z, 1.0, c0=torch.tensor([5.0], dtype=torch.float64))
    assert abs(float(do[0, -1, -1] - sat[0, -1, -1])) < 1e-6
    er = torch.full((B, Dd), -5.0, dtype=torch.float64)
    do, sat = physics.integrate_do(zero, er, k, th, lw, z, 1.0)
    kO2 = physics.k600_to_ko2(10.0, 20.0)
    assert float(sat[0, -1, -1] - do[0, -1, -1]) == pytest.approx(5.0 / kO2, rel=1e-4)


def test_integrator_stable_for_huge_k600():
    gpp, er, k, th, lw, z = _rand_case()
    do, sat = physics.integrate_do(gpp, er, k * 50, th, lw, z, 1.0)
    assert torch.isfinite(do).all() and float(torch.max(torch.abs(do - sat))) < 1.0


# ----------------------------------------------------------------------------- model

def _norm(nv=len(D.TARGETS)):
    return {"y_mean": np.zeros(nv, "float32") + 10, "y_std": np.ones(nv, "float32") * 5}


def test_model_shapes_and_physics_do():
    m = SimModel(_norm(), n_dyn=len(D.DYN_NAMES), n_static=len(D.STATIC_NAMES), hidden=16)
    assert m.net.head.out_features == N_RAW == 7       # network never emits DO
    B, T = 2, 20
    lw = torch.as_tensor(physics.hourly_light_weights(np.full(B, 40.0), np.arange(100, 100 + T)))
    out = m(torch.randn(B, T, len(D.DYN_NAMES)), torch.randn(B, len(D.STATIC_NAMES)),
            q_m3s=torch.full((B, T), 5.0), slope=torch.full((B,), 1e-3), light_w=lw,
            pressure=torch.ones(B))
    for k in ("temp_c_mean", "do_mgl_mean", "do_mgl_min", "do_mgl_max", "gpp", "er", "k600"):
        assert out[k].shape == (B, T)
    assert (out["gpp"] >= 0).all() and (out["er"] <= 0).all() and (out["k600"] > 0).all()
    assert (out["do_mgl_min"] <= out["do_mgl_mean"] + 1e-5).all() and (out["do_mgl_mean"] <= out["do_mgl_max"] + 1e-5).all()
    loss = out["do_mgl_mean"].mean()
    loss.backward()
    assert m.net.head.weight.grad is not None and torch.isfinite(m.net.head.weight.grad).all()


def test_inputs_contain_no_wq_observations():
    import re

    pat = re.compile(r"(temp_c|do_mgl|spcond|turb_|turbid|no3|chl_|^ph_|_obs)", re.I)
    assert not [n for n in D.DYN_NAMES + D.STATIC_NAMES if pat.search(n)]


def test_target_transforms_roundtrip():
    x = np.array([0.0, 1.0, 50.0, 3000.0])
    for v in D.TARGETS:
        assert np.allclose(D.from_model(v, D.to_model(v, x)), x, atol=1e-8)


# ----------------------------------------------------------------------------- scenario rules

def test_scenario_ranges_and_forbidden_levers():
    for k, (lo, hi) in LIMITS.items():
        with pytest.raises(ScenarioRangeError):
            Scenario(**{k: hi + 0.01})
        with pytest.raises(ScenarioRangeError):
            Scenario(**{k: lo - 0.01})
    with pytest.raises(ScenarioRangeError):
        Scenario.from_dict({"nitrate_scale": 0.7})
    with pytest.raises(ScenarioRangeError):
        Scenario.from_dict({"latent_shift": 1.0})
    assert Scenario(precip_scale=1.2).kind() == "model_association"
    assert Scenario(q_scale=0.8, precip_scale=0.8).kind() == "physical_forcing"
    assert Scenario().is_baseline


def test_perturb_holds_rh_and_flags():
    from sentinel.models.sim import scenario as SC

    S, T = 2, 400
    dates = pd.date_range("2007-10-01", periods=T)
    f = np.zeros((S, T, len(D.FORCING_RAW)), "float32")
    f[..., D.FORCING_RAW.index("tmax_c")] = 25
    f[..., D.FORCING_RAW.index("tmin_c")] = 15
    f[..., D.FORCING_RAW.index("vpd_kpa")] = 1.0
    f[..., D.FORCING_RAW.index("srad_wm2")] = 200
    q = np.full((S, T), 3.0, "float32")
    d = D.SimData(sites=["a", "b"], role=np.array(["val", "val"]), basin_group=np.array(["g", "g"]), dates=dates,
                  forcing=f, q_m3s=q, q_missing=np.zeros((S, T), bool), y=np.full((S, T, 8), np.nan),
                  prov=np.zeros((S, T, 8), bool), static_raw=np.zeros((S, len(D.STATIC_NAMES))),
                  lat=np.array([40.0, 41.0]), lon=np.zeros(2), elev=np.zeros(2), slope=np.full(2, 1e-3),
                  pressure=np.ones(2), authorised={"mode": "validation"})
    f2, q2, flags = SC.perturb(d, [0, 1], Scenario(dT_air=2.0, q_scale=0.5, sw_scale=0.5))
    assert np.allclose(f2[..., D.FORCING_RAW.index("tmax_c")], 27)
    es = SC._es_kpa
    assert np.allclose(f2[..., D.FORCING_RAW.index("vpd_kpa")], es(22.0) / es(20.0), rtol=1e-5)
    assert np.allclose(f2[..., D.FORCING_RAW.index("srad_wm2")], 100)
    assert np.allclose(q2, 1.5)
    assert flags["q_extrap"].all() and flags["tair_extrap"].all()


# ----------------------------------------------------------------------------- hindcast scoring

def test_row_crps_matches_metric():
    rs = np.random.default_rng(0)
    y = rs.normal(size=200)
    q = np.sort(rs.normal(size=(200, len(QUANTILES))), axis=1)
    assert row_crps(y, q).mean() == pytest.approx(crps_from_quantiles(y, q, QUANTILES), rel=1e-12)


def test_simec_recovers_ar1_phi():
    rs = np.random.default_rng(0)
    S, T, phi = 3, 3000, 0.8
    r = np.zeros((S, T))
    for t in range(1, T):
        r[:, t] = phi * r[:, t - 1] + rs.normal(size=S)
    sim = rs.normal(size=(S, T))
    ec = SimEC(sim + r, sim, np.arange(T) < 2500, np.array([0, 1]), horizons=(1, 3))
    assert ec.phi_site[1] == pytest.approx(np.full(S, phi), abs=0.05)
    assert ec.phi_pool[3] == pytest.approx(phi ** 3, abs=0.06)


# ----------------------------------------------------------------------------- guard

def test_guard_validation_mode_never_reaches_test_block():
    a = guard.authorise("validation")
    assert a["data_end"] == D.TIME_BLOCKS["val"][1] and "test" not in a["roles"]
    with pytest.raises(guard.ProtectedDataError):
        guard.authorise("test")
    if not guard.FREEZE_FILE.exists():
        with pytest.raises(guard.ProtectedDataError):
            guard.authorise("test", allow_test=True)


def test_naivefit_equals_m1_baselines_on_train_mask():
    from sentinel.models.baselines.sonde_baselines import SondeBaselines
    from sentinel.models.sim.dsst import NaiveFit
    from sentinel.models.sonde.data import CLIM_END, Cube, climatology

    rs = np.random.default_rng(0)
    dates = pd.date_range("2016-10-01", "2022-12-31", freq="D")
    S, T = 2, len(dates)
    doy = dates.dayofyear.to_numpy()
    y = (10 + 8 * np.sin(2 * np.pi * doy / 365.25))[None, :] + rs.normal(size=(S, T))
    y[:, rs.random(T) < 0.1] = np.nan
    cube = Cube(["a", "b"], dates, ["temp_c_mean"], y[..., None].astype("float32"), np.zeros((S, T, 1), bool))
    cm, cs = climatology(cube)
    sb = SondeBaselines(cube, cm, cs, "temp_c_mean", horizons=(3,))
    fit = np.asarray(dates <= pd.Timestamp(CLIM_END))
    nf = NaiveFit(cube.y[:, :, 0].astype("float64"), dates, fit, (3,))
    t_idx = np.arange(T - 400, T - 10)
    for m in ("climatology", "persistence"):
        a, b = sb.forecast(m, t_idx, 3), nf.forecast(m, t_idx, 3)
        assert np.allclose(a, b, atol=1e-4, equal_nan=True)


def test_q_fill_only_short_gaps_interpolated():
    dates = pd.date_range("2007-10-01", periods=800)
    q = np.full((1, 800), 100.0)        # cfs
    q[0, 400:420] = np.nan               # 20-day gap -> climatology fill (flagged), not interpolation
    q[0, 500:505] = np.nan               # 5-day gap -> interpolated
    q[0, 400:420] = np.nan
    q2 = q.copy()
    q2[0, 0:400] = np.linspace(50, 150, 400)
    filled, miss = D._fill_q(q2, dates)
    assert miss[0, 400:420].all() and miss[0, 500:505].all()
    lin = np.exp(np.interp(np.arange(500, 505), [499, 505], np.log(q2[0, [499, 505]] * 0.028316846592 + 1e-3))) - 1e-3
    assert np.allclose(filled[0, 500:505], lin, rtol=1e-4)
    # the long gap is NOT interpolated: its first days differ from the straight line between its ends
    ends = np.log(q2[0, [399, 420]] * 0.028316846592 + 1e-3)
    line = np.exp(np.interp(np.arange(400, 407), [399, 420], ends)) - 1e-3
    assert not np.allclose(filled[0, 400:407], line, rtol=1e-3)
    assert np.isfinite(filled).all()


def test_metab_struct_scales_gpp_with_light_and_er_with_temperature():
    import torch

    from sentinel.models.sim import data as D
    from sentinel.models.sim.model import IK_WM2, THETA_ER, SimModel

    n_dyn, n_st = len(D.DYN_NAMES), len(D.STATIC_NAMES)
    norm = {"y_mean": np.zeros(len(D.TARGETS), "float32"), "y_std": np.ones(len(D.TARGETS), "float32"),
            "dyn_mean": np.zeros(n_dyn, "float32"), "dyn_std": np.ones(n_dyn, "float32")}
    torch.manual_seed(0)
    m0 = SimModel(norm, n_dyn, n_st)
    m1 = SimModel(norm, n_dyn, n_st, metab_struct=True, drop_dyn=["doy_sin", "doy_cos"])
    m1.net.load_state_dict(m0.net.state_dict())
    m0.eval(), m1.eval()
    B, T = 2, 5
    dyn = torch.zeros(B, T, n_dyn)
    dyn[..., D.DYN_NAMES.index("srad_wm2")] = 300.0
    st = torch.zeros(B, n_st)
    from sentinel.models.sim import physics

    lw = torch.as_tensor(physics.hourly_light_weights(np.full(B, 40.0), np.arange(100, 100 + T)), dtype=torch.float32)
    kw = dict(q_m3s=torch.full((B, T), 5.0), slope=torch.full((B,), 1e-3), pressure=torch.ones(B), light_w=lw)
    with torch.no_grad():
        o0, o1 = m0(dyn, st, **kw), m1(dyn, st, **kw)
    # drop_dyn zeroes inputs that are already 0 here → same raw net outputs; GPP scaled by tanh(300/150)
    assert torch.allclose(o1["gpp"], o0["gpp"] * np.tanh(300.0 / IK_WM2), rtol=1e-5)
    fac = THETA_ER ** (o1["temp_c_mean"].clamp(min=0) - 20.0)
    assert torch.allclose(o1["er"], o0["er"] * fac, rtol=1e-5)
