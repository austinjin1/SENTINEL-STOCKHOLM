# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""M6 architecture-v2, D2 ThermOx-T temperature stage (``sentinel/hydromet_sota/thermox/nets.py``): code-path and
property tests on synthetic arrays. No real observation is read and no number printed here is a research result."""

import json
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sentinel.hydromet_sota.gso_mt import data as VD  # noqa: E402
from sentinel.hydromet_sota.gso_mt import nets as N  # noqa: E402
from sentinel.hydromet_sota.gso_mt import runner as VR  # noqa: E402
from sentinel.hydromet_sota.gso_mt.synthetic import synth_v2  # noqa: E402
from sentinel.hydromet_sota.gso_mt.trainer import TrainCfg, Trainer  # noqa: E402
from sentinel.hydromet_sota.thermox import nets as TN  # noqa: E402
from sentinel.hydromet_sota.thermox import physics as TP  # noqa: E402

torch.set_num_threads(2)
NET = {"static_hidden": [16, 8], "static_emb": 6, "static_noise": 0.5, "gru_hidden": 10, "n_newton": 4,
       "te_range": [-5, 45], "diel": {"enabled": True, "cycles": 2, "steps": 24}, "mixing": "full",
       "warmup_epochs": 2, "powell_prior": False, "prior": {"shade": 0.3, "wind_shelter": 0.6}, "residual_cap": 1.0}
CFG = dict(arm="t", seq_len=60, predict_last=20, batch_size=16, steps_per_epoch=3, max_epochs=3, patience=10, seed=3,
           grab_share=0.5)
QUIET = lambda *_: None  # noqa: E731

# a hand-made context: 14 dynamic channels in physical units with mean 0 and standard deviation 1 (so x is physical)
DYN = ("tmean_c", "tmax_c", "tmin_c", "srad_wm2", "wind_ms", "sph_kgkg", "ws_tmean_c", "ws_srad_wm2", "ws_wind_ms",
       "ws_sph_kgkg", "cloud_ratio", "photoperiod_h")
CTX = {"dyn": {k: [j, 0.0, 1.0] for j, k in enumerate(DYN)}, "cal": [12, 13], "has_upstream": True,
       "static": {"elev_m": [0, 500.0, 300.0, False], "ws_area_km2": [1, 2.0, 1.0, True]},
       "label": {"t_mean": 12.0, "t_std": 8.0, "logq_std": 1.5}}


def _inputs(B=3, T=400, seed=0, start_doy=1):
    g = torch.Generator().manual_seed(seed)
    doy = (torch.arange(T) + start_doy - 1) % 365 + 1.0
    ang = 2.0 * math.pi * (doy - 1.0) / 365.25
    season = -torch.cos(2.0 * math.pi * (doy - 20.0) / 365.0)
    tm = 10.0 + 10.0 * season[None] + 2.0 * torch.randn(B, T, generator=g)
    x = torch.zeros(B, T, 14)
    x[..., 0], x[..., 1], x[..., 2] = tm, tm + 5.0, tm - 5.0
    x[..., 3] = 180.0 + 100.0 * season[None] + 10.0 * torch.rand(B, T, generator=g)
    x[..., 4] = 2.0 + torch.rand(B, T, generator=g)
    x[..., 5] = 0.004 + 0.003 * (season[None] + 1.0)
    x[..., 6:10] = x[..., [0, 3, 4, 5]] + 0.1
    x[..., 10] = 0.4 + 0.4 * torch.rand(B, T, generator=g)
    x[..., 11] = 12.0 + 3.0 * season[None]
    x[..., 12], x[..., 13] = torch.sin(ang)[None], torch.cos(ang)[None]
    s = torch.randn(B, 5, generator=g)
    return x, s, doy


def _net(**over):
    torch.manual_seed(0)
    return TN.make_net(14, 5, {**NET, **over}, CTX)


def _batch(B=3, L=5, seed=1):
    g = torch.Generator().manual_seed(seed)
    r = lambda *sh: torch.randn(*sh, generator=g)  # noqa: E731
    return {"yd": r(B, L, 11), "md": torch.ones(B, L, 11), "yh": r(B, L, 24, 2), "mh": torch.ones(B, L, 24, 2),
            "hfrac": torch.zeros(B), "yi": r(B, L, 2), "mi": torch.ones(B, L, 2),
            "hi": torch.full((B, L, 2), 13.0), "pc": torch.tensor([0, 1, -1])[:B]}


# ------------------------------------------------------------------------------------------------ bounded parameters
def test_parameters_start_at_their_prior_medians_and_stay_inside_their_bounds():
    th = TN.bounded(torch.tensor(TN.prior_bias()))
    for name, lo, hi, med, _ in TN.PARAMS:
        assert th[name].item() == pytest.approx(med, rel=1e-5), name
        assert lo < med < hi
    assert th["shade_max"].item() == pytest.approx(TP.PRIOR["shade"], rel=1e-5)
    assert th["wind_shelter"].item() == pytest.approx(TP.PRIOR["wind_shelter"], rel=1e-5)
    for raw in (-50.0, 50.0):
        ex = TN.bounded(torch.full((len(TN.PARAMS),), raw))
        for name, lo, hi, _, _ in TN.PARAMS:
            assert lo - 1e-5 * (hi - lo) <= ex[name].item() <= hi + 1e-5 * (hi - lo), name
    net = _net()
    _, s, _ = _inputs()
    net.eval()
    got = TN.bounded(net.par(net.enc(s)))
    for name, lo, hi, med, _ in TN.PARAMS:                  # a fresh network is close to the prior at every site
        assert torch.all((got[name] - med).abs() < 0.15 * (hi - lo)), name


# ------------------------------------------------------------------------------------------------ numerical helpers
def test_exponential_integrator_matches_the_closed_form():
    k = torch.full((1, 6), 0.7)
    const = torch.full((1, 6), 4.0)
    assert torch.allclose(TN.integrate_daily(const, k), const)
    ts = torch.tensor([[0.0, 10.0, 10.0, 10.0, 10.0, 10.0]])            # step at day 1, start at T* = 0
    got = TN.integrate_daily(ts, k)
    e = math.exp(-0.7)
    tw, want = 0.0, [0.0]
    for _ in range(5):
        want.append(10.0 + (tw - 10.0) * (1.0 - e) / 0.7)               # mean of the exact solution over the day
        tw = 10.0 + (tw - 10.0) * e
    assert torch.allclose(got, torch.tensor([want]), atol=1e-5)
    fast = TN.integrate_daily(ts, torch.full((1, 6), 200.0))
    assert torch.allclose(fast, ts, atol=0.06)                           # strong exchange: the water follows T*
    slow = TN.integrate_daily(ts, torch.full((1, 6), 1e-4))
    assert slow.abs().max() < 0.01                                       # no exchange: the water stays where it was


def test_closed_forms_equal_the_step_by_step_recurrences():
    g = torch.Generator().manual_seed(5)
    x = 10.0 + 8.0 * torch.randn(4, 365, generator=g)
    tau = torch.tensor([5.0, 30.0, 90.0, 365.0], requires_grad=True)
    a, b = TN.ema(x, tau), TN.ema_loop(x, tau)
    assert torch.allclose(a, b, atol=2e-4)
    ga, gb = torch.autograd.grad(a.sum(), tau)[0], torch.autograd.grad(b.sum(), tau)[0]
    assert torch.allclose(ga, gb, rtol=2e-3, atol=1e-4)
    k = torch.exp(torch.linspace(math.log(0.02), math.log(150.0), 4))[:, None] \
        * (0.5 + torch.rand(4, 365, generator=g))
    k.requires_grad_(True)
    a, b = TN.integrate_daily(x, k), TN.integrate_daily_loop(x, k)
    assert torch.allclose(a, b, atol=5e-4)
    ga, gb = torch.autograd.grad(a.sum(), k)[0], torch.autograd.grad(b.sum(), k)[0]
    assert torch.allclose(ga, gb, rtol=5e-3, atol=5e-3)


def test_kernel_lag_and_shortwave_shape():
    x = torch.tensor([[1.0, 1.0, 1.0, 1.0], [0.0, 4.0, 0.0, 0.0]])
    y = TN.ema(x, torch.tensor([5.0, 2.0]))
    assert torch.allclose(y[0], x[0])
    a = 1.0 - math.exp(-0.5)
    y0 = 1.0 + a * (0.0 - 1.0)                                           # started at the window mean (1.0)
    y1 = y0 + a * (4.0 - y0)
    assert y[1, 0].item() == pytest.approx(y0, rel=1e-5) and y[1, 1].item() == pytest.approx(y1, rel=1e-5)
    z = torch.arange(6.0)[None].repeat(3, 1)
    lg = TN.lagged(z, torch.tensor([0.0, 2.0, 1.5]))
    assert torch.allclose(lg[0], z[0])
    assert torch.allclose(lg[1], torch.tensor([0.0, 0.0, 0.0, 1.0, 2.0, 3.0]))
    assert torch.allclose(lg[2], torch.tensor([0.0, 0.0, 0.5, 1.5, 2.5, 3.5]))
    sh = TN.sw_shape(torch.tensor([12.0, 16.0, 0.0, 24.0]), 24)
    assert torch.allclose(sh.mean(dim=-1), torch.ones(4), atol=1e-5)
    assert sh[0, :6].abs().max() == 0 and sh[0, 18:].abs().max() == 0 and int(sh[0].argmax()) in (11, 12)
    assert (sh[1] > 0).sum() == 16 and torch.allclose(sh[2], torch.ones(24))
    assert sh.min() >= 0


def test_diel_response_is_periodic_damped_and_delayed():
    h = (torch.arange(24.0) + 0.5)
    f = torch.cos((h - 12.0) * (2.0 * math.pi / 24.0))[None]            # forcing with its maximum at noon
    fast = TN.diel_anomaly(f, torch.tensor([20.0]), 2)
    slow = TN.diel_anomaly(f, torch.tensor([0.1]), 2)
    assert fast.mean().abs() < 1e-6 and slow.mean().abs() < 1e-6
    amp = lambda c: (c.max() - c.min()).item() / 2.0  # noqa: E731
    assert amp(fast) == pytest.approx(1.0, abs=0.05) and 0.05 < amp(slow) < 0.5
    assert 12 <= int(fast[0].argmax()) <= 13 and 15 <= int(slow[0].argmax()) <= 19   # a deep reach peaks later
    # 2 cycles are close to the periodic state for the exchange rates of a stream (k · step ≥ 0.1)
    assert torch.allclose(slow, TN.diel_anomaly(f, torch.tensor([0.1]), 8), atol=0.03)
    assert torch.allclose(TN.diel_anomaly(torch.zeros(2, 24), torch.tensor([0.3, 0.3]), 2), torch.zeros(2, 24))


# ------------------------------------------------------------------------------------------------ the network
@pytest.mark.parametrize("over", [{}, {"mixing": "atmosphere_only"}, {"diel": {"enabled": False}},
                                  {"residual_cap": 0.0}, {"diel": {"enabled": True, "months": [6, 7, 8, 9]}}])
def test_output_contract_and_finite_gradients(over):
    net = _net(**over)
    x, s, _ = _inputs(T=120)
    L = 5
    out = net(x, s, L)
    Ld = L + N.PAD
    assert out["mu"].shape == (3, Ld, 2) and out["curve"].shape == (3, Ld, 24, 2) and out["direct"] is None
    assert out["sig_d"].shape == (3, Ld, N.N_DAILY) and out["sig_i"].shape == (3, Ld, 2)
    assert out["aux_mu"].shape == (3, Ld, N.N_AUX) and out["aux_sig"].shape == (3, Ld, N.N_AUX)
    assert out["curve"].mean(dim=2).abs().max() < 1e-4                   # the curve is an anomaly around the mean
    assert net.sigma_src().shape == (6, 2)
    loss = N.total_loss(net, N.loss_terms(net, out, _batch(L=L), L), 0.5)
    loss.backward()
    g = [p.grad for p in net.parameters() if p.grad is not None]
    assert g and all(torch.isfinite(v).all() for v in g)
    assert net.par.weight.grad.abs().sum() > 0                           # the physical parameters receive gradient
    assert (net.res is None) == (float({**NET, **over}["residual_cap"]) == 0.0)


def test_equilibrium_temperature_is_the_shared_routine_on_the_physical_inputs():
    net = _net(mixing="atmosphere_only", residual_cap=0.0).eval()
    x, s, _ = _inputs(T=90)
    with torch.no_grad():
        ph = net(x, s, 5, return_physics=True)["physics"]
    elev = (s[:, 0] * 300.0 + 500.0).numpy()
    p = TP.pressure_kpa(elev)[:, None]
    assert np.allclose(ph["p_kpa"].numpy(), p[:, 0], rtol=1e-5)
    kw = TP.surface_drivers(tmean=x[..., 0].numpy().astype("float64"), srad=x[..., 3].numpy().astype("float64"),
                            wind10=x[..., 4].numpy().astype("float64"), sph=x[..., 5].numpy().astype("float64"),
                            p_kpa=p, cloud_ratio=x[..., 10].numpy().astype("float64"),
                            shade=ph["shade"].numpy().astype("float64"),
                            wind_shelter=ph["params"]["wind_shelter"].numpy().astype("float64")[:, None])
    te, kx = TP.equilibrium_temperature(**kw)
    assert np.allclose(ph["te"].numpy(), te, atol=2e-3) and np.allclose(ph["k_exchange"].numpy(), kx, rtol=1e-3)
    assert torch.allclose(ph["t_star"], ph["te"].clamp(min=0.0))         # no other source without mixing
    assert TP.TE_RANGE[0] <= ph["te"].min() and ph["te"].max() <= TP.TE_RANGE[1]
    assert ph["tw_mean"].min() >= -1e-4                                  # liquid water
    # the lag of the upstream term follows the watershed area: sqrt(A) / speed, at most MAX_LAG_D
    area = 10.0 ** (s[:, 1] * 1.0 + 2.0) - 1e-3
    assert torch.allclose(ph["lag_d"], torch.clamp(area.clamp(min=0).sqrt() / TN.TRAVEL_SPEED_KM_D, 0, TN.MAX_LAG_D))


def test_shade_follows_the_season_and_summer_water_is_warmer_than_winter_water():
    net = _net(residual_cap=0.0).eval()
    x, s, doy = _inputs(T=400)
    with torch.no_grad():
        ph = net(x, s, 5, return_physics=True)["physics"]
    summer, winter = (doy > 170) & (doy < 250), (doy < 40) | (doy > 340)
    assert torch.all(ph["shade"][:, summer].mean(1) > 1.5 * ph["shade"][:, winter].mean(1))
    assert ph["shade"].max() <= 0.95 and ph["shade"].min() >= 0
    assert torch.all(ph["tw_mean"][:, summer].mean(1) > ph["tw_mean"][:, winter].mean(1) + 5.0)
    assert torch.all(ph["depth"] > 0.05 * math.exp(-0.6 * TN.Q_EXP_CLAMP) - 1e-6) and torch.isfinite(ph["k_day"]).all()
    assert 0 <= ph["melt"].min() and ph["melt"].max() < 0.2              # melt share starts near zero


def test_residual_is_frozen_during_warm_up_and_capped_afterwards():
    net = _net(residual_cap=1.0, warmup_epochs=2).eval()
    with torch.no_grad():
        net.res.weight.fill_(5.0)
        net.res.bias.fill_(3.0)
    x, s, _ = _inputs(T=90)
    for epoch, on in ((0, False), (1, False), (2, True), (7, True)):
        net.set_epoch(epoch)
        with torch.no_grad():
            r = net(x, s, 5, return_physics=True)["physics"]["residual"]
        assert (r.abs().max() > 0) == on and r.abs().max() <= 1.0 + 1e-6
    assert "res_on" in net.state_dict()                                  # the switch is stored with the weights
    free = _net(residual_cap=1.0, warmup_epochs=0)
    assert free.res_on.item() == 1.0
    assert torch.all(free.res.weight == 0) and torch.all(free.res.bias == 0)   # the residual starts at exactly 0


def test_fallback_months_use_the_physical_curve_only_on_those_days():
    x, s, doy = _inputs(T=400)
    L = 300
    full, part, off = _net().eval(), _net(diel={"enabled": True, "months": [6, 7, 8, 9]}).eval(), \
        _net(diel={"enabled": False}).eval()
    part.load_state_dict(full.state_dict())
    off.load_state_dict(full.state_dict())
    with torch.no_grad():
        cf, cp, co = (n(x, s, L)["curve"] for n in (full, part, off))
    d = doy[-(L + N.PAD):]
    jjas = (d >= 152) & (d <= 273)                                       # 1 June .. 30 September
    assert jjas.sum() == 122
    assert torch.allclose(cp[:, jjas, :, 0], cf[:, jjas, :, 0], atol=1e-5)
    assert torch.allclose(cp[:, ~jjas, :, 0], co[:, ~jjas, :, 0], atol=1e-6)
    assert not torch.allclose(cf[:, ~jjas, :, 0], co[:, ~jjas, :, 0], atol=1e-3)
    assert torch.equal(cp[..., 1], cf[..., 1])                           # the DO curve is the statistical head
    with torch.no_grad():                                                # the physical curve peaks in the afternoon
        peak = cf[:, jjas, :, 0].argmax(dim=-1).float()
    assert 12 <= peak.median() <= 19


def test_configuration_guards():
    for bad, msg in (({"powell_prior": True}, "Powell"), ({"te_range": [-5, 40]}, "te_range"),
                     ({"prior": {"shade": 0.5, "wind_shelter": 0.6}}, "prior"), ({"mixing": "x"}, "mixing"),
                     ({"diel": {"enabled": True, "steps": 12}}, "24 steps")):
        with pytest.raises(ValueError, match=msg):
            _net(**bad)
    ctx = json.loads(json.dumps(CTX))                                    # the context is plain JSON
    assert TN.make_net(14, 5, NET, ctx) is not None


# ------------------------------------------------------------------------------------------------ shared trainer
@pytest.fixture(scope="module")
def v2():
    return synth_v2()


def test_shared_trainer_fits_scores_and_round_trips(v2, tmp_path):
    t = Trainer(TrainCfg(net=NET, **CFG), make=TN.make_net)
    info = t.fit(v2, device="cpu", log=QUIET, assert_finite=True, checkpoint_path=tmp_path / "m.pt")
    assert info["n_dense"] == len(v2.fit_idx) and np.isfinite(info["best_J"])
    assert float(t.state["res_on"]) == (1.0 if info["best_epoch"] >= NET["warmup_epochs"] else 0.0)
    ctx = TN.physics_context(t.scales, t.info["inputs"]["dynamic"], t.info["inputs"]["static"])
    assert ctx["has_upstream"] is False and set(ctx["static"]) == {"elev_m", "ws_area_km2"}
    jt = VD.DAILY.index("T_mean")
    assert ctx["label"]["t_mean"] == pytest.approx(float(t.scales.label_mean[jt]))
    t2 = Trainer.load(tmp_path / "m.pt", make=TN.make_net)
    dyn, st, _, _ = VD.make_inputs(v2.data, v2.solar, v2.vaa, t2.scales)
    need = np.zeros((2, len(v2.data.dates)), bool)
    need[:, -150:] = True
    a = t.predict(dyn, st, v2.inner_idx, need, hours=v2.inst_hour[v2.inner_idx], device="cpu")
    b = t2.predict(dyn, st, v2.inner_idx, need, hours=v2.inst_hour[v2.inner_idx], device="cpu")
    np.testing.assert_array_equal(a["daily"], b["daily"])
    assert np.isfinite(a["daily"][:, -150:]).all()
    with pytest.raises(ValueError, match="fitted on"):
        t.predict(dyn[..., :-1], st, v2.inner_idx, need, device="cpu")


def test_warm_up_epochs_reach_the_network_through_the_trainer(v2):
    t = Trainer(TrainCfg(net=NET, **{**CFG, "max_epochs": 1}), make=TN.make_net)
    t.fit(v2, device="cpu", log=QUIET)
    assert float(t.state["res_on"]) == 0.0 and float(t.state["res.weight"].abs().sum()) == 0.0


# ------------------------------------------------------------------------------------------------ config and kill rule
def test_d2_config_builds_and_lambda_q_sets_the_discharge_weight():
    for point, lam, cap in (("g1", 1.0, 0.0), ("g2", 1.0, 1.0), ("g3", 0.3, 1.0)):
        cfg = VR.load_config(VR.CFG_DIR / "d2.yaml", point)
        assert cfg["model"] == "thermox" and float(cfg["net"]["residual_cap"]) == cap
        tc = VR.train_cfg(VR.resolve_d2_diel({**cfg, "_smoke": True}), 0)
        assert tc.aux_w["logq"] == lam and {k: v for k, v in tc.aux_w.items() if k != "logq"} == \
            {k: v for k, v in N.AUX_W.items() if k != "logq"}
        assert TN.make_net(14, 5, tc.net, CTX) is not None
    d1 = VR.train_cfg(VR.load_config(VR.CFG_DIR / "d1.yaml", "g1"), 0)
    assert d1.aux_w is None                                              # other arms keep the default weights


def test_kill_rule_decision_and_diel_days(tmp_path):
    assert VR.d2_kill_decision(0.3, None, 0.5) == "all_days" and VR.d2_kill_decision(0.5, None, 0.5) == "all_days"
    assert VR.d2_kill_decision(0.7, None, 0.5) == "needs_step_2"
    assert VR.d2_kill_decision(0.7, 0.4, 0.5) == "fallback_months" and VR.d2_kill_decision(0.7, 0.6, 0.5) == "dropped"
    cfg = VR.load_config(VR.CFG_DIR / "d2.yaml", "g1")
    assert cfg["net"]["diel"]["days"] == "rule"
    sm = VR.resolve_d2_diel({**cfg, "_smoke": True})
    assert sm["net"]["diel"]["months"] is None and "days" not in sm["net"]["diel"]
    assert VR.resolve_d2_diel({**cfg, "_smoke": True}, fallback=True)["net"]["diel"]["months"] == [6, 7, 8, 9]
    assert cfg["net"]["diel"]["days"] == "rule"                          # the loaded config is not changed
    rec = tmp_path / "kill_rule.json"
    with pytest.raises(VR.NotStampedError, match="does not exist"):
        VR.resolve_d2_diel(cfg, record=rec)
    sha = VR.arm_release_token("d2").split()[-1]
    rec.write_text(json.dumps({"decision": "all_days", "nets_sha256": "0" * 64}))
    with pytest.raises(VR.NotStampedError, match="another state"):       # measured on an older network file
        VR.resolve_d2_diel(cfg, record=rec)
    for dec, want in (("all_days", None), ("fallback_months", [6, 7, 8, 9])):
        rec.write_text(json.dumps({"decision": dec, "nets_sha256": sha}))
        assert VR.resolve_d2_diel(cfg, record=rec)["net"]["diel"]["months"] == want
    for dec in ("dropped", "needs_step_2"):
        rec.write_text(json.dumps({"decision": dec, "nets_sha256": sha}))
        with pytest.raises(VR.NotStampedError, match="not runnable"):
            VR.resolve_d2_diel(cfg, record=rec)
    nod = VR.load_config(VR.CFG_DIR / "d2.yaml", "abl_nodiel", base="g1") if "abl_nodiel" in _points("d2") else None
    if nod is not None:                                                  # diel off: no record is needed
        assert VR.resolve_d2_diel(nod, record=rec)["net"]["diel"]["enabled"] is False


def _points(arm):
    import yaml

    doc = yaml.safe_load((VR.CFG_DIR / f"{arm}.yaml").read_text())
    return set(doc.get("grid", {})) | set(doc.get("ablations", {}))


def test_kill_rule_record_holds_times_and_hashes_only(tmp_path):
    cfg = VR.load_config(VR.CFG_DIR / "d2.yaml", "g1", smoke=True)

    def smoke(name, s, n=50, smoke=True, blocks=("train",)):
        d = tmp_path / name
        d.mkdir()
        (d / "train_info.json").write_text(json.dumps({"smoke": smoke, "s_per_step_median": s, "n_params": 7,
                                                       "epochs": 2, "gpu_memory": {"max_allocated_mb": 1.0},
                                                       "history": [{"s_per_step": s}, {"s_per_step": s}]}))
        (d / "data_info.json").write_text(json.dumps({"blocks": list(blocks), "n_locations_used": n}))
        return d

    rec = tmp_path / "kr" / "kill_rule.json"
    out = VR.record_d2_kill_rule(cfg, smoke("a", 0.2), record=rec, register=False)
    assert out["decision"] == "all_days" and json.loads(rec.read_text())["decision"] == "all_days"
    assert out["max_s_per_step"] == 0.5 and out["batch_size"] == 256 and len(out["nets_sha256"]) == 64
    assert VR.arm_release_token("d2").endswith(out["nets_sha256"])
    assert VR.record_d2_kill_rule(cfg, smoke("b", 0.9), record=rec, register=False)["decision"] == "needs_step_2"
    assert VR.record_d2_kill_rule(cfg, tmp_path / "b", smoke("c", 0.4), record=rec,
                                  register=False)["decision"] == "fallback_months"
    assert VR.record_d2_kill_rule(cfg, tmp_path / "b", smoke("d", 0.8), record=rec,
                                  register=False)["decision"] == "dropped"
    with pytest.raises(VR.NotStampedError, match="not runnable"):
        VR.resolve_d2_diel(VR.load_config(VR.CFG_DIR / "d2.yaml", "g1"), record=rec)
    for bad in (smoke("e", 0.2, n=51), smoke("f", 0.2, smoke=False), smoke("g", 0.2, blocks=("train", "val"))):
        with pytest.raises(RuntimeError):
            VR.record_d2_kill_rule(cfg, bad, record=rec, register=False)
