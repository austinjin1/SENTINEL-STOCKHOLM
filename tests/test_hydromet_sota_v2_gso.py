# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""M6 architecture-v2 (GSO-MT, R-TD, common trainer and criterion): code-path and property tests on synthetic arrays.
No real observation is read and no number printed here is a research result."""

import math

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from sentinel.hydromet_sota.gso_mt import criterion as C  # noqa: E402
from sentinel.hydromet_sota.gso_mt import data as VD  # noqa: E402
from sentinel.hydromet_sota.gso_mt import grab as G  # noqa: E402
from sentinel.hydromet_sota.gso_mt import nets as N  # noqa: E402
from sentinel.hydromet_sota.gso_mt import runner as VR  # noqa: E402
from sentinel.hydromet_sota.gso_mt.solar import MIX_HOURS, local_solar_hour, solar_channels  # noqa: E402
from sentinel.hydromet_sota.gso_mt.synthetic import synth_v2  # noqa: E402
from sentinel.hydromet_sota.gso_mt.trainer import ResumeNeeded, TrainCfg, Trainer, _tiles  # noqa: E402

torch.set_num_threads(2)
NET = {"hidden": 12, "heads": "diel", "static": "film", "dropout": 0.2, "static_noise": 0.3, "attr_dropout": 0.1}
CFG = dict(arm="t", seq_len=60, predict_last=20, batch_size=16, steps_per_epoch=4, max_epochs=4, patience=10, seed=3,
           grab_share=0.5)
QUIET = lambda *_: None  # noqa: E731


@pytest.fixture(scope="module")
def v2():
    return synth_v2()


# ------------------------------------------------------------------------------------------------ solar and grab rules
def test_solar_channels_and_local_hour():
    doy = np.array([80, 172, 355])
    ch = solar_channels(np.array([0.0, 60.0]), np.array([0.0, 500.0]), doy, np.full((2, 3), 200.0, "float32"))
    assert ch.shape == (2, 3, 4)
    np.testing.assert_allclose(ch[0, :, 0], 12.0, atol=0.05)                 # equator: 12 h all year
    assert 18.0 < ch[1, 1, 0] < 19.2 and 5.0 < ch[1, 2, 0] < 6.2             # 60°N: solstices
    assert (ch[..., 2] >= 0).all() and (ch[..., 2] <= 1.2).all()
    np.testing.assert_allclose(ch[1, 1, 3], 90 - (60 - 23.45), atol=0.2)     # noon elevation at the June solstice
    np.testing.assert_allclose(local_solar_hour(np.array([18.0, 2.0]), np.array([-90.0, -90.0])), [12.0, 20.0])


def test_provider_class_first_match():
    assert G.provider_class("NWIS", "USGS-CO", "USGS Colorado Water Science Center", None) == "usgs"
    assert G.provider_class("STORET", "REDLAKE_WQX", "Red Lake Band of Chippewa Indians", None) == "tribal"
    assert G.provider_class("STORET", "X", "Some County Health", "Yes") == "tribal"
    assert G.provider_class("STORET", "RWATCH", "Colorado River Watch volunteer program", None) == "volunteer"
    assert G.provider_class("STORET", "21COL001", "anything", None) == "state"
    assert G.provider_class("STORET", "ABC", "Department of Environmental Quality", None) == "state"
    assert G.provider_class("STORET", "ABC", "City of Springfield", None) == "local"
    assert G.provider_class("STORET", "ABC", "Acme Consulting", None) == "other"


def test_collapse_rules():
    d = pd.Timestamp("2015-06-01")
    rows = [("a", d, "T", 10.0, 9.0), ("a", d, "T", 12.0, 10.0),                        # same visit: mean / mean hour
            ("b", d, "T", 10.0, 8.0), ("b", d, "T", 20.0, 15.0),                        # wide span: first timed sample
            ("c", d, "DO", 7.0, np.nan)]                                                # no time
    rows += [("lg", d, "T", 10.0 + h, h + 0.25) for h in range(24)]                      # logger day
    inst, daily, counts = G.collapse(pd.DataFrame(rows, columns=["site_id", "date", "var", "value", "hour"]))
    g = inst[inst["kind"] == "grab"].set_index("site_id")
    assert g.loc["a", "value"] == 11.0 and g.loc["a", "hour"] == 9.5
    assert g.loc["b", "value"] == 10.0 and g.loc["b", "hour"] == 8.0
    assert np.isnan(g.loc["c", "hour"]) and g.loc["c", "value"] == 7.0
    lg = inst[inst["kind"] == "logger"]
    assert sorted(np.floor(lg["hour"]).astype(int)) == list(G.FIXED_HOURS)
    assert len(daily) == 1 and daily.iloc[0]["mean"] == pytest.approx(21.5) and daily.iloc[0]["max"] == 33.0
    assert counts["visit_days"] == 2 and counts["wide_span_days"] == 1 and counts["logger_days"] == 1
    assert counts["rows_dropped_in_wide_span_days"] == 1 and counts["inst_rows_without_time"] == 1


# ------------------------------------------------------------------------------------------------ operator
def test_diel_curve_zero_mean_and_peak():
    th = torch.tensor([2 * math.pi * 15 / 24, 2 * math.pi * 6 / 24])
    cv = N.diel_curve(torch.tensor([2.0, 0.5]), torch.cos(th), torch.sin(th), torch.tensor([0.0, 0.5]))
    assert cv.shape == (2, 24)
    np.testing.assert_allclose(cv.mean(-1).numpy(), 0.0, atol=1e-6)
    assert int(cv[0].argmax()) == 15 and cv[0].max().item() == pytest.approx(2.0, abs=1e-5)
    assert int(cv[1].argmax()) == 6 and cv[1].max().item() == pytest.approx(0.75, abs=1e-5)
    # periodic linear interpolation: grid points exact, 23.5 h is the mean of 23 h and 0 h
    h = torch.tensor([15.0, 6.0])
    np.testing.assert_allclose(N.interp_curve(cv, h).numpy(), [2.0, 0.75], atol=1e-5)
    v = N.interp_curve(cv, torch.tensor([23.5, 47.5]))
    np.testing.assert_allclose(v.numpy(), ((cv[:, 23] + cv[:, 0]) / 2).numpy(), atol=1e-6)


def _fake_out(B=3, L=5, seed=0, direct=False):
    g = torch.Generator().manual_seed(seed)
    Ld = L + N.PAD
    th = torch.rand(B, Ld, 2, generator=g) * 2 * math.pi
    cv = N.diel_curve(torch.rand(B, Ld, 2, generator=g) + 0.1, torch.cos(th), torch.sin(th),
                      0.5 * torch.rand(B, Ld, 2, generator=g)).permute(0, 1, 3, 2)
    return {"mu": torch.randn(B, Ld, 2, generator=g), "curve": cv,
            "direct": torch.randn(B, Ld, 5, generator=g) if direct else None,
            "sig_d": torch.rand(B, Ld, 7, generator=g) + 0.5, "sig_i": torch.rand(B, Ld, 2, generator=g) + 0.5,
            "aux_mu": torch.randn(B, Ld, 4, generator=g), "aux_sig": torch.rand(B, Ld, 4, generator=g) + 0.5}


def test_daily_stats_from_curve_and_direct():
    L = 5
    out = _fake_out(L=L)
    ds = N.daily_stats(out, L)
    tmax = out["mu"][..., 0] + out["curve"][..., 0].amax(-1)
    np.testing.assert_allclose(ds[..., 1].numpy(), tmax[:, -L:].numpy(), atol=1e-6)
    for k in range(L):                                                       # trailing 7-day mean of the daily max
        e = N.PAD + k
        np.testing.assert_allclose(ds[:, k, 3].numpy(), tmax[:, e - 6:e + 1].mean(1).numpy(), atol=1e-5)
    assert (ds[..., 1] >= ds[..., 0]).all() and (ds[..., 2] <= ds[..., 0]).all()
    assert (ds[..., 5] <= ds[..., 4]).all() and (ds[..., 6] >= ds[..., 4]).all()
    out2 = _fake_out(L=L, direct=True)                                       # independent heads: no operator
    ds2 = N.daily_stats(out2, L)
    np.testing.assert_array_equal(ds2[..., 3].numpy(), out2["direct"][:, -L:, 2].numpy())
    np.testing.assert_array_equal(ds2[..., 4].numpy(), out2["mu"][:, -L:, 1].numpy())


def test_inst_point_timed_and_untimed():
    L = 5
    out = _fake_out(L=L)
    hour = torch.full((3, L, 2), 13.0)
    hour[0, 0, 0] = float("nan")
    p = N.inst_point(out, L, hour)
    cv = out["curve"][:, -L:]
    np.testing.assert_allclose(p[1, 2, 1].item(), (out["mu"][1, -L + 2, 1] + cv[1, 2, 13, 1]).item(), atol=1e-6)
    mix = cv[0, 0, torch.as_tensor(MIX_HOURS), 0].mean()
    np.testing.assert_allclose(p[0, 0, 0].item(), (out["mu"][0, -L, 0] + mix).item(), atol=1e-6)
    out["curve"] = None                                                      # no diel information: the mean
    np.testing.assert_array_equal(N.inst_point(out, L, hour).numpy(), out["mu"][:, -L:].numpy())


def _batch(B=3, L=5, seed=1):
    g = torch.Generator().manual_seed(seed)
    b = {"yd": torch.randn(B, L, 11, generator=g), "md": (torch.rand(B, L, 11, generator=g) < 0.7).float(),
         "yh": torch.randn(B, L, 24, 2, generator=g), "mh": (torch.rand(B, L, 24, 2, generator=g) < 0.5).float(),
         "hfrac": torch.rand(B, generator=g), "yi": torch.randn(B, L, 2, generator=g),
         "mi": (torch.rand(B, L, 2, generator=g) < 0.6).float(), "hi": torch.rand(B, L, 2, generator=g) * 24,
         "pc": torch.tensor([0, 3, 5])[:B]}
    b["mi"][0, 0, 0], b["hi"][0, 0, 0] = 1.0, float("nan")
    return b


class _Src:
    def sigma_src(self):
        return torch.full((6, 2), 0.3)


def test_untimed_grab_row_is_the_uniform_daytime_mixture():
    L = 5
    out, b = _fake_out(L=L), _batch(L=L)
    b["mi"] = torch.zeros_like(b["mi"])
    b["mi"][0, 0, 0] = 1.0                                                   # a single untimed T row
    t = N.loss_terms(_Src(), out, b, L)
    mu, cv = out["mu"][0, -L, 0], out["curve"][0, -L, :, 0]
    sg = math.sqrt(out["sig_i"][0, -L, 0].item() ** 2 + 0.3 ** 2)
    dens = [math.exp(-0.5 * ((b["yi"][0, 0, 0] - mu - cv[h]).item() / sg) ** 2) / (sg * math.sqrt(2 * math.pi))
            for h in MIX_HOURS]
    assert t["grab_T"].item() == pytest.approx(-math.log(np.mean(dens)), rel=1e-5)
    assert "grab_DO" not in t


def test_total_loss_weights_and_variable_filter():
    L = 5
    out, b = _fake_out(L=L), _batch(L=L)
    src = _Src()
    t = N.loss_terms(src, out, b, L)
    assert set(t) == set(VD.DAILY) | {"hour_T", "hour_DO", "grab_T", "grab_DO"}
    dense = sum(N.DENSE_W[k] * t[k] for k in N.DENSE_W) / sum(N.DENSE_W.values())
    grab = (t["grab_T"] + t["grab_DO"]) / 2
    aux = sum(N.AUX_W[k] * t[k] for k in N.AUX_W)
    prior = N.SRC_PRIOR_W * 0.5 * ((0.3 / N.SRC_PRIOR_SCALE) ** 2) * 12
    assert N.total_loss(src, t, 0.25).item() == pytest.approx((0.75 * dense + 0.25 * grab + aux).item() + prior, rel=1e-5)
    tT = N.total_loss(src, t, 0.5, variables=("T",), aux_w={})
    kT = [k for k in N.DENSE_W if k.split("_")[0] == "T" or k == "hour_T"]
    dT = sum(N.DENSE_W[k] * t[k] for k in kT) / sum(N.DENSE_W[k] for k in kT)
    assert tT.item() == pytest.approx((0.5 * dT + 0.5 * t["grab_T"]).item() + prior, rel=1e-5)
    only = {k: v for k, v in t.items() if not k.startswith("grab_")}        # no grab rows: dense weight 1, no prior
    assert N.total_loss(src, only, 0.5).item() == pytest.approx((dense + aux).item(), rel=1e-5)


@pytest.mark.parametrize("heads,static", [("diel", "film"), ("indep_op", "concat"), ("indep", "film")])
def test_networks_have_finite_gradients(heads, static):
    net = N.make_net(7, 5, {**NET, "heads": heads, "static": static})
    L = 5
    out = net(torch.randn(3, 30, 7), torch.randn(3, 5), L)
    assert out["mu"].shape == (3, L + N.PAD, 2)
    assert (out["curve"] is None) == (heads == "indep") and (out["direct"] is None) == (heads == "diel")
    loss = N.total_loss(net, N.loss_terms(net, out, _batch(L=L), L), 0.5)
    loss.backward()
    g = [p.grad for p in net.parameters() if p.grad is not None]
    assert g and all(torch.isfinite(x).all() for x in g)
    if heads != "indep":                                                     # the phase initialisation peaks at 15 h
        net.eval()
        with torch.no_grad():
            cv = net.decode(torch.zeros(1, 1, net.out.out_features) + net.out.bias)["curve"]
        assert int(cv[0, 0, :, 0].argmax()) == 15


# ------------------------------------------------------------------------------------------------ criterion
def test_criterion_formula_and_station_cap():
    rs = np.random.default_rng(0)
    y = rs.normal(size=(3, 50, 11)).astype("float32")
    p = y[:, :, :7] + 0.5
    y[0, :10] = np.nan
    cf = {"temp_c_mean": 1.0, "temp_c_7dadm": 2.0, "do_mgl_mean": 1.0, "do_mgl_min": 4.0}
    js = C.j_sonde(p, y, cf)
    assert js["j_sonde"] == pytest.approx(0.25 * (1 + 2 + 1 + 4) / 4, rel=1e-5)
    assert js["n_cells"]["temp_c_mean"] == 140
    assert C.j_sonde(p, y, cf, variables=("T",))["j_sonde"] == pytest.approx(0.25 * 1.5, rel=1e-5)
    yi = np.full((2, 100, 2), np.nan, "float32")
    yi[0, :90], yi[1, :10] = 0.0, 0.0                                        # 90 rows vs 10 rows
    pi = np.zeros_like(yi)
    pi[0], pi[1] = 1.0, 2.0                                                  # squared errors 1 and 4
    jg = C.j_grab(pi, yi, cap=30)
    assert jg["by_variable"]["T"] == pytest.approx((30 * 1 + 10 * 4) / 40)
    assert C.combine(js, jg) == pytest.approx(0.8 * js["j_sonde"] + 0.2 * jg["j_grab"])
    assert C.combine(js, None) == js["j_sonde"]


def test_tiles_cover_every_needed_day_once():
    rs = np.random.default_rng(1)
    need = rs.random((4, 500)) < 0.05
    need[3] = False
    cov = np.zeros_like(need, dtype=int)
    for i, e in _tiles(need, 60, 20):
        assert e >= 59
        cov[i, e - 19:e + 1] += 1
    assert cov.max() <= 1
    assert (cov[need & (np.arange(500) >= 59)[None]] == 1).all()


# ------------------------------------------------------------------------------------------------ trainer
def _hist(t):
    return [{k: h.get(k) for k in ("epoch", "train_loss", "J", "j_sonde", "j_grab")} for h in t.history]


def test_trainer_runs_scores_and_round_trips(v2, tmp_path):
    t = Trainer(TrainCfg(net=NET, **CFG))
    info = t.fit(v2, device="cpu", log=QUIET, assert_finite=True, checkpoint_path=tmp_path / "m.pt")
    assert info["n_dense"] == len(v2.fit_idx) and info["n_grab"] == len(v2.grab_fit_idx)
    h = t.history[info["best_epoch"]]
    assert h["J"] == pytest.approx(0.8 * h["j_sonde"] + 0.2 * h["j_grab"])
    assert set(h["by_target"]) == set(VD.PRIMARY)
    t2 = Trainer.load(tmp_path / "m.pt")
    dyn, st, _, _ = VD.make_inputs(v2.data, v2.solar, v2.vaa, t2.scales)
    need = np.zeros((2, len(v2.data.dates)), bool)
    need[:, -200:] = True
    a = t.predict(dyn, st, v2.inner_idx, need, hours=v2.inst_hour[v2.inner_idx], device="cpu")
    b = t2.predict(dyn, st, v2.inner_idx, need, hours=v2.inst_hour[v2.inner_idx], device="cpu")
    np.testing.assert_array_equal(a["daily"], b["daily"])
    assert np.isfinite(a["daily"][:, -200:]).all() and np.isnan(a["daily"][:, :-200]).all()
    assert a["inst"].shape == (2, len(v2.data.dates), 2)


def test_trainer_resume_is_exact(v2, tmp_path):
    ref = Trainer(TrainCfg(net=NET, **CFG))
    ref.fit(v2, device="cpu", log=QUIET)
    rp, jobs = tmp_path / "resume.pt", 0
    while True:
        m = Trainer(TrainCfg(net=NET, **CFG))
        jobs += 1
        try:
            m.fit(v2, device="cpu", log=QUIET, resume_path=rp, job_minutes=1e-9)
            break
        except ResumeNeeded:
            pass
    assert jobs >= 3 and _hist(ref) == _hist(m)
    for k in ref.state:
        assert torch.equal(ref.state[k], m.state[k]), k
    other = Trainer(TrainCfg(net=NET, **{**CFG, "seed": 4}))
    with pytest.raises(RuntimeError, match="another configuration"):
        other.fit(v2, device="cpu", log=QUIET, resume_path=rp)


def test_no_grab_variant_uses_j_sonde_alone_and_temperature_only_stage(v2):
    t = Trainer(TrainCfg(net={**NET, "heads": "indep"}, **{**CFG, "grab_share": 0.0}))
    info = t.fit(v2.subset(np.r_[v2.fit_idx, v2.inner_idx]), device="cpu", log=QUIET)
    assert info["n_grab"] == 0 and info["best_j_grab"] is None and info["best_J"] == info["best_j_sonde"]
    assert not any(k.startswith("grab_") for k in t.history[-1]["train_terms"])
    tt = Trainer(TrainCfg(net=NET, variables=("T",), **CFG))
    it = tt.fit(v2, device="cpu", log=QUIET)
    assert set(k for k, x in it["best_by_target"].items() if x is not None) == {"temp_c_mean", "temp_c_7dadm"}
    assert not any("DO" in k for k in tt.history[-1]["train_terms"])


def test_refit_uses_fit_and_inner_without_validation(v2):
    t = Trainer(TrainCfg(net=NET, **CFG))
    info = t.fit(v2, refit_epochs=2, device="cpu", log=QUIET)
    assert info["epochs"] == 2 and info["best_J"] is None and "J" not in t.history[-1]
    assert info["n_dense"] == len(v2.fit_idx) + len(v2.inner_idx)
    assert info["n_grab"] == len(v2.grab_fit_idx) + len(v2.grab_inner_idx)


def test_trainer_refuses_non_train_roles(v2):
    bad = v2.subset(np.arange(len(v2.data.sites)))
    bad.data.roles = bad.data.roles.copy()
    bad.data.roles[0] = "val"
    with pytest.raises(ValueError, match="train-role"):
        Trainer(TrainCfg(net=NET, **CFG)).fit(bad, device="cpu", log=QUIET)


def test_scales_use_final_fit_sites_only(v2):
    sc = VD.fit_scales(v2)
    assert sc.norm.fit_sites == [v2.data.sites[i] for i in v2.final_fit_idx]
    j = VD.DAILY.index("T_mean")
    x = v2.daily[v2.final_fit_idx][:, :, j]
    assert sc.label_mean[j] == pytest.approx(np.nanmean(x.astype("float64")))
    assert sc.label_std[VD.DAILY.index("T_7dadm")] == sc.label_std[j]       # one temperature axis
    back = VD.Scales.from_state(sc.state())
    np.testing.assert_array_equal(back.label_std, sc.label_std)
    assert back.crit_factor() == sc.crit_factor()


def test_stamp_guard_needs_registry_row_with_current_hash_and_read_only_file(tmp_path, monkeypatch):
    import hashlib
    import json as _json
    import os

    from sentinel.hydromet_sota.gso_mt import runner as VR

    am = tmp_path / "amendment.md"
    am.write_text("draft v1\n")
    reg = tmp_path / "registry.jsonl"
    monkeypatch.setattr(VR, "AMENDMENT", am)
    with pytest.raises(VR.NotStampedError):                       # no registry at all
        VR.assert_stamped(registry_path=reg)

    def row(sha, ws=VR.WORKSTREAM, exp="hydromet_arch_v2/prereg_snapshot", by="lead"):
        return _json.dumps({"id": "abc123", "workstream": ws, "experiment": exp, "timestamp": "t",
                            "extra": {"sha256": {VR.AMENDMENT_REL: sha}, "written_by": by}}) + "\n"
    sha = hashlib.sha256(am.read_bytes()).hexdigest()
    reg.write_text(row("0" * 64) + row(sha, ws="M1-sonde") + row(sha, exp="M6V2/inner_v1")
                   + row(sha, by="M6 arch-v2 engineer") + row(sha, by=None) + row(sha, exp="hydromet_sota/prereg_snapshot"))
    with pytest.raises(VR.NotStampedError):                       # other hash, workstream, experiment or writer
        VR.assert_stamped(registry_path=reg)
    with reg.open("a") as fh:
        fh.write(row(sha))
    with pytest.raises(VR.NotStampedError, match="read-only"):    # row exists, file still writable
        VR.assert_stamped(registry_path=reg)
    os.chmod(am, 0o444)
    st = VR.assert_stamped(registry_path=reg)
    assert st["stamped"] and st["registry_id"] == "abc123" and st["amendment_sha256"] == sha
    os.chmod(am, 0o644)
    am.write_text("draft v2\n")                                   # an edit after the stamp closes the gate again
    os.chmod(am, 0o444)
    with pytest.raises(VR.NotStampedError):
        VR.assert_stamped(registry_path=reg)


def test_real_amendment_is_not_stamped_yet_or_stamp_is_consistent():
    """Before the stamp the guard is closed; after it the row's hash equals the file's (never a half state)."""
    from sentinel.hydromet_sota.gso_mt import runner as VR

    st = VR.stamp_status()
    assert st["stamped"] == (bool(st["registry_id"]) and st["read_only"])


def test_pool_guard_refuses_reserved_sites_in_every_id_form(tmp_path):
    """Lead ruling 2026-11-10 (repo-wide guard): every arch-v2 table, loader and training record passes its complete
    location list through ``runner.assert_pool_clean`` (shared helper + M6 guard + embedded-number scan)."""
    from sentinel.data import reserved as shared
    from sentinel.hydromet_sota import reserved as R

    some = sorted(shared.reserved_ids())[0]
    ok = ["01646500", "WQP:USGS-01646500", "WQP:21FLA_WQX-ABC12", "S0003"]
    assert VR.assert_pool_clean(ok) == len(ok)
    for bad in (some, f"USGS-{some}", f"WQP:USGS-{some}", f"WQP:ORG-{some}-A"):
        with pytest.raises((shared.ReservedSiteError, R.ReservedSiteError)):
            VR.assert_pool_clean(ok + [bad])
    cfg = {"arm": "d1", "point": "g1", "_smoke": True}
    with pytest.raises((shared.ReservedSiteError, R.ReservedSiteError)):       # refused before a run record opens
        VR.training_run(cfg, tmp_path, seed=0, refit=False, sites=ok + [some])
    assert not list(tmp_path.iterdir())


def test_v2_files_import_the_shared_reserved_helper():
    """Same static rule as tests/test_reserved_loaders.py, applied to every arch-v2 file that names a shared tier."""
    import re
    from pathlib import Path

    root = Path(VR.__file__).resolve().parents[3]
    tier = re.compile(r"sonde_daily|sonde_hourly|usgs_iv|SITE_DIR|SONDE_INDEX")
    files = (list((root / "sentinel" / "hydromet_sota").glob("*/*.py"))
             + list((root / "scripts").glob("*/sota_hydromet_v2_*.py")))
    assert len(files) > 5
    bad = [f.name for f in files if tier.search(f.read_text())
           and not re.search(r"(data\.reserved|from \.reserved) import", f.read_text())]
    assert not bad, bad


def test_grab_share_is_required_and_rtd_share_is_fixed():
    """Lead rulings §2.2: R-TD's grab share is fixed at 0.5; the form without grab rows is its registered variant."""
    d1 = VR.load_config(VR.CFG_DIR / "d1.yaml", "g1")
    assert VR.train_cfg(d1, 0).grab_share == 0.5
    rtd = VR.load_config(VR.CFG_DIR / "rtd.yaml", "fixed", smoke=True)
    assert VR.train_cfg(rtd, 0).grab_share == 0.5 and rtd["variant_of"] is None and rtd["data"]["grab"] is True
    pub = VR.load_config(VR.CFG_DIR / "rtd.yaml", "published", smoke=True)
    assert VR.train_cfg(pub, 0).grab_share == 0.0 and pub["variant_of"] == "fixed" and pub["data"]["grab"] is False
    assert {k: v for k, v in pub["net"].items()} == {k: v for k, v in rtd["net"].items()}   # one configuration
    with pytest.raises(ValueError, match="required"):
        Trainer(TrainCfg(net=NET, **{k: v for k, v in CFG.items() if k != "grab_share"}))
