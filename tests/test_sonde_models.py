# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Tests for the SENTINEL-Sonde models (GBM, LSTM, Chronos-2 wrappers, event heads, guard).

Synthetic arrays here only exercise code paths and leakage properties; no test output is a
research result. No network access, no real data (the Chronos-2 test uses a tiny randomly
initialised model).
"""

import json

import numpy as np
import pandas as pd
import pytest

from sentinel.data.sonde_splits import TIME_BLOCKS, SondeSplit, make_split
from sentinel.eval import ew_scoring as EWS
from sentinel.eval.splits import assert_disjoint
from sentinel.models.baselines.sonde_baselines import QUANTILES, SondeBaselines
from sentinel.models.sonde import common, guard
from sentinel.models.sonde import events as E
from sentinel.models.sonde.data import Cube, climatology, to_model_space
from sentinel.models.sonde.evaluate import score_rows, score_samples
from sentinel.models.sonde.features import FeatureBuilder
from sentinel.models.sonde.gbm import GBMConfig, SondeGBM, fit_cal_masks

VARS = ("temp_c_mean", "do_mgl_min", "turb_fnu_mean", "discharge_cfs_mean")


def synth_data(S=6, roles=None, variables=VARS, seed=0, end="2023-09-30", miss=0.05) -> common.SondeData:
    dates = pd.date_range("2007-10-01", end, freq="D")
    rs = np.random.default_rng(seed)
    T, V = len(dates), len(variables)
    season = np.sin(2 * np.pi * (dates.dayofyear.to_numpy() - 110) / 365.25)
    y = np.empty((S, T, V), np.float32)
    for i in range(S):
        for j, v in enumerate(variables):
            e = rs.normal(size=T)
            a = np.empty(T)
            a[0] = e[0]
            for t in range(1, T):
                a[t] = 0.9 * a[t - 1] + 0.45 * e[t]
            if v == "temp_c_mean":
                x = 12 + 8 * season + a
            elif v == "do_mgl_min":
                x = (6.0 if i % 2 == 0 else 8.5) - 2.5 * season + 0.8 * a
            elif v == "turb_fnu_mean":
                x = np.exp(1.5 + 0.8 * a)
            else:
                x = np.exp(4 + 0.5 * season + 0.6 * a)
            x[rs.random(T) < miss] = np.nan
            y[i, :, j] = x
    sites = [f"S{i:02d}" for i in range(S)]
    cube = Cube(sites, dates, list(variables), y, np.zeros_like(y, bool))
    cm, cs = climatology(cube)
    roles = np.array(roles or (["train"] * (S - 2) + ["val"] * 2))
    split = SondeSplit({s: r for s, r in zip(sites, roles)}, {s: f"g{s}" for s in sites})
    static = pd.DataFrame({"lat": np.linspace(35, 45, S), "lon": np.linspace(-95, -80, S), "huc2": "07",
                           "elev_m": 100.0, "log_drain_area_km2": 2.0}, index=sites)
    return common.SondeData(cube, cm, cs, roles, static, split, {"temporary": True, "final": False},
                            {}, ("train", "val"), common.HORIZONS)


@pytest.fixture(scope="module")
def data():
    return synth_data()


# ----------------------------------------------------------------------------- guard

def test_guard_requires_flag_and_freeze_file(tmp_path):
    assert guard.check_access(("train", "val"), ("train", "val"))["protected"] == []
    for blocks, roles in ((("test",), ()), (("test_b",), ()), (("val",), ("test",))):
        with pytest.raises(guard.ProtectedDataError):
            guard.check_access(blocks, roles)
        with pytest.raises(guard.ProtectedDataError):  # flag but no freeze file
            guard.check_access(blocks, roles, allow_test=True, freeze_file=tmp_path / "FREEZE_sonde.json")
    bad = tmp_path / "bad.json"
    bad.write_text("not json")
    with pytest.raises(guard.ProtectedDataError):
        guard.check_access(("test",), (), allow_test=True, freeze_file=bad)
    ff = tmp_path / "FREEZE_sonde.json"
    sp = SondeSplit({"a": "train"}, {"a": "g"})
    good = {"created": "2026-11-15", "split_sha256": sp.sha256(), "scoring_code_sha256": guard.scoring_code_sha256(),
            "weights_sha256": {"gbm": "a" * 64}, "thresholds_sha256": "b" * 64}
    ff.write_text(json.dumps({"anything": 1}))           # review probe P4: arbitrary JSON must not unlock
    with pytest.raises(guard.ProtectedDataError):
        guard.check_access(("test",), (), allow_test=True, freeze_file=ff)
    for k in guard.FREEZE_KEYS:                           # every required key is enforced
        ff.write_text(json.dumps({kk: v for kk, v in good.items() if kk != k}))
        with pytest.raises(guard.ProtectedDataError):
            guard.check_access(("test",), (), allow_test=True, freeze_file=ff, split=sp)
    ff.write_text(json.dumps(good))
    rec = guard.check_access(("test",), ("test",), allow_test=True, freeze_file=ff, split=sp)
    assert rec["freeze_sha256"] and rec["protected"] == ["test", "role:test"]
    with pytest.raises(guard.ProtectedDataError):  # freeze file names another split
        guard.check_access(("test",), (), allow_test=True, freeze_file=ff,
                           split=SondeSplit({"b": "train"}, {"b": "g"}))
    ff.write_text(json.dumps({**good, "scoring_code_sha256": "0" * 64}))
    with pytest.raises(guard.ProtectedDataError):  # scoring code changed after the freeze
        guard.check_access(("test",), (), allow_test=True, freeze_file=ff, split=sp)
    ck = tmp_path / "ck.bin"
    ck.write_bytes(b"weights")
    fr = {**good, "weights_sha256": {"gbm": guard.artifact_sha256(ck)}}
    assert guard.verify_weights(fr, "gbm", ck)
    ck.write_bytes(b"other weights")
    with pytest.raises(guard.ProtectedDataError):
        guard.verify_weights(fr, "gbm", ck)
    with pytest.raises(guard.ProtectedDataError):
        guard.verify_thresholds(good, ck)


def test_guard_load_range_and_prepare_refuses_test_sites():
    assert guard.data_end_date(("train", "val")) == TIME_BLOCKS["val"][1]
    assert guard.data_end_date(("train",)) == TIME_BLOCKS["train"][1]
    with pytest.raises(guard.ProtectedDataError):
        guard.assert_cube_within(pd.date_range("2023-09-01", "2023-10-02"), ("train", "val"))
    sp = SondeSplit({"a": "train", "b": "test"}, {"a": "g1", "b": "g2"})
    with pytest.raises(guard.ProtectedDataError):  # raised before any file is read
        common.prepare_data(sp, {}, ["a", "b"], blocks=("train", "val"))
    with pytest.raises(guard.ProtectedDataError):
        common.prepare_data(sp, {}, ["a"], blocks=("train", "val", "test"))


def test_dataset_refuses_unauthorised_block(data):
    with pytest.raises(guard.ProtectedDataError):
        data.issue_idx("test", 1)


# ----------------------------------------------------------------------------- splits

def test_split_roles_disjoint_and_nonfinal_not_registrable(tmp_path):
    hucs = [f"{h:04d}{k:04d}" for h in range(1, 31) for k in range(3)]
    sites = pd.DataFrame({"site_no": [f"{i:08d}" for i in range(len(hucs))], "huc_cd": hucs})
    sp = make_split(sites, None, seed=1)
    assert_disjoint(train=sp.sites("train"), val=sp.sites("val"), test=sp.sites("test"))
    g = pd.DataFrame({"g": pd.Series(sp.basin_group), "r": pd.Series(sp.site_role)})
    assert (g.groupby("g")["r"].nunique() == 1).all()
    p = tmp_path / "sonde_split_dev.json"
    p.write_text(sp.to_json())
    sp2, info = common.load_split(p)
    assert sp2.sha256() == sp.sha256() and info["temporary"] and not info["final"]
    from sentinel.models.sonde.runner import assert_registrable

    with pytest.raises(RuntimeError):
        assert_registrable({"_smoke": False}, info)
    with pytest.raises(RuntimeError):
        assert_registrable({"_smoke": True}, {"temporary": False, "final": True})


def test_models_never_train_on_val_or_test_sites(data):
    tr = data.site_idx(("train",))
    assert set(data.roles[tr]) == {"train"}
    gbm = SondeGBM(GBMConfig(max_rounds=5), (1,))
    si, ti = gbm.training_rows(data, "temp_c_mean", 1, tr)
    assert set(data.roles[si]) == {"train"}
    assert data.cube.dates[ti].max() + pd.Timedelta(days=1) <= pd.Timestamp(TIME_BLOCKS["train"][1])


# ----------------------------------------------------------------------------- leakage: features

def test_features_causal(data):
    """Perturbing every value after issue date t0 must not change any feature at t0."""
    t0 = int(np.searchsorted(data.cube.dates, pd.Timestamp("2022-06-15")))
    si = np.arange(len(data.sites))
    ti = np.full(si.size, t0)
    fb = FeatureBuilder(data.cube, data.clim_mean, data.clim_std, data.static)
    X = fb.table("do_mgl_min", si, ti, 7, threshold_model=3.0)
    y2 = data.cube.y.copy()
    y2[:, t0 + 1:] = np.random.default_rng(5).normal(50, 20, size=y2[:, t0 + 1:].shape)
    cube2 = Cube(data.cube.sites, data.cube.dates, data.cube.variables, y2, data.cube.prov)
    fb2 = FeatureBuilder(cube2, data.clim_mean, data.clim_std, data.static)
    X2 = fb2.table("do_mgl_min", si, ti, 7, threshold_model=3.0)
    pd.testing.assert_frame_equal(X, X2)
    # ... while perturbing the issue day itself does change them (the test has power)
    y3 = data.cube.y.copy()
    y3[:, t0] += 5
    fb3 = FeatureBuilder(Cube(data.cube.sites, data.cube.dates, data.cube.variables, y3, data.cube.prov),
                         data.clim_mean, data.clim_std, data.static)
    assert not X.equals(fb3.table("do_mgl_min", si, ti, 7, threshold_model=3.0))


def test_rolling_helpers_match_pandas():
    from sentinel.models.sonde import features as F

    rs = np.random.default_rng(0)
    z = rs.normal(size=(3, 60))
    z[rs.random(z.shape) < 0.2] = np.nan
    ref = pd.DataFrame(z.T).rolling(7, min_periods=2).mean().to_numpy().T
    np.testing.assert_allclose(F.rolling_mean(z, 7, 2), ref, equal_nan=True)
    ref_sd = pd.DataFrame(z.T).rolling(7, min_periods=3).std().to_numpy().T
    np.testing.assert_allclose(F.rolling_std(z, 7, 3), ref_sd, equal_nan=True, atol=1e-10)
    ref_min = pd.DataFrame(z.T).rolling(7, min_periods=1).min().to_numpy().T
    np.testing.assert_allclose(F.rolling_extreme(z, 7, "min"), ref_min, equal_nan=True)
    x = np.arange(10, dtype=float)[None, :] * 2.0 + 1
    assert np.allclose(F.rolling_slope(x, 7)[0, 6:], 2.0)


# ----------------------------------------------------------------------------- leakage: GBM slices

def test_gbm_residual_slice_disjoint_from_fit_rows(data):
    cfg = GBMConfig()
    d = data.cube.dates
    ti = np.arange(len(d) - 20)
    for h in (1, 7, 14):
        fit, cal = fit_cal_masks(d, ti, h, cfg)
        assert not (fit & cal).any()
        assert (d[ti[fit]] + pd.Timedelta(days=h)).max() <= pd.Timestamp(cfg.fit_end)
        assert d[ti[cal]].min() >= pd.Timestamp(cfg.cal_start) + pd.Timedelta(days=cfg.cal_embargo_days)
        assert (d[ti[cal]] + pd.Timedelta(days=h)).max() <= pd.Timestamp(TIME_BLOCKS["train"][1])


def test_gbm_tiny_fit_predict(data):
    gbm = SondeGBM(GBMConfig(max_rounds=20, num_leaves=7, min_data_in_leaf=20, num_threads=2), (1, 3))
    tr = data.site_idx(("train",))
    d = gbm.fit_var(data, "temp_c_mean", train_sites=tr, es_rows=None, log=lambda *_: None)
    for h, rec in d.items():
        assert pd.Timestamp(rec["fit_issue_max"]) + pd.Timedelta(days=h) <= pd.Timestamp("2019-09-30")
        assert pd.Timestamp(rec["cal_issue_min"]) >= pd.Timestamp("2019-11-12")
    rows = common.common_rows(data, "temp_c_mean", "val")
    q = gbm.predict_rows(data, "temp_c_mean", rows)
    assert q.shape == (len(rows), 2, len(QUANTILES)) and np.isfinite(q).all()
    assert (np.diff(q, axis=-1) >= -1e-9).all()


# ----------------------------------------------------------------------------- common rows / scoring

def test_common_rows_match_baseline_scorable(data):
    v = "do_mgl_min"
    rows = common.common_rows(data, v, "val")
    b = SondeBaselines(data.cube, data.clim_mean, data.clim_std, v, data.horizons)
    for h in (1, 7):
        t_h = data.issue_idx("val", h)
        m = b.scorable(t_h, h)
        want = {(s, int(t)) for s, k in zip(*np.nonzero(m)) for t in [t_h[k]]}
        got = {(int(s), int(t)) for s, t in zip(rows.si[rows.score_mask[h]], rows.ti[rows.score_mask[h]])}
        assert got == want
    assert rows.issue_date.min() >= np.datetime64(pd.Timestamp(TIME_BLOCKS["val"][0]) + pd.Timedelta(days=42))


def test_score_rows_identical_to_score_samples():
    rs = np.random.default_rng(1)
    S, n = 3, 40
    q = np.sort(rs.normal(size=(S, n, len(QUANTILES))), axis=-1)
    obs = rs.normal(size=(S, n))
    mask = rs.random((S, n)) < 0.7
    dates = pd.date_range("2022-01-01", periods=n)
    t_idx = np.arange(n)
    a = score_samples("temp_c_mean", q, obs, mask, ["a", "b", "c"], dates, t_idx, 3, "m")
    si, ti = np.nonzero(mask)
    b = score_rows("temp_c_mean", q[si, ti], obs[si, ti], np.array(["a", "b", "c"])[si], dates[ti], 3, "m",
                   clip=False)
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)


def test_forecast_file_roundtrip(tmp_path, data):
    rows = common.common_rows(data, "temp_c_mean", "val")
    q = common.baseline_forecast(data, "temp_c_mean", rows, "damped_anomaly")
    p = common.save_forecast(tmp_path / "f.npz", model="damped_anomaly", rows=rows, q=q)
    fc = common.load_forecast(p)
    np.testing.assert_array_equal(common.align_forecast(fc, rows), q)


# ----------------------------------------------------------------------------- normalisation

def test_normalisation_uses_training_years_only(data):
    from sentinel.models.sonde.lstm import SondeTensors

    t_val = data.cube.dates > pd.Timestamp(TIME_BLOCKS["train"][1])
    y2 = data.cube.y.copy()
    y2[:, t_val] = y2[:, t_val] * 3 + 100
    cube2 = Cube(data.cube.sites, data.cube.dates, data.cube.variables, y2, data.cube.prov)
    cm2, cs2 = climatology(cube2)
    np.testing.assert_array_equal(cm2, data.clim_mean)
    np.testing.assert_array_equal(cs2, data.clim_std)
    d2 = common.SondeData(cube2, cm2, cs2, data.roles, data.static, data.split, data.split_info, {},
                          data.blocks, data.horizons)
    a, b = SondeTensors(data, variables=VARS), SondeTensors(d2, variables=VARS)
    np.testing.assert_array_equal(a.m_site, b.m_site)
    np.testing.assert_array_equal(a.s_site, b.s_site)
    np.testing.assert_array_equal(a.static, b.static)
    assert a.frozen_stats() == b.frozen_stats()


# ----------------------------------------------------------------------------- LSTM (tiny, CPU)

def test_lstm_tiny_monotone_and_causal(data):
    torch = pytest.importorskip("torch")
    from sentinel.models.sonde.lstm import LSTMConfig, SondeLSTM, SondeTensors

    cfg = LSTMConfig(context=30, hidden=8, static_hidden=4, head_hidden=8, batch_size=16, max_epochs=1,
                     steps_per_epoch=3, patience=1, amp=False, val_max_samples=64)
    m = SondeLSTM(cfg, data.horizons, list(VARS))
    info = m.fit(data, device="cpu", log=lambda *_: None)
    assert np.isfinite(info["best_val_loss"])
    rows = {v: common.common_rows(data, v, "val") for v in VARS}
    fc = m.predict_rows(rows)
    for v, q in fc.items():
        assert q.shape == (len(rows[v]), len(data.horizons), len(QUANTILES)) and np.isfinite(q).all()
        assert (np.diff(q, axis=-1) >= -1e-5).all()
    # input truncation: changing data after the issue date leaves the forecast unchanged
    t0 = int(np.searchsorted(data.cube.dates, pd.Timestamp("2022-07-01")))
    si, ti = np.arange(len(data.sites)), np.full(len(data.sites), t0)
    base = m._forward_np(si, ti)
    y2 = data.cube.y.copy()
    y2[:, t0 + 1:] = 99.0
    d2 = common.SondeData(Cube(data.cube.sites, data.cube.dates, data.cube.variables, y2, data.cube.prov),
                          data.clim_mean, data.clim_std, data.roles, data.static, data.split, data.split_info,
                          {}, data.blocks, data.horizons)
    m.tensors = SondeTensors(d2, variables=m.variables, horizons=m.horizons, context=cfg.context,
                             frozen=m.tensors.frozen_stats())
    m._to_device(m.tensors)
    np.testing.assert_allclose(m._forward_np(si, ti), base, rtol=1e-5, atol=1e-6)
    y2[:, t0] += 3.0  # ...but the issue-day value does matter
    m.tensors = SondeTensors(d2, variables=m.variables, horizons=m.horizons, context=cfg.context,
                             frozen=m.tensors.frozen_stats())
    m._to_device(m.tensors)
    assert not np.allclose(m._forward_np(si, ti), base)
    del torch


def test_pinball_loss_masks_missing():
    torch = pytest.importorskip("torch")
    from sentinel.models.sonde.lstm import pinball_loss

    taus = torch.tensor(QUANTILES, dtype=torch.float32)
    q = torch.zeros(2, 2, 1, len(QUANTILES))
    y = torch.tensor([[[1.0], [float("nan")]], [[1.0], [float("nan")]]])
    loss = pinball_loss(q, y, taus, balance=True)
    expect = float(np.mean(np.maximum(QUANTILES * 1.0, (QUANTILES - 1) * 1.0)))
    assert abs(float(loss) - expect) < 1e-6


# ----------------------------------------------------------------------------- Chronos-2 (tiny random model)

def _tiny_chronos():
    pytest.importorskip("chronos")
    from chronos import Chronos2Pipeline
    from chronos.chronos2 import Chronos2Model
    from chronos.chronos2.config import Chronos2CoreConfig

    levels = [0.01] + [round(x, 2) for x in np.arange(0.05, 0.951, 0.05)] + [0.99]
    cfg = Chronos2CoreConfig(d_model=16, d_kv=8, d_ff=32, num_layers=1, num_heads=2, chronos_config={
        "context_length": 64, "input_patch_size": 16, "input_patch_stride": 16, "output_patch_size": 16,
        "max_output_patches": 1, "quantiles": levels, "use_reg_token": True, "use_arcsinh": True})
    import torch

    torch.manual_seed(0)
    return Chronos2Pipeline(Chronos2Model(cfg).eval())


def test_chronos_windows_are_causal():
    from sentinel.models.sonde.chronos import padded_series, windows

    z = np.arange(20, dtype=float)[None, :].repeat(2, 0)
    w = windows(padded_series(z, 8), np.array([0, 1]), np.array([3, 10]), 8)
    assert np.isnan(w[0, :4]).all() and w[0, -1] == 3 and w[1, -1] == 10 and np.nanmax(w[1]) == 10


def test_chronos_tiny_uni_and_multi(data):
    from sentinel.models.sonde.chronos import ChronosConfig, ChronosForecaster

    pipe = _tiny_chronos()
    rows = {v: common.common_rows(data, v, "val") for v in ("temp_c_mean", "do_mgl_min")}
    rows = {v: _subset(r, 30) for v, r in rows.items()}
    for mode in ("uni", "multi"):
        f = ChronosForecaster(pipe, ChronosConfig(mode=mode, context=32, batch_size=64, chunk_rows=40), data.horizons)
        out = f.predict(data, rows)
        for v, q in out.items():
            assert q.shape == (len(rows[v]), len(data.horizons), len(QUANTILES)) and np.isfinite(q).all()
            assert (np.diff(q, axis=-1) >= 0).all()


def _subset(r: common.Rows, n: int) -> common.Rows:
    k = slice(0, n)
    return common.Rows(r.var, r.block, r.si[k], r.ti[k], r.site_no[k], r.issue_date[k], r.role[k],
                       {h: m[k] for h, m in r.score_mask.items()}, {h: o[k] for h, o in r.obs.items()})


def test_chronos_ft_inputs_stay_in_train_block(data):
    from sentinel.models.sonde.chronos_ft import ChronosFTConfig, build_train_inputs, build_val_inputs

    cfg = ChronosFTConfig(context=64, min_obs=50, val_cut_every_days=60, val_max_sites=None)
    tr, info = build_train_inputs(data, cfg)
    end = int(np.searchsorted(data.cube.dates, pd.Timestamp(TIME_BLOCKS["train"][1]), side="right"))
    assert info["last_train_date"] <= TIME_BLOCKS["train"][1] and len(tr) == len(data.site_idx(("train",)))
    assert all(x["context"].shape[1] <= end for x in tr)
    va, vinfo = build_val_inputs(data, cfg)
    assert len(va) > 0 and pd.Timestamp(vinfo["first_issue"]) >= pd.Timestamp(TIME_BLOCKS["val"][0]) + pd.Timedelta(days=42)


# ----------------------------------------------------------------------------- events

def test_exceedance_prob_matches_gaussian_cdf():
    from scipy.stats import norm

    q = norm.ppf(QUANTILES)[None, :].repeat(5, 0)
    thr = [-1.0, -0.3, 0.0, 0.7, 1.2]
    for k, t in enumerate(thr):
        p = E.exceedance_prob(q[k:k + 1], t, "below")[0]
        assert abs(p - norm.cdf(t)) < 0.012
        assert abs(E.exceedance_prob(q[k:k + 1], t, "above")[0] - (1 - norm.cdf(t))) < 0.012
    tail = E.exceedance_prob(q[:1], -4.0, "below")[0]
    assert 0.0 <= tail < 0.05
    grid = np.linspace(-3, 3, 50)
    ps = [E.exceedance_prob(q[:1], g, "below")[0] for g in grid]
    assert (np.diff(ps) >= -1e-12).all()


def test_onset_prob_combinations_ordered():
    rs = np.random.default_rng(2)
    q = np.sort(rs.normal(4.0, 1.5, size=(50, len((1, 2, 3, 5, 7, 10, 14)), len(QUANTILES))), axis=-1)
    p_max = E.onset_prob_from_quantiles(q, (1, 2, 3, 5, 7, 10, 14), 7, "H1_hypoxia", combine="max")
    p_ind = E.onset_prob_from_quantiles(q, (1, 2, 3, 5, 7, 10, 14), 7, "H1_hypoxia", combine="indep")
    assert ((p_ind >= p_max - 1e-12) & (p_max >= 0) & (p_ind <= 1)).all()


def test_event_rows_and_labels_respect_blocks(data):
    lab = E.hazard_labels(data, "H1_hypoxia", (1, 3))
    tr = E.event_rows(data, lab, 3, "train", roles=("train",))
    va = E.event_rows(data, lab, 3, "val")
    assert tr.y.sum() > 0
    assert set(data.roles[tr.si]) == {"train"}
    assert (data.cube.dates[tr.ti] + pd.Timedelta(days=3)).max() <= pd.Timestamp(TIME_BLOCKS["train"][1])
    assert data.cube.dates[va.ti].min() >= pd.Timestamp(TIME_BLOCKS["val"][0]) + pd.Timedelta(days=42)
    assert not lab.event[va.si, va.ti].any()  # issue days inside an event are never scored


def test_harmonic_climatology_seasonal_and_pooled():
    rs = np.random.default_rng(3)
    doy = rs.integers(0, 365, 20000)
    p_true = np.where((doy > 150) & (doy < 250), 0.05, 0.002)
    y = (rs.random(doy.size) < p_true).astype(int)
    site = np.where(rs.random(doy.size) < 0.5, "a", "b")
    y_b = np.zeros(int((site == "b").sum()), int)  # site b never has onsets
    y2 = y.copy()
    y2[site == "b"] = y_b
    clim = E.HarmonicClimatology(K=2, ridge=5.0).fit(doy, y2, {"a": (doy[site == "a"], y2[site == "a"]),
                                                              "b": (doy[site == "b"], y2[site == "b"])})
    pa = clim.predict(np.array(["a", "a"]), np.array([200, 20]))
    pb = clim.predict(np.array(["b"]), np.array([200]))
    assert pa[0] > 3 * pa[1]
    assert 0 < pb[0] < pa[0]


def test_state_persistence_and_platt_out_of_fold():
    rs = np.random.default_rng(4)
    m = rs.exponential(2.0, 5000)
    y = (rs.random(5000) < 0.3 * np.exp(-m)).astype(int)
    sp = E.StatePersistence(10).fit(m, y)
    assert sp.predict(np.array([0.05]))[0] > sp.predict(np.array([6.0]))[0]
    p = np.clip(rs.random(400), 0.01, 0.99)
    yy = (rs.random(400) < p).astype(int)
    fold = np.repeat([2022, 2023], 200)
    out = E.crossfit_platt(p, yy, fold)
    yy2 = yy.copy()
    yy2[fold == 2022] = 1 - yy2[fold == 2022]       # change fold-2022 labels only
    out2 = E.crossfit_platt(p, yy2, fold)
    np.testing.assert_allclose(out[fold == 2022], out2[fold == 2022])  # fold-2022 output uses only 2023 data


def test_episode_rules_and_budget_calibration():
    assert EWS.alarm_episodes(np.array([0, 1, 1, 0, 1, 0, 0, 1], bool)) == [(1, 2), (4, 4), (7, 7)]
    idx = pd.date_range("2022-01-01", periods=10)
    r = EWS.episode_outcomes(idx, np.array([0, 1, 1, 0, 0, 0, 1, 0, 0, 0], bool), pd.DatetimeIndex(["2022-01-04"]), 3)
    assert r["n_hit_episodes"] == 1 and r["n_false_episodes"] == 1 and r["n_onsets_warned"] == 1
    rs = np.random.default_rng(6)
    frames, onsets = {}, {}
    for s in ("a", "b"):
        d = pd.date_range("2022-01-01", periods=300)
        frames[s] = pd.DataFrame({"score": rs.random(300), "usable": rs.random(300) < 0.9}, index=d)
        onsets[s] = pd.DatetimeIndex([d[100], d[200]])
    cal = EWS.calibrate_episode_threshold(frames, onsets, 3, budget_per_30d=1.0)
    rate, _ = EWS.false_alarm_rate(frames, onsets, cal["threshold"], 3)
    assert rate <= 1.0 and cal["fa_per_site_30d"] == rate
    bss = EWS.brier_skill(np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.2, 0.8]), np.full(4, 0.5))
    assert bss["bss"] > 0


def test_load_cube_waits_or_skips_missing(tmp_path, monkeypatch):
    from sentinel.models.sonde import data as D

    monkeypatch.setattr(D, "SITE_DIR", tmp_path)
    d = pd.date_range("2008-01-01", periods=5)
    pd.DataFrame({"temp_c_mean": np.arange(5, dtype="float32"), "temp_c_mean_prov": False},
                 index=pd.Index(d, name="date")).to_parquet(tmp_path / "A.parquet")
    c = D.load_cube(["A", "B"], ["temp_c_mean"], start="2008-01-01", end="2008-01-05")
    assert np.isfinite(c.y[0, :, 0]).all() and np.isnan(c.y[1]).all()
    c2 = D.load_cube(["A", "B"], ["temp_c_mean"], start="2008-01-01", end="2008-01-05", retries=1, wait_s=0.01)
    np.testing.assert_array_equal(c.y, c2.y)
    assert to_model_space("turb_fnu_median", np.array([0.0]))[0] == pytest.approx(np.log(0.1))


def test_vectorised_episodes_match_reference_and_chunking_penalises_standing_alarms():
    rs = np.random.default_rng(7)
    frames, onsets = {}, {}
    for s in "abc":
        d = pd.date_range("2022-01-01", periods=150)
        sc = rs.random(150)
        sc[rs.random(150) < 0.1] = np.nan
        frames[s] = pd.DataFrame({"score": sc, "usable": rs.random(150) < 0.9}, index=d)
        onsets[s] = pd.DatetimeIndex(sorted(rs.choice(d, 3, replace=False)))
    for h in (1, 3, 7):
        for ml in (None, h):
            for thr in (0.0, 0.5, 0.95):
                st = EWS.StackedFrames(frames, onsets, h).stats(thr, ml)
                ref = dict.fromkeys(("n_episodes", "n_hit_episodes", "n_false_episodes", "n_onsets",
                                     "n_onsets_warned"), 0)
                for s, f in frames.items():
                    use = f["usable"].to_numpy() & np.isfinite(f["score"].to_numpy())
                    r = EWS.episode_outcomes(f.index, use & (f["score"].to_numpy() >= thr), onsets[s], h, ml)
                    for k in ref:
                        ref[k] += r[k]
                assert all(st[k] == ref[k] for k in ref)
    # a standing alarm (always on, no gaps) is one "episode" as maximal runs, many when chunked
    d = pd.date_range("2022-01-01", periods=300)
    f = {"x": pd.DataFrame({"score": np.ones(300), "usable": np.ones(300, bool)}, index=d)}
    o = {"x": pd.DatetimeIndex([])}
    assert EWS.StackedFrames(f, o, 3).stats(0.5)["fa_per_site_30d"] == pytest.approx(0.1)
    assert EWS.StackedFrames(f, o, 3).stats(0.5, 3)["fa_per_site_30d"] == pytest.approx(10.0)


def test_lstm_last_anomaly_is_latest_recent_observation():
    torch = pytest.importorskip("torch")
    from sentinel.models.sonde.lstm import LSTMConfig, _build_model

    V = 2
    m = _build_model(3 * V + 2, 3, 24, V, 2, len(QUANTILES), LSTMConfig(hidden=4, static_hidden=2, head_hidden=4), (1, 2))
    x = torch.zeros(1, 6, 3 * V + 2)
    x[0, :, 0] = torch.arange(6.0)            # var 0 anomalies 0..5
    x[0, [1, 3], 2 * V] = 1.0                 # var 0 observed on days 1 and 3 only (day 3 = 2 days before issue)
    x[0, 0, 1] = 9.0
    x[0, 0, 2 * V + 1] = 1.0                  # var 1 observed only 5 days before issue -> too stale
    a = m.last_anomaly(x)
    assert float(a[0, 0]) == 3.0 and float(a[0, 1]) == 0.0



def test_quantile_clipping_to_plausible_range():
    from sentinel.models.sonde.evaluate import clip_plausible, plausible_bounds

    assert plausible_bounds("turb_fnu_mean") == (0.0, 10000.0)      # CLIP_ZERO lower bound
    assert plausible_bounds("temp_c_mean") == (-1.0, 40.0)
    q = np.array([[to_model_space("turb_fnu_mean", np.array([1e8]))[0]] * 19])
    c = clip_plausible("turb_fnu_mean", q)
    assert np.allclose(c, to_model_space("turb_fnu_mean", np.array([1e4]))[0])
    r = score_rows("turb_fnu_mean", q, np.array([to_model_space("turb_fnu_mean", np.array([10.0]))[0]]),
                   ["a"], pd.DatetimeIndex(["2022-01-01"]), 7, "m")
    assert r["q95"].iloc[0] == pytest.approx(1e4, rel=1e-4)
    d = synth_data()
    rows = common.common_rows(d, "turb_fnu_mean", "val")
    qq = common.baseline_forecast(d, "turb_fnu_mean", rows, "climatology") + 50.0     # absurd forecast
    with np.load(common.save_forecast(Path_tmp() / "c.npz", model="x", rows=rows, q=qq)) as z:
        assert z["q"].max() <= to_model_space("turb_fnu_mean", np.array([1e4]))[0] + 1e-5


def Path_tmp():
    import tempfile
    from pathlib import Path

    return Path(tempfile.mkdtemp())


def test_provisional_targets_are_not_scored():
    d = synth_data()
    j = d.cube.var_index("temp_c_mean")
    prov = d.cube.prov.copy()
    vmask = d.cube.dates > pd.Timestamp("2022-06-01")
    prov[:, vmask, j] = True
    d2 = common.SondeData(Cube(d.cube.sites, d.cube.dates, d.cube.variables, d.cube.y, prov), d.clim_mean,
                          d.clim_std, d.roles, d.static, d.split, d.split_info, {}, d.blocks, d.horizons)
    r1, r2 = common.common_rows(d, "temp_c_mean", "val"), common.common_rows(d2, "temp_c_mean", "val")
    for h in (1, 7):
        tgt = d.cube.dates[np.minimum(r2.ti + h, len(d.cube.dates) - 1)]
        assert not (r2.score_mask[h] & (tgt > pd.Timestamp("2022-06-01"))).any()
        assert r2.n_dropped_provisional[h] > 0 and r1.n_dropped_provisional[h] == 0
        assert (r2.score_mask[h] <= r1.score_mask[h]).all()


def test_alarm_day_budget_rejects_standing_alarm_at_every_lead():
    """C1: a 7-in-30 standing alarm (no events) breaks the 1/30 alarm-day budget for L = 1, 3, 7."""
    from sentinel.eval.hazards import daily_onsets, issue_labels

    hz = E.HAZARDS["H1_hypoxia"]
    n_sites, n_days = 30, 300
    for L in (1, 3, 7):
        ys, vs, ss, sites = [], [], [], []
        for k in range(n_sites):
            x = np.full(n_days, 8.0)
            ev, on = daily_onsets(x, hz)
            y, v = issue_labels(x, ev, on, L, refractory_days=hz.refractory_days)
            sc = ((np.arange(n_days) % 30) < 7).astype(float)
            ys.append(y); vs.append(v); ss.append(sc); sites.append(np.full(n_days, k))
        y, v, sc, site = map(np.concatenate, (ys, vs, ss, sites))
        st = EWS.alarm_day_stats(sc, y, v, site, threshold=0.5, budget_per_30d=1.0)
        assert not st["meets_budget"] and st["pooled_alarm_days_per_30d"] == pytest.approx(7.0, abs=0.2)
        thr = EWS.alarm_day_threshold(sc, y, v, budget_per_30d=1.0)
        st2 = EWS.alarm_day_stats(sc, y, v, site, thr, 1.0)
        assert st2["meets_budget"] and st2["time_in_alarm"] == 0.0     # calibration forbids the alarm


def test_alarm_day_threshold_meets_budget_on_random_scores():
    rs = np.random.default_rng(8)
    sc, y = rs.random(9000), (rs.random(9000) < 0.01).astype(int)
    valid, site = rs.random(9000) < 0.95, np.repeat(np.arange(30), 300)
    for b in (1.0, 2.0):
        thr = EWS.alarm_day_threshold(sc, y, valid, b)
        st = EWS.alarm_day_stats(sc, y, valid, site, thr, b)
        assert st["meets_budget"] and st["pooled_alarm_days_per_30d"] > 0.7 * b


def test_clean_placebos_identical_when_unrestricted_and_clean_otherwise():
    from sentinel.eval import early_warning as EWm

    d = pd.date_range("2022-01-01", periods=400)
    usable = np.ones(400, bool)
    evs = [EWm.Event("s", d[100], d[103], d[100], 1.0, 96.0, False)]
    a = EWm.sample_placebo_onsets(d, usable, evs, n=20, horizon="3D", seed=3)
    b = EWS.sample_clean_placebos(d, usable, evs, np.ones(400, bool), n=20, horizon="3D", seed=3)
    assert a.equals(b)
    cand = np.zeros(400, bool); cand[200:260] = True
    c = EWS.sample_clean_placebos(d, usable, evs, cand, n=20, horizon="3D", seed=3)
    assert len(c) and ((c >= d[200]) & (c <= d[259])).all()


def test_placebo_candidates_exclude_event_and_unclean_days(data):
    lab = E.hazard_labels(data, "H1_hypoxia", (3,))
    cand = E.placebo_candidates(data, lab)
    assert not (cand & lab.event).any()
    hz = E.HAZARDS["H1_hypoxia"]
    x = data.cube.y[:, :, data.cube.var_index("do_mgl_min")]
    s, t = np.nonzero(cand)
    for k in range(1, hz.pre_days + 1):
        assert (x[s, t - k] >= hz.threshold + hz.pre_margin).all()



def test_lstm_static_standardisation_bounded_for_single_site_variables():
    from sentinel.models.sonde.lstm import SondeTensors

    d = synth_data()
    j = d.cube.var_index("turb_fnu_mean")
    y = d.cube.y.copy()
    tr = d.site_idx(("train",))
    y[tr[1:], :, j] = np.nan          # only ONE train-role site has the variable
    c = Cube(d.cube.sites, d.cube.dates, d.cube.variables, y, d.cube.prov)
    cm, cs = climatology(c)
    d2 = common.SondeData(c, cm, cs, d.roles, d.static, d.split, d.split_info, {}, d.blocks, d.horizons)
    t = SondeTensors(d2, variables=VARS)
    assert np.abs(t.static).max() <= 10 and np.isfinite(t.static).all()


def test_training_prepare_refuses_protected_blocks_and_test_sites(tmp_path):
    import argparse

    from sentinel.models.sonde import runner

    for blocks, inc in (("val,test", False), ("test_b", False), ("val", True)):
        a = argparse.Namespace(eval_blocks=blocks, include_test_sites=inc, split=None, temp_split=False,
                               allow_test=True, freeze_file=None)
        with pytest.raises(guard.ProtectedDataError):
            runner.prepare(a, {"_smoke": True}, tmp_path, training=True)



def test_two_stage_bootstrap_wider_than_block_bootstrap_with_group_effects():
    from sentinel.eval.bootstrap import bootstrap_ci

    rs = np.random.default_rng(9)
    outer = np.repeat(np.arange(6), 400)                  # 6 basin groups
    inner = np.repeat(np.arange(60), 40)                  # 10 site-months per group
    x = rs.normal(size=outer.size) + np.repeat(rs.normal(0, 2, 6), 400)   # strong group effect
    one = bootstrap_ci(np.mean, x, groups=inner, n_boot=400, seed=1)
    two = EWS.two_stage_bootstrap(np.mean, x, outer=outer, inner=inner, n_boot=400, seed=1)
    assert two["n_outer"] == 6 and two["n_inner"] == 60
    assert (two["ci_high"] - two["ci_low"]) > 2 * (one["ci_high"] - one["ci_low"])
    r = EWS.brier_skill(np.array([0, 1] * 50), np.array([0.2, 0.7] * 50), np.full(100, 0.5),
                        site=np.repeat(["a", "b"], 50), dates=pd.date_range("2022-01-01", periods=100),
                        n_boot=50, basin={"a": "g1", "b": "g2"})
    assert r["ci_unit"] == "basin_group > site-month" and r["n_basin_groups"] == 2 and "bss_sm_ci_low" in r



def test_block_sum_bootstrap_matches_row_bootstrap_and_delta_brier():
    rs = np.random.default_rng(11)
    n = 3000
    site = np.repeat([f"s{i}" for i in range(15)], n // 15)
    dates = np.tile(pd.date_range("2022-01-01", periods=n // 15).to_numpy(), 15)
    y = (rs.random(n) < 0.05).astype(float)
    p = np.clip(0.05 + 0.3 * y + rs.normal(0, 0.05, n), 0, 1)
    q = np.full(n, 0.05)
    basin = {f"s{i}": f"g{i % 4}" for i in range(15)}
    r = EWS.brier_skill(y, p, q, site=site, dates=dates, n_boot=500, basin=basin)
    g = EWS.site_month_groups(site, dates)
    bg = np.array([basin[x] for x in site])
    slow = EWS.two_stage_bootstrap(lambda a, b, c: 1 - np.mean((b - a) ** 2) / np.mean((c - a) ** 2), y, p, q,
                                   outer=bg, inner=g, n_boot=500, seed=0)
    assert abs(r["bss_ci_low"] - slow["ci_low"]) < 0.05 and abs(r["bss_ci_high"] - slow["ci_high"]) < 0.05
    d = EWS.delta_brier(y, p, q, site=site, dates=dates, basin=basin, n_boot=300)
    assert d["dbs"] == pytest.approx(np.mean((p - y) ** 2) - np.mean((q - y) ** 2))
    assert d["dbs_ci_low"] <= d["dbs"] <= d["dbs_ci_high"] and d["dbs_ci_high"] < 0   # p clearly better


def test_diebold_mariano_detects_difference_and_null():
    from sentinel.models.sonde.evaluate import diebold_mariano

    rs = np.random.default_rng(12)
    e = rs.normal(size=700)
    ar = np.convolve(e, np.ones(5) / 5, mode="same")            # serially correlated loss differential
    assert diebold_mariano(ar - 0.3, 7)["p_value"] < 1e-3
    null = [diebold_mariano(np.convolve(rs.normal(size=700), np.ones(5) / 5, mode="same"), 7)["p_value"]
            for _ in range(200)]
    assert 0.01 < np.mean(np.array(null) < 0.05) < 0.12           # roughly nominal size under HAC


def test_weather_is_lagged_and_never_future(tmp_path, monkeypatch):
    """load_weather: index t holds day t − 1 (published by the end of day t); no same-day value."""
    from sentinel.models.sonde import common as C

    d = pd.date_range("2008-01-01", periods=10)
    rows = pd.DataFrame({"site_key": "USGS-A", "date": d.date, **{v: np.arange(10, dtype="float32") + 100 * k
                                                                  for k, v in enumerate(C.WEATHER_VARS)}})
    f = tmp_path / "f.parquet"
    rows.to_parquet(f)
    monkeypatch.setattr(C, "FORCINGS", f)
    cube = Cube(["A", "B"], d, ["temp_c_mean"], np.zeros((2, 10, 1), "float32"), np.zeros((2, 10, 1), bool))
    w = C.load_weather(cube)
    assert np.isnan(w[0, 0, 0]) and w[0, 1, 0] == 0.0 and w[0, 9, 0] == 8.0   # day t sees day t-1
    assert np.isnan(w[1]).all()


def test_lstm_with_weather_channels_runs_and_stays_causal(data):
    pytest.importorskip("torch")
    from sentinel.models.sonde.lstm import LSTMConfig, SondeLSTM

    rs = np.random.default_rng(3)
    d2 = common.SondeData(data.cube, data.clim_mean, data.clim_std, data.roles, data.static, data.split,
                          data.split_info, {}, data.blocks, data.horizons,
                          weather=rs.normal(size=(len(data.sites), len(data.cube.dates), 6)).astype("float32"))
    m = SondeLSTM(LSTMConfig(context=20, hidden=8, static_hidden=4, head_hidden=8, batch_size=16, max_epochs=1,
                             steps_per_epoch=2, amp=False, val_max_samples=32, use_weather=True), data.horizons, list(VARS))
    m.fit(d2, device="cpu", log=lambda *_: None)
    assert m.tensors.dyn.shape[-1] == 3 * len(VARS) + 2 + 7 and m.tensors.weather_stats is not None


def test_weather_and_virtual_normalisation_use_training_years_only(data):
    from sentinel.models.sonde.lstm import SondeTensors
    from sentinel.models.sonde.virtual import VirtualLSTM

    rs = np.random.default_rng(4)
    w = rs.normal(size=(len(data.sites), len(data.cube.dates), 6)).astype("float32")
    w2 = w.copy()
    late = data.cube.dates > pd.Timestamp(TIME_BLOCKS["train"][1])
    w2[:, late] += 50.0
    mk = lambda ww, y=data.cube.y: common.SondeData(Cube(data.cube.sites, data.cube.dates, data.cube.variables, y,  # noqa: E731
                                                         data.cube.prov), data.clim_mean, data.clim_std, data.roles,
                                                    data.static, data.split, data.split_info, {}, data.blocks,
                                                    data.horizons, weather=ww)
    a = SondeTensors(mk(w), variables=VARS, use_weather=True).weather_stats
    b = SondeTensors(mk(w2), variables=VARS, use_weather=True).weather_stats
    np.testing.assert_allclose(a, b)
    y2 = data.cube.y.copy()
    y2[:, late] = y2[:, late] * 5 + 7
    d1, s1 = VirtualLSTM()._tensors(mk(w), "do_mgl_min")
    d2, s2 = VirtualLSTM()._tensors(mk(w2, y2), "do_mgl_min")
    tr = data.cube.dates <= pd.Timestamp(TIME_BLOCKS["train"][1])
    np.testing.assert_array_equal(d1[:, tr], d2[:, tr])   # training-period inputs unaffected by val-year values


def test_virtual_sensing_withholds_sensor_group_and_respects_roles(data):
    from sentinel.models.sonde.virtual import VirtualGBM, VirtualLSTM, VLSTMConfig, virtual_features, virtual_rows, withheld

    assert withheld("do_mgl_min") == {"do_mgl_mean", "do_mgl_min"}
    fb = FeatureBuilder(data.cube, data.clim_mean, data.clim_std, data.static)
    tr = virtual_rows(data, "do_mgl_min", "train", ("train",))
    es = virtual_rows(data, "do_mgl_min", "val", ("train",))
    ev = virtual_rows(data, "do_mgl_min", "val", ("val",))
    assert set(data.roles[tr.si]) == {"train"} and set(data.roles[ev.si]) == {"val"}
    assert data.cube.dates[tr.ti].max() <= pd.Timestamp(TIME_BLOCKS["train"][1])
    assert data.cube.dates[ev.ti].min() >= pd.Timestamp(TIME_BLOCKS["val"][0]) + pd.Timedelta(days=42)
    X = virtual_features(fb, data, "do_mgl_min", ev.si[:50], ev.ti[:50])
    assert not any(c.startswith("do_mgl") for c in X.columns)
    # perturbing the withheld variable at the evaluated site must not change its features
    y2 = data.cube.y.copy()
    y2[:, :, data.cube.var_index("do_mgl_min")] += 3.0
    c2 = Cube(data.cube.sites, data.cube.dates, data.cube.variables, y2, data.cube.prov)
    fb2 = FeatureBuilder(c2, data.clim_mean, data.clim_std, data.static)
    d2 = common.SondeData(c2, data.clim_mean, data.clim_std, data.roles, data.static, data.split, data.split_info,
                          {}, data.blocks, data.horizons)
    pd.testing.assert_frame_equal(X, virtual_features(fb2, d2, "do_mgl_min", ev.si[:50], ev.ti[:50]))
    g = VirtualGBM(num_threads=2, max_rounds=10)
    g.fit(fb, data, "do_mgl_min", tr, es)
    assert np.isfinite(g.predict(fb, data, "do_mgl_min", ev)).all()
    p, info = VirtualLSTM(VLSTMConfig(context=10, hidden=8, batch_size=32, max_epochs=1, steps_per_epoch=3,
                                      amp=False)).fit_predict(data, "do_mgl_min", tr, es, ev, device="cpu",
                                                             log=lambda *_: None)
    assert p.shape == ev.y.shape and np.isfinite(p).all()


def test_subdaily_local_day_attribution_is_causal():
    from sentinel.models.sonde.subdaily import FEATS, site_daily_from_hourly

    t = pd.date_range("2022-07-01 06:00", periods=48, freq="h", tz="UTC")     # = local 00:00 for UTC-6
    do = np.full(48, 8.0)
    do[3] = 2.0                         # local 03:00 on day 1 -> day-1 night minimum
    do[47] = 1.0                        # local 23:00 on day 2 (ends at local midnight) -> belongs to day 2
    df = pd.DataFrame({"do_mgl_mean": do, "do_mgl_prov": False, "temp_c_mean": 20.0, "temp_c_prov": False},
                      index=pd.DatetimeIndex(t, name="time_utc"))
    dates = pd.date_range("2022-07-01", "2022-07-03")
    f = site_daily_from_hourly(df, -6.0, dates)
    k = FEATS.index("do_night_min")
    assert f[0, k] == 2.0 and f[1, k] == 8.0 and np.isnan(f[2, k])
    assert f[1, FEATS.index("do_amp")] == 7.0 and np.isnan(f[2, FEATS.index("do_amp")])
    df.loc[df.index[5], "do_mgl_prov"] = True; df.loc[df.index[5], "do_mgl_mean"] = 0.0
    assert site_daily_from_hourly(df, -6.0, dates)[0, k] == 2.0          # provisional value ignored


def test_timesfm_decile_conversion_close_to_gaussian():
    from scipy.stats import norm

    from sentinel.models.sonde.timesfm_tsfm import DECILES, deciles_to_19

    q = deciles_to_19(norm.ppf(DECILES)[None, :])[0]
    assert (np.diff(q) > 0).all()
    np.testing.assert_allclose(q, norm.ppf(QUANTILES), atol=1e-9)        # exact for a Gaussian
    sk = deciles_to_19(np.exp(norm.ppf(DECILES))[None, :])[0]            # skewed (log-normal) shape
    assert (np.diff(sk) > 0).all() and sk[0] > 0


def test_chronos_weather_past_covariates_layout_and_causality(data):
    from sentinel.models.sonde.chronos import ChronosConfig, ChronosForecaster

    pipe = _tiny_chronos()
    rs = np.random.default_rng(5)
    d2 = common.SondeData(data.cube, data.clim_mean, data.clim_std, data.roles, data.static, data.split,
                          data.split_info, {}, data.blocks, data.horizons,
                          weather=rs.normal(size=(len(data.sites), len(data.cube.dates), 6)).astype("float32"))
    f = ChronosForecaster(pipe, ChronosConfig(mode="multi", context=32, weather_covariates=True), data.horizons)
    t0 = int(np.searchsorted(data.cube.dates, pd.Timestamp("2022-06-01")))
    inp, tv = f.prepared_inputs(d2, 0, np.array([t0]), list(VARS))
    x = inp[0]
    k = x["n_targets"]
    assert x["n_covariates"] == 6 + k and x["n_future_covariates"] == k
    np.testing.assert_allclose(x["context"][k:k + 6, -1].numpy()[1:], d2.weather[0, t0, 1:], rtol=1e-6)  # ends at t0
    assert np.isnan(x["future_covariates"][:k + 6].numpy()).all()          # weather never in the future
    rows = {v: _subset(common.common_rows(d2, v, "val"), 10) for v in ("temp_c_mean",)}
    out = f.predict(d2, rows)
    assert np.isfinite(out["temp_c_mean"]).all()


def test_chloride_attach_and_quantile_mapping(tmp_path, monkeypatch, data):
    from sentinel.models.sonde import common as C

    q = np.sort(np.random.default_rng(1).normal(6, 0.3, size=(4, 7, len(QUANTILES))), axis=-1)
    a, b = np.full(4, 2.0), np.full(4, 0.25)
    m = E.sc_quantiles_to_chloride(q, a, b)
    np.testing.assert_allclose(m, 2.0 + 0.25 * (np.exp(q) - 1.0))           # exact monotone mapping
    (tmp_path / "sites").mkdir()
    d = pd.DataFrame({"cl_mgl_mean": 100.0, "cl_mgl_4dmean": 100.0, "cl_mgl_max": 120.0},
                     index=pd.Index(data.cube.dates, name="date"))
    d.to_parquet(tmp_path / "sites" / f"{data.sites[0]}.parquet")
    monkeypatch.setattr(C, "CHLORIDE_DIR", tmp_path)
    vars0 = list(VARS) + ["spcond_uscm_mean"]
    y = np.concatenate([data.cube.y, data.cube.y[:, :, :1] * 0 + 300], axis=-1)
    cube = Cube(data.sites, data.cube.dates, vars0, y, np.zeros_like(y, bool))
    cm, cs = climatology(cube)
    d2 = common.SondeData(cube, cm, cs, data.roles, data.static, data.split, data.split_info, {}, data.blocks,
                          data.horizons)
    C.attach_chloride(d2)
    j = d2.cube.var_index("cl_mgl_4dmean")
    assert np.isfinite(d2.cube.y[0, :, j]).all() and np.isnan(d2.cube.y[1, :, j]).all()
    assert d2.clim_mean.shape[-1] == len(d2.cube.variables) and np.isfinite(d2.clim_mean[0, :, j]).all()


def test_chronos_ft_inner_validation_uses_only_heldout_train_sites(data):
    from sentinel.models.sonde.chronos_ft import ChronosFTConfig, build_train_inputs, build_val_inputs, inner_split

    sites = [f"S{i:02d}" for i in range(10)]
    roles = np.array(["train"] * 8 + ["val"] * 2)
    d = synth_data(S=10, roles=list(roles))
    d.split.basin_group.update({s: f"g{i % 4}" for i, s in enumerate(sites)})
    cfg = ChronosFTConfig(context=64, min_obs=50, val_cut_every_days=60, val_max_sites=None, inner_val=True,
                          inner_val_frac=0.3)
    fit, held = inner_split(d, cfg)
    assert set(d.roles[held]) == {"train"} and not set(fit) & set(held) and held.size
    gf = {d.split.basin_group[d.sites[s]] for s in fit}
    gh = {d.split.basin_group[d.sites[s]] for s in held}
    assert not gf & gh                                    # whole basin groups held out
    tr, _ = build_train_inputs(d, cfg)
    assert len(tr) == fit.size
    va, vinfo = build_val_inputs(d, cfg)
    assert len(va) > 0 and vinfo["n_sites"] == held.size
    assert pd.Timestamp(vinfo["first_issue"]) >= pd.Timestamp("2019-11-12")
    end = pd.Timestamp(TIME_BLOCKS["train"][1])
    # every inner-validation window ends inside the TRAIN block (no val data)
    assert all(x["context"].shape[1] == cfg.context + 14 for x in va)
    del end


def test_stack_logistic_roundtrip_and_monotone_gbm():
    """EW-v2 cycle 2: the stack serialises exactly; monotone constraints make P(onset) non-increasing
    in the safe-side margin."""
    rng = np.random.default_rng(0)
    n = 4000
    margin = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(3 + 2 * margin))).astype(int)
    raw = {"a": 1 / (1 + np.exp(2 + 1.5 * margin)), "b": np.clip(rng.random(n), 0, 1)}
    st = E.StackLogistic(["a", "b"], 1.0).fit(raw, y)
    st2 = E.StackLogistic.from_dict(json.loads(json.dumps(st.to_dict())))
    assert np.allclose(st.predict(raw), st2.predict(raw))
    X = pd.DataFrame({"margin_last": margin, "noise": rng.normal(size=n)})
    cfg = E.EventGBMConfig(max_rounds=60, min_data_in_leaf=20, num_threads=1, monotone={"margin_last": -1})
    m = E.EventGBM(cfg).fit(X, y)
    grid = pd.DataFrame({"margin_last": np.linspace(-3, 3, 50), "noise": np.zeros(50)})
    assert np.all(np.diff(m.predict(grid)) <= 1e-12)


def test_freeze_record_may_list_its_own_scoring_files(tmp_path):
    """An EW-only freeze record hashes EW_SCORING_CODE; a list missing required files is refused."""
    files = list(guard.EW_SCORING_CODE)
    rec = {"created": "2026-11-15", "split_sha256": "x" * 64, "weights_sha256": {"none": "0"},
           "thresholds_sha256": "t" * 64, "scoring_code_files": files,
           "scoring_code_sha256": guard.scoring_code_sha256(files=files)}
    ff = tmp_path / "FREEZE_ew.json"
    ff.write_text(json.dumps(rec))
    assert guard.load_freeze(ff)["scoring_code_files"] == files
    ff.write_text(json.dumps({**rec, "scoring_code_files": files[:-1]}))
    with pytest.raises(guard.ProtectedDataError):
        guard.load_freeze(ff)
    ff.write_text(json.dumps({**rec, "scoring_code_sha256": "0" * 64}))
    with pytest.raises(guard.ProtectedDataError):
        guard.load_freeze(ff)


def test_event_gbm_bagging_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"margin_last": rng.normal(size=3000), "b": rng.normal(size=3000)})
    y = (rng.random(3000) < 1 / (1 + np.exp(2 + 2 * X["margin_last"]))).astype(int)
    m = E.EventGBM(E.EventGBMConfig(max_rounds=40, min_data_in_leaf=20, num_threads=1, n_bags=3)).fit(X, y)
    assert len(m.boosters) == 3
    f = tmp_path / "event_gbm.json"
    f.write_text(m.to_json())
    assert np.allclose(E.EventGBM.from_file(f).predict(X), m.predict(X))
