# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Tests for SENTINEL-HydroMet (M6): no sensor-derived inputs, role disjointness, train-only
normalisation, causal forcing windows, physics components and the test-set guard.

Synthetic arrays here only exercise code paths and leakage properties; no test output is a
research result. No network access and no real data (one schema check skips if the processed
StreamCat file is absent).
"""

import json

import numpy as np
import pandas as pd
import pytest

from sentinel.data.sonde_splits import SondeSplit
from sentinel.models.hydromet import baselines as B
from sentinel.models.hydromet import data as D
from sentinel.models.hydromet import guard, runner
from sentinel.models.hydromet.evaluate import common_mask, site_metrics, summarize
from sentinel.models.hydromet.features import FeatureSet, forcing_features, trailing
from sentinel.models.hydromet.physics import air2stream_simulate, do_sat_mgl, pressure_atm_from_elevation

STATIC_EXTRA = ["wsareasqkm", "bfiws", "tmean9120ws", "precip9120ws", "elevws", "pctimp2019ws", "pcturbop2019ws"]


def synth(S=8, roles=None, end="2023-09-30", seed=0, groups=None, blocks=("train", "val")) -> D.HydroMetData:
    roles = roles or ["train"] * 5 + ["val"] * 2 + ["test"]
    groups = groups or ["g1", "g1", "g2", "g3", "g3", "g4", "g4", "g5"][:S]
    sites = [f"{i:08d}" for i in range(S)]
    dates = pd.date_range(D.DATA_START, end, freq="D")
    T = len(dates)
    rs = np.random.default_rng(seed)
    doy = dates.dayofyear.to_numpy()
    season = 12 * np.sin(2 * np.pi * (doy - 110) / 365.25)
    forc = np.zeros((S, T, len(D.FORCING_VARS)), np.float32)
    for i in range(S):
        tmean = 10 + season + rs.normal(0, 3, T) + i
        forc[i, :, D.FORCING_VARS.index("tmax_c")] = tmean + 5
        forc[i, :, D.FORCING_VARS.index("tmin_c")] = tmean - 5
        forc[i, :, D.FORCING_VARS.index("tmean_c")] = tmean
        forc[i, :, D.FORCING_VARS.index("precip_mm")] = rs.exponential(2, T) * (rs.random(T) < 0.3)
        for v in ("srad_wm2", "wind_ms", "rhmax_pct", "rhmin_pct", "sph_kgkg", "vpd_kpa", "pet_mm"):
            forc[i, :, D.FORCING_VARS.index(v)] = rs.uniform(0.5, 2, T)
    y = np.full((S, T, len(D.TARGETS)), np.nan, np.float32)
    prov = np.zeros_like(y, dtype=bool)
    for i in range(S):
        ta = forc[i, :, D.FORCING_VARS.index("tmean_c")]
        tw = np.maximum(pd.Series(ta).rolling(7, min_periods=1).mean().to_numpy() * 0.8 + 2, 0)
        y[i, :, 0] = tw
        y[i, :, 1] = tw + 2
        y[i, :, 2] = do_sat_mgl(tw) * 0.9
        y[i, :, 3] = do_sat_mgl(tw) * 0.8
        y[i, :, 4] = 300 + 50 * i + rs.normal(0, 20, T)
        y[i, :, 5] = np.exp(rs.normal(1, 1, T))
        y[i, :, 6] = np.exp(rs.normal(0, 0.5, T))
    y[:, :400] = np.nan                       # some missing history
    y[rs.random(y.shape) < 0.1] = np.nan
    prov[:, -60:] = True                      # provisional tail
    static = pd.DataFrame({"lat": 35 + rs.uniform(0, 10, S), "lon": -100 + rs.uniform(0, 20, S),
                           "elev_m": rs.uniform(0, 1500, S), **{c: rs.uniform(1, 100, S) for c in STATIC_EXTRA},
                           "bfiws": rs.uniform(10, 80, S), "runoffws": rs.uniform(10, 900, S),
                           "elevcat": rs.uniform(0, 1500, S)}, index=sites)
    split = SondeSplit(dict(zip(sites, roles)), dict(zip(sites, groups)))
    return D.HydroMetData(sites, np.array(roles), np.array(groups), dates, forc, y, prov, static, split,
                          {"final": False, "sha256": split.sha256()}, {}, tuple(blocks))


@pytest.fixture(scope="module")
def data():
    return synth()


# ----------------------------------------------------------------------------- physics

def test_do_sat_benson_krause_table():
    # docs/research/06 §a.6 table (1 atm)
    t = np.array([0, 5, 10, 15, 20, 25, 30.0])
    np.testing.assert_allclose(do_sat_mgl(t), [14.62, 12.77, 11.29, 10.08, 9.09, 8.26, 7.56], atol=0.01)
    assert abs((do_sat_mgl(20.0) - do_sat_mgl(22.0)) - 0.35) < 0.02          # +2 °C at 20 °C
    p = pressure_atm_from_elevation(1500.0)
    assert 0.82 < p < 0.85 and 0.8 < do_sat_mgl(20.0, p) / do_sat_mgl(20.0) < 0.86
    torch = pytest.importorskip("torch")
    np.testing.assert_allclose(do_sat_mgl(torch.tensor(t)).numpy(), do_sat_mgl(t), rtol=1e-10)


def test_air2stream_causal_and_numpy_torch_agree():
    rs = np.random.default_rng(0)
    ta = rs.normal(10, 5, (3, 200))
    doy = (np.arange(200) % 365) + 1
    params = {k: np.array([0.3, 0.2, 0.1]) if k == "a1" else np.full(3, 0.1 if k != "a6" else 0.6)
              for k in ("a1", "a2", "a3", "a5", "a6")}
    tw = air2stream_simulate(ta, doy, params)
    ta2 = ta.copy()
    ta2[:, 120:] += 50
    tw2 = air2stream_simulate(ta2, doy, params)
    np.testing.assert_allclose(tw[:, :120], tw2[:, :120])
    assert not np.allclose(tw[:, 120:], tw2[:, 120:])
    torch = pytest.importorskip("torch")
    tw_t = air2stream_simulate(torch.tensor(ta), doy, {k: torch.tensor(v) for k, v in params.items()})
    np.testing.assert_allclose(tw_t.numpy(), tw, rtol=1e-8, atol=1e-8)


# ----------------------------------------------------------------------------- inputs: no sensor data

def test_feature_columns_are_only_forcing_static_calendar(data):
    fs = FeatureSet.build(data)
    cols = fs.columns()
    groups = D.assert_allowed_inputs(cols, static_cols=list(data.static.columns), forcing_feature_cols=fs.dyn_names)
    assert set(groups["forcing"]) == set(fs.dyn_names) and all(c.startswith("f_") for c in groups["forcing"])
    assert set(groups["calendar"]) == set(D.CALENDAR_COLS)
    assert set(groups["static"]) == set(data.static.columns)
    for c in cols:
        assert not D.SENSOR_PATTERN.search(c), c


@pytest.mark.parametrize("bad", ["temp_c_mean", "do_mgl_min", "discharge_cfs_mean", "gage_ft_mean", "spcond_uscm_mean",
                                 "turb_fnu_mean", "no3no2_mgl_mean", "has_do", "n_upstream_sites", "ph_median"])
def test_sensor_named_columns_are_refused(data, bad):
    fs = FeatureSet.build(data)
    with pytest.raises(ValueError):
        D.assert_allowed_inputs(fs.columns() + [bad], static_cols=list(data.static.columns) + [bad],
                                forcing_feature_cols=fs.dyn_names)
    with pytest.raises(ValueError):   # a column outside the three groups is refused too
        fs.design_matrix(np.array([0]), np.array([500]), fs.columns()[:3] + [bad])


def test_urban_landcover_names_not_flagged():
    for c in ("pcturbop2019ws", "pcturbhi2019cat", "sph_kgkg", "f_sph_kgkg_m1", "precip_minus_evtcat", "n_tin_2017cat"):
        assert not D.SENSOR_PATTERN.search(c), c


def test_real_streamcat_columns_pass_whitelist():
    if not D.STREAMCAT_FILE.is_file():
        pytest.skip("processed StreamCat file not present")
    cols = D.streamcat_feature_columns()
    D.assert_allowed_inputs(cols + D.SITE_STATIC, static_cols=cols, forcing_feature_cols=[])


def test_inputs_invariant_to_targets(data):
    """Feature/input arrays are computed without reading any target/sensor value."""
    fs1 = FeatureSet.build(data)
    d2 = synth(seed=0)
    d2.y = np.random.default_rng(5).normal(size=d2.y.shape).astype("float32")
    fs2 = FeatureSet.build(d2)
    np.testing.assert_array_equal(fs1.dyn, fs2.dyn)
    pd.testing.assert_frame_equal(fs1.static, fs2.static)
    pytest.importorskip("torch")
    from sentinel.models.hydromet.lstm import build_inputs

    fit, _ = D.inner_split(data, 0.2, 0)
    norm = D.Normaliser.fit(data, fit)
    a, s1, _, _ = build_inputs(data, norm, 0)
    b, s2, _, _ = build_inputs(d2, norm, 0)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(s1, s2)


def _fit_predict_all(d, iters=5):
    fs = FeatureSet.build(d)
    fit, inner = D.inner_split(d, 0.2, 0)
    cfg = {"models": ["doy_location_gbm", "forcing_gbm", "air2water_linear", "air2stream_regional",
                      "dosat_ratio[air2stream_regional]"],
           "lgbm": {"n_estimators": 30, "n_jobs": 1, "max_rows_per_site": 300, "early_stopping_rounds": 5},
           "air2stream": {"iters": iters, "min_obs": 100, "min_calib_nse": -10.0, "k_donors": 2}}
    suite = B.fit_suite(cfg, d, fs, fit, inner, log=lambda *_: None)
    si = d.site_idx(["val"])
    ti = np.flatnonzero(d.block_mask("val"))
    return {n: m.predict(d, fs, si, ti) for n, m in suite.models.items()}


def test_baseline_predictions_ignore_val_targets_and_val_block():
    """Changing every target value outside train-role sites × train block (val sites, test site,
    and the val block at train sites) must not change any baseline prediction."""
    d1 = synth()
    d2 = synth()
    tm = d2.block_mask("train")
    rs = np.random.default_rng(1)
    noise = rs.normal(0, 5, d2.y.shape).astype("float32")
    keep = np.zeros(d2.y.shape, bool)
    keep[d2.site_idx(["train"])] = True
    keep[:, ~tm] = False
    d2.y = np.where(keep, d2.y, d2.y + noise)
    p1, p2 = _fit_predict_all(d1), _fit_predict_all(d2)
    for name in p1:
        for v in p1[name]:
            np.testing.assert_allclose(p1[name][v], p2[name][v], rtol=1e-6, err_msg=f"{name}/{v}")


# ----------------------------------------------------------------------------- roles

def test_inner_split_disjoint_grouped_and_train_only(data):
    fit, inner = D.inner_split(data, 0.3, 0)
    assert len(inner) and len(fit)
    assert set(fit).isdisjoint(inner)
    assert (data.roles[np.r_[fit, inner]] == "train").all()
    assert set(data.basin_group[fit]).isdisjoint(set(data.basin_group[inner]))


def test_fitting_on_non_train_sites_is_refused(data):
    fs = FeatureSet.build(data)
    val = data.site_idx(["val"])
    fit, inner = D.inner_split(data, 0.2, 0)
    with pytest.raises(ValueError):
        B.doy_location_gbm().fit(data, fs, val, inner)
    with pytest.raises(ValueError):
        B.AirWaterRegression().fit(data, fs, fit, val)
    with pytest.raises(ValueError):
        D.Normaliser.fit(data, np.r_[fit, val])
    pytest.importorskip("torch")
    from sentinel.models.hydromet.lstm import EALSTMConfig, HydroMetEALSTM

    with pytest.raises(ValueError):
        HydroMetEALSTM(EALSTMConfig()).fit(data, fit, val, device="cpu")


def test_block_mask_embargo_and_unauthorised_block(data):
    m = data.block_mask("val")
    first = data.dates[m][0]
    assert first == pd.Timestamp("2021-10-01") + pd.Timedelta(days=42)
    with pytest.raises(guard.ProtectedDataError):
        data.block_mask("test")


# ----------------------------------------------------------------------------- normalisation

def test_normaliser_uses_train_sites_and_train_block_only():
    d1 = synth()
    fit, _ = D.inner_split(d1, 0.2, 0)
    n1 = D.Normaliser.fit(d1, fit)
    d2 = synth()
    other = np.setdiff1d(np.arange(len(d2.sites)), fit)
    d2.forc[other] *= 50                                  # non-fit sites
    d2.forc[:, ~d2.block_mask("train")] += 100            # outside the train block
    d2.y[other] *= 10
    d2.y[:, ~d2.block_mask("train")] *= 10
    d2.static.iloc[other] = d2.static.iloc[other] * 7
    n2 = D.Normaliser.fit(d2, fit)
    for k in ("forc_mean", "forc_std", "static_mean", "static_std", "target_mean", "target_std"):
        np.testing.assert_allclose(getattr(n1, k), getattr(n2, k), rtol=1e-6, err_msg=k)
    tm = d1.block_mask("train")
    j = D.FORCING_VARS.index("tmean_c")
    np.testing.assert_allclose(n1.forc_mean[j], d1.forc[fit][:, tm, j].astype("float64").mean(), rtol=1e-6)


# ----------------------------------------------------------------------------- causality

@pytest.mark.parametrize("lag", [0, 1, 3])
def test_forcing_features_are_causal(data, lag):
    t0 = 3000
    f1, names = forcing_features(data.forc, data.static["elev_m"].to_numpy(), lag)
    forc2 = data.forc.copy()
    forc2[:, t0 + 1:] = forc2[:, t0 + 1:] * 3 + 7          # perturb strictly after t0
    f2, _ = forcing_features(forc2, data.static["elev_m"].to_numpy(), lag)
    np.testing.assert_array_equal(f1[:, : t0 + 1 + lag], f2[:, : t0 + 1 + lag])
    assert not np.allclose(np.nan_to_num(f1[:, t0 + 1 + lag:]), np.nan_to_num(f2[:, t0 + 1 + lag:]))
    if lag:   # with lag L the same-day (t0) forcing must not affect day t0's features
        forc3 = data.forc.copy()
        forc3[:, t0] += 25
        f3, _ = forcing_features(forc3, data.static["elev_m"].to_numpy(), lag)
        np.testing.assert_array_equal(f1[:, t0 + lag - 1], f3[:, t0 + lag - 1])


def test_trailing_window_definition():
    x = np.arange(10, dtype=float)[None, :]
    np.testing.assert_allclose(trailing(x, 3, "mean")[0, 2:], [1, 2, 3, 4, 5, 6, 7, 8])
    assert np.isnan(trailing(x, 3, "mean")[0, :2]).all()
    np.testing.assert_allclose(trailing(x, 2, "sum")[0, 1:4], [1, 3, 5])


def test_ealstm_output_is_causal(data):
    torch = pytest.importorskip("torch")
    from sentinel.models.hydromet.lstm import EALSTMConfig, make_model

    cfg = EALSTMConfig(hidden=8, static_emb=4, dropout=0.0)
    m = make_model(15, 6, cfg).eval()
    x = torch.randn(2, 50, 15)
    s = torch.randn(2, 6)
    x2 = x.clone()
    x2[:, 30:] += 5.0
    with torch.no_grad():
        o1, o2 = m(x, s, 50), m(x2, s, 50)
    torch.testing.assert_close(o1[:, :30], o2[:, :30])
    assert not torch.allclose(o1[:, 30:], o2[:, 30:])


def test_ealstm_fit_predict_smoke_cpu(data):
    pytest.importorskip("torch")
    from sentinel.models.hydromet.lstm import EALSTMConfig, HydroMetEALSTM

    fit, inner = D.inner_split(data, 0.2, 0)
    for physics in (False, True):
        cfg = EALSTMConfig(seq_len=60, predict_last=20, hidden=8, static_emb=4, batch_size=8, max_epochs=1,
                           steps_per_epoch=2, val_max_windows=10, physics_do=physics)
        m = HydroMetEALSTM(cfg)
        m.fit(data, fit, inner, device="cpu", log=lambda *_: None)
        si, ti = data.site_idx(["val"]), np.flatnonzero(data.block_mask("val"))
        p = m.predict(data, si, ti, device="cpu")
        assert set(p) == set(D.TARGETS)
        assert all(np.isfinite(v).all() for v in p.values())
        assert set(m.info["inputs"]["static"]) <= set(data.static.columns)


# ----------------------------------------------------------------------------- targets & scoring

def test_7dadm_strict_trailing_mean():
    x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, np.nan, 10, 11, 12, 13, 14, 15, 16]], float)
    out = D.trailing_mean_strict(x, 7)
    assert np.isnan(out[0, :6]).all()
    assert out[0, 6] == pytest.approx(4.0) and out[0, 7] == pytest.approx(5.0)
    assert np.isnan(out[0, 8:15]).all() and out[0, 15] == pytest.approx(13.0)


def test_common_mask_identical_rows_and_approved_only(data):
    si, ti = data.site_idx(["val"]), np.flatnonzero(data.block_mask("val"))
    v = "temp_c_mean"
    obs = data.y[si][:, ti, data.vi(v)]
    a = obs + 1.0
    b = obs.copy()
    b[:, :5] = np.nan
    m, drops = common_mask(data, v, "val", si, ti, {"a": {v: a}, "b": {v: b}})
    assert not m[:, :5].any() and drops["a"] == 0 and drops["b"] > 0
    assert not (m & data.prov[si][:, ti, data.vi(v)]).any()
    df, info = site_metrics(data, {"a": {v: a}, "b": {v: b}}, si, ti, "val", targets=[v])
    assert set(df["model"]) == {"a", "b"} and (df.groupby("site_no")["n"].nunique() == 1).all()
    s = summarize(df, refs=("b",), n_boot=50)
    r = s[(s.model == "a") & (s.metric == "rmse")].iloc[0]
    assert r["median"] == pytest.approx(1.0, abs=1e-6) and r["ci_unit"] == "basin_group" and r["n_groups"] >= 1


# ----------------------------------------------------------------------------- guard & modes

def _claim_sandbox(monkeypatch, tmp_path, ff):
    """Single-pass guard (pre-test review C1): protected data need an open claim of this freeze; sandboxed."""
    from sentinel.core import paths as _paths
    from sentinel.models.hydromet import testlook as TL
    import sentinel.core.registry as R

    monkeypatch.setattr(_paths, "RUNS_ROOT", tmp_path / "runs_sandbox")
    monkeypatch.setattr(guard, "FREEZE_DIR", tmp_path)
    monkeypatch.setattr(TL, "registry_rows", lambda exp: 0)
    monkeypatch.setattr(R, "log_result", lambda **k: None)
    TL.claim("M6/test", guard.load_freeze(ff, check_code=False))


def _look_blocks(models=("ealstm",)):
    comps = {v: {"gbm": "forcing_gbm", "physics": "air2stream_regional"} for v in
             ("temp_c_mean", "temp_c_7dadm", "do_mgl_mean", "do_mgl_min")}
    return {"M6/test": {"block": "test", "roles": ["test"], "dataset": "sonde", "primary_models": list(models),
                        "n_hypotheses": 4 * len(models), "comparators": comps, "artifacts": {}, "data_snapshots": []}}


def test_guard_refuses_test_without_flag_and_freeze(tmp_path, data, monkeypatch):
    with pytest.raises(guard.ProtectedDataError):
        guard.check_access(("train", "test"), ("train",))
    with pytest.raises(guard.ProtectedDataError):
        guard.check_access(("train", "val"), ("test",))
    with pytest.raises(guard.ProtectedDataError):
        guard.check_access(("train", "test"), ("train",), allow_test=True, freeze_file=tmp_path / "missing.json")
    monkeypatch.setattr(guard, "FREEZE_DIR", tmp_path)
    good = {"created": "2026-10-22", "status": "FROZEN", "authorised_by": "lead (test)", "looks": _look_blocks(),
            "split_sha256": data.split.sha256(),
            "scoring_code_sha256": guard.scoring_code_sha256(), "weights_sha256": {"x": "0" * 64}}
    ff = tmp_path / "FREEZE_hydromet.json"
    ff.write_text(json.dumps(good))
    with pytest.raises(guard.ProtectedDataError):   # valid freeze but no open single-pass claim
        guard.check_access(("train", "test"), ("test",), allow_test=True, freeze_file=ff, split=data.split)
    _claim_sandbox(monkeypatch, tmp_path, ff)
    rec = guard.check_access(("train", "test"), ("test",), allow_test=True, freeze_file=ff, split=data.split)
    assert rec["freeze_file"] == str(ff)
    with pytest.raises(guard.ProtectedDataError):   # training code never gets protected data
        guard.check_access(("train", "test"), ("test",), allow_test=True, freeze_file=ff, split=data.split,
                           training=True)
    ff.write_text(json.dumps({**good, "scoring_code_sha256": "f" * 64}))
    with pytest.raises(guard.ProtectedDataError):
        guard.check_access(("train", "test"), (), allow_test=True, freeze_file=ff, split=data.split)
    assert guard.check_access(("train", "val"), ("train", "val"))["protected"] == []


def test_prepare_refuses_test_in_training():
    class A:
        eval_blocks = "test"
        split = None
    with pytest.raises(guard.ProtectedDataError):
        runner.prepare(A(), {"_smoke": True}, None, training=True)


def test_modes():
    assert runner.check_mode({"mode": "nowcast", "forcing_lag_days": 0}) == 0
    assert runner.check_mode({"mode": "lagged", "forcing_lag_days": 2}) == 2
    with pytest.raises(ValueError):
        runner.check_mode({"mode": "nowcast", "forcing_lag_days": 1})
    with pytest.raises(NotImplementedError):
        runner.check_mode({"mode": "forecast_nwp", "forcing_lag_days": 0})


def test_registry_refuses_smoke_and_nonfinal():
    with pytest.raises(RuntimeError):
        runner.assert_registrable({"_smoke": True}, {"final": True})
    with pytest.raises(RuntimeError):
        runner.assert_registrable({"_smoke": False}, {"final": False})


# ----------------------------------------------------------------------------- pre-registered analysis

def _site_frame(rs, models, var="temp_c_mean", n=12, shift=None, n_groups=4):
    rows = []
    for m in models:
        for i in range(n):
            rows.append({"model": m, "var": var, "site_no": f"{i:03d}", "role": "val", "n": 100,
                         "basin_group": f"G{i % n_groups}", "kge": 0.5 + (shift or {}).get(m, 0.0) + rs.normal(0, 0.01)})
    return pd.DataFrame(rows)


def test_group_bootstrap_is_primary_and_wider_than_site_bootstrap():
    from sentinel.models.hydromet.evaluate import summarize

    # two basin groups with very different skill: site resampling ignores the clustering
    rows = [{"model": "m", "var": "temp_c_mean", "site_no": f"{i:03d}", "role": "val", "n": 100,
             "basin_group": "A" if i < 20 else "B", "kge": (0.9 if i < 20 else 0.1) + 0.001 * i} for i in range(40)]
    s = summarize(pd.DataFrame(rows), n_boot=500, metrics=("kge",)).iloc[0]
    assert s["ci_unit"] == "basin_group" and s["n_groups"] == 2 and s["n_sites"] == 40
    assert (s["ci_high"] - s["ci_low"]) > (s["ci_high_site"] - s["ci_low_site"])


def test_pooled_metrics_match_metrics_module_and_use_groups(data):
    from sentinel.eval import metrics as Mx
    from sentinel.models.hydromet.evaluate import pooled_metrics

    si, ti = data.site_idx(["train", "val"]), np.flatnonzero(data.block_mask("val"))
    obs = data.y[si][:, ti, data.vi("temp_c_mean")]
    noisy = obs + np.random.default_rng(3).normal(0, 1.5, obs.shape).astype("float32")
    df = pooled_metrics(data, {"a": {"temp_c_mean": noisy}}, si, ti, "val", n_boot=100, targets=["temp_c_mean"])
    p = df.set_index("metric")
    m = data.target_mask("temp_c_mean", "val", si, approved_only=True)[si][:, ti]
    y, q = obs[m].astype(float), noisy[m].astype(float)
    assert p.loc["pooled_nse", "value"] == pytest.approx(Mx.nse(y, q), rel=1e-6)
    assert p.loc["pooled_kge", "value"] == pytest.approx(Mx.kge(y, q), rel=1e-6)
    assert p.loc["pooled_rmse", "value"] == pytest.approx(Mx.rmse(y, q), rel=1e-6)
    assert (df["ci_unit"] == "basin_group").all() and (df["n_groups"] == len(set(data.basin_group[si]))).all()


def test_select_comparators_two_families_and_iut_holm_seed_rule():
    from sentinel.models.hydromet import evaluate as E

    rs = np.random.default_rng(0)
    base = ["forcing_gbm", "doy_location_gbm", "air2stream_regional", "air2water_linear", "air2water_mohseni"]
    df = _site_frame(rs, ["ealstm", "ealstm_s0", "ealstm_s1"] + base, n=24, n_groups=8,
                     shift={"ealstm": 0.2, "ealstm_s0": 0.18, "ealstm_s1": 0.19, "forcing_gbm": 0.05,
                            "air2water_mohseni": 0.08, "doy_location_gbm": -0.3})
    comps = E.select_comparators(df)
    assert comps == {"temp_c_mean": {"gbm": "forcing_gbm", "physics": "air2water_mohseni"}}
    res = E.primary_comparison(df, comps, models=("ealstm",), endpoints=("temp_c_mean",),
                               seed_models={"ealstm": ["ealstm_s0", "ealstm_s1"]}, n_boot=300)
    r = res.iloc[0]
    assert r["ci_unit"] == "basin_group" and r["gbm_n_groups"] == 8
    assert r["gbm_diff"] == pytest.approx(0.15, abs=0.02) and r["physics_diff"] == pytest.approx(0.12, abs=0.02)
    assert r["p_iut"] == max(r["gbm_p"], r["physics_p"]) and bool(r["won"])
    # every-seed rule: one seed worse than the physics comparator -> not won, whatever the ensemble does
    df2 = df.copy()
    df2.loc[df2["model"] == "ealstm_s1", "kge"] -= 0.15
    r2 = E.primary_comparison(df2, comps, models=("ealstm",), endpoints=("temp_c_mean",),
                              seed_models={"ealstm": ["ealstm_s0", "ealstm_s1"]}, n_boot=300).iloc[0]
    assert not r2["physics_every_seed_positive"] and not bool(r2["won"])
    # beating only one family is not a win (intersection-union)
    df3 = df.copy()
    df3.loc[df3["model"] == "air2water_mohseni", "kge"] += 0.2
    r3 = E.primary_comparison(df3, comps, models=("ealstm",), endpoints=("temp_c_mean",),
                              seed_models={"ealstm": ["ealstm_s0", "ealstm_s1"]}, n_boot=300).iloc[0]
    assert r3["gbm_diff"] > 0 and r3["physics_diff"] < 0 and not bool(r3["won"])
    with pytest.raises(ValueError):
        E.select_comparators(df.assign(role="test"), role="test")


def test_seed_ensemble_mean():
    from sentinel.models.hydromet.pipeline import seed_ensembles

    a, b = np.full((2, 3), 10.0, np.float32), np.full((2, 3), 12.0, np.float32)
    ens, members = seed_ensembles({"ealstm_s0": {"temp_c_mean": a}, "ealstm_s1": {"temp_c_mean": b}, "forcing_gbm": {}})
    np.testing.assert_allclose(ens["ealstm"]["temp_c_mean"], 11.0)
    assert members == {"ealstm": ["ealstm_s0", "ealstm_s1"]} and "forcing_gbm" not in ens
    assert seed_ensembles({"ealstm_s0": {"temp_c_mean": a}}) == ({}, {})


# ----------------------------------------------------------------------------- guard: split, freeze coverage

def test_non_final_split_refused(tmp_path):
    sp = SondeSplit({"01": "train", "02": "test"}, {"01": "a", "02": "b"}, notes="DEV split")
    f = tmp_path / "dev.json"
    f.write_text(sp.to_json())
    with pytest.raises(guard.ProtectedDataError):
        D.load_split(f)
    dev = D.paths.REPO_ROOT / "docs" / "preregistration" / "sonde_split_dev.json"
    if dev.is_file():
        with pytest.raises(guard.ProtectedDataError):
            D.load_split(dev)
    with pytest.raises(ValueError):   # smoke never selects val-role (or test-role) sites
        D.smoke_sites(sp, n_train=5, n_val=3)


def test_freeze_hash_covers_all_model_code_configs_and_protocol(tmp_path):
    import shutil

    files = guard.frozen_files()
    root = D.paths.REPO_ROOT
    for f in root.glob("sentinel/models/hydromet/*.py"):
        assert str(f.relative_to(root)) in files
    for f in root.glob("configs/hydromet/*.yaml"):
        assert str(f.relative_to(root)) in files
    assert "docs/preregistration/hydromet_protocol.md" in files and "sentinel/models/hydromet/lstm.py" in files
    for rel in files:
        if (root / rel).is_file():
            (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(root / rel, tmp_path / rel)
    h0 = guard.scoring_code_sha256(tmp_path)
    with open(tmp_path / "sentinel/models/hydromet/baselines.py", "a") as fh:
        fh.write("\n# changed\n")
    assert guard.scoring_code_sha256(tmp_path) != h0
    (tmp_path / "configs/hydromet/new.yaml").write_text("x: 1\n")   # an added config also changes the hash
    h1 = guard.scoring_code_sha256(tmp_path)
    (tmp_path / "configs/hydromet/new.yaml").unlink()
    assert guard.scoring_code_sha256(tmp_path) != h1


def test_static_ablation_no_gauge_derived(data):
    st = D.apply_static_ablation(data.static, "no_gauge_derived")
    assert not set(D.GAUGE_DERIVED_STATIC) & set(st.columns)
    np.testing.assert_allclose(st["elev_m"], data.static["elevcat"])
    assert D.apply_static_ablation(data.static, None) is data.static
    with pytest.raises(ValueError):
        D.apply_static_ablation(data.static, "something_else")


def test_pseudo_roles_inner_and_fit(data):
    d = synth()
    fit, inner = D.inner_split(d, 0.3, 0)
    d.info["inner_sites"] = [d.sites[i] for i in inner]
    np.testing.assert_array_equal(np.sort(d.site_idx(["inner"])), np.sort(inner))
    np.testing.assert_array_equal(np.sort(d.site_idx(["fit"])), np.sort(fit))
    assert d.role_label(int(inner[0])) == "inner" and d.role_label(int(fit[0])) == "train"


# ----------------------------------------------------------------------------- end-to-end

def test_ealstm_predictions_ignore_targets_outside_train_cell():
    pytest.importorskip("torch")
    from sentinel.models.hydromet.lstm import EALSTMConfig, HydroMetEALSTM

    def run(d):
        fit, inner = D.inner_split(d, 0.2, 0)
        m = HydroMetEALSTM(EALSTMConfig(seq_len=60, predict_last=20, hidden=8, static_emb=4, batch_size=8, max_epochs=2,
                                        steps_per_epoch=3, val_max_windows=10, physics_do=True, dropout=0.0))
        m.fit(d, fit, inner, device="cpu", log=lambda *_: None)
        return m.predict(d, d.site_idx(["val"]), np.flatnonzero(d.block_mask("val")), device="cpu")

    d1, d2 = synth(), synth()
    keep = np.zeros(d2.y.shape, bool)
    keep[d2.site_idx(["train"])] = True
    keep[:, ~d2.block_mask("train")] = False
    d2.y = np.where(keep, d2.y, d2.y + np.random.default_rng(9).normal(0, 5, d2.y.shape).astype("float32"))
    p1, p2 = run(d1), run(d2)
    for v in p1:
        np.testing.assert_array_equal(p1[v], p2[v], err_msg=v)


def _e2e_world(tmp_path):
    """Synthetic world with test-role sites, fitted baselines and 2 EA-LSTM seeds saved like real runs."""
    pytest.importorskip("torch")
    import json as _json

    from sentinel.models.hydromet.features import FeatureSet
    from sentinel.models.hydromet.lstm import EALSTMConfig, HydroMetEALSTM

    roles = ["train"] * 6 + ["val"] * 2 + ["test"] * 4
    groups = ["g1", "g1", "g2", "g3", "g3", "g6", "g4", "g4", "g5", "g5", "g7", "g8"]
    d = synth(S=12, roles=roles, groups=groups, end="2025-09-30", blocks=("train", "val", "test"))
    fs = FeatureSet.build(d)
    fit, inner = D.inner_split(d, 0.2, 0)
    cfg = {"models": ["doy_location_gbm", "forcing_gbm", "air2water_linear", "air2water_mohseni", "air2stream_regional",
                      "dosat_ratio[air2stream_regional]", "dosat_ratio[doy_location_gbm]"],
           "lgbm": {"n_estimators": 20, "n_jobs": 1, "max_rows_per_site": 200, "early_stopping_rounds": 5},
           "air2stream": {"iters": 3, "min_obs": 100, "min_calib_nse": -10.0, "k_donors": 2}}
    suite = B.fit_suite(cfg, d, fs, fit, inner, log=lambda *_: None)
    brun = tmp_path / "base"
    for n, m in suite.models.items():
        m.save(brun / "models" / f"{n}.pkl")
    lrun = tmp_path / "lstm"
    for seed in (0, 1):
        m = HydroMetEALSTM(EALSTMConfig(seq_len=60, predict_last=20, hidden=8, static_emb=4, batch_size=8, max_epochs=1,
                                        steps_per_epoch=2, val_max_windows=10, seed=seed))
        m.info["model_name"] = "ealstm"
        m.fit(d, fit, inner, device="cpu", log=lambda *_: None)
        m.save(lrun / f"g0_s{seed}" / "ealstm_best.pt")
    m.save(lrun / "g1_s0" / "ealstm_best.pt")             # a NON-selected grid point: must be ignored
    (lrun / "selection.json").write_text(_json.dumps({"selected_index": 0}))
    return d


def test_predict_score_end_to_end_with_freeze(tmp_path, monkeypatch):
    """C2: per-seed artifact keys survive the frozen test path; C3c: provenance is enforced."""
    import json as _json

    from sentinel.models.hydromet import evaluate as E
    from sentinel.models.hydromet import pipeline

    d = _e2e_world(tmp_path)
    arts = pipeline.discover_artifacts([tmp_path / "base", tmp_path / "lstm"])
    assert {"ealstm_s0", "ealstm_s1", "forcing_gbm", "air2stream_regional"} <= set(arts)
    assert str(arts["ealstm_s0"]).endswith("g0_s0/ealstm_best.pt")
    comps = {v: {"gbm": "forcing_gbm", "physics": "air2stream_regional" if v.startswith("temp")
                 else "dosat_ratio[air2stream_regional]"} for v in E.PRIMARY_TARGETS}
    monkeypatch.setattr(guard, "FREEZE_DIR", tmp_path)
    freeze = {"created": "2026-10-22", "status": "FROZEN", "authorised_by": "lead (test)",
              "looks": _look_blocks(E.SENTINEL_MODELS),
              "split_sha256": d.split.sha256(), "scoring_code_sha256": guard.scoring_code_sha256(),
              "weights_sha256": {k: guard.artifact_sha256(p) for k, p in arts.items()}, "comparators": comps}
    ff = tmp_path / "FREEZE_hydromet.json"
    ff.write_text(_json.dumps(freeze))
    _claim_sandbox(monkeypatch, tmp_path, ff)
    access = guard.check_access(("train", "test"), ("test",), allow_test=True, freeze_file=ff, split=d.split)
    si, ti = d.site_idx(["test"]), np.flatnonzero(d.block_mask("test"))
    out = tmp_path / "pred_test"
    files = pipeline.predict_artifacts(d, arts, si, ti, "test", out, lag=0, access=access, log=lambda *_: None)
    assert len(files) == len(arts)
    preds, _ = pipeline.load_prediction_set([out], d, si, ti, "test", smoke=False, access=access)
    assert {"ealstm_s0", "ealstm_s1"} <= set(preds)
    with pytest.raises(guard.ProtectedDataError):   # C2: the model family must come from the freeze on test
        pipeline.score(d, preds, si, ti, "test", ("test",), tmp_path / "score0", n_boot=50, min_days=60, primary=True,
                       access=access, log=lambda *_: None)
    res = pipeline.score(d, preds, si, ti, "test", ("test",), tmp_path / "score", n_boot=50, min_days=60, primary=True,
                         access=access, log=lambda *_: None, primary_models=list(E.SENTINEL_MODELS))
    assert res["members"] == {"ealstm": ["ealstm_s0", "ealstm_s1"]}
    prim = res["primary"]
    tested = prim[prim["status"] == "tested"]
    assert set(tested["var"]) == set(E.PRIMARY_TARGETS) and set(tested["model"]) == {"ealstm"}
    assert (tested["gbm_comparator"] == "forcing_gbm").all() and prim["n_hypotheses"].iloc[0] == 4 * len(E.SENTINEL_MODELS)
    assert (prim[prim["model"].isin(["ealstm_physdo", "ealstm_gbmblend"])]["status"] == "not available").all()
    # C3c: a prediction file whose artifact hash is not the frozen one is refused
    bad = tmp_path / "pred_bad"
    bad.mkdir()
    pf = E.load_predictions(files[0])
    hdr = dict(pf["header"], artifact_sha256="0" * 64)
    E.save_predictions(bad / f"{hdr['model']}__test.npz", model=hdr["model"],
                       preds={v: pf["pred"][k] for k, v in enumerate(pf["targets"])}, sites=pf["sites"],
                       dates=pd.DatetimeIndex(pf["dates"].astype("datetime64[ns]")), block="test",
                       meta={k: v for k, v in hdr.items() if k not in ("model", "block")})
    with pytest.raises(guard.ProtectedDataError):
        pipeline.load_prediction_set([bad], d, si, ti, "test", smoke=False, access=access)
    # an artifact that differs from the frozen weights is refused before predicting
    tampered = dict(arts)
    tampered["forcing_gbm"] = arts["doy_location_gbm"]
    with pytest.raises(guard.ProtectedDataError):
        pipeline.predict_artifacts(d, tampered, si, ti, "test", tmp_path / "x", lag=0, access=access)
    # smoke artifacts never touch protected data
    with pytest.raises(guard.ProtectedDataError):
        pipeline.predict_artifacts(d, arts, si, ti, "test", tmp_path / "y", lag=0, access=access, smoke=True)


def test_inner_split_greedy_fill_skips_large_groups():
    roles = ["train"] * 20 + ["val"]
    groups = ["BIG"] * 12 + [f"s{i}" for i in range(8)] + ["v"]
    d = synth(S=21, roles=roles, groups=groups)
    for seed in range(5):
        fit, inner = D.inner_split(d, 0.3, seed)
        assert 0 < len(inner) <= 0.3 * 20 and "BIG" not in set(d.basin_group[inner])
        assert set(d.basin_group[fit]).isdisjoint(d.basin_group[inner])


def test_untestable_hypotheses_stay_in_holm_family():
    from sentinel.models.hydromet import evaluate as E

    rs = np.random.default_rng(0)
    df = _site_frame(rs, ["ealstm", "forcing_gbm", "air2stream_regional"], n=24, n_groups=8,
                     shift={"ealstm": 0.2})
    comps = {"temp_c_mean": {"gbm": "forcing_gbm", "physics": "air2stream_regional"}}
    res = E.primary_comparison(df, comps, models=("ealstm", "ealstm_physdo"), endpoints=("temp_c_mean",), n_boot=200)
    assert len(res) == 2 and res["n_hypotheses"].iloc[0] == 2
    na = res[res["model"] == "ealstm_physdo"].iloc[0]
    assert na["status"] == "not available" and na["p_iut"] == 1.0 and not bool(na["won"])


def test_secondary_comparisons_pairs_and_group_unit():
    from sentinel.models.hydromet import evaluate as E

    rs = np.random.default_rng(1)
    df = _site_frame(rs, ["ealstm_s0", "ealstm_nogauge_s0", "forcing_gbm", "forcing_gbm_nogauge"], n=24, n_groups=8,
                     shift={"ealstm_nogauge_s0": -0.1, "forcing_gbm_nogauge": -0.02})
    sec = E.secondary_comparisons(df, n_boot=300)
    # only pairs whose two models are present; physics-DO / air2stream pairs are skipped
    assert set(sec["model"]) == {"ealstm_nogauge_s0", "forcing_gbm_nogauge"}
    r = sec[sec["model"] == "ealstm_nogauge_s0"].iloc[0]
    assert r["reference"] == "ealstm_s0" and r["ci_unit"] == "basin_group" and r["n_groups"] == 8
    assert r["diff"] == pytest.approx(-0.1, abs=0.02) and r["ci_high"] < 0
    assert (sec["analysis"] == "static_ablation_no_gauge_derived").all()


def test_seed_blends_pair_seeds_in_model_space():
    from sentinel.models.hydromet.pipeline import seed_blends, seed_ensembles

    a0, a1 = np.full((2, 3), 10.0, np.float32), np.full((2, 3), 12.0, np.float32)
    g0, g1 = np.full((2, 3), 14.0, np.float32), np.full((2, 3), 16.0, np.float32)
    preds = {"ealstm_s0": {"temp_c_mean": a0}, "ealstm_s1": {"temp_c_mean": a1},
             "forcing_gbm_bag_s0": {"temp_c_mean": g0}, "forcing_gbm_bag_s1": {"temp_c_mean": g1}}
    bl = seed_blends(preds)
    assert set(bl) == {"ealstm_gbmblend_s0", "ealstm_gbmblend_s1"}
    np.testing.assert_allclose(bl["ealstm_gbmblend_s0"]["temp_c_mean"], 12.0)
    ens, members = seed_ensembles({**preds, **bl})
    np.testing.assert_allclose(ens["ealstm_gbmblend"]["temp_c_mean"], 13.0)
    assert members["ealstm_gbmblend"] == ["ealstm_gbmblend_s0", "ealstm_gbmblend_s1"]
    # log target: blend is the geometric mean (model space = log(x + 0.01))
    n = seed_blends({"ealstm_s0": {"no3no2_mgl_mean": np.array([[0.99]], np.float32)},
                     "forcing_gbm_bag_s0": {"no3no2_mgl_mean": np.array([[99.99]], np.float32)}})
    np.testing.assert_allclose(n["ealstm_gbmblend_s0"]["no3no2_mgl_mean"], 9.99, rtol=1e-4)
    assert seed_blends({"ealstm_s0": {"temp_c_mean": a0}}) == {}


def test_ealstm_refit_without_inner_runs_fixed_epochs_and_keeps_last():
    """Protocol v0.8 refit: no inner sites -> exactly max_epochs epochs, never early-stopped, last state kept."""
    pytest.importorskip("torch")
    import torch

    from sentinel.models.hydromet.lstm import EALSTMConfig, HydroMetEALSTM

    fit, inner = D.inner_split(data_ := synth(), 0.2, 0)
    cfg = EALSTMConfig(seq_len=60, predict_last=20, hidden=8, static_emb=4, batch_size=8, max_epochs=3,
                       steps_per_epoch=2, val_max_windows=10, patience=1)
    m = HydroMetEALSTM(cfg)
    m.fit(data_, np.r_[fit, inner], np.zeros(0, int), device="cpu", log=lambda *_: None)
    assert len(m.history) == 3 and m.info["n_inner_sites"] == 0 and m.info["n_fit_sites"] == len(fit) + len(inner)
    m2 = HydroMetEALSTM(dataclasses_replace(cfg, max_epochs=2))
    m2.fit(data_, np.r_[fit, inner], np.zeros(0, int), device="cpu", log=lambda *_: None)
    # same seed: epoch-2 run differs from the 3-epoch run's final state (the last epoch is kept, not the first)
    assert any(not torch.equal(m.state[k], m2.state[k]) for k in m.state)


def dataclasses_replace(obj, **kw):
    import dataclasses

    return dataclasses.replace(obj, **kw)


def test_compact_static_set():
    st = pd.DataFrame({c: [1.0, 2.0] for c in D.COMPACT_STATIC + ("pctcrop2019cat", "iwi")})
    out = D.apply_static_ablation(st, "compact")
    assert list(out.columns) == list(D.COMPACT_STATIC) and "elev_m" in out and "iwi" not in out
    with pytest.raises(ValueError):
        D.apply_static_ablation(st.drop(columns=["bfiws"]), "compact")
