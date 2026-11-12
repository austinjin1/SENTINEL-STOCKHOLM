# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Unit tests (synthetic data only) for the SENTINEL-Sonde v2 components: NWP alignment, bias correction, error
model, GBM-NWP features, physics features, quantile interpolation, networks (R-NH, SONDE-HF, masked mean, CMAL),
the E2/E2-B weighting rule and the Chronos-2-cov input builder."""

from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import torch

from sentinel.models.sonde_v2 import ensemble as E
from sentinel.models.sonde_v2 import nets as N
from sentinel.models.sonde_v2 import nwp as NWP
from sentinel.models.sonde_v2 import physics_features as PF
from sentinel.models.sonde_v2.gbm_nwp import interp_quantiles

DATES = pd.date_range("2020-01-01", periods=400, freq="D")


# ----------------------------------------------------------------------------- NWP

def test_from_daily_table_alignment():
    df = pd.DataFrame({"site_key": ["USGS-1", "USGS-1"], "date_local": ["2020-01-10", "2020-01-10"],
                       "lead_day": [1, 3], "tmean_c": [5.0, 7.0], "tmax_c": [9, 11], "tmin_c": [1, 3],
                       "precip_mm": [0.0, 2.0], "srad_wm2": [100.0, 90.0]})
    t = NWP.from_daily_table(df, ["1", "2"], DATES, name="gfs")
    i9, i7 = DATES.get_loc(pd.Timestamp("2020-01-09")), DATES.get_loc(pd.Timestamp("2020-01-07"))
    assert t.x[0, i9, 0, NWP.F_IDX["tmean_c"]] == 5.0          # issue 01-09, lead 1 → valid 01-10
    assert t.x[0, i7, 2, NWP.F_IDX["tmean_c"]] == 7.0          # issue 01-07, lead 3 → valid 01-10
    assert np.isnan(t.x[1]).all() and t.m.sum() == 2


def test_observed_future_alignment():
    w = np.arange(10, dtype=np.float32).reshape(1, 10, 1)
    f = NWP.observed_future(w, 3)
    assert f[0, 0, 0, 0] == 1 and f[0, 0, 2, 0] == 3 and np.isnan(f[0, 8, 1, 0])


def _obs_fc(S=4, seed=0):
    rs = np.random.default_rng(seed)
    obs = rs.normal(10, 5, (S, len(DATES), 2, len(NWP.FIELDS))).astype(np.float32)
    obs[..., NWP.F_IDX["precip_mm"]] = np.abs(obs[..., NWP.F_IDX["precip_mm"]])
    fc = (obs - 2.0) / 0.8                                        # obs = 2 + 0.8 fc
    fc[..., NWP.F_IDX["precip_mm"]] = obs[..., NWP.F_IDX["precip_mm"]] * 2   # obs = 0.5 fc
    return obs, fc.astype(np.float32)


def test_bias_map_recovers_linear_map_from_fit_window_only():
    obs, fc = _obs_fc()
    fit = np.zeros(len(DATES), bool)
    fit[:250] = True
    fc_bad = fc.copy()
    fc_bad[:, ~fit] += 100.0                                     # outside the fit window: must not matter
    bm = NWP.fit_bias_map(fc_bad, obs, DATES, fit, groups=np.zeros(4), min_days=10)
    t = NWP.F_IDX["tmean_c"]
    assert np.allclose(bm.b[:, :, t], 0.8, atol=1e-3) and np.allclose(bm.a[:, :, t], 2.0, atol=1e-2)
    assert np.allclose(bm.b[:, :, NWP.F_IDX["precip_mm"]], 0.5, atol=1e-3)
    corr = bm.apply(fc, DATES)
    assert np.allclose(corr[..., t], obs[..., t], atol=1e-2)
    assert (corr[..., NWP.F_IDX["precip_mm"]] >= 0).all()


def test_error_model_fit_and_sample():
    rs = np.random.default_rng(1)
    obs = rs.normal(10, 3, (20, len(DATES), 3, len(NWP.FIELDS))).astype(np.float32)
    fc = obs + rs.normal(0.5, 1.5, obs.shape).astype(np.float32)
    em = NWP.fit_error_model(fc, obs, DATES, np.ones(len(DATES), bool), L=5)
    t = NWP.F_IDX["tmean_c"]
    assert abs(em.mean[0, t].mean() - 0.5) < 0.1 and abs(em.sd[0, t].mean() - 1.5) < 0.1
    assert em.sd[4, t].mean() > em.sd[2, t].mean()             # leads past the product horizon are inflated
    pp = em.sample(NWP.observed_future(obs[:, :, 0], 5), DATES, np.random.default_rng(0))
    assert pp.shape == (20, len(DATES), 5, len(NWP.FIELDS))


def test_gbm_nwp_blocks_values():
    x = np.full((1, 2, 7, len(NWP.FIELDS)), np.nan, np.float32)
    x[0, 0, :, NWP.F_IDX["tmean_c"]] = [1, 2, 3, 4, 5, 6, 7]
    x[0, 0, :, NWP.F_IDX["precip_mm"]] = [0, 1, 0, 2, 0, 0, 0]
    b = NWP.gbm_nwp_blocks(x, 3, base_c=1.5, decay=0.5)
    assert b["nwp_dd_cum"][0, 0] == pytest.approx(0.5 + 1.5)
    # exponentially weighted FORECAST DEGREE-DAYS (stamped feature; lead ruling M), weights decay^(hh - lead)
    assert b["nwp_dd_ew"][0, 0] == pytest.approx(0.0 * 0.25 + 0.5 * 0.5 + 1.5) and "nwp_tmean_ew" not in b
    assert b["nwp_p_sum"][0, 0] == 1 and b["nwp_tmean_target"][0, 0] == 3
    assert np.isnan(b["nwp_p_sum"][0, 1]) and b["nwp_avail_t"][0, 1] == 0
    # product missing: NaN, never 0 (verifier item 6), for every trajectory summary
    for k in ("nwp_dd_cum", "nwp_dd_ew", "nwp_p_max", "nwp_tmean_mean", "nwp_sw_mean", "nwp_tmean_target"):
        assert np.isnan(b[k][0, 1]), k
    assert set(k for k in b if k.startswith("nwp_tmean_l")) == {"nwp_tmean_l1", "nwp_tmean_l2", "nwp_tmean_l3"}
    b14 = NWP.gbm_nwp_blocks(x, 14)
    assert b14["nwp_tmean_target"][0, 0] == 7                  # beyond the product horizon: last lead
    assert "nwp_tmean_l7" in b14 and "nwp_tmean_l8" not in b14  # "leads 1..h" = leads 1..min(h, 7)
    assert b14["nwp_dd_cum"][0, 0] == pytest.approx(sum(range(1, 8)))


def test_masked_mean_products():
    a = np.array([1.0, np.nan, 3.0])
    b = np.array([3.0, np.nan, np.nan])
    x, n = NWP.masked_mean_products([a, b])
    assert x[0] == 2 and np.isnan(x[1]) and x[2] == 3 and list(n) == [2, 0, 1]


# ----------------------------------------------------------------------------- physics features

def test_physics_features():
    tw = PF.air2stream_forecast(np.array([5.0]), np.full((1, 60), 20.0))
    assert tw[0, -1] == pytest.approx(PF.A0 + PF.A1 * 20, abs=1e-3) and tw[0, 0] > 5
    assert PF.dosat_forecast(np.array([[20.0]]), np.array([0.0]))[0, 0] == pytest.approx(9.09, abs=0.05)
    assert PF.sc_dilution(np.array([500.0]), np.array([100.0]), np.array([0.0]), np.array([50.0]), 3)[0] == 500
    assert PF.sc_dilution(np.array([500.0]), np.array([100.0]), np.array([30.0]), np.array([50.0]), 3)[0] < 500
    assert np.isnan(PF.k600_raymond(np.array([np.nan]), np.array([0.01]))[0])
    assert PF.k600_raymond(np.array([100.0]), np.array([0.01]))[0] > 0


def test_interp_quantiles():
    from scipy.stats import norm

    levels = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
    qk = norm.ppf(levels)[None, :] * 2 + 1                    # exact normal(1, 2) quantiles at the knots
    q19 = interp_quantiles(qk)
    lv19 = np.round(np.arange(0.05, 0.951, 0.05), 2)
    assert np.all(np.diff(q19, axis=1) >= 0)
    assert np.allclose(q19[0], norm.ppf(lv19) * 2 + 1, atol=1e-9)   # linear in z-space reproduces a normal


# ----------------------------------------------------------------------------- networks

def _nwp_inputs(B, L, fields=(5, 3)):
    xs = [torch.randn(B, L, f) for f in fields]
    ms = [torch.ones(B, L, dtype=torch.bool), torch.zeros(B, L, dtype=torch.bool)]
    return xs, ms


def test_masked_mean_ignores_masked_products():
    torch.manual_seed(0)
    mm = N.MaskedMeanNWP([5, 3], d=8)
    xs, ms = _nwp_inputs(2, 4)
    a = mm(xs, ms)
    xs2 = [xs[0], xs[1] * 100 + float("nan")]
    b = mm(xs2, ms)
    assert torch.allclose(a, b)
    none = mm(xs, [torch.zeros(2, 4, dtype=torch.bool)] * 2)
    assert torch.allclose(none, mm.none.expand_as(none))


def test_monotone_head_and_cmal_quantiles():
    torch.manual_seed(0)
    h = N.MonotoneQuantileHead(16, 3)
    q = h(torch.randn(5, 16))
    assert q.shape == (5, 3, 19) and (q.diff(dim=-1) >= 0).all()
    par = {"logit": torch.zeros(1, 1, 1), "mu": torch.zeros(1, 1, 1), "b": torch.ones(1, 1, 1),
           "tau": torch.full((1, 1, 1), 0.5)}
    qq = N.CMALHead.quantiles(par)                              # symmetric Laplace(0, scale b/(τ(1-τ))=... )
    assert torch.allclose(N.CMALHead.cdf({k: v[..., None, :] for k, v in par.items()}, qq), N.TAUS[None, None],
                          atol=1e-4)
    assert (qq.diff(dim=-1) > 0).all()
    y = torch.tensor([[0.3]])
    assert torch.isfinite(N.CMALHead.nll(par, y))


def test_rnh_forward_shapes_and_backward():
    torch.manual_seed(0)
    for head in ("cmal", "quantile"):
        m = N.RNH(n_dyn=6, n_static=4, n_huc=3, nwp_fields=[5, 3], L=7, hidden=16, head=head)
        xs, ms = _nwp_inputs(4, 7)
        out = m(torch.randn(4, 30, 6), torch.randn(4, 4), torch.zeros(4, dtype=torch.long), xs, ms,
                torch.randn(4, 7, 4))
        q = m.quantiles(out)
        assert q.shape == (4, 7, 1, 19) and (q.diff(dim=-1) >= -1e-5).all()
        y = torch.randn(4, 7, 1)
        loss = N.CMALHead.nll(out, y) if head == "cmal" else N.pinball(q, y)
        loss.backward()


@pytest.mark.parametrize("n_hourly", [0, 3])
def test_sonde_hf_constraints(n_hourly):
    torch.manual_seed(0)
    targets = ["do_mean", "do_min", "do_max", "hyp_hours"]
    m = N.SondeHF(n_dyn=6, n_static=4, n_huc=3, nwp_fields=[5, 3], targets=targets, n_hourly=n_hourly, L=7,
                  hidden_d=16, hidden_h=8, hidden_f=16, order=[("do_min", "do_mean", -1), ("do_max", "do_mean", 1)],
                  bounded={"hyp_hours": (0.0, 24.0)}, anchor_targets=["do_mean"])
    B = 5
    xs, ms = _nwp_inputs(B, 7)
    cm = torch.rand(B, 7, 4) * 5 + 5
    cs = torch.rand(B, 7, 4) + 0.5
    xh = torch.randn(B, 336, n_hourly) if n_hourly else None
    q = m(torch.randn(B, 40, 6), torch.randn(B, 4), torch.zeros(B, dtype=torch.long), xs, ms, torch.randn(B, 7, 4),
          x_hourly=xh, has_hourly=torch.tensor([True, False, True, True, False]) if n_hourly else None,
          a_last=torch.randn(B, 1), clim_mean=cm, clim_sd=cs)
    assert q.shape == (B, 7, 4, 19)
    y = cm[..., None] + cs[..., None] * q                       # original units
    assert (y[:, :, 1] <= y[:, :, 0] + 1e-4).all() and (y[:, :, 2] >= y[:, :, 0] - 1e-4).all()
    assert (y[:, :, 3] >= -1e-4).all() and (y[:, :, 3] <= 24 + 1e-4).all()
    assert (q.diff(dim=-1) >= -1e-4).all()
    N.pinball(q, torch.randn(B, 7, 4)).backward()


# ----------------------------------------------------------------------------- ensemble rule

def test_fit_weights_prefers_better_member():
    rs = np.random.default_rng(0)
    y = rs.normal(0, 1, 4000)
    lv = np.round(np.arange(0.05, 0.951, 0.05), 2)
    from scipy.stats import norm
    good = (y * 0.0)[:, None] + norm.ppf(lv)[None, :]              # calibrated N(0,1)
    bad = good + 3.0
    w = E.fit_weights(np.stack([bad, good]), y)
    assert w.sum() == pytest.approx(1.0) and (w >= 0).all() and w[1] > 0.9


def test_rule_identical_for_e2_and_e2b_and_cross_fit():
    rs = np.random.default_rng(1)
    oof = {("t", h): rs.normal(0, 1, (3, 300, 19)).cumsum(-1) * 0.1 for h in (1, 3, 10)}
    y = {k: rs.normal(0, 1, 300) for k in oof}
    fold = {k: np.arange(300) % 5 for k in oof}
    w1 = E.fit_rule(oof, y, "W1")
    w2 = E.fit_rule(oof, y, "W1")
    assert all(np.allclose(w1[k], w2[k]) for k in w1) and len(w1) == 3
    sc = E.cross_fitted_score(oof, y, fold, "W2")
    assert set(sc) == set(oof) and all(np.isfinite(v) for v in sc.values())


# ----------------------------------------------------------------------------- Chronos-2-cov input builder

def test_chronos_cov_appends_known_future_rows():
    from sentinel.models.sonde.chronos import ChronosConfig, ChronosForecaster
    from sentinel.models.sonde_v2.chronos_cov import ChronosCovForecaster

    pipe = SimpleNamespace(quantiles=sorted(set(round(float(q), 4)
                                                for q in np.r_[np.round(np.arange(0.05, 0.951, 0.05), 2), 0.01])))
    cfg = ChronosConfig(mode="multi", context=8)
    S, T = 1, 30
    nwp_x = np.zeros((S, T, 7, len(NWP.FIELDS)), np.float32)
    nwp_x[..., NWP.F_IDX["tmean_c"]] = 12.0
    f = ChronosCovForecaster(pipe, cfg, nwp_x=nwp_x)
    base = {"context": torch.zeros(2, 8), "future_covariates": torch.full((2, 14), float("nan")),
            "n_targets": 2, "n_covariates": 0, "n_future_covariates": 0}
    data = SimpleNamespace(weather=np.ones((S, T, 6), np.float32))
    with mock.patch.object(ChronosForecaster, "prepared_inputs", return_value=([dict(base)], [["a", "b"]])):
        inputs, tv = f.prepared_inputs(data, 0, np.array([20]), ["a", "b"])
    inp = inputs[0]
    assert inp["context"].shape == (5, 8) and inp["future_covariates"].shape == (5, 14)
    assert inp["n_covariates"] == 3 and inp["n_future_covariates"] == 3
    assert float(inp["future_covariates"][2, 0]) == 12.0 and torch.isnan(inp["future_covariates"][2, 7])


def test_reference_window_climatology_uses_first_two_years_only():
    from sentinel.models.sonde.data import Cube
    from sentinel.models.sonde_v2.prep import reference_window_climatology

    dates = pd.date_range("2010-01-01", "2016-12-31", freq="D")
    doy = dates.dayofyear.to_numpy()
    y = np.full((4, len(dates), 1), np.nan, np.float32)
    y[0, :, 0] = 10 + 5 * np.sin(2 * np.pi * doy / 365.25)
    y[0, dates >= "2012-01-01", 0] += 100.0                      # after the 2-year window: must be ignored
    y[1, :200, 0] = 1.0                                          # < 365 observed days: no climatology
    y[2, -729:, 0] = 3.0                                         # window one day longer than the loaded dates
    y[3, -730:, 0] = 3.0                                         # window ends on the last loaded date
    cube = Cube(["a", "b", "c", "d"], dates, ["temp_c_mean"], y, np.zeros_like(y, dtype=bool))
    from sentinel.models.sonde_v2.prep import NEVER, reference_window
    m, s, start = reference_window(cube)
    assert np.isnan(m[2]).all() and start[2, 0] == NEVER         # never a truncated window (dev rulings F(3))
    assert np.isfinite(m[3]).all() and start[3, 0] == len(dates) + 42      # climatology, but no issue day loaded
    assert np.nanmax(m[0]) < 16 and np.nanmin(m[0]) > 4 and np.isnan(m[1]).all()
    assert abs(m[0, 90, 0] - (10 + 5 * np.sin(2 * np.pi * 91 / 365.25))) < 0.5


def test_hourly_targets_and_naive_references():
    from sentinel.models.sonde_v2 import hourly as HR

    d0 = pd.Timestamp("2015-01-01")
    T = 800
    hrs = np.arange(T * 24)
    x = (8 + 2 * np.sin(2 * np.pi * (hrs % 24) / 24)).astype(np.float32)
    x[500 * 24 + 3: 500 * 24 + 6] = 2.0                         # 3 hypoxic hours on day 500
    clim = HR.ref_window_hourly_clim(x, d0)
    assert clim is not None and np.isfinite(clim[0]).all()
    days = np.array([100, 499])
    y = HR.targets_at(x, days)
    assert y.shape == (2, 5) and y[0, 0] == x[101 * 24 + 5]
    hh = HR.hypoxia_hours(x, days, thr=3.0)
    assert hh[1, 0] == 3.0 and hh[0, 0] == 0.0
    diel = HR.naive_point("diel", x, clim, d0, np.array([100]))
    assert np.allclose(diel, HR.targets_at(x, np.array([100])), atol=0.2)   # periodic signal: diel ≈ truth
    pers = HR.naive_point("persistence", x, clim, d0, np.array([100]))
    assert np.allclose(pers, x[101 * 24 - 1])


# ============================================================================= v2 drivers and rules (synthetic)

def _script(name):
    """Import ``scripts/train/<name>.py`` as a module (nothing runs at import)."""
    import importlib.util
    import sys
    from pathlib import Path

    if name in sys.modules:
        return sys.modules[name]
    p = Path(__file__).resolve().parents[1] / "scripts" / "train" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


SYN_VARS = ["temp_c_mean", "do_mgl_mean", "do_mgl_min", "discharge_cfs_mean", "spcond_uscm_mean", "do_pctsat"]


def _synth_data(seed=0):
    """10 sites (8 train, 2 val; 5 basin pairs), 2007-10-01..2023-09-30, AR(1) series with a seasonal cycle.
    Site 1: nothing before 2015-01-01. Site 2: temperature has 100 observed days in its first 730 (no reference
    climatology), data again from day 1000. Site 3: no DO."""
    from scipy.signal import lfilter

    from sentinel.data.sonde_splits import SondeSplit
    from sentinel.models.sonde import common
    from sentinel.models.sonde.data import Cube
    from sentinel.models.sonde_v2 import prep as PR

    rng = np.random.default_rng(seed)
    dates = pd.date_range("2007-10-01", "2023-09-30", freq="D")
    S, T = 10, len(dates)
    sites = [f"{i:08d}" for i in range(1, S + 1)]
    seas = np.cos(2 * np.pi * (dates.dayofyear.to_numpy() - 200) / 365.25)

    def ar(sd):
        return lfilter([1.0], [1.0, -0.9], rng.normal(0, sd, (S, T)), axis=1)

    temp = 13 + 8 * seas[None] + ar(0.4) + rng.normal(0, 1, (S, 1))
    do = 14.6 - 0.33 * temp + ar(0.1)
    q = np.exp(4 + ar(0.15) + rng.normal(0, 0.5, (S, 1)))
    sc = 400 * (q / q.mean(axis=1, keepdims=True)) ** -0.3 * np.exp(ar(0.02))
    y = np.stack([temp, do, do - 0.8 - np.abs(ar(0.05)), q, sc, 100 * do / (14.6 - 0.33 * temp + 0.3)],
                 axis=2).astype(np.float32)
    t15 = dates.get_loc(pd.Timestamp("2015-01-01"))
    y[1, :t15] = np.nan
    y[2, 100:1000, 0] = np.nan
    y[3, :, [1, 2, 5]] = np.nan
    cube = Cube(sites, dates, list(SYN_VARS), y, np.zeros(y.shape, bool))
    role = {s: ("val" if i >= 8 else "train") for i, s in enumerate(sites)}
    basin = {s: f"B{i // 2}" for i, s in enumerate(sites)}
    split = SondeSplit(role, basin)
    static = pd.DataFrame({"lat": np.linspace(30, 45, S), "lon": np.linspace(-120, -75, S),
                           "huc2": [f"{1 + i // 5:02d}" for i in range(S)], "elev_m": np.linspace(0, 900, S),
                           "log_drain_area_km2": np.linspace(3, 8, S), "slope": np.linspace(0.001, 0.02, S)},
                          index=sites)
    data = PR.SondeDataV2(cube, None, None, np.array([role[s] for s in sites]), static, split,
                          {"temporary": True, "final": False}, {}, ("train", "val"), common.HORIZONS)
    PR.apply_reference_window_climatology(data)
    return data


@pytest.fixture(scope="module")
def synth_v2():
    return _synth_data()


def test_reference_window_issue_rule(synth_v2):
    from sentinel.models.sonde_v2 import prep as PR

    data, c = synth_v2, synth_v2.cube
    jt, jd = c.var_index("temp_c_mean"), c.var_index("do_mgl_mean")
    t15 = c.dates.get_loc(pd.Timestamp("2015-01-01"))
    assert (PR.REF_DAYS, PR.REF_EMBARGO, PR.REF_MIN_OBS) == (730, 42, 365)
    assert data.ref_issue_start[0, jt] == 772                     # first observation on the cube start
    assert data.ref_issue_start[1, jt] == t15 + 772               # late first observation
    assert data.ref_issue_start[2, jt] == PR.NEVER                # < 365 observed days in the first 730
    assert data.ref_issue_start[3, jd] == PR.NEVER                # never observed
    assert not np.isfinite(data.clim_mean[2, :, jt]).any()
    assert not data.issue_ok("temp_c_mean", np.array([0]), np.array([771]))[0]
    assert data.issue_ok("temp_c_mean", np.array([0]), np.array([772]))[0]
    for block in ("train", "val"):
        r = data.rows("temp_c_mean", block)
        assert len(r) and (r.ti >= data.ref_issue_start[r.si, jt]).all() and 2 not in set(r.si.tolist())
        assert all(m.shape == r.si.shape for m in r.score_mask.values())
    r = data.rows("temp_c_mean", "train")
    assert r.ti[r.si == 0].min() == 772 and r.ti[r.si == 1].min() == t15 + 772
    n = data.n_dropped_ref_window[("temp_c_mean", "train")]
    assert n["rows"] >= 2 * 700 and set(n["scored"]) == set(data.horizons)   # sites 0 and 4..: first 771 issue days
    assert 3 not in set(data.rows("do_mgl_mean", "train").si.tolist())


def test_do_pctsat_transform():
    from sentinel.models.sonde_v2 import prep as PR

    q = np.array([[4.545, 9.09, 10.0], [9.09, 9.09, 9.09]])
    out = PR.do_pctsat_from_quantiles(q, np.array([20.0, 20.0]), np.array([0.0, np.nan]))
    assert out.shape == (2, 3) and out[0, 1] == pytest.approx(100, abs=0.6) and out[0, 0] == pytest.approx(50, abs=0.3)
    assert out[1, 1] == pytest.approx(out[0, 1])                  # elevation NaN → 0 m
    assert (np.diff(out, axis=1) >= 0).all()                      # monotone in the DO quantiles
    hi = PR.do_pctsat_from_quantiles(q[:1], np.array([20.0]), np.array([2000.0]))
    assert hi[0, 1] > out[0, 1]                                   # lower pressure → lower saturation → higher %


def test_k600_broadcasts_site_slope():
    q = np.full((3, 5), 100.0)
    q[0, 0] = np.nan
    k = PF.k600_raymond(q, np.array([0.01, 0.0, 0.02])[:, None])
    assert k.shape == (3, 5) and np.isnan(k[0, 0]) and np.isnan(k[1]).all() and (k[2] > k[0, 1]).all()


def test_bias_map_valid_end_excludes_pairs_past_the_fit_window():
    obs, fc = _obs_fc()
    fit = np.zeros(len(DATES), bool)
    fit[:250] = True
    end = DATES[250]                                              # lead 2 of issue day 249 is valid on day 251
    late = (np.arange(len(DATES))[:, None] + np.arange(1, 3)[None, :]) > 250
    obs_bad = obs.copy()
    obs_bad[:, late] += 50.0                                      # pairs valid after ``end``: must not matter
    t = NWP.F_IDX["tmean_c"]
    bm = NWP.fit_bias_map(fc, obs_bad, DATES, fit, groups=np.zeros(4), min_days=10, valid_end=end)
    assert np.allclose(bm.b[:, :, t], 0.8, atol=1e-3) and np.allclose(bm.a[:, :, t], 2.0, atol=1e-2)
    bm0 = NWP.fit_bias_map(fc, obs_bad, DATES, fit, groups=np.zeros(4), min_days=10)
    mo = DATES[249].month - 1
    assert not np.allclose(bm0.a[:, 1, t, mo], 2.0, atol=1e-2)    # without the rule the late pair leaks in


def test_load_gefs_requires_every_month(tmp_path, synth_v2):
    from sentinel.models.sonde_v2 import prep as PR

    assert len(PR.month_range("200710", "201909")) == 144 and PR.month_range("201911", "202002")[-1] == "202002"
    d = tmp_path / "reforecast"
    d.mkdir()
    s0 = synth_v2.sites[0]
    row = {"site": [s0, "99999999"], "issue": pd.to_datetime(["2008-01-05", "2008-01-05"])}
    for k in range(1, 8):
        row.update({f"tmax_l{k}": [200 + k, 1], f"tmin_l{k}": [100, 1], f"p_l{k}": [-32768, 1], f"sw_l{k}": [150, 1]})
    pd.DataFrame(row).to_parquet(d / "200801.parquet")
    x = PR.load_gefs(synth_v2, "reforecast", months=("200801", "200801"), root=tmp_path)
    ti = synth_v2.cube.dates.get_loc(pd.Timestamp("2008-01-05"))
    assert x.shape == (10, len(synth_v2.cube.dates), 7, len(NWP.FIELDS))
    assert x[0, ti, 2, NWP.F_IDX["tmax_c"]] == pytest.approx(20.3) and x[0, ti, 0, NWP.F_IDX["tmin_c"]] == 10.0
    assert x[0, ti, 2, NWP.F_IDX["tmean_c"]] == pytest.approx(15.15) and x[0, ti, 0, NWP.F_IDX["srad_wm2"]] == 150
    assert np.isnan(x[0, ti, :, NWP.F_IDX["precip_mm"]]).all()    # −32768 = missing, not −3276.8
    assert np.isfinite(x).sum() == 7 * 4                          # unknown site ignored, nothing else filled
    with pytest.raises(FileNotFoundError, match="1 of 2 required month files missing"):
        PR.load_gefs(synth_v2, "reforecast", months=("200801", "200802"), root=tmp_path)


# ----------------------------------------------------------------------------- stamp guard (verifier item F04)

def _tmp_repo(tmp_path):
    """A tiny repo: three v2 code files, the deviations file with their sha256 list (read-only), a registry."""
    import os

    from sentinel.models.sonde_v2 import parity

    root = tmp_path / "repo"
    for rel, txt in (("sentinel/models/sonde_v2/a.py", "A = 1\n"), ("scripts/train/sonde_v2_x.py", "X = 1\n"),
                     ("tests/test_sonde_v2.py", "T = 1\n")):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(txt)
    dev = root / parity.DEV_DIR / parity.DEV_MAIN
    dev.parent.mkdir(parents=True)
    dev.write_text(f"# deviations\n\n{parity.MARK_BEGIN}\n{parity.code_list_text(root)}\n{parity.MARK_END}\n")
    os.chmod(dev, 0o444)
    return root, dev, tmp_path / "registry.jsonl"


def _stamp_row(root, reg, **over):
    import hashlib
    import json

    from sentinel.models.sonde_v2 import parity

    files = sorted((root / parity.DEV_DIR).glob(parity.DEV_GLOB))
    extra = {"sha256": {str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
             "written_by": "lead", "code_sha256": parity.code_snapshot_sha256(root)}
    row = {"id": over.pop("id", "row1"), "workstream": over.pop("workstream", "M1-sonde"),
           "experiment": "sonde/prereg_snapshot", "extra": {**extra, **over}}
    with open(reg, "a") as fh:
        fh.write(json.dumps(row) + "\n")


def test_stamp_guard_states(tmp_path):
    import os

    from sentinel.models.sonde_v2 import parity

    def st(root, reg):
        return parity.stamp_status(repo_root=root, registry_path=reg)

    empty = tmp_path / "empty"
    empty.mkdir()
    assert "does not exist" in st(empty, tmp_path / "r.jsonl")["reason"]
    root, dev, reg = _tmp_repo(tmp_path)
    assert st(root, reg)["reason"] == "registry not found"
    _stamp_row(root, reg, id="a", written_by="M1")                 # not written by the lead
    _stamp_row(root, reg, id="b", code_sha256="0" * 64)            # other code state
    _stamp_row(root, reg, id="c", workstream="M6-hydro")           # other workstream
    _stamp_row(root, reg, id="d", sha256={str(dev.relative_to(root)): "0" * 64})   # other file state
    s = st(root, reg)
    assert not s["stamped"] and "no lead-written" in s["reason"]
    with pytest.raises(parity.StampError, match="refusing: unit test"):
        parity.require_stamp("unit test", repo_root=root, registry_path=reg)
    _stamp_row(root, reg, id="ok")
    s = parity.require_stamp("unit test", repo_root=root, registry_path=reg)
    assert s["stamped"] and s["row_id"] == "ok" and s["code_sha256"] == parity.code_snapshot_sha256(root)
    os.chmod(dev, 0o644)                                           # writable deviations file
    assert "writable" in st(root, reg)["reason"]
    os.chmod(dev, 0o444)
    code = root / "sentinel/models/sonde_v2/a.py"
    code.write_text("A = 2\n")                                     # edited code file
    assert "changed or missing: ['sentinel/models/sonde_v2/a.py']" in st(root, reg)["reason"]
    code.write_text("A = 1\n")
    assert st(root, reg)["stamped"]
    new = root / "scripts/train/sonde_v2_m1h.py"
    new.write_text("Y = 1\n")                                      # new unlisted code file
    assert "unlisted: ['scripts/train/sonde_v2_m1h.py']" in st(root, reg)["reason"]
    # a later dated deviation file lists the new file; it needs its own sha256 and the new code hash in a lead row
    later = root / parity.DEV_DIR / "sonde_arch_v2_deviations_v2_a.md"
    later.write_text(f"{parity.MARK_BEGIN}\n{parity.code_list_text(root)}\n{parity.MARK_END}\n")
    os.chmod(later, 0o444)
    assert "no lead-written" in st(root, reg)["reason"]
    _stamp_row(root, reg, id="ok2")
    assert st(root, reg)["row_id"] == "ok2"
    os.chmod(dev, 0o644)
    dev.write_text(dev.read_text() + "edit\n")                     # edited deviations file
    os.chmod(dev, 0o444)
    assert not st(root, reg)["stamped"]


def test_stamp_guard_needs_code_list(tmp_path):
    import os

    from sentinel.models.sonde_v2 import parity

    root, dev, reg = _tmp_repo(tmp_path)
    os.chmod(dev, 0o644)
    dev.write_text("# deviations without a code list\n")
    os.chmod(dev, 0o444)
    _stamp_row(root, reg)
    assert "no code sha256 list" in parity.stamp_status(repo_root=root, registry_path=reg)["reason"]


@pytest.fixture
def tmp_run(tmp_path, monkeypatch):
    """Redirect repo root, registry and run roots of parity and both drivers to a temporary directory."""
    from sentinel.core import paths
    from sentinel.models.sonde_v2 import parity

    cv, s1 = _script("sonde_v2_cv"), _script("sonde_v2_step1")
    root, dev, reg = _tmp_repo(tmp_path)
    runs = tmp_path / "runs"
    monkeypatch.setattr(paths, "REPO_ROOT", root)
    monkeypatch.setattr(paths, "REGISTRY_PATH", reg)
    monkeypatch.setattr(parity, "LOG", runs / "v2_cv" / "PARITY_LOG.jsonl")
    monkeypatch.setattr(parity, "DECLARATION", runs / "v2_cv" / "DECLARATION.json")
    monkeypatch.setattr(cv, "CV", runs / "v2_cv")
    monkeypatch.setattr(cv, "FOLDS", runs / "v2_cv" / "FOLDS.json")
    monkeypatch.setattr(s1, "RUN", runs / "v2_step1")

    def never(*a, **k):
        raise AssertionError("data was loaded without a stamp/declaration")

    monkeypatch.setattr(cv, "load_data", never)
    monkeypatch.setattr(s1, "setup", never)
    return SimpleNamespace(root=root, dev=dev, reg=reg, runs=runs, cv=cv, s1=s1)


def _args(arm="sonde_hf", config="C1", fold=0, seed=0, target=None):
    return SimpleNamespace(arm=arm, config=config, fold=fold, seed=seed, target=target)


def test_every_driver_phase_refuses_without_stamp(tmp_run):
    from sentinel.models.sonde_v2 import parity

    cv, s1 = tmp_run.cv, tmp_run.s1
    calls = [cv.phase_declare, cv.phase_folds, lambda: cv.phase_parity(_args()), lambda: cv.phase_combine(_args()),
             s1.phase_declare, s1.phase_hf, s1.phase_gbm, s1.phase_eval]
    calls += [lambda a=a, c=c: cv.phase_run(_args(a, c)) for a, cs in cv.D.CONFIGS.items() for c in cs]
    for f in calls:
        with pytest.raises(parity.StampError):
            f()
    assert not tmp_run.runs.exists() and not tmp_run.reg.exists()   # nothing written, no registry row


def test_drivers_refuse_without_declaration_and_declare_once(tmp_run, monkeypatch):
    import json

    from sentinel.models.sonde_v2 import declared as D
    from sentinel.models.sonde_v2 import parity

    cv, s1 = tmp_run.cv, tmp_run.s1
    _stamp_row(tmp_run.root, tmp_run.reg)
    for f in (lambda: cv.phase_run(_args()), lambda: cv.phase_combine(_args()), s1.phase_hf, s1.phase_gbm,
              s1.phase_eval):
        with pytest.raises(parity.ParityError, match="no declaration record"):
            f()
    assert cv.phase_declare() == 0
    rec = json.loads(parity.DECLARATION.read_text())
    assert rec["sha256"] == D.declaration_sha256() and rec["stamp_row"] == "row1"
    assert not (parity.DECLARATION.stat().st_mode & 0o222)
    rows = [json.loads(ln) for ln in tmp_run.reg.read_text().splitlines()]
    assert rows[-1]["experiment"] == "sonde_arch_v2/declaration" and rows[-1]["extra"]["sha256"] == rec["sha256"]
    assert parity.read_log()[0]["run"] == "DECLARATION"
    n = len(rows)
    assert s1.phase_declare() == 0 and len(tmp_run.reg.read_text().splitlines()) == n    # idempotent
    assert parity.require_declared("sonde_hf", "C1") == rec["sha256"]
    with pytest.raises(parity.ParityError, match="not declared for arm"):
        parity.require_declared("sonde_hf", "C9")
    with pytest.raises(SystemExit, match="seed 3 is not declared"):
        cv.phase_run(_args(seed=3))
    # GBM arms and step 1 stay closed while the GBM settings are not ruled
    monkeypatch.setitem(D.GBM_SETTINGS, "ruled", False)
    monkeypatch.setattr(parity, "require_declared", lambda *a, **k: rec["sha256"])
    for f in (lambda: cv.phase_run(_args("gbm_nwp", "Q19")), lambda: cv.phase_run(_args("pf_gbm", "LS")),
              s1.phase_hf, s1.phase_gbm, s1.phase_eval):
        with pytest.raises(RuntimeError, match="not ruled"):
            f()
    monkeypatch.undo()


def test_changed_configuration_list_closes_the_drivers(tmp_run, monkeypatch):
    from sentinel.models.sonde_v2 import declared as D
    from sentinel.models.sonde_v2 import parity

    _stamp_row(tmp_run.root, tmp_run.reg)
    parity.declare(log=lambda m: None)
    monkeypatch.setitem(D.CONFIGS["rnh"], "R6", {"head": "cmal", "hidden_f": 512})     # a sixth, undeclared config
    with pytest.raises(parity.ParityError, match="not the declared one"):
        parity.require_declared("rnh", "R6")
    with pytest.raises(parity.ParityError, match="not the declared one"):
        parity.require_declared("sonde_hf", "C1")               # the whole declaration hash changed
    with pytest.raises(parity.ParityError, match="different declaration is already on record"):
        parity.declare(log=lambda m: None)


def test_declared_lists():
    from sentinel.models.sonde_v2 import declared as D

    assert all(len(c) <= 5 for c in D.CONFIGS.values()) and len(D.CONFIGS["rnh"]) == 5
    for arm in ("gbm_nwp", "pf_gbm"):
        assert D.MANDATORY[arm] == "Q19" and D.CONFIGS[arm]["Q19"]["quantile_levels"] == 19
        assert D.arm_list(arm)["gbm_settings"] is D.GBM_SETTINGS
        assert set(D.GBM_SETTINGS["max_fit_rows"]) == set(D.CONFIGS[arm])
        assert all(v is None for v in D.GBM_SETTINGS["max_fit_rows"].values())        # G-2: full rows
    assert all(c.get("physics") for c in D.CONFIGS["pf_gbm"].values())
    assert not any(c.get("physics") for c in D.CONFIGS["gbm_nwp"].values())
    assert D.STEP1["comparator"] == ["gbm_nwp", "Q19"] and D.STEP1["candidate"] == ["sonde_hf", "C1"]
    assert D.STEP1["threshold"] == 0.01 and D.PRODUCT == "gefs_c00" and D.ES_FRACTION == 0.15
    d = D.declaration()
    assert set(d["arms"]) == set(D.CONFIGS) and set(d["arm_sha256"]) == set(D.CONFIGS)
    for arm in ("sonde_hf", "rnh"):                               # identical optimiser settings, stage 1 and 2
        t = d["arms"][arm]["train_config_defaults"]
        assert t["weight_decay"] == 1e-5 and t["es_max_samples"] == 30_000 and d["arms"][arm]["stage2"] is D.STAGE2
    a, b = (d["arms"][k]["train_config_defaults"] for k in ("sonde_hf", "rnh"))
    assert {k: a[k] for k in ("lr", "weight_decay", "batch_size", "patience", "es_max_samples", "grad_clip")} == \
           {k: b[k] for k in ("lr", "weight_decay", "batch_size", "patience", "es_max_samples", "grad_clip")}
    assert "do_pctsat" in D.TRANSFORMED and not set(D.SCORED) & set(D.AUX_REPORTED)


def test_gbm_config_follows_the_lead_ruling(monkeypatch):
    """``GBM_SETTINGS`` is the lead's GBM ruling (G-2, G-3, G-5, G-6, G-7): the ruling file on disk, the recipe of
    the frozen ``GBMConfig``, device gpu, full rows for every configuration, and no cap read by any driver."""
    import hashlib

    from sentinel.core import paths
    from sentinel.models.sonde.gbm import GBMConfig
    from sentinel.models.sonde_v2 import declared as D

    g = D.GBM_SETTINGS
    rp = paths.REPO_ROOT / g["ruling"]["file"]
    assert g["ruled"] and rp.name == "sonde_arch_v2_gbm_lead_ruling_20261110.md"
    assert hashlib.sha256(rp.read_bytes()).hexdigest() == g["ruling"]["sha256"]
    assert g["device"] == "gpu" and g["device_settings"] == {"device_type": "gpu", "gpu_use_dp": False,
                                                            "max_bin": 255}
    assert g["cv_gpu_hours_limit"] == 300 and g["resume"]["process_limit_min"] == 115
    assert g["resume"]["factor"] == 1.5 and g["resume"]["first_booster_allowance_min"] == 40
    assert g["resume"]["booster_limit_h"] == 2
    d = GBMConfig()                                                   # frozen file: the recipe of G-5
    for k in ("learning_rate", "max_rounds", "early_stopping_rounds", "num_leaves", "min_data_in_leaf",
              "feature_fraction", "bagging_fraction", "bagging_freq", "lambda_l2", "es_max_rows", "seed"):
        assert g[k] == getattr(d, k), k
    c, p = D.gbm_config("gbm_nwp", "Q19"), D.gbm_config("pf_gbm", "LS")
    assert (c.max_fit_rows, p.max_fit_rows) == (None, None) and p.physics and not c.physics
    assert c.device == "gpu" and c.max_bin == 255 and c.num_threads == 6 and c.bagging_freq == 1
    assert c.learning_rate == 0.05 and c.max_rounds == 2000 and c.quantile_levels == 19
    assert D.gbm_config("gbm_nwp", "Q").quantile_levels == 7 and p.quantile_mode == "location_scale"
    monkeypatch.setitem(D.GBM_SETTINGS, "max_fit_rows", {"Q19": None, "Q": 500_000, "LS": None})
    with pytest.raises(RuntimeError, match="G-2"):                    # a cap would need a new lead ruling
        D.gbm_config("gbm_nwp", "Q")
    monkeypatch.setitem(D.GBM_SETTINGS, "max_fit_rows", {"Q19": None, "Q": None, "LS": None})
    monkeypatch.setitem(D.GBM_SETTINGS, "device_settings", {"device_type": "gpu", "gpu_use_dp": True,
                                                            "max_bin": 255})
    with pytest.raises(RuntimeError, match="G-6"):
        D.gbm_config("gbm_nwp", "Q19")
    with pytest.raises(KeyError):
        D.gbm_config("gbm_nwp", "Q99")
    with pytest.raises(KeyError):
        D.gbm_config("rnh", "R1")
    monkeypatch.setitem(D.GBM_SETTINGS, "ruled", False)
    with pytest.raises(RuntimeError, match="not ruled"):
        D.gbm_config("gbm_nwp", "Q19")


def test_no_driver_reads_the_cap_ladder():
    """G-9: the cap ladder stays in ``GBM_SETTINGS`` for the timing benchmark script and is read by nothing else."""
    from sentinel.core import paths
    from sentinel.models.sonde_v2 import declared as D

    assert D.GBM_SETTINGS["cap_ladder"][:2] == [None, 1_500_000]
    for rel in ("scripts/train/sonde_v2_step1.py", "scripts/train/sonde_v2_cv.py",
                "sentinel/models/sonde_v2/gbm_nwp.py", "sentinel/models/sonde_v2/gbm_resume.py"):
        src = (paths.REPO_ROOT / rel).read_text()
        assert "cap_ladder" not in src and "max_fit_rows\"]" not in src, rel
    import inspect
    assert "cap_ladder" not in inspect.getsource(D.gbm_config)
    assert "cap_ladder" in (paths.REPO_ROOT / "scripts/train/sonde_v2_gbm_timing.py").read_text()


# ----------------------------------------------------------------------------- tensors, networks, training

FIT, ES, HELD = np.arange(0, 6), np.array([6, 7]), np.array([8, 9])


def _synth_product(data, seed=1, first=None):
    """(S, T, 7, F) issued forecasts; NaN before ``first`` (an archive that starts late)."""
    rng = np.random.default_rng(seed)
    S, T = len(data.sites), len(data.cube.dates)
    x = rng.normal(0, 1, (S, T, 7, len(NWP.FIELDS))).astype(np.float32)
    x[..., NWP.F_IDX["tmean_c"]] += 12 + 8 * np.cos(2 * np.pi * (data.cube.dates.dayofyear.to_numpy() - 200)
                                                    / 365.25)[None, :, None]
    x[..., NWP.F_IDX["precip_mm"]] = np.abs(x[..., NWP.F_IDX["precip_mm"]]) * 4
    if first is not None:
        x[:, :data.cube.dates.get_loc(pd.Timestamp(first))] = np.nan
    return x


def _tensors(data, fit=FIT):
    from sentinel.models.sonde_v2.data_v2 import V2Tensors
    from sentinel.models.sonde_v2.prep import NOT_MODELLED

    ten = V2Tensors(data, variables=[v for v in data.cube.variables if v not in NOT_MODELLED], L=14,
                    use_weather=False, hourly=False, stat_sites=fit)
    x = _synth_product(data)
    ten.set_products("train", x, stat_sites=fit)
    ten.set_products("real", _synth_product(data, first="2012-01-01"))
    ten.set_products("eval", x)
    return ten


@pytest.fixture(scope="module")
def synth_ten(synth_v2):
    return _tensors(synth_v2)


def _chronos_pipe():
    return SimpleNamespace(quantiles=sorted(set(round(float(q), 4)
                                                for q in np.r_[np.round(np.arange(0.05, 0.951, 0.05), 2), 0.01])))


def test_chronos_cov_never_reads_the_transformed_target():
    from sentinel.models.sonde.chronos import ChronosConfig, ChronosForecaster
    from sentinel.models.sonde_v2.chronos_cov import ChronosCovForecaster
    from sentinel.models.sonde_v2.prep import NOT_MODELLED

    assert NOT_MODELLED == ("do_pctsat",)
    f = ChronosCovForecaster(_chronos_pipe(), ChronosConfig(mode="multi", context=8),
                             nwp_x=np.zeros((1, 30, 7, len(NWP.FIELDS)), np.float32))
    with mock.patch.object(ChronosForecaster, "prepared_inputs", return_value=([None], [["a"]])) as p:
        f.prepared_inputs(SimpleNamespace(weather=None), 0, np.array([20]), ["temp_c_mean", "do_pctsat", "do_mgl_mean"])
    assert p.call_args[0][3] == ["temp_c_mean", "do_mgl_mean"]


def _tiny_cfg(kind="sonde_hf", **kw):
    from sentinel.models.sonde_v2 import train_v2 as TV

    base = dict(kind=kind, hourly=False, hidden_d=16, hidden_h=8, hidden_f=16, batch_size=8, steps_per_epoch=2,
                max_epochs=1, patience=2, es_max_samples=16, amp=False)
    return TV.TrainConfig(**{**base, **kw})


def test_fit_site_constants_and_roles_untouched(synth_v2, synth_ten):
    from sentinel.models.sonde_v2.data_v2 import V2Tensors

    roles = synth_v2.roles.copy()
    other = V2Tensors(synth_v2, variables=synth_ten.variables, L=14, use_weather=False, stat_sites=np.array([0, 1]))
    assert (synth_v2.roles == roles).all() and "do_pctsat" not in synth_ten.variables
    assert not np.allclose(other.st.static, synth_ten.st.static)       # standardised with other sites' statistics
    assert synth_ten.issue_start.shape == (10, len(synth_ten.variables))


def test_single_product_is_enforced(synth_v2, synth_ten):
    from sentinel.models.sonde_v2 import train_v2 as TV
    from sentinel.models.sonde_v2.chronos_cov import ChronosCovForecaster
    from sentinel.models.sonde_v2.data_v2 import V2Tensors
    from sentinel.models.sonde_v2.gbm_nwp import _one_product

    x = _synth_product(synth_v2)
    for bad in ([x], [x, x], (x, x), x[..., :3], x[:, :, 0]):
        with pytest.raises(ValueError, match="exactly one declared NWP product"):
            synth_ten.set_products("eval2", bad)
    assert "eval2" not in synth_ten.products
    fresh = V2Tensors(synth_v2, variables=synth_ten.variables, L=14, use_weather=False, stat_sites=FIT)
    with pytest.raises(ValueError, match="train product first"):
        fresh.set_products("real", x)
    with pytest.raises(ValueError, match="train product first"):
        fresh.set_products("train", x)                                  # no fit sites for the standardisation
    with pytest.raises(RuntimeError, match="< 1000 finite train-block values"):
        fresh.set_products("train", _synth_product(synth_v2, first="2021-09-20"), stat_sites=FIT)
    for bad in ([x], [x, x]):
        with pytest.raises(ValueError):
            _one_product(bad)
        with pytest.raises(ValueError, match="exactly one declared NWP product"):
            ChronosCovForecaster(_chronos_pipe(), None, nwp_x=bad)
    assert _one_product(x) is x
    with pytest.raises(KeyError, match="no NWP product registered"):
        synth_ten.batch(np.array([0]), np.array([4000]), "gfs")
    cfg = _tiny_cfg()
    for kind in ("sonde_hf", "rnh"):                                    # one projection = one product slot
        m = TV.build_model(_tiny_cfg(kind, rnh_target="temp_c_mean"), synth_ten)
        assert sum(isinstance(c, N.MaskedMeanNWP) and len(c.proj) == 1 for c in m.modules()) == 1
    assert cfg.weight_decay == 1e-5 and cfg.lr == 1e-3


def test_val_block_batches_have_a_non_empty_product_slot(synth_v2, synth_ten):
    from sentinel.models.sonde_v2 import train_v2 as TV

    cv = _script("sonde_v2_cv")
    rows = cv.held_rows(synth_v2, "temp_c_mean", HELD)
    assert set(rows.si.tolist()) == set(HELD.tolist())
    assert rows.issue_date.min() >= np.datetime64("2021-10-01") + np.timedelta64(42, "D")   # val block + embargo
    si, ti = rows.si[::37].astype(np.int64), rows.ti[::37].astype(np.int64)
    b = synth_ten.batch(si, ti, "eval")
    assert len(b["nwp_x"]) == 1 and len(b["nwp_m"]) == 1
    assert b["nwp_x"][0].shape == (si.size, 14, 2 * len(NWP.FIELDS))
    assert b["nwp_m"][0][:, :7].all() and not b["nwp_m"][0][:, 7:].any()     # 7 forecast leads, none beyond
    assert np.abs(b["nwp_x"][0][:, :7, :len(NWP.FIELDS)]).sum() > 0
    assert cv.eval_slot_share(_synth_product(synth_v2), si, ti, "unit") == 1.0
    with pytest.raises(RuntimeError, match="refusing to write forecasts"):
        cv.eval_slot_share(_synth_product(synth_v2, first="2023-01-01"), rows.si, rows.ti, "unit")
    blank = dict(b, nwp_x=[np.zeros_like(b["nwp_x"][0])], nwp_m=[np.zeros_like(b["nwp_m"][0])])
    for kind in ("sonde_hf", "rnh"):                                    # the slot reaches the output of both
        cfg = _tiny_cfg(kind, rnh_target="temp_c_mean")
        torch.manual_seed(0)
        m = TV.build_model(cfg, synth_ten).eval()
        with torch.no_grad():
            o1 = TV.forward(m, cfg, TV._to(b, "cpu", cfg), synth_ten)
            o2 = TV.forward(m, cfg, TV._to(blank, "cpu", cfg), synth_ten)
        d = (o1 - o2).abs()
        assert float(d[:, :7].max()) > 1e-6, kind


def test_training_rows_are_labelled_for_the_arm_target(synth_v2, synth_ten):
    from sentinel.models.sonde_v2 import train_v2 as TV

    ten = synth_ten
    all_si, all_ti = ten.train_rows(FIT, "train")
    kd, kt = ten.variables.index("do_mgl_mean"), ten.variables.index("temp_c_mean")
    cfg = _tiny_cfg("rnh", rnh_target="do_mgl_mean")
    assert TV.loss_targets(cfg, ten) == ["do_mgl_mean"]
    si, ti = ten.train_rows(FIT, "train", TV.loss_targets(cfg, ten))
    assert 0 < si.size < all_si.size and 3 not in set(si.tolist()) and 3 in set(all_si.tolist())
    y = ten.batch(si[::101], ti[::101], "train")["y"]
    assert np.isfinite(y[:, :, kd]).any(axis=1).all()                   # every R-NH row has a label of ITS target
    assert synth_v2.cube.dates[ti.max()] + pd.Timedelta(days=14) <= pd.Timestamp("2021-09-30")
    assert (ti >= ten.issue_start[si, kd]).all()
    # reference-window rule in the labels: site 2 has no temperature label at all, site 1 none before its start
    st_si, st_ti = ten.train_rows(FIT, "train", ["temp_c_mean"])
    assert 2 not in set(st_si.tolist()) and 2 in set(all_si.tolist())
    s2 = all_si == 2
    assert not np.isfinite(ten.batch(all_si[s2][::200], all_ti[s2][::200], "train")["y"][:, :, kt]).any()
    c3 = _tiny_cfg(multitask=False)
    assert TV.loss_targets(c3, ten) == [v for v in ten.variables if v in c3.primary]
    assert TV.loss_targets(_tiny_cfg(), ten) == list(ten.variables)
    es_si, _ = ten.train_rows(ES, "train", ["do_mgl_mean"])
    assert set(es_si.tolist()) <= set(ES.tolist()) and es_si.size


def test_c4_differs_from_c1_in_head_and_loss(synth_ten):
    from sentinel.models.sonde_v2 import declared as D
    from sentinel.models.sonde_v2 import train_v2 as TV

    assert D.CONFIGS["sonde_hf"]["C4"] == {"head": "cmal"} and D.CONFIGS["sonde_hf"]["C1"] == {}
    c1, c4 = _tiny_cfg(**D.train_overrides("sonde_hf", "C1")), _tiny_cfg(**D.train_overrides("sonde_hf", "C4"))
    ten = synth_ten
    si, ti = ten.train_rows(FIT, "train")
    b = ten.batch(si[::997][:16], ti[::997][:16], "train")
    res = {}
    for name, cfg in (("C1", c1), ("C4", c4)):
        torch.manual_seed(0)
        m = TV.build_model(cfg, ten)
        bt = TV._to(b, "cpu", cfg)
        out = TV.forward(m, cfg, bt, ten)
        loss = TV.loss_fn(out, cfg, bt, ten)
        loss.backward()
        q = TV.predict(m, cfg, ten, si[::997][:16], ti[::997][:16], "train", "cpu")
        res[name] = (m, out, float(loss), q)
    h1 = [type(c).__name__ for c in res["C1"][0].modules()]
    h4 = [type(c).__name__ for c in res["C4"][0].modules()]
    assert "MonotoneQuantileHead" in h1 and "CMALHead" not in h1
    assert "CMALHead" in h4 and "MonotoneQuantileHead" not in h4
    assert torch.is_tensor(res["C1"][1]) and res["C1"][1].shape[-1] == 19          # 19 monotone quantiles
    assert set(res["C4"][1]) >= {"logit", "mu", "b"}                               # mixture parameters
    assert np.isfinite(res["C1"][2]) and np.isfinite(res["C4"][2]) and abs(res["C1"][2] - res["C4"][2]) > 1e-6
    for name in ("C1", "C4"):
        q = res[name][3]
        assert q.shape == (16, 14, len(ten.variables), 19)
        assert (np.diff(q[np.isfinite(q).all(-1)], axis=-1) >= 0).all()
    assert not np.allclose(np.nan_to_num(res["C1"][3]), np.nan_to_num(res["C4"][3]))


def test_predict_subsets_and_no_climatology(synth_ten):
    from sentinel.models.sonde_v2 import train_v2 as TV

    ten = synth_ten
    kt = ten.variables.index("temp_c_mean")
    si = np.array([0, 2, 5, 8], dtype=np.int64)
    ti = np.full(4, 5000, dtype=np.int64)
    cfg = _tiny_cfg()
    torch.manual_seed(0)
    m = TV.build_model(cfg, ten)
    full = TV.predict(m, cfg, ten, si, ti, "eval", "cpu", bs=3)
    sub = TV.predict(m, cfg, ten, si, ti, "eval", "cpu", bs=2, leads=[2, 4, 6], var_idx=[kt, 0])
    assert np.allclose(sub, full[:, [2, 4, 6]][:, :, [kt, 0]], equal_nan=True, atol=1e-5)
    assert np.isnan(full[1, :, kt]).all() and np.isfinite(full[0, :, kt]).all()     # site 2: no temperature climatology
    r = _tiny_cfg("rnh", rnh_target="temp_c_mean", head="cmal")
    mr = TV.build_model(r, ten)
    qr = TV.predict(mr, r, ten, si, ti, "eval", "cpu", leads=[0, 13])
    assert qr.shape == (4, 2, 1, 19) and np.isnan(qr[1]).all() and np.isfinite(qr[0]).all()
    assert (np.diff(qr[0], axis=-1) >= 0).all()


def test_weight_decay_and_lr_come_from_the_configuration(tmp_path, synth_ten):
    """Verifier item F08: a saved optimiser state must not overwrite weight decay / learning rate (resume and
    stage 2), and the early-stopping sample is capped at ``es_max_samples``."""
    from sentinel.models.sonde_v2 import train_v2 as TV

    ten = synth_ten
    si, ti = ten.train_rows(FIT, "train")
    esi, eti = ten.train_rows(ES, "train")
    cfg = _tiny_cfg()
    seen = []
    real_step = torch.optim.AdamW.step

    def spy(self, *a, **k):
        seen.append((self.param_groups[0]["lr"], self.param_groups[0]["weight_decay"]))
        return real_step(self, *a, **k)

    with mock.patch.object(torch.optim.AdamW, "step", spy):
        r = TV.train(cfg, ten, si, ti, esi, eti, tmp_path / "s1", device="cpu", log=lambda m: None)
        assert r["complete"] and r["epoch"] == 1 and (tmp_path / "s1" / "best.pt").exists()
        assert r["optimizer"]["weight_decay"] == 1e-5 and r["optimizer"]["n_es_rows"] == 16
        assert r["optimizer"]["n_train_rows"] == si.size and r["optimizer"]["use"] == "train"
        assert set(seen) == {(1e-3, 1e-5)}
        # tamper with the saved optimiser state, then resume for a second epoch
        c = torch.load(tmp_path / "s1" / "last.pt", weights_only=False)
        for g in c["opt"]["param_groups"]:
            g["lr"], g["weight_decay"] = 0.5, 0.01                      # e.g. the AdamW default weight decay
        torch.save(c, tmp_path / "s1" / "last.pt")
        seen.clear()
        import dataclasses
        r = TV.train(dataclasses.replace(cfg, max_epochs=2), ten, si, ti, esi, eti, tmp_path / "s1", device="cpu",
                     log=lambda m: None)
        assert r["epoch"] == 2 and len(seen) == 2 and set(seen) == {(1e-3, 1e-5)}
        # stage 2: lr / 3, same weight decay, initialised from the stage-1 best weights, archived product
        seen.clear()
        s2 = dataclasses.replace(cfg, lr=cfg.lr / 3)
        r2 = TV.train(s2, ten, si, ti, esi, eti, tmp_path / "s2", device="cpu", log=lambda m: None,
                      init=tmp_path / "s1" / "best.pt", use="real")
        assert set(seen) == {(cfg.lr / 3, 1e-5)} and r2["optimizer"]["use"] == "real"
        assert r2["optimizer"]["lr"] == cfg.lr / 3 and r2["optimizer"]["weight_decay"] == 1e-5


# ----------------------------------------------------------------------------- GBM references

def test_stratified_cap():
    from sentinel.models.sonde_v2.gbm_nwp import stratified_cap

    si = np.repeat(np.arange(4), [1000, 500, 300, 200])
    p = stratified_cap(si, 400, np.random.default_rng(0))
    assert p.size == 400 and (np.diff(p) > 0).all() and np.bincount(si[p]).tolist() == [200, 100, 60, 40]
    assert np.array_equal(p, stratified_cap(si, 400, np.random.default_rng(0)))
    assert not np.array_equal(p, stratified_cap(si, 400, np.random.default_rng(1)))
    for cap in (333, 7, 1999):
        q = stratified_cap(si, cap, np.random.default_rng(2))
        assert q.size == cap and np.unique(q).size == cap
    for cap in (None, 2000, 5000):
        assert np.array_equal(stratified_cap(si, cap, np.random.default_rng(0)), np.arange(2000))


def test_gbm_references_rows_fit_and_predict(synth_v2):
    from sentinel.models.sonde_v2.gbm_nwp import GBMNWPConfig, SondeGBMNWP

    cv = _script("sonde_v2_cv")
    data, x = synth_v2, _synth_product(synth_v2)
    jt = data.cube.var_index("temp_c_mean")
    kw = dict(max_rounds=5, early_stopping_rounds=5, num_threads=2, es_max_rows=500)   # G-2: no max_fit_rows
    m = SondeGBMNWP(GBMNWPConfig(quantile_levels=7, **kw), horizons=(3,))
    si, ti, y, s, n = m.labelled_rows(data, "temp_c_mean", 3, FIT, cap=3000, salt=0)
    assert si.size == 3000 < n and np.isfinite(y).all() and np.isfinite(s).all()
    assert (ti >= data.ref_issue_start[si, jt]).all() and set(si.tolist()) == {0, 1, 3, 4, 5}   # site 2: no climatology
    si2, ti2, *_ = m.labelled_rows(data, "temp_c_mean", 3, FIT, cap=3000, salt=0)
    si3, ti3, *_ = m.labelled_rows(data, "temp_c_mean", 3, FIT, cap=3000, salt=1)
    assert np.array_equal(ti, ti2) and not np.array_equal(ti, ti3)
    out = m.fit_var_nwp(data, "temp_c_mean", train_sites=FIT, es_sites=ES, nwp_fit=x, log=lambda s: None)
    info = m.rows_info[("temp_c_mean", 3)]
    assert info["n_fit"] == info["n_fit_total"] == n and info["max_fit_rows"] is None   # G-2: every masked row
    assert info["n_es"] == 500 < info["n_es_total"]
    assert info["fit_sites"] == 5 and info["es_sites"] == 2 and info["fit_target_max"] <= "2021-09-30"
    assert len(out[3]["rounds"]) == 7 and 0 <= out[3]["es_cov90"] <= 1
    assert set(out[3]["boosters"]) == {"q05", "q10", "q25", "q50", "q75", "q90", "q95"}
    assert all(b["device"] == "cpu" and b["gpu"] == -1 and b["rounds_run"] >= 1 and b["fitted_for_config"] == "adhoc"
               for b in out[3]["boosters"].values())
    with pytest.raises(ValueError, match="G-2"):
        SondeGBMNWP(GBMNWPConfig(quantile_levels=7, max_fit_rows=100, **kw), horizons=(3,)).fit_var_nwp(
            data, "temp_c_mean", train_sites=FIT, es_sites=ES, nwp_fit=x, log=lambda s: None)
    cols = m.feature_columns
    assert {"nwp_dd_ew", "nwp_dd_cum", "nwp_tmean_target", "det_clearsky_sw", "det_day_length"} <= set(cols)
    assert not any("pctsat" in c or c.startswith("pf_") or "tmean_ew" in c for c in cols)
    rows = cv.held_rows(data, "temp_c_mean", HELD)
    q = m.predict_rows_nwp(data, "temp_c_mean", rows, x)
    assert q.shape == (len(rows), 1, 19) and np.isfinite(q).all() and (np.diff(q, axis=-1) >= 0).all()
    with pytest.raises(ValueError, match="exactly one declared NWP product"):
        m.predict_rows_nwp(data, "temp_c_mean", rows, [x])
    # PF-GBM, location-scale: physics columns present, residual quantiles from the early-stopping rows
    p = SondeGBMNWP(GBMNWPConfig(quantile_mode="location_scale", physics=True, **kw), horizons=(3,))
    o = p.fit_var_nwp(data, "do_mgl_mean", train_sites=FIT, es_sites=ES, nwp_fit=x, log=lambda s: None)
    assert {"pf_tw_target", "pf_dosat_target", "pf_k600", "pf_sc_mix"} <= set(p.feature_columns)
    assert o[3]["n_residual_rows"] == o[3]["n_es"] == 500 and len(o[3]["rounds"]) == 2
    assert set(o[3]["boosters"]) == {"ls_loc", "ls_scale"}
    assert p.rows_info[("do_mgl_mean", 3)]["fit_sites"] == 5                      # site 3 has no DO
    qd = p.predict_rows_nwp(data, "do_mgl_mean", cv.held_rows(data, "do_mgl_mean", HELD), x)
    assert np.isfinite(qd).all() and (np.diff(qd, axis=-1) >= 0).all()
    # too few labelled rows: the horizon is skipped and recorded, never fitted on a handful of rows
    sk = SondeGBMNWP(GBMNWPConfig(quantile_levels=7, **kw), horizons=(3,))
    o = sk.fit_var_nwp(data, "do_mgl_mean", train_sites=np.array([3]), es_sites=ES, nwp_fit=x, log=lambda s: None)
    assert o[3]["skipped"] and o[3]["n_fit"] == 0 and ("do_mgl_mean", 3) not in sk.qmodels
    assert cv.held_rows(data, "do_mgl_mean", np.array([3])) is None


# ----------------------------------------------------------------------------- GBM booster store and resume

def test_gbm_process_clock_rule():
    """G-3: a process starts a booster only if elapsed + 1.5 x the longest booster time seen (40 min while none)
    is at most 115 minutes."""
    from sentinel.models.sonde_v2 import gbm_resume as GR

    t = {"now": 0.0}
    c = GR.ProcessClock(t0=0.0, now=lambda: t["now"])
    assert c.may_start() and c.need() == 40 * 60                      # no booster time seen yet: 40 min of room
    t["now"] = 75 * 60 + 1
    assert not c.may_start()                                          # 75 min + 40 min > 115 min
    t["now"] = 10 * 60
    c.seen(600.0)
    c.seen(1200.0)
    c.seen(300.0)
    assert c.longest == 1200.0 and c.need() == 1800.0 and c.may_start()
    t["now"] = 95 * 60 + 1
    assert not c.may_start() and c.may_predict()                      # 95 min + 30 min > 115 min
    t["now"] = 105 * 60 + 1
    assert not c.may_predict()                                        # forecasts start at most 105 min in
    assert (GR.PROCESS_LIMIT_S, GR.BOOSTER_LIMIT_S, GR.ABORT_AT_S, GR.EXIT_RESUME) == (6900, 7200, 7140, 75)


def _gbm_run(data, x, store, *, config="Q", levels=7, stop_after=None, var="temp_c_mean"):
    """One GBM process over ``var`` h=3 into ``store``; ``stop_after``: budget reached after n new boosters."""
    from sentinel.models.sonde_v2 import gbm_resume as GR
    from sentinel.models.sonde_v2.gbm_nwp import GBMNWPConfig, SondeGBMNWP

    cfg = GBMNWPConfig(quantile_levels=levels, max_rounds=4, early_stopping_rounds=4, num_threads=2,
                       es_max_rows=400)
    m = SondeGBMNWP(cfg, horizons=(3,))
    proc = GR.GBMProcess("step1", "test", cfg, log=lambda s: None)
    if stop_after is not None:
        inner, n = proc.before_booster, []

        def before(what):
            if len(n) >= stop_after:
                raise GR.ProcessBudgetReached(what)
            n.append(what)
            inner(what)

        proc.before_booster = before
    out = m.fit_var_nwp(data, var, train_sites=FIT, es_sites=ES, nwp_fit=x, store_root=store, process=proc,
                        config=config, log=lambda s: None)
    return m, out, proc


def test_gbm_boosters_are_saved_read_back_and_resumed(synth_v2, tmp_path):
    """G-3: every booster is saved to its own file with its stopping round and wall time and is read back from
    that file; a booster that has a record is never fitted again; a resumed unit is the uninterrupted one."""
    import json

    from sentinel.models.sonde_v2 import gbm_resume as GR

    cv = _script("sonde_v2_cv")
    data, x = synth_v2, _synth_product(synth_v2)
    rows = cv.held_rows(data, "temp_c_mean", HELD)
    whole, out, p0 = _gbm_run(data, x, tmp_path / "whole")
    q_whole = whole.predict_rows_nwp(data, "temp_c_mean", rows, x)
    assert len(p0.fitted) == 7 and not p0.loaded
    d = tmp_path / "whole" / "temp_c_mean" / "h3"
    assert {f.name for f in d.glob("q*.txt")} == {f"q{a:02d}.txt" for a in (5, 10, 25, 50, 75, 90, 95)}
    meta = json.loads((d / "q50.json").read_text())
    assert meta["rounds_run"] >= meta["best_iteration"] >= 1 and meta["wall_s"] > 0 and meta["n_trees"] >= 1
    assert meta["file_vs_fitted_max_abs_diff"] == 0.0 and meta["pid"] == p0.pid and meta["device"] == "cpu"
    assert meta["n_fit"] == out[3]["n_fit"] and meta["fitted_for_config"] == "Q"
    # a process that may not start the next booster: the finished ones stay, the unit is continued by the next
    with pytest.raises(GR.ProcessBudgetReached):
        _gbm_run(data, x, tmp_path / "resumed", stop_after=3)
    dr = tmp_path / "resumed" / "temp_c_mean" / "h3"
    assert len(list(dr.glob("q*.json"))) == 3
    rec = json.loads((dr / "unit__Q.json").read_text())
    assert len(rec["processes"]) == 1 and rec["processes"][0]["end"] == "budget"
    assert len(rec["processes"][0]["fitted"]) == 3 and rec["n_fit"] == out[3]["n_fit"]
    m2, out2, p2 = _gbm_run(data, x, tmp_path / "resumed")
    assert len(p2.loaded) == 3 and len(p2.fitted) == 4            # the three finished boosters are not refitted
    rec = json.loads((dr / "unit__Q.json").read_text())
    assert [p["end"] for p in rec["processes"]] == ["budget", "complete"] and len(rec["processes"]) == 2
    assert rec["fit_rows_sha256"] == out[3]["fit_rows_sha256"] and rec["x_fit_sha256"]
    q_res = m2.predict_rows_nwp(data, "temp_c_mean", rows, x)
    assert np.array_equal(q_res, q_whole)                         # same rows, seeds and settings as uninterrupted
    assert out2[3]["rounds"] == out[3]["rounds"] and out2[3]["es_cov90"] == out[3]["es_cov90"]
    # a record that does not belong to these rows, seeds or settings is a hard error, never a silent refit
    j = dr / "q50.json"
    j.write_text(json.dumps({**json.loads(j.read_text()), "key": "0" * 64}))
    with pytest.raises(GR.ResumeError, match="other rows, seed or settings"):
        _gbm_run(data, x, tmp_path / "resumed")
    # two processes never fit the same unit at the same time
    lock = GR.unit_lock(tmp_path / "whole" / "temp_c_mean")
    with pytest.raises(GR.UnitBusy):
        _gbm_run(data, x, tmp_path / "whole")
    lock.close()


def test_gbm_q_reuses_the_identical_q19_boosters(synth_v2, tmp_path):
    """G-3: a booster of another configuration of the same arm, seed, fold, variable and horizon with an identical
    key is used instead of a second fit; the quantile levels of Q are seven of the nineteen of Q19."""
    from sentinel.models.sonde_v2 import gbm_resume as GR

    data, x = synth_v2, _synth_product(synth_v2)
    store = tmp_path / "arm"
    _, o19, p19 = _gbm_run(data, x, store, config="Q19", levels=19)
    assert len(p19.fitted) == 19 and len(o19[3]["rounds"]) == 19
    _, o7, p7 = _gbm_run(data, x, store, config="Q", levels=7)
    assert len(p7.loaded) == 7 and not p7.fitted
    assert [b["fitted_for_config"] for b in o7[3]["boosters"].values()] == ["Q19"] * 7
    assert o7[3]["rounds"] == [o19[3]["boosters"][n]["best_iteration"] for n in GR.booster_names(
        {"quantile_levels": 7})]
    assert GR.booster_names({"quantile_levels": 7}) == ["q05", "q10", "q25", "q50", "q75", "q90", "q95"]
    assert set(GR.booster_names({"quantile_levels": 7})) < set(GR.booster_names({"quantile_levels": 19}))


def test_gbm_fit_rows_are_exactly_the_masked_rows(synth_v2, tmp_path):
    """G-2: the fit rows of a unit are exactly the rows that pass the masks of the row rule — recomputed here from
    the frozen ``training_rows``, the reference-window issue rule and the finite-target mask."""
    import json

    from sentinel.models.sonde_v2 import gbm_resume as GR

    data, x = synth_v2, _synth_product(synth_v2)
    m, out, _ = _gbm_run(data, x, tmp_path / "s")
    si, ti = m.training_rows(data, "temp_c_mean", 3, FIT)
    keep = data.issue_ok("temp_c_mean", si, ti)
    si, ti = si[keep], ti[keep]
    y, sc = m._target(m._builder(data), "temp_c_mean", si, ti, 3)
    ok = np.isfinite(y) & np.isfinite(sc)
    si, ti = si[ok], ti[ok]
    assert out[3]["n_fit"] == si.size == out[3]["n_fit_total"]
    assert out[3]["fit_rows_sha256"] == GR.array_sha(si, ti)
    rec = json.loads((tmp_path / "s" / "temp_c_mean" / "h3" / "unit__Q.json").read_text())
    assert rec["fit_rows_sha256"] == GR.array_sha(si, ti) and rec["max_fit_rows"] is None


def test_gbm_gpu_params_and_gpu_hour_rows(synth_v2, tmp_path, monkeypatch):
    """G-6: a GPU fit is ``device_type=gpu``, ``gpu_use_dp=False``, the declared ``max_bin``, no row-wise force and
    at most 6 threads; G-7(a): every process writes its GPU wall time to the registry."""
    from sentinel.core import paths
    from sentinel.models.sonde_v2 import gbm_resume as GR
    from sentinel.models.sonde_v2.gbm_nwp import GBMNWPConfig, SondeGBMNWP

    g = SondeGBMNWP(GBMNWPConfig(device="gpu", num_threads=6))._params()
    assert g["device_type"] == "gpu" and g["gpu_use_dp"] is False and g["max_bin"] == 255
    assert g["gpu_device_id"] == 0 and "force_row_wise" not in g and g["num_threads"] == 6
    assert g["deterministic"] is True and g["seed"] == 0
    with pytest.raises(ValueError):
        SondeGBMNWP(GBMNWPConfig(device="mps"))._params()
    with pytest.raises(ValueError, match="GPU lock"):                 # a GPU fit needs the driver's process
        SondeGBMNWP(GBMNWPConfig(device="gpu"), horizons=(3,)).fit_var_nwp(
            synth_v2, "temp_c_mean", train_sites=FIT, es_sites=ES, nwp_fit=_synth_product(synth_v2),
            log=lambda s: None)
    reg = tmp_path / "registry.jsonl"
    monkeypatch.setattr(paths, "REGISTRY_PATH", reg)
    cfg = GBMNWPConfig(num_threads=2)
    p = GR.GBMProcess("cv", "cv_gbm_nwp_Q19_s0_f0", cfg, log=lambda s: None)
    p.fitted = [["temp_c_mean/h3", "q50", 12.5]]
    p.end = "complete"
    row = p.finish()
    assert p.finish() is None                                         # one row per process
    assert row["experiment"] == "sonde_arch_v2/gbm_gpu_hours" and row["metric"] == "gpu_hours"
    assert row["split"] == "cv" and row["n"] == 1 and row["value"] >= 0 and row["extra"]["end"] == "complete"
    assert row["extra"]["booster_fit_s"] == 12.5 and row["extra"]["pid"] == p.pid
    assert GR.gpu_hours_total("cv", reg) == row["value"] and GR.gpu_hours_total("step1", reg) == 0.0


def test_gbm_cv_budget_projection_and_gate(tmp_path):
    """G-7(c, d): the CV projection from the step-1 stopping rounds, and the gate that refuses a CV GBM process
    without the projection, above the limit, or above it after the running total — until the lead rules."""
    import json

    from sentinel.core.registry import log_result
    from sentinel.models.sonde_v2 import declared as D
    from sentinel.models.sonde_v2 import gbm_resume as GR

    metas = [{"name": "q50", "rounds_run": 300, "best_iteration": 200, "wall_s": 600.0, "n_fit": 1_000_000}]
    rows = {f"{k}|temp_c_mean|{h}": {"n_fit": 5_000_000, "n_es": 400_000, "skipped": False}
            for k in range(5) for h in (3, 7)}
    rows["0|do_mgl_mean|3"] = {"n_fit": 10, "n_es": 2, "skipped": True}
    configs = {a: D.CONFIGS[a] for a in ("gbm_nwp", "pf_gbm")}
    pr = GR.cv_projection(metas, rows, configs, seed=0)
    # one booster: 300 rounds x (600 s / 300 / 1e6 rows) x 5,000,000 rows = 3000 s; 10 units per arm; the 21
    # distinct boosters of Q19, Q and LS per unit (Q's seven are Q19's, counted once in the total)
    assert pr["step1"]["mean_rounds_run"] == 300 and pr["step1"]["mean_s_per_round_and_row"] == 2e-6
    assert pr["total_gpu_h"] == pytest.approx(2 * 10 * 21 * 3000 / 3600)
    per = pr["gpu_hours_per_arm_and_config_alone"]
    assert set(per) == {f"{a}|{c}" for a in configs for c in ("Q19", "Q", "LS")}
    assert per["gbm_nwp|Q19"] == pytest.approx(10 * 19 * 3000 / 3600)
    assert per["gbm_nwp|Q"] == pytest.approx(10 * 7 * 3000 / 3600) and per["pf_gbm|LS"] == pytest.approx(
        10 * 2 * 3000 / 3600)
    assert len(pr["units"]) == 20 and all(len(u["boosters"]) == 21 for u in pr["units"])    # skipped unit left out
    assert pr["limit_gpu_h"] == 300.0 and "no scaling for the number of feature columns" in pr["formula"]
    with pytest.raises(GR.BudgetError, match="no step-1 booster record"):
        GR.cv_projection([], rows, configs, seed=0)
    # the gate
    bf, store, reg = tmp_path / "gbm_budget.json", tmp_path / "boosters", tmp_path / "registry.jsonl"
    with pytest.raises(GR.BudgetError, match="G-7"):
        GR.cv_gate(bf, store, registry_path=reg)
    bf.write_text(json.dumps(pr))
    with pytest.raises(GR.BudgetError, match="above the limit"):                     # 350 GPU-h > 300
        GR.cv_gate(bf, store, registry_path=reg)
    log_result(workstream="M1-sonde", experiment="sonde/lead_gbm_budget_ruling", model="lead", metric="gpu_hours",
               value=400.0, split="cv", notes="test", extra={"written_by": "lead", "gpu_hours_limit": 400},
               registry_path=reg)
    acc = GR.cv_gate(bf, store, registry_path=reg)                                   # released by the lead's row
    assert acc["limit_gpu_h"] == 400.0 and acc["lead_row"] and acc["spent_cv_gpu_h"] == 0.0
    assert acc["projected_remainder_gpu_h"] == pytest.approx(pr["total_gpu_h"])       # nothing fitted yet
    d = store / "gbm_nwp_s0" / "fold0" / "temp_c_mean" / "h3"
    d.mkdir(parents=True)
    for n in GR.booster_names({"quantile_levels": 19}):
        (d / f"{n}.json").write_text("{}")
    assert GR.cv_gate(bf, store, registry_path=reg)["projected_remainder_gpu_h"] == pytest.approx(
        (2 * 10 * 21 - 19) * 3000 / 3600)
    log_result(workstream="M1-sonde", experiment=GR.GPU_EXPERIMENT, model="cv_x", metric="gpu_hours", value=250.0,
               split="cv", notes="test", registry_path=reg)
    with pytest.raises(GR.BudgetError, match="above the limit"):     # spent + projected remainder > 400
        GR.cv_gate(bf, store, registry_path=reg)


# ----------------------------------------------------------------------------- parity log, folds, step 1

def test_parity_record_and_check(tmp_path, synth_v2):
    from sentinel.models.sonde_v2 import parity

    log = tmp_path / "PARITY.jsonl"
    reg = pd.DataFrame({"site_type": "ST"}, index=synth_v2.sites)

    def rec(arm, config, fold=0, fit=FIT, es=ES, held=HELD, sha=None, use="cv"):
        with mock.patch("sentinel.models.sonde.common.site_registry", return_value=reg), \
                mock.patch.object(parity, "DECLARATION", tmp_path / "none.json"):
            return parity.record(f"{arm}_{config}_f{fold}", data=synth_v2, use=use, arm=arm, config=config,
                                 fold=fold, fit_sites=fit, es_sites=es, held_sites=held, n_rows=10,
                                 feature_columns=["a"], input_sha256=sha or {"gefs_bc": "x"}, log_path=log)

    r = rec("sonde_hf", "C1")
    assert r["n_train_sites"] == 6 and r["n_held_sites"] == 2 and r["site_types"] == {"ST": 6}
    assert r["static_columns"] == list(synth_v2.static.columns) and r["n_loaded_sites"] == 10
    rec("gbm_nwp", "Q19")
    z = rec("chronos_cov", "ZS", fit=np.array([], dtype=int), es=np.array([], dtype=int))   # zero-shot: no fit sites
    assert z["train_sites_sha256"] is None and z["es_sites_sha256"] is None
    ok = parity.check("cv", 0, require_arms=[("sonde_hf", "C1"), ("gbm_nwp", "Q19"), ("chronos_cov", "ZS")],
                      log_path=log)
    assert ok["n_records"] == 3 and ok["train_sites_sha256"] == r["train_sites_sha256"]
    with pytest.raises(parity.ParityError, match="no parity record for"):
        parity.check("cv", 0, require_arms=[("rnh", "R1")], log_path=log)
    with pytest.raises(parity.ParityError, match="fold 1: no parity record"):
        parity.check("cv", 1, log_path=log)
    rec("rnh", "R1", fold=1)
    rec("pf_gbm", "Q19", fold=1, held=np.array([8]))
    with pytest.raises(parity.ParityError, match="held_sites_sha256"):
        parity.check("cv", 1, log_path=log)
    rec("rnh", "R1", fold=2)
    rec("pf_gbm", "Q19", fold=2, fit=np.arange(0, 5))
    with pytest.raises(parity.ParityError, match="train_sites_sha256"):
        parity.check("cv", 2, log_path=log)
    rec("rnh", "R1", fold=3)
    rec("pf_gbm", "Q19", fold=3, es=np.array([6]))
    with pytest.raises(parity.ParityError, match="es_sites_sha256"):
        parity.check("cv", 3, log_path=log)
    rec("rnh", "R1", fold=4)
    rec("pf_gbm", "Q19", fold=4, sha={"gefs_bc": "y"})
    with pytest.raises(parity.ParityError, match=r"input_sha256\[gefs_bc\]"):
        parity.check("cv", 4, log_path=log)
    rec("sonde_hf", "C1", use="step1", sha={"pp": "p1"})
    rec("gbm_nwp", "Q19", use="step1", sha={"pp": "p2"})
    with pytest.raises(parity.ParityError, match=r"input_sha256\[pp\]"):
        parity.check("step1", 0, log_path=log)


def test_cv_fold_sites(tmp_path, synth_v2, monkeypatch):
    import json

    from sentinel.models.sonde_v2.data_v2 import basin_folds

    cv = _script("sonde_v2_cv")
    data = synth_v2
    basin = np.array([data.split.basin_group[s] for s in data.sites])
    f = basin_folds(np.arange(10), basin, 5, cv.SEED)
    assert np.array_equal(f, basin_folds(np.arange(10), basin, 5, cv.SEED)) and sorted(np.bincount(f)) == [2] * 5
    assert all(np.unique(f[basin == b]).size == 1 for b in np.unique(basin))       # whole basin groups
    folds = tmp_path / "FOLDS.json"
    folds.write_text(json.dumps({"folds": {str(k): np.asarray(data.sites)[f == k].tolist() + ["not_loaded"]
                                           for k in range(5)}}))
    monkeypatch.setattr(cv, "FOLDS", folds)
    seen = []
    for k in range(5):
        fit, es, held = cv.fold_sites(data, k)
        assert not set(fit) & set(es) and not set(fit) & set(held) and not set(es) & set(held)
        assert sorted(np.r_[fit, es, held].tolist()) == list(range(10)) and held.size == 2 and es.size == 2
        assert not set(basin[es]) & set(basin[fit]) and not set(basin[held]) & set(basin[np.r_[fit, es]])
        again = cv.fold_sites(data, k)
        assert all(np.array_equal(a, b) for a, b in zip((fit, es, held), again))
        seen += held.tolist()
    assert sorted(seen) == list(range(10))                                         # every site held out once


def test_step1_site_sets_rows_and_statistic(synth_v2):
    s1 = _script("sonde_v2_step1")
    data = synth_v2
    fit, es, held = s1.site_sets(data)
    tr = set(data.site_idx(["train"]).tolist())
    assert set(fit) | set(es) | set(held) == tr and not set(fit) & set(held) and not set(es) & set(held)
    basin = np.array([data.split.basin_group[s] for s in data.sites])
    assert not set(basin[held]) & set(basin[np.r_[fit, es]]) and es.size and held.size
    si, ti = s1.eval_rows(data, held)
    d = data.cube.dates
    assert set(si.tolist()) <= set(held.tolist()) and si.size
    assert d[ti.min()] >= pd.Timestamp("2015-10-01") and d[ti.max()] + pd.Timedelta(days=14) <= pd.Timestamp("2021-09-30")
    ok = np.zeros(si.size, bool)
    for v in s1.VARS_DECISION:
        ok |= data.issue_ok(v, si, ti)
    assert ok.all()
    # statistic: mean over horizons of the median over sites of 1 − CRPS_hf / CRPS_gbm; sites with < min_rows dropped
    assert s1.crps_pinball(np.array([0.0]), np.zeros((1, 19)))[0] == 0
    rng = np.random.default_rng(0)
    n = 400
    sj = np.repeat(np.arange(4), 100)
    sj[-90:] = 9                                                 # site 3 keeps 10 rows (< 30), site 9 has 90
    y = {k: rng.normal(0, 1, n) for k in range(3)}
    okh = {k: np.ones(n, bool) for k in range(3)}
    from scipy.stats import norm
    z = norm.ppf(np.round(np.arange(0.05, 0.951, 0.05), 2))
    q_good = np.broadcast_to(z[None, None, :], (n, 3, 19)).copy()
    q_bad = q_good + 1.5
    st = s1.step1_statistic(sj, y, okh, q_good, q_bad, 30)
    assert st[0]["n_sites"] == 4 and st["mean"] > 0.3 and st[0]["n_rows"] == 400   # 4 of 5 sites have >= 30 rows
    assert s1.step1_statistic(sj, y, okh, q_bad, q_good, 30)["mean"] < 0
    assert abs(s1.step1_statistic(sj, y, okh, q_good, q_good, 30)["mean"]) < 1e-12
    q_nan = q_good.copy()
    q_nan[:5] = np.nan
    assert s1.step1_statistic(sj, y, okh, q_nan, q_bad, 30)[0]["n_rows_without_forecast"] == 5


def test_step1_eval_refuses_unequal_inputs(tmp_run, monkeypatch):
    """Verifier item P2-12/F07: the decision is never computed when the two phases read different perfect-prog
    samples or predicted different rows."""
    s1 = tmp_run.s1
    monkeypatch.setattr(s1, "guard", lambda what: {"row_id": "x"})
    monkeypatch.setattr(s1, "log_result", None, raising=False)
    s1.RUN.mkdir(parents=True)
    base = dict(q=np.zeros((3, 3, 2, 19), np.float32), si=np.arange(3), ti=np.arange(3) + 4000,
                sites=np.array(["a", "b", "c"]), pp_sha256="h1")
    np.savez(s1.RUN / "pred_hf.npz", **base)
    np.savez(s1.RUN / "pred_gbm.npz", **{**base, "pp_sha256": "h2"})
    with pytest.raises(RuntimeError, match="perfect-prog input of the HF phase is not that of the GBM phase"):
        s1.phase_eval()
    np.savez(s1.RUN / "pred_gbm.npz", **{**base, "ti": np.arange(3) + 4001})
    with pytest.raises(RuntimeError, match="different rows"):
        s1.phase_eval()
    np.savez(s1.RUN / "pred_gbm.npz", **base)
    from sentinel.models.sonde_v2 import parity
    with pytest.raises(parity.ParityError, match="no parity record"):
        s1.phase_eval()                                                  # no decision without the parity records
    assert not (s1.RUN / "STEP1_RESULT.json").exists()


# ----------------------------------------------------------------------------- the one product (GEFS c00)

def _synth_obs_weather(data, seed=3):
    rng = np.random.default_rng(seed)
    S, T = len(data.sites), len(data.cube.dates)
    seas = np.cos(2 * np.pi * (data.cube.dates.dayofyear.to_numpy() - 200) / 365.25)[None, :]
    o = np.zeros((S, T, len(NWP.FIELDS)), np.float32)
    tm = 12 + 9 * seas + rng.normal(0, 3, (S, T))
    o[..., NWP.F_IDX["tmax_c"]], o[..., NWP.F_IDX["tmin_c"]], o[..., NWP.F_IDX["tmean_c"]] = tm + 5, tm - 5, tm
    o[..., NWP.F_IDX["precip_mm"]] = np.maximum(rng.gamma(0.4, 6.0, (S, T)) - 1.0, 0)
    o[..., NWP.F_IDX["srad_wm2"]] = 180 + 90 * seas + rng.normal(0, 25, (S, T))
    return o


def _write_gefs(root, product, data, obs, first, last, n_sites=None, seed=0):
    """Month files in the layout of ``scripts/data/extract_gefs.py``: forecast = observation of the valid day
    + 1 °C warm bias + noise (temperature), × 1.25 wet bias (precipitation)."""
    from sentinel.models.sonde_v2 import prep as PR

    rng = np.random.default_rng(seed)
    d = root / product
    d.mkdir(parents=True)
    dates, T = data.cube.dates, len(data.cube.dates)
    S = np.arange(n_sites or len(data.sites))
    first, last = pd.Timestamp(first), pd.Timestamp(last)
    for m in PR.month_range(first.strftime("%Y%m"), last.strftime("%Y%m")):
        tsel = np.flatnonzero((dates.strftime("%Y%m") == m) & (dates >= first) & (dates <= last))
        si, ti = np.repeat(S, tsel.size), np.tile(tsel, S.size)
        df = {"site": np.asarray(data.sites)[si], "issue": dates[ti]}
        for k in range(1, 8):
            ok, tv = (ti + k) < T, np.minimum(ti + k, T - 1)

            def enc(v, sc):
                return np.where(ok, np.round(v * sc), -32768).astype(np.int16)

            e = rng.normal(0, 1.0, si.size)
            df[f"tmax_l{k}"] = enc(obs[si, tv, NWP.F_IDX["tmax_c"]] + 1.0 + e, 10)
            df[f"tmin_l{k}"] = enc(obs[si, tv, NWP.F_IDX["tmin_c"]] + 1.0 + e, 10)
            df[f"p_l{k}"] = enc(obs[si, tv, NWP.F_IDX["precip_mm"]] * 1.25, 10)
            df[f"sw_l{k}"] = enc(obs[si, tv, NWP.F_IDX["srad_wm2"]] + rng.normal(0, 10, si.size), 1)
        pd.DataFrame(df).to_parquet(d / f"{m}.parquet")


def test_products_build_single_product(tmp_path, synth_v2):
    """Hard errors for a missing month and for thin coverage; bias map and error model fitted on the reforecast
    only; the perfect-prog sample is reproducible by hash; the 2020 archive gap stays empty (no substitution)."""
    from sentinel.models.sonde_v2 import prep as PR
    from sentinel.models.sonde_v2 import products as P

    data = synth_v2
    assert PR.GEFS_REQUIRED["step1"] == {"reforecast": ("200710", "201912")}
    assert PR.GEFS_REQUIRED["cv"]["operational"] == ("202009", "202309") and P.PRODUCT == "gefs_c00"
    assert set(PR.month_range("200710", "201909")) <= set(PR.month_range(*PR.GEFS_REQUIRED["step1"]["reforecast"]))
    obs = _synth_obs_weather(data)
    root = tmp_path / "gefs"
    _write_gefs(root, "reforecast", data, obs, "2007-10-01", "2019-12-31")
    quiet = dict(root=root, log=lambda m: None)
    with mock.patch.object(P, "observed_weather_fields", return_value=obs):
        a = P.build(data, use="step1", **quiet)
        b = P.build(data, use="step1", **quiet)
        c = P.build(data, use="step1", seed=1, **quiet)
        with pytest.raises(FileNotFoundError, match="operational: 37 of 37 required month files missing"):
            P.build(data, use="cv", **quiet)
        _write_gefs(root, "operational", data, obs, "2020-09-23", "2023-09-30", seed=1)
        cvp = P.build(data, use="cv", **quiet)
        thin = tmp_path / "thin"
        (thin / "reforecast").parent.mkdir()
        (thin / "reforecast").symlink_to(root / "reforecast")
        _write_gefs(thin, "operational", data, obs, "2020-09-23", "2023-09-30", n_sites=8, seed=1)
        with pytest.raises(RuntimeError, match="GEFS coverage below 0.9: .*operational"):
            P.build(data, use="cv", root=thin, log=lambda m: None)
        (root / "reforecast" / "201305.parquet").rename(tmp_path / "201305.parquet")
        with pytest.raises(FileNotFoundError, match="reforecast: 1 of 147 required month files missing"):
            P.build(data, use="step1", **quiet)
    ia, ic = a.info, cvp.info
    assert ia["pp_sha256"] == b.info["pp_sha256"] and ia["pp_sha256"] != c.info["pp_sha256"]     # seed → sample
    assert ia["error_model"]["sha256"] == c.info["error_model"]["sha256"]
    assert ia["gefs_bc_sha256"] == b.info["gefs_bc_sha256"] and ia["product"] == "gefs_c00"
    assert ia["error_model"]["fit_end_valid_day"] == "2019-09-30" and ia["error_model"]["fit_sites"] == 8
    assert min(ia["coverage"].values()) == 1.0 and set(ic["coverage"]) > set(ia["coverage"])
    d, t = data.cube.dates, NWP.F_IDX["tmean_c"]
    gap = np.asarray((d >= "2020-01-01") & (d <= "2020-09-22"))
    ref = np.asarray(d <= "2019-12-31")
    assert not np.isfinite(a.gefs[:, ~ref]).any()                           # step 1: reforecast only
    assert not np.isfinite(cvp.gefs[:, gap]).any()                          # CV: the archive gap stays empty
    assert np.isfinite(cvp.gefs[:, np.asarray(d >= "2020-09-23") & np.asarray(d <= "2023-09-23")][..., t]).all()
    assert np.array_equal(a.gefs[:, ref], cvp.gefs[:, ref], equal_nan=True)  # same reforecast-only bias map
    # the bias map removes the +1 °C bias on the reforecast and — fitted on the reforecast only — is applied
    # unchanged to the operational product
    op = np.asarray(d >= "2020-09-23") & np.asarray(d <= "2023-09-20")
    assert abs(float(np.nanmean((a.gefs - a.obs_f)[:, ref][..., t]))) < 0.05
    assert abs(float(np.nanmean((cvp.gefs - cvp.obs_f)[:, op][..., t]))) < 0.1
    src = ia["bias_fit"]["source_cells"]
    assert ia["bias_fit"]["sites_all_identity_tmean"] == 0 and ia["bias_fit"]["slope_clip"] == [0.5, 1.5]
    assert isinstance(src, dict) and src
    # perfect-prog sample: observed future + sampled error, finite wherever the observed future is
    fin = np.isfinite(a.obs_f[..., t])
    assert np.isfinite(a.pp[..., t][fin]).all() and a.pp.shape == a.obs_f.shape
    err = (a.pp - a.obs_f)[..., t][fin]
    assert 0.5 < float(err.std()) < 1.5 and 0.5 < ia["error_model"]["sd_l1_tmean"][0] < 1.5


# ----------------------------------------------------------------------------- driver rehearsal (synthetic, CPU)

TINY_NET = dict(hourly=False, hidden_d=16, hidden_h=8, hidden_f=16, batch_size=8, steps_per_epoch=2, max_epochs=1,
                patience=2, es_max_samples=16, amp=False)


class _FakeChronos:
    """Stands in for the Chronos-2 pipeline: checks the row layout of every input and returns the last finite
    context value of each target plus a quantile ramp."""

    def __init__(self):
        self.quantiles = _chronos_pipe().quantiles
        self.n_inputs, self.future_ok = 0, 0

    def predict(self, inputs, prediction_length, batch_size, context_length):
        out = []
        ramp = torch.linspace(-1, 1, len(self.quantiles))[None, :, None]
        for inp in inputs:
            k, c, f = inp["n_targets"], inp["context"], inp["future_covariates"]
            assert c.shape[0] == f.shape[0] == k + inp["n_covariates"] and f.shape[1] == prediction_length
            assert inp["n_future_covariates"] == k + 3 and torch.isnan(f[:k]).all()
            assert torch.isnan(f[-3:, 7:]).all()                       # the product has 7 leads
            self.future_ok += int(torch.isfinite(f[-3:, :7]).all())
            last = torch.stack([row[torch.isfinite(row)][-1] for row in c[:k]])
            out.append(last[:, None, None] + ramp.expand(k, -1, prediction_length))
            self.n_inputs += 1
        return out


@pytest.fixture
def rehearsal(tmp_run, synth_v2, monkeypatch):
    """Stamp, declaration and every external input of the two drivers replaced by synthetic stand-ins (CPU)."""
    import json
    from copy import copy

    from sentinel.core import paths
    from sentinel.models.sonde import chronos as CH
    from sentinel.models.sonde import common, gpu_budget, runner
    from sentinel.models.sonde_v2 import declared as D
    from sentinel.models.sonde_v2 import prep as PR
    from sentinel.models.sonde_v2 import products as P

    cv, s1 = tmp_run.cv, tmp_run.s1
    data = copy(synth_v2)
    rng = np.random.default_rng(5)
    data.weather = rng.normal(5, 2, (len(data.sites), len(data.cube.dates), len(common.WEATHER_VARS))).astype(np.float32)
    data._rows = {}
    prod = SimpleNamespace(pp=_synth_product(data, seed=1), gefs=_synth_product(data, seed=2, first="2012-01-01"),
                           info={"pp_sha256": "pp-sha", "gefs_bc_sha256": "gefs-sha"})
    for arm, c in (("sonde_hf", "C1"), ("rnh", "R1")):
        monkeypatch.setitem(D.CONFIGS[arm], c, {**D.CONFIGS[arm][c], **TINY_NET})
    monkeypatch.setitem(D.STAGE2, "max_epochs", 1)
    monkeypatch.setitem(D.CONFIGS["chronos_cov"], "ZS", {**D.CONFIGS["chronos_cov"]["ZS"], "context": 32})
    for k, v in dict(ruled=True, max_rounds=3, early_stopping_rounds=3, num_threads=2, es_max_rows=300,
                     device="cpu", device_settings=None,
                     max_fit_rows={"Q19": None, "Q": None, "LS": None}).items():
        monkeypatch.setitem(D.GBM_SETTINGS, k, v)
    cv.CV.mkdir(parents=True, exist_ok=True)       # CV GBM gate (G-7d): a projection of 0 GPU-h, nothing spent
    (cv.CV / "gbm_budget.json").write_text(json.dumps({"total_gpu_h": 0.0, "units": []}))
    proc = tmp_run.runs / "processed"
    (proc / "sites").mkdir(parents=True)
    pd.DataFrame({"site_key": [f"USGS-{s}" for s in data.sites], "utc_offset_h": -6.0}).to_parquet(
        proc / "sites" / "sites.parquet")
    monkeypatch.setattr(paths, "PROCESSED", proc)
    monkeypatch.setattr(common, "site_registry", lambda *a, **k: pd.DataFrame({"site_type": "ST"}, index=data.sites))
    monkeypatch.setattr(common, "load_split", lambda *a, **k: (data.split, {}))
    monkeypatch.setattr(gpu_budget, "widen", lambda *a, **k: None)
    monkeypatch.setattr(runner, "setup_gpu", lambda *a, **k: -1)
    monkeypatch.setattr(cv, "DEVICE", "cpu")
    monkeypatch.setattr(s1, "DEVICE", "cpu")
    monkeypatch.setattr(cv, "load_data", lambda: data)
    monkeypatch.setattr(cv, "log", lambda m: None)
    monkeypatch.setattr(s1, "log", lambda m: None)
    monkeypatch.setattr(P, "build", lambda d, **k: prod)
    monkeypatch.setattr(PR, "prepare_v2", lambda **k: data)
    monkeypatch.setattr(s1, "setup", lambda: (data, prod, FIT, ES, HELD, {"n_fit_sites": int(FIT.size)}))
    pipe = _FakeChronos()
    monkeypatch.setattr(CH, "load_pipeline", lambda *a, **k: pipe)
    import scripts.eval.sonde_predict as SP
    monkeypatch.setattr(SP, "chronos_snapshot", lambda m: "snapshot")
    _stamp_row(tmp_run.root, tmp_run.reg)
    return SimpleNamespace(cv=cv, s1=s1, data=data, prod=prod, pipe=pipe, run=tmp_run)


def _registry(reg):
    import json
    return [json.loads(ln) for ln in reg.read_text().splitlines()]


def test_step1_rehearsal_declare_hf_gbm_eval(rehearsal):
    """Every step-1 phase end to end on synthetic data: files, parity records, one decision, registry rows."""
    import json

    from sentinel.models.sonde_v2 import parity

    s1, data = rehearsal.s1, rehearsal.data
    s1.RUN.mkdir(parents=True)
    assert s1.phase_declare() == 0 and s1.phase_hf() == 0 and s1.phase_gbm() == 0
    a, b = np.load(s1.RUN / "pred_hf.npz"), np.load(s1.RUN / "pred_gbm.npz")
    n = a["si"].size
    assert n > 0 and a["q"].shape == b["q"].shape == (n, 3, 2, 19) and set(a["si"].tolist()) <= set(HELD.tolist())
    assert np.isfinite(a["q"]).all() and np.isfinite(b["q"]).all()
    assert (data.cube.dates[a["ti"]] >= pd.Timestamp("2015-10-01")).all()
    assert (data.cube.dates[a["ti"]] + pd.Timedelta(days=14) <= pd.Timestamp("2021-09-30")).all()
    recs = {r["arm"]: r for r in parity.read_log() if r.get("use") == "step1"}
    assert set(recs) == {"sonde_hf", "gbm_nwp"}
    assert recs["sonde_hf"]["optimizer"]["stage1"]["weight_decay"] == 1e-5 and recs["sonde_hf"]["n_train_rows"] > 0
    assert recs["gbm_nwp"]["fit_window"]["per_variable_horizon"]["temp_c_mean|h3"]["fit_target_max"] <= "2021-09-30"
    r7 = recs["gbm_nwp"]["rows_per_variable_horizon"]["do_mgl_mean|h7"]
    assert r7["n_fit"] == r7["n_fit_total"] > 0 and r7["max_fit_rows"] is None and len(r7["fit_rows_sha256"]) == 64
    assert recs["gbm_nwp"]["gbm_ruling"]["sha256"] == \
        "3028fc1bd0509152e7428ad8a0ccbe27dece086027da9713bc9dae20bbf32c5b"
    pr = recs["gbm_nwp"]["gbm_processes"]
    assert len(pr) == 2 * 3 and all(p["processes"][0]["pid"] == pr[0]["processes"][0]["pid"] for p in pr)
    assert recs["gbm_nwp"]["boosters_per_variable_horizon"]["temp_c_mean|h3"]["q50"]["best_iteration"] >= 1
    assert s1.phase_hf() == 0                                           # resumes from DONE, predicts the same rows
    assert np.array_equal(np.load(s1.RUN / "pred_hf.npz")["q"], a["q"])
    n_reg = len(_registry(rehearsal.run.reg))
    assert s1.phase_eval() == 0
    res = json.loads((s1.RUN / "STEP1_RESULT.json").read_text())
    assert res["decision"].startswith(("BUILD", "DO NOT BUILD")) and res["stamp_row"] == "row1"
    assert all(np.isfinite(res[v]) for v in ("temp_c_mean", "do_mgl_mean"))
    assert res["temp_c_mean_h3"]["n_sites"] == 2 and res["temp_c_mean_h3"]["n_rows_without_forecast"] == 0
    rows = _registry(rehearsal.run.reg)[n_reg:]
    assert [r["target"] for r in rows] == ["temp_c_mean", "do_mgl_mean"]
    assert all(r["experiment"] == "sonde_arch_v2/step1" and r["extra"]["pp_sha256"] == "pp-sha" for r in rows)


def test_cv_rehearsal_every_arm_parity_combine(rehearsal):
    """CV driver end to end on synthetic data (CPU): folds, both networks (two stages), both GBM references, the
    Chronos-2-cov runner (stub pipeline), cross-arm parity, combine. No score is computed anywhere."""
    import json

    from sentinel.models.sonde import common
    from sentinel.models.sonde_v2 import declared as D
    from sentinel.models.sonde_v2 import parity

    cv, data = rehearsal.cv, rehearsal.data
    assert cv.phase_declare() == 0 and cv.phase_folds() == 0
    with pytest.raises(SystemExit, match="folds are fixed once"):
        cv.phase_folds()
    scored = [v for v in D.SCORED if v in data.cube.variables]
    for k in range(5):
        assert cv.phase_run(_args("sonde_hf", "C1", fold=k)) == 0
        assert cv.phase_run(_args("pf_gbm", "LS", fold=k)) == 0
    assert cv.phase_run(_args("sonde_hf", "C1", fold=0)) == 0                       # finished fold: nothing redone
    assert cv.phase_parity(_args()) == 0
    chk = json.loads((cv.CV / "PARITY_CHECK.json").read_text())
    assert chk["arms"] == [["pf_gbm", "LS"], ["sonde_hf", "C1"]] and len(chk["folds"]) == 5
    # SONDE-HF writes the scored and the auxiliary variables and the transformed DO % saturation
    d0 = cv.CV / "sonde_hf_C1_s0" / "fold0"
    done = json.loads((d0 / "OOF_DONE").read_text())
    names = {f.name.split("__")[1] for f in (d0 / "forecasts").glob("*.npz")}
    assert names == (set(done["variables"]) - set(done["no_held_rows"])) | {"do_pctsat"}
    assert set(scored) <= names and "discharge_cfs_mean" in names
    assert json.loads((d0 / "stage2" / "DONE").read_text())["optimizer"]["weight_decay"] == 1e-5
    fold, held = cv.fold_sites(data, 0)[2], None
    for v in sorted(names):
        fc, r = common.load_forecast(d0 / "forecasts" / f"sonde_hf_C1__{v}__val.npz"), cv.held_rows(data, v, fold)
        assert fc["q"].shape == (len(r), 7, 19) and np.isfinite(fc["q"]).all()
        assert (np.diff(fc["q"], axis=-1) >= 0).all() and set(fc["site_no"]) <= {data.sites[i] for i in fold}
        assert (fc["issue_date"] > np.datetime64("2021-09-30")).all() and fc["header"]["oof"] is True
    assert common.load_forecast(d0 / "forecasts" / "sonde_hf_C1__do_pctsat__val.npz")["header"]["transformed"]
    recs = [r for r in parity.read_log() if r.get("use") == "cv" and r["fold"] == 0]
    hf, gb = (next(r for r in recs if r["arm"] == a) for a in ("sonde_hf", "pf_gbm"))
    assert hf["optimizer"]["stage2"] == {"name": "AdamW", "lr": pytest.approx(1e-3 / 3), "weight_decay": 1e-5,
                                         "max_epochs": 1, "patience": 3}
    assert hf["input_sha256"] == {"pp": "pp-sha", "gefs_bc": "gefs-sha"} and hf["fit_window"]["target_max"] <= "2021-09-30"
    assert gb["ls_residual_rows"] == "early-stopping rows" and gb["train_sites_sha256"] == hf["train_sites_sha256"]
    assert set(gb["rows_per_variable_horizon"]) == {f"{v}|h{h}" for v in scored for h in D.HORIZONS}
    assert all(d["n_fit"] == d["n_fit_total"] and d["n_es"] <= 300          # G-2: full rows, ES rows capped
               for d in gb["rows_per_variable_horizon"].values())
    assert gb["device"] == "cpu" and gb["gbm_ruling"]["file"].endswith("gbm_lead_ruling_20261110.md")
    assert gb["boosters_per_variable_horizon"][f"{scored[0]}|h3"]["ls_loc"]["wall_s"] >= 0
    # the other arms on fold 0
    assert cv.phase_run(_args("gbm_nwp", "Q19", fold=0, target="temp_c_mean")) == 0   # one variable: not complete
    assert not (cv.CV / "gbm_nwp_Q19_s0" / "fold0" / "OOF_DONE").exists()
    assert cv.phase_run(_args("gbm_nwp", "Q19", fold=0)) == 0
    with pytest.raises(SystemExit, match="--target must be one of the scored"):
        cv.phase_run(_args("rnh", "R1", fold=0))
    with pytest.raises(SystemExit, match="no --target"):
        cv.phase_run(_args("sonde_hf", "C2", fold=0, target="temp_c_mean"))
    assert cv.phase_run(_args("rnh", "R1", fold=0, target="do_mgl_mean")) == 0
    rn = [f.name for f in (cv.CV / "rnh_R1_do_mgl_mean_s0" / "fold0" / "forecasts").glob("*.npz")]
    assert rn == ["rnh_R1__do_mgl_mean__val.npz"]
    assert cv.phase_run(_args("chronos_cov", "ZS", fold=0)) == 0
    pipe = rehearsal.pipe
    assert pipe.n_inputs > 0 and pipe.future_ok == pipe.n_inputs               # every val-block row had the product
    for arm, tag in (("gbm_nwp", "gbm_nwp_Q19"), ("chronos_cov", "chronos_cov_ZS")):
        fs = sorted((cv.CV / f"{tag}_s0" / "fold0" / "forecasts").glob("*.npz"))
        assert [f.name.split("__")[1] for f in fs] == sorted(scored)
        for f in fs:
            fc = common.load_forecast(f)
            r = cv.held_rows(data, f.name.split("__")[1], fold)
            assert fc["q"].shape == (len(r), 7, 19) and np.isfinite(fc["q"]).all(), f.name
    arms = [("sonde_hf", "C1"), ("pf_gbm", "LS"), ("gbm_nwp", "Q19"), ("rnh", "R1"), ("chronos_cov", "ZS")]
    ok = parity.check("cv", 0, require_arms=arms)
    assert ok["n_records"] >= 5
    with pytest.raises(parity.ParityError, match="no parity record for"):
        cv.phase_parity(_args())                                       # three arms have fold 0 only
    with pytest.raises(parity.ParityError, match="fold 1: no parity record for"):
        cv.phase_combine(_args("gbm_nwp", "Q19"))                      # combine needs all five folds
    # combine: every held-out val row of every site exactly once
    for arm, c in (("sonde_hf", "C1"), ("pf_gbm", "LS")):
        assert cv.phase_combine(_args(arm, c)) == 0
        for v in scored:
            fc = common.load_forecast(cv.CV / f"{arm}_{c}_s0" / "oof" / "forecasts" / f"{arm}_{c}__{v}__val.npz")
            rows = data.rows(v, "val")
            assert np.isfinite(common.align_forecast(fc, rows, D.HORIZONS)).all() and len(fc["q"]) == len(rows)
            assert fc["header"]["oof_folds"] == 5
    # a fold without a forecast file of a variable is accepted only when the run recorded "no held-out rows"
    f1 = cv.CV / "pf_gbm_LS_s0" / "fold1" / "forecasts" / "pf_gbm_LS__do_mgl_min__val.npz"
    f1.unlink()
    with pytest.raises(SystemExit, match=r"no forecast file in fold\(s\) \[1\]"):
        cv.phase_combine(_args("pf_gbm", "LS"))
    od = cv.CV / "pf_gbm_LS_s0" / "fold1" / "OOF_DONE"
    od.write_text(json.dumps({**json.loads(od.read_text()), "no_held_rows": ["do_mgl_min"]}))
    assert cv.phase_combine(_args("pf_gbm", "LS")) == 0
    reg = _registry(rehearsal.run.reg)
    mets = [r.get("metric") for r in reg]
    assert mets.count("oof_written") == 13
    gh = [r for r in reg if r.get("metric") == "gpu_hours"]             # G-7a: one row per GBM process
    assert len(gh) == 5 + 1 + 1 and {r["split"] for r in gh} == {"cv"} and all(r["value"] >= 0 for r in gh)
    assert all(r["extra"]["device"] == "cpu" and r["extra"]["gpu"] == -1 for r in gh)
    failed = [r["notes"] for r in reg if r.get("metric") == "run_failed"]     # the two refused calls above
    assert len(failed) == 2 and "--target must be one of" in failed[0] and "no --target" in failed[1]
    assert all(r["extra"]["stamp_row"] == "row1" for r in reg if r.get("metric") == "oof_written")
    del held
