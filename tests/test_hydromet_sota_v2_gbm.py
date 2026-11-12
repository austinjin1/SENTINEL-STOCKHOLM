# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""GBM references of the M6 architecture-v2 round (grab-sample formulation; synthetic arrays only)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("lightgbm")

from sentinel.hydromet_sota.gso_mt import gbm_ref as G  # noqa: E402
from sentinel.hydromet_sota.gso_mt.data import DAILY, fit_scales  # noqa: E402
from sentinel.hydromet_sota.gso_mt.synthetic import synth_v2  # noqa: E402
from sentinel.models.hydromet.features import FeatureSet  # noqa: E402

J = {k: DAILY.index(k) for k in DAILY}


def tiny():
    return G.lgbm_config({"num_leaves": 7, "n_estimators": 40, "min_child_samples": 5, "early_stopping_rounds": 10,
                          "learning_rate": 0.2, "no_inner_n_estimators": 30}, n_jobs=1)


@pytest.fixture(scope="module")
def v2():
    return synth_v2(8, 3, 6, 3, seed=1)


@pytest.fixture(scope="module")
def fs(v2):
    return FeatureSet.build(v2.data, lag=0)


def test_hour_block_shapes_products_and_missing_hour():
    x3 = np.array([[10.0, 2.0, 100.0], [1.0, 1.0, 1.0]])
    b = G.hour_block(np.array([6.0, np.nan]), x3, np.array([0.0, 1.0]))
    assert b.shape == (2, G.N_EXTRA) and len(G.extra_names()) == G.N_EXTRA
    assert b[0, 0] == pytest.approx(1.0) and abs(b[0, 1]) < 1e-6            # sin, cos of 6 h
    assert b[0, 4:7] == pytest.approx([10.0, 2.0, 100.0])                   # sin1 x (tmax, tmin, srad)
    assert np.isnan(b[1, :-1]).all() and b[1, -1] == 1.0 and b[0, -1] == 0.0


def test_inst_rows_kinds_caps_and_grab_switch(v2):
    r = G.inst_rows(v2, 0, v2.fit_idx, v2.grab_fit_idx, cap=50, seed=0)
    assert set(np.unique(r["src"])) == {0, 1, 2}
    d, h, g = (r["src"] == k for k in (0, 1, 2))
    assert np.isnan(r["hour"][d]).all() and (r["kind"][d] == 1).all() and (r["kind"][~d] == 0).all()
    assert np.isin(np.floor(r["hour"][h]), G.FIXED_HOURS).all()
    assert np.bincount(r["si"][d]).max() <= 50 and np.bincount(r["si"][h]).max() <= 50
    assert np.isin(r["si"][g], v2.grab_fit_idx).all() and np.isin(r["si"][~g], v2.fit_idx).all()
    assert g.sum() == np.isfinite(v2.inst_val[v2.grab_fit_idx][:, :, 0]).sum()          # grab rows are never capped
    assert np.allclose(r["y"][g], v2.inst_val[r["si"][g], r["ti"][g], 0])
    assert np.allclose(r["y"][d], v2.daily[r["si"][d], r["ti"][d], J["T_mean"]])
    r0 = G.inst_rows(v2, 0, v2.fit_idx, v2.grab_fit_idx, cap=50, seed=0, grab=False)
    assert (r0["src"] != 2).all()


def test_inst_model_learns_the_diel_cycle_and_weights_grab_rows(v2, fs):
    m = G.InstGBM("T", fs.columns(), tiny(), seed=0).fit(v2, fs, cap=400, share=0.5, log=lambda *_: None)
    n_dense, n_grab = sum(m.info["rows_by_src"][:2]), m.info["rows_by_src"][2]
    assert m.info["grab_row_weight"] == pytest.approx(n_dense / n_grab)       # share 0.5: equal total weights
    si = np.repeat(v2.inner_idx[:1], 40)
    ti = np.arange(2000, 2040)
    aft, night = m.at(fs, si, ti, np.full(40, 15.0)), m.at(fs, si, ti, np.full(40, 3.0))
    assert (aft - night).mean() > 0.3                                         # synthetic peak at 15 h
    mean24 = m.mean_over(fs, si, ti, G.HOURS24)
    assert np.all(mean24 < aft + 1e-6 + 2.0) and abs((mean24 - 0.5 * (aft + night)).mean()) < 1.5
    unt = m.at(fs, si, ti, np.full(40, np.nan))
    assert np.allclose(unt, m.mean_over(fs, si, ti, G.MIX_HOURS.astype(float)))
    it = m.it
    m.refit(v2, fs, log=lambda *_: None)
    assert m.model.n_estimators == max(it, 10) and m.info["refit_rows"] > m.info["n_train"]


def test_reference_selects_per_target_scores_and_predicts(v2, fs):
    sc = fit_scales(v2)
    ref = G.GBMRef("g1", tiny(), grid=((300, 0.5), (300, 0.25), (100, 0.5)), seed=0)
    out = ref.fit(v2, fs, sc, log=lambda *_: None)
    assert set(ref.sel) == {"temp_c_mean", "temp_c_7dadm", "do_mgl_mean", "do_mgl_min"}
    for v, rows in ref.info["candidates"].items():
        assert len(rows) == 5                                                 # 3 derived + 2 direct (distinct caps)
        best = min(rows, key=lambda r: r["inner_mse"])
        assert [best["form"], best["max_rows_per_site"], best["grab_share"]] == ref.info["selected"][v]
    assert set(ref.sel_inst) == {"T", "DO"} and ref.info["J"] is not None
    assert ref.info["J"] == pytest.approx(0.8 * ref.info["j_sonde"] + 0.2 * ref.info["j_grab"])
    y = v2.daily[v2.inner_idx]
    for k in ("T_mean", "T_7dadm", "DO_mean", "DO_min"):                       # a prediction on every inner label cell
        assert np.isfinite(out["daily"][:, :, J[k]][np.isfinite(y[:, :, J[k]])]).all()
    assert np.isfinite(out["inst"][np.isfinite(v2.inst_val[v2.grab_inner_idx])]).all()
    # only the selected members survive, refitted
    assert all("refit_rows" in m.info for m in list(ref.inst.values()) + list(ref.amp.values()) + list(ref.direct.values()))
    nt, nd = G.need_days(y)
    p = ref.predict_daily(fs, v2.inner_idx, nt, nd)
    for k in ("T_mean", "T_7dadm", "DO_mean", "DO_min"):
        assert np.isfinite(p[:, :, J[k]][np.isfinite(y[:, :, J[k]])]).all()
    a, t = np.nonzero(np.isfinite(v2.inst_val[v2.grab_inner_idx][:, :, 0]))
    gi = v2.grab_inner_idx[a]
    assert np.isfinite(ref.predict_inst(fs, "T", gi, t, v2.inst_hour[gi, t, 0])).all()


def test_no_grab_variant_trains_without_grab_rows_but_is_scored_on_the_inner_grab_set(v2, fs):
    ref = G.GBMRef("g1.nograb", tiny(), grid=((300, 0.0),), seed=0)
    ref.fit(v2, fs, fit_scales(v2), log=lambda *_: None)
    assert all(m["rows_by_src"][2] == 0 for m in ref.info["members"]["inst"].values())
    assert ref.info["j_grab"] is not None


def test_derived_7dadm_is_the_strict_trailing_mean_of_the_derived_daily_max(v2, fs):
    ref = G.GBMRef("g1", tiny(), grid=((300, 0.5),), seed=0)
    ref.fit(v2, fs, fit_scales(v2), log=lambda *_: None)
    ref.inst[("T", 300, 0.5)] = ref.inst.get(("T", 300, 0.5)) or G.InstGBM("T", fs.columns(), tiny()).fit(
        v2, fs, cap=300, share=0.5, log=lambda *_: None)
    ref.amp[("T", 300)] = ref.amp.get(("T", 300)) or G.DailyGBM("amp", fs.columns(), tiny()).fit(
        v2.data, fs, ref._amp_label(v2, "T"), v2.fit_idx, v2.inner_idx, cap=300, log=lambda *_: None)
    idx = v2.inner_idx[:1]
    nt = np.zeros((1, v2.daily.shape[1]), bool)
    nt[0, 1000:1010] = True
    d = ref._derived(fs, idx, nt, np.zeros_like(nt), 300, 0.5)
    assert np.isnan(d[0, 1000:1006, J["T_7dadm"]]).all() and np.isfinite(d[0, 1006:1010, J["T_7dadm"]]).all()
    assert d[0, 1006, J["T_7dadm"]] == pytest.approx(d[0, 1000:1007, J["T_max"]].mean(), abs=1e-4)
    assert (d[0, 1000:1010, J["T_max"]] >= d[0, 1000:1010, J["T_mean"]] - 1e-6).all()


def test_physics_and_stacked_inputs_are_added_and_pass_the_input_allow_list(v2, fs):
    te = v2.data.forc[:, :, 10].astype("float64")                             # any (S, T) driver stands in for Te
    f2 = G.physics_features(fs, te=te, solar=v2.solar, vaa=v2.vaa, elev_m=v2.data.static["elev_m"].to_numpy())
    assert len(f2.dyn_names) == len(fs.dyn_names) + len(G.TE_WINDOWS) + 3
    assert all(c in f2.static.columns for c in v2.vaa.columns)
    f3, models, info = G.add_stacks(v2, f2, tiny(), 0, log=lambda *_: None, max_rows_per_site=200)
    assert len(f3.dyn_names) == len(f2.dyn_names) + 3 * len(G.STACKED) and set(models) == set(G.STACKED)
    X = f3.design_matrix(np.array([0, 1]), np.array([500, 600]), f3.columns())   # raises on a refused column name
    assert X.shape[1] == len(f3.columns())
    j = f3.dyn_names.index("f_stk_tmax")
    assert np.isfinite(f3.dyn[:, 400:, j]).all()                              # fit, inner and grab locations all filled
    f4 = G.apply_stacks(f2, models)
    oth = np.setdiff1d(np.arange(len(v2.data.sites)), v2.fit_idx)
    assert np.allclose(f4.dyn[oth][:, :, j], f3.dyn[oth][:, :, j], equal_nan=True)   # full-fit outside the fit sites
    assert not np.allclose(f4.dyn[v2.fit_idx][:, 400:, j], f3.dyn[v2.fit_idx][:, 400:, j])   # out-of-fold at fit sites


def test_every_reference_gets_the_v2_input_additions_and_g2_adds_only_physics(v2, fs):
    f1 = G.v2_inputs(fs, v2.solar, v2.vaa)                                    # G1 (amendment §3)
    assert f1.dyn_names[-4:] == [f"f_solar_{c}" for c in G.SOLAR_CHANNELS]
    assert list(f1.static.columns[-3:]) == list(v2.vaa.columns) and f1.dyn.shape[-1] == fs.dyn.shape[-1] + 4
    te = v2.data.forc[:, :, 10].astype("float64")
    f2 = G.physics_features(f1, te=te, elev_m=v2.data.static["elev_m"].to_numpy())      # G2 on top of G1
    assert f2.dyn_names[len(f1.dyn_names):] == [f"f_te_m{w}" for w in G.TE_WINDOWS] + ["f_dosat_te"]
    assert len(set(f2.columns())) == len(f2.columns())                        # no column twice
    assert G.v2_inputs(fs, v2.solar, None).static.shape[1] == fs.static.shape[1]         # T-E schema: no VAA


def test_fixed_selection_fits_only_the_selected_members_and_selects_nothing(v2, fs):
    sc = fit_scales(v2)
    fixed = {"selected": {"temp_c_mean": ["derived", 300, 0.5], "temp_c_7dadm": ["direct", 100, None],
                          "do_mgl_mean": ["direct", 300, None], "do_mgl_min": ["derived", 100, 0.25]},
             "selected_inst": {"T": [300, 0.5], "DO": [300, 0.5]}}
    ref = G.GBMRef("g1", tiny(), seed=1, fixed=fixed)
    out = ref.fit(v2, fs, sc, log=lambda *_: None)
    assert ref.info["fixed_selection"] and ref.info["selected"] == fixed["selected"]
    assert ref.info["selected_inst"] == fixed["selected_inst"]
    assert set(ref.inst) == {("T", 300, 0.5), ("DO", 100, 0.25), ("DO", 300, 0.5)}
    assert set(ref.amp) == {("T", 300), ("DO", 100)} and set(ref.direct) == {("T_7dadm", 100), ("DO_mean", 300)}
    y = v2.daily[v2.inner_idx]
    for k in ("T_mean", "T_7dadm", "DO_mean", "DO_min"):
        assert np.isfinite(out["daily"][:, :, J[k]][np.isfinite(y[:, :, J[k]])]).all()
    assert ref.info["J"] is not None
    # the no-grab variant of the same selection: shares set to 0, same forms and caps
    ng = G.no_grab_selection(fixed)
    assert ng["selected"]["temp_c_mean"] == ["derived", 300, 0.0] and ng["selected"]["temp_c_7dadm"] == ["direct", 100, None]
    assert ng["selected_inst"] == {"T": [300, 0.0], "DO": [300, 0.0]}
    r2 = G.GBMRef("g1.nograb", tiny(), seed=0, fixed=ng)
    r2.fit(v2, fs, sc, log=lambda *_: None)
    assert all(m["rows_by_src"][2] == 0 for m in r2.info["members"]["inst"].values())
