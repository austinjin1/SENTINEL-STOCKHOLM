# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Tabular models and row frames of M7 architecture round v2 — synthetic data only (no real row is loaded)."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from sentinel.models.lite import baselines as B
from sentinel.models.litev2 import labels2b as L
from sentinel.models.litev2 import rows as RW
from sentinel.models.litev2 import tabular as T

FAST = (0,)


def _synthetic(n=320, n_wb=40, seed=0):
    rng = np.random.default_rng(seed)
    cols = T.parity_columns()
    X = pd.DataFrame(rng.uniform(0.1, 1.0, size=(n, len(cols))), columns=cols)
    wb = np.array([f"wb{int(i)}" for i in rng.integers(0, n_wb, n)])
    sig = 2.0 * X["c9s_R"].to_numpy() - X["c9s_B"].to_numpy()
    Y = pd.DataFrame({f"y_{q}": np.nan for q in T.QUANTITIES}, index=X.index)
    Y["y_secchi"] = -sig + 0.05 * rng.normal(size=n)
    unit = rng.integers(0, 3, n)                         # 0 NTU, 1 FNU, 2 NTRU; one unit per row
    off = np.array([0.0, 0.5, -0.5])
    for k, u in enumerate(("turbidity", "turb_fnu", "turb_ntru")):
        Y.loc[unit == k, f"y_{u}"] = (sig + off[k] + 0.05 * rng.normal(size=n))[unit == k]
    Y.loc[::2, "y_tss"] = (sig + 0.05 * rng.normal(size=n))[::2]
    Y.loc[::5, "y_ssc"] = (sig + 0.05 * rng.normal(size=n))[::5]
    Y.loc[::3, "y_chla"] = (X["c9s_G"].to_numpy() + 0.05 * rng.normal(size=n))[::3]
    return X, Y, wb, sig


def test_feature_lists():
    assert len(T.chip_columns()) == 45
    cols = T.parity_columns()
    assert len(cols) == 98 and len(T.SCENE_COLS) == 50 and cols[-3:] == ["lat", "lon", "doy"]
    assert set(T.SCENE_COLS) <= set(cols)
    assert T.QUANTITIES == (*L.TARGETS, "turb_ntru")


def test_anchor_fold_rule():
    for w in ("N2:wb12", "p2:BLC-1", "F:wb3"):
        assert T.anchor_fold(w) == int(hashlib.sha256(("SENTINEL-Lite-anchor|" + w).encode()).hexdigest()[:8], 16) % 5
    wb = np.array(["a", "b", "a", "c", "b"])
    f = T.folds(wb)
    assert f[0] == f[2] and f[1] == f[4] and set(f) <= set(range(5))


def test_fit_gbm_equals_the_frozen_function():
    X, Y, _, _ = _synthetic(200)
    cols = T.chip_columns()
    y = Y["y_secchi"].to_numpy().copy()
    y[::7] = np.nan
    frozen, _ = B.gbm(X.iloc[:150], y[:150], X.iloc[150:], cols=cols)
    mine = T.predict_gbm(T.fit_gbm(X.iloc[:150][cols].to_numpy(), y[:150]), X.iloc[150:][cols].to_numpy())
    assert np.allclose(frozen, mine, atol=1e-12)
    assert np.isnan(T.predict_gbm(T.fit_gbm(X[cols].to_numpy(), np.full(len(X), np.nan)), X[cols].to_numpy())).all()


def test_pooled_turbidity_uses_unit_indicators():
    X, Y, _, _ = _synthetic(600)
    Xp, yp = T.turbidity_stack(X.to_numpy(), Y)
    n_u = [int(Y[f"y_{u}"].notna().sum()) for u in T.TURB_UNITS]
    assert len(yp) == sum(n_u) and Xp.shape[1] == 100
    assert Xp[:n_u[0], -2:].sum() == 0 and (Xp[n_u[0]:n_u[0] + n_u[1], -2] == 1).all() and (Xp[-n_u[2]:, -1] == 1).all()
    m = T.fit_parity(X, Y, seeds=FAST)
    p = T.predict_parity(m, X)
    assert list(p.columns) == list(T.QUANTITIES) and np.isfinite(p.to_numpy()).all()
    assert (p["turb_fnu"] - p["turbidity"]).mean() > 0.3 and (p["turb_ntru"] - p["turbidity"]).mean() < -0.3
    m7 = T.fit_parity(X, Y, pooled=False, seeds=FAST)                       # S7: per unit, NTRU unused
    p7 = T.predict_parity(m7, X)
    assert p7["turb_ntru"].isna().all() and np.isfinite(p7[["turbidity", "turb_fnu"]].to_numpy()).all()
    assert np.allclose(p7["secchi"], p["secchi"])                           # differs for turbidity only
    assert not np.allclose(p7["turbidity"], p["turbidity"])


def test_out_of_fold_anchor_never_sees_its_own_water_body():
    X, Y, wb, _ = _synthetic(300)
    fold = T.folds(wb)
    a, info = T.oof_parity(X, Y, wb, seeds=FAST)
    assert np.isfinite(a.to_numpy()).all() and sum(info["rows_per_fold"]) == len(X)
    Y2 = Y.copy()
    k = int(np.bincount(fold).argmax())
    Y2.loc[fold == k] = Y2.loc[fold == k] + 3.0                             # change the labels of fold k only
    a2, _ = T.oof_parity(X, Y2, wb, seeds=FAST)
    assert np.allclose(a.loc[fold == k], a2.loc[fold == k])                # its own anchors do not move
    assert not np.allclose(a.loc[fold != k], a2.loc[fold != k])            # the others do


def test_anchors_and_frozen_list_shapes():
    X, Y, wb, _ = _synthetic(300)
    rows = pd.concat([Y, pd.DataFrame({"lat": X["lat"] * 40, "lon": -X["lon"] * 100, "site_id": wb})], axis=1)
    out = T.anchors(X.iloc[:240], Y.iloc[:240], wb[:240], {"ev": X.iloc[240:]}, seeds=FAST)
    assert out["train"].shape == (240, 7) and out["ev"].shape == (60, 7)
    assert set(T.parity_predictions(out["model"], X.iloc[240:])) == set(L.TARGETS)
    fl = T.frozen_list(rows.iloc[:240], X.iloc[:240], rows.iloc[240:], X.iloc[240:])
    assert {"S1_train_median", "S2_loc_knn", "S3_loc_season_gbm", "S4_gbm_rgbn", "S5_gbm_ctx"} <= set(fl)
    assert set(fl["S5_gbm_ctx"]) == set(L.TARGETS)
    forms = {k for k in fl if k.startswith("P1_")}
    assert forms == {f"P1_{f}" for t in ("secchi", "turbidity", "chla") for f in B.BAND_RATIO_FORMS[t]}
    assert set(fl["P1_red_nir_log"]) == {"turbidity", "turb_fnu", "tss", "ssc"}     # FNU and SSC use those forms


def test_s8_columns_and_run():
    assert "a_secchi" not in T.s8_extra_columns("secchi") and "a_turb_ntru" in T.s8_extra_columns("secchi")
    tp = T.s8_extra_columns("turb_pooled")
    assert not {"a_turbidity", "a_turb_fnu", "a_turb_ntru"} & set(tp) and set(T.TEACHER_COLS) <= set(tp)
    X, Y, wb, sig = _synthetic(240)
    teacher = pd.DataFrame({c: sig + 0.1 for c in T.TEACHER_COLS})
    teacher.iloc[::4] = np.nan                                               # no pairing / masked
    preds, info = T.fit_predict_s8(X.iloc[:200], Y.iloc[:200], wb[:200], teacher.iloc[:200], X.iloc[200:], seeds=FAST)
    assert set(preds) == set(L.TARGETS) and all(np.isfinite(v).all() and len(v) == 40 for v in preds.values())
    assert info["n_teacher_rows"]["tmu_secchi"] == 150


def test_teacher_chip_means_are_trust_weighted():
    kd = {"t_mu": np.array([[[1.0, np.nan], [3.0, 2.0]], [[np.nan, np.nan], [np.nan, 5.0]]], np.float32),
          "t_w": np.array([[[1.0, 0.0], [3.0, 0.0]], [[0.0, 0.0], [0.0, 2.0]]], np.float32)}
    T_COLS = T.TEACHER_COLS
    try:
        T.TEACHER_COLS = ("tmu_a", "tmu_b")
        m = T.teacher_chip_means(kd)
    finally:
        T.TEACHER_COLS = T_COLS
    assert np.isclose(m.iloc[0, 0], 2.5) and np.isnan(m.iloc[0, 1]) and np.isnan(m.iloc[1, 0]) and m.iloc[1, 1] == 5.0


def test_aft_bounds_and_fit():
    y = np.array([0.0, np.nan, np.nan, np.nan, 1.0])
    cens = np.array([0, -1, 1, 0, 0])
    lim = np.array([np.nan, -1.0, 2.0, np.nan, np.nan])
    use, lo, hi = T.aft_bounds(y, cens, lim)
    assert use.tolist() == [True, True, True, False, True]
    assert np.allclose(lo, [1.0, 0.0, 100.0, 10.0]) and np.allclose(hi[[0, 1, 3]], [1.0, 0.1, 10.0]) and np.isinf(hi[2])
    rng = np.random.default_rng(1)
    X = rng.normal(size=(600, 5))
    truth = 0.8 * X[:, 0]
    yl = truth + 0.05 * rng.normal(size=600)
    c = np.where(yl < -0.5, -1, 0)                                           # left-censored below the limit
    p, info = T.fit_predict_aft(X[:500], np.where(c[:500] != 0, np.nan, yl[:500]), c[:500],
                                np.where(c[:500] != 0, -0.5, np.nan), X[500:], seeds=FAST)
    assert info["n_fit"] == 500 and info["n_left"] == int((c[:500] != 0).sum()) > 20 and info["n_right"] == 0
    assert np.corrcoef(p, truth[500:])[0, 1] > 0.9
    rows = pd.DataFrame({f"cens_{t}": 0 for t in L.TARGETS}, index=range(30))
    rows.loc[:24, "cens_secchi"] = 1
    rows.loc[:9, "cens_tss"] = -1
    assert T.aft_targets(rows) == {"secchi": 25}


def test_tabpfn_row_rule():
    wb = np.array(["a"] * 4 + ["b"] * 3 + ["c"] * 5 + ["d"] * 2)
    assert T.tabpfn_keep(wb, cap=14).all()
    keep = T.tabpfn_keep(wb, cap=8)
    order = sorted("abcd", key=lambda w: hashlib.sha256(("SENTINEL-Lite-tabpfn|" + w).encode()).hexdigest())
    size, total, want = {"a": 4, "b": 3, "c": 5, "d": 2}, 0, set()
    for w in order:
        if total + size[w] > 8:
            break
        want.add(w)
        total += size[w]
    assert set(wb[keep]) == want and keep.sum() == total <= 8
    for w in set(wb):                                                        # whole water bodies only
        assert keep[wb == w].all() or not keep[wb == w].any()


def test_cyfi_and_context_blocks():
    rng = np.random.default_rng(2)
    chips = rng.integers(0, 255, size=(3, 4, 256, 256), dtype=np.uint8)
    chips[2, 3] = 255                                                        # no water pixel on the third chip
    rows = pd.DataFrame({"naip_date": pd.to_datetime(["2019-06-01", "2020-08-15", "2021-10-03"]),
                         "dt_label_days": [0.0, -1.0, np.nan], "lat": [40.0, 30.0, 45.0], "lon": [-90.0, -100.0, -75.0],
                         "doy": [152, 228, 276], "gsd_m": [0.6, 0.6, 1.0]})
    cy = T.cyfi_features(rows, chips)
    assert cy.shape == (3, 30) and cy["cy_month"].tolist() == [6.0, 8.0, 10.0] and cy["cy_pct_water"].iloc[2] == 0.0
    assert np.isfinite(cy.drop(columns="cy_dt_label_days").to_numpy()).all()
    scene = pd.DataFrame(rng.uniform(1, 200, size=(3, 50)), columns=list(T.SCENE_COLS))
    sc = T.context_block(rows, scene)
    assert sc.shape == (3, 56)
    assert np.allclose(sc[["c_x", "c_y", "c_z"]].pow(2).sum(axis=1), 1.0)
    X = T.feature_table(rows, chips, scene)
    assert list(X.columns) == T.parity_columns() and np.isfinite(X.to_numpy()).all()


def test_stacked_and_pca():
    X, Y, _, _ = _synthetic(200)
    rng = np.random.default_rng(3)
    E = rng.normal(size=(200, 40))
    p = T.pca_fit(E[:160])
    z = T.pca_apply(p, E[:160])
    assert z.shape == (160, 16) and np.allclose(z.mean(axis=0), 0.0, atol=1e-9)
    assert np.allclose(np.cov(z.to_numpy().T) - np.diag(np.diag(np.cov(z.to_numpy().T))), 0.0, atol=1e-8)
    preds, info = T.fit_predict_stacked(X.iloc[:160], Y.iloc[:160], E[:160], X.iloc[160:], E[160:], seeds=FAST)
    assert set(preds) == set(L.TARGETS) and info["n_pcs"] == 16 and len(preds["secchi"]) == 40


def _frame(n=4):
    d = {c: [None] * n for c in RW.KEY_COLS}
    d["uid"] = [f"c{i}" for i in range(n)]
    d["role"] = ["train"] * n
    df = pd.DataFrame(d)
    for c in (*RW.Y_COLS, *RW.LIM_COLS):
        df[c] = np.nan
    for c in RW.CENS_COLS:
        df[c] = 0
    return df


def test_row_frame_rules():
    RW.check(_frame())
    bad = _frame()
    bad.loc[1, "uid"] = "c0"
    with pytest.raises(RW.RowError, match="duplicate"):
        RW.check(bad)
    bad = _frame()
    bad.loc[0, ["y_secchi", "cens_secchi", "lim_secchi"]] = [0.1, 1, 0.5]
    with pytest.raises(RW.RowError, match="censored row also has"):
        RW.check(bad)
    bad = _frame()
    bad.loc[0, "cens_tss"] = -1
    with pytest.raises(RW.RowError, match="no limit"):
        RW.check(bad)
    for role in ("test", "test_n", "test_time", "test_time_n", "test_p2"):
        bad = _frame()
        bad.loc[2, "role"] = role
        with pytest.raises(RW.RowError, match="protected"):
            RW.check(bad)
    with pytest.raises(RW.RowError):
        RW.p2_rows(["test_p2"])
    with pytest.raises(RW.RowError):
        RW.eval_rows("test")
