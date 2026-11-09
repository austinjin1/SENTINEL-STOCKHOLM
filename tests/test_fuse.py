# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Tests for SENTINEL-Fuse (synthetic arrays only — allowed inside unit tests, R9)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from sentinel.data import fuse_tiers as ftd
from sentinel.models.fuse import baselines as bl
from sentinel.models.fuse import plugins as pg
from sentinel.models.fuse import splits as sp
from sentinel.models.fuse.data import FoldPreprocessor, ModalitySpec, group_columns
from sentinel.models.fuse.model import FuseConfig, FuseModel, crps_gaussian_t, masked_loss
from sentinel.models.fuse.train import TrainConfig, dropout_keep, pattern_weights

torch.set_num_threads(2)


def _batch(B, groups, present, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = {}
    for name, gs in groups.items():
        G, K = len(gs), max(gs)
        x = torch.randn(B, G, K, generator=g)
        m = (torch.rand(B, G, K, generator=g) > 0.2).float()
        for gi, k in enumerate(gs):
            m[:, gi, k:] = 0
        out[name] = {"x": x, "m": m, "present": present[name].clone(), "age": torch.rand(B, generator=g) * 30}
    return out


TARGETS = [("do", 7, "gauss"), ("turb", 7, "gauss"), ("H1", 7, "binary"), ("lab_tn", 0, "gauss")]


def _model(groups, seed=0):
    torch.manual_seed(seed)
    m = FuseModel(FuseConfig(d=32, n_latents=8, n_layers=2, n_heads=4, tok_hidden=16, dropout=0.0), groups, TARGETS)
    return m.eval()


def test_absent_modality_equals_not_provided():
    groups = {"sonde": [3, 2], "chem": [2, 2, 1], "wx": [4]}
    B = 6
    pres = {"sonde": torch.ones(B, dtype=torch.bool), "chem": torch.zeros(B, dtype=torch.bool),
            "wx": torch.ones(B, dtype=torch.bool)}
    model = _model(groups)
    b = _batch(B, groups, pres)
    with torch.no_grad():
        a = model(b)
        b2 = {k: v for k, v in b.items() if k != "chem"}
        c = model(b2)
    assert torch.equal(a, c)


def test_plug_and_play_bit_identical_when_absent():
    groups = {"sonde": [3, 2], "wx": [4]}
    B = 8
    model = _model(groups)
    pres = {"sonde": torch.tensor([1, 1, 0, 1, 0, 1, 1, 1], dtype=torch.bool), "wx": torch.ones(B, dtype=torch.bool)}
    b = _batch(B, groups, pres)
    with torch.no_grad():
        before = model(b).clone()
    new_params = model.add_modality("micro", [4, 4], freeze_core=True)
    assert all(p.requires_grad for p in new_params)
    assert not any(p.requires_grad for n, p in model.named_parameters()
                   if not (n.startswith("tokenizers.micro") or n.startswith("cross.micro")))
    # new modality absent everywhere -> bit-identical
    b_abs = dict(b)
    b_abs.update(_batch(B, {"micro": [4, 4]}, {"micro": torch.zeros(B, dtype=torch.bool)}, seed=3))
    with torch.no_grad():
        after = model(b_abs)
    assert torch.equal(before, after)
    # present for some rows -> only those rows change
    pm = torch.tensor([1, 0, 0, 1, 0, 0, 0, 0], dtype=torch.bool)
    b_some = dict(b)
    b_some.update(_batch(B, {"micro": [4, 4]}, {"micro": pm}, seed=3))
    with torch.no_grad():
        mixed = model(b_some)
    assert torch.equal(mixed[~pm], before[~pm])
    assert not torch.equal(mixed[pm], before[pm])


def test_row_independence():
    groups = {"sonde": [3, 2], "wx": [4]}
    model = _model(groups)
    B = 5
    pres = {"sonde": torch.ones(B, dtype=torch.bool), "wx": torch.ones(B, dtype=torch.bool)}
    b = _batch(B, groups, pres)
    with torch.no_grad():
        full = model(b)
        one = model({k: {kk: vv[2:3] for kk, vv in v.items()} for k, v in b.items()})
    assert torch.allclose(full[2:3], one, atol=1e-5)


def test_masked_loss_censoring_and_masks():
    out = torch.zeros(4, 2, 2, requires_grad=True)
    y = torch.tensor([[0.5, 1.0], [0.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
    obs = torch.tensor([[1, 1], [0, 1], [1, 0], [0, 0]], dtype=torch.bool)
    cens = torch.tensor([[0, 0], [0, 0], [1, 0], [0, 0]], dtype=torch.bool)
    is_bin = torch.tensor([False, True])
    loss, info = masked_loss(out, y, obs, cens, is_bin, beta=0.0)
    assert torch.isfinite(loss) and info["n_active_targets"] == 2
    loss.backward()
    # censored row 2 (bound -1 below mu=0): Tobit pushes mu down (positive gradient on mu)
    assert out.grad[2, 0, 0] > 0
    # unobserved rows get no gradient
    assert out.grad[3].abs().sum() == 0


def test_crps_matches_numpy():
    y = np.array([0.3, -1.0, 2.0])
    mu = np.array([0.0, 0.0, 1.0])
    s = np.array([1.0, 0.5, 2.0])
    a = crps_gaussian_t(torch.tensor(y), torch.tensor(mu), torch.tensor(s)).numpy()
    b = bl.crps_gauss_rows(y, mu, s)
    assert np.allclose(a, b, atol=1e-6)


def test_dropout_schedule_never_adds_absent():
    groups = {"sonde": [2], "chem": [2], "micro": [2], "wx": [2]}
    B = 400
    g = torch.Generator().manual_seed(1)
    pres = {m: torch.rand(B, generator=g) > 0.4 for m in groups}
    b = _batch(B, groups, pres)
    keep = dropout_keep(b, {"wx"}, torch.Generator().manual_seed(0), TrainConfig(), active=True)
    for m, k in keep.items():
        assert not (k & ~pres[m]).any()
    assert "wx" not in keep
    n_kept = torch.stack([keep[m] for m in keep], 1).sum(1)
    n_avail = torch.stack([pres[m] for m in keep], 1).sum(1)
    frac_full = ((n_kept == n_avail) & (n_avail >= 2)).float().mean()
    assert 0.1 < frac_full < 0.6
    assert dropout_keep(b, {"wx"}, torch.Generator().manual_seed(0), TrainConfig(), active=False) is None


def test_pattern_weights_upweight_rare():
    tens = {"a": {"present": np.array([1] * 90 + [0] * 10, bool)}, "hub": {"present": np.ones(100, bool)}}
    w = pattern_weights(tens, {"hub"}, 0.5)
    assert w[95] > w[0] and np.isclose(w.sum(), 1)


def test_latest_event_strict_and_age():
    cal = pd.date_range("2020-01-01", "2020-01-10")
    ev = pd.Series(pd.to_datetime(["2020-01-03", "2020-01-07"]))
    pos, age = ftd.latest_event(cal, ev, max_age=3, strictly_before=True)
    assert pos[2] == -1 and pos[3] == 0 and age[3] == 1          # 01-03 not visible on 01-03
    assert pos[6] == -1 and pos[7] == 1                          # 01-07 visible from 01-08 (01-03 too old)
    pos2, _ = ftd.latest_event(cal, ev, max_age=3, strictly_before=False)
    assert pos2[2] == 0 and pos2[6] == 1
    p3, a3 = ftd.latest_event(cal, pd.Series([], dtype="datetime64[ns]"), 3, True)
    assert (p3 == -1).all() and np.isnan(a3).all()


def test_sensor_features_causal():
    cal = pd.date_range("2020-01-01", periods=60)
    rs = np.random.default_rng(0)
    z = pd.DataFrame({"v__mean": rs.normal(size=60), "v__min": rs.normal(size=60) - 1,
                      "v__max": rs.normal(size=60) + 1}, index=cal)
    f1 = ftd.sensor_features(z, cal, "sonde", "v")
    z2 = z.copy()
    z2.iloc[40:] = 99.0      # change the future
    f2 = ftd.sensor_features(z2, cal, "sonde", "v")
    pd.testing.assert_frame_equal(f1.iloc[:40], f2.iloc[:40])


def test_station_daily_drops_provisional_and_prefers_downstream():
    df = pd.DataFrame({"date": pd.to_datetime(["2020-01-01"] * 3 + ["2020-01-02"] * 2),
                       "hor": ["101", "102", "112", "102", "112"], "ver": ["100"] * 5,
                       "x_mean": [1.0, 2.0, 3.0, np.nan, 5.0], "x_min": [0.0] * 5, "x_max": [9.0] * 5,
                       "provisional": [False, False, False, False, True]})
    s = ftd.station_daily(df, ["x"])
    assert s.loc["2020-01-01", "x__mean"] == 2.0      # 102 preferred, 101 never used
    assert "2020-01-02" not in s.index.strftime("%Y-%m-%d")  # 112 provisional dropped, 102 NaN


def test_any_exceed_label():
    x = np.array([5, 5, 5, 2, 5, 5, 5, 5, 5, 5], float)
    y, v = ftd._any_exceed(x, "H1_hypoxia", 2)
    assert y[1] == 1 and y[2] == 1 and y[3] == 0


def test_group_columns():
    cols = ["sonde__do__last", "sonde__do__m7", "sonde__ph__last", "static__a", "static__b"]
    g, n = group_columns("sonde", cols, chunk=1)
    assert g[0] == [0, 1] and g[1] == [2] and n[:2] == ["do", "ph"] and len(g) == 4


def test_fold_preprocessor_fits_train_only():
    df = pd.DataFrame({"a__x": np.r_[np.zeros(50), np.full(50, 100.0)], "a__present": True})
    spec = ModalitySpec("a", ["a__x"], [[0]], ["x"], "a__present")
    tr = np.r_[np.ones(50, bool), np.zeros(50, bool)]
    pre = FoldPreprocessor([spec]).fit(df, tr)
    assert pre.center_["a"][0] == 0.0
    t = pre.tensors(df)["a"]
    assert t["x"].shape == (100, 1, 1) and t["x"][-1, 0, 0] == 10.0   # clipped, never refit on eval rows


def test_tier_d_split_rules():
    rows = []
    for i in range(12):
        for j, kind in enumerate(["Aquatic Wadeable Stream", "Aquatic Lake" if i % 4 == 0 else "Aquatic Wadeable Stream",
                                  "Aquatic Non-wadeable Stream" if i % 5 == 0 else "Aquatic Wadeable Stream"]):
            rows.append({"site": f"S{i}{j}", "domain": f"D{i:02d}", "aquatic_type": kind})
    s = pd.DataFrame(rows)
    obj = sp.make_tier_d_split(s, min_test_sites=6, max_test_sites=9)
    test = {k for k, v in obj["site_role"].items() if v == "test"}
    dev = {k for k, v in obj["site_role"].items() if v == "dev"}
    assert not test & dev and len(test) >= 6
    assert obj["counts"]["test_lakes"] >= 1 and obj["counts"]["test_rivers"] >= 1
    doms_t = set(s[s.site.isin(test)].domain)
    assert not doms_t & set(s[s.site.isin(dev)].domain)      # whole domains held out


def test_tier_c_split_buffer_and_disjoint():
    rs = np.random.default_rng(0)
    n = 600
    r = pd.DataFrame({"site_id": [f"s{i}" for i in range(n)], "lat": rs.uniform(30, 45, n), "lon": rs.uniform(-100, -80, n),
                      "huc2": rs.choice([f"{h:02d}" for h in range(1, 12)], n)})
    obj = sp.make_tier_c_split(r)
    role = pd.Series(obj["site_role"])
    xy = r.set_index("site_id")
    from sentinel.eval.splits import nearest_distance_km
    dev = xy.loc[role[role == "dev"].index]
    tst = xy.loc[role[role == "test_spatial"].index]
    d = nearest_distance_km(dev.lat, dev.lon, tst.lat, tst.lon)
    assert (d > 5.0).all()
    assert set(xy.loc[role[role == "test_region"].index, "huc2"]).isdisjoint(set(dev["huc2"]))
    assert set(obj["dev_folds"]["block20"]) == set(dev.index)


def test_save_split_never_overwrites(tmp_path):
    p = tmp_path / "s.json"
    sha = sp.save_split({"a": 1}, p)
    with pytest.raises(FileExistsError):
        sp.save_split({"a": 2}, p)
    assert sp.load_split(p, sha) == {"a": 1}
    with pytest.raises(ValueError):
        sp.load_split(p, "0" * 64)


def test_crossfit_check():
    class P:
        name = "m3"
        model_registry_id = None
        training_sites = {"A", "B"}
        fold_aware = False

        def featurize(self, rows, exclude_sites=None):
            return pd.DataFrame({"m3__present": True}, index=rows.index)

    rows = pd.DataFrame({"site": ["A", "C"]})
    with pytest.raises(pg.CrossFitError):
        pg.check_crossfit(P(), rows)
    excl = pd.Series([{"A"}, set()], index=rows.index)
    assert pg.check_crossfit(P(), rows, per_row_exclusions=excl)["ok"]
    with pytest.raises(pg.CrossFitError):
        pg.check_crossfit(P(), rows, test_sites={"B"}, per_row_exclusions=excl)


def test_acceptance_test():
    ok = pg.acceptance_test({"do": (0.05, 0.01, 0.09), "turb": (0.0, -0.005, 0.006)})
    assert ok["accepted"]
    bad = pg.acceptance_test({"do": (0.05, 0.01, 0.09), "turb": (-0.05, -0.08, -0.02)})
    assert not bad["accepted"] and bad["non_inferiority_failures"] == ["turb"]


def test_mixture_moments():
    mu, sd = bl.mixture_moments(np.array([[0.0, 2.0]]), np.array([[1.0, 1.0]]), np.array([[1, 1]]))
    assert np.isclose(mu[0], 1.0) and np.isclose(sd[0], np.sqrt(2.0))
    mu, sd = bl.mixture_moments(np.array([[0.0, 2.0]]), np.array([[1.0, 1.0]]), np.array([[1, 0]]))
    assert np.isclose(mu[0], 0.0) and np.isclose(sd[0], 1.0)


# --------------------------------------------------------------------------- review fixes (2026-11-08)
def test_censored_lab_target_bound_is_detection_limit():
    """C1: a censored lab value becomes its detection limit (bound), never the sub-DL published value;
    censored rows without a DL are dropped (unobserved)."""
    from sentinel.models.fuse.data import Target, tier_d_target_matrix
    df = pd.DataFrame({"ylab__tp_mgl": [0.05, 0.001, 0.002, np.nan],
                       "ylab__tp_mgl__cens": [False, True, True, False],
                       "ylab__tp_mgl__dl": [0.004, 0.004, np.nan, np.nan]})
    Y, M, C = tier_d_target_matrix(df, [Target("ylab__tp_mgl", "gauss", "lab_tp_mgl", 0, cens_col="ylab__tp_mgl__cens")])
    assert np.isclose(Y[0, 0], np.log(0.05 + ftd.LAB_LOG_OFFSET["tp_mgl"]))
    assert np.isclose(Y[1, 0], np.log(0.004 + ftd.LAB_LOG_OFFSET["tp_mgl"]))     # bound = DL, not 0.001
    assert C[1, 0] and not C[0, 0]
    assert not M[2, 0] and not M[3, 0]                                               # censored w/o DL dropped


def test_anchor_removed_when_owner_modality_dropped():
    groups = {"sonde": [3, 2], "chem": [2, 2], "wx": [4]}
    B = 5
    pres = {k: torch.ones(B, dtype=torch.bool) for k in groups}
    model = _model(groups)
    b = _batch(B, groups, pres)
    T = len(TARGETS)
    anc = {"z": torch.full((B, T), 2.5), "ok": torch.ones(B, T, dtype=torch.bool),
           "owner": ["sonde", "sonde", None, "chem"]}
    with torch.no_grad():
        base = model(b)
        a = model(b, None, anc)
        keep = {"sonde": torch.zeros(B, dtype=torch.bool), "chem": torch.ones(B, dtype=torch.bool)}
        dropped = model(b, keep, anc)
        dropped_noanc = model(b, keep)
    assert torch.allclose(a[:, 0, 0], base[:, 0, 0] + 2.5)
    assert torch.equal(a[:, 2], base[:, 2])                       # binary target untouched
    assert torch.equal(dropped[:, 0], dropped_noanc[:, 0])        # owner dropped -> no anchor
    assert torch.allclose(dropped[:, 3, 0], dropped_noanc[:, 3, 0] + 2.5)


def test_cluster_ci_small_cluster_factor():
    from sentinel.models.fuse import stats as fst
    rs = np.random.default_rng(0)
    a = rs.gamma(2, 1, 400)
    b = a * 1.1 + rs.normal(0, 0.05, 400).clip(-0.5, 0.5)
    g = np.repeat(np.arange(8), 50)
    r = fst.cluster_ci(fst.skill, a, b, groups=g, n_boot=200)
    assert np.isclose(r["factor"], np.sqrt(8 / 7))
    assert r["ci_low"] <= r["ci_low_raw"] <= r["point"] <= r["ci_high_raw"] <= r["ci_high"]
    j = fst.leave_one_cluster_out(fst.skill, a, b, clusters=g // 2)
    assert len(j["by_cluster"]) == 4 and j["min"] <= r["point"] <= j["max"] + 1e-9


def test_hard_strata_use_training_thresholds_and_tails():
    from sentinel.models.fuse import stats as fst
    ytr = np.arange(100, dtype=float)
    y = np.array([5.0, 50.0, 95.0, np.nan])
    s = fst.hard_strata(y, y - 1, ytr, ytr - 1, "turb_fnu", np.array([1, 0, 1, 1], bool), np.array([3, 1, 5, 5]),
                        np.array([0, 0, 100, 0.0]), np.arange(100, dtype=float))
    assert s["ext_p90"].tolist() == [False, False, True, False]
    assert s["own_absent"].tolist() == [False, True, False, False]
    assert s["sparse_le2"].tolist() == [False, True, False, False]
    assert s["storm_p95"].tolist() == [False, False, True, False]
    lo = fst.hard_strata(y, y, ytr, ytr, "do_mgl", np.ones(4, bool), np.ones(4), np.zeros(4), np.zeros(100))
    assert lo["ext_p10"].tolist() == [True, False, False, False]


def test_single_hor_daily_and_trailing_anomaly_are_causal():
    cal = pd.date_range("2020-01-01", periods=400, freq="D")
    rows = []
    for hor, lvl, n in ((101, 100.0, 300), (102, 200.0, 400)):
        for i, d in enumerate(cal[:n]):
            rows.append({"date": d, "hor": hor, "ver": 100, "wse_m_mean": lvl + np.sin(i / 10),
                         "wse_m_min": lvl, "wse_m_max": lvl + 1, "provisional": False})
    z = ftd.single_hor_daily(pd.DataFrame(rows), "wse_m", cal)
    assert z["hor_used"].iloc[0] == 102 and z["wse_m__mean"].min() > 150        # one HOR only, never mixed
    zt = z.drop(columns="hor_used")
    a1 = ftd.trailing_anomaly(zt, "wse_m", cal, 365)
    zt2 = zt.copy()
    zt2.loc[zt2.index > cal[200], "wse_m__mean"] += 50.0                          # change the future only
    a2 = ftd.trailing_anomaly(zt2, "wse_m", cal, 365)
    pd.testing.assert_series_equal(a1["wse_m__mean"].loc[:cal[200]], a2["wse_m__mean"].loc[:cal[200]])


def test_bio_block_latest_event_on_or_before_and_age_cap():
    cal = pd.date_range("2020-01-01", periods=300, freq="D")
    ev = pd.DataFrame({"site": "X", "date": pd.to_datetime(["2020-01-10", "2020-03-01"]), "metric": [1.0, 2.0]})
    b = ftd.bio_block("macroinv", ev, cal)
    assert not b.loc["2020-01-09", "macroinv__present"]
    assert b.loc["2020-01-10", "macroinv__metric"] == 1.0 and b.loc["2020-01-10", "macroinv__age"] == 0
    assert b.loc["2020-03-01", "macroinv__metric"] == 2.0
    last_ok = pd.Timestamp("2020-03-01") + pd.Timedelta(days=ftd.BIO_MAX_AGE)
    assert b.loc[last_ok, "macroinv__present"] and not b.loc[last_ok + pd.Timedelta(days=1), "macroinv__present"]


def test_p3_statistic_sign_and_joint_sites():
    from sentinel.models.fuse import stats as fst
    rs = np.random.default_rng(1)
    s = np.repeat(np.array(["a", "b", "c", "d", "e"]), 40)
    w = rs.gamma(2, 1, 200)
    parts = [(s, w, w * 1.2, w * 2), (s[:120], w[:120], w[:120] * 1.1, w[:120] * 2)]
    r = fst.p3_statistic(parts, n_boot=200)
    assert np.isclose(r["point"], np.mean([0.2 / 2, 0.1 / 2]))
    assert 0 < r["ci_low"] <= r["point"] <= r["ci_high"] and r["n_groups"] == 5
