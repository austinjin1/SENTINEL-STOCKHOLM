# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Unit tests for D1 SafeSwim (synthetic arrays only; no data files except the frozen split JSON)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sentinel.models.downstream import common as C
from sentinel.models.downstream import safeswim as S


def _bd(rows):
    df = pd.DataFrame(rows, columns=["beach_key", "date", "log_gmean"])
    df["date"] = pd.to_datetime(df["date"])
    df["y"] = (10 ** df["log_gmean"] > 235).astype(int)
    return df


def test_beach_days_gmean_and_strict_threshold():
    t = pd.DataFrame({"beach_key": ["a", "a", "b"], "date": ["2010-06-01", "2010-06-01", "2010-06-01"],
                      "result": [100.0, 400.0, 235.0]})
    bd = S.beach_days(t, 235.0)
    a = bd[bd.beach_key == "a"].iloc[0]
    assert a["n_tests"] == 2 and np.isclose(a["gmean"], 200.0)
    assert a["y"] == 0                          # gmean 200 <= 235
    assert bd[bd.beach_key == "b"].iloc[0]["y"] == 0   # strict '>' threshold


def test_beach_days_excludes_inland_humboldt():
    t = pd.DataFrame({"beach_key": ["humboldt", "a"], "date": ["2010-06-01"] * 2, "result": [5000.0, 10.0]})
    assert set(S.beach_days(t, 235.0)["beach_key"]) == {"a"}


def test_persistence_uses_only_earlier_dates():
    bd = _bd([("a", "2010-06-01", 1.0), ("a", "2010-06-02", 3.0), ("a", "2010-06-06", 2.0), ("b", "2010-06-02", 2.5)])
    p = S.persistence_features(bd)
    assert np.isnan(p.loc[0, "prev_log"]) and np.isnan(p.loc[3, "prev_log"])
    assert p.loc[1, "prev_log"] == 1.0 and p.loc[1, "prev_gap_d"] == 1
    assert p.loc[2, "prev_log"] == 3.0 and p.loc[2, "persist_eligible"] == 0   # gap 4 d > 3 d rule


def test_culture_never_same_day():
    bd = _bd([("a", "2010-06-01", 1.0)])
    with pytest.raises(C.ProtectedDataError):
        S.network_features(bd, bd, ["a"], kind="culture", same_day=True, prefix="x_")


def test_previous_network_value_is_strictly_earlier():
    src = _bd([("in1", "2022-06-01", 1.0), ("in1", "2022-06-03", 4.0), ("in2", "2022-06-03", 2.0)])
    q = _bd([("a", "2022-06-03", 0.0), ("a", "2022-06-01", 0.0)])
    prev = S.network_features(q, src, ["in1", "in2"], kind="qpcr", same_day=False, prefix="p_")
    assert prev.loc[0, "p_mean_log"] == 1.0 and prev.loc[0, "p_gap_d"] == 2     # not the same-day 4.0/2.0
    assert np.isnan(prev.loc[1, "p_mean_log"])                                   # nothing before the first day
    sd = S.network_features(q, src, ["in1", "in2"], kind="qpcr", same_day=True, prefix="s_")
    assert sd.loc[0, "s_mean_log"] == 3.0 and sd.loc[0, "s_n"] == 2


def test_exclude_own_beach_from_network():
    src = _bd([("a", "2022-06-01", 4.0), ("b", "2022-06-01", 1.0)])
    q = _bd([("a", "2022-06-02", 0.0)])
    f = S.network_features(q, src, ["a", "b"], kind="qpcr", same_day=False, prefix="n_", exclude_own=True)
    assert f.loc[0, "n_mean_log"] == 1.0


def test_assign_roles_and_drop_protected():
    split = {"assignments": {"d1b_year_role": {"2021": "train", "2022": "val", "2024": "test"},
                             "beach_lat_band": {"a": 0, "inp": 0}},
             "meta": {"d1b_eval_beaches": ["a"]}}
    bd = _bd([("a", "2020-06-01", 1.0), ("a", "2021-06-01", 1.0), ("a", "2022-06-01", 1.0),
              ("a", "2024-06-01", 1.0), ("inp", "2022-06-01", 1.0), ("nocoord", "2022-06-01", 1.0)])
    r = S.assign_roles(bd, "d1b", split)
    assert set(r["beach_key"]) == {"a"}
    kept = C.drop_protected(r)
    assert sorted(kept["role"]) == ["train", "val"]
    assert kept.attrs["n_dropped_protected"] == 2          # 2020 (no role) and 2024 (test)
    C.assert_no_test(kept)
    with pytest.raises(C.ProtectedDataError):
        C.assert_no_test(r)


def test_sens_at_spec():
    y = np.array([0] * 20 + [1] * 4)
    s = np.r_[np.arange(20) / 20.0, [0.99, 0.98, 0.5, 0.1]]
    v = S.sens_at_spec(y, s, 0.85)
    assert 0.0 <= v <= 1.0 and np.isclose(v, 0.5)


def test_frozen_split_matches_module_constants():
    split, sha = C.load_split(S.APP)
    assert split["assignments"]["d1a_year_role"] == {str(k): v for k, v in S.D1A_YEARS.items()}
    assert split["assignments"]["d1b_year_role"] == {str(k): v for k, v in S.D1B_YEARS.items()}
    assert "2020" not in split["assignments"]["d1b_year_role"]
    assert not set(split["meta"]["d1b_input_beaches"]) & set(split["meta"]["d1b_eval_beaches"])
    assert len(sha) == 64


def test_feature_columns_exclude_labels_and_incumbent():
    X = pd.DataFrame(columns=["y", "log_gmean", "gmean", "official_pred", "doy", "dow", "latitude", "prev_log",
                              "prev_gap_d", "prev_exceed", "inp_sd_mean_log", "wx_precip_sum_1d", "wx_cell_km"])
    fc = S.feature_columns(X, "d1b", "forecast")
    assert not {"y", "log_gmean", "gmean", "official_pred", "inp_sd_mean_log", "wx_cell_km"} & set(fc)
    assert "inp_sd_mean_log" in S.feature_columns(X, "d1b", "nowcast")


def test_fast_auroc_matches_sklearn_with_ties():
    from sklearn.metrics import roc_auc_score

    rs = np.random.default_rng(1)
    y = rs.integers(0, 2, 300)
    s = np.round(rs.normal(size=300), 1)
    assert np.isclose(S.fast_auroc(y, s), roc_auc_score(y, s))
    assert np.isnan(S.fast_auroc(np.zeros(5), np.arange(5.0)))


def test_hgb_has_no_random_holdout():
    assert S.MODEL_CONFIGS["gbm_hgb"]["early_stopping"] is False
