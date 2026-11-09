# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Unit tests for D2 ToxinWatch (synthetic arrays only)."""

import numpy as np
import pandas as pd
import pytest

from sentinel.models.downstream import common as C
from sentinel.models.downstream import toxinwatch as T


def test_waterbody_key_rules():
    assert T.waterbody_key("Green Lake (West Beach)") == "green"
    assert T.waterbody_key("Log Boom Park (Lake Washington)") == "washington"
    assert T.waterbody_key("Moses Lake, BH Park") == "moses"
    assert T.waterbody_key("Juanita Beach") == "washington"  # curated alias
    assert T.waterbody_key("Snake River - Chief Looking Glass Park") == "snake river"
    assert T.waterbody_key("Lake  Sammamish") == "sammamish"


def test_lake_clusters_name_county_and_distance():
    sites = pd.DataFrame({
        "site_id": ["1", "2", "3", "4", "5"],
        "site_name": ["Clear Lake", "Clear Lake", "Clear Lake", "Pond A", "Pond B"],
        "county": ["King", "King", "Pierce", "Kitsap", "Kitsap"],
        "latitude": [47.0, 47.3, 46.0, 48.0, 48.005],
        "longitude": [-122.0, -122.3, -122.0, -123.0, -123.0],
    })
    cl = T.lake_clusters(sites)
    assert cl["1"] == cl["2"]          # same name, same county
    assert cl["1"] != cl["3"]          # same name, other county, far away
    assert cl["4"] == cl["5"]          # different names but < 2 km apart


def _hist(dates, ys):
    return pd.DataFrame({"lake_cluster": "L", "date": pd.to_datetime(dates), "y": ys,
                         "max_value": [20.0 if y else 0.5 for y in ys], "n": 1}).sort_values("date")


def test_prior_features_respect_embargo():
    hist = _hist(["2020-07-01", "2020-07-20"], [0.0, 1.0])
    rows = pd.DataFrame({"lake_cluster": "L", "date": pd.to_datetime(["2020-07-23", "2020-07-27", "2020-07-28"])},
                        index=[10, 11, 12])
    f = T.prior_features(rows, hist, embargo_days=7, p0=0.1)
    # 07-23: the 07-20 exceedance is 3 days old -> invisible; only the 07-01 non-exceedance is known
    assert f.at[10, "tw_prior_n"] == 1 and f.at[10, "tw_last_exceed"] == 0.0
    # 07-27: exactly 7 days after 07-20 -> visible
    assert f.at[11, "tw_prior_n"] == 2 and f.at[11, "tw_last_exceed"] == 1.0
    assert f.at[12, "tw_prior_exceed_same_year"] == 1.0
    f3 = T.prior_features(rows, hist, embargo_days=3, p0=0.1)
    assert f3.at[10, "tw_last_exceed"] == 1.0


def test_prior_features_never_use_same_or_future_day():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2015-06-01", periods=60, freq="D")
    hist = _hist(dates, rng.integers(0, 2, 60).astype(float))
    rows = pd.DataFrame({"lake_cluster": "L", "date": dates})
    f = T.prior_features(rows, hist, embargo_days=7, p0=0.1)
    for i, d in enumerate(dates):
        assert f["tw_prior_n"].iloc[i] == (hist["date"] <= d - pd.Timedelta(days=7)).sum()


def test_assign_rows_and_drop_protected():
    sd = pd.DataFrame({"lake_cluster": ["a", "a", "a", "b", "c", "a"],
                       "date": pd.to_datetime(["2010-07-01", "2020-07-01", "2023-07-01", "2010-07-01", "2010-07-01",
                                               "2018-12-31"])})
    out = T.assign_rows(sd, {"a": "train", "b": "val", "c": "test"})
    assert out["subset"].tolist() == ["train", "val_known", "test", "val_new_pre2019", "test", "train"]
    later = T.assign_rows(sd.assign(date=pd.Timestamp("2020-07-01")), {"a": "train", "b": "val", "c": "test"})
    assert later["subset"].tolist()[3] == "val_new"
    kept = C.drop_protected(out)
    assert (kept["role"] != "test").all() and len(kept) == 4
    unknown = T.assign_rows(sd.assign(lake_cluster="zzz"), {"a": "train"})
    assert (unknown["role"] == "test").all()  # clusters absent from the frozen split are protected


def test_st_glmm_design_shape_and_no_nan():
    rows = pd.DataFrame({"date": pd.to_datetime(["2019-07-01", "2019-08-01"])})
    X = pd.DataFrame({"tw_last_exceed": [np.nan, 1.0], "tw_days_since_last": [np.nan, 10.0],
                      "tw_prior_rate_s": [0.1, 0.4]})
    D = T.st_glmm_design(rows, X, np.array([0.1, 0.2]))
    assert D.shape == (2, 10) and np.isfinite(D).all()


def test_split_frozen_file_matches_protocol_sha():
    import hashlib
    p = C.PREREG / T.SPLIT_FILE
    if not p.exists():
        pytest.skip("split not frozen in this checkout")
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    assert sha == T.SPLIT_SHA
    assert sha in (C.PREREG / "downstream_toxinwatch.md").read_text()
    C.load_split("toxinwatch", expect_sha=T.SPLIT_SHA, path=p)


def test_v2_lake_roles_giant_cluster_to_train_and_balanced():
    sd = pd.DataFrame({"lake_cluster": ["big"] * 500 + [f"l{i}" for i in range(40) for _ in range(5)]
                       + [f"r{i}" for i in range(5) for _ in range(210)]})
    role = T.lake_role_assignment(sd)
    assert role["big"] == "train"
    r = pd.Series(role)
    small = sd[sd.lake_cluster.str.startswith("l")].groupby("lake_cluster").size()
    share = small.groupby(r.reindex(small.index)).sum() / small.sum()
    assert abs(share["test"] - 0.2) < 0.06 and abs(share["val"] - 0.2) < 0.06


def test_iteration1_site_and_intensity_features_respect_embargo():
    sitehist = pd.DataFrame({"site_id": "s1", "date": pd.to_datetime(["2020-07-01", "2020-07-20"]), "y": [1.0, 0.0]})
    hist = _hist(["2020-07-01", "2020-07-20"], [1.0, 0.0])
    rows = pd.DataFrame({"site_id": "s1", "lake_cluster": "L", "date": pd.to_datetime(["2020-07-23", "2020-07-27"])},
                        index=[0, 1])
    f = T.site_and_intensity_features(rows, sitehist, hist, embargo_days=7, p0=0.1)
    assert f.at[0, "tw_site_prior_n"] == 1 and f.at[0, "tw_site_last_exceed"] == 1.0
    assert f.at[0, "tw_lake_n_30d"] == 1 and f.at[1, "tw_lake_n_30d"] == 2
    assert f.at[1, "tw_site_last_exceed"] == 0.0


def test_stglmm_v2_fits_and_predicts_synthetic():
    rng = np.random.default_rng(1)
    n = 300
    rows = pd.DataFrame({"date": pd.Timestamp("2015-06-01") + pd.to_timedelta(rng.integers(0, 120, n), "D"),
                         "lake_cluster": rng.choice(["a", "b", "c"], n)})
    X = pd.DataFrame({"tw_last_exceed": rng.choice([0.0, 1.0, np.nan], n), "tw_days_since_last": rng.uniform(7, 60, n),
                      "tw_prior_rate_s": rng.uniform(0.01, 0.5, n), "tw_last_log_value": rng.normal(0, 1, n),
                      "tw_prior_max_log_365": rng.normal(0, 1, n), "lat": rng.uniform(46, 48, n),
                      "lon": rng.uniform(-123, -120, n), "wx_tmax_mean_7d": rng.normal(25, 3, n)})
    y = rng.integers(0, 2, n)
    m = T.StGlmm(C_reg=1.0, wx_cols=["wx_tmax_mean_7d"], variant="v2").fit(rows, X, np.full(n, 0.1), y)
    p = m.predict_proba(rows.assign(lake_cluster="zzz"), X, np.full(n, 0.1))  # unseen lake -> no one-hot
    assert p.shape == (n,) and np.all((p > 0) & (p < 1))
