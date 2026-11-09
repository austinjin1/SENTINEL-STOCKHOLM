# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Unit tests for D3 KillAlert (synthetic arrays only; no data files, no network)."""

import numpy as np
import pandas as pd
import pytest

from sentinel.models.downstream import common as C
from sentinel.models.downstream import killalert as K
from sentinel.models.downstream import weather as W


def _events(rows):
    df = pd.DataFrame(rows, columns=["event_id", "date", "latitude", "longitude"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_role_of_year_blocks():
    assert K.role_of_year(2000) == "train" and K.role_of_year(2009) == "train"
    assert K.role_of_year(2010) == "val" and K.role_of_year(2017) == "val"
    assert K.role_of_year(2018) == "test" and K.role_of_year(1999) is None


def test_referent_dates_other_years_only_and_leap_day():
    r = K.referent_dates("2004-02-29", range(2000, 2010), half_window=3)
    assert len(r) == 9 * 7
    assert all(d.year != 2004 for d in r)
    assert pd.Timestamp("2001-02-28") in r  # Feb 29 mapped to Feb 28
    assert max(abs((d - K._replace_year(pd.Timestamp("2004-02-29"), d.year)).days) for d in r) <= 3


def test_near_any_event_space_and_time():
    ev = _events([("1", "2005-07-10", 42.0, -93.0)])
    cands = pd.to_datetime(["2005-07-20", "2005-08-20", "2005-07-11"])
    near = K.near_any_event(cands, 42.0, -93.0, ev, km=5, days=30)
    assert near.tolist() == [True, False, True]
    far = K.near_any_event(cands, 42.2, -93.0, ev, km=5, days=30)  # ~22 km away
    assert not far.any()


def test_case_crossover_referents_never_near_events_and_stay_in_block():
    ev = _events([("1", "2005-07-10", 42.0, -93.0), ("2", "2007-07-12", 42.001, -93.001),
                  ("3", "2012-03-01", 41.0, -92.0)])
    for c in ("is_env_wq", "is_env_or_natsusp", "est_location"):
        ev[c] = False
    ev["cause_origin"], ev["waterbody_type"] = "Natural", "Lake"
    ev["role"] = ev["date"].dt.year.map(K.role_of_year)
    cc = K.build_case_crossover(ev, ev, last_date=pd.Timestamp("2026-01-01"))
    ref = cc[cc["is_case"] == 0]
    for r in ref.itertuples():
        assert K.role_of_year(r.date.year) == r.role                        # same block years only
        assert not K.near_any_event([r.date], r.lat, r.lon, ev, km=K.EXCL_KM, days=K.EXCL_DAYS)[0]
    s1 = ref[ref["stratum"] == "1"]
    assert not ((s1["date"].dt.year == 2007)).any()                          # 2007 window blocked by event 2
    assert (cc.groupby("stratum")["is_case"].sum() == 1).all()


def test_weather_windows_end_day_before():
    dates = pd.date_range("2010-01-01", "2010-03-01")
    s = pd.DataFrame({"site_key": "X", "date": dates, "precip_mm": 0.0, "tmax_c": 0.0, "tmin_c": 0.0,
                      "srad_wm2": 0.0, "vpd_kpa": 0.0, "wind_ms": 0.0})
    s.loc[s["date"] == "2010-02-01", "precip_mm"] = 50.0
    q = pd.DataFrame({"wx_site_key": ["X", "X"], "date": pd.to_datetime(["2010-02-01", "2010-02-02"])})
    f = W.antecedent_features(q, s)
    assert f["wx_precip_sum_1d"].tolist() == [0.0, 50.0]                     # same-day rain is not an input


def test_gauge_anoms_end_day_before_and_clim_train_only():
    idx = pd.date_range("2000-01-01", "2011-12-31")
    x = pd.Series(10.0, index=idx)
    x[idx.year == 2011] = 1000.0                                             # non-train year shift
    a = K.gauge_anoms_from_series(x, range(2000, 2010), log=True)
    assert abs(a.loc["2005-06-10", "anom_7d"]) < 1e-9
    assert a.loc["2011-06-10", "anom_7d"] == pytest.approx(2.0)              # climatology excludes 2011
    y = x.copy()
    y["2005-06-10"] = 1e6
    b = K.gauge_anoms_from_series(y, range(2000, 2010), log=True)
    assert b.loc["2005-06-10", "change_3d"] == pytest.approx(a.loc["2005-06-10", "change_3d"])  # same-day not used


def test_matched_concordance():
    strata = np.repeat(["a", "b"], 3)
    y = np.array([1, 0, 0, 1, 0, 0])
    assert K.matched_concordance(y, np.array([3, 1, 2, 5, 4, 0]), strata) == 1.0
    assert K.matched_concordance(y, -np.array([3, 1, 2, 5, 4, 0]), strata) == 0.0
    assert K.matched_concordance(y, np.ones(6), strata) == 0.5


def test_conditional_logit_recovers_sign():
    rs = np.random.default_rng(0)
    n_s, m = 200, 10
    X = rs.normal(size=(n_s * m, 2))
    strata = np.repeat(np.arange(n_s), m)
    y = np.zeros(n_s * m, int)
    for k in range(n_s):
        g = np.arange(k * m, (k + 1) * m)
        p = np.exp(1.5 * X[g, 0]); p /= p.sum()
        y[g[rs.choice(m, p=p)]] = 1
    mdl = K.ConditionalLogit(l2=0.1).fit(X, y, strata)
    assert mdl.coef_[0] > 0.8 and abs(mdl.coef_[1]) < 0.4
    assert K.matched_concordance(y, mdl.decision_function(X), strata) > 0.65


def test_drop_protected_removes_test_rows():
    df = pd.DataFrame({"role": ["train", "val", "test", None], "x": range(4)})
    out = C.drop_protected(df)
    assert out["role"].tolist() == ["train", "val"]
    with pytest.raises(C.ProtectedDataError):
        C.assert_no_test(df)


def test_bootstrap_mean_paired_and_clusters():
    v = pd.Series([1.0, 0.8, 0.9, 0.7], index=list("abcd"))
    o = pd.Series([0.5, 0.5, 0.5, 0.5], index=list("abcd"))
    r = K.bootstrap_mean(v, other=o, n_boot=200)
    assert r["point"] == pytest.approx(0.35) and r["ci_low"] > 0 and r["p_value"] < 0.05
    r2 = K.bootstrap_mean(v, clusters=pd.Series([1, 1, 2, 2], index=list("abcd")), n_boot=200)
    assert r2["n_clusters"] == 2


def test_inner_cv_select_runs_on_synthetic_strata():
    rs = np.random.default_rng(1)
    n_s, m = 60, 8
    X = pd.DataFrame(rs.normal(size=(n_s * m, 2)), columns=["a", "b"])
    strata = np.repeat(np.arange(n_s), m)
    y = np.zeros(n_s * m, int)
    y[np.arange(n_s) * m + X["a"].to_numpy().reshape(n_s, m).argmax(1)] = 1
    best, log = K.inner_cv_select("clogit", [{"l2": 0.1}, {"l2": 10.0}], X, y, strata, n_folds=3)
    assert len(log) == 2 and best in ({"l2": 0.1}, {"l2": 10.0})
    assert max(r["inner_cv_concordance"] for r in log) > 0.8


def test_referents_clipped_to_block_years():
    ev = _events([("1", "2005-12-30", 42.0, -93.0), ("2", "2005-01-02", 41.0, -92.0)])
    for c in ("is_env_wq", "is_env_or_natsusp", "est_location"):
        ev[c] = False
    ev["cause_origin"], ev["waterbody_type"] = "Natural", "Lake"
    ev["role"] = "train"
    cc = K.build_case_crossover(ev, ev, last_date=pd.Timestamp("2026-01-01"))
    ref = cc[cc["is_case"] == 0]
    assert ref["date"].dt.year.between(2000, 2009).all()
    assert not ((ref["date"] >= "2010-01-01") | (ref["date"] < "2000-01-01")).any()


def test_design_gauge_missingness_is_location_level_only():
    cc = pd.DataFrame({"g_logq_anom_7d": [np.nan, 0.3, np.nan], "g_logq_change_3d": [np.nan, 0.1, np.nan],
                       "g_q_site": ["0545", "0545", None], "year_c": [0.0, 0.1, 0.2]})
    X = K.design(cc, K.GAUGE_FEATURES + ["year_c"])
    assert X.isna().sum().sum() == 0
    assert X["g_no_gauge"].tolist() == [0.0, 0.0, 1.0] and X["g_logq_anom_7d"].iloc[0] == 0.0


def test_delaware_cause_harmonisation_rules():
    from sentinel.data.sources import events_delaware_fishkill as DE

    assert DE.harmonise_cause("Low DO")[:2] == ("Natural", "Environmental")
    assert DE.harmonise_cause("Winter die-off after ice out")[1] == "Environmental"
    assert DE.harmonise_cause("Lethal levels of DO caused by runoff of chicken manure")[0] == "Anthropogenic"
    assert DE.harmonise_cause("Probable epizootic-gill disease")[1] == "Disease/Parasite/Stress"
    assert DE.harmonise_cause(None)[0] == "Unknown" and DE.harmonise_cause("unknown")[0] == "Unknown"
    assert DE.waterbody_type("Silver Lake (Dover)") == "Lake" and DE.waterbody_type("Pike Creek") == "River/Stream"


def test_delaware_events_dedupe_species_rows():
    from sentinel.data.sources import events_delaware_fishkill as DE

    rows = [{"location": "X Pond", "date": "2010-07-01T00:00:00.000", "lat": "39.1", "long": "-75.5",
             "species": s, "no_killed": "10", "probable_cause": "Low DO"} for s in ("Bluegill", "Bass")]
    ev = DE.to_events(DE.parse_rows(rows))
    assert len(ev) == 1 and ev["est_fish_killed"].iloc[0] == 20 and ev["is_env_wq"].iloc[0]
