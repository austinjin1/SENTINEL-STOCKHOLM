# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Tests for the shared downstream plumbing (synthetic arrays only)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sentinel.models.downstream import common as C
from sentinel.models.downstream import crossfit as X
from sentinel.models.downstream import weather as W
from sentinel.models.downstream import biointegrity as BI


def test_freeze_split_is_immutable(tmp_path):
    p = tmp_path / "s.json"
    sha = C.freeze_split("x", {"unit_role": {"a": "train", "b": "val", "c": "test"}}, {"m": 1}, path=p)
    assert sha == C.freeze_split("x", {"unit_role": {"a": "train", "b": "val", "c": "test"}}, {"m": 1}, path=p)
    with pytest.raises(C.ProtectedDataError):
        C.freeze_split("x", {"unit_role": {"a": "val", "b": "val", "c": "test"}}, {"m": 1}, path=p)
    body, sha2 = C.load_split("x", path=p, expect_sha=sha)
    assert sha2 == sha and body["assignments"]["unit_role"]["c"] == "test"
    with pytest.raises(C.ProtectedDataError):
        C.load_split("x", path=p, expect_sha="0" * 64)


def test_role_maps_validated(tmp_path):
    with pytest.raises(ValueError):
        C.freeze_split("x", {"unit_role": {"a": "holdout"}}, {}, path=tmp_path / "s.json")
    C.freeze_split("x", {"unit_cluster": {"a": "L1"}}, {}, path=tmp_path / "t.json")  # non-role maps free


def test_guard_refuses_test():
    df = pd.DataFrame({"role": ["train", "val", "test", None]})
    out = C.drop_protected(df)
    assert set(out["role"]) == {"train", "val"} and out.attrs["n_dropped_protected"] == 2
    C.assert_no_test(out)
    with pytest.raises(C.ProtectedDataError):
        C.assert_no_test(df)
    with pytest.raises(C.ProtectedDataError):
        C.check_roles("x", ["val", "test"])
    with pytest.raises(C.ProtectedDataError):
        C.check_roles("x", ["test"], allow_test=True, training=True)
    C.check_roles("x", ["train", "val"])


def test_registry_refuses_test_split():
    with pytest.raises(C.ProtectedDataError):
        C.log_metric(app="x", experiment="e", model="m", metric="auroc", value=0.7, split="test_2024")


def test_binary_scores_and_paired():
    rs = np.random.default_rng(0)
    y = rs.integers(0, 2, 400).astype(float)
    good = y + rs.normal(0, 0.7, 400)
    bad = rs.normal(0, 1, 400)
    g = np.repeat(np.arange(40), 10)
    sc = C.binary_scores(y, good, g, n_boot=200)
    assert sc["auroc"] > 0.75 and sc["auroc_ci"][0] < sc["auroc"] < sc["auroc_ci"][1]
    pr = C.paired(lambda a, b: __import__("sentinel.eval.metrics", fromlist=["auroc"]).auroc(a, b),
                  y, good, bad, g, n_boot=200)
    assert pr["diff"] > 0 and pr["ci_low"] > 0


def test_crossfit_hook():
    class P:
        name, model_registry_id = "m6", "abc"
        training_sites = {"USGS-1"}
        training_end = pd.Timestamp("2021-09-30")

    ev = pd.DataFrame({"upstream_site_key": ["USGS-2"], "decision_time": [pd.Timestamp("2022-06-01")]})
    assert X.check_crossfit(P(), ev)["n_units"] == 1
    with pytest.raises(X.CrossFitError):
        X.check_crossfit(P(), ev.assign(upstream_site_key="USGS-1"))
    with pytest.raises(X.CrossFitError):
        X.check_crossfit(P(), ev.assign(decision_time=pd.Timestamp("2021-01-01")))
    with pytest.raises(X.CrossFitError):
        X.phase1_no_upstream(["wx_precip", "m6_do_pred"])


def test_antecedent_windows_end_day_before():
    dates = pd.date_range("2020-01-01", periods=40, freq="D")
    s = pd.DataFrame({"site_key": "S", "date": dates, "precip_mm": np.arange(40.0), "tmax_c": 0.0, "tmin_c": 0.0,
                      "srad_wm2": 0.0, "vpd_kpa": 0.0, "wind_ms": 0.0})
    q = pd.DataFrame({"wx_site_key": ["S"], "date": [dates[20]]})
    f = W.antecedent_features(q, s, windows=(1, 3))
    assert f["wx_precip_sum_1d"].iloc[0] == 19.0            # day d-1 only, never day d (=20)
    assert f["wx_precip_sum_3d"].iloc[0] == 17 + 18 + 19
    f0 = W.antecedent_features(q, s, windows=(1,), lag0=True)
    assert f0["wx_precip_sum_1d"].iloc[0] == 20.0


def test_biointegrity_baselines_exclude_same_location():
    tr = pd.DataFrame({"lat": [40.0, 40.0, 41.0], "lon": [-90.0, -90.0, -91.0], "location_id": ["A", "A", "B"],
                       "y_poor": [1.0, 1.0, 0.0], "ag_eco9": ["UMW", "UMW", "UMW"], "y_ord": [2.0, 2.0, 0.0],
                       "date": pd.to_datetime(["2008-06-01", "2013-06-01", "2009-06-01"])})
    ev = pd.DataFrame({"lat": [40.0], "lon": [-90.0], "location_id": ["A"], "ag_eco9": ["UMW"]})
    assert BI.knn_rate(tr, ev, k=1)[0] == 0.0               # same-location training reaches excluded
    assert BI.persistence(tr, ev)[0] == 2.0
    assert abs(BI.ecoregion_rate(tr, ev)[0] - 2 / 3) < 1e-9


def test_resolve_comids_prefers_published():
    from sentinel.data import downstream_links as DL

    pts = pd.DataFrame({"unit_source": ["nrsa", "nrsa"], "unit_id": ["a", "b"], "lat": [40.0, 41.0],
                        "lon": [-90.0, -91.0], "comid_published": pd.Series(["123", pd.NA], dtype="string"),
                        "comid_published_source": ["site_file", None]})
    snaps = pd.DataFrame({"unit_source": ["nrsa"], "unit_id": ["b"], "comid_snap": ["456"], "snap_status": ["ok"]})
    out = DL.resolve_comids(pts, snaps)
    assert out["comid"].tolist() == ["123", "456"]
    assert out["comid_method"].tolist() == ["published:site_file", "nldi_position"]
    assert DL.parse_position_snap({"status": 404})["comid_snap"] is None
    rec = {"status": 200, "body": {"features": [{"properties": {"source": "comid", "identifier": "789"}}]}}
    assert DL.parse_position_snap(rec)["comid_snap"] == "789"


def test_biointegrity_excludes_model_derived_streamcat():
    for c in ("iwi", "ici", "bankfulldepth", "bankfullwidth", "wettedwidth", "thalwegdepth"):
        assert c in BI.EXCLUDED_STREAMCAT
    assert "prg_bmmi0809" in __import__("sentinel.data.sources.streamcat", fromlist=["EXCLUDED"]).EXCLUDED


def test_validate_published_comids():
    from sentinel.data import downstream_links as DL

    pts = pd.DataFrame({"unit_source": "nrsa", "unit_id": ["a", "b", "c"], "lat": 0.0, "lon": 0.0,
                        "comid_published": pd.Series(["123", "1.03E+11", "999"], dtype="string"),
                        "comid_published_source": "site_info"})
    out = DL.validate_published(pts, {"123"})
    assert out["comid_published"].isna().tolist() == [False, True, True]
    assert out["comid_published_invalid"].tolist()[1:] == ["1.03E+11", "999"]


def test_cpu_slot_fifo(tmp_path, monkeypatch):
    from sentinel.core import paths as Pth

    monkeypatch.setattr(Pth, "RUNS_ROOT", tmp_path)
    with C.cpu_slot("a", poll_s=0.01) as k:
        assert k == 0
        assert not list((tmp_path / "downstream" / "locks" / "queue").iterdir())   # ticket released
    # a stale ticket of a dead pid must not block
    q = tmp_path / "downstream" / "locks" / "queue"
    (q / f"{1:020d}_999999999").write_text("dead\n")
    with C.cpu_slot("b", poll_s=0.01) as k:
        assert k == 0


def test_fast_metrics_match_sklearn():
    from sklearn.metrics import average_precision_score, roc_auc_score

    rs = np.random.default_rng(3)
    for _ in range(20):
        y = rs.integers(0, 2, 200).astype(float)
        s = np.round(rs.normal(size=200), 1)          # many ties
        assert np.isclose(C.fast_auroc(y, s), roc_auc_score(y, s))
        assert np.isclose(C.fast_auprc(y, s), average_precision_score(y, s))
    assert np.isnan(C.fast_auroc(np.zeros(4), np.arange(4.0)))
