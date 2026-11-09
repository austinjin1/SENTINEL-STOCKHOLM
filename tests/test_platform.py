# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Tests for the SENTINEL Console: API contract, metric guard, upload validation, live client, pages.

All data here are tiny synthetic tables written to a temporary $SENTINEL_DATA (synthetic arrays
are allowed only inside unit tests, CONTRIBUTING_V2.md); their manifests are flagged
``synthetic=True``. No network access: the USGS live client runs against a fake session and the
app pages are rendered in-process with Streamlit's AppTest.
"""

from __future__ import annotations

import importlib
import io
import json
import logging
import sys
import types
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from sentinel.core import paths as core_paths
from sentinel.core.manifest import write_manifest
from sentinel.core.registry import log_result

REPO = Path(__file__).resolve().parents[1]
SITE_A, SITE_B = "01234567", "07654321"  # synthetic test sites
SECRET = "test-key-DO-NOT-LEAK-7f3a"


# ----------------------------------------------------------------------------- fixtures


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Point SENTINEL_DATA / SENTINEL_RUNS at a temp dir (env set before the platform reads paths)."""
    data, runs, repo = tmp_path / "data", tmp_path / "runs", tmp_path / "repo"
    monkeypatch.setenv("SENTINEL_DATA", str(data))
    monkeypatch.setenv("SENTINEL_RUNS", str(runs))
    monkeypatch.setenv("SENTINEL_API_URL", "http://127.0.0.1:9")  # nothing listens there
    monkeypatch.delenv("API_USGS_PAT", raising=False)
    monkeypatch.delenv("SENTINEL_LIVE", raising=False)
    importlib.reload(core_paths)  # paths are module constants computed from the environment
    from sentinel.platform import config, plugins

    for d in (repo / "docs" / "model_cards", repo / "docs" / "preregistration", repo / "results"):
        d.mkdir(parents=True)
    (repo / "models.yaml").write_text("version: 1\nmodels: []\n")
    config.set_overrides(
        registry_path=repo / "results" / "registry.jsonl",
        repo_root=repo,
        model_cards_dir=repo / "docs" / "model_cards",
        models_yaml=repo / "models.yaml",
        results_md=repo / "RESULTS.md",
        research_plan=repo / "RESEARCH_PLAN.md",
        prereg_dir=repo / "docs" / "preregistration",
        keys_file=tmp_path / "keys.env",
    )
    ns = types.SimpleNamespace(
        root=tmp_path, data=data, runs=runs, repo=repo, registry=repo / "results" / "registry.jsonl"
    )
    yield ns
    config.clear_overrides()
    plugins._ep_cache.clear()
    monkeypatch.undo()
    importlib.reload(core_paths)


def _client(live_client=None):
    from sentinel.platform.api.app import create_app
    from sentinel.platform.config import Settings

    return TestClient(create_app(Settings.from_env(), live_client=live_client))


def _write_daily(data: Path) -> None:
    d = data / "processed" / "sonde_daily" / "sites"
    d.mkdir(parents=True)
    dates = pd.date_range("2024-01-01", periods=10, freq="D", name="date")
    do = np.array([8.0, 7.5, 6.0, 2.5, 6.5, np.nan, 7.0, 7.2, 7.1, 6.9], dtype="float32")
    temp = np.linspace(10, 12, 10).astype("float32")
    prov = np.array([False] * 7 + [True] * 3)
    df = pd.DataFrame(
        {
            "do_mgl_min": do,
            "do_mgl_min_prov": prov,
            "do_mgl_min_ts": "111",
            "temp_c_mean": temp,
            "temp_c_mean_prov": prov,
            "temp_c_mean_ts": "222",
        },
        index=dates,
    )
    df.to_parquet(d / f"{SITE_A}.parquet")
    idx = pd.DataFrame(
        {
            "site_no": [SITE_A, SITE_A],
            "var": ["do_mgl_min", "temp_c_mean"],
            "n_days": [9, 10],
            "first": [dates[0], dates[0]],
            "last": [dates[-1], dates[-1]],
            "frac_provisional": [3 / 9, 0.3],
        }
    )
    idx.to_parquet(data / "processed" / "sonde_daily" / "index.parquet")
    write_manifest(
        "sonde_daily",
        files=[d / f"{SITE_A}.parquet"],
        sources=["manifest:usgs_dv_raw"],
        description="unit-test fixture (synthetic)",
        synthetic=True,
        stats={"retrieved": "test"},
    )


def _write_hourly(data: Path) -> None:
    d = data / "processed" / "sonde_hourly" / "sites"
    d.mkdir(parents=True)
    t = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC", name="time_utc")
    v = np.linspace(6, 9, 48).astype("float32")
    df = pd.DataFrame(
        {
            "do_mgl_mean": v,
            "do_mgl_min": v - 0.5,
            "do_mgl_max": v + 0.5,
            "do_mgl_n": np.full(48, 4, dtype="int16"),
            "do_mgl_prov": np.arange(48) >= 40,
        },
        index=t,
    )
    df.to_parquet(d / f"{SITE_A}.parquet")


def _write_catalog(data: Path) -> None:
    d = data / "raw" / "usgs_dv"
    d.mkdir(parents=True)
    pd.DataFrame(
        {
            "agency_cd": ["USGS"] * 3,
            "site_no": [SITE_A, SITE_A, SITE_B],
            "station_nm": ["TEST CREEK A", "TEST CREEK A", "TEST LAKE B"],
            "site_tp_cd": ["ST", "ST", "LK"],
            "dec_lat_va": ["40.5", "40.5", "35.25"],
            "dec_long_va": ["-90.25", "-90.25", "-100.75"],
            "huc_cd": ["07090001", "07090001", "11100101"],
            "state": ["IL", "IL", "TX"],
            "parm_cd": ["00300", "00010", "00010"],
        }
    ).to_parquet(d / "dv_series_catalog.parquet")


def _full_data(env) -> None:
    _write_daily(env.data)
    _write_hourly(env.data)
    _write_catalog(env.data)


def _log(env, **kw):
    base = dict(
        workstream="sonde",
        experiment="test_exp",
        model="sentinel_sonde_test",
        metric="crps",
        value=0.5,
        split="val",
        n=10,
        ci_low=0.4,
        ci_high=0.6,
        ci_method="site block bootstrap",
        notes="unit-test entry (synthetic)",
        registry_path=env.registry,
    )
    base.update(kw)
    return log_result(**base)


def _card(env, name="sentinel_sonde_test.md", models=("sentinel_sonde_test",)):
    (env.repo / "docs" / "model_cards" / name).write_text(
        "---\nmodel_id: sentinel_sonde_test\nname: Test sonde model\nmodality: sonde\n"
        f"registry_models: {list(models)}\nstatus: unit test\n---\n# Test card\nNo metrics here.\n"
    )


SONDE_CSV = "time_utc,do_mgl,temp_c\n2024-01-01T00:00:00Z,8.1,10.0\n2024-01-01T01:00:00Z,8.0,10.1\n"


# ----------------------------------------------------------------------------- API: empty states


def test_health_and_empty_states_when_nothing_exists(env):
    c = _client()
    h = c.get("/health").json()
    assert h["status"] == "ok"
    assert h["data"]["daily_index"] is False and h["data"]["hourly_sites"] == 0
    assert h["registry"]["exists"] is False and h["registry"]["has_validated_results"] is False
    assert h["live"]["api_key_configured"] is False
    assert h["live_trial"]["started"] is False
    s = c.get("/sites").json()
    assert s["total"] == 0 and s["sites"] == []
    assert any("no site catalogue" in n for n in s["provenance"]["notes"])
    m = c.get("/models").json()
    assert m["cards"] == [] and "no model cards" in m["message"]
    assert all(v["status"] == "model not yet trained" for v in m["plugins"]["by_modality"].values())
    r = c.get("/registry").json()
    assert r["count"] == 0 and "does not exist yet" in r["message"]
    assert c.get(f"/sites/{SITE_A}/observations").status_code == 404


# ----------------------------------------------------------------------------- API: sites


def test_sites_use_catalog_fallback_with_availability_and_provenance(env):
    _full_data(env)
    c = _client()
    body = c.get("/sites").json()
    assert body["total"] == 1  # default: only sites with local data
    a = body["sites"][0]
    assert a["id"] == SITE_A and a["name"] == "TEST CREEK A" and a["site_type_label"] == "Stream"
    assert a["lat"] == pytest.approx(40.5) and a["lon"] == pytest.approx(-90.25) and a["huc"] == "07090001"
    assert a["daily_variables"] == ["do_mgl", "temp_c"] and a["hourly_variables"] == ["do_mgl"]
    assert a["record_start"] == "2024-01-01" and a["record_end"] == "2024-01-10"
    prov = body["provenance"]
    assert any("sites.parquet not found" in n for n in prov["notes"])
    daily = next(m for m in prov["manifests"] if m.get("name") == "sonde_daily")
    assert daily["status"] == "present" and daily["synthetic"] is True and daily["created_utc"]
    assert daily["upstream"][0]["name"] == "usgs_dv_raw" and daily["upstream"][0]["status"] == "missing"
    assert prov["synthetic"] is True
    assert c.get("/sites", params={"has_data": "any"}).json()["total"] == 2
    assert c.get("/sites", params={"var": "do_mgl", "tier": "hourly"}).json()["total"] == 1
    assert c.get("/sites", params={"var": "ph"}).json()["total"] == 0
    assert c.get("/sites", params={"huc": "0709", "has_data": "any"}).json()["total"] == 1
    assert (
        c.get("/sites", params={"bbox": "-101,35,-100,36", "has_data": "any"}).json()["sites"][0]["id"]
        == SITE_B
    )
    assert c.get("/sites", params={"bbox": "1,2,3"}).status_code == 422


def test_sites_table_is_cross_checked_against_raw_catalogue(env):
    _full_data(env)
    d = env.data / "processed" / "sites"
    d.mkdir(parents=True)
    pd.DataFrame(
        {
            "site_id": [SITE_A, SITE_B],
            "name": ["Test Creek A (table)", "Wrong row"],
            "lat": [40.5, 10.0],
            "lon": [-90.25, 10.0],
            "site_type": ["ST", "LK"],
            "state": ["IL", "ME"],
            "huc8": ["07090001", "01010001"],
            "network": ["USGS", "USGS"],
        }
    ).to_parquet(d / "sites.parquet")
    body = _client().get("/sites", params={"has_data": "any"}).json()
    by = {s["id"]: s for s in body["sites"]}
    assert by[SITE_A]["catalog_source"] == "sites_table" and by[SITE_A]["catalog_conflict"] is False
    assert by[SITE_B]["catalog_source"] == "usgs_dv_catalog" and by[SITE_B]["catalog_conflict"] is True
    assert by[SITE_B]["state"] == "TX" and by[SITE_B]["lat"] == pytest.approx(35.25)
    assert any("disagrees with the raw USGS" in n for n in body["provenance"]["notes"])


# ----------------------------------------------------------------------------- API: observations


def test_daily_observations_values_flags_filters_and_provenance(env):
    _full_data(env)
    c = _client()
    r = c.get(f"/sites/{SITE_A}/observations", params={"var": "do_mgl", "start": "2024-01-03"})
    assert r.status_code == 200
    b = r.json()
    assert [v["name"] for v in b["variables"]] == ["do_mgl_min"] and b["variables"][0]["unit"] == "mg/L"
    rows = b["rows"]
    assert rows[0] == {
        "date": "2024-01-03",
        "variable": "do_mgl_min",
        "value": 6.0,
        "provisional": False,
        "ts_id": "111",
    }
    assert [x["date"] for x in rows] == [
        "2024-01-03",
        "2024-01-04",
        "2024-01-05",
        "2024-01-07",
        "2024-01-08",
        "2024-01-09",
        "2024-01-10",
    ]  # the NaN day is absent, not filled
    assert rows[1]["value"] == 2.5 and [x["provisional"] for x in rows][-3:] == [True, True, True]
    p = b["provenance"]
    assert p["manifests"][0]["name"] == "sonde_daily" and p["manifests"][0]["synthetic"] is True
    assert (
        p["approval"]["n_provisional"] == 3 and p["approval"]["n_approved"] == 4 and "rule" in p["approval"]
    )
    assert (
        p["files"][0]["path"].endswith(f"sonde_daily/sites/{SITE_A}.parquet")
        and p["files"][0]["modified_utc"]
    )
    page = c.get(f"/sites/{SITE_A}/observations", params={"var": "do_mgl_min,temp_c_mean", "limit": 5}).json()
    assert page["total_rows"] == 19 and page["count"] == 5 and page["next_offset"] == 5 and page["truncated"]
    assert c.get(f"/sites/USGS-{SITE_A}/observations").status_code == 200
    assert c.get(f"/sites/{SITE_A}/observations", params={"var": "ph"}).status_code == 422
    assert c.get(f"/sites/{SITE_A}/observations", params={"start": "not-a-date"}).status_code == 422
    assert c.get(f"/sites/{SITE_B}/observations").status_code == 404
    assert c.get("/sites/bad.id/observations").status_code == 422


def test_hourly_observations(env):
    _full_data(env)
    b = (
        _client()
        .get(
            f"/sites/{SITE_A}/observations",
            params={"tier": "hourly", "var": "do_mgl_mean", "start": "2024-01-02", "end": "2024-01-02"},
        )
        .json()
    )
    assert b["total_rows"] == 24
    assert b["rows"][0]["time_utc"] == "2024-01-02T00:00:00Z" and b["rows"][0]["n_raw"] == 4
    assert sum(r["provisional"] for r in b["rows"]) == 8
    assert b["provenance"]["manifests"][0]["name"] == "sonde_hourly"
    assert b["provenance"]["manifests"][0]["status"] == "missing"  # reported, never filled in


def test_site_detail_lists_hazard_lines_from_hazards_module(env):
    from sentinel.eval.hazards import HAZARDS

    _full_data(env)
    b = _client().get(f"/sites/{SITE_A}").json()
    keys = {h["key"] for h in b["hazards"]}
    assert {"H1_hypoxia", "H1_hypoxia_t4", "H1_hypoxia_t2"} <= keys
    h1 = next(h for h in b["hazards"] if h["key"] == "H1_hypoxia" and h["applies_to"] == "do_mgl_min")
    assert h1["threshold"] == HAZARDS["H1_hypoxia"].threshold and h1["anchor"] == HAZARDS["H1_hypoxia"].anchor


# ----------------------------------------------------------------------------- API: live


class _Resp:
    def __init__(self, status, payload=None, headers=None):
        self.status_code, self._payload, self.headers = status, payload, headers or {}

    def json(self):
        return self._payload


class _Session:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append({"url": url, "params": params, "headers": dict(headers or {})})
        return self.responses.pop(0)


def _feature(code, value, unit, status="Provisional"):
    return {
        "type": "Feature",
        "id": "x",
        "geometry": None,
        "properties": {
            "time_series_id": "ts1",
            "monitoring_location_id": f"USGS-{SITE_A}",
            "parameter_code": code,
            "statistic_id": "00011",
            "time": "2026-10-22T08:45:00+00:00",
            "value": value,
            "unit_of_measure": unit,
            "approval_status": status,
            "qualifier": None,
            "last_modified": "2026-10-22T09:01:10+00:00",
        },
    }


def test_live_values_are_labelled_and_the_key_never_leaks(env, caplog):
    from sentinel.platform.live import LiveClient

    (env.root / "keys.env").write_text(f"export API_USGS_PAT='{SECRET}'\n")
    doc = {
        "type": "FeatureCollection",
        "features": [_feature("00300", "8.8", "mg/l"), _feature("99999", None, "x")],
        "links": [],
        "timeStamp": "2026-10-22T09:25:13Z",
    }
    sess = _Session([_Resp(200, doc)])
    caplog.set_level(logging.DEBUG)
    c = _client(LiveClient(session=sess, keys_file=str(env.root / "keys.env")))
    r = c.get(f"/live/{SITE_A}")
    assert r.status_code == 200
    b = r.json()
    assert b["data_status"] == "provisional, live" and "subject to revision" in b["notice"]
    do = next(v for v in b["values"] if v["parameter_code"] == "00300")
    assert do["value"] == 8.8 and do["variable"] == "do_mgl" and do["approval_status"] == "Provisional"
    assert next(v for v in b["values"] if v["parameter_code"] == "99999")["value"] is None
    assert b["provenance"]["authenticated"] is True and b["provenance"]["collection"] == "latest-continuous"
    assert sess.calls[0]["headers"]["X-Api-Key"] == SECRET  # sent as a header ...
    assert SECRET not in json.dumps(sess.calls[0]["params"])  # ... never in the URL
    assert SECRET not in r.text and SECRET not in caplog.text  # never returned or logged
    assert c.get(f"/live/{SITE_A}").json()["provenance"]["cache"] == "hit" and len(sess.calls) == 1
    assert c.get("/health").json()["live"]["api_key_configured"] is True
    assert c.get("/live/NEON-ARIK").status_code == 422


def test_live_rate_limit_is_honoured_without_calling_usgs_again(env):
    from sentinel.platform.live import LiveClient

    sess = _Session([_Resp(429, {}, {"Retry-After": "120"})])
    c = _client(LiveClient(session=sess))
    r = c.get(f"/live/{SITE_A}")
    assert r.status_code == 503 and r.headers["Retry-After"] == "120"
    assert r.json()["data_status"] == "provisional, live"
    r2 = c.get(f"/live/{SITE_B}", params={"window_hours": 6})
    assert r2.status_code == 503 and len(sess.calls) == 1  # blocked locally
    assert c.get("/health").json()["live"]["rate_limited_for_s"] > 0


def test_live_can_be_disabled(env, monkeypatch):
    monkeypatch.setenv("SENTINEL_LIVE", "0")
    r = _client().get(f"/live/{SITE_A}")
    assert r.status_code == 503 and r.json()["status"] == "live fetch disabled"


# ----------------------------------------------------------------------------- API: models & registry


def test_models_join_cards_to_current_registry_entries(env):
    old = _log(env, value=0.9)
    new = _log(env, value=0.8, supersedes=old["id"])
    base = _log(env, model="climatology", is_baseline=True, split="test", value=1.1)
    _card(env)
    (env.repo / "docs" / "model_cards" / "README.md").write_text("template, not a card")
    b = _client().get("/models").json()
    assert [c["file"] for c in b["cards"]] == ["sentinel_sonde_test.md"]
    ents = b["cards"][0]["registry_entries"]
    assert [e["id"] for e in ents] == [new["id"]]  # superseded entry hidden
    g = ents[0]["guarded"]
    assert g["value"] == 0.8 and g["ci_low"] == 0.4 and new["id"] in g["display"]
    assert b["registry_models_without_card"][0]["model"] == "climatology"
    assert b["registry_models_without_card"][0]["registry_entries"][0]["id"] == base["id"]
    assert b["registry"]["has_validated_results"] is False  # baseline on test, model on val


def test_registry_endpoint_filters_and_is_read_only(env):
    a = _log(env)
    _log(env, workstream="orbit", model="m2", experiment="e2")
    c = _client()
    assert c.get("/registry").json()["count"] == 2
    b = c.get("/registry", params={"workstream": "sonde"}).json()
    assert b["count"] == 1 and b["entries"][0]["id"] == a["id"] and "config" not in b["entries"][0]
    assert c.get("/registry", params={"model": "nope"}).json()["count"] == 0
    for method in ("post", "put", "delete"):
        assert getattr(c, method)("/registry").status_code == 405


def test_validated_results_flag_needs_non_baseline_test_entry(env):
    from sentinel.platform import registry_view
    from sentinel.platform.config import Settings

    _log(env, split="t-F2F/occurrence/test", is_baseline=True)
    assert registry_view.status(Settings.from_env())["has_validated_results"] is False
    _log(env, split="t-F2F/occurrence/test")
    st = registry_view.status(Settings.from_env())
    assert st["has_validated_results"] is True and st["n_test_non_baseline"] == 1
    assert registry_view.is_test_split("test_b") and not registry_view.is_test_split("t-F2F/all")


# ----------------------------------------------------------------------------- API: predict


def test_predict_validates_then_answers_503_without_a_model(env):
    c = _client()
    r = c.post("/predict/sonde", files={"file": ("x.csv", SONDE_CSV)})
    assert r.status_code == 503
    b = r.json()
    assert b["status"] == "model not yet trained" and b["validation"]["ok"] is True
    assert b["validation"]["summary"]["rows"] == 2
    bad = c.post("/predict/sonde", files={"file": ("x.csv", "time_utc,turbidity\n2024-01-01T00:00Z,5\n")})
    assert bad.status_code == 422 and "turb_fnu" in json.dumps(bad.json()["errors"])
    assert c.post("/predict/radar", files={"file": ("x.csv", SONDE_CSV)}).status_code == 404
    assert c.post("/predict/orbit", files={"file": ("x.csv", SONDE_CSV)}).status_code == 422
    for m in ("orbit", "lite", "microbe", "chemtox", "bioassess"):
        assert c.get("/models").json()["plugins"]["by_modality"][m]["status"] == "model not yet trained"


def test_predict_enforces_size_limits(env):
    from sentinel.platform import config

    config.set_overrides(upload_limits={m: 64 for m in config.MODALITIES})
    r = _client().post("/predict/sonde", files={"file": ("x.csv", SONDE_CSV * 10)})
    assert r.status_code == 413 and r.json()["status"] == "upload too large"


def _plugin_yaml(env, entry="sentinel_test_plugin:predict"):
    (env.repo / "models.yaml").write_text(
        "version: 1\nmodels:\n  - modality: sonde\n    model_id: sentinel_sonde_test\n"
        f'    entry_point: "{entry}"\n    required_inputs: [time_utc, do_mgl]\n'
        "    registry_experiment: test_exp\n    model_card: docs/model_cards/sentinel_sonde_test.md\n"
    )


def test_predict_serves_registered_validated_plugin_with_registry_metrics(env, monkeypatch):
    mod = types.ModuleType("sentinel_test_plugin")

    def predict(upload):
        df = upload.data
        return {
            "predictions": [
                {
                    "time_utc": str(df["time_utc"].iloc[-1]),
                    "target": "do_mgl",
                    "echo_of_last_input": float(df["do_mgl"].iloc[-1]),
                }
            ],
            "units": {"do_mgl": "mg/L"},
            "model_version": "unit-test",
        }

    mod.predict = predict
    monkeypatch.setitem(sys.modules, "sentinel_test_plugin", mod)
    _plugin_yaml(env)
    c = _client()
    r = c.post("/predict/sonde", files={"file": ("x.csv", SONDE_CSV)})
    assert r.status_code == 503 and r.json()["status"] == "model not validated"  # no card, no registry entry
    _card(env)
    entry = _log(env)
    r = c.post("/predict/sonde", files={"file": ("x.csv", SONDE_CSV)})
    assert r.status_code == 200
    b = r.json()
    assert b["label"].startswith("MODEL OUTPUT") and b["predictions"][0]["echo_of_last_input"] == 8.0
    assert [m["id"] for m in b["validated_metrics"]] == [entry["id"]] and b["validated_metrics"][0][
        "value"
    ] == 0.5
    miss = c.post("/predict/sonde", files={"file": ("x.csv", "time_utc,temp_c\n2024-01-01T00:00Z,5\n")})
    assert miss.status_code == 422 and "do_mgl" in json.dumps(miss.json()["errors"])


def test_predict_rejects_plugins_that_report_metrics(env, monkeypatch):
    mod = types.ModuleType("sentinel_bad_plugin")
    mod.predict = lambda upload: {"predictions": [], "units": {"x": "y"}, "model_version": "t", "accuracy": 1}
    monkeypatch.setitem(sys.modules, "sentinel_bad_plugin", mod)
    _plugin_yaml(env, "sentinel_bad_plugin:predict")
    _card(env)
    _log(env)
    r = _client().post("/predict/sonde", files={"file": ("x.csv", SONDE_CSV)})
    assert r.status_code == 500 and "registry" in r.json()["detail"]


def test_unbuilt_components_answer_501(env):
    c = _client()
    assert c.get("/alerts").status_code == 501 and "not implemented" in c.get("/alerts").json()["status"]
    assert c.post("/simulate").status_code == 501
    for path in (f"/forecast/{SITE_A}", f"/causal/{SITE_A}", "/downstream/safeswim"):
        assert c.get(path).status_code == 501


def test_provenance_endpoint(env):
    _write_daily(env.data)
    c = _client()
    assert c.get("/provenance/sonde_daily").json()["synthetic"] is True
    assert c.get("/provenance/usgs_dv_raw").status_code == 404
    assert c.get("/provenance/..%2Fetc").status_code in (404, 422)


# ----------------------------------------------------------------------------- guard


def test_guard_accepts_only_current_registry_entries(env):
    from sentinel.platform import guard

    old = _log(env, value=0.9)
    new = _log(env, value=0.8, supersedes=old["id"])
    reg = env.registry
    for raw in (
        0.93,
        1,
        np.float64(0.93),
        np.float32(0.5),
        Decimal("0.9"),
        "0.93",
        "93%",
        True,
        np.array([0.9]),
    ):
        with pytest.raises(guard.MetricGuardError):
            guard.guarded_metric(raw, registry_path=reg)
    with pytest.raises(guard.MetricGuardError, match="unknown"):
        guard.guarded_metric("deadbeef0000", registry_path=reg)
    with pytest.raises(guard.MetricGuardError, match="superseded"):
        guard.guarded_metric(old["id"], registry_path=reg)
    with pytest.raises(guard.MetricGuardError, match="does not match"):
        guard.guarded_metric({**new, "value": 0.99}, registry_path=reg)  # hand-edited row
    with pytest.raises(guard.MetricGuardError):
        guard.guarded_metric({"value": 0.8, "metric": "crps"}, registry_path=reg)  # no id
    g = guard.guarded_metric(new["id"], registry_path=reg)
    assert g.value == 0.8 and g.id == new["id"]
    assert new["id"] in guard.render_metric(new, registry_path=reg)
    assert guard.metric_records([pd.Series(new)], registry_path=reg)[0]["ci_high"] == 0.6


def test_guard_accepts_registry_ids_that_look_numeric(env):
    from sentinel.platform import guard

    env.registry.write_text(
        "".join(
            json.dumps(
                {
                    "id": i,
                    "value": v,
                    "metric": "crps",
                    "model": "m",
                    "experiment": "e",
                    "split": "val",
                    "supersedes": None,
                }
            )
            + "\n"
            for i, v in (("123456789012", 0.5), ("1234e5678901", 0.6))
        )
    )
    assert guard.guarded_metric("123456789012", registry_path=env.registry).value == 0.5
    assert guard.guarded_metric("1234e5678901", registry_path=env.registry).value == 0.6
    with pytest.raises(guard.MetricGuardError, match="raw number"):
        guard.guarded_metric("0.5", registry_path=env.registry)


def test_no_typed_in_metric_claims_or_rng_in_platform_sources():
    from sentinel.platform import guard

    findings = guard.scan_platform_sources(
        [REPO / "docs" / "platform.md", REPO / "docs" / "model_cards" / "README.md"]
    )
    assert not findings, "typed-in claims in platform sources:\n" + "\n".join(map(str, findings))
    for bad in (
        "R² 0.9",
        "NSE = 0.86",
        "accuracy 95%",
        "skill score of 0.4",
        "± 0.05",
        "25/32 events detected",
        "state-of-the-art",
        "AUROC: .91",
        "OBSERVED",
        "every figure here is real",
    ):
        assert guard.scan_text(bad), bad
    for ok in (
        "SENTINEL v2.0.0.dev0",
        "python 3.11",
        "HTTP 503",
        "§12",
        "{x:.1%}",
        "width: 100%",
        "% saturation",
        "port 8501",
        "2026-10-22",
    ):
        assert not guard.scan_text(ok), ok


def test_scanner_flags_claims_in_python_strings_and_rng(tmp_path):
    from sentinel.platform import guard

    (tmp_path / "bad.py").write_text(
        'import numpy as np\nTEXT = f"NSE {1} and accuracy 0.9"\nx = np.random.rand(3)\n'
    )
    (tmp_path / "ok.py").write_text('TEXT = f"value {v:.1%} for site {s}"\n')
    rules = {f.rule for f in guard.scan_platform_sources(platform_dir=tmp_path)}
    assert "metric word followed by a number" in rules and "random-number generator in platform code" in rules
    assert all(f.path.endswith("bad.py") for f in guard.scan_platform_sources(platform_dir=tmp_path))


# ----------------------------------------------------------------------------- validation


def _v(modality, name, content, **kw):
    from sentinel.platform.validation import validate

    return validate(modality, name, content if isinstance(content, bytes) else content.encode(), **kw)


def test_sonde_validation_rules():
    ok = _v(
        "sonde",
        "a.csv",
        "time,do_mgl,temp_c_qualifier\n2024-01-01T00:00:00-05:00,-999999,Eqp\n"
        "2024-01-01T01:00:00-05:00,7.9,\n",
    )
    assert (
        ok.ok
        and ok.data["do_mgl"].isna().sum() == 1
        and any("missing-value code" in w["message"] for w in ok.warnings)
    )
    assert not _v("sonde", "a.csv", "time,do_mgl\n2024-01-01T00:00:00,7\n").ok  # no UTC offset
    assert not _v("sonde", "a.csv", "time_utc,temp_c\n2024-01-01T00:00Z,75\n").ok  # degF in temp_c
    assert not _v(
        "sonde", "a.csv", "time_utc,do_mgl\n2024-01-01T00:00Z,7\n2024-01-01T00:00Z,8\n"
    ).ok  # dup time
    assert not _v("sonde", "a.csv", "time_utc,do_mgl\n2024-01-01T00:00Z,abc\n").ok
    assert not _v("sonde", "a.csv", "time_utc,chlorophyll\n2024-01-01T00:00Z,5\n").ok  # unit-less name
    daily = _v("sonde", "a.csv", "date,do_mgl_min,do_mgl_mean\n2024-01-02,6,7\n2024-01-01,5,6\n")
    assert daily.ok and any("sorted" in w["message"] for w in daily.warnings)
    assert not _v("sonde", "a.txt", "").ok and not _v("sonde", "a.parquet", "x").ok
    req = _v("sonde", "a.csv", SONDE_CSV, required_inputs=("time_utc", "do_mgl", "spcond_uscm"))
    assert not req.ok and "spcond_uscm" in json.dumps(req.errors)


def _npz(**arrays) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def test_raster_validation_orbit_and_lite():
    chip = np.ones((5, 16, 16), dtype="float32")
    names = np.array(["B02", "B03", "B04", "B05", "B08"])
    rep = _v("orbit", "c.npz", _npz(chip=chip, band_names=names), required_inputs=("B05",))
    assert rep.ok and rep.summary["bands"] == 5
    assert not _v("orbit", "c.npz", _npz(chip=chip), required_inputs=("B05",)).ok  # unnamed bands
    assert not _v("orbit", "c.npz", _npz(chip=np.ones((16, 16)))).ok  # not 3-D
    assert not _v(
        "orbit", "c.npz", _npz(chip=np.array([{"a": 1}], dtype=object))
    ).ok  # pickled object refused
    assert not _v("orbit", "c.npz", b"not a zip").ok
    lite = _v(
        "lite",
        "d.npz",
        _npz(rgbn=np.dstack([np.full((8, 8), i, "uint8") for i in range(4)])),
        options={"band_order": "N,R,G,B"},
    )
    assert lite.ok and lite.data["rgbn"][0, 0, 0] == 1 and lite.data["rgbn"][3, 0, 0] == 0
    assert not _v("lite", "d.npz", _npz(rgbn=np.ones((3, 8, 8)))).ok  # needs 4 bands
    assert not _v("lite", "d.npz", _npz(rgbn=np.ones((4, 8, 8))), options={"band_order": "R,G,B,B"}).ok


def test_geotiff_validation():
    pytest.importorskip("rasterio")
    from rasterio.io import MemoryFile
    from rasterio.transform import from_origin

    with MemoryFile() as mem:
        with mem.open(
            driver="GTiff",
            width=8,
            height=8,
            count=4,
            dtype="uint16",
            crs="EPSG:32615",
            transform=from_origin(500000, 4500000, 1, 1),
        ) as ds:
            ds.write(np.ones((4, 8, 8), dtype="uint16"))
        data = mem.read()
    rep = _v("lite", "img.tif", data)
    assert rep.ok and rep.summary["crs"] == "EPSG:32615" and rep.summary["height"] == 8
    assert _v("orbit", "img.tif", data).ok


def test_table_validation_microbe_chemtox_bioassess():
    assert _v("microbe", "t.tsv", "#OTU ID\tS1\tS2\nasv1\t10\t0\nasv2\t3\t5\n").ok
    assert not _v("microbe", "t.tsv", "#OTU ID\tS1\nasv1\t-1\n").ok
    assert not _v("microbe", "t.csv", "asv_id,S1\nasv1,1\nasv1,2\n").ok
    assert not _v("microbe", "t.csv", "name,S1\nasv1,1\n").ok
    from sentinel.platform.validation import cas_checksum_ok

    assert cas_checksum_ok("7732-18-5") and not cas_checksum_ok("7732-18-4")
    chem = _v("chemtox", "c.txt", "7732-18-5\nCCO\n")
    assert chem.ok and chem.summary["cas"] == 1 and chem.summary["smiles"] == 1
    assert not _v("chemtox", "c.txt", "7732-18-4\n").ok
    assert not _v("chemtox", "c.txt", "C1CC(\n").ok
    assert _v("chemtox", "c.csv", "name,smiles\nethanol,CCO\n").ok
    assert not _v("chemtox", "c.txt", "CCO\n", required_inputs=("cas",)).ok
    assert _v(
        "bioassess",
        "b.csv",
        "sample_id,taxon,count,lat,lon\ns1,Baetis,4,40.1,-90\ns1,Hydropsyche,2,40.1,-90\n",
    ).ok
    assert not _v("bioassess", "b.csv", "sample_id,taxon\ns1,Baetis\n").ok
    assert not _v("bioassess", "b.csv", "sample_id,taxon,count\ns1,Baetis,4\ns1,Baetis,2\n").ok
    assert not _v("bioassess", "b.csv", "sample_id,taxon,count,lat\ns1,Baetis,4,95\n").ok


# ----------------------------------------------------------------------------- small modules


def test_every_variable_has_unit_and_label():
    from sentinel.data.sources.usgs_legacy import ALL_PARAMS
    from sentinel.platform import variables as V

    for name in ALL_PARAMS.values():
        assert V.UNITS.get(name) and V.LABELS.get(name), name
    assert V.split_column("discharge_tf_cfs_mean") == ("discharge_tf_cfs", "mean")


def test_status_helpers_read_files_verbatim(env):
    from sentinel.platform import status
    from sentinel.platform.config import Settings

    (env.repo / "RESULTS.md").write_text("# Results\n**Last updated**: today\n**Status**: In progress\n")
    (env.repo / "RESEARCH_PLAN.md").write_text(
        "# Plan\n## 3. Integrity rules\n- R1 rule\n---\n## 4. Next\nx\n"
    )
    s = Settings.from_env()
    h = status.results_headline(s)
    assert h["Status"] == "In progress" and "Threshold status" not in h
    assert status.plan_section(s, 3) == "## 3. Integrity rules\n- R1 rule"
    assert status.live_trial(s)["started"] is False
    (env.runs / "live").mkdir(parents=True)
    (env.runs / "live" / "2026-12-01.csv").write_text("reference_datetime,site_id\n")
    assert status.live_trial(s)["started"] is True


# ----------------------------------------------------------------------------- app pages (no server)


def _view(view):
    import importlib as _il

    _il.import_module(f"sentinel.platform.app.views.{view}").render()


VIEWS = ["home", "map_view", "site_page", "models_page", "upload", "early_warning", "integrity"]


@pytest.mark.parametrize("view", VIEWS)
def test_pages_render_with_data(env, view):
    from streamlit.testing.v1 import AppTest

    _full_data(env)
    _card(env)
    _log(env)
    (env.repo / "RESULTS.md").write_text("**Status**: In progress\n")
    at = AppTest.from_function(_view, args=(view,), default_timeout=60)
    at.session_state["site_id"] = SITE_A
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    if view == "early_warning":
        assert any("has not started" in w.value for w in at.warning)
    if view == "map_view":
        assert any("SYNTHETIC" in e.value for e in at.error)  # test manifests are synthetic


@pytest.mark.parametrize("view", VIEWS)
def test_pages_render_empty_states(env, view):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(_view, args=(view,), default_timeout=60)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    if view in ("map_view", "site_page", "models_page"):
        assert at.info, f"{view} should explain what is missing"


def test_upload_page_reports_validation_and_missing_model(env, monkeypatch):
    from streamlit.testing.v1 import AppTest

    from sentinel.platform.app import ui

    client = _client()

    def via_testclient(s, method, path, *, timeout=60.0, **kw):  # in-process API, no network
        r = client.request(method, path, files=kw.get("files"), data=kw.get("data"))
        return r.status_code, r.json()

    monkeypatch.setattr(ui, "api_request", via_testclient)
    at = AppTest.from_function(_view, args=("upload",), default_timeout=60)
    at.run()
    at.file_uploader[0].upload("x.csv", SONDE_CSV.encode(), "text/csv").run()
    at.button[0].click().run()
    assert not at.exception
    assert any("model not yet trained" in w.value for w in at.warning)
    assert any("passed validation" in m.value for m in at.success)
    at.file_uploader[0].upload("y.csv", b"time_utc,turbidity\n2024-01-01T00:00Z,5\n", "text/csv").run()
    at.button[0].click().run()
    assert any("failed validation" in e.value for e in at.error)


def test_entry_script_banner_tracks_registry(env):
    from streamlit.testing.v1 import AppTest

    script = str(REPO / "sentinel" / "platform" / "app" / "streamlit_app.py")
    at = AppTest.from_file(script, default_timeout=60)
    at.run()
    assert not at.exception and any("No validated v2 results yet" in w.value for w in at.warning)
    _log(env, split="test_a")
    at = AppTest.from_file(script, default_timeout=60)
    at.run()
    assert not at.exception and not any("No validated v2 results yet" in w.value for w in at.warning)
