# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""M2 Orbit single-look enforcement (orbit_protocol.md §9b). No real data, no real registry writes."""

import hashlib
import json
import os

import numpy as np
import pytest

from sentinel.core import paths
from sentinel.models.orbit import guard as G
from sentinel.models.orbit import testlook as TL
from sentinel.models.sonde.guard import ProtectedDataError


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RUNS_ROOT", tmp_path / "runs")
    rows = {"n": 0}
    monkeypatch.setattr(TL, "registry_rows", lambda exp: rows["n"])
    import sentinel.core.registry as R

    monkeypatch.setattr(R, "log_result", lambda **k: None)
    return rows


def _freeze(sha="a" * 64):
    return {"_path": "/x/FREEZE_orbit.json", "_sha256": sha, "scoring_code_sha256": "c" * 64, "data_version": "dev_vX"}


def test_second_claim_refused_marker_readonly(sandbox):
    TL.claim("M2/test", _freeze())
    assert TL.require_open_claim(_freeze()) == "M2/test"
    with pytest.raises(ProtectedDataError, match="already claimed"):
        TL.claim("M2/test", _freeze())
    assert oct(os.stat(TL.marker("M2/test")).st_mode & 0o777) == "0o444"


def test_registry_rows_refuse_claim(sandbox):
    sandbox["n"] = 2
    with pytest.raises(ProtectedDataError, match="registry already holds"):
        TL.claim("M2U/test", _freeze())
    assert not TL.marker("M2U/test").exists()


def test_rewritten_freeze_has_no_open_claim(sandbox):
    TL.claim("M2/test", _freeze("a" * 64))
    with pytest.raises(ProtectedDataError, match="open test-look claim"):
        TL.require_open_claim(_freeze("b" * 64))


def test_sealed_predictions_scored_marker(sandbox):
    TL.claim("M2/test", _freeze())
    d = TL.look_dir("M2/test") / "predictions"
    d.mkdir(parents=True)
    (d / "cell_headline.parquet").write_bytes(b"one")
    (d / "cells.json").write_text("{}")
    TL.seal_predictions("M2/test", d)
    assert oct(os.stat(d / "cell_headline.parquet").st_mode & 0o777) == "0o444"
    with pytest.raises(ProtectedDataError, match="already sealed"):
        TL.seal_predictions("M2/test", d)
    assert TL.check_sealed("M2/test") == d
    os.chmod(d / "cell_headline.parquet", 0o644)
    (d / "cell_headline.parquet").write_bytes(b"tampered")
    with pytest.raises(ProtectedDataError, match="modified"):
        TL.check_sealed("M2/test")
    (d / "cell_headline.parquet").write_bytes(b"one")
    (d / "cell_extra.parquet").write_bytes(b"x")
    with pytest.raises(ProtectedDataError, match="unsealed files"):
        TL.check_sealed("M2/test")
    (d / "cell_extra.parquet").unlink()
    TL.mark_scored("M2/test", {"n": 1})
    with pytest.raises(ProtectedDataError, match="already scored"):
        TL.check_sealed("M2/test")
    assert TL.open_claims(_freeze()["_sha256"]) == []
    with pytest.raises(ProtectedDataError, match="open test-look claim"):
        TL.require_open_claim(_freeze())


def test_data_pins_detect_change(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "PROCESSED", tmp_path)
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.parquet").write_bytes(b"1")
    monkeypatch.setattr(TL, "DATA_PIN_SPECS", {"a": ("a", ["*.parquet"])})
    freeze = {"data_pins": TL.make_data_pins()}
    TL.verify_data_pins(freeze)
    (tmp_path / "a" / "y.parquet").write_bytes(b"2")          # an added file changes the pin
    with pytest.raises(ProtectedDataError, match="content changed"):
        TL.verify_data_pins(freeze)
    with pytest.raises(ProtectedDataError, match="no pin"):
        TL.verify_data_pins({"data_pins": {}})


def test_candidate_never_unlocks(tmp_path, monkeypatch):
    """Only FREEZE_orbit.json unlocks; a candidate path or a non-FROZEN status is refused before anything else."""
    cand = tmp_path / "orbit_freeze_candidate.json"
    frz = tmp_path / "FREEZE_orbit.json"
    monkeypatch.setattr(G, "FREEZE_FILE", frz)
    monkeypatch.setattr(G, "CANDIDATE_FILE", cand)
    body = {k: "x" for k in G.FREEZE_KEYS}
    body.update(weights_sha256={"m": "s"}, data_pins={"a": {"x": 1}}, status="CANDIDATE")
    cand.write_text(json.dumps(body))
    with pytest.raises(ProtectedDataError, match="only .* unlocks"):
        G.load_freeze(cand)
    frz.write_text(json.dumps(body))                         # candidate copied without the lead's authorisation
    with pytest.raises(ProtectedDataError, match="status"):
        G.load_freeze()
    body2 = dict(body, status="FROZEN", authorised_by="lead", authorised="2026-11-10",
                 candidate_sha256=hashlib.sha256(cand.read_bytes()).hexdigest(), weights_sha256={"m": "t"})
    frz.write_text(json.dumps(body2))                        # a key differs from the reviewed candidate
    with pytest.raises(ProtectedDataError, match="differs from the reviewed candidate"):
        G.load_freeze()
    frz.write_text(json.dumps(dict(body2, candidate_sha256="0" * 64)))
    with pytest.raises(ProtectedDataError, match="candidate_sha256"):
        G.load_freeze()
    with pytest.raises(ProtectedDataError):                  # check_rows with allow_test also goes through it
        G.check_rows(np.array([True]), allow_test=True)


def test_event_code_in_scoring_hash():
    files = G.frozen_files()
    for f in G.EVENT_CODE:
        assert f in files
    assert "sentinel/models/orbit/testlook.py" in files
    assert "scripts/eval/orbit_test.py" in files
