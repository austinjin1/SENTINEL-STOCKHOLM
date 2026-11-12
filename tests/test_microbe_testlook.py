# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Single-look guard of the M3 test scorer (sentinel/models/microbe/testlook.py). All files live in tmp_path;
the real registry, runs folder and freeze directory are never touched."""

import json

import pytest

from sentinel.core import paths
from sentinel.models.microbe import testlook as TL


@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setattr(paths, "REGISTRY_PATH", tmp_path / "registry.jsonl")
    fdir = tmp_path / "prereg"
    fdir.mkdir()
    (fdir / "microbe_split.json").write_text("{}")
    monkeypatch.setattr(TL, "FREEZE_DIR", fdir)
    return tmp_path, fdir


def _freeze(fdir, name=TL.FREEZE_NAME, **over):
    rec = {"created": "2026-11-10", "status": "FROZEN", "authorised_by": "lead", "model": {"name": "x"},
           "split_sha256": TL.sha_file(fdir / "microbe_split.json"), "scoring_code_sha256": "abc",
           "looks": {"M3/test": {}}, "data_snapshots": {"d": {}}, "artifacts": {"a": "b"}}
    rec.update(over)
    (fdir / name).write_text(json.dumps(rec))
    return fdir / name


def test_candidate_and_unauthorised_freezes_are_refused(sandbox):
    _, fdir = sandbox
    with pytest.raises(TL.ProtectedDataError):
        TL.load_freeze(_freeze(fdir, TL.CANDIDATE_NAME), check_code=False)            # candidate file name
    for over in ({"status": "CANDIDATE"}, {"authorised_by": ""}, {"NOT_A_FREEZE": True}, {"candidate": True}):
        with pytest.raises(TL.ProtectedDataError):
            TL.load_freeze(_freeze(fdir, **over), check_code=False)
    fr = TL.load_freeze(_freeze(fdir), check_code=False)
    assert fr["_sha256"]
    with pytest.raises(TL.ProtectedDataError):                                         # code hash checked
        TL.load_freeze(_freeze(fdir))


def test_claim_is_exclusive_and_refuses_existing_test_rows(sandbox):
    tmp, fdir = sandbox
    fr = TL.load_freeze(_freeze(fdir), check_code=False)
    paths.REGISTRY_PATH.write_text(json.dumps({"workstream": TL.WORKSTREAM, "experiment": "microbe_test",
                                               "metric": "r2"}) + "\n")
    with pytest.raises(TL.ProtectedDataError):
        TL.claim("M3/test", fr)
    paths.REGISTRY_PATH.write_text("")
    TL.claim("M3/test", fr)
    assert TL.look_dir("M3/test").joinpath("TEST_LOOK.json").exists()
    with pytest.raises(TL.ProtectedDataError):                                         # claim row now exists
        TL.claim("M3/test", fr)


def test_access_closes_at_sealing_and_sealed_folder_is_verified(sandbox):
    tmp, fdir = sandbox
    fr = TL.load_freeze(_freeze(fdir), check_code=False)
    with pytest.raises(TL.ProtectedDataError):
        TL.require_open_claim()
    TL.claim("M3/test", fr)
    assert TL.require_open_claim(fr["_sha256"]) == "M3/test"
    d = TL.pred_dir("M3/test")
    d.mkdir(parents=True)
    (d / "timeout__x.parquet").write_bytes(b"pred")
    TL.seal_predictions("M3/test")
    with pytest.raises(TL.ProtectedDataError):                                         # access closed
        TL.require_open_claim()
    assert TL.check_sealed("M3/test") == d
    (d / "extra.parquet").write_bytes(b"x")                                            # unlisted file
    with pytest.raises(TL.ProtectedDataError):
        TL.check_sealed("M3/test")
    (d / "extra.parquet").unlink()
    TL.assert_no_scoring_rows("M3/test")
    with open(paths.REGISTRY_PATH, "a") as fh:                                         # a scoring row exists
        fh.write(json.dumps({"workstream": TL.WORKSTREAM, "experiment": "M3/test", "metric": "r2"}) + "\n")
    with pytest.raises(TL.ProtectedDataError):
        TL.assert_no_scoring_rows("M3/test")                                           # --rescore-from-saved refused
    TL.mark_scored("M3/test", {})
    with pytest.raises(TL.ProtectedDataError):
        TL.check_sealed("M3/test")


def test_sealed_files_must_match_registry_sealed_row(sandbox):
    tmp, fdir = sandbox
    fr = TL.load_freeze(_freeze(fdir), check_code=False)
    TL.claim("M3/test", fr)
    d = TL.pred_dir("M3/test")
    d.mkdir(parents=True)
    (d / "p.parquet").write_bytes(b"pred")
    TL.seal_predictions("M3/test")
    rows = [json.loads(x) for x in paths.REGISTRY_PATH.read_text().splitlines()]
    for r in rows:
        if r.get("metric") == "predictions_sealed":
            r["extra"]["sha256sums"] = "0" * 64
    paths.REGISTRY_PATH.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(TL.ProtectedDataError):
        TL.check_sealed("M3/test")


def test_holdout_loader_needs_open_claim(sandbox):
    from sentinel.models.microbe.design import load_design

    with pytest.raises(TL.ProtectedDataError):
        load_design(allow_holdout=True)


def test_code_hash_covers_events_import_closure():
    files = set(TL.frozen_files())
    for f in ("sentinel/eval/events.py", "sentinel/eval/ew_scoring.py", "sentinel/eval/hazards.py",
              "sentinel/eval/early_warning.py", "sentinel/eval/metrics.py", "sentinel/eval/bootstrap.py",
              "sentinel/eval/__init__.py", "sentinel/events/microbe.py", "scripts/eval/microbe_final.py",
              "sentinel/models/microbe/testlook.py", "sentinel/models/microbe/baselines.py",
              "configs/events/microbe.yaml"):
        assert f in files, f


def test_claim_is_tied_to_pid_and_host(sandbox, monkeypatch):
    import os

    tmp, fdir = sandbox
    fr = TL.load_freeze(_freeze(fdir), check_code=False)
    TL.claim("M3/test", fr)
    assert TL.require_open_claim() == "M3/test"
    monkeypatch.setattr(os, "getpid", lambda: -1)            # another process
    with pytest.raises(TL.ProtectedDataError):
        TL.require_open_claim()
