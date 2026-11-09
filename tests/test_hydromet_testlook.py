# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""M6 single-pass test-look guard (pre-test review 2026-11-09 C1–C3). No real data, no real registry writes."""

import json
import os
import subprocess
import sys

import pytest

from sentinel.core import paths
from sentinel.models.hydromet import testlook as TL
from sentinel.models.sonde.guard import ProtectedDataError


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RUNS_ROOT", tmp_path / "runs")
    rows = {"n": 0}
    monkeypatch.setattr(TL, "registry_rows", lambda exp: rows["n"])
    import sentinel.core.registry as R

    monkeypatch.setattr(R, "log_result", lambda **k: None)
    return rows


def _blk(dataset, n_models=1):
    comps = {"temp_c_mean": {}, "do_mgl_mean": {}}
    return {"block": "test", "roles": ["test"], "dataset": dataset, "primary_models": ["m"] * n_models,
            "n_hypotheses": 2 * n_models, "comparators": comps, "artifacts": {}, "data_snapshots": []}


def _freeze(sha="a" * 64):
    return {"_path": "/x/FREEZE_hydromet.json", "_sha256": sha, "scoring_code_sha256": "c" * 64,
            "looks": {"M6/test": _blk("sonde"), "M6N/test": _blk("trackn")}}


def test_second_look_refused(sandbox):
    TL.claim("M6/test", _freeze())
    assert TL.require_open_claim(_freeze()) == "M6/test"
    with pytest.raises(ProtectedDataError, match="already claimed"):
        TL.claim("M6/test", _freeze())
    assert oct(os.stat(TL.marker("M6/test")).st_mode & 0o777) == "0o444"


def test_registry_rows_refuse_claim(sandbox):
    sandbox["n"] = 3
    with pytest.raises(ProtectedDataError, match="registry already holds"):
        TL.claim("M6N/test", _freeze())
    assert not TL.marker("M6N/test").exists()


def test_rewritten_freeze_refused(sandbox):
    TL.claim("M6/test", _freeze("a" * 64))
    with pytest.raises(ProtectedDataError, match="open test-look claim"):
        TL.require_open_claim(_freeze("b" * 64))        # protected data under a rewritten freeze
    with pytest.raises(ProtectedDataError, match="already claimed"):
        TL.claim("M6/test", _freeze("b" * 64))          # and no second look with it


def test_sealed_predictions_and_scored_marker(sandbox, tmp_path):
    TL.claim("M6/test", _freeze())
    d = TL.look_dir("M6/test") / "predictions"
    (d / "R1").mkdir(parents=True)
    (d / "a__test.npz").write_bytes(b"one")
    (d / "R1" / "b__test.npz").write_bytes(b"two")
    TL.seal_predictions("M6/test", d)
    assert TL.check_sealed("M6/test") == d
    with pytest.raises(ProtectedDataError):
        TL.seal_predictions("M6/test", d)              # predictions are written once
    os.chmod(d / "a__test.npz", 0o644)
    (d / "a__test.npz").write_bytes(b"changed")
    with pytest.raises(ProtectedDataError, match="modified"):
        TL.check_sealed("M6/test")
    TL.mark_scored("M6/test", {"n": 1})
    with pytest.raises(ProtectedDataError, match="already scored"):
        TL.check_sealed("M6/test")
    with pytest.raises(ProtectedDataError):
        TL.require_open_claim(_freeze())                # nothing protected loads after scoring


def test_look_block_family_size_enforced():
    blk = {"block": "test", "roles": ["test"], "dataset": "trackn", "primary_models": ["a", "b"], "n_hypotheses": 3,
           "comparators": {"temp_c_mean": {}, "do_mgl_mean": {}}, "artifacts": {}, "data_snapshots": []}
    with pytest.raises(ProtectedDataError, match="n_hypotheses"):
        TL.look_block({"looks": {"M6N/test": blk}}, "M6N/test")
    blk["n_hypotheses"] = 4
    assert TL.look_block({"looks": {"M6N/test": blk}}, "M6N/test")["n_hypotheses"] == 4
    with pytest.raises(ProtectedDataError, match="no block"):
        TL.look_block({"looks": {}}, "M6/test")


@pytest.mark.parametrize("script,args", [
    ("hydromet_score.py", ["--config", "configs/hydromet/common.yaml", "--predictions", "x", "--block", "test",
                           "--roles", "test", "--primary", "--models", "ealstm_ws_full"]),
    ("hydromet_predict.py", ["--run", "x", "--block", "test", "--roles", "test", "--allow-test", "--out", "y"]),
    ("hydromet_track_n.py", ["score", "--block", "test", "--roles", "test", "--allow-test", "--out", "y",
                             "--predictions", "x", "--comparators", "c.json"]),
])
def test_cli_scorers_refuse_protected(script, args):
    r = subprocess.run([sys.executable, "-B", str(paths.REPO_ROOT / "scripts" / "eval" / script), *args],
                       capture_output=True, text=True, cwd=paths.REPO_ROOT, timeout=300)
    assert r.returncode != 0 and "hydromet_test_look.py" in (r.stderr + r.stdout)


def test_guard_needs_open_claim(sandbox, monkeypatch):
    from sentinel.models.hydromet import guard

    monkeypatch.setattr(guard, "load_freeze", lambda *a, **k: _freeze())
    with pytest.raises(ProtectedDataError, match="open test-look claim"):
        guard.check_access(("train", "test"), {"test"}, allow_test=True)
    TL.claim("M6/test", _freeze())
    assert guard.check_access(("train", "test"), {"test"}, allow_test=True)["look"] == "M6/test"


def test_claim_scope_is_its_own_cell(sandbox):
    TL.claim("M6N/test", _freeze())
    assert TL.require_open_claim(_freeze(), blocks=("train", "test"), roles={"test"}, dataset="trackn") == "M6N/test"
    with pytest.raises(ProtectedDataError, match="covering that cell"):   # sonde test sites under a track-N claim
        TL.require_open_claim(_freeze(), blocks=("train", "test"), roles={"test"}, dataset="sonde")
    with pytest.raises(ProtectedDataError, match="covering that cell"):   # test_b under a test claim
        TL.require_open_claim(_freeze(), blocks=("train", "test_b"), roles=(), dataset="trackn")


def test_abort_closes_claim_and_logged_resume(sandbox):
    TL.claim("M6/test", _freeze())
    TL.abort("M6/test", "Traceback: boom")
    with pytest.raises(ProtectedDataError):
        TL.require_open_claim(_freeze(), blocks=("test",), roles={"test"})
    assert TL.resume_aborted("M6/test", _freeze()) == 1
    assert TL.require_open_claim(_freeze(), blocks=("test",), roles={"test"}) == "M6/test"
    with pytest.raises(ProtectedDataError):                                # only an aborted look can be resumed
        TL.resume_aborted("M6/test", _freeze())
    TL.abort("M6/test", "again")
    assert TL.resume_aborted("M6/test", _freeze()) == 2
    TL.abort("M6/test", "third")
    with pytest.raises(ProtectedDataError, match="resumed 2 times"):
        TL.resume_aborted("M6/test", _freeze())
    with pytest.raises(ProtectedDataError, match="different freeze"):
        TL.resume_aborted("M6/test", _freeze("b" * 64))


def test_sealed_folder_rejects_extra_files(sandbox):
    TL.claim("M6/test", _freeze())
    d = TL.look_dir("M6/test") / "predictions"
    d.mkdir(parents=True)
    (d / "a__test.npz").write_bytes(b"one")
    TL.seal_predictions("M6/test", d)
    (d / "R2").mkdir()
    (d / "R2" / "a__test.npz").write_bytes(b"planted")
    with pytest.raises(ProtectedDataError, match="not in SHA256SUMS"):
        TL.check_sealed("M6/test")
    with pytest.raises(ProtectedDataError, match="sealed — use"):
        TL.abort("M6/test", "after sealing")


def test_candidate_and_rehearsal_records_are_not_freezes(tmp_path, monkeypatch):
    from sentinel.models.hydromet import guard

    cand = paths.REPO_ROOT / "docs" / "preregistration" / "hydromet_freeze_candidate.json"
    with pytest.raises(ProtectedDataError, match="not a freeze"):
        guard.load_freeze(cand, check_code=False)
    monkeypatch.setattr(guard, "FREEZE_DIR", tmp_path)
    base = {"created": "x", "split_sha256": "s", "scoring_code_sha256": "c", "weights_sha256": {"a": "b"}}
    for extra, msg in [({"NOT_A_FREEZE": "x", "status": "FROZEN", "authorised_by": "l"}, "never a freeze"),
                       ({"partial_rehearsal_only": True, "status": "FROZEN", "authorised_by": "l"}, "never a freeze"),
                       ({"status": "CANDIDATE", "authorised_by": "l"}, "status 'FROZEN'"),
                       ({"status": "FROZEN"}, "authorised_by")]:
        ff = tmp_path / "FREEZE_hydromet.json"
        ff.write_text(json.dumps({**base, **extra}))
        with pytest.raises(ProtectedDataError, match=msg):
            guard.load_freeze(ff, check_code=False)
    other = tmp_path / "my_freeze.json"
    other.write_text(json.dumps({**base, "status": "FROZEN", "authorised_by": "l"}))
    with pytest.raises(ProtectedDataError, match="not a freeze"):
        guard.load_freeze(other, check_code=False)
    ok = tmp_path / "FREEZE_hydromet.json"
    ok.write_text(json.dumps({**base, "status": "FROZEN", "authorised_by": "lead"}))
    assert guard.load_freeze(ok, check_code=False)["status"] == "FROZEN"


def test_test_look_cli_refuses_candidate_as_freeze():
    cand = paths.REPO_ROOT / "docs" / "preregistration" / "hydromet_freeze_candidate.json"
    r = subprocess.run([sys.executable, "-B", str(paths.REPO_ROOT / "scripts" / "eval" / "hydromet_test_look.py"),
                        "--look", "M6/test", "--freeze-file", str(cand)], capture_output=True, text=True,
                       cwd=paths.REPO_ROOT, timeout=300)
    assert r.returncode != 0 and "FREEZE_hydromet.json" in (r.stderr + r.stdout)
    assert not TL.marker("M6/test").exists()
