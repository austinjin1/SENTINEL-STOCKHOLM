# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Single-look guard of ``M2S/test`` (``sentinel/models/orbit_v2/looks.py``; arch-v2 amendment §5; rulings R-17,
R-21, R-23, R-24). Every guard test of ``tests/test_orbit_testlook.py`` has its mirror here (same name where the
subject is the same), plus the tests of what R-17, R-21 and R-24 add. No real data, no real registry write and no
write under the real runs root: every test runs on a temporary ``paths.RUNS_ROOT`` and the real
``sentinel_runs/orbit/test_looks`` tree must be exactly as before."""

import hashlib
import json
import os
import stat
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sentinel.core import paths
from sentinel.models.orbit import testlook as TL1
from sentinel.models.orbit_v2 import looks as LK
from sentinel.models.sonde.guard import ProtectedDataError

LOOK = "M2S/test"
REAL_RUNS_ROOT = Path(paths.RUNS_ROOT).resolve()        # captured at import, before any fixture changes it
REAL_LOOKS = REAL_RUNS_ROOT / "orbit" / LK.ROOT
_REAL = {"write": LK._excl_write}                       # the unguarded marker writer
V1_STATE = {"looks": tuple(TL1.LOOKS), "writer": TL1._excl_write, "claim": TL1.claim, "look_dir": TL1.look_dir}


def _under_real_runs_root(p) -> bool:
    try:
        Path(p).resolve().relative_to(REAL_RUNS_ROOT)
        return True
    except ValueError:
        return False


def _tree(root: Path) -> dict:
    """``root`` and everything under it: type, size, mtime, mode. Names and metadata only; no file is opened."""
    if not root.exists():
        return {}
    out = {}
    for q in [root, *sorted(root.rglob("*"))]:
        st = q.lstat()
        out[q.relative_to(root).as_posix()] = (stat.S_IFMT(st.st_mode), st.st_size if stat.S_ISREG(st.st_mode) else None,
                                               st.st_mtime_ns, stat.S_IMODE(st.st_mode))
    return out


def _guarded_write(p, rec):
    assert not _under_real_runs_root(p), f"a test tried to write {p} under the real runs root"
    return _REAL["write"](p, rec)


def _no_registry(**k):
    raise AssertionError("a test tried to write a real registry row")


@pytest.fixture(scope="module", autouse=True)
def _base_layer(tmp_path_factory):
    """Base layer for the whole module, with its own MonkeyPatch: a temporary runs root, a marker writer that
    refuses any path under the real runs root, and no registry write. A test-level ``monkeypatch.undo()`` falls back
    to this layer, never to the real root. Only attributes of the v2 module are replaced; v1 ``testlook`` is left
    as imported (checked on teardown, R-21)."""
    before = _tree(REAL_LOOKS)
    mp = pytest.MonkeyPatch()
    mp.setattr(paths, "RUNS_ROOT", tmp_path_factory.mktemp("orbit_v2_runs_base"))
    mp.setattr(LK, "_excl_write", _guarded_write)
    mp.setattr(LK, "_log_result", _no_registry)
    yield
    mp.undo()
    assert _tree(REAL_LOOKS) == before, f"the real {REAL_LOOKS} changed while this test module ran"
    assert (tuple(TL1.LOOKS), TL1._excl_write, TL1.claim, TL1.look_dir) == tuple(V1_STATE.values())


@pytest.fixture(autouse=True)
def _real_test_looks_untouched(tmp_path, monkeypatch):
    before = _tree(REAL_LOOKS)
    monkeypatch.setattr(paths, "RUNS_ROOT", tmp_path / "runs")
    yield
    assert not _under_real_runs_root(paths.RUNS_ROOT), "paths.RUNS_ROOT points at the real runs root inside a test"
    assert LK._excl_write is _guarded_write, "the marker-writer guard was removed inside a test"
    assert _tree(REAL_LOOKS) == before, f"the real {REAL_LOOKS} changed during this test"


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RUNS_ROOT", tmp_path / "runs")
    rows = {"n": 0}
    monkeypatch.setattr(LK, "registry_rows", lambda exp, **k: rows["n"])
    monkeypatch.setattr(LK, "control_row", lambda exp, metric: False)
    monkeypatch.setattr(LK, "_log_result", lambda **k: None)
    return rows


def _freeze(sha="a" * 64):
    return {"_path": "/x/FREEZE_orbit_S.json", "_sha256": sha, "scoring_code_sha256": "c" * 64,
            "data_version": "dev_vXm"}


SEALED = {"v1_networks": 2, "v2": True, "track_n": False, "comparators": ["mdn_b", "gbm", "knn", "mdn_b+gbm"]}


def _required():
    return LK.sealed_columns(SEALED)


def _cell_table(path: Path, cols, n=4, value=0.5):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"row_id": [f"r{i}" for i in range(n)], **{c: np.full(n, value) for c in cols}}).to_parquet(path)


# ------------------------------------------------------------------------------------ mirrors of the v1 guard tests
def test_second_claim_refused_marker_readonly(sandbox):
    LK.claim(LOOK, _freeze())
    assert LK.require_open_claim(_freeze()) == LOOK
    with pytest.raises(ProtectedDataError, match="already claimed"):
        LK.claim(LOOK, _freeze())
    assert oct(os.stat(LK.marker(LOOK)).st_mode & 0o777) == "0o444"


def test_registry_rows_refuse_claim(sandbox):
    sandbox["n"] = 2
    with pytest.raises(ProtectedDataError, match="registry already holds"):
        LK.claim(LOOK, _freeze())
    assert not LK.marker(LOOK).exists()


def test_rewritten_freeze_has_no_open_claim(sandbox):
    LK.claim(LOOK, _freeze("a" * 64))
    with pytest.raises(ProtectedDataError, match="open M2S/test claim"):
        LK.require_open_claim(_freeze("b" * 64))


def test_sealed_predictions_scored_marker(sandbox):
    LK.claim(LOOK, _freeze())
    d = LK.look_dir(LOOK) / "predictions"
    _cell_table(d / "cell_testxarchive_lake.parquet", _required())
    (d / "cells.json").write_text("{}")
    LK.seal_predictions(LOOK, d, freeze=_freeze(), required=_required())
    assert oct(os.stat(d / "cell_testxarchive_lake.parquet").st_mode & 0o777) == "0o444"
    with pytest.raises(ProtectedDataError, match="already sealed"):
        LK.seal_predictions(LOOK, d, freeze=_freeze(), required=_required())
    assert LK.check_sealed(LOOK) == d
    good = (d / "cell_testxarchive_lake.parquet").read_bytes()
    os.chmod(d / "cell_testxarchive_lake.parquet", 0o644)
    _cell_table(d / "cell_testxarchive_lake.parquet", _required(), value=0.7)
    with pytest.raises(ProtectedDataError, match="modified"):
        LK.check_sealed(LOOK)
    (d / "cell_testxarchive_lake.parquet").write_bytes(good)
    _cell_table(d / "cell_extra.parquet", _required())
    with pytest.raises(ProtectedDataError, match="unsealed files"):
        LK.check_sealed(LOOK)
    (d / "cell_extra.parquet").unlink()
    LK.mark_scored(LOOK, {"n": 1})
    with pytest.raises(ProtectedDataError, match="already scored"):
        LK.check_sealed(LOOK)
    assert LK.open_claims(_freeze()["_sha256"]) == []
    with pytest.raises(ProtectedDataError, match="open M2S/test claim"):
        LK.require_open_claim(_freeze())


def test_data_pins_detect_change(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "PROCESSED", tmp_path)
    monkeypatch.setattr(paths, "RUNS_ROOT", tmp_path / "runs")
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.parquet").write_bytes(b"1")
    (tmp_path / "runs" / "orbit" / "v2" / "pin").mkdir(parents=True)
    (tmp_path / "runs" / "orbit" / "v2" / "pin" / "pin.json").write_text("{}")
    monkeypatch.setattr(LK, "DATA_PIN_SPECS", {"a": ("a", ["*.parquet"]), "p": ("RUNS:orbit/v2/pin", ["pin.json"])})
    freeze = {"data_pins": LK.make_data_pins()}
    assert set(LK.verify_data_pins(freeze)) == {"a", "p"}
    (tmp_path / "a" / "y.parquet").write_bytes(b"2")          # an added file changes the pin
    with pytest.raises(ProtectedDataError, match="content changed"):
        LK.verify_data_pins(freeze)
    (tmp_path / "a" / "y.parquet").unlink()
    LK.verify_data_pins(freeze)
    (tmp_path / "runs" / "orbit" / "v2" / "pin" / "pin.json").write_text('{"x": 1}')     # a runs-root pin
    with pytest.raises(ProtectedDataError, match="p: content changed"):
        LK.verify_data_pins(freeze)
    with pytest.raises(ProtectedDataError, match="no pin"):
        LK.verify_data_pins({"data_pins": {}})
    narrowed = {"data_pins": {"a": LK.pin("a", ["x.parquet"]), "p": freeze["data_pins"]["p"]}}
    with pytest.raises(ProtectedDataError, match="patterns differ"):                     # a pin of fewer files
        LK.verify_data_pins(narrowed)
    monkeypatch.setattr(LK, "DATA_PIN_SPECS", {"e": ("a", ["*.none"])})
    with pytest.raises(ProtectedDataError, match="no file"):                             # an empty pinned set
        LK.verify_data_pins({"data_pins": LK.make_data_pins()})


def _freeze_files(tmp_path, monkeypatch, **over):
    """A complete, valid candidate + freeze pair on temporary files; returns (body of the freeze, paths)."""
    cand, frz, v1 = (tmp_path / "orbit_S_freeze_candidate.json", tmp_path / "FREEZE_orbit_S.json",
                     tmp_path / "FREEZE_orbit.json")
    monkeypatch.setattr(LK, "FREEZE_FILE", frz)
    monkeypatch.setattr(LK, "CANDIDATE_FILE", cand)
    monkeypatch.setattr(LK, "V1_FREEZE_FILE", v1)
    monkeypatch.setattr(paths, "MANIFESTS", tmp_path / "manifests")
    (tmp_path / "manifests").mkdir(exist_ok=True)
    man = tmp_path / "manifests" / "orbit_dev_vXm.json"
    man.write_text(json.dumps({"files": []}))
    v1.write_text(json.dumps({"weights_sha256": {"SENTINEL_s0": "w0", "SENTINEL_s1": "w1"}}))
    stamp = {"ok": True}

    def require_stamp(what, *a, **k):
        if not stamp["ok"]:
            raise LK.ST.StampError(f"{what}: refused before the lead's stamp")
        return {"stamped": True}

    lst = {"ok": True}

    def load_list(*a, **k):
        if not lst["ok"]:
            raise LK.M7.M7ListError("the M7 keys list differs from the sha256 of C-1")
        return pd.DataFrame()

    monkeypatch.setattr(LK.ST, "require_stamp", require_stamp)
    monkeypatch.setattr(LK.M7, "load_list", load_list)
    monkeypatch.setattr(LK, "stamped_sha256", lambda root=None: {"amendment": "s1", "rulings": "s2"})
    monkeypatch.setattr(LK, "scoring_code_sha256", lambda root=None: "code")
    body = {k: "x" for k in LK.FREEZE_KEYS}
    body.update(look=LOOK, split_sha256=LK.SP.split_sha256(), scoring_code_sha256="code", weights_sha256={"m": "s"},
                data_version="dev_vXm", data_manifest_sha256=hashlib.sha256(man.read_bytes()).hexdigest(),
                data_pins={"a": {"x": 1}}, stamped_sha256={"amendment": "s1", "rulings": "s2"},
                test_s_primary_cell="scene_disjoint", power_status={"small_lakes": "descriptive"},
                v1_weights_sha256={"SENTINEL_s0": "w0", "SENTINEL_s1": "w1"},
                v1_freeze_sha256=hashlib.sha256(v1.read_bytes()).hexdigest(), config_sha256={"a0": "h"},
                m7_keys_sha256=LK.M7.LIST_SHA256, sealed_models=dict(SEALED), status="CANDIDATE")
    body.update(over)
    cand.write_text(json.dumps(body))
    full = dict(body, status="FROZEN", authorised_by="lead", authorised="2026-11-10",
                candidate_sha256=hashlib.sha256(cand.read_bytes()).hexdigest())
    return full, {"cand": cand, "frz": frz, "v1": v1, "man": man, "stamp": stamp, "lst": lst}


def test_candidate_never_unlocks(tmp_path, monkeypatch):
    """Only FREEZE_orbit_S.json unlocks; a candidate path, the v1 freeze or a non-FROZEN status is refused."""
    full, f = _freeze_files(tmp_path, monkeypatch)
    with pytest.raises(ProtectedDataError, match="only .* unlocks"):
        LK.load_freeze(f["cand"])
    with pytest.raises(ProtectedDataError, match="only .* unlocks"):
        LK.load_freeze(f["v1"])                                   # the v1 freeze never unlocks Test-S
    with pytest.raises(ProtectedDataError, match="does not exist"):
        LK.load_freeze()
    f["frz"].write_text(f["cand"].read_text())                    # candidate copied without the lead's authorisation
    with pytest.raises(ProtectedDataError, match="status"):
        LK.load_freeze()
    f["frz"].write_text(json.dumps(dict(full, weights_sha256={"m": "t"})))     # a key differs from the candidate
    with pytest.raises(ProtectedDataError, match="differs from the reviewed candidate"):
        LK.load_freeze()
    f["frz"].write_text(json.dumps(dict(full, candidate_sha256="0" * 64)))
    with pytest.raises(ProtectedDataError, match="candidate_sha256"):
        LK.load_freeze()
    with pytest.raises(ProtectedDataError):                       # check_rows with allow_test also goes through it
        LK.check_rows(np.array([True]), allow_test=True)
    f["frz"].write_text(json.dumps(full))
    fr = LK.load_freeze()
    assert fr["_sha256"] == hashlib.sha256(f["frz"].read_bytes()).hexdigest()
    with pytest.raises(ProtectedDataError, match="open M2S/test claim"):      # a valid freeze alone opens nothing
        LK.check_rows(np.array([True]), allow_test=True)


def test_v1_and_v2_code_in_scoring_hash():
    files = LK.frozen_files()
    for f in LK.G1.frozen_files():                                # every file of the v1 hash
        assert f in files
    for f in ("sentinel/models/orbit_v2/looks.py", "sentinel/models/orbit_v2/m7keys.py",
              "sentinel/models/orbit_v2/targets.py", "sentinel/models/orbit_v2/context.py",
              "sentinel/models/orbit_v2/devm.py", "sentinel/models/orbit_v2/stamp.py",
              "sentinel/models/orbit_v2/comparators.py", "scripts/train/orbitv2_power.py",
              "scripts/data/orbitv2_build_dev_v5m.py", LK.ST.AMENDMENT, LK.ST.RULINGS):
        assert f in files, f
    assert any(f.startswith("configs/orbit_v2/") for f in files)


def test_access_closed_at_sealing_and_seal_row(sandbox, monkeypatch):
    rows = []
    monkeypatch.setattr(LK, "_log_result", lambda **k: rows.append(k))
    LK.claim(LOOK, _freeze())
    d = LK.look_dir(LOOK) / "predictions"
    _cell_table(d / "cell_testxarchive_lake.parquet", _required())
    LK.seal_predictions(LOOK, d, freeze=_freeze(), required=_required())
    assert [r["metric"] for r in rows] == ["look_claimed", "predictions_sealed"]
    assert {r["experiment"] for r in rows} == {"orbit_TEST_S_dev_vXm"} and {r["workstream"] for r in rows} == {"M2-orbit"}
    assert LK.open_claims(_freeze()["_sha256"]) == []           # protected rows can no longer be loaded
    with pytest.raises(ProtectedDataError, match="already sealed"):
        LK.require_open_claim(_freeze())


def test_registry_control_rows_close_access_without_markers(sandbox, monkeypatch):
    """Deleting a marker cannot re-open access: the append-only registry rows close it too."""
    LK.claim(LOOK, _freeze())
    assert LK.open_claims(_freeze()["_sha256"]) == [LOOK]
    monkeypatch.setattr(LK, "control_row", lambda exp, metric: metric == "predictions_sealed")
    assert LK.open_claims(_freeze()["_sha256"]) == []
    monkeypatch.setattr(LK, "control_row", lambda exp, metric: metric == "look_aborted")
    assert LK.open_claims(_freeze()["_sha256"]) == []


def test_rescore_refused_once_scores_exist(sandbox, monkeypatch):
    seen = {"n": 0}
    monkeypatch.setattr(LK, "scoring_rows", lambda exp: seen["n"])
    LK.require_no_scores(LOOK, _freeze())                       # crash between seal and scores: allowed
    seen["n"] = 5
    with pytest.raises(ProtectedDataError, match="scoring rows"):
        LK.require_no_scores(LOOK, _freeze())


def test_import_closure_hashed():
    """Import every orbit_v2 module and load every orbitv2 script in a fresh interpreter, then require every loaded
    sentinel module to be in the M2S scoring-code hash (infrastructure under sentinel/core excepted, as in v1)."""
    import subprocess
    import sys

    code = r'''
import importlib, importlib.util, json, sys
from pathlib import Path
root = Path.cwd()
for f in sorted((root / "sentinel/models/orbit_v2").glob("*.py")):
    if f.stem != "__init__":
        importlib.import_module(f"sentinel.models.orbit_v2.{f.stem}")
for g in ("scripts/train/orbitv2_*.py", "scripts/eval/orbitv2_*.py", "scripts/data/orbitv2_*.py"):
    for f in sorted(root.glob(g)):
        if str(f.relative_to(root)) in SKIP:
            continue
        spec = importlib.util.spec_from_file_location("closure_" + f.stem, f)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
print(json.dumps(sorted(m.__file__ for n, m in sys.modules.items()
                        if n.startswith("sentinel") and getattr(m, "__file__", None))))
'''
    root = paths.REPO_ROOT
    env = {**os.environ, "SENTINEL_RUNS": str(paths.RUNS_ROOT)}
    assert not _under_real_runs_root(env["SENTINEL_RUNS"])
    for f in LK.RUNS_AT_IMPORT:                    # hashed, but a stand-alone script without a main guard: not loaded
        assert f in LK.frozen_files() and "__main__" not in (root / f).read_text()
    out = subprocess.run([sys.executable, "-B", "-c", f"SKIP = {list(LK.RUNS_AT_IMPORT)!r}\n" + code], cwd=root,
                         capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr[-2000:]
    loaded = {str(Path(f).resolve().relative_to(root.resolve())) for f in json.loads(out.stdout.splitlines()[-1])}
    loaded = {f for f in loaded if not f.startswith(LK.NOT_IN_CLOSURE)}
    missing = sorted(loaded - set(LK.frozen_files()))
    assert not missing, f"modules loaded by the Test-S path outside the M2S scoring-code hash: {missing}"
    assert "sentinel/models/orbit/testlook.py" in loaded and "sentinel/models/orbit_v2/looks.py" in loaded


def test_claim_bound_to_process_and_abort_closes(sandbox, monkeypatch):
    LK.claim(LOOK, _freeze())
    rec = json.loads(LK.marker(LOOK).read_text())
    assert rec["pid"] == os.getpid()
    with monkeypatch.context() as other:                               # another process: no access
        other.setattr(LK.os, "getpid", lambda: rec["pid"] + 1)
        assert LK.open_claims(_freeze()["_sha256"]) == []
    with monkeypatch.context() as other:                               # another host: no access
        other.setattr(LK.socket, "gethostname", lambda: "elsewhere")
        assert LK.open_claims(_freeze()["_sha256"]) == []
    assert LK.open_claims(_freeze()["_sha256"]) == [LOOK]              # the claiming process: access open
    LK.mark_aborted(LOOK, _freeze(), "KeyboardInterrupt()")
    aborted = LK.look_dir(LOOK) / "ABORTED.json"
    assert aborted.exists() and not _under_real_runs_root(aborted)
    assert LK.open_claims(_freeze()["_sha256"]) == []


def test_no_write_under_real_runs_root_even_after_undo(monkeypatch):
    """A test-level ``monkeypatch.undo()`` falls back to the module's temporary root, and the marker writer refuses
    a real path before anything is created (R-24: no test may write under the real runs root)."""
    written = []
    monkeypatch.setitem(_REAL, "write", lambda p, rec: written.append(str(p)))
    monkeypatch.undo()
    assert not _under_real_runs_root(paths.RUNS_ROOT)
    assert not _under_real_runs_root(LK.look_dir(LOOK))
    assert LK._excl_write is _guarded_write
    monkeypatch.setitem(_REAL, "write", lambda p, rec: written.append(str(p)))
    probe = REAL_LOOKS / "M2S_test" / "PROBE_never_written.json"
    with pytest.raises(AssertionError, match="under the real runs root"):
        LK._excl_write(probe, {})
    with pytest.raises(AssertionError, match="real registry row"):
        LK.mark_aborted(LOOK, _freeze(), "probe")
    assert written and not any(_under_real_runs_root(w) for w in written)
    assert not probe.exists() and not probe.parent.exists() or not probe.exists()


def test_real_test_looks_tree_is_snapshotted():
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "test_looks"
        (root / "M2S_test").mkdir(parents=True)
        a = _tree(root)
        (root / "M2S_test" / "ABORTED.json").write_text("{}")
        b = _tree(root)
        assert a != b and "M2S_test/ABORTED.json" in b
        os.utime(root / "M2S_test" / "ABORTED.json", ns=(1, 1))
        assert _tree(root) != b
    assert _tree(Path(d) / "missing") == {}


# ------------------------------------------------------------------------------------ what R-17, R-21, R-24 add
def test_claim_refuses_a_non_empty_look_folder(sandbox):
    """R-24: any file or folder in the look folder stops the claim; nothing is written."""
    d = LK.look_dir(LOOK)
    d.mkdir(parents=True)
    LK.claim(LOOK, _freeze(), register=False)                   # an empty folder is fine
    os.chmod(LK.marker(LOOK), 0o644)
    LK.marker(LOOK).unlink()
    for stray in ("ABORTED.json", "notes.txt", "predictions/x.parquet", "empty_folder/"):
        q = d / stray
        if stray.endswith("/"):
            q.mkdir(parents=True)
        else:
            q.parent.mkdir(parents=True, exist_ok=True)
            q.write_text("{}")
        with pytest.raises(ProtectedDataError, match="not empty"):
            LK.claim(LOOK, _freeze())
        assert not LK.marker(LOOK).exists()
        if q.is_dir():
            q.rmdir()
        else:
            q.unlink()
            if q.parent != d:
                q.parent.rmdir()
    LK.claim(LOOK, _freeze())                                   # empty again: the claim starts
    assert LK.folder_entries(LOOK) == ["TEST_LOOK.json"]


def test_only_the_m2s_look_in_its_own_folder_and_experiment(sandbox):
    assert LK.LOOKS == ("M2S/test",)
    assert LK.look_dir(LOOK) == paths.RUNS_ROOT / "orbit" / "test_looks" / "M2S_test"
    assert LK.experiment(LOOK, "dev_v5m") == "orbit_TEST_S_dev_v5m"
    assert LK.FREEZE_FILE.name == "FREEZE_orbit_S.json" and LK.CANDIDATE_FILE.name == "orbit_S_freeze_candidate.json"
    assert LK.FREEZE_FILE != LK.V1_FREEZE_FILE
    for v1_look in TL1.LOOKS:
        with pytest.raises(ProtectedDataError, match="unknown test look"):
            LK.look_dir(v1_look)
        with pytest.raises(ProtectedDataError, match="unknown test look"):
            LK.claim(v1_look, _freeze())
        assert LK.experiment(LOOK, "dev_v5") != TL1.experiment(v1_look, "dev_v5")
    # a v1 claim in the same root opens nothing here, and this look's claim opens nothing in v1
    v1_marker = paths.RUNS_ROOT / "orbit" / "test_looks" / "M2_test" / "TEST_LOOK.json"
    _guarded_write(v1_marker, {"look": "M2/test", "freeze_sha256": "a" * 64, "pid": os.getpid(),
                               "host": LK.socket.gethostname()})
    assert LK.open_claims("a" * 64) == []
    LK.claim(LOOK, _freeze("a" * 64))
    assert LK.open_claims("a" * 64) == [LOOK]


def test_look_specific_parts_are_not_the_v1_functions():
    """R-21: look-agnostic helpers are imported from v1; the look-specific parts are this module's own; v1 is not
    patched."""
    assert _REAL["write"] is TL1._excl_write and LK._sha_file is TL1._sha_file and LK.content_pin is TL1.content_pin
    for name in ("look_dir", "marker", "experiment", "claim", "open_claims", "require_open_claim", "mark_aborted",
                 "seal_predictions", "check_sealed", "mark_scored", "verify_data_pins", "make_data_pins",
                 "require_no_scores", "registry_rows"):
        assert getattr(LK, name).__module__ == "sentinel.models.orbit_v2.looks", name
    assert LK.load_freeze.__module__ == LK.check_rows.__module__ == "sentinel.models.orbit_v2.looks"
    assert tuple(TL1.LOOKS) == V1_STATE["looks"] == ("M2/test", "M2U/test")
    assert TL1._excl_write is V1_STATE["writer"] and TL1.claim is V1_STATE["claim"]
    src = Path(LK.__file__).read_text()
    assert "setattr(" not in src and "monkeypatch" not in src and "importlib" not in src
    import ast

    v1_imports = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ImportFrom) and "testlook" in (node.module or ""):
            v1_imports.append((node.module, sorted(a.name for a in node.names)))
        if isinstance(node, (ast.Import, ast.ImportFrom)):               # no handle on the v1 module itself
            assert not any(a.name.split(".")[-1] == "testlook" for a in node.names)
    assert v1_imports == [("orbit.testlook", ["_excl_write", "_sha_file", "content_pin"])]


def test_pins_cover_every_file_the_protected_loader_reads():
    """R-21: wqp_discrete/variable=*, both context chip stores, processed/orbit_wqp_surface; plus the context store
    pin and the C-1 keys list."""
    s = LK.DATA_PIN_SPECS
    assert s["wqp_discrete"][0] == "wqp_discrete" and "variable=*/data.parquet" in s["wqp_discrete"][1]
    assert s["wqp_surface"] == ("orbit_wqp_surface", ["surface_samples.parquet", "surface_samples.meta.json"])
    for store in LK.TR.__name__ and ("wqp", "aquacorr"):
        d, pats = s[f"chips_{store}"]
        assert d == f"s2_chips/{store}" and {"index.parquet", "index/*", "shards/*", "layers.json"} <= set(pats)
    from sentinel.models.orbit_v2 import context as CX
    from sentinel.models.orbit_v2 import m7keys as M7
    from sentinel.models.orbit_v2 import targets as T

    assert set(CX.STORES) == {"wqp", "aquacorr"}
    assert CX.pin_dir() == paths.RUNS_ROOT / s["context_store_pin"][0][5:]
    assert set(s["context_store_pin"][1]) == {CX.PIN_CHIPS, CX.PIN_INFO}
    assert M7.LIST_PATH.name in s["m7_keys_list"][1] and M7.LIST_PATH.parent.name == Path(s["m7_keys_list"][0]).name
    for t in T.SPECS:                                           # every new-target table matches the pinned pattern
        rel = T.source_file(t).relative_to(paths.PROCESSED / "wqp_discrete")
        assert rel.match("variable=*/data.parquet")
    for name, (d, pats) in TL1.DATA_PIN_SPECS.items():          # nothing the v1 loader reads is left out
        assert s[name] == (d, pats), name


def test_seal_needs_unit_network_and_comparator_columns(sandbox):
    """R-17: the reported model, 3 seed units, 9 single networks, every frozen v1 network and every claim
    comparator are sealed before scoring."""
    req = _required()
    assert req[:3] == ["v1", "v1@s0", "v1@s1"]
    assert [c for c in req if c.startswith("v2")] == (
        ["v2", "v2@unit0", "v2@unit1", "v2@unit2"] + [f"v2@u{u}i{i}" for u in range(3) for i in range(3)])
    assert req[-4:] == ["cmp:mdn_b", "cmp:gbm", "cmp:knn", "cmp:mdn_b+gbm"]
    assert len(LK.sealed_columns(dict(SEALED, track_n=True))) == len(req) + 13
    for bad in ({}, dict(SEALED, v1_networks=0), dict(SEALED, v2=False), dict(SEALED, comparators=[])):
        with pytest.raises(ProtectedDataError, match="sealed_models"):
            LK.sealed_columns(bad)
    LK.claim(LOOK, _freeze())
    d = LK.look_dir(LOOK) / "predictions"
    for drop in ("v2@unit1", "v2@u2i2", "v1@s1", "cmp:mdn_b+gbm"):
        _cell_table(d / "cell_testxarchive_lake.parquet", [c for c in req if c != drop])
        with pytest.raises(ProtectedDataError, match="R-17"):
            LK.seal_predictions(LOOK, d, freeze=_freeze(), required=req)
        (d / "cell_testxarchive_lake.parquet").unlink()
    _cell_table(d / "cell_testxarchive_lake.parquet", req)
    for weak in (None, [], ["v1", "v2"], [c for c in req if "@unit" not in c], [c for c in req if "@u0i" not in c],
                 [c for c in req if not c.startswith("cmp:")]):
        with pytest.raises(ProtectedDataError, match="R-17"):
            LK.seal_predictions(LOOK, d, freeze=_freeze(), required=weak)
    (d / "notes.json").write_text("{}")
    os.remove(d / "cell_testxarchive_lake.parquet")
    with pytest.raises(ProtectedDataError, match="no cell_"):
        LK.seal_predictions(LOOK, d, freeze=_freeze(), required=req)
    assert not (LK.look_dir(LOOK) / "PREDICTIONS_SAVED.json").exists() and not (d / "SHA256SUMS").exists()
    assert LK.open_claims(_freeze()["_sha256"]) == [LOOK]       # a refused seal leaves the claim open, unsealed


def _keys(**cols):
    n = len(next(iter(cols.values())))
    base = {"role": ["test"] * n, "date": ["2020-07-01"] * n, "in_lake": [True] * n, "buffer_excluded": [False] * n,
            "protected": [True] * n}
    return pd.DataFrame({**base, **cols})


def test_test_s_cells_from_keys_and_nothing_else_protected():
    df = _keys(role=["test", "test", "test", "test", "test", "val", "train", "test"],
               date=["2015-06-22", "2015-06-23", "2023-12-31", "2024-01-01", "2021-05-05", "2021-05-05", "2019-01-01",
                     "2022-08-08"],
               in_lake=[True, True, False, True, True, True, True, True],
               buffer_excluded=[False] * 7 + [True])
    assert LK.test_s_cell(df).tolist() == ["", "testxarchive_lake", "testxarchive_nonlake", "", "testxarchive_lake",
                                           "", "", ""]
    assert (LK.ARCHIVE_DATE_MIN, LK.ARCHIVE_DATE_MAX) == ("2015-06-23", "2023-12-31")
    LK.require_test_s_only(df.iloc[[1, 2, 4]])
    with pytest.raises(ProtectedDataError, match="1 protected rows are not Test-S rows"):
        LK.require_test_s_only(df.iloc[[1, 3]])                 # a headline row (test role, 2024) never enters
    dev = df.iloc[[1, 5]].assign(protected=[True, False])
    with pytest.raises(ProtectedDataError, match="1 rows are not Test-S rows"):
        LK.require_test_s_only(dev)
    with pytest.raises(ProtectedDataError, match="training code never loads"):
        LK.check_rows(np.array([True, False]), training=True, allow_test=True)
    with pytest.raises(ProtectedDataError, match="--allow-test"):
        LK.check_rows(np.array([True]))
    assert LK.check_rows(np.array([False, False]))["n_protected"] == 0


def test_c1_rows_leave_every_cell():
    from sentinel.models.orbit_v2 import m7keys as M7

    lst = pd.DataFrame({"m2_waterbody": ["L1", "L9"], "site_id": ["S5", "S7"],
                        "_s2_day": M7._days(pd.Series(["2020-07-01", "2021-01-01"])),
                        "_sample_day": M7._days(pd.Series(["2020-07-03", "2019-03-03"]))})
    df = pd.DataFrame({"waterbody": ["L1", "L1", "L2", "L3"], "sensing_date": ["2020-07-02", "2020-07-09", "2020-07-01",
                                                                             "2019-03-03"],
                       "site_id": ["S1", "S1", "S2", "S7"], "date": ["2020-07-02", "2020-07-09", "2020-07-01",
                                                                    "2019-03-03"], "x": [1, 2, 3, 4]})
    out, rec = LK.apply_c1(df, lst)
    assert out.x.tolist() == [2, 3] and rec["n_matched"] == 2
    with pytest.raises(KeyError):
        LK.apply_c1(df.drop(columns="sensing_date"), lst)


def test_freeze_checks_stamp_v1_weights_keys_list_manifest_and_code(tmp_path, monkeypatch):
    full, f = _freeze_files(tmp_path, monkeypatch)
    f["frz"].write_text(json.dumps(full))
    LK.load_freeze()
    f["stamp"]["ok"] = False                                    # R-23: no valid stamp row
    with pytest.raises(ProtectedDataError, match="stamp"):
        LK.load_freeze()
    f["stamp"]["ok"] = True
    monkeypatch.setattr(LK, "stamped_sha256", lambda root=None: {"amendment": "s1", "rulings": "CHANGED"})
    with pytest.raises(ProtectedDataError, match="stamped_sha256"):
        LK.load_freeze()
    monkeypatch.setattr(LK, "stamped_sha256", lambda root=None: {"amendment": "s1", "rulings": "s2"})
    v1_text = f["v1"].read_text()
    f["v1"].write_text(json.dumps({"weights_sha256": {"SENTINEL_s0": "w0", "SENTINEL_s1": "OTHER"}}))
    with pytest.raises(ProtectedDataError, match="v1_freeze_sha256"):
        LK.load_freeze()
    f["v1"].unlink()
    with pytest.raises(ProtectedDataError, match="v1_freeze_sha256"):      # no lead-written v1 freeze: no Test-S look
        LK.load_freeze()
    f["v1"].write_text(v1_text)
    f["lst"]["ok"] = False                                      # C-1: the keys list changed or is missing
    with pytest.raises(ProtectedDataError, match="C-1"):
        LK.load_freeze()
    f["lst"]["ok"] = True
    man_text = f["man"].read_text()
    f["man"].write_text(json.dumps({"files": [], "x": 1}))
    with pytest.raises(ProtectedDataError, match="data manifest"):
        LK.load_freeze()
    f["man"].write_text(man_text)
    monkeypatch.setattr(LK, "scoring_code_sha256", lambda root=None: "edited")
    with pytest.raises(ProtectedDataError, match="scoring code changed"):
        LK.load_freeze()
    assert LK.load_freeze(check_code=False)["look"] == LOOK


@pytest.mark.parametrize("over, msg", [
    ({"look": "M2/test"}, "freeze look"),
    ({"test_s_primary_cell": "largest"}, "test_s_primary_cell"),
    ({"v1_weights_sha256": {"SENTINEL_s0": "w0"}}, "v1_weights_sha256 differs"),
    ({"m7_keys_sha256": "0" * 64}, "m7_keys_sha256"),
    ({"split_sha256": "0" * 64}, "split_sha256"),
    ({"sealed_models": ["v1"]}, "must be a JSON object"),
    ({"power_status": {}}, "lacks required keys"),
])
def test_freeze_field_rules(tmp_path, monkeypatch, over, msg):
    full, f = _freeze_files(tmp_path, monkeypatch, **over)
    f["frz"].write_text(json.dumps(full))
    with pytest.raises(ProtectedDataError, match=msg):
        LK.load_freeze()
