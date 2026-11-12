# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""M2 Orbit single-look enforcement (orbit_protocol.md §9b). No real data, no real registry writes, and no write
under the real runs root: every test runs on a temporary ``paths.RUNS_ROOT`` and the real
``sentinel_runs/orbit/test_looks`` must be exactly as it was before (fixtures below)."""

import hashlib
import json
import os
import stat
from pathlib import Path

import numpy as np
import pytest

from sentinel.core import paths
from sentinel.models.orbit import guard as G
from sentinel.models.orbit import testlook as TL
from sentinel.models.sonde.guard import ProtectedDataError


REAL_RUNS_ROOT = Path(paths.RUNS_ROOT).resolve()        # captured at import, before any fixture changes it
REAL_LOOKS = REAL_RUNS_ROOT / "orbit" / TL.ROOT
_REAL = {"write": TL._excl_write}                       # the unguarded marker writer


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
    """Base layer for the whole module, installed with its own MonkeyPatch (not the per-test one): the runs root is
    a temporary directory, the marker writer refuses any path under the real runs root, and registry writes are
    refused. A test-level ``monkeypatch.undo()`` therefore falls back to this layer and never to the real root.
    (That is how ``test_claim_bound_to_process_and_abort_closes`` wrote ``ABORTED.json`` into the real
    ``orbit/test_looks/M2_test`` on 2026-11-09.) On teardown the real ``test_looks`` tree must be unchanged."""
    import sentinel.core.registry as R

    before = _tree(REAL_LOOKS)
    mp = pytest.MonkeyPatch()
    mp.setattr(paths, "RUNS_ROOT", tmp_path_factory.mktemp("orbit_runs_base"))
    mp.setattr(TL, "_excl_write", _guarded_write)
    mp.setattr(R, "log_result", _no_registry)
    yield
    mp.undo()
    assert _tree(REAL_LOOKS) == before, f"the real {REAL_LOOKS} changed while this test module ran"


@pytest.fixture(autouse=True)
def _real_test_looks_untouched(tmp_path, monkeypatch):
    """Every test: its own temporary runs root; afterwards the runs root must still be temporary and the real
    ``sentinel_runs/orbit/test_looks`` tree must be exactly as before the test."""
    before = _tree(REAL_LOOKS)
    monkeypatch.setattr(paths, "RUNS_ROOT", tmp_path / "runs")
    yield
    assert not _under_real_runs_root(paths.RUNS_ROOT), "paths.RUNS_ROOT points at the real runs root inside a test"
    assert TL._excl_write is _guarded_write, "the marker-writer guard was removed inside a test"
    assert _tree(REAL_LOOKS) == before, f"the real {REAL_LOOKS} changed during this test"


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
    TL.seal_predictions("M2/test", d, freeze=_freeze())
    assert oct(os.stat(d / "cell_headline.parquet").st_mode & 0o777) == "0o444"
    with pytest.raises(ProtectedDataError, match="already sealed"):
        TL.seal_predictions("M2/test", d, freeze=_freeze())
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


def test_access_closed_at_sealing_and_seal_row(sandbox, monkeypatch):
    rows = []
    import sentinel.core.registry as R

    monkeypatch.setattr(R, "log_result", lambda **k: rows.append(k))
    TL.claim("M2/test", _freeze())
    d = TL.look_dir("M2/test") / "predictions"
    d.mkdir(parents=True)
    (d / "cell_headline.parquet").write_bytes(b"one")
    TL.seal_predictions("M2/test", d, freeze=_freeze())
    assert [r["metric"] for r in rows] == ["look_claimed", "predictions_sealed"]
    assert TL.open_claims(_freeze()["_sha256"]) == []           # protected rows can no longer be loaded
    with pytest.raises(ProtectedDataError, match="already sealed"):
        TL.require_open_claim(_freeze())          # message: "...or its predictions are already sealed"


def test_rescore_refused_once_scores_exist(sandbox, monkeypatch):
    seen = {"n": 0}
    monkeypatch.setattr(TL, "scoring_rows", lambda exp: seen["n"])
    TL.require_no_scores("M2/test", _freeze())                  # crash between seal and scores: allowed
    seen["n"] = 5
    with pytest.raises(ProtectedDataError, match="scoring rows"):
        TL.require_no_scores("M2/test", _freeze())


def test_event_import_closure_hashed():
    """Run the event path in a fresh interpreter, then require every loaded sentinel module to be in the orbit
    scoring-code hash (hazards.py sets the event thresholds; M7 review 2026-11-09)."""
    import subprocess
    import sys

    code = r'''
import json, sys
import numpy as np
from sentinel.events import orbit as EO
EV = EO.EV()
rng = np.random.default_rng(0)
y_lin = 10 ** rng.normal(1.1, 0.5, 400)
for code in EO.E4A_CODES:
    y = EO.labels(y_lin, code)
    w = np.full((400, 2), 0.5); mu = np.log10(y_lin)[:, None] + rng.normal(0, 0.2, (400, 2)); sg = np.full((400, 2), 0.2)
    p = EO.mixture_p(w, mu, sg, code); q = EO.gaussian_p(mu[:, 0], 0.3, code)
    cal = EV.recalibrate(p, y); pc = cal(p)
    g = np.arange(400) % 40
    EV.state_metrics(y, pc, groups=g, p_ref=np.full(400, y.mean()), n_boot=20)
    EV.compare_routes(y, pc, q, groups=g, n_boot=20)
for code, t in EO.E3_DEFS:
    EO.labels(y_lin * 5, code)
EO.spectral_indices.__call__
print(json.dumps(sorted(m.__file__ for n, m in sys.modules.items()
                        if n.startswith("sentinel") and getattr(m, "__file__", None))))
'''
    root = paths.REPO_ROOT
    env = {**os.environ, "SENTINEL_RUNS": str(paths.RUNS_ROOT)}      # the fresh interpreter gets the temporary root too
    assert not _under_real_runs_root(env["SENTINEL_RUNS"])
    out = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True, check=True, env=env)
    loaded = {str(__import__("pathlib").Path(f).resolve().relative_to(root.resolve())) for f in json.loads(out.stdout)}
    missing = sorted(loaded - set(G.frozen_files()))
    assert not missing, f"event-path modules outside the orbit scoring-code hash: {missing}"
    assert "sentinel/eval/hazards.py" in loaded


def test_claim_bound_to_process_and_abort_closes(sandbox, monkeypatch):
    TL.claim("M2/test", _freeze())
    rec = json.loads(TL.marker("M2/test").read_text())
    assert rec["pid"] == os.getpid()
    with monkeypatch.context() as other:                               # another process: no access
        other.setattr(TL.os, "getpid", lambda: rec["pid"] + 1)
        assert TL.open_claims(_freeze()["_sha256"]) == []
    # only the pid patch is undone here; the temporary runs root and the registry stub stay in place
    assert TL.open_claims(_freeze()["_sha256"]) == ["M2/test"]         # the claiming process: access open
    TL.mark_aborted("M2/test", _freeze(), "KeyboardInterrupt()")
    aborted = TL.look_dir("M2/test") / "ABORTED.json"
    assert aborted.exists() and not _under_real_runs_root(aborted)
    assert TL.open_claims(_freeze()["_sha256"]) == []


def test_no_write_under_real_runs_root_even_after_undo(monkeypatch):
    """Regression (2026-11-09): a test-level ``monkeypatch.undo()`` restored the real runs root and ``mark_aborted``
    wrote ``ABORTED.json`` there. Now the undo falls back to the module's temporary root, and the marker writer
    refuses a real path before anything is created. The unguarded writer is replaced by a recorder here, so this
    test cannot write to the real root even if the guard failed."""
    written = []
    monkeypatch.setitem(_REAL, "write", lambda p, rec: written.append(str(p)))
    monkeypatch.undo()                                   # the old failure mode (also restores _REAL["write"])
    assert not _under_real_runs_root(paths.RUNS_ROOT)
    assert not _under_real_runs_root(TL.look_dir("M2/test"))
    assert TL._excl_write is _guarded_write
    monkeypatch.setitem(_REAL, "write", lambda p, rec: written.append(str(p)))
    probe = REAL_LOOKS / "M2_test" / "PROBE_never_written.json"
    with pytest.raises(AssertionError, match="under the real runs root"):
        TL._excl_write(probe, {})
    with pytest.raises(AssertionError, match="real registry row"):
        TL.mark_aborted("M2/test", _freeze(), "probe")   # temporary root: marker goes to the recorder; registry refused
    assert written and not any(_under_real_runs_root(w) for w in written)
    assert not probe.exists()


def test_real_test_looks_tree_is_snapshotted():
    """The untouched-check compares names, sizes, mtimes and modes of the real ``orbit/test_looks`` (no file is
    opened). It must see a change: the same function on a temporary tree detects an added and a rewritten file."""
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "test_looks"
        (root / "M2_test").mkdir(parents=True)
        a = _tree(root)
        (root / "M2_test" / "ABORTED.json").write_text("{}")
        b = _tree(root)
        assert a != b and "M2_test/ABORTED.json" in b
        os.utime(root / "M2_test" / "ABORTED.json", ns=(1, 1))
        assert _tree(root) != b
    assert _tree(Path(d) / "missing") == {}


def test_conformal_cqr_waterbody_weighted():
    """§5b: offsets widen overconfident intervals to nominal coverage; water bodies, not rows, are weighted."""
    from sentinel.models.orbit import conformal as CF

    rng = np.random.default_rng(0)
    n = 4000
    g = rng.integers(0, 400, n)
    y = rng.normal(0, 1, n)
    Q = {0.025: np.full(n, -1.0), 0.16: np.full(n, -0.5), 0.5: np.zeros(n), 0.84: np.full(n, 0.5),
         0.975: np.full(n, 1.0)}                                    # too narrow for N(0, 1)
    off = CF.fit(y, Q, g)
    assert off["68"] > 0 and off["95"] > 0
    iv = CF.crossfit_intervals(y, Q, g)
    assert abs(CF.coverage(y, *iv["95"]) - 0.95) < 0.02
    assert abs(CF.coverage(y, *iv["68"]) - 0.68) < 0.03
    # one huge water body with many rows does not dominate the weighted quantile
    s = np.r_[np.zeros(1000), np.ones(10)]
    gg = np.r_[np.zeros(1000, int), np.arange(1, 11)]
    assert CF.wb_quantile(s, gg, 0.05) == 1.0
