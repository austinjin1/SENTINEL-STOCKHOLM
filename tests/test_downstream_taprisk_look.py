# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""D4 hardened look: each attack of the 2026-11-09 pre-test review must be refused (synthetic fixtures, temp registry)."""
import json
import os

import pandas as pd
import pytest

import sentinel.core.registry as R
from sentinel.models.downstream import common as C
from sentinel.models.downstream import taprisk_look as LK
from sentinel.models.downstream import taprisk_registry as REG
from sentinel.models.downstream import testlook as TL

_REAL_VERIFY_FOLDERS = LK.verify_pinned_folders

SPLIT = "a4a1fc7bb5f997d0a656c527c884ed4691c59da341eff81d0d424bbd3e5b3298"


@pytest.fixture
def env(tmp_path, monkeypatch):
    rows = []

    def log_result(**kw):
        rows.append({**kw, "id": f"r{len(rows)}"})
        return rows[-1]

    monkeypatch.setattr(R, "log_result", log_result)
    monkeypatch.setattr(C, "freeze_file", lambda app: tmp_path / f"FREEZE_downstream_{app}.json")
    monkeypatch.setattr(LK, "candidate_path", lambda: tmp_path / "FREEZE_CANDIDATE_downstream_taprisk.json")
    monkeypatch.setattr(TL, "look_dir", lambda app: tmp_path / app / "test_look")
    monkeypatch.setattr(LK, "_rows", lambda: [r for r in rows if str(r["experiment"]).startswith("taprisk/")])
    monkeypatch.setattr(TL, "registry_test_rows", lambda app: sum(1 for r in rows if str(r["experiment"]).startswith(app)
                                                                  and str(r["split"]).startswith("test")))
    monkeypatch.setattr(TL, "verify_data", lambda app: {})          # real pins are exercised by their own test
    monkeypatch.setattr(LK, "verify_pinned_folders", lambda app="taprisk": None)
    monkeypatch.setattr(LK, "env_pin", lambda: "envpin")
    monkeypatch.setattr(C, "check_roles", lambda *a, **k: None)
    cand = {"status": "CANDIDATE — not frozen", "split_sha256": SPLIT, "scoring_code_sha256": C.scoring_code_sha256("taprisk"),
            "registry_scoring_code_sha256": REG.scoring_code_sha256(),
            "paths": {"runs_root": str(LK.paths.RUNS_ROOT.resolve()), "data_root": str(LK.paths.DATA_ROOT.resolve())},
            "environment": {"pip_freeze_noeditable_sha256": "envpin"}}
    (tmp_path / "FREEZE_CANDIDATE_downstream_taprisk.json").write_text(json.dumps(cand))
    return tmp_path, rows, cand


def _good_freeze(tmp, cand):
    fr = {**cand, "status": "FROZEN — lead", "authorised_by": "lead", "authorised_at": "2026-11-10", "frozen": "2026-11-10",
          "candidate_record_sha256": TL.file_sha(tmp / "FREEZE_CANDIDATE_downstream_taprisk.json")}
    C.freeze_file("taprisk").write_text(json.dumps(fr))
    return fr


def test_p1_candidate_copy_refused(env):
    tmp, rows, cand = env
    C.freeze_file("taprisk").write_text((tmp / "FREEZE_CANDIDATE_downstream_taprisk.json").read_text())
    with pytest.raises(C.ProtectedDataError):
        LK.load_freeze()
    bad = {**cand, "status": "FROZEN", "authorised_by": "x", "authorised_at": "y", "frozen": "z", "candidate_record_sha256": "0" * 64}
    C.freeze_file("taprisk").write_text(json.dumps(bad))
    with pytest.raises(C.ProtectedDataError, match="candidate"):
        LK.load_freeze()
    _good_freeze(tmp, cand)
    LK.load_freeze()


def _claim_and_seal(tmp, cand):
    _good_freeze(tmp, cand)
    LK.claim(SPLIT)
    LK.require_open(SPLIT)                                   # open inside the claiming process
    LK.seal(SPLIT, pd.DataFrame({"y": [0, 1], "s": [0.1, 0.9]}))


def test_p4_access_closed_at_sealing_and_after_scoring(env):
    tmp, rows, cand = env
    _claim_and_seal(tmp, cand)
    with pytest.raises(C.ProtectedDataError):
        LK.require_open(SPLIT)
    LK.mark_scored({"claim_met": False})
    with pytest.raises(C.ProtectedDataError):
        LK.require_open(SPLIT)


def test_p5_no_rescore_once_scoring_rows_exist(env):
    tmp, rows, cand = env
    _claim_and_seal(tmp, cand)
    rows.append({"experiment": "taprisk/test_look", "split": "test_known_d4a_gw_nitrate", "extra": {}})
    with pytest.raises(C.ProtectedDataError, match="scoring rows"):
        LK.rescore_load(SPLIT)


def test_rescore_allowed_only_from_registry_seal(env):
    tmp, rows, cand = env
    _claim_and_seal(tmp, cand)
    assert len(LK.rescore_load(SPLIT)) == 2


def test_p7c_tampered_predictions_and_marker_refused(env):
    tmp, rows, cand = env
    _claim_and_seal(tmp, cand)
    d = TL.look_dir("taprisk")
    p = d / "predictions.parquet"
    os.chmod(p, 0o644)
    pd.DataFrame({"y": [0, 1], "s": [0.0, 1.0]}).to_parquet(p, index=False)
    m = d / "PREDICTIONS_SAVED.json"
    os.chmod(m, 0o644)
    rec = json.loads(m.read_text())
    rec["sha256"] = TL.file_sha(p)
    m.write_text(json.dumps(rec))
    with pytest.raises(C.ProtectedDataError, match="registry seal"):
        LK.rescore_load(SPLIT)


def test_unlisted_file_refused(env):
    tmp, rows, cand = env
    _claim_and_seal(tmp, cand)
    (TL.look_dir("taprisk") / "extra.parquet").write_text("x")
    with pytest.raises(C.ProtectedDataError, match="unlisted"):
        LK.rescore_load(SPLIT)


def test_p6_abort_closes_claim_and_p6d_second_look_refused(env):
    tmp, rows, cand = env
    _good_freeze(tmp, cand)
    LK.claim(SPLIT)
    try:
        raise KeyboardInterrupt
    except BaseException as e:
        LK.abort(SPLIT, repr(e))
    assert (TL.look_dir("taprisk") / "ABORTED.json").exists()
    assert any(r["split"] == "test_aborted" for r in rows)
    with pytest.raises(C.ProtectedDataError):
        LK.require_open(SPLIT)
    for f in TL.look_dir("taprisk").iterdir():                # attacker deletes every marker
        os.chmod(f, 0o644)
        f.unlink()
    with pytest.raises(C.ProtectedDataError):
        LK.claim(SPLIT)                                       # registry claim/abort rows still refuse


def test_wrong_process_refused(env, monkeypatch):
    tmp, rows, cand = env
    _good_freeze(tmp, cand)
    LK.claim(SPLIT)
    monkeypatch.setattr(os, "getpid", lambda: 1)
    with pytest.raises(C.ProtectedDataError, match="claiming process"):
        LK.require_open(SPLIT)


def test_p8b_closure_hash_change_refused(env, monkeypatch):
    tmp, rows, cand = env
    _good_freeze(tmp, cand)
    LK.claim(SPLIT)
    monkeypatch.setattr(REG, "scoring_code_sha256", lambda: "changed")
    with pytest.raises(C.ProtectedDataError, match="closure"):
        LK.require_open(SPLIT)
    assert "sentinel/core/registry.py" in REG.scoring_files() and "sentinel/core/paths.py" in REG.scoring_files()


def test_score_test_mode_end_to_end(env):
    """The full D4 score() in test mode through the registry path on a synthetic sealed prediction frame."""
    import importlib.util
    from pathlib import Path

    import numpy as np

    tmp, rows, cand = env
    _good_freeze(tmp, cand)
    LK.claim(SPLIT)
    spec = importlib.util.spec_from_file_location("tp_d4", Path(__file__).resolve().parents[1] / "scripts/train/downstream_taprisk_test.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rng = np.random.default_rng(0)
    parts = []
    for task in mod.TASKS:
        for name in ("test_known", "test_new"):
            n = 300
            y = (rng.random(n) < 0.2).astype(float)
            p = pd.DataFrame({"pwsid": [f"p{i}" for i in range(n)], "state_code": rng.integers(0, 12, n).astype(str),
                              "huc8": rng.integers(0, 40, n).astype(str), "origin": 2019, "y": y, "task": task, "set": name})
            for k in mod.MODELS:
                p[f"s_{k}"] = y + rng.normal(0, 0.5 if k == mod.CHALLENGER else 1.0, n)
            for k in mod.SEED_CANDS:
                for i in range(3):
                    p[f"s_{k}_seed{i}"] = p[f"s_{k}"]
            parts.append(p)
    for t in ("log10_haa5", "log10_toc", "log10_bromide"):
        n = 120
        yy = rng.normal(1, 0.5, n)
        parts.append(pd.DataFrame({"pwsid": [f"u{i}" for i in range(n)], "state_code": "1", "huc8": rng.integers(0, 20, n).astype(str),
                                   "y_num": yy, "qgbm_lo": yy - 1, "qgbm_med": yy + rng.normal(0, 0.2, n), "qgbm_hi": yy + 1,
                                   "pennino_rf_reg": yy, "state_median": 1.0, "history_median": 1.0, "train_median": 1.0,
                                   "task": f"numeric_{t}", "set": "test_states"}))
    pred = pd.concat(parts, ignore_index=True)
    LK.seal(SPLIT, pred)
    res = mod.score(LK.rescore_load(SPLIT), "test", SPLIT)
    assert set(res["numeric"]) == {"numeric_log10_haa5", "numeric_log10_toc", "numeric_log10_bromide"}
    assert any(r["split"] == "test_states_numeric" for r in rows)
    assert any(str(r["metric"]).startswith("auroc_diff_paired_state") for r in rows)
    assert isinstance(res["claim_met"], bool)


def test_e3_interrupt_while_writing_claim_row_closes_the_claim(env, monkeypatch):
    """E3 (A12): an interrupt between the O_EXCL marker and the test_claim row must leave ABORTED + a registry row,
    and a later look must be refused even after every look file is deleted."""
    tmp, rows, cand = env
    _good_freeze(tmp, cand)
    real = TL.log_test_metric

    def boom(**kw):
        if kw.get("split") == "test_claim":
            raise KeyboardInterrupt
        return real(**kw)

    monkeypatch.setattr(TL, "log_test_metric", boom)
    with pytest.raises(KeyboardInterrupt):
        LK.claim(SPLIT)
    assert (TL.look_dir("taprisk") / "ABORTED.json").exists()
    assert any(r["split"] == "test_aborted" for r in rows)
    monkeypatch.setattr(TL, "log_test_metric", real)
    for f in sorted(TL.look_dir("taprisk").iterdir()):       # attacker deletes every look file
        os.chmod(f, 0o644)
        f.unlink()
    with pytest.raises(C.ProtectedDataError):
        LK.claim(SPLIT)


def test_e4_rescore_refused_after_closure_code_edit(env, monkeypatch):
    """E4 (A10d): an edit to a closure-only module (e.g. taprisk_numeric.py) must stop a rescore."""
    tmp, rows, cand = env
    _claim_and_seal(tmp, cand)
    LK.rescore_load(SPLIT)                                   # clean state rescores
    monkeypatch.setattr(REG, "scoring_code_sha256", lambda: "edited-after-the-freeze")
    with pytest.raises(C.ProtectedDataError, match="closure"):
        LK.rescore_load(SPLIT)


def test_e2_environment_pin_excludes_editable(env):
    """E2: the D4 env pin must not carry the editable/git-HEAD line, and a mismatch is refused."""
    import subprocess
    import sys as _sys

    tmp, rows, cand = env
    fr = _good_freeze(tmp, cand)
    out = subprocess.run([_sys.executable, "-m", "pip", "freeze", "--exclude-editable"],
                         capture_output=True, text=True, check=True).stdout
    assert "-e git+" not in out and "#egg=" not in out
    bad = {**fr, "environment": {"pip_freeze_noeditable_sha256": "0" * 64}}
    C.freeze_file("taprisk").write_text(json.dumps(bad))
    with pytest.raises(C.ProtectedDataError, match="environment"):
        LK.verify_frozen_state(SPLIT)


def test_wa_subfolder_in_pinned_folder_refused(env, monkeypatch, tmp_path):
    """W-a: a file in a SUB-folder of a pinned partition must be refused (hive discovery would read it)."""
    import json as _json

    from sentinel.core import paths as P

    root = tmp_path / "data"
    (root / "processed" / "part").mkdir(parents=True)
    (root / "processed" / "part" / "data.parquet").write_text("x")
    monkeypatch.setattr(P, "DATA_ROOT", root)
    monkeypatch.setattr(LK.paths, "DATA_ROOT", root)
    pins = tmp_path / "pins.json"
    pins.write_text(_json.dumps({"files": {"processed/part/data.parquet": "0"}, "expected": {}}))
    monkeypatch.setattr(TL, "pins_path", lambda app: pins)
    monkeypatch.setattr(LK, "verify_pinned_folders", LK.verify_pinned_folders.__wrapped__
                        if hasattr(LK.verify_pinned_folders, "__wrapped__") else _REAL_VERIFY_FOLDERS)
    _REAL_VERIFY_FOLDERS()
    (root / "processed" / "part" / "sub").mkdir()
    (root / "processed" / "part" / "sub" / "x.parquet").write_text("y")
    with pytest.raises(C.ProtectedDataError, match="sub-folders"):
        _REAL_VERIFY_FOLDERS()


def test_e3_script_level_interrupt_on_claim_row(env, monkeypatch):
    """E3 through the script's main(): an interrupt while the claim row is written must close the look (ABORTED +
    test_aborted row), never reach predict(), and leave no claim row and no sealed predictions."""
    import contextlib
    import importlib.util
    from pathlib import Path

    tmp, rows, cand = env
    _good_freeze(tmp, cand)
    spec = importlib.util.spec_from_file_location("tp_d4_e3", Path(__file__).resolve().parents[1] / "scripts/train/downstream_taprisk_test.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod.C, "cpu_slot", lambda name: contextlib.nullcontext())
    called = []
    monkeypatch.setattr(mod, "predict", lambda *a, **k: called.append(1))
    real = TL.log_test_metric

    def boom(**kw):
        if kw.get("split") == "test_claim":
            raise KeyboardInterrupt
        return real(**kw)

    monkeypatch.setattr(TL, "log_test_metric", boom)
    with pytest.raises(BaseException) as ei:
        mod.main(["--allow-test"])
    assert isinstance(ei.value, (KeyboardInterrupt, FileExistsError))
    d = TL.look_dir("taprisk")
    assert not called
    assert (d / "ABORTED.json").exists() and not (d / "PREDICTIONS_SAVED.json").exists()
    assert any(r["split"] == "test_aborted" for r in rows)
    assert not any(r["split"] == "test_claim" for r in rows)
    monkeypatch.setattr(TL, "log_test_metric", real)
    with pytest.raises(C.ProtectedDataError):                 # the look is spent: a second attempt is refused
        mod.main(["--allow-test"])
