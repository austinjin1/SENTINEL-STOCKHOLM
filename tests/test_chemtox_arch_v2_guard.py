"""Tests for rulings B1 (i)-(v), C-1 and G-1 … G-10 on the M4 arch-v2 single-look guard.

Requirements: ``docs/preregistration/chemtox_arch_v2_lead_rulings.md`` (sha256 ``553be413…``) and
``docs/preregistration/chemtox_arch_v2_lead_guard_rulings_01.md`` (sha256 ``63b7184c…``), which
wins over addendum 01 rev 1. The stamped amendment and stamped rulings win over both.

**No test here reads a real protected table, before or after the pins exist (ruling G-2).** Every
test redirects the guard's paths at temporary directories and builds its own synthetic split,
frame, raw tables, registry and record files. The handful of tests that describe the *pre-pins*
state assert the pins file is absent and **skip before calling any reader** once it is present;
they are not deleted and are never made to pass by a read.
"""

from __future__ import annotations

import json
import os
import re
import stat
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sentinel.models.chemtox import arch_v2 as A
from sentinel.models.chemtox import arch_v2_guard as G

BLOCK = "T1_test"
FAKE_FREEZE_TEXT = "cleared by the lead\n"


def _skip_if_pins_exist() -> None:
    """G-2: the real-path tests must stop BEFORE any reader call once the pins exist."""
    if G.pins_path().exists():
        pytest.skip("ruling G-2: the pins file exists, so calling a reader on the real paths "
                    "would read a protected table; this test describes the pre-pins state only")


# --------------------------------------------------------------------------------- the sandbox

@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """A complete, synthetic guard world: runs dir, repo-relative files, split, frame, registry."""
    repo = tmp_path / "repo"
    runs = tmp_path / "runs" / "chemtox" / "arch_v2"
    data = tmp_path / "data"
    for d in (repo / "docs" / "preregistration", repo / "results", runs, data):
        d.mkdir(parents=True, exist_ok=True)

    # Rulings G-9, M10 and M12 need all three sealed blocks present, and in particular a
    # T6a_test row whose hold-out unit is RETAINED: that is the only way a sealed CELL can exist
    # on chemistry the hold-out rule does not already remove.
    #   1,2 train_pool (units 0,1)        3 T1_test   (unit 2, HELD)
    #   4   T6b_test  (unit 3, HELD)      5 T6a_test  (unit 4, RETAINED, out of window)
    split = pd.DataFrame({
        "result_id": ["1", "2", "3", "4", "5"],
        "test_cas": ["100", "200", "300", "400", "500"],
        "t1_cluster": [0, 1, 2, 3, 4],
        "holdout_unit": [0, 1, 2, 3, 4],
        "duration_h": [48.0, 48.0, 48.0, 720.0, 720.0],
        "duration_band": ["w24_96", "w24_96", "w24_96", "gt168_720", "gt168_720"],
        "in_adore_domain": [True] * 5,
        "shared_reference": [False] * 5,
        "block": ["train_pool", "train_pool", BLOCK, "T6b_test", "T6a_test"],
    })
    split_path = data / "split.parquet"
    split.to_parquet(split_path, index=False)

    frame = pd.DataFrame({"result_id": ["1", "2", "3", "4", "5"],
                          G.TARGET: [-4.0, -5.0, -6.0, -7.0, -8.0],
                          "test_cas": ["100", "200", "300", "400", "500"],
                          "latin_name": ["Danio rerio"] * 5,
                          "duration_h": [48.0, 48.0, 48.0, 720.0, 720.0]})
    frame_path = data / "frame.parquet"
    frame.to_parquet(frame_path, index=False)

    # G-7: every input that decides which rows come back is a pinned path, so the sandbox must
    # provide all of them -- including the two raw tables (G-2).
    pro_path = data / "prospective.parquet"
    pd.DataFrame({"result_id": ["999"]}).to_parquet(pro_path, index=False)
    sa_path = data / "split_assignments.parquet"
    pd.DataFrame({"challenge": ["t-F2F"], "result_id": ["998"], "split": ["split_occurrence"],
                  "role": ["test"]}).to_parquet(sa_path, index=False)
    spent_path = repo / "docs" / "preregistration" / "spent.json"
    spent_path.write_text(json.dumps({"result_ids": ["998"]}) + "\n")
    clusters = data / "clusters.csv"
    pd.DataFrame({"test_cas": ["100", "200", "300", "400", "500"],
                  "chem_smiles": ["CCO"] * 5,
                  "parent_key": ["PK1", "PK2", "PK3", "PK4", "PK5"],
                  "t1_cluster": [0, 1, 2, 3, 4],
                  "holdout_unit": [0, 1, 2, 3, 4]}).to_csv(clusters, index=False)
    h4_results = data / "raw_results.parquet"
    h4_tests = data / "raw_tests.parquet"
    pd.DataFrame({"result_id": [], "test_id": [], "endpoint": [], "bcf1_mean_op": [],
                  "bcf1_mean": [], "bcf1_unit": [], "obs_duration_mean": [],
                  "obs_duration_unit": []}).to_parquet(h4_results, index=False)
    pd.DataFrame({"test_id": [], "test_cas": [], "species_number": []}).to_parquet(
        h4_tests, index=False)

    labels_path = data / "protected_labels_v2.parquet"
    registry = repo / "results" / "registry.jsonl"
    # the frozen loader and _spent_sets() both assert that at least one ADORE test partition was
    # scored (ruling L1), so the synthetic registry carries one row naming the synthetic partition
    registry.write_text(json.dumps({"experiment": "adore_chemtox_test", "workstream": "chemtox",
                                    "split": "t-F2F/split_occurrence/test", "metric": "rmse",
                                    "value": 0.9}) + "\n")

    monkeypatch.setattr(G, "REPO", repo)
    monkeypatch.setattr(G, "RUNS", runs)
    monkeypatch.setattr(G, "REPORT_DIR", runs / "aux_exclusions")
    monkeypatch.setattr(G, "LABELS_MARKER", runs / "PROTECTED_LABELS.json")
    monkeypatch.setattr(G, "REGISTRY", registry)
    monkeypatch.setattr(G, "LABELS_PATH", labels_path)
    monkeypatch.setattr(G, "H4_RESULTS", h4_results)
    monkeypatch.setattr(G, "H4_TESTS", h4_tests)
    monkeypatch.setattr(A, "SPLIT_PATH", split_path)
    monkeypatch.setattr(A, "ECOTOX", frame_path)
    monkeypatch.setattr(A, "PROSPECTIVE", pro_path)
    monkeypatch.setattr(A, "ADORE_SPLITS", sa_path)
    monkeypatch.setattr(A, "SPENT_IDS_PATH", spent_path)
    monkeypatch.setattr(A, "CLUSTER_PATH", clusters)
    monkeypatch.setattr(A, "REGISTRY", registry)
    for rel in (G.AMENDMENT, G.RULINGS):
        q = repo / rel
        q.parent.mkdir(parents=True, exist_ok=True)
        q.write_text(f"synthetic {rel}\n")
    monkeypatch.setattr(G, "PINNED_REPO_FILES", (G.AMENDMENT, G.RULINGS))

    return {"repo": repo, "runs": runs, "data": data, "split": split_path,
            "frame": frame_path, "labels": labels_path, "registry": registry,
            "clusters": clusters, "h4_results": h4_results, "h4_tests": h4_tests}


def _write_registry_row(sandbox, *, written_by="lead", **extra):
    row = {"experiment": G.PREREG_EXPERIMENT, "workstream": "chemtox", "metric": "stamp",
           "value": 1.0, "extra": {"written_by": written_by, **extra}}
    with open(sandbox["registry"], "a") as fh:
        fh.write(json.dumps(row) + "\n")


def _write_pins(sandbox, *, vouch=True, mode=0o444, mutate=None):
    """Write a pins file and (by default) the lead row that vouches for it (G-1)."""
    pins = G.make_pins()
    if mutate is not None:
        mutate(pins)
    p = G.pins_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        os.chmod(p, 0o644)
    p.write_text(json.dumps(pins, indent=1, sort_keys=True) + "\n")
    os.chmod(p, mode)
    if vouch:
        _write_registry_row(sandbox, pins_sha256=G.file_sha(p))
    return p


def _write_freeze_file(block=BLOCK, text=FAKE_FREEZE_TEXT) -> str:
    f = G.freeze_path(block)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(text)
    return G.file_sha(f)


def _write_lead_record(sha, block=BLOCK, *, mode=0o444):
    p = G.freeze_record_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        os.chmod(p, 0o644)
    p.write_text(json.dumps({"written_by": "lead",
                             "blocks": {block: {"freeze_sha256": sha}}}, indent=1) + "\n")
    os.chmod(p, mode)
    return p


def _fully_unlocked(sandbox, block=BLOCK):
    """Pins + lead row + FREEZE + both freeze records: the only state a look may proceed in."""
    sha = _write_freeze_file(block)
    _write_lead_record(sha, block)
    _write_registry_row(sandbox, block=block, freeze_sha256=sha)
    _write_pins(sandbox)
    return sha


def _pins_only(sandbox):
    """Pins + lead row, but no FREEZE: readers work, looks do not."""
    _write_pins(sandbox)


# ========================================================= B1 (i) / C-1 — the two-source FREEZE check

def test_i_refuses_when_the_lead_record_file_is_missing(sandbox):
    _write_freeze_file()
    _write_registry_row(sandbox, block=BLOCK, freeze_sha256="a" * 64)
    with pytest.raises(G.ProtectedBlockError, match="no lead-written FREEZE record"):
        G.freeze_sha(BLOCK)


def test_i_refuses_when_the_record_file_is_writable(sandbox):
    sha = _write_freeze_file()
    _write_lead_record(sha, mode=0o644)
    _write_registry_row(sandbox, block=BLOCK, freeze_sha256=sha)
    with pytest.raises(G.ProtectedBlockError, match="must be 0444"):
        G.freeze_sha(BLOCK)


def test_i_refuses_when_the_registry_row_is_missing(sandbox):
    sha = _write_freeze_file()
    _write_lead_record(sha)
    with pytest.raises(G.ProtectedBlockError, match="second, independent source"):
        G.freeze_sha(BLOCK)


def test_i_refuses_when_the_registry_row_is_not_written_by_the_lead(sandbox):
    sha = _write_freeze_file()
    _write_lead_record(sha)
    _write_registry_row(sandbox, written_by="chemtox", block=BLOCK, freeze_sha256=sha)
    with pytest.raises(G.ProtectedBlockError, match="written_by"):
        G.freeze_sha(BLOCK)


def test_i_refuses_when_the_two_lead_records_disagree(sandbox):
    sha = _write_freeze_file()
    _write_lead_record(sha)
    _write_registry_row(sandbox, block=BLOCK, freeze_sha256="b" * 64)
    with pytest.raises(G.ProtectedBlockError, match="two records disagree"):
        G.freeze_sha(BLOCK)


def test_i_refuses_a_freeze_file_the_lead_did_not_write(sandbox):
    """The core of B1 (i): a workstream-written FREEZE is void."""
    sha = _write_freeze_file()
    _write_lead_record(sha)
    _write_registry_row(sandbox, block=BLOCK, freeze_sha256=sha)
    G.freeze_path(BLOCK).write_text("forged by the workstream\n")   # same path, different bytes
    with pytest.raises(G.ProtectedBlockError, match="VOID"):
        G.freeze_sha(BLOCK)


def test_i_refuses_two_conflicting_registry_rows(sandbox):
    sha = _write_freeze_file()
    _write_lead_record(sha)
    _write_registry_row(sandbox, block=BLOCK, freeze_sha256=sha)
    _write_registry_row(sandbox, block=BLOCK, freeze_sha256="c" * 64)
    with pytest.raises(G.ProtectedBlockError, match="different FREEZE sha256"):
        G.freeze_sha(BLOCK)


def test_i_accepts_only_when_file_and_both_records_agree(sandbox):
    sha = _fully_unlocked(sandbox)
    assert G.freeze_sha(BLOCK) == sha
    assert G.lead_recorded_freeze_sha(BLOCK) == sha


def test_i_record_path_is_a_guard_constant_under_docs_preregistration():
    assert G.FREEZE_RECORD == "docs/preregistration/chemtox_arch_v2_freeze_shas.json"
    assert G.PREREG_EXPERIMENT == "chemtox/prereg_snapshot"


# ===================================================== B1 (ii) — verify_freeze inside require_authorised

def test_ii_require_authorised_reruns_verify_freeze(sandbox):
    _fully_unlocked(sandbox)
    G.authorise(BLOCK)
    # a pinned file changes after authorisation: every later use must now refuse
    (sandbox["repo"] / G.AMENDMENT).write_text("tampered\n")
    with pytest.raises(G.ProtectedBlockError, match="on disk"):
        G.require_authorised(BLOCK)


def test_ii_require_authorised_refuses_without_pins(sandbox):
    _fully_unlocked(sandbox)
    G.authorise(BLOCK)
    G.pins_path().unlink()
    with pytest.raises(G.ProtectedBlockError, match="not been stamped"):
        G.require_authorised(BLOCK)


def test_ii_load_test_block_refused_after_scored(sandbox):
    _fully_unlocked(sandbox)
    G.authorise(BLOCK)
    G.save_predictions(BLOCK, pd.DataFrame({"result_id": ["3"], "pred": [-6.0]}))
    G.mark_scored(BLOCK, {"rmse": 1.0})
    with pytest.raises(G.ProtectedBlockError, match="look is over"):
        G.load_test_block(BLOCK)


def test_ii_load_test_block_refused_after_abort(sandbox):
    _fully_unlocked(sandbox)
    with pytest.raises(RuntimeError, match="boom"):
        with G.scored_pass(BLOCK):
            raise RuntimeError("boom")
    assert (G.block_dir(BLOCK) / "ABORTED.json").is_file()
    with pytest.raises(G.ProtectedBlockError, match="look is over"):
        G.load_test_block(BLOCK)


def test_ii_load_test_block_refused_without_authorisation(sandbox):
    _fully_unlocked(sandbox)
    with pytest.raises(G.ProtectedBlockError, match="not authorised"):
        G.load_test_block(BLOCK)


def test_ii_load_test_block_returns_only_that_block(sandbox):
    _fully_unlocked(sandbox)
    G.authorise(BLOCK)
    rows = G.load_test_block(BLOCK)
    assert sorted(rows["result_id"]) == ["3"]
    assert G.TARGET in rows.columns


# ========================================== B1 (iii) — verify_artifacts before mark_scored

def test_iii_finish_verifies_then_scores(sandbox):
    _fully_unlocked(sandbox)
    with G.scored_pass(BLOCK) as look:
        look.save_predictions(pd.DataFrame({"result_id": ["3"], "pred": [-6.0]}))
        v = look.finish({"rmse": 1.23})
    assert v["verified"] >= 2
    rec = json.loads((G.block_dir(BLOCK) / "SCORED.json").read_text())
    assert rec["artifact_verification"]["verified"] == v["verified"]


def test_iii_finish_refuses_when_an_artefact_changed(sandbox):
    _fully_unlocked(sandbox)
    with pytest.raises(G.ProtectedBlockError, match="on disk"):
        with G.scored_pass(BLOCK) as look:
            look.save_predictions(pd.DataFrame({"result_id": ["3"], "pred": [-6.0]}))
            p = G.block_dir(BLOCK) / "predictions.parquet"
            os.chmod(p, 0o644)
            p.write_bytes(b"corrupted")
            look.finish({"rmse": 1.0})
    assert not (G.block_dir(BLOCK) / "SCORED.json").exists()


def test_iii_mark_scored_cannot_be_bypassed(sandbox):
    """Calling mark_scored directly still runs verify_artifacts."""
    _fully_unlocked(sandbox)
    G.authorise(BLOCK)
    with pytest.raises(G.ProtectedBlockError, match="nothing to verify"):
        G.mark_scored(BLOCK, {"rmse": 1.0})


def test_iii_mark_scored_refuses_a_fake_verification(sandbox):
    _fully_unlocked(sandbox)
    G.authorise(BLOCK)
    G.save_predictions(BLOCK, pd.DataFrame({"result_id": ["3"], "pred": [-6.0]}))
    with pytest.raises(G.ProtectedBlockError, match="has not passed"):
        G.mark_scored(BLOCK, {"rmse": 1.0}, verification={"verified": 0})


def test_iii_verify_artifacts_catches_an_unrecorded_file(sandbox):
    _fully_unlocked(sandbox)
    G.authorise(BLOCK)
    G.save_predictions(BLOCK, pd.DataFrame({"result_id": ["3"], "pred": [-6.0]}))
    (G.block_dir(BLOCK) / "sneaked_in.txt").write_text("x")
    with pytest.raises(G.ProtectedBlockError, match="absent from SHA256SUMS"):
        G.verify_artifacts(G.block_dir(BLOCK))


def test_iii_scored_pass_refuses_to_seal_without_finish(sandbox):
    _fully_unlocked(sandbox)
    with pytest.raises(G.ProtectedBlockError, match="left no SCORED.json"):
        with G.scored_pass(BLOCK) as look:
            look.save_predictions(pd.DataFrame({"result_id": ["3"], "pred": [-6.0]}))


# ================================================= B1 (iv) — ABORTED.json records what was written

def test_iv_aborted_records_the_actual_inventory(sandbox):
    _fully_unlocked(sandbox)
    with pytest.raises(ValueError):
        with G.scored_pass(BLOCK) as look:
            look.save_predictions(pd.DataFrame({"result_id": ["3"], "pred": [-6.0]}))
            raise ValueError("after predictions")
    rec = json.loads((G.block_dir(BLOCK) / "ABORTED.json").read_text())
    names = {f["name"] for f in rec["files"]}
    assert "predictions.parquet" in names and "AUTHORISED.json" in names
    assert rec["predictions_written"] is True
    assert rec["markers"]["AUTHORISED.json"] is True
    assert rec["markers"]["SCORED.json"] is False
    assert rec["error"] == "ValueError"
    for f in rec["files"]:
        assert len(f["sha256"]) == 64 and f["bytes"] > 0


def test_iv_aborted_before_predictions_says_so(sandbox):
    _fully_unlocked(sandbox)
    with pytest.raises(RuntimeError):
        with G.scored_pass(BLOCK):
            raise RuntimeError("early")
    rec = json.loads((G.block_dir(BLOCK) / "ABORTED.json").read_text())
    assert rec["predictions_written"] is False
    assert "predictions.parquet" not in {f["name"] for f in rec["files"]}


def test_iv_abort_spends_the_look(sandbox):
    """An abort after AUTHORISED.json exists spends the block's look; there is no second one."""
    _fully_unlocked(sandbox)
    with pytest.raises(RuntimeError):
        with G.scored_pass(BLOCK):
            raise RuntimeError("boom")
    rec = json.loads((G.block_dir(BLOCK) / "ABORTED.json").read_text())
    assert rec["look_spent"] is True and "no second look" in rec["look_spent_reason"]
    with pytest.raises(G.ProtectedBlockError):
        G.authorise(BLOCK)
    with pytest.raises(G.ProtectedBlockError):
        with G.scored_pass(BLOCK):
            pass


def test_iv_a_second_abort_does_not_mask_the_error(sandbox):
    _fully_unlocked(sandbox)
    for msg in ("first", "second"):
        with pytest.raises(G.ProtectedBlockError if msg == "second" else RuntimeError):
            with G.scored_pass(BLOCK):
                raise RuntimeError(msg)
    assert (G.block_dir(BLOCK) / "ABORTED.json").is_file()


def test_iv_keyboardinterrupt_is_recorded_and_reraised(sandbox):
    _fully_unlocked(sandbox)
    with pytest.raises(KeyboardInterrupt):
        with G.scored_pass(BLOCK):
            raise KeyboardInterrupt
    rec = json.loads((G.block_dir(BLOCK) / "ABORTED.json").read_text())
    assert rec["error"] == "KeyboardInterrupt"
    assert not (G.block_dir(BLOCK) / "SCORED.json").exists()


# ============================================ B1 (v) — pins cover ecotox and the labels artefact

def test_v_pins_cover_ecotox_and_the_labels_path(sandbox):
    _fully_unlocked(sandbox)
    pins = json.loads(G.pins_path().read_text())
    assert str(A.ECOTOX) in pins["data"]
    assert str(G.LABELS_PATH) in pins["data"]
    assert pins["data"][str(G.LABELS_PATH)] is None, "not built before the first look"
    assert pins["freeze_record"] == G.FREEZE_RECORD


def test_v_verify_freeze_refuses_pins_without_ecotox(sandbox):
    _write_freeze_file()
    _write_pins(sandbox, mutate=lambda q: q["data"].pop(str(A.ECOTOX), None))
    with pytest.raises(G.ProtectedBlockError, match="omit pinned data path"):
        G.verify_freeze()


def test_v_verify_freeze_refuses_pins_without_the_labels_path(sandbox):
    _write_freeze_file()
    _write_pins(sandbox, mutate=lambda q: q["data"].pop(str(G.LABELS_PATH), None))
    with pytest.raises(G.ProtectedBlockError, match="omit pinned data path"):
        G.verify_freeze()


def test_v_a_changed_ecotox_hash_stops_everything(sandbox):
    """C-5: a changed hash of the frame holding every sealed target stops every T1/T6 step."""
    _fully_unlocked(sandbox)
    pd.DataFrame({"result_id": ["1"], G.TARGET: [0.0], "test_cas": ["100"],
                  "latin_name": ["Danio rerio"], "duration_h": [48.0]}).to_parquet(
        A.ECOTOX, index=False)
    with pytest.raises(G.ProtectedBlockError, match="on disk"):
        G.verify_freeze()
    with pytest.raises(G.ProtectedBlockError):
        G.load_training_pool()
    with pytest.raises(G.ProtectedBlockError):
        G.authorise(BLOCK)


def test_v_labels_artefact_is_pinned_once_built(sandbox):
    _fully_unlocked(sandbox)
    G.authorise(BLOCK)
    assert Path(G.LABELS_PATH).exists()
    pins = G.make_pins()
    assert pins["data"][str(G.LABELS_PATH)] == G.file_sha(G.LABELS_PATH)
    assert stat.S_IMODE(Path(G.LABELS_PATH).stat().st_mode) == 0o444


def test_v_require_labels_flag(sandbox):
    _fully_unlocked(sandbox)
    with pytest.raises(G.ProtectedBlockError, match="null value"):
        G.verify_freeze(require_labels=True)


# ================================================================= the rest of the look discipline

def test_authorise_refuses_without_pins(sandbox):
    sha = _write_freeze_file()
    _write_lead_record(sha)
    _write_registry_row(sandbox, block=BLOCK, freeze_sha256=sha)
    with pytest.raises(G.ProtectedBlockError, match="not been stamped"):
        G.authorise(BLOCK)


def test_authorise_refuses_without_a_freeze_file(sandbox):
    _write_lead_record("d" * 64)
    _write_registry_row(sandbox, block=BLOCK, freeze_sha256="d" * 64)
    _write_pins(sandbox)
    with pytest.raises(G.ProtectedBlockError, match="unlocked by the LEAD"):
        G.authorise(BLOCK)


def test_authorise_is_single_use(sandbox):
    """G-5: the registry claim refuses the second authorise before the local marker is even read."""
    _fully_unlocked(sandbox)
    G.authorise(BLOCK)
    with pytest.raises(G.ProtectedBlockError, match="look claim"):
        G.authorise(BLOCK)


def test_authorise_refuses_when_the_registry_holds_block_rows(sandbox):
    _fully_unlocked(sandbox)
    with open(sandbox["registry"], "a") as fh:
        fh.write(json.dumps({"experiment": G.EXPERIMENT, "split": f"{BLOCK}/x",
                             "metric": "rmse", "value": 1.0}) + "\n")
    with pytest.raises(G.ProtectedBlockError, match="second look"):
        G.authorise(BLOCK)


def test_authorise_records_both_lead_sources_and_the_code_hash(sandbox):
    sha = _fully_unlocked(sandbox)
    rec = G.authorise(BLOCK)
    assert rec["freeze_sha256"] == rec["freeze_sha256_lead_record"] == sha
    assert rec["freeze_record_file"] == G.FREEZE_RECORD
    assert len(rec["code_sha"]) == 64 and rec["n_modules_hashed"] > 3
    assert rec["input_hashes"]["ecotox"] == G.file_sha(A.ECOTOX)
    assert rec["pins_sha256"] == G.pins_sha256()
    assert rec["lead_row"]["workstream"] == "chemtox"
    assert rec["runs_root"] == str(Path(G.RUNS).resolve())


def test_predictions_are_write_once(sandbox):
    _fully_unlocked(sandbox)
    G.authorise(BLOCK)
    G.save_predictions(BLOCK, pd.DataFrame({"result_id": ["3"], "pred": [-6.0]}))
    with pytest.raises(G.ProtectedBlockError, match="write-once"):
        G.save_predictions(BLOCK, pd.DataFrame({"result_id": ["3"], "pred": [-7.0]}))


def test_load_training_pool_returns_pool_rows_only(sandbox):
    _fully_unlocked(sandbox)
    pool = G.load_training_pool()
    assert sorted(pool["result_id"]) == ["1", "2"]


def test_load_training_pool_refuses_without_pins(sandbox):
    with pytest.raises(G.ProtectedBlockError, match="not been stamped"):
        G.load_training_pool()


def test_build_protected_labels_refuses_without_the_lead_records(sandbox):
    _write_freeze_file()
    _write_pins(sandbox)
    with pytest.raises(G.ProtectedBlockError, match="no lead-written FREEZE record"):
        G.build_protected_labels(block=BLOCK)


def test_seal_dir_closes_access(sandbox):
    _fully_unlocked(sandbox)
    with G.scored_pass(BLOCK) as look:
        look.save_predictions(pd.DataFrame({"result_id": ["3"], "pred": [-6.0]}))
        look.finish({"rmse": 1.0})
    d = G.block_dir(BLOCK)
    for f in d.rglob("*"):
        if f.is_file():
            assert stat.S_IMODE(f.stat().st_mode) == 0o444, f
    assert G.verify_artifacts(d)["verified"] >= 3


def test_look_state_reads_nothing(sandbox):
    st = G.look_state(BLOCK)
    assert st["block"] == BLOCK and st["markers"]["AUTHORISED.json"] is False
    assert st["labels_artefact"] is False


def test_unknown_block_is_refused_everywhere(sandbox):
    for fn in (G.block_dir, G.freeze_path, G.registry_rows, G.look_state,
               G.lead_recorded_freeze_sha, G.require_authorised):
        with pytest.raises(G.ProtectedBlockError, match="unknown block"):
            fn("T9_test")


def test_guard_does_not_import_frozen_track_code():
    body = Path(G.__file__).read_text().split('"""', 2)[2]
    assert "unseen_taxa" not in body


def test_live_guard_still_refuses_everything():
    """Against the REAL paths: no pins file, so nothing is unlocked (the stamp is partial)."""
    _skip_if_pins_exist()
    with pytest.raises(G.ProtectedBlockError, match="not been stamped"):
        G.verify_freeze()
    with pytest.raises(G.ProtectedBlockError, match="not been stamped"):
        G.load_training_pool()
    for b in G.BLOCKS:
        assert G.registry_rows(b) == 0
        assert G.look_state(b)["markers"]["AUTHORISED.json"] is False
    assert not Path(G.LABELS_PATH).exists()
    assert not G.pins_path().exists()
    assert not G.freeze_record_path().exists()


def test_code_hash_cannot_be_blanked_by_redirecting_paths(sandbox):
    """The claim's code_sha anchors on the module's own location, not on the patched REPO."""
    c = G.import_closure_sha()
    assert c["n_modules"] > 3
    assert "sentinel/models/chemtox/arch_v2_guard.py" in c["modules"]
    assert "sentinel/models/chemtox/arch_v2.py" in c["modules"]


# ============================================ addendum 01 (N2-A) — the guarded auxiliary readers
#
# Synthetic frames in temporary directories only. No universe row is read: the sandbox points
# A.ECOTOX, A.PROSPECTIVE, A.ADORE_SPLITS, A.CLUSTER_PATH and the raw H4 tables at small frames
# built here, so the real ecotox_aquatic.parquet is never opened.

_AUX_COLS = ["result_id", "test_cas", "endpoint_norm", "censored", "log10_conc_mol_L",
             "chem_smiles", "duration_h", "latin_name", "class", "reference_number",
             "chem_structure_class", "chem_inchikey"]


def _aux_frame(rows):
    """Build an ecotox-shaped frame from (result_id, cas, endpoint, censored, duration, species)."""
    recs = []
    for rid, cas, endpoint, cens, dur, sp in rows:
        recs.append({"result_id": rid, "test_cas": cas, "endpoint_norm": endpoint,
                     "censored": cens, "log10_conc_mol_L": -5.0, "chem_smiles": "CCO",
                     "duration_h": dur, "latin_name": sp, "class": "Actinopterygii",
                     "reference_number": "R1", "chem_structure_class": "organic",
                     "chem_inchikey": "KEY"})
    return pd.DataFrame(recs, columns=_AUX_COLS)


@pytest.fixture
def aux_sandbox(sandbox, tmp_path, monkeypatch):
    """`sandbox` plus the extra frames `arch_v2.auxiliary_rows` needs, all synthetic."""
    data = sandbox["data"]
    # primary frame: ids 1,2 pool / 3,4 sealed, plus chronic auxiliary ids 10..13
    # the sealed rows must match the split of `sandbox`, or a planted sealed CELL would not be
    # in the sealed set and the re-check would have nothing to find (G-9)
    frame = _aux_frame([
        ("1", "100", "LC50", False, 48.0, "Danio rerio"),
        ("2", "200", "LC50", False, 48.0, "Danio rerio"),
        ("3", "300", "LC50", False, 48.0, "Danio rerio"),
        ("4", "400", "LC50", False, 720.0, "Danio rerio"),
        ("5", "500", "LC50", False, 720.0, "Danio rerio"),
        ("10", "100", "NOEC", False, 48.0, "Danio rerio"),      # pool chemistry, admissible
        ("11", "100", "NOEC", False, 720.0, "Danio rerio"),     # long duration: out on T6
        ("12", "300", "NOEC", False, 48.0, "Danio rerio"),      # held-out chemistry: always out
        ("13", "200", "NOEC", False, 48.0, "Danio rerio"),      # pool chemistry, admissible
    ])
    fp = data / "frame_aux.parquet"
    frame.to_parquet(fp, index=False)
    monkeypatch.setattr(A, "ECOTOX", fp)

    pro = data / "prospective.parquet"
    pd.DataFrame({"result_id": ["999"]}).to_parquet(pro, index=False)
    monkeypatch.setattr(A, "PROSPECTIVE", pro)

    sa = data / "split_assignments.parquet"
    pd.DataFrame({"challenge": ["t-F2F"], "result_id": ["998"], "split": ["split_occurrence"],
                  "role": ["test"]}).to_parquet(sa, index=False)
    monkeypatch.setattr(A, "ADORE_SPLITS", sa)
    monkeypatch.setattr(A, "REGISTRY", sandbox["registry"])
    spent = data / "spent_aux.json"
    spent.write_text(json.dumps({"result_ids": ["998"]}) + "\n")
    monkeypatch.setattr(A, "SPENT_IDS_PATH", spent)
    # the frozen loader asserts that at least one ADORE test partition was scored (ruling L1), so
    # the synthetic registry carries one row naming the synthetic split_assignments partition.
    with open(sandbox["registry"], "a") as fh:
        fh.write(json.dumps({"experiment": "adore_chemtox_test", "workstream": "chemtox",
                             "split": "t-F2F/split_occurrence/test", "metric": "rmse",
                             "value": 0.9}) + "\n")

    clusters = data / "clusters.csv"
    pd.DataFrame({"test_cas": ["100", "200", "300", "400", "500"],
                  "chem_smiles": ["CCO"] * 5,
                  "parent_key": ["PK1", "PK2", "PK3", "PK4", "PK5"],
                  "t1_cluster": [0, 1, 2, 3, 4],
                  "holdout_unit": [0, 1, 2, 3, 4]}).to_csv(clusters, index=False)
    monkeypatch.setattr(A, "CLUSTER_PATH", clusters)

    monkeypatch.setattr(G, "REPORT_DIR", tmp_path / "aux_exclusions")
    return sandbox


def test_aux_reader_refuses_without_pins(aux_sandbox):
    """Nothing may run before the lead writes the pins file (S2, addendum 01 §3)."""
    with pytest.raises(G.ProtectedBlockError, match="not been stamped"):
        G.load_auxiliary_rows("H3", "T1")
    with pytest.raises(G.ProtectedBlockError, match="not been stamped"):
        G.load_h4_rows("T1")


def test_aux_reader_refuses_a_caller_supplied_path(aux_sandbox):
    _fully_unlocked(aux_sandbox)
    for kw in ({"ecotox": "/tmp/x.parquet"}, {"cluster_table": pd.DataFrame()},
               {"registry": "/tmp/r.jsonl"}, {"splits": "/tmp/s.parquet"},
               {"prospective": "/tmp/p.parquet"}, {"split": pd.DataFrame()}):
        with pytest.raises(G.ProtectedBlockError, match="refused"):
            G.load_auxiliary_rows("H3", "T1", **kw)
        with pytest.raises(G.ProtectedBlockError, match="refused"):
            G.load_h4_rows("T1", **kw)


def test_aux_reader_rejects_unexpected_arguments(aux_sandbox):
    _fully_unlocked(aux_sandbox)
    with pytest.raises(G.ProtectedBlockError, match="unexpected arguments"):
        G.load_auxiliary_rows("H3", "T1", nonsense=1)


def test_aux_reader_returns_only_admissible_rows(aux_sandbox):
    _fully_unlocked(aux_sandbox)
    rows, rep = G.load_auxiliary_rows("H3", "T1")
    assert sorted(rows["result_id"]) == ["10", "11", "13"], "held-out chemistry 12 must be gone"
    assert rep["guarded_by"] == "arch_v2_guard.load_auxiliary_rows"
    # G-9: the re-check reports all seven, each zero on the happy path
    for key in G.SEVEN_EXCLUSIONS:
        assert rep["recheck"][key] == 0, key
    assert rep["recheck"]["sealed_result_ids"] == 0
    # G-4: the guard joined the target itself
    assert rep["target_column"] == G.TARGET
    assert G.TARGET in rows.columns and rows[G.TARGET].notna().all()
    assert Path(rep["report_path"]).is_file()


def test_aux_reader_applies_the_t6_duration_cap(aux_sandbox):
    _fully_unlocked(aux_sandbox)
    rows, rep = G.load_auxiliary_rows("H3", "T6")
    assert sorted(rows["result_id"]) == ["10", "13"], "the 720 h row must be gone on T6"
    assert rows["duration_h"].max() <= A.DURATION_WINDOW[1]
    assert rep["removed_outside_24_96h_for_T6"] >= 1


def test_aux_reader_writes_a_counts_only_exclusion_report(aux_sandbox):
    """G-8: 0444, with the pins sha, the time, the pid and every input hash; counts only."""
    _fully_unlocked(aux_sandbox)
    _rows, rep = G.load_auxiliary_rows("H3", "T1")
    path = Path(rep["report_path"])
    rec = json.loads(path.read_text())
    blob = json.dumps(rec)
    for banned in ("sd_target", "replicate_floor"):
        assert banned not in blob, banned
    # the target's NAME appears (G-4 requires the reader to return it); no target VALUE does
    assert rec["target_column"] == G.TARGET
    for v in (-4.0, -5.0, -6.0, -7.0):
        assert str(v) not in blob, v
    assert rec["recheck"]["rows"] == 3
    assert rec["pins_sha256"] == G.pins_sha256()
    assert rec["pid"] == os.getpid() and rec["written_at"]
    for key in ("split", "ecotox", "prospective", "adore_splits", "h4_results", "h4_tests"):
        assert len(rec["input_hashes"][key]) == 64, key
    assert stat.S_IMODE(path.stat().st_mode) == 0o444


def test_aux_recheck_catches_a_planted_sealed_result_id(aux_sandbox, monkeypatch):
    """The re-check must fire even if auxiliary_rows' own filters were bypassed."""
    _fully_unlocked(aux_sandbox)
    real = A.auxiliary_rows

    def leaky(target, split, **kw):
        rows, rep = real(target, split, **kw)
        planted = _aux_frame([("3", "100", "NOEC", False, 48.0, "Danio rerio")])
        return pd.concat([rows, planted[rows.columns]], ignore_index=True), rep

    monkeypatch.setattr(A, "auxiliary_rows", leaky)
    with pytest.raises(G.ProtectedBlockError, match="sealed_result_ids"):
        G.load_auxiliary_rows("H3", "T1")


def test_aux_recheck_catches_a_planted_sealed_cell(aux_sandbox, monkeypatch):
    _fully_unlocked(aux_sandbox)
    real = A.auxiliary_rows

    def leaky(target, split, **kw):
        rows, rep = real(target, split, **kw)
        # id is not sealed, but (cas, species, duration) matches sealed row 3
        planted = _aux_frame([("77", "300", "NOEC", False, 48.0, "Danio rerio")])
        return pd.concat([rows, planted[rows.columns]], ignore_index=True), rep

    monkeypatch.setattr(A, "auxiliary_rows", leaky)
    with pytest.raises(G.ProtectedBlockError, match="same_cell_as_sealed_row|held_out_unit_chemistry"):
        G.load_auxiliary_rows("H3", "T1")


def test_aux_recheck_catches_planted_held_out_unit_chemistry(aux_sandbox, monkeypatch):
    _fully_unlocked(aux_sandbox)
    real = A.auxiliary_rows

    def leaky(target, split, **kw):
        rows, rep = real(target, split, **kw)
        # held-out unit 3 (cas 400), but a cell and id that are not sealed
        planted = _aux_frame([("78", "400", "NOEC", False, 96.0, "Oncorhynchus mykiss")])
        return pd.concat([rows, planted[rows.columns]], ignore_index=True), rep

    monkeypatch.setattr(A, "auxiliary_rows", leaky)
    with pytest.raises(G.ProtectedBlockError, match="held-out hold-out unit"):
        G.load_auxiliary_rows("H3", "T1")


def test_aux_recheck_message_carries_counts_only(aux_sandbox, monkeypatch):
    _fully_unlocked(aux_sandbox)
    real = A.auxiliary_rows

    def leaky(target, split, **kw):
        rows, rep = real(target, split, **kw)
        planted = _aux_frame([("3", "100", "NOEC", False, 48.0, "Danio rerio")])
        return pd.concat([rows, planted[rows.columns]], ignore_index=True), rep

    monkeypatch.setattr(A, "auxiliary_rows", leaky)
    with pytest.raises(G.ProtectedBlockError) as exc:
        G.load_auxiliary_rows("H3", "T1")
    msg = str(exc.value)
    assert "found 1 row(s)" in msg
    assert "-5.0" not in msg and "-6.0" not in msg


def test_h4_reader_track_validation(aux_sandbox):
    _fully_unlocked(aux_sandbox)
    with pytest.raises(G.ProtectedBlockError, match="track must be"):
        G.load_h4_rows("T9")


def test_h4_report_script_reads_no_parquet():
    """Addendum 01 §3: the script calls the guard and opens nothing itself."""
    src = Path("scripts/eval/chemtox_arch_v2_h4_report.py").read_text()
    assert "read_parquet" not in src
    assert "load_h4_rows" in src


def test_h4_pure_module_does_no_io():
    from sentinel.models.chemtox import arch_v2_h4 as H4
    src = Path(H4.__file__).read_text()
    for banned in ("read_parquet", "read_csv", "open(", "Path("):
        assert banned not in src, banned


def test_h4_pure_module_rules_on_synthetic_frames():
    """The join/QC rules, exercised without touching any real table."""
    from sentinel.models.chemtox import arch_v2_h4 as H4
    results = pd.DataFrame({
        "result_id": ["1", "2", "3", "4"],
        "test_id": ["t1", "t2", "t3", "t4"],
        "endpoint": ["BCF", "BCFD", "LC50", "BAF"],
        "bcf1_mean_op": ["", "", "", "<"],
        "bcf1_mean": ["100", "50", "1", "1e9"],
        "bcf1_unit": ["L/kg", "L/kg dry wt", "L/kg", "RA"],
        "obs_duration_mean": ["48", "2", "48", "48"],
        "obs_duration_unit": ["h", "d", "h", "h"],
    })
    fam = H4.attach_basis(H4.admit_units(H4.parse_values(H4.select_bcf_family(results))))
    assert sorted(fam["result_id"]) == ["1", "2", "4"], "the LC50 row is not BCF family"
    assert dict(zip(fam["result_id"], fam["basis"])) == {"1": "wet", "2": "dry", "4": "wet"}
    assert dict(zip(fam["result_id"], fam["unit_ok"]))["4"] is False, "RA is not a BCF unit"
    tests = pd.DataFrame({"test_id": ["t1", "t2", "t4"], "test_cas": ["00100-1", "200", "400"],
                          "species_number": ["s1", "s1", "s1"]})
    fam = H4.attach_duration_h(H4.join_structures(
        fam, tests, {"1001": "CCO", "200": "CCC"}, {"s1": "Danio rerio"}))
    assert dict(zip(fam["result_id"], fam["duration_h"]))["2"] == 48.0, "2 d -> 48 h"
    kept, out = H4.trim_implausible(fam)
    assert sorted(kept["result_id"]) == ["1", "2"] and out == 0
    rep = H4.qc_report(len(results), fam, kept, out)
    assert rep["bcf_family_rows"] == 3 and rep["unfiltered_usable"]["rows"] == 2
    assert "bcf1_mean" in rep["value_column_finding"]


def test_guard_still_refuses_the_new_readers_on_real_paths():
    """Against the REAL paths: no pins file, so neither guarded reader can run."""
    _skip_if_pins_exist()
    with pytest.raises(G.ProtectedBlockError, match="not been stamped"):
        G.load_auxiliary_rows("H3", "T1")
    with pytest.raises(G.ProtectedBlockError, match="not been stamped"):
        G.load_h4_rows("T6")
    assert not G.pins_path().exists()


# ==================================================== G-1  the gate must require the LEAD (F1)

def test_g1_pins_without_a_lead_row_are_refused(sandbox):
    """F1's headline: a two-key JSON any workstream can write is not a gate."""
    _write_pins(sandbox, vouch=False)
    with pytest.raises(G.ProtectedBlockError, match="LEAD vouches"):
        G.verify_freeze()


def test_g1_a_writable_pins_file_is_refused(sandbox):
    _write_pins(sandbox, mode=0o644)
    with pytest.raises(G.ProtectedBlockError, match="writable"):
        G.verify_freeze()


def test_g1_a_lead_row_for_a_different_pins_sha_is_refused(sandbox):
    """The row must vouch for THIS file, not for any file."""
    _write_pins(sandbox, vouch=False)
    _write_registry_row(sandbox, pins_sha256="f" * 64)
    with pytest.raises(G.ProtectedBlockError, match="LEAD vouches"):
        G.verify_freeze()


def test_g1_a_non_lead_written_by_is_refused(sandbox):
    p = _write_pins(sandbox, vouch=False)
    _write_registry_row(sandbox, written_by="chemtox", pins_sha256=G.file_sha(p))
    with pytest.raises(G.ProtectedBlockError, match="LEAD vouches"):
        G.verify_freeze()


def test_g1_rewriting_the_pins_after_the_row_is_refused(sandbox):
    """Edit the pins and the lead's row no longer matches, so the gate closes again."""
    _write_pins(sandbox)
    G.verify_freeze()
    p = G.pins_path()
    os.chmod(p, 0o644)
    pins = json.loads(p.read_text())
    pins["blocks"] = ["T1_test"]
    p.write_text(json.dumps(pins, indent=1, sort_keys=True) + "\n")
    os.chmod(p, 0o444)
    with pytest.raises(G.ProtectedBlockError, match="LEAD vouches"):
        G.verify_freeze()


def test_g1_a_missing_pinned_repo_file_entry_is_refused(sandbox):
    _write_pins(sandbox, mutate=lambda q: q["files"].pop(G.RULINGS, None))
    with pytest.raises(G.ProtectedBlockError, match="omit .* pinned repo file"):
        G.verify_freeze()


def test_g1_an_unknown_entry_is_refused(sandbox):
    _write_pins(sandbox, mutate=lambda q: q["files"].update({"docs/not_pinned.md": "a" * 64}))
    with pytest.raises(G.ProtectedBlockError, match="unknown entries"):
        G.verify_freeze()
    _write_pins(sandbox, mutate=lambda q: q.update({"surprise": 1}))
    with pytest.raises(G.ProtectedBlockError, match="unknown top-level"):
        G.verify_freeze()


def test_g1_a_null_value_other_than_labels_is_refused(sandbox):
    _write_pins(sandbox, mutate=lambda q: q["data"].update({str(A.ECOTOX): None}))
    with pytest.raises(G.ProtectedBlockError, match="null value"):
        G.verify_freeze()


def test_g1_a_non_hex_value_is_refused(sandbox):
    _write_pins(sandbox, mutate=lambda q: q["data"].update({str(A.SPLIT_PATH): "nope"}))
    with pytest.raises(G.ProtectedBlockError, match="not a 64-hex"):
        G.verify_freeze()


def test_g1_an_honest_pins_file_passes_and_returns_its_sha(sandbox):
    p = _write_pins(sandbox)
    out = G.verify_freeze()
    assert out["pins_sha256"] == G.file_sha(p)
    assert out["lead_row"]["workstream"] == "chemtox"


def test_g1_every_reader_records_the_pins_sha(aux_sandbox):
    _fully_unlocked(aux_sandbox)
    pool = G.load_training_pool()
    assert pool.attrs["pins_sha256"] == G.pins_sha256()
    _rows, rep = G.load_auxiliary_rows("H3", "T1")
    rec = json.loads(Path(rep["report_path"]).read_text())
    assert rec["pins_sha256"] == G.pins_sha256()


def test_g1_a_changed_data_file_still_closes_the_gate(sandbox):
    """M15's property: the hash comparison itself must still bite."""
    _write_pins(sandbox)
    G.verify_freeze()
    pd.DataFrame({"result_id": ["9"]}).to_parquet(A.PROSPECTIVE, index=False)
    with pytest.raises(G.ProtectedBlockError, match="on disk"):
        G.verify_freeze()


# ======================================================= G-5  the runs root and the claim row

def test_g5_the_runs_root_is_pinned(sandbox):
    _write_pins(sandbox)
    assert json.loads(G.pins_path().read_text())["runs_root"] == str(Path(G.RUNS).resolve())


def test_g5_a_different_runs_root_is_refused(sandbox, tmp_path, monkeypatch):
    _write_pins(sandbox)
    G.verify_freeze()
    monkeypatch.setattr(G, "RUNS", tmp_path / "other_runs")
    with pytest.raises(G.ProtectedBlockError, match="runs root"):
        G.verify_freeze()


def test_g5_a_copied_freeze_in_a_fresh_runs_root_cannot_take_a_second_look(
        sandbox, tmp_path, monkeypatch):
    """Probe P6: the exact escape the review demonstrated."""
    _fully_unlocked(sandbox)
    with pytest.raises(RuntimeError, match="boom"):
        with G.scored_pass(BLOCK):
            raise RuntimeError("boom")
    assert G._claim_rows(BLOCK) == 1
    # a fresh runs root with a copy of the lead's FREEZE
    fresh = tmp_path / "fresh_runs" / "chemtox" / "arch_v2" / BLOCK
    fresh.mkdir(parents=True)
    (fresh / "FREEZE").write_text(FAKE_FREEZE_TEXT)
    monkeypatch.setattr(G, "RUNS", fresh.parent)
    # the pinned runs root closes it first; even past that, the registry claim does
    with pytest.raises(G.ProtectedBlockError, match="runs root"):
        G.authorise(BLOCK)
    monkeypatch.setattr(G, "RUNS", Path(json.loads(G.pins_path().read_text())["runs_root"]))
    with pytest.raises(G.ProtectedBlockError, match="look claim"):
        G.authorise(BLOCK)


def test_g5_the_claim_row_does_not_block_the_same_look(sandbox):
    """The claim must stop a SECOND authorise, not the later steps of the first look."""
    _fully_unlocked(sandbox)
    with G.scored_pass(BLOCK) as look:
        rows = look.load()
        assert len(rows) == 1
        look.save_predictions(pd.DataFrame({"result_id": ["3"], "pred": [-6.0]}))
        look.finish({"rmse": 1.0})
    assert G._claim_rows(BLOCK) == 1
    assert (G.block_dir(BLOCK) / "SCORED.json").is_file()


def test_g5_the_claim_row_is_not_a_prereg_snapshot_row(sandbox):
    _fully_unlocked(sandbox)
    G.authorise(BLOCK)
    rows = [json.loads(l) for l in open(sandbox["registry"]) if l.strip()]
    claims = [r for r in rows if r.get("experiment") == G.CLAIM_EXPERIMENT]
    assert len(claims) == 1
    assert claims[0]["extra"]["block"] == BLOCK
    assert all(r.get("experiment") != G.PREREG_EXPERIMENT or r["extra"].get("written_by") == "lead"
               for r in rows), "M4 writes no chemtox/prereg_snapshot row"


# ========================================================= G-6  the protected-labels artefact

def test_g6_a_planted_labels_file_is_refused(sandbox):
    """Probe P7: labels planted before the look must never be served."""
    _fully_unlocked(sandbox)
    pd.DataFrame({"result_id": ["3", "4", "5"], G.TARGET: [0.0, 0.0, 0.0]}).to_parquet(
        G.LABELS_PATH, index=False)
    with pytest.raises(G.ProtectedBlockError, match="did not write it"):
        G.authorise(BLOCK)


def test_g6_labels_swapped_after_the_build_are_refused(sandbox):
    """Probe P7b: a swap after the build must be caught at the read."""
    _fully_unlocked(sandbox)
    G.authorise(BLOCK)
    os.chmod(G.LABELS_PATH, 0o644)
    pd.DataFrame({"result_id": ["3", "4", "5"], G.TARGET: [0.0, 0.0, 0.0]}).to_parquet(
        G.LABELS_PATH, index=False)
    with pytest.raises(G.ProtectedBlockError, match="swapped"):
        G.load_test_block(BLOCK)


def test_g6_the_build_is_recorded_and_hash_checked(sandbox):
    _fully_unlocked(sandbox)
    rec = G.authorise(BLOCK)
    marker = json.loads(G.LABELS_MARKER.read_text())
    assert marker["sha256"] == G.file_sha(G.LABELS_PATH)
    assert rec["protected_labels"]["sha256"] == marker["sha256"]
    assert stat.S_IMODE(Path(G.LABELS_PATH).stat().st_mode) == 0o444


def test_g6_a_second_block_reuses_the_recorded_artefact(sandbox):
    """The artefact covers all three blocks, so block 2 must not be refused."""
    _fully_unlocked(sandbox)
    G.authorise(BLOCK)
    sha = _write_freeze_file("T6a_test")
    _write_lead_record(sha, "T6a_test")
    _write_registry_row(sandbox, block="T6a_test", freeze_sha256=sha)
    # the lead record file now holds only T6a_test, so re-add T1 for completeness
    rec = G.authorise("T6a_test")
    assert rec["protected_labels"]["sha256"] == G.file_sha(G.LABELS_PATH)


def test_g6_load_test_block_checks_the_artefact_against_the_pinned_frame(sandbox):
    """A labels row whose value differs from the pinned frame is not the sealed truth.

    The claim record catches a swap first, so this test re-records every hash and leaves only the
    frame comparison able to fire.
    """
    _fully_unlocked(sandbox)
    G.authorise(BLOCK)
    os.chmod(G.LABELS_PATH, 0o644)
    pd.DataFrame({"result_id": ["3", "4", "5"], G.TARGET: [0.0, -7.0, -8.0]}).to_parquet(
        G.LABELS_PATH, index=False)
    new_sha = G.file_sha(G.LABELS_PATH)
    for marker in (G.LABELS_MARKER, G.block_dir(BLOCK) / "AUTHORISED.json"):
        os.chmod(marker, 0o644)
        rec = json.loads(marker.read_text())
        if "sha256" in rec:
            rec["sha256"] = new_sha
        if "protected_labels" in rec:
            rec["protected_labels"]["sha256"] = new_sha
        marker.write_text(json.dumps(rec))
    with pytest.raises(G.ProtectedBlockError, match="differs from"):
        G.load_test_block(BLOCK)


def test_g6_a_swap_is_also_caught_by_the_claim_record(sandbox):
    _fully_unlocked(sandbox)
    G.authorise(BLOCK)
    os.chmod(G.LABELS_PATH, 0o644)
    pd.DataFrame({"result_id": ["3", "4", "5"], G.TARGET: [0.0, 0.0, 0.0]}).to_parquet(
        G.LABELS_PATH, index=False)
    os.chmod(G.LABELS_MARKER, 0o644)
    m = json.loads(G.LABELS_MARKER.read_text())
    m["sha256"] = G.file_sha(G.LABELS_PATH)
    G.LABELS_MARKER.write_text(json.dumps(m))
    with pytest.raises(G.ProtectedBlockError, match="swapped after the look"):
        G.load_test_block(BLOCK)


# ======================================= G-9  the re-check is fail-closed and covers all seven

def _plant(aux_sandbox, monkeypatch, rows):
    """Make the frozen loader return extra rows, so only the re-check can catch them."""
    real = A.auxiliary_rows

    def leaky(target, split, **kw):
        out, rep = real(target, split, **kw)
        planted = _aux_frame(rows)
        return pd.concat([out, planted[out.columns]], ignore_index=True), rep

    monkeypatch.setattr(A, "auxiliary_rows", leaky)


@pytest.mark.parametrize("cas", [" 400", "400.0", "0400", "40-0"])
def test_g9_held_out_chemistry_in_other_forms_raises(aux_sandbox, monkeypatch, cas):
    """Probe P4b: the forms that slipped past the old bare astype(str) compare."""
    _fully_unlocked(aux_sandbox)
    _plant(aux_sandbox, monkeypatch, [("91", cas, "NOEC", False, 48.0, "Danio rerio")])
    with pytest.raises(G.ProtectedBlockError, match="held_out_unit_chemistry"):
        G.load_auxiliary_rows("H3", "T1")


@pytest.mark.parametrize("cas", [None, "ZZZ-unknown"])
def test_g9_unmapped_chemistry_fails_closed(aux_sandbox, monkeypatch, cas):
    """G-9: chemistry that maps to no hold-out unit RAISES; it is not merely counted."""
    _fully_unlocked(aux_sandbox)
    _plant(aux_sandbox, monkeypatch, [("92", cas, "NOEC", False, 48.0, "Danio rerio")])
    with pytest.raises(G.ProtectedBlockError, match="chemistry_not_in_split"):
        G.load_auxiliary_rows("H3", "T1")


def test_g9_a_t6a_sealed_cell_on_retained_chemistry_raises(aux_sandbox, monkeypatch):
    """The case that needs a T6a_test row: retained unit, sealed cell. Kills M9 and M12."""
    _fully_unlocked(aux_sandbox)
    _plant(aux_sandbox, monkeypatch, [("93", "500", "NOEC", False, 720.0, "Danio rerio")])
    with pytest.raises(G.ProtectedBlockError, match="same_cell_as_sealed_row"):
        G.load_auxiliary_rows("H3", "T1")


def test_g9_a_t6b_held_unit_raises(aux_sandbox, monkeypatch):
    """Kills M10: held units must come from T1_test AND T6b_test."""
    _fully_unlocked(aux_sandbox)
    _plant(aux_sandbox, monkeypatch, [("94", "400", "NOEC", False, 48.0, "Oncorhynchus mykiss")])
    with pytest.raises(G.ProtectedBlockError, match="held_out_unit_chemistry"):
        G.load_auxiliary_rows("H3", "T1")


def test_g9_a_sealed_result_id_raises_for_h2_too(aux_sandbox, monkeypatch):
    """G-9 asks for an H2 case, not only H3."""
    _fully_unlocked(aux_sandbox)
    _plant(aux_sandbox, monkeypatch, [("3", "100", "LC50", True, 48.0, "Danio rerio")])
    with pytest.raises(G.ProtectedBlockError, match="sealed_result_ids"):
        G.load_auxiliary_rows("H2", "T1")


def test_g9_a_t2_reserved_class_raises(aux_sandbox, monkeypatch):
    _fully_unlocked(aux_sandbox)
    rows = _aux_frame([("95", "100", "NOEC", False, 48.0, "Daphnia magna")])
    rows["class"] = "Insecta"
    real = A.auxiliary_rows

    def leaky(target, split, **kw):
        out, rep = real(target, split, **kw)
        return pd.concat([out, rows[out.columns]], ignore_index=True), rep

    monkeypatch.setattr(A, "auxiliary_rows", leaky)
    with pytest.raises(G.ProtectedBlockError, match="t2_reserved_classes"):
        G.load_auxiliary_rows("H3", "T1")


def test_g9_a_spent_prospective_id_raises(aux_sandbox, monkeypatch):
    _fully_unlocked(aux_sandbox)
    _plant(aux_sandbox, monkeypatch, [("999", "100", "NOEC", False, 48.0, "Danio rerio")])
    with pytest.raises(G.ProtectedBlockError, match="spent_prospective"):
        G.load_auxiliary_rows("H3", "T1")


def test_g9_a_spent_adore_id_raises(aux_sandbox, monkeypatch):
    _fully_unlocked(aux_sandbox)
    _plant(aux_sandbox, monkeypatch, [("998", "100", "NOEC", False, 48.0, "Danio rerio")])
    with pytest.raises(G.ProtectedBlockError, match="spent_adore"):
        G.load_auxiliary_rows("H3", "T1")


def test_g9_an_out_of_window_row_raises_on_t6(aux_sandbox, monkeypatch):
    """Kills M14's H2/H3 analogue: the T6 window is re-checked, not only filtered."""
    _fully_unlocked(aux_sandbox)
    _plant(aux_sandbox, monkeypatch, [("96", "100", "NOEC", False, 720.0, "Danio rerio")])
    with pytest.raises(G.ProtectedBlockError, match="outside_24_96h_for_T6"):
        G.load_auxiliary_rows("H3", "T6")


def test_g9_the_recheck_reports_all_seven_keys(aux_sandbox):
    _fully_unlocked(aux_sandbox)
    _rows, rep = G.load_auxiliary_rows("H3", "T1")
    assert set(G.SEVEN_EXCLUSIONS) <= set(rep["recheck"])


def test_g9_the_spent_adore_set_must_match_the_pinned_list(aux_sandbox):
    """G-7: a partition scored since the freeze closes every reader."""
    _fully_unlocked(aux_sandbox)
    G.load_auxiliary_rows("H3", "T1")
    os.chmod(A.SPENT_IDS_PATH, 0o644)
    Path(A.SPENT_IDS_PATH).write_text(json.dumps({"result_ids": ["998", "997"]}) + "\n")
    _write_pins(aux_sandbox)        # re-pin so only the set comparison can fail
    with pytest.raises(G.ProtectedBlockError, match="spent-ADORE set"):
        G.load_auxiliary_rows("H3", "T1")


# ================================================================ G-8  reports are never lost

def test_g8_a_second_report_gets_a_counter_and_the_first_survives(aux_sandbox):
    _fully_unlocked(aux_sandbox)
    _r1, rep1 = G.load_auxiliary_rows("H3", "T1")
    _r2, rep2 = G.load_auxiliary_rows("H3", "T1")
    p1, p2 = Path(rep1["report_path"]), Path(rep2["report_path"])
    assert p1 != p2 and p1.is_file() and p2.is_file()
    assert p2.name.endswith(".1.json")
    for p in (p1, p2):
        assert stat.S_IMODE(p.stat().st_mode) == 0o444


# ============================================================================ G-4  the targets

def test_g4_h2_and_h3_come_back_with_their_target(aux_sandbox):
    _fully_unlocked(aux_sandbox)
    for target in ("H2", "H3"):
        rows, rep = G.load_auxiliary_rows(target, "T1")
        assert rep["target_column"] == G.TARGET
        assert G.TARGET in rows.columns and rows[G.TARGET].notna().all()
        assert G.H4_TARGET in rep["targets_refused"]


def test_g4_the_reader_refuses_h4_and_h5(aux_sandbox):
    _fully_unlocked(aux_sandbox)
    for bad in ("H4", "H5", "H9"):
        with pytest.raises(G.ProtectedBlockError, match="handles H2 and H3"):
            G.load_auxiliary_rows(bad, "T1")


def test_g4_h5_comes_from_the_pool_only(aux_sandbox):
    _fully_unlocked(aux_sandbox)
    pool, rep = G.load_h5_pool_pairs()
    assert rep["source"] == "load_training_pool only"
    assert rep["supervised_acr_target"] is False
    assert sorted(pool["result_id"]) == ["1", "2"]
    with pytest.raises(G.ProtectedBlockError, match="takes no arguments"):
        G.load_h5_pool_pairs(rows=pool)


# ============================================================ G-10  the env switch is gone

def test_g10_the_env_switch_is_removed():
    body = Path("tests/test_chemtox_arch_v2.py").read_text()
    assert "M4_ALLOW_DIRECT_ECOTOX_READ" not in body
    assert "ruling N2" in body or "N2:" in body


def test_g10_the_guard_names_five_sanctioned_readers():
    doc = G.__doc__ or ""
    for name in ("load_training_pool", "build_protected_labels", "load_test_block",
                 "load_auxiliary_rows", "load_h4_rows"):
        assert name in doc, name
    assert "Five" in doc or "five" in doc


# ============================================== G-3  load_h4_rows: all seven exclusions, tested
#
# The review found H4 applied four of the seven and had no test at all (F3). This fixture builds a
# synthetic raw-ECOTOX world in which exactly one row is removed by each of the seven, so every
# counter is exercised and the mutations M7, M8, M9 and M14 are killed.

@pytest.fixture
def h4_sandbox(sandbox, tmp_path, monkeypatch):
    data = sandbox["data"]
    # species s1 = Danio rerio (Actinopterygii); s9 = an Insecta species (a T2 reserved class)
    frame = pd.DataFrame({
        "result_id": ["1", "2", "3", "4", "5"],
        "test_cas": ["100", "200", "300", "400", "500"],
        "chem_smiles": ["CCO"] * 5,
        "latin_name": ["Danio rerio"] * 5,
        "species_number": ["s1"] * 5,
        "class": ["Actinopterygii"] * 5,
        "duration_h": [48.0, 48.0, 48.0, 720.0, 720.0],
        G.TARGET: [-4.0, -5.0, -6.0, -7.0, -8.0],
    })
    extra = pd.DataFrame({
        "result_id": ["90"], "test_cas": ["777"], "chem_smiles": ["CCC"],
        "latin_name": ["Chironomus riparius"], "species_number": ["s9"], "class": ["Insecta"],
        "duration_h": [48.0], G.TARGET: [-3.0]})
    fp = data / "frame_h4.parquet"
    pd.concat([frame, extra], ignore_index=True).to_parquet(fp, index=False)
    monkeypatch.setattr(A, "ECOTOX", fp)

    # one raw BCF row removed by each of the seven, plus two that survive
    raw = [
        ("ok1", "t1", "BCF", "100", 48.0),      # survives
        ("ok2", "t2", "BCF", "100", 48.0),      # survives
        ("cls", "t9", "BCF", "100", 48.0),      # 1 T2 reserved class (species s9 -> Insecta)
        ("999", "t1", "BCF", "100", 48.0),      # 2 spent prospective id
        ("998", "t1", "BCF", "100", 48.0),      # 3 spent ADORE id
        ("hld", "t3", "BCF", "100", 48.0),      # 4 held-out unit (cas 300)
        ("unm", "t7", "BCF", "100", 48.0),      # 5 chemistry not in the split (cas 777)
        ("cel", "t5", "BCF", "100", 720.0),     # 6 sealed (CAS, species) pair (cas 500)
        ("lng", "t2", "BCF", "100", 720.0),     # 7 outside 24-96 h -> removed on T6 only
    ]
    results = pd.DataFrame({
        "result_id": [r[0] for r in raw],
        "test_id": [r[1] for r in raw],
        "endpoint": [r[2] for r in raw],
        "bcf1_mean_op": [""] * len(raw),
        "bcf1_mean": [r[3] for r in raw],
        "bcf1_unit": ["L/kg"] * len(raw),
        "obs_duration_mean": [str(r[4]) for r in raw],
        "obs_duration_unit": ["h"] * len(raw),
    })
    tests = pd.DataFrame({
        "test_id": ["t1", "t2", "t3", "t5", "t7", "t9"],
        "test_cas": ["100", "200", "300", "500", "777", "100"],
        "species_number": ["s1", "s1", "s1", "s1", "s1", "s9"],
    })
    results.to_parquet(sandbox["h4_results"], index=False)
    tests.to_parquet(sandbox["h4_tests"], index=False)
    return sandbox


def test_g3_h4_happy_path_returns_only_admissible_rows(h4_sandbox):
    _fully_unlocked(h4_sandbox)
    rows, rep = G.load_h4_rows("T1")
    assert sorted(rows["result_id"]) == ["lng", "ok1", "ok2"], sorted(rows["result_id"])
    assert rep["target_column"] == G.H4_TARGET
    assert G.H4_TARGET in rows.columns and rows[G.H4_TARGET].notna().all()
    assert rep["cell_rule"].startswith("pair")


@pytest.mark.parametrize("key", list(G.SEVEN_EXCLUSIONS))
def test_g3_every_one_of_the_seven_exclusions_removes_a_row(h4_sandbox, key):
    """F3: the old reader applied four of seven. Each must now have its own non-zero counter."""
    _fully_unlocked(h4_sandbox)
    track = "T6" if key == "outside_24_96h_for_T6" else "T1"
    _rows, rep = G.load_h4_rows(track)
    steps = rep["admissible_for_ablation_A5"][track]
    assert steps[f"removed_{key}"] == 1, (key, steps)
    assert steps["exclusions_applied"] == list(G.SEVEN_EXCLUSIONS)


def test_g3_h4_applies_the_t6_window(h4_sandbox):
    _fully_unlocked(h4_sandbox)
    rows, rep = G.load_h4_rows("T6")
    assert sorted(rows["result_id"]) == ["ok1", "ok2"]
    assert rows["duration_h"].max() <= A.DURATION_WINDOW[1]
    assert rep["admissible_for_ablation_A5"]["T6"]["removed_outside_24_96h_for_T6"] == 1


def test_g3_h4_recheck_reports_all_seven_as_zero_on_the_happy_path(h4_sandbox):
    _fully_unlocked(h4_sandbox)
    _rows, rep = G.load_h4_rows("T1")
    for key in G.SEVEN_EXCLUSIONS:
        assert rep["recheck"][key] == 0, key
    assert rep["recheck"]["sealed_result_ids"] == 0
    assert rep["recheck"]["cell_rule"] == "pair"


@pytest.mark.parametrize("planted,expect,blind", [
    # cas 400 is a HELD unit (T6b_test) whose sealed pair is (400, Danio rerio), so a
    # different species keeps the cell rule out of the way and isolates the unit rule
    (("p_held", "400", "Oncorhynchus mykiss", 48.0), "held_out_unit_chemistry",
     "held_units"),
    (("p_cell", "500", "Danio rerio", 720.0), "same_cell_as_sealed_row", "pairs"),
    (("p_unmapped", "ZZZ", "Danio rerio", 48.0), "chemistry_not_in_split", "map"),
    (("3", "100", "Danio rerio", 48.0), "sealed_result_ids", None),
    (("999", "100", "Danio rerio", 48.0), "spent_prospective", "prospective"),
    (("998", "100", "Danio rerio", 48.0), "spent_adore", "adore"),
])
def test_g3_h4_recheck_is_independent_of_its_exclusions(h4_sandbox, monkeypatch, planted,
                                                        expect, blind):
    """Kills M7, M8, M9, M10 and M12.

    Each case blinds ONLY the exclusion step under test, plants a row that the blinded step would
    let through, and requires the re-check — which rebuilds its own sets from the pinned files — to
    refuse. The re-check reports the first violation in the stamped order, so blinding one step at
    a time is what makes each assertion specific.
    """
    from sentinel.models.chemtox import arch_v2_h4 as H4
    _fully_unlocked(h4_sandbox)
    real_trim = H4.trim_implausible
    rid, cas, species, dur = planted

    def leaky(frame):
        kept, out = real_trim(frame)
        row = kept.iloc[[0]].copy()
        row["result_id"] = rid
        row["cas_norm"] = cas
        row["latin_name"] = species
        row["duration_h"] = dur
        row["class"] = "Actinopterygii"
        return pd.concat([kept, row], ignore_index=True), out

    monkeypatch.setattr(H4, "trim_implausible", leaky)

    honest_ctx, honest_spent = G._sealed_context(), G._spent_sets()
    if blind in ("held_units", "pairs", "map"):
        ctx = dict(honest_ctx)
        if blind == "held_units":
            ctx["held_units"] = set()
        elif blind == "pairs":
            ctx["sealed_result_ids"] = set()      # so the exclusions' sealed pairs are empty
        else:
            ctx["cas_unit_norm"] = {**honest_ctx["cas_unit_norm"], cas: 0}
        monkeypatch.setattr(G, "_sealed_context", lambda: ctx)
    elif blind in ("prospective", "adore"):
        spent = {**honest_spent, blind: set()}
        monkeypatch.setattr(G, "_spent_sets", lambda: spent)

    with pytest.raises(G.ProtectedBlockError, match=expect):
        G.load_h4_rows("T1")


def test_g3_h4_recheck_catches_a_long_duration_row_on_t6(h4_sandbox, monkeypatch):
    """Kills M14: the T6 window is re-checked, not only filtered."""
    from sentinel.models.chemtox import arch_v2_h4 as H4
    _fully_unlocked(h4_sandbox)
    real_trim = H4.trim_implausible

    def leaky(frame):
        kept, out = real_trim(frame)
        row = kept.iloc[[0]].copy()
        row["result_id"] = "p_long"
        row["cas_norm"] = "100"
        row["latin_name"] = "Danio rerio"
        row["duration_h"] = 26298.0
        row["class"] = "Actinopterygii"
        return pd.concat([kept, row], ignore_index=True), out

    monkeypatch.setattr(H4, "trim_implausible", leaky)
    # blind only the exclusion's window step; the re-check reads A.DURATION_WINDOW afresh
    real_cut_window = A.DURATION_WINDOW
    monkeypatch.setattr(G, "_h4_window_for_exclusion", lambda: (0.0, 1e9), raising=False)
    with pytest.raises(G.ProtectedBlockError, match="outside_24_96h_for_T6"):
        G.load_h4_rows("T6")
    assert A.DURATION_WINDOW == real_cut_window


def test_g3_h4_writes_a_report_with_the_pins_sha_and_input_hashes(h4_sandbox):
    _fully_unlocked(h4_sandbox)
    _rows, rep = G.load_h4_rows("T1")
    path = Path(rep["report_path"])
    rec = json.loads(path.read_text())
    assert rec["pins_sha256"] == G.pins_sha256()
    assert rec["pid"] == os.getpid()
    for key in ("split", "ecotox", "prospective", "adore_splits", "h4_results", "h4_tests"):
        assert len(rec["input_hashes"][key]) == 64, key
    assert stat.S_IMODE(path.stat().st_mode) == 0o444
    assert "SUPERSEDED" in rec["superseded_counts"]


def test_g3_h4_report_is_never_overwritten(h4_sandbox):
    _fully_unlocked(h4_sandbox)
    _r1, rep1 = G.load_h4_rows("T1")
    _r2, rep2 = G.load_h4_rows("T1")
    assert Path(rep1["report_path"]).is_file() and Path(rep2["report_path"]).is_file()
    assert rep1["report_path"] != rep2["report_path"]


def test_g3_h4_refuses_without_pins_and_refuses_caller_paths(h4_sandbox):
    with pytest.raises(G.ProtectedBlockError, match="not been stamped"):
        G.load_h4_rows("T1")
    _fully_unlocked(h4_sandbox)
    with pytest.raises(G.ProtectedBlockError, match="refused"):
        G.load_h4_rows("T1", h4_results="/tmp/x.parquet")
    with pytest.raises(G.ProtectedBlockError, match="refused"):
        G.load_h4_rows("T1", out="/tmp/out.json")
