# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Step 0 of the lead's stamp message (2026-11-10) for the M3 architecture round v2.

(a) the stamp gate and the deviations-file release gate at every entry point that can write a development number;
(b) the hardened protected loaders (lead ruling I-03); (c) the unit tests amendment §13 lists as owed that are not
in ``tests/test_microbe_niche.py``; (d) row-and-key-only counts of protected blocks (lead ruling I-04).

SYNTHETIC data only for every fit (declared; RESEARCH_PLAN R9): the simulator of ``tests/microbe_synth.py`` and
hand-made tables in ``tmp_path``. The tests that touch real files read ids, counts and file hashes only; the
protected label file is never opened, and no test writes to the real registry, runs folder or training log.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))

import microbe_synth as SY  # noqa: E402

from sentinel.core import paths  # noqa: E402
from sentinel.core import runlog  # noqa: E402
from sentinel.data import microbe_geo as G  # noqa: E402
from sentinel.models.microbe import archv2 as AV  # noqa: E402
from sentinel.models.microbe import baselines as B  # noqa: E402
from sentinel.models.microbe import features as F  # noqa: E402
from sentinel.models.microbe import niche as N  # noqa: E402
from sentinel.models.microbe import splits as SP  # noqa: E402
from sentinel.models.microbe import stamp as ST  # noqa: E402
from sentinel.models.microbe import testlook as TL  # noqa: E402
from sentinel.models.microbe import v4mr as VM  # noqa: E402
from sentinel.models.microbe.design import TARGET_NAMES  # noqa: E402

REPO = paths.REPO_ROOT
REAL_GEO = paths.PROCESSED / G.DATASET
REAL_WQ = paths.PROCESSED / "microbe_wq"
SMALL = {"xgb": {"n_estimators": 12, "learning_rate": 0.2, "max_depth": 3, "subsample": 0.8,
                 "colsample_bytree": 0.5, "min_child_weight": 1, "reg_lambda": 1.0},
         "rf": {"n_estimators": 15, "min_samples_leaf": 3, "max_features": 0.3}}
FAST = {"n_eff_grid": (30.0, 300.0), "newton_steps": 4}
GEO_AUX = [c for c in N.MASTER if c not in TARGET_NAMES]


@pytest.fixture(scope="module")
def synth():
    return SY.make_design(seed=1)


def _sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _stamp_row(rel: str, sha: str, rid: str, workstream: str = "lead", written_by: str | None = "lead") -> str:
    extra = {"sha256": {rel: sha}, **({"written_by": written_by} if written_by is not None else {})}
    return json.dumps({"id": rid, "workstream": workstream, "experiment": ST.STAMP_EXPERIMENT,
                       "timestamp": "2026-11-10T00:00:00", "extra": extra}) + "\n"


@pytest.fixture()
def fake_repo(tmp_path):
    """A repository root with an amendment file, and an empty registry."""
    root = tmp_path / "repo"
    (root / ST.DEVIATIONS_DIR).mkdir(parents=True)
    (root / ST.AMENDMENT).write_text("amendment text v1\n")
    reg = tmp_path / "fake_registry.jsonl"
    reg.write_text("")
    return root, reg


def _stamp(root: Path, reg: Path, rel: str = ST.AMENDMENT, rid: str = "stamp0001") -> None:
    with reg.open("a") as fh:
        fh.write(_stamp_row(rel, _sha(root / rel), rid))


# ============================================================================================ (a) stamp gates

def test_require_stamp_needs_the_registry_row_of_the_current_file_content(fake_repo):
    root, reg = fake_repo
    with pytest.raises(ST.StampError):
        ST.require_stamp("x", reg, root)
    with reg.open("a") as fh:                                # a row of another experiment or another hash is no stamp
        fh.write(json.dumps({"id": "a", "experiment": "other", "extra": {"sha256": {ST.AMENDMENT: _sha(root / ST.AMENDMENT)}}}) + "\n")
        fh.write(_stamp_row(ST.AMENDMENT, "0" * 64, "b"))
    with pytest.raises(ST.StampError):
        ST.require_stamp("x", reg, root)
    _stamp(root, reg)
    assert ST.require_stamp("x", reg, root)["row_id"] == "stamp0001"
    (root / ST.AMENDMENT).write_text("amendment text v2\n")  # an edit after the stamp closes the gate
    with pytest.raises(ST.StampError):
        ST.require_stamp("x", reg, root)


def test_only_rows_written_by_the_lead_are_stamps(fake_repo):
    """dev01 lead ruling D11: a row of the stamp experiment counts only with ``extra.written_by`` == "lead"."""
    root, reg = fake_repo
    sha = _sha(root / ST.AMENDMENT)
    with reg.open("a") as fh:                                # the right hash, but not marked as the lead's row
        fh.write(_stamp_row(ST.AMENDMENT, sha, "nowriter", written_by=None))
        fh.write(_stamp_row(ST.AMENDMENT, sha, "m3writer", workstream="M3-microbe", written_by="M3"))
        fh.write(_stamp_row(ST.AMENDMENT, sha, "leadws", workstream="lead", written_by="Lead"))
    st = ST.stamp_status(reg, root)
    assert not st["stamped"] and st["n_rows"] == 3 and st["row_id"] is None
    with pytest.raises(ST.StampError):
        ST.require_stamp("x", reg, root)
    _stamp(root, reg)
    assert ST.require_stamp("x", reg, root)["row_id"] == "stamp0001"
    rel = f"{ST.DEVIATIONS_DIR}/microbe_arch_v2_deviations_03.md"
    (root / rel).write_text("**Releases:** A0\n")
    with reg.open("a") as fh:                                # the same for a deviations file
        fh.write(_stamp_row(rel, _sha(root / rel), "dev_nowriter", written_by=None))
        fh.write(_stamp_row(rel, _sha(root / rel), "dev_m3", written_by="M3-microbe"))
    assert ST._stamped_shas(reg).get(rel) is None
    with pytest.raises(ST.StampError):
        ST.require_release("x", "A0", reg, root)
    with pytest.raises(ST.StampError):
        AV.gate("niche_s_a0", "lodo", "P", registry_path=reg, repo_root=root)
    _stamp(root, reg, rel, "dev_lead")
    assert ST.require_release("x", "A0", reg, root)["row_id"] == "dev_lead"
    assert AV.gate("niche_s_a0", "lodo", "P", registry_path=reg, repo_root=root)["release"]["row_id"] == "dev_lead"
    assert ST.WRITTEN_BY == "lead"


def test_the_real_stamp_rows_carry_the_lead_mark():
    """Both real rows carry the field (dev01 rulings D11); read-only check of the real registry, ids and hashes."""
    st = ST.stamp_status()
    if not st["stamped"]:
        pytest.skip("the amendment is not stamped in this checkout")
    rows = [json.loads(x) for x in paths.REGISTRY_PATH.read_text().splitlines() if ST.STAMP_EXPERIMENT in x]
    rows = [r for r in rows if r.get("experiment") == ST.STAMP_EXPERIMENT]
    assert rows and all((r.get("extra") or {}).get("written_by") == "lead" for r in rows)


@pytest.mark.parametrize("key", list(ST.RELEASE_KEYS))
def test_require_release_needs_a_stamped_deviations_file_that_declares_the_key(fake_repo, key):
    root, reg = fake_repo
    rel = f"{ST.DEVIATIONS_DIR}/microbe_arch_v2_deviations_07.md"
    other = "EA" if key != "EA" else "D2"
    with pytest.raises(ST.StampError):                       # nothing stamped at all
        ST.require_release("x", key, reg, root)
    _stamp(root, reg)
    with pytest.raises(ST.StampError):                       # amendment stamped, no deviations file
        ST.require_release("x", key, reg, root)
    (root / rel).write_text(f"# deviations\n\n**Releases:** {other}\n\ntext naming {key} elsewhere\n")
    _stamp(root, reg, rel, "dev_other")
    with pytest.raises(ST.StampError):                       # a stamped file that releases something else
        ST.require_release("x", key, reg, root)
    (root / rel).write_text(f"# deviations\n\n**Releases:** {other}, {key}\n")
    with pytest.raises(ST.StampError) as e:                  # declared, but this content carries no registry row
        ST.require_release("x", key, reg, root)
    assert rel in str(e.value)
    _stamp(root, reg, rel, "dev_ok")
    st = ST.require_release("x", key, reg, root)
    assert st["released"] and st["row_id"] == "dev_ok" and st["file"] == rel and st["sha256"] == _sha(root / rel)
    (root / rel).write_text(f"# deviations (edited)\n\n**Releases:** {other}, {key}\n")
    with pytest.raises(ST.StampError):                       # an edit after its stamp closes the gate again
        ST.require_release("x", key, reg, root)
    (root / rel).write_text(f"# deviations\n\n**Releases:** {other}, {key}\n")
    assert ST.require_release("x", key, reg, root)["released"]
    (root / ST.AMENDMENT).write_text("edited amendment\n")   # the amendment stamp is required as well
    with pytest.raises(ST.StampError):
        ST.require_release("x", key, reg, root)


def test_release_keys_are_closed_and_the_row_must_name_the_file(fake_repo):
    root, reg = fake_repo
    _stamp(root, reg)
    with pytest.raises(KeyError):
        ST.release_status("D9", reg, root)
    rel = f"{ST.DEVIATIONS_DIR}/microbe_arch_v2_deviations_01.md"
    (root / rel).write_text("**Releases:** D2\n")
    with reg.open("a") as fh:                                # the right hash under another path is no stamp
        fh.write(_stamp_row(f"{ST.DEVIATIONS_DIR}/microbe_arch_v2_deviations_02.md", _sha(root / rel), "wrongpath"))
    with pytest.raises(ST.StampError):
        ST.require_release("x", "D2", reg, root)
    (root / ST.DEVIATIONS_DIR / "notes_deviations.md").write_text("**Releases:** H4\n")   # not a deviations file name
    _stamp(root, reg, f"{ST.DEVIATIONS_DIR}/notes_deviations.md", "notdev")
    with pytest.raises(ST.StampError):
        ST.require_release("x", "H4", reg, root)
    assert ST.declared_releases("a\n**Releases:** D2, EA\nb\n**Releases:** H4\n") == ("D2", "EA", "H4")
    assert set(ST.RELEASE_KEYS) == {"D2", "EA", "H4", "GAS", "A0", "XGB_V4CLR_MO", "CALIBRATION", "NICHE_S_QMP",
                                    "GEO_POWER"}
    assert ST.declared_releases("Releases: D2\n** Releases:** EA\n") == ()


ALL_NAMES = tuple(N.NAMES) + tuple(AV.WA.NAMES) + tuple(VM.NAMES) + tuple(AV.BASE_TABLES)


def test_gate_refuses_every_model_of_the_round_without_the_stamp_row(fake_repo):
    root, reg = fake_repo
    for name in ALL_NAMES + ("niche_geo_phys", "xgb_chained_physics"):
        for fam in ("P", "G"):
            with pytest.raises(ST.StampError):
                AV.gate(name, "lodo", fam, registry_path=reg, repo_root=root, slot8_row="x")


def test_gate_rules_after_the_stamp(fake_repo):
    root, reg = fake_repo
    _stamp(root, reg)
    kw = dict(registry_path=reg, repo_root=root)
    for name in ("niche_s", "niche_s_tax", "xgb_nichez_context", "xgb_v4mr", "xgb_v4clr_3000", "xgb_gfo", "rf_v4clr",
                 "tabpfn_v4clr", "wa_tol", "wa_pls", "xgb_clr_stack", "xgb_v4clr", "context_rf"):
        for fam in ("P", "G"):
            assert AV.gate(name, "lodo", fam, **kw)["stamp"]["stamped"]
    # held by a deviations file: D2, the EA variant, and the post-stamp readings of deviations file 01
    for name in ("niche_s_a0", "xgb_v4clr_mo", "niche_s_qmp", "niche_s_doy", "niche_geo_phys", "niche_geo_phys_a2",
                 "niche_geo_phys_a3", "xgb_chained_physics",
                 "xgb_v4clr_mo_qmp"):                        # dev01 D1: a +QMP twin needs its base model's release
        with pytest.raises(ST.StampError):
            AV.gate(name, "lodo", "G", g10_row="x", **kw)
    assert AV.release_keys("xgb_v4clr_mo_qmp") == ("XGB_V4CLR_MO",) and AV.release_keys("niche_s_qmp") == ("NICHE_S_QMP",)
    assert AV.release_keys("xgb_v4clr_qmp") == () and AV.qmp_base("xgb_v4mr_stack_qmp") == "xgb_v4mr_stack"
    assert AV.qmp_base("niche_s") == "niche_s"
    rel = f"{ST.DEVIATIONS_DIR}/microbe_arch_v2_deviations_01.md"
    (root / rel).write_text("**Releases:** A0, XGB_V4CLR_MO, CALIBRATION, NICHE_S_QMP\n")
    _stamp(root, reg, rel, "dev01")
    for name in ("niche_s_a0", "xgb_v4clr_mo"):
        assert AV.gate(name, "lodo", "P", **kw)["release"]["row_id"] == "dev01"
    assert AV.gate("xgb_v4clr_mo_qmp", "lodo", "G", **kw)["releases"]["XGB_V4CLR_MO"]["row_id"] == "dev01"
    with pytest.raises(AV.GateError):                        # released, but the G10 choice is not logged (dev01 D8)
        AV.gate("niche_s_qmp", "lodo", "G", **kw)
    for name in ("niche_s_doy", "niche_geo_phys", "xgb_chained_physics"):      # EA and D2 stay closed
        with pytest.raises(ST.StampError):
            AV.gate(name, "lodo", "G", **kw)
    # released scheme, families, +QMP rules
    for scheme in ("loso", "primer_v4v5_to_v3v4", "primer_v3v4_to_v4v5", "release_002_to_001", "lodo_x001", "x"):
        with pytest.raises(AV.GateError):
            AV.gate("niche_s", scheme, "P", **kw)
    with pytest.raises(AV.GateError):
        AV.gate("niche_s", "lodo", "E", **kw)
    with pytest.raises(KeyError):
        AV.gate("not_a_model", "lodo", "P", **kw)
    for name in ("xgb_v4clr_qmp", "xgb_v4mr_qmp", "xgb_context_qmp", "xgb_v4clr_mo_qmp"):
        with pytest.raises(AV.GateError):
            AV.gate(name, "lodo", "P", **kw)                                   # GEO track only
        with pytest.raises(AV.GateError):
            AV.gate(name, "lodo", "G", confound_done=False, **kw)              # confound check first
        assert AV.gate(name, "lodo", "G", confound_done=True, **kw)
    # dev01 D8: niche_s_qmp needs the one logged G10 choice, and it must name G1
    def g10(rid, entry="G1", rows=("r1", "r2"), experiment=AV.G10_CHOICE_EXPERIMENT):
        return json.dumps({"id": rid, "workstream": "M3-microbe", "experiment": experiment,
                           "extra": {"g10_entry": entry, "rows": list(rows)}}) + "\n"

    with pytest.raises(AV.GateError):
        AV.gate("niche_s_qmp", "lodo", "G", g10_row="choice1", **kw)           # no such row
    before = reg.read_text()
    for bad in (g10("choice1", experiment="microbe_arch_v2_cv"), g10("choice1", entry="G7"),
                g10("choice1", entry="G10"), g10("choice1", rows=()), g10("other")):
        reg.write_text(before + bad)
        with pytest.raises(AV.GateError):
            AV.gate("niche_s_qmp", "lodo", "G", g10_row="choice1", **kw)
    reg.write_text(before + g10("choice1"))
    with pytest.raises(AV.GateError):
        AV.gate("niche_s_qmp", "lodo", "P", g10_row="choice1", **kw)           # GEO track only
    with pytest.raises(AV.GateError):
        AV.gate("niche_s_qmp", "lodo", "G", g10_row="choice1", confound_done=False, **kw)
    ok = AV.gate("niche_s_qmp", "lodo", "G", g10_row="choice1", **kw)
    assert ok["g10_row"]["g10_entry"] == "G1" and ok["g10_row"]["rows"] == ["r1", "r2"] and ok["release"]["released"]
    reg.write_text(before + g10("choice1") + g10("choice2"))                   # chosen once: two rows close the gate
    for rid in ("choice1", "choice2"):
        with pytest.raises(AV.GateError):
            AV.gate("niche_s_qmp", "lodo", "G", g10_row=rid, **kw)
    reg.write_text(before)
    assert AV.G10_ONLY == {"niche_s_qmp": "G1"}
    # slot #8 needs the registry row of the lead's written D1-gate evaluation
    with pytest.raises(AV.GateError):
        AV.gate("xgb_v4mr_stack", "lodo", "P", **kw)
    with pytest.raises(AV.GateError):
        AV.gate("xgb_v4mr_stack", "lodo", "P", slot8_row="nope", **kw)
    def row(rid, experiment=AV.D1_GATE_EXPERIMENT, **extra):
        return json.dumps({"id": rid, "workstream": "M3-microbe", "experiment": experiment, "extra": extra}) + "\n"

    with reg.open("a") as fh:
        fh.write(row("otherrow", "microbe/prereg_snapshot", d1_gate="fail", written_by="lead"))
        fh.write(row("nooutcome", written_by="lead"))
        fh.write(row("passrow", d1_gate="pass", written_by="lead"))
        fh.write(row("nowriter", d1_gate="fail"))
        fh.write(row("m3writer", d1_gate="fail", written_by="M3"))
        fh.write(row("failrow", d1_gate="fail", written_by="lead"))
    # not a gate evaluation / no outcome / gate passed (-> D2) / not marked as written by the lead
    for bad in ("otherrow", "nooutcome", "passrow", "nowriter", "m3writer"):
        with pytest.raises(AV.GateError):
            AV.gate("xgb_v4mr_stack", "lodo", "P", slot8_row=bad, **kw)
    ok = AV.gate("xgb_v4mr_stack", "lodo", "P", slot8_row="failrow", **kw)["slot8_row"]
    assert ok["d1_gate"] == "fail" and ok["written_by"] == "lead"
    # dev01 D1: the +QMP twin of the slot-#8 model needs the same row
    with pytest.raises(AV.GateError):
        AV.gate("xgb_v4mr_stack_qmp", "lodo", "G", **kw)
    for bad in ("nope", "otherrow", "nooutcome", "passrow", "nowriter", "m3writer"):
        with pytest.raises(AV.GateError):
            AV.gate("xgb_v4mr_stack_qmp", "lodo", "G", slot8_row=bad, **kw)
    with pytest.raises(AV.GateError):
        AV.gate("xgb_v4mr_stack_qmp", "lodo", "G", slot8_row="failrow", confound_done=False, **kw)
    with pytest.raises(AV.GateError):
        AV.gate("xgb_v4mr_stack_qmp", "lodo", "P", slot8_row="failrow", **kw)
    assert AV.gate("xgb_v4mr_stack_qmp", "lodo", "G", slot8_row="failrow", **kw)["slot8_row"]["d1_gate"] == "fail"


def test_no_m3_code_writes_a_d1_gate_evaluation():
    """Ruling B6: the gate is evaluated by the lead. The experiment name appears in the gate module only."""
    hits = []
    for folder in ("sentinel/models/microbe", "sentinel/data", "scripts/train", "scripts/eval", "scripts/data"):
        for f in (REPO / folder).glob("*.py"):
            if "microbe" in f.name or folder == "sentinel/models/microbe":
                txt = f.read_text()
                if AV.D1_GATE_EXPERIMENT in txt or "d1_gate_evaluation" in txt:
                    hits.append(str(f.relative_to(REPO)))
    assert hits == ["sentinel/models/microbe/archv2.py"]


def test_roles_registry_flags_and_eligibility():
    assert AV.role("niche_s") == "slot#6" and AV.role("xgb_v4mr") == "slot#7" and AV.role("xgb_v4mr_stack") == "slot#8"
    assert AV.role("niche_s_a0") == "ablation A0" and AV.role("niche_s_tax") == "ablation A1"
    assert AV.role("xgb_nichez_context") == "control" and AV.role("xgb_v4clr_qmp") == "+QMP twin"
    assert not AV.is_baseline("niche_s") and not AV.is_baseline("xgb_v4mr") and not AV.is_baseline("niche_s_qmp")
    assert AV.is_baseline("wa_tol") and AV.is_baseline("xgb_v4clr") and AV.is_baseline("xgb_nichez_context")
    assert set(AV.NOT_ELIGIBLE_HM3) == {"xgb_v4mr", "xgb_v4mr_stack"}                 # ruling B12
    assert {"niche_s_tax", "niche_s_a0", "xgb_nichez_context"} <= set(N.NEVER_SELECTABLE)
    assert not AV.stochastic("niche_s") and not AV.stochastic("wa_pls") and AV.stochastic("xgb_nichez_context")
    assert AV.stochastic("xgb_v4mr") and not AV.stochastic("ridge_v4clr") and AV.needs_gpu("tabpfn_v4clr")
    assert AV.RELEASED_SCHEMES == ("lodo",) and len(AV.FAMILIES["P"]) == 8 and len(AV.FAMILIES["G"]) == 8
    assert set(AV.RELEASE_KEY.values()) <= set(ST.RELEASE_KEYS)


# ------------------------------------------------------------------------------------- entry points (scripts)

def _load_script(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    """Registry, runs folder and training log in tmp_path; thread variables restored afterwards."""
    for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
              "PYTHONDONTWRITEBYTECODE", "HF_HUB_OFFLINE"):
        monkeypatch.setenv(v, os.environ.get(v, "1"))
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(paths, "RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setattr(paths, "REGISTRY_PATH", tmp_path / "registry.jsonl")
    monkeypatch.setattr(runlog, "TRAINING_LOG_MD", tmp_path / "TRAINING_LOG.md")
    monkeypatch.setattr(runlog, "TRAINING_LOG_JSONL", tmp_path / "training_runs.jsonl")
    paths.REGISTRY_PATH.write_text("")
    return tmp_path


def _stamp_real_amendment(reg: Path) -> None:
    """A stamp row for the amendment as it is on disk (the tmp registry holds nothing else)."""
    with reg.open("a") as fh:
        fh.write(_stamp_row(ST.AMENDMENT, ST.amendment_sha256(), "teststamp"))


class _Reached(Exception):
    """Raised by the patched data loader: the gates were passed."""


@pytest.fixture()
def runner(sandbox, monkeypatch):
    R = _load_script("scripts/train/microbe_arch_v2_cv.py", "microbe_arch_v2_cv_under_test")

    def no_data(*a, **k):
        raise _Reached()

    monkeypatch.setattr(R, "load", no_data)
    return R


DEV_ARGV = [
    ["--models", "niche_s"], ["--models", "niche_s", "--families", "G"], ["--models", "niche_s_tax"],
    ["--models", "niche_s_a0"], ["--models", "xgb_nichez_context"], ["--models", "xgb_v4mr"],
    ["--models", "xgb_v4clr_3000", "xgb_gfo"], ["--models", "rf_v4clr"], ["--models", "tabpfn_v4clr"],
    ["--models", "xgb_v4clr_mo"], ["--models", "wa_tol", "wa_pls"], ["--models", "xgb_clr_stack"],
    ["--models", "xgb_v4mr_stack", "--slot8-row", "x"],
    ["--models", "global_mean", "context_ridge", "context_rf", "xgb_v4clr", "--families", "G"],
    ["--models", "xgb_v4clr_qmp", "xgb_context_qmp", "niche_s_qmp", "--families", "G"],
    ["--models", "niche_s_doy"], ["--models", "niche_s_doy", "--families", "G"],
    ["--ensemble", "ridge_v4clr", "xgb_v4clr", "--families", "G"], ["--tier0-best", "--families", "G"],
    ["--calibration", "niche_s"], ["--calibration", "wa_tol", "--families", "G"],
    ["--models", "niche_s", "xgb_nichez_context", "--shard-folds", "0", "1"], ["--models", "niche_s", "--merge-shards"],
    ["--models", "niche_s_a0", "--shard-folds", "0"], ["--models", "niche_s_qmp", "--families", "G", "--g10-row", "x"],
    ["--models", "xgb_v4clr_mo_qmp", "xgb_v4mr_stack_qmp", "--families", "G", "--slot8-row", "x"],
]


@pytest.mark.parametrize("argv", DEV_ARGV, ids=lambda a: " ".join(a))
def test_every_development_entry_point_refuses_without_the_registry_row(runner, sandbox, argv):
    with pytest.raises(ST.StampError):                       # empty registry: no stamp row
        runner.main(argv)
    assert paths.REGISTRY_PATH.read_text() == ""             # nothing logged
    assert not (sandbox / "runs").exists()                   # nothing written, no data loaded (load is patched)


def test_runner_gates_after_the_stamp(runner, sandbox):
    _stamp_real_amendment(paths.REGISTRY_PATH)
    before = paths.REGISTRY_PATH.read_text()
    for argv in (["--models", "niche_s"], ["--models", "xgb_v4mr", "xgb_v4clr_3000", "xgb_gfo"],
                 ["--models", "xgb_v4clr", "--families", "G"], ["--tier0-best", "--families", "G"],
                 ["--ensemble", "ridge_v4clr", "xgb_v4clr", "--families", "G"]):
        with pytest.raises(_Reached):                        # released by the stamp: the gates pass
            runner.main(argv)
    for argv in (["--models", "niche_s_a0"], ["--models", "xgb_v4clr_mo"], ["--calibration", "niche_s"],
                 ["--models", "niche_s_qmp", "--families", "G"], ["--models", "niche_s_doy"],
                 ["--models", "niche_s", "niche_s_a0"], ["--models", "niche_s_a0", "--shard-folds", "0"],
                 ["--models", "niche_s_a0", "--merge-shards"],
                 ["--models", "xgb_v4clr_mo_qmp", "--families", "G"]):                # dev01 D1: the base's key
        with pytest.raises(ST.StampError):                   # need a stamped deviations file
            runner.main(argv)
    for argv in (["--models", "niche_s", "--schemes", "loso"], ["--models", "niche_s", "--schemes", "lodo", "lodo_x001"],
                 ["--models", "xgb_v4clr"],                  # baseline table on family P
                 ["--models", "xgb_v4clr_qmp", "--families", "G"],                    # no confound file yet
                 ["--models", "xgb_v4clr_qmp", "--families", "P", "G"],
                 ["--models", "xgb_v4mr_stack"], ["--models", "xgb_v4mr_stack", "--slot8-row", "teststamp_x"],
                 ["--models", "xgb_v4mr_stack_qmp", "--families", "G"]):
        with pytest.raises(AV.GateError):
            runner.main(argv)
    for argv in (["--models", "niche_s", "--shard-folds", "0"], ["--models", "niche_s", "--merge-shards"]):
        with pytest.raises(_Reached):                        # released by the stamp: the gates pass
            runner.main(argv)
    for argv in (["--shard-folds", "0"], ["--merge-shards"], ["--tier0-best", "--merge-shards"],
                 ["--models", "niche_s", "--shard-folds", "0", "--merge-shards"],
                 ["--models", "niche_s", "--shard-folds", "0", "--schemes", "lodo", "lodo"]):
        with pytest.raises(SystemExit):
            runner.main(argv)
    with pytest.raises(KeyError):
        runner.main(["--models", "not_a_model"])
    with pytest.raises(SystemExit):                          # exactly one mode
        runner.main(["--models", "niche_s", "--tier0-best"])
    assert paths.REGISTRY_PATH.read_text() == before
    # the confound file opens the +QMP gate (family G only)
    f = runner.confound_file("")
    f.parent.mkdir(parents=True)
    f.write_text("{}")
    with pytest.raises(_Reached):
        runner.main(["--models", "xgb_v4clr_qmp", "xgb_context_qmp", "--families", "G"])
    for argv in (["--models", "xgb_v4mr_stack_qmp", "--families", "G"],               # dev01 D1: slot-#8 row
                 ["--models", "xgb_v4mr_stack_qmp", "--families", "G", "--slot8-row", "teststamp"]):
        with pytest.raises(AV.GateError):
            runner.main(argv)
    assert "niche_s_qmp" in runner.COUNTED and runner.NICHE_THREADS == 1 and runner.MAX_THREADS == 6
    # the smoke test is train-only: allowed without a stamp, and it reaches the loader
    paths.REGISTRY_PATH.write_text("")
    with pytest.raises(_Reached):
        runner.main(["--smoke-train-only"])


def test_confound_script_refuses_without_the_registry_row(sandbox, monkeypatch):
    C = _load_script("scripts/eval/microbe_qmp_confound.py", "microbe_qmp_confound_under_test")

    def boom(*a, **k):
        raise AssertionError("data opened before the stamp check")

    monkeypatch.setattr(G, "dataset_dir", boom)
    with pytest.raises(ST.StampError):
        C.main([])
    assert paths.REGISTRY_PATH.read_text() == "" and not (sandbox / "runs").exists()


def test_confound_share_hand_worked_and_label_free(sandbox, monkeypatch):
    C = _load_script("scripts/eval/microbe_qmp_confound.py", "microbe_qmp_confound_under_test2")
    # two laboratories, values (1, 3) and (5, 7): between SS = 4 * 2^2 = 16, total SS = 20
    e = C.eta2(np.array([1.0, 3.0, 5.0, 7.0, np.nan, 9.0]), np.array(["a", "a", "b", "b", "b", None], object))
    assert e["n"] == 4 and e["n_groups"] == 2 and e["share"] == pytest.approx(0.8)
    assert np.isnan(C.eta2(np.array([1.0, 2.0, 3.0]), np.array(["a", "a", "a"], object))["share"])     # one group
    assert np.isnan(C.eta2(np.array([2.0, 2.0, 2.0]), np.array(["a", "b", "b"], object))["share"])     # no variance
    # the script on a synthetic dataset directory: reads the two QMP tables and ids only
    d = sandbox / "geo"
    d.mkdir()
    ids = [f"X{i}" for i in range(40)]
    rng = np.random.default_rng(0)
    lab = np.array(["labA", "labB"])[np.arange(40) % 2]
    feat = pd.DataFrame({"sample_id": ids})
    for c in G.QMP_FEATURES:
        feat[c] = rng.normal(0, 0.1, 40) + (c == "qmp_load") * 3.0 * (lab == "labB")
        feat[f"{c}_missing"] = False
    feat.to_parquet(d / "qmp_features.parquet")
    pd.DataFrame({"sample_id": ids, "n_dna_samples": 1, "n_qpcr_records": 1, "laboratory": lab,
                  "batch": [f"b{i % 5}" for i in range(40)]}).to_parquet(d / "qmp_meta.parquet")
    (d / "ids.json").write_text(json.dumps({"dev": ids[:20], "train1718": ids[20:30], "holdout": ids[30:35],
                                            "reserve": ids[35:]}))
    _stamp_real_amendment(paths.REGISTRY_PATH)
    monkeypatch.setattr(G, "dataset_dir", lambda root=None: d)
    opened = []
    real_read = pd.read_parquet
    monkeypatch.setattr(pd, "read_parquet", lambda p, *a, **k: (opened.append(Path(p).name), real_read(p, *a, **k))[1])
    assert C.main([]) == 0
    assert sorted(opened) == ["qmp_features.parquet", "qmp_meta.parquet"]            # no target table
    out = json.loads((sandbox / "runs" / "microbe" / "cv_archv2" / "qmp_confound.json").read_text())
    assert out["descriptive_only"] is True                                            # planted laboratory effect
    assert out["laboratory_share_of_load_index"]["development"] > 0.9
    assert out["shares"]["development"]["qmp_log_dna"]["laboratory"]["share"] < 0.3
    rows = [json.loads(x) for x in paths.REGISTRY_PATH.read_text().splitlines()][1:]
    assert rows and {r["experiment"] for r in rows} == {"microbe_arch_v2_qmp_confound"}
    assert {r["split"] for r in rows} == {"neon/development", "neon/all_blocks"}


def _qmp_dataset(d: Path, lab, load_shift) -> list:
    """SYNTHETIC QMP tables in ``d``: 40 events (20 development), ``qmp_load`` shifted by ``load_shift`` per event."""
    d.mkdir()
    ids = [f"X{i}" for i in range(40)]
    rng = np.random.default_rng(0)
    feat = pd.DataFrame({"sample_id": ids})
    for c in G.QMP_FEATURES:
        feat[c] = rng.normal(0, 0.1, 40) + (c == "qmp_load") * np.asarray(load_shift, float)
        feat[f"{c}_missing"] = False
    feat.to_parquet(d / "qmp_features.parquet")
    pd.DataFrame({"sample_id": ids, "n_dna_samples": 1, "n_qpcr_records": 1, "laboratory": lab,
                  "batch": [f"b{i % 5}" for i in range(40)]}).to_parquet(d / "qmp_meta.parquet")
    (d / "ids.json").write_text(json.dumps({"dev": ids[:20], "train1718": ids[20:30], "holdout": ids[30:35],
                                            "reserve": ids[35:]}))
    return ids


@pytest.mark.parametrize("case", ["one_laboratory", "laboratory_missing", "no_variance"])
def test_confound_rule_fires_on_a_non_finite_laboratory_share(sandbox, monkeypatch, case):
    """dev01 lead ruling D6: a laboratory share of ``qmp_load`` that is not finite on either event set means the
    rule has fired, although no finite share is over the threshold."""
    C = _load_script("scripts/eval/microbe_qmp_confound.py", f"microbe_qmp_confound_under_test_{case}")
    assert C.THRESHOLD == 0.5
    assert C.rule_fired({"development": 0.2, "all_blocks": 0.5}) is False             # "more than 50%"
    assert C.rule_fired({"development": 0.2, "all_blocks": 0.5001}) is True
    assert C.rule_fired({"development": float("nan"), "all_blocks": 0.1}) is True
    assert C.rule_fired({"development": 0.1, "all_blocks": float("nan")}) is True
    assert C.rule_fired({"development": float("inf"), "all_blocks": 0.1}) is True
    lab = np.array(["labA", "labB"], object)[np.arange(40) % 2]
    shift = np.zeros(40)
    if case == "one_laboratory":                             # development events: one group
        lab[:20] = "labA"
    elif case == "laboratory_missing":                       # development events: laboratory unknown
        lab[:20] = None
    d = sandbox / "geo"
    _qmp_dataset(d, lab, shift)
    if case == "no_variance":                                # development events: constant load index
        f = pd.read_parquet(d / "qmp_features.parquet")
        f.loc[:19, "qmp_load"] = 1.0
        f.to_parquet(d / "qmp_features.parquet")
    _stamp_real_amendment(paths.REGISTRY_PATH)
    monkeypatch.setattr(G, "dataset_dir", lambda root=None: d)
    assert C.main([]) == 0
    out = json.loads((sandbox / "runs" / "microbe" / "cv_archv2" / "qmp_confound.json").read_text())
    lab_share = out["laboratory_share_of_load_index"]
    assert np.isnan(lab_share["development"]) and lab_share["all_blocks"] < C.THRESHOLD
    assert out["descriptive_only"] is True
    # ruling D7: the script runs once and the first output stands
    p = sandbox / "runs" / "microbe" / "cv_archv2" / "qmp_confound.json"
    before, reg_before = p.read_bytes(), paths.REGISTRY_PATH.read_bytes()

    def boom(*a, **k):
        raise AssertionError("data opened although the output exists")

    monkeypatch.setattr(G, "dataset_dir", boom)
    with pytest.raises(RuntimeError, match="runs once"):
        C.main([])
    assert p.read_bytes() == before and paths.REGISTRY_PATH.read_bytes() == reg_before


# ------------------------------------------------------------------------- the runner on a SYNTHETIC design

def _folds(d):
    return SP.make_folds(d.samples, "lodo")


def test_runner_end_to_end_on_a_synthetic_design(sandbox, monkeypatch, synth):
    d, _ = synth
    R = _load_script("scripts/train/microbe_arch_v2_cv.py", "microbe_arch_v2_cv_under_test_e2e")
    folds = _folds(d)
    monkeypatch.setattr(R, "load", lambda scheme, AV_: (d, folds, [f.to_dict() for f in folds]))
    monkeypatch.setitem(B.DEFAULTS, "rf", SMALL["rf"])
    monkeypatch.setitem(B.DEFAULTS, "xgb", SMALL["xgb"])
    _stamp_real_amendment(paths.REGISTRY_PATH)
    base = sandbox / "runs" / "microbe" / "cv_archv2" / "lodo"

    def rows():
        return [json.loads(x) for x in paths.REGISTRY_PATH.read_text().splitlines()][1:]

    # a comparator on both families: one fit pass, a file and registry rows per family
    assert R.main(["--models", "wa_tol", "--n-boot", "20"]) == 0
    for fam in ("P", "G"):
        o = pd.read_parquet(base / fam / "wa_tol.parquet")
        assert list(o.columns) == list(AV.FAMILIES[fam]) + ["fold"] and len(o) == len(d.samples)
        assert (base / fam / "wa_tol_scores.json").is_file() and (base / fam / "wa_tol_diag.json").is_file()
    rr = rows()
    assert {r["experiment"] for r in rr} == {"microbe_arch_v2_cv"} and {r["split"] for r in rr} == {"neon_dev/lodo_cv"}
    assert {r["workstream"] for r in rr} == {"M3-microbe"} and all(r["is_baseline"] for r in rr)
    r2 = [r for r in rr if r["metric"] == "r2"]
    assert {r["target"] for r in r2} == set(AV.FAMILIES["P"]) | set(AV.FAMILIES["G"])
    assert all(r["config"]["stamp_row"] == "teststamp" and r["config"]["family"] in "PG" for r in r2)
    assert json.loads((base / "folds.json").read_text()) == [f.to_dict() for f in folds]
    # a comparator is not rerun silently
    with pytest.raises(AV.GateError):
        R.main(["--models", "wa_tol", "--n-boot", "20"])
    assert R.main(["--models", "wa_tol", "--n-boot", "20", "--families", "G", "--overwrite"]) == 0
    # a counted model is never rerun, with or without --overwrite
    (base / "P" / "niche_s.parquet").write_text("exists")
    n0 = len(rows())
    for extra in ([], ["--overwrite"]):
        with pytest.raises(AV.GateError):
            R.main(["--models", "niche_s", "--n-boot", "20", *extra])
    assert len(rows()) == n0
    # tier-0 tables on family G, tier0_best, an ensemble
    assert R.main(["--models", "global_mean", "context_ridge", "context_rf", "--families", "G", "--n-boot", "20",
                   "--seeds", "0", "1"]) == 0
    assert (base / "G" / "context_rf_seed1.parquet").is_file() and not (base / "G" / "context_ridge_seed1.parquet").exists()
    assert R.main(["--tier0-best", "--families", "G", "--n-boot", "20"]) == 0
    choice = json.loads((base / "G" / "tier0_best_choice.json").read_text())
    assert set(choice) == set(AV.FAMILIES["G"])
    for t, c in choice.items():
        assert c["model"] == max(c["lodo_r2"], key=c["lodo_r2"].get)
    assert R.main(["--ensemble", "context_ridge", "context_rf", "--families", "G", "--n-boot", "20"]) == 0
    e = pd.read_parquet(base / "G" / "ens_context_ridge+context_rf.parquet")
    a, b = (pd.read_parquet(base / "G" / f"{m}.parquet") for m in ("context_ridge", "context_rf"))
    assert np.allclose(e["ph"], (a["ph"] + b["ph"].reindex(a.index)) / 2)
    with pytest.raises(AV.GateError):
        R.main(["--tier0-best", "--families", "P", "--n-boot", "20"])
    # a failed run gets a registry row and no metric
    n0 = len(rows())
    monkeypatch.setattr(AV, "oof", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("planted failure")))
    with pytest.raises(RuntimeError, match="planted failure"):
        R.main(["--models", "wa_pls", "--n-boot", "20"])
    new = rows()[n0:]
    assert len(new) == 1 and new[0]["metric"] == "run_failed" and "planted failure" in new[0]["notes"]
    assert not (base / "P" / "wa_pls.parquet").exists()
    log = [json.loads(x) for x in (sandbox / "training_runs.jsonl").read_text().splitlines()]
    assert log[-1]["status"].endswith("Failed") and "wa_pls" in log[-1]["experiment"]


def test_runner_cpu_budget_stops_before_any_metric(sandbox, monkeypatch, synth):
    d, _ = synth
    R = _load_script("scripts/train/microbe_arch_v2_cv.py", "microbe_arch_v2_cv_under_test_budget")
    folds = _folds(d)
    monkeypatch.setattr(R, "load", lambda scheme, AV_: (d, folds, [f.to_dict() for f in folds]))
    _stamp_real_amendment(paths.REGISTRY_PATH)
    monkeypatch.setitem(R.CPU_BUDGET_H, "wa_tol", 6.0)
    clock = iter(np.arange(0.0, 1000.0, 2.5))                # 2.5 CPU-h per call: over 6 h after the third fold
    monkeypatch.setattr(R, "cpu_hours", lambda: float(next(clock)))
    with pytest.raises(RuntimeError, match="CPU budget"):
        R.main(["--models", "wa_tol", "--n-boot", "20"])
    rows = [json.loads(x) for x in paths.REGISTRY_PATH.read_text().splitlines()][1:]
    assert [r["metric"] for r in rows] == ["run_failed"]
    assert not list((sandbox / "runs").rglob("wa_tol.parquet"))
    # xgb_v4mr: 30.0 per family run since the lead ruling of 2026-11-10 §V-2/§V-4 (registry row fa48ff4c72bf)
    assert R.CPU_BUDGET_H["niche_s"] == 6.0 and R.CPU_BUDGET_H["xgb_v4mr"] == 30.0 and R.CPU_BUDGET_H["xgb_v4mr_stack"] == 12.0
    assert R.MAX_THREADS == 6 and set(R.COUNTED) >= {"niche_s", "xgb_v4mr", "xgb_v4mr_stack", "niche_s_a0", "niche_s_tax"}


def test_runner_calibration_needs_its_release_and_logs_the_diagnostic(sandbox, monkeypatch, synth):
    d, _ = synth
    R = _load_script("scripts/train/microbe_arch_v2_cv.py", "microbe_arch_v2_cv_under_test_calib")
    folds = _folds(d)
    monkeypatch.setattr(R, "load", lambda scheme, AV_: (d, folds, [f.to_dict() for f in folds]))
    _stamp_real_amendment(paths.REGISTRY_PATH)
    assert R.main(["--models", "wa_tol", "--n-boot", "20", "--families", "G"]) == 0
    with pytest.raises(ST.StampError):
        R.main(["--calibration", "wa_tol", "--families", "G"])
    released = {"released": True, "row_id": "dev01"}
    monkeypatch.setattr(ST, "require_release", lambda what, key, *a, **k: released if key == "CALIBRATION" else 1 / 0)
    assert R.main(["--calibration", "wa_tol", "--families", "G"]) == 0
    c = json.loads((sandbox / "runs" / "microbe" / "cv_archv2" / "lodo" / "G" / "wa_tol_calibration.json").read_text())
    assert set(c) == set(AV.FAMILIES["G"]) and np.isfinite(c["ph"]["slope_within_domain"])
    rows = [json.loads(x) for x in paths.REGISTRY_PATH.read_text().splitlines()]
    assert {"calib_slope_within_domain", "calib_sd_ratio", "calib_sd_ratio_within_domain"} <= {r.get("metric") for r in rows}


def _synthetic_runner(monkeypatch, synth, name):
    d, _ = synth
    R = _load_script("scripts/train/microbe_arch_v2_cv.py", name)
    folds = _folds(d)
    monkeypatch.setattr(R, "load", lambda scheme, AV_: (d, folds, [f.to_dict() for f in folds]))
    monkeypatch.setitem(B.DEFAULTS, "xgb", SMALL["xgb"])
    monkeypatch.setitem(N.DEFAULTS, "n_eff_grid", FAST["n_eff_grid"])     # speed only; the same for both paths
    monkeypatch.setitem(N.DEFAULTS, "newton_steps", FAST["newton_steps"])
    _stamp_real_amendment(paths.REGISTRY_PATH)
    return R, d, folds


def _registry_rows():
    return [json.loads(x) for x in paths.REGISTRY_PATH.read_text().splitlines()][1:]


def test_fold_shards_merge_to_exactly_the_unsharded_table(sandbox, monkeypatch, synth):
    """Lead ruling of 2026-11-10 (execution log E14): sharded output equals unsharded output, exact equality of the
    merged prediction table; a shard writes predictions and CPU time only. SYNTHETIC design."""
    R, d, folds = _synthetic_runner(monkeypatch, synth, "microbe_arch_v2_cv_under_test_shards")
    models = ["niche_s", "xgb_nichez_context"]
    common = ["--n-boot", "20", "--seeds", "0", "1"]
    N._FOLD_CACHE.clear()
    assert R.main(["--models", *models, *common, "--tag", "_whole"]) == 0
    n_whole = len(_registry_rows())
    assert n_whole > 0 and len(folds) == 6
    root = sandbox / "runs" / "microbe" / "cv_archv2_shard"
    sh = ["--models", *models, "--seeds", "0", "1", "--tag", "_shard"]
    for k, group in enumerate(([4, 0], [2], [5, 1, 3])):     # three shard processes, folds in no particular order
        N._FOLD_CACHE.clear()                                # a new process starts with an empty fold cache
        assert R.main([*sh, "--shard-folds", *map(str, group)]) == 0
        assert len(_registry_rows()) == n_whole              # no registry row, so no metric
        files = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
        assert files and all(f.startswith("lodo/_shards/") or f == "lodo/folds.json" for f in files)   # no score,
        assert not any(f.endswith("_scores.json") for f in files)                         # no merged table
        if k == 1:                                           # the merge needs every fold
            with pytest.raises(AV.GateError, match=r"\[1, 3, 5\]"):
                R.main([*sh, "--merge-shards", "--n-boot", "20"])
            assert len(_registry_rows()) == n_whole
    with pytest.raises(AV.GateError):                        # a fold is fitted once
        R.main([*sh, "--shard-folds", "2"])
    with pytest.raises(AV.GateError):                        # fold numbers are 0..n-1
        R.main([*sh, "--shard-folds", "6"])
    with pytest.raises(AV.GateError):                        # approved for the inversion-based models only
        R.main(["--models", "wa_tol", "--shard-folds", "0", "--tag", "_shard"])
    meta = json.loads((root / "lodo" / "_shards" / "niche_s" / "fold03.json").read_text())
    assert meta["n_threads"] == 1 and meta["fold"] == 3 and meta["fold_name"] == folds[3].name
    assert meta["cpu_hours"] >= 0 and meta["shard_folds"] == [1, 3, 5] and set(meta["code_sha256"]) == set(R.CODE_FILES)
    assert set(meta) == {"model", "fold", "fold_name", "n_folds", "seeds", "targets", "n_test", "cpu_hours", "wall_s",
                         "n_threads", "pid", "shard_folds", "config", "stamp_row", "code_sha256", "diag"}
    assert meta["diag"]["fold"] == 3 and meta["n_test"] == len(folds[3].test)
    with pytest.raises(AV.GateError):                        # another configuration than the shards'
        R.main(["--models", "xgb_nichez_context", "--seeds", "0", "1", "2", "--tag", "_shard", "--merge-shards"])
    assert len(_registry_rows()) == n_whole
    N._FOLD_CACHE.clear()
    assert R.main([*sh, "--merge-shards", "--n-boot", "20"]) == 0
    whole = sandbox / "runs" / "microbe" / "cv_archv2_whole" / "lodo"
    n_files = 0
    for fam in ("P", "G"):
        for f in ("niche_s.parquet", "xgb_nichez_context.parquet", "xgb_nichez_context_seed0.parquet",
                  "xgb_nichez_context_seed1.parquet"):
            a, b = pd.read_parquet(whole / fam / f), pd.read_parquet(root / "lodo" / fam / f)
            assert a.equals(b) and list(a.index) == list(b.index) and list(a.columns) == list(b.columns)
            assert a.to_numpy().tobytes() == b.to_numpy().tobytes()                   # bit for bit
            n_files += 1
        for name in models:                                  # same scores from the same table
            assert json.loads((whole / fam / f"{name}_scores.json").read_text()) == \
                json.loads((root / "lodo" / fam / f"{name}_scores.json").read_text())
        da, db = (json.loads((x / fam / "niche_s_diag.json").read_text()) for x in (whole, root / "lodo"))
        assert json.dumps(da) == json.dumps(db) and [x["fold"] for x in da] == list(range(6))
    assert n_files == 8
    rows = _registry_rows()
    assert len(rows) == 2 * n_whole
    merged = [r for r in rows[n_whole:] if r["metric"] == "r2"]
    assert merged and all(r["config"]["sharded"] and r["config"]["n_jobs"] == 1 for r in merged)
    assert json.dumps([(r["model"], r["metric"], r.get("target"), r["value"]) for r in rows[:n_whole]]) == \
        json.dumps([(r["model"], r["metric"], r.get("target"), r["value"]) for r in rows[n_whole:]])
    for extra in ([], ["--overwrite"]):                      # merged once: a counted model is never rerun
        with pytest.raises(AV.GateError):
            R.main([*sh, "--merge-shards", "--n-boot", "20", *extra])
    assert len(_registry_rows()) == 2 * n_whole


# Lead ruling B (execution log E20): fold shards of the long comparators / block models, with the family set each
# shard set runs on. xgb_v4mr and xgb_v4clr_3000 run ONE family per shard set, each with its own merge-and-score
# step and its own CPU figure (lead ruling on xgb_v4mr of 2026-11-10 §V-3, §V-10); xgb_clr_stack stays one shard
# set for both families (its auxiliary out-of-fold cost is shared, execution log E21).
SHARD_B = {"rf_clr_tuned": (["G"],), "xgb_clr_tuned": (["G"],), "xgb_clr_stack": (["P", "G"],),
           "xgb_v4mr": (["P"], ["G"]), "xgb_v4clr_3000": (["P"], ["G"])}
SHARD_CASES = [(m, f) for m, fs in SHARD_B.items() for f in fs]


@pytest.mark.parametrize("model,fams", SHARD_CASES, ids=lambda x: x if isinstance(x, str) else "".join(x))
def test_comparator_fold_shards_merge_to_exactly_the_unsharded_table(sandbox, monkeypatch, synth, model, fams):
    """Lead ruling B (execution log E20): for EACH sharded comparator the merged table equals the unsharded table
    bit for bit, whatever the grouping and the order of the folds; a shard writes predictions and CPU time only;
    one model per call; one thread per process. SYNTHETIC design."""
    R, d, folds = _synthetic_runner(monkeypatch, synth,
                                    f"microbe_arch_v2_cv_under_test_shards_{model}_{''.join(fams)}")
    assert set(R.SHARD_COMPARATORS) == set(SHARD_B) and all(R.n_threads(m, AV) == 1 for m in SHARD_B)
    assert R.n_threads("xgb_gfo", AV) == R.n_threads("xgb_clr", AV) == R.MAX_THREADS == 6
    monkeypatch.setitem(B.DEFAULTS, "rf", SMALL["rf"])
    for k in list(B.TUNE_GRIDS):                             # speed only; the same grid for both paths
        monkeypatch.setitem(B.TUNE_GRIDS, k, B.TUNE_GRIDS[k][:2])
    common = ["--models", model, "--families", *fams, "--seeds", "0", "1"]
    # the shard set of a comparator carries its family set; the inversion-based models keep one set for both
    assert R.shard_set(model, fams) == f"{model}__{''.join(fams)}" and R.shard_set("niche_s", fams) == "niche_s"
    sdir = R.shard_dir("lodo", model, "_shard", fams)

    def new_process():                                       # a new process starts with empty per-fold caches
        VM._STACK_CACHE.clear()
        VM._MO_CACHE.clear()

    new_process()
    assert R.main([*common, "--n-boot", "20", "--tag", "_whole"]) == 0
    whole_rows = _registry_rows()
    n_whole = len(whole_rows)
    assert n_whole > 0 and len(folds) == 6
    assert all(r["config"]["n_jobs"] == 1 for r in whole_rows if r["metric"] == "r2")
    root = sandbox / "runs" / "microbe" / "cv_archv2_shard"
    sh = [*common, "--tag", "_shard"]
    other = next((m for m in SHARD_B if m != model and any(f == fams for f in SHARD_B[m])),
                 "xgb_v4mr" if model != "xgb_v4mr" else "xgb_v4clr_3000")   # gates on the same families
    for bad in (["--models", model, other], ["--models", model, "niche_s"]):      # one model at a time
        with pytest.raises(AV.GateError, match="one model at a time"):
            R.main([*bad, "--families", *fams, "--seeds", "0", "1", "--tag", "_shard", "--shard-folds", "0"])
    with pytest.raises(AV.GateError, match="overwrite"):
        R.main([*sh, "--shard-folds", "0", "--overwrite"])
    assert not root.exists() or not list(root.rglob("*.parquet"))
    for k, group in enumerate(([3, 5], [0], [4, 2, 1])):     # three shard processes, folds in no particular order
        new_process()
        assert R.main([*sh, "--shard-folds", *map(str, group)]) == 0
        assert len(_registry_rows()) == n_whole              # no registry row, so no metric
        files = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
        assert files and all(f.startswith(f"lodo/_shards/{R.shard_set(model, fams)}/") or f == "lodo/folds.json"
                             for f in files)
        assert not any(f.endswith("_scores.json") for f in files)
        if k == 1:                                           # the merge needs every fold
            with pytest.raises(AV.GateError, match=r"\[1, 2, 4\]"):
                R.main([*sh, "--merge-shards", "--n-boot", "20"])
            assert len(_registry_rows()) == n_whole
    with pytest.raises(AV.GateError):                        # a fold is fitted once
        R.main([*sh, "--shard-folds", "4"])
    meta = json.loads((sdir / "fold02.json").read_text())
    assert meta["n_threads"] == 1 and meta["config"]["n_jobs"] == 1 and meta["fold"] == 2
    assert meta["shard_folds"] == [1, 2, 4] and meta["seeds"] == [0, 1] and meta["cpu_hours"] >= 0
    assert meta["targets"] == [t for fam in fams for t in AV.FAMILIES[fam]] and meta["diag"] is None
    with pytest.raises(AV.GateError):                        # another configuration than the shards'
        R.main(["--models", model, "--families", *fams, "--seeds", "0", "1", "2", "--tag", "_shard",
                "--merge-shards"])
    with pytest.raises(AV.GateError, match="overwrite"):
        R.main([*sh, "--merge-shards", "--n-boot", "20", "--overwrite"])
    assert len(_registry_rows()) == n_whole
    new_process()
    assert R.main([*sh, "--merge-shards", "--n-boot", "20"]) == 0
    whole = sandbox / "runs" / "microbe" / "cv_archv2_whole" / "lodo"
    for fam in fams:
        for f in (f"{model}.parquet", f"{model}_seed0.parquet", f"{model}_seed1.parquet"):
            a, b = pd.read_parquet(whole / fam / f), pd.read_parquet(root / "lodo" / fam / f)
            assert a.equals(b) and list(a.index) == list(b.index) and list(a.columns) == list(b.columns)
            assert a.to_numpy().tobytes() == b.to_numpy().tobytes()                   # bit for bit
            assert np.isfinite(a[list(AV.FAMILIES[fam])].to_numpy()).all()
        s0, s1 = (pd.read_parquet(whole / fam / f"{model}_seed{s}.parquet") for s in (0, 1))
        assert not s0.equals(s1)                             # the two seeds are two fits
        assert json.loads((whole / fam / f"{model}_scores.json").read_text()) == \
            json.loads((root / "lodo" / fam / f"{model}_scores.json").read_text())
    rows = _registry_rows()
    assert len(rows) == 2 * n_whole
    merged = [r for r in rows[n_whole:] if r["metric"] == "r2"]
    assert merged and all(r["config"]["sharded"] and r["config"]["n_jobs"] == 1 and
                          r["config"]["n_shard_processes"] == 1 for r in merged)     # one test process
    assert json.dumps([(r["model"], r["metric"], r.get("target"), r["value"]) for r in rows[:n_whole]]) == \
        json.dumps([(r["model"], r["metric"], r.get("target"), r["value"]) for r in rows[n_whole:]])
    with pytest.raises(AV.GateError):                        # merged once
        R.main([*sh, "--merge-shards", "--n-boot", "20"])
    assert len(_registry_rows()) == 2 * n_whole
    if len(fams) == 1 and AV.backend(model) != "base":   # the tuned grids of the existing track run on G only
        # the other family is a shard set of its own and is not blocked (lead ruling of 2026-11-10 §V-3)
        n_before = len(_registry_rows())
        o_fam = "G" if fams[0] == "P" else "P"
        new_process()
        assert R.main(["--models", model, "--families", o_fam, "--seeds", "0", "1", "--tag", "_shard",
                       "--shard-folds", "0"]) == 0
        o_dir = R.shard_dir("lodo", model, "_shard", [o_fam])
        assert o_dir != sdir and (o_dir / "fold00.json").is_file()
        assert json.loads((o_dir / "fold00.json").read_text())["targets"] == list(AV.FAMILIES[o_fam])
        assert len(_registry_rows()) == n_before             # a shard writes no metric


def test_merge_refuses_to_score_when_the_shards_exceed_the_cpu_budget(sandbox, monkeypatch, synth):
    R, d, folds = _synthetic_runner(monkeypatch, synth, "microbe_arch_v2_cv_under_test_shard_budget")
    N._FOLD_CACHE.clear()
    clock = iter(np.arange(0.0, 1000.0, 1.1))                # 1.1 CPU-h per fold: 6.6 CPU-h over the 6 folds
    monkeypatch.setattr(R, "cpu_hours", lambda: float(next(clock)))
    assert R.main(["--models", "niche_s", "--shard-folds", "0", "1", "2", "--tag", "_over"]) == 0
    assert R.main(["--models", "niche_s", "--shard-folds", "3", "4", "5", "--tag", "_over"]) == 0
    assert _registry_rows() == []
    real_read = pd.read_parquet

    def boom(*a, **k):
        raise AssertionError("a prediction file was read although the CPU budget is exceeded")

    monkeypatch.setattr(pd, "read_parquet", boom)
    with pytest.raises(RuntimeError, match="CPU budget"):
        R.main(["--models", "niche_s", "--merge-shards", "--n-boot", "20", "--tag", "_over"])
    rows = _registry_rows()
    assert [r["metric"] for r in rows] == ["run_failed"] and "6.60 CPU-h" in rows[0]["notes"]
    assert not list((sandbox / "runs").rglob("niche_s.parquet")) and not list((sandbox / "runs").rglob("*_scores.json"))
    monkeypatch.setattr(pd, "read_parquet", real_read)
    # within the budget: 0.9 CPU-h per fold, 5.4 CPU-h in total; one code state is required
    clock = iter(np.arange(0.0, 1000.0, 0.9))
    assert R.main(["--models", "niche_s", "--shard-folds", *map(str, range(6)), "--tag", "_under"]) == 0
    f4 = sandbox / "runs" / "microbe" / "cv_archv2_under" / "lodo" / "_shards" / "niche_s" / "fold04.json"
    good = f4.read_text()
    m = json.loads(good)
    m["code_sha256"] = {k: "0" * 64 for k in m["code_sha256"]}
    f4.write_text(json.dumps(m))
    with pytest.raises(AV.GateError, match="code_sha256"):
        R.main(["--models", "niche_s", "--merge-shards", "--n-boot", "20", "--tag", "_under"])
    f4.write_text(good)
    assert R.main(["--models", "niche_s", "--merge-shards", "--n-boot", "20", "--tag", "_under"]) == 0
    new = _registry_rows()[1:]
    assert new and all(r["metric"] != "run_failed" for r in new) and any("5.40 CPU-h" in r["notes"] for r in new)
    assert R.CPU_BUDGET_H["niche_s"] == 6.0 and R.CPU_BUDGET_H["niche_s_a0"] == 6.0 and R.CPU_BUDGET_H["niche_s_tax"] == 6.0
    # lead ruling on xgb_v4mr of 2026-11-10 §V-2/§V-4 (registry row fa48ff4c72bf): 30 CPU-h per family run for
    # xgb_v4mr, no other entry changed
    assert R.CPU_BUDGET_H == {"niche_s": 6.0, "niche_s_tax": 6.0, "niche_s_a0": 6.0, "niche_s_qmp": 6.0,
                              "xgb_v4mr": 30.0, "xgb_v4mr_stack": 12.0}


def test_geo_power_script_refuses_without_its_release(sandbox, monkeypatch):
    P = _load_script("scripts/eval/microbe_geo_power.py", "microbe_geo_power_under_test")
    from sentinel.models.microbe import design as D

    def boom(*a, **k):
        raise AssertionError("data opened before the release check")

    monkeypatch.setattr(D, "load_design", boom)
    with pytest.raises(ST.StampError):                       # no stamp at all
        P.main([])
    _stamp_real_amendment(paths.REGISTRY_PATH)
    with pytest.raises(ST.StampError):                       # the amendment stamp alone does not release GEO_POWER
        P.main([])
    monkeypatch.setattr(ST, "require_release", lambda what, key, *a, **k: {"released": True, "row_id": "dev01",
                                                                           "file": "f"} if key == "GEO_POWER" else 1 / 0)
    monkeypatch.setattr(paths, "REPO_ROOT", sandbox / "repo")
    with pytest.raises(AV.GateError) as e:                   # released, but no GEO entry is finished
        P.main([])
    assert all(g in str(e.value) for g in P.ENTRIES) and all(c in str(e.value) for c in P.COMPARATORS)
    assert not (sandbox / "repo").exists() and len(paths.REGISTRY_PATH.read_text().splitlines()) == 1


def test_geo_power_not_run_references(tmp_path):
    """dev01 lead ruling D4: 'reported as not run' needs a registry row id or a heading of the execution log."""
    P = _load_script("scripts/eval/microbe_geo_power.py", "microbe_geo_power_under_test_refs")
    reg, log = tmp_path / "registry.jsonl", tmp_path / "log.md"
    reg.write_text(json.dumps({"id": "abc123def456", "metric": "run_failed"}) + "\nnot json abc999\n")
    log.write_text("# log\n\n### E21 — xgb_v4mr not run (budget)\n\ntext E22\n\n### E23\n")
    assert P.not_run_reference("abc123def456", reg, log) and P.not_run_reference("log:E21", reg, log)
    assert P.not_run_reference("log:E23", reg, log)
    for bad in ("abc123", "abc999", "", "log:", "log:E2", "log:E22", "E21", "log:E24"):
        assert not P.not_run_reference(bad, reg, log)
    assert not P.not_run_reference("abc123def456", tmp_path / "none.jsonl", log)
    assert not P.not_run_reference("log:E21", reg, tmp_path / "none.md")


def test_geo_power_file_hand_checked_and_written_once(sandbox, monkeypatch, synth):
    """Deviations 01 P1 as changed by the dev01 lead rulings D2 (scaling sqrt(1,215/214) for every target), D3 (every
    §7.4 comparator; the two context comparators for G1 and G8) and D4 (SE from the interval width; an entry
    reported as not run contributes no pair). SYNTHETIC predictions."""
    from scipy.stats import norm
    from sentinel.models.microbe import design as D
    from sentinel.models.microbe import evaluate as E

    d, _ = synth
    P = _load_script("scripts/eval/microbe_geo_power.py", "microbe_geo_power_under_test2")
    assert abs(P.mdd(0.1) - (norm.ppf(1 - 0.05 / 16) + norm.ppf(0.8)) * 0.1) < 1e-12 and abs(P.mdd(0.1) - 0.357599) < 1e-5
    assert P.ENTRIES == ("G1", "G3", "G7", "G8", "G9")
    assert P.COMPARATORS == ("tier0_best", "rf_clr", "xgb_clr", "rf_clr_tuned", "xgb_clr_tuned")
    assert P.CONTEXT_COMPARATORS == ("rf_clr_context", "xgb_clr_context") and P.CONTEXT_ENTRIES == ("G1", "G8")
    assert P.comparators_of("G1") == P.COMPARATORS + P.CONTEXT_COMPARATORS == P.comparators_of("G8")
    assert P.comparators_of("G3") == P.comparators_of("G7") == P.comparators_of("G9") == P.COMPARATORS
    assert (P.N_BOOT, P.BOOT_SEED, P.HOLDOUT_EVENTS, P.DEV_EVENTS, P.THRESHOLD) == (1000, 0, 214, 1215, 0.30)
    monkeypatch.setattr(P, "N_BOOT", 40)
    monkeypatch.setattr(P.paired_se, "__kwdefaults__", {"n_boot": 40, "seed": 0})
    monkeypatch.setattr(D, "load_design", lambda *a, **k: d)
    monkeypatch.setattr(AV, "attach_master", lambda design, root=None: design)
    monkeypatch.setattr(ST, "require_release", lambda what, key, *a, **k: {"released": True, "row_id": "dev01",
                                                                           "file": "f"} if key == "GEO_POWER" else 1 / 0)
    targets = list(AV.FAMILIES["G"])
    S = d.samples
    region, site = S["domain_id"].astype(str).to_numpy(), S["site_id"].astype(str).to_numpy()
    out = sandbox / "runs" / "microbe" / "cv_archv2" / "lodo" / "G"
    out.mkdir(parents=True)
    rng = np.random.default_rng(3)
    comps = list(P.COMPARATORS + P.CONTEXT_COMPARATORS)
    run = ("G1", "G7", "G8", "G9")                           # G3 (xgb_v4mr) is not run in the first part
    names = [AV.GEO_ENTRIES[g] for g in P.ENTRIES] + comps
    frames = {}
    for k, n in enumerate(names):
        frames[n] = S[targets].astype(float) + rng.normal(0, 0.3 + 0.1 * k, (len(S), len(targets)))
        if n != "xgb_v4mr":
            frames[n].to_parquet(out / f"{n}.parquet")
            if n != "tier0_best":                            # tier0_best has no score file yet: not finished
                (out / f"{n}_scores.json").write_text("{}")
    s0 = frames["xgb_v4clr"] + 1.0                           # a stored seed-0 file is the one that is used
    s0.to_parquet(out / "xgb_v4clr_seed0.parquet")
    frames["xgb_v4clr"] = s0

    def check(res, entries, not_run):
        want = [f"{g}:{AV.GEO_ENTRIES[g]} vs {c}" for g in entries for c in P.comparators_of(g)]
        assert res["n_pairs"] == len(want) == 5 * len(entries) + 2 * len([g for g in entries if g in ("G1", "G8")])
        assert res["entries"] == {g: AV.GEO_ENTRIES[g] for g in entries} and res["not_run"] == not_run
        assert res["comparators"] == comps and res["n_events"] == len(S)
        assert res["prediction_files"]["G7:xgb_v4clr"] == "xgb_v4clr_seed0.parquet"
        assert res["prediction_files"]["G1:niche_s"] == "niche_s.parquet"
        scale = np.sqrt(1215 / 214)
        for t in targets:
            r = res["power"][t]
            assert list(r["pairs"]) == want
            ses = {}
            for g in entries:
                n = AV.GEO_ENTRIES[g]
                for c in P.comparators_of(g):
                    o = E.paired_vs(S[t].to_numpy(float), frames[n][t], frames[c][t], region, site, n_boot=40, seed=0)
                    ses[f"{g}:{n} vs {c}"] = (o["ci_high"] - o["ci_low"]) / (2 * norm.ppf(0.975))
                    assert abs(r["pairs"][f"{g}:{n} vs {c}"]["se"] - ses[f"{g}:{n} vs {c}"]) < 1e-12
            worst = max(ses, key=ses.get)
            assert r["pair_of_max"] == worst and abs(r["paired_se_site_max"] - ses[worst]) < 1e-12
            assert r["n_dev"] == int(S[t].notna().sum()) and abs(r["holdout_scale"] - scale) < 1e-12   # D2
            hold = (norm.ppf(1 - 0.05 / 16) + norm.ppf(0.8)) * ses[worst] * scale
            assert abs(r["mdd_holdout_alpha_0.05/8"] - hold) < 1e-12 and r["underpowered"] == bool(hold > 0.30)
            assert abs(r["mdd_dev_alpha_0.05/8"] - P.mdd(ses[worst])) < 1e-12
        assert res["underpowered_targets"] == [t for t in targets if res["power"][t]["underpowered"]]

    # ---- part 1: G3 reported as not run
    repo = sandbox / "repo"
    monkeypatch.setattr(paths, "REPO_ROOT", repo)
    (repo / "docs" / "preregistration").mkdir(parents=True)
    (repo / P.LOG_REL).write_text("# log\n\n### E99 — xgb_v4mr not run\n")
    with pytest.raises(AV.GateError) as e:
        P.main([])
    assert "G3 (xgb_v4mr)" in str(e.value) and "tier0_best" in str(e.value) and "G1 (" not in str(e.value)
    (out / "tier0_best_scores.json").write_text("{}")
    for argv in ([],                                         # G3 neither finished nor reported: blocks
                 ["--not-run", "G3=log:E98"], ["--not-run", "G3=nosuchrow"], ["--not-run", "G3="],
                 ["--not-run", "G2=log:E99"], ["--not-run", "G5=log:E99"],
                 ["--not-run", "G3=log:E99", "G1=log:E99"]):  # G1 is finished: it cannot be reported as not run
        with pytest.raises(AV.GateError):
            P.main(argv)
    (out / "rf_clr_tuned_scores.json").unlink()              # a missing comparator blocks as well
    with pytest.raises(AV.GateError, match="rf_clr_tuned"):
        P.main(["--not-run", "G3=log:E99"])
    (out / "rf_clr_tuned_scores.json").write_text("{}")
    assert not (repo / P.OUT_REL).exists() and len(paths.REGISTRY_PATH.read_text().splitlines()) == 0
    assert P.main(["--not-run", "G3=log:E99"]) == 0
    check(json.loads((repo / P.OUT_REL).read_text()), run, {"G3": "log:E99"})
    rows = [json.loads(x) for x in paths.REGISTRY_PATH.read_text().splitlines()]
    assert {r["experiment"] for r in rows} == {P.EXPERIMENT} and len(rows) == 3 * len(targets)
    assert {r["split"] for r in rows} == {"neon_dev/lodo_cv"}
    assert all(r["config"]["not_run"] == {"G3": "log:E99"} and r["config"]["holdout_scale"] == np.sqrt(1215 / 214)
               for r in rows)
    for argv in ([], ["--not-run", "G3=log:E99"]):           # written once
        with pytest.raises(AV.GateError):
            P.main(argv)
    assert len(paths.REGISTRY_PATH.read_text().splitlines()) == 3 * len(targets)
    # ---- part 2 (another repository root): all five entries finished, 29 pairs
    repo2 = sandbox / "repo2"
    monkeypatch.setattr(paths, "REPO_ROOT", repo2)
    (repo2 / "docs" / "preregistration").mkdir(parents=True)
    (repo2 / P.LOG_REL).write_text("### E99\n")
    frames["xgb_v4mr"].to_parquet(out / "xgb_v4mr.parquet")
    (out / "xgb_v4mr_scores.json").write_text("{}")
    with pytest.raises(AV.GateError):                        # finished: no longer reportable as not run
        P.main(["--not-run", "G3=log:E99"])
    assert P.main([]) == 0
    res = json.loads((repo2 / P.OUT_REL).read_text())
    assert res["n_pairs"] == 29
    check(res, P.ENTRIES, {})


# ================================================================================ (b) hardened protected loaders

def _no_claim(monkeypatch, tmp_path):
    """No open claim can exist: the runs folder is an empty tmp folder. Any parquet read is an error."""
    monkeypatch.setattr(paths, "RUNS_ROOT", tmp_path / "runs_no_claim")

    def boom(*a, **k):
        raise AssertionError("a table was opened although the claim check must refuse first")

    monkeypatch.setattr(pd, "read_parquet", boom)


@pytest.mark.skipif(not REAL_GEO.is_dir() or not REAL_WQ.is_dir(), reason="real M3 data not on this machine")
def test_real_roots_need_the_open_claim_however_they_are_passed(tmp_path, monkeypatch):
    from sentinel.models.microbe.design import load_design

    _no_claim(monkeypatch, tmp_path)
    link_geo, link_wq = tmp_path / "link_geo", tmp_path / "link_wq"
    link_geo.symlink_to(REAL_GEO, target_is_directory=True)
    link_wq.symlink_to(REAL_WQ, target_is_directory=True)
    rel_geo = os.path.relpath(REAL_GEO, Path.cwd())
    for root in (None, REAL_GEO, str(REAL_GEO), link_geo, rel_geo, REAL_GEO / ".." / G.DATASET):
        with pytest.raises(TL.ProtectedDataError):
            G.load_protected_labels(root)
    for root in (None, REAL_WQ, str(REAL_WQ), link_wq, REAL_WQ.parent / "microbe_wq_v1"):
        with pytest.raises(TL.ProtectedDataError):
            load_design(allow_holdout=True, root=root)
    with pytest.raises(TL.ProtectedDataError):
        load_design(allow_holdout=True, with_release_v1=True)


def _synthetic_labels(ids) -> pd.DataFrame:
    """SYNTHETIC label table (random numbers)."""
    rng = np.random.default_rng(3)
    return pd.DataFrame({"sample_id": list(ids), "ph": rng.normal(7, 1, len(ids))})


def test_only_a_declared_synthetic_root_is_exempt(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RUNS_ROOT", tmp_path / "runs_no_claim")
    real_ids = sorted(TL.real_event_ids())
    fake_ids = [f"SYN{i:03d}" for i in range(12)]
    assert not set(fake_ids) & set(real_ids)

    def make(name, ids, marker=True):
        d = tmp_path / name
        d.mkdir()
        _synthetic_labels(ids).to_parquet(d / "protected_labels.parquet")
        if marker:
            (d / TL.SYNTHETIC_MARKER).write_text("synthetic test root\n")
        return d

    ok = make("ok", fake_ids)
    assert TL.is_synthetic_root(ok, "protected_labels.parquet")
    assert list(G.load_protected_labels(ok).index) == fake_ids              # exempt: no claim needed
    assert TL.require_claim_for_root(ok, "protected_labels.parquet") is None
    # 1. no marker
    no_marker = make("no_marker", fake_ids, marker=False)
    # 4. a marker on a table that carries real event ids (a copy of real data fails even with the marker)
    real_copy = make("real_copy", fake_ids[:5] + real_ids[:1])
    # 3. a marker next to a link that leaves the directory
    linked = make("linked", fake_ids)
    (linked / "more.parquet").symlink_to(ok / "protected_labels.parquet")
    # 2. a marker inside the data root
    monkeypatch.setattr(paths, "DATA_ROOT", tmp_path / "data")
    inside = tmp_path / "data" / "x"
    inside.parent.mkdir()
    _ = make("data/x", fake_ids)
    # a link to a synthetic root is the synthetic root; a root without the id file is not synthetic
    alias = tmp_path / "alias"
    alias.symlink_to(ok, target_is_directory=True)
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / TL.SYNTHETIC_MARKER).write_text("x")
    assert TL.is_synthetic_root(alias, "protected_labels.parquet")
    for bad in (no_marker, real_copy, linked, inside, empty, None, tmp_path / "does_not_exist"):
        assert not TL.is_synthetic_root(bad, "protected_labels.parquet"), bad
        with pytest.raises(TL.ProtectedDataError):
            TL.require_claim_for_root(bad, "protected_labels.parquet")
    for bad in (no_marker, real_copy, linked, inside):
        with pytest.raises(TL.ProtectedDataError):
            G.load_protected_labels(bad)


def test_load_design_with_holdout_needs_the_marker_on_a_synthetic_root(tmp_path, monkeypatch, synth):
    from sentinel.models.microbe.design import load_design

    monkeypatch.setattr(paths, "RUNS_ROOT", tmp_path / "runs_no_claim")
    d, _ = synth
    root = tmp_path / "wq"
    root.mkdir()
    s = d.samples.drop(columns=list(N.MASTER)).reset_index(drop=True)
    for c in ("raw_tn_mgl", "raw_tp_mgl", "raw_doc_mgl", "raw_no3_mgl", "raw_sc_uscm", "raw_turb_fnu", "raw_do_mgl",
              "raw_chla_ugl"):
        s[c] = 1.0                                           # SYNTHETIC constants
    s.to_parquet(root / "samples.parquet")
    d.tokens.to_parquet(root / "tokens.parquet")
    d.sequences.to_parquet(root / "sequences.parquet")
    rule = {"kind": "date_from", "date": "2019-06-01"}
    assert len(load_design(root=root, holdout_rule=rule).samples) < len(s)            # development rows: no claim
    with pytest.raises(TL.ProtectedDataError):                                         # holdout rows: refused
        load_design(root=root, holdout_rule=rule, allow_holdout=True)
    (root / TL.SYNTHETIC_MARKER).write_text("synthetic test root\n")
    full = load_design(root=root, holdout_rule=rule, allow_holdout=True)
    assert len(full.samples) == len(s) and full.includes_holdout


@pytest.mark.skipif(not (REAL_GEO / "ids.json").is_file(), reason="real GEO dataset not on this machine")
def test_real_ids_are_all_in_the_frozen_split_and_the_protected_file_is_closed():
    ids = G.load_ids()
    real = TL.real_event_ids()
    for blk in G.BLOCKS:                                     # condition 4 of is_synthetic_root covers every block
        assert set(ids[blk]) <= real, blk
    mode = stat.S_IMODE(os.stat(REAL_GEO / "protected_labels.parquet").st_mode)
    assert mode & ~0o440 == 0, oct(mode)                     # 0440 or stricter


# ======================================================================== (c) owed unit tests (amendment §13)

def _swap(d, S):
    d2 = copy.copy(d)
    d2.samples = S
    return d2


def test_block_features_equal_the_baseline_featuriser(synth):
    d, _ = synth
    tr, te = SY.lodo_folds(d.samples)[0]
    rtr, rte = VM._rows(d, tr), VM._rows(d, te)
    v4 = B.v4_tokens(d)
    for rank, toks in (("genus", d.tokens), ("family", d.tokens), ("order", d.tokens), ("v4", v4)):
        for cap in (500, 7):                                 # 7: ties in prevalence and a pooled "other" column
            ref = F.GenusCLR(min_prev=0.05, max_features=cap, pseudo=0.5, rank=rank).fit(toks, tr)
            rc = VM.rank_counts(d, rank)
            keep = VM.block_fit(rc, rtr, cap, 0.05)
            assert list(rc.labels[keep]) == list(ref.keep_), (rank, cap)
            for rows, ids in ((rtr, tr), (rte, te)):
                assert np.allclose(VM.block_transform(rc, keep, rows, 0.5), ref.transform(toks, ids, "clr"), atol=1e-6)
    # whole feature matrices and predictions of the re-implemented tables
    for name in ("xgb_clr", "xgb_v4clr", "xgb_clr_context", "xgb_v4clr_context", "ridge_clr", "ridge_v4clr", "rf_clr",
                 "context_ridge", "context_rf"):
        a = VM.fit_predict(name, d, tr, te, targets=["log_sc", "ph"], seed=0, n_jobs=1, params=SMALL,
                           _spec_table=VM.REIMPL)
        b = B.fit_predict(name, d, tr, te, targets=["log_sc", "ph"], seed=0, n_jobs=1, params=SMALL)
        assert np.allclose(a.to_numpy(), b.to_numpy(), atol=1e-5), name
    fx = VM.features("xgb_v4mr", d, tr, te, SMALL)
    assert fx["Xtr"].shape[0] == len(tr) and fx["Xte"].shape[0] == len(te) and len(fx["names"]) == fx["Xtr"].shape[1]
    assert {n.split(":")[0] for n in fx["names"] if ":" in n} == {"v4", "genus", "family", "order"}
    assert np.isfinite(fx["Xtr"]).all() and fx["qmp_cols"].size == 0


def test_no_held_out_label_in_any_stacking_feature(synth):
    d, _ = synth
    S = d.samples
    tr, te = SY.lodo_folds(S)[0]
    M = list(N.MASTER)
    VM._STACK_CACHE.clear()
    A_tr, A_te = VM.stack_features("xgb_clr", d, tr, te, n_jobs=1, params=SMALL)
    assert list(A_tr.index) == list(tr) and list(A_te.index) == list(te) and list(A_tr.columns) == M
    assert np.isfinite(A_tr.to_numpy()).all() and np.isfinite(A_te.to_numpy()).all()
    # every held-out label replaced: no stacking feature moves, on training or held-out rows
    S2 = S.copy()
    S2.loc[te, M] = 12345.0
    VM._STACK_CACHE.clear()
    B_tr, B_te = VM.stack_features("xgb_clr", _swap(d, S2), tr, te, n_jobs=1, params=SMALL)
    assert np.array_equal(A_tr.to_numpy(), B_tr.to_numpy()) and np.array_equal(A_te.to_numpy(), B_te.to_numpy())
    # the labels of one training domain replaced: its own rows' features do not move (they come from models fitted
    # on the other training domains), the other rows' features do
    dom = S.loc[tr, "domain_id"].to_numpy()
    own = tr[dom == dom[0]]
    rest = tr[dom != dom[0]]
    S3 = S.copy()
    S3.loc[own, M] = S3.loc[own, M] + 5.0
    VM._STACK_CACHE.clear()
    C_tr, C_te = VM.stack_features("xgb_clr", _swap(d, S3), tr, te, n_jobs=1, params=SMALL)
    assert np.array_equal(A_tr.loc[own].to_numpy(), C_tr.loc[own].to_numpy())
    assert not np.allclose(A_tr.loc[rest].to_numpy(), C_tr.loc[rest].to_numpy())
    assert not np.allclose(A_te.to_numpy(), C_te.to_numpy())
    # the stacked models: finite, different from their base, unchanged by held-out labels
    for name, base_table in (("xgb_clr_stack", VM.REIMPL), ("xgb_v4mr_stack", VM.ALL)):
        VM._STACK_CACHE.clear()
        p1 = VM.fit_predict(name, d, tr, te, targets=["log_sc", "ph"], seed=0, n_jobs=1, params=SMALL)
        VM._STACK_CACHE.clear()
        p2 = VM.fit_predict(name, _swap(d, S2), tr, te, targets=["log_sc", "ph"], seed=0, n_jobs=1, params=SMALL)
        assert np.isfinite(p1.to_numpy()).all() and np.array_equal(p1.to_numpy(), p2.to_numpy()), name
        base = VM.fit_predict(VM.ALL[name]["stack"], d, tr, te, targets=["log_sc", "ph"], seed=0, n_jobs=1,
                              params=SMALL, _spec_table=base_table)
        assert not np.allclose(p1.to_numpy(), base.to_numpy()), name


def test_stacked_model_never_receives_its_own_target_as_a_feature(synth, monkeypatch):
    import xgboost as xgb

    d, _ = synth
    tr, te = SY.lodo_folds(d.samples)[1]
    VM._STACK_CACHE.clear()
    n_base = VM.features("xgb_clr_stack", d, tr, te, SMALL)["Xtr"].shape[1]
    A_tr, _ = VM.stack_features("xgb_clr", d, tr, te, n_jobs=1, params=SMALL)
    seen = []
    real_fit = xgb.XGBRegressor.fit

    def spy(self, X, y, *a, **k):
        seen.append(np.asarray(X))
        return real_fit(self, X, y, *a, **k)

    monkeypatch.setattr(xgb.XGBRegressor, "fit", spy)
    VM.fit_predict("xgb_clr_stack", d, tr, te, targets=["ph"], seed=0, n_jobs=1, params=SMALL)
    X = seen[-1]
    assert X.shape[1] == n_base + len(N.MASTER) - 1                          # the 16 other dimensions
    ok = np.isfinite(d.samples.loc[tr, "ph"].to_numpy(float))
    others = [c for c in N.MASTER if c != "ph"]
    assert np.allclose(X[:, n_base:], A_tr.loc[tr[ok], others].to_numpy(np.float32))


def test_masked_multi_output_model(synth, monkeypatch):
    import xgboost as xgb

    rng = np.random.default_rng(5)
    X = rng.normal(0, 1, (160, 6)).astype(np.float32)
    Y = np.column_stack([X[:, 0] + 0.1 * rng.normal(size=160), 2.0 - X[:, 1] + 0.1 * rng.normal(size=160),
                         10.0 * X[:, 2] + rng.normal(size=160)])
    Xte = rng.normal(0, 1, (30, 6)).astype(np.float32)
    p = {**SMALL["xgb"], "subsample": 1.0, "colsample_bytree": 1.0}
    # the objective: zero gradient and Hessian where the label is masked, squared error elsewhere
    Mk = np.array([[True, False], [True, True]])
    g, h = VM.masked_squared_error(Mk)(np.array([[1.0, 99.0], [2.0, 3.0]]), np.array([[0.5, 7.0], [2.5, 1.0]]))
    assert np.array_equal(g, [[-0.5, 0.0], [0.5, -2.0]]) and np.array_equal(h, [[1.0, 0.0], [1.0, 1.0]])
    # with every label present it is the built-in squared error on the standardised targets
    mine = VM.masked_mo_fit_predict(X, Y, Xte, seed=0, n_jobs=1, xgb_params=p)
    Z = (Y - Y.mean(0)) / Y.std(0)
    ref = xgb.XGBRegressor(**p, random_state=0, n_jobs=1, tree_method="hist", multi_strategy="multi_output_tree",
                           base_score=0.0).fit(X, Z).predict(Xte) * Y.std(0) + Y.mean(0)
    assert mine.shape == (30, 3) and np.allclose(mine, ref, atol=1e-3 * Y.std(0).max())
    # a masked label is not filled: the value stored under the mask does not matter
    M = rng.random(Z.shape) > 0.3
    M[:, 0] = True
    preds = []
    for filler in (0.0, 123.0):
        m = xgb.XGBRegressor(**p, random_state=0, n_jobs=1, tree_method="hist", multi_strategy="multi_output_tree",
                             objective=VM.masked_squared_error(M), base_score=0.0)
        preds.append(m.fit(X, np.where(M, Z, filler)).predict(Xte))
    assert np.array_equal(preds[0], preds[1])
    # ... and it is not the model that fills the gaps with the mean (0 after standardisation) and counts them
    filled = xgb.XGBRegressor(**p, random_state=0, n_jobs=1, tree_method="hist", multi_strategy="multi_output_tree",
                              base_score=0.0).fit(X, np.where(M, Z, 0.0)).predict(Xte)
    assert not np.allclose(preds[0], filled, atol=1e-4)
    # missing labels: rows without any label are dropped, a dimension with < 10 labels is left out (NaN)
    Y2 = Y.copy()
    Y2[::3, 1] = np.nan
    Y2[:5] = np.nan
    Y2[:, 2] = np.nan
    Y2[10:17, 2] = 1.0 + np.arange(7)
    out = VM.masked_mo_fit_predict(X, Y2, Xte, seed=0, n_jobs=1, xgb_params=p)
    assert np.isfinite(out[:, :2]).all() and np.isnan(out[:, 2]).all()
    assert np.corrcoef(out[:, 0], Xte[:, 0])[0, 1] > 0.8 and np.corrcoef(out[:, 1], -Xte[:, 1])[0, 1] > 0.8
    assert np.isnan(VM.masked_mo_fit_predict(X, np.full_like(Y, np.nan), Xte, seed=0, n_jobs=1, xgb_params=p)).all()

    # design level: ONE model on the 17 master dimensions serves both families; held-out labels are not read
    d, _ = synth
    tr, te = SY.lodo_folds(d.samples)[0]
    calls = []
    real = VM.masked_mo_fit_predict
    monkeypatch.setattr(VM, "masked_mo_fit_predict", lambda *a, **k: (calls.append(a[1].shape), real(*a, **k))[1])
    VM._MO_CACHE.clear()
    P = VM.fit_predict("xgb_v4clr_mo", d, tr, te, targets=TARGET_NAMES, seed=0, n_jobs=1, params=SMALL)
    Gp = VM.fit_predict("xgb_v4clr_mo", d, tr, te, targets=G.GEO_TARGETS, seed=0, n_jobs=1, params=SMALL)
    both = VM.fit_predict("xgb_v4clr_mo", d, tr, te, targets=list(TARGET_NAMES) + list(G.GEO_TARGETS), seed=0,
                          n_jobs=1, params=SMALL)
    assert calls == [(len(tr), len(N.MASTER))]
    assert np.isfinite(P.to_numpy()).all() and np.isfinite(Gp.to_numpy()).all()
    assert np.array_equal(both[list(TARGET_NAMES)].to_numpy(), P.to_numpy())
    assert np.array_equal(both[list(G.GEO_TARGETS)].to_numpy(), Gp.to_numpy())
    S2 = d.samples.copy()
    S2.loc[te, list(N.MASTER)] = 12345.0
    VM._MO_CACHE.clear()
    P2 = VM.fit_predict("xgb_v4clr_mo", _swap(d, S2), tr, te, targets=TARGET_NAMES, seed=0, n_jobs=1, params=SMALL)
    assert np.array_equal(P.to_numpy(), P2.to_numpy())
    VM.fit_predict("xgb_v4clr_mo", d, tr, te, targets=TARGET_NAMES, seed=1, n_jobs=1, params=SMALL)
    assert len(calls) == 3                                                    # another seed is another fit
    with pytest.raises(KeyError):                                             # needs the master columns
        VM.fit_predict("xgb_v4clr_mo", _swap(d, d.samples.drop(columns=["log_ca"])), tr, te, targets=["ph"],
                       params=SMALL)
    with pytest.raises(ValueError):
        VM.fit_predict("xgb_v4clr_mo", d, tr, te, targets=["latitude"], params=SMALL)   # not a master dimension


def _qmp_table(ids, seed=7) -> pd.DataFrame:
    """SYNTHETIC QMP block with missing entries."""
    rng = np.random.default_rng(seed)
    q = pd.DataFrame(rng.normal(0, 1, (len(ids), 4)), index=pd.Index(list(ids)).astype(str), columns=list(G.QMP_FEATURES))
    q.iloc[::5, 0] = np.nan
    q.iloc[::7, 3] = np.nan
    return q


def test_qmp_twins_and_niche_variants(synth):
    d, _ = synth
    tr, te = SY.lodo_folds(d.samples)[2]
    q = _qmp_table(d.samples.index)
    pq_ = {**SMALL, "qmp_table": q}
    fx = VM.features("xgb_v4clr_qmp", d, tr, te, pq_)
    fx0 = VM.features("xgb_v4clr", d, tr, te, SMALL)
    assert fx["Xtr"].shape[1] == fx0["Xtr"].shape[1] + 8 and list(fx["names"][-8:]) == list(G.QMP_FEATURES + G.QMP_FLAGS)
    assert np.array_equal(fx["Xtr"][:, :fx0["Xtr"].shape[1]], fx0["Xtr"])
    c = fx["qmp_cols"]
    assert np.allclose(fx["Xtr"][:, c], q.loc[tr].to_numpy(np.float32), equal_nan=True)     # trees receive NaN
    assert np.array_equal(fx["Xtr"][:, c + 4], np.isnan(q.loc[tr].to_numpy()).astype(np.float32))
    for name in ("xgb_v4clr_qmp", "ridge_v4clr_qmp", "xgb_context_qmp", "xgb_v4mr_qmp"):
        p = VM.fit_predict(name, d, tr, te, targets=["log_sc", "ph"], seed=0, n_jobs=1, params=pq_)
        assert np.isfinite(p.to_numpy()).all(), name
    # the QMP block is an input like any other: a different block gives different predictions
    a = VM.fit_predict("xgb_context_qmp", d, tr, te, targets=["ph"], seed=0, n_jobs=1, params=pq_)
    b = VM.fit_predict("xgb_context_qmp", d, tr, te, targets=["ph"], seed=0, n_jobs=1,
                       params={**SMALL, "qmp_table": _qmp_table(d.samples.index, seed=8)})
    assert not np.allclose(a.to_numpy(), b.to_numpy())

    base = N.fit_predict("niche_s", d, tr, te, targets=N.MASTER, params=FAST)
    # +QMP twin of the inversion (reading R14): QMP block in the context prior
    tw = N.fit_predict("niche_s_qmp", d, tr, te, targets=N.MASTER, params={**FAST, "qmp_table": q})
    assert np.isfinite(tw.to_numpy()).all() and not np.allclose(tw.to_numpy(), base.to_numpy())
    # ablation A0 (reading R15): flat prior, kappa = 0, no slope clip, one N_eff
    a0 = N.fit_predict("niche_s_a0", d, tr, te, targets=N.MASTER, params={"newton_steps": 4})
    diag = N.LAST["result"].diag
    assert np.isfinite(a0.to_numpy()).all() and not np.allclose(a0.to_numpy(), base.to_numpy())
    assert diag["context_prior"] is False and diag["kappa"] == 0.0 and float(diag["n_eff"]) == 100.0
    v = {**N.DEFAULTS, **N.VARIANTS["niche_s_a0"]}
    assert v["slope_clip"] is None and v["n_eff_grid"] == (100.0,) and v["flat_prior_ridge"] == 1e-6
    s, i = N.deshrink_fit(np.array([0.0, 1.0, 2.0, 3.0]), np.array([0.0, 10.0, 20.0, 30.0]), clip=None)
    assert s == pytest.approx(10.0) and i == pytest.approx(0.0)                # no clip
    assert N.deshrink_fit(np.array([0.0, 1.0, 2.0, 3.0]), np.array([0.0, 10.0, 20.0, 30.0]))[0] == pytest.approx(3.0)
    # A0 does not read the context at all: changing the context of the held-out rows changes nothing
    S2 = d.samples.copy()
    S2.loc[te, "latitude"] = S2.loc[te, "latitude"] + 20.0
    S2.loc[te, "collect_date"] = S2.loc[te, "collect_date"] + pd.Timedelta(days=90)
    a0b = N.fit_predict("niche_s_a0", _swap(d, S2), tr, te, targets=N.MASTER, params={"newton_steps": 4})
    assert np.allclose(a0.to_numpy(), a0b.to_numpy())
    moved = N.fit_predict("niche_s", _swap(d, S2), tr, te, targets=N.MASTER, params=FAST)
    assert not np.allclose(moved.to_numpy(), base.to_numpy())


def test_niche_fold_cache_serves_families_seeds_and_the_control(synth, monkeypatch):
    d, _ = synth
    folds = SY.lodo_folds(d.samples)
    N._FOLD_CACHE.clear()
    first = [N.fit_predict("niche_s", d, tr, te, targets=TARGET_NAMES, params=FAST) for tr, te in folds[:3]]
    assert len(N._FOLD_CACHE) == 3

    def boom(*a, **k):
        raise AssertionError("the fold was fitted again")

    monkeypatch.setattr(N, "fit_fold", boom)
    for (tr, te), p in zip(folds[:3], first):
        g = N.fit_predict("niche_s", d, tr, te, targets=G.GEO_TARGETS, params=FAST)          # the other family
        assert np.isfinite(g.to_numpy()).all()
        assert N.fit_predict("niche_s", d, tr, te, targets=TARGET_NAMES, params=FAST, seed=2).equals(p)
        c = N.fit_predict(N.CONTROL, d, tr, te, targets=["log_sc"], params=FAST, seed=1)     # the control
        assert np.isfinite(c.to_numpy()).all()
    with pytest.raises(AssertionError):                                       # another variant is another fit
        N.fit_predict("niche_s_tax", d, *folds[0], targets=TARGET_NAMES, params=FAST)
    with pytest.raises(AssertionError):                                       # so is another training label
        S2 = d.samples.copy()
        S2.loc[folds[0][0][:5], "ph"] += 1.0
        N.fit_predict("niche_s", _swap(d, S2), *folds[0], targets=TARGET_NAMES, params=FAST)
    assert N.FOLD_CACHE_SIZE >= 2 * 19


def test_oof_loop_equals_one_call_per_fold_and_seed(synth):
    d, _ = synth
    folds = _folds(d)
    tg = ["log_sc", "log_ca"]
    seen = []
    per_seed, diags = AV.oof("xgb_clr", d, folds, targets=tg, seeds=(0, 1), n_jobs=1, params=SMALL,
                             on_fold=lambda i, f, dg: seen.append((i, f.name, dg)))
    assert sorted(per_seed) == [0, 1] and [s[0] for s in seen] == list(range(len(folds))) and diags == []
    for s in (0, 1):
        ref = pd.concat([B.fit_predict("xgb_clr", d, f.train, f.test, targets=tg, seed=s, n_jobs=1, params=SMALL)
                         for f in folds])
        assert np.array_equal(per_seed[s][tg].to_numpy(), ref.to_numpy()) and list(per_seed[s].index) == list(ref.index)
        assert sorted(per_seed[s].index) == sorted(d.samples.index)
        assert (per_seed[s]["fold"].to_numpy() == np.concatenate([[i] * len(f.test) for i, f in enumerate(folds)])).all()
    assert not np.allclose(per_seed[0][tg].to_numpy(), per_seed[1][tg].to_numpy())
    w, wd = AV.oof("wa_tol", d, folds[:2], targets=tg)
    assert len(wd) == 2 and wd[0]["fold"] == 0 and wd[0]["n_units"] > 0
    nz, nd = AV.oof("niche_s", d, folds[:1], targets=tg, params=FAST)
    assert "test_first_step_fail_share" in nd[0] and "n_test" in nd[0] and "n_test_without_reads" in nd[0]
    e = AV.ensemble([per_seed[0], per_seed[1]], tg)
    assert np.allclose(e.to_numpy(), (per_seed[0][tg].to_numpy() + per_seed[1][tg].to_numpy()) / 2)
    with pytest.raises(ValueError):
        AV.ensemble([per_seed[0], per_seed[1].iloc[:-1]], tg)


def test_calibration_hand_worked():
    # two domains; within-domain deviations: predicted (-1, +1), observed (-2, +2) in a and (-1, +1) in b
    y = np.array([8.0, 12.0, 19.0, 21.0] * 3)
    p = np.array([4.0, 6.0, 9.0, 11.0] * 3)
    dom = np.array(["a", "a", "b", "b"] * 3)
    c = AV.calibration(y, p, dom)
    assert c["n"] == 12 and c["slope_within_domain"] == pytest.approx((2 + 2 + 1 + 1) / 4)        # 1.5
    assert c["sd_ratio_within_domain"] == pytest.approx(1.0 / np.sqrt(2.5))
    assert c["sd_ratio"] == pytest.approx(np.std(p[:4]) / np.std(y[:4]))
    assert c["slope_pooled"] == pytest.approx(np.polyfit(p, y, 1)[0])
    y2, p2 = y.copy(), p.copy()
    y2[0], p2[1] = np.nan, np.nan                                             # rows with a missing value are dropped
    assert AV.calibration(y2, p2, dom)["n"] == 10
    assert np.isnan(AV.calibration(y[:8], p[:8], dom[:8])["slope_within_domain"])                  # fewer than 10 rows
    assert AV.calibration(y, y, dom)["slope_within_domain"] == pytest.approx(1.0)


# ------------------------------------------------------------- development loader, GEO build, partition script

def _geo_root(tmp_path, d, protected=()):
    """SYNTHETIC GEO dataset directory for the events of the synthetic design."""
    root = tmp_path / "geo_synth"
    root.mkdir()
    S = d.samples
    tab = pd.DataFrame({"sample_id": S.index.to_numpy()})
    for c in GEO_AUX:
        tab[c] = S[c].to_numpy(float)
    tab.to_parquet(root / "dev_targets.parquet")
    ids = [str(i) for i in S.index]
    (root / "ids.json").write_text(json.dumps({"dev": [i for i in ids if i not in protected], "train1718": [],
                                               "holdout": list(protected), "reserve": []}))
    return root


def test_attach_master_and_the_development_loader_refuse_protected_rows(tmp_path, synth):
    d, _ = synth
    bare = _swap(d, d.samples.drop(columns=GEO_AUX))
    root = _geo_root(tmp_path, d)
    full = AV.attach_master(bare, root)
    assert full is not bare and "ph" not in bare.samples.columns
    assert np.allclose(full.samples[list(N.MASTER)].to_numpy(float), d.samples[list(N.MASTER)].to_numpy(float),
                       equal_nan=True)
    assert full.tokens is d.tokens                                           # caches keyed on the token table still hit
    with pytest.raises(G.ProtectedRowError):
        AV.attach_master(_swap(bare, bare.samples), root) if False else AV.attach_master(
            type(bare)(bare.samples, bare.tokens, bare.sequences, bare.holdout_rule, True), root)
    # a protected id in the request, and a protected row inside the development table, both refuse
    ids = [str(i) for i in d.samples.index]
    blocks = json.loads((root / "ids.json").read_text())
    (root / "ids.json").write_text(json.dumps({**blocks, "dev": ids[1:], "holdout": ids[:1]}))
    with pytest.raises(G.ProtectedRowError):
        G.load_dev_targets(ids[1:], root)                                    # the table itself holds a protected row
    tab = pd.read_parquet(root / "dev_targets.parquet")
    tab[tab["sample_id"] != ids[0]].to_parquet(root / "dev_targets.parquet")
    assert len(G.load_dev_targets(ids[1:4], root)) == 3
    with pytest.raises(G.ProtectedRowError):
        G.load_dev_targets(ids[:3], root)
    with pytest.raises(G.ProtectedRowError):
        AV.attach_master(bare, root)
    with pytest.raises(KeyError):
        G.load_dev_targets(["NOT_AN_EVENT"], root)


@pytest.mark.skipif(not (REAL_GEO / "dev_targets.parquet").is_file(), reason="real GEO dataset not on this machine")
def test_real_development_loader_refuses_every_protected_id(monkeypatch):
    ids = G.load_ids()
    opened = []
    real_read = pd.read_parquet
    monkeypatch.setattr(pd, "read_parquet", lambda p, *a, **k: (opened.append(Path(p).name), real_read(p, *a, **k))[1])
    tab = G.load_dev_targets()
    assert set(tab.index) == set(ids["dev"]) | set(ids["train1718"])
    assert not set(tab.index) & (set(ids["holdout"]) | set(ids["reserve"]))
    for blk in G.PROTECTED_BLOCKS:
        for i in (ids[blk][0], ids[blk][-1]):
            with pytest.raises(G.ProtectedRowError):
                G.load_dev_targets([ids["dev"][0], i])
    assert len(G.load_dev_targets(ids["dev"][:5])) == 5
    assert set(opened) == {"dev_targets.parquet"}                            # the protected file was never opened


# Appendix A.1 of the stamped amendment, development blocks: column -> (usable, values, censored)
A1_DEV = {"lab_ph": (1165, 1165, 0), "cl_mgl": (1178, 1171, 20), "so4_mgl": (1178, 1176, 23), "ca_mgl": (1117, 1105, 15),
          "dic_mgl": (1171, 1171, 0), "field_do_sat_pct": (1174, 1174, 0), "nh4_n_mgl": (1182, 1141, 247),
          "uv_absorbance_254_nm_au": (1182, 1175, 7), "field_temp_c": (1194, None, 0),
          "field_spcond_uscm": (1191, None, 0), "field_do_mgl": (1172, None, 0)}
# column -> (usable, censored) of the .001 2017-18 training block
A1_TRAIN1718 = {"lab_ph": (472, 0), "cl_mgl": (503, 23), "so4_mgl": (503, 10), "ca_mgl": (503, 0), "dic_mgl": (493, 0),
                "field_do_sat_pct": (493, 0), "nh4_n_mgl": (493, 220), "uv_absorbance_254_nm_au": (506, 4),
                "field_temp_c": (505, None), "field_spcond_uscm": (503, None), "field_do_mgl": (494, None)}
A1_EVENTS = {"dev": 1215, "holdout": 214, "train1718": 517, "reserve": 190}
A1_ALL7 = {"dev": 1018, "train1718": 423}


@pytest.mark.skipif(not (REAL_GEO / "dev_targets.parquet").is_file(), reason="real GEO dataset not on this machine")
def test_geo_build_reproduces_appendix_a1(monkeypatch):
    """Counts only. Development blocks: per-variable counts from the development table. Protected blocks: row and
    key counts from ids and the stored counts file; the protected label file is not opened."""
    opened = []
    real_read = pd.read_parquet
    monkeypatch.setattr(pd, "read_parquet", lambda p, *a, **k: (opened.append(Path(p).name), real_read(p, *a, **k))[1])
    ids = G.load_ids()
    assert {b: len(ids[b]) for b in G.BLOCKS} == A1_EVENTS
    assert len(set().union(*(ids[b] for b in G.BLOCKS))) == sum(A1_EVENTS.values())        # blocks are disjoint
    tab = G.load_dev_targets()
    by_col = {col: t for t, (col, _, _) in G.ALL_SPECS.items()}
    assert set(by_col) == set(A1_DEV)
    for blk, table in (("dev", A1_DEV), ("train1718", A1_TRAIN1718)):
        sub = tab.loc[ids[blk]]
        assert len(sub) == A1_EVENTS[blk]
        for col, exp in table.items():
            t = by_col[col]
            usable = sub[t].notna().to_numpy()
            assert int(usable.sum()) == exp[0], (blk, col)
            cens = int(sub[f"{t}_censored"].sum())
            if blk == "dev":
                if exp[1] is not None:
                    assert int((usable & ~sub[f"{t}_imputed"].to_numpy(bool)).sum()) == exp[1], (blk, col)
                assert cens == exp[2], (blk, col)
            elif exp[1] is not None:
                assert cens == exp[1], (blk, col)
        seven = [t for t in G.GEO_TARGETS if t != "log_nh4"]
        vals = sub[seven].notna().to_numpy() & ~sub[[f"{t}_imputed" for t in seven]].to_numpy(bool)
        assert int(vals.all(1).sum()) == A1_ALL7[blk], blk
    # the evidence file of Appendix B and the stored counts agree on every development-block number
    ev = json.loads((REPO / "docs/preregistration/microbe_arch_v2_evidence/geo_counts.json").read_text())["blocks"]
    stored = json.loads((REAL_GEO / "counts.json").read_text())
    for blk in ("dev", "train1718"):
        e, s = ev[G.EVIDENCE_BLOCK[blk]], stored[blk]
        for k in ("n_events", "n_with_chem_parent", "sites", "domains", "n_qpcr_16s"):
            assert s[k] == e[k], (blk, k)
        for col, r in s["vars"].items():
            for k, v in r.items():
                assert e["vars"][col][k] == v, (blk, col, k)
            assert r["n_usable"] == (A1_DEV if blk == "dev" else A1_TRAIN1718)[col][0]
    for blk in G.PROTECTED_BLOCKS:                           # row and key counts only (lead ruling I-04)
        e, s = ev[G.EVIDENCE_BLOCK[blk]], stored[blk]
        assert s["n_events"] == e["n_events"] == A1_EVENTS[blk] and s["n_with_chem_parent"] == e["n_with_chem_parent"]
    assert set(opened) == {"dev_targets.parquet"}


def test_protected_block_counts_are_row_and_key_counts_only():
    meta = pd.DataFrame({"site_id": ["a", "a", "b"], "domain_id": ["D1", "D1", "D2"], "chem_match": ["parent"] * 3,
                         "chem_parent": ["p1", None, "p3"]})

    class NoLabels(pd.DataFrame):                            # any column access is an error
        def __getitem__(self, key):
            raise AssertionError(f"label column {key!r} read for a protected block")

    out = G.target_counts(NoLabels(index=range(3)), meta, keys_only=True)
    assert out == {"n_events": 3, "n_with_chem_parent": 2, "sites": 2, "domains": 2}
    tab = pd.DataFrame({"ph": [7.0, np.nan, 6.5], "ph_censored": [False, False, True], "ph_imputed": [False, False, True]})
    full = G.target_counts(tab, meta, {"ph": ("lab_ph", "identity", None)})
    assert full["vars"]["lab_ph"]["n_usable"] == 2 and full["vars"]["lab_ph"]["n"] == 1 and full["n_events"] == 3
    src = (REPO / "scripts/data/build_microbe_geo.py").read_text()
    assert "keys_only=protected" in src and "protected = blk in G.PROTECTED_BLOCKS" in src


def _appendix_b() -> dict:
    text = (REPO / ST.AMENDMENT).read_text()
    tail = text[text.index("## Appendix B."):]
    return dict(re.findall(r"^\| `([^`]+)`[^|]*\| `([0-9a-f]{64})` \|$", tail, re.M))


def test_partition_script_and_evidence_files_reproduce_appendix_b(tmp_path):
    table = _appendix_b()
    assert len(table) == 6
    for rel, sha in table.items():
        assert _sha(REPO / rel) == sha, rel
    ev = "docs/preregistration/microbe_arch_v2_evidence"
    out = tmp_path / "partition" / "ea_partition.json"
    out.parent.mkdir()
    r = subprocess.run([sys.executable, "-I", str(REPO / "scripts/data/microbe_ea_partition.py"),
                        str(REPO / ev / "ena_PRJEB90117_samples.tsv"), str(out)], capture_output=True, text=True,
                       cwd=tmp_path, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    assert _sha(out) == table[f"{ev}/ea_partition.json"]
