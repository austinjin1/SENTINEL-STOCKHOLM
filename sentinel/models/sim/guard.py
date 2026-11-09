# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Test-set guard for SENTINEL-Sim (RESEARCH_PLAN.md §3 R6; docs/preregistration/sim_protocol.md §2).

Protected data: time blocks ``test`` (Test-A, WY2024–2025) and ``test_b``, and every test-role site of
the frozen sonde split in ANY block. Development (``mode="validation"``) keeps only train-/val-role sites and dates ≤ the end of
the validation block (per-site files are read whole by the shared M1 loader and immediately cut, so
protected values are never retained, used or cached).
``mode="test"`` additionally requires ``allow_test=True`` and the freeze file
``docs/preregistration/FREEZE_sim.json`` whose ``split_sha256`` and ``frozen_code_sha256`` match the
current split and code (:func:`frozen_code_sha256`).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ...core import paths
from ...data.sonde_splits import TIME_BLOCKS

FREEZE_FILE = paths.REPO_ROOT / "docs" / "preregistration" / "FREEZE_sim.json"
SPLIT_FILE = paths.REPO_ROOT / "docs" / "preregistration" / "sonde_split.json"
SPLIT_SHA256 = "fd4bb14fe7304d05310e286e9aced22675fd6d07aa4dcac6415ed039a1e06a80"
FROZEN_CODE = ("sentinel/models/sim/physics.py", "sentinel/models/sim/data.py", "sentinel/models/sim/model.py",
               "sentinel/models/sim/baselines.py", "sentinel/models/sim/hindcast.py", "sentinel/models/sim/guard.py",
               "sentinel/models/sim/runner.py", "sentinel/models/sim/checks.py", "sentinel/models/sim/latent.py",
               "sentinel/models/sim/dsst.py", "sentinel/models/sim/scenario.py",
               "sentinel/models/hydromet/physics.py", "sentinel/models/baselines/sonde_baselines.py",
               "sentinel/models/sonde/data.py", "sentinel/data/sonde_splits.py",
               "sentinel/eval/metrics.py", "sentinel/eval/bootstrap.py",
               "scripts/eval/sim_hindcast.py", "scripts/eval/sim_checks.py",
               "sentinel/models/sim/strata.py", "scripts/eval/sim_hard_track.py", "scripts/train/sim_train.py",
               "docs/preregistration/sim_protocol.md", "docs/preregistration/sim_cal_sites.json",
               "docs/preregistration/sim_hard_track.json")


class ProtectedDataError(PermissionError):
    pass


def file_sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def frozen_code_sha256(root: Path | None = None) -> str:
    root = Path(root) if root else paths.REPO_ROOT
    h = hashlib.sha256()
    for rel in FROZEN_CODE:
        h.update(rel.encode())
        f = root / rel
        h.update(f.read_bytes() if f.exists() else b"<missing>")
    return h.hexdigest()


def check_split(path: Path = SPLIT_FILE) -> None:
    sha = file_sha256(path)
    if sha != SPLIT_SHA256:
        raise ProtectedDataError(f"split {path} sha256 {sha[:12]}… != frozen {SPLIT_SHA256[:12]}…")


def authorise(mode: str = "validation", *, allow_test: bool = False) -> dict:
    """Return {'roles': allowed site roles, 'data_end': last date that may be loaded, 'mode'}."""
    check_split()
    if mode == "validation":
        return {"mode": mode, "roles": ("train", "val"), "data_end": TIME_BLOCKS["val"][1]}
    if mode == "test":
        if not allow_test:
            raise ProtectedDataError("mode='test' needs allow_test=True (--allow-test)")
        if not FREEZE_FILE.exists():
            raise ProtectedDataError(f"freeze file {FREEZE_FILE} missing — test data stay locked")
        fz = json.loads(FREEZE_FILE.read_text())
        if fz.get("split_sha256") != SPLIT_SHA256:
            raise ProtectedDataError("freeze file split hash mismatch")
        if fz.get("frozen_code_sha256") != frozen_code_sha256():
            raise ProtectedDataError("frozen code changed since FREEZE_sim.json was written")
        return {"mode": mode, "roles": ("train", "val", "test"), "data_end": TIME_BLOCKS["test"][1],
                "freeze": fz}
    raise ValueError(mode)


def verify_weights(checkpoints: dict, freeze: dict) -> None:
    """Every checkpoint used for test scoring must match ``freeze['weights_sha256'][run]``."""
    want = freeze.get("weights_sha256") or {}
    for run, path in checkpoints.items():
        if run not in want:
            raise ProtectedDataError(f"run {run} not in FREEZE_sim.json weights_sha256")
        if file_sha256(Path(path)) != want[run]:
            raise ProtectedDataError(f"checkpoint of {run} changed since the freeze")
