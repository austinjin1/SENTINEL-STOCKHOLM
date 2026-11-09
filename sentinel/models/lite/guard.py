# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Test-set guard for SENTINEL-Lite (RESEARCH_PLAN.md §3 R6; protocol §8).

Protected roles = ``test`` (Test-A, unseen water bodies) and ``test_time`` (Test-B, future NAIP cycle).
Training code never receives protected chips (:func:`assert_no_protected`). Scoring protected chips
requires ``allow_test=True`` AND a valid ``docs/preregistration/FREEZE_lite.json`` whose
``split_sha256`` matches the frozen split, whose ``scoring_code_sha256`` matches the current code and
whose ``weights_sha256`` lists every scored artifact. The freeze file is NOT created in this phase.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ...core import paths
from ..sonde.guard import ProtectedDataError, artifact_sha256  # noqa: F401  (generic helpers, M1-owned)
from .splits import PROTECTED_ROLES, split_sha256

SPLIT_FILE = paths.REPO_ROOT / "docs" / "preregistration" / "lite_split.json"
FREEZE_FILE = paths.REPO_ROOT / "docs" / "preregistration" / "FREEZE_lite.json"
FREEZE_KEYS = ("created", "split_sha256", "scoring_code_sha256", "weights_sha256")
FROZEN_GLOBS = ("sentinel/models/lite/*.py", "configs/lite/*.yaml", "scripts/train/lite_*.py",
                "scripts/eval/lite_*.py", "docs/preregistration/lite_protocol.md")
SHARED_CODE = ("sentinel/eval/metrics.py", "sentinel/eval/bootstrap.py", "sentinel/eval/splits.py",
               "sentinel/data/wqp_harmonize.py", "sentinel/data/chips.py", "sentinel/models/bioassess/data.py",
               "sentinel/models/sonde/guard.py")
# The split hash fixed in the protocol before any model selection (protocol §3).
# National track (protocol §12): set when lite_split_national.json is frozen, before any model selection on it.
FROZEN_NATIONAL_SPLIT_SHA256 = "8e3b1ee00c8a5d2b6c9f9b291c15ae0b3d329abaa98c1d22887f2ec874cf7d19"
FROZEN_SPLIT_SHA256 = "46ed6e1f8724a2ef4aded2f4251c151d322a920d3c1df0d90874bb81a3397475"


def frozen_files(root: Path | None = None) -> list[str]:
    root = Path(root) if root else paths.REPO_ROOT
    files = set(SHARED_CODE)
    for g in FROZEN_GLOBS:
        files |= {str(f.relative_to(root)) for f in root.glob(g) if f.is_file()}
    return sorted(files)


def scoring_code_sha256(root: Path | None = None) -> str:
    root = Path(root) if root else paths.REPO_ROOT
    h = hashlib.sha256()
    for rel in frozen_files(root):
        f = root / rel
        h.update(rel.encode())
        h.update(f.read_bytes() if f.is_file() else b"<missing>")
    return h.hexdigest()


def load_split(path: str | Path | None = None, *, check_frozen: bool = True) -> dict:
    p = Path(path) if path else SPLIT_FILE
    split = json.loads(p.read_text())
    sha = split_sha256(split)
    if check_frozen and FROZEN_SPLIT_SHA256 != "PENDING" and sha != FROZEN_SPLIT_SHA256:
        raise ProtectedDataError(f"split {p} sha256 {sha[:12]} != frozen {FROZEN_SPLIT_SHA256[:12]}")
    split["_sha256"] = sha
    split["_path"] = str(p)
    return split


def assert_no_protected(roles) -> None:
    bad = sorted({str(r) for r in roles} & set(PROTECTED_ROLES))
    if bad:
        raise ProtectedDataError(f"training/selection code received protected roles {bad}")


def load_freeze(freeze_file: str | Path | None = None, *, split: dict | None = None) -> dict:
    ff = Path(freeze_file) if freeze_file else FREEZE_FILE
    if not ff.is_file():
        raise ProtectedDataError(f"freeze file {ff} does not exist")
    try:
        freeze = json.loads(ff.read_text())
    except ValueError as e:
        raise ProtectedDataError(f"freeze file {ff} is not valid JSON: {e}") from e
    missing = [k for k in FREEZE_KEYS if not freeze.get(k)]
    if missing:
        raise ProtectedDataError(f"freeze file lacks {missing}")
    if not isinstance(freeze["weights_sha256"], dict):
        raise ProtectedDataError("weights_sha256 must map model -> sha256")
    if split is not None and freeze["split_sha256"] != split_sha256(split):
        raise ProtectedDataError("freeze split_sha256 does not match the split in use")
    if freeze["scoring_code_sha256"] != scoring_code_sha256():
        raise ProtectedDataError("scoring code changed since the freeze")
    freeze["_sha256"] = hashlib.sha256(ff.read_bytes()).hexdigest()
    return freeze


def check_access(roles, *, allow_test: bool = False, training: bool = False, split: dict | None = None,
                 freeze_file=None) -> dict:
    prot = sorted({str(r) for r in roles} & set(PROTECTED_ROLES))
    rec = {"roles": sorted({str(r) for r in roles}), "protected": prot, "freeze_sha256": None}
    if not prot:
        return rec
    if training:
        raise ProtectedDataError(f"training code never loads protected roles {prot}")
    if not allow_test:
        raise ProtectedDataError(f"refusing protected roles {prot}: pass --allow-test after the freeze")
    freeze = load_freeze(freeze_file, split=split)
    rec["freeze_sha256"] = freeze["_sha256"]
    rec["freeze"] = freeze
    return rec


def verify_weights(freeze: dict, model: str, artifact: str | Path) -> str:
    sha = artifact_sha256(artifact)
    if freeze["weights_sha256"].get(model) != sha:
        raise ProtectedDataError(f"{artifact} sha256 {sha[:12]} != frozen weights_sha256[{model!r}]")
    return sha
