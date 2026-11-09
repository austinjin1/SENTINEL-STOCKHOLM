# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Test-set guard for SENTINEL-HydroMet (RESEARCH_PLAN.md §3 R6).

Protected data = time blocks ``test`` / ``test_b`` and every site whose spatial role is ``test`` (in
any block). Loading or scoring protected data requires BOTH an explicit opt-in (``allow_test=True``
/ ``--allow-test``) AND a valid ``docs/preregistration/FREEZE_hydromet.json`` with keys
:data:`FREEZE_KEYS` (``created``, ``split_sha256`` = split in use, ``scoring_code_sha256`` =
:func:`scoring_code_sha256` of the current code, ``weights_sha256`` = {model: sha256 of its saved
artifact}); optional ``comparators`` ({primary endpoint: baseline}, selected on val only) and ``looks`` (per test
look: models, Holm family, comparators, artifacts, data pins; :mod:`.testlook`). Protected data are loadable only
while an unscored single-pass claim of this freeze exists (:func:`.testlook.require_open_claim`).
Training code refuses protected data unconditionally. Generic helpers (block order,
artifact hashing) are imported from the M1 guard, which this module does not modify.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ...core import paths
from ..sonde.guard import (PROTECTED_BLOCKS, PROTECTED_ROLES, ProtectedDataError, artifact_sha256,  # noqa: F401
                           assert_cube_within, data_end_date, protected_items)

FREEZE_FILE = paths.REPO_ROOT / "docs" / "preregistration" / "FREEZE_hydromet.json"
FREEZE_KEYS = ("created", "split_sha256", "scoring_code_sha256", "weights_sha256")
# Everything whose behaviour determines a frozen test prediction or score (review C3b): ALL HydroMet model
# code (pickled baselines execute the current class code), configs, drivers, the protocol, and the shared
# code they import. Globs are expanded at hash time, so an added file also changes the hash.
FROZEN_GLOBS = ("sentinel/models/hydromet/*.py", "configs/hydromet/*.yaml", "scripts/train/hydromet_*.py",
                "scripts/eval/hydromet_*.py", "docs/preregistration/hydromet_protocol.md")
SHARED_CODE = ("sentinel/eval/metrics.py", "sentinel/eval/bootstrap.py", "sentinel/eval/splits.py",
               "sentinel/data/sonde_splits.py", "sentinel/models/sonde/data.py", "sentinel/models/sonde/guard.py",
               "sentinel/models/sonde/runner.py", "sentinel/core/registry.py", "sentinel/data/hydromet_upstream.py")


def frozen_files(root: Path | None = None) -> list[str]:
    root = Path(root) if root else paths.REPO_ROOT
    files = set(SHARED_CODE)
    for g in FROZEN_GLOBS:
        files |= {str(f.relative_to(root)) for f in root.glob(g) if f.is_file()}
    return sorted(files)


SCORING_CODE = tuple(frozen_files())   # snapshot for inspection; hashing re-expands the globs


def scoring_code_sha256(root: Path | None = None) -> str:
    root = Path(root) if root else paths.REPO_ROOT
    h = hashlib.sha256()
    for rel in frozen_files(root):
        f = root / rel
        h.update(rel.encode())
        h.update(f.read_bytes() if f.is_file() else b"<missing>")
    return h.hexdigest()


def load_freeze(freeze_file: str | Path | None = None, *, split=None, check_code: bool = True) -> dict:
    ff = Path(freeze_file) if freeze_file is not None else FREEZE_FILE
    if not ff.is_file():
        raise ProtectedDataError(f"freeze file {ff} does not exist")
    try:
        freeze = json.loads(ff.read_text())
    except (OSError, ValueError) as e:
        raise ProtectedDataError(f"freeze file {ff} is not valid JSON: {e}") from e
    if not isinstance(freeze, dict):
        raise ProtectedDataError(f"freeze file {ff} must be a JSON object")
    missing = [k for k in FREEZE_KEYS if not freeze.get(k)]
    if missing:
        raise ProtectedDataError(f"freeze file {ff} lacks required keys {missing}")
    if not isinstance(freeze["weights_sha256"], dict):
        raise ProtectedDataError("freeze file weights_sha256 must map model name -> sha256")
    if split is not None and freeze["split_sha256"] != split.sha256():
        raise ProtectedDataError("freeze file split_sha256 does not match the split in use")
    if check_code and freeze["scoring_code_sha256"] != scoring_code_sha256():
        raise ProtectedDataError("scoring code changed since the freeze (scoring_code_sha256 mismatch)")
    freeze["_path"] = str(ff)
    freeze["_sha256"] = hashlib.sha256(ff.read_bytes()).hexdigest()
    return freeze


def verify_weights(freeze: dict, model: str, artifact: str | Path) -> str:
    sha = artifact_sha256(artifact)
    want = freeze["weights_sha256"].get(model)
    if want != sha:
        raise ProtectedDataError(f"artifact {artifact} sha256 {sha[:12]} != frozen weights_sha256[{model!r}]")
    return sha


def check_access(blocks=(), roles=(), *, allow_test: bool = False, freeze_file=None, split=None,
                 training: bool = False) -> dict:
    """Authorise blocks/roles; raise :class:`ProtectedDataError` for protected data unless
    ``allow_test`` AND a valid freeze file — and always when ``training`` is True."""
    prot = protected_items(blocks, roles)
    rec = {"blocks": sorted(set(blocks)), "roles": sorted(set(roles)), "protected": prot,
           "freeze_file": None, "freeze_sha256": None}
    if not prot:
        return rec
    if training:
        raise ProtectedDataError(f"training code never loads protected data {prot}; test predictions come "
                                 "only from scripts/eval/hydromet_predict.py with frozen artifacts")
    if not allow_test:
        raise ProtectedDataError(f"refusing protected data {prot}: pass --allow-test (only after the "
                                 f"configuration is frozen in {FREEZE_FILE.name})")
    try:
        freeze = load_freeze(freeze_file, split=split)
    except ProtectedDataError as e:
        raise ProtectedDataError(f"refusing protected data {prot}: {e}") from e
    from .testlook import require_open_claim   # single-pass guard (pre-test review C1)

    look = require_open_claim(freeze)
    rec.update({"freeze_file": freeze["_path"], "freeze_sha256": freeze["_sha256"], "freeze": freeze, "look": look})
    return rec
