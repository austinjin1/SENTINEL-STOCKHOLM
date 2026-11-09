# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Test-set guard for SENTINEL-Sonde (RESEARCH_PLAN.md §3 R6, §12.5).

Protected data = time blocks ``test`` (WY2024–2025) and ``test_b`` (WY2026), and every site whose
spatial role is ``test`` (in any time block, including the training years — that is the
"unseen × train-period" cell). Loading or scoring protected data requires BOTH

  1. an explicit opt-in (``allow_test=True`` in code, ``--allow-test`` on the command line), AND
  2. the freeze file ``docs/preregistration/FREEZE_sonde.json``, a JSON object with the required keys
     :data:`FREEZE_KEYS`: ``created`` (date), ``split_sha256`` (must equal the split in use),
     ``scoring_code_sha256`` (must equal :func:`scoring_code_sha256` of the current code),
     ``weights_sha256`` ({model name: sha256 of its checkpoint, see :func:`artifact_sha256}`; checked
     by scripts/eval/sonde_predict.py) and ``thresholds_sha256`` (sha256 of the frozen EW calibration
     file; checked by the score phase of scripts/train/sonde_events.py).

Without them, :func:`check_access` raises :class:`ProtectedDataError`, and loaded arrays are cut at
the end of the last *permitted* block (:func:`data_end_date`) and to permitted site roles. (Site
files are read whole and then cut in memory, so protected values are transiently read from disk but
never enter any array, feature, model or score during development.)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ...core import paths
from ...data.sonde_splits import TIME_BLOCKS

FREEZE_FILE = paths.REPO_ROOT / "docs" / "preregistration" / "FREEZE_sonde.json"
FREEZE_FILE_EW = paths.REPO_ROOT / "docs" / "preregistration" / "FREEZE_ew.json"   # EW-v2-only record
FREEZE_KEYS = ("created", "split_sha256", "scoring_code_sha256", "weights_sha256", "thresholds_sha256")
# Code whose behaviour determines a test score (hashed into the freeze record).
SCORING_CODE = ("sentinel/eval/metrics.py", "sentinel/eval/bootstrap.py", "sentinel/eval/early_warning.py",
                "sentinel/eval/hazards.py", "sentinel/eval/ew_scoring.py", "sentinel/data/sonde_splits.py",
                "sentinel/data/sonde_daily.py", "sentinel/models/baselines/sonde_baselines.py",
                "sentinel/models/sonde/data.py", "sentinel/models/sonde/common.py",
                "sentinel/models/sonde/evaluate.py", "sentinel/models/sonde/events.py",
                "sentinel/models/sonde/guard.py", "scripts/eval/sonde_score.py", "scripts/eval/sonde_predict.py",
                "scripts/train/sonde_events.py")
# Minimum file set an EW-v2-only freeze record must hash (event labels, features, heads, scoring).
EW_SCORING_CODE = ("sentinel/eval/metrics.py", "sentinel/eval/bootstrap.py", "sentinel/eval/early_warning.py",
                   "sentinel/eval/hazards.py", "sentinel/eval/ew_scoring.py", "sentinel/eval/ew_controls.py",
                   "sentinel/eval/chloride.py", "sentinel/eval/splits.py", "sentinel/models/sonde/evaluate.py",
                   "sentinel/data/sonde_splits.py", "sentinel/data/sonde_daily.py",
                   "sentinel/models/baselines/sonde_baselines.py", "sentinel/models/sonde/data.py",
                   "sentinel/models/sonde/common.py", "sentinel/models/sonde/features.py",
                   "sentinel/models/sonde/subdaily.py", "sentinel/models/sonde/events.py",
                   "sentinel/models/sonde/guard.py", "sentinel/models/sonde/runner.py",
                   "scripts/train/sonde_events.py", "configs/sonde/common.yaml", "configs/sonde/events.yaml")
PROTECTED_BLOCKS = ("test", "test_b")
PROTECTED_ROLES = ("test",)
BLOCK_ORDER = ("train", "val", "test", "test_b")


class ProtectedDataError(PermissionError):
    """Raised when code asks for test-block or test-site data without flag + freeze file."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scoring_code_sha256(root: Path | None = None, files=None) -> str:
    """SHA-256 over (relative path, content) of ``files`` (default :data:`SCORING_CODE`). A freeze
    record may list its own ``scoring_code_files`` (e.g. the EW-v2 freeze, made before the M1
    forecasting freeze), so that each record is self-describing and later edits to files outside
    that list cannot invalidate it."""
    root = Path(root) if root else paths.REPO_ROOT
    h = hashlib.sha256()
    for rel in (SCORING_CODE if files is None else tuple(files)):
        f = root / rel
        h.update(rel.encode())
        h.update(f.read_bytes() if f.is_file() else b"<missing>")
    return h.hexdigest()


def artifact_sha256(path: str | Path) -> str:
    """SHA-256 of a checkpoint: a file's bytes, or for a directory the hash of its sorted
    (relative path, file sha256) list (e.g. GBM model dirs, Chronos checkpoints)."""
    p = Path(path)
    if p.is_file():
        return _sha256(p)
    if not p.is_dir():
        raise FileNotFoundError(p)
    h = hashlib.sha256()
    for f in sorted(x for x in p.rglob("*") if x.is_file()):
        h.update(str(f.relative_to(p)).encode())
        h.update(_sha256(f).encode())
    return h.hexdigest()


def load_freeze(freeze_file: str | Path | None = None, *, split=None, check_code: bool = True) -> dict:
    """Load and validate the freeze record; raise ProtectedDataError on any problem."""
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
    files = freeze.get("scoring_code_files")
    if files is not None and (not isinstance(files, list)
                              or not (set(files) >= set(SCORING_CODE) or set(files) >= set(EW_SCORING_CODE))):
        raise ProtectedDataError("freeze file scoring_code_files must cover SCORING_CODE or EW_SCORING_CODE")
    if check_code and freeze["scoring_code_sha256"] != scoring_code_sha256(files=files):
        raise ProtectedDataError("scoring code changed since the freeze (scoring_code_sha256 mismatch)")
    freeze["_path"], freeze["_sha256"] = str(ff), _sha256(ff)
    return freeze


def verify_weights(freeze: dict, model: str, checkpoint: str | Path) -> str:
    sha = artifact_sha256(checkpoint)
    want = freeze["weights_sha256"].get(model)
    if want != sha:
        raise ProtectedDataError(f"checkpoint {checkpoint} sha256 {sha[:12]} does not match frozen "
                                 f"weights_sha256[{model!r}] = {str(want)[:12]}")
    return sha


def verify_thresholds(freeze: dict, calibration_file: str | Path) -> str:
    sha = _sha256(Path(calibration_file))
    if freeze["thresholds_sha256"] != sha:
        raise ProtectedDataError(f"EW calibration file sha256 {sha[:12]} does not match frozen thresholds_sha256")
    return sha


def protected_items(blocks=(), roles=()) -> list[str]:
    blocks, roles = set(blocks), set(roles)
    unknown = blocks - set(TIME_BLOCKS)
    if unknown:
        raise ValueError(f"unknown time block(s) {sorted(unknown)}; known: {sorted(TIME_BLOCKS)}")
    return sorted(blocks & set(PROTECTED_BLOCKS)) + sorted(f"role:{r}" for r in roles & set(PROTECTED_ROLES))


def check_access(blocks=(), roles=(), *, allow_test: bool = False, freeze_file: str | Path | None = None,
                 split=None) -> dict:
    """Authorise access to the given time blocks / site roles; raise if protected data is
    requested without ``allow_test`` AND a valid freeze file. Returns an access record that
    drivers store next to their outputs."""
    prot = protected_items(blocks, roles)
    rec = {"blocks": sorted(set(blocks)), "roles": sorted(set(roles)), "protected": prot,
           "freeze_file": None, "freeze_sha256": None}
    if not prot:
        return rec
    if not allow_test:
        raise ProtectedDataError(
            f"refusing to load/score protected data {prot}: pass allow_test=True / --allow-test "
            f"(only after the configuration is frozen in {FREEZE_FILE.name})")
    try:
        freeze = load_freeze(freeze_file, split=split)
    except ProtectedDataError as e:
        raise ProtectedDataError(f"refusing to load/score protected data {prot}: {e}") from e
    rec.update({"freeze_file": freeze["_path"], "freeze_sha256": freeze["_sha256"], "freeze": freeze})
    return rec


def data_end_date(blocks) -> str:
    """Last calendar date that may be loaded when only ``blocks`` are authorised.

    Loading always starts at the training block; it ends at the end of the latest requested
    block, so e.g. ``("train", "val")`` never reads WY2024+ values.
    """
    blocks = set(blocks)
    if not blocks:
        raise ValueError("no blocks requested")
    last = max(blocks, key=BLOCK_ORDER.index)
    return TIME_BLOCKS[last][1]


def assert_cube_within(dates, blocks) -> None:
    """Defensive check: a loaded date index must not extend past the authorised blocks."""
    import pandas as pd

    end = pd.Timestamp(data_end_date(blocks))
    if len(dates) and pd.Timestamp(dates[-1]) > end:
        raise ProtectedDataError(f"data extend to {dates[-1]} but only blocks {sorted(blocks)} "
                                 f"(<= {end.date()}) are authorised")
