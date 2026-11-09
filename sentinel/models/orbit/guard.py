# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Test-set guard for SENTINEL-Orbit (RESEARCH_PLAN.md §3 R6; orbit_protocol.md §9).

Protected rows = every row whose spatial role is ``test`` (any year) and every row in the ``test`` /
``test_b`` time blocks (any role) — see :mod:`.split`. Loading protected rows requires BOTH an explicit
opt-in (``allow_test=True`` / ``--allow-test``) AND a valid ``docs/preregistration/FREEZE_orbit.json``
with keys :data:`FREEZE_KEYS`:

* ``created``, ``split_sha256`` (= sha256 of docs/preregistration/orbit_split_v3.json), ``data_version``
  and ``data_manifest_sha256`` (sha256 of manifests/orbit_<data_version>.json),
* ``scoring_code_sha256`` (= :func:`scoring_code_sha256` of the current code),
* ``weights_sha256`` ({model: sha256 of its saved artifact}), ``comparators`` ({endpoint: baseline},
  selected on validation only).

Training code (``training=True``) refuses protected rows unconditionally. Generic helpers
(``ProtectedDataError``, ``artifact_sha256``) are imported from the M1 guard (not modified).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ...core import paths
from ..sonde.guard import ProtectedDataError, artifact_sha256  # noqa: F401
from . import split as SP

FREEZE_FILE = paths.REPO_ROOT / "docs" / "preregistration" / "FREEZE_orbit.json"
# the reviewed candidate (written by scripts/eval/orbit_prepare_freeze.py); it NEVER unlocks protected rows
CANDIDATE_FILE = paths.REPO_ROOT / "docs" / "preregistration" / "orbit_freeze_candidate.json"
FREEZE_KEYS = ("created", "split_sha256", "scoring_code_sha256", "weights_sha256", "comparators", "data_version",
               "data_manifest_sha256", "data_pins", "status", "authorised_by", "authorised", "candidate_sha256")
# keys the lead adds when copying the candidate to FREEZE_orbit.json; every other key must equal the candidate's
AUTH_KEYS = ("status", "authorised_by", "authorised", "candidate_sha256")
FROZEN_GLOBS = ("sentinel/models/orbit/*.py", "configs/orbit/*.yaml", "scripts/train/orbit_*.py",
                "scripts/eval/orbit_*.py", "docs/preregistration/orbit_protocol.md",
                "docs/preregistration/orbit_split_v3.json")
# M7-facing teacher code and post-hoc diagnostics do not determine a test score (review W9)
NOT_FROZEN = ("sentinel/models/orbit/teacher.py", "scripts/train/orbit_teacher.py",
              "scripts/train/orbit_gloria_trackr.py")      # the placebo (pre-registered H3 control) IS frozen
SHARED_CODE = ("sentinel/eval/metrics.py", "sentinel/eval/bootstrap.py", "sentinel/eval/splits.py",
               "sentinel/data/sources/hydrolakes.py", "sentinel/data/chips.py", "sentinel/data/sources/stac.py")
# event track (orbit_protocol.md §12.5; event_layer_plan.md §4.1): outside the orbit globs, added DELIBERATELY to the
# scoring-code hash because orbit_test.py scores the event endpoints in the same look
EVENT_CODE = ("sentinel/eval/events.py", "sentinel/events/__init__.py", "sentinel/events/orbit.py",
              "scripts/eval/events_orbit.py", "configs/events/orbit.yaml")


def frozen_files(root: Path | None = None) -> list[str]:
    root = Path(root) if root else paths.REPO_ROOT
    files = set(SHARED_CODE) | set(EVENT_CODE)
    for g in FROZEN_GLOBS:
        files |= {str(f.relative_to(root)) for f in root.glob(g) if f.is_file()}
    return sorted(files - set(NOT_FROZEN))


def scoring_code_sha256(root: Path | None = None) -> str:
    root = Path(root) if root else paths.REPO_ROOT
    h = hashlib.sha256()
    for rel in frozen_files(root):
        f = root / rel
        h.update(rel.encode())
        h.update(f.read_bytes() if f.is_file() else b"<missing>")
    return h.hexdigest()


def load_freeze(freeze_file: str | Path | None = None, *, check_code: bool = True) -> dict:
    """Only the lead-written ``FREEZE_orbit.json`` unlocks protected rows (orbit_protocol.md §9b): any other path
    (e.g. the candidate) is refused, ``status`` must be ``FROZEN`` with ``authorised_by``/``authorised``, and
    ``candidate_sha256`` must match the reviewed candidate, whose keys (except the lead's AUTH_KEYS) must be equal."""
    ff = Path(freeze_file) if freeze_file is not None else FREEZE_FILE
    if ff.resolve() != FREEZE_FILE.resolve():
        raise ProtectedDataError(f"only {FREEZE_FILE} unlocks protected rows (got {ff})")
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
    if freeze["status"] != "FROZEN":
        raise ProtectedDataError(f"freeze status {freeze['status']!r} != 'FROZEN' (a candidate never unlocks test)")
    if not CANDIDATE_FILE.is_file() or hashlib.sha256(CANDIDATE_FILE.read_bytes()).hexdigest() != freeze["candidate_sha256"]:
        raise ProtectedDataError("candidate_sha256 does not match docs/preregistration/orbit_freeze_candidate.json")
    cand = json.loads(CANDIDATE_FILE.read_text())
    diff = sorted(k for k in set(cand) | set(freeze) if k not in AUTH_KEYS and cand.get(k) != freeze.get(k))
    if diff:
        raise ProtectedDataError(f"FREEZE_orbit.json differs from the reviewed candidate in {diff}")
    if freeze["split_sha256"] != SP.split_sha256():
        raise ProtectedDataError("freeze file split_sha256 does not match docs/preregistration/orbit_split_v3.json")
    man_path = paths.MANIFESTS / f"orbit_{freeze['data_version']}.json"     # review W9: pin the data
    if not man_path.is_file() or hashlib.sha256(man_path.read_bytes()).hexdigest() != freeze["data_manifest_sha256"]:
        raise ProtectedDataError(f"data manifest {man_path.name} missing or sha256 != frozen data_manifest_sha256")
    from ...core.manifest import verify_manifest

    problems = verify_manifest(f"orbit_{freeze['data_version']}")
    if problems:
        raise ProtectedDataError(f"frozen data files changed: {problems[:3]}")
    if check_code and freeze["scoring_code_sha256"] != scoring_code_sha256():
        raise ProtectedDataError("scoring code changed since the freeze (scoring_code_sha256 mismatch)")
    freeze["_path"] = str(ff)
    freeze["_sha256"] = hashlib.sha256(ff.read_bytes()).hexdigest()
    return freeze


def verify_weights(freeze: dict, model: str, artifact: str | Path) -> str:
    sha = artifact_sha256(artifact)
    if freeze["weights_sha256"].get(model) != sha:
        raise ProtectedDataError(f"artifact {artifact} sha256 {sha[:12]} != frozen weights_sha256[{model!r}]")
    return sha


def check_rows(protected_mask, *, allow_test: bool = False, training: bool = False, freeze_file=None,
               what: str = "rows") -> dict:
    """Authorise a set of rows given their ``protected`` flags (from :func:`split.assign`).

    Raises :class:`ProtectedDataError` if any row is protected, unless ``allow_test`` and a valid freeze
    file — and always when ``training`` is True."""
    n_prot = int(np.asarray(protected_mask, bool).sum())
    rec = {"n_protected": n_prot, "freeze_file": None, "freeze_sha256": None}
    if n_prot == 0:
        return rec
    if training:
        raise ProtectedDataError(f"training code never loads protected {what} ({n_prot} test-role or "
                                 "test-year rows); test predictions come only from scripts/eval/orbit_predict.py")
    if not allow_test:
        raise ProtectedDataError(f"refusing {n_prot} protected {what}: pass --allow-test (only after the "
                                 f"configuration is frozen in {FREEZE_FILE.name})")
    freeze = load_freeze(freeze_file)
    from .testlook import require_open_claim       # protected rows only inside an open, unscored claim (§9b)

    rec.update({"freeze_file": freeze["_path"], "freeze_sha256": freeze["_sha256"],
                "look": require_open_claim(freeze)})
    return rec


def drop_protected(df, *, allow_test: bool = False, freeze_file=None, log=print):
    """Return ``df`` without protected rows unless access is authorised (then unchanged).
    Used by the data builder so protected labels are never written to development caches."""
    prot = df["protected"].to_numpy(bool)
    if not prot.any():
        return df, {"n_protected": 0}
    if allow_test:
        rec = check_rows(prot, allow_test=True, freeze_file=freeze_file)
        return df, rec
    log(f"[guard] dropping {int(prot.sum())} protected rows (test role / test years) before any label use")
    return df[~prot].reset_index(drop=True), {"n_protected_dropped": int(prot.sum())}
