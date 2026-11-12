# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Lead-stamp guard of SENTINEL-Lite architecture round v2 (M7; verifier ruling F9, lead order 2026-11-10 15:05 EST).

No ``lite`` or ``litev2`` loader returns a ``val``, ``val_n`` or ``val_p2`` row to a v2 arm, a comparator or a
re-run until :func:`require_stamp` has found ONE registry row with

* ``workstream`` = ``"M7-lite"``, ``experiment`` = ``"lite/prereg_snapshot"``, ``extra.written_by`` = ``"lead"``,
* ``extra.sha256`` mapping each of the four files of :data:`STAMPED_FILES` (repo-relative path) to the sha256 the
  file has on disk now,

and all four files are read-only on disk (no write bit), and the amendment still has the hash it had when it was
submitted (:data:`AMENDMENT_SHA256`). The files are hashed as they are now: an edit or an append after the stamp
(the deviations file is append-only) closes the guard until the lead stamps the new content.

Allowed before the stamp: code, tests, label / index / mask builds (protected rows: counts only), synthetic and
train-only smoke tests. Same convention as ``sentinel/models/microbe/stamp.py`` and ``orbit_v2/stamp.py``.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from ...core import paths

WORKSTREAM = "M7-lite"
STAMP_EXPERIMENT = "lite/prereg_snapshot"
WRITTEN_BY = "lead"
AMENDMENT = "docs/preregistration/lite_arch_v2_amendment.md"
ADDENDUM = "docs/preregistration/lite_arch_v2_label_addendum.md"
DEVIATIONS = "docs/preregistration/lite_arch_v2_deviations.md"
RULINGS = "docs/preregistration/lite_arch_v2_lead_rulings.md"
STAMPED_FILES = (AMENDMENT, ADDENDUM, DEVIATIONS, RULINGS)
# v0.3 as reviewed (docs/preregistration/reviews/arch_v2_20261110/M7_verify.json): stamped byte for byte
AMENDMENT_SHA256 = "2b369018d2fc2a340594790756b3edbef3b1de28a38cf6073eb5908111908a56"
VAL_ROLES = ("val", "val_n", "val_p2")


class StampError(PermissionError):
    """A validation row of architecture round v2 was requested before the lead's stamp."""


def file_state(repo_root: Path | None = None) -> dict:
    """{path: {'exists', 'sha256', 'read_only'}} of the four stamped files as they are on disk now."""
    root = Path(repo_root) if repo_root else paths.REPO_ROOT
    out = {}
    for rel in STAMPED_FILES:
        f = root / rel
        if not f.is_file():
            out[rel] = {"exists": False, "sha256": None, "read_only": False}
            continue
        out[rel] = {"exists": True, "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
                    "read_only": (os.stat(f).st_mode & 0o222) == 0}
    return out


def stamp_status(registry_path: Path | None = None, repo_root: Path | None = None) -> dict:
    """{'stamped', 'reason', 'row_id', 'timestamp', 'n_rows', 'n_lead_rows', 'files'}.

    ``n_rows``: registry rows of the stamp experiment and workstream; ``n_lead_rows``: those written by the lead.
    ``reason`` is ``"stamped"`` or says what is missing (first failing condition)."""
    files = file_state(repo_root)
    reg = Path(registry_path) if registry_path else paths.REGISTRY_PATH
    out = {"stamped": False, "reason": "", "row_id": None, "timestamp": None, "n_rows": 0, "n_lead_rows": 0,
           "files": files}
    hit = None
    if reg.is_file():
        with reg.open(encoding="utf-8") as fh:
            for line in fh:
                if STAMP_EXPERIMENT not in line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("experiment") != STAMP_EXPERIMENT or r.get("workstream") != WORKSTREAM:
                    continue
                out["n_rows"] += 1
                extra = r.get("extra") or {}
                if extra.get("written_by") != WRITTEN_BY:
                    continue
                out["n_lead_rows"] += 1
                shas = extra.get("sha256") or {}
                if isinstance(shas, dict) and all(
                        files[f]["exists"] and shas.get(f) == files[f]["sha256"] for f in STAMPED_FILES):
                    hit = r
    missing = [f for f in STAMPED_FILES if not files[f]["exists"]]
    writable = [f for f in STAMPED_FILES if files[f]["exists"] and not files[f]["read_only"]]
    if missing:
        out["reason"] = f"not stamped: {len(missing)} of the four files do not exist ({', '.join(missing)})"
    elif files[AMENDMENT]["sha256"] != AMENDMENT_SHA256:
        out["reason"] = "not stamped: the amendment differs from the reviewed v0.3"
    elif hit is None:
        out["reason"] = (f"not stamped: no {STAMP_EXPERIMENT} row of workstream {WORKSTREAM} written by the "
                         f"{WRITTEN_BY} holds the four files at their current hashes "
                         f"({out['n_rows']} rows of the experiment, {out['n_lead_rows']} by the lead)")
    elif writable:
        out["reason"] = f"not stamped: {len(writable)} of the four files are not read-only ({', '.join(writable)})"
    else:
        out.update(stamped=True, reason="stamped", row_id=hit.get("id"), timestamp=hit.get("timestamp"))
    return out


def is_stamped(registry_path: Path | None = None, repo_root: Path | None = None) -> bool:
    return bool(stamp_status(registry_path, repo_root)["stamped"])


def require_stamp(what: str, registry_path: Path | None = None, repo_root: Path | None = None) -> dict:
    """Raise :class:`StampError` unless the round is stamped. ``what``: what was asked for (for the message)."""
    st = stamp_status(registry_path, repo_root)
    if not st["stamped"]:
        raise StampError(f"{what}: refused — {st['reason']}. Allowed before the stamp: code, tests, label / index / "
                         "mask builds (protected rows: counts only), synthetic and train-only smoke tests.")
    return st


def guard_roles(roles, what: str, registry_path: Path | None = None, repo_root: Path | None = None) -> None:
    """Call before returning rows of ``roles`` to a v2 arm, a comparator or a re-run: validation roles need the
    stamp; training roles pass."""
    asked = sorted({str(r) for r in roles} & set(VAL_ROLES))
    if asked:
        require_stamp(f"{what}: rows of roles {asked}", registry_path, repo_root)


__all__ = ["WORKSTREAM", "STAMP_EXPERIMENT", "WRITTEN_BY", "STAMPED_FILES", "AMENDMENT_SHA256", "VAL_ROLES",
           "StampError", "file_state", "stamp_status", "is_stamped", "require_stamp", "guard_roles"]


if __name__ == "__main__":
    s = stamp_status()
    print(json.dumps({k: v for k, v in s.items()}, indent=1))
