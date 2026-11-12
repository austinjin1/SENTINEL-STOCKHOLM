# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Lead-stamp gate for architecture round v2 (LEAD_DECISIONS_20261110.md rule 2; lead ruling R-23).

The stamp is ONE registry row of experiment ``orbit/prereg_snapshot`` that the lead wrote. It counts only if

* ``extra.sha256`` maps BOTH :data:`AMENDMENT` and :data:`RULINGS` to the sha256 of the files as they are on disk
  now (an edit of either file after the stamp closes the gate again);
* both files are read-only (no write bit for anyone);
* ``extra.written_by == "lead"``.

Every code path that would produce a validation number of a new arm, of a parity comparator refit or of a Track N
target calls :func:`require_stamp` first. No code of this workstream writes a row of the stamp experiment.
"""

from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path

from ...core import paths

AMENDMENT = "docs/preregistration/orbit_arch_v2_amendment.md"
RULINGS = "docs/preregistration/orbit_arch_v2_lead_rulings.md"
STAMPED_FILES = (AMENDMENT, RULINGS)
STAMP_EXPERIMENT = "orbit/prereg_snapshot"
WRITTEN_BY = "lead"


class StampError(RuntimeError):
    """A validation number of architecture round v2 was requested without a valid lead stamp."""


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def amendment_sha256(repo_root: Path | None = None) -> str:
    return _sha((repo_root or paths.REPO_ROOT) / AMENDMENT)


def file_state(repo_root: Path | None = None) -> dict:
    """{path: {'exists', 'sha256', 'read_only'}} of the two stamped files."""
    root = repo_root or paths.REPO_ROOT
    out = {}
    for rel in STAMPED_FILES:
        p = root / rel
        if p.is_file():
            out[rel] = {"exists": True, "sha256": _sha(p),
                        "read_only": not (p.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))}
        else:
            out[rel] = {"exists": False, "sha256": None, "read_only": False}
    return out


def stamp_status(registry_path: Path | None = None, repo_root: Path | None = None) -> dict:
    """{'stamped', 'reason', 'amendment_sha256', 'rulings_sha256', 'row_id', 'timestamp', 'n_rows'}.

    ``n_rows`` counts registry rows of the stamp experiment, valid or not. With several valid rows the last one is
    reported. ``reason`` says why the gate is closed (or 'stamped')."""
    files = file_state(repo_root)
    out = {"stamped": False, "reason": "", "amendment_sha256": files[AMENDMENT]["sha256"],
           "rulings_sha256": files[RULINGS]["sha256"], "row_id": None, "timestamp": None, "n_rows": 0}
    missing = [rel for rel, f in files.items() if not f["exists"]]
    if missing:
        out["reason"] = f"missing file(s) {missing}"
        return out
    writable = [rel for rel, f in files.items() if not f["read_only"]]
    reg = Path(registry_path) if registry_path else (repo_root or paths.REPO_ROOT) / "results" / "registry.jsonl"
    seen = {"hash": 0, "lead": 0}
    if reg.is_file():
        with reg.open() as fh:
            for line in fh:
                if STAMP_EXPERIMENT not in line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(r, dict) or r.get("experiment") != STAMP_EXPERIMENT:
                    continue
                out["n_rows"] += 1
                extra = r.get("extra") if isinstance(r.get("extra"), dict) else {}
                shas = extra.get("sha256") if isinstance(extra.get("sha256"), dict) else {}
                if any(shas.get(rel) != files[rel]["sha256"] for rel in STAMPED_FILES):
                    continue
                seen["hash"] += 1
                if extra.get("written_by") != WRITTEN_BY:
                    continue
                seen["lead"] += 1
                if not writable:
                    out.update(stamped=True, row_id=r.get("id"), timestamp=r.get("timestamp"))
    if out["stamped"]:
        out["reason"] = "stamped"
    elif writable:
        out["reason"] = f"not read-only: {writable}"
    elif seen["hash"] and not seen["lead"]:
        out["reason"] = f"{seen['hash']} row(s) carry both hashes but none has extra.written_by == {WRITTEN_BY!r}"
    else:
        out["reason"] = (f"no row of {STAMP_EXPERIMENT} maps both {AMENDMENT} and {RULINGS} to their current "
                         f"sha256 ({out['n_rows']} rows of the experiment)")
    return out


def require_stamp(what: str, registry_path: Path | None = None, repo_root: Path | None = None) -> dict:
    st = stamp_status(registry_path, repo_root)
    if not st["stamped"]:
        raise StampError(f"{what}: refused before the lead's stamp of {AMENDMENT} and {RULINGS} ({st['reason']}). "
                         "Allowed without it: code, tests, keys-only builds, synthetic / train-only smoke tests.")
    return st
