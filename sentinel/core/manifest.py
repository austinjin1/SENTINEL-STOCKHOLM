# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Dataset manifests: what was downloaded, from where, when, and with what checksums.

Every dataset under ``SENTINEL_DATA`` gets a manifest JSON in ``SENTINEL_DATA/manifests``.
Results in the registry cite manifests by name, so any number can be traced to the
exact files (sha256) it was computed from.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Any, Iterable

from . import paths
from .registry import code_state, file_sha256


def write_manifest(
    name: str,
    *,
    files: Iterable[str | Path],
    sources: Iterable[str],
    description: str,
    license: str = "unknown",
    synthetic: bool = False,
    stats: dict[str, Any] | None = None,
    hash_files: bool = True,
    max_hash_bytes: int = 2 << 30,
) -> Path:
    """Write ``manifests/<name>.json``. ``synthetic`` must be True for any generated data."""
    files = [Path(f) for f in files]
    records = []
    for f in sorted(files):
        if not f.exists():
            raise FileNotFoundError(f)
        size = f.stat().st_size
        rec = {"path": str(f), "bytes": size}
        if hash_files and size <= max_hash_bytes:
            rec["sha256"] = file_sha256(f)
        records.append(rec)
    manifest = {
        "name": name,
        "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "description": description,
        "sources": list(sources),
        "license": license,
        "synthetic": bool(synthetic),
        "n_files": len(records),
        "total_bytes": sum(r["bytes"] for r in records),
        "files": records,
        "stats": stats or {},
        "code": code_state(),
    }
    out = paths.ensure(paths.MANIFESTS) / f"{name}.json"
    out.write_text(json.dumps(manifest, indent=2, default=str))
    return out


def read_manifest(name: str) -> dict[str, Any]:
    return json.loads((paths.MANIFESTS / f"{name}.json").read_text())


def verify_manifest(name: str) -> list[str]:
    """Return a list of problems (missing files or checksum mismatches)."""
    problems = []
    for rec in read_manifest(name)["files"]:
        p = Path(rec["path"])
        if not p.exists():
            problems.append(f"missing: {p}")
        elif "sha256" in rec and file_sha256(p) != rec["sha256"]:
            problems.append(f"checksum mismatch: {p}")
    return problems
