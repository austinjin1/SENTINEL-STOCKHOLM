# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Write-time guard for M4: refuse any ``supersedes`` that does not name an existing row id.

A5 delta review A5.3-a asked for a check that refuses a ``supersedes`` pointing at a non-existent
id. ``sentinel/core/registry.py`` is shared by all seven workstreams, so the refusal is **not**
added there unilaterally — that is a cross-workstream decision for the lead. M4 routes every
registry write that carries ``supersedes`` through :func:`log_result_checked` instead.

This guard would have caught all three of my supersede mistakes: two guessed row ids, and version
*names* (``"prereg_snapshot_v6"``) written into a field that must hold a row *id*.
"""

from __future__ import annotations

import json

from sentinel.core import paths, registry


def existing_ids() -> set[str]:
    path = paths.REPO_ROOT / "results" / "registry.jsonl"
    return {json.loads(l)["id"] for l in path.read_text().splitlines() if l.strip()}


def log_result_checked(**kw):
    """`registry.log_result`, refusing a dangling or malformed ``supersedes``."""
    sup = kw.get("supersedes")
    if sup is not None:
        if sup not in existing_ids():
            raise ValueError(
                f"refusing to write: supersedes={sup!r} names no existing registry row. "
                "`supersedes` must be a row id read from the registry, never a guessed id and "
                "never a version name such as 'prereg_snapshot_v6'.")
    return registry.log_result(**kw)
