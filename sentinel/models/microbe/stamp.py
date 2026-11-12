# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Lead-stamp gate for the M3 architecture round v2 (LEAD_DECISIONS_20261110.md rule 2).

Before the lead stamps ``docs/preregistration/microbe_arch_v2_amendment.md`` (a registry row of experiment
``microbe/prereg_snapshot`` whose ``extra.sha256`` maps the amendment path to its current sha256 and whose
``extra.written_by`` is "lead"), only code, tests,
data builds (protected rows: counts only) and synthetic / train-only smoke tests are allowed. Every code path that
would produce a development number of a new arm (D1-D4), of a parity comparator added by the amendment, or on any
GEO / GAS / EA target calls :func:`require_stamp` first.

The gate hashes the file as it is on disk now: an edit after the stamp closes the gate again. Same convention as
``sentinel/models/orbit_v2/stamp.py``.

Second gate (lead stamp message of 2026-11-10, step 0a; rulings file L3): work the stamp did not release — D2
NicheGeo-Phys, the added EA family, H4, GAS-3 — and every post-stamp reading that fixes a design choice needs a
deviations file ``docs/preregistration/microbe_arch_v2_deviations_NN.md`` that

* declares what it releases on one line ``**Releases:** KEY[, KEY…]`` (keys: :data:`RELEASE_KEYS`), and
* is stamped by the lead: a registry row of experiment ``microbe/prereg_snapshot`` whose ``extra.sha256`` maps the
  file's repo-relative path to its current sha256.

Only rows with ``extra.written_by == "lead"`` count, for the amendment and for a deviations file (dev01 lead ruling
D11). M3 writes no row of that experiment.

:func:`require_release` checks the amendment stamp and then this. A deviations file edited after its stamp closes
its gate again.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from sentinel.core import paths

AMENDMENT = "docs/preregistration/microbe_arch_v2_amendment.md"
STAMP_EXPERIMENT = "microbe/prereg_snapshot"
DEVIATIONS_DIR = "docs/preregistration"
DEVIATIONS_PATTERN = "microbe_arch_v2_deviations_*.md"
RELEASE_LINE = re.compile(r"^\*\*Releases:\*\*\s*(.+?)\s*$", re.M)
WRITTEN_BY = "lead"                 # dev01 lead ruling D11: only rows with extra.written_by == "lead" count
# D2 = NicheGeo-Phys (with its ablations A2, A3); EA = the added EA family (any EA label or fit); H4 = the
# letter-group-blocked hypothesis; GAS = GAS-3; the others are the post-stamp readings of deviations file 01.
RELEASE_KEYS = ("D2", "EA", "H4", "GAS", "A0", "XGB_V4CLR_MO", "CALIBRATION", "NICHE_S_QMP", "GEO_POWER")


class StampError(RuntimeError):
    """A development number of architecture round v2 was requested before the lead's stamp."""


def amendment_sha256(repo_root: Path | None = None) -> str:
    return hashlib.sha256(((repo_root or paths.REPO_ROOT) / AMENDMENT).read_bytes()).hexdigest()


def stamp_status(registry_path: Path | None = None, repo_root: Path | None = None) -> dict:
    """{'stamped', 'amendment_sha256', 'row_id', 'timestamp', 'n_rows'}; ``n_rows`` counts registry rows of the
    stamp experiment whether or not they match the current file content or carry ``extra.written_by`` = "lead";
    only rows that carry it can stamp."""
    sha = amendment_sha256(repo_root)
    reg = Path(registry_path) if registry_path else paths.REGISTRY_PATH
    out = {"stamped": False, "amendment_sha256": sha, "row_id": None, "timestamp": None, "n_rows": 0}
    if not reg.is_file():
        return out
    with reg.open() as fh:
        for line in fh:
            if STAMP_EXPERIMENT not in line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("experiment") != STAMP_EXPERIMENT:
                continue
            out["n_rows"] += 1
            if (r.get("extra") or {}).get("written_by") != WRITTEN_BY:
                continue
            shas = (r.get("extra") or {}).get("sha256") or {}
            if isinstance(shas, dict) and shas.get(AMENDMENT) == sha:
                out.update(stamped=True, row_id=r.get("id"), timestamp=r.get("timestamp"))
    return out


def require_stamp(what: str, registry_path: Path | None = None, repo_root: Path | None = None) -> dict:
    st = stamp_status(registry_path, repo_root)
    if not st["stamped"]:
        raise StampError(
            f"{what}: refused before the lead's stamp of {AMENDMENT} (sha256 {st['amendment_sha256'][:12]}…; "
            f"{st['n_rows']} rows of {STAMP_EXPERIMENT} in the registry, none written by the lead for this file "
            "content). "
            "Allowed before the stamp: code, tests, data builds, synthetic / train-only smoke tests.")
    return st


def _stamped_shas(registry_path: Path | None = None) -> dict:
    """{repo-relative path: {sha256: (row id, timestamp)}} over the rows of the stamp experiment that carry
    ``extra.written_by`` = "lead"."""
    reg = Path(registry_path) if registry_path else paths.REGISTRY_PATH
    out: dict = {}
    if not reg.is_file():
        return out
    with reg.open() as fh:
        for line in fh:
            if STAMP_EXPERIMENT not in line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("experiment") != STAMP_EXPERIMENT:
                continue
            if (r.get("extra") or {}).get("written_by") != WRITTEN_BY:
                continue
            shas = (r.get("extra") or {}).get("sha256") or {}
            if isinstance(shas, dict):
                for k, v in shas.items():
                    out.setdefault(str(k), {})[str(v)] = (r.get("id"), r.get("timestamp"))
    return out


def declared_releases(text: str) -> tuple:
    keys = []
    for m in RELEASE_LINE.finditer(text):
        keys += [k.strip() for k in m.group(1).split(",") if k.strip()]
    return tuple(keys)


def release_status(key: str, registry_path: Path | None = None, repo_root: Path | None = None) -> dict:
    """{'released', 'key', 'file', 'sha256', 'row_id', 'timestamp', 'declared_in'}. ``declared_in`` lists every
    deviations file that declares the key, stamped or not."""
    if key not in RELEASE_KEYS:
        raise KeyError(f"unknown release key {key!r}; known: {RELEASE_KEYS}")
    root = Path(repo_root) if repo_root else paths.REPO_ROOT
    stamped = _stamped_shas(registry_path)
    out = {"released": False, "key": key, "file": None, "sha256": None, "row_id": None, "timestamp": None,
           "declared_in": []}
    for f in sorted((root / DEVIATIONS_DIR).glob(DEVIATIONS_PATTERN)):
        raw = f.read_bytes()
        if key not in declared_releases(raw.decode("utf-8", "replace")):
            continue
        rel = f"{DEVIATIONS_DIR}/{f.name}"
        sha = hashlib.sha256(raw).hexdigest()
        out["declared_in"].append(rel)
        hit = stamped.get(rel, {}).get(sha)
        if hit:
            out.update(released=True, file=rel, sha256=sha, row_id=hit[0], timestamp=hit[1])
    return out


def require_release(what: str, key: str, registry_path: Path | None = None, repo_root: Path | None = None) -> dict:
    """Amendment stamp AND a lead-stamped deviations file that releases ``key``."""
    require_stamp(what, registry_path, repo_root)
    st = release_status(key, registry_path, repo_root)
    if not st["released"]:
        where = ", ".join(st["declared_in"]) or "no deviations file declares it"
        raise StampError(
            f"{what}: '{key}' is not released. It needs a {DEVIATIONS_DIR}/{DEVIATIONS_PATTERN} file with the line "
            f"'**Releases:** {key}' and a {STAMP_EXPERIMENT} registry row carrying that file's current sha256 "
            f"(declared in: {where}; not stamped at its current content).")
    return st
