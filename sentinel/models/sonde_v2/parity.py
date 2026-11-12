# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Stamp guard, declaration record and parity log of the SENTINEL-Sonde architecture round v2.

* :func:`require_stamp` (verifier item F04): no driver writes a prediction or a registry row unless
  (a) ``docs/preregistration/sonde_arch_v2_deviations_v2.md`` (and every later ``sonde_arch_v2_deviations_v2_*.md``)
  exists and is read-only, (b) ONE registry row of workstream ``M1-sonde``, experiment ``sonde/prereg_snapshot``
  with ``extra.written_by == "lead"`` maps every such file to the sha256 it has on disk now and carries
  ``extra.code_sha256`` equal to :func:`code_snapshot_sha256` of the code on disk now, and (c) the sha256 list
  between the ``CODE-SHA256`` markers of those files (a later file overrides an earlier one per path) lists every
  v2 code file with the hash it has on disk now. Any edit of a listed file, a new unlisted v2 code file or an edit
  of a deviations file closes the guard until the lead stamps the new state.
* :func:`declare` / :func:`require_declared` (lead ruling L, verifier item P2-10): the declaration record
  (:func:`declared.declaration`) is written once, its sha256 goes to the parity log and the registry, and a driver
  refuses to run when the code's current declaration differs from the declared one.
* :func:`record` / :func:`check` (verifier item F15): one JSON line per run (site hashes, static columns, input
  hashes, row counts, fit window, configuration); :func:`check` compares the arms of a fold and raises on a
  mismatch. It runs before any fold is combined and before any CV score.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from sentinel.core import paths

from . import declared as D

LOG = paths.RUNS_ROOT / "sonde" / "v2_cv" / "PARITY_LOG.jsonl"
DECLARATION = paths.RUNS_ROOT / "sonde" / "v2_cv" / "DECLARATION.json"
WORKSTREAM = "M1-sonde"
STAMP_EXPERIMENT = "sonde/prereg_snapshot"
DECL_EXPERIMENT = "sonde_arch_v2/declaration"
DEV_DIR = "docs/preregistration"
DEV_MAIN = "sonde_arch_v2_deviations_v2.md"
DEV_GLOB = "sonde_arch_v2_deviations_v2*.md"
CODE_GLOBS = ("sentinel/models/sonde_v2/*.py", "scripts/train/sonde_v2_*.py", "tests/test_sonde_v2*.py")
MARK_BEGIN, MARK_END = "<!-- CODE-SHA256-BEGIN -->", "<!-- CODE-SHA256-END -->"


class StampError(PermissionError):
    """A prediction or registry row of architecture round v2 was requested without a valid lead stamp."""


class ParityError(RuntimeError):
    """Arms of one fold disagree on sites, statics or inputs, or a run is not declared."""


# ----------------------------------------------------------------------------------------------- stamp guard

def _sha_file(f: Path) -> str:
    return hashlib.sha256(f.read_bytes()).hexdigest()


def code_files(repo_root: Path | None = None) -> dict:
    """{repo-relative path: sha256} of every v2 code file on disk now."""
    root = Path(repo_root) if repo_root else paths.REPO_ROOT
    out = {}
    for g in CODE_GLOBS:
        for f in sorted(root.glob(g)):
            if f.is_file():
                out[str(f.relative_to(root))] = _sha_file(f)
    return dict(sorted(out.items()))


def code_list_text(repo_root: Path | None = None) -> str:
    """The sha256 list as it is written between the markers of the deviations file (``<sha256>  <path>``)."""
    return "\n".join(f"{h}  {p}" for p, h in code_files(repo_root).items())


def code_snapshot_sha256(repo_root: Path | None = None) -> str:
    return hashlib.sha256(code_list_text(repo_root).encode()).hexdigest()


def _listed_code(files: list[Path]) -> dict:
    listed = {}
    for f in files:                                   # sorted by name: a later file overrides an earlier one
        txt = f.read_text(encoding="utf-8")
        a, b = txt.find(MARK_BEGIN), txt.find(MARK_END)
        if a < 0 or b < a:
            continue
        for line in txt[a + len(MARK_BEGIN):b].splitlines():
            parts = line.strip().strip("`").split()
            if len(parts) == 2 and len(parts[0]) == 64:
                listed[parts[1]] = parts[0]
    return listed


def stamp_status(*, repo_root: Path | None = None, registry_path: Path | None = None) -> dict:
    root = Path(repo_root) if repo_root else paths.REPO_ROOT
    reg = Path(registry_path) if registry_path else paths.REGISTRY_PATH
    out = {"stamped": False, "reason": "", "row_id": None, "files": {}, "code_sha256": None}
    d = root / DEV_DIR
    files = sorted(d.glob(DEV_GLOB))
    if not (d / DEV_MAIN).is_file():
        out["reason"] = f"{DEV_DIR}/{DEV_MAIN} does not exist"
        return out
    state = {}
    for f in files:
        rel = str(f.relative_to(root))
        state[rel] = _sha_file(f)
        if os.stat(f).st_mode & 0o222:
            out["reason"] = f"{rel} is writable (must be read-only)"
            return out
    out["files"] = state
    on_disk = code_files(root)
    listed = _listed_code(files)
    if not listed:
        out["reason"] = "no code sha256 list between the CODE-SHA256 markers"
        return out
    unlisted = sorted(set(on_disk) - set(listed))
    changed = sorted(p for p, h in listed.items() if on_disk.get(p) != h)
    if unlisted or changed:
        out["reason"] = f"v2 code differs from the stamped list (unlisted: {unlisted}; changed or missing: {changed})"
        return out
    code_sha = code_snapshot_sha256(root)
    out["code_sha256"] = code_sha
    if not reg.is_file():
        out["reason"] = "registry not found"
        return out
    n = 0
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
            n += 1
            ex = r.get("extra") or {}
            sha = ex.get("sha256") or {}
            if ex.get("written_by") == "lead" and all(sha.get(p) == h for p, h in state.items()) \
                    and ex.get("code_sha256") == code_sha:
                out.update(stamped=True, reason="stamped", row_id=r.get("id"))
    if not out["stamped"]:
        out["reason"] = (f"no lead-written {STAMP_EXPERIMENT} row carries the current sha256 of {sorted(state)} "
                         f"and code_sha256 {code_sha[:12]} ({n} rows of that experiment)")
    return out


def require_stamp(what: str = "", *, repo_root: Path | None = None, registry_path: Path | None = None) -> dict:
    s = stamp_status(repo_root=repo_root, registry_path=registry_path)
    if not s["stamped"]:
        raise StampError(f"architecture round v2 is not stamped ({s['reason']}); refusing: {what}")
    return s


# ----------------------------------------------------------------------------------------------- declaration

def declare(*, log=print) -> dict:
    """Write the declaration record once (parity log + registry). Idempotent for an identical declaration."""
    from sentinel.core.registry import log_result

    stamp = require_stamp("declaration record")
    dec = D.declaration()
    sha = D.declaration_sha256()
    if DECLARATION.exists():
        old = json.loads(DECLARATION.read_text())
        if old["sha256"] != sha:
            raise ParityError(f"a different declaration is already on record ({old['sha256'][:12]} != {sha[:12]}); "
                              "a changed list needs a dated, stamped deviation and a new run root")
        log(f"[declare] already declared: {sha[:12]}")
        return old
    rec = {"utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"), "sha256": sha,
           "stamp_row": stamp["row_id"], "code_sha256": stamp["code_sha256"], "declaration": dec}
    DECLARATION.parent.mkdir(parents=True, exist_ok=True)
    DECLARATION.write_text(json.dumps(rec, indent=1, sort_keys=True, default=str))
    os.chmod(DECLARATION, 0o444)
    with open(LOG, "a") as fh:
        fh.write(json.dumps({"utc": rec["utc"], "run": "DECLARATION", "declaration_sha256": sha,
                             "arm_sha256": dec["arm_sha256"], "declaration": dec}, default=str) + "\n")
    r = log_result(workstream=WORKSTREAM, experiment=DECL_EXPERIMENT, model="declaration", metric="n_arms",
                   value=float(len(dec["arms"])), split="prereg",
                   notes="configuration lists of every arm, declared before the first run of any arm",
                   extra={"sha256": sha, "arm_sha256": dec["arm_sha256"], "stamp_row": stamp["row_id"],
                          "file": str(DECLARATION)})
    log(f"[declare] declaration {sha[:12]} logged (registry {r['id']})")
    return rec


def require_declared(arm: str, cname: str, *, registry_path: Path | None = None) -> str:
    """The declared sha256, or ParityError: no declaration, a changed list, or an undeclared configuration."""
    if cname not in D.CONFIGS.get(arm, {}):
        raise ParityError(f"configuration {cname!r} is not declared for arm {arm!r}")
    if not DECLARATION.exists():
        raise ParityError("no declaration record (run sonde_v2_cv.py --phase declare first)")
    rec = json.loads(DECLARATION.read_text())
    if rec["sha256"] != D.declaration_sha256() or rec["declaration"]["arm_sha256"].get(arm) != D._sha(D.arm_list(arm)):
        raise ParityError(f"the configuration list of {arm} in the code is not the declared one "
                          f"({rec['sha256'][:12]}); refusing to run")
    reg = Path(registry_path) if registry_path else paths.REGISTRY_PATH
    ok = False
    if reg.is_file():
        with reg.open(encoding="utf-8") as fh:
            ok = any(DECL_EXPERIMENT in ln and rec["sha256"] in ln for ln in fh)
    if not ok:
        raise ParityError("the declaration sha256 is not in the registry")
    return rec["sha256"]


# ----------------------------------------------------------------------------------------------- parity log

def sites_sha256(data, idx) -> str:
    return hashlib.sha256("\n".join(sorted(data.sites[i] for i in np.unique(idx))).encode()).hexdigest()


def record(run: str, *, data, use: str, arm: str, config: str, fold: int, fit_sites, es_sites, held_sites,
           n_rows: int, feature_columns: list[str], input_sha256: dict, fit_window: dict | None = None,
           extra: dict | None = None, log_path: Path | None = None) -> dict:
    """Append one parity record. ``use`` = "step1" | "cv"; ``input_sha256`` = hashes of the input arrays the run
    read (``pp`` = perfect-prog sample, ``gefs_bc`` = archived bias-corrected product)."""
    from sentinel.models.sonde import common
    reg = common.site_registry()
    sites = [data.sites[i] for i in np.unique(fit_sites)]
    types = reg.reindex(sites)["site_type"].fillna("?").value_counts().to_dict() if sites else {}
    rec = {"utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"), "run": run, "use": use,
           "arm": arm, "config": config, "fold": int(fold),
           "declaration_sha256": json.loads(DECLARATION.read_text())["sha256"] if DECLARATION.exists() else None,
           "n_train_rows": int(n_rows), "n_train_sites": len(sites),
           "train_sites_sha256": sites_sha256(data, fit_sites) if len(sites) else None,
           "es_sites_sha256": sites_sha256(data, es_sites) if len(np.atleast_1d(es_sites)) else None,
           "held_sites_sha256": sites_sha256(data, held_sites), "n_held_sites": int(np.unique(held_sites).size),
           "n_loaded_sites": len(data.sites),
           "site_types": {str(k): int(v) for k, v in types.items()},
           "n_feature_columns": len(feature_columns),
           "feature_columns_sha256": hashlib.sha256("\n".join(feature_columns).encode()).hexdigest(),
           "static_columns": [str(c) for c in data.static.columns],
           "input_sha256": dict(input_sha256), "fit_window": fit_window or {}, **(extra or {})}
    p = Path(log_path) if log_path else LOG
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")
    return rec


def read_log(log_path: Path | None = None) -> list[dict]:
    p = Path(log_path) if log_path else LOG
    if not p.exists():
        return []
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]


def check(use: str, fold: int, *, require_arms=(), log_path: Path | None = None) -> dict:
    """Cross-arm parity of one fold: every record of (use, fold) must agree on the held-out site set, the static
    columns, the loaded site count and every input hash it shares with another record; every FITTED arm must also
    agree on the fit-site and early-stopping-site sets. ``require_arms``: (arm, config) pairs that must be present.
    Raises :class:`ParityError`; returns a summary."""
    recs = [r for r in read_log(log_path) if r.get("use") == use and r.get("fold") == fold]
    have = {(r["arm"], r["config"]) for r in recs}
    missing = [a for a in require_arms if tuple(a) not in have]
    if missing:
        raise ParityError(f"{use} fold {fold}: no parity record for {missing}")
    if not recs:
        raise ParityError(f"{use} fold {fold}: no parity record")
    bad = []

    def same(key, rows, get):
        vals = {json.dumps(get(r), sort_keys=True) for r in rows}
        if len(vals) > 1:
            bad.append(f"{key}: {len(vals)} different values across {sorted({r['run'] for r in rows})}")

    same("held_sites_sha256", recs, lambda r: r["held_sites_sha256"])
    same("static_columns", recs, lambda r: r["static_columns"])
    same("n_loaded_sites", recs, lambda r: r["n_loaded_sites"])
    same("declaration_sha256", recs, lambda r: r["declaration_sha256"])
    fitted = [r for r in recs if r.get("train_sites_sha256")]
    same("train_sites_sha256", fitted, lambda r: r["train_sites_sha256"])
    same("es_sites_sha256", fitted, lambda r: r["es_sites_sha256"])
    for key in sorted({k for r in recs for k in r.get("input_sha256", {})}):
        same(f"input_sha256[{key}]", [r for r in recs if key in r["input_sha256"]], lambda r: r["input_sha256"][key])
    if bad:
        raise ParityError(f"{use} fold {fold}: " + "; ".join(bad))
    return {"use": use, "fold": fold, "n_records": len(recs), "arms": sorted(f"{a}_{c}" for a, c in have),
            "held_sites_sha256": recs[0]["held_sites_sha256"],
            "train_sites_sha256": fitted[0]["train_sites_sha256"] if fitted else None,
            "input_sha256": {k: v for r in recs for k, v in r["input_sha256"].items()}}
