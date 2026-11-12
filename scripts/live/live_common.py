# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Shared pieces of the EW-v2 live prospective trial (Test-C; amendment
``docs/preregistration/ew_v2_live_amendment.md``).

* Frozen trial configuration ``configs/live/ew_v2_live.json`` (written once by ``build_live_config.py``).
* Lead-written freeze record ``docs/preregistration/FREEZE_ew_live.json``. Without it only ``--mode dryrun``
  (train/val-role sites, separate output tree, separate hash log) and ``--mode replay`` (val-block identity
  check) run.
* Hash-chained issue log ``results/live/ew_v2_live_hashlog.jsonl``: every on-time issue appends one row
  ``{seq, issue_date, issued_utc, files{name: sha256}, code_sha256, config_sha256, prev_row_sha256,
  row_sha256}``; ``row_sha256`` = SHA-256 of the canonical JSON of the row without ``row_sha256``. Editing or
  inserting any past row changes every later hash. The lead commits and pushes the log (public timestamp).
* Code hash = SHA-256 over the import closure (``sentinel.*``) of the live entry points, plus the scripts in
  this directory and ``scripts/train/sonde_events.py`` (imported by path).

Nothing here touches files hashed by ``FREEZE_ew.json``; frozen code is imported, never modified.
"""

from __future__ import annotations

import ast
import fcntl
import hashlib
import importlib.util
import json
import os
import socket
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sentinel.core import paths  # noqa: E402

CONFIG = REPO / "configs" / "live" / "ew_v2_live.json"
FREEZE_LIVE = REPO / "docs" / "preregistration" / "FREEZE_ew_live.json"
HASHLOG = REPO / "results" / "live" / "ew_v2_live_hashlog.jsonl"
LIVE_ROOT = paths.RUNS_ROOT / "live" / "ew_v2"
DRYRUN_ROOT = LIVE_ROOT / "dryrun"
# One continuous live block (WY2027 + WY2028). Issues 2026-12-01 .. 2028-09-30 (review C-L2: the whole span of the
# pre-specified extension is frozen up front). Must equal cfg["trial"]["live_block"] (checked at every issue).
LIVE_BLOCK = ("2026-10-01", "2028-09-30")
SCHEMA_VERSION = "ew_v2_live_issue/2"
# ISSUING closure only (review W-L13): the scorer and one-off helpers are deliberately OUTSIDE it, so later work on
# them can never change the frozen issuing hash.
ENTRY_POINTS = ("scripts/live/live_common.py", "scripts/live/live_data.py", "scripts/live/issue.py",
                "scripts/live/fetch_usgs.py", "scripts/live/fetch_gridmet.py", "scripts/live/log_status.py",
                "scripts/live/stamp.py", "scripts/live/push_evidence.py", "scripts/live/verify_hashlog.py",
                "scripts/train/sonde_events.py")
EXTRA_FILES = ("scripts/live/run_daily.sh", "configs/sonde/events.yaml", "configs/sonde/common.yaml")
TIMESTAMP_DIR = REPO / "results" / "live" / "timestamps"


# --------------------------------------------------------------------------- hashing
def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()


def _module_file(mod: str) -> Path | None:
    base = REPO.joinpath(*mod.split("."))
    if (base / "__init__.py").is_file():
        return base / "__init__.py"
    if base.with_suffix(".py").is_file():
        return base.with_suffix(".py")
    return None


def _imports(f: Path) -> set[str]:
    tree = ast.parse(f.read_text(), filename=str(f))
    pkg = list(f.relative_to(REPO).parent.parts)
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names if a.name.split(".")[0] == "sentinel"}
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = pkg[: len(pkg) - (node.level - 1)] if node.level > 1 else pkg
                mod = ".".join(base + ([node.module] if node.module else []))
            else:
                mod = node.module or ""
            if mod.split(".")[0] != "sentinel":
                continue
            out.add(mod)
            out |= {f"{mod}.{a.name}" for a in node.names}
    return out


def code_files() -> list[str]:
    """Import closure of the live entry points (+ package __init__ files) and the extra files."""
    todo = [REPO / e for e in ENTRY_POINTS if (REPO / e).is_file()]
    seen: set[Path] = set()
    while todo:
        f = todo.pop()
        if f in seen:
            continue
        seen.add(f)
        parts = f.relative_to(REPO).parts
        for i in range(1, len(parts)):
            init = REPO.joinpath(*parts[:i], "__init__.py")
            if init.is_file() and init not in seen:
                todo.append(init)
        for mod in _imports(f):
            mf = _module_file(mod)
            if mf is not None and mf not in seen:
                todo.append(mf)
    files = {str(p.relative_to(REPO)) for p in seen}
    files |= {e for e in EXTRA_FILES if (REPO / e).is_file()}
    return sorted(files)


def code_sha256() -> str:
    h = hashlib.sha256()
    for rel in code_files():
        h.update(rel.encode())
        h.update((REPO / rel).read_bytes())
    return h.hexdigest()


# --------------------------------------------------------------------------- config / freeze
def load_config(path: Path = CONFIG) -> tuple[dict, str]:
    if not path.is_file():
        raise SystemExit(f"{path} missing — run scripts/live/build_live_config.py first")
    return json.loads(path.read_text()), sha256_file(path)


def load_freeze_live(cfg_sha: str, *, check_code: bool = True) -> dict:
    """The lead-written live freeze record; refuses on any mismatch (config, code, calibration files)."""
    if not FREEZE_LIVE.is_file():
        raise PermissionError(f"{FREEZE_LIVE} does not exist: live issuance (all roles) is locked; "
                              "use --mode dryrun (train/val-role sites) until the lead freezes the amendment")
    fr = json.loads(FREEZE_LIVE.read_text())
    if any(m in json.dumps(fr.get("status", "")) for m in ("NOT_A_FREEZE", "candidate")):
        raise PermissionError("FREEZE_ew_live.json is marked as a candidate / not a freeze")
    if "<" in json.dumps(fr):
        raise PermissionError("FREEZE_ew_live.json still contains <placeholders>")
    if fr.get("env_lock_sha256") != env_lock_sha256():
        raise PermissionError("Python environment differs from the frozen environment lock")
    if fr.get("config_sha256") != cfg_sha:
        raise PermissionError("live config differs from the frozen one")
    smap = REPO / "configs" / "live" / "ew_v2_live_series_map.json"
    if not smap.is_file() or fr.get("series_map_sha256") != sha256_file(smap):
        raise PermissionError("frozen series map missing or differs from FREEZE_ew_live.json")
    if check_code and fr.get("code_sha256") != code_sha256():
        raise PermissionError("live code differs from the frozen code (import-closure hash)")
    return fr


def verify_calibrations(cfg: dict) -> dict:
    """Every calibration file and every artifact it lists must match the frozen hashes."""
    out = {}
    for hz, m in cfg["models"].items():
        cf = Path(m["calibration"])
        sha = sha256_file(cf)
        if sha != m["calibration_sha256"]:
            raise PermissionError(f"{hz}: calibration file {cf} changed ({sha[:12]} != {m['calibration_sha256'][:12]})")
        cal = json.loads(cf.read_text())
        for key, ent in cal["entries"].items():
            if ent["hazard"] != hz:
                continue
            for name, art in ent["artifacts"].items():
                f = cf.parent / art["file"]
                if sha256_file(f) != art["sha256"]:
                    raise PermissionError(f"artifact {f} changed since calibration")
        out[hz] = sha
    return out


def verify_static_pins(cfg: dict) -> dict:
    """Pinned static inputs read at issue time (review W-L1): refuse on a missing, empty or changed file."""
    out = {}
    for key, rel in (("chloride_regressions_sha256", paths.PROCESSED / "chloride_derived" / "regressions.parquet"),
                     ("sites_parquet_sha256", paths.PROCESSED / "sites" / "sites.parquet"),
                     ("gridmet_site_cells_sha256", paths.RAW / "gridmet" / "site_cells.parquet"),
                     ("split_file_sha256", REPO / "docs" / "preregistration" / "sonde_split.json")):
        p = Path(rel)
        if not p.is_file() or p.stat().st_size == 0:
            raise PermissionError(f"pinned input {p} missing or empty")
        if sha256_file(p) != cfg["data_pins"][key]:
            raise PermissionError(f"pinned input {p} changed")
        out[key] = cfg["data_pins"][key]
    if tuple(cfg["trial"]["live_block"]) != LIVE_BLOCK:
        raise PermissionError("config live block differs from the code's LIVE_BLOCK")
    return out


def env_lock_sha256() -> str:
    """SHA-256 of the running interpreter's version and every installed distribution 'name==version' (review W-L2)."""
    import importlib.metadata as md
    import platform

    pk = sorted({f"{(d.metadata['Name'] or '').lower()}=={d.version}" for d in md.distributions()})
    return hashlib.sha256(("python==" + platform.python_version() + "\n" + "\n".join(pk)).encode()).hexdigest()


def freeze_sha() -> str | None:
    return sha256_file(FREEZE_LIVE) if FREEZE_LIVE.is_file() else None


# --------------------------------------------------------------------------- hash log
def _row_hash(row: dict) -> str:
    return hashlib.sha256(canonical({k: v for k, v in row.items() if k != "row_sha256"})).hexdigest()


def read_hashlog(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def append_hashlog(path: Path, rec: dict) -> dict:
    """Append one chained row under an exclusive lock. Every scheduled day gets a row (review C-L1): ``status`` is
    ``issued`` or ``failed`` (with stage + exception class). At most ONE ``issued`` row per issue date; failed rows
    may repeat (retries)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        fh.seek(0)
        rows = [json.loads(ln) for ln in fh.read().splitlines() if ln.strip()]
        if rec.get("status") == "issued" and any(r["issue_date"] == rec["issue_date"] and r.get("status") == "issued"
                                                 for r in rows):
            raise FileExistsError(f"hash log already holds an issued row for {rec['issue_date']}")
        row = {"seq": len(rows), **rec, "host": socket.gethostname(), "pid": os.getpid(),
               "prev_row_sha256": rows[-1]["row_sha256"] if rows else "0" * 64}
        row["row_sha256"] = _row_hash(row)
        fh.seek(0, os.SEEK_END)
        fh.write(json.dumps(row, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
        fcntl.flock(fh, fcntl.LOCK_UN)
    return row


def verify_chain(rows: list[dict]) -> list[str]:
    errs, prev = [], "0" * 64
    for i, r in enumerate(rows):
        if r.get("seq") != i:
            errs.append(f"row {i}: seq {r.get('seq')}")
        if r.get("prev_row_sha256") != prev:
            errs.append(f"row {i}: broken chain")
        if _row_hash(r) != r.get("row_sha256"):
            errs.append(f"row {i}: row hash mismatch")
        prev = r.get("row_sha256")
    return errs


# --------------------------------------------------------------------------- frozen script import
def sonde_events_module():
    """``scripts/train/sonde_events.py`` (hashed by FREEZE_ew) imported by path, unmodified."""
    name = "_frozen_sonde_events"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / "train" / "sonde_events.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def excl_write(p: Path, rec: dict) -> None:
    """Atomic create-only write (O_CREAT | O_EXCL), then read-only."""
    p.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(fd, "w") as fh:
        fh.write(json.dumps(rec, indent=1, sort_keys=True, default=str))
