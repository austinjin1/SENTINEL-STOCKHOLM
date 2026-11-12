# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Single-look enforcement for the M5-X test pass (protocol ``bioassess_x_protocol.md`` §9 as amended by v11).

The look is four separate commands of ``scripts/eval/bioassess_x_final.py``, all run from the pinned code snapshot:

1. **claim** — :func:`load_freeze` accepts ONLY the lead-written ``docs/preregistration/bioassess_x_freeze.json``
   (status ``frozen``; equal to the candidate record on every key but the freeze keys; ``candidate_sha256`` and
   ``scoring_code_sha256`` match; :func:`verify_snapshot`), then :func:`verify_data_pins` and
   :func:`verify_artifacts` — all before the claim, so a mismatch never uses the look. :func:`claim` refuses when
   the registry already holds any ``bioassess_x_test`` row, creates ``TEST_LOOK.json`` (O_EXCL, 0444) and writes
   the claim-time registry row. The claim process writes the label-free test inputs (:func:`mark_inputs`).
2. **chunk** (any number, any order, in parallel) — :func:`require_open_claim`; units are write-once
   (:func:`unit_write`: one directory per unit with its own sha256, moved into place; an existing unit is never
   replaced). A failed or killed chunk leaves finished units in place and does not end the look.
3. **seal** — :func:`seal_predictions` (0444, ``SHA256SUMS``, one ``predictions_sealed`` registry row holding the
   sha256 of every file, then ``PREDICTIONS_SAVED.json``). From here on no unit can be written or used.
4. **score** — :func:`begin_scoring` (the seal must verify; O_EXCL ``SCORING_STARTED.json``); test labels load only
   through :func:`require_scoring`; every scoring row reaches the registry in one locked append
   (:func:`flush_rows`); then the O_EXCL ``SCORED.json``. Scoring is never resumed: if it stops before the append,
   ``--rescore-from-saved`` starts again from the sealed predictions (``RESCORE_<n>.json``).

Code snapshot (v11 item 3): the pinned files are copied once to ``$SENTINEL_RUNS/bioassess_x/code_<sha>/`` (files
0444, directories 0555) with the marker ``M5X_CODE_SNAPSHOT.json``. When this module runs from a snapshot, the freeze
file, the candidate record and the registry are those of the live repository named in the marker.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import socket
import time
from pathlib import Path

from sentinel.core import paths

EXPERIMENT = "bioassess_x_test"
SNAPSHOT_MARKER = "M5X_CODE_SNAPSHOT.json"
SNAPSHOT_ROOT = paths.RUNS_ROOT / "bioassess_x"


def _live_root() -> Path:
    m = paths.REPO_ROOT / SNAPSHOT_MARKER
    return Path(json.loads(m.read_text())["live_root"]) if m.is_file() else paths.REPO_ROOT


LIVE_ROOT = _live_root()                 # the repository that holds the freeze file, the candidate and the registry
FREEZE_FILE = LIVE_ROOT / "docs/preregistration/bioassess_x_freeze.json"
CANDIDATE_FILE = LIVE_ROOT / "docs/preregistration/bioassess_x_freeze_candidate.json"
REGISTRY = LIVE_ROOT / "results" / "registry.jsonl"
# Scoring code = the static import closure (every ``sentinel`` module reachable by ``import`` statements, including
# imports inside functions) of these entry points, plus the R scripts the scorer runs. Covers e.g. events.py ->
# hazards.py / ew_scoring.py -> early_warning.py, and the frozen M5 modules (M7-review lesson 1).
SCORING_ENTRY = ("scripts/eval/bioassess_x_final.py", "scripts/eval/bioassess_x_dev.py",
                 "sentinel/events/bioassess_x.py", "sentinel/eval/events.py", "sentinel/models/bioassess/xtestlook.py",
                 "sentinel/core/gpu.py")       # gpu.py: run as ``python -m`` by the GPU chunks, not imported
SCORING_EXTRA = ("scripts/eval/bioassess_x_bioinfer.R", "scripts/eval/bioassess_x_gjam.R")
FREEZE_KEYS = ("status", "candidate_sha256", "scoring_code_sha256", "data_snapshots", "candidates", "comparators",
               "seeds", "n_boot", "n_boot_confirmatory", "alpha", "code_snapshot")
# keys the lead's freeze record may add to (or, for ``status``, change in) the candidate record; every other key
# must be equal in both files
FREEZE_ONLY_KEYS = ("status", "frozen_at", "candidate_sha256", "frozen_by", "review", "note")
# data pins: role -> (manifest, processed loader dir, exact files the M5-X loaders read)
DATA_PINS = {
    "nars_streams": ("nars_streams", "nars_streams", ("visits.parquet", "benthic_counts.parquet",
                                                     "chemistry_wide.parquet", "chemistry_long.parquet")),
    "nars_lakes": ("nars_lakes", "nars_lakes", ("visits.parquet", "benthic_counts.parquet", "chemistry_wide.parquet",
                                               "chemistry_long.parquet")),
    "uk_ea_ecology": ("uk_ea_ecology", "uk_ea_ecology", ("sites.parquet", "samples_invertebrate.parquet",
                                                        "counts_invertebrate.parquet",
                                                        "chemistry_matched_wide_pm30d_invertebrate.parquet")),
    "taxonomy_harmonised": ("taxonomy_harmonised", "taxonomy_harmonised", ("taxon_map.parquet",)),
    "fuse_tier_c": ("fuse_tier_c", "fuse_tier_c", ("rows.parquet",)),
    "s2_wqp_matchups_hist": ("s2_wqp_matchups", "s2_wqp_matchups_hist", ("samples.parquet",)),
}


class SingleLookError(RuntimeError):
    pass


def look_dir(root: Path | None = None) -> Path:
    return Path(root) if root is not None else paths.RUNS_ROOT / "bioassess" / "x_track" / "test_look"


def _reg(registry_path=None) -> Path:
    return Path(registry_path) if registry_path else REGISTRY


def sha_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def excl_write(p: Path, rec: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    with os.fdopen(fd, "w") as fh:
        json.dump(rec, fh, indent=1, default=str)


def _module_file(mod: str) -> Path | None:
    base = paths.REPO_ROOT / Path(*mod.split("."))
    if base.with_suffix(".py").is_file():
        return base.with_suffix(".py")
    if (base / "__init__.py").is_file():
        return base / "__init__.py"
    return None


def _imports_of(f: Path) -> set[str]:
    import ast

    tree = ast.parse(f.read_text(), filename=str(f))
    rel = f.relative_to(paths.REPO_ROOT).with_suffix("")
    pkg = list(rel.parts[:-1]) if rel.name != "__init__" else list(rel.parts[:-1])
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = pkg[: len(pkg) - (node.level - 1)] if node.level > 1 else pkg
                root = ".".join(base + ([node.module] if node.module else []))
            else:
                root = node.module or ""
            out.add(root)
            out |= {f"{root}.{a.name}" for a in node.names}
    return {m for m in out if m == "sentinel" or m.startswith("sentinel.")}


def code_closure(entries=SCORING_ENTRY) -> list[str]:
    """Repo-relative files of the static ``sentinel`` import closure of ``entries`` (incl. parent packages)."""
    seen, todo = set(), [paths.REPO_ROOT / e for e in entries]
    while todo:
        f = todo.pop()
        if f in seen or not f.is_file():
            continue
        seen.add(f)
        for mod in _imports_of(f):
            parts = mod.split(".")
            for k in range(1, len(parts) + 1):            # the module and every parent package
                mf = _module_file(".".join(parts[:k]))
                if mf is not None and mf not in seen:
                    todo.append(mf)
    return sorted(str(f.relative_to(paths.REPO_ROOT)) for f in seen)


def scoring_code_files() -> list[str]:
    return code_closure() + list(SCORING_EXTRA)


def scoring_code_sha256() -> str:
    h = hashlib.sha256()
    for rel in scoring_code_files():
        f = paths.REPO_ROOT / rel
        h.update(f"{rel}\t{sha_file(f) if f.is_file() else 'MISSING'}\n".encode())
    return h.hexdigest()


def verify_artifacts(rec: dict) -> None:
    """Every artifact the record lists (path -> sha256) must match — checked by every command."""
    bad = [p for p, h in (rec.get("artifacts") or {}).items() if not Path(p).is_file() or sha_file(p) != h]
    if not rec.get("artifacts"):
        bad.append("record lists no artifacts")
    if bad:
        raise SingleLookError(f"artifacts changed or missing: {bad[:10]}")


# --------------------------------------------------------------------------- code snapshot (v11 item 3)
def _tree_sha(files: dict) -> str:
    return hashlib.sha256("".join(f"{r}\t{files[r]}\n" for r in sorted(files)).encode()).hexdigest()


def _snapshot_files(d: Path) -> set[str]:
    return {f.relative_to(d).as_posix() for f in d.rglob("*") if f.is_file()} - {SNAPSHOT_MARKER}


def make_snapshot(rel_files, *, dest_root: Path | None = None, src_root: Path | None = None) -> dict:
    """Copy the repo-relative ``rel_files`` of ``src_root`` (default: the running tree) to a read-only snapshot
    ``<dest_root>/code_<sha16>/`` (files 0444, directories 0555) and return its record block
    ``{dir, tree_sha256, files: {rel: sha256}, marker_sha256, live_root}``. The directory name is the hash of the
    contents, so an identical snapshot is reused and a different one never overwrites it."""
    src = Path(src_root) if src_root is not None else paths.REPO_ROOT
    files = {str(r): sha_file(src / r) for r in sorted(set(map(str, rel_files)))}
    tree = _tree_sha(files)
    root = Path(dest_root) if dest_root is not None else SNAPSHOT_ROOT
    d = root / f"code_{tree[:16]}"
    live = str(LIVE_ROOT if src_root is None else src)
    if not d.exists():
        tmp = root / f".code_{tree[:16]}.tmp{os.getpid()}"
        if tmp.exists():
            shutil.rmtree(tmp)
        for rel in files:
            (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src / rel, tmp / rel)
        (tmp / SNAPSHOT_MARKER).write_text(json.dumps(
            {"live_root": live, "tree_sha256": tree, "n_files": len(files),
             "created": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=1))
        for f in tmp.rglob("*"):
            os.chmod(f, 0o555 if f.is_dir() else 0o444)
        os.rename(tmp, d)
        os.chmod(d, 0o555)
    block = {"dir": str(d), "tree_sha256": tree, "files": files, "marker_sha256": sha_file(d / SNAPSHOT_MARKER),
             "live_root": json.loads((d / SNAPSHOT_MARKER).read_text())["live_root"]}
    verify_snapshot({"code_snapshot": block}, require_running=False)
    return block


def verify_snapshot(rec: dict, *, require_running: bool = True) -> Path:
    """The record's snapshot must hold exactly the listed files with the listed hashes (and its marker); with
    ``require_running`` the running code tree must BE that snapshot. The live tree is never looked at."""
    cs = rec.get("code_snapshot") or {}
    if not cs.get("dir") or not cs.get("files"):
        raise SingleLookError("record has no code_snapshot block")
    d = Path(cs["dir"])
    if not d.is_dir():
        raise SingleLookError(f"code snapshot {d} does not exist")
    if require_running and d.resolve() != paths.REPO_ROOT.resolve():
        raise SingleLookError(f"not running from the pinned code snapshot {d} (running tree: {paths.REPO_ROOT})")
    present, listed = _snapshot_files(d), set(cs["files"])
    if present != listed:
        raise SingleLookError(f"code snapshot file set changed: unlisted {sorted(present - listed)[:5]}, "
                              f"missing {sorted(listed - present)[:5]}")
    bad = [r for r, h in cs["files"].items() if sha_file(d / r) != h]
    if bad:
        raise SingleLookError(f"code snapshot files changed: {bad[:5]}")
    if _tree_sha(cs["files"]) != cs.get("tree_sha256"):
        raise SingleLookError("code snapshot tree hash does not match its file list")
    mk = d / SNAPSHOT_MARKER
    if not mk.is_file() or sha_file(mk) != cs.get("marker_sha256"):
        raise SingleLookError("code snapshot marker changed")
    return d


# --------------------------------------------------------------------------- data pins
def content_pin(loader_dir: str, files) -> str:
    root = paths.PROCESSED / loader_dir
    h = hashlib.sha256()
    for f in sorted(files):
        h.update(f"{f}\t{sha_file(root / f)}\n".encode())
    return h.hexdigest()


def make_data_pins() -> dict:
    out = {}
    for role, (manifest, ldir, files) in DATA_PINS.items():
        mf = paths.MANIFESTS / f"{manifest}.json"
        out[role] = {"manifest": manifest, "manifest_sha256": sha_file(mf), "loader_dir_rel": ldir,
                     "files": list(files), "content_sha256": content_pin(ldir, files)}
    return out


def verify_data_pins(rec: dict) -> dict:
    pins = rec.get("data_snapshots") or {}
    bad = []
    for role in DATA_PINS:
        pin = pins.get(role)
        if not pin:
            bad.append(f"{role}: no pin")
            continue
        mf = paths.MANIFESTS / f"{pin['manifest']}.json"
        if not mf.is_file() or sha_file(mf) != pin["manifest_sha256"]:
            bad.append(f"{role}: manifest {pin['manifest']} changed")
        if content_pin(pin["loader_dir_rel"], pin["files"]) != pin["content_sha256"]:
            bad.append(f"{role}: content changed")
    if bad:
        raise SingleLookError(f"data pins do not match: {bad}")
    return {r: pins[r]["content_sha256"] for r in DATA_PINS}


# --------------------------------------------------------------------------- freeze
def load_freeze(freeze_file: Path | None = None, *, require_running: bool = True) -> dict:
    """The lead-written FREEZE file only (fixed path); never the candidate record. It must equal the candidate
    record on every key except :data:`FREEZE_ONLY_KEYS`, and its code snapshot must verify."""
    ff = Path(freeze_file) if freeze_file is not None else FREEZE_FILE
    if ff.resolve() == CANDIDATE_FILE.resolve():
        raise SingleLookError("the candidate record is never accepted; the lead writes bioassess_x_freeze.json")
    if not ff.is_file():
        raise SingleLookError(f"freeze file {ff} does not exist (the lead has not frozen M5-X)")
    rec = json.loads(ff.read_text())
    missing = [k for k in FREEZE_KEYS if not rec.get(k)]
    if missing:
        raise SingleLookError(f"freeze file lacks {missing}")
    if rec["status"] != "frozen":
        raise SingleLookError("freeze file status is not 'frozen' (a candidate record?)")
    if not CANDIDATE_FILE.is_file() or sha_file(CANDIDATE_FILE) != rec["candidate_sha256"]:
        raise SingleLookError("freeze record does not match the candidate record it was made from")
    cand = json.loads(CANDIDATE_FILE.read_text())
    diff = sorted(k for k in set(cand) | set(rec) if k not in FREEZE_ONLY_KEYS and cand.get(k) != rec.get(k))
    if diff:
        raise SingleLookError(f"freeze record differs from the candidate record in {diff[:10]}")
    if rec["scoring_code_sha256"] != scoring_code_sha256():
        raise SingleLookError("scoring code changed since the freeze")
    verify_snapshot(rec, require_running=require_running)
    rec["_path"], rec["_sha256"] = str(ff), sha_file(ff)
    return rec


# --------------------------------------------------------------------------- claim
def registry_rows(experiment: str = EXPERIMENT, registry_path=None) -> list[dict]:
    reg = _reg(registry_path)
    out = []
    if reg.is_file():
        with reg.open() as fh:
            for line in fh:
                if experiment in line and json.loads(line).get("experiment") == experiment:
                    out.append(json.loads(line))
    return out


NON_SCORING_METRICS = ("look_claimed", "predictions_sealed", "look_aborted")


def scoring_rows(registry_path=None) -> list[dict]:
    return [r for r in registry_rows(EXPERIMENT, registry_path) if r.get("metric") not in NON_SCORING_METRICS]


def sealed_rows(registry_path=None) -> list[dict]:
    return [r for r in registry_rows(EXPERIMENT, registry_path) if r.get("metric") == "predictions_sealed"]


def _log(registry_path, **kw):
    from sentinel.core.registry import log_result

    return log_result(workstream="bioassess", experiment=EXPERIMENT, model="m5x_test_look",
                      registry_path=_reg(registry_path), **kw)


def claim(rec: dict, authorised_by: str, run_dir: Path, *, root: Path | None = None, registry_path=None) -> dict:
    n = len(registry_rows(EXPERIMENT, registry_path))
    if n:
        raise SingleLookError(f"registry already holds {n} {EXPERIMENT} rows — the M5-X test look was used")
    c = {"look": EXPERIMENT, "freeze_file": rec.get("_path"), "freeze_sha256": rec.get("_sha256"),
         "scoring_code_sha256": rec.get("scoring_code_sha256"), "authorised_by": authorised_by,
         "run_dir": str(run_dir), "claimed": time.strftime("%Y-%m-%dT%H:%M:%S"), "pid": os.getpid(),
         "host": socket.gethostname()}
    try:
        excl_write(look_dir(root) / "TEST_LOOK.json", c)
    except FileExistsError as e:
        raise SingleLookError(f"M5-X test look already claimed ({look_dir(root) / 'TEST_LOOK.json'})") from e
    _log(registry_path, metric="look_claimed", value=1.0, split="M5-X claim (single pass)",
         notes="claim-time row (single-look guard)",
         extra={k: c[k] for k in ("freeze_sha256", "authorised_by", "run_dir")})
    return c


def open_claim(root: Path | None = None) -> dict | None:
    m = look_dir(root) / "TEST_LOOK.json"
    if not m.is_file() or (look_dir(root) / "SCORED.json").exists():
        return None
    return json.loads(m.read_text())


def is_sealed(root: Path | None = None, registry_path=None) -> bool:
    return bool((look_dir(root) / "PREDICTIONS_SAVED.json").exists() or sealed_rows(registry_path)
                or scoring_rows(registry_path))


def require_open_claim(root: Path | None = None, registry_path=None, *, run_dir: Path | None = None,
                       freeze_sha256: str | None = None) -> dict:
    """The prediction phase (test INPUTS, units) is open only between the claim and the sealing, for the claimed
    run directory and the claimed freeze record. Any number of processes may work under it (v11 item 2)."""
    c = open_claim(root)
    if c is None:
        raise SingleLookError("M5-X test inputs need an open, unscored test-look claim (scripts/eval/bioassess_x_final.py)")
    if is_sealed(root, registry_path):
        raise SingleLookError("the prediction phase closed at sealing: no test input or unit can be used any more")
    sha = freeze_sha256 if freeze_sha256 is not None else (sha_file(FREEZE_FILE) if FREEZE_FILE.is_file() else None)
    if sha is not None and c.get("freeze_sha256") != sha:
        raise SingleLookError("the open claim belongs to a different freeze record")
    if run_dir is not None and c.get("run_dir") != str(run_dir):
        raise SingleLookError(f"the open claim belongs to run directory {c.get('run_dir')}, not {run_dir}")
    return c


def mark_inputs(info: dict, *, root: Path | None = None) -> None:
    """Written once by the claim process after the label-free test inputs and the unit plan are on disk."""
    try:
        excl_write(look_dir(root) / "INPUTS.json", {"written": time.strftime("%Y-%m-%dT%H:%M:%S"), **info})
    except FileExistsError as e:
        raise SingleLookError("the test inputs of this look were already written") from e


def inputs_info(root: Path | None = None) -> dict:
    m = look_dir(root) / "INPUTS.json"
    if not m.is_file():
        raise SingleLookError("the claim did not finish writing the label-free test inputs (INPUTS.json missing)")
    return json.loads(m.read_text())


# --------------------------------------------------------------------------- write-once units (v11 item 2)
def unit_dir(run_dir: Path, key: str) -> Path:
    return Path(run_dir) / "units" / key


def unit_exists(run_dir: Path, key: str) -> bool:
    return (unit_dir(run_dir, key) / "unit.json").is_file()


def unit_write(run_dir: Path, key: str, df, meta: dict) -> bool:
    """Write unit ``key`` once: ``pred.parquet`` (or none for a recorded failure) + ``unit.json`` with its sha256,
    built in a temporary directory and moved into place. Returns False (and writes nothing) if the unit exists."""
    final = unit_dir(run_dir, key)
    if final.exists():
        return False
    tmp = final.parent / f".tmp_{key}_{socket.gethostname()}_{os.getpid()}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    sha = None
    if df is not None:
        df.to_parquet(tmp / "pred.parquet")
        sha = sha_file(tmp / "pred.parquet")
        os.chmod(tmp / "pred.parquet", 0o444)
    (tmp / "unit.json").write_text(json.dumps({**meta, "key": key, "sha256": sha, "pid": os.getpid(),
                                               "host": socket.gethostname(),
                                               "written": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=1, default=str))
    os.chmod(tmp / "unit.json", 0o444)
    try:
        os.rename(tmp, final)                       # fails if another process moved the unit in first
    except OSError:
        shutil.rmtree(tmp, ignore_errors=True)
        return False
    os.chmod(final, 0o555)
    return True


def unit_read(run_dir: Path, key: str):
    """(frame or None for a recorded failure, sidecar) of a finished unit, sha-verified."""
    import pandas as pd

    d = unit_dir(run_dir, key)
    meta = json.loads((d / "unit.json").read_text())
    if meta.get("key") != key:
        raise SingleLookError(f"unit {key}: sidecar belongs to another unit")
    if meta.get("sha256") is None:
        if (d / "pred.parquet").exists():
            raise SingleLookError(f"unit {key}: a failed unit holds predictions")
        return None, meta
    if not (d / "pred.parquet").is_file() or sha_file(d / "pred.parquet") != meta["sha256"]:
        raise SingleLookError(f"unit {key} was modified or lost")
    return pd.read_parquet(d / "pred.parquet"), meta


def unit_lock(run_dir: Path, key: str) -> bool:
    """Advisory per-unit work lock so parallel chunks do not fit the same unit twice. A lock whose owner process
    is gone (same host) is taken over. Correctness never depends on it: units are write-once."""
    f = Path(run_dir) / "locks" / f"{key}.lock"
    f.parent.mkdir(parents=True, exist_ok=True)
    me = {"pid": os.getpid(), "host": socket.gethostname(), "t": time.time()}
    for _ in range(2):
        try:
            fd = os.open(f, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            with os.fdopen(fd, "w") as fh:
                json.dump(me, fh)
            return True
        except FileExistsError:
            try:
                o = json.loads(f.read_text())
            except (OSError, ValueError):
                return False
            if o.get("host") != me["host"] or os.path.exists(f"/proc/{o.get('pid')}"):
                return False
            try:
                f.unlink()                          # stale: the owner process no longer exists
            except FileNotFoundError:
                pass
    return False


def unit_unlock(run_dir: Path, key: str) -> None:
    try:
        (Path(run_dir) / "locks" / f"{key}.lock").unlink()
    except FileNotFoundError:
        pass


# --------------------------------------------------------------------------- sealed predictions
def _verify_sealed_dir(ref: dict) -> Path:
    d = Path(ref["dir"])
    if not (d / "SHA256SUMS").is_file() or sha_file(d / "SHA256SUMS") != ref["sha256sums"]:
        raise SingleLookError("SHA256SUMS missing or modified")
    present = {f.relative_to(d).as_posix() for f in d.rglob("*") if f.is_file()} - {"SHA256SUMS"}
    if present != set(ref["files"]):
        raise SingleLookError(f"sealed folder differs from the sealing row: unlisted {sorted(present - set(ref['files']))[:5]}, "
                              f"missing {sorted(set(ref['files']) - present)[:5]}")
    for n, h in ref["files"].items():
        if sha_file(d / n) != h:
            raise SingleLookError(f"sealed file {n} modified")
    return d


def seal_predictions(pred_dir: Path, *, root: Path | None = None, registry_path=None) -> dict:
    """Seal once: 0444 files, O_EXCL ``SHA256SUMS``, ONE registry row holding every file's sha256 (the reference
    ``check_sealed`` verifies against; the commit point), then O_EXCL ``PREDICTIONS_SAVED.json``. If a crash fell
    between the row and the marker, a second call verifies the folder against the row and writes the marker."""
    pred_dir = Path(pred_dir)
    marker = look_dir(root) / "PREDICTIONS_SAVED.json"
    rows = sealed_rows(registry_path)
    if marker.exists() or scoring_rows(registry_path) or len(rows) > 1:
        raise SingleLookError("predictions already sealed — they are sealed once")
    if rows:                                        # row written, marker lost: finish the same sealing
        ref = rows[0]["extra"]
        if Path(ref["dir"]) != pred_dir:
            raise SingleLookError("the sealing row names another predictions folder")
        _verify_sealed_dir(ref)
        rec = {"dir": str(pred_dir), "n_files": len(ref["files"]), "sha256sums": ref["sha256sums"],
               "saved": time.strftime("%Y-%m-%dT%H:%M:%S"), "completed_after_crash": True}
        excl_write(marker, rec)
        return rec
    if (pred_dir / "SHA256SUMS").exists():
        raise SingleLookError("predictions folder already carries SHA256SUMS but no sealing row exists")
    files = sorted(f for f in pred_dir.rglob("*") if f.is_file())
    if not files:
        raise SingleLookError(f"no predictions to seal in {pred_dir}")
    sums = {f.relative_to(pred_dir).as_posix(): sha_file(f) for f in files}
    for f in files:
        os.chmod(f, 0o444)
    sp = pred_dir / "SHA256SUMS"
    fd = os.open(sp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    with os.fdopen(fd, "w") as fh:
        fh.write("".join(f"{s}  {n}\n" for n, s in sums.items()))
    rec = {"dir": str(pred_dir), "n_files": len(files), "sha256sums": sha_file(sp),
           "saved": time.strftime("%Y-%m-%dT%H:%M:%S")}
    _log(registry_path, metric="predictions_sealed", value=float(len(files)), split="M5-X sealed predictions",
         notes="sealing row (single-look guard): the sha256 of every sealed file",
         extra={"dir": str(pred_dir), "sha256sums": rec["sha256sums"], "files": sums})
    excl_write(marker, rec)
    return rec


def check_sealed(*, root: Path | None = None, registry_path=None, allow_scored: bool = False) -> Path:
    """Verify the sealed folder against the registry's sealing row; refuse once any scoring row exists, after
    SCORED.json, on any unlisted file or any changed file. ``allow_scored`` (verification of a finished look only)
    skips the two "already scored" refusals; it never opens the label loader."""
    if not allow_scored:
        if (look_dir(root) / "SCORED.json").exists():
            raise SingleLookError("already scored — nothing can be re-scored")
        if scoring_rows(registry_path):
            raise SingleLookError("scoring rows exist in the registry — re-scoring is refused")
    rows = sealed_rows(registry_path)
    if len(rows) != 1:
        raise SingleLookError(f"expected exactly one sealing row in the registry, found {len(rows)}")
    if not (look_dir(root) / "PREDICTIONS_SAVED.json").is_file():
        raise SingleLookError("PREDICTIONS_SAVED.json missing: run the seal command again to complete the sealing")
    return _verify_sealed_dir(rows[0]["extra"])


# --------------------------------------------------------------------------- single-shot scoring (v11 item 2.5)
def _scoring_marks(root: Path | None = None) -> list[Path]:
    d = look_dir(root)
    out = [d / "SCORING_STARTED.json"] if (d / "SCORING_STARTED.json").is_file() else []
    return out + sorted(d.glob("RESCORE_*.json"), key=lambda p: int(p.stem.split("_")[1]))


def begin_scoring(authorised_by: str, run_dir: Path, *, rescore: bool = False, root: Path | None = None,
                  registry_path=None) -> Path:
    """Open the scoring phase for THIS process. The seal must verify. The first call writes the O_EXCL
    ``SCORING_STARTED.json``; a second ``--score`` is refused (no resume). ``rescore`` (scoring stopped before the
    registry append) writes ``RESCORE_<n>.json`` and starts again from the sealed predictions."""
    c = open_claim(root)
    if c is None or c.get("run_dir") != str(run_dir):
        raise SingleLookError("scoring needs the claimed, unscored look of this run directory")
    pred_dir = check_sealed(root=root, registry_path=registry_path)
    me = {"pid": os.getpid(), "host": socket.gethostname(), "authorised_by": authorised_by,
          "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "run_dir": str(run_dir)}
    marks = _scoring_marks(root)
    if not rescore:
        try:
            excl_write(look_dir(root) / "SCORING_STARTED.json", me)
        except FileExistsError as e:
            raise SingleLookError("scoring was already started: it is never resumed (use --rescore-from-saved, "
                                  "which starts again from the sealed predictions)") from e
    else:
        if not marks:
            raise SingleLookError("--rescore-from-saved needs a scoring run that was started and did not finish")
        excl_write(look_dir(root) / f"RESCORE_{len(marks)}.json", me)
    return pred_dir


def require_scoring(root: Path | None = None, registry_path=None) -> Path:
    """Test LABELS may be loaded only by the process that opened the scoring phase, with the seal verified and no
    scoring row in the registry."""
    marks = _scoring_marks(root)
    if not marks:
        raise SingleLookError("M5-X test labels load only after sealing, inside the scoring command")
    pred_dir = check_sealed(root=root, registry_path=registry_path)
    o = json.loads(marks[-1].read_text())
    if (o.get("pid"), o.get("host")) != (os.getpid(), socket.gethostname()):
        raise SingleLookError("the scoring phase is held by another process")
    return pred_dir


def flush_rows(pending: Path, *, registry_path=None) -> int:
    """Append every pending scoring row to the registry in ONE locked write. Refused when scoring rows exist."""
    lines = [ln for ln in Path(pending).read_text().splitlines() if ln.strip()]
    bad = [ln for ln in lines if json.loads(ln).get("experiment") != EXPERIMENT
           or json.loads(ln).get("metric") in NON_SCORING_METRICS]
    if not lines or bad:
        raise SingleLookError("pending registry rows are empty or not scoring rows of this look")
    reg = _reg(registry_path)
    reg.parent.mkdir(parents=True, exist_ok=True)
    with open(reg, "a", encoding="utf-8") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            if scoring_rows(registry_path):
                raise SingleLookError("scoring rows already exist in the registry — nothing is written twice")
            fh.write("".join(ln + "\n" for ln in lines))
            fh.flush()
            os.fsync(fh.fileno())
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)
    return len(lines)


def record_abort(exc: BaseException, *, root: Path | None = None, registry_path=None) -> None:
    """A scoring run that stops is recorded: O_EXCL ABORTED marker + a non-scoring registry row. (A failed
    prediction chunk is not an abort: its finished units stay and another chunk continues.)"""
    rec = {"aborted": time.strftime("%Y-%m-%dT%H:%M:%S"), "error": f"{type(exc).__name__}: {exc}"[:2000],
           "pid": os.getpid(), "host": socket.gethostname()}
    try:
        excl_write(look_dir(root) / f"ABORTED_{int(time.time())}_{os.getpid()}.json", rec)
    finally:
        _log(registry_path, metric="look_aborted", value=1.0, split="M5-X abort", notes="scoring run aborted",
             extra=rec)


def mark_scored(summary: dict, *, root: Path | None = None) -> None:
    excl_write(look_dir(root) / "SCORED.json", {"scored": time.strftime("%Y-%m-%dT%H:%M:%S"), **summary})
