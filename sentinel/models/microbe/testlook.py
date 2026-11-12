# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Single-look test guard for SENTINEL-Microbe (protocol §8–§9, §10.5, events protocol §7).

Pattern of ``sentinel/models/hydromet/testlook.py`` plus the M7-review extras requested by the lead
(2026-11-09).

**Freeze.** Only the lead-written ``docs/preregistration/FREEZE_microbe.json`` (``status == "FROZEN"``, non-empty
``authorised_by``, no candidate marker) unlocks a test look. ``FREEZE_microbe_candidate.json`` (or any file
carrying :data:`NOT_FREEZE_MARKERS`) is refused. The freeze pins:
- the split sha256;
- the scoring-code sha256: the full import closure of the scoring entry points, including the shared
  ``sentinel.eval.events`` and everything it imports (``ew_scoring``, ``hazards``, ``early_warning``,
  ``metrics``, ``bootstrap``, package ``__init__`` files), plus the M3 configs and protocols;
- data content pins;
- every artifact it lists.

**Order** (``scripts/eval/microbe_final.py``):
1. :func:`load_freeze` → :func:`verify_data_pins` → :func:`verify_artifacts`. All of this happens before the
   claim, so a mismatch never burns the look.
2. :func:`claim`:
   - refuses if the registry holds ANY M3 test row (:data:`TEST_EXPERIMENTS`, or a ``neon_holdout``/``test``
     split);
   - then creates ``TEST_LOOK.json`` with O_CREAT | O_EXCL (0444) and writes a ``look_claimed`` registry row.
3. Holdout rows are loadable (``design.load_design(allow_holdout=True)``) only while an OPEN claim of this
   freeze exists that is held by the calling process (pid and host recorded in ``TEST_LOOK.json``). A claim is open if it is claimed, not sealed, not scored and not aborted (:func:`require_open_claim`).
   **Test access closes at sealing.**
4. :func:`seal_predictions` writes the predictions once:
   - files set to 0444, plus ``SHA256SUMS``;
   - the O_EXCL ``PREDICTIONS_SAVED.json``;
   - a ``predictions_sealed`` registry row carrying the SHA256SUMS digest.
5. Scoring reads only the sealed files (:func:`check_sealed`). That check verifies the digest against
   ``PREDICTIONS_SAVED.json`` **and** the registry sealed row, and refuses unlisted files.
6. Registry rows are written, then the O_EXCL ``SCORED.json``. After that nothing can be re-scored.
7. ``--rescore-from-saved`` is allowed only between sealing and the first scoring row: once any scoring
   row exists it is refused (:func:`assert_no_scoring_rows`).

Event-track errors never fail the primary look. They are recorded as ``event_track_not_run`` (registry
row + entry in ``SCORED.json``).
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import socket
import time
from pathlib import Path

from sentinel.core import paths

WORKSTREAM = "M3-microbe"


class ProtectedDataError(RuntimeError):
    """Raised whenever the M3 test guard refuses an action."""


FREEZE_DIR = paths.REPO_ROOT / "docs" / "preregistration"
FREEZE_NAME = "FREEZE_microbe.json"
CANDIDATE_NAME = "FREEZE_microbe_candidate.json"
NOT_FREEZE_MARKERS = ("NOT_A_FREEZE", "candidate", "partial_rehearsal_only")
FREEZE_KEYS = ("created", "status", "authorised_by", "split_sha256", "scoring_code_sha256", "model", "looks",
               "data_snapshots", "artifacts")
LOOKS = ("M3/test", "M3/test_b")
TEST_EXPERIMENTS = ("M3/test", "M3/test_b", "microbe_test", "microbe_events_test", "microbe_test_b")
SETUP_METRICS = ("look_claimed", "predictions_sealed", "look_aborted")
# Scoring entry points: their sentinel.* import closure is hashed (plus package __init__ files).
ENTRY_POINTS = ("scripts/eval/microbe_final.py", "scripts/eval/events_microbe.py", "sentinel/events/microbe.py",
                "sentinel/models/microbe/testlook.py")
FROZEN_GLOBS = ("sentinel/models/microbe/*.py", "configs/microbe/*.yaml", "configs/events/microbe.yaml",
                "scripts/train/microbe_*.py", "docs/preregistration/microbe_protocol.md",
                "docs/preregistration/events_microbe_protocol.md", "docs/preregistration/microbe_split.json")


# --------------------------------------------------------------------------- code hash
def _module_file(mod: str, root: Path) -> Path | None:
    base = root.joinpath(*mod.split("."))
    if (base / "__init__.py").is_file():
        return base / "__init__.py"
    if base.with_suffix(".py").is_file():
        return base.with_suffix(".py")
    return None


def _imports(f: Path, root: Path) -> set[str]:
    """``sentinel.*`` modules imported by file ``f`` (absolute and relative, any nesting level)."""
    tree = ast.parse(f.read_text(), filename=str(f))
    rel = f.relative_to(root)
    pkg = list(rel.parent.parts)
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
            out |= {f"{mod}.{a.name}" for a in node.names}       # ``from pkg import module``
    return out


def import_closure(root: Path | None = None, entry_points=ENTRY_POINTS) -> list[str]:
    """Repo files reachable from the entry points through ``sentinel.*`` imports, plus every package
    ``__init__.py`` on the way (imports execute them)."""
    root = Path(root) if root else paths.REPO_ROOT
    todo = [root / e for e in entry_points if (root / e).is_file()]
    seen: set[Path] = set()
    while todo:
        f = todo.pop()
        if f in seen:
            continue
        seen.add(f)
        parts = f.relative_to(root).parts
        for i in range(1, len(parts)):                       # parent package __init__ files
            init = root.joinpath(*parts[:i], "__init__.py")
            if init.is_file() and init not in seen:
                todo.append(init)
        for mod in _imports(f, root):
            mf = _module_file(mod, root)
            if mf is not None and mf not in seen:
                todo.append(mf)
    return sorted(str(p.relative_to(root)) for p in seen)


def frozen_files(root: Path | None = None) -> list[str]:
    root = Path(root) if root else paths.REPO_ROOT
    files = set(import_closure(root))
    for g in FROZEN_GLOBS:
        files |= {str(f.relative_to(root)) for f in root.glob(g) if f.is_file()}
    return sorted(files)


def scoring_code_sha256(root: Path | None = None) -> str:
    root = Path(root) if root else paths.REPO_ROOT
    h = hashlib.sha256()
    for rel in frozen_files(root):
        h.update(rel.encode())
        h.update((root / rel).read_bytes())
    return h.hexdigest()


def sha_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- freeze
def load_freeze(freeze_file=None, *, check_code: bool = True) -> dict:
    ff = Path(freeze_file) if freeze_file is not None else FREEZE_DIR / FREEZE_NAME
    if ff.name != FREEZE_NAME or ff.resolve().parent != Path(FREEZE_DIR).resolve():
        raise ProtectedDataError(f"{ff} is not the M3 freeze: only {FREEZE_DIR / FREEZE_NAME} (lead-written) unlocks a test")
    if not ff.is_file():
        raise ProtectedDataError(f"freeze file {ff} does not exist")
    try:
        fr = json.loads(ff.read_text())
    except ValueError as e:
        raise ProtectedDataError(f"{ff} is not valid JSON: {e}") from e
    if not isinstance(fr, dict):
        raise ProtectedDataError(f"{ff} must be a JSON object")
    marked = [k for k in NOT_FREEZE_MARKERS if k in fr]
    if marked or str(fr.get("status", "")).upper() != "FROZEN" or not fr.get("authorised_by"):
        raise ProtectedDataError(f"{ff}: needs status 'FROZEN', a lead 'authorised_by' and no candidate marker {marked}")
    missing = [k for k in FREEZE_KEYS if not fr.get(k)]
    if missing:
        raise ProtectedDataError(f"{ff} lacks required keys {missing}")
    split = FREEZE_DIR / "microbe_split.json"
    if sha_file(split) != fr["split_sha256"]:
        raise ProtectedDataError("microbe_split.json changed since the freeze")
    if check_code and fr["scoring_code_sha256"] != scoring_code_sha256():
        raise ProtectedDataError("scoring code changed since the freeze (scoring_code_sha256 mismatch)")
    fr["_path"] = str(ff)
    fr["_sha256"] = sha_file(ff)
    return fr


def content_pin(rel_dir: str, patterns) -> str:
    root = paths.DATA_ROOT / rel_dir
    files = sorted({f for p in patterns for f in root.glob(p) if f.is_file()})
    if not files:
        raise ProtectedDataError(f"data pin {rel_dir}/{patterns}: no files")
    h = hashlib.sha256()
    for f in files:
        h.update(f"{f.relative_to(root).as_posix()}\t{sha_file(f)}\n".encode())
    return h.hexdigest()


def verify_data_pins(freeze: dict) -> dict:
    bad, out = [], {}
    for role, pin in freeze["data_snapshots"].items():
        try:
            got = content_pin(pin["rel_dir"], pin["patterns"])
        except ProtectedDataError as e:
            bad.append(f"{role}: {e}")
            continue
        if got != pin["content_sha256"]:
            bad.append(f"{role}: content changed")
        out[role] = got
    if bad:
        raise ProtectedDataError(f"data pins do not match: {bad}")
    return out


def verify_artifacts(freeze: dict) -> dict:
    bad = []
    for rel, want in freeze["artifacts"].items():
        p = Path(rel) if Path(rel).is_absolute() else paths.REPO_ROOT / rel
        if not p.is_file() or sha_file(p) != want:
            bad.append(rel)
    if bad:
        raise ProtectedDataError(f"frozen artifacts missing or changed: {bad}")
    return freeze["artifacts"]


# --------------------------------------------------------------------------- registry
def _registry_rows():
    reg = paths.REGISTRY_PATH
    if not Path(reg).is_file():
        return
    with open(reg) as fh:
        for line in fh:
            if WORKSTREAM not in line:
                continue
            r = json.loads(line)
            if r.get("workstream") == WORKSTREAM:
                yield r


def is_test_row(r: dict) -> bool:
    split = str(r.get("split", ""))
    return r.get("experiment") in TEST_EXPERIMENTS or split.startswith("neon_holdout") or split.startswith("M3/test")


def test_rows(look: str | None = None) -> list[dict]:
    return [r for r in _registry_rows() if is_test_row(r) and (look is None or r.get("experiment") == look)]


def assert_no_scoring_rows(look: str) -> None:
    n = [r for r in test_rows(look) if r.get("metric") not in SETUP_METRICS]
    if n:
        raise ProtectedDataError(f"{look}: {len(n)} scoring rows already in the registry — re-scoring refused")


def _log(look, metric, value, **extra):
    from sentinel.core.registry import log_result

    log_result(workstream=WORKSTREAM, experiment=look, model="microbe_test_look", metric=metric, value=float(value),
               split=f"{look} single pass", config={"look": look}, extra=extra, data_manifests=["microbe_wq"])


# --------------------------------------------------------------------------- claim / access
def look_dir(look: str) -> Path:
    if look not in LOOKS:
        raise ProtectedDataError(f"unknown look {look!r}")
    return paths.RUNS_ROOT / "microbe" / "test_looks" / look.replace("/", "_")


def _excl_write(p: Path, rec: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    with os.fdopen(fd, "w") as fh:
        json.dump(rec, fh, indent=1, default=str)


def claim(look: str, freeze: dict) -> dict:
    rows = test_rows()
    if rows:
        raise ProtectedDataError(f"registry already holds {len(rows)} M3 test rows — the test look was used")
    rec = {"look": look, "freeze_file": freeze["_path"], "freeze_sha256": freeze["_sha256"],
           "scoring_code_sha256": freeze["scoring_code_sha256"], "claimed": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "pid": os.getpid(), "host": socket.gethostname()}
    try:
        _excl_write(look_dir(look) / "TEST_LOOK.json", rec)
    except FileExistsError as e:
        raise ProtectedDataError(f"{look}: already claimed ({look_dir(look) / 'TEST_LOOK.json'})") from e
    _log(look, "look_claimed", 1.0, freeze_sha256=freeze["_sha256"])
    return rec


def _state(look: str) -> dict:
    d = look_dir(look)
    return {k: (d / f).exists() for k, f in (("claimed", "TEST_LOOK.json"), ("sealed", "PREDICTIONS_SAVED.json"),
                                             ("scored", "SCORED.json"), ("aborted", "ABORTED.json"))}


def require_open_claim(freeze_sha: str | None = None) -> str:
    """Holdout access: an open claim (claimed, unsealed, unscored, not aborted) of the current freeze."""
    for look in LOOKS:
        st = _state(look)
        if st["claimed"] and not (st["sealed"] or st["scored"] or st["aborted"]):
            rec = json.loads((look_dir(look) / "TEST_LOOK.json").read_text())
            mine = rec.get("pid") == os.getpid() and rec.get("host") == socket.gethostname()
            if mine and (freeze_sha is None or rec.get("freeze_sha256") == freeze_sha):
                return look
    raise ProtectedDataError("holdout rows need an open M3 test-look claim held by THIS process (pid/host; "
                             "scripts/eval/microbe_final.py); access closes when predictions are sealed")


# --------------------------------------------------------------------------- protected loaders (lead ruling I-03)

SYNTHETIC_MARKER = "SYNTHETIC_TEST_ROOT"
SPLIT_FILE = paths.REPO_ROOT / "docs" / "preregistration" / "microbe_split.json"       # never monkeypatched


def real_event_ids() -> frozenset:
    """Every NEON event id of the frozen split (development, holdout, both ``.001`` blocks). Ids are not labels."""
    rec = json.loads(SPLIT_FILE.read_text())
    out = set()
    for k in ("development_ids", "holdout_ids", "release_v1_ids"):
        out.update(map(str, rec[k]))
    if not out:
        raise ProtectedDataError("the frozen split lists no event ids")
    return frozenset(out)


def _inside(p: Path, parent: Path) -> bool:
    try:
        p.relative_to(parent)
        return True
    except ValueError:
        return False


def is_synthetic_root(root, id_file: str, id_col: str = "sample_id") -> bool:
    """True only for a declared synthetic test root. ALL of the following must hold; anything else is treated as a
    root holding real data, however it was passed (argument, symlink, copy):

    1. the directory holds the marker file :data:`SYNTHETIC_MARKER`;
    2. its resolved path is outside the project data root and the processed-data root;
    3. no file in it resolves to a path outside it (no symlink to real data);
    4. none of the ids in ``id_file`` is an event id of the frozen split (a copy of real data fails here even with
       a marker). ``id_file`` is opened only if 1–3 hold, and only its id column is read.

    Any error (missing split file, unreadable id column) counts as "not synthetic"."""
    try:
        if root is None:
            return False
        r = Path(root).resolve()
        if not (r / SYNTHETIC_MARKER).is_file():
            return False
        home_data = (Path.home() / "STOCKHOLM" / "sentinel_data").resolve()
        for real in {paths.DATA_ROOT.resolve(), paths.PROCESSED.resolve(), home_data}:
            if r == real or _inside(r, real):
                return False
        for f in r.iterdir():
            if not _inside(f.resolve(), r):
                return False
        import pyarrow.parquet as pq

        ids = set(map(str, pq.read_table(r / id_file, columns=[id_col]).column(id_col).to_pylist()))
        return not (ids & real_event_ids())
    except ProtectedDataError:
        raise
    except Exception:
        return False


def require_claim_for_root(root, id_file: str, id_col: str = "sample_id") -> str | None:
    """Gate of every loader that can return holdout or reserve rows: the open claim is required for every root
    that holds real data; only a declared synthetic test root (:func:`is_synthetic_root`) is exempt. Returns the
    open look, or None for a synthetic root."""
    if is_synthetic_root(root, id_file, id_col):
        return None
    return require_open_claim()


def abort(look: str, reason: str) -> None:
    if _state(look)["sealed"]:
        raise ProtectedDataError(f"{look}: predictions are sealed — use --rescore-from-saved, not abort")
    _excl_write(look_dir(look) / "ABORTED.json", {"aborted": time.strftime("%Y-%m-%dT%H:%M:%S"), "reason": reason[-4000:]})
    _log(look, "look_aborted", 1.0, reason=reason[-500:])


# --------------------------------------------------------------------------- sealing
def pred_dir(look: str) -> Path:
    return look_dir(look) / "predictions"


def seal_predictions(look: str) -> dict:
    d = pred_dir(look)
    if _state(look)["sealed"] or (d / "SHA256SUMS").exists():
        raise ProtectedDataError(f"{look}: predictions already sealed — written once")
    files = sorted(f for f in d.rglob("*") if f.is_file())
    if not files:
        raise ProtectedDataError(f"{look}: nothing to seal in {d}")
    sums = {f.relative_to(d).as_posix(): sha_file(f) for f in files}
    for f in files:
        os.chmod(f, 0o444)
    sp = d / "SHA256SUMS"
    fd = os.open(sp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    with os.fdopen(fd, "w") as fh:
        fh.write("".join(f"{s}  {n}\n" for n, s in sums.items()))
    rec = {"dir": str(d), "n_files": len(files), "sha256sums": sha_file(sp), "saved": time.strftime("%Y-%m-%dT%H:%M:%S")}
    _excl_write(look_dir(look) / "PREDICTIONS_SAVED.json", rec)      # test access closes here
    _log(look, "predictions_sealed", float(len(files)), sha256sums=rec["sha256sums"])
    return rec


def check_sealed(look: str) -> Path:
    st = _state(look)
    if st["scored"]:
        raise ProtectedDataError(f"{look}: already scored — nothing can be re-scored")
    if not st["sealed"]:
        raise ProtectedDataError(f"{look}: no sealed predictions")
    rec = json.loads((look_dir(look) / "PREDICTIONS_SAVED.json").read_text())
    d = Path(rec["dir"])
    digest = sha_file(d / "SHA256SUMS")
    if digest != rec["sha256sums"]:
        raise ProtectedDataError(f"{look}: SHA256SUMS modified")
    sealed_rows = [r for r in test_rows(look) if r.get("metric") == "predictions_sealed"]
    if len(sealed_rows) != 1 or (sealed_rows[0].get("extra") or {}).get("sha256sums") != digest:
        raise ProtectedDataError(f"{look}: sealed files do not match the registry sealed row")
    listed = dict(line.split("  ", 1)[::-1] for line in (d / "SHA256SUMS").read_text().splitlines())
    present = {f.relative_to(d).as_posix() for f in d.rglob("*") if f.is_file() and f.name != "SHA256SUMS"}
    extra = sorted(present - set(listed))
    if extra:
        raise ProtectedDataError(f"{look}: unlisted files in the sealed folder: {extra[:5]}")
    for n, s in listed.items():
        if not (d / n).is_file() or sha_file(d / n) != s:
            raise ProtectedDataError(f"{look}: sealed file {n} modified or missing")
    return d


def mark_scored(look: str, summary: dict) -> None:
    _excl_write(look_dir(look) / "SCORED.json", {"scored": time.strftime("%Y-%m-%dT%H:%M:%S"), **summary})


def record_event_track_not_run(look: str, err: str) -> None:
    _log(look, "event_track_not_run", 1.0, error=err[-1000:])
