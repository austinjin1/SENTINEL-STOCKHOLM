# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Single-look enforcement for the M5-X test pass (pattern of ``sentinel/models/hydromet/testlook.py``).

Order (``scripts/eval/bioassess_x_final.py``):
1. :func:`load_freeze` — ONLY the lead-written ``docs/preregistration/bioassess_x_freeze.json`` (fixed path; the
   candidate record is never accepted): status ``frozen``, ``candidate_sha256`` = sha256 of the candidate file,
   ``scoring_code_sha256`` = the current scoring code; then :func:`verify_data_pins` (manifest sha + content sha
   over the exact files the loaders read) — both before the claim, so a mismatch never burns the look;
2. :func:`claim` — refuses when the registry already holds any ``bioassess_x_test`` row, then atomically creates
   ``$SENTINEL_RUNS/bioassess/x_track/test_look/TEST_LOOK.json`` (O_CREAT | O_EXCL, 0444) and writes the
   claim-time registry row (``bioassess_x_test`` / ``look_claimed``);
3. test rows are loadable only while an unscored claim of THIS freeze exists (:func:`require_open_claim`, called by
   ``xdev.load_test_rows``);
4. predictions are written once and sealed (0444, ``SHA256SUMS``, O_EXCL ``PREDICTIONS_SAVED.json``);
5. scoring + registry rows, then the O_EXCL ``SCORED.json`` marker; nothing can be re-scored after it.
Crash recovery: before step 4, the pre-registered single resume (O_EXCL ``RESUMED.json``; cached units reused);
between 4 and 5, ``--rescore-from-saved`` scores the sealed, sha-checked predictions.
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
import time
from pathlib import Path

from sentinel.core import paths

EXPERIMENT = "bioassess_x_test"
FREEZE_FILE = paths.REPO_ROOT / "docs/preregistration/bioassess_x_freeze.json"
CANDIDATE_FILE = paths.REPO_ROOT / "docs/preregistration/bioassess_x_freeze_candidate.json"
SCORING_CODE = ("scripts/eval/bioassess_x_final.py", "sentinel/models/bioassess/xtestlook.py",
                "sentinel/models/bioassess/xdev.py", "sentinel/models/bioassess/xbioset.py",
                "sentinel/models/bioassess/xdata.py", "sentinel/models/bioassess/xtrack.py",
                "scripts/eval/bioassess_x_bioinfer.R", "scripts/eval/bioassess_x_gjam.R",
                "sentinel/events/bioassess_x.py", "sentinel/eval/events.py")
FREEZE_KEYS = ("status", "candidate_sha256", "scoring_code_sha256", "data_snapshots", "candidates", "comparators",
               "seeds", "n_boot", "n_boot_confirmatory", "alpha")
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


def scoring_code_sha256() -> str:
    h = hashlib.sha256()
    for rel in SCORING_CODE:
        f = paths.REPO_ROOT / rel
        h.update(f"{rel}\t{sha_file(f) if f.is_file() else 'MISSING'}\n".encode())
    return h.hexdigest()


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
def load_freeze(freeze_file: Path | None = None) -> dict:
    """The lead-written FREEZE file only (fixed path); never the candidate record."""
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
    if rec["scoring_code_sha256"] != scoring_code_sha256():
        raise SingleLookError("scoring code changed since the freeze")
    rec["_path"], rec["_sha256"] = str(ff), sha_file(ff)
    return rec


# --------------------------------------------------------------------------- claim
def registry_rows(experiment: str = EXPERIMENT, registry_path=None) -> list[dict]:
    reg = Path(registry_path) if registry_path else paths.REGISTRY_PATH
    out = []
    if reg.is_file():
        with reg.open() as fh:
            for line in fh:
                if experiment in line and json.loads(line).get("experiment") == experiment:
                    out.append(json.loads(line))
    return out


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
    from sentinel.core.registry import log_result

    log_result(workstream="bioassess", experiment=EXPERIMENT, model="m5x_test_look", metric="look_claimed", value=1.0,
               split="M5-X claim (single pass)", notes="claim-time row (single-look guard)", registry_path=registry_path,
               extra={k: c[k] for k in ("freeze_sha256", "authorised_by", "run_dir")})
    return c


def open_claim(root: Path | None = None) -> dict | None:
    m = look_dir(root) / "TEST_LOOK.json"
    if not m.is_file() or (look_dir(root) / "SCORED.json").exists():
        return None
    return json.loads(m.read_text())


def require_open_claim(root: Path | None = None) -> dict:
    """Test rows may be loaded only while an unscored claim of the current freeze exists."""
    c = open_claim(root)
    if c is None:
        raise SingleLookError("M5-X test rows need an open, unscored test-look claim (scripts/eval/bioassess_x_final.py)")
    if FREEZE_FILE.is_file() and c.get("freeze_sha256") != sha_file(FREEZE_FILE):
        raise SingleLookError("the open claim belongs to a different freeze record")
    return c


def claim_resume(authorised_by: str, run_dir: Path, *, root: Path | None = None, registry_path=None) -> dict:
    """Pre-registered single resume: claimed, not sealed, not scored, only the claim row in the registry, never resumed."""
    c = open_claim(root)
    if c is None or c["run_dir"] != str(run_dir) or c["authorised_by"] != authorised_by:
        raise SingleLookError("resume needs the open claim with the same run directory and authorisation")
    if (look_dir(root) / "PREDICTIONS_SAVED.json").exists():
        raise SingleLookError("predictions already sealed: use --rescore-from-saved, not --resume")
    rows = registry_rows(EXPERIMENT, registry_path)
    if any(r.get("metric") != "look_claimed" for r in rows):
        raise SingleLookError("result rows exist: the pass cannot be resumed")
    try:
        excl_write(look_dir(root) / "RESUMED.json", {"resumed": time.strftime("%Y-%m-%dT%H:%M:%S"), **c})
    except FileExistsError as e:
        raise SingleLookError("the single resume was already used") from e
    return c


# --------------------------------------------------------------------------- sealed predictions
def seal_predictions(pred_dir: Path, *, root: Path | None = None) -> dict:
    pred_dir = Path(pred_dir)
    if (look_dir(root) / "PREDICTIONS_SAVED.json").exists() or (pred_dir / "SHA256SUMS").exists():
        raise SingleLookError("predictions already sealed — they are written once")
    files = sorted(pred_dir.rglob("*.parquet"))
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
    excl_write(look_dir(root) / "PREDICTIONS_SAVED.json", rec)
    return rec


def check_sealed(*, root: Path | None = None) -> Path:
    if (look_dir(root) / "SCORED.json").exists():
        raise SingleLookError("already scored — nothing can be re-scored")
    p = look_dir(root) / "PREDICTIONS_SAVED.json"
    if not p.exists():
        raise SingleLookError("no sealed predictions")
    rec = json.loads(p.read_text())
    d = Path(rec["dir"])
    if sha_file(d / "SHA256SUMS") != rec["sha256sums"]:
        raise SingleLookError("SHA256SUMS modified")
    for line in (d / "SHA256SUMS").read_text().splitlines():
        s, n = line.split("  ", 1)
        if sha_file(d / n) != s:
            raise SingleLookError(f"sealed prediction {n} modified")
    return d


def mark_scored(summary: dict, *, root: Path | None = None) -> None:
    excl_write(look_dir(root) / "SCORED.json", {"scored": time.strftime("%Y-%m-%dT%H:%M:%S"), **summary})
