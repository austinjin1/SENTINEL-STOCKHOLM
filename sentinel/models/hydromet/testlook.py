# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Single-pass guard for the M6 test looks (pre-test review 2026-11-09, C1–C3; pattern of
``sentinel/models/downstream/testlook.py`` and EW-v2).

Looks (protocol §8, §10, §12): ``M6/test`` (primary), ``M6N/test`` (track N), ``M6R/test`` (track R, separate
freeze file ``FREEZE_hydromet_trackR.json`` with the same scoring-code hash), ``M6/test_b`` (later, own data pin).
Each look's models, Holm family size, comparators, artifacts and data pins come ONLY from its freeze block
``freeze["looks"][<look>]`` — never from the command line.

Order (``scripts/eval/hydromet_test_look.py``):
1. ``guard.load_freeze`` (split, code hash) and ``verify_data_pins`` — before the claim, so a data mismatch never
   burns the look;
2. ``claim`` — refuses when the registry already holds rows of the look's experiment, then atomically creates
   ``$SENTINEL_RUNS/hydromet/test_looks/<look>/TEST_LOOK.json`` (O_CREAT | O_EXCL, 0444) and writes a
   ``look_claimed`` registry row;
3. protected data are loadable only while an unscored claim of THIS freeze exists (``guard.check_access`` →
   :func:`require_open_claim`);
4. predictions are written once, sealed 0444, with ``SHA256SUMS`` and an O_EXCL ``PREDICTIONS_SAVED.json``;
5. scoring + registry rows, then the O_EXCL ``SCORED.json`` marker. After it, nothing can be re-scored.
Crash recovery between 4 and 5: ``--rescore-from-saved`` scores the sealed, sha-checked predictions.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import time
from pathlib import Path

from ...core import paths
from ..sonde.guard import ProtectedDataError

LOOKS = ("M6/test", "M6/test_b", "M6N/test", "M6R/test")
LOOK_KEYS = ("block", "roles", "primary_models", "n_hypotheses", "comparators", "artifacts", "data_snapshots")


def look_dir(look: str) -> Path:
    if look not in LOOKS:
        raise ProtectedDataError(f"unknown test look {look!r}")
    return paths.RUNS_ROOT / "hydromet" / "test_looks" / look.replace("/", "_")


def marker(look: str) -> Path:
    return look_dir(look) / "TEST_LOOK.json"


def _sha_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _excl_write(p: Path, rec: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    with os.fdopen(fd, "w") as fh:
        json.dump(rec, fh, indent=1, default=str)


def look_block(freeze: dict, look: str) -> dict:
    blk = (freeze.get("looks") or {}).get(look)
    if not blk:
        raise ProtectedDataError(f"freeze {freeze.get('_path')} has no block for look {look}")
    missing = [k for k in LOOK_KEYS if k not in blk]
    if missing:
        raise ProtectedDataError(f"freeze block {look} lacks {missing}")
    if int(blk["n_hypotheses"]) != len(blk["primary_models"]) * len(blk["comparators"]):
        raise ProtectedDataError(f"freeze block {look}: n_hypotheses != models × endpoints")
    return blk


# --------------------------------------------------------------------------- data pins
def content_pin(loader_dir: str, patterns) -> str:
    root = paths.PROCESSED / loader_dir
    files = sorted({f for p in patterns for f in root.glob(p) if f.is_file()})
    h = hashlib.sha256()
    for f in files:
        h.update(f"{f.relative_to(root).as_posix()}\t{_sha_file(f)}\n".encode())
    return h.hexdigest()


def verify_data_pins(freeze: dict, look: str) -> dict:
    """Every pin listed by the look must match (manifest sha + content sha over the files the loader reads)."""
    blk = look_block(freeze, look)
    pins = freeze.get("data_snapshots") or {}
    bad = []
    for role in blk["data_snapshots"]:
        pin = pins.get(role)
        if not pin:
            bad.append(f"{role}: no pin")
            continue
        mf = paths.MANIFESTS / f"{pin['manifest']}.json"
        if not mf.is_file() or _sha_file(mf) != pin["manifest_sha256"]:
            bad.append(f"{role}: manifest {pin['manifest']} changed")
        if content_pin(pin["loader_dir_rel"], pin["patterns"]) != pin["content_sha256"]:
            bad.append(f"{role}: content changed")
    if bad:
        raise ProtectedDataError(f"{look}: data pins do not match: {bad}")
    return {r: pins[r]["content_sha256"] for r in blk["data_snapshots"]}


# --------------------------------------------------------------------------- claim
def registry_rows(experiment: str) -> int:
    reg = paths.REPO_ROOT / "results" / "registry.jsonl"
    n = 0
    if reg.is_file():
        with reg.open() as fh:
            for line in fh:
                if experiment not in line:
                    continue
                if json.loads(line).get("experiment") == experiment:
                    n += 1
    return n


def claim(look: str, freeze: dict) -> dict:
    n = registry_rows(look)
    if n:
        raise ProtectedDataError(f"{look}: registry already holds {n} rows — the test look was used")
    rec = {"look": look, "freeze_file": freeze["_path"], "freeze_sha256": freeze["_sha256"],
           "scoring_code_sha256": freeze["scoring_code_sha256"], "claimed": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "pid": os.getpid(), "host": socket.gethostname()}
    try:
        _excl_write(marker(look), rec)
    except FileExistsError as e:
        raise ProtectedDataError(f"{look}: test look already claimed ({marker(look)})") from e
    from ...core.registry import log_result

    log_result(workstream="M6-hydromet", experiment=look, model="hydromet_test_look", metric="look_claimed", value=1.0,
               split=f"{look} claim (single pass)", config={"look": look}, notes="claim-time row (single-pass guard)",
               extra={"freeze_sha256": freeze["_sha256"], "marker": str(marker(look))})
    return rec


def open_claims(freeze_sha: str) -> list[str]:
    """Looks claimed under this freeze and not yet scored."""
    out = []
    root = paths.RUNS_ROOT / "hydromet" / "test_looks"
    for m in sorted(root.glob("*/TEST_LOOK.json")) if root.is_dir() else []:
        rec = json.loads(m.read_text())
        if rec.get("freeze_sha256") == freeze_sha and not (m.parent / "SCORED.json").exists():
            out.append(rec["look"])
    return out


def require_open_claim(freeze: dict) -> str:
    claims = open_claims(freeze["_sha256"])
    if not claims:
        raise ProtectedDataError("protected data need an open test-look claim of this freeze "
                                 "(scripts/eval/hydromet_test_look.py); none exists or the look was scored")
    return claims[0]


# --------------------------------------------------------------------------- sealed predictions
def seal_predictions(look: str, pred_dir: Path) -> dict:
    if (look_dir(look) / "PREDICTIONS_SAVED.json").exists() or (Path(pred_dir) / "SHA256SUMS").exists():
        raise ProtectedDataError(f"{look}: predictions already sealed — they are written once")
    files = sorted(Path(pred_dir).rglob("*.npz"))
    if not files:
        raise ProtectedDataError(f"{look}: no predictions to seal in {pred_dir}")
    sums = {f.relative_to(pred_dir).as_posix(): _sha_file(f) for f in files}
    for f in files:
        os.chmod(f, 0o444)
    sp = Path(pred_dir) / "SHA256SUMS"
    fd = os.open(sp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    with os.fdopen(fd, "w") as fh:
        fh.write("".join(f"{s}  {n}\n" for n, s in sums.items()))
    rec = {"dir": str(pred_dir), "n_files": len(files), "sha256sums": _sha_file(sp),
           "saved": time.strftime("%Y-%m-%dT%H:%M:%S")}
    _excl_write(look_dir(look) / "PREDICTIONS_SAVED.json", rec)
    return rec


def check_sealed(look: str) -> Path:
    if (look_dir(look) / "SCORED.json").exists():
        raise ProtectedDataError(f"{look}: already scored — nothing can be re-scored")
    p = look_dir(look) / "PREDICTIONS_SAVED.json"
    if not p.exists():
        raise ProtectedDataError(f"{look}: no sealed predictions (the look crashed before saving; report it)")
    rec = json.loads(p.read_text())
    d = Path(rec["dir"])
    if _sha_file(d / "SHA256SUMS") != rec["sha256sums"]:
        raise ProtectedDataError(f"{look}: SHA256SUMS modified")
    for line in (d / "SHA256SUMS").read_text().splitlines():
        s, n = line.split("  ", 1)
        if _sha_file(d / n) != s:
            raise ProtectedDataError(f"{look}: sealed prediction {n} modified")
    return d


def mark_scored(look: str, summary: dict) -> None:
    _excl_write(look_dir(look) / "SCORED.json", {"scored": time.strftime("%Y-%m-%dT%H:%M:%S"), **summary})
