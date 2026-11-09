# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""One-time scoring guard for the ChemTox unseen-taxa blocks (review item C-M4U-6).

Follows the pattern `sentinel.models.downstream.testlook` already established: a pinned data
freeze, an exclusive-write authorisation marker per block, a registry refusal, a choke-point
loader that no stage-2 script can bypass, write-once predictions and sealed outputs.

Order for every scored pass::

    verify_freeze()                       # the two parquet files and the plan must match their pins
    authorise(block, code_sha=...)        # refuses if the registry already holds rows for the block
    rows = load_test_block(block)         # the ONLY way stage-2 code may read test rows
    save_predictions(block, preds)        # write-once, before any registry write
    ... score, log_result ...
    mark_scored(block, summary)           # nothing can be scored again afterwards

Pins live in ``docs/preregistration/chemtox_unseen_taxa_pins.json`` and are written by
:func:`make_pins` at freeze time. The markers live under
``$SENTINEL_RUNS/chemtox/unseen_taxa/<block>/``.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import time
from pathlib import Path

from ...core import paths

PLAN = "docs/preregistration/chemtox_unseen_taxa_plan.md"
PINS = "docs/preregistration/chemtox_unseen_taxa_pins.json"
DATA_DIR = "processed/chemtox_unseen_taxa"
EXPERIMENT = "chemtox_unseen_taxa_test"
BLOCKS = ("insects", "molluscs", "amphibians")


class ProtectedBlockError(RuntimeError):
    """Raised whenever a test block would be read, re-read or re-scored outside the protocol."""


def file_sha(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def pins_path() -> Path:
    return paths.REPO_ROOT / PINS


def data_path(name: str) -> Path:
    return paths.DATA_ROOT / DATA_DIR / name


def make_pins() -> dict:
    """Pin the frozen artefacts (written once, at freeze time)."""
    body = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "files": {n: file_sha(data_path(n)) for n in ("test_blocks.parquet",
                                                      "training_pool.parquet")},
        "plan_sha256": file_sha(paths.REPO_ROOT / PLAN),
        "module_sha256": file_sha(paths.REPO_ROOT / "sentinel/models/chemtox/unseen_taxa.py"),
        "guard_sha256": None,
    }
    body["guard_sha256"] = file_sha(Path(__file__))
    pins_path().write_text(json.dumps(body, indent=1, sort_keys=True))
    return body


def verify_freeze(*, check_plan: bool = True) -> dict:
    """Every pinned file must still match. Raises otherwise; returns the pins."""
    p = pins_path()
    if not p.is_file():
        raise ProtectedBlockError(f"{PINS} does not exist — no scoring without a pinned freeze")
    pins = json.loads(p.read_text())
    bad = [n for n, sha in pins["files"].items()
           if not data_path(n).is_file() or file_sha(data_path(n)) != sha]
    if bad:
        raise ProtectedBlockError(f"frozen data changed or missing: {bad}")
    if check_plan and file_sha(paths.REPO_ROOT / PLAN) != pins["plan_sha256"]:
        raise ProtectedBlockError(
            "the protocol has changed since the freeze — re-freeze and re-review before scoring")
    # the code that defines the features, the folds and this guard is pinned too (delta review
    # W-D-1): without these two checks `unseen_taxa.py` could change after the freeze undetected
    for key, rel in (("module_sha256", "sentinel/models/chemtox/unseen_taxa.py"),
                     ("guard_sha256", "sentinel/models/chemtox/unseen_taxa_guard.py")):
        want = pins.get(key)
        got = file_sha(Path(__file__).parent / Path(rel).name) if rel.endswith(Path(__file__).name) \
            else file_sha(paths.REPO_ROOT / rel)
        if want and got != want:
            raise ProtectedBlockError(
                f"{rel} changed since the freeze ({key}) — re-freeze and re-review before scoring")
    return pins


def block_dir(block: str) -> Path:
    if block not in BLOCKS:
        raise KeyError(block)
    return paths.RUNS_ROOT / "chemtox" / "unseen_taxa" / block


def _marker(block: str, name: str) -> Path:
    return block_dir(block) / name


def _excl_write(p: Path, rec: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    with os.fdopen(fd, "w") as fh:
        json.dump(rec, fh, indent=1)


def registry_rows(block: str) -> int:
    """Rows this block already has in the registry (a non-zero count forbids a second look)."""
    reg = paths.REPO_ROOT / "results" / "registry.jsonl"
    n = 0
    if reg.is_file():
        with reg.open() as fh:
            for line in fh:
                if EXPERIMENT not in line:
                    continue
                r = json.loads(line)
                if r.get("experiment") == EXPERIMENT and f"/{block}/" in str(r.get("split", "")):
                    n += 1
    return n


def authorise(block: str, *, code_sha: str, note: str = "") -> dict:
    """The only entry to a scored pass: verify the freeze, refuse a second look, claim the block."""
    pins = verify_freeze()
    n = registry_rows(block)
    if n:
        raise ProtectedBlockError(
            f"{block}: the registry already holds {n} row(s) for this block — the single look is spent")
    rec = {"block": block, "claimed": time.strftime("%Y-%m-%dT%H:%M:%S"), "pid": os.getpid(),
           "host": socket.gethostname(), "code_sha256": code_sha,
           "data_sha256": pins["files"]["test_blocks.parquet"],
           "plan_sha256": pins["plan_sha256"], "note": note}
    try:
        _excl_write(_marker(block, "AUTHORISED.json"), rec)
    except FileExistsError as e:
        raise ProtectedBlockError(f"{block}: already claimed ({_marker(block, 'AUTHORISED.json')})") from e
    return rec


def require_authorised(block: str) -> dict:
    """Raise unless this block has a live claim belonging to the current freeze."""
    pins = verify_freeze()
    p = _marker(block, "AUTHORISED.json")
    if not p.is_file():
        raise ProtectedBlockError(f"{block}: test rows requested without an authorised look")
    rec = json.loads(p.read_text())
    if rec.get("data_sha256") != pins["files"]["test_blocks.parquet"]:
        raise ProtectedBlockError(f"{block}: the claim belongs to a different data freeze")
    if _marker(block, "SCORED.json").is_file():
        raise ProtectedBlockError(f"{block}: already scored — the single look is spent")
    return rec


def load_test_block(block: str, *, columns=None):
    """The ONLY way stage-2 code may read test rows: authorised, hash-verified, logged."""
    import pandas as pd

    require_authorised(block)
    df = pd.read_parquet(data_path("test_blocks.parquet"), columns=columns)
    return df[df["block"] == block].reset_index(drop=True)


def load_training_pool(*, columns=None):
    """The training pool is not protected (it holds no test row), but it is hash-verified."""
    import pandas as pd

    verify_freeze(check_plan=False)
    return pd.read_parquet(data_path("training_pool.parquet"), columns=columns)


def save_predictions(block: str, pred) -> dict:
    """Write every prediction (incl. per-seed) ONCE, before any registry write."""
    require_authorised(block)
    if _marker(block, "PREDICTIONS_SAVED.json").exists():
        raise ProtectedBlockError(f"{block}: predictions already saved — re-fitting is not allowed")
    d = block_dir(block)
    d.mkdir(parents=True, exist_ok=True)
    out = d / "predictions.parquet"
    pred.to_parquet(out, index=False)
    os.chmod(out, 0o444)
    rec = {"file": str(out), "sha256": file_sha(out), "n_rows": int(len(pred)),
           "saved": time.strftime("%Y-%m-%dT%H:%M:%S")}
    _excl_write(_marker(block, "PREDICTIONS_SAVED.json"), rec)
    return rec


def mark_scored(block: str, summary: dict) -> None:
    """Close the look. Also seals every file in the block directory and writes SHA256SUMS."""
    _excl_write(_marker(block, "SCORED.json"),
                {"scored": time.strftime("%Y-%m-%dT%H:%M:%S"), **summary})
    seal_dir(block_dir(block))


def seal_dir(d: Path) -> Path:
    """``chmod 0444`` every file in ``d`` and write a ``SHA256SUMS`` sidecar."""
    d = Path(d)
    files = sorted(p for p in d.iterdir() if p.is_file() and p.name != "SHA256SUMS")
    lines = [f"{file_sha(p)}  {p.name}" for p in files]
    sums = d / "SHA256SUMS"
    if sums.exists():
        os.chmod(sums, 0o644)
    sums.write_text("\n".join(lines) + "\n")
    for p in files + [sums]:
        os.chmod(p, 0o444)
    return sums
