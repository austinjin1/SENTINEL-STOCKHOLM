# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Single-pass guard for the SENTINEL-Orbit test looks (orbit_protocol.md §9b; pattern of
``sentinel/models/hydromet/testlook.py``).

Looks:
* ``M2/test``  — headline (test role x 2024-25) + Test-B + pooled + track-E strata + event track (§12);
* ``M2U/test`` — track U (§11.2).

Order (``scripts/eval/orbit_test.py``):
1. ``guard.load_freeze`` (only docs/preregistration/FREEZE_orbit.json, status FROZEN, lead authorisation, code hash,
   split, dev manifest) and :func:`verify_data_pins` — before the claim, so a data mismatch never burns the look;
2. :func:`claim` — refuses when the registry already holds rows of the look's experiment, then atomically creates
   ``$SENTINEL_RUNS/orbit/test_looks/<look>/TEST_LOOK.json`` (O_CREAT | O_EXCL, 0444) and writes a ``look_claimed``
   registry row;
3. protected rows are loadable only while an unscored claim of THIS freeze exists (``guard.check_rows`` →
   :func:`require_open_claim`);
4. predictions are written once, sealed 0444, with ``SHA256SUMS`` and an O_EXCL ``PREDICTIONS_SAVED.json``;
5. scoring reads only the sealed files; registry rows; then the O_EXCL ``SCORED.json`` marker.
Crash recovery between 4 and 5: ``--rescore-from-saved`` scores the sealed, sha-checked predictions.

The development dry run uses the same functions under ``root=test_looks_dryrun`` (no freeze, no protected rows).
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

LOOKS = ("M2/test", "M2U/test")
LOOK_EXPERIMENT = {"M2/test": "orbit_TEST_{v}", "M2U/test": "orbit_TEST_{v}_trackU"}
ROOT = "test_looks"
DRYRUN_ROOT = "test_looks_dryrun"

# Source files the protected loader (data.build_table + split + track U) reads, relative to paths.PROCESSED.
# Pinned by content (sha256 of every file) in FREEZE data_pins; written by scripts/eval/orbit_prepare_freeze.py.
DATA_PIN_SPECS = {
    "wqp_matchups": ("", ["s2_wqp_matchups/samples.parquet", "s2_wqp_matchups/matchups.parquet",
                          "s2_wqp_matchups_hist/samples.parquet", "s2_wqp_matchups_hist/matchups.parquet",
                          "s2_wqp_matchups_2026/samples.parquet", "s2_wqp_matchups_2026/matchups.parquet"]),
    "aquacorr_tables": ("aquacorr_global", ["s2_chip_requests.parquet", "s2_chip_plan.parquet"]),
    "chips_wqp": ("s2_chips/wqp", ["index.parquet", "index/*", "layers.json", "shards/*"]),
    "chips_aquacorr": ("s2_chips/aquacorr", ["index.parquet", "index/*", "layers.json", "shards/*"]),
    "hydrolakes": ("hydrolakes", ["hydrolakes_na_eu.parquet"]),
    "wbd_huc12": ("wbd_huc12", ["huc12.parquet"]),
    "gloria_srf": ("gloria", ["srf_info.parquet"]),
}


def look_dir(look: str, root: str = ROOT) -> Path:
    if look not in LOOKS:
        raise ProtectedDataError(f"unknown test look {look!r}")
    return paths.RUNS_ROOT / "orbit" / root / look.replace("/", "_")


def marker(look: str, root: str = ROOT) -> Path:
    return look_dir(look, root) / "TEST_LOOK.json"


def experiment(look: str, version: str) -> str:
    return LOOK_EXPERIMENT[look].format(v=version)


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


# --------------------------------------------------------------------------- data pins
def pin_files(loader_dir: str, patterns) -> list[Path]:
    root = paths.PROCESSED / loader_dir if loader_dir else paths.PROCESSED
    return sorted({f for p in patterns for f in root.glob(p) if f.is_file()})


def content_pin(loader_dir: str, patterns) -> dict:
    """sha256 over (relative path, sha256) of every file matching ``patterns`` under PROCESSED/loader_dir."""
    root = paths.PROCESSED / loader_dir if loader_dir else paths.PROCESSED
    files = pin_files(loader_dir, patterns)
    h = hashlib.sha256()
    nbytes = 0
    for f in files:
        h.update(f"{f.relative_to(root).as_posix()}\t{_sha_file(f)}\n".encode())
        nbytes += f.stat().st_size
    return {"loader_dir_rel": loader_dir, "patterns": list(patterns), "n_files": len(files), "bytes": nbytes,
            "content_sha256": h.hexdigest()}


def make_data_pins() -> dict:
    return {name: content_pin(d, pats) for name, (d, pats) in DATA_PIN_SPECS.items()}


def verify_data_pins(freeze: dict) -> dict:
    """Every pin of the freeze must match, and every DATA_PIN_SPECS entry must be pinned."""
    pins = freeze.get("data_pins") or {}
    bad = [f"{k}: no pin" for k in DATA_PIN_SPECS if k not in pins]
    for name, pin in pins.items():
        now = content_pin(pin["loader_dir_rel"], pin["patterns"])
        if now["content_sha256"] != pin["content_sha256"] or now["n_files"] != pin["n_files"]:
            bad.append(f"{name}: content changed ({now['n_files']} files vs {pin['n_files']})")
    if bad:
        raise ProtectedDataError(f"data pins do not match: {bad}")
    return {k: v["content_sha256"] for k, v in pins.items()}


# --------------------------------------------------------------------------- claim
def registry_rows(experiment_name: str) -> int:
    reg = paths.REPO_ROOT / "results" / "registry.jsonl"
    n = 0
    if reg.is_file():
        with reg.open() as fh:
            for line in fh:
                if experiment_name not in line:
                    continue
                if json.loads(line).get("experiment") == experiment_name:
                    n += 1
    return n


def claim(look: str, freeze: dict, *, root: str = ROOT, register: bool = True) -> dict:
    exp = experiment(look, freeze["data_version"])
    n = registry_rows(exp)
    if n:
        raise ProtectedDataError(f"{look}: registry already holds {n} rows of {exp} — the test look was used")
    rec = {"look": look, "experiment": exp, "freeze_file": freeze.get("_path"), "freeze_sha256": freeze["_sha256"],
           "scoring_code_sha256": freeze.get("scoring_code_sha256"), "claimed": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "pid": os.getpid(), "host": socket.gethostname()}
    try:
        _excl_write(marker(look, root), rec)
    except FileExistsError as e:
        raise ProtectedDataError(f"{look}: test look already claimed ({marker(look, root)})") from e
    if register:
        from ...core.registry import log_result

        log_result(workstream="M2-orbit", experiment=exp, model="orbit_test_look", metric="look_claimed", value=1.0,
                   split=f"{look} claim (single pass)", config={"look": look},
                   notes="claim-time row (single-pass guard, orbit_protocol.md §9b)",
                   extra={"freeze_sha256": freeze["_sha256"], "marker": str(marker(look, root))})
    return rec


def open_claims(freeze_sha: str, root: str = ROOT) -> list[str]:
    """Looks claimed under this freeze and not yet scored."""
    out = []
    base = paths.RUNS_ROOT / "orbit" / root
    for m in sorted(base.glob("*/TEST_LOOK.json")) if base.is_dir() else []:
        rec = json.loads(m.read_text())
        if rec.get("freeze_sha256") == freeze_sha and not (m.parent / "SCORED.json").exists():
            out.append(rec["look"])
    return out


def require_open_claim(freeze: dict) -> str:
    claims = open_claims(freeze["_sha256"])
    if not claims:
        raise ProtectedDataError("protected rows need an open test-look claim of this freeze "
                                 "(scripts/eval/orbit_test.py); none exists or the look was already scored")
    return claims[0]


# --------------------------------------------------------------------------- sealed predictions
PRED_SUFFIXES = (".parquet", ".npz", ".json")


def seal_predictions(look: str, pred_dir: Path, root: str = ROOT) -> dict:
    pred_dir = Path(pred_dir)
    if (look_dir(look, root) / "PREDICTIONS_SAVED.json").exists() or (pred_dir / "SHA256SUMS").exists():
        raise ProtectedDataError(f"{look}: predictions already sealed — they are written once")
    files = sorted(f for f in pred_dir.rglob("*") if f.is_file() and f.suffix in PRED_SUFFIXES)
    if not files:
        raise ProtectedDataError(f"{look}: no predictions to seal in {pred_dir}")
    sums = {f.relative_to(pred_dir).as_posix(): _sha_file(f) for f in files}
    for f in files:
        os.chmod(f, 0o444)
    sp = pred_dir / "SHA256SUMS"
    fd = os.open(sp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    with os.fdopen(fd, "w") as fh:
        fh.write("".join(f"{s}  {n}\n" for n, s in sums.items()))
    rec = {"dir": str(pred_dir), "n_files": len(files), "sha256sums": _sha_file(sp),
           "saved": time.strftime("%Y-%m-%dT%H:%M:%S")}
    _excl_write(look_dir(look, root) / "PREDICTIONS_SAVED.json", rec)
    return rec


def check_sealed(look: str, root: str = ROOT) -> Path:
    if (look_dir(look, root) / "SCORED.json").exists():
        raise ProtectedDataError(f"{look}: already scored — nothing can be re-scored")
    p = look_dir(look, root) / "PREDICTIONS_SAVED.json"
    if not p.exists():
        raise ProtectedDataError(f"{look}: no sealed predictions (the look crashed before saving; report it)")
    rec = json.loads(p.read_text())
    d = Path(rec["dir"])
    if _sha_file(d / "SHA256SUMS") != rec["sha256sums"]:
        raise ProtectedDataError(f"{look}: SHA256SUMS modified")
    listed = set()
    for line in (d / "SHA256SUMS").read_text().splitlines():
        s, n = line.split("  ", 1)
        listed.add(n)
        if _sha_file(d / n) != s:
            raise ProtectedDataError(f"{look}: sealed prediction {n} modified")
    extra = {f.relative_to(d).as_posix() for f in d.rglob("*") if f.is_file() and f.suffix in PRED_SUFFIXES} - listed
    if extra:
        raise ProtectedDataError(f"{look}: unsealed files next to the sealed predictions: {sorted(extra)[:3]}")
    return d


def mark_scored(look: str, summary: dict, root: str = ROOT) -> None:
    _excl_write(look_dir(look, root) / "SCORED.json", {"scored": time.strftime("%Y-%m-%dT%H:%M:%S"), **summary})
