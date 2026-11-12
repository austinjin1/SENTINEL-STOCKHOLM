# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Single-look guard for the SENTINEL-Lite test scoring (pre-test review 2026-11-09, C1–C3; port of
``sentinel/models/hydromet/testlook.py``).

One look, ``lite/test``. It covers Test-A (``test``) and Test-B (``test_time``) of the original 4-state track. The
models, comparators, B, device and data pins come ONLY from ``docs/preregistration/FREEZE_lite.json``, never from
the command line.

Order (``scripts/eval/lite_score_test.py``):
1. :func:`load_frozen`: only ``guard.FREEZE_FILE``, which must have ``status == "FROZEN"``, ``authorised_by`` /
   ``authorised_at``, the frozen split and the current scoring-code hash. Then :func:`verify_data_pins`. Both run
   before the claim, so a mismatch never burns the look.
2. :func:`claim`:
   - refuse if the registry holds any ``lite_test*`` row;
   - atomically create ``$SENTINEL_RUNS/lite/test_look/TEST_LOOK.json`` (O_CREAT | O_EXCL, 0444);
   - write a ``look_claimed`` registry row (experiment ``lite_test_claim``).
3. Protected roles load only while an unscored claim of THIS freeze exists (``guard.check_access`` →
   :func:`require_open_claim`).
4. All predictions are written once, sealed 0444, with ``SHA256SUMS`` and an O_EXCL ``PREDICTIONS_SAVED.json``.
5. Scoring runs only from the sealed predictions. Registry rows are buffered and written only after every block has
   been scored, then the O_EXCL ``SCORED.json`` is written. After that, nothing can be re-scored.

Crash recovery between steps 4 and 5: ``--rescore-from-saved`` scores the sealed, sha-checked predictions.
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

LOOK = "lite/test"
REGISTRY_PREFIX = "lite_test"
FROZEN_STATUS = "FROZEN"


def look_dir() -> Path:
    return paths.RUNS_ROOT / "lite" / "test_look"


def marker() -> Path:
    return look_dir() / "TEST_LOOK.json"


def sha_file(p) -> str:
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


# --------------------------------------------------------------------------- freeze
def validate_frozen(freeze: dict) -> None:
    """Review C2: only an authorised, frozen record unlocks test (a candidate never does)."""
    st = str(freeze.get("status", ""))
    if st != FROZEN_STATUS or "CANDIDATE" in st.upper():
        raise ProtectedDataError(f"freeze status {st!r} is not {FROZEN_STATUS!r} (candidates never unlock test)")
    for k in ("authorised_by", "authorised_at"):
        if not freeze.get(k):
            raise ProtectedDataError(f"freeze lacks {k} (the lead's authorisation)")
    if freeze.get("scoring_code_sha256") in (None, "", "PENDING"):
        raise ProtectedDataError("freeze scoring_code_sha256 is PENDING")


def load_frozen() -> dict:
    """The one accepted freeze file (guard.FREEZE_FILE), fully validated."""
    from . import guard as G

    freeze = G.load_freeze(G.FREEZE_FILE, split=G.load_split())
    validate_frozen(freeze)
    check_packages(freeze)
    freeze["_path"] = str(G.FREEZE_FILE)
    return freeze


def check_packages(freeze: dict) -> None:
    """Delta review W-b: the running environment must have exactly the package versions recorded in the freeze
    (the ``sentinel2`` conda env)."""
    from . import guard as G

    rec = freeze.get("packages") or {}
    now = G.package_versions()
    bad = {k: (rec.get(k), now.get(k)) for k in set(rec) | set(now) if rec.get(k) != now.get(k)}
    if not rec or bad:
        raise ProtectedDataError(f"package versions differ from the freeze (frozen, running): {bad or 'none recorded'}"
                                 " — run in /home/bcheng/miniconda3/envs/sentinel2")


REQUIRED_MODELS = ("primary", "nokd", "deployable_int8_seed0")       # H1, H2, H-G6 (§8a)


def verify_artifacts(freeze: dict) -> dict:
    """Delta review D2: before the claim, every artifact the look reads must exist and match its frozen sha:
    every ``weights_sha256`` entry (model seeds, pool.json, ONNX, export report, event artifacts), every run dir,
    the events module, the teacher manifest + its checkpoints + seen_points. A freeze whose ``not_run`` names a
    model used by H1/H2/H-G6 is refused."""
    from . import guard as G

    stale = sorted(str(p) for p in (paths.RUNS_ROOT / "lite" / "cache").glob("eval_test*"))
    if stale:                                          # E3b: no test chip cache may exist before the look
        raise ProtectedDataError(f"pre-existing test chip cache(s): {stale}")
    w = dict(freeze.get("weights_sha256") or {})
    paths_of: dict[str, Path] = {}
    models = freeze.get("models") or {}
    for need in REQUIRED_MODELS:
        if need not in models or need in (freeze.get("not_run") or {}):
            raise ProtectedDataError(f"freeze lacks model {need!r} needed by H1/H2/H-G6")
    for name, spec in models.items():
        if spec.get("kind") == "torch":
            run = Path(spec["run"])
            if not run.is_dir():
                raise ProtectedDataError(f"run dir of {name} missing: {run}")
            for sd in spec["seeds"]:
                paths_of[f"{name}_seed{sd}"] = run / f"model_seed{sd}.pt"
            if (run / "pool.json").exists():
                paths_of[f"{name}_pool_json"] = run / "pool.json"
        elif spec.get("kind") == "onnx":
            paths_of[name] = Path(spec["path"])
    paths_of["export_report_seed0"] = Path(freeze["export_report"])
    ev = freeze.get("events") or {}
    for k in ("heads", "comparators", "calibrators"):
        if ev:
            paths_of[f"events_{k}"] = Path(ev[k])
    missing_hash = sorted(set(paths_of) - set(w))
    unused_hash = sorted(set(w) - set(paths_of))
    if missing_hash or unused_hash:
        raise ProtectedDataError(f"weights_sha256 / artifacts mismatch: unhashed {missing_hash}, unknown {unused_hash}")
    bad = [k for k, pth in paths_of.items() if not pth.is_file() or G.artifact_sha256(pth) != w[k]]
    if bad:
        raise ProtectedDataError(f"artifact sha256 mismatch or missing: {bad}")
    if ev:
        if sha_file(Path(ev["events_module"])) != ev["events_module_sha256"]:
            raise ProtectedDataError("sentinel/eval/events.py differs from the frozen release")
    tm = Path(freeze["teacher_manifest"])
    if sha_file(tm) != freeze["teacher_manifest_sha256"]:
        raise ProtectedDataError("teacher manifest sha256 differs from the freeze")
    man = json.loads(tm.read_text())
    for c, sha in zip(man["checkpoints"], man["sha256"]):
        if not Path(c).is_file() or G.artifact_sha256(c) != sha:
            raise ProtectedDataError(f"teacher checkpoint {c} sha mismatch")
    if G.artifact_sha256(man["seen_points"]) != man["seen_points_sha256"]:
        raise ProtectedDataError("teacher seen_points sha mismatch")
    return {"n_artifacts": len(paths_of)}


# --------------------------------------------------------------------------- data pins (review C3)
def files_pin(files) -> str:
    """Content sha over (relative path, file sha) of the given files under the data/run roots."""
    h = hashlib.sha256()
    for f in sorted({Path(x) for x in files}):
        if not f.is_file():
            raise ProtectedDataError(f"pinned file missing: {f}")
        h.update(f"{f.as_posix()}\t{sha_file(f)}\n".encode())
    return h.hexdigest()


def compute_data_pins(split: dict) -> dict:
    """Pins over every file the scorer reads for data (coordinates/ids/paths only; no label or chip is decoded):
    the NAIP chip-store shards holding train/test/test_time chip ids + the store index, the DOQQ scene stats, the
    train chip cache, the S2 teacher-store shards + index used by the red-edge gap, the label table, and the
    manifests ``naip_matchups`` / ``lite_labels``."""
    import pandas as pd

    from . import data as D

    lab = pd.read_parquet(D.labels_path(), columns=["naip_chip_id", "s2_chip_id"])
    lab["role"] = lab["naip_chip_id"].astype(str).map(split["chip_role"])
    use = lab[lab["role"].isin(["train", "test", "test_time"])]
    pins = {}
    ns = D.naip_store()
    idx = ns.index.set_index("chip_id")
    shards = sorted(set(idx.loc[use["naip_chip_id"].astype(str), "shard"].astype(str)))
    pins["naip_chips"] = {"files": [str(ns.root / "index.parquet")] + [str(ns.root / "shards" / s) for s in shards]}
    st = D.teacher_store()
    sidx = st.index.set_index("chip_id")
    sids = use["s2_chip_id"].dropna().astype(str)
    sids = sids[sids.isin(sidx.index)]
    s2_shards = sorted(set(sidx.loc[sids, "shard"].astype(str)))
    pins["s2_teacher_chips"] = {"files": [str(st.root / "index.parquet")] + [str(st.root / "shards" / s) for s in s2_shards]}
    pins["scene_stats"] = {"files": [str(paths.PROCESSED / "naip_matchups" / "naip_scene_stats.parquet")]}
    tr = D.role_frame(split, ("train",), training=True)
    key = hashlib.sha256("|".join(tr["naip_chip_id"].astype(str)).encode()).hexdigest()[:12]
    pins["train_chip_cache"] = {"files": [str(paths.RUNS_ROOT / "lite" / "cache" / f"train_{key}.npy")]}
    pins["labels"] = {"files": [str(D.labels_path())]}
    for name in ("naip_matchups", "lite_labels"):
        pins[f"manifest_{name}"] = {"files": [str(paths.MANIFESTS / f"{name}.json")]}
    for k, v in pins.items():
        v["n_files"] = len(v["files"])
        v["content_sha256"] = files_pin(v["files"])
    return pins


def verify_data_pins(freeze: dict) -> dict:
    """Frozen pins must match on disk AND equal pins recomputed through the live loader paths (delta review W-a: a
    different SENTINEL_DATA / SENTINEL_RUNS would make the loaders read unpinned files)."""
    from . import guard as G

    pins = freeze.get("data_snapshots") or {}
    live = compute_data_pins(G.load_split())
    for k, v in live.items():
        fz = pins.get(k)
        if fz is None or [str(x) for x in fz["files"]] != [str(x) for x in v["files"]] \
                or fz["content_sha256"] != v["content_sha256"]:
            raise ProtectedDataError(f"{LOOK}: data pin {k!r} differs from what the loaders read now "
                                     f"(SENTINEL_DATA={paths.DATA_ROOT}, SENTINEL_RUNS={paths.RUNS_ROOT})")
    need = ("naip_chips", "s2_teacher_chips", "scene_stats", "train_chip_cache", "labels", "manifest_naip_matchups",
            "manifest_lite_labels")
    bad = [k for k in need if k not in pins]
    for k, pin in pins.items():
        try:
            if files_pin(pin["files"]) != pin["content_sha256"]:
                bad.append(f"{k}: content changed")
        except ProtectedDataError as e:
            bad.append(f"{k}: {e}")
    if bad:
        raise ProtectedDataError(f"{LOOK}: data pins do not match: {bad}")
    return {k: v["content_sha256"] for k, v in pins.items()}


# --------------------------------------------------------------------------- claim (review C1)
CLAIM_EXP = f"{REGISTRY_PREFIX}_claim"
SEALED_EXP = f"{REGISTRY_PREFIX}_sealed"
ABORT_EXP = f"{REGISTRY_PREFIX}_aborted"
NON_SCORING = (CLAIM_EXP, SEALED_EXP, ABORT_EXP)


def registry_entries(prefix: str = REGISTRY_PREFIX) -> list[dict]:
    """All registry rows whose experiment starts with ``prefix`` (append-only record of the look)."""
    reg = paths.REPO_ROOT / "results" / "registry.jsonl"
    out = []
    if reg.is_file():
        with reg.open() as fh:
            for line in fh:
                if prefix not in line:
                    continue
                r = json.loads(line)
                if str(r.get("experiment", "")).startswith(prefix):
                    out.append(r)
    return out


def scoring_rows() -> list[dict]:
    """Registry rows written by scoring (anything lite_test* except the claim / sealed / aborted rows)."""
    return [r for r in registry_entries() if r.get("experiment") not in NON_SCORING]


def sealed_row() -> dict | None:
    rows = [r for r in registry_entries() if r.get("experiment") == SEALED_EXP]
    if len(rows) > 1:
        raise ProtectedDataError(f"{LOOK}: {len(rows)} sealed rows in the registry")
    return rows[0] if rows else None


def registry_rows(prefix: str = REGISTRY_PREFIX) -> int:
    reg = paths.REPO_ROOT / "results" / "registry.jsonl"
    n = 0
    if reg.is_file():
        with reg.open() as fh:
            for line in fh:
                if prefix not in line:
                    continue
                if str(json.loads(line).get("experiment", "")).startswith(prefix):
                    n += 1
    return n


def claim(freeze: dict) -> dict:
    n = registry_rows()
    if n:
        raise ProtectedDataError(f"{LOOK}: registry already holds {n} {REGISTRY_PREFIX}* rows — the test look was used")
    rec = {"look": LOOK, "freeze_file": freeze["_path"], "freeze_sha256": freeze["_sha256"],
           "scoring_code_sha256": freeze["scoring_code_sha256"], "claimed": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "pid": os.getpid(), "host": socket.gethostname()}
    try:
        _excl_write(marker(), rec)
    except FileExistsError as e:
        raise ProtectedDataError(f"{LOOK}: test look already claimed ({marker()})") from e
    from ...core.registry import log_result

    log_result(workstream="M7-lite", experiment=f"{REGISTRY_PREFIX}_claim", model="lite_test_look", metric="look_claimed",
               value=1.0, split="test claim (single pass)", config={"look": LOOK},
               notes="claim-time row (single-look guard)", extra={"freeze_sha256": freeze["_sha256"], "marker": str(marker())})
    return rec


def open_claim(freeze_sha: str) -> bool:
    """True only between the claim and the seal (delta review D1c): once predictions are sealed (marker file OR the
    append-only ``lite_test_sealed`` registry row), test rows are never loadable again — deleting SCORED.json or
    PREDICTIONS_SAVED.json cannot re-open access."""
    m = marker()
    if not m.is_file():
        return False
    rec = json.loads(m.read_text())
    if rec.get("freeze_sha256") != freeze_sha:
        return False
    if rec.get("pid") != os.getpid() or rec.get("host") != socket.gethostname():
        return False                                   # E3a: only the claiming process may load test rows
    if (look_dir() / "SCORED.json").exists() or (look_dir() / "PREDICTIONS_SAVED.json").exists() \
            or (look_dir() / "ABORTED.json").exists():
        return False
    return sealed_row() is None and not any(r.get("experiment") == ABORT_EXP for r in registry_entries())


def require_open_claim(freeze: dict) -> None:
    if not open_claim(freeze["_sha256"]):
        raise ProtectedDataError("protected roles need an open test-look claim of this freeze "
                                 "(scripts/eval/lite_score_test.py); none exists or the look was scored")


# --------------------------------------------------------------------------- sealed predictions
def pred_dir() -> Path:
    return look_dir() / "predictions"


def seal_predictions(freeze: dict) -> dict:
    """Seal once: 0444 files + SHA256SUMS + O_EXCL PREDICTIONS_SAVED.json + an append-only ``lite_test_sealed`` registry
    row carrying the SHA256SUMS sha and the freeze sha (delta review D1a)."""
    d = pred_dir()
    if (look_dir() / "PREDICTIONS_SAVED.json").exists() or (d / "SHA256SUMS").exists() or sealed_row() is not None:
        raise ProtectedDataError(f"{LOOK}: predictions already sealed — they are written once")
    files = sorted(f for f in d.rglob("*") if f.is_file())
    if not files:
        raise ProtectedDataError(f"{LOOK}: no predictions to seal in {d}")
    sums = {f.relative_to(d).as_posix(): sha_file(f) for f in files}
    for f in files:
        os.chmod(f, 0o444)
    sp = d / "SHA256SUMS"
    fd = os.open(sp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    with os.fdopen(fd, "w") as fh:
        fh.write("".join(f"{s}  {n}\n" for n, s in sums.items()))
    rec = {"dir": str(d), "n_files": len(files), "sha256sums": sha_file(sp), "freeze_sha256": freeze["_sha256"],
           "saved": time.strftime("%Y-%m-%dT%H:%M:%S")}
    _excl_write(look_dir() / "PREDICTIONS_SAVED.json", rec)
    from ...core.registry import log_result

    log_result(workstream="M7-lite", experiment=SEALED_EXP, model="lite_test_look", metric="predictions_sealed",
               value=1.0, split="test sealed (single pass)", config={"look": LOOK},
               notes="sealed-predictions anchor (delta review D1a)",
               extra={"sha256sums": rec["sha256sums"], "freeze_sha256": freeze["_sha256"], "n_files": len(files),
                      "dir": str(d)})
    return rec


def check_sealed(freeze: dict) -> Path:
    """Sealed predictions usable for (re)scoring only if: not yet scored (no SCORED.json and no scoring registry row),
    the claim marker, PREDICTIONS_SAVED.json and the ``lite_test_sealed`` registry row all name THIS freeze, the
    SHA256SUMS sha equals the registry anchor, every listed file matches, and no unlisted file is present."""
    if (look_dir() / "SCORED.json").exists():
        raise ProtectedDataError(f"{LOOK}: already scored — nothing can be re-scored")
    n = len(scoring_rows())
    if n:
        raise ProtectedDataError(f"{LOOK}: the registry already holds {n} scoring rows — nothing can be re-scored")
    reg = sealed_row()
    if reg is None:
        raise ProtectedDataError(f"{LOOK}: no lite_test_sealed registry row (the look crashed before sealing; report it)")
    ex = reg.get("extra") or {}
    if ex.get("freeze_sha256") != freeze["_sha256"]:
        raise ProtectedDataError(f"{LOOK}: the sealed predictions belong to another freeze")
    if not marker().is_file() or json.loads(marker().read_text()).get("freeze_sha256") != freeze["_sha256"]:
        raise ProtectedDataError(f"{LOOK}: the claim marker does not name this freeze")
    p = look_dir() / "PREDICTIONS_SAVED.json"
    if not p.exists():
        raise ProtectedDataError(f"{LOOK}: PREDICTIONS_SAVED.json missing")
    rec = json.loads(p.read_text())
    d = Path(rec["dir"])
    if rec.get("freeze_sha256") != freeze["_sha256"]:
        raise ProtectedDataError(f"{LOOK}: PREDICTIONS_SAVED.json names another freeze")
    s_now = sha_file(d / "SHA256SUMS")
    if s_now != rec["sha256sums"] or s_now != ex.get("sha256sums"):
        raise ProtectedDataError(f"{LOOK}: SHA256SUMS differs from the registry anchor")
    listed = set()
    for line in (d / "SHA256SUMS").read_text().splitlines():
        s, n_ = line.split("  ", 1)
        listed.add(n_)
        if sha_file(d / n_) != s:
            raise ProtectedDataError(f"{LOOK}: sealed prediction {n_} modified")
    extra_files = sorted(f.relative_to(d).as_posix() for f in d.rglob("*") if f.is_file()
                         and f.relative_to(d).as_posix() not in listed and f.name != "SHA256SUMS")
    if extra_files:
        raise ProtectedDataError(f"{LOOK}: unlisted files in the sealed directory: {extra_files[:5]}")
    return d


def mark_aborted(freeze: dict, error: str) -> None:
    """A crash after the claim and before sealing (M6 pattern): O_EXCL ABORTED.json + an append-only registry row.
    The look stays spent (the claim row refuses a new claim); report it to the lead."""
    try:
        _excl_write(look_dir() / "ABORTED.json", {"aborted": time.strftime("%Y-%m-%dT%H:%M:%S"), "error": error[:2000],
                                                 "freeze_sha256": freeze.get("_sha256")})
    finally:
        from ...core.registry import log_result

        log_result(workstream="M7-lite", experiment=ABORT_EXP, model="lite_test_look", metric="look_aborted", value=1.0,
                   split="test aborted before sealing", config={"look": LOOK}, notes="crash after claim, before seal",
                   extra={"error": error[:500], "freeze_sha256": freeze.get("_sha256")})


def mark_scored(summary: dict) -> None:
    _excl_write(look_dir() / "SCORED.json", {"scored": time.strftime("%Y-%m-%dT%H:%M:%S"), **summary})
