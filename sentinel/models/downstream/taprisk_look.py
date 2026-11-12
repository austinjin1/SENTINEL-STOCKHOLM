# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""D4 TapRisk hardened single look (pre-test review 2026-11-09 C1–C4, C7, W6; protocol §11).

Wraps the shared primitives of ``testlook.py`` (FREEZE-hashed by five spent looks → never edited) and adds:

* **pre-claim** (:func:`pre_claim`): FREEZE must be lead-written (status FROZEN, ``authorised_by``/``authorised_at``,
  ``candidate_record_sha256`` = sha of the candidate, text ≠ candidate, no "CANDIDATE" in status); D4 registry /
  import-closure hash; pinned ``SENTINEL_RUNS``/``SENTINEL_DATA``; data + environment pins; no unpinned files inside
  pinned folders (W6);
* **claim** (:func:`claim`): shared O_EXCL marker + registry test-row refusal, then immediately a ``test_claim``
  registry row carrying pid/host/freeze sha (the claim survives marker deletion or another ``SENTINEL_RUNS``);
* **access** (:func:`require_open`): test rows only inside the claiming process (pid/host), only while the registry
  holds this claim and no seal/abort/score, only with matching hashes — closed at sealing;
* **seal** (:func:`seal`): predictions written once (shared ``save_predictions``), set 0444, ``SHA256SUMS``, and a
  ``test_sealed`` registry row with the sha — before any scoring row;
* **abort** (:func:`abort`): any ``BaseException`` before sealing → O_EXCL ``ABORTED.json`` + ``test_aborted`` row;
* **rescore** (:func:`rescore_load`): only with registry claim + seal rows, only while no scoring row exists, only from
  the file whose sha equals the **registry** seal value, only if the look folder holds no unlisted file.
"""

from __future__ import annotations

import json
import os
import socket
import time

from ...core import paths
from . import common as C
from . import testlook as TL

APP = "taprisk"
SCORING_EXPERIMENT = f"{APP}/test_look"
CLAIM_EXPERIMENT = f"{APP}/test_look_claim"
ALLOWED_BEFORE_SCORING = {"TEST_LOOK.json", "PREDICTIONS_SAVED.json", "predictions.parquet", "SHA256SUMS"}


# --------------------------------------------------------------------------- registry view
def _rows() -> list[dict]:
    reg = paths.REPO_ROOT / "results" / "registry.jsonl"
    out = []
    if reg.is_file():
        with reg.open() as fh:
            for line in fh:
                if f'"{APP}/' in line:
                    r = json.loads(line)
                    if r.get("workstream") == C.WORKSTREAM and str(r.get("experiment", "")).startswith(APP + "/"):
                        out.append(r)
    return out


def registry_state() -> dict:
    rows = _rows()
    claim = [r for r in rows if r["experiment"] == CLAIM_EXPERIMENT and r["split"] == "test_claim"]
    sealed = [r for r in rows if r["experiment"] == CLAIM_EXPERIMENT and r["split"] == "test_sealed"]
    aborted = [r for r in rows if r["experiment"] == CLAIM_EXPERIMENT and r["split"] == "test_aborted"]
    scoring = [r for r in rows if r["experiment"] == SCORING_EXPERIMENT]
    return {"claim": claim, "sealed": sealed, "aborted": aborted, "scoring": scoring}


# --------------------------------------------------------------------------- pre-claim checks
def candidate_path():
    return C.PREREG / f"FREEZE_CANDIDATE_downstream_{APP}.json"


def load_freeze() -> dict:
    ff = C.freeze_file(APP)
    if not ff.is_file():
        raise C.ProtectedDataError("no FREEZE_downstream_taprisk.json")
    text = ff.read_text()
    fr = json.loads(text)
    status = str(fr.get("status", ""))
    if not status.startswith("FROZEN") or "CANDIDATE" in status.upper():
        raise C.ProtectedDataError(f"FREEZE status is not a lead freeze: {status[:60]!r}")
    for k in ("authorised_by", "authorised_at", "frozen", "candidate_record_sha256"):
        if not fr.get(k):
            raise C.ProtectedDataError(f"FREEZE lacks {k}")
    cand = candidate_path()
    if not cand.is_file() or TL.file_sha(cand) != fr["candidate_record_sha256"]:
        raise C.ProtectedDataError("FREEZE does not reference the current candidate record")
    if text == cand.read_text():
        raise C.ProtectedDataError("FREEZE is a verbatim copy of the candidate")
    return fr


def verify_pinned_folders(app: str = APP) -> None:
    """W6: every folder that holds a pinned file may contain only pinned files (logs ignored)."""
    pins = json.loads(TL.pins_path(app).read_text())["files"]
    folders = {(paths.DATA_ROOT / rel).parent for rel in pins}
    pinned = {str(paths.DATA_ROOT / rel) for rel in pins}
    extra, subdirs = [], []
    for d in folders:                                   # W-a: recurse; a sub-folder would be read by hive discovery
        for q in d.rglob("*"):
            if q.is_dir():
                subdirs.append(str(q))
            elif q.is_file() and q.suffix != ".log" and str(q) not in pinned:
                extra.append(str(q))
    if subdirs:
        raise C.ProtectedDataError(f"sub-folders inside pinned folders: {subdirs[:3]}")
    if extra:
        raise C.ProtectedDataError(f"unpinned files inside pinned folders: {extra[:3]}")


def env_pin() -> str:
    """E2: environment pin WITHOUT the editable install line (``-e git+...@<HEAD>``), so an unrelated git commit does
    not move it. The repo code itself is covered by the import-closure hash."""
    import subprocess
    import sys

    out = subprocess.run([sys.executable, "-m", "pip", "freeze", "--exclude-editable"],
                         capture_output=True, text=True, check=True).stdout
    return C.sha256_text(out)


def verify_frozen_state(split_sha: str) -> dict:
    """Every check that must hold for BOTH a claim and a rescore (E4): lead freeze, split, import-closure hash,
    pinned paths, data pins, environment pin, pinned-folder contents."""
    from . import taprisk_registry as REG

    fr = load_freeze()
    if fr.get("split_sha256") != split_sha:
        raise C.ProtectedDataError("split sha mismatch")
    if fr.get("registry_scoring_code_sha256") != REG.scoring_code_sha256():
        raise C.ProtectedDataError("D4 import-closure hash changed since the freeze")
    pinned = fr.get("paths", {})
    if pinned.get("runs_root") != str(paths.RUNS_ROOT.resolve()) or pinned.get("data_root") != str(paths.DATA_ROOT.resolve()):
        raise C.ProtectedDataError("SENTINEL_RUNS / SENTINEL_DATA differ from the pinned paths")
    want_env = (fr.get("environment") or {}).get("pip_freeze_noeditable_sha256")
    if not want_env:
        raise C.ProtectedDataError("FREEZE lacks environment.pip_freeze_noeditable_sha256")
    if env_pin() != want_env:
        raise C.ProtectedDataError("python environment (pip freeze --exclude-editable) differs from the pinned one")
    TL.verify_data(APP)
    verify_pinned_folders()
    C.check_roles(APP, {"train", "val", "test"}, allow_test=True, split_sha=split_sha)
    return fr


def pre_claim(split_sha: str) -> dict:
    """Everything that can fail must fail here, before the claim."""
    fr = verify_frozen_state(split_sha)
    st = registry_state()
    if any(st.values()):
        raise C.ProtectedDataError(f"registry already holds D4 look rows: { {k: len(v) for k, v in st.items()} }")
    return fr


# --------------------------------------------------------------------------- claim / access / seal / abort
def claim(split_sha: str) -> dict:
    """E3: once the O_EXCL marker exists, ANY failure while writing the claim row (interrupt, registry error) must
    close the claim with an ABORTED marker + ``test_aborted`` row — never leave a marker-only state."""
    rec = TL.authorise(APP, split_sha)          # shared: check_roles + registry test rows == 0 + O_EXCL marker
    try:
        TL.log_test_metric(app=APP, split_sha=split_sha, experiment="test_look_claim", model="claim", metric="claimed",
                           value=1.0, split="test_claim", extra={"pid": rec["pid"], "host": rec["host"],
                                                                 "freeze_sha256": rec["freeze_sha256"], "claimed": rec["claimed"]})
    except BaseException as e:   # noqa: BLE001 — incl. KeyboardInterrupt / SystemExit
        abort(split_sha, f"claim row not written: {e!r}")
        raise
    return rec


def require_open(split_sha: str) -> None:
    from . import taprisk_registry as REG

    TL.require_authorised(APP, split_sha)
    mk = json.loads(TL.marker_path(APP).read_text())
    if mk.get("pid") != os.getpid() or mk.get("host") != socket.gethostname():
        raise C.ProtectedDataError("test rows are reachable only inside the claiming process")
    d = TL.look_dir(APP)
    if any((d / x).exists() for x in ("PREDICTIONS_SAVED.json", "SCORED.json", "ABORTED.json")):
        raise C.ProtectedDataError("test access is closed (sealed, scored or aborted)")
    st = registry_state()
    mine = [r for r in st["claim"] if (r.get("extra") or {}).get("pid") == os.getpid()]
    if not mine or st["sealed"] or st["aborted"] or st["scoring"]:
        raise C.ProtectedDataError("no open registry claim of this process (or the look is sealed/aborted/scored)")
    fr = json.loads(C.freeze_file(APP).read_text())
    if fr.get("registry_scoring_code_sha256") != REG.scoring_code_sha256():
        raise C.ProtectedDataError("D4 import-closure hash changed")


def seal(split_sha: str, pred) -> dict:
    rec = TL.save_predictions(APP, split_sha, pred)        # O_EXCL PREDICTIONS_SAVED + sha
    p = TL.look_dir(APP) / "predictions.parquet"
    os.chmod(p, 0o444)
    TL._excl_write(TL.look_dir(APP) / "SHA256SUMS", {"predictions.parquet": rec["sha256"]})
    TL.log_test_metric(app=APP, split_sha=split_sha, experiment="test_look_claim", model="seal", metric="n_rows",
                       value=float(rec["n_rows"]), split="test_sealed", extra={"sha256": rec["sha256"], "pid": os.getpid()})
    return rec


def abort(split_sha: str, reason: str) -> None:
    d = TL.look_dir(APP)
    if (d / "PREDICTIONS_SAVED.json").exists():
        return                                             # sealed: only a rescore (if no scoring row) remains
    try:
        TL._excl_write(d / "ABORTED.json", {"reason": reason[:500], "pid": os.getpid(),
                                            "aborted": time.strftime("%Y-%m-%dT%H:%M:%S")})
    finally:
        from ...core.registry import log_result

        log_result(workstream=C.WORKSTREAM, experiment=CLAIM_EXPERIMENT, model="abort", metric="aborted", value=1.0,
                   split="test_aborted", n=None, notes=reason[:500],
                   extra={"pid": os.getpid(), "host": socket.gethostname(), "split_sha256": split_sha, "test_look": True})


def rescore_load(split_sha: str):
    import pandas as pd

    verify_frozen_state(split_sha)          # E4: freeze, split, closure hash, paths, pins, env, folders
    TL.require_authorised(APP, split_sha)
    st = registry_state()
    if not st["claim"] or not st["sealed"]:
        raise C.ProtectedDataError("rescore needs registry claim and seal rows")
    if st["scoring"]:
        raise C.ProtectedDataError("scoring rows exist — a partial-scoring crash goes to the lead, no rescore")
    d = TL.look_dir(APP)
    if (d / "SCORED.json").exists() or (d / "ABORTED.json").exists():
        raise C.ProtectedDataError("look already scored or aborted")
    unlisted = sorted(set(os.listdir(d)) - ALLOWED_BEFORE_SCORING)
    if unlisted:
        raise C.ProtectedDataError(f"unlisted files in the look folder: {unlisted}")
    want = st["sealed"][-1]["extra"]["sha256"]
    p = d / "predictions.parquet"
    if TL.file_sha(p) != want:
        raise C.ProtectedDataError("sealed predictions differ from the registry seal")
    return pd.read_parquet(p)


def mark_scored(summary: dict) -> None:
    TL.mark_scored(APP, summary)
