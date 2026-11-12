#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Primary-eligibility verification for the EW-v2 live trial (review C-D1). Scorer side; OUTSIDE the issuing closure.
Runs in the dedicated ``sentinel2-live`` environment (opentimestamps library + client).

Evidence is never trusted from local metadata:
  1. Integrity of the evidence record. Every evidence file listed in a later hash-log row (``evidence_refs``) must
     exist with that SHA-256. A missing or altered file is RESTORED from the public pushed copy if that copy matches the
     committed SHA-256; otherwise it is reported as unrecoverable and its row is "issued but not eligible" (review C-E2).
     Deleting a proof after the next row is written is therefore always detected and reported.
  2. Third-party verification of each proof. A temporary COPY of the ``.ots`` proof is upgraded from the calendars
     (``ots upgrade``; the committed file is untouched). The detached file digest must equal SHA-256 of the canonical row
     bytes. For every Bitcoin block-header attestation, the attested message (merkle root) and the block time are
     checked against TWO independent block explorers, which must agree.
  3. Existence time of row i = min(own attestation time, attestation time of any later row j), because row j chains
     to row i through ``prev_row_sha256``.
  4. Primary-eligible iff status ``issued``, ``on_time``, and existence time ≤ D 23:59:59 UTC.
``self_test()`` must pass (client present, ≥ 2 calendars and both explorers reachable, explorers agree on the genesis
merkle root) or the scorer refuses to claim a look.

Calendar note: the OTS protocol retrieves upgrades by the COMMITMENT inside each proof (derived from the calendar's
reply and a client nonce), not by the row hash. A proof is therefore only retrievable via its file, which is why the
files are hash-committed into the next row (step 1).
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
import requests

import live_common as LC
import stamp as ST

EXPLORERS = ("https://blockstream.info/api", "https://mempool.space/api")
GENESIS_MERKLE = "4a5e1e4baab89f3a32518a88c31bc87f618f76673e2cc77ab2127b7afdeda33b"
UA = {"User-Agent": "SENTINEL-v2 live trial (timestamp verification)"}


class EvidenceDestroyed(PermissionError):
    pass


class VerificationUnavailable(PermissionError):
    pass


def _get(url, timeout=30):
    r = requests.get(url, timeout=timeout, headers=UA)
    r.raise_for_status()
    return r


def block_header(height: int) -> tuple[str, int]:
    seen = []
    for e in EXPLORERS:
        h = _get(f"{e}/block-height/{int(height)}").text.strip()
        b = _get(f"{e}/block/{h}").json()
        seen.append((b["merkle_root"], int(b["timestamp"])))
    if len(set(seen)) != 1:
        raise VerificationUnavailable(f"explorers disagree on block {height}")
    return seen[0]


def self_test() -> dict:
    out = {}
    try:
        import opentimestamps  # noqa: F401
    except ImportError as e:
        raise VerificationUnavailable("opentimestamps library missing (run in the sentinel2-live env)") from e
    if not ST.OTS_EXE.is_file():
        raise VerificationUnavailable("ots client missing")
    ok = 0
    for c in ST.CALENDARS:
        try:
            ok += requests.get(c, timeout=20, headers=UA).status_code < 500
        except requests.RequestException:
            pass
    if ok < 2:
        raise VerificationUnavailable(f"only {ok} calendars reachable")
    try:
        mr, _ = block_header(0)
    except requests.RequestException as e:
        raise VerificationUnavailable(f"block explorers unreachable ({e.__class__.__name__})") from e
    if mr != GENESIS_MERKLE:
        raise VerificationUnavailable("explorer returned a wrong genesis merkle root")
    out.update({"calendars_reachable": ok, "explorers": list(EXPLORERS)})
    return out


def _public_copy(name: str, remote: str = "origin", branch: str = "v2-rebuild") -> bytes | None:
    """The pushed (public) copy of an evidence file, or None (review C-E2 recommendation: restore, don't refuse)."""
    rel = (ST.evidence_dir("live") / name).relative_to(LC.REPO).as_posix()
    subprocess.run(["git", "-C", str(LC.REPO), "fetch", "--quiet", remote, branch], capture_output=True, timeout=300,
                   check=False)
    r = subprocess.run(["git", "-C", str(LC.REPO), "show", f"{remote}/{branch}:{rel}"], capture_output=True,
                       timeout=120, check=False)
    return r.stdout if r.returncode == 0 else None


def check_refs(rows: list[dict], mode: str, *, restore: bool = True) -> dict:
    """Every evidence file referenced by a later row must exist with that SHA-256. A missing or altered file is
    RESTORED from the public pushed copy when that copy matches the committed SHA-256; otherwise it is reported as
    unrecoverable, and the affected row is 'issued but not eligible' (it never empties the look). Returns
    {"n_refs", "restored": [...], "unrecoverable": {row_sha256: [files]}}."""
    d = ST.evidence_dir(mode)
    out = {"n_refs": 0, "restored": [], "unrecoverable": {}}
    by_name = {}
    for r in rows:
        for name, sha in (r.get("evidence_refs") or {}).items():
            by_name[name] = sha
    for name, sha in by_name.items():
        out["n_refs"] += 1
        f = d / name
        if f.is_file() and LC.sha256_file(f) == sha:
            continue
        pub = _public_copy(name) if (restore and mode == "live") else None
        if pub is not None and hashlib.sha256(pub).hexdigest() == sha:
            f.write_bytes(pub)
            out["restored"].append(name)
            continue
        row_sha = name.split("_")[1].split(".")[0] if "_" in name else name     # <seq>_<sha12>.*
        out["unrecoverable"].setdefault(row_sha, []).append(name)
    return out


def proof_time(row_file: Path, ots_file: Path) -> pd.Timestamp | None:
    from opentimestamps.core.notary import BitcoinBlockHeaderAttestation
    from opentimestamps.core.serialize import StreamDeserializationContext
    from opentimestamps.core.timestamp import DetachedTimestampFile

    with tempfile.TemporaryDirectory() as td:
        tp = Path(td) / ots_file.name
        shutil.copy(ots_file, tp)
        subprocess.run([str(ST.OTS_EXE), "upgrade", str(tp)], capture_output=True, text=True, timeout=600, check=False)
        with open(tp, "rb") as fh:
            det = DetachedTimestampFile.deserialize(StreamDeserializationContext(fh))
    if det.file_digest != hashlib.sha256(row_file.read_bytes()).digest():
        raise EvidenceDestroyed(f"{ots_file.name} does not commit to {row_file.name}")
    times = []
    for msg, att in det.timestamp.all_attestations():
        if isinstance(att, BitcoinBlockHeaderAttestation):
            mr, t = block_header(att.height)
            if msg[::-1].hex() == mr:
                times.append(pd.Timestamp(t, unit="s", tz="UTC"))
    return min(times) if times else None


def primary_eligibility(mode: str = "live") -> pd.DataFrame:
    path = LC.HASHLOG if mode == "live" else LC.DRYRUN_ROOT / "dryrun_hashlog.jsonl"
    rows = LC.read_hashlog(path)
    refs = check_refs(rows, mode)
    lost = set(refs["unrecoverable"])                                   # sha12 of rows whose proof is unrecoverable
    ev = ST.load_evidence(mode)
    d = ST.evidence_dir(mode)
    own = {}
    for r in rows:
        t = None
        for rec in ([] if r["row_sha256"][:12] in lost else ev.get(r["row_sha256"], [])):
            if rec.get("ots_file") and (d / rec["ots_file"]).is_file() and (d / rec["row_file"]).is_file():
                tt = proof_time(d / rec["row_file"], d / rec["ots_file"])
                t = tt if t is None or (tt is not None and tt < t) else t
        own[r["seq"]] = t
    out, later = [], None
    for r in sorted(rows, key=lambda x: -x["seq"]):                  # existence time via chaining (step 3)
        cand = [x for x in (own[r["seq"]], later) if x is not None]
        exist = min(cand) if cand else None
        later = exist
        if r.get("status", "issued") == "issued":
            deadline = pd.Timestamp(f"{r['issue_date']} 23:59:59", tz="UTC")
            out.append({"issue_date": r["issue_date"], "row_sha256": r["row_sha256"], "on_time": bool(r.get("on_time")),
                        "own_attestation": own[r["seq"]], "existence_time": exist,
                        "proof_unrecoverable": r["row_sha256"][:12] in lost,
                        "primary_eligible": bool(r.get("on_time") and exist is not None and exist <= deadline)})
    df = pd.DataFrame(out)
    df.attrs["evidence_refs"] = refs
    return df
