#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Independent same-day timestamps for hash-log rows: OpenTimestamps ONLY (review C-L1 / C-D1). Hash-only submission.

For each hash-log row without evidence: write the stamped file ``<seq>_<sha12>.row.json`` (= the canonical row bytes,
whose SHA-256 is ``row_sha256``) and run the OFFICIAL client ``ots stamp`` (opentimestamps-client, dedicated env
``sentinel2-live``) against three public calendars. The client adds a random nonce and submits only a hash; no
forecast, site or value leaves the host. The resulting ``.ots`` proof is upgraded to a Bitcoin attestation at scoring
time (``ots_verify.py``, outside the issuing closure).

Deletion protection (C-D1): every evidence file (row file, ``.ots``, ``.evidence.json``) is listed with its SHA-256 in
the NEXT hash-log row (``evidence_refs``; ``pending_evidence_refs``), which is itself chained and stamped. A referenced
evidence file that is later missing or altered makes the scorer refuse the look.

ACTIVATION: only if the lead-written ``FREEZE_ew_live.json`` lists ``"ots"`` in ``timestamp_backends`` (set in the
freeze record from the start, after the owner's approval; the record is never edited afterwards). Dry runs never
submit. The one-shot approval test is ``ots_selftest.py`` (not run without approval).

    python -B scripts/live/stamp.py --mode live      # stamps every hash-log row that has no evidence yet
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import subprocess
import sys
from pathlib import Path

import live_common as LC

CALENDARS = ("https://a.pool.opentimestamps.org", "https://b.pool.opentimestamps.org",
             "https://a.pool.eternitywall.com")
OTS_EXE = Path("/home/bcheng/miniconda3/envs/sentinel2-live/bin/ots")


def evidence_dir(mode: str) -> Path:
    return LC.TIMESTAMP_DIR if mode == "live" else LC.DRYRUN_ROOT / "timestamps"


def row_bytes(row: dict) -> bytes:
    return LC.canonical({k: v for k, v in row.items() if k != "row_sha256"})


def pending_evidence_refs(mode: str) -> dict:
    """{file name: sha256} of every evidence file not yet referenced by a hash-log row (goes into the next row)."""
    d = evidence_dir(mode)
    if not d.is_dir():
        return {}
    path = LC.HASHLOG if mode == "live" else LC.DRYRUN_ROOT / "dryrun_hashlog.jsonl"
    referenced = set()
    for r in LC.read_hashlog(path):
        referenced |= set((r.get("evidence_refs") or {}))
    return {f.name: LC.sha256_file(f) for f in sorted(d.iterdir()) if f.is_file() and f.name not in referenced}


def stamp_row(row: dict, out: Path) -> dict:
    base = f"{row['seq']:05d}_{row['row_sha256'][:12]}"
    rf = out / f"{base}.row.json"
    rf.write_bytes(row_bytes(row))
    if LC.sha256_file(rf) != row["row_sha256"]:
        raise ValueError("row hash does not match its canonical bytes")
    cmd = [str(OTS_EXE), "stamp", "-m", "2"] + sum((["-c", c] for c in CALENDARS), []) + [str(rf)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=False)
    proof = rf.with_name(rf.name + ".ots")
    rec = {"backend": "ots", "row_sha256": row["row_sha256"], "seq": row["seq"], "issue_date": row["issue_date"],
           "submitted_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(), "returncode": r.returncode,
           "calendars": list(CALENDARS), "row_file": rf.name}
    if r.returncode == 0 and proof.is_file():
        rec.update({"ots_file": proof.name, "ots_sha256": LC.sha256_file(proof)})
    else:
        rec["error"] = (r.stderr or r.stdout)[-200:]
    return rec


def load_evidence(mode: str) -> dict:
    d = evidence_dir(mode)
    out = {}
    if d.is_dir():
        for f in sorted(d.glob("*.evidence.json")):
            rec = json.loads(f.read_text())
            out.setdefault(rec["row_sha256"], []).append(rec)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("live", "dryrun"), default="live")
    a = ap.parse_args()
    if a.mode != "live" or not LC.FREEZE_LIVE.is_file():
        print("timestamping disabled (dry run, or no lead-written FREEZE_ew_live.json): nothing submitted")
        return 0
    fr = json.loads(LC.FREEZE_LIVE.read_text())
    if "ots" not in (fr.get("timestamp_backends") or []):
        print("OpenTimestamps not approved in FREEZE_ew_live.json: nothing submitted")
        return 0
    if not OTS_EXE.is_file():
        print("ots client missing in the sentinel2-live environment: nothing submitted (day stays unstamped)")
        return 1
    out = evidence_dir(a.mode)
    out.mkdir(parents=True, exist_ok=True)
    have = load_evidence(a.mode)
    n = 0
    for row in LC.read_hashlog(LC.HASHLOG):
        if row["row_sha256"] in have:
            continue
        try:
            rec = stamp_row(row, out)
        except Exception as e:  # noqa: BLE001
            rec = {"backend": "ots", "row_sha256": row["row_sha256"], "seq": row["seq"], "error": e.__class__.__name__}
        LC.excl_write(out / f"{row['seq']:05d}_{row['row_sha256'][:12]}.evidence.json", rec)
        n += 1
    print(f"stamped {n} row(s) with OpenTimestamps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
