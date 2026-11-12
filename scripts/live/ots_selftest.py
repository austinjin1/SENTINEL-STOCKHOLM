#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""ONE-SHOT OpenTimestamps approval test (review C-D2). DO NOT RUN without the project owner's approval of external
hash-only submission (the lead records the approval). Run in the ``sentinel2-live`` environment.

What it does:
  1. writes a test file holding only the text "SENTINEL-v2 EW live trial OTS self-test <UTC time>" (no data) under
     ``dryrun/ots_selftest/``;
  2. ``ots stamp`` it with the same calendars as ``stamp.py`` (one hash-only submission per calendar);
  3. polls ``ots_verify.proof_time`` every 30 min for up to 12 h, until a Bitcoin attestation is verified against both
     explorers;
  4. also runs the official ``ots verify`` for comparison and prints both block times.
Success criterion (for the freeze): a verified block time within 12 h, equal from both paths.

    python -B scripts/live/ots_selftest.py --i-have-owner-approval
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import subprocess
import sys
import time

import live_common as LC
import ots_verify as OV
import stamp as ST


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--i-have-owner-approval", action="store_true", required=True)
    ap.add_argument("--max-hours", type=float, default=12.0)
    a = ap.parse_args()
    print(json.dumps(OV.self_test()))
    d = LC.DRYRUN_ROOT / "ots_selftest"
    d.mkdir(parents=True, exist_ok=True)
    now = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    f = d / f"selftest_{now.replace(':', '')}.txt"
    f.write_text(f"SENTINEL-v2 EW live trial OTS self-test {now}\n")
    cmd = [str(ST.OTS_EXE), "stamp", "-m", "2"] + sum((["-c", c] for c in ST.CALENDARS), []) + [str(f)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=False)
    print("stamp rc", r.returncode, (r.stderr or "")[-300:])
    proof = f.with_name(f.name + ".ots")
    t0 = time.time()
    while time.time() - t0 < a.max_hours * 3600:
        t = OV.proof_time(f, proof)
        if t is not None:
            v = subprocess.run([str(ST.OTS_EXE), "verify", str(proof)], capture_output=True, text=True, timeout=600,
                               check=False, env={"TZ": "UTC", "PATH": "/usr/bin:/bin"})
            print(json.dumps({"verified_block_time_utc": str(t), "ots_verify_output": (v.stdout + v.stderr)[-400:],
                              "elapsed_h": round((time.time() - t0) / 3600, 2)}))
            return 0
        time.sleep(1800)
    print("no Bitcoin attestation within the time limit")
    return 1


if __name__ == "__main__":
    sys.exit(main())
