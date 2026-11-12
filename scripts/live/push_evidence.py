#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Push the hash log + OTS proofs right after stamping (review C-E2 recommendation). DORMANT until enabled.

Runs (from ``run_daily.sh``, owner's cron) only if the lead-written ``FREEZE_ew_live.json`` contains
``"auto_push": {"enabled": true, "remote": "<name>", "branch": "<branch>"}`` (set at the freeze, after the owner
approves push access). It stages ONLY ``results/live/ew_v2_live_hashlog.jsonl`` and ``results/live/timestamps/``,
commits with a fixed message and pushes to that remote/branch. Agents never run it. Dry runs never push.

    python -B scripts/live/push_evidence.py --mode live
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import subprocess
import sys

import live_common as LC

PATHS = ("results/live/ew_v2_live_hashlog.jsonl", "results/live/timestamps")


def git(*args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(LC.REPO), *args], capture_output=True, text=True, timeout=300, check=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("live", "dryrun"), default="live")
    a = ap.parse_args()
    if a.mode != "live" or not LC.FREEZE_LIVE.is_file():
        print("auto-push disabled (dry run, or no freeze record): nothing pushed")
        return 0
    cfg = (json.loads(LC.FREEZE_LIVE.read_text()).get("auto_push") or {})
    if not cfg.get("enabled"):
        print("auto-push not enabled in FREEZE_ew_live.json: nothing pushed")
        return 0
    existing = [p for p in PATHS if (LC.REPO / p).exists()]
    if not existing:
        return 0
    git("add", "--", *existing)
    if git("diff", "--cached", "--quiet").returncode == 0:
        print("nothing new to push")
        return 0
    msg = f"EW-v2 live: hash log + OTS proofs {_dt.datetime.now(_dt.timezone.utc).date()}"
    c = git("commit", "-m", msg, "--", *existing)
    p = git("push", cfg.get("remote", "origin"), f"HEAD:{cfg.get('branch', 'v2-rebuild')}")
    print(f"commit rc={c.returncode} push rc={p.returncode}")
    return 0 if p.returncode == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
