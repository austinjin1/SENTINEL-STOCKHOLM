#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Status row for a scheduled issue day (review C-L1). Called by ``run_daily.sh`` on EXIT, whatever happened.

If the hash log already holds an ``issued`` row for the day, nothing is written. Otherwise a chained ``failed`` row
records the failing stage, the exit code, the exception class (last ``Error:`` line of the stage log; no message
text, no data) and whether an orphan issue directory exists (crash after the issue was written but before its row).

    python -B scripts/live/log_status.py --mode live --date 2026-12-01 --stage issue --rc 1 --stage-log <file>
"""

from __future__ import annotations

import argparse
import datetime as _dt
import re
import sys
from pathlib import Path

import live_common as LC
import stamp as ST


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("live", "dryrun"), default="live")
    ap.add_argument("--date", required=True)
    ap.add_argument("--stage", default="unknown")
    ap.add_argument("--rc", type=int, default=0)
    ap.add_argument("--stage-log", default=None)
    a = ap.parse_args()
    path = LC.HASHLOG if a.mode == "live" else LC.DRYRUN_ROOT / "dryrun_hashlog.jsonl"
    root = LC.LIVE_ROOT if a.mode == "live" else LC.DRYRUN_ROOT
    rows = LC.read_hashlog(path)
    if any(r["issue_date"] == a.date and r.get("status") == "issued" for r in rows):
        return 0
    err = None
    if a.stage_log and Path(a.stage_log).is_file():
        lines = [ln for ln in Path(a.stage_log).read_text(errors="replace").splitlines()
                 if re.match(r"^[A-Za-z_][\w.]*(Error|Exception|Exit)\b", ln)]
        err = lines[-1].split(":")[0][:80] if lines else None
        if err is None and a.rc != 0:     # SystemExit("msg") prints only the message (a code message, never data)
            tail = [ln for ln in Path(a.stage_log).read_text(errors="replace").splitlines() if ln.strip()]
            err = "SystemExit: " + tail[-1][:80] if tail else "SystemExit"
    try:
        cfg_sha = LC.sha256_file(LC.CONFIG)
    except FileNotFoundError:
        cfg_sha = None
    orphan = (root / "issues" / a.date / "manifest.json").exists()
    LC.append_hashlog(path, {"issue_date": a.date, "status": "failed", "mode": a.mode, "stage": a.stage, "rc": a.rc,
                             "error_class": err, "orphan_issue_dir": orphan,
                             "logged_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                             "files": {}, "code_sha256": LC.code_sha256(), "config_sha256": cfg_sha,
                             "freeze_sha256": LC.freeze_sha(), "env_lock_sha256": LC.env_lock_sha256(),
                             "evidence_refs": ST.pending_evidence_refs(a.mode)})
    print(f"failed-status row written for {a.date} (stage {a.stage}, rc {a.rc}, {err})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
