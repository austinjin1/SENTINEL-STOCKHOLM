#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""M5-F French pull through the Hub'eau APIs (protocol ``bioassess_f_protocol.md`` §2.1, §2.3; lead rulings L1-L3).

Step B (biology): ``stations_hydrobio`` by department, then ``indices`` and ``taxons`` by department x support
(13, 10) x calendar year, 2007-01-01 to the day of the first run. Step C (chemistry) is not in this script: it
needs the lock list and the lead's gate-G2 row, and the fetcher refuses it before that.

Everything goes through ``ftrack.LockedFetcher``: stamp guard before every request, one request at a time, at
least 0.5 s apart, every response gzip-compressed into the store of ``ftrack`` and every request in the manifest.
This script parses no record. It prints counts only.

Run with isolated Python (French files are untrusted data, L3), one chunk of at most ``--max-minutes``; a rerun
resumes from the manifest:

    python -I scripts/data/bioassess_f_pull.py --step B --max-minutes 100

Exit codes: 0 step complete; 3 time limit reached (run again); 4 stopped by a gate or an HTTP error (report).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time

from sentinel.core import paths
from sentinel.models.bioassess import ftrack as F


class TimeUp(Exception):
    pass


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["B"], default="B")
    ap.add_argument("--departments", default=None, help="comma-separated subset (probe runs); default: all 96")
    ap.add_argument("--endpoints", default="stations_hydrobio,indices,taxons")
    ap.add_argument("--max-minutes", type=float, default=100.0)
    ap.add_argument("--retries", type=int, default=4)
    args = ap.parse_args()

    row = F.assert_stamped()                                  # gate G0 (L1), before anything else
    ok, free = F.disk_ok()
    log(f"stamp row {row.get('id')}; disk free {free:.1f} GB (floor {F.DISK_FLOOR_GB:.0f} GB); ok={ok}")
    if not ok:
        return 4
    state_f = paths.MANIFESTS / "naiades_f_pull_state.json"
    state = json.loads(state_f.read_text()) if state_f.exists() else {}
    if "download_date" not in state:                          # fixed at the first run; later chunks reuse it
        state["download_date"] = dt.datetime.now(dt.timezone.utc).date().isoformat()
        state_f.parent.mkdir(parents=True, exist_ok=True)
        state_f.write_text(json.dumps(state, indent=1))
    today = dt.date.fromisoformat(state["download_date"])
    deadline = time.monotonic() + 60.0 * args.max_minutes
    sent = {"n": 0}

    def transport(url, headers):
        if time.monotonic() > deadline:
            raise TimeUp
        sent["n"] += 1
        if sent["n"] % 200 == 0:
            log(f"requests sent in this chunk: {sent['n']}")
        return F._http_get(url, headers)

    fx = F.LockedFetcher(transport=transport, today=today, retries=args.retries)
    deps = args.departments.split(",") if args.departments else None
    blocks = F.biology_blocks(today, deps)
    log(f"download date {today}; M5-F data on disk {fx.used_gb():.3f} GB of {F.NEW_DATA_CAP_GB:.0f} GB; "
        f"manifest rows {len(fx.manifest_rows())}")
    try:
        for ep in args.endpoints.split(","):
            unsplit: list = []
            res = fx.run_step(ep, blocks[ep], unsplit=unsplit)
            res["unsplit_blocks"] = [{"params": p, "records": n} for p, n in unsplit]
            log("step done:", json.dumps(res, sort_keys=True))
            if deps is None:
                state.setdefault("steps", {})[ep] = res
                state_f.write_text(json.dumps(state, indent=1))
    except TimeUp:
        log(f"time limit of {args.max_minutes:.0f} min reached after {sent['n']} requests; run again to resume")
        return 3
    except (F.FetchRefused, RuntimeError) as e:
        log(f"STOPPED ({type(e).__name__}): {e}")
        return 4
    log(f"complete; requests in this chunk {sent['n']}; M5-F data on disk {fx.used_gb():.3f} GB; "
        f"peak {fx.peak_bytes / 1e9:.3f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
