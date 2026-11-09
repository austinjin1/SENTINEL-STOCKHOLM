#!/usr/bin/env python
"""Parallel driver for build_usgs_npz.build_one (network I/O bound).

Uses a thread pool (default 8 workers) so the many-year iv fetches overlap.
Writes _build_results.json incrementally. Window kept to ~3 yr for speed while
retaining ample runway for episodes and broad clean records for controls.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_usgs_npz as B  # noqa: E402

OUTDIR = B.OUTDIR


def main():
    cand_csv = sys.argv[1]
    start = sys.argv[2] if len(sys.argv) > 2 else "2022-06-01"
    end = sys.argv[3] if len(sys.argv) > 3 else "2025-06-30"
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    cands = pd.read_csv(cand_csv, dtype=str)

    def work(row):
        site = str(row["site_no"]).strip()
        station = str(row.get("station_nm", ""))
        try:
            return B.build_one(site, station, start, end)
        except Exception as e:
            return {"site": site, "station": station, "decision": "ERROR",
                    "reason": f"{type(e).__name__}: {str(e)[:120]}"}

    results = []
    rows = [r for _, r in cands.iterrows()]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(work, r): str(r["site_no"]).strip() for r in rows}
        for i, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            results.append(rec)
            print(f"[{i}/{len(rows)}] {rec['site']} -> {rec['decision']} "
                  f"{rec.get('reason','')} doMin={rec.get('do_clean_min')} "
                  f"onset={rec.get('onset_utc')} runway={rec.get('runway_days')}", flush=True)
            (OUTDIR / "_build_results.json").write_text(
                json.dumps(results, indent=2, default=str))

    ne = sum(1 for x in results if x["decision"] == "EPISODE")
    nc = sum(1 for x in results if x["decision"] == "CONTROL")
    nd = sum(1 for x in results if x["decision"] == "DISCARD")
    nerr = sum(1 for x in results if x["decision"] == "ERROR")
    print(f"\nDONE. EPISODE={ne} CONTROL={nc} DISCARD={nd} ERROR={nerr} of {len(results)}")


if __name__ == "__main__":
    main()
