#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Verify the EW-v2 live hash log OFFLINE (review C-L1/C-D1/W-D2; no git dates, no network).

Checks: chain integrity; SHA-256 of every issued file; every evidence file referenced by a later row
(``evidence_refs``) exists unchanged; scheduled days with NO row (missing) or only ``failed`` rows; orphan issue
directories and stale temp directories; issued rows without a local OTS proof. Primary eligibility (third-party OTS
verification) is computed ONLY by the scorer (``ots_verify.py``, outside the issuing closure).

    python -B scripts/live/verify_hashlog.py [--dryrun]
"""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

import live_common as LC
import stamp as ST


def check_evidence_refs(rows, mode) -> list[str]:
    """Every evidence file referenced by a row (``evidence_refs``) must exist unchanged (review C-D1)."""
    d = ST.evidence_dir(mode)
    errs = []
    for r in rows:
        for name, sha in (r.get("evidence_refs") or {}).items():
            f = d / name
            if not f.is_file():
                errs.append(f"row {r['seq']}: referenced evidence {name} missing")
            elif LC.sha256_file(f) != sha:
                errs.append(f"row {r['seq']}: referenced evidence {name} altered")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dryrun", action="store_true")
    ap.add_argument("--no-ots", action="store_true", help="kept for compatibility; this script never goes online")
    a = ap.parse_args()
    mode = "dryrun" if a.dryrun else "live"
    path = LC.DRYRUN_ROOT / "dryrun_hashlog.jsonl" if a.dryrun else LC.HASHLOG
    root = LC.DRYRUN_ROOT if a.dryrun else LC.LIVE_ROOT
    rows = LC.read_hashlog(path)
    errs = LC.verify_chain(rows)
    issued = {}
    for r in rows:
        if r.get("status", "issued") == "issued":
            issued[r["issue_date"]] = r
            idir = root / "issues" / r["issue_date"]
            for name, sha in r["files"].items():
                f = idir / name
                if not f.is_file():
                    errs.append(f"{r['issue_date']}: {name} missing")
                elif LC.sha256_file(f) != sha:
                    errs.append(f"{r['issue_date']}: {name} sha mismatch")
    dirs = sorted(p.name for p in (root / "issues").glob("*")) if (root / "issues").is_dir() else []
    orphans = [d for d in dirs if not d.startswith(".tmp_") and d not in issued]
    temps = [d for d in dirs if d.startswith(".tmp_")]
    res = {"log": str(path), "n_rows": len(rows), "n_issued": len(issued),
           "n_failed_rows": sum(r.get("status") == "failed" for r in rows),
           "orphan_issue_dirs": orphans, "stale_temp_dirs": temps, "errors": errs}
    if rows:
        try:
            cfg, _ = LC.load_config()
            start = pd.Timestamp(cfg["trial"]["issue_start"]) if not a.dryrun else pd.Timestamp(rows[0]["issue_date"])
            end = min(pd.Timestamp.now(tz="UTC").tz_localize(None).normalize(), pd.Timestamp(cfg["trial"]["issue_end"]))
            days = [d.date().isoformat() for d in pd.date_range(start, end)]
            have = {r["issue_date"] for r in rows}
            res["missing_days_no_row"] = [d for d in days if d not in have]
            res["failed_only_days"] = sorted({r["issue_date"] for r in rows} - set(issued))
        except FileNotFoundError:
            pass
    errs += check_evidence_refs(rows, mode)
    ev = ST.load_evidence(mode)
    res["n_issued_without_local_proof"] = sum(1 for r in issued.values() if not any(
        e.get("ots_file") for e in ev.get(r["row_sha256"], [])))
    res["eligibility"] = "computed only by the scorer (ots_verify.py, third-party verification)"
    print(json.dumps(res, indent=1, default=str))
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
