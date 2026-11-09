#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""M5-X pre-registration checks and snapshot (no model is fitted; no target value is used).

1. verifies the frozen split table's sha256;
2. builds the frozen-test exclusion set of every workstream touching NRSA / NLA / UK EA (item-7 ruling);
3. validates the split table and every fold's training pool (no test id, no frozen-test row, nothing within
   1 km of a fold test location) and writes per-fold pool sizes;
4. with ``--snapshot``: logs the registry ordering-evidence row ``bioassess_x/prereg_snapshot`` (sha256 + mtime
   of the protocol, code, tests, split, power and pool files).

    python -B scripts/eval/bioassess_x_prereg.py [--snapshot]
"""
from __future__ import annotations

import argparse
import collections
import json
import time
from pathlib import Path

import pandas as pd

from sentinel.core import paths
from sentinel.core.registry import file_sha256, log_result
from sentinel.models.bioassess import xtrack as X

SPLIT_SHA256 = "18a35ef3ba9ddd87a5be2e8c4ff1a18d29bea8869318f940b150b1306feee459"
XDIR = paths.RUNS_ROOT / "bioassess" / "x_track"
REPO = paths.REPO_ROOT


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", action="store_true")
    a = ap.parse_args()
    split = XDIR / "splits_x_v1.parquet"
    if file_sha256(split) != SPLIT_SHA256:
        raise SystemExit("split table changed since pre-registration")
    S = pd.read_parquet(split)
    frozen = X.frozen_test_keys()
    reasons = collections.Counter(frozen.values())
    bad = X.validate_splits(S)
    bad2, rep = X.validate_training_pools(S, frozen)
    R = pd.DataFrame(rep)
    summary = {
        "split_sha256": SPLIT_SHA256, "frozen_split_files": {k: file_sha256(REPO / v) for k, v in X.FROZEN_SPLIT_FILES.items()},
        "frozen_rows_by_reason": dict(reasons), "validate_splits": bad, "validate_training_pools": bad2,
        "per_fold": rep,
        "pool_ranges": {b: {c: [int(R[R.block == b][c].min()), int(R[R.block == b][c].max())]
                            for c in ("final", "removed_frozen", "removed_buffer", "n_nla", "n_nrsa", "n_uk")}
                        for b in ("X-L", "X-U")},
    }
    out = XDIR / "pools_x_v1.json"
    out.write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({k: summary[k] for k in ("frozen_rows_by_reason", "validate_splits", "validate_training_pools",
                                              "pool_ranges")}, indent=1, default=str))
    if bad or bad2:
        raise SystemExit("validation failed")
    if a.snapshot:
        files = [REPO / "docs/preregistration/bioassess_x_protocol.md", REPO / "sentinel/models/bioassess/xtrack.py",
                 REPO / "scripts/eval/bioassess_x_power.py", REPO / "scripts/eval/bioassess_x_prereg.py",
                 REPO / "tests/test_bioassess_models.py", split, XDIR / "splits_x_v1.json", XDIR / "power_x_v1.json", out]
        files += [REPO / v for v in X.FROZEN_SPLIT_FILES.values()]
        snap = {str(f): {"sha256": file_sha256(f), "mtime": time.strftime("%Y-%m-%dT%H:%M:%S",
                                                                          time.localtime(f.stat().st_mtime))} for f in files}
        r = log_result(workstream="bioassess", experiment="bioassess_x/prereg_snapshot", model="prereg_snapshot",
                       metric="n_files", value=float(len(snap)), split="prereg", n=len(snap), supersedes="47130a305f69",
                       notes="M5-X pre-registration v3 (re-review C-MXR-1/2, M2 ruling, W-MXR-2/3 addressed); "
                             "no track-X fit or selection yet",
                       extra={"files": snap, "snapshot_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                              "split_content_sha256": "59ab19bfcf0b3d4d4d7c6eb977d4d31c8620da269630d8d82923ae39d52cb3fa"})
        print("snapshot", r.get("id") if isinstance(r, dict) else r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
