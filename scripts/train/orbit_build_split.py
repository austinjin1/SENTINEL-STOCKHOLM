#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""SENTINEL-Orbit benchmark build (orbit_protocol.md §2–§3).

1. ``--freeze-rule``: write the frozen split RULE file docs/preregistration/orbit_split.json (refuses to
   overwrite) and print its sha256. Done once, before any model selection.
2. Default: assemble the matchup table (AquaCorr-Global L2A chips + WQP matchups), image QC, de-duplication,
   split assignment; protected rows (test role / test years) are dropped before any label use; extract the
   per-chip water-pixel cache for development rows -> $SENTINEL_RUNS/orbit/cache/dev_v1/; write the split
   report docs/preregistration/orbit_split_report_<version>.json (counts only for protected cells).

    python -B scripts/train/orbit_build_split.py --freeze-rule
    python -B scripts/train/orbit_build_split.py --workers 6
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "6")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths  # noqa: E402
from sentinel.core.manifest import write_manifest  # noqa: E402
from sentinel.models.orbit import data as D  # noqa: E402
from sentinel.models.orbit import split as SP  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze-rule", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--version", default=D.CACHE_VERSION)
    a = ap.parse_args()
    if a.freeze_rule:
        sha = SP.write_split_file(extra={"created": "2026-10-24", "note": "split v3: supersedes v2 (and v1) after the M2 "
                                         "pre-training review (C1, N1), before any model selection; roles are a pure "
                                         "function of (lon, lat, date) + HydroLAKES"})
        print(f"wrote {SP.SPLIT_FILE} sha256 {sha}")
        return 0
    ver = SP.verify_split_file()
    print("split file", ver)
    df, info = D.build_table(log=print)
    info["split_file"] = ver
    info["qc_rules"] = D.QC_RULES
    info["valid_ranges"] = D.VALID
    pix = D.extract_pixels(df, workers=a.workers, log=print)
    info["pixels"] = {"p_max": D.P_MAX, "win": D.WIN, "rows_with_ge1_pixel": int((pix["n_pix"] > 0).sum()),
                      "median_n_pix": float(pd_median(pix["n_pix"]))}
    info["buffer_excluded_rows"] = int(df.buffer_excluded.sum())
    d = D.write_cache(df, pix, info, a.version)
    rep = {k: info[k] for k in ("buffer_excluded_rows", "cells_all_rows", "split_distance_km", "targets_by_cell", "n_rows_final",
                                "image_qc_pass", "n_rows_with_chip", "by_source", "access", "split_file",
                                "protected_cell_counts")}
    rp = paths.REPO_ROOT / "docs" / "preregistration" / f"orbit_split_report_{a.version}.json"
    rp.write_text(json.dumps(rep, indent=1, default=str) + "\n")
    write_manifest(f"orbit_{a.version}", files=[d / "table.parquet", d / "pixels.npz", d / "info.json"],
                   sources=["manifest:s2_chips_aquacorr"] + [f"manifest:{n}" for n in D.WQP_MATCHUP_SETS
                                                                if (paths.PROCESSED / n / "samples.parquet").exists()] + ["manifest:hydrolakes",
                            "manifest:aquacorr_global"],
                   description="SENTINEL-Orbit development matchup table (train/val roles x train/val years only; "
                               "protected rows never written) + per-chip water-pixel cache",
                   license="derived: AquaCorr CC-BY-4.0, WQP public, Copernicus Sentinel data, JRC GSW, HydroLAKES CC-BY-4.0",
                   synthetic=False, stats=json.loads(json.dumps(rep, default=str)))
    print(json.dumps(rep, indent=1, default=str)[:4000])
    return 0


def pd_median(x):
    import numpy as np

    return float(np.median(x)) if len(x) else float("nan")


if __name__ == "__main__":
    raise SystemExit(main())
