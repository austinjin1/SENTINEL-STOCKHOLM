#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Download USGS daily values (min/mean/max) for every site with continuous WQ or temperature data.

Step 1 builds a per-state DV series catalog; step 2 downloads 10-site batches
(resume-safe); step 3 writes a manifest. Raw output: $SENTINEL_DATA/raw/usgs_dv/.

    python scripts/data/download_usgs_dv.py --start 2000-10-01 --end 2026-10-20
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths  # noqa: E402
from sentinel.core.manifest import write_manifest  # noqa: E402
from sentinel.data.sources import usgs_legacy as U  # noqa: E402

log = logging.getLogger("usgs_dv")


def build_catalog(out: Path) -> pd.DataFrame:
    cat_path = out / "dv_series_catalog.parquet"
    if cat_path.exists():
        return pd.read_parquet(cat_path)
    frames = []
    for st in U.STATES:
        try:
            df = U.site_series_catalog(st, list(U.WQ_PARAMS), "dv")
        except RuntimeError as e:
            log.warning("catalog %s failed: %s", st, e)
            continue
        log.info("catalog %s: %d series rows", st, len(df))
        if not df.empty:
            frames.append(df)
        time.sleep(1.0)
    cat = pd.concat(frames, ignore_index=True)
    cat.to_parquet(cat_path, index=False)
    return cat


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2000-10-01")
    ap.add_argument("--end", default="2026-10-20")
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--sleep", type=float, default=1.0)
    ap.add_argument("--stats", default="00001,00002,00003",
                    help="DV statistic codes; the median pass uses --stats 00008 --batch-dir batches_median")
    ap.add_argument("--batch-dir", default="batches")
    a = ap.parse_args()
    out = paths.raw("usgs_dv")
    bdir = out / a.batch_dir
    bdir.mkdir(exist_ok=True)
    stats = a.stats.split(",")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(out / "download.log"), logging.StreamHandler()])
    cat = build_catalog(out)
    cat = cat[(cat["data_type_cd"] == "dv") & cat["parm_cd"].isin(U.WQ_PARAMS)]
    if stats != ["00001", "00002", "00003"]:
        cat = cat[cat["stat_cd"].isin(stats)]
    sites = sorted(cat["site_no"].unique())
    log.info("%d sites with DV water-quality series", len(sites))
    params = list(U.ALL_PARAMS)
    n_rows = 0
    for b in range(0, len(sites), a.batch):
        chunk = sites[b:b + a.batch]
        dest = bdir / f"batch_{b // a.batch:05d}.parquet"
        if dest.exists():
            continue
        try:
            df = U.get_dv(chunk, params, a.start, a.end, stat_codes=stats)
        except RuntimeError as e:
            log.error("batch %d failed: %s", b // a.batch, e)
            continue
        if df.empty:
            df = pd.DataFrame(columns=["site_no", "parm_cd", "stat_cd", "ts_id", "value",
                                       "qualifier", "date"])
        df.to_parquet(dest, index=False, compression="zstd")
        n_rows += len(df)
        log.info("batch %d/%d: %d rows (total %d)", b // a.batch + 1,
                 (len(sites) + a.batch - 1) // a.batch, len(df), n_rows)
        time.sleep(a.sleep)
    files = sorted(bdir.glob("*.parquet")) + [out / "dv_series_catalog.parquet"]
    write_manifest("usgs_dv_raw" if a.batch_dir == "batches" else f"usgs_dv_raw_{a.batch_dir}", files=files, sources=[U.SITE_URL, U.DV_URL],
                   description=f"USGS NWIS daily values (min/mean/max) {a.start}..{a.end} for sites "
                               "with DV water-quality series; missing codes masked to NaN.",
                   license="US public domain", stats={"n_sites": len(sites)})
    log.info("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
