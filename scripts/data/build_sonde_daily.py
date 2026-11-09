#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Build per-site daily SENTINEL-Sonde tables from raw USGS DV batches.

Reads every batch directory under $SENTINEL_DATA/raw/usgs_dv/ (``batches`` = min/mean/max,
``batches_median`` = daily median), groups rows by site, applies the QC/splicing rules in
sentinel/data/sonde_daily.py and writes:
  processed/sonde_daily/sites/<site_no>.parquet  (wide, one row per date)
  processed/sonde_daily/index.parquet            (per-site/variable coverage)
  processed/sonde_daily/splice_log.parquet, qc_counts.json, manifest 'sonde_daily'

    python scripts/data/build_sonde_daily.py --rebuild     # full rebuild (deterministic)

Approval-status refresh (2026-11-08): re-pulled batches for a date window take precedence over
the original pull inside that window, per (parm, stat, ts_id) present in the refresh; rows outside
the window are untouched (train/val blocks end 2023-09-30, so a WY2024+ refresh cannot change them).
Built into a NEW output directory so that readers of the current tables are never disturbed:

    python scripts/data/build_sonde_daily.py --rebuild --out-name sonde_daily_refresh_20261108 \
        --refresh-dirs batches_refresh_20261108,batches_median_refresh_20261108 \
        --refresh-window 2023-10-01,2026-11-08
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from sentinel.core import paths
from sentinel.core.manifest import write_manifest
from sentinel.data.sonde_daily import clean_long, splice_site

OUT = paths.processed("sonde_daily")
SITES = OUT / "sites"
BATCH_DIRS = ("batches", "batches_median")


def site_files(dirs=BATCH_DIRS) -> dict[str, list[str]]:
    m = defaultdict(list)
    for d in dirs:
        for f in sorted((paths.RAW / "usgs_dv" / d).glob("batch_*.parquet")):
            for s in pd.read_parquet(f, columns=["site_no"])["site_no"].astype(str).unique():
                m[s].append(str(f))
    return dict(m)


KEY = ["parm_cd", "stat_cd", "ts_id"]


def apply_refresh(base: pd.DataFrame, ref: pd.DataFrame, lo, hi) -> pd.DataFrame:
    """Refresh rows replace base rows dated inside [lo, hi] for every (parm, stat, ts_id) the
    refresh contains; base rows outside the window or for series absent from the refresh stay."""
    if ref.empty:
        return base
    d = pd.to_datetime(base["date"])
    inwin = (d >= lo) & (d <= hi)
    keys = pd.MultiIndex.from_frame(ref[KEY].astype(str).drop_duplicates())
    hit = pd.MultiIndex.from_frame(base[KEY].astype(str)).isin(keys)
    out = pd.concat([base[~(inwin & hit)], ref], ignore_index=True)
    return out.drop_duplicates(subset=["site_no", *KEY, "date"], keep="last")


def process_site(args) -> dict:
    site, files = args[0], args[1]
    ref_files, win = (args[2], args[3]) if len(args) > 2 else ([], None)
    keep_dir = args[4] if len(args) > 4 else None
    df = pd.concat([pd.read_parquet(f, filters=[("site_no", "==", site)]) for f in files],
                   ignore_index=True) if files else pd.DataFrame()
    if ref_files:
        ref = pd.concat([pd.read_parquet(f, filters=[("site_no", "==", site)]) for f in ref_files],
                        ignore_index=True)
        ref = ref[(pd.to_datetime(ref["date"]) >= win[0]) & (pd.to_datetime(ref["date"]) <= win[1])]
        df = apply_refresh(df, ref, *win) if len(df) else ref
    if df.empty:
        return {"site": site, "counts": {}, "index": [], "splices": []}
    df["site_no"] = df["site_no"].astype(str)
    clean, counts = clean_long(df)
    wide, log = splice_site(clean)
    if keep_dir is not None and (keep_dir / f"{site}.parquet").is_file():
        # rows dated before the refresh window are taken verbatim from the calibration snapshot: the
        # splice primary-series choice depends on series length, which a refresh can change
        old = pd.read_parquet(keep_dir / f"{site}.parquet")
        old = old[old.index < win[0]]
        cols = list(dict.fromkeys(list(wide.columns) + list(old.columns)))
        wide = pd.concat([old.reindex(columns=cols), wide[wide.index >= win[0]].reindex(columns=cols)]).sort_index()
        for c in cols:
            if c.endswith("_prov"):
                wide[c] = wide[c].astype("boolean").fillna(False).astype(bool)
            elif c.endswith("_ts"):
                wide[c] = wide[c].astype(object)
            else:
                wide[c] = wide[c].astype("float32")
        wide = wide.dropna(how="all", subset=[c for c in cols if not c.endswith(("_prov", "_ts"))])
    rows = []
    if not wide.empty:
        wide.to_parquet(SITES / f"{site}.parquet", compression="zstd")
        for c in wide.columns:
            if c.endswith("_prov") or c.endswith("_ts"):
                continue
            s = wide[c].dropna()
            if s.empty:
                continue
            rows.append({"site_no": site, "var": c, "n_days": int(s.size), "first": s.index.min(),
                         "last": s.index.max(),
                         "frac_provisional": float(wide.loc[s.index, f"{c}_prov"].mean())})
    return {"site": site, "counts": counts, "index": rows,
            "splices": [{"site_no": site, **r} for r in log]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--rebuild", action="store_true", help="delete and rebuild all site tables")
    ap.add_argument("--out-name", default="sonde_daily")
    ap.add_argument("--refresh-dirs", default="", help="comma list of re-pulled batch dirs (precedence in window)")
    ap.add_argument("--refresh-window", default="", help="lo,hi dates of the refresh pull")
    ap.add_argument("--keep-before-from", default="",
                    help="snapshot whose rows dated before the refresh window are kept verbatim (e.g. sonde_daily)")
    a = ap.parse_args()
    global OUT, SITES
    OUT = paths.processed(a.out_name)
    SITES = OUT / "sites"
    rdirs = [d for d in a.refresh_dirs.split(",") if d]
    if rdirs and len(a.refresh_window.split(",")) != 2:
        raise SystemExit("--refresh-dirs needs --refresh-window lo,hi")
    win = tuple(pd.Timestamp(x) for x in a.refresh_window.split(",")) if rdirs else None
    if a.rebuild and SITES.exists():
        shutil.rmtree(SITES)
    SITES.mkdir(parents=True, exist_ok=True)
    mapping = site_files()
    rmap = site_files(rdirs) if rdirs else {}
    keep = (paths.processed(a.keep_before_from) / "sites") if a.keep_before_from else None
    if keep is not None and keep.resolve() == SITES.resolve():
        raise SystemExit("--keep-before-from must differ from the output snapshot")
    jobs = [(s, mapping.get(s, []), rmap.get(s, []), win, keep) if rdirs else (s, mapping[s])
            for s in sorted(set(mapping) | set(rmap))]
    print(f"{len(jobs)} sites across {sum(len(v) for v in mapping.values())} site-file pairs"
          f"{f'; refresh {len(rmap)} sites from {rdirs} window {a.refresh_window}' if rdirs else ''}")
    idx_rows, spl_rows, totals = [], [], {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for res in ex.map(process_site, jobs, chunksize=8):
            idx_rows += res["index"]
            spl_rows += res["splices"]
            for k, v in res["counts"].items():
                totals[k] = totals.get(k, 0) + v
    idx = pd.DataFrame(idx_rows)
    idx.to_parquet(OUT / "index.parquet", index=False)
    pd.DataFrame(spl_rows).to_parquet(OUT / "splice_log.parquet", index=False)
    (OUT / "qc_counts.json").write_text(json.dumps(totals, indent=2))
    for stale in ("batches_done.json",):
        (OUT / stale).unlink(missing_ok=True)
    print("qc counts:", totals)
    files = sorted(SITES.glob("*.parquet")) + [OUT / "index.parquet", OUT / "splice_log.parquet",
                                              OUT / "qc_counts.json"]
    write_manifest(a.out_name, files=files,
                   sources=["manifest:usgs_dv_raw", "manifest:usgs_dv_raw_batches_median"]
                   + [f"manifest:usgs_dv_raw_{d}" for d in rdirs],
                   description="SENTINEL-Sonde daily tier: per-site wide daily min/mean/max/median tables "
                               "built from USGS DV with qualifier/plausibility masking and traceable "
                               "series splicing (sentinel/data/sonde_daily.py).",
                   license="US public domain", hash_files=False,
                   stats={"n_sites": int(idx["site_no"].nunique()), "qc_counts": totals,
                          "n_site_vars": int(len(idx))})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
