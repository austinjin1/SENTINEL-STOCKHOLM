#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Build hourly SENTINEL-Sonde tables from raw 15-min USGS IV files.

Processes only sites whose IV download is complete (every expected site-year has a .parquet
or .empty marker), so it can run while the downloader is still working. Output:
$SENTINEL_DATA/processed/sonde_hourly/sites/<site>.parquet, splice_log.parquet.

    python scripts/data/build_sonde_hourly.py [--finalize]

Approval-status refresh (2026-11-08): with --refresh-raw usgs_iv_refresh_20261108 --refresh-start
2023-10-01 --out-name sonde_hourly_refresh_20261108 --keep-before-from sonde_hourly, raw 15-min rows
at or after the refresh start come from the re-pull for every (parm, ts_id) it contains, and hourly
rows before the refresh start are copied verbatim from the calibration snapshot.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from sentinel.core import paths
from sentinel.core.manifest import write_manifest
from sentinel.data.sonde_hourly import clean_iv, hourly_site
from sentinel.data.sources.usgs_legacy import WQ_PARAMS

RAW = paths.RAW / "usgs_iv"
OUT = paths.processed("sonde_hourly")
SITES = OUT / "sites"


def site_complete(site_dir, years) -> bool:
    return all((site_dir / f"{y}.parquet").exists() or (site_dir / f"{y}.empty").exists() for y in years)


REFRESH = None   # (raw dir, start Timestamp UTC, keep-before snapshot sites dir) — set in main()


def build(site: str) -> list[dict]:
    files = sorted((RAW / site).glob("*.parquet"))
    if not files:
        return []
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if REFRESH is not None:
        rdir, start, keep = REFRESH
        rf = sorted((rdir / site).glob("*.parquet"))
        if rf:
            ref = pd.concat([pd.read_parquet(f) for f in rf], ignore_index=True)
            ref = ref[pd.to_datetime(ref["time_utc"], utc=True) >= start]
            key = ["parm_cd", "ts_id"]
            hit = pd.MultiIndex.from_frame(df[key].astype(str)).isin(
                pd.MultiIndex.from_frame(ref[key].astype(str).drop_duplicates()))
            late = pd.to_datetime(df["time_utc"], utc=True) >= start
            df = pd.concat([df[~(hit & late)], ref], ignore_index=True).drop_duplicates(
                subset=["site_no", "parm_cd", "ts_id", "time_utc"], keep="last")
    wide, log = hourly_site(clean_iv(df))
    if REFRESH is not None and REFRESH[2] is not None and (REFRESH[2] / f"{site}.parquet").is_file() and not wide.empty:
        old = pd.read_parquet(REFRESH[2] / f"{site}.parquet")
        st = REFRESH[1] if wide.index.tz is not None else REFRESH[1].tz_localize(None)
        old = old[old.index < st]
        cols = list(dict.fromkeys(list(wide.columns) + list(old.columns)))
        wide = pd.concat([old.reindex(columns=cols), wide[wide.index >= st].reindex(columns=cols)]).sort_index()
        for c in cols:
            dt = old.dtypes[c] if c in old.columns else wide[c].dtype
            if str(dt) == "bool":
                wide[c] = wide[c].astype("boolean").fillna(False).astype(bool)
            elif str(dt).startswith("int"):
                wide[c] = wide[c].fillna(0).astype(dt)
            else:
                wide[c] = wide[c].astype(dt)
    if not wide.empty:
        wide.to_parquet(SITES / f"{site}.parquet", compression="zstd")
    return [{"site_no": site, **r} for r in log]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--finalize", action="store_true")
    ap.add_argument("--out-name", default="sonde_hourly")
    ap.add_argument("--refresh-raw", default="")
    ap.add_argument("--refresh-start", default="2023-10-01")
    ap.add_argument("--keep-before-from", default="")
    a = ap.parse_args()
    global OUT, SITES, REFRESH
    OUT = paths.processed(a.out_name)
    SITES = OUT / "sites"
    if a.refresh_raw:
        keep = paths.processed(a.keep_before_from) / "sites" if a.keep_before_from else None
        REFRESH = (paths.RAW / a.refresh_raw, pd.Timestamp(a.refresh_start, tz="UTC"), keep)
    SITES.mkdir(parents=True, exist_ok=True)
    cat = pd.read_parquet(paths.RAW / "usgs_dv" / "dv_series_catalog.parquet")
    # Same span rule as scripts/data/download_usgs_iv.py (water-quality DV series only).
    cat = cat[(cat["data_type_cd"] == "dv") & cat["parm_cd"].isin(WQ_PARAMS)]
    span = cat.groupby("site_no").agg(b=("begin_date", "min"), e=("end_date", "max"))
    todo = []
    for site_dir in sorted(p for p in RAW.iterdir() if p.is_dir()):
        site = site_dir.name
        if (SITES / f"{site}.parquet").exists() or site not in span.index:
            continue
        b = max(pd.Timestamp(span.loc[site, "b"]), pd.Timestamp("2007-10-01"))
        e = min(pd.Timestamp(span.loc[site, "e"]), pd.Timestamp("2026-10-20"))
        if site_complete(site_dir, range(b.year, e.year + 1)):
            todo.append(site)
    print(f"{len(todo)} complete sites to build")
    logs = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for rows in ex.map(build, todo):
            logs += rows
    p = OUT / "splice_log.parquet"
    if logs:
        new = pd.DataFrame(logs)
        if p.exists():
            new = pd.concat([pd.read_parquet(p), new], ignore_index=True)
        new.to_parquet(p, index=False)
    if a.finalize:
        files = sorted(SITES.glob("*.parquet"))
        write_manifest(a.out_name, files=files + ([p] if p.exists() else []),
                       sources=["manifest:usgs_iv_raw"] + ([f"manifest:{a.refresh_raw}_raw"] if a.refresh_raw else []), hash_files=False, license="US public domain",
                       description="SENTINEL-Sonde hourly tier (UTC hourly mean/min/max/count per variable) "
                                   "from USGS IV with QC masking and traceable splicing.",
                       stats={"n_sites": len(files)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
