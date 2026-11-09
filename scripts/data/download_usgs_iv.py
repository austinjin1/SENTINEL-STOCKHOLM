#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Download USGS instantaneous (15-min) values for the SENTINEL-Sonde core sites.

Uses the legacy NWIS IV service in quarterly windows (it rejects long spans) while it is
still online (decommissioned 2027-02-22; slowdowns from 2026-11-16). Output: one Parquet
per site-year under $SENTINEL_DATA/raw/usgs_iv/<site>/<year>.parquet (resume-safe; an
empty marker file records site-years with no data).

    python scripts/data/download_usgs_iv.py --workers 3
"""

from __future__ import annotations

import argparse
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from sentinel.core import paths
from sentinel.core.manifest import write_manifest
from sentinel.data.sources import usgs_legacy as U

log = logging.getLogger("usgs_iv")
SURFACE_TYPES = {"ST", "ST-TS", "ST-CA", "ST-DCH", "LK", "ES", "WE"}
_lock = threading.Lock()


def spring_dst(year: int) -> pd.Timestamp:
    """US DST start (2007+): second Sunday of March."""
    first = pd.Timestamp(year=year, month=3, day=1)
    return first + pd.Timedelta(days=(6 - first.weekday()) % 7 + 7)


# The legacy IV service returns HTTP 503 for any window containing the local spring-forward
# hour (verified 2026-10-22). The DST day is therefore fetched in UTC pieces and pieces that
# still fail are skipped (<= 4 h lost per site-year for CONUS sites; logged).
DST_PIECES = [("00:00", "06:59"), ("07:00", "10:59"), ("11:00", "23:59")]  # 07-11Z = US spring-forward hours


def site_windows(begin: pd.Timestamp, end: pd.Timestamp):
    """Yield (year, [(start, end, optional), ...]) windows covering [begin, end]."""
    for year in range(begin.year, end.year + 1):
        dst = spring_dst(year)
        raw = []
        for q in range(4):
            s = pd.Timestamp(year=year, month=3 * q + 1, day=1)
            e = s + pd.offsets.QuarterEnd(0)
            if q == 0:
                raw += [(s, dst - pd.Timedelta(days=1)), (dst + pd.Timedelta(days=1), e)]
            else:
                raw.append((s, e))
        wins = []
        for s, e in raw:
            if e < begin or s > end:
                continue
            wins.append((max(s, begin).strftime("%Y-%m-%d"), min(e, end).strftime("%Y-%m-%d"), False))
        if begin <= dst <= end:
            d = dst.strftime("%Y-%m-%d")
            wins += [(f"{d}T{a}Z", f"{d}T{b}Z", True) for a, b in DST_PIECES]
        if wins:
            yield year, wins


def fetch_site_year(site: str, year: int, wins, params, out_dir) -> int:
    dest = out_dir / site / f"{year}.parquet"
    empty = out_dir / site / f"{year}.empty"
    if dest.exists() or empty.exists():
        return -1
    frames = []
    for s, e, optional in wins:
        try:
            df = U.get_iv(site, params, s, e)
        except U.ServiceUnavailable:
            if not optional:
                raise
            log.info("%s %s..%s skipped (503 on DST piece)", site, s, e)
            continue
        if not df.empty:
            frames.append(df)
        time.sleep(0.1)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        empty.touch()
        return 0
    df = pd.concat(frames, ignore_index=True)
    df = df[["site_no", "time_utc", "parm_cd", "ts_id", "value", "qualifier", "tz_cd"]]
    df["qualifier"] = df["qualifier"].astype("category")
    df["parm_cd"] = df["parm_cd"].astype("category")
    df["tz_cd"] = df["tz_cd"].astype("category")
    tmp = dest.with_suffix(".tmp")
    df.to_parquet(tmp, index=False, compression="zstd")
    tmp.rename(dest)
    return len(df)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--start", default="2007-10-01")
    ap.add_argument("--end", default="2026-10-20")
    ap.add_argument("--out-name", default="usgs_iv",
                    help="raw dataset directory (e.g. usgs_iv_refresh_20261108 for an approval-status re-pull)")
    a = ap.parse_args()
    out = paths.raw(a.out_name)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(out / "download.log"), logging.StreamHandler()])
    sites = pd.read_parquet(paths.raw("usgs_dv") / "iv_core_sites.parquet")
    sites = sites[sites["site_tp_cd"].isin(SURFACE_TYPES)]
    cat = pd.read_parquet(paths.raw("usgs_dv") / "dv_series_catalog.parquet")
    cat = cat[(cat["data_type_cd"] == "dv") & cat["parm_cd"].isin(U.WQ_PARAMS)]
    span = cat.groupby("site_no").agg(b=("begin_date", "min"), e=("end_date", "max"))
    g0, g1 = pd.Timestamp(a.start), pd.Timestamp(a.end)
    params = list(U.ALL_PARAMS)
    jobs = []
    for site in sites["site_no"]:
        b = max(pd.Timestamp(span.loc[site, "b"]), g0)
        e = min(pd.Timestamp(span.loc[site, "e"]), g1)
        if b > e:
            continue
        for year, wins in site_windows(b, e):
            jobs.append((site, year, wins))
    log.info("%d sites, %d site-years to fetch", sites.shape[0], len(jobs))
    done = rows = fails = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(fetch_site_year, s, y, w, params, out): (s, y) for s, y, w in jobs}
        for f in as_completed(futs):
            s, y = futs[f]
            try:
                n = f.result()
                done += 1
                rows += max(n, 0)
            except Exception as e:  # noqa: BLE001 — log and continue; rerun resumes
                fails += 1
                log.error("%s %d failed: %s", s, y, str(e)[:300])
            if (done + fails) % 100 == 0:
                log.info("progress %d/%d site-years, %d rows, %d failures", done + fails, len(jobs), rows, fails)
    files = sorted(out.rglob("*.parquet"))
    write_manifest("usgs_iv_raw" if a.out_name == "usgs_iv" else f"{a.out_name}_raw", files=files, sources=[U.IV_URL],
                   description=f"USGS NWIS instantaneous values {a.start}..{a.end}, quarterly legacy-IV "
                               f"requests for {sites.shape[0]} SENTINEL-Sonde surface-water core sites; "
                               "missing codes masked; times converted to UTC from tz_cd.",
                   license="US public domain", hash_files=False,
                   stats={"n_sites": int(sites.shape[0]), "n_site_years": len(jobs), "failures": fails})
    log.info("done: %d site-years, %d rows, %d failures", done, rows, fails)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
