#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Download hydrometeorological drivers for D1 SafeSwim: USGS daily discharge (4 Chicago-area gauges),
NOAA CO-OPS Calumet Harbor daily mean water level, NOAA NDBC buoys 45007/45174 (hourly → daily).

Sequential requests, ≥ 1 s apart, generic User-Agent, retries with backoff. Raw responses are kept
(small, < 20 MB) under raw('safeswim_drivers'); tidy Parquet under processed('safeswim_drivers');
manifest `safeswim_drivers`; datasheet docs/datasheets/safeswim_drivers.md.

    python -B scripts/data/download_safeswim_drivers.py --start-year 2005 --end-year 2026
"""

from __future__ import annotations

import argparse
import datetime as _dt
import gzip
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths  # noqa: E402
from sentinel.core.manifest import write_manifest  # noqa: E402
from sentinel.data.sources import safeswim_drivers as SD  # noqa: E402

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": SD.USER_AGENT, "Accept-Encoding": "gzip"})


def get(url: str, params: dict | None = None, *, retries: int = 5, ok404: bool = False) -> requests.Response | None:
    last = None
    for k in range(retries):
        try:
            r = SESSION.get(url, params=params, timeout=120)
            if r.status_code == 200:
                time.sleep(1.0)
                return r
            if r.status_code == 404 and ok404:
                time.sleep(1.0)
                return None
            if r.status_code == 429:
                time.sleep(float(r.headers.get("Retry-After", 60)))
                continue
            last = f"HTTP {r.status_code}"
        except requests.RequestException as e:
            last = repr(e)
        time.sleep(5 * (k + 1))
    raise RuntimeError(f"GET failed {url} {params}: {last}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--start-year", type=int, default=2005)
    ap.add_argument("--end-year", type=int, default=2026)
    a = ap.parse_args()
    raw = paths.raw("safeswim_drivers")
    out = paths.processed("safeswim_drivers")
    stamp = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    stats: dict = {"retrieved_utc": stamp, "years": [a.start_year, a.end_year]}

    # USGS daily discharge
    frames = []
    for site in SD.GAUGES:
        url, params = SD.usgs_dv_request(site, f"{a.start_year}-01-01", f"{a.end_year}-12-31")
        r = get(url, params)
        (raw / f"usgs_dv_{site}.json.gz").write_bytes(gzip.compress(r.content))
        df = SD.parse_usgs_dv(r.json(), site)
        frames.append(df)
        stats[f"usgs_{site}"] = {"n_days": int(len(df)), "n_valid": int(df["discharge_cfs"].notna().sum()),
                                 "first": str(df["date"].min().date()) if len(df) else None,
                                 "last": str(df["date"].max().date()) if len(df) else None}
    q = pd.concat(frames, ignore_index=True)
    q["unit"] = "ft3/s"
    q.to_parquet(out / "usgs_discharge_daily.parquet", index=False, compression="zstd")

    # CO-OPS daily mean water level
    frames = []
    for y in range(a.start_year, a.end_year + 1):
        url, params = SD.coops_request(y)
        r = get(url, params)
        payload = r.json()
        (raw / f"coops_{SD.COOPS_STATION}_{y}.json.gz").write_bytes(gzip.compress(r.content))
        frames.append(SD.parse_coops_daily(payload))
    wl = pd.concat(frames, ignore_index=True).drop_duplicates("date")
    wl["station"] = SD.COOPS_STATION
    wl.to_parquet(out / "coops_water_level_daily.parquet", index=False, compression="zstd")
    stats["coops"] = {"n_days": int(len(wl)), "n_valid": int(wl["water_level_m_igld"].notna().sum())}

    # NDBC buoys
    daily, masked, years_ok = [], {}, {}
    for st in SD.BUOYS:
        years_ok[st] = []
        for y in range(a.start_year, a.end_year + 1):
            r = get(SD.NDBC.format(station=st, year=y), ok404=True)
            if r is None:
                continue
            (raw / f"ndbc_{st}h{y}.txt.gz").write_bytes(r.content)
            h = SD.parse_ndbc_stdmet(r.content, st)
            masked[f"{st}_{y}"] = h.attrs.get("n_masked", 0)
            daily.append(SD.ndbc_daily(h))
            years_ok[st].append(y)
    nd = pd.concat(daily, ignore_index=True)
    nd.to_parquet(out / "ndbc_daily.parquet", index=False, compression="zstd")
    stats["ndbc"] = {"years": years_ok, "n_station_days": int(len(nd)), "n_masked_codes": masked}

    files = sorted(out.glob("*.parquet")) + sorted(raw.glob("*"))
    write_manifest("safeswim_drivers", files=files,
                   sources=[SD.USGS_DV, SD.COOPS, SD.NDBC.format(station="<id>", year="<year>")],
                   description="Chicago SafeSwim hydrometeorological drivers: USGS daily discharge (4 gauges), "
                               "CO-OPS 9087044 daily mean water level (IGLD85), NDBC 45007/45174 daily wave/wind/"
                               "water-temperature summaries",
                   license="US public domain (USGS, NOAA)", synthetic=False, stats=stats)
    print(json.dumps(stats, indent=1, default=str)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
