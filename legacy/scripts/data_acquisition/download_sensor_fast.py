#!/usr/bin/env python3
"""Fast sensor data download — daily values instead of instantaneous.

The downstream pipeline (download_s2_massive.py) resamples IV data to daily
with df.resample("1D").mean(), so daily values (DV) are equivalent but ~100x
smaller per response. Also batches up to 50 stations per API request.

MIT License — Bryan Cheng, 2026
"""

import json
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT))

SITE_INFO_DIR = PROJECT / "data" / "raw" / "hydrology" / "nhdplus" / "cache" / "site_info"
SENSOR_DIR = PROJECT / "data" / "raw" / "sensor" / "full"
SENSOR_DIR.mkdir(parents=True, exist_ok=True)

PARAM_CODES = "00010,00300,00400,63680,00095"
PARAM_MAP = {
    "00010": "Temp",
    "00300": "DO",
    "00400": "pH",
    "63680": "Turb",
    "00095": "SpCond",
}

# DV endpoint — smaller batches are more reliable
BATCH_SIZE = 5
MAX_WORKERS = 32
REQUEST_TIMEOUT = 120


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def download_dv_batch(site_nos):
    """Download daily values for a batch of stations.

    Returns dict of {site_no: DataFrame} for successful downloads.
    """
    results = {}
    sites_str = ",".join(site_nos)

    url = "https://waterservices.usgs.gov/nwis/dv/"
    params = {
        "format": "json",
        "sites": sites_str,
        "parameterCd": PARAM_CODES,
        "startDT": "2015-01-01",
        "endDT": "2026-01-01",
        "siteStatus": "all",
        "statCd": "00003",  # mean daily value
    }

    try:
        resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        ts_list = data.get("value", {}).get("timeSeries", [])
        if not ts_list:
            return results

        # Group time series by site
        site_series = {}  # site_no -> {col_name: Series}
        for ts in ts_list:
            source_info = ts.get("sourceInfo", {})
            site_code = source_info.get("siteCode", [{}])
            if isinstance(site_code, list):
                site_no = site_code[0].get("value", "") if site_code else ""
            else:
                site_no = str(site_code)

            if not site_no:
                continue

            var_code = ts.get("variable", {}).get("variableCode", [{}])[0].get("value", "")
            col_name = PARAM_MAP.get(var_code, var_code)
            values = ts.get("values", [{}])[0].get("value", [])
            if not values:
                continue

            times = []
            vals = []
            for v in values:
                try:
                    t = pd.to_datetime(v["dateTime"], utc=True)
                    val = float(v["value"])
                    if val < -999:
                        continue
                    times.append(t)
                    vals.append(val)
                except (ValueError, KeyError):
                    continue

            if times:
                s = pd.Series(vals, index=pd.DatetimeIndex(times), name=col_name)
                if site_no not in site_series:
                    site_series[site_no] = {}
                site_series[site_no][col_name] = s

        # Build DataFrames
        for site_no, series_dict in site_series.items():
            if not series_dict:
                continue
            df = pd.DataFrame(series_dict)
            df.index.name = "datetime"
            df = df.sort_index()
            df = df[~df.index.duplicated(keep="first")]
            if len(df) >= 30:  # at least 30 days of data
                results[site_no] = df

    except Exception as e:
        log(f"  Batch error ({len(site_nos)} sites): {e}")

    return results


def download_batch_job(site_nos):
    """Wrapper for thread pool — downloads batch and saves parquets."""
    saved = 0
    failed = 0
    try:
        batch_results = download_dv_batch(site_nos)
        for site_no, df in batch_results.items():
            out_path = SENSOR_DIR / f"{site_no}.parquet"
            if not out_path.exists():
                df.to_parquet(out_path)
                saved += 1
    except Exception:
        failed = len(site_nos)
    return saved, failed


def main():
    log("=" * 60)
    log("=== FAST Sensor Data Download (Daily Values, Batched) ===")
    log("=" * 60)

    # Find all stations that need data
    all_stations = {f.stem for f in SITE_INFO_DIR.glob("*.json")}
    existing = {f.stem for f in SENSOR_DIR.glob("*.parquet")}
    need = sorted(all_stations - existing)

    log(f"Total stations with site info: {len(all_stations)}")
    log(f"Already have parquet data: {len(existing)}")
    log(f"Need to download: {len(need)}")
    log(f"Batch size: {BATCH_SIZE} stations per request")
    log(f"Workers: {MAX_WORKERS}")

    if not need:
        log("Nothing to download!")
        return

    # Build batches
    batches = []
    for i in range(0, len(need), BATCH_SIZE):
        batches.append(need[i:i + BATCH_SIZE])

    log(f"Total batches: {len(batches)}")
    log(f"Estimated time: ~{len(batches) / MAX_WORKERS * 10 / 60:.1f} minutes")

    # Execute
    completed = 0
    total_saved = 0
    total_failed = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(download_batch_job, batch): batch for batch in batches}
        for future in as_completed(futures):
            try:
                saved, failed = future.result()
                total_saved += saved
                total_failed += failed
            except Exception:
                pass

            completed += 1
            if completed % 5 == 0 or completed == len(batches):
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                remaining = (len(batches) - completed) / rate if rate > 0 else 0
                log(f"  Batches: {completed}/{len(batches)} | "
                    f"Saved: {total_saved} | "
                    f"Rate: {rate:.1f} batch/s | "
                    f"ETA: {remaining/60:.1f} min")

    elapsed = time.time() - start_time
    final_count = len(list(SENSOR_DIR.glob("*.parquet")))

    log(f"\n{'=' * 60}")
    log(f"Download complete in {elapsed/60:.1f} minutes")
    log(f"New parquets saved: {total_saved}")
    log(f"Total parquets: {final_count}")
    log("DONE")


if __name__ == "__main__":
    main()
