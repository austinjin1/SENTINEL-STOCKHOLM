#!/usr/bin/env python3
"""Discover additional USGS stations with water quality data.

Queries the USGS NWIS web service for stream stations that have
instantaneous value (IV) data for water quality parameters:
  00010 = Temperature
  00300 = Dissolved Oxygen
  00400 = pH
  63680 = Turbidity
  00095 = Specific Conductance

Downloads site metadata and IV data for each new station found,
saving to the same directories used by the existing pipeline.

MIT License -- Bryan Cheng, 2026
"""

import json
import io
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

SITE_INFO_DIR = PROJECT / "data" / "raw" / "hydrology" / "nhdplus" / "cache" / "site_info"
SENSOR_DIR = PROJECT / "data" / "raw" / "sensor" / "full"
SITE_INFO_DIR.mkdir(parents=True, exist_ok=True)
SENSOR_DIR.mkdir(parents=True, exist_ok=True)

# USGS parameter codes for water quality
PARAM_CODES = "00010,00300,00400,63680,00095"
PARAM_MAP = {
    "00010": "Temp",
    "00300": "DO",
    "00400": "pH",
    "63680": "Turb",
    "00095": "SpCond",
}

# State FIPS codes - query by state to avoid timeouts on massive requests
STATE_CODES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
    "DC", "PR", "VI",
]

MAX_IV_WORKERS = 8
REQUEST_TIMEOUT = 120


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def get_existing_stations():
    """Return set of station IDs that already have site_info JSON."""
    return {f.stem for f in SITE_INFO_DIR.glob("*.json")}


def discover_sites_for_state(state_code):
    """Query NWIS for stream stations with WQ IV data in a state."""
    url = "https://waterservices.usgs.gov/nwis/site/"
    params = {
        "format": "rdb",
        "stateCd": state_code,
        "parameterCd": PARAM_CODES,
        "siteType": "ST",
        "siteStatus": "all",
        "hasDataTypeCd": "iv",
    }
    try:
        resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return parse_rdb_sites(resp.text, state_code)
    except Exception as e:
        log(f"  Warning: failed to query state {state_code}: {e}")
        return []


def parse_rdb_sites(rdb_text, state_code):
    """Parse USGS RDB format response into list of station dicts."""
    sites = []
    lines = rdb_text.strip().split("\n")
    # Skip comment lines (start with #)
    data_lines = [l for l in lines if not l.startswith("#")]
    if len(data_lines) < 2:
        return sites

    header = data_lines[0].split("\t")
    # Second line is format spec, skip it
    # Data starts at line index 2

    # Find column indices
    col_map = {}
    for i, col in enumerate(header):
        col_map[col.strip()] = i

    needed = {
        "site_no": None,
        "station_nm": None,
        "dec_lat_va": None,
        "dec_long_va": None,
        "drain_area_va": None,
        "huc_cd": None,
        "state_cd": None,
        "site_tp_cd": None,
    }
    for key in needed:
        if key in col_map:
            needed[key] = col_map[key]

    if needed["site_no"] is None or needed["dec_lat_va"] is None:
        return sites

    for line in data_lines[2:]:
        fields = line.split("\t")
        if len(fields) <= max(v for v in needed.values() if v is not None):
            continue
        try:
            site_no = fields[needed["site_no"]].strip()
            lat_str = fields[needed["dec_lat_va"]].strip() if needed["dec_lat_va"] is not None else ""
            lon_str = fields[needed["dec_long_va"]].strip() if needed["dec_long_va"] is not None else ""
            if not site_no or not lat_str or not lon_str:
                continue
            lat = float(lat_str)
            lon = float(lon_str)
            if lat == 0 and lon == 0:
                continue

            station_nm = fields[needed["station_nm"]].strip() if needed["station_nm"] is not None else ""
            drain_area = ""
            if needed["drain_area_va"] is not None:
                drain_area = fields[needed["drain_area_va"]].strip()
            huc_cd = ""
            if needed["huc_cd"] is not None:
                huc_cd = fields[needed["huc_cd"]].strip()
            state_cd_val = ""
            if needed["state_cd"] is not None:
                state_cd_val = fields[needed["state_cd"]].strip()
            site_tp = "ST"
            if needed["site_tp_cd"] is not None:
                site_tp = fields[needed["site_tp_cd"]].strip()

            sites.append({
                "site_no": site_no,
                "station_nm": station_nm,
                "lat": lat,
                "lon": lon,
                "drain_area_sq_mi": float(drain_area) if drain_area else None,
                "huc_cd": huc_cd,
                "state_cd": state_cd_val,
                "site_type": site_tp,
            })
        except (ValueError, IndexError):
            continue

    return sites


def save_site_info(site):
    """Save station info JSON."""
    out_path = SITE_INFO_DIR / f"{site['site_no']}.json"
    if out_path.exists():
        return  # don't overwrite
    with open(out_path, "w") as f:
        json.dump(site, f, indent=2)


def download_iv_data(site_no):
    """Download IV data for a station from USGS NWIS.

    Downloads 10 years of instantaneous values, resamples to daily,
    and saves as parquet.
    """
    out_path = SENSOR_DIR / f"{site_no}.parquet"
    if out_path.exists():
        return True  # already have it

    url = "https://waterservices.usgs.gov/nwis/iv/"
    params = {
        "format": "json",
        "sites": site_no,
        "parameterCd": PARAM_CODES,
        "period": "P3650D",
        "siteStatus": "all",
    }

    try:
        resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        ts_list = data.get("value", {}).get("timeSeries", [])
        if not ts_list:
            return False

        all_series = {}
        for ts in ts_list:
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
                all_series[col_name] = s

        if not all_series:
            return False

        df = pd.DataFrame(all_series)
        df.index.name = "datetime"
        df = df.sort_index()
        # Remove duplicates
        df = df[~df.index.duplicated(keep="first")]

        df.to_parquet(out_path)
        return True

    except Exception as e:
        return False


def main():
    log("=== USGS Station Discovery ===")
    log(f"Site info dir: {SITE_INFO_DIR}")
    log(f"Sensor dir: {SENSOR_DIR}")

    existing = get_existing_stations()
    log(f"Existing stations with site info: {len(existing)}")
    existing_parquet = {f.stem for f in SENSOR_DIR.glob("*.parquet")}
    log(f"Existing stations with sensor data: {len(existing_parquet)}")

    # Phase 1: Discover sites from all states
    log("\n--- Phase 1: Discovering stations by state ---")
    all_sites = {}  # site_no -> site_dict
    for i, state in enumerate(STATE_CODES):
        sites = discover_sites_for_state(state)
        new_count = 0
        for site in sites:
            sid = site["site_no"]
            if sid not in all_sites:
                all_sites[sid] = site
                if sid not in existing:
                    new_count += 1
        if (i + 1) % 5 == 0 or new_count > 0:
            log(f"  [{i+1}/{len(STATE_CODES)}] {state}: {len(sites)} sites found, "
                f"{new_count} new | total unique: {len(all_sites)}")

    new_sites = {sid: s for sid, s in all_sites.items() if sid not in existing}
    log(f"\nTotal unique stations found: {len(all_sites)}")
    log(f"New stations (not in existing): {len(new_sites)}")

    # Phase 2: Save site info JSONs for new stations
    log("\n--- Phase 2: Saving site info JSONs ---")
    saved = 0
    for sid, site in new_sites.items():
        save_site_info(site)
        saved += 1
    log(f"Saved {saved} new site info JSONs")

    # Phase 3: Download IV data for stations that don't have parquet files
    # Include both new stations and existing ones missing parquet
    all_station_ids = set(all_sites.keys()) | existing
    need_iv = [sid for sid in all_station_ids if sid not in existing_parquet]
    log(f"\n--- Phase 3: Downloading IV data for {len(need_iv)} stations ---")

    completed = 0
    success = 0
    errors = 0

    with ThreadPoolExecutor(max_workers=MAX_IV_WORKERS) as pool:
        futures = {pool.submit(download_iv_data, sid): sid for sid in need_iv}
        for future in as_completed(futures):
            sid = futures[future]
            try:
                result = future.result()
                if result:
                    success += 1
                else:
                    errors += 1
            except Exception:
                errors += 1
            completed += 1
            if completed % 50 == 0:
                log(f"  IV download progress: {completed}/{len(need_iv)} "
                    f"| success={success} errors={errors}")

    log(f"\nIV download complete: {success} success, {errors} errors "
        f"out of {completed} attempted")

    # Final counts
    final_site_info = len(list(SITE_INFO_DIR.glob("*.json")))
    final_parquet = len(list(SENSOR_DIR.glob("*.parquet")))
    log(f"\n=== Final Counts ===")
    log(f"Site info JSONs: {final_site_info}")
    log(f"Sensor parquets: {final_parquet}")
    log("DONE")


if __name__ == "__main__":
    main()
