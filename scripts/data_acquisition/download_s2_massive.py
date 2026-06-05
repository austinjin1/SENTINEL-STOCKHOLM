#!/usr/bin/env python3
"""Massive Sentinel-2 tile download for WaterDroneNet expansion.

Downloads 4-band (B02,B03,B04,B08) 224x224 S2 tiles from Microsoft
Planetary Computer for all USGS stations, sampling every 5 days from
2017-01-01 to 2025-12-31 (~657 dates per station).

Tiles are cached as .npz files in data/processed/satellite/drone_tiles/.
Existing cached tiles are skipped. No NPZ assembly -- training loads
directly from the tile cache.

Target: 500K+ cached tiles (from ~19K currently).

MIT License -- Bryan Cheng, 2026
"""

import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT))

SENSOR_DIR = PROJECT / "data" / "raw" / "sensor" / "full"
SITE_INFO_DIR = PROJECT / "data" / "raw" / "hydrology" / "nhdplus" / "cache" / "site_info"
TILE_CACHE = PROJECT / "data" / "processed" / "satellite" / "drone_tiles"
TILE_CACHE.mkdir(parents=True, exist_ok=True)

BANDS = ["B02", "B03", "B04", "B08"]  # Blue, Green, Red, NIR
PATCH_SIZE = 224
MAX_CLOUD = 30
TARGET_COLS = ["DO", "pH", "Turb", "Temp", "SpCond"]
PARAM_RANGES = {
    "DO": (0, 20),
    "pH": (4, 10),
    "Turb": (0, 1000),
    "Temp": (-5, 45),
    "SpCond": (0, 50000),
}

# Dense sampling: every 5 days from 2017-01-01 to 2025-12-31
SAMPLE_DATES = []
_d = datetime(2017, 1, 1)
_end = datetime(2025, 12, 31)
while _d <= _end:
    SAMPLE_DATES.append(_d.strftime("%Y-%m-%d"))
    _d += timedelta(days=5)

MAX_WORKERS = 24  # parallel download workers
LOG_INTERVAL = 200  # log every N tiles
MAX_RETRIES = 3  # retry failed downloads


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_station_coords():
    """Load all station coordinates from site_info JSONs."""
    coords = {}
    for f in SITE_INFO_DIR.glob("*.json"):
        try:
            d = json.loads(f.read_text())
            coords[d["site_no"]] = (float(d["lat"]), float(d["lon"]))
        except Exception:
            pass
    return coords


def download_tile(lat, lon, date_str, station_id):
    """Download single 4-band S2 tile with retry logic. Returns True/False."""
    import planetary_computer
    import pystac_client
    import rasterio
    from pyproj import Transformer

    cache_path = TILE_CACHE / f"{station_id}_{date_str}.npz"
    if cache_path.exists():
        # Verify cache integrity
        try:
            arr = np.load(cache_path)["image"]
            if arr.shape == (len(BANDS), PATCH_SIZE, PATCH_SIZE):
                return True  # valid cache hit
        except Exception:
            cache_path.unlink(missing_ok=True)

    for attempt in range(MAX_RETRIES):
        try:
            catalog = pystac_client.Client.open(
                "https://planetarycomputer.microsoft.com/api/stac/v1",
                modifier=planetary_computer.sign_inplace,
            )

            bbox = [lon - 0.015, lat - 0.015, lon + 0.015, lat + 0.015]
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            t_start = (dt - timedelta(days=5)).strftime("%Y-%m-%d")
            t_end = (dt + timedelta(days=5)).strftime("%Y-%m-%d")

            search = catalog.search(
                collections=["sentinel-2-l2a"],
                bbox=bbox,
                datetime=f"{t_start}/{t_end}",
                query={"eo:cloud_cover": {"lt": MAX_CLOUD}},
                max_items=5,
            )
            items = list(search.items())
            if not items:
                return False  # no imagery available, don't retry

            target_ts = dt.timestamp()
            items.sort(key=lambda it: abs(
                datetime.fromisoformat(
                    it.properties["datetime"].replace("Z", "+00:00")
                ).timestamp() - target_ts
            ))
            item = items[0]
            signed = planetary_computer.sign(item)

            bands_data = []
            for band_name in BANDS:
                asset = signed.assets.get(band_name)
                if asset is None:
                    return False
                with rasterio.open(asset.href) as src:
                    transformer = Transformer.from_crs(
                        "EPSG:4326", src.crs, always_xy=True
                    )
                    x, y = transformer.transform(lon, lat)
                    py_idx, px_idx = src.index(x, y)
                    half = PATCH_SIZE // 2
                    if not (half <= px_idx < src.width - half and
                            half <= py_idx < src.height - half):
                        return False
                    window = rasterio.windows.Window(
                        px_idx - half, py_idx - half, PATCH_SIZE, PATCH_SIZE
                    )
                    data = src.read(1, window=window).astype(np.float32) / 10000.0
                    if data.shape != (PATCH_SIZE, PATCH_SIZE):
                        return False
                    bands_data.append(data)

            if len(bands_data) != len(BANDS):
                return False

            image = np.stack(bands_data, axis=0)
            if np.isnan(image).any() or image.max() < 0.001:
                return False

            np.savez_compressed(cache_path, image=image)
            return True

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)  # exponential backoff: 1s, 2s
                continue
            return False

    return False


def download_tile_job(args):
    """Wrapper for thread pool."""
    station_id, lat, lon, date_str = args
    try:
        result = download_tile(lat, lon, date_str, station_id)
        return (station_id, date_str, result)
    except Exception:
        return (station_id, date_str, False)


def main():
    import pandas as pd

    log("=" * 60)
    log("=== WaterDroneNet MASSIVE S2 Tile Download ===")
    log("=" * 60)
    log(f"Bands: {BANDS}")
    log(f"Patch size: {PATCH_SIZE}x{PATCH_SIZE}")
    log(f"Workers: {MAX_WORKERS}")
    log(f"Sample dates: {len(SAMPLE_DATES)} (every 5 days, 2017-2025)")
    log(f"Max cloud cover: {MAX_CLOUD}%")
    log(f"Retries per tile: {MAX_RETRIES}")

    # Load station coordinates
    coords = load_station_coords()
    log(f"\nStations with GPS coordinates: {len(coords)}")

    # Find stations with sensor data
    sensor_stations = {f.stem for f in SENSOR_DIR.glob("*.parquet")}
    valid_stations = sorted(set(coords.keys()) & sensor_stations)
    log(f"Stations with GPS + sensor data: {len(valid_stations)}")

    # Count existing tiles
    existing_tiles = {f.stem for f in TILE_CACHE.glob("*.npz")}
    log(f"Already cached tiles: {len(existing_tiles)}")

    # Pre-load WQ data for pairing check
    log("\nLoading water quality data for pairing check...")
    station_wq_dates = {}  # sid -> set of date strings with valid WQ
    loaded = 0
    for sid in valid_stations:
        fpath = SENSOR_DIR / f"{sid}.parquet"
        try:
            df = pd.read_parquet(fpath)
            if not isinstance(df.index, pd.DatetimeIndex):
                if "datetime" in df.columns:
                    df = df.set_index("datetime")
                elif len(df.columns) > 0:
                    # Try to parse index
                    df.index = pd.to_datetime(df.index, utc=True)

            df.index = pd.to_datetime(df.index, utc=True)
            daily = df.resample("1D").mean()

            # Check which columns exist
            available_cols = [c for c in TARGET_COLS if c in daily.columns]
            if len(available_cols) < 2:
                continue

            # Apply range filters
            for col in available_cols:
                lo, hi = PARAM_RANGES[col]
                daily.loc[(daily[col] < lo) | (daily[col] > hi), col] = np.nan

            # Find dates with at least 2 valid parameters
            valid_dates = set()
            for idx, row in daily.iterrows():
                n_valid = sum(1 for c in available_cols if pd.notna(row.get(c, np.nan)))
                if n_valid >= 2:
                    valid_dates.add(idx.strftime("%Y-%m-%d"))

            if valid_dates:
                station_wq_dates[sid] = valid_dates
                loaded += 1

        except Exception:
            pass

        if loaded % 100 == 0 and loaded > 0:
            log(f"  Loaded WQ for {loaded} stations...")

    log(f"Stations with usable WQ data: {len(station_wq_dates)}")

    # Build download jobs
    log("\nBuilding download job list...")
    jobs = []
    jobs_by_station = {}
    skipped_cached = 0
    skipped_no_wq = 0

    for sid in valid_stations:
        if sid not in station_wq_dates:
            continue
        lat, lon = coords[sid]
        wq_dates = station_wq_dates[sid]
        station_jobs = 0

        for date_str in SAMPLE_DATES:
            cache_key = f"{sid}_{date_str}"

            # Check if already cached
            if cache_key in existing_tiles:
                skipped_cached += 1
                continue

            # Check if WQ data exists within +/-5 days
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            has_wq = False
            for delta in range(-5, 6):
                check = (dt + timedelta(days=delta)).strftime("%Y-%m-%d")
                if check in wq_dates:
                    has_wq = True
                    break

            if not has_wq:
                skipped_no_wq += 1
                continue

            jobs.append((sid, lat, lon, date_str))
            station_jobs += 1

        if station_jobs > 0:
            jobs_by_station[sid] = station_jobs

    log(f"Total download jobs: {len(jobs)}")
    log(f"Stations with jobs: {len(jobs_by_station)}")
    log(f"Skipped (already cached): {skipped_cached}")
    log(f"Skipped (no WQ match): {skipped_no_wq}")
    log(f"Estimated target: ~{len(jobs) + skipped_cached + len(existing_tiles)} total tiles")

    if not jobs:
        log("No new tiles to download!")
        return

    # Execute downloads in parallel
    log(f"\n{'=' * 60}")
    log("Starting parallel downloads...")
    log(f"{'=' * 60}")

    completed = 0
    success = 0
    errors = 0
    cache_hits = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(download_tile_job, job): job for job in jobs}
        for future in as_completed(futures):
            try:
                sid, date_str, result = future.result()
                if result:
                    success += 1
                else:
                    errors += 1
            except Exception:
                errors += 1

            completed += 1
            if completed % LOG_INTERVAL == 0:
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                eta_hrs = (len(jobs) - completed) / rate / 3600 if rate > 0 else 0
                log(f"  Progress: {completed}/{len(jobs)} "
                    f"({100*completed/len(jobs):.1f}%) | "
                    f"success={success} errors={errors} | "
                    f"rate={rate:.1f}/s | "
                    f"ETA={eta_hrs:.1f}h")

    elapsed = time.time() - start_time
    log(f"\n{'=' * 60}")
    log("Download complete!")
    log(f"  Total attempted: {completed}")
    log(f"  New tiles downloaded: {success}")
    log(f"  Failed/no imagery: {errors}")
    log(f"  Time: {elapsed/3600:.1f} hours ({elapsed:.0f}s)")
    log(f"  Rate: {completed/elapsed:.1f} tiles/sec")

    # Final cache count
    final_count = len(list(TILE_CACHE.glob("*.npz")))
    log(f"\n  Total tiles in cache: {final_count}")
    log(f"  Target: 500,000")
    log(f"  Progress: {100*final_count/500000:.1f}%")
    log("DONE")


if __name__ == "__main__":
    main()
