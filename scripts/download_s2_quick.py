#!/usr/bin/env python3
"""Quick S2 tile download to reach 80K target from existing stations.

Only downloads tiles for stations that already have sensor parquets,
using dates not yet cached. Stops at 80K total.

MIT License — Bryan Cheng, 2026
"""

import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

SITE_INFO_DIR = PROJECT / "data" / "raw" / "hydrology" / "nhdplus" / "cache" / "site_info"
SENSOR_DIR = PROJECT / "data" / "raw" / "sensor" / "full"
TILE_CACHE = PROJECT / "data" / "processed" / "satellite" / "drone_tiles"

BANDS = ["B02", "B03", "B04", "B08"]
PATCH_SIZE = 224
MAX_CLOUD = 30
TARGET = 80000
MAX_WORKERS = 24


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def download_tile(lat, lon, date_str, station_id):
    import planetary_computer
    import pystac_client
    import rasterio
    from pyproj import Transformer

    cache_path = TILE_CACHE / f"{station_id}_{date_str}.npz"
    if cache_path.exists():
        return True

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
            collections=["sentinel-2-l2a"], bbox=bbox,
            datetime=f"{t_start}/{t_end}",
            query={"eo:cloud_cover": {"lt": MAX_CLOUD}}, max_items=3,
        )
        items = list(search.items())
        if not items:
            return False

        item = planetary_computer.sign(items[0])
        bands_data = []
        for band_name in BANDS:
            asset = item.assets.get(band_name)
            if asset is None:
                return False
            with rasterio.open(asset.href) as src:
                transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                x, y = transformer.transform(lon, lat)
                py_idx, px_idx = src.index(x, y)
                half = PATCH_SIZE // 2
                if not (half <= px_idx < src.width - half and half <= py_idx < src.height - half):
                    return False
                window = rasterio.windows.Window(px_idx - half, py_idx - half, PATCH_SIZE, PATCH_SIZE)
                data = src.read(1, window=window).astype(np.float32) / 10000.0
                if data.shape != (PATCH_SIZE, PATCH_SIZE):
                    return False
                bands_data.append(data)

        image = np.stack(bands_data, axis=0)
        if np.isnan(image).any() or image.max() < 0.001:
            return False
        np.savez_compressed(cache_path, image=image)
        return True
    except Exception:
        return False


def main():
    log(f"=== Quick S2 Download → {TARGET} tiles ===")

    existing = {f.stem for f in TILE_CACHE.glob("*.npz")}
    current = len(existing)
    need = TARGET - current
    log(f"Current: {current} | Need: {need}")

    if need <= 0:
        log("Already at target!")
        return

    # Load station coords
    coords = {}
    for f in SITE_INFO_DIR.glob("*.json"):
        try:
            d = json.loads(f.read_text())
            coords[d["site_no"]] = (float(d["lat"]), float(d["lon"]))
        except Exception:
            pass

    sensor_stations = {f.stem for f in SENSOR_DIR.glob("*.parquet")}
    valid = sorted(set(coords.keys()) & sensor_stations)
    log(f"Stations with GPS + sensor: {len(valid)}")

    # Generate candidate dates not already cached
    dates = []
    d = datetime(2020, 1, 1)
    end = datetime(2025, 12, 31)
    while d <= end:
        dates.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=10)

    jobs = []
    for sid in valid:
        lat, lon = coords[sid]
        for date_str in dates:
            if f"{sid}_{date_str}" not in existing:
                jobs.append((sid, lat, lon, date_str))
            if len(jobs) >= need * 3:  # overshoot since many will fail
                break
        if len(jobs) >= need * 3:
            break

    log(f"Jobs queued: {len(jobs)} (aiming for {need} successes)")

    completed = 0
    success = 0
    start = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {}
        for job in jobs:
            if success >= need:
                break
            futures[pool.submit(download_tile, job[1], job[2], job[3], job[0])] = job

        for future in as_completed(futures):
            try:
                if future.result():
                    success += 1
            except Exception:
                pass
            completed += 1
            if completed % 100 == 0:
                elapsed = time.time() - start
                rate = success / elapsed if elapsed > 0 else 0
                log(f"  {completed} attempted | {success} new | "
                    f"{rate:.1f}/s | total={current + success}")
            if success >= need:
                break

    final = len(list(TILE_CACHE.glob("*.npz")))
    log(f"\nDone! Final count: {final} tiles")


if __name__ == "__main__":
    main()
