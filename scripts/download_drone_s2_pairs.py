#!/usr/bin/env python3
"""Download RGB+NIR Sentinel-2 patches paired with USGS water quality data.

Builds the training dataset for WaterDroneNet (SENTINEL Mini).
Uses 4 bands (B02=Blue, B03=Green, B04=Red, B08=NIR) to simulate
what a drone with a multispectral camera would capture.

Uses parallel downloads for speed. Caches tiles to disk so interrupted
runs can resume. Assembles final NPZ from cached tiles at the end.

Output: data/processed/satellite/drone_wq_pairs.npz

MIT License — Bryan Cheng, 2026
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

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

SENSOR_DIR = PROJECT / "data" / "raw" / "sensor" / "full"
SITE_INFO_DIR = PROJECT / "data" / "raw" / "hydrology" / "nhdplus" / "cache" / "site_info"
OUT_DIR = PROJECT / "data" / "processed" / "satellite"
TILE_CACHE = OUT_DIR / "drone_tiles"
TILE_CACHE.mkdir(parents=True, exist_ok=True)

BANDS = ["B02", "B03", "B04", "B08"]  # Blue, Green, Red, NIR
PATCH_SIZE = 224
MAX_CLOUD = 30
TARGET_COLS = ["DO", "pH", "Turb", "Temp", "SpCond"]
PARAM_RANGES = {"DO": (0, 20), "pH": (4, 10), "Turb": (0, 1000),
                "Temp": (-5, 45), "SpCond": (0, 50000)}

# Sample dates: seasonal coverage across S2 era
SAMPLE_DATES = []
for year in range(2018, 2026):
    for month in range(1, 13):
        SAMPLE_DATES.append(f"{year}-{month:02d}-15")

MAX_WORKERS = 8  # parallel downloads
MAX_DATES_PER_STATION = 40


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_station_coords():
    coords = {}
    for f in SITE_INFO_DIR.glob("*.json"):
        try:
            d = json.loads(f.read_text())
            coords[d["site_no"]] = (float(d["lat"]), float(d["lon"]))
        except Exception:
            pass
    return coords


def download_tile(lat, lon, date_str, station_id):
    """Download single 4-band S2 tile. Returns image array or None."""
    import planetary_computer
    import pystac_client
    import rasterio
    from pyproj import Transformer

    cache_path = TILE_CACHE / f"{station_id}_{date_str}.npz"
    if cache_path.exists():
        try:
            return np.load(cache_path)["image"]
        except Exception:
            cache_path.unlink(missing_ok=True)

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
            return None

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
                return None
            with rasterio.open(asset.href) as src:
                transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                x, y = transformer.transform(lon, lat)
                py_idx, px_idx = src.index(x, y)
                half = PATCH_SIZE // 2
                if not (half <= px_idx < src.width - half and
                        half <= py_idx < src.height - half):
                    return None
                window = rasterio.windows.Window(
                    px_idx - half, py_idx - half, PATCH_SIZE, PATCH_SIZE)
                data = src.read(1, window=window).astype(np.float32) / 10000.0
                if data.shape != (PATCH_SIZE, PATCH_SIZE):
                    return None
                bands_data.append(data)

        if len(bands_data) != len(BANDS):
            return None

        image = np.stack(bands_data, axis=0)
        if np.isnan(image).any() or image.max() < 0.001:
            return None

        np.savez_compressed(cache_path, image=image)
        return image
    except Exception:
        return None


def download_tile_job(args):
    """Wrapper for thread pool."""
    station_id, lat, lon, date_str = args
    try:
        img = download_tile(lat, lon, date_str, station_id)
        return (station_id, date_str, img)
    except Exception:
        return (station_id, date_str, None)


def main():
    import pandas as pd

    log("=== WaterDroneNet S2 Data Download (Parallel) ===")
    log(f"Bands: {BANDS}, Workers: {MAX_WORKERS}")

    coords = load_station_coords()
    sensor_stations = {f.stem for f in SENSOR_DIR.glob("*.parquet")}
    valid_stations = sorted(set(coords.keys()) & sensor_stations)
    log(f"Stations with GPS + sensor data: {len(valid_stations)}")

    existing_tiles = {f.stem for f in TILE_CACHE.glob("*.npz")}
    log(f"Already cached tiles: {len(existing_tiles)}")

    # Pre-load WQ data for all stations
    log("Loading water quality data...")
    station_wq = {}
    for sid in valid_stations:
        fpath = SENSOR_DIR / f"{sid}.parquet"
        try:
            df = pd.read_parquet(fpath)
            if not isinstance(df.index, pd.DatetimeIndex):
                continue
            df.index = pd.to_datetime(df.index, utc=True)
            daily = df.resample("1D").mean()
            for col in TARGET_COLS:
                if col not in daily.columns:
                    daily[col] = np.nan
                else:
                    lo, hi = PARAM_RANGES[col]
                    daily.loc[(daily[col] < lo) | (daily[col] > hi), col] = np.nan
            wq = {}
            for idx, row in daily.iterrows():
                t = np.array([row[c] for c in TARGET_COLS], dtype=np.float32)
                if np.isfinite(t).sum() >= 2:
                    wq[idx.strftime("%Y-%m-%d")] = t
            if wq:
                station_wq[sid] = wq
        except Exception:
            pass
    log(f"Stations with WQ data: {len(station_wq)}")

    # Build download jobs: (station_id, lat, lon, date_str)
    # Only for tiles not already cached
    jobs = []
    for sid in valid_stations:
        if sid not in station_wq:
            continue
        lat, lon = coords[sid]
        wq = station_wq[sid]
        n_added = 0
        for date_str in SAMPLE_DATES:
            if n_added >= MAX_DATES_PER_STATION:
                break
            cache_key = f"{sid}_{date_str}"
            # Check if WQ data exists within ±3 days
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            has_wq = False
            for delta in range(-3, 4):
                check = (dt + timedelta(days=delta)).strftime("%Y-%m-%d")
                if check in wq:
                    has_wq = True
                    break
            if not has_wq:
                continue
            # Submit whether cached or not (download_tile handles cache)
            jobs.append((sid, lat, lon, date_str))
            n_added += 1

    log(f"Download jobs: {len(jobs)} (incl. {len(existing_tiles)} cached)")

    # Execute downloads in parallel
    results = {}  # (station_id, date_str) -> image
    completed = 0
    errors = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(download_tile_job, job): job for job in jobs}
        for future in as_completed(futures):
            try:
                sid, date_str, img = future.result()
                if img is not None:
                    results[(sid, date_str)] = img
            except Exception:
                errors += 1
            completed += 1
            if completed % 100 == 0:
                log(f"  {completed}/{len(jobs)} done | "
                    f"{len(results)} tiles | {errors} errors")

    log(f"\nDownload complete: {len(results)} tiles from {completed} jobs")

    # Pair tiles with WQ targets
    images, targets, metadata = [], [], []
    for (sid, date_str), img in results.items():
        wq = station_wq.get(sid, {})
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        best_t, best_d = None, 999
        for delta in range(-3, 4):
            check = (dt + timedelta(days=delta)).strftime("%Y-%m-%d")
            if check in wq and abs(delta) < best_d:
                best_d = abs(delta)
                best_t = wq[check]
        if best_t is None:
            continue
        images.append(img)
        targets.append(best_t)
        lat, lon = coords.get(sid, (0, 0))
        metadata.append({"site_id": sid, "date": date_str,
                         "lat": lat, "lon": lon})

    log(f"Paired samples: {len(images)}")

    if not images:
        log("No pairs - check network access")
        return

    imgs = np.stack(images)
    tgts = np.stack(targets)

    out_file = OUT_DIR / "drone_wq_pairs.npz"
    np.savez_compressed(out_file, images=imgs, targets=tgts,
                        metadata=json.dumps(metadata),
                        target_names=json.dumps(TARGET_COLS),
                        bands=json.dumps(BANDS))

    log(f"\nSaved: {out_file}")
    log(f"Shape: images={imgs.shape}, targets={tgts.shape}")
    log(f"Stations: {len(set(m['site_id'] for m in metadata))}")

    for j, col in enumerate(TARGET_COLS):
        valid = tgts[~np.isnan(tgts[:, j]), j]
        log(f"  {col}: {len(valid)}/{len(tgts)} valid "
            f"(mean={valid.mean():.2f}, std={valid.std():.2f})")

    log("DONE")


if __name__ == "__main__":
    main()
