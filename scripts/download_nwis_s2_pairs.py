#!/usr/bin/env python3
"""Download tightly co-registered USGS NWIS + Sentinel-2 pairs for HydroViT.

Improvements over v2 (GRQA+WQP) dataset:
  - USGS station GPS is precise to ~10m (vs GRQA loose coordinates)
  - ±3-day temporal window (vs ±15-day in expand script)
  - Focuses on temperature (best current R²=0.526 → push to >0.55)
  - Also captures DO, pH, turbidity for other parameters

Target: 2,000+ new pairs from up to 1,130 USGS NWIS stations.
Combined with existing 847 pairs → 2,800+ total training samples.

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
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SEQ_DIR = PROJECT_ROOT / "data" / "processed" / "sensor" / "real_50k"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "satellite"
TILE_CACHE = OUT_DIR / "nwis_tiles"
TILE_CACHE.mkdir(parents=True, exist_ok=True)

COLLECTION = "sentinel-2-l2a"
BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
PATCH_SIZE = 224
MAX_CLOUD = 30
MAX_WORKERS = 4   # conservative — PC rate limits
TARGET_NEW_PAIRS = 2000
MAX_STATIONS = 400   # process at most this many stations
MAX_DATES_PER_STATION = 20  # max pairs per station

# NWIS parameter codes → target vector index
PARAM_CODES = {
    "00010": 11,   # water temp °C
    "00300": 7,    # dissolved oxygen mg/L
    "00400": 10,   # pH
    "63680": 1,    # turbidity FNU
    "00060": -1,   # discharge — not in WQ target vector
    "00095": -1,   # specific conductance — not in target
}
NUM_PARAMS = 16

# Sample dates evenly across Sentinel-2 era, spread by season
SAMPLE_DATES = []
for year in range(2018, 2024):
    for month in [1, 3, 5, 7, 9, 11]:  # every 2 months
        SAMPLE_DATES.append(f"{year}-{month:02d}-15")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def get_station_ids():
    """Extract unique station IDs from sequence filenames."""
    ids = set()
    for f in SEQ_DIR.glob("*.npz"):
        parts = f.stem.split("_seq")
        if len(parts) == 2:
            ids.add(parts[0])
    log(f"Found {len(ids)} unique station IDs")
    return sorted(ids)


def fetch_station_coords(station_ids, batch_size=100):
    """Fetch decimal GPS coordinates from USGS Site Service."""
    coords = {}
    for i in range(0, len(station_ids), batch_size):
        batch = station_ids[i:i + batch_size]
        url = (
            "https://waterservices.usgs.gov/nwis/site/"
            f"?format=rdb&sites={','.join(batch)}&siteOutput=expanded"
        )
        try:
            r = requests.get(url, timeout=30)
            if r.status_code != 200:
                continue
            for line in r.text.splitlines():
                if line.startswith("#") or line.startswith("agency_cd") or line.startswith("5s"):
                    continue
                parts = line.split("\t")
                if len(parts) >= 8:
                    try:
                        site_no = parts[1].strip()
                        lat = float(parts[6].strip())   # dec_lat_va
                        lon = float(parts[7].strip())   # dec_long_va
                        if -90 <= lat <= 90 and -180 <= lon <= 180:
                            coords[site_no] = (lat, lon)
                    except (ValueError, IndexError):
                        pass
        except Exception as e:
            log(f"  Coord fetch error batch {i//batch_size}: {e}")
        time.sleep(0.3)
    log(f"GPS coords: {len(coords)}/{len(station_ids)} stations")
    return coords


def fetch_nwis_daily(station_id, date_str):
    """Fetch WQ measurements for ±1 day around date_str from NWIS IV."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    start = (dt - timedelta(days=1)).strftime("%Y-%m-%d")
    end = (dt + timedelta(days=1)).strftime("%Y-%m-%d")
    pcodes = "00010,00300,00400,63680"

    url = (
        f"https://waterservices.usgs.gov/nwis/iv/"
        f"?format=json&sites={station_id}"
        f"&startDT={start}&endDT={end}"
        f"&parameterCd={pcodes}&siteStatus=all"
    )
    try:
        r = requests.get(url, timeout=20)
        if r.status_code != 200:
            return None
        data = r.json()
        ts_list = data.get("value", {}).get("timeSeries", [])

        params = {}
        for ts in ts_list:
            pcode = ts.get("variable", {}).get("variableCode", [{}])[0].get("value", "")
            if pcode not in PARAM_CODES:
                continue
            pidx = PARAM_CODES[pcode]
            if pidx < 0:
                continue
            vals = []
            for v in ts.get("values", [{}])[0].get("value", []):
                try:
                    vals.append(float(v["value"]))
                except (ValueError, KeyError):
                    pass
            if vals:
                params[pidx] = float(np.mean(vals))

        return params if params else None
    except Exception:
        return None


def download_tile(lat, lon, date_str, station_id):
    """Download 224×224 Sentinel-2 patch for given location/date."""
    import planetary_computer
    import pystac_client
    import rasterio
    from pyproj import Transformer
    from datetime import datetime, timedelta

    cache_path = TILE_CACHE / f"{station_id}_{date_str}.npz"
    if cache_path.exists():
        d = np.load(cache_path)
        return d["image"], float(d["cloud_cover"]), str(d["actual_date"])

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    bbox = [lon - 0.015, lat - 0.015, lon + 0.015, lat + 0.015]
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    t_start = (dt - timedelta(days=3)).strftime("%Y-%m-%d")
    t_end = (dt + timedelta(days=3)).strftime("%Y-%m-%d")

    search = catalog.search(
        collections=[COLLECTION],
        bbox=bbox,
        datetime=f"{t_start}/{t_end}",
        query={"eo:cloud_cover": {"lt": MAX_CLOUD}},
        max_items=5,
    )
    items = list(search.items())
    if not items:
        return None

    # Pick item closest in time to the target date
    target_ts = dt.timestamp()
    items.sort(key=lambda it: abs(
        datetime.fromisoformat(
            it.properties["datetime"].replace("Z", "+00:00")
        ).timestamp() - target_ts
    ))
    item = items[0]
    signed_item = planetary_computer.sign(item)

    bands_data = []
    for band_name in BANDS:
        asset = signed_item.assets.get(band_name)
        if asset is None:
            return None
        try:
            with rasterio.open(asset.href) as src:
                transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                x, y = transformer.transform(lon, lat)
                py_idx, px_idx = src.index(x, y)
                half = PATCH_SIZE // 2

                if not (half <= px_idx < src.width - half and
                        half <= py_idx < src.height - half):
                    return None

                window = rasterio.windows.Window(
                    px_idx - half, py_idx - half, PATCH_SIZE, PATCH_SIZE
                )
                data = src.read(1, window=window).astype(np.float32)
                data = data / 10000.0

                if data.shape != (PATCH_SIZE, PATCH_SIZE):
                    from scipy.ndimage import zoom
                    scale = PATCH_SIZE / max(data.shape)
                    data = zoom(data, scale, order=1)[:PATCH_SIZE, :PATCH_SIZE]

                bands_data.append(data)
        except Exception:
            return None

    if len(bands_data) != len(BANDS):
        return None

    image = np.stack(bands_data, axis=0)

    if np.isnan(image).any() or image.max() < 0.001:
        return None

    cloud_cover = item.properties.get("eo:cloud_cover", 0)
    actual_date = item.properties.get("datetime", date_str)[:10]

    np.savez_compressed(cache_path, image=image, cloud_cover=cloud_cover, actual_date=actual_date)
    return image, float(cloud_cover), actual_date


def process_station(station_id, lat, lon):
    """Collect pairs for one station across sample dates."""
    results = []
    for date_str in SAMPLE_DATES:
        if len(results) >= MAX_DATES_PER_STATION:
            break

        # Fetch WQ data first (fast API) — skip if no temperature
        params = fetch_nwis_daily(station_id, date_str)
        if params is None or 11 not in params:  # must have temperature
            continue

        # Download S2 tile (slower)
        try:
            tile_result = download_tile(lat, lon, date_str, station_id)
        except Exception:
            continue

        if tile_result is None:
            continue

        image, cloud_cover, actual_date = tile_result

        target = np.full(NUM_PARAMS, np.nan, dtype=np.float32)
        for pidx, val in params.items():
            target[pidx] = val

        meta = {
            "site_id": station_id,
            "date": date_str,
            "actual_date": actual_date,
            "lat": lat,
            "lon": lon,
            "cloud_cover": cloud_cover,
            "source": "USGS_NWIS",
        }
        results.append((image, target, meta))
        time.sleep(0.1)  # be nice to PC API

    return results


def main():
    log("=== USGS NWIS + Sentinel-2 v3 Data Collection ===")
    log(f"Target: {TARGET_NEW_PAIRS} new pairs from NWIS stations")

    # Step 1: Station IDs
    station_ids = get_station_ids()
    if not station_ids:
        log("ERROR: No station IDs in SEQ_DIR")
        return

    # Step 2: GPS coordinates
    log("Fetching station GPS from USGS...")
    coords = fetch_station_coords(station_ids)
    valid = [(sid, lat, lon) for sid, (lat, lon) in coords.items()]
    log(f"{len(valid)} stations with valid GPS, processing up to {MAX_STATIONS}")
    valid = valid[:MAX_STATIONS]

    # Step 3: Collect pairs (sequential — PC doesn't like bursts)
    all_images, all_targets, all_meta = [], [], []

    for i, (station_id, lat, lon) in enumerate(valid):
        try:
            results = process_station(station_id, lat, lon)
            for image, target, meta in results:
                all_images.append(image)
                all_targets.append(target)
                all_meta.append(meta)
        except Exception as e:
            pass

        if (i + 1) % 20 == 0:
            log(f"  {i+1}/{len(valid)} stations | {len(all_images)} pairs collected")

        if len(all_images) >= TARGET_NEW_PAIRS:
            log(f"  Reached target {TARGET_NEW_PAIRS} pairs at station {i+1}")
            break

        time.sleep(0.2)

    log(f"New pairs collected: {len(all_images)}")

    if not all_images:
        log("No new pairs — check network/API access")
        return

    # Step 4: Combine with existing 847 pairs
    existing_file = OUT_DIR / "paired_wq_expanded.npz"
    if existing_file.exists():
        log("Loading existing 847 pairs...")
        ex = np.load(existing_file, allow_pickle=True)
        ex_images = ex["images"]
        ex_targets = ex["targets"]
        try:
            ex_meta = json.loads(str(ex["metadata"]))
        except Exception:
            ex_meta = []
    else:
        ex_images = np.zeros((0, 10, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
        ex_targets = np.zeros((0, NUM_PARAMS), dtype=np.float32)
        ex_meta = []

    new_images = np.stack(all_images)
    new_targets = np.stack(all_targets)
    combined_images = np.concatenate([ex_images, new_images], axis=0)
    combined_targets = np.concatenate([ex_targets, new_targets], axis=0)
    combined_meta = (ex_meta if isinstance(ex_meta, list) else []) + all_meta

    log(f"Combined: {len(combined_images)} pairs "
        f"({len(ex_images)} existing + {len(new_images)} new NWIS)")

    # Step 5: Save
    out_file = OUT_DIR / "paired_wq_v3.npz"
    np.savez_compressed(
        out_file,
        images=combined_images,
        targets=combined_targets,
        metadata=json.dumps(combined_meta),
        param_names=json.dumps([
            "chl_a", "turbidity", "secchi_depth", "cdom", "tss",
            "total_nitrogen", "total_phosphorus", "dissolved_oxygen",
            "ammonia", "nitrate", "ph", "water_temp", "phycocyanin",
            "oil_probability", "acdom", "pollution_anomaly_index"
        ]),
    )

    # Stats
    params_names = ["chl_a","turb","secchi","cdom","TSS","TN","TP","DO","NH4","NO3","pH","temp","PC","oil","aCDOM","PAI"]
    log("Valid pairs per parameter in combined dataset:")
    for i, p in enumerate(params_names):
        n = int((~np.isnan(combined_targets[:, i])).sum())
        log(f"  {p}: {n}/{len(combined_targets)}")

    log(f"Saved {out_file}")

    # Save status
    status = {
        "n_existing": int(len(ex_images)),
        "n_new": int(len(new_images)),
        "n_combined": int(len(combined_images)),
        "stations_processed": i + 1,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (OUT_DIR / "nwis_s2_status.json").write_text(json.dumps(status, indent=2))
    log("DONE")


if __name__ == "__main__":
    main()
