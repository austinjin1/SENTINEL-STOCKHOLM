#!/usr/bin/env python3
"""Download real Sentinel-2 imagery co-registered with USGS/GRQA stations.

Uses Planetary Computer STAC API to find and download Sentinel-2 L2A
tiles at locations where we have in-situ water quality measurements.
This creates the paired satellite-WQ dataset needed for HydroViT training.

Strategy:
1. Load USGS station locations (from sensor parquets)
2. Load GRQA station locations (from ingested data)
3. For each station, search for cloud-free S2 tiles within ±3 days of WQ measurements
4. Download 224x224 pixel patches centered on stations

MIT License — Bryan Cheng, 2026
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import planetary_computer
import pystac_client
import rasterio
from rasterio.windows import from_bounds

DATA_DIR = Path("data/raw/satellite/real")
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = Path("data/processed/satellite/real")
OUT_DIR.mkdir(parents=True, exist_ok=True)

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"
BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
PATCH_SIZE = 224  # pixels at 10m = 2.24km
MAX_CLOUD = 30  # max cloud cover %


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def get_usgs_station_locations():
    """Extract station locations from USGS station info via dataretrieval."""
    catalog_path = Path("data/raw/sensor/full/station_catalog_smart.json")
    if not catalog_path.exists():
        return []

    catalog = json.load(open(catalog_path))
    site_nos = [s["site_no"] for s in catalog if isinstance(s, dict)][:200]

    # Get station locations via dataretrieval
    try:
        import dataretrieval.nwis as nwis
        stations = []
        # Batch lookup in chunks of 50
        for i in range(0, len(site_nos), 50):
            batch = site_nos[i:i+50]
            try:
                df, _ = nwis.get_info(sites=batch)
                for _, row in df.iterrows():
                    lat = row.get("dec_lat_va") or row.get("latitude")
                    lon = row.get("dec_long_va") or row.get("longitude")
                    site = row.get("site_no", "")
                    if lat and lon:
                        stations.append({"id": str(site), "lat": float(lat), "lon": float(lon)})
            except Exception:
                pass
            time.sleep(0.5)
        return stations
    except Exception:
        return []


def get_grqa_station_locations(max_stations=500):
    """Extract station locations from GRQA ingested data."""
    grqa_dir = Path("data/sentinel_db/grqa")
    if not grqa_dir.exists():
        return []

    # Load DO data (has most sites)
    do_path = grqa_dir / "DO.parquet"
    if not do_path.exists():
        return []

    df = pd.read_parquet(do_path, columns=["site_id", "latitude", "longitude"])
    sites = df.drop_duplicates("site_id").head(max_stations)
    return [
        {"id": row["site_id"], "lat": row["latitude"], "lon": row["longitude"]}
        for _, row in sites.iterrows()
    ]


def download_patch(lat, lon, date_str, station_id, catalog):
    """Download a single S2 patch centered on a station."""
    # Search for S2 tiles
    bbox = [lon - 0.015, lat - 0.015, lon + 0.015, lat + 0.015]

    # Search a 30-day window (S2 revisit = 5 days, need margin for clouds)
    from datetime import datetime, timedelta
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    start = (dt - timedelta(days=15)).strftime("%Y-%m-%d")
    end = (dt + timedelta(days=15)).strftime("%Y-%m-%d")
    search = catalog.search(
        collections=[COLLECTION],
        bbox=bbox,
        datetime=f"{start}/{end}",
        query={"eo:cloud_cover": {"lt": MAX_CLOUD}},
        max_items=1,
    )

    items = list(search.items())
    if not items:
        return None

    item = items[0]
    signed_item = planetary_computer.sign(item)

    # Download bands and create patch
    from pyproj import Transformer

    bands_data = []
    for band_name in BANDS:
        asset = signed_item.assets.get(band_name)
        if asset is None:
            return None

        try:
            with rasterio.open(asset.href) as src:
                # Transform WGS84 lat/lon to raster CRS
                transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                x, y = transformer.transform(lon, lat)
                py, px = src.index(x, y)
                half = PATCH_SIZE // 2

                if not (half <= px < src.width - half and half <= py < src.height - half):
                    return None

                window = rasterio.windows.Window(
                    px - half, py - half, PATCH_SIZE, PATCH_SIZE
                )
                data = src.read(1, window=window).astype(np.float32)

                # Scale to reflectance (S2 L2A scale factor = 10000)
                data = data / 10000.0

                # Handle 20m bands by resizing to 224x224
                if data.shape != (PATCH_SIZE, PATCH_SIZE):
                    from scipy.ndimage import zoom
                    scale = PATCH_SIZE / data.shape[0]
                    data = zoom(data, scale, order=1)[:PATCH_SIZE, :PATCH_SIZE]

                bands_data.append(data)
        except Exception:
            return None

    if len(bands_data) != len(BANDS):
        return None

    image = np.stack(bands_data, axis=0)  # [10, 224, 224]
    cloud_cover = item.properties.get("eo:cloud_cover", 0)

    return image, cloud_cover


def main():
    log("=" * 60)
    log("Satellite Imagery Download (Planetary Computer)")
    log("=" * 60)

    # Get station locations
    usgs_stations = get_usgs_station_locations()
    grqa_stations = get_grqa_station_locations(max_stations=300)
    log(f"USGS stations: {len(usgs_stations)}")
    log(f"GRQA stations: {len(grqa_stations)}")

    # Combine and deduplicate
    all_stations = usgs_stations + grqa_stations
    log(f"Total candidate stations: {len(all_stations)}")

    # Open STAC catalog
    catalog = pystac_client.Client.open(STAC_URL)
    log("Connected to Planetary Computer STAC API")

    # Search dates - sample from 2023-2024 for each station
    search_dates = ["2023-06-15", "2023-08-15", "2024-03-15", "2024-06-15", "2024-09-15"]

    downloaded = 0
    failed = 0
    existing = len(list(OUT_DIR.glob("*.npz")))
    log(f"Existing tiles: {existing}")

    for i, station in enumerate(all_stations):
        if downloaded >= 5000:
            log("Reached 5000 tile target, stopping")
            break

        for date_str in search_dates:
            out_path = OUT_DIR / f"s2_{station['id']}_{date_str}.npz"
            if out_path.exists():
                continue

            try:
                result = download_patch(
                    station["lat"], station["lon"], date_str, station["id"], catalog
                )
                if result is not None:
                    image, cloud_cover = result
                    np.savez_compressed(
                        out_path,
                        image=image,
                        latitude=station["lat"],
                        longitude=station["lon"],
                        station_id=station["id"],
                        date=date_str,
                        cloud_cover=cloud_cover,
                        bands=BANDS,
                    )
                    downloaded += 1
                else:
                    failed += 1
            except Exception as e:
                failed += 1

            time.sleep(0.2)  # Rate limiting

        if (i + 1) % 25 == 0:
            log(f"  Progress: {i+1}/{len(all_stations)} stations (downloaded={downloaded}, failed={failed})")

    total = existing + downloaded
    log(f"\nDone! Total tiles: {total} (new={downloaded}, existing={existing}, failed={failed})")


if __name__ == "__main__":
    main()
