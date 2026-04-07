#!/usr/bin/env python3
"""Expand HydroViT paired satellite-WQ dataset significantly.

Strategy: Instead of downloading tiles at fixed dates and hoping for WQ matches,
we flip the approach — find WQ measurements first (from GRQA + EPA WQP),
then download Sentinel-2 tiles at those exact locations and dates.

GRQA has 220K+ site-date pairs in the Sentinel-2 era (2015-2020) with
coordinates directly in the data. EPA WQP adds 2020-2024 coverage.

Target: 500-2000 paired samples (up from 74).

MIT License — Bryan Cheng, 2026
"""

import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

# ── Config ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
GRQA_DIR = DATA_DIR / "sentinel_db" / "grqa"
WQP_DIR = DATA_DIR / "raw" / "epa_wqp"
WQP_SITES_DIR = DATA_DIR / "raw" / "grqa" / "GRQA_source_data" / "WQP" / "raw" / "download_2020-11-16"
OUT_DIR = DATA_DIR / "processed" / "satellite" / "expanded"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"
BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
PATCH_SIZE = 224
MAX_CLOUD = 30
MAX_WORKERS = 6  # concurrent downloads
TARGET_PAIRS = 1500  # target number of paired samples

# GRQA parameter → target vector index mapping
GRQA_PARAM_MAP = {
    "TSS": 4,      # total suspended solids
    "TN": 5,       # total nitrogen
    "TP": 6,       # total phosphorus
    "DO": 7,       # dissolved oxygen
    "NH4N": 8,     # ammonia
    "NO3N": 9,     # nitrate
    "pH": 10,      # pH
    "TEMP": 11,    # water temperature
    "PC": 12,      # phycocyanin
}

# EPA WQP characteristic → target vector index mapping
WQP_PARAM_MAP = {
    "Chlorophyll a": 0,
    "Turbidity": 1,
    "Total suspended solids": 4,
    "Phosphorus": 6,
    "Dissolved oxygen (DO)": 7,
    "Ammonia": 8,
    "Nitrate": 9,
    "pH": 10,
    "Temperature, water": 11,
}

NUM_PARAMS = 16


def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ── Step 1: Extract WQ site-date pairs from GRQA ──────────────────────

def load_grqa_pairs():
    """Load all GRQA site-date pairs in the Sentinel-2 era (2015-2020)."""
    log("Loading GRQA data (2015-2020)...")
    records = []

    for param_name, param_idx in GRQA_PARAM_MAP.items():
        fpath = GRQA_DIR / f"{param_name}.parquet"
        if not fpath.exists():
            continue

        df = pd.read_parquet(fpath, columns=["site_id", "latitude", "longitude", "timestamp", "value"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["latitude", "longitude", "timestamp", "value"])

        # Filter to Sentinel-2 era
        df = df[(df["timestamp"] >= "2015-06-23") & (df["timestamp"] <= "2020-12-31")]
        df["date"] = df["timestamp"].dt.strftime("%Y-%m-%d")
        df["param_idx"] = param_idx
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"])

        records.append(df[["site_id", "latitude", "longitude", "date", "param_idx", "value"]])
        log(f"  GRQA {param_name}: {len(df)} records")

    if not records:
        return pd.DataFrame()

    combined = pd.concat(records, ignore_index=True)
    log(f"  Total GRQA records: {len(combined)}")
    return combined


def load_wqp_pairs():
    """Load EPA WQP site-date pairs with coordinates from lookup."""
    log("Loading EPA WQP data...")

    # Build coordinate lookup from site CSVs
    coord_lookup = {}
    if WQP_SITES_DIR.exists():
        for csv_file in WQP_SITES_DIR.glob("WQP_*_sites.csv"):
            try:
                df = pd.read_csv(csv_file, usecols=[
                    "MonitoringLocationIdentifier",
                    "LatitudeMeasure", "LongitudeMeasure"
                ], low_memory=False)
                df = df.dropna(subset=["LatitudeMeasure", "LongitudeMeasure"])
                for _, row in df.iterrows():
                    coord_lookup[row["MonitoringLocationIdentifier"]] = (
                        float(row["LatitudeMeasure"]),
                        float(row["LongitudeMeasure"])
                    )
            except Exception:
                continue
    log(f"  WQP coordinate lookup: {len(coord_lookup)} sites")

    if not coord_lookup:
        return pd.DataFrame()

    # Load WQP measurements and join with coordinates
    records = []
    target_chars = set(WQP_PARAM_MAP.keys())

    for parquet_file in sorted(WQP_DIR.glob("wqp_huc*.parquet")):
        try:
            df = pd.read_parquet(parquet_file, columns=[
                "MonitoringLocationIdentifier", "ActivityStartDate",
                "CharacteristicName", "ResultMeasureValue"
            ])
            df = df[df["CharacteristicName"].isin(target_chars)]
            df["value"] = pd.to_numeric(df["ResultMeasureValue"], errors="coerce")
            df = df.dropna(subset=["value"])
            df["date"] = pd.to_datetime(df["ActivityStartDate"], errors="coerce").dt.strftime("%Y-%m-%d")
            df = df.dropna(subset=["date"])

            # Join with coordinates
            df["coords"] = df["MonitoringLocationIdentifier"].map(coord_lookup)
            df = df.dropna(subset=["coords"])
            df["latitude"] = df["coords"].apply(lambda c: c[0])
            df["longitude"] = df["coords"].apply(lambda c: c[1])
            df["param_idx"] = df["CharacteristicName"].map(WQP_PARAM_MAP)
            df["site_id"] = df["MonitoringLocationIdentifier"]

            records.append(df[["site_id", "latitude", "longitude", "date", "param_idx", "value"]])
        except Exception:
            continue

    if not records:
        return pd.DataFrame()

    combined = pd.concat(records, ignore_index=True)
    # Filter to Sentinel-2 era (2015+)
    combined = combined[combined["date"] >= "2015-06-23"]
    log(f"  Total WQP records with coordinates: {len(combined)}")
    return combined


# ── Step 2: Select diverse site-date pairs ────────────────────────────

def select_pairs(grqa_df, wqp_df, target_n=TARGET_PAIRS):
    """Select diverse site-date pairs maximizing parameter coverage and geographic spread."""
    log("Selecting diverse site-date pairs...")

    # Combine sources
    dfs = []
    if len(grqa_df) > 0:
        grqa_df = grqa_df.copy()
        grqa_df["source"] = "grqa"
        dfs.append(grqa_df)
    if len(wqp_df) > 0:
        wqp_df = wqp_df.copy()
        wqp_df["source"] = "wqp"
        dfs.append(wqp_df)

    if not dfs:
        log("ERROR: No WQ data found!")
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)

    # Group by site+date to count parameter coverage per sample
    grouped = combined.groupby(["site_id", "date", "latitude", "longitude"]).agg(
        n_params=("param_idx", "nunique"),
        params=("param_idx", lambda x: list(set(x))),
    ).reset_index()

    log(f"  Total unique site-date pairs: {len(grouped)}")
    log(f"  Parameter coverage distribution:")
    for n in range(1, 10):
        count = (grouped["n_params"] == n).sum()
        if count > 0:
            log(f"    {n} params: {count} pairs")

    # Prioritize: (1) sites with most params, (2) optically active params (TSS=4, turbidity=1, chl_a=0)
    optical_params = {0, 1, 4, 12}  # chl_a, turbidity, tss, phycocyanin
    grouped["has_optical"] = grouped["params"].apply(lambda ps: len(set(ps) & optical_params) > 0)
    grouped["score"] = grouped["n_params"] * 2 + grouped["has_optical"].astype(int) * 3

    # Sort by score descending, then diversify geographically
    grouped = grouped.sort_values("score", ascending=False)

    # Take top candidates (3x target to allow for download failures)
    candidates = grouped.head(target_n * 3)

    # Geographic diversification: grid the world into 1° cells, max 5 per cell
    candidates["lat_bin"] = (candidates["latitude"] // 1).astype(int)
    candidates["lon_bin"] = (candidates["longitude"] // 1).astype(int)
    candidates["cell"] = candidates["lat_bin"].astype(str) + "_" + candidates["lon_bin"].astype(str)

    selected = []
    cell_counts = {}
    max_per_cell = 8

    for _, row in candidates.iterrows():
        cell = row["cell"]
        cell_counts.setdefault(cell, 0)
        if cell_counts[cell] < max_per_cell:
            selected.append(row)
            cell_counts[cell] += 1
            if len(selected) >= target_n:
                break

    result = pd.DataFrame(selected)
    log(f"  Selected {len(result)} pairs across {len(cell_counts)} geographic cells")
    log(f"  Mean params per pair: {result['n_params'].mean():.1f}")
    log(f"  Pairs with optical params: {result['has_optical'].sum()}")
    return result


# ── Step 3: Download Sentinel-2 tiles ─────────────────────────────────

def download_tile(lat, lon, date_str, station_id, catalog):
    """Download a single S2 tile. Returns (image, cloud_cover) or None."""
    import planetary_computer
    import pystac_client
    import rasterio

    bbox = [lon - 0.015, lat - 0.015, lon + 0.015, lat + 0.015]

    from datetime import datetime, timedelta
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    start = (dt - timedelta(days=15)).strftime("%Y-%m-%d")
    end = (dt + timedelta(days=15)).strftime("%Y-%m-%d")

    search = catalog.search(
        collections=[COLLECTION],
        bbox=bbox,
        datetime=f"{start}/{end}",
        query={"eo:cloud_cover": {"lt": MAX_CLOUD}},
        max_items=3,
    )

    items = list(search.items())
    if not items:
        return None

    # Pick item closest in time
    target_ts = dt.timestamp()
    items.sort(key=lambda it: abs(
        datetime.fromisoformat(it.properties["datetime"].replace("Z", "+00:00")).timestamp() - target_ts
    ))
    item = items[0]
    signed_item = planetary_computer.sign(item)

    from pyproj import Transformer
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

                if not (half <= px_idx < src.width - half and half <= py_idx < src.height - half):
                    return None

                window = rasterio.windows.Window(
                    px_idx - half, py_idx - half, PATCH_SIZE, PATCH_SIZE
                )
                data = src.read(1, window=window).astype(np.float32)
                data = data / 10000.0  # S2 L2A scale factor

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

    # Basic quality check: reject all-zero or all-nan tiles
    if np.isnan(image).any() or image.max() < 0.001:
        return None

    cloud_cover = item.properties.get("eo:cloud_cover", 0)
    actual_date = item.properties.get("datetime", date_str)[:10]

    return image, cloud_cover, actual_date


def download_worker(args):
    """Worker function for parallel downloads."""
    idx, row, catalog = args
    site_id = str(row["site_id"])
    lat, lon = float(row["latitude"]), float(row["longitude"])
    date_str = str(row["date"])

    out_path = OUT_DIR / f"s2_{site_id}_{date_str}.npz"
    if out_path.exists():
        return idx, "exists", out_path

    try:
        result = download_tile(lat, lon, date_str, site_id, catalog)
        if result is None:
            return idx, "no_tile", None

        image, cloud_cover, actual_date = result
        np.savez_compressed(
            out_path,
            image=image,
            latitude=lat,
            longitude=lon,
            station_id=site_id,
            date=actual_date,
            request_date=date_str,
            cloud_cover=cloud_cover,
            bands=BANDS,
        )
        return idx, "ok", out_path
    except Exception as e:
        return idx, f"error:{str(e)[:50]}", None


# ── Step 4: Build paired dataset ──────────────────────────────────────

def build_paired_dataset(selected_pairs, wq_records):
    """Build the final paired_wq.npz with expanded data."""
    log("Building expanded paired dataset...")

    # Load all downloaded tiles
    tile_files = sorted(OUT_DIR.glob("s2_*.npz"))
    log(f"  Found {len(tile_files)} downloaded tiles")

    # Also include existing tiles from the original set
    orig_dir = DATA_DIR / "processed" / "satellite" / "real"
    orig_tiles = sorted(orig_dir.glob("s2_*.npz")) if orig_dir.exists() else []
    log(f"  Found {len(orig_tiles)} original tiles")

    # Build lookup: (site_id, date) → WQ target vector
    wq_lookup = {}
    for _, row in wq_records.iterrows():
        key = (str(row["site_id"]), str(row["date"]))
        if key not in wq_lookup:
            wq_lookup[key] = np.full(NUM_PARAMS, np.nan, dtype=np.float32)
        wq_lookup[key][int(row["param_idx"])] = float(row["value"])

    images = []
    targets = []
    metadata = []

    for tile_path in tile_files + orig_tiles:
        try:
            tile = np.load(tile_path, allow_pickle=True)
            site_id = str(tile["station_id"])
            date = str(tile["date"])
            request_date = str(tile.get("request_date", date))

            # Try exact date match first, then request date
            target = wq_lookup.get((site_id, date))
            if target is None:
                target = wq_lookup.get((site_id, request_date))
            if target is None:
                # Try ±3 day window
                from datetime import datetime, timedelta
                try:
                    dt = datetime.strptime(date, "%Y-%m-%d")
                    for offset in range(-3, 4):
                        check_date = (dt + timedelta(days=offset)).strftime("%Y-%m-%d")
                        target = wq_lookup.get((site_id, check_date))
                        if target is not None:
                            break
                except Exception:
                    pass

            if target is None:
                continue

            # Check we have at least 1 non-NaN param
            if np.all(np.isnan(target)):
                continue

            image = tile["image"]
            if image.shape != (10, PATCH_SIZE, PATCH_SIZE):
                continue

            images.append(image)
            targets.append(target)
            metadata.append({
                "site_id": site_id,
                "date": date,
                "lat": float(tile["latitude"]),
                "lon": float(tile["longitude"]),
                "cloud_cover": float(tile.get("cloud_cover", 0)),
            })
        except Exception:
            continue

    if not images:
        log("ERROR: No paired samples found!")
        return 0

    images = np.stack(images, axis=0)  # [N, 10, 224, 224]
    targets = np.stack(targets, axis=0)  # [N, 16]

    # Save expanded dataset
    out_path = DATA_DIR / "processed" / "satellite" / "paired_wq_expanded.npz"
    np.savez_compressed(
        out_path,
        images=images,
        targets=targets,
        metadata=json.dumps(metadata),
        param_names=json.dumps([
            "chl_a", "turbidity", "secchi_depth", "cdom", "tss",
            "total_nitrogen", "total_phosphorus", "dissolved_oxygen",
            "ammonia", "nitrate", "ph", "water_temp", "phycocyanin",
            "oil_probability", "acdom", "pollution_anomaly_index"
        ]),
    )

    log(f"  Saved {len(images)} paired samples to {out_path}")
    log(f"  Image shape: {images.shape}")
    log(f"  Target shape: {targets.shape}")

    # Print per-parameter coverage
    log("  Per-parameter coverage:")
    param_names = ["chl_a", "turbidity", "secchi", "cdom", "tss", "TN", "TP",
                   "DO", "NH4N", "NO3N", "pH", "temp", "PC", "oil", "acdom", "PAI"]
    for i, name in enumerate(param_names):
        n_valid = np.sum(~np.isnan(targets[:, i]))
        pct = 100 * n_valid / len(targets)
        if n_valid > 0:
            log(f"    {name}: {n_valid}/{len(targets)} ({pct:.0f}%)")

    return len(images)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    log("=" * 60)
    log("Expanding HydroViT paired satellite-WQ dataset")
    log("=" * 60)

    # Step 1: Load WQ data
    grqa_df = load_grqa_pairs()
    wqp_df = load_wqp_pairs()

    total_wq = len(grqa_df) + len(wqp_df)
    if total_wq == 0:
        log("ERROR: No WQ data found!")
        return

    # Step 2: Select diverse pairs
    selected = select_pairs(grqa_df, wqp_df, target_n=TARGET_PAIRS)
    if len(selected) == 0:
        log("ERROR: No pairs selected!")
        return

    # Step 3: Download tiles
    log(f"\nDownloading Sentinel-2 tiles for {len(selected)} site-date pairs...")
    log(f"Using {MAX_WORKERS} parallel workers")

    import pystac_client
    catalog = pystac_client.Client.open(STAC_URL)

    # Check existing
    existing = set(f.stem for f in OUT_DIR.glob("s2_*.npz"))
    to_download = []
    skipped = 0
    for idx, row in selected.iterrows():
        key = f"s2_{row['site_id']}_{row['date']}"
        if key in existing:
            skipped += 1
        else:
            to_download.append((idx, row, catalog))

    log(f"  Already downloaded: {skipped}")
    log(f"  To download: {len(to_download)}")

    downloaded = 0
    failed = 0
    t0 = time.time()

    # Sequential download with rate limiting (Planetary Computer needs this)
    for i, (idx, row, cat) in enumerate(to_download):
        result_idx, status, path = download_worker((idx, row, cat))

        if status == "ok":
            downloaded += 1
        elif status == "exists":
            skipped += 1
        else:
            failed += 1

        if (i + 1) % 25 == 0 or (i + 1) == len(to_download):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(to_download) - i - 1) / rate if rate > 0 else 0
            log(f"  Progress: {i+1}/{len(to_download)} "
                f"(ok={downloaded}, skip={skipped}, fail={failed}) "
                f"[{rate:.1f}/s, ETA {eta/60:.0f}m]")

        time.sleep(0.3)  # Rate limit

    log(f"\nDownload complete: {downloaded} new + {skipped} existing, {failed} failed")
    log(f"Total time: {(time.time()-t0)/60:.1f} minutes")

    # Step 4: Build paired dataset
    all_wq = pd.concat([grqa_df, wqp_df], ignore_index=True) if len(wqp_df) > 0 else grqa_df
    n_paired = build_paired_dataset(selected, all_wq)
    log(f"\nFinal paired dataset: {n_paired} samples")
    log("=" * 60)


if __name__ == "__main__":
    main()
