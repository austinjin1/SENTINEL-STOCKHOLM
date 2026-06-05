#!/usr/bin/env python3
"""Co-register Sentinel-2 satellite imagery with in-situ water quality measurements.

Matches satellite tiles to EPA WQP and GRQA ground-truth measurements by
spatial proximity (5 km) and temporal proximity (+-3 days).

Outputs data/processed/satellite/paired_wq.npz with:
  - images: [N, 10, 224, 224]  float32
  - targets: [N, 16]           float32  (NaN where no measurement)
  - tile_indices: [N]          indices into original tile list
  - metadata: dict with tile info

MIT License -- Bryan Cheng, 2026
"""

import glob
import math
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from sentinel.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SATELLITE_DIR = Path("data/processed/satellite/real")
EPA_DIR = Path("data/raw/epa_wqp")
GRQA_DIR = Path("data/sentinel_db/grqa")
WQP_SITES_DIR = Path("data/raw/grqa/GRQA_source_data/WQP/raw/download_2020-11-16")
OUTPUT_PATH = Path("data/processed/satellite/paired_wq.npz")

SPATIAL_RADIUS_KM = 5.0
TEMPORAL_WINDOW_DAYS = 3

# Parameter mapping: HydroViT index -> (EPA CharacteristicName list, GRQA filename, unit_conversion)
# HydroViT params (from parameter_head.py):
#  0: chl_a, 1: turbidity, 2: secchi_depth, 3: cdom, 4: tss,
#  5: total_nitrogen, 6: total_phosphorus, 7: dissolved_oxygen,
#  8: ammonia, 9: nitrate, 10: ph, 11: water_temp
#  12: phycocyanin, 13: oil_probability, 14: acdom, 15: pollution_anomaly_index

PARAM_MAP = {
    0:  {"epa": ["Chlorophyll a"],
         "grqa": None},
    1:  {"epa": ["Turbidity"],
         "grqa": None},
    2:  {"epa": [],
         "grqa": None},  # Secchi depth - rarely in these datasets
    3:  {"epa": [],
         "grqa": None},  # CDOM
    4:  {"epa": ["Total suspended solids"],
         "grqa": "TSS.parquet"},
    5:  {"epa": [],
         "grqa": "TN.parquet"},
    6:  {"epa": ["Phosphorus"],
         "grqa": "TP.parquet"},
    7:  {"epa": ["Dissolved oxygen (DO)"],
         "grqa": "DO.parquet"},
    8:  {"epa": ["Ammonia"],
         "grqa": "NH4N.parquet"},
    9:  {"epa": ["Nitrate"],
         "grqa": "NO3N.parquet"},
    10: {"epa": ["pH"],
         "grqa": "pH.parquet"},
    11: {"epa": ["Temperature, water"],
         "grqa": "TEMP.parquet"},
    12: {"epa": [],
         "grqa": "PC.parquet"},  # Phycocyanin
    13: {"epa": [],
         "grqa": None},  # Oil probability - not in these datasets
    14: {"epa": [],
         "grqa": None},  # ACDOM
    15: {"epa": [],
         "grqa": None},  # Pollution anomaly index
}


def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance in km."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def load_satellite_metadata():
    """Load lat, lon, date from all satellite tiles."""
    files = sorted(glob.glob(str(SATELLITE_DIR / "*.npz")))
    logger.info(f"Found {len(files)} satellite tiles")

    records = []
    for i, f in enumerate(files):
        d = np.load(f, allow_pickle=True)
        records.append({
            "idx": i,
            "file": f,
            "lat": float(d["latitude"]),
            "lon": float(d["longitude"]),
            "date": str(d["date"]),
            "station_id": str(d["station_id"]),
        })
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    logger.info(f"Satellite tiles: lat [{df.lat.min():.2f}, {df.lat.max():.2f}], "
                f"lon [{df.lon.min():.2f}, {df.lon.max():.2f}], "
                f"dates [{df.date.min().date()}, {df.date.max().date()}]")
    return df


def build_wqp_station_lookup():
    """Build MonitoringLocationIdentifier -> (lat, lon) from WQP site files."""
    site_files = sorted(glob.glob(str(WQP_SITES_DIR / "*_sites.csv")))
    logger.info(f"Loading station coordinates from {len(site_files)} WQP site files")
    dfs = []
    for f in site_files:
        try:
            df = pd.read_csv(f,
                             usecols=["MonitoringLocationIdentifier",
                                      "LatitudeMeasure", "LongitudeMeasure"],
                             low_memory=False)
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Skipping {f}: {e}")
    if not dfs:
        return {}
    all_sites = pd.concat(dfs).drop_duplicates("MonitoringLocationIdentifier")
    all_sites = all_sites.dropna(subset=["LatitudeMeasure", "LongitudeMeasure"])
    lookup = {}
    for _, row in all_sites.iterrows():
        lookup[row["MonitoringLocationIdentifier"]] = (
            float(row["LatitudeMeasure"]),
            float(row["LongitudeMeasure"]),
        )
    logger.info(f"Built station lookup: {len(lookup)} unique stations with coordinates")
    return lookup


def load_epa_wqp_data(station_lookup):
    """Load all EPA WQP parquet files and merge with station coordinates."""
    files = sorted(glob.glob(str(EPA_DIR / "wqp_huc*.parquet")))
    logger.info(f"Loading {len(files)} EPA WQP parquet files")

    # Only load characteristics we care about
    target_chars = set()
    for pmap in PARAM_MAP.values():
        target_chars.update(pmap["epa"])

    dfs = []
    for f in files:
        df = pd.read_parquet(f)
        # Filter to relevant characteristics
        df = df[df["CharacteristicName"].isin(target_chars)]
        # Filter to valid numeric values
        df["ResultMeasureValue"] = pd.to_numeric(df["ResultMeasureValue"], errors="coerce")
        df = df.dropna(subset=["ResultMeasureValue"])
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    epa = pd.concat(dfs, ignore_index=True)
    logger.info(f"EPA WQP: {len(epa)} records after filtering")

    # Resolve coordinates from station lookup
    lats, lons = [], []
    for sid in epa["MonitoringLocationIdentifier"]:
        if sid in station_lookup:
            lat, lon = station_lookup[sid]
            lats.append(lat)
            lons.append(lon)
        else:
            lats.append(np.nan)
            lons.append(np.nan)
    epa["lat"] = lats
    epa["lon"] = lons
    epa = epa.dropna(subset=["lat", "lon"])
    epa["date"] = pd.to_datetime(epa["ActivityStartDate"], errors="coerce")
    epa = epa.dropna(subset=["date"])
    logger.info(f"EPA WQP with coordinates: {len(epa)} records")
    return epa


def load_grqa_data():
    """Load relevant GRQA parquet files."""
    dfs = []
    for param_idx, pmap in PARAM_MAP.items():
        grqa_file = pmap["grqa"]
        if grqa_file is None:
            continue
        fpath = GRQA_DIR / grqa_file
        if not fpath.exists():
            logger.warning(f"GRQA file not found: {fpath}")
            continue
        df = pd.read_parquet(fpath)
        df = df.dropna(subset=["latitude", "longitude", "value", "timestamp"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"])
        df["date"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["date"])
        df["param_idx"] = param_idx
        df = df[["latitude", "longitude", "date", "value", "param_idx"]].rename(
            columns={"latitude": "lat", "longitude": "lon"}
        )
        dfs.append(df)
        logger.info(f"  GRQA {grqa_file}: {len(df)} records")

    if not dfs:
        return pd.DataFrame()
    grqa = pd.concat(dfs, ignore_index=True)
    logger.info(f"GRQA total: {len(grqa)} records")
    return grqa


def build_epa_param_index(epa_df):
    """Map EPA CharacteristicName -> param_idx."""
    char_to_idx = {}
    for param_idx, pmap in PARAM_MAP.items():
        for char_name in pmap["epa"]:
            char_to_idx[char_name] = param_idx
    return char_to_idx


def coregister(sat_df, epa_df, grqa_df):
    """Spatially and temporally match satellite tiles with in-situ measurements.

    For each satellite tile, find all in-situ measurements within 5km and +-3 days.
    When multiple measurements exist for a parameter, take the median.
    """
    logger.info("=" * 60)
    logger.info("CO-REGISTRATION")
    logger.info(f"  Spatial radius: {SPATIAL_RADIUS_KM} km")
    logger.info(f"  Temporal window: +/-{TEMPORAL_WINDOW_DAYS} days")
    logger.info("=" * 60)

    # Build unified measurement table: (lat, lon, date, param_idx, value)
    measurements = []

    # EPA measurements
    if len(epa_df) > 0:
        char_to_idx = build_epa_param_index(epa_df)
        epa_meas = epa_df[epa_df["CharacteristicName"].isin(char_to_idx)].copy()
        epa_meas["param_idx"] = epa_meas["CharacteristicName"].map(char_to_idx)
        epa_meas = epa_meas[["lat", "lon", "date", "param_idx", "ResultMeasureValue"]].rename(
            columns={"ResultMeasureValue": "value"}
        )
        measurements.append(epa_meas)
        logger.info(f"  EPA measurements: {len(epa_meas)}")

    # GRQA measurements
    if len(grqa_df) > 0:
        measurements.append(grqa_df)
        logger.info(f"  GRQA measurements: {len(grqa_df)}")

    if not measurements:
        logger.error("No in-situ measurements available!")
        return [], {}

    all_meas = pd.concat(measurements, ignore_index=True)
    # Remove extreme outliers (negative values for params that should be positive)
    all_meas = all_meas[all_meas["value"] > -1e6]
    all_meas = all_meas[all_meas["value"] < 1e6]
    logger.info(f"  Total measurements: {len(all_meas)}")

    # Build spatial index using KD-tree on measurement locations
    # Convert to radians for approximate euclidean distance
    meas_lat_rad = np.radians(all_meas["lat"].values)
    meas_lon_rad = np.radians(all_meas["lon"].values)
    R_EARTH = 6371.0
    meas_x = R_EARTH * np.cos(meas_lat_rad) * meas_lon_rad
    meas_y = R_EARTH * meas_lat_rad
    meas_coords = np.column_stack([meas_x, meas_y])
    tree = cKDTree(meas_coords)

    meas_dates = all_meas["date"].values  # numpy datetime64
    meas_param_idx = all_meas["param_idx"].values
    meas_values = all_meas["value"].values

    paired = []
    n_with_data = 0
    param_counts = np.zeros(16, dtype=int)

    for row_idx, row in sat_df.iterrows():
        tile_lat, tile_lon = row["lat"], row["lon"]
        tile_date = row["date"]

        # Convert tile location to same coordinate system
        lat_rad = np.radians(tile_lat)
        lon_rad = np.radians(tile_lon)
        x = R_EARTH * np.cos(lat_rad) * lon_rad
        y = R_EARTH * lat_rad

        # Query KD-tree for points within SPATIAL_RADIUS_KM
        candidate_indices = tree.query_ball_point([x, y], r=SPATIAL_RADIUS_KM)
        if not candidate_indices:
            continue

        # Filter by temporal window
        td_window = np.timedelta64(TEMPORAL_WINDOW_DAYS, "D")
        cand_dates = meas_dates[candidate_indices]
        tile_date_np = np.datetime64(tile_date)
        time_mask = np.abs(cand_dates - tile_date_np) <= td_window
        matched_indices = np.array(candidate_indices)[time_mask]

        if len(matched_indices) == 0:
            continue

        # Aggregate: median per parameter
        targets = np.full(16, np.nan, dtype=np.float32)
        for mi in matched_indices:
            pidx = int(meas_param_idx[mi])
            val = float(meas_values[mi])
            if np.isnan(targets[pidx]):
                targets[pidx] = val
            else:
                # Running list not efficient, just accumulate (will re-scan below)
                pass

        # Re-scan to compute median for each param
        for pidx in range(16):
            vals = []
            for mi in matched_indices:
                if int(meas_param_idx[mi]) == pidx:
                    vals.append(float(meas_values[mi]))
            if vals:
                targets[pidx] = np.median(vals)
                param_counts[pidx] += 1

        n_params_found = np.sum(~np.isnan(targets))
        if n_params_found > 0:
            paired.append({
                "tile_idx": row["idx"],
                "tile_file": row["file"],
                "targets": targets,
                "n_params": n_params_found,
            })
            n_with_data += 1

    logger.info(f"Co-registered tiles: {n_with_data} / {len(sat_df)} ({100*n_with_data/len(sat_df):.1f}%)")
    logger.info("Parameters matched per param index:")
    from sentinel.models.satellite_encoder.parameter_head import PARAM_NAMES
    for i, name in enumerate(PARAM_NAMES):
        logger.info(f"  {i:2d} {name:>25s}: {param_counts[i]:5d} tiles")

    return paired, {"param_counts": param_counts}


def save_paired_dataset(paired):
    """Save paired (image, targets) dataset."""
    if not paired:
        logger.error("No paired samples to save!")
        return

    logger.info(f"Saving {len(paired)} paired samples to {OUTPUT_PATH}")

    images = []
    targets = []
    tile_indices = []
    tile_files = []

    for p in paired:
        f = np.load(p["tile_file"], allow_pickle=True)
        img = f["image"].astype(np.float32)  # (10, 224, 224)
        images.append(img)
        targets.append(p["targets"])
        tile_indices.append(p["tile_idx"])
        tile_files.append(p["tile_file"])

    images = np.stack(images)      # (N, 10, 224, 224)
    targets = np.stack(targets)    # (N, 16)
    tile_indices = np.array(tile_indices)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_PATH,
        images=images,
        targets=targets,
        tile_indices=tile_indices,
        tile_files=np.array(tile_files, dtype=str),
    )
    logger.info(f"Saved: images {images.shape}, targets {targets.shape}")
    logger.info(f"Non-NaN target density: {(~np.isnan(targets)).sum() / targets.size:.3f}")

    # Per-param stats
    from sentinel.models.satellite_encoder.parameter_head import PARAM_NAMES
    for i, name in enumerate(PARAM_NAMES):
        col = targets[:, i]
        valid = col[~np.isnan(col)]
        if len(valid) > 0:
            logger.info(f"  {name:>25s}: n={len(valid):5d}, "
                        f"mean={valid.mean():.3f}, std={valid.std():.3f}, "
                        f"min={valid.min():.3f}, max={valid.max():.3f}")


def main():
    t0 = time.time()

    # Step 1: Load satellite metadata
    sat_df = load_satellite_metadata()

    # Step 2: Build station coordinate lookup
    station_lookup = build_wqp_station_lookup()

    # Step 3: Load in-situ data
    epa_df = load_epa_wqp_data(station_lookup)
    grqa_df = load_grqa_data()

    # Step 4: Co-register
    paired, stats = coregister(sat_df, epa_df, grqa_df)

    # Step 5: Save
    save_paired_dataset(paired)

    elapsed = time.time() - t0
    logger.info(f"Total time: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
