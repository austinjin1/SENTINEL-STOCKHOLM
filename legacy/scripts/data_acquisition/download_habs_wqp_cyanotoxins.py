#!/usr/bin/env python3
"""
Download cyanotoxin data from WQP using WQX3.0 API.

The legacy API doesn't support some cyanotoxin names (hyphens cause 400 errors).
The WQX3.0 API handles them correctly. This script downloads cyanotoxin data
and merges it into the main HAB indicators parquet file.
"""

import os
import sys
import time
import logging
import warnings

import pandas as pd
from dataretrieval import wqp

BASE_DIR = "/home/bcheng/SENTINEL"
OUT_DIR = os.path.join(BASE_DIR, "data/processed/biology/noaa_habs")
LOG_FILE = os.path.join(BASE_DIR, "logs/download_habs_wqp.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a'),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

STATES = {
    "US:39": "Ohio",
    "US:12": "Florida",
    "US:06": "California",
    "US:26": "Michigan",
    "US:55": "Wisconsin",
    "US:36": "New York",
    "US:27": "Minnesota",
    "US:22": "Louisiana",
}

# Cyanotoxin names that work with WQX3.0 API
# Query each individually to avoid multi-name issues
CYANOTOXIN_NAMES = [
    "Microcystin",
    "Microcystin LR",
    "Microcystin-LR",
    "Microcystins and Nodularins",
    "Cylindrospermopsin",
    "Anatoxin-a",
    "Saxitoxin",
    "Total microcystins",
    "Cyanobacteria",
    "Blue-green algae (cyanobacteria), phycocyanin",
]

DATE_LO = "01-01-2015"
DATE_HI = "12-31-2025"


def main():
    logger.info("")
    logger.info("=" * 70)
    logger.info("WQP Cyanotoxin Download (WQX3.0 API)")
    logger.info("=" * 70)

    all_frames = []
    total_t0 = time.time()

    for char_name in CYANOTOXIN_NAMES:
        for statecode, state_name in STATES.items():
            t0 = time.time()
            try:
                df, md = wqp.get_results(
                    characteristicName=char_name,
                    statecode=statecode,
                    startDateLo=DATE_LO,
                    startDateHi=DATE_HI,
                    legacy=False,
                    dataProfile='narrow',
                )
                elapsed = time.time() - t0
                if len(df) > 0:
                    all_frames.append(df)
                    logger.info(f"  {char_name} | {state_name}: {len(df):,} records ({elapsed:.1f}s)")
                # Skip logging zeros to reduce noise
            except Exception as e:
                elapsed = time.time() - t0
                err = str(e)
                if "Bad Request" in err:
                    # Name doesn't exist in WQP - skip silently
                    pass
                elif "No data" in err or "204" in err:
                    pass
                else:
                    logger.warning(f"  {char_name} | {state_name}: ERROR - {err[:100]} ({elapsed:.1f}s)")

    if not all_frames:
        logger.info("No cyanotoxin data found via WQX3.0 API.")
        return

    # Combine
    logger.info("\nCombining cyanotoxin data...")
    cyano = pd.concat(all_frames, ignore_index=True)
    logger.info(f"Total cyanotoxin records: {len(cyano):,}")

    # Deduplicate
    dedup_cols = [c for c in [
        "Location_Identifier",
        "Activity_StartDate",
        "Characteristic_Name",
        "Result_Measure",
        "Activity_StartTime",
    ] if c in cyano.columns]

    if dedup_cols:
        before = len(cyano)
        cyano = cyano.drop_duplicates(subset=dedup_cols)
        logger.info(f"Dedup: {before:,} -> {len(cyano):,}")

    # Summary
    char_col = "Characteristic_Name" if "Characteristic_Name" in cyano.columns else "CharacteristicName"
    if char_col in cyano.columns:
        logger.info(f"\nCyanotoxin records by type:")
        for name, count in cyano[char_col].value_counts().items():
            logger.info(f"  {name}: {count:,}")

    # Fix object columns for parquet
    for col in cyano.select_dtypes(include="object").columns:
        cyano[col] = cyano[col].astype(str)

    # Save cyanotoxin-specific file
    cyano_path = os.path.join(OUT_DIR, "wqp_hab_cyanotoxins.parquet")
    cyano.to_parquet(cyano_path, index=False, engine="pyarrow")
    file_size = os.path.getsize(cyano_path) / (1024 * 1024)
    logger.info(f"\nSaved: {cyano_path}")
    logger.info(f"  Records: {len(cyano):,}")
    logger.info(f"  Columns: {cyano.shape[1]}")
    logger.info(f"  File size: {file_size:.2f} MB")

    # Note: WQX3.0 columns differ from legacy columns, so we save as a separate file
    # rather than trying to merge into the legacy-format main file.
    # The main file (wqp_hab_indicators.parquet) has chlorophyll-a + phycocyanin from legacy API.
    # This file (wqp_hab_cyanotoxins.parquet) has cyanotoxin data from WQX3.0 API.

    total_elapsed = time.time() - total_t0
    logger.info(f"\nCyanotoxin download time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    logger.info("Done.")


if __name__ == "__main__":
    main()
