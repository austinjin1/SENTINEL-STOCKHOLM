#!/usr/bin/env python3
"""
Download HAB-related water quality data from USGS Water Quality Portal (WQP).

Alternative to NOAA HABSOS which is DNS-blocked on this cluster.
Downloads chlorophyll-a and cyanotoxin/microcystin data for HAB-prone states.

Output: /home/bcheng/SENTINEL/data/processed/biology/noaa_habs/wqp_hab_indicators.parquet
"""

import os
import sys
import time
import logging
import warnings
from datetime import datetime

import pandas as pd
from dataretrieval import wqp

# Setup
BASE_DIR = "/home/bcheng/SENTINEL"
OUT_DIR = os.path.join(BASE_DIR, "data/processed/biology/noaa_habs")
LOG_FILE = os.path.join(BASE_DIR, "logs/download_habs_wqp.log")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# Configure logging to both file and stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)

# Target states for HAB data (FIPS codes)
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

# HAB indicator characteristic names to query
# Group 1: Chlorophyll-a variants
CHLOROPHYLL_CHARS = [
    "Chlorophyll a",
    "Chlorophyll a, corrected for pheophytin",
    "Chlorophyll a, free of pheophytin",
    "Chlorophyll a (probe relative fluorescence)",
    "Chlorophyll a, uncorrected for pheophytin",
]

# Group 2: Cyanotoxins / microcystin
CYANOTOXIN_CHARS = [
    "Microcystin",
    "Microcystin LR",
    "Microcystins and Nodularins",
    "Cylindrospermopsin",
    "Anatoxin-a",
    "Saxitoxin",
]

# Group 3: Phycocyanin (cyanobacteria pigment indicator)
PHYCOCYANIN_CHARS = [
    "Phycocyanin",
    "Phycocyanin (probe relative fluorescence)",
]

# Combine all characteristic names as semicolon-separated string (WQP API format)
ALL_CHARS_LIST = CHLOROPHYLL_CHARS + CYANOTOXIN_CHARS + PHYCOCYANIN_CHARS
ALL_CHARS = ";".join(ALL_CHARS_LIST)
CHLOROPHYLL_CHARS_STR = ";".join(CHLOROPHYLL_CHARS)

DATE_LO = "01-01-2015"
DATE_HI = "12-31-2025"


def download_state(statecode: str, state_name: str) -> pd.DataFrame:
    """Download HAB indicator data for a single state."""
    logger.info(f"  Querying {state_name} ({statecode})...")
    t0 = time.time()

    try:
        df, md = wqp.get_results(
            characteristicName=ALL_CHARS,
            statecode=statecode,
            startDateLo=DATE_LO,
            startDateHi=DATE_HI,
        )
        elapsed = time.time() - t0
        logger.info(f"  {state_name}: {len(df):,} records in {elapsed:.1f}s")
        return df
    except Exception as e:
        logger.error(f"  {state_name} FAILED: {e}")
        # Retry with just chlorophyll-a (smaller query)
        try:
            logger.info(f"  Retrying {state_name} with chlorophyll-a only...")
            df, md = wqp.get_results(
                characteristicName=CHLOROPHYLL_CHARS_STR,
                statecode=statecode,
                startDateLo=DATE_LO,
                startDateHi=DATE_HI,
            )
            elapsed = time.time() - t0
            logger.info(f"  {state_name} (retry): {len(df):,} records in {elapsed:.1f}s")
            return df
        except Exception as e2:
            logger.error(f"  {state_name} retry also FAILED: {e2}")
            return pd.DataFrame()


def main():
    logger.info("=" * 70)
    logger.info("WQP HAB Indicator Data Download")
    logger.info(f"Date range: {DATE_LO} to {DATE_HI}")
    logger.info(f"States: {', '.join(STATES.values())}")
    logger.info(f"Characteristics: {len(ALL_CHARS_LIST)} types")
    logger.info("=" * 70)

    all_frames = []
    total_t0 = time.time()

    for statecode, state_name in STATES.items():
        df = download_state(statecode, state_name)
        if len(df) > 0:
            all_frames.append(df)

    if not all_frames:
        logger.error("No data downloaded from any state. Exiting.")
        sys.exit(1)

    # Combine all dataframes
    logger.info("Combining all state data...")
    combined = pd.concat(all_frames, ignore_index=True)
    logger.info(f"Total records before dedup: {len(combined):,}")

    # Deduplicate
    # Use a subset of columns that uniquely identify a measurement
    dedup_cols = [c for c in [
        "MonitoringLocationIdentifier",
        "ActivityStartDate",
        "CharacteristicName",
        "ResultMeasureValue",
        "ActivityStartTime/Time",
    ] if c in combined.columns]

    if dedup_cols:
        before = len(combined)
        combined = combined.drop_duplicates(subset=dedup_cols)
        logger.info(f"Dedup: {before:,} -> {len(combined):,} ({before - len(combined):,} duplicates removed)")

    # Summary statistics
    logger.info("\n--- Summary by CharacteristicName ---")
    if "CharacteristicName" in combined.columns:
        char_counts = combined["CharacteristicName"].value_counts()
        for char_name, count in char_counts.items():
            logger.info(f"  {char_name}: {count:,}")

    logger.info("\n--- Summary by State ---")
    if "StateName" in combined.columns:
        state_counts = combined["StateName"].value_counts()
        for sn, count in state_counts.items():
            logger.info(f"  {sn}: {count:,}")
    elif "StateCode" in combined.columns:
        state_counts = combined["StateCode"].value_counts()
        for sc, count in state_counts.items():
            logger.info(f"  {sc}: {count:,}")

    # Fix mixed-type columns before saving to parquet
    # This avoids the ArrowInvalid error that crashed the BioData download
    logger.info("\nConverting object columns to string for parquet compatibility...")
    obj_cols = combined.select_dtypes(include="object").columns
    for col in obj_cols:
        combined[col] = combined[col].astype(str)
    logger.info(f"  Converted {len(obj_cols)} object columns to string")

    # Save to parquet
    out_path = os.path.join(OUT_DIR, "wqp_hab_indicators.parquet")
    combined.to_parquet(out_path, index=False, engine="pyarrow")
    file_size_mb = os.path.getsize(out_path) / (1024 * 1024)
    logger.info(f"\nSaved: {out_path}")
    logger.info(f"  Records: {len(combined):,}")
    logger.info(f"  Columns: {combined.shape[1]}")
    logger.info(f"  File size: {file_size_mb:.1f} MB")

    # Also save a small metadata CSV with column info
    meta_path = os.path.join(OUT_DIR, "wqp_hab_indicators_columns.csv")
    col_info = pd.DataFrame({
        "column": combined.columns,
        "dtype": [str(combined[c].dtype) for c in combined.columns],
        "non_null_count": [combined[c].notna().sum() for c in combined.columns],
        "sample_value": [str(combined[c].dropna().iloc[0]) if combined[c].notna().any() else "N/A" for c in combined.columns],
    })
    col_info.to_csv(meta_path, index=False)
    logger.info(f"Column metadata: {meta_path}")

    total_elapsed = time.time() - total_t0
    logger.info(f"\nTotal download time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    logger.info("Done.")


if __name__ == "__main__":
    main()
