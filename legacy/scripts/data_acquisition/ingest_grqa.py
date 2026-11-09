#!/usr/bin/env python3
"""Ingest GRQA v1.3 data into SENTINEL-DB format.

Reads the 42 GRQA parameter CSV files, applies ontology mapping, H3 spatial
indexing, quality tier assignment, and saves harmonized parquet files.

GRQA is already semi-harmonized but uses its own parameter codes and naming.
This script maps them to the SENTINEL canonical ontology.

MIT License — Bryan Cheng, 2026
"""

import json
import time
from pathlib import Path

import h3
import numpy as np
import pandas as pd

GRQA_DIR = Path("data/raw/grqa/GRQA_data_v1.3")
OUT_DIR = Path("data/sentinel_db/grqa")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# GRQA parameter code → SENTINEL canonical parameter mapping
GRQA_TO_CANONICAL = {
    "DO": "dissolved_oxygen",
    "DOSAT": "dissolved_oxygen_saturation",
    "TEMP": "water_temperature",
    "pH": "ph",
    "TSS": "total_suspended_solids",
    "TN": "total_nitrogen",
    "TP": "total_phosphorus",
    "NO3N": "nitrate",
    "NO2N": "nitrite",
    "NH4N": "ammonium",
    "TAN": "ammonia",
    "TKN": "total_kjeldahl_nitrogen",
    "DIN": "nitrate_nitrite",
    "DOC": "dissolved_organic_carbon",
    "TOC": "total_organic_carbon",
    "BOD": "biological_oxygen_demand",
    "BOD5": "biological_oxygen_demand",
    "COD": "chemical_oxygen_demand",
    "DIP": "orthophosphate",
    "TDP": "dissolved_phosphorus",
    "DKN": "total_kjeldahl_nitrogen",
    "PC": "chlorophyll_a",
}

# GRQA units → SENTINEL canonical units
UNIT_MAP = {
    "mg/l": "mg/L",
    "mg/L": "mg/L",
    "ug/l": "ug/L",
    "µg/l": "ug/L",
    "%": "% sat",
    "°c": "degC",
    "deg c": "degC",
    "-": "pH units",
}

H3_RESOLUTION = 8  # ~0.74 km² hexagons


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def process_grqa_file(csv_path, param_code):
    """Process a single GRQA parameter CSV into SENTINEL-DB format."""
    canonical_name = GRQA_TO_CANONICAL.get(param_code)
    if canonical_name is None:
        return None, 0

    df = pd.read_csv(csv_path, sep=";", low_memory=False)

    if len(df) == 0:
        return None, 0

    # Extract and clean columns
    records = pd.DataFrame()
    records["canonical_param"] = canonical_name
    records["value"] = pd.to_numeric(df["obs_value"], errors="coerce")
    records["latitude"] = pd.to_numeric(df["lat_wgs84"], errors="coerce")
    records["longitude"] = pd.to_numeric(df["lon_wgs84"], errors="coerce")
    records["timestamp"] = pd.to_datetime(df["obs_date"], errors="coerce")
    records["site_id"] = df["site_id"].astype(str)
    records["site_name"] = df.get("site_name", "").astype(str)
    records["country"] = df.get("site_country", "").astype(str)
    records["source"] = "GRQA"
    records["raw_param_name"] = df.get("param_name", param_code).astype(str)

    # Unit harmonization
    raw_unit = df.get("unit", "").astype(str).str.strip().str.lower()
    records["unit"] = raw_unit.map(lambda u: UNIT_MAP.get(u, u))

    # Quality indicators from GRQA
    records["iqr_outlier"] = df.get("obs_iqr_outlier", 0)
    records["detection_limit_flag"] = df.get("detection_limit_flag", "").astype(str)

    # Drop rows with missing essential fields
    records = records.dropna(subset=["value", "latitude", "longitude", "timestamp"])

    # Filter invalid coordinates
    records = records[
        (records["latitude"].between(-90, 90)) &
        (records["longitude"].between(-180, 180))
    ]

    # Assign H3 spatial index
    records["h3_index"] = records.apply(
        lambda r: h3.latlng_to_cell(r["latitude"], r["longitude"], H3_RESOLUTION),
        axis=1,
    )

    # Assign quality tier
    # GRQA data is from verified sources (USGS, GEMS/Water, GLORICH, WQP) → Q1
    records["quality_tier"] = "Q1"
    # Downgrade outliers to Q2
    records.loc[records["iqr_outlier"] == 1, "quality_tier"] = "Q2"
    # Downgrade detection-limit values to Q2
    records.loc[records["detection_limit_flag"] == "<", "quality_tier"] = "Q2"

    return records, len(records)


def main():
    log("=" * 60)
    log("GRQA → SENTINEL-DB Ingest")
    log("=" * 60)

    csv_files = sorted(GRQA_DIR.glob("*_GRQA.csv"))
    log(f"Found {len(csv_files)} GRQA parameter files")

    total_records = 0
    param_stats = {}
    all_dfs = []

    for csv_path in csv_files:
        param_code = csv_path.stem.replace("_GRQA", "")
        records, n = process_grqa_file(csv_path, param_code)

        if records is not None and n > 0:
            # Save per-parameter parquet
            out_path = OUT_DIR / f"{param_code}.parquet"
            records.to_parquet(out_path, index=False)

            all_dfs.append(records)
            total_records += n
            param_stats[param_code] = {
                "canonical": GRQA_TO_CANONICAL.get(param_code, "unmapped"),
                "records": n,
                "countries": records["country"].nunique(),
                "sites": records["site_id"].nunique(),
                "date_range": f"{records['timestamp'].min().date()} to {records['timestamp'].max().date()}",
            }
            log(f"  {param_code:>8s} → {GRQA_TO_CANONICAL.get(param_code, 'SKIP'):25s}: {n:>10,} records ({records['country'].nunique()} countries, {records['site_id'].nunique():,} sites)")
        else:
            log(f"  {param_code:>8s} → SKIPPED (no canonical mapping)")

    # Save combined index
    log(f"\nTotal ingested: {total_records:,} records")

    # Save summary
    summary = {
        "total_records": total_records,
        "n_parameters": len(param_stats),
        "parameters": param_stats,
    }
    with open(OUT_DIR / "ingest_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Country distribution
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        country_counts = combined["country"].value_counts().head(20)
        log("\nTop 20 countries:")
        for country, count in country_counts.items():
            log(f"  {country:20s}: {count:>10,}")

        # Unique sites and H3 cells
        log(f"\nUnique sites: {combined['site_id'].nunique():,}")
        log(f"Unique H3 cells: {combined['h3_index'].nunique():,}")
        log(f"Date range: {combined['timestamp'].min()} to {combined['timestamp'].max()}")

    log("DONE")


if __name__ == "__main__":
    main()
