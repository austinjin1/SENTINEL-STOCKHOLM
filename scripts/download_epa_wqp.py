#!/usr/bin/env python3
"""Download real water quality data from EPA Water Quality Portal.

Uses the dataretrieval package to pull discrete sample results from
https://www.waterqualitydata.us/

Strategy: Download by HUC2 region (21 major basins) for key parameters.
Focus on last 10 years to keep manageable while still getting millions of records.

MIT License — Bryan Cheng, 2026
"""

import json
import time
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data/raw/epa_wqp")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Key water quality parameters to download (EPA characteristic names)
TARGET_CHARACTERISTICS = [
    "Dissolved oxygen (DO)",
    "pH",
    "Temperature, water",
    "Specific conductance",
    "Turbidity",
    "Nitrogen, mixed forms",
    "Nitrate",
    "Ammonia",
    "Phosphorus",
    "Chlorophyll a",
    "Total dissolved solids",
    "Total suspended solids",
]

# HUC2 regions (major US drainage basins)
HUC2_REGIONS = {
    "01": "New England",
    "02": "Mid-Atlantic",
    "03": "South Atlantic-Gulf",
    "04": "Great Lakes",
    "05": "Ohio",
    "06": "Tennessee",
    "07": "Upper Mississippi",
    "08": "Lower Mississippi",
    "09": "Souris-Red-Rainy",
    "10": "Missouri",
    "11": "Arkansas-White-Red",
    "12": "Texas-Gulf",
    "13": "Rio Grande",
    "14": "Upper Colorado",
    "15": "Lower Colorado",
    "16": "Great Basin",
    "17": "Pacific Northwest",
    "18": "California",
}


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def download_wqp_by_huc2(huc2, characteristics, start_date="01-01-2020", end_date="12-31-2025"):
    """Download WQP data for a HUC2 region."""
    import dataretrieval.wqp as wqp

    dest = DATA_DIR / f"wqp_huc{huc2}.parquet"
    if dest.exists() and dest.stat().st_size > 1000:
        df = pd.read_parquet(dest)
        log(f"  Already exists: HUC {huc2} ({len(df):,} records)")
        return len(df)

    log(f"  Downloading HUC {huc2} ({HUC2_REGIONS.get(huc2, 'Unknown')})...")

    all_dfs = []
    for char in characteristics:
        try:
            df, md = wqp.get_results(
                huc=huc2,
                characteristicName=char,
                startDateLo=start_date,
                startDateHi=end_date,
            )
            if len(df) > 0:
                all_dfs.append(df)
                log(f"    {char}: {len(df):,} records")
            time.sleep(0.5)  # Rate limiting
        except Exception as e:
            log(f"    {char}: FAILED ({e})")
            time.sleep(2)

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        # Keep essential columns
        keep_cols = [c for c in [
            "MonitoringLocationIdentifier",
            "MonitoringLocationName",
            "MonitoringLocationTypeName",
            "HUCEightDigitCode",
            "ActivityStartDate",
            "ActivityStartTime/Time",
            "CharacteristicName",
            "ResultMeasureValue",
            "ResultMeasure/MeasureUnitCode",
            "ResultStatusIdentifier",
            "ResultValueTypeName",
            "ProviderName",
            "OrganizationIdentifier",
            "ActivityMediaName",
            "LatitudeMeasure",
            "LongitudeMeasure",
        ] if c in combined.columns]

        combined = combined[keep_cols]
        # Convert mixed-type columns to string to avoid parquet errors
        for col in combined.columns:
            if combined[col].dtype == object:
                combined[col] = combined[col].astype(str)
        combined.to_parquet(dest, index=False)
        log(f"    Saved: {len(combined):,} total records -> {dest.name}")
        return len(combined)
    else:
        log(f"    No data for HUC {huc2}")
        return 0


def main():
    log("=" * 60)
    log("EPA Water Quality Portal Download")
    log(f"Target: {len(TARGET_CHARACTERISTICS)} parameters across {len(HUC2_REGIONS)} HUC2 basins")
    log(f"Date range: 2020-01-01 to 2025-12-31")
    log("=" * 60)

    total_records = 0
    results = {}

    for huc2, name in HUC2_REGIONS.items():
        log(f"\nHUC {huc2}: {name}")
        n = download_wqp_by_huc2(huc2, TARGET_CHARACTERISTICS)
        total_records += n
        results[huc2] = {"name": name, "records": n}

    log("\n" + "=" * 60)
    log("EPA WQP Download Summary")
    log("=" * 60)
    for huc2, info in sorted(results.items()):
        log(f"  HUC {huc2} ({info['name']:25s}): {info['records']:>10,} records")
    log(f"  {'TOTAL':34s}: {total_records:>10,} records")

    # Save summary
    summary = {"total_records": total_records, "by_huc2": results}
    with open(DATA_DIR / "download_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Quick inspection of combined data
    log("\nData inspection:")
    parquets = list(DATA_DIR.glob("*.parquet"))
    if parquets:
        sample = pd.read_parquet(parquets[0])
        log(f"  Sample file: {parquets[0].name}")
        log(f"  Columns: {list(sample.columns)}")
        if "CharacteristicName" in sample.columns:
            log(f"  Parameters: {sample['CharacteristicName'].value_counts().head(10).to_dict()}")

    log("DONE")


if __name__ == "__main__":
    main()
