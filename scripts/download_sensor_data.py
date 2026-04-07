#!/usr/bin/env python3
"""Download and preprocess USGS NWIS sensor data for AquaSSM training.

Starts with a focused subset of states for rapid iteration, then can be
expanded to all 50 states.

MIT License — Bryan Cheng, 2026
"""

import argparse
import json
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from sentinel.utils.logging import get_logger

logger = get_logger(__name__)

# USGS parameter codes for the 6 target parameters
PARAMETER_CODES = {
    "00300": "DO",       # Dissolved Oxygen (mg/L)
    "00400": "pH",       # pH
    "00095": "SpCond",   # Specific Conductance (uS/cm)
    "00010": "Temp",     # Water Temperature (degC)
    "63680": "Turb",     # Turbidity (FNU)
    "00090": "ORP",      # Oxidation-Reduction Potential (mV)
}

# Start with high-data-density states for rapid iteration
PRIORITY_STATES = ["OH", "PA", "NY", "VA", "MD", "NC", "GA", "FL", "TX", "CA"]


def discover_stations(states, min_params=3, delay=1.0):
    """Find USGS continuous monitoring stations with target parameters."""
    import dataretrieval.nwis as nwis

    all_stations = []
    for state in states:
        logger.info(f"Discovering stations in {state}...")
        try:
            site_info = nwis.get_info(
                stateCd=state,
                siteType="ST",
                parameterCd=list(PARAMETER_CODES.keys()),
                siteStatus="active",
            )
            if site_info is not None and len(site_info) > 0:
                df = site_info[0] if isinstance(site_info, tuple) else site_info
                if hasattr(df, 'reset_index'):
                    df = df.reset_index()

                # Get unique site numbers
                if 'site_no' in df.columns:
                    sites = df['site_no'].unique().tolist()
                elif df.index.name == 'site_no':
                    sites = df.index.unique().tolist()
                else:
                    sites = []

                for site in sites:
                    all_stations.append({
                        "site_no": str(site),
                        "state": state,
                    })
                logger.info(f"  {state}: {len(sites)} stations found")
        except Exception as e:
            logger.warning(f"  {state}: Failed - {e}")
        time.sleep(delay)

    logger.info(f"Total stations discovered: {len(all_stations)}")
    return all_stations


def download_station_data(site_no, start_date, end_date, output_dir):
    """Download instantaneous values for a single station."""
    import dataretrieval.nwis as nwis

    output_file = output_dir / f"{site_no}.parquet"
    if output_file.exists():
        return True

    try:
        df, _ = nwis.get_iv(
            sites=site_no,
            parameterCd=list(PARAMETER_CODES.keys()),
            start=start_date,
            end=end_date,
        )

        if df is None or len(df) == 0:
            return False

        # Rename columns to standard names
        rename_map = {}
        for code, name in PARAMETER_CODES.items():
            for col in df.columns:
                if code in col and '_cd' not in col:
                    rename_map[col] = name
                    break

        df = df.rename(columns=rename_map)

        # Keep only the target columns that exist
        keep_cols = [c for c in PARAMETER_CODES.values() if c in df.columns]
        if len(keep_cols) < 3:
            return False

        df = df[keep_cols]
        df.to_parquet(output_file)
        return True

    except Exception as e:
        logger.debug(f"  {site_no}: {e}")
        return False


def preprocess_for_aquassm(raw_dir, output_dir, seq_length=2000, overlap=0.1):
    """Convert raw station parquets to AquaSSM training format.

    Output: .npz files with:
        - values: (T, 6) float32 — normalized parameter values
        - delta_ts: (T,) float32 — time gaps in seconds
        - mask: (T, 6) bool — validity mask
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_files = sorted(raw_dir.glob("*.parquet"))
    logger.info(f"Preprocessing {len(parquet_files)} station files...")

    total_sequences = 0
    station_stats = []

    for pf in parquet_files:
        try:
            df = pd.read_parquet(pf)
            if len(df) < 100:
                continue

            site_no = pf.stem

            # Ensure datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                continue

            df = df.sort_index()

            # Compute time deltas in seconds
            timestamps = df.index.astype(np.int64) // 10**9  # Unix seconds
            delta_ts = np.diff(timestamps, prepend=timestamps[0]).astype(np.float32)
            delta_ts[0] = 0.0

            # Fill all 6 parameters with NaN where missing
            all_params = ["DO", "pH", "SpCond", "Temp", "Turb", "ORP"]
            for p in all_params:
                if p not in df.columns:
                    df[p] = np.nan

            values = df[all_params].values.astype(np.float32)
            mask = ~np.isnan(values)

            # Rolling z-score normalization (90-day window)
            window_size = min(8640, len(values))  # ~90 days at 15-min
            for i in range(6):
                col = values[:, i].copy()
                valid = mask[:, i]
                if valid.sum() < 10:
                    continue
                # Compute running mean/std
                valid_vals = col[valid]
                running_mean = np.convolve(
                    np.where(valid, col, 0),
                    np.ones(window_size) / window_size,
                    mode='same'
                )
                running_sq_mean = np.convolve(
                    np.where(valid, col**2, 0),
                    np.ones(window_size) / window_size,
                    mode='same'
                )
                running_std = np.sqrt(np.maximum(running_sq_mean - running_mean**2, 1e-6))
                values[:, i] = np.where(valid, (col - running_mean) / running_std, 0.0)

            # Replace NaN with 0 (masked positions)
            values = np.nan_to_num(values, nan=0.0)

            # Extract overlapping sequences
            stride = int(seq_length * (1 - overlap))
            n_seqs = max(1, (len(values) - seq_length) // stride + 1)

            for j in range(n_seqs):
                start = j * stride
                end = start + seq_length
                if end > len(values):
                    break

                seq_values = values[start:end]
                seq_delta = delta_ts[start:end]
                seq_mask = mask[start:end]

                # Skip sequences with <50% valid data
                if seq_mask.mean() < 0.5:
                    continue

                out_file = output_dir / f"{site_no}_seq{j:04d}.npz"
                np.savez_compressed(
                    out_file,
                    values=seq_values,
                    delta_ts=seq_delta,
                    mask=seq_mask,
                    site_no=site_no,
                )
                total_sequences += 1

            station_stats.append({
                "site_no": site_no,
                "n_records": len(df),
                "n_sequences": n_seqs,
                "param_coverage": {p: float(mask[:, i].mean()) for i, p in enumerate(all_params)},
            })

        except Exception as e:
            logger.warning(f"  {pf.stem}: preprocessing failed - {e}")

    # Save stats
    stats_file = output_dir / "preprocessing_stats.json"
    with open(stats_file, "w") as f:
        json.dump({
            "total_sequences": total_sequences,
            "total_stations": len(station_stats),
            "stations": station_stats,
        }, f, indent=2)

    logger.info(f"Preprocessing complete: {total_sequences} sequences from {len(station_stats)} stations")
    return total_sequences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", nargs="+", default=PRIORITY_STATES)
    parser.add_argument("--start-date", default="2020-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--data-dir", default="data/")
    parser.add_argument("--max-stations", type=int, default=200)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    raw_dir = data_dir / "raw" / "sensor"
    processed_dir = data_dir / "processed" / "sensor" / "pretrain"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        # Step 1: Discover stations
        logger.info("=" * 60)
        logger.info("Step 1: Discovering USGS NWIS stations")
        logger.info("=" * 60)
        stations = discover_stations(args.states)

        # Save station catalog
        catalog_file = raw_dir / "station_catalog.json"
        with open(catalog_file, "w") as f:
            json.dump(stations, f, indent=2)
        logger.info(f"Saved {len(stations)} stations to {catalog_file}")

        # Step 2: Download data
        logger.info("=" * 60)
        logger.info(f"Step 2: Downloading data for up to {args.max_stations} stations")
        logger.info("=" * 60)

        downloaded = 0
        for i, station in enumerate(stations[:args.max_stations]):
            site_no = station["site_no"]
            success = download_station_data(
                site_no, args.start_date, args.end_date, raw_dir
            )
            if success:
                downloaded += 1
            if (i + 1) % 10 == 0:
                logger.info(f"  Progress: {i+1}/{min(len(stations), args.max_stations)}, downloaded: {downloaded}")
            time.sleep(0.5)  # Rate limiting

        logger.info(f"Downloaded data for {downloaded} stations")

    # Step 3: Preprocess
    logger.info("=" * 60)
    logger.info("Step 3: Preprocessing for AquaSSM")
    logger.info("=" * 60)
    n_seqs = preprocess_for_aquassm(raw_dir, processed_dir)
    logger.info(f"Final: {n_seqs} training sequences ready")


if __name__ == "__main__":
    main()
