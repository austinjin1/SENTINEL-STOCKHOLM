#!/usr/bin/env python3
"""Download large-scale USGS NWIS water quality sensor data for SENTINEL.

Uses a smart discovery strategy: queries what_sites() per parameter with
hasDataTypeCd='iv' to find stations that actually have instantaneous value
data, then intersects to find multi-parameter water quality monitoring sites.

Downloads IV data year-by-year and saves per-station parquet files, then
preprocesses into AquaSSM training format (.npz).

Target: 500+ stations, 5000+ sequences.

MIT License -- Bryan Cheng, 2026
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PARAMETER_CODES = {
    "00300": "DO",       # Dissolved Oxygen (mg/L)
    "00400": "pH",       # pH
    "00095": "SpCond",   # Specific Conductance (uS/cm)
    "00010": "Temp",     # Water Temperature (degC)
    "63680": "Turb",     # Turbidity (FNU)
}

# Canonical column order (6 columns for AquaSSM compatibility; ORP is placeholder)
ALL_PARAMS = ["DO", "pH", "SpCond", "Temp", "Turb", "ORP"]

# All 50 US states + DC, ordered by expected data density
ALL_STATES = [
    "OH", "PA", "NY", "VA", "MD", "NC", "GA", "FL", "TX", "CA",
    "IL", "IN", "MI", "WI", "MN", "IA", "MO", "KY", "TN", "AL",
    "SC", "WV", "NJ", "CT", "MA", "OR", "WA", "CO", "NE", "KS",
    "OK", "AR", "LA", "MS", "NM", "AZ", "UT", "NV", "ID", "MT",
    "WY", "ND", "SD", "ME", "NH", "VT", "RI", "DE", "HI", "AK",
    "DC",
]

# Date range chunks (USGS IV API handles ~1 year well)
DATE_CHUNKS = [
    ("2022-01-01", "2022-12-31"),
    ("2023-01-01", "2023-12-31"),
    ("2024-01-01", "2024-12-31"),
    ("2025-01-01", "2025-12-31"),
]

MIN_PARAMS = 3          # Minimum number of the 5 target params a station must have
SEQ_LENGTH = 512        # Sequence length for AquaSSM
OVERLAP_FRAC = 0.25     # 25% overlap between sequences
MIN_SEQ_DATA = 0.40     # Minimum fraction of valid data in a sequence
VALUE_CLAMP = 5.0       # Clamp z-scored values to [-5, 5]


def log(msg):
    """Simple timestamped logging to stdout."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Step 1: Smart station discovery
# ---------------------------------------------------------------------------

def discover_stations_smart(states, min_params=MIN_PARAMS, delay=1.0):
    """Find USGS stations with IV data for multiple water quality parameters.

    Strategy: For each state and each parameter code, query what_sites() with
    hasDataTypeCd='iv' to find sites that actually serve IV data. Then count
    how many of the 5 target parameters each site has, and keep sites with
    at least min_params.
    """
    import dataretrieval.nwis as nwis

    # site_no -> set of parameter names it has
    site_params = {}
    # site_no -> state
    site_state = {}

    for state in states:
        log(f"Discovering IV stations in {state}...")
        state_sites_found = 0

        for pcode, pname in PARAMETER_CODES.items():
            try:
                result = nwis.what_sites(
                    stateCd=state,
                    parameterCd=pcode,
                    hasDataTypeCd="iv",
                )
                df = result[0] if isinstance(result, tuple) else result

                if df is None or len(df) == 0:
                    continue

                if hasattr(df, 'reset_index'):
                    df = df.reset_index()

                if 'site_no' in df.columns:
                    sites = df['site_no'].unique()
                else:
                    continue

                for site in sites:
                    site_str = str(site).strip()
                    if site_str not in site_params:
                        site_params[site_str] = set()
                        site_state[site_str] = state
                    site_params[site_str].add(pname)

            except Exception as e:
                log(f"  {state}/{pcode}: error - {e}")

            time.sleep(0.3)  # Brief pause between parameter queries

        # Count how many qualified sites this state contributed
        state_qualified = sum(
            1 for s, params in site_params.items()
            if site_state[s] == state and len(params) >= min_params
        )
        log(f"  {state}: {state_qualified} stations with >= {min_params} params")
        time.sleep(delay)

    # Filter to stations with >= min_params
    qualified = []
    param_distribution = Counter()
    for site_no, params in site_params.items():
        n = len(params)
        param_distribution[n] += 1
        if n >= min_params:
            qualified.append({
                "site_no": site_no,
                "state": site_state[site_no],
                "iv_params": sorted(list(params)),
                "n_params": n,
            })

    # Sort by number of parameters (descending), then by site_no
    qualified.sort(key=lambda x: (-x["n_params"], x["site_no"]))

    log(f"Parameter distribution across all sites:")
    for n in sorted(param_distribution.keys()):
        log(f"  {n} params: {param_distribution[n]} sites")
    log(f"Qualified stations (>= {min_params} params): {len(qualified)}")

    return qualified


# ---------------------------------------------------------------------------
# Step 2: Download IV data per station
# ---------------------------------------------------------------------------

def download_station_data(site_no, output_dir, delay=0.5):
    """Download instantaneous values for a single station across all date chunks.

    Downloads year-by-year and concatenates. Saves as parquet.
    Returns True if successful (station has >= MIN_PARAMS parameters with data).
    """
    import dataretrieval.nwis as nwis

    output_file = output_dir / f"{site_no}.parquet"
    if output_file.exists():
        try:
            existing = pd.read_parquet(output_file)
            if len(existing) >= 100:
                return True
        except Exception:
            pass

    chunks = []
    for start_date, end_date in DATE_CHUNKS:
        try:
            df, _ = nwis.get_iv(
                sites=site_no,
                parameterCd=list(PARAMETER_CODES.keys()),
                start=start_date,
                end=end_date,
            )
            if df is not None and len(df) > 0:
                chunks.append(df)
        except Exception:
            pass
        time.sleep(delay)

    if not chunks:
        return False

    combined = pd.concat(chunks, axis=0)
    combined = combined.sort_index()
    combined = combined[~combined.index.duplicated(keep='first')]

    # Rename columns: map parameter codes to standard names
    rename_map = {}
    for code, name in PARAMETER_CODES.items():
        for col in combined.columns:
            if code in col and '_cd' not in col and col != 'site_no':
                # Prefer exact match, but accept partial (e.g., 00010_instream)
                rename_map[col] = name
                break

    combined = combined.rename(columns=rename_map)

    # Deduplicate columns: if there are multiple columns mapping to same name,
    # keep the first
    seen = set()
    keep = []
    for col in combined.columns:
        if col in PARAMETER_CODES.values() and col not in seen:
            keep.append(col)
            seen.add(col)

    if len(keep) < MIN_PARAMS:
        return False

    combined = combined[keep]
    combined = combined.dropna(how='all')

    if len(combined) < 100:
        return False

    combined.to_parquet(output_file)
    return True


# ---------------------------------------------------------------------------
# Step 3: Preprocess into AquaSSM format
# ---------------------------------------------------------------------------

def preprocess_station(parquet_path, output_dir, global_stats=None):
    """Preprocess a single station parquet into .npz sequences.

    Returns (n_sequences, n_records) or (0, 0) on failure.
    """
    try:
        df = pd.read_parquet(parquet_path)
    except Exception:
        return 0, 0

    if len(df) < SEQ_LENGTH:
        return 0, 0

    site_no = parquet_path.stem

    if not isinstance(df.index, pd.DatetimeIndex):
        return 0, 0

    df = df.sort_index()

    # Compute time deltas in seconds
    timestamps = df.index.astype(np.int64) // 10**9
    delta_ts = np.diff(timestamps, prepend=timestamps[0]).astype(np.float32)
    delta_ts[0] = 0.0

    # Ensure all 6 parameter columns exist (ORP will be all NaN)
    for p in ALL_PARAMS:
        if p not in df.columns:
            df[p] = np.nan

    values = df[ALL_PARAMS].values.astype(np.float64)
    mask = ~np.isnan(values)

    # Z-score normalization per parameter
    for i, param in enumerate(ALL_PARAMS):
        valid = mask[:, i]
        if valid.sum() < 10:
            values[:, i] = 0.0
            continue

        col = values[:, i].copy()
        if global_stats and param in global_stats and global_stats[param]["count"] > 0:
            mu = global_stats[param]["mean"]
            sigma = global_stats[param]["std"]
        else:
            mu = np.nanmean(col[valid])
            sigma = np.nanstd(col[valid])

        if sigma < 1e-8:
            sigma = 1.0

        values[:, i] = np.where(valid, (col - mu) / sigma, 0.0)

    # Clamp to [-5, 5]
    values = np.clip(values, -VALUE_CLAMP, VALUE_CLAMP)
    values = np.nan_to_num(values, nan=0.0).astype(np.float32)

    # Labels: all 0 (normal) for real sensor data
    labels = np.zeros(len(values), dtype=np.float32)

    # Extract overlapping sequences
    stride = max(1, int(SEQ_LENGTH * (1.0 - OVERLAP_FRAC)))
    n_seqs = 0

    for j in range((len(values) - SEQ_LENGTH) // stride + 1):
        start = j * stride
        end = start + SEQ_LENGTH
        if end > len(values):
            break

        seq_values = values[start:end].copy()
        seq_delta = delta_ts[start:end].copy()
        seq_mask = mask[start:end].copy()
        seq_labels = labels[start:end].copy()

        # CRITICAL: delta_ts[0] MUST be 0 for every sequence
        seq_delta[0] = 0.0

        # Skip sequences with too little valid data
        if seq_mask.mean() < MIN_SEQ_DATA:
            continue

        out_file = output_dir / f"{site_no}_seq{j:05d}.npz"
        np.savez_compressed(
            out_file,
            values=seq_values,        # (512, 6) float32
            delta_ts=seq_delta,        # (512,)   float32
            labels=seq_labels,         # (512,)   float32
            mask=seq_mask,             # (512, 6) bool
        )
        n_seqs += 1

    return n_seqs, len(df)


def compute_global_stats(raw_dir):
    """Compute per-parameter mean and std across all downloaded stations."""
    log("Computing global parameter statistics...")
    param_accum = {p: {"sum": 0.0, "sum_sq": 0.0, "count": 0, "min": float('inf'), "max": float('-inf')} for p in ALL_PARAMS}

    parquet_files = sorted(raw_dir.glob("*.parquet"))
    for pf in parquet_files:
        try:
            df = pd.read_parquet(pf)
            for p in ALL_PARAMS:
                if p in df.columns:
                    vals = df[p].dropna().values
                    if len(vals) > 0:
                        param_accum[p]["sum"] += vals.sum()
                        param_accum[p]["sum_sq"] += (vals ** 2).sum()
                        param_accum[p]["count"] += len(vals)
                        param_accum[p]["min"] = min(param_accum[p]["min"], float(vals.min()))
                        param_accum[p]["max"] = max(param_accum[p]["max"], float(vals.max()))
        except Exception:
            continue

    stats = {}
    for p in ALL_PARAMS:
        acc = param_accum[p]
        if acc["count"] > 1:
            mean = acc["sum"] / acc["count"]
            var = acc["sum_sq"] / acc["count"] - mean ** 2
            std = max(np.sqrt(max(var, 0.0)), 1e-8)
            stats[p] = {
                "mean": float(mean),
                "std": float(std),
                "count": int(acc["count"]),
                "min": float(acc["min"]),
                "max": float(acc["max"]),
            }
            log(f"  {p}: mean={mean:.4f}, std={std:.4f}, n={acc['count']}")
        else:
            stats[p] = {"mean": 0.0, "std": 1.0, "count": 0, "min": 0.0, "max": 0.0}
            log(f"  {p}: NO DATA")

    return stats


def preprocess_all(raw_dir, output_dir):
    """Preprocess all station parquets into AquaSSM .npz sequences."""
    output_dir.mkdir(parents=True, exist_ok=True)

    global_stats = compute_global_stats(raw_dir)

    stats_file = output_dir / "normalization_stats.json"
    with open(stats_file, "w") as f:
        json.dump(global_stats, f, indent=2)
    log(f"Saved normalization stats to {stats_file}")

    parquet_files = sorted(raw_dir.glob("*.parquet"))
    log(f"Preprocessing {len(parquet_files)} station files into sequences...")

    total_seqs = 0
    total_records = 0
    station_results = []

    for i, pf in enumerate(parquet_files):
        n_seqs, n_records = preprocess_station(pf, output_dir, global_stats)
        total_seqs += n_seqs
        total_records += n_records
        if n_seqs > 0:
            station_results.append({
                "site_no": pf.stem,
                "n_records": n_records,
                "n_sequences": n_seqs,
            })
        if (i + 1) % 50 == 0 or (i + 1) == len(parquet_files):
            log(f"  Preprocessed {i+1}/{len(parquet_files)} stations, "
                f"{total_seqs} sequences from {len(station_results)} stations so far")

    final_stats = {
        "total_sequences": total_seqs,
        "total_records": total_records,
        "total_stations_with_sequences": len(station_results),
        "total_parquet_files": len(parquet_files),
        "seq_length": SEQ_LENGTH,
        "overlap_fraction": OVERLAP_FRAC,
        "value_clamp": VALUE_CLAMP,
        "min_seq_data_fraction": MIN_SEQ_DATA,
        "parameters": ALL_PARAMS,
        "normalization_stats": global_stats,
        "stations": station_results,
    }
    with open(output_dir / "preprocessing_stats.json", "w") as f:
        json.dump(final_stats, f, indent=2)

    log(f"Preprocessing complete: {total_seqs} sequences from {len(station_results)} stations")
    return total_seqs, len(station_results)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Download large-scale USGS NWIS sensor data")
    parser.add_argument("--states", nargs="+", default=None,
                        help="State codes to query (default: all 50 + DC)")
    parser.add_argument("--data-dir", default="/home/bcheng/SENTINEL/data",
                        help="Base data directory")
    parser.add_argument("--max-stations", type=int, default=2000,
                        help="Maximum stations to download")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download, only preprocess existing parquets")
    parser.add_argument("--skip-preprocess", action="store_true",
                        help="Skip preprocessing, only download")
    parser.add_argument("--station-delay", type=float, default=0.5,
                        help="Delay between station download API calls (seconds)")
    parser.add_argument("--state-delay", type=float, default=1.0,
                        help="Delay between state discovery calls (seconds)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    raw_dir = data_dir / "raw" / "sensor" / "full"
    processed_dir = data_dir / "processed" / "sensor" / "full"
    raw_dir.mkdir(parents=True, exist_ok=True)

    states = args.states if args.states else ALL_STATES

    if not args.skip_download:
        # ---------------------------------------------------------------
        # Step 1: Smart station discovery
        # ---------------------------------------------------------------
        log("=" * 70)
        log("STEP 1: Discovering USGS NWIS water quality stations (smart mode)")
        log("=" * 70)

        catalog_file = raw_dir / "station_catalog_smart.json"

        if catalog_file.exists():
            with open(catalog_file) as f:
                stations = json.load(f)
            log(f"Loaded cached smart catalog: {len(stations)} qualified stations")
        else:
            stations = discover_stations_smart(states, min_params=MIN_PARAMS, delay=args.state_delay)
            with open(catalog_file, "w") as f:
                json.dump(stations, f, indent=2)
            log(f"Saved {len(stations)} qualified stations to {catalog_file}")

        # ---------------------------------------------------------------
        # Step 2: Download IV data for each qualified station
        # ---------------------------------------------------------------
        n_to_download = min(len(stations), args.max_stations)
        log("=" * 70)
        log(f"STEP 2: Downloading IV data for {n_to_download} qualified stations")
        log("=" * 70)

        existing_parquets = set(p.stem for p in raw_dir.glob("*.parquet"))
        log(f"Already have {len(existing_parquets)} parquet files")

        downloaded = 0
        failed = 0
        skipped = 0

        for i, station in enumerate(stations[:n_to_download]):
            site_no = station["site_no"]

            if site_no in existing_parquets:
                skipped += 1
                downloaded += 1  # Count existing as downloaded
                continue

            try:
                success = download_station_data(site_no, raw_dir, delay=args.station_delay)
                if success:
                    downloaded += 1
                    existing_parquets.add(site_no)
                else:
                    failed += 1
            except Exception as e:
                log(f"  {site_no}: EXCEPTION - {e}")
                failed += 1

            if (i + 1) % 25 == 0 or (i + 1) == n_to_download:
                log(f"  Progress: {i+1}/{n_to_download} "
                    f"(downloaded={downloaded}, failed={failed}, skipped_existing={skipped})")

        log(f"Download complete: {downloaded} stations with data, {failed} failed")

    if not args.skip_preprocess:
        # ---------------------------------------------------------------
        # Step 3: Preprocess into AquaSSM format
        # ---------------------------------------------------------------
        log("=" * 70)
        log("STEP 3: Preprocessing into AquaSSM format")
        log("=" * 70)

        n_seqs, n_stations = preprocess_all(raw_dir, processed_dir)

        log("=" * 70)
        log("PIPELINE COMPLETE")
        log(f"  Stations with sequences: {n_stations}")
        log(f"  Total sequences: {n_seqs}")
        log(f"  Raw data: {raw_dir}")
        log(f"  Processed data: {processed_dir}")
        log("=" * 70)


if __name__ == "__main__":
    main()
