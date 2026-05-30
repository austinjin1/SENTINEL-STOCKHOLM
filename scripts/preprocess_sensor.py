#!/usr/bin/env python3
"""Standalone preprocessing of raw USGS NWIS sensor parquet files.

Reads each raw parquet file from data/raw/sensor/full/ (produced by
download_sensor.py, possibly still running) and produces AquaSSM-compatible
.npz sequences in data/processed/sensor/full/.

Can be run independently of the download process — it only touches
already-written parquet files and writes to a separate output directory.

Processing pipeline per station:
  1. Load parquet (DatetimeIndex, columns ⊆ {DO, pH, SpCond, Temp, Turb})
  2. Sort by time, add missing parameter columns as NaN
  3. Compute global normalization stats across all available stations (first pass)
  4. Z-score normalize each parameter using global stats
  5. Generate anomaly labels via z-score of first derivative (per-param)
  6. Create overlapping sliding windows of length 512
  7. Save each window as .npz with keys: values, delta_ts, labels, mask

Output format matches USGSSensorDataset in train_aquassm_v2.py:
  - values:   (512, 6) float32 — z-scored sensor parameters
  - delta_ts: (512,)   float32 — time gaps in seconds
  - labels:   (512,)   float32 — per-timestep anomaly labels (0=normal, 1=anomaly)
  - mask:     (512, 6) bool    — per-parameter validity

MIT License -- Bryan Cheng, SENTINEL project, 2026
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration — matches download_sensor.py exactly
# ---------------------------------------------------------------------------

# Canonical column order (6 columns for AquaSSM compatibility; ORP is placeholder)
ALL_PARAMS = ["DO", "pH", "SpCond", "Temp", "Turb", "ORP"]

SEQ_LENGTH = 512         # Sequence length per window
OVERLAP_FRAC = 0.25      # 25 % overlap between windows
MIN_SEQ_DATA = 0.40      # Minimum fraction of valid data in a window
VALUE_CLAMP = 5.0        # Clamp z-scored values to [-5, 5]

# Anomaly labeling parameters
ANOMALY_ZSCORE_THRESH = 4.0   # z-score threshold on derivative to flag anomaly
ANOMALY_WINDOW = 3            # Half-window for temporal smoothing of labels


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Pass 1: Compute global normalization statistics
# ---------------------------------------------------------------------------

def compute_global_stats(raw_dir: Path) -> dict:
    """Compute per-parameter mean/std across all raw station parquets.

    Uses Welford-style two-pass (sum, sum_sq) accumulation which is
    numerically stable enough for our data ranges.
    """
    log("Pass 1: Computing global parameter statistics ...")

    accum = {
        p: {"sum": 0.0, "sum_sq": 0.0, "count": 0,
            "min": float("inf"), "max": float("-inf")}
        for p in ALL_PARAMS
    }

    parquet_files = sorted(raw_dir.glob("*.parquet"))
    for pf in parquet_files:
        try:
            df = pd.read_parquet(pf)
        except Exception:
            continue
        for p in ALL_PARAMS:
            if p not in df.columns:
                continue
            vals = df[p].dropna().values.astype(np.float64)
            # Filter out USGS sentinel values (-999999) and physically impossible values
            vals = vals[(vals > -1e5) & (vals < 1e5)]
            if len(vals) == 0:
                continue
            accum[p]["sum"] += vals.sum()
            accum[p]["sum_sq"] += (vals ** 2).sum()
            accum[p]["count"] += len(vals)
            accum[p]["min"] = min(accum[p]["min"], float(vals.min()))
            accum[p]["max"] = max(accum[p]["max"], float(vals.max()))

    stats: dict = {}
    for p in ALL_PARAMS:
        a = accum[p]
        if a["count"] > 1:
            mean = a["sum"] / a["count"]
            var = a["sum_sq"] / a["count"] - mean ** 2
            std = max(np.sqrt(max(var, 0.0)), 1e-8)
            stats[p] = {
                "mean": float(mean),
                "std": float(std),
                "count": int(a["count"]),
                "min": float(a["min"]),
                "max": float(a["max"]),
            }
            log(f"  {p:>6s}: mean={mean:10.4f}  std={std:8.4f}  n={a['count']}")
        else:
            stats[p] = {"mean": 0.0, "std": 1.0, "count": 0, "min": 0.0, "max": 0.0}
            log(f"  {p:>6s}: NO DATA (will be zero-filled)")

    return stats


# ---------------------------------------------------------------------------
# Pass 1b: Compute per-parameter derivative statistics (for anomaly labels)
# ---------------------------------------------------------------------------

def compute_derivative_stats(raw_dir: Path) -> dict:
    """Compute global mean/std of the first discrete derivative for each param.

    These statistics are used to z-score the derivative and apply a threshold
    for statistical anomaly labeling.
    """
    log("Pass 1b: Computing derivative statistics for anomaly labeling ...")

    accum = {
        p: {"sum": 0.0, "sum_sq": 0.0, "count": 0}
        for p in ALL_PARAMS
    }

    parquet_files = sorted(raw_dir.glob("*.parquet"))
    for pf in parquet_files:
        try:
            df = pd.read_parquet(pf)
        except Exception:
            continue
        df = df.sort_index()
        for p in ALL_PARAMS:
            if p not in df.columns:
                continue
            col = df[p].values.astype(np.float64)
            # Filter out USGS sentinel values (-999999)
            col[(col < -1e5) | (col > 1e5)] = np.nan
            valid = ~np.isnan(col)
            if valid.sum() < 10:
                continue
            # Forward difference; NaN where either neighbor is NaN
            deriv = np.diff(col)
            valid_d = ~np.isnan(deriv)
            d = deriv[valid_d]
            if len(d) == 0:
                continue
            accum[p]["sum"] += d.sum()
            accum[p]["sum_sq"] += (d ** 2).sum()
            accum[p]["count"] += len(d)

    d_stats: dict = {}
    for p in ALL_PARAMS:
        a = accum[p]
        if a["count"] > 1:
            mean = a["sum"] / a["count"]
            var = a["sum_sq"] / a["count"] - mean ** 2
            std = max(np.sqrt(max(var, 0.0)), 1e-8)
            d_stats[p] = {"mean": float(mean), "std": float(std), "count": int(a["count"])}
            log(f"  d({p:>6s}): mean={mean:12.6f}  std={std:10.6f}  n={a['count']}")
        else:
            d_stats[p] = {"mean": 0.0, "std": 1.0, "count": 0}

    return d_stats


# ---------------------------------------------------------------------------
# Anomaly labeling
# ---------------------------------------------------------------------------

def label_anomalies(
    df: pd.DataFrame,
    d_stats: dict,
    threshold: float = ANOMALY_ZSCORE_THRESH,
    half_window: int = ANOMALY_WINDOW,
) -> np.ndarray:
    """Create per-timestep anomaly labels using z-score of the derivative.

    A timestep is labeled anomalous (1.0) if ANY parameter's derivative
    z-score exceeds the threshold.  We also expand each detected point by
    ±half_window steps to capture the onset/recovery of the event.

    Returns: labels array of shape (len(df),), dtype float32
    """
    n = len(df)
    labels = np.zeros(n, dtype=np.float32)

    for p in ALL_PARAMS:
        if p not in df.columns:
            continue
        ds = d_stats.get(p)
        if ds is None or ds["count"] == 0:
            continue

        col = df[p].values.astype(np.float64)
        deriv = np.diff(col, prepend=np.nan)  # length n, first element is NaN
        valid = ~np.isnan(deriv)
        if valid.sum() == 0:
            continue

        z = np.zeros(n, dtype=np.float64)
        z[valid] = np.abs((deriv[valid] - ds["mean"]) / ds["std"])

        # Mark points exceeding threshold
        spike = z > threshold
        if not spike.any():
            continue

        # Expand by ±half_window to capture event context
        spike_idx = np.where(spike)[0]
        for idx in spike_idx:
            lo = max(0, idx - half_window)
            hi = min(n, idx + half_window + 1)
            labels[lo:hi] = 1.0

    return labels


# ---------------------------------------------------------------------------
# Pass 2: Process each station
# ---------------------------------------------------------------------------

def preprocess_station(
    parquet_path: Path,
    output_dir: Path,
    global_stats: dict,
    d_stats: dict,
) -> tuple[int, int]:
    """Preprocess a single station parquet into .npz sequence windows.

    Returns (n_sequences_created, n_total_records).
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

    # ------------------------------------------------------------------
    # Time deltas (seconds since previous reading)
    # ------------------------------------------------------------------
    timestamps = df.index.astype(np.int64) // 10**9
    delta_ts = np.diff(timestamps, prepend=timestamps[0]).astype(np.float32)
    delta_ts[0] = 0.0

    # ------------------------------------------------------------------
    # Ensure all 6 canonical parameter columns exist
    # ------------------------------------------------------------------
    for p in ALL_PARAMS:
        if p not in df.columns:
            df[p] = np.nan

    # ------------------------------------------------------------------
    # Anomaly labels (before normalization — works on raw scale)
    # ------------------------------------------------------------------
    labels = label_anomalies(df, d_stats)

    # ------------------------------------------------------------------
    # Z-score normalization
    # ------------------------------------------------------------------
    values = df[ALL_PARAMS].values.astype(np.float64)
    # Filter out USGS sentinel values (-999999) and physically impossible values
    values[(values < -1e5) | (values > 1e5)] = np.nan
    mask = ~np.isnan(values)

    for i, param in enumerate(ALL_PARAMS):
        valid = mask[:, i]
        if valid.sum() < 10:
            values[:, i] = 0.0
            continue

        gs = global_stats.get(param, {})
        if gs.get("count", 0) > 0:
            mu = gs["mean"]
            sigma = gs["std"]
        else:
            mu = float(np.nanmean(values[valid, i]))
            sigma = float(np.nanstd(values[valid, i]))

        if sigma < 1e-8:
            sigma = 1.0

        values[:, i] = np.where(valid, (values[:, i] - mu) / sigma, 0.0)

    # Clamp and cast
    values = np.clip(values, -VALUE_CLAMP, VALUE_CLAMP)
    values = np.nan_to_num(values, nan=0.0).astype(np.float32)

    # ------------------------------------------------------------------
    # Sliding window extraction
    # ------------------------------------------------------------------
    stride = max(1, int(SEQ_LENGTH * (1.0 - OVERLAP_FRAC)))
    n_seqs = 0

    for j in range((len(values) - SEQ_LENGTH) // stride + 1):
        start = j * stride
        end = start + SEQ_LENGTH
        if end > len(values):
            break

        seq_values = values[start:end].copy()       # (512, 6)
        seq_delta = delta_ts[start:end].copy()       # (512,)
        seq_mask = mask[start:end].copy()             # (512, 6)
        seq_labels = labels[start:end].copy()         # (512,)

        # CRITICAL: first delta_t of every sequence must be 0
        seq_delta[0] = 0.0

        # Skip windows with too little valid data
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess raw USGS NWIS sensor parquets into AquaSSM .npz sequences"
    )
    parser.add_argument(
        "--data-dir", default="/home/bcheng/SENTINEL/data",
        help="Base data directory (default: /home/bcheng/SENTINEL/data)",
    )
    parser.add_argument(
        "--anomaly-threshold", type=float, default=ANOMALY_ZSCORE_THRESH,
        help=f"Z-score threshold on derivative for anomaly labeling (default: {ANOMALY_ZSCORE_THRESH})",
    )
    parser.add_argument(
        "--seq-length", type=int, default=SEQ_LENGTH,
        help=f"Sequence window length (default: {SEQ_LENGTH})",
    )
    parser.add_argument(
        "--overlap", type=float, default=OVERLAP_FRAC,
        help=f"Overlap fraction between windows (default: {OVERLAP_FRAC})",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    raw_dir = data_dir / "raw" / "sensor" / "full"
    processed_dir = data_dir / "processed" / "sensor" / "full"
    processed_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        log(f"ERROR: Raw directory {raw_dir} does not exist.")
        sys.exit(1)

    parquet_files = sorted(raw_dir.glob("*.parquet"))
    log(f"Found {len(parquet_files)} raw parquet files in {raw_dir}")
    if len(parquet_files) == 0:
        log("Nothing to do.")
        sys.exit(0)

    # ------------------------------------------------------------------
    # Pass 1: Global statistics
    # ------------------------------------------------------------------
    global_stats = compute_global_stats(raw_dir)
    d_stats = compute_derivative_stats(raw_dir)

    # Save normalization + derivative stats for reproducibility
    stats_file = processed_dir / "normalization_stats.json"
    with open(stats_file, "w") as f:
        json.dump({"normalization": global_stats, "derivative": d_stats}, f, indent=2)
    log(f"Saved statistics to {stats_file}")

    # ------------------------------------------------------------------
    # Pass 2: Process each station
    # ------------------------------------------------------------------
    log(f"Pass 2: Processing {len(parquet_files)} stations into sequences ...")

    total_seqs = 0
    total_records = 0
    station_results = []
    anomaly_seq_count = 0

    for i, pf in enumerate(parquet_files):
        n_seqs, n_records = preprocess_station(
            pf, processed_dir, global_stats, d_stats,
        )
        total_seqs += n_seqs
        total_records += n_records

        if n_seqs > 0:
            # Quick count of sequences with any anomaly label
            for j in range(n_seqs):
                # We don't re-read here; approximate from station-level
                pass
            station_results.append({
                "site_no": pf.stem,
                "n_records": n_records,
                "n_sequences": n_seqs,
            })

        if (i + 1) % 25 == 0 or (i + 1) == len(parquet_files):
            log(
                f"  [{i + 1}/{len(parquet_files)}] "
                f"{total_seqs} sequences from {len(station_results)} stations"
            )

    # Count anomaly prevalence in output
    npz_files = sorted(processed_dir.glob("*.npz"))
    for nf in npz_files:
        d = np.load(nf)
        if (d["labels"] > 0).any():
            anomaly_seq_count += 1

    anomaly_pct = 100.0 * anomaly_seq_count / max(len(npz_files), 1)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    summary = {
        "total_sequences": total_seqs,
        "total_records": total_records,
        "total_stations_with_sequences": len(station_results),
        "total_parquet_files": len(parquet_files),
        "anomaly_sequences": anomaly_seq_count,
        "anomaly_pct": round(anomaly_pct, 2),
        "seq_length": args.seq_length,
        "overlap_fraction": args.overlap,
        "value_clamp": VALUE_CLAMP,
        "min_seq_data_fraction": MIN_SEQ_DATA,
        "anomaly_zscore_threshold": args.anomaly_threshold,
        "anomaly_expand_half_window": ANOMALY_WINDOW,
        "parameters": ALL_PARAMS,
        "normalization_stats": global_stats,
        "derivative_stats": d_stats,
        "stations": station_results,
    }
    summary_file = processed_dir / "preprocessing_stats.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    log("=" * 70)
    log("PREPROCESSING COMPLETE")
    log(f"  Stations processed:  {len(station_results)}")
    log(f"  Total sequences:     {total_seqs}")
    log(f"  Anomaly sequences:   {anomaly_seq_count} ({anomaly_pct:.1f}%)")
    log(f"  Raw data:            {raw_dir}")
    log(f"  Processed data:      {processed_dir}")
    log(f"  Stats:               {stats_file}")
    log("=" * 70)


if __name__ == "__main__":
    main()
