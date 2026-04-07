#!/usr/bin/env python3
"""Process downloaded USGS NWIS parquet files into AquaSSM training data.

Reads raw parquet files from data/raw/sensor/full/, applies quality filtering
and normalization, and produces training-ready .npz sequences for AquaSSM.

Output format per .npz:
  values: [T, 6] - normalized sensor values (DO, pH, SpCond, Temp, Turb, ORP)
  delta_ts: [T] - time gaps in seconds between consecutive observations
  timestamps: [T] - Unix timestamps
  labels: [T] - anomaly labels (0=normal, 1=anomaly based on EPA thresholds)
  station_id: str - USGS station identifier
  has_anomaly: bool - whether any timestep in the sequence is anomalous

MIT License — Bryan Cheng, 2026
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw/sensor/full")
OUT_DIR = Path("data/processed/sensor/real")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Canonical column names and USGS parameter codes
PARAM_MAP = {
    "00300": "DO",
    "00400": "pH",
    "00095": "SpCond",
    "00010": "Temp",
    "63680": "Turb",
    "00090": "ORP",
}
ALL_PARAMS = ["DO", "pH", "SpCond", "Temp", "Turb", "ORP"]

# EPA water quality thresholds for anomaly labeling
# Values outside these ranges are labeled as anomalous
ANOMALY_THRESHOLDS = {
    "DO": (2.0, 20.0),        # mg/L - fish kills below 2
    "pH": (5.0, 9.5),         # standard units
    "SpCond": (0, 5000),      # uS/cm - freshwater typical max
    "Temp": (0, 35),          # degC
    "Turb": (0, 500),         # NTU - severe turbidity
    "ORP": (-200, 700),       # mV
}

# Normalization constants (approximate from large USGS datasets)
NORM_MEAN = {"DO": 9.0, "pH": 7.5, "SpCond": 500, "Temp": 15, "Turb": 20, "ORP": 200}
NORM_STD = {"DO": 3.0, "pH": 1.0, "SpCond": 400, "Temp": 8, "Turb": 50, "ORP": 150}

SEQ_LENGTH = 512
OVERLAP = 128
MIN_VALID_FRAC = 0.4


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def process_parquet(parquet_path):
    """Process a single station parquet file into training sequences."""
    station_id = parquet_path.stem
    df = pd.read_parquet(parquet_path)

    if len(df) < 100:
        return 0

    # The parquets have named columns: Temp, SpCond, DO, pH, Turb
    # and a datetime index
    value_cols = {}
    for param in ALL_PARAMS:
        if param in df.columns:
            value_cols[param] = param

    if len(value_cols) < 3:
        return 0

    # Get datetime from index or column
    if isinstance(df.index, pd.DatetimeIndex) or df.index.name == "datetime":
        df = df.reset_index()
        df.rename(columns={df.columns[0]: "datetime"}, inplace=True)
    elif "datetime" in df.columns:
        pass
    else:
        for col in df.columns:
            if "date" in col.lower() or "time" in col.lower():
                df.rename(columns={col: "datetime"}, inplace=True)
                break
        else:
            return 0

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    df = df.dropna(subset=["datetime"]).sort_values("datetime")

    if len(df) < 100:
        return 0

    # Build values array [T, 6]
    T = len(df)
    values = np.full((T, 6), np.nan, dtype=np.float32)
    for i, param in enumerate(ALL_PARAMS):
        if param in value_cols:
            col = value_cols[param]
            vals = pd.to_numeric(df[col], errors="coerce").values
            values[:, i] = vals

    # Compute timestamps and delta_ts
    timestamps = df["datetime"].values.astype("int64") // 10**9  # Unix seconds
    timestamps = timestamps.astype(np.float64)
    delta_ts = np.zeros(T, dtype=np.float32)
    delta_ts[1:] = np.diff(timestamps).astype(np.float32)
    delta_ts = np.clip(delta_ts, 0, 86400)  # Cap at 1 day

    # Label anomalies based on EPA thresholds
    labels = np.zeros(T, dtype=np.int64)
    for i, param in enumerate(ALL_PARAMS):
        if param in ANOMALY_THRESHOLDS:
            lo, hi = ANOMALY_THRESHOLDS[param]
            v = values[:, i]
            anomalous = ((v < lo) | (v > hi)) & ~np.isnan(v)
            labels[anomalous] = 1

    # Normalize values
    for i, param in enumerate(ALL_PARAMS):
        if param in NORM_MEAN:
            values[:, i] = (values[:, i] - NORM_MEAN[param]) / NORM_STD[param]

    # Replace NaN with 0 after normalization
    values = np.nan_to_num(values, nan=0.0)
    values = np.clip(values, -5, 5)

    # Sliding window extraction
    n_seqs = 0
    step = SEQ_LENGTH - OVERLAP
    for start in range(0, T - SEQ_LENGTH + 1, step):
        end = start + SEQ_LENGTH
        v = values[start:end]
        dt = delta_ts[start:end]
        dt[0] = 0  # First delta_t is always 0
        ts_seq = timestamps[start:end]
        lb = labels[start:end]

        # Check validity
        valid_frac = np.mean(np.any(v != 0, axis=1))
        if valid_frac < MIN_VALID_FRAC:
            continue

        has_anomaly = int(lb.any())

        out_path = OUT_DIR / f"{station_id}_seq{n_seqs:04d}.npz"
        np.savez_compressed(
            out_path,
            values=v.astype(np.float32),
            delta_ts=dt.astype(np.float32),
            timestamps=ts_seq.astype(np.float64),
            labels=lb,
            station_id=station_id,
            has_anomaly=has_anomaly,
        )
        n_seqs += 1

    return n_seqs


def main():
    log("=" * 60)
    log("Processing USGS NWIS Parquets → AquaSSM Training Data")
    log("=" * 60)

    parquets = sorted(RAW_DIR.glob("*.parquet"))
    log(f"Found {len(parquets)} parquet files in {RAW_DIR}")

    # Check existing processed files
    existing = set(f.stem.split("_seq")[0] for f in OUT_DIR.glob("*.npz"))
    log(f"Already processed: {len(existing)} stations")

    total_seqs = 0
    processed = 0
    skipped = 0

    for i, pq in enumerate(parquets):
        station = pq.stem
        if station in existing:
            skipped += 1
            continue

        try:
            n = process_parquet(pq)
            total_seqs += n
            processed += 1
            if (processed) % 25 == 0:
                log(f"  Progress: {i+1}/{len(parquets)} (processed={processed}, seqs={total_seqs}, skipped={skipped})")
        except Exception as e:
            log(f"  Error processing {pq.name}: {e}")

    # Count total
    total_files = len(list(OUT_DIR.glob("*.npz")))
    anomaly_count = 0
    for f in OUT_DIR.glob("*.npz"):
        d = np.load(f, allow_pickle=True)
        if int(d.get("has_anomaly", 0)):
            anomaly_count += 1

    log(f"\nDone! Total sequences: {total_files}")
    log(f"  Normal: {total_files - anomaly_count}")
    log(f"  Anomalous: {anomaly_count}")
    log(f"  Anomaly rate: {anomaly_count / max(total_files, 1) * 100:.1f}%")

    # Save stats
    stats = {
        "total_parquets": len(parquets),
        "processed_stations": processed,
        "total_sequences": total_files,
        "anomalous_sequences": anomaly_count,
        "seq_length": SEQ_LENGTH,
        "overlap": OVERLAP,
    }
    with open(OUT_DIR / "processing_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    log("DONE")


if __name__ == "__main__":
    main()
