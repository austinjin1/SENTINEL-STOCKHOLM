#!/usr/bin/env python3
"""Generate synthetic sensor data for AquaSSM smoke testing.

Creates realistic water quality time series with known anomalies
for rapid model development and testing before real data is available.

MIT License — Bryan Cheng, 2026
"""

import json
from pathlib import Path

import numpy as np

np.random.seed(42)

OUTPUT_DIR = Path("data/processed/sensor/synthetic")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_STATIONS = 50
SEQ_LENGTH = 2000
N_PARAMS = 6  # DO, pH, SpCond, Temp, Turb, ORP

# Realistic parameter ranges (after z-score normalization, centered ~0)
PARAM_NAMES = ["DO", "pH", "SpCond", "Temp", "Turb", "ORP"]

# Baseline statistics (unnormalized) for generating realistic patterns
PARAM_STATS = {
    "DO": {"mean": 8.5, "std": 2.0, "min": 0, "max": 15},
    "pH": {"mean": 7.5, "std": 0.5, "min": 5, "max": 9},
    "SpCond": {"mean": 400, "std": 200, "min": 50, "max": 2000},
    "Temp": {"mean": 15, "std": 8, "min": 0, "max": 35},
    "Turb": {"mean": 10, "std": 20, "min": 0, "max": 500},
    "ORP": {"mean": 200, "std": 100, "min": -200, "max": 600},
}


def generate_station_timeseries(n_steps, irregular=True):
    """Generate a realistic water quality time series."""
    # Time gaps (irregular sampling)
    if irregular:
        # Mix of regular 15-min and random gaps
        delta_ts = np.random.choice(
            [900, 900, 900, 900, 1800, 3600, 7200, 14400],  # 15min, 30min, 1h, 2h, 4h
            size=n_steps,
            p=[0.6, 0.15, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01],
        ).astype(np.float32)
    else:
        delta_ts = np.full(n_steps, 900, dtype=np.float32)
    delta_ts[0] = 0.0

    # Generate correlated parameters with diurnal patterns
    t = np.cumsum(delta_ts) / 3600  # hours
    values = np.zeros((n_steps, N_PARAMS), dtype=np.float32)

    # Temperature: diurnal cycle
    temp = 15 + 8 * np.sin(2 * np.pi * t / 24) + np.random.randn(n_steps) * 0.5
    values[:, 3] = temp

    # DO: inversely correlated with temperature + diurnal photosynthesis
    do = 14.6 - 0.39 * temp + 1.5 * np.sin(2 * np.pi * t / 24 + np.pi/3) + np.random.randn(n_steps) * 0.3
    values[:, 0] = do

    # pH: slight diurnal from photosynthesis
    values[:, 1] = 7.5 + 0.3 * np.sin(2 * np.pi * t / 24) + np.random.randn(n_steps) * 0.1

    # SpCond: slow drift + noise
    values[:, 2] = 400 + 50 * np.sin(2 * np.pi * t / (24 * 30)) + np.random.randn(n_steps) * 20

    # Turb: base + storm events
    turb_base = 5 + np.abs(np.random.randn(n_steps)) * 2
    values[:, 4] = turb_base

    # ORP: correlated with DO
    values[:, 5] = 200 + 50 * (do - 8.5) / 2.0 + np.random.randn(n_steps) * 10

    # Z-score normalize
    for i in range(N_PARAMS):
        mean = values[:, i].mean()
        std = values[:, i].std() + 1e-6
        values[:, i] = (values[:, i] - mean) / std

    # Mask: ~95% valid
    mask = np.random.rand(n_steps, N_PARAMS) > 0.05

    return values, delta_ts, mask


def inject_anomaly(values, delta_ts, anomaly_type="contamination"):
    """Inject a known anomaly into the time series."""
    n_steps = len(values)
    labels = np.zeros(n_steps, dtype=np.int64)  # 0=normal

    # Anomaly window: 10-20% of sequence
    start = np.random.randint(n_steps // 4, n_steps // 2)
    duration = np.random.randint(n_steps // 10, n_steps // 5)
    end = min(start + duration, n_steps)

    if anomaly_type == "contamination":
        # DO drops, turbidity spikes, pH shifts
        values[start:end, 0] -= 2.5  # DO drop
        values[start:end, 4] += 3.0  # Turbidity spike
        values[start:end, 1] -= 1.0  # pH drop
        labels[start:end] = 1
    elif anomaly_type == "sensor_drift":
        # Gradual drift in one parameter
        drift = np.linspace(0, 3.0, end - start)
        param_idx = np.random.randint(0, N_PARAMS)
        values[start:end, param_idx] += drift
        labels[start:end] = 2
    elif anomaly_type == "thermal":
        # Temperature spike
        values[start:end, 3] += 3.0
        values[start:end, 0] -= 1.5  # DO drops with temp
        labels[start:end] = 3

    return values, labels


def main():
    total_normal = 0
    total_anomaly = 0

    for i in range(N_STATIONS):
        values, delta_ts, mask = generate_station_timeseries(SEQ_LENGTH)

        if i < N_STATIONS // 2:
            # Normal station
            labels = np.zeros(SEQ_LENGTH, dtype=np.int64)
            np.savez_compressed(
                OUTPUT_DIR / f"normal_{i:04d}.npz",
                values=values,
                delta_ts=delta_ts,
                mask=mask,
                labels=labels,
                site_no=f"synthetic_normal_{i:04d}",
            )
            total_normal += 1
        else:
            # Anomaly station
            anom_type = np.random.choice(["contamination", "sensor_drift", "thermal"])
            values, labels = inject_anomaly(values, delta_ts, anom_type)
            np.savez_compressed(
                OUTPUT_DIR / f"anomaly_{i:04d}.npz",
                values=values,
                delta_ts=delta_ts,
                mask=mask,
                labels=labels,
                anomaly_type=anom_type,
                site_no=f"synthetic_anomaly_{i:04d}",
            )
            total_anomaly += 1

    stats = {
        "total_sequences": N_STATIONS,
        "normal": total_normal,
        "anomaly": total_anomaly,
        "seq_length": SEQ_LENGTH,
        "n_params": N_PARAMS,
    }
    with open(OUTPUT_DIR / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Generated {total_normal} normal + {total_anomaly} anomaly sequences")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
