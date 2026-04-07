#!/usr/bin/env python3
"""Generate synthetic data for all 5 modalities for fusion smoke testing.

Creates minimal but realistic synthetic data so we can test the full
SENTINEL pipeline end-to-end without waiting for real data downloads.

MIT License — Bryan Cheng, 2026
"""

import json
from pathlib import Path

import numpy as np
import torch

np.random.seed(42)

BASE_DIR = Path("data/processed/synthetic_multimodal")
BASE_DIR.mkdir(parents=True, exist_ok=True)

N_SAMPLES = 100  # Per modality
N_SITES = 20     # Simulated monitoring sites


def generate_sensor_data():
    """Sensor time series: (T, 6) — DO, pH, SpCond, Temp, Turb, ORP."""
    out_dir = BASE_DIR / "sensor"
    out_dir.mkdir(exist_ok=True)

    for i in range(N_SAMPLES):
        T = np.random.randint(200, 500)
        values = np.random.randn(T, 6).astype(np.float32)
        delta_ts = np.random.choice([900, 1800, 3600], size=T).astype(np.float32)
        delta_ts[0] = 0
        labels = np.zeros(T, dtype=np.int64)

        # 50% chance of anomaly
        if np.random.rand() > 0.5:
            start = np.random.randint(T // 4, T // 2)
            end = min(start + T // 5, T)
            values[start:end, 0] -= 2.5  # DO drop
            values[start:end, 4] += 3.0  # Turbidity spike
            labels[start:end] = 1

        np.savez_compressed(
            out_dir / f"sensor_{i:04d}.npz",
            values=values, delta_ts=delta_ts, labels=labels,
            site_id=i % N_SITES, has_anomaly=int((labels > 0).any()),
        )
    print(f"Sensor: {N_SAMPLES} sequences")


def generate_satellite_data():
    """Satellite tiles: (13, 64, 64) — 13-band Sentinel-2 imagery."""
    out_dir = BASE_DIR / "satellite"
    out_dir.mkdir(exist_ok=True)

    # 16 water quality parameters as regression targets
    param_names = [
        "chla", "turbidity", "secchi", "cdom", "tss", "tn", "tp", "do",
        "ammonia", "nitrate", "ph", "temp", "phycocyanin", "oil_prob",
        "acdom", "pai",
    ]

    for i in range(N_SAMPLES):
        # Simulated satellite tile (smaller for speed: 64x64 instead of 224x224)
        image = np.random.randn(13, 64, 64).astype(np.float32) * 0.1 + 0.3
        image = np.clip(image, 0, 1)  # reflectance [0, 1]

        # Random water quality targets (some NaN for missing)
        targets = np.random.randn(16).astype(np.float32)
        mask = np.random.rand(16) > 0.3  # 30% missing
        targets[~mask] = np.nan

        np.savez_compressed(
            out_dir / f"satellite_{i:04d}.npz",
            image=image, targets=targets, param_names=param_names,
            site_id=i % N_SITES, cloud_fraction=np.random.rand(),
        )
    print(f"Satellite: {N_SAMPLES} tiles")


def generate_microbial_data():
    """Microbial community: (5000,) CLR-transformed OTU abundances."""
    out_dir = BASE_DIR / "microbial"
    out_dir.mkdir(exist_ok=True)

    source_types = ["nutrient", "heavy_metals", "sewage", "oil_petrochemical",
                     "thermal", "sediment", "pharmaceutical", "acid_mine"]

    for i in range(N_SAMPLES):
        # Sparse CLR abundances (most taxa absent)
        n_otus = 5000
        abundances = np.zeros(n_otus, dtype=np.float32)
        n_present = np.random.randint(50, 500)
        present_idx = np.random.choice(n_otus, n_present, replace=False)
        abundances[present_idx] = np.random.randn(n_present).astype(np.float32)

        # Source label
        source_label = np.random.randint(0, len(source_types))

        np.savez_compressed(
            out_dir / f"microbial_{i:04d}.npz",
            abundances=abundances,
            source_label=source_label,
            source_name=source_types[source_label],
            site_id=i % N_SITES,
        )
    print(f"Microbial: {N_SAMPLES} samples")


def generate_molecular_data():
    """Molecular expression: (1000,) gene expression values."""
    out_dir = BASE_DIR / "molecular"
    out_dir.mkdir(exist_ok=True)

    contaminant_classes = ["pah", "heavy_metal", "endocrine", "pesticide",
                           "pharmaceutical", "solvent", "nutrient"]

    for i in range(N_SAMPLES):
        n_genes = 1000
        expression = np.random.randn(n_genes).astype(np.float32)

        # Contaminant label
        label = np.random.randint(0, len(contaminant_classes))

        np.savez_compressed(
            out_dir / f"molecular_{i:04d}.npz",
            expression=expression,
            contaminant_label=label,
            contaminant_name=contaminant_classes[label],
            site_id=i % N_SITES,
        )
    print(f"Molecular: {N_SAMPLES} samples")


def generate_behavioral_data():
    """Behavioral trajectories: (T, n_keypoints, 2) pose sequences."""
    out_dir = BASE_DIR / "behavioral"
    out_dir.mkdir(exist_ok=True)

    for i in range(N_SAMPLES):
        T = np.random.randint(50, 200)
        n_keypoints = 12  # Daphnia

        # Normal trajectory: smooth movement
        keypoints = np.cumsum(np.random.randn(T, n_keypoints, 2) * 0.1, axis=0).astype(np.float32)
        features = np.random.randn(T, 16).astype(np.float32)  # velocity, acceleration, etc.
        timestamps = np.arange(T, dtype=np.float32) / 30.0  # 30 FPS

        # 50% anomalous
        is_anomaly = int(np.random.rand() > 0.5)
        if is_anomaly:
            start = np.random.randint(T // 4, T // 2)
            keypoints[start:] += np.random.randn(T - start, n_keypoints, 2) * 2.0  # erratic
            features[start:, 0] *= 3.0  # velocity spike

        np.savez_compressed(
            out_dir / f"behavioral_{i:04d}.npz",
            keypoints=keypoints,
            features=features,
            timestamps=timestamps,
            is_anomaly=is_anomaly,
            organism="daphnia",
            site_id=i % N_SITES,
        )
    print(f"Behavioral: {N_SAMPLES} trajectories")


def generate_multimodal_index():
    """Create index linking samples across modalities by site_id."""
    index = {}
    for site_id in range(N_SITES):
        index[str(site_id)] = {
            "sensor": [f"sensor_{i:04d}" for i in range(N_SAMPLES) if i % N_SITES == site_id],
            "satellite": [f"satellite_{i:04d}" for i in range(N_SAMPLES) if i % N_SITES == site_id],
            "microbial": [f"microbial_{i:04d}" for i in range(N_SAMPLES) if i % N_SITES == site_id],
            "molecular": [f"molecular_{i:04d}" for i in range(N_SAMPLES) if i % N_SITES == site_id],
            "behavioral": [f"behavioral_{i:04d}" for i in range(N_SAMPLES) if i % N_SITES == site_id],
        }

    with open(BASE_DIR / "multimodal_index.json", "w") as f:
        json.dump(index, f, indent=2)
    print(f"Multimodal index: {N_SITES} sites, {N_SAMPLES} samples per modality")


if __name__ == "__main__":
    print("Generating synthetic multimodal data for SENTINEL...")
    generate_sensor_data()
    generate_satellite_data()
    generate_microbial_data()
    generate_molecular_data()
    generate_behavioral_data()
    generate_multimodal_index()
    print(f"\nAll data saved to {BASE_DIR}")
