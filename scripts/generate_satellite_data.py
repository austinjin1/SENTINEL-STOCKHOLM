#!/usr/bin/env python3
"""Generate realistic synthetic satellite data for HydroViT training.

Real Sentinel-2/3 data requires Google Earth Engine auth. This generates
realistic water pixel patches with known water quality parameters for
training and evaluation.

MIT License — Bryan Cheng, 2026
"""

import json
from pathlib import Path

import numpy as np

np.random.seed(2026)

OUTPUT_DIR = Path("data/processed/satellite")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_SAMPLES = 500
IMG_SIZE = 224  # ViT input size
N_BANDS = 13    # 10 S2 + 3 S3 bands

# 16 water quality parameters
PARAM_NAMES = [
    "chla", "turbidity", "secchi", "cdom", "tss", "tn", "tp", "do",
    "ammonia", "nitrate", "ph", "temp", "phycocyanin", "oil_prob",
    "acdom", "pai",
]

# Realistic parameter statistics (for generating targets)
PARAM_STATS = {
    "chla":        {"mean": 15.0,  "std": 20.0,  "min": 0.1,   "max": 300.0},
    "turbidity":   {"mean": 10.0,  "std": 15.0,  "min": 0.1,   "max": 200.0},
    "secchi":      {"mean": 2.0,   "std": 1.5,   "min": 0.05,  "max": 10.0},
    "cdom":        {"mean": 3.0,   "std": 2.0,   "min": 0.1,   "max": 20.0},
    "tss":         {"mean": 20.0,  "std": 25.0,  "min": 0.5,   "max": 200.0},
    "tn":          {"mean": 1.5,   "std": 1.0,   "min": 0.05,  "max": 10.0},
    "tp":          {"mean": 0.05,  "std": 0.05,  "min": 0.001, "max": 0.5},
    "do":          {"mean": 8.5,   "std": 2.0,   "min": 0.0,   "max": 15.0},
    "ammonia":     {"mean": 0.1,   "std": 0.15,  "min": 0.001, "max": 2.0},
    "nitrate":     {"mean": 1.0,   "std": 1.5,   "min": 0.01,  "max": 10.0},
    "ph":          {"mean": 7.5,   "std": 0.5,   "min": 5.0,   "max": 9.5},
    "temp":        {"mean": 18.0,  "std": 8.0,   "min": 0.0,   "max": 35.0},
    "phycocyanin": {"mean": 2.0,   "std": 5.0,   "min": 0.0,   "max": 50.0},
    "oil_prob":    {"mean": 0.02,  "std": 0.05,  "min": 0.0,   "max": 1.0},
    "acdom":       {"mean": 1.0,   "std": 0.8,   "min": 0.1,   "max": 5.0},
    "pai":         {"mean": 0.0,   "std": 1.0,   "min": -3.0,  "max": 5.0},
}

# Sentinel-2 band center wavelengths (nm) for the 10 bands we use
S2_BANDS = [490, 560, 665, 705, 740, 783, 842, 865, 1610, 2190]
# Plus 3 S3 OLCI bands
S3_BANDS = [412, 443, 510]


def generate_water_reflectance(wq_params):
    """Generate synthetic water reflectance spectra from WQ parameters.

    Uses simplified bio-optical relationships:
    - High chlorophyll → higher green reflectance, lower blue
    - High turbidity → higher reflectance across all bands
    - High CDOM → lower blue reflectance
    """
    chla = wq_params[0]  # chlorophyll-a
    turb = wq_params[1]  # turbidity
    cdom = wq_params[3]  # CDOM

    # Base water reflectance (low, typical of clear water)
    base = np.array([0.02, 0.03, 0.01, 0.008, 0.005, 0.003, 0.002, 0.001, 0.0005, 0.0002,
                     0.025, 0.028, 0.032])  # 10 S2 + 3 S3

    # Chlorophyll effect: boost green (560nm), absorption at blue and red
    chla_effect = np.array([-0.001, 0.002, -0.001, 0.001, 0.001, 0.0005, 0.0002, 0.0001, 0, 0,
                            -0.001, -0.0005, 0.001]) * np.log1p(chla)

    # Turbidity: increases reflectance across all bands
    turb_effect = base * 0.1 * np.log1p(turb)

    # CDOM: absorbs blue light
    cdom_effect = np.array([-0.003, -0.001, 0, 0, 0, 0, 0, 0, 0, 0,
                            -0.004, -0.003, -0.001]) * np.log1p(cdom)

    reflectance = base + chla_effect + turb_effect + cdom_effect
    reflectance = np.clip(reflectance, 0.0001, 0.5)

    return reflectance


def generate_tile(reflectance, size=224):
    """Generate a 2D tile with spatial variation around base reflectance."""
    tile = np.zeros((len(reflectance), size, size), dtype=np.float32)

    # Smooth spatial variation (Gaussian random field approximation)
    for b in range(len(reflectance)):
        # Base + low-frequency spatial variation
        noise = np.random.randn(size // 16, size // 16) * reflectance[b] * 0.2
        # Upsample with interpolation
        from scipy.ndimage import zoom
        noise_full = zoom(noise, 16, order=1)[:size, :size]
        tile[b] = reflectance[b] + noise_full

    tile = np.clip(tile, 0.0001, 0.5)
    return tile


def main():
    print(f"Generating {N_SAMPLES} synthetic satellite tiles...")

    stats = {"n_samples": N_SAMPLES, "n_bands": N_BANDS, "img_size": IMG_SIZE}

    for i in range(N_SAMPLES):
        # Generate random water quality parameters
        targets = np.zeros(16, dtype=np.float32)
        for j, name in enumerate(PARAM_NAMES):
            s = PARAM_STATS[name]
            val = np.random.lognormal(
                np.log(s["mean"]), s["std"] / s["mean"]
            ) if s["mean"] > 0 else np.random.randn() * s["std"]
            targets[j] = np.clip(val, s["min"], s["max"])

        # Generate reflectance from WQ parameters
        reflectance = generate_water_reflectance(targets)

        # Generate spatial tile
        tile = generate_tile(reflectance, IMG_SIZE)

        # Random cloud fraction
        cloud_frac = np.random.beta(2, 10)  # mostly clear

        # Random missing parameters (30% chance each is NaN)
        mask = np.random.rand(16) > 0.3
        targets_masked = targets.copy()
        targets_masked[~mask] = np.nan

        # Anomaly flag (10% pollution events)
        is_anomaly = int(np.random.rand() < 0.1)
        if is_anomaly:
            # Boost turbidity and reduce Secchi (pollution signature)
            targets[1] *= 5  # turbidity spike
            targets[2] /= 3  # secchi drop
            targets[15] = 3.0  # high PAI
            tile[2] *= 1.5  # red reflectance boost (sediment)
            tile[4] *= 1.3  # NIR boost

        np.savez_compressed(
            OUTPUT_DIR / f"tile_{i:05d}.npz",
            image=tile,           # (13, 224, 224)
            targets=targets_masked,  # (16,) with NaN for missing
            targets_full=targets,    # (16,) no NaN (for evaluation)
            cloud_fraction=cloud_frac,
            is_anomaly=is_anomaly,
            param_names=PARAM_NAMES,
        )

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{N_SAMPLES}")

    # Save metadata
    stats["param_names"] = PARAM_NAMES
    stats["param_stats"] = PARAM_STATS
    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Done. Saved {N_SAMPLES} tiles to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
