#!/usr/bin/env python3
"""
Expand HydroViT training data from ~2861 to ~5700 pairs using augmentation.

Augmentation strategies applied to training subset only (70% of 2861 = ~2003 samples):
  1. Spectral noise (×0.5 of train) → ~1001 new samples
  2. Spatial augmentation (×0.5 of train) → ~1001 new samples
  3. Seasonal simulation (×0.3 of train) → ~601 new samples

Total new: ~2603 augmented + 2861 original = ~5464 pairs (within 1.5–2× target)

PARAM_NAMES order (from parameter_head.py):
  0=chl_a, 1=turbidity, 2=secchi_depth, 3=cdom, 4=tss, 5=total_nitrogen,
  6=total_phosphorus, 7=dissolved_oxygen, 8=ammonia, 9=nitrate, 10=ph,
  11=water_temp, 12=phycocyanin, 13=oil_probability, 14=acdom, 15=pollution_anomaly_index

MIT License -- Bryan Cheng, 2026
"""

import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path("/home/bcheng/SENTINEL")
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data/processed/satellite"
INPUT_PATH = DATA_DIR / "paired_wq_v3.npz"
OUTPUT_PATH = DATA_DIR / "paired_wq_v5.npz"

WATER_TEMP_IDX = 11
# Band layout for Sentinel-2 (10 bands: B2,B3,B4,B8,B5,B6,B7,B8A,B11,B12)
# green=B3 idx=1, red=B4 idx=2, NIR=B8 idx=3, B8A idx=7
BAND_GREEN = 1
BAND_RED = 2
BAND_NIR = 3
BAND_NIR2 = 7

RNG = np.random.default_rng(42)


def load_data():
    print(f"Loading {INPUT_PATH} ...")
    t0 = time.time()
    d = np.load(str(INPUT_PATH))
    images = d["images"].astype(np.float32)   # (2861, 10, 224, 224)
    targets = d["targets"].astype(np.float32) # (2861, 16)
    print(f"  Loaded in {time.time()-t0:.1f}s: images={images.shape}, targets={targets.shape}")
    return images, targets


def get_train_indices(n_total, seed=42):
    """Reproduce the exact train/val/test split from train_hydrovit_wq_v6.py."""
    import torch
    n_train = max(1, int(0.7 * n_total))
    n_val = max(1, int(0.15 * n_total))
    n_test = n_total - n_train - n_val
    if n_test < 1:
        n_test = 1
        n_train = n_total - n_val - n_test

    indices = torch.randperm(n_total, generator=torch.Generator().manual_seed(seed)).numpy()
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    print(f"  Split: {len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test")
    return train_idx, val_idx, test_idx


# ─── Augmentation 1: Spectral noise ────────────────────────────────────────
def spectral_noise_augmentation(images, targets, n_aug):
    """Add Gaussian noise + random per-band scaling to simulate atmospheric variability."""
    print(f"  Spectral noise: augmenting {n_aug} samples ...")
    n_orig, C, H, W = images.shape

    # Compute per-band std across all training samples
    band_stds = images.std(axis=(0, 2, 3))  # shape (10,)

    chosen = RNG.integers(0, n_orig, size=n_aug)
    aug_images = images[chosen].copy()
    aug_targets = targets[chosen].copy()

    for b in range(C):
        noise_std = 0.02 * band_stds[b]
        noise = RNG.normal(0, noise_std, size=(n_aug, H, W)).astype(np.float32)
        scale = RNG.uniform(0.97, 1.03, size=(n_aug, 1, 1)).astype(np.float32)
        aug_images[:, b, :, :] = aug_images[:, b, :, :] * scale + noise

    # Clip to valid reflectance range (no negatives)
    aug_images = np.clip(aug_images, 0.0, None)
    return aug_images, aug_targets


# ─── Augmentation 2: Spatial augmentation ──────────────────────────────────
def spatial_augmentation(images, targets, n_aug):
    """Random flips + 90/180/270 degree rotations. WQ targets unchanged."""
    print(f"  Spatial augmentation: augmenting {n_aug} samples ...")
    n_orig = images.shape[0]

    chosen = RNG.integers(0, n_orig, size=n_aug)
    aug_images = images[chosen].copy()
    aug_targets = targets[chosen].copy()

    flip_h = RNG.random(n_aug) < 0.5
    flip_v = RNG.random(n_aug) < 0.5
    rot_k = RNG.integers(0, 4, size=n_aug)  # 0,1,2,3 → 0°,90°,180°,270°

    for i in range(n_aug):
        img = aug_images[i]
        if flip_h[i]:
            img = img[:, :, ::-1].copy()
        if flip_v[i]:
            img = img[:, ::-1, :].copy()
        if rot_k[i] > 0:
            img = np.rot90(img, k=rot_k[i], axes=(1, 2)).copy()
        aug_images[i] = img

    return aug_images, aug_targets


# ─── Augmentation 3: Seasonal simulation ───────────────────────────────────
def seasonal_augmentation(images, targets, n_aug):
    """Simulate seasonal spectral shifts (summer/winter) with adjusted water_temp."""
    print(f"  Seasonal simulation: augmenting {n_aug} samples ...")
    n_orig, C, H, W = images.shape

    chosen = RNG.integers(0, n_orig, size=n_aug)
    aug_images = images[chosen].copy()
    aug_targets = targets[chosen].copy()

    is_summer = RNG.random(n_aug) < 0.5

    for i in range(n_aug):
        img = aug_images[i]
        if is_summer[i]:
            # Summer: more algae → greener/redder, less NIR clarity
            g_fac = RNG.uniform(1.1, 1.3)
            r_fac = RNG.uniform(1.1, 1.3)
            nir_fac = RNG.uniform(0.8, 0.95)
            img[BAND_GREEN] *= g_fac
            img[BAND_RED] *= r_fac
            img[BAND_NIR] *= nir_fac
            img[BAND_NIR2] *= nir_fac
            # Adjust water_temp upward
            if not np.isnan(aug_targets[i, WATER_TEMP_IDX]):
                aug_targets[i, WATER_TEMP_IDX] += RNG.uniform(0, 8)
        else:
            # Winter: less algae → less green/red, more NIR
            g_fac = RNG.uniform(0.8, 0.95)
            r_fac = RNG.uniform(0.8, 0.95)
            nir_fac = RNG.uniform(1.05, 1.2)
            img[BAND_GREEN] *= g_fac
            img[BAND_RED] *= r_fac
            img[BAND_NIR] *= nir_fac
            img[BAND_NIR2] *= nir_fac
            # Adjust water_temp downward
            if not np.isnan(aug_targets[i, WATER_TEMP_IDX]):
                aug_targets[i, WATER_TEMP_IDX] -= RNG.uniform(0, 8)

        # Perturb other targets by ±10%
        for j in range(16):
            if j != WATER_TEMP_IDX and not np.isnan(aug_targets[i, j]):
                aug_targets[i, j] *= RNG.uniform(0.9, 1.1)

        aug_images[i] = np.clip(img, 0.0, None)

    return aug_images, aug_targets


def main():
    t0 = time.time()
    print("=" * 60)
    print("HydroViT Data Expansion v5")
    print("=" * 60)

    images, targets = load_data()
    n_total = len(images)

    train_idx, val_idx, test_idx = get_train_indices(n_total)
    n_train = len(train_idx)

    train_images = images[train_idx]
    train_targets = targets[train_idx]

    # Compute augmentation counts
    n_spectral = int(round(0.5 * n_train))   # ~1001
    n_spatial = int(round(0.5 * n_train))    # ~1001
    n_seasonal = int(round(0.3 * n_train))   # ~601

    print(f"\nAugmentation plan:")
    print(f"  Original samples:         {n_total}")
    print(f"  Training subset:          {n_train}")
    print(f"  Spectral noise augs:      {n_spectral}")
    print(f"  Spatial augs:             {n_spatial}")
    print(f"  Seasonal simulation augs: {n_seasonal}")
    total_new = n_total + n_spectral + n_spatial + n_seasonal
    print(f"  Expected total:           {total_new}")

    # Run augmentations
    spec_imgs, spec_tgts = spectral_noise_augmentation(train_images, train_targets, n_spectral)
    spat_imgs, spat_tgts = spatial_augmentation(train_images, train_targets, n_spatial)
    seas_imgs, seas_tgts = seasonal_augmentation(train_images, train_targets, n_seasonal)

    # Concatenate: originals + augmented
    all_images = np.concatenate([images, spec_imgs, spat_imgs, seas_imgs], axis=0)
    all_targets = np.concatenate([targets, spec_tgts, spat_tgts, seas_tgts], axis=0)

    print(f"\nFinal dataset: {all_images.shape} images, {all_targets.shape} targets")
    print(f"Expansion ratio: {len(all_images)/n_total:.2f}x")

    print(f"Saving to {OUTPUT_PATH} ...")
    t_save = time.time()
    np.savez_compressed(str(OUTPUT_PATH), images=all_images, targets=all_targets)
    print(f"Saved in {time.time()-t_save:.1f}s")

    total_elapsed = time.time() - t0
    print(f"\nTotal elapsed: {total_elapsed/60:.1f} min")
    print("Data expansion complete.")


if __name__ == "__main__":
    main()
