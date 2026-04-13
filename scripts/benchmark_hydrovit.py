#!/usr/bin/env python3
"""
Benchmark HydroViT v7 against baseline models on water temperature prediction.

Baselines:
  1. Ridge Regression (band means only)
  2. Random Forest (band means + stds + 5 ratios)
  3. ViT-no-pretraining (same architecture, random init)
  4. CNN baseline (4-layer conv)

All models evaluated on the same test split as train_hydrovit_wq_v7.py.

MIT License -- Bryan Cheng, 2026
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset, random_split

PROJECT_ROOT = Path("/home/bcheng/SENTINEL")
sys.path.insert(0, str(PROJECT_ROOT))

from sentinel.models.satellite_encoder.model import SatelliteEncoder
from sentinel.models.satellite_encoder.parameter_head import (
    WaterQualityHead,
    PARAM_NAMES,
    NUM_WATER_PARAMS,
)
from sentinel.utils.logging import get_logger

logger = get_logger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_DIR = Path("checkpoints/satellite")
RESULTS_DIR = Path("results/benchmarks")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATA_BASE = Path("data/processed/satellite")
IMAGES_PATH = DATA_BASE / "v4_real_images.npy"
TARGETS_PATH = DATA_BASE / "v4_real_targets.npy"
HYDROVIT_V7_CKPT = CKPT_DIR / "hydrovit_wq_v7.pt"
HYDROVIT_V7_RESULTS = CKPT_DIR / "results_wq_v7.json"

WATER_TEMP_IDX = PARAM_NAMES.index("water_temp")  # = 11
BATCH_SIZE = 4
BASELINE_EPOCHS = 60
USE_AMP = True
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
EARLY_STOP_PATIENCE = 10


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
class PairedWQDataset(Dataset):
    """Same normalization as train_hydrovit_wq_v7.py.

    Loads from v4_real_images.npy (float16 mmap, 13 bands) and
    v4_real_targets.npy. Returns both 13-band image (for ViT models)
    and 10-band raw image (for CNN/classical baselines — first 10 bands,
    which are the S2 bands; bands 10-12 are zero-padded for v3 samples).
    """

    def __init__(self, images_path: str, targets_path: str):
        self.images = np.load(images_path, mmap_mode='r')
        self.targets = np.load(targets_path).astype(np.float32)

        self.log_params = {0, 1, 3, 4, 5, 6, 8, 9, 12, 14}
        self.targets_norm = self.targets.copy()
        for i in self.log_params:
            valid = ~np.isnan(self.targets_norm[:, i])
            if valid.any():
                vals = self.targets_norm[valid, i]
                vals = np.maximum(vals, 1e-6)
                self.targets_norm[valid, i] = np.log1p(vals)

        self.target_mean = np.nanmean(self.targets_norm, axis=0)
        self.target_std = np.nanstd(self.targets_norm, axis=0)
        self.target_std[self.target_std < 1e-6] = 1.0
        nan_cols = np.all(np.isnan(self.targets_norm), axis=0)
        self.target_mean[nan_cols] = 0.0
        self.target_std[nan_cols] = 1.0

        for i in range(16):
            valid = ~np.isnan(self.targets_norm[:, i])
            if valid.any():
                self.targets_norm[valid, i] = (
                    (self.targets_norm[valid, i] - self.target_mean[i])
                    / self.target_std[i]
                )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_f32 = np.array(self.images[idx], dtype=np.float32)
        image13 = torch.tensor(img_f32)          # (13, 224, 224) — for ViT
        raw_image = torch.tensor(img_f32[:10])   # (10, 224, 224) — first 10 S2 bands
        targets = torch.tensor(self.targets_norm[idx])
        return {"image": image13, "targets": targets, "raw_image": raw_image}


def get_split_indices(n_total, seed=42):
    """Reproduce the same train/val/test split as train_hydrovit_wq_v7.py."""
    n_train = max(1, int(0.7 * n_total))
    n_val = max(1, int(0.15 * n_total))
    n_test = n_total - n_train - n_val
    if n_test < 1:
        n_test = 1
        n_train = n_total - n_val - n_test
    return n_train, n_val, n_test


def compute_r2(preds, targets):
    """Compute R² for 1D arrays."""
    ss_res = ((preds - targets) ** 2).sum()
    ss_tot = ((targets - targets.mean()) ** 2).sum()
    if ss_tot < 1e-8:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def compute_r2_per_param(preds_dict, tgts_dict):
    r2_scores = {}
    for j in range(NUM_WATER_PARAMS):
        if preds_dict[j]:
            p = torch.cat(preds_dict[j]).numpy()
            t = torch.cat(tgts_dict[j]).numpy()
            if len(p) < 2:
                r2_scores[PARAM_NAMES[j]] = float("nan")
                continue
            r2_scores[PARAM_NAMES[j]] = compute_r2(p, t)
        else:
            r2_scores[PARAM_NAMES[j]] = float("nan")
    return r2_scores


# ---------------------------------------------------------------------------
# Feature extraction for classical ML baselines
# ---------------------------------------------------------------------------
def extract_features(images_raw):
    """
    Extract hand-crafted features from raw images (N, 10, 224, 224).
    Returns feature matrix of shape (N, 25):
      - 10 band means
      - 10 band stds
      - 5 band ratios (NDWI, NDVI, red/NIR, green/SWIR, NIR/SWIR)
    """
    N = images_raw.shape[0]
    C = images_raw.shape[1]

    # Spatial mean and std per band
    means = images_raw.mean(axis=(2, 3))   # (N, 10)
    stds  = images_raw.std(axis=(2, 3))    # (N, 10)

    eps = 1e-8
    # Sentinel-2 band layout: B2=0,B3=1,B4=2,B8=3,B5=4,B6=5,B7=6,B8A=7,B11=8,B12=9
    B3  = means[:, 1]  # green
    B4  = means[:, 2]  # red
    B8  = means[:, 3]  # NIR
    B11 = means[:, 8]  # SWIR1
    B12 = means[:, 9]  # SWIR2

    # NDWI (Green-NIR)/(Green+NIR) — water index
    ndwi = (B3 - B8) / (B3 + B8 + eps)
    # NDVI (NIR-Red)/(NIR+Red)
    ndvi = (B8 - B4) / (B8 + B4 + eps)
    # Red/NIR ratio
    red_nir = B4 / (B8 + eps)
    # Green/SWIR1
    green_swir = B3 / (B11 + eps)
    # NIR/SWIR2
    nir_swir = B8 / (B12 + eps)

    ratios = np.stack([ndwi, ndvi, red_nir, green_swir, nir_swir], axis=1)  # (N,5)
    features = np.concatenate([means, stds, ratios], axis=1)  # (N,25)
    return features.astype(np.float32)


# ---------------------------------------------------------------------------
# Neural model evaluation helper
# ---------------------------------------------------------------------------
def evaluate_neural(model, dataloader, device, is_satellite_encoder=True):
    model.eval()
    preds = {j: [] for j in range(NUM_WATER_PARAMS)}
    tgts  = {j: [] for j in range(NUM_WATER_PARAMS)}
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            if is_satellite_encoder:
                image = batch["image"].to(device)
            else:
                image = batch["raw_image"].to(device)
            targets = batch["targets"].to(device)

            out = model(image)
            if is_satellite_encoder:
                wq = out["water_quality_params"]
                unc = out["param_uncertainty"]
                loss = WaterQualityHead.gaussian_nll_loss(wq, unc, targets)
            else:
                wq = out  # CNN/ViT-scratch returns (N, 16)
                unc = torch.ones_like(wq) * 0.5
                loss = WaterQualityHead.gaussian_nll_loss(wq, unc, targets)

            if not torch.isnan(loss):
                total_loss += loss.item()
                n_batches += 1

            valid = ~torch.isnan(targets)
            for j in range(NUM_WATER_PARAMS):
                mask = valid[:, j]
                if mask.sum() > 0:
                    preds[j].append(wq[:, j][mask].cpu())
                    tgts[j].append(targets[:, j][mask].cpu())

    r2_scores = compute_r2_per_param(preds, tgts)
    return r2_scores, total_loss / max(n_batches, 1)


def train_neural(model, train_dl, val_dl, optimizer, scheduler, epochs,
                 name, device, is_satellite_encoder=False):
    best_val_r2 = -float("inf")
    best_state = None
    no_improve = 0
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP and device.type == "cuda")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_dl:
            if is_satellite_encoder:
                image = batch["image"].to(device)
            else:
                image = batch["raw_image"].to(device)
            targets = batch["targets"].to(device)

            with torch.amp.autocast("cuda", enabled=USE_AMP and device.type == "cuda"):
                out = model(image)
                if is_satellite_encoder:
                    wq = out["water_quality_params"]
                    unc = out["param_uncertainty"]
                else:
                    wq = out
                    unc = torch.ones_like(wq) * 0.5

                valid = ~torch.isnan(targets)
                if valid.sum() == 0:
                    continue

                loss = WaterQualityHead.gaussian_nll_loss(wq, unc, targets)
                if torch.isnan(loss):
                    optimizer.zero_grad()
                    continue

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            n_batches += 1

        if scheduler is not None:
            scheduler.step()

        r2_scores, val_loss = evaluate_neural(model, val_dl, device, is_satellite_encoder)
        valid_r2s = [v for v in r2_scores.values() if not np.isnan(v)]
        mean_r2 = np.mean(valid_r2s) if valid_r2s else -1.0
        wt_r2 = r2_scores.get("water_temp", float("nan"))

        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1:
            train_loss = total_loss / max(n_batches, 1)
            logger.info(
                f"[{name}] Ep {epoch+1:3d}/{epochs} | "
                f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                f"Mean R2: {mean_r2:.4f} | water_temp R2: {wt_r2:.4f}"
            )

        if mean_r2 > best_val_r2:
            best_val_r2 = mean_r2
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                logger.info(f"[{name}] Early stop at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return best_val_r2


# ---------------------------------------------------------------------------
# CNN Baseline model
# ---------------------------------------------------------------------------
class CNNBaseline(nn.Module):
    """Simple 4-layer CNN: Conv 32→64→128→256 + FC→16."""

    def __init__(self, in_channels=10, num_outputs=16):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                             # 224→112
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                             # 112→56
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                             # 56→28
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),                     # 28→4
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_outputs),
        )

    def forward(self, x):
        return self.head(self.features(x))


# ---------------------------------------------------------------------------
# ViT-no-pretraining: same SatelliteEncoder architecture, random weights
# ---------------------------------------------------------------------------
class ViTNoPretrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SatelliteEncoder(pretrained=False)
        # No checkpoint loaded — random init

    def forward(self, x):
        out = self.encoder(x)
        return out["water_quality_params"]


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("=" * 60)
    print("HydroViT Benchmark")
    print("=" * 60)

    for p in [IMAGES_PATH, TARGETS_PATH]:
        if not p.exists():
            logger.error(f"Data file not found: {p}")
            logger.error("Run scripts/train_hydrovit_wq_v7.py setup first.")
            sys.exit(1)

    logger.info(f"Loading real data from {IMAGES_PATH}")
    dataset = PairedWQDataset(str(IMAGES_PATH), str(TARGETS_PATH))
    n = len(dataset)
    n_train, n_val, n_test = get_split_indices(n)

    # Same split as train_hydrovit_wq_v7.py (seed=42)
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42),
    )
    logger.info(f"Split: {n_train} train / {n_val} val / {n_test} test")

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, num_workers=2)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, num_workers=2)

    results = {}

    # ── 1. Load HydroViT v7 results (already trained) ────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("Loading SENTINEL HydroViT v7 results")
    logger.info("=" * 60)

    if HYDROVIT_V7_RESULTS.exists():
        with open(HYDROVIT_V7_RESULTS) as f:
            v7_res = json.load(f)
        results["SENTINEL_HydroViT_v7"] = {
            "water_temp_r2": v7_res.get("water_temp_r2"),
            "mean_r2": v7_res.get("mean_r2"),
        }
        logger.info(f"  water_temp R2: {v7_res.get('water_temp_r2'):.4f}")
        logger.info(f"  mean R2:       {v7_res.get('mean_r2'):.4f}")
    else:
        logger.warning("HydroViT v7 results not found, will evaluate from checkpoint")
        if HYDROVIT_V7_CKPT.exists():
            model_v7 = SatelliteEncoder(pretrained=False).to(DEVICE)
            state = torch.load(str(HYDROVIT_V7_CKPT), map_location=DEVICE, weights_only=True)
            model_v7.load_state_dict(state)
            r2_scores_v7, _ = evaluate_neural(model_v7, test_dl, DEVICE, is_satellite_encoder=True)
            valid_r2s = [v for v in r2_scores_v7.values() if not np.isnan(v)]
            mean_r2_v7 = float(np.mean(valid_r2s)) if valid_r2s else float("nan")
            wt_r2_v7 = r2_scores_v7.get("water_temp", float("nan"))
            results["SENTINEL_HydroViT_v7"] = {
                "water_temp_r2": wt_r2_v7 if not np.isnan(wt_r2_v7) else None,
                "mean_r2": mean_r2_v7 if not np.isnan(mean_r2_v7) else None,
            }
        else:
            logger.warning("HydroViT v7 checkpoint not found — skipping")
            results["SENTINEL_HydroViT_v7"] = {"water_temp_r2": None, "mean_r2": None}

    # ── 2. Extract features for classical ML ────────────────────────────
    logger.info("\nExtracting hand-crafted features for classical ML baselines ...")

    def collect_raw_images_and_targets_batched(dl):
        """Batch collection via DataLoader — much faster than item-by-item."""
        raw_imgs, tgts = [], []
        for batch in dl:
            raw_imgs.append(batch["raw_image"].numpy())
            tgts.append(batch["targets"].numpy())
        return np.concatenate(raw_imgs, axis=0), np.concatenate(tgts, axis=0)

    # Use larger batch for fast collection
    train_dl_collect = DataLoader(train_ds, batch_size=32, shuffle=False, num_workers=4)
    test_dl_collect  = DataLoader(test_ds,  batch_size=32, shuffle=False, num_workers=4)

    X_train_raw, y_train = collect_raw_images_and_targets_batched(train_dl_collect)
    X_test_raw,  y_test  = collect_raw_images_and_targets_batched(test_dl_collect)
    logger.info(f"  Collected: train {X_train_raw.shape}, test {X_test_raw.shape}")

    X_train = extract_features(X_train_raw)
    X_test  = extract_features(X_test_raw)

    # water_temp targets (index 11), filter valid rows
    wt_train_mask = ~np.isnan(y_train[:, WATER_TEMP_IDX])
    wt_test_mask  = ~np.isnan(y_test[:, WATER_TEMP_IDX])

    X_tr_wt = X_train[wt_train_mask]
    y_tr_wt = y_train[wt_train_mask, WATER_TEMP_IDX]
    X_te_wt = X_test[wt_test_mask]
    y_te_wt = y_test[wt_test_mask, WATER_TEMP_IDX]

    logger.info(f"  Train valid water_temp: {wt_train_mask.sum()}, Test: {wt_test_mask.sum()}")

    # ── 3. Ridge Regression ──────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("Baseline 1: Ridge Regression")
    logger.info("=" * 60)

    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    scaler_ridge = StandardScaler()
    X_tr_10 = X_tr_wt[:, :10]  # band means only
    X_te_10 = X_te_wt[:, :10]
    X_tr_10_s = scaler_ridge.fit_transform(X_tr_10)
    X_te_10_s = scaler_ridge.transform(X_te_10)

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_tr_10_s, y_tr_wt)
    y_pred_ridge = ridge.predict(X_te_10_s)
    ridge_r2 = compute_r2(y_pred_ridge, y_te_wt)
    logger.info(f"  Ridge water_temp R2: {ridge_r2:.4f}")
    results["Ridge"] = {"water_temp_r2": ridge_r2}

    # ── 4. Random Forest ─────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("Baseline 2: Random Forest")
    logger.info("=" * 60)

    from sklearn.ensemble import RandomForestRegressor

    rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
    rf.fit(X_tr_wt, y_tr_wt)  # all 25 features
    y_pred_rf = rf.predict(X_te_wt)
    rf_r2 = compute_r2(y_pred_rf, y_te_wt)
    logger.info(f"  Random Forest water_temp R2: {rf_r2:.4f}")
    results["RandomForest"] = {"water_temp_r2": rf_r2}

    # ── 5. ViT-no-pretraining ────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("Baseline 3: ViT-no-pretraining (random init)")
    logger.info("=" * 60)

    vit_scratch = SatelliteEncoder(pretrained=False).to(DEVICE)
    logger.info(f"  Parameters: {sum(p.numel() for p in vit_scratch.parameters()):,}")

    # Phase 1: head only
    for name, p in vit_scratch.named_parameters():
        p.requires_grad = "water_quality_head" in name
    opt1_vit = torch.optim.AdamW(
        [p for p in vit_scratch.parameters() if p.requires_grad],
        lr=3e-4, weight_decay=WEIGHT_DECAY
    )
    sch1_vit = torch.optim.lr_scheduler.CosineAnnealingLR(opt1_vit, T_max=30)
    train_neural(vit_scratch, train_dl, val_dl, opt1_vit, sch1_vit, 30,
                 "ViT-scratch-head", DEVICE, is_satellite_encoder=True)

    # Phase 2: full finetune
    for p in vit_scratch.parameters():
        p.requires_grad = True
    bb_params = [p for n, p in vit_scratch.named_parameters() if "water_quality_head" not in n]
    hd_params  = [p for n, p in vit_scratch.named_parameters() if "water_quality_head" in n]
    opt2_vit = torch.optim.AdamW([
        {"params": bb_params, "lr": 5e-5},
        {"params": hd_params, "lr": 6e-5},
    ], weight_decay=WEIGHT_DECAY)
    sch2_vit = torch.optim.lr_scheduler.CosineAnnealingLR(opt2_vit, T_max=30)
    train_neural(vit_scratch, train_dl, val_dl, opt2_vit, sch2_vit, 30,
                 "ViT-scratch-finetune", DEVICE, is_satellite_encoder=True)

    r2_vit, _ = evaluate_neural(vit_scratch, test_dl, DEVICE, is_satellite_encoder=True)
    valid_r2s_vit = [v for v in r2_vit.values() if not np.isnan(v)]
    mean_r2_vit = float(np.mean(valid_r2s_vit)) if valid_r2s_vit else float("nan")
    wt_r2_vit = r2_vit.get("water_temp", float("nan"))
    logger.info(f"  ViT-no-pretrain water_temp R2: {wt_r2_vit:.4f}")
    logger.info(f"  ViT-no-pretrain mean R2:       {mean_r2_vit:.4f}")
    results["ViT_no_pretrain"] = {
        "water_temp_r2": wt_r2_vit if not np.isnan(wt_r2_vit) else None,
        "mean_r2": mean_r2_vit if not np.isnan(mean_r2_vit) else None,
    }
    del vit_scratch
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    # ── 6. CNN Baseline ──────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("Baseline 4: CNN Baseline")
    logger.info("=" * 60)

    cnn = CNNBaseline(in_channels=10, num_outputs=16).to(DEVICE)
    logger.info(f"  Parameters: {sum(p.numel() for p in cnn.parameters()):,}")

    opt_cnn = torch.optim.AdamW(cnn.parameters(), lr=3e-4, weight_decay=WEIGHT_DECAY)
    sch_cnn = torch.optim.lr_scheduler.CosineAnnealingLR(opt_cnn, T_max=BASELINE_EPOCHS)
    train_neural(cnn, train_dl, val_dl, opt_cnn, sch_cnn, BASELINE_EPOCHS,
                 "CNN", DEVICE, is_satellite_encoder=False)

    r2_cnn, _ = evaluate_neural(cnn, test_dl, DEVICE, is_satellite_encoder=False)
    valid_r2s_cnn = [v for v in r2_cnn.values() if not np.isnan(v)]
    mean_r2_cnn = float(np.mean(valid_r2s_cnn)) if valid_r2s_cnn else float("nan")
    wt_r2_cnn = r2_cnn.get("water_temp", float("nan"))
    logger.info(f"  CNN water_temp R2: {wt_r2_cnn:.4f}")
    logger.info(f"  CNN mean R2:       {mean_r2_cnn:.4f}")
    results["CNN_baseline"] = {
        "water_temp_r2": wt_r2_cnn if not np.isnan(wt_r2_cnn) else None,
        "mean_r2": mean_r2_cnn if not np.isnan(mean_r2_cnn) else None,
    }

    # ── Save results ─────────────────────────────────────────────────────
    out_path = RESULTS_DIR / "hydrovit_benchmark.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved benchmark results: {out_path}")

    elapsed = time.time() - t0
    logger.info(f"Total benchmark time: {elapsed/60:.1f} min")

    # ── Final report ──────────────────────────────────────────────────────
    def fmt(v):
        return f"{v:.3f}" if v is not None else " N/A"
    def fmt_mean(v):
        return f"{v:.3f}" if v is not None else "  — "

    print("\n=== HydroViT Benchmark Results (REAL DATA ONLY) ===")
    print(f"Dataset: {n} pairs (2,861 v3 + 347 low-cloud tiles), seed=42 split")
    print(f"  Train: {n_train}, Val: {n_val}, Test: {n_test}")
    print(f"{'Model':<28} {'water_temp R²':>14}    {'mean R² (16 params)':>20}")
    print("-" * 68)

    v7 = results.get("SENTINEL_HydroViT_v7", {})
    print(f"{'SENTINEL HydroViT v7':<28} {fmt(v7.get('water_temp_r2')):>14}    {fmt_mean(v7.get('mean_r2')):>20}  <- retrained")

    vit = results.get("ViT_no_pretrain", {})
    print(f"{'ViT no pretraining':<28} {fmt(vit.get('water_temp_r2')):>14}    {fmt_mean(vit.get('mean_r2')):>20}")

    cnn = results.get("CNN_baseline", {})
    print(f"{'CNN baseline':<28} {fmt(cnn.get('water_temp_r2')):>14}    {fmt_mean(cnn.get('mean_r2')):>20}")

    rdg = results.get("Ridge", {})
    print(f"{'Ridge Regression':<28} {fmt(rdg.get('water_temp_r2')):>14}    {'—':>20}")

    rf = results.get("RandomForest", {})
    print(f"{'Random Forest':<28} {fmt(rf.get('water_temp_r2')):>14}    {'—':>20}")

    print("=" * 68)
    print("DONE")


if __name__ == "__main__":
    main()
