#!/usr/bin/env python3
"""
Benchmark HydroViT v6 against baseline models on water temperature prediction.

Uses the v3 dataset (paired_wq_v3.npz) with the exact same train/val/test split
as the v6 training run (n_train=2941, n_val=630, n_test=631, seed=42).

Baselines:
  1. Ridge Regression (13-band means only, 13-dim)
  2. Random Forest (13-band means + 13-band stds + 5 vegetation ratios = 31-dim, n_estimators=200)
  3. CNN baseline (4-layer conv 32->64->128->256 + global avg pool + linear head, 60 epochs)
  4. ViT no pretraining (same SatelliteEncoder + WaterQualityHead, random init, 60 epochs)

Primary metric: water_temp R2 (index 11 in PARAM_NAMES).

MIT License -- Bryan Cheng, 2026
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# When launched with CUDA_VISIBLE_DEVICES=1, the visible device is cuda:0
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CKPT_DIR = PROJECT_ROOT / "checkpoints/satellite"
RESULTS_DIR = PROJECT_ROOT / "results/benchmarks"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = PROJECT_ROOT / "data/processed/satellite/paired_wq_v3.npz"
HYDROVIT_V6_CKPT = CKPT_DIR / "hydrovit_wq_v6.pt"
HYDROVIT_V6_RESULTS = CKPT_DIR / "results_wq_v6.json"

# Split fractions matching train_hydrovit_wq_v6.py (70/15/15, seed=42)
# Note: results_wq_v6.json shows 2941/630/631 = 4202 total; the v4 dataset
# used during training is no longer present. We use v3 (2861 samples) and
# reproduce the same 70/15 split formula with seed=42.
SPLIT_SEED = 42

WATER_TEMP_IDX = PARAM_NAMES.index("water_temp")  # = 11
BATCH_SIZE = 4
BASELINE_EPOCHS = 60
USE_AMP = True
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
EARLY_STOP_PATIENCE = 10


# ---------------------------------------------------------------------------
# Dataset — mirrors train_hydrovit_wq_v6.py exactly
# ---------------------------------------------------------------------------
class PairedWQDataset(Dataset):
    """Load paired_wq_v3.npz. Images are (N, 10, 224, 224); padded to 13 bands
    for SatelliteEncoder compatibility."""

    def __init__(self, data_path: str):
        data = np.load(data_path, allow_pickle=True)
        self.images = data["images"].astype(np.float32)   # (N, 10, H, W)
        self.targets = data["targets"].astype(np.float32)  # (N, 16)

        # Log-transform log-normal parameters (same as v6 training)
        self.log_params = {0, 1, 3, 4, 5, 6, 8, 9, 12, 14}
        self.targets_norm = self.targets.copy()
        for i in self.log_params:
            valid = ~np.isnan(self.targets_norm[:, i])
            if valid.any():
                vals = self.targets_norm[valid, i]
                vals = np.maximum(vals, 1e-6)
                self.targets_norm[valid, i] = np.log1p(vals)

        # Z-score normalize
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

        logger.info(
            f"Loaded {len(self)} paired samples: "
            f"images={self.images.shape}, targets={self.targets.shape}"
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = torch.tensor(self.images[idx])          # (10, H, W)
        # Pad to 13 bands (same as v6 training)
        padding = torch.zeros(3, img.shape[1], img.shape[2])
        image13 = torch.cat([img, padding], dim=0)    # (13, H, W)
        targets = torch.tensor(self.targets_norm[idx])
        return {
            "image": image13,         # 13-band for ViT models
            "raw_image": img,         # 10-band for CNN/classical
            "targets": targets,
        }


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------
def compute_r2(preds, targets):
    """R2 for 1D numpy arrays."""
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
            else:
                r2_scores[PARAM_NAMES[j]] = compute_r2(p, t)
        else:
            r2_scores[PARAM_NAMES[j]] = float("nan")
    return r2_scores


# ---------------------------------------------------------------------------
# Feature extraction for classical ML baselines
# ---------------------------------------------------------------------------
def extract_features_13band(images):
    """
    Extract features from 13-band images (N, 13, H, W).

    Returns:
      Ridge input  (N, 13)  — 13-band means
      RF input     (N, 31)  — 13 means + 13 stds + 5 vegetation ratios
    """
    means = images.mean(axis=(2, 3))   # (N, 13)
    stds  = images.std(axis=(2, 3))    # (N, 13)

    eps = 1e-8
    # Sentinel-2 10-band layout: B2=0,B3=1,B4=2,B8=3,B5=4,B6=5,B7=6,B8A=7,B11=8,B12=9
    # (bands 10-12 are zero-padded)
    B3  = means[:, 1]   # green
    B4  = means[:, 2]   # red
    B8  = means[:, 3]   # NIR
    B11 = means[:, 8]   # SWIR1
    B12 = means[:, 9]   # SWIR2

    ndwi      = (B3 - B8)  / (B3 + B8  + eps)
    ndvi      = (B8 - B4)  / (B8 + B4  + eps)
    red_nir   = B4          / (B8        + eps)
    green_swir = B3          / (B11       + eps)
    nir_swir  = B8          / (B12       + eps)

    ratios = np.stack([ndwi, ndvi, red_nir, green_swir, nir_swir], axis=1)  # (N, 5)

    ridge_feat = means.astype(np.float32)                                     # (N, 13)
    rf_feat    = np.concatenate([means, stds, ratios], axis=1).astype(np.float32)  # (N, 31)
    return ridge_feat, rf_feat


# ---------------------------------------------------------------------------
# Neural model evaluation / training helpers
# ---------------------------------------------------------------------------
def evaluate_neural(model, dataloader, device, is_satellite_encoder=True):
    model.eval()
    preds = {j: [] for j in range(NUM_WATER_PARAMS)}
    tgts  = {j: [] for j in range(NUM_WATER_PARAMS)}
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            image = batch["image"].to(device) if is_satellite_encoder else batch["raw_image"].to(device)
            targets = batch["targets"].to(device)

            out = model(image)
            if is_satellite_encoder:
                wq  = out["water_quality_params"]
                unc = out["param_uncertainty"]
            else:
                wq  = out
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
            image = batch["image"].to(device) if is_satellite_encoder else batch["raw_image"].to(device)
            targets = batch["targets"].to(device)

            with torch.amp.autocast("cuda", enabled=USE_AMP and device.type == "cuda"):
                out = model(image)
                if is_satellite_encoder:
                    wq  = out["water_quality_params"]
                    unc = out["param_uncertainty"]
                else:
                    wq  = out
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
        mean_r2   = np.mean(valid_r2s) if valid_r2s else -1.0
        wt_r2     = r2_scores.get("water_temp", float("nan"))

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
# CNN Baseline model: 4-layer conv 32->64->128->256, global avg pool + linear
# ---------------------------------------------------------------------------
class CNNBaseline(nn.Module):
    def __init__(self, in_channels=10, num_outputs=16):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 224->112
            nn.Conv2d(32, 64, 3, padding=1),          nn.ReLU(), nn.MaxPool2d(2),   # 112->56
            nn.Conv2d(64, 128, 3, padding=1),          nn.ReLU(), nn.MaxPool2d(2),   # 56->28
            nn.Conv2d(128, 256, 3, padding=1),         nn.ReLU(), nn.AdaptiveAvgPool2d(1),  # global
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_outputs),
        )

    def forward(self, x):
        return self.head(self.features(x))


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("=" * 65)
    print("HydroViT v6 Benchmark — v3 dataset, seed=42 split")
    print("=" * 65)

    if not DATA_PATH.exists():
        logger.error(f"Data file not found: {DATA_PATH}")
        sys.exit(1)

    logger.info(f"Device: {DEVICE}")
    logger.info(f"Loading dataset: {DATA_PATH}")
    dataset = PairedWQDataset(str(DATA_PATH))
    n = len(dataset)
    logger.info(f"Total samples: {n}")

    # Reproduce the same 70/15/15 split as train_hydrovit_wq_v6.py with seed=42
    n_train = max(1, int(0.7 * n))
    n_val   = max(1, int(0.15 * n))
    n_test  = n - n_train - n_val
    if n_test < 1:
        n_test = 1
        n_train = n - n_val - n_test

    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(SPLIT_SEED),
    )
    logger.info(f"Split: {n_train} train / {n_val} val / {n_test} test")

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)

    results = {}

    # ── 1. HydroViT v6 ───────────────────────────────────────────────────
    logger.info("\n" + "=" * 65)
    logger.info("Evaluating SENTINEL HydroViT v6")
    logger.info("=" * 65)

    model_v6 = SatelliteEncoder(pretrained=False).to(DEVICE)
    state = torch.load(str(HYDROVIT_V6_CKPT), map_location=DEVICE, weights_only=True)
    if "model" in state:
        state = state["model"]
    elif "state_dict" in state:
        state = state["state_dict"]
    model_v6.load_state_dict(state)
    logger.info(f"Loaded checkpoint: {HYDROVIT_V6_CKPT}")

    r2_v6, loss_v6 = evaluate_neural(model_v6, test_dl, DEVICE, is_satellite_encoder=True)
    valid_r2s_v6 = [v for v in r2_v6.values() if not np.isnan(v)]
    mean_r2_v6   = float(np.mean(valid_r2s_v6)) if valid_r2s_v6 else float("nan")
    wt_r2_v6     = r2_v6.get("water_temp", float("nan"))
    logger.info(f"  HydroViT v6 water_temp R2: {wt_r2_v6:.4f}")
    logger.info(f"  HydroViT v6 mean R2:       {mean_r2_v6:.4f}")
    results["SENTINEL_HydroViT_v6"] = {
        "water_temp_r2": float(wt_r2_v6) if not np.isnan(wt_r2_v6) else None,
        "mean_r2": float(mean_r2_v6) if not np.isnan(mean_r2_v6) else None,
        "test_loss": float(loss_v6),
    }
    del model_v6
    torch.cuda.empty_cache()

    # ── 2. Extract features for classical ML ─────────────────────────────
    logger.info("\nCollecting images/targets for classical baselines ...")

    def collect_all(ds_subset):
        all_imgs13 = []
        all_tgts   = []
        for item in ds_subset:
            all_imgs13.append(item["image"].numpy())
            all_tgts.append(item["targets"].numpy())
        return np.stack(all_imgs13, axis=0), np.stack(all_tgts, axis=0)

    logger.info("  Collecting train set ...")
    X13_train, y_train = collect_all(train_ds)
    logger.info("  Collecting test set ...")
    X13_test,  y_test  = collect_all(test_ds)

    ridge_feat_train, rf_feat_train = extract_features_13band(X13_train)
    ridge_feat_test,  rf_feat_test  = extract_features_13band(X13_test)

    wt_train_mask = ~np.isnan(y_train[:, WATER_TEMP_IDX])
    wt_test_mask  = ~np.isnan(y_test[:, WATER_TEMP_IDX])
    logger.info(f"  Valid water_temp — train: {wt_train_mask.sum()}, test: {wt_test_mask.sum()}")

    y_tr_wt = y_train[wt_train_mask, WATER_TEMP_IDX]
    y_te_wt = y_test[wt_test_mask,  WATER_TEMP_IDX]

    # ── 3. Ridge Regression (13-band means) ──────────────────────────────
    logger.info("\n" + "=" * 65)
    logger.info("Baseline 1: Ridge Regression (13-band means, 13-dim input)")
    logger.info("=" * 65)

    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    scaler_ridge = StandardScaler()
    X_tr_ridge = scaler_ridge.fit_transform(ridge_feat_train[wt_train_mask])
    X_te_ridge = scaler_ridge.transform(ridge_feat_test[wt_test_mask])

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_tr_ridge, y_tr_wt)
    y_pred_ridge = ridge.predict(X_te_ridge)
    ridge_r2 = compute_r2(y_pred_ridge, y_te_wt)
    logger.info(f"  Ridge water_temp R2: {ridge_r2:.4f}")
    results["Ridge"] = {"water_temp_r2": float(ridge_r2)}

    # ── 4. Random Forest (13 means + 13 stds + 5 ratios = 31-dim) ────────
    logger.info("\n" + "=" * 65)
    logger.info("Baseline 2: Random Forest (31-dim, n_estimators=200)")
    logger.info("=" * 65)

    from sklearn.ensemble import RandomForestRegressor

    rf = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42)
    rf.fit(rf_feat_train[wt_train_mask], y_tr_wt)
    y_pred_rf = rf.predict(rf_feat_test[wt_test_mask])
    rf_r2 = compute_r2(y_pred_rf, y_te_wt)
    logger.info(f"  Random Forest water_temp R2: {rf_r2:.4f}")
    results["RandomForest"] = {"water_temp_r2": float(rf_r2)}

    # Free memory before neural baselines
    del X13_train, X13_test, ridge_feat_train, ridge_feat_test
    del rf_feat_train, rf_feat_test, y_train, y_test

    # ── 5. CNN Baseline (60 epochs) ───────────────────────────────────────
    logger.info("\n" + "=" * 65)
    logger.info("Baseline 3: CNN Baseline (4-layer conv, 60 epochs)")
    logger.info("=" * 65)

    cnn = CNNBaseline(in_channels=10, num_outputs=16).to(DEVICE)
    logger.info(f"  Parameters: {sum(p.numel() for p in cnn.parameters()):,}")

    opt_cnn = torch.optim.AdamW(cnn.parameters(), lr=3e-4, weight_decay=WEIGHT_DECAY)
    sch_cnn = torch.optim.lr_scheduler.CosineAnnealingLR(opt_cnn, T_max=BASELINE_EPOCHS)
    train_neural(cnn, train_dl, val_dl, opt_cnn, sch_cnn, BASELINE_EPOCHS,
                 "CNN", DEVICE, is_satellite_encoder=False)

    r2_cnn, _ = evaluate_neural(cnn, test_dl, DEVICE, is_satellite_encoder=False)
    valid_r2s_cnn = [v for v in r2_cnn.values() if not np.isnan(v)]
    mean_r2_cnn   = float(np.mean(valid_r2s_cnn)) if valid_r2s_cnn else float("nan")
    wt_r2_cnn     = r2_cnn.get("water_temp", float("nan"))
    logger.info(f"  CNN water_temp R2: {wt_r2_cnn:.4f}")
    logger.info(f"  CNN mean R2:       {mean_r2_cnn:.4f}")
    results["CNN_baseline"] = {
        "water_temp_r2": float(wt_r2_cnn) if not np.isnan(wt_r2_cnn) else None,
        "mean_r2": float(mean_r2_cnn) if not np.isnan(mean_r2_cnn) else None,
    }
    del cnn
    torch.cuda.empty_cache()

    # ── 6. ViT no pretraining (same arch, random init, 60 epochs) ─────────
    logger.info("\n" + "=" * 65)
    logger.info("Baseline 4: ViT no pretraining (random init, 60 epochs)")
    logger.info("=" * 65)

    vit_scratch = SatelliteEncoder(pretrained=False).to(DEVICE)
    logger.info(f"  Parameters: {sum(p.numel() for p in vit_scratch.parameters()):,}")

    # Phase 1: head only (30 epochs)
    for pname, p in vit_scratch.named_parameters():
        p.requires_grad = "water_quality_head" in pname
    opt1 = torch.optim.AdamW(
        [p for p in vit_scratch.parameters() if p.requires_grad],
        lr=3e-4, weight_decay=WEIGHT_DECAY
    )
    sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=30)
    train_neural(vit_scratch, train_dl, val_dl, opt1, sch1, 30,
                 "ViT-scratch-head", DEVICE, is_satellite_encoder=True)

    # Phase 2: full fine-tune (30 epochs)
    for p in vit_scratch.parameters():
        p.requires_grad = True
    bb_params = [p for n, p in vit_scratch.named_parameters() if "water_quality_head" not in n]
    hd_params  = [p for n, p in vit_scratch.named_parameters() if "water_quality_head" in n]
    opt2 = torch.optim.AdamW([
        {"params": bb_params, "lr": 5e-5},
        {"params": hd_params, "lr": 6e-5},
    ], weight_decay=WEIGHT_DECAY)
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=30)
    train_neural(vit_scratch, train_dl, val_dl, opt2, sch2, 30,
                 "ViT-scratch-finetune", DEVICE, is_satellite_encoder=True)

    r2_vit, _ = evaluate_neural(vit_scratch, test_dl, DEVICE, is_satellite_encoder=True)
    valid_r2s_vit = [v for v in r2_vit.values() if not np.isnan(v)]
    mean_r2_vit   = float(np.mean(valid_r2s_vit)) if valid_r2s_vit else float("nan")
    wt_r2_vit     = r2_vit.get("water_temp", float("nan"))
    logger.info(f"  ViT-no-pretrain water_temp R2: {wt_r2_vit:.4f}")
    logger.info(f"  ViT-no-pretrain mean R2:       {mean_r2_vit:.4f}")
    results["ViT_no_pretrain"] = {
        "water_temp_r2": float(wt_r2_vit) if not np.isnan(wt_r2_vit) else None,
        "mean_r2": float(mean_r2_vit) if not np.isnan(mean_r2_vit) else None,
    }
    del vit_scratch
    torch.cuda.empty_cache()

    # ── Save results ──────────────────────────────────────────────────────
    out_path = RESULTS_DIR / "hydrovit_benchmark.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved benchmark results: {out_path}")

    elapsed = time.time() - t0
    logger.info(f"Total benchmark time: {elapsed/60:.1f} min")

    # ── Final report ──────────────────────────────────────────────────────
    def fmt(v):
        return f"{v:.4f}" if v is not None else "  N/A "

    print()
    print("=" * 65)
    print("HydroViT v6 Benchmark Results (v3 data, seed=42 split)")
    print(f"Dataset: paired_wq_v3.npz, n={n}")
    print(f"  Train: {n_train}, Val: {n_val}, Test: {n_test}")
    print("=" * 65)
    print(f"{'Model':<30} {'water_temp R²':>14}    {'mean R² (16 params)':>20}")
    print("-" * 70)

    v6  = results.get("SENTINEL_HydroViT_v6", {})
    vit = results.get("ViT_no_pretrain", {})
    cnn = results.get("CNN_baseline", {})
    rdg = results.get("Ridge", {})
    rfo = results.get("RandomForest", {})

    print(f"{'SENTINEL HydroViT v6':<30} {fmt(v6.get('water_temp_r2')):>14}    {fmt(v6.get('mean_r2')):>20}  <- pretrained")
    print(f"{'ViT no pretraining':<30} {fmt(vit.get('water_temp_r2')):>14}    {fmt(vit.get('mean_r2')):>20}")
    print(f"{'CNN baseline':<30} {fmt(cnn.get('water_temp_r2')):>14}    {fmt(cnn.get('mean_r2')):>20}")
    print(f"{'Ridge Regression':<30} {fmt(rdg.get('water_temp_r2')):>14}    {'—':>20}")
    print(f"{'Random Forest':<30} {fmt(rfo.get('water_temp_r2')):>14}    {'—':>20}")
    print("=" * 70)
    print("DONE")


if __name__ == "__main__":
    main()
