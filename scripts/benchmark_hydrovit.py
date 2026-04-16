#!/usr/bin/env python3
"""Benchmark HydroViT v8 vs v6, CNN, ViT-scratch, Ridge, RF.

All models evaluated on the SAME v5 test split (seed=42, 70/15/15).
Primary metric: water_temp R2 (index 11).
Secondary: mean R2 across all 16 parameters.

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

# Import HydroViTV8 from v8 training script
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from train_hydrovit_wq_v8 import (
    HydroViTV8,
    SpectralBandAttention,
    DeepWQHead,
    weighted_mse,
    PARAM_WEIGHTS,
    WATER_TEMP_IDX,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CKPT_DIR = PROJECT_ROOT / "checkpoints/satellite"
RESULTS_DIR = PROJECT_ROOT / "results/benchmarks"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = PROJECT_ROOT / "data/processed/satellite/paired_wq_v5.npz"
V8_CKPT = CKPT_DIR / "hydrovit_wq_v8.pt"
V6_CKPT = CKPT_DIR / "hydrovit_wq_v6.pt"

SPLIT_SEED = 42
BATCH_SIZE = 8
USE_AMP = True
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
BASELINE_EPOCHS = 80
EARLY_STOP_PATIENCE = 15

LOG_PARAMS = {0, 1, 3, 4, 5, 6, 8, 9, 12, 14}


# ---------------------------------------------------------------------------
# Unified Dataset — returns both 10-band and 13-band images
# ---------------------------------------------------------------------------
class PairedWQDatasetWith13(Dataset):
    """Returns image (10-band) and image13 (13-band padded) for all models."""

    def __init__(self, data_path: str):
        data = np.load(data_path, allow_pickle=True)
        self.images = data["images"].astype(np.float32)
        self.targets = data["targets"].astype(np.float32)

        targets_norm = self.targets.copy()
        for i in LOG_PARAMS:
            valid = ~np.isnan(targets_norm[:, i])
            if valid.any():
                vals = np.maximum(targets_norm[valid, i], 1e-6)
                targets_norm[valid, i] = np.log1p(vals)

        self.target_mean = np.nanmean(targets_norm, axis=0)
        self.target_std = np.nanstd(targets_norm, axis=0)
        self.target_std[self.target_std < 1e-6] = 1.0
        nan_cols = np.all(np.isnan(targets_norm), axis=0)
        self.target_mean[nan_cols] = 0.0
        self.target_std[nan_cols] = 1.0

        for i in range(NUM_WATER_PARAMS):
            valid = ~np.isnan(targets_norm[:, i])
            if valid.any():
                targets_norm[valid, i] = (
                    (targets_norm[valid, i] - self.target_mean[i])
                    / self.target_std[i]
                )

        self.targets_norm = targets_norm
        logger.info(f"Loaded {len(self)} samples: images={self.images.shape}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = torch.tensor(self.images[idx])          # (10, H, W)
        pad = torch.zeros(3, img.shape[1], img.shape[2])
        img13 = torch.cat([img, pad], dim=0)           # (13, H, W)
        targets = torch.tensor(self.targets_norm[idx])
        return {
            "image":   img,     # 10-band for HydroViT v8 / CNN
            "image13": img13,   # 13-band for v6 / ViT-scratch
            "targets": targets,
        }


# ---------------------------------------------------------------------------
# R² helper
# ---------------------------------------------------------------------------
def _compute_r2(preds_dict, tgts_dict):
    r2 = {}
    for j in range(NUM_WATER_PARAMS):
        if preds_dict[j]:
            p = torch.cat(preds_dict[j]).numpy()
            t = torch.cat(tgts_dict[j]).numpy()
            if len(p) < 2:
                r2[PARAM_NAMES[j]] = float("nan")
            else:
                ss_res = ((p - t) ** 2).sum()
                ss_tot = ((t - t.mean()) ** 2).sum()
                r2[PARAM_NAMES[j]] = float(1 - ss_res / ss_tot) if ss_tot > 1e-8 else 0.0
        else:
            r2[PARAM_NAMES[j]] = float("nan")
    return r2


# ---------------------------------------------------------------------------
# HydroViT v8 evaluation  (model returns [B, 16] directly)
# ---------------------------------------------------------------------------
def evaluate_v8(model, dataloader, device):
    model.eval()
    preds = {j: [] for j in range(NUM_WATER_PARAMS)}
    tgts  = {j: [] for j in range(NUM_WATER_PARAMS)}
    with torch.no_grad():
        for batch in dataloader:
            image   = batch["image"].to(device)
            targets = batch["targets"].to(device)
            pred    = model(image)                   # [B, 16]
            valid   = ~torch.isnan(targets)
            for j in range(NUM_WATER_PARAMS):
                mask = valid[:, j]
                if mask.sum() > 0:
                    preds[j].append(pred[:, j][mask].cpu())
                    tgts[j].append(targets[:, j][mask].cpu())
    return _compute_r2(preds, tgts)


# ---------------------------------------------------------------------------
# HydroViT v6 evaluation  (model returns dict with "water_quality_params")
# ---------------------------------------------------------------------------
def evaluate_v6_model(model, dataloader, device):
    model.eval()
    preds = {j: [] for j in range(NUM_WATER_PARAMS)}
    tgts  = {j: [] for j in range(NUM_WATER_PARAMS)}
    with torch.no_grad():
        for batch in dataloader:
            image   = batch["image13"].to(device)
            targets = batch["targets"].to(device)
            out     = model(image)
            wq      = out["water_quality_params"]
            valid   = ~torch.isnan(targets)
            for j in range(NUM_WATER_PARAMS):
                mask = valid[:, j]
                if mask.sum() > 0:
                    preds[j].append(wq[:, j][mask].cpu())
                    tgts[j].append(targets[:, j][mask].cpu())
    return _compute_r2(preds, tgts)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
def train_v6_style(model, train_dl, val_dl, device, epochs, name):
    """Two-phase training for SatelliteEncoder (ViT-scratch)."""
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP and device.type == "cuda")
    best_val_r2, best_state = -float("inf"), None
    no_improve = 0

    def _epoch(opt):
        nonlocal no_improve, best_val_r2, best_state
        model.train()
        for batch in train_dl:
            img = batch["image13"].to(device)
            tgt = batch["targets"].to(device)
            with torch.amp.autocast("cuda", enabled=USE_AMP and device.type == "cuda"):
                out  = model(img)
                wq   = out["water_quality_params"]
                unc  = out["param_uncertainty"]
                loss = WaterQualityHead.gaussian_nll_loss(wq, unc, tgt)
                if torch.isnan(loss):
                    opt.zero_grad(); continue
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(opt); scaler.update()

    half = epochs // 2
    # Phase 1
    for p in model.parameters():
        p.requires_grad = False
    for n, p in model.named_parameters():
        if "water_quality_head" in n:
            p.requires_grad = True
    opt1 = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                              lr=3e-4, weight_decay=WEIGHT_DECAY)
    sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=half)
    for ep in range(half):
        _epoch(opt1); sch1.step()
        r2s = evaluate_v6_model(model, val_dl, device)
        mr2 = float(np.mean([v for v in r2s.values() if not np.isnan(v)] or [-1]))
        if mr2 > best_val_r2:
            best_val_r2 = mr2
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                logger.info(f"[{name}] Phase1 early stop ep {ep+1}")
                break
        if (ep + 1) % 10 == 0:
            logger.info(f"[{name}] P1 Ep {ep+1}/{half} | meanR2={mr2:.4f} wt={r2s.get('water_temp', float('nan')):.4f}")

    # Phase 2
    for p in model.parameters():
        p.requires_grad = True
    bb = [p for n, p in model.named_parameters() if "water_quality_head" not in n]
    hd = [p for n, p in model.named_parameters() if "water_quality_head" in n]
    opt2 = torch.optim.AdamW([{"params": bb, "lr": 5e-5}, {"params": hd, "lr": 6e-5}],
                              weight_decay=WEIGHT_DECAY)
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=half)
    no_improve = 0
    for ep in range(half):
        _epoch(opt2); sch2.step()
        r2s = evaluate_v6_model(model, val_dl, device)
        mr2 = float(np.mean([v for v in r2s.values() if not np.isnan(v)] or [-1]))
        if mr2 > best_val_r2:
            best_val_r2 = mr2
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                logger.info(f"[{name}] Phase2 early stop ep {ep+1}")
                break
        if (ep + 1) % 10 == 0:
            logger.info(f"[{name}] P2 Ep {ep+1}/{half} | meanR2={mr2:.4f} wt={r2s.get('water_temp', float('nan')):.4f}")

    if best_state:
        model.load_state_dict(best_state)
    return best_val_r2


# ---------------------------------------------------------------------------
# CNN Baseline
# ---------------------------------------------------------------------------
class CNNBaseline(nn.Module):
    def __init__(self, in_channels=10, num_outputs=16):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),           nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),           nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),          nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_outputs),
        )

    def forward(self, x):
        return self.head(self.features(x))


def evaluate_cnn(model, dataloader, device):
    model.eval()
    preds = {j: [] for j in range(NUM_WATER_PARAMS)}
    tgts  = {j: [] for j in range(NUM_WATER_PARAMS)}
    with torch.no_grad():
        for batch in dataloader:
            image   = batch["image"].to(device)
            targets = batch["targets"].to(device)
            pred    = model(image)
            valid   = ~torch.isnan(targets)
            for j in range(NUM_WATER_PARAMS):
                mask = valid[:, j]
                if mask.sum() > 0:
                    preds[j].append(pred[:, j][mask].cpu())
                    tgts[j].append(targets[:, j][mask].cpu())
    return _compute_r2(preds, tgts)


def train_cnn(model, train_dl, val_dl, device, epochs, name):
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP and device.type == "cuda")
    best_val_r2, best_state, no_improve = -float("inf"), None, 0
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    w = PARAM_WEIGHTS.to(device)

    for ep in range(epochs):
        model.train()
        for batch in train_dl:
            img = batch["image"].to(device)
            tgt = batch["targets"].to(device)
            with torch.amp.autocast("cuda", enabled=USE_AMP and device.type == "cuda"):
                pred = model(img)
                loss = weighted_mse(pred, tgt, w)
                if torch.isnan(loss):
                    opt.zero_grad(); continue
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(opt); scaler.update()
        sch.step()

        r2s = evaluate_cnn(model, val_dl, device)
        mr2 = float(np.mean([v for v in r2s.values() if not np.isnan(v)] or [-1]))
        if mr2 > best_val_r2:
            best_val_r2 = mr2
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                logger.info(f"[{name}] Early stop ep {ep+1}")
                break
        if (ep + 1) % 10 == 0:
            logger.info(f"[{name}] Ep {ep+1}/{epochs} | meanR2={mr2:.4f} wt={r2s.get('water_temp', float('nan')):.4f}")

    if best_state:
        model.load_state_dict(best_state)
    return best_val_r2


# ---------------------------------------------------------------------------
# Classical ML feature extraction
# ---------------------------------------------------------------------------
def extract_features(images_10band):
    means = images_10band.mean(axis=(2, 3))
    stds  = images_10band.std(axis=(2, 3))
    eps   = 1e-8
    B3 = means[:, 1]; B4 = means[:, 2]; B8 = means[:, 3]
    B11 = means[:, 8]; B12 = means[:, 9]
    ndwi      = (B3 - B8)  / (B3 + B8  + eps)
    ndvi      = (B8 - B4)  / (B8 + B4  + eps)
    red_nir   = B4          / (B8        + eps)
    green_swir = B3          / (B11       + eps)
    nir_swir  = B8          / (B12       + eps)
    ratios = np.stack([ndwi, ndvi, red_nir, green_swir, nir_swir], axis=1)
    return np.concatenate([means, stds, ratios], axis=1).astype(np.float32)


def compute_r2_1d(preds, targets):
    ss_res = ((preds - targets) ** 2).sum()
    ss_tot = ((targets - targets.mean()) ** 2).sum()
    return float(1.0 - ss_res / ss_tot) if ss_tot > 1e-8 else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("HydroViT v8 Benchmark — v5 dataset, seed=42 split")
    logger.info("=" * 70)
    logger.info(f"Device: {DEVICE}")

    if not DATA_PATH.exists():
        logger.error(f"Dataset not found: {DATA_PATH}")
        sys.exit(1)

    # Load data — use IDENTICAL split to training script (np.random.default_rng(42))
    # CRITICAL: training script uses np.random.default_rng(SEED=42).permutation(N)
    #           with test=idx[:n_test], val=idx[n_test:n_test+n_val], train=idx[n_test+n_val:]
    #           Any other split method causes data leakage between train and benchmark test.
    dataset = PairedWQDatasetWith13(str(DATA_PATH))
    n = len(dataset)
    rng = np.random.default_rng(SPLIT_SEED)
    idx = rng.permutation(n)
    n_test  = int(0.15 * n)
    n_val   = int(0.15 * n)
    n_train = n - n_test - n_val
    test_idx  = idx[:n_test]
    val_idx   = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]

    from torch.utils.data import Subset
    train_ds = Subset(dataset, train_idx.tolist())
    val_ds   = Subset(dataset, val_idx.tolist())
    test_ds  = Subset(dataset, test_idx.tolist())
    logger.info(f"Split (numpy rng seed={SPLIT_SEED}): {n_train} train / {n_val} val / {n_test} test")
    logger.info("  (Matches training script split exactly — no data leakage)")

    nw = min(4, torch.get_num_threads())
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=nw, pin_memory=True, drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, num_workers=nw, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, num_workers=nw, pin_memory=True)

    results = {}

    # ── 1. HydroViT v8 ──────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("Evaluating HydroViT v8")
    logger.info("=" * 70)

    if V8_CKPT.exists():
        model_v8 = HydroViTV8(pretrained_ckpt=None, in_bands=10).to(DEVICE)
        state = torch.load(str(V8_CKPT), map_location=DEVICE, weights_only=True)
        model_v8.load_state_dict(state)
        logger.info(f"Loaded checkpoint: {V8_CKPT}")
        r2_v8 = evaluate_v8(model_v8, test_dl, DEVICE)
        valid_r2s = [v for v in r2_v8.values() if not np.isnan(v)]
        mean_r2_v8 = float(np.mean(valid_r2s)) if valid_r2s else float("nan")
        wt_r2_v8 = r2_v8.get("water_temp", float("nan"))
        logger.info(f"  v8 water_temp R2: {wt_r2_v8:.4f}")
        logger.info(f"  v8 mean R2:       {mean_r2_v8:.4f}")
        results["HydroViT_v8"] = {
            "water_temp_r2": float(wt_r2_v8),
            "mean_r2": float(mean_r2_v8),
            "per_param_r2": {k: (float(v) if not np.isnan(v) else None) for k, v in r2_v8.items()},
        }
        del model_v8
        torch.cuda.empty_cache()
    else:
        logger.warning(f"v8 checkpoint not found: {V8_CKPT} — skipping")
        results["HydroViT_v8"] = {"note": "checkpoint not found"}

    # ── 2. HydroViT v6 ──────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("Evaluating HydroViT v6 (previous best)")
    logger.info("=" * 70)

    if V6_CKPT.exists():
        model_v6 = SatelliteEncoder(pretrained=False).to(DEVICE)
        state = torch.load(str(V6_CKPT), map_location=DEVICE, weights_only=True)
        if "model" in state:
            state = state["model"]
        elif "state_dict" in state:
            state = state["state_dict"]
        model_v6.load_state_dict(state, strict=False)
        logger.info(f"Loaded checkpoint: {V6_CKPT}")
        r2_v6 = evaluate_v6_model(model_v6, test_dl, DEVICE)
        valid_r2s = [v for v in r2_v6.values() if not np.isnan(v)]
        mean_r2_v6 = float(np.mean(valid_r2s)) if valid_r2s else float("nan")
        wt_r2_v6 = r2_v6.get("water_temp", float("nan"))
        logger.info(f"  v6 water_temp R2: {wt_r2_v6:.4f}")
        logger.info(f"  v6 mean R2:       {mean_r2_v6:.4f}")
        results["HydroViT_v6"] = {
            "water_temp_r2": float(wt_r2_v6),
            "mean_r2": float(mean_r2_v6),
            "note": "evaluated on v5 test split (same seed=42)",
        }
        del model_v6
        torch.cuda.empty_cache()
    else:
        logger.warning(f"v6 checkpoint not found: {V6_CKPT}")
        results["HydroViT_v6"] = {"note": "checkpoint not found"}

    # ── 3. CNN Baseline ──────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("Baseline: CNN (4-layer, from scratch)")
    logger.info("=" * 70)

    cnn = CNNBaseline(in_channels=10, num_outputs=16).to(DEVICE)
    logger.info(f"  CNN params: {sum(p.numel() for p in cnn.parameters()):,}")
    train_cnn(cnn, train_dl, val_dl, DEVICE, BASELINE_EPOCHS, "CNN")
    r2_cnn = evaluate_cnn(cnn, test_dl, DEVICE)
    valid_r2s = [v for v in r2_cnn.values() if not np.isnan(v)]
    mean_r2_cnn = float(np.mean(valid_r2s)) if valid_r2s else float("nan")
    wt_r2_cnn = r2_cnn.get("water_temp", float("nan"))
    logger.info(f"  CNN water_temp R2: {wt_r2_cnn:.4f}")
    logger.info(f"  CNN mean R2:       {mean_r2_cnn:.4f}")
    results["CNN_baseline"] = {
        "water_temp_r2": float(wt_r2_cnn),
        "mean_r2": float(mean_r2_cnn),
        "per_param_r2": {k: (float(v) if not np.isnan(v) else None) for k, v in r2_cnn.items()},
    }
    del cnn
    torch.cuda.empty_cache()

    # ── 4. ViT no pretraining ────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("Baseline: ViT no pretraining (random init)")
    logger.info("=" * 70)

    vit_scratch = SatelliteEncoder(pretrained=False).to(DEVICE)
    logger.info(f"  ViT-scratch params: {sum(p.numel() for p in vit_scratch.parameters()):,}")
    train_v6_style(vit_scratch, train_dl, val_dl, DEVICE, BASELINE_EPOCHS, "ViT-scratch")
    r2_vit = evaluate_v6_model(vit_scratch, test_dl, DEVICE)
    valid_r2s = [v for v in r2_vit.values() if not np.isnan(v)]
    mean_r2_vit = float(np.mean(valid_r2s)) if valid_r2s else float("nan")
    wt_r2_vit = r2_vit.get("water_temp", float("nan"))
    logger.info(f"  ViT-scratch water_temp R2: {wt_r2_vit:.4f}")
    results["ViT_no_pretrain"] = {
        "water_temp_r2": float(wt_r2_vit),
        "mean_r2": float(mean_r2_vit),
    }
    del vit_scratch
    torch.cuda.empty_cache()

    # ── 5. Ridge & RF ────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("Classical baselines: Ridge + Random Forest")
    logger.info("=" * 70)

    all_train_imgs, all_train_tgts = [], []
    all_test_imgs,  all_test_tgts  = [], []
    for item in train_ds:
        all_train_imgs.append(item["image"].numpy())
        all_train_tgts.append(item["targets"].numpy())
    for item in test_ds:
        all_test_imgs.append(item["image"].numpy())
        all_test_tgts.append(item["targets"].numpy())

    X_train = np.stack(all_train_imgs)
    y_train = np.stack(all_train_tgts)
    X_test  = np.stack(all_test_imgs)
    y_test  = np.stack(all_test_tgts)

    feat_train = extract_features(X_train)
    feat_test  = extract_features(X_test)

    wt_tr_mask = ~np.isnan(y_train[:, WATER_TEMP_IDX])
    wt_te_mask = ~np.isnan(y_test[:, WATER_TEMP_IDX])
    y_tr_wt = y_train[wt_tr_mask, WATER_TEMP_IDX]
    y_te_wt = y_test[wt_te_mask,  WATER_TEMP_IDX]

    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler

    scaler_sk = StandardScaler()
    X_tr_sc = scaler_sk.fit_transform(feat_train[wt_tr_mask])
    X_te_sc = scaler_sk.transform(feat_test[wt_te_mask])

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_tr_sc, y_tr_wt)
    ridge_r2 = compute_r2_1d(ridge.predict(X_te_sc), y_te_wt)
    logger.info(f"  Ridge water_temp R2: {ridge_r2:.4f}")
    results["Ridge"] = {"water_temp_r2": float(ridge_r2)}

    rf = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42)
    rf.fit(feat_train[wt_tr_mask], y_tr_wt)
    rf_r2 = compute_r2_1d(rf.predict(feat_test[wt_te_mask]), y_te_wt)
    logger.info(f"  Random Forest water_temp R2: {rf_r2:.4f}")
    results["RandomForest"] = {"water_temp_r2": float(rf_r2)}

    # ── Save & Report ────────────────────────────────────────────────────────
    out_path = RESULTS_DIR / "hydrovit_v8_benchmark.json"
    results["metadata"] = {
        "dataset": str(DATA_PATH),
        "n_total": n, "n_train": n_train, "n_val": n_val, "n_test": n_test,
        "split_seed": SPLIT_SEED,
        "elapsed_seconds": time.time() - t0,
    }
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved: {out_path}")

    def fmt(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "  N/A "
        return f"{v:.4f}"

    v8  = results.get("HydroViT_v8", {})
    v6  = results.get("HydroViT_v6", {})
    cnn = results.get("CNN_baseline", {})
    vit = results.get("ViT_no_pretrain", {})
    rdg = results.get("Ridge", {})
    rfo = results.get("RandomForest", {})

    print()
    print("=" * 75)
    print("HydroViT v8 Benchmark Results — v5 dataset (5,464 samples), seed=42")
    print(f"  n_train={n_train}, n_val={n_val}, n_test={n_test}")
    print("=" * 75)
    print(f"{'Model':<35} {'water_temp R²':>14}  {'mean R² (16)':>14}")
    print("-" * 75)
    print(f"{'HydroViT v8 (SpectralAttn)':<35} {fmt(v8.get('water_temp_r2')):>14}  {fmt(v8.get('mean_r2')):>14}  <- NEW")
    print(f"{'HydroViT v6 (prev best)':<35} {fmt(v6.get('water_temp_r2')):>14}  {fmt(v6.get('mean_r2')):>14}")
    print(f"{'CNN baseline (4-layer)':<35} {fmt(cnn.get('water_temp_r2')):>14}  {fmt(cnn.get('mean_r2')):>14}")
    print(f"{'ViT no pretraining':<35} {fmt(vit.get('water_temp_r2')):>14}  {fmt(vit.get('mean_r2')):>14}")
    print(f"{'Ridge Regression':<35} {fmt(rdg.get('water_temp_r2')):>14}  {'—':>14}")
    print(f"{'Random Forest':<35} {fmt(rfo.get('water_temp_r2')):>14}  {'—':>14}")
    print("=" * 75)
    print(f"Total time: {(time.time() - t0) / 60:.1f} min")
    print("DONE")


if __name__ == "__main__":
    main()
