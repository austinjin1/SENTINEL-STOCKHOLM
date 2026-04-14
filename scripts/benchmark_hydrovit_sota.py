#!/usr/bin/env python3
"""benchmark_hydrovit_sota.py — Add DenseNet121 + ResNet50 to HydroViT benchmark.

Implements transfer-learning baselines from:
  HydroVision (arXiv 2509.01882, Sep 2025):
    "Predicting Optically Active Parameters in Surface Water Using Computer Vision"
    Best model: DenseNet121 (ImageNet pretrained, fine-tuned)
    Reported R²: CDOM=0.898, Chl=0.788, Chl-α=0.678, Turbidity=0.498
    Note: water_temp NOT included in HydroVision — our primary metric.

Both models evaluated on SAME paired_wq_v5.npz test split (seed=42, 70/15/15).
Appends results to results/benchmarks/hydrovit_v8_benchmark.json.

Bryan Cheng, SENTINEL project, 2026
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.models import densenet121, resnet50, DenseNet121_Weights, ResNet50_Weights

PROJECT_ROOT = Path("/home/bcheng/SENTINEL")
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from sentinel.utils.logging import get_logger

# Import data + eval helpers from existing benchmark
from benchmark_hydrovit_v8 import (
    PairedWQDatasetWith13,
    _compute_r2,
    WATER_TEMP_IDX,
    PARAM_NAMES,
    NUM_WATER_PARAMS,
    LOG_PARAMS,
    SPLIT_SEED,
    BATCH_SIZE,
    GRAD_CLIP,
    USE_AMP,
)

logger = get_logger(__name__)
DEVICE       = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
RESULTS_DIR  = PROJECT_ROOT / "results" / "benchmarks"
DATA_PATH    = PROJECT_ROOT / "data" / "processed" / "satellite" / "paired_wq_v5.npz"
RESULTS_PATH = RESULTS_DIR / "hydrovit_v8_benchmark.json"

FINETUNE_EPOCHS = 80
EARLY_STOP_PAT  = 15
LR_HEAD         = 1e-3
LR_BACKBONE     = 1e-4
WEIGHT_DECAY    = 0.01


# ─────────────────────────────────────────────────────────────────────────────
# Transfer-learning wrappers
# ─────────────────────────────────────────────────────────────────────────────
class DenseNet121WQ(nn.Module):
    """DenseNet121 pretrained on ImageNet, fine-tuned for 16-param water quality.

    First conv adapted from 3→10 channels (average pretrained weights,
    replicate to cover all 10 spectral bands) following HydroVision (arXiv 2509.01882).
    """

    def __init__(self, n_params: int = NUM_WATER_PARAMS, in_channels: int = 10):
        super().__init__()
        base = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)

        # Adapt first conv: 3→in_channels
        old_conv = base.features.conv0   # Conv2d(3, 64, 7, stride=2, padding=3)
        new_conv = nn.Conv2d(
            in_channels, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        # Initialize: average the 3 RGB channel weights, tile to in_channels
        with torch.no_grad():
            avg_w = old_conv.weight.mean(dim=1, keepdim=True)   # (64, 1, 7, 7)
            new_conv.weight.copy_(avg_w.repeat(1, in_channels, 1, 1))
        base.features.conv0 = new_conv

        # Replace classifier
        in_feats = base.classifier.in_features
        base.classifier = nn.Sequential(
            nn.Linear(in_feats, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_params),
        )
        self.model = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ResNet50WQ(nn.Module):
    """ResNet50 pretrained on ImageNet, fine-tuned for 16-param water quality."""

    def __init__(self, n_params: int = NUM_WATER_PARAMS, in_channels: int = 10):
        super().__init__()
        base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        # Adapt first conv
        old_conv = base.conv1
        new_conv = nn.Conv2d(
            in_channels, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        with torch.no_grad():
            avg_w = old_conv.weight.mean(dim=1, keepdim=True)
            new_conv.weight.copy_(avg_w.repeat(1, in_channels, 1, 1))
        base.conv1 = new_conv

        in_feats = base.fc.in_features
        base.fc = nn.Sequential(
            nn.Linear(in_feats, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_params),
        )
        self.model = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ─────────────────────────────────────────────────────────────────────────────
# NaN-aware MSE loss
# ─────────────────────────────────────────────────────────────────────────────
def nan_mse(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    valid = ~torch.isnan(tgt)
    if valid.sum() == 0:
        return pred.sum() * 0.0
    return F.mse_loss(pred[valid], tgt[valid])


# ─────────────────────────────────────────────────────────────────────────────
# Train transfer-learning model
# ─────────────────────────────────────────────────────────────────────────────
def train_transfer(
    model: nn.Module,
    train_dl: DataLoader,
    val_dl: DataLoader,
    name: str,
    epochs: int = FINETUNE_EPOCHS,
    patience: int = EARLY_STOP_PAT,
) -> nn.Module:
    # Two param groups: lower LR for backbone, higher for head
    head_params, back_params = [], []
    for n, p in model.named_parameters():
        if "classifier" in n or "fc" in n:
            head_params.append(p)
        else:
            back_params.append(p)

    optimizer = torch.optim.AdamW([
        {"params": back_params, "lr": LR_BACKBONE},
        {"params": head_params, "lr": LR_HEAD},
    ], weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP and DEVICE.type == "cuda")

    best_val = float("inf")
    no_imp   = 0
    best_state = None
    model.to(DEVICE)

    for ep in range(1, epochs + 1):
        model.train()
        for batch in train_dl:
            img = batch["image"].to(DEVICE)
            tgt = batch["targets"].to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=USE_AMP and DEVICE.type == "cuda"):
                pred = model(img)
                loss = nan_mse(pred, tgt)
                if torch.isnan(loss):
                    continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dl:
                img = batch["image"].to(DEVICE)
                tgt = batch["targets"].to(DEVICE)
                with torch.amp.autocast("cuda", enabled=USE_AMP and DEVICE.type == "cuda"):
                    pred = model(img)
                    loss = nan_mse(pred, tgt)
                if not torch.isnan(loss):
                    val_loss += loss.item()

        if val_loss < best_val:
            best_val = val_loss
            no_imp   = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_imp += 1

        if ep % 20 == 0 or ep == 1:
            logger.info(f"  [{name}] Epoch {ep}/{epochs} | val_loss={val_loss:.4f}")

        if no_imp >= patience:
            logger.info(f"  Early stopping at epoch {ep} (best val_loss={best_val:.4f})")
            break

    if best_state:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def evaluate_model(model: nn.Module, dataloader: DataLoader) -> dict:
    model.eval()
    preds = {j: [] for j in range(NUM_WATER_PARAMS)}
    tgts  = {j: [] for j in range(NUM_WATER_PARAMS)}
    for batch in dataloader:
        img    = batch["image"].to(DEVICE)
        targets = batch["targets"].to(DEVICE)
        with torch.amp.autocast("cuda", enabled=USE_AMP and DEVICE.type == "cuda"):
            pred = model(img)
        valid = ~torch.isnan(targets)
        for j in range(NUM_WATER_PARAMS):
            mask = valid[:, j]
            if mask.sum() > 0:
                preds[j].append(pred[:, j][mask].cpu())
                tgts[j].append(targets[:, j][mask].cpu())
    return _compute_r2(preds, tgts)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    torch.manual_seed(SPLIT_SEED)
    np.random.seed(SPLIT_SEED)
    t0 = time.time()

    logger.info("HydroViT SOTA benchmark: DenseNet121 + ResNet50 (HydroVision arXiv 2509.01882)")
    logger.info(f"Device: {DEVICE}")

    # ── Load dataset (same as benchmark_hydrovit_v8.py) ──────────────────────
    full_ds = PairedWQDatasetWith13(str(DATA_PATH))
    N = len(full_ds)
    n_tr = int(0.70 * N)
    n_va = int(0.15 * N)
    n_te = N - n_tr - n_va
    g = torch.Generator().manual_seed(SPLIT_SEED)
    tr_ds, va_ds, te_ds = random_split(full_ds, [n_tr, n_va, n_te], generator=g)
    logger.info(f"Split: {n_tr} train / {n_va} val / {n_te} test")

    tr_dl = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    va_dl = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    te_dl = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Load existing results
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            results = json.load(f)
    else:
        results = {}

    sota_results = {}

    # ── DenseNet121 ───────────────────────────────────────────────────────────
    logger.info(f"\n--- DenseNet121 (ImageNet pretrained, HydroVision-style, {FINETUNE_EPOCHS} epochs) ---")
    dn121 = DenseNet121WQ(n_params=NUM_WATER_PARAMS, in_channels=10)
    n_params = sum(p.numel() for p in dn121.parameters())
    logger.info(f"  DenseNet121: {n_params:,} params")
    dn121 = train_transfer(dn121, tr_dl, va_dl, "DenseNet121")
    r2_dn121 = evaluate_model(dn121, te_dl)
    wt_r2 = r2_dn121.get("water_temp", float("nan"))
    mean_r2 = float(np.nanmean([v for v in r2_dn121.values() if v is not None]))
    logger.info(f"  DenseNet121 water_temp R²={wt_r2:.4f} | mean R²={mean_r2:.4f}")
    sota_results["DenseNet121 (HydroVision-style)"] = {
        "water_temp_r2": round(wt_r2, 4),
        "mean_r2":       round(mean_r2, 4),
        "per_param_r2":  {k: (round(v, 4) if v is not None else None)
                          for k, v in r2_dn121.items()},
        "n_params":      n_params,
        "reference":     "arXiv 2509.01882 — HydroVision: Predicting Optically Active Parameters, Sep 2025",
        "note":          "ImageNet pretrained, first conv adapted 3→10 channels",
    }
    del dn121
    torch.cuda.empty_cache()

    # ── ResNet50 ──────────────────────────────────────────────────────────────
    logger.info(f"\n--- ResNet50 (ImageNet pretrained, {FINETUNE_EPOCHS} epochs) ---")
    rn50 = ResNet50WQ(n_params=NUM_WATER_PARAMS, in_channels=10)
    n_params_rn = sum(p.numel() for p in rn50.parameters())
    logger.info(f"  ResNet50: {n_params_rn:,} params")
    rn50 = train_transfer(rn50, tr_dl, va_dl, "ResNet50")
    r2_rn50 = evaluate_model(rn50, te_dl)
    wt_r2_rn = r2_rn50.get("water_temp", float("nan"))
    mean_r2_rn = float(np.nanmean([v for v in r2_rn50.values() if v is not None]))
    logger.info(f"  ResNet50 water_temp R²={wt_r2_rn:.4f} | mean R²={mean_r2_rn:.4f}")
    sota_results["ResNet50 (ImageNet pretrained)"] = {
        "water_temp_r2": round(wt_r2_rn, 4),
        "mean_r2":       round(mean_r2_rn, 4),
        "per_param_r2":  {k: (round(v, 4) if v is not None else None)
                          for k, v in r2_rn50.items()},
        "n_params":      n_params_rn,
        "note":          "ImageNet pretrained, first conv adapted 3→10 channels",
    }

    elapsed = time.time() - t0
    results.update(sota_results)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {RESULTS_PATH}  (elapsed {elapsed/60:.1f} min)")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("HydroViT v8 vs. Published SOTA — Satellite Water Quality (water_temp R²)")
    print(f"{'Model':<38} {'Water Temp R²':>14} {'Mean R²':>9}")
    print("-" * 70)
    our_models = {
        "HydroViT v8 (ours)":     results.get("HydroViT_v8", {}).get("water_temp_r2"),
        "CNN baseline (ours)":    results.get("CNN_baseline", {}).get("water_temp_r2"),
        "ViT (no pretrain, ours)":results.get("ViT_no_pretrain", {}).get("water_temp_r2"),
        "RandomForest (ours)":    results.get("RandomForest", {}).get("water_temp_r2"),
        "Ridge (ours)":           results.get("Ridge", {}).get("water_temp_r2"),
    }
    sota_map = {
        "DenseNet121 (HydroVision-style)": sota_results["DenseNet121 (HydroVision-style)"],
        "ResNet50 (ImageNet pretrained)":  sota_results["ResNet50 (ImageNet pretrained)"],
    }
    all_rows = list(our_models.items()) + [
        (k, v.get("water_temp_r2")) for k, v in sota_map.items()
    ]
    for name, r2 in sorted(all_rows, key=lambda x: x[1] or -99, reverse=True):
        ref = " ← published" if "HydroVision" in name else ""
        r2_str = f"{r2:.4f}" if r2 is not None else "  N/A"
        mean_str = ""
        if name in sota_map:
            mean_str = f"{sota_map[name]['mean_r2']:.4f}"
        print(f"  {name:<36} {r2_str:>14} {mean_str:>9}{ref}")
    print("=" * 70)


if __name__ == "__main__":
    main()
