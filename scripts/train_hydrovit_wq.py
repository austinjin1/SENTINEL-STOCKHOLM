#!/usr/bin/env python3
"""Fine-tune HydroViT's water quality regression head on co-registered data.

v5 improvements:
  - Gradual unfreezing (head → last blocks → full backbone)
  - Early stopping with patience=20
  - Params with zero data excluded from loss
  - Per-param loss weighting by sample count
  - Quality-weighted loss from co-registration temporal decay
  - Stronger regularization (weight_decay=0.05, dropout=0.3)
  - Lower backbone LR (1e-5)

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
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CKPT_DIR = Path("checkpoints/satellite")
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# Data priority chain
PAIRED_DATA_V4 = Path("data/processed/satellite/paired_wq_v4.npz")
PAIRED_DATA_V3 = Path("data/processed/satellite/paired_wq_v3.npz")
PAIRED_DATA_EXPANDED = Path("data/processed/satellite/paired_wq_expanded.npz")
PAIRED_DATA_ORIG = Path("data/processed/satellite/paired_wq.npz")
if PAIRED_DATA_V4.exists():
    PAIRED_DATA = PAIRED_DATA_V4
elif PAIRED_DATA_V3.exists():
    PAIRED_DATA = PAIRED_DATA_V3
elif PAIRED_DATA_EXPANDED.exists():
    PAIRED_DATA = PAIRED_DATA_EXPANDED
else:
    PAIRED_DATA = PAIRED_DATA_ORIG
PRETRAINED_CKPT = CKPT_DIR / "hydrovit_real_mae.pt"
OUTPUT_CKPT = CKPT_DIR / "hydrovit_wq_v5.pt"

# Training hyperparams — tuned for RTX 4060 8GB
BATCH_SIZE = 4
HEAD_LR = 3e-4
BACKBONE_LR = 1e-5       # 5x lower than before
HEAD_EPOCHS = 100         # reduced from 200
FINETUNE_EPOCHS = 100     # reduced from 200
USE_AMP = True
WEIGHT_DECAY = 0.05       # 5x stronger regularization
GRAD_CLIP = 1.0
EARLY_STOP_PATIENCE = 20

# Params with actual data (skip 2=secchi, 3=cdom, 13=oil, 14=acdom, 15=pollution)
ACTIVE_PARAMS = {0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12}
OPTICAL_PARAMS = {0, 1, 4}  # chl_a, turbidity, tss


class PairedWQDataset(Dataset):
    """Dataset of satellite images paired with water quality ground truth."""

    def __init__(self, data_path: str, augment: bool = False):
        self.augment = augment
        data = np.load(data_path, allow_pickle=True)
        self.images = data["images"].astype(np.float32)   # (N, 10, 224, 224)
        self.targets = data["targets"].astype(np.float32)  # (N, 16)

        # Load quality weights if available
        if "quality_weights" in data:
            self.quality_weights = data["quality_weights"].astype(np.float32)
        else:
            self.quality_weights = np.ones(len(self.images), dtype=np.float32)

        # Compute per-param sample counts for loss weighting
        self.param_counts = np.zeros(NUM_WATER_PARAMS, dtype=np.float32)
        for j in range(NUM_WATER_PARAMS):
            self.param_counts[j] = np.sum(~np.isnan(self.targets[:, j]))

        # Log-transform for log-normal parameters before z-scoring
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

        for i in range(NUM_WATER_PARAMS):
            valid = ~np.isnan(self.targets_norm[:, i])
            if valid.any():
                self.targets_norm[valid, i] = (
                    (self.targets_norm[valid, i] - self.target_mean[i]) / self.target_std[i]
                )

        logger.info(f"Loaded {len(self)} paired samples, "
                    f"images: {self.images.shape}, targets: {self.targets.shape}")
        active_density = 0
        for j in ACTIVE_PARAMS:
            n = self.param_counts[j]
            if n > 0:
                logger.info(f"  {PARAM_NAMES[j]:>20s}: {int(n):>5d} samples ({100*n/len(self):.0f}%)")
                active_density += n
        logger.info(f"  Active param density: {active_density / (len(self) * len(ACTIVE_PARAMS)):.3f}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx])  # (10, 224, 224)

        if self.augment:
            if torch.rand(1).item() > 0.5:
                image = image.flip(-1)
            if torch.rand(1).item() > 0.5:
                image = image.flip(-2)
            noise = 1.0 + 0.05 * (2 * torch.rand(10, 1, 1) - 1)
            image = image * noise

        padding = torch.zeros(3, image.shape[1], image.shape[2])
        image13 = torch.cat([image, padding], dim=0)  # (13, 224, 224)

        targets = torch.tensor(self.targets_norm[idx])  # (16,)
        qw = torch.tensor(self.quality_weights[idx])    # scalar
        return {"image": image13, "targets": targets, "quality_weight": qw}


def weighted_gaussian_nll(wq, unc, targets, quality_weights, param_weights):
    """Gaussian NLL loss with per-sample quality weights and per-param weights.

    Only computes loss on ACTIVE_PARAMS (skips params with no data).
    """
    total_loss = torch.tensor(0.0, device=wq.device)
    n_valid = 0

    for j in ACTIVE_PARAMS:
        valid = ~torch.isnan(targets[:, j])
        if valid.sum() == 0:
            continue
        mu = wq[valid, j]
        log_var = unc[valid, j]
        y = targets[valid, j]
        qw = quality_weights[valid]

        # Gaussian NLL: 0.5 * (log_var + (y - mu)^2 / exp(log_var))
        nll = 0.5 * (log_var + (y - mu) ** 2 / (log_var.exp() + 1e-6))

        # Apply quality weights (higher weight for tighter co-registration)
        weighted_nll = (nll * qw).mean()

        # Apply per-param weight
        total_loss = total_loss + weighted_nll * param_weights[j]
        n_valid += 1

    if n_valid == 0:
        return torch.tensor(float("nan"), device=wq.device)
    return total_loss / n_valid


def compute_r2_per_param(preds_dict, tgts_dict):
    r2_scores = {}
    for j in range(NUM_WATER_PARAMS):
        if preds_dict[j]:
            p = torch.cat(preds_dict[j])
            t = torch.cat(tgts_dict[j])
            if len(p) < 2:
                r2_scores[PARAM_NAMES[j]] = float("nan")
                continue
            ss_res = ((p - t) ** 2).sum()
            ss_tot = ((t - t.mean()) ** 2).sum()
            if ss_tot > 1e-8:
                r2_scores[PARAM_NAMES[j]] = (1 - ss_res / ss_tot).item()
            else:
                r2_scores[PARAM_NAMES[j]] = 0.0
        else:
            r2_scores[PARAM_NAMES[j]] = float("nan")
    return r2_scores


def evaluate(model, dataloader, device, param_weights):
    model.eval()
    preds = {j: [] for j in range(NUM_WATER_PARAMS)}
    tgts = {j: [] for j in range(NUM_WATER_PARAMS)}
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            image = batch["image"].to(device)
            targets = batch["targets"].to(device)
            qw = batch["quality_weight"].to(device)
            out = model(image)
            wq = out["water_quality_params"]
            unc = out["param_uncertainty"]

            loss = weighted_gaussian_nll(wq, unc, targets, qw, param_weights)
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
    mean_loss = total_loss / max(n_batches, 1)
    return r2_scores, mean_loss


def get_mean_r2(r2_scores):
    valid = [v for k, v in r2_scores.items()
             if not np.isnan(v) and PARAM_NAMES.index(k) in ACTIVE_PARAMS]
    return np.mean(valid) if valid else -1.0


def get_optical_r2(r2_scores):
    vals = []
    for j in OPTICAL_PARAMS:
        name = PARAM_NAMES[j]
        if name in r2_scores and not np.isnan(r2_scores[name]):
            vals.append(r2_scores[name])
    return np.mean(vals) if vals else -1.0


def train_phase(model, train_dl, val_dl, optimizer, scheduler, epochs,
                phase_name, device, param_weights, patience=None):
    """Train loop with optional early stopping. Returns best val R^2."""
    best_val_r2 = -float("inf")
    best_state = None
    no_improve = 0
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP and device.type == "cuda")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_dl:
            image = batch["image"].to(device)
            targets = batch["targets"].to(device)
            qw = batch["quality_weight"].to(device)

            with torch.amp.autocast("cuda", enabled=USE_AMP and device.type == "cuda"):
                out = model(image)
                wq = out["water_quality_params"]
                unc = out["param_uncertainty"]

                loss = weighted_gaussian_nll(wq, unc, targets, qw, param_weights)
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

        # Evaluate
        r2_scores, val_loss = evaluate(model, val_dl, device, param_weights)
        mean_r2 = get_mean_r2(r2_scores)
        optical_r2 = get_optical_r2(r2_scores)

        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
            train_loss = total_loss / max(n_batches, 1)
            logger.info(
                f"[{phase_name}] Ep {epoch+1:3d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Mean R2: {mean_r2:.4f} | Optical R2: {optical_r2:.4f}"
            )
            # Log per-param R2 every 25 epochs
            if (epoch + 1) % 25 == 0:
                for name in PARAM_NAMES:
                    idx = PARAM_NAMES.index(name)
                    if idx in ACTIVE_PARAMS and not np.isnan(r2_scores[name]):
                        logger.info(f"    {name:>20s}: R2={r2_scores[name]:.4f}")

        if mean_r2 > best_val_r2:
            best_val_r2 = mean_r2
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if patience and no_improve >= patience:
            logger.info(f"  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return best_val_r2


def unfreeze_last_n_blocks(model, n):
    """Unfreeze the last n transformer blocks in the ViT backbone."""
    # First freeze everything
    for p in model.parameters():
        p.requires_grad = False
    # Always unfreeze head
    for name, p in model.named_parameters():
        if "water_quality_head" in name:
            p.requires_grad = True

    # Unfreeze last n blocks
    if n > 0:
        # Find all backbone block parameters
        block_names = []
        for name, _ in model.named_parameters():
            if "backbone" in name and "blocks." in name:
                # Extract block index
                parts = name.split("blocks.")
                if len(parts) > 1:
                    block_idx = int(parts[1].split(".")[0])
                    if block_idx not in [b for b, _ in block_names]:
                        block_names.append((block_idx, name.split(f"blocks.{block_idx}")[0]))

        # Get total number of blocks
        all_block_indices = set()
        for name, _ in model.named_parameters():
            if "backbone" in name and "blocks." in name:
                parts = name.split("blocks.")
                if len(parts) > 1:
                    try:
                        all_block_indices.add(int(parts[1].split(".")[0]))
                    except ValueError:
                        pass

        if all_block_indices:
            total_blocks = max(all_block_indices) + 1
            unfreeze_from = total_blocks - n

            for name, p in model.named_parameters():
                if "backbone" in name and "blocks." in name:
                    parts = name.split("blocks.")
                    if len(parts) > 1:
                        try:
                            block_idx = int(parts[1].split(".")[0])
                            if block_idx >= unfreeze_from:
                                p.requires_grad = True
                        except ValueError:
                            pass

        # Also unfreeze norm and projection layers
        for name, p in model.named_parameters():
            if any(k in name for k in ["norm", "proj", "embedding_projection",
                                        "shared_projection", "multi_res", "temporal"]):
                p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")


def main():
    t0 = time.time()

    if not PAIRED_DATA.exists():
        logger.error(f"Paired dataset not found: {PAIRED_DATA}")
        sys.exit(1)

    logger.info(f"Using dataset: {PAIRED_DATA}")

    # ---------------------------------------------------------------
    # 1. Load data
    # ---------------------------------------------------------------
    dataset = PairedWQDataset(str(PAIRED_DATA), augment=True)
    dataset_eval = PairedWQDataset(str(PAIRED_DATA), augment=False)
    n = len(dataset)
    n_train = max(1, int(0.7 * n))
    n_val = max(1, int(0.15 * n))
    n_test = n - n_train - n_val
    if n_test < 1:
        n_test = 1
        n_train = n - n_val - n_test

    gen = torch.Generator().manual_seed(42)
    train_ds, _, _ = random_split(dataset, [n_train, n_val, n_test], generator=gen)
    gen2 = torch.Generator().manual_seed(42)
    _, val_ds, test_ds = random_split(dataset_eval, [n_train, n_val, n_test], generator=gen2)
    logger.info(f"Split: {n_train} train / {n_val} val / {n_test} test")
    logger.info(f"Config: batch={BATCH_SIZE}, AMP={USE_AMP}, wd={WEIGHT_DECAY}, "
                f"backbone_lr={BACKBONE_LR}, patience={EARLY_STOP_PATIENCE}")

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=0)

    # Compute per-param loss weights (inversely proportional to sqrt of count)
    # so rare params get more weight
    param_weights = torch.ones(NUM_WATER_PARAMS)
    counts = dataset.param_counts
    max_count = float(counts.max())
    for j in ACTIVE_PARAMS:
        if counts[j] > 0:
            param_weights[j] = float(1.0 / np.sqrt(counts[j] / max_count))
    # Zero out inactive params
    for j in range(NUM_WATER_PARAMS):
        if j not in ACTIVE_PARAMS:
            param_weights[j] = 0.0
    # Normalize so mean active weight = 1
    active_idx = list(ACTIVE_PARAMS)
    active_w = param_weights[active_idx]
    param_weights[active_idx] = active_w / active_w.mean()
    param_weights = param_weights.to(DEVICE)
    logger.info("Per-param loss weights:")
    for j in ACTIVE_PARAMS:
        logger.info(f"  {PARAM_NAMES[j]:>20s}: {param_weights[j]:.3f} ({int(counts[j])} samples)")

    # ---------------------------------------------------------------
    # 2. Load model with MAE-pretrained weights
    # ---------------------------------------------------------------
    logger.info(f"Loading HydroViT with pretrained weights from {PRETRAINED_CKPT}")
    model = SatelliteEncoder(pretrained=False).to(DEVICE)

    if PRETRAINED_CKPT.exists():
        state = torch.load(str(PRETRAINED_CKPT), map_location=DEVICE, weights_only=True)
        if "model" in state:
            state = state["model"]
        elif "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        logger.info(f"Loaded checkpoint: {len(missing)} missing, {len(unexpected)} unexpected keys")
    else:
        logger.warning(f"Checkpoint not found: {PRETRAINED_CKPT}, using random init")

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"HydroViT: {n_params:,} parameters")

    # ---------------------------------------------------------------
    # 3. Phase 1: Train WQ head only (backbone frozen) — 100 epochs
    # ---------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("PHASE 1: Train WQ Head (backbone frozen)")
    logger.info("=" * 60)

    for name, p in model.named_parameters():
        p.requires_grad = "water_quality_head" in name

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    optimizer1 = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=HEAD_LR, weight_decay=WEIGHT_DECAY,
    )
    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=HEAD_EPOCHS)

    best_r2_phase1 = train_phase(
        model, train_dl, val_dl, optimizer1, scheduler1,
        HEAD_EPOCHS, "Head", DEVICE, param_weights,
        patience=EARLY_STOP_PATIENCE,
    )
    logger.info(f"Phase 1 best val R2: {best_r2_phase1:.4f}")

    # ---------------------------------------------------------------
    # 4. Phase 2: Gradual unfreezing — last 2 blocks
    # ---------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("PHASE 2a: Unfreeze last 2 ViT blocks")
    logger.info("=" * 60)

    unfreeze_last_n_blocks(model, 2)

    optimizer2a = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and "water_quality_head" not in n],
         "lr": BACKBONE_LR},
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and "water_quality_head" in n],
         "lr": HEAD_LR * 0.1},
    ], weight_decay=WEIGHT_DECAY)
    scheduler2a = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2a, T_max=50)

    best_r2_2a = train_phase(
        model, train_dl, val_dl, optimizer2a, scheduler2a,
        50, "Unfreeze-2", DEVICE, param_weights,
        patience=EARLY_STOP_PATIENCE,
    )
    logger.info(f"Phase 2a best val R2: {best_r2_2a:.4f}")

    # ---------------------------------------------------------------
    # 5. Phase 2b: Unfreeze last 4 blocks
    # ---------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("PHASE 2b: Unfreeze last 4 ViT blocks")
    logger.info("=" * 60)

    unfreeze_last_n_blocks(model, 4)

    optimizer2b = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and "water_quality_head" not in n],
         "lr": BACKBONE_LR * 0.5},
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and "water_quality_head" in n],
         "lr": HEAD_LR * 0.05},
    ], weight_decay=WEIGHT_DECAY)
    scheduler2b = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2b, T_max=50)

    best_r2_2b = train_phase(
        model, train_dl, val_dl, optimizer2b, scheduler2b,
        50, "Unfreeze-4", DEVICE, param_weights,
        patience=EARLY_STOP_PATIENCE,
    )
    logger.info(f"Phase 2b best val R2: {best_r2_2b:.4f}")

    # ---------------------------------------------------------------
    # 6. Phase 2c: Full fine-tune at very low LR
    # ---------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("PHASE 2c: Full fine-tune (all params)")
    logger.info("=" * 60)

    for p in model.parameters():
        p.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Trainable: {trainable:,} / {n_params:,} (100%)")

    optimizer2c = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters()
                    if "water_quality_head" not in n],
         "lr": BACKBONE_LR * 0.2},  # very low: 2e-6
        {"params": [p for n, p in model.named_parameters()
                    if "water_quality_head" in n],
         "lr": HEAD_LR * 0.02},     # very low: 6e-6
    ], weight_decay=WEIGHT_DECAY)
    scheduler2c = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2c, T_max=50)

    best_r2_2c = train_phase(
        model, train_dl, val_dl, optimizer2c, scheduler2c,
        50, "FullFT", DEVICE, param_weights,
        patience=EARLY_STOP_PATIENCE,
    )
    logger.info(f"Phase 2c best val R2: {best_r2_2c:.4f}")

    # ---------------------------------------------------------------
    # 7. Final test evaluation
    # ---------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("TEST EVALUATION")
    logger.info("=" * 60)

    r2_scores, test_loss = evaluate(model, test_dl, DEVICE, param_weights)

    for name in PARAM_NAMES:
        r2 = r2_scores[name]
        idx = PARAM_NAMES.index(name)
        if idx not in ACTIVE_PARAMS:
            continue
        status = ""
        if not np.isnan(r2):
            if idx in OPTICAL_PARAMS:
                status = " [OPTICAL]" + (" OK" if r2 > 0.55 else " BELOW")
        logger.info(f"  {name:>25s}: R2 = {r2:>8.4f}{status}")

    mean_r2 = get_mean_r2(r2_scores)
    optical_r2 = get_optical_r2(r2_scores)
    logger.info(f"  Mean R2 (active params): {mean_r2:.4f}")
    logger.info(f"  Mean R2 (optical params): {optical_r2:.4f}")

    if mean_r2 > 0.65:
        logger.info("  >>> EXCELLENT (mean R2 > 0.65) <<<")
    elif mean_r2 > 0.55:
        logger.info("  >>> GOOD (mean R2 > 0.55) <<<")
    elif mean_r2 > 0.30:
        logger.info("  >>> ACCEPTABLE (mean R2 > 0.30) <<<")
    else:
        logger.info(f"  >>> BELOW THRESHOLD ({mean_r2:.4f}) <<<")

    # ---------------------------------------------------------------
    # 8. Save checkpoint and results
    # ---------------------------------------------------------------
    torch.save(model.state_dict(), OUTPUT_CKPT)
    logger.info(f"Saved checkpoint: {OUTPUT_CKPT}")

    elapsed = time.time() - t0
    results = {
        "mean_r2": mean_r2,
        "optical_mean_r2": optical_r2,
        "best_val_r2_phase1": best_r2_phase1,
        "best_val_r2_phase2a": best_r2_2a,
        "best_val_r2_phase2b": best_r2_2b,
        "best_val_r2_phase2c": best_r2_2c,
        "per_param_r2": {k: v if not np.isnan(v) else None for k, v in r2_scores.items()},
        "test_loss": test_loss,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "elapsed_seconds": elapsed,
        "config": {
            "batch_size": BATCH_SIZE,
            "head_lr": HEAD_LR,
            "backbone_lr": BACKBONE_LR,
            "weight_decay": WEIGHT_DECAY,
            "head_epochs": HEAD_EPOCHS,
            "finetune_epochs": FINETUNE_EPOCHS,
            "early_stop_patience": EARLY_STOP_PATIENCE,
            "active_params": list(ACTIVE_PARAMS),
        },
    }
    results_path = CKPT_DIR / "results_wq_v5.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved: {results_path}")
    logger.info(f"Total time: {elapsed/60:.1f} min")
    logger.info("DONE")


if __name__ == "__main__":
    main()
