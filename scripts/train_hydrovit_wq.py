#!/usr/bin/env python3
"""Fine-tune HydroViT's water quality regression head on co-registered data.

Loads:
  - Paired dataset from coregister_satellite_wq.py
  - HydroViT MAE-pretrained backbone checkpoint

Trains the WaterQualityHead while keeping backbone frozen (phase 1),
then optionally fine-tunes backbone at lower LR (phase 2).

Reports per-parameter R^2.
Saves checkpoint to checkpoints/satellite/hydrovit_wq_finetuned.pt

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
# Priority: v3 (NWIS+GRQA) > expanded (GRQA+WQP) > original
PAIRED_DATA_V3 = Path("data/processed/satellite/paired_wq_v3.npz")
PAIRED_DATA_EXPANDED = Path("data/processed/satellite/paired_wq_expanded.npz")
PAIRED_DATA_ORIG = Path("data/processed/satellite/paired_wq.npz")
if PAIRED_DATA_V3.exists():
    PAIRED_DATA = PAIRED_DATA_V3
elif PAIRED_DATA_EXPANDED.exists():
    PAIRED_DATA = PAIRED_DATA_EXPANDED
else:
    PAIRED_DATA = PAIRED_DATA_ORIG
PRETRAINED_CKPT = CKPT_DIR / "hydrovit_real_mae.pt"
OUTPUT_CKPT = CKPT_DIR / "hydrovit_wq_v3.pt"

# Training hyperparams
BATCH_SIZE = 8
HEAD_LR = 3e-4
BACKBONE_LR = 5e-5
HEAD_EPOCHS = 100
FINETUNE_EPOCHS = 100
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0

# Optically-active parameter indices (target R^2 > 0.55)
OPTICAL_PARAMS = {0, 1, 2, 3, 4}  # chl_a, turbidity, secchi, cdom, tss


class PairedWQDataset(Dataset):
    """Dataset of satellite images paired with water quality ground truth."""

    def __init__(self, data_path: str):
        data = np.load(data_path, allow_pickle=True)
        self.images = data["images"].astype(np.float32)   # (N, 10, 224, 224)
        self.targets = data["targets"].astype(np.float32)  # (N, 16)

        # Compute per-parameter normalization stats from non-NaN values
        self.target_mean = np.nanmean(self.targets, axis=0)
        self.target_std = np.nanstd(self.targets, axis=0)
        self.target_std[self.target_std < 1e-6] = 1.0
        # Handle all-NaN columns
        nan_cols = np.all(np.isnan(self.targets), axis=0)
        self.target_mean[nan_cols] = 0.0
        self.target_std[nan_cols] = 1.0

        # Log-transform for log-normal parameters before z-scoring
        self.log_params = {0, 1, 3, 4, 5, 6, 8, 9, 12, 14}
        self.targets_norm = self.targets.copy()
        for i in self.log_params:
            valid = ~np.isnan(self.targets_norm[:, i])
            if valid.any():
                vals = self.targets_norm[valid, i]
                vals = np.maximum(vals, 1e-6)  # avoid log(0)
                self.targets_norm[valid, i] = np.log1p(vals)

        # Recompute stats on transformed targets
        self.target_mean = np.nanmean(self.targets_norm, axis=0)
        self.target_std = np.nanstd(self.targets_norm, axis=0)
        self.target_std[self.target_std < 1e-6] = 1.0
        nan_cols = np.all(np.isnan(self.targets_norm), axis=0)
        self.target_mean[nan_cols] = 0.0
        self.target_std[nan_cols] = 1.0

        # Z-score normalize
        for i in range(16):
            valid = ~np.isnan(self.targets_norm[:, i])
            if valid.any():
                self.targets_norm[valid, i] = (
                    (self.targets_norm[valid, i] - self.target_mean[i]) / self.target_std[i]
                )

        logger.info(f"Loaded {len(self)} paired samples, "
                    f"images: {self.images.shape}, targets: {self.targets.shape}")
        logger.info(f"Non-NaN target density: {(~np.isnan(self.targets)).sum() / self.targets.size:.3f}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx])  # (10, 224, 224)
        # Pad from 10 bands to 13 bands (add 3 zero channels for S3 OLCI)
        padding = torch.zeros(3, image.shape[1], image.shape[2])
        image13 = torch.cat([image, padding], dim=0)  # (13, 224, 224)

        targets = torch.tensor(self.targets_norm[idx])  # (16,) with NaN
        return {"image": image13, "targets": targets}


def compute_r2_per_param(preds_dict, tgts_dict):
    """Compute R^2 per parameter from accumulated predictions/targets.

    Args:
        preds_dict: {param_idx: list of tensors}
        tgts_dict: {param_idx: list of tensors}

    Returns:
        dict of {param_name: r2_score}
    """
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


def evaluate(model, dataloader, device):
    """Evaluate model on a dataloader, returning per-param R^2 and mean loss."""
    model.eval()
    preds = {j: [] for j in range(NUM_WATER_PARAMS)}
    tgts = {j: [] for j in range(NUM_WATER_PARAMS)}
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            image = batch["image"].to(device)
            targets = batch["targets"].to(device)
            out = model(image)
            wq = out["water_quality_params"]
            unc = out["param_uncertainty"]

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
    mean_loss = total_loss / max(n_batches, 1)
    return r2_scores, mean_loss


def train_phase(model, train_dl, val_dl, optimizer, scheduler, epochs, phase_name, device):
    """Train loop for one phase. Returns best val R^2."""
    best_val_r2 = -float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_dl:
            image = batch["image"].to(device)
            targets = batch["targets"].to(device)

            out = model(image)
            wq = out["water_quality_params"]
            unc = out["param_uncertainty"]

            valid = ~torch.isnan(targets)
            if valid.sum() == 0:
                continue

            loss = WaterQualityHead.gaussian_nll_loss(wq, unc, targets)
            if torch.isnan(loss):
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        if scheduler is not None:
            scheduler.step()

        # Evaluate
        r2_scores, val_loss = evaluate(model, val_dl, device)
        valid_r2s = [v for v in r2_scores.values() if not np.isnan(v)]
        mean_r2 = np.mean(valid_r2s) if valid_r2s else -1.0

        # Optically-active params mean R^2
        optical_r2s = []
        for j in OPTICAL_PARAMS:
            name = PARAM_NAMES[j]
            if name in r2_scores and not np.isnan(r2_scores[name]):
                optical_r2s.append(r2_scores[name])
        optical_mean_r2 = np.mean(optical_r2s) if optical_r2s else -1.0

        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
            train_loss = total_loss / max(n_batches, 1)
            logger.info(
                f"[{phase_name}] Ep {epoch+1:3d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Mean R2: {mean_r2:.4f} | Optical R2: {optical_mean_r2:.4f}"
            )

        if mean_r2 > best_val_r2:
            best_val_r2 = mean_r2
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Restore best state
    if best_state is not None:
        model.load_state_dict(best_state)

    return best_val_r2


def main():
    t0 = time.time()

    if not PAIRED_DATA.exists():
        logger.error(f"Paired dataset not found: {PAIRED_DATA}")
        logger.error("Run scripts/coregister_satellite_wq.py first")
        sys.exit(1)

    # ---------------------------------------------------------------
    # 1. Load data
    # ---------------------------------------------------------------
    dataset = PairedWQDataset(str(PAIRED_DATA))
    n = len(dataset)
    n_train = max(1, int(0.7 * n))
    n_val = max(1, int(0.15 * n))
    n_test = n - n_train - n_val
    if n_test < 1:
        n_test = 1
        n_train = n - n_val - n_test

    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42),
    )
    logger.info(f"Split: {n_train} train / {n_val} val / {n_test} test")

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=0)

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
    # 3. Phase 1: Train water_quality_head only (frozen backbone)
    # ---------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("PHASE 1: Train WQ Head (backbone frozen)")
    logger.info("=" * 60)

    # Freeze everything except water_quality_head
    for name, p in model.named_parameters():
        p.requires_grad = "water_quality_head" in name

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    optimizer1 = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=HEAD_LR,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=HEAD_EPOCHS)

    best_r2_phase1 = train_phase(
        model, train_dl, val_dl, optimizer1, scheduler1,
        HEAD_EPOCHS, "Head", DEVICE,
    )
    logger.info(f"Phase 1 best val R2: {best_r2_phase1:.4f}")

    # ---------------------------------------------------------------
    # 4. Phase 2: Fine-tune backbone + head at lower LR
    # ---------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("PHASE 2: Fine-tune backbone + head")
    logger.info("=" * 60)

    # Unfreeze everything
    for p in model.parameters():
        p.requires_grad = True

    # Differential learning rates
    backbone_params = []
    head_params = []
    for name, p in model.named_parameters():
        if "water_quality_head" in name:
            head_params.append(p)
        else:
            backbone_params.append(p)

    optimizer2 = torch.optim.AdamW([
        {"params": backbone_params, "lr": BACKBONE_LR},
        {"params": head_params, "lr": HEAD_LR * 0.2},  # Lower LR after phase 1
    ], weight_decay=WEIGHT_DECAY)

    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=FINETUNE_EPOCHS)

    best_r2_phase2 = train_phase(
        model, train_dl, val_dl, optimizer2, scheduler2,
        FINETUNE_EPOCHS, "Finetune", DEVICE,
    )
    logger.info(f"Phase 2 best val R2: {best_r2_phase2:.4f}")

    # ---------------------------------------------------------------
    # 5. Final test evaluation
    # ---------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("TEST EVALUATION")
    logger.info("=" * 60)

    r2_scores, test_loss = evaluate(model, test_dl, DEVICE)

    for name in PARAM_NAMES:
        r2 = r2_scores[name]
        status = ""
        if not np.isnan(r2):
            idx = PARAM_NAMES.index(name)
            if idx in OPTICAL_PARAMS:
                status = " [OPTICAL]" + (" OK" if r2 > 0.55 else " BELOW")
        logger.info(f"  {name:>25s}: R2 = {r2:>8.4f}{status}")

    valid_r2s = [v for v in r2_scores.values() if not np.isnan(v)]
    mean_r2 = np.mean(valid_r2s) if valid_r2s else -1.0
    logger.info(f"  Mean R2 (all params with data): {mean_r2:.4f}")

    optical_r2s = []
    for j in OPTICAL_PARAMS:
        name = PARAM_NAMES[j]
        if name in r2_scores and not np.isnan(r2_scores[name]):
            optical_r2s.append(r2_scores[name])
    optical_mean_r2 = np.mean(optical_r2s) if optical_r2s else -1.0
    logger.info(f"  Mean R2 (optical params): {optical_mean_r2:.4f}")

    if optical_mean_r2 > 0.55:
        logger.info("  >>> OPTICAL PARAMS TARGET MET (R2 > 0.55) <<<")
    elif mean_r2 > 0.30:
        logger.info("  >>> ACCEPTABLE (mean R2 > 0.30) <<<")
    else:
        logger.info(f"  >>> BELOW THRESHOLD ({mean_r2:.4f}) <<<")

    # ---------------------------------------------------------------
    # 6. Save checkpoint and results
    # ---------------------------------------------------------------
    torch.save(model.state_dict(), OUTPUT_CKPT)
    logger.info(f"Saved checkpoint: {OUTPUT_CKPT}")

    elapsed = time.time() - t0
    results = {
        "mean_r2": mean_r2,
        "optical_mean_r2": optical_mean_r2,
        "best_val_r2_phase1": best_r2_phase1,
        "best_val_r2_phase2": best_r2_phase2,
        "per_param_r2": {k: v if not np.isnan(v) else None for k, v in r2_scores.items()},
        "test_loss": test_loss,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "elapsed_seconds": elapsed,
    }
    results_path = CKPT_DIR / "results_wq_finetuned.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved: {results_path}")
    logger.info(f"Total time: {elapsed/60:.1f} min")
    logger.info("DONE")


if __name__ == "__main__":
    main()
