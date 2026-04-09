#!/usr/bin/env python3
"""Fine-tune HydroViT v6 — match collaborator's exact setup, just more data.

The collaborator achieved mean R2=0.123, water_temp R2=0.674 with:
  - 2,861 pairs, batch=8, head_lr=3e-4, backbone_lr=5e-5
  - weight_decay=0.01, 100+100 epochs, gaussian_nll_loss
  - No augmentation, no quality weighting, no param weighting

This script: SAME hyperparams, but v4 dataset (4,202 pairs).
Only differences: batch=4 (VRAM), AMP (VRAM).

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
# Config — matches collaborator's setup exactly
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CKPT_DIR = Path("checkpoints/satellite")
CKPT_DIR.mkdir(parents=True, exist_ok=True)

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
OUTPUT_CKPT = CKPT_DIR / "hydrovit_wq_v6.pt"

# EXACT collaborator hyperparams (only batch=4 and AMP for VRAM)
BATCH_SIZE = 4              # was 8, reduced for RTX 4060
HEAD_LR = 3e-4              # same
BACKBONE_LR = 5e-5          # same
HEAD_EPOCHS = 100           # same
FINETUNE_EPOCHS = 100       # same
USE_AMP = True              # added for VRAM
WEIGHT_DECAY = 0.01         # same
GRAD_CLIP = 1.0             # same

OPTICAL_PARAMS = {0, 1, 2, 3, 4}


class PairedWQDataset(Dataset):
    """Exact same dataset class as collaborator's."""

    def __init__(self, data_path: str):
        data = np.load(data_path, allow_pickle=True)
        self.images = data["images"].astype(np.float32)
        self.targets = data["targets"].astype(np.float32)

        # Log-transform for log-normal parameters
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
                    (self.targets_norm[valid, i] - self.target_mean[i]) / self.target_std[i]
                )

        logger.info(f"Loaded {len(self)} paired samples, "
                    f"images: {self.images.shape}, targets: {self.targets.shape}")
        logger.info(f"Non-NaN target density: "
                    f"{(~np.isnan(self.targets)).sum() / self.targets.size:.3f}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx])
        padding = torch.zeros(3, image.shape[1], image.shape[2])
        image13 = torch.cat([image, padding], dim=0)
        targets = torch.tensor(self.targets_norm[idx])
        return {"image": image13, "targets": targets}


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


def evaluate(model, dataloader, device):
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
    best_val_r2 = -float("inf")
    best_state = None
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP and device.type == "cuda")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_dl:
            image = batch["image"].to(device)
            targets = batch["targets"].to(device)

            with torch.amp.autocast("cuda", enabled=USE_AMP and device.type == "cuda"):
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
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            n_batches += 1

        if scheduler is not None:
            scheduler.step()

        r2_scores, val_loss = evaluate(model, val_dl, device)
        valid_r2s = [v for v in r2_scores.values() if not np.isnan(v)]
        mean_r2 = np.mean(valid_r2s) if valid_r2s else -1.0

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
            # Per-param R2 every 25 epochs
            if (epoch + 1) % 25 == 0 or epoch == epochs - 1:
                for name in PARAM_NAMES:
                    r2 = r2_scores[name]
                    if not np.isnan(r2):
                        logger.info(f"    {name:>25s}: R2={r2:.4f}")

        if mean_r2 > best_val_r2:
            best_val_r2 = mean_r2
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return best_val_r2


def main():
    t0 = time.time()

    if not PAIRED_DATA.exists():
        logger.error(f"Paired dataset not found: {PAIRED_DATA}")
        sys.exit(1)

    logger.info(f"Dataset: {PAIRED_DATA}")
    logger.info(f"Matching collaborator setup: batch={BATCH_SIZE}, head_lr={HEAD_LR}, "
                f"backbone_lr={BACKBONE_LR}, wd={WEIGHT_DECAY}, "
                f"epochs={HEAD_EPOCHS}+{FINETUNE_EPOCHS}")

    # 1. Load data
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

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=0)

    # 2. Load model
    logger.info(f"Loading pretrained weights from {PRETRAINED_CKPT}")
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

    logger.info(f"HydroViT: {sum(p.numel() for p in model.parameters()):,} parameters")

    # 3. Phase 1: Head only
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

    best_r2_p1 = train_phase(model, train_dl, val_dl, optimizer1, scheduler1,
                              HEAD_EPOCHS, "Head", DEVICE)
    logger.info(f"Phase 1 best val R2: {best_r2_p1:.4f}")

    # 4. Phase 2: Full fine-tune
    logger.info("=" * 60)
    logger.info("PHASE 2: Fine-tune backbone + head")
    logger.info("=" * 60)

    for p in model.parameters():
        p.requires_grad = True

    backbone_params = []
    head_params = []
    for name, p in model.named_parameters():
        if "water_quality_head" in name:
            head_params.append(p)
        else:
            backbone_params.append(p)

    optimizer2 = torch.optim.AdamW([
        {"params": backbone_params, "lr": BACKBONE_LR},
        {"params": head_params, "lr": HEAD_LR * 0.2},
    ], weight_decay=WEIGHT_DECAY)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=FINETUNE_EPOCHS)

    best_r2_p2 = train_phase(model, train_dl, val_dl, optimizer2, scheduler2,
                              FINETUNE_EPOCHS, "Finetune", DEVICE)
    logger.info(f"Phase 2 best val R2: {best_r2_p2:.4f}")

    # 5. Test evaluation
    logger.info("=" * 60)
    logger.info("TEST EVALUATION")
    logger.info("=" * 60)

    r2_scores, test_loss = evaluate(model, test_dl, DEVICE)

    for name in PARAM_NAMES:
        r2 = r2_scores[name]
        if not np.isnan(r2):
            idx = PARAM_NAMES.index(name)
            opt = " [OPTICAL]" if idx in OPTICAL_PARAMS else ""
            logger.info(f"  {name:>25s}: R2 = {r2:>8.4f}{opt}")

    valid_r2s = [v for v in r2_scores.values() if not np.isnan(v)]
    mean_r2 = np.mean(valid_r2s) if valid_r2s else -1.0
    logger.info(f"  Mean R2: {mean_r2:.4f}")

    # 6. Save
    torch.save(model.state_dict(), OUTPUT_CKPT)
    logger.info(f"Saved checkpoint: {OUTPUT_CKPT}")

    elapsed = time.time() - t0
    results = {
        "mean_r2": mean_r2,
        "best_val_r2_phase1": best_r2_p1,
        "best_val_r2_phase2": best_r2_p2,
        "per_param_r2": {k: v if not np.isnan(v) else None for k, v in r2_scores.items()},
        "test_loss": test_loss,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "elapsed_seconds": elapsed,
    }
    with open(CKPT_DIR / "results_wq_v6.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Total time: {elapsed/60:.1f} min")
    logger.info("DONE")


if __name__ == "__main__":
    main()
