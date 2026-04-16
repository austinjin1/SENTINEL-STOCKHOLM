#!/usr/bin/env python3
"""HydroViT v8 — SpectralAttention + deeper head + v5 data (5,464 pairs).

Key improvements over v6 (R²=0.760 water_temp):
  1. SpectralBandAttention: learned per-pixel band reweighting before ViT
  2. Bigger WQ head: 384→512→256→16 with residual + dropout
  3. More data: v5 dataset (5,464 pairs, 1.9× v3)
  4. Weighted MSE: water_temp 3×, other params 1×
  5. Longer finetuning: 150+150 epochs (vs 100+100)
  6. StepLR decay instead of fixed LR

MIT License — Bryan Cheng, 2026
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
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sentinel.models.satellite_encoder.model import SatelliteEncoder
from sentinel.models.satellite_encoder.parameter_head import PARAM_NAMES, NUM_WATER_PARAMS
from sentinel.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_DIR = Path("checkpoints/satellite")
CKPT_DIR.mkdir(parents=True, exist_ok=True)

PAIRED_DATA = Path("data/processed/satellite/paired_wq_v5.npz")
if not PAIRED_DATA.exists():
    PAIRED_DATA = Path("data/processed/satellite/paired_wq_v3.npz")
    logger.warning("v5 not found, falling back to v3")

PRETRAINED_CKPT = CKPT_DIR / "hydrovit_real_mae.pt"
OUTPUT_CKPT     = CKPT_DIR / "hydrovit_wq_v8.pt"
RESULTS_JSON    = CKPT_DIR / "results_wq_v8.json"

SEED         = 42
BATCH_SIZE   = 8
HEAD_LR      = 3e-4
BACKBONE_LR  = 5e-5
HEAD_EPOCHS  = 150
FINETUNE_EPOCHS = 150
WEIGHT_DECAY = 0.01
GRAD_CLIP    = 1.0
USE_AMP      = True

# water_temp is param index 11
WATER_TEMP_IDX = PARAM_NAMES.index("water_temp") if "water_temp" in PARAM_NAMES else 11
PARAM_WEIGHTS = torch.ones(NUM_WATER_PARAMS)
PARAM_WEIGHTS[WATER_TEMP_IDX] = 3.0   # water_temp weighted 3×

torch.manual_seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# SpectralBandAttention — key new module
# ---------------------------------------------------------------------------
class SpectralBandAttention(nn.Module):
    """Per-pixel learned reweighting of spectral bands.

    Given input [B, C, H, W], computes a channel attention weight per pixel:
      1. Global avg pool → [B, C] → FC → ReLU → FC → Sigmoid → [B, C]
      2. Per-pixel: compute local channel statistics [B, C, H, W]
      3. Fuse global + local → final gate [B, C, H, W]
      4. Output: x * gate (soft band selection)

    This is a lightweight spatial-spectral attention (≈ CBAM channel branch
    extended with local context), with ~2× the parameters of SE but better
    spatial specificity for water body detection.
    """

    def __init__(self, in_channels: int, reduction: int = 4):
        super().__init__()
        mid = max(in_channels // reduction, 4)
        # Global path (SE-style)
        self.global_fc = nn.Sequential(
            nn.Linear(in_channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, in_channels),
            nn.Sigmoid(),
        )
        # Local path: 1×1 conv over channels per pixel
        self.local_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, in_channels, 1, bias=False),
            nn.Sigmoid(),
        )
        # Learnable mix parameter
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Global branch
        g = x.mean(dim=(2, 3))                      # [B, C]
        g = self.global_fc(g).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        # Local branch
        lc = self.local_conv(x)                     # [B, C, H, W]
        # Fuse
        alpha = self.alpha.sigmoid()
        gate = alpha * g + (1 - alpha) * lc         # [B, C, H, W]
        return x * gate


# ---------------------------------------------------------------------------
# Deeper WQ Head
# ---------------------------------------------------------------------------
class DeepWQHead(nn.Module):
    """384 → 512 → 256 → 16 with residual + dropout.

    Separate mean prediction per parameter. No log_var (use weighted MSE).
    """

    def __init__(self, input_dim: int = 384, hidden: int = 512, n_params: int = 16):
        super().__init__()
        self.bn_in = nn.LayerNorm(input_dim)
        self.fc1   = nn.Linear(input_dim, hidden)
        self.bn1   = nn.LayerNorm(hidden)
        self.fc2   = nn.Linear(hidden, 256)
        self.bn2   = nn.LayerNorm(256)
        self.drop  = nn.Dropout(0.2)
        # Residual projection if dims differ
        self.res   = nn.Linear(input_dim, 256) if input_dim != 256 else nn.Identity()
        self.out   = nn.Linear(256, n_params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.bn_in(x)
        h = F.gelu(self.bn1(self.fc1(h)))
        h = self.drop(h)
        h = self.bn2(self.fc2(h))
        h = h + self.res(x)        # residual
        h = F.gelu(h)
        return self.out(h)


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------
class HydroViTV8(nn.Module):
    """SpectralAttention + ViT-S/16 backbone + DeepWQHead."""

    def __init__(self, pretrained_ckpt: Path | None = None, in_bands: int = 10):
        super().__init__()
        self.spectral_attn = SpectralBandAttention(in_channels=in_bands)
        # Pad to 13 channels for SatelliteEncoder (expects 10 S2 + 3 S3 bands)
        self.encoder = SatelliteEncoder()
        # SatelliteEncoder projects to shared_embed_dim=256, not raw ViT dim=384
        self.wq_head = DeepWQHead(input_dim=self.encoder.shared_embed_dim)

        if pretrained_ckpt and pretrained_ckpt.exists():
            try:
                ckpt = torch.load(str(pretrained_ckpt), map_location="cpu", weights_only=False)
                state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
                # Load backbone only (ignore head)
                missing, unexpected = self.encoder.load_state_dict(state, strict=False)
                logger.info(f"Loaded pretrained backbone: {len(missing)} missing, {len(unexpected)} unexpected keys")
            except Exception as e:
                logger.warning(f"Could not load pretrained ckpt: {e}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 10, H, W]  (10-band Sentinel-2)
        Returns:
            preds: [B, 16]
        """
        x = self.spectral_attn(x)            # [B, 10, H, W]
        # Pad to 13 channels
        pad = torch.zeros(x.shape[0], 3, x.shape[2], x.shape[3], device=x.device)
        x13 = torch.cat([x, pad], dim=1)     # [B, 13, H, W]
        out = self.encoder(x13)
        emb = out["embedding"]   # [B, 256] — shared projected embedding
        return self.wq_head(emb)             # [B, 16]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PairedWQDataset(Dataset):
    def __init__(self, data_path: str, idxs: np.ndarray):
        data = np.load(data_path, allow_pickle=True)
        images  = data["images"].astype(np.float32)[idxs]
        targets = data["targets"].astype(np.float32)[idxs]

        self.images = images
        self.log_params = {0, 1, 3, 4, 5, 6, 8, 9, 12, 14}
        targets_norm = targets.copy()
        for i in self.log_params:
            valid = ~np.isnan(targets_norm[:, i])
            if valid.any():
                vals = np.maximum(targets_norm[valid, i], 1e-6)
                targets_norm[valid, i] = np.log1p(vals)

        self.mean = np.nanmean(targets_norm, axis=0)
        self.std  = np.nanstd(targets_norm, axis=0)
        self.std[self.std < 1e-6] = 1.0
        self.mean[np.all(np.isnan(targets_norm), axis=0)] = 0.0

        for i in range(targets_norm.shape[1]):
            valid = ~np.isnan(targets_norm[:, i])
            if valid.any():
                targets_norm[valid, i] = (targets_norm[valid, i] - self.mean[i]) / self.std[i]

        self.targets_norm = targets_norm

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return {
            "image":   torch.tensor(self.images[idx]),
            "targets": torch.tensor(self.targets_norm[idx]),
        }


# ---------------------------------------------------------------------------
# Loss + metrics
# ---------------------------------------------------------------------------
def weighted_mse(preds: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    valid = ~torch.isnan(targets)
    if valid.sum() == 0:
        return torch.tensor(0.0, device=preds.device, requires_grad=True)
    w = weights.to(preds.device).unsqueeze(0).expand_as(targets)
    diff2 = (preds - targets.nan_to_num(0.0)) ** 2
    return (diff2 * w * valid.float()).sum() / (w * valid.float()).sum().clamp(min=1e-6)


def compute_r2(preds: dict, tgts: dict) -> dict:
    r2 = {}
    for j in range(NUM_WATER_PARAMS):
        if preds[j]:
            p = torch.cat(preds[j])
            t = torch.cat(tgts[j])
            ss_res = ((p - t) ** 2).sum()
            ss_tot = ((t - t.mean()) ** 2).sum()
            r2[PARAM_NAMES[j]] = (1 - ss_res / ss_tot).item() if ss_tot > 1e-8 else 0.0
        else:
            r2[PARAM_NAMES[j]] = float("nan")
    return r2


def evaluate(model, loader, device):
    model.eval()
    preds = {j: [] for j in range(NUM_WATER_PARAMS)}
    tgts  = {j: [] for j in range(NUM_WATER_PARAMS)}
    total_loss, n_b = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            img = batch["image"].to(device)
            tgt = batch["targets"].to(device)
            pred = model(img)
            loss = weighted_mse(pred, tgt, PARAM_WEIGHTS)
            if not torch.isnan(loss):
                total_loss += loss.item(); n_b += 1
            valid = ~torch.isnan(tgt)
            for j in range(NUM_WATER_PARAMS):
                mask = valid[:, j]
                if mask.sum() > 0:
                    preds[j].append(pred[:, j][mask].cpu())
                    tgts[j].append(tgt[:, j][mask].cpu())
    r2 = compute_r2(preds, tgts)
    return r2, total_loss / max(n_b, 1)


# ---------------------------------------------------------------------------
# Training phase
# ---------------------------------------------------------------------------
def train_phase(model, train_dl, val_dl, optimizer, scheduler, epochs, label, device):
    best_wt_r2 = -float("inf")
    best_state = None
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP and device.type == "cuda")

    for ep in range(epochs):
        model.train()
        total_loss, n_b = 0.0, 0
        for batch in train_dl:
            img = batch["image"].to(device)
            tgt = batch["targets"].to(device)
            with torch.amp.autocast("cuda", enabled=USE_AMP and device.type == "cuda"):
                pred = model(img)
                loss = weighted_mse(pred, tgt, PARAM_WEIGHTS)
            if torch.isnan(loss):
                optimizer.zero_grad(); continue
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer); scaler.update()
            total_loss += loss.item(); n_b += 1

        if scheduler:
            scheduler.step()

        if (ep + 1) % 10 == 0 or ep == epochs - 1:
            r2, val_loss = evaluate(model, val_dl, device)
            valid_r2s = [v for v in r2.values() if not np.isnan(v)]
            mean_r2 = np.mean(valid_r2s) if valid_r2s else -1.0
            wt_r2 = r2.get("water_temp", float("nan"))
            logger.info(
                f"[{label}] Ep {ep+1:3d}/{epochs} | "
                f"tr={total_loss/max(n_b,1):.4f} val={val_loss:.4f} | "
                f"mean_R2={mean_r2:.4f} water_temp_R2={wt_r2:.4f}"
            )
            if not np.isnan(wt_r2) and wt_r2 > best_wt_r2:
                best_wt_r2 = wt_r2
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                logger.info(f"  -> New best water_temp R2: {best_wt_r2:.4f}")

    if best_state:
        model.load_state_dict(best_state)
    return best_wt_r2


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("HydroViT v8 — SpectralAttention + deeper head + v5 data")
    logger.info(f"  Data:   {PAIRED_DATA}")
    logger.info(f"  Device: {DEVICE}")
    logger.info("=" * 70)

    # ── Data split ────────────────────────────────────────────────────────
    data = np.load(str(PAIRED_DATA), allow_pickle=True)
    N = len(data["images"])
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(N)
    n_test = int(0.15 * N)
    n_val  = int(0.15 * N)
    test_idx  = idx[:n_test]
    val_idx   = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]
    logger.info(f"Split: {len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test")

    train_ds = PairedWQDataset(str(PAIRED_DATA), train_idx)
    val_ds   = PairedWQDataset(str(PAIRED_DATA), val_idx)
    test_ds  = PairedWQDataset(str(PAIRED_DATA), test_idx)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────
    model = HydroViTV8(pretrained_ckpt=PRETRAINED_CKPT).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {n_params:,} trainable parameters")

    # ── Phase 1: Train head only (freeze backbone) ─────────────────────
    logger.info("\n--- Phase 1: Head-only training ---")
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    opt1 = torch.optim.AdamW(
        [p for p in list(model.spectral_attn.parameters()) + list(model.wq_head.parameters()) if p.requires_grad],
        lr=HEAD_LR, weight_decay=WEIGHT_DECAY
    )
    sch1 = torch.optim.lr_scheduler.StepLR(opt1, step_size=HEAD_EPOCHS // 3, gamma=0.5)
    best_p1 = train_phase(model, train_dl, val_dl, opt1, sch1, HEAD_EPOCHS, "Head", DEVICE)
    logger.info(f"Phase 1 best water_temp R2: {best_p1:.4f}")

    # ── Phase 2: Full fine-tune ─────────────────────────────────────────
    logger.info("\n--- Phase 2: Full fine-tune ---")
    for p in model.encoder.parameters():
        p.requires_grad_(True)
    opt2 = torch.optim.AdamW([
        {"params": model.spectral_attn.parameters(), "lr": HEAD_LR},
        {"params": model.wq_head.parameters(),       "lr": HEAD_LR},
        {"params": model.encoder.parameters(),       "lr": BACKBONE_LR},
    ], weight_decay=WEIGHT_DECAY)
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=FINETUNE_EPOCHS, eta_min=1e-6)
    best_p2 = train_phase(model, train_dl, val_dl, opt2, sch2, FINETUNE_EPOCHS, "Finetune", DEVICE)
    logger.info(f"Phase 2 best water_temp R2: {best_p2:.4f}")

    # ── Final test evaluation ──────────────────────────────────────────
    logger.info("\n--- Final test evaluation ---")
    r2_test, _ = evaluate(model, test_dl, DEVICE)
    valid_r2s = [v for v in r2_test.values() if not np.isnan(v)]
    mean_r2 = float(np.mean(valid_r2s)) if valid_r2s else -1.0
    wt_r2 = r2_test.get("water_temp", float("nan"))
    logger.info(f"  water_temp R2 = {wt_r2:.4f}")
    logger.info(f"  mean R2       = {mean_r2:.4f}")
    for name, r2 in r2_test.items():
        if not np.isnan(r2):
            marker = " ***" if name == "water_temp" else ""
            logger.info(f"    {name:>30s}: R2={r2:.4f}{marker}")

    # ── Save checkpoint ──────────────────────────────────────────────────
    torch.save(model.state_dict(), str(OUTPUT_CKPT))
    logger.info(f"Saved: {OUTPUT_CKPT}")

    elapsed = time.time() - t0
    results = {
        "model": "HydroViT_v8",
        "architecture": "SpectralBandAttention + ViT-S/16 + DeepWQHead",
        "n_params": n_params,
        "data": str(PAIRED_DATA),
        "n_train": int(len(train_idx)),
        "n_val":   int(len(val_idx)),
        "n_test":  int(len(test_idx)),
        "water_temp_r2": float(wt_r2),
        "mean_r2": float(mean_r2),
        "per_param_r2": {k: (float(v) if not np.isnan(v) else None) for k, v in r2_test.items()},
        "best_val_r2_phase1": float(best_p1),
        "best_val_r2_phase2": float(best_p2),
        "elapsed_seconds": round(elapsed, 1),
    }
    with open(str(RESULTS_JSON), "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results: {RESULTS_JSON}")
    logger.info(f"Total elapsed: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
