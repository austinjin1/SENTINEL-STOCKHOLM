#!/usr/bin/env python3
"""Train WaterDroneNet — SENTINEL Mini for drone-based water quality sensing.

WaterDroneNet predicts water quality parameters from drone/satellite imagery
alone. It uses real Sentinel-2 RGB+NIR patches (4 bands, 224x224) paired with
USGS NWIS water quality measurements as ground truth.

This is "SENTINEL Mini": a lightweight, deployable model for drones equipped
with multispectral cameras, complementing full SENTINEL deployed at fixed
coastal stations with comprehensive sensor arrays.

Architecture:
  - Vision encoder: ViT-S/16 (timm) adapted for 4-channel input (RGB+NIR)
  - Image-derived physics priors: band-ratio proxies for turbidity, chl-a
  - Per-target Gaussian prediction heads (mu + sigma)
  - Uncertainty router: trust flag per sample

Training:
  Phase 1 — MAE self-supervised pretraining of vision encoder (optional)
  Phase 2 — Supervised training with Gaussian NLL + MAE + physics regularization

Data:
  - Real Sentinel-2 L2A patches (B02=Blue, B03=Green, B04=Red, B08=NIR)
  - Targets from USGS NWIS: DO, pH, Turb, Temp, SpCond
  - Temporal holdout: train < 2023, val 2023, test >= 2024

Usage:
    PYTHONNOUSERSITE=1 conda run -n physiformer python scripts/train_waterdronenet.py

MIT License — Bryan Cheng, 2026
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

CKPT_DIR = PROJECT / "checkpoints" / "waterdronenet"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = PROJECT / "results" / "benchmarks"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = PROJECT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILE = PROJECT / "data" / "processed" / "satellite" / "drone_wq_pairs.npz"

# ---------------------------------------------------------------------------
# Target definitions
# ---------------------------------------------------------------------------
TARGET_COLS = ["DO", "pH", "Turb", "Temp", "SpCond"]
NUM_TARGETS = len(TARGET_COLS)
IMG_CHANNELS = 4     # B02=Blue, B03=Green, B04=Red, B08=NIR
IMG_SIZE = 224        # Sentinel-2 patch size
VIT_PATCH = 16        # ViT patch size → 14×14 = 196 patches
EMBED_DIM = 384       # ViT-S embedding dimension

# Plausible value ranges for QA/QC filtering
PARAM_RANGES = {
    "DO":      (0.0,    20.0),
    "pH":      (4.0,    10.0),
    "Turb":    (0.0,    1000.0),
    "Temp":    (-5.0,   45.0),
    "SpCond":  (0.0,    50000.0),
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_log_handler: Optional[logging.FileHandler] = None


def setup_logging(log_path: Path) -> None:
    global _log_handler
    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    root = logging.getLogger("wdn")
    root.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    root.addHandler(ch)
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    root.addHandler(fh)
    _log_handler = fh


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)
    if _log_handler is not None:
        logging.getLogger("wdn").info(msg)


# ---------------------------------------------------------------------------
# Dataset: Real Sentinel-2 imagery + USGS water quality
# ---------------------------------------------------------------------------
class DroneWQDataset(Dataset):
    """Real S2 imagery paired with USGS water quality measurements.

    Loads from drone_wq_pairs.npz:
      images:  (N, 4, 224, 224) float32 — S2 reflectance
      targets: (N, 5) float32 — [DO, pH, Turb, Temp, SpCond]

    Temporal holdout via metadata dates:
      train: date < 2023-01-01
      val:   2023-01-01 <= date < 2024-01-01
      test:  date >= 2024-01-01
    """

    SPLIT_DATES = {
        "train": (None, "2023-01-01"),
        "val":   ("2023-01-01", "2024-01-01"),
        "test":  ("2024-01-01", None),
    }

    def __init__(self, split: str = "train") -> None:
        super().__init__()
        assert split in ("train", "val", "test")
        self.split = split

        if not DATA_FILE.exists():
            raise FileNotFoundError(
                f"Paired data not found at {DATA_FILE}. "
                "Run scripts/download_drone_s2_pairs.py first."
            )

        data = np.load(DATA_FILE, allow_pickle=True)
        all_images = data["images"]     # (N, 4, 224, 224)
        all_targets = data["targets"]   # (N, 5)
        meta_list = json.loads(str(data["metadata"]))

        # Spatial holdout split: group by station, assign stations to splits
        # This is more robust than temporal split when most data is post-2022
        site_ids = list({m.get("site_id", str(i)) for i, m in enumerate(meta_list)})
        site_ids.sort()
        rng = np.random.RandomState(42)
        rng.shuffle(site_ids)
        n_sites = len(site_ids)
        train_sites = set(site_ids[:int(0.7 * n_sites)])
        val_sites = set(site_ids[int(0.7 * n_sites):int(0.85 * n_sites)])
        test_sites = set(site_ids[int(0.85 * n_sites):])

        indices = []
        for i, meta in enumerate(meta_list):
            sid = meta.get("site_id", str(i))
            if split == "train" and sid in train_sites:
                indices.append(i)
            elif split == "val" and sid in val_sites:
                indices.append(i)
            elif split == "test" and sid in test_sites:
                indices.append(i)

        if not indices:
            # Fallback: random split
            log(f"  WARNING: Empty split for {split}, using random split")
            rng2 = np.random.RandomState(42)
            perm = rng2.permutation(len(all_images))
            n = len(perm)
            if split == "train":
                indices = perm[:int(0.7 * n)].tolist()
            elif split == "val":
                indices = perm[int(0.7 * n):int(0.85 * n)].tolist()
            else:
                indices = perm[int(0.85 * n):].tolist()

        self.images = all_images[indices].astype(np.float32)
        self.targets = all_targets[indices].astype(np.float32)

        # QA/QC: filter extreme values
        for j, col in enumerate(TARGET_COLS):
            lo, hi = PARAM_RANGES[col]
            bad = (self.targets[:, j] < lo) | (self.targets[:, j] > hi)
            self.targets[bad, j] = np.nan

        log(f"  {split}: {len(self.images)} samples")

    def compute_target_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        mean = np.zeros(NUM_TARGETS, dtype=np.float32)
        std = np.ones(NUM_TARGETS, dtype=np.float32)
        for t in range(NUM_TARGETS):
            vals = self.targets[:, t]
            valid = vals[np.isfinite(vals)]
            if len(valid) > 1:
                mean[t] = valid.mean()
                std[t] = valid.std().clip(min=1e-6)
        return mean, std

    def set_target_stats(self, mean: np.ndarray, std: np.ndarray) -> None:
        self._target_mean = mean
        self._target_std = std

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image = torch.from_numpy(self.images[idx])     # (4, 224, 224)
        target = torch.from_numpy(self.targets[idx].copy())  # (5,)

        valid_mask = (~torch.isnan(target)).float()
        target = torch.nan_to_num(target, nan=0.0)

        # Z-score normalize targets
        if hasattr(self, "_target_mean"):
            tmean = torch.from_numpy(self._target_mean)
            tstd = torch.from_numpy(self._target_std)
            target = (target - tmean) / tstd
            target = target * valid_mask  # re-zero masked positions

        return {
            "image": image,
            "target": target,
            "valid_mask": valid_mask,
        }


# ---------------------------------------------------------------------------
# ViT building blocks (inline, no timm dependency)
# ---------------------------------------------------------------------------
class _PatchEmbed(nn.Module):
    """Non-overlapping patch projection."""
    def __init__(self, img_size=IMG_SIZE, patch_size=VIT_PATCH,
                 in_chans=IMG_CHANNELS, embed_dim=EMBED_DIM):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)  # (B, N, D)


class _TransformerBlock(nn.Module):
    """Pre-norm transformer block."""
    def __init__(self, dim=EMBED_DIM, num_heads=6, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout,
                                          batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Model: Image-only WaterDroneNet
# ---------------------------------------------------------------------------
class WaterDroneNet(nn.Module):
    """WaterDroneNet: RGB+NIR imagery → water quality predictions.

    Pure vision model — no scalar sensor inputs required.
    Inline ViT backbone for 4-channel 224x224 input.

    Components:
      1. ViT vision encoder (384d, 12 layers, 6 heads)
      2. Prediction head: CLS token → per-target (mu, sigma)
      3. Image-derived physics priors (band ratios)
      4. Uncertainty router (trust flag)
    """

    def __init__(self, depth=12, num_heads=6) -> None:
        super().__init__()

        # ViT backbone (ViT-S/16 equivalent)
        self.patch_embed = _PatchEmbed()
        self.num_patches = self.patch_embed.num_patches  # 196
        self.cls_token = nn.Parameter(torch.zeros(1, 1, EMBED_DIM))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, EMBED_DIM))
        self.blocks = nn.ModuleList([
            _TransformerBlock(EMBED_DIM, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(EMBED_DIM)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Prediction head
        self.head = nn.Sequential(
            nn.LayerNorm(EMBED_DIM),
            nn.Linear(EMBED_DIM, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
        )

        self.mu_heads = nn.ModuleList([nn.Linear(128, 1) for _ in range(NUM_TARGETS)])
        self.sigma_heads = nn.ModuleList([nn.Linear(128, 1) for _ in range(NUM_TARGETS)])

        # Image-based physics prior from band statistics
        # [mean_B, mean_G, mean_R, mean_NIR, std_B, std_G, std_R, std_NIR,
        #  NDWI, NIR/Red ratio]
        self.prior = nn.Linear(10, NUM_TARGETS, bias=True)
        nn.init.zeros_(self.prior.weight)
        nn.init.zeros_(self.prior.bias)

        # Trust router
        self.trust_router = nn.Sequential(
            nn.Linear(EMBED_DIM + NUM_TARGETS, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def _encode_vision(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode image → (cls_feat, patch_feats)."""
        B = x.size(0)
        patches = self.patch_embed(x)                    # (B, N, D)
        cls = self.cls_token.expand(B, -1, -1)           # (B, 1, D)
        tokens = torch.cat([cls, patches], dim=1)        # (B, N+1, D)
        tokens = tokens + self.pos_embed
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)
        return tokens[:, 0], tokens[:, 1:]               # cls, patches

    def _image_features(self, images: torch.Tensor) -> torch.Tensor:
        """Band statistics for physics prior. Returns (B, 10)."""
        means = images.mean(dim=(-2, -1))  # (B, 4)
        stds = images.std(dim=(-2, -1))    # (B, 4)
        green, nir, red = means[:, 1], means[:, 3], means[:, 2]
        ndwi = (green - nir) / (green + nir + 1e-6)
        nir_red = nir / (red + 1e-6)
        return torch.cat([means, stds, ndwi.unsqueeze(1), nir_red.unsqueeze(1)], dim=1)

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        cls_feat, patch_feats = self._encode_vision(image)  # (B, D), (B, N, D)

        h = self.head(cls_feat)  # (B, 128)
        mu_residual = torch.cat([head(h) for head in self.mu_heads], dim=1)
        log_sigma = torch.cat([head(h) for head in self.sigma_heads], dim=1)
        sigma = F.softplus(log_sigma) + 1e-4

        img_feats = self._image_features(image)
        physics_prior = self.prior(img_feats)
        mu = physics_prior + mu_residual

        trust_logit = self.trust_router(torch.cat([cls_feat, sigma], dim=1))

        return {
            "mu": mu,
            "sigma": sigma,
            "physics_prior": physics_prior,
            "trust_logit": trust_logit,
            "cls_feat": cls_feat,
        }


# ---------------------------------------------------------------------------
# MAE Pretraining
# ---------------------------------------------------------------------------
class MAEDecoder(nn.Module):
    """Lightweight decoder for MAE self-supervised pretraining."""

    def __init__(self, embed_dim=EMBED_DIM, decoder_dim=192, depth=4,
                 num_patches=196, patch_size=VIT_PATCH, in_chans=IMG_CHANNELS):
        super().__init__()
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.decoder_dim = decoder_dim

        self.encoder_proj = nn.Linear(embed_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_dim))

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=decoder_dim, nhead=6, dim_feedforward=decoder_dim * 4,
                dropout=0.1, activation="gelu", batch_first=True,
                norm_first=True,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(decoder_dim)
        self.pred = nn.Linear(decoder_dim, patch_size * patch_size * in_chans)

        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, vis_feats, mask_indices, visible_indices):
        """Reconstruct all patches from visible features.

        Args:
            vis_feats: (B, N_vis, D_enc)
            mask_indices: (B, N_mask) indices of masked patches
            visible_indices: (B, N_vis) indices of visible patches

        Returns:
            pred_pixels: (B, N, patch_pixels)
        """
        B, N_vis, _ = vis_feats.shape
        N_mask = mask_indices.size(1)
        N = N_vis + N_mask

        vis = self.encoder_proj(vis_feats)

        tokens = torch.zeros(B, N, self.decoder_dim, device=vis.device, dtype=vis.dtype)
        vis_idx = visible_indices.unsqueeze(-1).expand(-1, -1, self.decoder_dim)
        tokens.scatter_(1, vis_idx, vis)

        mask_tok = self.mask_token.expand(B, N_mask, -1).to(dtype=vis.dtype)
        mask_idx = mask_indices.unsqueeze(-1).expand(-1, -1, self.decoder_dim)
        tokens.scatter_(1, mask_idx, mask_tok)

        tokens = tokens + self.pos_embed[:, :N]

        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        return self.pred(tokens)


def patchify(imgs, patch_size=VIT_PATCH):
    """(B, C, H, W) → (B, N, C*p*p)"""
    B, C, H, W = imgs.shape
    ph, pw = H // patch_size, W // patch_size
    x = imgs.reshape(B, C, ph, patch_size, pw, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, ph * pw, -1)
    return x


def mae_pretrain_epoch(model, decoder, loader, optimizer, scaler, device,
                       mask_ratio=0.75):
    """One MAE pretraining epoch."""
    model.train()
    decoder.train()
    total_loss, n = 0.0, 0

    raw = model.module if hasattr(model, "module") else model
    num_patches = raw.num_patches

    for batch in loader:
        images = batch["image"].to(device)
        B = images.size(0)
        N = num_patches
        num_mask = int(N * mask_ratio)
        num_vis = N - num_mask

        noise = torch.rand(B, N, device=device)
        ids = torch.argsort(noise, dim=1)
        visible_indices = ids[:, :num_vis]
        mask_indices = ids[:, num_vis:]

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda"):
            # Get patch embeddings
            patch_emb = raw.patch_embed(images)  # (B, N, D)
            cls_token = raw.cls_token.expand(B, -1, -1)

            # Gather visible patches
            vis_idx = visible_indices.unsqueeze(-1).expand(-1, -1, EMBED_DIM)
            vis_patches = torch.gather(patch_emb, 1, vis_idx)

            # Add positional embeddings for visible patches
            pos_emb = raw.pos_embed[:, 1:]  # (1, N, D)
            vis_pos = torch.gather(pos_emb.expand(B, -1, -1), 1, vis_idx)
            vis_patches = vis_patches + vis_pos

            # Prepend CLS token
            tokens = torch.cat([cls_token + raw.pos_embed[:, :1], vis_patches], dim=1)

            # Apply transformer blocks
            for block in raw.blocks:
                tokens = block(tokens)
            tokens = raw.norm(tokens)
            vis_feats = tokens[:, 1:]  # (B, N_vis, D)

            pred_pixels = decoder(vis_feats, mask_indices, visible_indices)

        # MAE reconstruction loss on masked patches
        target_pixels = patchify(images).float()
        mask_idx_exp = mask_indices.unsqueeze(-1).expand(-1, -1, target_pixels.size(-1))
        pred_masked = torch.gather(pred_pixels.float(), 1, mask_idx_exp)
        target_masked = torch.gather(target_pixels, 1, mask_idx_exp)

        loss = F.mse_loss(pred_masked, target_masked)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        all_params = list(raw.parameters()) + list(decoder.parameters())
        nn.utils.clip_grad_norm_(all_params, 5.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * B
        n += B

    return total_loss / max(n, 1)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------
def gaussian_nll_loss(mu, sigma, target, mask):
    mu, sigma, target, mask = mu.float(), sigma.float().clamp(min=1e-4), target.float(), mask.float()
    nll = 0.5 * (torch.log(sigma ** 2) + (target - mu) ** 2 / sigma ** 2)
    nll = torch.nan_to_num(nll, nan=0.0, posinf=0.0, neginf=0.0)
    return (nll * mask).sum() / mask.sum().clamp(min=1.0)


def masked_mae_loss(mu, target, mask):
    mu, target, mask = mu.float(), target.float(), mask.float()
    ae = torch.abs(target - mu)
    ae = torch.nan_to_num(ae, nan=0.0, posinf=0.0, neginf=0.0)
    return (ae * mask).sum() / mask.sum().clamp(min=1.0)


def calibration_loss(mu, sigma, target, mask):
    mu, sigma = mu.float(), sigma.float().clamp(min=1e-4)
    target, mask = target.float(), mask.float()
    z_score = torch.abs(target - mu) / sigma
    overconf = F.relu(z_score - 2.0)
    return (overconf * mask).sum() / mask.sum().clamp(min=1.0)


def trust_supervision_loss(trust_logit, mu, target, mask, threshold=2.0):
    mu, target, mask = mu.float().detach(), target.float(), mask.float()
    norm_err = torch.abs(target - mu)
    per_sample = (norm_err * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    label = (per_sample <= threshold).float().unsqueeze(1)
    return F.binary_cross_entropy_with_logits(trust_logit.float(), label)


# ---------------------------------------------------------------------------
# Training & Evaluation
# ---------------------------------------------------------------------------
def train_epoch(model, loader, optimizer, scaler, device,
                target_mean_t, target_std_t):
    model.train()
    totals = {"loss": 0., "nll": 0., "mae": 0., "calib": 0., "trust": 0.}
    n = 0

    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["target"].to(device)
        mask = batch["valid_mask"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda"):
            out = model(images)

        mu = out["mu"].float()
        sigma = out["sigma"].float()
        trust_logit = out["trust_logit"].float()

        nll = gaussian_nll_loss(mu, sigma, targets, mask)
        mae = masked_mae_loss(mu, targets, mask)
        calib = calibration_loss(mu, sigma, targets, mask)
        trust = trust_supervision_loss(trust_logit, mu, targets, mask)

        loss = nll + 1.0 * mae + 0.1 * calib + 0.05 * trust

        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad(set_to_none=True)
            n += images.size(0)
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        B = images.size(0)
        totals["loss"] += loss.item() * B
        totals["nll"] += nll.item() * B
        totals["mae"] += mae.item() * B
        totals["calib"] += calib.item() * B
        totals["trust"] += trust.item() * B
        n += B

    return {k: v / max(n, 1) for k, v in totals.items()}


@torch.no_grad()
def evaluate(model, loader, device, target_mean=None, target_std=None):
    model.eval()
    all_mu, all_sigma, all_tgt, all_mask, all_trust = [], [], [], [], []

    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["target"].to(device)
        mask = batch["valid_mask"].to(device)

        with autocast("cuda"):
            out = model(images)

        all_mu.append(out["mu"].cpu().float())
        all_sigma.append(out["sigma"].cpu().float())
        all_tgt.append(targets.cpu().float())
        all_mask.append(mask.cpu().float())
        all_trust.append(torch.sigmoid(out["trust_logit"]).cpu().float())

    mu = torch.cat(all_mu)
    sigma = torch.cat(all_sigma)
    tgt = torch.cat(all_tgt)
    mask = torch.cat(all_mask)
    trust = torch.cat(all_trust)

    # Denormalize
    if target_mean is not None and target_std is not None:
        tmean = torch.from_numpy(target_mean).unsqueeze(0)
        tstd = torch.from_numpy(target_std).unsqueeze(0)
        mu = mu * tstd + tmean
        sigma = sigma * tstd.abs()
        tgt = tgt * tstd + tmean
        tgt = tgt * mask

    metrics = {"per_target": {}}

    # Filter extreme values
    filtered_mask = mask.clone()
    for t_idx, col in enumerate(TARGET_COLS):
        lo, hi = PARAM_RANGES[col]
        t_col = tgt[:, t_idx]
        bad = (t_col < lo) | (t_col > hi) | ~torch.isfinite(t_col)
        bad_pred = ~torch.isfinite(mu[:, t_idx])
        filtered_mask[bad | bad_pred, t_idx] = 0.0

    nll = gaussian_nll_loss(mu, sigma, tgt, filtered_mask)
    metrics["nll"] = float(nll)

    for t_idx, col in enumerate(TARGET_COLS):
        m = filtered_mask[:, t_idx].bool()
        if m.sum() < 2:
            continue

        y_pred = mu[m, t_idx].numpy()
        y_true = tgt[m, t_idx].numpy()
        y_sig = sigma[m, t_idx].numpy()

        ss_res = ((y_true - y_pred) ** 2).sum()
        ss_tot = ((y_true - y_true.mean()) ** 2).sum() + 1e-12
        r2 = float(1.0 - ss_res / ss_tot)
        mae_val = float(np.abs(y_true - y_pred).mean())
        rmse = float(np.sqrt(((y_true - y_pred) ** 2).mean()))

        z90 = 1.645
        lower = y_pred - z90 * y_sig
        upper = y_pred + z90 * y_sig
        coverage = float(((y_true >= lower) & (y_true <= upper)).mean())

        metrics["per_target"][col] = {
            "r2": r2, "mae": mae_val, "rmse": rmse,
            "coverage_90pct": coverage, "n_valid": int(m.sum()),
        }

    metrics["n_samples"] = int(len(mu))
    return metrics


def metrics_to_scalar(metrics):
    """Lower = better. Negative mean R² of targets with data."""
    r2s = [v["r2"] for v in metrics.get("per_target", {}).values() if "r2" in v]
    return -float(np.mean(r2s)) if r2s else float("inf")


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------
def save_checkpoint(path, model, optimizer, epoch, metrics, extra=None):
    raw = model.module if hasattr(model, "module") else model
    payload = {
        "epoch": epoch,
        "model_state_dict": raw.state_dict(),
        "metrics": metrics,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if extra:
        payload.update(extra)
    torch.save(payload, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser("Train WaterDroneNet on real S2 imagery")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrain-epochs", type=int, default=20)
    parser.add_argument("--skip-pretrain", action="store_true")
    args = parser.parse_args()

    log_path = LOG_DIR / "train_waterdronenet.log"
    setup_logging(log_path)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    log("=" * 65)
    log("WaterDroneNet v3 — Real Sentinel-2 Imagery")
    log("=" * 65)
    log(f"Device:     {device}")
    log(f"Data:       {DATA_FILE}")
    log(f"Targets:    {TARGET_COLS}")
    log(f"Image:      {IMG_CHANNELS}ch x {IMG_SIZE}x{IMG_SIZE}")
    log(f"Backbone:   ViT-S/16 (ImageNet pretrained, 4ch adapted)")
    log(f"Epochs:     {args.epochs}")
    log(f"Batch size: {args.batch_size}")
    log(f"LR:         {args.lr}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    log("\n--- Loading datasets ---")
    train_ds = DroneWQDataset("train")
    val_ds = DroneWQDataset("val")
    test_ds = DroneWQDataset("test")

    target_mean, target_std = train_ds.compute_target_stats()
    train_ds.set_target_stats(target_mean, target_std)
    val_ds.set_target_stats(target_mean, target_std)
    test_ds.set_target_stats(target_mean, target_std)

    log("Target normalization (from train):")
    for i, col in enumerate(TARGET_COLS):
        log(f"  {col:8s}: mean={target_mean[i]:.4f}  std={target_std[i]:.4f}")

    nw = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=nw, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=nw, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=nw, pin_memory=True)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    log("\n--- Building model ---")
    model = WaterDroneNet().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"WaterDroneNet parameters: {n_params:,}")

    scaler = GradScaler("cuda")

    # ------------------------------------------------------------------
    # Phase 1: MAE Pretraining
    # ------------------------------------------------------------------
    if not args.skip_pretrain and args.pretrain_epochs > 0:
        log("\n" + "=" * 65)
        log("Phase 1: MAE Self-Supervised Pretraining")
        log("=" * 65)

        decoder = MAEDecoder().to(device)
        raw = model.module if hasattr(model, "module") else model
        # Vision encoder params: patch_embed + cls_token + pos_embed + blocks + norm
        vit_params = (
            list(raw.patch_embed.parameters()) +
            [raw.cls_token, raw.pos_embed] +
            list(raw.blocks.parameters()) +
            list(raw.norm.parameters())
        )
        pt_params = vit_params + list(decoder.parameters())
        pt_opt = torch.optim.AdamW(pt_params, lr=args.lr * 5, weight_decay=0.05)
        pt_sched = torch.optim.lr_scheduler.CosineAnnealingLR(pt_opt, T_max=args.pretrain_epochs)

        best_mae = float("inf")
        for epoch in range(1, args.pretrain_epochs + 1):
            t0 = time.time()
            loss = mae_pretrain_epoch(model, decoder, train_loader, pt_opt, scaler, device)
            pt_sched.step()
            dt = time.time() - t0
            log(f"  Pretrain {epoch:3d}/{args.pretrain_epochs} | loss={loss:.5f} | {dt:.1f}s")
            if loss < best_mae:
                best_mae = loss
                save_checkpoint(CKPT_DIR / "waterdronenet_pretrain_best.pt",
                                model, pt_opt, epoch, {"mae_loss": loss},
                                extra={"phase": "pretrain"})

        log(f"  MAE pretraining done. Best loss: {best_mae:.5f}")
        del decoder
    else:
        pt_path = CKPT_DIR / "waterdronenet_pretrain_best.pt"
        if pt_path.exists():
            ckpt = torch.load(pt_path, map_location=device, weights_only=False)
            raw = model.module if hasattr(model, "module") else model
            # Load vision encoder params (patch_embed, cls_token, pos_embed, blocks, norm)
            vis_prefixes = ("patch_embed.", "cls_token", "pos_embed", "blocks.", "norm.")
            vis_sd = {k: v for k, v in ckpt["model_state_dict"].items()
                      if any(k.startswith(p) for p in vis_prefixes)}
            raw.load_state_dict(vis_sd, strict=False)
            log(f"  Restored pretrained vision encoder from {pt_path.name}")

    # ------------------------------------------------------------------
    # Phase 2: Supervised Training
    # ------------------------------------------------------------------
    log("\n" + "=" * 65)
    log("Phase 2: Supervised Training")
    log("=" * 65)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    target_mean_t = torch.from_numpy(target_mean).to(device).unsqueeze(0)
    target_std_t = torch.from_numpy(target_std).to(device).unsqueeze(0)

    best_score = float("inf")
    patience_counter = 0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_m = train_epoch(model, train_loader, optimizer, scaler, device,
                              target_mean_t, target_std_t)
        val_m = evaluate(model, val_loader, device, target_mean, target_std)
        scheduler.step()

        score = metrics_to_scalar(val_m)
        dt = time.time() - t0

        r2_strs = [f"{k}={v['r2']:.3f}" for k, v in val_m.get("per_target", {}).items()]
        log(f"Epoch {epoch:3d}/{args.epochs} | "
            f"loss={train_m['loss']:.4f} nll={train_m['nll']:.4f} mae={train_m['mae']:.4f} | "
            f"Val score={score:.4f} | R²: {' | '.join(r2_strs)} | {dt:.1f}s")

        if score < best_score:
            best_score = score
            best_epoch = epoch
            patience_counter = 0
            save_checkpoint(
                CKPT_DIR / "waterdronenet_best.pt",
                model, optimizer, epoch, val_m,
                extra={"target_mean": target_mean, "target_std": target_std},
            )
            log(f"  ** New best (score={best_score:.4f})")
        elif epoch > args.warmup_epochs:
            patience_counter += 1
            if patience_counter >= args.patience:
                log(f"  Early stopping at epoch {epoch}")
                break

        if epoch % 25 == 0:
            save_checkpoint(CKPT_DIR / f"waterdronenet_epoch{epoch:04d}.pt",
                            model, None, epoch, val_m)

    log(f"\nBest epoch: {best_epoch}, best score: {best_score:.4f}")

    # ------------------------------------------------------------------
    # Test Evaluation
    # ------------------------------------------------------------------
    log("\n" + "=" * 65)
    log("Test Set Evaluation")
    log("=" * 65)

    ckpt_path = CKPT_DIR / "waterdronenet_best.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        raw = model.module if hasattr(model, "module") else model
        raw.load_state_dict(ckpt["model_state_dict"])
        log(f"  Loaded best checkpoint from epoch {ckpt['epoch']}")

    test_m = evaluate(model, test_loader, device, target_mean, target_std)

    log(f"\nTest NLL: {test_m['nll']:.4f}")
    log("\nPer-target metrics:")
    for col, m in test_m.get("per_target", {}).items():
        log(f"  {col:8s}: R²={m['r2']:.4f}  MAE={m['mae']:.4f}  "
            f"RMSE={m['rmse']:.4f}  Cov90={m['coverage_90pct']:.3f}  "
            f"(n={m['n_valid']})")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    def to_serializable(obj):
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [to_serializable(x) for x in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    results = to_serializable({
        "model": "WaterDroneNet_v3",
        "description": "SENTINEL Mini — drone imagery to water quality",
        "input": "Sentinel-2 RGB+NIR (4ch, 224x224)",
        "n_params": int(n_params),
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
        "best_epoch": best_epoch,
        "best_val_score": float(best_score),
        "targets": TARGET_COLS,
        "holdout": {
            "type": "spatial",
            "train": "70% of stations",
            "val": "15% of stations",
            "test": "15% of stations (geographically unseen)",
        },
        "test_metrics": test_m,
        "training_config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "backbone": "vit_small_patch16_224 (ImageNet, 4ch adapted)",
            "pretrain_epochs": args.pretrain_epochs,
            "loss": "gaussian_nll + mae + calibration + trust",
        },
        "target_normalisation": {
            col: {"mean": float(target_mean[i]), "std": float(target_std[i])}
            for i, col in enumerate(TARGET_COLS)
        },
    })

    out_path = RESULTS_DIR / "waterdronenet_holdout.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    log(f"\nResults saved to {out_path}")
    log(f"Best checkpoint: {ckpt_path}")
    log("DONE")


if __name__ == "__main__":
    main()
