#!/usr/bin/env python3
"""Train WaterDroneNet+pH — imagery + pH strip measurement → water quality.

Same as train_waterdronenet.py but with pH as an INPUT feature (from strip)
rather than a predicted target. Predicts 4 remaining parameters:
  DO, Turb, Temp, SpCond

pH is injected into the prediction head via concatenation with the CLS token.
The ViT backbone processes imagery identically — we just give it pH as
additional context for predicting the other parameters.

This simulates the operational setup: drone photographs water with RGB+NIR
camera AND dips a pH strip, photographs it. The pH reading feeds into the
model alongside the imagery to improve DO/Turb/Temp/SpCond predictions.

MIT License — Bryan Cheng, 2026
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

CKPT_DIR = PROJECT / "checkpoints" / "waterdronenet"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = PROJECT / "results" / "benchmarks"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = PROJECT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILE = PROJECT / "data" / "processed" / "satellite" / "drone_wq_pairs.npz"

# pH is now INPUT, predict the remaining 4
INPUT_PH_IDX = 1            # index of pH in the original 5-target array
ALL_COLS = ["DO", "pH", "Turb", "Temp", "SpCond"]
TARGET_COLS = ["DO", "Turb", "Temp", "SpCond"]  # pH removed from targets
TARGET_INDICES = [0, 2, 3, 4]   # indices into original 5-col array
NUM_TARGETS = len(TARGET_COLS)   # 4

IMG_CHANNELS = 4
IMG_SIZE = 224
VIT_PATCH = 16
EMBED_DIM = 384

PARAM_RANGES = {
    "DO":     (0.0, 20.0),
    "pH":     (4.0, 10.0),
    "Turb":   (0.0, 1000.0),
    "Temp":   (-5.0, 45.0),
    "SpCond": (0.0, 50000.0),
}

PH_STRIP_NOISE_STD = 0.3  # simulate strip measurement noise


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class DroneWQDatasetWithPH(Dataset):
    """Same data as DroneWQDataset but pH is an input feature, not a target.

    Returns:
        image: (4, 224, 224)
        ph_input: (1,) — simulated pH strip reading (true pH + noise)
        target: (4,) — [DO, Turb, Temp, SpCond] z-scored
        valid_mask: (4,) — per-target validity
        ph_valid: (1,) — whether pH measurement is valid
    """

    def __init__(self, split="train", ph_noise_std=PH_STRIP_NOISE_STD):
        super().__init__()
        self.split = split
        self.ph_noise_std = ph_noise_std

        data = np.load(DATA_FILE, allow_pickle=True)
        all_images = data["images"]
        all_targets = data["targets"]  # (N, 5) — [DO, pH, Turb, Temp, SpCond]
        meta_list = json.loads(str(data["metadata"]))

        # Spatial holdout split (same as original)
        site_ids = sorted({m.get("site_id", str(i)) for i, m in enumerate(meta_list)})
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

        self.images = all_images[indices].astype(np.float32)
        all_tgts = all_targets[indices].astype(np.float32)

        # QA/QC filter
        for j, col in enumerate(ALL_COLS):
            lo, hi = PARAM_RANGES[col]
            bad = (all_tgts[:, j] < lo) | (all_tgts[:, j] > hi)
            all_tgts[bad, j] = np.nan

        # Split pH (input) from targets
        self.ph_values = all_tgts[:, INPUT_PH_IDX].copy()  # (N,)
        self.targets = all_tgts[:, TARGET_INDICES].copy()    # (N, 4)

        log(f"  {split}: {len(self.images)} samples, "
            f"pH valid: {np.isfinite(self.ph_values).sum()}")

    def compute_target_stats(self):
        mean = np.zeros(NUM_TARGETS, dtype=np.float32)
        std = np.ones(NUM_TARGETS, dtype=np.float32)
        for t in range(NUM_TARGETS):
            vals = self.targets[:, t]
            valid = vals[np.isfinite(vals)]
            if len(valid) > 1:
                mean[t] = valid.mean()
                std[t] = valid.std().clip(min=1e-6)
        return mean, std

    def compute_ph_stats(self):
        valid = self.ph_values[np.isfinite(self.ph_values)]
        return float(valid.mean()), float(valid.std().clip(min=1e-6))

    def set_stats(self, target_mean, target_std, ph_mean, ph_std):
        self._target_mean = target_mean
        self._target_std = target_std
        self._ph_mean = ph_mean
        self._ph_std = ph_std

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.from_numpy(self.images[idx])
        target = torch.from_numpy(self.targets[idx].copy())
        ph = self.ph_values[idx]

        valid_mask = (~torch.isnan(target)).float()
        target = torch.nan_to_num(target, nan=0.0)

        # pH input: simulate strip noise during training
        ph_valid = 1.0 if np.isfinite(ph) else 0.0
        if np.isfinite(ph):
            if self.split == "train":
                # Add strip measurement noise during training
                ph_noisy = ph + np.random.normal(0, self.ph_noise_std)
            else:
                # No noise at eval (strip gives what it gives)
                ph_noisy = ph
            # Z-score normalize pH
            ph_z = (ph_noisy - self._ph_mean) / self._ph_std
        else:
            ph_z = 0.0

        # Z-score normalize targets
        if hasattr(self, "_target_mean"):
            tmean = torch.from_numpy(self._target_mean)
            tstd = torch.from_numpy(self._target_std)
            target = (target - tmean) / tstd
            target = target * valid_mask

        return {
            "image": image,
            "ph_input": torch.tensor([ph_z], dtype=torch.float32),
            "ph_valid": torch.tensor([ph_valid], dtype=torch.float32),
            "target": target,
            "valid_mask": valid_mask,
        }


# ---------------------------------------------------------------------------
# ViT blocks (same as original)
# ---------------------------------------------------------------------------
class _PatchEmbed(nn.Module):
    def __init__(self, img_size=IMG_SIZE, patch_size=VIT_PATCH,
                 in_chans=IMG_CHANNELS, embed_dim=EMBED_DIM):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class _TransformerBlock(nn.Module):
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
# Model: WaterDroneNet + pH
# ---------------------------------------------------------------------------
class WaterDroneNetPH(nn.Module):
    """WaterDroneNet with pH strip input.

    Same ViT backbone, but pH measurement is injected into the prediction
    head via concatenation with the CLS token features.

    Input: 4-ch imagery (B, 4, 224, 224) + pH scalar (B, 1)
    Output: 4 predictions (DO, Turb, Temp, SpCond) with uncertainty
    """

    def __init__(self, depth=12, num_heads=6):
        super().__init__()

        # ViT backbone (identical to original)
        self.patch_embed = _PatchEmbed()
        self.num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, EMBED_DIM))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, EMBED_DIM))
        self.blocks = nn.ModuleList([
            _TransformerBlock(EMBED_DIM, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(EMBED_DIM)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # pH scalar encoder: 1 → 32 dim embedding
        self.ph_encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),
            nn.Linear(32, 32),
        )

        # Prediction head: CLS (384) + pH embed (32) = 416 → predictions
        HEAD_IN = EMBED_DIM + 32  # 416
        self.head = nn.Sequential(
            nn.LayerNorm(HEAD_IN),
            nn.Linear(HEAD_IN, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
        )

        self.mu_heads = nn.ModuleList([nn.Linear(128, 1) for _ in range(NUM_TARGETS)])
        self.sigma_heads = nn.ModuleList([nn.Linear(128, 1) for _ in range(NUM_TARGETS)])

        # Physics prior: band stats (10) + pH (1) = 11 features
        self.prior = nn.Linear(11, NUM_TARGETS, bias=True)
        nn.init.zeros_(self.prior.weight)
        nn.init.zeros_(self.prior.bias)

        # Trust router
        self.trust_router = nn.Sequential(
            nn.Linear(EMBED_DIM + NUM_TARGETS, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def _encode_vision(self, x):
        B = x.size(0)
        patches = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, patches], dim=1)
        tokens = tokens + self.pos_embed
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)
        return tokens[:, 0], tokens[:, 1:]

    def _image_features(self, images):
        means = images.mean(dim=(-2, -1))
        stds = images.std(dim=(-2, -1))
        green, nir, red = means[:, 1], means[:, 3], means[:, 2]
        ndwi = (green - nir) / (green + nir + 1e-6)
        nir_red = nir / (red + 1e-6)
        return torch.cat([means, stds, ndwi.unsqueeze(1), nir_red.unsqueeze(1)], dim=1)

    def forward(self, image, ph_input, ph_valid=None):
        """
        Args:
            image: (B, 4, 224, 224)
            ph_input: (B, 1) — z-scored pH measurement
            ph_valid: (B, 1) — mask for valid pH (optional)
        """
        cls_feat, _ = self._encode_vision(image)  # (B, 384)

        # Encode pH
        ph_embed = self.ph_encoder(ph_input)  # (B, 32)
        if ph_valid is not None:
            ph_embed = ph_embed * ph_valid  # zero out if pH invalid

        # Concatenate CLS + pH for prediction
        combined = torch.cat([cls_feat, ph_embed], dim=1)  # (B, 416)

        h = self.head(combined)  # (B, 128)
        mu_residual = torch.cat([head(h) for head in self.mu_heads], dim=1)
        log_sigma = torch.cat([head(h) for head in self.sigma_heads], dim=1)
        sigma = F.softplus(log_sigma) + 1e-4

        # Physics prior: band stats + raw pH
        img_feats = self._image_features(image)  # (B, 10)
        prior_input = torch.cat([img_feats, ph_input], dim=1)  # (B, 11)
        physics_prior = self.prior(prior_input)
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
# Loss functions (same as original)
# ---------------------------------------------------------------------------
def gaussian_nll_loss(mu, sigma, target, mask):
    mu, sigma, target, mask = mu.float(), sigma.float().clamp(min=1e-4), target.float(), mask.float()
    nll = 0.5 * (torch.log(sigma ** 2) + (target - mu) ** 2 / sigma ** 2)
    nll = torch.nan_to_num(nll, nan=0.0, posinf=0.0, neginf=0.0)
    return (nll * mask).sum() / mask.sum().clamp(min=1.0)


def masked_mae_loss(mu, target, mask):
    ae = torch.abs(target.float() - mu.float())
    ae = torch.nan_to_num(ae, nan=0.0)
    return (ae * mask.float()).sum() / mask.float().sum().clamp(min=1.0)


def calibration_loss(mu, sigma, target, mask):
    z_score = torch.abs(target.float() - mu.float()) / sigma.float().clamp(min=1e-4)
    overconf = F.relu(z_score - 2.0)
    return (overconf * mask.float()).sum() / mask.float().sum().clamp(min=1.0)


def trust_supervision_loss(trust_logit, mu, target, mask, threshold=2.0):
    norm_err = torch.abs(target.float() - mu.float().detach())
    per_sample = (norm_err * mask.float()).sum(dim=1) / mask.float().sum(dim=1).clamp(min=1.0)
    label = (per_sample <= threshold).float().unsqueeze(1)
    return F.binary_cross_entropy_with_logits(trust_logit.float(), label)


# ---------------------------------------------------------------------------
# Training & Evaluation
# ---------------------------------------------------------------------------
def train_epoch(model, loader, optimizer, scaler, device):
    model.train()
    totals = {"loss": 0., "nll": 0., "mae": 0.}
    n = 0

    for batch in loader:
        images = batch["image"].to(device)
        ph = batch["ph_input"].to(device)
        ph_valid = batch["ph_valid"].to(device)
        targets = batch["target"].to(device)
        mask = batch["valid_mask"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda"):
            out = model(images, ph, ph_valid)

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
        n += B

    return {k: v / max(n, 1) for k, v in totals.items()}


@torch.no_grad()
def evaluate(model, loader, device, target_mean=None, target_std=None):
    model.eval()
    all_mu, all_sigma, all_tgt, all_mask = [], [], [], []

    for batch in loader:
        images = batch["image"].to(device)
        ph = batch["ph_input"].to(device)
        ph_valid = batch["ph_valid"].to(device)
        targets = batch["target"].to(device)
        mask = batch["valid_mask"].to(device)

        with autocast("cuda"):
            out = model(images, ph, ph_valid)

        all_mu.append(out["mu"].cpu().float())
        all_sigma.append(out["sigma"].cpu().float())
        all_tgt.append(targets.cpu().float())
        all_mask.append(mask.cpu().float())

    mu = torch.cat(all_mu)
    sigma = torch.cat(all_sigma)
    tgt = torch.cat(all_tgt)
    mask = torch.cat(all_mask)

    # Denormalize
    if target_mean is not None and target_std is not None:
        tmean = torch.from_numpy(target_mean).unsqueeze(0)
        tstd = torch.from_numpy(target_std).unsqueeze(0)
        mu = mu * tstd + tmean
        sigma = sigma * tstd.abs()
        tgt = tgt * tstd + tmean
        tgt = tgt * mask

    metrics = {"per_target": {}}

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

        # Pearson r
        pearson = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 2 else 0.0

        z90 = 1.645
        lower = y_pred - z90 * y_sig
        upper = y_pred + z90 * y_sig
        coverage = float(((y_true >= lower) & (y_true <= upper)).mean())

        metrics["per_target"][col] = {
            "r2": r2, "mae": mae_val, "rmse": rmse,
            "pearson_r": pearson,
            "coverage_90pct": coverage, "n_valid": int(m.sum()),
        }

    metrics["n_samples"] = int(len(mu))
    return metrics


def metrics_to_scalar(metrics):
    r2s = [v["r2"] for v in metrics.get("per_target", {}).values() if "r2" in v]
    return -float(np.mean(r2s)) if r2s else float("inf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser("Train WaterDroneNet+pH")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrain-weights", action="store_true",
                        help="Init ViT from imagery-only pretrained checkpoint")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    log("=" * 65)
    log("WaterDroneNet+pH — Imagery + pH Strip → Water Quality")
    log("=" * 65)
    log(f"Device:     {device}")
    log(f"Input:      4ch imagery + pH strip measurement")
    log(f"Targets:    {TARGET_COLS} (pH is INPUT, not target)")
    log(f"Strip noise: ±{PH_STRIP_NOISE_STD} pH units (during training)")

    # Data
    log("\n--- Loading datasets ---")
    train_ds = DroneWQDatasetWithPH("train")
    val_ds = DroneWQDatasetWithPH("val", ph_noise_std=0.0)  # no noise at eval
    test_ds = DroneWQDatasetWithPH("test", ph_noise_std=0.0)

    target_mean, target_std = train_ds.compute_target_stats()
    ph_mean, ph_std = train_ds.compute_ph_stats()
    log(f"pH stats: mean={ph_mean:.3f}, std={ph_std:.3f}")

    train_ds.set_stats(target_mean, target_std, ph_mean, ph_std)
    val_ds.set_stats(target_mean, target_std, ph_mean, ph_std)
    test_ds.set_stats(target_mean, target_std, ph_mean, ph_std)

    log("Target normalization:")
    for i, col in enumerate(TARGET_COLS):
        log(f"  {col:8s}: mean={target_mean[i]:.4f}  std={target_std[i]:.4f}")

    nw = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=nw, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=nw, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=nw, pin_memory=True)

    # Model
    log("\n--- Building model ---")
    model = WaterDroneNetPH().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"WaterDroneNet+pH parameters: {n_params:,}")

    # Optionally init ViT from imagery-only pretrained weights
    pretrain_path = CKPT_DIR / "waterdronenet_pretrain_best.pt"
    if args.pretrain_weights and pretrain_path.exists():
        ckpt = torch.load(pretrain_path, map_location=device, weights_only=False)
        vis_prefixes = ("patch_embed.", "cls_token", "pos_embed", "blocks.", "norm.")
        vis_sd = {k: v for k, v in ckpt["model_state_dict"].items()
                  if any(k.startswith(p) for p in vis_prefixes)}
        model.load_state_dict(vis_sd, strict=False)
        log(f"  Loaded pretrained ViT weights from {pretrain_path.name}")
    elif pretrain_path.exists():
        # Always try to load pretrained ViT
        ckpt = torch.load(pretrain_path, map_location=device, weights_only=False)
        vis_prefixes = ("patch_embed.", "cls_token", "pos_embed", "blocks.", "norm.")
        vis_sd = {k: v for k, v in ckpt["model_state_dict"].items()
                  if any(k.startswith(p) for p in vis_prefixes)}
        model.load_state_dict(vis_sd, strict=False)
        log(f"  Loaded pretrained ViT weights from {pretrain_path.name}")

    scaler = GradScaler("cuda")

    # Training
    log("\n" + "=" * 65)
    log("Training: Imagery + pH → DO, Turb, Temp, SpCond")
    log("=" * 65)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_score = float("inf")
    patience_counter = 0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_m = train_epoch(model, train_loader, optimizer, scaler, device)
        val_m = evaluate(model, val_loader, device, target_mean, target_std)
        scheduler.step()

        score = metrics_to_scalar(val_m)
        dt = time.time() - t0

        r2_strs = [f"{k}={v['r2']:.3f}" for k, v in val_m.get("per_target", {}).items()]
        log(f"Epoch {epoch:3d}/{args.epochs} | "
            f"loss={train_m['loss']:.4f} nll={train_m['nll']:.4f} mae={train_m['mae']:.4f} | "
            f"Val R²: {' | '.join(r2_strs)} | {dt:.1f}s")

        if score < best_score:
            best_score = score
            best_epoch = epoch
            patience_counter = 0

            raw = model.module if hasattr(model, "module") else model
            torch.save({
                "epoch": epoch,
                "model_state_dict": raw.state_dict(),
                "metrics": val_m,
                "target_mean": target_mean,
                "target_std": target_std,
                "ph_mean": ph_mean,
                "ph_std": ph_std,
            }, CKPT_DIR / "waterdronenet_ph_best.pt")
            log(f"  ** New best (score={best_score:.4f})")
        elif epoch > args.warmup_epochs:
            patience_counter += 1
            if patience_counter >= args.patience:
                log(f"  Early stopping at epoch {epoch}")
                break

    log(f"\nBest epoch: {best_epoch}, best score: {best_score:.4f}")

    # Test
    log("\n" + "=" * 65)
    log("Test Set Evaluation")
    log("=" * 65)

    ckpt = torch.load(CKPT_DIR / "waterdronenet_ph_best.pt",
                       map_location=device, weights_only=False)
    raw = model.module if hasattr(model, "module") else model
    raw.load_state_dict(ckpt["model_state_dict"])
    log(f"  Loaded best checkpoint from epoch {ckpt['epoch']}")

    test_m = evaluate(model, test_loader, device, target_mean, target_std)

    log(f"\nTest NLL: {test_m['nll']:.4f}")
    log(f"\n{'Param':<8} {'R²':>8} {'Pearson r':>10} {'MAE':>10} {'RMSE':>10} {'Cov90':>8} {'n':>6}")
    log(f"{'─'*8} {'─'*8} {'─'*10} {'─'*10} {'─'*10} {'─'*8} {'─'*6}")

    for col, m in test_m.get("per_target", {}).items():
        log(f"{col:<8} {m['r2']:>8.4f} {m['pearson_r']:>10.4f} {m['mae']:>10.4f} "
            f"{m['rmse']:>10.4f} {m['coverage_90pct']:>8.3f} {m['n_valid']:>6}")

    # Load original results for comparison
    orig_path = RESULTS_DIR / "waterdronenet_holdout.json"
    if orig_path.exists():
        with open(orig_path) as f:
            orig = json.load(f)

        log(f"\n{'─' * 70}")
        log("COMPARISON: Imagery-Only vs Imagery + pH Strip")
        log(f"{'─' * 70}")
        log(f"\n{'Param':<8} {'Img R²':>8} {'pH R²':>8} {'Change':>8} "
            f"{'Img MAE':>10} {'pH MAE':>10} {'Change':>10}")
        log(f"{'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*10} {'─'*10} {'─'*10}")

        for col in TARGET_COLS:
            if col in orig["test_metrics"]["per_target"] and col in test_m["per_target"]:
                o = orig["test_metrics"]["per_target"][col]
                n = test_m["per_target"][col]
                r2_diff = n["r2"] - o["r2"]
                mae_diff = n["mae"] - o["mae"]
                log(f"{col:<8} {o['r2']:>8.3f} {n['r2']:>8.3f} {r2_diff:>+8.3f} "
                    f"{o['mae']:>10.3f} {n['mae']:>10.3f} {mae_diff:>+10.3f}")

    # Save results
    def to_ser(obj):
        if isinstance(obj, dict):
            return {k: to_ser(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [to_ser(x) for x in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    results = to_ser({
        "model": "WaterDroneNet+pH",
        "description": "SENTINEL Mini — imagery + pH strip → water quality",
        "input": "Sentinel-2 RGB+NIR (4ch, 224x224) + pH strip measurement",
        "n_params": int(n_params),
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
        "best_epoch": best_epoch,
        "best_val_score": float(best_score),
        "targets": TARGET_COLS,
        "ph_input": {"mean": ph_mean, "std": ph_std, "noise_std": PH_STRIP_NOISE_STD},
        "holdout": {
            "type": "spatial",
            "train": "70% of stations",
            "val": "15% of stations",
            "test": "15% of stations (geographically unseen)",
        },
        "test_metrics": test_m,
    })

    out_path = RESULTS_DIR / "waterdronenet_ph_holdout.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    log(f"\nResults saved to {out_path}")
    log("DONE")


if __name__ == "__main__":
    main()
