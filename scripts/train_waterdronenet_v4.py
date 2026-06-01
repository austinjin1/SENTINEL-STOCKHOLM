#!/usr/bin/env python3
"""Train WaterDroneNet V4 — HydroDenseNet + pH strip → water quality.

Custom architecture built ON TOP OF DenseNet121, not just using it as backbone:

  1. Spectral Attention Stem — learnable 4-band interactions before DenseNet
  2. Multi-Scale Feature Pyramid — taps all 4 DenseBlock outputs (not just final)
  3. CBAM attention at each scale — channel + spatial attention
  4. FiLM conditioning — pH modulates visual features INSIDE the network
  5. Multi-pool aggregation — avg + max + std captures spatial heterogeneity
  6. Per-target expert decoders — each WQ parameter gets its own specialist
  7. Spectral-spatial cross-attention — band ratios attend to spatial features

Key insight: vanilla DenseNet121 throws away spatial info via global avg pool
and treats all bands equally. Water quality prediction needs multi-scale spatial
reasoning (turbidity plumes are large, algal texture is fine) and spectral
band interactions (NDWI, NIR/Red) are the physics, not just features.

Previous results:
  ViT-S/16:          DO R²=0.26, Temp R²=0.51
  ViT-S/16 + pH:     DO R²=0.35, Temp R²=0.53
  DenseNet121 naive:  (untrained — this script replaces it)

MIT License — Bryan Cheng, 2026
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# GPU setup — respect CUDA_VISIBLE_DEVICES if already set, default to GPU 2
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
import torchvision.models as tv_models

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

CKPT_DIR = PROJECT / "checkpoints" / "waterdronenet"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = PROJECT / "results" / "benchmarks"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = PROJECT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Data paths
TILE_DIR = PROJECT / "data" / "processed" / "satellite" / "drone_tiles"
SENSOR_DIR = PROJECT / "data" / "raw" / "sensor" / "full"
SITE_INFO_DIR = PROJECT / "data" / "raw" / "hydrology" / "nhdplus" / "cache" / "site_info"

# pH is INPUT, predict the remaining 4
ALL_COLS = ["DO", "pH", "Turb", "Temp", "SpCond"]
TARGET_COLS = ["DO", "Turb", "Temp", "SpCond"]
NUM_TARGETS = len(TARGET_COLS)  # 4

IMG_CHANNELS = 4
IMG_SIZE = 224
DENSENET_FEAT_DIM = 1024  # DenseNet121 output features

PARAM_RANGES = {
    "DO":     (0.0, 20.0),
    "pH":     (4.0, 10.0),
    "Turb":   (0.0, 1000.0),
    "Temp":   (-5.0, 45.0),
    "SpCond": (0.0, 50000.0),
}

PH_STRIP_NOISE_STD = 0.3  # simulate strip measurement noise
DATE_WINDOW_DAYS = 3       # ±3 day window for matching WQ data

# Number of extended band ratio features
NUM_BAND_FEATURES = 15


def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Dataset: Lazy tile-based loading from individual .npz files
# ---------------------------------------------------------------------------
class TileDataset(Dataset):
    """Lazy-loading dataset from individual .npz tile files.

    Scans data/processed/satellite/drone_tiles/*.npz for all tiles.
    Parses filename: {station_id}_{YYYY-MM-DD}.npz
    Loads WQ from parquet for that station/date (daily mean, ±3 day window).
    pH becomes input feature, [DO, Turb, Temp, SpCond] are targets.
    Spatial holdout split: 70/15/15 by station (seed=42).

    Returns:
        image: (4, 224, 224)
        ph_input: (1,) — z-scored pH strip reading
        ph_valid: (1,) — whether pH measurement is valid
        target: (4,) — [DO, Turb, Temp, SpCond] z-scored
        valid_mask: (4,) — per-target validity
    """

    def __init__(self, split: str = "train", ph_noise_std: float = PH_STRIP_NOISE_STD,
                 augment: bool = False):
        super().__init__()
        self.split = split
        self.ph_noise_std = ph_noise_std
        self.augment = augment and (split == "train")

        # Discover all tile files
        all_tiles = sorted(TILE_DIR.glob("*.npz"))
        if not all_tiles:
            raise FileNotFoundError(f"No .npz tiles found in {TILE_DIR}")

        # Parse filenames → (station_id, date_str, path)
        tile_info: List[Tuple[str, str, Path]] = []
        for p in all_tiles:
            stem = p.stem  # e.g., "01302020_2022-01-05"
            parts = stem.rsplit("_", 1)
            if len(parts) == 2:
                station_id, date_str = parts
                tile_info.append((station_id, date_str, p))

        # Spatial holdout split by station
        all_stations = sorted({t[0] for t in tile_info})
        rng = np.random.RandomState(42)
        rng.shuffle(all_stations)
        n = len(all_stations)
        train_stations = set(all_stations[:int(0.7 * n)])
        val_stations = set(all_stations[int(0.7 * n):int(0.85 * n)])
        test_stations = set(all_stations[int(0.85 * n):])

        if split == "train":
            split_stations = train_stations
        elif split == "val":
            split_stations = val_stations
        elif split == "test":
            split_stations = test_stations
        else:
            raise ValueError(f"Unknown split: {split}")

        # Filter tiles for this split
        self.tiles = [(sid, ds, p) for sid, ds, p in tile_info if sid in split_stations]
        self.station_ids = split_stations

        # Pre-load WQ data for all stations in this split (parquet → dict)
        self._wq_cache: Dict[str, pd.DataFrame] = {}
        self._load_wq_data()

        # Match tiles to WQ data
        self.samples: List[Tuple[Path, np.ndarray, float]] = []
        self._match_tiles_to_wq()

        log(f"  {split}: {len(self.samples)} samples from "
            f"{len(split_stations)} stations, "
            f"pH valid: {sum(1 for _, _, ph in self.samples if np.isfinite(ph))}")

    def _load_wq_data(self):
        """Load parquet WQ data for all stations in the split."""
        for sid in self.station_ids:
            parquet_path = SENSOR_DIR / f"{sid}.parquet"
            if parquet_path.exists():
                try:
                    df = pd.read_parquet(parquet_path)
                    # Ensure datetime index
                    if not isinstance(df.index, pd.DatetimeIndex):
                        df.index = pd.to_datetime(df.index)
                    # Localize to UTC if needed, then drop tz for matching
                    if df.index.tz is not None:
                        df.index = df.index.tz_localize(None)
                    self._wq_cache[sid] = df
                except Exception as e:
                    log(f"    Warning: Failed to load {parquet_path}: {e}")

    def _match_tiles_to_wq(self):
        """Match each tile to WQ data within ±DATE_WINDOW_DAYS."""
        for station_id, date_str, tile_path in self.tiles:
            if station_id not in self._wq_cache:
                continue

            df = self._wq_cache[station_id]
            try:
                target_date = pd.Timestamp(date_str)
            except Exception:
                continue

            # Get data within ±3 day window
            window_start = target_date - timedelta(days=DATE_WINDOW_DAYS)
            window_end = target_date + timedelta(days=DATE_WINDOW_DAYS + 1)
            window = df[(df.index >= window_start) & (df.index < window_end)]

            if len(window) == 0:
                continue

            # Daily mean of WQ parameters
            target_vals = np.full(NUM_TARGETS, np.nan, dtype=np.float32)
            ph_val = np.nan

            for i, col in enumerate(TARGET_COLS):
                if col in window.columns:
                    vals = window[col].dropna()
                    if len(vals) > 0:
                        mean_val = float(vals.mean())
                        lo, hi = PARAM_RANGES[col]
                        if lo <= mean_val <= hi:
                            target_vals[i] = mean_val

            if "pH" in window.columns:
                ph_series = window["pH"].dropna()
                if len(ph_series) > 0:
                    ph_val = float(ph_series.mean())
                    lo, hi = PARAM_RANGES["pH"]
                    if not (lo <= ph_val <= hi):
                        ph_val = np.nan

            # Only keep samples with at least one valid target
            if np.all(np.isnan(target_vals)):
                continue

            self.samples.append((tile_path, target_vals, ph_val))

    def compute_target_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute mean/std for each target from training data."""
        all_targets = np.array([s[1] for s in self.samples], dtype=np.float32)
        mean = np.zeros(NUM_TARGETS, dtype=np.float32)
        std = np.ones(NUM_TARGETS, dtype=np.float32)
        for t in range(NUM_TARGETS):
            vals = all_targets[:, t]
            valid = vals[np.isfinite(vals)]
            if len(valid) > 1:
                mean[t] = valid.mean()
                std[t] = max(valid.std(), 1e-6)
        return mean, std

    def compute_ph_stats(self) -> Tuple[float, float]:
        """Compute mean/std for pH from training data."""
        ph_vals = np.array([s[2] for s in self.samples], dtype=np.float32)
        valid = ph_vals[np.isfinite(ph_vals)]
        if len(valid) > 1:
            return float(valid.mean()), float(max(valid.std(), 1e-6))
        return 7.0, 1.0

    def set_stats(self, target_mean: np.ndarray, target_std: np.ndarray,
                  ph_mean: float, ph_std: float):
        self._target_mean = target_mean
        self._target_std = target_std
        self._ph_mean = ph_mean
        self._ph_std = ph_std

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        tile_path, target_vals, ph_val = self.samples[idx]

        # Lazy load image from disk
        data = np.load(tile_path)
        image = torch.from_numpy(data["image"].astype(np.float32))  # (4, 224, 224)

        # Data augmentation (training only)
        if self.augment:
            # Random horizontal flip
            if torch.rand(1).item() < 0.5:
                image = image.flip(-1)
            # Random vertical flip
            if torch.rand(1).item() < 0.5:
                image = image.flip(-2)
            # Random 90° rotation (0, 90, 180, 270)
            k = torch.randint(0, 4, (1,)).item()
            if k > 0:
                image = torch.rot90(image, k, dims=(-2, -1))

        # Targets
        target = torch.from_numpy(target_vals.copy())
        valid_mask = (~torch.isnan(target)).float()
        target = torch.nan_to_num(target, nan=0.0)

        # pH input
        ph_valid_flag = 1.0 if np.isfinite(ph_val) else 0.0
        if np.isfinite(ph_val):
            if self.split == "train":
                ph_noisy = ph_val + np.random.normal(0, self.ph_noise_std)
            else:
                ph_noisy = ph_val
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
            "ph_valid": torch.tensor([ph_valid_flag], dtype=torch.float32),
            "target": target,
            "valid_mask": valid_mask,
        }


# ---------------------------------------------------------------------------
# CBAM: Convolutional Block Attention Module (channel + spatial)
# ---------------------------------------------------------------------------
class ChannelAttention(nn.Module):
    """Channel attention: avg pool + max pool → shared MLP → sigmoid."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) → (B, C, H, W)"""
        B, C, H, W = x.shape
        avg_out = self.mlp(x.mean(dim=(2, 3)))           # (B, C)
        max_out = self.mlp(x.amax(dim=(2, 3)))            # (B, C)
        scale = torch.sigmoid(avg_out + max_out)           # (B, C)
        return x * scale.unsqueeze(-1).unsqueeze(-1)


class SpatialAttention(nn.Module):
    """Spatial attention: channel-wise avg+max → conv → sigmoid."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=pad, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) → (B, C, H, W)"""
        avg_out = x.mean(dim=1, keepdim=True)   # (B, 1, H, W)
        max_out = x.amax(dim=1, keepdim=True)    # (B, 1, H, W)
        combined = torch.cat([avg_out, max_out], dim=1)  # (B, 2, H, W)
        scale = torch.sigmoid(self.conv(combined))        # (B, 1, H, W)
        return x * scale


class CBAM(nn.Module):
    """CBAM: sequential channel + spatial attention."""

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


# ---------------------------------------------------------------------------
# FiLM: Feature-wise Linear Modulation (pH conditions visual features)
# ---------------------------------------------------------------------------
class FiLMLayer(nn.Module):
    """Generate per-channel affine params (gamma, beta) from a conditioning vector.

    Instead of concatenating pH AFTER the backbone, FiLM lets pH
    modulate visual feature extraction INSIDE the network.
    """

    def __init__(self, cond_dim: int, channels: int):
        super().__init__()
        self.fc = nn.Linear(cond_dim, channels * 2)
        # Init near identity: gamma≈1, beta≈0
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        with torch.no_grad():
            self.fc.bias[:channels] = 1.0  # gamma = 1

    def forward(self, feat_map: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        feat_map: (B, C, H, W)
        cond: (B, cond_dim)
        Returns: (B, C, H, W) modulated feature map
        """
        params = self.fc(cond)                           # (B, 2*C)
        gamma, beta = params.chunk(2, dim=1)              # each (B, C)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)         # (B, C, 1, 1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return feat_map * gamma + beta


# ---------------------------------------------------------------------------
# Spectral Attention Stem — learnable band interactions
# ---------------------------------------------------------------------------
class SpectralStem(nn.Module):
    """Learns spectral band interactions BEFORE the DenseNet backbone.

    Instead of just replacing conv0 from 3→4 channels, we first learn
    meaningful spectral combinations (like NDWI, NIR/Red) as trainable
    convolutions, then feed the enriched representation into DenseNet.

    4 bands → 12 spectral features → 64 channels (matching DenseNet conv0 output)
    """

    def __init__(self):
        super().__init__()
        # Stage 1: Per-pixel spectral mixing (1x1 conv learns band combinations)
        self.spectral_mix = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=1, bias=False),  # learn band ratios
            nn.BatchNorm2d(16),
            nn.GELU(),
        )
        # Stage 2: Spatial context (3x3 depthwise for local texture)
        self.spatial_ctx = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1, groups=16, bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
        )
        # Stage 3: Combine original bands + learned features → DenseNet-compatible
        # 4 (original) + 16 (spectral) = 20 → 64 channels with 7x7 conv stride 2
        self.fuse = nn.Conv2d(20, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.fuse_bn = nn.BatchNorm2d(64)

    def init_from_densenet(self, old_conv_weight: torch.Tensor):
        """Transfer RGB weights from pretrained DenseNet conv0 into fuse layer."""
        with torch.no_grad():
            # old_conv_weight: (64, 3, 7, 7) — copy into first 4 input channels
            self.fuse.weight[:, :3, :, :] = old_conv_weight
            self.fuse.weight[:, 3, :, :] = old_conv_weight.mean(dim=1)  # NIR from RGB mean
            # Remaining 16 channels (spectral features) init small
            nn.init.kaiming_normal_(self.fuse.weight[:, 4:, :, :], mode='fan_out')
            self.fuse.weight[:, 4:, :, :] *= 0.1  # start small, let backbone dominate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 4, 224, 224) → (B, 64, 112, 112) matching DenseNet post-conv0"""
        spectral = self.spectral_mix(x)      # (B, 16, 224, 224)
        spectral = self.spatial_ctx(spectral) # (B, 16, 224, 224)
        combined = torch.cat([x, spectral], dim=1)  # (B, 20, 224, 224)
        out = self.fuse(combined)             # (B, 64, 112, 112)
        out = self.fuse_bn(out)
        return F.relu(out)


# ---------------------------------------------------------------------------
# Multi-Scale Feature Aggregation
# ---------------------------------------------------------------------------
class MultiScaleAggregator(nn.Module):
    """Tap DenseNet at multiple scales, apply CBAM, aggregate.

    DenseNet121 dense block output channels:
      DenseBlock1: 256 channels, 56×56
      DenseBlock2: 512 channels, 28×28
      DenseBlock3: 1024 channels, 14×14
      DenseBlock4: 1024 channels, 7×7

    We project each to a common dim, apply CBAM attention, then use
    multi-pool aggregation (avg + max + std) for each.
    """

    SCALE_DIM = 128  # project each scale to this dim

    def __init__(self, ph_cond_dim: int = 32):
        super().__init__()
        # DenseNet121 block output channels
        block_channels = [256, 512, 1024, 1024]
        n_scales = len(block_channels)

        # Per-scale: 1x1 projection + CBAM + FiLM
        self.projections = nn.ModuleList()
        self.cbams = nn.ModuleList()
        self.films = nn.ModuleList()

        for ch in block_channels:
            self.projections.append(nn.Sequential(
                nn.Conv2d(ch, self.SCALE_DIM, 1, bias=False),
                nn.BatchNorm2d(self.SCALE_DIM),
                nn.ReLU(inplace=True),
            ))
            self.cbams.append(CBAM(self.SCALE_DIM, reduction=8))
            self.films.append(FiLMLayer(ph_cond_dim, self.SCALE_DIM))

        # Multi-pool: avg + max + std = 3 vectors per scale
        # Total: n_scales * SCALE_DIM * 3
        self.total_dim = n_scales * self.SCALE_DIM * 3
        self.out_norm = nn.LayerNorm(self.total_dim)

    def forward(self, scale_features: list, ph_embed: torch.Tensor) -> torch.Tensor:
        """
        scale_features: list of 4 tensors from DenseNet dense blocks
        ph_embed: (B, 32) pH conditioning vector

        Returns: (B, total_dim) aggregated multi-scale features
        """
        parts = []
        for i, feat in enumerate(scale_features):
            # Project to common dim
            proj = self.projections[i](feat)          # (B, 128, H_i, W_i)
            # FiLM conditioning from pH
            proj = self.films[i](proj, ph_embed)      # (B, 128, H_i, W_i)
            # CBAM attention
            proj = self.cbams[i](proj)                 # (B, 128, H_i, W_i)
            # Multi-pool aggregation
            avg_pool = proj.mean(dim=(2, 3))           # (B, 128)
            max_pool = proj.amax(dim=(2, 3))           # (B, 128)
            std_pool = proj.std(dim=(2, 3))            # (B, 128)
            parts.extend([avg_pool, max_pool, std_pool])

        combined = torch.cat(parts, dim=1)  # (B, 4 * 128 * 3 = 1536)
        return self.out_norm(combined)


# ---------------------------------------------------------------------------
# Per-Target Expert Decoder
# ---------------------------------------------------------------------------
class TargetExpert(nn.Module):
    """Specialized decoder for a single water quality parameter.

    Each target gets its own MLP so it can learn which features matter.
    DO cares about different band patterns than turbidity.
    """

    def __init__(self, feat_dim: int, band_feat_dim: int):
        super().__init__()
        total_in = feat_dim + band_feat_dim  # multi-scale + band ratios
        self.net = nn.Sequential(
            nn.Linear(total_in, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
        )
        self.mu_head = nn.Linear(64, 1)
        self.sigma_head = nn.Linear(64, 1)

    def forward(self, feat: torch.Tensor, band_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        feat: (B, feat_dim) multi-scale features
        band_feat: (B, band_feat_dim) physics band ratio features

        Returns: mu (B, 1), sigma (B, 1)
        """
        h = self.net(torch.cat([feat, band_feat], dim=1))
        mu = self.mu_head(h)
        sigma = F.softplus(self.sigma_head(h)) + 1e-4
        return mu, sigma


# ---------------------------------------------------------------------------
# Band Ratio Feature Extractor (Physics Prior — with gradient flow)
# ---------------------------------------------------------------------------
class BandRatioEncoder(nn.Module):
    """Compute physics-informed band ratios and encode them.

    Unlike the old BandRatioFeatures which was disconnected from the backbone,
    this version encodes band ratios into a learned representation that feeds
    into both the physics prior AND the per-target expert decoders.

    Band ordering: [Blue, Green, Red, NIR] (channels 0-3)
    """

    RATIO_DIM = 15  # raw band ratio features
    ENCODED_DIM = 48  # encoded representation

    def __init__(self):
        super().__init__()
        # Encode raw ratios into a richer representation
        self.encoder = nn.Sequential(
            nn.Linear(self.RATIO_DIM + 1, 32),  # +1 for pH
            nn.GELU(),
            nn.Linear(32, self.ENCODED_DIM),
            nn.GELU(),
        )
        # Physics prior: direct linear from encoded → targets
        self.prior = nn.Linear(self.ENCODED_DIM, NUM_TARGETS)
        nn.init.zeros_(self.prior.weight)
        nn.init.zeros_(self.prior.bias)

    def _compute_ratios(self, images: torch.Tensor) -> torch.Tensor:
        """Compute 15 band ratio features from 4-band imagery."""
        B = images.size(0)
        eps = 1e-6
        flat = images.view(B, 4, -1)  # (B, 4, H*W)

        means = flat.mean(dim=-1)  # (B, 4)
        stds = flat.std(dim=-1)    # (B, 4)
        blue, green, red, nir = means[:, 0], means[:, 1], means[:, 2], means[:, 3]

        ndwi = (green - nir) / (green + nir + eps)
        nir_red = nir / (red + eps)
        green_blue = green / (blue + eps)
        sediment = (red - green) / (red + green + eps)
        nir_var = flat[:, 3, :].var(dim=-1)

        # Band correlations
        blue_f, nir_f = flat[:, 0, :], flat[:, 3, :]
        green_f, red_f = flat[:, 1, :], flat[:, 2, :]

        corr_bn = ((blue_f - blue_f.mean(-1, keepdim=True)) *
                   (nir_f - nir_f.mean(-1, keepdim=True))).mean(-1) / (
                   blue_f.std(-1) * nir_f.std(-1) + eps)
        corr_gr = ((green_f - green_f.mean(-1, keepdim=True)) *
                   (red_f - red_f.mean(-1, keepdim=True))).mean(-1) / (
                   green_f.std(-1) * red_f.std(-1) + eps)

        return torch.stack([
            means[:, 0], means[:, 1], means[:, 2], means[:, 3],
            stds[:, 0], stds[:, 1], stds[:, 2], stds[:, 3],
            ndwi, nir_red, green_blue, sediment, nir_var,
            corr_bn, corr_gr,
        ], dim=1)  # (B, 15)

    def forward(self, images: torch.Tensor, ph_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            encoded: (B, ENCODED_DIM) — for per-target experts
            prior: (B, NUM_TARGETS) — physics prior prediction
        """
        ratios = self._compute_ratios(images)  # (B, 15)
        inp = torch.cat([ratios, ph_input], dim=1)  # (B, 16)
        encoded = self.encoder(inp)  # (B, 48)
        prior = self.prior(encoded)  # (B, 4)
        return encoded, prior


# ---------------------------------------------------------------------------
# HydroDenseNet — purpose-built for water quality from satellite imagery
# ---------------------------------------------------------------------------
class WaterDroneNetV4(nn.Module):
    """HydroDenseNet: DenseNet121 backbone with water-quality-specific design.

    Key innovations over vanilla DenseNet121:

    1. SpectralStem: learnable band interactions (1×1 spectral mixing +
       depthwise spatial context) BEFORE the DenseNet body. Learns
       domain-relevant spectral combinations like NDWI and NIR/Red ratios
       as trainable convolutions instead of hardcoded.

    2. Multi-Scale Feature Pyramid: taps all 4 DenseBlock outputs
       (256ch@56×56, 512ch@28×28, 1024ch@14×14, 1024ch@7×7) instead
       of just the final 1024-dim vector. Water quality signals exist
       at multiple scales — turbidity plumes are large-scale, algal
       texture is fine-grained.

    3. CBAM attention at each scale: channel attention identifies which
       feature maps matter, spatial attention localizes WHERE in the tile
       the signal is (water vs. bank, plume vs. clear).

    4. FiLM conditioning: pH modulates visual features INSIDE the
       multi-scale aggregation, not just concatenated after. This lets
       the model learn pH-dependent visual patterns (e.g., algae looks
       different at pH 6 vs pH 8).

    5. Multi-pool aggregation: avg + max + std pooling per scale.
       A half-turbid tile has high std; uniformly turbid has low std.
       This spatial heterogeneity is lost by global average pooling alone.

    6. Per-target expert decoders: each WQ parameter (DO, Turb, Temp,
       SpCond) has its own MLP. DO depends on temperature + algal color,
       turbidity depends on NIR backscatter — they need different features.

    7. Band ratio encoder with gradient flow: band ratios are encoded
       into a learned representation that feeds into both the physics
       prior AND each expert decoder, with full gradient flow.

    Architecture diagram:
      4ch imagery
        ├─→ SpectralStem ──→ DenseNet body ──→ 4 scale feature maps
        │                                        ├─ FiLM(pH) + CBAM at each
        │                                        └─ avg+max+std pool ──→ 1536-dim
        ├─→ BandRatioEncoder(+pH) ──→ 48-dim encoded + physics prior
        │
        └─→ Per-target experts:  [1536 + 48] ──→ mu_i, sigma_i  (×4 targets)
                                  + physics_prior residual
    """

    PH_EMBED_DIM = 32

    def __init__(self):
        super().__init__()

        # ----- Spectral Attention Stem -----
        self.stem = SpectralStem()

        # ----- DenseNet121 body (sans conv0 — replaced by SpectralStem) -----
        densenet = tv_models.densenet121(weights="IMAGENET1K_V1")

        # Transfer pretrained conv0 weights to SpectralStem
        self.stem.init_from_densenet(densenet.features.conv0.weight.data)

        # Extract DenseNet body: norm0, relu0, pool0, then 4 denseblocks + transitions
        self.dn_norm0 = densenet.features.norm0
        self.dn_pool0 = densenet.features.pool0
        self.dn_block1 = densenet.features.denseblock1    # → 256ch, 56×56
        self.dn_trans1 = densenet.features.transition1    # → 128ch, 28×28
        self.dn_block2 = densenet.features.denseblock2    # → 512ch, 28×28
        self.dn_trans2 = densenet.features.transition2    # → 256ch, 14×14
        self.dn_block3 = densenet.features.denseblock3    # → 1024ch, 14×14
        self.dn_trans3 = densenet.features.transition3    # → 512ch, 7×7
        self.dn_block4 = densenet.features.denseblock4    # → 1024ch, 7×7
        self.dn_final_norm = densenet.features.norm5       # final BatchNorm

        # ----- pH encoder (for FiLM conditioning) -----
        self.ph_encoder = nn.Sequential(
            nn.Linear(1, self.PH_EMBED_DIM),
            nn.GELU(),
            nn.Linear(self.PH_EMBED_DIM, self.PH_EMBED_DIM),
        )

        # ----- Multi-scale aggregation with CBAM + FiLM -----
        self.multi_scale = MultiScaleAggregator(ph_cond_dim=self.PH_EMBED_DIM)
        ms_dim = self.multi_scale.total_dim  # 1536

        # ----- Band ratio encoder -----
        self.band_encoder = BandRatioEncoder()
        band_enc_dim = BandRatioEncoder.ENCODED_DIM  # 48

        # ----- Per-target expert decoders -----
        self.experts = nn.ModuleList([
            TargetExpert(ms_dim, band_enc_dim) for _ in range(NUM_TARGETS)
        ])

        # ----- Trust router -----
        self.trust_router = nn.Sequential(
            nn.Linear(ms_dim + NUM_TARGETS, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def _extract_multiscale(self, x: torch.Tensor) -> list:
        """Run DenseNet body, return feature maps from each dense block.

        Args:
            x: (B, 64, 112, 112) — output of SpectralStem

        Returns:
            list of 4 feature map tensors at different scales
        """
        x = self.dn_norm0(x)
        x = self.dn_pool0(x)           # (B, 64, 56, 56)

        s1 = self.dn_block1(x)         # (B, 256, 56, 56)
        x = self.dn_trans1(s1)         # (B, 128, 28, 28)

        s2 = self.dn_block2(x)         # (B, 512, 28, 28)
        x = self.dn_trans2(s2)         # (B, 256, 14, 14)

        s3 = self.dn_block3(x)         # (B, 1024, 14, 14)
        x = self.dn_trans3(s3)         # (B, 512, 7, 7)

        s4 = self.dn_block4(x)         # (B, 1024, 7, 7)
        s4 = self.dn_final_norm(s4)
        s4 = F.relu(s4)

        return [s1, s2, s3, s4]

    def forward(self, image: torch.Tensor, ph_input: torch.Tensor,
                ph_valid: Optional[torch.Tensor] = None) -> dict:
        """
        Args:
            image: (B, 4, 224, 224)
            ph_input: (B, 1) — z-scored pH measurement
            ph_valid: (B, 1) — mask for valid pH (optional)

        Returns:
            dict with keys: mu, sigma, physics_prior, trust_logit, backbone_feat
        """
        # pH conditioning
        ph_embed = self.ph_encoder(ph_input)   # (B, 32)
        if ph_valid is not None:
            ph_embed = ph_embed * ph_valid      # zero out if invalid

        # Spectral stem → DenseNet body → multi-scale features
        stem_out = self.stem(image)             # (B, 64, 112, 112)
        scale_feats = self._extract_multiscale(stem_out)  # 4 feature maps

        # Multi-scale aggregation with CBAM + FiLM(pH)
        ms_feat = self.multi_scale(scale_feats, ph_embed)  # (B, 1536)

        # Band ratio encoding
        band_enc, physics_prior = self.band_encoder(image, ph_input)  # (B, 48), (B, 4)

        # Per-target expert predictions
        mus, sigmas = [], []
        for expert in self.experts:
            mu_i, sigma_i = expert(ms_feat, band_enc)
            mus.append(mu_i)
            sigmas.append(sigma_i)

        mu_residual = torch.cat(mus, dim=1)       # (B, 4)
        sigma = torch.cat(sigmas, dim=1)           # (B, 4)
        mu = physics_prior + mu_residual           # residual over physics

        # Trust router
        trust_input = torch.cat([ms_feat, sigma], dim=1)
        trust_logit = self.trust_router(trust_input)  # (B, 1)

        return {
            "mu": mu,
            "sigma": sigma,
            "physics_prior": physics_prior,
            "trust_logit": trust_logit,
            "backbone_feat": ms_feat,
        }


# ---------------------------------------------------------------------------
# Loss functions (same as original WaterDroneNet)
# ---------------------------------------------------------------------------
def gaussian_nll_loss(mu: torch.Tensor, sigma: torch.Tensor,
                      target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mu = mu.float()
    sigma = sigma.float().clamp(min=1e-4)
    target = target.float()
    mask = mask.float()
    nll = 0.5 * (torch.log(sigma ** 2) + (target - mu) ** 2 / sigma ** 2)
    nll = torch.nan_to_num(nll, nan=0.0, posinf=0.0, neginf=0.0)
    return (nll * mask).sum() / mask.sum().clamp(min=1.0)


def masked_mae_loss(mu: torch.Tensor, target: torch.Tensor,
                    mask: torch.Tensor) -> torch.Tensor:
    ae = torch.abs(target.float() - mu.float())
    ae = torch.nan_to_num(ae, nan=0.0)
    return (ae * mask.float()).sum() / mask.float().sum().clamp(min=1.0)


def calibration_loss(mu: torch.Tensor, sigma: torch.Tensor,
                     target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    z_score = torch.abs(target.float() - mu.float()) / sigma.float().clamp(min=1e-4)
    overconf = F.relu(z_score - 2.0)
    return (overconf * mask.float()).sum() / mask.float().sum().clamp(min=1.0)


def trust_supervision_loss(trust_logit: torch.Tensor, mu: torch.Tensor,
                           target: torch.Tensor, mask: torch.Tensor,
                           threshold: float = 2.0) -> torch.Tensor:
    norm_err = torch.abs(target.float() - mu.float().detach())
    per_sample = (norm_err * mask.float()).sum(dim=1) / mask.float().sum(dim=1).clamp(min=1.0)
    label = (per_sample <= threshold).float().unsqueeze(1)
    return F.binary_cross_entropy_with_logits(trust_logit.float(), label)


# ---------------------------------------------------------------------------
# Training & Evaluation
# ---------------------------------------------------------------------------
def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                scaler: GradScaler, device: torch.device) -> dict:
    model.train()
    totals = {"loss": 0.0, "nll": 0.0, "mae": 0.0}
    n = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        ph = batch["ph_input"].to(device, non_blocking=True)
        ph_valid = batch["ph_valid"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        mask = batch["valid_mask"].to(device, non_blocking=True)

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
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             target_mean: Optional[np.ndarray] = None,
             target_std: Optional[np.ndarray] = None) -> dict:
    model.eval()
    all_mu, all_sigma, all_tgt, all_mask = [], [], [], []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        ph = batch["ph_input"].to(device, non_blocking=True)
        ph_valid = batch["ph_valid"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        mask = batch["valid_mask"].to(device, non_blocking=True)

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

    # Denormalize to original scale
    if target_mean is not None and target_std is not None:
        tmean = torch.from_numpy(target_mean).unsqueeze(0)
        tstd = torch.from_numpy(target_std).unsqueeze(0)
        mu = mu * tstd + tmean
        sigma = sigma * tstd.abs()
        tgt = tgt * tstd + tmean
        tgt = tgt * mask

    metrics: Dict = {"per_target": {}}

    # Apply QA/QC filter to denormalized values
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

        # R²
        ss_res = ((y_true - y_pred) ** 2).sum()
        ss_tot = ((y_true - y_true.mean()) ** 2).sum() + 1e-12
        r2 = float(1.0 - ss_res / ss_tot)

        # MAE
        mae_val = float(np.abs(y_true - y_pred).mean())

        # RMSE
        rmse = float(np.sqrt(((y_true - y_pred) ** 2).mean()))

        # Pearson r
        if len(y_true) > 2:
            pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
            if np.isnan(pearson):
                pearson = 0.0
        else:
            pearson = 0.0

        # 90% prediction interval coverage
        z90 = 1.645
        lower = y_pred - z90 * y_sig
        upper = y_pred + z90 * y_sig
        coverage = float(((y_true >= lower) & (y_true <= upper)).mean())

        metrics["per_target"][col] = {
            "r2": r2,
            "mae": mae_val,
            "rmse": rmse,
            "pearson_r": pearson,
            "coverage_90pct": coverage,
            "n_valid": int(m.sum()),
        }

    metrics["n_samples"] = int(len(mu))
    return metrics


def metrics_to_scalar(metrics: dict) -> float:
    """Convert metrics dict to a single scalar for best-model selection (lower = better)."""
    r2s = [v["r2"] for v in metrics.get("per_target", {}).values() if "r2" in v]
    return -float(np.mean(r2s)) if r2s else float("inf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser("Train WaterDroneNet V4 (DenseNet121)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    # Multi-GPU setup
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        device = torch.device("cuda:0")
        log(f"CUDA available: {num_gpus} GPU(s)")
        for i in range(num_gpus):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            log(f"  GPU {i}: {name} ({mem:.0f} GB)")
    else:
        device = torch.device("cpu")
        num_gpus = 0
        log("WARNING: No CUDA GPUs available, using CPU")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    log("=" * 70)
    log("WaterDroneNet V4 — DenseNet121 + pH Strip → Water Quality")
    log("=" * 70)
    log(f"Architecture: HydroDenseNet (DenseNet121 body, custom design)")
    log(f"Stem:        SpectralStem (learnable 4-band interactions)")
    log(f"Scales:      Multi-Scale Feature Pyramid (4 DenseBlock taps)")
    log(f"Attention:   CBAM (channel + spatial) at each scale")
    log(f"Conditioning: FiLM (pH modulates visual features inside network)")
    log(f"Pooling:     Multi-pool (avg + max + std per scale)")
    log(f"Decoders:    Per-target expert MLPs (4 specialists)")
    log(f"Physics:     Band ratio encoder (15 ratios → 48-dim, gradient flow)")
    log(f"Input:       4ch imagery + pH strip measurement")
    log(f"Targets:     {TARGET_COLS} (pH is INPUT, not target)")
    log(f"Multi-GPU:   DataParallel across {max(num_gpus, 1)} GPU(s)")
    log(f"Batch size:  {args.batch_size}")
    log(f"LR:          {args.lr}")
    log(f"Epochs:      {args.epochs}")
    log(f"Patience:    {args.patience}")

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------
    log("\n--- Loading datasets (lazy tile loading) ---")
    train_ds = TileDataset("train", ph_noise_std=PH_STRIP_NOISE_STD, augment=True)
    val_ds = TileDataset("val", ph_noise_std=0.0, augment=False)
    test_ds = TileDataset("test", ph_noise_std=0.0, augment=False)

    # Compute normalization stats from training set
    target_mean, target_std = train_ds.compute_target_stats()
    ph_mean, ph_std = train_ds.compute_ph_stats()
    log(f"pH stats: mean={ph_mean:.3f}, std={ph_std:.3f}")

    train_ds.set_stats(target_mean, target_std, ph_mean, ph_std)
    val_ds.set_stats(target_mean, target_std, ph_mean, ph_std)
    test_ds.set_stats(target_mean, target_std, ph_mean, ph_std)

    log("Target normalization:")
    for i, col in enumerate(TARGET_COLS):
        log(f"  {col:8s}: mean={target_mean[i]:.4f}  std={target_std[i]:.4f}")

    nw = min(args.num_workers, os.cpu_count() or 1)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=nw, pin_memory=True, drop_last=True,
        persistent_workers=nw > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=nw, pin_memory=True,
        persistent_workers=nw > 0,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=nw, pin_memory=True,
        persistent_workers=nw > 0,
    )

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    log("\n--- Building WaterDroneNet V4 ---")
    model = WaterDroneNetV4().to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _dn_modules = [model.dn_norm0, model.dn_pool0, model.dn_block1, model.dn_trans1,
                    model.dn_block2, model.dn_trans2, model.dn_block3, model.dn_trans3,
                    model.dn_block4, model.dn_final_norm]
    n_backbone = sum(p.numel() for m in _dn_modules for p in m.parameters() if p.requires_grad)
    log(f"Total parameters:    {n_params:,}")
    log(f"Backbone parameters: {n_backbone:,} (DenseNet121)")
    log(f"Head parameters:     {n_params - n_backbone:,}")

    # Multi-GPU with DataParallel
    if num_gpus > 1:
        model = nn.DataParallel(model)
        log(f"Wrapped in DataParallel across {num_gpus} GPUs")

    # -----------------------------------------------------------------------
    # Optimizer & Scheduler
    # -----------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs,
    )
    scaler = GradScaler("cuda")

    # -----------------------------------------------------------------------
    # Training Loop
    # -----------------------------------------------------------------------
    log("\n" + "=" * 70)
    log("Training: DenseNet121 + pH → DO, Turb, Temp, SpCond")
    log("=" * 70)

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
        lr_now = optimizer.param_groups[0]["lr"]

        # Per-target R² and Pearson r
        r2_strs = []
        pr_strs = []
        for col in TARGET_COLS:
            if col in val_m.get("per_target", {}):
                r2_strs.append(f"{col}={val_m['per_target'][col]['r2']:.3f}")
                pr_strs.append(f"{col}={val_m['per_target'][col]['pearson_r']:.3f}")
            else:
                r2_strs.append(f"{col}=N/A")
                pr_strs.append(f"{col}=N/A")

        log(f"Epoch {epoch:3d}/{args.epochs} | "
            f"loss={train_m['loss']:.4f} nll={train_m['nll']:.4f} mae={train_m['mae']:.4f} | "
            f"lr={lr_now:.2e} | {dt:.1f}s")
        log(f"  Val R²:      {' | '.join(r2_strs)}")
        log(f"  Val Pearson: {' | '.join(pr_strs)}")

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
                "args": vars(args),
                "architecture": "HydroDenseNet_V4",
            }, CKPT_DIR / "waterdronenet_v4_best.pt")
            log(f"  ** New best (mean R²={-best_score:.4f}) — saved checkpoint")
        elif epoch > args.warmup_epochs:
            patience_counter += 1
            if patience_counter >= args.patience:
                log(f"  Early stopping at epoch {epoch} (patience={args.patience})")
                break

    log(f"\nBest epoch: {best_epoch}, best mean R²: {-best_score:.4f}")

    # -----------------------------------------------------------------------
    # Test Set Evaluation
    # -----------------------------------------------------------------------
    log("\n" + "=" * 70)
    log("Test Set Evaluation (Spatial Holdout)")
    log("=" * 70)

    ckpt = torch.load(CKPT_DIR / "waterdronenet_v4_best.pt",
                       map_location=device, weights_only=False)
    raw = model.module if hasattr(model, "module") else model
    raw.load_state_dict(ckpt["model_state_dict"])
    log(f"Loaded best checkpoint from epoch {ckpt['epoch']}")

    test_m = evaluate(model, test_loader, device, target_mean, target_std)

    log(f"\nTest NLL: {test_m['nll']:.4f}")
    log(f"\n{'Param':<8} {'R²':>8} {'Pearson r':>10} {'MAE':>10} {'RMSE':>10} "
        f"{'Cov90':>8} {'n':>6}")
    log(f"{'---'*20}")

    for col in TARGET_COLS:
        if col in test_m.get("per_target", {}):
            m = test_m["per_target"][col]
            log(f"{col:<8} {m['r2']:>8.4f} {m['pearson_r']:>10.4f} "
                f"{m['mae']:>10.4f} {m['rmse']:>10.4f} "
                f"{m['coverage_90pct']:>8.3f} {m['n_valid']:>6}")

    # -----------------------------------------------------------------------
    # Comparison with previous WaterDroneNet versions
    # -----------------------------------------------------------------------
    log(f"\n{'=' * 70}")
    log("COMPARISON: HydroDenseNet vs Previous Versions")
    log(f"{'=' * 70}")

    comparisons = [
        ("Img-Only (ViT)", RESULTS_DIR / "waterdronenet_holdout.json"),
        ("+pH (ViT)",      RESULTS_DIR / "waterdronenet_ph_holdout.json"),
    ]

    for label, path in comparisons:
        if not path.exists():
            log(f"  {label}: results file not found ({path.name})")
            continue

        with open(path) as f:
            prev = json.load(f)

        log(f"\n--- HydroDenseNet vs {label} ---")
        log(f"{'Param':<8} {'Prev R²':>10} {'V4 R²':>10} {'Delta':>8} "
            f"{'Prev MAE':>10} {'V4 MAE':>10} {'Delta':>10}")
        log(f"{'---'*20}")

        prev_tgt_metrics = prev.get("test_metrics", {}).get("per_target", {})
        for col in TARGET_COLS:
            if col in prev_tgt_metrics and col in test_m.get("per_target", {}):
                o = prev_tgt_metrics[col]
                n = test_m["per_target"][col]
                r2_diff = n["r2"] - o["r2"]
                mae_diff = n["mae"] - o["mae"]
                log(f"{col:<8} {o['r2']:>10.4f} {n['r2']:>10.4f} {r2_diff:>+8.4f} "
                    f"{o['mae']:>10.3f} {n['mae']:>10.3f} {mae_diff:>+10.3f}")
            elif col in test_m.get("per_target", {}):
                n = test_m["per_target"][col]
                log(f"{col:<8} {'N/A':>10} {n['r2']:>10.4f} {'--':>8} "
                    f"{'N/A':>10} {n['mae']:>10.3f} {'--':>10}")

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
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
        "model": "HydroDenseNet (WaterDroneNet V4)",
        "description": "Custom architecture built on DenseNet121 for water quality",
        "backbone": "DenseNet121 body with SpectralStem + Multi-Scale FPN",
        "input": "Sentinel-2 RGB+NIR (4ch, 224x224) + pH strip measurement",
        "improvements": [
            "SpectralStem: learnable band interactions before DenseNet body",
            "Multi-Scale Feature Pyramid: taps all 4 DenseBlock outputs",
            "CBAM (channel + spatial) attention at each scale",
            "FiLM conditioning: pH modulates visual features inside the network",
            "Multi-pool aggregation: avg + max + std captures spatial heterogeneity",
            "Per-target expert decoders: DO/Turb/Temp/SpCond each specialized",
            "Band ratio encoder with gradient flow (15 → 48-dim learned)",
            "Multi-GPU DataParallel training",
            "Lazy tile loading (individual .npz files)",
            "Data augmentation (flip + rotate)",
        ],
        "n_params": int(n_params),
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
        "best_epoch": best_epoch,
        "best_val_score": float(best_score),
        "targets": TARGET_COLS,
        "ph_input": {
            "mean": ph_mean,
            "std": ph_std,
            "noise_std": PH_STRIP_NOISE_STD,
        },
        "holdout": {
            "type": "spatial",
            "train": "70% of stations",
            "val": "15% of stations",
            "test": "15% of stations (geographically unseen)",
            "seed": 42,
        },
        "hyperparameters": {
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "patience": args.patience,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealing",
            "mixed_precision": True,
        },
        "test_metrics": test_m,
    })

    out_path = RESULTS_DIR / "waterdronenet_v4_holdout.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    log(f"\nResults saved to {out_path}")
    log(f"Checkpoint at  {CKPT_DIR / 'waterdronenet_v4_best.pt'}")
    log("DONE")


if __name__ == "__main__":
    main()
