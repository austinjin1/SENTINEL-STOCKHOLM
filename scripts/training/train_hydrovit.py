#!/usr/bin/env python3
"""HydroViT — CNN-ViT hybrid with strict temporal-spatial holdout splits.

Replaces the random 70/15/15 split of train_hydrovit.py (v9) with the
SENTINEL temporal-spatial holdout protocol:
  - Train:  2015-2022, spatial folds A-C
  - Val:    2023,       spatial fold D
  - Test:   2024-2026,  spatial fold E

Data source: REAL satellite tiles only — no synthetic data.
  Primary:  Individual .npz tiles from data/processed/satellite/real/
  Fallback: data/processed/satellite/paired_wq_v5.npz

Architecture (identical to v9):
  raw_10ch
    -> PerParamBandAttention (shared gate)
    -> LocalCNNExtractor  -> multi-scale features [32, 64, 128]
    -> CNNToViTAdapter    -> 13ch fused map
    -> SatelliteEncoder   -> ViT CLS embedding [256]
    -> MultiScaleCNNAggregator -> CNN embedding [256]
    -> DeepWQHeadV2       -> [16] predictions

Training protocol:
  Phase 1: Head only (backbone frozen), lr=3e-4, 80 epochs
  Phase 2: Full fine-tune, lr=3e-5 backbone / 3e-4 head, 120 epochs
  Early stopping on validation R² with patience=20

MIT License — Bryan Cheng, 2026
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sentinel.data.splits import (
    SplitConfig,
    assign_spatial_fold,
    split_indices,
    summarize_split,
)
from sentinel.models.satellite_encoder.model import SatelliteEncoder
from sentinel.models.satellite_encoder.parameter_head import (
    PARAM_NAMES,
    NUM_WATER_PARAMS,
)
from sentinel.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CKPT_DIR = PROJECT_ROOT / "checkpoints" / "satellite"
RESULTS_DIR = PROJECT_ROOT / "results" / "benchmarks"
LOGS_DIR = PROJECT_ROOT / "logs"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Data paths
REAL_TILE_DIR = PROJECT_ROOT / "data" / "processed" / "satellite" / "real"
PAIRED_DATA_V5 = PROJECT_ROOT / "data" / "processed" / "satellite" / "paired_wq_v5.npz"
PAIRED_DATA_V3 = PROJECT_ROOT / "data" / "processed" / "satellite" / "paired_wq_v3.npz"

PRETRAINED_CKPT = CKPT_DIR / "hydrovit_real_mae.pt"
OUTPUT_CKPT = CKPT_DIR / "hydrovit_v2_best.pt"
RESULTS_JSON = RESULTS_DIR / "hydrovit_v2_holdout.json"
LOG_FILE = LOGS_DIR / "train_hydrovit_v2.log"

SEED = 42
BATCH_SIZE = 8
GRAD_ACCUM = 4  # effective batch = 32
HEAD_LR = 3e-4
BACKBONE_LR = 3e-5
HEAD_EPOCHS = 80
FINETUNE_EPOCHS = 120
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
USE_AMP = True
EARLY_STOP_PATIENCE = 20

# Per-parameter loss weights
WATER_TEMP_IDX = PARAM_NAMES.index("water_temp") if "water_temp" in PARAM_NAMES else 11
CHL_A_IDX = PARAM_NAMES.index("chl_a") if "chl_a" in PARAM_NAMES else 0
TURBIDITY_IDX = PARAM_NAMES.index("turbidity") if "turbidity" in PARAM_NAMES else 1
PHYCOCYANIN_IDX = PARAM_NAMES.index("phycocyanin") if "phycocyanin" in PARAM_NAMES else 12

PARAM_WEIGHTS = torch.ones(NUM_WATER_PARAMS)
PARAM_WEIGHTS[WATER_TEMP_IDX] = 3.0
PARAM_WEIGHTS[CHL_A_IDX] = 3.0
PARAM_WEIGHTS[TURBIDITY_IDX] = 2.0
PARAM_WEIGHTS[PHYCOCYANIN_IDX] = 2.0

# Reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ===========================================================================
# Model components (identical to v9)
# ===========================================================================

class LocalCNNExtractor(nn.Module):
    """3-layer stride-1 CNN that builds multi-scale local features.

    Input:  [B, in_chans, H, W]  (10 bands)
    Output: [B, 128, H, W] + intermediate feature maps at 32 and 64 ch

    All convolutions are stride-1 with padding so spatial dimensions are
    fully preserved.  Receptive field: 3->5->9 (90m at 10m/px).
    """

    def __init__(self, in_chans: int = 10):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_chans, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        self.conv3_dw = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2, groups=64, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        self.conv3_pw = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor):
        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        f3 = self.conv3_pw(self.conv3_dw(f2))
        return f3, [f1, f2]


class CNNToViTAdapter(nn.Module):
    """Project CNN 128-ch feature map back to the ViT input space (13 ch)."""

    def __init__(self, cnn_out_channels: int = 128, in_chans: int = 10, vit_in_chans: int = 13):
        super().__init__()
        combined = in_chans + cnn_out_channels
        self.proj = nn.Sequential(
            nn.Conv2d(combined, vit_in_chans, kernel_size=1, bias=False),
            nn.BatchNorm2d(vit_in_chans),
        )
        self.blend = nn.Parameter(torch.tensor(0.5))

    def forward(self, x_raw: torch.Tensor, x_cnn: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([x_raw, x_cnn], dim=1)
        return self.proj(combined)


class MultiScaleCNNAggregator(nn.Module):
    """Global avg-pool each CNN feature map and project to embed_dim."""

    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(224, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, feats: list[torch.Tensor], out128: torch.Tensor) -> torch.Tensor:
        pooled = [F.adaptive_avg_pool2d(f, 1).flatten(1) for f in feats]
        pooled.append(F.adaptive_avg_pool2d(out128, 1).flatten(1))
        x = torch.cat(pooled, dim=1)
        return self.proj(x)


class PerParamBandAttention(nn.Module):
    """Learnable band weighting per output parameter with physics-informed init."""

    def __init__(self, in_chans: int = 10, n_params: int = 16):
        super().__init__()
        w = torch.ones(n_params, in_chans)
        w[0, 2] = 2.0; w[0, 3] = 2.0; w[0, 4] = 2.5  # chl_a: green, red, red-edge
        w[1, 6] = 1.5; w[1, 8] = 1.5  # turbidity: NIR, SWIR1
        w[12, 3] = 1.5; w[12, 4] = 2.0; w[12, 5] = 2.0; w[12, 6] = 1.5  # phycocyanin
        w[11, 8] = 1.5; w[11, 9] = 1.5  # water_temp: SWIR
        self.band_weights = nn.Parameter(w)

    def get_channel_gate(self) -> torch.Tensor:
        avg_w = self.band_weights.mean(0)
        return torch.softmax(avg_w, dim=0) * avg_w.shape[0]

    def get_per_param_weights(self) -> torch.Tensor:
        return torch.softmax(self.band_weights, dim=1)


class DeepWQHeadV2(nn.Module):
    """Dual-stream head: ViT embedding + CNN multi-scale aggregation.

    Streams fuse via concat -> 512 -> residual blocks -> 16 outputs.
    """

    def __init__(
        self,
        vit_dim: int = 256,
        cnn_dim: int = 256,
        hidden: int = 512,
        n_params: int = 16,
    ):
        super().__init__()
        fused_dim = vit_dim + cnn_dim

        self.bn_in = nn.LayerNorm(fused_dim)
        self.fc1 = nn.Linear(fused_dim, hidden)
        self.bn1 = nn.LayerNorm(hidden)
        self.drop1 = nn.Dropout(0.15)

        self.fc2 = nn.Linear(hidden, 384)
        self.bn2 = nn.LayerNorm(384)
        self.drop2 = nn.Dropout(0.1)

        self.fc3 = nn.Linear(384, 256)
        self.bn3 = nn.LayerNorm(256)

        self.res1 = nn.Linear(fused_dim, hidden)
        self.res2 = nn.Linear(hidden, 384)
        self.res3 = nn.Linear(384, 256)

        self.out = nn.Linear(256, n_params)

    def forward(self, vit_emb: torch.Tensor, cnn_emb: torch.Tensor) -> torch.Tensor:
        x = torch.cat([vit_emb, cnn_emb], dim=1)
        x = self.bn_in(x)
        h = F.gelu(self.bn1(self.fc1(x)))
        h = self.drop1(h)
        h = h + self.res1(x)
        h2 = F.gelu(self.bn2(self.fc2(h)))
        h2 = self.drop2(h2)
        h2 = h2 + self.res2(h)
        h3 = F.gelu(self.bn3(self.fc3(h2)))
        h3 = h3 + self.res3(h2)
        return self.out(h3)


class HydroViTV2(nn.Module):
    """CNN-ViT hybrid for water quality estimation (v2 with holdout splits).

    Architecture is identical to HydroViTV9 — only data splitting differs.

    Pipeline:
      raw_10ch
        -> PerParamBandAttention (shared gate)
        -> LocalCNNExtractor  -> multi-scale features [32, 64, 128]
        -> CNNToViTAdapter    -> 13ch fused map
        -> SatelliteEncoder   -> ViT CLS embedding [256]
        -> MultiScaleCNNAggregator -> CNN embedding [256]
        -> DeepWQHeadV2       -> [16] predictions
    """

    def __init__(self, pretrained_ckpt: Path | None = None, in_bands: int = 10):
        super().__init__()
        self.in_bands = in_bands

        self.band_attn = PerParamBandAttention(in_chans=in_bands)
        self.cnn = LocalCNNExtractor(in_chans=in_bands)
        self.adapter = CNNToViTAdapter(cnn_out_channels=128, in_chans=in_bands, vit_in_chans=13)
        self.encoder = SatelliteEncoder()
        self.cnn_agg = MultiScaleCNNAggregator(embed_dim=self.encoder.shared_embed_dim)
        self.wq_head = DeepWQHeadV2(
            vit_dim=self.encoder.shared_embed_dim,
            cnn_dim=self.encoder.shared_embed_dim,
        )

        if pretrained_ckpt and pretrained_ckpt.exists():
            try:
                ckpt = torch.load(str(pretrained_ckpt), map_location="cpu", weights_only=False)
                state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
                missing, unexpected = self.encoder.load_state_dict(state, strict=False)
                logger.info(
                    f"Loaded pretrained backbone: {len(missing)} missing, "
                    f"{len(unexpected)} unexpected keys"
                )
            except Exception as e:
                logger.warning(f"Could not load pretrained ckpt: {e}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 10, H, W]
        Returns:
            preds: [B, 16]
        """
        gate = self.band_attn.get_channel_gate()
        x_gated = x * gate.view(1, -1, 1, 1)
        cnn_out, cnn_feats = self.cnn(x_gated)
        x13 = self.adapter(x_gated, cnn_out)
        enc_out = self.encoder(x13)
        vit_emb = enc_out["embedding"]
        cnn_emb = self.cnn_agg(cnn_feats, cnn_out)
        return self.wq_head(vit_emb, cnn_emb)


# ===========================================================================
# Data loading — REAL data only
# ===========================================================================

def load_individual_tiles(tile_dir: Path) -> dict:
    """Load individual real Sentinel-2 .npz tiles.

    Each file contains: image (10,224,224), targets (16,),
    station_id (str), date (str), latitude (float), longitude (float).

    Returns dict with keys: images, targets, station_ids, dates, latitudes, longitudes
    """
    npz_files = sorted(tile_dir.glob("*.npz"))
    if not npz_files:
        return {}

    images = []
    targets = []
    station_ids = []
    dates = []
    latitudes = []
    longitudes = []
    skipped = 0

    for f in npz_files:
        try:
            tile = np.load(str(f), allow_pickle=True)
        except Exception as e:
            logger.warning(f"Failed to load {f.name}: {e}")
            skipped += 1
            continue

        # Validate required fields
        if "image" not in tile or "targets" not in tile:
            logger.warning(f"Skipping {f.name}: missing image or targets")
            skipped += 1
            continue

        img = tile["image"]
        tgt = tile["targets"]
        if img.shape != (10, 224, 224):
            logger.warning(f"Skipping {f.name}: unexpected image shape {img.shape}")
            skipped += 1
            continue
        if tgt.shape != (16,):
            logger.warning(f"Skipping {f.name}: unexpected targets shape {tgt.shape}")
            skipped += 1
            continue

        images.append(img.astype(np.float32))
        targets.append(tgt.astype(np.float32))

        # Extract metadata for splitting
        sid = str(tile["station_id"]) if "station_id" in tile else "unknown"
        date = str(tile["date"]) if "date" in tile else "2020-01-01"
        lat = float(tile["latitude"]) if "latitude" in tile else 0.0
        lon = float(tile["longitude"]) if "longitude" in tile else 0.0

        station_ids.append(sid)
        dates.append(date)
        latitudes.append(lat)
        longitudes.append(lon)

    if skipped > 0:
        logger.warning(f"Skipped {skipped} tiles due to errors")

    if not images:
        return {}

    return {
        "images": np.stack(images),
        "targets": np.stack(targets),
        "station_ids": np.array(station_ids),
        "dates": np.array(dates),
        "latitudes": np.array(latitudes),
        "longitudes": np.array(longitudes),
    }


def load_paired_npz(path: Path) -> dict:
    """Load the paired WQ .npz fallback file.

    Expected arrays: images (N,10,224,224), targets (N,16).
    Optional: station_ids, dates.

    Returns dict with same keys as load_individual_tiles.
    """
    data = np.load(str(path), allow_pickle=True)

    images = data["images"].astype(np.float32)
    targets = data["targets"].astype(np.float32)
    N = len(images)

    # Extract station_ids if present
    if "station_ids" in data:
        station_ids = np.array([str(s) for s in data["station_ids"]])
    elif "site_ids" in data:
        station_ids = np.array([str(s) for s in data["site_ids"]])
    else:
        # Generate deterministic pseudo station IDs from data hash
        logger.warning(
            "No station_ids in paired npz — generating pseudo IDs from row index. "
            "Spatial split will be approximate."
        )
        station_ids = np.array([f"site_{i:05d}" for i in range(N)])

    # Extract dates if present
    if "dates" in data:
        dates = np.array([str(d) for d in data["dates"]])
    elif "timestamps" in data:
        dates = np.array([str(d) for d in data["timestamps"]])
    else:
        logger.warning(
            "No dates in paired npz — temporal split will not be possible. "
            "All samples will use spatial-only splitting."
        )
        dates = None

    # Extract coordinates if present (not strictly needed for splitting
    # since we use hash-based spatial folds, but useful for logging)
    latitudes = data.get("latitudes", np.zeros(N))
    longitudes = data.get("longitudes", np.zeros(N))
    if isinstance(latitudes, np.lib.npyio.NpzFile):
        latitudes = np.zeros(N)
    if isinstance(longitudes, np.lib.npyio.NpzFile):
        longitudes = np.zeros(N)

    result = {
        "images": images,
        "targets": targets,
        "station_ids": station_ids,
        "latitudes": np.array(latitudes, dtype=np.float64),
        "longitudes": np.array(longitudes, dtype=np.float64),
    }
    if dates is not None:
        result["dates"] = dates
    return result


def load_data() -> dict:
    """Load real satellite data with metadata for splitting.

    Tries individual tiles first, then falls back to paired .npz.
    Fails with clear error if no data is available.
    """
    # Strategy 1: Individual real tiles
    if REAL_TILE_DIR.is_dir():
        data = load_individual_tiles(REAL_TILE_DIR)
        if data and len(data["images"]) > 0:
            logger.info(
                f"Loaded {len(data['images'])} individual real tiles "
                f"from {REAL_TILE_DIR}"
            )
            return data
        else:
            logger.warning(f"No valid tiles found in {REAL_TILE_DIR}")

    # Strategy 2: Paired .npz fallback
    for paired_path in [PAIRED_DATA_V5, PAIRED_DATA_V3]:
        if paired_path.exists():
            logger.info(f"Falling back to paired .npz: {paired_path}")
            data = load_paired_npz(paired_path)
            if len(data["images"]) > 0:
                logger.info(f"Loaded {len(data['images'])} samples from {paired_path}")
                return data

    # No data available — fail with clear instructions
    raise FileNotFoundError(
        "No real satellite data found. Expected one of:\n"
        f"  1. Individual .npz tiles in: {REAL_TILE_DIR}/\n"
        f"  2. Paired dataset at: {PAIRED_DATA_V5}\n"
        f"  3. Paired dataset at: {PAIRED_DATA_V3}\n\n"
        "Run scripts/download_satellite_real.py first to acquire data, then\n"
        "scripts/coregister_satellite_wq.py to create paired tiles."
    )


# ===========================================================================
# Dataset with target normalization
# ===========================================================================

class HoldoutWQDataset(Dataset):
    """Water quality dataset with log-transform + z-score normalization.

    Normalization is computed from training data only and applied to all splits
    to prevent information leakage.
    """

    # Parameters that benefit from log1p transform (right-skewed distributions)
    LOG_PARAMS = {0, 1, 3, 4, 5, 6, 8, 9, 12, 14}

    def __init__(
        self,
        images: np.ndarray,
        targets: np.ndarray,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
    ):
        self.images = images.astype(np.float32)
        targets = targets.astype(np.float32).copy()

        # Apply log1p to right-skewed parameters
        for i in self.LOG_PARAMS:
            valid = ~np.isnan(targets[:, i])
            if valid.any():
                targets[valid, i] = np.log1p(np.maximum(targets[valid, i], 1e-6))

        if mean is None or std is None:
            # Compute normalization from this split (should only be train)
            self.mean = np.nanmean(targets, axis=0)
            self.std = np.nanstd(targets, axis=0)
            self.std[self.std < 1e-6] = 1.0
            # Handle all-NaN columns
            all_nan_cols = np.all(np.isnan(targets), axis=0)
            self.mean[all_nan_cols] = 0.0
        else:
            self.mean = mean
            self.std = std

        # Z-score normalize
        for i in range(targets.shape[1]):
            valid = ~np.isnan(targets[:, i])
            if valid.any():
                targets[valid, i] = (targets[valid, i] - self.mean[i]) / self.std[i]

        self.targets_norm = targets

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return {
            "image": torch.tensor(self.images[idx]),
            "targets": torch.tensor(self.targets_norm[idx]),
        }


# ===========================================================================
# Temporal-spatial split with leakage verification
# ===========================================================================

def build_holdout_splits(
    data: dict,
) -> tuple[dict[str, list[int]], dict]:
    """Apply SENTINEL temporal-spatial holdout splits to the data.

    Returns:
        split_idx: dict mapping "train"/"val"/"test" to index lists
        split_meta: dict with split statistics for logging and saving
    """
    config = SplitConfig()
    station_ids = list(data["station_ids"])
    N = len(station_ids)

    # Determine if we have temporal information
    has_dates = "dates" in data and data["dates"] is not None
    if has_dates:
        dates = list(data["dates"])
        logger.info(f"Using temporal-spatial holdout (N={N}, have dates)")
    else:
        dates = None
        logger.info(f"Using spatial-only holdout (N={N}, no dates available)")

    # Apply split
    split_idx = split_indices(
        site_ids=station_ids,
        timestamps=dates,
        config=config,
    )

    # Build metadata for logging
    split_meta = {"n_total": N}

    for split_name in ["train", "val", "test"]:
        idxs = split_idx[split_name]
        info = {"n_samples": len(idxs)}

        if idxs:
            sites = set(station_ids[i] for i in idxs)
            info["n_sites"] = len(sites)
            info["sites_sample"] = sorted(sites)[:5]

            if has_dates:
                split_dates = [dates[i] for i in idxs]
                info["date_min"] = min(split_dates)
                info["date_max"] = max(split_dates)

            # Per-parameter coverage: count non-NaN entries
            tgt_slice = data["targets"][idxs]
            coverage = {}
            for j, name in enumerate(PARAM_NAMES):
                n_valid = int(np.sum(~np.isnan(tgt_slice[:, j])))
                if n_valid > 0:
                    coverage[name] = n_valid
            info["param_coverage"] = coverage
        else:
            info["n_sites"] = 0

        split_meta[split_name] = info

    # Leakage verification: check site overlap between splits
    train_sites = set(station_ids[i] for i in split_idx["train"]) if split_idx["train"] else set()
    val_sites = set(station_ids[i] for i in split_idx["val"]) if split_idx["val"] else set()
    test_sites = set(station_ids[i] for i in split_idx["test"]) if split_idx["test"] else set()

    tv_leak = train_sites & val_sites
    tt_leak = train_sites & test_sites
    vt_leak = val_sites & test_sites

    split_meta["leakage_check"] = {
        "train_val_overlap": len(tv_leak),
        "train_test_overlap": len(tt_leak),
        "val_test_overlap": len(vt_leak),
        "leaked_sites": sorted(tv_leak | tt_leak | vt_leak)[:10],
    }

    if tv_leak or tt_leak or vt_leak:
        logger.error(
            f"SITE LEAKAGE DETECTED! "
            f"train-val={len(tv_leak)}, train-test={len(tt_leak)}, "
            f"val-test={len(vt_leak)}"
        )
        raise RuntimeError(
            "Temporal-spatial holdout has site leakage — this should not happen. "
            "Check sentinel.data.splits for bugs."
        )
    else:
        logger.info("Leakage check PASSED: zero site overlap between train/val/test")

    # Excluded samples
    included = set()
    for idxs in split_idx.values():
        included.update(idxs)
    n_excluded = N - len(included)
    split_meta["n_excluded"] = n_excluded
    if n_excluded > 0:
        logger.info(
            f"Excluded {n_excluded} samples ({100*n_excluded/N:.1f}%) "
            "that fell into cross-split (e.g., train-site in val-period)"
        )

    return split_idx, split_meta


def log_split_summary(split_meta: dict) -> None:
    """Log a formatted summary of split statistics."""
    logger.info("=" * 70)
    logger.info("TEMPORAL-SPATIAL HOLDOUT SPLIT SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  Total samples: {split_meta['n_total']}")
    logger.info(f"  Excluded:      {split_meta.get('n_excluded', 0)}")

    for split_name in ["train", "val", "test"]:
        info = split_meta[split_name]
        n = info["n_samples"]
        ns = info["n_sites"]
        pct = 100 * n / max(split_meta["n_total"], 1)

        date_str = ""
        if "date_min" in info:
            date_str = f" | dates: {info['date_min']} to {info['date_max']}"

        coverage = info.get("param_coverage", {})
        n_params_covered = len(coverage)

        logger.info(
            f"  {split_name:>5}: {n:5d} samples ({pct:5.1f}%) | "
            f"{ns:3d} sites | {n_params_covered} params{date_str}"
        )

    leak = split_meta["leakage_check"]
    status = "PASS" if (
        leak["train_val_overlap"] == 0
        and leak["train_test_overlap"] == 0
        and leak["val_test_overlap"] == 0
    ) else "FAIL"
    logger.info(f"  Leakage: {status}")
    logger.info("=" * 70)


# ===========================================================================
# Loss + metrics
# ===========================================================================

def weighted_mse(
    preds: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Weighted MSE with NaN masking."""
    valid = ~torch.isnan(targets)
    if valid.sum() == 0:
        return torch.tensor(0.0, device=preds.device, requires_grad=True)
    w = weights.to(preds.device).unsqueeze(0).expand_as(targets)
    diff2 = (preds - targets.nan_to_num(0.0)) ** 2
    return (diff2 * w * valid.float()).sum() / (w * valid.float()).sum().clamp(min=1e-6)


def compute_r2(preds: dict, tgts: dict) -> dict:
    """Compute per-parameter R-squared."""
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


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict, float]:
    """Evaluate model on a data loader. Returns (per-param R², mean loss)."""
    model.eval()
    preds = {j: [] for j in range(NUM_WATER_PARAMS)}
    tgts = {j: [] for j in range(NUM_WATER_PARAMS)}
    total_loss, n_b = 0.0, 0

    for batch in loader:
        img = batch["image"].to(device)
        tgt = batch["targets"].to(device)

        with torch.amp.autocast("cuda", enabled=USE_AMP and device.type == "cuda"):
            pred = model(img)
            loss = weighted_mse(pred, tgt, PARAM_WEIGHTS)

        if not torch.isnan(loss):
            total_loss += loss.item()
            n_b += 1

        valid = ~torch.isnan(tgt)
        for j in range(NUM_WATER_PARAMS):
            mask = valid[:, j]
            if mask.sum() > 0:
                preds[j].append(pred[:, j][mask].float().cpu())
                tgts[j].append(tgt[:, j][mask].float().cpu())

    r2 = compute_r2(preds, tgts)
    return r2, total_loss / max(n_b, 1)


# ===========================================================================
# Training with early stopping
# ===========================================================================

class EarlyStopping:
    """Early stopping tracker based on validation R-squared."""

    def __init__(self, patience: int = 20, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = -float("inf")
        self.counter = 0
        self.best_state = None
        self.should_stop = False

    def step(self, score: float, model: nn.Module) -> bool:
        """Update tracker. Returns True if improved."""
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            self.best_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False

    def restore_best(self, model: nn.Module) -> None:
        """Load best weights into model."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def train_phase(
    model: nn.Module,
    train_dl: DataLoader,
    val_dl: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    epochs: int,
    label: str,
    device: torch.device,
    grad_accum: int = 1,
    early_stop_patience: int = EARLY_STOP_PATIENCE,
    eval_every: int = 5,
) -> tuple[float, float]:
    """Train for a phase with gradient accumulation and early stopping.

    Returns (best_mean_r2, best_water_temp_r2).
    """
    early_stop = EarlyStopping(patience=early_stop_patience)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP and device.type == "cuda")

    for ep in range(epochs):
        model.train()
        total_loss, n_b = 0.0, 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_dl):
            img = batch["image"].to(device)
            tgt = batch["targets"].to(device)

            with torch.amp.autocast("cuda", enabled=USE_AMP and device.type == "cuda"):
                pred = model(img)
                loss = weighted_mse(pred, tgt, PARAM_WEIGHTS)
                loss = loss / grad_accum

            if torch.isnan(loss):
                optimizer.zero_grad()
                continue

            scaler.scale(loss).backward()

            if (step + 1) % grad_accum == 0 or (step + 1) == len(train_dl):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * grad_accum
            n_b += 1

        if scheduler:
            scheduler.step()

        # Evaluate periodically
        if (ep + 1) % eval_every == 0 or ep == epochs - 1:
            r2, val_loss = evaluate(model, val_dl, device)
            valid_r2s = [v for v in r2.values() if not np.isnan(v)]
            mean_r2 = np.mean(valid_r2s) if valid_r2s else -1.0
            wt_r2 = r2.get("water_temp", float("nan"))
            chl_r2 = r2.get("chl_a", float("nan"))

            improved = early_stop.step(mean_r2, model)
            marker = " *BEST*" if improved else ""

            logger.info(
                f"[{label}] Ep {ep+1:3d}/{epochs} | "
                f"tr={total_loss/max(n_b,1):.4f} val={val_loss:.4f} | "
                f"mean_R2={mean_r2:.4f} wt={wt_r2:.4f} chl_a={chl_r2:.4f} | "
                f"ES={early_stop.counter}/{early_stop.patience}{marker}"
            )

            if early_stop.should_stop:
                logger.info(
                    f"[{label}] Early stopping at epoch {ep+1} "
                    f"(best mean R2={early_stop.best_score:.4f})"
                )
                break

    # Restore best model
    early_stop.restore_best(model)
    return early_stop.best_score, early_stop.best_score


# ===========================================================================
# Main
# ===========================================================================

def main():
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("HydroViT v2 — CNN-ViT hybrid with temporal-spatial holdout")
    logger.info(f"  Device: {DEVICE}")
    logger.info(f"  Seed:   {SEED}")
    logger.info(f"  Grad accum: {GRAD_ACCUM} (effective batch={BATCH_SIZE * GRAD_ACCUM})")
    logger.info(f"  AMP: {USE_AMP}")
    logger.info(f"  Early stopping patience: {EARLY_STOP_PATIENCE}")
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load REAL data
    # ------------------------------------------------------------------
    logger.info("\n--- Loading real satellite data ---")
    data = load_data()
    N = len(data["images"])
    logger.info(f"Total samples loaded: {N}")
    logger.info(f"Image shape: {data['images'].shape}")
    logger.info(f"Targets shape: {data['targets'].shape}")
    logger.info(f"Unique stations: {len(set(data['station_ids']))}")

    if "dates" in data:
        logger.info(f"Date range: {min(data['dates'])} to {max(data['dates'])}")

    # ------------------------------------------------------------------
    # 2. Apply temporal-spatial holdout splits
    # ------------------------------------------------------------------
    logger.info("\n--- Applying temporal-spatial holdout splits ---")
    split_idx, split_meta = build_holdout_splits(data)
    log_split_summary(split_meta)

    # Validate minimum split sizes
    n_train = len(split_idx["train"])
    n_val = len(split_idx["val"])
    n_test = len(split_idx["test"])

    if n_train < 10:
        raise RuntimeError(
            f"Training split has only {n_train} samples — need at least 10. "
            "Check data availability and split configuration."
        )
    if n_val < 5:
        raise RuntimeError(
            f"Validation split has only {n_val} samples — need at least 5. "
            "Check data availability and split configuration."
        )
    if n_test < 5:
        logger.warning(
            f"Test split has only {n_test} samples — results may be unreliable."
        )

    # ------------------------------------------------------------------
    # 3. Create datasets (normalize using train stats only)
    # ------------------------------------------------------------------
    train_images = data["images"][split_idx["train"]]
    train_targets = data["targets"][split_idx["train"]]
    val_images = data["images"][split_idx["val"]]
    val_targets = data["targets"][split_idx["val"]]
    test_images = data["images"][split_idx["test"]]
    test_targets = data["targets"][split_idx["test"]]

    # Compute normalization from training split only
    train_ds = HoldoutWQDataset(train_images, train_targets)

    # Apply train normalization to val/test (prevents target leakage)
    val_ds = HoldoutWQDataset(val_images, val_targets, mean=train_ds.mean, std=train_ds.std)
    test_ds = HoldoutWQDataset(test_images, test_targets, mean=train_ds.mean, std=train_ds.std)

    logger.info(
        f"Datasets created: {len(train_ds)} train / "
        f"{len(val_ds)} val / {len(test_ds)} test"
    )

    # DataLoaders
    num_workers = min(4, os.cpu_count() or 1)
    train_dl = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=False,
    )
    val_dl = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    test_dl = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    # ------------------------------------------------------------------
    # 4. Build model
    # ------------------------------------------------------------------
    logger.info("\n--- Building HydroViT v2 model ---")
    model = HydroViTV2(pretrained_ckpt=PRETRAINED_CKPT).to(DEVICE)
    n_params_total = sum(p.numel() for p in model.parameters())
    n_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {n_params_total:,} total / {n_params_trainable:,} trainable params")

    cnn_params = sum(
        p.numel() for p in (
            list(model.cnn.parameters())
            + list(model.adapter.parameters())
            + list(model.cnn_agg.parameters())
        )
    )
    vit_params = sum(p.numel() for p in model.encoder.parameters())
    head_params = sum(p.numel() for p in model.wq_head.parameters())
    logger.info(f"  CNN extractor+adapter+agg: {cnn_params:,}")
    logger.info(f"  ViT encoder:               {vit_params:,}")
    logger.info(f"  WQ head:                   {head_params:,}")

    # ------------------------------------------------------------------
    # 5. Phase 1: Train CNN + head (freeze ViT backbone)
    # ------------------------------------------------------------------
    logger.info("\n--- Phase 1: CNN + head training (ViT backbone frozen) ---")
    for p in model.encoder.parameters():
        p.requires_grad_(False)

    phase1_params = (
        list(model.band_attn.parameters())
        + list(model.cnn.parameters())
        + list(model.adapter.parameters())
        + list(model.cnn_agg.parameters())
        + list(model.wq_head.parameters())
    )
    opt1 = torch.optim.AdamW(
        [p for p in phase1_params if p.requires_grad],
        lr=HEAD_LR,
        weight_decay=WEIGHT_DECAY,
    )
    sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt1, T_max=HEAD_EPOCHS, eta_min=1e-5
    )
    best_r2_p1, _ = train_phase(
        model, train_dl, val_dl, opt1, sch1, HEAD_EPOCHS,
        "Phase1-CNN+Head", DEVICE, grad_accum=GRAD_ACCUM,
        early_stop_patience=EARLY_STOP_PATIENCE,
    )
    logger.info(f"Phase 1 best val mean R2: {best_r2_p1:.4f}")

    # ------------------------------------------------------------------
    # 6. Phase 2: Full fine-tune (unfreeze ViT, lower LR for backbone)
    # ------------------------------------------------------------------
    logger.info("\n--- Phase 2: Full fine-tune ---")
    for p in model.encoder.parameters():
        p.requires_grad_(True)

    opt2 = torch.optim.AdamW(
        [
            {"params": model.band_attn.parameters(), "lr": HEAD_LR},
            {"params": model.cnn.parameters(), "lr": HEAD_LR},
            {"params": model.adapter.parameters(), "lr": HEAD_LR},
            {"params": model.cnn_agg.parameters(), "lr": HEAD_LR},
            {"params": model.wq_head.parameters(), "lr": HEAD_LR},
            {"params": model.encoder.parameters(), "lr": BACKBONE_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt2, T_max=FINETUNE_EPOCHS, eta_min=1e-6
    )
    best_r2_p2, _ = train_phase(
        model, train_dl, val_dl, opt2, sch2, FINETUNE_EPOCHS,
        "Phase2-Finetune", DEVICE, grad_accum=GRAD_ACCUM,
        early_stop_patience=EARLY_STOP_PATIENCE,
    )
    logger.info(f"Phase 2 best val mean R2: {best_r2_p2:.4f}")

    # ------------------------------------------------------------------
    # 7. Final test evaluation
    # ------------------------------------------------------------------
    logger.info("\n--- Final test evaluation ---")
    r2_test, test_loss = evaluate(model, test_dl, DEVICE)
    valid_r2s = [v for v in r2_test.values() if not np.isnan(v)]
    mean_r2_test = float(np.mean(valid_r2s)) if valid_r2s else -1.0
    wt_r2_test = r2_test.get("water_temp", float("nan"))
    chl_r2_test = r2_test.get("chl_a", float("nan"))

    # Per-parameter results table
    logger.info("\n" + "=" * 60)
    logger.info(f"{'Parameter':>25} | {'Test R2':>10} | {'Status':>8}")
    logger.info("-" * 60)
    for name, r2_val in r2_test.items():
        if np.isnan(r2_val):
            logger.info(f"{name:>25} |       N/A  |   NO DATA")
        else:
            status = "OK" if r2_val > 0.0 else "POOR"
            logger.info(f"{name:>25} | {r2_val:>10.4f} | {status:>8}")

    logger.info("-" * 60)
    logger.info(f"{'MEAN (valid params)':>25} | {mean_r2_test:>10.4f} |")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 8. Save checkpoint
    # ------------------------------------------------------------------
    ckpt_data = {
        "model_state_dict": model.state_dict(),
        "model_class": "HydroViTV2",
        "architecture": (
            "LocalCNNExtractor + CNNToViTAdapter + SatelliteEncoder (ViT-S/16) "
            "+ MultiScaleCNNAgg + DeepWQHeadV2"
        ),
        "split_protocol": "temporal-spatial holdout",
        "split_config": {
            "train": "2015-2022, folds A-C",
            "val": "2023, fold D",
            "test": "2024-2026, fold E",
        },
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "best_val_r2_phase1": float(best_r2_p1),
        "best_val_r2_phase2": float(best_r2_p2),
        "test_mean_r2": float(mean_r2_test),
        "seed": SEED,
        "normalization": {
            "mean": train_ds.mean.tolist(),
            "std": train_ds.std.tolist(),
        },
    }
    torch.save(ckpt_data, str(OUTPUT_CKPT))
    logger.info(f"\nSaved checkpoint: {OUTPUT_CKPT}")

    # ------------------------------------------------------------------
    # 9. Save results JSON
    # ------------------------------------------------------------------
    elapsed = time.time() - t0
    results = {
        "model": "HydroViT_v2",
        "architecture": (
            "LocalCNNExtractor + CNNToViTAdapter + SatelliteEncoder (ViT-S/16) "
            "+ MultiScaleCNNAgg + DeepWQHeadV2"
        ),
        "split_protocol": "temporal-spatial holdout (NOT random)",
        "split_config": {
            "temporal": {
                "train": "2015-2022",
                "val": "2023",
                "test": "2024-2026",
            },
            "spatial": {
                "train": "folds A-C",
                "val": "fold D",
                "test": "fold E",
                "method": "SHA-256 hash-based assignment",
            },
        },
        "split_statistics": {
            k: v for k, v in split_meta.items()
            if k not in ("param_coverage",)
        },
        "data_source": "REAL satellite tiles only (no synthetic)",
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "n_excluded": split_meta.get("n_excluded", 0),
        "hyperparameters": {
            "seed": SEED,
            "batch_size": BATCH_SIZE,
            "grad_accum": GRAD_ACCUM,
            "effective_batch_size": BATCH_SIZE * GRAD_ACCUM,
            "head_lr": HEAD_LR,
            "backbone_lr": BACKBONE_LR,
            "head_epochs": HEAD_EPOCHS,
            "finetune_epochs": FINETUNE_EPOCHS,
            "weight_decay": WEIGHT_DECAY,
            "grad_clip": GRAD_CLIP,
            "amp": USE_AMP,
            "early_stop_patience": EARLY_STOP_PATIENCE,
        },
        "training": {
            "best_val_r2_phase1": float(best_r2_p1),
            "best_val_r2_phase2": float(best_r2_p2),
        },
        "test_results": {
            "mean_r2": float(mean_r2_test),
            "water_temp_r2": float(wt_r2_test),
            "chl_a_r2": float(chl_r2_test),
            "per_param_r2": {
                k: (float(v) if not np.isnan(v) else None)
                for k, v in r2_test.items()
            },
            "test_loss": float(test_loss),
        },
        "leakage_check": split_meta["leakage_check"],
        "elapsed_seconds": round(elapsed, 1),
        "elapsed_minutes": round(elapsed / 60, 1),
        "timestamp": datetime.now().isoformat(),
    }

    with open(str(RESULTS_JSON), "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results: {RESULTS_JSON}")
    logger.info(f"Total elapsed: {elapsed/60:.1f} min")

    # Final summary
    logger.info("\n=== FINAL SUMMARY ===")
    logger.info(f"  Split: {n_train} train / {n_val} val / {n_test} test "
                f"(temporal-spatial holdout)")
    logger.info(f"  Test mean R2:     {mean_r2_test:.4f}")
    logger.info(f"  Test water_temp:  {wt_r2_test:.4f}")
    logger.info(f"  Test chl_a:       {chl_r2_test:.4f}")
    logger.info(f"  Leakage:          NONE (verified)")
    logger.info(f"  Checkpoint:       {OUTPUT_CKPT}")
    logger.info(f"  Results:          {RESULTS_JSON}")


if __name__ == "__main__":
    main()
