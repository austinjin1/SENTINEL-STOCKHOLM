#!/usr/bin/env python3
"""HydroViT v9 — CNN-ViT hybrid with per-parameter band attention.

Key improvements over v8 (mean R²=0.662, Chl-a=0.364):
  1. CNN-ViT hybrid: 3-layer stride-1 CNN (32→64→128ch) before ViT encoder.
     CNN captures fine-grained local texture for algal bloom patches.
     ViT handles global long-range dependencies. Multi-scale feature fusion
     from CNN + ViT CLS concatenated → richer representation.
  2. Per-parameter band attention: learnable band weights per output param.
     Bands 3,4,5 (green, red, red-edge) physically linked to Chl-a.
  3. Weighted loss: Chl-a 3×, turbidity/phycocyanin 2×, water_temp 3×.
  4. Deeper head with skip connections: 512→384→256 with two residuals.
  5. 120 epochs phase 2 cosine annealing (up from 80 effective in v8).
  6. Gradient accumulation: effective batch = 32 (4 accum × batch 8).

Architecture motivation:
  - DenseNet121 wins on Chl-a (0.781 vs 0.364) because dense convolutions
    build rich multi-scale local features at every layer. ViT-S/16 patches
    (16×16 = 160m ground footprint) are too coarse for algal bloom patches.
  - Prepending a 3-layer CNN (all stride-1, so no spatial downsampling)
    projects the 10-band input into a 128-channel locally-contextual feature
    map at the same resolution. The CNN output is then treated as a new
    "spectral+spatial" input to the ViT patch embedder via an adapter.
  - Multi-scale CNN features (32, 64, 128 channels) are pooled and
    concatenated to the ViT CLS token before the regression head.

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
# Use cuda:2 — it has the most free memory (18432 MiB used vs 81920 MiB total)
DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
CKPT_DIR   = Path("checkpoints/satellite")
RESULTS_DIR = Path("results/benchmarks")
LOGS_DIR   = Path("logs")
CKPT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

PAIRED_DATA = Path("data/processed/satellite/paired_wq_v5.npz")
if not PAIRED_DATA.exists():
    PAIRED_DATA = Path("data/processed/satellite/paired_wq_v3.npz")
    logger.warning("v5 not found, falling back to v3")

PRETRAINED_CKPT = CKPT_DIR / "hydrovit_real_mae.pt"
OUTPUT_CKPT     = CKPT_DIR / "hydrovit_wq_v9.pt"
RESULTS_JSON    = RESULTS_DIR / "hydrovit_v9_results.json"
LOG_FILE        = LOGS_DIR / "train_hydrovit_v9.log"

SEED            = 42
BATCH_SIZE      = 8
GRAD_ACCUM      = 4        # effective batch = 32
HEAD_LR         = 3e-4
BACKBONE_LR     = 3e-5
HEAD_EPOCHS     = 80
FINETUNE_EPOCHS = 120
WEIGHT_DECAY    = 0.01
GRAD_CLIP       = 1.0
USE_AMP         = True

# ─── Per-parameter loss weights ────────────────────────────────────────────
# Optically-active params that need the most improvement:
#   chl_a (idx 0): 3×  — critical -0.417 gap vs DenseNet
#   turbidity (idx 1): 2×
#   phycocyanin (idx 12): 2×
#   water_temp (idx 11): 3×  — preserve strong performance
WATER_TEMP_IDX  = PARAM_NAMES.index("water_temp") if "water_temp" in PARAM_NAMES else 11
CHL_A_IDX       = PARAM_NAMES.index("chl_a")       if "chl_a"       in PARAM_NAMES else 0
TURBIDITY_IDX   = PARAM_NAMES.index("turbidity")   if "turbidity"   in PARAM_NAMES else 1
PHYCOCYANIN_IDX = PARAM_NAMES.index("phycocyanin") if "phycocyanin" in PARAM_NAMES else 12

PARAM_WEIGHTS = torch.ones(NUM_WATER_PARAMS)
PARAM_WEIGHTS[WATER_TEMP_IDX]  = 3.0
PARAM_WEIGHTS[CHL_A_IDX]       = 3.0   # critical gap
PARAM_WEIGHTS[TURBIDITY_IDX]   = 2.0
PARAM_WEIGHTS[PHYCOCYANIN_IDX] = 2.0

torch.manual_seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Local CNN feature extractor (stride-1, no spatial downsampling)
# ---------------------------------------------------------------------------
class LocalCNNExtractor(nn.Module):
    """3-layer stride-1 CNN that builds multi-scale local features.

    Input:  [B, in_chans, H, W]   (10 bands)
    Output: [B, 128, H, W]  + intermediate feature maps at 32 and 64 ch

    All convolutions are stride-1 with padding=1 so spatial dimensions
    are fully preserved.  Depthwise-separable conv at the end keeps
    parameters low while maximising receptive field.

    Receptive field:
      Layer 1: 3×3  → rf = 3
      Layer 2: 3×3  → rf = 5
      Layer 3: 5×5  → rf = 9
    At 10m/px this covers a 90m neighbourhood — well-matched to algal
    patch scales (tens to hundreds of metres).
    """

    def __init__(self, in_chans: int = 10):
        super().__init__()
        # Layer 1: 10 → 32
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_chans, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        # Layer 2: 32 → 64
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        # Layer 3: 64 → 128 (depthwise-separable for efficiency)
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
        """
        Returns:
            out128: [B, 128, H, W]
            feats:  list of intermediate feature maps [32ch, 64ch]
        """
        f1 = self.conv1(x)          # [B, 32, H, W]
        f2 = self.conv2(f1)         # [B, 64, H, W]
        f3 = self.conv3_pw(self.conv3_dw(f2))  # [B, 128, H, W]
        return f3, [f1, f2]


# ---------------------------------------------------------------------------
# CNN → ViT adapter (projects 128-ch map to 13-ch for SatelliteEncoder)
# ---------------------------------------------------------------------------
class CNNToViTAdapter(nn.Module):
    """Project CNN 128-ch feature map back to the ViT input space.

    We run the ViT encoder on a fused representation: original 10-band
    image + CNN 128-ch feature map blended via a learned 1×1 projection
    down to 13 channels (the SatelliteEncoder input format).

    This lets the ViT 'see' CNN-extracted local context alongside the raw
    spectral signal, at no extra spatial resolution cost.
    """

    def __init__(self, cnn_out_channels: int = 128, in_chans: int = 10, vit_in_chans: int = 13):
        super().__init__()
        combined = in_chans + cnn_out_channels   # 10 + 128 = 138
        self.proj = nn.Sequential(
            nn.Conv2d(combined, vit_in_chans, kernel_size=1, bias=False),
            nn.BatchNorm2d(vit_in_chans),
        )
        # learnable blend weight — start at 0.5 (equal mix)
        self.blend = nn.Parameter(torch.tensor(0.5))

    def forward(self, x_raw: torch.Tensor, x_cnn: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_raw: [B, 10, H, W]  original spectral input
            x_cnn: [B, 128, H, W] CNN output
        Returns:
            fused: [B, 13, H, W]  ready for SatelliteEncoder
        """
        combined = torch.cat([x_raw, x_cnn], dim=1)  # [B, 138, H, W]
        return self.proj(combined)                    # [B, 13, H, W]


# ---------------------------------------------------------------------------
# Multi-scale pooled CNN features → fixed-dim vector
# ---------------------------------------------------------------------------
class MultiScaleCNNAggregator(nn.Module):
    """Global avg-pool each CNN feature map and project to embed_dim.

    Feature maps:  [B, 32, H, W], [B, 64, H, W], [B, 128, H, W]
    After pooling: [B, 32], [B, 64], [B, 128]  → concat → [B, 224]
    Project:       [B, 224] → [B, embed_dim]
    """

    def __init__(self, embed_dim: int = 256):
        super().__init__()
        # 32 + 64 + 128 = 224
        self.proj = nn.Sequential(
            nn.Linear(224, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, feats: list[torch.Tensor], out128: torch.Tensor) -> torch.Tensor:
        pooled = [F.adaptive_avg_pool2d(f, 1).flatten(1) for f in feats]
        pooled.append(F.adaptive_avg_pool2d(out128, 1).flatten(1))
        x = torch.cat(pooled, dim=1)   # [B, 224]
        return self.proj(x)            # [B, embed_dim]


# ---------------------------------------------------------------------------
# Per-parameter band attention
# ---------------------------------------------------------------------------
class PerParamBandAttention(nn.Module):
    """Learnable band weighting per output parameter.

    Physically motivated initialization:
      - chl_a (idx 0): bands 2,3,4 (green, red, red-edge) weighted high
      - turbidity (idx 1): all bands roughly equal
      - phycocyanin (idx 12): bands 4,5,6 (red, red-edge, NIR) weighted high
      Others: uniform init.

    Output: soft-weighted input x' = x * weights[param_idx, :] broadcast
    over spatial dims. This is applied before the CNN extractor.

    Note: for the WQ head we also concatenate these per-param projections.
    """

    def __init__(self, in_chans: int = 10, n_params: int = 16):
        super().__init__()
        # [n_params, in_chans] learnable weights
        w = torch.ones(n_params, in_chans)
        # Physics-informed init: Chl-a (idx 0) — boost green (2), red (3), red-edge (4)
        w[0, 2] = 2.0; w[0, 3] = 2.0; w[0, 4] = 2.5   # green, red, red-edge
        # Turbidity (idx 1) — boost NIR (6), SWIR1 (8)
        w[1, 6] = 1.5; w[1, 8] = 1.5
        # Phycocyanin (idx 12) — boost red (3), red-edge (4), NIR (5,6)
        w[12, 3] = 1.5; w[12, 4] = 2.0; w[12, 5] = 2.0; w[12, 6] = 1.5
        # Water temp (idx 11) — SWIR more informative
        w[11, 8] = 1.5; w[11, 9] = 1.5
        self.band_weights = nn.Parameter(w)  # learnable from init

    def get_channel_gate(self) -> torch.Tensor:
        """Return softmax-normalized band attention gate for global use.

        Returns [in_chans] — average across all params (for the shared CNN).
        """
        # Average across params, then softmax-normalize
        avg_w = self.band_weights.mean(0)    # [in_chans]
        return torch.softmax(avg_w, dim=0) * avg_w.shape[0]  # scale to ~1 mean

    def get_per_param_weights(self) -> torch.Tensor:
        """Return [n_params, in_chans] softmax weights for regression head use."""
        return torch.softmax(self.band_weights, dim=1)  # softmax over bands


# ---------------------------------------------------------------------------
# Deep WQ Head v2 — CNN + ViT dual-stream input
# ---------------------------------------------------------------------------
class DeepWQHeadV2(nn.Module):
    """Dual-stream head: ViT embedding + CNN multi-scale aggregation.

    Streams:
      1. ViT stream: 256-dim projected ViT embedding
      2. CNN stream: 256-dim multi-scale CNN features

    Fusion: concat → 512 → residual blocks → 16 outputs.
    """

    def __init__(
        self,
        vit_dim: int = 256,
        cnn_dim: int = 256,
        hidden: int = 512,
        n_params: int = 16,
    ):
        super().__init__()
        fused_dim = vit_dim + cnn_dim  # 512

        self.bn_in = nn.LayerNorm(fused_dim)

        # Block 1: 512 → 512
        self.fc1  = nn.Linear(fused_dim, hidden)
        self.bn1  = nn.LayerNorm(hidden)
        self.drop1 = nn.Dropout(0.15)

        # Block 2: 512 → 384
        self.fc2  = nn.Linear(hidden, 384)
        self.bn2  = nn.LayerNorm(384)
        self.drop2 = nn.Dropout(0.1)

        # Block 3: 384 → 256
        self.fc3  = nn.Linear(384, 256)
        self.bn3  = nn.LayerNorm(256)

        # Residual connections
        self.res1 = nn.Linear(fused_dim, hidden)  # 512 → 512
        self.res2 = nn.Linear(hidden, 384)        # 512 → 384
        self.res3 = nn.Linear(384, 256)           # 384 → 256

        self.out  = nn.Linear(256, n_params)

    def forward(self, vit_emb: torch.Tensor, cnn_emb: torch.Tensor) -> torch.Tensor:
        x = torch.cat([vit_emb, cnn_emb], dim=1)   # [B, 512]
        x = self.bn_in(x)

        # Block 1
        h = F.gelu(self.bn1(self.fc1(x)))
        h = self.drop1(h)
        h = h + self.res1(x)                        # residual

        # Block 2
        h2 = F.gelu(self.bn2(self.fc2(h)))
        h2 = self.drop2(h2)
        h2 = h2 + self.res2(h)                      # residual

        # Block 3
        h3 = F.gelu(self.bn3(self.fc3(h2)))
        h3 = h3 + self.res3(h2)                     # residual

        return self.out(h3)


# ---------------------------------------------------------------------------
# Full HydroViT v9 model
# ---------------------------------------------------------------------------
class HydroViTV9(nn.Module):
    """CNN-ViT hybrid for water quality estimation.

    Pipeline:
      raw_10ch
        → PerParamBandAttention (shared gate)
        → LocalCNNExtractor  → multi-scale features [32, 64, 128]
        → CNNToViTAdapter    → 13ch fused map
        → SatelliteEncoder   → ViT CLS embedding [256]
        → MultiScaleCNNAggregator → CNN embedding [256]
        → DeepWQHeadV2       → [16] predictions
    """

    def __init__(self, pretrained_ckpt: Path | None = None, in_bands: int = 10):
        super().__init__()
        self.in_bands = in_bands

        # Per-parameter band attention (shared gate for CNN path)
        self.band_attn = PerParamBandAttention(in_chans=in_bands)

        # Local CNN feature extractor (stride-1, no downsampling)
        self.cnn = LocalCNNExtractor(in_chans=in_bands)

        # CNN → ViT adapter (projects combined 138ch → 13ch for ViT)
        self.adapter = CNNToViTAdapter(cnn_out_channels=128, in_chans=in_bands, vit_in_chans=13)

        # ViT encoder (SatelliteEncoder, projects to 256-dim)
        self.encoder = SatelliteEncoder()

        # Multi-scale CNN feature aggregator → 256
        self.cnn_agg = MultiScaleCNNAggregator(embed_dim=self.encoder.shared_embed_dim)

        # Dual-stream WQ head
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
            x: [B, 10, H, W]  (10-band Sentinel-2)
        Returns:
            preds: [B, 16]
        """
        # 1. Apply shared band gate (global average across params)
        gate = self.band_attn.get_channel_gate()          # [10]
        x_gated = x * gate.view(1, -1, 1, 1)             # [B, 10, H, W]

        # 2. CNN multi-scale features
        cnn_out, cnn_feats = self.cnn(x_gated)            # [B, 128, H, W], [[32], [64]]

        # 3. Fuse CNN output with raw input → ViT input (13ch)
        x13 = self.adapter(x_gated, cnn_out)              # [B, 13, H, W]

        # 4. ViT encoder → 256-dim embedding
        enc_out = self.encoder(x13)
        vit_emb = enc_out["embedding"]                    # [B, 256]

        # 5. Multi-scale CNN aggregation → 256-dim
        cnn_emb = self.cnn_agg(cnn_feats, cnn_out)        # [B, 256]

        # 6. Dual-stream WQ head
        return self.wq_head(vit_emb, cnn_emb)             # [B, 16]


# ---------------------------------------------------------------------------
# Dataset (identical to v8 — same split, same normalization)
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
# Loss + metrics (identical interface to v8)
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
# Training phase (with gradient accumulation)
# ---------------------------------------------------------------------------
def train_phase(model, train_dl, val_dl, optimizer, scheduler, epochs, label, device,
                grad_accum: int = 1):
    best_mean_r2 = -float("inf")
    best_wt_r2   = -float("inf")
    best_state   = None
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
                loss = loss / grad_accum   # scale for accumulation

            if torch.isnan(loss):
                optimizer.zero_grad(); continue

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

        if (ep + 1) % 10 == 0 or ep == epochs - 1:
            r2, val_loss = evaluate(model, val_dl, device)
            valid_r2s = [v for v in r2.values() if not np.isnan(v)]
            mean_r2 = np.mean(valid_r2s) if valid_r2s else -1.0
            wt_r2   = r2.get("water_temp", float("nan"))
            chl_r2  = r2.get("chl_a", float("nan"))
            logger.info(
                f"[{label}] Ep {ep+1:3d}/{epochs} | "
                f"tr={total_loss/max(n_b,1):.4f} val={val_loss:.4f} | "
                f"mean_R2={mean_r2:.4f} wt={wt_r2:.4f} chl_a={chl_r2:.4f}"
            )
            # Save best by mean R² (not just water_temp) to improve all params
            if not np.isnan(mean_r2) and mean_r2 > best_mean_r2:
                best_mean_r2 = mean_r2
                best_wt_r2   = wt_r2
                best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                logger.info(f"  -> New best mean R2: {best_mean_r2:.4f} (wt={best_wt_r2:.4f})")

    if best_state:
        model.load_state_dict(best_state)
    return best_wt_r2, best_mean_r2


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("HydroViT v9 — CNN-ViT hybrid + per-param band attention")
    logger.info(f"  Data:   {PAIRED_DATA}")
    logger.info(f"  Device: {DEVICE}")
    logger.info(f"  Grad accum: {GRAD_ACCUM} (effective batch={BATCH_SIZE * GRAD_ACCUM})")
    logger.info("=" * 70)

    # ── Data split (IDENTICAL to v8) ─────────────────────────────────────
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

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True, persistent_workers=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True, persistent_workers=True)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True, persistent_workers=True)

    # ── Model ─────────────────────────────────────────────────────────────
    model = HydroViTV9(pretrained_ckpt=PRETRAINED_CKPT).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {n_params:,} trainable parameters")

    # Component counts
    cnn_params  = sum(p.numel() for p in list(model.cnn.parameters()) +
                      list(model.adapter.parameters()) +
                      list(model.cnn_agg.parameters()))
    vit_params  = sum(p.numel() for p in model.encoder.parameters())
    head_params = sum(p.numel() for p in model.wq_head.parameters())
    logger.info(f"  CNN extractor+adapter+agg: {cnn_params:,}")
    logger.info(f"  ViT encoder:               {vit_params:,}")
    logger.info(f"  WQ head:                   {head_params:,}")

    # ── Phase 1: Train CNN + head (freeze ViT backbone) ──────────────────
    logger.info("\n--- Phase 1: CNN + head training (ViT frozen) ---")
    for p in model.encoder.parameters():
        p.requires_grad_(False)

    phase1_params = (
        list(model.band_attn.parameters()) +
        list(model.cnn.parameters()) +
        list(model.adapter.parameters()) +
        list(model.cnn_agg.parameters()) +
        list(model.wq_head.parameters())
    )
    opt1 = torch.optim.AdamW(
        [p for p in phase1_params if p.requires_grad],
        lr=HEAD_LR, weight_decay=WEIGHT_DECAY
    )
    sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=HEAD_EPOCHS, eta_min=1e-5)
    best_wt1, best_mean1 = train_phase(
        model, train_dl, val_dl, opt1, sch1, HEAD_EPOCHS, "Phase1-CNN+Head",
        DEVICE, grad_accum=GRAD_ACCUM
    )
    logger.info(f"Phase 1 best: water_temp R2={best_wt1:.4f}, mean R2={best_mean1:.4f}")

    # ── Phase 2: Full fine-tune (unfreeze ViT, lower LR) ─────────────────
    logger.info("\n--- Phase 2: Full fine-tune ---")
    for p in model.encoder.parameters():
        p.requires_grad_(True)

    opt2 = torch.optim.AdamW([
        {"params": model.band_attn.parameters(),  "lr": HEAD_LR},
        {"params": model.cnn.parameters(),        "lr": HEAD_LR},
        {"params": model.adapter.parameters(),    "lr": HEAD_LR},
        {"params": model.cnn_agg.parameters(),    "lr": HEAD_LR},
        {"params": model.wq_head.parameters(),    "lr": HEAD_LR},
        {"params": model.encoder.parameters(),    "lr": BACKBONE_LR},
    ], weight_decay=WEIGHT_DECAY)
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=FINETUNE_EPOCHS, eta_min=1e-6)
    best_wt2, best_mean2 = train_phase(
        model, train_dl, val_dl, opt2, sch2, FINETUNE_EPOCHS, "Phase2-Finetune",
        DEVICE, grad_accum=GRAD_ACCUM
    )
    logger.info(f"Phase 2 best: water_temp R2={best_wt2:.4f}, mean R2={best_mean2:.4f}")

    # ── Final test evaluation ──────────────────────────────────────────────
    logger.info("\n--- Final test evaluation ---")
    r2_test, _ = evaluate(model, test_dl, DEVICE)
    valid_r2s = [v for v in r2_test.values() if not np.isnan(v)]
    mean_r2   = float(np.mean(valid_r2s)) if valid_r2s else -1.0
    wt_r2     = r2_test.get("water_temp", float("nan"))

    # ── Comparison table ──────────────────────────────────────────────────
    densenet_ref = {
        "chl_a": 0.7806, "turbidity": 0.7783, "tss": 0.6634,
        "total_nitrogen": 0.713, "total_phosphorus": 0.7584,
        "dissolved_oxygen": 0.7721, "ammonia": 0.5783, "nitrate": 0.822,
        "ph": 0.6352, "water_temp": 0.884, "phycocyanin": 0.3464,
    }
    hydrovit_v8 = {
        "chl_a": 0.364, "turbidity": 0.733, "tss": 0.747,
        "total_nitrogen": 0.632, "total_phosphorus": 0.738,
        "dissolved_oxygen": 0.772, "ammonia": 0.557, "nitrate": 0.746,
        "ph": 0.657, "water_temp": 0.872, "phycocyanin": 0.467,
    }

    logger.info("\n" + "=" * 80)
    logger.info(f"{'Parameter':>25} | {'DenseNet121':>11} | {'HydroViT v8':>11} | "
                f"{'HydroViT v9':>11} | {'v9 vs DN':>9}")
    logger.info("-" * 80)
    for name, r2v9 in r2_test.items():
        if np.isnan(r2v9):
            continue
        dn  = densenet_ref.get(name, float("nan"))
        v8  = hydrovit_v8.get(name,  float("nan"))
        gap = r2v9 - dn if not np.isnan(dn) else float("nan")
        gap_str = f"{gap:+.4f}" if not np.isnan(gap) else "   N/A"
        dn_str  = f"{dn:.4f}"   if not np.isnan(dn) else "   N/A"
        v8_str  = f"{v8:.4f}"   if not np.isnan(v8) else "   N/A"
        marker  = " ***" if name == "water_temp" else (" +++" if gap > 0.02 else "")
        logger.info(
            f"{name:>25} | {dn_str:>11} | {v8_str:>11} | "
            f"{r2v9:>11.4f} | {gap_str:>9}{marker}"
        )
    logger.info("-" * 80)
    logger.info(f"{'MEAN (valid params)':>25} | {'0.7029':>11} | {'0.6622':>11} | "
                f"{mean_r2:>11.4f} | {'':>9}")
    logger.info("=" * 80)

    # ── Save checkpoint ────────────────────────────────────────────────────
    torch.save(model.state_dict(), str(OUTPUT_CKPT))
    logger.info(f"\nSaved checkpoint: {OUTPUT_CKPT}")

    elapsed = time.time() - t0
    results = {
        "model": "HydroViT_v9",
        "architecture": "LocalCNNExtractor + CNNToViTAdapter + SatelliteEncoder (ViT-S/16) + MultiScaleCNNAgg + DeepWQHeadV2",
        "key_changes_vs_v8": [
            "CNN-ViT hybrid: 3-layer stride-1 CNN (10→32→64→128ch) before ViT",
            "CNNToViTAdapter: fuses raw spectral + CNN features into 13ch ViT input",
            "MultiScaleCNNAggregator: pools all 3 CNN feature maps into 256-dim",
            "DeepWQHeadV2: dual-stream (ViT 256 + CNN 256) with 3 residual blocks",
            "PerParamBandAttention: physics-initialized band weights (Chl-a: bands 2,3,4)",
            "Weighted loss: chl_a=3x, water_temp=3x, turbidity=2x, phycocyanin=2x",
            "120 epochs fine-tune (vs 80 in v8) with cosine annealing",
            "Grad accum 4x (effective batch=32)",
            "Best model saved by mean R² across all params (not just water_temp)",
        ],
        "n_params": n_params,
        "data": str(PAIRED_DATA),
        "n_train": int(len(train_idx)),
        "n_val":   int(len(val_idx)),
        "n_test":  int(len(test_idx)),
        "water_temp_r2": float(wt_r2),
        "mean_r2": float(mean_r2),
        "per_param_r2": {k: (float(v) if not np.isnan(v) else None) for k, v in r2_test.items()},
        "best_val_wt_r2_phase1": float(best_wt1),
        "best_val_mean_r2_phase1": float(best_mean1),
        "best_val_wt_r2_phase2": float(best_wt2),
        "best_val_mean_r2_phase2": float(best_mean2),
        "vs_densenet121": {
            "densenet_mean_r2": 0.7029,
            "densenet_water_temp_r2": 0.884,
            "delta_mean_r2": round(float(mean_r2) - 0.7029, 4),
            "delta_water_temp_r2": round(float(wt_r2) - 0.884, 4),
        },
        "vs_hydrovit_v8": {
            "v8_mean_r2": 0.6622,
            "v8_water_temp_r2": 0.8716,
            "delta_mean_r2": round(float(mean_r2) - 0.6622, 4),
            "delta_water_temp_r2": round(float(wt_r2) - 0.8716, 4),
        },
        "elapsed_seconds": round(elapsed, 1),
    }

    with open(str(RESULTS_JSON), "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results: {RESULTS_JSON}")
    logger.info(f"Total elapsed: {elapsed/60:.1f} min")

    # Quick summary
    logger.info("\n=== FINAL SUMMARY ===")
    logger.info(f"  water_temp R²:  {wt_r2:.4f}  (v8=0.8716, DenseNet=0.884)")
    logger.info(f"  mean R²:        {mean_r2:.4f}  (v8=0.6622, DenseNet=0.7029)")
    logger.info(f"  chl_a R²:       {r2_test.get('chl_a', float('nan')):.4f}  (v8=0.364, DenseNet=0.7806)")


if __name__ == "__main__":
    main()
