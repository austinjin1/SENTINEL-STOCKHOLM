"""WaterDroneNet: Image-only water quality prediction from drone/satellite imagery.

SENTINEL Mini — a lightweight, deployable model for drones equipped with
a dual-camera payload: Raspberry Pi Camera Module 3 Wide (RGB) + Raspberry Pi NoIR Camera
Module V2 (8MP, 1080P30) for near-infrared capture. Predicts 5 key water
quality parameters from 4-band (RGB + NIR) imagery alone, without
requiring any scalar sensor inputs.

Architecture:

1. **Inline ViT-S/16 encoder** — 12-layer, 6-head, 384-d vision transformer
   processing 4-channel 224×224 patches. No timm dependency; fully
   self-contained for edge deployment.

2. **Image-derived physics priors** — band statistics (means, stds, NDWI,
   NIR/Red ratio) → linear prior for each target. The model learns
   residuals on top of physics, not from scratch.

3. **Per-target Gaussian heads** — each target gets (mu, sigma) prediction
   with calibrated uncertainty via softplus parameterisation.

4. **Trust router** — per-sample trust flag combining embedding features
   and predicted uncertainty to flag low-confidence predictions.

Trained on real Sentinel-2 L2A patches paired with USGS NWIS water quality
measurements. Spatial holdout evaluation ensures generalization to
geographically unseen stations.

References
----------
- Benson & Krause (1984) for DO saturation curve.
- Bayesian deep learning uncertainty via softplus-parameterised sigma
  (Nix & Weigend, 1994).
- MAE: Masked Autoencoders Are Scalable Vision Learners (He et al., 2022).

MIT License — Bryan Cheng, 2026
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Number of input channels: Blue (B02), Green (B03), Red (B04), NIR (B08).
DRONE_IN_CHANS: int = 4

#: ViT-Small embedding dimension.
DRONE_EMBED_DIM: int = 384

#: Spatial image size expected by the encoder.
DRONE_IMG_SIZE: int = 224

#: ViT patch size.
VIT_PATCH_SIZE: int = 16

#: Output embedding dimension (SENTINEL shared space).
FUSED_DIM: int = 256

#: Number of scalar inputs (legacy compatibility, set to 0 for image-only).
NUM_SCALAR_INPUTS: int = 0

#: Water quality targets predicted by WaterDroneNet.
TARGET_PARAMS: Tuple[str, ...] = (
    "DO",       # Dissolved Oxygen (mg/L)
    "pH",       # pH units
    "Turb",     # Turbidity (NTU)
    "Temp",     # Temperature (°C)
    "SpCond",   # Specific Conductance (μS/cm)
)
NUM_TARGETS: int = len(TARGET_PARAMS)

#: Trust-score thresholds for the three-tier flag system.
TRUST_GREEN_THRESHOLD: float = 0.7
TRUST_RED_THRESHOLD: float = 0.3

#: Plausible value ranges for QA/QC filtering.
PARAM_RANGES: Dict[str, Tuple[float, float]] = {
    "DO":      (0.0,    20.0),
    "pH":      (4.0,    10.0),
    "Turb":    (0.0,    1000.0),
    "Temp":    (-5.0,   45.0),
    "SpCond":  (0.0,    50000.0),
}


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TargetPrediction:
    """Probabilistic prediction for a single water quality target.

    Attributes:
        mu: Predicted mean value, shape ``[B]``.
        sigma: Predicted aleatoric uncertainty (std), shape ``[B]``.
        physics_prior_value: Value of the physics prior for this target, shape ``[B]``.
    """
    mu: torch.Tensor
    sigma: torch.Tensor
    physics_prior_value: torch.Tensor


@dataclass
class WaterDroneNetOutput:
    """Full structured output from a WaterDroneNet forward pass.

    Attributes:
        predictions: Dict mapping target name → TargetPrediction.
        trust_scores: Dict mapping target name → trust score in [0, 1].
        trust_flags: Dict mapping target name → list of flag strings.
        fused_embedding: CLS token embedding, shape ``[B, 384]``.
    """
    predictions: Dict[str, TargetPrediction]
    trust_scores: Dict[str, torch.Tensor]
    trust_flags: Dict[str, List[str]]
    fused_embedding: torch.Tensor


# ---------------------------------------------------------------------------
# ViT building blocks (inline — no external dependency)
# ---------------------------------------------------------------------------

class _PatchEmbed(nn.Module):
    """Non-overlapping patch projection for 4-channel input."""

    def __init__(
        self,
        img_size: int = DRONE_IMG_SIZE,
        patch_size: int = VIT_PATCH_SIZE,
        in_chans: int = DRONE_IN_CHANS,
        embed_dim: int = DRONE_EMBED_DIM,
    ):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) → (B, N, D)"""
        return self.proj(x).flatten(2).transpose(1, 2)


class _TransformerBlock(nn.Module):
    """Pre-norm transformer block with multi-head self-attention."""

    def __init__(
        self,
        dim: int = DRONE_EMBED_DIM,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class WaterDroneNet(nn.Module):
    """WaterDroneNet: RGB+NIR imagery → water quality predictions.

    Pure vision model — no scalar sensor inputs required. Designed for
    deployment on drones with multispectral cameras (SENTINEL Mini).

    Architecture:
      1. Inline ViT-S/16 backbone (384d, 12 layers, 6 heads)
      2. Per-target prediction heads: CLS token → (mu, sigma) per target
      3. Image-derived physics priors from band statistics
      4. Trust router for per-sample confidence flagging

    Args:
        depth: Number of transformer layers. Default 12.
        num_heads: Number of attention heads per layer. Default 6.
        embed_dim: Embedding dimension. Default 384.
        dropout: Dropout rate. Default 0.1.

    Example:
        >>> model = WaterDroneNet()
        >>> images = torch.randn(2, 4, 224, 224)  # RGB+NIR
        >>> out = model(images)
        >>> out["mu"].shape  # (2, 5)
        >>> out["sigma"].shape  # (2, 5)
    """

    def __init__(
        self,
        depth: int = 12,
        num_heads: int = 6,
        embed_dim: int = DRONE_EMBED_DIM,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        # ViT backbone (ViT-S/16 equivalent)
        self.patch_embed = _PatchEmbed(embed_dim=embed_dim)
        self.num_patches = self.patch_embed.num_patches  # 196
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim)
        )
        self.blocks = nn.ModuleList([
            _TransformerBlock(embed_dim, num_heads, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Prediction head
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
        )

        # Per-target mu and sigma heads
        self.mu_heads = nn.ModuleList([
            nn.Linear(128, 1) for _ in range(NUM_TARGETS)
        ])
        self.sigma_heads = nn.ModuleList([
            nn.Linear(128, 1) for _ in range(NUM_TARGETS)
        ])

        # Image-based physics prior from band statistics
        # Input features: [mean_B, mean_G, mean_R, mean_NIR,
        #                   std_B, std_G, std_R, std_NIR, NDWI, NIR/Red]
        self.prior = nn.Linear(10, NUM_TARGETS, bias=True)
        nn.init.zeros_(self.prior.weight)
        nn.init.zeros_(self.prior.bias)

        # Trust router
        self.trust_router = nn.Sequential(
            nn.Linear(embed_dim + NUM_TARGETS, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def _encode_vision(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode image through ViT.

        Args:
            x: Input image tensor, shape ``[B, 4, 224, 224]``.

        Returns:
            Tuple of (cls_feat, patch_feats):
            - cls_feat: CLS token, shape ``[B, D]``.
            - patch_feats: Patch tokens, shape ``[B, N, D]``.
        """
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
        """Compute band statistics for physics prior.

        Args:
            images: Input images, shape ``[B, 4, H, W]``.

        Returns:
            Feature vector of shape ``[B, 10]``:
            [mean_B, mean_G, mean_R, mean_NIR, std_B, std_G, std_R, std_NIR,
             NDWI, NIR/Red ratio]
        """
        means = images.mean(dim=(-2, -1))  # (B, 4)
        stds = images.std(dim=(-2, -1))    # (B, 4)
        green, nir, red = means[:, 1], means[:, 3], means[:, 2]
        ndwi = (green - nir) / (green + nir + 1e-6)
        nir_red = nir / (red + 1e-6)
        return torch.cat([
            means, stds, ndwi.unsqueeze(1), nir_red.unsqueeze(1)
        ], dim=1)

    def forward(
        self, image: torch.Tensor, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass: image → water quality predictions.

        Args:
            image: RGB+NIR drone/satellite imagery, shape ``[B, 4, 224, 224]``.
            **kwargs: Ignored (for API compatibility).

        Returns:
            Dict with keys:
            - ``mu``: Predicted means, shape ``[B, 5]``.
            - ``sigma``: Predicted uncertainties, shape ``[B, 5]``.
            - ``physics_prior``: Physics prior values, shape ``[B, 5]``.
            - ``trust_logit``: Trust score logit, shape ``[B, 1]``.
            - ``cls_feat``: CLS token embedding, shape ``[B, D]``.
        """
        cls_feat, patch_feats = self._encode_vision(image)

        h = self.head(cls_feat)  # (B, 128)
        mu_residual = torch.cat(
            [head(h) for head in self.mu_heads], dim=1
        )  # (B, 5)
        log_sigma = torch.cat(
            [head(h) for head in self.sigma_heads], dim=1
        )  # (B, 5)
        sigma = F.softplus(log_sigma) + 1e-4

        img_feats = self._image_features(image)
        physics_prior = self.prior(img_feats)
        mu = physics_prior + mu_residual

        trust_logit = self.trust_router(
            torch.cat([cls_feat, sigma], dim=1)
        )

        return {
            "mu": mu,
            "sigma": sigma,
            "physics_prior": physics_prior,
            "trust_logit": trust_logit,
            "cls_feat": cls_feat,
        }

    def predict_structured(
        self, image: torch.Tensor
    ) -> WaterDroneNetOutput:
        """Forward pass returning structured output with trust flags.

        Args:
            image: Input imagery, shape ``[B, 4, 224, 224]``.

        Returns:
            WaterDroneNetOutput with per-target predictions and trust flags.
        """
        raw = self.forward(image)
        mu = raw["mu"]
        sigma = raw["sigma"]
        prior = raw["physics_prior"]
        trust = torch.sigmoid(raw["trust_logit"]).squeeze(-1)

        predictions = {}
        trust_scores = {}
        trust_flags = {}

        for i, name in enumerate(TARGET_PARAMS):
            predictions[name] = TargetPrediction(
                mu=mu[:, i],
                sigma=sigma[:, i],
                physics_prior_value=prior[:, i],
            )
            trust_scores[name] = trust
            flags = []
            for j in range(mu.size(0)):
                t = trust[j].item()
                if t >= TRUST_GREEN_THRESHOLD:
                    flags.append("green")
                elif t >= TRUST_RED_THRESHOLD:
                    flags.append("yellow")
                else:
                    flags.append("red")
            trust_flags[name] = flags

        return WaterDroneNetOutput(
            predictions=predictions,
            trust_scores=trust_scores,
            trust_flags=trust_flags,
            fused_embedding=raw["cls_feat"],
        )


# ---------------------------------------------------------------------------
# MAE Decoder for self-supervised pretraining
# ---------------------------------------------------------------------------

class MAEDecoder(nn.Module):
    """Lightweight decoder for MAE self-supervised pretraining.

    Reconstructs masked patches from visible patches, enabling
    unsupervised representation learning on unlabelled imagery.

    Args:
        embed_dim: Encoder embedding dimension.
        decoder_dim: Decoder embedding dimension.
        depth: Number of decoder transformer layers.
        num_patches: Total number of patches.
        patch_size: Spatial patch size.
        in_chans: Number of input channels.
    """

    def __init__(
        self,
        embed_dim: int = DRONE_EMBED_DIM,
        decoder_dim: int = 192,
        depth: int = 4,
        num_patches: int = 196,
        patch_size: int = VIT_PATCH_SIZE,
        in_chans: int = DRONE_IN_CHANS,
    ):
        super().__init__()
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.decoder_dim = decoder_dim

        self.encoder_proj = nn.Linear(embed_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, decoder_dim)
        )

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=decoder_dim,
                nhead=6,
                dim_feedforward=decoder_dim * 4,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(decoder_dim)
        self.pred = nn.Linear(
            decoder_dim, patch_size * patch_size * in_chans
        )

        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(
        self,
        vis_feats: torch.Tensor,
        mask_indices: torch.Tensor,
        visible_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruct all patches from visible features.

        Args:
            vis_feats: Visible patch features, shape ``[B, N_vis, D_enc]``.
            mask_indices: Indices of masked patches, shape ``[B, N_mask]``.
            visible_indices: Indices of visible patches, shape ``[B, N_vis]``.

        Returns:
            Predicted pixel values for all patches, shape ``[B, N, C*p*p]``.
        """
        B, N_vis, _ = vis_feats.shape
        N_mask = mask_indices.size(1)
        N = N_vis + N_mask

        vis = self.encoder_proj(vis_feats)

        tokens = torch.zeros(
            B, N, self.decoder_dim, device=vis.device, dtype=vis.dtype
        )
        vis_idx = visible_indices.unsqueeze(-1).expand(
            -1, -1, self.decoder_dim
        )
        tokens.scatter_(1, vis_idx, vis)

        mask_tok = self.mask_token.expand(B, N_mask, -1).to(dtype=vis.dtype)
        mask_idx = mask_indices.unsqueeze(-1).expand(
            -1, -1, self.decoder_dim
        )
        tokens.scatter_(1, mask_idx, mask_tok)

        tokens = tokens + self.pos_embed[:, :N]

        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        return self.pred(tokens)
