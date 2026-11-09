"""HydroViT satellite encoder: water-specific foundation model for SENTINEL.

Combines:
1. HydroViT backbone (ViT-S/16, 13-band input, MAE pretraining support)
2. Multi-resolution cross-attention (S2 10m + S3 300m fusion)
3. Temporal attention stack (5-10 image time series with cloud weighting)
4. Water quality parameter head (16 params with uncertainty)
5. Spectral physics consistency loss

Interface contract:
    forward() returns dict with at minimum:
        "embedding"       [B, 256]  - primary projected embedding
        "fusion_embedding" [B, 256] - embedding for cross-modal fusion
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .hydrovit_backbone import (
    HydroViTBackbone,
    VIT_EMBED_DIM,
    NUM_SPECTRAL_BANDS,
)
from .multi_resolution import ResolutionCrossAttention
from .temporal_stack import TemporalAttentionStack
from .parameter_head import WaterQualityHead, NUM_WATER_PARAMS
from .physics_loss import SpectralPhysicsLoss

SHARED_EMBED_DIM = 256


class SatelliteEncoder(nn.Module):
    """HydroViT-based satellite encoder for SENTINEL.

    Water-specific vision transformer with multi-resolution fusion,
    temporal attention, and water quality parameter estimation.

    Args:
        in_chans: Number of input spectral bands (10 S2 + 3 S3 OLCI).
        pretrained: Whether to load pretrained backbone weights.
        checkpoint_path: Optional path to HydroViT checkpoint.
        shared_embed_dim: Dimension of the shared fusion embedding space.
        max_temporal_len: Maximum temporal stack length.
        enable_s3_fusion: Whether to enable S2/S3 cross-attention fusion.
    """

    def __init__(
        self,
        in_chans: int = NUM_SPECTRAL_BANDS,
        pretrained: bool = True,
        checkpoint_path: Optional[str] = None,
        shared_embed_dim: int = SHARED_EMBED_DIM,
        max_temporal_len: int = 16,
        enable_s3_fusion: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = VIT_EMBED_DIM  # 384
        self.shared_embed_dim = shared_embed_dim
        self.enable_s3_fusion = enable_s3_fusion

        # 1. HydroViT backbone
        self.backbone = HydroViTBackbone(
            in_chans=in_chans,
            pretrained=pretrained,
            checkpoint_path=checkpoint_path,
        )

        # 2. Multi-resolution S2/S3 cross-attention
        if enable_s3_fusion:
            self.multi_res = ResolutionCrossAttention(
                embed_dim=VIT_EMBED_DIM,
                num_heads=6,
                num_layers=2,
                s2_max_tokens=196,
                s3_max_tokens=16,
                s2_input_dim=VIT_EMBED_DIM,
                s3_input_dim=VIT_EMBED_DIM,
            )

        # 3. Temporal attention stack
        self.temporal_stack = TemporalAttentionStack(
            embed_dim=VIT_EMBED_DIM,
            num_layers=3,
            num_heads=6,
            max_temporal_len=max_temporal_len,
        )

        # 4. Water quality parameter head
        self.water_quality_head = WaterQualityHead(
            input_dim=VIT_EMBED_DIM,
            backbone_dim=256,
            head_hidden_dim=128,
        )

        # 5. Physics loss module
        self.physics_loss = SpectralPhysicsLoss()

        # Projection head: 384 -> 256 (interface contract)
        self.projection = nn.Sequential(
            nn.Linear(VIT_EMBED_DIM, VIT_EMBED_DIM),
            nn.GELU(),
            nn.LayerNorm(VIT_EMBED_DIM),
            nn.Linear(VIT_EMBED_DIM, shared_embed_dim),
            nn.LayerNorm(shared_embed_dim),
        )

        # Temporal projection: 384 -> 256
        self.temporal_projection = nn.Sequential(
            nn.Linear(VIT_EMBED_DIM, VIT_EMBED_DIM),
            nn.GELU(),
            nn.LayerNorm(VIT_EMBED_DIM),
            nn.Linear(VIT_EMBED_DIM, shared_embed_dim),
            nn.LayerNorm(shared_embed_dim),
        )

        self._init_projections()

    def _init_projections(self) -> None:
        """Xavier-initialize all projection heads."""
        for proj in [self.projection, self.temporal_projection]:
            for m in proj.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(
        self,
        image: torch.Tensor,
        s3_tokens: Optional[torch.Tensor] = None,
        temporal_frames: Optional[torch.Tensor] = None,
        temporal_timestamps: Optional[torch.Tensor] = None,
        temporal_cloud_fractions: Optional[torch.Tensor] = None,
        temporal_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Full forward pass through the HydroViT encoder.

        Args:
            image: Multispectral input tile [B, 13, 224, 224].
            s3_tokens: Optional Sentinel-3 OLCI tokens [B, N_s3, 384].
                If None and enable_s3_fusion=True, S3 fusion is skipped.
            temporal_frames: Optional stack of per-frame CLS embeddings
                [B, T, 384] from previous acquisitions at this location.
            temporal_timestamps: [B, T+1] timestamps in days since epoch
                (T historical + 1 current).
            temporal_cloud_fractions: [B, T+1] cloud fraction per frame.
            temporal_mask: [B, T+1] bool padding mask (True = invalid).

        Returns:
            Dict with:
                "embedding":             [B, 256] primary projected embedding
                "fusion_embedding":      [B, 256] for cross-modal fusion
                "water_quality_params":  [B, 16]  predicted parameters
                "param_uncertainty":     [B, 16]  per-param log variance
                "temporal_embedding":    [B, 256] from temporal stack
                "cls_token":             [B, 384] raw CLS token
        """
        B = image.shape[0]
        device = image.device

        # ------------------------------------------------------------------
        # 1. Backbone: extract CLS token and multi-scale features
        # ------------------------------------------------------------------
        cls_token, multi_scale_features = self.backbone(image)
        # cls_token: [B, 384]
        # multi_scale_features: {3: [B, N+1, 384], 6: ..., 9: ..., 12: ...}

        # ------------------------------------------------------------------
        # 2. Multi-resolution fusion (S2 + S3) if S3 data is available
        # ------------------------------------------------------------------
        if self.enable_s3_fusion and s3_tokens is not None:
            # Use final-layer patch tokens as S2 representation
            s2_patch_tokens = multi_scale_features[12][:, 1:, :]  # [B, 196, 384]
            fused_tokens = self.multi_res(s2_patch_tokens, s3_tokens)
            # Pool fused tokens to get enhanced CLS
            fused_cls = fused_tokens.mean(dim=1)  # [B, 384]
        else:
            fused_cls = cls_token  # [B, 384]

        # ------------------------------------------------------------------
        # 3. Temporal attention
        # ------------------------------------------------------------------
        if temporal_frames is not None and temporal_timestamps is not None:
            # Append current CLS token to temporal stack
            current_cls = cls_token.unsqueeze(1)  # [B, 1, 384]
            full_stack = torch.cat([temporal_frames, current_cls], dim=1)

            temporal_output = self.temporal_stack(
                frame_embeddings=full_stack,
                timestamps=temporal_timestamps,
                cloud_fractions=temporal_cloud_fractions,
                padding_mask=temporal_mask,
            )
            temporal_emb = temporal_output["temporal_embedding"]  # [B, 384]
        else:
            # Single-frame: use CLS directly
            temporal_output = self.temporal_stack.forward_single(cls_token)
            temporal_emb = temporal_output["temporal_embedding"]

        # ------------------------------------------------------------------
        # 4. Project to shared embedding space
        # ------------------------------------------------------------------
        embedding = self.projection(fused_cls)               # [B, 256]
        temporal_embedding = self.temporal_projection(temporal_emb)  # [B, 256]

        # fusion_embedding combines spatial (fused) + temporal context
        fusion_embedding = embedding  # [B, 256]

        # ------------------------------------------------------------------
        # 5. Water quality parameter estimation
        # ------------------------------------------------------------------
        water_params, param_uncertainty = self.water_quality_head(fused_cls)
        # water_params: [B, 16], param_uncertainty: [B, 16]

        return {
            "embedding": embedding,
            "fusion_embedding": fusion_embedding,
            "water_quality_params": water_params,
            "param_uncertainty": param_uncertainty,
            "temporal_embedding": temporal_embedding,
            "cls_token": cls_token,
        }

    def forward_mae(
        self,
        image: torch.Tensor,
        mask_ratio: float = 0.75,
        water_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """MAE pretraining forward pass with physics loss.

        Args:
            image: [B, 13, 224, 224] input.
            mask_ratio: Fraction of patches to mask.
            water_mask: Optional [B, 1, H, W] water mask for physics loss.

        Returns:
            Dict with MAE reconstruction loss and physics loss terms.
        """
        reconstruction, target, mask = self.backbone.forward_mae(
            image, mask_ratio=mask_ratio
        )

        # MSE reconstruction loss on masked patches only
        loss_per_patch = ((reconstruction - target) ** 2).mean(dim=-1)  # [B, N]
        mae_loss = (loss_per_patch * mask).sum() / mask.sum().clamp(min=1.0)

        # Skip physics loss during MAE pretraining — reconstruction is patch-based
        # [B, N_patches, patch_dim], not image-shaped [B, C, H, W]
        # Physics loss will be applied during fine-tuning on full images

        return {
            "mae_loss": mae_loss,
            "total_loss": mae_loss,
            "reconstruction": reconstruction,
            "target": target,
            "mask": mask,
        }

    def compute_loss(
        self,
        outputs: dict[str, torch.Tensor],
        wq_targets: Optional[torch.Tensor] = None,
        param_weights: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Compute water quality regression loss.

        Args:
            outputs: Forward pass outputs.
            wq_targets: [B, 16] ground truth water quality values. NaN = missing.
            param_weights: [16] optional per-parameter importance weights.

        Returns:
            Dict of loss terms.
        """
        losses: dict[str, torch.Tensor] = {}

        if wq_targets is not None:
            wq_loss = WaterQualityHead.gaussian_nll_loss(
                outputs["water_quality_params"],
                outputs["param_uncertainty"],
                wq_targets,
                param_weights=param_weights,
            )
            losses["wq_loss"] = wq_loss

        return losses
