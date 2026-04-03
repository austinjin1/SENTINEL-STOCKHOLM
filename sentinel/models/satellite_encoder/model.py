"""Full satellite encoder combining ViT backbone, UPerNet segmentation, and temporal change detection.

This module provides the complete satellite modality encoder that:
1. Extracts multi-scale features from 10-band multispectral imagery via ViT-S/16
2. Produces per-pixel anomaly segmentation via UPerNet
3. Detects temporal changes via transformer over rolling CLS embeddings
4. Projects embeddings to a shared 256-dim fusion space
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .backbone import SatelliteViTBackbone, VIT_EMBED_DIM, NUM_SPECTRAL_BANDS
from .segmentation import UPerNetHead, SegmentationLoss
from .temporal import TemporalChangeDetector

SHARED_EMBED_DIM = 256


class SatelliteEncoder(nn.Module):
    """Complete satellite modality encoder for SENTINEL.

    Combines the ViT-S/16 backbone, UPerNet segmentation head, and temporal
    change detection module into a single encoder that produces spatial
    features, temporal embeddings, and anomaly scores.

    Args:
        in_chans: Number of input spectral bands. Default 10.
        pretrained: Whether to load pretrained backbone weights. Default True.
        checkpoint_path: Optional path to SSL4EO-S12 checkpoint.
        shared_embed_dim: Dimension of the shared fusion embedding space.
        temporal_buffer_size: Number of historical CLS tokens for change detection.
    """

    def __init__(
        self,
        in_chans: int = NUM_SPECTRAL_BANDS,
        pretrained: bool = True,
        checkpoint_path: Optional[str] = None,
        shared_embed_dim: int = SHARED_EMBED_DIM,
        temporal_buffer_size: int = 10,
    ) -> None:
        super().__init__()

        # ViT-S/16 backbone
        self.backbone = SatelliteViTBackbone(
            in_chans=in_chans,
            pretrained=pretrained,
            checkpoint_path=checkpoint_path,
        )

        # UPerNet segmentation head
        self.segmentation_head = UPerNetHead(
            in_channels=VIT_EMBED_DIM,
            fpn_channels=256,
            num_levels=4,
        )

        # Temporal change detection
        self.temporal_module = TemporalChangeDetector(
            embed_dim=VIT_EMBED_DIM,
            buffer_size=temporal_buffer_size,
        )

        # Projection head: 384-dim -> shared 256-dim embedding space
        self.projection = nn.Sequential(
            nn.Linear(VIT_EMBED_DIM, VIT_EMBED_DIM),
            nn.GELU(),
            nn.LayerNorm(VIT_EMBED_DIM),
            nn.Linear(VIT_EMBED_DIM, shared_embed_dim),
            nn.LayerNorm(shared_embed_dim),
        )

        # Segmentation loss (used during training)
        self.seg_loss = SegmentationLoss()

        self._init_projection()

    def _init_projection(self) -> None:
        """Initialize projection head weights."""
        for m in self.projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        image: torch.Tensor,
        temporal_cls_buffer: Optional[torch.Tensor] = None,
        temporal_timestamps: Optional[torch.Tensor] = None,
        temporal_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Full forward pass through the satellite encoder.

        Args:
            image: Multispectral input tile [B, 10, 224, 224].
            temporal_cls_buffer: Rolling buffer of CLS embeddings
                [B, T, 384] for temporal change detection. If None,
                temporal outputs use only the current frame.
            temporal_timestamps: Acquisition timestamps in days since epoch
                [B, T]. Required if temporal_cls_buffer is provided.
            temporal_mask: Boolean padding mask [B, T] for temporal buffer.
                True = padded/invalid position.

        Returns:
            Dict with:
                'spatial_features': Segmentation output dict with
                    'anomaly_prob' [B, 1, H, W] and 'class_logits' [B, C, H, W].
                'temporal_embedding': Temporal context embedding [B, 256].
                'change_anomaly_score': Tile-level change score [B].
                'cls_token': Raw CLS embedding [B, 384] for buffer update.
                'fusion_embedding': Projected embedding [B, 256] for fusion.
        """
        B = image.shape[0]

        # 1. Backbone forward pass
        cls_token, multi_scale_features = self.backbone(image)
        # cls_token: [B, 384]
        # multi_scale_features: {3: [B, N+1, 384], 6: ..., 9: ..., 12: ...}

        # 2. Reshape features to spatial maps for segmentation
        spatial_features = self.backbone.get_spatial_features(multi_scale_features)
        # {3: [B, 384, 14, 14], 6: ..., 9: ..., 12: ...}

        # 3. Segmentation head
        seg_output = self.segmentation_head(spatial_features)

        # 4. Temporal change detection
        if temporal_cls_buffer is not None and temporal_timestamps is not None:
            # Append current CLS token to buffer for temporal processing
            current_cls = cls_token.unsqueeze(1)  # [B, 1, 384]
            full_buffer = torch.cat(
                [temporal_cls_buffer, current_cls], dim=1
            )  # [B, T+1, 384]

            # Extend timestamps with a placeholder for current
            # (caller should provide the actual current timestamp)
            current_ts = temporal_timestamps[:, -1:] + 5.0  # ~5-day revisit
            full_timestamps = torch.cat(
                [temporal_timestamps, current_ts], dim=1
            )

            # Extend mask if present
            if temporal_mask is not None:
                current_mask = torch.zeros(
                    B, 1, dtype=torch.bool, device=image.device
                )
                full_mask = torch.cat([temporal_mask, current_mask], dim=1)
            else:
                full_mask = None

            temporal_output = self.temporal_module(
                full_buffer, full_timestamps, full_mask
            )
        else:
            # No temporal context: use current CLS token directly
            temporal_output = {
                "temporal_embedding": cls_token,
                "change_anomaly_score": torch.zeros(B, device=image.device),
            }

        # 5. Project temporal embedding to shared space
        fusion_embedding = self.projection(temporal_output["temporal_embedding"])

        return {
            "spatial_features": seg_output,
            "temporal_embedding": fusion_embedding,
            "change_anomaly_score": temporal_output["change_anomaly_score"],
            "cls_token": cls_token,
            "fusion_embedding": fusion_embedding,
        }

    def compute_loss(
        self,
        outputs: dict[str, torch.Tensor],
        seg_targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute segmentation loss for training.

        Args:
            outputs: Forward pass outputs.
            seg_targets: Ground truth class indices [B, H, W].

        Returns:
            Dict of loss values.
        """
        return self.seg_loss(outputs["spatial_features"]["class_logits"], seg_targets)
