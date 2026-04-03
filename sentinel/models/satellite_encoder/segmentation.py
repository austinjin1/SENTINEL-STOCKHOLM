"""UPerNet segmentation head for satellite anomaly detection.

Takes multi-scale ViT features from layers [3, 6, 9, 12] and produces
per-pixel anomaly probability and class logits via feature pyramid fusion.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# Anomaly class definitions
ANOMALY_CLASSES = [
    "algal_bloom",
    "turbidity_plume",
    "oil_sheen",
    "thermal_anomaly",
    "discoloration",
    "foam_surfactant",
    "normal",
]
NUM_ANOMALY_CLASSES = len(ANOMALY_CLASSES)

# Default class weights for handling imbalance (normal class is most frequent)
DEFAULT_CLASS_WEIGHTS = torch.tensor(
    [3.0, 3.0, 5.0, 4.0, 3.0, 5.0, 0.5], dtype=torch.float32
)


class ConvBNReLU(nn.Module):
    """Conv2d + BatchNorm + ReLU block."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PPM(nn.Module):
    """Pyramid Pooling Module from PSPNet.

    Applies multi-scale pooling to capture global context at different
    spatial granularities.
    """

    def __init__(
        self,
        in_channels: int,
        pool_channels: int,
        pool_scales: tuple[int, ...] = (1, 2, 3, 6),
    ) -> None:
        super().__init__()
        self.pool_scales = pool_scales
        self.stages = nn.ModuleList()
        for _ in pool_scales:
            self.stages.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(output_size=_),
                    ConvBNReLU(in_channels, pool_channels, kernel_size=1),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool at multiple scales, upsample, and concatenate with input."""
        h, w = x.shape[2:]
        out = [x]
        for stage in self.stages:
            pooled = stage(x)
            upsampled = F.interpolate(
                pooled, size=(h, w), mode="bilinear", align_corners=False
            )
            out.append(upsampled)
        return torch.cat(out, dim=1)


class FPN(nn.Module):
    """Feature Pyramid Network for multi-scale feature fusion.

    Takes features from multiple ViT layers and produces aligned feature
    maps at the same spatial resolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 256,
        num_levels: int = 4,
    ) -> None:
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for _ in range(num_levels):
            self.lateral_convs.append(ConvBNReLU(in_channels, out_channels, 1))
            self.fpn_convs.append(ConvBNReLU(out_channels, out_channels, 3))

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        """Fuse multi-scale features via lateral connections.

        Args:
            features: List of feature maps [B, C, H, W] from different layers,
                ordered from shallowest to deepest.

        Returns:
            List of fused feature maps, all at the same spatial resolution.
        """
        # Lateral projections
        laterals = [
            conv(feat) for conv, feat in zip(self.lateral_convs, features)
        ]

        # Top-down pathway
        for i in range(len(laterals) - 1, 0, -1):
            h, w = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=(h, w), mode="bilinear", align_corners=False
            )

        # FPN convolutions
        fpn_out = [conv(lat) for conv, lat in zip(self.fpn_convs, laterals)]
        return fpn_out


class UPerNetHead(nn.Module):
    """Unified Perceptual Parsing Network segmentation head.

    Combines FPN fusion with Pyramid Pooling for robust multi-scale
    segmentation of satellite anomalies.

    Args:
        in_channels: Channel dimension of ViT features (384 for ViT-S).
        fpn_channels: Intermediate FPN channel dimension.
        num_levels: Number of feature pyramid levels.
        pool_scales: Spatial scales for the PPM module.
    """

    def __init__(
        self,
        in_channels: int = 384,
        fpn_channels: int = 256,
        num_levels: int = 4,
        pool_scales: tuple[int, ...] = (1, 2, 3, 6),
    ) -> None:
        super().__init__()
        self.num_levels = num_levels

        # Pyramid Pooling on deepest feature
        self.ppm = PPM(in_channels, fpn_channels // len(pool_scales), pool_scales)
        ppm_out_channels = in_channels + (fpn_channels // len(pool_scales)) * len(
            pool_scales
        )
        self.ppm_bottleneck = ConvBNReLU(ppm_out_channels, in_channels, 3)

        # Feature Pyramid Network
        self.fpn = FPN(in_channels, fpn_channels, num_levels)

        # Final fusion: concatenate all FPN levels + apply bottleneck
        self.fusion_conv = ConvBNReLU(fpn_channels * num_levels, fpn_channels, 3)

        # Anomaly probability head (binary: anomaly vs normal)
        self.anomaly_head = nn.Sequential(
            ConvBNReLU(fpn_channels, fpn_channels // 2, 3),
            nn.Conv2d(fpn_channels // 2, 1, kernel_size=1),
        )

        # Anomaly class logits head
        self.class_head = nn.Sequential(
            ConvBNReLU(fpn_channels, fpn_channels // 2, 3),
            nn.Conv2d(fpn_channels // 2, NUM_ANOMALY_CLASSES, kernel_size=1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize convolution and batch norm weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self, multi_scale_features: dict[int, torch.Tensor], spatial_size: int = 14
    ) -> dict[str, torch.Tensor]:
        """Forward pass through UPerNet segmentation head.

        Args:
            multi_scale_features: Dict mapping layer indices to spatial
                feature maps of shape [B, C, H, W].
            spatial_size: Spatial dimension of feature maps (H=W).

        Returns:
            Dict with keys:
                'anomaly_prob': Per-pixel anomaly probability [B, 1, H, W].
                'class_logits': Per-pixel class logits [B, num_classes, H, W].
        """
        # Sort features by layer index (ascending = shallow to deep)
        sorted_keys = sorted(multi_scale_features.keys())
        features = [multi_scale_features[k] for k in sorted_keys]

        # Apply PPM to deepest feature
        features[-1] = self.ppm_bottleneck(self.ppm(features[-1]))

        # FPN fusion
        fpn_features = self.fpn(features)

        # Resize all to the spatial size of the shallowest feature
        target_h, target_w = fpn_features[0].shape[2:]
        aligned = []
        for feat in fpn_features:
            if feat.shape[2:] != (target_h, target_w):
                feat = F.interpolate(
                    feat, size=(target_h, target_w),
                    mode="bilinear", align_corners=False,
                )
            aligned.append(feat)

        # Concatenate and fuse
        fused = self.fusion_conv(torch.cat(aligned, dim=1))

        # Prediction heads
        anomaly_prob = torch.sigmoid(self.anomaly_head(fused))
        class_logits = self.class_head(fused)

        return {
            "anomaly_prob": anomaly_prob,
            "class_logits": class_logits,
        }


class SegmentationLoss(nn.Module):
    """Combined weighted cross-entropy + Dice loss for segmentation.

    The cross-entropy handles class imbalance via per-class weights,
    while the Dice loss improves boundary quality.

    Args:
        class_weights: Per-class weights for cross-entropy. Shape [num_classes].
        dice_weight: Relative weight of Dice loss vs CE loss.
        ce_weight: Relative weight of CE loss.
        smooth: Smoothing constant for Dice loss to avoid division by zero.
    """

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.smooth = smooth

        if class_weights is None:
            class_weights = DEFAULT_CLASS_WEIGHTS
        self.register_buffer("class_weights", class_weights)

    def dice_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Per-class Dice loss averaged over classes.

        Args:
            pred: Softmax probabilities [B, C, H, W].
            target: One-hot encoded targets [B, C, H, W].

        Returns:
            Scalar Dice loss.
        """
        dims = (0, 2, 3)  # Sum over batch and spatial dims
        intersection = (pred * target).sum(dim=dims)
        cardinality = pred.sum(dim=dims) + target.sum(dim=dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()

    def forward(
        self,
        class_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute combined segmentation loss.

        Args:
            class_logits: Raw logits [B, num_classes, H, W].
            targets: Ground truth class indices [B, H, W] (long tensor).

        Returns:
            Dict with 'total', 'ce', and 'dice' loss values.
        """
        # Weighted cross-entropy
        ce_loss = F.cross_entropy(
            class_logits, targets, weight=self.class_weights
        )

        # Dice loss on softmax probabilities
        pred_probs = F.softmax(class_logits, dim=1)
        target_onehot = F.one_hot(
            targets, num_classes=NUM_ANOMALY_CLASSES
        ).permute(0, 3, 1, 2).float()
        dice = self.dice_loss(pred_probs, target_onehot)

        total = self.ce_weight * ce_loss + self.dice_weight * dice
        return {"total": total, "ce": ce_loss, "dice": dice}
