"""ViT-Small backbone for multispectral satellite imagery.

Architecture: ViT-S/16 with 22M parameters, adapted for 10-band Sentinel-2 input.
Patch size 16x16 pixels corresponds to 160m x 160m ground footprint at 10m resolution.
Pretrained weights loaded from SSL4EO-S12 via timm with band adaptation.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import timm

logger = logging.getLogger(__name__)

# Sentinel-2 band configuration (10 bands used):
# B2(Blue), B3(Green), B4(Red), B5(VRE1), B6(VRE2), B7(VRE3),
# B8(NIR), B8A(Narrow NIR), B11(SWIR1), B12(SWIR2)
NUM_SPECTRAL_BANDS = 10

# ViT-Small architecture constants
VIT_EMBED_DIM = 384
VIT_NUM_HEADS = 6
VIT_NUM_LAYERS = 12
VIT_PATCH_SIZE = 16
VIT_IMAGE_SIZE = 224


def adapt_patch_embed_weights(
    pretrained_weight: torch.Tensor,
    in_chans: int = NUM_SPECTRAL_BANDS,
) -> torch.Tensor:
    """Adapt 3-channel pretrained patch embedding weights to N-channel input.

    Strategy: tile the RGB weights cyclically across the N input channels,
    then scale by 3/N to preserve activation magnitude.

    Args:
        pretrained_weight: Shape [embed_dim, 3, patch_h, patch_w].
        in_chans: Number of input channels (default 10).

    Returns:
        Adapted weight tensor of shape [embed_dim, in_chans, patch_h, patch_w].
    """
    embed_dim, orig_chans, ph, pw = pretrained_weight.shape
    # Tile cyclically along channel dimension
    repeats = (in_chans // orig_chans) + 1
    tiled = pretrained_weight.repeat(1, repeats, 1, 1)[:, :in_chans, :, :]
    # Scale to preserve expected activation magnitude
    tiled = tiled * (orig_chans / in_chans)
    return tiled


def load_ssl4eo_weights(
    model: nn.Module,
    checkpoint_path: Optional[str] = None,
    in_chans: int = NUM_SPECTRAL_BANDS,
) -> None:
    """Load SSL4EO-S12 pretrained weights with band adaptation.

    If no checkpoint path is provided, loads the default timm pretrained
    weights for vit_small_patch16_224 and adapts the patch embedding.

    Args:
        model: The ViT model to load weights into.
        checkpoint_path: Optional path to an SSL4EO-S12 checkpoint file.
        in_chans: Number of input spectral bands.
    """
    if checkpoint_path is not None:
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        # Handle nested state dicts (e.g., {"model": {...}, "optimizer": {...}})
        if "model" in state_dict:
            state_dict = state_dict["model"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
    else:
        # Load default timm pretrained weights
        pretrained_model = timm.create_model(
            "vit_small_patch16_224", pretrained=True, in_chans=3
        )
        state_dict = pretrained_model.state_dict()
        del pretrained_model

    # Adapt patch embedding weights from 3 channels to in_chans
    patch_key = "patch_embed.proj.weight"
    if patch_key in state_dict and state_dict[patch_key].shape[1] != in_chans:
        logger.info(
            "Adapting patch embedding from %d to %d channels",
            state_dict[patch_key].shape[1],
            in_chans,
        )
        state_dict[patch_key] = adapt_patch_embed_weights(
            state_dict[patch_key], in_chans
        )

    # Load with strict=False to handle any minor mismatches
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning("Missing keys when loading weights: %s", missing)
    if unexpected:
        logger.warning("Unexpected keys when loading weights: %s", unexpected)


class SatelliteViTBackbone(nn.Module):
    """Vision Transformer Small backbone for multispectral satellite imagery.

    Architecture:
        - ViT-S/16: 12 layers, 6 heads, embedding dim 384
        - Patch size: 16x16 (160m ground footprint at 10m/px)
        - Input: 10-band multispectral tile resized to 224x224

    The model exposes intermediate features from layers [3, 6, 9, 12]
    for use by the UPerNet segmentation head.

    Args:
        in_chans: Number of input spectral bands. Default 10.
        img_size: Input image spatial size. Default 224.
        pretrained: Whether to load pretrained weights. Default True.
        checkpoint_path: Optional path to SSL4EO-S12 checkpoint.
        feature_layers: Tuple of layer indices to extract features from.
    """

    def __init__(
        self,
        in_chans: int = NUM_SPECTRAL_BANDS,
        img_size: int = VIT_IMAGE_SIZE,
        pretrained: bool = True,
        checkpoint_path: Optional[str] = None,
        feature_layers: tuple[int, ...] = (3, 6, 9, 12),
    ) -> None:
        super().__init__()
        self.in_chans = in_chans
        self.img_size = img_size
        self.embed_dim = VIT_EMBED_DIM
        self.feature_layers = feature_layers
        self.num_patches_per_side = img_size // VIT_PATCH_SIZE

        # Create ViT-S/16 via timm with 10-channel input
        self.vit = timm.create_model(
            "vit_small_patch16_224",
            pretrained=False,
            in_chans=in_chans,
            img_size=img_size,
            num_classes=0,  # Remove classification head
        )

        # Load pretrained weights with band adaptation
        if pretrained:
            load_ssl4eo_weights(self.vit, checkpoint_path, in_chans)

        # Register hooks for intermediate feature extraction
        self._features: dict[int, torch.Tensor] = {}
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward hooks on transformer blocks to capture features."""
        for layer_idx in self.feature_layers:
            # timm ViT blocks are 0-indexed; layer 3 = blocks[2]
            block_idx = layer_idx - 1
            block = self.vit.blocks[block_idx]
            block.register_forward_hook(self._make_hook(layer_idx))

    def _make_hook(self, layer_idx: int):
        """Create a forward hook closure for the given layer."""
        def hook(module, input, output):
            self._features[layer_idx] = output
        return hook

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        """Forward pass through ViT backbone.

        Args:
            x: Input tensor of shape [B, 10, 224, 224].

        Returns:
            cls_token: [CLS] token embedding, shape [B, 384].
            multi_scale_features: Dict mapping layer index to feature tensors
                of shape [B, N_patches, 384] (includes CLS token at position 0).
        """
        self._features.clear()

        # Forward through full ViT; timm returns [B, embed_dim] when num_classes=0
        cls_token = self.vit(x)  # [B, 384]

        # Collect intermediate features (already captured by hooks)
        multi_scale_features = {k: v for k, v in self._features.items()}

        return cls_token, multi_scale_features

    def get_spatial_features(
        self, features: dict[int, torch.Tensor]
    ) -> dict[int, torch.Tensor]:
        """Reshape patch token features into spatial feature maps.

        Strips the [CLS] token and reshapes [B, N+1, D] to [B, D, H, W].

        Args:
            features: Dict of layer features from forward().

        Returns:
            Dict mapping layer index to spatial feature maps [B, D, H, W].
        """
        spatial = {}
        h = w = self.num_patches_per_side
        for layer_idx, feat in features.items():
            # Remove CLS token (position 0)
            patch_tokens = feat[:, 1:, :]  # [B, N, D]
            B, N, D = patch_tokens.shape
            spatial[layer_idx] = patch_tokens.transpose(1, 2).reshape(B, D, h, w)
        return spatial
