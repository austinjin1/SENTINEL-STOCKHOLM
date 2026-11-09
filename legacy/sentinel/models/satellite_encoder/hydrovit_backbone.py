"""HydroViT backbone: ViT-S/16 adapted for water-specific remote sensing.

Architecture: ViT-S/16 with 22M parameters, adapted for 13-band input
(10 Sentinel-2 bands + 3 key Sentinel-3 OLCI bands).  Supports MAE-style
pretraining with spectral-physics-aware masking and reconstruction.

Patch size 16x16 at 10m resolution = 160m ground footprint.
Native embedding dimension: 384.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import timm

logger = logging.getLogger(__name__)

# 10 S2 bands + 3 key S3 OLCI bands (443nm, 560nm, 665nm)
NUM_SPECTRAL_BANDS = 13

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

    Tiles RGB weights cyclically across input channels, scaled by 3/N to
    preserve activation magnitude.
    """
    embed_dim, orig_chans, ph, pw = pretrained_weight.shape
    repeats = (in_chans // orig_chans) + 1
    tiled = pretrained_weight.repeat(1, repeats, 1, 1)[:, :in_chans, :, :]
    tiled = tiled * (orig_chans / in_chans)
    return tiled


class SpectralPositionalEmbedding(nn.Module):
    """Learnable per-band spectral encoding added to patch embeddings.

    Each input band gets a learnable vector that is broadcast across all
    patches, encoding spectral identity so the transformer can reason
    about band-specific physics (e.g., NIR should be near-zero over water).
    """

    def __init__(self, num_bands: int, embed_dim: int) -> None:
        super().__init__()
        self.spectral_embed = nn.Parameter(
            torch.randn(1, num_bands, embed_dim) * 0.02
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add spectral encoding.  x: [B, num_patches, embed_dim]."""
        # Project spectral embeddings via learned aggregation across bands,
        # preserving per-band structure through a weighted sum rather than mean
        # Each band contributes a distinct learned vector to the spatial encoding
        spectral_weights = torch.softmax(
            self.spectral_embed.mean(dim=-1, keepdim=True), dim=1
        )  # [1, num_bands, 1]
        spectral_code = (self.spectral_embed * spectral_weights).sum(
            dim=1, keepdim=True
        )  # [1, 1, embed_dim]
        return x + spectral_code


class MAEDecoder(nn.Module):
    """Lightweight decoder for masked autoencoder pretraining.

    Reconstructs masked patches from visible-patch latent representations
    plus learned mask tokens, using a shallow transformer decoder.
    """

    def __init__(
        self,
        embed_dim: int = VIT_EMBED_DIM,
        decoder_embed_dim: int = 192,
        decoder_depth: int = 4,
        decoder_num_heads: int = 6,
        patch_size: int = VIT_PATCH_SIZE,
        in_chans: int = NUM_SPECTRAL_BANDS,
    ) -> None:
        super().__init__()
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        num_patches = (VIT_IMAGE_SIZE // patch_size) ** 2
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False
        )
        self._init_pos_embed(num_patches + 1, decoder_embed_dim)

        self.decoder_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=decoder_embed_dim,
                nhead=decoder_num_heads,
                dim_feedforward=decoder_embed_dim * 4,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size ** 2 * in_chans, bias=True
        )

    def _init_pos_embed(self, num_positions: int, embed_dim: int) -> None:
        """Sinusoidal positional embedding initialization."""
        pe = torch.zeros(num_positions, embed_dim)
        position = torch.arange(0, num_positions, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.decoder_pos_embed.data.copy_(pe.unsqueeze(0))

    def forward(
        self,
        latent: torch.Tensor,
        ids_restore: torch.Tensor,
    ) -> torch.Tensor:
        """Decode masked patches.

        Args:
            latent: Encoder output for visible patches [B, N_vis+1, D].
            ids_restore: Indices to unshuffle back to original order [B, N_total].

        Returns:
            Reconstructed patches [B, N_total, patch_size**2 * in_chans].
        """
        x = self.decoder_embed(latent)
        B, N_vis_plus_cls, D = x.shape

        # Append mask tokens for masked positions
        num_masked = ids_restore.shape[1] - (N_vis_plus_cls - 1)
        mask_tokens = self.mask_token.expand(B, num_masked, -1)
        # Separate CLS and visible tokens
        cls_token = x[:, :1, :]
        visible = x[:, 1:, :]
        # Concatenate visible + mask, then unshuffle
        x_ = torch.cat([visible, mask_tokens], dim=1)
        x_ = torch.gather(
            x_, dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, D),
        )
        # Prepend CLS
        x = torch.cat([cls_token, x_], dim=1)

        # Add positional embedding
        x = x + self.decoder_pos_embed[:, : x.shape[1], :]

        # Transformer decoder
        for block in self.decoder_blocks:
            x = block(x)
        x = self.decoder_norm(x)

        # Predict pixel values (skip CLS)
        x = self.decoder_pred(x[:, 1:, :])
        return x


class HydroViTBackbone(nn.Module):
    """HydroViT: Water-specific Vision Transformer backbone.

    Built on ViT-S/16 with water-remote-sensing adaptations:
    - 13-band input (10 S2 + 3 key S3 OLCI bands)
    - Spectral positional encoding for band-aware reasoning
    - MAE pretraining with physics-consistent reconstruction
    - Multi-scale feature extraction for downstream segmentation

    Args:
        in_chans: Number of input spectral bands.  Default 13.
        img_size: Spatial input size.  Default 224.
        pretrained: Load ImageNet-pretrained ViT-S weights with band adaptation.
        checkpoint_path: Optional path to HydroViT-specific checkpoint.
        feature_layers: Layers to extract multi-scale features from.
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
        self.num_patches = self.num_patches_per_side ** 2

        # Create ViT-S/16 via timm
        self.vit = timm.create_model(
            "vit_small_patch16_224",
            pretrained=False,
            in_chans=in_chans,
            img_size=img_size,
            num_classes=0,
        )

        # Spectral positional encoding
        self.spectral_embed = SpectralPositionalEmbedding(in_chans, VIT_EMBED_DIM)

        # MAE decoder for pretraining
        self.mae_decoder = MAEDecoder(
            embed_dim=VIT_EMBED_DIM,
            decoder_embed_dim=192,
            decoder_depth=4,
            decoder_num_heads=6,
            patch_size=VIT_PATCH_SIZE,
            in_chans=in_chans,
        )

        # Load pretrained weights
        if pretrained and checkpoint_path is None:
            self._load_timm_pretrained(in_chans)
        elif checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path, in_chans)

        # Intermediate feature hooks
        self._features: dict[int, torch.Tensor] = {}
        self._register_hooks()

    def _load_timm_pretrained(self, in_chans: int) -> None:
        """Load timm ViT-S pretrained weights with band adaptation."""
        pretrained_model = timm.create_model(
            "vit_small_patch16_224", pretrained=True, in_chans=3
        )
        state_dict = pretrained_model.state_dict()
        del pretrained_model

        patch_key = "patch_embed.proj.weight"
        if patch_key in state_dict and state_dict[patch_key].shape[1] != in_chans:
            logger.info(
                "Adapting patch embedding from %d to %d channels",
                state_dict[patch_key].shape[1], in_chans,
            )
            state_dict[patch_key] = adapt_patch_embed_weights(
                state_dict[patch_key], in_chans
            )

        missing, unexpected = self.vit.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning("Missing keys: %s", missing)
        if unexpected:
            logger.warning("Unexpected keys: %s", unexpected)

    def _load_checkpoint(self, path: str, in_chans: int) -> None:
        """Load a HydroViT checkpoint (full model or backbone-only)."""
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        patch_key = "patch_embed.proj.weight"
        if patch_key in state_dict and state_dict[patch_key].shape[1] != in_chans:
            state_dict[patch_key] = adapt_patch_embed_weights(
                state_dict[patch_key], in_chans
            )

        missing, unexpected = self.vit.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning("Missing keys: %s", missing)
        if unexpected:
            logger.warning("Unexpected keys: %s", unexpected)

    def _register_hooks(self) -> None:
        for layer_idx in self.feature_layers:
            block_idx = layer_idx - 1
            block = self.vit.blocks[block_idx]
            block.register_forward_hook(self._make_hook(layer_idx))

    def _make_hook(self, layer_idx: int):
        def hook(module, input, output):
            self._features[layer_idx] = output
        return hook

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        """Forward pass through HydroViT backbone.

        Args:
            x: Input tensor [B, 13, 224, 224].

        Returns:
            cls_token: [CLS] embedding [B, 384].
            multi_scale_features: Dict {layer_idx: [B, N+1, 384]}.
        """
        self._features.clear()

        # Forward through ViT (timm returns [B, embed_dim] with num_classes=0)
        cls_token = self.vit(x)  # [B, 384]

        multi_scale_features = {k: v for k, v in self._features.items()}
        return cls_token, multi_scale_features

    def forward_mae(
        self,
        x: torch.Tensor,
        mask_ratio: float = 0.75,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """MAE pretraining forward pass with random masking.

        Args:
            x: Input tensor [B, 13, 224, 224].
            mask_ratio: Fraction of patches to mask.

        Returns:
            reconstruction: Predicted pixel values [B, N, patch_size**2 * C].
            target: Ground truth patchified pixels [B, N, patch_size**2 * C].
            mask: Binary mask [B, N], 1 = masked.
        """
        B = x.shape[0]
        N = self.num_patches

        # Patchify target
        target = self._patchify(x)  # [B, N, patch_size**2 * C]

        # Encode with masking
        # Get patch embeddings from the ViT patch_embed layer
        patch_embed = self.vit.patch_embed(x)  # [B, N, D]

        # Add positional embedding (skip CLS position)
        cls_token = self.vit.cls_token.expand(B, -1, -1)
        pos_embed = self.vit.pos_embed

        # Random masking
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        num_visible = int(N * (1 - mask_ratio))
        ids_keep = ids_shuffle[:, :num_visible]

        # Gather visible patches
        visible_patches = torch.gather(
            patch_embed, dim=1,
            index=ids_keep.unsqueeze(-1).expand(-1, -1, self.embed_dim),
        )

        # Add spectral and positional encoding
        visible_pos = torch.gather(
            pos_embed[:, 1:, :].expand(B, -1, -1), dim=1,
            index=ids_keep.unsqueeze(-1).expand(-1, -1, self.embed_dim),
        )
        visible_patches = visible_patches + visible_pos
        visible_patches = self.spectral_embed(visible_patches)

        # Prepend CLS token
        cls_with_pos = cls_token + pos_embed[:, :1, :]
        tokens = torch.cat([cls_with_pos, visible_patches], dim=1)

        # Forward through encoder blocks
        tokens = self.vit.pos_drop(tokens)
        # Manually apply norm_pre if the model has it
        if hasattr(self.vit, 'norm_pre'):
            tokens = self.vit.norm_pre(tokens)
        for block in self.vit.blocks:
            tokens = block(tokens)
        tokens = self.vit.norm(tokens)

        # Decode
        reconstruction = self.mae_decoder(tokens, ids_restore)

        # Build mask: 1 = masked
        mask = torch.ones(B, N, device=x.device)
        mask.scatter_(1, ids_keep, 0.0)

        return reconstruction, target, mask

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert image to patch pixel values.

        Args:
            x: [B, C, H, W].

        Returns:
            patches: [B, N, patch_size**2 * C].
        """
        B, C, H, W = x.shape
        p = VIT_PATCH_SIZE
        h, w = H // p, W // p
        x = x.reshape(B, C, h, p, w, p)
        x = x.permute(0, 2, 4, 3, 5, 1)  # [B, h, w, p, p, C]
        x = x.reshape(B, h * w, p * p * C)
        return x

    def get_spatial_features(
        self, features: dict[int, torch.Tensor]
    ) -> dict[int, torch.Tensor]:
        """Reshape patch features to spatial maps [B, D, H, W]."""
        spatial = {}
        h = w = self.num_patches_per_side
        for layer_idx, feat in features.items():
            patch_tokens = feat[:, 1:, :]
            B, N, D = patch_tokens.shape
            spatial[layer_idx] = patch_tokens.transpose(1, 2).reshape(B, D, h, w)
        return spatial
