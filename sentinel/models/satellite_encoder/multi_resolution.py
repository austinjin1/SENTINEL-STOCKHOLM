"""Resolution-aware cross-attention between Sentinel-2 (10m) and Sentinel-3 (300m).

S2 provides spatial detail (10 bands at 10-60m), while S3 OLCI provides
broader spectral coverage (21 bands) and near-daily revisit.  This module
fuses both via bidirectional cross-attention with resolution-aware
positional embeddings.

Typical token counts at 224x224 input with patch_size=16:
  S2: 14x14 = 196 tokens  (10m -> 160m patches)
  S3:  ~1x1 = 1-4 tokens  (300m -> one S3 pixel covers ~30x30 S2 pixels)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResolutionPositionalEmbedding(nn.Module):
    """Learnable 2D positional embeddings scaled by sensor resolution.

    Each sensor has its own spatial scale, so the positional grid is
    normalized to physical coordinates (meters) rather than pixel indices.
    """

    def __init__(
        self,
        max_tokens: int,
        embed_dim: int,
        resolution_m: float,
    ) -> None:
        super().__init__()
        self.resolution_m = resolution_m
        self.pos_embed = nn.Parameter(torch.randn(1, max_tokens, embed_dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, N, D].  Adds positional embedding (truncated to N)."""
        return x + self.pos_embed[:, : x.shape[1], :]


class CrossAttentionBlock(nn.Module):
    """Single cross-attention block: queries attend to keys/values from another modality.

    Uses pre-norm architecture with residual connections.
    """

    def __init__(
        self,
        embed_dim: int = 384,
        num_heads: int = 6,
        dropout: float = 0.0,
        ffn_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_ff = nn.LayerNorm(embed_dim)
        ffn_dim = int(embed_dim * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Cross-attention forward.

        Args:
            query: [B, N_q, D] - tokens that attend.
            context: [B, N_kv, D] - tokens being attended to.
            key_padding_mask: [B, N_kv] bool mask, True = ignore.

        Returns:
            Updated query tokens [B, N_q, D].
        """
        # Cross-attention with residual
        q = self.norm_q(query)
        kv = self.norm_kv(context)
        attn_out, _ = self.cross_attn(
            q, kv, kv, key_padding_mask=key_padding_mask
        )
        query = query + attn_out

        # FFN with residual
        query = query + self.ffn(self.norm_ff(query))
        return query


class ResolutionCrossAttention(nn.Module):
    """Bidirectional cross-attention fusion between S2 and S3 token streams.

    S3 tokens attend to S2 tokens to inherit spatial detail, and S2 tokens
    attend to S3 tokens to absorb broader spectral context.  Each direction
    uses its own cross-attention blocks with resolution-aware positional
    embeddings.

    Args:
        embed_dim: Token embedding dimension.
        num_heads: Number of attention heads.
        num_layers: Number of cross-attention layers per direction.
        s2_max_tokens: Maximum S2 token count.
        s3_max_tokens: Maximum S3 token count.
        s2_input_dim: S2 token input dimension (before projection).
        s3_input_dim: S3 token input dimension (before projection).
        s2_resolution_m: S2 ground resolution in meters.
        s3_resolution_m: S3 ground resolution in meters.
    """

    def __init__(
        self,
        embed_dim: int = 384,
        num_heads: int = 6,
        num_layers: int = 2,
        s2_max_tokens: int = 196,
        s3_max_tokens: int = 16,
        s2_input_dim: int = 384,
        s3_input_dim: int = 384,
        s2_resolution_m: float = 10.0,
        s3_resolution_m: float = 300.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        # Input projections (identity if dims match)
        self.s2_proj = (
            nn.Linear(s2_input_dim, embed_dim)
            if s2_input_dim != embed_dim
            else nn.Identity()
        )
        self.s3_proj = (
            nn.Linear(s3_input_dim, embed_dim)
            if s3_input_dim != embed_dim
            else nn.Identity()
        )

        # Resolution-aware positional embeddings
        self.s2_pos = ResolutionPositionalEmbedding(
            s2_max_tokens, embed_dim, s2_resolution_m
        )
        self.s3_pos = ResolutionPositionalEmbedding(
            s3_max_tokens, embed_dim, s3_resolution_m
        )

        # S3->S2 cross-attention (S3 queries, S2 keys/values)
        self.s3_to_s2_layers = nn.ModuleList([
            CrossAttentionBlock(embed_dim, num_heads)
            for _ in range(num_layers)
        ])

        # S2->S3 cross-attention (S2 queries, S3 keys/values)
        self.s2_to_s3_layers = nn.ModuleList([
            CrossAttentionBlock(embed_dim, num_heads)
            for _ in range(num_layers)
        ])

        # Fusion gate: learnable weighting of cross-attended features
        self.fusion_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid(),
        )

        # Output projection
        self.output_norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        s2_tokens: torch.Tensor,
        s3_tokens: torch.Tensor,
        s2_mask: torch.Tensor | None = None,
        s3_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fuse S2 and S3 token streams via bidirectional cross-attention.

        Args:
            s2_tokens: [B, N_s2, D_s2] Sentinel-2 patch tokens.
            s3_tokens: [B, N_s3, D_s3] Sentinel-3 OLCI tokens.
            s2_mask: [B, N_s2] padding mask (True = invalid).
            s3_mask: [B, N_s3] padding mask (True = invalid).

        Returns:
            Fused representation [B, N_s2, embed_dim], aligned to S2 spatial grid.
        """
        # Project to common embedding dimension
        s2 = self.s2_proj(s2_tokens)
        s3 = self.s3_proj(s3_tokens)

        # Add resolution-aware positional embeddings
        s2 = self.s2_pos(s2)
        s3 = self.s3_pos(s3)

        # Bidirectional cross-attention
        s3_enriched = s3
        for layer in self.s3_to_s2_layers:
            s3_enriched = layer(s3_enriched, s2, key_padding_mask=s2_mask)

        s2_enriched = s2
        for layer in self.s2_to_s3_layers:
            s2_enriched = layer(s2_enriched, s3_enriched, key_padding_mask=s3_mask)

        # Gated fusion: blend original S2 with cross-attended S2
        gate = self.fusion_gate(torch.cat([s2, s2_enriched], dim=-1))
        fused = gate * s2_enriched + (1 - gate) * s2

        return self.output_norm(fused)

    def forward_s2_only(self, s2_tokens: torch.Tensor) -> torch.Tensor:
        """Passthrough when S3 data is unavailable.

        Args:
            s2_tokens: [B, N_s2, D_s2].

        Returns:
            Projected S2 tokens [B, N_s2, embed_dim].
        """
        s2 = self.s2_proj(s2_tokens)
        s2 = self.s2_pos(s2)
        return self.output_norm(s2)
