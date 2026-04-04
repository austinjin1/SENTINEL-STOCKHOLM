"""Temporal cross-attention over multi-date satellite image stacks.

Processes a stack of T images at the same location with timestamps and
cloud masks.  Separates persistent features (bathymetry, coastline) from
transient signals (pollution plumes, algal blooms) via cloud-confidence-
weighted temporal attention.

Input: [T, C, H, W] image stack with per-frame timestamps and cloud masks.
Output: persistent embedding [B, D] + transient embedding [B, D].
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEncoding(nn.Module):
    """Sinusoidal encoding of timestamps (days since epoch).

    Encodes temporal position using the standard sin/cos formulation so
    the model can reason about time gaps between acquisitions.
    """

    def __init__(self, embed_dim: int, max_period: float = 365.25) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.max_period = max_period
        # Pre-compute frequency bands
        half = embed_dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(half, dtype=torch.float)
            / half
        )
        self.register_buffer("freqs", freqs)  # [D/2]

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """Encode timestamps.

        Args:
            timestamps: [B, T] days since epoch (float).

        Returns:
            Time embeddings [B, T, embed_dim].
        """
        # timestamps: [B, T] -> [B, T, 1]
        t = timestamps.unsqueeze(-1).float()
        # [B, T, D/2]
        args = t * self.freqs.unsqueeze(0).unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class CloudConfidenceGate(nn.Module):
    """Convert cloud masks to soft attention weights.

    Cloud-free frames get high weight; heavily clouded frames are
    downweighted but not completely zeroed (residual information in
    cloud-free patches within a cloudy frame).

    Args:
        min_weight: Minimum attention weight even for fully cloudy frames.
    """

    def __init__(self, min_weight: float = 0.05) -> None:
        super().__init__()
        self.min_weight = min_weight

    def forward(self, cloud_fraction: torch.Tensor) -> torch.Tensor:
        """Compute per-frame confidence from cloud fraction.

        Args:
            cloud_fraction: [B, T] fraction of cloudy pixels per frame, in [0, 1].

        Returns:
            Confidence weights [B, T] in [min_weight, 1].
        """
        confidence = 1.0 - cloud_fraction.clamp(0.0, 1.0)
        return confidence * (1.0 - self.min_weight) + self.min_weight


class TemporalCrossAttentionLayer(nn.Module):
    """Single temporal cross-attention layer with confidence weighting."""

    def __init__(
        self,
        embed_dim: int = 384,
        num_heads: int = 6,
        dropout: float = 0.1,
        ffn_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
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
        x: torch.Tensor,
        confidence_weights: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Temporal self-attention with cloud-confidence bias.

        Args:
            x: [B, T, D] temporal token sequence.
            confidence_weights: [B, T] per-frame confidence. Used to bias
                attention toward cloud-free frames.
            padding_mask: [B, T] bool, True = padded frame.

        Returns:
            Updated tokens [B, T, D].
        """
        B, T, D = x.shape

        # Build attention bias from confidence weights
        attn_mask = None
        if confidence_weights is not None:
            # Bias: queries attend preferentially to high-confidence keys
            # [B, T] -> [B, 1, T] broadcast over query positions
            log_conf = torch.log(confidence_weights.clamp(min=1e-6))
            # [B, T, T]: each query sees same key bias
            attn_bias = log_conf.unsqueeze(1).expand(B, T, T)
            # MultiheadAttention expects [B*num_heads, T, T] or None
            # We'll add it after pre-norm manually isn't supported,
            # so we scale the input instead (approximate approach)
            # Actually, use attn_mask parameter: additive mask
            num_heads = self.self_attn.num_heads
            attn_mask = attn_bias.unsqueeze(1).expand(
                B, num_heads, T, T
            ).reshape(B * num_heads, T, T)

        # Pre-norm self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(
            x_norm, x_norm, x_norm,
            attn_mask=attn_mask,
            key_padding_mask=padding_mask,
        )
        x = x + attn_out

        # FFN
        x = x + self.ffn(self.norm2(x))
        return x


class TemporalAttentionStack(nn.Module):
    """Temporal attention stack for multi-date satellite imagery.

    Processes a temporal stack of per-image CLS embeddings (or pooled
    spatial features) with timestamps and cloud confidence, producing:
    - Persistent embedding: time-averaged features (bathymetry, coastline)
    - Transient embedding: deviation from temporal mean (plumes, blooms)
    - Fused temporal embedding: combined representation

    Args:
        embed_dim: Embedding dimension per frame.
        num_layers: Number of temporal attention layers.
        num_heads: Number of attention heads.
        max_temporal_len: Maximum number of frames in a stack.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        embed_dim: int = 384,
        num_layers: int = 3,
        num_heads: int = 6,
        max_temporal_len: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        # Time encoding
        self.time_encoder = SinusoidalTimeEncoding(embed_dim)

        # Cloud confidence gate
        self.cloud_gate = CloudConfidenceGate(min_weight=0.05)

        # Temporal attention layers
        self.layers = nn.ModuleList([
            TemporalCrossAttentionLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Learnable query tokens for persistent vs transient
        self.persistent_query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.transient_query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Cross-attention to extract persistent and transient from temporal sequence
        self.persistent_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.transient_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )

        # Output norms
        self.persistent_norm = nn.LayerNorm(embed_dim)
        self.transient_norm = nn.LayerNorm(embed_dim)

        # Fusion of persistent + transient -> single temporal embedding
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        frame_embeddings: torch.Tensor,
        timestamps: torch.Tensor,
        cloud_fractions: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Process temporal image stack.

        Args:
            frame_embeddings: [B, T, D] per-frame CLS embeddings.
            timestamps: [B, T] acquisition times (days since epoch).
            cloud_fractions: [B, T] cloud coverage fraction per frame.
                If None, all frames assumed cloud-free.
            padding_mask: [B, T] bool, True = padded/invalid frame.

        Returns:
            Dict with:
                "temporal_embedding": [B, D] fused temporal representation.
                "persistent_embedding": [B, D] stable/persistent features.
                "transient_embedding": [B, D] transient/change features.
                "confidence_weights": [B, T] computed cloud confidence.
        """
        B, T, D = frame_embeddings.shape
        device = frame_embeddings.device

        # Add temporal positional encoding
        time_embed = self.time_encoder(timestamps)
        x = frame_embeddings + time_embed

        # Cloud confidence weights
        if cloud_fractions is not None:
            confidence = self.cloud_gate(cloud_fractions)
        else:
            confidence = torch.ones(B, T, device=device)

        # Temporal self-attention layers
        for layer in self.layers:
            x = layer(x, confidence_weights=confidence, padding_mask=padding_mask)

        # Extract persistent features via query attention
        p_query = self.persistent_query.expand(B, -1, -1)  # [B, 1, D]
        persistent, _ = self.persistent_attn(
            p_query, x, x, key_padding_mask=padding_mask
        )
        persistent = self.persistent_norm(persistent.squeeze(1))  # [B, D]

        # Extract transient features
        # Compute temporal deviation: each frame minus confidence-weighted mean
        if padding_mask is not None:
            valid = (~padding_mask).float().unsqueeze(-1)  # [B, T, 1]
        else:
            valid = torch.ones(B, T, 1, device=device)
        conf_expanded = confidence.unsqueeze(-1) * valid  # [B, T, 1]
        temporal_mean = (x * conf_expanded).sum(dim=1, keepdim=True) / conf_expanded.sum(dim=1, keepdim=True).clamp(min=1e-6)
        deviations = x - temporal_mean  # [B, T, D]

        t_query = self.transient_query.expand(B, -1, -1)
        transient, _ = self.transient_attn(
            t_query, deviations, deviations, key_padding_mask=padding_mask
        )
        transient = self.transient_norm(transient.squeeze(1))  # [B, D]

        # Fuse persistent + transient
        temporal_embedding = self.fusion(
            torch.cat([persistent, transient], dim=-1)
        )

        return {
            "temporal_embedding": temporal_embedding,
            "persistent_embedding": persistent,
            "transient_embedding": transient,
            "confidence_weights": confidence,
        }

    def forward_single(self, embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        """Handle single-frame case (no temporal context).

        Args:
            embedding: [B, D] single frame CLS embedding.

        Returns:
            Same dict structure with temporal_embedding = input embedding,
            transient set to zero.
        """
        B, D = embedding.shape
        device = embedding.device
        return {
            "temporal_embedding": embedding,
            "persistent_embedding": embedding,
            "transient_embedding": torch.zeros(B, D, device=device),
            "confidence_weights": torch.ones(B, 1, device=device),
        }
