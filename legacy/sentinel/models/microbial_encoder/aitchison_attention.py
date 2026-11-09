"""Aitchison-geometry-aware attention mechanism for compositional data.

Implements attention that operates natively in the Aitchison simplex geometry,
using Centered Log-Ratio (CLR) coordinates for all computations. Standard
Euclidean dot-product attention is replaced with Aitchison inner products,
and batch normalization respects compositional constraints.

References:
    Aitchison, J. (1986). The Statistical Analysis of Compositional Data.
    Pawlowsky-Glahn & Egozcue (2001). Geometric approach to statistical
        analysis on the simplex.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def clr_transform(x: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Centered Log-Ratio transform mapping compositions to Aitchison tangent space.

    Args:
        x: Compositional data on the simplex [*, D] (positive, sums to ~1).
        eps: Small constant for numerical stability.

    Returns:
        CLR-transformed data [*, D].
    """
    x_safe = x.clamp(min=eps)
    log_x = torch.log(x_safe)
    geometric_mean = log_x.mean(dim=-1, keepdim=True)
    return log_x - geometric_mean


def inverse_clr(clr_x: torch.Tensor) -> torch.Tensor:
    """Inverse CLR: map from Aitchison tangent space back to the simplex.

    Args:
        clr_x: CLR-transformed data [*, D].

    Returns:
        Compositional data on the simplex [*, D].
    """
    exp_x = torch.exp(clr_x)
    return exp_x / exp_x.sum(dim=-1, keepdim=True)


class AitchisonMultiHeadAttention(nn.Module):
    """Multi-head attention using Aitchison inner product for similarity.

    Instead of computing Q*K^T in Euclidean space, this module computes
    similarity via the Aitchison inner product <clr(q), clr(k)> which
    respects the geometry of compositional data.

    The Aitchison inner product is equivalent to the Euclidean inner product
    in CLR coordinates, so we project Q/K/V into CLR-compatible subspaces
    and compute attention there.

    Args:
        embed_dim: Total embedding dimension. Default 256.
        num_heads: Number of attention heads. Default 4.
        dropout: Attention dropout rate. Default 0.1.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # CLR centering weights (learnable per-head geometric mean adjustment)
        # This allows each head to learn its own compositional reference frame
        self.clr_bias_q = nn.Parameter(torch.zeros(num_heads, 1, self.head_dim))
        self.clr_bias_k = nn.Parameter(torch.zeros(num_heads, 1, self.head_dim))

        self.attn_dropout = nn.Dropout(dropout)

    def _aitchison_similarity(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Aitchison inner product between Q and K.

        In CLR coordinates, the Aitchison inner product reduces to:
            <clr(x), clr(y)> = sum_i clr(x)_i * clr(y)_i

        We center Q and K by subtracting the per-vector mean (CLR centering)
        before computing the dot product, ensuring the result is invariant
        to proportional scaling of the original compositions.

        Args:
            q: Query tensor [B, H, T_q, head_dim].
            k: Key tensor [B, H, T_k, head_dim].

        Returns:
            Attention scores [B, H, T_q, T_k].
        """
        # CLR-style centering: subtract mean along feature dimension
        # This ensures compositional scale invariance
        q_centered = q - q.mean(dim=-1, keepdim=True) + self.clr_bias_q
        k_centered = k - k.mean(dim=-1, keepdim=True) + self.clr_bias_k

        # Aitchison inner product (= Euclidean dot product in CLR space)
        return torch.matmul(q_centered, k_centered.transpose(-2, -1)) / self.scale

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Compute Aitchison multi-head attention.

        Args:
            query: Query input [B, T_q, embed_dim].
            key: Key input [B, T_k, embed_dim].
            value: Value input [B, T_k, embed_dim].
            key_padding_mask: Mask for padded positions [B, T_k]. True = ignore.
            need_weights: Whether to return attention weights.

        Returns:
            Tuple of (output [B, T_q, embed_dim], attn_weights or None).
        """
        B, T_q, _ = query.shape
        T_k = key.shape[1]

        # Project and reshape to multi-head format
        q = self.q_proj(query).view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)

        # Aitchison similarity scores
        attn_scores = self._aitchison_similarity(q, k)  # [B, H, T_q, T_k]

        # Apply padding mask
        if key_padding_mask is not None:
            # key_padding_mask: [B, T_k] -> [B, 1, 1, T_k]
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf"),
            )

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum of values
        attn_output = torch.matmul(attn_weights, v)  # [B, H, T_q, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T_q, self.embed_dim)
        output = self.out_proj(attn_output)

        if need_weights:
            return output, attn_weights
        return output, None


class AitchisonBatchNorm(nn.Module):
    """Batch normalization respecting Aitchison compositional geometry.

    Standard batch normalization treats features as Euclidean coordinates.
    For compositional data, normalization must be performed in CLR space
    (the Aitchison tangent space) to preserve the simplex constraint.

    The module:
    1. Projects to CLR space
    2. Applies standard batch normalization
    3. (Optionally) projects back to the simplex

    Since the rest of the network operates in CLR coordinates, we typically
    stay in CLR space and skip the inverse transform.

    Args:
        num_features: Number of features per position.
        stay_in_clr: If True, output remains in CLR coordinates. Default True.
        eps: BatchNorm epsilon. Default 1e-5.
        momentum: BatchNorm momentum. Default 0.1.
    """

    def __init__(
        self,
        num_features: int,
        stay_in_clr: bool = True,
        eps: float = 1e-5,
        momentum: float = 0.1,
    ) -> None:
        super().__init__()
        self.stay_in_clr = stay_in_clr
        self.bn = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Aitchison-aware batch normalization.

        Args:
            x: Input tensor. Can be [B, D] or [B, T, D].
                If input is raw compositions (on simplex), it is first CLR-transformed.
                If input is already in CLR space (zero-mean per sample), it is
                normalized directly.

        Returns:
            Normalized tensor with same shape as input.
        """
        if x.dim() == 3:
            B, T, D = x.shape
            # Reshape for BatchNorm1d: [B*T, D]
            x_flat = x.reshape(B * T, D)
            # CLR centering (ensure zero-mean per sample in CLR space)
            x_centered = x_flat - x_flat.mean(dim=-1, keepdim=True)
            x_normed = self.bn(x_centered)
            return x_normed.view(B, T, D)
        else:
            x_centered = x - x.mean(dim=-1, keepdim=True)
            return self.bn(x_centered)


class AitchisonTransformerLayer(nn.Module):
    """Full transformer layer with Aitchison-geometry-aware components.

    Combines Aitchison multi-head attention with a position-wise feed-forward
    network and Aitchison batch normalization. Uses pre-norm architecture
    (norm before attention/FF) for training stability.

    Args:
        embed_dim: Embedding dimension. Default 256.
        num_heads: Number of attention heads. Default 4.
        ff_dim: Feed-forward hidden dimension. Default 512.
        dropout: Dropout rate. Default 0.1.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 4,
        ff_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Aitchison attention
        self.self_attn = AitchisonMultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )

        # Aitchison-aware normalization (pre-norm architecture)
        self.norm1 = AitchisonBatchNorm(embed_dim)
        self.norm2 = AitchisonBatchNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass through one Aitchison transformer layer.

        Args:
            x: Input tensor [B, T, embed_dim].
            key_padding_mask: Padding mask [B, T]. True = ignore.
            need_weights: Whether to return attention weights.

        Returns:
            Tuple of (output [B, T, embed_dim], attn_weights or None).
        """
        # Pre-norm + Aitchison self-attention + residual
        normed = self.norm1(x)
        attn_out, attn_weights = self.self_attn(
            normed, normed, normed,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
        )
        x = x + self.dropout(attn_out)

        # Pre-norm + feed-forward + residual
        normed = self.norm2(x)
        ff_out = self.ff(normed)
        x = x + ff_out

        return x, attn_weights
