"""Perceiver IO cross-attention for multimodal fusion.

Replaces the old :class:`CrossModalTemporalAttention` with a full
Perceiver IO encode-process-decode architecture:

1. **Encode**: modality tokens attend *to* the latent array -- the
   latent array is updated by cross-attending over all modality tokens
   (latents as queries, modality tokens as keys/values).
2. **Process**: self-attention within the latent array (configurable
   depth, default 2 layers) allows information mixing.
3. **Decode**: a learned output query attends *to* the latent array to
   produce the fused state vector for downstream heads.

Temporal decay is injected as additive log-bias to the encode-step
attention logits, and missing modalities are handled via per-modality
learned "no data" tokens.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from sentinel.models.fusion.embedding_registry import (
    MODALITY_IDS,
    NUM_MODALITIES,
    SHARED_EMBEDDING_DIM,
)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class MultiHeadCrossAttention(nn.Module):
    """Multi-head cross-attention: queries attend to keys/values.

    Args:
        d_model: Model dimensionality.
        num_heads: Number of attention heads.
        dropout: Attention dropout probability.
    """

    def __init__(
        self,
        d_model: int = SHARED_EMBEDDING_DIM,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.scale = math.sqrt(self.d_k)

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_out = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self) -> None:
        for linear in (self.W_q, self.W_k, self.W_v, self.W_out):
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute multi-head cross-attention.

        Args:
            query: ``[B, Nq, D]``
            key: ``[B, Nk, D]``
            value: ``[B, Nk, D]``
            attn_bias: Optional additive bias for logits,
                broadcastable to ``[B, H, Nq, Nk]``.

        Returns:
            output: ``[B, Nq, D]``
            attn_weights: ``[B, H, Nq, Nk]``
        """
        B, Nq, _ = query.shape
        Nk = key.shape[1]
        H, d_k = self.num_heads, self.d_k

        Q = self.W_q(query).view(B, Nq, H, d_k).transpose(1, 2)   # [B, H, Nq, d_k]
        K = self.W_k(key).view(B, Nk, H, d_k).transpose(1, 2)     # [B, H, Nk, d_k]
        V = self.W_v(value).view(B, Nk, H, d_k).transpose(1, 2)   # [B, H, Nk, d_k]

        logits = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, H, Nq, Nk]

        if attn_bias is not None:
            logits = logits + attn_bias

        attn_weights = F.softmax(logits, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        out = torch.matmul(attn_weights, V)                         # [B, H, Nq, d_k]
        out = out.transpose(1, 2).contiguous().view(B, Nq, self.d_model)
        out = self.W_out(out)

        return out, attn_weights


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with pre-LayerNorm and residual.

    Args:
        d_model: Model dimensionality.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        d_model: int = SHARED_EMBEDDING_DIM,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = MultiHeadCrossAttention(d_model, num_heads, dropout)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self._init_ffn()

    def _init_ffn(self) -> None:
        for module in self.ffn.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Self-attention + FFN with residual connections.

        Args:
            x: ``[B, N, D]``

        Returns:
            ``[B, N, D]``
        """
        normed = self.norm(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        x = x + self.ffn(x)
        return x


# ---------------------------------------------------------------------------
# Perceiver IO cross-attention module
# ---------------------------------------------------------------------------

class PerceiverCrossAttention(nn.Module):
    """Perceiver IO encode-process-decode for multimodal fusion.

    The module maintains learned "no data" tokens (one per modality) and
    a learned output query for the decode step.

    Args:
        d_model: Shared embedding dimensionality.
        num_heads: Number of attention heads.
        num_process_layers: Number of self-attention layers in the
            process step.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        d_model: int = SHARED_EMBEDDING_DIM,
        num_heads: int = 8,
        num_process_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # --- Encode step: latents (Q) attend to modality tokens (KV) ---
        self.encode_norm_latent = nn.LayerNorm(d_model)
        self.encode_norm_input = nn.LayerNorm(d_model)
        self.encode_cross_attn = MultiHeadCrossAttention(
            d_model, num_heads, dropout
        )
        self.encode_ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

        # --- Process step: self-attention within latent array ---
        self.process_layers = nn.ModuleList([
            MultiHeadSelfAttention(d_model, num_heads, dropout)
            for _ in range(num_process_layers)
        ])

        # --- Decode step: output query attends to latents ---
        self.decode_query = nn.Parameter(
            torch.randn(1, 1, d_model) * 0.02
        )
        self.decode_norm_query = nn.LayerNorm(d_model)
        self.decode_norm_latent = nn.LayerNorm(d_model)
        self.decode_cross_attn = MultiHeadCrossAttention(
            d_model, num_heads, dropout
        )

        # --- No-data tokens (one per modality) ---
        self.no_data_tokens = nn.ParameterDict({
            mid: nn.Parameter(torch.randn(d_model) * 0.02)
            for mid in MODALITY_IDS
        })

        self._init_ffn()

    def _init_ffn(self) -> None:
        for module in self.encode_ffn.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        latents: torch.Tensor,
        modality_embeddings: Dict[str, Optional[torch.Tensor]],
        temporal_bias: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the full Perceiver IO encode-process-decode.

        Args:
            latents: Latent array, shape ``[B, N, D]``.
            modality_embeddings: Mapping from modality id to projected
                embedding ``[B, D]`` or ``None`` if unavailable.
            temporal_bias: Additive log-bias for encode attention,
                shape ``[B, K]`` where ``K = NUM_MODALITIES``.

        Returns:
            updated_latents: Updated latent array ``[B, N, D]``.
            fused_state: Decoded output vector ``[B, D]``.
            encode_attn_weights: Encode attention ``[B, H, N, K]``.
        """
        B = latents.shape[0]
        device = latents.device

        # --- Assemble modality token matrix [B, K, D] ---
        token_list: list[torch.Tensor] = []
        for mid in MODALITY_IDS:
            emb = modality_embeddings.get(mid)
            if emb is None:
                token = self.no_data_tokens[mid].unsqueeze(0).expand(B, -1)
            else:
                if emb.dim() == 1:
                    emb = emb.unsqueeze(0).expand(B, -1)
                elif emb.shape[0] != B:
                    emb = emb.expand(B, -1)
                token = emb
            token_list.append(token)

        input_tokens = torch.stack(token_list, dim=1)  # [B, K, D]

        # --- Encode: latents cross-attend to modality tokens ---
        # Temporal bias: [B, K] -> [B, 1, 1, K] for broadcast over heads and queries
        attn_bias = temporal_bias.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, K]

        normed_latents = self.encode_norm_latent(latents)
        normed_input = self.encode_norm_input(input_tokens)

        cross_out, encode_attn = self.encode_cross_attn(
            query=normed_latents,
            key=normed_input,
            value=normed_input,
            attn_bias=attn_bias,
        )
        latents = latents + cross_out
        latents = latents + self.encode_ffn(latents)

        # --- Process: self-attention within latent array ---
        for layer in self.process_layers:
            latents = layer(latents)

        # --- Decode: output query cross-attends to latents ---
        decode_q = self.decode_query.expand(B, -1, -1)  # [B, 1, D]
        normed_q = self.decode_norm_query(decode_q)
        normed_lat = self.decode_norm_latent(latents)

        decoded, _ = self.decode_cross_attn(
            query=normed_q,
            key=normed_lat,
            value=normed_lat,
        )
        fused_state = decoded.squeeze(1)  # [B, D]

        return latents, fused_state, encode_attn
