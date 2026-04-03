"""Cross-modal temporal attention for asynchronous multimodal fusion.

When a new observation arrives from any modality at time *t_q*, this
module collects all modality embeddings, applies temporal decay and
confidence weighting, and computes multi-head attention to produce
a fused representation that is maximally informative given the
current evidence from all modalities.

Missing modalities (never observed) are handled via learned
modality-specific "no data" tokens so the attention mechanism always
operates over a fixed set of *K* keys.
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


class CrossModalTemporalAttention(nn.Module):
    """Multi-head attention across modality embeddings with temporal bias.

    The query is derived from the *triggering* modality's embedding (the
    one that just arrived).  Keys and values come from **all** modalities
    (including the triggering one).  Temporal decay weights are injected
    as additive bias to the pre-softmax attention logits:

    .. math::

        \\text{logit}_{ij} = \\frac{q_i \\cdot k_j}{\\sqrt{d_k}}
            + \\log(w_j)

    where :math:`w_j = \\text{decay}_j \\times \\text{confidence}_j`.

    Args:
        d_model: Embedding dimensionality (must equal
            :data:`SHARED_EMBEDDING_DIM`).
        num_heads: Number of attention heads.
        dropout: Attention dropout probability during training.
    """

    def __init__(
        self,
        d_model: int = SHARED_EMBEDDING_DIM,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, (
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        )
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.scale = math.sqrt(self.d_k)
        self.dropout_p = dropout

        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_out = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(d_model)

        # Learned "no data" tokens -- one per modality.  These are used
        # when a modality has *never* produced an observation.
        self.no_data_tokens = nn.ParameterDict(
            {
                mid: nn.Parameter(torch.randn(d_model) * 0.02)
                for mid in MODALITY_IDS
            }
        )

        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        for linear in (self.W_q, self.W_k, self.W_v, self.W_out):
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        query_embedding: torch.Tensor,
        modality_embeddings: Dict[str, Optional[torch.Tensor]],
        decay_weights: Dict[str, torch.Tensor],
        confidences: Dict[str, float],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute cross-modal attention.

        Args:
            query_embedding: The embedding of the *triggering* modality,
                shape ``[B, d_model]`` or ``[d_model]``.
            modality_embeddings: Mapping from modality id to projected
                embedding of shape ``[B, d_model]`` or ``[d_model]``, or
                ``None`` if no data has ever been received.
            decay_weights: Mapping from modality id to scalar decay
                weight in ``(0, 1]``.
            confidences: Mapping from modality id to confidence in
                ``[0, 1]``.

        Returns:
            fused: Fused representation, shape ``[B, d_model]``.
            attn_weights: Attention weights, shape ``[B, num_heads, K]``
                where ``K = len(MODALITY_IDS)``.
        """
        # Ensure batch dimension.
        if query_embedding.dim() == 1:
            query_embedding = query_embedding.unsqueeze(0)
        B = query_embedding.shape[0]
        device = query_embedding.device

        # Assemble key/value matrix  [B, K, d_model] and weight vector.
        kv_list: list[torch.Tensor] = []
        weight_list: list[torch.Tensor] = []

        for mid in MODALITY_IDS:
            emb = modality_embeddings.get(mid)
            if emb is None:
                # Use learned no-data token, expanded to batch.
                token = self.no_data_tokens[mid].unsqueeze(0).expand(B, -1)
                kv_list.append(token)
                # No-data slots get a small but non-zero weight so the
                # model can learn to ignore them.
                weight_list.append(
                    torch.full((B,), 0.01, device=device)
                )
            else:
                if emb.dim() == 1:
                    emb = emb.unsqueeze(0).expand(B, -1)
                kv_list.append(emb)
                # Combined weight = decay * confidence.
                dw = decay_weights.get(mid, torch.tensor(1.0, device=device))
                conf = confidences.get(mid, 1.0)
                if isinstance(dw, torch.Tensor):
                    w = dw * conf
                else:
                    w = torch.tensor(float(dw) * float(conf), device=device)
                if w.dim() == 0:
                    w = w.unsqueeze(0).expand(B)
                weight_list.append(w)

        # [B, K, d_model]
        kv = torch.stack(kv_list, dim=1)
        # [B, K]
        weights = torch.stack(weight_list, dim=1).to(device)

        # --- Multi-head attention ---
        K_seq = kv.shape[1]  # = NUM_MODALITIES

        Q = self.W_q(query_embedding)                     # [B, d]
        K_proj = self.W_k(kv)                             # [B, K, d]
        V = self.W_v(kv)                                  # [B, K, d]

        # Reshape to [B, num_heads, *, d_k]
        Q = Q.view(B, 1, self.num_heads, self.d_k).transpose(1, 2)        # [B, h, 1, d_k]
        K_proj = K_proj.view(B, K_seq, self.num_heads, self.d_k).transpose(1, 2)  # [B, h, K, d_k]
        V = V.view(B, K_seq, self.num_heads, self.d_k).transpose(1, 2)            # [B, h, K, d_k]

        # Scaled dot-product attention logits.
        logits = torch.matmul(Q, K_proj.transpose(-2, -1)) / self.scale  # [B, h, 1, K]

        # Additive temporal-confidence bias: log(w) clamped for stability.
        log_w = torch.log(weights.clamp(min=1e-8))           # [B, K]
        bias = log_w.unsqueeze(1).unsqueeze(2)                # [B, 1, 1, K]
        logits = logits + bias

        attn = F.softmax(logits, dim=-1)                      # [B, h, 1, K]
        attn = self.attn_dropout(attn)

        # Weighted sum of values.
        out = torch.matmul(attn, V)                           # [B, h, 1, d_k]
        out = out.squeeze(2).transpose(1, 2).contiguous().view(B, self.d_model)  # [B, d]

        fused = self.out_norm(self.W_out(out))

        # Return attention weights for diagnostics.
        attn_weights = attn.squeeze(2)  # [B, h, K]

        return fused, attn_weights
