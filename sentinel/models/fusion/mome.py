"""Mixture-of-Modality-Experts (MoME) Fusion Layer.

A more sophisticated alternative to the existing Perceiver IO fusion
architecture.  Instead of late fusion via cross-attention to a learned
latent array, MoME performs *earlier* fusion by tokenizing each
modality's embedding, tagging tokens with modality type embeddings,
and processing them through shared self-attention with modality-specific
FFN experts -- the approach used by ImageBind and Gemini-style
multimodal models.

Architecture overview::

    modality embeddings  --->  ModalityTokenizer (per-modality)
        [B, D]                    |
                                  v
                        tagged tokens [B, K_total, D]
                                  |
                     +------------+------------+
                     |  MoMETransformerLayer x L |
                     |                          |
                     |   shared self-attention  |
                     |          |               |
                     |   ExpertRouter           |
                     |          |               |
                     |   per-modality FFN       |
                     +------------+------------+
                                  |
                                  v
                         [CLS] pooling or mean
                                  |
                                  v
                         fused_embedding [B, D]

Key differences from PerceiverIOFusion:

* **Tokenization**: each modality embedding is split into multiple
  tokens (satellite=8, sensor=4, microbial=2, molecular=2,
  behavioral=4) rather than treated as a single vector.
* **Shared attention**: all modality tokens attend to each other
  directly, enabling earlier cross-modal interaction.
* **Expert routing**: a learned router dispatches each token to the
  appropriate modality-specific FFN expert, preserving modality-
  specific processing capacity.
* **Load balancing**: auxiliary loss prevents expert collapse.
* **Temporal decay**: attention scores are modulated by temporal
  distance between modality observations.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from sentinel.models.fusion.embedding_registry import (
    MODALITY_IDS,
    NUM_MODALITIES,
    SHARED_EMBEDDING_DIM,
)
from sentinel.models.fusion.temporal_decay import DEFAULT_TAU_PRIORS

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------

# Number of tokens each modality embedding is split into.
DEFAULT_TOKENS_PER_MODALITY: Dict[str, int] = {
    "satellite": 8,
    "sensor": 4,
    "microbial": 2,
    "molecular": 2,
    "behavioral": 4,
}


# -----------------------------------------------------------------------
# Output container
# -----------------------------------------------------------------------

@dataclass
class MoMEOutput:
    """Container for MoME fusion outputs.

    Designed to be interface-compatible with :class:`FusionOutput`
    from the Perceiver IO fusion layer.

    Attributes:
        fused_embedding: Fused representation, shape ``[B, D]``.
            This is the primary input to all downstream output heads
            (equivalent to ``FusionOutput.fused_state``).
        latent_state: Full token sequence after all transformer layers,
            shape ``[B, K_total, D]``.  Can be cached for recurrent
            use (equivalent to ``FusionOutput.latent_state``).
        attn_weights: Attention weights from the last transformer
            layer, shape ``[B, H, K_total, K_total]``.
        router_probs: Router probability distributions from all layers,
            list of ``[B, K_total, num_experts]`` tensors.
        load_balancing_loss: Auxiliary loss encouraging even expert
            utilization, scalar tensor.
        expert_usage: Per-expert fraction of tokens routed, shape
            ``[num_experts]``.
    """

    fused_embedding: torch.Tensor
    latent_state: torch.Tensor
    attn_weights: torch.Tensor
    router_probs: List[torch.Tensor]
    load_balancing_loss: torch.Tensor
    expert_usage: torch.Tensor

    def to_dict(self) -> Dict[str, torch.Tensor]:
        """Return a dict matching the PerceiverIOFusion output interface."""
        return {
            "fused_embedding": self.fused_embedding,
            "fused_state": self.fused_embedding,   # alias for compat
            "latent_state": self.latent_state,
            "load_balancing_loss": self.load_balancing_loss,
        }


# =======================================================================
# ModalityTokenizer
# =======================================================================

class ModalityTokenizer(nn.Module):
    """Convert each modality's embedding into a sequence of tokens.

    For a modality with ``K`` tokens, the embedding ``[B, D]`` is
    projected to ``[B, K, D]`` via a learned linear expansion, then
    augmented with:

    1. A learnable modality type embedding (shared across all ``K``
       tokens of the same modality).
    2. A temporal positional encoding based on the observation
       timestamp.

    Args:
        shared_dim: Token dimensionality.
        tokens_per_modality: Mapping from modality id to number of
            tokens ``K``.
        modalities: Ordered list of modality identifiers.
        max_time_scale: Maximum timescale for sinusoidal temporal
            encoding (seconds).  Default 10 days.
    """

    def __init__(
        self,
        shared_dim: int = SHARED_EMBEDDING_DIM,
        tokens_per_modality: Optional[Dict[str, int]] = None,
        modalities: Optional[List[str]] = None,
        max_time_scale: float = 864_000.0,
    ) -> None:
        super().__init__()
        self.shared_dim = shared_dim
        self.modalities = list(modalities or MODALITY_IDS)
        self.tokens_per_modality = tokens_per_modality or DEFAULT_TOKENS_PER_MODALITY
        self.max_time_scale = max_time_scale

        # Per-modality linear tokenizer: [D] -> [K * D]
        self.tokenizers = nn.ModuleDict({
            mid: nn.Linear(shared_dim, self.tokens_per_modality[mid] * shared_dim)
            for mid in self.modalities
        })

        # Layer norm applied after tokenization.
        self.token_norm = nn.LayerNorm(shared_dim)

        # Learnable modality type embeddings.
        self.modality_embeddings = nn.ParameterDict({
            mid: nn.Parameter(torch.randn(shared_dim) * 0.02)
            for mid in self.modalities
        })

        # Modality index lookup (for temporal encoding).
        self._mid_to_idx = {mid: i for i, mid in enumerate(self.modalities)}

        # Temporal encoding projection.
        self.time_proj = nn.Linear(shared_dim, shared_dim, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        for mid in self.modalities:
            nn.init.xavier_uniform_(self.tokenizers[mid].weight)
            nn.init.zeros_(self.tokenizers[mid].bias)
        nn.init.xavier_uniform_(self.time_proj.weight)

    def _sinusoidal_time_embedding(
        self, timestamp: torch.Tensor, dim: int, device: torch.device
    ) -> torch.Tensor:
        """Generate sinusoidal positional encoding from timestamps.

        Args:
            timestamp: Scalar or ``[B]`` tensor of timestamps (seconds).
            dim: Embedding dimensionality.
            device: Target device.

        Returns:
            Time embedding of shape ``[B, dim]``.
        """
        if timestamp.dim() == 0:
            timestamp = timestamp.unsqueeze(0)

        half = dim // 2
        freqs = torch.exp(
            -math.log(self.max_time_scale)
            * torch.arange(half, dtype=torch.float32, device=device)
            / half
        )  # [half]
        # [B, 1] * [1, half] -> [B, half]
        args = timestamp.unsqueeze(-1).float() * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, dim]
        if dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

    def tokenize(
        self,
        modality_id: str,
        embedding: torch.Tensor,
        timestamp: torch.Tensor,
    ) -> torch.Tensor:
        """Tokenize a single modality's embedding.

        Args:
            modality_id: Which modality.
            embedding: Shape ``[B, D]``.
            timestamp: Shape ``[B]`` or scalar -- observation time.

        Returns:
            Token sequence ``[B, K, D]`` with modality type and
            temporal embeddings added.
        """
        B = embedding.shape[0]
        K = self.tokens_per_modality[modality_id]
        D = self.shared_dim
        device = embedding.device

        # Linear expansion: [B, D] -> [B, K*D] -> [B, K, D]
        tokens = self.tokenizers[modality_id](embedding)
        tokens = tokens.view(B, K, D)
        tokens = self.token_norm(tokens)

        # Add modality type embedding.
        mod_emb = self.modality_embeddings[modality_id]  # [D]
        tokens = tokens + mod_emb.unsqueeze(0).unsqueeze(0)  # broadcast

        # Add temporal positional encoding.
        time_emb = self._sinusoidal_time_embedding(timestamp, D, device)  # [B, D]
        time_emb = self.time_proj(time_emb)  # [B, D]
        tokens = tokens + time_emb.unsqueeze(1)  # [B, 1, D] broadcast over K

        return tokens

    def forward(
        self,
        modality_data: List[Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """Tokenize and concatenate all available modalities.

        Args:
            modality_data: List of ``(modality_id, embedding, timestamp,
                confidence)`` tuples.  Embeddings have shape ``[B, D]``,
                timestamps and confidences are ``[B]`` or scalar tensors.

        Returns:
            tokens: Concatenated token sequence ``[B, K_total, D]``.
            modality_ids_per_token: Integer tensor ``[K_total]`` mapping
                each token position to its modality index in
                ``self.modalities``.
            present_modalities: List of modality ids that were present.
        """
        all_tokens: List[torch.Tensor] = []
        modality_indices: List[int] = []
        present: List[str] = []

        for mid, emb, ts, _conf in modality_data:
            K = self.tokens_per_modality[mid]
            tokens = self.tokenize(mid, emb, ts)  # [B, K, D]
            all_tokens.append(tokens)
            modality_indices.extend([self._mid_to_idx[mid]] * K)
            present.append(mid)

        # Concatenate along token dimension.
        all_tokens_cat = torch.cat(all_tokens, dim=1)  # [B, K_total, D]
        modality_ids_per_token = torch.tensor(
            modality_indices,
            dtype=torch.long,
            device=all_tokens_cat.device,
        )

        return all_tokens_cat, modality_ids_per_token, present


# =======================================================================
# ExpertRouter
# =======================================================================

class ExpertRouter(nn.Module):
    """Learned token-to-expert routing with load balancing loss.

    Given a token embedding, produces a probability distribution over
    experts.  Supports soft gating (weighted combination of all experts)
    and top-k hard gating (only the top-k experts are activated).

    The auxiliary load-balancing loss follows the Switch Transformer
    formulation: it encourages the fraction of tokens dispatched to
    each expert to be uniform.

    Args:
        d_model: Token embedding dimensionality.
        num_experts: Number of experts.
        top_k: Number of experts activated per token.  If ``0`` or
            ``num_experts``, uses soft gating (all experts weighted).
        aux_loss_weight: Weight for the load-balancing loss.
    """

    def __init__(
        self,
        d_model: int = SHARED_EMBEDDING_DIM,
        num_experts: int = NUM_MODALITIES,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.use_soft = self.top_k <= 0 or self.top_k >= num_experts
        self.aux_loss_weight = aux_loss_weight

        self.gate = nn.Linear(d_model, num_experts, bias=False)
        nn.init.xavier_uniform_(self.gate.weight)

    def forward(
        self, tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Route tokens to experts.

        Args:
            tokens: Token embeddings ``[B, T, D]``.

        Returns:
            router_weights: Expert weights per token,
                ``[B, T, num_experts]``.  For top-k routing, non-
                selected experts have weight 0.
            aux_loss: Scalar load-balancing loss.
        """
        B, T, D = tokens.shape

        logits = self.gate(tokens)  # [B, T, E]
        probs = F.softmax(logits, dim=-1)  # [B, T, E]

        if self.use_soft:
            router_weights = probs
        else:
            # Top-k selection: keep only top_k experts per token.
            topk_vals, topk_idx = torch.topk(probs, self.top_k, dim=-1)
            # Re-normalize the top-k weights.
            topk_weights = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + 1e-8)
            # Scatter back to full expert dimension.
            router_weights = torch.zeros_like(probs)
            router_weights.scatter_(-1, topk_idx, topk_weights)

        # --- Load balancing loss (Switch Transformer Eq. 4) ---
        # f_i = fraction of tokens routed to expert i (using argmax).
        expert_assignments = probs.argmax(dim=-1)  # [B, T]
        # One-hot: [B, T, E]
        one_hot = F.one_hot(expert_assignments, self.num_experts).float()
        # f_i: mean fraction across batch and tokens.
        f = one_hot.mean(dim=(0, 1))  # [E]
        # P_i: mean router probability per expert.
        P = probs.mean(dim=(0, 1))  # [E]
        # Loss = E * sum(f_i * P_i)
        aux_loss = self.aux_loss_weight * self.num_experts * (f * P).sum()

        return router_weights, aux_loss


# =======================================================================
# TemporalDecayAttention
# =======================================================================

class TemporalDecayAttention(nn.Module):
    """Self-attention with temporal decay modulation.

    Standard multi-head self-attention where the raw attention scores
    are modulated by the temporal distance between observations from
    different modalities.  Tokens from the same modality have zero
    temporal distance; tokens from different modalities have distance
    equal to ``|t_i - t_j|``.

    The decay is parameterized as::

        decay(dt) = exp(-dt / tau_ij)

    where ``tau_ij`` is a learnable per-modality-pair time constant
    initialized from physically-motivated priors.

    Optionally uses PyTorch's ``scaled_dot_product_attention`` with
    flash attention when available.

    Args:
        d_model: Model dimensionality.
        num_heads: Number of attention heads.
        dropout: Attention dropout probability.
        tau_priors: Per-modality initial tau values (seconds).
        modalities: Ordered modality identifiers.
    """

    def __init__(
        self,
        d_model: int = SHARED_EMBEDDING_DIM,
        num_heads: int = 8,
        dropout: float = 0.1,
        tau_priors: Optional[Dict[str, float]] = None,
        modalities: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, (
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        )

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.scale = math.sqrt(self.d_k)
        self.dropout = nn.Dropout(dropout)

        self.modalities = list(modalities or MODALITY_IDS)
        self.num_modalities = len(self.modalities)
        tau_priors = tau_priors or DEFAULT_TAU_PRIORS

        # Learnable log-tau for modality pairs.
        # Initialize as geometric mean of per-modality priors.
        _TAU_MIN = 60.0
        init_log_tau = torch.zeros(self.num_modalities, self.num_modalities)
        for i, mi in enumerate(self.modalities):
            for j, mj in enumerate(self.modalities):
                geo_mean = math.sqrt(tau_priors[mi] * tau_priors[mj])
                init_log_tau[i, j] = math.log(max(geo_mean - _TAU_MIN, 1.0))
        self.log_tau = nn.Parameter(init_log_tau)
        self._tau_min = _TAU_MIN

        # QKV projections.
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_out = nn.Linear(d_model, d_model)

        self._init_weights()

        # Check for flash attention availability.
        self._has_flash = hasattr(F, "scaled_dot_product_attention")

    def _init_weights(self) -> None:
        for linear in (self.W_q, self.W_k, self.W_v, self.W_out):
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)

    def _symmetric_tau(self) -> torch.Tensor:
        """Return the symmetric positive tau matrix ``[M, M]``."""
        sym_log = (self.log_tau + self.log_tau.t()) / 2.0
        return torch.exp(sym_log) + self._tau_min

    def _compute_temporal_bias(
        self,
        timestamps_per_token: torch.Tensor,
        modality_ids_per_token: torch.Tensor,
    ) -> torch.Tensor:
        """Compute additive attention bias from temporal distances.

        Args:
            timestamps_per_token: ``[T]`` timestamp for each token
                (tokens from the same modality share a timestamp).
            modality_ids_per_token: ``[T]`` integer modality index
                per token.

        Returns:
            Temporal bias ``[1, 1, T, T]`` broadcastable over batch
            and heads.
        """
        T = timestamps_per_token.shape[0]
        device = timestamps_per_token.device

        # Pairwise absolute time differences: [T, T]
        dt = (timestamps_per_token.unsqueeze(1) - timestamps_per_token.unsqueeze(0)).abs()

        # Look up tau for each token pair: [T, T]
        tau_matrix = self._symmetric_tau()  # [M, M]
        mi = modality_ids_per_token  # [T]
        tau_per_pair = tau_matrix[mi.unsqueeze(1), mi.unsqueeze(0)]  # [T, T]

        # Compute log-decay: log(exp(-dt / tau)) = -dt / tau
        log_decay = -dt / tau_per_pair  # [T, T]

        return log_decay.unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]

    def forward(
        self,
        x: torch.Tensor,
        timestamps_per_token: torch.Tensor,
        modality_ids_per_token: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute temporally-modulated self-attention.

        Args:
            x: Input tokens ``[B, T, D]``.
            timestamps_per_token: ``[T]`` timestamps per token position.
            modality_ids_per_token: ``[T]`` modality index per token.
            attn_mask: Optional boolean mask ``[B, T]`` where ``True``
                indicates valid tokens.

        Returns:
            output: Attended tokens ``[B, T, D]``.
            attn_weights: Attention weights ``[B, H, T, T]``.
        """
        B, T, D = x.shape
        H, d_k = self.num_heads, self.d_k

        Q = self.W_q(x).view(B, T, H, d_k).transpose(1, 2)  # [B, H, T, d_k]
        K = self.W_k(x).view(B, T, H, d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, H, d_k).transpose(1, 2)

        # Temporal decay bias: [1, 1, T, T]
        temporal_bias = self._compute_temporal_bias(
            timestamps_per_token, modality_ids_per_token
        )

        # Raw attention logits.
        logits = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, H, T, T]
        logits = logits + temporal_bias

        # Apply attention mask (mask out padded/missing tokens).
        if attn_mask is not None:
            # attn_mask: [B, T] -> [B, 1, 1, T]
            mask_2d = attn_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
            logits = logits.masked_fill(~mask_2d, float("-inf"))

        attn_weights = F.softmax(logits, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Handle NaN from all-masked rows.
        attn_weights = attn_weights.nan_to_num(0.0)

        out = torch.matmul(attn_weights, V)  # [B, H, T, d_k]
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.W_out(out)

        return out, attn_weights


# =======================================================================
# MoMETransformerLayer
# =======================================================================

class MoMETransformerLayer(nn.Module):
    """Single Mixture-of-Modality-Experts transformer layer.

    Consists of:

    1. Pre-norm multi-head self-attention with temporal decay
       (shared across all modalities).
    2. Router network dispatching tokens to modality-specific experts.
    3. Per-modality FFN experts (one per modality).
    4. LayerNorm + residual connections.

    Args:
        d_model: Token dimensionality.
        num_heads: Attention heads.
        num_experts: Number of FFN experts.
        expert_dim: Hidden dimension of each expert FFN.
        dropout: Dropout probability.
        router_top_k: Top-k expert selection (0 = soft gating).
        aux_loss_weight: Load-balancing loss weight.
        tau_priors: Per-modality temporal decay priors.
        modalities: Ordered modality identifiers.
    """

    def __init__(
        self,
        d_model: int = SHARED_EMBEDDING_DIM,
        num_heads: int = 8,
        num_experts: int = NUM_MODALITIES,
        expert_dim: int = 512,
        dropout: float = 0.1,
        router_top_k: int = 2,
        aux_loss_weight: float = 0.01,
        tau_priors: Optional[Dict[str, float]] = None,
        modalities: Optional[List[str]] = None,
    ) -> None:
        super().__init__()

        # Pre-norm attention.
        self.attn_norm = nn.LayerNorm(d_model)
        self.attention = TemporalDecayAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            tau_priors=tau_priors,
            modalities=modalities,
        )
        self.attn_dropout = nn.Dropout(dropout)

        # Expert router.
        self.router = ExpertRouter(
            d_model=d_model,
            num_experts=num_experts,
            top_k=router_top_k,
            aux_loss_weight=aux_loss_weight,
        )

        # Per-expert FFN layers.
        self.expert_norm = nn.LayerNorm(d_model)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, expert_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(expert_dim, d_model),
                nn.Dropout(dropout),
            )
            for _ in range(num_experts)
        ])

        self._init_expert_weights()

    def _init_expert_weights(self) -> None:
        for expert in self.experts:
            for module in expert.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        timestamps_per_token: torch.Tensor,
        modality_ids_per_token: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process tokens through attention + routed expert FFN.

        Args:
            x: Token embeddings ``[B, T, D]``.
            timestamps_per_token: ``[T]`` timestamp per token.
            modality_ids_per_token: ``[T]`` modality index per token.
            attn_mask: Optional ``[B, T]`` boolean mask (True = valid).

        Returns:
            x: Updated tokens ``[B, T, D]``.
            attn_weights: Attention weights ``[B, H, T, T]``.
            router_weights: Expert routing probabilities
                ``[B, T, num_experts]``.
            aux_loss: Scalar load-balancing loss.
        """
        # --- Self-attention with temporal decay ---
        residual = x
        x_norm = self.attn_norm(x)
        attn_out, attn_weights = self.attention(
            x_norm, timestamps_per_token, modality_ids_per_token, attn_mask
        )
        x = residual + self.attn_dropout(attn_out)

        # --- Expert routing and FFN ---
        residual = x
        x_norm = self.expert_norm(x)
        router_weights, aux_loss = self.router(x_norm)  # [B, T, E]

        # Compute expert outputs and combine via router weights.
        # For efficiency, we compute all experts and weight the outputs
        # rather than conditionally executing (works well for small E).
        B, T, D = x_norm.shape
        expert_outputs = torch.stack(
            [expert(x_norm) for expert in self.experts], dim=-2
        )  # [B, T, E, D]

        # Weighted combination: [B, T, E, 1] * [B, T, E, D] -> sum -> [B, T, D]
        combined = (router_weights.unsqueeze(-1) * expert_outputs).sum(dim=-2)
        x = residual + combined

        return x, attn_weights, router_weights, aux_loss


# =======================================================================
# MoMEFusion (main model)
# =======================================================================

class MoMEFusion(nn.Module):
    """Mixture-of-Modality-Experts fusion model.

    Drop-in replacement for :class:`PerceiverIOFusion`.  Accepts a list
    of modality observations and produces a fused representation by:

    1. Tokenizing each modality's embedding into ``K`` tokens with
       modality type and temporal embeddings.
    2. Processing the concatenated token sequence through ``L``
       transformer layers with shared self-attention and per-modality
       FFN experts.
    3. Pooling the output tokens into a single fused vector.

    The output interface matches PerceiverIOFusion: the returned dict
    contains ``"fused_embedding"`` (``[B, D]``) and ``"latent_state"``
    (``[B, K_total, D]``).

    Args:
        shared_dim: Shared embedding dimensionality.
        num_heads: Number of attention heads.
        num_layers: Number of MoME transformer layers.
        num_experts: Number of FFN experts (typically one per modality).
        expert_dim: Hidden dimension of each expert FFN.
        dropout: Dropout probability.
        router_top_k: Top-k expert selection (0 = soft gating).
        aux_loss_weight: Load-balancing loss weight.
        tokens_per_modality: Number of tokens per modality.
        modalities: Ordered list of modality identifiers.
        tau_priors: Per-modality temporal decay priors (seconds).
        pool_mode: Token pooling mode: ``"cls"`` (prepend a [CLS] token)
            or ``"mean"`` (mean-pool over all tokens).

    Usage::

        fusion = MoMEFusion()

        out = fusion([
            ("satellite", sat_emb, sat_ts, sat_conf),
            ("sensor", sen_emb, sen_ts, sen_conf),
        ])

        fused = out.fused_embedding  # [B, 256]
        loss += out.load_balancing_loss
    """

    def __init__(
        self,
        shared_dim: int = SHARED_EMBEDDING_DIM,
        num_heads: int = 8,
        num_layers: int = 4,
        num_experts: int = NUM_MODALITIES,
        expert_dim: int = 512,
        dropout: float = 0.1,
        router_top_k: int = 2,
        aux_loss_weight: float = 0.01,
        tokens_per_modality: Optional[Dict[str, int]] = None,
        modalities: Optional[List[str]] = None,
        tau_priors: Optional[Dict[str, float]] = None,
        pool_mode: str = "cls",
    ) -> None:
        super().__init__()
        self.shared_dim = shared_dim
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.modalities = list(modalities or MODALITY_IDS)
        self.pool_mode = pool_mode
        self.tokens_per_modality = tokens_per_modality or DEFAULT_TOKENS_PER_MODALITY

        # Tokenizer.
        self.tokenizer = ModalityTokenizer(
            shared_dim=shared_dim,
            tokens_per_modality=self.tokens_per_modality,
            modalities=self.modalities,
        )

        # [CLS] token for pooling.
        if pool_mode == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, shared_dim) * 0.02)
        else:
            self.cls_token = None

        # Transformer layers.
        self.layers = nn.ModuleList([
            MoMETransformerLayer(
                d_model=shared_dim,
                num_heads=num_heads,
                num_experts=num_experts,
                expert_dim=expert_dim,
                dropout=dropout,
                router_top_k=router_top_k,
                aux_loss_weight=aux_loss_weight,
                tau_priors=tau_priors,
                modalities=self.modalities,
            )
            for _ in range(num_layers)
        ])

        # Output projection (from pooled tokens to fused embedding).
        self.output_norm = nn.LayerNorm(shared_dim)
        self.output_proj = nn.Linear(shared_dim, shared_dim)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

        # Confidence scaling: per-token scalar gate from confidence.
        self.confidence_scale = nn.Sequential(
            nn.Linear(1, shared_dim),
            nn.Sigmoid(),
        )

        logger.info(
            "MoMEFusion: %d layers, %d experts, %d heads, "
            "tokens_per_modality=%s, pool_mode=%s",
            num_layers,
            num_experts,
            num_heads,
            self.tokens_per_modality,
            pool_mode,
        )

    def _total_tokens(self, present_modalities: List[str]) -> int:
        """Total token count for the given present modalities."""
        total = sum(self.tokens_per_modality[m] for m in present_modalities)
        if self.pool_mode == "cls":
            total += 1  # [CLS] token
        return total

    def forward(
        self,
        modality_data: List[Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> MoMEOutput:
        """Fuse multimodal observations.

        Args:
            modality_data: List of ``(modality_id, embedding, timestamp,
                confidence)`` tuples.  Each embedding has shape
                ``[B, D]`` (already projected to shared space).
                Timestamps and confidences are ``[B]`` or scalar
                tensors.  Missing modalities are simply omitted from
                the list.

        Returns:
            :class:`MoMEOutput` with fused embedding and diagnostics.
        """
        if len(modality_data) == 0:
            raise ValueError("At least one modality must be provided.")

        # --- 1. Tokenize ---
        tokens, modality_ids, present = self.tokenizer(modality_data)
        # tokens: [B, K_total, D], modality_ids: [K_total]
        B = tokens.shape[0]
        device = tokens.device

        # --- 2. Apply confidence scaling per modality's tokens ---
        # Build the scale factors for each token chunk, then multiply
        # in a single non-in-place operation to keep autograd happy.
        scale_chunks: List[torch.Tensor] = []
        for mid, _emb, _ts, conf in modality_data:
            K = self.tokens_per_modality[mid]
            # conf: scalar or [B] -> [B, 1]
            if conf.dim() == 0:
                conf = conf.unsqueeze(0).expand(B)
            conf_input = conf.unsqueeze(-1)  # [B, 1]
            scale = self.confidence_scale(conf_input)  # [B, D]
            # Expand scale to [B, K, D] for this modality's tokens.
            scale_chunks.append(scale.unsqueeze(1).expand(-1, K, -1))

        # Concatenate all scales: [B, K_total, D]
        all_scales = torch.cat(scale_chunks, dim=1)
        tokens = tokens * all_scales

        # --- 3. Build timestamp vector for all tokens ---
        ts_list: List[torch.Tensor] = []
        for mid, _emb, ts, _conf in modality_data:
            K = self.tokens_per_modality[mid]
            if ts.dim() == 0:
                ts = ts.unsqueeze(0)
            # Use the first element of the batch for the shared timestamp
            # (tokens share the observation time per modality).
            ts_scalar = ts[0].detach()
            ts_list.extend([ts_scalar] * K)

        timestamps_per_token = torch.stack(ts_list)  # [K_total]

        # --- 4. Prepend [CLS] token if using cls pooling ---
        if self.pool_mode == "cls" and self.cls_token is not None:
            cls_expanded = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
            tokens = torch.cat([cls_expanded, tokens], dim=1)  # [B, 1+K_total, D]

            # CLS token gets modality id = num_modalities (special)
            # and timestamp = mean of all timestamps.
            cls_mod_id = torch.tensor(
                [self.num_experts],  # out-of-range modality index handled by decay
                dtype=torch.long,
                device=device,
            )

            # Extend tau matrix to accommodate CLS token.
            # CLS uses the mean timestamp of all present modalities.
            cls_ts = timestamps_per_token.mean().unsqueeze(0)
            timestamps_per_token = torch.cat([cls_ts, timestamps_per_token])
            modality_ids = torch.cat([cls_mod_id, modality_ids])

            # We need the attention to handle the CLS modality index.
            # The TemporalDecayAttention tau matrix is [M, M]; we need
            # [M+1, M+1] to accommodate CLS.  Instead, we map CLS
            # tokens to modality index 0 for decay purposes (CLS has
            # zero temporal bias since its timestamp is the mean).
            modality_ids[0] = 0  # map CLS to first modality for tau lookup

        # --- 5. Pass through MoME transformer layers ---
        all_router_probs: List[torch.Tensor] = []
        total_aux_loss = torch.tensor(0.0, device=device)
        last_attn_weights = None

        for layer in self.layers:
            tokens, attn_w, router_w, aux_loss = layer(
                tokens,
                timestamps_per_token,
                modality_ids,
            )
            all_router_probs.append(router_w)
            total_aux_loss = total_aux_loss + aux_loss
            last_attn_weights = attn_w

        # Average aux loss over layers.
        total_aux_loss = total_aux_loss / self.num_layers

        # --- 6. Pool tokens to fused embedding ---
        if self.pool_mode == "cls":
            pooled = tokens[:, 0, :]  # [B, D]  -- CLS token
        else:
            pooled = tokens.mean(dim=1)  # [B, D]  -- mean pool

        fused = self.output_proj(self.output_norm(pooled))  # [B, D]

        # --- 7. Compute expert usage statistics ---
        # Stack all router probs and compute per-expert mean.
        all_probs = torch.stack(all_router_probs, dim=0)  # [L, B, T, E]
        expert_usage = all_probs.mean(dim=(0, 1, 2))  # [E]

        return MoMEOutput(
            fused_embedding=fused,
            latent_state=tokens,
            attn_weights=last_attn_weights,
            router_probs=all_router_probs,
            load_balancing_loss=total_aux_loss,
            expert_usage=expert_usage,
        )

    # ------------------------------------------------------------------
    # Compatibility with PerceiverIOFusion interface
    # ------------------------------------------------------------------

    def fuse_single(
        self,
        modality_id: str,
        embedding: torch.Tensor,
        timestamp: float,
        confidence: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Process a single modality observation (convenience wrapper).

        This provides an interface similar to
        :meth:`PerceiverIOFusion.forward` for single-modality updates.

        Args:
            modality_id: Which modality.
            embedding: Shape ``[B, D]`` or ``[D]``.
            timestamp: Observation time (seconds).
            confidence: Confidence score in ``[0, 1]``.

        Returns:
            Dict with ``"fused_embedding"`` and ``"latent_state"``.
        """
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        device = embedding.device

        ts_tensor = torch.tensor([timestamp], dtype=torch.float32, device=device)
        conf_tensor = torch.tensor([confidence], dtype=torch.float32, device=device)

        out = self.forward([
            (modality_id, embedding, ts_tensor, conf_tensor),
        ])
        return out.to_dict()

    def reset_registry(self) -> None:
        """No-op for interface compatibility with PerceiverIOFusion."""
        pass

    def initial_latent_state(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Return an initial latent state (zeros) for compatibility.

        Unlike PerceiverIOFusion, MoME does not maintain a persistent
        latent state between calls.  This returns a zero tensor that
        can be passed to downstream code expecting a latent state.

        Args:
            batch_size: Number of independent tracks.
            device: Target device.

        Returns:
            Zero tensor ``[B, 1, D]``.
        """
        state = torch.zeros(batch_size, 1, self.shared_dim)
        if device is not None:
            state = state.to(device)
        return state
