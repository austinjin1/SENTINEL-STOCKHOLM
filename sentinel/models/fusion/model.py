"""Cross-Modal Temporal Attention Fusion Layer.

This is the core architectural innovation of SENTINEL.  It handles
asynchronous, multi-rate data from four modalities with different
update cadences and information persistence by combining:

1. **Projection bank** -- maps each modality to a shared 256-d space.
2. **Embedding registry** -- bookkeeps the latest embedding + metadata.
3. **Temporal decay** -- learned per-modality exponential decay.
4. **Cross-modal attention** -- 8-head attention with temporal bias.
5. **GRU state** -- persistent waterway state updated at each event.

A single forward call represents one *observation event*: a new reading
arrives from some modality and the waterway state is updated.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from sentinel.models.fusion.attention import CrossModalTemporalAttention
from sentinel.models.fusion.embedding_registry import (
    MODALITY_IDS,
    SHARED_EMBEDDING_DIM,
    EmbeddingRegistry,
)
from sentinel.models.fusion.projections import NATIVE_DIMS, ProjectionBank
from sentinel.models.fusion.state import WaterwayStateGRU
from sentinel.models.fusion.temporal_decay import TemporalDecay

logger = logging.getLogger(__name__)


@dataclass
class FusionOutput:
    """Container for a single fusion step's outputs.

    Attributes:
        fused_state: Layer-normed GRU output, shape ``[B, 256]``.
        raw_state: Raw GRU hidden state for recurrence, shape ``[B, 256]``.
        attn_weights: Attention weights, shape ``[B, 8, K]``.
        decay_weights: Per-modality decay values used this step.
    """

    fused_state: torch.Tensor
    raw_state: torch.Tensor
    attn_weights: torch.Tensor
    decay_weights: Dict[str, torch.Tensor]


class CrossModalTemporalFusion(nn.Module):
    """End-to-end fusion layer for asynchronous multimodal observations.

    Usage::

        fusion = CrossModalTemporalFusion()
        state = None

        # Satellite observation arrives at t=0
        out = fusion(
            modality_id="satellite",
            raw_embedding=satellite_encoder_output,   # [B, 384]
            timestamp=0.0,
            confidence=0.95,
            state=state,
        )
        state = out.raw_state

        # Sensor reading arrives at t=3600
        out = fusion(
            modality_id="sensor",
            raw_embedding=sensor_encoder_output,      # [B, 256]
            timestamp=3600.0,
            confidence=0.88,
            state=state,
        )
        state = out.raw_state
        anomaly_input = out.fused_state  # feed to output heads

    Args:
        shared_dim: Shared embedding dimensionality.
        native_dims: Per-modality native encoder dims.
        num_heads: Attention heads.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        shared_dim: int = SHARED_EMBEDDING_DIM,
        native_dims: Dict[str, int] | None = None,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.shared_dim = shared_dim
        native_dims = native_dims or NATIVE_DIMS

        # Sub-modules.
        self.projection_bank = ProjectionBank(native_dims, shared_dim)
        self.temporal_decay = TemporalDecay()
        self.cross_modal_attention = CrossModalTemporalAttention(
            d_model=shared_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.gru_state = WaterwayStateGRU(
            state_dim=shared_dim,
            dropout=dropout,
        )

        # Non-learnable bookkeeping (lives on CPU; tensors copied as needed).
        self.registry = EmbeddingRegistry()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        modality_id: str,
        raw_embedding: torch.Tensor,
        timestamp: float,
        confidence: float = 1.0,
        state: Optional[torch.Tensor] = None,
    ) -> FusionOutput:
        """Process one observation event and update waterway state.

        Args:
            modality_id: Which modality produced this observation.
            raw_embedding: Native-dim embedding from the encoder,
                shape ``[B, native_dim]`` or ``[native_dim]``.
            timestamp: Absolute time of the observation (seconds).
            confidence: Encoder-reported confidence in ``[0, 1]``.
            state: Previous GRU hidden state, or ``None`` for initial.

        Returns:
            :class:`FusionOutput` with fused state and diagnostics.
        """
        device = raw_embedding.device

        # --- 1. Project to shared space ---
        projected = self.projection_bank(modality_id, raw_embedding)
        if projected.dim() == 1:
            projected = projected.unsqueeze(0)

        # --- 2. Update registry ---
        self.registry.update(
            modality_id,
            projected.detach().squeeze(0) if projected.shape[0] == 1 else projected[0].detach(),
            timestamp,
            confidence,
        )

        # --- 3. Gather all modality embeddings and compute decay ---
        modality_embeddings: Dict[str, Optional[torch.Tensor]] = {}
        decay_weights: Dict[str, torch.Tensor] = {}
        confidences: Dict[str, float] = {}

        for mid in MODALITY_IDS:
            entry = self.registry.get_entry(mid)
            if entry is None:
                modality_embeddings[mid] = None
                decay_weights[mid] = torch.tensor(0.0, device=device)
                confidences[mid] = 0.0
            else:
                staleness = self.registry.get_staleness(mid, timestamp)
                staleness_t = torch.tensor(staleness, dtype=torch.float32, device=device)
                dw = self.temporal_decay(staleness_t, mid)

                modality_embeddings[mid] = entry.embedding.to(device)
                decay_weights[mid] = dw
                confidences[mid] = entry.confidence

        # The triggering modality always uses its freshly projected
        # embedding (with gradients) rather than the detached registry copy.
        modality_embeddings[modality_id] = projected.squeeze(0)
        decay_weights[modality_id] = torch.tensor(1.0, device=device)
        confidences[modality_id] = confidence

        # --- 4. Cross-modal attention ---
        fused, attn_weights = self.cross_modal_attention(
            query_embedding=projected,
            modality_embeddings=modality_embeddings,
            decay_weights=decay_weights,
            confidences=confidences,
        )

        # --- 5. GRU state update ---
        output, new_state = self.gru_state(fused, state)

        return FusionOutput(
            fused_state=output,
            raw_state=new_state,
            attn_weights=attn_weights,
            decay_weights=decay_weights,
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def reset_registry(self) -> None:
        """Clear the embedding registry (e.g. between episodes)."""
        self.registry.reset()

    def initial_state(
        self,
        batch_size: int = 1,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Return a zero-initialized GRU state."""
        return self.gru_state.initial_state(batch_size, device)
