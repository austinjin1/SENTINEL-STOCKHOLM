"""Perceiver IO Multimodal Fusion Layer.

This is the core architectural component of SENTINEL.  It handles
asynchronous, multi-rate data from five modalities with different
update cadences and information persistence by combining:

1. **Projection bank** -- maps each modality to a shared 256-d space.
2. **Embedding registry** -- bookkeeps the latest embedding + metadata.
3. **Temporal decay** -- learned per-modality-pair exponential decay.
4. **Confidence gating** -- calibrated per-modality confidence gates.
5. **Perceiver IO** -- cross-attention with a learned latent array
   replaces the previous cross-modal attention + GRU architecture.

A single forward call represents one *observation event*: a new reading
arrives from some modality and the waterway state is updated.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from sentinel.models.fusion.confidence_gating import ConfidenceGate
from sentinel.models.fusion.embedding_registry import (
    MODALITY_IDS,
    NUM_MODALITIES,
    SHARED_EMBEDDING_DIM,
    EmbeddingRegistry,
)
from sentinel.models.fusion.latent_array import LatentArray
from sentinel.models.fusion.perceiver_attention import PerceiverCrossAttention
from sentinel.models.fusion.projections import NATIVE_DIMS, ProjectionBank
from sentinel.models.fusion.temporal_decay import TemporalDecay

logger = logging.getLogger(__name__)


@dataclass
class FusionOutput:
    """Container for a single fusion step's outputs.

    Attributes:
        fused_state: Decoded output from the Perceiver IO latent array,
            shape ``[B, 256]``.  This is the primary input to all
            downstream output heads.
        latent_state: Updated latent array for recurrence, shape
            ``[B, N, 256]``.  Pass this back as ``latent_state`` in the
            next forward call to maintain temporal continuity.
        attn_weights: Encode-step attention weights,
            shape ``[B, 8, N, K]``.
        decay_weights: Per-modality decay values used this step.
    """

    fused_state: torch.Tensor
    latent_state: torch.Tensor
    attn_weights: torch.Tensor
    decay_weights: Dict[str, torch.Tensor]


class PerceiverIOFusion(nn.Module):
    """Perceiver IO fusion layer for asynchronous multimodal observations.

    Replaces the previous CrossModalTemporalFusion (cross-modal
    attention + GRU) with a Perceiver IO architecture that uses a
    learned latent array as the compressed waterway state.

    Usage::

        fusion = PerceiverIOFusion()
        latent_state = None

        # Satellite observation arrives at t=0
        out = fusion(
            modality_id="satellite",
            raw_embedding=satellite_encoder_output,   # [B, 384]
            timestamp=0.0,
            confidence=0.95,
            latent_state=latent_state,
        )
        latent_state = out.latent_state

        # Sensor reading arrives at t=3600
        out = fusion(
            modality_id="sensor",
            raw_embedding=sensor_encoder_output,      # [B, 256]
            timestamp=3600.0,
            confidence=0.88,
            latent_state=latent_state,
        )
        latent_state = out.latent_state
        anomaly_input = out.fused_state  # feed to output heads

    Args:
        shared_dim: Shared embedding dimensionality.
        native_dims: Per-modality native encoder dims.
        num_latents: Number of latent vectors in the latent array.
        num_heads: Attention heads.
        num_process_layers: Self-attention layers in the Perceiver
            process step.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        shared_dim: int = SHARED_EMBEDDING_DIM,
        native_dims: Dict[str, int] | None = None,
        num_latents: int = 256,
        num_heads: int = 8,
        num_process_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.shared_dim = shared_dim
        self.num_latents = num_latents
        native_dims = native_dims or NATIVE_DIMS

        # Sub-modules.
        self.projection_bank = ProjectionBank(native_dims, shared_dim)
        self.temporal_decay = TemporalDecay()
        self.confidence_gate = ConfidenceGate(d_model=shared_dim)
        self.latent_array = LatentArray(
            num_latents=num_latents,
            latent_dim=shared_dim,
        )
        self.perceiver = PerceiverCrossAttention(
            d_model=shared_dim,
            num_heads=num_heads,
            num_process_layers=num_process_layers,
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
        latent_state: Optional[torch.Tensor] = None,
    ) -> FusionOutput:
        """Process one observation event and update waterway state.

        Args:
            modality_id: Which modality produced this observation.
            raw_embedding: Native-dim embedding from the encoder,
                shape ``[B, native_dim]`` or ``[native_dim]``.
            timestamp: Absolute time of the observation (seconds).
            confidence: Encoder-reported confidence in ``[0, 1]``.
            latent_state: Previous latent array state ``[B, N, D]``,
                or ``None`` for initial (uses learned initialization).

        Returns:
            :class:`FusionOutput` with fused state and diagnostics.
        """
        device = raw_embedding.device

        # --- 1. Project to shared space ---
        projected = self.projection_bank(modality_id, raw_embedding)
        if projected.dim() == 1:
            projected = projected.unsqueeze(0)
        B = projected.shape[0]

        # --- 2. Update registry ---
        self.registry.update(
            modality_id,
            projected.detach().squeeze(0) if B == 1 else projected[0].detach(),
            timestamp,
            confidence,
        )

        # --- 3. Gather all modality embeddings, staleness, and confidence ---
        modality_embeddings: Dict[str, Optional[torch.Tensor]] = {}
        decay_weights: Dict[str, torch.Tensor] = {}
        confidences: Dict[str, torch.Tensor] = {}

        staleness_list: list[float] = []
        confidence_list: list[float] = []

        for mid in MODALITY_IDS:
            entry = self.registry.get_entry(mid)
            if entry is None:
                modality_embeddings[mid] = None
                staleness_list.append(float("inf"))
                confidence_list.append(0.0)
                decay_weights[mid] = torch.tensor(0.0, device=device)
                confidences[mid] = torch.tensor(0.0, device=device)
            else:
                staleness = self.registry.get_staleness(mid, timestamp)
                staleness_list.append(staleness)
                confidence_list.append(entry.confidence)

                emb = entry.embedding.to(device)
                modality_embeddings[mid] = emb
                confidences[mid] = torch.tensor(
                    entry.confidence, dtype=torch.float32, device=device
                )

        # The triggering modality always uses its freshly projected
        # embedding (with gradients) rather than the detached registry copy.
        modality_embeddings[modality_id] = projected.squeeze(0)
        confidences[modality_id] = torch.tensor(
            confidence, dtype=torch.float32, device=device
        )
        # Triggering modality has zero staleness.
        idx = list(MODALITY_IDS).index(modality_id)
        staleness_list[idx] = 0.0
        confidence_list[idx] = confidence

        # --- 4. Compute temporal decay bias ---
        staleness_t = torch.tensor(
            staleness_list, dtype=torch.float32, device=device
        )
        # Clamp inf staleness to a large but finite value for exp safety.
        staleness_t = torch.clamp(staleness_t, max=1e7)

        confidence_t = torch.tensor(
            confidence_list, dtype=torch.float32, device=device
        )

        # Compute log-bias: [K] -> [B, K] by expanding.
        log_bias = self.temporal_decay.compute_log_bias(
            staleness_t.unsqueeze(0).expand(B, -1),
            confidence_t.unsqueeze(0).expand(B, -1),
            query_modality=modality_id,
        )

        # Also store per-modality scalar decay weights for diagnostics.
        decay_vec = self.temporal_decay.forward(staleness_t, modality_id)
        for i, mid in enumerate(MODALITY_IDS):
            decay_weights[mid] = decay_vec[i]

        # --- 5. Confidence gating ---
        gated_embeddings = self.confidence_gate(
            modality_embeddings, confidences
        )

        # --- 6. Initialize or reuse latent state ---
        if latent_state is None:
            latents = self.latent_array.get_latents(B).to(device)
        else:
            latents = latent_state

        # --- 7. Perceiver IO encode-process-decode ---
        updated_latents, fused_state, encode_attn = self.perceiver(
            latents=latents,
            modality_embeddings=gated_embeddings,
            temporal_bias=log_bias,
        )

        return FusionOutput(
            fused_state=fused_state,
            latent_state=updated_latents,
            attn_weights=encode_attn,
            decay_weights=decay_weights,
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def reset_registry(self) -> None:
        """Clear the embedding registry (e.g. between episodes)."""
        self.registry.reset()

    def initial_latent_state(
        self,
        batch_size: int = 1,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Return the learned initial latent state.

        Args:
            batch_size: Number of independent waterway tracks.
            device: Target device.

        Returns:
            Tensor of shape ``[B, N, D]``.
        """
        latents = self.latent_array.get_latents(batch_size)
        if device is not None:
            latents = latents.to(device)
        return latents
