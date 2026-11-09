"""Calibrated, data-driven confidence gating for multimodal fusion.

Each modality's contribution to cross-attention is multiplicatively
modulated by a gate in ``[0, 1]`` so that uninformative observations
(cloudy satellite imagery, drifting sensors, low-coverage metagenomics)
are gracefully downweighted rather than masked entirely.

BUGFIX (audit item c): the previous gate depended *only* on a scalar
``confidence`` argument.  Every caller in the shipped fusion path passed
the SAME hardcoded ``confidence=0.9`` for every sample and modality, so
the gate collapsed to the constant ``sigmoid(0.9) == 0.71`` for all
inputs -- it could never suppress a weak modality on a per-sample basis,
which is the entire point of confidence gating.  The gate is now
*data-driven*: a learned FiLM-style head reads the projected embedding
itself (``Linear(D -> 1) -> sigmoid``) to produce a per-sample,
per-modality gate.  The optional scalar ``confidence`` is retained (as an
additive logit bias) so the public API is unchanged, but the gate now
varies with the *content* of each observation.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from sentinel.models.fusion.embedding_registry import (
    MODALITY_IDS,
    SHARED_EMBEDDING_DIM,
)


class ConfidenceGate(nn.Module):
    """Per-modality, data-driven FiLM-style confidence gating.

    Each modality has:

    * a learned FiLM-style head ``Linear(d_model -> 1)`` that reads the
      *projected embedding* and produces a per-sample logit, and
    * a learned temperature (log-space, positive) that calibrates the
      sharpness of the sigmoid.

    Architecture per modality::

        content_logit = W_mid @ embedding + b_mid          # [B]
        gate = sigmoid((content_logit + confidence) / temperature)  # [B]
        gated_embedding = embedding * gate                 # [B, D]

    The scalar/tensor ``confidence`` from the encoder is folded in as an
    additive logit so the gate can still be nudged by an external
    reliability signal, but -- crucially -- the gate now depends on the
    *content* of each observation and can therefore suppress a weak
    modality on a per-sample basis.

    Args:
        d_model: Embedding dimensionality (input to the FiLM head).
        init_temperature: Initial temperature for the sigmoid.
            Higher = softer gating.  Default 1.0.
    """

    def __init__(
        self,
        d_model: int = SHARED_EMBEDDING_DIM,
        init_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # Per-modality FiLM-style content gate: Linear(D -> 1).
        self.gate_head = nn.ModuleDict({
            mid: nn.Linear(d_model, 1)
            for mid in MODALITY_IDS
        })
        # Init near-open (bias>0 -> sigmoid starts ~0.7) so training begins
        # with all modalities meaningfully contributing, then learns to
        # suppress the weak one per-sample.
        for mid in MODALITY_IDS:
            nn.init.xavier_uniform_(self.gate_head[mid].weight)
            nn.init.constant_(self.gate_head[mid].bias, 1.0)

        # Per-modality log-temperature.
        self.log_temperature = nn.ParameterDict({
            mid: nn.Parameter(
                torch.tensor(float(torch.tensor(init_temperature).log()))
            )
            for mid in MODALITY_IDS
        })

    def get_temperature(self, modality_id: str) -> torch.Tensor:
        """Return the positive temperature for *modality_id*."""
        return self.log_temperature[modality_id].exp()

    def gate_single(
        self,
        modality_id: str,
        embedding: torch.Tensor,
        confidence: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the data-driven gate value for one modality.

        Args:
            modality_id: Modality identifier.
            embedding: Projected embedding, shape ``[B, D]`` or ``[D]``.
            confidence: Optional external reliability score(s), broadcast
                to ``[B]``.  Added as a logit bias.  ``None`` -> 0.

        Returns:
            Gate value(s) in ``[0, 1]``, shape ``[B]``.
        """
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        content_logit = self.gate_head[modality_id](embedding).squeeze(-1)  # [B]
        if confidence is not None:
            content_logit = content_logit + confidence
        temp = self.get_temperature(modality_id)
        return torch.sigmoid(content_logit / temp)

    def forward(
        self,
        embeddings: Dict[str, Optional[torch.Tensor]],
        confidences: Dict[str, torch.Tensor],
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Apply data-driven confidence gating to all modality embeddings.

        Args:
            embeddings: Mapping from modality id to embedding tensor
                ``[B, D]`` (or ``[D]``) or ``None`` if the modality has
                no data.
            confidences: Mapping from modality id to raw confidence
                tensor ``[B]`` or scalar.  Used as an additive logit
                bias; may be empty.

        Returns:
            Gated embeddings with the same structure as *embeddings*.
            ``None``-valued entries are passed through unchanged.
        """
        gated: Dict[str, Optional[torch.Tensor]] = {}
        for mid in MODALITY_IDS:
            emb = embeddings.get(mid)
            if emb is None:
                gated[mid] = None
                continue
            conf = confidences.get(mid)
            gate = self.gate_single(mid, emb, conf)  # [B]
            # Broadcast gate [B] -> [B, 1] for element-wise multiply.
            if emb.dim() == 2 and gate.dim() == 1:
                gate = gate.unsqueeze(-1)
            elif emb.dim() == 1:
                gate = gate.reshape(())  # scalar for [D] embedding
            gated[mid] = emb * gate
        return gated
