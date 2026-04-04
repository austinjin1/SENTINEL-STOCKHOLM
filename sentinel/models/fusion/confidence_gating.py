"""Calibrated confidence-weighted gating for multimodal fusion.

Each encoder outputs a raw confidence score that is passed through a
learned temperature-scaled sigmoid to produce a calibrated gate in
``[0, 1]``.  The gate multiplicatively modulates the modality's
contribution to cross-attention, so that low-confidence observations
(cloudy satellite imagery, drifting sensors, low-coverage metagenomics)
are gracefully downweighted rather than masked entirely.
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
    """Per-modality confidence gating with learned temperature scaling.

    Each modality has an independent temperature parameter that
    calibrates how sharply the confidence score gates the embedding.
    Temperatures are stored in log-space for positivity.

    Architecture per modality::

        raw_confidence  -->  sigmoid(confidence / temperature)  -->  gate
        embedding * gate  -->  gated_embedding

    Args:
        d_model: Embedding dimensionality (for optional learned bias).
        init_temperature: Initial temperature for the sigmoid.
            Higher = softer gating.  Default 1.0 (identity sigmoid).
    """

    def __init__(
        self,
        d_model: int = SHARED_EMBEDDING_DIM,
        init_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # Per-modality log-temperature.
        self.log_temperature = nn.ParameterDict({
            mid: nn.Parameter(
                torch.tensor(float(torch.tensor(init_temperature).log()))
            )
            for mid in MODALITY_IDS
        })

        # Optional per-modality bias (shifts the gating threshold).
        self.bias = nn.ParameterDict({
            mid: nn.Parameter(torch.zeros(1))
            for mid in MODALITY_IDS
        })

    def get_temperature(self, modality_id: str) -> torch.Tensor:
        """Return the positive temperature for *modality_id*."""
        return self.log_temperature[modality_id].exp()

    def gate_single(
        self,
        modality_id: str,
        confidence: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the calibrated gate value for one modality.

        Args:
            modality_id: Modality identifier.
            confidence: Raw confidence score(s), shape ``[B]`` or scalar.

        Returns:
            Gate value(s) in ``[0, 1]``, same shape as *confidence*.
        """
        temp = self.get_temperature(modality_id)
        bias = self.bias[modality_id]
        return torch.sigmoid((confidence + bias) / temp)

    def forward(
        self,
        embeddings: Dict[str, Optional[torch.Tensor]],
        confidences: Dict[str, torch.Tensor],
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Apply confidence gating to all modality embeddings.

        Args:
            embeddings: Mapping from modality id to embedding tensor
                ``[B, D]`` or ``None`` if the modality has no data.
            confidences: Mapping from modality id to raw confidence
                tensor ``[B]`` or scalar.

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
            if conf is None:
                gated[mid] = emb
                continue
            gate = self.gate_single(mid, conf)
            # Broadcast gate [B] -> [B, 1] for element-wise multiply.
            if gate.dim() == 0:
                gate = gate.unsqueeze(0)
            if gate.dim() == 1 and emb.dim() == 2:
                gate = gate.unsqueeze(-1)
            gated[mid] = emb * gate
        return gated
