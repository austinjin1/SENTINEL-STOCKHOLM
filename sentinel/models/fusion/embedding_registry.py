"""Embedding registry for asynchronous multimodal fusion.

Maintains timestamped embeddings from each modality encoder, tracking
confidence and staleness to enable the cross-modal temporal attention
mechanism to weight contributions appropriately.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import torch


# Canonical modality identifiers and ordering.
MODALITY_IDS: tuple[str, ...] = ("satellite", "sensor", "microbial", "molecular")
NUM_MODALITIES: int = len(MODALITY_IDS)
SHARED_EMBEDDING_DIM: int = 256


@dataclass
class ModalityEntry:
    """A single modality's most-recent embedding and metadata.

    Attributes:
        embedding: Projected embedding in shared space, shape ``[d]``.
        timestamp: Absolute time (seconds since epoch) when the embedding
            was produced by the upstream encoder.
        dimension: Dimensionality of the embedding (always
            :data:`SHARED_EMBEDDING_DIM`).
        confidence: Encoder-reported confidence in ``[0, 1]``.
        staleness: Seconds elapsed since *timestamp* at the last query.
            Recomputed lazily by :meth:`EmbeddingRegistry.get_staleness`.
    """

    embedding: torch.Tensor
    timestamp: float
    dimension: int = SHARED_EMBEDDING_DIM
    confidence: float = 1.0
    staleness: float = 0.0


class EmbeddingRegistry:
    """Thread-safe registry of the most recent embedding per modality.

    The registry does **not** own any learnable parameters; it is a
    bookkeeping structure consumed by the fusion layer.

    Args:
        device: Torch device for zero-initialized placeholder embeddings.
    """

    def __init__(self, device: torch.device = torch.device("cpu")) -> None:
        self.device = device
        self._entries: dict[str, Optional[ModalityEntry]] = {
            mid: None for mid in MODALITY_IDS
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        modality_id: str,
        embedding: torch.Tensor,
        timestamp: float,
        confidence: float = 1.0,
    ) -> None:
        """Record a new embedding for *modality_id*.

        Args:
            modality_id: One of :data:`MODALITY_IDS`.
            embedding: Tensor of shape ``[d]`` in shared space.
            timestamp: Absolute time when the observation was produced.
            confidence: Encoder-reported confidence in ``[0, 1]``.

        Raises:
            ValueError: If *modality_id* is unknown or *embedding* has
                the wrong dimensionality.
        """
        self._validate_modality(modality_id)
        if embedding.shape[-1] != SHARED_EMBEDDING_DIM:
            raise ValueError(
                f"Expected embedding dim {SHARED_EMBEDDING_DIM}, "
                f"got {embedding.shape[-1]}"
            )
        self._entries[modality_id] = ModalityEntry(
            embedding=embedding.detach(),
            timestamp=timestamp,
            confidence=float(confidence),
            staleness=0.0,
        )

    def get_entry(self, modality_id: str) -> Optional[ModalityEntry]:
        """Return the stored entry for *modality_id*, or ``None``."""
        self._validate_modality(modality_id)
        return self._entries[modality_id]

    def has_data(self, modality_id: str) -> bool:
        """Return ``True`` if *modality_id* has ever received data."""
        return self._entries.get(modality_id) is not None

    def get_staleness(
        self, modality_id: str, query_time: float
    ) -> float:
        """Compute seconds between last update and *query_time*.

        Returns ``float('inf')`` if no data has ever been received for
        the modality.
        """
        entry = self._entries.get(modality_id)
        if entry is None:
            return float("inf")
        staleness = max(0.0, query_time - entry.timestamp)
        entry.staleness = staleness
        return staleness

    def get_all_embeddings(
        self, query_time: float
    ) -> dict[str, Optional[ModalityEntry]]:
        """Return a snapshot of all entries with updated staleness values.

        Args:
            query_time: Reference time for staleness computation.

        Returns:
            Dict mapping modality id to :class:`ModalityEntry` or
            ``None`` if no data has been received.
        """
        for mid in MODALITY_IDS:
            self.get_staleness(mid, query_time)
        return dict(self._entries)

    def reset(self) -> None:
        """Clear all stored embeddings (e.g. between episodes)."""
        for mid in MODALITY_IDS:
            self._entries[mid] = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _validate_modality(self, modality_id: str) -> None:
        if modality_id not in self._entries:
            raise ValueError(
                f"Unknown modality '{modality_id}'. "
                f"Expected one of {MODALITY_IDS}"
            )

    def __repr__(self) -> str:  # pragma: no cover
        active = [mid for mid, e in self._entries.items() if e is not None]
        return f"EmbeddingRegistry(active={active})"
