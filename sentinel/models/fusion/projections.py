"""Modality-specific projection layers to shared embedding space.

Each upstream encoder produces embeddings of different native
dimensionality.  These projection heads map every modality into a
common ``d=256`` space so that cross-modal attention can operate
uniformly.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from sentinel.models.fusion.embedding_registry import (
    MODALITY_IDS,
    SHARED_EMBEDDING_DIM,
)

# Native embedding dimensions produced by each encoder.
NATIVE_DIMS: Dict[str, int] = {
    "satellite": 384,   # ViT-S/16 CLS token
    "sensor": 256,      # Temporal transformer
    "microbial": 256,   # Community encoder
    "molecular": 128,   # Pathway encoder
}


class ModalityProjection(nn.Module):
    """Linear + LayerNorm projection from native dim to shared space.

    Args:
        in_dim: Native embedding dimensionality.
        out_dim: Shared space dimensionality.
    """

    def __init__(self, in_dim: int, out_dim: int = SHARED_EMBEDDING_DIM) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project input embedding(s) to shared space.

        Args:
            x: Tensor of shape ``[..., in_dim]``.

        Returns:
            Projected tensor of shape ``[..., out_dim]``.
        """
        return self.norm(self.linear(x))


class ProjectionBank(nn.Module):
    """Collection of per-modality projection heads.

    Maintains one :class:`ModalityProjection` for every modality in
    :data:`MODALITY_IDS` and exposes a dictionary-style interface.

    Args:
        native_dims: Mapping from modality id to native embedding dim.
            Defaults to :data:`NATIVE_DIMS`.
        shared_dim: Target shared dimensionality.  Default 256.
    """

    def __init__(
        self,
        native_dims: Dict[str, int] | None = None,
        shared_dim: int = SHARED_EMBEDDING_DIM,
    ) -> None:
        super().__init__()
        native_dims = native_dims or NATIVE_DIMS
        self.shared_dim = shared_dim
        self.projections = nn.ModuleDict(
            {
                mid: ModalityProjection(native_dims[mid], shared_dim)
                for mid in MODALITY_IDS
            }
        )

    def forward(
        self, modality_id: str, embedding: torch.Tensor
    ) -> torch.Tensor:
        """Project a single modality embedding.

        Args:
            modality_id: Identifier in :data:`MODALITY_IDS`.
            embedding: Tensor of shape ``[..., native_dim]``.

        Returns:
            Tensor of shape ``[..., shared_dim]``.

        Raises:
            KeyError: If *modality_id* is not recognized.
        """
        return self.projections[modality_id](embedding)

    def project_all(
        self, embeddings: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Project a batch of modality embeddings.

        Args:
            embeddings: Mapping from modality id to native embedding.

        Returns:
            Mapping from modality id to projected embedding.
        """
        return {
            mid: self.forward(mid, emb)
            for mid, emb in embeddings.items()
        }
