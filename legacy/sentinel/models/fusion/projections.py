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
    "behavioral": 256,  # Behavioral pattern encoder
}


class ModalityProjection(nn.Module):
    """Nonlinear projection from native dim to shared space.

    BUGFIX (audit item b): the previous implementation was a single
    ``LayerNorm(Linear(x))`` head, i.e. a purely *linear* map into the
    shared space.  Because every downstream fusion operation the GBM
    baseline exploited relies on a nonlinear interaction between the
    projected modalities, a linear projection head throws away exactly
    the representational capacity that lets fusion beat a single
    modality.  The head is now ``Linear -> GELU -> LayerNorm -> Linear``
    with a residual-friendly output ``LayerNorm``, preserving the
    ``[..., out_dim]`` output shape/dim.

    Args:
        in_dim: Native embedding dimensionality.
        out_dim: Shared space dimensionality.
        hidden_dim: Width of the nonlinear bottleneck.  Defaults to
            ``out_dim``.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = SHARED_EMBEDDING_DIM,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or out_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.hidden_norm = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        for linear in (self.fc1, self.fc2):
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project input embedding(s) to shared space.

        Args:
            x: Tensor of shape ``[..., in_dim]``.

        Returns:
            Projected tensor of shape ``[..., out_dim]``.
        """
        h = self.hidden_norm(self.act(self.fc1(x)))
        return self.norm(self.fc2(h))


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
