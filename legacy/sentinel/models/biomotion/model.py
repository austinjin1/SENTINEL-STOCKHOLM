"""Complete BioMotion encoder — SENTINEL's 5th modality.

Encodes aquatic organism behavioral trajectories (pose keypoints +
behavioral features) from multiple species into a shared 256-dim
embedding for multi-modal fusion, with integrated diffusion-based
anomaly detection.

Output contract::

    {
        "embedding":            Tensor[B, 256],          # main embedding
        "fusion_embedding":     Tensor[B, 256],          # projected for fusion
        "anomaly_score":        Tensor[B],               # ensemble anomaly
        "per_organism_scores":  dict[str, Tensor],       # per-species anomaly
        "denoising_difficulty": Tensor[B],               # raw diffusion anomaly
        "organism_embeddings":  dict[str, Tensor],       # per-species embeddings
    }
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .multi_organism import (
    MultiOrganismEnsemble,
    SPECIES_ORDER,
    SPECIES_FEATURE_DIM,
)
from .trajectory_encoder import EMBED_DIM

SHARED_EMBED_DIM: int = 256


class BioMotionEncoder(nn.Module):
    """Complete BioMotion modality encoder for SENTINEL.

    Wraps the ``MultiOrganismEnsemble`` (which in turn wraps per-species
    ``OrganismEncoder`` instances containing ``PoseEncoder`` and
    ``TrajectoryDiffusionEncoder``), adds the standard projection head
    for the fusion embedding space, and exposes the canonical SENTINEL
    encoder interface.

    Args:
        species_list: Ordered list of species to model.
        embed_dim: Internal embedding dimension (native dim).
        shared_embed_dim: Fusion embedding dimension (must be 256).
        species_feature_dims: Per-species behavioral feature dimensions.
        nhead: Cross-organism attention heads.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        species_list: list[str] | None = None,
        embed_dim: int = EMBED_DIM,
        shared_embed_dim: int = SHARED_EMBED_DIM,
        species_feature_dims: dict[str, int] | None = None,
        nhead: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.species_list = species_list or list(SPECIES_ORDER)
        self.embed_dim = embed_dim
        self.shared_embed_dim = shared_embed_dim

        # Multi-organism ensemble backbone
        self.ensemble = MultiOrganismEnsemble(
            species_list=self.species_list,
            embed_dim=embed_dim,
            nhead=nhead,
            species_feature_dims=species_feature_dims,
            dropout=dropout,
        )

        # Projection to shared fusion embedding space
        # Contract: Linear -> GELU -> LayerNorm -> Linear -> LayerNorm
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, shared_embed_dim),
            nn.GELU(),
            nn.LayerNorm(shared_embed_dim),
            nn.Linear(shared_embed_dim, shared_embed_dim),
            nn.LayerNorm(shared_embed_dim),
        )

        self._init_projection()

    def _init_projection(self) -> None:
        """Xavier-initialise all linear layers in the projection head."""
        for m in self.projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        organism_inputs: dict[str, dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Full forward pass through the BioMotion encoder.

        Args:
            organism_inputs: Dict keyed by species name, each value a dict
                containing:

                - ``"keypoints"``: ``(B, T, n_keypoints, 2)``
                - ``"features"``: ``(B, T, feature_dim)``
                - ``"timestamps"``: (optional) ``(B, T)``
                - ``"padding_mask"``: (optional) ``(B, T)``

        Returns:
            Dict with:

            - ``"embedding"``: ``(B, 256)`` — main embedding from the
              cross-organism attention fusion.
            - ``"fusion_embedding"``: ``(B, 256)`` — projected embedding
              for the SENTINEL multi-modal fusion layer.
            - ``"anomaly_score"``: ``(B,)`` — ensemble anomaly score
              combining all organisms.
            - ``"per_organism_scores"``: ``{species: (B,)}`` — per-species
              diffusion-based anomaly scores.
            - ``"denoising_difficulty"``: ``(B,)`` — mean per-organism
              denoising difficulty (raw diffusion anomaly signal).
            - ``"organism_embeddings"``: ``{species: (B, embed_dim)}`` —
              per-species fused embeddings before cross-organism attention.
        """
        ensemble_out = self.ensemble(organism_inputs)

        embedding = ensemble_out["embedding"]  # (B, embed_dim)
        fusion_embedding = self.projection(embedding)  # (B, shared_embed_dim)

        # Compute mean denoising difficulty across present organisms
        per_scores = ensemble_out["per_organism_scores"]
        if per_scores:
            denoising_difficulty = torch.stack(
                list(per_scores.values()), dim=0
            ).mean(dim=0)
        else:
            B = embedding.shape[0]
            denoising_difficulty = torch.zeros(B, device=embedding.device)

        return {
            "embedding": embedding,
            "fusion_embedding": fusion_embedding,
            "anomaly_score": ensemble_out["ensemble_anomaly_score"],
            "per_organism_scores": per_scores,
            "denoising_difficulty": denoising_difficulty,
            "organism_embeddings": ensemble_out["organism_embeddings"],
        }

    def forward_single_species(
        self,
        species: str,
        keypoints: torch.Tensor,
        features: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Convenience method for single-species inference.

        Args:
            species: Species name.
            keypoints: ``(B, T, n_keypoints, 2)``.
            features: ``(B, T, feature_dim)``.
            timestamps: (optional) ``(B, T)``.
            padding_mask: (optional) ``(B, T)``.

        Returns:
            Same output dict as :meth:`forward`.
        """
        organism_inputs = {
            species: {
                "keypoints": keypoints,
                "features": features,
            }
        }
        if timestamps is not None:
            organism_inputs[species]["timestamps"] = timestamps
        if padding_mask is not None:
            organism_inputs[species]["padding_mask"] = padding_mask

        return self.forward(organism_inputs)
