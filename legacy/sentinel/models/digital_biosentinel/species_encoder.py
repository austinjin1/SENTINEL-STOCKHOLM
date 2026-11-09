"""Species identity encoder using hierarchical taxonomic embedding.

Encodes species into a 64-dimensional embedding that respects the
taxonomic hierarchy: phylum > class > order > family > genus > species.

This means closely related species (same genus) produce similar
embeddings, enabling the dose-response model to generalise across
species even when direct toxicity data is sparse.

Training data source: ECOTOX species table (~13,000 species).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Taxonomic rank names in order of increasing specificity
# ---------------------------------------------------------------------------
TAXONOMIC_RANKS: List[str] = [
    "phylum",
    "class",
    "order",
    "family",
    "genus",
    "species",
]

NUM_RANKS = len(TAXONOMIC_RANKS)

# ---------------------------------------------------------------------------
# Trophic-level categories (ordinal, but we embed them)
# ---------------------------------------------------------------------------
TROPHIC_LEVELS: List[str] = [
    "primary_producer",   # algae, plants
    "primary_consumer",   # herbivorous invertebrates
    "secondary_consumer", # predatory invertebrates, small fish
    "tertiary_consumer",  # large fish, top predators
    "decomposer",         # bacteria, fungi
]

TROPHIC_LEVEL_TO_IDX: Dict[str, int] = {
    lvl: idx for idx, lvl in enumerate(TROPHIC_LEVELS)
}


class SpeciesEncoder(nn.Module):
    """Hierarchical species encoder.

    Each taxonomic rank has its own embedding table.  The species
    representation is built by summing the per-rank embeddings
    (with learned scale weights) and adding a trophic-level embedding,
    then projecting through a small MLP to the final embedding space.

    Parameters
    ----------
    vocab_sizes : Dict[str, int]
        Mapping from rank name to vocabulary size for that rank.
        E.g. ``{"phylum": 40, "class": 120, ...}``.
    embedding_dim : int
        Output embedding dimensionality (default 64).
    rank_embed_dim : int
        Internal per-rank embedding dimensionality (default 32).
    num_trophic_levels : int
        Number of trophic-level categories (default 5).
    dropout : float
        Dropout in the projection MLP.
    """

    def __init__(
        self,
        vocab_sizes: Dict[str, int],
        embedding_dim: int = 64,
        rank_embed_dim: int = 32,
        num_trophic_levels: int = len(TROPHIC_LEVELS),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.rank_embed_dim = rank_embed_dim

        # --- Per-rank embedding tables -----------------------------------------
        # Index 0 in each table is reserved for "unknown / missing".
        self.rank_embeddings = nn.ModuleDict()
        for rank in TAXONOMIC_RANKS:
            vs = vocab_sizes.get(rank, 1)
            self.rank_embeddings[rank] = nn.Embedding(
                num_embeddings=vs + 1,  # +1 for unknown at 0
                embedding_dim=rank_embed_dim,
                padding_idx=0,
            )

        # Learned importance weight per rank (higher for more specific ranks)
        self.rank_weights = nn.Parameter(torch.ones(NUM_RANKS))

        # --- Trophic-level embedding -------------------------------------------
        self.trophic_embedding = nn.Embedding(
            num_embeddings=num_trophic_levels,
            embedding_dim=rank_embed_dim,
        )

        # --- Projection MLP ---------------------------------------------------
        # Input: rank_embed_dim (summed ranks + trophic)
        self.projection = nn.Sequential(
            nn.Linear(rank_embed_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, embedding_dim),
        )

        self.layer_norm = nn.LayerNorm(embedding_dim)
        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        for rank in TAXONOMIC_RANKS:
            emb = self.rank_embeddings[rank]
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)
            with torch.no_grad():
                emb.weight[0].zero_()
        nn.init.normal_(self.trophic_embedding.weight, mean=0.0, std=0.02)
        # Initialise rank weights so deeper (more specific) ranks start heavier
        with torch.no_grad():
            for i in range(NUM_RANKS):
                self.rank_weights[i] = 1.0 + 0.2 * i  # phylum=1.0 … species=2.0
        for module in self.projection:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    def forward(
        self,
        taxonomy_indices: Dict[str, torch.Tensor],
        trophic_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a batch of species.

        Parameters
        ----------
        taxonomy_indices : Dict[str, Tensor[B]]
            Per-rank vocabulary indices.  Keys must be from
            ``TAXONOMIC_RANKS``.  Use ``0`` for unknown / missing.
        trophic_idx : Tensor[B]  (long)
            Trophic-level index (see ``TROPHIC_LEVEL_TO_IDX``).

        Returns
        -------
        embedding : Tensor[B, embedding_dim]
        """
        # Softmax over rank weights so they sum to 1
        weights = F.softmax(self.rank_weights, dim=0)  # [NUM_RANKS]

        combined = None
        for i, rank in enumerate(TAXONOMIC_RANKS):
            if rank not in taxonomy_indices:
                continue
            idx = taxonomy_indices[rank]  # [B]
            emb = self.rank_embeddings[rank](idx)  # [B, rank_embed_dim]
            weighted = weights[i] * emb
            if combined is None:
                combined = weighted
            else:
                combined = combined + weighted

        if combined is None:
            raise ValueError(
                "taxonomy_indices must contain at least one rank from "
                f"{TAXONOMIC_RANKS}"
            )

        # Add trophic-level embedding
        trophic_emb = self.trophic_embedding(trophic_idx)  # [B, rank_embed_dim]
        combined = combined + trophic_emb

        embedding = self.projection(combined)  # [B, embedding_dim]
        embedding = self.layer_norm(embedding)

        return embedding

    # ------------------------------------------------------------------
    def taxonomic_distance_loss(
        self,
        embeddings: torch.Tensor,
        taxonomy_indices: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Auxiliary loss: species sharing more taxonomic ranks should
        have closer embeddings.

        Computes a margin-ranking loss where the margin is proportional
        to the taxonomic distance between species pairs.

        Parameters
        ----------
        embeddings : Tensor[B, D]
        taxonomy_indices : Dict[str, Tensor[B]]

        Returns
        -------
        loss : scalar Tensor
        """
        B = embeddings.size(0)
        if B < 2:
            return embeddings.new_tensor(0.0)

        # Compute pairwise taxonomic overlap (count of shared ranks)
        overlap = torch.zeros(B, B, device=embeddings.device)
        for rank in TAXONOMIC_RANKS:
            if rank not in taxonomy_indices:
                continue
            idx = taxonomy_indices[rank]  # [B]
            # Two species share this rank if both are non-zero and equal
            valid = idx != 0  # [B]
            match = (idx.unsqueeze(0) == idx.unsqueeze(1)) & valid.unsqueeze(
                0
            ) & valid.unsqueeze(1)
            overlap += match.float()

        # Normalise overlap to [0, 1]
        overlap = overlap / NUM_RANKS

        # Pairwise L2 distance in embedding space
        normed = F.normalize(embeddings, dim=-1)
        dist = torch.cdist(normed.unsqueeze(0), normed.unsqueeze(0)).squeeze(0)

        # Loss: species with high overlap should have low distance
        # target_dist = 1 - overlap (close relatives -> small target distance)
        target_dist = 1.0 - overlap

        # Mask out diagonal
        mask = ~torch.eye(B, dtype=torch.bool, device=embeddings.device)
        loss = F.mse_loss(dist[mask], target_dist[mask])

        return loss
