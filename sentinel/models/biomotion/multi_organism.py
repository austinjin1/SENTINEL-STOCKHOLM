"""Multi-organism ensemble with shared anomaly reasoning.

Different aquatic organisms detect different contaminant classes:
  - Daphnia magna  -> organophosphates, carbamates
  - Mussels        -> heavy metals (Cu, Zn, Pb, Cd)
  - Fish           -> neurotoxins, general toxicants

This module provides:
  - ``OrganismEncoder``: species-specific encoder wrapping PoseEncoder +
    TrajectoryDiffusionEncoder with tuned parameters.
  - ``MultiOrganismEnsemble``: cross-organism attention for contaminant
    class inference from joint anomaly patterns.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pose_encoder import PoseEncoder, SPECIES_KEYPOINTS, POSE_DIM
from .trajectory_encoder import TrajectoryDiffusionEncoder, EMBED_DIM


# Default behavioral feature dimensions per species.
SPECIES_FEATURE_DIM: dict[str, int] = {
    "daphnia": 16,
    "mussel": 8,
    "fish": 32,
}

# Ordered species list for consistent indexing.
SPECIES_ORDER: list[str] = ["daphnia", "mussel", "fish"]


# ---------------------------------------------------------------------------
# Per-organism encoder
# ---------------------------------------------------------------------------


class OrganismEncoder(nn.Module):
    """Species-specific trajectory encoder.

    Wraps a PoseEncoder and TrajectoryDiffusionEncoder with species-tuned
    parameters.  Fuses pose and trajectory embeddings into a single
    per-organism representation.

    Args:
        species: Species name (``"daphnia"``, ``"mussel"``, ``"fish"``).
        n_keypoints: Number of pose keypoints for this species.
        feature_dim: Per-frame behavioral feature dimension.
        pose_dim: Pose encoder output dimension.
        embed_dim: Trajectory encoder / output embedding dimension.
        pose_nhead: Pose transformer heads.
        pose_num_layers: Pose transformer layers.
        traj_nhead: Trajectory denoiser heads.
        traj_num_layers: Trajectory denoiser layers.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        species: str,
        n_keypoints: int | None = None,
        feature_dim: int | None = None,
        pose_dim: int = POSE_DIM,
        embed_dim: int = EMBED_DIM,
        pose_nhead: int = 4,
        pose_num_layers: int = 3,
        traj_nhead: int = 4,
        traj_num_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.species = species
        self.n_keypoints = n_keypoints or SPECIES_KEYPOINTS.get(species, 22)
        self.feature_dim = feature_dim or SPECIES_FEATURE_DIM.get(species, 16)
        self.embed_dim = embed_dim

        # Pose encoder
        self.pose_encoder = PoseEncoder(
            pose_dim=pose_dim,
            max_keypoints=self.n_keypoints,
            nhead=pose_nhead,
            num_layers=pose_num_layers,
            dropout=dropout,
        )

        # Trajectory diffusion encoder
        self.trajectory_encoder = TrajectoryDiffusionEncoder(
            feature_dim=self.feature_dim,
            embed_dim=embed_dim,
            nhead=traj_nhead,
            num_layers=traj_num_layers,
            dropout=dropout,
        )

        # Fusion of pose and trajectory embeddings
        self.fusion = nn.Sequential(
            nn.Linear(pose_dim + embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )
        self._init_fusion()

    def _init_fusion(self) -> None:
        for m in self.fusion.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        keypoints: torch.Tensor,
        features: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass for a single organism species.

        Args:
            keypoints: ``(B, T, n_keypoints, 2)`` pose keypoints.
            features: ``(B, T, feature_dim)`` behavioral features.
            timestamps: ``(B, T)`` frame timestamps in seconds.
            padding_mask: ``(B, T)`` True for padded positions.

        Returns:
            Dict with:
              - ``"embedding"``: fused embedding ``(B, embed_dim)``
              - ``"anomaly_score"``: diffusion-based anomaly ``(B,)``
              - ``"pose_embedding"``: pose-only embedding ``(B, pose_dim)``
              - ``"trajectory_embedding"``: trajectory-only embedding ``(B, embed_dim)``
        """
        pose_emb = self.pose_encoder(keypoints, timestamps, padding_mask)
        traj_emb = self.trajectory_encoder.forward_encode(features, padding_mask)
        anomaly_score = self.trajectory_encoder.compute_anomaly_score(
            features, padding_mask
        )

        # Fuse pose and trajectory
        combined = torch.cat([pose_emb, traj_emb], dim=-1)
        fused = self.fusion(combined)

        return {
            "embedding": fused,
            "anomaly_score": anomaly_score,
            "pose_embedding": pose_emb,
            "trajectory_embedding": traj_emb,
        }


# ---------------------------------------------------------------------------
# Cross-organism attention
# ---------------------------------------------------------------------------


class CrossOrganismAttention(nn.Module):
    """Cross-organism attention for joint anomaly reasoning.

    Given per-organism embeddings and anomaly scores, computes
    attention-weighted fusion.  The key insight: if Daphnia shows an
    anomaly but fish do not, this pattern is informative for contaminant
    class inference (e.g. organophosphates vs neurotoxins).

    Args:
        embed_dim: Per-organism embedding dimension.
        num_species: Number of organism types.
        nhead: Number of attention heads.
    """

    def __init__(
        self,
        embed_dim: int = EMBED_DIM,
        num_species: int = len(SPECIES_ORDER),
        nhead: int = 4,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        # Anomaly score embedding: project scalar score to embed_dim
        self.score_proj = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Species-type embedding (learnable)
        self.species_embed = nn.Embedding(num_species, embed_dim)

        # Cross-attention (self-attention across organism tokens)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=nhead,
            batch_first=True,
        )
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)

        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        organism_embeddings: list[torch.Tensor],
        anomaly_scores: list[torch.Tensor],
        organism_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Cross-organism attention fusion.

        Args:
            organism_embeddings: List of ``(B, embed_dim)`` per-organism.
            anomaly_scores: List of ``(B,)`` per-organism anomaly scores.
            organism_mask: ``(B, num_species)`` True for organisms not
                present in this batch sample.

        Returns:
            Fused embedding ``(B, embed_dim)``.
        """
        B = organism_embeddings[0].shape[0]
        S = len(organism_embeddings)
        device = organism_embeddings[0].device

        # Stack organism tokens: (B, S, embed_dim)
        tokens = torch.stack(organism_embeddings, dim=1)

        # Add species-type embeddings
        species_idx = torch.arange(S, device=device)
        species_emb = self.species_embed(species_idx).unsqueeze(0)  # (1, S, embed_dim)
        tokens = tokens + species_emb

        # Add anomaly score information
        scores = torch.stack(anomaly_scores, dim=1).unsqueeze(-1)  # (B, S, 1)
        score_emb = self.score_proj(scores)  # (B, S, embed_dim)
        tokens = tokens + score_emb

        # Cross-attention across organisms
        attn_out, _ = self.cross_attn(
            tokens, tokens, tokens,
            key_padding_mask=organism_mask,
        )
        tokens = self.layer_norm1(tokens + attn_out)
        tokens = self.layer_norm2(tokens + self.ffn(tokens))

        # Pool across organisms (masked mean)
        if organism_mask is not None:
            valid = (~organism_mask).unsqueeze(-1).float()
            fused = (tokens * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        else:
            fused = tokens.mean(dim=1)

        return fused  # (B, embed_dim)


# ---------------------------------------------------------------------------
# Multi-organism ensemble
# ---------------------------------------------------------------------------


class MultiOrganismEnsemble(nn.Module):
    """Multi-organism ensemble with shared anomaly reasoning.

    Maintains one OrganismEncoder per species and a cross-organism attention
    layer for joint anomaly reasoning and contaminant-class inference.

    Args:
        species_list: Ordered list of species names.
        embed_dim: Shared embedding dimension.
        nhead: Cross-attention heads.
        species_feature_dims: Per-species behavioral feature dimensions.
            Defaults to ``SPECIES_FEATURE_DIM``.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        species_list: list[str] | None = None,
        embed_dim: int = EMBED_DIM,
        nhead: int = 4,
        species_feature_dims: dict[str, int] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.species_list = species_list or list(SPECIES_ORDER)
        self.embed_dim = embed_dim

        feat_dims = species_feature_dims or SPECIES_FEATURE_DIM

        # Per-organism encoders (as ModuleDict for proper parameter registration)
        self.organism_encoders = nn.ModuleDict()
        for sp in self.species_list:
            self.organism_encoders[sp] = OrganismEncoder(
                species=sp,
                feature_dim=feat_dims.get(sp, 16),
                embed_dim=embed_dim,
                dropout=dropout,
            )

        # Cross-organism attention
        self.cross_attention = CrossOrganismAttention(
            embed_dim=embed_dim,
            num_species=len(self.species_list),
            nhead=nhead,
        )

        # Ensemble anomaly scorer: from per-organism scores + fused embedding
        self.anomaly_head = nn.Sequential(
            nn.Linear(embed_dim + len(self.species_list), embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
        )
        self._init_anomaly_head()

    def _init_anomaly_head(self) -> None:
        for m in self.anomaly_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        organism_inputs: dict[str, dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Forward pass through the multi-organism ensemble.

        Args:
            organism_inputs: Dict keyed by species name, each containing:
              - ``"keypoints"``: ``(B, T, n_keypoints, 2)``
              - ``"features"``: ``(B, T, feature_dim)``
              - ``"timestamps"``: (optional) ``(B, T)``
              - ``"padding_mask"``: (optional) ``(B, T)``

        Returns:
            Dict with:
              - ``"embedding"``: cross-organism fused ``(B, embed_dim)``
              - ``"per_organism_scores"``: ``{species: (B,)}``
              - ``"ensemble_anomaly_score"``: ``(B,)``
              - ``"organism_embeddings"``: ``{species: (B, embed_dim)}``
        """
        embeddings: list[torch.Tensor] = []
        scores: list[torch.Tensor] = []
        per_organism_scores: dict[str, torch.Tensor] = {}
        organism_embeddings: dict[str, torch.Tensor] = {}

        for sp in self.species_list:
            if sp not in organism_inputs:
                # Species not present — skip (will need masking)
                continue
            inp = organism_inputs[sp]
            encoder: OrganismEncoder = self.organism_encoders[sp]  # type: ignore[assignment]
            out = encoder(
                keypoints=inp["keypoints"],
                features=inp["features"],
                timestamps=inp.get("timestamps"),
                padding_mask=inp.get("padding_mask"),
            )
            embeddings.append(out["embedding"])
            scores.append(out["anomaly_score"])
            per_organism_scores[sp] = out["anomaly_score"]
            organism_embeddings[sp] = out["embedding"]

        # Handle case where not all species are present: pad with zeros
        B = embeddings[0].shape[0]
        device = embeddings[0].device

        all_embeddings: list[torch.Tensor] = []
        all_scores: list[torch.Tensor] = []
        mask_list: list[bool] = []

        for sp in self.species_list:
            if sp in organism_embeddings:
                all_embeddings.append(organism_embeddings[sp])
                all_scores.append(per_organism_scores[sp])
                mask_list.append(False)
            else:
                all_embeddings.append(torch.zeros(B, self.embed_dim, device=device))
                all_scores.append(torch.zeros(B, device=device))
                mask_list.append(True)

        organism_mask: Optional[torch.Tensor] = None
        if any(mask_list):
            organism_mask = torch.tensor(
                mask_list, device=device, dtype=torch.bool
            ).unsqueeze(0).expand(B, -1)

        # Cross-organism attention fusion
        fused = self.cross_attention(all_embeddings, all_scores, organism_mask)

        # Ensemble anomaly score
        score_vec = torch.stack(all_scores, dim=-1)  # (B, num_species)
        anomaly_input = torch.cat([fused, score_vec], dim=-1)
        ensemble_score = self.anomaly_head(anomaly_input).squeeze(-1)  # (B,)

        return {
            "embedding": fused,
            "per_organism_scores": per_organism_scores,
            "ensemble_anomaly_score": ensemble_score,
            "organism_embeddings": organism_embeddings,
        }
