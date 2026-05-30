"""Sentinel Species Health Index -- Phase 3.1 of SENTINEL 2.0.

Hierarchical occupancy + abundance model that predicts daily health
indices for six keystone freshwater indicator species.  The model
consumes the 256-dimensional multimodal environmental embedding
produced by the SENTINEL fusion layer (or any single encoder) and
optional site-level covariates, then outputs per-species:

* Health score (0--100 continuous scale)
* 90 % confidence interval (via MC-dropout)
* Occupancy probability P(species present | environment)
* Temporal trend (declining / stable / improving) with rate
* Primary environmental stressor attribution with confidence

Architecture overview::

    SENTINEL embedding  (B, 256)
    Site covariates      (B, 5)   [lat, lon, elev, stream_order, drainage_area]
          |
          v
    EnvironmentalEncoder  -->  env_repr (B, 256)
          |
          +-- SpeciesEmbedding (6 x 64-d learnable)
          |
          v
    HierarchicalOccupancyModel
        Level 1: occupancy    P(present | env)
        Level 2: abundance    E[N | present, env]
        Level 3: health score H(abundance, env quality)
          |
          v
    StressorAttribution  -->  primary stressor per species
    TrendEstimator       -->  temporal trend from health history

Uncertainty estimation uses MC-dropout (20 stochastic passes at
inference) following Gal & Ghahramani (2016).

Target species
--------------
1. Freshwater mussels   (Unionidae)                -- federally endangered
2. Mayflies             (Ephemeroptera)             -- textbook bioindicator
3. Brook trout          (Salvelinus fontinalis)     -- cold-water indicator
4. Hellbender           (Cryptobranchus alleganiensis) -- skin-breathing
5. Freshwater pearl mussel (Margaritifera margaritifera) -- filter-feeding
6. American eel         (Anguilla rostrata)         -- catadromous integrator
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SHARED_EMBEDDING_DIM: int = 256
SPECIES_EMBEDDING_DIM: int = 64
SITE_COVARIATE_DIM: int = 5     # lat, lon, elevation, stream_order, drainage_area
SITE_PROJECTED_DIM: int = 64
HIDDEN_DIMS: Tuple[int, ...] = (512, 256, 128)
NUM_SPECIES: int = 6
MC_DROPOUT_SAMPLES: int = 20

# Stressor categories that the model can attribute.
STRESSOR_CATEGORIES: Tuple[str, ...] = (
    "temperature",
    "dissolved_oxygen",
    "pH",
    "turbidity",
    "nutrients",
    "metals",
    "organic_pollutants",
)
NUM_STRESSORS: int = len(STRESSOR_CATEGORIES)

# Trend labels (ordinal).
TREND_LABELS: Tuple[str, ...] = ("declining", "stable", "improving")
NUM_TRENDS: int = len(TREND_LABELS)


# ---------------------------------------------------------------------------
# Species metadata
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KeystoneSpecies:
    """Metadata for a keystone freshwater indicator species."""

    index: int
    common_name: str
    taxon: str
    role: str
    primary_sensitivity: str


KEYSTONE_SPECIES: Tuple[KeystoneSpecies, ...] = (
    KeystoneSpecies(
        index=0,
        common_name="Freshwater mussels",
        taxon="Unionidae",
        role="Filter-feeding bivalves; federally endangered, 70% of NA species imperiled",
        primary_sensitivity="sediment, dissolved_oxygen, temperature",
    ),
    KeystoneSpecies(
        index=1,
        common_name="Mayflies",
        taxon="Ephemeroptera",
        role="Textbook aquatic bioindicator; sensitive to water quality degradation",
        primary_sensitivity="dissolved_oxygen, pH, organic_pollutants",
    ),
    KeystoneSpecies(
        index=2,
        common_name="Brook trout",
        taxon="Salvelinus fontinalis",
        role="Cold-water salmonid indicator; requires high DO, low temperature",
        primary_sensitivity="temperature, dissolved_oxygen, turbidity",
    ),
    KeystoneSpecies(
        index=3,
        common_name="Hellbender salamander",
        taxon="Cryptobranchus alleganiensis",
        role="Skin-breathing amphibian; hypersensitive to water quality",
        primary_sensitivity="dissolved_oxygen, temperature, metals",
    ),
    KeystoneSpecies(
        index=4,
        common_name="Freshwater pearl mussel",
        taxon="Margaritifera margaritifera",
        role="Long-lived filter-feeder; canary species for watershed health",
        primary_sensitivity="turbidity, nutrients, metals",
    ),
    KeystoneSpecies(
        index=5,
        common_name="American eel",
        taxon="Anguilla rostrata",
        role="Catadromous fish; integrates conditions across large watersheds",
        primary_sensitivity="temperature, organic_pollutants, turbidity",
    ),
)

SPECIES_INDEX_TO_NAME: Dict[int, str] = {
    sp.index: sp.common_name for sp in KEYSTONE_SPECIES
}

SPECIES_NAME_TO_INDEX: Dict[str, int] = {
    sp.common_name: sp.index for sp in KEYSTONE_SPECIES
}


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SpeciesHealthPrediction:
    """Health prediction for a single species at a single site.

    Attributes
    ----------
    species_name : str
        Common name of the species.
    health_score : float
        Continuous health index on a 0--100 scale.
    confidence_low : float
        Lower bound of the 90 % confidence interval.
    confidence_high : float
        Upper bound of the 90 % confidence interval.
    occupancy_prob : float
        Probability that the species is present at the site.
    trend : str
        One of ``"declining"``, ``"stable"``, ``"improving"``.
    trend_rate : float
        Estimated health-score change per month.
    primary_stressor : str
        Most impactful environmental stressor from
        :data:`STRESSOR_CATEGORIES`.
    stressor_confidence : float
        Confidence in the stressor attribution (0--1).
    """

    species_name: str
    health_score: float
    confidence_low: float
    confidence_high: float
    occupancy_prob: float
    trend: str
    trend_rate: float
    primary_stressor: str
    stressor_confidence: float


@dataclass
class SpeciesHealthOutput:
    """Batched tensor output from :class:`SentinelSpeciesHealthIndex`.

    All tensor shapes assume batch size ``B`` and ``S`` species.

    Attributes
    ----------
    health_scores : Tensor[B, S]
        Health scores in [0, 100].
    confidence_low : Tensor[B, S]
        Lower 90 % CI bound.
    confidence_high : Tensor[B, S]
        Upper 90 % CI bound.
    occupancy_probs : Tensor[B, S]
        Occupancy probabilities in [0, 1].
    abundance : Tensor[B, S]
        Conditional expected abundance (log-scale, given presence).
    stressor_logits : Tensor[B, S, num_stressors]
        Raw logits for stressor attribution.
    stressor_probs : Tensor[B, S, num_stressors]
        Softmax stressor probabilities.
    """

    health_scores: torch.Tensor
    confidence_low: torch.Tensor
    confidence_high: torch.Tensor
    occupancy_probs: torch.Tensor
    abundance: torch.Tensor
    stressor_logits: torch.Tensor
    stressor_probs: torch.Tensor


# ---------------------------------------------------------------------------
# Sub-module 1: Species Embedding
# ---------------------------------------------------------------------------

class SpeciesEmbedding(nn.Module):
    """Learnable embedding for the six keystone indicator species.

    Each species gets a 64-dimensional embedding that encodes its
    taxonomic identity, physiological traits, and known environmental
    sensitivities.  The embeddings enable cross-species transfer
    learning: the shared environmental encoder learns features that
    are relevant across all species, while the species embedding
    modulates the prediction heads to capture species-specific
    responses.

    Parameters
    ----------
    num_species : int
        Number of species (default 6).
    embedding_dim : int
        Dimensionality of each species vector (default 64).
    """

    def __init__(
        self,
        num_species: int = NUM_SPECIES,
        embedding_dim: int = SPECIES_EMBEDDING_DIM,
    ) -> None:
        super().__init__()
        self.num_species = num_species
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(
            num_embeddings=num_species,
            embedding_dim=embedding_dim,
        )
        self.layer_norm = nn.LayerNorm(embedding_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(
        self,
        species_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return species embeddings.

        Parameters
        ----------
        species_idx : Tensor[...], optional
            Long tensor of species indices.  If ``None``, returns the
            full embedding table ``[S, D]`` for all species.

        Returns
        -------
        Tensor
            Shape ``[..., D]`` if ``species_idx`` is given, or
            ``[S, D]`` for the full table.
        """
        if species_idx is None:
            species_idx = torch.arange(
                self.num_species,
                device=self.embedding.weight.device,
            )
        emb = self.embedding(species_idx)
        return self.layer_norm(emb)

    def pairwise_similarity(self) -> torch.Tensor:
        """Compute pairwise cosine similarity between all species.

        Useful for diagnosing whether the learned embeddings capture
        expected taxonomic relationships (e.g. both mussel species
        should be more similar to each other than to the eel).

        Returns
        -------
        Tensor[S, S]
            Cosine similarity matrix.
        """
        emb = self.forward()  # [S, D]
        normed = F.normalize(emb, dim=-1)
        return normed @ normed.T


# ---------------------------------------------------------------------------
# Sub-module 2: Environmental Encoder
# ---------------------------------------------------------------------------

class EnvironmentalEncoder(nn.Module):
    """Project SENTINEL embedding + site covariates to a shared space.

    Combines the 256-d multimodal environmental embedding from the
    SENTINEL fusion layer with site-level geographic covariates
    (latitude, longitude, elevation, stream order, drainage area)
    to produce a unified environmental representation.

    Parameters
    ----------
    embedding_dim : int
        Dimensionality of the SENTINEL embedding input (default 256).
    site_covariate_dim : int
        Number of site-level covariate features (default 5).
    site_projected_dim : int
        Dimensionality to project site covariates into (default 64).
    output_dim : int
        Output representation dimensionality (default 256).
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        embedding_dim: int = SHARED_EMBEDDING_DIM,
        site_covariate_dim: int = SITE_COVARIATE_DIM,
        site_projected_dim: int = SITE_PROJECTED_DIM,
        output_dim: int = SHARED_EMBEDDING_DIM,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

        # Project site covariates to a learned representation.
        self.site_projection = nn.Sequential(
            nn.Linear(site_covariate_dim, site_projected_dim),
            nn.GELU(),
            nn.LayerNorm(site_projected_dim),
        )

        # Fuse SENTINEL embedding with projected site covariates.
        fused_dim = embedding_dim + site_projected_dim
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fused_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(output_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        sentinel_embedding: torch.Tensor,
        site_covariates: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode environmental context.

        Parameters
        ----------
        sentinel_embedding : Tensor[B, 256]
            Multimodal embedding from the SENTINEL fusion layer or a
            single-modality encoder.
        site_covariates : Tensor[B, 5], optional
            Site-level geographic features ``[lat, lon, elevation,
            stream_order, drainage_area]``.  If ``None``, zeros are
            used (the model remains functional without site data).

        Returns
        -------
        Tensor[B, 256]
            Unified environmental representation.
        """
        B = sentinel_embedding.size(0)
        device = sentinel_embedding.device

        if site_covariates is None:
            site_covariates = torch.zeros(
                B, SITE_COVARIATE_DIM, device=device
            )

        site_repr = self.site_projection(site_covariates)  # [B, 64]
        fused = torch.cat([sentinel_embedding, site_repr], dim=-1)  # [B, 320]
        return self.fusion_mlp(fused)  # [B, 256]


# ---------------------------------------------------------------------------
# Sub-module 3: Hierarchical Occupancy Model
# ---------------------------------------------------------------------------

class HierarchicalOccupancyModel(nn.Module):
    """Three-level hierarchical predictor: occupancy -> abundance -> health.

    This is the core prediction engine.  It uses a shared environmental
    encoder backbone with species-specific prediction heads at each
    hierarchical level:

    * **Level 1 -- Occupancy**: Binary probability that the species is
      present at the site given current environmental conditions.
    * **Level 2 -- Abundance**: Expected abundance (log-scale) of the
      species conditional on presence.
    * **Level 3 -- Health score**: Continuous 0--100 index integrating
      abundance with environmental quality indicators.

    The backbone is shared across species, but each species has its own
    set of lightweight output heads.  The species embedding is
    concatenated with the environmental representation before the
    shared backbone, allowing the network to learn species-specific
    environmental responses.

    MC-dropout is applied throughout the backbone to enable Bayesian
    uncertainty estimation at inference time.

    Parameters
    ----------
    env_dim : int
        Dimensionality of the environmental representation (default 256).
    species_dim : int
        Dimensionality of species embeddings (default 64).
    hidden_dims : tuple of int
        Hidden layer sizes for the shared backbone (default (512, 256, 128)).
    num_species : int
        Number of species (default 6).
    dropout : float
        Dropout probability for MC-dropout layers (default 0.3).
    """

    def __init__(
        self,
        env_dim: int = SHARED_EMBEDDING_DIM,
        species_dim: int = SPECIES_EMBEDDING_DIM,
        hidden_dims: Tuple[int, ...] = HIDDEN_DIMS,
        num_species: int = NUM_SPECIES,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_species = num_species
        self.dropout_p = dropout

        # --- Shared backbone (env_repr + species_emb -> hidden) ----------------
        input_dim = env_dim + species_dim  # 256 + 64 = 320
        layers: list[nn.Module] = []
        in_features = input_dim
        for out_features in hidden_dims:
            layers.extend([
                nn.Linear(in_features, out_features),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_features = out_features
        self.backbone = nn.Sequential(*layers)

        backbone_out = hidden_dims[-1]  # 128

        # --- Per-species heads -------------------------------------------------
        # Each species gets its own lightweight heads so the shared backbone
        # learns transferable features while species-specific responses are
        # captured in the heads.
        self.occupancy_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(backbone_out, 64),
                nn.GELU(),
                nn.Linear(64, 1),
            )
            for _ in range(num_species)
        ])

        self.abundance_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(backbone_out, 64),
                nn.GELU(),
                nn.Linear(64, 1),
                nn.Softplus(),  # abundance is non-negative
            )
            for _ in range(num_species)
        ])

        self.health_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(backbone_out + 2, 64),  # +2 for occupancy + abundance
                nn.GELU(),
                nn.Linear(64, 1),
            )
            for _ in range(num_species)
        ])

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Bias occupancy toward moderate presence (logit ~0 -> 0.5 probability).
        for head in self.occupancy_heads:
            nn.init.zeros_(head[-1].bias)

    def forward(
        self,
        env_repr: torch.Tensor,
        species_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Hierarchical prediction for all species.

        Parameters
        ----------
        env_repr : Tensor[B, env_dim]
            Environmental representation from :class:`EnvironmentalEncoder`.
        species_emb : Tensor[S, species_dim]
            Embedding table for all ``S`` species.

        Returns
        -------
        occupancy : Tensor[B, S]
            Occupancy probabilities in [0, 1].
        abundance : Tensor[B, S]
            Conditional expected abundance (log-scale), non-negative.
        health_scores : Tensor[B, S]
            Health scores clamped to [0, 100].
        """
        B = env_repr.size(0)
        S = species_emb.size(0)

        occupancy_list: list[torch.Tensor] = []
        abundance_list: list[torch.Tensor] = []
        health_list: list[torch.Tensor] = []

        for s in range(S):
            # Broadcast species embedding across the batch.
            sp_emb = species_emb[s].unsqueeze(0).expand(B, -1)  # [B, 64]
            x = torch.cat([env_repr, sp_emb], dim=-1)  # [B, 320]

            h = self.backbone(x)  # [B, 128]

            # Level 1: occupancy
            occ_logit = self.occupancy_heads[s](h).squeeze(-1)  # [B]
            occ_prob = torch.sigmoid(occ_logit)

            # Level 2: abundance (conditional on presence)
            abund = self.abundance_heads[s](h).squeeze(-1)  # [B]

            # Level 3: health score (integrates backbone features + occ + abund)
            health_input = torch.cat([
                h,
                occ_prob.unsqueeze(-1),
                abund.unsqueeze(-1),
            ], dim=-1)  # [B, 130]
            health_raw = self.health_heads[s](health_input).squeeze(-1)  # [B]
            health_score = torch.sigmoid(health_raw) * 100.0  # [0, 100]

            occupancy_list.append(occ_prob)
            abundance_list.append(abund)
            health_list.append(health_score)

        occupancy = torch.stack(occupancy_list, dim=1)    # [B, S]
        abundance = torch.stack(abundance_list, dim=1)    # [B, S]
        health_scores = torch.stack(health_list, dim=1)   # [B, S]

        return occupancy, abundance, health_scores


# ---------------------------------------------------------------------------
# Sub-module 4: Stressor Attribution
# ---------------------------------------------------------------------------

class StressorAttribution(nn.Module):
    """Identify the primary environmental stressor for each species.

    Uses an attention mechanism over the SENTINEL embedding features
    to determine which environmental stressor category is most
    impactful for each species at each site.

    The model learns species-specific attention patterns: e.g. brook
    trout attend strongly to temperature-related features while
    mussels attend to sediment and dissolved oxygen features.

    Parameters
    ----------
    env_dim : int
        Environmental representation dimensionality (default 256).
    species_dim : int
        Species embedding dimensionality (default 64).
    num_stressors : int
        Number of stressor categories (default 7).
    num_attention_heads : int
        Number of attention heads for the stressor query (default 4).
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        env_dim: int = SHARED_EMBEDDING_DIM,
        species_dim: int = SPECIES_EMBEDDING_DIM,
        num_stressors: int = NUM_STRESSORS,
        num_attention_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_stressors = num_stressors
        self.env_dim = env_dim

        # Learnable query vectors for each stressor category.
        # These act as "stressor detectors" that attend to the
        # environmental representation.
        self.stressor_queries = nn.Parameter(
            torch.randn(num_stressors, env_dim) * 0.02
        )

        # Species-conditioned key/value projection.
        combined_dim = env_dim + species_dim  # 320
        self.key_proj = nn.Linear(combined_dim, env_dim)
        self.value_proj = nn.Linear(combined_dim, env_dim)

        # Final classification from attended features.
        self.classifier = nn.Sequential(
            nn.Linear(env_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

        self.scale = math.sqrt(env_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        env_repr: torch.Tensor,
        species_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Attribute primary stressor for each species.

        Parameters
        ----------
        env_repr : Tensor[B, env_dim]
            Environmental representation.
        species_emb : Tensor[S, species_dim]
            Species embedding table.

        Returns
        -------
        stressor_logits : Tensor[B, S, num_stressors]
            Raw logits for each stressor category.
        stressor_probs : Tensor[B, S, num_stressors]
            Softmax probabilities over stressor categories.
        """
        B = env_repr.size(0)
        S = species_emb.size(0)
        C = self.num_stressors

        logits_list: list[torch.Tensor] = []

        for s in range(S):
            sp_emb = species_emb[s].unsqueeze(0).expand(B, -1)  # [B, 64]
            combined = torch.cat([env_repr, sp_emb], dim=-1)  # [B, 320]

            # Project to key space.
            keys = self.key_proj(combined)   # [B, 256]
            values = self.value_proj(combined)  # [B, 256]

            # Compute attention: stressor queries attend to keys.
            # queries: [C, 256], keys: [B, 256] -> attn: [B, C]
            attn_scores = torch.matmul(
                self.stressor_queries, keys.T  # [C, 256] x [256, B] -> [C, B]
            ).T / self.scale  # [B, C]
            attn_weights = F.softmax(attn_scores, dim=-1)  # [B, C]

            # Weighted values per stressor: [B, C, 256]
            # attn_weights: [B, C] -> [B, C, 1], values: [B, 1, 256]
            attended = attn_weights.unsqueeze(-1) * values.unsqueeze(1)  # [B, C, 256]

            # Classify each stressor channel.
            stressor_logit = self.classifier(attended).squeeze(-1)  # [B, C]
            logits_list.append(stressor_logit)

        stressor_logits = torch.stack(logits_list, dim=1)  # [B, S, C]
        stressor_probs = F.softmax(stressor_logits, dim=-1)  # [B, S, C]

        return stressor_logits, stressor_probs


# ---------------------------------------------------------------------------
# Sub-module 5: Trend Estimator
# ---------------------------------------------------------------------------

class TrendEstimator(nn.Module):
    """Estimate temporal health trends from a sequence of health scores.

    Given a time series of health scores for a species at a site,
    classifies the trend as declining / stable / improving and
    estimates the rate of change (score units per month).

    The estimator uses a small 1-D convolutional network to capture
    local temporal patterns, followed by a GRU for longer-range
    dependencies.

    Parameters
    ----------
    input_dim : int
        Number of features per timestep (default 1, just health score).
    hidden_dim : int
        GRU hidden state size.
    num_trends : int
        Number of trend classes (default 3).
    min_sequence_length : int
        Minimum number of timesteps required (default 7).
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        num_trends: int = NUM_TRENDS,
        min_sequence_length: int = 7,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.min_sequence_length = min_sequence_length
        self.hidden_dim = hidden_dim

        # Temporal feature extraction (1D conv over time).
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.GELU(),
        )

        # Recurrent layer for longer dependencies.
        self.gru = nn.GRU(
            input_size=64,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )

        # Classification head.
        self.trend_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_trends),
        )

        # Rate regression head (score change per month).
        self.rate_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        health_sequence: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Estimate trend from health score sequence.

        Parameters
        ----------
        health_sequence : Tensor[B, T, 1] or Tensor[B, T]
            Time series of health scores.  ``T`` must be at least
            :attr:`min_sequence_length`.

        Returns
        -------
        trend_logits : Tensor[B, num_trends]
            Logits over trend classes.
        trend_probs : Tensor[B, num_trends]
            Softmax probabilities.
        trend_rate : Tensor[B]
            Estimated health-score change per month.
        """
        if health_sequence.dim() == 2:
            health_sequence = health_sequence.unsqueeze(-1)  # [B, T, 1]

        B, T, _ = health_sequence.shape

        # Conv1d expects [B, C, T].
        x = health_sequence.permute(0, 2, 1)  # [B, 1, T]
        x = self.temporal_conv(x)             # [B, 64, T]
        x = x.permute(0, 2, 1)               # [B, T, 64]

        # GRU: take final hidden state.
        _, h_n = self.gru(x)  # h_n: [1, B, hidden_dim]
        h = h_n.squeeze(0)    # [B, hidden_dim]

        trend_logits = self.trend_head(h)            # [B, num_trends]
        trend_probs = F.softmax(trend_logits, dim=-1)
        trend_rate = self.rate_head(h).squeeze(-1)   # [B]

        return trend_logits, trend_probs, trend_rate


# ---------------------------------------------------------------------------
# Main model: Sentinel Species Health Index
# ---------------------------------------------------------------------------

class SentinelSpeciesHealthIndex(nn.Module):
    """Hierarchical occupancy + abundance model for species health.

    This is the top-level model that composes all sub-modules into a
    single ``nn.Module``.  It can be used:

    * **Standalone**: given a SENTINEL embedding and optional site
      covariates, predict health indices for all six keystone species.
    * **As a head**: attached to the SENTINEL fusion model, consuming
      the ``fused_state`` output.

    Parameters
    ----------
    embedding_dim : int
        Dimensionality of the input SENTINEL embedding (default 256).
    species_embedding_dim : int
        Species embedding dimensionality (default 64).
    site_covariate_dim : int
        Number of site-level covariate features (default 5).
    hidden_dims : tuple of int
        Hidden layer sizes for the occupancy model backbone.
    num_species : int
        Number of target species (default 6).
    dropout : float
        Dropout probability for MC-dropout layers (default 0.3).
    mc_samples : int
        Number of MC-dropout forward passes for uncertainty (default 20).

    Example
    -------
    >>> model = SentinelSpeciesHealthIndex()
    >>> embedding = torch.randn(4, 256)  # batch of 4 sites
    >>> covariates = torch.randn(4, 5)
    >>> output = model(embedding, site_covariates=covariates)
    >>> output.health_scores.shape
    torch.Size([4, 6])
    """

    def __init__(
        self,
        embedding_dim: int = SHARED_EMBEDDING_DIM,
        species_embedding_dim: int = SPECIES_EMBEDDING_DIM,
        site_covariate_dim: int = SITE_COVARIATE_DIM,
        hidden_dims: Tuple[int, ...] = HIDDEN_DIMS,
        num_species: int = NUM_SPECIES,
        dropout: float = 0.3,
        mc_samples: int = MC_DROPOUT_SAMPLES,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_species = num_species
        self.mc_samples = mc_samples
        self.dropout_p = dropout

        # --- Sub-modules -------------------------------------------------------
        self.species_embedding = SpeciesEmbedding(
            num_species=num_species,
            embedding_dim=species_embedding_dim,
        )

        self.env_encoder = EnvironmentalEncoder(
            embedding_dim=embedding_dim,
            site_covariate_dim=site_covariate_dim,
            site_projected_dim=SITE_PROJECTED_DIM,
            output_dim=embedding_dim,
            dropout=dropout * 0.33,  # lighter dropout in encoder
        )

        self.occupancy_model = HierarchicalOccupancyModel(
            env_dim=embedding_dim,
            species_dim=species_embedding_dim,
            hidden_dims=hidden_dims,
            num_species=num_species,
            dropout=dropout,
        )

        self.stressor_attribution = StressorAttribution(
            env_dim=embedding_dim,
            species_dim=species_embedding_dim,
            num_stressors=NUM_STRESSORS,
            dropout=dropout * 0.33,
        )

        self.trend_estimator = TrendEstimator(
            input_dim=1,
            hidden_dim=64,
            num_trends=NUM_TRENDS,
            dropout=dropout * 0.33,
        )

    # ------------------------------------------------------------------
    # MC-dropout helpers
    # ------------------------------------------------------------------

    def _enable_dropout(self) -> None:
        """Set all Dropout modules to train mode (stochastic)."""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def _disable_dropout(self) -> None:
        """Set all Dropout modules back to eval mode (deterministic)."""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.eval()

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------

    def _single_forward(
        self,
        sentinel_embedding: torch.Tensor,
        site_covariates: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor]:
        """Single deterministic forward pass.

        Returns
        -------
        occupancy : Tensor[B, S]
        abundance : Tensor[B, S]
        health_scores : Tensor[B, S]
        stressor_logits : Tensor[B, S, C]
        stressor_probs : Tensor[B, S, C]
        """
        # Encode environment.
        env_repr = self.env_encoder(sentinel_embedding, site_covariates)

        # Get species embeddings (full table).
        species_emb = self.species_embedding()  # [S, 64]

        # Hierarchical prediction.
        occupancy, abundance, health_scores = self.occupancy_model(
            env_repr, species_emb,
        )

        # Stressor attribution.
        stressor_logits, stressor_probs = self.stressor_attribution(
            env_repr, species_emb,
        )

        return occupancy, abundance, health_scores, stressor_logits, stressor_probs

    def forward(
        self,
        sentinel_embedding: torch.Tensor,
        site_covariates: Optional[torch.Tensor] = None,
        use_mc_dropout: bool = False,
    ) -> SpeciesHealthOutput:
        """Forward pass with optional MC-dropout uncertainty.

        Parameters
        ----------
        sentinel_embedding : Tensor[B, 256]
            Multimodal environmental embedding from the SENTINEL fusion
            layer or any single-modality encoder.
        site_covariates : Tensor[B, 5], optional
            Site-level features: ``[lat, lon, elevation, stream_order,
            drainage_area]``.  Omit if not available.
        use_mc_dropout : bool
            If ``True`` and the model is in eval mode, run
            :attr:`mc_samples` stochastic forward passes and compute
            90 % confidence intervals from the empirical distribution.

        Returns
        -------
        SpeciesHealthOutput
            Batched predictions for all species.
        """
        if not use_mc_dropout or self.training:
            occ, abund, health, stressor_logits, stressor_probs = (
                self._single_forward(sentinel_embedding, site_covariates)
            )
            # Without MC-dropout, CI is a point estimate (no spread).
            return SpeciesHealthOutput(
                health_scores=health,
                confidence_low=health,
                confidence_high=health,
                occupancy_probs=occ,
                abundance=abund,
                stressor_logits=stressor_logits,
                stressor_probs=stressor_probs,
            )

        # --- MC-dropout inference ----------------------------------------------
        self._enable_dropout()

        mc_health: list[torch.Tensor] = []
        mc_occ: list[torch.Tensor] = []
        mc_abund: list[torch.Tensor] = []
        mc_stressor_logits: list[torch.Tensor] = []

        for _ in range(self.mc_samples):
            occ, abund, health, s_logits, _ = self._single_forward(
                sentinel_embedding, site_covariates,
            )
            mc_health.append(health)
            mc_occ.append(occ)
            mc_abund.append(abund)
            mc_stressor_logits.append(s_logits)

        self._disable_dropout()

        # Stack: [N, B, S] for health/occ/abund, [N, B, S, C] for stressors.
        mc_health_t = torch.stack(mc_health, dim=0)         # [N, B, S]
        mc_occ_t = torch.stack(mc_occ, dim=0)               # [N, B, S]
        mc_abund_t = torch.stack(mc_abund, dim=0)            # [N, B, S]
        mc_stressor_t = torch.stack(mc_stressor_logits, dim=0)  # [N, B, S, C]

        # Mean predictions.
        health_mean = mc_health_t.mean(dim=0)
        occ_mean = mc_occ_t.mean(dim=0)
        abund_mean = mc_abund_t.mean(dim=0)
        stressor_logits_mean = mc_stressor_t.mean(dim=0)
        stressor_probs_mean = F.softmax(stressor_logits_mean, dim=-1)

        # 90 % confidence interval: 5th and 95th percentiles.
        ci_low = torch.quantile(mc_health_t, 0.05, dim=0)
        ci_high = torch.quantile(mc_health_t, 0.95, dim=0)

        return SpeciesHealthOutput(
            health_scores=health_mean,
            confidence_low=ci_low,
            confidence_high=ci_high,
            occupancy_probs=occ_mean,
            abundance=abund_mean,
            stressor_logits=stressor_logits_mean,
            stressor_probs=stressor_probs_mean,
        )

    # ------------------------------------------------------------------
    # Trend estimation (separate call, requires temporal data)
    # ------------------------------------------------------------------

    def estimate_trends(
        self,
        health_history: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Estimate temporal health trends from historical scores.

        This is called separately from :meth:`forward` because it
        requires a time series of health scores (typically accumulated
        over days/weeks), whereas ``forward`` operates on a single
        observation.

        Parameters
        ----------
        health_history : Tensor[B, T] or Tensor[B, T, 1]
            Time series of health scores for one species at ``B``
            sites, with ``T`` timesteps.  ``T`` should be at least
            :attr:`TrendEstimator.min_sequence_length`.

        Returns
        -------
        trend_logits : Tensor[B, 3]
            Logits for (declining, stable, improving).
        trend_probs : Tensor[B, 3]
            Softmax probabilities.
        trend_rate : Tensor[B]
            Estimated health-score change per month.
        """
        return self.trend_estimator(health_history)

    # ------------------------------------------------------------------
    # High-level inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        sentinel_embedding: torch.Tensor,
        site_covariates: Optional[torch.Tensor] = None,
        health_history: Optional[torch.Tensor] = None,
        use_mc_dropout: bool = True,
    ) -> List[List[SpeciesHealthPrediction]]:
        """High-level inference producing structured predictions.

        Convenience method that runs the forward pass with MC-dropout
        and packages results into :class:`SpeciesHealthPrediction`
        dataclass instances.

        Parameters
        ----------
        sentinel_embedding : Tensor[B, 256]
            Multimodal SENTINEL embedding.
        site_covariates : Tensor[B, 5], optional
            Site-level geographic features.
        health_history : Tensor[B, T, S] or None, optional
            Historical health scores for trend estimation.  Shape is
            ``[B, T, S]`` where ``S`` is the number of species and
            ``T`` is the number of timesteps.  If ``None``, trend
            defaults to ``"stable"`` with rate 0.
        use_mc_dropout : bool
            Whether to use MC-dropout for uncertainty.

        Returns
        -------
        list of list of SpeciesHealthPrediction
            Outer list is over batch elements, inner list is over
            species.
        """
        self.eval()
        output = self.forward(
            sentinel_embedding, site_covariates,
            use_mc_dropout=use_mc_dropout,
        )

        B = sentinel_embedding.size(0)
        S = self.num_species

        # Estimate trends if history is provided.
        trend_labels: list[list[str]] = []
        trend_rates: list[list[float]] = []
        if health_history is not None and health_history.size(1) >= self.trend_estimator.min_sequence_length:
            for s in range(S):
                # Extract per-species history: [B, T]
                sp_history = health_history[:, :, s]
                _, trend_probs, rate = self.estimate_trends(sp_history)
                trend_idx = trend_probs.argmax(dim=-1)  # [B]
                for b in range(B):
                    if len(trend_labels) <= b:
                        trend_labels.append([])
                        trend_rates.append([])
                    trend_labels[b].append(TREND_LABELS[trend_idx[b].item()])
                    trend_rates[b].append(rate[b].item())
        else:
            trend_labels = [["stable"] * S for _ in range(B)]
            trend_rates = [[0.0] * S for _ in range(B)]

        # Package results.
        results: List[List[SpeciesHealthPrediction]] = []
        for b in range(B):
            site_preds: List[SpeciesHealthPrediction] = []
            for s in range(S):
                species = KEYSTONE_SPECIES[s]

                # Primary stressor.
                stressor_idx = output.stressor_probs[b, s].argmax().item()
                stressor_name = STRESSOR_CATEGORIES[stressor_idx]
                stressor_conf = output.stressor_probs[b, s, stressor_idx].item()

                pred = SpeciesHealthPrediction(
                    species_name=species.common_name,
                    health_score=output.health_scores[b, s].item(),
                    confidence_low=output.confidence_low[b, s].item(),
                    confidence_high=output.confidence_high[b, s].item(),
                    occupancy_prob=output.occupancy_probs[b, s].item(),
                    trend=trend_labels[b][s],
                    trend_rate=trend_rates[b][s],
                    primary_stressor=stressor_name,
                    stressor_confidence=stressor_conf,
                )
                site_preds.append(pred)
            results.append(site_preds)

        return results

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    @staticmethod
    def compute_loss(
        output: SpeciesHealthOutput,
        targets: Dict[str, torch.Tensor],
        loss_weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute multi-task training loss.

        Parameters
        ----------
        output : SpeciesHealthOutput
            Model predictions.
        targets : dict
            Expected keys (all optional, missing keys are skipped):

            - ``"health_scores"`` : Tensor[B, S] -- target health (0--100).
            - ``"occupancy"`` : Tensor[B, S] -- binary occupancy labels.
            - ``"abundance"`` : Tensor[B, S] -- log-abundance targets.
            - ``"stressor_labels"`` : Tensor[B, S] -- long, stressor
              class indices.

        loss_weights : dict, optional
            Per-task weighting factors.  Defaults to balanced weights.

        Returns
        -------
        total_loss : scalar Tensor
        per_task : Dict[str, Tensor]
            Individual task losses for logging.
        """
        if loss_weights is None:
            loss_weights = {
                "health": 1.0,
                "occupancy": 1.0,
                "abundance": 0.5,
                "stressor": 0.3,
            }

        losses: Dict[str, torch.Tensor] = {}
        device = output.health_scores.device

        if "health_scores" in targets:
            losses["health"] = F.mse_loss(
                output.health_scores, targets["health_scores"].float(),
            )

        if "occupancy" in targets:
            losses["occupancy"] = F.binary_cross_entropy(
                output.occupancy_probs,
                targets["occupancy"].float(),
            )

        if "abundance" in targets:
            losses["abundance"] = F.mse_loss(
                output.abundance, targets["abundance"].float(),
            )

        if "stressor_labels" in targets:
            # Reshape for cross-entropy: [B*S, C] vs [B*S].
            B, S, C = output.stressor_logits.shape
            logits_flat = output.stressor_logits.reshape(B * S, C)
            labels_flat = targets["stressor_labels"].reshape(B * S)
            losses["stressor"] = F.cross_entropy(logits_flat, labels_flat)

        total = torch.tensor(0.0, device=device)
        for key, loss_val in losses.items():
            w = loss_weights.get(key, 1.0)
            total = total + w * loss_val

        return total, losses
