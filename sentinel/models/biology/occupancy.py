"""Species Occupancy + eDNA Community Forecasting (Phase 3.5).

Predicts changes in species occupancy over 30/90/365-day horizons for
200+ freshwater species, and generates 16S community composition from
environmental embeddings.

Architecture overview
---------------------

**OccupancyShiftModel**::

    SENTINEL embedding (B, 256)
        --> HierarchicalBayesianEncoder (256 -> 512 -> 256 -> 128)
            --> SpeciesTaxonomicEmbedding (200+ species, 64-d learnable)
            --> TaxonomicPriorModule (family/order/class priors)
                --> per-species delta-occupancy heads for 30/90/365 day
                    horizons
            --> MC-dropout uncertainty (20 stochastic passes)

The hierarchical Bayesian-style approach is implemented as a neural
approximation: learnable per-species priors are regularised toward
taxonomic-group means, mimicking partial pooling in a hierarchical
model.

**eDNACommunityPredictor**::

    AquaSSM embedding (B, 256) + HydroViT embedding (B, 256)
        --> FusionEncoder (512 -> 256)
        --> CommunityDecoder (VAE-style generative model)
            --> 16S OTU probability table (B, 5000)
            --> log-abundance profile (B, 5000)

The generative model uses a VAE architecture with reparameterisation
trick to capture the multimodal nature of microbial community
distributions.

Uncertainty estimation uses MC-dropout (20 stochastic passes at
inference) following Gal & Ghahramani (2016).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Constants
# ============================================================================

SHARED_EMBEDDING_DIM: int = 256
SPECIES_EMBEDDING_DIM: int = 64
NUM_SPECIES: int = 200
NUM_OTUS: int = 5000
MC_DROPOUT_SAMPLES: int = 20
DROPOUT_P: float = 0.2

FORECAST_HORIZONS_DAYS: Tuple[int, ...] = (30, 90, 365)
NUM_HORIZONS: int = len(FORECAST_HORIZONS_DAYS)

# Taxonomic hierarchy levels
NUM_FAMILIES: int = 40
NUM_ORDERS: int = 15
NUM_CLASSES: int = 6

# VAE latent dimension for eDNA community generation
VAE_LATENT_DIM: int = 128


# ============================================================================
# Taxonomic groupings for hierarchical prior
# ============================================================================

class TaxonomicLevel(IntEnum):
    """Levels in the taxonomic hierarchy used for partial pooling."""
    SPECIES = 0
    FAMILY = 1
    ORDER = 2
    CLASS = 3


@dataclass(frozen=True)
class SpeciesMetadata:
    """Metadata for a target freshwater species."""

    index: int
    common_name: str
    taxon: str
    family_idx: int
    order_idx: int
    class_idx: int
    conservation_status: str
    habitat_preference: str


# Representative subset of 200 species across major freshwater taxa.
# In production, the full catalog is loaded from a database; here we
# define the mapping structure.
_DEFAULT_FAMILY_MAP: List[int] = [i % NUM_FAMILIES for i in range(NUM_SPECIES)]
_DEFAULT_ORDER_MAP: List[int] = [i % NUM_ORDERS for i in range(NUM_SPECIES)]
_DEFAULT_CLASS_MAP: List[int] = [i % NUM_CLASSES for i in range(NUM_SPECIES)]


# ============================================================================
# Occupancy shift alert levels
# ============================================================================

class OccupancyAlertLevel(IntEnum):
    """Alert levels for occupancy decline magnitude."""
    STABLE = 0          # |delta| < 0.05
    MINOR_DECLINE = 1   # 0.05 <= |delta| < 0.15
    MODERATE_DECLINE = 2  # 0.15 <= |delta| < 0.30
    SEVERE_DECLINE = 3    # |delta| >= 0.30


ALERT_THRESHOLDS: Dict[str, float] = {
    "minor": 0.05,
    "moderate": 0.15,
    "severe": 0.30,
}


# ============================================================================
# Output dataclasses
# ============================================================================

@dataclass
class OccupancyShiftOutput:
    """Output of the OccupancyShiftModel.

    All occupancy deltas are in [-1, 1] representing the predicted
    change in occupancy probability over the given horizon.

    Attributes
    ----------
    delta_occupancy : Tensor[B, S, H]
        Predicted occupancy change for S species over H horizons.
    baseline_occupancy : Tensor[B, S]
        Current estimated occupancy probability for each species.
    alert_levels : Tensor[B, S, H]
        Integer alert level per species per horizon.
    species_embeddings : Tensor[S, D]
        Learned species embedding table (for downstream analysis).
    mc_delta_mean : Tensor[B, S, H] or None
        MC-dropout posterior mean of delta occupancy.
    mc_delta_std : Tensor[B, S, H] or None
        MC-dropout posterior std of delta occupancy.
    taxonomic_prior_loss : Tensor or None
        Regularisation loss encouraging partial pooling toward
        taxonomic-group means.
    """

    delta_occupancy: torch.Tensor
    baseline_occupancy: torch.Tensor
    alert_levels: torch.Tensor
    species_embeddings: torch.Tensor

    mc_delta_mean: Optional[torch.Tensor] = None
    mc_delta_std: Optional[torch.Tensor] = None
    taxonomic_prior_loss: Optional[torch.Tensor] = None


@dataclass
class eDNACommunityOutput:
    """Output of the eDNACommunityPredictor.

    Attributes
    ----------
    otu_log_abundance : Tensor[B, num_otus]
        Predicted log-abundance for each OTU.
    otu_presence_prob : Tensor[B, num_otus]
        Predicted presence probability for each OTU.
    latent_mean : Tensor[B, latent_dim]
        VAE posterior mean.
    latent_logvar : Tensor[B, latent_dim]
        VAE posterior log-variance.
    kl_divergence : Tensor
        KL divergence of posterior from prior.
    mc_log_abundance_mean : Tensor[B, num_otus] or None
        MC-dropout posterior mean of log-abundance.
    mc_log_abundance_std : Tensor[B, num_otus] or None
        MC-dropout posterior std of log-abundance.
    """

    otu_log_abundance: torch.Tensor
    otu_presence_prob: torch.Tensor
    latent_mean: torch.Tensor
    latent_logvar: torch.Tensor
    kl_divergence: torch.Tensor

    mc_log_abundance_mean: Optional[torch.Tensor] = None
    mc_log_abundance_std: Optional[torch.Tensor] = None


# ============================================================================
# Sub-module 1: Taxonomic Prior Module
# ============================================================================

class TaxonomicPriorModule(nn.Module):
    """Learnable hierarchical priors for partial pooling.

    Implements a neural analogue of hierarchical Bayesian partial
    pooling.  Each species embedding is regularised toward its
    family-level mean, which is in turn regularised toward its
    order-level mean.  This structure enables information sharing
    across taxonomically related species, which is critical for
    rare/data-sparse species where direct estimates are unreliable.

    Parameters
    ----------
    num_species : int
        Number of target species (default 200).
    num_families : int
        Number of taxonomic families (default 40).
    num_orders : int
        Number of taxonomic orders (default 15).
    num_classes : int
        Number of taxonomic classes (default 6).
    embedding_dim : int
        Dimensionality of species embeddings (default 64).
    prior_strength : float
        Regularisation strength toward group means (default 0.1).
    """

    def __init__(
        self,
        num_species: int = NUM_SPECIES,
        num_families: int = NUM_FAMILIES,
        num_orders: int = NUM_ORDERS,
        num_classes: int = NUM_CLASSES,
        embedding_dim: int = SPECIES_EMBEDDING_DIM,
        prior_strength: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_species = num_species
        self.num_families = num_families
        self.num_orders = num_orders
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.prior_strength = prior_strength

        # Learnable group-level means
        self.family_mean = nn.Embedding(num_families, embedding_dim)
        self.order_mean = nn.Embedding(num_orders, embedding_dim)
        self.class_mean = nn.Embedding(num_classes, embedding_dim)

        # Taxonomic mapping buffers (species -> family/order/class)
        self.register_buffer(
            "family_map",
            torch.tensor(_DEFAULT_FAMILY_MAP, dtype=torch.long),
        )
        self.register_buffer(
            "order_map",
            torch.tensor(_DEFAULT_ORDER_MAP, dtype=torch.long),
        )
        self.register_buffer(
            "class_map",
            torch.tensor(_DEFAULT_CLASS_MAP, dtype=torch.long),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.family_mean.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.order_mean.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.class_mean.weight, mean=0.0, std=0.02)

    def load_taxonomy(
        self,
        family_map: torch.Tensor,
        order_map: torch.Tensor,
        class_map: torch.Tensor,
    ) -> None:
        """Load species-to-group mappings from external taxonomy database.

        Parameters
        ----------
        family_map : Tensor[num_species]
            Maps each species index to its family index.
        order_map : Tensor[num_species]
            Maps each species index to its order index.
        class_map : Tensor[num_species]
            Maps each species index to its class index.
        """
        assert family_map.shape == (self.num_species,)
        assert order_map.shape == (self.num_species,)
        assert class_map.shape == (self.num_species,)
        self.family_map.copy_(family_map)
        self.order_map.copy_(order_map)
        self.class_map.copy_(class_map)

    def compute_prior_loss(
        self,
        species_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Compute hierarchical prior regularisation loss.

        Encourages species embeddings to stay close to their
        taxonomic-group means, implementing partial pooling:

        L = prior_strength * (
            sum_s ||emb_s - family_mean[f(s)]||^2
            + 0.5 * sum_f ||family_mean[f] - order_mean[o(f)]||^2
            + 0.25 * sum_o ||order_mean[o] - class_mean[c(o)]||^2
        )

        Parameters
        ----------
        species_embeddings : Tensor[S, D]
            Current species embedding table.

        Returns
        -------
        Tensor (scalar)
            Hierarchical prior loss.
        """
        S = species_embeddings.shape[0]
        device = species_embeddings.device

        # Species -> family level
        family_targets = self.family_mean(self.family_map[:S])  # [S, D]
        sp_to_fam_loss = F.mse_loss(species_embeddings, family_targets)

        # Family -> order level
        family_embs = self.family_mean.weight  # [F, D]
        order_targets_for_fam = self.order_mean(
            self.order_map[:self.num_families].clamp(max=self.num_orders - 1)
        )  # [F, D]
        fam_to_ord_loss = F.mse_loss(family_embs, order_targets_for_fam)

        # Order -> class level
        order_embs = self.order_mean.weight  # [O, D]
        class_targets_for_ord = self.class_mean(
            self.class_map[:self.num_orders].clamp(max=self.num_classes - 1)
        )  # [O, D]
        ord_to_cls_loss = F.mse_loss(order_embs, class_targets_for_ord)

        total = self.prior_strength * (
            sp_to_fam_loss + 0.5 * fam_to_ord_loss + 0.25 * ord_to_cls_loss
        )
        return total

    def get_species_prior(
        self,
        species_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get the hierarchical prior embedding for species.

        Combines family, order, and class-level information as a
        weighted sum for species with limited direct data.

        Parameters
        ----------
        species_idx : Tensor, optional
            Species indices.  If None, returns priors for all species.

        Returns
        -------
        Tensor[..., D]
            Prior embedding combining all taxonomic levels.
        """
        if species_idx is None:
            species_idx = torch.arange(
                self.num_species,
                device=self.family_mean.weight.device,
            )

        fam_idx = self.family_map[species_idx]
        ord_idx = self.order_map[species_idx]
        cls_idx = self.class_map[species_idx]

        fam_emb = self.family_mean(fam_idx)     # [..., D]
        ord_emb = self.order_mean(ord_idx)       # [..., D]
        cls_emb = self.class_mean(cls_idx)       # [..., D]

        # Weighted combination (more specific levels weighted higher)
        return 0.6 * fam_emb + 0.3 * ord_emb + 0.1 * cls_emb


# ============================================================================
# Sub-module 2: Species Embedding with Taxonomic Prior
# ============================================================================

class SpeciesTaxonomicEmbedding(nn.Module):
    """Learnable species embedding with hierarchical taxonomic prior.

    Each of the 200+ target species gets a 64-d embedding that is
    regularised toward its taxonomic group mean through the
    TaxonomicPriorModule.  For data-rich species the embedding can
    deviate significantly from the prior; for rare species it stays
    close, enabling effective transfer learning.

    Parameters
    ----------
    num_species : int
        Number of target species (default 200).
    embedding_dim : int
        Dimensionality of species embeddings (default 64).
    """

    def __init__(
        self,
        num_species: int = NUM_SPECIES,
        embedding_dim: int = SPECIES_EMBEDDING_DIM,
    ) -> None:
        super().__init__()
        self.num_species = num_species
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(num_species, embedding_dim)
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
            Long tensor of species indices.  If None, returns
            the full embedding table [S, D].

        Returns
        -------
        Tensor[..., D]
        """
        if species_idx is None:
            species_idx = torch.arange(
                self.num_species,
                device=self.embedding.weight.device,
            )
        emb = self.embedding(species_idx)
        return self.layer_norm(emb)


# ============================================================================
# Sub-module 3: Hierarchical Bayesian Encoder
# ============================================================================

class HierarchicalBayesianEncoder(nn.Module):
    """Shared environmental encoder with MC-dropout for occupancy prediction.

    Processes the 256-d SENTINEL embedding through a deep MLP with
    dropout at every layer, enabling MC-dropout uncertainty estimation
    at inference time.

    Architecture::

        SENTINEL embedding (B, 256)
            --> Linear(256, 512) + GELU + Dropout
            --> Linear(512, 256) + GELU + Dropout
            --> LayerNorm
            --> Linear(256, 128) + GELU + Dropout
            --> output (B, 128)

    Parameters
    ----------
    embedding_dim : int
        Input SENTINEL embedding dimension (default 256).
    hidden_dims : tuple
        Hidden layer sizes (default (512, 256, 128)).
    dropout : float
        Dropout probability (default 0.2).
    """

    def __init__(
        self,
        embedding_dim: int = SHARED_EMBEDDING_DIM,
        hidden_dims: Tuple[int, ...] = (512, 256, 128),
        dropout: float = DROPOUT_P,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

        layers: List[nn.Module] = []
        in_features = embedding_dim
        for i, out_features in enumerate(hidden_dims):
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            if i == 1:  # LayerNorm after second hidden layer
                layers.append(nn.LayerNorm(out_features))
            in_features = out_features

        self.encoder = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """Encode the SENTINEL embedding.

        Parameters
        ----------
        embedding : Tensor[B, 256]
            SENTINEL environmental embedding.

        Returns
        -------
        Tensor[B, 128]
            Encoded environmental representation.
        """
        return self.encoder(embedding)


# ============================================================================
# Sub-module 4: Per-Species Occupancy Heads
# ============================================================================

class OccupancyHeadBank(nn.Module):
    """Bank of per-species occupancy shift prediction heads.

    Uses a shared backbone followed by species-modulated lightweight
    heads.  Each head predicts the occupancy delta for 3 horizons
    (30, 90, 365 days) and a baseline occupancy estimate.

    Parameters
    ----------
    env_dim : int
        Environmental encoder output dimension (default 128).
    species_dim : int
        Species embedding dimension (default 64).
    num_species : int
        Number of species (default 200).
    num_horizons : int
        Number of forecast horizons (default 3).
    dropout : float
        Dropout probability (default 0.2).
    """

    def __init__(
        self,
        env_dim: int = 128,
        species_dim: int = SPECIES_EMBEDDING_DIM,
        num_species: int = NUM_SPECIES,
        num_horizons: int = NUM_HORIZONS,
        dropout: float = DROPOUT_P,
    ) -> None:
        super().__init__()
        self.num_species = num_species
        self.num_horizons = num_horizons

        combined_dim = env_dim + species_dim  # 192

        # Shared feature extractor (across species)
        self.shared_backbone = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Baseline occupancy head (one per species group to save params)
        # We use a FiLM-style conditioning: species embedding modulates
        # features via learned scale + shift
        self.baseline_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        # Delta occupancy heads: predict change at each horizon
        self.delta_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
                nn.Tanh(),  # delta in [-1, 1]
            )
            for _ in range(num_horizons)
        ])

        # FiLM conditioning: scale and shift from species embedding
        self.film_gamma = nn.Linear(species_dim, 128)
        self.film_beta = nn.Linear(species_dim, 128)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Initialize FiLM to identity transform
        nn.init.ones_(self.film_gamma.weight)
        nn.init.zeros_(self.film_gamma.bias)
        nn.init.zeros_(self.film_beta.weight)
        nn.init.zeros_(self.film_beta.bias)

    def forward(
        self,
        env_repr: torch.Tensor,
        species_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict occupancy baseline and deltas for all species.

        Parameters
        ----------
        env_repr : Tensor[B, env_dim]
            Encoded environment.
        species_emb : Tensor[S, species_dim]
            Species embedding table.

        Returns
        -------
        baseline : Tensor[B, S]
            Baseline occupancy probabilities in [0, 1].
        deltas : Tensor[B, S, H]
            Predicted occupancy deltas in [-1, 1] for each horizon.
        """
        B = env_repr.size(0)
        S = species_emb.size(0)

        baseline_list: List[torch.Tensor] = []
        delta_list: List[torch.Tensor] = []

        for s in range(S):
            sp_emb = species_emb[s].unsqueeze(0).expand(B, -1)  # [B, D_sp]
            combined = torch.cat([env_repr, sp_emb], dim=-1)    # [B, 192]

            shared_feat = self.shared_backbone(combined)  # [B, 128]

            # FiLM conditioning: modulate shared features by species identity
            gamma = self.film_gamma(sp_emb)  # [B, 128]
            beta = self.film_beta(sp_emb)     # [B, 128]
            modulated = gamma * shared_feat + beta  # [B, 128]

            # Baseline occupancy
            base_logit = self.baseline_head(modulated).squeeze(-1)  # [B]
            baseline_list.append(torch.sigmoid(base_logit))

            # Delta for each horizon
            horizon_deltas: List[torch.Tensor] = []
            for h in range(self.num_horizons):
                delta = self.delta_heads[h](modulated).squeeze(-1)  # [B]
                horizon_deltas.append(delta)
            delta_list.append(torch.stack(horizon_deltas, dim=-1))  # [B, H]

        baseline = torch.stack(baseline_list, dim=1)  # [B, S]
        deltas = torch.stack(delta_list, dim=1)        # [B, S, H]

        return baseline, deltas


# ============================================================================
# Sub-module 5: Temporal Context Encoding
# ============================================================================

def encode_seasonal_context(
    day_of_year: torch.Tensor,
    *,
    max_days: float = 365.25,
) -> torch.Tensor:
    """Build a seasonal context vector from day-of-year.

    Parameters
    ----------
    day_of_year : Tensor[B]
        Day of the year (1-366).

    Returns
    -------
    Tensor[B, 8]
        [sin(annual), cos(annual), sin(semi-annual), cos(semi-annual),
         spring, summer, fall, winter]
    """
    angle = 2.0 * math.pi * day_of_year / max_days
    sin_annual = torch.sin(angle)
    cos_annual = torch.cos(angle)
    sin_semi = torch.sin(2.0 * angle)
    cos_semi = torch.cos(2.0 * angle)

    season = torch.zeros(day_of_year.shape[0], 4, device=day_of_year.device)
    doy = day_of_year.long()
    spring = (doy >= 60) & (doy < 152)
    summer = (doy >= 152) & (doy < 244)
    fall = (doy >= 244) & (doy < 335)
    winter = ~(spring | summer | fall)
    season[:, 0] = spring.float()
    season[:, 1] = summer.float()
    season[:, 2] = fall.float()
    season[:, 3] = winter.float()

    return torch.cat([
        sin_annual.unsqueeze(-1),
        cos_annual.unsqueeze(-1),
        sin_semi.unsqueeze(-1),
        cos_semi.unsqueeze(-1),
        season,
    ], dim=-1)


# ============================================================================
# Main Model 1: OccupancyShiftModel
# ============================================================================

class OccupancyShiftModel(nn.Module):
    """Multi-species occupancy shift forecaster.

    Predicts changes in species occupancy probability over 30, 90, and
    365-day horizons for 200+ freshwater species.  Uses a hierarchical
    Bayesian-style approach with taxonomic partial pooling and
    MC-dropout uncertainty.

    Parameters
    ----------
    embedding_dim : int
        SENTINEL embedding dimension (default 256).
    num_species : int
        Number of target species (default 200).
    species_embedding_dim : int
        Species embedding dimension (default 64).
    encoder_hidden_dims : tuple
        Hidden layer sizes for the environmental encoder.
    dropout : float
        Dropout probability for MC-dropout (default 0.2).
    mc_samples : int
        Number of MC-dropout forward passes (default 20).
    prior_strength : float
        Hierarchical prior regularisation strength (default 0.1).

    Example
    -------
    >>> model = OccupancyShiftModel()
    >>> embedding = torch.randn(4, 256)
    >>> output = model(embedding)
    >>> output.delta_occupancy.shape
    torch.Size([4, 200, 3])
    """

    def __init__(
        self,
        embedding_dim: int = SHARED_EMBEDDING_DIM,
        num_species: int = NUM_SPECIES,
        species_embedding_dim: int = SPECIES_EMBEDDING_DIM,
        encoder_hidden_dims: Tuple[int, ...] = (512, 256, 128),
        dropout: float = DROPOUT_P,
        mc_samples: int = MC_DROPOUT_SAMPLES,
        prior_strength: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_species = num_species
        self.mc_samples = mc_samples
        self.dropout_p = dropout

        # Sub-modules
        self.env_encoder = HierarchicalBayesianEncoder(
            embedding_dim=embedding_dim,
            hidden_dims=encoder_hidden_dims,
            dropout=dropout,
        )

        self.species_embedding = SpeciesTaxonomicEmbedding(
            num_species=num_species,
            embedding_dim=species_embedding_dim,
        )

        self.taxonomic_prior = TaxonomicPriorModule(
            num_species=num_species,
            embedding_dim=species_embedding_dim,
            prior_strength=prior_strength,
        )

        self.head_bank = OccupancyHeadBank(
            env_dim=encoder_hidden_dims[-1],
            species_dim=species_embedding_dim,
            num_species=num_species,
            num_horizons=NUM_HORIZONS,
            dropout=dropout,
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single deterministic forward pass.

        Returns
        -------
        baseline : Tensor[B, S]
        deltas : Tensor[B, S, H]
        species_emb : Tensor[S, D]
        prior_loss : Tensor (scalar)
        """
        env_repr = self.env_encoder(sentinel_embedding)
        species_emb = self.species_embedding()
        prior_loss = self.taxonomic_prior.compute_prior_loss(species_emb)
        baseline, deltas = self.head_bank(env_repr, species_emb)
        return baseline, deltas, species_emb, prior_loss

    def _compute_alert_levels(
        self,
        deltas: torch.Tensor,
    ) -> torch.Tensor:
        """Assign alert levels based on occupancy decline magnitude.

        Parameters
        ----------
        deltas : Tensor[B, S, H]
            Predicted occupancy deltas.

        Returns
        -------
        Tensor[B, S, H] of long
            Alert level for each species at each horizon.
        """
        abs_delta = deltas.abs()
        alerts = torch.zeros_like(deltas, dtype=torch.long)
        alerts[abs_delta >= ALERT_THRESHOLDS["minor"]] = OccupancyAlertLevel.MINOR_DECLINE
        alerts[abs_delta >= ALERT_THRESHOLDS["moderate"]] = OccupancyAlertLevel.MODERATE_DECLINE
        alerts[abs_delta >= ALERT_THRESHOLDS["severe"]] = OccupancyAlertLevel.SEVERE_DECLINE
        # Only mark negative deltas (declines) as alerts
        alerts[deltas > 0] = OccupancyAlertLevel.STABLE
        return alerts

    def forward(
        self,
        sentinel_embedding: torch.Tensor,
        use_mc_dropout: bool = False,
    ) -> OccupancyShiftOutput:
        """Forward pass with optional MC-dropout uncertainty.

        Parameters
        ----------
        sentinel_embedding : Tensor[B, 256]
            SENTINEL environmental embedding.
        use_mc_dropout : bool
            If True and in eval mode, run mc_samples stochastic
            forward passes for uncertainty estimation.

        Returns
        -------
        OccupancyShiftOutput
        """
        if not use_mc_dropout or self.training:
            baseline, deltas, species_emb, prior_loss = self._single_forward(
                sentinel_embedding,
            )
            return OccupancyShiftOutput(
                delta_occupancy=deltas,
                baseline_occupancy=baseline,
                alert_levels=self._compute_alert_levels(deltas),
                species_embeddings=species_emb,
                taxonomic_prior_loss=prior_loss,
            )

        # --- MC-dropout inference ---
        self._enable_dropout()

        mc_deltas: List[torch.Tensor] = []
        mc_baselines: List[torch.Tensor] = []

        with torch.no_grad():
            for _ in range(self.mc_samples):
                baseline, deltas, species_emb, prior_loss = self._single_forward(
                    sentinel_embedding,
                )
                mc_deltas.append(deltas)
                mc_baselines.append(baseline)

        self._disable_dropout()

        mc_deltas_stack = torch.stack(mc_deltas, dim=0)      # [N, B, S, H]
        mc_baselines_stack = torch.stack(mc_baselines, dim=0)  # [N, B, S]

        delta_mean = mc_deltas_stack.mean(dim=0)   # [B, S, H]
        delta_std = mc_deltas_stack.std(dim=0)     # [B, S, H]
        baseline_mean = mc_baselines_stack.mean(dim=0)  # [B, S]

        return OccupancyShiftOutput(
            delta_occupancy=delta_mean,
            baseline_occupancy=baseline_mean,
            alert_levels=self._compute_alert_levels(delta_mean),
            species_embeddings=species_emb,
            mc_delta_mean=delta_mean,
            mc_delta_std=delta_std,
            taxonomic_prior_loss=prior_loss,
        )

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    @staticmethod
    def compute_loss(
        output: OccupancyShiftOutput,
        targets: Dict[str, torch.Tensor],
        loss_weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute multi-task training loss.

        Parameters
        ----------
        output : OccupancyShiftOutput
            Model predictions.
        targets : dict
            Expected keys (all optional, missing keys are skipped):

            - ``"delta_occupancy"`` : Tensor[B, S, H] -- target deltas.
            - ``"baseline_occupancy"`` : Tensor[B, S] -- target baselines.

        loss_weights : dict, optional
            Per-task weighting factors.

        Returns
        -------
        total_loss : scalar Tensor
        per_task : Dict[str, Tensor]
        """
        if loss_weights is None:
            loss_weights = {
                "delta": 1.0,
                "baseline": 0.5,
                "prior": 0.1,
            }

        losses: Dict[str, torch.Tensor] = {}
        device = output.delta_occupancy.device

        if "delta_occupancy" in targets:
            losses["delta"] = F.mse_loss(
                output.delta_occupancy,
                targets["delta_occupancy"].float(),
            )

        if "baseline_occupancy" in targets:
            losses["baseline"] = F.binary_cross_entropy(
                output.baseline_occupancy,
                targets["baseline_occupancy"].float(),
            )

        if output.taxonomic_prior_loss is not None:
            losses["prior"] = output.taxonomic_prior_loss

        total = torch.tensor(0.0, device=device)
        for key, loss_val in losses.items():
            w = loss_weights.get(key, 1.0)
            total = total + w * loss_val

        return total, losses


# ============================================================================
# Sub-module 6: Embedding Fusion for eDNA
# ============================================================================

class DualEmbeddingFusion(nn.Module):
    """Fuse AquaSSM and HydroViT embeddings for community prediction.

    Combines two 256-d embeddings from complementary models:
    - AquaSSM: temporal water chemistry patterns
    - HydroViT: spatial hydrological features from satellite

    Architecture::

        AquaSSM (B, 256) + HydroViT (B, 256)
            --> concat (B, 512)
            --> Linear(512, 256) + GELU + Dropout
            --> Linear(256, 256) + LayerNorm
            --> output (B, 256)

    Parameters
    ----------
    embedding_dim : int
        Dimension of each input embedding (default 256).
    dropout : float
        Dropout probability (default 0.2).
    """

    def __init__(
        self,
        embedding_dim: int = SHARED_EMBEDDING_DIM,
        dropout: float = DROPOUT_P,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

        # Cross-attention for refined fusion
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(embedding_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        aqua_embedding: torch.Tensor,
        hydro_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse AquaSSM and HydroViT embeddings.

        Parameters
        ----------
        aqua_embedding : Tensor[B, 256]
            AquaSSM temporal embedding.
        hydro_embedding : Tensor[B, 256]
            HydroViT spatial embedding.

        Returns
        -------
        Tensor[B, 256]
            Fused environmental embedding.
        """
        # Concatenation-based fusion
        concat = torch.cat([aqua_embedding, hydro_embedding], dim=-1)
        fused = self.fusion(concat)  # [B, 256]

        # Cross-attention refinement
        # Treat each embedding as a sequence element for attention
        seq = torch.stack([aqua_embedding, hydro_embedding], dim=1)  # [B, 2, 256]
        attn_out, _ = self.cross_attention(
            fused.unsqueeze(1), seq, seq,
        )  # [B, 1, 256]
        refined = self.cross_norm(fused + attn_out.squeeze(1))

        return refined


# ============================================================================
# Sub-module 7: VAE Community Decoder
# ============================================================================

class CommunityVAEDecoder(nn.Module):
    """VAE-style decoder for 16S community composition generation.

    Encodes the fused environmental embedding into a latent space,
    then decodes to a 16S OTU composition table.  The VAE structure
    captures the multimodal nature of microbial communities: similar
    environments can support different community configurations.

    Architecture::

        fused_embedding (B, 256)
            --> Encoder: Linear(256, 256) -> GELU -> Dropout
                       -> mu head: Linear(256, latent_dim)
                       -> logvar head: Linear(256, latent_dim)
            --> Reparameterisation: z = mu + std * eps
            --> Decoder: Linear(latent_dim, 512) -> GELU -> Dropout
                       -> Linear(512, 1024) -> GELU -> Dropout
                       -> presence head: Linear(1024, num_otus) + Sigmoid
                       -> abundance head: Linear(1024, num_otus)

    Parameters
    ----------
    input_dim : int
        Fused embedding dimension (default 256).
    latent_dim : int
        VAE latent dimension (default 128).
    num_otus : int
        Number of OTUs to predict (default 5000).
    dropout : float
        Dropout probability (default 0.2).
    """

    def __init__(
        self,
        input_dim: int = SHARED_EMBEDDING_DIM,
        latent_dim: int = VAE_LATENT_DIM,
        num_otus: int = NUM_OTUS,
        dropout: float = DROPOUT_P,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_otus = num_otus

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.mu_head = nn.Linear(256, latent_dim)
        self.logvar_head = nn.Linear(256, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 2048),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Output heads
        self.presence_head = nn.Sequential(
            nn.Linear(2048, num_otus),
        )
        self.abundance_head = nn.Sequential(
            nn.Linear(2048, num_otus),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _reparameterise(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """Reparameterisation trick: z = mu + std * epsilon.

        Parameters
        ----------
        mu : Tensor[B, latent_dim]
        logvar : Tensor[B, latent_dim]

        Returns
        -------
        Tensor[B, latent_dim]
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + std * eps
        return mu

    def forward(
        self,
        fused_embedding: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate 16S community composition from environment.

        Parameters
        ----------
        fused_embedding : Tensor[B, 256]
            Fused environmental embedding.

        Returns
        -------
        otu_log_abundance : Tensor[B, num_otus]
        otu_presence_prob : Tensor[B, num_otus]
        mu : Tensor[B, latent_dim]
        logvar : Tensor[B, latent_dim]
        """
        # Encode
        h = self.encoder(fused_embedding)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)

        # Sample latent
        z = self._reparameterise(mu, logvar)

        # Decode
        decoded = self.decoder(z)
        otu_log_abundance = self.abundance_head(decoded)
        otu_presence_logit = self.presence_head(decoded)
        otu_presence_prob = torch.sigmoid(otu_presence_logit)

        return otu_log_abundance, otu_presence_prob, mu, logvar


# ============================================================================
# Main Model 2: eDNACommunityPredictor
# ============================================================================

class eDNACommunityPredictor(nn.Module):
    """Generative model for 16S community composition from environment.

    Given AquaSSM and HydroViT embeddings (capturing temporal water
    chemistry and spatial hydrology respectively), predicts the expected
    16S rRNA community composition at a monitoring site.  This is a
    generative model that can produce diverse community samples for a
    given environment, reflecting the inherent stochasticity of
    microbial community assembly.

    Parameters
    ----------
    embedding_dim : int
        Input embedding dimension (default 256).
    latent_dim : int
        VAE latent space dimension (default 128).
    num_otus : int
        Number of 16S OTUs to predict (default 5000).
    dropout : float
        Dropout probability (default 0.2).
    mc_samples : int
        Number of MC-dropout forward passes (default 20).
    kl_weight : float
        Weight for KL divergence loss (beta-VAE parameter, default 0.01).

    Example
    -------
    >>> model = eDNACommunityPredictor()
    >>> aqua = torch.randn(4, 256)
    >>> hydro = torch.randn(4, 256)
    >>> output = model(aqua, hydro)
    >>> output.otu_log_abundance.shape
    torch.Size([4, 5000])
    """

    def __init__(
        self,
        embedding_dim: int = SHARED_EMBEDDING_DIM,
        latent_dim: int = VAE_LATENT_DIM,
        num_otus: int = NUM_OTUS,
        dropout: float = DROPOUT_P,
        mc_samples: int = MC_DROPOUT_SAMPLES,
        kl_weight: float = 0.01,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_otus = num_otus
        self.mc_samples = mc_samples
        self.kl_weight = kl_weight

        # Sub-modules
        self.fusion = DualEmbeddingFusion(
            embedding_dim=embedding_dim,
            dropout=dropout,
        )

        self.vae_decoder = CommunityVAEDecoder(
            input_dim=embedding_dim,
            latent_dim=latent_dim,
            num_otus=num_otus,
            dropout=dropout,
        )

    # ------------------------------------------------------------------
    # MC-dropout helpers
    # ------------------------------------------------------------------

    def _enable_dropout(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def _disable_dropout(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.eval()

    # ------------------------------------------------------------------
    # KL divergence
    # ------------------------------------------------------------------

    @staticmethod
    def _kl_divergence(
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KL(q(z|x) || p(z)) where p(z) = N(0, I).

        Parameters
        ----------
        mu : Tensor[B, D]
        logvar : Tensor[B, D]

        Returns
        -------
        Tensor (scalar)
        """
        return -0.5 * torch.mean(
            1.0 + logvar - mu.pow(2) - logvar.exp()
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _single_forward(
        self,
        aqua_embedding: torch.Tensor,
        hydro_embedding: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single forward pass.

        Returns
        -------
        otu_log_abundance, otu_presence_prob, mu, logvar, kl_div
        """
        fused = self.fusion(aqua_embedding, hydro_embedding)
        otu_log_abund, otu_pres_prob, mu, logvar = self.vae_decoder(fused)
        kl_div = self._kl_divergence(mu, logvar)
        return otu_log_abund, otu_pres_prob, mu, logvar, kl_div

    def forward(
        self,
        aqua_embedding: torch.Tensor,
        hydro_embedding: torch.Tensor,
        use_mc_dropout: bool = False,
    ) -> eDNACommunityOutput:
        """Forward pass with optional MC-dropout uncertainty.

        Parameters
        ----------
        aqua_embedding : Tensor[B, 256]
            AquaSSM temporal water chemistry embedding.
        hydro_embedding : Tensor[B, 256]
            HydroViT spatial hydrological embedding.
        use_mc_dropout : bool
            If True and in eval mode, run mc_samples stochastic
            forward passes.

        Returns
        -------
        eDNACommunityOutput
        """
        if not use_mc_dropout or self.training:
            log_abund, pres_prob, mu, logvar, kl_div = self._single_forward(
                aqua_embedding, hydro_embedding,
            )
            return eDNACommunityOutput(
                otu_log_abundance=log_abund,
                otu_presence_prob=pres_prob,
                latent_mean=mu,
                latent_logvar=logvar,
                kl_divergence=kl_div,
            )

        # --- MC-dropout inference ---
        self._enable_dropout()
        mc_log_abund: List[torch.Tensor] = []

        with torch.no_grad():
            for _ in range(self.mc_samples):
                log_abund, pres_prob, mu, logvar, kl_div = self._single_forward(
                    aqua_embedding, hydro_embedding,
                )
                mc_log_abund.append(log_abund)

        self._disable_dropout()

        mc_stack = torch.stack(mc_log_abund, dim=0)  # [N, B, OTUs]
        log_abund_mean = mc_stack.mean(dim=0)
        log_abund_std = mc_stack.std(dim=0)

        return eDNACommunityOutput(
            otu_log_abundance=log_abund_mean,
            otu_presence_prob=pres_prob,
            latent_mean=mu,
            latent_logvar=logvar,
            kl_divergence=kl_div,
            mc_log_abundance_mean=log_abund_mean,
            mc_log_abundance_std=log_abund_std,
        )

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        output: eDNACommunityOutput,
        targets: Dict[str, torch.Tensor],
        loss_weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute VAE reconstruction + KL loss.

        Parameters
        ----------
        output : eDNACommunityOutput
            Model predictions.
        targets : dict
            Expected keys (all optional, missing keys are skipped):

            - ``"otu_log_abundance"`` : Tensor[B, num_otus]
            - ``"otu_presence"`` : Tensor[B, num_otus] binary

        loss_weights : dict, optional
            Per-task weighting factors.

        Returns
        -------
        total_loss : scalar Tensor
        per_task : Dict[str, Tensor]
        """
        if loss_weights is None:
            loss_weights = {
                "reconstruction": 1.0,
                "presence": 0.5,
                "kl": self.kl_weight,
            }

        losses: Dict[str, torch.Tensor] = {}
        device = output.otu_log_abundance.device

        if "otu_log_abundance" in targets:
            losses["reconstruction"] = F.mse_loss(
                output.otu_log_abundance,
                targets["otu_log_abundance"].float(),
            )

        if "otu_presence" in targets:
            losses["presence"] = F.binary_cross_entropy(
                output.otu_presence_prob,
                targets["otu_presence"].float(),
            )

        losses["kl"] = output.kl_divergence

        total = torch.tensor(0.0, device=device)
        for key, loss_val in losses.items():
            w = loss_weights.get(key, 1.0)
            total = total + w * loss_val

        return total, losses
