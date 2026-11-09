"""Endangered Species Recovery Planner for the Digital Aquatic Ecosystem Twin.

For each of ~150 federally listed freshwater species, uses the twin in
counterfactual mode to identify optimal recovery strategies.  The planner
evaluates habitat suitability at candidate sites, ranks intervention
options via learned attention, predicts recovery timelines, and produces
cost-constrained recovery plans.

The module layers:

1. **SpeciesHabitatRequirements** -- dataclass capturing water-quality
   tolerances, substrate preferences, and flow requirements for each
   species, drawn from USFWS recovery plans and NatureServe assessments.
2. **HabitatAssessment** -- per-site suitability evaluation with
   limiting-factor analysis and predicted suitability under intervention.
3. **RecoveryPlan** -- multi-site recovery plan with ranked interventions,
   cost estimates, predicted habitat expansion, and confidence intervals.
4. **RecoveryPlanner** -- ``nn.Module`` with habitat scoring, attention-
   based intervention ranking, and timeline prediction sub-networks.

References
----------
* USFWS. Recovery plans for federally listed species, ecos.fws.gov.
* NatureServe (2023). NatureServe Explorer species database.
* Strayer, D.L. et al. (2004). Effects of land cover on stream
  ecosystems. Freshwater Biology 49:1-20.
* Haag, W.R. (2012). North American Freshwater Mussels. Cambridge Univ.
* Jelks, H.L. et al. (2008). Conservation status of imperiled North
  American freshwater and diadromous fishes. Fisheries 33:372-407.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SENTINEL_EMBEDDING_DIM: int = 256
DEFAULT_NUM_SPECIES: int = 150
DEFAULT_NUM_ENV_PARAMS: int = 10
DEFAULT_HIDDEN_DIM: int = 128
MC_DROPOUT_SAMPLES: int = 20


# ---------------------------------------------------------------------------
# Intervention catalog
# ---------------------------------------------------------------------------

INTERVENTION_CATALOG: Dict[str, Dict[str, Any]] = {
    "riparian_buffer": {
        "description": "Native vegetation buffer along stream corridor",
        "unit": "acre",
        "cost_per_unit": 2_500.0,
        "typical_area_range": (5.0, 500.0),
        "effect_lag_years": 2.0,
        "primary_benefits": ["temperature", "sediment", "nutrients"],
    },
    "wetland_construction": {
        "description": "Constructed treatment wetland for nutrient removal",
        "unit": "acre",
        "cost_per_unit": 35_000.0,
        "typical_area_range": (1.0, 50.0),
        "effect_lag_years": 3.0,
        "primary_benefits": ["nutrients", "sediment", "habitat"],
    },
    "dam_removal": {
        "description": "Full or partial dam removal for fish passage",
        "unit": "project",
        "cost_range": (500_000.0, 5_000_000.0),
        "cost_per_unit": 2_500_000.0,
        "typical_area_range": (1.0, 1.0),
        "effect_lag_years": 1.0,
        "primary_benefits": ["connectivity", "flow", "sediment_transport"],
    },
    "agricultural_bmp": {
        "description": "Best management practices on agricultural land",
        "unit": "acre",
        "cost_per_unit": 50.0,
        "typical_area_range": (50.0, 5_000.0),
        "effect_lag_years": 1.0,
        "primary_benefits": ["nutrients", "sediment", "pesticides"],
    },
    "point_source_upgrade": {
        "description": "Wastewater treatment plant upgrade (tertiary)",
        "unit": "project",
        "cost_range": (1_000_000.0, 10_000_000.0),
        "cost_per_unit": 5_000_000.0,
        "typical_area_range": (1.0, 1.0),
        "effect_lag_years": 0.5,
        "primary_benefits": ["nutrients", "BOD", "pathogens"],
    },
    "flow_augmentation": {
        "description": "Environmental flow releases or water purchases",
        "unit": "year",
        "cost_per_unit": 100_000.0,
        "typical_area_range": (1.0, 1.0),
        "effect_lag_years": 0.25,
        "primary_benefits": ["flow", "temperature", "dissolved_oxygen"],
    },
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SpeciesHabitatRequirements:
    """Water-quality and habitat requirements for a federally listed species."""

    species_name: str
    scientific_name: str
    federal_status: str  # "endangered", "threatened", "candidate"
    temp_range: Tuple[float, float]  # degrees Celsius
    do_minimum: float  # mg/L dissolved oxygen
    ph_range: Tuple[float, float]
    substrate_preferences: List[str]
    flow_requirements: str
    additional_constraints: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        valid_statuses = {"endangered", "threatened", "candidate"}
        if self.federal_status not in valid_statuses:
            raise ValueError(
                f"federal_status must be one of {valid_statuses}, "
                f"got {self.federal_status!r}"
            )


@dataclass
class HabitatAssessment:
    """Per-site habitat suitability assessment for a species."""

    species: SpeciesHabitatRequirements
    site_id: str
    current_suitability: float  # 0-1 continuous scale
    limiting_factors: List[Tuple[str, float]]  # (factor_name, deficit)
    predicted_suitability_under_intervention: Dict[str, float]
    recovery_timeline_years: float

    def __post_init__(self) -> None:
        self.current_suitability = max(0.0, min(1.0, self.current_suitability))
        if self.recovery_timeline_years < 0:
            self.recovery_timeline_years = 0.0


@dataclass
class RecoveryPlan:
    """Multi-site recovery plan for a federally listed species."""

    species: SpeciesHabitatRequirements
    target_sites: List[str]
    interventions: List[Dict[str, Any]]
    # Each intervention dict: {type, magnitude, cost_estimate,
    #                          predicted_habitat_gain, site_id}
    total_cost_estimate: float
    predicted_habitat_expansion_pct: float
    confidence_interval: Tuple[float, float]  # (lower, upper) on expansion

    @property
    def cost_per_pct_habitat(self) -> float:
        """Cost per percentage-point of habitat expansion."""
        if self.predicted_habitat_expansion_pct <= 0:
            return float("inf")
        return self.total_cost_estimate / self.predicted_habitat_expansion_pct


# ---------------------------------------------------------------------------
# Priority species database
# ---------------------------------------------------------------------------

PRIORITY_SPECIES: List[SpeciesHabitatRequirements] = [
    SpeciesHabitatRequirements(
        species_name="Higgins eye pearlymussel",
        scientific_name="Lampsilis higginsii",
        federal_status="endangered",
        temp_range=(10.0, 28.0),
        do_minimum=5.0,
        ph_range=(7.0, 8.5),
        substrate_preferences=["gravel", "sand", "cobble"],
        flow_requirements="moderate current, stable substrate",
        additional_constraints={"host_fish": ["Sauger", "Walleye", "Yellow perch"]},
    ),
    SpeciesHabitatRequirements(
        species_name="Topeka shiner",
        scientific_name="Notropis topeka",
        federal_status="endangered",
        temp_range=(5.0, 30.0),
        do_minimum=4.0,
        ph_range=(6.5, 8.5),
        substrate_preferences=["sand", "gravel", "silt"],
        flow_requirements="low-gradient pools in small prairie streams",
    ),
    SpeciesHabitatRequirements(
        species_name="Pallid sturgeon",
        scientific_name="Scaphirhynchus albus",
        federal_status="endangered",
        temp_range=(5.0, 25.0),
        do_minimum=5.0,
        ph_range=(7.0, 8.5),
        substrate_preferences=["sand", "gravel"],
        flow_requirements="large turbid rivers with natural flow regime",
        additional_constraints={"min_river_km": 200},
    ),
    SpeciesHabitatRequirements(
        species_name="Ozark hellbender",
        scientific_name="Cryptobranchus alleganiensis bishopi",
        federal_status="endangered",
        temp_range=(8.0, 22.0),
        do_minimum=7.0,
        ph_range=(6.8, 8.0),
        substrate_preferences=["large_flat_rocks", "boulders", "cobble"],
        flow_requirements="clear, fast-flowing Ozark streams",
    ),
    SpeciesHabitatRequirements(
        species_name="Cumberland darter",
        scientific_name="Etheostoma susanae",
        federal_status="endangered",
        temp_range=(8.0, 22.0),
        do_minimum=6.0,
        ph_range=(6.5, 8.0),
        substrate_preferences=["gravel", "cobble", "boulder"],
        flow_requirements="headwater streams with riffle-pool sequences",
    ),
    SpeciesHabitatRequirements(
        species_name="Snuffbox mussel",
        scientific_name="Epioblasma triquetra",
        federal_status="endangered",
        temp_range=(10.0, 26.0),
        do_minimum=5.5,
        ph_range=(7.0, 8.5),
        substrate_preferences=["gravel", "coarse_sand"],
        flow_requirements="swift current in clean streams and rivers",
        additional_constraints={"host_fish": ["Logperch"]},
    ),
    SpeciesHabitatRequirements(
        species_name="Diamond darter",
        scientific_name="Crystallaria cincotta",
        federal_status="endangered",
        temp_range=(10.0, 24.0),
        do_minimum=6.0,
        ph_range=(6.5, 8.0),
        substrate_preferences=["sand", "gravel"],
        flow_requirements="medium rivers with sand/gravel runs",
    ),
    SpeciesHabitatRequirements(
        species_name="Rayed bean mussel",
        scientific_name="Villosa fabalis",
        federal_status="endangered",
        temp_range=(10.0, 27.0),
        do_minimum=5.0,
        ph_range=(7.0, 8.5),
        substrate_preferences=["gravel", "sand", "cobble"],
        flow_requirements="small headwater streams with stable substrate",
        additional_constraints={"host_fish": ["Tippecanoe darter", "Greenside darter"]},
    ),
    SpeciesHabitatRequirements(
        species_name="Alabama sturgeon",
        scientific_name="Scaphirhynchus suttkusi",
        federal_status="endangered",
        temp_range=(10.0, 28.0),
        do_minimum=5.0,
        ph_range=(6.5, 8.5),
        substrate_preferences=["sand", "gravel", "clay"],
        flow_requirements="large free-flowing rivers of the Mobile Basin",
    ),
    SpeciesHabitatRequirements(
        species_name="Relict darter",
        scientific_name="Etheostoma chienense",
        federal_status="endangered",
        temp_range=(8.0, 24.0),
        do_minimum=5.5,
        ph_range=(6.5, 8.0),
        substrate_preferences=["gravel", "cobble"],
        flow_requirements="spring-fed streams in western Kentucky",
    ),
    SpeciesHabitatRequirements(
        species_name="Laurel dace",
        scientific_name="Chrosomus saylori",
        federal_status="endangered",
        temp_range=(8.0, 20.0),
        do_minimum=6.5,
        ph_range=(6.5, 7.5),
        substrate_preferences=["gravel", "sand", "leaf_litter"],
        flow_requirements="small headwater streams in Walden Ridge, TN",
    ),
    SpeciesHabitatRequirements(
        species_name="Spectaclecase mussel",
        scientific_name="Cumberlandia monodonta",
        federal_status="endangered",
        temp_range=(8.0, 28.0),
        do_minimum=5.0,
        ph_range=(7.0, 8.5),
        substrate_preferences=["boulders", "sheltered_crevices"],
        flow_requirements="large rivers with stable boulder substrates",
    ),
    SpeciesHabitatRequirements(
        species_name="Yellowfin madtom",
        scientific_name="Noturus flavipinnis",
        federal_status="threatened",
        temp_range=(10.0, 24.0),
        do_minimum=5.5,
        ph_range=(6.5, 8.0),
        substrate_preferences=["cobble", "boulder", "slab_rock"],
        flow_requirements="moderate-gradient streams in upper Tennessee River",
    ),
    SpeciesHabitatRequirements(
        species_name="Sheepnose mussel",
        scientific_name="Plethobasus cyphyus",
        federal_status="endangered",
        temp_range=(10.0, 28.0),
        do_minimum=5.0,
        ph_range=(7.0, 8.5),
        substrate_preferences=["gravel", "sand"],
        flow_requirements="medium to large rivers with moderate current",
        additional_constraints={"host_fish": ["Sauger"]},
    ),
    SpeciesHabitatRequirements(
        species_name="Scioto madtom",
        scientific_name="Noturus trautmani",
        federal_status="endangered",
        temp_range=(10.0, 22.0),
        do_minimum=6.0,
        ph_range=(6.5, 8.0),
        substrate_preferences=["gravel", "cobble"],
        flow_requirements="riffle habitats in central Ohio streams",
    ),
    SpeciesHabitatRequirements(
        species_name="Spring pygmy sunfish",
        scientific_name="Elassoma alabamae",
        federal_status="threatened",
        temp_range=(12.0, 22.0),
        do_minimum=5.0,
        ph_range=(6.5, 7.5),
        substrate_preferences=["aquatic_vegetation", "detritus"],
        flow_requirements="spring-fed pools with dense vegetation",
    ),
    SpeciesHabitatRequirements(
        species_name="Shovelnose sturgeon",
        scientific_name="Scaphirhynchus platorynchus",
        federal_status="candidate",
        temp_range=(5.0, 28.0),
        do_minimum=4.5,
        ph_range=(6.5, 8.5),
        substrate_preferences=["sand", "gravel"],
        flow_requirements="large rivers with sand/gravel channel bottoms",
    ),
    SpeciesHabitatRequirements(
        species_name="Rabbitsfoot mussel",
        scientific_name="Theliderma cylindrica",
        federal_status="threatened",
        temp_range=(10.0, 28.0),
        do_minimum=5.0,
        ph_range=(7.0, 8.5),
        substrate_preferences=["gravel", "sand", "cobble"],
        flow_requirements="small to medium streams with stable gravel beds",
        additional_constraints={"host_fish": ["Rosyface shiner", "Bluntnose minnow"]},
    ),
    SpeciesHabitatRequirements(
        species_name="Candy darter",
        scientific_name="Etheostoma osburni",
        federal_status="endangered",
        temp_range=(8.0, 22.0),
        do_minimum=6.0,
        ph_range=(6.5, 8.0),
        substrate_preferences=["cobble", "boulder"],
        flow_requirements="clear cool streams in the New River drainage",
        additional_constraints={"threat": "hybridization with variegate darter"},
    ),
    SpeciesHabitatRequirements(
        species_name="Neosho mucket mussel",
        scientific_name="Lampsilis rafinesqueana",
        federal_status="endangered",
        temp_range=(10.0, 28.0),
        do_minimum=5.0,
        ph_range=(7.0, 8.5),
        substrate_preferences=["gravel", "cobble"],
        flow_requirements="Ozark streams with stable cobble/gravel riffles",
        additional_constraints={"host_fish": ["Largemouth bass", "Smallmouth bass"]},
    ),
]


# ---------------------------------------------------------------------------
# Neural sub-networks
# ---------------------------------------------------------------------------

class _MLP(nn.Module):
    """Simple feed-forward MLP with ReLU activations and optional dropout."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_dim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(prev, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev = hidden_dim
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _InterventionAttentionRanker(nn.Module):
    """Attention-based ranker for intervention options.

    Given a site embedding and a set of intervention embeddings, produces
    a ranking score for each intervention using cross-attention.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.score_head = nn.Linear(embed_dim, 1)

    def forward(
        self,
        site_query: torch.Tensor,     # (B, D)
        intervention_keys: torch.Tensor,  # (B, K, D)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (scores [B, K], weighted_context [B, D])."""
        B, K, D = intervention_keys.shape

        # Project queries, keys, values
        Q = self.query_proj(site_query).unsqueeze(1)  # (B, 1, D)
        Kp = self.key_proj(intervention_keys)  # (B, K, D)
        V = self.value_proj(intervention_keys)  # (B, K, D)

        # Reshape for multi-head attention
        Q = Q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        Kp = Kp.view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, K, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn = (Q @ Kp.transpose(-2, -1)) / self.scale  # (B, H, 1, K)
        attn = F.softmax(attn, dim=-1)

        context = (attn @ V).transpose(1, 2).reshape(B, 1, D)  # (B, 1, D)
        context = context.squeeze(1)  # (B, D)

        # Per-intervention scores from value projection
        scores = self.score_head(
            intervention_keys + context.unsqueeze(1)
        ).squeeze(-1)  # (B, K)

        return scores, context


# ---------------------------------------------------------------------------
# RecoveryPlanner (nn.Module)
# ---------------------------------------------------------------------------

class RecoveryPlanner(nn.Module):
    """Endangered Species Recovery Planner.

    Uses learned sub-networks to:
    1. Score habitat suitability at candidate sites for a given species.
    2. Rank potential interventions using attention over a catalog.
    3. Predict recovery timelines under selected interventions.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of site embeddings (default: 256).
    num_species : int
        Number of species embedding slots (default: 150).
    num_env_params : int
        Number of environmental parameters tracked (default: 10).
    hidden_dim : int
        Hidden layer width (default: 128).
    """

    def __init__(
        self,
        embed_dim: int = SENTINEL_EMBEDDING_DIM,
        num_species: int = DEFAULT_NUM_SPECIES,
        num_env_params: int = DEFAULT_NUM_ENV_PARAMS,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_species = num_species
        self.num_env_params = num_env_params
        self.hidden_dim = hidden_dim

        # Learnable species embeddings
        self.species_embedding = nn.Embedding(num_species, embed_dim)

        # Intervention type embeddings (one per catalog entry)
        num_interventions = len(INTERVENTION_CATALOG)
        self.intervention_embedding = nn.Embedding(num_interventions, embed_dim)
        self._intervention_names = list(INTERVENTION_CATALOG.keys())

        # --- Sub-networks ---

        # Habitat scorer: site_embed concat species_embed -> suitability
        self.habitat_scorer = _MLP(
            in_dim=embed_dim * 2,
            hidden_dim=hidden_dim,
            out_dim=1,
            num_layers=3,
            dropout=0.1,
        )

        # Intervention ranker: attention over intervention options
        self.intervention_ranker = _InterventionAttentionRanker(
            embed_dim=embed_dim, num_heads=4,
        )

        # Timeline predictor: predicts recovery time in years
        self.timeline_predictor = _MLP(
            in_dim=embed_dim * 2 + num_env_params,
            hidden_dim=hidden_dim,
            out_dim=2,  # (mean_years, log_var_years)
            num_layers=3,
            dropout=0.1,
        )

        # Habitat gain predictor per intervention
        self.gain_predictor = _MLP(
            in_dim=embed_dim * 2,
            hidden_dim=hidden_dim,
            out_dim=1,
            num_layers=2,
            dropout=0.1,
        )

        # Confidence estimator
        self.confidence_head = _MLP(
            in_dim=embed_dim,
            hidden_dim=hidden_dim // 2,
            out_dim=2,  # (lower_offset, upper_offset) for CI
            num_layers=2,
            dropout=0.1,
        )

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------

    def forward(
        self,
        site_embedding: torch.Tensor,  # (B, D)
        species_idx: torch.Tensor,      # (B,) LongTensor
        interventions: Optional[torch.Tensor] = None,  # (B, K, D) or None
    ) -> Dict[str, torch.Tensor]:
        """Score habitat suitability and rank interventions.

        Parameters
        ----------
        site_embedding : Tensor (B, D)
            Per-site environmental embeddings from the SENTINEL encoder.
        species_idx : Tensor (B,) int64
            Index into the species embedding table.
        interventions : Tensor (B, K, D), optional
            Pre-computed intervention embeddings.  If *None*, the full
            catalog embeddings are used (broadcast across the batch).

        Returns
        -------
        Dict with keys:
            suitability : (B,) habitat suitability scores in [0, 1]
            intervention_scores : (B, K) ranking scores per intervention
            context : (B, D) attention-weighted intervention context
        """
        B = site_embedding.shape[0]

        # Species embedding lookup
        sp_embed = self.species_embedding(species_idx)  # (B, D)

        # Habitat suitability
        combined = torch.cat([site_embedding, sp_embed], dim=-1)  # (B, 2D)
        suitability = torch.sigmoid(self.habitat_scorer(combined)).squeeze(-1)

        # Intervention ranking
        if interventions is None:
            K = len(self._intervention_names)
            idx = torch.arange(K, device=site_embedding.device)
            interventions = self.intervention_embedding(idx)  # (K, D)
            interventions = interventions.unsqueeze(0).expand(B, -1, -1)

        scores, context = self.intervention_ranker(site_embedding, interventions)

        return {
            "suitability": suitability,
            "intervention_scores": scores,
            "context": context,
        }

    # ------------------------------------------------------------------
    # Habitat assessment
    # ------------------------------------------------------------------

    def assess_habitat(
        self,
        site_embedding: torch.Tensor,  # (1, D) or (D,)
        species_requirements: SpeciesHabitatRequirements,
        species_idx: int = 0,
        env_params: Optional[torch.Tensor] = None,  # (num_env_params,)
    ) -> float:
        """Compute scalar habitat suitability for one site-species pair.

        Parameters
        ----------
        site_embedding : Tensor
            Single-site embedding (1, D) or (D,).
        species_requirements : SpeciesHabitatRequirements
            Species habitat needs (used for interpretability context).
        species_idx : int
            Index into the species embedding table.
        env_params : Tensor, optional
            Current environmental parameters at the site.

        Returns
        -------
        float
            Habitat suitability in [0, 1].
        """
        self.eval()
        if site_embedding.dim() == 1:
            site_embedding = site_embedding.unsqueeze(0)

        sp_idx = torch.tensor([species_idx], device=site_embedding.device)
        with torch.no_grad():
            out = self.forward(site_embedding, sp_idx)
        return out["suitability"].item()

    # ------------------------------------------------------------------
    # Recovery planning
    # ------------------------------------------------------------------

    def plan_recovery(
        self,
        site_embeddings: torch.Tensor,     # (N, D)
        species_idx: int,
        budget_constraint: Optional[float] = None,
        env_params: Optional[torch.Tensor] = None,  # (N, num_env_params)
    ) -> RecoveryPlan:
        """Generate a cost-constrained recovery plan for a species.

        Parameters
        ----------
        site_embeddings : Tensor (N, D)
            Embeddings for *N* candidate sites.
        species_idx : int
            Index into species embedding + PRIORITY_SPECIES list.
        budget_constraint : float, optional
            Maximum total cost in USD.  Interventions are ranked by
            cost-effectiveness and added until the budget is exhausted.
        env_params : Tensor (N, num_env_params), optional
            Current environmental parameter values at each site.

        Returns
        -------
        RecoveryPlan
        """
        self.eval()
        N, D = site_embeddings.shape
        device = site_embeddings.device

        species_req = (
            PRIORITY_SPECIES[species_idx]
            if species_idx < len(PRIORITY_SPECIES)
            else PRIORITY_SPECIES[0]
        )

        sp_idx_tensor = torch.full((N,), species_idx, dtype=torch.long, device=device)

        with torch.no_grad():
            out = self.forward(site_embeddings, sp_idx_tensor)

        suitability = out["suitability"]  # (N,)
        int_scores = out["intervention_scores"]  # (N, K)

        K = len(self._intervention_names)

        # Compute per-(site, intervention) habitat gain predictions
        sp_embed = self.species_embedding(sp_idx_tensor)  # (N, D)
        gains = torch.zeros(N, K, device=device)
        for k in range(K):
            int_embed = self.intervention_embedding(
                torch.tensor([k], device=device)
            ).expand(N, -1)
            combo = torch.cat([site_embeddings + int_embed, sp_embed], dim=-1)
            gains[:, k] = torch.sigmoid(self.gain_predictor(combo)).squeeze(-1)

        # Compute per-intervention cost
        costs_per_unit = torch.tensor(
            [INTERVENTION_CATALOG[n]["cost_per_unit"] for n in self._intervention_names],
            device=device,
        )

        # Rank interventions by cost-effectiveness: gain / cost
        effectiveness = gains / (costs_per_unit.unsqueeze(0) + 1e-8)  # (N, K)

        # Flatten and sort by cost-effectiveness
        flat_eff = effectiveness.view(-1)
        flat_indices = torch.argsort(flat_eff, descending=True)

        selected: List[Dict[str, Any]] = []
        total_cost = 0.0
        target_sites_set: set = set()

        for idx_flat in flat_indices.tolist():
            site_i = idx_flat // K
            int_k = idx_flat % K
            int_name = self._intervention_names[int_k]
            catalog_entry = INTERVENTION_CATALOG[int_name]
            cost = float(catalog_entry["cost_per_unit"])
            gain = float(gains[site_i, int_k])

            if budget_constraint is not None and total_cost + cost > budget_constraint:
                continue

            # Skip marginal gains
            if gain < 0.01:
                continue

            selected.append({
                "type": int_name,
                "site_id": f"site_{site_i}",
                "magnitude": 1.0,
                "cost_estimate": cost,
                "predicted_habitat_gain": round(gain, 4),
            })
            target_sites_set.add(f"site_{site_i}")
            total_cost += cost

            # Limit to reasonable plan size
            if len(selected) >= 20:
                break

        total_gain_pct = sum(d["predicted_habitat_gain"] for d in selected) * 100.0

        # Confidence interval from the confidence head
        mean_embed = site_embeddings.mean(dim=0, keepdim=True)
        with torch.no_grad():
            ci_offsets = F.softplus(self.confidence_head(mean_embed)).squeeze(0)
        ci_lower = max(0.0, total_gain_pct - float(ci_offsets[0]) * 100.0)
        ci_upper = total_gain_pct + float(ci_offsets[1]) * 100.0

        return RecoveryPlan(
            species=species_req,
            target_sites=sorted(target_sites_set),
            interventions=selected,
            total_cost_estimate=total_cost,
            predicted_habitat_expansion_pct=round(total_gain_pct, 2),
            confidence_interval=(round(ci_lower, 2), round(ci_upper, 2)),
        )

    # ------------------------------------------------------------------
    # Timeline prediction
    # ------------------------------------------------------------------

    def predict_timeline(
        self,
        site_embedding: torch.Tensor,  # (B, D)
        species_idx: torch.Tensor,      # (B,)
        env_params: Optional[torch.Tensor] = None,  # (B, num_env_params)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict recovery timeline in years with uncertainty.

        Returns
        -------
        mean_years : Tensor (B,)
        std_years : Tensor (B,)
        """
        B = site_embedding.shape[0]
        sp_embed = self.species_embedding(species_idx)

        if env_params is None:
            env_params = torch.zeros(B, self.num_env_params, device=site_embedding.device)

        combined = torch.cat([site_embedding, sp_embed, env_params], dim=-1)
        out = self.timeline_predictor(combined)  # (B, 2)
        mean_years = F.softplus(out[:, 0])  # positive years
        std_years = F.softplus(out[:, 1])
        return mean_years, std_years


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

def _test_recovery_planner() -> None:
    """Smoke-test for RecoveryPlanner with random inputs."""
    import sys

    torch.manual_seed(42)
    device = torch.device("cpu")

    print("=" * 70)
    print("RecoveryPlanner -- standalone smoke test")
    print("=" * 70)

    # --- Verify data structures ---
    print(f"\nPRIORITY_SPECIES: {len(PRIORITY_SPECIES)} species loaded")
    for sp in PRIORITY_SPECIES[:3]:
        print(f"  {sp.species_name} ({sp.scientific_name}) -- {sp.federal_status}")
    print(f"INTERVENTION_CATALOG: {len(INTERVENTION_CATALOG)} types")
    for name, info in list(INTERVENTION_CATALOG.items())[:3]:
        print(f"  {name}: ${info['cost_per_unit']:,.0f}/{info['unit']}")

    # --- Instantiate model ---
    planner = RecoveryPlanner(
        embed_dim=256, num_species=150, num_env_params=10, hidden_dim=128,
    ).to(device)
    total_params = sum(p.numel() for p in planner.parameters())
    print(f"\nModel parameters: {total_params:,}")

    # --- Forward pass ---
    B, D = 4, 256
    site_emb = torch.randn(B, D, device=device)
    sp_idx = torch.randint(0, 20, (B,), device=device)

    out = planner(site_emb, sp_idx)
    print(f"\nForward pass:")
    print(f"  suitability shape: {out['suitability'].shape}")
    print(f"  suitability values: {out['suitability'].tolist()}")
    print(f"  intervention_scores shape: {out['intervention_scores'].shape}")

    # --- Habitat assessment ---
    score = planner.assess_habitat(
        site_emb[0], PRIORITY_SPECIES[0], species_idx=0,
    )
    print(f"\nHabitat assessment for {PRIORITY_SPECIES[0].species_name}: {score:.4f}")

    # --- Recovery plan ---
    plan = planner.plan_recovery(
        site_emb, species_idx=0, budget_constraint=1_000_000.0,
    )
    print(f"\nRecovery plan for {plan.species.species_name}:")
    print(f"  Target sites: {plan.target_sites}")
    print(f"  Interventions: {len(plan.interventions)}")
    print(f"  Total cost: ${plan.total_cost_estimate:,.0f}")
    print(f"  Habitat expansion: {plan.predicted_habitat_expansion_pct:.1f}%")
    print(f"  95% CI: ({plan.confidence_interval[0]:.1f}%, "
          f"{plan.confidence_interval[1]:.1f}%)")
    if plan.interventions:
        print(f"  Top intervention: {plan.interventions[0]}")

    # --- Timeline prediction ---
    mean_yr, std_yr = planner.predict_timeline(site_emb, sp_idx)
    print(f"\nTimeline prediction: {mean_yr.tolist()} +/- {std_yr.tolist()} years")

    # --- Gradient flow check ---
    planner.train()
    out2 = planner(site_emb, sp_idx)
    loss = out2["suitability"].sum()
    loss.backward()
    grad_ok = all(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in planner.parameters()
        if p.requires_grad
    )
    print(f"\nGradient flow check: {'PASS' if grad_ok else 'PARTIAL'}")

    print("\n" + "=" * 70)
    print("All RecoveryPlanner smoke tests passed.")
    print("=" * 70)


if __name__ == "__main__":
    _test_recovery_planner()
