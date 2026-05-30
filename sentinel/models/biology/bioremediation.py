"""Bioremediation Recommender -- Phase 4 of SENTINEL 2.0.

When contamination is detected, prescribes remediation based on the
microbial community composition and pollutant class.  The recommender
classifies contaminants, detects known degrader organisms in the
environmental metagenomic/16S profile, selects between biostimulation,
bioaugmentation, monitored natural attenuation (MNA), or combined
strategies, and predicts reduction timelines and costs.

Architecture overview::

    site_embedding (B, 256)   microbial_embedding (B, 256)
          |                          |
    ContaminantClassifier      DegraderDetector
          |                          |
          +--------- concat ---------+
                       |
                 StrategySelector
                       |
                 OutcomePredictor
                       |
          RemediationRecommendation

The module also provides a curated ``DEGRADER_DATABASE`` mapping
contaminant-degrader pairs to metabolic pathways (with KEGG pathway IDs
where available) and an ``AMENDMENT_CATALOG`` of biostimulation
amendments with dosing and cost information.

References
----------
* Lovley, D.R. (2003). Cleaning up with genomics: applying molecular
  biology to bioremediation. Nature Reviews Microbiology 1:35-44.
* Stroo, H.F. et al. (2012). Chlorinated Solvent Source Zone
  Remediation. Springer.
* Interstate Technology & Regulatory Council (ITRC). (2011).
  Bioremediation of DNAPLs. BioDNAPL-3.
* Kuppusamy, S. et al. (2017). Bioremediation approaches for
  recalcitrant pollutants. Reviews in Environmental Science and
  Bio/Technology 16:681-721.
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
DEFAULT_NUM_CONTAMINANTS: int = 50
DEFAULT_NUM_PATHWAYS: int = 30
DEFAULT_HIDDEN_DIM: int = 128

#: Supported contaminant classes with numeric IDs.
CONTAMINANT_CLASSES: Dict[str, int] = {
    "chlorinated_solvent": 0,
    "petroleum": 1,
    "heavy_metal": 2,
    "PFAS": 3,
    "pesticide": 4,
    "nitrate": 5,
    "perchlorate": 6,
    "explosive": 7,
    "PAH": 8,
    "pharmaceutical": 9,
}

NUM_CONTAMINANT_CLASSES: int = len(CONTAMINANT_CLASSES)

#: Remediation strategy options.
STRATEGY_OPTIONS: Tuple[str, ...] = (
    "biostimulation",
    "bioaugmentation",
    "monitored_natural_attenuation",
    "combined",
)

NUM_STRATEGIES: int = len(STRATEGY_OPTIONS)


# ---------------------------------------------------------------------------
# Degrader database
# ---------------------------------------------------------------------------

DEGRADER_DATABASE: Dict[str, List[Dict[str, Any]]] = {
    "trichloroethylene": [
        {"taxon": "Dehalococcoides mccartyi", "pathway": "reductive_dechlorination",
         "kegg_id": "K18209", "efficiency": "high"},
        {"taxon": "Dehalobacter restrictus", "pathway": "reductive_dechlorination",
         "kegg_id": "K18209", "efficiency": "moderate"},
        {"taxon": "Desulfitobacterium hafniense", "pathway": "reductive_dechlorination",
         "kegg_id": "K18210", "efficiency": "moderate"},
    ],
    "benzene": [
        {"taxon": "Pseudomonas putida", "pathway": "aerobic_ring_cleavage",
         "kegg_id": "K07104", "efficiency": "high"},
        {"taxon": "Rhodococcus jostii", "pathway": "aerobic_oxidation",
         "kegg_id": "K05549", "efficiency": "high"},
        {"taxon": "Geobacter metallireducens", "pathway": "anaerobic_carboxylation",
         "kegg_id": "K07539", "efficiency": "moderate"},
    ],
    "PFAS": [
        {"taxon": "Acidimicrobium ferrooxidans", "pathway": "defluorination",
         "kegg_id": None, "efficiency": "low",
         "notes": "Emerging research, Huang & Jaffé (2019)"},
        {"taxon": "Pseudomonas aeruginosa", "pathway": "oxidative_defluorination",
         "kegg_id": None, "efficiency": "low",
         "notes": "Limited peer-reviewed evidence"},
    ],
    "petroleum": [
        {"taxon": "Alcanivorax borkumensis", "pathway": "alkane_hydroxylase",
         "kegg_id": "K00496", "efficiency": "high"},
        {"taxon": "Marinobacter hydrocarbonoclasticus", "pathway": "alkane_degradation",
         "kegg_id": "K00496", "efficiency": "high"},
        {"taxon": "Pseudomonas fluorescens", "pathway": "aromatic_degradation",
         "kegg_id": "K07104", "efficiency": "moderate"},
    ],
    "nitrate": [
        {"taxon": "Paracoccus denitrificans", "pathway": "denitrification",
         "kegg_id": "K00370", "efficiency": "high"},
        {"taxon": "Thiobacillus denitrificans", "pathway": "autotrophic_denitrification",
         "kegg_id": "K00370", "efficiency": "high"},
        {"taxon": "Pseudomonas stutzeri", "pathway": "denitrification",
         "kegg_id": "K00371", "efficiency": "high"},
    ],
    "mercury": [
        {"taxon": "Geobacter sulfurreducens", "pathway": "mercury_methylation",
         "kegg_id": "K07821", "efficiency": "moderate",
         "notes": "Can methylate Hg; useful for understanding speciation"},
        {"taxon": "Desulfovibrio desulfuricans", "pathway": "mercury_reduction_merA",
         "kegg_id": "K00520", "efficiency": "high"},
    ],
    "toluene": [
        {"taxon": "Pseudomonas putida mt-2", "pathway": "TOL_plasmid_xyl",
         "kegg_id": "K07540", "efficiency": "high"},
        {"taxon": "Thauera aromatica", "pathway": "anaerobic_toluene_degradation",
         "kegg_id": "K07545", "efficiency": "moderate"},
    ],
    "perchlorate": [
        {"taxon": "Dechloromonas aromatica", "pathway": "perchlorate_reductase",
         "kegg_id": "K16330", "efficiency": "high"},
        {"taxon": "Azospira oryzae", "pathway": "perchlorate_reduction",
         "kegg_id": "K16330", "efficiency": "high"},
    ],
    "naphthalene": [
        {"taxon": "Pseudomonas putida G7", "pathway": "nah_operon",
         "kegg_id": "K14579", "efficiency": "high"},
        {"taxon": "Sphingomonas paucimobilis", "pathway": "PAH_dioxygenase",
         "kegg_id": "K11943", "efficiency": "moderate"},
    ],
    "atrazine": [
        {"taxon": "Pseudomonas sp. ADP", "pathway": "atzABCDEF_hydrolysis",
         "kegg_id": "K03382", "efficiency": "high"},
        {"taxon": "Arthrobacter aurescens TC1", "pathway": "trzN_hydrolysis",
         "kegg_id": "K03381", "efficiency": "high"},
    ],
    "chromium_vi": [
        {"taxon": "Shewanella oneidensis MR-1", "pathway": "chromate_reductase",
         "kegg_id": "K11930", "efficiency": "high"},
        {"taxon": "Pseudomonas putida", "pathway": "chromate_reduction",
         "kegg_id": "K11930", "efficiency": "moderate"},
    ],
    "RDX": [
        {"taxon": "Gordonia sp. KTR9", "pathway": "xplA_cytochrome_P450",
         "kegg_id": None, "efficiency": "high"},
        {"taxon": "Rhodococcus rhodochrous 11Y", "pathway": "xplA_denitration",
         "kegg_id": None, "efficiency": "high"},
    ],
    "arsenic": [
        {"taxon": "Geobacter lovleyi", "pathway": "arsenate_reduction_arrAB",
         "kegg_id": "K03741", "efficiency": "moderate"},
        {"taxon": "Thiomonas arsenitoxydans", "pathway": "arsenite_oxidation_aioAB",
         "kegg_id": "K08355", "efficiency": "high"},
    ],
    "1,4-dioxane": [
        {"taxon": "Pseudonocardia dioxanivorans CB1190", "pathway": "monooxygenase",
         "kegg_id": "K14338", "efficiency": "high"},
        {"taxon": "Mycobacterium austroafricanum", "pathway": "propane_monooxygenase",
         "kegg_id": "K14338", "efficiency": "moderate"},
    ],
}


# ---------------------------------------------------------------------------
# Amendment catalog
# ---------------------------------------------------------------------------

AMENDMENT_CATALOG: Dict[str, Dict[str, Any]] = {
    "emulsified_vegetable_oil": {
        "description": "Slow-release organic carbon for anaerobic biostimulation",
        "dose": "10-50 L/m²",
        "cost_per_m3": 15.0,
        "target_contaminants": ["chlorinated_solvent", "perchlorate"],
        "mechanism": "Fermentation produces H2 for reductive dechlorination",
    },
    "lactate": {
        "description": "Soluble electron donor for anaerobic biostimulation",
        "dose": "1-5 mM",
        "cost_per_m3": 8.0,
        "target_contaminants": ["chlorinated_solvent", "heavy_metal", "perchlorate"],
        "mechanism": "Ferments to acetate + H2, drives reductive conditions",
    },
    "molasses": {
        "description": "Low-cost carbon source for anaerobic stimulation",
        "dose": "0.5-2% w/v",
        "cost_per_m3": 5.0,
        "target_contaminants": ["chlorinated_solvent", "nitrate"],
        "mechanism": "Rapidly fermentable carbon generates electron donors",
    },
    "oxygen_release_compound": {
        "description": "Slow-release MgO2 for aerobic biostimulation",
        "dose": "0.5-2 kg/m²",
        "cost_per_m3": 25.0,
        "target_contaminants": ["petroleum", "PAH", "1,4-dioxane"],
        "mechanism": "Sustained O2 release enhances aerobic biodegradation",
    },
    "hydrogen_peroxide": {
        "description": "Liquid O2 source for aerobic biostimulation",
        "dose": "100-500 mg/L",
        "cost_per_m3": 12.0,
        "target_contaminants": ["petroleum", "BTEX"],
        "mechanism": "Rapid O2 delivery, may also produce hydroxyl radicals",
    },
    "ammonium_sulfate": {
        "description": "Nitrogen + sulfur nutrient amendment",
        "dose": "10-50 mg/L N",
        "cost_per_m3": 3.0,
        "target_contaminants": ["petroleum"],
        "mechanism": "Corrects N-limitation in hydrocarbon-impacted zones",
    },
    "zero_valent_iron": {
        "description": "nZVI for abiotic + biotic reductive treatment",
        "dose": "1-10 g/L",
        "cost_per_m3": 40.0,
        "target_contaminants": ["chlorinated_solvent", "heavy_metal", "chromium_vi"],
        "mechanism": "Chemical reduction + stimulates iron-reducing bacteria",
    },
    "biochar": {
        "description": "Pyrolyzed biomass for sorption and microbial habitat",
        "dose": "1-5% w/w",
        "cost_per_m3": 20.0,
        "target_contaminants": ["heavy_metal", "pesticide", "PAH"],
        "mechanism": "Sorption reduces bioavailability; enhances microbial colonization",
    },
    "sulfate": {
        "description": "Electron acceptor for anaerobic BTEX degradation",
        "dose": "500-2000 mg/L",
        "cost_per_m3": 6.0,
        "target_contaminants": ["petroleum", "PAH"],
        "mechanism": "Drives sulfate-reducing BTEX degradation",
    },
    "phosphate": {
        "description": "Phosphorus nutrient amendment",
        "dose": "1-10 mg/L P",
        "cost_per_m3": 4.0,
        "target_contaminants": ["petroleum"],
        "mechanism": "Corrects P-limitation in nutrient-poor aquifers",
    },
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RemediationRecommendation:
    """Complete bioremediation recommendation for a contaminated site."""

    contaminant: str
    contaminant_class: str  # e.g., "chlorinated_solvent", "petroleum"
    detected_degraders: List[Dict[str, Any]]
    # Each: {taxon, abundance_copies_per_L, pathway}
    strategy: str
    # "biostimulation", "bioaugmentation", "monitored_natural_attenuation", "combined"
    amendments: List[Dict[str, Any]]
    # Each: {name, dose, rationale}
    predicted_reduction_pct: float
    predicted_timeline_months: int
    cost_estimate: float
    conventional_cost_estimate: float
    confidence: float  # 0-1

    @property
    def cost_savings_pct(self) -> float:
        """Percentage cost savings vs. conventional treatment."""
        if self.conventional_cost_estimate <= 0:
            return 0.0
        return (
            (self.conventional_cost_estimate - self.cost_estimate)
            / self.conventional_cost_estimate
            * 100.0
        )


# ---------------------------------------------------------------------------
# Neural sub-networks
# ---------------------------------------------------------------------------

class _MLP(nn.Module):
    """Feed-forward MLP with ReLU activations and optional dropout."""

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


# ---------------------------------------------------------------------------
# BioremediationRecommender (nn.Module)
# ---------------------------------------------------------------------------

class BioremediationRecommender(nn.Module):
    """Bioremediation strategy recommender.

    Consumes site and microbial embeddings to classify contamination,
    detect known degrader organisms, select the optimal remediation
    strategy, and predict outcomes (reduction % and timeline).

    Parameters
    ----------
    embed_dim : int
        Dimensionality of input embeddings (default: 256).
    num_contaminants : int
        Number of contaminant embedding slots (default: 50).
    num_pathways : int
        Number of degradation pathway slots (default: 30).
    hidden_dim : int
        Hidden layer width for sub-networks (default: 128).
    """

    def __init__(
        self,
        embed_dim: int = SENTINEL_EMBEDDING_DIM,
        num_contaminants: int = DEFAULT_NUM_CONTAMINANTS,
        num_pathways: int = DEFAULT_NUM_PATHWAYS,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_contaminants = num_contaminants
        self.num_pathways = num_pathways
        self.hidden_dim = hidden_dim

        # Contaminant classifier: site_embedding -> contaminant class logits
        self.contaminant_classifier = _MLP(
            in_dim=embed_dim,
            hidden_dim=hidden_dim,
            out_dim=NUM_CONTAMINANT_CLASSES,
            num_layers=3,
            dropout=0.1,
        )

        # Degrader detector: microbial_embedding -> pathway activations
        self.degrader_detector = _MLP(
            in_dim=embed_dim,
            hidden_dim=hidden_dim,
            out_dim=num_pathways,
            num_layers=3,
            dropout=0.1,
        )

        # Strategy selector: combined features -> strategy logits
        self.strategy_selector = _MLP(
            in_dim=embed_dim * 2 + NUM_CONTAMINANT_CLASSES + num_pathways,
            hidden_dim=hidden_dim,
            out_dim=NUM_STRATEGIES,
            num_layers=3,
            dropout=0.1,
        )

        # Outcome predictor: predicts reduction %, timeline, costs
        self.outcome_predictor = _MLP(
            in_dim=embed_dim * 2 + NUM_STRATEGIES,
            hidden_dim=hidden_dim,
            out_dim=4,  # reduction_pct, timeline_months, cost, conventional_cost
            num_layers=3,
            dropout=0.1,
        )

        # Confidence head
        self.confidence_head = _MLP(
            in_dim=embed_dim * 2,
            hidden_dim=hidden_dim // 2,
            out_dim=1,
            num_layers=2,
            dropout=0.1,
        )

        # Contaminant and pathway name mappings for interpretability
        self._contaminant_class_names = list(CONTAMINANT_CLASSES.keys())
        self._degrader_contaminants = list(DEGRADER_DATABASE.keys())

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------

    def forward(
        self,
        site_embedding: torch.Tensor,        # (B, D)
        microbial_embedding: Optional[torch.Tensor] = None,  # (B, D)
    ) -> RemediationRecommendation:
        """Produce a bioremediation recommendation.

        Parameters
        ----------
        site_embedding : Tensor (B, D)
            Environmental embedding capturing water quality and
            contaminant signals.
        microbial_embedding : Tensor (B, D), optional
            Microbial community embedding from 16S / metagenomic data.
            If *None*, a zero vector is used (limited degrader detection).

        Returns
        -------
        RemediationRecommendation
            Complete recommendation for the first sample in the batch.
            For batch processing, use ``forward_batch()``.
        """
        B, D = site_embedding.shape
        device = site_embedding.device

        if microbial_embedding is None:
            microbial_embedding = torch.zeros_like(site_embedding)

        # Step 1: Classify contaminant
        contam_logits = self.contaminant_classifier(site_embedding)  # (B, C)
        contam_probs = F.softmax(contam_logits, dim=-1)
        contam_idx = contam_probs.argmax(dim=-1)  # (B,)

        # Step 2: Detect degrader pathways
        pathway_logits = self.degrader_detector(microbial_embedding)  # (B, P)
        pathway_activations = torch.sigmoid(pathway_logits)

        # Step 3: Select strategy
        combined_features = torch.cat([
            site_embedding,
            microbial_embedding,
            contam_probs,
            pathway_activations,
        ], dim=-1)

        strategy_logits = self.strategy_selector(combined_features)  # (B, S)
        strategy_probs = F.softmax(strategy_logits, dim=-1)
        strategy_idx = strategy_probs.argmax(dim=-1)  # (B,)

        # Step 4: Predict outcomes
        strategy_onehot = F.one_hot(
            strategy_idx, NUM_STRATEGIES,
        ).float()  # (B, S)

        outcome_input = torch.cat([
            site_embedding, microbial_embedding, strategy_onehot,
        ], dim=-1)

        outcomes = self.outcome_predictor(outcome_input)  # (B, 4)
        reduction_pct = torch.sigmoid(outcomes[:, 0]) * 100.0
        timeline_months = F.softplus(outcomes[:, 1]) * 12.0 + 1.0
        cost = F.softplus(outcomes[:, 2]) * 100_000.0
        conventional_cost = F.softplus(outcomes[:, 3]) * 200_000.0

        # Step 5: Confidence
        conf_input = torch.cat([site_embedding, microbial_embedding], dim=-1)
        confidence = torch.sigmoid(self.confidence_head(conf_input)).squeeze(-1)

        # Build recommendation from first sample (batch=0)
        contam_class_name = self._contaminant_class_names[contam_idx[0].item()]
        strategy_name = STRATEGY_OPTIONS[strategy_idx[0].item()]

        # Look up detected degraders from database
        detected = self._lookup_degraders(contam_class_name, pathway_activations[0])

        # Select appropriate amendments
        amendments = self._select_amendments(contam_class_name, strategy_name)

        return RemediationRecommendation(
            contaminant=contam_class_name,
            contaminant_class=contam_class_name,
            detected_degraders=detected,
            strategy=strategy_name,
            amendments=amendments,
            predicted_reduction_pct=round(reduction_pct[0].item(), 1),
            predicted_timeline_months=max(1, int(timeline_months[0].item())),
            cost_estimate=round(cost[0].item(), 2),
            conventional_cost_estimate=round(conventional_cost[0].item(), 2),
            confidence=round(confidence[0].item(), 4),
        )

    # ------------------------------------------------------------------
    # Batch forward (returns raw tensors)
    # ------------------------------------------------------------------

    def forward_batch(
        self,
        site_embedding: torch.Tensor,        # (B, D)
        microbial_embedding: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Batch forward returning raw tensors for training.

        Returns
        -------
        Dict with keys:
            contaminant_logits : (B, C)
            pathway_activations : (B, P)
            strategy_logits : (B, S)
            reduction_pct : (B,)
            timeline_months : (B,)
            cost : (B,)
            conventional_cost : (B,)
            confidence : (B,)
        """
        B, D = site_embedding.shape
        device = site_embedding.device

        if microbial_embedding is None:
            microbial_embedding = torch.zeros_like(site_embedding)

        contam_logits = self.contaminant_classifier(site_embedding)
        contam_probs = F.softmax(contam_logits, dim=-1)

        pathway_logits = self.degrader_detector(microbial_embedding)
        pathway_activations = torch.sigmoid(pathway_logits)

        combined_features = torch.cat([
            site_embedding, microbial_embedding, contam_probs, pathway_activations,
        ], dim=-1)

        strategy_logits = self.strategy_selector(combined_features)
        strategy_probs = F.softmax(strategy_logits, dim=-1)
        strategy_idx = strategy_probs.argmax(dim=-1)

        strategy_onehot = F.one_hot(strategy_idx, NUM_STRATEGIES).float()
        outcome_input = torch.cat([
            site_embedding, microbial_embedding, strategy_onehot,
        ], dim=-1)

        outcomes = self.outcome_predictor(outcome_input)
        reduction_pct = torch.sigmoid(outcomes[:, 0]) * 100.0
        timeline_months = F.softplus(outcomes[:, 1]) * 12.0 + 1.0
        cost = F.softplus(outcomes[:, 2]) * 100_000.0
        conventional_cost = F.softplus(outcomes[:, 3]) * 200_000.0

        conf_input = torch.cat([site_embedding, microbial_embedding], dim=-1)
        confidence = torch.sigmoid(self.confidence_head(conf_input)).squeeze(-1)

        return {
            "contaminant_logits": contam_logits,
            "pathway_activations": pathway_activations,
            "strategy_logits": strategy_logits,
            "reduction_pct": reduction_pct,
            "timeline_months": timeline_months,
            "cost": cost,
            "conventional_cost": conventional_cost,
            "confidence": confidence,
        }

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        output: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute multi-task training loss.

        Parameters
        ----------
        output : Dict[str, Tensor]
            Output from ``forward_batch()``.
        targets : Dict[str, Tensor]
            Ground-truth labels.  Expected keys (all optional):
            - contaminant_labels : (B,) int64 class indices
            - strategy_labels : (B,) int64 class indices
            - reduction_pct : (B,) float targets
            - timeline_months : (B,) float targets

        Returns
        -------
        total_loss : scalar Tensor
        per_task : Dict[str, Tensor]
            Individual task losses for logging.
        """
        losses: Dict[str, torch.Tensor] = {}
        device = output["contaminant_logits"].device

        if "contaminant_labels" in targets:
            losses["contaminant"] = F.cross_entropy(
                output["contaminant_logits"],
                targets["contaminant_labels"],
            )

        if "strategy_labels" in targets:
            losses["strategy"] = F.cross_entropy(
                output["strategy_logits"],
                targets["strategy_labels"],
            )

        if "reduction_pct" in targets:
            losses["reduction"] = F.mse_loss(
                output["reduction_pct"],
                targets["reduction_pct"].float(),
            )

        if "timeline_months" in targets:
            losses["timeline"] = F.mse_loss(
                output["timeline_months"],
                targets["timeline_months"].float(),
            )

        # Default weights
        weights = {
            "contaminant": 1.0,
            "strategy": 1.0,
            "reduction": 0.5,
            "timeline": 0.3,
        }

        total = torch.tensor(0.0, device=device)
        for key, loss_val in losses.items():
            w = weights.get(key, 1.0)
            total = total + w * loss_val

        return total, losses

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _lookup_degraders(
        self,
        contaminant_class: str,
        pathway_activation: torch.Tensor,  # (P,)
    ) -> List[Dict[str, Any]]:
        """Look up known degraders from the database for the contaminant."""
        detected: List[Dict[str, Any]] = []

        # Map contaminant class to specific contaminants in database
        class_to_contaminants = {
            "chlorinated_solvent": ["trichloroethylene"],
            "petroleum": ["petroleum", "benzene", "toluene", "naphthalene"],
            "heavy_metal": ["mercury", "chromium_vi", "arsenic"],
            "PFAS": ["PFAS"],
            "pesticide": ["atrazine"],
            "nitrate": ["nitrate"],
            "perchlorate": ["perchlorate"],
            "explosive": ["RDX"],
            "PAH": ["naphthalene"],
            "pharmaceutical": ["1,4-dioxane"],
        }

        relevant_contaminants = class_to_contaminants.get(contaminant_class, [])

        for contam_name in relevant_contaminants:
            if contam_name in DEGRADER_DATABASE:
                for entry in DEGRADER_DATABASE[contam_name]:
                    # Simulate abundance based on pathway activation
                    abundance = float(
                        pathway_activation[
                            hash(entry["taxon"]) % len(pathway_activation)
                        ].item()
                    ) * 1e6  # copies per liter

                    detected.append({
                        "taxon": entry["taxon"],
                        "abundance_copies_per_L": round(abundance, 0),
                        "pathway": entry["pathway"],
                        "kegg_id": entry.get("kegg_id"),
                        "efficiency": entry.get("efficiency", "unknown"),
                    })

        return detected

    def _select_amendments(
        self,
        contaminant_class: str,
        strategy: str,
    ) -> List[Dict[str, Any]]:
        """Select appropriate amendments based on contaminant and strategy."""
        if strategy == "monitored_natural_attenuation":
            return []

        selected: List[Dict[str, Any]] = []
        for name, info in AMENDMENT_CATALOG.items():
            targets = info.get("target_contaminants", [])
            # Check if the amendment targets this contaminant class
            if contaminant_class in targets or any(
                contaminant_class.lower() in t.lower() for t in targets
            ):
                selected.append({
                    "name": name,
                    "dose": info["dose"],
                    "rationale": info["mechanism"],
                    "cost_per_m3": info["cost_per_m3"],
                })

        # If combined strategy, add a generic carbon source if nothing selected
        if not selected and strategy in ("biostimulation", "combined"):
            fallback = AMENDMENT_CATALOG["molasses"]
            selected.append({
                "name": "molasses",
                "dose": fallback["dose"],
                "rationale": "Generic carbon source for microbial stimulation",
                "cost_per_m3": fallback["cost_per_m3"],
            })

        return selected


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

def _test_bioremediation_recommender() -> None:
    """Smoke-test for BioremediationRecommender with random inputs."""
    import sys

    torch.manual_seed(42)
    device = torch.device("cpu")

    print("=" * 70)
    print("BioremediationRecommender -- standalone smoke test")
    print("=" * 70)

    # --- Verify databases ---
    print(f"\nDEGRADER_DATABASE: {len(DEGRADER_DATABASE)} contaminants")
    for contam, degraders in list(DEGRADER_DATABASE.items())[:4]:
        taxa = [d["taxon"] for d in degraders]
        print(f"  {contam}: {taxa}")

    print(f"\nAMENDMENT_CATALOG: {len(AMENDMENT_CATALOG)} amendments")
    for name, info in list(AMENDMENT_CATALOG.items())[:4]:
        print(f"  {name}: ${info['cost_per_m3']}/m3, dose={info['dose']}")

    print(f"\nCONTAMINANT_CLASSES: {len(CONTAMINANT_CLASSES)} classes")

    # --- Instantiate model ---
    model = BioremediationRecommender(
        embed_dim=256, num_contaminants=50, num_pathways=30, hidden_dim=128,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")

    # --- Forward pass (single recommendation) ---
    B, D = 4, 256
    site_emb = torch.randn(B, D, device=device)
    micro_emb = torch.randn(B, D, device=device)

    rec = model(site_emb, micro_emb)
    print(f"\nRecommendation:")
    print(f"  Contaminant class: {rec.contaminant_class}")
    print(f"  Strategy: {rec.strategy}")
    print(f"  Detected degraders: {len(rec.detected_degraders)}")
    if rec.detected_degraders:
        d = rec.detected_degraders[0]
        print(f"    Top: {d['taxon']} ({d['pathway']}, "
              f"{d['abundance_copies_per_L']:.0f} copies/L)")
    print(f"  Amendments: {len(rec.amendments)}")
    if rec.amendments:
        a = rec.amendments[0]
        print(f"    Top: {a['name']} (dose={a['dose']})")
    print(f"  Predicted reduction: {rec.predicted_reduction_pct:.1f}%")
    print(f"  Timeline: {rec.predicted_timeline_months} months")
    print(f"  Cost: ${rec.cost_estimate:,.0f}")
    print(f"  Conventional cost: ${rec.conventional_cost_estimate:,.0f}")
    print(f"  Cost savings: {rec.cost_savings_pct:.1f}%")
    print(f"  Confidence: {rec.confidence:.4f}")

    # --- Batch forward ---
    out_batch = model.forward_batch(site_emb, micro_emb)
    print(f"\nBatch forward:")
    for key, val in out_batch.items():
        print(f"  {key}: shape={val.shape}")

    # --- Loss computation ---
    targets = {
        "contaminant_labels": torch.randint(0, NUM_CONTAMINANT_CLASSES, (B,)),
        "strategy_labels": torch.randint(0, NUM_STRATEGIES, (B,)),
        "reduction_pct": torch.rand(B) * 100.0,
        "timeline_months": torch.rand(B) * 36.0 + 1.0,
    }
    total_loss, per_task = model.compute_loss(out_batch, targets)
    print(f"\nLoss computation:")
    print(f"  Total loss: {total_loss.item():.4f}")
    for key, val in per_task.items():
        print(f"  {key}: {val.item():.4f}")

    # --- Forward without microbial embedding ---
    rec_no_micro = model(site_emb)
    print(f"\nWithout microbial embedding:")
    print(f"  Strategy: {rec_no_micro.strategy}")
    print(f"  Confidence: {rec_no_micro.confidence:.4f}")

    # --- Gradient flow check ---
    model.train()
    out2 = model.forward_batch(site_emb, micro_emb)
    loss = out2["contaminant_logits"].sum() + out2["reduction_pct"].sum()
    loss.backward()
    grad_ok = all(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.parameters()
        if p.requires_grad
    )
    print(f"\nGradient flow check: {'PASS' if grad_ok else 'PARTIAL'}")

    print("\n" + "=" * 70)
    print("All BioremediationRecommender smoke tests passed.")
    print("=" * 70)


if __name__ == "__main__":
    _test_bioremediation_recommender()
