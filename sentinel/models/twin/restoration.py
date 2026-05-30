"""Restoration outcome predictor for the Digital Aquatic Ecosystem Twin.

Extends the counterfactual engine to predict long-term ecological outcomes
of restoration projects, estimate cost-effectiveness, optimize intervention
portfolios, and recommend bioremediation strategies.

Modules
-------
RestorationProject
    Dataclass specifying a restoration project with type, area, cost, and
    site embedding.
RestorationOutcomePredictor
    Runs counterfactual simulation over the project lifetime and predicts
    IBI score changes, species community shifts, and indicator species
    return probabilities.
CostOptimizer
    Gradient-based optimization through the differentiable twin to find
    minimum-cost intervention packages that achieve a target ecological
    outcome (Pareto frontier of cost vs. benefit).
BioremediationRecommender
    Suggests biostimulation vs. bioaugmentation strategies based on detected
    microbial community composition and pollutant class.

References
----------
* Bernhardt, E.S. et al. (2005). Synthesizing U.S. river restoration
  efforts. Science 308:636-637.
* Palmer, M.A. et al. (2005). Standards for ecologically successful river
  restoration. J. Applied Ecology 42:208-217.
* Kadlec, R.H. & Wallace, S. (2009). Treatment Wetlands, 2nd ed.
* Karr, J.R. (1981). Assessment of biotic integrity using fish
  communities. Fisheries 6(6):21-27.
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from sentinel.models.twin.twin_engine import (
    STATE_VARS,
    NUM_STATE_VARS,
    FORECAST_HORIZONS,
    SENTINEL_EMBEDDING_DIM,
    DigitalTwinEngine,
    TwinOutput,
)
from sentinel.models.twin.counterfactual import (
    CounterfactualOutput,
    CounterfactualSimulator,
    InterventionLibrary,
    InterventionSpec,
    InterventionType,
    SpatialScope,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Standard restoration project types.
class ProjectType(str, Enum):
    """Supported restoration project categories."""

    RIPARIAN_BUFFER = "riparian_buffer"
    WETLAND_CONSTRUCTION = "wetland_construction"
    DAM_REMOVAL = "dam_removal"
    AGRICULTURAL_BMP = "agricultural_bmp"
    STORMWATER_MANAGEMENT = "stormwater_management"


#: Mapping from project types to their default intervention spec parameters.
#: Each project type maps to a list of (InterventionType, target_parameter)
#: pairs representing the primary ecological mechanisms.
_PROJECT_TO_INTERVENTIONS: Dict[
    ProjectType, List[Tuple[InterventionType, str, float]]
] = {
    ProjectType.RIPARIAN_BUFFER: [
        (InterventionType.RIPARIAN_BUFFER, "total_nitrogen", 0.5),
    ],
    ProjectType.WETLAND_CONSTRUCTION: [
        (InterventionType.WETLAND_RESTORATION, "total_nitrogen", 0.6),
    ],
    ProjectType.DAM_REMOVAL: [
        (InterventionType.DAM_REMOVAL, "dissolved_oxygen", 0.4),
    ],
    ProjectType.AGRICULTURAL_BMP: [
        (InterventionType.NUTRIENT_REDUCTION, "total_nitrogen", 0.3),
        (InterventionType.NUTRIENT_REDUCTION, "total_phosphorus", 0.25),
    ],
    ProjectType.STORMWATER_MANAGEMENT: [
        (InterventionType.POINT_SOURCE_CONTROL, "total_nitrogen", 0.35),
    ],
}

#: Per-hectare cost coefficients (USD/ha) for scaling intervention magnitude
#: by project area.  Derived from Bernhardt et al. (2005) meta-analysis.
_AREA_COST_COEFFICIENTS: Dict[ProjectType, float] = {
    ProjectType.RIPARIAN_BUFFER: 12_000.0,
    ProjectType.WETLAND_CONSTRUCTION: 45_000.0,
    ProjectType.DAM_REMOVAL: 250_000.0,
    ProjectType.AGRICULTURAL_BMP: 8_000.0,
    ProjectType.STORMWATER_MANAGEMENT: 35_000.0,
}

#: IBI reference thresholds (Karr 1981, adapted).
_IBI_THRESHOLDS: Dict[str, Tuple[float, str]] = {
    "excellent": (48.0, "Comparable to best situations without human disturbance"),
    "good": (40.0, "Decreased species richness, some sensitive species present"),
    "fair": (28.0, "Sensitive species rare, tolerant species common"),
    "poor": (16.0, "Few fish present, tolerant species dominate"),
    "very_poor": (0.0, "Few or no fish present"),
}

#: Indicator species and their habitat requirements mapped to state variable
#: thresholds.  Format: {species: {state_var: (min_val, max_val)}}.
_INDICATOR_SPECIES: Dict[str, Dict[str, Tuple[float, float]]] = {
    "brook_trout": {
        "dissolved_oxygen": (7.0, 14.0),
        "water_temperature": (0.0, 20.0),
        "ph": (6.0, 8.5),
        "total_nitrogen": (0.0, 2.0),
        "turbidity": (0.0, 10.0),
    },
    "smallmouth_bass": {
        "dissolved_oxygen": (5.0, 14.0),
        "water_temperature": (10.0, 30.0),
        "ph": (6.5, 9.0),
        "total_nitrogen": (0.0, 5.0),
        "turbidity": (0.0, 25.0),
    },
    "freshwater_mussel": {
        "dissolved_oxygen": (6.0, 14.0),
        "water_temperature": (5.0, 28.0),
        "ph": (6.5, 8.5),
        "total_nitrogen": (0.0, 3.0),
        "turbidity": (0.0, 15.0),
        "sediment": (0.0, 20.0),
    },
    "mayfly_ephemeroptera": {
        "dissolved_oxygen": (6.5, 14.0),
        "water_temperature": (0.0, 25.0),
        "ph": (6.0, 8.5),
        "total_nitrogen": (0.0, 2.5),
    },
    "stonefly_plecoptera": {
        "dissolved_oxygen": (7.0, 14.0),
        "water_temperature": (0.0, 22.0),
        "ph": (6.0, 8.0),
        "turbidity": (0.0, 8.0),
    },
    "darter_percina": {
        "dissolved_oxygen": (5.5, 14.0),
        "water_temperature": (5.0, 27.0),
        "ph": (6.5, 8.5),
        "total_nitrogen": (0.0, 4.0),
        "sediment": (0.0, 25.0),
    },
}


# ---------------------------------------------------------------------------
# RestorationProject dataclass
# ---------------------------------------------------------------------------

@dataclass
class RestorationProject:
    """Specification of a restoration project.

    Attributes:
        project_type: Category of restoration (see :class:`ProjectType`).
        area_hectares: Project area in hectares.
        cost_usd: Total estimated project cost in USD.
        site_embedding: SENTINEL fused embedding for the project site,
            shape ``[256]`` or ``[B, 256]``.
        expected_duration_years: Number of years over which the project
            is expected to deliver ecological benefits.
        description: Optional free-text description.
    """

    project_type: ProjectType
    area_hectares: float
    cost_usd: float
    site_embedding: torch.Tensor
    expected_duration_years: int = 5
    description: str = ""

    def __post_init__(self) -> None:
        self.project_type = ProjectType(self.project_type)
        if self.site_embedding.dim() == 1:
            self.site_embedding = self.site_embedding.unsqueeze(0)
        if self.site_embedding.size(-1) != SENTINEL_EMBEDDING_DIM:
            raise ValueError(
                f"site_embedding must have dim={SENTINEL_EMBEDDING_DIM}, "
                f"got {self.site_embedding.size(-1)}"
            )


# ---------------------------------------------------------------------------
# RestorationOutcome dataclass
# ---------------------------------------------------------------------------

@dataclass
class RestorationOutcome:
    """Predicted ecological outcome of a restoration project.

    Attributes:
        project: The restoration project that was evaluated.
        counterfactual_result: Full counterfactual simulation output.
        predicted_ibi_change: Predicted change in Index of Biotic
            Integrity score (Karr 1981 scale, 0-60).
        ibi_baseline: Estimated baseline IBI score.
        ibi_projected: Projected IBI score after restoration.
        ibi_rating: Qualitative IBI rating (e.g., "good", "fair").
        species_viability: Dict mapping indicator species names to
            their estimated habitat viability probability [0, 1].
        species_viability_baseline: Baseline species viability before
            restoration.
        community_shift: Summary of predicted community composition
            changes.
        cost_effectiveness: Dict with cost-per-unit ecological benefit
            metrics.
        five_year_trajectory: Predicted state trajectory over 5 years,
            shape ``[T, B, D]``.
        confidence_intervals: Optional CI dict from MC sampling.
    """

    project: RestorationProject
    counterfactual_result: CounterfactualOutput
    predicted_ibi_change: float
    ibi_baseline: float
    ibi_projected: float
    ibi_rating: str
    species_viability: Dict[str, float]
    species_viability_baseline: Dict[str, float]
    community_shift: Dict[str, Any]
    cost_effectiveness: Dict[str, float]
    five_year_trajectory: torch.Tensor
    confidence_intervals: Optional[Dict[str, torch.Tensor]] = None


# ---------------------------------------------------------------------------
# RestorationOutcomePredictor
# ---------------------------------------------------------------------------

class RestorationOutcomePredictor:
    """Predicts ecological outcomes of restoration projects.

    Combines the counterfactual engine with ecological assessment
    models to predict IBI score changes, indicator species viability,
    and cost-effectiveness of restoration projects.

    The predictor uses a simplified IBI scoring model based on state
    variable thresholds (Karr 1981) and species-habitat suitability
    matrices.

    Args:
        engine: Reference to the :class:`DigitalTwinEngine`.
        library: Intervention library; uses default if ``None``.
    """

    def __init__(
        self,
        engine: DigitalTwinEngine,
        library: Optional[InterventionLibrary] = None,
    ) -> None:
        self.engine = engine
        self.simulator = CounterfactualSimulator(engine, library)

    def _project_to_interventions(
        self,
        project: RestorationProject,
    ) -> List[InterventionSpec]:
        """Convert a restoration project to intervention specifications.

        Scales intervention magnitude by project area using empirical
        area-effectiveness curves.

        Args:
            project: The restoration project.

        Returns:
            List of :class:`InterventionSpec` objects.
        """
        templates = _PROJECT_TO_INTERVENTIONS.get(project.project_type, [])
        if not templates:
            warnings.warn(
                f"No intervention template for project type "
                f"'{project.project_type.value}'."
            )
            return []

        specs: List[InterventionSpec] = []
        for itype, target, base_magnitude in templates:
            # Scale magnitude by area (diminishing returns via log)
            # A 1-hectare project gets base_magnitude; 100 ha gets ~2x base
            area_scale = min(
                1.0 + 0.2 * math.log1p(project.area_hectares), 2.0
            )
            scaled_mag = min(base_magnitude * area_scale, 0.95)

            specs.append(InterventionSpec(
                intervention_type=itype,
                magnitude=scaled_mag,
                target_parameter=target,
                spatial_scope=SpatialScope.LOCAL,
                implementation_time_days=int(
                    project.expected_duration_years * 365 * 0.1
                ),
                description=(
                    f"{project.project_type.value}: {target} intervention "
                    f"at {scaled_mag:.0%} magnitude over "
                    f"{project.area_hectares:.1f} ha"
                ),
            ))

        return specs

    def _estimate_ibi(
        self,
        state: torch.Tensor,
    ) -> float:
        """Estimate an Index of Biotic Integrity score from state variables.

        Uses a simplified scoring model based on Karr (1981) with state
        variable thresholds.  The IBI is scored on a 0-60 scale across
        six metrics:

        1. Dissolved oxygen adequacy (0-10)
        2. Nutrient stress index (0-10)
        3. Algal bloom risk (0-10)
        4. Thermal suitability (0-10)
        5. Habitat quality (turbidity + sediment) (0-10)
        6. Overall water chemistry (pH + DOC) (0-10)

        Args:
            state: State vector ``[B, D]`` or ``[D]``.  If batched,
                the batch mean is used.

        Returns:
            Estimated IBI score on the 0-60 scale.
        """
        if state.dim() > 1:
            state = state.mean(dim=0)
        if state.dim() > 1:
            state = state.mean(dim=0)

        s = state.detach().cpu()

        # Unpack state variables
        do = s[0].item()
        bod = s[1].item()
        tn = s[2].item()
        tp = s[3].item()
        chla = s[4].item()
        temp = s[5].item()
        ph = s[6].item()
        turb = s[7].item()
        doc = s[8].item()
        sed = s[9].item()

        scores: List[float] = []

        # 1. DO adequacy: 10 at DO >= 8 mg/L, 0 at DO <= 2 mg/L
        do_score = max(0.0, min(10.0, (do - 2.0) / 6.0 * 10.0))
        scores.append(do_score)

        # 2. Nutrient stress: 10 at TN <= 0.5, 0 at TN >= 5.0
        n_score = max(0.0, min(10.0, (5.0 - tn) / 4.5 * 10.0))
        p_score = max(0.0, min(10.0, (0.1 - tp) / 0.09 * 10.0))
        nutrient_score = (n_score + p_score) / 2.0
        scores.append(nutrient_score)

        # 3. Algal bloom risk: 10 at chla <= 5 ug/L, 0 at chla >= 50
        algal_score = max(0.0, min(10.0, (50.0 - chla) / 45.0 * 10.0))
        scores.append(algal_score)

        # 4. Thermal suitability: 10 at 10-20 degC, decreasing outside
        if 10.0 <= temp <= 20.0:
            thermal_score = 10.0
        elif temp < 10.0:
            thermal_score = max(0.0, 10.0 - (10.0 - temp) * 0.5)
        else:
            thermal_score = max(0.0, 10.0 - (temp - 20.0) * 0.5)
        scores.append(thermal_score)

        # 5. Habitat quality: based on turbidity and sediment
        turb_score = max(0.0, min(10.0, (25.0 - turb) / 25.0 * 10.0))
        sed_score = max(0.0, min(10.0, (30.0 - sed) / 30.0 * 10.0))
        habitat_score = (turb_score + sed_score) / 2.0
        scores.append(habitat_score)

        # 6. Water chemistry: pH near neutral, reasonable DOC
        ph_score = max(0.0, min(10.0, 10.0 - abs(ph - 7.5) * 3.0))
        doc_score = max(0.0, min(10.0, (15.0 - doc) / 15.0 * 10.0))
        chem_score = (ph_score + doc_score) / 2.0
        scores.append(chem_score)

        return sum(scores)

    def _ibi_to_rating(self, ibi: float) -> str:
        """Convert an IBI score to a qualitative rating."""
        for rating, (threshold, _) in sorted(
            _IBI_THRESHOLDS.items(),
            key=lambda x: x[1][0],
            reverse=True,
        ):
            if ibi >= threshold:
                return rating
        return "very_poor"

    def _estimate_species_viability(
        self,
        state: torch.Tensor,
    ) -> Dict[str, float]:
        """Estimate habitat viability for indicator species.

        For each indicator species, computes the probability that the
        predicted water quality conditions fall within the species'
        tolerance ranges.  Uses a smooth sigmoid approximation to
        model threshold transitions.

        Args:
            state: State vector ``[B, D]`` or ``[D]``.

        Returns:
            Dict mapping species name to viability probability [0, 1].
        """
        if state.dim() > 1:
            state = state.mean(dim=0)
        if state.dim() > 1:
            state = state.mean(dim=0)

        s = state.detach().cpu()
        viability: Dict[str, float] = {}

        for species, requirements in _INDICATOR_SPECIES.items():
            # Compute suitability for each required variable
            suitabilities: List[float] = []

            for var_name, (lo, hi) in requirements.items():
                if var_name not in STATE_VARS:
                    continue
                idx = STATE_VARS.index(var_name)
                val = s[idx].item()

                # Smooth sigmoid boundaries with 10% buffer zones
                range_width = hi - lo
                buffer = range_width * 0.1

                # Lower boundary suitability
                lower_suit = 1.0 / (
                    1.0 + math.exp(-5.0 * (val - lo) / max(buffer, 0.01))
                )
                # Upper boundary suitability
                upper_suit = 1.0 / (
                    1.0 + math.exp(5.0 * (val - hi) / max(buffer, 0.01))
                )

                suitabilities.append(lower_suit * upper_suit)

            # Overall viability is the geometric mean of individual
            # suitabilities (all conditions must be met)
            if suitabilities:
                geo_mean = math.exp(
                    sum(math.log(max(s_val, 1e-10)) for s_val in suitabilities)
                    / len(suitabilities)
                )
                viability[species] = round(min(geo_mean, 1.0), 4)
            else:
                viability[species] = 0.0

        return viability

    def _compute_cost_effectiveness(
        self,
        project: RestorationProject,
        ibi_change: float,
        counterfactual: CounterfactualOutput,
    ) -> Dict[str, float]:
        """Compute cost-effectiveness metrics.

        Args:
            project: The restoration project.
            ibi_change: Predicted IBI change.
            counterfactual: Counterfactual simulation result.

        Returns:
            Dict of cost-effectiveness metrics.
        """
        cost = project.cost_usd
        area = project.area_hectares

        metrics: Dict[str, float] = {}

        # Cost per IBI point improvement
        if abs(ibi_change) > 0.01:
            metrics["cost_per_ibi_point"] = round(cost / abs(ibi_change), 2)
        else:
            metrics["cost_per_ibi_point"] = float("inf")

        # Cost per hectare
        if area > 0:
            metrics["cost_per_hectare"] = round(cost / area, 2)
        else:
            metrics["cost_per_hectare"] = float("inf")

        # Cost per kg nitrogen removed (estimated from trajectory delta)
        impact = counterfactual.ecological_impact_summary
        if "_aggregate" in impact:
            n_reduction = impact["_aggregate"].get(
                "nitrogen_reduction_mg_l", 0.0
            )
            if abs(n_reduction) > 1e-6:
                # Rough estimate: mg/L reduction * stream volume proxy
                # Using area as volume proxy (m^3 ~ ha * 10,000 * depth)
                est_volume_m3 = area * 10_000 * 1.0  # 1m avg depth
                kg_n_removed = n_reduction * est_volume_m3 / 1_000_000
                if abs(kg_n_removed) > 1e-6:
                    metrics["cost_per_kg_nitrogen"] = round(
                        cost / abs(kg_n_removed), 2
                    )

            do_improvement = impact["_aggregate"].get(
                "do_improvement_mg_l", 0.0
            )
            if abs(do_improvement) > 1e-6:
                metrics["cost_per_mg_l_do_improvement"] = round(
                    cost / abs(do_improvement), 2
                )

        # Benefit-cost ratio (simplified: IBI points * area / cost * 1000)
        if cost > 0:
            benefit_index = abs(ibi_change) * area
            metrics["benefit_cost_ratio"] = round(
                benefit_index / (cost / 1000.0), 6
            )
        else:
            metrics["benefit_cost_ratio"] = float("inf")

        # Annual cost (amortized over project duration)
        if project.expected_duration_years > 0:
            metrics["annual_cost"] = round(
                cost / project.expected_duration_years, 2
            )

        return metrics

    def predict(
        self,
        project: RestorationProject,
        n_mc_samples: int = 0,
    ) -> RestorationOutcome:
        """Predict the ecological outcome of a restoration project.

        Args:
            project: The restoration project to evaluate.
            n_mc_samples: Number of MC-dropout samples for confidence
                intervals.  If 0, a single deterministic simulation is
                used.

        Returns:
            :class:`RestorationOutcome` with predicted IBI change,
            species viability, cost-effectiveness, and full trajectory.
        """
        # Convert project to interventions
        interventions = self._project_to_interventions(project)
        if not interventions:
            raise ValueError(
                f"Could not generate interventions for project type "
                f"'{project.project_type.value}'."
            )

        # Use the standard forecast horizons supported by the engine.
        # The ForecastHead has a fixed horizon embedding table that only
        # accommodates FORECAST_HORIZONS entries, so we select from those.
        # For multi-year projections we run iterative 365-day simulations.
        horizons = FORECAST_HORIZONS

        # Run counterfactual simulation
        cf_result = self.simulator.simulate(
            embedding=project.site_embedding,
            interventions=interventions,
            horizons=horizons,
            n_mc_samples=n_mc_samples,
        )

        # Estimate baseline IBI from initial (t=0) baseline state
        baseline_state = cf_result.baseline_trajectory[0]  # [B, D]
        ibi_baseline = self._estimate_ibi(baseline_state)

        # Estimate projected IBI from final intervention state
        final_state = cf_result.intervention_trajectory[-1]  # [B, D]
        ibi_projected = self._estimate_ibi(final_state)
        ibi_change = ibi_projected - ibi_baseline
        ibi_rating = self._ibi_to_rating(ibi_projected)

        # Species viability
        viability_baseline = self._estimate_species_viability(baseline_state)
        viability_projected = self._estimate_species_viability(final_state)

        # Community shift summary
        community_shift = self._compute_community_shift(
            viability_baseline, viability_projected
        )

        # Cost-effectiveness
        cost_eff = self._compute_cost_effectiveness(
            project, ibi_change, cf_result
        )

        return RestorationOutcome(
            project=project,
            counterfactual_result=cf_result,
            predicted_ibi_change=round(ibi_change, 2),
            ibi_baseline=round(ibi_baseline, 2),
            ibi_projected=round(ibi_projected, 2),
            ibi_rating=ibi_rating,
            species_viability=viability_projected,
            species_viability_baseline=viability_baseline,
            community_shift=community_shift,
            cost_effectiveness=cost_eff,
            five_year_trajectory=cf_result.intervention_trajectory,
            confidence_intervals=cf_result.confidence_intervals,
        )

    def _compute_community_shift(
        self,
        baseline_viability: Dict[str, float],
        projected_viability: Dict[str, float],
    ) -> Dict[str, Any]:
        """Compute community composition shift summary.

        Args:
            baseline_viability: Species viability before restoration.
            projected_viability: Species viability after restoration.

        Returns:
            Dict summarizing community changes.
        """
        shift: Dict[str, Any] = {}

        # Per-species change
        species_changes: Dict[str, float] = {}
        newly_viable: List[str] = []
        lost_species: List[str] = []

        viable_threshold = 0.5

        for species in projected_viability:
            base = baseline_viability.get(species, 0.0)
            proj = projected_viability[species]
            species_changes[species] = round(proj - base, 4)

            if base < viable_threshold <= proj:
                newly_viable.append(species)
            elif proj < viable_threshold <= base:
                lost_species.append(species)

        shift["species_changes"] = species_changes
        shift["newly_viable_species"] = newly_viable
        shift["lost_species"] = lost_species
        shift["net_species_gain"] = len(newly_viable) - len(lost_species)

        # Biodiversity index change (Shannon-like from viabilities)
        def _shannon_proxy(viabilities: Dict[str, float]) -> float:
            probs = [v for v in viabilities.values() if v > 0.01]
            if not probs:
                return 0.0
            total = sum(probs)
            return -sum(
                (p / total) * math.log(p / total) for p in probs
            )

        shift["biodiversity_index_baseline"] = round(
            _shannon_proxy(baseline_viability), 4
        )
        shift["biodiversity_index_projected"] = round(
            _shannon_proxy(projected_viability), 4
        )

        return shift

    def compare_projects(
        self,
        projects: Sequence[RestorationProject],
        n_mc_samples: int = 0,
    ) -> List[RestorationOutcome]:
        """Compare multiple restoration projects.

        Args:
            projects: Restoration projects to evaluate.
            n_mc_samples: MC-dropout samples for CIs.

        Returns:
            List of :class:`RestorationOutcome`, one per project,
            sorted by cost-effectiveness (benefit-cost ratio, descending).
        """
        outcomes = [
            self.predict(project, n_mc_samples=n_mc_samples)
            for project in projects
        ]
        outcomes.sort(
            key=lambda o: o.cost_effectiveness.get("benefit_cost_ratio", 0.0),
            reverse=True,
        )
        return outcomes


# ---------------------------------------------------------------------------
# CostOptimizer
# ---------------------------------------------------------------------------

class CostOptimizer:
    """Finds minimum-cost intervention packages for target ecological outcomes.

    Uses gradient-based optimization through the differentiable twin to
    search over the continuous intervention magnitude space.  The optimizer
    finds the Pareto frontier of cost vs. ecological benefit.

    The key insight is that the Digital Twin is fully differentiable:
    we can backpropagate from the ecological outcome (e.g., DO at
    day 365) through the ODE integration and parameter modulation
    back to the intervention magnitudes, enabling efficient gradient-
    based search.

    Args:
        engine: Reference to the :class:`DigitalTwinEngine`.
        library: Intervention library; uses default if ``None``.
    """

    def __init__(
        self,
        engine: DigitalTwinEngine,
        library: Optional[InterventionLibrary] = None,
    ) -> None:
        self.engine = engine
        self.library = library or InterventionLibrary()
        self.simulator = CounterfactualSimulator(engine, library)

    def _compute_intervention_cost(
        self,
        magnitudes: torch.Tensor,
        project_types: Sequence[ProjectType],
        base_area_hectares: float = 10.0,
    ) -> torch.Tensor:
        """Estimate total intervention cost from magnitudes.

        Uses per-hectare cost coefficients scaled by magnitude.
        Cost scales super-linearly with magnitude (quadratic) to
        model diminishing returns.

        Args:
            magnitudes: Intervention magnitudes ``[N]``.
            project_types: Project type for each intervention.
            base_area_hectares: Reference area for cost estimation.

        Returns:
            Scalar total cost tensor.
        """
        total = torch.zeros(1, device=magnitudes.device, dtype=magnitudes.dtype)
        for i, ptype in enumerate(project_types):
            coeff = _AREA_COST_COEFFICIENTS.get(ptype, 20_000.0)
            # Quadratic cost scaling: cost = coeff * area * magnitude^2
            total = total + coeff * base_area_hectares * magnitudes[i] ** 2
        return total.squeeze()

    def optimize(
        self,
        embedding: torch.Tensor,
        target_variable: str,
        target_value: float,
        available_interventions: Optional[
            Sequence[Tuple[InterventionType, str]]
        ] = None,
        max_iterations: int = 100,
        learning_rate: float = 0.01,
        area_hectares: float = 10.0,
        penalty_weight: float = 10.0,
    ) -> Dict[str, Any]:
        """Find minimum-cost intervention package for a target outcome.

        Optimizes intervention magnitudes to minimize cost subject to
        achieving the target value for the specified state variable.
        Uses a Lagrangian relaxation approach.

        Args:
            embedding: SENTINEL fused embedding ``[B, 256]``.
            target_variable: State variable to target (e.g.,
                ``"dissolved_oxygen"``).
            target_value: Target value for the state variable at the
                end of the simulation.
            available_interventions: List of
                ``(InterventionType, target_parameter)`` pairs to
                consider.  If ``None``, uses all available templates.
            max_iterations: Maximum optimization iterations.
            learning_rate: Adam learning rate.
            area_hectares: Reference area for cost estimation.
            penalty_weight: Weight for the target constraint penalty.

        Returns:
            Dict with keys:
                ``"optimal_magnitudes"``: Optimized intervention
                    magnitudes.
                ``"optimal_cost"``: Estimated total cost.
                ``"achieved_value"``: Predicted value of the target
                    variable.
                ``"target_met"``: Whether the target was achieved.
                ``"interventions"``: List of intervention descriptions.
                ``"optimization_history"``: List of dicts tracking
                    cost and constraint violation per iteration.
        """
        if target_variable not in STATE_VARS:
            raise ValueError(
                f"Unknown target variable '{target_variable}'. "
                f"Must be one of {STATE_VARS}."
            )
        var_idx = STATE_VARS.index(target_variable)

        # Set up available interventions
        if available_interventions is None:
            available_interventions = [
                (InterventionType.NUTRIENT_REDUCTION, "total_nitrogen"),
                (InterventionType.NUTRIENT_REDUCTION, "total_phosphorus"),
                (InterventionType.RIPARIAN_BUFFER, "total_nitrogen"),
                (InterventionType.WETLAND_RESTORATION, "total_nitrogen"),
                (InterventionType.POINT_SOURCE_CONTROL, "total_nitrogen"),
            ]

        n_interventions = len(available_interventions)

        # Map intervention types to project types for costing
        _itype_to_ptype = {
            InterventionType.NUTRIENT_REDUCTION: ProjectType.AGRICULTURAL_BMP,
            InterventionType.DAM_REMOVAL: ProjectType.DAM_REMOVAL,
            InterventionType.RIPARIAN_BUFFER: ProjectType.RIPARIAN_BUFFER,
            InterventionType.WETLAND_RESTORATION: ProjectType.WETLAND_CONSTRUCTION,
            InterventionType.POINT_SOURCE_CONTROL: ProjectType.STORMWATER_MANAGEMENT,
        }
        project_types = [
            _itype_to_ptype.get(it, ProjectType.AGRICULTURAL_BMP)
            for it, _ in available_interventions
        ]

        # Learnable magnitudes (in logit space for bounded [0, 1])
        logit_magnitudes = nn.Parameter(
            torch.zeros(n_interventions, device=embedding.device)
        )
        optimizer = torch.optim.Adam([logit_magnitudes], lr=learning_rate)

        # Get baseline prediction for reference
        with torch.no_grad():
            baseline_out = self.engine(embedding, horizons=(365,))
            baseline_value = baseline_out.predictions[0, :, var_idx].mean()

        history: List[Dict[str, float]] = []

        for iteration in range(max_iterations):
            optimizer.zero_grad()

            # Convert logits to [0, 0.95] magnitudes
            magnitudes = torch.sigmoid(logit_magnitudes) * 0.95

            # Build intervention specs and compose modifiers
            specs: List[InterventionSpec] = []
            for i, (itype, target) in enumerate(available_interventions):
                specs.append(InterventionSpec(
                    intervention_type=itype,
                    magnitude=magnitudes[i].item(),
                    target_parameter=target,
                ))

            composed_ode_mods, composed_state_mods = (
                self.simulator._compose_modifiers(specs)
            )

            # Apply modifiers and run forward pass
            # We need gradients, so we use the engine directly
            state_mean, state_log_var, param_scale, param_shift = (
                self.engine.assimilator(embedding)
            )
            y0 = state_mean.clone()

            # Apply state modifiers
            for idx, mult in composed_state_mods.items():
                if idx < y0.size(-1):
                    y0[..., idx] = y0[..., idx] * mult

            # Apply ODE modifiers
            saved = self.simulator._apply_ode_modifiers(
                self.engine.ode, composed_ode_mods
            )

            try:
                t_eval, h_idx = self.engine._build_eval_times(
                    (365,), embedding.device, embedding.dtype
                )
                # Apply parameter modulation from assimilator
                saved_modulation = self.engine._apply_param_modulation(
                    param_scale, param_shift
                )

                try:
                    from sentinel.models.twin.twin_engine import odeint as _odeint
                    traj = _odeint(
                        self.engine.ode, y0, t_eval,
                        method=self.engine.ode_method,
                        rtol=self.engine.ode_rtol,
                        atol=self.engine.ode_atol,
                    )
                finally:
                    self.engine._restore_ode_params(saved_modulation)
            finally:
                self.simulator._restore_ode_params(self.engine.ode, saved)

            # Corrected trajectory
            corrections = self.engine.corrector(traj, embedding)
            corrected = traj + corrections

            # Extract target variable at final time step
            achieved = corrected[-1, :, var_idx].mean()

            # Compute cost
            cost = self._compute_intervention_cost(
                magnitudes, project_types, area_hectares
            )

            # Loss: cost + penalty for missing target
            # For variables we want to increase (DO), penalize if below target
            # For variables we want to decrease (nutrients), penalize if above
            constraint_violation = F.relu(target_value - achieved)
            loss = cost / 1e6 + penalty_weight * constraint_violation ** 2

            loss.backward()
            optimizer.step()

            history.append({
                "iteration": iteration,
                "cost": cost.item(),
                "achieved_value": achieved.item(),
                "constraint_violation": constraint_violation.item(),
                "loss": loss.item(),
            })

            # Early stopping if target met and cost stable
            if (iteration > 10
                    and constraint_violation.item() < 0.01
                    and len(history) > 5
                    and abs(history[-1]["cost"] - history[-5]["cost"])
                    / max(history[-1]["cost"], 1.0) < 0.001):
                logger.info(
                    "CostOptimizer converged at iteration %d.", iteration
                )
                break

        # Extract results
        with torch.no_grad():
            final_magnitudes = torch.sigmoid(logit_magnitudes) * 0.95
            final_cost = self._compute_intervention_cost(
                final_magnitudes, project_types, area_hectares
            )

        intervention_descriptions = []
        for i, (itype, target) in enumerate(available_interventions):
            intervention_descriptions.append({
                "type": itype.value,
                "target": target,
                "magnitude": round(final_magnitudes[i].item(), 4),
            })

        return {
            "optimal_magnitudes": final_magnitudes.detach().cpu().tolist(),
            "optimal_cost": round(final_cost.item(), 2),
            "achieved_value": round(history[-1]["achieved_value"], 4),
            "target_met": history[-1]["constraint_violation"] < 0.1,
            "interventions": intervention_descriptions,
            "optimization_history": history,
        }

    def pareto_frontier(
        self,
        embedding: torch.Tensor,
        target_variable: str,
        target_values: Sequence[float],
        available_interventions: Optional[
            Sequence[Tuple[InterventionType, str]]
        ] = None,
        max_iterations: int = 80,
        area_hectares: float = 10.0,
    ) -> List[Dict[str, Any]]:
        """Compute the Pareto frontier of cost vs. ecological benefit.

        Runs the optimizer at multiple target levels to trace out the
        efficient frontier showing the minimum cost to achieve each
        level of ecological improvement.

        Args:
            embedding: SENTINEL fused embedding ``[B, 256]``.
            target_variable: State variable to target.
            target_values: Sequence of target values to optimize for,
                each representing a different ambition level.
            available_interventions: Interventions to consider.
            max_iterations: Max iterations per target.
            area_hectares: Reference area.

        Returns:
            List of optimization results (one per target value),
            sorted by cost.
        """
        frontier: List[Dict[str, Any]] = []

        for target in sorted(target_values):
            result = self.optimize(
                embedding=embedding,
                target_variable=target_variable,
                target_value=target,
                available_interventions=available_interventions,
                max_iterations=max_iterations,
                area_hectares=area_hectares,
            )
            result["target_value"] = target
            frontier.append(result)

        # Sort by cost
        frontier.sort(key=lambda x: x["optimal_cost"])
        return frontier


# ---------------------------------------------------------------------------
# BioremediationRecommender
# ---------------------------------------------------------------------------

class BioremediationRecommender:
    """Recommends bioremediation strategies based on contamination context.

    When contamination is detected, suggests remediation approaches
    based on:

    * Detected microbial community composition (from MicroBiomeNet
      embeddings or source attribution).
    * Pollutant class (from source attribution classification).
    * Site conditions (from SENTINEL embedding / digital twin state).

    Recommends between:

    * **Biostimulation**: Enhancing activity of indigenous
      microorganisms by adding nutrients or electron acceptors.
    * **Bioaugmentation**: Introducing specialized microbial
      cultures to accelerate degradation.

    The recommender is rule-based with empirical decision boundaries
    from the bioremediation literature (Vidali 2001; Adams et al.
    2015).

    References
    ----------
    * Vidali, M. (2001). Bioremediation: An overview. Pure Appl.
      Chem. 73(7):1163-1172.
    * Adams, G.O. et al. (2015). Bioremediation, biostimulation and
      bioaugmentation: A review. Int. J. Environ. Bioremediation
      3(1):28-39.
    """

    #: Pollutant classes and their amenability to bioremediation.
    _POLLUTANT_REMEDIATION: Dict[str, Dict[str, Any]] = {
        "nutrient": {
            "primary_strategy": "biostimulation",
            "mechanism": (
                "Enhance denitrifying bacteria activity via carbon "
                "source addition (e.g., acetate, methanol)."
            ),
            "key_organisms": [
                "Pseudomonas", "Paracoccus", "Thiobacillus",
            ],
            "timeline_months": (3, 12),
            "success_rate": 0.85,
            "cost_per_hectare": 5_000.0,
        },
        "heavy_metals": {
            "primary_strategy": "bioaugmentation",
            "mechanism": (
                "Introduce metal-tolerant, biosorption-capable strains. "
                "Sulfate-reducing bacteria precipitate metals as sulfides."
            ),
            "key_organisms": [
                "Desulfovibrio", "Geobacter", "Shewanella",
            ],
            "timeline_months": (6, 24),
            "success_rate": 0.60,
            "cost_per_hectare": 15_000.0,
        },
        "pharmaceutical": {
            "primary_strategy": "bioaugmentation",
            "mechanism": (
                "Inoculate with specialized degraders targeting specific "
                "pharmaceutical compounds (e.g., ibuprofen, estrogens)."
            ),
            "key_organisms": [
                "Sphingomonas", "Rhodococcus", "Trametes versicolor",
            ],
            "timeline_months": (6, 18),
            "success_rate": 0.55,
            "cost_per_hectare": 20_000.0,
        },
        "oil_petrochemical": {
            "primary_strategy": "biostimulation",
            "mechanism": (
                "Stimulate hydrocarbon-degrading bacteria by adding "
                "nitrogen and phosphorus fertilizers (Oleophilic "
                "fertilizer approach)."
            ),
            "key_organisms": [
                "Alcanivorax", "Marinobacter", "Pseudomonas",
            ],
            "timeline_months": (3, 18),
            "success_rate": 0.75,
            "cost_per_hectare": 8_000.0,
        },
        "sewage": {
            "primary_strategy": "biostimulation",
            "mechanism": (
                "Aerate to promote aerobic degradation of organic matter "
                "and nitrification. Minimal inoculation needed."
            ),
            "key_organisms": [
                "Nitrosomonas", "Nitrobacter", "Zoogloea",
            ],
            "timeline_months": (1, 6),
            "success_rate": 0.90,
            "cost_per_hectare": 3_000.0,
        },
        "acid_mine": {
            "primary_strategy": "bioaugmentation",
            "mechanism": (
                "Construct passive bioreactors with sulfate-reducing "
                "bacteria to raise pH and precipitate dissolved metals."
            ),
            "key_organisms": [
                "Desulfovibrio", "Desulfosporosinus",
                "Acidithiobacillus (for iron oxidation)",
            ],
            "timeline_months": (12, 36),
            "success_rate": 0.50,
            "cost_per_hectare": 25_000.0,
        },
        "thermal": {
            "primary_strategy": "biostimulation",
            "mechanism": (
                "Riparian revegetation for canopy shading. Not a direct "
                "bioremediation target but ecosystem-based adaptation."
            ),
            "key_organisms": [],
            "timeline_months": (12, 60),
            "success_rate": 0.70,
            "cost_per_hectare": 12_000.0,
        },
        "sediment": {
            "primary_strategy": "biostimulation",
            "mechanism": (
                "Bioengineered erosion control using vegetation "
                "establishment and mycorrhizal fungi for soil binding."
            ),
            "key_organisms": [
                "Glomus (mycorrhizal)", "Rhizobium",
            ],
            "timeline_months": (6, 24),
            "success_rate": 0.65,
            "cost_per_hectare": 10_000.0,
        },
    }

    def recommend(
        self,
        pollutant_class: str,
        site_state: Optional[torch.Tensor] = None,
        microbial_embedding: Optional[torch.Tensor] = None,
        area_hectares: float = 1.0,
    ) -> Dict[str, Any]:
        """Generate a bioremediation recommendation.

        Args:
            pollutant_class: Detected pollutant class (must match one
                of the keys in ``_POLLUTANT_REMEDIATION``).
            site_state: Optional current site state vector ``[D]`` or
                ``[B, D]`` for context-aware recommendations.
            microbial_embedding: Optional microbial community embedding
                ``[256]`` from MicroBiomeNet.  If provided, the
                recommender checks whether key degrader organisms are
                likely present (via embedding similarity heuristic)
                to decide between biostimulation and bioaugmentation.
            area_hectares: Project area for cost estimation.

        Returns:
            Dict with recommendation details including:
                ``"strategy"``: ``"biostimulation"`` or
                    ``"bioaugmentation"``.
                ``"mechanism"``: Description of the remediation
                    mechanism.
                ``"key_organisms"``: List of key microbial taxa.
                ``"estimated_timeline_months"``: ``(min, max)`` tuple.
                ``"estimated_cost_usd"``: Total estimated cost.
                ``"success_probability"``: Literature-based success
                    rate [0, 1].
                ``"confidence"``: Confidence in the recommendation
                    (``"high"``, ``"medium"``, ``"low"``).
                ``"site_suitability_notes"``: Context from site
                    conditions.
        """
        pollutant_class = pollutant_class.lower().strip()

        if pollutant_class not in self._POLLUTANT_REMEDIATION:
            available = list(self._POLLUTANT_REMEDIATION.keys())
            raise ValueError(
                f"Unknown pollutant class '{pollutant_class}'. "
                f"Available: {available}"
            )

        info = self._POLLUTANT_REMEDIATION[pollutant_class]
        strategy = info["primary_strategy"]
        confidence = "medium"

        # Adjust strategy based on microbial embedding if available
        notes: List[str] = []
        if microbial_embedding is not None:
            # Heuristic: high-norm embeddings suggest diverse communities
            # that are more likely to already contain degraders
            if microbial_embedding.dim() > 1:
                microbial_embedding = microbial_embedding.mean(dim=0)
            emb_norm = microbial_embedding.norm().item()

            if emb_norm > 15.0:
                # Rich community detected -- biostimulation more likely
                # to work even if default is bioaugmentation
                if strategy == "bioaugmentation":
                    notes.append(
                        "Rich microbial community detected (embedding "
                        f"norm={emb_norm:.1f}). Indigenous degraders may "
                        "be present; consider biostimulation first."
                    )
                    # Don't override but flag as alternative
                confidence = "medium"
            elif emb_norm < 5.0:
                # Depauperate community -- bioaugmentation more likely needed
                if strategy == "biostimulation":
                    notes.append(
                        "Low microbial diversity detected (embedding "
                        f"norm={emb_norm:.1f}). Bioaugmentation may be "
                        "needed to supplement indigenous community."
                    )
                confidence = "low"
            else:
                confidence = "high"

        # Adjust based on site state if available
        if site_state is not None:
            if site_state.dim() > 1:
                site_state = site_state.mean(dim=0)
            s = site_state.detach().cpu()

            do = s[0].item() if s.size(0) > 0 else 0.0
            temp = s[5].item() if s.size(0) > 5 else 15.0
            ph = s[6].item() if s.size(0) > 6 else 7.0

            # Aerobic vs anaerobic conditions affect strategy
            if do < 2.0:
                notes.append(
                    f"Low dissolved oxygen ({do:.1f} mg/L) detected. "
                    "Anaerobic bioremediation pathways favored."
                )
                if pollutant_class in ("nutrient", "heavy_metals"):
                    notes.append(
                        "Consider sulfate-reducing bioremediation under "
                        "anaerobic conditions."
                    )
            elif do > 6.0:
                notes.append(
                    f"Well-oxygenated conditions ({do:.1f} mg/L). "
                    "Aerobic degradation pathways available."
                )

            # Temperature affects microbial activity
            if temp < 5.0:
                notes.append(
                    f"Low temperature ({temp:.1f} degC) will slow "
                    "microbial activity. Extend timeline estimate by 2x."
                )
            elif temp > 30.0:
                notes.append(
                    f"High temperature ({temp:.1f} degC) may stress "
                    "mesophilic degraders. Monitor for community shifts."
                )

            # pH affects many bioremediation processes
            if ph < 5.5:
                notes.append(
                    f"Acidic conditions (pH={ph:.1f}). Lime addition may "
                    "be needed to support microbial activity."
                )
            elif ph > 9.0:
                notes.append(
                    f"Alkaline conditions (pH={ph:.1f}). May limit "
                    "effectiveness of some degradation pathways."
                )

        timeline = info["timeline_months"]
        cost = info["cost_per_hectare"] * area_hectares

        return {
            "strategy": strategy,
            "mechanism": info["mechanism"],
            "key_organisms": info["key_organisms"],
            "estimated_timeline_months": timeline,
            "estimated_cost_usd": round(cost, 2),
            "success_probability": info["success_rate"],
            "confidence": confidence,
            "site_suitability_notes": notes,
            "pollutant_class": pollutant_class,
            "area_hectares": area_hectares,
        }

    def compare_strategies(
        self,
        pollutant_class: str,
        site_state: Optional[torch.Tensor] = None,
        area_hectares: float = 1.0,
    ) -> Dict[str, Dict[str, Any]]:
        """Compare biostimulation vs bioaugmentation for a given scenario.

        Returns side-by-side recommendations for both strategies.

        Args:
            pollutant_class: Detected pollutant class.
            site_state: Optional site state vector.
            area_hectares: Project area.

        Returns:
            Dict with ``"biostimulation"`` and ``"bioaugmentation"``
            keys, each containing a recommendation dict.
        """
        base_rec = self.recommend(
            pollutant_class, site_state=site_state,
            area_hectares=area_hectares,
        )

        # Generate alternative strategy recommendation
        alt_strategy = (
            "bioaugmentation"
            if base_rec["strategy"] == "biostimulation"
            else "biostimulation"
        )

        alt_rec = dict(base_rec)
        alt_rec["strategy"] = alt_strategy

        if alt_strategy == "bioaugmentation":
            alt_rec["estimated_cost_usd"] = round(
                base_rec["estimated_cost_usd"] * 1.5, 2
            )
            alt_rec["success_probability"] = round(
                min(base_rec["success_probability"] + 0.10, 0.95), 2
            )
            alt_rec["mechanism"] = (
                f"Bioaugmentation alternative: Introduce specialized "
                f"microbial cultures to supplement indigenous community. "
                f"Higher cost but potentially faster degradation."
            )
        else:
            alt_rec["estimated_cost_usd"] = round(
                base_rec["estimated_cost_usd"] * 0.6, 2
            )
            alt_rec["success_probability"] = round(
                max(base_rec["success_probability"] - 0.10, 0.20), 2
            )
            alt_rec["mechanism"] = (
                f"Biostimulation alternative: Enhance indigenous "
                f"microbial activity with nutrient/electron acceptor "
                f"amendments. Lower cost but requires suitable native "
                f"community."
            )

        return {
            base_rec["strategy"]: base_rec,
            alt_rec["strategy"]: alt_rec,
        }
