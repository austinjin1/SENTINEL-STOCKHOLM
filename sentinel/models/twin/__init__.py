"""Digital Aquatic Ecosystem Twin.

Hybrid neural-ODE + transformer architecture for multi-horizon ecosystem
forecasting.  The twin couples a physics-informed biogeochemical ODE with
learned neural corrections, using SENTINEL embeddings for data assimilation
(initial conditions and parameter posteriors).

Modules
-------
DigitalTwinEngine
    Top-level orchestrator that integrates physics and neural components.
BiogeochemicalODE
    Differentiable ODE describing coupled biogeochemical processes.
NeuralCorrector
    Residual network that learns the gap between physics and reality.
DataAssimilator
    Maps SENTINEL embeddings to ODE state vectors and parameter posteriors.
CounterfactualEngine
    Evaluates "what-if" intervention scenarios via paired simulations.
ForecastHead
    Multi-horizon prediction head with per-variable uncertainty.
CounterfactualSimulator
    Advanced counterfactual engine with intervention library, composed
    interventions, and MC-dropout confidence intervals.
InterventionSpec / InterventionLibrary
    Structured intervention definitions with empirically validated
    parameter modifications from peer-reviewed literature.
RestorationOutcomePredictor
    Predicts IBI score changes, species viability, and cost-effectiveness
    of restoration projects.
CostOptimizer
    Gradient-based optimization through the differentiable twin to find
    minimum-cost intervention packages.
BioremediationRecommender
    Suggests biostimulation vs bioaugmentation strategies based on
    contamination context.
RecoveryPlanner
    Endangered species recovery planner using counterfactual twin
    simulations to identify optimal habitat interventions.
"""

from sentinel.models.twin.twin_engine import (
    STATE_VARS,
    NUM_STATE_VARS,
    FORECAST_HORIZONS,
    BiogeochemicalODE,
    CounterfactualEngine,
    DataAssimilator,
    DigitalTwinEngine,
    ForecastHead,
    NeuralCorrector,
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

from sentinel.models.twin.restoration import (
    BioremediationRecommender,
    CostOptimizer,
    ProjectType,
    RestorationOutcome,
    RestorationOutcomePredictor,
    RestorationProject,
)

from sentinel.models.twin.recovery_planner import (
    HabitatAssessment,
    INTERVENTION_CATALOG,
    PRIORITY_SPECIES,
    RecoveryPlan,
    RecoveryPlanner,
    SpeciesHabitatRequirements,
)

__all__ = [
    # twin_engine
    "STATE_VARS",
    "NUM_STATE_VARS",
    "FORECAST_HORIZONS",
    "BiogeochemicalODE",
    "CounterfactualEngine",
    "DataAssimilator",
    "DigitalTwinEngine",
    "ForecastHead",
    "NeuralCorrector",
    "TwinOutput",
    # counterfactual
    "CounterfactualOutput",
    "CounterfactualSimulator",
    "InterventionLibrary",
    "InterventionSpec",
    "InterventionType",
    "SpatialScope",
    # restoration
    "BioremediationRecommender",
    "CostOptimizer",
    "ProjectType",
    "RestorationOutcome",
    "RestorationOutcomePredictor",
    "RestorationProject",
    # recovery_planner
    "HabitatAssessment",
    "INTERVENTION_CATALOG",
    "PRIORITY_SPECIES",
    "RecoveryPlan",
    "RecoveryPlanner",
    "SpeciesHabitatRequirements",
]
