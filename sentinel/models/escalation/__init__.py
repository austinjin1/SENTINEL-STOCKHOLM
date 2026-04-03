"""Cascade Escalation Controller — RL-based tiered monitoring for SENTINEL.

Decides which sensor modalities to activate at each timestep, escalating
when anomalies warrant investigation and de-escalating under normal
conditions.  Trained via PPO with curriculum learning; ships with an
interpretable decision-tree extraction for resource-constrained deployment.

Public API
----------
CascadeEscalationController
    Top-level controller (train, predict, extract protocol).
CascadeEscalationEnv
    Gymnasium MDP environment.
EscalationPolicyNetwork
    Standalone actor-critic MLP (outside SB3).
CurriculumScheduler
    Curriculum learning schedule manager.
"""

from sentinel.models.escalation.curriculum import (
    CurriculumCallback,
    CurriculumPhase,
    CurriculumScheduler,
)
from sentinel.models.escalation.decision_tree import (
    ExtractionResult,
    collect_policy_dataset,
    extract_decision_tree,
    format_protocol,
    save_protocol,
)
from sentinel.models.escalation.environment import (
    NUM_ACTIONS,
    NUM_TIERS,
    STATE_DIM,
    TIER_COMPUTE_COST,
    TIER_MODALITIES,
    CascadeEscalationEnv,
    ContaminationEvent,
    EpisodeScenario,
)
from sentinel.models.escalation.model import CascadeEscalationController
from sentinel.models.escalation.policy import (
    EscalationFeaturesExtractor,
    EscalationPolicyNetwork,
    create_ppo_agent,
)

__all__ = [
    # Top-level controller
    "CascadeEscalationController",
    # Environment
    "CascadeEscalationEnv",
    "ContaminationEvent",
    "EpisodeScenario",
    # Policy
    "EscalationPolicyNetwork",
    "EscalationFeaturesExtractor",
    "create_ppo_agent",
    # Curriculum
    "CurriculumScheduler",
    "CurriculumPhase",
    "CurriculumCallback",
    # Decision tree
    "ExtractionResult",
    "collect_policy_dataset",
    "extract_decision_tree",
    "format_protocol",
    "save_protocol",
    # Constants
    "NUM_ACTIONS",
    "NUM_TIERS",
    "STATE_DIM",
    "TIER_COMPUTE_COST",
    "TIER_MODALITIES",
]
