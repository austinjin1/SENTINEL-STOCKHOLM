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

# Lazy imports: curriculum, policy, decision_tree, and model depend on
# stable_baselines3 which triggers tensorboard → tensorflow.  Defer so
# that scripts which only need the environment (e.g. exp4) can import
# this package without a working TF installation.

def __getattr__(name):
    _curriculum_names = {"CurriculumCallback", "CurriculumPhase", "CurriculumScheduler"}
    _decision_tree_names = {
        "ExtractionResult", "collect_policy_dataset", "extract_decision_tree",
        "format_protocol", "save_protocol",
    }
    _policy_names = {"EscalationFeaturesExtractor", "EscalationPolicyNetwork", "create_ppo_agent"}
    _model_names = {"CascadeEscalationController"}

    if name in _curriculum_names:
        from sentinel.models.escalation import curriculum
        return getattr(curriculum, name)
    if name in _decision_tree_names:
        from sentinel.models.escalation import decision_tree
        return getattr(decision_tree, name)
    if name in _policy_names:
        from sentinel.models.escalation import policy
        return getattr(policy, name)
    if name in _model_names:
        from sentinel.models.escalation import model
        return getattr(model, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

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
