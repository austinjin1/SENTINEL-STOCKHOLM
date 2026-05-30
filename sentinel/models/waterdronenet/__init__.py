"""WaterDroneNet: Physics-conditioned water quality prediction from drone imagery.

Fuses RGB+NIR drone imagery with cheap scalar sensors (temperature, TDS)
to predict a full 12-parameter water quality panel with per-target
uncertainty estimates and trust flags.

Public API
----------
.. autosummary::

    WaterDroneNet
    WaterDroneNetOutput
    TargetPrediction
    DroneVisionEncoder
    ScalarEncoder
    FiLMConditionedFusion
    PhysicsResidualHead
    UncertaintyRouter
    TARGET_PARAMS
    NUM_TARGETS
    FUSED_DIM
    DRONE_EMBED_DIM
    DRONE_IN_CHANS
    TRUST_GREEN_THRESHOLD
    TRUST_RED_THRESHOLD
"""

from sentinel.models.waterdronenet.waterdronenet import (
    # Main model
    WaterDroneNet,
    # Output dataclasses
    WaterDroneNetOutput,
    TargetPrediction,
    # Sub-modules
    DroneVisionEncoder,
    ScalarEncoder,
    FiLMConditionedFusion,
    PhysicsResidualHead,
    UncertaintyRouter,
    # Constants
    TARGET_PARAMS,
    NUM_TARGETS,
    FUSED_DIM,
    DRONE_EMBED_DIM,
    DRONE_IN_CHANS,
    DRONE_IMG_SIZE,
    NUM_SCALAR_INPUTS,
    TRUST_GREEN_THRESHOLD,
    TRUST_RED_THRESHOLD,
)

__all__ = [
    # Main model
    "WaterDroneNet",
    # Output dataclasses
    "WaterDroneNetOutput",
    "TargetPrediction",
    # Sub-modules
    "DroneVisionEncoder",
    "ScalarEncoder",
    "FiLMConditionedFusion",
    "PhysicsResidualHead",
    "UncertaintyRouter",
    # Constants
    "TARGET_PARAMS",
    "NUM_TARGETS",
    "FUSED_DIM",
    "DRONE_EMBED_DIM",
    "DRONE_IN_CHANS",
    "DRONE_IMG_SIZE",
    "NUM_SCALAR_INPUTS",
    "TRUST_GREEN_THRESHOLD",
    "TRUST_RED_THRESHOLD",
]
