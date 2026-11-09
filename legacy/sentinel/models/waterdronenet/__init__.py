"""WaterDroneNet: Image-only water quality prediction from drone/satellite imagery.

SENTINEL Mini — a lightweight, deployable model for drones equipped with
a dual-camera payload: Raspberry Pi Camera Module 3 Wide (RGB) + Raspberry Pi NoIR Camera
Module V2 (8MP, 1080P30) for near-infrared capture. Predicts 5 key water
quality parameters from 4-band (RGB + NIR) imagery alone, without
requiring any scalar sensor inputs.

Public API
----------
.. autosummary::

    WaterDroneNet
    WaterDroneNetOutput
    TargetPrediction
    MAEDecoder
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
    # MAE pretraining decoder
    MAEDecoder,
    # Output dataclasses
    WaterDroneNetOutput,
    TargetPrediction,
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
    # MAE pretraining decoder
    "MAEDecoder",
    # Output dataclasses
    "WaterDroneNetOutput",
    "TargetPrediction",
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
