"""HydroViT satellite encoder: water-specific foundation model + temporal + multi-resolution."""

from .model import SatelliteEncoder
from .hydrovit_backbone import HydroViTBackbone
from .multi_resolution import ResolutionCrossAttention
from .temporal_stack import TemporalAttentionStack
from .parameter_head import WaterQualityHead, PARAM_NAMES, NUM_WATER_PARAMS
from .physics_loss import SpectralPhysicsLoss

__all__ = [
    "SatelliteEncoder",
    "HydroViTBackbone",
    "ResolutionCrossAttention",
    "TemporalAttentionStack",
    "WaterQualityHead",
    "SpectralPhysicsLoss",
    "PARAM_NAMES",
    "NUM_WATER_PARAMS",
]
