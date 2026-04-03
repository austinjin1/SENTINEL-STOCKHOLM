"""Microbial encoder: source attribution transformer + community trajectory VAE."""

from .model import MicrobialEncoder
from .source_attribution import SourceAttributionTransformer, CONTAMINATION_SOURCES
from .vae import CommunityTrajectoryVAE

__all__ = [
    "MicrobialEncoder",
    "SourceAttributionTransformer",
    "CommunityTrajectoryVAE",
    "CONTAMINATION_SOURCES",
]
