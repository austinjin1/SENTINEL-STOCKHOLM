"""Perceiver IO Multimodal Fusion Layer for SENTINEL.

Public API
----------
.. autosummary::

    PerceiverIOFusion
    FusionOutput
    SentinelOutputHeads
    SentinelHeadsOutput
    EmbeddingRegistry
    ProjectionBank
    TemporalDecay
    PerceiverCrossAttention
    LatentArray
    ConfidenceGate
    CrossModalConsistencyLoss
    AnomalyDetectionHead
    SourceAttributionHead
    BiosentinelIntegrationHead
    EscalationRecommendationHead
"""

from sentinel.models.fusion.confidence_gating import ConfidenceGate
from sentinel.models.fusion.consistency_loss import CrossModalConsistencyLoss
from sentinel.models.fusion.embedding_registry import (
    MODALITY_IDS,
    NUM_MODALITIES,
    SHARED_EMBEDDING_DIM,
    EmbeddingRegistry,
    ModalityEntry,
)
from sentinel.models.fusion.heads import (
    AnomalyDetectionHead,
    AnomalyOutput,
    BiosentinelIntegrationHead,
    BiosentinelOutput,
    EscalationOutput,
    EscalationRecommendationHead,
    SentinelHeadsOutput,
    SentinelOutputHeads,
    SourceAttributionHead,
    SourceAttributionOutput,
)
from sentinel.models.fusion.latent_array import LatentArray
from sentinel.models.fusion.model import FusionOutput, PerceiverIOFusion
from sentinel.models.fusion.perceiver_attention import PerceiverCrossAttention
from sentinel.models.fusion.projections import (
    NATIVE_DIMS,
    ModalityProjection,
    ProjectionBank,
)
from sentinel.models.fusion.temporal_decay import TemporalDecay

# Backwards compatibility aliases.
CrossModalTemporalFusion = PerceiverIOFusion

__all__ = [
    # Core fusion
    "PerceiverIOFusion",
    "CrossModalTemporalFusion",  # backwards compat alias
    "FusionOutput",
    # Sub-modules
    "PerceiverCrossAttention",
    "LatentArray",
    "ConfidenceGate",
    "CrossModalConsistencyLoss",
    "EmbeddingRegistry",
    "ModalityEntry",
    "ProjectionBank",
    "ModalityProjection",
    "TemporalDecay",
    # Output heads
    "AnomalyDetectionHead",
    "SourceAttributionHead",
    "BiosentinelIntegrationHead",
    "EscalationRecommendationHead",
    "SentinelOutputHeads",
    # Output dataclasses
    "AnomalyOutput",
    "SourceAttributionOutput",
    "BiosentinelOutput",
    "EscalationOutput",
    "SentinelHeadsOutput",
    # Constants
    "MODALITY_IDS",
    "NUM_MODALITIES",
    "SHARED_EMBEDDING_DIM",
    "NATIVE_DIMS",
]
