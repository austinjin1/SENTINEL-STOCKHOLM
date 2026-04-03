"""Cross-Modal Temporal Attention Fusion Layer for SENTINEL.

Public API
----------
.. autosummary::

    CrossModalTemporalFusion
    FusionOutput
    SentinelOutputHeads
    SentinelHeadsOutput
    EmbeddingRegistry
    ProjectionBank
    TemporalDecay
    CrossModalTemporalAttention
    WaterwayStateGRU
    AnomalyDetectionHead
    SourceAttributionHead
    BiosentinelIntegrationHead
"""

from sentinel.models.fusion.attention import CrossModalTemporalAttention
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
    SentinelHeadsOutput,
    SentinelOutputHeads,
    SourceAttributionHead,
    SourceAttributionOutput,
)
from sentinel.models.fusion.model import CrossModalTemporalFusion, FusionOutput
from sentinel.models.fusion.projections import (
    NATIVE_DIMS,
    ModalityProjection,
    ProjectionBank,
)
from sentinel.models.fusion.state import WaterwayStateGRU
from sentinel.models.fusion.temporal_decay import TemporalDecay

__all__ = [
    # Core fusion
    "CrossModalTemporalFusion",
    "FusionOutput",
    # Sub-modules
    "CrossModalTemporalAttention",
    "EmbeddingRegistry",
    "ModalityEntry",
    "ProjectionBank",
    "ModalityProjection",
    "TemporalDecay",
    "WaterwayStateGRU",
    # Output heads
    "AnomalyDetectionHead",
    "SourceAttributionHead",
    "BiosentinelIntegrationHead",
    "SentinelOutputHeads",
    # Output dataclasses
    "AnomalyOutput",
    "SourceAttributionOutput",
    "BiosentinelOutput",
    "SentinelHeadsOutput",
    # Constants
    "MODALITY_IDS",
    "NUM_MODALITIES",
    "SHARED_EMBEDDING_DIM",
    "NATIVE_DIMS",
]
