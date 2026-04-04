"""Novel theoretical contributions for SENTINEL.

This package contains five modules implementing the core theoretical
contributions of the SENTINEL framework:

1. **HEMA** --- Hierarchical Environmental Modality Alignment
2. **Causal Discovery** --- Heterogeneous Temporal Causal Discovery
3. **Sensor Placement** --- Information-Theoretic Sensor Placement
4. **Aitchison NN** --- Compositionally Coherent Neural Networks
5. **Conformal** --- Multimodal Conformal Anomaly Detection
"""

from sentinel.models.theory.aitchison_nn import (
    AitchisonBatchNorm,
    AitchisonLinear,
    AitchisonMLP,
    aitchison_distance,
    aitchison_inner_product,
    closure,
    clr_transform,
    compositional_coherence_test,
    inv_clr,
)
from sentinel.models.theory.causal_discovery import (
    CausalEdge,
    CausalGraph,
    CrossSpaceConditionalIndependence,
    FDRController,
    HeterogeneousPCMCI,
    SpaceType,
)
from sentinel.models.theory.conformal import (
    AnomalyPrediction,
    ChangePoint,
    ChangePointDetector,
    ConformalAnomalyDetector,
    GeometryAwareNonconformityScore,
    MultimodalConformalEnsemble,
)
from sentinel.models.theory.hema import (
    HEMALoss,
    ModalityAligner,
)
from sentinel.models.theory.sensor_placement import (
    CandidateSensor,
    GNNSurrogate,
    Placement,
    SubmodularObjective,
    optimize_placement,
)

__all__ = [
    # HEMA
    "HEMALoss",
    "ModalityAligner",
    # Causal Discovery
    "CausalEdge",
    "CausalGraph",
    "CrossSpaceConditionalIndependence",
    "FDRController",
    "HeterogeneousPCMCI",
    "SpaceType",
    # Sensor Placement
    "CandidateSensor",
    "GNNSurrogate",
    "Placement",
    "SubmodularObjective",
    "optimize_placement",
    # Aitchison NN
    "AitchisonBatchNorm",
    "AitchisonLinear",
    "AitchisonMLP",
    "aitchison_distance",
    "aitchison_inner_product",
    "closure",
    "clr_transform",
    "compositional_coherence_test",
    "inv_clr",
    # Conformal
    "AnomalyPrediction",
    "ChangePoint",
    "ChangePointDetector",
    "ConformalAnomalyDetector",
    "GeometryAwareNonconformityScore",
    "MultimodalConformalEnsemble",
]
