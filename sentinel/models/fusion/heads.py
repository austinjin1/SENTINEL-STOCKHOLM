"""Output heads for the SENTINEL fusion layer.

Four heads consume the fused waterway state to produce actionable
predictions:

1. **AnomalyDetectionHead** -- real-time anomaly detection with
   severity and type classification.
2. **SourceAttributionHead** -- contaminant source identification with
   confidence-calibrated probabilities.
3. **BiosentinelIntegrationHead** -- translates the fused state into
   a predicted chemistry vector for the Digital Biosentinel ecological
   impact model.
4. **EscalationRecommendationHead** -- recommends a monitoring tier
   and escalation urgency based on the fused state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from sentinel.models.fusion.embedding_registry import SHARED_EMBEDDING_DIM

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Anomaly type classes.
ANOMALY_TYPES: tuple[str, ...] = (
    "chemical_spike",
    "dissolved_oxygen_drop",
    "turbidity_surge",
    "ph_deviation",
    "temperature_anomaly",
    "nutrient_bloom",
    "microbial_shift",
    "toxic_compound",
)
NUM_ANOMALY_TYPES: int = len(ANOMALY_TYPES)

# Alert levels (ordinal).
ALERT_LEVELS: tuple[str, ...] = ("no_event", "low", "high")
NUM_ALERT_LEVELS: int = len(ALERT_LEVELS)

# Contaminant source classes.
CONTAMINANT_CLASSES: tuple[str, ...] = (
    "nutrient",
    "heavy_metals",
    "thermal",
    "pharmaceutical",
    "sediment",
    "oil_petrochemical",
    "sewage",
    "acid_mine",
)
NUM_CONTAMINANT_CLASSES: int = len(CONTAMINANT_CLASSES)

# Chemistry parameters predicted for the Digital Biosentinel.
CHEMISTRY_PARAMS: tuple[str, ...] = (
    "dissolved_oxygen",
    "ph",
    "temperature",
    "turbidity",
    "nitrate",
    "phosphate",
    "ammonia",
    "conductivity",
    "total_organic_carbon",
    "heavy_metal_index",
    "pharmaceutical_index",
    "hydrocarbon_index",
)
NUM_CHEMISTRY_PARAMS: int = len(CHEMISTRY_PARAMS)


# ---------------------------------------------------------------------------
# Dataclasses for structured output
# ---------------------------------------------------------------------------

@dataclass
class AnomalyOutput:
    """Structured output from :class:`AnomalyDetectionHead`.

    Attributes:
        anomaly_probability: Scalar probability that *any* anomaly is
            occurring, in ``[0, 1]``.
        severity_score: Scalar severity in ``[0, 1]``
            (0 = negligible, 1 = critical).
        anomaly_type_logits: Raw logits for multilabel anomaly type
            classification, shape ``[B, C_anomaly]``.
        anomaly_type_probs: Sigmoid probabilities per anomaly type.
        alert_level_logits: Logits for ordinal alert level, shape
            ``[B, 3]``.
        alert_level_probs: Softmax probabilities over alert levels.
    """

    anomaly_probability: torch.Tensor
    severity_score: torch.Tensor
    anomaly_type_logits: torch.Tensor
    anomaly_type_probs: torch.Tensor
    alert_level_logits: torch.Tensor
    alert_level_probs: torch.Tensor


@dataclass
class SourceAttributionOutput:
    """Structured output from :class:`SourceAttributionHead`.

    Attributes:
        class_logits: Raw logits, shape ``[B, C_classes]``.
        class_probs: Softmax probabilities over contaminant classes.
        confidence: Calibrated confidence score in ``[0, 1]``.
        top_class_idx: Index of highest-probability class.
        modifier_class_idx: Index of second-highest class (possible
            modifier / co-contaminant).
    """

    class_logits: torch.Tensor
    class_probs: torch.Tensor
    confidence: torch.Tensor
    top_class_idx: torch.Tensor
    modifier_class_idx: torch.Tensor


@dataclass
class BiosentinelOutput:
    """Structured output from :class:`BiosentinelIntegrationHead`.

    Attributes:
        chemistry_pred: Predicted chemistry concentrations, shape
            ``[B, C_chem]``.
        chemistry_log_var: Log-variance for each chemistry parameter
            (uncertainty), shape ``[B, C_chem]``.
    """

    chemistry_pred: torch.Tensor
    chemistry_log_var: torch.Tensor


# =========================================================================
# Head 1: Anomaly Detection
# =========================================================================

class AnomalyDetectionHead(nn.Module):
    """Multi-task anomaly detection head.

    Consumes the fused waterway state and produces:

    * Binary anomaly probability (any anomaly vs normal).
    * Continuous severity score.
    * Multilabel anomaly type classification.
    * Ordinal alert level (no_event / low / high).

    Architecture::

        fused_state [256]
            |
            MLP (256 -> 128, GELU, Dropout)
            |
        +---+---+---+
        |   |   |   |
       p(a) sev types alert

    Args:
        state_dim: Fused state dimensionality.
        hidden_dim: Intermediate MLP width.
        num_anomaly_types: Number of anomaly type classes.
        num_alert_levels: Number of alert levels.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        state_dim: int = SHARED_EMBEDDING_DIM,
        hidden_dim: int = 128,
        num_anomaly_types: int = NUM_ANOMALY_TYPES,
        num_alert_levels: int = NUM_ALERT_LEVELS,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.shared_mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # Sub-heads.
        self.anomaly_head = nn.Linear(hidden_dim, 1)
        self.severity_head = nn.Linear(hidden_dim, 1)
        self.type_head = nn.Linear(hidden_dim, num_anomaly_types)
        self.alert_head = nn.Linear(hidden_dim, num_alert_levels)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Bias the anomaly head toward "no anomaly" at initialization
        # to match the class prior (anomalies are rare).
        nn.init.constant_(self.anomaly_head.bias, -2.0)

    def forward(self, fused_state: torch.Tensor) -> AnomalyOutput:
        """Predict anomaly properties from fused state.

        Args:
            fused_state: Shape ``[B, state_dim]``.

        Returns:
            :class:`AnomalyOutput` with all predictions.
        """
        h = self.shared_mlp(fused_state)

        anomaly_logit = self.anomaly_head(h).squeeze(-1)
        anomaly_prob = torch.sigmoid(anomaly_logit)

        severity_logit = self.severity_head(h).squeeze(-1)
        severity = torch.sigmoid(severity_logit)

        type_logits = self.type_head(h)
        type_probs = torch.sigmoid(type_logits)

        alert_logits = self.alert_head(h)
        alert_probs = F.softmax(alert_logits, dim=-1)

        return AnomalyOutput(
            anomaly_probability=anomaly_prob,
            severity_score=severity,
            anomaly_type_logits=type_logits,
            anomaly_type_probs=type_probs,
            alert_level_logits=alert_logits,
            alert_level_probs=alert_probs,
        )


# =========================================================================
# Head 2: Source Attribution
# =========================================================================

class SourceAttributionHead(nn.Module):
    """Contaminant source identification head.

    Produces a probability distribution over 8 contaminant source
    classes together with a calibrated confidence score and "most
    likely" / "possible modifier" labels.

    Architecture::

        fused_state [256]
            -> Linear(256, 128), GELU, Dropout
            -> Linear(128, C_classes)  [class logits]
            -> Linear(128, 1)          [confidence]

    Args:
        state_dim: Fused state dimensionality.
        hidden_dim: Intermediate MLP width.
        num_classes: Number of contaminant source classes.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        state_dim: int = SHARED_EMBEDDING_DIM,
        hidden_dim: int = 128,
        num_classes: int = NUM_CONTAMINANT_CLASSES,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.class_head = nn.Linear(hidden_dim, num_classes)
        self.confidence_head = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, fused_state: torch.Tensor) -> SourceAttributionOutput:
        """Predict contaminant source from fused state.

        Args:
            fused_state: Shape ``[B, state_dim]``.

        Returns:
            :class:`SourceAttributionOutput`.
        """
        h = self.mlp(fused_state)

        class_logits = self.class_head(h)
        class_probs = F.softmax(class_logits, dim=-1)

        confidence = torch.sigmoid(self.confidence_head(h)).squeeze(-1)

        # Top-1 and top-2 classes.
        sorted_indices = class_probs.argsort(dim=-1, descending=True)
        top_class_idx = sorted_indices[:, 0]
        modifier_class_idx = sorted_indices[:, 1]

        return SourceAttributionOutput(
            class_logits=class_logits,
            class_probs=class_probs,
            confidence=confidence,
            top_class_idx=top_class_idx,
            modifier_class_idx=modifier_class_idx,
        )


# =========================================================================
# Head 3: Digital Biosentinel Integration
# =========================================================================

class BiosentinelIntegrationHead(nn.Module):
    """Maps fused state to chemistry vector for the Digital Biosentinel.

    Takes the fused environmental state and the source attribution
    probability vector as joint input and predicts the chemistry
    concentrations that the Digital Biosentinel model expects.

    The head also predicts per-parameter log-variance to express
    uncertainty, enabling downstream probabilistic ecological modelling.

    Architecture::

        [fused_state (256) ; source_probs (8)]  =  264
            -> Linear(264, 128), GELU, Dropout
            -> Linear(128, 64),  GELU, Dropout
            -> Linear(64, C_chem)     [mean predictions]
            -> Linear(64, C_chem)     [log-variance]

    Args:
        state_dim: Fused state dimensionality.
        num_source_classes: Dimensionality of source attribution vector.
        num_chemistry_params: Number of chemistry parameters to predict.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        state_dim: int = SHARED_EMBEDDING_DIM,
        num_source_classes: int = NUM_CONTAMINANT_CLASSES,
        num_chemistry_params: int = NUM_CHEMISTRY_PARAMS,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        input_dim = state_dim + num_source_classes

        self.shared_mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.mean_head = nn.Linear(64, num_chemistry_params)
        self.log_var_head = nn.Linear(64, num_chemistry_params)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Initialize log-variance to small values (low initial uncertainty).
        nn.init.constant_(self.log_var_head.bias, -2.0)
        nn.init.zeros_(self.log_var_head.weight)

    def forward(
        self,
        fused_state: torch.Tensor,
        source_probs: torch.Tensor,
    ) -> BiosentinelOutput:
        """Predict chemistry concentrations for the Digital Biosentinel.

        Args:
            fused_state: Shape ``[B, state_dim]``.
            source_probs: Source attribution probabilities from
                :class:`SourceAttributionHead`, shape ``[B, C_classes]``.

        Returns:
            :class:`BiosentinelOutput` with mean predictions and
            log-variance uncertainty estimates.
        """
        x = torch.cat([fused_state, source_probs], dim=-1)
        h = self.shared_mlp(x)

        chemistry_pred = self.mean_head(h)
        chemistry_log_var = self.log_var_head(h)

        return BiosentinelOutput(
            chemistry_pred=chemistry_pred,
            chemistry_log_var=chemistry_log_var,
        )


# =========================================================================
# Head 4: Escalation Recommendation
# =========================================================================

# Monitoring tiers (ordinal).
MONITORING_TIERS: tuple[str, ...] = (
    "routine",       # Tier 0: normal scheduled monitoring
    "elevated",      # Tier 1: increase sampling frequency
    "intensive",     # Tier 2: deploy additional sensors / field team
    "emergency",     # Tier 3: immediate response required
)
NUM_MONITORING_TIERS: int = len(MONITORING_TIERS)


@dataclass
class EscalationOutput:
    """Structured output from :class:`EscalationRecommendationHead`.

    Attributes:
        tier_logits: Raw logits over monitoring tiers, shape
            ``[B, 4]``.
        tier_probs: Softmax probabilities over monitoring tiers.
        recommended_tier: Index of the recommended monitoring tier.
        escalation_urgency: Urgency score in ``[0, 1]``
            (0 = no urgency, 1 = immediate action needed).
    """

    tier_logits: torch.Tensor
    tier_probs: torch.Tensor
    recommended_tier: torch.Tensor
    escalation_urgency: torch.Tensor


class EscalationRecommendationHead(nn.Module):
    """Monitoring escalation recommendation head.

    Consumes the fused waterway state and produces:

    * Recommended monitoring tier (logits over 4 tiers).
    * Escalation urgency score (continuous 0-1).

    Architecture::

        fused_state [256]
            |
            MLP (256 -> 128, GELU, Dropout)
            |
        +-------+--------+
        |                |
       tiers           urgency
       [B, 4]          [B]

    Args:
        state_dim: Fused state dimensionality.
        hidden_dim: Intermediate MLP width.
        num_tiers: Number of monitoring tiers.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        state_dim: int = SHARED_EMBEDDING_DIM,
        hidden_dim: int = 128,
        num_tiers: int = NUM_MONITORING_TIERS,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.tier_head = nn.Linear(hidden_dim, num_tiers)
        self.urgency_head = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Bias toward "routine" tier at initialization.
        nn.init.constant_(self.tier_head.bias[0], 1.0)
        # Bias urgency toward low at initialization.
        nn.init.constant_(self.urgency_head.bias, -2.0)

    def forward(self, fused_state: torch.Tensor) -> EscalationOutput:
        """Predict escalation recommendation from fused state.

        Args:
            fused_state: Shape ``[B, state_dim]``.

        Returns:
            :class:`EscalationOutput`.
        """
        h = self.mlp(fused_state)

        tier_logits = self.tier_head(h)
        tier_probs = F.softmax(tier_logits, dim=-1)
        recommended_tier = tier_probs.argmax(dim=-1)

        urgency = torch.sigmoid(self.urgency_head(h)).squeeze(-1)

        return EscalationOutput(
            tier_logits=tier_logits,
            tier_probs=tier_probs,
            recommended_tier=recommended_tier,
            escalation_urgency=urgency,
        )


# =========================================================================
# Combined head wrapper
# =========================================================================

@dataclass
class SentinelHeadsOutput:
    """Combined output from all four heads.

    Attributes:
        anomaly: Output from :class:`AnomalyDetectionHead`.
        source: Output from :class:`SourceAttributionHead`.
        biosentinel: Output from :class:`BiosentinelIntegrationHead`.
        escalation: Output from :class:`EscalationRecommendationHead`.
    """

    anomaly: AnomalyOutput
    source: SourceAttributionOutput
    biosentinel: BiosentinelOutput
    escalation: EscalationOutput


class SentinelOutputHeads(nn.Module):
    """Convenience wrapper running all four output heads.

    Args:
        state_dim: Fused state dimensionality.
        dropout: Shared dropout probability.
    """

    def __init__(
        self,
        state_dim: int = SHARED_EMBEDDING_DIM,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.anomaly_head = AnomalyDetectionHead(
            state_dim=state_dim, dropout=dropout
        )
        self.source_head = SourceAttributionHead(
            state_dim=state_dim, dropout=dropout
        )
        self.biosentinel_head = BiosentinelIntegrationHead(
            state_dim=state_dim, dropout=dropout
        )
        self.escalation_head = EscalationRecommendationHead(
            state_dim=state_dim, dropout=dropout
        )

    def forward(self, fused_state: torch.Tensor) -> SentinelHeadsOutput:
        """Run all four heads on the fused waterway state.

        The source attribution probabilities are automatically piped
        into the biosentinel integration head.

        Args:
            fused_state: Shape ``[B, state_dim]``.

        Returns:
            :class:`SentinelHeadsOutput` with all predictions.
        """
        anomaly_out = self.anomaly_head(fused_state)
        source_out = self.source_head(fused_state)
        biosentinel_out = self.biosentinel_head(
            fused_state, source_out.class_probs
        )
        escalation_out = self.escalation_head(fused_state)
        return SentinelHeadsOutput(
            anomaly=anomaly_out,
            source=source_out,
            biosentinel=biosentinel_out,
            escalation=escalation_out,
        )
