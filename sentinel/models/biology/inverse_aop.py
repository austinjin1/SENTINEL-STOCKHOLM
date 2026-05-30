"""Inverse AOP-Wiki Pathway Activation Prediction (Phase 3.7).

Predicts which Adverse Outcome Pathways (AOPs) are activated in
resident fish populations given environmental sensor + satellite data,
**without** requiring molecular biomarker data.

This inverts the direction of ToxiGene (which maps chemical exposure
to pathway activation): instead of going chemical -> gene expression
-> pathway, this module goes environment -> predicted pathway
activation, enabling non-invasive real-time AOP monitoring from
remote sensing and in-situ sensors alone.

Architecture overview
---------------------
::

    SENTINEL embedding (B, 256)
    + Environmental sensor data (B, E_sensor)
    + Satellite features (B, S_sat)
        --> MultiSourceEncoder (fuses all inputs -> 256-d)
        --> EnvironmentToPathwayAttention
            (cross-attention: environment queries AOP pathway keys)
        --> Per-AOP prediction heads
            --> activation probability (B, 7)
            --> severity score (B, 7) in [0, 1]
            --> confidence score (B, 7) in [0, 1]
        --> MC-dropout uncertainty (20 stochastic passes)

Seven major AOP categories (AOP-Wiki):
    1. Estrogenic         -- endocrine disruption (17beta-estradiol axis)
    2. Androgenic         -- androgen receptor disruption
    3. Thyroid            -- thyroid hormone axis disruption
    4. Oxidative stress   -- ROS-mediated cellular damage
    5. Neurotoxicity      -- acetylcholinesterase inhibition, neuronal damage
    6. Hepatotoxicity     -- liver damage, metabolic disruption
    7. Immunotoxicity     -- immune suppression, inflammatory response

Key environmental predictors:
    - Temperature, pH, dissolved oxygen (oxidative stress trigger)
    - Turbidity, suspended sediments (contaminant carriers)
    - NDVI, chlorophyll-a (eutrophication -> cyanotoxin exposure)
    - Land use (agricultural -> pesticides, urban -> pharmaceuticals)
    - Satellite-derived water quality indices
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Constants
# ============================================================================

SHARED_EMBEDDING_DIM: int = 256
MC_DROPOUT_SAMPLES: int = 20
DROPOUT_P: float = 0.2

# Environmental sensor features
NUM_SENSOR_FEATURES: int = 16
SENSOR_FEATURE_NAMES: Tuple[str, ...] = (
    "temperature_C",
    "pH",
    "dissolved_oxygen_mg_L",
    "conductivity_uS_cm",
    "turbidity_NTU",
    "total_dissolved_solids_mg_L",
    "nitrate_mg_L",
    "phosphate_mg_L",
    "ammonia_mg_L",
    "chlorophyll_a_ug_L",
    "dissolved_organic_carbon_mg_L",
    "oxidation_reduction_potential_mV",
    "total_suspended_solids_mg_L",
    "alkalinity_mg_L",
    "hardness_mg_L",
    "flow_rate_m3_s",
)

# Satellite-derived features
NUM_SATELLITE_FEATURES: int = 12
SATELLITE_FEATURE_NAMES: Tuple[str, ...] = (
    "NDVI",
    "NDWI",
    "chlorophyll_index",
    "turbidity_index",
    "surface_temperature_K",
    "land_use_agriculture_frac",
    "land_use_urban_frac",
    "land_use_forest_frac",
    "impervious_surface_frac",
    "wetland_proximity_km",
    "upstream_agriculture_area_km2",
    "point_source_density_per_km2",
)


# ---------------------------------------------------------------------------
# AOP categories
# ---------------------------------------------------------------------------

class AOPCategory(IntEnum):
    """Seven major AOP categories from AOP-Wiki."""
    ESTROGENIC = 0
    ANDROGENIC = 1
    THYROID = 2
    OXIDATIVE_STRESS = 3
    NEUROTOXICITY = 4
    HEPATOTOXICITY = 5
    IMMUNOTOXICITY = 6


NUM_AOP_CATEGORIES: int = len(AOPCategory)

AOP_NAMES: Tuple[str, ...] = (
    "Estrogenic",
    "Androgenic",
    "Thyroid",
    "Oxidative Stress",
    "Neurotoxicity",
    "Hepatotoxicity",
    "Immunotoxicity",
)

AOP_INDEX_TO_NAME: Dict[int, str] = {i: name for i, name in enumerate(AOP_NAMES)}
AOP_NAME_TO_INDEX: Dict[str, int] = {name: i for i, name in enumerate(AOP_NAMES)}


@dataclass(frozen=True)
class AOPDescription:
    """Detailed description of an AOP category."""

    index: int
    name: str
    molecular_target: str
    key_events: str
    adverse_outcome: str
    environmental_drivers: str


AOP_CATALOG: Tuple[AOPDescription, ...] = (
    AOPDescription(
        index=0,
        name="Estrogenic",
        molecular_target="Estrogen receptor alpha (ERalpha)",
        key_events="ER binding -> vitellogenin induction -> impaired reproduction",
        adverse_outcome="Population decline via reproductive failure",
        environmental_drivers="EE2, E2, BPA, nonylphenol, WWTP effluent",
    ),
    AOPDescription(
        index=1,
        name="Androgenic",
        molecular_target="Androgen receptor (AR)",
        key_events="AR antagonism -> reduced spiggin -> impaired spermatogenesis",
        adverse_outcome="Reproductive impairment, intersex",
        environmental_drivers="Flutamide, vinclozolin, pulp mill effluent",
    ),
    AOPDescription(
        index=2,
        name="Thyroid",
        molecular_target="Thyroid hormone receptors (TR), TPO",
        key_events="TPO inhibition -> reduced T3/T4 -> altered metamorphosis/growth",
        adverse_outcome="Developmental abnormalities, growth impairment",
        environmental_drivers="Perchlorate, PTU, triclosan, PBDEs",
    ),
    AOPDescription(
        index=3,
        name="Oxidative Stress",
        molecular_target="Nrf2/Keap1, mitochondrial ETC",
        key_events="ROS generation -> lipid peroxidation -> cell death",
        adverse_outcome="Tissue damage, organ failure",
        environmental_drivers="Metals (Cu, Zn), nanoparticles, low DO, high temperature",
    ),
    AOPDescription(
        index=4,
        name="Neurotoxicity",
        molecular_target="Acetylcholinesterase (AChE), NMDA receptors",
        key_events="AChE inhibition -> acetylcholine accumulation -> seizures",
        adverse_outcome="Behavioral changes, mortality",
        environmental_drivers="Organophosphates, carbamates, neonicotinoids",
    ),
    AOPDescription(
        index=5,
        name="Hepatotoxicity",
        molecular_target="PXR, CAR, PPARs, cytochrome P450s",
        key_events="Metabolic enzyme induction -> hepatocyte hypertrophy -> liver tumors",
        adverse_outcome="Liver failure, metabolic disruption",
        environmental_drivers="PAHs, PCBs, dioxins, cyanotoxins (microcystin)",
    ),
    AOPDescription(
        index=6,
        name="Immunotoxicity",
        molecular_target="AhR, NF-kB, complement system",
        key_events="Immune suppression -> reduced pathogen resistance",
        adverse_outcome="Increased disease susceptibility, population decline",
        environmental_drivers="PAHs, PCBs, heavy metals, microplastics",
    ),
)


# ---------------------------------------------------------------------------
# Alert and severity thresholds
# ---------------------------------------------------------------------------

class AOPAlertLevel(IntEnum):
    """Alert levels for AOP activation."""
    INACTIVE = 0          # P(activation) < 0.2
    LOW_CONCERN = 1       # 0.2 <= P < 0.4
    MODERATE_CONCERN = 2  # 0.4 <= P < 0.6
    HIGH_CONCERN = 3      # 0.6 <= P < 0.8
    CRITICAL = 4          # P >= 0.8

AOP_ALERT_THRESHOLDS: Dict[str, float] = {
    "low": 0.2,
    "moderate": 0.4,
    "high": 0.6,
    "critical": 0.8,
}


# ============================================================================
# Output dataclasses
# ============================================================================

@dataclass
class AOPPredictionOutput:
    """Output of the InverseAOPPredictor.

    Attributes
    ----------
    activation_prob : Tensor[B, 7]
        Probability that each AOP category is activated.
    severity_score : Tensor[B, 7]
        Predicted severity of activation (0 = minimal, 1 = severe).
    confidence : Tensor[B, 7]
        Model confidence in each prediction (0-1).
    alert_levels : Tensor[B, 7]
        Integer alert level per AOP category.
    attention_weights : Tensor[B, 7, E]
        Attention weights mapping environmental features to each AOP.
    mc_activation_mean : Tensor[B, 7] or None
        MC-dropout posterior mean of activation probability.
    mc_activation_std : Tensor[B, 7] or None
        MC-dropout posterior std of activation probability.
    mc_severity_mean : Tensor[B, 7] or None
        MC-dropout posterior mean of severity score.
    mc_severity_std : Tensor[B, 7] or None
        MC-dropout posterior std of severity score.
    """

    activation_prob: torch.Tensor
    severity_score: torch.Tensor
    confidence: torch.Tensor
    alert_levels: torch.Tensor
    attention_weights: torch.Tensor

    mc_activation_mean: Optional[torch.Tensor] = None
    mc_activation_std: Optional[torch.Tensor] = None
    mc_severity_mean: Optional[torch.Tensor] = None
    mc_severity_std: Optional[torch.Tensor] = None


# ============================================================================
# Sub-module 1: Multi-Source Input Encoder
# ============================================================================

class MultiSourceEncoder(nn.Module):
    """Encode and fuse SENTINEL embedding, sensor data, and satellite features.

    Three-stream encoder that projects each input modality into a
    shared space, then fuses them with gated attention to produce a
    unified environmental representation.

    Parameters
    ----------
    embedding_dim : int
        SENTINEL embedding dimension (default 256).
    num_sensor_features : int
        Number of in-situ sensor features (default 16).
    num_satellite_features : int
        Number of satellite-derived features (default 12).
    output_dim : int
        Output representation dimension (default 256).
    dropout : float
        Dropout probability (default 0.2).
    """

    def __init__(
        self,
        embedding_dim: int = SHARED_EMBEDDING_DIM,
        num_sensor_features: int = NUM_SENSOR_FEATURES,
        num_satellite_features: int = NUM_SATELLITE_FEATURES,
        output_dim: int = 256,
        dropout: float = DROPOUT_P,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim

        # Stream 1: SENTINEL embedding projection
        self.sentinel_proj = nn.Sequential(
            nn.Linear(embedding_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Stream 2: Sensor data projection
        self.sensor_proj = nn.Sequential(
            nn.Linear(num_sensor_features, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Stream 3: Satellite feature projection
        self.satellite_proj = nn.Sequential(
            nn.Linear(num_satellite_features, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Gated fusion: learn how much to weight each stream
        self.gate = nn.Sequential(
            nn.Linear(output_dim * 3, output_dim * 3),
            nn.Sigmoid(),
        )

        # Final fusion
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * 3, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(output_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        sentinel_embedding: torch.Tensor,
        sensor_data: Optional[torch.Tensor] = None,
        satellite_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fuse multi-source environmental data.

        Parameters
        ----------
        sentinel_embedding : Tensor[B, 256]
            SENTINEL environmental embedding.
        sensor_data : Tensor[B, 16], optional
            In-situ sensor measurements.  Zeros if unavailable.
        satellite_features : Tensor[B, 12], optional
            Satellite-derived features.  Zeros if unavailable.

        Returns
        -------
        Tensor[B, 256]
            Fused environmental representation.
        """
        B = sentinel_embedding.size(0)
        device = sentinel_embedding.device

        if sensor_data is None:
            sensor_data = torch.zeros(
                B, NUM_SENSOR_FEATURES, device=device,
            )
        if satellite_features is None:
            satellite_features = torch.zeros(
                B, NUM_SATELLITE_FEATURES, device=device,
            )

        # Project each stream
        s1 = self.sentinel_proj(sentinel_embedding)      # [B, 256]
        s2 = self.sensor_proj(sensor_data)                # [B, 256]
        s3 = self.satellite_proj(satellite_features)      # [B, 256]

        # Concatenate and gate
        concat = torch.cat([s1, s2, s3], dim=-1)  # [B, 768]
        gates = self.gate(concat)                  # [B, 768]
        gated = concat * gates                     # [B, 768]

        # Fuse to output dimension
        return self.fusion(gated)  # [B, 256]


# ============================================================================
# Sub-module 2: AOP Pathway Embeddings
# ============================================================================

class AOPPathwayEmbedding(nn.Module):
    """Learnable embeddings for the 7 AOP pathway categories.

    Each AOP category gets a 256-d embedding that captures its
    known molecular initiating events, key events, and adverse
    outcomes.  These embeddings serve as ``keys`` in the cross-
    attention mechanism, allowing the model to learn which
    environmental features predict which pathway activations.

    Parameters
    ----------
    num_categories : int
        Number of AOP categories (default 7).
    embedding_dim : int
        Embedding dimension (default 256).
    """

    def __init__(
        self,
        num_categories: int = NUM_AOP_CATEGORIES,
        embedding_dim: int = 256,
    ) -> None:
        super().__init__()
        self.num_categories = num_categories
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(num_categories, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self) -> torch.Tensor:
        """Return the full AOP embedding table.

        Returns
        -------
        Tensor[7, 256]
            AOP pathway embeddings.
        """
        idx = torch.arange(
            self.num_categories,
            device=self.embedding.weight.device,
        )
        return self.layer_norm(self.embedding(idx))


# ============================================================================
# Sub-module 3: Environment-to-Pathway Cross-Attention
# ============================================================================

class EnvironmentToPathwayAttention(nn.Module):
    """Cross-attention from environmental features to AOP pathways.

    Uses the environmental representation as the ``query`` and the
    AOP pathway embeddings as ``keys`` and ``values``.  This learns
    which environmental conditions activate which molecular pathways,
    implementing the inverse AOP mapping.

    The attention weights provide interpretable explanations:
    for each AOP, we can see which environmental features contributed
    most to the predicted activation.

    Parameters
    ----------
    env_dim : int
        Environmental representation dimension (default 256).
    aop_dim : int
        AOP pathway embedding dimension (default 256).
    num_heads : int
        Number of attention heads (default 8).
    dropout : float
        Dropout probability (default 0.2).
    """

    def __init__(
        self,
        env_dim: int = 256,
        aop_dim: int = 256,
        num_heads: int = 8,
        dropout: float = DROPOUT_P,
    ) -> None:
        super().__init__()
        self.env_dim = env_dim
        self.aop_dim = aop_dim
        self.num_heads = num_heads

        # Multi-head cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=aop_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Query projection for environment
        self.query_proj = nn.Sequential(
            nn.Linear(env_dim, aop_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Post-attention processing
        self.post_attention = nn.Sequential(
            nn.Linear(aop_dim, aop_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(aop_dim),
        )

        # Feature-level attention for interpretability
        # Maps env features to per-AOP attention scores
        self.feature_attention = nn.Sequential(
            nn.Linear(env_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, NUM_AOP_CATEGORIES * env_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        env_repr: torch.Tensor,
        aop_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cross-attend from environment to AOP pathways.

        Parameters
        ----------
        env_repr : Tensor[B, 256]
            Fused environmental representation.
        aop_embeddings : Tensor[7, 256]
            AOP pathway embedding table.

        Returns
        -------
        aop_features : Tensor[B, 7, 256]
            AOP-specific environmental features.
        attention_weights : Tensor[B, 7, 256]
            Feature-level attention weights for interpretability.
        """
        B = env_repr.size(0)
        K = aop_embeddings.size(0)  # 7

        # Project environment to query space
        query = self.query_proj(env_repr)  # [B, 256]

        # Expand query for each AOP
        query_expanded = query.unsqueeze(1).expand(B, K, -1)  # [B, 7, 256]

        # AOP embeddings as keys and values
        kv = aop_embeddings.unsqueeze(0).expand(B, -1, -1)  # [B, 7, 256]

        # Cross-attention
        attn_output, _ = self.cross_attention(
            query_expanded, kv, kv,
        )  # [B, 7, 256]

        # Post-attention processing with residual
        aop_features = self.post_attention(attn_output + query_expanded)

        # Feature-level attention for interpretability
        feat_attn = self.feature_attention(env_repr)  # [B, 7*256]
        feat_attn = feat_attn.view(B, K, -1)           # [B, 7, 256]
        attention_weights = torch.sigmoid(feat_attn)   # [B, 7, 256]

        return aop_features, attention_weights


# ============================================================================
# Sub-module 4: AOP Prediction Heads
# ============================================================================

class AOPPredictionHeads(nn.Module):
    """Per-AOP prediction heads for activation, severity, and confidence.

    Each AOP category has its own prediction head that takes the
    AOP-specific features from the cross-attention module and
    predicts:
    - activation probability (is this AOP active?)
    - severity score (how severe is the activation?)
    - confidence score (how confident is the model?)

    Parameters
    ----------
    input_dim : int
        AOP feature dimension from cross-attention (default 256).
    num_categories : int
        Number of AOP categories (default 7).
    hidden_dim : int
        Hidden layer size per head (default 128).
    dropout : float
        Dropout probability (default 0.2).
    """

    def __init__(
        self,
        input_dim: int = 256,
        num_categories: int = NUM_AOP_CATEGORIES,
        hidden_dim: int = 128,
        dropout: float = DROPOUT_P,
    ) -> None:
        super().__init__()
        self.num_categories = num_categories

        # Shared feature extractor per AOP
        self.shared_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 64),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for _ in range(num_categories)
        ])

        # Activation probability head
        self.activation_heads = nn.ModuleList([
            nn.Linear(64, 1) for _ in range(num_categories)
        ])

        # Severity score head
        self.severity_heads = nn.ModuleList([
            nn.Linear(64, 1) for _ in range(num_categories)
        ])

        # Confidence score head (epistemic uncertainty estimate)
        self.confidence_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(64, 32),
                nn.GELU(),
                nn.Linear(32, 1),
            )
            for _ in range(num_categories)
        ])

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        aop_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict activation, severity, and confidence for each AOP.

        Parameters
        ----------
        aop_features : Tensor[B, 7, 256]
            AOP-specific features from cross-attention.

        Returns
        -------
        activation_prob : Tensor[B, 7]
            Activation probabilities in [0, 1].
        severity_score : Tensor[B, 7]
            Severity scores in [0, 1].
        confidence : Tensor[B, 7]
            Confidence scores in [0, 1].
        """
        B = aop_features.size(0)

        act_list: List[torch.Tensor] = []
        sev_list: List[torch.Tensor] = []
        conf_list: List[torch.Tensor] = []

        for k in range(self.num_categories):
            feat = aop_features[:, k, :]  # [B, 256]
            shared = self.shared_layers[k](feat)  # [B, 64]

            act = torch.sigmoid(self.activation_heads[k](shared).squeeze(-1))
            sev = torch.sigmoid(self.severity_heads[k](shared).squeeze(-1))
            conf = torch.sigmoid(self.confidence_heads[k](shared).squeeze(-1))

            act_list.append(act)
            sev_list.append(sev)
            conf_list.append(conf)

        activation_prob = torch.stack(act_list, dim=-1)   # [B, 7]
        severity_score = torch.stack(sev_list, dim=-1)     # [B, 7]
        confidence = torch.stack(conf_list, dim=-1)        # [B, 7]

        return activation_prob, severity_score, confidence


# ============================================================================
# Main Model: InverseAOPPredictor
# ============================================================================

class InverseAOPPredictor(nn.Module):
    """Inverse AOP pathway activation predictor.

    Given environmental sensor and satellite data (without molecular
    biomarker data), predicts which Adverse Outcome Pathways are
    activated in resident fish populations.  This inverts ToxiGene's
    direction: instead of chemical -> pathway, it maps
    environment -> pathway.

    Parameters
    ----------
    embedding_dim : int
        SENTINEL embedding dimension (default 256).
    num_sensor_features : int
        Number of in-situ sensor features (default 16).
    num_satellite_features : int
        Number of satellite-derived features (default 12).
    num_attention_heads : int
        Number of attention heads for cross-attention (default 8).
    dropout : float
        Dropout probability (default 0.2).
    mc_samples : int
        Number of MC-dropout forward passes (default 20).

    Example
    -------
    >>> model = InverseAOPPredictor()
    >>> embedding = torch.randn(4, 256)
    >>> sensor = torch.randn(4, 16)
    >>> satellite = torch.randn(4, 12)
    >>> output = model(embedding, sensor, satellite)
    >>> output.activation_prob.shape
    torch.Size([4, 7])
    """

    def __init__(
        self,
        embedding_dim: int = SHARED_EMBEDDING_DIM,
        num_sensor_features: int = NUM_SENSOR_FEATURES,
        num_satellite_features: int = NUM_SATELLITE_FEATURES,
        num_attention_heads: int = 8,
        dropout: float = DROPOUT_P,
        mc_samples: int = MC_DROPOUT_SAMPLES,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.mc_samples = mc_samples
        self.dropout_p = dropout

        # Sub-modules
        self.multi_source_encoder = MultiSourceEncoder(
            embedding_dim=embedding_dim,
            num_sensor_features=num_sensor_features,
            num_satellite_features=num_satellite_features,
            output_dim=256,
            dropout=dropout,
        )

        self.aop_embeddings = AOPPathwayEmbedding(
            num_categories=NUM_AOP_CATEGORIES,
            embedding_dim=256,
        )

        self.env_to_pathway = EnvironmentToPathwayAttention(
            env_dim=256,
            aop_dim=256,
            num_heads=num_attention_heads,
            dropout=dropout,
        )

        self.prediction_heads = AOPPredictionHeads(
            input_dim=256,
            num_categories=NUM_AOP_CATEGORIES,
            hidden_dim=128,
            dropout=dropout,
        )

    # ------------------------------------------------------------------
    # MC-dropout helpers
    # ------------------------------------------------------------------

    def _enable_dropout(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def _disable_dropout(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.eval()

    # ------------------------------------------------------------------
    # Alert level assignment
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_alert_levels(activation_prob: torch.Tensor) -> torch.Tensor:
        """Assign alert levels based on activation probability.

        Parameters
        ----------
        activation_prob : Tensor[B, 7]

        Returns
        -------
        Tensor[B, 7] of long
        """
        alerts = torch.zeros_like(activation_prob, dtype=torch.long)
        alerts[activation_prob >= AOP_ALERT_THRESHOLDS["low"]] = AOPAlertLevel.LOW_CONCERN
        alerts[activation_prob >= AOP_ALERT_THRESHOLDS["moderate"]] = AOPAlertLevel.MODERATE_CONCERN
        alerts[activation_prob >= AOP_ALERT_THRESHOLDS["high"]] = AOPAlertLevel.HIGH_CONCERN
        alerts[activation_prob >= AOP_ALERT_THRESHOLDS["critical"]] = AOPAlertLevel.CRITICAL
        return alerts

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------

    def _single_forward(
        self,
        sentinel_embedding: torch.Tensor,
        sensor_data: Optional[torch.Tensor] = None,
        satellite_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single deterministic forward pass.

        Returns
        -------
        activation_prob, severity_score, confidence, attention_weights
        """
        # Encode multi-source input
        env_repr = self.multi_source_encoder(
            sentinel_embedding, sensor_data, satellite_features,
        )

        # Get AOP pathway embeddings
        aop_emb = self.aop_embeddings()  # [7, 256]

        # Cross-attention: environment -> pathways
        aop_features, attention_weights = self.env_to_pathway(
            env_repr, aop_emb,
        )

        # Predict activation, severity, confidence
        activation_prob, severity_score, confidence = self.prediction_heads(
            aop_features,
        )

        return activation_prob, severity_score, confidence, attention_weights

    def forward(
        self,
        sentinel_embedding: torch.Tensor,
        sensor_data: Optional[torch.Tensor] = None,
        satellite_features: Optional[torch.Tensor] = None,
        use_mc_dropout: bool = False,
    ) -> AOPPredictionOutput:
        """Forward pass with optional MC-dropout uncertainty.

        Parameters
        ----------
        sentinel_embedding : Tensor[B, 256]
            SENTINEL environmental embedding.
        sensor_data : Tensor[B, 16], optional
            In-situ sensor measurements.
        satellite_features : Tensor[B, 12], optional
            Satellite-derived features.
        use_mc_dropout : bool
            If True and in eval mode, run mc_samples stochastic
            forward passes.

        Returns
        -------
        AOPPredictionOutput
        """
        if not use_mc_dropout or self.training:
            act, sev, conf, attn = self._single_forward(
                sentinel_embedding, sensor_data, satellite_features,
            )
            return AOPPredictionOutput(
                activation_prob=act,
                severity_score=sev,
                confidence=conf,
                alert_levels=self._compute_alert_levels(act),
                attention_weights=attn,
            )

        # --- MC-dropout inference ---
        self._enable_dropout()

        mc_act: List[torch.Tensor] = []
        mc_sev: List[torch.Tensor] = []

        with torch.no_grad():
            for _ in range(self.mc_samples):
                act, sev, conf, attn = self._single_forward(
                    sentinel_embedding, sensor_data, satellite_features,
                )
                mc_act.append(act)
                mc_sev.append(sev)

        self._disable_dropout()

        act_stack = torch.stack(mc_act, dim=0)  # [N, B, 7]
        sev_stack = torch.stack(mc_sev, dim=0)  # [N, B, 7]

        act_mean = act_stack.mean(dim=0)
        act_std = act_stack.std(dim=0)
        sev_mean = sev_stack.mean(dim=0)
        sev_std = sev_stack.std(dim=0)

        return AOPPredictionOutput(
            activation_prob=act_mean,
            severity_score=sev_mean,
            confidence=conf,
            alert_levels=self._compute_alert_levels(act_mean),
            attention_weights=attn,
            mc_activation_mean=act_mean,
            mc_activation_std=act_std,
            mc_severity_mean=sev_mean,
            mc_severity_std=sev_std,
        )

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    @staticmethod
    def compute_loss(
        output: AOPPredictionOutput,
        targets: Dict[str, torch.Tensor],
        loss_weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute multi-task training loss.

        Parameters
        ----------
        output : AOPPredictionOutput
            Model predictions.
        targets : dict
            Expected keys (all optional, missing keys are skipped):

            - ``"activation"`` : Tensor[B, 7] -- binary activation labels.
            - ``"severity"`` : Tensor[B, 7] -- severity scores (0-1).

        loss_weights : dict, optional
            Per-task weighting factors.

        Returns
        -------
        total_loss : scalar Tensor
        per_task : Dict[str, Tensor]
        """
        if loss_weights is None:
            loss_weights = {
                "activation": 1.0,
                "severity": 0.5,
                "confidence_calibration": 0.1,
            }

        losses: Dict[str, torch.Tensor] = {}
        device = output.activation_prob.device

        if "activation" in targets:
            losses["activation"] = F.binary_cross_entropy(
                output.activation_prob,
                targets["activation"].float(),
            )

        if "severity" in targets:
            losses["severity"] = F.mse_loss(
                output.severity_score,
                targets["severity"].float(),
            )

        # Confidence calibration: confidence should correlate with accuracy
        if "activation" in targets:
            # Correct predictions have error = 0; wrong ones have error close to 1
            pred_binary = (output.activation_prob > 0.5).float()
            accuracy = (pred_binary == targets["activation"].float()).float()
            # Confidence should match accuracy
            losses["confidence_calibration"] = F.mse_loss(
                output.confidence,
                accuracy,
            )

        total = torch.tensor(0.0, device=device)
        for key, loss_val in losses.items():
            w = loss_weights.get(key, 1.0)
            total = total + w * loss_val

        return total, losses

    # ------------------------------------------------------------------
    # Interpretability: top environmental drivers per AOP
    # ------------------------------------------------------------------

    def get_top_drivers(
        self,
        output: AOPPredictionOutput,
        top_k: int = 5,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Extract the top environmental drivers for each active AOP.

        Uses the attention weights to identify which environmental
        features most strongly activate each AOP pathway.

        Parameters
        ----------
        output : AOPPredictionOutput
            Model output (single sample, B=1).
        top_k : int
            Number of top drivers to return per AOP.

        Returns
        -------
        Dict[str, List[Tuple[str, float]]]
            Maps AOP name to list of (feature_name, attention_score).
        """
        all_features = list(SENSOR_FEATURE_NAMES) + list(SATELLITE_FEATURE_NAMES)
        attn = output.attention_weights[0]  # [7, E]

        drivers: Dict[str, List[Tuple[str, float]]] = {}

        for k in range(NUM_AOP_CATEGORIES):
            aop_name = AOP_NAMES[k]
            aop_attn = attn[k]  # [E]

            # Get top-k feature indices
            num_features = min(len(all_features), aop_attn.size(0))
            top_vals, top_idx = torch.topk(
                aop_attn[:num_features], min(top_k, num_features),
            )

            driver_list: List[Tuple[str, float]] = []
            for i in range(top_idx.size(0)):
                idx = top_idx[i].item()
                score = top_vals[i].item()
                if idx < len(all_features):
                    driver_list.append((all_features[idx], score))

            drivers[aop_name] = driver_list

        return drivers
