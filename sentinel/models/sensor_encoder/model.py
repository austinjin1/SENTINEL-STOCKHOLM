"""Full sensor encoder combining TCN, MPP pretraining, and anomaly detection.

Provides the complete sensor modality encoder that:
1. Encodes 7-day, 6-parameter time series via causal TCN
2. Supports self-supervised pretraining via masked parameter prediction
3. Detects anomalies via reconstruction error analysis
4. Classifies anomaly type and sensor health
5. Projects to shared 256-dim fusion embedding space
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .tcn import SensorTCN, NUM_PARAMETERS, OUTPUT_DIM
from .mpp import MaskedParameterPrediction, MPPHead
from .anomaly import (
    ReconstructionAnomalyDetector,
    AnomalyClassifier,
    SensorHealthClassifier,
)

SHARED_EMBED_DIM = 256


class SensorEncoder(nn.Module):
    """Complete sensor modality encoder for SENTINEL.

    Combines the TCN backbone with MPP self-supervised pretraining,
    reconstruction-based anomaly detection, anomaly classification,
    and sensor health monitoring.

    Args:
        num_params: Number of sensor parameters. Default 6.
        tcn_channels: Channel widths per TCN layer.
        tcn_kernel_size: TCN convolution kernel size.
        tcn_dilations: TCN dilation factors.
        dropout: Dropout rate for TCN.
        output_dim: TCN output embedding dimension.
        shared_embed_dim: Shared fusion embedding dimension.
        error_threshold: Z-score threshold for anomaly detection.
    """

    def __init__(
        self,
        num_params: int = NUM_PARAMETERS,
        tcn_channels: list[int] | None = None,
        tcn_kernel_size: int = 7,
        tcn_dilations: list[int] | None = None,
        dropout: float = 0.2,
        output_dim: int = OUTPUT_DIM,
        shared_embed_dim: int = SHARED_EMBED_DIM,
        error_threshold: float = 3.0,
    ) -> None:
        super().__init__()

        # TCN backbone
        self.tcn = SensorTCN(
            in_channels=num_params,
            channels=tcn_channels,
            kernel_size=tcn_kernel_size,
            dilations=tcn_dilations,
            dropout=dropout,
            output_dim=output_dim,
        )

        # MPP reconstruction head (shared between pretraining and anomaly detection)
        self.reconstruction_head = MPPHead(
            feature_dim=256, num_params=num_params
        )

        # MPP pretraining module
        self.mpp = MaskedParameterPrediction(
            tcn=self.tcn,
            num_params=num_params,
            feature_dim=256,
        )
        # Point MPP's reconstruction head to ours (shared weights)
        self.mpp.reconstruction_head = self.reconstruction_head

        # Anomaly detection
        self.anomaly_detector = ReconstructionAnomalyDetector(
            tcn=self.tcn,
            reconstruction_head=self.reconstruction_head,
            num_params=num_params,
            error_threshold=error_threshold,
        )

        # Anomaly classification
        self.anomaly_classifier = AnomalyClassifier(num_params=num_params)

        # Sensor health
        self.sensor_health = SensorHealthClassifier(num_params=num_params)

        # Projection to shared embedding space
        self.projection = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
            nn.Linear(output_dim, shared_embed_dim),
            nn.LayerNorm(shared_embed_dim),
        )

        self._init_projection()

    def _init_projection(self) -> None:
        for m in self.projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_pretrain(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        masked_params: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass for MPP pretraining.

        Args:
            x: Sensor readings [B, T, P].
            mask: Optional precomputed mask [B, P, T].
            masked_params: Optional masked parameter indices [B].

        Returns:
            MPP loss and predictions dict.
        """
        return self.mpp(x, mask, masked_params)

    def forward(
        self,
        x: torch.Tensor,
        compute_anomaly: bool = True,
    ) -> dict[str, torch.Tensor | dict]:
        """Full forward pass through the sensor encoder.

        Args:
            x: Sensor readings [B, T=672, P=6].
            compute_anomaly: Whether to run anomaly detection (expensive
                at inference due to P sequential forward passes).

        Returns:
            Dict with:
                'embedding': Projected embedding [B, 256] for fusion.
                'anomaly_scores': Dict of anomaly detection results (if enabled).
                'sensor_health': Dict of per-sensor health status (if enabled).
                'tcn_embedding': Raw TCN embedding [B, 256].
        """
        # TCN forward pass
        tcn_embedding, temporal_features = self.tcn(x)

        # Project to shared space
        embedding = self.projection(tcn_embedding)

        result: dict[str, torch.Tensor | dict] = {
            "embedding": embedding,
            "tcn_embedding": tcn_embedding,
            "fusion_embedding": embedding,
        }

        if compute_anomaly:
            # Reconstruction-based anomaly detection
            error_results = self.anomaly_detector.compute_reconstruction_errors(x)

            # Classify anomaly type
            anomaly_class = self.anomaly_classifier(
                error_results["raw_errors"],
                error_results["normalized_errors"],
            )

            # Sensor health assessment
            health = self.sensor_health(error_results["raw_errors"])

            result["anomaly_scores"] = {
                "mean_errors": error_results["mean_errors"],
                "normalized_errors": error_results["normalized_errors"],
                "max_errors": error_results["max_errors"],
                "anomaly_type": anomaly_class["predicted_type"],
                "anomaly_probs": anomaly_class["probabilities"],
                "num_affected_params": anomaly_class["num_affected_params"],
            }
            result["sensor_health"] = {
                "health_status": health["health_status"],
                "health_probs": health["health_probs"],
            }
        else:
            result["anomaly_scores"] = {}
            result["sensor_health"] = {}

        return result
