"""Full AquaSSM sensor encoder combining SSM backbone, MPP, anomaly detection, and health.

Provides the complete sensor modality encoder that:
1. Encodes irregular water quality time series via continuous-time multi-scale SSM
2. Supports self-supervised pretraining via masked parameter prediction
3. Detects anomalies via reconstruction error analysis
4. Classifies anomaly type and sensor health
5. Projects to shared 256-dim fusion embedding space
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .aqua_ssm import AquaSSM, NUM_PARAMETERS, OUTPUT_DIM
from .mpp import MaskedParameterPrediction, MPPHead
from .anomaly import ReconstructionAnomalyDetector, AnomalyClassifier
from .sensor_health import SensorHealthSentinel

SHARED_EMBED_DIM = 256


class SensorEncoder(nn.Module):
    """Complete AquaSSM sensor modality encoder for SENTINEL.

    Combines the continuous-time multi-scale SSM backbone with MPP
    self-supervised pretraining, reconstruction-based anomaly detection,
    anomaly classification, and sensor health monitoring.

    Args:
        num_params: Number of sensor parameters. Default 6.
        output_dim: SSM output embedding dimension. Default 256.
        shared_embed_dim: Shared fusion embedding dimension. Default 256.
        hidden_dim_per_scale: SSM hidden state per timescale channel.
        timescales: Characteristic timescales in seconds (list of 8).
        dropout: Dropout rate for SSM.
        error_threshold: Z-score threshold for anomaly detection.
    """

    def __init__(
        self,
        num_params: int = NUM_PARAMETERS,
        output_dim: int = OUTPUT_DIM,
        shared_embed_dim: int = SHARED_EMBED_DIM,
        hidden_dim_per_scale: int = 32,
        timescales: Optional[list[float]] = None,
        dropout: float = 0.1,
        error_threshold: float = 3.0,
    ) -> None:
        super().__init__()

        # AquaSSM backbone
        self.ssm = AquaSSM(
            num_params=num_params,
            output_dim=output_dim,
            hidden_dim_per_scale=hidden_dim_per_scale,
            timescales=timescales,
            dropout=dropout,
        )

        # MPP reconstruction head (shared between pretraining and anomaly detection)
        self.reconstruction_head = MPPHead(
            feature_dim=output_dim, num_params=num_params
        )

        # MPP pretraining module
        self.mpp = MaskedParameterPrediction(
            ssm=self.ssm,
            num_params=num_params,
            feature_dim=output_dim,
        )
        # Share reconstruction head weights
        self.mpp.reconstruction_head = self.reconstruction_head

        # Anomaly detection
        self.anomaly_detector = ReconstructionAnomalyDetector(
            ssm=self.ssm,
            reconstruction_head=self.reconstruction_head,
            num_params=num_params,
            error_threshold=error_threshold,
        )

        # Anomaly classification
        self.anomaly_classifier = AnomalyClassifier(num_params=num_params)

        # Sensor health sentinel
        self.sensor_health = SensorHealthSentinel(num_params=num_params)

        # Projection to shared embedding space
        # Pattern: Linear(256,256) -> GELU -> LayerNorm(256) -> Linear(256,256) -> LayerNorm(256)
        self.projection = nn.Sequential(
            nn.Linear(output_dim, shared_embed_dim),
            nn.GELU(),
            nn.LayerNorm(shared_embed_dim),
            nn.Linear(shared_embed_dim, shared_embed_dim),
            nn.LayerNorm(shared_embed_dim),
        )

        self._init_projection()

    def _init_projection(self) -> None:
        """Xavier init for all Linear layers in projection."""
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
        delta_ts: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass for MPP pretraining.

        Args:
            x: Sensor readings [B, T, P].
            mask: Optional precomputed mask [B, T, P].
            masked_params: Optional masked parameter indices [B].
            delta_ts: Optional time gaps [B, T]. Default 900s.

        Returns:
            MPP loss and predictions dict.
        """
        return self.mpp(x, mask, masked_params, delta_ts=delta_ts)

    def forward(
        self,
        x: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
        delta_ts: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
        compute_anomaly: bool = True,
    ) -> dict[str, torch.Tensor | dict]:
        """Full forward pass through the AquaSSM sensor encoder.

        Args:
            x: Sensor readings [B, T, P=6].
            timestamps: Absolute timestamps [B, T] (seconds). Optional.
            delta_ts: Time gaps between observations [B, T] (seconds).
                Default 900s (15 min regular intervals).
            masks: Per-parameter validity masks [B, T, P]. Default all-valid.
            compute_anomaly: Whether to run anomaly detection (expensive
                at inference due to P sequential forward passes).

        Returns:
            Dict with:
                'embedding': Projected embedding [B, 256] for fusion.
                'fusion_embedding': Same as embedding [B, 256].
                'ssm_embedding': Raw SSM output before projection [B, 256].
                'anomaly_scores': Dict of anomaly detection results (if enabled).
                'sensor_health': Dict of per-sensor health status (if enabled).
        """
        B, T, P = x.shape
        device = x.device

        # Default timestamps and delta_ts for regular 15-min intervals
        if timestamps is None:
            timestamps = torch.zeros(B, T, device=device)
        if delta_ts is None:
            delta_ts = torch.full((B, T), 900.0, device=device)

        # SSM forward pass
        ssm_result = self.ssm(timestamps, x, delta_ts, masks)
        ssm_embedding = ssm_result["embedding"]  # [B, 256]

        # Project to shared space
        embedding = self.projection(ssm_embedding)  # [B, 256]

        result: dict[str, torch.Tensor | dict] = {
            "embedding": embedding,
            "fusion_embedding": embedding,
            "ssm_embedding": ssm_embedding,
        }

        if compute_anomaly:
            # Reconstruction-based anomaly detection
            error_results = self.anomaly_detector.compute_reconstruction_errors(
                x, delta_ts=delta_ts
            )

            # Classify anomaly type
            anomaly_class = self.anomaly_classifier(
                error_results["raw_errors"],
                error_results["normalized_errors"],
            )

            # Sensor health assessment (needs SSM temporal features)
            temporal_features = error_results["temporal_features"]  # [B, T, D]
            health = self.sensor_health(
                error_results["raw_errors"],
                temporal_features,
            )

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
                "anomaly_weights": health["anomaly_weights"],
            }
        else:
            result["anomaly_scores"] = {}
            result["sensor_health"] = {}

        return result
