"""Anomaly detection via SSM reconstruction error analysis.

At inference, sequentially masks each parameter and reconstructs it from
the remaining parameters plus temporal context through the AquaSSM backbone.
The pattern of reconstruction errors across parameters enables classification:
- Multi-parameter high error -> real contamination event
- Single-parameter high + others normal -> sensor malfunction
- Slowly increasing single-parameter error -> sensor drift
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aqua_ssm import AquaSSM, NUM_PARAMETERS, OUTPUT_DIM
from .mpp import MPPHead, PARAMETER_NAMES


class ReconstructionAnomalyDetector(nn.Module):
    """Anomaly detection through systematic parameter masking and SSM reconstruction.

    During inference, each parameter is individually masked across the full
    temporal window and reconstructed via the SSM. The per-parameter
    reconstruction error pattern reveals the nature of anomalies.

    Args:
        ssm: Shared AquaSSM backbone (pretrained via MPP).
        reconstruction_head: MPP reconstruction head (pretrained).
        num_params: Number of sensor parameters.
        error_threshold: Z-score threshold for flagging anomalous error.
    """

    def __init__(
        self,
        ssm: AquaSSM,
        reconstruction_head: MPPHead,
        num_params: int = NUM_PARAMETERS,
        error_threshold: float = 3.0,
    ) -> None:
        super().__init__()
        self.ssm = ssm
        self.reconstruction_head = reconstruction_head
        self.num_params = num_params
        self.error_threshold = error_threshold

        # Running error statistics for normalization
        self.register_buffer(
            "param_std", torch.ones(num_params, dtype=torch.float32)
        )
        self.register_buffer(
            "param_mean_error", torch.zeros(num_params, dtype=torch.float32)
        )
        self.register_buffer(
            "running_count", torch.tensor(0, dtype=torch.long)
        )

    def update_statistics(
        self, mean_errors: torch.Tensor, std_errors: torch.Tensor
    ) -> None:
        """Update running error statistics from a training batch.

        Uses exponential moving average for stable online updates.

        Args:
            mean_errors: Per-parameter mean reconstruction errors [B, P].
            std_errors: Per-parameter std of reconstruction errors [B, P].
        """
        batch_mean = mean_errors.mean(dim=0)
        batch_std = std_errors.mean(dim=0)
        n = self.running_count.item()
        m = mean_errors.shape[0]

        if n == 0:
            self.param_mean_error.copy_(batch_mean)
            self.param_std.copy_(batch_std.clamp(min=1e-6))
        else:
            alpha = min(m / (n + m), 0.1)
            self.param_mean_error.lerp_(batch_mean, alpha)
            self.param_std.lerp_(batch_std.clamp(min=1e-6), alpha)

        self.running_count.add_(m)

    @torch.no_grad()
    def compute_reconstruction_errors(
        self,
        x: torch.Tensor,
        delta_ts: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Sequentially mask each parameter and compute reconstruction errors.

        Args:
            x: Sensor readings [B, T, P].
            delta_ts: Time gaps [B, T]. Default 900s.

        Returns:
            Dict with:
                'raw_errors': Per-parameter absolute errors [B, P, T].
                'mean_errors': Time-averaged errors per parameter [B, P].
                'normalized_errors': Z-score normalized errors [B, P].
                'max_errors': Maximum error per parameter [B, P].
                'temporal_features': SSM temporal features from last pass [B, T, D].
        """
        B, T, P = x.shape
        all_errors = torch.zeros(B, P, T, device=x.device)
        last_temporal_features = None

        for param_idx in range(P):
            # Mask entire parameter
            x_masked = x.clone()
            x_masked[:, :, param_idx] = 0.0

            # Create validity mask
            validity_mask = torch.ones(B, T, P, device=x.device)
            validity_mask[:, :, param_idx] = 0.0

            # Forward through SSM
            _, temporal_outputs = self.ssm.forward_with_values(
                x_masked, delta_ts=delta_ts, masks=validity_mask
            )

            # Average scale outputs for reconstruction
            temporal_features = torch.stack(temporal_outputs, dim=0).mean(dim=0)
            last_temporal_features = temporal_features

            # Reconstruct
            predictions = self.reconstruction_head(temporal_features)  # [B, T, P]

            # Absolute error for this parameter
            all_errors[:, param_idx, :] = torch.abs(
                predictions[:, :, param_idx] - x[:, :, param_idx]
            )

        # Aggregate statistics
        mean_errors = all_errors.mean(dim=2)  # [B, P]
        max_errors = all_errors.max(dim=2).values  # [B, P]
        normalized_errors = (
            (mean_errors - self.param_mean_error) / self.param_std.clamp(min=1e-6)
        )

        return {
            "raw_errors": all_errors,
            "mean_errors": mean_errors,
            "normalized_errors": normalized_errors,
            "max_errors": max_errors,
            "temporal_features": last_temporal_features,
        }


class AnomalyClassifier(nn.Module):
    """Classify anomaly type from reconstruction error patterns.

    Uses error statistics to distinguish contamination events, sensor
    malfunctions, and sensor drift.

    Classification logic:
        - Multiple parameters high error -> contamination event
        - Single parameter high + others normal -> sensor malfunction
        - Slowly increasing single-parameter error -> sensor drift

    Args:
        num_params: Number of sensor parameters.
        hidden_dim: Hidden dimension for the classifier MLP.
    """

    ANOMALY_TYPES = [
        "normal",
        "contamination_event",
        "sensor_malfunction",
        "sensor_drift",
    ]

    def __init__(
        self,
        num_params: int = NUM_PARAMETERS,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        # Input: per-param (mean, var, max, onset_rate) + global num_affected
        input_dim = num_params * 4 + 1
        self.num_params = num_params

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim // 2, len(self.ANOMALY_TYPES)),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def compute_error_statistics(
        self,
        raw_errors: torch.Tensor,
        normalized_errors: torch.Tensor,
        threshold: float = 3.0,
    ) -> torch.Tensor:
        """Compute error statistics features for classification.

        Args:
            raw_errors: Per-parameter absolute errors [B, P, T].
            normalized_errors: Z-score normalized mean errors [B, P].
            threshold: Z-score threshold for "high error".

        Returns:
            Feature vector [B, P*4 + 1].
        """
        B, P, T = raw_errors.shape

        mean_err = raw_errors.mean(dim=2)
        var_err = raw_errors.var(dim=2)
        max_err = raw_errors.max(dim=2).values

        # Onset rate (slope of error over time)
        t_axis = torch.arange(T, dtype=torch.float32, device=raw_errors.device)
        t_centered = t_axis - t_axis.mean()
        t_var = (t_centered ** 2).sum().clamp(min=1e-6)
        onset_rate = (raw_errors * t_centered.view(1, 1, -1)).sum(dim=2) / t_var

        num_affected = (
            (normalized_errors > threshold).float().sum(dim=1, keepdim=True)
        )

        features = torch.cat(
            [mean_err, var_err, max_err, onset_rate, num_affected], dim=1
        )
        return features

    def forward(
        self,
        raw_errors: torch.Tensor,
        normalized_errors: torch.Tensor,
        threshold: float = 3.0,
    ) -> dict[str, torch.Tensor]:
        """Classify anomaly type from error patterns.

        Args:
            raw_errors: Per-parameter absolute errors [B, P, T].
            normalized_errors: Z-score normalized mean errors [B, P].
            threshold: Z-score threshold.

        Returns:
            Dict with:
                'logits': Classification logits [B, 4].
                'probabilities': Softmax probabilities [B, 4].
                'predicted_type': Predicted anomaly type index [B].
                'num_affected_params': Count of affected parameters [B].
        """
        features = self.compute_error_statistics(
            raw_errors, normalized_errors, threshold
        )
        logits = self.classifier(features)
        probs = F.softmax(logits, dim=-1)
        predicted = logits.argmax(dim=-1)
        num_affected = (normalized_errors > threshold).float().sum(dim=1)

        return {
            "logits": logits,
            "probabilities": probs,
            "predicted_type": predicted,
            "num_affected_params": num_affected,
        }
