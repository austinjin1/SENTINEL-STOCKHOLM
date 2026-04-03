"""Anomaly detection via reconstruction error analysis for sensor data.

At inference, sequentially masks each parameter and reconstructs it from
the remaining parameters plus temporal context. The pattern of reconstruction
errors across parameters enables classification of anomaly types:
- Multi-parameter high error -> real contamination event
- Single-parameter high + others normal -> sensor malfunction
- Slowly increasing single-parameter error -> sensor drift
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tcn import SensorTCN, NUM_PARAMETERS
from .mpp import MPPHead, PARAMETER_NAMES


class ReconstructionAnomalyDetector(nn.Module):
    """Anomaly detection through systematic parameter masking and reconstruction.

    During inference, each parameter is individually masked across the full
    temporal window and reconstructed. The reconstruction error pattern
    reveals the nature of anomalies.

    Args:
        tcn: Shared TCN backbone (pretrained via MPP).
        reconstruction_head: MPP reconstruction head (pretrained).
        num_params: Number of sensor parameters.
        feature_dim: TCN output feature dimension.
        error_threshold: Z-score threshold for flagging anomalous error.
    """

    def __init__(
        self,
        tcn: SensorTCN,
        reconstruction_head: MPPHead,
        num_params: int = NUM_PARAMETERS,
        feature_dim: int = 256,
        error_threshold: float = 3.0,
    ) -> None:
        super().__init__()
        self.tcn = tcn
        self.reconstruction_head = reconstruction_head
        self.num_params = num_params
        self.error_threshold = error_threshold

        # Learned per-parameter standard deviation from training distribution
        # Initialized to ones; updated via running statistics during training
        self.register_buffer(
            "param_std", torch.ones(num_params, dtype=torch.float32)
        )
        self.register_buffer(
            "param_mean_error", torch.zeros(num_params, dtype=torch.float32)
        )
        # Running count for online statistics updates
        self.register_buffer(
            "running_count", torch.tensor(0, dtype=torch.long)
        )

    def update_statistics(
        self, mean_errors: torch.Tensor, std_errors: torch.Tensor
    ) -> None:
        """Update running error statistics from a training batch.

        Uses Welford's online algorithm for numerically stable updates.

        Args:
            mean_errors: Per-parameter mean reconstruction errors [B, P].
            std_errors: Per-parameter std of reconstruction errors [B, P].
        """
        batch_mean = mean_errors.mean(dim=0)  # [P]
        batch_std = std_errors.mean(dim=0)  # [P]
        n = self.running_count.item()
        m = mean_errors.shape[0]

        if n == 0:
            self.param_mean_error.copy_(batch_mean)
            self.param_std.copy_(batch_std.clamp(min=1e-6))
        else:
            # Exponential moving average
            alpha = min(m / (n + m), 0.1)
            self.param_mean_error.lerp_(batch_mean, alpha)
            self.param_std.lerp_(batch_std.clamp(min=1e-6), alpha)

        self.running_count.add_(m)

    @torch.no_grad()
    def compute_reconstruction_errors(
        self, x: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Sequentially mask each parameter and compute reconstruction errors.

        Args:
            x: Sensor readings [B, T, P].

        Returns:
            Dict with:
                'raw_errors': Per-parameter absolute errors [B, P, T].
                'mean_errors': Time-averaged errors per parameter [B, P].
                'normalized_errors': Errors normalized by learned std [B, P].
                'max_errors': Maximum error per parameter [B, P].
        """
        B, T, P = x.shape
        all_errors = torch.zeros(B, P, T, device=x.device)

        for param_idx in range(P):
            # Mask entire parameter
            x_masked = x.transpose(1, 2).clone()  # [B, P, T]
            x_masked[:, param_idx, :] = 0.0

            # Reconstruct
            _, temporal_features = self.tcn(x_masked.transpose(1, 2))
            predictions = self.reconstruction_head(temporal_features)  # [B, P, T]

            # Absolute error for this parameter
            x_orig = x.transpose(1, 2)  # [B, P, T]
            all_errors[:, param_idx, :] = torch.abs(
                predictions[:, param_idx, :] - x_orig[:, param_idx, :]
            )

        # Aggregate statistics
        mean_errors = all_errors.mean(dim=2)  # [B, P]
        max_errors = all_errors.max(dim=2).values  # [B, P]
        normalized_errors = (mean_errors - self.param_mean_error) / self.param_std.clamp(min=1e-6)

        return {
            "raw_errors": all_errors,
            "mean_errors": mean_errors,
            "normalized_errors": normalized_errors,
            "max_errors": max_errors,
        }


class AnomalyClassifier(nn.Module):
    """Classify anomaly type from reconstruction error patterns.

    Uses error statistics (mean, variance, onset rate, number of affected
    parameters) to distinguish between contamination events, sensor
    malfunctions, and sensor drift.

    Classification logic:
        - Multiple parameters high error -> contamination event
        - Single parameter high + others normal -> sensor malfunction
        - Slowly increasing single-parameter error -> sensor drift

    Args:
        num_params: Number of sensor parameters.
        hidden_dim: Hidden dimension for the classifier MLP.
    """

    # Anomaly type labels
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
        # Input features per parameter: mean_error, var_error, onset_rate, max_error
        # Plus global: num_affected_params
        input_dim = num_params * 4 + 1
        self.num_params = num_params

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
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
            Feature vector [B, P*4 + 1] for the classifier.
        """
        B, P, T = raw_errors.shape

        # Per-parameter statistics
        mean_err = raw_errors.mean(dim=2)  # [B, P]
        var_err = raw_errors.var(dim=2)  # [B, P]
        max_err = raw_errors.max(dim=2).values  # [B, P]

        # Onset rate: slope of error over time (positive = increasing)
        # Simple linear regression slope over time
        t_axis = torch.arange(T, dtype=torch.float32, device=raw_errors.device)
        t_mean = t_axis.mean()
        t_centered = t_axis - t_mean
        t_var = (t_centered ** 2).sum()

        # [B, P, T] * [T] -> sum over T -> [B, P]
        onset_rate = (raw_errors * t_centered.view(1, 1, -1)).sum(dim=2) / t_var.clamp(min=1e-6)

        # Number of affected parameters (high normalized error)
        num_affected = (normalized_errors > threshold).float().sum(dim=1, keepdim=True)  # [B, 1]

        # Concatenate features
        features = torch.cat(
            [mean_err, var_err, max_err, onset_rate, num_affected], dim=1
        )  # [B, P*4 + 1]

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
            threshold: Z-score threshold for anomaly detection.

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


class SensorHealthClassifier(nn.Module):
    """MLP classifier for per-sensor health status.

    Evaluates each sensor's health based on its error statistics
    (mean, variance, onset rate, number of co-affected parameters).

    Args:
        num_params: Number of sensor parameters.
        hidden_dim: MLP hidden dimension.
    """

    HEALTH_STATES = ["healthy", "degraded", "failing", "offline"]

    def __init__(
        self,
        num_params: int = NUM_PARAMETERS,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        # Per-sensor features: mean_error, var_error, onset_rate, max_error
        input_dim = 4
        self.num_params = num_params

        # Shared MLP applied independently to each sensor
        self.health_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, len(self.HEALTH_STATES)),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, raw_errors: torch.Tensor) -> dict[str, torch.Tensor]:
        """Classify health status for each sensor.

        Args:
            raw_errors: Per-parameter absolute errors [B, P, T].

        Returns:
            Dict with:
                'health_logits': Per-sensor health logits [B, P, 4].
                'health_probs': Per-sensor health probabilities [B, P, 4].
                'health_status': Per-sensor predicted status index [B, P].
        """
        B, P, T = raw_errors.shape

        # Per-sensor statistics
        mean_err = raw_errors.mean(dim=2)  # [B, P]
        var_err = raw_errors.var(dim=2)  # [B, P]
        max_err = raw_errors.max(dim=2).values  # [B, P]

        # Onset rate
        t_axis = torch.arange(T, dtype=torch.float32, device=raw_errors.device)
        t_centered = t_axis - t_axis.mean()
        t_var = (t_centered ** 2).sum().clamp(min=1e-6)
        onset_rate = (raw_errors * t_centered.view(1, 1, -1)).sum(dim=2) / t_var

        # Stack per-sensor features: [B, P, 4]
        features = torch.stack([mean_err, var_err, max_err, onset_rate], dim=-1)

        # Apply MLP to each sensor independently
        # Reshape to [B*P, 4], apply MLP, reshape back
        logits = self.health_mlp(features.view(B * P, -1)).view(B, P, -1)
        probs = F.softmax(logits, dim=-1)
        status = logits.argmax(dim=-1)

        return {
            "health_logits": logits,
            "health_probs": probs,
            "health_status": status,
        }
