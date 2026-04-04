"""Sensor Health Sentinel: auxiliary head for sensor status classification.

Classifies each sensor's operational status from SSM hidden states and
per-parameter reconstruction errors. When a sensor is flagged as unhealthy,
its anomaly contribution is down-weighted to prevent false alarms.

Classes: normal, drift, fouling, failure, calibration_needed
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aqua_ssm import NUM_PARAMETERS

# Health status labels
HEALTH_CLASSES = [
    "normal",
    "drift",
    "fouling",
    "failure",
    "calibration_needed",
]
NUM_HEALTH_CLASSES = len(HEALTH_CLASSES)


class SensorHealthSentinel(nn.Module):
    """MLP classifier for per-sensor health status.

    Combines SSM hidden state statistics with per-parameter reconstruction
    error statistics to classify sensor operational health.

    Input features per sensor (10 dims):
        - Mean reconstruction error (1)
        - Std of reconstruction error (1)
        - Max reconstruction error (1)
        - Error trend (slope over time) (1)
        - SSM hidden state mean (1)
        - SSM hidden state std (1)
        - Error autocorrelation at lag-1 (1)
        - Fraction of timesteps above 2-sigma (1)
        - Fraction of timesteps above 3-sigma (1)
        - Error kurtosis (1)

    Args:
        num_params: Number of sensor parameters.
        input_features_per_sensor: Number of input features per sensor.
        hidden_dim: MLP hidden dimension.
        num_classes: Number of health status classes.
    """

    def __init__(
        self,
        num_params: int = NUM_PARAMETERS,
        input_features_per_sensor: int = 10,
        hidden_dim: int = 64,
        num_classes: int = NUM_HEALTH_CLASSES,
    ) -> None:
        super().__init__()
        self.num_params = num_params
        self.num_classes = num_classes
        self.input_features_per_sensor = input_features_per_sensor

        # Shared MLP applied independently to each sensor
        self.classifier = nn.Sequential(
            nn.Linear(input_features_per_sensor, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        # Learnable anomaly down-weighting per health status
        # normal=1.0, drift=0.5, fouling=0.3, failure=0.1, calibration=0.2
        self.register_buffer(
            "health_weight_prior",
            torch.tensor([1.0, 0.5, 0.3, 0.1, 0.2], dtype=torch.float32),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def compute_error_features(
        self,
        raw_errors: torch.Tensor,
        ssm_hidden: torch.Tensor,
    ) -> torch.Tensor:
        """Extract per-sensor statistical features from errors and hidden states.

        Args:
            raw_errors: Per-parameter reconstruction errors [B, P, T].
            ssm_hidden: SSM temporal hidden features [B, T, D].

        Returns:
            Per-sensor feature vectors [B, P, input_features_per_sensor].
        """
        B, P, T = raw_errors.shape

        # Error statistics
        mean_err = raw_errors.mean(dim=2)  # [B, P]
        std_err = raw_errors.std(dim=2)  # [B, P]
        max_err = raw_errors.max(dim=2).values  # [B, P]

        # Error trend (slope via linear regression)
        t_axis = torch.arange(T, dtype=torch.float32, device=raw_errors.device)
        t_centered = t_axis - t_axis.mean()
        t_var = (t_centered ** 2).sum().clamp(min=1e-6)
        trend = (raw_errors * t_centered.view(1, 1, -1)).sum(dim=2) / t_var  # [B, P]

        # SSM hidden state statistics (averaged over time, projected per param)
        # Use mean and std of temporal features as global context
        ssm_mean = ssm_hidden.mean(dim=1)  # [B, D]
        ssm_std = ssm_hidden.std(dim=1)  # [B, D]
        # Reduce to scalar per param by taking slices
        D = ssm_hidden.shape[-1]
        step = max(1, D // P)
        ssm_mean_per_param = ssm_mean[:, :P * step:step][:, :P]  # [B, P]
        ssm_std_per_param = ssm_std[:, :P * step:step][:, :P]  # [B, P]

        # Error autocorrelation at lag-1
        if T > 1:
            err_centered = raw_errors - mean_err.unsqueeze(-1)
            autocorr = (err_centered[:, :, 1:] * err_centered[:, :, :-1]).mean(dim=2)
            autocorr = autocorr / (std_err ** 2).clamp(min=1e-6)  # [B, P]
        else:
            autocorr = torch.zeros(B, P, device=raw_errors.device)

        # Fraction above thresholds
        threshold_2sig = mean_err.unsqueeze(-1) + 2 * std_err.unsqueeze(-1)
        threshold_3sig = mean_err.unsqueeze(-1) + 3 * std_err.unsqueeze(-1)
        frac_2sig = (raw_errors > threshold_2sig).float().mean(dim=2)  # [B, P]
        frac_3sig = (raw_errors > threshold_3sig).float().mean(dim=2)  # [B, P]

        # Kurtosis
        err_centered = raw_errors - mean_err.unsqueeze(-1)
        kurtosis = (err_centered ** 4).mean(dim=2) / (std_err ** 4).clamp(min=1e-6) - 3.0  # [B, P]

        # Stack all features: [B, P, 10]
        features = torch.stack([
            mean_err, std_err, max_err, trend,
            ssm_mean_per_param, ssm_std_per_param,
            autocorr, frac_2sig, frac_3sig, kurtosis,
        ], dim=-1)

        return features

    def forward(
        self,
        raw_errors: torch.Tensor,
        ssm_hidden: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Classify health status for each sensor.

        Args:
            raw_errors: Per-parameter reconstruction errors [B, P, T].
            ssm_hidden: SSM temporal features [B, T, D].

        Returns:
            Dict with:
                'health_logits': Per-sensor logits [B, P, num_classes].
                'health_probs': Per-sensor probabilities [B, P, num_classes].
                'health_status': Per-sensor predicted class index [B, P].
                'anomaly_weights': Per-sensor anomaly down-weights [B, P].
        """
        B, P, T = raw_errors.shape

        features = self.compute_error_features(raw_errors, ssm_hidden)  # [B, P, 10]

        # Apply shared MLP to each sensor independently
        logits = self.classifier(features.view(B * P, -1)).view(B, P, -1)  # [B, P, C]
        probs = F.softmax(logits, dim=-1)
        status = logits.argmax(dim=-1)  # [B, P]

        # Compute anomaly down-weights based on health status
        # Soft weighting: weighted average of prior weights by health probabilities
        anomaly_weights = (probs * self.health_weight_prior.view(1, 1, -1)).sum(dim=-1)  # [B, P]

        return {
            "health_logits": logits,
            "health_probs": probs,
            "health_status": status,
            "anomaly_weights": anomaly_weights,
        }
