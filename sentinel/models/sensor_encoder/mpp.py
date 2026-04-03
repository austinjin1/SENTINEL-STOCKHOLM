"""Masked Parameter Prediction (MPP) for self-supervised sensor pretraining.

Analogous to BERT's masked language modeling but for multivariate water
quality time series. Randomly masks one complete parameter within a
contiguous temporal window and trains the TCN to reconstruct the masked
values from remaining parameters and unmasked temporal context.
"""

from __future__ import annotations

import random
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tcn import SensorTCN, NUM_PARAMETERS, LOOKBACK_STEPS

# Parameter names and default loss weights (DO weighted higher)
PARAMETER_NAMES = ["pH", "DO", "turbidity", "conductivity", "temperature", "ORP"]
DEFAULT_PARAM_WEIGHTS = torch.tensor(
    [1.0, 2.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32
)


class MPPHead(nn.Module):
    """Reconstruction head for masked parameter prediction.

    Takes per-timestep features from the TCN and reconstructs masked
    parameter values.

    Args:
        feature_dim: Channel dimension of TCN temporal features.
        num_params: Number of output parameters to reconstruct.
    """

    def __init__(self, feature_dim: int = 256, num_params: int = NUM_PARAMETERS) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim // 2, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(feature_dim // 2, num_params, kernel_size=1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, temporal_features: torch.Tensor) -> torch.Tensor:
        """Reconstruct all parameters from temporal features.

        Args:
            temporal_features: TCN features [B, feature_dim, T].

        Returns:
            Reconstructed values [B, num_params, T].
        """
        return self.head(temporal_features)


class MaskedParameterPrediction(nn.Module):
    """Self-supervised pretraining via masked parameter prediction.

    Training procedure:
    1. For each sample, randomly select one parameter to mask.
    2. Within that parameter, mask a contiguous sub-window (25-75% of lookback).
    3. Set masked values to zero in the input.
    4. Train the TCN + reconstruction head to predict masked values.
    5. Loss: weighted MSE on masked values only.

    Args:
        tcn: The SensorTCN backbone (shared with downstream tasks).
        num_params: Number of sensor parameters.
        feature_dim: TCN output feature dimension.
        param_weights: Per-parameter loss weights. DO is weighted higher.
        mask_ratio_range: (min, max) fraction of temporal window to mask.
    """

    def __init__(
        self,
        tcn: SensorTCN,
        num_params: int = NUM_PARAMETERS,
        feature_dim: int = 256,
        param_weights: Optional[torch.Tensor] = None,
        mask_ratio_range: tuple[float, float] = (0.25, 0.75),
    ) -> None:
        super().__init__()
        self.tcn = tcn
        self.num_params = num_params
        self.mask_ratio_range = mask_ratio_range
        self.reconstruction_head = MPPHead(feature_dim, num_params)

        if param_weights is None:
            param_weights = DEFAULT_PARAM_WEIGHTS
        self.register_buffer("param_weights", param_weights)

    def generate_mask(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate masking pattern: one parameter per sample, contiguous window.

        Args:
            batch_size: Number of samples.
            seq_len: Temporal sequence length.
            device: Target device.

        Returns:
            mask: Boolean tensor [B, P, T] where True = masked.
            masked_params: Index of masked parameter per sample [B].
        """
        mask = torch.zeros(batch_size, self.num_params, seq_len, dtype=torch.bool, device=device)
        masked_params = torch.zeros(batch_size, dtype=torch.long, device=device)

        for i in range(batch_size):
            # Randomly select parameter to mask
            param_idx = random.randint(0, self.num_params - 1)
            masked_params[i] = param_idx

            # Random contiguous window (25-75% of sequence)
            ratio = random.uniform(*self.mask_ratio_range)
            window_len = max(1, int(seq_len * ratio))
            start = random.randint(0, seq_len - window_len)
            mask[i, param_idx, start : start + window_len] = True

        return mask, masked_params

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        masked_params: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass for MPP pretraining.

        Args:
            x: Raw sensor readings [B, T, P].
            mask: Optional precomputed mask [B, P, T]. If None, generated
                automatically.
            masked_params: Optional parameter indices [B]. Required if mask
                is provided.

        Returns:
            Dict with:
                'loss': Weighted MSE loss on masked values.
                'predictions': Reconstructed values [B, P, T].
                'mask': Applied mask [B, P, T].
                'masked_params': Masked parameter indices [B].
        """
        B, T, P = x.shape

        # Generate mask if not provided
        if mask is None:
            mask, masked_params = self.generate_mask(B, T, x.device)
        else:
            assert masked_params is not None, "masked_params required with custom mask"

        # Transpose to [B, P, T] for masking
        x_input = x.transpose(1, 2).clone()  # [B, P, T]

        # Store original values for loss computation
        x_original = x_input.clone()

        # Zero out masked values
        x_input[mask] = 0.0

        # Forward through TCN (expects [B, T, P])
        _, temporal_features = self.tcn(x_input.transpose(1, 2))  # [B, 256, T]

        # Reconstruct all parameters
        predictions = self.reconstruction_head(temporal_features)  # [B, P, T]

        # Compute weighted MSE loss on masked values only
        error = (predictions - x_original) ** 2  # [B, P, T]
        error = error * mask.float()  # Only masked positions

        # Weight by parameter importance
        weights = self.param_weights.view(1, -1, 1).expand_as(error)
        weighted_error = error * weights

        # Mean over masked positions
        num_masked = mask.float().sum().clamp(min=1.0)
        loss = weighted_error.sum() / num_masked

        return {
            "loss": loss,
            "predictions": predictions,
            "mask": mask,
            "masked_params": masked_params,
        }

    @torch.no_grad()
    def predict_masked(
        self,
        x: torch.Tensor,
        param_idx: int,
        mask_start: int,
        mask_end: int,
    ) -> torch.Tensor:
        """Predict values for a specific masked region (inference utility).

        Args:
            x: Raw sensor readings [B, T, P].
            param_idx: Index of parameter to mask and predict.
            mask_start: Start index of mask window.
            mask_end: End index of mask window (exclusive).

        Returns:
            Predicted values for the masked region [B, mask_end - mask_start].
        """
        B, T, P = x.shape
        x_input = x.transpose(1, 2).clone()  # [B, P, T]

        # Zero out the specified region
        x_input[:, param_idx, mask_start:mask_end] = 0.0

        _, temporal_features = self.tcn(x_input.transpose(1, 2))
        predictions = self.reconstruction_head(temporal_features)  # [B, P, T]

        return predictions[:, param_idx, mask_start:mask_end]
