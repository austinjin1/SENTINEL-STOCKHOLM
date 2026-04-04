"""Masked Parameter Prediction (MPP) for self-supervised AquaSSM pretraining.

Analogous to BERT's masked language modeling but for multivariate water
quality time series. Randomly masks one complete parameter within a
contiguous temporal window and trains the SSM to reconstruct the masked
values from remaining parameters and temporal context.

Adapted from TCN-based v1 to work with AquaSSM backbone.
"""

from __future__ import annotations

import random
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aqua_ssm import AquaSSM, NUM_PARAMETERS, OUTPUT_DIM

# Parameter names and default loss weights (DO weighted higher)
PARAMETER_NAMES = ["pH", "DO", "turbidity", "conductivity", "temperature", "ORP"]
DEFAULT_PARAM_WEIGHTS = torch.tensor(
    [1.0, 2.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32
)


class MPPHead(nn.Module):
    """Reconstruction head for masked parameter prediction.

    Takes per-timestep SSM features and reconstructs masked parameter values.
    Uses a small MLP per timestep (implemented as 1D convolutions for efficiency).

    Args:
        feature_dim: Dimension of SSM temporal features.
        num_params: Number of output parameters to reconstruct.
    """

    def __init__(
        self,
        feature_dim: int = OUTPUT_DIM,
        num_params: int = NUM_PARAMETERS,
    ) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, num_params),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, temporal_features: torch.Tensor) -> torch.Tensor:
        """Reconstruct all parameters from temporal features.

        Args:
            temporal_features: SSM features [B, T, feature_dim].

        Returns:
            Reconstructed values [B, T, num_params].
        """
        return self.head(temporal_features)  # [B, T, num_params]


class MaskedParameterPrediction(nn.Module):
    """Self-supervised pretraining via masked parameter prediction on AquaSSM.

    Training procedure:
    1. For each sample, randomly select one parameter to mask.
    2. Within that parameter, mask a contiguous sub-window (25-75% of sequence).
    3. Set masked values to zero in the input.
    4. Train the SSM + reconstruction head to predict masked values.
    5. Loss: weighted MSE on masked values only (DO weighted 2x).

    Args:
        ssm: The AquaSSM backbone (shared with downstream tasks).
        num_params: Number of sensor parameters.
        feature_dim: SSM output feature dimension.
        param_weights: Per-parameter loss weights.
        mask_ratio_range: (min, max) fraction of temporal window to mask.
    """

    def __init__(
        self,
        ssm: AquaSSM,
        num_params: int = NUM_PARAMETERS,
        feature_dim: int = OUTPUT_DIM,
        param_weights: Optional[torch.Tensor] = None,
        mask_ratio_range: tuple[float, float] = (0.25, 0.75),
    ) -> None:
        super().__init__()
        self.ssm = ssm
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
            mask: Boolean tensor [B, T, P] where True = masked.
            masked_params: Index of masked parameter per sample [B].
        """
        mask = torch.zeros(
            batch_size, seq_len, self.num_params, dtype=torch.bool, device=device
        )
        masked_params = torch.zeros(batch_size, dtype=torch.long, device=device)

        for i in range(batch_size):
            param_idx = random.randint(0, self.num_params - 1)
            masked_params[i] = param_idx

            ratio = random.uniform(*self.mask_ratio_range)
            window_len = max(1, int(seq_len * ratio))
            start = random.randint(0, seq_len - window_len)
            mask[i, start : start + window_len, param_idx] = True

        return mask, masked_params

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        masked_params: Optional[torch.Tensor] = None,
        delta_ts: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass for MPP pretraining.

        Args:
            x: Raw sensor readings [B, T, P].
            mask: Optional precomputed mask [B, T, P]. If None, generated.
            masked_params: Optional parameter indices [B].
            delta_ts: Optional time gaps [B, T]. Default 900s.

        Returns:
            Dict with:
                'loss': Weighted MSE loss on masked values.
                'predictions': Reconstructed values [B, T, P].
                'mask': Applied mask [B, T, P].
                'masked_params': Masked parameter indices [B].
        """
        B, T, P = x.shape

        # Generate mask if not provided
        if mask is None:
            mask, masked_params = self.generate_mask(B, T, x.device)
        else:
            assert masked_params is not None, "masked_params required with custom mask"

        # Store original values
        x_original = x.clone()  # [B, T, P]

        # Zero out masked values
        x_input = x.clone()
        x_input[mask] = 0.0

        # Create validity mask (inverse of masking)
        validity_mask = (~mask).float()  # [B, T, P]

        # Forward through SSM
        _, temporal_outputs = self.ssm.forward_with_values(
            x_input, delta_ts=delta_ts, masks=validity_mask
        )

        # Use first scale's temporal output for reconstruction
        # Average across all scales for richer features
        temporal_features = torch.stack(
            [out for out in temporal_outputs], dim=0
        ).mean(dim=0)  # [B, T, output_dim]

        # Reconstruct all parameters
        predictions = self.reconstruction_head(temporal_features)  # [B, T, P]

        # Compute weighted MSE loss on masked values only
        error = (predictions - x_original) ** 2  # [B, T, P]
        error = error * mask.float()

        # Weight by parameter importance
        weights = self.param_weights.view(1, 1, -1).expand_as(error)
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
        delta_ts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict values for a specific masked region (inference utility).

        Args:
            x: Raw sensor readings [B, T, P].
            param_idx: Index of parameter to mask and predict.
            mask_start: Start index of mask window.
            mask_end: End index of mask window (exclusive).
            delta_ts: Optional time gaps [B, T].

        Returns:
            Predicted values for the masked region [B, mask_end - mask_start].
        """
        B, T, P = x.shape
        x_input = x.clone()
        x_input[:, mask_start:mask_end, param_idx] = 0.0

        validity_mask = torch.ones(B, T, P, device=x.device)
        validity_mask[:, mask_start:mask_end, param_idx] = 0.0

        _, temporal_outputs = self.ssm.forward_with_values(
            x_input, delta_ts=delta_ts, masks=validity_mask
        )

        temporal_features = torch.stack(temporal_outputs, dim=0).mean(dim=0)
        predictions = self.reconstruction_head(temporal_features)

        return predictions[:, mask_start:mask_end, param_idx]
