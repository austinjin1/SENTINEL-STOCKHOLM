"""Temporal Convolutional Network for water quality sensor time series.

Processes 7 days of 15-minute interval readings (T=672) across 6 parameters
(pH, DO, turbidity, conductivity, temperature, ORP) using causal dilated
convolutions to produce a fixed 256-dim embedding.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Input configuration
LOOKBACK_STEPS = 672  # 7 days * 24 hours * (60/15) = 672 timesteps
NUM_PARAMETERS = 6    # pH, DO, turbidity, conductivity, temperature, ORP
OUTPUT_DIM = 256


class CausalConv1d(nn.Module):
    """Causal 1D convolution that prevents future information leakage.

    Uses left-padding to ensure the output at time t only depends on
    inputs at times <= t.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size.
        dilation: Dilation factor.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=0,  # Manual causal padding
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply causal convolution.

        Args:
            x: Input tensor [B, C, T].

        Returns:
            Output tensor [B, C_out, T] (same temporal length).
        """
        # Left-pad to maintain causality
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class TCNBlock(nn.Module):
    """Single TCN residual block with dilated causal convolutions.

    Architecture per block:
        CausalConv -> BatchNorm -> ELU -> Dropout ->
        CausalConv -> BatchNorm -> ELU -> Dropout + Residual

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size.
        dilation: Dilation factor.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        dilation: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            CausalConv1d(in_channels, out_channels, kernel_size, dilation),
            nn.BatchNorm1d(out_channels),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
            CausalConv1d(out_channels, out_channels, kernel_size, dilation),
            nn.BatchNorm1d(out_channels),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
        )

        # 1x1 conv for residual connection when channel dims differ
        self.residual = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

        self.activation = nn.ELU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with residual connection.

        Args:
            x: Input tensor [B, C, T].

        Returns:
            Output tensor [B, C_out, T].
        """
        return self.activation(self.net(x) + self.residual(x))


class SensorTCN(nn.Module):
    """Temporal Convolutional Network for water quality sensor data.

    3-layer TCN with increasing channel widths and dilation factors,
    producing a fixed-size embedding from variable-length time series.

    Architecture:
        Layer 1: channels=64,  dilation=1, receptive field = 13
        Layer 2: channels=128, dilation=2, receptive field = 37
        Layer 3: channels=256, dilation=4, receptive field = 85
        Total receptive field: 85 timesteps (~21 hours at 15-min intervals)

    Args:
        in_channels: Number of input parameters. Default 6.
        channels: Channel widths per TCN layer. Default [64, 128, 256].
        kernel_size: Convolution kernel size. Default 7.
        dilations: Dilation factors per layer. Default [1, 2, 4].
        dropout: Dropout rate. Default 0.2.
        output_dim: Output embedding dimension. Default 256.
    """

    def __init__(
        self,
        in_channels: int = NUM_PARAMETERS,
        channels: list[int] | None = None,
        kernel_size: int = 7,
        dilations: list[int] | None = None,
        dropout: float = 0.2,
        output_dim: int = OUTPUT_DIM,
    ) -> None:
        super().__init__()

        if channels is None:
            channels = [64, 128, 256]
        if dilations is None:
            dilations = [1, 2, 4]

        assert len(channels) == len(dilations), (
            "channels and dilations must have the same length"
        )

        self.output_dim = output_dim

        # Build TCN layers
        layers: list[nn.Module] = []
        prev_ch = in_channels
        for ch, dil in zip(channels, dilations):
            layers.append(TCNBlock(prev_ch, ch, kernel_size, dil, dropout))
            prev_ch = ch
        self.tcn = nn.Sequential(*layers)

        # Global pooling + projection to output_dim
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Sequential(
            nn.Linear(channels[-1], output_dim),
            nn.LayerNorm(output_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through TCN.

        Args:
            x: Sensor readings [B, T=672, P=6] (batch, time, parameters).

        Returns:
            embedding: Global embedding [B, output_dim].
            temporal_features: Per-timestep features [B, channels[-1], T]
                for use by the anomaly detection module.
        """
        # TCN expects [B, C, T] format
        x = x.transpose(1, 2)  # [B, P, T]

        # Forward through TCN blocks
        temporal_features = self.tcn(x)  # [B, 256, T]

        # Global average pooling for embedding
        pooled = self.pool(temporal_features).squeeze(-1)  # [B, 256]
        embedding = self.projection(pooled)  # [B, output_dim]

        return embedding, temporal_features
