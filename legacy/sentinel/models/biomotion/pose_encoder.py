"""Pose encoder for SLEAP keypoint sequences.

Encodes per-frame pose keypoints into fixed-dimensional temporal
representations via per-frame MLP projection followed by a temporal
transformer with sinusoidal positional encoding for irregular timestamps.

Supports variable keypoint counts across species (Daphnia 12, mussel 8,
fish 22) through a max-keypoint padding strategy with masking.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# Species keypoint counts used across the SENTINEL aquatic biomonitoring system.
SPECIES_KEYPOINTS: dict[str, int] = {
    "daphnia": 12,
    "mussel": 8,
    "fish": 22,
}

MAX_KEYPOINTS: int = max(SPECIES_KEYPOINTS.values())  # 22

# Default pose embedding dimension.
POSE_DIM: int = 128


# ---------------------------------------------------------------------------
# Sinusoidal positional encoding for irregular timestamps
# ---------------------------------------------------------------------------


class SinusoidalTimestampEncoding(nn.Module):
    """Sinusoidal positional encoding driven by absolute timestamps.

    Unlike standard positional encoding that uses integer positions, this
    variant encodes continuous-valued timestamps (in seconds) so the model
    can handle irregularly-sampled trajectories.

    Args:
        d_model: Encoding dimension (must be even).
        max_period: Maximum period for the sinusoidal basis functions
            (seconds).  Controls the lowest frequency.
    """

    def __init__(self, d_model: int = POSE_DIM, max_period: float = 1000.0) -> None:
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even, got {d_model}")
        self.d_model = d_model
        self.max_period = max_period

        # Precompute frequency bands: shape (d_model // 2,)
        half = d_model // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32) / half
        )
        self.register_buffer("freqs", freqs)

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """Compute sinusoidal encoding for given timestamps.

        Args:
            timestamps: Timestamps in seconds, shape ``(B, T)`` or ``(T,)``.

        Returns:
            Positional encoding, shape ``(..., T, d_model)``.
        """
        # timestamps: (..., T) -> (..., T, 1)
        t = timestamps.unsqueeze(-1).float()
        # freqs: (half,) -> (1, half)
        args = t * self.freqs.unsqueeze(0)  # (..., T, half)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


# ---------------------------------------------------------------------------
# Per-frame MLP
# ---------------------------------------------------------------------------


class FrameEncoder(nn.Module):
    """Per-frame keypoint encoder: flatten + two-layer MLP.

    Args:
        max_keypoints: Maximum number of keypoints (padded dimension).
        hidden_dim: Hidden layer width.
        output_dim: Output dimension per frame.
    """

    def __init__(
        self,
        max_keypoints: int = MAX_KEYPOINTS,
        hidden_dim: int = 256,
        output_dim: int = POSE_DIM,
    ) -> None:
        super().__init__()
        input_dim = max_keypoints * 2  # XY per keypoint
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, keypoints: torch.Tensor) -> torch.Tensor:
        """Encode per-frame keypoints.

        Args:
            keypoints: Shape ``(B, T, max_keypoints, 2)``.

        Returns:
            Per-frame embeddings ``(B, T, output_dim)``.
        """
        B, T, K, _ = keypoints.shape
        flat = keypoints.reshape(B, T, K * 2)
        return self.mlp(flat)


# ---------------------------------------------------------------------------
# Temporal Transformer
# ---------------------------------------------------------------------------


class TemporalTransformer(nn.Module):
    """Lightweight transformer encoder for temporal pose sequences.

    Args:
        d_model: Feature dimension.
        nhead: Number of attention heads.
        num_layers: Number of transformer encoder layers.
        dim_feedforward: Feed-forward hidden dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        d_model: int = POSE_DIM,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the temporal transformer.

        Args:
            x: Input features ``(B, T, d_model)``.
            mask: Key-padding mask ``(B, T)`` where ``True`` indicates
                padded (invalid) positions.

        Returns:
            Contextualized features ``(B, T, d_model)``.
        """
        return self.encoder(x, src_key_padding_mask=mask)


# ---------------------------------------------------------------------------
# Full Pose Encoder
# ---------------------------------------------------------------------------


class PoseEncoder(nn.Module):
    """Encode SLEAP keypoint sequences into a fixed-dim representation.

    Pipeline:
      1. Pad keypoints to ``max_keypoints`` with masking for species with
         fewer keypoints.
      2. Per-frame MLP: flatten + Linear -> ReLU -> Linear -> ``(B, T, pose_dim)``.
      3. Add sinusoidal positional encoding from timestamps.
      4. Temporal transformer over the pose sequence.
      5. Masked mean pooling -> ``(B, pose_dim)``.

    Args:
        pose_dim: Pose embedding dimension.
        max_keypoints: Maximum keypoint count across species.
        frame_hidden_dim: Hidden dim in per-frame MLP.
        nhead: Transformer attention heads.
        num_layers: Transformer encoder layers.
        dim_feedforward: Transformer FF hidden dim.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        pose_dim: int = POSE_DIM,
        max_keypoints: int = MAX_KEYPOINTS,
        frame_hidden_dim: int = 256,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.pose_dim = pose_dim
        self.max_keypoints = max_keypoints

        self.frame_encoder = FrameEncoder(
            max_keypoints=max_keypoints,
            hidden_dim=frame_hidden_dim,
            output_dim=pose_dim,
        )
        self.timestamp_encoding = SinusoidalTimestampEncoding(d_model=pose_dim)
        self.layer_norm = nn.LayerNorm(pose_dim)
        self.temporal_transformer = TemporalTransformer(
            d_model=pose_dim,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

    def forward(
        self,
        keypoints: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode a batch of keypoint sequences.

        Args:
            keypoints: Keypoint coordinates ``(B, T, n_keypoints, 2)``.
                If ``n_keypoints < max_keypoints``, zero-padded internally.
            timestamps: Frame timestamps in seconds ``(B, T)``.  If None,
                integer indices are used as surrogate timestamps.
            padding_mask: Boolean mask ``(B, T)`` where ``True`` marks
                padded (invalid) time steps.

        Returns:
            Pooled pose embedding ``(B, pose_dim)``.
        """
        B, T, K, _ = keypoints.shape

        # Pad keypoints to max_keypoints if needed
        if K < self.max_keypoints:
            pad = torch.zeros(
                B, T, self.max_keypoints - K, 2,
                device=keypoints.device, dtype=keypoints.dtype,
            )
            keypoints = torch.cat([keypoints, pad], dim=2)

        # Per-frame encoding: (B, T, pose_dim)
        frame_emb = self.frame_encoder(keypoints)

        # Positional encoding from timestamps
        if timestamps is None:
            timestamps = torch.arange(T, device=keypoints.device, dtype=torch.float32)
            timestamps = timestamps.unsqueeze(0).expand(B, -1)
        pos_enc = self.timestamp_encoding(timestamps)  # (B, T, pose_dim)
        frame_emb = self.layer_norm(frame_emb + pos_enc)

        # Temporal transformer
        contextualized = self.temporal_transformer(frame_emb, mask=padding_mask)

        # Masked mean pooling
        if padding_mask is not None:
            # padding_mask: True = invalid -> invert for valid mask
            valid = ~padding_mask  # (B, T)
            valid_f = valid.unsqueeze(-1).float()  # (B, T, 1)
            pooled = (contextualized * valid_f).sum(dim=1) / valid_f.sum(dim=1).clamp(min=1.0)
        else:
            pooled = contextualized.mean(dim=1)

        return pooled  # (B, pose_dim)
