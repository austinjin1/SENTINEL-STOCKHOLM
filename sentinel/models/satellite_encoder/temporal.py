"""Temporal change detection module for satellite imagery.

Maintains a rolling buffer of [CLS] token embeddings and uses a
2-layer transformer encoder with sinusoidal temporal encodings to
detect changes over irregular acquisition intervals.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


BUFFER_SIZE = 10
EMBED_DIM = 384


class SinusoidalTemporalEncoding(nn.Module):
    """Sinusoidal positional encoding based on actual acquisition timestamps.

    Unlike fixed positional encodings, this generates embeddings from
    continuous timestamp values (in days since epoch), supporting irregular
    satellite revisit intervals (e.g., Sentinel-2's 5-day cadence with
    cloud-related gaps).

    Args:
        d_model: Embedding dimension.
        max_period: Maximum period for sinusoidal encoding (days).
    """

    def __init__(self, d_model: int = EMBED_DIM, max_period: float = 365.0) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_period = max_period

        # Precompute frequency bands
        half_dim = d_model // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(0, half_dim, dtype=torch.float32)
            / half_dim
        )
        self.register_buffer("freqs", freqs)  # [half_dim]

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """Generate temporal encodings from timestamps.

        Args:
            timestamps: Acquisition timestamps in days since epoch.
                Shape [B, T] or [T].

        Returns:
            Temporal encoding of shape [B, T, d_model] or [T, d_model].
        """
        # Ensure timestamps have at least 2 dims
        squeeze = False
        if timestamps.dim() == 1:
            timestamps = timestamps.unsqueeze(0)
            squeeze = True

        # [B, T, 1] * [half_dim] -> [B, T, half_dim]
        args = timestamps.unsqueeze(-1) * self.freqs
        encoding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        # Handle odd d_model
        if self.d_model % 2 == 1:
            encoding = F.pad(encoding, (0, 1))

        if squeeze:
            encoding = encoding.squeeze(0)
        return encoding


class TemporalChangeDetector(nn.Module):
    """Temporal change detection via transformer over rolling CLS embeddings.

    Maintains a rolling buffer of the last 10 [CLS] token embeddings per
    tile and processes them through a lightweight transformer encoder with
    sinusoidal temporal encodings derived from actual acquisition timestamps.

    The change anomaly score is computed as the cosine distance between the
    current embedding and the mean of previous embeddings, normalized by a
    learned standard deviation estimate.

    Args:
        embed_dim: Dimension of [CLS] token embeddings. Default 384.
        buffer_size: Number of historical embeddings to retain. Default 10.
        nhead: Number of attention heads in temporal transformer. Default 6.
        num_layers: Number of transformer encoder layers. Default 2.
        dropout: Dropout rate. Default 0.1.
    """

    def __init__(
        self,
        embed_dim: int = EMBED_DIM,
        buffer_size: int = BUFFER_SIZE,
        nhead: int = 6,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.buffer_size = buffer_size

        # Temporal encoding from timestamps
        self.temporal_encoding = SinusoidalTemporalEncoding(embed_dim)

        # Lightweight transformer encoder for temporal reasoning
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Layer norm on input embeddings
        self.input_norm = nn.LayerNorm(embed_dim)

        # Learned standard deviation for normalizing change scores
        self.log_sigma = nn.Parameter(torch.zeros(1))

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize transformer weights."""
        for p in self.transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        cls_embeddings: torch.Tensor,
        timestamps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Process a temporal sequence of CLS embeddings.

        Args:
            cls_embeddings: Rolling buffer of [CLS] tokens.
                Shape [B, T, embed_dim] where T <= buffer_size.
                The last position (T-1) is the current observation.
            timestamps: Acquisition timestamps in days since epoch.
                Shape [B, T].
            attention_mask: Optional boolean mask. True = ignore position.
                Shape [B, T]. Used when buffer has fewer than buffer_size entries.

        Returns:
            Dict with:
                'temporal_embedding': Contextualized embedding [B, embed_dim].
                'change_anomaly_score': Per-sample anomaly score [B].
        """
        B, T, D = cls_embeddings.shape

        # Add sinusoidal temporal encoding
        temp_enc = self.temporal_encoding(timestamps)  # [B, T, D]
        x = self.input_norm(cls_embeddings) + temp_enc

        # Convert boolean padding mask to format expected by PyTorch transformer
        # PyTorch: True = ignore
        src_key_padding_mask = attention_mask  # [B, T] or None

        # Temporal transformer
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        # [B, T, D]

        # Current embedding = last position output
        current_embedding = x[:, -1, :]  # [B, D]

        # Change anomaly score: cosine distance from historical mean
        if T > 1:
            # Mean of previous embeddings (exclude current)
            if attention_mask is not None:
                # Mask out padded positions from history
                hist_mask = attention_mask[:, :-1]  # [B, T-1]
                hist = x[:, :-1, :]  # [B, T-1, D]
                # Set masked positions to zero
                hist = hist.masked_fill(hist_mask.unsqueeze(-1), 0.0)
                valid_count = (~hist_mask).sum(dim=1, keepdim=True).clamp(min=1)
                hist_mean = hist.sum(dim=1) / valid_count.float()  # [B, D]
            else:
                hist_mean = x[:, :-1, :].mean(dim=1)  # [B, D]

            # Cosine similarity -> cosine distance
            cos_sim = F.cosine_similarity(current_embedding, hist_mean, dim=-1)
            cos_distance = 1.0 - cos_sim  # [B]

            # Normalize by learned sigma
            sigma = torch.exp(self.log_sigma).clamp(min=1e-6)
            change_score = cos_distance / sigma
        else:
            # Single observation: no history to compare
            change_score = torch.zeros(B, device=cls_embeddings.device)

        return {
            "temporal_embedding": current_embedding,
            "change_anomaly_score": change_score,
        }


class RollingBuffer:
    """Fixed-size rolling buffer for [CLS] embeddings and timestamps.

    Manages a per-tile FIFO buffer for temporal context during inference.
    Not an nn.Module -- this is a stateful inference utility.

    Args:
        buffer_size: Maximum number of entries to retain.
        embed_dim: Dimension of embeddings.
        device: Torch device.
    """

    def __init__(
        self,
        buffer_size: int = BUFFER_SIZE,
        embed_dim: int = EMBED_DIM,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.buffer_size = buffer_size
        self.embed_dim = embed_dim
        self.device = device

        self.embeddings: list[torch.Tensor] = []
        self.timestamps: list[float] = []

    def push(self, embedding: torch.Tensor, timestamp: float) -> None:
        """Add a new embedding and timestamp to the buffer.

        Args:
            embedding: [CLS] token embedding, shape [embed_dim].
            timestamp: Acquisition timestamp in days since epoch.
        """
        self.embeddings.append(embedding.detach().to(self.device))
        self.timestamps.append(timestamp)

        # Evict oldest if over capacity
        if len(self.embeddings) > self.buffer_size:
            self.embeddings.pop(0)
            self.timestamps.pop(0)

    @torch.no_grad()
    def get_tensors(self) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Get stacked tensors for the temporal module.

        Returns:
            embeddings: Shape [1, T, embed_dim].
            timestamps: Shape [1, T].
            attention_mask: Shape [1, T] with True for padded positions,
                or None if buffer is full.
        """
        T = len(self.embeddings)
        if T == 0:
            raise ValueError("Buffer is empty; push at least one embedding.")

        emb = torch.stack(self.embeddings, dim=0).unsqueeze(0)  # [1, T, D]
        ts = torch.tensor(
            self.timestamps, dtype=torch.float32, device=self.device
        ).unsqueeze(0)  # [1, T]

        # Pad to buffer_size if needed
        if T < self.buffer_size:
            pad_len = self.buffer_size - T
            emb = F.pad(emb, (0, 0, pad_len, 0))  # Pad on left (older side)
            ts = F.pad(ts, (pad_len, 0))
            mask = torch.zeros(1, self.buffer_size, dtype=torch.bool, device=self.device)
            mask[:, :pad_len] = True
            return emb, ts, mask

        return emb, ts, None

    def clear(self) -> None:
        """Clear the buffer."""
        self.embeddings.clear()
        self.timestamps.clear()

    def __len__(self) -> int:
        return len(self.embeddings)
