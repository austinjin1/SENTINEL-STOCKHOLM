"""Diffusion-pretrained trajectory encoder for behavioral anomaly detection.

Core insight: learn normal behavior by training to denoise corrupted
trajectories.  Trajectories that are hard to denoise (high reconstruction
error) are behaviorally abnormal -- the denoising score IS the anomaly
signal.

Architecture:
  - Transformer-based denoiser operating on behavioral feature sequences
  - Cosine noise schedule (Nichol & Dhariwal 2021)
  - Multi-scale denoising score aggregation for robust anomaly detection
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Cosine noise schedule (DDPM, Nichol & Dhariwal 2021)
# ---------------------------------------------------------------------------

EMBED_DIM: int = 256
NUM_DIFFUSION_STEPS: int = 1000


def cosine_alpha_bar_schedule(
    num_steps: int = NUM_DIFFUSION_STEPS,
    s: float = 0.008,
) -> torch.Tensor:
    """Cosine schedule for cumulative alpha_bar values.

    Args:
        num_steps: Total diffusion steps.
        s: Small offset to prevent singularity at t=0.

    Returns:
        alpha_bar values, shape ``(num_steps + 1,)``.
    """
    steps = torch.arange(num_steps + 1, dtype=torch.float64) / num_steps
    f = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = f / f[0]
    return alpha_bar.float()


# ---------------------------------------------------------------------------
# Noise-level embedding (sinusoidal)
# ---------------------------------------------------------------------------


class NoiseEmbedding(nn.Module):
    """Sinusoidal embedding for continuous noise level / diffusion timestep.

    Args:
        embed_dim: Output embedding dimension.
    """

    def __init__(self, embed_dim: int = EMBED_DIM) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, noise_level: torch.Tensor) -> torch.Tensor:
        """Embed noise level.

        Args:
            noise_level: Scalar or ``(B,)`` noise levels in ``[0, 1]``.

        Returns:
            Noise embeddings ``(B, embed_dim)``.
        """
        if noise_level.dim() == 0:
            noise_level = noise_level.unsqueeze(0)
        half = self.embed_dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=noise_level.device, dtype=torch.float32)
            / half
        )
        args = noise_level.unsqueeze(-1) * freqs.unsqueeze(0)  # (B, half)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, embed_dim)
        return self.mlp(emb)


# ---------------------------------------------------------------------------
# Transformer-based denoiser
# ---------------------------------------------------------------------------


class TransformerDenoiser(nn.Module):
    """Transformer denoiser for trajectory sequences.

    Takes noised feature sequences and a noise-level embedding, predicts the
    noise component (epsilon prediction).

    Args:
        feature_dim: Dimension of per-frame behavioral features.
        d_model: Internal transformer dimension.
        nhead: Number of attention heads.
        num_layers: Number of transformer encoder layers.
        dim_feedforward: Feed-forward hidden dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        feature_dim: int = 32,
        d_model: int = EMBED_DIM,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Linear(feature_dim, d_model)

        # Noise conditioning: additive bias from noise embedding
        self.noise_proj = nn.Linear(d_model, d_model)

        # Learnable positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, 2048, d_model) * 0.02)
        self.layer_norm_in = nn.LayerNorm(d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection back to feature space
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, feature_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) and "transformer" not in name:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        noised_features: torch.Tensor,
        noise_embedding: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict noise from noised features.

        Args:
            noised_features: ``(B, T, feature_dim)`` noised behavioral features.
            noise_embedding: ``(B, d_model)`` noise-level embedding.
            padding_mask: ``(B, T)`` True for padded positions.

        Returns:
            Predicted noise ``(B, T, feature_dim)``.
        """
        B, T, _ = noised_features.shape

        # Project input to d_model
        h = self.input_proj(noised_features)  # (B, T, d_model)

        # Add positional encoding
        h = h + self.pos_embed[:, :T, :]

        # Condition on noise level via additive bias
        noise_bias = self.noise_proj(noise_embedding).unsqueeze(1)  # (B, 1, d_model)
        h = self.layer_norm_in(h + noise_bias)

        # Transformer
        h = self.transformer(h, src_key_padding_mask=padding_mask)

        # Project back to feature space
        return self.output_proj(h)


# ---------------------------------------------------------------------------
# Main encoder
# ---------------------------------------------------------------------------


class TrajectoryDiffusionEncoder(nn.Module):
    """Diffusion-pretrained trajectory encoder.

    Learns normal behavioral patterns by training to denoise corrupted
    feature trajectories.  At inference, the denoising difficulty
    (reconstruction error across noise levels) serves as an anomaly score.

    Args:
        feature_dim: Per-frame behavioral feature dimension.
        embed_dim: Latent embedding dimension.
        nhead: Transformer attention heads.
        num_layers: Transformer encoder layers.
        dim_feedforward: Transformer FF hidden dim.
        dropout: Dropout rate.
        num_diffusion_steps: Discretised diffusion steps for the schedule.
        num_eval_noise_levels: Number of noise levels to evaluate for
            anomaly scoring.
    """

    def __init__(
        self,
        feature_dim: int = 32,
        embed_dim: int = EMBED_DIM,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        num_diffusion_steps: int = NUM_DIFFUSION_STEPS,
        num_eval_noise_levels: int = 10,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim
        self.num_eval_noise_levels = num_eval_noise_levels

        # Cosine noise schedule
        alpha_bar = cosine_alpha_bar_schedule(num_diffusion_steps)
        self.register_buffer("alpha_bar", alpha_bar)
        self.num_diffusion_steps = num_diffusion_steps

        # Noise embedding
        self.noise_embedding = NoiseEmbedding(embed_dim=embed_dim)

        # Denoiser
        self.denoiser = TransformerDenoiser(
            feature_dim=feature_dim,
            d_model=embed_dim,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        # Encoding head: project denoiser hidden states to embedding
        self.encoding_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )
        self.encoding_input_proj = nn.Linear(feature_dim, embed_dim)
        self.encoding_layer_norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in [self.encoding_proj, self.encoding_input_proj]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Sequential):
                for sub in m.modules():
                    if isinstance(sub, nn.Linear):
                        nn.init.xavier_uniform_(sub.weight)
                        if sub.bias is not None:
                            nn.init.zeros_(sub.bias)

    # ----- Diffusion forward process -----

    def _sample_noise_level(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample random diffusion timesteps uniformly.

        Returns:
            Integer timesteps ``(B,)`` in ``[1, num_diffusion_steps]``.
        """
        return torch.randint(
            1, self.num_diffusion_steps + 1, (batch_size,), device=device
        )

    def _corrupt(
        self,
        features: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply forward diffusion to feature sequences.

        Args:
            features: Clean features ``(B, T, feature_dim)``.
            timesteps: Diffusion timesteps ``(B,)`` as integers.

        Returns:
            Tuple of (noised_features, noise).
        """
        ab = self.alpha_bar[timesteps]  # (B,)
        sqrt_ab = ab.sqrt().view(-1, 1, 1)  # (B, 1, 1)
        sqrt_one_minus_ab = (1.0 - ab).sqrt().view(-1, 1, 1)

        noise = torch.randn_like(features)
        noised = sqrt_ab * features + sqrt_one_minus_ab * noise
        return noised, noise

    # ----- Training interface -----

    def forward_denoise(
        self,
        corrupted_features: torch.Tensor,
        noise_level: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict noise from corrupted features (training target).

        Args:
            corrupted_features: ``(B, T, feature_dim)`` noised features.
            noise_level: ``(B,)`` noise levels in ``[0, 1]``.
            padding_mask: ``(B, T)`` True for padded positions.

        Returns:
            Predicted noise ``(B, T, feature_dim)``.
        """
        noise_emb = self.noise_embedding(noise_level)
        return self.denoiser(corrupted_features, noise_emb, padding_mask)

    def compute_training_loss(
        self,
        features: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Compute diffusion denoising training loss.

        Samples random timesteps, corrupts features, and computes MSE
        between predicted and actual noise.

        Args:
            features: Clean features ``(B, T, feature_dim)``.
            padding_mask: ``(B, T)`` True for padded positions.

        Returns:
            Dict with ``"loss"`` (scalar) and ``"predicted_noise"``.
        """
        B = features.shape[0]
        timesteps = self._sample_noise_level(B, features.device)
        noise_level = timesteps.float() / self.num_diffusion_steps  # [0, 1]

        noised, noise = self._corrupt(features, timesteps)
        predicted_noise = self.forward_denoise(noised, noise_level, padding_mask)

        # Mask out padded positions for loss computation
        if padding_mask is not None:
            valid = ~padding_mask  # (B, T)
            valid_f = valid.unsqueeze(-1).float()  # (B, T, 1)
            mse = ((predicted_noise - noise) ** 2 * valid_f).sum() / valid_f.sum().clamp(min=1.0) / self.feature_dim
        else:
            mse = F.mse_loss(predicted_noise, noise)

        return {"loss": mse, "predicted_noise": predicted_noise}

    # ----- Encoding interface -----

    def forward_encode(
        self,
        features: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode clean trajectory features into a fixed embedding.

        Uses the denoiser at zero noise level to extract representations,
        then pools over time.

        Args:
            features: Clean features ``(B, T, feature_dim)``.
            padding_mask: ``(B, T)`` True for padded positions.

        Returns:
            Trajectory embedding ``(B, embed_dim)``.
        """
        B, T, _ = features.shape

        # Project features to embed_dim for the encoding pathway
        h = self.encoding_input_proj(features)  # (B, T, embed_dim)

        # Add positional encoding from denoiser
        h = h + self.denoiser.pos_embed[:, :T, :]

        # Condition on zero noise (clean signal)
        zero_noise = torch.zeros(B, device=features.device)
        noise_emb = self.noise_embedding(zero_noise)
        noise_bias = self.denoiser.noise_proj(noise_emb).unsqueeze(1)
        h = self.encoding_layer_norm(h + noise_bias)

        # Pass through transformer
        h = self.denoiser.transformer(h, src_key_padding_mask=padding_mask)

        # Masked mean pooling
        if padding_mask is not None:
            valid = (~padding_mask).unsqueeze(-1).float()
            pooled = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        else:
            pooled = h.mean(dim=1)

        return self.encoding_proj(pooled)  # (B, embed_dim)

    # ----- Anomaly scoring -----

    @torch.no_grad()
    def compute_anomaly_score(
        self,
        features: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        num_noise_levels: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute anomaly score as average denoising difficulty.

        Evaluates denoising error across multiple noise levels.  Normal
        trajectories (learned during pretraining) are easy to denoise;
        anomalous trajectories produce high reconstruction error.

        Args:
            features: Clean features ``(B, T, feature_dim)``.
            padding_mask: ``(B, T)`` True for padded positions.
            num_noise_levels: Number of noise levels to evaluate.
                Defaults to ``self.num_eval_noise_levels``.

        Returns:
            Per-sample anomaly score ``(B,)``.
        """
        num_levels = num_noise_levels or self.num_eval_noise_levels
        B = features.shape[0]
        device = features.device

        # Evaluate at evenly-spaced noise levels in (0, 1]
        noise_fracs = torch.linspace(
            1.0 / num_levels, 1.0, num_levels, device=device
        )

        total_error = torch.zeros(B, device=device)

        for frac in noise_fracs:
            timestep = max(1, int(frac.item() * self.num_diffusion_steps))
            timesteps = torch.full((B,), timestep, device=device, dtype=torch.long)
            noise_level = torch.full((B,), frac.item(), device=device)

            noised, noise = self._corrupt(features, timesteps)
            predicted_noise = self.forward_denoise(noised, noise_level, padding_mask)

            # Per-sample MSE
            if padding_mask is not None:
                valid = (~padding_mask).unsqueeze(-1).float()
                per_sample = ((predicted_noise - noise) ** 2 * valid).sum(dim=(1, 2)) / (
                    valid.sum(dim=(1, 2)).clamp(min=1.0)
                )
            else:
                per_sample = ((predicted_noise - noise) ** 2).mean(dim=(1, 2))

            total_error += per_sample

        return total_error / num_levels
