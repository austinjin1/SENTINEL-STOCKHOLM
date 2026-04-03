"""Community Trajectory Variational Autoencoder for microbial health assessment.

Learns a latent representation of healthy microbial community profiles
from EPA NARS reference condition sites. Anomalous communities (pushed
outside the normal distribution by contamination) exhibit high
reconstruction error, enabling health scoring.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CommunityTrajectoryVAE(nn.Module):
    """Variational Autoencoder for microbial community profiles.

    Trained on healthy reference communities from EPA NARS to learn
    the manifold of normal microbial composition. New samples are
    scored by reconstruction error -- high error indicates the
    community has been displaced from its normal state.

    Architecture:
        Encoder: input_dim -> 512 -> 256 -> latent_dim (mu + logvar)
        Decoder: latent_dim -> 256 -> 512 -> input_dim

    Args:
        input_dim: Dimension of CLR-transformed ASV abundance vector.
        hidden_dims: Encoder/decoder hidden layer dimensions.
        latent_dim: Latent space dimension. Default 32.
        dropout: Dropout rate. Default 0.1.
        beta: KL divergence weight (beta-VAE). Default 1.0.
    """

    def __init__(
        self,
        input_dim: int = 5000,
        hidden_dims: list[int] | None = None,
        latent_dim: int = 32,
        dropout: float = 0.1,
        beta: float = 1.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta

        if hidden_dims is None:
            hidden_dims = [512, 256]

        # Encoder: input -> 512 -> 256 -> (mu, logvar)
        encoder_layers: list[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim

        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

        # Decoder: latent -> 256 -> 512 -> input
        decoder_layers: list[nn.Module] = []
        reversed_dims = list(reversed(hidden_dims))
        prev_dim = latent_dim
        for h_dim in reversed_dims:
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim

        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        # Running statistics for anomaly scoring normalization
        self.register_buffer(
            "train_recon_mean", torch.tensor(0.0)
        )
        self.register_buffer(
            "train_recon_std", torch.tensor(1.0)
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with Xavier uniform for linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters.

        Args:
            x: CLR-transformed ASV abundances [B, input_dim].

        Returns:
            mu: Mean of latent distribution [B, latent_dim].
            logvar: Log-variance of latent distribution [B, latent_dim].
        """
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """Reparameterization trick for differentiable sampling.

        Args:
            mu: Mean [B, latent_dim].
            logvar: Log-variance [B, latent_dim].

        Returns:
            Sampled latent vector [B, latent_dim].
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            # At inference, use mean (deterministic)
            return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstructed input.

        Args:
            z: Latent vector [B, latent_dim].

        Returns:
            Reconstructed CLR abundances [B, input_dim].
        """
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Full VAE forward pass.

        Args:
            x: CLR-transformed ASV abundances [B, input_dim].

        Returns:
            Dict with:
                'reconstruction': Reconstructed input [B, input_dim].
                'mu': Latent mean [B, latent_dim].
                'logvar': Latent log-variance [B, latent_dim].
                'z': Sampled latent vector [B, latent_dim].
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)

        return {
            "reconstruction": reconstruction,
            "mu": mu,
            "logvar": logvar,
            "z": z,
        }

    def compute_loss(
        self,
        x: torch.Tensor,
        outputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute VAE loss: reconstruction + KL divergence.

        Args:
            x: Original input [B, input_dim].
            outputs: Forward pass outputs dict.

        Returns:
            Dict with 'total', 'reconstruction', and 'kl' losses.
        """
        recon = outputs["reconstruction"]
        mu = outputs["mu"]
        logvar = outputs["logvar"]

        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(recon, x, reduction="mean")

        # KL divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kl_loss = -0.5 * torch.mean(
            torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        )

        total = recon_loss + self.beta * kl_loss

        return {
            "total": total,
            "reconstruction": recon_loss,
            "kl": kl_loss,
        }

    @torch.no_grad()
    def compute_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Compute community health anomaly score via reconstruction error.

        Higher scores indicate communities pushed further from the healthy
        reference distribution.

        Args:
            x: CLR-transformed ASV abundances [B, input_dim].

        Returns:
            Anomaly scores [B], normalized by training distribution statistics.
        """
        outputs = self.forward(x)
        recon = outputs["reconstruction"]

        # Per-sample reconstruction error (MSE)
        recon_error = F.mse_loss(recon, x, reduction="none").mean(dim=1)  # [B]

        # Normalize by training statistics
        normalized = (recon_error - self.train_recon_mean) / self.train_recon_std.clamp(min=1e-6)

        return normalized

    def update_training_statistics(self, recon_errors: torch.Tensor) -> None:
        """Update running statistics from training reconstruction errors.

        Call this at the end of each training epoch with accumulated errors.

        Args:
            recon_errors: Per-sample reconstruction errors from training [N].
        """
        self.train_recon_mean.copy_(recon_errors.mean())
        self.train_recon_std.copy_(recon_errors.std().clamp(min=1e-6))
