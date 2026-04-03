"""Full microbial encoder combining source attribution and community health VAE.

Provides the complete microbial modality encoder that:
1. Classifies contamination sources from microbial community profiles
2. Assesses community health via VAE reconstruction error
3. Identifies indicator species via attention weight extraction
4. Projects to shared 256-dim fusion embedding space
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .source_attribution import (
    SourceAttributionTransformer,
    MAX_ASV_FEATURES,
    EMBED_DIM,
    NUM_SOURCES,
)
from .vae import CommunityTrajectoryVAE

SHARED_EMBED_DIM = 256


class MicrobialEncoder(nn.Module):
    """Complete microbial modality encoder for SENTINEL.

    Combines the source attribution transformer (for contamination
    classification and indicator species discovery) with the community
    trajectory VAE (for health scoring against reference conditions).

    Args:
        input_dim: Number of ASV features. Default 5000.
        embed_dim: Transformer embedding dimension. Default 256.
        num_heads: Transformer attention heads. Default 4.
        num_layers: Transformer layers. Default 4.
        ff_dim: Transformer feed-forward dimension. Default 512.
        latent_dim: VAE latent dimension. Default 32.
        dropout: Dropout rate. Default 0.1.
        shared_embed_dim: Shared fusion embedding dimension. Default 256.
        num_sources: Number of contamination source types. Default 8.
        vae_beta: KL divergence weight for VAE. Default 1.0.
    """

    def __init__(
        self,
        input_dim: int = MAX_ASV_FEATURES,
        embed_dim: int = EMBED_DIM,
        num_heads: int = 4,
        num_layers: int = 4,
        ff_dim: int = 512,
        latent_dim: int = 32,
        dropout: float = 0.1,
        shared_embed_dim: int = SHARED_EMBED_DIM,
        num_sources: int = NUM_SOURCES,
        vae_beta: float = 1.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim

        # Source attribution transformer
        self.source_attribution = SourceAttributionTransformer(
            input_dim=input_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=dropout,
            num_sources=num_sources,
        )

        # Community trajectory VAE
        self.vae = CommunityTrajectoryVAE(
            input_dim=input_dim,
            latent_dim=latent_dim,
            dropout=dropout,
            beta=vae_beta,
        )

        # Fusion of transformer CLS embedding + VAE latent for projection
        self.fusion_layer = nn.Sequential(
            nn.Linear(embed_dim + latent_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )

        # Projection to shared embedding space
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, shared_embed_dim),
            nn.LayerNorm(shared_embed_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in [self.fusion_layer, self.projection]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        extract_indicators: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Full forward pass through the microbial encoder.

        Args:
            x: CLR-transformed ASV abundances [B, input_dim].
            extract_indicators: Whether to compute indicator species weights
                (requires additional forward pass for attention extraction).

        Returns:
            Dict with:
                'embedding': Projected embedding [B, 256] for fusion.
                'fusion_embedding': Same as embedding (for consistency).
                'source_logits': Contamination source logits [B, num_sources].
                'source_probs': Source probabilities [B, num_sources].
                'community_health_score': VAE-based health score [B].
                'indicator_species_weights': Per-ASV importance [B, input_dim]
                    (only if extract_indicators=True).
                'vae_outputs': VAE forward pass outputs dict.
        """
        # Source attribution
        source_output = self.source_attribution(x)
        # source_output: source_logits, source_probs, embedding

        # Community health via VAE
        vae_output = self.vae(x)
        community_health = self.vae.compute_anomaly_score(x)

        # Fuse transformer CLS embedding with VAE latent
        combined = torch.cat(
            [source_output["embedding"], vae_output["mu"]], dim=-1
        )
        fused = self.fusion_layer(combined)

        # Project to shared space
        embedding = self.projection(fused)

        result: dict[str, torch.Tensor] = {
            "embedding": embedding,
            "fusion_embedding": embedding,
            "source_logits": source_output["source_logits"],
            "source_probs": source_output["source_probs"],
            "community_health_score": community_health,
            "vae_outputs": vae_output,
        }

        # Optionally extract indicator species weights
        if extract_indicators:
            indicator_weights = self.source_attribution.get_indicator_species_weights(x)
            result["indicator_species_weights"] = indicator_weights
        else:
            result["indicator_species_weights"] = torch.zeros(
                x.shape[0], self.input_dim, device=x.device
            )

        return result

    def compute_loss(
        self,
        x: torch.Tensor,
        outputs: dict[str, torch.Tensor],
        source_targets: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Compute combined training loss.

        Args:
            x: Original input [B, input_dim].
            outputs: Forward pass outputs.
            source_targets: Ground truth source labels [B] (long tensor).
                If None, only VAE loss is computed.

        Returns:
            Dict of loss components.
        """
        losses = {}

        # VAE loss
        vae_losses = self.vae.compute_loss(x, outputs["vae_outputs"])
        losses["vae_total"] = vae_losses["total"]
        losses["vae_recon"] = vae_losses["reconstruction"]
        losses["vae_kl"] = vae_losses["kl"]

        # Source attribution loss
        if source_targets is not None:
            source_loss = nn.functional.cross_entropy(
                outputs["source_logits"], source_targets
            )
            losses["source_attribution"] = source_loss
            losses["total"] = vae_losses["total"] + source_loss
        else:
            losses["total"] = vae_losses["total"]

        return losses
