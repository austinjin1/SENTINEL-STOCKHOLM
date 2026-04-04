"""Learned latent array for Perceiver IO fusion.

Replaces the single 256-dim GRU hidden state with an array of N=256
learned latent vectors, each of dimension D=256.  The latent array
serves as a compressed "state of the waterway" representation that is
iteratively refined through cross-attention with incoming modality
tokens.

The latent array is a set of **learnable parameters** -- during
inference they are initialized to the learned values and then updated
via cross-attention within each forward pass (the parameters themselves
provide the initialization; gradient updates during training shape the
prior).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from sentinel.models.fusion.embedding_registry import SHARED_EMBEDDING_DIM


class LatentArray(nn.Module):
    """Array of learned latent vectors for Perceiver IO.

    The latent array acts as a bottleneck that compresses information
    from all modalities into a fixed-size representation.  Cross-
    attention writes information *into* the latents (encode step), self-
    attention mixes information *within* the latents (process step), and
    a decode cross-attention reads information *out* of the latents.

    Args:
        num_latents: Number of latent vectors ``N``.
        latent_dim: Dimensionality of each latent vector ``D``.
    """

    def __init__(
        self,
        num_latents: int = 256,
        latent_dim: int = SHARED_EMBEDDING_DIM,
    ) -> None:
        super().__init__()
        self.num_latents = num_latents
        self.latent_dim = latent_dim

        # Learnable initial latent array.
        self.latents = nn.Parameter(
            torch.randn(num_latents, latent_dim) * 0.02
        )

    def get_latents(self, batch_size: int = 1) -> torch.Tensor:
        """Return the latent array expanded to a batch dimension.

        Args:
            batch_size: Number of independent waterway tracks.

        Returns:
            Tensor of shape ``[B, N, D]``.
        """
        return self.latents.unsqueeze(0).expand(batch_size, -1, -1)

    def extra_repr(self) -> str:
        return (
            f"num_latents={self.num_latents}, "
            f"latent_dim={self.latent_dim}"
        )
