"""Hierarchical Environmental Modality Alignment (HEMA).

Physics-informed contrastive learning that aligns representations across
environmental monitoring modalities (satellite imagery, in-situ sensors,
microbial sequencing, etc.) into a shared embedding space.

Positive pairs are formed from *physical co-occurrence* --- a satellite pixel
and a sensor reading from the same location-time window --- rather than from
data augmentation.  Negative pairs are drawn from different locations or
different time windows.  An InfoNCE contrastive objective with a learned
temperature is augmented with physics-informed similarity adjustments so that
known cross-modal correlations (e.g., high chlorophyll-a satellite signal
should align with low dissolved oxygen sensor reading) are respected.

The :class:`ModalityAligner` projects per-modality embeddings into the shared
space and provides a transfer error bound via a MINE-based mutual information
estimator.

References
----------
[1] Oord, van den, Li, & Vinyals (2018). "Representation Learning with
    Contrastive Predictive Coding." arXiv:1807.03748.
[2] Belghazi et al. (2018). "Mutual Information Neural Estimation."
    ICML 2018.
[3] Radford et al. (2021). "Learning Transferable Visual Models From
    Natural Language Supervision." (CLIP) ICML 2021.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from sentinel.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Physics-informed correlation priors
# ---------------------------------------------------------------------------

# Default known cross-modal correlations.  Positive values mean the two
# signals should be positively correlated in alignment space; negative
# values mean anti-correlated (e.g., high chlorophyll satellite signal
# typically coincides with low dissolved oxygen).
_DEFAULT_PHYSICS_PRIORS: Dict[Tuple[str, str], float] = {
    ("satellite_chlorophyll", "sensor_dissolved_oxygen"): -0.6,
    ("satellite_chlorophyll", "sensor_chlorophyll"): 0.9,
    ("satellite_turbidity", "sensor_turbidity"): 0.85,
    ("satellite_temperature", "sensor_temperature"): 0.95,
    ("sensor_dissolved_oxygen", "microbial_anaerobic_ratio"): -0.7,
}


# ---------------------------------------------------------------------------
# HEMA contrastive loss
# ---------------------------------------------------------------------------


class HEMALoss(nn.Module):
    """InfoNCE contrastive loss with physics-informed similarity adjustment.

    Given a batch of positive pairs ``(z_a, z_b)`` where ``z_a`` and ``z_b``
    are embeddings from two different modalities for the same
    location-time window, and the remaining batch elements serve as
    negatives, the loss is:

    .. math::

        \\mathcal{L} = -\\log \\frac{\\exp(\\text{sim}(z_a^i, z_b^i) / \\tau)}
            {\\sum_{j=1}^{N} \\exp(\\text{sim}(z_a^i, z_b^j) / \\tau)}

    where :math:`\\tau` is a learned temperature parameter and ``sim``
    includes a physics-informed correction term.

    Args:
        embed_dim: Dimensionality of the shared embedding space.
        init_temperature: Initial value for the learned log-temperature.
        physics_weight: Scalar weight for the physics-informed correction.
        physics_priors: Optional dict mapping ``(modality_a, modality_b)``
            pairs to expected correlation signs/magnitudes.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        init_temperature: float = 0.07,
        physics_weight: float = 0.1,
        physics_priors: Optional[Dict[Tuple[str, str], float]] = None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.physics_weight = physics_weight
        self.physics_priors = physics_priors or _DEFAULT_PHYSICS_PRIORS

        # Learned log-temperature (clamped during forward for stability).
        self.log_temperature = nn.Parameter(
            torch.tensor(math.log(init_temperature))
        )

        # Learnable physics-correction MLP: takes concatenated embeddings
        # and outputs a scalar similarity correction.
        self.physics_correction = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, 1),
        )

    # -- helpers -------------------------------------------------------------

    @property
    def temperature(self) -> torch.Tensor:
        """Current temperature (always positive)."""
        return self.log_temperature.exp().clamp(min=1e-4, max=100.0)

    def _cosine_similarity_matrix(
        self, z_a: torch.Tensor, z_b: torch.Tensor
    ) -> torch.Tensor:
        """Compute pairwise cosine similarity ``(N, N)``."""
        z_a = F.normalize(z_a, dim=-1)
        z_b = F.normalize(z_b, dim=-1)
        return z_a @ z_b.T

    def _physics_correction_matrix(
        self, z_a: torch.Tensor, z_b: torch.Tensor
    ) -> torch.Tensor:
        """Compute pairwise physics-informed correction ``(N, N)``."""
        N = z_a.size(0)
        # Expand for pairwise: (N, 1, D) x (1, N, D) -> (N, N, 2D)
        z_a_exp = z_a.unsqueeze(1).expand(N, N, -1)
        z_b_exp = z_b.unsqueeze(0).expand(N, N, -1)
        paired = torch.cat([z_a_exp, z_b_exp], dim=-1)  # (N, N, 2D)
        correction = self.physics_correction(paired).squeeze(-1)  # (N, N)
        return correction

    # -- forward -------------------------------------------------------------

    def forward(
        self,
        z_a: torch.Tensor,
        z_b: torch.Tensor,
        modality_a: str = "",
        modality_b: str = "",
    ) -> Dict[str, torch.Tensor]:
        """Compute HEMA contrastive loss.

        Args:
            z_a: Embeddings from modality A, shape ``(N, D)``.
            z_b: Embeddings from modality B, shape ``(N, D)``.
                  ``z_a[i]`` and ``z_b[i]`` form a positive pair.
            modality_a: Name of modality A (for physics prior lookup).
            modality_b: Name of modality B (for physics prior lookup).

        Returns:
            Dictionary with keys ``loss`` (scalar), ``logits_per_a`` ``(N, N)``,
            ``logits_per_b`` ``(N, N)``, ``temperature`` (scalar).
        """
        assert z_a.shape == z_b.shape, (
            f"Shape mismatch: z_a {z_a.shape} vs z_b {z_b.shape}"
        )
        N = z_a.size(0)

        # Base cosine similarity
        sim = self._cosine_similarity_matrix(z_a, z_b)  # (N, N)

        # Physics-informed correction
        if self.physics_weight > 0.0:
            correction = self._physics_correction_matrix(z_a, z_b)
            sim = sim + self.physics_weight * correction

        # Scale by learned temperature
        logits = sim / self.temperature  # (N, N)

        # Symmetric InfoNCE: loss from both directions
        labels = torch.arange(N, device=z_a.device)
        loss_a = F.cross_entropy(logits, labels)
        loss_b = F.cross_entropy(logits.T, labels)
        loss = (loss_a + loss_b) / 2.0

        # Physics prior regularization: if we know the expected correlation
        # sign, penalize misalignment on the diagonal (positive pairs).
        prior_key = (modality_a, modality_b)
        prior_key_rev = (modality_b, modality_a)
        expected_corr = self.physics_priors.get(
            prior_key, self.physics_priors.get(prior_key_rev, None)
        )
        if expected_corr is not None:
            diag_sim = torch.diagonal(sim)  # (N,)
            # Push diagonal similarity toward expected correlation sign
            target = torch.full_like(diag_sim, expected_corr)
            physics_reg = F.mse_loss(torch.tanh(diag_sim), target)
            loss = loss + self.physics_weight * physics_reg

        return {
            "loss": loss,
            "logits_per_a": logits,
            "logits_per_b": logits.T,
            "temperature": self.temperature.detach(),
        }


# ---------------------------------------------------------------------------
# MINE mutual information estimator
# ---------------------------------------------------------------------------


class _MINEStatisticsNetwork(nn.Module):
    """Statistics network T(x, y) for MINE estimation.

    The mutual information lower bound is:

    .. math::

        I(X; Y) \\geq \\mathbb{E}_{p(x,y)}[T(x,y)]
            - \\log \\mathbb{E}_{p(x)p(y)}[e^{T(x,y)}]
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Evaluate T(x, y). Both inputs shape ``(N, D)``."""
        return self.net(torch.cat([x, y], dim=-1))  # (N, 1)


# ---------------------------------------------------------------------------
# Modality aligner
# ---------------------------------------------------------------------------


class ModalityAligner(nn.Module):
    """Projects per-modality embeddings into a shared alignment space.

    Each registered modality gets its own projection head.  After
    projection, embeddings live in a common ``align_dim``-dimensional
    space where cross-modal similarity is meaningful.

    A MINE-based mutual information estimator quantifies how much
    information is shared between any two modality projections, which
    provides an upper bound on the transfer error between modalities
    (Ben-David et al., 2010).

    Args:
        modality_dims: Mapping from modality name to its native
            embedding dimensionality.
        align_dim: Dimensionality of the shared alignment space.
        hidden_dim: Hidden layer size in projection heads.
        mine_hidden_dim: Hidden layer size in the MINE statistics network.
    """

    def __init__(
        self,
        modality_dims: Dict[str, int],
        align_dim: int = 256,
        hidden_dim: int = 512,
        mine_hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.align_dim = align_dim
        self.modality_names = sorted(modality_dims.keys())

        # Per-modality projection heads
        self.projectors = nn.ModuleDict()
        for name, in_dim in modality_dims.items():
            self.projectors[name] = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, align_dim),
                nn.LayerNorm(align_dim),
            )

        # MINE statistics network (shared across modality pairs)
        self._mine_net = _MINEStatisticsNetwork(align_dim, mine_hidden_dim)
        # Exponential moving average for MINE (for stable gradients)
        self.register_buffer("_mine_ema", torch.tensor(1.0))

    def project(self, embedding: torch.Tensor, modality: str) -> torch.Tensor:
        """Project a modality-native embedding into the alignment space.

        Args:
            embedding: Shape ``(N, D_modality)``.
            modality: Name of the modality (must be registered).

        Returns:
            L2-normalized projection, shape ``(N, align_dim)``.
        """
        if modality not in self.projectors:
            raise ValueError(
                f"Unknown modality '{modality}'. "
                f"Registered: {list(self.projectors.keys())}"
            )
        projected = self.projectors[modality](embedding)
        return F.normalize(projected, dim=-1)

    def estimate_mutual_information(
        self,
        embed_a: torch.Tensor,
        embed_b: torch.Tensor,
        *,
        ema_decay: float = 0.99,
    ) -> torch.Tensor:
        """Estimate mutual information between two embedding sets via MINE.

        Uses the Donsker-Varadhan representation:

        .. math::

            I(A; B) \\geq \\sup_T \\{\\mathbb{E}_{p(a,b)}[T(a,b)]
                - \\log \\mathbb{E}_{p(a)p(b)}[e^{T(a,b)}]\\}

        The exponential moving average trick from Belghazi et al. (2018) is
        used for stable gradient estimation.

        Args:
            embed_a: Embeddings from modality A, shape ``(N, D)``.
            embed_b: Embeddings from modality B, shape ``(N, D)``.
            ema_decay: Decay factor for the exponential moving average.

        Returns:
            Estimated MI (scalar tensor, in nats).
        """
        N = embed_a.size(0)

        # Joint samples: (a_i, b_i)
        t_joint = self._mine_net(embed_a, embed_b)  # (N, 1)

        # Marginal samples: (a_i, b_{\pi(i)}) with random permutation
        perm = torch.randperm(N, device=embed_b.device)
        t_marginal = self._mine_net(embed_a, embed_b[perm])  # (N, 1)

        # Donsker-Varadhan bound with EMA correction
        joint_mean = t_joint.mean()
        exp_marginal = t_marginal.exp()
        exp_mean = exp_marginal.mean()

        # Update EMA for stable gradient (detach to avoid backprop through EMA)
        if self.training:
            self._mine_ema = (
                ema_decay * self._mine_ema + (1.0 - ema_decay) * exp_mean.detach()
            )

        # Corrected log-mean-exp using EMA
        mi_estimate = joint_mean - torch.log(exp_mean + 1e-8)

        return mi_estimate.squeeze()

    def transfer_error_bound(
        self,
        embed_a: torch.Tensor,
        embed_b: torch.Tensor,
    ) -> torch.Tensor:
        """Upper bound on cross-modal transfer error.

        Based on the information-theoretic bound: transfer error is bounded
        by a decreasing function of the mutual information between the
        aligned representations (Ben-David et al., 2010; Xu & Raginsky, 2017).

        .. math::

            \\epsilon_{\\text{transfer}} \\leq \\sqrt{\\frac{2 \\ln 2}{I(A; B)}}

        Args:
            embed_a: Aligned embeddings from modality A, shape ``(N, D)``.
            embed_b: Aligned embeddings from modality B, shape ``(N, D)``.

        Returns:
            Estimated upper bound on transfer error (scalar).
        """
        mi = self.estimate_mutual_information(embed_a, embed_b)
        # Clamp MI to be positive (MINE can underestimate)
        mi_clamped = mi.clamp(min=1e-6)
        bound = torch.sqrt(2.0 * math.log(2.0) / mi_clamped)
        return bound

    def forward(
        self,
        embeddings: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Project all modality embeddings into the shared space.

        Args:
            embeddings: Mapping from modality name to embedding tensor,
                each of shape ``(N, D_modality)``.

        Returns:
            Mapping from modality name to aligned embedding, each of
            shape ``(N, align_dim)``.
        """
        aligned = {}
        for name, emb in embeddings.items():
            aligned[name] = self.project(emb, name)
        return aligned
