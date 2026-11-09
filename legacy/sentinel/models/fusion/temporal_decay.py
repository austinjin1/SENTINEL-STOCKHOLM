"""Learned temporal decay functions for multimodal fusion.

Each modality-pair's information decays at a characteristic rate that
reflects how informative one modality's past observation is when a new
observation arrives from another modality.  The decay matrix is
symmetric: ``tau[i, j] = tau[j, i]``.

Physically-motivated priors per modality:

* **Satellite** imagery changes slowly (cloud cover, land use) --
  expected tau ~5 days (432 000 s).
* **Sensor** readings reflect fast-changing water chemistry --
  expected tau ~2 hours (7 200 s).
* **Microbial** community composition shifts over days --
  expected tau ~7 days (604 800 s).
* **Molecular** pathway activation persists for days --
  expected tau ~3 days (259 200 s).
* **Behavioral** patterns (wildlife/human activity) change quickly --
  expected tau ~5 minutes (300 s).

The pairwise prior ``tau[i, j]`` is initialized as the geometric mean
of the per-modality priors: ``sqrt(tau_i * tau_j)``.

The decay parameters are **learned end-to-end** but initialized to the
physically-motivated pairwise priors.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn

from sentinel.models.fusion.embedding_registry import (
    MODALITY_IDS,
    NUM_MODALITIES,
)

# Physically-motivated priors (seconds).
DEFAULT_TAU_PRIORS: Dict[str, float] = {
    "satellite": 432_000.0,    # ~5 days
    "sensor": 7_200.0,         # ~2 hours
    "microbial": 604_800.0,    # ~7 days
    "molecular": 259_200.0,    # ~3 days
    "behavioral": 300.0,       # ~5 minutes
}

# Minimum allowed tau to avoid division-by-zero / exploding gradients.
_TAU_MIN: float = 60.0  # 1 minute floor


def _build_pairwise_priors(
    tau_priors: Dict[str, float],
) -> torch.Tensor:
    """Build a [NUM_MODALITIES, NUM_MODALITIES] matrix of geometric-mean priors."""
    priors = torch.zeros(NUM_MODALITIES, NUM_MODALITIES)
    for i, mid_i in enumerate(MODALITY_IDS):
        for j, mid_j in enumerate(MODALITY_IDS):
            priors[i, j] = math.sqrt(tau_priors[mid_i] * tau_priors[mid_j])
    return priors


class TemporalDecay(nn.Module):
    """Per-modality-pair exponential decay with learned time constants.

    For staleness ``s`` between modalities *i* and *j*:

    .. math::

        w(s) = \\exp\\bigl(-s / \\tau_{ij}\\bigr)

    ``tau`` is stored in log-space to guarantee positivity and numerical
    stability during gradient descent.  The matrix is kept **symmetric**
    by learning only the upper triangle and mirroring it.

    Args:
        tau_priors: Initial per-modality tau values (seconds).  Pairwise
            priors are computed as geometric means.
    """

    def __init__(
        self,
        tau_priors: Dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        tau_priors = tau_priors or DEFAULT_TAU_PRIORS

        pairwise = _build_pairwise_priors(tau_priors)

        # Store log(tau - tau_min) so that tau = exp(log_tau) + _TAU_MIN
        # is always > _TAU_MIN.
        init_values = torch.log(torch.clamp(pairwise - _TAU_MIN, min=1.0))
        self.log_tau = nn.Parameter(init_values)  # [M, M]

        # Index map (not a parameter).
        self._mid_to_idx: Dict[str, int] = {
            mid: i for i, mid in enumerate(MODALITY_IDS)
        }

    # ------------------------------------------------------------------
    # Symmetry enforcement
    # ------------------------------------------------------------------

    def _symmetric_log_tau(self) -> torch.Tensor:
        """Return symmetrized log_tau: (log_tau + log_tau^T) / 2."""
        return (self.log_tau + self.log_tau.t()) / 2.0

    def _symmetric_tau(self) -> torch.Tensor:
        """Return the full symmetric positive tau matrix."""
        return torch.exp(self._symmetric_log_tau()) + _TAU_MIN

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_tau(self, modality_i: str, modality_j: str) -> torch.Tensor:
        """Return the current tau for modality pair (i, j) in seconds."""
        i = self._mid_to_idx[modality_i]
        j = self._mid_to_idx[modality_j]
        return self._symmetric_tau()[i, j]

    def get_all_tau(self) -> torch.Tensor:
        """Return the full ``[M, M]`` tau matrix."""
        return self._symmetric_tau()

    def forward(
        self,
        staleness: torch.Tensor,
        source_modality: str,
    ) -> torch.Tensor:
        """Compute decay weights from *source_modality* to all modalities.

        Args:
            staleness: Scalar or 1-D tensor of staleness values (seconds),
                shape ``[K]`` where ``K = NUM_MODALITIES``.
            source_modality: The modality that just produced data.  The
                decay is computed as ``exp(-staleness[j] / tau[source, j])``
                for every target modality *j*.

        Returns:
            Decay weights of shape ``[K]`` in ``(0, 1]``.
        """
        i = self._mid_to_idx[source_modality]
        tau_row = self._symmetric_tau()[i]  # [K]
        return torch.exp(-staleness / tau_row)

    def forward_all(
        self,
        staleness_vec: torch.Tensor,
        query_modality: str,
    ) -> torch.Tensor:
        """Compute decay weights for all modalities relative to *query_modality*.

        Args:
            staleness_vec: Tensor of shape ``[NUM_MODALITIES]`` giving
                the staleness in seconds for each modality (ordered as
                :data:`MODALITY_IDS`).
            query_modality: The modality whose perspective we are computing
                from.

        Returns:
            Decay weights of shape ``[NUM_MODALITIES]`` in ``(0, 1]``.
        """
        i = self._mid_to_idx[query_modality]
        tau_row = self._symmetric_tau()[i]  # [K]
        return torch.exp(-staleness_vec / tau_row)

    def compute_log_bias(
        self,
        staleness_per_modality: torch.Tensor,
        confidences_per_modality: torch.Tensor,
        query_modality: str,
    ) -> torch.Tensor:
        """Compute additive log-bias for attention logits.

        Combines temporal decay with confidence:

        .. math::

            \\text{bias}_j = \\log\\bigl(
                \\exp(-s_j / \\tau_{q,j}) \\times c_j + \\epsilon
            \\bigr)

        Args:
            staleness_per_modality: ``[B, K]`` staleness per modality.
            confidences_per_modality: ``[B, K]`` confidence per modality.
            query_modality: The triggering modality.

        Returns:
            Log-bias ``[B, K]`` suitable for adding to attention logits.
        """
        i = self._mid_to_idx[query_modality]
        tau_row = self._symmetric_tau()[i]  # [K]
        # tau_row is on the parameter device; move staleness there
        decay = torch.exp(-staleness_per_modality / tau_row.unsqueeze(0))  # [B, K]
        combined = decay * confidences_per_modality
        return torch.log(combined.clamp(min=1e-8))
