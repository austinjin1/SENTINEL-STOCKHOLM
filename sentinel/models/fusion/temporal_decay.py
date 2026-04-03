"""Learned temporal decay functions for multimodal fusion.

Each modality's information decays at a characteristic rate that
reflects the physical persistence of the underlying signal:

* **Satellite** imagery changes slowly (cloud cover, land use) --
  expected tau ~5 days (432 000 s).
* **Sensor** readings reflect fast-changing water chemistry --
  expected tau ~2 hours (7 200 s).
* **Microbial** community composition shifts over days --
  expected tau ~7 days (604 800 s).
* **Molecular** pathway activation persists for days --
  expected tau ~3 days (259 200 s).

The decay parameters ``tau`` are **learned end-to-end** but initialized
to the physically-motivated priors above.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn

from sentinel.models.fusion.embedding_registry import MODALITY_IDS

# Physically-motivated priors (seconds).
DEFAULT_TAU_PRIORS: Dict[str, float] = {
    "satellite": 432_000.0,    # ~5 days
    "sensor": 7_200.0,         # ~2 hours
    "microbial": 604_800.0,    # ~7 days
    "molecular": 259_200.0,    # ~3 days
}

# Minimum allowed tau to avoid division-by-zero / exploding gradients.
_TAU_MIN: float = 60.0  # 1 minute floor


class TemporalDecay(nn.Module):
    """Per-modality exponential decay with learned time constants.

    For staleness ``s`` and learned time constant ``tau``:

    .. math::

        w(s) = \\exp\\bigl(-s / \\tau\\bigr)

    ``tau`` is stored in log-space to guarantee positivity and numerical
    stability during gradient descent.

    Args:
        tau_priors: Initial tau values per modality (seconds).  Defaults
            to :data:`DEFAULT_TAU_PRIORS`.
    """

    def __init__(
        self,
        tau_priors: Dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        tau_priors = tau_priors or DEFAULT_TAU_PRIORS

        # Store log(tau) so that tau = exp(log_tau) + _TAU_MIN is always
        # positive.  Using exp ensures exact recovery of the prior at
        # initialization and well-behaved gradients for large tau values.
        init_values = torch.tensor(
            [math.log(max(tau_priors[mid] - _TAU_MIN, 1.0)) for mid in MODALITY_IDS],
            dtype=torch.float32,
        )
        self.log_tau = nn.Parameter(init_values)  # [num_modalities]

        # Handy index map (not a parameter).
        self._mid_to_idx: Dict[str, int] = {
            mid: i for i, mid in enumerate(MODALITY_IDS)
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_tau(self, modality_id: str) -> torch.Tensor:
        """Return the current (positive) tau for *modality_id* in seconds."""
        idx = self._mid_to_idx[modality_id]
        return torch.exp(self.log_tau[idx]) + _TAU_MIN

    def get_all_tau(self) -> Dict[str, torch.Tensor]:
        """Return a dict mapping every modality to its current tau."""
        taus = torch.exp(self.log_tau) + _TAU_MIN
        return {mid: taus[i] for i, mid in enumerate(MODALITY_IDS)}

    def forward(
        self,
        staleness: torch.Tensor,
        modality_id: str,
    ) -> torch.Tensor:
        """Compute decay weight for a single modality.

        Args:
            staleness: Scalar or tensor of staleness values (seconds).
                Must be non-negative.
            modality_id: One of :data:`MODALITY_IDS`.

        Returns:
            Decay weight(s) in ``(0, 1]``, same shape as *staleness*.
        """
        tau = self.get_tau(modality_id)
        return torch.exp(-staleness / tau)

    def forward_all(
        self,
        staleness_vec: torch.Tensor,
    ) -> torch.Tensor:
        """Compute decay weights for all modalities simultaneously.

        Args:
            staleness_vec: Tensor of shape ``[num_modalities]`` giving
                the staleness in seconds for each modality (ordered as
                :data:`MODALITY_IDS`).

        Returns:
            Decay weights of shape ``[num_modalities]`` in ``(0, 1]``.
        """
        taus = torch.exp(self.log_tau) + _TAU_MIN
        return torch.exp(-staleness_vec / taus)
