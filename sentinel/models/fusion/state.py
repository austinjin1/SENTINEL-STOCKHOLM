"""GRU-based waterway state module.

Maintains a running "state of the waterway" vector that persists
between observations and smoothly integrates new evidence from the
cross-modal attention output.  The GRU cell is a natural fit because:

1. It handles variable inter-observation intervals gracefully.
2. The update gate learns how much new evidence to incorporate vs
   how much prior state to retain.
3. It is lightweight (single cell, no sequence unrolling needed since
   we receive one fused embedding per event).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from sentinel.models.fusion.embedding_registry import SHARED_EMBEDDING_DIM


class WaterwayStateGRU(nn.Module):
    """GRU cell that maintains a persistent waterway state.

    At each observation event the fused cross-modal embedding is fed as
    input and the hidden state is updated in-place.  The hidden state
    can be queried at any time as the current "state of the waterway".

    Args:
        state_dim: Dimensionality of the hidden state and input.
            Must match :data:`SHARED_EMBEDDING_DIM`.
        dropout: Dropout applied to the GRU input during training.
    """

    def __init__(
        self,
        state_dim: int = SHARED_EMBEDDING_DIM,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim

        self.gru_cell = nn.GRUCell(
            input_size=state_dim,
            hidden_size=state_dim,
        )
        self.input_dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(state_dim)

        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Orthogonal init for GRU recurrent weights; Xavier for input."""
        for name, param in self.gru_cell.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def initial_state(
        self,
        batch_size: int = 1,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Return a zero-initialized hidden state.

        Args:
            batch_size: Number of independent waterway tracks.
            device: Target device.

        Returns:
            Tensor of shape ``[B, state_dim]``.
        """
        return torch.zeros(
            batch_size,
            self.state_dim,
            device=device,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        fused: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Integrate a new fused observation into the waterway state.

        Args:
            fused: Cross-modal attention output, shape ``[B, state_dim]``
                or ``[state_dim]``.
            state: Previous hidden state, shape ``[B, state_dim]``.
                If ``None``, a zero state is created automatically.

        Returns:
            output: Layer-normed updated state for downstream heads,
                shape ``[B, state_dim]``.
            new_state: Raw GRU hidden state to feed back at next step,
                shape ``[B, state_dim]``.
        """
        if fused.dim() == 1:
            fused = fused.unsqueeze(0)
        B = fused.shape[0]

        if state is None:
            state = self.initial_state(B, device=fused.device)

        x = self.input_dropout(fused)
        new_state = self.gru_cell(x, state)
        output = self.layer_norm(new_state)

        return output, new_state
