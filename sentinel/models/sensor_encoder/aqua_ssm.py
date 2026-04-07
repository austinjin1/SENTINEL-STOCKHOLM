"""AquaSSM: Continuous-time selective state space model for irregular sensor data.

Combines Mamba-style selective SSM efficiency with Neural CDE irregular-time
handling. Multi-scale SSM bank captures dynamics from hourly to yearly
timescales, with learned step size adapting to observation gaps.

Architecture:
    Input embedding -> MultiScaleSSMBank (8 timescales) -> GatedMixing -> [B, 256]
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Sensor configuration
NUM_PARAMETERS = 6  # pH, DO, turbidity, conductivity, temperature, ORP
OUTPUT_DIM = 256

# Characteristic timescales in seconds (log-spaced from 1h to 365d)
DEFAULT_TIMESCALES = [
    3600.0,       # 1 hour
    14400.0,      # 4 hours
    43200.0,      # 12 hours
    172800.0,     # 2 days
    604800.0,     # 7 days
    2592000.0,    # 30 days
    7776000.0,    # 90 days
    31536000.0,   # 365 days
]
NUM_SCALES = len(DEFAULT_TIMESCALES)


class StepSizeMLP(nn.Module):
    """Learned step size function: delta_t = f_theta(gap, x_prev).

    Maps the raw observation gap and previous hidden state summary to an
    effective step size that controls state transition dynamics.

    Small gaps -> near-linear dynamics (small delta_t).
    Large gaps -> complex state transitions (large delta_t).

    Args:
        hidden_dim: Dimension of the SSM hidden state.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        # Input: [gap_scalar, hidden_state_summary] -> delta_t
        self.net = nn.Sequential(
            nn.Linear(1 + hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),  # Ensure positive step size
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, gap: torch.Tensor, h_prev: torch.Tensor
    ) -> torch.Tensor:
        """Compute effective step size.

        Args:
            gap: Time gap since last observation [B, 1].
            h_prev: Previous hidden state [B, hidden_dim].

        Returns:
            Effective step size [B, 1].
        """
        inp = torch.cat([gap, h_prev], dim=-1)  # [B, 1 + hidden_dim]
        return self.net(inp)  # [B, 1]


class ContinuousTimeSSMCell(nn.Module):
    """Core continuous-time selective state space model cell.

    Implements a discretized linear SSM with learned step size:
        h_t = exp(A * delta_t) * h_{t-1} + B * delta_t * x_t
        y_t = C * h_t + D * x_t

    The step size delta_t adapts to irregular observation gaps via a
    learned function of the gap duration and previous state.

    Args:
        input_dim: Dimension of input features.
        hidden_dim: SSM state dimension.
        characteristic_timescale: Initial timescale in seconds for this cell.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        characteristic_timescale: float = 3600.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.characteristic_timescale = characteristic_timescale

        # State transition matrix A (diagonal for efficiency, initialized for stability)
        # Initialize as negative values scaled by characteristic timescale
        # so that exp(A * tau) decays with time constant ~ characteristic_timescale
        init_A = -torch.ones(hidden_dim) / characteristic_timescale
        self.log_neg_A = nn.Parameter(torch.log(-init_A))  # Store log(-A) for stability

        # Input-to-state projection B
        self.B = nn.Linear(input_dim, hidden_dim, bias=False)

        # State-to-output projection C
        self.C = nn.Linear(hidden_dim, input_dim, bias=False)

        # Skip connection D
        self.D = nn.Parameter(torch.ones(input_dim) * 0.1)

        # Learned step size
        self.step_size_fn = StepSizeMLP(hidden_dim)

        # Layer norm on output
        self.layer_norm = nn.LayerNorm(input_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.B.weight)
        nn.init.xavier_uniform_(self.C.weight)

    @property
    def A(self) -> torch.Tensor:
        """Diagonal of state transition matrix (guaranteed negative)."""
        return -torch.exp(self.log_neg_A)  # [hidden_dim]

    def forward_step(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor,
        gap: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single SSM step with continuous-time discretization.

        Args:
            x_t: Input at current timestep [B, input_dim].
            h_prev: Previous hidden state [B, hidden_dim].
            gap: Time since last observation [B, 1] (in seconds).

        Returns:
            y_t: Output [B, input_dim].
            h_t: Updated hidden state [B, hidden_dim].
        """
        # Compute effective step size from gap and previous state
        delta_t = self.step_size_fn(gap, h_prev)  # [B, 1]

        # Discretize: exact solution for diagonal A
        # Clamp the exponent to prevent overflow/underflow
        A_diag = self.A  # [hidden_dim], guaranteed negative
        exponent = A_diag.unsqueeze(0) * delta_t  # [B, hidden_dim]
        exponent = exponent.clamp(min=-20.0, max=0.0)  # exp(-20)≈2e-9, exp(0)=1
        A_bar = torch.exp(exponent)  # [B, hidden_dim], in (0, 1]
        B_bar = self.B(x_t) * delta_t  # [B, hidden_dim]

        # State update with hidden state clamping for stability
        h_t = A_bar * h_prev + B_bar  # [B, hidden_dim]
        h_t = h_t.clamp(min=-50.0, max=50.0)  # prevent hidden state explosion

        # Output
        y_t = self.C(h_t) + self.D * x_t  # [B, input_dim]
        y_t = self.layer_norm(y_t)

        return y_t, h_t

    def forward_sequence(
        self,
        x: torch.Tensor,
        delta_ts: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        h_init: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process full sequence through the SSM cell.

        Args:
            x: Input features [B, T, input_dim].
            delta_ts: Time gaps between observations [B, T] (seconds).
            mask: Valid observation mask [B, T, input_dim]. 1=valid, 0=missing.
            h_init: Initial hidden state [B, hidden_dim]. Default zeros.

        Returns:
            outputs: Per-timestep outputs [B, T, input_dim].
            h_final: Final hidden state [B, hidden_dim].
        """
        B, T, D = x.shape
        device = x.device

        if h_init is None:
            h_init = torch.zeros(B, self.hidden_dim, device=device)

        # Apply mask: zero out missing observations
        if mask is not None:
            x = x * mask

        outputs = []
        h = h_init
        for t in range(T):
            gap = delta_ts[:, t : t + 1]  # [B, 1]
            y_t, h = self.forward_step(x[:, t], h, gap)
            outputs.append(y_t)

        outputs = torch.stack(outputs, dim=1)  # [B, T, input_dim]
        return outputs, h


class GatedMixingLayer(nn.Module):
    """Gated mixing of multi-scale SSM outputs.

    Computes per-scale gates from the concatenation of all scale outputs,
    then produces a weighted combination.

    Args:
        input_dim: Per-scale feature dimension.
        num_scales: Number of SSM scales to combine.
    """

    def __init__(self, input_dim: int, num_scales: int = NUM_SCALES) -> None:
        super().__init__()
        self.num_scales = num_scales
        self.input_dim = input_dim

        # Gate network: concat of all scales -> per-scale gate
        self.gate_net = nn.Sequential(
            nn.Linear(input_dim * num_scales, input_dim * num_scales),
            nn.SiLU(inplace=True),
            nn.Linear(input_dim * num_scales, num_scales),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, scale_outputs: list[torch.Tensor]) -> torch.Tensor:
        """Mix multi-scale outputs with learned gates.

        Args:
            scale_outputs: List of num_scales tensors, each [B, dim].

        Returns:
            Mixed output [B, dim].
        """
        concat = torch.cat(scale_outputs, dim=-1)  # [B, num_scales * dim]
        gates = torch.sigmoid(self.gate_net(concat))  # [B, num_scales]

        # Weighted combination
        stacked = torch.stack(scale_outputs, dim=1)  # [B, num_scales, dim]
        gated = stacked * gates.unsqueeze(-1)  # [B, num_scales, dim]
        return gated.sum(dim=1)  # [B, dim]


class MultiScaleSSMBank(nn.Module):
    """Bank of parallel SSM channels at different characteristic timescales.

    Each channel captures dynamics at a specific timescale (1h to 365d),
    and a gated mixing layer combines them into a single representation.

    Args:
        input_dim: Input feature dimension per timestep.
        hidden_dim_per_scale: Hidden state dimension per SSM channel.
        timescales: List of characteristic timescales in seconds.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim_per_scale: int = 32,
        timescales: Optional[list[float]] = None,
    ) -> None:
        super().__init__()
        if timescales is None:
            timescales = DEFAULT_TIMESCALES
        self.timescales = timescales
        self.num_scales = len(timescales)
        self.hidden_dim_per_scale = hidden_dim_per_scale

        # Create one SSM cell per timescale
        self.ssm_cells = nn.ModuleList([
            ContinuousTimeSSMCell(
                input_dim=input_dim,
                hidden_dim=hidden_dim_per_scale,
                characteristic_timescale=tau,
            )
            for tau in timescales
        ])

        # Gated mixing of multi-scale outputs
        self.mixer = GatedMixingLayer(input_dim, self.num_scales)

    def forward(
        self,
        x: torch.Tensor,
        delta_ts: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """Process sequence through all SSM scales and mix.

        Args:
            x: Input features [B, T, input_dim].
            delta_ts: Time gaps [B, T] (seconds).
            mask: Observation mask [B, T, input_dim].

        Returns:
            mixed_output: Gated mixture of final states [B, input_dim].
            all_outputs: Per-scale temporal outputs, each [B, T, input_dim].
            all_finals: Per-scale final hidden states, each [B, hidden_dim].
        """
        all_outputs = []
        all_finals = []
        scale_embeddings = []

        for cell in self.ssm_cells:
            outputs, h_final = cell.forward_sequence(x, delta_ts, mask)
            all_outputs.append(outputs)
            all_finals.append(h_final)
            # Use last timestep output as scale embedding
            scale_embeddings.append(outputs[:, -1])  # [B, input_dim]

        mixed = self.mixer(scale_embeddings)  # [B, input_dim]
        return mixed, all_outputs, all_finals


class AquaSSM(nn.Module):
    """Full AquaSSM model: continuous-time multi-scale SSM for water quality.

    Processes irregular time series of 6 water quality parameters through
    a multi-scale SSM bank, producing a fixed 256-dim embedding.

    Input:
        timestamps: Absolute timestamps [B, T] (Unix seconds).
        values: Sensor readings [B, T, 6].
        delta_ts: Time gaps between observations [B, T] (seconds).
        masks: Per-parameter validity masks [B, T, 6]. 1=valid, 0=missing.

    Output:
        embedding: [B, 256] final embedding (last hidden state pooled).
        temporal_outputs: Per-scale temporal features for auxiliary heads.
        scale_finals: Per-scale final hidden states.

    Args:
        num_params: Number of sensor parameters.
        output_dim: Output embedding dimension.
        hidden_dim_per_scale: Hidden state size per SSM channel.
        timescales: Characteristic timescales in seconds.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        num_params: int = NUM_PARAMETERS,
        output_dim: int = OUTPUT_DIM,
        hidden_dim_per_scale: int = 32,
        timescales: Optional[list[float]] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_params = num_params
        self.output_dim = output_dim

        # Input embedding: project raw parameters to internal dimension
        self.input_proj = nn.Sequential(
            nn.Linear(num_params, output_dim),
            nn.LayerNorm(output_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Multi-scale SSM bank operates on projected features
        self.ssm_bank = MultiScaleSSMBank(
            input_dim=output_dim,
            hidden_dim_per_scale=hidden_dim_per_scale,
            timescales=timescales,
        )

        # Output projection from mixed scale output to final embedding
        self.output_proj = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in [self.input_proj, self.output_proj]:
            for sub in m.modules():
                if isinstance(sub, nn.Linear):
                    nn.init.xavier_uniform_(sub.weight)
                    if sub.bias is not None:
                        nn.init.zeros_(sub.bias)

    def forward(
        self,
        timestamps: torch.Tensor,
        values: torch.Tensor,
        delta_ts: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """Forward pass through AquaSSM.

        Args:
            timestamps: Absolute timestamps [B, T] (seconds, for reference).
            values: Sensor readings [B, T, num_params].
            delta_ts: Time gaps between observations [B, T] (seconds).
            masks: Per-parameter validity mask [B, T, num_params]. Default all-valid.

        Returns:
            Dict with:
                'embedding': Final SSM embedding [B, output_dim].
                'temporal_outputs': List of per-scale outputs [B, T, output_dim].
                'scale_finals': List of per-scale final hidden states.
        """
        B, T, P = values.shape

        # Project input parameters to internal dimension
        x = self.input_proj(values)  # [B, T, output_dim]

        # Expand mask to match projected dimension if provided
        projected_mask: Optional[torch.Tensor] = None
        if masks is not None:
            # Any parameter missing -> mask entire projected vector at that timestep
            # Use min across params: if any param missing, mask the timestep partially
            # Actually, broadcast: [B, T, P] -> [B, T, 1] -> [B, T, output_dim]
            timestep_valid = masks.min(dim=-1, keepdim=True).values  # [B, T, 1]
            projected_mask = timestep_valid.expand_as(x)

        # Process through multi-scale SSM bank
        mixed, temporal_outputs, scale_finals = self.ssm_bank(
            x, delta_ts, projected_mask
        )  # mixed: [B, output_dim]

        # Final output projection
        embedding = self.output_proj(mixed)  # [B, output_dim]

        return {
            "embedding": embedding,
            "temporal_outputs": temporal_outputs,
            "scale_finals": scale_finals,
        }

    def forward_with_values(
        self,
        x: torch.Tensor,
        delta_ts: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Simplified forward for compatibility with MPP/anomaly modules.

        Accepts raw values [B, T, P] and returns (embedding, temporal_outputs).
        Uses default 15-min gaps if delta_ts not provided.

        Args:
            x: Sensor readings [B, T, P].
            delta_ts: Optional time gaps [B, T]. Default 900s (15 min).
            masks: Optional validity mask [B, T, P].

        Returns:
            embedding: [B, output_dim].
            temporal_outputs: List of per-scale outputs [B, T, output_dim].
        """
        B, T, P = x.shape
        device = x.device

        if delta_ts is None:
            delta_ts = torch.full((B, T), 900.0, device=device)

        timestamps = torch.zeros(B, T, device=device)  # placeholder

        result = self.forward(timestamps, x, delta_ts, masks)
        return result["embedding"], result["temporal_outputs"]
