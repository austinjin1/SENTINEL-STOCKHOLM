"""Neural ODE on the Aitchison tangent space for temporal community trajectories.

Microbial communities evolve on the simplex (compositions must sum to 1).
Rather than modeling dynamics directly on this constrained space, we work
in CLR coordinates (the Aitchison tangent space) where standard ODE tools
apply without constraint violations.

Healthy communities oscillate seasonally within a bounded region of CLR
space. Contamination events push trajectories outside this region:
- Acute contamination: sudden jumps in CLR space
- Chronic degradation: slow directional drift

The Neural ODE learns the healthy vector field and detects deviations.

References:
    Chen et al. (2018). Neural Ordinary Differential Equations. NeurIPS.
    Pawlowsky-Glahn & Egozcue (2001). Geometric approach to statistical
        analysis on the simplex.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CLRVectorField(nn.Module):
    """Learned vector field on the CLR tangent space.

    Defines dx/dt = f(x, t) where x is the CLR-transformed community state
    and t is time. The vector field is constrained to produce outputs that
    sum to zero (CLR constraint: compositions have zero-sum log-ratios).

    Args:
        state_dim: Dimension of the CLR state vector. Default 256.
        hidden_dim: Hidden layer dimension. Default 512.
        time_embed_dim: Dimension of time embedding. Default 32.
    """

    def __init__(
        self,
        state_dim: int = 256,
        hidden_dim: int = 512,
        time_embed_dim: int = 32,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim

        # Time embedding: sinusoidal encoding of scalar time
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.GELU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Vector field network: (state, time) -> dx/dt
        self.net = nn.Sequential(
            nn.Linear(state_dim + time_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Compute vector field dx/dt at state x and time t.

        Args:
            t: Scalar time value (0-dimensional or 1-dimensional tensor).
            x: CLR state vector [B, state_dim].

        Returns:
            dx/dt: Time derivative in CLR space [B, state_dim].
                Centered (zero-sum) to maintain CLR constraint.
        """
        B = x.shape[0]

        # Embed time
        t_val = t.float().reshape(1, 1).expand(B, 1)
        t_emb = self.time_embed(t_val)  # [B, time_embed_dim]

        # Concatenate state and time
        xt = torch.cat([x, t_emb], dim=-1)

        # Compute vector field
        dxdt = self.net(xt)

        # Center to maintain CLR constraint (sum to zero)
        dxdt = dxdt - dxdt.mean(dim=-1, keepdim=True)

        return dxdt


class SimplexNeuralODE(nn.Module):
    """Neural ODE on the Aitchison tangent space for temporal community trajectories.

    Integrates a learned vector field in CLR coordinates to model how microbial
    communities evolve over time. Detects anomalous trajectory deviations that
    indicate contamination events.

    Uses a simple Euler integrator by default (no torchdiffeq dependency),
    with optional support for adaptive-step solvers if torchdiffeq is available.

    Args:
        input_dim: Dimension of CLR-transformed community vector. Default 5000.
        latent_dim: Dimension of trajectory latent embedding. Default 256.
        hidden_dim: Hidden dimension for vector field and encoder. Default 512.
        n_euler_steps: Number of Euler integration steps between timepoints.
            Default 10.
        dropout: Dropout rate. Default 0.1.
    """

    def __init__(
        self,
        input_dim: int = 5000,
        latent_dim: int = 256,
        hidden_dim: int = 512,
        n_euler_steps: int = 10,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_euler_steps = n_euler_steps

        # Encoder: project CLR community vector to latent ODE state
        self.state_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

        # Learned vector field on CLR tangent space
        self.vector_field = CLRVectorField(
            state_dim=latent_dim,
            hidden_dim=hidden_dim,
        )

        # Trajectory aggregator: summarize the full trajectory into a single embedding
        self.trajectory_attention = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )
        self.trajectory_norm = nn.LayerNorm(latent_dim)

        # Anomaly detection head: compare predicted vs observed trajectory
        self.anomaly_head = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # Running statistics for anomaly score normalization
        self.register_buffer("anomaly_mean", torch.tensor(0.0))
        self.register_buffer("anomaly_std", torch.tensor(1.0))

        # Check for torchdiffeq
        self._has_torchdiffeq = False
        try:
            import torchdiffeq
            self._has_torchdiffeq = True
        except ImportError:
            pass

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _euler_integrate(
        self,
        z0: torch.Tensor,
        t_span: torch.Tensor,
    ) -> torch.Tensor:
        """Integrate the vector field using fixed-step Euler method.

        Args:
            z0: Initial state [B, latent_dim].
            t_span: Time points to integrate to [T] (monotonically increasing).

        Returns:
            Trajectory states [B, T, latent_dim] at each time point.
        """
        B = z0.shape[0]
        T = t_span.shape[0]
        device = z0.device

        trajectory = [z0]
        z = z0

        for i in range(T - 1):
            t_start = t_span[i]
            t_end = t_span[i + 1]
            dt = (t_end - t_start) / self.n_euler_steps

            # Sub-steps between consecutive time points
            for step in range(self.n_euler_steps):
                t_current = t_start + step * dt
                dz = self.vector_field(t_current, z)
                z = z + dt * dz

            trajectory.append(z)

        return torch.stack(trajectory, dim=1)  # [B, T, latent_dim]

    def _adaptive_integrate(
        self,
        z0: torch.Tensor,
        t_span: torch.Tensor,
    ) -> torch.Tensor:
        """Integrate using torchdiffeq adaptive solver (if available).

        Args:
            z0: Initial state [B, latent_dim].
            t_span: Time points [T].

        Returns:
            Trajectory states [B, T, latent_dim].
        """
        import torchdiffeq

        # torchdiffeq.odeint returns [T, B, latent_dim]
        trajectory = torchdiffeq.odeint(
            self.vector_field,
            z0,
            t_span,
            method="dopri5",
            rtol=1e-4,
            atol=1e-4,
        )
        return trajectory.permute(1, 0, 2)  # [B, T, latent_dim]

    def encode_trajectory(
        self,
        clr_sequence: torch.Tensor,
        timestamps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode a temporal sequence of community profiles.

        Args:
            clr_sequence: CLR-transformed community profiles at each timepoint
                [B, T, input_dim].
            timestamps: Observation timestamps [B, T] or [T] (shared across batch).
                Should be normalized to [0, 1] range.

        Returns:
            Tuple of:
                - trajectory_embedding: Summary of the trajectory [B, latent_dim].
                - predicted_trajectory: ODE-predicted states [B, T, latent_dim].
                - observed_states: Encoded observed states [B, T, latent_dim].
        """
        B, T, D = clr_sequence.shape

        # Encode each observed timepoint to latent state
        observed_flat = clr_sequence.reshape(B * T, D)
        states_flat = self.state_encoder(observed_flat)
        observed_states = states_flat.view(B, T, self.latent_dim)

        # Normalize timestamps to [0, 1] if not already
        if timestamps.dim() == 1:
            t_span = timestamps
        else:
            t_span = timestamps[0]  # Assume shared timestamps across batch

        # Integrate from initial observed state
        z0 = observed_states[:, 0, :]  # [B, latent_dim]

        if self._has_torchdiffeq:
            predicted_trajectory = self._adaptive_integrate(z0, t_span)
        else:
            predicted_trajectory = self._euler_integrate(z0, t_span)

        # Aggregate trajectory into a single embedding via self-attention
        attended, _ = self.trajectory_attention(
            observed_states, observed_states, observed_states
        )
        trajectory_embedding = self.trajectory_norm(attended.mean(dim=1))

        return trajectory_embedding, predicted_trajectory, observed_states

    def compute_anomaly_score(
        self,
        predicted_trajectory: torch.Tensor,
        observed_states: torch.Tensor,
    ) -> torch.Tensor:
        """Compute anomaly score from trajectory deviation.

        Compares ODE-predicted trajectory with observed states. Large
        deviations indicate the community is not following healthy dynamics.

        Args:
            predicted_trajectory: ODE predictions [B, T, latent_dim].
            observed_states: Encoded observations [B, T, latent_dim].

        Returns:
            Anomaly scores [B], higher = more anomalous.
        """
        # Per-timepoint deviation in CLR space
        deviation = (predicted_trajectory - observed_states).pow(2).mean(dim=-1)  # [B, T]

        # Summary statistics of deviation trajectory
        mean_dev = deviation.mean(dim=1)  # [B]
        max_dev = deviation.max(dim=1).values  # [B]

        # Combine mean and max deviation for anomaly scoring
        combined = torch.cat([
            mean_dev.unsqueeze(-1),
            max_dev.unsqueeze(-1),
        ], dim=-1)  # [B, 2]

        # Pad to match anomaly_head input dimension
        # Use trajectory embedding statistics as additional context
        trajectory_stats = torch.cat([
            predicted_trajectory.mean(dim=1),  # [B, latent_dim]
            observed_states.mean(dim=1),  # [B, latent_dim]
        ], dim=-1)  # [B, latent_dim * 2]

        raw_score = self.anomaly_head(trajectory_stats).squeeze(-1)  # [B]

        # Add deviation magnitude as direct signal
        raw_score = raw_score + mean_dev

        # Normalize by running statistics
        normalized = (raw_score - self.anomaly_mean) / self.anomaly_std.clamp(min=1e-6)

        return normalized

    def forward(
        self,
        clr_sequence: torch.Tensor,
        timestamps: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Full forward pass: encode trajectory, predict dynamics, score anomaly.

        Args:
            clr_sequence: CLR-transformed community profiles [B, T, input_dim].
            timestamps: Observation timestamps [B, T] or [T].

        Returns:
            Dict with:
                - 'trajectory_embedding': Summary embedding [B, latent_dim].
                - 'anomaly_score': Trajectory anomaly score [B].
                - 'predicted_trajectory': ODE predictions [B, T, latent_dim].
                - 'observed_states': Encoded observations [B, T, latent_dim].
        """
        trajectory_embedding, predicted, observed = self.encode_trajectory(
            clr_sequence, timestamps
        )

        anomaly_score = self.compute_anomaly_score(predicted, observed)

        return {
            "trajectory_embedding": trajectory_embedding,
            "anomaly_score": anomaly_score,
            "predicted_trajectory": predicted,
            "observed_states": observed,
        }

    def forward_single_timepoint(
        self,
        clr_abundances: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Handle single-timepoint input (no temporal data available).

        When only a single community snapshot is available (no time series),
        encode it directly and produce a baseline anomaly score based on
        deviation from the learned latent manifold.

        Args:
            clr_abundances: CLR-transformed abundances [B, input_dim].

        Returns:
            Tuple of:
                - state_embedding: Encoded state [B, latent_dim].
                - anomaly_score: Anomaly score [B] (based on latent norm).
        """
        state = self.state_encoder(clr_abundances)  # [B, latent_dim]

        # For single timepoint, anomaly = distance from origin in latent space
        # (healthy states should cluster near the origin after training)
        latent_norm = state.pow(2).mean(dim=-1)  # [B]
        anomaly_score = (latent_norm - self.anomaly_mean) / self.anomaly_std.clamp(min=1e-6)

        return state, anomaly_score

    def update_anomaly_statistics(self, scores: torch.Tensor) -> None:
        """Update running statistics for anomaly score normalization.

        Args:
            scores: Raw anomaly scores from training data [N].
        """
        self.anomaly_mean.copy_(scores.mean())
        self.anomaly_std.copy_(scores.std().clamp(min=1e-6))
