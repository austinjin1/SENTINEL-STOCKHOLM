"""Digital Aquatic Ecosystem Twin -- core engine.

Implements a hybrid neural-ODE + transformer architecture for multi-horizon
freshwater ecosystem forecasting.  The engine couples a differentiable
biogeochemical ODE (Streeter-Phelps-extended) with a neural corrector that
learns residuals, using SENTINEL embeddings (256-d) for Bayesian data
assimilation of initial conditions and ODE parameter posteriors.

Architecture overview::

    SENTINEL embedding (256-d)
            |
      DataAssimilator
       /           \\
    state_0      ODE params (posterior)
       |              |
       +-----> BiogeochemicalODE -----> raw trajectory
                                            |
                                     NeuralCorrector  <-- embedding
                                            |
                                      corrected trajectory
                                            |
                                       ForecastHead
                                            |
                               [1, 7, 14, 30, 90, 365]-day
                                predictions + uncertainty

The ``CounterfactualEngine`` wraps the forward pipeline to evaluate
paired simulations (baseline vs. intervention) for "what-if" analysis.

ODE integration uses ``torchdiffeq.odeint`` when available and falls back
to a simple fixed-step Euler integrator otherwise, keeping the module
dependency-light for environments without ``torchdiffeq``.

All predictions include 90% confidence intervals estimated via MC-dropout
at inference time.
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try to import torchdiffeq; fall back to Euler integration if unavailable.
# ---------------------------------------------------------------------------

try:
    from torchdiffeq import odeint as _torchdiffeq_odeint

    _HAS_TORCHDIFFEQ = True
    logger.info("torchdiffeq available -- using adaptive ODE solver.")
except ImportError:
    _HAS_TORCHDIFFEQ = False
    logger.info("torchdiffeq not found -- falling back to Euler integration.")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STATE_VARS: Tuple[str, ...] = (
    "dissolved_oxygen",    # mg/L   -- reaeration, photosynthesis, respiration
    "bod",                 # mg/L   -- biochemical oxygen demand
    "total_nitrogen",      # mg/L   -- nitrification, denitrification, uptake
    "total_phosphorus",    # mg/L   -- sorption, uptake, release from sediment
    "chlorophyll_a",       # ug/L   -- algal growth/death, nutrient limitation
    "water_temperature",   # degC   -- heat balance
    "ph",                  # -      -- CO2 equilibrium
    "turbidity",           # NTU    -- settling, resuspension
    "doc",                 # mg/L   -- dissolved organic carbon
    "sediment",            # mg/L   -- erosion, settling
)

NUM_STATE_VARS: int = len(STATE_VARS)  # 10

#: Forecast horizons in days.
FORECAST_HORIZONS: Tuple[int, ...] = (1, 7, 14, 30, 90, 365)

#: Dimensionality of the SENTINEL shared embedding space.
SENTINEL_EMBEDDING_DIM: int = 256


# ---------------------------------------------------------------------------
# Utility: Euler integrator (fallback when torchdiffeq is absent)
# ---------------------------------------------------------------------------

def _euler_integrate(
    func: nn.Module,
    y0: torch.Tensor,
    t: torch.Tensor,
    *,
    steps_per_unit: int = 20,
) -> torch.Tensor:
    """Fixed-step Euler integration of an ODE.

    Args:
        func: Callable ``f(t, y) -> dy/dt`` where ``y`` has shape ``[B, D]``.
        y0: Initial state ``[B, D]``.
        t: 1-D tensor of evaluation times (must be sorted ascending).
        steps_per_unit: Number of Euler steps per unit time interval.

    Returns:
        Trajectory tensor of shape ``[T, B, D]`` evaluated at each time in
        ``t`` (including ``t[0]``).
    """
    trajectory = [y0]
    y = y0
    for i in range(len(t) - 1):
        t_start = t[i]
        t_end = t[i + 1]
        dt_total = (t_end - t_start).item()
        n_steps = max(1, int(math.ceil(abs(dt_total) * steps_per_unit)))
        dt = dt_total / n_steps
        t_cur = t_start.item()
        for _ in range(n_steps):
            t_tensor = torch.tensor(t_cur, dtype=y.dtype, device=y.device)
            dy = func(t_tensor, y)
            y = y + dt * dy
            t_cur += dt
        trajectory.append(y)
    return torch.stack(trajectory, dim=0)  # [T, B, D]


def odeint(
    func: nn.Module,
    y0: torch.Tensor,
    t: torch.Tensor,
    method: str = "dopri5",
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> torch.Tensor:
    """Unified ODE integration interface.

    Delegates to ``torchdiffeq.odeint`` when available, otherwise falls
    back to :func:`_euler_integrate`.

    Args:
        func: ODE right-hand side ``f(t, y)``.
        y0: Initial condition ``[B, D]``.
        t: Evaluation times ``[T]``.
        method: Solver method (only used with torchdiffeq).
        rtol: Relative tolerance (only used with torchdiffeq).
        atol: Absolute tolerance (only used with torchdiffeq).

    Returns:
        Trajectory ``[T, B, D]``.
    """
    if _HAS_TORCHDIFFEQ:
        return _torchdiffeq_odeint(func, y0, t, method=method, rtol=rtol, atol=atol)
    return _euler_integrate(func, y0, t)


# =========================================================================== #
#  BiogeochemicalODE                                                          #
# =========================================================================== #


class BiogeochemicalODE(nn.Module):
    """Differentiable biogeochemical ODE for coupled water quality dynamics.

    Extends the classical Streeter-Phelps DO-BOD model to a 10-variable
    system covering oxygen, nutrients, algae, temperature, pH, turbidity,
    dissolved organic carbon, and suspended sediment.

    All process rates and half-saturation constants are **learnable
    parameters** initialized from literature values so the model can be
    trained end-to-end while staying in a physically reasonable regime.

    State vector order matches :data:`STATE_VARS`.

    Forward signature follows ``torchdiffeq`` convention:
    ``f(t, y) -> dy/dt``.

    Literature defaults
    -------------------
    * Streeter & Phelps (1925) -- DO-BOD coupling.
    * Chapra (2008) *Surface Water-Quality Modeling* -- nutrient kinetics.
    * Bowie et al. (1985) EPA/600/3-85/040 -- algal growth parameters.
    """

    def __init__(self) -> None:
        super().__init__()

        # --- Oxygen / BOD (Streeter-Phelps extended) -----------------------
        # k_d: BOD deoxygenation rate [1/day]
        self.log_k_d = nn.Parameter(torch.tensor(math.log(0.23)))
        # k_a: reaeration rate [1/day]
        self.log_k_a = nn.Parameter(torch.tensor(math.log(0.40)))
        # photosynthetic O2 production per unit chl-a [mg O2 / (ug chl-a * day)]
        self.log_photo_rate = nn.Parameter(torch.tensor(math.log(0.10)))
        # community respiration rate [mg O2 / (L * day)]
        self.log_resp_rate = nn.Parameter(torch.tensor(math.log(0.05)))

        # --- Nutrients -----------------------------------------------------
        # nitrification rate [1/day]
        self.log_nitrif = nn.Parameter(torch.tensor(math.log(0.10)))
        # denitrification rate [1/day]
        self.log_denitrif = nn.Parameter(torch.tensor(math.log(0.05)))
        # nitrogen half-saturation for algal uptake [mg/L]
        self.log_k_n = nn.Parameter(torch.tensor(math.log(0.03)))
        # phosphorus half-saturation for algal uptake [mg/L]
        self.log_k_p = nn.Parameter(torch.tensor(math.log(0.005)))
        # phosphorus sorption rate [1/day]
        self.log_p_sorption = nn.Parameter(torch.tensor(math.log(0.02)))
        # sediment phosphorus release rate [mg/(L*day)]
        self.log_p_release = nn.Parameter(torch.tensor(math.log(0.005)))

        # --- Algae (chlorophyll-a proxy) -----------------------------------
        # max specific growth rate [1/day]
        self.log_mu_max = nn.Parameter(torch.tensor(math.log(1.5)))
        # algal death/respiration rate [1/day]
        self.log_algal_death = nn.Parameter(torch.tensor(math.log(0.10)))
        # optimal temperature for growth [degC]
        self.t_opt = nn.Parameter(torch.tensor(22.0))
        # temperature width parameter [degC]
        self.log_t_width = nn.Parameter(torch.tensor(math.log(8.0)))

        # --- Temperature ---------------------------------------------------
        # heat exchange coefficient [1/day] (Newton cooling toward equilibrium)
        self.log_heat_exchange = nn.Parameter(torch.tensor(math.log(0.15)))
        # equilibrium temperature [degC] (latent; can be overridden per site)
        self.t_eq = nn.Parameter(torch.tensor(18.0))

        # --- pH / carbonate ------------------------------------------------
        # CO2 invasion rate [1/day]
        self.log_co2_exchange = nn.Parameter(torch.tensor(math.log(0.10)))
        # equilibrium pH
        self.ph_eq = nn.Parameter(torch.tensor(7.5))

        # --- Turbidity / sediment ------------------------------------------
        # particle settling velocity [1/day]
        self.log_settling = nn.Parameter(torch.tensor(math.log(0.30)))
        # resuspension rate [1/day]
        self.log_resuspension = nn.Parameter(torch.tensor(math.log(0.02)))
        # erosion input rate [mg/(L*day)]
        self.log_erosion = nn.Parameter(torch.tensor(math.log(0.01)))

        # --- DOC -----------------------------------------------------------
        # DOC decomposition rate [1/day]
        self.log_doc_decay = nn.Parameter(torch.tensor(math.log(0.03)))
        # DOC leaching input [mg/(L*day)]
        self.log_doc_input = nn.Parameter(torch.tensor(math.log(0.01)))

    # -- helpers to enforce positivity via exp -----------------------------
    def _pos(self, log_param: nn.Parameter) -> torch.Tensor:
        """Return exp(log_param) to guarantee positivity."""
        return log_param.exp()

    def _do_saturation(self, temp: torch.Tensor) -> torch.Tensor:
        """Temperature-dependent DO saturation (Benson & Krause 1984).

        Args:
            temp: Water temperature [degC], any shape.

        Returns:
            Maximum dissolved oxygen [mg/L], same shape.
        """
        return 14.62 - 0.3898 * temp + 0.006969 * temp ** 2 - 5.897e-5 * temp ** 3

    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute dy/dt for the coupled biogeochemical system.

        Args:
            t: Current time (scalar tensor, in days).  Not explicitly used
               in autonomous dynamics but required by the ODE interface.
            y: State vector ``[B, 10]`` ordered as :data:`STATE_VARS`.

        Returns:
            Time derivatives ``[B, 10]``, same shape as ``y``.
        """
        # Unpack state -- clamp to physically plausible ranges to prevent
        # numerical blow-up during early training.
        do   = y[..., 0].clamp(min=0.0)           # dissolved oxygen
        bod  = y[..., 1].clamp(min=0.0)           # BOD
        tn   = y[..., 2].clamp(min=0.0)           # total nitrogen
        tp   = y[..., 3].clamp(min=0.0)           # total phosphorus
        chla = y[..., 4].clamp(min=0.0)           # chlorophyll-a
        temp = y[..., 5].clamp(min=-2.0, max=45.0)  # temperature
        ph   = y[..., 6].clamp(min=2.0, max=14.0)   # pH
        turb = y[..., 7].clamp(min=0.0)           # turbidity
        doc  = y[..., 8].clamp(min=0.0)           # DOC
        sed  = y[..., 9].clamp(min=0.0)           # sediment

        # --- Retrieve positive parameters --------------------------------
        k_d   = self._pos(self.log_k_d)
        k_a   = self._pos(self.log_k_a)
        photo = self._pos(self.log_photo_rate)
        resp  = self._pos(self.log_resp_rate)

        nitrif     = self._pos(self.log_nitrif)
        denitrif   = self._pos(self.log_denitrif)
        k_n        = self._pos(self.log_k_n)
        k_p        = self._pos(self.log_k_p)
        p_sorption = self._pos(self.log_p_sorption)
        p_release  = self._pos(self.log_p_release)

        mu_max     = self._pos(self.log_mu_max)
        algal_death = self._pos(self.log_algal_death)
        t_width    = self._pos(self.log_t_width)

        heat_ex    = self._pos(self.log_heat_exchange)
        co2_ex     = self._pos(self.log_co2_exchange)

        settling     = self._pos(self.log_settling)
        resuspension = self._pos(self.log_resuspension)
        erosion      = self._pos(self.log_erosion)

        doc_decay = self._pos(self.log_doc_decay)
        doc_input = self._pos(self.log_doc_input)

        # --- Derived quantities ------------------------------------------
        do_sat = self._do_saturation(temp)

        # Nutrient limitation (Monod kinetics)
        n_lim = tn / (tn + k_n)  # nitrogen limitation [0, 1]
        p_lim = tp / (tp + k_p)  # phosphorus limitation [0, 1]

        # Temperature limitation for algal growth (Gaussian envelope)
        temp_lim = torch.exp(-((temp - self.t_opt) ** 2) / (2.0 * t_width ** 2))

        # Algal specific growth rate
        mu = mu_max * n_lim * p_lim * temp_lim

        # --- State derivatives -------------------------------------------
        # 0. Dissolved oxygen (Streeter-Phelps + photosynthesis - respiration)
        d_do = (
            k_a * (do_sat - do)       # reaeration
            - k_d * bod               # BOD deoxygenation
            + photo * chla            # algal photosynthesis
            - resp                    # community respiration
            - 4.57 * nitrif * tn      # oxygen demand from nitrification
        )

        # 1. BOD
        d_bod = (
            -k_d * bod                # decay
            + 0.5 * algal_death * chla  # dead algae -> BOD
            + 0.3 * doc_decay * doc   # DOC decomposition adds BOD
        )

        # 2. Total nitrogen
        # Yield coefficient: algal N content ~ 0.07 mg N / ug chl-a
        d_tn = (
            -nitrif * tn              # nitrification loss
            - denitrif * tn           # denitrification loss
            - 0.07 * mu * chla        # algal uptake
            + 0.07 * algal_death * chla  # recycling from dead algae
        )

        # 3. Total phosphorus
        # Yield coefficient: algal P content ~ 0.01 mg P / ug chl-a
        d_tp = (
            -p_sorption * tp          # adsorption to particles
            - 0.01 * mu * chla        # algal uptake
            + 0.01 * algal_death * chla  # recycling
            + p_release               # release from sediments
        )

        # 4. Chlorophyll-a (algal biomass proxy)
        d_chla = (
            mu * chla                 # growth
            - algal_death * chla      # death / respiration
            - settling * 0.1 * chla   # sinking loss (fraction settles)
        )

        # 5. Water temperature (Newton's law of cooling)
        d_temp = heat_ex * (self.t_eq - temp)

        # 6. pH (relaxation toward CO2 equilibrium)
        # Simplified: photosynthesis raises pH, respiration lowers it.
        d_ph = (
            co2_ex * (self.ph_eq - ph)
            + 0.002 * photo * chla    # photosynthesis raises pH
            - 0.001 * resp            # respiration lowers pH
        )

        # 7. Turbidity
        d_turb = (
            -settling * turb          # settling clears turbidity
            + resuspension * sed      # resuspension adds turbidity
        )

        # 8. DOC
        d_doc = (
            -doc_decay * doc          # microbial decomposition
            + doc_input               # allochthonous input
            + 0.2 * algal_death * chla  # exudation from dead algae
        )

        # 9. Suspended sediment
        d_sed = (
            erosion                   # erosion input
            - settling * sed          # settling
            + resuspension * sed * 0.1  # partial resuspension feedback
        )

        return torch.stack(
            [d_do, d_bod, d_tn, d_tp, d_chla, d_temp, d_ph, d_turb, d_doc, d_sed],
            dim=-1,
        )


# =========================================================================== #
#  NeuralCorrector                                                            #
# =========================================================================== #


class NeuralCorrector(nn.Module):
    """Learns additive residuals between the physics ODE and reality.

    The corrector receives the raw physics-predicted trajectory concatenated
    with the SENTINEL embedding and outputs a correction term of the same
    shape as the state vector.  By construction the correction is
    *additive*, so the hybrid model is always at least as expressive as
    the pure physics model.

    MC-dropout is used at inference time to estimate prediction uncertainty.

    Architecture::

        [physics_state (10) ; embedding (256)] -> MLP -> correction (10)

    Args:
        state_dim: Dimensionality of the state vector.
        embedding_dim: Dimensionality of the SENTINEL embedding.
        hidden_dim: Width of hidden layers.
        num_layers: Number of hidden layers.
        dropout: Dropout probability (also used at inference for MC-dropout).
    """

    def __init__(
        self,
        state_dim: int = NUM_STATE_VARS,
        embedding_dim: int = SENTINEL_EMBEDDING_DIM,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        in_dim = state_dim + embedding_dim
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else state_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.LayerNorm(out_dim))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
            in_dim = out_dim

        self.mlp = nn.Sequential(*layers)

        # Small initial scale so corrections start near zero.
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize final layer near zero so corrections start small."""
        final_linear = self.mlp[-1]
        if isinstance(final_linear, nn.Linear):
            nn.init.zeros_(final_linear.bias)
            nn.init.normal_(final_linear.weight, std=1e-3)

    def forward(
        self,
        physics_state: torch.Tensor,
        embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Predict additive correction to the physics-model output.

        Args:
            physics_state: Physics ODE prediction ``[B, D]`` or
                ``[T, B, D]``.
            embedding: SENTINEL embedding ``[B, E]``.  Will be broadcast
                along the time dimension when ``physics_state`` is 3-D.

        Returns:
            Correction tensor, same shape as ``physics_state``.
        """
        if physics_state.dim() == 3:
            # [T, B, D] -- expand embedding to match time dimension.
            T = physics_state.size(0)
            emb = embedding.unsqueeze(0).expand(T, -1, -1)  # [T, B, E]
        else:
            emb = embedding  # [B, E]

        x = torch.cat([physics_state, emb], dim=-1)
        return self.mlp(x)


# =========================================================================== #
#  DataAssimilator                                                            #
# =========================================================================== #


class DataAssimilator(nn.Module):
    """Bayesian data assimilation from SENTINEL embedding.

    Maps the 256-d SENTINEL fused embedding to:

    1. **Initial state vector** (10-d mean + 10-d log-variance) encoding
       a Gaussian belief about the current ecosystem state.
    2. **ODE parameter posteriors** -- affine modulation of each learnable
       ODE parameter, allowing the embedding to condition the physics.

    The state mean and variance are produced by separate linear heads so
    the network can cleanly separate location from uncertainty.

    Args:
        embedding_dim: Dimensionality of the SENTINEL embedding.
        state_dim: Dimensionality of the state vector.
        num_ode_params: Number of ODE parameters to modulate.
        hidden_dim: Width of shared hidden layer.
    """

    # Number of learnable parameters in BiogeochemicalODE.
    _DEFAULT_NUM_ODE_PARAMS: int = 24  # counted from __init__

    def __init__(
        self,
        embedding_dim: int = SENTINEL_EMBEDDING_DIM,
        state_dim: int = NUM_STATE_VARS,
        num_ode_params: int = _DEFAULT_NUM_ODE_PARAMS,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.num_ode_params = num_ode_params

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # State posterior heads
        self.state_mean = nn.Linear(hidden_dim, state_dim)
        self.state_log_var = nn.Linear(hidden_dim, state_dim)

        # ODE parameter modulation: predicts (scale, shift) for each param.
        # scale is passed through sigmoid and centered at 1, shift is small.
        self.param_modulation = nn.Linear(hidden_dim, num_ode_params * 2)

        self._init_weights()

    def _init_weights(self) -> None:
        """Start with near-identity modulation and reasonable state priors."""
        nn.init.zeros_(self.param_modulation.bias)
        nn.init.normal_(self.param_modulation.weight, std=1e-3)
        # Log-variance initialized to small variance (log(0.01) ~ -4.6)
        nn.init.constant_(self.state_log_var.bias, -4.6)

    def forward(
        self,
        embedding: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Assimilate SENTINEL embedding into ODE initial conditions.

        Args:
            embedding: SENTINEL fused embedding ``[B, 256]``.

        Returns:
            Tuple of:
                state_mean: Posterior mean of state vector ``[B, 10]``.
                state_log_var: Log-variance of state vector ``[B, 10]``.
                param_scale: Multiplicative modulation ``[B, P]``, centered
                    near 1.0.  Apply as ``ode_param * param_scale``.
                param_shift: Additive modulation ``[B, P]``, near 0.
        """
        h = self.encoder(embedding)  # [B, H]

        mean = self.state_mean(h)          # [B, 10]
        log_var = self.state_log_var(h)    # [B, 10]

        # Parameter modulation: split into scale and shift.
        mod = self.param_modulation(h)     # [B, 2*P]
        raw_scale = mod[..., : self.num_ode_params]
        raw_shift = mod[..., self.num_ode_params :]

        # Scale centered at 1 via sigmoid * 2 (range [0, 2]).
        param_scale = torch.sigmoid(raw_scale) * 2.0
        # Shift stays small via tanh * 0.1.
        param_shift = torch.tanh(raw_shift) * 0.1

        return mean, log_var, param_scale, param_shift

    def sample_state(
        self,
        mean: torch.Tensor,
        log_var: torch.Tensor,
        *,
        n_samples: int = 1,
    ) -> torch.Tensor:
        """Draw reparameterized samples from the state posterior.

        Args:
            mean: State posterior mean ``[B, D]``.
            log_var: State posterior log-variance ``[B, D]``.
            n_samples: Number of samples to draw.

        Returns:
            Samples ``[n_samples, B, D]`` (or ``[B, D]`` if n_samples=1).
        """
        std = (0.5 * log_var).exp()
        if n_samples == 1:
            eps = torch.randn_like(mean)
            return mean + eps * std
        eps = torch.randn(n_samples, *mean.shape, device=mean.device, dtype=mean.dtype)
        return mean.unsqueeze(0) + eps * std.unsqueeze(0)


# =========================================================================== #
#  ForecastHead                                                               #
# =========================================================================== #


class ForecastHead(nn.Module):
    """Multi-horizon prediction head with per-variable uncertainty.

    Given a continuous ODE trajectory, extracts predictions at the
    canonical forecast horizons and estimates per-variable 90% confidence
    intervals via a lightweight uncertainty MLP.

    The head also supports MC-dropout uncertainty (controlled externally by
    keeping dropout active during inference).

    Args:
        state_dim: Dimensionality of the state vector.
        embedding_dim: Dimensionality of the SENTINEL embedding.
        hidden_dim: Hidden dimension of the uncertainty estimator.
        horizons: Forecast horizons in days.
        dropout: Dropout probability for MC-dropout uncertainty.
    """

    def __init__(
        self,
        state_dim: int = NUM_STATE_VARS,
        embedding_dim: int = SENTINEL_EMBEDDING_DIM,
        hidden_dim: int = 128,
        horizons: Tuple[int, ...] = FORECAST_HORIZONS,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.horizons = horizons
        self.num_horizons = len(horizons)

        # Per-horizon uncertainty estimator: predicts log-scale for each
        # variable at each horizon.
        self.uncertainty_net = nn.Sequential(
            nn.Linear(state_dim + embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, state_dim),  # log-scale per variable
        )

        # Horizon embedding (so the network can distinguish horizons).
        self.horizon_embedding = nn.Embedding(self.num_horizons, embedding_dim)

    def forward(
        self,
        trajectory: torch.Tensor,
        horizon_indices: torch.Tensor,
        embedding: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract predictions and uncertainty at requested horizons.

        Args:
            trajectory: State trajectory ``[T, B, D]`` from the ODE.
                ``T`` should correspond to evaluation times including all
                requested horizons.
            horizon_indices: Long tensor ``[H]`` indexing into the time
                dimension of ``trajectory`` for each desired horizon.
            embedding: SENTINEL embedding ``[B, E]``.

        Returns:
            Tuple of:
                predictions: ``[H, B, D]`` predicted state at each horizon.
                lower_90: ``[H, B, D]`` lower bound of 90% CI.
                upper_90: ``[H, B, D]`` upper bound of 90% CI.
        """
        # Gather predictions at horizon time-points.
        preds = trajectory[horizon_indices]  # [H, B, D]

        H, B, D = preds.shape

        # Uncertainty estimation per horizon.
        emb_expanded = embedding.unsqueeze(0).expand(H, -1, -1)  # [H, B, E]

        # Add horizon embedding.
        h_emb = self.horizon_embedding(
            torch.arange(H, device=embedding.device)
        )  # [H, E]
        h_emb = h_emb.unsqueeze(1).expand(-1, B, -1)  # [H, B, E]
        emb_with_horizon = emb_expanded + h_emb  # [H, B, E]

        unc_input = torch.cat([preds, emb_with_horizon], dim=-1)  # [H, B, D+E]
        log_scale = self.uncertainty_net(unc_input)  # [H, B, D]
        scale = F.softplus(log_scale) + 1e-4  # ensure positive

        # 90% CI: z_{0.95} ~ 1.645
        z = 1.6449
        lower_90 = preds - z * scale
        upper_90 = preds + z * scale

        return preds, lower_90, upper_90


# =========================================================================== #
#  CounterfactualEngine                                                       #
# =========================================================================== #


class CounterfactualEngine:
    """Evaluates counterfactual "what-if" scenarios.

    Given an intervention specification (parameter name -> multiplier),
    runs paired forward simulations -- one with and one without the
    intervention -- and returns the difference.

    This enables questions like:

    * *"What if nitrogen loading were reduced by 30%?"*
    * *"What if the reaeration rate doubled due to a new weir?"*

    The engine does **not** own the ODE or corrector; it wraps a
    :class:`DigitalTwinEngine` reference and modifies ODE parameters
    in-place (restoring them afterward).

    Note: This is intentionally *not* an ``nn.Module`` because it holds
    no learnable parameters and storing a back-reference to the parent
    ``DigitalTwinEngine`` would create a circular reference that breaks
    ``repr()``.

    Args:
        engine: Reference to the parent :class:`DigitalTwinEngine`.
    """

    def __init__(self, engine: "DigitalTwinEngine") -> None:
        self._engine = engine

    @torch.no_grad()
    def evaluate(
        self,
        embedding: torch.Tensor,
        intervention: Dict[str, float],
        horizons: Optional[Tuple[int, ...]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run a counterfactual analysis.

        Args:
            embedding: SENTINEL embedding ``[B, 256]``.
            intervention: Mapping from ODE parameter name (e.g.
                ``"log_k_d"``, ``"log_nitrif"``) to a **multiplier**
                applied to the *positive* parameter.  For example,
                ``{"log_nitrif": 0.7}`` means "reduce nitrification rate
                to 70% of its current value".  Internally the log-parameter
                is shifted by ``log(multiplier)``.
            horizons: Optional override of forecast horizons.

        Returns:
            Dictionary with keys:
                ``"baseline"``: ``[H, B, D]`` baseline predictions.
                ``"counterfactual"``: ``[H, B, D]`` intervention predictions.
                ``"delta"``: ``[H, B, D]`` counterfactual - baseline.
                ``"horizons"``: tuple of horizon days used.
        """
        horizons = horizons or FORECAST_HORIZONS

        # --- Baseline run --------------------------------------------------
        baseline_out = self._engine(embedding, horizons=horizons)
        baseline_preds = baseline_out.predictions  # [H, B, D]

        # --- Apply intervention --------------------------------------------
        ode = self._engine.ode
        saved: Dict[str, torch.Tensor] = {}
        for param_name, multiplier in intervention.items():
            if not hasattr(ode, param_name):
                warnings.warn(
                    f"BiogeochemicalODE has no parameter '{param_name}'; skipping."
                )
                continue
            param = getattr(ode, param_name)
            saved[param_name] = param.data.clone()
            # For log-parameters, adding log(multiplier) is equivalent to
            # multiplying the positive parameter.
            if param_name.startswith("log_"):
                param.data.add_(math.log(max(multiplier, 1e-8)))
            else:
                param.data.mul_(multiplier)

        # --- Counterfactual run -------------------------------------------
        cf_out = self._engine(embedding, horizons=horizons)
        cf_preds = cf_out.predictions

        # --- Restore parameters -------------------------------------------
        for param_name, original_data in saved.items():
            getattr(ode, param_name).data.copy_(original_data)

        return {
            "baseline": baseline_preds,
            "counterfactual": cf_preds,
            "delta": cf_preds - baseline_preds,
            "horizons": horizons,
        }


# =========================================================================== #
#  TwinOutput                                                                 #
# =========================================================================== #


@dataclass
class TwinOutput:
    """Container for DigitalTwinEngine outputs.

    Attributes:
        predictions: Predicted ecosystem state at each forecast horizon,
            shape ``[H, B, D]``.
        lower_90: Lower bound of 90% confidence interval ``[H, B, D]``.
        upper_90: Upper bound of 90% confidence interval ``[H, B, D]``.
        trajectory: Full ODE trajectory ``[T, B, D]`` at all integration
            time-points (including horizons).
        state_mean: Assimilated state posterior mean ``[B, D]``.
        state_log_var: Assimilated state posterior log-variance ``[B, D]``.
        physics_trajectory: Raw physics trajectory before neural
            correction ``[T, B, D]``.
        corrections: Neural corrector output ``[T, B, D]``.
    """

    predictions: torch.Tensor
    lower_90: torch.Tensor
    upper_90: torch.Tensor
    trajectory: torch.Tensor
    state_mean: torch.Tensor
    state_log_var: torch.Tensor
    physics_trajectory: torch.Tensor
    corrections: torch.Tensor


# =========================================================================== #
#  DigitalTwinEngine                                                          #
# =========================================================================== #


class DigitalTwinEngine(nn.Module):
    """Main orchestrator for the Digital Aquatic Ecosystem Twin.

    Integrates:

    * :class:`DataAssimilator` -- maps SENTINEL embedding to initial
      conditions and ODE parameter posteriors.
    * :class:`BiogeochemicalODE` -- physics-informed state evolution.
    * :class:`NeuralCorrector` -- learns additive residuals.
    * :class:`ForecastHead` -- multi-horizon predictions + uncertainty.
    * :class:`CounterfactualEngine` -- "what-if" scenario analysis.

    Forward pass:

    1. Assimilate embedding -> state_0, parameter modulation.
    2. Integrate ODE from t=0 to t=max(horizons) days.
    3. Apply neural correction to the full trajectory.
    4. Extract predictions at requested horizons + uncertainty.

    MC-dropout uncertainty is obtained by calling :meth:`predict_with_uncertainty`
    which runs multiple stochastic forward passes.

    Args:
        embedding_dim: Dimensionality of the SENTINEL embedding.
        hidden_dim: Hidden dimension for sub-networks.
        corrector_layers: Number of hidden layers in the NeuralCorrector.
        dropout: Dropout probability (used for both regularization and
            MC-dropout uncertainty).
        ode_method: ODE solver method (passed to torchdiffeq).
        ode_rtol: Relative tolerance for the ODE solver.
        ode_atol: Absolute tolerance for the ODE solver.
        euler_steps_per_day: Euler steps per day (only used when torchdiffeq
            is unavailable).
    """

    def __init__(
        self,
        embedding_dim: int = SENTINEL_EMBEDDING_DIM,
        hidden_dim: int = 256,
        corrector_layers: int = 3,
        dropout: float = 0.10,
        ode_method: str = "dopri5",
        ode_rtol: float = 1e-5,
        ode_atol: float = 1e-6,
        euler_steps_per_day: int = 20,
    ) -> None:
        super().__init__()

        self.ode_method = ode_method
        self.ode_rtol = ode_rtol
        self.ode_atol = ode_atol
        self.euler_steps_per_day = euler_steps_per_day

        # Sub-modules
        self.assimilator = DataAssimilator(
            embedding_dim=embedding_dim,
            state_dim=NUM_STATE_VARS,
            hidden_dim=hidden_dim,
        )
        self.ode = BiogeochemicalODE()
        self.corrector = NeuralCorrector(
            state_dim=NUM_STATE_VARS,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=corrector_layers,
            dropout=dropout,
        )
        self.forecast_head = ForecastHead(
            state_dim=NUM_STATE_VARS,
            embedding_dim=embedding_dim,
            dropout=dropout,
        )
        self.counterfactual = CounterfactualEngine(self)

    def _build_eval_times(
        self,
        horizons: Tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build ODE evaluation time grid and horizon index map.

        Creates a time tensor starting at 0 with the requested horizon
        days as evaluation points.

        Args:
            horizons: Forecast horizons in days.
            device: Target device.
            dtype: Target dtype.

        Returns:
            Tuple of (eval_times ``[T]``, horizon_indices ``[H]``).
        """
        # Always start at t=0 (initial condition).
        times = sorted(set([0] + list(horizons)))
        t = torch.tensor(times, device=device, dtype=dtype)

        # Map each horizon to its index in the time tensor.
        time_to_idx = {tv: i for i, tv in enumerate(times)}
        horizon_indices = torch.tensor(
            [time_to_idx[h] for h in horizons],
            device=device,
            dtype=torch.long,
        )
        return t, horizon_indices

    def _apply_param_modulation(
        self,
        param_scale: torch.Tensor,
        param_shift: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Apply data-assimilated parameter modulation to the ODE.

        For batch size > 1 we take the batch mean of scale/shift
        (the ODE parameters are shared across the batch).

        Args:
            param_scale: ``[B, P]`` multiplicative modulation.
            param_shift: ``[B, P]`` additive modulation.

        Returns:
            Dictionary mapping parameter names to their original data
            (for restoration after the forward pass).
        """
        saved: Dict[str, torch.Tensor] = {}
        # Collect all learnable ODE parameters in definition order.
        ode_params: list[Tuple[str, nn.Parameter]] = [
            (name, param)
            for name, param in self.ode.named_parameters()
        ]

        # Batch mean modulation (ODE params are not batched).
        scale = param_scale.mean(dim=0)  # [P]
        shift = param_shift.mean(dim=0)  # [P]

        for i, (name, param) in enumerate(ode_params):
            if i >= param_scale.size(-1):
                break
            saved[name] = param.data.clone()
            param.data = param.data * scale[i] + shift[i]

        return saved

    def _restore_ode_params(self, saved: Dict[str, torch.Tensor]) -> None:
        """Restore ODE parameters from saved values."""
        for name, data in saved.items():
            param = dict(self.ode.named_parameters())[name]
            param.data.copy_(data)

    def forward(
        self,
        embedding: torch.Tensor,
        horizons: Tuple[int, ...] = FORECAST_HORIZONS,
        state_override: Optional[torch.Tensor] = None,
    ) -> TwinOutput:
        """Run the Digital Twin forward pass.

        Args:
            embedding: SENTINEL fused embedding ``[B, 256]``.
            horizons: Forecast horizons in days.
            state_override: Optional ``[B, 10]`` initial state to use
                instead of the assimilated state (e.g., for known ground
                truth).

        Returns:
            :class:`TwinOutput` containing predictions, uncertainty, and
            intermediate quantities.
        """
        B = embedding.size(0)
        device = embedding.device
        dtype = embedding.dtype

        # --- 1. Data assimilation -----------------------------------------
        state_mean, state_log_var, param_scale, param_shift = self.assimilator(
            embedding
        )

        # Use assimilated or overridden initial state.
        if state_override is not None:
            y0 = state_override
        else:
            # During training: sample from posterior (reparameterization trick).
            # During eval: use the mean.
            if self.training:
                y0 = self.assimilator.sample_state(state_mean, state_log_var)
            else:
                y0 = state_mean

        # --- 2. Apply parameter modulation --------------------------------
        saved_params = self._apply_param_modulation(param_scale, param_shift)

        # --- 3. Integrate ODE ---------------------------------------------
        t_eval, horizon_idx = self._build_eval_times(horizons, device, dtype)

        try:
            physics_traj = odeint(
                self.ode,
                y0,
                t_eval,
                method=self.ode_method,
                rtol=self.ode_rtol,
                atol=self.ode_atol,
            )  # [T, B, D]
        finally:
            # Always restore ODE parameters, even if integration fails.
            self._restore_ode_params(saved_params)

        # --- 4. Neural correction -----------------------------------------
        corrections = self.corrector(physics_traj, embedding)  # [T, B, D]
        corrected_traj = physics_traj + corrections

        # --- 5. Forecast head ---------------------------------------------
        predictions, lower_90, upper_90 = self.forecast_head(
            corrected_traj, horizon_idx, embedding
        )

        return TwinOutput(
            predictions=predictions,
            lower_90=lower_90,
            upper_90=upper_90,
            trajectory=corrected_traj,
            state_mean=state_mean,
            state_log_var=state_log_var,
            physics_trajectory=physics_traj,
            corrections=corrections,
        )

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        embedding: torch.Tensor,
        horizons: Tuple[int, ...] = FORECAST_HORIZONS,
        n_samples: int = 50,
        state_override: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Monte Carlo dropout uncertainty estimation.

        Runs ``n_samples`` stochastic forward passes with dropout active
        and aggregates predictions to produce calibrated uncertainty.

        Args:
            embedding: SENTINEL embedding ``[B, 256]``.
            horizons: Forecast horizons in days.
            n_samples: Number of MC-dropout forward passes.
            state_override: Optional initial state override.

        Returns:
            Dictionary with keys:
                ``"mean"``: ``[H, B, D]`` mean prediction.
                ``"std"``: ``[H, B, D]`` standard deviation.
                ``"lower_90"``: ``[H, B, D]`` 5th percentile.
                ``"upper_90"``: ``[H, B, D]`` 95th percentile.
                ``"median"``: ``[H, B, D]`` median prediction.
                ``"samples"``: ``[S, H, B, D]`` all MC samples.
        """
        # Enable dropout during inference for MC estimation.
        was_training = self.training
        self.train()

        samples = []
        for _ in range(n_samples):
            out = self.forward(
                embedding, horizons=horizons, state_override=state_override
            )
            samples.append(out.predictions)

        if not was_training:
            self.eval()

        # Stack samples: [S, H, B, D]
        samples_tensor = torch.stack(samples, dim=0)

        return {
            "mean": samples_tensor.mean(dim=0),
            "std": samples_tensor.std(dim=0),
            "lower_90": samples_tensor.quantile(0.05, dim=0),
            "upper_90": samples_tensor.quantile(0.95, dim=0),
            "median": samples_tensor.median(dim=0).values,
            "samples": samples_tensor,
        }

    def physics_only_forward(
        self,
        embedding: torch.Tensor,
        horizons: Tuple[int, ...] = FORECAST_HORIZONS,
        state_override: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run only the physics ODE without neural correction.

        Useful as a baseline to quantify how much the neural corrector
        contributes.

        Args:
            embedding: SENTINEL embedding ``[B, 256]``.
            horizons: Forecast horizons in days.
            state_override: Optional initial state override.

        Returns:
            Physics-only trajectory ``[T, B, D]``.
        """
        state_mean, _, param_scale, param_shift = self.assimilator(embedding)
        y0 = state_override if state_override is not None else state_mean

        saved_params = self._apply_param_modulation(param_scale, param_shift)
        t_eval, _ = self._build_eval_times(
            horizons, embedding.device, embedding.dtype
        )
        try:
            traj = odeint(
                self.ode, y0, t_eval,
                method=self.ode_method,
                rtol=self.ode_rtol,
                atol=self.ode_atol,
            )
        finally:
            self._restore_ode_params(saved_params)
        return traj

    def extra_repr(self) -> str:
        return (
            f"state_vars={NUM_STATE_VARS}, "
            f"embedding_dim={SENTINEL_EMBEDDING_DIM}, "
            f"horizons={FORECAST_HORIZONS}, "
            f"ode_method='{self.ode_method}', "
            f"has_torchdiffeq={_HAS_TORCHDIFFEQ}"
        )
