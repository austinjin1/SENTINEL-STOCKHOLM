"""Climate coupling for SENTINEL Digital Twin — Phase 4.2.

Ingests NOAA CFSv2 and ERA5 forecast fields as exogenous drivers for
multi-horizon ecosystem forecasting.  Enables climate-amplified
contamination forecasts on multi-decade horizons.

Climate drivers:
  - Precipitation (mm/day)
  - Air temperature (°C)
  - Solar radiation (W/m²)
  - Wind speed (m/s)
  - Relative humidity (%)
  - Soil moisture (fractional)
  - Snow water equivalent (mm)
  - Evapotranspiration (mm/day)

The module provides:
  1. ClimateEncoder: embeds multi-variable climate forecasts
  2. ClimateModulator: modulates BiogeochemicalODE parameters based on
     climate forcing (e.g., temperature → reaction rates, precip → flow)
  3. SeasonalPrior: learned seasonal baseline for each state variable
  4. ClimateScenarioRunner: runs digital twin under IPCC SSP scenarios

MIT License — Bryan Cheng, 2026
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLIMATE_VARS: Tuple[str, ...] = (
    "precipitation",
    "air_temperature",
    "solar_radiation",
    "wind_speed",
    "relative_humidity",
    "soil_moisture",
    "snow_water_equivalent",
    "evapotranspiration",
)
NUM_CLIMATE_VARS: int = len(CLIMATE_VARS)
CLIMATE_EMBED_DIM: int = 128
SENTINEL_EMBED_DIM: int = 256
NUM_STATE_VARS: int = 10

# IPCC SSP scenarios
SSP_SCENARIOS: Dict[str, Dict[str, float]] = {
    "SSP1-2.6": {"temp_delta": 1.8, "precip_delta": 1.05, "name": "Sustainability"},
    "SSP2-4.5": {"temp_delta": 2.7, "precip_delta": 1.08, "name": "Middle of the Road"},
    "SSP3-7.0": {"temp_delta": 3.6, "precip_delta": 1.12, "name": "Regional Rivalry"},
    "SSP5-8.5": {"temp_delta": 4.4, "precip_delta": 1.15, "name": "Fossil-Fueled Development"},
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ClimateForcing:
    """Container for climate forcing data."""
    # Raw climate variables: (B, T, C) where C = NUM_CLIMATE_VARS
    variables: torch.Tensor
    # Day-of-year for seasonal encoding: (B, T)
    day_of_year: torch.Tensor
    # Climate embedding: (B, T, climate_embed_dim)
    embedding: Optional[torch.Tensor] = None
    # SSP scenario adjustments applied
    scenario: Optional[str] = None


@dataclass
class ClimateModulatedState:
    """State with climate modulation applied."""
    # Modulated ODE parameters: (B, num_params)
    ode_param_scale: torch.Tensor
    ode_param_shift: torch.Tensor
    # Climate contribution to state derivatives: (B, NUM_STATE_VARS)
    climate_forcing_term: torch.Tensor
    # Seasonal baseline: (B, NUM_STATE_VARS)
    seasonal_baseline: torch.Tensor


# ---------------------------------------------------------------------------
# Climate Encoder
# ---------------------------------------------------------------------------

class ClimateEncoder(nn.Module):
    """Encode multi-variable climate time series into dense embeddings.

    Uses a 1D temporal convolution + self-attention architecture to capture
    both local weather patterns and long-range climate trends.

    Parameters
    ----------
    num_vars : int
        Number of input climate variables.
    embed_dim : int
        Output embedding dimension per time step.
    num_heads : int
        Number of attention heads.
    num_layers : int
        Number of transformer encoder layers.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        num_vars: int = NUM_CLIMATE_VARS,
        embed_dim: int = CLIMATE_EMBED_DIM,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_vars = num_vars
        self.embed_dim = embed_dim

        # Temporal convolution for local pattern extraction
        self.conv1 = nn.Conv1d(num_vars, embed_dim // 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim // 2, embed_dim, kernel_size=3, padding=1)
        self.conv_norm = nn.LayerNorm(embed_dim)

        # Seasonal encoding
        self.seasonal_embed = nn.Linear(4, embed_dim)  # sin/cos of doy + 2 harmonics

        # Transformer for temporal dependencies
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def _seasonal_encoding(self, day_of_year: torch.Tensor) -> torch.Tensor:
        """Create seasonal encoding from day of year."""
        doy_rad = day_of_year.float() * (2 * math.pi / 365.25)
        return torch.stack([
            torch.sin(doy_rad),
            torch.cos(doy_rad),
            torch.sin(2 * doy_rad),
            torch.cos(2 * doy_rad),
        ], dim=-1)

    def forward(
        self,
        climate_vars: torch.Tensor,
        day_of_year: torch.Tensor,
    ) -> torch.Tensor:
        """Encode climate time series.

        Args:
            climate_vars: (B, T, C) climate variable time series.
            day_of_year: (B, T) day of year for each time step.

        Returns:
            Climate embedding (B, T, embed_dim).
        """
        B, T, C = climate_vars.shape

        # 1D convolution: (B, C, T) -> (B, embed_dim, T) -> (B, T, embed_dim)
        x = climate_vars.permute(0, 2, 1)
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)
        x = self.conv_norm(x)

        # Add seasonal encoding
        seasonal = self._seasonal_encoding(day_of_year)
        seasonal_emb = self.seasonal_embed(seasonal)
        x = x + seasonal_emb

        # Self-attention over time
        x = self.transformer(x)

        return self.output_proj(x)


# ---------------------------------------------------------------------------
# Climate Modulator
# ---------------------------------------------------------------------------

class ClimateModulator(nn.Module):
    """Modulate ODE parameters and state derivatives based on climate.

    Maps climate embeddings to:
    1. Scale/shift for BiogeochemicalODE parameters (e.g., temperature →
       reaction rate Arrhenius adjustment)
    2. Additive forcing terms for state derivatives (e.g., precipitation →
       nutrient loading)
    3. Seasonal baseline state expectations

    Parameters
    ----------
    climate_dim : int
        Climate embedding dimension.
    sentinel_dim : int
        SENTINEL embedding dimension.
    num_ode_params : int
        Number of ODE parameters to modulate.
    num_states : int
        Number of state variables.
    """

    def __init__(
        self,
        climate_dim: int = CLIMATE_EMBED_DIM,
        sentinel_dim: int = SENTINEL_EMBED_DIM,
        num_ode_params: int = 24,
        num_states: int = NUM_STATE_VARS,
    ) -> None:
        super().__init__()
        combined_dim = climate_dim + sentinel_dim

        # ODE parameter modulation
        self.param_modulator = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, num_ode_params * 2),  # scale + shift
        )
        self.num_ode_params = num_ode_params

        # Direct forcing terms
        self.forcing_net = nn.Sequential(
            nn.Linear(climate_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_states),
            nn.Tanh(),  # Bounded forcing
        )

        # Forcing magnitude (learnable per-state)
        self.forcing_scale = nn.Parameter(torch.ones(num_states) * 0.1)

        # Seasonal baseline
        self.seasonal_baseline = SeasonalPrior(num_states=num_states)

    def forward(
        self,
        climate_embedding: torch.Tensor,
        sentinel_embedding: torch.Tensor,
        day_of_year: torch.Tensor,
    ) -> ClimateModulatedState:
        """Compute climate modulation.

        Args:
            climate_embedding: (B, climate_dim) -- summary of climate forecast.
            sentinel_embedding: (B, sentinel_dim) -- SENTINEL multimodal embedding.
            day_of_year: (B,) -- current day of year.

        Returns:
            ClimateModulatedState with modulated parameters and forcings.
        """
        # Combined embedding for parameter modulation
        combined = torch.cat([climate_embedding, sentinel_embedding], dim=-1)
        param_out = self.param_modulator(combined)
        param_scale = param_out[:, :self.num_ode_params]
        param_shift = param_out[:, self.num_ode_params:]

        # Bounded scale: exp(tanh(x)) ∈ [0.37, 2.72]
        param_scale = torch.exp(torch.tanh(param_scale))

        # Direct climate forcing
        forcing = self.forcing_net(climate_embedding) * self.forcing_scale

        # Seasonal baseline
        seasonal = self.seasonal_baseline(day_of_year)

        return ClimateModulatedState(
            ode_param_scale=param_scale,
            ode_param_shift=param_shift,
            climate_forcing_term=forcing,
            seasonal_baseline=seasonal,
        )


# ---------------------------------------------------------------------------
# Seasonal Prior
# ---------------------------------------------------------------------------

class SeasonalPrior(nn.Module):
    """Learned seasonal baseline for each state variable.

    Uses Fourier basis functions to capture annual and semi-annual
    cycles in ecological state variables.

    Parameters
    ----------
    num_states : int
        Number of state variables.
    num_harmonics : int
        Number of Fourier harmonics (1 = annual, 2 = +semiannual, etc.).
    """

    def __init__(
        self,
        num_states: int = NUM_STATE_VARS,
        num_harmonics: int = 3,
    ) -> None:
        super().__init__()
        self.num_harmonics = num_harmonics
        # Fourier coefficients: (num_states, 2 * num_harmonics + 1)
        # Layout: [mean, a1, b1, a2, b2, a3, b3]
        self.coefficients = nn.Parameter(
            torch.zeros(num_states, 2 * num_harmonics + 1)
        )
        # Initialize means to typical freshwater values
        with torch.no_grad():
            self.coefficients[:, 0] = torch.tensor([
                8.0,   # DO (mg/L)
                3.0,   # BOD
                1.5,   # TN
                0.1,   # TP
                5.0,   # Chl-a
                15.0,  # Temp
                7.5,   # pH
                10.0,  # Turbidity
                5.0,   # DOC
                50.0,  # Sediment
            ])
            # Temperature has strong annual cycle
            self.coefficients[5, 1] = 10.0  # 10°C amplitude
            # DO inversely correlated with temperature
            self.coefficients[0, 1] = -2.0
            # Chl-a peaks in summer
            self.coefficients[4, 1] = 3.0

    def forward(self, day_of_year: torch.Tensor) -> torch.Tensor:
        """Compute seasonal baseline.

        Args:
            day_of_year: (B,) or (B, 1) day of year.

        Returns:
            Seasonal baseline state (B, num_states).
        """
        if day_of_year.dim() > 1:
            day_of_year = day_of_year.squeeze(-1)

        doy_rad = day_of_year.float() * (2 * math.pi / 365.25)
        B = doy_rad.shape[0]
        D = self.coefficients.shape[0]

        # Build Fourier basis: [1, cos(wt), sin(wt), cos(2wt), sin(2wt), ...]
        basis = [torch.ones(B, 1, device=doy_rad.device)]
        for k in range(1, self.num_harmonics + 1):
            basis.append(torch.cos(k * doy_rad).unsqueeze(-1))
            basis.append(torch.sin(k * doy_rad).unsqueeze(-1))
        basis = torch.cat(basis, dim=-1)  # (B, 2H+1)

        # Multiply: (D, 2H+1) @ (B, 2H+1, 1) -> (B, D)
        seasonal = torch.mm(basis, self.coefficients.t())  # (B, D)

        return seasonal


# ---------------------------------------------------------------------------
# Climate Scenario Runner
# ---------------------------------------------------------------------------

class ClimateScenarioRunner(nn.Module):
    """Run digital twin under IPCC SSP climate scenarios.

    Applies scenario-specific adjustments to baseline climate forcing:
    - Temperature offsets (global mean surface temperature change)
    - Precipitation scaling (relative change in precipitation)
    - Extreme event frequency multipliers

    Parameters
    ----------
    climate_encoder : ClimateEncoder
        Pre-trained climate encoder.
    climate_modulator : ClimateModulator
        Pre-trained climate modulator.
    """

    def __init__(
        self,
        climate_encoder: ClimateEncoder,
        climate_modulator: ClimateModulator,
    ) -> None:
        super().__init__()
        self.encoder = climate_encoder
        self.modulator = climate_modulator

    def apply_scenario(
        self,
        baseline_climate: torch.Tensor,
        scenario: str,
        years_ahead: float = 0.0,
    ) -> torch.Tensor:
        """Apply SSP scenario adjustments to baseline climate.

        Args:
            baseline_climate: (B, T, C) baseline climate variables.
            scenario: SSP scenario name (e.g., "SSP2-4.5").
            years_ahead: Years into the future (for time-varying adjustments).

        Returns:
            Adjusted climate variables (B, T, C).
        """
        if scenario not in SSP_SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario}. "
                             f"Choose from {list(SSP_SCENARIOS.keys())}")

        params = SSP_SCENARIOS[scenario]
        adjusted = baseline_climate.clone()

        # Linear interpolation of warming over time
        # Assume full delta by 2100, current year ~2026, baseline ~2020
        fraction = min(1.0, years_ahead / 80.0)

        # Temperature adjustment (index 1 = air_temperature)
        temp_delta = params["temp_delta"] * fraction
        adjusted[:, :, 1] = adjusted[:, :, 1] + temp_delta

        # Precipitation scaling (index 0 = precipitation)
        precip_scale = 1.0 + (params["precip_delta"] - 1.0) * fraction
        adjusted[:, :, 0] = adjusted[:, :, 0] * precip_scale

        # Evapotranspiration increases with temperature (index 7)
        et_increase = temp_delta * 0.05  # ~5% per °C
        adjusted[:, :, 7] = adjusted[:, :, 7] * (1.0 + et_increase)

        # Snow water equivalent decreases (index 6)
        swe_decrease = min(0.5, temp_delta * 0.1)
        adjusted[:, :, 6] = adjusted[:, :, 6] * (1.0 - swe_decrease)

        return adjusted

    def run_scenario(
        self,
        twin_engine: nn.Module,
        sentinel_embedding: torch.Tensor,
        baseline_climate: torch.Tensor,
        day_of_year: torch.Tensor,
        scenario: str,
        years_ahead: float = 10.0,
        horizons: Tuple[int, ...] = (1, 7, 14, 30, 90, 365),
    ) -> Dict[str, torch.Tensor]:
        """Run full scenario simulation.

        Args:
            twin_engine: DigitalTwinEngine instance.
            sentinel_embedding: (B, 256) current SENTINEL embedding.
            baseline_climate: (B, T, C) baseline climate variables.
            day_of_year: (B, T) day of year.
            scenario: SSP scenario name.
            years_ahead: Years into the future.
            horizons: Forecast horizons in days.

        Returns:
            Dictionary with scenario predictions and comparisons.
        """
        # Baseline run
        baseline_clim_emb = self.encoder(baseline_climate, day_of_year)
        baseline_clim_summary = baseline_clim_emb.mean(dim=1)  # (B, climate_dim)
        doy_scalar = day_of_year[:, 0] if day_of_year.dim() > 1 else day_of_year

        baseline_mod = self.modulator(baseline_clim_summary, sentinel_embedding, doy_scalar)
        baseline_pred = twin_engine(sentinel_embedding, horizons=horizons)

        # Scenario run
        adjusted_climate = self.apply_scenario(baseline_climate, scenario, years_ahead)
        scenario_clim_emb = self.encoder(adjusted_climate, day_of_year)
        scenario_clim_summary = scenario_clim_emb.mean(dim=1)
        scenario_mod = self.modulator(scenario_clim_summary, sentinel_embedding, doy_scalar)
        scenario_pred = twin_engine(sentinel_embedding, horizons=horizons)

        # Compute deltas
        return {
            "baseline_predictions": baseline_pred.predictions,
            "scenario_predictions": scenario_pred.predictions,
            "delta": scenario_pred.predictions - baseline_pred.predictions,
            "climate_forcing_baseline": baseline_mod.climate_forcing_term,
            "climate_forcing_scenario": scenario_mod.climate_forcing_term,
            "scenario": scenario,
            "years_ahead": years_ahead,
            "seasonal_baseline": baseline_mod.seasonal_baseline,
        }


# ---------------------------------------------------------------------------
# Integration helper
# ---------------------------------------------------------------------------

class ClimateCoupledTwin(nn.Module):
    """Digital twin with integrated climate coupling.

    Wraps DigitalTwinEngine + ClimateEncoder + ClimateModulator into
    a single module for climate-aware ecosystem forecasting.
    """

    def __init__(
        self,
        twin_engine: nn.Module,
        climate_embed_dim: int = CLIMATE_EMBED_DIM,
        num_climate_vars: int = NUM_CLIMATE_VARS,
    ) -> None:
        super().__init__()
        self.twin = twin_engine
        self.climate_encoder = ClimateEncoder(
            num_vars=num_climate_vars,
            embed_dim=climate_embed_dim,
        )
        self.climate_modulator = ClimateModulator(
            climate_dim=climate_embed_dim,
        )
        self.scenario_runner = ClimateScenarioRunner(
            self.climate_encoder,
            self.climate_modulator,
        )

        # Projection: climate embedding → augmented SENTINEL embedding
        self.climate_projection = nn.Sequential(
            nn.Linear(SENTINEL_EMBED_DIM + climate_embed_dim, SENTINEL_EMBED_DIM),
            nn.GELU(),
            nn.Linear(SENTINEL_EMBED_DIM, SENTINEL_EMBED_DIM),
        )

    def forward(
        self,
        sentinel_embedding: torch.Tensor,
        climate_vars: Optional[torch.Tensor] = None,
        day_of_year: Optional[torch.Tensor] = None,
        horizons: Tuple[int, ...] = (1, 7, 14, 30, 90, 365),
        state_override: Optional[torch.Tensor] = None,
    ):
        """Climate-augmented digital twin forward pass.

        If climate data is provided, augments the SENTINEL embedding
        with climate information before running the twin.

        Args:
            sentinel_embedding: (B, 256) SENTINEL embedding.
            climate_vars: (B, T, C) climate time series (optional).
            day_of_year: (B, T) day of year (optional).
            horizons: Forecast horizons in days.
            state_override: Optional initial state override.

        Returns:
            TwinOutput from the underlying twin engine.
        """
        if climate_vars is not None and day_of_year is not None:
            # Encode climate
            clim_emb = self.climate_encoder(climate_vars, day_of_year)
            clim_summary = clim_emb.mean(dim=1)  # (B, climate_dim)

            # Augment SENTINEL embedding with climate info
            combined = torch.cat([sentinel_embedding, clim_summary], dim=-1)
            augmented_emb = self.climate_projection(combined)
        else:
            augmented_emb = sentinel_embedding

        return self.twin(augmented_emb, horizons=horizons, state_override=state_override)
