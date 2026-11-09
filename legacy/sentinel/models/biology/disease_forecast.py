"""Disease outbreak forecasting heads for SENTINEL (Phase 3.3).

Consumes the 256-dim environmental embeddings produced by the core SENTINEL
model and predicts disease / pathogen risk at monitoring sites 7 and 14 days
ahead.  Each forecaster outputs point predictions **and** calibrated
uncertainty estimates via MC-dropout and conformal prediction intervals.

Target diseases
---------------
1. **Cyanotoxin concentrations** -- microcystin-LR, anatoxin-a,
   cylindrospermopsin.  Regression + WHO-threshold exceedance probability.
2. **Vibrio risk index** -- V. vulnificus, V. parahaemolyticus.
   Classification (risk index 0-1, expected case rate).
3. **Naegleria fowleri habitat probability** -- binary suitability for
   primary amebic meningoencephalitis (97 % fatality rate).
4. **Schistosomiasis snail-host habitat** -- intermediate host
   (Biomphalaria / Bulinus) suitability.

Architecture overview
---------------------
::

    SENTINEL embedding (B, 256) + temporal context
        --> shared environmental encoder (256+T -> 256 -> 128)
            --> disease-specific prediction heads
                --> point predictions + MC-dropout uncertainty

All heads support two forecast horizons (7 and 14 days).
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Constants & thresholds
# ============================================================================

# --- Cyanotoxins -----------------------------------------------------------

CYANOTOXIN_NAMES: List[str] = [
    "microcystin_LR",
    "anatoxin_a",
    "cylindrospermopsin",
]

# WHO thresholds (ng/L)
WHO_DRINKING_WATER_THRESHOLD: Dict[str, float] = {
    "microcystin_LR": 1_000.0,       # 1 ug/L = 1000 ng/L
    "anatoxin_a": 30_000.0,          # provisional
    "cylindrospermopsin": 700.0,     # provisional
}

WHO_RECREATIONAL_THRESHOLD: Dict[str, float] = {
    "microcystin_LR": 24_000.0,     # 24 ug/L = 24000 ng/L
    "anatoxin_a": 60_000.0,         # provisional
    "cylindrospermopsin": 6_000.0,  # provisional
}

# --- Vibrio ----------------------------------------------------------------

VIBRIO_SPECIES: List[str] = [
    "V_vulnificus",
    "V_parahaemolyticus",
]

# Environmental driver thresholds
VIBRIO_TEMP_THRESHOLD_C: float = 20.0   # growth onset
VIBRIO_SALINITY_RANGE_PPT: Tuple[float, float] = (5.0, 25.0)

# --- Naegleria fowleri -----------------------------------------------------

NAEGLERIA_TEMP_THRESHOLD_C: float = 30.0   # proliferation onset

# --- Forecast horizons -----------------------------------------------------

FORECAST_HORIZONS_DAYS: List[int] = [7, 14]
NUM_HORIZONS: int = len(FORECAST_HORIZONS_DAYS)

# --- Temporal encoding ------------------------------------------------------

TEMPORAL_DIM: int = 6  # sin(doy) + cos(doy) + 4 season one-hot


# ============================================================================
# Dataclass outputs
# ============================================================================

@dataclass
class CyanotoxinOutput:
    """Output of the cyanotoxin forecaster.

    All concentration tensors are in **log10(ng/L)** space.
    """

    # Point predictions: (B, 6) = 3 toxins x 2 horizons
    log_concentration: torch.Tensor

    # Threshold exceedance probabilities: (B, 6) -- P(conc > WHO drinking)
    drinking_exceedance_prob: torch.Tensor

    # Threshold exceedance probabilities: (B, 6) -- P(conc > WHO recreational)
    recreational_exceedance_prob: torch.Tensor

    # Uncertainty -- populated when MC-dropout is used
    mc_log_conc_mean: Optional[torch.Tensor] = None   # (B, 6)
    mc_log_conc_std: Optional[torch.Tensor] = None     # (B, 6)

    # Conformal prediction interval half-widths (set after calibration)
    conformal_half_width: Optional[torch.Tensor] = None  # (B, 6)


@dataclass
class VibrioOutput:
    """Output of the Vibrio risk forecaster."""

    # Risk index 0-1: (B, 4) = 2 species x 2 horizons
    risk_index: torch.Tensor

    # Expected case rate per 100k: (B, 4)
    expected_case_rate: torch.Tensor

    # MC-dropout uncertainty
    mc_risk_mean: Optional[torch.Tensor] = None   # (B, 4)
    mc_risk_std: Optional[torch.Tensor] = None     # (B, 4)


@dataclass
class NaegleriaOutput:
    """Output of the Naegleria fowleri habitat forecaster."""

    # Habitat suitability probability: (B, 2) = 2 horizons
    habitat_probability: torch.Tensor

    # MC-dropout uncertainty
    mc_prob_mean: Optional[torch.Tensor] = None  # (B, 2)
    mc_prob_std: Optional[torch.Tensor] = None    # (B, 2)


@dataclass
class SchistosomiasisOutput:
    """Output of the schistosomiasis snail-host habitat forecaster."""

    # Snail habitat suitability probability: (B, 2) = 2 horizons
    habitat_probability: torch.Tensor

    # MC-dropout uncertainty
    mc_prob_mean: Optional[torch.Tensor] = None  # (B, 2)
    mc_prob_std: Optional[torch.Tensor] = None    # (B, 2)


class AlertLevel(enum.IntEnum):
    """Risk alert levels for integrated disease assessment."""

    LOW = 0
    MODERATE = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class DiseaseRiskSummary:
    """Comprehensive per-site disease risk assessment."""

    cyanotoxin: CyanotoxinOutput
    vibrio: VibrioOutput
    naegleria: NaegleriaOutput
    schistosomiasis: SchistosomiasisOutput

    # Overall alert level per site: (B,) of AlertLevel ints
    alert_level: torch.Tensor

    # Human-readable reason for each site's alert level
    alert_reasons: Optional[List[List[str]]] = None


# ============================================================================
# Temporal encoding helpers
# ============================================================================

def encode_temporal_context(
    day_of_year: torch.Tensor,
    *,
    max_days: float = 365.25,
) -> torch.Tensor:
    """Build a temporal context vector from day-of-year.

    Parameters
    ----------
    day_of_year : Tensor[B]
        Day of the year (1-366).  Float for smooth interpolation.

    Returns
    -------
    Tensor[B, 6]
        [sin(annual), cos(annual), spring, summer, fall, winter]
    """
    # Sinusoidal annual cycle
    angle = 2.0 * math.pi * day_of_year / max_days
    sin_doy = torch.sin(angle)
    cos_doy = torch.cos(angle)

    # Season one-hot (meteorological seasons, Northern Hemisphere)
    season = torch.zeros(day_of_year.shape[0], 4, device=day_of_year.device)
    doy = day_of_year.long()
    spring = (doy >= 60) & (doy < 152)    # Mar-May
    summer = (doy >= 152) & (doy < 244)   # Jun-Aug
    fall = (doy >= 244) & (doy < 335)     # Sep-Nov
    winter = ~(spring | summer | fall)     # Dec-Feb

    season[:, 0] = spring.float()
    season[:, 1] = summer.float()
    season[:, 2] = fall.float()
    season[:, 3] = winter.float()

    return torch.cat([sin_doy.unsqueeze(-1), cos_doy.unsqueeze(-1), season], dim=-1)


# ============================================================================
# Base class
# ============================================================================

class DiseaseForecaster(nn.Module):
    """Abstract base class for disease-specific forecasting heads.

    Provides:
        - Shared environmental encoder (SENTINEL embed + temporal -> 128)
        - MC-dropout infrastructure (enable/disable, N forward passes)
        - Weight initialisation

    Subclasses must implement ``_disease_forward`` and ``forward``.

    Parameters
    ----------
    embedding_dim : int
        Dimensionality of the SENTINEL environmental embedding (default 256).
    temporal_dim : int
        Dimensionality of the temporal context vector (default 6).
    hidden_dim : int
        Hidden layer size in the shared encoder (default 256).
    encoder_out_dim : int
        Output size of the shared encoder (default 128).
    dropout : float
        Dropout probability for MC-dropout (default 0.2).
    mc_samples : int
        Number of stochastic forward passes for MC-dropout (default 20).
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        temporal_dim: int = TEMPORAL_DIM,
        hidden_dim: int = 256,
        encoder_out_dim: int = 128,
        dropout: float = 0.2,
        mc_samples: int = 20,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.temporal_dim = temporal_dim
        self.encoder_out_dim = encoder_out_dim
        self.dropout_p = dropout
        self.mc_samples = mc_samples

        # Shared environmental encoder
        self.env_encoder = nn.Sequential(
            nn.Linear(embedding_dim + temporal_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, encoder_out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        """Kaiming initialisation for all Linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    def _encode(
        self,
        embedding: torch.Tensor,
        temporal: torch.Tensor,
    ) -> torch.Tensor:
        """Run the shared encoder.

        Parameters
        ----------
        embedding : Tensor[B, 256]
            SENTINEL environmental embedding.
        temporal : Tensor[B, temporal_dim]
            Temporal context vector (from ``encode_temporal_context``).

        Returns
        -------
        Tensor[B, encoder_out_dim]
        """
        x = torch.cat([embedding, temporal], dim=-1)
        return self.env_encoder(x)

    # ------------------------------------------------------------------
    def _enable_dropout(self) -> None:
        """Set all Dropout modules to train mode (stochastic)."""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def _disable_dropout(self) -> None:
        """Set all Dropout modules back to eval mode (deterministic)."""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.eval()


# ============================================================================
# Conformal prediction calibration buffer
# ============================================================================

class ConformalCalibrator:
    """Lightweight split-conformal prediction interval calibrator.

    Maintains a buffer of non-conformity scores (absolute residuals)
    collected on a calibration set.  At inference time, computes the
    ``(1 - alpha)`` quantile to produce marginal coverage guarantees.

    Parameters
    ----------
    alpha : float
        Mis-coverage rate (default 0.10 for 90 % coverage).
    max_buffer : int
        Maximum number of calibration residuals to store.
    """

    def __init__(self, alpha: float = 0.10, max_buffer: int = 50_000) -> None:
        self.alpha = alpha
        self.max_buffer = max_buffer
        self._scores: List[torch.Tensor] = []

    # ------------------------------------------------------------------
    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """Append absolute residuals from a calibration batch.

        Parameters
        ----------
        predictions : Tensor[B, D]
        targets : Tensor[B, D]
        """
        residuals = (predictions - targets).abs().detach().cpu()
        self._scores.append(residuals)
        # Trim if too large
        all_scores = torch.cat(self._scores, dim=0)
        if all_scores.shape[0] > self.max_buffer:
            all_scores = all_scores[-self.max_buffer:]
            self._scores = [all_scores]

    # ------------------------------------------------------------------
    def quantile(self) -> torch.Tensor:
        """Compute the conformal quantile across calibration scores.

        Returns
        -------
        Tensor[D]
            Per-output-dimension half-width for the prediction interval.
        """
        if not self._scores:
            raise RuntimeError(
                "ConformalCalibrator has no calibration scores.  "
                "Call .update() on a held-out calibration set first."
            )
        all_scores = torch.cat(self._scores, dim=0)  # (N, D)
        n = all_scores.shape[0]
        # Finite-sample correction: ceil((n+1)(1-alpha)) / n
        q_level = min(math.ceil((n + 1) * (1.0 - self.alpha)) / n, 1.0)
        return torch.quantile(all_scores, q_level, dim=0)  # (D,)

    @property
    def num_scores(self) -> int:
        if not self._scores:
            return 0
        return sum(s.shape[0] for s in self._scores)


# ============================================================================
# 1. Cyanotoxin forecaster
# ============================================================================

class CyanotoxinForecaster(DiseaseForecaster):
    """Predict cyanotoxin concentrations 7 and 14 days ahead.

    Outputs log10(ng/L) concentrations for three toxins at two horizons,
    plus WHO drinking-water and recreational threshold exceedance
    probabilities.

    Output dimension: 6 = 3 toxins x 2 horizons.

    Parameters
    ----------
    embedding_dim, temporal_dim, hidden_dim, encoder_out_dim, dropout,
    mc_samples : see ``DiseaseForecaster``.
    conformal_alpha : float
        Mis-coverage rate for conformal prediction intervals (default 0.10).
    """

    TOXIN_NAMES: List[str] = CYANOTOXIN_NAMES
    NUM_TOXINS: int = len(CYANOTOXIN_NAMES)
    NUM_OUTPUTS: int = NUM_TOXINS * NUM_HORIZONS  # 6

    # WHO thresholds in log10(ng/L)
    DRINKING_THRESHOLDS_LOG: List[float] = [
        math.log10(WHO_DRINKING_WATER_THRESHOLD[t]) for t in CYANOTOXIN_NAMES
    ]
    RECREATIONAL_THRESHOLDS_LOG: List[float] = [
        math.log10(WHO_RECREATIONAL_THRESHOLD[t]) for t in CYANOTOXIN_NAMES
    ]

    def __init__(
        self,
        embedding_dim: int = 256,
        temporal_dim: int = TEMPORAL_DIM,
        hidden_dim: int = 256,
        encoder_out_dim: int = 128,
        dropout: float = 0.2,
        mc_samples: int = 20,
        conformal_alpha: float = 0.10,
    ) -> None:
        super().__init__(
            embedding_dim=embedding_dim,
            temporal_dim=temporal_dim,
            hidden_dim=hidden_dim,
            encoder_out_dim=encoder_out_dim,
            dropout=dropout,
            mc_samples=mc_samples,
        )

        # Regression head: log10(concentration) for 3 toxins x 2 horizons
        self.concentration_head = nn.Sequential(
            nn.Linear(encoder_out_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, self.NUM_OUTPUTS),
        )

        # Exceedance classification head -- predicts logits for P(conc > threshold)
        # Separate head so it can learn sharp decision boundaries
        self.drinking_exceedance_head = nn.Sequential(
            nn.Linear(encoder_out_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, self.NUM_OUTPUTS),
        )

        self.recreational_exceedance_head = nn.Sequential(
            nn.Linear(encoder_out_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, self.NUM_OUTPUTS),
        )

        # Conformal calibrator (per-output dimension)
        self.conformal = ConformalCalibrator(alpha=conformal_alpha)

        self._init_weights()

    # ------------------------------------------------------------------
    def _disease_forward(
        self, h: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single forward through disease-specific heads.

        Returns (log_conc, drink_logits, rec_logits).
        """
        log_conc = self.concentration_head(h)             # (B, 6)
        drink_logits = self.drinking_exceedance_head(h)   # (B, 6)
        rec_logits = self.recreational_exceedance_head(h) # (B, 6)
        return log_conc, drink_logits, rec_logits

    # ------------------------------------------------------------------
    def forward(
        self,
        embedding: torch.Tensor,
        temporal: torch.Tensor,
        use_mc_dropout: bool = False,
    ) -> CyanotoxinOutput:
        """Forward pass.

        Parameters
        ----------
        embedding : Tensor[B, 256]
            SENTINEL environmental embedding.
        temporal : Tensor[B, temporal_dim]
            Temporal context vector.
        use_mc_dropout : bool
            If True, perform ``mc_samples`` stochastic forward passes.

        Returns
        -------
        CyanotoxinOutput
        """
        h = self._encode(embedding, temporal)

        if not use_mc_dropout or self.training:
            log_conc, drink_logits, rec_logits = self._disease_forward(h)
            return CyanotoxinOutput(
                log_concentration=log_conc,
                drinking_exceedance_prob=torch.sigmoid(drink_logits),
                recreational_exceedance_prob=torch.sigmoid(rec_logits),
            )

        # --- MC-Dropout inference -----------------------------------------
        self._enable_dropout()

        mc_conc: List[torch.Tensor] = []
        mc_drink: List[torch.Tensor] = []
        mc_rec: List[torch.Tensor] = []
        for _ in range(self.mc_samples):
            h_mc = self._encode(embedding, temporal)
            lc, dl, rl = self._disease_forward(h_mc)
            mc_conc.append(lc)
            mc_drink.append(torch.sigmoid(dl))
            mc_rec.append(torch.sigmoid(rl))

        self._disable_dropout()

        mc_conc_t = torch.stack(mc_conc, dim=0)    # (N, B, 6)
        mc_drink_t = torch.stack(mc_drink, dim=0)
        mc_rec_t = torch.stack(mc_rec, dim=0)

        conc_mean = mc_conc_t.mean(dim=0)
        conc_std = mc_conc_t.std(dim=0)

        # Conformal interval (if calibrated)
        conf_hw: Optional[torch.Tensor] = None
        if self.conformal.num_scores > 0:
            q = self.conformal.quantile().to(conc_mean.device)
            conf_hw = q.unsqueeze(0).expand_as(conc_mean)

        return CyanotoxinOutput(
            log_concentration=conc_mean,
            drinking_exceedance_prob=mc_drink_t.mean(dim=0),
            recreational_exceedance_prob=mc_rec_t.mean(dim=0),
            mc_log_conc_mean=conc_mean,
            mc_log_conc_std=conc_std,
            conformal_half_width=conf_hw,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def compute_loss(
        output: CyanotoxinOutput,
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Multi-task loss: regression + exceedance classification.

        Parameters
        ----------
        output : CyanotoxinOutput
        targets : dict
            ``"log_concentration"`` : Tensor[B, 6] -- true log10(ng/L).
            ``"drinking_exceedance"`` : Tensor[B, 6] -- binary (0/1).
            ``"recreational_exceedance"`` : Tensor[B, 6] -- binary (0/1).

        Returns
        -------
        total_loss, per_task_losses
        """
        losses: Dict[str, torch.Tensor] = {}
        device = output.log_concentration.device

        if "log_concentration" in targets:
            pred_lc = torch.nan_to_num(output.log_concentration, nan=0.0)
            tgt_lc = torch.nan_to_num(targets["log_concentration"], nan=0.0)
            losses["concentration_mse"] = F.mse_loss(pred_lc, tgt_lc)

        if "drinking_exceedance" in targets:
            pred_drink = torch.nan_to_num(output.drinking_exceedance_prob, nan=0.5).clamp(1e-7, 1 - 1e-7)
            tgt_drink = torch.nan_to_num(targets["drinking_exceedance"].float(), nan=0.0).clamp(0, 1)
            losses["drinking_bce"] = F.binary_cross_entropy(pred_drink, tgt_drink)

        if "recreational_exceedance" in targets:
            pred_rec = torch.nan_to_num(output.recreational_exceedance_prob, nan=0.5).clamp(1e-7, 1 - 1e-7)
            tgt_rec = torch.nan_to_num(targets["recreational_exceedance"].float(), nan=0.0).clamp(0, 1)
            losses["recreational_bce"] = F.binary_cross_entropy(pred_rec, tgt_rec)

        total = torch.tensor(0.0, device=device)
        weights = {
            "concentration_mse": 1.0,
            "drinking_bce": 0.5,
            "recreational_bce": 0.3,
        }
        for key, val in losses.items():
            total = total + weights.get(key, 1.0) * val

        return total, losses


# ============================================================================
# 2. Vibrio risk forecaster
# ============================================================================

class VibrioRiskForecaster(DiseaseForecaster):
    """Predict Vibrio species risk index for coastal/estuarine waters.

    Explicitly models water temperature and salinity as additional inputs
    because they are the primary environmental drivers of Vibrio growth.

    Output dimension: 4 = 2 species x 2 horizons.

    Parameters
    ----------
    embedding_dim, temporal_dim, hidden_dim, encoder_out_dim, dropout,
    mc_samples : see ``DiseaseForecaster``.
    covariate_dim : int
        Extra covariates appended to the encoder output (default 2:
        water temperature in Celsius, salinity in ppt).
    """

    SPECIES_NAMES: List[str] = VIBRIO_SPECIES
    NUM_SPECIES: int = len(VIBRIO_SPECIES)
    NUM_OUTPUTS: int = NUM_SPECIES * NUM_HORIZONS  # 4

    TEMP_THRESHOLD_C: float = VIBRIO_TEMP_THRESHOLD_C
    SALINITY_RANGE_PPT: Tuple[float, float] = VIBRIO_SALINITY_RANGE_PPT

    def __init__(
        self,
        embedding_dim: int = 256,
        temporal_dim: int = TEMPORAL_DIM,
        hidden_dim: int = 256,
        encoder_out_dim: int = 128,
        dropout: float = 0.2,
        mc_samples: int = 20,
        covariate_dim: int = 2,
    ) -> None:
        super().__init__(
            embedding_dim=embedding_dim,
            temporal_dim=temporal_dim,
            hidden_dim=hidden_dim,
            encoder_out_dim=encoder_out_dim,
            dropout=dropout,
            mc_samples=mc_samples,
        )
        self.covariate_dim = covariate_dim

        # Temperature / salinity interaction encoder
        self.covariate_encoder = nn.Sequential(
            nn.Linear(covariate_dim, 32),
            nn.GELU(),
            nn.Linear(32, 32),
        )

        head_input_dim = encoder_out_dim + 32

        # Risk index head: sigmoid -> [0, 1]
        self.risk_head = nn.Sequential(
            nn.Linear(head_input_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, self.NUM_OUTPUTS),
            nn.Sigmoid(),
        )

        # Case rate head: softplus -> non-negative
        self.case_rate_head = nn.Sequential(
            nn.Linear(head_input_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, self.NUM_OUTPUTS),
            nn.Softplus(),
        )

        self._init_weights()

    # ------------------------------------------------------------------
    def _build_head_input(
        self,
        h_env: torch.Tensor,
        covariates: torch.Tensor,
    ) -> torch.Tensor:
        """Combine encoder output with covariate features."""
        h_cov = self.covariate_encoder(covariates)  # (B, 32)
        return torch.cat([h_env, h_cov], dim=-1)    # (B, 128+32)

    # ------------------------------------------------------------------
    def _disease_forward(
        self, h: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single forward through risk + case rate heads."""
        risk = self.risk_head(h)            # (B, 4)
        case_rate = self.case_rate_head(h)  # (B, 4)
        return risk, case_rate

    # ------------------------------------------------------------------
    def forward(
        self,
        embedding: torch.Tensor,
        temporal: torch.Tensor,
        covariates: torch.Tensor,
        use_mc_dropout: bool = False,
    ) -> VibrioOutput:
        """Forward pass.

        Parameters
        ----------
        embedding : Tensor[B, 256]
            SENTINEL environmental embedding.
        temporal : Tensor[B, temporal_dim]
            Temporal context vector.
        covariates : Tensor[B, 2]
            Column 0: water temperature (Celsius).
            Column 1: salinity (ppt).
        use_mc_dropout : bool
            If True, perform ``mc_samples`` stochastic forward passes.

        Returns
        -------
        VibrioOutput
        """
        h_env = self._encode(embedding, temporal)
        h = self._build_head_input(h_env, covariates)

        if not use_mc_dropout or self.training:
            risk, case_rate = self._disease_forward(h)
            return VibrioOutput(risk_index=risk, expected_case_rate=case_rate)

        # --- MC-Dropout inference -----------------------------------------
        self._enable_dropout()

        mc_risk: List[torch.Tensor] = []
        mc_case: List[torch.Tensor] = []
        for _ in range(self.mc_samples):
            h_env_mc = self._encode(embedding, temporal)
            h_mc = self._build_head_input(h_env_mc, covariates)
            r, c = self._disease_forward(h_mc)
            mc_risk.append(r)
            mc_case.append(c)

        self._disable_dropout()

        mc_risk_t = torch.stack(mc_risk, dim=0)  # (N, B, 4)
        mc_case_t = torch.stack(mc_case, dim=0)

        return VibrioOutput(
            risk_index=mc_risk_t.mean(dim=0),
            expected_case_rate=mc_case_t.mean(dim=0),
            mc_risk_mean=mc_risk_t.mean(dim=0),
            mc_risk_std=mc_risk_t.std(dim=0),
        )

    # ------------------------------------------------------------------
    @staticmethod
    def compute_loss(
        output: VibrioOutput,
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Loss for Vibrio risk forecasting.

        Parameters
        ----------
        output : VibrioOutput
        targets : dict
            ``"risk_index"`` : Tensor[B, 4] -- true risk [0, 1].
            ``"case_rate"`` : Tensor[B, 4] -- true cases per 100k.

        Returns
        -------
        total_loss, per_task_losses
        """
        losses: Dict[str, torch.Tensor] = {}
        device = output.risk_index.device

        if "risk_index" in targets:
            pred_ri = torch.nan_to_num(output.risk_index, nan=0.5).clamp(1e-7, 1 - 1e-7)
            tgt_ri = torch.nan_to_num(targets["risk_index"].float(), nan=0.0).clamp(0, 1)
            losses["risk_bce"] = F.binary_cross_entropy(pred_ri, tgt_ri)

        if "case_rate" in targets:
            # Poisson-like loss: treating case_rate as a rate parameter
            losses["case_rate_mse"] = F.mse_loss(
                output.expected_case_rate, targets["case_rate"].float(),
            )

        total = torch.tensor(0.0, device=device)
        weights = {"risk_bce": 1.0, "case_rate_mse": 0.5}
        for key, val in losses.items():
            total = total + weights.get(key, 1.0) * val

        return total, losses


# ============================================================================
# 3. Naegleria fowleri forecaster
# ============================================================================

class NaegleriaForecaster(DiseaseForecaster):
    """Predict Naegleria fowleri habitat suitability probability.

    Primary amebic meningoencephalitis (PAM) has a 97 % fatality rate.
    The model incorporates a temperature-dependent logistic prior centered
    at 30 deg-C, reflecting the strong biological constraint on Naegleria
    proliferation.

    Output dimension: 2 = 2 horizons.

    Parameters
    ----------
    embedding_dim, temporal_dim, hidden_dim, encoder_out_dim, dropout,
    mc_samples : see ``DiseaseForecaster``.
    temp_center_c : float
        Centre of the temperature logistic prior (default 30.0 deg-C).
    temp_scale : float
        Steepness of the temperature logistic (default 2.0; higher = sharper).
    """

    TEMP_THRESHOLD_C: float = NAEGLERIA_TEMP_THRESHOLD_C

    def __init__(
        self,
        embedding_dim: int = 256,
        temporal_dim: int = TEMPORAL_DIM,
        hidden_dim: int = 256,
        encoder_out_dim: int = 128,
        dropout: float = 0.2,
        mc_samples: int = 20,
        temp_center_c: float = 30.0,
        temp_scale: float = 2.0,
    ) -> None:
        super().__init__(
            embedding_dim=embedding_dim,
            temporal_dim=temporal_dim,
            hidden_dim=hidden_dim,
            encoder_out_dim=encoder_out_dim,
            dropout=dropout,
            mc_samples=mc_samples,
        )
        self.temp_center_c = temp_center_c
        self.temp_scale = temp_scale

        # Covariate encoder for water temperature + chlorine residual
        # Input: (temperature_C, chlorine_mg_L)
        self.covariate_encoder = nn.Sequential(
            nn.Linear(2, 16),
            nn.GELU(),
            nn.Linear(16, 16),
        )

        head_input_dim = encoder_out_dim + 16

        # Habitat suitability head
        self.habitat_head = nn.Sequential(
            nn.Linear(head_input_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, NUM_HORIZONS),
        )

        self._init_weights()

    # ------------------------------------------------------------------
    def _temperature_prior(self, temperature_c: torch.Tensor) -> torch.Tensor:
        """Logistic prior: sigma((T - T_center) / scale).

        Encodes the biological fact that Naegleria fowleri proliferates
        almost exclusively at water temperatures > 30 deg-C.

        Parameters
        ----------
        temperature_c : Tensor[B]
            Water temperature in Celsius.

        Returns
        -------
        Tensor[B]
            Prior probability in [0, 1].
        """
        return torch.sigmoid(
            (temperature_c - self.temp_center_c) / self.temp_scale
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        embedding: torch.Tensor,
        temporal: torch.Tensor,
        covariates: torch.Tensor,
        use_mc_dropout: bool = False,
    ) -> NaegleriaOutput:
        """Forward pass.

        Parameters
        ----------
        embedding : Tensor[B, 256]
        temporal : Tensor[B, temporal_dim]
        covariates : Tensor[B, 2]
            Column 0: water temperature (Celsius).
            Column 1: chlorine residual (mg/L).
        use_mc_dropout : bool

        Returns
        -------
        NaegleriaOutput
        """
        temperature_c = covariates[:, 0]
        temp_prior = self._temperature_prior(temperature_c)  # (B,)

        def _single_pass() -> torch.Tensor:
            h_env = self._encode(embedding, temporal)
            h_cov = self.covariate_encoder(covariates)  # (B, 16)
            h = torch.cat([h_env, h_cov], dim=-1)
            logits = self.habitat_head(h)  # (B, 2)
            # Multiply by temperature prior to enforce biological constraint
            prob = torch.sigmoid(logits) * temp_prior.unsqueeze(-1)
            return prob

        if not use_mc_dropout or self.training:
            prob = _single_pass()
            return NaegleriaOutput(habitat_probability=prob)

        # --- MC-Dropout inference -----------------------------------------
        self._enable_dropout()
        mc_probs: List[torch.Tensor] = []
        for _ in range(self.mc_samples):
            mc_probs.append(_single_pass())
        self._disable_dropout()

        mc_t = torch.stack(mc_probs, dim=0)  # (N, B, 2)
        return NaegleriaOutput(
            habitat_probability=mc_t.mean(dim=0),
            mc_prob_mean=mc_t.mean(dim=0),
            mc_prob_std=mc_t.std(dim=0),
        )

    # ------------------------------------------------------------------
    @staticmethod
    def compute_loss(
        output: NaegleriaOutput,
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Binary cross-entropy loss for habitat suitability.

        Parameters
        ----------
        output : NaegleriaOutput
        targets : dict
            ``"habitat"`` : Tensor[B, 2] -- binary (0/1).
        """
        losses: Dict[str, torch.Tensor] = {}
        device = output.habitat_probability.device

        if "habitat" in targets:
            pred_hab = torch.nan_to_num(output.habitat_probability, nan=0.5).clamp(1e-7, 1 - 1e-7)
            tgt_hab = torch.nan_to_num(targets["habitat"].float(), nan=0.0).clamp(0, 1)
            losses["habitat_bce"] = F.binary_cross_entropy(pred_hab, tgt_hab)

        total = torch.tensor(0.0, device=device)
        for val in losses.values():
            total = total + val
        return total, losses


# ============================================================================
# 4. Schistosomiasis snail-host forecaster
# ============================================================================

class SchistosomiasisForecaster(DiseaseForecaster):
    """Predict schistosomiasis intermediate-host (snail) habitat suitability.

    Models habitat suitability for Biomphalaria and Bulinus freshwater
    snails in tropical / subtropical environments.  Key drivers include
    latitude, water temperature, and surrounding vegetation cover.

    Output dimension: 2 = 2 horizons.

    Parameters
    ----------
    embedding_dim, temporal_dim, hidden_dim, encoder_out_dim, dropout,
    mc_samples : see ``DiseaseForecaster``.
    covariate_dim : int
        Extra covariates (default 3: water temperature, latitude,
        NDVI vegetation index).
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        temporal_dim: int = TEMPORAL_DIM,
        hidden_dim: int = 256,
        encoder_out_dim: int = 128,
        dropout: float = 0.2,
        mc_samples: int = 20,
        covariate_dim: int = 3,
    ) -> None:
        super().__init__(
            embedding_dim=embedding_dim,
            temporal_dim=temporal_dim,
            hidden_dim=hidden_dim,
            encoder_out_dim=encoder_out_dim,
            dropout=dropout,
            mc_samples=mc_samples,
        )
        self.covariate_dim = covariate_dim

        # Covariate encoder: water_temp, latitude, NDVI
        self.covariate_encoder = nn.Sequential(
            nn.Linear(covariate_dim, 32),
            nn.GELU(),
            nn.Linear(32, 32),
        )

        head_input_dim = encoder_out_dim + 32

        # Habitat suitability head
        self.habitat_head = nn.Sequential(
            nn.Linear(head_input_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, NUM_HORIZONS),
            nn.Sigmoid(),
        )

        self._init_weights()

    # ------------------------------------------------------------------
    def forward(
        self,
        embedding: torch.Tensor,
        temporal: torch.Tensor,
        covariates: torch.Tensor,
        use_mc_dropout: bool = False,
    ) -> SchistosomiasisOutput:
        """Forward pass.

        Parameters
        ----------
        embedding : Tensor[B, 256]
        temporal : Tensor[B, temporal_dim]
        covariates : Tensor[B, 3]
            Column 0: water temperature (Celsius).
            Column 1: latitude (degrees, -90 to 90).
            Column 2: NDVI vegetation index (0 to 1).
        use_mc_dropout : bool

        Returns
        -------
        SchistosomiasisOutput
        """

        def _single_pass() -> torch.Tensor:
            h_env = self._encode(embedding, temporal)
            h_cov = self.covariate_encoder(covariates)  # (B, 32)
            h = torch.cat([h_env, h_cov], dim=-1)
            return self.habitat_head(h)  # (B, 2)

        if not use_mc_dropout or self.training:
            prob = _single_pass()
            return SchistosomiasisOutput(habitat_probability=prob)

        # --- MC-Dropout inference -----------------------------------------
        self._enable_dropout()
        mc_probs: List[torch.Tensor] = []
        for _ in range(self.mc_samples):
            mc_probs.append(_single_pass())
        self._disable_dropout()

        mc_t = torch.stack(mc_probs, dim=0)  # (N, B, 2)
        return SchistosomiasisOutput(
            habitat_probability=mc_t.mean(dim=0),
            mc_prob_mean=mc_t.mean(dim=0),
            mc_prob_std=mc_t.std(dim=0),
        )

    # ------------------------------------------------------------------
    @staticmethod
    def compute_loss(
        output: SchistosomiasisOutput,
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Binary cross-entropy loss for snail habitat suitability.

        Parameters
        ----------
        output : SchistosomiasisOutput
        targets : dict
            ``"habitat"`` : Tensor[B, 2] -- binary (0/1).
        """
        losses: Dict[str, torch.Tensor] = {}
        device = output.habitat_probability.device

        if "habitat" in targets:
            pred_hab = torch.nan_to_num(output.habitat_probability, nan=0.5).clamp(1e-7, 1 - 1e-7)
            tgt_hab = torch.nan_to_num(targets["habitat"].float(), nan=0.0).clamp(0, 1)
            losses["habitat_bce"] = F.binary_cross_entropy(pred_hab, tgt_hab)

        total = torch.tensor(0.0, device=device)
        for val in losses.values():
            total = total + val
        return total, losses


# ============================================================================
# 5. Integrated disease risk
# ============================================================================

class IntegratedDiseaseRisk(nn.Module):
    """Combines all four disease forecasters into a single module.

    Produces a comprehensive per-site risk assessment and generates
    alert levels (LOW / MODERATE / HIGH / CRITICAL) based on thresholds.

    Parameters
    ----------
    embedding_dim : int
        SENTINEL embedding dimension (default 256).
    dropout : float
        Shared dropout rate (default 0.2).
    mc_samples : int
        MC-dropout passes (default 20).
    """

    # Alert thresholds -------------------------------------------------
    # Cyanotoxin: any toxin above WHO drinking-water threshold
    # Vibrio: risk index above 0.6
    # Naegleria: habitat probability above 0.5
    # Schistosomiasis: habitat probability above 0.5

    ALERT_THRESHOLDS: Dict[str, Dict[str, float]] = {
        "cyanotoxin_moderate": {"exceedance_prob": 0.3},
        "cyanotoxin_high": {"exceedance_prob": 0.6},
        "cyanotoxin_critical": {"exceedance_prob": 0.85},
        "vibrio_moderate": {"risk_index": 0.3},
        "vibrio_high": {"risk_index": 0.6},
        "vibrio_critical": {"risk_index": 0.85},
        "naegleria_moderate": {"habitat_prob": 0.25},
        "naegleria_high": {"habitat_prob": 0.5},
        "naegleria_critical": {"habitat_prob": 0.75},
        "schisto_moderate": {"habitat_prob": 0.3},
        "schisto_high": {"habitat_prob": 0.6},
        "schisto_critical": {"habitat_prob": 0.85},
    }

    def __init__(
        self,
        embedding_dim: int = 256,
        dropout: float = 0.2,
        mc_samples: int = 20,
    ) -> None:
        super().__init__()

        shared_kwargs: Dict[str, Any] = dict(
            embedding_dim=embedding_dim,
            dropout=dropout,
            mc_samples=mc_samples,
        )

        self.cyanotoxin = CyanotoxinForecaster(**shared_kwargs)
        self.vibrio = VibrioRiskForecaster(**shared_kwargs)
        self.naegleria = NaegleriaForecaster(**shared_kwargs)
        self.schistosomiasis = SchistosomiasisForecaster(**shared_kwargs)

    # ------------------------------------------------------------------
    def forward(
        self,
        embedding: torch.Tensor,
        day_of_year: torch.Tensor,
        vibrio_covariates: Optional[torch.Tensor] = None,
        naegleria_covariates: Optional[torch.Tensor] = None,
        schisto_covariates: Optional[torch.Tensor] = None,
        use_mc_dropout: bool = False,
    ) -> DiseaseRiskSummary:
        """Run all disease forecasters and compute alert levels.

        Parameters
        ----------
        embedding : Tensor[B, 256]
            SENTINEL environmental embedding.
        day_of_year : Tensor[B]
            Day of year (1-366) for temporal encoding.
        vibrio_covariates : Tensor[B, 2], optional
            (water_temp_C, salinity_ppt).  If None, zeros are used.
        naegleria_covariates : Tensor[B, 2], optional
            (water_temp_C, chlorine_mg_L).  If None, zeros are used.
        schisto_covariates : Tensor[B, 3], optional
            (water_temp_C, latitude_deg, ndvi).  If None, zeros are used.
        use_mc_dropout : bool
            If True, run MC-dropout for uncertainty estimation.

        Returns
        -------
        DiseaseRiskSummary
        """
        B = embedding.shape[0]
        device = embedding.device

        # Build temporal context
        temporal = encode_temporal_context(day_of_year)

        # Default covariates (zeros) when not provided
        if vibrio_covariates is None:
            vibrio_covariates = torch.zeros(B, 2, device=device)
        if naegleria_covariates is None:
            naegleria_covariates = torch.zeros(B, 2, device=device)
        if schisto_covariates is None:
            schisto_covariates = torch.zeros(B, 3, device=device)

        # Run all forecasters
        cyano_out = self.cyanotoxin(embedding, temporal, use_mc_dropout=use_mc_dropout)
        vibrio_out = self.vibrio(embedding, temporal, vibrio_covariates, use_mc_dropout=use_mc_dropout)
        naeg_out = self.naegleria(embedding, temporal, naegleria_covariates, use_mc_dropout=use_mc_dropout)
        schisto_out = self.schistosomiasis(embedding, temporal, schisto_covariates, use_mc_dropout=use_mc_dropout)

        # Compute alert levels
        alert_level, alert_reasons = self._compute_alerts(
            cyano_out, vibrio_out, naeg_out, schisto_out,
        )

        return DiseaseRiskSummary(
            cyanotoxin=cyano_out,
            vibrio=vibrio_out,
            naegleria=naeg_out,
            schistosomiasis=schisto_out,
            alert_level=alert_level,
            alert_reasons=alert_reasons,
        )

    # ------------------------------------------------------------------
    def _compute_alerts(
        self,
        cyano: CyanotoxinOutput,
        vibrio: VibrioOutput,
        naegleria: NaegleriaOutput,
        schisto: SchistosomiasisOutput,
    ) -> Tuple[torch.Tensor, List[List[str]]]:
        """Determine per-site alert levels from forecaster outputs.

        The overall alert level for a site is the maximum across all
        disease categories.

        Returns
        -------
        alert_levels : Tensor[B]   (int values 0-3 mapping to AlertLevel)
        alert_reasons : List[List[str]]   per-site list of reason strings
        """
        B = cyano.log_concentration.shape[0]
        device = cyano.log_concentration.device

        site_levels = torch.zeros(B, dtype=torch.long, device=device)
        all_reasons: List[List[str]] = [[] for _ in range(B)]

        # --- Cyanotoxin alerts --------------------------------------------
        # Use the maximum drinking-water exceedance probability across
        # all toxins and horizons as the trigger metric.
        max_drink_exc = cyano.drinking_exceedance_prob.max(dim=-1).values  # (B,)

        crit = max_drink_exc >= self.ALERT_THRESHOLDS["cyanotoxin_critical"]["exceedance_prob"]
        high = (~crit) & (max_drink_exc >= self.ALERT_THRESHOLDS["cyanotoxin_high"]["exceedance_prob"])
        mod = (~crit) & (~high) & (max_drink_exc >= self.ALERT_THRESHOLDS["cyanotoxin_moderate"]["exceedance_prob"])

        for i in range(B):
            if crit[i]:
                site_levels[i] = max(site_levels[i].item(), AlertLevel.CRITICAL)
                all_reasons[i].append("Cyanotoxin: CRITICAL -- high exceedance probability")
            elif high[i]:
                site_levels[i] = max(site_levels[i].item(), AlertLevel.HIGH)
                all_reasons[i].append("Cyanotoxin: HIGH -- elevated exceedance probability")
            elif mod[i]:
                site_levels[i] = max(site_levels[i].item(), AlertLevel.MODERATE)
                all_reasons[i].append("Cyanotoxin: MODERATE -- possible exceedance")

        # --- Vibrio alerts ------------------------------------------------
        max_vibrio_risk = vibrio.risk_index.max(dim=-1).values  # (B,)

        crit = max_vibrio_risk >= self.ALERT_THRESHOLDS["vibrio_critical"]["risk_index"]
        high = (~crit) & (max_vibrio_risk >= self.ALERT_THRESHOLDS["vibrio_high"]["risk_index"])
        mod = (~crit) & (~high) & (max_vibrio_risk >= self.ALERT_THRESHOLDS["vibrio_moderate"]["risk_index"])

        for i in range(B):
            if crit[i]:
                site_levels[i] = max(site_levels[i].item(), AlertLevel.CRITICAL)
                all_reasons[i].append("Vibrio: CRITICAL -- very high risk index")
            elif high[i]:
                site_levels[i] = max(site_levels[i].item(), AlertLevel.HIGH)
                all_reasons[i].append("Vibrio: HIGH -- elevated risk index")
            elif mod[i]:
                site_levels[i] = max(site_levels[i].item(), AlertLevel.MODERATE)
                all_reasons[i].append("Vibrio: MODERATE -- moderate risk index")

        # --- Naegleria alerts ---------------------------------------------
        max_naeg = naegleria.habitat_probability.max(dim=-1).values  # (B,)

        crit = max_naeg >= self.ALERT_THRESHOLDS["naegleria_critical"]["habitat_prob"]
        high = (~crit) & (max_naeg >= self.ALERT_THRESHOLDS["naegleria_high"]["habitat_prob"])
        mod = (~crit) & (~high) & (max_naeg >= self.ALERT_THRESHOLDS["naegleria_moderate"]["habitat_prob"])

        for i in range(B):
            if crit[i]:
                site_levels[i] = max(site_levels[i].item(), AlertLevel.CRITICAL)
                all_reasons[i].append("Naegleria: CRITICAL -- high habitat suitability (97% fatality)")
            elif high[i]:
                site_levels[i] = max(site_levels[i].item(), AlertLevel.HIGH)
                all_reasons[i].append("Naegleria: HIGH -- elevated habitat suitability")
            elif mod[i]:
                site_levels[i] = max(site_levels[i].item(), AlertLevel.MODERATE)
                all_reasons[i].append("Naegleria: MODERATE -- possible habitat suitability")

        # --- Schistosomiasis alerts ---------------------------------------
        max_schisto = schisto.habitat_probability.max(dim=-1).values  # (B,)

        crit = max_schisto >= self.ALERT_THRESHOLDS["schisto_critical"]["habitat_prob"]
        high = (~crit) & (max_schisto >= self.ALERT_THRESHOLDS["schisto_high"]["habitat_prob"])
        mod = (~crit) & (~high) & (max_schisto >= self.ALERT_THRESHOLDS["schisto_moderate"]["habitat_prob"])

        for i in range(B):
            if crit[i]:
                site_levels[i] = max(site_levels[i].item(), AlertLevel.CRITICAL)
                all_reasons[i].append("Schistosomiasis: CRITICAL -- high snail habitat suitability")
            elif high[i]:
                site_levels[i] = max(site_levels[i].item(), AlertLevel.HIGH)
                all_reasons[i].append("Schistosomiasis: HIGH -- elevated snail habitat suitability")
            elif mod[i]:
                site_levels[i] = max(site_levels[i].item(), AlertLevel.MODERATE)
                all_reasons[i].append("Schistosomiasis: MODERATE -- possible snail habitat")

        # Set LOW for sites with no triggers
        for i in range(B):
            if not all_reasons[i]:
                all_reasons[i].append("All disease risks LOW")

        return site_levels, all_reasons

    # ------------------------------------------------------------------
    def compute_loss(
        self,
        summary: DiseaseRiskSummary,
        targets: Dict[str, Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Combined multi-task loss across all disease forecasters.

        Parameters
        ----------
        summary : DiseaseRiskSummary
        targets : dict of dicts
            Outer keys: ``"cyanotoxin"``, ``"vibrio"``, ``"naegleria"``,
            ``"schistosomiasis"``.  Inner keys are forecaster-specific
            (see individual ``compute_loss`` methods).

        Returns
        -------
        total_loss, per_disease_losses
        """
        all_losses: Dict[str, torch.Tensor] = {}
        device = summary.alert_level.device
        total = torch.tensor(0.0, device=device)

        if "cyanotoxin" in targets:
            loss, sub = CyanotoxinForecaster.compute_loss(
                summary.cyanotoxin, targets["cyanotoxin"],
            )
            all_losses["cyanotoxin"] = loss
            total = total + loss

        if "vibrio" in targets:
            loss, sub = VibrioRiskForecaster.compute_loss(
                summary.vibrio, targets["vibrio"],
            )
            all_losses["vibrio"] = loss
            total = total + loss

        if "naegleria" in targets:
            loss, sub = NaegleriaForecaster.compute_loss(
                summary.naegleria, targets["naegleria"],
            )
            # Up-weight Naegleria due to extreme fatality rate
            all_losses["naegleria"] = loss
            total = total + 2.0 * loss

        if "schistosomiasis" in targets:
            loss, sub = SchistosomiasisForecaster.compute_loss(
                summary.schistosomiasis, targets["schistosomiasis"],
            )
            all_losses["schistosomiasis"] = loss
            total = total + loss

        return total, all_losses
