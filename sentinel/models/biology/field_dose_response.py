"""Lab-to-Field Dose-Response Extrapolation (Phase 3.6).

Bridges the EPA ECOTOX database (~268K lab toxicity records) to field
observations by learning a mapping from controlled lab dose-response
curves to predicted effects under realistic environmental conditions.

The core challenge: lab LC50/EC50 values are measured under controlled
conditions (fixed temperature, pH, dissolved oxygen, single chemicals)
that rarely match field reality.  This module learns to extrapolate
from those lab values to predicted field effects given the actual
environmental conditions captured by SENTINEL embeddings.

Architecture overview
---------------------

**FieldDoseResponseModel**::

    SENTINEL embedding (B, 256)
    + contaminant concentrations from USGS (B, C_chem)
    + species identity embedding (B, 64)
        --> EnvironmentalContextEncoder (256 + C_chem -> 256)
        --> DoseResponseCurvePredictor
            --> 4-parameter log-logistic curve parameters (EC50, slope, min, max)
            --> species sensitivity distribution (SSD) parameters
        --> BenchmarkPredictor
            --> HC5 (hazardous concentration for 5% of species)
            --> margin of safety
        --> MC-dropout uncertainty (20 stochastic passes)

Species Sensitivity Distribution (SSD):
    Assumes log-normal distribution of species sensitivities.
    Given predicted effect concentrations for multiple species,
    fits log(EC50) ~ N(mu, sigma) and derives HC5 as the 5th
    percentile of the fitted distribution.

USGS contaminant mapping:
    Takes real-time contaminant concentrations from USGS monitoring
    sites (pesticides, metals, nutrients, pharmaceuticals) and computes
    the implied exposure relative to species-specific effect thresholds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Constants
# ============================================================================

SHARED_EMBEDDING_DIM: int = 256
SPECIES_EMBEDDING_DIM: int = 64
MC_DROPOUT_SAMPLES: int = 20
DROPOUT_P: float = 0.2

# Number of USGS-monitored contaminant classes
NUM_CONTAMINANT_CLASSES: int = 24

# Contaminant categories tracked by USGS
CONTAMINANT_CATEGORIES: Tuple[str, ...] = (
    "atrazine",
    "metolachlor",
    "glyphosate",
    "chlorpyrifos",
    "diazinon",
    "malathion",
    "carbaryl",
    "imidacloprid",
    "fipronil",
    "bifenthrin",
    "arsenic",
    "cadmium",
    "chromium",
    "copper",
    "lead",
    "mercury",
    "zinc",
    "nitrate",
    "phosphate",
    "ammonia",
    "acetaminophen",
    "ibuprofen",
    "estradiol_17b",
    "ethinylestradiol",
)

CONTAMINANT_INDEX: Dict[str, int] = {
    name: i for i, name in enumerate(CONTAMINANT_CATEGORIES)
}

# Number of species with ECOTOX lab data
NUM_ECOTOX_SPECIES: int = 150

# Dose-response curve parameters
NUM_DR_PARAMS: int = 4  # EC50, slope (Hill), min_effect, max_effect


# ============================================================================
# Risk thresholds
# ============================================================================

class ExposureRiskLevel(IntEnum):
    """Risk level based on exposure relative to effect thresholds."""
    NEGLIGIBLE = 0    # exposure/EC50 < 0.01
    LOW = 1           # 0.01 <= ratio < 0.1
    MODERATE = 2      # 0.1 <= ratio < 0.5
    HIGH = 3          # 0.5 <= ratio < 1.0
    CRITICAL = 4      # ratio >= 1.0 (exposure exceeds effect concentration)


EXPOSURE_RISK_THRESHOLDS: Dict[str, float] = {
    "negligible": 0.01,
    "low": 0.1,
    "moderate": 0.5,
    "high": 1.0,
}

# SSD percentile for regulatory benchmark
HC5_PERCENTILE: float = 0.05  # 5th percentile of species sensitivity
HC5_CONFIDENCE: float = 0.95  # 95% confidence interval


# ============================================================================
# Output dataclasses
# ============================================================================

@dataclass
class DoseResponseCurveParams:
    """Predicted dose-response curve parameters.

    The 4-parameter log-logistic model:
        effect(c) = min_effect + (max_effect - min_effect) /
                    (1 + (c / EC50)^(-slope))

    Attributes
    ----------
    log_ec50 : Tensor[B, S]
        Log10 of the predicted effect concentration (ug/L).
    slope : Tensor[B, S]
        Hill slope (steepness of the dose-response curve).
    min_effect : Tensor[B, S]
        Minimum effect level (baseline).
    max_effect : Tensor[B, S]
        Maximum effect level (full response).
    """

    log_ec50: torch.Tensor
    slope: torch.Tensor
    min_effect: torch.Tensor
    max_effect: torch.Tensor


@dataclass
class SSDOutput:
    """Species Sensitivity Distribution output.

    Attributes
    ----------
    log_ec50_mean : Tensor[B, C]
        Mean of log(EC50) distribution per contaminant.
    log_ec50_std : Tensor[B, C]
        Std of log(EC50) distribution per contaminant.
    hc5 : Tensor[B, C]
        Hazardous concentration for 5% of species (log10 ug/L).
    hc5_lower : Tensor[B, C]
        Lower 95% CI on HC5.
    hc5_upper : Tensor[B, C]
        Upper 95% CI on HC5.
    """

    log_ec50_mean: torch.Tensor
    log_ec50_std: torch.Tensor
    hc5: torch.Tensor
    hc5_lower: torch.Tensor
    hc5_upper: torch.Tensor


@dataclass
class FieldDoseResponseOutput:
    """Complete output of the FieldDoseResponseModel.

    Attributes
    ----------
    curve_params : DoseResponseCurveParams
        Predicted dose-response curve parameters per species.
    implied_exposure : Tensor[B, S, C]
        Implied exposure ratio (measured conc / predicted EC50)
        for each species and contaminant.
    risk_levels : Tensor[B, S, C]
        Integer risk level per species per contaminant.
    ssd : SSDOutput
        Species sensitivity distribution parameters.
    margin_of_safety : Tensor[B, C]
        Ratio of HC5 to measured environmental concentration.
    mc_log_ec50_mean : Tensor[B, S] or None
        MC-dropout mean of log EC50.
    mc_log_ec50_std : Tensor[B, S] or None
        MC-dropout std of log EC50.
    mc_hc5_mean : Tensor[B, C] or None
        MC-dropout mean of HC5.
    mc_hc5_std : Tensor[B, C] or None
        MC-dropout std of HC5.
    """

    curve_params: DoseResponseCurveParams
    implied_exposure: torch.Tensor
    risk_levels: torch.Tensor
    ssd: SSDOutput
    margin_of_safety: torch.Tensor

    mc_log_ec50_mean: Optional[torch.Tensor] = None
    mc_log_ec50_std: Optional[torch.Tensor] = None
    mc_hc5_mean: Optional[torch.Tensor] = None
    mc_hc5_std: Optional[torch.Tensor] = None


# ============================================================================
# Sub-module 1: Environmental Context Encoder
# ============================================================================

class EnvironmentalContextEncoder(nn.Module):
    """Encode SENTINEL embedding + contaminant concentrations.

    Fuses the multimodal environmental embedding with measured
    contaminant concentrations to create a context vector that
    captures both the general environmental state (temperature,
    pH, DO, etc.) and the specific chemical exposure profile.

    Parameters
    ----------
    embedding_dim : int
        SENTINEL embedding dimension (default 256).
    num_contaminants : int
        Number of contaminant concentration features (default 24).
    output_dim : int
        Output representation dimension (default 256).
    dropout : float
        Dropout probability (default 0.2).
    """

    def __init__(
        self,
        embedding_dim: int = SHARED_EMBEDDING_DIM,
        num_contaminants: int = NUM_CONTAMINANT_CLASSES,
        output_dim: int = 256,
        dropout: float = DROPOUT_P,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_contaminants = num_contaminants

        # Log-transform and project contaminant concentrations
        self.contaminant_projection = nn.Sequential(
            nn.Linear(num_contaminants, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
        )

        # Fuse embedding + projected contaminants
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim + 64, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        sentinel_embedding: torch.Tensor,
        contaminant_conc: torch.Tensor,
    ) -> torch.Tensor:
        """Encode environmental context.

        Parameters
        ----------
        sentinel_embedding : Tensor[B, 256]
            SENTINEL environmental embedding.
        contaminant_conc : Tensor[B, C_chem]
            Log10-transformed contaminant concentrations from USGS.

        Returns
        -------
        Tensor[B, 256]
            Fused environmental + chemical context.
        """
        chem_repr = self.contaminant_projection(contaminant_conc)  # [B, 64]
        combined = torch.cat([sentinel_embedding, chem_repr], dim=-1)  # [B, 320]
        return self.fusion(combined)  # [B, 256]


# ============================================================================
# Sub-module 2: Species Embedding for Dose-Response
# ============================================================================

class ECOTOXSpeciesEmbedding(nn.Module):
    """Learnable embeddings for species with ECOTOX lab data.

    Encodes species identity with an embedding that captures
    taxonomic traits, body size, metabolic rate, and known
    sensitivity patterns from the ECOTOX database.

    Parameters
    ----------
    num_species : int
        Number of species in the ECOTOX database (default 150).
    embedding_dim : int
        Embedding dimensionality (default 64).
    """

    def __init__(
        self,
        num_species: int = NUM_ECOTOX_SPECIES,
        embedding_dim: int = SPECIES_EMBEDDING_DIM,
    ) -> None:
        super().__init__()
        self.num_species = num_species
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(num_species, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)

        # Traits encoder: optional side information (body mass, trophic level, etc.)
        self.traits_encoder = nn.Sequential(
            nn.Linear(8, 32),
            nn.GELU(),
            nn.Linear(32, embedding_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        for module in self.traits_encoder.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        species_idx: Optional[torch.Tensor] = None,
        species_traits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return species embeddings, optionally augmented with traits.

        Parameters
        ----------
        species_idx : Tensor[...], optional
            Long tensor of species indices.  If None, returns the
            full table [S, D].
        species_traits : Tensor[..., 8], optional
            Species trait features (body mass, trophic level, etc.).
            If provided, added to the learnable embedding.

        Returns
        -------
        Tensor[..., D]
        """
        if species_idx is None:
            species_idx = torch.arange(
                self.num_species,
                device=self.embedding.weight.device,
            )

        emb = self.embedding(species_idx)

        if species_traits is not None:
            traits_emb = self.traits_encoder(species_traits)
            emb = emb + traits_emb

        return self.layer_norm(emb)


# ============================================================================
# Sub-module 3: Dose-Response Curve Predictor
# ============================================================================

class DoseResponseCurvePredictor(nn.Module):
    """Predict 4-parameter log-logistic dose-response curves.

    For each species, predicts the parameters of a log-logistic
    dose-response curve that describes the relationship between
    contaminant concentration and biological effect under the
    actual field conditions captured by the SENTINEL embedding.

    The key insight is that lab-derived EC50 values are adjusted
    by the environmental context: temperature, pH, dissolved
    organic carbon, and other water quality parameters all modify
    bioavailability and toxicity.

    Parameters
    ----------
    env_dim : int
        Environmental context dimension (default 256).
    species_dim : int
        Species embedding dimension (default 64).
    num_contaminants : int
        Number of contaminant classes (default 24).
    hidden_dim : int
        Hidden layer size (default 256).
    dropout : float
        Dropout probability (default 0.2).
    """

    def __init__(
        self,
        env_dim: int = 256,
        species_dim: int = SPECIES_EMBEDDING_DIM,
        num_contaminants: int = NUM_CONTAMINANT_CLASSES,
        hidden_dim: int = 256,
        dropout: float = DROPOUT_P,
    ) -> None:
        super().__init__()
        self.env_dim = env_dim
        self.num_contaminants = num_contaminants

        combined_dim = env_dim + species_dim  # 320

        # Shared feature extractor
        self.backbone = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )

        # Per-contaminant heads that predict 4 DR curve parameters
        self.dr_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(64, NUM_DR_PARAMS),
            )
            for _ in range(num_contaminants)
        ])

        # Lab-to-field adjustment factors (learnable bias per contaminant)
        # Initialized near 1.0 (no adjustment) and learned from data
        self.lab_field_adjustment = nn.Parameter(
            torch.zeros(num_contaminants, NUM_DR_PARAMS)
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        env_context: torch.Tensor,
        species_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict dose-response curve parameters.

        Parameters
        ----------
        env_context : Tensor[B, env_dim]
            Environmental context embedding.
        species_emb : Tensor[B, species_dim]
            Species embedding for the target species.

        Returns
        -------
        log_ec50 : Tensor[B, C]
            Log10 EC50 per contaminant.
        slope : Tensor[B, C]
            Hill slope per contaminant.
        min_effect : Tensor[B, C]
            Minimum effect per contaminant.
        max_effect : Tensor[B, C]
            Maximum effect per contaminant.
        """
        combined = torch.cat([env_context, species_emb], dim=-1)
        backbone_out = self.backbone(combined)  # [B, 256]

        log_ec50_list: List[torch.Tensor] = []
        slope_list: List[torch.Tensor] = []
        min_eff_list: List[torch.Tensor] = []
        max_eff_list: List[torch.Tensor] = []

        for c in range(self.num_contaminants):
            params = self.dr_heads[c](backbone_out)  # [B, 4]
            params = params + self.lab_field_adjustment[c].unsqueeze(0)

            log_ec50_list.append(params[:, 0])
            slope_list.append(F.softplus(params[:, 1]) + 0.1)  # slope > 0
            min_eff_list.append(torch.sigmoid(params[:, 2]))     # in [0, 1]
            max_eff_list.append(torch.sigmoid(params[:, 3]))     # in [0, 1]

        log_ec50 = torch.stack(log_ec50_list, dim=-1)    # [B, C]
        slope = torch.stack(slope_list, dim=-1)           # [B, C]
        min_effect = torch.stack(min_eff_list, dim=-1)   # [B, C]
        max_effect = torch.stack(max_eff_list, dim=-1)   # [B, C]

        return log_ec50, slope, min_effect, max_effect


# ============================================================================
# Sub-module 4: Species Sensitivity Distribution
# ============================================================================

class SpeciesSensitivityDistribution(nn.Module):
    """Fit log-normal species sensitivity distribution and derive HC5.

    Given predicted EC50 values for multiple species, fits a
    log-normal distribution and computes the 5th percentile
    (HC5), which is the standard regulatory benchmark for
    environmental risk assessment.

    The SSD assumes log(EC50) ~ N(mu, sigma^2) across species.
    HC5 = 10^(mu - 1.645 * sigma) for the 5th percentile.

    Confidence intervals on HC5 are estimated via the delta
    method applied to the normal distribution parameters.

    Parameters
    ----------
    num_contaminants : int
        Number of contaminant classes (default 24).
    """

    def __init__(
        self,
        num_contaminants: int = NUM_CONTAMINANT_CLASSES,
    ) -> None:
        super().__init__()
        self.num_contaminants = num_contaminants

        # Z-score for the 5th percentile of the standard normal
        self.register_buffer(
            "z_hc5",
            torch.tensor(-1.6449),  # norm.ppf(0.05)
        )

        # Z-score for 95% confidence interval
        self.register_buffer(
            "z_ci",
            torch.tensor(1.96),  # norm.ppf(0.975)
        )

    def forward(
        self,
        log_ec50: torch.Tensor,
    ) -> SSDOutput:
        """Fit SSD and compute HC5 with confidence intervals.

        Parameters
        ----------
        log_ec50 : Tensor[B, S, C]
            Log10 EC50 predictions for S species and C contaminants.

        Returns
        -------
        SSDOutput
            Contains SSD parameters and HC5 with confidence intervals.
        """
        B, S, C = log_ec50.shape

        # Fit log-normal: MLE estimates of mu and sigma
        mu = log_ec50.mean(dim=1)      # [B, C]
        sigma = log_ec50.std(dim=1)     # [B, C]
        sigma = sigma.clamp(min=1e-6)   # numerical stability

        # HC5: 5th percentile of the log-normal distribution
        # HC5 = mu + z_005 * sigma  (in log10 space)
        hc5 = mu + self.z_hc5 * sigma  # [B, C]

        # Confidence intervals via delta method
        # SE(HC5) = sqrt(1/n + z^2/(2*(n-1))) * sigma
        n = float(S)
        se_factor = math.sqrt(1.0 / n + (self.z_hc5.item() ** 2) / (2.0 * (n - 1)))
        hc5_se = se_factor * sigma  # [B, C]

        hc5_lower = hc5 - self.z_ci * hc5_se
        hc5_upper = hc5 + self.z_ci * hc5_se

        return SSDOutput(
            log_ec50_mean=mu,
            log_ec50_std=sigma,
            hc5=hc5,
            hc5_lower=hc5_lower,
            hc5_upper=hc5_upper,
        )


# ============================================================================
# Sub-module 5: Implied Exposure Calculator
# ============================================================================

class ImpliedExposureCalculator(nn.Module):
    """Compute implied exposure ratios from measured concentrations.

    For each contaminant, the implied exposure is the ratio of the
    measured environmental concentration to the predicted EC50 for
    each species.  Ratios > 1.0 indicate that the measured
    concentration exceeds the predicted effect concentration.

    Parameters
    ----------
    num_contaminants : int
        Number of contaminant classes (default 24).
    """

    def __init__(
        self,
        num_contaminants: int = NUM_CONTAMINANT_CLASSES,
    ) -> None:
        super().__init__()
        self.num_contaminants = num_contaminants

    def forward(
        self,
        log_ec50: torch.Tensor,
        log_contaminant_conc: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute implied exposure ratios and risk levels.

        Parameters
        ----------
        log_ec50 : Tensor[B, S, C]
            Log10 predicted EC50 per species per contaminant.
        log_contaminant_conc : Tensor[B, C]
            Log10 measured contaminant concentrations.

        Returns
        -------
        exposure_ratio : Tensor[B, S, C]
            Ratio of measured concentration to EC50.
        risk_levels : Tensor[B, S, C] of long
            Risk level per species per contaminant.
        """
        # Exposure ratio in log space: log(conc/EC50) = log(conc) - log(EC50)
        # Then exponentiate for the actual ratio
        log_ratio = log_contaminant_conc.unsqueeze(1) - log_ec50  # [B, S, C]
        exposure_ratio = torch.pow(10.0, log_ratio)

        # Assign risk levels based on thresholds
        risk_levels = torch.zeros_like(exposure_ratio, dtype=torch.long)
        risk_levels[exposure_ratio >= EXPOSURE_RISK_THRESHOLDS["negligible"]] = (
            ExposureRiskLevel.LOW
        )
        risk_levels[exposure_ratio >= EXPOSURE_RISK_THRESHOLDS["low"]] = (
            ExposureRiskLevel.MODERATE
        )
        risk_levels[exposure_ratio >= EXPOSURE_RISK_THRESHOLDS["moderate"]] = (
            ExposureRiskLevel.HIGH
        )
        risk_levels[exposure_ratio >= EXPOSURE_RISK_THRESHOLDS["high"]] = (
            ExposureRiskLevel.CRITICAL
        )

        return exposure_ratio, risk_levels


# ============================================================================
# Sub-module 6: Benchmark Prediction Head
# ============================================================================

class BenchmarkPredictionHead(nn.Module):
    """Predict regulatory benchmark metrics with uncertainty.

    Combines SSD-derived HC5 with measured concentrations to
    compute the margin of safety (MoS), a key regulatory metric:

        MoS = HC5 / measured_concentration

    MoS > 1: environment is likely safe for most species.
    MoS < 1: risk of adverse effects to sensitive species.

    Parameters
    ----------
    num_contaminants : int
        Number of contaminant classes (default 24).
    hidden_dim : int
        Hidden dimension for the refinement network (default 64).
    dropout : float
        Dropout probability (default 0.2).
    """

    def __init__(
        self,
        num_contaminants: int = NUM_CONTAMINANT_CLASSES,
        hidden_dim: int = 64,
        dropout: float = DROPOUT_P,
    ) -> None:
        super().__init__()
        self.num_contaminants = num_contaminants

        # Learned refinement of the analytical MoS estimate
        # Input: [raw_mos, hc5, hc5_lower, hc5_upper, measured_conc]
        self.mos_refiner = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        ssd_output: SSDOutput,
        log_contaminant_conc: torch.Tensor,
    ) -> torch.Tensor:
        """Compute margin of safety for each contaminant.

        Parameters
        ----------
        ssd_output : SSDOutput
            Species sensitivity distribution parameters.
        log_contaminant_conc : Tensor[B, C]
            Log10 measured contaminant concentrations.

        Returns
        -------
        margin_of_safety : Tensor[B, C]
            Refined margin of safety (log10 scale).
        """
        B, C = log_contaminant_conc.shape

        # Analytical MoS in log space
        raw_log_mos = ssd_output.hc5 - log_contaminant_conc  # [B, C]

        # Stack features for refinement
        features = torch.stack([
            raw_log_mos,
            ssd_output.hc5,
            ssd_output.hc5_lower,
            ssd_output.hc5_upper,
            log_contaminant_conc,
        ], dim=-1)  # [B, C, 5]

        # Refine per-contaminant
        refined = self.mos_refiner(features).squeeze(-1)  # [B, C]

        # Residual connection: learned refinement adjusts analytical estimate
        margin_of_safety = raw_log_mos + refined

        return margin_of_safety


# ============================================================================
# Main Model: FieldDoseResponseModel
# ============================================================================

class FieldDoseResponseModel(nn.Module):
    """Lab-to-field dose-response extrapolation model.

    Bridges 268K ECOTOX lab toxicity records to field observations by
    predicting how dose-response curves shift under real environmental
    conditions.  Takes a SENTINEL embedding + measured contaminant
    concentrations from USGS and predicts implied exposure, species
    sensitivity distributions, and regulatory benchmark metrics.

    Parameters
    ----------
    embedding_dim : int
        SENTINEL embedding dimension (default 256).
    num_contaminants : int
        Number of contaminant classes (default 24).
    num_species : int
        Number of ECOTOX species (default 150).
    species_embedding_dim : int
        Species embedding dimension (default 64).
    dropout : float
        Dropout probability (default 0.2).
    mc_samples : int
        Number of MC-dropout forward passes (default 20).

    Example
    -------
    >>> model = FieldDoseResponseModel()
    >>> embedding = torch.randn(4, 256)
    >>> concentrations = torch.randn(4, 24)
    >>> output = model(embedding, concentrations)
    >>> output.implied_exposure.shape
    torch.Size([4, 150, 24])
    """

    def __init__(
        self,
        embedding_dim: int = SHARED_EMBEDDING_DIM,
        num_contaminants: int = NUM_CONTAMINANT_CLASSES,
        num_species: int = NUM_ECOTOX_SPECIES,
        species_embedding_dim: int = SPECIES_EMBEDDING_DIM,
        dropout: float = DROPOUT_P,
        mc_samples: int = MC_DROPOUT_SAMPLES,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_contaminants = num_contaminants
        self.num_species = num_species
        self.mc_samples = mc_samples
        self.dropout_p = dropout

        # Sub-modules
        self.env_encoder = EnvironmentalContextEncoder(
            embedding_dim=embedding_dim,
            num_contaminants=num_contaminants,
            output_dim=256,
            dropout=dropout,
        )

        self.species_embedding = ECOTOXSpeciesEmbedding(
            num_species=num_species,
            embedding_dim=species_embedding_dim,
        )

        self.dr_predictor = DoseResponseCurvePredictor(
            env_dim=256,
            species_dim=species_embedding_dim,
            num_contaminants=num_contaminants,
            hidden_dim=256,
            dropout=dropout,
        )

        self.ssd = SpeciesSensitivityDistribution(
            num_contaminants=num_contaminants,
        )

        self.exposure_calc = ImpliedExposureCalculator(
            num_contaminants=num_contaminants,
        )

        self.benchmark_head = BenchmarkPredictionHead(
            num_contaminants=num_contaminants,
            hidden_dim=64,
            dropout=dropout,
        )

    # ------------------------------------------------------------------
    # MC-dropout helpers
    # ------------------------------------------------------------------

    def _enable_dropout(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def _disable_dropout(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.eval()

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------

    def _single_forward(
        self,
        sentinel_embedding: torch.Tensor,
        contaminant_conc: torch.Tensor,
    ) -> Tuple[
        DoseResponseCurveParams,
        torch.Tensor,
        torch.Tensor,
        SSDOutput,
        torch.Tensor,
    ]:
        """Single deterministic forward pass.

        Returns
        -------
        curve_params, implied_exposure, risk_levels, ssd_output, margin_of_safety
        """
        B = sentinel_embedding.size(0)
        device = sentinel_embedding.device

        # Encode environmental context
        env_context = self.env_encoder(sentinel_embedding, contaminant_conc)

        # Get species embeddings
        species_emb = self.species_embedding()  # [S, 64]
        S = species_emb.size(0)

        # Predict DR curves for each species
        all_log_ec50: List[torch.Tensor] = []
        all_slope: List[torch.Tensor] = []
        all_min_eff: List[torch.Tensor] = []
        all_max_eff: List[torch.Tensor] = []

        for s in range(S):
            sp_emb = species_emb[s].unsqueeze(0).expand(B, -1)
            log_ec50, slope, min_eff, max_eff = self.dr_predictor(
                env_context, sp_emb,
            )
            all_log_ec50.append(log_ec50)
            all_slope.append(slope)
            all_min_eff.append(min_eff)
            all_max_eff.append(max_eff)

        # Stack: [B, S, C]
        log_ec50_all = torch.stack(all_log_ec50, dim=1)
        slope_all = torch.stack(all_slope, dim=1)
        min_eff_all = torch.stack(all_min_eff, dim=1)
        max_eff_all = torch.stack(all_max_eff, dim=1)

        curve_params = DoseResponseCurveParams(
            log_ec50=log_ec50_all,
            slope=slope_all,
            min_effect=min_eff_all,
            max_effect=max_eff_all,
        )

        # Compute SSD
        ssd_output = self.ssd(log_ec50_all)

        # Compute implied exposure and risk
        implied_exposure, risk_levels = self.exposure_calc(
            log_ec50_all, contaminant_conc,
        )

        # Compute margin of safety
        margin_of_safety = self.benchmark_head(ssd_output, contaminant_conc)

        return curve_params, implied_exposure, risk_levels, ssd_output, margin_of_safety

    def forward(
        self,
        sentinel_embedding: torch.Tensor,
        contaminant_conc: torch.Tensor,
        use_mc_dropout: bool = False,
    ) -> FieldDoseResponseOutput:
        """Forward pass with optional MC-dropout uncertainty.

        Parameters
        ----------
        sentinel_embedding : Tensor[B, 256]
            SENTINEL environmental embedding.
        contaminant_conc : Tensor[B, C_chem]
            Log10-transformed contaminant concentrations from USGS.
        use_mc_dropout : bool
            If True and in eval mode, run mc_samples stochastic
            forward passes.

        Returns
        -------
        FieldDoseResponseOutput
        """
        if not use_mc_dropout or self.training:
            cr, ie, rl, ssd_out, mos = self._single_forward(
                sentinel_embedding, contaminant_conc,
            )
            return FieldDoseResponseOutput(
                curve_params=cr,
                implied_exposure=ie,
                risk_levels=rl,
                ssd=ssd_out,
                margin_of_safety=mos,
            )

        # --- MC-dropout inference ---
        self._enable_dropout()

        mc_log_ec50: List[torch.Tensor] = []
        mc_hc5: List[torch.Tensor] = []

        with torch.no_grad():
            for _ in range(self.mc_samples):
                cr, ie, rl, ssd_out, mos = self._single_forward(
                    sentinel_embedding, contaminant_conc,
                )
                mc_log_ec50.append(cr.log_ec50)
                mc_hc5.append(ssd_out.hc5)

        self._disable_dropout()

        ec50_stack = torch.stack(mc_log_ec50, dim=0)  # [N, B, S, C]
        hc5_stack = torch.stack(mc_hc5, dim=0)         # [N, B, C]

        return FieldDoseResponseOutput(
            curve_params=cr,
            implied_exposure=ie,
            risk_levels=rl,
            ssd=ssd_out,
            margin_of_safety=mos,
            mc_log_ec50_mean=ec50_stack.mean(dim=0).mean(dim=-1),  # [B, S]
            mc_log_ec50_std=ec50_stack.std(dim=0).mean(dim=-1),     # [B, S]
            mc_hc5_mean=hc5_stack.mean(dim=0),    # [B, C]
            mc_hc5_std=hc5_stack.std(dim=0),       # [B, C]
        )

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    @staticmethod
    def compute_loss(
        output: FieldDoseResponseOutput,
        targets: Dict[str, torch.Tensor],
        loss_weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute multi-task training loss.

        Parameters
        ----------
        output : FieldDoseResponseOutput
            Model predictions.
        targets : dict
            Expected keys (all optional, missing keys are skipped):

            - ``"log_ec50"`` : Tensor[B, S, C] -- lab-measured EC50.
            - ``"implied_exposure"`` : Tensor[B, S, C] -- field observations.
            - ``"hc5"`` : Tensor[B, C] -- regulatory HC5 values.

        loss_weights : dict, optional
            Per-task weighting factors.

        Returns
        -------
        total_loss : scalar Tensor
        per_task : Dict[str, Tensor]
        """
        if loss_weights is None:
            loss_weights = {
                "ec50": 1.0,
                "exposure": 0.5,
                "hc5": 0.3,
            }

        losses: Dict[str, torch.Tensor] = {}
        device = output.implied_exposure.device

        if "log_ec50" in targets:
            losses["ec50"] = F.mse_loss(
                output.curve_params.log_ec50,
                targets["log_ec50"].float(),
            )

        if "implied_exposure" in targets:
            losses["exposure"] = F.mse_loss(
                output.implied_exposure,
                targets["implied_exposure"].float(),
            )

        if "hc5" in targets:
            losses["hc5"] = F.mse_loss(
                output.ssd.hc5,
                targets["hc5"].float(),
            )

        total = torch.tensor(0.0, device=device)
        for key, loss_val in losses.items():
            w = loss_weights.get(key, 1.0)
            total = total + w * loss_val

        return total, losses

    # ------------------------------------------------------------------
    # Utility: evaluate dose-response curve at given concentrations
    # ------------------------------------------------------------------

    @staticmethod
    def evaluate_dose_response(
        log_concentration: torch.Tensor,
        curve_params: DoseResponseCurveParams,
    ) -> torch.Tensor:
        """Evaluate the 4-parameter log-logistic dose-response curve.

        Parameters
        ----------
        log_concentration : Tensor[B, C]
            Log10 concentration values to evaluate.
        curve_params : DoseResponseCurveParams
            Predicted curve parameters (from a single species).

        Returns
        -------
        Tensor[B, C]
            Predicted effect level at each concentration.
        """
        # 4PL model: effect = min + (max - min) / (1 + (c/EC50)^(-slope))
        log_ratio = log_concentration - curve_params.log_ec50  # [B, C]
        exponent = -curve_params.slope * log_ratio * math.log(10.0)
        exponent = exponent.clamp(-20.0, 20.0)  # numerical stability

        response = curve_params.min_effect + (
            (curve_params.max_effect - curve_params.min_effect)
            / (1.0 + torch.exp(exponent))
        )

        return response
