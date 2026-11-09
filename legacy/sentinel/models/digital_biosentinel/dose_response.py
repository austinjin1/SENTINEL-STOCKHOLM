"""Core dose-response model for the Digital Biosentinel.

Architecture
------------
Input vector (concatenation):
    chemical_embed   128
    concentration     1   (log10-scaled)
    species_embed    64
    exposure_dur      1   (log10-scaled hours)
    trophic_level     1   (numeric, 0-4)
    ─────────────────────
    Total            195

Shared backbone: 195 → 512 → 256 → 128  (ReLU + Dropout 0.3)

Output heads:
    mortality_head       sigmoid  → P(mortality)
    growth_head          linear   → growth inhibition  %
    reproduction_head    linear   → reproduction effect magnitude
    behavioral_head      sigmoid  → P(behavioral change)
    uncertainty_head     softplus → expected prediction error magnitude

Uncertainty estimation uses Monte Carlo (MC) dropout: at inference we run
N stochastic forward passes with dropout enabled and report the mean
prediction and inter-pass variance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DoseResponseOutput:
    """Structured output from the dose-response model."""

    mortality: torch.Tensor          # [B]  probability
    growth_inhibition: torch.Tensor  # [B]  percentage (0-100+)
    reproduction_effect: torch.Tensor  # [B]  magnitude
    behavioral_change: torch.Tensor  # [B]  probability
    uncertainty: torch.Tensor        # [B]  predicted error magnitude
    # Populated only when mc_dropout is used:
    mc_mortality_mean: Optional[torch.Tensor] = None
    mc_mortality_std: Optional[torch.Tensor] = None
    mc_growth_mean: Optional[torch.Tensor] = None
    mc_growth_std: Optional[torch.Tensor] = None
    mc_reproduction_mean: Optional[torch.Tensor] = None
    mc_reproduction_std: Optional[torch.Tensor] = None
    mc_behavioral_mean: Optional[torch.Tensor] = None
    mc_behavioral_std: Optional[torch.Tensor] = None


class DoseResponseModel(nn.Module):
    """Multi-head dose-response predictor with MC-dropout uncertainty.

    Parameters
    ----------
    chemical_dim : int
        Chemical embedding dimensionality (default 128).
    species_dim : int
        Species embedding dimensionality (default 64).
    backbone_dims : Tuple[int, ...]
        Hidden layer sizes for the shared backbone.
    dropout : float
        Dropout probability (used at both train and MC-inference time).
    mc_samples : int
        Number of stochastic forward passes for MC dropout (default 20).
    """

    def __init__(
        self,
        chemical_dim: int = 128,
        species_dim: int = 64,
        backbone_dims: Tuple[int, ...] = (512, 256, 128),
        dropout: float = 0.3,
        mc_samples: int = 20,
    ) -> None:
        super().__init__()
        self.mc_samples = mc_samples
        self.dropout_p = dropout

        # Input: chemical_embed + concentration + species_embed + duration + trophic
        input_dim = chemical_dim + 1 + species_dim + 1 + 1  # 195

        # --- Shared backbone ---------------------------------------------------
        layers = []
        in_features = input_dim
        for out_features in backbone_dims:
            layers.extend([
                nn.Linear(in_features, out_features),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_features = out_features
        self.backbone = nn.Sequential(*layers)

        backbone_out = backbone_dims[-1]

        # --- Output heads ------------------------------------------------------
        self.mortality_head = nn.Sequential(
            nn.Linear(backbone_out, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.growth_head = nn.Sequential(
            nn.Linear(backbone_out, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

        self.reproduction_head = nn.Sequential(
            nn.Linear(backbone_out, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

        self.behavioral_head = nn.Sequential(
            nn.Linear(backbone_out, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Uncertainty head: predicts expected absolute error magnitude
        self.uncertainty_head = nn.Sequential(
            nn.Linear(backbone_out, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Softplus(),  # ensure non-negative
        )

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    def _build_input(
        self,
        chemical_embed: torch.Tensor,
        log_concentration: torch.Tensor,
        species_embed: torch.Tensor,
        log_exposure_hours: torch.Tensor,
        trophic_level: torch.Tensor,
    ) -> torch.Tensor:
        """Concatenate all inputs into a single feature vector.

        All scalar tensors are expected to be [B] or [B, 1]; they will be
        reshaped to [B, 1] before concatenation.
        """
        def _ensure_2d(t: torch.Tensor) -> torch.Tensor:
            if t.dim() == 1:
                return t.unsqueeze(-1)
            return t

        return torch.cat(
            [
                chemical_embed,
                _ensure_2d(log_concentration),
                species_embed,
                _ensure_2d(log_exposure_hours),
                _ensure_2d(trophic_level),
            ],
            dim=-1,
        )

    # ------------------------------------------------------------------
    def _single_forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One forward pass through backbone + all heads."""
        h = self.backbone(x)
        mortality = self.mortality_head(h).squeeze(-1)
        growth = self.growth_head(h).squeeze(-1)
        reproduction = self.reproduction_head(h).squeeze(-1)
        behavioral = self.behavioral_head(h).squeeze(-1)
        uncertainty = self.uncertainty_head(h).squeeze(-1)
        return mortality, growth, reproduction, behavioral, uncertainty

    # ------------------------------------------------------------------
    def forward(
        self,
        chemical_embed: torch.Tensor,
        log_concentration: torch.Tensor,
        species_embed: torch.Tensor,
        log_exposure_hours: torch.Tensor,
        trophic_level: torch.Tensor,
        use_mc_dropout: bool = False,
    ) -> DoseResponseOutput:
        """Forward pass.

        Parameters
        ----------
        chemical_embed : Tensor[B, 128]
        log_concentration : Tensor[B]  — log10(mg/L)
        species_embed : Tensor[B, 64]
        log_exposure_hours : Tensor[B]  — log10(hours)
        trophic_level : Tensor[B]  — numeric (0-4)
        use_mc_dropout : bool
            If True, run ``mc_samples`` stochastic passes and populate
            the ``mc_*`` fields in the output.

        Returns
        -------
        DoseResponseOutput
        """
        x = self._build_input(
            chemical_embed, log_concentration, species_embed,
            log_exposure_hours, trophic_level,
        )

        if not use_mc_dropout or self.training:
            # Standard single forward pass
            mort, grow, repro, behav, unc = self._single_forward(x)
            return DoseResponseOutput(
                mortality=mort,
                growth_inhibition=grow,
                reproduction_effect=repro,
                behavioral_change=behav,
                uncertainty=unc,
            )

        # --- MC Dropout inference ----------------------------------------------
        # Enable dropout even in eval mode
        self._enable_dropout()

        mc_mort, mc_grow, mc_repro, mc_behav = [], [], [], []
        for _ in range(self.mc_samples):
            mort, grow, repro, behav, _ = self._single_forward(x)
            mc_mort.append(mort)
            mc_grow.append(grow)
            mc_repro.append(repro)
            mc_behav.append(behav)

        self._disable_dropout()

        mc_mort = torch.stack(mc_mort, dim=0)    # [N, B]
        mc_grow = torch.stack(mc_grow, dim=0)
        mc_repro = torch.stack(mc_repro, dim=0)
        mc_behav = torch.stack(mc_behav, dim=0)

        # Single deterministic pass for the uncertainty head value
        mort_mean = mc_mort.mean(dim=0)
        grow_mean = mc_grow.mean(dim=0)
        repro_mean = mc_repro.mean(dim=0)
        behav_mean = mc_behav.mean(dim=0)

        # Use the MC variance of mortality as the main uncertainty signal
        combined_unc = mc_mort.std(dim=0) + mc_grow.std(dim=0) * 0.01

        return DoseResponseOutput(
            mortality=mort_mean,
            growth_inhibition=grow_mean,
            reproduction_effect=repro_mean,
            behavioral_change=behav_mean,
            uncertainty=combined_unc,
            mc_mortality_mean=mort_mean,
            mc_mortality_std=mc_mort.std(dim=0),
            mc_growth_mean=grow_mean,
            mc_growth_std=mc_grow.std(dim=0),
            mc_reproduction_mean=repro_mean,
            mc_reproduction_std=mc_repro.std(dim=0),
            mc_behavioral_mean=behav_mean,
            mc_behavioral_std=mc_behav.std(dim=0),
        )

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

    # ------------------------------------------------------------------
    @staticmethod
    def compute_loss(
        output: DoseResponseOutput,
        targets: Dict[str, torch.Tensor],
        endpoint_weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the multi-task training loss.

        Parameters
        ----------
        output : DoseResponseOutput
        targets : dict
            Keys: ``"mortality"`` (binary), ``"growth_inhibition"`` (float),
            ``"reproduction_effect"`` (float), ``"behavioral_change"`` (binary).
            Missing keys are simply skipped.
        endpoint_weights : dict, optional
            Per-endpoint loss scaling factors.

        Returns
        -------
        total_loss : scalar Tensor
        per_endpoint : Dict[str, Tensor]
            Individual endpoint losses (for logging).
        """
        if endpoint_weights is None:
            endpoint_weights = {
                "mortality": 1.0,
                "growth_inhibition": 0.5,
                "reproduction_effect": 0.5,
                "behavioral_change": 0.8,
            }

        losses: Dict[str, torch.Tensor] = {}
        device = output.mortality.device

        if "mortality" in targets:
            losses["mortality"] = F.binary_cross_entropy(
                output.mortality, targets["mortality"].float(),
            )

        if "growth_inhibition" in targets:
            losses["growth_inhibition"] = F.mse_loss(
                output.growth_inhibition, targets["growth_inhibition"].float(),
            )

        if "reproduction_effect" in targets:
            losses["reproduction_effect"] = F.mse_loss(
                output.reproduction_effect, targets["reproduction_effect"].float(),
            )

        if "behavioral_change" in targets:
            losses["behavioral_change"] = F.binary_cross_entropy(
                output.behavioral_change, targets["behavioral_change"].float(),
            )

        # Uncertainty-aware weighting: penalise under-confident predictions
        # when the model is correct and over-confident when wrong.
        if "mortality" in targets:
            residual = (output.mortality - targets["mortality"].float()).abs()
            unc_loss = F.mse_loss(output.uncertainty, residual)
            losses["uncertainty"] = unc_loss * 0.1

        total = torch.tensor(0.0, device=device)
        for key, loss_val in losses.items():
            w = endpoint_weights.get(key, 1.0)
            total = total + w * loss_val

        return total, losses
