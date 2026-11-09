"""Water quality parameter regression head with uncertainty estimation.

Predicts 16 water quality parameters from the HydroViT embedding, each
with its own lightweight MLP head sharing a common backbone.  Uses
heteroscedastic aleatoric uncertainty (predicting mean + log_variance)
so the model can express confidence per parameter per sample.

Parameters (index order):
  0: chl_a              - Chlorophyll-a concentration (ug/L)
  1: turbidity          - Turbidity (NTU)
  2: secchi_depth       - Secchi disk depth (m)
  3: cdom               - Colored dissolved organic matter (m^-1)
  4: tss                - Total suspended solids (mg/L)
  5: total_nitrogen     - Total nitrogen (mg/L)
  6: total_phosphorus   - Total phosphorus (mg/L)
  7: dissolved_oxygen   - Dissolved oxygen (mg/L)
  8: ammonia            - Ammonia-N (mg/L)
  9: nitrate            - Nitrate-N (mg/L)
 10: ph                 - pH
 11: water_temp         - Water temperature (C)
 12: phycocyanin        - Phycocyanin concentration (ug/L)
 13: oil_probability    - Oil/hydrocarbon presence probability [0,1]
 14: acdom              - Absorption coeff of CDOM at 440nm (m^-1)
 15: pollution_anomaly  - Pollution anomaly index [0,1]
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_WATER_PARAMS = 16

PARAM_NAMES: tuple[str, ...] = (
    "chl_a",
    "turbidity",
    "secchi_depth",
    "cdom",
    "tss",
    "total_nitrogen",
    "total_phosphorus",
    "dissolved_oxygen",
    "ammonia",
    "nitrate",
    "ph",
    "water_temp",
    "phycocyanin",
    "oil_probability",
    "acdom",
    "pollution_anomaly_index",
)

# Physical bounds for clamping predictions (min, max).
# None means unbounded on that side.
PARAM_BOUNDS: dict[str, tuple[float | None, float | None]] = {
    "chl_a": (0.0, None),
    "turbidity": (0.0, None),
    "secchi_depth": (0.0, None),
    "cdom": (0.0, None),
    "tss": (0.0, None),
    "total_nitrogen": (0.0, None),
    "total_phosphorus": (0.0, None),
    "dissolved_oxygen": (0.0, None),
    "ammonia": (0.0, None),
    "nitrate": (0.0, None),
    "ph": (0.0, 14.0),
    "water_temp": (-2.0, 50.0),
    "phycocyanin": (0.0, None),
    "oil_probability": (0.0, 1.0),
    "acdom": (0.0, None),
    "pollution_anomaly_index": (0.0, 1.0),
}


class ParameterHead(nn.Module):
    """Single-parameter regression head with uncertainty.

    Predicts mean and log_variance for one water quality parameter.

    Args:
        input_dim: Input feature dimension.
        hidden_dim: Hidden layer dimension.
        param_name: Name of the parameter (for bounds clamping).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        param_name: str = "",
    ) -> None:
        super().__init__()
        self.param_name = param_name
        bounds = PARAM_BOUNDS.get(param_name, (None, None))
        self.lower_bound = bounds[0]
        self.upper_bound = bounds[1]

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
        )
        self.mean_head = nn.Linear(hidden_dim // 2, 1)
        self.logvar_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict mean and log_variance.

        Args:
            x: [B, input_dim].

        Returns:
            mean: [B, 1] predicted parameter value.
            log_var: [B, 1] log variance (uncertainty).
        """
        h = self.mlp(x)
        mean = self.mean_head(h)
        log_var = self.logvar_head(h)

        # Clamp log_var to prevent numerical instability
        log_var = log_var.clamp(-10.0, 10.0)

        # Apply physical bounds via soft clamping
        if self.lower_bound is not None and self.upper_bound is not None:
            mean = torch.sigmoid(mean) * (self.upper_bound - self.lower_bound) + self.lower_bound
        elif self.lower_bound is not None:
            mean = F.softplus(mean) + self.lower_bound
        elif self.upper_bound is not None:
            mean = self.upper_bound - F.softplus(-mean)

        return mean, log_var


class WaterQualityHead(nn.Module):
    """Multi-parameter water quality regression head.

    Shares a common backbone projection, then branches into 16 individual
    parameter heads, each predicting mean + log_variance.

    Args:
        input_dim: Input embedding dimension from HydroViT.
        backbone_dim: Shared backbone hidden dimension.
        head_hidden_dim: Per-parameter head hidden dimension.
    """

    def __init__(
        self,
        input_dim: int = 384,
        backbone_dim: int = 256,
        head_hidden_dim: int = 128,
    ) -> None:
        super().__init__()

        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, backbone_dim),
            nn.GELU(),
            nn.LayerNorm(backbone_dim),
            nn.Dropout(0.1),
            nn.Linear(backbone_dim, backbone_dim),
            nn.GELU(),
            nn.LayerNorm(backbone_dim),
        )

        # Per-parameter heads
        self.heads = nn.ModuleDict({
            name: ParameterHead(backbone_dim, head_hidden_dim, name)
            for name in PARAM_NAMES
        })

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict all 16 water quality parameters with uncertainty.

        Args:
            x: [B, input_dim] embedding from HydroViT.

        Returns:
            params: [B, 16] predicted parameter values.
            uncertainties: [B, 16] log variance per parameter.
        """
        B = x.shape[0]
        backbone_features = self.backbone(x)

        means = []
        log_vars = []
        for name in PARAM_NAMES:
            mean, log_var = self.heads[name](backbone_features)
            means.append(mean)
            log_vars.append(log_var)

        params = torch.cat(means, dim=-1)          # [B, 16]
        uncertainties = torch.cat(log_vars, dim=-1)  # [B, 16]

        return params, uncertainties

    @staticmethod
    def gaussian_nll_loss(
        predictions: torch.Tensor,
        uncertainties: torch.Tensor,
        targets: torch.Tensor,
        param_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Heteroscedastic Gaussian negative log-likelihood loss.

        Args:
            predictions: [B, 16] predicted means.
            uncertainties: [B, 16] predicted log variances.
            targets: [B, 16] ground truth values. NaN = missing.
            param_weights: [16] optional per-parameter importance weights.

        Returns:
            Scalar loss.
        """
        # Mask out NaN targets (missing ground truth)
        valid_mask = ~torch.isnan(targets)
        if not valid_mask.any():
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)

        # Replace NaN targets with 0 before computation (NaN * 0.0 = NaN in IEEE 754)
        targets_safe = targets.clone()
        targets_safe[~valid_mask] = 0.0

        # Gaussian NLL: 0.5 * (log_var + (y - mu)^2 / exp(log_var))
        precision = torch.exp(-uncertainties.clamp(-10, 10))
        nll = 0.5 * (uncertainties.clamp(-10, 10) + (targets_safe - predictions) ** 2 * precision)

        # Zero out invalid entries
        nll = nll * valid_mask.float()

        if param_weights is not None:
            nll = nll * param_weights.unsqueeze(0)

        return nll.sum() / valid_mask.float().sum().clamp(min=1.0)
