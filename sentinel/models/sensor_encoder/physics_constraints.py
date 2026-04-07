"""Physics-informed constraint losses for water quality sensor data.

Encodes known physical relationships between water quality parameters as
soft penalty losses. These act as regularizers during training, encouraging
predictions that respect physical/chemical laws.

Constraints:
    1. DO-Temperature: Dissolved oxygen saturation decreases with temperature.
    2. pH-Conductivity: Coupled through carbonate chemistry.
    3. Conductivity-TDS: Linear proportionality (Conductivity ~ k * TDS).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def do_saturation_max(temperature: torch.Tensor) -> torch.Tensor:
    """Compute maximum dissolved oxygen saturation at given temperature.

    Uses the Benson & Krause (1984) empirical formula for DO saturation
    in fresh water at 1 atm.

    DO_max(T) = 14.62 - 0.3898*T + 0.006969*T^2 - 5.897e-5*T^3

    where T is in degrees Celsius and DO_max is in mg/L.

    Args:
        temperature: Water temperature in Celsius [*].

    Returns:
        Maximum DO saturation in mg/L [*], same shape as input.
    """
    T = temperature
    return 14.62 - 0.3898 * T + 0.006969 * T ** 2 - 5.897e-5 * T ** 3


def do_temperature_constraint(
    do_pred: torch.Tensor,
    temp_pred: torch.Tensor,
) -> torch.Tensor:
    """Penalize predicted DO exceeding physical saturation limit.

    DO cannot exceed the temperature-dependent saturation maximum
    (barring supersaturation from algal blooms, which is rare).

    Args:
        do_pred: Predicted dissolved oxygen [*] (mg/L).
        temp_pred: Predicted temperature [*] (Celsius).

    Returns:
        Scalar penalty loss (mean of ReLU violations).
    """
    do_max = do_saturation_max(temp_pred)
    violation = F.relu(do_pred - do_max)  # Only penalize excess
    return violation.mean()


def ph_alkalinity_constraint(
    ph_pred: torch.Tensor,
    conductivity_pred: torch.Tensor,
) -> torch.Tensor:
    """Soft constraint coupling pH and conductivity via carbonate chemistry.

    In natural waters, pH and conductivity are related through dissolved
    ionic species (primarily carbonate/bicarbonate). Extreme pH with low
    conductivity is physically implausible.

    Penalty: if pH is extreme (< 5 or > 9) but conductivity is low,
    the model is penalized.

    Args:
        ph_pred: Predicted pH values [*].
        conductivity_pred: Predicted conductivity [*] (uS/cm).

    Returns:
        Scalar penalty loss.
    """
    # pH deviation from neutral range
    ph_extreme_low = F.relu(5.0 - ph_pred)   # penalty when pH < 5
    ph_extreme_high = F.relu(ph_pred - 9.0)  # penalty when pH > 9
    ph_deviation = ph_extreme_low + ph_extreme_high

    # Low conductivity indicator (below 50 uS/cm suggests very few ions)
    low_cond = F.relu(50.0 - conductivity_pred) / 50.0  # normalized [0, 1]

    # Penalty: extreme pH AND low conductivity
    violation = ph_deviation * low_cond
    return violation.mean()


def conductivity_tds_constraint(
    conductivity_pred: torch.Tensor,
    tds_pred: torch.Tensor,
    k_range: tuple[float, float] = (0.5, 0.9),
) -> torch.Tensor:
    """Penalize deviations from Conductivity ~ k * TDS relationship.

    For most natural waters, TDS (mg/L) = k * Conductivity (uS/cm),
    where k is typically 0.5-0.9 depending on ionic composition.

    Args:
        conductivity_pred: Predicted conductivity [*] (uS/cm).
        tds_pred: Predicted TDS [*] (mg/L).
        k_range: Valid range of proportionality constant (k_min, k_max).

    Returns:
        Scalar penalty loss.
    """
    k_min, k_max = k_range
    # Expected TDS range given conductivity
    tds_min = k_min * conductivity_pred
    tds_max = k_max * conductivity_pred

    # Penalize TDS outside expected range
    violation_low = F.relu(tds_min - tds_pred)
    violation_high = F.relu(tds_pred - tds_max)
    violation = violation_low + violation_high

    return violation.mean()


class PhysicsConstraintLoss(nn.Module):
    """Combined physics constraint loss with learnable weights.

    Aggregates individual physical constraint penalties with
    per-constraint learnable weights (log-space for positivity).

    The total loss is: sum_i exp(log_w_i) * constraint_i + sum_i log_w_i
    The second term prevents weights from collapsing to zero
    (Kendall et al., 2018 multi-task uncertainty weighting).

    Args:
        num_constraints: Number of physics constraints. Default 3.
        initial_weight: Initial weight for each constraint.
    """

    CONSTRAINT_NAMES = [
        "do_temperature",
        "ph_alkalinity",
        "conductivity_tds",
    ]

    def __init__(
        self,
        num_constraints: int = 3,
        initial_weight: float = 1.0,
    ) -> None:
        super().__init__()
        # Log-space weights for positivity and uncertainty weighting
        self.log_weights = nn.Parameter(
            torch.full((num_constraints,), math.log(initial_weight))
        )

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute combined physics constraint loss.

        Args:
            predictions: Dict with parameter predictions. Expected keys:
                'do': Dissolved oxygen predictions [*].
                'temperature': Temperature predictions [*].
                'ph': pH predictions [*].
                'conductivity': Conductivity predictions [*].
                'tds': TDS predictions [*] (optional, skip constraint if absent).

        Returns:
            Dict with:
                'total_loss': Combined weighted physics loss (scalar).
                'do_temperature_loss': DO-temperature constraint loss.
                'ph_alkalinity_loss': pH-alkalinity constraint loss.
                'conductivity_tds_loss': Conductivity-TDS loss (0 if no TDS).
                'constraint_weights': Current constraint weights [3].
        """
        losses = []

        # 1. DO-Temperature constraint
        if "do" in predictions and "temperature" in predictions:
            do_temp_loss = do_temperature_constraint(
                predictions["do"], predictions["temperature"]
            )
        else:
            do_temp_loss = torch.tensor(0.0, device=self.log_weights.device)
        losses.append(do_temp_loss)

        # 2. pH-Conductivity constraint
        if "ph" in predictions and "conductivity" in predictions:
            ph_alk_loss = ph_alkalinity_constraint(
                predictions["ph"], predictions["conductivity"]
            )
        else:
            ph_alk_loss = torch.tensor(0.0, device=self.log_weights.device)
        losses.append(ph_alk_loss)

        # 3. Conductivity-TDS constraint
        if "conductivity" in predictions and "tds" in predictions:
            cond_tds_loss = conductivity_tds_constraint(
                predictions["conductivity"], predictions["tds"]
            )
        else:
            cond_tds_loss = torch.tensor(0.0, device=self.log_weights.device)
        losses.append(cond_tds_loss)

        # Weighted combination with uncertainty regularization
        # Clamp log_weights to prevent instability from extreme values
        clamped_log_w = self.log_weights.clamp(min=-4.0, max=4.0)
        weights = torch.exp(clamped_log_w)
        total_loss = torch.tensor(0.0, device=self.log_weights.device)
        for w, log_w, loss in zip(weights, clamped_log_w, losses):
            total_loss = total_loss + w * loss + log_w  # uncertainty weighting

        return {
            "total_loss": total_loss,
            "do_temperature_loss": do_temp_loss,
            "ph_alkalinity_loss": ph_alk_loss,
            "conductivity_tds_loss": cond_tds_loss,
            "constraint_weights": weights.detach(),
        }
