"""Spectral physics consistency loss for water-leaving radiance.

Enforces semi-analytical constraints from ocean color models on predicted
or reconstructed spectral patches, ensuring physical plausibility:

1. Band ratio constraints: Rrs(B4)/Rrs(B3) within [0.1, 2.0] for water.
2. Negative reflectance penalty: Water-leaving radiance must be non-negative.
3. NIR water constraint: Rrs(NIR) near-zero for clear water.
4. SWIR negligibility: SWIR reflectance near-zero over water.
5. Spectral smoothness: Adjacent bands should not have extreme discontinuities.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# Band indices within the 13-band input:
# 0:B2(Blue) 1:B3(Green) 2:B4(Red) 3:B5(VRE1) 4:B6(VRE2) 5:B7(VRE3)
# 6:B8(NIR) 7:B8A(NarrowNIR) 8:B11(SWIR1) 9:B12(SWIR2)
# 10:OLCI-443 11:OLCI-560 12:OLCI-665
BAND_GREEN = 1       # B3  (560 nm)
BAND_RED = 2         # B4  (665 nm)
BAND_NIR = 6         # B8  (842 nm)
BAND_NIR_NARROW = 7  # B8A (865 nm)
BAND_SWIR1 = 8       # B11 (1610 nm)
BAND_SWIR2 = 9       # B12 (2190 nm)

# Physical bounds
RATIO_RED_GREEN_MIN = 0.1
RATIO_RED_GREEN_MAX = 2.0
NIR_CLEAR_WATER_MAX = 0.02   # Rrs threshold above which NIR is penalized
SWIR_CLEAR_WATER_MAX = 0.005


class SpectralPhysicsLoss(nn.Module):
    """Spectral physics consistency loss for water remote sensing.

    Computes soft penalties encouraging predicted reflectance values to
    satisfy known physical constraints for water-leaving radiance.

    Args:
        lambda_ratio: Weight for band ratio constraint.
        lambda_negative: Weight for negative reflectance penalty.
        lambda_nir: Weight for NIR water constraint.
        lambda_swir: Weight for SWIR negligibility constraint.
        lambda_smoothness: Weight for spectral smoothness constraint.
        turbidity_adaptive: If True, relax NIR constraint when turbidity
            indicators suggest high sediment load.
    """

    def __init__(
        self,
        lambda_ratio: float = 1.0,
        lambda_negative: float = 2.0,
        lambda_nir: float = 1.0,
        lambda_swir: float = 0.5,
        lambda_smoothness: float = 0.3,
        turbidity_adaptive: bool = True,
    ) -> None:
        super().__init__()
        self.lambda_ratio = lambda_ratio
        self.lambda_negative = lambda_negative
        self.lambda_nir = lambda_nir
        self.lambda_swir = lambda_swir
        self.lambda_smoothness = lambda_smoothness
        self.turbidity_adaptive = turbidity_adaptive

    def forward(
        self,
        predicted: torch.Tensor,
        water_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute spectral physics consistency losses.

        Args:
            predicted: Predicted reflectance values.  Can be:
                - Per-band image [B, C, H, W]  (C >= 13)
                - Patchified    [B, N, patch_size**2 * C]  (will be treated as flat)
            water_mask: Optional binary mask [B, 1, H, W] or [B, N].
                1 = water pixel, 0 = land/cloud.  If None, all pixels treated
                as water.

        Returns:
            Dict of individual loss terms and total weighted loss.
        """
        # Reshape to [B, C, ...] if patchified
        if predicted.dim() == 3:
            # Patchified: [B, N, patch_area * C] -> extract band means per patch
            predicted = self._extract_band_means_from_patches(predicted)

        B, C = predicted.shape[0], predicted.shape[1]
        device = predicted.device

        # Flatten spatial dims: [B, C, *spatial] -> [B, C, S]
        if predicted.dim() > 3:
            predicted = predicted.reshape(B, C, -1)
        elif predicted.dim() == 2:
            predicted = predicted.unsqueeze(-1)

        # Apply water mask
        if water_mask is not None:
            if water_mask.dim() > 2:
                water_mask = water_mask.reshape(B, 1, -1)
            else:
                water_mask = water_mask.unsqueeze(1)
            mask_weight = water_mask.float()
        else:
            mask_weight = torch.ones(B, 1, predicted.shape[-1], device=device)

        num_water = mask_weight.sum().clamp(min=1.0)

        # 1. Band ratio constraint: Rrs(Red)/Rrs(Green) in [0.1, 2.0]
        green = predicted[:, BAND_GREEN: BAND_GREEN + 1, :]
        red = predicted[:, BAND_RED: BAND_RED + 1, :]
        ratio = red / (green + 1e-8)
        ratio_violation_low = F.relu(RATIO_RED_GREEN_MIN - ratio)
        ratio_violation_high = F.relu(ratio - RATIO_RED_GREEN_MAX)
        loss_ratio = ((ratio_violation_low + ratio_violation_high) * mask_weight).sum() / num_water

        # 2. Negative reflectance penalty
        loss_negative = (F.relu(-predicted) * mask_weight).sum() / (num_water * C)

        # 3. NIR water constraint
        nir = predicted[:, BAND_NIR: BAND_NIR + 1, :]
        nir_narrow = predicted[:, BAND_NIR_NARROW: BAND_NIR_NARROW + 1, :]

        if self.turbidity_adaptive:
            # Estimate turbidity proxy from red/green ratio
            turbidity_proxy = (ratio.detach().clamp(0, 5) / 5.0)
            nir_threshold = NIR_CLEAR_WATER_MAX + turbidity_proxy * 0.1
        else:
            nir_threshold = NIR_CLEAR_WATER_MAX

        loss_nir = (
            (F.relu(nir - nir_threshold) * mask_weight).sum()
            + (F.relu(nir_narrow - nir_threshold) * mask_weight).sum()
        ) / num_water

        # 4. SWIR negligibility
        if C > BAND_SWIR2:
            swir1 = predicted[:, BAND_SWIR1: BAND_SWIR1 + 1, :]
            swir2 = predicted[:, BAND_SWIR2: BAND_SWIR2 + 1, :]
            loss_swir = (
                (F.relu(swir1 - SWIR_CLEAR_WATER_MAX) * mask_weight).sum()
                + (F.relu(swir2 - SWIR_CLEAR_WATER_MAX) * mask_weight).sum()
            ) / num_water
        else:
            loss_swir = torch.tensor(0.0, device=device)

        # 5. Spectral smoothness: penalize large jumps between adjacent bands
        band_diffs = torch.diff(predicted[:, :10, :], dim=1)  # First 10 S2 bands
        loss_smoothness = ((band_diffs.abs() * mask_weight).sum()) / (num_water * 9)

        # Total weighted loss
        total = (
            self.lambda_ratio * loss_ratio
            + self.lambda_negative * loss_negative
            + self.lambda_nir * loss_nir
            + self.lambda_swir * loss_swir
            + self.lambda_smoothness * loss_smoothness
        )

        return {
            "physics_loss": total,
            "loss_ratio": loss_ratio,
            "loss_negative": loss_negative,
            "loss_nir": loss_nir,
            "loss_swir": loss_swir,
            "loss_smoothness": loss_smoothness,
        }

    @staticmethod
    def _extract_band_means_from_patches(
        patches: torch.Tensor,
    ) -> torch.Tensor:
        """Extract per-band mean from patchified representation.

        Args:
            patches: [B, N, patch_area * C] where C=13 and patch_area = 16*16 = 256.

        Returns:
            Band means [B, 13, N].
        """
        B, N, D = patches.shape
        C = 13
        patch_area = D // C
        # [B, N, patch_area, C] -> mean over patch_area -> [B, N, C] -> [B, C, N]
        x = patches.reshape(B, N, patch_area, C)
        x = x.mean(dim=2)  # [B, N, C]
        return x.permute(0, 2, 1)  # [B, C, N]
