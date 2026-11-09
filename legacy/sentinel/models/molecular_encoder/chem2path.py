"""Chemistry-to-Pathway (Chem2Path) prediction model.

Multi-task architecture that predicts biological pathway activation profiles
from chemical identity and concentration. Shared MLP backbone with separate
per-pathway prediction heads enables transfer learning across pathways.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Biological pathways for water quality toxicity assessment
PATHWAYS = [
    "oxidative_stress",
    "endocrine_disruption",
    "neurotoxicity",
    "genotoxicity",
    "immunotoxicity",
    "hepatotoxicity",
    "nephrotoxicity",
]
NUM_PATHWAYS = len(PATHWAYS)


class Chem2PathBackbone(nn.Module):
    """Shared chemistry feature backbone.

    Processes chemical identity (one-hot class encoding) and log concentration
    into a shared feature representation used by all pathway heads.

    Args:
        num_chem_classes: Number of chemical class categories.
        backbone_dims: Dimensions of shared MLP layers.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        num_chem_classes: int = 50,
        backbone_dims: list[int] | None = None,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if backbone_dims is None:
            backbone_dims = [512, 256, 128]

        # Input: one-hot chemical class + log concentration
        input_dim = num_chem_classes + 1  # +1 for concentration

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for dim in backbone_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = dim

        self.backbone = nn.Sequential(*layers)
        self.output_dim = backbone_dims[-1]

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through shared backbone.

        Args:
            x: Concatenated [one_hot_class, log_concentration] [B, input_dim].

        Returns:
            Shared features [B, output_dim].
        """
        return self.backbone(x)


class PathwayHead(nn.Module):
    """Single pathway prediction head.

    Takes shared backbone features and predicts activation probability
    for one specific biological pathway.

    Args:
        input_dim: Backbone output dimension.
        pathway_name: Name of the pathway (for logging).
    """

    def __init__(self, input_dim: int = 128, pathway_name: str = "") -> None:
        super().__init__()
        self.pathway_name = pathway_name
        self.head = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict pathway activation probability.

        Args:
            features: Shared backbone features [B, input_dim].

        Returns:
            Activation probability [B, 1] (after sigmoid).
        """
        return torch.sigmoid(self.head(features))


class Chem2Path(nn.Module):
    """Chemistry-to-Pathway prediction model.

    Multi-task architecture with shared MLP backbone and separate per-pathway
    prediction heads. Each head predicts whether a given chemical at a given
    concentration activates a specific biological pathway.

    Args:
        num_chem_classes: Number of chemical class categories for one-hot encoding.
        backbone_dims: Hidden dimensions of the shared backbone MLP.
        num_pathways: Number of biological pathways to predict.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        num_chem_classes: int = 50,
        backbone_dims: list[int] | None = None,
        num_pathways: int = NUM_PATHWAYS,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_chem_classes = num_chem_classes
        self.num_pathways = num_pathways

        # Shared backbone
        self.backbone = Chem2PathBackbone(
            num_chem_classes=num_chem_classes,
            backbone_dims=backbone_dims,
            dropout=dropout,
        )

        # Per-pathway heads
        self.pathway_heads = nn.ModuleList([
            PathwayHead(self.backbone.output_dim, pathway_name=name)
            for name in PATHWAYS[:num_pathways]
        ])

    def forward(
        self,
        chem_class: torch.Tensor,
        log_concentration: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Predict pathway activation profile.

        Args:
            chem_class: One-hot chemical class encoding [B, num_chem_classes].
            log_concentration: Log10 concentration values [B, 1] or [B].

        Returns:
            Dict with:
                'pathway_activation': Activation probabilities [B, num_pathways].
                'backbone_features': Shared features [B, 128].
        """
        # Ensure concentration has correct shape
        if log_concentration.dim() == 1:
            log_concentration = log_concentration.unsqueeze(-1)

        # Concatenate inputs
        x = torch.cat([chem_class, log_concentration], dim=-1)

        # Shared backbone
        features = self.backbone(x)

        # Per-pathway predictions
        activations = torch.cat(
            [head(features) for head in self.pathway_heads], dim=-1
        )  # [B, num_pathways]

        return {
            "pathway_activation": activations,
            "backbone_features": features,
        }

    def compute_loss(
        self,
        predictions: dict[str, torch.Tensor],
        targets: torch.Tensor,
        pathway_weights: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Multi-task BCE loss across all pathways.

        Args:
            predictions: Forward pass outputs.
            targets: Binary pathway activation targets [B, num_pathways].
            pathway_weights: Optional per-pathway loss weights [num_pathways].

        Returns:
            Dict with 'total' loss and per-pathway losses.
        """
        pred = predictions["pathway_activation"]

        if pathway_weights is None:
            pathway_weights = torch.ones(
                self.num_pathways, device=pred.device
            )

        losses = {}
        total = torch.tensor(0.0, device=pred.device)

        for i, name in enumerate(PATHWAYS[: self.num_pathways]):
            loss_i = F.binary_cross_entropy(
                pred[:, i], targets[:, i].float(), reduction="mean"
            )
            losses[f"pathway_{name}"] = loss_i
            total = total + pathway_weights[i] * loss_i

        losses["total"] = total / self.num_pathways
        return losses
