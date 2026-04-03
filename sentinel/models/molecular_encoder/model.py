"""Full molecular encoder combining Chem2Path and information bottleneck.

Provides the complete molecular modality encoder that:
1. Predicts pathway activation profiles from chemical identity + concentration
2. Identifies minimal biomarker gene panels via gated selection
3. Projects to shared 256-dim fusion embedding space
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .chem2path import Chem2Path, NUM_PATHWAYS
from .bottleneck import InformationBottleneck

SHARED_EMBED_DIM = 256


class MolecularEncoder(nn.Module):
    """Complete molecular modality encoder for SENTINEL.

    Combines the Chem2Path multi-task model (chemistry -> pathway activation)
    with the information bottleneck (full transcriptome -> minimal biomarker
    panel). Both feed into a shared projection space for fusion.

    Args:
        num_chem_classes: Number of chemical class categories.
        num_pathways: Number of biological pathways.
        gene_input_dim: Full transcriptome dimension.
        bottleneck_hidden_dim: Bottleneck classifier hidden dim.
        lambda_l1: L1 penalty for gene selection.
        shared_embed_dim: Shared fusion embedding dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        num_chem_classes: int = 50,
        num_pathways: int = NUM_PATHWAYS,
        gene_input_dim: int = 20000,
        bottleneck_hidden_dim: int = 256,
        lambda_l1: float = 0.01,
        shared_embed_dim: int = SHARED_EMBED_DIM,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_chem_classes = num_chem_classes
        self.num_pathways = num_pathways

        # Chem2Path model
        self.chem2path = Chem2Path(
            num_chem_classes=num_chem_classes,
            num_pathways=num_pathways,
            dropout=dropout,
        )

        # Information bottleneck for biomarker panel
        self.bottleneck = InformationBottleneck(
            input_dim=gene_input_dim,
            hidden_dim=bottleneck_hidden_dim,
            num_pathways=num_pathways,
            lambda_l1=lambda_l1,
            dropout=dropout,
        )

        # Fusion of Chem2Path backbone features + pathway predictions
        # Chem2Path backbone outputs 128-dim, pathway activation is 7-dim
        chem_feat_dim = 128 + num_pathways
        self.fusion_layer = nn.Sequential(
            nn.Linear(chem_feat_dim, shared_embed_dim),
            nn.GELU(),
            nn.LayerNorm(shared_embed_dim),
        )

        # Projection to shared embedding space
        self.projection = nn.Sequential(
            nn.Linear(shared_embed_dim, shared_embed_dim),
            nn.GELU(),
            nn.LayerNorm(shared_embed_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in [self.fusion_layer, self.projection]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(
        self,
        chem_class: torch.Tensor,
        log_concentration: torch.Tensor,
        gene_expression: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Full forward pass through the molecular encoder.

        Args:
            chem_class: One-hot chemical class encoding [B, num_chem_classes].
            log_concentration: Log10 concentration [B, 1] or [B].
            gene_expression: Optional full gene expression vector [B, gene_dim]
                for bottleneck analysis. If None, bottleneck outputs are omitted.

        Returns:
            Dict with:
                'pathway_activation': Predicted pathway probabilities [B, num_pathways].
                'embedding': Projected embedding [B, 256] for fusion.
                'fusion_embedding': Same as embedding.
                'selected_genes': Boolean mask of selected genes [gene_dim]
                    (only if gene_expression provided).
                'bottleneck_logits': Bottleneck pathway logits [B, num_pathways]
                    (only if gene_expression provided).
        """
        # Chem2Path forward
        chem_output = self.chem2path(chem_class, log_concentration)
        pathway_activation = chem_output["pathway_activation"]
        backbone_features = chem_output["backbone_features"]

        # Fuse backbone features with pathway predictions
        fused = torch.cat([backbone_features, pathway_activation], dim=-1)
        fused = self.fusion_layer(fused)

        # Project to shared space
        embedding = self.projection(fused)

        result: dict[str, torch.Tensor] = {
            "pathway_activation": pathway_activation,
            "embedding": embedding,
            "fusion_embedding": embedding,
        }

        # Bottleneck analysis if gene expression is provided
        if gene_expression is not None:
            bn_output = self.bottleneck(gene_expression)
            result["selected_genes"] = self.bottleneck.selection.get_selected_mask()
            result["bottleneck_logits"] = bn_output["logits"]
            result["bottleneck_gates"] = bn_output["gates"]
            result["num_selected_genes"] = bn_output["num_selected"]
        else:
            # Placeholder outputs
            result["selected_genes"] = torch.zeros(
                self.bottleneck.input_dim,
                dtype=torch.bool,
                device=chem_class.device,
            )

        return result

    def compute_loss(
        self,
        outputs: dict[str, torch.Tensor],
        pathway_targets: torch.Tensor,
        gene_expression: Optional[torch.Tensor] = None,
        bottleneck_targets: Optional[torch.Tensor] = None,
        chem2path_weight: float = 1.0,
        bottleneck_weight: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """Compute combined training losses.

        Args:
            outputs: Forward pass outputs.
            pathway_targets: Binary pathway activation targets [B, num_pathways].
            gene_expression: Original gene expression [B, gene_dim] (for bottleneck).
            bottleneck_targets: Pathway targets for bottleneck [B, num_pathways].
            chem2path_weight: Weight for Chem2Path loss.
            bottleneck_weight: Weight for bottleneck loss.

        Returns:
            Dict of loss components.
        """
        losses = {}

        # Chem2Path loss
        chem_losses = self.chem2path.compute_loss(
            {"pathway_activation": outputs["pathway_activation"]},
            pathway_targets,
        )
        losses["chem2path"] = chem_losses["total"]

        total = chem2path_weight * chem_losses["total"]

        # Bottleneck loss
        if "bottleneck_logits" in outputs and bottleneck_targets is not None:
            bn_outputs = {
                "logits": outputs["bottleneck_logits"],
                "gates": outputs["bottleneck_gates"],
                "num_selected": outputs["num_selected_genes"],
                "selected": None,  # Not needed for loss
            }
            bn_losses = self.bottleneck.compute_loss(bn_outputs, bottleneck_targets)
            losses["bottleneck"] = bn_losses["total"]
            losses["bottleneck_l1"] = bn_losses["l1_penalty"]
            losses["num_selected_genes"] = bn_losses["num_selected"]
            total = total + bottleneck_weight * bn_losses["total"]

        losses["total"] = total
        return losses
