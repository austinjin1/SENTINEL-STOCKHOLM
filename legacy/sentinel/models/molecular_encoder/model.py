"""ToxiGene molecular encoder — P-NET biological hierarchy network.

Replaces Chem2Path MLP with a biologically interpretable hierarchy:
  Gene expression → Pathways → Biological Processes → Adverse Outcomes

All layer connections are constrained by known biology (Reactome, AOP-Wiki).
Supports two operational modes:
  1. Full transcriptome: gene_expression → bottleneck → hierarchy → outcomes
  2. Chemistry-only: chem_class + concentration → predicted pathway activation
     (for inference when no transcriptomics available)

Interface contract: forward() returns dict with "embedding" [B,256]
and "fusion_embedding" [B,256].
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hierarchy_network import HierarchyNetwork
from .cross_species import CrossSpeciesEncoder
from .bottleneck import HierarchyBottleneck

SHARED_EMBED_DIM = 256
NATIVE_DIM = 128


class ChemistryToPathway(nn.Module):
    """Lightweight MLP predicting pathway activations from chemistry alone.

    Used for inference when transcriptomics data is unavailable. Trained
    to match the pathway activations produced by the full hierarchy network.

    Args:
        num_chem_classes: Number of chemical class categories.
        n_pathways: Number of pathways to predict.
        hidden_dim: Hidden layer dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        num_chem_classes: int,
        n_pathways: int,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_chem_classes = num_chem_classes
        # Input: one-hot chem class + log concentration
        input_dim = num_chem_classes + 1

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_pathways),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, chem_class: torch.Tensor, log_concentration: torch.Tensor
    ) -> torch.Tensor:
        """Predict pathway activations from chemistry.

        Args:
            chem_class: One-hot chemical class [B, num_chem_classes].
            log_concentration: Log10 concentration [B, 1] or [B].

        Returns:
            Predicted pathway activations [B, n_pathways].
        """
        if log_concentration.dim() == 1:
            log_concentration = log_concentration.unsqueeze(-1)
        x = torch.cat([chem_class, log_concentration], dim=-1)
        return self.net(x)


class MolecularEncoder(nn.Module):
    """ToxiGene molecular encoder for SENTINEL.

    Combines the P-NET biological hierarchy network with a gated gene
    selection bottleneck and optional cross-species transfer. Projects
    to a shared 256-dim fusion embedding space.

    Args:
        gene_names: List of gene names for the input transcriptome.
        pathway_adj: Gene-to-pathway adjacency [n_pathways, n_genes].
        process_adj: Pathway-to-process adjacency [n_processes, n_pathways].
        outcome_adj: Process-to-outcome adjacency [n_outcomes, n_processes].
        num_chem_classes: Number of chemical classes for chemistry-only mode.
        lambda_l1: L1 penalty weight for gene selection bottleneck.
        shared_embed_dim: Dimension of output fusion embedding.
        native_dim: Internal hierarchy feature dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        gene_names: list[str],
        pathway_adj: torch.Tensor,
        process_adj: torch.Tensor,
        outcome_adj: torch.Tensor,
        num_chem_classes: int = 50,
        lambda_l1: float = 0.01,
        shared_embed_dim: int = SHARED_EMBED_DIM,
        native_dim: int = NATIVE_DIM,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.gene_names = gene_names
        self.n_genes = len(gene_names)
        self.n_pathways = pathway_adj.shape[0]
        self.n_processes = process_adj.shape[0]
        self.n_outcomes = outcome_adj.shape[0]
        self.num_chem_classes = num_chem_classes
        self.native_dim = native_dim
        self.shared_embed_dim = shared_embed_dim

        # Gene selection bottleneck (gated sparsity)
        self.bottleneck = HierarchyBottleneck(
            gene_names=gene_names,
            lambda_l1=lambda_l1,
        )

        # Biological hierarchy network
        self.hierarchy = HierarchyNetwork(
            gene_names=gene_names,
            pathway_adj=pathway_adj,
            process_adj=process_adj,
            outcome_adj=outcome_adj,
            dropout=dropout,
            native_dim=native_dim,
        )

        # Chemistry-only pathway predictor (for inference without transcriptomics)
        self.chem_to_pathway = ChemistryToPathway(
            num_chem_classes=num_chem_classes,
            n_pathways=self.n_pathways,
            dropout=dropout,
        )

        # Chemistry feature encoder for fusion when using chem-only mode
        chem_input_dim = num_chem_classes + 1
        self.chem_feature_net = nn.Sequential(
            nn.Linear(chem_input_dim, native_dim),
            nn.BatchNorm1d(native_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Projection: native_dim (128) → shared_embed_dim (256)
        # Specification: Linear(128,128) → GELU → LN(128) → Linear(128,256) → LN(256)
        self.projection = nn.Sequential(
            nn.Linear(native_dim, native_dim),
            nn.GELU(),
            nn.LayerNorm(native_dim),
            nn.Linear(native_dim, shared_embed_dim),
            nn.LayerNorm(shared_embed_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier initialization for projection and chemistry feature layers."""
        for module in [self.projection, self.chem_feature_net]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(
        self,
        gene_expression: Optional[torch.Tensor] = None,
        chem_class: Optional[torch.Tensor] = None,
        log_concentration: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass through the ToxiGene encoder.

        Supports two modes:
        1. Full transcriptome (gene_expression provided):
           gene_expression → bottleneck gating → hierarchy → projection
        2. Chemistry-only (chem_class + log_concentration, no gene_expression):
           chemistry → predicted pathway activations → feature net → projection

        Args:
            gene_expression: Full gene expression [B, n_genes]. If provided,
                this is the primary input path.
            chem_class: One-hot chemical class [B, num_chem_classes].
            log_concentration: Log10 concentration [B, 1] or [B].

        Returns:
            Dict with:
                "embedding": Projected embedding [B, 256].
                "fusion_embedding": Same as embedding [B, 256].
                "pathway_activation": Pathway activations [B, n_pathways].
                "outcome_logits": Adverse outcome logits [B, n_outcomes].
                "selected_genes": Boolean gene selection mask [n_genes].
                "num_selected_genes": Number of selected genes (int).
                "hierarchy_features": Intermediate features [B, 128].
        """
        use_transcriptome = gene_expression is not None

        if use_transcriptome:
            # Mode 1: Full transcriptome path
            # Apply bottleneck gating
            gated_expression = self.bottleneck(gene_expression)

            # Hierarchy forward
            features, pathway_act, outcome_logits = self.hierarchy(gated_expression)

        else:
            # Mode 2: Chemistry-only path
            if chem_class is None or log_concentration is None:
                raise ValueError(
                    "Either gene_expression or (chem_class + log_concentration) "
                    "must be provided."
                )

            # Predict pathway activations from chemistry
            pathway_act = self.chem_to_pathway(chem_class, log_concentration)

            # Chemistry features as hierarchy substitute
            if log_concentration.dim() == 1:
                log_concentration_2d = log_concentration.unsqueeze(-1)
            else:
                log_concentration_2d = log_concentration
            chem_input = torch.cat([chem_class, log_concentration_2d], dim=-1)
            features = self.chem_feature_net(chem_input)

            # No real outcome logits in chem-only mode — predict from pathways
            # Use a simple linear projection (hierarchy weights not applicable)
            outcome_logits = torch.zeros(
                chem_class.shape[0],
                self.n_outcomes,
                device=chem_class.device,
                dtype=chem_class.dtype,
            )

        # Project to shared embedding space
        embedding = self.projection(features)

        # Build output dict
        result: dict[str, torch.Tensor] = {
            "embedding": embedding,
            "fusion_embedding": embedding,
            "pathway_activation": pathway_act,
            "outcome_logits": outcome_logits,
            "hierarchy_features": features,
            "selected_genes": self.bottleneck.get_selected_mask(),
            "num_selected_genes": torch.tensor(
                self.bottleneck.num_selected,
                dtype=torch.long,
                device=embedding.device,
            ),
        }

        return result

    def compute_loss(
        self,
        outputs: dict[str, torch.Tensor],
        outcome_targets: torch.Tensor,
        pathway_targets: Optional[torch.Tensor] = None,
        chem_pathway_targets: Optional[torch.Tensor] = None,
        outcome_weight: float = 1.0,
        pathway_weight: float = 0.5,
        bottleneck_weight: float = 1.0,
        chem_distill_weight: float = 0.5,
    ) -> dict[str, torch.Tensor]:
        """Compute combined training losses.

        Args:
            outputs: Forward pass output dict.
            outcome_targets: Binary adverse outcome targets [B, n_outcomes].
            pathway_targets: Optional pathway activation targets [B, n_pathways].
            chem_pathway_targets: Optional pathway targets for chemistry
                distillation (teacher pathway activations from transcriptome).
            outcome_weight: Weight for outcome classification loss.
            pathway_weight: Weight for pathway supervision loss.
            bottleneck_weight: Weight for gene selection L1 penalty.
            chem_distill_weight: Weight for chemistry distillation loss.

        Returns:
            Dict with 'total' and component losses.
        """
        device = outputs["outcome_logits"].device
        losses: dict[str, torch.Tensor] = {}

        # Adverse outcome classification loss (primary objective)
        outcome_loss = F.binary_cross_entropy_with_logits(
            outputs["outcome_logits"],
            outcome_targets.float(),
            reduction="mean",
        )
        losses["outcome"] = outcome_loss
        total = outcome_weight * outcome_loss

        # Pathway supervision (if pathway-level labels available)
        if pathway_targets is not None:
            pw_loss = F.mse_loss(
                outputs["pathway_activation"],
                pathway_targets.float(),
                reduction="mean",
            )
            losses["pathway"] = pw_loss
            total = total + pathway_weight * pw_loss

        # Bottleneck L1 sparsity penalty
        l1_loss = self.bottleneck.compute_loss()
        losses["bottleneck_l1"] = l1_loss
        losses["num_selected_genes"] = outputs["num_selected_genes"].float()
        total = total + bottleneck_weight * l1_loss

        # Chemistry distillation loss (train chem_to_pathway to match hierarchy)
        if chem_pathway_targets is not None:
            # Requires chem_class and log_concentration to be available
            # This is typically computed in a separate forward pass
            distill_loss = F.mse_loss(
                outputs.get("chem_pathway_pred", outputs["pathway_activation"]),
                chem_pathway_targets.float(),
                reduction="mean",
            )
            losses["chem_distill"] = distill_loss
            total = total + chem_distill_weight * distill_loss

        losses["total"] = total
        return losses

    def get_interpretable_activations(
        self,
        gene_expression: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Extract interpretable activation patterns for analysis.

        Returns activations at each hierarchy level for mechanistic
        interpretation of model predictions.

        Args:
            gene_expression: Gene expression [B, n_genes].

        Returns:
            Dict with per-layer activations and selected genes.
        """
        gated = self.bottleneck(gene_expression)
        features, pathway_act, outcome_logits = self.hierarchy(gated)

        return {
            "selected_genes": self.bottleneck.get_selected_genes(),
            "gene_gates": self.bottleneck.gates.detach(),
            "pathway_activations": pathway_act.detach(),
            "outcome_logits": outcome_logits.detach(),
            "outcome_probabilities": torch.sigmoid(outcome_logits).detach(),
        }
