"""P-NET-inspired biological hierarchy network for interpretable toxicology.

Mirrors the Adverse Outcome Pathway (AOP) framework with three sparse layers:
  Layer 1: Gene → Pathway (constrained by Reactome gene sets)
  Layer 2: Pathway → Biological Process (constrained by Reactome hierarchy)
  Layer 3: Biological Process → Adverse Outcome (constrained by AOP-Wiki)

All connections are biologically constrained — only known relationships
have learnable weights. This makes the network interpretable by design:
activation patterns at each layer directly correspond to known
toxicological mechanisms.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SparseConstrainedLinear(nn.Module):
    """Linear layer with a fixed binary mask enforcing biological constraints.

    Only connections supported by the adjacency matrix have learnable weights;
    all others are permanently zero. This is implemented by registering the
    mask as a buffer (not a parameter) and applying it on every forward pass.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        mask: Binary adjacency matrix [out_features, in_features] where 1
            indicates a biologically supported connection.
        bias: Whether to include a bias term.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        mask: torch.Tensor,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Store mask as buffer (not trainable, but moves with device)
        self.register_buffer("mask", mask.float())

        # Full weight matrix — masked entries never update
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        # Zero out masked positions at init for cleanliness
        with torch.no_grad():
            self.weight.mul_(self.mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply masked linear transformation.

        Args:
            x: Input tensor [B, in_features].

        Returns:
            Output tensor [B, out_features].
        """
        masked_weight = self.weight * self.mask
        return nn.functional.linear(x, masked_weight, self.bias)

    def extra_repr(self) -> str:
        nnz = int(self.mask.sum().item())
        total = self.in_features * self.out_features
        density = nnz / max(total, 1) * 100
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"connections={nnz}/{total} ({density:.1f}% dense), "
            f"bias={self.bias is not None}"
        )


class HierarchyNetwork(nn.Module):
    """Biological hierarchy network mirroring the AOP framework.

    Three sparse layers map from gene expression through pathways and
    biological processes to adverse outcomes. Each layer's connectivity is
    constrained by known biology (Reactome, AOP-Wiki).

    Args:
        gene_names: List of gene names corresponding to input dimensions.
        pathway_adj: Sparse adjacency matrix [n_pathways, n_genes] from
            Reactome gene sets. Entry (i, j) = 1 means gene j belongs to
            pathway i.
        process_adj: Sparse adjacency matrix [n_processes, n_pathways] from
            Reactome hierarchy. Entry (i, j) = 1 means pathway j contributes
            to biological process i.
        outcome_adj: Sparse adjacency matrix [n_outcomes, n_processes] from
            AOP-Wiki. Entry (i, j) = 1 means biological process j leads to
            adverse outcome i.
        dropout: Dropout rate for regularization.
        native_dim: Native feature dimension for output (default 128).
    """

    def __init__(
        self,
        gene_names: list[str],
        pathway_adj: torch.Tensor,
        process_adj: torch.Tensor,
        outcome_adj: torch.Tensor,
        dropout: float = 0.3,
        native_dim: int = 128,
    ) -> None:
        super().__init__()
        self.gene_names = gene_names
        self.n_genes = len(gene_names)
        self.n_pathways = pathway_adj.shape[0]
        self.n_processes = process_adj.shape[0]
        self.n_outcomes = outcome_adj.shape[0]
        self.native_dim = native_dim

        # Layer 1: Gene → Pathway (Reactome gene sets)
        self.gene_to_pathway = SparseConstrainedLinear(
            self.n_genes, self.n_pathways, mask=pathway_adj
        )
        self.pathway_bn = nn.BatchNorm1d(self.n_pathways)
        self.pathway_drop = nn.Dropout(dropout)

        # Layer 2: Pathway → Biological Process (Reactome hierarchy)
        self.pathway_to_process = SparseConstrainedLinear(
            self.n_pathways, self.n_processes, mask=process_adj
        )
        self.process_bn = nn.BatchNorm1d(self.n_processes)
        self.process_drop = nn.Dropout(dropout)

        # Layer 3: Biological Process → Adverse Outcome (AOP-Wiki)
        self.process_to_outcome = SparseConstrainedLinear(
            self.n_processes, self.n_outcomes, mask=outcome_adj
        )

        # Feature aggregation: concatenate pathway + process activations
        # and project to native_dim for downstream use
        agg_dim = self.n_pathways + self.n_processes
        self.feature_proj = nn.Sequential(
            nn.Linear(agg_dim, native_dim),
            nn.BatchNorm1d(native_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self._init_proj_weights()

    def _init_proj_weights(self) -> None:
        for m in self.feature_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, gene_expression: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the biological hierarchy.

        Args:
            gene_expression: Gene expression values [B, n_genes].

        Returns:
            Tuple of:
                features: Aggregated hierarchy features [B, native_dim].
                pathway_activations: Pathway-level activations [B, n_pathways].
                outcome_logits: Adverse outcome logits [B, n_outcomes].
        """
        # Layer 1: Genes → Pathways
        pathway_act = self.gene_to_pathway(gene_expression)
        pathway_act = self.pathway_bn(pathway_act)
        pathway_act = torch.relu(pathway_act)
        pathway_act = self.pathway_drop(pathway_act)

        # Layer 2: Pathways → Biological Processes
        process_act = self.pathway_to_process(pathway_act)
        process_act = self.process_bn(process_act)
        process_act = torch.relu(process_act)
        process_act = self.process_drop(process_act)

        # Layer 3: Biological Processes → Adverse Outcomes
        outcome_logits = self.process_to_outcome(process_act)

        # Aggregate features from pathway and process layers
        aggregated = torch.cat([pathway_act, process_act], dim=-1)
        features = self.feature_proj(aggregated)

        return features, pathway_act, outcome_logits
