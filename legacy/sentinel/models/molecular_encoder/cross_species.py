"""Cross-species transfer learning via ortholog mapping.

Enables transfer learning between species (e.g., data-rich zebrafish to
data-poor Daphnia) by mapping species-specific gene expression into a
shared ortholog group space before feeding into the hierarchy network.

Ortholog mappings come from the data pipeline (e.g., Ensembl Compara,
OrthoFinder) and define which genes across species correspond to the
same ancestral gene.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .hierarchy_network import HierarchyNetwork


class OrthologAligner(nn.Module):
    """Maps species-specific gene expression to shared ortholog group space.

    Uses a sparse projection matrix derived from ortholog mapping tables.
    For one-to-many or many-to-many orthologs, expression values are
    averaged across the group.

    Args:
        n_species_genes: Number of genes in the species-specific space.
        n_ortholog_groups: Number of shared ortholog groups.
        mapping_matrix: Sparse projection matrix [n_ortholog_groups, n_species_genes]
            where entry (i, j) = 1/k means species gene j maps to ortholog
            group i (k = number of species genes mapping to that group, for
            averaging).
        species_name: Name of the species (for logging/identification).
    """

    def __init__(
        self,
        n_species_genes: int,
        n_ortholog_groups: int,
        mapping_matrix: torch.Tensor,
        species_name: str = "unknown",
    ) -> None:
        super().__init__()
        self.n_species_genes = n_species_genes
        self.n_ortholog_groups = n_ortholog_groups
        self.species_name = species_name

        # Fixed projection (not learnable — defined by biology)
        self.register_buffer("projection", mapping_matrix.float())

        # Learnable species-specific scaling: accounts for platform effects,
        # expression range differences, etc.
        self.species_scale = nn.Parameter(torch.ones(n_species_genes))
        self.species_bias = nn.Parameter(torch.zeros(n_species_genes))

    def forward(self, gene_expression: torch.Tensor) -> torch.Tensor:
        """Map species-specific expression to ortholog space.

        Args:
            gene_expression: Species gene expression [B, n_species_genes].

        Returns:
            Ortholog group expression [B, n_ortholog_groups].
        """
        # Apply species-specific normalization
        normalized = gene_expression * self.species_scale + self.species_bias

        # Project to shared ortholog space
        ortholog_expr = nn.functional.linear(normalized, self.projection)

        return ortholog_expr

    def extra_repr(self) -> str:
        nnz = int((self.projection != 0).sum().item())
        coverage = int((self.projection.sum(dim=1) > 0).sum().item())
        return (
            f"species={self.species_name}, "
            f"genes={self.n_species_genes} -> orthologs={self.n_ortholog_groups}, "
            f"mappings={nnz}, coverage={coverage}/{self.n_ortholog_groups}"
        )


class CrossSpeciesEncoder(nn.Module):
    """Wraps HierarchyNetwork with species-specific input adapters.

    For each registered species, gene expression is first mapped to a
    shared ortholog space, then fed through the common hierarchy network.
    This enables training on multi-species data and transfer learning
    from data-rich to data-poor species.

    Args:
        ortholog_gene_names: List of gene names in the shared ortholog space
            (these become the hierarchy network's gene inputs).
        pathway_adj: Pathway adjacency for HierarchyNetwork [n_pathways, n_ortholog_groups].
        process_adj: Process adjacency for HierarchyNetwork [n_processes, n_pathways].
        outcome_adj: Outcome adjacency for HierarchyNetwork [n_outcomes, n_processes].
        dropout: Dropout rate.
        native_dim: Native feature dimension (default 128).
    """

    def __init__(
        self,
        ortholog_gene_names: list[str],
        pathway_adj: torch.Tensor,
        process_adj: torch.Tensor,
        outcome_adj: torch.Tensor,
        dropout: float = 0.3,
        native_dim: int = 128,
    ) -> None:
        super().__init__()
        self.n_ortholog_groups = len(ortholog_gene_names)
        self.native_dim = native_dim

        # Shared hierarchy network operates in ortholog space
        self.hierarchy = HierarchyNetwork(
            gene_names=ortholog_gene_names,
            pathway_adj=pathway_adj,
            process_adj=process_adj,
            outcome_adj=outcome_adj,
            dropout=dropout,
            native_dim=native_dim,
        )

        # Species-specific input adapters
        self.aligners: nn.ModuleDict = nn.ModuleDict()

    def register_species(
        self,
        species_name: str,
        n_species_genes: int,
        mapping_matrix: torch.Tensor,
    ) -> None:
        """Register a new species with its ortholog mapping.

        Args:
            species_name: Unique species identifier (e.g., "danio_rerio",
                "daphnia_magna").
            n_species_genes: Number of genes measured for this species.
            mapping_matrix: Ortholog projection [n_ortholog_groups, n_species_genes].
        """
        aligner = OrthologAligner(
            n_species_genes=n_species_genes,
            n_ortholog_groups=self.n_ortholog_groups,
            mapping_matrix=mapping_matrix,
            species_name=species_name,
        )
        self.aligners[species_name] = aligner

    def forward(
        self,
        gene_expression: torch.Tensor,
        species: str,
        ortholog_expression: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with species-specific input adaptation.

        Two modes:
        1. Species-specific input: provide gene_expression + species name.
           Expression is mapped through the OrthologAligner first.
        2. Pre-aligned input: provide ortholog_expression directly (already
           in shared ortholog space). Species is ignored.

        Args:
            gene_expression: Species-specific gene expression [B, n_species_genes].
                Ignored if ortholog_expression is provided.
            species: Species identifier matching a registered aligner.
            ortholog_expression: Optional pre-aligned expression in ortholog
                space [B, n_ortholog_groups].

        Returns:
            Tuple of:
                features: Hierarchy features [B, native_dim].
                pathway_activations: [B, n_pathways].
                outcome_logits: [B, n_outcomes].

        Raises:
            KeyError: If species is not registered and no ortholog_expression given.
        """
        if ortholog_expression is not None:
            aligned = ortholog_expression
        else:
            if species not in self.aligners:
                raise KeyError(
                    f"Species '{species}' not registered. Available: "
                    f"{list(self.aligners.keys())}"
                )
            aligned = self.aligners[species](gene_expression)

        return self.hierarchy(aligned)

    @property
    def registered_species(self) -> list[str]:
        """List of registered species names."""
        return list(self.aligners.keys())
