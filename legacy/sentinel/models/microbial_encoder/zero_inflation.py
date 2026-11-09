"""Zero-inflation gating for structural vs sampling zeros in microbiome data.

Microbiome abundance tables are extremely sparse, but not all zeros are equal:
- Structural zeros: the taxon is truly absent from the environment (e.g.,
  a marine organism in a freshwater sample). These should remain zero.
- Sampling zeros: the taxon is present but undetected due to insufficient
  sequencing depth or PCR bias. These should be imputed.

This module uses a neural network conditioned on the non-zero taxa in each
sample to classify each zero entry and impute sampling zeros with a learned
prior.

References:
    Kaul et al. (2017). Analysis of Microbiome Data in the Presence of
        Excess Zeros. Frontiers in Microbiology.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ZeroInflationGate(nn.Module):
    """Neural gating mechanism for classifying and handling structural vs sampling zeros.

    For each zero entry in the abundance vector, predicts the probability that
    it is a structural zero (taxon truly absent) vs a sampling zero (missed
    detection). Sampling zeros are imputed with a learned prior conditioned
    on the co-occurring taxa, while structural zeros are masked to zero.

    Architecture:
        1. Context encoder: summarizes the non-zero taxa profile into a context vector.
        2. Zero classifier: predicts P(structural | context, position) per zero.
        3. Imputation network: generates imputed values for sampling zeros.

    Args:
        n_otus: Number of OTUs/ASVs. Default 5000.
        context_dim: Hidden dimension for context encoding. Default 256.
        dropout: Dropout rate. Default 0.1.
        structural_threshold: Probability threshold above which a zero is
            classified as structural. Default 0.5.
    """

    def __init__(
        self,
        n_otus: int = 5000,
        context_dim: int = 256,
        dropout: float = 0.1,
        structural_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.n_otus = n_otus
        self.context_dim = context_dim
        self.structural_threshold = structural_threshold

        # Context encoder: summarizes the non-zero portion of each sample
        # into a dense context vector capturing the community signature
        self.context_encoder = nn.Sequential(
            nn.Linear(n_otus, context_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(context_dim, context_dim),
            nn.LayerNorm(context_dim),
        )

        # Zero type classifier: for each OTU, predict P(structural zero)
        # conditioned on the community context
        # Uses a bilinear interaction: context vector x per-OTU embedding
        self.otu_embeddings = nn.Parameter(
            torch.randn(n_otus, context_dim) * 0.02
        )
        self.classifier_bias = nn.Parameter(torch.zeros(n_otus))

        # Imputation network: generates pseudo-counts for sampling zeros
        # conditioned on community context
        self.imputation_net = nn.Sequential(
            nn.Linear(context_dim, context_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(context_dim, n_otus),
        )

        # Learnable temperature for imputation magnitude
        self.imputation_scale = nn.Parameter(torch.tensor(0.1))

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier init for linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        raw_abundances: torch.Tensor,
        return_details: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Classify zeros and impute sampling zeros.

        Args:
            raw_abundances: Raw (non-CLR) abundance vector [B, n_otus].
                Values >= 0, with many zeros.
            return_details: If True, returns additional diagnostic tensors.

        Returns:
            Tuple of:
                - imputed_abundances: Abundance vector with sampling zeros
                  imputed [B, n_otus].
                - zero_type_mask: Per-entry classification [B, n_otus] where
                  1.0 = structural zero (hard masked), 0.0 = non-zero or
                  imputed sampling zero.
        """
        B, D = raw_abundances.shape

        # Identify zero entries
        is_zero = (raw_abundances == 0).float()  # [B, D]
        is_nonzero = 1.0 - is_zero

        # Encode community context from non-zero taxa
        # Use presence/absence weighted by abundance as input
        context = self.context_encoder(raw_abundances)  # [B, context_dim]

        # Classify each zero: P(structural_zero | context, otu_position)
        # Bilinear: context [B, context_dim] @ otu_embeddings^T [context_dim, D] + bias
        structural_logits = (
            torch.matmul(context, self.otu_embeddings.t()) + self.classifier_bias
        )  # [B, D]
        structural_prob = torch.sigmoid(structural_logits)  # [B, D]

        # Only classify zero entries; non-zero entries are by definition not structural zeros
        structural_prob = structural_prob * is_zero

        # Hard decision: classify as structural if P > threshold
        structural_mask = (structural_prob > self.structural_threshold).float()

        # Sampling zeros: zeros that are NOT structural
        sampling_zero_mask = is_zero * (1.0 - structural_mask)  # [B, D]

        # Generate imputed values for sampling zeros
        imputed_values = self.imputation_net(context)  # [B, D]
        # Use softplus to ensure positive imputed values, scaled down
        imputed_values = F.softplus(imputed_values) * self.imputation_scale.abs()

        # Combine: keep original non-zero values, impute sampling zeros, mask structural zeros
        imputed_abundances = (
            raw_abundances * is_nonzero  # original non-zero values
            + imputed_values * sampling_zero_mask  # imputed sampling zeros
            # structural zeros remain 0 (not added)
        )

        # zero_type_mask: 1.0 where structural zero, 0.0 elsewhere
        zero_type_mask = structural_mask

        return imputed_abundances, zero_type_mask

    @torch.no_grad()
    def get_zero_statistics(
        self, raw_abundances: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Compute diagnostic statistics about zero classification.

        Args:
            raw_abundances: Raw abundance vector [B, n_otus].

        Returns:
            Dict with:
                - 'n_zeros': Total zeros per sample [B].
                - 'n_structural': Structural zeros per sample [B].
                - 'n_sampling': Sampling zeros per sample [B].
                - 'structural_fraction': Fraction of zeros that are structural [B].
                - 'structural_prob': Per-entry structural probability [B, n_otus].
        """
        is_zero = (raw_abundances == 0).float()
        context = self.context_encoder(raw_abundances)
        structural_logits = (
            torch.matmul(context, self.otu_embeddings.t()) + self.classifier_bias
        )
        structural_prob = torch.sigmoid(structural_logits) * is_zero
        structural_mask = (structural_prob > self.structural_threshold).float()

        n_zeros = is_zero.sum(dim=1)
        n_structural = structural_mask.sum(dim=1)
        n_sampling = n_zeros - n_structural

        return {
            "n_zeros": n_zeros,
            "n_structural": n_structural,
            "n_sampling": n_sampling,
            "structural_fraction": n_structural / n_zeros.clamp(min=1),
            "structural_prob": structural_prob,
        }
