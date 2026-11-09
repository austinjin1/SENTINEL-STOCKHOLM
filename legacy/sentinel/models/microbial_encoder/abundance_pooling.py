"""Abundance-weighted soft attention pooling for compositional microbiome data.

Combines phylogenetic-aware sequence embeddings ("who is there") with
CLR-transformed abundance information ("how much") through a learned
attention mechanism operating in Aitchison geometry. This produces a
single sample-level embedding that captures both taxonomic identity
and community composition.

Extends the Abundance-Aware Set Transformer concept with compositional
geometry awareness.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AbundanceWeightedPooling(nn.Module):
    """Soft attention pooling weighted by CLR-transformed abundances.

    For each sample, produces a single embedding by attending over the
    set of ASV sequence embeddings, where attention weights are modulated
    by the CLR-transformed abundance of each taxon.

    The attention weight for ASV j in sample i is:
        alpha_{i,j} = softmax( f(seq_embed_j) * clr_abundance_{i,j} )

    where f is a learned linear projection. This naturally upweights
    taxa that are both phylogenetically informative (high f score) and
    abundant (high CLR value), while downweighting rare or uninformative
    taxa.

    Args:
        seq_embed_dim: Dimension of per-ASV sequence embeddings. Default 256.
        embed_dim: Output embedding dimension. Default 256.
        n_attention_heads: Number of parallel attention heads for pooling.
            Default 4.
        dropout: Dropout rate. Default 0.1.
    """

    def __init__(
        self,
        seq_embed_dim: int = 256,
        embed_dim: int = 256,
        n_attention_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.seq_embed_dim = seq_embed_dim
        self.embed_dim = embed_dim
        self.n_attention_heads = n_attention_heads

        # Score function: maps each ASV embedding to per-head attention logits
        # These logits are then modulated by CLR abundances
        self.score_proj = nn.Linear(seq_embed_dim, n_attention_heads)

        # Abundance gating: transforms CLR abundance into a gating signal
        # CLR values can be negative (below geometric mean) or positive (above)
        self.abundance_gate = nn.Sequential(
            nn.Linear(1, n_attention_heads),
            nn.Tanh(),
        )

        # Value projection: maps sequence embeddings to per-head value vectors
        assert embed_dim % n_attention_heads == 0
        self.head_dim = embed_dim // n_attention_heads
        self.value_proj = nn.Linear(seq_embed_dim, embed_dim)

        # Output projection after concatenating heads
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier initialization for all linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        sequence_embeddings: torch.Tensor,
        clr_abundances: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pool ASV embeddings into a single sample embedding.

        Args:
            sequence_embeddings: Per-ASV phylogenetic embeddings [n_otus, seq_embed_dim].
                Shared across all samples in the batch (same OTU table).
            clr_abundances: CLR-transformed abundance vectors [B, n_otus].
            padding_mask: Boolean mask [B, n_otus] where True = ASV is absent/padded.
                If None, no masking is applied.

        Returns:
            Tuple of:
                - Sample embeddings [B, embed_dim].
                - Attention weights [B, n_otus] (averaged across heads, useful for
                  indicator species discovery).
        """
        B = clr_abundances.shape[0]
        n_otus = clr_abundances.shape[1]

        # Expand sequence embeddings across batch: [n_otus, seq_dim] -> [B, n_otus, seq_dim]
        if sequence_embeddings.dim() == 2:
            seq_emb = sequence_embeddings.unsqueeze(0).expand(B, -1, -1)
        else:
            seq_emb = sequence_embeddings  # already [B, n_otus, seq_dim]

        # Compute base attention logits from sequence content: [B, n_otus, n_heads]
        base_scores = self.score_proj(seq_emb)

        # Compute abundance gating signal: [B, n_otus, 1] -> [B, n_otus, n_heads]
        abundance_signal = self.abundance_gate(clr_abundances.unsqueeze(-1))

        # Combined attention logits: element-wise product modulates phylogenetic
        # importance by compositional abundance
        attn_logits = base_scores * abundance_signal  # [B, n_otus, n_heads]

        # Apply padding mask (mask out absent/padded ASVs)
        if padding_mask is not None:
            attn_logits = attn_logits.masked_fill(
                padding_mask.unsqueeze(-1), float("-inf")
            )

        # Softmax over OTU dimension
        attn_weights = F.softmax(attn_logits, dim=1)  # [B, n_otus, n_heads]

        # Value projection: [B, n_otus, embed_dim]
        values = self.value_proj(seq_emb)
        values = values.view(B, n_otus, self.n_attention_heads, self.head_dim)

        # Weighted sum per head: [B, n_heads, head_dim]
        # attn_weights: [B, n_otus, n_heads] -> [B, n_otus, n_heads, 1]
        weighted = (attn_weights.unsqueeze(-1) * values).sum(dim=1)  # [B, n_heads, head_dim]

        # Concatenate heads: [B, embed_dim]
        pooled = weighted.view(B, self.embed_dim)

        # Final projection
        output = self.output_proj(pooled)

        # Return averaged attention weights across heads for interpretability
        avg_attn = attn_weights.mean(dim=-1)  # [B, n_otus]

        return output, avg_attn
