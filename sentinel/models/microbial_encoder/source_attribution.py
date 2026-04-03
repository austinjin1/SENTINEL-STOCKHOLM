"""Source attribution transformer for microbial community profiling.

Takes CLR-transformed ASV abundance vectors and uses self-attention to
learn taxon-taxon co-occurrence patterns for contamination source
classification. Attention weights enable indicator species discovery.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Maximum number of ASV features (amplicon sequence variants)
MAX_ASV_FEATURES = 5000
EMBED_DIM = 256

# Contamination source types
CONTAMINATION_SOURCES = [
    "nutrient",
    "heavy_metals",
    "thermal",
    "pharmaceutical",
    "sediment",
    "oil_petrochemical",
    "sewage",
    "acid_mine",
]
NUM_SOURCES = len(CONTAMINATION_SOURCES)


class ASVEmbedding(nn.Module):
    """Embedding layer for CLR-transformed ASV abundance vectors.

    Projects high-dimensional sparse ASV vectors to a dense embedding
    suitable for transformer processing. Uses a linear projection with
    learned positional embeddings per ASV position.

    Args:
        input_dim: Number of ASV features. Default 5000.
        embed_dim: Output embedding dimension. Default 256.
        dropout: Dropout rate. Default 0.1.
    """

    def __init__(
        self,
        input_dim: int = MAX_ASV_FEATURES,
        embed_dim: int = EMBED_DIM,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        # Project each ASV value to embed_dim
        self.value_projection = nn.Linear(1, embed_dim)

        # Learned positional embedding for each ASV position
        self.position_embedding = nn.Embedding(input_dim, embed_dim)

        # Learnable [CLS] token for aggregation
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed ASV abundance vector.

        Args:
            x: CLR-transformed ASV abundances [B, input_dim].

        Returns:
            Embedded sequence [B, input_dim + 1, embed_dim] with CLS prepended.
        """
        B, N = x.shape

        # Project each value: [B, N, 1] -> [B, N, embed_dim]
        values = self.value_projection(x.unsqueeze(-1))

        # Add positional embeddings
        positions = torch.arange(N, device=x.device)
        pos_emb = self.position_embedding(positions)  # [N, embed_dim]
        values = values + pos_emb.unsqueeze(0)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        sequence = torch.cat([cls_tokens, values], dim=1)  # [B, N+1, D]

        return self.dropout(self.norm(sequence))


class SourceAttributionTransformer(nn.Module):
    """Transformer encoder for microbial contamination source attribution.

    Self-attention over ASV positions learns taxon-taxon co-occurrence
    patterns that indicate specific contamination sources. The attention
    weights can be extracted to identify indicator species.

    Args:
        input_dim: Number of ASV features. Default 5000.
        embed_dim: Transformer embedding dimension. Default 256.
        num_heads: Number of attention heads. Default 4.
        num_layers: Number of transformer layers. Default 4.
        ff_dim: Feed-forward hidden dimension. Default 512.
        dropout: Dropout rate. Default 0.1.
        num_sources: Number of contamination source types. Default 8.
    """

    def __init__(
        self,
        input_dim: int = MAX_ASV_FEATURES,
        embed_dim: int = EMBED_DIM,
        num_heads: int = 4,
        num_layers: int = 4,
        ff_dim: int = 512,
        dropout: float = 0.1,
        num_sources: int = NUM_SOURCES,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        # ASV embedding layer
        self.embedding = ASVEmbedding(input_dim, embed_dim, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Classification head for contamination sources
        self.classification_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_sources),
        )

        # Store attention weights for indicator species extraction
        self._attention_weights: list[torch.Tensor] = []
        self._register_attention_hooks()

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # Re-initialize CLS token
        nn.init.trunc_normal_(self.embedding.cls_token, std=0.02)

    def _register_attention_hooks(self) -> None:
        """Register hooks to capture attention weights from each layer."""
        for layer in self.transformer.layers:
            layer.self_attn.register_forward_hook(self._attention_hook)

    def _attention_hook(self, module, input, output):
        """Capture attention weights from self-attention layers."""
        # For nn.MultiheadAttention with batch_first=True,
        # output is (attn_output, attn_weights) when need_weights=True
        # But by default need_weights=False in TransformerEncoderLayer.
        # We store what we can; full extraction done in get_attention_weights()
        pass

    def get_attention_weights(
        self, x: torch.Tensor
    ) -> list[torch.Tensor]:
        """Extract attention weights by running forward with need_weights=True.

        Args:
            x: CLR-transformed ASV abundances [B, input_dim].

        Returns:
            List of attention weight tensors [B, num_heads, seq_len, seq_len],
            one per transformer layer.
        """
        self.eval()
        attention_weights = []

        embedded = self.embedding(x)

        # Manually run through transformer layers to capture attention
        hidden = embedded
        for layer in self.transformer.layers:
            # Pre-norm
            normed = layer.norm1(hidden)
            # Self-attention with attention weights
            attn_out, attn_weights = layer.self_attn(
                normed, normed, normed, need_weights=True, average_attn_weights=False
            )
            attention_weights.append(attn_weights.detach())
            # Residual + FF
            hidden = hidden + attn_out
            hidden = hidden + layer._ff_block(layer.norm2(hidden))

        return attention_weights

    @torch.no_grad()
    def get_indicator_species_weights(
        self,
        x: torch.Tensor,
        layer_idx: int = -1,
    ) -> torch.Tensor:
        """Extract per-ASV importance weights via CLS attention.

        The attention from the CLS token to each ASV position indicates
        how important that taxon is for the classification decision.

        Args:
            x: CLR-transformed ASV abundances [B, input_dim].
            layer_idx: Which transformer layer's attention to use (-1 = last).

        Returns:
            Per-ASV importance weights [B, input_dim].
        """
        attn_weights = self.get_attention_weights(x)
        # Use specified layer
        attn = attn_weights[layer_idx]  # [B, heads, seq, seq]

        # Average over heads
        attn = attn.mean(dim=1)  # [B, seq, seq]

        # CLS token attention to ASV positions (CLS is at position 0)
        cls_attention = attn[:, 0, 1:]  # [B, input_dim]

        return cls_attention

    def forward(
        self, x: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Forward pass for source attribution.

        Args:
            x: CLR-transformed ASV abundances [B, input_dim].

        Returns:
            Dict with:
                'source_logits': Classification logits [B, num_sources].
                'source_probs': Softmax probabilities [B, num_sources].
                'embedding': CLS token embedding [B, embed_dim].
        """
        # Embed ASV vector
        embedded = self.embedding(x)  # [B, N+1, D]

        # Transformer encoding
        encoded = self.transformer(embedded)  # [B, N+1, D]

        # CLS token output
        cls_output = encoded[:, 0, :]  # [B, D]

        # Classification
        logits = self.classification_head(cls_output)
        probs = F.softmax(logits, dim=-1)

        return {
            "source_logits": logits,
            "source_probs": probs,
            "embedding": cls_output,
        }
