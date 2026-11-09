"""MicroBiomeNet: Aitchison-aware compositional deep learning encoder.

Complete microbial modality encoder for SENTINEL that replaces the original
Transformer+VAE architecture with a compositionally-aware pipeline:

1. Zero-inflation gating: classify and impute structural vs sampling zeros
2. DNABERT-S sequence encoding: phylogenetic-aware ASV embeddings
3. Abundance-weighted pooling: combine "who" with "how much" in CLR space
4. Aitchison transformer: self-attention respecting compositional geometry
5. Simplex Neural ODE: temporal trajectory modeling for health assessment
6. Source attribution: contamination source classification

Interface contract:
    forward() returns dict with "embedding" [B, 256] and "fusion_embedding" [B, 256]
    Projection: Linear(256,256) -> GELU -> LayerNorm(256) -> Linear(256,256) -> LayerNorm(256)
    Xavier init. Native dim: 256.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aitchison_attention import AitchisonTransformerLayer, clr_transform
from .sequence_encoder import DNABERTSequenceEncoder
from .abundance_pooling import AbundanceWeightedPooling
from .zero_inflation import ZeroInflationGate
from .simplex_ode import SimplexNeuralODE

# Constants
MAX_ASV_FEATURES = 5000
EMBED_DIM = 256
NUM_SOURCES = 8
SHARED_EMBED_DIM = 256

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


class MicrobialEncoder(nn.Module):
    """MicroBiomeNet: Aitchison-aware compositional deep learning encoder.

    Replaces the original Transformer+VAE with a compositionally-aware
    architecture that respects the geometry of microbial abundance data
    (Aitchison simplex geometry) at every stage.

    Pipeline:
        raw_abundances -> ZeroInflationGate -> CLR transform
            -> DNABERTSequenceEncoder (per-ASV phylogenetic embeddings)
            -> AbundanceWeightedPooling (sample embedding)
            -> AitchisonTransformer (compositional self-attention)
            -> SimplexNeuralODE (temporal trajectory / health scoring)
            -> Source attribution head
            -> Projection to 256-dim shared fusion space

    Args:
        input_dim: Number of ASV features. Default 5000.
        embed_dim: Internal embedding dimension. Default 256.
        num_heads: Attention heads for Aitchison transformer. Default 4.
        num_aitchison_layers: Number of Aitchison transformer layers. Default 4.
        ff_dim: Feed-forward hidden dimension. Default 512.
        dropout: Dropout rate. Default 0.1.
        shared_embed_dim: Shared fusion embedding dimension. Default 256.
        num_sources: Number of contamination source types. Default 8.
        freeze_dnabert: Whether to freeze DNABERT-S weights. Default True.
    """

    def __init__(
        self,
        input_dim: int = MAX_ASV_FEATURES,
        embed_dim: int = EMBED_DIM,
        num_heads: int = 4,
        num_aitchison_layers: int = 4,
        ff_dim: int = 512,
        dropout: float = 0.1,
        shared_embed_dim: int = SHARED_EMBED_DIM,
        num_sources: int = NUM_SOURCES,
        freeze_dnabert: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_sources = num_sources

        # --- Stage 1: Zero-inflation gating ---
        self.zero_gate = ZeroInflationGate(
            n_otus=input_dim,
            context_dim=embed_dim,
            dropout=dropout,
        )

        # --- Stage 2: DNABERT-S sequence encoder ---
        self.sequence_encoder = DNABERTSequenceEncoder(
            output_dim=embed_dim,
            max_otus=input_dim,
            freeze_backbone=freeze_dnabert,
        )

        # --- Stage 3: Abundance-weighted pooling ---
        self.abundance_pooling = AbundanceWeightedPooling(
            seq_embed_dim=embed_dim,
            embed_dim=embed_dim,
            n_attention_heads=num_heads,
            dropout=dropout,
        )

        # --- Stage 4: Aitchison transformer ---
        # Input projection from pooled embedding to sequence for transformer
        self.pre_transformer_proj = nn.Linear(embed_dim, embed_dim)
        self.aitchison_layers = nn.ModuleList([
            AitchisonTransformerLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout,
            )
            for _ in range(num_aitchison_layers)
        ])

        # --- Stage 5: Simplex Neural ODE (for temporal data / health scoring) ---
        self.simplex_ode = SimplexNeuralODE(
            input_dim=input_dim,
            latent_dim=embed_dim,
            hidden_dim=ff_dim,
            dropout=dropout,
        )

        # --- Stage 6: Source attribution head ---
        self.source_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_sources),
        )

        # --- Fusion and projection ---
        # Fuse transformer output with ODE trajectory embedding
        self.fusion_layer = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )

        # Projection to shared embedding space (contract: 256 -> 256)
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, shared_embed_dim),
            nn.LayerNorm(shared_embed_dim),
        )

        # --- CLR input projection for direct CLR input path ---
        # When raw abundances are not available (input is already CLR-transformed)
        self.clr_input_proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )

        # Cache for precomputed sequence embeddings
        self._cached_seq_embeddings: torch.Tensor | None = None

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier initialization for projection and head layers."""
        for module in [
            self.fusion_layer,
            self.projection,
            self.source_head,
            self.clr_input_proj,
        ]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        # Init pre-transformer projection
        nn.init.xavier_uniform_(self.pre_transformer_proj.weight)
        nn.init.zeros_(self.pre_transformer_proj.bias)

    def cache_sequence_embeddings(
        self,
        sequences: list[str] | None = None,
        n_otus: int | None = None,
    ) -> None:
        """Precompute and cache ASV sequence embeddings.

        Call once when the OTU table is defined. Avoids redundant DNABERT-S
        forward passes during training.

        Args:
            sequences: DNA sequences per ASV (for DNABERT-S).
            n_otus: Number of OTUs (for fallback embeddings).
        """
        device = next(self.parameters()).device
        with torch.no_grad():
            self._cached_seq_embeddings = self.sequence_encoder(
                sequences=sequences, n_otus=n_otus, device=device
            ).detach()

    def _get_seq_embeddings(self, n_otus: int) -> torch.Tensor:
        """Get cached or compute sequence embeddings.

        Args:
            n_otus: Number of OTUs in the current batch.

        Returns:
            Sequence embeddings [n_otus, embed_dim].
        """
        if self._cached_seq_embeddings is not None:
            return self._cached_seq_embeddings[:n_otus]

        device = next(self.parameters()).device
        return self.sequence_encoder(n_otus=n_otus, device=device)

    def forward(
        self,
        x: torch.Tensor,
        raw_abundances: torch.Tensor | None = None,
        clr_sequence: torch.Tensor | None = None,
        timestamps: torch.Tensor | None = None,
        extract_indicators: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Full forward pass through MicroBiomeNet.

        Supports three input modes:
        1. CLR-only (x only): Uses CLR input directly, skips zero-inflation.
        2. Raw + CLR (x + raw_abundances): Full pipeline with zero-inflation.
        3. Temporal (x + clr_sequence + timestamps): Full pipeline with ODE.

        Args:
            x: CLR-transformed ASV abundances [B, input_dim].
            raw_abundances: Optional raw (non-CLR) abundances [B, input_dim]
                for zero-inflation gating. If None, skip zero-inflation.
            clr_sequence: Optional temporal CLR sequence [B, T, input_dim]
                for trajectory modeling. If None, use single-timepoint mode.
            timestamps: Optional timestamps [B, T] or [T] for temporal data.
            extract_indicators: Whether to compute indicator species weights.

        Returns:
            Dict with:
                'embedding': Projected embedding [B, 256] for fusion.
                'fusion_embedding': Same as embedding (interface contract).
                'source_logits': Contamination source logits [B, num_sources].
                'source_probs': Source probabilities [B, num_sources].
                'community_health_score': ODE-based health anomaly score [B].
                'indicator_species_weights': Per-ASV importance [B, n_otus].
        """
        B = x.shape[0]
        n_otus = x.shape[1]

        # --- Stage 1: Zero-inflation gating ---
        zero_type_mask = torch.zeros(B, n_otus, device=x.device)
        if raw_abundances is not None:
            imputed, zero_type_mask = self.zero_gate(raw_abundances)
            # Re-compute CLR from imputed abundances
            x_clr = clr_transform(imputed + 1e-10)  # add epsilon for log safety
        else:
            x_clr = x  # input is already CLR-transformed

        # --- Stage 2: Get phylogenetic sequence embeddings ---
        seq_embeddings = self._get_seq_embeddings(n_otus)  # [n_otus, embed_dim]

        # --- Stage 3: Abundance-weighted pooling ---
        pooled, indicator_weights = self.abundance_pooling(
            sequence_embeddings=seq_embeddings,
            clr_abundances=x_clr,
        )  # pooled: [B, embed_dim], indicator_weights: [B, n_otus]

        # --- Stage 4: Aitchison transformer ---
        # Create a sequence by combining: [CLS-like pooled token, per-OTU projections]
        # For efficiency, we process the pooled embedding through transformer layers
        # as a short sequence (using the CLR input projection for context)
        clr_projected = self.clr_input_proj(x_clr)  # [B, embed_dim]

        # Stack pooled and CLR projections as a 2-token sequence for self-attention
        transformer_input = torch.stack([
            self.pre_transformer_proj(pooled),
            clr_projected,
        ], dim=1)  # [B, 2, embed_dim]

        attn_weights_all = []
        h = transformer_input
        for layer in self.aitchison_layers:
            h, attn_w = layer(h, need_weights=extract_indicators)
            if attn_w is not None:
                attn_weights_all.append(attn_w)

        # Use first token (pooled representation) as the transformer output
        transformer_out = h[:, 0, :]  # [B, embed_dim]

        # --- Stage 5: Simplex Neural ODE ---
        if clr_sequence is not None and timestamps is not None:
            ode_output = self.simplex_ode(clr_sequence, timestamps)
            trajectory_embedding = ode_output["trajectory_embedding"]
            community_health_score = ode_output["anomaly_score"]
        else:
            # Single timepoint: use simplified health scoring
            trajectory_embedding, community_health_score = (
                self.simplex_ode.forward_single_timepoint(x_clr)
            )

        # --- Stage 6: Source attribution ---
        source_logits = self.source_head(transformer_out)
        source_probs = F.softmax(source_logits, dim=-1)

        # --- Fusion and projection ---
        fused = self.fusion_layer(
            torch.cat([transformer_out, trajectory_embedding], dim=-1)
        )
        embedding = self.projection(fused)

        # --- Build output dict ---
        result: dict[str, torch.Tensor] = {
            "embedding": embedding,
            "fusion_embedding": embedding,
            "source_logits": source_logits,
            "source_probs": source_probs,
            "community_health_score": community_health_score,
            "indicator_species_weights": indicator_weights,
        }

        return result

    def compute_loss(
        self,
        x: torch.Tensor,
        outputs: dict[str, torch.Tensor],
        source_targets: Optional[torch.Tensor] = None,
        raw_abundances: Optional[torch.Tensor] = None,
        clr_sequence: Optional[torch.Tensor] = None,
        timestamps: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Compute combined training loss.

        Args:
            x: Original CLR input [B, input_dim].
            outputs: Forward pass outputs.
            source_targets: Ground truth source labels [B] (long tensor).
            raw_abundances: Raw abundances for zero-inflation supervision.
            clr_sequence: Temporal CLR data for trajectory loss.
            timestamps: Timestamps for trajectory loss.

        Returns:
            Dict of loss components.
        """
        losses: dict[str, torch.Tensor] = {}
        device = x.device
        total = torch.tensor(0.0, device=device)

        # Source attribution loss
        if source_targets is not None:
            source_loss = F.cross_entropy(
                outputs["source_logits"], source_targets
            )
            losses["source_attribution"] = source_loss
            total = total + source_loss

        # Trajectory prediction loss (if temporal data available)
        if clr_sequence is not None and timestamps is not None:
            ode_output = self.simplex_ode(clr_sequence, timestamps)
            trajectory_loss = F.mse_loss(
                ode_output["predicted_trajectory"],
                ode_output["observed_states"],
            )
            losses["trajectory"] = trajectory_loss
            total = total + trajectory_loss

        # Health score regularization: encourage healthy samples near zero
        health_reg = outputs["community_health_score"].pow(2).mean()
        losses["health_regularization"] = health_reg * 0.01
        total = total + losses["health_regularization"]

        losses["total"] = total
        return losses
