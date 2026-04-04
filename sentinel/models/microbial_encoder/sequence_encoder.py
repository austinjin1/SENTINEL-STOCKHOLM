"""DNABERT-S integration for phylogenetic-aware OTU/ASV sequence encoding.

Uses the pretrained DNABERT-S model (zhihan1996/DNABERT-S) to encode 16S rRNA
representative sequences per ASV into dense embeddings that capture phylogenetic
relationships. Phylogenetically similar taxa produce similar embeddings, enabling
the downstream model to generalize across related organisms.

References:
    Zhou et al. (2024). DNABERT-S: Learning Species-Aware DNA Embedding with
        Genome Foundation Models. arXiv:2402.08777.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Default DNABERT-S model identifier on HuggingFace
DNABERT_S_MODEL_ID = "zhihan1996/DNABERT-S"

# DNABERT-S output dimension (BERT-base hidden size)
DNABERT_S_DIM = 768

# Maximum sequence length for 16S rRNA fragments (in tokens)
MAX_SEQ_LENGTH = 512


class DNABERTSequenceEncoder(nn.Module):
    """Wraps DNABERT-S for encoding 16S rRNA representative sequences per ASV.

    Loads the pretrained DNABERT-S model from HuggingFace, tokenizes input
    DNA sequences, and produces per-ASV embeddings that capture phylogenetic
    relationships. Falls back to learned random embeddings if DNABERT-S is
    unavailable (e.g., no internet, missing dependencies).

    Args:
        output_dim: Output embedding dimension per ASV. Default 256.
        max_otus: Maximum number of OTUs/ASVs to encode. Default 5000.
        freeze_backbone: If True, freeze DNABERT-S weights. Default True.
        model_id: HuggingFace model identifier. Default "zhihan1996/DNABERT-S".
        max_seq_length: Maximum tokenized sequence length. Default 512.
    """

    def __init__(
        self,
        output_dim: int = 256,
        max_otus: int = 5000,
        freeze_backbone: bool = True,
        model_id: str = DNABERT_S_MODEL_ID,
        max_seq_length: int = MAX_SEQ_LENGTH,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.max_otus = max_otus
        self.freeze_backbone = freeze_backbone
        self.max_seq_length = max_seq_length
        self.using_dnabert = False

        # Try to load DNABERT-S; fall back to learned embeddings
        self._init_backbone(model_id)

        # Projection from backbone dim to output_dim
        backbone_dim = DNABERT_S_DIM if self.using_dnabert else output_dim
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
        )

        self._init_weights()

    def _init_backbone(self, model_id: str) -> None:
        """Attempt to load DNABERT-S; fall back to learned embeddings on failure."""
        try:
            from transformers import AutoTokenizer, AutoModel

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id, trust_remote_code=True
            )
            self.backbone = AutoModel.from_pretrained(
                model_id, trust_remote_code=True
            )

            if self.freeze_backbone:
                for param in self.backbone.parameters():
                    param.requires_grad = False

            self.using_dnabert = True
            logger.info("DNABERT-S loaded successfully from %s", model_id)

        except Exception as e:
            logger.warning(
                "DNABERT-S unavailable (%s). Using learned fallback embeddings.",
                str(e),
            )
            self.using_dnabert = False
            self.tokenizer = None
            self.backbone = None

            # Fallback: learned embeddings per OTU position
            self.fallback_embeddings = nn.Embedding(
                self.max_otus, self.output_dim
            )
            nn.init.xavier_uniform_(self.fallback_embeddings.weight)

    def _init_weights(self) -> None:
        """Xavier init for projection layers."""
        for m in self.projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _encode_with_dnabert(
        self,
        sequences: list[str],
        device: torch.device,
    ) -> torch.Tensor:
        """Encode DNA sequences using DNABERT-S.

        Args:
            sequences: List of DNA sequence strings (e.g., 16S rRNA fragments).
            device: Target device for output tensors.

        Returns:
            Sequence embeddings [n_seqs, DNABERT_S_DIM].
        """
        assert self.tokenizer is not None and self.backbone is not None

        # Tokenize all sequences
        encoded = self.tokenizer(
            sequences,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        # Forward through DNABERT-S
        if self.freeze_backbone:
            with torch.no_grad():
                outputs = self.backbone(**encoded)
        else:
            outputs = self.backbone(**encoded)

        # Use [CLS] token embedding as sequence representation
        # DNABERT-S outputs last_hidden_state [B, seq_len, hidden_dim]
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [n_seqs, 768]

        return cls_embeddings

    def forward(
        self,
        sequences: list[str] | None = None,
        n_otus: int | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Encode ASV representative sequences into dense embeddings.

        Either provide DNA sequences (for DNABERT-S encoding) or n_otus
        (for fallback/positional embeddings).

        Args:
            sequences: List of DNA sequence strings, one per ASV. If None
                and DNABERT-S is available, raises ValueError.
            n_otus: Number of OTUs. Required for fallback mode. If sequences
                is provided, this is inferred from len(sequences).
            device: Target device. Inferred from model parameters if None.

        Returns:
            Per-ASV sequence embeddings [n_otus, output_dim].
        """
        if device is None:
            device = next(self.parameters()).device

        if self.using_dnabert and sequences is not None:
            # Encode with DNABERT-S
            raw_embeddings = self._encode_with_dnabert(sequences, device)
            return self.projection(raw_embeddings)

        # Fallback: learned positional embeddings
        if n_otus is None:
            if sequences is not None:
                n_otus = len(sequences)
            else:
                raise ValueError(
                    "Must provide either sequences or n_otus."
                )

        n_otus = min(n_otus, self.max_otus)
        indices = torch.arange(n_otus, device=device)
        raw_embeddings = self.fallback_embeddings(indices)  # [n_otus, output_dim]

        # Fallback embeddings are already output_dim, but project for consistency
        return self.projection(
            # Pad to backbone dim if needed, or bypass projection
            raw_embeddings
        )

    def get_cached_embeddings(
        self,
        sequences: list[str],
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Precompute and cache sequence embeddings for a fixed OTU table.

        Call this once when the OTU table is defined, then reuse the cached
        embeddings during training/inference to avoid redundant DNABERT-S
        forward passes.

        Args:
            sequences: List of DNA sequence strings, one per ASV.
            device: Target device.

        Returns:
            Cached per-ASV embeddings [n_otus, output_dim].
        """
        with torch.no_grad():
            embeddings = self.forward(sequences=sequences, device=device)
        return embeddings.detach()
