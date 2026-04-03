"""Chemical identity encoder for the Digital Biosentinel.

Encodes chemical identity into a learned 128-dimensional embedding space
where structurally or functionally similar chemicals are mapped to nearby
vectors.  Unknown chemicals fall back to their chemical class centroid
with an inflated uncertainty flag.

Training data source: EPA ECOTOX chemical metadata (~12,000 chemicals).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Chemical class taxonomy used by ECOTOX / EPA classification
# ---------------------------------------------------------------------------
CHEMICAL_CLASSES: List[str] = [
    "organophosphate",
    "organochlorine",
    "carbamate",
    "pyrethroid",
    "neonicotinoid",
    "triazine",
    "phenol",
    "phthalate",
    "polycyclic_aromatic_hydrocarbon",
    "heavy_metal",
    "surfactant",
    "pharmaceutical",
    "per_polyfluoroalkyl",
    "nitroaromatic",
    "chlorinated_solvent",
    "petroleum_hydrocarbon",
    "inorganic_acid",
    "inorganic_base",
    "nutrient",
    "other",
]

CHEMICAL_CLASS_TO_IDX: Dict[str, int] = {
    cls: idx for idx, cls in enumerate(CHEMICAL_CLASSES)
}

# Number of coarse molecular-descriptor features that accompany each chemical
# (e.g. log_kow, molecular_weight, water_solubility, vapour_pressure …)
NUM_MOLECULAR_DESCRIPTORS = 8


class ChemicalEncoder(nn.Module):
    """Learned chemical embedding encoder.

    Each chemical in the vocabulary receives a trainable embedding vector.
    At training time these embeddings are regularised so that chemicals
    sharing the same class are pulled together (contrastive auxiliary loss).

    For chemicals *not* in the vocabulary (zero-shot), the encoder falls
    back to a class-level embedding plus a small MLP over molecular
    descriptors, and sets an ``is_unknown`` flag that downstream modules
    can use to inflate uncertainty.

    Parameters
    ----------
    num_chemicals : int
        Size of the known-chemical vocabulary (ECOTOX catalogue).
    embedding_dim : int
        Dimensionality of the output chemical embedding (default 128).
    num_classes : int
        Number of chemical classes (default ``len(CHEMICAL_CLASSES)``).
    num_descriptors : int
        Number of continuous molecular-descriptor features.
    dropout : float
        Dropout probability inside the descriptor MLP.
    """

    def __init__(
        self,
        num_chemicals: int,
        embedding_dim: int = 128,
        num_classes: int = len(CHEMICAL_CLASSES),
        num_descriptors: int = NUM_MOLECULAR_DESCRIPTORS,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_chemicals = num_chemicals
        self.num_classes = num_classes

        # --- Learned chemical embedding table ----------------------------------
        # Index 0 is reserved for "unknown / OOV" chemicals.
        self.chemical_embedding = nn.Embedding(
            num_embeddings=num_chemicals + 1,  # +1 for OOV at index 0
            embedding_dim=embedding_dim,
            padding_idx=0,
        )

        # --- Chemical-class embedding ------------------------------------------
        self.class_embedding = nn.Embedding(
            num_embeddings=num_classes,
            embedding_dim=embedding_dim,
        )

        # --- Descriptor MLP (fallback pathway for unknown chemicals) -----------
        self.descriptor_mlp = nn.Sequential(
            nn.Linear(num_descriptors, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, embedding_dim),
        )

        # --- Fusion gate: blend chemical-specific vs class+descriptor ----------
        self.gate = nn.Sequential(
            nn.Linear(embedding_dim * 2, 1),
            nn.Sigmoid(),
        )

        # Layer norm on the output embedding
        self.layer_norm = nn.LayerNorm(embedding_dim)

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        """Xavier-uniform for linear layers; normal for embeddings."""
        nn.init.normal_(self.chemical_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.class_embedding.weight, mean=0.0, std=0.02)
        # Keep padding vector at zero
        with torch.no_grad():
            self.chemical_embedding.weight[0].zero_()
        for module in self.descriptor_mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    def forward(
        self,
        chemical_idx: torch.Tensor,
        class_idx: torch.Tensor,
        descriptors: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of chemicals.

        Parameters
        ----------
        chemical_idx : Tensor[B]  (long)
            Index into the chemical vocabulary.  Use ``0`` for unknown.
        class_idx : Tensor[B]  (long)
            Chemical-class index (see ``CHEMICAL_CLASS_TO_IDX``).
        descriptors : Tensor[B, num_descriptors]  (float)
            Continuous molecular descriptors (log-Kow, MW, …).  Should
            already be standardised (zero-mean, unit-variance).

        Returns
        -------
        embedding : Tensor[B, embedding_dim]
            Chemical embedding vector.
        is_unknown : Tensor[B]  (bool)
            ``True`` where the chemical was OOV (index 0).
        """
        is_unknown = chemical_idx == 0  # [B]

        # Known-chemical pathway
        chem_emb = self.chemical_embedding(chemical_idx)  # [B, D]

        # Class + descriptor pathway (always computed; used as fallback)
        cls_emb = self.class_embedding(class_idx)  # [B, D]
        desc_emb = self.descriptor_mlp(descriptors)  # [B, D]
        fallback_emb = cls_emb + desc_emb  # [B, D]

        # Gate: for known chemicals, prefer the learned embedding;
        # for unknown, rely entirely on the fallback.
        gate_input = torch.cat([chem_emb, fallback_emb], dim=-1)  # [B, 2D]
        alpha = self.gate(gate_input)  # [B, 1]

        # Force alpha → 0 for unknowns so fallback dominates
        alpha = alpha * (~is_unknown).unsqueeze(-1).float()

        embedding = alpha * chem_emb + (1.0 - alpha) * fallback_emb
        embedding = self.layer_norm(embedding)

        return embedding, is_unknown

    # ------------------------------------------------------------------
    def class_contrastive_loss(
        self,
        embeddings: torch.Tensor,
        class_idx: torch.Tensor,
        temperature: float = 0.1,
    ) -> torch.Tensor:
        """Supervised contrastive loss that pulls same-class chemicals together.

        Parameters
        ----------
        embeddings : Tensor[B, D]
            Chemical embeddings from ``forward()``.
        class_idx : Tensor[B]  (long)
            Chemical-class labels.
        temperature : float
            Contrastive temperature (lower = sharper).

        Returns
        -------
        loss : scalar Tensor
        """
        B = embeddings.size(0)
        if B < 2:
            return embeddings.new_tensor(0.0)

        # L2-normalise
        normed = F.normalize(embeddings, dim=-1)
        sim = torch.mm(normed, normed.t()) / temperature  # [B, B]

        # Mask: same class = positive, different class = negative
        labels = class_idx.unsqueeze(0) == class_idx.unsqueeze(1)  # [B, B]
        # Exclude self-pairs
        self_mask = torch.eye(B, dtype=torch.bool, device=embeddings.device)
        labels = labels & ~self_mask

        # If no positive pairs exist, return 0
        if labels.sum() == 0:
            return embeddings.new_tensor(0.0)

        # Log-sum-exp over all non-self entries (denominator)
        logits_mask = ~self_mask
        exp_sim = torch.exp(sim) * logits_mask.float()
        log_denom = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # Mean of log-prob over positive pairs
        log_prob = sim - log_denom
        mean_log_prob = (log_prob * labels.float()).sum(dim=1) / labels.float().sum(
            dim=1
        ).clamp(min=1.0)

        loss = -mean_log_prob.mean()
        return loss
