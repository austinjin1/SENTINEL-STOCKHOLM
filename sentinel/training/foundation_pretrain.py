"""Joint multimodal self-supervised pretraining for the Water Foundation Model.

Implements cross-modal masked reconstruction pretraining across all five
SENTINEL modalities (sensor, satellite, microbial, molecular, behavioral).
All modality streams are randomly masked at the whole-modality level and the
model learns to reconstruct the masked modalities from the fused representation
of the unmasked ones, with an auxiliary cross-modal consistency loss.

Architecture overview:
    1. Freeze all five per-modality encoders (AquaSSM, HydroViT, MicroBiomeNet,
       ToxiGene, BioMotion) -- they produce fixed embeddings.
    2. Pass unmasked modalities through the Perceiver IO fusion layer normally.
    3. Replace masked modalities with learned [MASK] tokens in the fusion input.
    4. From the fused representation, per-modality decoder heads reconstruct
       the original encoder embeddings for the masked modalities.
    5. Losses: per-modality reconstruction MSE, cross-modal InfoNCE consistency,
       and modality presence prediction.

Training interface::

    objective = FoundationPretrainObjective(encoders, fusion)
    losses = objective.compute_foundation_loss(batch)
    losses["total_loss"].backward()

Usage::

    python -m sentinel.training.foundation_pretrain \\
        --data-dir data/aligned \\
        --encoder-dir outputs/pretrained_encoders \\
        --output-dir outputs/foundation
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from sentinel.models.fusion.embedding_registry import (
    MODALITY_IDS,
    NUM_MODALITIES,
    SHARED_EMBEDDING_DIM,
)
from sentinel.models.fusion.model import PerceiverIOFusion
from sentinel.models.fusion.perceiver_attention import PerceiverCrossAttention
from sentinel.models.fusion.latent_array import LatentArray

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SHARED_EMBED_DIM: int = 256

# Default reconstruction loss weights per modality.  Modalities with
# noisier or higher-dimensional native spaces receive lower weight so
# that no single modality dominates the total loss.
DEFAULT_MODALITY_WEIGHTS: Dict[str, float] = {
    "sensor": 1.0,
    "satellite": 1.0,
    "microbial": 0.8,
    "molecular": 0.8,
    "behavioral": 0.6,
}

# Cross-modal masking groups: when structured masking is enabled,
# correlated modalities are masked together.
CORRELATED_GROUPS: List[Tuple[str, ...]] = [
    ("sensor", "satellite"),
    ("microbial", "molecular"),
]


# ---------------------------------------------------------------------------
# JointMaskingStrategy
# ---------------------------------------------------------------------------


class JointMaskingStrategy(nn.Module):
    """Whole-modality masking strategy for foundation pretraining.

    Masks entire modality streams (not individual tokens within a modality)
    so the model must reconstruct a full modality from the remaining ones.
    Supports three masking modes:

    * **random** (default): each modality is independently masked with
      probability ``mask_prob``, subject to the constraint that at least
      one modality remains unmasked per sample.
    * **structured**: correlated modality groups (sensor+satellite,
      microbial+molecular) are masked together.
    * **cross_modal**: same as structured, used as an alias.

    Args:
        mask_prob: Per-modality masking probability. Default 0.3.
        mode: One of ``"random"``, ``"structured"``, ``"cross_modal"``.
        min_unmasked: Minimum number of modalities that must stay
            unmasked per sample. Default 1.
    """

    VALID_MODES = ("random", "structured", "cross_modal")

    def __init__(
        self,
        mask_prob: float = 0.3,
        mode: str = "random",
        min_unmasked: int = 1,
    ) -> None:
        super().__init__()
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Invalid masking mode '{mode}'. "
                f"Expected one of {self.VALID_MODES}"
            )
        self.mask_prob = mask_prob
        self.mode = mode
        self.min_unmasked = min_unmasked

    def forward(
        self,
        batch_size: int,
        available_modalities: List[str],
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Generate per-sample, per-modality mask decisions.

        Args:
            batch_size: Number of samples in the batch.
            available_modalities: Modality ids present in the current
                batch (only these can be masked).
            device: Target device for the mask tensors.

        Returns:
            Dictionary mapping modality id to a boolean mask tensor of
            shape ``[B]`` where ``True`` means the modality is **masked**
            (i.e. should be reconstructed).
        """
        masks: Dict[str, torch.Tensor] = {}
        n_available = len(available_modalities)

        if n_available <= self.min_unmasked:
            # Cannot mask anything -- all must remain visible.
            for mid in available_modalities:
                masks[mid] = torch.zeros(batch_size, dtype=torch.bool, device=device)
            return masks

        if self.mode == "random":
            masks = self._random_mask(
                batch_size, available_modalities, device
            )
        else:
            # structured / cross_modal
            masks = self._structured_mask(
                batch_size, available_modalities, device
            )

        return masks

    def _random_mask(
        self,
        batch_size: int,
        available: List[str],
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Independent Bernoulli masking with min-unmasked guarantee."""
        masks: Dict[str, torch.Tensor] = {}
        n = len(available)

        for _ in range(batch_size):
            sample_masks: Dict[str, bool] = {}
            for mid in available:
                sample_masks[mid] = random.random() < self.mask_prob

            # Ensure at least min_unmasked modalities are visible.
            unmasked = [m for m, v in sample_masks.items() if not v]
            if len(unmasked) < self.min_unmasked:
                # Randomly unmask some modalities.
                masked_list = [m for m, v in sample_masks.items() if v]
                random.shuffle(masked_list)
                n_to_unmask = self.min_unmasked - len(unmasked)
                for m in masked_list[:n_to_unmask]:
                    sample_masks[m] = False

            for mid in available:
                masks.setdefault(mid, []).append(sample_masks[mid])  # type: ignore[arg-type]

        return {
            mid: torch.tensor(vals, dtype=torch.bool, device=device)
            for mid, vals in masks.items()
        }

    def _structured_mask(
        self,
        batch_size: int,
        available: List[str],
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Group-correlated masking: correlated modalities masked together."""
        masks: Dict[str, list] = {mid: [] for mid in available}
        available_set = set(available)

        for _ in range(batch_size):
            sample_masks: Dict[str, bool] = {mid: False for mid in available}

            # First mask correlated groups together.
            for group in CORRELATED_GROUPS:
                group_available = [m for m in group if m in available_set]
                if len(group_available) >= 2 and random.random() < self.mask_prob:
                    for m in group_available:
                        sample_masks[m] = True

            # Then independently mask remaining modalities.
            for mid in available:
                if not sample_masks[mid] and random.random() < self.mask_prob:
                    sample_masks[mid] = True

            # Enforce min_unmasked constraint.
            unmasked = [m for m, v in sample_masks.items() if not v]
            if len(unmasked) < self.min_unmasked:
                masked_list = [m for m, v in sample_masks.items() if v]
                random.shuffle(masked_list)
                for m in masked_list[: self.min_unmasked - len(unmasked)]:
                    sample_masks[m] = False

            for mid in available:
                masks[mid].append(sample_masks[mid])

        return {
            mid: torch.tensor(vals, dtype=torch.bool, device=device)
            for mid, vals in masks.items()
        }


# ---------------------------------------------------------------------------
# Per-modality Reconstructor heads
# ---------------------------------------------------------------------------


class ModalityReconstructor(nn.Module):
    """Base per-modality decoder head: fused latent -> modality embedding.

    A lightweight MLP that maps from the shared fused representation back
    to the encoder's output embedding space so the model can reconstruct
    masked modality embeddings.

    Args:
        input_dim: Dimension of the fused representation.
        output_dim: Dimension of the target encoder embedding.
        hidden_dim: Hidden layer dimension. Defaults to 2x output_dim.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        input_dim: int = SHARED_EMBED_DIM,
        output_dim: int = SHARED_EMBED_DIM,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or output_dim * 2
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        """Reconstruct a modality embedding from the fused state.

        Args:
            fused: Fused representation ``[B, input_dim]``.

        Returns:
            Predicted modality embedding ``[B, output_dim]``.
        """
        return self.net(fused)


class SensorReconstructor(ModalityReconstructor):
    """Decoder head for the AquaSSM sensor encoder embedding space.

    Reconstructs the 256-dim projected sensor embedding from the fused
    latent.
    """

    def __init__(
        self,
        input_dim: int = SHARED_EMBED_DIM,
        output_dim: int = SHARED_EMBED_DIM,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            dropout=dropout,
        )


class SatelliteReconstructor(ModalityReconstructor):
    """Decoder head for the HydroViT satellite encoder embedding space.

    Reconstructs the 256-dim projected satellite embedding (spectral
    band features) from the fused latent.
    """

    def __init__(
        self,
        input_dim: int = SHARED_EMBED_DIM,
        output_dim: int = SHARED_EMBED_DIM,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            dropout=dropout,
        )


class MicrobialReconstructor(ModalityReconstructor):
    """Decoder head for the MicroBiomeNet microbial encoder embedding space.

    Reconstructs the 256-dim projected microbial abundance embedding
    from the fused latent.
    """

    def __init__(
        self,
        input_dim: int = SHARED_EMBED_DIM,
        output_dim: int = SHARED_EMBED_DIM,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            dropout=dropout,
        )


class MolecularReconstructor(ModalityReconstructor):
    """Decoder head for the ToxiGene molecular encoder embedding space.

    Reconstructs the 256-dim projected molecular expression signature
    from the fused latent.
    """

    def __init__(
        self,
        input_dim: int = SHARED_EMBED_DIM,
        output_dim: int = SHARED_EMBED_DIM,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            dropout=dropout,
        )


class BehavioralReconstructor(ModalityReconstructor):
    """Decoder head for the BioMotion behavioral encoder embedding space.

    Reconstructs the 256-dim projected behavioral trajectory features
    from the fused latent.
    """

    def __init__(
        self,
        input_dim: int = SHARED_EMBED_DIM,
        output_dim: int = SHARED_EMBED_DIM,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            dropout=dropout,
        )


# Registry mapping modality ids to their reconstructor class.
RECONSTRUCTOR_CLASSES: Dict[str, type] = {
    "sensor": SensorReconstructor,
    "satellite": SatelliteReconstructor,
    "microbial": MicrobialReconstructor,
    "molecular": MolecularReconstructor,
    "behavioral": BehavioralReconstructor,
}


# ---------------------------------------------------------------------------
# Modality presence prediction head
# ---------------------------------------------------------------------------


class ModalityPresenceHead(nn.Module):
    """Binary classification head predicting which modalities were present.

    From the fused representation, predicts a NUM_MODALITIES-dim logit
    vector indicating whether each modality was observed (unmasked) vs.
    masked in the input.

    Args:
        input_dim: Dimension of the fused representation.
        num_modalities: Number of modalities to predict.
        hidden_dim: Hidden layer dimension.
    """

    def __init__(
        self,
        input_dim: int = SHARED_EMBED_DIM,
        num_modalities: int = NUM_MODALITIES,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_modalities),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        """Predict modality presence logits.

        Args:
            fused: Fused representation ``[B, D]``.

        Returns:
            Logits ``[B, num_modalities]`` where positive = modality
            was present (unmasked).
        """
        return self.head(fused)


# ---------------------------------------------------------------------------
# FoundationPretrainObjective
# ---------------------------------------------------------------------------


class FoundationPretrainObjective(nn.Module):
    """Joint multimodal self-supervised pretraining objective.

    Takes all five frozen encoders and the Perceiver IO fusion model.
    At each step:

    1. Encode all available modalities through their frozen encoders.
    2. Apply the masking strategy to select which modalities to hide.
    3. Feed unmasked modality embeddings (and learned [MASK] tokens for
       masked ones) into the fusion layer.
    4. Decode the fused representation through per-modality reconstructor
       heads to predict the original masked embeddings.
    5. Compute reconstruction MSE, cross-modal consistency (InfoNCE),
       and modality presence prediction losses.

    Only the fusion layer, reconstructor heads, [MASK] tokens, and the
    modality presence head are trained; all encoders remain frozen.

    Args:
        fusion: The Perceiver IO fusion model.
        shared_embed_dim: Shared embedding dimensionality. Default 256.
        mask_prob: Masking probability. Default 0.3.
        mask_mode: Masking mode (``"random"``, ``"structured"``,
            ``"cross_modal"``). Default ``"random"``.
        modality_weights: Per-modality reconstruction loss weights.
        consistency_temperature: InfoNCE temperature. Default 0.07.
        consistency_weight: Weight for the cross-modal consistency loss.
            Default 0.1.
        presence_weight: Weight for the modality presence prediction loss.
            Default 0.05.
        dropout: Dropout for reconstructor heads. Default 0.1.
    """

    def __init__(
        self,
        fusion: PerceiverIOFusion,
        shared_embed_dim: int = SHARED_EMBED_DIM,
        mask_prob: float = 0.3,
        mask_mode: str = "random",
        modality_weights: Optional[Dict[str, float]] = None,
        consistency_temperature: float = 0.07,
        consistency_weight: float = 0.1,
        presence_weight: float = 0.05,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.shared_embed_dim = shared_embed_dim
        self.consistency_weight = consistency_weight
        self.presence_weight = presence_weight
        self.modality_weights = modality_weights or dict(DEFAULT_MODALITY_WEIGHTS)

        # Fusion model (trainable).
        self.fusion = fusion

        # Masking strategy.
        self.masking = JointMaskingStrategy(
            mask_prob=mask_prob,
            mode=mask_mode,
        )

        # Learned [MASK] tokens -- one per modality, in shared space.
        self.mask_tokens = nn.ParameterDict({
            mid: nn.Parameter(torch.randn(shared_embed_dim) * 0.02)
            for mid in MODALITY_IDS
        })

        # Per-modality reconstructor heads.
        self.reconstructors = nn.ModuleDict({
            mid: RECONSTRUCTOR_CLASSES[mid](
                input_dim=shared_embed_dim,
                output_dim=shared_embed_dim,
                dropout=dropout,
            )
            for mid in MODALITY_IDS
        })

        # Modality presence prediction head.
        self.presence_head = ModalityPresenceHead(
            input_dim=shared_embed_dim,
            num_modalities=NUM_MODALITIES,
        )

        # Learnable InfoNCE temperature (log-space).
        self.log_temperature = nn.Parameter(
            torch.tensor(consistency_temperature).log()
        )

    @property
    def temperature(self) -> torch.Tensor:
        """Current InfoNCE temperature."""
        return self.log_temperature.exp()

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def _fuse_with_masks(
        self,
        embeddings: Dict[str, torch.Tensor],
        masks: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Run the Perceiver IO fusion with masked modality tokens.

        For each modality:
        - If unmasked for a sample: use the actual (frozen) encoder embedding.
        - If masked for a sample: substitute the learned [MASK] token.

        We bypass the PerceiverIOFusion's sequential interface and directly
        invoke the Perceiver cross-attention on assembled token matrices.

        Args:
            embeddings: Per-modality projected embeddings ``{mid: [B, D]}``.
            masks: Per-modality boolean mask ``{mid: [B]}``, True = masked.

        Returns:
            Fused state ``[B, D]``.
        """
        B = next(iter(embeddings.values())).shape[0]
        device = next(iter(embeddings.values())).device

        # Assemble the input token matrix [B, K, D] with mask substitution.
        token_list: list[torch.Tensor] = []
        for mid in MODALITY_IDS:
            if mid in embeddings:
                emb = embeddings[mid]  # [B, D]
                mask = masks.get(mid)
                if mask is not None:
                    # mask: [B], True = masked.  Replace masked samples with
                    # the learned [MASK] token.
                    mask_token = self.mask_tokens[mid].unsqueeze(0).expand(B, -1)
                    mask_expanded = mask.unsqueeze(-1).float()  # [B, 1]
                    token = emb * (1.0 - mask_expanded) + mask_token * mask_expanded
                else:
                    token = emb
            else:
                # Modality not present at all -- use the Perceiver's
                # no-data token (handled by the perceiver itself).
                token = None

            if token is not None:
                token_list.append(token)
            else:
                # Use a zero vector; the perceiver's no-data token will
                # handle truly absent modalities in the attention.
                token_list.append(torch.zeros(B, self.shared_embed_dim, device=device))

        # Construct input tokens [B, K, D].
        input_tokens = torch.stack(token_list, dim=1)  # [B, NUM_MODALITIES, D]

        # Get initial latent state from the fusion model.
        latents = self.fusion.latent_array.get_latents(B).to(device)

        # Temporal bias: all modalities are simultaneous in foundation
        # pretraining, so zero staleness.  Build a zero log-bias.
        temporal_bias = torch.zeros(B, NUM_MODALITIES, device=device)

        # Normalise input and latents through the perceiver's encode step.
        perceiver = self.fusion.perceiver

        normed_latents = perceiver.encode_norm_latent(latents)
        normed_input = perceiver.encode_norm_input(input_tokens)

        cross_out, encode_attn = perceiver.encode_cross_attn(
            query=normed_latents,
            key=normed_input,
            value=normed_input,
            attn_bias=temporal_bias.unsqueeze(1).unsqueeze(2),
        )
        latents = latents + cross_out
        latents = latents + perceiver.encode_ffn(latents)

        # Process: self-attention within latent array.
        for layer in perceiver.process_layers:
            latents = layer(latents)

        # Decode: output query cross-attends to latents.
        decode_q = perceiver.decode_query.expand(B, -1, -1)
        normed_q = perceiver.decode_norm_query(decode_q)
        normed_lat = perceiver.decode_norm_latent(latents)

        decoded, _ = perceiver.decode_cross_attn(
            query=normed_q,
            key=normed_lat,
            value=normed_lat,
        )
        fused_state = decoded.squeeze(1)  # [B, D]

        return fused_state

    def _compute_reconstruction_losses(
        self,
        fused_state: torch.Tensor,
        target_embeddings: Dict[str, torch.Tensor],
        masks: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute per-modality MSE reconstruction loss on masked modalities.

        Args:
            fused_state: Fused representation ``[B, D]``.
            target_embeddings: Original (unmasked) encoder embeddings.
            masks: Boolean mask per modality ``[B]``, True = masked.

        Returns:
            Dict with ``"recon/<modality>"`` keys and scalar loss values,
            plus ``"reconstruction_total"`` as the weighted sum.
        """
        losses: Dict[str, torch.Tensor] = {}
        device = fused_state.device
        total = torch.tensor(0.0, device=device)

        for mid in MODALITY_IDS:
            if mid not in target_embeddings or mid not in masks:
                continue

            mask = masks[mid]  # [B]
            if not mask.any():
                # No samples were masked for this modality.
                continue

            target = target_embeddings[mid]  # [B, D]
            predicted = self.reconstructors[mid](fused_state)  # [B, D]

            # Compute MSE only on masked samples.
            masked_pred = predicted[mask]  # [M, D]
            masked_target = target[mask]  # [M, D]

            recon_loss = F.mse_loss(masked_pred, masked_target)
            weight = self.modality_weights.get(mid, 1.0)
            losses[f"recon/{mid}"] = recon_loss
            total = total + weight * recon_loss

        losses["reconstruction_total"] = total
        return losses

    def _compute_consistency_loss(
        self,
        embeddings: Dict[str, torch.Tensor],
        masks: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """InfoNCE cross-modal consistency loss on unmasked modalities.

        Encourages co-located unmasked modality embeddings to be similar
        across modalities (positive pairs) while being dissimilar to
        embeddings from different batch samples (negative pairs).

        Args:
            embeddings: Per-modality projected embeddings ``{mid: [B, D]}``.
            masks: Boolean mask per modality ``[B]``, True = masked.

        Returns:
            Scalar InfoNCE consistency loss.
        """
        device = next(iter(embeddings.values())).device

        # Collect unmasked embeddings.  For each sample, gather all
        # modality embeddings that were *not* masked.
        unmasked_embs: List[torch.Tensor] = []
        unmasked_sample_ids: List[int] = []
        unmasked_mod_ids: List[int] = []

        for mod_idx, mid in enumerate(MODALITY_IDS):
            if mid not in embeddings:
                continue
            emb = embeddings[mid]  # [B, D]
            mask = masks.get(mid)
            B = emb.shape[0]
            for b in range(B):
                if mask is None or not mask[b]:
                    unmasked_embs.append(emb[b])
                    unmasked_sample_ids.append(b)
                    unmasked_mod_ids.append(mod_idx)

        if len(unmasked_embs) < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Stack into [N, D].
        all_emb = torch.stack(unmasked_embs, dim=0)  # [N, D]
        sample_ids = torch.tensor(unmasked_sample_ids, device=device)
        mod_ids = torch.tensor(unmasked_mod_ids, device=device)
        N = all_emb.shape[0]

        # L2 normalise.
        all_emb_norm = F.normalize(all_emb, dim=-1)

        # Cosine similarity matrix.
        sim = torch.matmul(all_emb_norm, all_emb_norm.t()) / self.temperature

        # Positive mask: same sample, different modality.
        same_sample = sample_ids.unsqueeze(0) == sample_ids.unsqueeze(1)
        diff_mod = mod_ids.unsqueeze(0) != mod_ids.unsqueeze(1)
        positive_mask = same_sample & diff_mod  # [N, N]

        if not positive_mask.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Self-mask.
        self_mask = ~torch.eye(N, dtype=torch.bool, device=device)

        # InfoNCE (multi-positive formulation).
        exp_sim = torch.exp(sim) * self_mask.float()
        denominator = exp_sim.sum(dim=1, keepdim=True).clamp(min=1e-8)
        log_prob = sim - torch.log(denominator)

        has_positive = positive_mask.any(dim=1)
        if not has_positive.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        pos_log_prob = (log_prob * positive_mask.float()).sum(dim=1)
        num_positives = positive_mask.float().sum(dim=1).clamp(min=1.0)
        per_anchor_loss = -pos_log_prob / num_positives

        return per_anchor_loss[has_positive].mean()

    def _compute_presence_loss(
        self,
        fused_state: torch.Tensor,
        masks: Dict[str, torch.Tensor],
        available_modalities: List[str],
    ) -> torch.Tensor:
        """Binary cross-entropy loss for modality presence prediction.

        From the fused representation, predict which modalities were
        observed (unmasked) vs. masked.

        Args:
            fused_state: Fused representation ``[B, D]``.
            masks: Boolean mask per modality ``[B]``, True = masked.
            available_modalities: Modalities present in this batch.

        Returns:
            Scalar BCE loss.
        """
        B = fused_state.shape[0]
        device = fused_state.device

        # Build target: 1 = present (unmasked), 0 = masked.
        target = torch.ones(B, NUM_MODALITIES, device=device)
        for i, mid in enumerate(MODALITY_IDS):
            if mid not in available_modalities:
                # Not available at all -- mark as absent.
                target[:, i] = 0.0
            elif mid in masks:
                target[:, i] = (~masks[mid]).float()

        logits = self.presence_head(fused_state)  # [B, NUM_MODALITIES]
        return F.binary_cross_entropy_with_logits(logits, target)

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def compute_foundation_loss(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute the full foundation pretraining loss.

        Args:
            batch: Dictionary with per-modality data, keyed by modality
                name.  Each value is a tensor of shape ``[B, D]``
                representing the **projected** encoder embedding in the
                shared 256-dim space (computed with frozen encoders
                outside this call).  Optionally includes
                ``"location_ids": [B]`` for consistency loss.

        Returns:
            Dict with:
                ``"total_loss"``: Scalar total loss for backprop.
                ``"reconstruction_total"``: Weighted sum of per-modality
                    reconstruction MSE losses.
                ``"recon/<modality>"``: Per-modality reconstruction loss
                    (only for modalities that were masked in >= 1 sample).
                ``"consistency_loss"``: Cross-modal InfoNCE loss.
                ``"presence_loss"``: Modality presence prediction loss.
        """
        device = None
        available_modalities: List[str] = []
        embeddings: Dict[str, torch.Tensor] = {}

        for mid in MODALITY_IDS:
            if mid in batch:
                emb = batch[mid]
                if device is None:
                    device = emb.device
                embeddings[mid] = emb
                available_modalities.append(mid)

        if not available_modalities:
            raise ValueError(
                "Batch contains no modality embeddings.  "
                f"Expected keys from {MODALITY_IDS}"
            )

        B = next(iter(embeddings.values())).shape[0]
        assert device is not None

        # 1. Generate masks.
        masks = self.masking(B, available_modalities, device)

        # 2. Run fusion with masked inputs.
        fused_state = self._fuse_with_masks(embeddings, masks)

        # 3. Reconstruction losses (on masked modalities).
        recon_losses = self._compute_reconstruction_losses(
            fused_state, embeddings, masks
        )

        # 4. Cross-modal consistency loss (on unmasked modalities).
        consistency_loss = self._compute_consistency_loss(embeddings, masks)

        # 5. Modality presence prediction loss.
        presence_loss = self._compute_presence_loss(
            fused_state, masks, available_modalities
        )

        # 6. Total loss.
        total = (
            recon_losses["reconstruction_total"]
            + self.consistency_weight * consistency_loss
            + self.presence_weight * presence_loss
        )

        # Build output dict.
        losses: Dict[str, torch.Tensor] = {
            "total_loss": total,
            "reconstruction_total": recon_losses["reconstruction_total"],
            "consistency_loss": consistency_loss,
            "presence_loss": presence_loss,
        }
        # Include per-modality reconstruction losses.
        for key, val in recon_losses.items():
            if key.startswith("recon/"):
                losses[key] = val

        return losses

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Alias for :meth:`compute_foundation_loss` (nn.Module interface).

        Args:
            batch: See :meth:`compute_foundation_loss`.

        Returns:
            Loss dictionary.
        """
        return self.compute_foundation_loss(batch)


# ---------------------------------------------------------------------------
# Encoder wrapper: run frozen encoders and collect projected embeddings
# ---------------------------------------------------------------------------


class FrozenEncoderBank(nn.Module):
    """Wraps all five encoders, freezes them, and extracts projected embeddings.

    This is a convenience module for end-to-end foundation pretraining
    when raw modality data (not pre-computed embeddings) is available.
    Each encoder is set to eval mode with all parameters frozen, and
    forward passes are executed under ``torch.no_grad()``.

    Args:
        sensor_encoder: AquaSSM sensor encoder instance.
        satellite_encoder: HydroViT satellite encoder instance.
        microbial_encoder: MicroBiomeNet microbial encoder instance.
        molecular_encoder: ToxiGene molecular encoder instance.
        behavioral_encoder: BioMotion behavioral encoder instance.
    """

    def __init__(
        self,
        sensor_encoder: nn.Module,
        satellite_encoder: nn.Module,
        microbial_encoder: nn.Module,
        molecular_encoder: nn.Module,
        behavioral_encoder: nn.Module,
    ) -> None:
        super().__init__()
        self.encoders: Dict[str, nn.Module] = {}

        # Store and freeze each encoder.
        encoder_map = {
            "sensor": sensor_encoder,
            "satellite": satellite_encoder,
            "microbial": microbial_encoder,
            "molecular": molecular_encoder,
            "behavioral": behavioral_encoder,
        }

        for mid, encoder in encoder_map.items():
            encoder.eval()
            for param in encoder.parameters():
                param.requires_grad = False
            # Register as a submodule so it moves with .to(device).
            self.add_module(f"encoder_{mid}", encoder)
            self.encoders[mid] = encoder

    def train(self, mode: bool = True) -> "FrozenEncoderBank":
        """Override train() to keep encoders in eval mode always."""
        super().train(mode)
        for encoder in self.encoders.values():
            encoder.eval()
        return self

    @torch.no_grad()
    def encode_sensor(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode sensor data through frozen AquaSSM.

        Args:
            batch: Must contain ``"sensor_data"`` key with tensor
                ``[B, T, P]``.

        Returns:
            Projected embedding ``[B, 256]``.
        """
        x = batch["sensor_data"]
        delta_ts = batch.get("sensor_delta_ts")
        result = self.encoders["sensor"](
            x, delta_ts=delta_ts, compute_anomaly=False
        )
        return result["fusion_embedding"]

    @torch.no_grad()
    def encode_satellite(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode satellite data through frozen HydroViT.

        Args:
            batch: Must contain ``"satellite_image"`` key with tensor
                ``[B, 13, 224, 224]``.

        Returns:
            Projected embedding ``[B, 256]``.
        """
        image = batch["satellite_image"]
        s3_tokens = batch.get("s3_tokens")
        result = self.encoders["satellite"](image, s3_tokens=s3_tokens)
        return result["fusion_embedding"]

    @torch.no_grad()
    def encode_microbial(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode microbial data through frozen MicroBiomeNet.

        Args:
            batch: Must contain ``"microbial_abundances"`` key with
                tensor ``[B, n_otus]`` (CLR-transformed).

        Returns:
            Projected embedding ``[B, 256]``.
        """
        x = batch["microbial_abundances"]
        raw = batch.get("microbial_raw_abundances")
        result = self.encoders["microbial"](x, raw_abundances=raw)
        return result["fusion_embedding"]

    @torch.no_grad()
    def encode_molecular(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode molecular data through frozen ToxiGene.

        Args:
            batch: Must contain either ``"gene_expression"`` ``[B, n_genes]``
                or ``"chem_class"`` + ``"log_concentration"``.

        Returns:
            Projected embedding ``[B, 256]``.
        """
        gene_expr = batch.get("gene_expression")
        chem_class = batch.get("chem_class")
        log_conc = batch.get("log_concentration")
        result = self.encoders["molecular"](
            gene_expression=gene_expr,
            chem_class=chem_class,
            log_concentration=log_conc,
        )
        return result["fusion_embedding"]

    @torch.no_grad()
    def encode_behavioral(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode behavioral data through frozen BioMotion.

        Args:
            batch: Must contain ``"organism_inputs"`` dict keyed by
                species name.

        Returns:
            Projected embedding ``[B, 256]``.
        """
        organism_inputs = batch["organism_inputs"]
        result = self.encoders["behavioral"](organism_inputs)
        return result["fusion_embedding"]

    def encode_available(
        self,
        batch: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        """Encode all available modalities in the batch.

        Detects which modalities have data present in the batch and
        encodes them through the corresponding frozen encoder.

        Args:
            batch: Multi-modal batch dict with modality-specific keys.

        Returns:
            Dict mapping modality id to projected embedding ``[B, 256]``.
        """
        embeddings: Dict[str, torch.Tensor] = {}

        # Sensor
        if "sensor_data" in batch:
            embeddings["sensor"] = self.encode_sensor(batch)

        # Satellite
        if "satellite_image" in batch:
            embeddings["satellite"] = self.encode_satellite(batch)

        # Microbial
        if "microbial_abundances" in batch:
            embeddings["microbial"] = self.encode_microbial(batch)

        # Molecular
        if "gene_expression" in batch or "chem_class" in batch:
            embeddings["molecular"] = self.encode_molecular(batch)

        # Behavioral
        if "organism_inputs" in batch:
            embeddings["behavioral"] = self.encode_behavioral(batch)

        return embeddings


# ---------------------------------------------------------------------------
# Full training wrapper
# ---------------------------------------------------------------------------


class FoundationPretrainModel(nn.Module):
    """Complete foundation pretraining model combining frozen encoders,
    fusion, and the pretraining objective.

    This module is the top-level entry point for foundation pretraining.
    Given a raw multi-modal batch, it:

    1. Runs frozen encoders to produce projected embeddings.
    2. Passes the embeddings to :class:`FoundationPretrainObjective`
       which handles masking, fusion, reconstruction, and loss computation.

    Only the fusion layer parameters and reconstructor heads receive
    gradients.

    Args:
        encoder_bank: :class:`FrozenEncoderBank` wrapping all five encoders.
        objective: :class:`FoundationPretrainObjective` with fusion and heads.
    """

    def __init__(
        self,
        encoder_bank: FrozenEncoderBank,
        objective: FoundationPretrainObjective,
    ) -> None:
        super().__init__()
        self.encoder_bank = encoder_bank
        self.objective = objective

    def forward(
        self,
        batch: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        """End-to-end forward pass from raw modality data to losses.

        Args:
            batch: Multi-modal batch dict.  May contain:

                - ``"sensor_data"``: ``[B, T, P]``
                - ``"satellite_image"``: ``[B, 13, 224, 224]``
                - ``"microbial_abundances"``: ``[B, n_otus]``
                - ``"gene_expression"``: ``[B, n_genes]``
                - ``"organism_inputs"``: ``{species: {keypoints, features, ...}}``

                Or, if pre-computed embeddings are available:

                - ``"sensor"``: ``[B, 256]``
                - ``"satellite"``: ``[B, 256]``
                - etc.

        Returns:
            Loss dictionary from
            :meth:`FoundationPretrainObjective.compute_foundation_loss`.
        """
        # Check if embeddings are already pre-computed (keys match MODALITY_IDS).
        has_precomputed = any(mid in batch for mid in MODALITY_IDS)
        has_raw = any(
            k in batch
            for k in [
                "sensor_data",
                "satellite_image",
                "microbial_abundances",
                "gene_expression",
                "organism_inputs",
            ]
        )

        if has_precomputed and not has_raw:
            # Embeddings are already in the batch.
            embeddings = {
                mid: batch[mid] for mid in MODALITY_IDS if mid in batch
            }
        elif has_raw:
            # Run frozen encoders.
            embeddings = self.encoder_bank.encode_available(batch)
        else:
            raise ValueError(
                "Batch must contain either pre-computed modality embeddings "
                "or raw modality data."
            )

        return self.objective.compute_foundation_loss(embeddings)


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def build_foundation_objective(
    fusion: Optional[PerceiverIOFusion] = None,
    shared_embed_dim: int = SHARED_EMBED_DIM,
    mask_prob: float = 0.3,
    mask_mode: str = "random",
    consistency_weight: float = 0.1,
    presence_weight: float = 0.05,
    num_latents: int = 256,
    num_heads: int = 8,
    num_process_layers: int = 2,
    dropout: float = 0.1,
) -> FoundationPretrainObjective:
    """Build a :class:`FoundationPretrainObjective` with defaults.

    Args:
        fusion: Optional pre-existing fusion model.  If ``None``, a new
            :class:`PerceiverIOFusion` is created.
        shared_embed_dim: Shared embedding dim. Default 256.
        mask_prob: Masking probability. Default 0.3.
        mask_mode: Masking strategy. Default ``"random"``.
        consistency_weight: InfoNCE loss weight. Default 0.1.
        presence_weight: Presence prediction loss weight. Default 0.05.
        num_latents: Perceiver latent array size. Default 256.
        num_heads: Attention heads. Default 8.
        num_process_layers: Perceiver process layers. Default 2.
        dropout: Dropout rate. Default 0.1.

    Returns:
        Configured :class:`FoundationPretrainObjective`.
    """
    if fusion is None:
        fusion = PerceiverIOFusion(
            shared_dim=shared_embed_dim,
            num_latents=num_latents,
            num_heads=num_heads,
            num_process_layers=num_process_layers,
            dropout=dropout,
        )

    return FoundationPretrainObjective(
        fusion=fusion,
        shared_embed_dim=shared_embed_dim,
        mask_prob=mask_prob,
        mask_mode=mask_mode,
        consistency_weight=consistency_weight,
        presence_weight=presence_weight,
        dropout=dropout,
    )


def get_trainable_params(
    objective: FoundationPretrainObjective,
) -> List[nn.Parameter]:
    """Return only the trainable parameters for foundation pretraining.

    This excludes frozen encoder parameters and returns only:
    - Fusion layer parameters
    - Reconstructor head parameters
    - [MASK] token parameters
    - Modality presence head parameters
    - InfoNCE temperature

    Args:
        objective: The foundation pretraining objective module.

    Returns:
        List of trainable parameters suitable for an optimizer.
    """
    params: List[nn.Parameter] = []

    # Fusion layer
    params.extend(objective.fusion.parameters())

    # Reconstructors
    params.extend(objective.reconstructors.parameters())

    # Mask tokens
    for p in objective.mask_tokens.values():
        params.append(p)

    # Presence head
    params.extend(objective.presence_head.parameters())

    # Temperature
    params.append(objective.log_temperature)

    return params
