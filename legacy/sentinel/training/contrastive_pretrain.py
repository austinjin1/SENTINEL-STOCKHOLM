"""CLIP-style cross-modal contrastive pretraining for SENTINEL encoders.

Implements InfoNCE-based contrastive learning between pairs of modality
encoders to force shared representational structure.  This enables
zero-shot transfer when modalities are missing at inference time: a
query in one modality can retrieve matching observations in another.

Contrastive pairs and their temporal/spatial constraints::

    Sensor   <-> Satellite  : same H3 hex (res 8), within 48 hours
    Sensor   <-> Microbial  : same watershed,       within 30 days
    Microbial <-> Molecular  : paired EMP-GEO datasets
    Behavioral <-> Sensor    : matched ECOTOX concentrations

Architecture overview:

1. **CrossModalContrastiveLoss** -- CLIP-style symmetric InfoNCE loss with
   a learnable temperature parameter and optional hard negative mining.

2. **ProjectionHead** -- Learnable 2-layer MLP that projects 256-dim
   encoder embeddings into a 128-dim shared contrastive space with L2
   normalisation.

3. **ContrastivePretrainer** -- Training orchestrator that wraps a pair
   of *frozen* encoders, adds projection heads, and computes the
   contrastive objective across a batch of matched observations.

4. **PairSampler** -- Sampling utility that finds co-located observations
   across modality pairs given a multimodal index, respecting
   spatial (H3 / distance) and temporal constraints.

5. **ContrastiveEvaluator** -- Evaluation utilities: zero-shot retrieval
   (Recall@k) and embedding alignment (cosine similarity distribution).

6. **train_contrastive_epoch()** -- Standalone epoch-level training loop
   compatible with the SENTINEL trainer infrastructure.

Usage::

    python -m sentinel.training.contrastive_pretrain \\
        --pair sensor-satellite \\
        --encoder-dir outputs/pretrained_encoders \\
        --index-path data/aligned/multimodal_index.parquet \\
        --output-dir outputs/contrastive
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

from sentinel.models.fusion.embedding_registry import (
    MODALITY_IDS,
    SHARED_EMBEDDING_DIM,
)
from sentinel.utils.logging import get_logger, make_progress

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ENCODER_EMBED_DIM: int = SHARED_EMBEDDING_DIM  # 256
CONTRASTIVE_DIM: int = 128  # projection head output

# Canonical contrastive pair definitions with default temporal windows.
# Each entry: (modality_a, modality_b, temporal_window_seconds, spatial_mode)
CONTRASTIVE_PAIRS: Dict[str, Tuple[str, str, float, str]] = {
    "sensor-satellite": ("sensor", "satellite", 48 * 3600, "h3"),
    "sensor-microbial": ("sensor", "microbial", 30 * 86400, "watershed"),
    "microbial-molecular": ("microbial", "molecular", float("inf"), "paired"),
    "behavioral-sensor": ("behavioral", "sensor", float("inf"), "ecotox"),
}

# Maximum distance in meters for fallback spatial matching.
MAX_SPATIAL_DISTANCE_M: float = 500.0

# H3 resolution used for spatial co-location.
H3_RESOLUTION: int = 8


# ---------------------------------------------------------------------------
# CrossModalContrastiveLoss
# ---------------------------------------------------------------------------


class CrossModalContrastiveLoss(nn.Module):
    """CLIP-style symmetric InfoNCE loss for cross-modal contrastive learning.

    Given two batches of L2-normalised embeddings from different modalities,
    computes the symmetric contrastive loss: the average of the
    modality_a -> modality_b and modality_b -> modality_a cross-entropy
    losses over the similarity matrix scaled by a learnable temperature.

    Args:
        init_temperature: Initial value of the temperature parameter.
            Following CLIP, this is the *inverse* temperature (i.e. the
            logits are ``similarity / temperature``).  Default 0.07.
        learnable_temperature: If True (default), temperature is a
            learnable parameter updated via gradient descent.
        max_temperature: Upper clamp for the temperature to prevent
            training instability.  Default 100.0 (for 1/t scaling).
        hard_negative_mining: If True, upweight the hardest negatives
            in each row of the similarity matrix.
        hard_negative_fraction: Fraction of negatives to treat as
            hard negatives (top-k by similarity).  Default 0.2.
        label_smoothing: Optional label smoothing for the cross-entropy.
            Default 0.0 (no smoothing).
        symmetric: If True (default), compute loss in both directions
            and average.  If False, compute only a -> b.
    """

    def __init__(
        self,
        init_temperature: float = 0.07,
        learnable_temperature: bool = True,
        max_temperature: float = 100.0,
        hard_negative_mining: bool = False,
        hard_negative_fraction: float = 0.2,
        label_smoothing: float = 0.0,
        symmetric: bool = True,
    ) -> None:
        super().__init__()

        # Store temperature as log(1/t) so the effective scale is exp(log_scale).
        # This parameterisation keeps the scale positive without clamping.
        log_scale = math.log(1.0 / init_temperature)
        if learnable_temperature:
            self.log_scale = nn.Parameter(torch.tensor(log_scale))
        else:
            self.register_buffer("log_scale", torch.tensor(log_scale))

        self.max_log_scale = math.log(max_temperature)
        self.hard_negative_mining = hard_negative_mining
        self.hard_negative_fraction = hard_negative_fraction
        self.label_smoothing = label_smoothing
        self.symmetric = symmetric

    @property
    def temperature(self) -> float:
        """Current temperature value (scalar)."""
        return 1.0 / self.log_scale.exp().item()

    def forward(
        self,
        emb_a: torch.Tensor,
        emb_b: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute the symmetric InfoNCE contrastive loss.

        Args:
            emb_a: Embeddings from modality A, shape ``[B, D]``.
                Must be L2-normalised.
            emb_b: Embeddings from modality B, shape ``[B, D]``.
                Must be L2-normalised.
            weights: Optional per-sample weights ``[B]`` for
                importance-weighted contrastive learning.

        Returns:
            Dict with keys:
                ``loss``: Scalar contrastive loss.
                ``loss_a2b``: Loss for the a -> b direction.
                ``loss_b2a``: Loss for the b -> a direction.
                ``accuracy_a2b``: Top-1 retrieval accuracy a -> b.
                ``accuracy_b2a``: Top-1 retrieval accuracy b -> a.
                ``temperature``: Current temperature value.
        """
        B, D = emb_a.shape
        assert emb_b.shape == (B, D), (
            f"Shape mismatch: emb_a={emb_a.shape}, emb_b={emb_b.shape}"
        )

        # Clamp log_scale to prevent runaway temperature.
        log_scale = torch.clamp(self.log_scale, max=self.max_log_scale)
        scale = log_scale.exp()  # 1 / temperature

        # Cosine similarity matrix: [B, B]
        logits = emb_a @ emb_b.T * scale

        # Apply hard negative mining if requested.
        if self.hard_negative_mining:
            logits = self._apply_hard_negatives(logits)

        # Ground-truth: diagonal entries are positives.
        labels = torch.arange(B, device=emb_a.device)

        # a -> b direction.
        loss_a2b = F.cross_entropy(
            logits, labels, label_smoothing=self.label_smoothing,
            reduction="none",
        )
        # b -> a direction.
        loss_b2a = F.cross_entropy(
            logits.T, labels, label_smoothing=self.label_smoothing,
            reduction="none",
        )

        # Apply sample weights if provided.
        if weights is not None:
            assert weights.shape == (B,), f"weights shape {weights.shape} != ({B},)"
            w = weights / weights.sum()
            loss_a2b_mean = (loss_a2b * w).sum()
            loss_b2a_mean = (loss_b2a * w).sum()
        else:
            loss_a2b_mean = loss_a2b.mean()
            loss_b2a_mean = loss_b2a.mean()

        if self.symmetric:
            loss = (loss_a2b_mean + loss_b2a_mean) / 2.0
        else:
            loss = loss_a2b_mean

        # Retrieval accuracy (top-1).
        with torch.no_grad():
            acc_a2b = (logits.argmax(dim=1) == labels).float().mean()
            acc_b2a = (logits.T.argmax(dim=1) == labels).float().mean()

        return {
            "loss": loss,
            "loss_a2b": loss_a2b_mean.detach(),
            "loss_b2a": loss_b2a_mean.detach(),
            "accuracy_a2b": acc_a2b,
            "accuracy_b2a": acc_b2a,
            "temperature": torch.tensor(
                self.temperature, device=emb_a.device
            ),
        }

    def _apply_hard_negatives(self, logits: torch.Tensor) -> torch.Tensor:
        """Upweight hard negatives by zeroing out easy negatives.

        For each row, keep only the top-k hardest negatives (highest
        similarity) and the positive.  Easy negatives are set to
        ``-inf`` so they contribute zero probability.

        Args:
            logits: Similarity matrix ``[B, B]``.

        Returns:
            Modified logits with easy negatives masked.
        """
        B = logits.shape[0]
        k = max(1, int(B * self.hard_negative_fraction))

        # Mask the diagonal (positives) before finding hard negatives.
        diag_mask = torch.eye(B, dtype=torch.bool, device=logits.device)
        neg_logits = logits.masked_fill(diag_mask, float("-inf"))

        # Top-k hardest negatives per row.
        _, hard_indices = neg_logits.topk(k, dim=1)

        # Build a mask: True for positives + hard negatives.
        keep_mask = diag_mask.clone()
        keep_mask.scatter_(1, hard_indices, True)

        # Set easy negatives to -inf.
        return logits.masked_fill(~keep_mask, float("-inf"))


# ---------------------------------------------------------------------------
# ProjectionHead
# ---------------------------------------------------------------------------


class ProjectionHead(nn.Module):
    """Learnable MLP projection from encoder space to contrastive space.

    Architecture: Linear(in_dim, out_dim) -> BN -> ReLU -> Linear(out_dim, out_dim) -> L2 normalise.

    The projection head is the *only* learnable component during
    contrastive pretraining; the encoder weights remain frozen.

    Args:
        in_dim: Input embedding dimension (from encoder).  Default 256.
        out_dim: Output embedding dimension (contrastive space).
            Default 128.
    """

    def __init__(
        self,
        in_dim: int = ENCODER_EMBED_DIM,
        out_dim: int = CONTRASTIVE_DIM,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier-uniform initialisation for Linear layers."""
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project and L2-normalise.

        Args:
            x: Encoder embeddings ``[B, in_dim]``.

        Returns:
            L2-normalised projections ``[B, out_dim]``.
        """
        z = self.net(x)
        return F.normalize(z, p=2, dim=-1)


# ---------------------------------------------------------------------------
# ContrastivePretrainer
# ---------------------------------------------------------------------------


class ContrastivePretrainer(nn.Module):
    """Training orchestrator for cross-modal contrastive pretraining.

    Wraps a pair of frozen modality encoders with learnable projection
    heads and computes the CLIP-style InfoNCE loss over matched batches.

    Gradient flows **only** through the projection heads and the
    temperature parameter -- encoder weights are frozen.

    Args:
        encoder_a: Pretrained encoder for modality A.  Will be frozen.
        encoder_b: Pretrained encoder for modality B.  Will be frozen.
        modality_a: Name of modality A (e.g. ``"sensor"``).
        modality_b: Name of modality B (e.g. ``"satellite"``).
        encoder_dim: Encoder output embedding dimension.  Default 256.
        contrastive_dim: Contrastive projection dimension.  Default 128.
        init_temperature: Initial InfoNCE temperature.  Default 0.07.
        hard_negative_mining: Enable hard negative mining.
        hard_negative_fraction: Fraction of negatives to keep.
        label_smoothing: Cross-entropy label smoothing.
        symmetric: Use symmetric (both directions) loss.
    """

    def __init__(
        self,
        encoder_a: nn.Module,
        encoder_b: nn.Module,
        modality_a: str = "sensor",
        modality_b: str = "satellite",
        encoder_dim: int = ENCODER_EMBED_DIM,
        contrastive_dim: int = CONTRASTIVE_DIM,
        init_temperature: float = 0.07,
        hard_negative_mining: bool = False,
        hard_negative_fraction: float = 0.2,
        label_smoothing: float = 0.0,
        symmetric: bool = True,
    ) -> None:
        super().__init__()

        self.modality_a = modality_a
        self.modality_b = modality_b

        # Freeze encoders -- no gradient flows through them.
        self.encoder_a = encoder_a
        self.encoder_b = encoder_b
        self._freeze_encoder(self.encoder_a)
        self._freeze_encoder(self.encoder_b)

        # Learnable projection heads.
        self.proj_a = ProjectionHead(in_dim=encoder_dim, out_dim=contrastive_dim)
        self.proj_b = ProjectionHead(in_dim=encoder_dim, out_dim=contrastive_dim)

        # Contrastive loss.
        self.criterion = CrossModalContrastiveLoss(
            init_temperature=init_temperature,
            learnable_temperature=True,
            hard_negative_mining=hard_negative_mining,
            hard_negative_fraction=hard_negative_fraction,
            label_smoothing=label_smoothing,
            symmetric=symmetric,
        )

    @staticmethod
    def _freeze_encoder(encoder: nn.Module) -> None:
        """Freeze all parameters and set to eval mode."""
        encoder.eval()
        for param in encoder.parameters():
            param.requires_grad = False

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Return only the parameters that receive gradient updates.

        These are the two projection heads and the temperature parameter.
        """
        params: list[nn.Parameter] = []
        params.extend(self.proj_a.parameters())
        params.extend(self.proj_b.parameters())
        # Temperature is a parameter on the loss module.
        if isinstance(self.criterion.log_scale, nn.Parameter):
            params.append(self.criterion.log_scale)
        return params

    def forward(
        self,
        batch_a: Any,
        batch_b: Any,
        weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute contrastive loss for a matched batch.

        Args:
            batch_a: Input for encoder A.  Either a pre-computed
                embedding tensor ``[B, D]`` or a dict/tuple that
                the encoder accepts as ``forward()`` input.
            batch_b: Input for encoder B (same convention).
            weights: Optional per-sample importance weights ``[B]``.

        Returns:
            Dict of loss and metric tensors (see
            :meth:`CrossModalContrastiveLoss.forward`).
        """
        # Extract embeddings from encoders.
        emb_a = self._encode(self.encoder_a, batch_a)
        emb_b = self._encode(self.encoder_b, batch_b)

        # Project to contrastive space (includes L2 normalisation).
        z_a = self.proj_a(emb_a)
        z_b = self.proj_b(emb_b)

        # Compute contrastive loss.
        return self.criterion(z_a, z_b, weights=weights)

    @staticmethod
    def _encode(encoder: nn.Module, batch: Any) -> torch.Tensor:
        """Extract the ``"embedding"`` from an encoder's output.

        Handles three input conventions:
        1. Pre-computed embedding tensor ``[B, D]`` -- pass through.
        2. Dict input -- call ``encoder(**batch)`` and extract
           ``"embedding"``.
        3. Tuple/list input -- call ``encoder(*batch)`` and extract
           ``"embedding"``.

        Returns:
            Detached embedding tensor ``[B, D]``.
        """
        if isinstance(batch, torch.Tensor) and batch.ndim == 2:
            # Pre-computed embeddings.
            return batch.detach()

        with torch.no_grad():
            if isinstance(batch, dict):
                out = encoder(**batch)
            elif isinstance(batch, (tuple, list)):
                out = encoder(*batch)
            else:
                out = encoder(batch)

        if isinstance(out, dict):
            return out["embedding"].detach()
        elif isinstance(out, torch.Tensor):
            return out.detach()
        else:
            raise TypeError(
                f"Encoder returned unsupported type {type(out)}. "
                f"Expected dict with 'embedding' key or a Tensor."
            )


# ---------------------------------------------------------------------------
# PairSampler
# ---------------------------------------------------------------------------


@dataclass
class PairConfig:
    """Configuration for a single contrastive pair.

    Attributes:
        modality_a: First modality name.
        modality_b: Second modality name.
        temporal_window_s: Maximum allowed time difference in seconds
            between matched observations.  ``float('inf')`` disables
            temporal filtering.
        spatial_mode: One of ``"h3"`` (same H3 hex), ``"watershed"``
            (same watershed id), ``"paired"`` (pre-linked dataset ids),
            ``"ecotox"`` (matched via ECOTOX concentration records),
            or ``"distance"`` (within ``max_distance_m``).
        max_distance_m: Fallback maximum distance in meters for the
            ``"distance"`` spatial mode.
        h3_resolution: H3 resolution for the ``"h3"`` mode.
    """

    modality_a: str = "sensor"
    modality_b: str = "satellite"
    temporal_window_s: float = 48 * 3600
    spatial_mode: str = "h3"
    max_distance_m: float = MAX_SPATIAL_DISTANCE_M
    h3_resolution: int = H3_RESOLUTION


# Default pair configs matching the four contrastive objectives.
DEFAULT_PAIR_CONFIGS: Dict[str, PairConfig] = {
    "sensor-satellite": PairConfig(
        modality_a="sensor",
        modality_b="satellite",
        temporal_window_s=48 * 3600,
        spatial_mode="h3",
    ),
    "sensor-microbial": PairConfig(
        modality_a="sensor",
        modality_b="microbial",
        temporal_window_s=30 * 86400,
        spatial_mode="watershed",
    ),
    "microbial-molecular": PairConfig(
        modality_a="microbial",
        modality_b="molecular",
        temporal_window_s=float("inf"),
        spatial_mode="paired",
    ),
    "behavioral-sensor": PairConfig(
        modality_a="behavioral",
        modality_b="sensor",
        temporal_window_s=float("inf"),
        spatial_mode="ecotox",
    ),
}


class PairSampler:
    """Finds co-located observation pairs from a multimodal index.

    Given a multimodal index (a pandas DataFrame or list of dicts with
    columns like ``site_id``, ``h3_index``, ``watershed_id``,
    ``timestamp_s``, ``modality``, ``observation_id``), builds an
    inverted index for efficient retrieval of matching pairs that
    satisfy both spatial and temporal constraints.

    Args:
        index: Multimodal observation index.  Must contain columns:
            ``observation_id``, ``modality``, ``timestamp_s``,
            ``h3_index``, ``watershed_id``, and optionally
            ``dataset_id`` (for paired mode) and ``ecotox_group``
            (for ECOTOX matching).
        pair_config: Configuration for the contrastive pair.
        max_pairs_per_anchor: Maximum number of candidate partners
            to retain per anchor observation.
        seed: Random seed for reproducible sampling.
    """

    def __init__(
        self,
        index: Any,
        pair_config: PairConfig,
        max_pairs_per_anchor: int = 10,
        seed: int = 42,
    ) -> None:
        self.pair_config = pair_config
        self.max_pairs_per_anchor = max_pairs_per_anchor
        self.rng = np.random.RandomState(seed)

        # Build the inverted spatial index.
        self._pairs: List[Tuple[Any, Any]] = []
        self._build_index(index)

    def _build_index(self, index: Any) -> None:
        """Build matched pairs from the multimodal index.

        Iterates over observations for ``modality_a`` and finds all
        compatible ``modality_b`` observations satisfying spatial and
        temporal constraints.
        """
        import pandas as pd

        if not isinstance(index, pd.DataFrame):
            index = pd.DataFrame(index)

        cfg = self.pair_config

        # Separate observations by modality.
        obs_a = index[index["modality"] == cfg.modality_a].reset_index(drop=True)
        obs_b = index[index["modality"] == cfg.modality_b].reset_index(drop=True)

        if obs_a.empty or obs_b.empty:
            logger.warning(
                f"No observations for pair {cfg.modality_a}-{cfg.modality_b}. "
                f"Got {len(obs_a)} {cfg.modality_a}, {len(obs_b)} {cfg.modality_b}."
            )
            return

        # Build spatial lookup for modality_b.
        spatial_groups_b = self._group_by_spatial_key(obs_b, cfg.spatial_mode)

        n_pairs = 0
        for _, row_a in obs_a.iterrows():
            spatial_key = self._get_spatial_key(row_a, cfg.spatial_mode)
            if spatial_key is None:
                continue

            candidates = spatial_groups_b.get(spatial_key, [])
            if not candidates:
                continue

            # Temporal filtering.
            if math.isfinite(cfg.temporal_window_s):
                t_a = row_a.get("timestamp_s", 0.0)
                candidates = [
                    c for c in candidates
                    if abs(c.get("timestamp_s", 0.0) - t_a) <= cfg.temporal_window_s
                ]

            if not candidates:
                continue

            # Subsample if too many candidates.
            if len(candidates) > self.max_pairs_per_anchor:
                indices = self.rng.choice(
                    len(candidates), self.max_pairs_per_anchor, replace=False
                )
                candidates = [candidates[i] for i in indices]

            for c in candidates:
                self._pairs.append(
                    (row_a["observation_id"], c["observation_id"])
                )
                n_pairs += 1

        logger.info(
            f"PairSampler built {n_pairs} pairs for "
            f"{cfg.modality_a} <-> {cfg.modality_b}"
        )

    @staticmethod
    def _get_spatial_key(row: Any, spatial_mode: str) -> Optional[str]:
        """Extract the spatial grouping key from an observation row."""
        if spatial_mode == "h3":
            return row.get("h3_index")
        elif spatial_mode == "watershed":
            return row.get("watershed_id")
        elif spatial_mode == "paired":
            return row.get("dataset_id")
        elif spatial_mode == "ecotox":
            return row.get("ecotox_group")
        elif spatial_mode == "distance":
            # For distance mode, use H3 as a coarse filter, then refine.
            return row.get("h3_index")
        return None

    @staticmethod
    def _group_by_spatial_key(
        obs: Any, spatial_mode: str
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group observations by their spatial key for fast lookup."""
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for _, row in obs.iterrows():
            key = PairSampler._get_spatial_key(row, spatial_mode)
            if key is not None:
                groups.setdefault(key, []).append(row.to_dict())
        return groups

    def sample_batch(self, batch_size: int) -> List[Tuple[Any, Any]]:
        """Sample a batch of matched (obs_id_a, obs_id_b) pairs.

        Args:
            batch_size: Number of pairs to sample.

        Returns:
            List of ``(observation_id_a, observation_id_b)`` tuples.

        Raises:
            ValueError: If fewer pairs are available than requested.
        """
        if len(self._pairs) == 0:
            raise ValueError("No matched pairs available for sampling.")

        if batch_size > len(self._pairs):
            # Sample with replacement if not enough unique pairs.
            indices = self.rng.choice(
                len(self._pairs), batch_size, replace=True
            )
        else:
            indices = self.rng.choice(
                len(self._pairs), batch_size, replace=False
            )
        return [self._pairs[i] for i in indices]

    @property
    def num_pairs(self) -> int:
        """Total number of matched pairs in the index."""
        return len(self._pairs)

    def __len__(self) -> int:
        return len(self._pairs)


# ---------------------------------------------------------------------------
# ContrastivePairDataset
# ---------------------------------------------------------------------------


class ContrastivePairDataset(Dataset):
    """PyTorch Dataset wrapper for contrastive pair sampling.

    Wraps a :class:`PairSampler` and an embedding cache (or raw data
    loaders) to produce ``(embedding_a, embedding_b)`` pairs for
    training.

    For efficiency, this dataset expects pre-computed encoder embeddings
    stored as dicts mapping ``observation_id -> Tensor[D]``.

    Args:
        pair_sampler: A configured :class:`PairSampler`.
        embeddings_a: Dict mapping observation ids to modality-A
            embeddings ``[D]``.
        embeddings_b: Dict mapping observation ids to modality-B
            embeddings ``[D]``.
    """

    def __init__(
        self,
        pair_sampler: PairSampler,
        embeddings_a: Dict[Any, torch.Tensor],
        embeddings_b: Dict[Any, torch.Tensor],
    ) -> None:
        self.pairs = pair_sampler._pairs
        self.embeddings_a = embeddings_a
        self.embeddings_b = embeddings_b

        # Filter to pairs where both embeddings exist.
        self.valid_pairs = [
            (a, b) for a, b in self.pairs
            if a in self.embeddings_a and b in self.embeddings_b
        ]
        if len(self.valid_pairs) < len(self.pairs):
            logger.warning(
                f"Filtered {len(self.pairs) - len(self.valid_pairs)} pairs "
                f"with missing embeddings. {len(self.valid_pairs)} remain."
            )

    def __len__(self) -> int:
        return len(self.valid_pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        obs_a, obs_b = self.valid_pairs[idx]
        return self.embeddings_a[obs_a], self.embeddings_b[obs_b]


# ---------------------------------------------------------------------------
# ContrastiveEvaluator
# ---------------------------------------------------------------------------


class ContrastiveEvaluator:
    """Evaluation utilities for cross-modal contrastive pretraining.

    Computes zero-shot cross-modal retrieval metrics and embedding
    alignment statistics.

    Args:
        proj_a: Projection head for modality A.
        proj_b: Projection head for modality B.
        device: Torch device for computation.
    """

    def __init__(
        self,
        proj_a: ProjectionHead,
        proj_b: ProjectionHead,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.proj_a = proj_a
        self.proj_b = proj_b
        self.device = device

    @torch.no_grad()
    def zero_shot_retrieval(
        self,
        emb_a: torch.Tensor,
        emb_b: torch.Tensor,
        k_values: Sequence[int] = (1, 5, 10),
    ) -> Dict[str, float]:
        """Compute zero-shot cross-modal retrieval recall@k.

        Given matched pairs ``(emb_a[i], emb_b[i])``, project both
        into contrastive space and compute recall at various k values
        for both retrieval directions.

        Args:
            emb_a: Encoder embeddings for modality A ``[N, D]``.
            emb_b: Encoder embeddings for modality B ``[N, D]``.
            k_values: Tuple of k values for recall@k computation.

        Returns:
            Dict with keys like ``"recall@1_a2b"``, ``"recall@5_b2a"``
            etc., plus ``"mean_cosine_sim"`` for diagonal alignment.
        """
        self.proj_a.eval()
        self.proj_b.eval()

        emb_a = emb_a.to(self.device)
        emb_b = emb_b.to(self.device)

        z_a = self.proj_a(emb_a)  # [N, C], L2-normalised
        z_b = self.proj_b(emb_b)  # [N, C], L2-normalised

        N = z_a.shape[0]
        labels = torch.arange(N, device=self.device)

        # Similarity matrix.
        sim = z_a @ z_b.T  # [N, N]

        metrics: Dict[str, float] = {}

        # Recall@k in both directions.
        for k in k_values:
            if k > N:
                k = N

            # a -> b retrieval.
            _, topk_a2b = sim.topk(k, dim=1)
            hits_a2b = (topk_a2b == labels.unsqueeze(1)).any(dim=1).float()
            metrics[f"recall@{k}_a2b"] = hits_a2b.mean().item()

            # b -> a retrieval.
            _, topk_b2a = sim.T.topk(k, dim=1)
            hits_b2a = (topk_b2a == labels.unsqueeze(1)).any(dim=1).float()
            metrics[f"recall@{k}_b2a"] = hits_b2a.mean().item()

        # Embedding alignment: mean diagonal cosine similarity.
        diag_sim = (z_a * z_b).sum(dim=-1)
        metrics["mean_cosine_sim"] = diag_sim.mean().item()
        metrics["std_cosine_sim"] = diag_sim.std().item()
        metrics["median_cosine_sim"] = diag_sim.median().item()

        return metrics

    @torch.no_grad()
    def embedding_alignment_histogram(
        self,
        emb_a: torch.Tensor,
        emb_b: torch.Tensor,
        n_bins: int = 50,
    ) -> Dict[str, Any]:
        """Compute cosine similarity distribution between matched pairs.

        Args:
            emb_a: Encoder embeddings for modality A ``[N, D]``.
            emb_b: Encoder embeddings for modality B ``[N, D]``.
            n_bins: Number of histogram bins.

        Returns:
            Dict with ``"bin_edges"``, ``"counts"`` (positive pairs),
            and ``"neg_counts"`` (random negative pairs).
        """
        self.proj_a.eval()
        self.proj_b.eval()

        z_a = self.proj_a(emb_a.to(self.device))
        z_b = self.proj_b(emb_b.to(self.device))

        # Positive pair cosine similarities (diagonal).
        pos_sim = (z_a * z_b).sum(dim=-1).cpu().numpy()

        # Negative pair similarities: shift z_b by 1 position.
        z_b_neg = torch.roll(z_b, shifts=1, dims=0)
        neg_sim = (z_a * z_b_neg).sum(dim=-1).cpu().numpy()

        bin_edges = np.linspace(-1.0, 1.0, n_bins + 1)
        pos_counts, _ = np.histogram(pos_sim, bins=bin_edges)
        neg_counts, _ = np.histogram(neg_sim, bins=bin_edges)

        return {
            "bin_edges": bin_edges.tolist(),
            "pos_counts": pos_counts.tolist(),
            "neg_counts": neg_counts.tolist(),
            "pos_mean": float(pos_sim.mean()),
            "neg_mean": float(neg_sim.mean()),
        }


# ---------------------------------------------------------------------------
# ContrastivePretrainConfig
# ---------------------------------------------------------------------------


@dataclass
class ContrastivePretrainConfig:
    """Configuration for contrastive pretraining.

    Attributes:
        pair_name: Name of the contrastive pair (e.g. ``"sensor-satellite"``).
        lr: Learning rate for projection heads and temperature.
        weight_decay: AdamW weight decay.
        batch_size: Training batch size.
        epochs: Number of training epochs.
        warmup_epochs: Number of linear warmup epochs.
        encoder_dim: Encoder output dimension.
        contrastive_dim: Projection head output dimension.
        init_temperature: Initial temperature for InfoNCE loss.
        hard_negative_mining: Enable hard negative mining.
        hard_negative_fraction: Fraction of hard negatives.
        label_smoothing: Label smoothing for cross-entropy.
        symmetric: Use symmetric loss.
        max_grad_norm: Gradient clipping norm.
        eval_every_n_epochs: Evaluate retrieval metrics every N epochs.
        output_dir: Directory for checkpoints and logs.
        device: Training device.
        seed: Random seed.
    """

    pair_name: str = "sensor-satellite"
    lr: float = 3e-4
    weight_decay: float = 0.01
    batch_size: int = 256
    epochs: int = 100
    warmup_epochs: int = 10
    encoder_dim: int = ENCODER_EMBED_DIM
    contrastive_dim: int = CONTRASTIVE_DIM
    init_temperature: float = 0.07
    hard_negative_mining: bool = False
    hard_negative_fraction: float = 0.2
    label_smoothing: float = 0.0
    symmetric: bool = True
    max_grad_norm: float = 1.0
    eval_every_n_epochs: int = 5
    output_dir: str = "outputs/contrastive"
    device: str = "auto"
    seed: int = 42


# ---------------------------------------------------------------------------
# train_contrastive_epoch
# ---------------------------------------------------------------------------


def train_contrastive_epoch(
    pretrainer: ContrastivePretrainer,
    dataloader: DataLoader,
    optimizer: Optimizer,
    scheduler: Optional[LambdaLR] = None,
    epoch: int = 0,
    max_grad_norm: float = 1.0,
    device: torch.device = torch.device("cpu"),
    log_every_n_steps: int = 50,
) -> Dict[str, float]:
    """Run one epoch of contrastive pretraining.

    Iterates over the dataloader, computing the CLIP-style contrastive
    loss for each batch and updating the projection heads and
    temperature parameter.  Encoder weights remain frozen.

    Args:
        pretrainer: A configured :class:`ContrastivePretrainer` with
            frozen encoders and learnable projection heads.
        dataloader: DataLoader yielding ``(emb_a, emb_b)`` pairs,
            each of shape ``[B, D]``.
        optimizer: Optimizer over ``pretrainer.trainable_parameters()``.
        scheduler: Optional LR scheduler, stepped per batch.
        epoch: Current epoch number (for logging).
        max_grad_norm: Maximum gradient norm for clipping.
        device: Training device.
        log_every_n_steps: Console log frequency.

    Returns:
        Dict of epoch-averaged metrics: ``loss``, ``loss_a2b``,
        ``loss_b2a``, ``accuracy_a2b``, ``accuracy_b2a``,
        ``temperature``, ``lr``.
    """
    pretrainer.proj_a.train()
    pretrainer.proj_b.train()
    # Ensure encoders stay in eval mode.
    pretrainer.encoder_a.eval()
    pretrainer.encoder_b.eval()

    epoch_metrics: Dict[str, List[float]] = {
        "loss": [],
        "loss_a2b": [],
        "loss_b2a": [],
        "accuracy_a2b": [],
        "accuracy_b2a": [],
        "temperature": [],
    }

    progress = make_progress()
    progress.start()
    task = progress.add_task(
        f"Contrastive Epoch {epoch}", total=len(dataloader)
    )

    for step, batch in enumerate(dataloader):
        # Unpack batch: (emb_a, emb_b) or (emb_a, emb_b, weights).
        if isinstance(batch, (list, tuple)):
            if len(batch) == 3:
                emb_a, emb_b, weights = batch
                weights = weights.to(device)
            else:
                emb_a, emb_b = batch[:2]
                weights = None
        else:
            raise TypeError(
                f"Expected batch as tuple (emb_a, emb_b), got {type(batch)}"
            )

        emb_a = emb_a.to(device)
        emb_b = emb_b.to(device)

        # Forward through projection heads + contrastive loss.
        # Since we pass pre-computed embeddings, the encoder forward
        # is skipped (handled by ContrastivePretrainer._encode).
        result = pretrainer(emb_a, emb_b, weights=weights)

        loss = result["loss"]

        # Backward pass.
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient clipping over trainable parameters only.
        if max_grad_norm > 0:
            nn.utils.clip_grad_norm_(
                pretrainer.trainable_parameters(), max_grad_norm
            )

        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # Track metrics.
        for key in epoch_metrics:
            val = result.get(key, torch.tensor(0.0))
            epoch_metrics[key].append(
                val.item() if isinstance(val, torch.Tensor) else val
            )

        # Step logging.
        if (step + 1) % log_every_n_steps == 0:
            avg_loss = sum(epoch_metrics["loss"][-log_every_n_steps:]) / min(
                log_every_n_steps, len(epoch_metrics["loss"])
            )
            acc_a2b = sum(
                epoch_metrics["accuracy_a2b"][-log_every_n_steps:]
            ) / min(log_every_n_steps, len(epoch_metrics["accuracy_a2b"]))
            logger.info(
                f"Epoch {epoch} | Step {step + 1}/{len(dataloader)} | "
                f"loss={avg_loss:.4f} | acc_a2b={acc_a2b:.3f} | "
                f"temp={pretrainer.criterion.temperature:.4f} | "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )

        progress.update(
            task,
            advance=1,
            description=(
                f"Contrastive Epoch {epoch} "
                f"[loss={result['loss'].item():.4f} "
                f"acc={result['accuracy_a2b'].item():.3f}]"
            ),
        )

    progress.stop()

    # Compute epoch averages.
    avg_metrics: Dict[str, float] = {}
    for key, values in epoch_metrics.items():
        avg_metrics[key] = sum(values) / max(len(values), 1)

    avg_metrics["lr"] = optimizer.param_groups[0]["lr"]

    logger.info(
        f"Epoch {epoch} complete | "
        f"loss={avg_metrics['loss']:.4f} | "
        f"acc_a2b={avg_metrics['accuracy_a2b']:.3f} | "
        f"acc_b2a={avg_metrics['accuracy_b2a']:.3f} | "
        f"temp={avg_metrics['temperature']:.4f}"
    )

    return avg_metrics


# ---------------------------------------------------------------------------
# Full training loop
# ---------------------------------------------------------------------------


def run_contrastive_pretraining(
    pretrainer: ContrastivePretrainer,
    train_loader: DataLoader,
    val_emb_a: Optional[torch.Tensor] = None,
    val_emb_b: Optional[torch.Tensor] = None,
    config: Optional[ContrastivePretrainConfig] = None,
) -> Dict[str, Any]:
    """Run full contrastive pretraining with evaluation and checkpointing.

    Convenience function that wraps :func:`train_contrastive_epoch`
    with optimizer construction, LR scheduling, periodic evaluation,
    and checkpoint saving.

    Args:
        pretrainer: Configured :class:`ContrastivePretrainer`.
        train_loader: DataLoader of ``(emb_a, emb_b)`` training pairs.
        val_emb_a: Optional validation embeddings for modality A ``[N, D]``.
        val_emb_b: Optional validation embeddings for modality B ``[N, D]``.
        config: Training configuration.  Uses defaults if None.

    Returns:
        Dict with training history and best metrics.
    """
    from pathlib import Path

    if config is None:
        config = ContrastivePretrainConfig()

    # Resolve device.
    if config.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.device)

    pretrainer.to(device)

    # Seed.
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Optimizer: only trainable parameters (proj heads + temperature).
    optimizer = torch.optim.AdamW(
        pretrainer.trainable_parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    # Cosine schedule with linear warmup.
    total_steps = len(train_loader) * config.epochs
    warmup_steps = len(train_loader) * config.warmup_epochs

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.01 + 0.5 * 0.99 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)

    # Output directory.
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Evaluator.
    evaluator = ContrastiveEvaluator(
        proj_a=pretrainer.proj_a,
        proj_b=pretrainer.proj_b,
        device=device,
    )

    # Training history.
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc_a2b": [],
        "train_acc_b2a": [],
        "temperature": [],
    }
    best_loss = float("inf")
    best_epoch = -1

    for epoch in range(config.epochs):
        # Train one epoch.
        epoch_metrics = train_contrastive_epoch(
            pretrainer=pretrainer,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            max_grad_norm=config.max_grad_norm,
            device=device,
        )

        history["train_loss"].append(epoch_metrics["loss"])
        history["train_acc_a2b"].append(epoch_metrics["accuracy_a2b"])
        history["train_acc_b2a"].append(epoch_metrics["accuracy_b2a"])
        history["temperature"].append(epoch_metrics["temperature"])

        # Periodic evaluation.
        if (
            val_emb_a is not None
            and val_emb_b is not None
            and (epoch + 1) % config.eval_every_n_epochs == 0
        ):
            val_metrics = evaluator.zero_shot_retrieval(val_emb_a, val_emb_b)
            logger.info(
                f"Eval Epoch {epoch} | "
                + " | ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
            )

            for k, v in val_metrics.items():
                history.setdefault(f"val_{k}", []).append(v)

        # Checkpointing.
        if epoch_metrics["loss"] < best_loss:
            best_loss = epoch_metrics["loss"]
            best_epoch = epoch
            ckpt_path = output_dir / "best_contrastive.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "proj_a_state": pretrainer.proj_a.state_dict(),
                    "proj_b_state": pretrainer.proj_b.state_dict(),
                    "criterion_state": pretrainer.criterion.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "config": config.__dict__,
                    "modality_a": pretrainer.modality_a,
                    "modality_b": pretrainer.modality_b,
                    "loss": best_loss,
                },
                ckpt_path,
            )
            logger.info(f"New best model at epoch {epoch} (loss={best_loss:.4f})")

    # Final summary.
    summary = {
        "best_loss": best_loss,
        "best_epoch": best_epoch,
        "total_epochs": config.epochs,
        "pair": config.pair_name,
        "history": history,
    }

    import json

    summary_path = output_dir / "contrastive_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Training complete. Summary: {summary_path}")

    return summary
