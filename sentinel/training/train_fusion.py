"""
Perceiver IO fusion layer training pipeline for SENTINEL.

Staged training to solve the cold-start problem:

Stage 1: Freeze encoders, train fusion layer + output heads on co-located
          multi-modal data. Curriculum: modality pairs -> triplets -> full 5-modal.
Stage 2: End-to-end fine-tuning with unfrozen top encoder layers, lower lr,
          and cross-modal consistency loss.

Usage:
    python -m sentinel.training.train_fusion --stage 1 --data-dir data/aligned --encoder-dir outputs/pretrained_encoders
    python -m sentinel.training.train_fusion --stage 2 --data-dir data/aligned --checkpoint outputs/fusion/stage1/checkpoints/best_model.pt
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from sentinel.models.fusion.embedding_registry import (
    MODALITY_IDS,
    NUM_MODALITIES,
    SHARED_EMBEDDING_DIM,
)
from sentinel.models.fusion.heads import (
    ANOMALY_TYPES,
    ALERT_LEVELS,
    CONTAMINANT_CLASSES,
    NUM_ANOMALY_TYPES,
    NUM_ALERT_LEVELS,
    NUM_CONTAMINANT_CLASSES,
)
from sentinel.models.fusion.model import FusionOutput, PerceiverIOFusion
from sentinel.models.fusion.projections import NATIVE_DIMS
from sentinel.training.trainer import (
    BaseTrainer,
    EarlyStopping,
    MetricTracker,
    TrainerConfig,
    build_scheduler,
)
from sentinel.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Alignment Index Dataset
# ---------------------------------------------------------------------------


class AlignmentIndexDataset(Dataset):
    """Dataset for co-located multi-modal observations.

    Loads from an alignment index (JSON or directory structure) that maps
    site-time windows to available modality embeddings.

    Each sample is a dict with:
      - modality embeddings (pre-computed from frozen encoders)
      - timestamps per modality
      - confidence per modality
      - labels (anomaly, source class, alert level)
    """

    def __init__(
        self,
        data_dir: Path,
        min_modalities: int = 2,
        modality_filter: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.min_modalities = min_modalities
        self.modality_filter = modality_filter

        # Load alignment index
        index_path = self.data_dir / "alignment_index.json"
        if index_path.exists():
            with open(index_path, "r", encoding="utf-8") as f:
                self._index = json.load(f)
        else:
            # Build from directory structure: site_id/timestamp/*.pt
            self._index = self._build_index_from_dir()

        # Filter by minimum modalities
        self._samples = [
            entry for entry in self._index
            if self._count_modalities(entry) >= min_modalities
        ]

        logger.info(
            f"AlignmentIndexDataset: {len(self._samples)} samples with "
            f">={min_modalities} modalities from {self.data_dir}"
        )

    def _build_index_from_dir(self) -> List[Dict[str, Any]]:
        """Build alignment index from directory of .pt embedding files."""
        index: List[Dict[str, Any]] = []
        for site_dir in sorted(self.data_dir.iterdir()):
            if not site_dir.is_dir():
                continue
            for window_dir in sorted(site_dir.iterdir()):
                if not window_dir.is_dir():
                    continue
                entry: Dict[str, Any] = {
                    "site_id": site_dir.name,
                    "window": window_dir.name,
                    "modalities": {},
                    "labels": {},
                }
                for pt_file in window_dir.glob("*.pt"):
                    mod_name = pt_file.stem
                    if mod_name in MODALITY_IDS:
                        entry["modalities"][mod_name] = str(pt_file)

                # Load labels if present
                label_path = window_dir / "labels.json"
                if label_path.exists():
                    with open(label_path, "r", encoding="utf-8") as f:
                        entry["labels"] = json.load(f)

                if entry["modalities"]:
                    index.append(entry)
        return index

    def _count_modalities(self, entry: Dict[str, Any]) -> int:
        mods = entry.get("modalities", {})
        if self.modality_filter:
            return sum(1 for m in self.modality_filter if m in mods)
        return len(mods)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        entry = self._samples[idx]
        sample: Dict[str, Any] = {
            "site_id": entry.get("site_id", ""),
            "embeddings": {},
            "timestamps": {},
            "confidences": {},
            "modalities_present": [],
            "labels": entry.get("labels", {}),
        }

        for mod_id, path in entry["modalities"].items():
            if self.modality_filter and mod_id not in self.modality_filter:
                continue
            try:
                emb = torch.load(path, map_location="cpu", weights_only=True)
                if isinstance(emb, dict):
                    sample["embeddings"][mod_id] = emb.get("embedding", emb.get("data"))
                    sample["timestamps"][mod_id] = emb.get("timestamp", 0.0)
                    sample["confidences"][mod_id] = emb.get("confidence", 1.0)
                else:
                    sample["embeddings"][mod_id] = emb
                    sample["timestamps"][mod_id] = 0.0
                    sample["confidences"][mod_id] = 1.0
                sample["modalities_present"].append(mod_id)
            except Exception as e:
                logger.debug(f"Failed to load {path}: {e}")

        return sample


def fusion_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate multi-modal samples into a batch.

    Since fusion processes modalities sequentially (one observation event
    at a time), we structure the batch for sequential processing.
    """
    B = len(batch)

    # Collect all modalities present in any sample
    all_mods: set = set()
    for s in batch:
        all_mods.update(s["modalities_present"])

    # For each modality, stack embeddings (with masking for absent modalities)
    collated: Dict[str, Any] = {
        "batch_size": B,
        "modalities_present": [s["modalities_present"] for s in batch],
        "embeddings": {},
        "timestamps": {},
        "confidences": {},
        "labels": {},
    }

    for mod in sorted(all_mods):
        embs = []
        timestamps = []
        confidences = []
        mask = []
        for s in batch:
            if mod in s["embeddings"]:
                embs.append(s["embeddings"][mod])
                timestamps.append(s["timestamps"].get(mod, 0.0))
                confidences.append(s["confidences"].get(mod, 1.0))
                mask.append(True)
            else:
                # Placeholder zero embedding
                dim = NATIVE_DIMS.get(mod, SHARED_EMBEDDING_DIM)
                embs.append(torch.zeros(dim))
                timestamps.append(0.0)
                confidences.append(0.0)
                mask.append(False)

        collated["embeddings"][mod] = torch.stack(embs)
        collated["timestamps"][mod] = timestamps
        collated["confidences"][mod] = confidences
        collated[f"{mod}_mask"] = mask

    # Collate labels
    label_keys = set()
    for s in batch:
        label_keys.update(s.get("labels", {}).keys())

    for key in label_keys:
        vals = [s.get("labels", {}).get(key, float("nan")) for s in batch]
        if all(isinstance(v, (int, float)) for v in vals):
            collated["labels"][key] = torch.tensor(vals, dtype=torch.float32)

    return collated


# ---------------------------------------------------------------------------
# Curriculum for Stage 1
# ---------------------------------------------------------------------------

class ModalityCurriculum:
    """Curriculum scheduler that controls how many modalities are used.

    Progression: pairs -> triplets -> full 5-modal.
    """

    def __init__(
        self,
        total_epochs: int,
        pair_fraction: float = 0.3,
        triplet_fraction: float = 0.3,
    ) -> None:
        self.total_epochs = total_epochs
        self.pair_cutoff = int(total_epochs * pair_fraction)
        self.triplet_cutoff = int(total_epochs * (pair_fraction + triplet_fraction))

    def get_max_modalities(self, epoch: int) -> int:
        """Return the maximum number of modalities to use this epoch."""
        if epoch < self.pair_cutoff:
            return 2
        elif epoch < self.triplet_cutoff:
            return 3
        else:
            return NUM_MODALITIES

    def get_phase_name(self, epoch: int) -> str:
        if epoch < self.pair_cutoff:
            return "pairs"
        elif epoch < self.triplet_cutoff:
            return "triplets"
        return "full"


# ---------------------------------------------------------------------------
# Fusion training model wrapper
# ---------------------------------------------------------------------------


class FusionWithHeads(nn.Module):
    """PerceiverIOFusion with output heads for supervised training."""

    def __init__(
        self,
        fusion: PerceiverIOFusion,
        shared_dim: int = SHARED_EMBEDDING_DIM,
    ) -> None:
        super().__init__()
        self.fusion = fusion

        # Anomaly detection head
        self.anomaly_head = nn.Sequential(
            nn.Linear(shared_dim, shared_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(shared_dim // 2, 1),
        )

        # Alert level head (ordinal: no_event, low, high)
        self.alert_head = nn.Sequential(
            nn.Linear(shared_dim, shared_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(shared_dim // 2, NUM_ALERT_LEVELS),
        )

        # Source attribution head
        self.source_head = nn.Sequential(
            nn.Linear(shared_dim, shared_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(shared_dim // 2, NUM_CONTAMINANT_CLASSES),
        )

    def forward(
        self,
        modality_embeddings: Dict[str, torch.Tensor],
        timestamps: Dict[str, List[float]],
        confidences: Dict[str, List[float]],
        modality_order: List[str],
    ) -> Dict[str, torch.Tensor]:
        """Process modalities sequentially through fusion and apply heads.

        Args:
            modality_embeddings: {mod_id: (B, native_dim)}.
            timestamps: {mod_id: list of floats}.
            confidences: {mod_id: list of floats}.
            modality_order: Order in which to feed modalities.

        Returns:
            Dict with anomaly_logit, alert_logits, source_logits, fused_state.
        """
        latent_state = None
        fused_state = None

        for mod_id in modality_order:
            if mod_id not in modality_embeddings:
                continue

            emb = modality_embeddings[mod_id]
            ts = timestamps.get(mod_id, [0.0] * emb.shape[0])
            conf = confidences.get(mod_id, [1.0] * emb.shape[0])

            # Process each sample (fusion expects per-observation calls)
            B = emb.shape[0]
            batch_fused = []
            batch_latents = []

            for b in range(B):
                out = self.fusion(
                    modality_id=mod_id,
                    raw_embedding=emb[b],
                    timestamp=float(ts[b]) if isinstance(ts, list) else float(ts),
                    confidence=float(conf[b]) if isinstance(conf, list) else float(conf),
                    latent_state=latent_state[b:b+1] if latent_state is not None else None,
                )
                batch_fused.append(out.fused_state)
                batch_latents.append(out.latent_state)

            fused_state = torch.cat(batch_fused, dim=0)  # (B, 256)
            latent_state = torch.cat(batch_latents, dim=0)  # (B, N, 256)

        if fused_state is None:
            raise ValueError("No modalities processed")

        # Apply output heads
        anomaly_logit = self.anomaly_head(fused_state).squeeze(-1)  # (B,)
        alert_logits = self.alert_head(fused_state)  # (B, 3)
        source_logits = self.source_head(fused_state)  # (B, C)

        return {
            "anomaly_logit": anomaly_logit,
            "alert_logits": alert_logits,
            "source_logits": source_logits,
            "fused_state": fused_state,
        }


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class FusionStage1Config(TrainerConfig):
    """Stage 1: Freeze encoders, train fusion + heads."""

    lr: float = 1e-4
    batch_size: int = 32
    epochs: int = 50
    scheduler: str = "cosine"
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    wandb_run_name: str = "fusion-stage1-freeze-encoders"

    # Data
    data_dir: str = "data/aligned"
    min_modalities: int = 2

    # Curriculum fractions
    pair_fraction: float = 0.3
    triplet_fraction: float = 0.3

    # Fusion architecture
    num_latents: int = 256
    num_heads: int = 8
    num_process_layers: int = 2
    dropout: float = 0.1


@dataclass
class FusionStage2Config(TrainerConfig):
    """Stage 2: End-to-end fine-tuning with cross-modal consistency."""

    lr: float = 1e-5
    batch_size: int = 16
    epochs: int = 30
    scheduler: str = "cosine"
    weight_decay: float = 0.01
    warmup_steps: int = 500
    wandb_run_name: str = "fusion-stage2-e2e-finetune"

    # Data
    data_dir: str = "data/aligned"
    min_modalities: int = 3

    # Fine-tuning
    unfreeze_top_n_layers: int = 4
    consistency_loss_weight: float = 0.1

    # Architecture
    num_latents: int = 256
    num_heads: int = 8
    num_process_layers: int = 2
    dropout: float = 0.1

    # Stage 1 checkpoint
    stage1_checkpoint: str = ""


# ---------------------------------------------------------------------------
# Stage 1 Trainer
# ---------------------------------------------------------------------------


class FusionStage1Trainer(BaseTrainer):
    """Stage 1: Freeze encoders, train Perceiver IO fusion + output heads."""

    def __init__(self, config: FusionStage1Config) -> None:
        super().__init__(config)
        self.stage_config = config
        self.curriculum = ModalityCurriculum(
            total_epochs=config.epochs,
            pair_fraction=config.pair_fraction,
            triplet_fraction=config.triplet_fraction,
        )

    def build_model(self) -> nn.Module:
        fusion = PerceiverIOFusion(
            shared_dim=SHARED_EMBEDDING_DIM,
            num_latents=self.stage_config.num_latents,
            num_heads=self.stage_config.num_heads,
            num_process_layers=self.stage_config.num_process_layers,
            dropout=self.stage_config.dropout,
        )
        model = FusionWithHeads(fusion)
        return model

    def build_datasets(self) -> Tuple[Any, Any]:
        data_dir = Path(self.stage_config.data_dir)
        train_ds = AlignmentIndexDataset(
            data_dir=data_dir / "train",
            min_modalities=self.stage_config.min_modalities,
        )
        val_ds = AlignmentIndexDataset(
            data_dir=data_dir / "val",
            min_modalities=self.stage_config.min_modalities,
        )
        return train_ds, val_ds

    def _build_dataloader(self, dataset: Any, shuffle: bool = True) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            collate_fn=fusion_collate_fn,
            drop_last=True,
        )

    def _select_modalities(
        self,
        available: List[str],
        max_mods: int,
    ) -> List[str]:
        """Select up to max_mods modalities from available ones."""
        if len(available) <= max_mods:
            return available
        return sorted(random.sample(available, max_mods))

    def train_step(self, batch: Any) -> Dict[str, float]:
        assert self.model is not None and self.optimizer is not None

        max_mods = self.curriculum.get_max_modalities(self.current_epoch)
        B = batch["batch_size"]

        # Determine which modalities to use per the curriculum
        all_available = list(batch["embeddings"].keys())
        selected = self._select_modalities(all_available, max_mods)

        embeddings = {m: batch["embeddings"][m].to(self.device) for m in selected}
        timestamps = {m: batch["timestamps"][m] for m in selected}
        confidences = {m: batch["confidences"][m] for m in selected}

        output = self.model(
            modality_embeddings=embeddings,
            timestamps=timestamps,
            confidences=confidences,
            modality_order=selected,
        )

        losses: Dict[str, torch.Tensor] = {}
        total = torch.tensor(0.0, device=self.device)

        labels = batch.get("labels", {})

        # Anomaly detection loss
        if "anomaly" in labels:
            anomaly_labels = labels["anomaly"].to(self.device)
            mask = ~torch.isnan(anomaly_labels)
            if mask.any():
                anom_loss = F.binary_cross_entropy_with_logits(
                    output["anomaly_logit"][mask], anomaly_labels[mask],
                )
                losses["anomaly_loss"] = anom_loss
                total = total + anom_loss

        # Alert level loss
        if "alert_level" in labels:
            alert_labels = labels["alert_level"].to(self.device).long()
            mask = alert_labels >= 0
            if mask.any():
                alert_loss = F.cross_entropy(
                    output["alert_logits"][mask], alert_labels[mask],
                )
                losses["alert_loss"] = alert_loss
                total = total + alert_loss

        # Source attribution loss
        if "source_class" in labels:
            source_labels = labels["source_class"].to(self.device).long()
            mask = source_labels >= 0
            if mask.any():
                source_loss = F.cross_entropy(
                    output["source_logits"][mask], source_labels[mask],
                )
                losses["source_loss"] = source_loss
                total = total + source_loss

        # Fallback: if no labels, use reconstruction-style loss
        if not losses:
            total = output["fused_state"].pow(2).mean() * 0.01
            losses["regularization"] = total

        losses["loss"] = total

        total.backward()
        if self.config.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        return {k: v.detach() for k, v in losses.items()}

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        assert self.model is not None
        self.model.eval()
        all_losses: Dict[str, List[float]] = {}

        for batch in dataloader:
            all_available = list(batch["embeddings"].keys())
            embeddings = {m: batch["embeddings"][m].to(self.device) for m in all_available}
            timestamps = {m: batch["timestamps"][m] for m in all_available}
            confidences = {m: batch["confidences"][m] for m in all_available}

            output = self.model(
                modality_embeddings=embeddings,
                timestamps=timestamps,
                confidences=confidences,
                modality_order=all_available,
            )

            labels = batch.get("labels", {})
            total = torch.tensor(0.0, device=self.device)

            if "anomaly" in labels:
                anomaly_labels = labels["anomaly"].to(self.device)
                mask = ~torch.isnan(anomaly_labels)
                if mask.any():
                    loss = F.binary_cross_entropy_with_logits(
                        output["anomaly_logit"][mask], anomaly_labels[mask],
                    )
                    total = total + loss
                    all_losses.setdefault("anomaly_loss", []).append(loss.item())

            if "alert_level" in labels:
                alert_labels = labels["alert_level"].to(self.device).long()
                mask = alert_labels >= 0
                if mask.any():
                    loss = F.cross_entropy(output["alert_logits"][mask], alert_labels[mask])
                    total = total + loss
                    all_losses.setdefault("alert_loss", []).append(loss.item())

            if "source_class" in labels:
                source_labels = labels["source_class"].to(self.device).long()
                mask = source_labels >= 0
                if mask.any():
                    loss = F.cross_entropy(output["source_logits"][mask], source_labels[mask])
                    total = total + loss
                    all_losses.setdefault("source_loss", []).append(loss.item())

            all_losses.setdefault("loss", []).append(total.item())

        self.model.train()
        return {k: sum(v) / len(v) for k, v in all_losses.items() if v}


# ---------------------------------------------------------------------------
# Stage 2 Trainer
# ---------------------------------------------------------------------------


class FusionStage2Trainer(BaseTrainer):
    """Stage 2: End-to-end fine-tuning with cross-modal consistency loss."""

    def __init__(self, config: FusionStage2Config) -> None:
        super().__init__(config)
        self.stage_config = config

    def build_model(self) -> nn.Module:
        fusion = PerceiverIOFusion(
            shared_dim=SHARED_EMBEDDING_DIM,
            num_latents=self.stage_config.num_latents,
            num_heads=self.stage_config.num_heads,
            num_process_layers=self.stage_config.num_process_layers,
            dropout=self.stage_config.dropout,
        )
        model = FusionWithHeads(fusion)

        # Load Stage 1 checkpoint
        if self.stage_config.stage1_checkpoint:
            ckpt_path = Path(self.stage_config.stage1_checkpoint)
            if ckpt_path.exists():
                state = torch.load(str(ckpt_path), map_location=self.device, weights_only=False)
                model.load_state_dict(state["model_state_dict"], strict=False)
                logger.info(f"Loaded Stage 1 checkpoint from {ckpt_path}")
            else:
                logger.warning(f"Stage 1 checkpoint not found: {ckpt_path}")

        return model

    def build_datasets(self) -> Tuple[Any, Any]:
        data_dir = Path(self.stage_config.data_dir)
        train_ds = AlignmentIndexDataset(
            data_dir=data_dir / "train",
            min_modalities=self.stage_config.min_modalities,
        )
        val_ds = AlignmentIndexDataset(
            data_dir=data_dir / "val",
            min_modalities=self.stage_config.min_modalities,
        )
        return train_ds, val_ds

    def _build_dataloader(self, dataset: Any, shuffle: bool = True) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            collate_fn=fusion_collate_fn,
            drop_last=True,
        )

    def _compute_consistency_loss(
        self,
        embeddings: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Cross-modal consistency loss.

        Encourages modality embeddings projected to the shared space to
        be similar for observations of the same event.  Computed as the
        mean pairwise cosine distance between projected embeddings.
        """
        projected = []
        for mod_id, emb in embeddings.items():
            # Project through the fusion projection bank
            proj = self.model.fusion.projection_bank(mod_id, emb)
            projected.append(F.normalize(proj, dim=-1))

        if len(projected) < 2:
            return torch.tensor(0.0, device=self.device)

        # Pairwise cosine similarity -> maximize (minimize negative)
        total_dist = torch.tensor(0.0, device=self.device)
        n_pairs = 0
        for i in range(len(projected)):
            for j in range(i + 1, len(projected)):
                cos_sim = (projected[i] * projected[j]).sum(dim=-1).mean()
                total_dist = total_dist + (1.0 - cos_sim)
                n_pairs += 1

        return total_dist / max(n_pairs, 1)

    def train_step(self, batch: Any) -> Dict[str, float]:
        assert self.model is not None and self.optimizer is not None

        all_available = list(batch["embeddings"].keys())
        embeddings = {m: batch["embeddings"][m].to(self.device) for m in all_available}
        timestamps = {m: batch["timestamps"][m] for m in all_available}
        confidences = {m: batch["confidences"][m] for m in all_available}

        output = self.model(
            modality_embeddings=embeddings,
            timestamps=timestamps,
            confidences=confidences,
            modality_order=all_available,
        )

        losses: Dict[str, torch.Tensor] = {}
        total = torch.tensor(0.0, device=self.device)

        labels = batch.get("labels", {})

        # Task losses (same as Stage 1)
        if "anomaly" in labels:
            anomaly_labels = labels["anomaly"].to(self.device)
            mask = ~torch.isnan(anomaly_labels)
            if mask.any():
                anom_loss = F.binary_cross_entropy_with_logits(
                    output["anomaly_logit"][mask], anomaly_labels[mask],
                )
                losses["anomaly_loss"] = anom_loss
                total = total + anom_loss

        if "alert_level" in labels:
            alert_labels = labels["alert_level"].to(self.device).long()
            mask = alert_labels >= 0
            if mask.any():
                alert_loss = F.cross_entropy(
                    output["alert_logits"][mask], alert_labels[mask],
                )
                losses["alert_loss"] = alert_loss
                total = total + alert_loss

        if "source_class" in labels:
            source_labels = labels["source_class"].to(self.device).long()
            mask = source_labels >= 0
            if mask.any():
                source_loss = F.cross_entropy(
                    output["source_logits"][mask], source_labels[mask],
                )
                losses["source_loss"] = source_loss
                total = total + source_loss

        # Cross-modal consistency loss
        consistency = self._compute_consistency_loss(embeddings)
        losses["consistency_loss"] = consistency
        total = total + self.stage_config.consistency_loss_weight * consistency

        losses["loss"] = total

        total.backward()
        if self.config.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        return {k: v.detach() for k, v in losses.items()}

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        assert self.model is not None
        self.model.eval()
        all_losses: Dict[str, List[float]] = {}

        for batch in dataloader:
            all_available = list(batch["embeddings"].keys())
            embeddings = {m: batch["embeddings"][m].to(self.device) for m in all_available}
            timestamps = {m: batch["timestamps"][m] for m in all_available}
            confidences = {m: batch["confidences"][m] for m in all_available}

            output = self.model(
                modality_embeddings=embeddings,
                timestamps=timestamps,
                confidences=confidences,
                modality_order=all_available,
            )

            labels = batch.get("labels", {})
            total = torch.tensor(0.0, device=self.device)

            if "anomaly" in labels:
                anomaly_labels = labels["anomaly"].to(self.device)
                mask = ~torch.isnan(anomaly_labels)
                if mask.any():
                    loss = F.binary_cross_entropy_with_logits(
                        output["anomaly_logit"][mask], anomaly_labels[mask],
                    )
                    total = total + loss
                    all_losses.setdefault("anomaly_loss", []).append(loss.item())

            if "alert_level" in labels:
                alert_labels = labels["alert_level"].to(self.device).long()
                mask = alert_labels >= 0
                if mask.any():
                    loss = F.cross_entropy(output["alert_logits"][mask], alert_labels[mask])
                    total = total + loss
                    all_losses.setdefault("alert_loss", []).append(loss.item())

            if "source_class" in labels:
                source_labels = labels["source_class"].to(self.device).long()
                mask = source_labels >= 0
                if mask.any():
                    loss = F.cross_entropy(output["source_logits"][mask], source_labels[mask])
                    total = total + loss
                    all_losses.setdefault("source_loss", []).append(loss.item())

            # Consistency
            consistency = self._compute_consistency_loss(embeddings)
            all_losses.setdefault("consistency_loss", []).append(consistency.item())
            total = total + self.stage_config.consistency_loss_weight * consistency

            all_losses.setdefault("loss", []).append(total.item())

        self.model.train()
        return {k: sum(v) / len(v) for k, v in all_losses.items() if v}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SENTINEL Perceiver IO fusion training pipeline",
    )
    parser.add_argument(
        "--stage", type=int, required=True, choices=[1, 2],
        help="Training stage: 1=freeze encoders, 2=end-to-end fine-tuning",
    )
    parser.add_argument("--data-dir", type=str, default="data/aligned")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="",
                        help="Stage 1 checkpoint for Stage 2")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--min-modalities", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="sentinel")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume training from checkpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = args.output_dir or f"outputs/fusion/stage{args.stage}"

    if args.stage == 1:
        config = FusionStage1Config(
            data_dir=args.data_dir,
            output_dir=output_dir,
            device=args.device,
            seed=args.seed,
            use_wandb=not args.no_wandb,
            wandb_project=args.wandb_project,
            num_workers=args.num_workers,
            fp16=args.fp16,
        )
        if args.epochs is not None:
            config.epochs = args.epochs
        if args.batch_size is not None:
            config.batch_size = args.batch_size
        if args.lr is not None:
            config.lr = args.lr
        if args.min_modalities is not None:
            config.min_modalities = args.min_modalities

        trainer = FusionStage1Trainer(config)

    elif args.stage == 2:
        config = FusionStage2Config(
            data_dir=args.data_dir,
            output_dir=output_dir,
            stage1_checkpoint=args.checkpoint,
            device=args.device,
            seed=args.seed,
            use_wandb=not args.no_wandb,
            wandb_project=args.wandb_project,
            num_workers=args.num_workers,
            fp16=args.fp16,
        )
        if args.epochs is not None:
            config.epochs = args.epochs
        if args.batch_size is not None:
            config.batch_size = args.batch_size
        if args.lr is not None:
            config.lr = args.lr
        if args.min_modalities is not None:
            config.min_modalities = args.min_modalities

        trainer = FusionStage2Trainer(config)
    else:
        raise ValueError(f"Unknown stage: {args.stage}")

    logger.info(f"Starting Fusion Stage {args.stage} training")
    trainer.setup()

    if args.resume:
        trainer.load_checkpoint(args.resume)

    summary = trainer.train()
    logger.info(f"Stage {args.stage} complete: {json.dumps(summary, indent=2, default=str)}")


if __name__ == "__main__":
    main()
