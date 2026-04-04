"""
BioMotion encoder training pipeline — three-phase training for SENTINEL.

Phase 1: Diffusion pretraining — self-supervised denoising on normal trajectories.
Phase 2: Anomaly fine-tuning — supervised binary classification (normal vs anomalous).
Phase 3: Multi-organism ensemble — cross-organism attention with mixed-species batches.

Usage:
    python -m sentinel.training.train_biomotion --phase 1 --data-dir data/behavioral
    python -m sentinel.training.train_biomotion --phase 2 --data-dir data/behavioral --checkpoint outputs/biomotion/phase1/checkpoints/best_model.pt
    python -m sentinel.training.train_biomotion --phase 3 --data-dir data/behavioral --checkpoint outputs/biomotion/phase2/checkpoints/best_model.pt
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from sentinel.data.behavioral.trajectory import (
    TrajectoryDataset,
    TrajectoryRecord,
    corrupt_batch,
    split_trajectories,
)
from sentinel.models.biomotion.model import BioMotionEncoder, SHARED_EMBED_DIM
from sentinel.models.biomotion.multi_organism import (
    CrossOrganismAttention,
    MultiOrganismEnsemble,
    SPECIES_FEATURE_DIM,
    SPECIES_ORDER,
)
from sentinel.models.biomotion.trajectory_encoder import (
    EMBED_DIM,
    NUM_DIFFUSION_STEPS,
    TrajectoryDiffusionEncoder,
)
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
# Phase 1: Diffusion Pretraining
# ---------------------------------------------------------------------------


@dataclass
class DiffusionPretrainConfig(TrainerConfig):
    """Configuration for Phase 1 diffusion pretraining."""

    lr: float = 2e-4
    batch_size: int = 64
    epochs: int = 200
    warmup_steps: int = 2000
    scheduler: str = "cosine"
    weight_decay: float = 0.01
    noise_schedule: str = "cosine"
    max_trajectory_length: int = 1800
    wandb_run_name: str = "biomotion-diffusion-pretrain"

    # Data
    data_dir: str = "data/behavioral"
    species: List[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.species is None:
            self.species = list(SPECIES_ORDER)


class DiffusionPretrainer(BaseTrainer):
    """Phase 1: Self-supervised diffusion pretraining on normal trajectories.

    Corrupts trajectories with cosine noise schedule, trains denoiser to
    predict the added noise (epsilon prediction).
    """

    def __init__(self, config: DiffusionPretrainConfig) -> None:
        super().__init__(config)
        self.phase_config = config
        self.encoder: Optional[TrajectoryDiffusionEncoder] = None

    def build_model(self) -> nn.Module:
        species = self.phase_config.species[0] if self.phase_config.species else "daphnia"
        feature_dim = SPECIES_FEATURE_DIM.get(species, 32)
        self.encoder = TrajectoryDiffusionEncoder(
            feature_dim=feature_dim,
            embed_dim=EMBED_DIM,
        )
        return self.encoder

    def build_datasets(self) -> Tuple[Any, Any]:
        data_dir = Path(self.phase_config.data_dir)

        # Load only normal trajectories for pretraining
        train_ds = TrajectoryDataset(
            data_dir=data_dir / "train",
            label_filter=[0],  # normal only
            species_filter=self.phase_config.species,
            max_length=self.phase_config.max_trajectory_length,
        )
        val_ds = TrajectoryDataset(
            data_dir=data_dir / "val",
            label_filter=[0],
            species_filter=self.phase_config.species,
            max_length=self.phase_config.max_trajectory_length,
        )
        return train_ds, val_ds

    def _build_dataloader(self, dataset: Any, shuffle: bool = True) -> DataLoader:
        """Override to use TrajectoryDataset's custom collate_fn."""
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            collate_fn=TrajectoryDataset.collate_fn,
            drop_last=True,
        )

    def train_step(self, batch: Any) -> Dict[str, float]:
        assert self.model is not None and self.optimizer is not None

        features = batch["features"]  # (B, T, feat_dim)
        mask = batch["mask"]  # (B, T) True=valid
        padding_mask = ~mask  # True=padded for transformer

        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features).float().to(self.device)
        if isinstance(padding_mask, np.ndarray):
            padding_mask = torch.from_numpy(padding_mask).to(self.device)

        # Diffusion training loss
        loss_dict = self.encoder.compute_training_loss(features, padding_mask)
        loss = loss_dict["loss"]

        loss.backward()
        if self.config.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        return {"loss": loss.detach()}

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        assert self.model is not None
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            batch = self._to_device(batch)
            features = batch["features"]
            mask = batch["mask"]
            padding_mask = ~mask if isinstance(mask, torch.Tensor) else ~torch.from_numpy(mask).to(self.device)

            if isinstance(features, np.ndarray):
                features = torch.from_numpy(features).float().to(self.device)

            loss_dict = self.encoder.compute_training_loss(features, padding_mask)
            total_loss += loss_dict["loss"].item()
            n_batches += 1

        self.model.train()
        return {"loss": total_loss / max(n_batches, 1)}


# ---------------------------------------------------------------------------
# Phase 2: Anomaly Fine-tuning
# ---------------------------------------------------------------------------


@dataclass
class AnomalyFinetuneConfig(TrainerConfig):
    """Configuration for Phase 2 anomaly fine-tuning."""

    lr: float = 5e-5
    batch_size: int = 32
    epochs: int = 50
    scheduler: str = "cosine"
    weight_decay: float = 0.01
    warmup_steps: int = 500
    wandb_run_name: str = "biomotion-anomaly-finetune"

    # Data
    data_dir: str = "data/behavioral"
    species: List[str] = None  # type: ignore[assignment]
    pretrained_checkpoint: str = ""

    def __post_init__(self) -> None:
        if self.species is None:
            self.species = list(SPECIES_ORDER)


class AnomalyClassifier(nn.Module):
    """Wraps a TrajectoryDiffusionEncoder with a binary classification head."""

    def __init__(self, encoder: TrajectoryDiffusionEncoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(EMBED_DIM, EMBED_DIM // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(EMBED_DIM // 2, 1),
        )

    def forward(
        self,
        features: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return anomaly logit (B,)."""
        embedding = self.encoder.forward_encode(features, padding_mask)
        return self.classifier(embedding).squeeze(-1)


class AnomalyFinetuner(BaseTrainer):
    """Phase 2: Supervised anomaly detection fine-tuning."""

    def __init__(self, config: AnomalyFinetuneConfig) -> None:
        super().__init__(config)
        self.phase_config = config

    def build_model(self) -> nn.Module:
        species = self.phase_config.species[0] if self.phase_config.species else "daphnia"
        feature_dim = SPECIES_FEATURE_DIM.get(species, 32)
        encoder = TrajectoryDiffusionEncoder(
            feature_dim=feature_dim,
            embed_dim=EMBED_DIM,
        )

        # Load pretrained weights from Phase 1
        if self.phase_config.pretrained_checkpoint:
            ckpt_path = Path(self.phase_config.pretrained_checkpoint)
            if ckpt_path.exists():
                state = torch.load(str(ckpt_path), map_location=self.device, weights_only=False)
                encoder.load_state_dict(state["model_state_dict"], strict=False)
                logger.info(f"Loaded pretrained encoder from {ckpt_path}")
            else:
                logger.warning(f"Pretrained checkpoint not found: {ckpt_path}")

        model = AnomalyClassifier(encoder)
        return model

    def build_datasets(self) -> Tuple[Any, Any]:
        data_dir = Path(self.phase_config.data_dir)

        # Load both normal and anomalous trajectories
        train_ds = TrajectoryDataset(
            data_dir=data_dir / "train",
            species_filter=self.phase_config.species,
            max_length=1800,
        )
        val_ds = TrajectoryDataset(
            data_dir=data_dir / "val",
            species_filter=self.phase_config.species,
            max_length=1800,
        )
        return train_ds, val_ds

    def _build_dataloader(self, dataset: Any, shuffle: bool = True) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            collate_fn=TrajectoryDataset.collate_fn,
            drop_last=True,
        )

    def train_step(self, batch: Any) -> Dict[str, float]:
        assert self.model is not None and self.optimizer is not None

        features = batch["features"]
        labels = batch["labels"]
        mask = batch["mask"]
        padding_mask = ~mask

        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features).float().to(self.device)
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels).float().to(self.device)
        if isinstance(padding_mask, np.ndarray):
            padding_mask = torch.from_numpy(padding_mask).to(self.device)

        logits = self.model(features, padding_mask)
        loss = F.binary_cross_entropy_with_logits(logits, labels)

        loss.backward()
        if self.config.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        # Compute accuracy
        with torch.no_grad():
            preds = (torch.sigmoid(logits) > 0.5).float()
            acc = (preds == labels).float().mean()

        return {"loss": loss.detach(), "accuracy": acc.detach()}

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        assert self.model is not None
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_labels = []
        all_scores = []

        for batch in dataloader:
            batch = self._to_device(batch)
            features = batch["features"]
            labels = batch["labels"]
            mask = batch["mask"]
            padding_mask = ~mask if isinstance(mask, torch.Tensor) else ~torch.from_numpy(mask).to(self.device)

            if isinstance(features, np.ndarray):
                features = torch.from_numpy(features).float().to(self.device)
            if isinstance(labels, np.ndarray):
                labels = torch.from_numpy(labels).float().to(self.device)

            logits = self.model(features, padding_mask)
            loss = F.binary_cross_entropy_with_logits(logits, labels)

            preds = (torch.sigmoid(logits) > 0.5).float()
            total_loss += loss.item() * len(labels)
            total_correct += (preds == labels).sum().item()
            total_samples += len(labels)

            all_labels.append(labels.cpu())
            all_scores.append(torch.sigmoid(logits).cpu())

        self.model.train()

        avg_loss = total_loss / max(total_samples, 1)
        accuracy = total_correct / max(total_samples, 1)

        # Compute AUC if possible
        metrics = {"loss": avg_loss, "accuracy": accuracy}
        try:
            from sklearn.metrics import roc_auc_score
            all_labels_np = torch.cat(all_labels).numpy()
            all_scores_np = torch.cat(all_scores).numpy()
            if len(np.unique(all_labels_np)) > 1:
                metrics["auc"] = float(roc_auc_score(all_labels_np, all_scores_np))
        except ImportError:
            pass

        return metrics


# ---------------------------------------------------------------------------
# Phase 3: Multi-organism Ensemble Training
# ---------------------------------------------------------------------------


@dataclass
class EnsembleConfig(TrainerConfig):
    """Configuration for Phase 3 multi-organism ensemble training."""

    lr: float = 1e-4
    batch_size: int = 32
    epochs: int = 50
    scheduler: str = "cosine"
    weight_decay: float = 0.01
    warmup_steps: int = 500
    wandb_run_name: str = "biomotion-ensemble"

    # Data
    data_dir: str = "data/behavioral"
    species: List[str] = None  # type: ignore[assignment]
    pretrained_checkpoint: str = ""

    def __post_init__(self) -> None:
        if self.species is None:
            self.species = list(SPECIES_ORDER)


class MixedSpeciesDataset(Dataset):
    """Dataset that produces mixed-species batches for ensemble training.

    Each sample returns trajectory data for all available species at a
    given time window, enabling cross-organism attention training.
    """

    def __init__(
        self,
        data_dir: Path,
        species_list: List[str],
        max_length: int = 1800,
    ) -> None:
        super().__init__()
        self.species_list = species_list
        self.max_length = max_length

        # Load per-species datasets
        self._species_datasets: Dict[str, TrajectoryDataset] = {}
        for sp in species_list:
            ds = TrajectoryDataset(
                data_dir=data_dir,
                species_filter=[sp],
                max_length=max_length,
            )
            if len(ds) > 0:
                self._species_datasets[sp] = ds

        # Length = size of largest species dataset
        self._length = max(
            (len(ds) for ds in self._species_datasets.values()), default=0
        )

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        result: Dict[str, Any] = {"species_present": [], "label": 0}

        for sp, ds in self._species_datasets.items():
            sp_idx = idx % len(ds)
            sample = ds[sp_idx]
            result[sp] = sample
            result["species_present"].append(sp)
            # Use worst-case label
            if sample["label"] == 1:
                result["label"] = 1

        return result

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate mixed-species batch into organism_inputs format."""
        all_species = set()
        for sample in batch:
            all_species.update(sample["species_present"])

        organism_inputs: Dict[str, Dict[str, torch.Tensor]] = {}
        B = len(batch)

        for sp in all_species:
            # Collect samples for this species
            sp_samples = [s[sp] for s in batch if sp in s]
            if not sp_samples:
                continue

            collated = TrajectoryDataset.collate_fn(sp_samples)
            organism_inputs[sp] = {
                "keypoints": torch.from_numpy(collated["keypoints"]).float(),
                "features": torch.from_numpy(collated["features"]).float(),
                "padding_mask": torch.from_numpy(~collated["mask"]),
            }

        labels = torch.tensor([s["label"] for s in batch], dtype=torch.float32)
        return {"organism_inputs": organism_inputs, "labels": labels}


class EnsembleTrainer(BaseTrainer):
    """Phase 3: Train cross-organism attention with mixed-species batches."""

    def __init__(self, config: EnsembleConfig) -> None:
        super().__init__(config)
        self.phase_config = config

    def build_model(self) -> nn.Module:
        model = BioMotionEncoder(
            species_list=self.phase_config.species,
            embed_dim=EMBED_DIM,
            shared_embed_dim=SHARED_EMBED_DIM,
        )

        # Load pretrained weights from Phase 2 if available
        if self.phase_config.pretrained_checkpoint:
            ckpt_path = Path(self.phase_config.pretrained_checkpoint)
            if ckpt_path.exists():
                state = torch.load(str(ckpt_path), map_location=self.device, weights_only=False)
                model.load_state_dict(state["model_state_dict"], strict=False)
                logger.info(f"Loaded pretrained weights from {ckpt_path}")
            else:
                logger.warning(f"Pretrained checkpoint not found: {ckpt_path}")

        return model

    def build_datasets(self) -> Tuple[Any, Any]:
        data_dir = Path(self.phase_config.data_dir)
        train_ds = MixedSpeciesDataset(
            data_dir=data_dir / "train",
            species_list=self.phase_config.species,
        )
        val_ds = MixedSpeciesDataset(
            data_dir=data_dir / "val",
            species_list=self.phase_config.species,
        )
        return train_ds, val_ds

    def _build_dataloader(self, dataset: Any, shuffle: bool = True) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            collate_fn=MixedSpeciesDataset.collate_fn,
            drop_last=True,
        )

    def train_step(self, batch: Any) -> Dict[str, float]:
        assert self.model is not None and self.optimizer is not None

        organism_inputs = batch["organism_inputs"]
        labels = batch["labels"]

        # Move inputs to device
        for sp in organism_inputs:
            for k, v in organism_inputs[sp].items():
                if isinstance(v, torch.Tensor):
                    organism_inputs[sp][k] = v.to(self.device)
        labels = labels.to(self.device)

        output = self.model(organism_inputs)
        anomaly_score = output["anomaly_score"]

        # Binary cross-entropy on anomaly score
        loss = F.binary_cross_entropy(
            anomaly_score.clamp(1e-7, 1.0 - 1e-7), labels,
        )

        loss.backward()
        if self.config.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            preds = (anomaly_score > 0.5).float()
            acc = (preds == labels).float().mean()

        return {"loss": loss.detach(), "accuracy": acc.detach()}

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        assert self.model is not None
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch in dataloader:
            organism_inputs = batch["organism_inputs"]
            labels = batch["labels"]

            for sp in organism_inputs:
                for k, v in organism_inputs[sp].items():
                    if isinstance(v, torch.Tensor):
                        organism_inputs[sp][k] = v.to(self.device)
            labels = labels.to(self.device)

            output = self.model(organism_inputs)
            anomaly_score = output["anomaly_score"]

            loss = F.binary_cross_entropy(
                anomaly_score.clamp(1e-7, 1.0 - 1e-7), labels,
            )

            preds = (anomaly_score > 0.5).float()
            total_loss += loss.item() * len(labels)
            total_correct += (preds == labels).sum().item()
            total_samples += len(labels)

        self.model.train()
        return {
            "loss": total_loss / max(total_samples, 1),
            "accuracy": total_correct / max(total_samples, 1),
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SENTINEL BioMotion encoder training pipeline",
    )
    parser.add_argument(
        "--phase", type=int, required=True, choices=[1, 2, 3],
        help="Training phase: 1=diffusion pretrain, 2=anomaly finetune, 3=ensemble",
    )
    parser.add_argument("--data-dir", type=str, default="data/behavioral")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="",
                        help="Pretrained checkpoint (phases 2 and 3)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--species", nargs="+", default=None)
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

    output_dir = args.output_dir or f"outputs/biomotion/phase{args.phase}"

    if args.phase == 1:
        config = DiffusionPretrainConfig(
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
        if args.species is not None:
            config.species = args.species

        trainer = DiffusionPretrainer(config)

    elif args.phase == 2:
        config = AnomalyFinetuneConfig(
            data_dir=args.data_dir,
            output_dir=output_dir,
            pretrained_checkpoint=args.checkpoint,
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
        if args.species is not None:
            config.species = args.species

        trainer = AnomalyFinetuner(config)

    elif args.phase == 3:
        config = EnsembleConfig(
            data_dir=args.data_dir,
            output_dir=output_dir,
            pretrained_checkpoint=args.checkpoint,
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
        if args.species is not None:
            config.species = args.species

        trainer = EnsembleTrainer(config)
    else:
        raise ValueError(f"Unknown phase: {args.phase}")

    logger.info(f"Starting BioMotion Phase {args.phase} training")
    trainer.setup()

    if args.resume:
        trainer.load_checkpoint(args.resume)

    summary = trainer.train()
    logger.info(f"Phase {args.phase} complete: {json.dumps(summary, indent=2, default=str)}")


if __name__ == "__main__":
    main()
