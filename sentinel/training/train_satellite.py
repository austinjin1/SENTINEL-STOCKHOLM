"""
HydroViT satellite encoder training pipeline for SENTINEL.

Three-phase training:
  Phase 1: Masked Autoencoder (MAE) pretraining on water-pixel patches.
            75% mask ratio with combined reconstruction MSE + spectral physics
            consistency loss.
  Phase 2: 16-parameter supervised fine-tuning on co-registered satellite +
            in-situ measurement pairs. Gaussian NLL loss with per-parameter
            missing-value masking.
  Phase 3: Temporal stack training on sequences of 5-10 images per location.
            Self-supervised temporal prediction + supervised objectives.

Usage:
    python -m sentinel.training.train_satellite --phase 1 --data-dir data/satellite/mae
    python -m sentinel.training.train_satellite --phase 2 --data-dir data/satellite/pairs
    python -m sentinel.training.train_satellite --phase 3 --data-dir data/satellite/temporal
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from sentinel.models.satellite_encoder.model import SatelliteEncoder, SHARED_EMBED_DIM
from sentinel.models.satellite_encoder.hydrovit_backbone import (
    NUM_SPECTRAL_BANDS,
    VIT_EMBED_DIM,
)
from sentinel.models.satellite_encoder.parameter_head import (
    NUM_WATER_PARAMS,
    WaterQualityHead,
)
from sentinel.training.trainer import BaseTrainer, TrainerConfig, build_scheduler
from sentinel.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data augmentation
# ---------------------------------------------------------------------------

class SatelliteAugmentation:
    """Augmentation pipeline for satellite imagery.

    Includes random crop, horizontal/vertical flip, spectral jitter,
    and simulated cloud masking.
    """

    def __init__(
        self,
        crop_size: int = 224,
        flip_prob: float = 0.5,
        spectral_jitter: float = 0.05,
        cloud_mask_prob: float = 0.1,
        cloud_mask_size_range: Tuple[int, int] = (20, 60),
    ) -> None:
        self.crop_size = crop_size
        self.flip_prob = flip_prob
        self.spectral_jitter = spectral_jitter
        self.cloud_mask_prob = cloud_mask_prob
        self.cloud_mask_size_range = cloud_mask_size_range

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Apply augmentations to image (H, W, C)."""
        h, w = image.shape[:2]

        # Random crop
        if h > self.crop_size and w > self.crop_size:
            y = random.randint(0, h - self.crop_size)
            x = random.randint(0, w - self.crop_size)
            image = image[y:y + self.crop_size, x:x + self.crop_size]

        # Horizontal flip
        if random.random() < self.flip_prob:
            image = np.flip(image, axis=1).copy()

        # Vertical flip
        if random.random() < self.flip_prob:
            image = np.flip(image, axis=0).copy()

        # Spectral jitter
        if self.spectral_jitter > 0:
            noise = np.random.normal(
                0, self.spectral_jitter, size=(1, 1, image.shape[2])
            ).astype(np.float32)
            image = np.clip(image + noise, 0.0, 1.0)

        # Simulated cloud masking
        if random.random() < self.cloud_mask_prob:
            ch, cw = image.shape[:2]
            size = random.randint(*self.cloud_mask_size_range)
            cy = random.randint(0, max(0, ch - size))
            cx = random.randint(0, max(0, cw - size))
            image[cy:cy + size, cx:cx + size, :] = np.random.uniform(
                0.8, 1.0,
                size=(min(size, ch - cy), min(size, cw - cx), image.shape[2]),
            ).astype(np.float32)

        return image


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class MAEDataset(Dataset):
    """Dataset for MAE pretraining on water-pixel patches.

    Loads patches extracted by the data pipeline's extract_water_pixels()
    function. Each .npy file is a (H, W, C) float32 multispectral patch
    where C = NUM_SPECTRAL_BANDS (13 bands).

    Expected directory structure:
        data_dir/
            patches/    -- .npy files (224, 224, 13) float32
    """

    def __init__(
        self,
        data_dir: str | Path,
        augmentation: Optional[SatelliteAugmentation] = None,
        target_size: int = 224,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.target_size = target_size
        self.augmentation = augmentation or SatelliteAugmentation(crop_size=target_size)

        patches_dir = self.data_dir / "patches"
        if patches_dir.exists():
            self.patch_paths: List[Path] = sorted(patches_dir.glob("*.npy"))
        else:
            # Fall back to top-level .npy files
            self.patch_paths = sorted(self.data_dir.glob("*.npy"))

        logger.info(f"MAEDataset: {len(self.patch_paths)} patches from {self.data_dir}")

    def __len__(self) -> int:
        return len(self.patch_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        patch_path = self.patch_paths[idx]
        patch = np.load(patch_path).astype(np.float32)

        # Normalize reflectance if in uint16 range
        if patch.max() > 2.0:
            patch = patch / 10000.0

        # Ensure (H, W, C) format
        if patch.ndim == 3 and patch.shape[0] == NUM_SPECTRAL_BANDS:
            patch = patch.transpose(1, 2, 0)

        # Apply augmentations
        patch = self.augmentation(patch)

        # Resize to target size if needed
        if patch.shape[0] != self.target_size or patch.shape[1] != self.target_size:
            from scipy.ndimage import zoom as scipy_zoom
            factors = (
                self.target_size / patch.shape[0],
                self.target_size / patch.shape[1],
                1.0,
            )
            patch = scipy_zoom(patch, factors, order=1).astype(np.float32)

        # (H, W, C) -> (C, H, W)
        tensor = torch.from_numpy(patch.transpose(2, 0, 1))

        return {"image": tensor}


class WaterQualityPairDataset(Dataset):
    """Dataset of co-registered satellite images + in-situ water quality measurements.

    Each sample pairs a satellite patch with ground-truth values for up to
    16 water quality parameters. Missing parameters are encoded as NaN.

    Expected directory structure:
        data_dir/
            images/     -- .npy files (H, W, C) float32
            labels.json -- {image_id: {"params": [16 floats, NaN for missing],
                                        "timestamp": str, "station_id": str}}
    """

    def __init__(
        self,
        data_dir: str | Path,
        augmentation: Optional[SatelliteAugmentation] = None,
        target_size: int = 224,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.target_size = target_size
        self.augmentation = augmentation

        images_dir = self.data_dir / "images"
        self.image_paths: List[Path] = sorted(images_dir.glob("*.npy"))

        labels_path = self.data_dir / "labels.json"
        self.labels: Dict[str, Dict] = {}
        if labels_path.exists():
            with open(labels_path, "r", encoding="utf-8") as f:
                self.labels = json.load(f)

        logger.info(
            f"WaterQualityPairDataset: {len(self.image_paths)} images, "
            f"{len(self.labels)} labels"
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path = self.image_paths[idx]
        image = np.load(img_path).astype(np.float32)

        if image.max() > 2.0:
            image = image / 10000.0

        # Ensure (H, W, C)
        if image.ndim == 3 and image.shape[0] in (NUM_SPECTRAL_BANDS, 10):
            image = image.transpose(1, 2, 0)

        if self.augmentation is not None:
            image = self.augmentation(image)

        if image.shape[0] != self.target_size or image.shape[1] != self.target_size:
            from scipy.ndimage import zoom as scipy_zoom
            factors = (
                self.target_size / image.shape[0],
                self.target_size / image.shape[1],
                1.0,
            )
            image = scipy_zoom(image, factors, order=1).astype(np.float32)

        # Pad to 13 bands if only 10 (S2 only, no S3)
        if image.shape[2] < NUM_SPECTRAL_BANDS:
            pad_bands = NUM_SPECTRAL_BANDS - image.shape[2]
            image = np.pad(image, ((0, 0), (0, 0), (0, pad_bands)), mode="constant")

        tensor = torch.from_numpy(image.transpose(2, 0, 1))

        # Load water quality labels
        img_id = img_path.stem
        label_info = self.labels.get(img_id, {})
        params = label_info.get("params", [float("nan")] * NUM_WATER_PARAMS)
        params = np.array(params, dtype=np.float32)
        if len(params) < NUM_WATER_PARAMS:
            params = np.pad(
                params,
                (0, NUM_WATER_PARAMS - len(params)),
                constant_values=float("nan"),
            )

        return {
            "image": tensor,
            "wq_targets": torch.from_numpy(params),
        }


class TemporalStackDataset(Dataset):
    """Dataset of temporal image sequences for temporal attention training.

    Each sample is a sequence of 5-10 satellite images over the same location
    at different times, with optional water quality labels for the most
    recent frame.

    Expected directory structure:
        data_dir/
            sequences/
                <location_id>/
                    frames/     -- .npy files named by date (YYYYMMDD.npy)
                    metadata.json -- {"timestamps": [...], "cloud_fractions": [...],
                                      "wq_targets": [16 floats or null]}
    """

    def __init__(
        self,
        data_dir: str | Path,
        max_temporal_len: int = 10,
        min_temporal_len: int = 5,
        target_size: int = 224,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.max_temporal_len = max_temporal_len
        self.min_temporal_len = min_temporal_len
        self.target_size = target_size

        sequences_dir = self.data_dir / "sequences"
        self.sequence_dirs: List[Path] = []
        if sequences_dir.exists():
            for loc_dir in sorted(sequences_dir.iterdir()):
                if loc_dir.is_dir() and (loc_dir / "frames").exists():
                    frames = sorted((loc_dir / "frames").glob("*.npy"))
                    if len(frames) >= min_temporal_len:
                        self.sequence_dirs.append(loc_dir)

        logger.info(
            f"TemporalStackDataset: {len(self.sequence_dirs)} sequences "
            f"(min {min_temporal_len} frames)"
        )

    def __len__(self) -> int:
        return len(self.sequence_dirs)

    def _load_and_preprocess(self, path: Path) -> np.ndarray:
        """Load a single frame and preprocess."""
        img = np.load(path).astype(np.float32)
        if img.max() > 2.0:
            img = img / 10000.0
        if img.ndim == 3 and img.shape[0] in (NUM_SPECTRAL_BANDS, 10):
            img = img.transpose(1, 2, 0)
        if img.shape[0] != self.target_size or img.shape[1] != self.target_size:
            from scipy.ndimage import zoom as scipy_zoom
            factors = (
                self.target_size / img.shape[0],
                self.target_size / img.shape[1],
                1.0,
            )
            img = scipy_zoom(img, factors, order=1).astype(np.float32)
        if img.shape[2] < NUM_SPECTRAL_BANDS:
            pad = NUM_SPECTRAL_BANDS - img.shape[2]
            img = np.pad(img, ((0, 0), (0, 0), (0, pad)), mode="constant")
        return img

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        loc_dir = self.sequence_dirs[idx]
        frames_dir = loc_dir / "frames"

        # Load metadata
        meta_path = loc_dir / "metadata.json"
        metadata: Dict[str, Any] = {}
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

        frame_files = sorted(frames_dir.glob("*.npy"))

        # Subsample to max_temporal_len if needed
        if len(frame_files) > self.max_temporal_len:
            # Keep the last frame, randomly sample the rest
            last = frame_files[-1]
            others = random.sample(frame_files[:-1], self.max_temporal_len - 1)
            frame_files = sorted(others) + [last]

        T = len(frame_files)

        # Load all frames
        frames_list = []
        for fp in frame_files:
            img = self._load_and_preprocess(fp)
            frames_list.append(img.transpose(2, 0, 1))  # (C, H, W)
        frames = np.stack(frames_list, axis=0)  # (T, C, H, W)

        # Timestamps (days since epoch)
        timestamps = metadata.get("timestamps", list(range(T)))
        timestamps = np.array(timestamps[:T], dtype=np.float32)

        # Cloud fractions
        cloud_fractions = metadata.get("cloud_fractions", [0.0] * T)
        cloud_fractions = np.array(cloud_fractions[:T], dtype=np.float32)

        # Padding to max_temporal_len
        pad_len = self.max_temporal_len - T
        padding_mask = np.zeros(self.max_temporal_len, dtype=bool)
        if pad_len > 0:
            frames = np.pad(
                frames,
                ((pad_len, 0), (0, 0), (0, 0), (0, 0)),
                mode="constant",
            )
            timestamps = np.pad(timestamps, (pad_len, 0), constant_values=0)
            cloud_fractions = np.pad(cloud_fractions, (pad_len, 0), constant_values=1.0)
            padding_mask[:pad_len] = True

        # Water quality targets for the latest frame (if available)
        wq_targets = metadata.get("wq_targets", None)
        if wq_targets is not None:
            wq_targets = np.array(wq_targets, dtype=np.float32)
            if len(wq_targets) < NUM_WATER_PARAMS:
                wq_targets = np.pad(
                    wq_targets,
                    (0, NUM_WATER_PARAMS - len(wq_targets)),
                    constant_values=float("nan"),
                )
        else:
            wq_targets = np.full(NUM_WATER_PARAMS, float("nan"), dtype=np.float32)

        return {
            "frames": torch.from_numpy(frames),                  # (T, C, H, W)
            "timestamps": torch.from_numpy(timestamps),           # (T,)
            "cloud_fractions": torch.from_numpy(cloud_fractions), # (T,)
            "padding_mask": torch.from_numpy(padding_mask),       # (T,)
            "wq_targets": torch.from_numpy(wq_targets),          # (16,)
        }


# ---------------------------------------------------------------------------
# Phase 1: MAE Pretraining Trainer
# ---------------------------------------------------------------------------

@dataclass
class MAEPretrainConfig(TrainerConfig):
    """Configuration for Phase 1: MAE pretraining."""

    lr: float = 1.5e-4
    batch_size: int = 32
    epochs: int = 100
    warmup_epochs: int = 10
    scheduler: str = "cosine"
    weight_decay: float = 0.05
    wandb_run_name: str = "hydrovit-mae-pretrain"

    # Data
    data_dir: str = "data/satellite/mae"
    val_fraction: float = 0.1
    target_size: int = 224

    # MAE
    mask_ratio: float = 0.75
    physics_loss_weight: float = 0.1

    # Augmentation
    spectral_jitter: float = 0.03
    cloud_mask_prob: float = 0.1


class MAEPretrainTrainer(BaseTrainer):
    """Phase 1: MAE pretraining with spectral physics consistency loss."""

    def __init__(self, config: MAEPretrainConfig) -> None:
        super().__init__(config)
        self.mae_config = config

    def build_model(self) -> nn.Module:
        model = SatelliteEncoder(
            in_chans=NUM_SPECTRAL_BANDS,
            pretrained=True,
            enable_s3_fusion=False,
        )
        return model

    def build_datasets(self) -> Tuple[Dataset, Dataset]:
        augmentation = SatelliteAugmentation(
            crop_size=self.mae_config.target_size,
            spectral_jitter=self.mae_config.spectral_jitter,
            cloud_mask_prob=self.mae_config.cloud_mask_prob,
        )

        full_ds = MAEDataset(
            self.mae_config.data_dir,
            augmentation=augmentation,
            target_size=self.mae_config.target_size,
        )

        n = len(full_ds)
        n_val = max(1, int(n * self.mae_config.val_fraction))
        indices = list(range(n))
        random.shuffle(indices)
        train_ds = Subset(full_ds, indices[n_val:])
        val_ds = Subset(full_ds, indices[:n_val])

        logger.info(f"MAE Pretrain — Train: {len(train_ds)}, Val: {len(val_ds)}")
        return train_ds, val_ds

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        assert self.model is not None and self.optimizer is not None

        images = batch["image"]  # (B, C, H, W)

        # MAE forward pass through the SatelliteEncoder
        mae_output = self.model.forward_mae(
            images,
            mask_ratio=self.mae_config.mask_ratio,
        )

        mae_loss = mae_output["mae_loss"]

        # Spectral physics consistency loss
        physics_loss = torch.tensor(0.0, device=images.device)
        for key in ("ndwi_consistency", "nir_water_penalty", "spectral_smoothness"):
            if key in mae_output:
                physics_loss = physics_loss + mae_output[key]

        total_loss = mae_loss + self.mae_config.physics_loss_weight * physics_loss

        total_loss.backward()

        if self.config.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        return {
            "loss": total_loss.item(),
            "mae_loss": mae_loss.item(),
            "physics_loss": physics_loss.item(),
        }

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        assert self.model is not None
        self.model.eval()

        total_mae = 0.0
        total_physics = 0.0
        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            batch = self._to_device(batch)
            images = batch["image"]

            mae_output = self.model.forward_mae(
                images,
                mask_ratio=self.mae_config.mask_ratio,
            )

            mae_loss = mae_output["mae_loss"].item()
            physics_loss = 0.0
            for key in ("ndwi_consistency", "nir_water_penalty", "spectral_smoothness"):
                if key in mae_output:
                    physics_loss += mae_output[key].item()

            total_mae += mae_loss
            total_physics += physics_loss
            total_loss += mae_loss + self.mae_config.physics_loss_weight * physics_loss
            n_batches += 1

        self.model.train()
        n = max(n_batches, 1)
        return {
            "loss": total_loss / n,
            "mae_loss": total_mae / n,
            "physics_loss": total_physics / n,
        }


# ---------------------------------------------------------------------------
# Phase 2: Supervised Fine-tuning Trainer
# ---------------------------------------------------------------------------

@dataclass
class SupervisedFinetuneConfig(TrainerConfig):
    """Configuration for Phase 2: 16-parameter supervised fine-tuning."""

    lr: float = 1e-4
    batch_size: int = 32
    epochs: int = 50
    scheduler: str = "cosine"
    weight_decay: float = 0.05
    wandb_run_name: str = "hydrovit-supervised-finetune"

    # Data
    data_dir: str = "data/satellite/pairs"
    val_fraction: float = 0.2
    target_size: int = 224

    # Model
    pretrain_checkpoint: str = ""
    freeze_backbone_epochs: int = 5  # freeze backbone for first N epochs
    unfreeze_top_n: int = 4

    # Augmentation
    spectral_jitter: float = 0.02
    cloud_mask_prob: float = 0.0

    # Loss
    physics_loss_weight: float = 0.05


class SupervisedFinetuneTrainer(BaseTrainer):
    """Phase 2: 16-parameter water quality regression with Gaussian NLL loss."""

    def __init__(self, config: SupervisedFinetuneConfig) -> None:
        super().__init__(config)
        self.ft_config = config
        self._backbone_frozen = False

    def build_model(self) -> nn.Module:
        model = SatelliteEncoder(
            in_chans=NUM_SPECTRAL_BANDS,
            pretrained=True,
            enable_s3_fusion=False,
        )

        # Load MAE pretrained checkpoint
        if self.ft_config.pretrain_checkpoint:
            ckpt_path = Path(self.ft_config.pretrain_checkpoint)
            if ckpt_path.exists():
                logger.info(f"Loading MAE pretrained weights from {ckpt_path}")
                state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                model_state = state.get("model_state_dict", state)
                model.load_state_dict(model_state, strict=False)
                logger.info("MAE pretrained weights loaded")

        # Optionally freeze backbone initially
        if self.ft_config.freeze_backbone_epochs > 0:
            self.freeze_model(model.backbone)
            self._backbone_frozen = True
            logger.info(
                f"Backbone frozen for first {self.ft_config.freeze_backbone_epochs} epochs"
            )

        return model

    def build_datasets(self) -> Tuple[Dataset, Dataset]:
        augmentation = SatelliteAugmentation(
            crop_size=self.ft_config.target_size,
            spectral_jitter=self.ft_config.spectral_jitter,
            cloud_mask_prob=self.ft_config.cloud_mask_prob,
        )

        full_ds = WaterQualityPairDataset(
            self.ft_config.data_dir,
            augmentation=augmentation,
            target_size=self.ft_config.target_size,
        )

        n = len(full_ds)
        n_val = max(1, int(n * self.ft_config.val_fraction))
        indices = list(range(n))
        random.shuffle(indices)
        train_ds = Subset(full_ds, indices[n_val:])
        val_ds = Subset(full_ds, indices[:n_val])

        logger.info(f"Supervised Finetune — Train: {len(train_ds)}, Val: {len(val_ds)}")
        return train_ds, val_ds

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        assert self.model is not None and self.optimizer is not None

        # Unfreeze backbone after warmup epochs
        if (
            self._backbone_frozen
            and self.current_epoch >= self.ft_config.freeze_backbone_epochs
        ):
            self.unfreeze_top_layers(self.model.backbone, self.ft_config.unfreeze_top_n)
            self._backbone_frozen = False
            logger.info(
                f"Unfroze top {self.ft_config.unfreeze_top_n} backbone layers "
                f"at epoch {self.current_epoch}"
            )

        images = batch["image"]         # (B, C, H, W)
        wq_targets = batch["wq_targets"]  # (B, 16)

        # Forward pass
        outputs = self.model(images)

        # Gaussian NLL loss on water quality parameters
        wq_loss_dict = self.model.compute_loss(outputs, wq_targets=wq_targets)
        wq_loss = wq_loss_dict.get("wq_loss", torch.tensor(0.0, device=images.device))

        total_loss = wq_loss

        total_loss.backward()

        if self.config.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        # Compute per-parameter metrics for logging
        with torch.no_grad():
            valid_mask = ~torch.isnan(wq_targets)
            if valid_mask.any():
                abs_errors = torch.abs(
                    outputs["water_quality_params"] - wq_targets
                )
                abs_errors[~valid_mask] = 0.0
                n_valid = valid_mask.float().sum(dim=0).clamp(min=1)
                mean_ae = (abs_errors.sum(dim=0) / n_valid).mean().item()
            else:
                mean_ae = 0.0

        return {
            "loss": total_loss.item(),
            "wq_loss": wq_loss.item(),
            "mean_ae": mean_ae,
        }

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        assert self.model is not None
        self.model.eval()

        total_loss = 0.0
        all_preds = []
        all_targets = []
        n_batches = 0

        for batch in dataloader:
            batch = self._to_device(batch)
            images = batch["image"]
            wq_targets = batch["wq_targets"]

            outputs = self.model(images)
            loss_dict = self.model.compute_loss(outputs, wq_targets=wq_targets)
            wq_loss = loss_dict.get("wq_loss", torch.tensor(0.0))
            total_loss += wq_loss.item()
            n_batches += 1

            all_preds.append(outputs["water_quality_params"].cpu())
            all_targets.append(wq_targets.cpu())

        self.model.train()

        # Aggregate metrics
        preds = torch.cat(all_preds, dim=0)    # (N, 16)
        targets = torch.cat(all_targets, dim=0)  # (N, 16)
        valid = ~torch.isnan(targets)

        metrics: Dict[str, float] = {
            "loss": total_loss / max(n_batches, 1),
        }

        # Per-parameter MAE
        if valid.any():
            abs_err = torch.abs(preds - targets)
            abs_err[~valid] = 0.0
            n_valid_per_param = valid.float().sum(dim=0).clamp(min=1)
            param_mae = abs_err.sum(dim=0) / n_valid_per_param
            metrics["mean_mae"] = param_mae.mean().item()

            # R-squared per parameter (averaged over params with enough samples)
            for p_idx in range(NUM_WATER_PARAMS):
                mask = valid[:, p_idx]
                if mask.sum() > 10:
                    y = targets[mask, p_idx]
                    y_hat = preds[mask, p_idx]
                    ss_res = ((y - y_hat) ** 2).sum()
                    ss_tot = ((y - y.mean()) ** 2).sum().clamp(min=1e-6)
                    r2 = 1.0 - (ss_res / ss_tot)
                    metrics[f"r2_param_{p_idx}"] = r2.item()

        return metrics


# ---------------------------------------------------------------------------
# Phase 3: Temporal Stack Trainer
# ---------------------------------------------------------------------------

@dataclass
class TemporalStackConfig(TrainerConfig):
    """Configuration for Phase 3: temporal stack training."""

    lr: float = 5e-5
    batch_size: int = 8
    epochs: int = 50
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    weight_decay: float = 0.05
    wandb_run_name: str = "hydrovit-temporal-stack"

    # Data
    data_dir: str = "data/satellite/temporal"
    val_fraction: float = 0.2
    max_temporal_len: int = 10
    min_temporal_len: int = 5
    target_size: int = 224

    # Model
    pretrain_checkpoint: str = ""
    freeze_backbone: bool = True

    # Loss weights
    ssl_weight: float = 1.0
    supervised_weight: float = 0.5
    temporal_mask_ratio: float = 0.2


class TemporalStackTrainer(BaseTrainer):
    """Phase 3: Train temporal attention stack on multi-date imagery sequences."""

    def __init__(self, config: TemporalStackConfig) -> None:
        super().__init__(config)
        self.temp_config = config

    def build_model(self) -> nn.Module:
        model = SatelliteEncoder(
            in_chans=NUM_SPECTRAL_BANDS,
            pretrained=True,
            max_temporal_len=self.temp_config.max_temporal_len + 1,
            enable_s3_fusion=False,
        )

        # Load Phase 2 (or Phase 1) checkpoint
        if self.temp_config.pretrain_checkpoint:
            ckpt_path = Path(self.temp_config.pretrain_checkpoint)
            if ckpt_path.exists():
                logger.info(f"Loading pretrained weights from {ckpt_path}")
                state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                model_state = state.get("model_state_dict", state)
                model.load_state_dict(model_state, strict=False)
                logger.info("Pretrained weights loaded")

        # Freeze backbone + projection, train temporal stack only
        if self.temp_config.freeze_backbone:
            self.freeze_model(model.backbone)
            self.freeze_model(model.projection)
            self.freeze_model(model.water_quality_head)
            logger.info("Backbone, projection, and WQ head frozen; training temporal stack")

        return model

    def build_datasets(self) -> Tuple[Dataset, Dataset]:
        full_ds = TemporalStackDataset(
            self.temp_config.data_dir,
            max_temporal_len=self.temp_config.max_temporal_len,
            min_temporal_len=self.temp_config.min_temporal_len,
            target_size=self.temp_config.target_size,
        )

        n = len(full_ds)
        n_val = max(1, int(n * self.temp_config.val_fraction))
        indices = list(range(n))
        random.shuffle(indices)
        train_ds = Subset(full_ds, indices[n_val:])
        val_ds = Subset(full_ds, indices[:n_val])

        logger.info(f"Temporal Stack — Train: {len(train_ds)}, Val: {len(val_ds)}")
        return train_ds, val_ds

    def _extract_cls_tokens(
        self,
        frames: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Extract CLS tokens for each frame in the stack.

        Args:
            frames: (B, T, C, H, W) image stack.
            padding_mask: (B, T) True = padded/invalid.

        Returns:
            cls_tokens: (B, T, D) CLS embeddings per frame.
        """
        B, T, C, H, W = frames.shape

        # Process each frame through the backbone
        cls_tokens = torch.zeros(B, T, VIT_EMBED_DIM, device=frames.device)

        with torch.no_grad():
            for t in range(T):
                # Skip padded frames
                valid = ~padding_mask[:, t]
                if not valid.any():
                    continue

                frame_batch = frames[valid, t]  # (B_valid, C, H, W)
                cls, _ = self.model.backbone(frame_batch)
                cls_tokens[valid, t] = cls

        return cls_tokens

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        assert self.model is not None and self.optimizer is not None

        frames = batch["frames"]                # (B, T, C, H, W)
        timestamps = batch["timestamps"]         # (B, T)
        cloud_fractions = batch["cloud_fractions"]  # (B, T)
        padding_mask = batch["padding_mask"]     # (B, T)
        wq_targets = batch["wq_targets"]         # (B, 16)

        B, T, C, H, W = frames.shape

        # Extract CLS tokens for all frames (frozen backbone)
        cls_tokens = self._extract_cls_tokens(frames, padding_mask)

        # Split into historical (T-1) and current (last frame)
        historical_tokens = cls_tokens[:, :-1, :]     # (B, T-1, D)
        current_frame = frames[:, -1, :, :, :]        # (B, C, H, W)

        # Self-supervised: mask some historical frames and predict them
        original_tokens = historical_tokens.clone()
        ssl_loss = torch.tensor(0.0, device=frames.device)

        T_hist = T - 1
        valid_counts = (~padding_mask[:, :-1]).float().sum(dim=1)  # (B,)

        # Create temporal prediction mask
        temporal_mask = torch.zeros(B, T_hist, dtype=torch.bool, device=frames.device)
        for b in range(B):
            n_valid = int(valid_counts[b].item())
            if n_valid > 2:
                n_mask = max(1, int(n_valid * self.temp_config.temporal_mask_ratio))
                valid_idx = torch.where(~padding_mask[b, :-1])[0]
                # Don't mask the most recent historical frame
                maskable = valid_idx[:-1] if len(valid_idx) > 1 else valid_idx
                if len(maskable) > 0:
                    chosen = maskable[torch.randperm(len(maskable))[:n_mask]]
                    temporal_mask[b, chosen] = True

        # Zero out masked historical tokens
        historical_tokens[temporal_mask] = 0.0

        # Forward through the full model with temporal stack
        outputs = self.model(
            image=current_frame,
            temporal_frames=historical_tokens,
            temporal_timestamps=timestamps,
            temporal_cloud_fractions=cloud_fractions,
            temporal_mask=padding_mask,
        )

        losses: Dict[str, torch.Tensor] = {}

        # SSL loss: predict masked temporal embeddings from context
        if temporal_mask.any():
            # Re-extract temporal features from the temporal stack
            # Use the temporal_embedding output as a proxy for the full representation
            temporal_emb = outputs["temporal_embedding"]  # (B, 256)

            # Simple SSL: predict masked CLS tokens from temporal context
            # Use a projection from the temporal embedding
            if not hasattr(self, "_temporal_ssl_head"):
                self._temporal_ssl_head = nn.Linear(
                    SHARED_EMBED_DIM, VIT_EMBED_DIM
                ).to(frames.device)

            pred_cls = self._temporal_ssl_head(temporal_emb)  # (B, D)

            # Average masked tokens as target
            masked_targets = []
            for b in range(B):
                if temporal_mask[b].any():
                    masked_targets.append(original_tokens[b, temporal_mask[b]].mean(dim=0))
                else:
                    masked_targets.append(torch.zeros(VIT_EMBED_DIM, device=frames.device))
            masked_target = torch.stack(masked_targets, dim=0)

            has_masked = temporal_mask.any(dim=1)
            if has_masked.any():
                ssl_loss = F.mse_loss(pred_cls[has_masked], masked_target[has_masked])

        losses["ssl_loss"] = ssl_loss * self.temp_config.ssl_weight

        # Supervised loss: water quality prediction from temporal context
        valid_wq = ~torch.isnan(wq_targets).all(dim=1)
        if valid_wq.any():
            wq_loss_dict = self.model.compute_loss(
                outputs, wq_targets=wq_targets
            )
            supervised_loss = wq_loss_dict.get(
                "wq_loss", torch.tensor(0.0, device=frames.device)
            )
            losses["supervised_loss"] = supervised_loss * self.temp_config.supervised_weight
        else:
            losses["supervised_loss"] = torch.tensor(0.0, device=frames.device)

        total_loss = sum(losses.values())
        losses["loss"] = total_loss

        total_loss.backward()

        if self.config.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        assert self.model is not None
        self.model.eval()

        total_loss = 0.0
        total_ssl = 0.0
        total_supervised = 0.0
        n_batches = 0
        all_preds = []
        all_targets = []

        for batch in dataloader:
            batch = self._to_device(batch)
            frames = batch["frames"]
            timestamps = batch["timestamps"]
            cloud_fractions = batch["cloud_fractions"]
            padding_mask = batch["padding_mask"]
            wq_targets = batch["wq_targets"]

            B, T, C, H, W = frames.shape

            # Extract CLS tokens
            cls_tokens = self._extract_cls_tokens(frames, padding_mask)
            historical_tokens = cls_tokens[:, :-1, :]
            current_frame = frames[:, -1, :, :, :]

            outputs = self.model(
                image=current_frame,
                temporal_frames=historical_tokens,
                temporal_timestamps=timestamps,
                temporal_cloud_fractions=cloud_fractions,
                temporal_mask=padding_mask,
            )

            # Supervised loss
            valid_wq = ~torch.isnan(wq_targets).all(dim=1)
            if valid_wq.any():
                wq_loss_dict = self.model.compute_loss(outputs, wq_targets=wq_targets)
                supervised_loss = wq_loss_dict.get("wq_loss", torch.tensor(0.0))
                total_supervised += supervised_loss.item()
                total_loss += supervised_loss.item()

                all_preds.append(outputs["water_quality_params"].cpu())
                all_targets.append(wq_targets.cpu())

            n_batches += 1

        self.model.train()

        metrics: Dict[str, float] = {
            "loss": total_loss / max(n_batches, 1),
            "supervised_loss": total_supervised / max(n_batches, 1),
        }

        if all_preds:
            preds = torch.cat(all_preds, dim=0)
            targets = torch.cat(all_targets, dim=0)
            valid = ~torch.isnan(targets)
            if valid.any():
                abs_err = torch.abs(preds - targets)
                abs_err[~valid] = 0.0
                n_valid_per_param = valid.float().sum(dim=0).clamp(min=1)
                param_mae = abs_err.sum(dim=0) / n_valid_per_param
                metrics["mean_mae"] = param_mae.mean().item()

        return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SENTINEL HydroViT satellite encoder training pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--phase", type=int, required=True, choices=[1, 2, 3],
        help="Training phase (1=MAE pretrain, 2=supervised finetune, 3=temporal stack)",
    )
    parser.add_argument("--data-dir", type=str, default="data/satellite/mae")
    parser.add_argument("--output-dir", type=str, default="outputs/satellite")
    parser.add_argument("--config", type=str, default="", help="Path to YAML config override")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)

    # Common
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--val-fraction", type=float, default=None)

    # Phase 1 specific
    parser.add_argument("--mask-ratio", type=float, default=0.75)
    parser.add_argument("--physics-loss-weight", type=float, default=0.1)

    # Phase 2 specific
    parser.add_argument("--pretrain-checkpoint", type=str, default="")
    parser.add_argument("--freeze-backbone-epochs", type=int, default=5)

    # Phase 3 specific
    parser.add_argument("--max-temporal-len", type=int, default=10)
    parser.add_argument("--min-temporal-len", type=int, default=5)
    parser.add_argument("--temporal-mask-ratio", type=float, default=0.2)
    parser.add_argument("--freeze-backbone", action="store_true", default=True)
    parser.add_argument("--no-freeze-backbone", dest="freeze_backbone", action="store_false")

    return parser


def _load_yaml_config(path: str) -> Dict[str, Any]:
    """Load YAML config and return training.satellite section."""
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cfg.get("training", {}).get("satellite", {})
    except Exception as e:
        logger.warning(f"Could not load config from {path}: {e}")
        return {}


def main() -> None:
    args = build_argparser().parse_args()

    # Load YAML overrides
    yaml_cfg: Dict[str, Any] = {}
    if args.config:
        yaml_cfg = _load_yaml_config(args.config)

    if args.phase == 1:
        mae_cfg = yaml_cfg.get("mae_pretraining", {})
        config = MAEPretrainConfig(
            lr=args.lr or mae_cfg.get("lr", 1.5e-4),
            batch_size=args.batch_size or mae_cfg.get("batch_size", 32),
            epochs=args.epochs or mae_cfg.get("epochs", 100),
            warmup_epochs=mae_cfg.get("warmup_epochs", 10),
            weight_decay=mae_cfg.get("weight_decay", 0.05),
            scheduler=mae_cfg.get("scheduler", "cosine"),
            output_dir=args.output_dir,
            device=args.device,
            seed=args.seed,
            use_wandb=not args.no_wandb,
            data_dir=args.data_dir,
            val_fraction=args.val_fraction or 0.1,
            mask_ratio=args.mask_ratio,
            physics_loss_weight=args.physics_loss_weight,
        )
        trainer = MAEPretrainTrainer(config)
        trainer.setup()
        trainer.train()

    elif args.phase == 2:
        ft_cfg = yaml_cfg.get("finetuning", {})
        config = SupervisedFinetuneConfig(
            lr=args.lr or ft_cfg.get("lr", 1e-4),
            batch_size=args.batch_size or ft_cfg.get("batch_size", 32),
            epochs=args.epochs or ft_cfg.get("epochs", 50),
            weight_decay=ft_cfg.get("weight_decay", 0.05),
            scheduler=ft_cfg.get("scheduler", "cosine"),
            output_dir=args.output_dir,
            device=args.device,
            seed=args.seed,
            use_wandb=not args.no_wandb,
            data_dir=args.data_dir,
            val_fraction=args.val_fraction or 0.2,
            pretrain_checkpoint=args.pretrain_checkpoint,
            freeze_backbone_epochs=args.freeze_backbone_epochs,
        )
        trainer = SupervisedFinetuneTrainer(config)
        trainer.setup()
        trainer.train()

    elif args.phase == 3:
        config = TemporalStackConfig(
            lr=args.lr or 5e-5,
            batch_size=args.batch_size or 8,
            epochs=args.epochs or 50,
            output_dir=args.output_dir,
            device=args.device,
            seed=args.seed,
            use_wandb=not args.no_wandb,
            data_dir=args.data_dir,
            val_fraction=args.val_fraction or 0.2,
            max_temporal_len=args.max_temporal_len,
            min_temporal_len=args.min_temporal_len,
            temporal_mask_ratio=args.temporal_mask_ratio,
            pretrain_checkpoint=args.pretrain_checkpoint,
            freeze_backbone=args.freeze_backbone,
        )
        trainer = TemporalStackTrainer(config)
        trainer.setup()
        trainer.train()


if __name__ == "__main__":
    main()
