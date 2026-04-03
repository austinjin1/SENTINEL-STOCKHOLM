"""
Satellite encoder training pipeline for SENTINEL.

Three-phase training:
  Phase 1: Load SSL4EO-S12 ViT-S pretrained weights (adapt to 10-band input).
  Phase 2: Segmentation fine-tuning on DrivenData Tick Tick Bloom + MARIDA.
  Phase 3: Temporal transformer training on [CLS] embedding sequences.

Usage:
    python -m sentinel.training.train_satellite --phase 2 --data-dir data/satellite
    python -m sentinel.training.train_satellite --phase 3 --cls-embeddings-dir outputs/satellite/cls_embeddings
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from sentinel.models.satellite_encoder.backbone import (
    SatelliteViTBackbone,
    VIT_EMBED_DIM,
    NUM_SPECTRAL_BANDS,
)
from sentinel.models.satellite_encoder.segmentation import (
    ANOMALY_CLASSES,
    NUM_ANOMALY_CLASSES,
    SegmentationLoss,
    UPerNetHead,
)
from sentinel.models.satellite_encoder.temporal import (
    BUFFER_SIZE,
    EMBED_DIM,
    TemporalChangeDetector,
)
from sentinel.training.trainer import BaseTrainer, TrainerConfig, build_scheduler
from sentinel.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data augmentation
# ---------------------------------------------------------------------------

class SatelliteAugmentation:
    """Augmentation pipeline for satellite imagery training.

    Includes random crop, horizontal/vertical flip, spectral jitter,
    and simulated cloud masking.
    """

    def __init__(
        self,
        crop_size: int = 224,
        flip_prob: float = 0.5,
        spectral_jitter: float = 0.05,
        cloud_mask_prob: float = 0.2,
        cloud_mask_size_range: Tuple[int, int] = (20, 80),
    ) -> None:
        self.crop_size = crop_size
        self.flip_prob = flip_prob
        self.spectral_jitter = spectral_jitter
        self.cloud_mask_prob = cloud_mask_prob
        self.cloud_mask_size_range = cloud_mask_size_range

    def __call__(
        self, image: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply augmentations to image (H, W, C) and optional mask (H, W)."""
        h, w = image.shape[:2]

        # Random crop
        if h > self.crop_size and w > self.crop_size:
            y = random.randint(0, h - self.crop_size)
            x = random.randint(0, w - self.crop_size)
            image = image[y : y + self.crop_size, x : x + self.crop_size]
            if mask is not None:
                mask = mask[y : y + self.crop_size, x : x + self.crop_size]

        # Horizontal flip
        if random.random() < self.flip_prob:
            image = np.flip(image, axis=1).copy()
            if mask is not None:
                mask = np.flip(mask, axis=1).copy()

        # Vertical flip
        if random.random() < self.flip_prob:
            image = np.flip(image, axis=0).copy()
            if mask is not None:
                mask = np.flip(mask, axis=0).copy()

        # Spectral jitter: per-band additive noise
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
            # Set cloud-masked region to high reflectance (white cloud)
            image[cy : cy + size, cx : cx + size, :] = np.random.uniform(
                0.8, 1.0, size=(min(size, ch - cy), min(size, cw - cx), image.shape[2])
            ).astype(np.float32)

        return image, mask


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class TickTickBloomDataset(Dataset):
    """DrivenData Tick Tick Bloom dataset.

    ~23,000 in-situ cyanobacteria measurements paired with Sentinel-2 imagery.
    Labels are severity categories mapped to anomaly classes.

    Expected directory structure:
        data_dir/
            tiles/          -- .npy files (H, W, 10) float32
            labels.json     -- {tile_id: {"severity": int, "cyanobacteria_density": float, ...}}
    """

    # Severity bins -> anomaly class index (algal_bloom or normal)
    SEVERITY_TO_CLASS = {0: 6, 1: 0, 2: 0, 3: 0, 4: 0}  # 0=normal, 1-4=algal_bloom

    def __init__(
        self,
        data_dir: str | Path,
        augmentation: Optional[SatelliteAugmentation] = None,
        target_size: int = 224,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.target_size = target_size
        self.augmentation = augmentation

        tiles_dir = self.data_dir / "tiles"
        labels_path = self.data_dir / "labels.json"

        self.tile_paths: List[Path] = sorted(tiles_dir.glob("*.npy"))
        self.labels: Dict[str, Dict] = {}
        if labels_path.exists():
            with open(labels_path, "r", encoding="utf-8") as f:
                self.labels = json.load(f)

        logger.info(f"TickTickBloom: {len(self.tile_paths)} tiles, {len(self.labels)} labels")

    def __len__(self) -> int:
        return len(self.tile_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tile_path = self.tile_paths[idx]
        tile = np.load(tile_path).astype(np.float32)

        # Normalize uint16 reflectance if needed
        if tile.max() > 2.0:
            tile = tile / 10000.0

        tile_id = tile_path.stem
        label_info = self.labels.get(tile_id, {"severity": 0})
        severity = label_info.get("severity", 0)
        class_idx = self.SEVERITY_TO_CLASS.get(severity, 6)

        # Create pixel-level mask (uniform class for the tile)
        h, w = tile.shape[:2]
        mask = np.full((h, w), class_idx, dtype=np.int64)

        if self.augmentation is not None:
            tile, mask = self.augmentation(tile, mask)

        # Resize to target
        if tile.shape[0] != self.target_size or tile.shape[1] != self.target_size:
            from scipy.ndimage import zoom as scipy_zoom
            factors = (self.target_size / tile.shape[0], self.target_size / tile.shape[1], 1.0)
            tile = scipy_zoom(tile, factors, order=1)
            mask_factors = (self.target_size / mask.shape[0], self.target_size / mask.shape[1])
            mask = scipy_zoom(mask.astype(np.float32), mask_factors, order=0).astype(np.int64)

        # (H, W, C) -> (C, H, W)
        tile_tensor = torch.from_numpy(tile.transpose(2, 0, 1))
        mask_tensor = torch.from_numpy(mask)

        return {
            "image": tile_tensor,
            "mask": mask_tensor,
            "class_label": torch.tensor(class_idx, dtype=torch.long),
            "severity": torch.tensor(severity, dtype=torch.long),
        }


class MARIDADataset(Dataset):
    """MARIDA marine debris and anomaly detection dataset.

    1,381 patches with pixel-level annotations covering marine debris,
    algal blooms, turbidity, and other classes.

    Expected directory structure:
        data_dir/
            patches/        -- .npy files (H, W, 10) float32
            masks/          -- .npy files (H, W) int64 with class indices
    """

    # MARIDA class mapping to SENTINEL anomaly classes
    MARIDA_TO_ANOMALY = {
        0: 6,   # Marine water -> normal
        1: 0,   # Dense algae -> algal_bloom
        2: 0,   # Sparse algae -> algal_bloom
        3: 1,   # Turbid water -> turbidity_plume
        4: 2,   # Marine debris -> oil_sheen (closest analogue)
        5: 5,   # Foam -> foam_surfactant
        6: 4,   # Dense sargassum -> discoloration
        7: 4,   # Sparse sargassum -> discoloration
        8: 6,   # Cloud -> normal
        9: 6,   # Ship -> normal
    }

    def __init__(
        self,
        data_dir: str | Path,
        augmentation: Optional[SatelliteAugmentation] = None,
        target_size: int = 224,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.target_size = target_size
        self.augmentation = augmentation

        self.patch_paths: List[Path] = sorted((self.data_dir / "patches").glob("*.npy"))
        logger.info(f"MARIDA: {len(self.patch_paths)} patches")

    def __len__(self) -> int:
        return len(self.patch_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        patch_path = self.patch_paths[idx]
        patch = np.load(patch_path).astype(np.float32)
        if patch.max() > 2.0:
            patch = patch / 10000.0

        mask_path = self.data_dir / "masks" / patch_path.name
        if mask_path.exists():
            mask = np.load(mask_path).astype(np.int64)
            # Remap MARIDA classes to SENTINEL anomaly classes
            remapped = np.vectorize(lambda x: self.MARIDA_TO_ANOMALY.get(x, 6))(mask)
        else:
            remapped = np.full(patch.shape[:2], 6, dtype=np.int64)

        if self.augmentation is not None:
            patch, remapped = self.augmentation(patch, remapped)

        if patch.shape[0] != self.target_size or patch.shape[1] != self.target_size:
            from scipy.ndimage import zoom as scipy_zoom
            factors = (self.target_size / patch.shape[0], self.target_size / patch.shape[1], 1.0)
            patch = scipy_zoom(patch, factors, order=1)
            mask_factors = (self.target_size / remapped.shape[0], self.target_size / remapped.shape[1])
            remapped = scipy_zoom(remapped.astype(np.float32), mask_factors, order=0).astype(np.int64)

        tile_tensor = torch.from_numpy(patch.transpose(2, 0, 1))
        mask_tensor = torch.from_numpy(remapped)

        # Determine dominant class
        unique, counts = np.unique(remapped, return_counts=True)
        dominant_class = unique[counts.argmax()]

        return {
            "image": tile_tensor,
            "mask": mask_tensor,
            "class_label": torch.tensor(int(dominant_class), dtype=torch.long),
        }


class MergedSegmentationDataset(Dataset):
    """Merges TickTickBloom and MARIDA datasets into one."""

    def __init__(self, datasets: Sequence[Dataset]) -> None:
        self.datasets = datasets
        self.lengths = [len(d) for d in datasets]
        self.cumulative = []
        total = 0
        for l in self.lengths:
            self.cumulative.append(total)
            total += l
        self._total = total

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        for i, cum in enumerate(self.cumulative):
            if idx < cum + self.lengths[i]:
                return self.datasets[i][idx - cum]
        raise IndexError(f"Index {idx} out of range")


class TemporalCLSDataset(Dataset):
    """Dataset of [CLS] embedding sequences for temporal transformer training.

    Each sample is a sequence of T [CLS] embeddings from consecutive
    acquisitions of the same tile, plus timestamps and an optional
    binary anomaly label for the last acquisition.

    Expected directory structure:
        data_dir/
            sequences/      -- .npz files with keys: embeddings (T, D), timestamps (T,)
            labels.json     -- {sequence_id: {"anomaly": 0|1, "next_score": float}}
    """

    def __init__(
        self,
        data_dir: str | Path,
        max_seq_len: int = BUFFER_SIZE,
        mask_ratio: float = 0.15,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.max_seq_len = max_seq_len
        self.mask_ratio = mask_ratio

        self.sequence_paths: List[Path] = sorted(
            (self.data_dir / "sequences").glob("*.npz")
        )
        labels_path = self.data_dir / "labels.json"
        self.labels: Dict[str, Dict] = {}
        if labels_path.exists():
            with open(labels_path, "r", encoding="utf-8") as f:
                self.labels = json.load(f)

        logger.info(f"TemporalCLS: {len(self.sequence_paths)} sequences")

    def __len__(self) -> int:
        return len(self.sequence_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path = self.sequence_paths[idx]
        data = np.load(path)
        embeddings = data["embeddings"].astype(np.float32)  # (T, D)
        timestamps = data["timestamps"].astype(np.float32)  # (T,)
        T, D = embeddings.shape

        # Pad or truncate to max_seq_len
        attention_mask = np.zeros(self.max_seq_len, dtype=bool)
        if T < self.max_seq_len:
            pad_len = self.max_seq_len - T
            embeddings = np.pad(embeddings, ((pad_len, 0), (0, 0)))
            timestamps = np.pad(timestamps, (pad_len, 0))
            attention_mask[:pad_len] = True
        elif T > self.max_seq_len:
            embeddings = embeddings[-self.max_seq_len:]
            timestamps = timestamps[-self.max_seq_len:]

        # Masked temporal prediction: randomly mask some positions
        mask_pred = np.zeros(self.max_seq_len, dtype=bool)
        valid_positions = np.where(~attention_mask)[0]
        if len(valid_positions) > 1:
            n_mask = max(1, int(len(valid_positions) * self.mask_ratio))
            mask_indices = np.random.choice(
                valid_positions[:-1], size=min(n_mask, len(valid_positions) - 1), replace=False
            )
            mask_pred[mask_indices] = True

        # Labels
        seq_id = path.stem
        label_info = self.labels.get(seq_id, {"anomaly": 0})
        anomaly = label_info.get("anomaly", 0)

        return {
            "embeddings": torch.from_numpy(embeddings),
            "timestamps": torch.from_numpy(timestamps),
            "attention_mask": torch.from_numpy(attention_mask),
            "mask_pred": torch.from_numpy(mask_pred),
            "anomaly_label": torch.tensor(anomaly, dtype=torch.float32),
        }


# ---------------------------------------------------------------------------
# Full segmentation model
# ---------------------------------------------------------------------------

class SatelliteSegmentationModel(nn.Module):
    """ViT backbone + UPerNet segmentation head."""

    def __init__(
        self,
        pretrained: bool = True,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.backbone = SatelliteViTBackbone(
            pretrained=pretrained, checkpoint_path=checkpoint_path
        )
        self.seg_head = UPerNetHead(in_channels=VIT_EMBED_DIM)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        cls_token, multi_scale = self.backbone(x)
        spatial = self.backbone.get_spatial_features(multi_scale)
        seg_out = self.seg_head(spatial)
        seg_out["cls_token"] = cls_token
        return seg_out


# ---------------------------------------------------------------------------
# Phase 2: Segmentation trainer
# ---------------------------------------------------------------------------

@dataclass
class SegmentationConfig(TrainerConfig):
    lr: float = 1e-4
    batch_size: int = 32
    epochs: int = 50
    scheduler: str = "cosine"
    wandb_run_name: str = "satellite-segmentation"

    # Data
    ttb_data_dir: str = "data/satellite/tick_tick_bloom"
    marida_data_dir: str = "data/satellite/marida"
    val_fraction: float = 0.2
    target_size: int = 224

    # Augmentation
    spectral_jitter: float = 0.05
    cloud_mask_prob: float = 0.2

    # Model
    pretrained: bool = True
    ssl4eo_checkpoint: str = ""

    # Loss
    dice_weight: float = 1.0
    ce_weight: float = 1.0


class SegmentationTrainer(BaseTrainer):
    """Phase 2: Fine-tune segmentation on merged DrivenData + MARIDA."""

    def __init__(self, config: SegmentationConfig) -> None:
        super().__init__(config)
        self.seg_config = config
        self.criterion: Optional[SegmentationLoss] = None

    def build_model(self) -> nn.Module:
        ckpt = self.seg_config.ssl4eo_checkpoint or None
        model = SatelliteSegmentationModel(
            pretrained=self.seg_config.pretrained,
            checkpoint_path=ckpt,
        )
        self.criterion = SegmentationLoss(
            dice_weight=self.seg_config.dice_weight,
            ce_weight=self.seg_config.ce_weight,
        ).to(self.device)
        return model

    def build_datasets(self) -> Tuple[Dataset, Dataset]:
        augmentation = SatelliteAugmentation(
            crop_size=self.seg_config.target_size,
            spectral_jitter=self.seg_config.spectral_jitter,
            cloud_mask_prob=self.seg_config.cloud_mask_prob,
        )

        datasets: List[Dataset] = []

        ttb_dir = Path(self.seg_config.ttb_data_dir)
        if ttb_dir.exists():
            datasets.append(TickTickBloomDataset(ttb_dir, augmentation, self.seg_config.target_size))

        marida_dir = Path(self.seg_config.marida_data_dir)
        if marida_dir.exists():
            datasets.append(MARIDADataset(marida_dir, augmentation, self.seg_config.target_size))

        if not datasets:
            raise FileNotFoundError(
                f"No training data found at {ttb_dir} or {marida_dir}"
            )

        merged = MergedSegmentationDataset(datasets) if len(datasets) > 1 else datasets[0]

        # Stratified split by class label
        n = len(merged)
        indices = list(range(n))
        random.shuffle(indices)

        # Attempt stratified split
        class_indices: Dict[int, List[int]] = {}
        for i in indices:
            try:
                sample = merged[i]
                cls = sample["class_label"].item()
            except Exception:
                cls = 6  # default to normal
            class_indices.setdefault(cls, []).append(i)

        train_indices: List[int] = []
        val_indices: List[int] = []
        for cls, idx_list in class_indices.items():
            split = int(len(idx_list) * (1 - self.seg_config.val_fraction))
            train_indices.extend(idx_list[:split])
            val_indices.extend(idx_list[split:])

        random.shuffle(train_indices)
        random.shuffle(val_indices)

        train_ds = Subset(merged, train_indices)
        val_ds = Subset(merged, val_indices)

        logger.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
        return train_ds, val_ds

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        assert self.model is not None and self.optimizer is not None
        images = batch["image"]   # (B, C, H, W)
        masks = batch["mask"]     # (B, H, W)

        outputs = self.model(images)
        class_logits = outputs["class_logits"]  # (B, num_classes, h, w)

        # Resize mask to match output spatial size
        h_out, w_out = class_logits.shape[2:]
        if masks.shape[1] != h_out or masks.shape[2] != w_out:
            masks = F.interpolate(
                masks.unsqueeze(1).float(),
                size=(h_out, w_out),
                mode="nearest",
            ).squeeze(1).long()

        loss_dict = self.criterion(class_logits, masks)
        loss_dict["loss"].backward()

        if self.config.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        return {k: v.item() for k, v in loss_dict.items()}

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        assert self.model is not None
        self.model.eval()

        total_loss = 0.0
        total_ce = 0.0
        total_dice = 0.0
        correct_pixels = 0
        total_pixels = 0
        n_batches = 0

        for batch in dataloader:
            batch = self._to_device(batch)
            images = batch["image"]
            masks = batch["mask"]

            outputs = self.model(images)
            class_logits = outputs["class_logits"]

            h_out, w_out = class_logits.shape[2:]
            if masks.shape[1] != h_out or masks.shape[2] != w_out:
                masks = F.interpolate(
                    masks.unsqueeze(1).float(),
                    size=(h_out, w_out),
                    mode="nearest",
                ).squeeze(1).long()

            loss_dict = self.criterion(class_logits, masks)
            total_loss += loss_dict["total"].item()
            total_ce += loss_dict["ce"].item()
            total_dice += loss_dict["dice"].item()

            preds = class_logits.argmax(dim=1)
            correct_pixels += (preds == masks).sum().item()
            total_pixels += masks.numel()
            n_batches += 1

        self.model.train()

        n = max(n_batches, 1)
        return {
            "loss": total_loss / n,
            "ce": total_ce / n,
            "dice": total_dice / n,
            "pixel_acc": correct_pixels / max(total_pixels, 1),
        }


# ---------------------------------------------------------------------------
# Phase 3: Temporal trainer
# ---------------------------------------------------------------------------

@dataclass
class TemporalConfig(TrainerConfig):
    lr: float = 1e-4
    batch_size: int = 64
    epochs: int = 50
    scheduler: str = "cosine"
    wandb_run_name: str = "satellite-temporal"

    # Data
    cls_embeddings_dir: str = "outputs/satellite/cls_embeddings"
    val_fraction: float = 0.2
    max_seq_len: int = BUFFER_SIZE
    mask_ratio: float = 0.15

    # Loss weights
    ssl_weight: float = 1.0
    supervised_weight: float = 0.5


class TemporalTrainer(BaseTrainer):
    """Phase 3: Train temporal transformer on [CLS] embedding sequences."""

    def __init__(self, config: TemporalConfig) -> None:
        super().__init__(config)
        self.temp_config = config

    def build_model(self) -> nn.Module:
        model = TemporalChangeDetector(
            embed_dim=EMBED_DIM,
            buffer_size=self.temp_config.max_seq_len,
        )
        # Add a prediction head for masked temporal prediction
        model.mask_pred_head = nn.Linear(EMBED_DIM, EMBED_DIM)
        # Add a classification head for binary anomaly
        model.anomaly_cls_head = nn.Sequential(
            nn.Linear(EMBED_DIM, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        return model

    def build_datasets(self) -> Tuple[Dataset, Dataset]:
        data_dir = Path(self.temp_config.cls_embeddings_dir)
        full_ds = TemporalCLSDataset(
            data_dir,
            max_seq_len=self.temp_config.max_seq_len,
            mask_ratio=self.temp_config.mask_ratio,
        )

        n = len(full_ds)
        n_val = int(n * self.temp_config.val_fraction)
        indices = list(range(n))
        random.shuffle(indices)
        train_ds = Subset(full_ds, indices[n_val:])
        val_ds = Subset(full_ds, indices[:n_val])

        logger.info(f"Temporal train: {len(train_ds)}, val: {len(val_ds)}")
        return train_ds, val_ds

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        assert self.model is not None and self.optimizer is not None
        embeddings = batch["embeddings"]       # (B, T, D)
        timestamps = batch["timestamps"]       # (B, T)
        attention_mask = batch["attention_mask"] # (B, T)
        mask_pred = batch["mask_pred"]          # (B, T)
        anomaly_label = batch["anomaly_label"]  # (B,)

        # Store original embeddings for SSL target
        original_embeddings = embeddings.clone()

        # Zero out masked positions
        mask_expanded = mask_pred.unsqueeze(-1).float()
        embeddings = embeddings * (1.0 - mask_expanded)

        # Forward through temporal transformer
        out = self.model(embeddings, timestamps, attention_mask)

        losses = {}

        # Self-supervised: predict masked embeddings
        if mask_pred.any():
            # Get transformer output at masked positions for prediction
            # Re-run forward to get intermediate representations
            temp_enc = self.model.temporal_encoding(timestamps)
            x = self.model.input_norm(embeddings) + temp_enc
            x = self.model.transformer(x, src_key_padding_mask=attention_mask)

            pred_embeddings = self.model.mask_pred_head(x)
            mask_loss = F.mse_loss(
                pred_embeddings[mask_pred],
                original_embeddings[mask_pred],
            )
            losses["ssl_loss"] = mask_loss * self.temp_config.ssl_weight

        # Supervised: binary anomaly classification
        anomaly_logit = self.model.anomaly_cls_head(out["temporal_embedding"]).squeeze(-1)
        bce_loss = F.binary_cross_entropy_with_logits(anomaly_logit, anomaly_label)
        losses["anomaly_loss"] = bce_loss * self.temp_config.supervised_weight

        total = sum(losses.values())
        losses["loss"] = total
        total.backward()

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
        total_bce = 0.0
        correct = 0
        total = 0

        for batch in dataloader:
            batch = self._to_device(batch)
            embeddings = batch["embeddings"]
            timestamps = batch["timestamps"]
            attention_mask = batch["attention_mask"]
            anomaly_label = batch["anomaly_label"]

            out = self.model(embeddings, timestamps, attention_mask)
            anomaly_logit = self.model.anomaly_cls_head(
                out["temporal_embedding"]
            ).squeeze(-1)
            bce = F.binary_cross_entropy_with_logits(anomaly_logit, anomaly_label)
            total_bce += bce.item()

            preds = (torch.sigmoid(anomaly_logit) > 0.5).float()
            correct += (preds == anomaly_label).sum().item()
            total += anomaly_label.size(0)
            total_loss += bce.item()

        self.model.train()
        n = max(total, 1)
        n_batches = max(1, total // max(self.config.batch_size, 1))
        return {
            "loss": total_loss / max(n_batches, 1),
            "anomaly_accuracy": correct / n,
        }


# ---------------------------------------------------------------------------
# Phase 1: SSL4EO weight loading (no training)
# ---------------------------------------------------------------------------

def run_phase1(checkpoint_path: Optional[str] = None, output_dir: str = "outputs/satellite") -> Path:
    """Phase 1: Load and adapt SSL4EO-S12 pretrained weights.

    No training needed -- just loads ViT-S weights with 10-band adaptation
    and saves the adapted model.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("Phase 1: Loading SSL4EO-S12 ViT-S pretrained weights")
    model = SatelliteViTBackbone(
        pretrained=True,
        checkpoint_path=checkpoint_path,
    )

    # Verify forward pass
    dummy = torch.randn(1, NUM_SPECTRAL_BANDS, 224, 224)
    with torch.no_grad():
        cls_token, features = model(dummy)

    logger.info(f"CLS token shape: {cls_token.shape}")
    logger.info(f"Feature layers: {list(features.keys())}")
    for k, v in features.items():
        logger.info(f"  Layer {k}: {v.shape}")

    # Save adapted backbone
    save_path = out / "ssl4eo_adapted_backbone.pt"
    torch.save(model.state_dict(), save_path)
    logger.info(f"Adapted backbone saved to {save_path}")

    return save_path


def extract_cls_embeddings(
    model_path: str | Path,
    tiles_dir: str | Path,
    output_dir: str | Path,
    batch_size: int = 32,
    device: str = "auto",
) -> Path:
    """Extract [CLS] embeddings from all tiles for Phase 3.

    Processes all .npy tile files and saves embedding sequences
    grouped by tile ID.
    """
    tiles_dir = Path(tiles_dir)
    output_dir = Path(output_dir)
    (output_dir / "sequences").mkdir(parents=True, exist_ok=True)

    dev = torch.device(
        "cuda" if device == "auto" and torch.cuda.is_available() else
        device if device != "auto" else "cpu"
    )

    model = SatelliteViTBackbone(pretrained=False)
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=False)
    model.to(dev)
    model.eval()

    tile_files = sorted(tiles_dir.glob("*.npy"))
    logger.info(f"Extracting CLS embeddings from {len(tile_files)} tiles")

    # Group files by tile ID (assume format: tileID_date.npy)
    from collections import defaultdict
    tile_groups: Dict[str, List[Tuple[str, Path]]] = defaultdict(list)
    for tf in tile_files:
        parts = tf.stem.rsplit("_", 1)
        tile_id = parts[0] if len(parts) > 1 else tf.stem
        date_str = parts[1] if len(parts) > 1 else "0"
        tile_groups[tile_id].append((date_str, tf))

    # Sort each group by date
    for tid in tile_groups:
        tile_groups[tid].sort(key=lambda x: x[0])

    from sentinel.utils.logging import make_progress
    progress = make_progress()
    with progress:
        task = progress.add_task("Extracting CLS embeddings", total=len(tile_groups))
        for tile_id, group in tile_groups.items():
            embeddings_list = []
            timestamps_list = []

            for date_str, path in group:
                tile = np.load(path).astype(np.float32)
                if tile.max() > 2.0:
                    tile = tile / 10000.0
                if tile.shape[0] != 224 or tile.shape[1] != 224:
                    from scipy.ndimage import zoom as scipy_zoom
                    factors = (224 / tile.shape[0], 224 / tile.shape[1], 1.0)
                    tile = scipy_zoom(tile, factors, order=1)

                tensor = torch.from_numpy(tile.transpose(2, 0, 1)).unsqueeze(0).to(dev)
                with torch.no_grad():
                    cls_token, _ = model(tensor)
                embeddings_list.append(cls_token.cpu().numpy().squeeze(0))

                # Convert date to days-since-epoch
                try:
                    from datetime import datetime
                    dt = datetime.strptime(date_str, "%Y%m%d")
                    days = (dt - datetime(2000, 1, 1)).days
                except Exception:
                    days = len(timestamps_list)
                timestamps_list.append(float(days))

            if embeddings_list:
                np.savez(
                    output_dir / "sequences" / f"{tile_id}.npz",
                    embeddings=np.stack(embeddings_list),
                    timestamps=np.array(timestamps_list),
                )
            progress.advance(task)

    logger.info(f"Saved {len(tile_groups)} embedding sequences to {output_dir}")
    return output_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SENTINEL satellite encoder training pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2, 3],
                        help="Training phase (1=load weights, 2=segmentation, 3=temporal)")
    parser.add_argument("--output-dir", type=str, default="outputs/satellite")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)

    # Phase 1
    parser.add_argument("--ssl4eo-checkpoint", type=str, default="",
                        help="Path to SSL4EO-S12 checkpoint (Phase 1)")

    # Phase 2
    parser.add_argument("--ttb-data-dir", type=str, default="data/satellite/tick_tick_bloom")
    parser.add_argument("--marida-data-dir", type=str, default="data/satellite/marida")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--no-wandb", action="store_true")

    # Phase 3
    parser.add_argument("--cls-embeddings-dir", type=str,
                        default="outputs/satellite/cls_embeddings")
    parser.add_argument("--tiles-dir", type=str, default="data/satellite/processed")
    parser.add_argument("--backbone-path", type=str, default="")
    parser.add_argument("--mask-ratio", type=float, default=0.15)

    # Config overrides
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        help="Override config values: key=value")
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    if args.phase == 1:
        run_phase1(
            checkpoint_path=args.ssl4eo_checkpoint or None,
            output_dir=args.output_dir,
        )

    elif args.phase == 2:
        config = SegmentationConfig(
            lr=args.lr,
            batch_size=args.batch_size,
            epochs=args.epochs,
            output_dir=args.output_dir,
            device=args.device,
            seed=args.seed,
            use_wandb=not args.no_wandb,
            ttb_data_dir=args.ttb_data_dir,
            marida_data_dir=args.marida_data_dir,
            val_fraction=args.val_fraction,
            ssl4eo_checkpoint=args.ssl4eo_checkpoint,
        )
        trainer = SegmentationTrainer(config)
        trainer.setup()
        trainer.train()

    elif args.phase == 3:
        # Optionally extract CLS embeddings first
        emb_dir = Path(args.cls_embeddings_dir)
        if not (emb_dir / "sequences").exists() and args.backbone_path:
            logger.info("Extracting CLS embeddings before temporal training...")
            extract_cls_embeddings(
                model_path=args.backbone_path,
                tiles_dir=args.tiles_dir,
                output_dir=str(emb_dir),
                device=args.device,
            )

        config = TemporalConfig(
            lr=args.lr,
            batch_size=args.batch_size,
            epochs=args.epochs,
            output_dir=args.output_dir,
            device=args.device,
            seed=args.seed,
            use_wandb=not args.no_wandb,
            cls_embeddings_dir=args.cls_embeddings_dir,
            mask_ratio=args.mask_ratio,
        )
        trainer = TemporalTrainer(config)
        trainer.setup()
        trainer.train()


if __name__ == "__main__":
    main()
