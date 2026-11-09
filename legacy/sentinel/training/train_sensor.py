"""
AquaSSM sensor encoder training pipeline for SENTINEL.

Three-phase training:
  Phase 1: Self-supervised pretraining via Masked Parameter Prediction (MPP)
            on all USGS NWIS stations. Irregular time series input.
  Phase 2: Supervised anomaly fine-tuning with binary anomaly classification
            and contrastive loss, cross-referencing USGS/EPA data.
  Phase 3: Sensor health classifier training on simulated sensor failures
            (drift, fouling, failure) injected into clean data.

Usage:
    python -m sentinel.training.train_sensor --phase 1 --data-dir data/sensor/pretrain
    python -m sentinel.training.train_sensor --phase 2 --data-dir data/sensor/anomaly
    python -m sentinel.training.train_sensor --phase 3 --data-dir data/sensor/pretrain
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

from sentinel.models.sensor_encoder.model import SensorEncoder
from sentinel.models.sensor_encoder.aqua_ssm import NUM_PARAMETERS, OUTPUT_DIM
from sentinel.models.sensor_encoder.sensor_health import (
    HEALTH_CLASSES,
    NUM_HEALTH_CLASSES,
    SensorHealthSentinel,
)
from sentinel.models.sensor_encoder.physics_constraints import PhysicsConstraintLoss
from sentinel.training.trainer import BaseTrainer, TrainerConfig, build_scheduler
from sentinel.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class MPPPretrainDataset(Dataset):
    """Dataset for self-supervised MPP pretraining from preprocessed .npz files.

    Each .npz file (one per USGS NWIS station) contains:
        'values':   (T, P) float32 — sensor readings (P=6 parameters)
        'delta_ts': (T,) float32   — time gaps in seconds between observations
        'masks':    (T, P) float32 — per-parameter validity mask (1=valid, 0=missing)
        'station_id': str          — USGS station identifier

    Samples are drawn as sliding windows of fixed length from each station's
    time series, supporting the irregular time intervals from IrregularTimeSample
    format.
    """

    def __init__(
        self,
        data_dir: str | Path,
        window_length: int = 672,
        stride: int = 168,
        num_params: int = NUM_PARAMETERS,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.window_length = window_length
        self.stride = stride
        self.num_params = num_params

        # Index all windows across all station files
        self.windows: List[Tuple[Path, int]] = []  # (npz_path, start_idx)
        npz_files = sorted(self.data_dir.glob("*.npz"))

        for npz_path in npz_files:
            try:
                with np.load(npz_path) as data:
                    T = data["values"].shape[0]
            except Exception:
                continue
            if T < window_length:
                # Use the whole series padded later
                self.windows.append((npz_path, 0))
            else:
                for start in range(0, T - window_length + 1, stride):
                    self.windows.append((npz_path, start))

        logger.info(
            f"MPPPretrainDataset: {len(npz_files)} stations, "
            f"{len(self.windows)} windows (len={window_length}, stride={stride})"
        )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        npz_path, start = self.windows[idx]
        data = np.load(npz_path)

        values = data["values"].astype(np.float32)  # (T, P)
        delta_ts = data["delta_ts"].astype(np.float32)  # (T,)
        masks = data["masks"].astype(np.float32) if "masks" in data else np.ones_like(values)

        T = values.shape[0]
        end = min(start + self.window_length, T)
        values_w = values[start:end]
        delta_ts_w = delta_ts[start:end]
        masks_w = masks[start:end]

        # Pad if shorter than window_length
        actual_len = values_w.shape[0]
        if actual_len < self.window_length:
            pad_len = self.window_length - actual_len
            values_w = np.pad(values_w, ((0, pad_len), (0, 0)), mode="constant")
            delta_ts_w = np.pad(delta_ts_w, (0, pad_len), constant_values=900.0)
            masks_w = np.pad(masks_w, ((0, pad_len), (0, 0)), mode="constant")

        return {
            "values": torch.from_numpy(values_w),
            "delta_ts": torch.from_numpy(delta_ts_w),
            "masks": torch.from_numpy(masks_w),
        }


class AnomalyFinetuneDataset(Dataset):
    """Dataset for supervised anomaly fine-tuning.

    Cross-references USGS station data with EPA contamination event records.

    Expected directory structure:
        data_dir/
            samples/    -- .npz files with keys: values (T, P), delta_ts (T,),
                           masks (T, P), label (int: 0=normal, 1=anomaly)
            labels.json -- {sample_id: {"label": 0|1, "event_type": str, ...}}
    """

    def __init__(
        self,
        data_dir: str | Path,
        window_length: int = 672,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.window_length = window_length

        samples_dir = self.data_dir / "samples"
        self.sample_paths: List[Path] = sorted(samples_dir.glob("*.npz"))

        labels_path = self.data_dir / "labels.json"
        self.labels: Dict[str, Dict] = {}
        if labels_path.exists():
            with open(labels_path, "r", encoding="utf-8") as f:
                self.labels = json.load(f)

        logger.info(
            f"AnomalyFinetuneDataset: {len(self.sample_paths)} samples, "
            f"{len(self.labels)} labels"
        )

    def __len__(self) -> int:
        return len(self.sample_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path = self.sample_paths[idx]
        data = np.load(path)

        values = data["values"].astype(np.float32)
        delta_ts = data["delta_ts"].astype(np.float32)
        masks = data["masks"].astype(np.float32) if "masks" in data else np.ones_like(values)

        # Crop or pad to window_length
        T = values.shape[0]
        if T > self.window_length:
            start = random.randint(0, T - self.window_length)
            values = values[start:start + self.window_length]
            delta_ts = delta_ts[start:start + self.window_length]
            masks = masks[start:start + self.window_length]
        elif T < self.window_length:
            pad = self.window_length - T
            values = np.pad(values, ((0, pad), (0, 0)), mode="constant")
            delta_ts = np.pad(delta_ts, (0, pad), constant_values=900.0)
            masks = np.pad(masks, ((0, pad), (0, 0)), mode="constant")

        sample_id = path.stem
        label_info = self.labels.get(sample_id, {})
        label = label_info.get("label", 0)

        return {
            "values": torch.from_numpy(values),
            "delta_ts": torch.from_numpy(delta_ts),
            "masks": torch.from_numpy(masks),
            "label": torch.tensor(label, dtype=torch.float32),
        }


class SensorHealthDataset(Dataset):
    """Dataset for sensor health classifier training via simulated failures.

    Loads clean station data and injects simulated sensor failures:
        - drift:   linear trend added to one parameter
        - fouling: multiplicative noise increase on one parameter
        - failure: constant output on one parameter

    Labels: 0=normal, 1=drift, 2=fouling, 3=failure, 4=calibration_needed
    """

    FAULT_TYPES = ["normal", "drift", "fouling", "failure", "calibration_needed"]

    def __init__(
        self,
        data_dir: str | Path,
        window_length: int = 672,
        stride: int = 168,
        fault_probability: float = 0.8,
        num_params: int = NUM_PARAMETERS,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.window_length = window_length
        self.num_params = num_params
        self.fault_probability = fault_probability

        # Build window index from clean station data
        self.windows: List[Tuple[Path, int]] = []
        npz_files = sorted(self.data_dir.glob("*.npz"))

        for npz_path in npz_files:
            try:
                with np.load(npz_path) as data:
                    T = data["values"].shape[0]
            except Exception:
                continue
            if T < window_length:
                self.windows.append((npz_path, 0))
            else:
                for start in range(0, T - window_length + 1, stride):
                    self.windows.append((npz_path, start))

        logger.info(
            f"SensorHealthDataset: {len(npz_files)} stations, "
            f"{len(self.windows)} windows"
        )

    def __len__(self) -> int:
        return len(self.windows)

    def _inject_fault(
        self,
        values: np.ndarray,
        fault_type: int,
        param_idx: int,
    ) -> np.ndarray:
        """Inject a simulated sensor fault into one parameter.

        Args:
            values: Clean sensor readings (T, P).
            fault_type: 1=drift, 2=fouling, 3=failure, 4=calibration_needed.
            param_idx: Parameter index to corrupt.

        Returns:
            Corrupted values (T, P).
        """
        T = values.shape[0]
        corrupted = values.copy()
        param_std = max(np.std(values[:, param_idx]), 1e-6)

        if fault_type == 1:  # drift: linear trend
            drift_magnitude = np.random.uniform(0.5, 3.0) * param_std
            trend = np.linspace(0, drift_magnitude, T)
            if random.random() < 0.5:
                trend = -trend
            corrupted[:, param_idx] += trend

        elif fault_type == 2:  # fouling: increasing noise
            base_noise = np.random.uniform(1.0, 5.0) * param_std
            noise_scale = np.linspace(0.1, 1.0, T)
            noise = np.random.normal(0, base_noise, T) * noise_scale
            corrupted[:, param_idx] += noise

        elif fault_type == 3:  # failure: constant output
            onset = random.randint(0, max(T // 2, 1))
            constant_value = values[onset, param_idx]
            corrupted[onset:, param_idx] = constant_value

        elif fault_type == 4:  # calibration_needed: offset + slight drift
            offset = np.random.uniform(1.0, 3.0) * param_std
            if random.random() < 0.5:
                offset = -offset
            drift = np.linspace(0, 0.5 * param_std, T)
            corrupted[:, param_idx] += offset + drift

        return corrupted

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        npz_path, start = self.windows[idx]
        data = np.load(npz_path)

        values = data["values"].astype(np.float32)
        delta_ts = data["delta_ts"].astype(np.float32)

        T = values.shape[0]
        end = min(start + self.window_length, T)
        values_w = values[start:end]
        delta_ts_w = delta_ts[start:end]

        # Pad if needed
        actual_len = values_w.shape[0]
        if actual_len < self.window_length:
            pad = self.window_length - actual_len
            values_w = np.pad(values_w, ((0, pad), (0, 0)), mode="constant")
            delta_ts_w = np.pad(delta_ts_w, (0, pad), constant_values=900.0)

        # Per-sensor labels: [P] with class index per sensor
        health_labels = np.zeros(self.num_params, dtype=np.int64)

        # Decide whether to inject a fault
        if random.random() < self.fault_probability:
            # Pick 1-2 sensors to corrupt
            n_corrupt = random.choices([1, 2], weights=[0.7, 0.3])[0]
            corrupt_params = random.sample(range(self.num_params), n_corrupt)
            for param_idx in corrupt_params:
                fault_type = random.randint(1, 4)
                values_w = self._inject_fault(values_w, fault_type, param_idx)
                health_labels[param_idx] = fault_type

        return {
            "values": torch.from_numpy(values_w),
            "delta_ts": torch.from_numpy(delta_ts_w),
            "health_labels": torch.from_numpy(health_labels),
        }


# ---------------------------------------------------------------------------
# Phase 1: MPP Pretraining Trainer
# ---------------------------------------------------------------------------

@dataclass
class MPPPretrainConfig(TrainerConfig):
    """Configuration for Phase 1: self-supervised MPP pretraining."""

    lr: float = 5e-4
    batch_size: int = 256
    epochs: int = 100
    warmup_steps: int = 5000
    scheduler: str = "cosine"
    weight_decay: float = 0.01
    wandb_run_name: str = "sensor-mpp-pretrain"

    # Data
    data_dir: str = "data/sensor/pretrain"
    window_length: int = 672
    stride: int = 168
    val_fraction: float = 0.2
    spatial_split: bool = True  # hold out 20% of stations

    # MPP
    mask_ratio_min: float = 0.25
    mask_ratio_max: float = 0.75


class MPPPretrainTrainer(BaseTrainer):
    """Phase 1: Self-supervised pretraining via Masked Parameter Prediction."""

    def __init__(self, config: MPPPretrainConfig) -> None:
        super().__init__(config)
        self.mpp_config = config
        self.physics_loss = PhysicsConstraintLoss()
        self.physics_weight = 0.1  # weight for physics constraint regularization

    def build_model(self) -> nn.Module:
        model = SensorEncoder(
            num_params=NUM_PARAMETERS,
            output_dim=OUTPUT_DIM,
        )
        return model

    def build_datasets(self) -> Tuple[Dataset, Dataset]:
        data_dir = Path(self.mpp_config.data_dir)

        if self.mpp_config.spatial_split:
            # Spatial split: hold out 20% of stations entirely
            all_npz = sorted(data_dir.glob("*.npz"))
            random.shuffle(all_npz)
            n_val = max(1, int(len(all_npz) * self.mpp_config.val_fraction))
            val_stations = set(str(p) for p in all_npz[:n_val])
            train_stations = set(str(p) for p in all_npz[n_val:])

            # Build separate datasets for train/val station subsets
            train_ds = MPPPretrainDataset(
                data_dir,
                window_length=self.mpp_config.window_length,
                stride=self.mpp_config.stride,
            )
            # Filter windows by station
            train_windows = [
                w for w in train_ds.windows if str(w[0]) in train_stations
            ]
            val_windows = [
                w for w in train_ds.windows if str(w[0]) in val_stations
            ]
            train_ds.windows = train_windows
            val_ds = MPPPretrainDataset(
                data_dir,
                window_length=self.mpp_config.window_length,
                stride=self.mpp_config.stride,
            )
            val_ds.windows = val_windows
        else:
            full_ds = MPPPretrainDataset(
                data_dir,
                window_length=self.mpp_config.window_length,
                stride=self.mpp_config.stride,
            )
            n = len(full_ds)
            n_val = int(n * self.mpp_config.val_fraction)
            indices = list(range(n))
            random.shuffle(indices)
            train_ds = Subset(full_ds, indices[n_val:])
            val_ds = Subset(full_ds, indices[:n_val])

        logger.info(f"MPP Pretrain — Train: {len(train_ds)}, Val: {len(val_ds)}")
        return train_ds, val_ds

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        assert self.model is not None and self.optimizer is not None

        values = batch["values"]      # (B, T, P)
        delta_ts = batch["delta_ts"]  # (B, T)

        # Forward through MPP pretraining
        mpp_output = self.model.forward_pretrain(
            x=values,
            delta_ts=delta_ts,
        )

        mpp_loss = mpp_output["loss"]

        # Physics constraint regularization on reconstructed values
        pred_tensor = mpp_output["predictions"]  # (B, T, P)
        # Convert tensor to named dict for physics constraints
        _param_names = ["do", "ph", "conductivity", "temperature", "turb", "orp"]
        pred_dict = {name: pred_tensor[..., i] for i, name in enumerate(_param_names)}
        physics_out = self.physics_loss(pred_dict)
        physics_loss = physics_out["total_loss"]

        loss = mpp_loss + self.physics_weight * physics_loss

        loss.backward()

        if self.config.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        # Update running error statistics for anomaly normalization
        with torch.no_grad():
            self.model.anomaly_detector.update_statistics(predictions, values)

        return {
            "loss": loss.item(),
            "mpp_loss": mpp_loss.item(),
            "physics_loss": physics_loss.item(),
        }

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        assert self.model is not None
        self.model.eval()

        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            batch = self._to_device(batch)
            values = batch["values"]
            delta_ts = batch["delta_ts"]

            mpp_output = self.model.forward_pretrain(
                x=values,
                delta_ts=delta_ts,
            )
            total_loss += mpp_output["loss"].item()
            n_batches += 1

        self.model.train()
        return {"loss": total_loss / max(n_batches, 1)}


# ---------------------------------------------------------------------------
# Phase 2: Supervised Anomaly Fine-tuning Trainer
# ---------------------------------------------------------------------------

@dataclass
class AnomalyFinetuneConfig(TrainerConfig):
    """Configuration for Phase 2: supervised anomaly fine-tuning."""

    lr: float = 1e-4
    batch_size: int = 128
    epochs: int = 50
    scheduler: str = "cosine"
    weight_decay: float = 0.01
    wandb_run_name: str = "sensor-anomaly-finetune"

    # Data
    data_dir: str = "data/sensor/anomaly"
    window_length: int = 672
    val_fraction: float = 0.2

    # Loss
    contrastive_weight: float = 0.5
    contrastive_temperature: float = 0.07
    classification_weight: float = 1.0

    # Pretrained checkpoint
    pretrain_checkpoint: str = ""


class AnomalyClassificationHead(nn.Module):
    """Binary anomaly classification head on top of SensorEncoder embeddings."""

    def __init__(self, embed_dim: int = 256) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """Returns logits (B,)."""
        return self.head(embedding).squeeze(-1)


class AnomalyFinetuneModel(nn.Module):
    """Sensor encoder with anomaly classification head for Phase 2."""

    def __init__(
        self,
        sensor_encoder: SensorEncoder,
        embed_dim: int = 256,
    ) -> None:
        super().__init__()
        self.encoder = sensor_encoder
        self.cls_head = AnomalyClassificationHead(embed_dim)

    def forward(
        self,
        values: torch.Tensor,
        delta_ts: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        enc_out = self.encoder(
            x=values,
            delta_ts=delta_ts,
            masks=masks,
            compute_anomaly=False,
        )
        embedding = enc_out["embedding"]  # (B, 256)
        logits = self.cls_head(embedding)  # (B,)
        return {
            "embedding": embedding,
            "logits": logits,
        }


class AnomalyFinetuneTrainer(BaseTrainer):
    """Phase 2: Supervised anomaly detection with contrastive + BCE loss."""

    def __init__(self, config: AnomalyFinetuneConfig) -> None:
        super().__init__(config)
        self.anom_config = config

    def build_model(self) -> nn.Module:
        sensor_encoder = SensorEncoder(
            num_params=NUM_PARAMETERS,
            output_dim=OUTPUT_DIM,
        )

        # Load pretrained checkpoint if available
        if self.anom_config.pretrain_checkpoint:
            ckpt_path = Path(self.anom_config.pretrain_checkpoint)
            if ckpt_path.exists():
                logger.info(f"Loading pretrained weights from {ckpt_path}")
                state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                model_state = state.get("model_state_dict", state)
                # Filter to sensor encoder keys only
                enc_state = {
                    k.replace("encoder.", "", 1) if k.startswith("encoder.") else k: v
                    for k, v in model_state.items()
                }
                sensor_encoder.load_state_dict(enc_state, strict=False)
                logger.info("Pretrained weights loaded successfully")

        model = AnomalyFinetuneModel(sensor_encoder)
        return model

    def build_datasets(self) -> Tuple[Dataset, Dataset]:
        full_ds = AnomalyFinetuneDataset(
            self.anom_config.data_dir,
            window_length=self.anom_config.window_length,
        )

        n = len(full_ds)
        n_val = int(n * self.anom_config.val_fraction)
        indices = list(range(n))
        random.shuffle(indices)
        train_ds = Subset(full_ds, indices[n_val:])
        val_ds = Subset(full_ds, indices[:n_val])

        logger.info(f"Anomaly Finetune — Train: {len(train_ds)}, Val: {len(val_ds)}")
        return train_ds, val_ds

    def _contrastive_loss(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        temperature: float = 0.07,
    ) -> torch.Tensor:
        """Supervised contrastive loss (SupCon).

        Pulls together embeddings with the same label, pushes apart
        embeddings with different labels.

        Args:
            embeddings: Normalized embeddings (B, D).
            labels: Binary labels (B,).
            temperature: Softmax temperature.

        Returns:
            Scalar contrastive loss.
        """
        B = embeddings.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=-1)

        # Similarity matrix
        sim = torch.mm(embeddings, embeddings.t()) / temperature  # (B, B)

        # Mask: same label = positive pair
        labels_col = labels.unsqueeze(1)
        positive_mask = (labels_col == labels_col.t()).float()
        # Remove self-pairs
        identity = torch.eye(B, device=embeddings.device)
        positive_mask = positive_mask - identity

        # For numerical stability
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        # Log-sum-exp over all non-self pairs
        exp_sim = torch.exp(sim) * (1.0 - identity)
        log_sum_exp = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # Mean of log(positive_similarity / sum_all)
        log_prob = sim - log_sum_exp
        # Average over positive pairs
        n_positives = positive_mask.sum(dim=1).clamp(min=1)
        loss = -(positive_mask * log_prob).sum(dim=1) / n_positives

        return loss.mean()

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        assert self.model is not None and self.optimizer is not None

        values = batch["values"]
        delta_ts = batch["delta_ts"]
        masks = batch["masks"]
        labels = batch["label"]

        outputs = self.model(values, delta_ts, masks)
        embedding = outputs["embedding"]
        logits = outputs["logits"]

        # Binary cross-entropy loss
        bce_loss = F.binary_cross_entropy_with_logits(logits, labels)

        # Supervised contrastive loss
        contrastive = self._contrastive_loss(
            embedding, labels, self.anom_config.contrastive_temperature
        )

        total_loss = (
            self.anom_config.classification_weight * bce_loss
            + self.anom_config.contrastive_weight * contrastive
        )

        total_loss.backward()

        if self.config.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        return {
            "loss": total_loss.item(),
            "bce_loss": bce_loss.item(),
            "contrastive_loss": contrastive.item(),
        }

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        assert self.model is not None
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0
        all_probs = []
        all_labels = []

        for batch in dataloader:
            batch = self._to_device(batch)
            values = batch["values"]
            delta_ts = batch["delta_ts"]
            masks = batch["masks"]
            labels = batch["label"]

            outputs = self.model(values, delta_ts, masks)
            logits = outputs["logits"]

            bce_loss = F.binary_cross_entropy_with_logits(logits, labels)
            total_loss += bce_loss.item()

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

        self.model.train()

        all_probs_cat = torch.cat(all_probs)
        all_labels_cat = torch.cat(all_labels)

        # Compute AUROC if both classes present
        auroc = 0.0
        if len(all_labels_cat.unique()) > 1:
            try:
                from sklearn.metrics import roc_auc_score
                auroc = roc_auc_score(
                    all_labels_cat.numpy(), all_probs_cat.numpy()
                )
            except Exception:
                pass

        n_batches = max(total // max(self.config.batch_size, 1), 1)
        return {
            "loss": total_loss / n_batches,
            "accuracy": correct / max(total, 1),
            "auroc": auroc,
        }


# ---------------------------------------------------------------------------
# Phase 3: Sensor Health Classifier Trainer
# ---------------------------------------------------------------------------

@dataclass
class SensorHealthConfig(TrainerConfig):
    """Configuration for Phase 3: sensor health classifier training."""

    lr: float = 1e-3
    batch_size: int = 128
    epochs: int = 50
    scheduler: str = "cosine"
    weight_decay: float = 0.01
    wandb_run_name: str = "sensor-health-classifier"

    # Data
    data_dir: str = "data/sensor/pretrain"
    window_length: int = 672
    stride: int = 168
    val_fraction: float = 0.2
    fault_probability: float = 0.8

    # Pretrained checkpoint (Phase 1 or Phase 2)
    pretrain_checkpoint: str = ""


class SensorHealthModel(nn.Module):
    """Frozen SensorEncoder backbone + trainable SensorHealthSentinel head."""

    def __init__(
        self,
        sensor_encoder: SensorEncoder,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = sensor_encoder
        self.health_head = sensor_encoder.sensor_health

        if freeze_backbone:
            # Freeze SSM backbone and reconstruction head, keep health head trainable
            for name, param in sensor_encoder.named_parameters():
                if "sensor_health" not in name:
                    param.requires_grad = False

    def forward(
        self,
        values: torch.Tensor,
        delta_ts: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass: backbone -> reconstruction errors -> health classification.

        Args:
            values: Sensor readings (B, T, P).
            delta_ts: Time gaps (B, T).

        Returns:
            Dict with health logits, predictions, and error features.
        """
        # Get reconstruction errors via the anomaly detector
        error_results = self.encoder.anomaly_detector.compute_reconstruction_errors(
            values, delta_ts=delta_ts
        )

        raw_errors = error_results["raw_errors"]        # (B, P, T)
        temporal_features = error_results["temporal_features"]  # (B, T, D)

        # Health classification
        health_out = self.health_head(raw_errors, temporal_features)

        return {
            "health_logits": health_out["health_logits"],    # (B, P, C)
            "health_probs": health_out["health_probs"],      # (B, P, C)
            "health_status": health_out["health_status"],    # (B, P)
            "anomaly_weights": health_out["anomaly_weights"],  # (B, P)
        }


class SensorHealthTrainer(BaseTrainer):
    """Phase 3: Train SensorHealthSentinel on simulated sensor failures."""

    def __init__(self, config: SensorHealthConfig) -> None:
        super().__init__(config)
        self.health_config = config

    def build_model(self) -> nn.Module:
        sensor_encoder = SensorEncoder(
            num_params=NUM_PARAMETERS,
            output_dim=OUTPUT_DIM,
        )

        # Load pretrained checkpoint
        if self.health_config.pretrain_checkpoint:
            ckpt_path = Path(self.health_config.pretrain_checkpoint)
            if ckpt_path.exists():
                logger.info(f"Loading pretrained weights from {ckpt_path}")
                state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                model_state = state.get("model_state_dict", state)
                # Handle keys wrapped in AnomalyFinetuneModel or bare SensorEncoder
                cleaned = {}
                for k, v in model_state.items():
                    if k.startswith("encoder."):
                        cleaned[k[len("encoder."):]] = v
                    else:
                        cleaned[k] = v
                sensor_encoder.load_state_dict(cleaned, strict=False)
                logger.info("Pretrained weights loaded")

        model = SensorHealthModel(sensor_encoder, freeze_backbone=True)
        return model

    def build_datasets(self) -> Tuple[Dataset, Dataset]:
        full_ds = SensorHealthDataset(
            self.health_config.data_dir,
            window_length=self.health_config.window_length,
            stride=self.health_config.stride,
            fault_probability=self.health_config.fault_probability,
        )

        n = len(full_ds)
        n_val = int(n * self.health_config.val_fraction)
        indices = list(range(n))
        random.shuffle(indices)
        train_ds = Subset(full_ds, indices[n_val:])
        val_ds = Subset(full_ds, indices[:n_val])

        logger.info(f"Sensor Health — Train: {len(train_ds)}, Val: {len(val_ds)}")
        return train_ds, val_ds

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        assert self.model is not None and self.optimizer is not None

        values = batch["values"]          # (B, T, P)
        delta_ts = batch["delta_ts"]      # (B, T)
        health_labels = batch["health_labels"]  # (B, P) per-sensor class

        # Enable gradients for reconstruction error computation
        # (health head is trainable, backbone is frozen)
        with torch.no_grad():
            error_results = self.model.encoder.anomaly_detector.compute_reconstruction_errors(
                values, delta_ts=delta_ts
            )

        raw_errors = error_results["raw_errors"]
        temporal_features = error_results["temporal_features"]

        health_out = self.model.health_head(raw_errors, temporal_features)
        health_logits = health_out["health_logits"]  # (B, P, C)

        B, P, C = health_logits.shape
        # Cross-entropy loss per sensor, averaged
        loss = F.cross_entropy(
            health_logits.reshape(B * P, C),
            health_labels.reshape(B * P),
        )

        loss.backward()

        if self.config.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        # Accuracy
        preds = health_logits.argmax(dim=-1)  # (B, P)
        correct = (preds == health_labels).float().mean().item()

        return {
            "loss": loss.item(),
            "accuracy": correct,
        }

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        assert self.model is not None
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0
        per_class_correct = [0] * NUM_HEALTH_CLASSES
        per_class_total = [0] * NUM_HEALTH_CLASSES

        for batch in dataloader:
            batch = self._to_device(batch)
            values = batch["values"]
            delta_ts = batch["delta_ts"]
            health_labels = batch["health_labels"]

            error_results = self.model.encoder.anomaly_detector.compute_reconstruction_errors(
                values, delta_ts=delta_ts
            )
            raw_errors = error_results["raw_errors"]
            temporal_features = error_results["temporal_features"]

            health_out = self.model.health_head(raw_errors, temporal_features)
            health_logits = health_out["health_logits"]  # (B, P, C)

            B, P, C = health_logits.shape
            loss = F.cross_entropy(
                health_logits.reshape(B * P, C),
                health_labels.reshape(B * P),
            )
            total_loss += loss.item()

            preds = health_logits.argmax(dim=-1)
            correct += (preds == health_labels).sum().item()
            total += B * P

            for c in range(NUM_HEALTH_CLASSES):
                mask = health_labels == c
                per_class_total[c] += mask.sum().item()
                per_class_correct[c] += (preds[mask] == c).sum().item()

        self.model.train()

        n_batches = max(total // max(self.config.batch_size * NUM_PARAMETERS, 1), 1)
        metrics = {
            "loss": total_loss / max(n_batches, 1),
            "accuracy": correct / max(total, 1),
        }

        # Per-class accuracy
        for c, name in enumerate(HEALTH_CLASSES):
            if per_class_total[c] > 0:
                metrics[f"acc_{name}"] = per_class_correct[c] / per_class_total[c]

        return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SENTINEL AquaSSM sensor encoder training pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--phase", type=int, required=True, choices=[1, 2, 3],
        help="Training phase (1=MPP pretrain, 2=anomaly finetune, 3=sensor health)",
    )
    parser.add_argument("--data-dir", type=str, default="data/sensor/pretrain")
    parser.add_argument("--output-dir", type=str, default="outputs/sensor")
    parser.add_argument("--config", type=str, default="", help="Path to YAML config override")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)

    # Common training params
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--no-wandb", action="store_true")

    # Phase 1 specific
    parser.add_argument("--window-length", type=int, default=672)
    parser.add_argument("--stride", type=int, default=168)
    parser.add_argument("--val-fraction", type=float, default=0.2)

    # Phase 2 specific
    parser.add_argument("--pretrain-checkpoint", type=str, default="")
    parser.add_argument("--contrastive-weight", type=float, default=0.5)

    # Phase 3 specific
    parser.add_argument("--fault-probability", type=float, default=0.8)

    return parser


def _load_yaml_config(path: str) -> Dict[str, Any]:
    """Load YAML config and return training.sensor section."""
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cfg.get("training", {}).get("sensor", {})
    except Exception as e:
        logger.warning(f"Could not load config from {path}: {e}")
        return {}


def main() -> None:
    args = build_argparser().parse_args()

    # Load YAML overrides if provided
    yaml_cfg: Dict[str, Any] = {}
    if args.config:
        yaml_cfg = _load_yaml_config(args.config)

    if args.phase == 1:
        pretrain_cfg = yaml_cfg.get("pretraining", {})
        config = MPPPretrainConfig(
            lr=args.lr or pretrain_cfg.get("lr", 5e-4),
            batch_size=args.batch_size or pretrain_cfg.get("batch_size", 256),
            epochs=args.epochs or pretrain_cfg.get("epochs", 100),
            warmup_steps=pretrain_cfg.get("warmup_steps", 5000),
            weight_decay=pretrain_cfg.get("weight_decay", 0.01),
            scheduler=pretrain_cfg.get("scheduler", "cosine"),
            output_dir=args.output_dir,
            device=args.device,
            seed=args.seed,
            use_wandb=not args.no_wandb,
            data_dir=args.data_dir,
            window_length=args.window_length,
            stride=args.stride,
            val_fraction=args.val_fraction,
        )
        trainer = MPPPretrainTrainer(config)
        trainer.setup()
        trainer.train()

    elif args.phase == 2:
        finetune_cfg = yaml_cfg.get("finetuning", {})
        config = AnomalyFinetuneConfig(
            lr=args.lr or finetune_cfg.get("lr", 1e-4),
            batch_size=args.batch_size or finetune_cfg.get("batch_size", 128),
            epochs=args.epochs or finetune_cfg.get("epochs", 50),
            weight_decay=finetune_cfg.get("weight_decay", 0.01),
            scheduler=finetune_cfg.get("scheduler", "cosine"),
            output_dir=args.output_dir,
            device=args.device,
            seed=args.seed,
            use_wandb=not args.no_wandb,
            data_dir=args.data_dir,
            window_length=args.window_length,
            val_fraction=args.val_fraction,
            pretrain_checkpoint=args.pretrain_checkpoint,
            contrastive_weight=args.contrastive_weight,
        )
        trainer = AnomalyFinetuneTrainer(config)
        trainer.setup()
        trainer.train()

    elif args.phase == 3:
        config = SensorHealthConfig(
            lr=args.lr or 1e-3,
            batch_size=args.batch_size or 128,
            epochs=args.epochs or 50,
            output_dir=args.output_dir,
            device=args.device,
            seed=args.seed,
            use_wandb=not args.no_wandb,
            data_dir=args.data_dir,
            window_length=args.window_length,
            stride=args.stride,
            val_fraction=args.val_fraction,
            pretrain_checkpoint=args.pretrain_checkpoint,
            fault_probability=args.fault_probability,
        )
        trainer = SensorHealthTrainer(config)
        trainer.setup()
        trainer.train()


if __name__ == "__main__":
    main()
