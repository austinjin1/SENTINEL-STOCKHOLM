"""
MicroBiomeNet training pipeline for SENTINEL.

Four-phase training:
  Phase 1: Data preparation — load CLR matrices, representative sequences,
           zero-inflation annotations from prepare_microbiomenet_inputs output.
  Phase 2: Source attribution — train Aitchison attention on EPA NARS data
           with chemistry-based clustering labels. 5-fold CV by watershed.
  Phase 3: Simplex ODE — train temporal trajectory model on reference-condition
           sites. Reconstruction loss + KL on simplex trajectories.
  Phase 4: Attention analysis — extract per-class mean attention weights,
           rank indicator taxa, save as JSON.

Usage:
    python -m sentinel.training.train_microbial --phase 1 --data-dir data/microbial
    python -m sentinel.training.train_microbial --phase 2 --data-dir data/microbial/prepared
    python -m sentinel.training.train_microbial --phase 3 --data-dir data/microbial/prepared
    python -m sentinel.training.train_microbial --phase 4 --data-dir data/microbial/prepared --checkpoint outputs/microbial/checkpoints/best_model.pt
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from sentinel.models.microbial_encoder.model import (
    CONTAMINATION_SOURCES,
    EMBED_DIM,
    MAX_ASV_FEATURES,
    NUM_SOURCES,
    MicrobialEncoder,
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
# Datasets
# ---------------------------------------------------------------------------

class MicrobialDataset(Dataset):
    """CLR-transformed microbial community samples with metadata.

    Loads pre-computed outputs from prepare_microbiomenet_inputs:
      - CLR matrix: (n_samples, n_features) float32 .npy
      - Metadata JSON with sample IDs, source labels, watershed IDs
      - Optional: representative sequences JSON, zero-inflation annotations

    Args:
        data_dir: Directory containing prepared MicroBiomeNet inputs.
        split_indices: Optional subset indices for train/val split.
        max_features: Maximum ASV features (pads or truncates).
    """

    def __init__(
        self,
        data_dir: str | Path,
        split_indices: Optional[np.ndarray] = None,
        max_features: int = MAX_ASV_FEATURES,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.max_features = max_features

        # Load CLR matrix
        clr_path = self.data_dir / "clr" / "clr_matrix.npy"
        if not clr_path.exists():
            # Try flat layout
            clr_path = self.data_dir / "clr_matrix.npy"
        self.clr_matrix = np.load(clr_path).astype(np.float32)

        # Load metadata
        meta_path = self.data_dir / "clr" / "metadata.json"
        if not meta_path.exists():
            meta_path = self.data_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

        # Source labels (int indices into CONTAMINATION_SOURCES)
        self.source_labels = np.array(
            self.metadata.get("source_labels", [-1] * len(self.clr_matrix)),
            dtype=np.int64,
        )

        # Watershed IDs for stratified splitting
        self.watershed_ids = np.array(
            self.metadata.get("watershed_ids", list(range(len(self.clr_matrix)))),
        )

        # Feature names
        self.feature_names = self.metadata.get("feature_names", [])

        # Optional raw abundances for zero-inflation gating
        raw_path = self.data_dir / "simplex" / "proportions.npy"
        if not raw_path.exists():
            raw_path = self.data_dir / "raw_abundances.npy"
        self.raw_abundances = (
            np.load(raw_path).astype(np.float32) if raw_path.exists() else None
        )

        # Optional representative sequences
        seqs_path = self.data_dir / "rep_seqs.json"
        if seqs_path.exists():
            with open(seqs_path, "r", encoding="utf-8") as f:
                self.rep_sequences = json.load(f)
        else:
            self.rep_sequences = None

        # Subset if indices provided
        if split_indices is not None:
            self.clr_matrix = self.clr_matrix[split_indices]
            self.source_labels = self.source_labels[split_indices]
            self.watershed_ids = self.watershed_ids[split_indices]
            if self.raw_abundances is not None:
                self.raw_abundances = self.raw_abundances[split_indices]

        logger.info(
            f"MicrobialDataset: {len(self.clr_matrix)} samples, "
            f"{self.clr_matrix.shape[1]} features"
        )

    def __len__(self) -> int:
        return len(self.clr_matrix)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        clr = self.clr_matrix[idx]
        n_feat = clr.shape[0]

        # Pad or truncate to max_features
        if n_feat < self.max_features:
            clr = np.pad(clr, (0, self.max_features - n_feat))
        elif n_feat > self.max_features:
            clr = clr[: self.max_features]

        result: Dict[str, torch.Tensor] = {
            "clr": torch.from_numpy(clr),
            "source_label": torch.tensor(self.source_labels[idx], dtype=torch.long),
        }

        if self.raw_abundances is not None:
            raw = self.raw_abundances[idx]
            if raw.shape[0] < self.max_features:
                raw = np.pad(raw, (0, self.max_features - raw.shape[0]))
            elif raw.shape[0] > self.max_features:
                raw = raw[: self.max_features]
            result["raw_abundances"] = torch.from_numpy(raw)

        return result


class ReferenceConditionDataset(Dataset):
    """Temporal CLR sequences from reference-condition (healthy) sites.

    For training the Simplex ODE to model healthy community trajectories.
    Contaminated sites are used for validation: the model should produce
    high reconstruction error on contaminated samples.

    Expected layout:
        data_dir/
            reference_sequences/    -- .npz files with keys: clr_seq (T, D), timestamps (T,)
            contaminated_sequences/ -- .npz files (same format, for validation)
            reference_metadata.json -- site metadata

    Args:
        data_dir: Directory containing reference condition data.
        mode: "reference" for healthy sites, "contaminated" for val.
        max_features: Maximum ASV features.
    """

    def __init__(
        self,
        data_dir: str | Path,
        mode: str = "reference",
        max_features: int = MAX_ASV_FEATURES,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.max_features = max_features
        self.mode = mode

        if mode == "reference":
            seq_dir = self.data_dir / "reference_sequences"
        else:
            seq_dir = self.data_dir / "contaminated_sequences"

        if not seq_dir.exists():
            seq_dir.mkdir(parents=True, exist_ok=True)
            logger.warning(f"Sequence directory created (empty): {seq_dir}")

        self.sequence_paths: List[Path] = sorted(seq_dir.glob("*.npz"))
        logger.info(
            f"ReferenceConditionDataset ({mode}): {len(self.sequence_paths)} sequences"
        )

    def __len__(self) -> int:
        return len(self.sequence_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = np.load(self.sequence_paths[idx])
        clr_seq = data["clr_seq"].astype(np.float32)  # (T, D)
        timestamps = data["timestamps"].astype(np.float32)  # (T,)
        T, D = clr_seq.shape

        # Pad features to max_features
        if D < self.max_features:
            clr_seq = np.pad(clr_seq, ((0, 0), (0, self.max_features - D)))
        elif D > self.max_features:
            clr_seq = clr_seq[:, : self.max_features]

        # Also provide the last timepoint as the "current" CLR snapshot
        clr_current = clr_seq[-1]

        return {
            "clr_sequence": torch.from_numpy(clr_seq),
            "timestamps": torch.from_numpy(timestamps),
            "clr_current": torch.from_numpy(clr_current),
            "n_timepoints": torch.tensor(T, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Phase 2: Source Attribution Trainer
# ---------------------------------------------------------------------------

@dataclass
class SourceAttributionConfig(TrainerConfig):
    """Config for source attribution training (Phase 2)."""

    lr: float = 1e-3
    batch_size: int = 64
    epochs: int = 200
    weight_decay: float = 0.01
    scheduler: str = "cosine"
    patience: int = 15
    wandb_run_name: str = "microbial-source-attribution"

    # Data
    data_dir: str = "data/microbial/prepared"
    val_fraction: float = 0.2
    max_features: int = MAX_ASV_FEATURES

    # Model
    embed_dim: int = EMBED_DIM
    num_heads: int = 4
    num_aitchison_layers: int = 4
    ff_dim: int = 512
    dropout: float = 0.1
    num_sources: int = NUM_SOURCES
    freeze_dnabert: bool = True

    # Cross-validation
    n_folds: int = 5


class SourceAttributionTrainer(BaseTrainer):
    """Phase 2: Source attribution via Aitchison attention.

    Trains the MicrobialEncoder's source attribution head on EPA NARS data.
    Labels come from chemistry-based clustering of contamination types.
    """

    def __init__(
        self,
        config: SourceAttributionConfig,
        train_indices: Optional[np.ndarray] = None,
        val_indices: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(config)
        self.sa_config = config
        self.train_indices = train_indices
        self.val_indices = val_indices

    def build_model(self) -> nn.Module:
        model = MicrobialEncoder(
            input_dim=self.sa_config.max_features,
            embed_dim=self.sa_config.embed_dim,
            num_heads=self.sa_config.num_heads,
            num_aitchison_layers=self.sa_config.num_aitchison_layers,
            ff_dim=self.sa_config.ff_dim,
            dropout=self.sa_config.dropout,
            num_sources=self.sa_config.num_sources,
            freeze_dnabert=self.sa_config.freeze_dnabert,
        )
        return model

    def build_datasets(self) -> Tuple[Dataset, Dataset]:
        full_ds = MicrobialDataset(
            self.sa_config.data_dir,
            max_features=self.sa_config.max_features,
        )

        if self.train_indices is not None and self.val_indices is not None:
            train_ds = MicrobialDataset(
                self.sa_config.data_dir,
                split_indices=self.train_indices,
                max_features=self.sa_config.max_features,
            )
            val_ds = MicrobialDataset(
                self.sa_config.data_dir,
                split_indices=self.val_indices,
                max_features=self.sa_config.max_features,
            )
        else:
            # Random split
            n = len(full_ds)
            n_val = int(n * self.sa_config.val_fraction)
            indices = np.arange(n)
            np.random.shuffle(indices)
            train_ds = MicrobialDataset(
                self.sa_config.data_dir,
                split_indices=indices[n_val:],
                max_features=self.sa_config.max_features,
            )
            val_ds = MicrobialDataset(
                self.sa_config.data_dir,
                split_indices=indices[:n_val],
                max_features=self.sa_config.max_features,
            )

        logger.info(f"Source attribution: train={len(train_ds)}, val={len(val_ds)}")
        return train_ds, val_ds

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        assert self.model is not None and self.optimizer is not None

        clr = batch["clr"]
        source_labels = batch["source_label"]
        raw = batch.get("raw_abundances", None)

        # Forward
        outputs = self.model(x=clr, raw_abundances=raw)

        # Source attribution loss
        source_loss = F.cross_entropy(
            outputs["source_logits"], source_labels, ignore_index=-1
        )

        # Health score regularization
        health_reg = outputs["community_health_score"].pow(2).mean() * 0.01

        total_loss = source_loss + health_reg
        total_loss.backward()

        if self.config.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        return {
            "loss": total_loss.item(),
            "source_loss": source_loss.item(),
            "health_reg": health_reg.item(),
        }

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        assert self.model is not None
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0
        per_class_correct: Dict[int, int] = defaultdict(int)
        per_class_total: Dict[int, int] = defaultdict(int)

        for batch in dataloader:
            batch = self._to_device(batch)
            clr = batch["clr"]
            source_labels = batch["source_label"]
            raw = batch.get("raw_abundances", None)

            outputs = self.model(x=clr, raw_abundances=raw)
            loss = F.cross_entropy(
                outputs["source_logits"], source_labels, ignore_index=-1
            )
            total_loss += loss.item()

            # Accuracy (exclude unlabeled samples)
            valid_mask = source_labels >= 0
            if valid_mask.any():
                preds = outputs["source_logits"][valid_mask].argmax(dim=-1)
                labels = source_labels[valid_mask]
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                for pred, label in zip(preds.cpu().tolist(), labels.cpu().tolist()):
                    per_class_total[label] += 1
                    if pred == label:
                        per_class_correct[label] += 1

        self.model.train()

        n_batches = max(1, total // max(self.config.batch_size, 1))
        accuracy = correct / max(total, 1)

        # Per-class accuracy
        class_accs = {}
        for cls in sorted(per_class_total.keys()):
            cls_acc = per_class_correct[cls] / max(per_class_total[cls], 1)
            if cls < len(CONTAMINATION_SOURCES):
                class_accs[CONTAMINATION_SOURCES[cls]] = cls_acc

        result = {
            "loss": total_loss / max(n_batches, 1),
            "accuracy": accuracy,
        }
        return result


# ---------------------------------------------------------------------------
# Phase 3: Simplex ODE Trainer
# ---------------------------------------------------------------------------

@dataclass
class SimplexODEConfig(TrainerConfig):
    """Config for Simplex ODE training (Phase 3)."""

    lr: float = 1e-3
    batch_size: int = 32
    epochs: int = 300
    weight_decay: float = 0.01
    scheduler: str = "cosine"
    patience: int = 20
    wandb_run_name: str = "microbial-simplex-ode"

    # Data
    data_dir: str = "data/microbial/prepared"
    max_features: int = MAX_ASV_FEATURES

    # Model — load from Phase 2 checkpoint
    checkpoint_path: str = ""

    # Loss weights
    reconstruction_weight: float = 1.0
    kl_weight: float = 0.1

    # Model hyperparameters
    embed_dim: int = EMBED_DIM
    num_heads: int = 4
    num_aitchison_layers: int = 4
    ff_dim: int = 512
    dropout: float = 0.1
    num_sources: int = NUM_SOURCES


class SimplexODETrainer(BaseTrainer):
    """Phase 3: Train Simplex ODE on reference-condition sites.

    Trains the temporal trajectory model on healthy (reference-condition)
    sites from EPA NARS. The model learns to reconstruct community
    trajectories; contaminated sites should yield high reconstruction error.
    """

    def __init__(self, config: SimplexODEConfig) -> None:
        super().__init__(config)
        self.ode_config = config

    def build_model(self) -> nn.Module:
        model = MicrobialEncoder(
            input_dim=self.ode_config.max_features,
            embed_dim=self.ode_config.embed_dim,
            num_heads=self.ode_config.num_heads,
            num_aitchison_layers=self.ode_config.num_aitchison_layers,
            ff_dim=self.ode_config.ff_dim,
            dropout=self.ode_config.dropout,
            num_sources=self.ode_config.num_sources,
        )

        # Load Phase 2 checkpoint if available
        if self.ode_config.checkpoint_path:
            ckpt_path = Path(self.ode_config.checkpoint_path)
            if ckpt_path.exists():
                state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                model_state = state.get("model_state_dict", state)
                model.load_state_dict(model_state, strict=False)
                logger.info(f"Loaded Phase 2 checkpoint: {ckpt_path}")
            else:
                logger.warning(f"Checkpoint not found: {ckpt_path}")

        # Freeze source attribution head and Aitchison layers;
        # only train the Simplex ODE and fusion components
        self.freeze_model(model)
        for p in model.simplex_ode.parameters():
            p.requires_grad = True
        for p in model.fusion_layer.parameters():
            p.requires_grad = True
        for p in model.projection.parameters():
            p.requires_grad = True

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(
            f"Simplex ODE training: {trainable:,} / {total_params:,} params trainable"
        )

        return model

    def build_datasets(self) -> Tuple[Dataset, Dataset]:
        data_dir = Path(self.ode_config.data_dir)

        train_ds = ReferenceConditionDataset(
            data_dir, mode="reference", max_features=self.ode_config.max_features
        )
        val_ds = ReferenceConditionDataset(
            data_dir, mode="contaminated", max_features=self.ode_config.max_features
        )

        # If no contaminated validation data, split reference 80/20
        if len(val_ds) == 0:
            n = len(train_ds)
            n_val = max(1, int(n * 0.2))
            indices = list(range(n))
            random.shuffle(indices)
            val_ds = Subset(train_ds, indices[:n_val])
            train_ds = Subset(train_ds, indices[n_val:])

        logger.info(f"Simplex ODE: train={len(train_ds)}, val={len(val_ds)}")
        return train_ds, val_ds

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        assert self.model is not None and self.optimizer is not None

        clr_seq = batch["clr_sequence"]      # (B, T, D)
        timestamps = batch["timestamps"]     # (B, T)
        clr_current = batch["clr_current"]   # (B, D)

        # Forward with temporal data
        outputs = self.model(
            x=clr_current,
            clr_sequence=clr_seq,
            timestamps=timestamps,
        )

        # Reconstruction loss: run ODE separately for trajectory prediction
        ode_output = self.model.simplex_ode(clr_seq, timestamps)
        recon_loss = F.mse_loss(
            ode_output["predicted_trajectory"],
            ode_output["observed_states"],
        )

        # KL-like regularization on trajectory embedding
        # Encourage the trajectory embedding to be close to unit Gaussian
        traj_emb = ode_output["trajectory_embedding"]
        kl_loss = 0.5 * (traj_emb.pow(2).sum(dim=-1) - traj_emb.shape[-1]).mean()
        kl_loss = kl_loss.clamp(min=0.0)

        total_loss = (
            self.ode_config.reconstruction_weight * recon_loss
            + self.ode_config.kl_weight * kl_loss
        )
        total_loss.backward()

        if self.config.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        return {
            "loss": total_loss.item(),
            "recon_loss": recon_loss.item(),
            "kl_loss": kl_loss.item(),
        }

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        assert self.model is not None
        self.model.eval()

        total_recon = 0.0
        total_anomaly = 0.0
        n_batches = 0

        for batch in dataloader:
            batch = self._to_device(batch)
            clr_seq = batch["clr_sequence"]
            timestamps = batch["timestamps"]
            clr_current = batch["clr_current"]

            outputs = self.model(
                x=clr_current,
                clr_sequence=clr_seq,
                timestamps=timestamps,
            )

            ode_output = self.model.simplex_ode(clr_seq, timestamps)
            recon = F.mse_loss(
                ode_output["predicted_trajectory"],
                ode_output["observed_states"],
            )
            total_recon += recon.item()
            total_anomaly += outputs["community_health_score"].mean().item()
            n_batches += 1

        self.model.train()
        n = max(n_batches, 1)

        return {
            "loss": total_recon / n,
            "recon_error": total_recon / n,
            "mean_anomaly_score": total_anomaly / n,
        }


# ---------------------------------------------------------------------------
# Cross-validation utilities
# ---------------------------------------------------------------------------

def split_by_watershed(
    dataset: MicrobialDataset,
    n_folds: int = 5,
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate k-fold splits stratified by watershed ID.

    Ensures that all samples from the same watershed stay together in
    either train or validation, preventing spatial data leakage.

    Returns:
        List of (train_indices, val_indices) tuples.
    """
    rng = np.random.RandomState(seed)
    unique_watersheds = np.unique(dataset.watershed_ids)
    rng.shuffle(unique_watersheds)

    # Assign watersheds to folds
    fold_assignments = np.array_split(unique_watersheds, n_folds)

    folds = []
    for fold_idx in range(n_folds):
        val_watersheds = set(fold_assignments[fold_idx].tolist())
        val_mask = np.array(
            [ws in val_watersheds for ws in dataset.watershed_ids]
        )
        train_indices = np.where(~val_mask)[0]
        val_indices = np.where(val_mask)[0]
        folds.append((train_indices, val_indices))
        logger.info(
            f"Fold {fold_idx + 1}: train={len(train_indices)}, "
            f"val={len(val_indices)}, "
            f"val_watersheds={len(fold_assignments[fold_idx])}"
        )

    return folds


# ---------------------------------------------------------------------------
# Phase 1: Data preparation
# ---------------------------------------------------------------------------

def run_phase1(
    data_dir: str,
    output_dir: str,
    config_path: Optional[str] = None,
) -> Dict[str, str]:
    """Phase 1: Prepare MicroBiomeNet inputs from raw data.

    Expects the data directory to contain:
      - biom table (.biom or .biom.json)
      - metadata TSV
      - Optional: representative sequences (.qza or .fasta)

    Produces CLR matrix, zero-inflation annotations, temporal metadata.
    """
    data_dir_path = Path(data_dir)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Find input files
    biom_files = list(data_dir_path.glob("*.biom*"))
    meta_files = list(data_dir_path.glob("*metadata*.tsv")) + list(
        data_dir_path.glob("*metadata*.csv")
    )
    seq_files = list(data_dir_path.glob("*.qza")) + list(
        data_dir_path.glob("*rep_seqs*.fasta")
    )

    if not biom_files:
        logger.error(f"No BIOM files found in {data_dir_path}")
        return {"error": "No BIOM files found"}

    biom_path = biom_files[0]
    meta_path = meta_files[0] if meta_files else None
    seq_path = seq_files[0] if seq_files else None

    logger.info(f"BIOM: {biom_path}")
    logger.info(f"Metadata: {meta_path}")
    logger.info(f"Sequences: {seq_path}")

    # Load config for preprocessing parameters
    pseudocount = 0.5
    min_abundance = 0.001
    max_features = MAX_ASV_FEATURES

    if config_path:
        try:
            import yaml
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            microbial_cfg = cfg.get("data", {}).get("microbial", {})
            pseudocount = microbial_cfg.get("pseudocount", pseudocount)
            min_abundance = microbial_cfg.get("min_abundance", min_abundance)
            max_features = microbial_cfg.get("max_features", max_features)
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")

    from sentinel.data.microbial.preprocessing import prepare_microbiomenet_inputs

    result = prepare_microbiomenet_inputs(
        biom_path=str(biom_path),
        metadata_path=str(meta_path) if meta_path else "",
        output_dir=str(output_dir_path),
        rep_seqs_path=str(seq_path) if seq_path else None,
        pseudocount=pseudocount,
        min_abundance=min_abundance,
        max_features=max_features,
    )

    logger.info("Phase 1 complete. Outputs:")
    for key, path in result.items():
        logger.info(f"  {key}: {path}")

    return result


# ---------------------------------------------------------------------------
# Phase 2: Source attribution with cross-validation
# ---------------------------------------------------------------------------

def run_phase2(
    data_dir: str,
    output_dir: str,
    config_path: Optional[str] = None,
    n_folds: int = 5,
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 64,
    device: str = "auto",
    seed: int = 42,
    use_wandb: bool = True,
) -> Dict[str, Any]:
    """Phase 2: Source attribution training with 5-fold CV by watershed.

    Returns per-fold and aggregate metrics.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Build full dataset for splitting
    full_ds = MicrobialDataset(data_dir)
    folds = split_by_watershed(full_ds, n_folds=n_folds, seed=seed)

    all_fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        logger.info(f"\n{'='*60}")
        logger.info(f"FOLD {fold_idx + 1}/{n_folds}")
        logger.info(f"{'='*60}")

        fold_output = output_path / f"fold_{fold_idx}"

        config = SourceAttributionConfig(
            lr=lr,
            batch_size=batch_size,
            epochs=epochs,
            output_dir=str(fold_output),
            device=device,
            seed=seed + fold_idx,
            use_wandb=use_wandb,
            wandb_run_name=f"microbial-source-attr-fold{fold_idx}",
            data_dir=data_dir,
        )

        trainer = SourceAttributionTrainer(
            config,
            train_indices=train_idx,
            val_indices=val_idx,
        )
        trainer.setup()
        fold_result = trainer.train()
        all_fold_results.append(fold_result)

        logger.info(
            f"Fold {fold_idx + 1} — best val loss: {fold_result['best_val_metric']:.4f} "
            f"at epoch {fold_result['best_epoch']}"
        )

    # Aggregate cross-validation results
    cv_losses = [r["best_val_metric"] for r in all_fold_results]
    cv_summary = {
        "mean_val_loss": float(np.mean(cv_losses)),
        "std_val_loss": float(np.std(cv_losses)),
        "per_fold_best_loss": cv_losses,
        "per_fold_best_epoch": [r["best_epoch"] for r in all_fold_results],
        "n_folds": n_folds,
    }

    summary_path = output_path / "cv_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(cv_summary, f, indent=2)

    logger.info(f"\nCross-validation summary:")
    logger.info(f"  Mean val loss: {cv_summary['mean_val_loss']:.4f} "
                f"+/- {cv_summary['std_val_loss']:.4f}")
    logger.info(f"  Saved to {summary_path}")

    # Train final model on all data using best hyperparameters
    logger.info("\nTraining final model on all data...")
    final_config = SourceAttributionConfig(
        lr=lr,
        batch_size=batch_size,
        epochs=epochs,
        output_dir=str(output_path / "final"),
        device=device,
        seed=seed,
        use_wandb=use_wandb,
        wandb_run_name="microbial-source-attr-final",
        data_dir=data_dir,
    )
    final_trainer = SourceAttributionTrainer(final_config)
    final_trainer.setup()
    final_result = final_trainer.train()

    cv_summary["final_model"] = {
        "best_val_metric": final_result["best_val_metric"],
        "total_epochs": final_result["total_epochs"],
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(cv_summary, f, indent=2)

    return cv_summary


# ---------------------------------------------------------------------------
# Phase 3: Simplex ODE training
# ---------------------------------------------------------------------------

def run_phase3(
    data_dir: str,
    output_dir: str,
    checkpoint_path: str = "",
    epochs: int = 300,
    lr: float = 1e-3,
    batch_size: int = 32,
    device: str = "auto",
    seed: int = 42,
    use_wandb: bool = True,
) -> Dict[str, Any]:
    """Phase 3: Train Simplex ODE on reference-condition sites."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # If no explicit checkpoint, look for Phase 2 final best model
    if not checkpoint_path:
        candidate = Path(output_dir).parent / "final" / "checkpoints" / "best_model.pt"
        if candidate.exists():
            checkpoint_path = str(candidate)
            logger.info(f"Auto-detected Phase 2 checkpoint: {checkpoint_path}")

    config = SimplexODEConfig(
        lr=lr,
        batch_size=batch_size,
        epochs=epochs,
        output_dir=str(output_path),
        device=device,
        seed=seed,
        use_wandb=use_wandb,
        data_dir=data_dir,
        checkpoint_path=checkpoint_path,
    )

    trainer = SimplexODETrainer(config)
    trainer.setup()
    result = trainer.train()

    logger.info(
        f"Phase 3 complete. Best recon error: {result['best_val_metric']:.6f} "
        f"at epoch {result['best_epoch']}"
    )

    return result


# ---------------------------------------------------------------------------
# Phase 4: Attention analysis
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_phase4(
    data_dir: str,
    output_dir: str,
    checkpoint_path: str,
    device: str = "auto",
    max_features: int = MAX_ASV_FEATURES,
    top_k: int = 50,
) -> Dict[str, Any]:
    """Phase 4: Extract per-class mean attention weights, rank indicator taxa.

    Loads a converged model, runs inference with extract_indicators=True,
    computes per-class mean attention over correctly classified samples,
    and saves indicator species rankings as JSON.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Resolve device
    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    # Load model
    model = MicrobialEncoder(input_dim=max_features)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(model_state, strict=False)
    model.to(dev)
    model.eval()

    # Load dataset
    dataset = MicrobialDataset(data_dir, max_features=max_features)
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    # Accumulate per-class attention weights
    # attention weights are indicator_species_weights [B, n_features]
    class_attention_sum: Dict[int, np.ndarray] = {}
    class_attention_count: Dict[int, int] = {}

    for batch in loader:
        clr = batch["clr"].to(dev)
        labels = batch["source_label"]
        raw = batch.get("raw_abundances")
        if raw is not None:
            raw = raw.to(dev)

        outputs = model(x=clr, raw_abundances=raw, extract_indicators=True)
        preds = outputs["source_logits"].argmax(dim=-1).cpu()
        attn = outputs["indicator_species_weights"].cpu().numpy()

        # Only consider correctly classified samples
        for i in range(len(labels)):
            label = labels[i].item()
            if label < 0:
                continue
            if preds[i].item() != label:
                continue

            if label not in class_attention_sum:
                class_attention_sum[label] = np.zeros(attn.shape[1], dtype=np.float64)
                class_attention_count[label] = 0
            class_attention_sum[label] += attn[i].astype(np.float64)
            class_attention_count[label] += 1

    # Compute per-class mean attention and rank taxa
    indicator_rankings: Dict[str, List[Dict[str, Any]]] = {}
    feature_names = dataset.feature_names

    for cls_idx in sorted(class_attention_sum.keys()):
        count = class_attention_count[cls_idx]
        if count == 0:
            continue

        mean_attn = class_attention_sum[cls_idx] / count
        ranked_indices = np.argsort(-mean_attn)

        cls_name = (
            CONTAMINATION_SOURCES[cls_idx]
            if cls_idx < len(CONTAMINATION_SOURCES)
            else f"class_{cls_idx}"
        )

        top_taxa = []
        for rank, feat_idx in enumerate(ranked_indices[:top_k]):
            feat_name = (
                feature_names[feat_idx]
                if feat_idx < len(feature_names)
                else f"ASV_{feat_idx}"
            )
            top_taxa.append({
                "rank": rank + 1,
                "feature_index": int(feat_idx),
                "feature_name": feat_name,
                "mean_attention_weight": float(mean_attn[feat_idx]),
            })

        indicator_rankings[cls_name] = top_taxa
        logger.info(
            f"Class '{cls_name}' ({count} samples): "
            f"top indicator = {top_taxa[0]['feature_name']} "
            f"(attn={top_taxa[0]['mean_attention_weight']:.4f})"
        )

    # Summary statistics
    summary = {
        "n_classes_analyzed": len(indicator_rankings),
        "samples_per_class": {
            (CONTAMINATION_SOURCES[k] if k < len(CONTAMINATION_SOURCES) else f"class_{k}"): v
            for k, v in class_attention_count.items()
        },
        "top_k": top_k,
        "checkpoint": checkpoint_path,
    }

    # Save results
    rankings_path = output_path / "indicator_species_rankings.json"
    with open(rankings_path, "w", encoding="utf-8") as f:
        json.dump(
            {"summary": summary, "rankings": indicator_rankings},
            f,
            indent=2,
        )
    logger.info(f"Indicator species rankings saved to {rankings_path}")

    # Also save the full mean attention matrices for further analysis
    attention_matrices = {}
    for cls_idx, attn_sum in class_attention_sum.items():
        cls_name = (
            CONTAMINATION_SOURCES[cls_idx]
            if cls_idx < len(CONTAMINATION_SOURCES)
            else f"class_{cls_idx}"
        )
        attention_matrices[cls_name] = (attn_sum / class_attention_count[cls_idx]).tolist()

    attn_path = output_path / "per_class_mean_attention.json"
    with open(attn_path, "w", encoding="utf-8") as f:
        json.dump(attention_matrices, f)
    logger.info(f"Per-class mean attention saved to {attn_path}")

    return {"summary": summary, "rankings_path": str(rankings_path)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SENTINEL MicroBiomeNet training pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--phase",
        type=int,
        required=True,
        choices=[1, 2, 3, 4],
        help="Training phase (1=data prep, 2=source attribution, "
        "3=simplex ODE, 4=attention analysis)",
    )
    parser.add_argument("--data-dir", type=str, default="data/microbial",
                        help="Input data directory")
    parser.add_argument("--output-dir", type=str, default="outputs/microbial",
                        help="Output directory")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)

    # Training hyperparameters
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--no-wandb", action="store_true")

    # Phase 2 specific
    parser.add_argument("--n-folds", type=int, default=5,
                        help="Number of cross-validation folds (Phase 2)")

    # Phase 3 specific
    parser.add_argument("--checkpoint", type=str, default="",
                        help="Path to model checkpoint (Phase 3/4)")

    # Phase 4 specific
    parser.add_argument("--top-k", type=int, default=50,
                        help="Top K indicator taxa to report (Phase 4)")

    return parser


def main() -> None:
    args = build_argparser().parse_args()

    if args.phase == 1:
        run_phase1(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            config_path=args.config,
        )

    elif args.phase == 2:
        run_phase2(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            config_path=args.config,
            n_folds=args.n_folds,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            device=args.device,
            seed=args.seed,
            use_wandb=not args.no_wandb,
        )

    elif args.phase == 3:
        run_phase3(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            checkpoint_path=args.checkpoint,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            device=args.device,
            seed=args.seed,
            use_wandb=not args.no_wandb,
        )

    elif args.phase == 4:
        if not args.checkpoint:
            # Auto-detect from output dir
            candidate = Path(args.output_dir) / "final" / "checkpoints" / "best_model.pt"
            if candidate.exists():
                args.checkpoint = str(candidate)
            else:
                raise ValueError(
                    "--checkpoint is required for Phase 4, "
                    "or a best_model.pt must exist in output_dir/final/checkpoints/"
                )

        run_phase4(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            checkpoint_path=args.checkpoint,
            device=args.device,
            top_k=args.top_k,
        )


if __name__ == "__main__":
    main()
