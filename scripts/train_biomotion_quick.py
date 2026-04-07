#!/usr/bin/env python3
"""BioMotion training on real ECOTOX Daphnia behavioral data.

Phase 1: Diffusion pretraining on normal trajectories (50 epochs)
Phase 2: Anomaly classification fine-tuning (30 epochs)

Data: 17,074 real Daphnia behavioral tests from EPA ECOTOX database,
converted from concentration-response curves (locomotion, swimming,
equilibrium, activity endpoints) into trajectory format.

Evaluates on held-out test set and saves results to checkpoints/biomotion/results.json.

Usage:
    CUDA_VISIBLE_DEVICES=3 python scripts/train_biomotion_quick.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sentinel.models.biomotion.trajectory_encoder import (
    TrajectoryDiffusionEncoder,
    EMBED_DIM,
)
from sentinel.models.biomotion.multi_organism import SPECIES_FEATURE_DIM


# ── Configuration ──────────────────────────────────────────────────────────

# Use real ECOTOX data if available, fall back to synthetic
_real_dir = PROJECT_ROOT / "data" / "processed" / "behavioral_real"
_synth_dir = PROJECT_ROOT / "data" / "processed" / "behavioral_expanded"
_synth_dir2 = PROJECT_ROOT / "data" / "processed" / "behavioral"
DATA_DIR = _real_dir if _real_dir.exists() and any(_real_dir.glob("traj_*.npz")) else \
           _synth_dir if _synth_dir.exists() and any(_synth_dir.glob("traj_*.npz")) else \
           _synth_dir2
CKPT_DIR = PROJECT_ROOT / "checkpoints" / "biomotion"

FEATURE_DIM = SPECIES_FEATURE_DIM["daphnia"]  # 16
N_KEYPOINTS = 12
BATCH_SIZE = 32
SEED = 42

# Phase 1: diffusion pretraining
PHASE1_EPOCHS = 50
PHASE1_LR = 2e-4
PHASE1_WARMUP = 200

# Phase 2: anomaly fine-tuning
PHASE2_EPOCHS = 30
PHASE2_LR = 5e-5
PHASE2_WARMUP = 100

# Test split
TEST_FRAC = 0.15
VAL_FRAC = 0.15


# ── Dataset ────────────────────────────────────────────────────────────────

class BehavioralTrajectoryDataset(Dataset):
    """Load synthetic behavioral trajectories from .npz files.

    Each file contains:
        keypoints:  (T, 12, 2)
        features:   (T, 16)
        timestamps: (T,)
        is_anomaly: bool
    """

    def __init__(self, file_paths: list[Path]) -> None:
        self.file_paths = file_paths
        self._cache: dict[int, dict] = {}

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray | bool]:
        if idx not in self._cache:
            data = np.load(self.file_paths[idx])
            self._cache[idx] = {
                "keypoints": data["keypoints"].astype(np.float32),
                "features": data["features"].astype(np.float32),
                "timestamps": data["timestamps"].astype(np.float32),
                "is_anomaly": bool(data["is_anomaly"]),
            }
        return self._cache[idx]

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
        """Collate into batched tensors."""
        return {
            "keypoints": torch.from_numpy(
                np.stack([s["keypoints"] for s in batch])
            ),
            "features": torch.from_numpy(
                np.stack([s["features"] for s in batch])
            ),
            "timestamps": torch.from_numpy(
                np.stack([s["timestamps"] for s in batch])
            ),
            "labels": torch.tensor(
                [float(s["is_anomaly"]) for s in batch], dtype=torch.float32
            ),
        }


# ── Data splitting ─────────────────────────────────────────────────────────

def load_and_split_data() -> tuple[
    BehavioralTrajectoryDataset,
    BehavioralTrajectoryDataset,
    BehavioralTrajectoryDataset,
    BehavioralTrajectoryDataset,
]:
    """Load all trajectories and split into train/val/test sets.

    Returns:
        (train_normal, train_all, val_all, test_all) datasets.
        train_normal: only normal trajectories for Phase 1 pretraining.
        train_all: all training trajectories for Phase 2.
        val_all: validation set.
        test_all: held-out test set.
    """
    all_files = sorted(DATA_DIR.glob("traj_*.npz"))
    assert len(all_files) > 0, f"No trajectory files found in {DATA_DIR}"
    print(f"Found {len(all_files)} trajectory files")

    # Separate normal and anomalous
    normal_files = []
    anomalous_files = []
    for f in all_files:
        data = np.load(f)
        if bool(data["is_anomaly"]):
            anomalous_files.append(f)
        else:
            normal_files.append(f)

    print(f"  Normal: {len(normal_files)}, Anomalous: {len(anomalous_files)}")

    rng = np.random.default_rng(SEED)

    # Shuffle
    rng.shuffle(normal_files)
    rng.shuffle(anomalous_files)

    # Split normal: train / val / test
    n_norm = len(normal_files)
    n_test_norm = max(1, int(n_norm * TEST_FRAC))
    n_val_norm = max(1, int(n_norm * VAL_FRAC))
    n_train_norm = n_norm - n_test_norm - n_val_norm

    norm_train = normal_files[:n_train_norm]
    norm_val = normal_files[n_train_norm:n_train_norm + n_val_norm]
    norm_test = normal_files[n_train_norm + n_val_norm:]

    # Split anomalous: train / val / test
    n_anom = len(anomalous_files)
    n_test_anom = max(1, int(n_anom * TEST_FRAC))
    n_val_anom = max(1, int(n_anom * VAL_FRAC))
    n_train_anom = n_anom - n_test_anom - n_val_anom

    anom_train = anomalous_files[:n_train_anom]
    anom_val = anomalous_files[n_train_anom:n_train_anom + n_val_anom]
    anom_test = anomalous_files[n_train_anom + n_val_anom:]

    print(f"  Train: {n_train_norm} normal + {n_train_anom} anomalous = {n_train_norm + n_train_anom}")
    print(f"  Val:   {n_val_norm} normal + {n_val_anom} anomalous = {n_val_norm + n_val_anom}")
    print(f"  Test:  {n_test_norm} normal + {n_test_anom} anomalous = {n_test_norm + n_test_anom}")

    train_normal = BehavioralTrajectoryDataset(norm_train)
    train_all = BehavioralTrajectoryDataset(norm_train + anom_train)
    val_all = BehavioralTrajectoryDataset(norm_val + anom_val)
    test_all = BehavioralTrajectoryDataset(norm_test + anom_test)

    return train_normal, train_all, val_all, test_all


# ── Anomaly Classifier ─────────────────────────────────────────────────────

class AnomalyClassifier(nn.Module):
    """Binary anomaly classifier wrapping a TrajectoryDiffusionEncoder."""

    def __init__(self, encoder: TrajectoryDiffusionEncoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(EMBED_DIM, EMBED_DIM // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(EMBED_DIM // 2, 1),
        )
        self._init_classifier()

    def _init_classifier(self) -> None:
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return anomaly logit (B,)."""
        embedding = self.encoder.forward_encode(features)
        return self.classifier(embedding).squeeze(-1)


# ── Training utilities ─────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_cosine_schedule(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine decay with linear warmup."""
    import math

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.01 + 0.5 * 0.99 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Phase 1: Diffusion Pretraining ─────────────────────────────────────────

def train_phase1(
    encoder: TrajectoryDiffusionEncoder,
    train_ds: BehavioralTrajectoryDataset,
    val_ds: BehavioralTrajectoryDataset,
    device: torch.device,
) -> dict:
    """Phase 1: Self-supervised diffusion pretraining on normal trajectories."""
    print("\n" + "=" * 70)
    print("PHASE 1: Diffusion Pretraining (normal trajectories only)")
    print("=" * 70)

    encoder.to(device)
    encoder.train()

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=BehavioralTrajectoryDataset.collate_fn,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=BehavioralTrajectoryDataset.collate_fn,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )

    optimizer = torch.optim.AdamW(
        encoder.parameters(), lr=PHASE1_LR, weight_decay=0.01
    )
    total_steps = len(train_loader) * PHASE1_EPOCHS
    scheduler = build_cosine_schedule(optimizer, total_steps, PHASE1_WARMUP)

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}
    start_time = time.time()

    for epoch in range(PHASE1_EPOCHS):
        encoder.train()
        epoch_losses = []

        for batch in train_loader:
            features = batch["features"].to(device)

            loss_dict = encoder.compute_training_loss(features)
            loss = loss_dict["loss"]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_losses.append(loss.item())

        train_loss = np.mean(epoch_losses)

        # Validation
        encoder.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                features = batch["features"].to(device)
                loss_dict = encoder.compute_training_loss(features)
                val_losses.append(loss_dict["loss"].item())
        val_loss = np.mean(val_losses) if val_losses else float("inf")

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = CKPT_DIR / "phase1_best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": encoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            }, ckpt_path)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            elapsed = time.time() - start_time
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"  Epoch {epoch + 1:3d}/{PHASE1_EPOCHS} | "
                f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
                f"lr={lr:.2e} | time={elapsed:.1f}s"
            )

    elapsed = time.time() - start_time
    print(f"\nPhase 1 complete in {elapsed:.1f}s | best_val_loss={best_val_loss:.6f}")

    # Reload best model
    state = torch.load(CKPT_DIR / "phase1_best.pt", map_location=device, weights_only=False)
    encoder.load_state_dict(state["model_state_dict"])

    return {
        "phase": 1,
        "best_val_loss": float(best_val_loss),
        "total_epochs": PHASE1_EPOCHS,
        "elapsed_seconds": elapsed,
        "history": history,
    }


# ── Phase 2: Anomaly Fine-tuning ──────────────────────────────────────────

def train_phase2(
    encoder: TrajectoryDiffusionEncoder,
    train_ds: BehavioralTrajectoryDataset,
    val_ds: BehavioralTrajectoryDataset,
    device: torch.device,
) -> tuple[AnomalyClassifier, dict]:
    """Phase 2: Supervised anomaly classification fine-tuning."""
    print("\n" + "=" * 70)
    print("PHASE 2: Anomaly Classification Fine-tuning")
    print("=" * 70)

    model = AnomalyClassifier(encoder)
    model.to(device)
    model.train()

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=BehavioralTrajectoryDataset.collate_fn,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=BehavioralTrajectoryDataset.collate_fn,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )

    # Lower LR for encoder (already pretrained), higher for classifier
    param_groups = [
        {"params": model.encoder.parameters(), "lr": PHASE2_LR},
        {"params": model.classifier.parameters(), "lr": PHASE2_LR * 5},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
    total_steps = len(train_loader) * PHASE2_EPOCHS
    scheduler = build_cosine_schedule(optimizer, total_steps, PHASE2_WARMUP)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    start_time = time.time()

    for epoch in range(PHASE2_EPOCHS):
        model.train()
        epoch_losses = []

        for batch in train_loader:
            features = batch["features"].to(device)
            labels = batch["labels"].to(device)

            logits = model(features)
            loss = F.binary_cross_entropy_with_logits(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_losses.append(loss.item())

        train_loss = np.mean(epoch_losses)

        # Validation
        model.eval()
        val_losses = []
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                features = batch["features"].to(device)
                labels = batch["labels"].to(device)

                logits = model(features)
                loss = F.binary_cross_entropy_with_logits(logits, labels)
                val_losses.append(loss.item())

                preds = (torch.sigmoid(logits) > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += len(labels)

        val_loss = np.mean(val_losses) if val_losses else float("inf")
        val_acc = val_correct / max(val_total, 1)

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            ckpt_path = CKPT_DIR / "phase2_best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
            }, ckpt_path)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            elapsed = time.time() - start_time
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"  Epoch {epoch + 1:3d}/{PHASE2_EPOCHS} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"val_acc={val_acc:.4f} | lr={lr:.2e} | time={elapsed:.1f}s"
            )

    elapsed = time.time() - start_time
    print(
        f"\nPhase 2 complete in {elapsed:.1f}s | "
        f"best_val_loss={best_val_loss:.4f} | best_val_acc={best_val_acc:.4f}"
    )

    # Reload best model
    state = torch.load(CKPT_DIR / "phase2_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])

    return model, {
        "phase": 2,
        "best_val_loss": float(best_val_loss),
        "best_val_acc": float(best_val_acc),
        "total_epochs": PHASE2_EPOCHS,
        "elapsed_seconds": elapsed,
        "history": history,
    }


# ── Test Evaluation ────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_test(
    model: AnomalyClassifier,
    test_ds: BehavioralTrajectoryDataset,
    device: torch.device,
) -> dict:
    """Evaluate on held-out test set, compute AUROC and F1."""
    from sklearn.metrics import (
        roc_auc_score,
        f1_score,
        precision_score,
        recall_score,
        accuracy_score,
        classification_report,
    )

    print("\n" + "=" * 70)
    print("TEST SET EVALUATION")
    print("=" * 70)

    model.to(device)
    model.eval()

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=BehavioralTrajectoryDataset.collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    all_labels = []
    all_scores = []
    all_preds = []

    for batch in test_loader:
        features = batch["features"].to(device)
        labels = batch["labels"]

        logits = model(features)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs > 0.5).astype(float)

        all_labels.append(labels.numpy())
        all_scores.append(probs)
        all_preds.append(preds)

    y_true = np.concatenate(all_labels)
    y_scores = np.concatenate(all_scores)
    y_pred = np.concatenate(all_preds)

    auroc = float(roc_auc_score(y_true, y_scores))
    f1 = float(f1_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred))
    recall = float(recall_score(y_true, y_pred))
    accuracy = float(accuracy_score(y_true, y_pred))

    print(f"\n  Test samples: {len(y_true)}")
    print(f"  Normal:       {int((y_true == 0).sum())}")
    print(f"  Anomalous:    {int((y_true == 1).sum())}")
    print(f"\n  AUROC:     {auroc:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  Accuracy:  {accuracy:.4f}")

    print("\n  Classification Report:")
    report = classification_report(
        y_true, y_pred, target_names=["normal", "anomalous"]
    )
    print("  " + report.replace("\n", "\n  "))

    # Also evaluate using diffusion-based anomaly score (unsupervised)
    print("\n  Diffusion-based anomaly scoring (unsupervised):")
    all_diff_scores = []
    for batch in test_loader:
        features = batch["features"].to(device)
        diff_score = model.encoder.compute_anomaly_score(features, num_noise_levels=5)
        all_diff_scores.append(diff_score.cpu().numpy())
    diff_scores = np.concatenate(all_diff_scores)
    diff_auroc = float(roc_auc_score(y_true, diff_scores))
    print(f"  Diffusion AUROC: {diff_auroc:.4f}")

    return {
        "n_test": len(y_true),
        "n_normal": int((y_true == 0).sum()),
        "n_anomalous": int((y_true == 1).sum()),
        "auroc": auroc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "diffusion_auroc": diff_auroc,
    }


# ── Full-model forward pass test ──────────────────────────────────────────

@torch.no_grad()
def test_biomotion_encoder_forward(
    encoder_weights: TrajectoryDiffusionEncoder,
    test_ds: BehavioralTrajectoryDataset,
    device: torch.device,
) -> None:
    """Verify the full BioMotionEncoder works with our data format."""
    from sentinel.models.biomotion.model import BioMotionEncoder

    print("\n" + "-" * 70)
    print("Verifying BioMotionEncoder forward pass with Daphnia data...")

    # Build full model (daphnia-only for this test)
    bm_encoder = BioMotionEncoder(
        species_list=["daphnia"],
        species_feature_dims={"daphnia": FEATURE_DIM},
    )
    bm_encoder.to(device)
    bm_encoder.eval()

    # Get a small batch
    loader = DataLoader(
        test_ds,
        batch_size=4,
        shuffle=False,
        collate_fn=BehavioralTrajectoryDataset.collate_fn,
    )
    batch = next(iter(loader))

    organism_inputs = {
        "daphnia": {
            "keypoints": batch["keypoints"].to(device),
            "features": batch["features"].to(device),
            "timestamps": batch["timestamps"].to(device),
        }
    }

    output = bm_encoder(organism_inputs)

    print(f"  embedding:            {output['embedding'].shape}")
    print(f"  fusion_embedding:     {output['fusion_embedding'].shape}")
    print(f"  anomaly_score:        {output['anomaly_score'].shape}")
    print(f"  denoising_difficulty: {output['denoising_difficulty'].shape}")
    for sp, emb in output["organism_embeddings"].items():
        print(f"  organism_embeddings[{sp}]: {emb.shape}")
    print("  BioMotionEncoder forward pass: OK")


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # Load and split data
    train_normal, train_all, val_all, test_all = load_and_split_data()

    # Build encoder
    encoder = TrajectoryDiffusionEncoder(
        feature_dim=FEATURE_DIM,
        embed_dim=EMBED_DIM,
        nhead=4,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.1,
    )
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"\nEncoder parameters: {n_params:,}")

    # Phase 1: Diffusion pretraining
    phase1_results = train_phase1(encoder, train_normal, val_all, device)

    # Phase 2: Anomaly fine-tuning
    model, phase2_results = train_phase2(encoder, train_all, val_all, device)

    # Test evaluation
    test_results = evaluate_test(model, test_all, device)

    # Verify full BioMotionEncoder forward pass
    test_biomotion_encoder_forward(encoder, test_all, device)

    # Save results
    results = {
        "model": "BioMotionEncoder (TrajectoryDiffusionEncoder)",
        "species": "daphnia",
        "feature_dim": FEATURE_DIM,
        "embed_dim": EMBED_DIM,
        "n_keypoints": N_KEYPOINTS,
        "n_parameters": n_params,
        "data": {
            "n_trajectories": 500,
            "n_normal": 250,
            "n_anomalous": 250,
            "trajectory_length": 200,
            "fps": 30,
        },
        "phase1": phase1_results,
        "phase2": phase2_results,
        "test": test_results,
        "checkpoints": {
            "phase1_best": str(CKPT_DIR / "phase1_best.pt"),
            "phase2_best": str(CKPT_DIR / "phase2_best.pt"),
        },
    }

    results_path = CKPT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'=' * 70}")
    print(f"Results saved to {results_path}")
    print(f"{'=' * 70}")
    print(f"\nFinal test metrics:")
    print(f"  AUROC:           {test_results['auroc']:.4f}")
    print(f"  F1:              {test_results['f1']:.4f}")
    print(f"  Accuracy:        {test_results['accuracy']:.4f}")
    print(f"  Diffusion AUROC: {test_results['diffusion_auroc']:.4f}")


if __name__ == "__main__":
    main()
