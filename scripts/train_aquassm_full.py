#!/usr/bin/env python3
"""Full AquaSSM training pipeline: pretrain + anomaly fine-tune + evaluate.

MIT License — Bryan Cheng, 2026
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

from sentinel.models.sensor_encoder import SensorEncoder
from sentinel.models.sensor_encoder.physics_constraints import PhysicsConstraintLoss
from sentinel.utils.logging import get_logger

logger = get_logger(__name__)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = Path("checkpoints/sensor")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


class SensorSequenceDataset(Dataset):
    """Load preprocessed sensor sequences from .npz files."""

    def __init__(self, data_dir: Path, max_len: int = 1024):
        self.files = sorted(data_dir.glob("*.npz"))
        self.max_len = max_len

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        T = min(len(data["values"]), self.max_len)
        values = data["values"][:T].astype(np.float32)
        delta_ts = data["delta_ts"][:T].astype(np.float32)
        mask = data["mask"][:T].astype(np.float32) if "mask" in data else np.ones_like(values)

        # Labels: 0=normal, 1+=anomaly (if available)
        if "labels" in data:
            labels = data["labels"][:T].astype(np.int64)
            has_anomaly = int((labels > 0).any())
        else:
            labels = np.zeros(T, dtype=np.int64)
            has_anomaly = 0

        return {
            "values": torch.tensor(values),
            "delta_ts": torch.tensor(delta_ts),
            "mask": torch.tensor(mask),
            "labels": torch.tensor(labels),
            "has_anomaly": has_anomaly,
        }


def collate_fn(batch):
    """Pad sequences to same length within batch."""
    max_len = max(b["values"].shape[0] for b in batch)

    values = torch.zeros(len(batch), max_len, 6)
    delta_ts = torch.zeros(len(batch), max_len)
    masks = torch.zeros(len(batch), max_len, 6)
    labels = torch.zeros(len(batch), max_len, dtype=torch.long)
    has_anomaly = torch.tensor([b["has_anomaly"] for b in batch])

    for i, b in enumerate(batch):
        T = b["values"].shape[0]
        values[i, :T] = b["values"]
        delta_ts[i, :T] = b["delta_ts"]
        masks[i, :T] = b["mask"]
        labels[i, :T] = b["labels"]

    return {
        "values": values,
        "delta_ts": delta_ts,
        "mask": masks,
        "labels": labels,
        "has_anomaly": has_anomaly,
    }


def train_phase1_pretrain(model, train_dl, val_dl, epochs=50, lr=5e-4):
    """Phase 1: Self-supervised MPP pretraining."""
    logger.info("=" * 60)
    logger.info("PHASE 1: MPP Pretraining")
    logger.info("=" * 60)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    physics_loss_fn = PhysicsConstraintLoss().to(DEVICE)

    best_val_loss = float("inf")
    history = []

    for epoch in range(epochs):
        model.train()
        total_loss, total_mpp, total_phys = 0, 0, 0
        n_batches = 0

        for batch in train_dl:
            values = batch["values"].to(DEVICE)
            delta_ts = batch["delta_ts"].to(DEVICE)

            out = model.forward_pretrain(x=values, delta_ts=delta_ts)
            mpp_loss = out["loss"]

            if torch.isnan(mpp_loss):
                optimizer.zero_grad()
                continue

            # Physics constraints
            pred = out["predictions"]
            pnames = ["do", "ph", "conductivity", "temperature", "turb", "orp"]
            pred_dict = {n: pred[..., i] for i, n in enumerate(pnames)}
            phys_out = physics_loss_fn(pred_dict)
            phys_loss = phys_out["total_loss"].clamp(max=10.0)

            loss = mpp_loss + 0.1 * phys_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_mpp += mpp_loss.item()
            total_phys += phys_loss.item()
            n_batches += 1

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0
        val_n = 0
        with torch.no_grad():
            for batch in val_dl:
                values = batch["values"].to(DEVICE)
                delta_ts = batch["delta_ts"].to(DEVICE)
                out = model.forward_pretrain(x=values, delta_ts=delta_ts)
                if not torch.isnan(out["loss"]):
                    val_loss += out["loss"].item()
                    val_n += 1

        avg_train = total_loss / max(n_batches, 1)
        avg_val = val_loss / max(val_n, 1)

        history.append({
            "epoch": epoch + 1,
            "train_loss": avg_train,
            "train_mpp": total_mpp / max(n_batches, 1),
            "train_phys": total_phys / max(n_batches, 1),
            "val_loss": avg_val,
            "lr": scheduler.get_last_lr()[0],
        })

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                f"Epoch {epoch+1:3d}/{epochs} | Train: {avg_train:.4f} "
                f"(MPP: {total_mpp/max(n_batches,1):.4f}, Phys: {total_phys/max(n_batches,1):.4f}) "
                f"| Val: {avg_val:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}"
            )

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), CHECKPOINT_DIR / "aquassm_pretrained_best.pt")

    torch.save(model.state_dict(), CHECKPOINT_DIR / "aquassm_pretrained_final.pt")
    return history


def train_phase2_anomaly(model, train_dl, val_dl, epochs=30, lr=1e-4):
    """Phase 2: Supervised anomaly detection fine-tuning."""
    logger.info("=" * 60)
    logger.info("PHASE 2: Anomaly Fine-tuning")
    logger.info("=" * 60)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = []
    best_auroc = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0
        all_preds = []
        all_labels = []

        for batch in train_dl:
            values = batch["values"].to(DEVICE)
            delta_ts = batch["delta_ts"].to(DEVICE)
            has_anomaly = batch["has_anomaly"].float().to(DEVICE)

            # Forward pass with anomaly detection
            out = model(values, delta_ts, compute_anomaly=True)

            # Get anomaly score as max normalized error
            if "normalized_errors" in out["anomaly_scores"]:
                anomaly_score = out["anomaly_scores"]["normalized_errors"].mean(dim=-1)  # [B]
            else:
                anomaly_score = torch.zeros(len(values), device=DEVICE)

            # Binary cross-entropy on anomaly score
            anomaly_pred = torch.sigmoid(anomaly_score)
            loss = nn.functional.binary_cross_entropy(
                anomaly_pred, has_anomaly, reduction="mean"
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            all_preds.extend(anomaly_pred.detach().cpu().numpy())
            all_labels.extend(has_anomaly.cpu().numpy())

        scheduler.step()

        # Compute AUROC
        try:
            train_auroc = roc_auc_score(all_labels, all_preds)
        except ValueError:
            train_auroc = 0.5

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_dl:
                values = batch["values"].to(DEVICE)
                delta_ts = batch["delta_ts"].to(DEVICE)
                has_anomaly = batch["has_anomaly"].float()

                out = model(values, delta_ts, compute_anomaly=True)
                if "normalized_errors" in out["anomaly_scores"]:
                    score = out["anomaly_scores"]["normalized_errors"].mean(dim=-1)
                else:
                    score = torch.zeros(len(values))
                val_preds.extend(torch.sigmoid(score).cpu().numpy())
                val_labels.extend(has_anomaly.numpy())

        try:
            val_auroc = roc_auc_score(val_labels, val_preds)
        except ValueError:
            val_auroc = 0.5

        history.append({
            "epoch": epoch + 1,
            "train_loss": total_loss / max(n_batches, 1),
            "train_auroc": train_auroc,
            "val_auroc": val_auroc,
        })

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                f"Epoch {epoch+1:3d}/{epochs} | Loss: {total_loss/max(n_batches,1):.4f} "
                f"| Train AUROC: {train_auroc:.4f} | Val AUROC: {val_auroc:.4f}"
            )

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            torch.save(model.state_dict(), CHECKPOINT_DIR / "aquassm_anomaly_best.pt")

    return history, best_auroc


def evaluate(model, test_dl):
    """Final evaluation on test set."""
    logger.info("=" * 60)
    logger.info("EVALUATION")
    logger.info("=" * 60)

    model.eval()
    all_preds, all_labels = [], []
    embeddings = []

    with torch.no_grad():
        for batch in test_dl:
            values = batch["values"].to(DEVICE)
            delta_ts = batch["delta_ts"].to(DEVICE)
            has_anomaly = batch["has_anomaly"].float()

            out = model(values, delta_ts, compute_anomaly=True)

            if "normalized_errors" in out["anomaly_scores"]:
                score = out["anomaly_scores"]["normalized_errors"].mean(dim=-1)
            else:
                score = torch.zeros(len(values))

            all_preds.extend(torch.sigmoid(score).cpu().numpy())
            all_labels.extend(has_anomaly.numpy())
            embeddings.append(out["embedding"].cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    embeddings = np.concatenate(embeddings)

    try:
        auroc = roc_auc_score(all_labels, all_preds)
        auprc = average_precision_score(all_labels, all_preds)
    except ValueError:
        auroc = auprc = 0.5

    # Binary predictions at 0.5 threshold
    binary_preds = (all_preds > 0.5).astype(int)
    f1 = f1_score(all_labels, binary_preds, zero_division=0)

    results = {
        "auroc": float(auroc),
        "auprc": float(auprc),
        "f1": float(f1),
        "n_test": len(all_labels),
        "n_positive": int(all_labels.sum()),
        "embedding_shape": list(embeddings.shape),
    }

    logger.info(f"Test AUROC: {auroc:.4f}")
    logger.info(f"Test AUPRC: {auprc:.4f}")
    logger.info(f"Test F1:    {f1:.4f}")
    logger.info(f"N={len(all_labels)}, Positive={int(all_labels.sum())}")

    return results


def main():
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load datasets
    real_dir = Path("data/processed/sensor/pretrain")
    synth_dir = Path("data/processed/sensor/synthetic")

    real_ds = SensorSequenceDataset(real_dir, max_len=1024) if real_dir.exists() else None
    synth_ds = SensorSequenceDataset(synth_dir, max_len=1024) if synth_dir.exists() else None

    datasets = []
    if real_ds and len(real_ds) > 0:
        datasets.append(real_ds)
        logger.info(f"Real data: {len(real_ds)} sequences")
    if synth_ds and len(synth_ds) > 0:
        datasets.append(synth_ds)
        logger.info(f"Synthetic data: {len(synth_ds)} sequences")

    if not datasets:
        logger.error("No data found!")
        return

    full_ds = ConcatDataset(datasets)
    n_total = len(full_ds)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    n_test = n_total - n_train - n_val

    train_ds, val_ds, test_ds = random_split(
        full_ds, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    logger.info(f"Total: {n_total} | Train: {n_train} | Val: {n_val} | Test: {n_test}")

    train_dl = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=0)

    # Build model
    model = SensorEncoder().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: SensorEncoder ({n_params:,} parameters)")

    # Phase 1: Pretrain
    phase1_history = train_phase1_pretrain(model, train_dl, val_dl, epochs=50, lr=5e-4)

    # Phase 2: Anomaly fine-tune
    phase2_history, best_auroc = train_phase2_anomaly(model, train_dl, val_dl, epochs=30, lr=1e-4)

    # Evaluate
    test_results = evaluate(model, test_dl)

    # Save all results
    elapsed = time.time() - start_time
    run_results = {
        "timestamp": timestamp,
        "device": str(DEVICE),
        "n_params": n_params,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "phase1_history": phase1_history,
        "phase2_history": phase2_history,
        "test_results": test_results,
        "elapsed_seconds": elapsed,
        "best_val_auroc": best_auroc,
    }

    results_file = CHECKPOINT_DIR / f"run_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(run_results, f, indent=2)

    logger.info(f"\nTotal time: {elapsed/60:.1f} minutes")
    logger.info(f"Results saved to {results_file}")

    # Check against thresholds
    logger.info("\n" + "=" * 60)
    logger.info("THRESHOLD CHECK")
    logger.info("=" * 60)
    auroc = test_results["auroc"]
    if auroc > 0.85:
        logger.info(f"✓ AUROC {auroc:.4f} > 0.85 — HARD THRESHOLD MET")
    elif auroc > 0.70:
        logger.info(f"~ AUROC {auroc:.4f} > 0.70 — ACCEPTABLE (needs iteration)")
    else:
        logger.info(f"✗ AUROC {auroc:.4f} < 0.70 — BELOW THRESHOLD")


if __name__ == "__main__":
    main()
