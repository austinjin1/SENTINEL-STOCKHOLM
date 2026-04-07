#!/usr/bin/env python3
"""Phase 2: Anomaly detection fine-tuning for AquaSSM.

Uses the pretrained backbone embedding + a learnable classification head.
Loads the Phase 1 checkpoint and fine-tunes for binary anomaly detection.

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
from sentinel.utils.logging import get_logger

logger = get_logger(__name__)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = Path("checkpoints/sensor")


class SensorSequenceDataset(Dataset):
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

        if "labels" in data:
            labels = data["labels"][:T].astype(np.int64)
            has_anomaly = int((labels > 0).any())
        else:
            has_anomaly = 0

        return {
            "values": torch.tensor(values),
            "delta_ts": torch.tensor(delta_ts),
            "has_anomaly": has_anomaly,
        }


def collate_fn(batch):
    max_len = max(b["values"].shape[0] for b in batch)
    B = len(batch)
    values = torch.zeros(B, max_len, 6)
    delta_ts = torch.zeros(B, max_len)
    has_anomaly = torch.tensor([b["has_anomaly"] for b in batch])

    for i, b in enumerate(batch):
        T = b["values"].shape[0]
        values[i, :T] = b["values"]
        delta_ts[i, :T] = b["delta_ts"]

    return {"values": values, "delta_ts": delta_ts, "has_anomaly": has_anomaly}


class AnomalyHead(nn.Module):
    """Learnable head on top of frozen/fine-tuned SSM backbone."""
    def __init__(self, embed_dim=256):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, embedding):
        return self.head(embedding).squeeze(-1)  # [B]


def main():
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load data
    real_dir = Path("data/processed/sensor/pretrain")
    synth_dir = Path("data/processed/sensor/synthetic")

    datasets = []
    if real_dir.exists():
        ds = SensorSequenceDataset(real_dir)
        if len(ds) > 0:
            datasets.append(ds)
            logger.info(f"Real data: {len(ds)} sequences")
    if synth_dir.exists():
        ds = SensorSequenceDataset(synth_dir)
        if len(ds) > 0:
            datasets.append(ds)
            logger.info(f"Synthetic data: {len(ds)} sequences")

    full_ds = ConcatDataset(datasets)
    n = len(full_ds)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    n_test = n - n_train - n_val

    train_ds, val_ds, test_ds = random_split(
        full_ds, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    train_dl = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=0)

    logger.info(f"Train: {n_train} | Val: {n_val} | Test: {n_test}")

    # Load pretrained model
    model = SensorEncoder().to(DEVICE)
    ckpt_path = CHECKPOINT_DIR / "aquassm_pretrained_best.pt"
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        logger.info(f"Loaded pretrained checkpoint: {ckpt_path}")
    else:
        logger.warning("No pretrained checkpoint found — training from scratch")

    # Add anomaly classification head
    anomaly_head = AnomalyHead(embed_dim=256).to(DEVICE)

    # Optimizer: fine-tune backbone with lower LR, head with higher LR
    optimizer = torch.optim.AdamW([
        {"params": model.parameters(), "lr": 1e-5},
        {"params": anomaly_head.parameters(), "lr": 1e-3},
    ], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    logger.info("=" * 60)
    logger.info("PHASE 2: Anomaly Fine-tuning")
    logger.info("=" * 60)

    best_auroc = 0
    history = []

    for epoch in range(30):
        model.train()
        anomaly_head.train()
        total_loss, n_batches = 0, 0
        all_preds, all_labels = [], []

        for batch in train_dl:
            values = batch["values"].to(DEVICE)
            delta_ts = batch["delta_ts"].to(DEVICE)
            has_anomaly = batch["has_anomaly"].float().to(DEVICE)

            # Get embedding from backbone (with gradients)
            out = model(values, delta_ts, compute_anomaly=False)
            embedding = out["embedding"]  # [B, 256]

            # Anomaly prediction from head
            logit = anomaly_head(embedding)  # [B]
            pred = torch.sigmoid(logit)

            loss = nn.functional.binary_cross_entropy_with_logits(
                logit, has_anomaly, reduction="mean"
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(anomaly_head.parameters()), 1.0
            )
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            all_preds.extend(pred.detach().cpu().numpy())
            all_labels.extend(has_anomaly.cpu().numpy())

        scheduler.step()

        try:
            train_auroc = roc_auc_score(all_labels, all_preds)
        except ValueError:
            train_auroc = 0.5

        # Validation
        model.eval()
        anomaly_head.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_dl:
                values = batch["values"].to(DEVICE)
                delta_ts = batch["delta_ts"].to(DEVICE)
                has_anomaly = batch["has_anomaly"].float()

                out = model(values, delta_ts, compute_anomaly=False)
                logit = anomaly_head(out["embedding"])
                val_preds.extend(torch.sigmoid(logit).cpu().numpy())
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
                f"Epoch {epoch+1:3d}/30 | Loss: {total_loss/max(n_batches,1):.4f} "
                f"| Train AUROC: {train_auroc:.4f} | Val AUROC: {val_auroc:.4f}"
            )

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            torch.save({
                "model": model.state_dict(),
                "anomaly_head": anomaly_head.state_dict(),
            }, CHECKPOINT_DIR / "aquassm_anomaly_best.pt")

    # Test evaluation
    logger.info("=" * 60)
    logger.info("TEST EVALUATION")
    logger.info("=" * 60)

    model.eval()
    anomaly_head.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for batch in test_dl:
            values = batch["values"].to(DEVICE)
            delta_ts = batch["delta_ts"].to(DEVICE)
            has_anomaly = batch["has_anomaly"].float()

            out = model(values, delta_ts, compute_anomaly=False)
            logit = anomaly_head(out["embedding"])
            test_preds.extend(torch.sigmoid(logit).cpu().numpy())
            test_labels.extend(has_anomaly.numpy())

    test_preds = np.array(test_preds)
    test_labels = np.array(test_labels)

    try:
        auroc = roc_auc_score(test_labels, test_preds)
        auprc = average_precision_score(test_labels, test_preds)
    except ValueError:
        auroc = auprc = 0.5

    binary = (test_preds > 0.5).astype(int)
    f1 = f1_score(test_labels, binary, zero_division=0)

    logger.info(f"Test AUROC: {auroc:.4f}")
    logger.info(f"Test AUPRC: {auprc:.4f}")
    logger.info(f"Test F1:    {f1:.4f}")
    logger.info(f"N={len(test_labels)}, Positive={int(test_labels.sum())}")

    # Threshold check
    logger.info("\n" + "=" * 60)
    logger.info("THRESHOLD CHECK")
    logger.info("=" * 60)
    if auroc > 0.85:
        logger.info(f"HARD THRESHOLD MET: AUROC {auroc:.4f} > 0.85")
    elif auroc > 0.70:
        logger.info(f"ACCEPTABLE: AUROC {auroc:.4f} > 0.70 (needs iteration)")
    else:
        logger.info(f"BELOW THRESHOLD: AUROC {auroc:.4f} < 0.70")

    # Save results
    elapsed = time.time() - start_time
    results = {
        "timestamp": timestamp,
        "best_val_auroc": best_auroc,
        "test_auroc": auroc,
        "test_auprc": auprc,
        "test_f1": f1,
        "n_test": len(test_labels),
        "elapsed_seconds": elapsed,
        "history": history,
    }
    with open(CHECKPOINT_DIR / f"phase2_{timestamp}.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nTotal time: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
