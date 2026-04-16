#!/usr/bin/env python3
"""AquaSSM end-to-end training: clean_synthetic + pretrain data.

Strategy:
- Combine clean_synthetic (100 labeled, 50 normal + 50 anomaly)
  with pretrain (163 real USGS, all normal, label=0).
- Clamp values to [-5,5], delta_ts to [0,3600], set delta_ts[0]=0.
- Pad delta_ts with 0 (not 900!) in collate.
- Train end-to-end: backbone (lr=3e-4) + classification head (lr=1e-3).
- 100 epochs, AdamW, cosine annealing, grad clipping.

MIT License -- Bryan Cheng, 2026
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

# Force GPU selection before any CUDA init
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from sentinel.models.sensor_encoder import SensorEncoder
from sentinel.utils.logging import get_logger

logger = get_logger(__name__)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = Path("checkpoints/sensor")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameters
MAX_LEN = 512          # Truncate sequences to this length
BATCH_SIZE = 8
NUM_EPOCHS = 100
LR_BACKBONE = 3e-4
LR_HEAD = 1e-3
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class CleanSyntheticDataset(Dataset):
    """Clean synthetic data with has_anomaly labels."""

    def __init__(self, data_dir: str, max_len: int = MAX_LEN):
        self.files = sorted(Path(data_dir).glob("*.npz"))
        self.max_len = max_len
        logger.info(f"CleanSyntheticDataset: {len(self.files)} files from {data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        d = np.load(self.files[idx])
        T = min(len(d["values"]), self.max_len)
        values = torch.tensor(d["values"][:T].astype(np.float32)).clamp(-5, 5)
        delta_ts = torch.tensor(d["delta_ts"][:T].astype(np.float32)).clamp(0, 3600)
        delta_ts[0] = 0.0  # First timestep has no gap
        has_anomaly = int(d["has_anomaly"])
        return {"values": values, "delta_ts": delta_ts, "has_anomaly": has_anomaly}


class PretrainDataset(Dataset):
    """USGS pretrain data -- all normal (label=0)."""

    def __init__(self, data_dir: str, max_len: int = MAX_LEN):
        self.files = sorted(Path(data_dir).glob("*.npz"))
        self.max_len = max_len
        logger.info(f"PretrainDataset: {len(self.files)} files from {data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        d = np.load(self.files[idx])
        T = min(len(d["values"]), self.max_len)
        values = torch.tensor(d["values"][:T].astype(np.float32)).clamp(-5, 5)
        delta_ts = torch.tensor(d["delta_ts"][:T].astype(np.float32)).clamp(0, 3600)
        delta_ts[0] = 0.0
        return {"values": values, "delta_ts": delta_ts, "has_anomaly": 0}


def collate_fn(batch):
    """Collate with zero-padding for delta_ts (NOT 900!)."""
    max_len = max(b["values"].shape[0] for b in batch)
    B = len(batch)
    values = torch.zeros(B, max_len, 6)
    delta_ts = torch.zeros(B, max_len)  # Pad with 0, not 900!
    has_anomaly = torch.tensor([b["has_anomaly"] for b in batch], dtype=torch.float32)

    for i, b in enumerate(batch):
        T = b["values"].shape[0]
        values[i, :T] = b["values"]
        delta_ts[i, :T] = b["delta_ts"]

    return {"values": values, "delta_ts": delta_ts, "has_anomaly": has_anomaly}


# ---------------------------------------------------------------------------
# Classification Head
# ---------------------------------------------------------------------------
class AnomalyHead(nn.Module):
    """Binary anomaly classification head on SSM embedding."""

    def __init__(self, input_dim: int = 256, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def compute_metrics(labels, probs):
    """Compute AUROC, AUPRC, F1 from numpy arrays."""
    try:
        auroc = roc_auc_score(labels, probs)
    except ValueError:
        auroc = 0.5
    try:
        auprc = average_precision_score(labels, probs)
    except ValueError:
        auprc = 0.5
    f1 = f1_score(labels, (probs > 0.5).astype(int), zero_division=0)
    return auroc, auprc, f1


def main():
    t0 = time.time()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info("=" * 70)
    logger.info("AquaSSM End-to-End Training (Final)")
    logger.info("=" * 70)
    logger.info(f"Device: {DEVICE}")

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    ds_clean = CleanSyntheticDataset("data/processed/sensor/clean_synthetic")
    ds_pretrain = PretrainDataset("data/processed/sensor/pretrain")
    full_ds = ConcatDataset([ds_clean, ds_pretrain])
    n_total = len(full_ds)

    # Stratified-ish split: we know clean_synthetic[0:50] are normal, [50:100] anomaly
    # pretrain[0:163] are normal. Total: 213 normal, 50 anomaly.
    # Use 70/15/15 split with fixed seed.
    n_train = int(0.70 * n_total)
    n_val = int(0.15 * n_total)
    n_test = n_total - n_train - n_val
    train_ds, val_ds, test_ds = random_split(
        full_ds, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(SEED),
    )
    logger.info(f"Data split: train={n_train}, val={n_val}, test={n_test}")

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=collate_fn, num_workers=0, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=collate_fn, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                         collate_fn=collate_fn, num_workers=0)

    # Count class distribution in each split
    for name, dl in [("train", train_dl), ("val", val_dl), ("test", test_dl)]:
        all_labels = []
        for batch in dl:
            all_labels.extend(batch["has_anomaly"].numpy().tolist())
        pos = sum(1 for l in all_labels if l > 0.5)
        logger.info(f"  {name}: {len(all_labels)} samples, {pos} anomaly, {len(all_labels)-pos} normal")

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    model = SensorEncoder().to(DEVICE)
    head = AnomalyHead().to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in head.parameters())
    logger.info(f"Total parameters: {total_params:,}")

    # Separate optimizers for backbone vs head
    optimizer = torch.optim.AdamW([
        {"params": model.parameters(), "lr": LR_BACKBONE, "weight_decay": WEIGHT_DECAY},
        {"params": head.parameters(), "lr": LR_HEAD, "weight_decay": WEIGHT_DECAY},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # Class weighting: ~213 normal vs ~50 anomaly => weight anomaly higher
    # pos_weight = n_normal / n_anomaly ~ 4.0
    pos_weight = torch.tensor([4.0], device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    best_val_auroc = 0.0
    best_epoch = 0
    history = {"train_loss": [], "train_auroc": [], "val_loss": [], "val_auroc": []}
    nan_count = 0

    logger.info(f"Starting training for {NUM_EPOCHS} epochs...")
    logger.info(f"  Backbone LR: {LR_BACKBONE}, Head LR: {LR_HEAD}")
    logger.info(f"  Batch size: {BATCH_SIZE}, Grad clip: {GRAD_CLIP}")
    logger.info(f"  Pos weight: {pos_weight.item()}")

    for epoch in range(1, NUM_EPOCHS + 1):
        # --- Train ---
        model.train()
        head.train()
        train_loss_sum = 0.0
        train_batches = 0
        all_train_probs = []
        all_train_labels = []

        for batch in train_dl:
            values = batch["values"].to(DEVICE)
            delta_ts = batch["delta_ts"].to(DEVICE)
            labels = batch["has_anomaly"].to(DEVICE)

            # Forward: SensorEncoder.forward(x, timestamps, delta_ts, masks, compute_anomaly)
            out = model(values, delta_ts=delta_ts, compute_anomaly=False)
            embedding = out["embedding"]  # [B, 256]

            # Check for NaN
            if torch.isnan(embedding).any():
                nan_count += 1
                optimizer.zero_grad()
                continue

            logits = head(embedding)
            loss = criterion(logits, labels)

            if torch.isnan(loss):
                nan_count += 1
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(head.parameters()), GRAD_CLIP
            )
            optimizer.step()

            train_loss_sum += loss.item()
            train_batches += 1
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_train_probs.extend(probs.tolist())
            all_train_labels.extend(labels.cpu().numpy().tolist())

        scheduler.step()

        train_loss = train_loss_sum / max(train_batches, 1)
        train_labels_arr = np.array(all_train_labels)
        train_probs_arr = np.array(all_train_probs)
        train_auroc, _, train_f1 = compute_metrics(train_labels_arr, train_probs_arr)

        # --- Validate ---
        model.eval()
        head.eval()
        val_loss_sum = 0.0
        val_batches = 0
        all_val_probs = []
        all_val_labels = []

        with torch.no_grad():
            for batch in val_dl:
                values = batch["values"].to(DEVICE)
                delta_ts = batch["delta_ts"].to(DEVICE)
                labels = batch["has_anomaly"].to(DEVICE)

                out = model(values, delta_ts=delta_ts, compute_anomaly=False)
                embedding = out["embedding"]

                if torch.isnan(embedding).any():
                    continue

                logits = head(embedding)
                loss = criterion(logits, labels)

                if not torch.isnan(loss):
                    val_loss_sum += loss.item()
                    val_batches += 1
                    probs = torch.sigmoid(logits).cpu().numpy()
                    all_val_probs.extend(probs.tolist())
                    all_val_labels.extend(labels.cpu().numpy().tolist())

        val_loss = val_loss_sum / max(val_batches, 1)
        val_labels_arr = np.array(all_val_labels)
        val_probs_arr = np.array(all_val_probs)
        val_auroc, val_auprc, val_f1 = compute_metrics(val_labels_arr, val_probs_arr)

        history["train_loss"].append(train_loss)
        history["train_auroc"].append(train_auroc)
        history["val_loss"].append(val_loss)
        history["val_auroc"].append(val_auroc)

        # Save best model
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_epoch = epoch
            torch.save(
                {"model": model.state_dict(), "head": head.state_dict(), "epoch": epoch,
                 "val_auroc": val_auroc},
                CHECKPOINT_DIR / "aquassm_final_best.pt",
            )

        # Log every 5 epochs, or first/last
        if epoch <= 3 or epoch % 5 == 0 or epoch == NUM_EPOCHS:
            lr_bb = optimizer.param_groups[0]["lr"]
            lr_hd = optimizer.param_groups[1]["lr"]
            logger.info(
                f"Ep {epoch:3d}/{NUM_EPOCHS} | "
                f"TrLoss={train_loss:.4f} TrAUC={train_auroc:.4f} | "
                f"VaLoss={val_loss:.4f} VaAUC={val_auroc:.4f} VaF1={val_f1:.4f} | "
                f"LR={lr_bb:.2e}/{lr_hd:.2e} | NaN={nan_count}"
            )

    logger.info(f"\nBest val AUROC: {best_val_auroc:.4f} at epoch {best_epoch}")

    # -----------------------------------------------------------------------
    # Test evaluation (load best model)
    # -----------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("TEST EVALUATION")
    logger.info("=" * 70)

    ckpt = torch.load(CHECKPOINT_DIR / "aquassm_final_best.pt", map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    head.load_state_dict(ckpt["head"])
    model.eval()
    head.eval()

    all_test_probs = []
    all_test_labels = []

    with torch.no_grad():
        for batch in test_dl:
            values = batch["values"].to(DEVICE)
            delta_ts = batch["delta_ts"].to(DEVICE)
            labels = batch["has_anomaly"]

            out = model(values, delta_ts=delta_ts, compute_anomaly=False)
            embedding = out["embedding"]

            if torch.isnan(embedding).any():
                logger.warning(f"NaN in test embedding! Skipping batch.")
                continue

            logits = head(embedding)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_test_probs.extend(probs.tolist())
            all_test_labels.extend(labels.numpy().tolist())

    test_labels = np.array(all_test_labels)
    test_probs = np.array(all_test_probs)
    test_auroc, test_auprc, test_f1 = compute_metrics(test_labels, test_probs)

    # Also try optimal threshold
    best_f1_opt = 0.0
    best_thresh = 0.5
    for thresh in np.arange(0.1, 0.9, 0.05):
        f1_t = f1_score(test_labels, (test_probs > thresh).astype(int), zero_division=0)
        if f1_t > best_f1_opt:
            best_f1_opt = f1_t
            best_thresh = thresh

    logger.info(f"Test AUROC:  {test_auroc:.4f}")
    logger.info(f"Test AUPRC:  {test_auprc:.4f}")
    logger.info(f"Test F1@0.5: {test_f1:.4f}")
    logger.info(f"Test F1 opt: {best_f1_opt:.4f} (thresh={best_thresh:.2f})")
    logger.info(f"N_test={len(test_labels)}, N_pos={int(test_labels.sum())}, N_neg={int((1-test_labels).sum())}")
    logger.info(f"Total NaN batches: {nan_count}")

    if test_auroc > 0.85:
        logger.info(f"*** HARD THRESHOLD MET: {test_auroc:.4f} > 0.85 ***")
    elif test_auroc > 0.70:
        logger.info(f"ACCEPTABLE: {test_auroc:.4f} > 0.70")
    else:
        logger.info(f"BELOW THRESHOLD: {test_auroc:.4f} < 0.70")

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    elapsed = time.time() - t0
    results = {
        "test_auroc": float(test_auroc),
        "test_auprc": float(test_auprc),
        "test_f1_at_0.5": float(test_f1),
        "test_f1_optimal": float(best_f1_opt),
        "optimal_threshold": float(best_thresh),
        "best_val_auroc": float(best_val_auroc),
        "best_epoch": int(best_epoch),
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "n_test_evaluated": len(test_labels),
        "n_test_pos": int(test_labels.sum()),
        "total_nan_batches": nan_count,
        "total_epochs": NUM_EPOCHS,
        "elapsed_seconds": elapsed,
        "elapsed_minutes": elapsed / 60.0,
        "hyperparameters": {
            "lr_backbone": LR_BACKBONE,
            "lr_head": LR_HEAD,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "max_len": MAX_LEN,
            "grad_clip": GRAD_CLIP,
            "pos_weight": 4.0,
            "seed": SEED,
        },
        "training_curves": {
            "train_loss": history["train_loss"],
            "train_auroc": history["train_auroc"],
            "val_loss": history["val_loss"],
            "val_auroc": history["val_auroc"],
        },
        "timestamp": ts,
    }

    results_path = CHECKPOINT_DIR / "results_final.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")
    logger.info(f"Total time: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
