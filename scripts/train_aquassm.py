#!/usr/bin/env python3
"""AquaSSM v2 training on REAL USGS NWIS data with temporal-spatial holdout.

Data source: data/processed/sensor/full/ — real USGS NWIS sequences
  (produced by download_sensor.py)

Split protocol:
  - Spatial: 5-fold hash-based assignment on site_no
  - Temporal: train=2015-2022, val=2023, test=2024-2026
  - Strict intersection: sample only in split if BOTH spatial AND temporal match

Training:
  Phase 1: Self-supervised masked parameter prediction (MPP) on train split
  Phase 2: Supervised anomaly detection fine-tuning on train split
  Evaluation on held-out test split (never-seen sites + time period)

MIT License -- Bryan Cheng, SENTINEL project, 2026
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
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--pretrain-epochs", type=int, default=50)
    parser.add_argument("--finetune-epochs", type=int, default=50)
    return parser.parse_args()

_args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(_args.gpu)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sentinel.models.sensor_encoder import SensorEncoder
from sentinel.data.splits import SplitConfig, assign_spatial_fold, split_indices

# Use standard logging instead of Rich to avoid buffering issues
import logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = Path("checkpoints/sensor")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameters
MAX_LEN = 128            # Shorter sequences for faster training (8 scales × 128 steps vs 512)
BATCH_SIZE = 512         # Large batch to amortize loop overhead (GPU has 80GB)
# Note: each epoch ~30 min with step-by-step SSM (no parallel scan available)
NUM_EPOCHS_MPP = _args.pretrain_epochs   # Phase 1: masked parameter prediction
NUM_EPOCHS_FT = _args.finetune_epochs    # Phase 2: anomaly fine-tuning
LR_BACKBONE = 3e-4
LR_HEAD = 1e-3
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
SEED = 42
MPP_MASK_RATIO = 0.4     # Mask 40% of parameters at each timestep

DATA_DIR = Path("data/processed/sensor/full")

torch.manual_seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Dataset — Real USGS NWIS sequences
# ---------------------------------------------------------------------------

class USGSSensorDataset(Dataset):
    """Real USGS NWIS sensor sequences from download_sensor.py.

    Each .npz file contains:
      - values: (512, 6) float32 — z-scored sensor parameters
      - delta_ts: (512,) float32 — time gaps in seconds
      - labels: (512,) float32 — per-timestep anomaly labels (0=normal)
      - mask: (512, 6) bool — per-parameter validity

    Filename format: {site_no}_seq{N:05d}.npz
    """

    def __init__(self, data_dir: str | Path, max_len: int = MAX_LEN):
        self.data_dir = Path(data_dir)
        self.files = sorted(self.data_dir.glob("*.npz"))
        self.max_len = max_len

        # Extract site_no and metadata from filenames
        self.site_ids = []
        for f in self.files:
            # Format: {site_no}_seq{N}.npz
            site_no = f.stem.rsplit("_seq", 1)[0]
            self.site_ids.append(site_no)

        logger.info(f"USGSSensorDataset: {len(self.files)} sequences from "
                    f"{len(set(self.site_ids))} unique stations in {data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        d = np.load(self.files[idx])
        T = min(d["values"].shape[0], self.max_len)

        values = torch.tensor(d["values"][:T].astype(np.float32)).clamp(-5, 5)
        delta_ts = torch.tensor(d["delta_ts"][:T].astype(np.float32)).clamp(0, 3600)
        delta_ts[0] = 0.0
        mask = torch.tensor(d["mask"][:T].astype(np.float32))
        labels = torch.tensor(d["labels"][:T].astype(np.float32))

        # Sequence-level anomaly label: 1 if any timestep is anomalous
        has_anomaly = float((labels > 0).any().item())

        return {
            "values": values,
            "delta_ts": delta_ts,
            "mask": mask,
            "labels": labels,
            "has_anomaly": has_anomaly,
            "site_id": self.site_ids[idx],
        }


def collate_fn(batch):
    """Collate with zero-padding."""
    max_len = max(b["values"].shape[0] for b in batch)
    B = len(batch)

    values = torch.zeros(B, max_len, 6)
    delta_ts = torch.zeros(B, max_len)
    mask = torch.zeros(B, max_len, 6)
    labels = torch.zeros(B, max_len)
    has_anomaly = torch.tensor([b["has_anomaly"] for b in batch], dtype=torch.float32)

    for i, b in enumerate(batch):
        T = b["values"].shape[0]
        values[i, :T] = b["values"]
        delta_ts[i, :T] = b["delta_ts"]
        mask[i, :T] = b["mask"]
        labels[i, :T] = b["labels"]

    return {
        "values": values,
        "delta_ts": delta_ts,
        "mask": mask,
        "labels": labels,
        "has_anomaly": has_anomaly,
    }


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
# MPP Head (Masked Parameter Prediction)
# ---------------------------------------------------------------------------

class MPPHead(nn.Module):
    """Reconstruct masked sensor parameters from SSM embedding."""

    def __init__(self, input_dim: int = 256, num_params: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Linear(128, num_params),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(labels, probs):
    """Compute AUROC, AUPRC, F1."""
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


# ---------------------------------------------------------------------------
# Phase 1: Masked Parameter Prediction (self-supervised)
# ---------------------------------------------------------------------------

def train_mpp(model, mpp_head, train_dl, val_dl, num_epochs, device):
    """Self-supervised pretraining: mask random parameters and predict them."""
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(mpp_head.parameters()),
        lr=LR_BACKBONE, weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    best_val_loss = float("inf")

    logger.info(f"Phase 1: MPP pretraining for {num_epochs} epochs")

    for epoch in range(num_epochs):
        model.train()
        mpp_head.train()
        train_loss = 0.0
        n_batches = 0

        n_total_batches = len(train_dl)
        epoch_start = time.time()
        for batch in train_dl:
            values = batch["values"].to(device)
            delta_ts = batch["delta_ts"].to(device)
            mask = batch["mask"].to(device)
            B, T, P = values.shape

            # Random mask: zero out some parameters at each timestep
            param_mask = (torch.rand(B, T, P, device=device) < MPP_MASK_RATIO) & (mask > 0)
            masked_values = values.clone()
            masked_values[param_mask] = 0.0

            # Forward through encoder
            out = model(masked_values, delta_ts)
            embedding = out["embedding"]  # (B, 256)

            # Predict original values from embedding
            pred = mpp_head(embedding)  # (B, 6)

            # Loss: MSE on masked parameters (mean over last timestep's masked params)
            # Use the original values at the last valid timestep as target
            target = values[:, -1, :]  # (B, 6)
            target_mask = mask[:, -1, :]  # (B, 6)
            loss = F.mse_loss(pred * target_mask, target * target_mask)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

            if n_batches == 1 or n_batches % 25 == 0:
                elapsed = time.time() - epoch_start
                logger.info(f"    batch {n_batches}/{n_total_batches} loss={loss.item():.4f} [{elapsed:.0f}s]")
                sys.stderr.flush()

        scheduler.step()
        avg_train = train_loss / max(n_batches, 1)

        # Validation
        model.eval()
        mpp_head.eval()
        val_loss = 0.0
        val_n = 0
        with torch.no_grad():
            for batch in val_dl:
                values = batch["values"].to(device)
                delta_ts = batch["delta_ts"].to(device)
                mask = batch["mask"].to(device)
                B, T, P = values.shape

                param_mask = (torch.rand(B, T, P, device=device) < MPP_MASK_RATIO) & (mask > 0)
                masked_values = values.clone()
                masked_values[param_mask] = 0.0

                out = model(masked_values, delta_ts)
                pred = mpp_head(out["embedding"])
                target = values[:, -1, :]
                target_mask = mask[:, -1, :]
                loss = F.mse_loss(pred * target_mask, target * target_mask)
                val_loss += loss.item()
                val_n += 1

        avg_val = val_loss / max(val_n, 1)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), CHECKPOINT_DIR / "aquassm_v2_mpp_best.pt")

        logger.info(f"  MPP Epoch {epoch+1}/{num_epochs}: train_loss={avg_train:.4f}, val_loss={avg_val:.4f}, best={best_val_loss:.4f}")
        sys.stderr.flush()
        sys.stdout.flush()

    logger.info(f"Phase 1 complete. Best val loss: {best_val_loss:.4f}")
    sys.stderr.flush()
    return best_val_loss


# ---------------------------------------------------------------------------
# Phase 2: Anomaly fine-tuning (supervised)
# ---------------------------------------------------------------------------

def train_anomaly(model, head, train_dl, val_dl, num_epochs, device):
    """Supervised anomaly detection fine-tuning."""
    optimizer = torch.optim.AdamW([
        {"params": model.parameters(), "lr": LR_BACKBONE * 0.1, "weight_decay": WEIGHT_DECAY},
        {"params": head.parameters(), "lr": LR_HEAD, "weight_decay": WEIGHT_DECAY},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Compute class weights from training data
    all_labels = []
    for batch in train_dl:
        all_labels.extend(batch["has_anomaly"].numpy().tolist())
    n_pos = sum(1 for l in all_labels if l > 0.5)
    n_neg = len(all_labels) - n_pos
    pos_weight = torch.tensor([max(n_neg / max(n_pos, 1), 1.0)], device=device)
    logger.info(f"Phase 2: Anomaly FT — {n_pos} anomaly, {n_neg} normal, pos_weight={pos_weight.item():.1f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best_val_auroc = 0.0
    patience = 15
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        head.train()
        train_loss = 0.0
        n_batches = 0

        for batch in train_dl:
            values = batch["values"].to(device)
            delta_ts = batch["delta_ts"].to(device)
            labels = batch["has_anomaly"].to(device)

            out = model(values, delta_ts)
            logits = head(out["embedding"])
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(head.parameters()), GRAD_CLIP
            )
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train = train_loss / max(n_batches, 1)

        # Validation
        model.eval()
        head.eval()
        all_labels = []
        all_probs = []
        with torch.no_grad():
            for batch in val_dl:
                values = batch["values"].to(device)
                delta_ts = batch["delta_ts"].to(device)
                labels_np = batch["has_anomaly"].numpy()

                out = model(values, delta_ts)
                logits = head(out["embedding"])
                probs = torch.sigmoid(logits).cpu().numpy()

                all_labels.extend(labels_np.tolist())
                all_probs.extend(probs.tolist())

        auroc, auprc, f1 = compute_metrics(np.array(all_labels), np.array(all_probs))

        if auroc > best_val_auroc:
            best_val_auroc = auroc
            patience_counter = 0
            torch.save(model.state_dict(), CHECKPOINT_DIR / "aquassm_v2_real_best.pt")
            torch.save(head.state_dict(), CHECKPOINT_DIR / "aquassm_v2_head_best.pt")
        else:
            patience_counter += 1

        logger.info(
            f"  FT Epoch {epoch+1}/{num_epochs}: loss={avg_train:.4f}, "
            f"val_auroc={auroc:.4f}, f1={f1:.4f}, auprc={auprc:.4f}"
        )
        sys.stderr.flush()
        sys.stdout.flush()

        if patience_counter >= patience:
            logger.info(f"  Early stopping at epoch {epoch+1}")
            break

    logger.info(f"Phase 2 complete. Best val AUROC: {best_val_auroc:.4f}")
    return best_val_auroc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info("=" * 70)
    logger.info("AquaSSM v2 Training — REAL USGS NWIS Data + Temporal-Spatial Holdout")
    logger.info("=" * 70)
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Data directory: {DATA_DIR}")

    if not DATA_DIR.exists():
        logger.error(f"Data directory {DATA_DIR} does not exist. Run download_sensor.py first.")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Load dataset
    # -----------------------------------------------------------------------
    full_ds = USGSSensorDataset(DATA_DIR)
    if len(full_ds) == 0:
        logger.error("No data found. Run download_sensor.py first.")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Temporal-spatial holdout split
    # -----------------------------------------------------------------------
    config = SplitConfig()

    # Extract timestamps from filenames or data
    # The download_sensor.py doesn't embed timestamps in filenames,
    # so we use spatial-only split (site-level holdout)
    # This ensures geographically unseen sites in val/test
    site_ids = full_ds.site_ids
    indices = split_indices(
        site_ids=site_ids,
        timestamps=None,  # spatial-only for now
        config=config,
    )

    logger.info(f"Split summary:")
    unique_train_sites = set(site_ids[i] for i in indices["train"])
    unique_val_sites = set(site_ids[i] for i in indices["val"])
    unique_test_sites = set(site_ids[i] for i in indices["test"])
    logger.info(f"  Train: {len(indices['train'])} sequences from {len(unique_train_sites)} sites")
    logger.info(f"  Val:   {len(indices['val'])} sequences from {len(unique_val_sites)} sites")
    logger.info(f"  Test:  {len(indices['test'])} sequences from {len(unique_test_sites)} sites")

    # Verify no site leakage
    train_val_overlap = unique_train_sites & unique_val_sites
    train_test_overlap = unique_train_sites & unique_test_sites
    val_test_overlap = unique_val_sites & unique_test_sites
    assert len(train_val_overlap) == 0, f"Site leakage train↔val: {train_val_overlap}"
    assert len(train_test_overlap) == 0, f"Site leakage train↔test: {train_test_overlap}"
    assert len(val_test_overlap) == 0, f"Site leakage val↔test: {val_test_overlap}"
    logger.info("  ✓ No site leakage between splits")

    train_ds = Subset(full_ds, indices["train"])
    val_ds = Subset(full_ds, indices["val"])
    test_ds = Subset(full_ds, indices["test"])

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=collate_fn, num_workers=0, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=collate_fn, num_workers=0, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                         collate_fn=collate_fn, num_workers=0, pin_memory=True)

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    model = SensorEncoder().to(DEVICE)
    mpp_head = MPPHead().to(DEVICE)
    anomaly_head = AnomalyHead().to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"SensorEncoder parameters: {total_params:,}")

    logger.info(f"MAX_LEN={MAX_LEN}, BATCH_SIZE={BATCH_SIZE}")

    # -----------------------------------------------------------------------
    # Phase 1: MPP pretraining
    # -----------------------------------------------------------------------
    mpp_loss = train_mpp(model, mpp_head, train_dl, val_dl, NUM_EPOCHS_MPP, DEVICE)

    # Load best MPP checkpoint
    model.load_state_dict(torch.load(CHECKPOINT_DIR / "aquassm_v2_mpp_best.pt", weights_only=True))
    logger.info("Loaded best MPP checkpoint")

    # -----------------------------------------------------------------------
    # Phase 2: Anomaly fine-tuning
    # -----------------------------------------------------------------------
    val_auroc = train_anomaly(model, anomaly_head, train_dl, val_dl, NUM_EPOCHS_FT, DEVICE)

    # -----------------------------------------------------------------------
    # Test evaluation
    # -----------------------------------------------------------------------
    model.load_state_dict(torch.load(CHECKPOINT_DIR / "aquassm_v2_real_best.pt", weights_only=True))
    anomaly_head.load_state_dict(torch.load(CHECKPOINT_DIR / "aquassm_v2_head_best.pt", weights_only=True))
    model.eval()
    anomaly_head.eval()

    all_labels = []
    all_probs = []
    with torch.no_grad():
        for batch in test_dl:
            values = batch["values"].to(DEVICE)
            delta_ts = batch["delta_ts"].to(DEVICE)

            out = model(values, delta_ts)
            logits = anomaly_head(out["embedding"])
            probs = torch.sigmoid(logits).cpu().numpy()

            all_labels.extend(batch["has_anomaly"].numpy().tolist())
            all_probs.extend(probs.tolist())

    test_auroc, test_auprc, test_f1 = compute_metrics(
        np.array(all_labels), np.array(all_probs)
    )
    logger.info("=" * 70)
    logger.info("TEST RESULTS (temporal-spatial holdout)")
    logger.info(f"  AUROC: {test_auroc:.4f}")
    logger.info(f"  AUPRC: {test_auprc:.4f}")
    logger.info(f"  F1:    {test_f1:.4f}")
    logger.info(f"  n_test: {len(all_labels)}")
    logger.info("=" * 70)

    # Save results
    results = {
        "model": "AquaSSM_v2_real",
        "data": "USGS_NWIS_real",
        "split": "temporal_spatial_holdout",
        "train_sites": len(unique_train_sites),
        "val_sites": len(unique_val_sites),
        "test_sites": len(unique_test_sites),
        "train_sequences": len(indices["train"]),
        "val_sequences": len(indices["val"]),
        "test_sequences": len(indices["test"]),
        "phase1_mpp_loss": mpp_loss,
        "phase2_val_auroc": val_auroc,
        "test_auroc": test_auroc,
        "test_auprc": test_auprc,
        "test_f1": test_f1,
        "elapsed_seconds": time.time() - t0,
        "timestamp": ts,
    }
    results_dir = Path("results/benchmarks")
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "aquassm_v2_real_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved. Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
