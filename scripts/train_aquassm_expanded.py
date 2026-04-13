#!/usr/bin/env python3
"""AquaSSM expanded training: REAL DATA ONLY.

Sources:
  - data/processed/sensor/pretrain/  (162 real USGS, threshold-labeled)
  - data/processed/sensor/real_20k/  subset: 300 anomaly + 300 normal = 600 files

Total: 762 samples (~3x expansion from original 262).
NO synthetic data used.

Column ordering: [DO(0), pH(1), SpCond(2), Temp(3), Turb(4), ORP(5)]
Normalization: NORM_MEAN={DO:9,pH:7.5,SpCond:500,Temp:15,Turb:20,ORP:200}
               NORM_STD ={DO:3,pH:1,  SpCond:400,Temp:8, Turb:50, ORP:150}
Pretrain anomaly thresholds (normalized):
  DO < (4-9)/3 = -1.667
  pH < (6-7.5)/1 = -1.5  OR  pH > (9.5-7.5)/1 = 2.0
  SpCond > (1500-500)/400 = 2.5
  Turb > (300-20)/50 = 5.6

All data cached in RAM at init to avoid repeated disk I/O.
150 epochs, AdamW, cosine annealing, early stopping patience=15.

Bryan Cheng, SENTINEL project, 2026
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
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, "/home/bcheng/SENTINEL")

from sentinel.models.sensor_encoder import SensorEncoder
from sentinel.utils.logging import get_logger

logger = get_logger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = Path("checkpoints/sensor")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameters
# MAX_LEN=128: all anomaly events appear in first 64 timesteps; 128 captures full context
# while reducing ContinuousTimeSSMCell compute from 512->128 steps (~4x faster).
MAX_LEN    = 128
BATCH_SIZE = 16
NUM_EPOCHS = 150
LR_BACKBONE = 3e-4
LR_HEAD     = 1e-3
WEIGHT_DECAY = 0.01
GRAD_CLIP   = 1.0
PATIENCE    = 15
SEED        = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# Normalized threshold boundaries for pretrain anomaly labeling
DO_LOW_NORM      = (4.0   - 9.0)   / 3.0    # -1.6667
PH_LOW_NORM      = (6.0   - 7.5)   / 1.0    # -1.5
PH_HIGH_NORM     = (9.5   - 7.5)   / 1.0    #  2.0
SPCOND_HIGH_NORM = (1500.0- 500.0) / 400.0  #  2.5
TURB_HIGH_NORM   = (300.0 - 20.0)  / 50.0   #  5.6


def label_pretrain_by_thresholds(values: np.ndarray) -> int:
    """Return 1 if any timestep violates water quality thresholds."""
    do   = values[:, 0]
    ph   = values[:, 1]
    spc  = values[:, 2]
    turb = values[:, 4]
    if np.any(do   < DO_LOW_NORM):      return 1
    if np.any(ph   < PH_LOW_NORM):      return 1
    if np.any(ph   > PH_HIGH_NORM):     return 1
    if np.any(spc  > SPCOND_HIGH_NORM): return 1
    if np.any(turb > TURB_HIGH_NORM):   return 1
    return 0


# ---------------------------------------------------------------------------
# Datasets — all data cached in RAM
# ---------------------------------------------------------------------------

class PretrainRealDataset(Dataset):
    """162 real USGS pretrain files, threshold-labeled, cached in RAM."""

    def __init__(self, data_dir: str, max_len: int = MAX_LEN):
        files = sorted(Path(data_dir).glob("*.npz"))
        self.samples = []
        self.labels  = []
        for f in files:
            d = np.load(f)
            label = label_pretrain_by_thresholds(d["values"])
            T = min(len(d["values"]), max_len)
            v  = torch.tensor(d["values"][:T].astype(np.float32)).clamp(-5, 5)
            dt = torch.tensor(d["delta_ts"][:T].astype(np.float32)).clamp(0, 3600)
            dt[0] = 0.0
            self.samples.append((v, dt))
            self.labels.append(label)
        n_pos = sum(self.labels)
        logger.info(
            f"PretrainRealDataset: {len(files)} files from {data_dir} "
            f"| anomaly={n_pos}, normal={len(files)-n_pos} (cached in RAM)"
        )

    def __len__(self):             return len(self.samples)
    def __getitem__(self, idx):
        v, dt = self.samples[idx]
        return {"values": v, "delta_ts": dt, "has_anomaly": self.labels[idx]}


class Real20kDataset(Dataset):
    """Balanced real_20k subset, has_anomaly labels from file, cached in RAM."""

    def __init__(self, file_list: list, max_len: int = MAX_LEN):
        self.samples = []
        self.labels  = []
        for f in file_list:
            d = np.load(f)
            label = int(d["has_anomaly"])
            T = min(len(d["values"]), max_len)
            v  = torch.tensor(d["values"][:T].astype(np.float32)).clamp(-5, 5)
            dt = torch.tensor(d["delta_ts"][:T].astype(np.float32)).clamp(0, 3600)
            dt[0] = 0.0
            self.samples.append((v, dt))
            self.labels.append(label)
        n_pos = sum(self.labels)
        logger.info(
            f"Real20kDataset: {len(file_list)} files "
            f"| anomaly={n_pos}, normal={len(file_list)-n_pos} (cached in RAM)"
        )

    def __len__(self):             return len(self.samples)
    def __getitem__(self, idx):
        v, dt = self.samples[idx]
        return {"values": v, "delta_ts": dt, "has_anomaly": self.labels[idx]}


def collate_fn(batch):
    max_len = max(b["values"].shape[0] for b in batch)
    B = len(batch)
    values    = torch.zeros(B, max_len, 6)
    delta_ts  = torch.zeros(B, max_len)
    has_anomaly = torch.tensor([b["has_anomaly"] for b in batch], dtype=torch.float32)
    for i, b in enumerate(batch):
        T = b["values"].shape[0]
        values[i, :T]   = b["values"]
        delta_ts[i, :T] = b["delta_ts"]
    return {"values": values, "delta_ts": delta_ts, "has_anomaly": has_anomaly}


# ---------------------------------------------------------------------------
# Classification Head
# ---------------------------------------------------------------------------

class AnomalyHead(nn.Module):
    def __init__(self, input_dim: int = 256, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(labels, probs):
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


def compute_optimal_f1(labels, probs):
    best_f1, best_thresh = 0.0, 0.5
    for thresh in np.arange(0.1, 0.9, 0.05):
        f1 = f1_score(labels, (probs > thresh).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, thresh
    return best_f1, best_thresh


# ---------------------------------------------------------------------------
# Data selection
# ---------------------------------------------------------------------------

def select_balanced_real20k(target_anomaly: int = 300, target_normal: int = 300):
    rng = np.random.default_rng(SEED)
    all_files = sorted(Path("data/processed/sensor/real_20k").glob("*.npz"))
    anomaly_files, normal_files = [], []
    for f in all_files:
        if int(np.load(f)["has_anomaly"]) == 1:
            anomaly_files.append(f)
        else:
            normal_files.append(f)
    anomaly_files = [anomaly_files[i] for i in rng.permutation(len(anomaly_files))]
    normal_files  = [normal_files[i]  for i in rng.permutation(len(normal_files))]
    selected = anomaly_files[:target_anomaly] + normal_files[:target_normal]
    logger.info(
        f"real_20k selection: {len(selected[:target_anomaly])} anomaly + "
        f"{len(selected[target_anomaly:])} normal = {len(selected)} total"
    )
    return selected


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info("=" * 70)
    logger.info("AquaSSM Expanded Training — REAL DATA ONLY")
    logger.info("=" * 70)
    logger.info(f"Device: {DEVICE} | Torch {torch.__version__}")
    logger.info("Sources: pretrain/ (162 real USGS) + real_20k/ subset (600 real USGS)")

    # -----------------------------------------------------------------------
    # Load and cache datasets
    # -----------------------------------------------------------------------
    ds_pretrain = PretrainRealDataset("data/processed/sensor/pretrain")
    selected    = select_balanced_real20k(300, 300)
    ds_real20k  = Real20kDataset(selected)

    full_ds = ConcatDataset([ds_pretrain, ds_real20k])
    n_total = len(full_ds)
    logger.info(f"Total: {n_total} samples (pretrain={len(ds_pretrain)}, real_20k={len(ds_real20k)})")

    n_train = int(0.70 * n_total)
    n_val   = int(0.15 * n_total)
    n_test  = n_total - n_train - n_val
    train_ds, val_ds, test_ds = random_split(
        full_ds, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(SEED),
    )
    logger.info(f"Split: train={n_train}, val={n_val}, test={n_test}")

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=collate_fn, num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          collate_fn=collate_fn, num_workers=0)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          collate_fn=collate_fn, num_workers=0)

    for name, dl in [("train", train_dl), ("val", val_dl), ("test", test_dl)]:
        labs = []
        for b in dl: labs.extend(b["has_anomaly"].numpy().tolist())
        pos = sum(1 for l in labs if l > 0.5)
        logger.info(f"  {name}: {len(labs)} samples, {pos} anomaly, {len(labs)-pos} normal")

    # -----------------------------------------------------------------------
    # Model + optimizer
    # -----------------------------------------------------------------------
    model = SensorEncoder().to(DEVICE)
    head  = AnomalyHead().to(DEVICE)
    total_params = (
        sum(p.numel() for p in model.parameters())
        + sum(p.numel() for p in head.parameters())
    )
    logger.info(f"Total parameters: {total_params:,}")

    train_labels = []
    for b in train_dl: train_labels.extend(b["has_anomaly"].numpy().tolist())
    n_pos = sum(1 for l in train_labels if l > 0.5)
    n_neg = len(train_labels) - n_pos
    pos_w = float(max(1.0, min(n_neg / max(n_pos, 1), 10.0)))
    logger.info(f"pos_weight: {pos_w:.2f}  (n_pos={n_pos}, n_neg={n_neg})")
    pos_weight = torch.tensor([pos_w], device=DEVICE)

    optimizer = torch.optim.AdamW([
        {"params": model.parameters(), "lr": LR_BACKBONE, "weight_decay": WEIGHT_DECAY},
        {"params": head.parameters(),  "lr": LR_HEAD,     "weight_decay": WEIGHT_DECAY},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    best_val_auroc   = 0.0
    best_epoch       = 0
    patience_counter = 0
    history = {"train_loss": [], "train_auroc": [], "val_loss": [], "val_auroc": []}
    nan_count        = 0
    epochs_trained   = 0

    logger.info(f"Training: {NUM_EPOCHS} epochs max, patience={PATIENCE}")
    logger.info(f"  LR backbone={LR_BACKBONE}, head={LR_HEAD}, batch={BATCH_SIZE}, clip={GRAD_CLIP}")

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train(); head.train()
        tr_loss, tr_batches = 0.0, 0
        tr_probs, tr_labels = [], []

        for batch in train_dl:
            values   = batch["values"].to(DEVICE)
            delta_ts = batch["delta_ts"].to(DEVICE)
            labels   = batch["has_anomaly"].to(DEVICE)

            out = model(values, delta_ts=delta_ts, compute_anomaly=False)
            emb = out["embedding"]
            if torch.isnan(emb).any():
                nan_count += 1; optimizer.zero_grad(); continue

            logits = head(emb)
            loss   = criterion(logits, labels)
            if torch.isnan(loss):
                nan_count += 1; optimizer.zero_grad(); continue

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), GRAD_CLIP)
            optimizer.step()

            tr_loss += loss.item(); tr_batches += 1
            tr_probs.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
            tr_labels.extend(labels.cpu().numpy().tolist())

        scheduler.step()
        epochs_trained = epoch

        train_loss  = tr_loss / max(tr_batches, 1)
        train_auroc, _, _ = compute_metrics(np.array(tr_labels), np.array(tr_probs))

        model.eval(); head.eval()
        va_loss, va_batches = 0.0, 0
        va_probs, va_labels = [], []

        with torch.no_grad():
            for batch in val_dl:
                values   = batch["values"].to(DEVICE)
                delta_ts = batch["delta_ts"].to(DEVICE)
                labels   = batch["has_anomaly"].to(DEVICE)
                out = model(values, delta_ts=delta_ts, compute_anomaly=False)
                emb = out["embedding"]
                if torch.isnan(emb).any(): continue
                logits = head(emb)
                loss   = criterion(logits, labels)
                if not torch.isnan(loss):
                    va_loss += loss.item(); va_batches += 1
                    va_probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
                    va_labels.extend(labels.cpu().numpy().tolist())

        val_loss = va_loss / max(va_batches, 1)
        val_auroc, val_auprc, val_f1 = compute_metrics(np.array(va_labels), np.array(va_probs))

        history["train_loss"].append(train_loss)
        history["train_auroc"].append(train_auroc)
        history["val_loss"].append(val_loss)
        history["val_auroc"].append(val_auroc)

        improved = val_auroc > best_val_auroc
        if improved:
            best_val_auroc   = val_auroc
            best_epoch       = epoch
            patience_counter = 0
            torch.save(
                {"model": model.state_dict(), "head": head.state_dict(),
                 "epoch": epoch, "val_auroc": val_auroc},
                CHECKPOINT_DIR / "aquassm_expanded_best.pt",
            )
        else:
            patience_counter += 1

        if epoch <= 3 or epoch % 10 == 0 or epoch == NUM_EPOCHS or improved:
            lr_bb = optimizer.param_groups[0]["lr"]
            logger.info(
                f"Ep {epoch:3d}/{NUM_EPOCHS} | "
                f"TrLoss={train_loss:.4f} TrAUC={train_auroc:.4f} | "
                f"VaLoss={val_loss:.4f} VaAUC={val_auroc:.4f} VaF1={val_f1:.4f} | "
                f"LR={lr_bb:.2e} | Pat={patience_counter}/{PATIENCE} | NaN={nan_count}"
                + (" *" if improved else "")
            )

        if patience_counter >= PATIENCE:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    logger.info(f"Best val AUROC: {best_val_auroc:.4f} at epoch {best_epoch}")

    # -----------------------------------------------------------------------
    # Test evaluation
    # -----------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("TEST EVALUATION")
    logger.info("=" * 70)

    ckpt = torch.load(CHECKPOINT_DIR / "aquassm_expanded_best.pt", map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model"])
    head.load_state_dict(ckpt["head"])
    model.eval(); head.eval()

    te_probs, te_labels = [], []
    with torch.no_grad():
        for batch in test_dl:
            values   = batch["values"].to(DEVICE)
            delta_ts = batch["delta_ts"].to(DEVICE)
            labels   = batch["has_anomaly"]
            out = model(values, delta_ts=delta_ts, compute_anomaly=False)
            emb = out["embedding"]
            if torch.isnan(emb).any():
                logger.warning("NaN in test embedding — skipping batch.")
                continue
            logits = head(emb)
            te_probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            te_labels.extend(labels.numpy().tolist())

    test_labels = np.array(te_labels)
    test_probs  = np.array(te_probs)
    test_auroc, test_auprc, test_f1 = compute_metrics(test_labels, test_probs)
    best_f1_opt, best_thresh = compute_optimal_f1(test_labels, test_probs)
    test_acc = float(((test_probs > 0.5).astype(int) == test_labels.astype(int)).mean())

    logger.info(f"Test AUROC:  {test_auroc:.4f}")
    logger.info(f"Test AUPRC:  {test_auprc:.4f}")
    logger.info(f"Test F1@0.5: {test_f1:.4f}")
    logger.info(f"Test F1 opt: {best_f1_opt:.4f} (thresh={best_thresh:.2f})")
    logger.info(f"Test Acc:    {test_acc:.4f}")
    logger.info(f"N_test={len(test_labels)}, N_pos={int(test_labels.sum())}, N_neg={int((1-test_labels).sum())}")
    logger.info(f"NaN batches skipped: {nan_count}")

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
        "test_auroc":        float(test_auroc),
        "test_auprc":        float(test_auprc),
        "test_f1":           float(test_f1),
        "test_f1_optimal":   float(best_f1_opt),
        "test_acc":          float(test_acc),
        "optimal_threshold": float(best_thresh),
        "best_val_auroc":    float(best_val_auroc),
        "best_epoch":        int(best_epoch),
        "epochs_trained":    int(epochs_trained),
        "n_train": n_train, "n_val": n_val, "n_test": n_test,
        "n_test_evaluated":  len(te_labels),
        "n_test_pos":        int(test_labels.sum()),
        "n_total":           n_total,
        "n_pretrain":        len(ds_pretrain),
        "n_real20k":         len(ds_real20k),
        "data_sources":      ["pretrain (real USGS)", "real_20k subset (real USGS)"],
        "synthetic_data":    False,
        "total_nan_batches": nan_count,
        "elapsed_seconds":   elapsed,
        "elapsed_minutes":   elapsed / 60.0,
        "hyperparameters": {
            "lr_backbone": LR_BACKBONE, "lr_head": LR_HEAD,
            "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE,
            "max_len": MAX_LEN, "grad_clip": GRAD_CLIP,
            "pos_weight": pos_w, "patience": PATIENCE, "seed": SEED,
        },
        "training_curves": {
            "train_loss":  history["train_loss"],
            "train_auroc": history["train_auroc"],
            "val_loss":    history["val_loss"],
            "val_auroc":   history["val_auroc"],
        },
        "timestamp": ts,
    }

    results_path = CHECKPOINT_DIR / "results_expanded.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")
    logger.info(f"Checkpoint: checkpoints/sensor/aquassm_expanded_best.pt")
    logger.info(f"Total time: {elapsed/60:.1f} minutes")

    print("\n" + "=" * 60)
    print("=== AquaSSM Expanded (Real Data Only) — Results ===")
    print(f"  Data:          {n_total} samples ({n_total/262:.1f}x from 262)")
    print(f"  Real sources:  pretrain + real_20k subset")
    print(f"  Epochs:        {epochs_trained}/{NUM_EPOCHS}")
    print(f"  Test AUROC:    {test_auroc:.4f}")
    print(f"  Test AUPRC:    {test_auprc:.4f}")
    print(f"  Test F1@0.5:   {test_f1:.4f}")
    print(f"  Test F1 opt:   {best_f1_opt:.4f} (thresh={best_thresh:.2f})")
    print(f"  Test Accuracy: {test_acc:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
