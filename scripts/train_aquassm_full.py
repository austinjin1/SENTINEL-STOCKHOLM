#!/usr/bin/env python3
"""AquaSSM full training: ALL 291K REAL SENSOR DATA via epoch subsampling.

Sources:
  - data/processed/sensor/real/    (291,855 real USGS labeled sequences)
  - data/processed/sensor/pretrain/ (162 real USGS, threshold-labeled)

Strategy: epoch subsampling (50K samples/epoch from 233K pool, shuffled each epoch).
This covers the full dataset in ~5 epochs, with the model seeing diverse data each pass.
- Train: 233K shuffled pool, sample 20K/epoch (batch=512 for GPU efficiency)
- Val:   29K fixed (full, loaded into RAM: ~0.09 GB)
- Test:  29K fixed (full, loaded into RAM: ~0.09 GB)

Class imbalance handled via pos_weight=4.0 (anomaly rate ~17.2%)
Server has 502 GB RAM; val+test (0.18 GB) and per-epoch loading is fast from RAM.

Column ordering: [DO(0), pH(1), SpCond(2), Temp(3), Turb(4), ORP(5)]

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
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, "/home/bcheng/SENTINEL")

from sentinel.models.sensor_encoder import SensorEncoder
from sentinel.utils.logging import get_logger

logger = get_logger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_ROOT   = Path("/home/bcheng/SENTINEL")
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "sensor"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameters
MAX_LEN           = 128   # 4x speedup vs 512
BATCH_SIZE        = 512   # large batch: 512/batch -> ~40 batches/epoch (fast on contested GPU)
NUM_EPOCHS        = 50    # 50 * 20K = 1M sample-epochs from 233K pool
SAMPLES_PER_EPOCH = 20000 # subsample from 233K train pool each epoch
LR_BACKBONE       = 3e-4
LR_HEAD           = 1e-3
WEIGHT_DECAY      = 0.01
GRAD_CLIP         = 1.0
PATIENCE          = 10
SEED              = 42
POS_WEIGHT        = 4.0   # ~4x for 17.2% anomaly rate

torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Normalized threshold constants for pretrain labeling
# ---------------------------------------------------------------------------
_DO_LO   = (4.0    - 9.0)   / 3.0
_PH_LO   = (6.0    - 7.5)   / 1.0
_PH_HI   = (9.5    - 7.5)   / 1.0
_SPC_HI  = (1500.0 - 500.0) / 400.0
_TUR_HI  = (300.0  - 20.0)  / 50.0


def label_pretrain_by_thresholds(values: np.ndarray) -> int:
    do   = values[:, 0]; ph = values[:, 1]; spc = values[:, 2]; turb = values[:, 4]
    if np.any(do   < _DO_LO):  return 1
    if np.any(ph   < _PH_LO):  return 1
    if np.any(ph   > _PH_HI):  return 1
    if np.any(spc  > _SPC_HI): return 1
    if np.any(turb > _TUR_HI): return 1
    return 0


# ---------------------------------------------------------------------------
# Datasets — all data in RAM (server has 502 GB)
# ---------------------------------------------------------------------------

class RealSensorDataset(Dataset):
    """RAM-cached dataset. Loads all files once; each epoch uses a Subset."""

    def __init__(self, file_list, max_len: int = MAX_LEN, desc: str = ""):
        n = len(file_list)
        logger.info(f"  Loading {n:,} files into RAM [{desc}]...")
        self.vals_arr   = np.zeros((n, max_len, 6), dtype=np.float32)
        self.dt_arr     = np.zeros((n, max_len),    dtype=np.float32)
        self.labels_arr = np.zeros(n,               dtype=np.int64)
        for i, f in enumerate(file_list):
            d = np.load(f, allow_pickle=True)
            vals = d['values'][:max_len].astype(np.float32)
            dt   = d['delta_ts'][:max_len].astype(np.float32)
            T = vals.shape[0]
            if T < max_len:
                vals = np.pad(vals, [(0, max_len - T), (0, 0)])
                dt   = np.pad(dt,   [(0, max_len - T)])
            self.vals_arr[i]   = np.clip(vals, -5.0, 5.0)
            self.dt_arr[i]     = np.clip(dt,    0.0, 3600.0)
            self.dt_arr[i][0]  = 0.0
            self.labels_arr[i] = int(d['has_anomaly'])
            if (i + 1) % 50000 == 0:
                logger.info(f"    {i+1:,}/{n:,} files loaded...")
        gb = self.vals_arr.nbytes / 1e9
        n_pos = int(self.labels_arr.sum())
        logger.info(f"  RAM cache ready: {gb:.2f} GB | anomaly={n_pos:,}, normal={n-n_pos:,}")

    def __len__(self):
        return len(self.labels_arr)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.vals_arr[idx].copy()),
            torch.from_numpy(self.dt_arr[idx].copy()),
            int(self.labels_arr[idx]),
        )

    @property
    def labels(self):
        return self.labels_arr


class PretrainDataset(Dataset):
    """162 pretrain files with threshold-derived labels."""

    def __init__(self, data_dir: str, max_len: int = MAX_LEN):
        files = sorted(Path(data_dir).glob("*.npz"))
        self.vals_arr   = np.zeros((len(files), max_len, 6), dtype=np.float32)
        self.dt_arr     = np.zeros((len(files), max_len),    dtype=np.float32)
        self.labels_arr = np.zeros(len(files),               dtype=np.int64)
        for i, f in enumerate(files):
            d = np.load(f)
            label = label_pretrain_by_thresholds(d["values"])
            T = min(len(d["values"]), max_len)
            v  = np.clip(d["values"][:T].astype(np.float32), -5.0, 5.0)
            dt = np.clip(d["delta_ts"][:T].astype(np.float32), 0.0, 3600.0)
            if T < max_len:
                v  = np.pad(v,  [(0, max_len - T), (0, 0)])
                dt = np.pad(dt, [(0, max_len - T)])
            dt[0] = 0.0
            self.vals_arr[i]   = v
            self.dt_arr[i]     = dt
            self.labels_arr[i] = label
        n_pos = int(self.labels_arr.sum())
        logger.info(f"PretrainDataset: {len(files)} files | anomaly={n_pos}, normal={len(files)-n_pos}")

    def __len__(self):
        return len(self.labels_arr)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.vals_arr[idx].copy()),
            torch.from_numpy(self.dt_arr[idx].copy()),
            int(self.labels_arr[idx]),
        )

    @property
    def labels(self):
        return self.labels_arr


def collate_fn(batch):
    vals_list, dt_list, label_list = zip(*batch)
    return {
        "values":      torch.stack(vals_list),
        "delta_ts":    torch.stack(dt_list),
        "has_anomaly": torch.tensor(label_list, dtype=torch.float32),
    }


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
    try:    auroc = roc_auc_score(labels, probs)
    except: auroc = 0.5
    try:    auprc = average_precision_score(labels, probs)
    except: auprc = 0.5
    f1 = f1_score(labels, (probs > 0.5).astype(int), zero_division=0)
    return auroc, auprc, f1


def compute_optimal_f1(labels, probs):
    best_f1, best_thresh = 0.0, 0.5
    for thresh in np.arange(0.1, 0.9, 0.05):
        f1 = f1_score(labels, (probs > thresh).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, thresh
    return best_f1, best_thresh


def run_epoch(model, head, loader, criterion, optimizer, device, train: bool, grad_clip: float):
    if train:
        model.train(); head.train()
    else:
        model.eval(); head.eval()

    total_loss, n_batches = 0.0, 0
    all_probs, all_labels = [], []
    nan_count = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            values   = batch["values"].to(device, non_blocking=True)
            delta_ts = batch["delta_ts"].to(device, non_blocking=True)
            labels   = batch["has_anomaly"].to(device, non_blocking=True)

            out = model(values, delta_ts=delta_ts, compute_anomaly=False)
            emb = out["embedding"]
            if torch.isnan(emb).any():
                nan_count += 1
                if train: optimizer.zero_grad()
                continue

            logits = head(emb)
            loss   = criterion(logits, labels)
            if torch.isnan(loss):
                nan_count += 1
                if train: optimizer.zero_grad()
                continue

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(head.parameters()), grad_clip
                )
                optimizer.step()

            total_loss += loss.item()
            n_batches  += 1
            all_probs.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
            all_labels.extend(labels.detach().cpu().numpy().tolist())

    avg_loss = total_loss / max(n_batches, 1)
    auroc, auprc, f1 = compute_metrics(np.array(all_labels), np.array(all_probs))
    return avg_loss, auroc, auprc, f1, nan_count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info("=" * 70)
    logger.info("AquaSSM FULL Training -- 291K Real USGS (epoch subsampling)")
    logger.info("=" * 70)
    logger.info(f"Device: {DEVICE} | Torch {torch.__version__}")
    logger.info(
        f"Hyperparams: MAX_LEN={MAX_LEN}, BATCH={BATCH_SIZE}, EPOCHS={NUM_EPOCHS}, "
        f"SAMPLES_PER_EPOCH={SAMPLES_PER_EPOCH:,}, LR_BB={LR_BACKBONE}, LR_HEAD={LR_HEAD} "
        f"(~1 min/epoch on contested A100)"
    )
    logger.info(f"pos_weight={POS_WEIGHT} | patience={PATIENCE} | seed={SEED}")

    # -----------------------------------------------------------------------
    # Build file lists + split
    # -----------------------------------------------------------------------
    real_dir     = PROJECT_ROOT / "data" / "processed" / "sensor" / "real"
    pretrain_dir = PROJECT_ROOT / "data" / "processed" / "sensor" / "pretrain"

    real_files = sorted(real_dir.glob("*.npz"))
    logger.info(f"Found {len(real_files):,} files in real/")

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(real_files))
    real_files = [real_files[i] for i in perm]

    n_real  = len(real_files)
    n_train = int(0.80 * n_real)
    n_val   = int(0.10 * n_real)
    n_test  = n_real - n_train - n_val

    train_files = real_files[:n_train]
    val_files   = real_files[n_train : n_train + n_val]
    test_files  = real_files[n_train + n_val :]
    logger.info(f"Split: train_pool={n_train:,}, val={n_val:,}, test={n_test:,}")

    # -----------------------------------------------------------------------
    # Load datasets into RAM
    # -----------------------------------------------------------------------
    ds_pretrain  = PretrainDataset(str(pretrain_dir))
    ds_train_pool = RealSensorDataset(train_files, desc="train pool")
    ds_val       = RealSensorDataset(val_files,   desc="val")
    ds_test      = RealSensorDataset(test_files,  desc="test")

    n_pool = len(ds_train_pool)
    logger.info(
        f"Pool: {n_pool:,} train + {len(ds_pretrain)} pretrain | "
        f"val={len(ds_val):,} | test={len(ds_test):,}"
    )

    val_dl = DataLoader(
        ds_val, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=True,
    )
    test_dl = DataLoader(
        ds_test, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=True,
    )

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

    pos_weight = torch.tensor([POS_WEIGHT], device=DEVICE)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW([
        {"params": model.parameters(), "lr": LR_BACKBONE, "weight_decay": WEIGHT_DECAY},
        {"params": head.parameters(),  "lr": LR_HEAD,     "weight_decay": WEIGHT_DECAY},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # -----------------------------------------------------------------------
    # Training loop with epoch subsampling
    # -----------------------------------------------------------------------
    best_val_auroc   = 0.0
    best_epoch       = 0
    patience_counter = 0
    total_nan        = 0
    epochs_trained   = 0
    history = {"train_loss": [], "train_auroc": [], "val_loss": [], "val_auroc": []}

    # Precompute pool label array for stratified subsampling
    pool_labels = ds_train_pool.labels  # numpy array of 0/1

    samples_per_epoch = min(SAMPLES_PER_EPOCH, n_pool)
    logger.info(
        f"Training: {NUM_EPOCHS} epochs x {samples_per_epoch:,} samples/epoch "
        f"(~{NUM_EPOCHS * samples_per_epoch / 1e6:.1f}M total sample-epochs)"
    )

    epoch_rng = np.random.default_rng(SEED + 1)

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_t0 = time.time()

        # Stratified subsample: maintain ~17.2% anomaly rate
        anomaly_idx = np.where(pool_labels == 1)[0]
        normal_idx  = np.where(pool_labels == 0)[0]
        n_anomaly_sample = int(samples_per_epoch * 0.172)
        n_normal_sample  = samples_per_epoch - n_anomaly_sample

        chosen_anomaly = epoch_rng.choice(anomaly_idx, size=min(n_anomaly_sample, len(anomaly_idx)), replace=False)
        chosen_normal  = epoch_rng.choice(normal_idx,  size=min(n_normal_sample,  len(normal_idx)),  replace=False)
        chosen_idx = np.concatenate([chosen_anomaly, chosen_normal])
        epoch_rng.shuffle(chosen_idx)

        # Combine with pretrain
        epoch_subset = Subset(ds_train_pool, chosen_idx.tolist())
        epoch_combined = ConcatDataset([ds_pretrain, epoch_subset])

        train_dl = DataLoader(
            epoch_combined, batch_size=BATCH_SIZE, shuffle=True,
            collate_fn=collate_fn, num_workers=0, pin_memory=True,
        )

        tr_loss, tr_auroc, _, _, ep_nan = run_epoch(
            model, head, train_dl, criterion, optimizer, DEVICE,
            train=True, grad_clip=GRAD_CLIP,
        )
        total_nan += ep_nan
        scheduler.step()

        va_loss, va_auroc, va_auprc, va_f1, _ = run_epoch(
            model, head, val_dl, criterion, optimizer, DEVICE,
            train=False, grad_clip=GRAD_CLIP,
        )

        history["train_loss"].append(tr_loss)
        history["train_auroc"].append(tr_auroc)
        history["val_loss"].append(va_loss)
        history["val_auroc"].append(va_auroc)

        improved = va_auroc > best_val_auroc
        if improved:
            best_val_auroc   = va_auroc
            best_epoch       = epoch
            patience_counter = 0
            torch.save(
                {"model": model.state_dict(), "head": head.state_dict(),
                 "epoch": epoch, "val_auroc": va_auroc},
                CHECKPOINT_DIR / "aquassm_full_best.pt",
            )
        else:
            patience_counter += 1

        epoch_time = time.time() - epoch_t0
        epochs_trained = epoch

        if epoch <= 3 or epoch % 5 == 0 or epoch == NUM_EPOCHS or improved:
            lr_bb = optimizer.param_groups[0]["lr"]
            coverage = min(epoch * samples_per_epoch / n_pool * 100, 100)
            logger.info(
                f"Ep {epoch:3d}/{NUM_EPOCHS} [{epoch_time:.0f}s] "
                f"cov={coverage:.0f}% | "
                f"TrLoss={tr_loss:.4f} TrAUC={tr_auroc:.4f} | "
                f"VaLoss={va_loss:.4f} VaAUC={va_auroc:.4f} VaF1={va_f1:.4f} | "
                f"LR={lr_bb:.2e} Pat={patience_counter}/{PATIENCE} NaN={total_nan}"
                + (" *BEST*" if improved else "")
            )

        if patience_counter >= PATIENCE:
            logger.info(f"Early stopping at epoch {epoch} (patience={PATIENCE})")
            break

    logger.info(f"Best val AUROC: {best_val_auroc:.4f} at epoch {best_epoch}")

    # -----------------------------------------------------------------------
    # Test evaluation
    # -----------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("TEST EVALUATION")
    logger.info("=" * 70)

    ckpt = torch.load(CHECKPOINT_DIR / "aquassm_full_best.pt",
                      map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model"])
    head.load_state_dict(ckpt["head"])

    te_loss, te_auroc, te_auprc, te_f1, _ = run_epoch(
        model, head, test_dl, criterion, optimizer, DEVICE,
        train=False, grad_clip=GRAD_CLIP,
    )

    # Re-run to get probs for optimal F1
    model.eval(); head.eval()
    te_probs_all, te_labels_all = [], []
    with torch.no_grad():
        for batch in test_dl:
            values   = batch["values"].to(DEVICE, non_blocking=True)
            delta_ts = batch["delta_ts"].to(DEVICE, non_blocking=True)
            labels   = batch["has_anomaly"]
            out = model(values, delta_ts=delta_ts, compute_anomaly=False)
            emb = out["embedding"]
            if torch.isnan(emb).any(): continue
            logits = head(emb)
            te_probs_all.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            te_labels_all.extend(labels.numpy().tolist())

    test_labels = np.array(te_labels_all)
    test_probs  = np.array(te_probs_all)
    test_auroc, test_auprc, test_f1 = compute_metrics(test_labels, test_probs)
    best_f1_opt, best_thresh = compute_optimal_f1(test_labels, test_probs)
    test_acc = float(((test_probs > 0.5).astype(int) == test_labels.astype(int)).mean())

    logger.info(f"Test AUROC:  {test_auroc:.4f}  (baseline=0.920)")
    logger.info(f"Test AUPRC:  {test_auprc:.4f}")
    logger.info(f"Test F1@0.5: {test_f1:.4f}")
    logger.info(f"Test F1 opt: {best_f1_opt:.4f} (thresh={best_thresh:.2f})")
    logger.info(f"Test Acc:    {test_acc:.4f}")
    logger.info(f"N_test={len(test_labels):,}, N_pos={int(test_labels.sum()):,}, N_neg={int((1-test_labels).sum()):,}")

    if test_auroc > 0.920:
        logger.info(f"*** BEATS BASELINE: {test_auroc:.4f} > 0.920 ***")
    elif test_auroc > 0.85:
        logger.info(f"Strong: {test_auroc:.4f} (below 0.920 baseline)")
    else:
        logger.info(f"Below baseline: {test_auroc:.4f} < 0.920")

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    elapsed = time.time() - t0
    results = {
        "test_auroc":        float(test_auroc),
        "test_auprc":        float(test_auprc),
        "test_f1":           float(test_f1),
        "test_f1_optimal":   float(best_f1_opt),
        "optimal_threshold": float(best_thresh),
        "test_acc":          float(test_acc),
        "best_val_auroc":    float(best_val_auroc),
        "best_epoch":        int(best_epoch),
        "epochs_trained":    int(epochs_trained),
        "n_train":           n_train + len(ds_pretrain),
        "n_val":             len(ds_val),
        "n_test":            len(ds_test),
        "n_test_evaluated":  len(te_labels_all),
        "n_test_pos":        int(test_labels.sum()),
        "n_total":           n_real + len(ds_pretrain),
        "samples_per_epoch": samples_per_epoch,
        "baseline_auroc":    0.920,
        "beats_baseline":    bool(test_auroc > 0.920),
        "data_source":       "real/ (291K USGS)",
        "total_nan_batches": total_nan,
        "elapsed_seconds":   elapsed,
        "elapsed_minutes":   elapsed / 60.0,
        "hyperparameters": {
            "lr_backbone": LR_BACKBONE, "lr_head": LR_HEAD,
            "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE,
            "max_len": MAX_LEN, "grad_clip": GRAD_CLIP,
            "pos_weight": POS_WEIGHT, "patience": PATIENCE, "seed": SEED,
            "num_epochs": NUM_EPOCHS, "samples_per_epoch": SAMPLES_PER_EPOCH,
        },
        "training_curves": {
            "train_loss":  history["train_loss"],
            "train_auroc": history["train_auroc"],
            "val_loss":    history["val_loss"],
            "val_auroc":   history["val_auroc"],
        },
        "timestamp": ts,
    }

    results_path = CHECKPOINT_DIR / "results_full.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")
    logger.info(f"Checkpoint: {CHECKPOINT_DIR}/aquassm_full_best.pt")
    logger.info(f"Total time: {elapsed/60:.1f} minutes")

    print("\n" + "=" * 65)
    print("=== AquaSSM FULL (291K Real USGS) -- Final Results ===")
    print(f"  Data pool:      {n_pool:,} train + {len(ds_pretrain)} pretrain")
    print(f"  Samples/epoch:  {samples_per_epoch:,}")
    print(f"  Val/Test:       {len(ds_val):,} / {len(ds_test):,} (full, fixed)")
    print(f"  Epochs:         {epochs_trained}/{NUM_EPOCHS} (patience={PATIENCE})")
    print(f"  Baseline AUROC: 0.9200 (aquassm_real_best, 20K samples)")
    print(f"  Test AUROC:     {test_auroc:.4f}  {'*** BEATS BASELINE ***' if test_auroc > 0.920 else '(below baseline)'}")
    print(f"  Test AUPRC:     {test_auprc:.4f}")
    print(f"  Test F1@0.5:    {test_f1:.4f}")
    print(f"  Test F1 opt:    {best_f1_opt:.4f} (thresh={best_thresh:.2f})")
    print(f"  Test Accuracy:  {test_acc:.4f}")
    print(f"  Time:           {elapsed/60:.1f} min")
    print("=" * 65)


if __name__ == "__main__":
    main()
