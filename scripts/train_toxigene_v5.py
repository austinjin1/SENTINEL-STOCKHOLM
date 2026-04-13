#!/usr/bin/env python3
"""ToxiGene v5 — Full-gene SimpleMLP with BatchNorm (matches v2 SimpleMLP).

Root cause of v3/v4 gap vs v2 SimpleMLP (0.8896):
  v3/v4 SimpleMLP uses 5,000 selected genes, no BatchNorm.
  v2 SimpleMLP uses ALL 61,479 genes + BatchNorm → F1=0.8896.

Strategy:
  Reproduce v2's SimpleMLP exactly but with multi-label BCE loss:
    - Input: 61,479 genes, log1p-transformed → z-score → clip[-10,10]
    - Architecture: Linear(61479,512)→BN→ReLU→Drop(0.3)→Linear(512,256)→BN→ReLU→Drop(0.3)→Linear(256,7)
    - Loss: BCE with pos_weight (multi-label)
    - AdamW: lr=3e-4, weight_decay=0.01
    - Augmentation: Gaussian noise (CLR-like), gene dropout, Mixup
    - Early stopping patience=30

MIT License — Bryan Cheng, 2026
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path("/home/bcheng/SENTINEL")
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR  = PROJECT_ROOT / "data" / "processed" / "molecular"
CKPT_DIR  = PROJECT_ROOT / "checkpoints" / "molecular"
LOG_DIR   = PROJECT_ROOT / "logs"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

CKPT_PATH    = CKPT_DIR / "toxigene_v5_best.pt"
RESULTS_PATH = CKPT_DIR / "results_v5.json"
LOG_PATH     = LOG_DIR  / "train_toxigene_v5.log"

# ── Hyperparameters ───────────────────────────────────────────────────────────
HIDDEN_DIM     = 512
DROPOUT        = 0.3
BATCH_SIZE     = 64
EPOCHS         = 400
LR             = 3e-4
WEIGHT_DECAY   = 0.01
GRAD_CLIP      = 1.0
EARLY_STOP_PAT = 30
SEED           = 42

# Augmentation
NOISE_PROB     = 0.60
NOISE_STD      = 0.02
GENE_DROP_RATE = 0.15
MIXUP_PROB     = 0.40
MIXUP_ALPHA    = 0.3

N_TRAIN = 1187
N_VAL   = 254
N_TEST  = 256

OUTCOME_NAMES = [
    "reproductive_impairment",
    "growth_inhibition",
    "immunosuppression",
    "neurotoxicity",
    "hepatotoxicity",
    "oxidative_damage",
    "endocrine_disruption",
]


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
def _setup_logging() -> logging.Logger:
    try:
        from sentinel.utils.logging import get_logger
        log = get_logger("toxigene_v5")
    except Exception:
        log = logging.getLogger("toxigene_v5")
        if not log.handlers:
            log.setLevel(logging.INFO)
            ch = logging.StreamHandler(sys.stdout)
            ch.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S"))
            log.addHandler(ch)
            log.propagate = False
    fh = logging.FileHandler(str(LOG_PATH), mode="w")
    fh.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S"))
    fh.setLevel(logging.INFO)
    log.addHandler(fh)
    return log


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class ToxicologyDataset(Dataset):
    def __init__(self, expression: np.ndarray, labels: np.ndarray):
        self.X = torch.tensor(expression, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation
# ─────────────────────────────────────────────────────────────────────────────
def augment_batch(x: torch.Tensor, y: torch.Tensor, training: bool = True):
    if not training:
        return x, y.float()

    B = x.size(0)
    y_out = y.float()

    if torch.rand(1).item() < NOISE_PROB:
        x = x + torch.randn_like(x) * NOISE_STD

    mask = (torch.rand_like(x) > GENE_DROP_RATE).float()
    x = x * mask

    if torch.rand(1).item() < MIXUP_PROB:
        lam = float(np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA))
        perm = torch.randperm(B, device=x.device)
        x     = lam * x     + (1.0 - lam) * x[perm]
        y_out = lam * y_out + (1.0 - lam) * y_out[perm]

    return x, y_out


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
class ToxiGeneV5(nn.Module):
    """SimpleMLP with BatchNorm — matches v2 SimpleMLP architecture."""

    def __init__(self, n_genes: int, n_classes: int = 7,
                 hidden_dim: int = HIDDEN_DIM, dropout: float = DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_genes, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────
def compute_pos_weight(labels: np.ndarray) -> torch.Tensor:
    pos = labels.sum(axis=0).astype(np.float32)
    neg = labels.shape[0] - pos
    pos = np.maximum(pos, 1.0)
    return torch.tensor(neg / pos, dtype=torch.float32)


def multilabel_bce_loss(logits, targets, pos_weight=None):
    return F.binary_cross_entropy_with_logits(
        logits, targets,
        pos_weight=pos_weight.to(logits.device) if pos_weight is not None else None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, device, pos_weight):
    model.eval()
    all_preds, all_labels = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with autocast(device_type="cuda"):
            logits = model(x)
        preds = (torch.sigmoid(logits) > 0.5).float()
        all_preds.append(preds.cpu().numpy())
        all_labels.append(y.cpu().numpy())

    all_preds  = np.concatenate(all_preds,  axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_labels_bin = (all_labels > 0.5).astype(int)

    f1_macro = f1_score(all_labels_bin, all_preds.astype(int), average="macro", zero_division=0)
    acc      = accuracy_score(all_labels_bin, all_preds.astype(int))
    per_class_f1 = f1_score(all_labels_bin, all_preds.astype(int), average=None, zero_division=0).tolist()
    return {"f1": f1_macro, "acc": acc, "per_class_f1": per_class_f1}


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
def train_model(model, train_loader, val_loader, device, log, pos_weight,
                ckpt_path, name="ToxiGeneV5", epochs=EPOCHS, patience=EARLY_STOP_PAT):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler("cuda")

    best_val_f1 = 0.0
    best_epoch  = 0
    no_improve  = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        n_batches = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            x, y_aug = augment_batch(x, y, training=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda"):
                logits = model(x)
                loss = multilabel_bce_loss(logits, y_aug, pos_weight)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()
            n_batches += 1

        scheduler.step()

        val_metrics = evaluate(model, val_loader, device, pos_weight)
        val_f1 = val_metrics["f1"]

        if epoch % 10 == 0 or epoch == 1:
            log.info(
                f"[{name}] Epoch {epoch:3d}/{epochs} | "
                f"train_loss={total_train_loss/max(n_batches,1):.4f} | "
                f"val_F1={val_f1:.4f} | "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch  = epoch
            no_improve  = 0
            torch.save(model.state_dict(), str(ckpt_path))
        else:
            no_improve += 1

        if no_improve >= patience:
            log.info(f"  Early stopping at epoch {epoch} (best val F1={best_val_f1:.4f})")
            break

    return best_val_f1, best_epoch


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    t0 = time.time()
    log = _setup_logging()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # ── Load data (all 61,479 genes, already log1p from preprocessing) ────────
    expr_path   = DATA_DIR / "expression_matrix_v2_expanded.npy"
    labels_path = DATA_DIR / "outcome_labels_v2_expanded.npy"

    log.info(f"Loading: {expr_path.name}")
    expression = np.load(str(expr_path))          # (1697, 61479)
    labels     = np.load(str(labels_path)).astype(np.float32)  # (1697, 7)

    log.info(f"Expression: {expression.shape}, Labels: {labels.shape}")

    N, N_GENES = expression.shape
    log.info(f"Split: {N_TRAIN} train / {N_VAL} val / {N_TEST} test")

    # ── Deterministic split (identical to v3/v4 for fair comparison) ──────────
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(N)
    train_idx = idx[:N_TRAIN]
    val_idx   = idx[N_TRAIN:N_TRAIN + N_VAL]
    test_idx  = idx[N_TRAIN + N_VAL:]

    X_train = expression[train_idx].astype(np.float32)
    X_val   = expression[val_idx].astype(np.float32)
    X_test  = expression[test_idx].astype(np.float32)
    y_train = labels[train_idx]
    y_val   = labels[val_idx]
    y_test  = labels[test_idx]

    # ── Normalize: z-score (train stats) + clip ───────────────────────────────
    mu  = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std < 1e-6] = 1.0
    X_train = np.clip((X_train - mu) / std, -10.0, 10.0)
    X_val   = np.clip((X_val   - mu) / std, -10.0, 10.0)
    X_test  = np.clip((X_test  - mu) / std, -10.0, 10.0)
    log.info("Normalized: z-score (train stats) + clip[-10,10]")

    # ── pos_weight ────────────────────────────────────────────────────────────
    pos_weight = compute_pos_weight(y_train).to(device)
    log.info(f"BCE pos_weight: {pos_weight.tolist()}")

    # ── Dataloaders ───────────────────────────────────────────────────────────
    train_ds = ToxicologyDataset(X_train, y_train)
    val_ds   = ToxicologyDataset(X_val,   y_val)
    test_ds  = ToxicologyDataset(X_test,  y_test)

    g = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True, generator=g)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)

    # ── Build model ───────────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("Building ToxiGene v5 (SimpleMLP + BatchNorm, all genes) …")
    model = ToxiGeneV5(n_genes=N_GENES, n_classes=7).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"ToxiGene v5: {n_params:,} parameters")
    log.info("Training …")

    best_val_f1, best_epoch = train_model(
        model, train_loader, val_loader, device, log, pos_weight,
        ckpt_path=CKPT_PATH, name="ToxiGeneV5",
        epochs=EPOCHS, patience=EARLY_STOP_PAT,
    )
    log.info(f"Best val F1: {best_val_f1:.4f} at epoch {best_epoch}")

    # ── Test ──────────────────────────────────────────────────────────────────
    model.load_state_dict(torch.load(str(CKPT_PATH), map_location=device, weights_only=True))
    test_metrics = evaluate(model, test_loader, device, pos_weight)

    log.info("\n" + "=" * 60)
    log.info("TEST RESULTS: ToxiGene v5")
    log.info("=" * 60)
    log.info(f"  Macro F1  : {test_metrics['f1']:.4f}")
    log.info(f"  Accuracy  : {test_metrics['acc']:.4f}")
    log.info("  Per-class F1:")
    for cname, cf1 in zip(OUTCOME_NAMES, test_metrics["per_class_f1"]):
        log.info(f"    {cname:30s}: {cf1:.4f}")

    elapsed = time.time() - t0
    log.info(f"Elapsed: {elapsed:.1f}s")

    results = {
        "model": "ToxiGene_v5",
        "test_f1_macro": round(test_metrics["f1"], 6),
        "test_acc": round(test_metrics["acc"], 6),
        "per_class_f1": {
            name: round(f1, 6)
            for name, f1 in zip(OUTCOME_NAMES, test_metrics["per_class_f1"])
        },
        "n_train": N_TRAIN,
        "n_val": N_VAL,
        "n_test": N_TEST,
        "n_genes_input": N_GENES,
        "n_params": n_params,
        "best_val_f1": round(best_val_f1, 6),
        "best_epoch": best_epoch,
        "elapsed_s": round(elapsed, 2),
        "hyperparameters": {
            "n_genes": N_GENES,
            "hidden_dim": HIDDEN_DIM,
            "dropout": DROPOUT,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "early_stop_patience": EARLY_STOP_PAT,
            "normalization": "z-score + clip[-10,10]",
            "augmentation": {
                "gaussian_noise_prob": NOISE_PROB,
                "gaussian_noise_std": NOISE_STD,
                "gene_dropout_rate": GENE_DROP_RATE,
                "mixup_prob": MIXUP_PROB,
                "mixup_alpha": MIXUP_ALPHA,
            },
        },
    }

    with open(str(RESULTS_PATH), "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results saved to {RESULTS_PATH}")
    print(f"ToxiGene v5 TEST F1: {test_metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
