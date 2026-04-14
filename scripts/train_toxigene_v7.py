#!/usr/bin/env python3
"""ToxiGene v7 — Multi-task SimpleMLP (baseline arch + pathway supervision).

Diagnosis: transformer architectures (v2 83M, v6 6.8M) consistently lose to
SimpleMLP (31.6M) on 1187 training samples. Small-data regime favors simpler
inductive bias. v7 takes the WINNING baseline arch and adds the two proven
improvements from v6:
  1. Multi-task pathway supervision: BCE(7 outcomes) + λ*Huber(200 pathways)
     → richer gradient signal, better minority class representations
  2. Class-specific thresholds: per-class F1-optimal threshold on val set
     → especially helps neurotoxicity (18.5% pos) and immunosuppression (20.2%)

Architecture (ToxiGene v7):
  61479 genes
  → Linear(61479, 512) → BN → ReLU → Dropout(0.3)
  → Linear(512, 256)   → BN → ReLU → Dropout(0.3)   ← shared repr
  → outcome_head: Linear(256, 7)                      [BCE multi-label]
  → pathway_head: Linear(256, 128) → GELU → Linear(128, 200) → Softplus  [Huber]
  ~31.7M params

Training: identical to the SimpleMLP_v2 baseline (seed=42, same split),
augmentation kept minimal so as not to degrade the baseline's strong performance.

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
from sklearn.metrics import f1_score, accuracy_score

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR  = PROJECT_ROOT / "data" / "processed" / "molecular"
CKPT_DIR  = PROJECT_ROOT / "checkpoints" / "molecular"
LOG_DIR   = PROJECT_ROOT / "logs"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

CKPT_PATH    = CKPT_DIR / "toxigene_v7_best.pt"
RESULTS_PATH = CKPT_DIR / "results_v7.json"
LOG_PATH     = LOG_DIR  / "train_toxigene_v7.log"

# ── Hyperparameters ───────────────────────────────────────────────────────────
# Match baseline as closely as possible; add only pathway head
HIDDEN1        = 512
HIDDEN2        = 256
DROPOUT        = 0.3
N_PATHWAY      = 200
PATHWAY_LAMBDA = 0.3      # BCE + 0.3 * Huber
PATHWAY_HIDDEN = 128

BATCH_SIZE    = 64
EPOCHS        = 400
LR            = 3e-4
WEIGHT_DECAY  = 0.01      # lighter than v6 (0.05) — matches baseline
GRAD_CLIP     = 1.0
EARLY_STOP    = 50
SEED          = 42

# Minimal augmentation: only gene dropout (no Mixup — baseline had none)
GENE_DROP_RATE = 0.10
NOISE_PROB     = 0.40
NOISE_STD      = 0.01

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
        log = get_logger("toxigene_v7")
    except Exception:
        log = logging.getLogger("toxigene_v7")
        if not log.handlers:
            log.setLevel(logging.INFO)
            ch = logging.StreamHandler(sys.stdout)
            ch.setFormatter(logging.Formatter(
                "[%(asctime)s] %(message)s", datefmt="%H:%M:%S"))
            log.addHandler(ch)
            log.propagate = False
    fh = logging.FileHandler(str(LOG_PATH), mode="w")
    fh.setFormatter(logging.Formatter(
        "[%(asctime)s] %(message)s", datefmt="%H:%M:%S"))
    fh.setLevel(logging.INFO)
    log.addHandler(fh)
    return log


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class ToxicologyDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, p: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.p = torch.tensor(p, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx], self.p[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Model: SimpleMLP + pathway head
# ─────────────────────────────────────────────────────────────────────────────
class ToxiGeneV7(nn.Module):
    """SimpleMLP backbone with dual-head multi-task output.

    Shared representation (256-d) feeds both:
      - outcome_head: 7-class multi-label classification
      - pathway_head: 200-dim pathway activity regression (auxiliary task)
    """

    def __init__(
        self,
        n_genes: int,
        hidden1: int = HIDDEN1,
        hidden2: int = HIDDEN2,
        dropout: float = DROPOUT,
        n_outcomes: int = 7,
        n_pathways: int = N_PATHWAY,
        pathway_hidden: int = PATHWAY_HIDDEN,
    ):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(n_genes, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.outcome_head = nn.Linear(hidden2, n_outcomes)

        self.pathway_head = nn.Sequential(
            nn.Linear(hidden2, pathway_hidden),
            nn.GELU(),
            nn.Linear(pathway_hidden, n_pathways),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.backbone(x)
        return {
            "outcome_logits": self.outcome_head(h),
            "pathway_pred":   self.pathway_head(h),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Loss & helpers
# ─────────────────────────────────────────────────────────────────────────────
def compute_pos_weight(labels: np.ndarray) -> torch.Tensor:
    pos = np.maximum(labels.sum(axis=0).astype(np.float32), 1.0)
    neg = labels.shape[0] - pos
    return torch.tensor(neg / pos, dtype=torch.float32)


def multitask_loss(
    out: dict,
    y_outcome: torch.Tensor,
    y_pathway: torch.Tensor,
    pos_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    outcome_loss = F.binary_cross_entropy_with_logits(
        out["outcome_logits"], y_outcome,
        pos_weight=pos_weight.to(y_outcome.device),
    )
    pathway_loss = F.huber_loss(out["pathway_pred"], y_pathway, delta=1.0)
    total = outcome_loss + PATHWAY_LAMBDA * pathway_loss
    return total, outcome_loss, pathway_loss


def augment(x: torch.Tensor, training: bool) -> torch.Tensor:
    if not training:
        return x
    if torch.rand(1).item() < NOISE_PROB:
        x = x + torch.randn_like(x) * NOISE_STD
    mask = (torch.rand_like(x) > GENE_DROP_RATE).float()
    return x * mask


# ─────────────────────────────────────────────────────────────────────────────
# Threshold optimization (from v6)
# ─────────────────────────────────────────────────────────────────────────────
def optimize_thresholds(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    grid = np.linspace(0.2, 0.8, 25)
    thresholds = np.zeros(labels.shape[1])
    for c in range(labels.shape[1]):
        best_f1, best_t = 0.0, 0.5
        for t in grid:
            preds = (probs[:, c] > t).astype(int)
            f1 = f1_score(labels[:, c].astype(int), preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[c] = best_t
    return thresholds


@torch.no_grad()
def collect_probs(model, loader, device):
    model.eval()
    probs_list, labels_list = [], []
    for x, y, p in loader:
        x = x.to(device)
        with autocast(device_type="cuda"):
            out = model(x)
        probs_list.append(torch.sigmoid(out["outcome_logits"]).cpu().numpy())
        labels_list.append(y.numpy())
    return np.concatenate(probs_list), np.concatenate(labels_list)


def evaluate(model, loader, device, thresholds=None):
    probs, labels = collect_probs(model, loader, device)
    labels_bin = (labels > 0.5).astype(int)
    if thresholds is None:
        thresholds = np.full(labels.shape[1], 0.5)
    preds = np.stack(
        [(probs[:, c] > thresholds[c]).astype(int) for c in range(labels.shape[1])],
        axis=1,
    )
    f1_macro = f1_score(labels_bin, preds, average="macro", zero_division=0)
    acc      = accuracy_score(labels_bin, preds)
    per_cls  = f1_score(labels_bin, preds, average=None, zero_division=0).tolist()
    return {"f1": f1_macro, "acc": acc, "per_class_f1": per_cls}


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────
def train(model, train_loader, val_loader, device, log, pos_weight):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler = GradScaler("cuda")

    best_val_f1 = 0.0
    best_epoch  = 0
    no_improve  = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        n_batches = 0

        for x, y, p in train_loader:
            x, y, p = x.to(device), y.to(device), p.to(device)
            x = augment(x, training=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda"):
                out = model(x)
                total, _, _ = multitask_loss(out, y, p, pos_weight)

            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            loss_sum  += total.item()
            n_batches += 1

        scheduler.step()

        # Validate with class-specific thresholds
        val_probs, val_labels = collect_probs(model, val_loader, device)
        val_labels_bin = (val_labels > 0.5).astype(int)
        thresholds = optimize_thresholds(val_probs, val_labels_bin)
        val_preds = np.stack(
            [(val_probs[:, c] > thresholds[c]).astype(int)
             for c in range(val_labels.shape[1])], axis=1)
        val_f1 = f1_score(val_labels_bin, val_preds, average="macro", zero_division=0)

        if epoch % 10 == 0 or epoch == 1:
            log.info(
                f"[ToxiGeneV7] Epoch {epoch:3d}/{EPOCHS} | "
                f"loss={loss_sum/max(n_batches,1):.4f} | "
                f"val_F1={val_f1:.4f} | "
                f"thresholds=[{','.join(f'{t:.2f}' for t in thresholds)}] | "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch  = epoch
            no_improve  = 0
            torch.save(
                {"state_dict": model.state_dict(), "thresholds": thresholds},
                str(CKPT_PATH),
            )
        else:
            no_improve += 1

        if no_improve >= EARLY_STOP:
            log.info(f"  Early stopping at epoch {epoch} (best val F1={best_val_f1:.4f})")
            break

    return best_val_f1, best_epoch


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    t0  = time.time()
    log = _setup_logging()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # ── Load data ────────────────────────────────────────────────────────────
    log.info("Loading expression + outcome + pathway data …")
    X = np.load(str(DATA_DIR / "expression_matrix_v2_expanded.npy"))   # (1697, 61479)
    y = np.load(str(DATA_DIR / "outcome_labels_v2_expanded.npy")).astype(np.float32)
    p = np.load(str(DATA_DIR / "pathway_labels_v2_expanded.npy")).astype(np.float32)
    log.info(f"  X: {X.shape}, y: {y.shape}, p: {p.shape}")

    # ── Split (same seed=42, same indices as v2/v3/v6) ───────────────────────
    N   = len(X)
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(N)
    tr_idx = idx[:N_TRAIN]
    va_idx = idx[N_TRAIN:N_TRAIN + N_VAL]
    te_idx = idx[N_TRAIN + N_VAL:]

    X_tr, y_tr, p_tr = X[tr_idx], y[tr_idx], p[tr_idx]
    X_va, y_va, p_va = X[va_idx], y[va_idx], p[va_idx]
    X_te, y_te, p_te = X[te_idx], y[te_idx], p[te_idx]
    log.info(f"Split: {N_TRAIN} / {N_VAL} / {N_TEST}")

    # ── Normalize expression (z-score + clip, train stats) ───────────────────
    mu  = X_tr.mean(axis=0, keepdims=True)
    std = X_tr.std(axis=0,  keepdims=True)
    std[std < 1e-6] = 1.0
    X_tr = np.clip((X_tr - mu) / std, -10.0, 10.0).astype(np.float32)
    X_va = np.clip((X_va - mu) / std, -10.0, 10.0).astype(np.float32)
    X_te = np.clip((X_te - mu) / std, -10.0, 10.0).astype(np.float32)

    # ── Normalize pathway labels (z-score, train stats) ──────────────────────
    p_mu  = p_tr.mean(axis=0)
    p_std = p_tr.std(axis=0)
    p_std[p_std < 1e-6] = 1.0
    p_tr  = ((p_tr - p_mu) / p_std).astype(np.float32)
    p_va  = ((p_va - p_mu) / p_std).astype(np.float32)
    p_te  = ((p_te - p_mu) / p_std).astype(np.float32)
    log.info("Normalized expression and pathway labels")

    # ── Class weights ─────────────────────────────────────────────────────────
    pos_weight = compute_pos_weight(y_tr).to(device)
    log.info(f"pos_weight: {[f'{w:.2f}' for w in pos_weight.tolist()]}")
    log.info("Training class counts:")
    for i, name in enumerate(OUTCOME_NAMES):
        log.info(f"  {name:30s}: {int(y_tr[:,i].sum()):4d} pos")

    # ── Dataloaders ───────────────────────────────────────────────────────────
    g = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        ToxicologyDataset(X_tr, y_tr, p_tr),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=2,
        pin_memory=True, generator=g,
    )
    val_loader = DataLoader(
        ToxicologyDataset(X_va, y_va, p_va),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True,
    )
    test_loader = DataLoader(
        ToxicologyDataset(X_te, y_te, p_te),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True,
    )

    # ── Build model ───────────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("Building ToxiGene v7 (multi-task SimpleMLP) …")
    model = ToxiGeneV7(
        n_genes=X_tr.shape[1],
        hidden1=HIDDEN1, hidden2=HIDDEN2, dropout=DROPOUT,
        n_outcomes=7, n_pathways=N_PATHWAY, pathway_hidden=PATHWAY_HIDDEN,
    ).to(device)
    n_params = sum(param.numel() for param in model.parameters())
    log.info(f"ToxiGene v7: {n_params:,} parameters")
    log.info(f"  Backbone: SimpleMLP ({HIDDEN1}→{HIDDEN2}) + pathway head")
    log.info(f"  Multi-task: outcome BCE + {PATHWAY_LAMBDA}×pathway Huber")
    log.info("Training …")

    best_val_f1, best_epoch = train(
        model, train_loader, val_loader, device, log, pos_weight)
    log.info(f"Best val F1: {best_val_f1:.4f} at epoch {best_epoch}")

    # ── Test evaluation ───────────────────────────────────────────────────────
    log.info("Loading best checkpoint …")
    ckpt = torch.load(str(CKPT_PATH), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    best_thresholds = ckpt["thresholds"]
    log.info(f"  Best thresholds: {[f'{t:.2f}' for t in best_thresholds]}")

    test_m    = evaluate(model, test_loader, device, best_thresholds)
    test_m_05 = evaluate(model, test_loader, device, None)

    log.info("\n" + "=" * 60)
    log.info("TEST RESULTS: ToxiGene v7")
    log.info("=" * 60)
    log.info(f"  Macro F1  : {test_m['f1']:.4f}")
    log.info(f"  Accuracy  : {test_m['acc']:.4f}")
    log.info("  Per-class F1:")
    for cname, cf1 in zip(OUTCOME_NAMES, test_m["per_class_f1"]):
        log.info(f"    {cname:30s}: {cf1:.4f}")
    log.info(f"  Macro F1 (threshold=0.5): {test_m_05['f1']:.4f}")

    elapsed = time.time() - t0
    log.info(f"Elapsed: {elapsed:.1f}s")

    results = {
        "model": "ToxiGene_v7",
        "test_f1_macro": round(test_m["f1"], 6),
        "test_f1_macro_t05": round(test_m_05["f1"], 6),
        "test_acc": round(test_m["acc"], 6),
        "per_class_f1": {
            name: round(f1, 6)
            for name, f1 in zip(OUTCOME_NAMES, test_m["per_class_f1"])
        },
        "best_thresholds": {
            name: round(float(t), 3)
            for name, t in zip(OUTCOME_NAMES, best_thresholds)
        },
        "baseline_simpleMLP_f1": 0.8896,
        "delta_vs_baseline": round(test_m["f1"] - 0.8896, 4),
        "n_train": N_TRAIN,
        "n_val": N_VAL,
        "n_test": N_TEST,
        "n_genes_input": int(X_tr.shape[1]),
        "n_pathway_targets": N_PATHWAY,
        "n_params": n_params,
        "best_val_f1": round(best_val_f1, 6),
        "best_epoch": best_epoch,
        "elapsed_s": round(elapsed, 2),
        "hyperparameters": {
            "hidden1": HIDDEN1,
            "hidden2": HIDDEN2,
            "dropout": DROPOUT,
            "pathway_lambda": PATHWAY_LAMBDA,
            "pathway_hidden": PATHWAY_HIDDEN,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "early_stop_patience": EARLY_STOP,
            "gene_drop_rate": GENE_DROP_RATE,
            "noise_prob": NOISE_PROB,
        },
    }

    with open(str(RESULTS_PATH), "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results saved to {RESULTS_PATH}")
    print(
        f"ToxiGene v7 TEST F1: {test_m['f1']:.4f}  "
        f"(baseline: 0.8896, delta: {test_m['f1']-0.8896:+.4f})"
    )


if __name__ == "__main__":
    main()
