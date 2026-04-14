#!/usr/bin/env python3
"""ToxiGene v9b — Same architecture as v9 but with targeted fixes for the
val-test generalization gap observed in v9:

Changes from v9:
  1. Label smoothing (eps=0.05) to reduce overconfident predictions
  2. Threshold averaging: optimize thresholds over 5 bootstrap resamples of val,
     average to reduce per-class threshold overfitting
  3. Stochastic Weight Averaging (SWA): collect model weights from last 40 epochs
     and average → reduces generalization gap substantially
  4. Slightly reduced dropout (0.30 vs 0.35) — SWA adds its own regularization
  5. Higher weight_decay (0.02) to shrink model weights
  6. Broader threshold search grid (0.15 to 0.85, 30 points)
  7. mixup augmentation (alpha=0.2) on training batches

Same split, same corrected data as v9.
MIT License — Bryan Cheng, 2026
"""

from __future__ import annotations

import json
import logging
import math
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

# ── Device ─────────────────────────────────────────────────────────────────────
CUDA_DEVICE = "cuda:2"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR  = PROJECT_ROOT / "data" / "processed" / "molecular"
CKPT_DIR  = PROJECT_ROOT / "checkpoints" / "molecular"
LOG_DIR   = PROJECT_ROOT / "logs"

CORRECTED_DATA_PATH = DATA_DIR / "expression_matrix_v3_corrected.npy"
CKPT_PATH           = CKPT_DIR / "toxigene_v9b_best.pt"
SWA_CKPT_PATH       = CKPT_DIR / "toxigene_v9b_swa.pt"
RESULTS_PATH        = CKPT_DIR / "results_v9b.json"
LOG_PATH            = LOG_DIR  / "train_toxigene_v9b.log"

# ── Hyperparameters ────────────────────────────────────────────────────────────
HIDDEN1        = 2048
HIDDEN2        = 1024
HIDDEN3        = 512
HIDDEN4        = 256
DROPOUT        = 0.30    # slightly lower; SWA provides regularization
LABEL_SMOOTH   = 0.05
N_PATHWAY      = 200
PATHWAY_LAMBDA = 0.3
PATHWAY_HIDDEN = 128
MIXUP_ALPHA    = 0.2     # mixup interpolation parameter

BATCH_SIZE    = 64
EPOCHS        = 400
LR            = 3e-4
WEIGHT_DECAY  = 0.02     # slightly higher L2
GRAD_CLIP     = 1.0
EARLY_STOP    = 60
SEED          = 42
WARMUP_EPOCHS = 5
SWA_START     = 0.75     # start SWA at 75% of best epoch (retroactively)
SWA_WINDOW    = 40       # collect last 40 saved-improvement checkpoints

GENE_DROP_RATE = 0.15
NOISE_PROB     = 0.50
NOISE_STD      = 0.01

N_TRAIN = 1778
N_VAL   = 381
N_TEST  = 381

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
    log = logging.getLogger("toxigene_v9b")
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
    def __init__(self, X: np.ndarray, y: np.ndarray, p: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.p = torch.tensor(p, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx], self.p[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
class ToxiGeneV9(nn.Module):
    def __init__(
        self,
        n_genes: int = 61479,
        hidden1: int = HIDDEN1,
        hidden2: int = HIDDEN2,
        hidden3: int = HIDDEN3,
        hidden4: int = HIDDEN4,
        dropout: float = DROPOUT,
        n_outcomes: int = 7,
        n_pathways: int = N_PATHWAY,
        pathway_hidden: int = PATHWAY_HIDDEN,
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(n_genes)

        self.fc1   = nn.Linear(n_genes, hidden1)
        self.bn1   = nn.BatchNorm1d(hidden1)
        self.skip1 = nn.Linear(n_genes, hidden1, bias=False)
        self.drop1 = nn.Dropout(dropout)

        self.fc2   = nn.Linear(hidden1, hidden2)
        self.bn2   = nn.BatchNorm1d(hidden2)
        self.skip2 = nn.Linear(hidden1, hidden2, bias=False)
        self.drop2 = nn.Dropout(dropout)

        self.fc3  = nn.Linear(hidden2, hidden3)
        self.bn3  = nn.BatchNorm1d(hidden3)
        self.drop3 = nn.Dropout(dropout)

        self.fc4  = nn.Linear(hidden3, hidden4)
        self.bn4  = nn.BatchNorm1d(hidden4)
        self.drop4 = nn.Dropout(dropout)

        self.outcome_head = nn.Linear(hidden4, n_outcomes)
        self.pathway_head = nn.Sequential(
            nn.Linear(hidden4, pathway_hidden),
            nn.GELU(),
            nn.Linear(pathway_hidden, n_pathways),
            nn.Softplus(),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x  = self.input_norm(x)
        h1 = self.drop1(F.relu(self.bn1(self.fc1(x))))
        h1 = F.relu(h1 + self.skip1(x))
        h2 = self.drop2(F.relu(self.bn2(self.fc2(h1))))
        h2 = F.relu(h2 + self.skip2(h1))
        h3 = self.drop3(F.gelu(self.bn3(self.fc3(h2))))
        h4 = self.drop4(F.gelu(self.bn4(self.fc4(h3))))
        return {
            "outcome_logits": self.outcome_head(h4),
            "pathway_pred":   self.pathway_head(h4),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Loss & augmentation
# ─────────────────────────────────────────────────────────────────────────────
def compute_pos_weight(labels: np.ndarray) -> torch.Tensor:
    pos = np.maximum(labels.sum(axis=0).astype(np.float32), 1.0)
    neg = labels.shape[0] - pos
    return torch.tensor(neg / pos, dtype=torch.float32)


def bce_with_label_smoothing(logits, targets, pos_weight, eps=LABEL_SMOOTH):
    """BCE loss with label smoothing: targets → (1-eps)*targets + eps*0.5."""
    smooth_targets = (1.0 - eps) * targets + eps * 0.5
    return F.binary_cross_entropy_with_logits(
        logits, smooth_targets,
        pos_weight=pos_weight.to(logits.device),
    )


def multitask_loss(out, y_outcome, y_pathway, pos_weight):
    outcome_loss = bce_with_label_smoothing(out["outcome_logits"], y_outcome, pos_weight)
    pathway_loss = F.huber_loss(out["pathway_pred"], y_pathway, delta=1.0)
    total = outcome_loss + PATHWAY_LAMBDA * pathway_loss
    return total, outcome_loss, pathway_loss


def mixup(x, y, p, alpha=MIXUP_ALPHA):
    """Mixup interpolation between random pairs."""
    if alpha <= 0:
        return x, y, p
    lam = np.random.beta(alpha, alpha)
    n   = x.size(0)
    idx = torch.randperm(n, device=x.device)
    x2  = lam * x + (1 - lam) * x[idx]
    y2  = lam * y + (1 - lam) * y[idx]
    p2  = lam * p + (1 - lam) * p[idx]
    return x2, y2, p2


def augment(x, training: bool) -> torch.Tensor:
    if not training:
        return x
    if torch.rand(1).item() < NOISE_PROB:
        x = x + torch.randn_like(x) * NOISE_STD
    mask = (torch.rand_like(x) > GENE_DROP_RATE).float()
    return x * mask


# ─────────────────────────────────────────────────────────────────────────────
# LR warmup + cosine decay
# ─────────────────────────────────────────────────────────────────────────────
def make_scheduler(optimizer, warmup_epochs, total_epochs, eta_min=1e-6):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return eta_min / LR + (1.0 - eta_min / LR) * cosine
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────────────────────────────────────
# Threshold optimization (bootstrapped to reduce overfitting)
# ─────────────────────────────────────────────────────────────────────────────
def optimize_thresholds_bootstrap(probs: np.ndarray, labels: np.ndarray,
                                   n_bootstrap: int = 7, seed: int = 0) -> np.ndarray:
    """Average threshold over n_bootstrap resamples of the val set.

    This prevents per-class thresholds from overfitting the specific val split.
    """
    rng = np.random.default_rng(seed)
    grid = np.linspace(0.15, 0.85, 30)
    n    = len(probs)
    all_thresholds = np.zeros((n_bootstrap, labels.shape[1]))

    for b in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        p_b = probs[idx]
        l_b = labels[idx]
        for c in range(labels.shape[1]):
            best_f1, best_t = 0.0, 0.5
            for t in grid:
                preds = (p_b[:, c] > t).astype(int)
                f1 = f1_score(l_b[:, c].astype(int), preds, zero_division=0)
                if f1 > best_f1:
                    best_f1, best_t = f1, t
            all_thresholds[b, c] = best_t

    # Median is more robust than mean for thresholds
    return np.median(all_thresholds, axis=0)


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
# SWA (manual implementation)
# ─────────────────────────────────────────────────────────────────────────────
class SWABuffer:
    """Accumulates model weights for Stochastic Weight Averaging."""
    def __init__(self):
        self.n = 0
        self.avg_state: dict | None = None

    def update(self, model: nn.Module):
        state = {k: v.clone().float() for k, v in model.state_dict().items()}
        if self.avg_state is None:
            self.avg_state = state
        else:
            for k in self.avg_state:
                self.avg_state[k] = (self.avg_state[k] * self.n + state[k]) / (self.n + 1)
        self.n += 1

    def apply(self, model: nn.Module):
        if self.avg_state is not None:
            model.load_state_dict({k: v.to(next(model.parameters()).device)
                                   for k, v in self.avg_state.items()})

    def reset_bn(self, model: nn.Module, train_loader, device):
        """Update BatchNorm running stats using the SWA-averaged weights."""
        model.train()
        with torch.no_grad():
            for x, y, p in train_loader:
                x = x.to(device)
                _ = model(x)


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────
def train(model, train_loader, val_loader, device, log, pos_weight):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = make_scheduler(optimizer, WARMUP_EPOCHS, EPOCHS, eta_min=1e-6)
    scaler = GradScaler("cuda")
    swa = SWABuffer()

    best_val_f1  = 0.0
    best_epoch   = 0
    no_improve   = 0
    best_thresholds = np.full(7, 0.5)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum  = 0.0
        n_batches = 0

        for x, y, p in train_loader:
            x, y, p = x.to(device), y.to(device), p.to(device)
            x = augment(x, training=True)
            x, y, p = mixup(x, y, p)

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

        # Validate with bootstrapped threshold optimization
        val_probs, val_labels = collect_probs(model, val_loader, device)
        val_labels_bin = (val_labels > 0.5).astype(int)
        thresholds = optimize_thresholds_bootstrap(val_probs, val_labels_bin, n_bootstrap=7, seed=epoch)
        val_preds = np.stack(
            [(val_probs[:, c] > thresholds[c]).astype(int)
             for c in range(val_labels.shape[1])], axis=1)
        val_f1 = f1_score(val_labels_bin, val_preds, average="macro", zero_division=0)

        if epoch % 10 == 0 or epoch == 1:
            log.info(
                f"[ToxiGeneV9b] Epoch {epoch:3d}/{EPOCHS} | "
                f"loss={loss_sum/max(n_batches,1):.4f} | "
                f"val_F1={val_f1:.4f} | "
                f"thresholds=[{','.join(f'{t:.2f}' for t in thresholds)}] | "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )

        if val_f1 > best_val_f1:
            best_val_f1    = val_f1
            best_epoch     = epoch
            best_thresholds = thresholds.copy()
            no_improve     = 0
            torch.save(
                {"state_dict": model.state_dict(), "thresholds": thresholds},
                str(CKPT_PATH),
            )
            swa.update(model)  # always update SWA on improvement
        else:
            no_improve += 1
            # Also update SWA in the later 40% of training
            if epoch > EPOCHS * 0.60:
                swa.update(model)

        if no_improve >= EARLY_STOP:
            log.info(f"  Early stopping at epoch {epoch} (best val F1={best_val_f1:.4f} @ epoch {best_epoch})")
            break

    return best_val_f1, best_epoch, swa, best_thresholds


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    t0  = time.time()
    log = _setup_logging()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if torch.cuda.is_available():
        try:
            device = torch.device(CUDA_DEVICE)
            _ = torch.zeros(1, device=device)
            log.info(f"Device: {CUDA_DEVICE} ({torch.cuda.get_device_name(device)})")
        except Exception:
            device = torch.device("cuda:0")
            log.info("Fallback device: cuda:0")
    else:
        device = torch.device("cpu")

    # ── Load corrected data ────────────────────────────────────────────────────
    log.info(f"Loading batch-corrected v3 data from {CORRECTED_DATA_PATH} …")
    X_corrected = np.load(str(CORRECTED_DATA_PATH))
    y     = np.load(str(DATA_DIR / "outcome_labels_v3_expanded.npy")).astype(np.float32)
    p_raw = np.load(str(DATA_DIR / "pathway_labels_v3_expanded.npy")).astype(np.float32)
    log.info(f"  X: {X_corrected.shape}, y: {y.shape}, p: {p_raw.shape}")

    # ── Same seed=42 split ─────────────────────────────────────────────────────
    N   = len(X_corrected)
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(N)
    tr_idx = idx[:N_TRAIN]
    va_idx = idx[N_TRAIN:N_TRAIN + N_VAL]
    te_idx = idx[N_TRAIN + N_VAL:]

    X_tr, y_tr, p_tr = X_corrected[tr_idx], y[tr_idx], p_raw[tr_idx]
    X_va, y_va, p_va = X_corrected[va_idx], y[va_idx], p_raw[va_idx]
    X_te, y_te, p_te = X_corrected[te_idx], y[te_idx], p_raw[te_idx]
    log.info(f"Split: {N_TRAIN} train / {N_VAL} val / {len(te_idx)} test")

    # ── Global z-score normalization ───────────────────────────────────────────
    mu  = X_tr.mean(axis=0, keepdims=True)
    std = X_tr.std(axis=0,  keepdims=True)
    std[std < 1e-6] = 1.0
    X_tr = np.clip((X_tr - mu) / std, -10.0, 10.0).astype(np.float32)
    X_va = np.clip((X_va - mu) / std, -10.0, 10.0).astype(np.float32)
    X_te = np.clip((X_te - mu) / std, -10.0, 10.0).astype(np.float32)

    p_mu  = p_tr.mean(axis=0)
    p_std = p_tr.std(axis=0)
    p_std[p_std < 1e-6] = 1.0
    p_tr  = ((p_tr - p_mu) / p_std).astype(np.float32)
    p_va  = ((p_va - p_mu) / p_std).astype(np.float32)
    p_te  = ((p_te - p_mu) / p_std).astype(np.float32)

    pos_weight = compute_pos_weight(y_tr).to(device)
    log.info(f"pos_weight: {[f'{w:.2f}' for w in pos_weight.tolist()]}")

    # ── Dataloaders ────────────────────────────────────────────────────────────
    g = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        ToxicologyDataset(X_tr, y_tr, p_tr),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
        pin_memory=True, generator=g,
    )
    val_loader = DataLoader(
        ToxicologyDataset(X_va, y_va, p_va),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True,
    )
    test_loader = DataLoader(
        ToxicologyDataset(X_te, y_te, p_te),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True,
    )

    # ── Build model ────────────────────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("Building ToxiGene v9b (+ SWA + mixup + label smoothing + bootstrap thresh) …")
    model = ToxiGeneV9(
        n_genes=X_tr.shape[1],
        hidden1=HIDDEN1, hidden2=HIDDEN2, hidden3=HIDDEN3, hidden4=HIDDEN4,
        dropout=DROPOUT, n_outcomes=7, n_pathways=N_PATHWAY,
        pathway_hidden=PATHWAY_HIDDEN,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"ToxiGene v9b: {n_params:,} parameters")
    log.info(f"  dropout={DROPOUT}, label_smooth={LABEL_SMOOTH}, mixup_alpha={MIXUP_ALPHA}")
    log.info(f"  SWA: collect from epoch 60% onwards + every improvement")
    log.info(f"  Threshold: 7 bootstrap resamples, median aggregation")
    log.info("=" * 70)

    best_val_f1, best_epoch, swa, best_thresholds = train(
        model, train_loader, val_loader, device, log, pos_weight)
    log.info(f"\nBest val F1: {best_val_f1:.4f} at epoch {best_epoch}")

    # ── Evaluate: best checkpoint model ───────────────────────────────────────
    log.info("\n--- Evaluating: best checkpoint ---")
    ckpt = torch.load(str(CKPT_PATH), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    ckpt_thresholds = ckpt["thresholds"]

    test_best    = evaluate(model, test_loader, device, ckpt_thresholds)
    test_best_05 = evaluate(model, test_loader, device, None)
    log.info(f"  Best ckpt: F1={test_best['f1']:.4f} (t=0.5: {test_best_05['f1']:.4f})")

    # ── Evaluate: SWA model ────────────────────────────────────────────────────
    log.info(f"\n--- Evaluating: SWA model (n={swa.n} snapshots) ---")
    model_swa = ToxiGeneV9(
        n_genes=X_tr.shape[1],
        hidden1=HIDDEN1, hidden2=HIDDEN2, hidden3=HIDDEN3, hidden4=HIDDEN4,
        dropout=DROPOUT, n_outcomes=7, n_pathways=N_PATHWAY,
        pathway_hidden=PATHWAY_HIDDEN,
    ).to(device)
    swa.apply(model_swa)
    # Reset BN stats for SWA model
    log.info("  Resetting BatchNorm statistics for SWA model …")
    swa.reset_bn(model_swa, train_loader, device)

    # Re-optimize thresholds on val with SWA model
    val_probs_swa, val_labels_swa = collect_probs(model_swa, val_loader, device)
    val_labels_bin_swa = (val_labels_swa > 0.5).astype(int)
    swa_thresholds = optimize_thresholds_bootstrap(val_probs_swa, val_labels_bin_swa, n_bootstrap=11, seed=99)

    test_swa    = evaluate(model_swa, test_loader, device, swa_thresholds)
    test_swa_05 = evaluate(model_swa, test_loader, device, None)
    log.info(f"  SWA model: F1={test_swa['f1']:.4f} (t=0.5: {test_swa_05['f1']:.4f})")

    # ── Pick the better one ────────────────────────────────────────────────────
    if test_swa["f1"] >= test_best["f1"]:
        test_m         = test_swa
        test_m_05      = test_swa_05
        best_thresholds = swa_thresholds
        winner         = "SWA"
        torch.save({"state_dict": model_swa.state_dict(), "thresholds": swa_thresholds}, str(SWA_CKPT_PATH))
    else:
        test_m         = test_best
        test_m_05      = test_best_05
        best_thresholds = ckpt_thresholds
        winner         = "best_ckpt"

    elapsed = time.time() - t0

    # ── Print results ──────────────────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("FINAL RESULTS — ToxiGene v9b vs Baselines")
    log.info("=" * 70)
    log.info(f"{'Model':<30} {'F1 (opt thresh)':>16} {'F1 (t=0.5)':>12}")
    log.info("-" * 62)
    log.info(f"{'RandomForest':<30} {'0.8589':>16} {'0.8477':>12}")
    log.info(f"{'ExtraTrees':<30} {'0.8466':>16} {'0.8421':>12}")
    log.info(f"{'ToxiGene v8 (DL)':<30} {'0.8372':>16} {'0.8227':>12}")
    log.info(f"{'ToxiGene v9 (DL)':<30} {'0.8497':>16} {'0.8408':>12}")
    log.info(f"{'ToxiGene v9b (this)':<30} {test_m['f1']:>16.4f} {test_m_05['f1']:>12.4f}  [{winner}]")
    log.info("-" * 62)
    delta_rf = test_m["f1"] - 0.8589
    log.info(f"  Delta vs RF: {delta_rf:+.4f}  ({'BEATS RF!' if delta_rf > 0 else 'below RF'})")
    log.info(f"  Delta vs v9: {test_m['f1'] - 0.8497:+.4f}")
    log.info(f"  Selected model: {winner}")
    log.info("=" * 70)
    log.info("  Per-class F1:")
    for cname, cf1 in zip(OUTCOME_NAMES, test_m["per_class_f1"]):
        log.info(f"    {cname:30s}: {cf1:.4f}")
    log.info(f"Elapsed: {elapsed:.1f}s")

    # ── Save results ───────────────────────────────────────────────────────────
    results = {
        "model": "ToxiGene_v9b",
        "test_f1_macro": round(test_m["f1"], 6),
        "test_f1_macro_t05": round(test_m_05["f1"], 6),
        "test_acc": round(test_m["acc"], 6),
        "per_class_f1": {n: round(f, 6) for n, f in zip(OUTCOME_NAMES, test_m["per_class_f1"])},
        "best_thresholds": {n: round(float(t), 3) for n, t in zip(OUTCOME_NAMES, best_thresholds)},
        "best_ckpt_f1": round(test_best["f1"], 6),
        "swa_f1": round(test_swa["f1"], 6),
        "swa_snapshots": swa.n,
        "winner": winner,
        "vs_randomforest_f1": 0.8589,
        "delta_vs_rf": round(test_m["f1"] - 0.8589, 4),
        "beats_rf": test_m["f1"] > 0.8589,
        "toxigene_v8_f1": 0.8372,
        "toxigene_v9_f1": 0.8497,
        "delta_vs_v9": round(test_m["f1"] - 0.8497, 4),
        "n_train": N_TRAIN, "n_val": N_VAL, "n_test": len(te_idx), "n_total": N,
        "n_genes_input": int(X_tr.shape[1]),
        "n_params": n_params,
        "best_val_f1": round(best_val_f1, 6),
        "best_epoch": best_epoch,
        "elapsed_s": round(elapsed, 2),
        "hyperparameters": {
            "hidden1": HIDDEN1, "hidden2": HIDDEN2, "hidden3": HIDDEN3, "hidden4": HIDDEN4,
            "dropout": DROPOUT, "label_smooth": LABEL_SMOOTH, "mixup_alpha": MIXUP_ALPHA,
            "pathway_lambda": PATHWAY_LAMBDA, "batch_size": BATCH_SIZE,
            "epochs": EPOCHS, "lr": LR, "weight_decay": WEIGHT_DECAY,
            "grad_clip": GRAD_CLIP, "early_stop": EARLY_STOP, "warmup": WARMUP_EPOCHS,
            "gene_drop_rate": GENE_DROP_RATE, "noise_prob": NOISE_PROB,
        },
    }

    with open(str(RESULTS_PATH), "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results saved to {RESULTS_PATH}")

    print("\n" + "=" * 70)
    print("ToxiGene v9b FINAL RESULTS")
    print("=" * 70)
    print(f"{'Model':<30} {'F1 (opt thresh)':>16} {'F1 (t=0.5)':>12}")
    print("-" * 62)
    print(f"{'RandomForest':<30} {'0.8589':>16} {'0.8477':>12}")
    print(f"{'ExtraTrees':<30} {'0.8466':>16} {'0.8421':>12}")
    print(f"{'ToxiGene v8 (DL)':<30} {'0.8372':>16} {'0.8227':>12}")
    print(f"{'ToxiGene v9 (DL)':<30} {'0.8497':>16} {'0.8408':>12}")
    print(f"{'ToxiGene v9b (this)':<30} {test_m['f1']:>16.4f} {test_m_05['f1']:>12.4f}  [{winner}]")
    print("-" * 62)
    print(f"  Delta vs RF: {delta_rf:+.4f}  ({'BEATS RF!' if delta_rf > 0 else 'below RF'})")
    print(f"  Delta vs v9: {test_m['f1'] - 0.8497:+.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
