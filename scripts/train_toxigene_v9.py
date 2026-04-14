#!/usr/bin/env python3
"""ToxiGene v9 — Residual MLP with per-GSE batch correction on v3 data.

Key changes from v8:
  1. Per-GSE reference-batch normalization (Option A+C combined):
     - 843 new samples grouped by GSE accession
     - Each GSE batch shifted+scaled to match v2 gene-wise mean/std
     - Then global z-score + clip[-10,10] applied on top
  2. Residual backbone: 61479→2048 (skip 61479→2048), 2048→1024 (skip), 1024→512→256
  3. LayerNorm on input (sample-level, robust to batch distribution shifts)
  4. Higher dropout (0.35) and noise (gene_drop=0.15, noise_prob=0.50)
  5. LR warmup (5 epochs) + cosine decay
  6. Gradient clipping max_norm=1.0

Architecture (~196M params):
  input_norm: LayerNorm(61479)
  block1: Linear(61479,2048)+BN+ReLU+Drop | skip: Linear(61479,2048) → sum → ReLU
  block2: Linear(2048,1024)+BN+ReLU+Drop  | skip: Linear(2048,1024)  → sum → ReLU
  fc3: Linear(1024,512) → BN → GELU → Drop
  fc4: Linear(512,256)  → BN → GELU → Drop
  outcome_head: Linear(256,7)
  pathway_head: Linear(256,128)→GELU→Linear(128,200)→Softplus

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

# ── Device: prefer cuda:2 (least loaded), fallback to cuda:0 ─────────────────
CUDA_DEVICE = "cuda:2"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR  = PROJECT_ROOT / "data" / "processed" / "molecular"
CKPT_DIR  = PROJECT_ROOT / "checkpoints" / "molecular"
LOG_DIR   = PROJECT_ROOT / "logs"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

CORRECTED_DATA_PATH = DATA_DIR / "expression_matrix_v3_corrected.npy"
CKPT_PATH           = CKPT_DIR / "toxigene_v9_best.pt"
RESULTS_PATH        = CKPT_DIR / "results_v9.json"
LOG_PATH            = LOG_DIR  / "train_toxigene_v9.log"

# ── Hyperparameters ────────────────────────────────────────────────────────────
HIDDEN1        = 2048
HIDDEN2        = 1024
HIDDEN3        = 512
HIDDEN4        = 256
DROPOUT        = 0.35
N_PATHWAY      = 200
PATHWAY_LAMBDA = 0.3
PATHWAY_HIDDEN = 128

BATCH_SIZE    = 64
EPOCHS        = 400
LR            = 3e-4
WEIGHT_DECAY  = 0.01
GRAD_CLIP     = 1.0
EARLY_STOP    = 60
SEED          = 42
WARMUP_EPOCHS = 5

GENE_DROP_RATE = 0.15
NOISE_PROB     = 0.50
NOISE_STD      = 0.01

# Exact same split as v8 (seed=42, 70/15/15 on 2540 samples)
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
    log = logging.getLogger("toxigene_v9")
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
# Step 1: Per-GSE batch correction
# ─────────────────────────────────────────────────────────────────────────────
def apply_batch_correction(X_raw: np.ndarray, geo_meta: list, log) -> np.ndarray:
    """Per-GSE reference-batch normalization.

    Strategy:
    - The first 1697 rows are v2 samples (reference distribution, clean)
    - Rows 1697-2539 are new GEO samples, each tagged with a GSE in geo_meta
    - For each GSE group:
        1. Compute per-gene mean and std of that group
        2. Shift/scale each gene so the group matches v2's gene-wise mean/std
        3. This corrects platform-level offset and scaling differences
    - Then apply global z-score + clip[-10,10] using v2 (training reference) stats

    This is Reference Batch Normalization (analogous to ComBat's reference batch mode).
    """
    X = X_raw.astype(np.float64)  # work in float64 for precision
    n_v2 = 1697
    n_genes = X.shape[1]

    log.info(f"Applying per-GSE batch correction: {len(geo_meta)} new samples, {n_v2} reference (v2)")

    # Reference statistics from v2 samples
    ref_mean = X[:n_v2].mean(axis=0)           # (n_genes,)
    ref_std  = X[:n_v2].std(axis=0)            # (n_genes,)
    ref_std[ref_std < 1e-8] = 1.0

    log.info(f"  Reference (v2): global mean={ref_mean.mean():.4f}, global std={ref_std.mean():.4f}")

    # Build GSE → list of row indices (in the full X array, rows 1697+)
    gse_to_indices: dict[str, list[int]] = {}
    for i, m in enumerate(geo_meta):
        gse = m["gse"]
        if gse not in gse_to_indices:
            gse_to_indices[gse] = []
        gse_to_indices[gse].append(n_v2 + i)

    # Per-GSE correction
    total_corrected = 0
    for gse, idxs in sorted(gse_to_indices.items(), key=lambda kv: -len(kv[1])):
        batch = X[idxs]                               # (n_batch, n_genes)
        batch_mean = batch.mean(axis=0)               # (n_genes,)
        batch_std  = batch.std(axis=0)                # (n_genes,)
        batch_std[batch_std < 1e-8] = 1.0

        # Normalize batch to reference distribution (per-gene)
        # x_corrected = (x - batch_mean) / batch_std * ref_std + ref_mean
        batch_corrected = (batch - batch_mean) / batch_std * ref_std + ref_mean
        X[idxs] = batch_corrected

        # Diagnostic stats
        delta_mean = (batch_mean - ref_mean).mean()
        delta_std  = (batch_std / ref_std).mean()
        log.info(
            f"  {gse} (n={len(idxs):3d}): "
            f"pre_delta_mean={delta_mean:+.4f}, "
            f"pre_std_ratio={delta_std:.4f} → corrected"
        )
        total_corrected += len(idxs)

    log.info(f"  Total corrected: {total_corrected} new samples + {n_v2} v2 samples unchanged")
    log.info(f"  Post-correction: mean={X[n_v2:].mean():.4f}, std={X[n_v2:].std():.4f}")

    return X.astype(np.float32)


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
# Model: ToxiGeneV9 — Residual MLP with LayerNorm input
# ─────────────────────────────────────────────────────────────────────────────
class ToxiGeneV9(nn.Module):
    """Residual MLP backbone with dual-head multi-task output.

    Architecture:
      LayerNorm(n_genes)                        # sample-level normalization
      ResBlock1: fc1(n→2048)+BN+ReLU+Drop, skip1(n→2048) → sum → ReLU
      ResBlock2: fc2(2048→1024)+BN+ReLU+Drop,  skip2(2048→1024) → sum → ReLU
      fc3(1024→512) → BN → GELU → Drop
      fc4(512→256)  → BN → GELU → Drop
      outcome_head: Linear(256→7)               [BCE multi-label]
      pathway_head: Linear(256→128)→GELU→Linear(128→200)→Softplus  [Huber]
    """

    def __init__(
        self,
        n_genes: int = 61479,
        hidden1: int = HIDDEN1,    # 2048
        hidden2: int = HIDDEN2,    # 1024
        hidden3: int = HIDDEN3,    # 512
        hidden4: int = HIDDEN4,    # 256
        dropout: float = DROPOUT,
        n_outcomes: int = 7,
        n_pathways: int = N_PATHWAY,
        pathway_hidden: int = PATHWAY_HIDDEN,
    ):
        super().__init__()

        # Sample-level input normalization (robust to batch distribution shifts)
        self.input_norm = nn.LayerNorm(n_genes)

        # Residual block 1: n_genes → hidden1 (2048)
        self.fc1   = nn.Linear(n_genes, hidden1)
        self.bn1   = nn.BatchNorm1d(hidden1)
        self.skip1 = nn.Linear(n_genes, hidden1, bias=False)
        self.drop1 = nn.Dropout(dropout)

        # Residual block 2: hidden1 → hidden2 (1024)
        self.fc2   = nn.Linear(hidden1, hidden2)
        self.bn2   = nn.BatchNorm1d(hidden2)
        self.skip2 = nn.Linear(hidden1, hidden2, bias=False)
        self.drop2 = nn.Dropout(dropout)

        # Feed-forward layers
        self.fc3  = nn.Linear(hidden2, hidden3)
        self.bn3  = nn.BatchNorm1d(hidden3)
        self.drop3 = nn.Dropout(dropout)

        self.fc4  = nn.Linear(hidden3, hidden4)
        self.bn4  = nn.BatchNorm1d(hidden4)
        self.drop4 = nn.Dropout(dropout)

        # Task heads
        self.outcome_head = nn.Linear(hidden4, n_outcomes)

        self.pathway_head = nn.Sequential(
            nn.Linear(hidden4, pathway_hidden),
            nn.GELU(),
            nn.Linear(pathway_hidden, n_pathways),
            nn.Softplus(),
        )

        self._init_weights()

    def _init_weights(self):
        """Kaiming initialization for all linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # Input normalization (LayerNorm = per-sample)
        x = self.input_norm(x)

        # Residual block 1
        h1 = self.drop1(F.relu(self.bn1(self.fc1(x))))
        h1 = F.relu(h1 + self.skip1(x))

        # Residual block 2
        h2 = self.drop2(F.relu(self.bn2(self.fc2(h1))))
        h2 = F.relu(h2 + self.skip2(h1))

        # Feed-forward
        h3 = self.drop3(F.gelu(self.bn3(self.fc3(h2))))
        h4 = self.drop4(F.gelu(self.bn4(self.fc4(h3))))

        return {
            "outcome_logits": self.outcome_head(h4),
            "pathway_pred":   self.pathway_head(h4),
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
# LR warmup + cosine decay scheduler
# ─────────────────────────────────────────────────────────────────────────────
def make_scheduler(optimizer, warmup_epochs: int, total_epochs: int, eta_min: float = 1e-6):
    """Linear warmup for warmup_epochs, then cosine decay."""
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return eta_min / LR + (1.0 - eta_min / LR) * cosine
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────────────────────────────────────
# Threshold optimization
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
    scheduler = make_scheduler(optimizer, WARMUP_EPOCHS, EPOCHS, eta_min=1e-6)
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

        # Validate with class-specific threshold optimization
        val_probs, val_labels = collect_probs(model, val_loader, device)
        val_labels_bin = (val_labels > 0.5).astype(int)
        thresholds = optimize_thresholds(val_probs, val_labels_bin)
        val_preds = np.stack(
            [(val_probs[:, c] > thresholds[c]).astype(int)
             for c in range(val_labels.shape[1])], axis=1)
        val_f1 = f1_score(val_labels_bin, val_preds, average="macro", zero_division=0)

        if epoch % 10 == 0 or epoch == 1:
            log.info(
                f"[ToxiGeneV9] Epoch {epoch:3d}/{EPOCHS} | "
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
            log.info(f"  Early stopping at epoch {epoch} (best val F1={best_val_f1:.4f} @ epoch {best_epoch})")
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

    # ── Device selection ───────────────────────────────────────────────────────
    if torch.cuda.is_available():
        try:
            device = torch.device(CUDA_DEVICE)
            # Test allocation
            _ = torch.zeros(1, device=device)
            log.info(f"Device: {CUDA_DEVICE} ({torch.cuda.get_device_name(device)})")
        except Exception:
            device = torch.device("cuda:0")
            log.info(f"Fallback device: cuda:0")
    else:
        device = torch.device("cpu")
        log.info("Device: CPU")

    # ── Load raw data ──────────────────────────────────────────────────────────
    log.info("Loading v3 expression + outcome + pathway data …")
    X_raw = np.load(str(DATA_DIR / "expression_matrix_v3_expanded.npy"))
    y     = np.load(str(DATA_DIR / "outcome_labels_v3_expanded.npy")).astype(np.float32)
    p_raw = np.load(str(DATA_DIR / "pathway_labels_v3_expanded.npy")).astype(np.float32)
    log.info(f"  X: {X_raw.shape}, y: {y.shape}, p: {p_raw.shape}")

    # ── Step 1: Per-GSE batch correction ──────────────────────────────────────
    if CORRECTED_DATA_PATH.exists():
        log.info(f"Loading pre-corrected data from {CORRECTED_DATA_PATH}")
        X_corrected = np.load(str(CORRECTED_DATA_PATH))
    else:
        log.info("Computing per-GSE batch correction (will save for reuse) …")
        with open(str(DATA_DIR / "geo_v3_metadata.json")) as f:
            geo_meta = json.load(f)
        X_corrected = apply_batch_correction(X_raw, geo_meta, log)
        np.save(str(CORRECTED_DATA_PATH), X_corrected)
        log.info(f"Saved corrected data → {CORRECTED_DATA_PATH}")

    # ── Step 2: Same seed=42 split as v8 ──────────────────────────────────────
    N   = len(X_corrected)
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(N)
    tr_idx = idx[:N_TRAIN]
    va_idx = idx[N_TRAIN:N_TRAIN + N_VAL]
    te_idx = idx[N_TRAIN + N_VAL:]

    X_tr, y_tr, p_tr = X_corrected[tr_idx], y[tr_idx], p_raw[tr_idx]
    X_va, y_va, p_va = X_corrected[va_idx], y[va_idx], p_raw[va_idx]
    X_te, y_te, p_te = X_corrected[te_idx], y[te_idx], p_raw[te_idx]
    log.info(f"Split: {N_TRAIN} train / {N_VAL} val / {len(te_idx)} test (total={N})")

    # ── Step 3: Global z-score normalization (train stats, same as v8) ────────
    mu  = X_tr.mean(axis=0, keepdims=True)
    std = X_tr.std(axis=0,  keepdims=True)
    std[std < 1e-6] = 1.0
    X_tr = np.clip((X_tr - mu) / std, -10.0, 10.0).astype(np.float32)
    X_va = np.clip((X_va - mu) / std, -10.0, 10.0).astype(np.float32)
    X_te = np.clip((X_te - mu) / std, -10.0, 10.0).astype(np.float32)

    # ── Step 4: Normalize pathway labels ──────────────────────────────────────
    p_mu  = p_tr.mean(axis=0)
    p_std = p_tr.std(axis=0)
    p_std[p_std < 1e-6] = 1.0
    p_tr  = ((p_tr - p_mu) / p_std).astype(np.float32)
    p_va  = ((p_va - p_mu) / p_std).astype(np.float32)
    p_te  = ((p_te - p_mu) / p_std).astype(np.float32)
    log.info("Normalized expression (train-stats z-score + clip[-10,10]) and pathway labels")

    # ── Class weights ──────────────────────────────────────────────────────────
    pos_weight = compute_pos_weight(y_tr).to(device)
    log.info(f"pos_weight: {[f'{w:.2f}' for w in pos_weight.tolist()]}")
    log.info("Training class counts:")
    for i, name in enumerate(OUTCOME_NAMES):
        log.info(f"  {name:30s}: {int(y_tr[:,i].sum()):4d} pos")

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

    # ── Build ToxiGene v9 ──────────────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("Building ToxiGene v9 (residual MLP + per-GSE batch correction) …")
    model = ToxiGeneV9(
        n_genes=X_tr.shape[1],
        hidden1=HIDDEN1, hidden2=HIDDEN2, hidden3=HIDDEN3, hidden4=HIDDEN4,
        dropout=DROPOUT, n_outcomes=7, n_pathways=N_PATHWAY,
        pathway_hidden=PATHWAY_HIDDEN,
    ).to(device)

    n_params = sum(param.numel() for param in model.parameters())
    log.info(f"ToxiGene v9: {n_params:,} parameters")
    log.info(f"  Backbone: ResidualMLP ({HIDDEN1}→{HIDDEN2}→{HIDDEN3}→{HIDDEN4}) + LayerNorm input")
    log.info(f"  Residual shortcuts: {X_tr.shape[1]}→{HIDDEN1}, {HIDDEN1}→{HIDDEN2}")
    log.info(f"  Multi-task: outcome BCE + {PATHWAY_LAMBDA}×pathway Huber")
    log.info(f"  Regularization: dropout={DROPOUT}, gene_drop={GENE_DROP_RATE}, noise_prob={NOISE_PROB}")
    log.info(f"  LR schedule: {WARMUP_EPOCHS} warmup epochs + cosine decay")
    log.info(f"  Batch correction: per-GSE reference normalization to v2 distribution")
    log.info(f"  Data: v3 corrected ({N} samples)")
    log.info("=" * 70)
    log.info("Training …")

    best_val_f1, best_epoch = train(
        model, train_loader, val_loader, device, log, pos_weight)
    log.info(f"\nBest val F1: {best_val_f1:.4f} at epoch {best_epoch}")

    # ── Test evaluation ────────────────────────────────────────────────────────
    log.info("Loading best checkpoint …")
    ckpt = torch.load(str(CKPT_PATH), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    best_thresholds = ckpt["thresholds"]
    log.info(f"  Best thresholds: {[f'{t:.2f}' for t in best_thresholds]}")

    test_m    = evaluate(model, test_loader, device, best_thresholds)
    test_m_05 = evaluate(model, test_loader, device, None)

    elapsed = time.time() - t0

    # ── Print results ──────────────────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("FINAL RESULTS — ToxiGene v9 vs Baselines")
    log.info("=" * 70)
    log.info(f"{'Model':<30} {'F1 (opt thresh)':>16} {'F1 (t=0.5)':>12}")
    log.info("-" * 62)
    log.info(f"{'RandomForest':<30} {'0.8589':>16} {'0.8477':>12}")
    log.info(f"{'ExtraTrees':<30} {'0.8466':>16} {'0.8421':>12}")
    log.info(f"{'ToxiGene v8 (DL)':<30} {'0.8372':>16} {'0.8227':>12}")
    log.info(f"{'ToxiGene v9 (this)':<30} {test_m['f1']:>16.4f} {test_m_05['f1']:>12.4f}")
    log.info("-" * 62)
    delta_rf = test_m["f1"] - 0.8589
    log.info(f"  Delta vs RF: {delta_rf:+.4f}  ({'BEATS RF' if delta_rf > 0 else 'below RF'})")
    log.info("=" * 70)
    log.info(f"  Macro F1  : {test_m['f1']:.4f}")
    log.info(f"  Accuracy  : {test_m['acc']:.4f}")
    log.info("  Per-class F1:")
    for cname, cf1 in zip(OUTCOME_NAMES, test_m["per_class_f1"]):
        log.info(f"    {cname:30s}: {cf1:.4f}")
    log.info(f"  Macro F1 (threshold=0.5): {test_m_05['f1']:.4f}")
    log.info(f"Elapsed: {elapsed:.1f}s")

    # ── Save results ───────────────────────────────────────────────────────────
    results = {
        "model": "ToxiGene_v9",
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
        "vs_randomforest_f1": 0.8589,
        "delta_vs_rf": round(test_m["f1"] - 0.8589, 4),
        "beats_rf": test_m["f1"] > 0.8589,
        "toxigene_v8_f1": 0.8372,
        "delta_vs_v8": round(test_m["f1"] - 0.8372, 4),
        "n_train": N_TRAIN,
        "n_val": N_VAL,
        "n_test": len(te_idx),
        "n_total": N,
        "n_genes_input": int(X_tr.shape[1]),
        "n_pathway_targets": N_PATHWAY,
        "n_params": n_params,
        "best_val_f1": round(best_val_f1, 6),
        "best_epoch": best_epoch,
        "elapsed_s": round(elapsed, 2),
        "batch_correction": "per-GSE reference-batch normalization (v2 as reference)",
        "architecture": {
            "input_norm": "LayerNorm",
            "residual_blocks": [
                f"{X_tr.shape[1]}→{HIDDEN1} (with shortcut)",
                f"{HIDDEN1}→{HIDDEN2} (with shortcut)",
            ],
            "ffn_layers": [f"{HIDDEN2}→{HIDDEN3}", f"{HIDDEN3}→{HIDDEN4}"],
            "heads": ["outcome: Linear(256,7)", "pathway: Linear(256,128)→GELU→Linear(128,200)→Softplus"],
        },
        "hyperparameters": {
            "hidden1": HIDDEN1,
            "hidden2": HIDDEN2,
            "hidden3": HIDDEN3,
            "hidden4": HIDDEN4,
            "dropout": DROPOUT,
            "pathway_lambda": PATHWAY_LAMBDA,
            "pathway_hidden": PATHWAY_HIDDEN,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "grad_clip": GRAD_CLIP,
            "early_stop_patience": EARLY_STOP,
            "warmup_epochs": WARMUP_EPOCHS,
            "gene_drop_rate": GENE_DROP_RATE,
            "noise_prob": NOISE_PROB,
            "noise_std": NOISE_STD,
        },
    }

    with open(str(RESULTS_PATH), "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results saved to {RESULTS_PATH}")

    print("\n" + "=" * 70)
    print("ToxiGene v9 FINAL RESULTS")
    print("=" * 70)
    print(f"{'Model':<30} {'F1 (opt thresh)':>16} {'F1 (t=0.5)':>12}")
    print("-" * 62)
    print(f"{'RandomForest':<30} {'0.8589':>16} {'0.8477':>12}")
    print(f"{'ExtraTrees':<30} {'0.8466':>16} {'0.8421':>12}")
    print(f"{'ToxiGene v8 (DL)':<30} {'0.8372':>16} {'0.8227':>12}")
    print(f"{'ToxiGene v9 (this)':<30} {test_m['f1']:>16.4f} {test_m_05['f1']:>12.4f}")
    print("-" * 62)
    print(f"  Delta vs RF: {delta_rf:+.4f}  ({'BEATS RF!' if delta_rf > 0 else 'below RF'})")
    print(f"  Delta vs v8: {test_m['f1'] - 0.8372:+.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
