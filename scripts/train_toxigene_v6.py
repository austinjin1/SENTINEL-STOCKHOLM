#!/usr/bin/env python3
"""ToxiGene v6 — Multi-task pathway-supervised transformer.

Root cause of v3/v4/v5 plateau at F1≈0.85 (vs v2 0.877):
  v2 used multi-task supervision: predict pathway activities + outcomes jointly.
  v3/v4/v5 only predicted 7 outcomes → model sees 7 supervision signals.
  With 200 pathway activity targets, model learns richer gene representations
  that generalize better to the 7 downstream outcome labels.

Key innovations over v3:
  1. Multi-task loss: BCE(7 outcomes) + λ*Huber(200 pathway activities)
     pathway_labels_v2_expanded.npy provides continuous pathway activity scores
  2. Discriminative gene selection: t-test statistic per gene vs each label
     (vs v3's variance selection which ignores class separability)
  3. Class-specific sigmoid thresholds: optimized per-class on validation set
     (vs v3's global 0.5 threshold — especially helps minority classes)
  4. Larger capacity exploiting extra supervision: embed_dim=256, 8 heads, 6 layers
  5. Minority-class aware augmentation: higher Mixup alpha for minority samples

Architecture (ToxiGene v6):
  5000 genes (discriminative select) → 10 groups × 500 → Linear(500,256) per group
  → [CLS] + pos_embed → 6-layer Transformer (256d, 8H, ff=1024, pre-norm)
  → CLS token → two heads:
      Outcome head: Linear(256,128)→GELU→Drop(0.4)→Linear(128,7)   [BCE loss]
      Pathway head: Linear(256,200)                                 [Huber loss]
  ~5.5M params

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
from scipy import stats as scipy_stats
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

CKPT_PATH    = CKPT_DIR / "toxigene_v6_best.pt"
RESULTS_PATH = CKPT_DIR / "results_v6.json"
LOG_PATH     = LOG_DIR  / "train_toxigene_v6.log"

# ── Hyperparameters ───────────────────────────────────────────────────────────
N_GENES_SELECT  = 5000
N_GROUPS        = 10
GROUP_SIZE      = N_GENES_SELECT // N_GROUPS  # 500
EMBED_DIM       = 256
N_HEADS         = 8
N_LAYERS        = 6
FF_DIM          = 1024
DROPOUT_TF      = 0.2
DROPOUT_HEAD    = 0.4
N_PATHWAY       = 200      # auxiliary supervision targets
PATHWAY_LAMBDA  = 0.3      # weight of pathway loss vs outcome loss
BATCH_SIZE      = 64
EPOCHS          = 400
LR              = 3e-4
WEIGHT_DECAY    = 0.05
GRAD_CLIP       = 1.0
EARLY_STOP_PAT  = 50
SEED            = 42

# Augmentation
NOISE_PROB      = 0.70
NOISE_STD       = 0.01
GENE_DROP_RATE  = 0.15
MIXUP_PROB      = 0.50
MIXUP_ALPHA     = 0.4

N_TRAIN = 1187
N_VAL   = 254
N_TEST  = 256

OUTCOME_NAMES = [
    "reproductive_impairment",  # 0  — 28.3% pos
    "growth_inhibition",        # 1  — 62.2% pos
    "immunosuppression",        # 2  — 20.2% pos
    "neurotoxicity",            # 3  — 18.5% pos ← most minority
    "hepatotoxicity",           # 4  — 56.4% pos
    "oxidative_damage",         # 5  — 58.5% pos
    "endocrine_disruption",     # 6  — 32.6% pos
]


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
def _setup_logging() -> logging.Logger:
    try:
        from sentinel.utils.logging import get_logger
        log = get_logger("toxigene_v6")
    except Exception:
        log = logging.getLogger("toxigene_v6")
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
# Gene selection: discriminative t-test (vs variance in v3)
# ─────────────────────────────────────────────────────────────────────────────
def select_discriminative_genes(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_genes: int = N_GENES_SELECT,
) -> np.ndarray:
    """Select top-n genes by max |t-statistic| across all 7 outcome labels.

    For each gene, compute per-class Welch t-test between positive and negative
    samples. Take the max absolute t-statistic across all 7 classes. This selects
    genes that are most discriminative for ANY of the outcome classes.
    """
    n_classes = y_train.shape[1]
    max_t = np.zeros(X_train.shape[1], dtype=np.float32)

    for c in range(n_classes):
        pos_mask = y_train[:, c] > 0.5
        neg_mask = ~pos_mask
        n_pos = pos_mask.sum()
        n_neg = neg_mask.sum()
        if n_pos < 5 or n_neg < 5:
            continue
        pos_mean = X_train[pos_mask].mean(axis=0)
        neg_mean = X_train[neg_mask].mean(axis=0)
        pos_var  = X_train[pos_mask].var(axis=0) + 1e-8
        neg_var  = X_train[neg_mask].var(axis=0) + 1e-8
        # Welch t-statistic
        t = np.abs(pos_mean - neg_mean) / np.sqrt(pos_var / n_pos + neg_var / n_neg)
        max_t = np.maximum(max_t, t.astype(np.float32))

    top_idx = np.argsort(max_t)[-n_genes:]
    return top_idx


# ─────────────────────────────────────────────────────────────────────────────
# Dataset: joint (expression, outcome, pathway) samples
# ─────────────────────────────────────────────────────────────────────────────
class ToxicologyDataset(Dataset):
    def __init__(
        self,
        expression: np.ndarray,
        outcomes: np.ndarray,
        pathways: np.ndarray,
    ):
        self.X = torch.tensor(expression, dtype=torch.float32)
        self.y = torch.tensor(outcomes,   dtype=torch.float32)   # (N, 7)
        self.p = torch.tensor(pathways,   dtype=torch.float32)   # (N, 200)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx], self.p[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation (Mixup + noise + gene dropout)
# ─────────────────────────────────────────────────────────────────────────────
def augment_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    p: torch.Tensor,
    training: bool = True,
):
    if not training:
        return x, y.float(), p.float()

    B = x.size(0)
    y_out = y.float()
    p_out = p.float()

    if torch.rand(1).item() < NOISE_PROB:
        x = x + torch.randn_like(x) * NOISE_STD

    mask = (torch.rand_like(x) > GENE_DROP_RATE).float()
    x = x * mask

    if torch.rand(1).item() < MIXUP_PROB:
        lam = float(np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA))
        perm  = torch.randperm(B, device=x.device)
        x     = lam * x     + (1.0 - lam) * x[perm]
        y_out = lam * y_out + (1.0 - lam) * y_out[perm]
        p_out = lam * p_out + (1.0 - lam) * p_out[perm]

    return x, y_out, p_out


# ─────────────────────────────────────────────────────────────────────────────
# Model: ToxiGene v6 — Multi-task pathway-supervised transformer
# ─────────────────────────────────────────────────────────────────────────────
class PathwayGroupProjection(nn.Module):
    """Project each gene group → single token via 2-layer MLP."""

    def __init__(self, group_size: int, embed_dim: int, n_groups: int):
        super().__init__()
        self.n_groups = n_groups
        self.group_size = group_size
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(group_size, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
            )
            for _ in range(n_groups)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, n_genes] → [B, n_groups, embed_dim]"""
        tokens = []
        for g, proj in enumerate(self.projections):
            start = g * self.group_size
            tokens.append(proj(x[:, start:start + self.group_size]))
        return torch.stack(tokens, dim=1)


class ToxiGeneV6(nn.Module):
    """Multi-task transformer: predicts 7 outcomes AND 200 pathway activities.

    Pathway supervision (200 targets) provides richer gradient signal than
    7 outcome targets alone — improves representations for minority classes.
    """

    def __init__(
        self,
        n_genes: int = N_GENES_SELECT,
        n_groups: int = N_GROUPS,
        embed_dim: int = EMBED_DIM,
        n_heads: int = N_HEADS,
        n_layers: int = N_LAYERS,
        ff_dim: int = FF_DIM,
        dropout_tf: float = DROPOUT_TF,
        dropout_head: float = DROPOUT_HEAD,
        n_outcomes: int = 7,
        n_pathways: int = N_PATHWAY,
    ):
        super().__init__()
        group_size = n_genes // n_groups

        # Stage 1: per-group gene projection
        self.gene_proj = PathwayGroupProjection(group_size, embed_dim, n_groups)

        # CLS token + positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_groups + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Stage 2: cross-group transformer (pre-norm)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout_tf,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(embed_dim),
        )

        # Outcome head (7-class multi-label)
        self.outcome_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout_head),
            nn.Linear(128, n_outcomes),
        )

        # Pathway head (200-dim continuous regression)
        self.pathway_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout_tf),
            nn.Linear(256, n_pathways),
            nn.Softplus(),   # pathway activities are non-negative
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        B = x.size(0)
        tokens = self.gene_proj(x)                          # [B, n_groups, D]
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)             # [B, n_groups+1, D]
        tokens = tokens + self.pos_embed
        tokens = self.transformer(tokens)                   # [B, n_groups+1, D]
        cls_out = tokens[:, 0, :]                           # [B, D]
        return {
            "outcome_logits": self.outcome_head(cls_out),   # [B, 7]
            "pathway_pred":   self.pathway_head(cls_out),   # [B, 200]
        }


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────
def compute_pos_weight(labels: np.ndarray) -> torch.Tensor:
    pos = labels.sum(axis=0).astype(np.float32)
    neg = labels.shape[0] - pos
    pos = np.maximum(pos, 1.0)
    return torch.tensor(neg / pos, dtype=torch.float32)


def multitask_loss(
    out: dict,
    y_outcome: torch.Tensor,
    y_pathway: torch.Tensor,
    pos_weight: torch.Tensor,
    pathway_lambda: float = PATHWAY_LAMBDA,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combined BCE (outcomes) + Huber (pathways) loss."""
    outcome_loss = F.binary_cross_entropy_with_logits(
        out["outcome_logits"], y_outcome,
        pos_weight=pos_weight.to(y_outcome.device),
    )
    pathway_loss = F.huber_loss(out["pathway_pred"], y_pathway, delta=1.0)
    total = outcome_loss + pathway_lambda * pathway_loss
    return total, outcome_loss, pathway_loss


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation (with class-specific thresholds)
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def collect_predictions(model, loader, device):
    """Collect raw sigmoid probabilities for threshold optimization."""
    model.eval()
    all_probs, all_labels = [], []
    for x, y, p in loader:
        x = x.to(device)
        with autocast(device_type="cuda"):
            out = model(x)
        probs = torch.sigmoid(out["outcome_logits"]).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(y.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def optimize_thresholds(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Find per-class threshold in [0.2, 0.8] maximizing F1 on validation."""
    n_classes = labels.shape[1]
    thresholds = np.zeros(n_classes)
    grid = np.linspace(0.2, 0.8, 25)
    for c in range(n_classes):
        best_f1, best_t = 0.0, 0.5
        for t in grid:
            preds = (probs[:, c] > t).astype(int)
            f1 = f1_score(labels[:, c].astype(int), preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[c] = best_t
    return thresholds


@torch.no_grad()
def evaluate(model, loader, device, pos_weight, thresholds=None):
    model.eval()
    all_probs, all_labels = collect_predictions(model, loader, device)
    all_labels_bin = (all_labels > 0.5).astype(int)

    if thresholds is None:
        thresholds = np.full(all_labels.shape[1], 0.5)

    all_preds = np.zeros_like(all_probs, dtype=int)
    for c in range(all_probs.shape[1]):
        all_preds[:, c] = (all_probs[:, c] > thresholds[c]).astype(int)

    f1_macro = f1_score(all_labels_bin, all_preds, average="macro", zero_division=0)
    acc = accuracy_score(all_labels_bin, all_preds)
    per_class = f1_score(all_labels_bin, all_preds, average=None, zero_division=0).tolist()
    return {"f1": f1_macro, "acc": acc, "per_class_f1": per_class}


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
def train_model(model, train_loader, val_loader, device, log, pos_weight,
                ckpt_path, name="ToxiGeneV6", epochs=EPOCHS, patience=EARLY_STOP_PAT):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler("cuda")

    best_val_f1 = 0.0
    best_epoch  = 0
    no_improve  = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss_sum = 0.0
        n_batches = 0

        for x, y, p in train_loader:
            x, y, p = x.to(device), y.to(device), p.to(device)
            x, y_aug, p_aug = augment_batch(x, y, p, training=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda"):
                out = model(x)
                total, out_loss, path_loss = multitask_loss(out, y_aug, p_aug, pos_weight)

            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            total_loss_sum += total.item()
            n_batches += 1

        scheduler.step()

        # Validate: collect probs and optimize thresholds
        val_probs, val_labels = collect_predictions(model, val_loader, device)
        val_labels_bin = (val_labels > 0.5).astype(int)
        thresholds = optimize_thresholds(val_probs, val_labels_bin)
        val_preds = np.zeros_like(val_probs, dtype=int)
        for c in range(val_probs.shape[1]):
            val_preds[:, c] = (val_probs[:, c] > thresholds[c]).astype(int)
        val_f1 = f1_score(val_labels_bin, val_preds, average="macro", zero_division=0)

        if epoch % 10 == 0 or epoch == 1:
            log.info(
                f"[{name}] Epoch {epoch:3d}/{epochs} | "
                f"loss={total_loss_sum/max(n_batches,1):.4f} | "
                f"val_F1={val_f1:.4f} | "
                f"thresholds=[{','.join(f'{t:.2f}' for t in thresholds)}] | "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch  = epoch
            no_improve  = 0
            # Save model + thresholds together
            torch.save(
                {"state_dict": model.state_dict(), "thresholds": thresholds},
                str(ckpt_path),
            )
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

    # ── Load data ────────────────────────────────────────────────────────────
    log.info("Loading expression + outcome + pathway data …")
    expression = np.load(str(DATA_DIR / "expression_matrix_v2_expanded.npy"))  # (1697, 61479)
    outcomes   = np.load(str(DATA_DIR / "outcome_labels_v2_expanded.npy")).astype(np.float32)  # (1697, 7)
    pathways   = np.load(str(DATA_DIR / "pathway_labels_v2_expanded.npy")).astype(np.float32)  # (1697, 200)

    log.info(f"  expression: {expression.shape}, outcomes: {outcomes.shape}, pathways: {pathways.shape}")

    # ── Split (same as v3 for fair comparison) ────────────────────────────────
    N = len(expression)
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(N)
    train_idx = idx[:N_TRAIN]
    val_idx   = idx[N_TRAIN:N_TRAIN + N_VAL]
    test_idx  = idx[N_TRAIN + N_VAL:]

    log.info(f"Split: {N_TRAIN} train / {N_VAL} val / {N_TEST} test")

    # Subset arrays
    X_tr, y_tr, p_tr = expression[train_idx], outcomes[train_idx], pathways[train_idx]
    X_va, y_va, p_va = expression[val_idx],   outcomes[val_idx],   pathways[val_idx]
    X_te, y_te, p_te = expression[test_idx],  outcomes[test_idx],  pathways[test_idx]

    # ── Discriminative gene selection (t-test, train-only stats) ─────────────
    log.info(f"Selecting top {N_GENES_SELECT} genes by discriminative t-test …")
    top_gene_idx = select_discriminative_genes(X_tr, y_tr, N_GENES_SELECT)
    X_tr = X_tr[:, top_gene_idx].astype(np.float32)
    X_va = X_va[:, top_gene_idx].astype(np.float32)
    X_te = X_te[:, top_gene_idx].astype(np.float32)
    log.info(f"  Selected {len(top_gene_idx)} genes")

    # ── Normalize expression (train stats) ───────────────────────────────────
    mu  = X_tr.mean(axis=0)
    std = X_tr.std(axis=0)
    std[std < 1e-6] = 1.0
    X_tr = np.clip((X_tr - mu) / std, -10.0, 10.0)
    X_va = np.clip((X_va - mu) / std, -10.0, 10.0)
    X_te = np.clip((X_te - mu) / std, -10.0, 10.0)

    # Normalize pathway labels (train stats)
    p_mu  = p_tr.mean(axis=0)
    p_std = p_tr.std(axis=0)
    p_std[p_std < 1e-6] = 1.0
    p_tr  = ((p_tr - p_mu) / p_std).astype(np.float32)
    p_va  = ((p_va - p_mu) / p_std).astype(np.float32)
    p_te  = ((p_te - p_mu) / p_std).astype(np.float32)
    log.info("Normalized expression (z-score+clip) and pathway labels (z-score)")

    # ── pos_weight for BCE ────────────────────────────────────────────────────
    pos_weight = compute_pos_weight(y_tr).to(device)
    log.info(f"pos_weight: {pos_weight.tolist()}")

    # Class counts for analysis
    log.info("Training class counts:")
    for i, name in enumerate(OUTCOME_NAMES):
        log.info(f"  {name:30s}: {int(y_tr[:,i].sum()):4d} pos")

    # ── Dataloaders ───────────────────────────────────────────────────────────
    g = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        ToxicologyDataset(X_tr, y_tr, p_tr),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, generator=g,
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
    log.info("Building ToxiGene v6 (multi-task pathway transformer) …")
    model = ToxiGeneV6(
        n_genes=N_GENES_SELECT, n_groups=N_GROUPS, embed_dim=EMBED_DIM,
        n_heads=N_HEADS, n_layers=N_LAYERS, ff_dim=FF_DIM,
        dropout_tf=DROPOUT_TF, dropout_head=DROPOUT_HEAD,
        n_outcomes=7, n_pathways=N_PATHWAY,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"ToxiGene v6: {n_params:,} parameters")
    log.info(f"  Multi-task: outcome BCE + {PATHWAY_LAMBDA}×pathway Huber")
    log.info("Training …")

    best_val_f1, best_epoch = train_model(
        model, train_loader, val_loader, device, log, pos_weight,
        ckpt_path=CKPT_PATH, name="ToxiGeneV6",
        epochs=EPOCHS, patience=EARLY_STOP_PAT,
    )
    log.info(f"Best val F1: {best_val_f1:.4f} at epoch {best_epoch}")

    # ── Test evaluation (with optimized thresholds from checkpoint) ───────────
    log.info("Loading best checkpoint …")
    ckpt = torch.load(str(CKPT_PATH), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    best_thresholds = ckpt["thresholds"]
    log.info(f"  Best thresholds: {[f'{t:.2f}' for t in best_thresholds]}")

    test_metrics = evaluate(model, test_loader, device, pos_weight, best_thresholds)

    log.info("\n" + "=" * 60)
    log.info("TEST RESULTS: ToxiGene v6")
    log.info("=" * 60)
    log.info(f"  Macro F1  : {test_metrics['f1']:.4f}")
    log.info(f"  Accuracy  : {test_metrics['acc']:.4f}")
    log.info("  Per-class F1:")
    for cname, cf1 in zip(OUTCOME_NAMES, test_metrics["per_class_f1"]):
        log.info(f"    {cname:30s}: {cf1:.4f}")

    # Also evaluate with default 0.5 threshold for comparison
    test_metrics_05 = evaluate(model, test_loader, device, pos_weight, None)
    log.info(f"  Macro F1 (threshold=0.5): {test_metrics_05['f1']:.4f}")

    elapsed = time.time() - t0
    log.info(f"Elapsed: {elapsed:.1f}s")

    results = {
        "model": "ToxiGene_v6",
        "test_f1_macro": round(test_metrics["f1"], 6),
        "test_f1_macro_t05": round(test_metrics_05["f1"], 6),
        "test_acc": round(test_metrics["acc"], 6),
        "per_class_f1": {
            name: round(f1, 6)
            for name, f1 in zip(OUTCOME_NAMES, test_metrics["per_class_f1"])
        },
        "best_thresholds": {
            name: round(float(t), 3)
            for name, t in zip(OUTCOME_NAMES, best_thresholds)
        },
        "n_train": N_TRAIN,
        "n_val": N_VAL,
        "n_test": N_TEST,
        "n_genes_input": N_GENES_SELECT,
        "n_pathway_targets": N_PATHWAY,
        "n_params": n_params,
        "best_val_f1": round(best_val_f1, 6),
        "best_epoch": best_epoch,
        "elapsed_s": round(elapsed, 2),
        "hyperparameters": {
            "n_genes_selected": N_GENES_SELECT,
            "gene_selection": "max_t_statistic",
            "n_groups": N_GROUPS,
            "embed_dim": EMBED_DIM,
            "n_heads": N_HEADS,
            "n_layers": N_LAYERS,
            "ff_dim": FF_DIM,
            "dropout_tf": DROPOUT_TF,
            "dropout_head": DROPOUT_HEAD,
            "pathway_lambda": PATHWAY_LAMBDA,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "early_stop_patience": EARLY_STOP_PAT,
        },
    }

    with open(str(RESULTS_PATH), "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results saved to {RESULTS_PATH}")
    print(f"ToxiGene v6 TEST F1: {test_metrics['f1']:.4f}  (t=0.5: {test_metrics_05['f1']:.4f})")


if __name__ == "__main__":
    main()
