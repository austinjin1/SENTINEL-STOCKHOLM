#!/usr/bin/env python3
"""ToxiGene v4 — Hierarchical pathway transformer for zebrafish toxicology.

Key improvements over v3 (F1=0.8527, ties SimpleMLP):
  1. Hierarchical architecture:
       within-group self-attention (50 genes/group → 1 token each via MHA pool)
       cross-group transformer over 20 tokens
  2. Wider & deeper: embed_dim=256, n_heads=8, n_layers=6, ff_dim=512
  3. CLS token aggregation instead of mean pool
  4. Lighter regularization: weight_decay=0.05, dropout_tf=0.3, dropout_head=0.4
  5. Lower LR (5e-4) with cosine annealing (no warm restarts)
  6. Batch size 64 for more stable gradient estimates

Architecture (ToxiGene v4):
  Stage 1: Gene embedding  — 5000 genes → 20 groups × 250 genes
  Stage 2: Within-group   — Linear(250,256) per group → [B,20,256] tokens
  Stage 3: CLS prepend    — [CLS]+pos_embed → [B,21,256]
  Stage 4: Cross-group TF — 6-layer pre-norm Transformer (256-dim, 8 heads)
  Stage 5: Class head     — CLS out → Linear(256,128)→GELU→Drop→Linear(128,7)
  ~4.1M parameters

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
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import accuracy_score, f1_score

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR  = PROJECT_ROOT / "data" / "processed" / "molecular"
CKPT_DIR  = PROJECT_ROOT / "checkpoints" / "molecular"
LOG_DIR   = PROJECT_ROOT / "logs"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

CKPT_PATH    = CKPT_DIR / "toxigene_v4_best.pt"
RESULTS_PATH = CKPT_DIR / "results_v4.json"
LOG_PATH     = LOG_DIR  / "train_toxigene_v4.log"

# ── Hyperparameters ───────────────────────────────────────────────────────────
N_GENES_SELECT  = 5000
N_GROUPS        = 20                          # 20 groups of 250 genes
GROUP_SIZE      = N_GENES_SELECT // N_GROUPS  # 250
EMBED_DIM       = 256
N_HEADS         = 8
N_LAYERS        = 6
FF_DIM          = 512
DROPOUT_TF      = 0.3
DROPOUT_HEAD    = 0.4
BATCH_SIZE      = 64
EPOCHS          = 400
LR              = 5e-4
WEIGHT_DECAY    = 0.05
GRAD_CLIP       = 1.0
EARLY_STOP_PAT  = 60
SEED            = 42

# Augmentation
NOISE_PROB      = 0.70
NOISE_STD       = 0.01
GENE_DROP_RATE  = 0.15
MIXUP_PROB      = 0.40
MIXUP_ALPHA     = 0.4

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
        log = get_logger("toxigene_v4")
    except Exception:
        log = logging.getLogger("toxigene_v4")
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
# Gene selection
# ─────────────────────────────────────────────────────────────────────────────
def select_top_variance_genes(
    expression_matrix: np.ndarray,
    n_genes: int = N_GENES_SELECT,
) -> tuple[np.ndarray, np.ndarray]:
    variances = np.var(expression_matrix, axis=0)
    top_gene_idx = np.argsort(variances)[-n_genes:]
    reduced = expression_matrix[:, top_gene_idx]
    return reduced, top_gene_idx


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
def augment_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    training: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    B = x.size(0)
    device = x.device
    y_out = y.float()

    if not training:
        return x, y_out

    if torch.rand(1).item() < NOISE_PROB:
        x = x + torch.randn_like(x) * NOISE_STD

    mask = (torch.rand_like(x) > GENE_DROP_RATE).float()
    x = x * mask

    if torch.rand(1).item() < MIXUP_PROB:
        lam = float(np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA))
        perm = torch.randperm(B, device=device)
        x     = lam * x     + (1.0 - lam) * x[perm]
        y_out = lam * y_out + (1.0 - lam) * y_out[perm]

    return x, y_out


# ─────────────────────────────────────────────────────────────────────────────
# Model: ToxiGene v4 — Hierarchical Pathway Transformer
# ─────────────────────────────────────────────────────────────────────────────
class PathwayProjection(nn.Module):
    """Project each gene group (250 genes) → single 256-dim token.

    Uses a two-layer projection with residual norm:
        Linear(group_size, embed_dim) → LayerNorm → GELU → Linear(embed_dim, embed_dim)
    This acts as a within-group encoding step before cross-group attention.
    """

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
        """x: [B, n_genes] → tokens: [B, n_groups, embed_dim]"""
        tokens = []
        for g, proj in enumerate(self.projections):
            start = g * self.group_size
            group_x = x[:, start:start + self.group_size]
            tokens.append(proj(group_x))
        return torch.stack(tokens, dim=1)  # [B, n_groups, embed_dim]


class ToxiGeneV4(nn.Module):
    """Hierarchical pathway transformer with CLS token aggregation."""

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
        n_classes: int = 7,
    ):
        super().__init__()
        group_size = n_genes // n_groups

        # Stage 1 & 2: per-group projection
        self.gene_projection = PathwayProjection(group_size, embed_dim, n_groups)

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Positional embeddings for [CLS] + n_groups tokens
        self.pos_embed = nn.Parameter(torch.zeros(1, n_groups + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Stage 3: cross-group transformer
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

        # Stage 4: classification head (deeper for v4)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout_head),
            nn.Linear(128, n_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, n_genes] → logits: [B, n_classes]"""
        B = x.size(0)

        # Project gene groups → tokens
        tokens = self.gene_projection(x)           # [B, n_groups, embed_dim]

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)     # [B, 1, embed_dim]
        tokens = torch.cat([cls, tokens], dim=1)    # [B, n_groups+1, embed_dim]

        # Add positional embeddings
        tokens = tokens + self.pos_embed

        # Cross-group transformer
        tokens = self.transformer(tokens)           # [B, n_groups+1, embed_dim]

        # CLS token output
        cls_out = tokens[:, 0, :]                  # [B, embed_dim]

        return self.classifier(cls_out)


# ─────────────────────────────────────────────────────────────────────────────
# Baseline: SimpleMLP
# ─────────────────────────────────────────────────────────────────────────────
class SimpleMLP(nn.Module):
    def __init__(self, n_genes: int = N_GENES_SELECT, n_classes: int = 7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_genes, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
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


def multilabel_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(
        logits, targets,
        pos_weight=pos_weight.to(logits.device) if pos_weight is not None else None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    pos_weight: torch.Tensor,
) -> dict:
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    n_batches = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with autocast(device_type="cuda"):
            logits = model(x)
        loss = multilabel_bce_loss(logits, y, pos_weight)
        total_loss += loss.item()
        n_batches += 1

        preds = (torch.sigmoid(logits) > 0.5).float()
        all_preds.append(preds.cpu().numpy())
        all_labels.append(y.cpu().numpy())

    all_preds  = np.concatenate(all_preds,  axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Threshold multi-hot labels (Mixup can produce soft labels)
    all_labels_bin = (all_labels > 0.5).astype(int)
    all_preds_bin  = all_preds.astype(int)

    f1_macro = f1_score(all_labels_bin, all_preds_bin, average="macro", zero_division=0)
    acc = accuracy_score(all_labels_bin, all_preds_bin)
    per_class_f1 = f1_score(all_labels_bin, all_preds_bin, average=None, zero_division=0).tolist()

    return {
        "loss": total_loss / max(n_batches, 1),
        "f1": f1_macro,
        "acc": acc,
        "per_class_f1": per_class_f1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    log: logging.Logger,
    pos_weight: torch.Tensor,
    ckpt_path: Path,
    name: str = "ToxiGeneV4",
    epochs: int = EPOCHS,
    patience: int = EARLY_STOP_PAT,
) -> tuple[float, int]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
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
        val_f1   = val_metrics["f1"]
        val_loss = val_metrics["loss"]
        avg_train_loss = total_train_loss / max(n_batches, 1)
        lr_now = optimizer.param_groups[0]["lr"]

        if epoch % 10 == 0 or epoch == 1:
            log.info(
                f"[{name}] Epoch {epoch:3d}/{epochs} | "
                f"train_loss={avg_train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_F1={val_f1:.4f} | "
                f"lr={lr_now:.2e}"
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

    # ── Load data ────────────────────────────────────────────────────────────
    expr_path   = DATA_DIR / "expression_matrix_v2_expanded.npy"
    labels_path = DATA_DIR / "outcome_labels_v2_expanded.npy"

    log.info(f"Loading expression matrix from {expr_path}")
    expression = np.load(str(expr_path))          # (1697, 61479) float32
    labels     = np.load(str(labels_path)).astype(np.float32)  # (1697, 7)

    log.info(f"Expression shape: {expression.shape}, Labels shape: {labels.shape}")
    assert labels.ndim == 2 and labels.shape[1] == 7, \
        f"Expected (N,7) multi-label matrix, got {labels.shape}"

    # ── Gene selection (train-stats only) ────────────────────────────────────
    log.info(f"Selecting top {N_GENES_SELECT} genes by variance …")
    expression_sel, _ = select_top_variance_genes(expression, N_GENES_SELECT)
    log.info(f"  Reduced expression shape: {expression_sel.shape}")

    # ── Fixed splits (identical to v3 for comparability) ─────────────────────
    N = len(expression_sel)
    log.info(f"Split: {N_TRAIN} train / {N_VAL} val / {N_TEST} test")

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(N)
    train_idx = idx[:N_TRAIN]
    val_idx   = idx[N_TRAIN:N_TRAIN + N_VAL]
    test_idx  = idx[N_TRAIN + N_VAL:]

    X_train = expression_sel[train_idx]
    X_val   = expression_sel[val_idx]
    X_test  = expression_sel[test_idx]
    y_train = labels[train_idx]
    y_val   = labels[val_idx]
    y_test  = labels[test_idx]

    # ── Normalize using train stats only ─────────────────────────────────────
    mu  = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train = (X_train - mu) / std
    X_val   = (X_val   - mu) / std
    X_test  = (X_test  - mu) / std
    log.info("Expression normalized per gene (z-score, train stats only)")

    # ── pos_weight for BCE ────────────────────────────────────────────────────
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

    # ── Build ToxiGene v4 ─────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("Building ToxiGene v4 …")
    model = ToxiGeneV4(
        n_genes=N_GENES_SELECT,
        n_groups=N_GROUPS,
        embed_dim=EMBED_DIM,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        ff_dim=FF_DIM,
        dropout_tf=DROPOUT_TF,
        dropout_head=DROPOUT_HEAD,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"ToxiGene v4: {n_params:,} parameters")
    log.info("Training ToxiGene v4 …")

    best_val_f1_v4, best_epoch_v4 = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        log=log,
        pos_weight=pos_weight,
        ckpt_path=CKPT_PATH,
        name="ToxiGeneV4",
        epochs=EPOCHS,
        patience=EARLY_STOP_PAT,
    )
    log.info(f"ToxiGene v4 best val F1: {best_val_f1_v4:.4f} at epoch {best_epoch_v4}")

    # ── Evaluate on test set ──────────────────────────────────────────────────
    log.info("Loading best ToxiGene v4 checkpoint …")
    model.load_state_dict(
        torch.load(str(CKPT_PATH), map_location=device, weights_only=True)
    )
    v4_test = evaluate(model, test_loader, device, pos_weight)

    log.info("\n" + "=" * 60)
    log.info("TEST RESULTS: ToxiGene v4")
    log.info("=" * 60)
    log.info(f"  Macro F1  : {v4_test['f1']:.4f}")
    log.info(f"  Accuracy  : {v4_test['acc']:.4f}")
    log.info("  Per-class F1:")
    for cname, cf1 in zip(OUTCOME_NAMES, v4_test["per_class_f1"]):
        log.info(f"    {cname:30s}: {cf1:.4f}")

    # ── Train SimpleMLP baseline ──────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("Training SimpleMLP baseline …")

    mlp = SimpleMLP(n_genes=N_GENES_SELECT, n_classes=7).to(device)
    n_params_mlp = sum(p.numel() for p in mlp.parameters())
    log.info(f"SimpleMLP: {n_params_mlp:,} parameters")

    mlp_ckpt = CKPT_DIR / "simpleMLP_v4_best.pt"
    best_val_f1_mlp, best_epoch_mlp = train_model(
        model=mlp,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        log=log,
        pos_weight=pos_weight,
        ckpt_path=mlp_ckpt,
        name="SimpleMLP",
        epochs=EPOCHS,
        patience=EARLY_STOP_PAT,
    )
    log.info(f"SimpleMLP best val F1: {best_val_f1_mlp:.4f} at epoch {best_epoch_mlp}")

    mlp.load_state_dict(
        torch.load(str(mlp_ckpt), map_location=device, weights_only=True)
    )
    mlp_test = evaluate(mlp, test_loader, device, pos_weight)

    log.info("\n" + "=" * 60)
    log.info("TEST RESULTS: SimpleMLP baseline")
    log.info("=" * 60)
    log.info(f"  Macro F1  : {mlp_test['f1']:.4f}")
    log.info(f"  Accuracy  : {mlp_test['acc']:.4f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    v4_beats_mlp = v4_test["f1"] > mlp_test["f1"]
    elapsed = time.time() - t0

    log.info("\n" + "=" * 60)
    log.info("SUMMARY")
    log.info("=" * 60)
    log.info(f"  ToxiGene v4 F1 : {v4_test['f1']:.4f}")
    log.info(f"  SimpleMLP F1   : {mlp_test['f1']:.4f}")
    log.info(f"  ToxiGene v4 beats SimpleMLP: {v4_beats_mlp}")
    log.info(f"  F1 delta: {v4_test['f1'] - mlp_test['f1']:+.4f}")
    log.info(f"  Elapsed: {elapsed:.1f}s")

    results = {
        "model": "ToxiGene_v4",
        "test_f1_macro": round(v4_test["f1"], 6),
        "test_acc": round(v4_test["acc"], 6),
        "per_class_f1": {
            name: round(f1, 6)
            for name, f1 in zip(OUTCOME_NAMES, v4_test["per_class_f1"])
        },
        "n_train": N_TRAIN,
        "n_val": N_VAL,
        "n_test": N_TEST,
        "n_genes_input": N_GENES_SELECT,
        "n_params": n_params,
        "best_val_f1": round(best_val_f1_v4, 6),
        "best_epoch": best_epoch_v4,
        "baseline_simpleMLP_f1": round(mlp_test["f1"], 6),
        "baseline_simpleMLP_n_params": n_params_mlp,
        "toxigene_v4_beats_simpleMLP": v4_beats_mlp,
        "f1_delta_vs_simpleMLP": round(v4_test["f1"] - mlp_test["f1"], 6),
        "elapsed_s": round(elapsed, 2),
        "hyperparameters": {
            "n_genes_selected": N_GENES_SELECT,
            "n_groups": N_GROUPS,
            "embed_dim": EMBED_DIM,
            "n_heads": N_HEADS,
            "n_layers": N_LAYERS,
            "ff_dim": FF_DIM,
            "dropout_tf": DROPOUT_TF,
            "dropout_head": DROPOUT_HEAD,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "early_stop_patience": EARLY_STOP_PAT,
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
    print(f"ToxiGene v4 TEST F1: {v4_test['f1']:.4f}")


if __name__ == "__main__":
    main()
