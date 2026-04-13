#!/usr/bin/env python3
"""ToxiGene v3 — Compact pathway-grouped transformer for zebrafish toxicology.

Root cause of v2 failure: 83M params severely overfits 1,697 samples → F1=0.8770.
SimpleMLP baseline (31M) achieves F1=0.8896.

Fix strategy:
  1. Gene selection: top-5000 genes by variance (61,479 → 5,000)
  2. Architecture: ~3M params, pathway-grouped transformer
  3. Strong regularization: weight_decay=0.1, dropout=0.4/0.5, label_smoothing=0.2
  4. Data augmentation: Gaussian noise, gene dropout, Mixup
  5. Early stopping on val macro F1 (patience=50)

Architecture (ToxiGene v3):
  - 10 pathway groups × 500 genes each
  - Each group projected to 128-dim token via Linear(500, 128)
  - 4-layer Transformer (embed_dim=128, nheads=4, ff_dim=256, dropout=0.4, pre-norm)
  - Mean pool 10 tokens → 128-dim
  - Class head: Linear(128,64) → GELU → Dropout(0.5) → Linear(64,7)
  - Total: ~3M parameters

MIT License — Bryan Cheng, 2026
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

# ── Project root & sys.path ───────────────────────────────────────────────────
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

CKPT_PATH    = CKPT_DIR / "toxigene_v3_best.pt"
RESULTS_PATH = CKPT_DIR / "results_v3.json"
LOG_PATH     = LOG_DIR  / "train_toxigene_v3.log"

# ── Hyperparameters ───────────────────────────────────────────────────────────
N_GENES_SELECT  = 5000
N_GROUPS        = 10
GROUP_SIZE      = N_GENES_SELECT // N_GROUPS   # 500
EMBED_DIM       = 128
N_HEADS         = 4
N_LAYERS        = 4
FF_DIM          = 256
DROPOUT_TF      = 0.4
DROPOUT_HEAD    = 0.5
BATCH_SIZE      = 32
EPOCHS          = 300
LR              = 1e-3
WEIGHT_DECAY    = 0.1
GRAD_CLIP       = 1.0
EARLY_STOP_PAT  = 50
LABEL_SMOOTHING = 0.2
SEED            = 42

# Augmentation
NOISE_PROB      = 0.70
NOISE_STD       = 0.01
GENE_DROP_RATE  = 0.20
MIXUP_PROB      = 0.40
MIXUP_ALPHA     = 0.4

N_TRAIN = 1187
N_VAL   = 254
N_TEST  = 256

OUTCOME_NAMES = [
    "reproductive_impairment",  # 0
    "growth_inhibition",        # 1
    "immunosuppression",        # 2
    "neurotoxicity",            # 3
    "hepatotoxicity",           # 4
    "oxidative_damage",         # 5
    "endocrine_disruption",     # 6
]


# ─────────────────────────────────────────────────────────────────────────────
# Logging — file + sentinel Rich logger
# ─────────────────────────────────────────────────────────────────────────────
def _setup_logging() -> logging.Logger:
    """Configure a logger that writes to both console (Rich) and a log file."""
    try:
        from sentinel.utils.logging import get_logger
        log = get_logger("toxigene_v3")
    except Exception:
        log = logging.getLogger("toxigene_v3")
        if not log.handlers:
            log.setLevel(logging.INFO)
            ch = logging.StreamHandler(sys.stdout)
            ch.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S"))
            log.addHandler(ch)
            log.propagate = False

    # Also tee to file
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
    """Return (reduced_matrix, top_gene_indices) selecting top-n by variance."""
    variances = np.var(expression_matrix, axis=0)
    top_gene_idx = np.argsort(variances)[-n_genes:]
    reduced = expression_matrix[:, top_gene_idx]
    return reduced, top_gene_idx


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class ToxicologyDataset(Dataset):
    """Multi-label toxicology dataset. Labels are (N, 7) float32 multi-hot vectors."""

    def __init__(self, expression: np.ndarray, labels: np.ndarray):
        self.X = torch.tensor(expression, dtype=torch.float32)
        # labels shape: (N, 7) float32 multi-hot
        self.y = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation helpers
# ─────────────────────────────────────────────────────────────────────────────
def augment_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    training: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Gaussian noise, gene dropout, and/or Mixup to a batch.

    y is already (B, 7) float32 multi-hot. Returns (augmented_x, augmented_y).
    """
    B = x.size(0)
    device = x.device
    y_out = y.float()

    if not training:
        return x, y_out

    # 1. Gaussian noise (70% of batches)
    if torch.rand(1).item() < NOISE_PROB:
        x = x + torch.randn_like(x) * NOISE_STD

    # 2. Random gene dropout per sample
    mask = (torch.rand_like(x) > GENE_DROP_RATE).float()
    x = x * mask

    # 3. Mixup (40% of batches) — blend multi-hot labels directly
    if torch.rand(1).item() < MIXUP_PROB:
        lam = float(np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA))
        perm = torch.randperm(B, device=device)
        x     = lam * x     + (1.0 - lam) * x[perm]
        y_out = lam * y_out + (1.0 - lam) * y_out[perm]

    return x, y_out


# ─────────────────────────────────────────────────────────────────────────────
# Model: ToxiGene v3
# ─────────────────────────────────────────────────────────────────────────────
class PathwayGroupProjection(nn.Module):
    """Project 10 non-overlapping gene groups of 500 each to 128-dim tokens."""

    def __init__(
        self,
        n_genes: int = N_GENES_SELECT,
        n_groups: int = N_GROUPS,
        embed_dim: int = EMBED_DIM,
    ):
        super().__init__()
        self.n_groups = n_groups
        self.group_size = n_genes // n_groups
        # One Linear per group — avoids a single 5000×128 matrix that would
        # mix all genes together before regularization can operate on groups
        self.projections = nn.ModuleList([
            nn.Linear(self.group_size, embed_dim)
            for _ in range(n_groups)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, n_genes]
        Returns:
            tokens: [B, n_groups, embed_dim]
        """
        tokens = []
        for g, proj in enumerate(self.projections):
            start = g * self.group_size
            end = start + self.group_size
            group_x = x[:, start:end]           # [B, group_size]
            t = proj(group_x)                    # [B, embed_dim]
            tokens.append(t)
        tokens = torch.stack(tokens, dim=1)      # [B, n_groups, embed_dim]
        tokens = self.norm(tokens)
        return tokens


class ToxiGeneV3(nn.Module):
    """Compact pathway-grouped transformer for zebrafish toxicology.

    Stage 1 — Gene projection:
        Split 5000 genes into 10 groups of 500. Each group projected to
        128-dim via an independent linear layer → 10 tokens.

    Stage 2 — Transformer encoder:
        4 layers, embed_dim=128, nheads=4, ff_dim=256, dropout=0.4, pre-norm.

    Stage 3 — Aggregation:
        Mean pool all 10 tokens → 128-dim.

    Stage 4 — Class head:
        Linear(128, 64) → GELU → Dropout(0.5) → Linear(64, 7).
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
        n_classes: int = 7,
    ):
        super().__init__()

        # Stage 1: gene group projection
        self.gene_projection = PathwayGroupProjection(
            n_genes=n_genes,
            n_groups=n_groups,
            embed_dim=embed_dim,
        )

        # Learnable positional embeddings for the n_groups tokens
        self.pos_embed = nn.Parameter(torch.zeros(1, n_groups, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Stage 2: transformer encoder (pre-norm via norm_first=True)
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

        # Stage 4: classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout_head),
            nn.Linear(64, n_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, n_genes] float32

        Returns:
            logits: [B, n_classes]
        """
        # Stage 1: project gene groups → tokens
        tokens = self.gene_projection(x)         # [B, n_groups, embed_dim]
        tokens = tokens + self.pos_embed         # add positional embeddings

        # Stage 2: transformer
        tokens = self.transformer(tokens)        # [B, n_groups, embed_dim]

        # Stage 3: mean pool
        pooled = tokens.mean(dim=1)              # [B, embed_dim]

        # Stage 4: classify
        logits = self.classifier(pooled)         # [B, n_classes]
        return logits


# ─────────────────────────────────────────────────────────────────────────────
# Baseline: SimpleMLP (same as v2 baseline, operating on 5000 genes)
# ─────────────────────────────────────────────────────────────────────────────
class SimpleMLP(nn.Module):
    """SimpleMLP baseline: Linear(5000,512)→ReLU→Drop(0.3)→
    Linear(512,256)→ReLU→Drop(0.3)→Linear(256,7)."""

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
# Loss: cross-entropy with class weights + label smoothing
# ─────────────────────────────────────────────────────────────────────────────
def compute_pos_weight(labels: np.ndarray) -> torch.Tensor:
    """Compute pos_weight for BCE: (n_neg / n_pos) per class.
    labels: (N, 7) float32 multi-hot
    """
    pos = labels.sum(axis=0).astype(np.float32)
    neg = labels.shape[0] - pos
    pos = np.maximum(pos, 1.0)
    pos_weight = neg / pos
    return torch.tensor(pos_weight, dtype=torch.float32)


def multilabel_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Binary CE loss for multi-label classification."""
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
    """Multi-label evaluation: sigmoid threshold=0.5, macro F1 over 7 outcomes."""
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

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float().cpu().numpy()
        all_preds.append(preds)
        all_labels.append(y.cpu().numpy())

    all_preds  = np.concatenate(all_preds,  axis=0)   # (N, 7)
    all_labels = np.concatenate(all_labels, axis=0)   # (N, 7)

    f1         = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
    acc        = float(accuracy_score(all_labels.flatten(), all_preds.flatten()))
    per_class  = f1_score(all_labels, all_preds, average=None, zero_division=0).tolist()

    return {
        "loss": total_loss / max(n_batches, 1),
        "f1": f1,
        "acc": acc,
        "per_class_f1": per_class,
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
    name: str = "ToxiGeneV3",
    epochs: int = EPOCHS,
    patience: int = EARLY_STOP_PAT,
) -> tuple[float, int]:
    """Train model with early stopping. Returns (best_val_f1, best_epoch)."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scaler = GradScaler(device="cuda")

    best_val_f1 = -1.0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, n_batches = 0.0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            x_aug, y_aug = augment_batch(x, y, training=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda"):
                logits = model(x_aug)
                loss = multilabel_bce_loss(logits, y_aug, pos_weight)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            n_batches += 1

        scheduler.step()
        val_metrics = evaluate(model, val_loader, device, pos_weight)
        val_f1, val_loss = val_metrics["f1"], val_metrics["loss"]

        if epoch % 10 == 0 or epoch == 1:
            log.info(f"[{name}] Epoch {epoch:3d}/{epochs} | "
                     f"train_loss={train_loss/max(n_batches,1):.4f} | "
                     f"val_loss={val_loss:.4f} | val_F1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1, best_epoch, patience_counter = val_f1, epoch, 0
            torch.save(model.state_dict(), str(ckpt_path))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log.info(f"[{name}] Early stopping at epoch {epoch} "
                         f"(best val F1={best_val_f1:.4f} at epoch {best_epoch})")
                break

    return best_val_f1, best_epoch


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    log = _setup_logging()
    t0 = time.time()

    log.info("=" * 60)
    log.info("ToxiGene v3 — compact pathway-grouped transformer")
    log.info("=" * 60)

    # ── Device ─────────────────────────────────────────────────────────────
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    if device.type == "cuda":
        log.info(f"  GPU: {torch.cuda.get_device_name(0)}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ── Load data ──────────────────────────────────────────────────────────
    expr_path = DATA_DIR / "expression_matrix_v2_expanded.npy"
    label_path_v2 = DATA_DIR / "labels_v2.npy"

    log.info(f"Loading expression matrix: {expr_path}")
    expression_matrix = np.load(str(expr_path))  # (1697, 61479) float32
    log.info(f"  Expression shape: {expression_matrix.shape}")

    # Labels: multi-hot (1697, 7) float32 — ToxiGene is multi-label
    outcome_path = DATA_DIR / "outcome_labels_v2_expanded.npy"
    log.info(f"Loading multi-label outcomes from {outcome_path}")
    labels = np.load(str(outcome_path)).astype(np.float32)   # (1697, 7)
    log.info(f"  Labels shape: {labels.shape}, avg labels/sample={labels.sum(axis=1).mean():.2f}")
    log.info(f"  Labels/class: {labels.sum(axis=0).astype(int).tolist()}")

    assert len(expression_matrix) == len(labels), (
        f"Expression/label count mismatch: {len(expression_matrix)} vs {len(labels)}"
    )
    assert expression_matrix.shape[0] == 1697, (
        f"Expected 1697 samples, got {expression_matrix.shape[0]}"
    )

    # ── Gene selection: top-5000 by variance ──────────────────────────────
    log.info(f"Selecting top {N_GENES_SELECT} genes by variance …")
    expression_matrix, top_gene_idx = select_top_variance_genes(
        expression_matrix, n_genes=N_GENES_SELECT
    )
    log.info(f"  Reduced expression shape: {expression_matrix.shape}")

    # ── Dataset split (MUST match v2 exactly) ─────────────────────────────
    # Use torch.utils.data.random_split with seed=42 and [1187, 254, 256]
    full_dataset = ToxicologyDataset(expression_matrix, labels)

    split_generator = torch.Generator().manual_seed(SEED)
    train_ds, val_ds, test_ds = random_split(
        full_dataset, [N_TRAIN, N_VAL, N_TEST], generator=split_generator
    )
    log.info(f"Split: {N_TRAIN} train / {N_VAL} val / {N_TEST} test")

    # ── Normalization (computed on training split only) ────────────────────
    # Extract training indices and compute statistics on raw (pre-tensor) data
    train_indices = list(train_ds.indices)
    X_train_raw = expression_matrix[train_indices]  # (1187, 5000)

    train_mean = np.mean(X_train_raw, axis=0)                      # (5000,)
    train_std  = np.std(X_train_raw, axis=0) + 1e-8                # (5000,)

    # Normalize the full dataset in-place using training statistics
    expression_normalized = (expression_matrix - train_mean) / train_std
    expression_normalized = expression_normalized.astype(np.float32)

    # Rebuild datasets on normalized data (split indices unchanged)
    full_dataset_norm = ToxicologyDataset(expression_normalized, labels)

    split_generator2 = torch.Generator().manual_seed(SEED)
    train_ds_norm, val_ds_norm, test_ds_norm = random_split(
        full_dataset_norm, [N_TRAIN, N_VAL, N_TEST], generator=split_generator2
    )

    log.info("Expression normalized per gene (z-score, train stats only)")

    # ── pos_weight for BCE from training labels ───────────────────────────
    train_labels_arr = labels[train_indices]   # (1187, 7) float32
    pos_weight = compute_pos_weight(train_labels_arr)
    log.info(f"BCE pos_weight: {pos_weight.numpy().round(2).tolist()}")

    # ── DataLoaders ───────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_ds_norm, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=(device.type == "cuda"),
        persistent_workers=True, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds_norm, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=(device.type == "cuda"),
        persistent_workers=True,
    )
    test_loader = DataLoader(
        test_ds_norm, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=(device.type == "cuda"),
        persistent_workers=True,
    )

    # ─────────────────────────────────────────────────────────────────────
    # Train ToxiGene v3
    # ─────────────────────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("Building ToxiGene v3 …")
    model = ToxiGeneV3(
        n_genes=N_GENES_SELECT,
        n_groups=N_GROUPS,
        embed_dim=EMBED_DIM,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        ff_dim=FF_DIM,
        dropout_tf=DROPOUT_TF,
        dropout_head=DROPOUT_HEAD,
        n_classes=7,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"ToxiGene v3: {n_params:,} parameters")

    log.info("Training ToxiGene v3 …")
    best_val_f1_v3, best_epoch_v3 = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        log=log,
        pos_weight=pos_weight,
        ckpt_path=CKPT_PATH,
        name="ToxiGeneV3",
        epochs=EPOCHS,
        patience=EARLY_STOP_PAT,
    )
    log.info(f"ToxiGene v3 best val F1: {best_val_f1_v3:.4f} at epoch {best_epoch_v3}")

    # ── Evaluate ToxiGene v3 on test set ──────────────────────────────────
    log.info("Loading best ToxiGene v3 checkpoint …")
    model.load_state_dict(
        torch.load(str(CKPT_PATH), map_location=device, weights_only=True)
    )
    v3_test = evaluate(model, test_loader, device, pos_weight)

    log.info("\n" + "=" * 60)
    log.info("TEST RESULTS: ToxiGene v3")
    log.info("=" * 60)
    log.info(f"  Macro F1  : {v3_test['f1']:.4f}")
    log.info(f"  Accuracy  : {v3_test['acc']:.4f}")
    log.info("  Per-class F1:")
    for cname, cf1 in zip(OUTCOME_NAMES, v3_test["per_class_f1"]):
        log.info(f"    {cname:30s}: {cf1:.4f}")

    # ─────────────────────────────────────────────────────────────────────
    # Train SimpleMLP baseline (on same 5000-gene normalized data)
    # ─────────────────────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("Training SimpleMLP baseline …")

    mlp = SimpleMLP(n_genes=N_GENES_SELECT, n_classes=7).to(device)
    n_params_mlp = sum(p.numel() for p in mlp.parameters())
    log.info(f"SimpleMLP: {n_params_mlp:,} parameters")

    mlp_ckpt = CKPT_DIR / "simpleMLP_v3_best.pt"
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

    # ─────────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────────
    v3_beats_mlp = v3_test["f1"] > mlp_test["f1"]
    elapsed = time.time() - t0

    log.info("\n" + "=" * 60)
    log.info("COMPARISON SUMMARY")
    log.info("=" * 60)
    log.info(f"  ToxiGene v3 test F1 : {v3_test['f1']:.4f}")
    log.info(f"  SimpleMLP test F1   : {mlp_test['f1']:.4f}")
    log.info(f"  ToxiGene v3 beats SimpleMLP: {v3_beats_mlp}")
    log.info(f"  Elapsed: {elapsed:.1f}s")

    # ─────────────────────────────────────────────────────────────────────
    # Save results JSON
    # ─────────────────────────────────────────────────────────────────────
    per_class_dict = {
        name: round(float(f1), 6)
        for name, f1 in zip(OUTCOME_NAMES, v3_test["per_class_f1"])
    }
    results = {
        "model": "ToxiGene_v3",
        "test_f1_macro": round(float(v3_test["f1"]), 6),
        "test_acc": round(float(v3_test["acc"]), 6),
        "per_class_f1": per_class_dict,
        "n_train": N_TRAIN,
        "n_val": N_VAL,
        "n_test": N_TEST,
        "n_genes_input": N_GENES_SELECT,
        "n_params": n_params,
        "best_val_f1": round(float(best_val_f1_v3), 6),
        "best_epoch": best_epoch_v3,
        "baseline_simpleMLP_f1": round(float(mlp_test["f1"]), 6),
        "baseline_simpleMLP_n_params": n_params_mlp,
        "toxigene_v3_beats_simpleMLP": v3_beats_mlp,
        "f1_delta_vs_simpleMLP": round(
            float(v3_test["f1"]) - float(mlp_test["f1"]), 6
        ),
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
            "label_smoothing": LABEL_SMOOTHING,
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

    # ── Final print (required) ─────────────────────────────────────────────
    print(f"ToxiGene v3 TEST F1: {v3_test['f1']:.4f}")


if __name__ == "__main__":
    main()
