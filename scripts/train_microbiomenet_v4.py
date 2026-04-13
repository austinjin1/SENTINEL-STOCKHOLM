#!/usr/bin/env python3
"""MicroBiomeNet v4 — Restored sparse OTU attention with focal loss.

Root cause of v3 regression (0.7742 vs v2 0.8989):
  Conv1d tokenization (k=50,s=50 → 100 tokens) destroyed per-OTU identity.
  saline_sediment: 0.91→0.59, freshwater_impacted: 0.72→0.58.

Fix strategy:
  1. Restore v2's SparseOTUAttentionGate + PhylogeneticOTUEmbedding + GlobalAttentionPooling
  2. Add focal loss (gamma=2) to further target hard classes (freshwater_impacted)
  3. Fix data loading: io.BytesIO to avoid SIGBUS on RHEL9 NFS mmap
  4. Longer training: 150 epochs, patience=30 (after ep 40)
  5. Cosine annealing (no warm restarts) for stable convergence

Architecture (identical to v2):
  CLR(raw_abundances)
    → SparseOTUAttentionGate(5000→top_k=256)
    → PhylogeneticOTUEmbedding(5000, embed_dim=256)
    → Transformer[top-256 tokens](6L, 8H, ff=1024, drop=0.15)
    → GlobalAttentionPooling
    → Linear(256→512→256→8)
  11.7M parameters

MIT License — Bryan Cheng, 2026
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, random_split
from sklearn.metrics import f1_score, accuracy_score

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR  = PROJECT_ROOT / "data" / "processed" / "microbial" / "emp_16s"
CKPT_DIR  = PROJECT_ROOT / "checkpoints" / "microbial"
LOG_DIR   = PROJECT_ROOT / "logs"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

CKPT_PATH    = CKPT_DIR / "microbiomenet_v4_best.pt"
RESULTS_PATH = CKPT_DIR / "results_v4.json"
LOG_PATH     = LOG_DIR  / "train_microbiomenet_v4.log"

# ── Hyperparameters (identical to v2 except epochs/patience/scheduler) ────────
INPUT_DIM   = 5000
EMBED_DIM   = 256
NUM_HEADS   = 8
NUM_LAYERS  = 6
FF_DIM      = 1024
DROPOUT     = 0.15
TOP_K       = 256
NUM_SOURCES = 8
BATCH_SIZE  = 64
EPOCHS      = 150
LR          = 3e-4
WEIGHT_DECAY = 0.02
SEED        = 42

# Focal loss gamma (0 = standard CE, 2 = standard focal)
FOCAL_GAMMA = 2.0
LABEL_SMOOTHING = 0.05

SOURCE_NAMES = [
    "freshwater_natural",   # 0
    "freshwater_impacted",  # 1
    "saline_water",         # 2
    "freshwater_sediment",  # 3
    "saline_sediment",      # 4
    "soil_runoff",          # 5
    "animal_fecal",         # 6
    "plant_associated",     # 7
]


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
def _setup_logging() -> logging.Logger:
    try:
        from sentinel.utils.logging import get_logger
        log = get_logger("microbiomenet_v4")
    except Exception:
        log = logging.getLogger("microbiomenet_v4")
        if not log.handlers:
            log.setLevel(logging.INFO)
            ch = logging.StreamHandler(sys.stdout)
            ch.setFormatter(logging.Formatter(
                "%(asctime)s [INFO] %(message)s", datefmt="%H:%M:%S"))
            log.addHandler(ch)
            log.propagate = False
    fh = logging.FileHandler(str(LOG_PATH), mode="w")
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    fh.setLevel(logging.INFO)
    log.addHandler(fh)
    return log


# ─────────────────────────────────────────────────────────────────────────────
# CLR transform
# ─────────────────────────────────────────────────────────────────────────────
def clr_transform_np(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.maximum(x, eps)
    log_x = np.log(x)
    return (log_x - log_x.mean()).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Focal loss
# ─────────────────────────────────────────────────────────────────────────────
class FocalCrossEntropyLoss(nn.Module):
    """Focal loss variant of cross-entropy (handles class imbalance)."""

    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.05,
                 weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Standard label-smoothed CE first
        ce = F.cross_entropy(
            logits, targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        if self.gamma == 0.0:
            return ce.mean()
        # Focal weighting: (1 - p_t)^gamma
        probs = F.softmax(logits.detach(), dim=-1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_w = (1.0 - pt) ** self.gamma
        return (focal_w * ce).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Model components (identical to v2)
# ─────────────────────────────────────────────────────────────────────────────
class SparseOTUAttentionGate(nn.Module):
    """Sparse attention gate: identify top-k discriminative OTUs per sample."""

    def __init__(self, input_dim: int = INPUT_DIM, k: int = TOP_K) -> None:
        super().__init__()
        self.k = k
        self.scorer = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Linear(512, input_dim),
        )

    def forward(self, clr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.scorer(clr)
        importance = torch.sigmoid(scores)

        if self.training:
            topk_vals, _ = torch.topk(importance, self.k, dim=-1)
            threshold = topk_vals[:, -1:].detach()
            soft_mask = torch.sigmoid(10.0 * (importance - threshold))
            gated = clr * soft_mask
        else:
            topk_vals, topk_idx = torch.topk(importance, self.k, dim=-1)
            mask = torch.zeros_like(importance)
            mask.scatter_(-1, topk_idx, 1.0)
            gated = clr * mask

        return gated, importance


class PhylogeneticOTUEmbedding(nn.Module):
    """Per-OTU token embedding initialized from hierarchical positional encoding."""

    def __init__(self, n_otus: int = INPUT_DIM, embed_dim: int = EMBED_DIM) -> None:
        super().__init__()
        self.n_otus = n_otus
        self.embed_dim = embed_dim
        self.otu_embedding = nn.Embedding(n_otus, embed_dim)
        self._phylo_init()
        self.value_proj = nn.Linear(1, embed_dim)

    def _phylo_init(self) -> None:
        n, d = self.n_otus, self.embed_dim
        pe = torch.zeros(n, d)
        pos = torch.arange(n, dtype=torch.float).unsqueeze(1)
        div1 = torch.exp(torch.arange(0, 64, 2).float() * -(np.log(10000) / 64))
        pe[:, 0:64:2]  = torch.sin(pos / n * np.pi * div1)
        pe[:, 1:64:2]  = torch.cos(pos / n * np.pi * div1)
        div2 = torch.exp(torch.arange(0, 64, 2).float() * -(np.log(1000) / 64))
        pe[:, 64:128:2] = torch.sin(pos / n * np.pi * 10 * div2)
        pe[:, 65:129:2] = torch.cos(pos / n * np.pi * 10 * div2)
        div3 = torch.exp(torch.arange(0, 128, 2).float() * -(np.log(100) / 128))
        pe[:, 128:256:2] = torch.sin(pos / n * np.pi * 100 * div3)
        pe[:, 129:257:2] = torch.cos(pos / n * np.pi * 100 * div3)
        with torch.no_grad():
            self.otu_embedding.weight.copy_(pe)

    def forward(self, clr: torch.Tensor) -> torch.Tensor:
        """clr: [B, D] → tokens: [B, D, embed_dim]"""
        pos_ids = torch.arange(clr.shape[1], device=clr.device)
        pos_emb = self.otu_embedding(pos_ids)               # [D, embed_dim]
        val_emb = self.value_proj(clr.unsqueeze(-1))        # [B, D, embed_dim]
        return pos_emb.unsqueeze(0) + val_emb               # [B, D, embed_dim]


class GlobalAttentionPooling(nn.Module):
    """Learnable query-based attention pooling over OTU tokens."""

    def __init__(self, embed_dim: int = EMBED_DIM) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.attn  = nn.MultiheadAttention(embed_dim, num_heads=4,
                                            batch_first=True, dropout=0.0)
        self.norm  = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B = x.shape[0]
        q = self.query.expand(B, -1, -1)
        out, attn_w = self.attn(q, x, x, need_weights=True, average_attn_weights=True)
        pooled = self.norm(out.squeeze(1))
        return pooled, attn_w.squeeze(1)


class MicroBiomeNetV4(nn.Module):
    """MicroBiomeNet v4 — identical to v2 architecture."""

    def __init__(
        self,
        input_dim:   int   = INPUT_DIM,
        embed_dim:   int   = EMBED_DIM,
        num_heads:   int   = NUM_HEADS,
        num_layers:  int   = NUM_LAYERS,
        ff_dim:      int   = FF_DIM,
        dropout:     float = DROPOUT,
        top_k:       int   = TOP_K,
        num_classes: int   = NUM_SOURCES,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.sparse_gate = SparseOTUAttentionGate(input_dim, k=top_k)
        self.otu_embed   = PhylogeneticOTUEmbedding(input_dim, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.pool = GlobalAttentionPooling(embed_dim)

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear) and m is not self.otu_embed.otu_embedding:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, clr: torch.Tensor) -> dict[str, torch.Tensor]:
        # Sparse gate
        gated_clr, otu_importance = self.sparse_gate(clr)  # [B, D]

        # OTU embeddings
        tokens = self.otu_embed(gated_clr)                  # [B, D, E]

        # Select top-k token positions for transformer efficiency
        B, D, E = tokens.shape
        k = self.sparse_gate.k
        with torch.no_grad():
            _, topk_idx = torch.topk(otu_importance.detach(), k, dim=-1)  # [B, k]
        topk_idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, E)
        tokens_topk  = torch.gather(tokens, 1, topk_idx_exp)              # [B, k, E]

        # Transformer
        h = self.transformer(tokens_topk)                   # [B, k, E]

        # Global attention pooling
        pooled, _ = self.pool(h)                            # [B, E]

        logits = self.classifier(pooled)
        return {"logits": logits, "probs": F.softmax(logits, dim=-1)}


# ─────────────────────────────────────────────────────────────────────────────
# Dataset (io.BytesIO to avoid SIGBUS on RHEL9 NFS mmap)
# ─────────────────────────────────────────────────────────────────────────────
class EMP16SDataset(Dataset):
    def __init__(
        self,
        files: list[Path],
        augment: bool = False,
        dirichlet_alpha: float = 50.0,
        subsample_rate: float = 0.15,
    ) -> None:
        self.augment = augment
        self.dirichlet_alpha = dirichlet_alpha
        self.subsample_rate = subsample_rate
        self._labels: list[int] = []
        valid: list[Path] = []

        for f in files:
            try:
                with open(f, "rb") as fh:
                    raw = fh.read()
                d = np.load(io.BytesIO(raw), allow_pickle=True)
                abund = d["abundances"].astype(np.float32)
                if abund.sum() < 1e-8:
                    continue
                self._labels.append(int(d["source_label"]))
                valid.append(f)
            except Exception:
                continue
        self.files = valid

    def __len__(self) -> int:
        return len(self.files)

    @property
    def labels(self) -> list[int]:
        return self._labels

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        with open(self.files[idx], "rb") as fh:
            raw = fh.read()
        d = np.load(io.BytesIO(raw), allow_pickle=True)
        abund = d["abundances"].astype(np.float32)
        label = int(d["source_label"])

        total = abund.sum()
        if total > 0:
            abund = abund / total

        if self.augment:
            abund = self._augment(abund)

        clr = clr_transform_np(abund)
        return torch.tensor(clr, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    def _augment(self, x: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng()
        # Dirichlet noise (simplex-preserving)
        if rng.random() < 0.5:
            alpha = x * self.dirichlet_alpha + 1e-3
            x = rng.dirichlet(alpha).astype(np.float32)
        # OTU subsampling
        if rng.random() < self.subsample_rate:
            nonzero = np.where(x > 0)[0]
            if len(nonzero) > 20:
                n_drop = rng.integers(1, max(2, int(len(nonzero) * 0.1)))
                drop_idx = rng.choice(nonzero, size=n_drop, replace=False)
                x[drop_idx] = 0.0
                s = x.sum()
                if s > 1e-8:
                    x = x / s
        # Aitchison perturbation
        if rng.random() < 0.3:
            noise = rng.lognormal(0, 0.05, size=x.shape)
            x = x * noise
            s = x.sum()
            if s > 1e-8:
                x = x / s
        return x.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────────────────────────────────────
def make_sampler(labels: list[int]) -> WeightedRandomSampler:
    counts = Counter(labels)
    total  = len(labels)
    n_cls  = len(counts)
    weights = [total / (n_cls * counts[l]) for l in labels]
    return WeightedRandomSampler(weights, len(weights), replacement=True)


def mixup(clr: torch.Tensor, labels: torch.Tensor, alpha: float = 0.2):
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    idx = torch.randperm(clr.size(0), device=clr.device)
    return lam * clr + (1 - lam) * clr[idx], labels, labels[idx], lam


def train_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss, nb = 0.0, 0
    all_preds, all_labels = [], []

    for clr, labels in loader:
        clr, labels = clr.to(device), labels.to(device)

        if np.random.random() < 0.5:
            clr_m, la, lb, lam = mixup(clr, labels, alpha=0.2)
            with torch.amp.autocast("cuda"):
                out  = model(clr_m)
                loss = lam * criterion(out["logits"], la) + (1 - lam) * criterion(out["logits"], lb)
            preds = out["logits"].argmax(1).cpu()
        else:
            with torch.amp.autocast("cuda"):
                out  = model(clr)
                loss = criterion(out["logits"], labels)
            preds = out["logits"].argmax(1).cpu()

        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad()
            continue

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        nb += 1
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().tolist())

    return total_loss / max(nb, 1), f1_score(all_labels, all_preds, average="macro", zero_division=0)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, nb = 0.0, 0
    all_preds, all_labels = [], []

    for clr, labels in loader:
        clr, labels = clr.to(device), labels.to(device)
        out  = model(clr)
        loss = criterion(out["logits"], labels)
        preds = out["logits"].argmax(1).cpu()
        total_loss += loss.item()
        nb += 1
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().tolist())

    return (
        total_loss / max(nb, 1),
        f1_score(all_labels, all_preds, average="macro", zero_division=0),
        all_preds,
        all_labels,
    )


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
    if device.type == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Load data ────────────────────────────────────────────────────────────
    all_files = sorted(DATA_DIR.glob("*.npz"))
    log.info(f"Total EMP 16S files: {len(all_files)}")

    # Scan labels with BytesIO
    _labels: list[int] = []
    _valid:  list[Path] = []
    for f in all_files:
        try:
            with open(f, "rb") as fh:
                raw = fh.read()
            d = np.load(io.BytesIO(raw), allow_pickle=True)
            abund = d["abundances"].astype(np.float32)
            if abund.sum() < 1e-8:
                continue
            _labels.append(int(d["source_label"]))
            _valid.append(f)
        except Exception:
            continue

    n = len(_valid)
    log.info(f"Valid samples: {n}")
    cnt = Counter(_labels)
    for lid in sorted(cnt):
        name = SOURCE_NAMES[lid] if lid < len(SOURCE_NAMES) else f"class_{lid}"
        log.info(f"  {name:>25}: {cnt[lid]:,}")

    # ── 70/15/15 split (same split as v2/v3 for fair comparison) ─────────────
    n_tr = int(0.70 * n)
    n_va = int(0.15 * n)
    n_te = n - n_tr - n_va

    gen = torch.Generator().manual_seed(SEED)
    tr_idx, va_idx, te_idx = random_split(range(n), [n_tr, n_va, n_te], generator=gen)
    tr_idx = list(tr_idx)
    va_idx = list(va_idx)
    te_idx = list(te_idx)

    log.info(f"Split: {n_tr} train / {n_va} val / {n_te} test")

    tr_files = [_valid[i] for i in tr_idx]
    va_files = [_valid[i] for i in va_idx]
    te_files = [_valid[i] for i in te_idx]

    tr_ds = EMP16SDataset(tr_files, augment=True)
    va_ds = EMP16SDataset(va_files, augment=False)
    te_ds = EMP16SDataset(te_files, augment=False)

    log.info(f"Dataset loaded: {len(tr_ds)} / {len(va_ds)} / {len(te_ds)}")

    sampler = make_sampler(tr_ds.labels)
    tr_dl = DataLoader(tr_ds, batch_size=BATCH_SIZE, sampler=sampler,
                       num_workers=4, pin_memory=True, drop_last=True)
    va_dl = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=2, pin_memory=True)
    te_dl = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=2, pin_memory=True)

    # ── Build model ───────────────────────────────────────────────────────────
    model = MicroBiomeNetV4(
        input_dim=INPUT_DIM, embed_dim=EMBED_DIM, num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS, ff_dim=FF_DIM, dropout=DROPOUT,
        top_k=TOP_K, num_classes=NUM_SOURCES,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"MicroBiomeNetV4: {n_params:,} parameters")

    # ── Loss: focal CE with class weights ─────────────────────────────────────
    # Compute inverse-frequency class weights for WeightedRandomSampler +
    # focal loss (double coverage of class imbalance)
    counts_arr = np.array([cnt.get(i, 1) for i in range(NUM_SOURCES)], dtype=np.float32)
    class_weights = torch.tensor(counts_arr.sum() / (NUM_SOURCES * counts_arr), dtype=torch.float32).to(device)
    criterion = FocalCrossEntropyLoss(
        gamma=FOCAL_GAMMA,
        label_smoothing=LABEL_SMOOTHING,
        weight=class_weights,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_f1 = 0.0
    no_improve  = 0
    patience    = 30

    log.info("=" * 70)
    log.info("TRAINING")
    log.info("=" * 70)

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_f1 = train_epoch(model, tr_dl, optimizer, criterion, scaler, device)
        va_loss, va_f1, _, _ = eval_epoch(model, va_dl, criterion, device)
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            log.info(
                f"Ep {epoch:3d}/{EPOCHS} | "
                f"tr_loss={tr_loss:.4f} tr_f1={tr_f1:.4f} | "
                f"va_loss={va_loss:.4f} va_f1={va_f1:.4f} | "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )

        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            no_improve  = 0
            torch.save(model.state_dict(), CKPT_PATH)
            log.info(f"  -> New best val F1: {va_f1:.4f}")
        else:
            no_improve += 1

        if no_improve >= patience and epoch > 40:
            log.info(f"Early stopping at epoch {epoch} (patience={patience})")
            break

    # ── Test evaluation ───────────────────────────────────────────────────────
    log.info("=" * 70)
    log.info("TEST EVALUATION")
    log.info("=" * 70)

    if CKPT_PATH.exists():
        model.load_state_dict(torch.load(str(CKPT_PATH), map_location=device, weights_only=True))
        log.info(f"Loaded checkpoint: {CKPT_PATH.name}")

    _, te_f1, te_preds, te_labels = eval_epoch(model, te_dl, criterion, device)
    te_acc = accuracy_score(te_labels, te_preds)
    per_class_f1 = f1_score(te_labels, te_preds, average=None, zero_division=0)
    unique_labels = sorted(set(te_labels))

    log.info(f"Macro F1  : {te_f1:.4f}")
    log.info(f"Accuracy  : {te_acc:.4f}")
    log.info(f"Best val  : {best_val_f1:.4f}")
    log.info("Per-class F1:")
    per_class_dict = {}
    for i, lid in enumerate(unique_labels):
        name = SOURCE_NAMES[lid] if lid < len(SOURCE_NAMES) else f"class_{lid}"
        cf1  = per_class_f1[i] if i < len(per_class_f1) else 0.0
        per_class_dict[name] = round(float(cf1), 6)
        log.info(f"  {name:>25}: {cf1:.4f}")

    elapsed = time.time() - t0
    log.info(f"Elapsed: {elapsed:.1f}s")

    results = {
        "model": "MicroBiomeNetV4",
        "test_macro_f1": round(float(te_f1), 6),
        "test_accuracy": round(float(te_acc), 6),
        "best_val_f1": round(float(best_val_f1), 6),
        "per_class_f1": per_class_dict,
        "n_train": len(tr_ds),
        "n_val": len(va_ds),
        "n_test": len(te_ds),
        "n_total": len(tr_ds) + len(va_ds) + len(te_ds),
        "n_classes": NUM_SOURCES,
        "architecture": {
            "input_dim": INPUT_DIM,
            "embed_dim": EMBED_DIM,
            "num_heads": NUM_HEADS,
            "num_layers": NUM_LAYERS,
            "ff_dim": FF_DIM,
            "dropout": DROPOUT,
            "top_k_sparse": TOP_K,
            "focal_gamma": FOCAL_GAMMA,
            "n_params": n_params,
        },
        "training": {
            "epochs": EPOCHS,
            "lr": LR,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
            "scheduler": "cosine_annealing",
        },
        "elapsed_seconds": round(elapsed, 2),
    }

    with open(str(RESULTS_PATH), "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results saved to {RESULTS_PATH}")
    print(f"MicroBiomeNet v4 TEST F1: {te_f1:.4f}")


if __name__ == "__main__":
    main()
