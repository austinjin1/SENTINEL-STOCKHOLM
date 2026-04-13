#!/usr/bin/env python3
"""MicroBiomeNet v5 — v2 architecture, exact training config, extended budget.

Root cause analysis:
  v2 (0.8989): CrossEntropyLoss(LS=0.05) + WeightedRandomSampler + WarmRestarts
  v4 (0.8515): FocalCE(gamma=2) + class_weights + WeightedRandomSampler + CosineAnnealing
  The over-regularization (triple class-imbalance handling) degraded performance.

Fix:
  Restore v2's exact training config:
    - CrossEntropyLoss(label_smoothing=0.05), NO focal, NO extra class weights
    - CosineAnnealingWarmRestarts(T_0=25, T_mult=2) like v2
    - WeightedRandomSampler only (single imbalance correction)
  Improvements over v2:
    - io.BytesIO data loading (SIGBUS safe on RHEL9 NFS)
    - Extended budget: 150 epochs (v2 had 100), patience=25 (v2 had 20)

Architecture (identical to v2):
  CLR → SparseOTUAttentionGate(top_k=256) → PhylogeneticOTUEmbedding
  → Transformer[256 tokens, 6L, 8H, ff=1024] → GlobalAttentionPooling
  → Linear(256→512→256→8)   [11.7M params]

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

CKPT_PATH    = CKPT_DIR / "microbiomenet_v5_best.pt"
RESULTS_PATH = CKPT_DIR / "results_v5.json"
LOG_PATH     = LOG_DIR  / "train_microbiomenet_v5.log"

# ── Hyperparameters (exact v2 except epochs=150, patience=25) ─────────────────
INPUT_DIM    = 5000
EMBED_DIM    = 256
NUM_HEADS    = 8
NUM_LAYERS   = 6
FF_DIM       = 1024
DROPOUT      = 0.15
TOP_K        = 256
NUM_SOURCES  = 8
BATCH_SIZE   = 64
EPOCHS       = 150
LR           = 3e-4
WEIGHT_DECAY = 0.02
SEED         = 42

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
        log = get_logger("microbiomenet_v5")
    except Exception:
        log = logging.getLogger("microbiomenet_v5")
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
# Model (identical to v2 / v4)
# ─────────────────────────────────────────────────────────────────────────────
class SparseOTUAttentionGate(nn.Module):
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
        pe[:, 0:64:2]   = torch.sin(pos / n * np.pi * div1)
        pe[:, 1:64:2]   = torch.cos(pos / n * np.pi * div1)
        div2 = torch.exp(torch.arange(0, 64, 2).float() * -(np.log(1000) / 64))
        pe[:, 64:128:2]  = torch.sin(pos / n * np.pi * 10 * div2)
        pe[:, 65:129:2]  = torch.cos(pos / n * np.pi * 10 * div2)
        div3 = torch.exp(torch.arange(0, 128, 2).float() * -(np.log(100) / 128))
        pe[:, 128:256:2] = torch.sin(pos / n * np.pi * 100 * div3)
        pe[:, 129:257:2] = torch.cos(pos / n * np.pi * 100 * div3)
        with torch.no_grad():
            self.otu_embedding.weight.copy_(pe)

    def forward(self, clr: torch.Tensor) -> torch.Tensor:
        pos_ids = torch.arange(clr.shape[1], device=clr.device)
        pos_emb = self.otu_embedding(pos_ids)
        val_emb = self.value_proj(clr.unsqueeze(-1))
        return pos_emb.unsqueeze(0) + val_emb


class GlobalAttentionPooling(nn.Module):
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


class MicroBiomeNetV5(nn.Module):
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
        gated_clr, otu_importance = self.sparse_gate(clr)
        tokens = self.otu_embed(gated_clr)
        B, D, E = tokens.shape
        k = self.sparse_gate.k
        with torch.no_grad():
            _, topk_idx = torch.topk(otu_importance.detach(), k, dim=-1)
        topk_idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, E)
        tokens_topk  = torch.gather(tokens, 1, topk_idx_exp)
        h = self.transformer(tokens_topk)
        pooled, _ = self.pool(h)
        logits = self.classifier(pooled)
        return {"logits": logits, "probs": F.softmax(logits, dim=-1)}


# ─────────────────────────────────────────────────────────────────────────────
# Dataset (io.BytesIO — SIGBUS safe)
# ─────────────────────────────────────────────────────────────────────────────
class EMP16SDataset(Dataset):
    def __init__(self, files: list[Path], augment: bool = False) -> None:
        self.augment = augment
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
        if rng.random() < 0.5:
            alpha = x * 50.0 + 1e-3
            x = rng.dirichlet(alpha).astype(np.float32)
        if rng.random() < 0.15:
            nonzero = np.where(x > 0)[0]
            if len(nonzero) > 20:
                n_drop = rng.integers(1, max(2, int(len(nonzero) * 0.1)))
                drop_idx = rng.choice(nonzero, size=n_drop, replace=False)
                x[drop_idx] = 0.0
                s = x.sum()
                if s > 1e-8:
                    x = x / s
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

    # ── Load data ────────────────────────────────────────────────────────────
    all_files = sorted(DATA_DIR.glob("*.npz"))
    log.info(f"Total EMP 16S files: {len(all_files)}")

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

    # ── 70/15/15 split (identical to v2 — same torch.Generator seed) ─────────
    n_tr = int(0.70 * n)
    n_va = int(0.15 * n)
    n_te = n - n_tr - n_va

    gen = torch.Generator().manual_seed(SEED)
    tr_idx, va_idx, te_idx = random_split(range(n), [n_tr, n_va, n_te], generator=gen)
    tr_idx = list(tr_idx)
    va_idx = list(va_idx)
    te_idx = list(te_idx)

    log.info(f"Split: {n_tr} train / {n_va} val / {n_te} test")

    tr_ds = EMP16SDataset([_valid[i] for i in tr_idx], augment=True)
    va_ds = EMP16SDataset([_valid[i] for i in va_idx], augment=False)
    te_ds = EMP16SDataset([_valid[i] for i in te_idx], augment=False)

    log.info(f"Dataset loaded: {len(tr_ds)} / {len(va_ds)} / {len(te_ds)}")

    sampler = make_sampler(tr_ds.labels)
    tr_dl = DataLoader(tr_ds, batch_size=BATCH_SIZE, sampler=sampler,
                       num_workers=4, pin_memory=True, drop_last=True)
    va_dl = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=2, pin_memory=True)
    te_dl = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=2, pin_memory=True)

    # ── Build model ───────────────────────────────────────────────────────────
    model = MicroBiomeNetV5(
        input_dim=INPUT_DIM, embed_dim=EMBED_DIM, num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS, ff_dim=FF_DIM, dropout=DROPOUT,
        top_k=TOP_K, num_classes=NUM_SOURCES,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"MicroBiomeNetV5: {n_params:,} parameters")

    # ── Exact v2 loss + scheduler ─────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)           # same as v2
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=25, T_mult=2, eta_min=1e-6,                  # same as v2
    )
    scaler = torch.amp.GradScaler("cuda")

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_f1 = 0.0
    no_improve  = 0
    patience    = 25   # v2 had 20; +5 for extended budget

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
        cf1 = per_class_f1[i] if i < len(per_class_f1) else 0.0
        per_class_dict[name] = round(float(cf1), 6)
        log.info(f"  {name:>25}: {cf1:.4f}")

    elapsed = time.time() - t0
    log.info(f"Elapsed: {elapsed:.1f}s")

    results = {
        "model": "MicroBiomeNetV5",
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
            "n_params": n_params,
        },
        "training": {
            "epochs": EPOCHS,
            "lr": LR,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
            "loss": "CrossEntropyLoss(label_smoothing=0.05)",
            "scheduler": "CosineAnnealingWarmRestarts(T_0=25, T_mult=2)",
            "patience": patience,
        },
        "elapsed_seconds": round(elapsed, 2),
    }

    with open(str(RESULTS_PATH), "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results saved to {RESULTS_PATH}")
    print(f"MicroBiomeNet v5 TEST F1: {te_f1:.4f}")


if __name__ == "__main__":
    main()
