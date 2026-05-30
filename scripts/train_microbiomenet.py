#!/usr/bin/env python3
"""MicroBiomeNet v3 — Temporal-spatial holdout evaluation on real EMP 16S data.

Motivation:
  v5 (current best) uses random 70/15/15 splits, which allows site leakage:
  samples from the same site can appear in train AND test, inflating metrics.
  This script uses strict spatial holdout via sentinel.data.splits to give
  honest generalization estimates.

Split protocol:
  Uses split_indices_spatial_only() since EMP samples lack strong temporal
  metadata.  Sites are assigned to folds via deterministic SHA-256 hashing
  of site_id (assign_spatial_fold):
    Folds 0,1,2 -> train
    Fold 3      -> val
    Fold 4      -> test
  Zero site overlap between splits is verified before training.

Architecture (identical to v5/v2 — MicroBiomeNetV5):
  CLR -> SparseOTUAttentionGate(top_k=256) -> PhylogeneticOTUEmbedding
  -> Transformer[256 tokens, 6L, 8H, ff=1024] -> GlobalAttentionPooling
  -> Linear(256->512->256->8)   [11.7M params]

Training config (same as v5):
  - CrossEntropyLoss(label_smoothing=0.05)
  - AdamW(lr=3e-4, weight_decay=0.02)
  - CosineAnnealingWarmRestarts(T_0=25, T_mult=2, eta_min=1e-6)
  - WeightedRandomSampler for class imbalance
  - Mixed precision (AMP + GradScaler)
  - Gradient clipping at 1.0
  - 150 epochs, patience=25, early stopping after epoch 40
  - Augmentation: Mixup (50%), Dirichlet noise (50%), OTU subsampling (15%),
    lognormal noise (30%)

Data:
  Real EMP 16S rRNA per-sample .npz files ONLY.
  NO synthetic data.

Saves:
  checkpoints/microbial/microbiomenet_v3_best.pt
  results/benchmarks/microbiomenet_v3_holdout.json
Logs to: logs/train_microbiomenet_v3.log

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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import f1_score, accuracy_score

from sentinel.data.splits import (
    assign_spatial_fold,
    split_indices_spatial_only,
    SplitConfig,
)

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR     = PROJECT_ROOT / "data" / "processed" / "microbial" / "emp_16s"
CKPT_DIR     = PROJECT_ROOT / "checkpoints" / "microbial"
LOG_DIR      = PROJECT_ROOT / "logs"
RESULTS_DIR  = PROJECT_ROOT / "results" / "benchmarks"

CKPT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CKPT_PATH    = CKPT_DIR / "microbiomenet_v3_best.pt"
RESULTS_PATH = RESULTS_DIR / "microbiomenet_v3_holdout.json"
LOG_PATH     = LOG_DIR / "train_microbiomenet_v3.log"

# ── Hyperparameters (same as v5/v2) ──────────────────────────────────────────
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
        log = get_logger("microbiomenet_v3")
    except Exception:
        log = logging.getLogger("microbiomenet_v3")
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
# Model (identical to v5)
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
    """Architecture identical to v5/v2. Named V5 for checkpoint compatibility."""

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
# Dataset (io.BytesIO — SIGBUS safe on RHEL9 NFS)
# ─────────────────────────────────────────────────────────────────────────────
class EMP16SDataset(Dataset):
    """EMP 16S dataset that loads from pre-validated file lists.

    Unlike v5 which scans files during __init__, this version receives
    pre-validated (file, label) pairs from the split logic to avoid
    redundant file reads.
    """

    def __init__(
        self,
        files: list[Path],
        labels: list[int],
        augment: bool = False,
    ) -> None:
        assert len(files) == len(labels), (
            f"files ({len(files)}) and labels ({len(labels)}) length mismatch"
        )
        self.files = files
        self._labels = labels
        self.augment = augment

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
        label = self._labels[idx]
        total = abund.sum()
        if total > 0:
            abund = abund / total
        if self.augment:
            abund = self._augment(abund)
        clr = clr_transform_np(abund)
        return torch.tensor(clr, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    def _augment(self, x: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng()
        # Dirichlet noise (50%)
        if rng.random() < 0.5:
            alpha = x * 50.0 + 1e-3
            x = rng.dirichlet(alpha).astype(np.float32)
        # OTU subsampling (15%)
        if rng.random() < 0.15:
            nonzero = np.where(x > 0)[0]
            if len(nonzero) > 20:
                n_drop = rng.integers(1, max(2, int(len(nonzero) * 0.1)))
                drop_idx = rng.choice(nonzero, size=n_drop, replace=False)
                x[drop_idx] = 0.0
                s = x.sum()
                if s > 1e-8:
                    x = x / s
        # Lognormal noise (30%)
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
# Data loading with spatial holdout splits
# ─────────────────────────────────────────────────────────────────────────────
def load_and_split_data(log: logging.Logger) -> tuple[
    list[Path], list[int], list[str],
    dict[str, list[int]],
]:
    """Load all real EMP 16S .npz files and assign spatial holdout splits.

    Returns:
        valid_files: List of valid .npz file paths.
        valid_labels: Corresponding source labels.
        valid_site_ids: Corresponding site_id strings.
        split_idx: Dict mapping 'train'/'val'/'test' to index lists.
    """
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Data directory not found: {DATA_DIR}\n"
            f"Run 'python scripts/download_emp_microbiome.py' first to download "
            f"real EMP 16S data."
        )

    all_files = sorted(DATA_DIR.glob("*.npz"))
    if len(all_files) == 0:
        raise FileNotFoundError(
            f"No .npz files found in {DATA_DIR}\n"
            f"Run 'python scripts/download_emp_microbiome.py' first to download "
            f"real EMP 16S data."
        )

    log.info(f"Scanning {len(all_files)} .npz files in {DATA_DIR}")

    valid_files: list[Path] = []
    valid_labels: list[int] = []
    valid_site_ids: list[str] = []
    skipped = 0

    for f in all_files:
        try:
            with open(f, "rb") as fh:
                raw = fh.read()
            d = np.load(io.BytesIO(raw), allow_pickle=True)

            abund = d["abundances"].astype(np.float32)
            if abund.shape[0] != INPUT_DIM:
                log.warning(f"Skipping {f.name}: unexpected dim {abund.shape[0]}")
                skipped += 1
                continue
            if abund.sum() < 1e-8:
                skipped += 1
                continue

            label = int(d["source_label"])
            if label < 0 or label >= NUM_SOURCES:
                log.warning(f"Skipping {f.name}: invalid label {label}")
                skipped += 1
                continue

            # Extract site_id — stored as str or int in the npz
            site_id_raw = d["site_id"]
            if hasattr(site_id_raw, "item"):
                site_id = str(site_id_raw.item())
            else:
                site_id = str(site_id_raw)

            valid_files.append(f)
            valid_labels.append(label)
            valid_site_ids.append(site_id)
        except Exception as e:
            log.warning(f"Skipping {f.name}: {e}")
            skipped += 1
            continue

    n = len(valid_files)
    if n == 0:
        raise RuntimeError(
            f"No valid samples found in {DATA_DIR}. "
            f"All {len(all_files)} files were skipped."
        )

    log.info(f"Valid samples: {n} (skipped {skipped})")

    # ── Log overall class distribution ──────────────────────────────────────
    cnt = Counter(valid_labels)
    log.info("Overall class distribution:")
    for lid in sorted(cnt):
        name = SOURCE_NAMES[lid] if lid < len(SOURCE_NAMES) else f"class_{lid}"
        log.info(f"  {name:>25}: {cnt[lid]:>6,}  ({100*cnt[lid]/n:.1f}%)")

    # ── Spatial holdout split ───────────────────────────────────────────────
    log.info("Applying spatial holdout split (hash-based site assignment)...")
    log.info("  Folds 0,1,2 -> train | Fold 3 -> val | Fold 4 -> test")

    split_idx = split_indices_spatial_only(valid_site_ids)

    # ── Verify zero site leakage ────────────────────────────────────────────
    split_sites: dict[str, set[str]] = {}
    for split_name, indices in split_idx.items():
        sites = set(valid_site_ids[i] for i in indices)
        split_sites[split_name] = sites

    for a in ["train", "val", "test"]:
        for b in ["train", "val", "test"]:
            if a >= b:
                continue
            overlap = split_sites[a] & split_sites[b]
            if overlap:
                raise RuntimeError(
                    f"SITE LEAKAGE DETECTED between {a} and {b}! "
                    f"{len(overlap)} overlapping sites: {list(overlap)[:5]}..."
                )
    log.info("Site leakage check PASSED: zero overlap between splits")

    # ── Log per-split statistics ────────────────────────────────────────────
    for split_name in ["train", "val", "test"]:
        indices = split_idx[split_name]
        n_split = len(indices)
        n_sites = len(split_sites[split_name])
        split_labels = [valid_labels[i] for i in indices]
        split_cnt = Counter(split_labels)

        log.info(f"\n{'=' * 50}")
        log.info(f"{split_name.upper()} split: {n_split} samples, {n_sites} unique sites")
        log.info(f"{'=' * 50}")
        for lid in sorted(split_cnt):
            name = SOURCE_NAMES[lid] if lid < len(SOURCE_NAMES) else f"class_{lid}"
            pct = 100 * split_cnt[lid] / n_split if n_split > 0 else 0
            log.info(f"  {name:>25}: {split_cnt[lid]:>6,}  ({pct:.1f}%)")

        # Log fold distribution for this split
        fold_counts = Counter(
            assign_spatial_fold(valid_site_ids[i]) for i in indices
        )
        log.info(f"  Fold distribution: {dict(sorted(fold_counts.items()))}")

    return valid_files, valid_labels, valid_site_ids, split_idx


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    t0 = time.time()
    log = _setup_logging()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    log.info("=" * 70)
    log.info("MicroBiomeNet v3 — Spatial Holdout Training (REAL DATA ONLY)")
    log.info("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    if device.type == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")
        log.info(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")

    # ── Load data with spatial holdout splits ────────────────────────────────
    valid_files, valid_labels, valid_site_ids, split_idx = load_and_split_data(log)

    # Build datasets from split indices
    tr_files  = [valid_files[i]  for i in split_idx["train"]]
    tr_labels = [valid_labels[i] for i in split_idx["train"]]
    va_files  = [valid_files[i]  for i in split_idx["val"]]
    va_labels = [valid_labels[i] for i in split_idx["val"]]
    te_files  = [valid_files[i]  for i in split_idx["test"]]
    te_labels = [valid_labels[i] for i in split_idx["test"]]

    tr_ds = EMP16SDataset(tr_files, tr_labels, augment=True)
    va_ds = EMP16SDataset(va_files, va_labels, augment=False)
    te_ds = EMP16SDataset(te_files, te_labels, augment=False)

    log.info(f"\nDataset sizes: {len(tr_ds)} train / {len(va_ds)} val / {len(te_ds)} test")

    if len(tr_ds) == 0 or len(va_ds) == 0:
        raise RuntimeError(
            "Train or val split is empty after spatial holdout. "
            "Check that data contains diverse site_ids."
        )

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

    # ── Loss, optimizer, scheduler (same as v5) ───────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=25, T_mult=2, eta_min=1e-6,
    )
    scaler = torch.amp.GradScaler("cuda")

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_f1 = 0.0
    no_improve  = 0
    patience    = 25

    log.info("\n" + "=" * 70)
    log.info("TRAINING (spatial holdout — no site leakage)")
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
            log.info(f"  -> New best val F1: {va_f1:.4f}  (saved to {CKPT_PATH.name})")
        else:
            no_improve += 1

        if no_improve >= patience and epoch > 40:
            log.info(f"Early stopping at epoch {epoch} (patience={patience})")
            break

    # ── Test evaluation ───────────────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("TEST EVALUATION (spatial holdout — held-out sites)")
    log.info("=" * 70)

    if CKPT_PATH.exists():
        model.load_state_dict(torch.load(str(CKPT_PATH), map_location=device, weights_only=True))
        log.info(f"Loaded best checkpoint from {CKPT_PATH}")

    if len(te_ds) > 0:
        _, te_f1, te_preds, te_labels_eval = eval_epoch(model, te_dl, criterion, device)
        te_acc = accuracy_score(te_labels_eval, te_preds)
        per_class_f1 = f1_score(te_labels_eval, te_preds, average=None, zero_division=0)
        unique_labels = sorted(set(te_labels_eval))

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
    else:
        log.warning("Test split is empty — no test evaluation performed")
        te_f1 = 0.0
        te_acc = 0.0
        per_class_dict = {}

    elapsed = time.time() - t0
    log.info(f"\nElapsed: {elapsed:.1f}s")

    # ── Compute split metadata for results ────────────────────────────────────
    train_sites = set(valid_site_ids[i] for i in split_idx["train"])
    val_sites   = set(valid_site_ids[i] for i in split_idx["val"])
    test_sites  = set(valid_site_ids[i] for i in split_idx["test"])

    results = {
        "model": "MicroBiomeNetV5",
        "version": "v3_spatial_holdout",
        "split_protocol": "spatial_holdout_hash_based",
        "split_description": (
            "Strict spatial holdout using sentinel.data.splits. "
            "Sites assigned to folds via SHA-256 hash of site_id. "
            "Folds 0,1,2 -> train, Fold 3 -> val, Fold 4 -> test. "
            "Zero site leakage verified."
        ),
        "data_source": "Real EMP 16S rRNA only (no synthetic data)",
        "test_macro_f1": round(float(te_f1), 6),
        "test_accuracy": round(float(te_acc), 6),
        "best_val_f1": round(float(best_val_f1), 6),
        "per_class_f1": per_class_dict,
        "n_train": len(tr_ds),
        "n_val": len(va_ds),
        "n_test": len(te_ds),
        "n_total": len(tr_ds) + len(va_ds) + len(te_ds),
        "n_train_sites": len(train_sites),
        "n_val_sites": len(val_sites),
        "n_test_sites": len(test_sites),
        "site_leakage": {
            "train_val_overlap": 0,
            "train_test_overlap": 0,
            "val_test_overlap": 0,
        },
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
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
            "loss": "CrossEntropyLoss(label_smoothing=0.05)",
            "optimizer": "AdamW(lr=3e-4, weight_decay=0.02)",
            "scheduler": "CosineAnnealingWarmRestarts(T_0=25, T_mult=2, eta_min=1e-6)",
            "patience": patience,
            "gradient_clip": 1.0,
            "mixed_precision": True,
            "augmentation": "Mixup(50%), Dirichlet(50%), OTU_subsample(15%), lognormal(30%)",
        },
        "elapsed_seconds": round(elapsed, 2),
        "checkpoint_path": str(CKPT_PATH),
    }

    with open(str(RESULTS_PATH), "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results saved to {RESULTS_PATH}")

    print(f"\nMicroBiomeNet v3 (spatial holdout) TEST F1: {te_f1:.4f}")
    print(f"Results: {RESULTS_PATH}")
    print(f"Checkpoint: {CKPT_PATH}")


if __name__ == "__main__":
    main()
