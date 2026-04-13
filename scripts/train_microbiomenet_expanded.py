#!/usr/bin/env python3
"""MicroBiomeNet expanded training on consolidated real data.

Trains on consolidated_real/ dataset: EMP 16S + NARS + real NRSA WQ
Total: ~82,317 samples (~4x expansion from 20,244 baseline).

Architecture: same as train_microbiomenet_emp.py (MicrobialEncoder, input_dim=5000, 8 classes)
Training: 100 epochs, early stopping patience=10, class-balanced sampling, mixup.

MIT License — Bryan Cheng, 2026
"""

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import f1_score, accuracy_score, classification_report

sys.path.insert(0, "/home/bcheng/SENTINEL")

# Patch DNABERT-S before importing the model to avoid bus error on large model load.
# The model uses learned fallback embeddings which work identically for training.
import sentinel.models.microbial_encoder.sequence_encoder as _se
def _safe_init_backbone(self, model_id):
    import logging
    logging.getLogger(__name__).warning(
        "DNABERT-S skipped (bus error protection). Using learned fallback embeddings."
    )
    self.using_dnabert = False
    self.tokenizer = None
    self.backbone = None
    self.fallback_embeddings = nn.Embedding(self.max_otus, self.output_dim)
    nn.init.xavier_uniform_(self.fallback_embeddings.weight)
_se.DNABERTSequenceEncoder._init_backbone = _safe_init_backbone

from sentinel.models.microbial_encoder.model import MicrobialEncoder
from sentinel.models.microbial_encoder.aitchison_attention import clr_transform
from sentinel.utils.logging import get_logger

logger = get_logger(__name__)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CKPT = Path("checkpoints/microbial")
CKPT.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path("data/processed/microbial/consolidated_real")

CLASS_NAMES = [
    "freshwater_natural",
    "freshwater_impacted",
    "saline_water",
    "freshwater_sediment",
    "saline_sediment",
    "soil_runoff",
    "animal_fecal",
    "plant_associated",
]
NUM_CLASSES = 8


class ConsolidatedDataset(Dataset):
    """Dataset for consolidated multi-source real data.

    Loads the single consolidated.npz and provides CLR-transformed features.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.augment = augment

    def __len__(self):
        return len(self.X)

    @property
    def labels(self):
        return self.y.tolist()

    def __getitem__(self, idx):
        abundances = self.X[idx].copy()
        label = int(self.y[idx])

        # Ensure sums to 1 (relative abundance)
        total = abundances.sum()
        if total > 0:
            abundances = abundances / total

        # Compositional augmentation
        if self.augment:
            abundances = _augment_composition(abundances)

        abundances_t = torch.tensor(abundances, dtype=torch.float32)

        # CLR transform with pseudocount
        pseudo = 1e-6
        x = abundances_t + pseudo
        x = x / x.sum()
        clr = clr_transform(x.unsqueeze(0)).squeeze(0)
        clr = clr.clamp(-8, 8).float()

        return {
            "abundances": abundances_t,
            "clr": clr,
            "source_label": label,
        }


def _augment_composition(x: np.ndarray) -> np.ndarray:
    """Aitchison multiplicative perturbation + random zero-out."""
    rng = np.random.default_rng()

    if rng.random() < 0.5:
        noise = rng.lognormal(0, 0.1, size=x.shape)
        x = x * noise
        s = x.sum()
        if s > 0:
            x = x / s

    if rng.random() < 0.3:
        nonzero = np.where(x > 0)[0]
        if len(nonzero) > 10:
            n_drop = rng.integers(1, max(2, len(nonzero) // 10))
            drop_idx = rng.choice(nonzero, size=n_drop, replace=False)
            x[drop_idx] = 0
            s = x.sum()
            if s > 0:
                x = x / s

    return x


def make_balanced_sampler(labels):
    """Inverse-frequency weighted sampler for class balance."""
    label_list = list(labels)
    class_counts = Counter(label_list)
    n_classes = len(class_counts)
    total = len(label_list)
    weights = [total / (n_classes * class_counts[l]) for l in label_list]
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def mixup_criterion(crit, pred, ya, yb, lam):
    return lam * crit(pred, ya) + (1 - lam) * crit(pred, yb)


def train(model, tr_dl, va_dl, epochs=100, lr=5e-4, patience=10):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.02)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_f1, no_improve = 0.0, 0

    logger.info("=" * 60)
    logger.info("EXPANDED MICROBIOMENET TRAINING")
    logger.info(f"Device: {DEVICE}, Epochs: {epochs}, Patience: {patience}")
    logger.info("=" * 60)

    for epoch in range(epochs):
        model.train()
        total_loss, nb = 0.0, 0
        all_preds, all_labels = [], []

        for batch in tr_dl:
            clr = batch["clr"].to(DEVICE)
            labels = batch["source_label"].to(DEVICE)

            if np.random.random() < 0.5:
                clr_m, ya, yb, lam = mixup_data(clr, labels)
                outputs = model(x=clr_m)
                loss = mixup_criterion(criterion, outputs["source_logits"], ya, yb, lam)
                preds = outputs["source_logits"].argmax(-1).cpu()
            else:
                outputs = model(x=clr)
                loss_dict = model.compute_loss(x=clr, outputs=outputs, source_targets=labels)
                loss = loss_dict["total"]
                preds = outputs["source_logits"].argmax(-1).cpu()

            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            nb += 1
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().tolist())

        scheduler.step()
        train_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        # Validation
        model.eval()
        va_preds, va_labels = [], []
        with torch.no_grad():
            for batch in va_dl:
                clr = batch["clr"].to(DEVICE)
                labels = batch["source_label"]
                outputs = model(x=clr)
                preds = outputs["source_logits"].argmax(-1).cpu()
                va_preds.extend(preds.tolist())
                va_labels.extend(labels.tolist())

        val_f1 = f1_score(va_labels, va_preds, average="macro", zero_division=0)
        val_acc = accuracy_score(va_labels, va_preds)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                f"Ep {epoch+1:3d}/{epochs} | Loss: {total_loss/max(nb,1):.4f} | "
                f"Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f} | "
                f"Val Acc: {val_acc:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}"
            )

        if val_f1 > best_f1:
            best_f1 = val_f1
            no_improve = 0
            torch.save(model.state_dict(), CKPT / "microbiomenet_expanded_best.pt")
            logger.info(f"  -> New best Val F1: {val_f1:.4f} (saved)")
        else:
            no_improve += 1

        if no_improve >= patience and epoch > 20:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    return best_f1


def main():
    t0 = time.time()
    logger.info("Loading consolidated dataset...")

    data = np.load(DATA_DIR / "consolidated.npz", allow_pickle=True)
    X = data["X"]
    y = data["y"]
    sources = data["sources"]

    n = len(X)
    logger.info(f"Total samples: {n:,}, features: {X.shape[1]}")
    lc = Counter(y.tolist())
    for k in sorted(lc):
        name = CLASS_NAMES[k] if k < len(CLASS_NAMES) else f"class_{k}"
        logger.info(f"  {name:>25}: {lc[k]:,}")

    sc = Counter(sources.tolist())
    for src, cnt in sorted(sc.items()):
        logger.info(f"  source {src}: {cnt:,}")

    # Split 70/15/15
    rng = np.random.default_rng(42)
    idx = rng.permutation(n)
    n_tr = int(0.70 * n)
    n_va = int(0.15 * n)
    tr_idx = idx[:n_tr]
    va_idx = idx[n_tr:n_tr+n_va]
    te_idx = idx[n_tr+n_va:]

    tr_ds = ConsolidatedDataset(X[tr_idx], y[tr_idx], augment=True)
    va_ds = ConsolidatedDataset(X[va_idx], y[va_idx], augment=False)
    te_ds = ConsolidatedDataset(X[te_idx], y[te_idx], augment=False)

    sampler = make_balanced_sampler(tr_ds.labels)
    tr_dl = DataLoader(tr_ds, batch_size=128, sampler=sampler, num_workers=4,
                       pin_memory=True, drop_last=True)
    va_dl = DataLoader(va_ds, batch_size=128, num_workers=4, pin_memory=True)
    te_dl = DataLoader(te_ds, batch_size=128, num_workers=4, pin_memory=True)

    logger.info(f"Splits: {len(tr_ds)}/{len(va_ds)}/{len(te_ds)} (train/val/test)")
    logger.info(f"Device: {DEVICE}")

    # Build model — same architecture, 8 classes, input_dim=5000
    model = MicrobialEncoder(
        input_dim=5000,
        embed_dim=256,
        num_heads=4,
        num_aitchison_layers=4,
        ff_dim=512,
        dropout=0.15,
        num_sources=NUM_CLASSES,
        freeze_dnabert=True,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"MicroBiomeNet: {n_params:,} parameters")
    model.cache_sequence_embeddings(n_otus=5000)

    best_val_f1 = train(model, tr_dl, va_dl, epochs=100, lr=5e-4, patience=10)

    # Reload best
    best_path = CKPT / "microbiomenet_expanded_best.pt"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=DEVICE, weights_only=True))
        logger.info("Reloaded best checkpoint.")

    # Test
    model.eval()
    te_preds, te_labels = [], []
    with torch.no_grad():
        for batch in te_dl:
            clr = batch["clr"].to(DEVICE)
            labels = batch["source_label"]
            outputs = model(x=clr)
            preds = outputs["source_logits"].argmax(-1).cpu()
            te_preds.extend(preds.tolist())
            te_labels.extend(labels.tolist())

    test_f1 = f1_score(te_labels, te_preds, average="macro", zero_division=0)
    test_acc = accuracy_score(te_labels, te_preds)
    per_class_f1 = f1_score(te_labels, te_preds, average=None, zero_division=0)

    logger.info("=" * 60)
    logger.info("TEST RESULTS — Expanded Multi-Source Real Data")
    logger.info("=" * 60)
    unique_lbls = sorted(set(te_labels))
    for i, lid in enumerate(unique_lbls):
        name = CLASS_NAMES[lid] if lid < len(CLASS_NAMES) else f"class_{lid}"
        if i < len(per_class_f1):
            logger.info(f"  {name:>25}: F1 = {per_class_f1[i]:.4f}")

    logger.info(f"\n  Macro F1:    {test_f1:.4f}")
    logger.info(f"  Accuracy:    {test_acc:.4f}")
    logger.info(f"  Best Val F1: {best_val_f1:.4f}")

    if test_f1 > 0.70:
        logger.info("*** HARD THRESHOLD MET (F1 > 0.70) ***")
    elif test_f1 > 0.50:
        logger.info("ACCEPTABLE (F1 > 0.50)")
    else:
        logger.info(f"BELOW THRESHOLD ({test_f1:.4f})")

    report = classification_report(
        te_labels, te_preds,
        target_names=[CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"cls_{i}" for i in sorted(set(te_labels))],
        zero_division=0,
    )
    logger.info(f"\n{report}")

    elapsed = time.time() - t0
    results = {
        "test_f1": float(test_f1),
        "test_acc": float(test_acc),
        "best_val_f1": float(best_val_f1),
        "per_class_f1": {
            CLASS_NAMES[lid]: float(per_class_f1[i])
            for i, lid in enumerate(unique_lbls)
            if lid < len(CLASS_NAMES) and i < len(per_class_f1)
        },
        "n_train": len(tr_ds),
        "n_val": len(va_ds),
        "n_test": len(te_ds),
        "n_total": n,
        "data_sources": dict(sc),
        "elapsed_seconds": elapsed,
    }
    with open(CKPT / "results_expanded.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {CKPT / 'results_expanded.json'}")
    logger.info(f"Time: {elapsed/60:.1f}m")


if __name__ == "__main__":
    main()
