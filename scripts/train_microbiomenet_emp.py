#!/usr/bin/env python3
"""Retrain MicroBiomeNet on real EMP 16S rRNA OTU data.

Uses Earth Microbiome Project deblur-processed 16S ASV tables with
proper compositional data on the simplex. Applies CLR transform
and trains the Aitchison-geometry-aware encoder for source attribution.

Key improvements over chemistry-only training:
- Real compositional OTU data (relative abundances sum to 1)
- Class-balanced sampling to handle imbalanced environment types
- Label smoothing + mixup augmentation for better generalization
- Cosine annealing with warm restarts

Target: Push F1 past 0.70 threshold (currently 0.698 on NARS chemistry).

MIT License — Bryan Cheng, 2026
"""

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import (
    DataLoader, Dataset, WeightedRandomSampler, random_split, Subset
)
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report, confusion_matrix
)

from sentinel.models.microbial_encoder.model import MicrobialEncoder
from sentinel.models.microbial_encoder.aitchison_attention import clr_transform
from sentinel.utils.logging import get_logger

logger = get_logger(__name__)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CKPT = Path("checkpoints/microbial")
CKPT.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path("data/processed/microbial/emp_16s")

# EMP source classes
SOURCE_NAMES = [
    "freshwater_natural",
    "freshwater_impacted",
    "saline_water",
    "freshwater_sediment",
    "saline_sediment",
    "soil_runoff",
    "animal_fecal",
    "plant_associated",
]
NUM_SOURCES = len(SOURCE_NAMES)


class EMP16SDataset(Dataset):
    """Dataset for EMP 16S rRNA OTU data.

    Loads per-sample .npz files containing relative abundance vectors
    on the simplex (sum to 1). Applies CLR transform for model input.
    """

    def __init__(self, data_dir, max_samples=None, augment=False):
        self.files = sorted(Path(data_dir).glob("*.npz"))
        if max_samples:
            self.files = self.files[:max_samples]
        self.augment = augment

        # Pre-load labels for balanced sampling
        self._labels = []
        valid_files = []
        for f in self.files:
            try:
                data = np.load(f, allow_pickle=True)
                label = int(data["source_label"])
                abund = data["abundances"]
                # Skip empty samples
                if abund.sum() < 1e-8:
                    continue
                self._labels.append(label)
                valid_files.append(f)
            except Exception:
                continue
        self.files = valid_files

    def __len__(self):
        return len(self.files)

    @property
    def labels(self):
        return self._labels

    def __getitem__(self, idx):
        data = np.load(self.files[idx], allow_pickle=True)
        abundances = data["abundances"].astype(np.float32)
        source_label = int(data["source_label"])

        # Ensure proper compositional data (sum to 1)
        if abundances.sum() > 0:
            abundances = abundances / abundances.sum()

        # Data augmentation for training
        if self.augment:
            abundances = self._augment_composition(abundances)

        # Convert to tensor
        abundances = torch.tensor(abundances)

        # CLR transform with pseudocount for zeros
        # Use small pseudocount to handle sparse OTU data
        pseudocount = 1e-6
        abund_with_pseudo = abundances + pseudocount
        abund_with_pseudo = abund_with_pseudo / abund_with_pseudo.sum()
        clr = clr_transform(abund_with_pseudo.unsqueeze(0)).squeeze(0)
        clr = clr.clamp(-8, 8).float()  # Ensure float32

        return {
            "abundances": abundances.float(),
            "clr": clr,
            "source_label": source_label,
        }

    @staticmethod
    def _augment_composition(x: np.ndarray) -> np.ndarray:
        """Compositional data augmentation preserving simplex constraint.

        Uses multiplicative perturbation (Aitchison perturbation) which
        is the proper way to add noise on the simplex.
        """
        rng = np.random.default_rng()

        # 1. Multiplicative noise (Aitchison perturbation)
        if rng.random() < 0.5:
            noise = rng.lognormal(0, 0.1, size=x.shape)
            x = x * noise
            if x.sum() > 0:
                x = x / x.sum()

        # 2. Random zero-out (simulates undetected taxa)
        if rng.random() < 0.3:
            nonzero = np.where(x > 0)[0]
            if len(nonzero) > 10:
                n_drop = rng.integers(1, max(2, len(nonzero) // 10))
                drop_idx = rng.choice(nonzero, size=n_drop, replace=False)
                x[drop_idx] = 0
                if x.sum() > 0:
                    x = x / x.sum()

        return x


def make_balanced_sampler(dataset: EMP16SDataset) -> WeightedRandomSampler:
    """Create a weighted sampler that balances class frequencies."""
    labels = dataset.labels
    class_counts = Counter(labels)
    n_classes = len(class_counts)
    total = len(labels)

    # Inverse frequency weighting
    weights = []
    for label in labels:
        w = total / (n_classes * class_counts[label])
        weights.append(w)

    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )


def mixup_data(x, y, alpha=0.2):
    """Mixup augmentation for compositional data.

    In CLR space, mixup is a weighted average which corresponds to
    a power mean on the simplex.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute mixup loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_source_classification(
    model, tr_dl, va_dl, epochs=80, lr=5e-4, use_mixup=True
):
    """Train source attribution classifier with class balancing and mixup."""
    logger.info("=" * 60)
    logger.info("EMP 16S SOURCE CLASSIFICATION TRAINING")
    logger.info("=" * 60)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.02)

    # Cosine annealing with warm restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )

    # Label smoothing cross-entropy
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_f1 = 0.0
    patience = 15
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        total_loss, nb = 0, 0
        all_preds, all_labels = [], []

        for batch in tr_dl:
            clr = batch["clr"].to(DEVICE)
            labels = batch["source_label"].to(DEVICE)

            # Mixup augmentation
            if use_mixup and np.random.random() < 0.5:
                clr_mixed, labels_a, labels_b, lam = mixup_data(clr, labels)
                outputs = model(x=clr_mixed)
                loss = mixup_criterion(
                    criterion, outputs["source_logits"], labels_a, labels_b, lam
                )
                preds = outputs["source_logits"].argmax(dim=-1).cpu()
            else:
                outputs = model(x=clr)
                loss_dict = model.compute_loss(
                    x=clr, outputs=outputs, source_targets=labels
                )
                loss = loss_dict["total"]
                preds = outputs["source_logits"].argmax(dim=-1).cpu()

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
                preds = outputs["source_logits"].argmax(dim=-1).cpu()
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
            torch.save(model.state_dict(), CKPT / "microbiomenet_real_best.pt")
            logger.info(f"  -> New best Val F1: {val_f1:.4f} (saved)")
        else:
            no_improve += 1

        if no_improve >= patience and epoch > 30:
            logger.info(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

    return best_f1


def main():
    t0 = time.time()

    logger.info("Loading EMP 16S rRNA OTU dataset...")
    logger.info(f"Data directory: {DATA_DIR}")
    logger.info(f"Device: {DEVICE}")

    # Load dataset with augmentation for training
    full_ds = EMP16SDataset(DATA_DIR, max_samples=None, augment=False)
    n = len(full_ds)
    logger.info(f"Total samples: {n}")

    # Class distribution
    label_counts = Counter(full_ds.labels)
    for label_id in sorted(label_counts):
        name = SOURCE_NAMES[label_id] if label_id < len(SOURCE_NAMES) else f"class_{label_id}"
        logger.info(f"  {name:>25}: {label_counts[label_id]:,}")

    # Split: 70/15/15
    n_tr = int(0.70 * n)
    n_va = int(0.15 * n)
    n_te = n - n_tr - n_va

    generator = torch.Generator().manual_seed(42)
    tr_indices, va_indices, te_indices = random_split(
        range(n), [n_tr, n_va, n_te], generator=generator
    )

    # Create augmented training dataset wrapper
    class AugmentedSubset(Dataset):
        def __init__(self, dataset, indices, augment=True):
            self.dataset = dataset
            self.indices = list(indices)
            self.augment = augment
            self._labels = [dataset.labels[i] for i in self.indices]

        def __len__(self):
            return len(self.indices)

        @property
        def labels(self):
            return self._labels

        def __getitem__(self, idx):
            real_idx = self.indices[idx]
            data = np.load(self.dataset.files[real_idx], allow_pickle=True)
            abundances = data["abundances"].astype(np.float32)
            source_label = int(data["source_label"])

            if abundances.sum() > 0:
                abundances = abundances / abundances.sum()

            if self.augment:
                abundances = EMP16SDataset._augment_composition(abundances)

            abundances = torch.tensor(abundances)
            pseudocount = 1e-6
            abund_with_pseudo = abundances + pseudocount
            abund_with_pseudo = abund_with_pseudo / abund_with_pseudo.sum()
            clr = clr_transform(abund_with_pseudo.unsqueeze(0)).squeeze(0)
            clr = clr.clamp(-8, 8).float()  # Ensure float32

            return {
                "abundances": abundances.float(),
                "clr": clr,
                "source_label": source_label,
            }

    tr_ds = AugmentedSubset(full_ds, tr_indices, augment=True)
    va_ds = AugmentedSubset(full_ds, va_indices, augment=False)
    te_ds = AugmentedSubset(full_ds, te_indices, augment=False)

    # Balanced sampler for training
    sampler = make_balanced_sampler(tr_ds)

    tr_dl = DataLoader(
        tr_ds, batch_size=64, sampler=sampler, num_workers=2,
        pin_memory=True, drop_last=True,
    )
    va_dl = DataLoader(va_ds, batch_size=64, num_workers=2, pin_memory=True)
    te_dl = DataLoader(te_ds, batch_size=64, num_workers=2, pin_memory=True)

    logger.info(f"Splits: {n_tr}/{n_va}/{n_te} (train/val/test)")
    logger.info(f"Features per sample: {full_ds[0]['clr'].shape[0]}")

    # Build model
    model = MicrobialEncoder(
        input_dim=5000,
        embed_dim=256,
        num_heads=4,
        num_aitchison_layers=4,
        ff_dim=512,
        dropout=0.15,
        num_sources=NUM_SOURCES,
        freeze_dnabert=True,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"MicroBiomeNet: {n_params:,} parameters")

    # Cache sequence embeddings
    model.cache_sequence_embeddings(n_otus=5000)

    # Train
    best_f1 = train_source_classification(
        model, tr_dl, va_dl,
        epochs=80, lr=5e-4, use_mixup=True,
    )

    # Reload best checkpoint
    best_path = CKPT / "microbiomenet_real_best.pt"
    if best_path.exists():
        model.load_state_dict(
            torch.load(best_path, map_location=DEVICE, weights_only=True)
        )

    # Test evaluation
    model.eval()
    te_preds, te_labels = [], []
    with torch.no_grad():
        for batch in te_dl:
            clr = batch["clr"].to(DEVICE)
            labels = batch["source_label"]
            outputs = model(x=clr)
            preds = outputs["source_logits"].argmax(dim=-1).cpu()
            te_preds.extend(preds.tolist())
            te_labels.extend(labels.tolist())

    test_f1 = f1_score(te_labels, te_preds, average="macro", zero_division=0)
    test_acc = accuracy_score(te_labels, te_preds)
    per_class_f1 = f1_score(te_labels, te_preds, average=None, zero_division=0)

    logger.info("=" * 60)
    logger.info("TEST RESULTS — EMP 16S Real OTU Data")
    logger.info("=" * 60)

    # Per-class F1
    unique_labels = sorted(set(te_labels))
    for i, label_id in enumerate(unique_labels):
        name = SOURCE_NAMES[label_id] if label_id < len(SOURCE_NAMES) else f"class_{label_id}"
        if i < len(per_class_f1):
            logger.info(f"  {name:>25}: F1 = {per_class_f1[i]:.4f}")

    logger.info(f"\n  Macro F1:  {test_f1:.4f}")
    logger.info(f"  Accuracy:  {test_acc:.4f}")
    logger.info(f"  Best Val F1: {best_f1:.4f}")

    if test_f1 > 0.70:
        logger.info("*** HARD THRESHOLD MET (F1 > 0.70) ***")
    elif test_f1 > 0.50:
        logger.info("ACCEPTABLE (F1 > 0.50 but below 0.70)")
    else:
        logger.info(f"BELOW THRESHOLD ({test_f1:.4f})")

    # Confusion matrix
    cm = confusion_matrix(te_labels, te_preds)
    logger.info("\nConfusion Matrix:")
    logger.info(f"\n{cm}")

    # Full classification report
    target_names = [
        SOURCE_NAMES[i] if i < len(SOURCE_NAMES) else f"class_{i}"
        for i in sorted(set(te_labels))
    ]
    report = classification_report(
        te_labels, te_preds, target_names=target_names, zero_division=0
    )
    logger.info(f"\n{report}")

    elapsed = time.time() - t0
    results = {
        "test_macro_f1": float(test_f1),
        "test_accuracy": float(test_acc),
        "best_val_f1": float(best_f1),
        "per_class_f1": {
            SOURCE_NAMES[label_id]: float(per_class_f1[i])
            for i, label_id in enumerate(sorted(set(te_labels)))
            if label_id < len(SOURCE_NAMES) and i < len(per_class_f1)
        },
        "elapsed_seconds": elapsed,
        "n_train": n_tr,
        "n_val": n_va,
        "n_test": n_te,
        "n_total": n,
        "n_classes": NUM_SOURCES,
        "data": "EMP_16S_release1",
        "class_counts": {
            SOURCE_NAMES[k]: v
            for k, v in sorted(label_counts.items())
            if k < len(SOURCE_NAMES)
        },
    }
    with open(CKPT / "results_real.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nTime: {elapsed/60:.1f}m")
    logger.info(f"Results saved to {CKPT / 'results_real.json'}")
    logger.info("DONE")


if __name__ == "__main__":
    main()
