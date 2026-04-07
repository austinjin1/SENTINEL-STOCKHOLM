#!/usr/bin/env python3
"""MicroBiomeNet training: source attribution from microbial community composition.

Phase 1: Self-supervised contrastive pretraining on CLR-transformed abundances
Phase 2: Supervised source classification (8 pollution source types)

MIT License — Bryan Cheng, 2026
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import f1_score, roc_auc_score

from sentinel.models.microbial_encoder.model import MicrobialEncoder
from sentinel.models.microbial_encoder.aitchison_attention import clr_transform
from sentinel.utils.logging import get_logger

logger = get_logger(__name__)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CKPT = Path("checkpoints/microbial")
CKPT.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path("data/processed/microbial/nars")

NUM_SOURCES = 8
SOURCE_NAMES = [
    "nutrient", "heavy_metals", "thermal", "pharmaceutical",
    "sediment", "oil_petrochemical", "sewage", "acid_mine",
]


class MicrobialDataset(Dataset):
    def __init__(self, data_dir, max_samples=None):
        self.files = sorted(Path(data_dir).glob("*.npz"))
        if max_samples:
            self.files = self.files[:max_samples]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx], allow_pickle=True)
        abundances = torch.tensor(data["abundances"].astype(np.float32))
        source_label = int(data["source_label"])
        # CLR transform with clamping to prevent gradient explosion
        clr = clr_transform(abundances.unsqueeze(0) + 1e-10).squeeze(0)
        clr = clr.clamp(-5, 5)  # Prevent extreme values from sparse taxa
        return {"abundances": abundances, "clr": clr, "source_label": source_label}


def train_source_classification(model, tr_dl, va_dl, epochs=60, lr=1e-3):
    """Train source attribution classifier."""
    logger.info("=" * 60)
    logger.info("SOURCE CLASSIFICATION TRAINING")
    logger.info("=" * 60)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    # Warmup + cosine decay
    warmup_epochs = 5
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    best_f1 = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss, nb = 0, 0
        all_preds, all_labels = [], []

        for batch in tr_dl:
            clr = batch["clr"].to(DEVICE)
            labels = batch["source_label"].to(DEVICE)

            outputs = model(x=clr)
            loss_dict = model.compute_loss(
                x=clr, outputs=outputs, source_targets=labels
            )
            loss = loss_dict["total"]

            if torch.isnan(loss):
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            nb += 1

            preds = outputs["source_logits"].argmax(dim=-1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().tolist())

        scheduler.step()
        train_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        # Validation
        model.eval()
        va_preds, va_labels, va_probs = [], [], []
        with torch.no_grad():
            for batch in va_dl:
                clr = batch["clr"].to(DEVICE)
                labels = batch["source_label"]
                outputs = model(x=clr)
                preds = outputs["source_logits"].argmax(dim=-1).cpu()
                probs = outputs["source_probs"].cpu()
                va_preds.extend(preds.tolist())
                va_labels.extend(labels.tolist())
                va_probs.append(probs)

        val_f1 = f1_score(va_labels, va_preds, average="macro", zero_division=0)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                f"Ep {epoch+1:3d}/{epochs} | Loss: {total_loss/max(nb,1):.4f} | "
                f"Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}"
            )

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), CKPT / "microbiomenet_best.pt")

    return best_f1


def main():
    t0 = time.time()

    ds = MicrobialDataset(DATA_DIR, max_samples=50000)
    n = len(ds)
    n_tr = int(0.7 * n)
    n_va = int(0.15 * n)
    n_te = n - n_tr - n_va
    tr, va, te = random_split(
        ds, [n_tr, n_va, n_te], generator=torch.Generator().manual_seed(42)
    )

    tr_dl = DataLoader(tr, batch_size=64, shuffle=True, num_workers=0)
    va_dl = DataLoader(va, batch_size=64, num_workers=0)
    te_dl = DataLoader(te, batch_size=64, num_workers=0)

    logger.info(f"Data: {n_tr}/{n_va}/{n_te} samples, {ds[0]['clr'].shape[0]} features")

    # Use smaller model for training efficiency
    model = MicrobialEncoder(
        input_dim=5000,
        embed_dim=256,
        num_heads=4,
        num_aitchison_layers=4,
        ff_dim=512,
        dropout=0.1,
        freeze_dnabert=True,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"MicroBiomeNet: {n_params:,} parameters")

    # Cache sequence embeddings (avoids DNABERT-S per-batch)
    model.cache_sequence_embeddings(n_otus=5000)

    best_f1 = train_source_classification(model, tr_dl, va_dl, epochs=60, lr=3e-4)

    # Reload best and test
    best_path = CKPT / "microbiomenet_best.pt"
    if best_path.exists():
        model.load_state_dict(
            torch.load(best_path, map_location=DEVICE, weights_only=True)
        )

    model.eval()
    te_preds, te_labels, te_probs = [], [], []
    with torch.no_grad():
        for batch in te_dl:
            clr = batch["clr"].to(DEVICE)
            labels = batch["source_label"]
            outputs = model(x=clr)
            preds = outputs["source_logits"].argmax(dim=-1).cpu()
            probs = outputs["source_probs"].cpu()
            te_preds.extend(preds.tolist())
            te_labels.extend(labels.tolist())
            te_probs.append(probs)

    test_f1 = f1_score(te_labels, te_preds, average="macro", zero_division=0)
    test_acc = sum(p == l for p, l in zip(te_preds, te_labels)) / len(te_labels)

    # Per-class F1
    per_class_f1 = f1_score(te_labels, te_preds, average=None, zero_division=0)
    logger.info("=" * 60)
    logger.info("TEST RESULTS")
    logger.info("=" * 60)
    for i, name in enumerate(SOURCE_NAMES):
        if i < len(per_class_f1):
            logger.info(f"  {name:>20}: F1 = {per_class_f1[i]:.4f}")
    logger.info(f"\n  Macro F1: {test_f1:.4f}")
    logger.info(f"  Accuracy: {test_acc:.4f}")

    if test_f1 > 0.70:
        logger.info("*** HARD THRESHOLD MET ***")
    elif test_f1 > 0.50:
        logger.info("ACCEPTABLE")
    else:
        logger.info(f"BELOW THRESHOLD ({test_f1:.4f})")

    elapsed = time.time() - t0
    results = {
        "test_macro_f1": test_f1,
        "test_accuracy": test_acc,
        "best_val_f1": best_f1,
        "per_class_f1": {
            SOURCE_NAMES[i]: float(per_class_f1[i])
            for i in range(min(len(SOURCE_NAMES), len(per_class_f1)))
        },
        "elapsed": elapsed,
        "n_train": n_tr,
        "n_test": n_te,
    }
    with open(CKPT / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Time: {elapsed/60:.1f}m")
    logger.info("DONE")


if __name__ == "__main__":
    main()
