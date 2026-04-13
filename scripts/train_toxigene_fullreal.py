#!/usr/bin/env python3
"""ToxiGene training on strictly real data (1000 real zebrafish samples).

Identical architecture to train_toxigene_expanded.py with:
  - Loads expression_matrix_fullreal.npy (1000 real zebrafish samples, no synthetic)
  - 150 epochs, cosine LR decay, early stopping patience=15
  - AdamW + gradient clipping
  - Saves best model to checkpoints/molecular/toxigene_fullreal_best.pt
  - Saves results to checkpoints/molecular/results_fullreal.json

Why fullreal vs v2 (augmented)?
  All 15 GEO files are wrong organism/study type (human/mouse/cattle).
  ECOTOX training data has 32 engineered chemical-endpoint features, not gene expression.
  ecotox/real/ is empty. Dose-response profiles have no expression vectors.
  v2 rows 1000-1800 are Gaussian synthetic augmentations.
  → fullreal = strictly the 1000 real zebrafish measurements.

MIT License — Bryan Cheng, 2026
"""

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import f1_score, roc_auc_score
from scipy import sparse

from sentinel.models.molecular_encoder.model import MolecularEncoder
from sentinel.utils.logging import get_logger

logger = get_logger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT = Path("checkpoints/molecular")
CKPT.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path("data/processed/molecular")

EPOCHS = 150
BATCH_SIZE = 32
EARLY_STOP_PATIENCE = 15
SEED = 42


class MolecularDataset(Dataset):
    def __init__(self, expression, outcomes, pathways=None):
        self.expression = torch.tensor(expression.astype(np.float32))
        self.outcomes   = torch.tensor(outcomes.astype(np.float32))
        self.pathways   = (
            torch.tensor(pathways.astype(np.float32)) if pathways is not None else None
        )

    def __len__(self):
        return len(self.expression)

    def __getitem__(self, idx):
        item = {
            "expression": self.expression[idx],
            "outcomes":   self.outcomes[idx],
        }
        if self.pathways is not None:
            item["pathways"] = self.pathways[idx]
        return item


def load_sparse_adj(path):
    d = np.load(path)
    shape = tuple(d["shape"])
    mat = sparse.csr_matrix((d["data"], d["indices"], d["indptr"]), shape=shape)
    return torch.tensor(mat.toarray(), dtype=torch.float32)


def main():
    t0 = time.time()

    # ── Load fullreal data ─────────────────────────────────────────────────
    expr_path = DATA_DIR / "expression_matrix_fullreal.npy"
    if not expr_path.exists():
        logger.error("expression_matrix_fullreal.npy not found. Run expand_toxigene_data_real.py first.")
        sys.exit(1)

    expression = np.load(expr_path)
    outcomes   = np.load(DATA_DIR / "outcome_labels_fullreal.npy")
    pathways   = np.load(DATA_DIR / "pathway_labels_fullreal.npy")
    gene_names = json.load(open(DATA_DIR / "gene_names.json"))

    pathway_adj = load_sparse_adj(DATA_DIR / "hierarchy_layer0_gene_to_pathway.npz")
    process_adj = load_sparse_adj(DATA_DIR / "hierarchy_layer1_pathway_to_process.npz")
    outcome_adj = load_sparse_adj(DATA_DIR / "hierarchy_layer2_process_to_outcome.npz")

    logger.info(
        f"Full-real data: {expression.shape[0]} samples, {expression.shape[1]} genes, "
        f"{outcomes.shape[1]} outcomes, {pathway_adj.shape[0]} pathways"
    )
    logger.info(
        f"Hierarchy: genes({expression.shape[1]}) → pathways({pathway_adj.shape[0]}) "
        f"→ processes({process_adj.shape[0]}) → outcomes({outcome_adj.shape[0]})"
    )

    # Normalize expression to z-scores
    expr_mean = expression.mean(axis=0)
    expr_std  = expression.std(axis=0)
    expr_std[expr_std < 1e-6] = 1.0
    expression = (expression - expr_mean) / expr_std

    # Dataset / splits: 70 / 15 / 15
    ds = MolecularDataset(expression, outcomes, pathways)
    n  = len(ds)
    n_tr = int(0.70 * n)
    n_va = int(0.15 * n)
    n_te = n - n_tr - n_va

    tr, va, te = random_split(
        ds, [n_tr, n_va, n_te],
        generator=torch.Generator().manual_seed(SEED)
    )
    logger.info(f"Split: {n_tr} train / {n_va} val / {n_te} test")

    tr_dl = DataLoader(tr, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    va_dl = DataLoader(va, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)
    te_dl = DataLoader(te, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)

    # ── Build model ────────────────────────────────────────────────────────
    model = MolecularEncoder(
        gene_names=gene_names,
        pathway_adj=pathway_adj,
        process_adj=process_adj,
        outcome_adj=outcome_adj,
        num_chem_classes=50,
        lambda_l1=0.01,
        dropout=0.2,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"ToxiGene: {n_params:,} parameters on {DEVICE}")

    # Optimizer + cosine LR scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_f1    = 0.0
    best_ckpt  = CKPT / "toxigene_fullreal_best.pt"
    epochs_no_improve = 0

    # ── Training loop ──────────────────────────────────────────────────────
    for epoch in range(EPOCHS):
        model.train()
        total_loss, nb = 0, 0
        all_preds, all_labels = [], []

        for batch in tr_dl:
            expr            = batch["expression"].to(DEVICE)
            outcome_targets = batch["outcomes"].to(DEVICE)
            pathway_targets = batch.get("pathways")
            if pathway_targets is not None:
                pathway_targets = pathway_targets.to(DEVICE)

            outputs = model(gene_expression=expr)
            losses  = model.compute_loss(
                outputs=outputs,
                outcome_targets=outcome_targets,
                pathway_targets=pathway_targets,
            )
            loss = losses["total"]

            if torch.isnan(loss):
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            nb += 1

            preds = (torch.sigmoid(outputs["outcome_logits"]) > 0.5).float().cpu()
            all_preds.append(preds)
            all_labels.append(outcome_targets.cpu())

        scheduler.step()
        if nb == 0:
            continue

        all_preds  = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        train_f1   = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        # Validation
        model.eval()
        va_preds, va_labels = [], []
        with torch.no_grad():
            for batch in va_dl:
                expr    = batch["expression"].to(DEVICE)
                outputs = model(gene_expression=expr)
                preds   = (torch.sigmoid(outputs["outcome_logits"]) > 0.5).float().cpu()
                va_preds.append(preds)
                va_labels.append(batch["outcomes"])

        va_preds  = torch.cat(va_preds).numpy()
        va_labels = torch.cat(va_labels).numpy()
        val_f1    = f1_score(va_labels, va_preds, average="macro", zero_division=0)
        n_sel     = outputs["num_selected_genes"].item()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"Ep {epoch+1:3d}/{EPOCHS} | Loss: {total_loss/nb:.4f} | "
                f"Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f} | "
                f"Genes: {n_sel} | Patience: {epochs_no_improve}/{EARLY_STOP_PATIENCE}"
            )

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), best_ckpt)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            logger.info(f"Early stopping at epoch {epoch+1} (patience={EARLY_STOP_PATIENCE})")
            break

    # ── Reload best and evaluate on test ──────────────────────────────────
    if best_ckpt.exists():
        model.load_state_dict(torch.load(best_ckpt, map_location=DEVICE, weights_only=True))
        logger.info(f"Loaded best checkpoint (val F1={best_f1:.4f})")

    model.eval()
    te_preds, te_labels, te_probs = [], [], []
    with torch.no_grad():
        for batch in te_dl:
            expr    = batch["expression"].to(DEVICE)
            outputs = model(gene_expression=expr)
            probs   = torch.sigmoid(outputs["outcome_logits"]).cpu()
            preds   = (probs > 0.5).float()
            te_preds.append(preds)
            te_labels.append(batch["outcomes"])
            te_probs.append(probs)

    te_preds  = torch.cat(te_preds).numpy()
    te_labels = torch.cat(te_labels).numpy()
    te_probs  = torch.cat(te_probs).numpy()

    test_f1      = f1_score(te_labels, te_preds, average="macro", zero_division=0)
    test_acc     = float((te_preds == te_labels).mean())
    per_class_f1 = f1_score(te_labels, te_preds, average=None, zero_division=0)

    try:
        auroc = float(roc_auc_score(te_labels, te_probs, average="macro"))
    except Exception:
        auroc = None

    # Gene selection analysis
    try:
        selected_genes = model.bottleneck.get_selected_genes()
        n_selected     = model.bottleneck.num_selected
    except Exception:
        selected_genes = []
        n_selected     = 0

    outcome_names = [
        "reproductive_impairment", "growth_inhibition", "immunosuppression",
        "neurotoxicity", "hepatotoxicity", "oxidative_damage", "endocrine_disruption"
    ]

    logger.info("=" * 60)
    logger.info("TEST RESULTS (full-real dataset)")
    logger.info("=" * 60)
    for i, (name, f) in enumerate(zip(outcome_names, per_class_f1)):
        logger.info(f"  {name}: F1 = {f:.4f}")
    logger.info(f"\n  Macro F1:  {test_f1:.4f}")
    logger.info(f"  Accuracy:  {test_acc:.4f}")
    if auroc is not None:
        logger.info(f"  AUROC:     {auroc:.4f}")
    logger.info(f"  Selected genes: {n_selected}/{len(gene_names)}")
    if selected_genes:
        logger.info(f"  Top selected: {selected_genes[:20]}")

    if test_f1 > 0.80:
        logger.info("*** HARD THRESHOLD MET ***")
    elif test_f1 > 0.60:
        logger.info("ACCEPTABLE")
    else:
        logger.info(f"BELOW THRESHOLD ({test_f1:.4f})")

    print(f"\n=== ToxiGene Full-Real Results ===")
    print(f"Test Macro-F1  : {test_f1:.4f}")
    print(f"Test Accuracy  : {test_acc:.4f}")
    if auroc is not None:
        print(f"Test AUROC     : {auroc:.4f}")
    print(f"Best Val F1    : {best_f1:.4f}")
    print(f"Train/Val/Test : {n_tr}/{n_va}/{n_te}")
    print(f"Per-class F1   :")
    for name, f in zip(outcome_names, per_class_f1):
        print(f"  {name}: {f:.4f}")

    elapsed = time.time() - t0
    results = {
        "model":              "ToxiGene_fullreal",
        "dataset":            "expression_matrix_fullreal.npy",
        "n_real_samples":     int(expression.shape[0]),
        "n_synthetic_samples": 0,
        "test_f1_macro":      float(test_f1),
        "test_acc":           test_acc,
        "test_auroc_macro":   auroc,
        "best_val_f1":        float(best_f1),
        "per_class_f1":       {name: float(f) for name, f in zip(outcome_names, per_class_f1)},
        "n_train":            n_tr,
        "n_val":              n_va,
        "n_test":             n_te,
        "epochs_trained":     epoch + 1,
        "max_epochs":         EPOCHS,
        "early_stop_patience": EARLY_STOP_PATIENCE,
        "n_selected_genes":   n_selected,
        "selected_genes_top20": selected_genes[:20] if selected_genes else [],
        "elapsed_s":          elapsed,
        "data_sources": {
            "expression_matrix_fullreal": "1000 real zebrafish samples — USED",
            "expression_matrix_v2_extra_800": "Synthetic Gaussian augmentation — EXCLUDED",
            "geo_files": "Wrong organism/study type — INCOMPATIBLE",
            "ecotox_training": "32 chemical-endpoint features, not expression — INCOMPATIBLE",
            "ecotox_real": "Empty directory — SKIPPED",
            "dose_response_profiles": "Chemical metadata only — INCOMPATIBLE"
        }
    }
    out_path = CKPT / "results_fullreal.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    logger.info(f"Results saved to {out_path}")
    logger.info(f"Time: {elapsed/60:.1f}m")
    logger.info("DONE")


if __name__ == "__main__":
    main()
