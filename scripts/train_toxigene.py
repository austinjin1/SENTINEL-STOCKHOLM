#!/usr/bin/env python3
"""ToxiGene training: pathway-informed transcriptomic toxicity classification.

Trains the biological hierarchy network (gene → pathway → process → outcome)
with gene selection bottleneck for minimal biomarker panel discovery.

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
from scipy import sparse

from sentinel.models.molecular_encoder.model import MolecularEncoder
from sentinel.utils.logging import get_logger

logger = get_logger(__name__)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CKPT = Path("checkpoints/molecular")
CKPT.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path("data/processed/molecular")


class MolecularDataset(Dataset):
    def __init__(self, expression, outcomes, pathways=None):
        self.expression = torch.tensor(expression.astype(np.float32))
        self.outcomes = torch.tensor(outcomes.astype(np.float32))
        self.pathways = (
            torch.tensor(pathways.astype(np.float32)) if pathways is not None else None
        )

    def __len__(self):
        return len(self.expression)

    def __getitem__(self, idx):
        item = {
            "expression": self.expression[idx],
            "outcomes": self.outcomes[idx],
        }
        if self.pathways is not None:
            item["pathways"] = self.pathways[idx]
        return item


def load_sparse_adj(path):
    """Load a sparse adjacency matrix saved as npz."""
    d = np.load(path)
    shape = tuple(d["shape"])
    mat = sparse.csr_matrix(
        (d["data"], d["indices"], d["indptr"]), shape=shape
    )
    return torch.tensor(mat.toarray(), dtype=torch.float32)


def main():
    t0 = time.time()

    # Load data
    expression = np.load(DATA_DIR / "expression_matrix.npy")
    outcomes = np.load(DATA_DIR / "outcome_labels.npy")
    pathways = np.load(DATA_DIR / "pathway_labels.npy")
    gene_names = json.load(open(DATA_DIR / "gene_names.json"))

    # Load hierarchy adjacency matrices
    pathway_adj = load_sparse_adj(DATA_DIR / "hierarchy_layer0_gene_to_pathway.npz")
    process_adj = load_sparse_adj(DATA_DIR / "hierarchy_layer1_pathway_to_process.npz")
    outcome_adj = load_sparse_adj(DATA_DIR / "hierarchy_layer2_process_to_outcome.npz")

    logger.info(
        f"Data: {expression.shape[0]} samples, {expression.shape[1]} genes, "
        f"{outcomes.shape[1]} outcomes, {pathway_adj.shape[0]} pathways"
    )
    logger.info(
        f"Hierarchy: genes({expression.shape[1]}) → pathways({pathway_adj.shape[0]}) "
        f"→ processes({process_adj.shape[0]}) → outcomes({outcome_adj.shape[0]})"
    )

    # Normalize expression to z-scores
    expr_mean = expression.mean(axis=0)
    expr_std = expression.std(axis=0)
    expr_std[expr_std < 1e-6] = 1.0
    expression = (expression - expr_mean) / expr_std

    # Dataset
    ds = MolecularDataset(expression, outcomes, pathways)
    n = len(ds)
    n_tr = int(0.7 * n)
    n_va = int(0.15 * n)
    n_te = n - n_tr - n_va
    tr, va, te = random_split(
        ds, [n_tr, n_va, n_te], generator=torch.Generator().manual_seed(42)
    )

    tr_dl = DataLoader(tr, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    va_dl = DataLoader(va, batch_size=32, num_workers=2, pin_memory=True)
    te_dl = DataLoader(te, batch_size=32, num_workers=2, pin_memory=True)

    logger.info(f"Split: {n_tr}/{n_va}/{n_te}")

    # Build model
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
    logger.info(f"ToxiGene: {n_params:,} parameters")

    # Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)
    best_f1 = 0.0

    for epoch in range(80):
        model.train()
        total_loss, nb = 0, 0
        all_preds, all_labels = [], []

        for batch in tr_dl:
            expr = batch["expression"].to(DEVICE)
            outcome_targets = batch["outcomes"].to(DEVICE)
            pathway_targets = batch.get("pathways")
            if pathway_targets is not None:
                pathway_targets = pathway_targets.to(DEVICE)

            outputs = model(gene_expression=expr)
            losses = model.compute_loss(
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

        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        train_f1 = f1_score(
            all_labels, all_preds, average="macro", zero_division=0
        )

        # Validation
        model.eval()
        va_preds, va_labels = [], []
        with torch.no_grad():
            for batch in va_dl:
                expr = batch["expression"].to(DEVICE)
                outputs = model(gene_expression=expr)
                preds = (torch.sigmoid(outputs["outcome_logits"]) > 0.5).float().cpu()
                va_preds.append(preds)
                va_labels.append(batch["outcomes"])

        va_preds = torch.cat(va_preds).numpy()
        va_labels = torch.cat(va_labels).numpy()
        val_f1 = f1_score(va_labels, va_preds, average="macro", zero_division=0)
        n_selected = outputs["num_selected_genes"].item()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"Ep {epoch+1:3d}/80 | Loss: {total_loss/nb:.4f} | "
                f"Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f} | "
                f"Genes: {n_selected}"
            )

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), CKPT / "toxigene_best.pt")

    # Reload best
    best_path = CKPT / "toxigene_best.pt"
    if best_path.exists():
        model.load_state_dict(
            torch.load(best_path, map_location=DEVICE, weights_only=True)
        )

    # Test
    model.eval()
    te_preds, te_labels, te_probs = [], [], []
    with torch.no_grad():
        for batch in te_dl:
            expr = batch["expression"].to(DEVICE)
            outputs = model(gene_expression=expr)
            probs = torch.sigmoid(outputs["outcome_logits"]).cpu()
            preds = (probs > 0.5).float()
            te_preds.append(preds)
            te_labels.append(batch["outcomes"])
            te_probs.append(probs)

    te_preds = torch.cat(te_preds).numpy()
    te_labels = torch.cat(te_labels).numpy()
    te_probs = torch.cat(te_probs).numpy()

    test_f1 = f1_score(te_labels, te_preds, average="macro", zero_division=0)
    test_acc = (te_preds == te_labels).mean()

    # Per-outcome metrics
    n_outcomes = te_labels.shape[1]
    per_outcome_f1 = f1_score(te_labels, te_preds, average=None, zero_division=0)

    # Gene selection analysis
    selected_genes = model.bottleneck.get_selected_genes()
    n_selected = model.bottleneck.num_selected

    logger.info("=" * 60)
    logger.info("TEST RESULTS")
    logger.info("=" * 60)
    for i in range(n_outcomes):
        logger.info(f"  Outcome {i}: F1 = {per_outcome_f1[i]:.4f}")
    logger.info(f"\n  Macro F1: {test_f1:.4f}")
    logger.info(f"  Accuracy: {test_acc:.4f}")
    logger.info(f"  Selected genes: {n_selected}/{len(gene_names)}")
    if selected_genes:
        logger.info(f"  Top selected: {selected_genes[:20]}")

    if test_f1 > 0.80:
        logger.info("*** HARD THRESHOLD MET ***")
    elif test_f1 > 0.60:
        logger.info("ACCEPTABLE")
    else:
        logger.info(f"BELOW THRESHOLD ({test_f1:.4f})")

    elapsed = time.time() - t0
    results = {
        "test_macro_f1": float(test_f1),
        "test_accuracy": float(test_acc),
        "best_val_f1": float(best_f1),
        "per_outcome_f1": [float(x) for x in per_outcome_f1],
        "n_selected_genes": n_selected,
        "selected_genes_top20": selected_genes[:20] if selected_genes else [],
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
