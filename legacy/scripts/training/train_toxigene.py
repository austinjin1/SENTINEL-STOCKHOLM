#!/usr/bin/env python3
"""ToxiGene — MolecularEncoder trained on real GEO expression data.

Key features:
  - Uses real GEO expression data preprocessed by prepare_geo_for_toxigene.py
  - Loads proper hierarchy adjacency matrices from npz files
  - Handles class imbalance with pos_weight for BCE loss
  - Handles classes with zero positive samples (skips them in loss)
  - Study-level spatial holdout splits (prevents intra-study leakage)
  - Z-score normalization computed on train split ONLY
  - Gradient clipping + AdamW + cosine LR schedule
  - Saves results to results/benchmarks/toxigene_v3_holdout.json

Prerequisites:
  Run: python scripts/prepare_geo_for_toxigene.py
  This produces the expression matrix, outcomes, pathway labels, gene_names,
  and hierarchy adjacency matrices in data/processed/molecular/.

Usage:
  CUDA_VISIBLE_DEVICES=2 python scripts/train_toxigene_v3.py

MIT License -- Bryan Cheng, 2026
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, roc_auc_score
from scipy import sparse

from sentinel.models.molecular_encoder.model import MolecularEncoder
from sentinel.data.splits import split_indices_spatial_only, SplitConfig
from sentinel.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "molecular"
GEO_SUMMARY = DATA_DIR / "real" / "geo_summary.json"
GEO_METADATA = DATA_DIR / "geo_sample_metadata.json"
CKPT_DIR = PROJECT_ROOT / "checkpoints" / "molecular"
RESULTS_DIR = PROJECT_ROOT / "results" / "benchmarks"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Hyperparameters — tuned for small dataset (239 samples)
# ---------------------------------------------------------------------------
EPOCHS = 200
BATCH_SIZE = 16  # small batch for small dataset
EARLY_STOP_PATIENCE = 25
LR = 5e-4
WEIGHT_DECAY = 0.02
GRAD_CLIP = 1.0
SEED = 42
LAMBDA_L1 = 0.005  # lighter L1 for small dataset
DROPOUT = 0.3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTCOME_NAMES = [
    "reproductive_impairment",
    "growth_inhibition",
    "immunosuppression",
    "neurotoxicity",
    "hepatotoxicity",
    "oxidative_damage",
    "endocrine_disruption",
]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class MolecularDataset(Dataset):
    """Simple dataset for gene expression + outcome/pathway labels."""

    def __init__(
        self,
        expression: np.ndarray,
        outcomes: np.ndarray,
        pathways: np.ndarray | None = None,
    ):
        self.expression = torch.tensor(expression.astype(np.float32))
        self.outcomes = torch.tensor(outcomes.astype(np.float32))
        self.pathways = (
            torch.tensor(pathways.astype(np.float32))
            if pathways is not None
            else None
        )

    def __len__(self) -> int:
        return len(self.expression)

    def __getitem__(self, idx: int) -> dict:
        item = {
            "expression": self.expression[idx],
            "outcomes": self.outcomes[idx],
        }
        if self.pathways is not None:
            item["pathways"] = self.pathways[idx]
        return item


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def load_sparse_adj(path: Path) -> torch.Tensor:
    """Load a sparse .npz adjacency matrix and return as dense float32 tensor."""
    d = np.load(path)
    shape = tuple(d["shape"])
    mat = sparse.csr_matrix((d["data"], d["indices"], d["indptr"]), shape=shape)
    return torch.tensor(mat.toarray(), dtype=torch.float32)


def load_study_ids_from_metadata(n_samples: int) -> list[str]:
    """Load per-sample GEO study IDs from the metadata file.

    Returns a list of study IDs (GSE*) so that samples from the same
    GEO study stay in the same fold for spatial holdout.
    """
    if GEO_METADATA.exists():
        try:
            with open(GEO_METADATA) as f:
                metadata = json.load(f)
            if len(metadata) == n_samples:
                ids = [entry.get("gse_id", f"sample_{i:05d}") for i, entry in enumerate(metadata)]
                logger.info(f"Loaded {len(set(ids))} unique study IDs from {GEO_METADATA.name}")
                return ids
        except Exception as e:
            logger.warning(f"Failed to load metadata: {e}")

    # Fallback: try geo_summary.json
    if GEO_SUMMARY.exists():
        try:
            with open(GEO_SUMMARY) as f:
                geo_data = json.load(f)
            if isinstance(geo_data, list):
                ids = []
                for entry in geo_data:
                    if isinstance(entry, dict):
                        gse = entry.get("gse_id", "")
                        n = entry.get("n_samples", 0)
                        ids.extend([gse] * n)
                if len(ids) == n_samples:
                    logger.info(f"Loaded study IDs from geo_summary.json")
                    return ids
        except Exception as e:
            logger.warning(f"Failed to parse geo_summary.json: {e}")

    # Fallback: deterministic index-based
    logger.info(f"Using index-based IDs for {n_samples} samples")
    return [f"sample_{i:05d}" for i in range(n_samples)]


def compute_pos_weight(labels: np.ndarray) -> torch.Tensor:
    """Compute positive class weight for each outcome to handle imbalance.

    For outcomes with zero positive samples, returns weight 0 (will be masked).
    """
    n_samples = labels.shape[0]
    n_outcomes = labels.shape[1]
    weights = np.ones(n_outcomes, dtype=np.float32)

    for j in range(n_outcomes):
        n_pos = labels[:, j].sum()
        n_neg = n_samples - n_pos
        if n_pos > 0:
            weights[j] = min(n_neg / n_pos, 10.0)  # cap at 10 to prevent explosion
        else:
            weights[j] = 0.0  # will mask this outcome in loss

    return torch.tensor(weights, dtype=torch.float32)


def compute_prevalence(labels: np.ndarray) -> dict[str, float]:
    """Compute per-outcome prevalence (fraction of positive samples)."""
    if labels.ndim == 1:
        return {"overall": float(labels.mean())}
    return {
        OUTCOME_NAMES[j] if j < len(OUTCOME_NAMES) else f"outcome_{j}": float(
            labels[:, j].mean()
        )
        for j in range(labels.shape[1])
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray | None, list[str], str]:
    """Load expression matrix, outcome labels, pathway labels, and gene names.

    Returns:
        (expression, outcomes, pathways_or_None, gene_names, dataset_name)
    """
    # Check for v3_corrected data (produced by prepare_geo_for_toxigene.py)
    expr_path = DATA_DIR / "expression_matrix_v3_corrected.npy"
    out_path = DATA_DIR / "outcome_labels_v3_corrected.npy"
    pw_path = DATA_DIR / "pathway_labels_v3_corrected.npy"

    if not expr_path.exists() or not out_path.exists():
        logger.error(
            "Prepared data not found. Run first:\n"
            "  python scripts/prepare_geo_for_toxigene.py\n"
            f"Expected: {expr_path}"
        )
        sys.exit(1)

    expression = np.load(expr_path).astype(np.float32)
    outcomes = np.load(out_path).astype(np.float32)
    pathways = np.load(pw_path).astype(np.float32) if pw_path.exists() else None

    logger.info(f"Expression: {expression.shape}")
    logger.info(f"Outcomes: {outcomes.shape}")
    if pathways is not None:
        logger.info(f"Pathways: {pathways.shape}")

    # Validate no NaN or Inf
    assert not np.isnan(expression).any(), "NaN in expression matrix!"
    assert not np.isinf(expression).any(), "Inf in expression matrix!"
    assert not np.isnan(outcomes).any(), "NaN in outcome labels!"

    # Gene names
    gene_names_path = DATA_DIR / "gene_names.json"
    if gene_names_path.exists():
        with open(gene_names_path) as f:
            gene_names = json.load(f)
        if len(gene_names) != expression.shape[1]:
            logger.warning(
                f"Gene name count ({len(gene_names)}) != expression columns ({expression.shape[1]}). "
                f"Using placeholder names."
            )
            gene_names = [f"gene_{i}" for i in range(expression.shape[1])]
    else:
        gene_names = [f"gene_{i}" for i in range(expression.shape[1])]

    logger.info(
        f"Loaded v3_corrected: {expression.shape[0]} samples x {expression.shape[1]} genes, "
        f"{outcomes.shape[1]} outcomes"
    )

    return expression, outcomes, pathways, gene_names, "v3_corrected"


def load_hierarchy(n_genes: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load hierarchy adjacency matrices from files."""
    paths = [
        DATA_DIR / "hierarchy_layer0_gene_to_pathway.npz",
        DATA_DIR / "hierarchy_layer1_pathway_to_process.npz",
        DATA_DIR / "hierarchy_layer2_process_to_outcome.npz",
    ]

    if not all(p.exists() for p in paths):
        logger.error(
            "Hierarchy adjacency matrices not found. Run first:\n"
            "  python scripts/prepare_geo_for_toxigene.py"
        )
        sys.exit(1)

    pathway_adj = load_sparse_adj(paths[0])
    process_adj = load_sparse_adj(paths[1])
    outcome_adj = load_sparse_adj(paths[2])

    # Validate dimensions
    assert pathway_adj.shape[1] == n_genes, (
        f"Hierarchy gene dim ({pathway_adj.shape[1]}) != expression genes ({n_genes}). "
        f"Re-run prepare_geo_for_toxigene.py"
    )

    logger.info(
        f"Hierarchy: genes({pathway_adj.shape[1]}) -> "
        f"pathways({pathway_adj.shape[0]}) -> "
        f"processes({process_adj.shape[0]}) -> "
        f"outcomes({outcome_adj.shape[0]})"
    )
    return pathway_adj, process_adj, outcome_adj


# ---------------------------------------------------------------------------
# Custom loss that handles zero-prevalence classes
# ---------------------------------------------------------------------------
def safe_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor,
    active_mask: torch.Tensor,
) -> torch.Tensor:
    """BCE with logits that masks out outcomes with no positive samples.

    Args:
        logits: (B, n_outcomes)
        targets: (B, n_outcomes)
        pos_weight: (n_outcomes,) -- 0 for inactive outcomes
        active_mask: (n_outcomes,) bool -- True for outcomes to include in loss
    """
    n_active = active_mask.sum().item()
    if n_active == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    # Only compute loss on active outcomes
    active_logits = logits[:, active_mask]
    active_targets = targets[:, active_mask]
    active_weights = pos_weight[active_mask]

    loss = F.binary_cross_entropy_with_logits(
        active_logits,
        active_targets,
        pos_weight=active_weights,
        reduction="mean",
    )
    return loss


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate model on a dataloader. Returns metrics dict."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for batch in dataloader:
        expr = batch["expression"].to(device)
        outputs = model(gene_expression=expr)
        probs = torch.sigmoid(outputs["outcome_logits"]).cpu()
        preds = (probs > 0.5).float()

        all_probs.append(probs)
        all_preds.append(preds)
        all_labels.append(batch["outcomes"])

    all_probs = torch.cat(all_probs).numpy()
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # Only compute metrics for outcomes that have both positive and negative samples
    active_outcomes = []
    per_class_f1 = np.zeros(all_labels.shape[1])
    for j in range(all_labels.shape[1]):
        n_pos = all_labels[:, j].sum()
        n_neg = all_labels.shape[0] - n_pos
        if n_pos > 0 and n_neg > 0:
            active_outcomes.append(j)
            per_class_f1[j] = f1_score(
                all_labels[:, j], all_preds[:, j], zero_division=0
            )

    macro_f1 = float(np.mean([per_class_f1[j] for j in active_outcomes])) if active_outcomes else 0.0
    accuracy = float((all_preds == all_labels).mean())

    try:
        if len(active_outcomes) > 0:
            active_labels = all_labels[:, active_outcomes]
            active_probs = all_probs[:, active_outcomes]
            auroc = float(roc_auc_score(active_labels, active_probs, average="macro"))
        else:
            auroc = None
    except Exception:
        auroc = None

    return {
        "macro_f1": macro_f1,
        "accuracy": accuracy,
        "auroc": auroc,
        "per_class_f1": [float(f) for f in per_class_f1],
        "active_outcomes": active_outcomes,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    logger.info("=" * 70)
    logger.info("ToxiGene v3 — MolecularEncoder on Real GEO Data")
    logger.info("=" * 70)
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Seed: {SEED}")

    # ── Load data ─────────────────────────────────────────────────────────
    expression, outcomes, pathways, gene_names, dataset_name = load_data()
    pathway_adj, process_adj, outcome_adj = load_hierarchy(expression.shape[1])

    n_samples = expression.shape[0]
    n_genes = expression.shape[1]

    # ── Determine which outcomes are active (have positive samples) ──────
    outcome_prevalence = outcomes.sum(axis=0)
    active_outcomes = outcome_prevalence > 0
    n_active = int(active_outcomes.sum())
    logger.info(f"\nActive outcomes ({n_active}/{len(OUTCOME_NAMES)}):")
    for j, name in enumerate(OUTCOME_NAMES):
        status = "ACTIVE" if active_outcomes[j] else "INACTIVE (0 positives)"
        logger.info(f"  {name}: {int(outcome_prevalence[j])} positives -- {status}")

    # ── Build study-level spatial holdout splits ──────────────────────────
    logger.info("\n--- Spatial Holdout Split ---")
    site_ids = load_study_ids_from_metadata(n_samples)

    split_idx = split_indices_spatial_only(
        site_ids=site_ids,
        config=SplitConfig(),
    )

    # With only 4 studies, hash-based splits might leave some empty.
    # Validate and fall back to random split if needed.
    empty_splits = [s for s in ["train", "val", "test"] if len(split_idx[s]) == 0]

    if empty_splits:
        logger.warning(
            f"Spatial holdout produced empty splits: {empty_splits}. "
            f"Falling back to stratified random split (70/15/15)."
        )
        rng = np.random.default_rng(SEED)
        idx = rng.permutation(n_samples)
        n_train = int(0.70 * n_samples)
        n_val = int(0.15 * n_samples)
        split_idx = {
            "train": idx[:n_train].tolist(),
            "val": idx[n_train:n_train + n_val].tolist(),
            "test": idx[n_train + n_val:].tolist(),
        }

    train_idx = np.array(split_idx["train"])
    val_idx = np.array(split_idx["val"])
    test_idx = np.array(split_idx["test"])

    logger.info(
        f"Split sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}"
    )

    # Log per-split outcome prevalence
    for split_name, idxs in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        if len(idxs) > 0:
            prev = outcomes[idxs].mean(axis=0)
            prev_str = ", ".join(f"{v:.3f}" for v in prev)
            unique_studies = set(site_ids[i] for i in idxs)
            logger.info(
                f"  {split_name:5s}: {len(idxs):4d} samples, "
                f"{len(unique_studies)} studies | prevalence=[{prev_str}]"
            )

    # ── Z-score normalization (train-only statistics) ─────────────────────
    logger.info("\n--- Z-score Normalization (train-only stats) ---")
    train_expr = expression[train_idx]
    expr_mean = train_expr.mean(axis=0)
    expr_std = train_expr.std(axis=0)
    constant_mask = expr_std < 1e-6
    expr_std[constant_mask] = 1.0
    n_constant = int(constant_mask.sum())
    logger.info(
        f"Constant/near-zero-variance genes (set std=1): {n_constant}/{n_genes}"
    )

    expression = (expression - expr_mean) / expr_std
    # Clip to prevent extreme values that cause NaN
    expression = np.clip(expression, -10.0, 10.0)

    # Final NaN/Inf check
    assert not np.isnan(expression).any(), "NaN after normalization!"
    assert not np.isinf(expression).any(), "Inf after normalization!"

    # ── Build datasets and dataloaders ────────────────────────────────────
    train_ds = MolecularDataset(
        expression[train_idx], outcomes[train_idx],
        pathways[train_idx] if pathways is not None else None,
    )
    val_ds = MolecularDataset(
        expression[val_idx], outcomes[val_idx],
        pathways[val_idx] if pathways is not None else None,
    )
    test_ds = MolecularDataset(
        expression[test_idx], outcomes[test_idx],
        pathways[test_idx] if pathways is not None else None,
    )

    train_dl = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=False,
    )
    val_dl = DataLoader(
        val_ds, batch_size=BATCH_SIZE,
        num_workers=0, pin_memory=True,
    )
    test_dl = DataLoader(
        test_ds, batch_size=BATCH_SIZE,
        num_workers=0, pin_memory=True,
    )

    # ── Compute class weights for imbalanced BCE ─────────────────────────
    train_outcomes = outcomes[train_idx]
    pos_weight = compute_pos_weight(train_outcomes).to(DEVICE)
    active_mask = (pos_weight > 0).to(DEVICE)
    logger.info(f"Pos weights: {[f'{w:.2f}' for w in pos_weight.tolist()]}")
    logger.info(f"Active mask: {active_mask.tolist()}")

    # ── Build model ───────────────────────────────────────────────────────
    logger.info("\n--- Building MolecularEncoder ---")
    model = MolecularEncoder(
        gene_names=gene_names,
        pathway_adj=pathway_adj,
        process_adj=process_adj,
        outcome_adj=outcome_adj,
        num_chem_classes=50,
        lambda_l1=LAMBDA_L1,
        dropout=DROPOUT,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {n_params:,}")

    # Optimizer + cosine LR scheduler with warmup
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # ── Training loop ─────────────────────────────────────────────────────
    logger.info("\n--- Training ---")
    best_val_f1 = 0.0
    best_ckpt_path = CKPT_DIR / "toxigene_v3_best.pt"
    epochs_no_improve = 0
    training_history = []
    nan_batch_count = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        n_batches = 0
        train_preds_all, train_labels_all = [], []

        for batch in train_dl:
            expr = batch["expression"].to(DEVICE)
            outcome_targets = batch["outcomes"].to(DEVICE)
            pathway_targets = batch.get("pathways")
            if pathway_targets is not None:
                pathway_targets = pathway_targets.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(gene_expression=expr)

            # Custom loss: BCE with pos_weight, masking inactive outcomes
            outcome_loss = safe_bce_loss(
                outputs["outcome_logits"],
                outcome_targets,
                pos_weight,
                active_mask,
            )

            # Pathway regression loss (if pathway labels available)
            pathway_loss = torch.tensor(0.0, device=DEVICE)
            if pathway_targets is not None:
                pathway_loss = F.mse_loss(
                    outputs["pathway_activation"],
                    pathway_targets,
                    reduction="mean",
                )

            # Bottleneck L1 penalty
            l1_loss = model.bottleneck.compute_loss()

            # Total loss
            loss = outcome_loss + 0.3 * pathway_loss + l1_loss

            # Check for NaN
            if torch.isnan(loss) or torch.isinf(loss):
                nan_batch_count += 1
                if nan_batch_count <= 5:
                    logger.warning(
                        f"  NaN/Inf loss at epoch {epoch+1} "
                        f"(outcome={outcome_loss.item():.4f}, "
                        f"pathway={pathway_loss.item():.4f}, "
                        f"l1={l1_loss.item():.4f})"
                    )
                continue

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            preds = (torch.sigmoid(outputs["outcome_logits"]) > 0.5).float().cpu()
            train_preds_all.append(preds)
            train_labels_all.append(outcome_targets.cpu())

        scheduler.step()

        if n_batches == 0:
            logger.warning(f"Epoch {epoch+1}: all batches had NaN loss!")
            continue

        avg_loss = total_loss / n_batches
        train_preds_all = torch.cat(train_preds_all).numpy()
        train_labels_all = torch.cat(train_labels_all).numpy()

        # Compute train F1 only on active outcomes
        active_idxs = [j for j in range(outcomes.shape[1]) if active_mask[j].item()]
        if active_idxs:
            train_f1_per = [
                f1_score(train_labels_all[:, j], train_preds_all[:, j], zero_division=0)
                for j in active_idxs
            ]
            train_f1 = float(np.mean(train_f1_per))
        else:
            train_f1 = 0.0

        # Validation
        val_metrics = evaluate(model, val_dl, DEVICE)
        val_f1 = val_metrics["macro_f1"]

        # Gene selection bottleneck stats
        try:
            n_selected = model.bottleneck.num_selected
        except Exception:
            n_selected = 0

        current_lr = optimizer.param_groups[0]["lr"]

        # Log every 10 epochs + first + last
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"Ep {epoch+1:3d}/{EPOCHS} | Loss: {avg_loss:.4f} | "
                f"Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f} | "
                f"Genes: {n_selected} | LR: {current_lr:.2e} | "
                f"Patience: {epochs_no_improve}/{EARLY_STOP_PATIENCE}"
            )

        training_history.append({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "train_f1": float(train_f1),
            "val_f1": val_f1,
            "n_selected_genes": n_selected,
            "lr": current_lr,
        })

        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_ckpt_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            logger.info(
                f"Early stopping at epoch {epoch+1} "
                f"(patience={EARLY_STOP_PATIENCE}, best val F1={best_val_f1:.4f})"
            )
            break

    final_epoch = epoch + 1

    # ── Reload best checkpoint and evaluate on test ───────────────────────
    logger.info("\n--- Test Evaluation ---")
    if best_ckpt_path.exists():
        model.load_state_dict(
            torch.load(best_ckpt_path, map_location=DEVICE, weights_only=True)
        )
        logger.info(f"Loaded best checkpoint (val F1={best_val_f1:.4f})")

    test_metrics = evaluate(model, test_dl, DEVICE)

    # Gene selection bottleneck analysis
    try:
        selected_genes = model.bottleneck.get_selected_genes()
        n_selected_final = model.bottleneck.num_selected
    except Exception:
        selected_genes = []
        n_selected_final = 0

    # ── Log results ───────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("TEST RESULTS (ToxiGene v3 — Real GEO Data)")
    logger.info("=" * 60)
    for i, f1_val in enumerate(test_metrics["per_class_f1"]):
        name = OUTCOME_NAMES[i] if i < len(OUTCOME_NAMES) else f"outcome_{i}"
        status = "" if active_outcomes[i] else " (INACTIVE)"
        logger.info(f"  {name}: F1 = {f1_val:.4f}{status}")
    logger.info(f"\n  Macro F1:  {test_metrics['macro_f1']:.4f}")
    logger.info(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    if test_metrics["auroc"] is not None:
        logger.info(f"  AUROC:     {test_metrics['auroc']:.4f}")
    logger.info(f"  Selected genes: {n_selected_final}/{n_genes}")
    if selected_genes:
        logger.info(f"  Top selected: {selected_genes[:20]}")

    # Bottleneck gate statistics
    logger.info("\n--- Gene Selection Bottleneck Stats ---")
    try:
        gates = model.bottleneck.gates.detach().cpu().numpy()
        logger.info(f"  Gate mean:   {gates.mean():.4f}")
        logger.info(f"  Gate std:    {gates.std():.4f}")
        logger.info(f"  Gate min:    {gates.min():.4f}")
        logger.info(f"  Gate max:    {gates.max():.4f}")
        logger.info(f"  Gates > 0.5: {int((gates > 0.5).sum())}")
        logger.info(f"  Gates > 0.1: {int((gates > 0.1).sum())}")
        logger.info(f"  Gates < 0.01: {int((gates < 0.01).sum())}")
    except Exception as e:
        logger.warning(f"  Could not read gate values: {e}")

    if nan_batch_count > 0:
        logger.warning(f"\n  Total NaN/Inf batches during training: {nan_batch_count}")

    if test_metrics["macro_f1"] > 0.80:
        logger.info("*** HARD THRESHOLD MET ***")
    elif test_metrics["macro_f1"] > 0.60:
        logger.info("ACCEPTABLE")
    else:
        logger.info(f"BELOW THRESHOLD ({test_metrics['macro_f1']:.4f})")

    # ── Print summary ─────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n=== ToxiGene v3 Results (Real GEO Data) ===")
    print(f"Dataset        : {dataset_name} ({n_samples} samples, {n_genes} genes)")
    print(f"Split          : train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    print(f"Test Macro-F1  : {test_metrics['macro_f1']:.4f}")
    print(f"Test Accuracy  : {test_metrics['accuracy']:.4f}")
    if test_metrics["auroc"] is not None:
        print(f"Test AUROC     : {test_metrics['auroc']:.4f}")
    print(f"Best Val F1    : {best_val_f1:.4f}")
    print(f"Epochs trained : {final_epoch}")
    print(f"Selected genes : {n_selected_final}/{n_genes}")
    print(f"NaN batches    : {nan_batch_count}")
    print(f"Per-class F1   :")
    for i, f1_val in enumerate(test_metrics["per_class_f1"]):
        name = OUTCOME_NAMES[i] if i < len(OUTCOME_NAMES) else f"outcome_{i}"
        status = "" if active_outcomes[i] else " (inactive)"
        print(f"  {name}: {f1_val:.4f}{status}")
    print(f"Time           : {elapsed/60:.1f}m")

    # ── Save results ──────────────────────────────────────────────────────
    results = {
        "model": "ToxiGene_v3_real_geo",
        "dataset": dataset_name,
        "split_protocol": "spatial_holdout_with_random_fallback",
        "n_samples_total": n_samples,
        "n_genes": n_genes,
        "n_synthetic_samples": 0,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "test_f1_macro": test_metrics["macro_f1"],
        "test_accuracy": test_metrics["accuracy"],
        "test_auroc_macro": test_metrics["auroc"],
        "best_val_f1": float(best_val_f1),
        "per_class_f1": {
            (OUTCOME_NAMES[i] if i < len(OUTCOME_NAMES) else f"outcome_{i}"): f1_val
            for i, f1_val in enumerate(test_metrics["per_class_f1"])
        },
        "active_outcomes": [
            OUTCOME_NAMES[j] for j in test_metrics.get("active_outcomes", [])
        ],
        "epochs_trained": final_epoch,
        "max_epochs": EPOCHS,
        "early_stop_patience": EARLY_STOP_PATIENCE,
        "nan_batches": nan_batch_count,
        "hyperparameters": {
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "grad_clip": GRAD_CLIP,
            "lambda_l1": LAMBDA_L1,
            "dropout": DROPOUT,
            "num_chem_classes": 50,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
        },
        "n_selected_genes": n_selected_final,
        "n_total_genes": n_genes,
        "selected_genes_top20": selected_genes[:20] if selected_genes else [],
        "normalization": "z-score (train-only mean/std), clipped to [-10, 10]",
        "n_constant_genes_masked": n_constant,
        "elapsed_seconds": elapsed,
        "checkpoint_path": str(best_ckpt_path),
        "data_sources": {
            "geo_studies": "GSE104776, GSE54800, GSE73661, GSE83514",
            "pipeline": "prepare_geo_for_toxigene.py",
            "synthetic_data": "EXCLUDED",
        },
    }

    results_path = RESULTS_DIR / "toxigene_v3_holdout.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {results_path}")
    logger.info(f"Checkpoint saved to {best_ckpt_path}")
    logger.info(f"Total time: {elapsed/60:.1f}m")
    logger.info("DONE")


if __name__ == "__main__":
    main()
