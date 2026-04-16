#!/usr/bin/env python3
"""Benchmark ToxiGene (full-real) against classical ML baselines.

Baselines:
  - Random Forest
  - XGBoost / Gradient Boosted Trees
  - Logistic Regression (one-vs-rest)
  - Simple MLP (no pathway hierarchy)
  - PCA + Logistic Regression

All baselines use the same 70/15/15 train/val/test split (seed=42) as training.
Evaluates on the held-out test set with macro F1, accuracy, and per-class F1.

Results saved to results/benchmarks/toxigene_fullreal_benchmark.json.

MIT License — Bryan Cheng, 2026
"""

import json
import sys
import time
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy import sparse

from sentinel.models.molecular_encoder.model import MolecularEncoder
from sentinel.utils.logging import get_logger

warnings.filterwarnings("ignore")

logger = get_logger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR  = Path("data/processed/molecular")
CKPT_DIR  = Path("checkpoints/molecular")
RESULTS_DIR = Path("results/benchmarks")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED       = 42
BATCH_SIZE = 32

OUTCOME_NAMES = [
    "reproductive_impairment", "growth_inhibition", "immunosuppression",
    "neurotoxicity", "hepatotoxicity", "oxidative_damage", "endocrine_disruption"
]


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
        item = {"expression": self.expression[idx], "outcomes": self.outcomes[idx]}
        if self.pathways is not None:
            item["pathways"] = self.pathways[idx]
        return item


def load_sparse_adj(path):
    d = np.load(path)
    shape = tuple(d["shape"])
    mat = sparse.csr_matrix((d["data"], d["indices"], d["indptr"]), shape=shape)
    return torch.tensor(mat.toarray(), dtype=torch.float32)


def metrics(y_true, y_pred, y_prob=None, name=""):
    """Compute macro F1, accuracy, and per-class F1."""
    f1   = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    acc  = float(accuracy_score(y_true.ravel(), y_pred.ravel()))
    pcf1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    auroc = None
    if y_prob is not None:
        try:
            auroc = float(roc_auc_score(y_true, y_prob, average="macro"))
        except Exception:
            pass
    result = {
        "f1_macro":     f1,
        "accuracy":     acc,
        "auroc_macro":  auroc,
        "per_class_f1": {n: float(v) for n, v in zip(OUTCOME_NAMES, pcf1)},
    }
    logger.info(f"  [{name}] F1={f1:.4f}  Acc={acc:.4f}"
                + (f"  AUROC={auroc:.4f}" if auroc is not None else ""))
    return result


# ── Simple MLP (no pathway hierarchy) ────────────────────────────────────────

class SimpleMLP(nn.Module):
    def __init__(self, n_genes, n_outcomes, hidden=512, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_genes, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_outcomes),
        )

    def forward(self, x):
        return self.net(x)


def train_simple_mlp(X_tr, y_tr, X_va, y_va, X_te, y_te,
                     n_outcomes, epochs=100, patience=15):
    """Train a simple MLP with early stopping; return test predictions."""
    n_genes = X_tr.shape[1]
    model = SimpleMLP(n_genes, n_outcomes).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    X_va_t = torch.tensor(X_va, dtype=torch.float32)
    y_va_t = torch.tensor(y_va, dtype=torch.float32)
    X_te_t = torch.tensor(X_te, dtype=torch.float32).to(DEVICE)

    # Mini-batch training
    tr_ds  = torch.utils.data.TensorDataset(X_tr_t, y_tr_t)
    tr_dl  = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True)

    best_val = 0.0
    best_sd  = None
    no_imp   = 0

    for ep in range(epochs):
        model.train()
        for xb, yb in tr_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss   = nn.functional.binary_cross_entropy_with_logits(logits, yb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            va_logits = model(X_va_t.to(DEVICE))
            va_preds  = (torch.sigmoid(va_logits) > 0.5).float().cpu().numpy()
        val_f1 = f1_score(y_va, va_preds, average="macro", zero_division=0)

        if val_f1 > best_val:
            best_val = val_f1
            best_sd  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_imp   = 0
        else:
            no_imp += 1
        if no_imp >= patience:
            break

    if best_sd is not None:
        model.load_state_dict(best_sd)
    model.eval()
    with torch.no_grad():
        te_probs = torch.sigmoid(model(X_te_t)).cpu().numpy()
    te_preds = (te_probs > 0.5).astype(float)
    return te_preds, te_probs


# ── Sentinel ToxiGene evaluation ─────────────────────────────────────────────

def eval_toxigene(te_dl, gene_names, pathway_adj, process_adj, outcome_adj):
    """Load the fullreal checkpoint and evaluate on test loader."""
    ckpt = CKPT_DIR / "toxigene_fullreal_best.pt"
    if not ckpt.exists():
        logger.warning("toxigene_fullreal_best.pt not found. Trying toxigene_expanded_best.pt.")
        ckpt = CKPT_DIR / "toxigene_expanded_best.pt"
    if not ckpt.exists():
        logger.warning("No fullreal checkpoint found. Trying toxigene_best.pt.")
        ckpt = CKPT_DIR / "toxigene_best.pt"
    if not ckpt.exists():
        logger.error("No ToxiGene checkpoint found.")
        return None

    model = MolecularEncoder(
        gene_names=gene_names,
        pathway_adj=pathway_adj,
        process_adj=process_adj,
        outcome_adj=outcome_adj,
        num_chem_classes=50,
        lambda_l1=0.01,
        dropout=0.2,
    ).to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
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

    return te_preds, te_labels, te_probs, str(ckpt.name)


def main():
    t0 = time.time()

    # ── Load data ──────────────────────────────────────────────────────────
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

    # Normalize
    expr_mean = expression.mean(axis=0)
    expr_std  = expression.std(axis=0)
    expr_std[expr_std < 1e-6] = 1.0
    expression_norm = (expression - expr_mean) / expr_std

    # Reproducible split (same seed as training)
    n  = len(expression_norm)
    n_tr = int(0.70 * n)
    n_va = int(0.15 * n)
    n_te = n - n_tr - n_va

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(n)

    # For classical sklearn models use numpy arrays
    # We need the same split as torch random_split(seed=42)
    # torch random_split with manual_seed=42 uses a specific permutation;
    # reproduce it here:
    g = torch.Generator().manual_seed(SEED)
    ds_full = MolecularDataset(expression_norm, outcomes, pathways)
    tr_ds, va_ds, te_ds = random_split(ds_full, [n_tr, n_va, n_te], generator=g)

    te_dl = DataLoader(te_ds, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)

    # Extract numpy arrays from splits
    def ds_to_numpy(ds_subset):
        xs, ys = [], []
        for item in ds_subset:
            xs.append(item["expression"].numpy())
            ys.append(item["outcomes"].numpy())
        return np.array(xs), np.array(ys)

    logger.info("Extracting split arrays for sklearn baselines...")
    X_tr, y_tr = ds_to_numpy(tr_ds)
    X_va, y_va = ds_to_numpy(va_ds)
    X_te, y_te = ds_to_numpy(te_ds)

    logger.info(f"Split: {len(X_tr)} train / {len(X_va)} val / {len(X_te)} test")

    results = {
        "dataset":           "expression_matrix_fullreal.npy",
        "n_real_samples":    int(n),
        "n_train":           n_tr,
        "n_val":             n_va,
        "n_test":            n_te,
        "seed":              SEED,
        "outcome_names":     OUTCOME_NAMES,
        "models":            {},
    }

    # ── 1. SENTINEL ToxiGene (fullreal checkpoint) ────────────────────────
    logger.info("Evaluating SENTINEL ToxiGene...")
    ret = eval_toxigene(te_dl, gene_names, pathway_adj, process_adj, outcome_adj)
    if ret is not None:
        te_preds, te_labels, te_probs, ckpt_name = ret
        results["models"]["SENTINEL_ToxiGene"] = metrics(
            te_labels, te_preds, te_probs, "SENTINEL_ToxiGene"
        )
        results["models"]["SENTINEL_ToxiGene"]["checkpoint"] = ckpt_name

    # ── 2. Random Forest ──────────────────────────────────────────────────
    logger.info("Training Random Forest...")
    t1 = time.time()
    rf = OneVsRestClassifier(
        RandomForestClassifier(
            n_estimators=200, max_depth=None, min_samples_leaf=1,
            n_jobs=-1, random_state=SEED
        )
    )
    rf.fit(X_tr, y_tr)
    rf_pred = rf.predict(X_te)
    try:
        rf_prob = rf.predict_proba(X_te)
    except Exception:
        rf_prob = None
    results["models"]["RandomForest"] = metrics(y_te, rf_pred, rf_prob, "RandomForest")
    results["models"]["RandomForest"]["fit_time_s"] = round(time.time() - t1, 2)

    # ── 3. XGBoost / Gradient Boosted Trees ──────────────────────────────
    logger.info("Training XGBoost (GBT)...")
    t1 = time.time()
    try:
        from xgboost import XGBClassifier
        xgb_model = OneVsRestClassifier(
            XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                use_label_encoder=False, eval_metric="logloss",
                n_jobs=-1, random_state=SEED, verbosity=0
            )
        )
        xgb_model.fit(X_tr, y_tr)
        xgb_pred = xgb_model.predict(X_te)
        try:
            xgb_prob = xgb_model.predict_proba(X_te)
        except Exception:
            xgb_prob = None
        results["models"]["XGBoost"] = metrics(y_te, xgb_pred, xgb_prob, "XGBoost")
        results["models"]["XGBoost"]["model"] = "XGBoost(GBT)"
    except ImportError:
        logger.warning("XGBoost not installed. Falling back to sklearn GradientBoosting.")
        gbm = OneVsRestClassifier(
            GradientBoostingClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                subsample=0.8, random_state=SEED
            )
        )
        gbm.fit(X_tr, y_tr)
        gbm_pred = gbm.predict(X_te)
        results["models"]["XGBoost"] = metrics(y_te, gbm_pred, name="GBT(sklearn)")
        results["models"]["XGBoost"]["model"] = "GradientBoosting(sklearn)"
    results["models"]["XGBoost"]["fit_time_s"] = round(time.time() - t1, 2)

    # ── 4. Logistic Regression (one-vs-rest) ─────────────────────────────
    logger.info("Training Logistic Regression...")
    t1 = time.time()
    lr_model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    OneVsRestClassifier(
            LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs",
                               random_state=SEED, n_jobs=-1)
        )),
    ])
    lr_model.fit(X_tr, y_tr)
    lr_pred = lr_model.predict(X_te)
    try:
        lr_prob = lr_model.predict_proba(X_te)
    except Exception:
        lr_prob = None
    results["models"]["LogisticRegression"] = metrics(y_te, lr_pred, lr_prob, "LogisticRegression")
    results["models"]["LogisticRegression"]["fit_time_s"] = round(time.time() - t1, 2)

    # ── 5. Simple MLP (no pathway hierarchy) ─────────────────────────────
    logger.info("Training Simple MLP...")
    t1 = time.time()
    mlp_pred, mlp_prob = train_simple_mlp(
        X_tr, y_tr, X_va, y_va, X_te, y_te,
        n_outcomes=outcomes.shape[1], epochs=100, patience=15
    )
    results["models"]["SimpleMLP"] = metrics(y_te, mlp_pred, mlp_prob, "SimpleMLP")
    results["models"]["SimpleMLP"]["fit_time_s"] = round(time.time() - t1, 2)
    results["models"]["SimpleMLP"]["note"] = "2-layer MLP, no pathway hierarchy"

    # ── 6. PCA + Logistic Regression ─────────────────────────────────────
    logger.info("Training PCA + LR...")
    t1 = time.time()
    pca_lr = Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=50, random_state=SEED)),
        ("clf",    OneVsRestClassifier(
            LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs",
                               random_state=SEED, n_jobs=-1)
        )),
    ])
    pca_lr.fit(X_tr, y_tr)
    pca_lr_pred = pca_lr.predict(X_te)
    try:
        pca_lr_prob = pca_lr.predict_proba(X_te)
    except Exception:
        pca_lr_prob = None
    results["models"]["PCA_LR"] = metrics(y_te, pca_lr_pred, pca_lr_prob, "PCA_LR")
    results["models"]["PCA_LR"]["fit_time_s"] = round(time.time() - t1, 2)
    results["models"]["PCA_LR"]["note"] = "PCA(50 components) + LogisticRegression"

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    results["benchmark_elapsed_s"] = round(elapsed, 2)

    out_path = RESULTS_DIR / "toxigene_fullreal_benchmark.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("TOXIGENE FULL-REAL BENCHMARK RESULTS")
    print("=" * 70)
    print(f"{'Model':<25} {'F1 (macro)':>10} {'Accuracy':>10} {'AUROC':>8}")
    print("-" * 60)
    for name, res in results["models"].items():
        auroc_str = f"{res['auroc_macro']:.4f}" if res.get("auroc_macro") else "  N/A  "
        print(f"{name:<25} {res['f1_macro']:>10.4f} {res['accuracy']:>10.4f} {auroc_str:>8}")
    print("=" * 70)
    print(f"Benchmark saved to: {out_path}")
    print(f"Total time: {elapsed/60:.1f}m")


if __name__ == "__main__":
    main()
