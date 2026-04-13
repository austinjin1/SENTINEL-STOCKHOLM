#!/usr/bin/env python3
"""Benchmark ToxiGene against classical ML and simple deep learning baselines.

Uses the SAME test split (seed=42, 70/15/15) as train_toxigene_expanded.py.
Baselines: Random Forest, Logistic Regression, XGBoost/GBT, Simple MLP, PCA+LR.

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
from sklearn.metrics import f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.decomposition import PCA
from scipy import sparse

from sentinel.models.molecular_encoder.model import MolecularEncoder
from sentinel.utils.logging import get_logger

logger = get_logger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT     = Path("checkpoints/molecular")
DATA_DIR = Path("data/processed/molecular")
RESULTS  = Path("results/benchmarks")
RESULTS.mkdir(parents=True, exist_ok=True)

BATCH_SIZE   = 32
MLP_EPOCHS   = 80
MLP_LR       = 1e-3


# ── Dataset (same as training scripts) ────────────────────────────────────────
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


# ── Simple MLP baseline (no pathway constraints) ──────────────────────────────
class SimpleMLP(nn.Module):
    def __init__(self, in_dim=1000, hidden1=512, hidden2=256, out_dim=7, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,   hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def train_simple_mlp(X_tr, y_tr, X_va, y_va, n_outcomes, device):
    """Train a 3-layer MLP for 80 epochs and return val-best model."""
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    X_va_t = torch.tensor(X_va, dtype=torch.float32).to(device)
    y_va_t = torch.tensor(y_va, dtype=torch.float32)

    ds_tr = torch.utils.data.TensorDataset(X_tr_t, y_tr_t)
    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True)

    model = SimpleMLP(in_dim=X_tr.shape[1], out_dim=n_outcomes).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=MLP_LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MLP_EPOCHS)
    crit  = nn.BCEWithLogitsLoss()

    best_f1, best_state = 0.0, None

    for epoch in range(MLP_EPOCHS):
        model.train()
        for Xb, yb in dl_tr:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(Xb), yb)
            loss.backward()
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            logits = model(X_va_t)
            preds  = (torch.sigmoid(logits).cpu() > 0.5).float().numpy()
        val_f1 = f1_score(y_va, preds, average="macro", zero_division=0)
        if val_f1 > best_f1:
            best_f1   = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def eval_mlp(model, X_te, y_te, device):
    model.eval()
    X_t = torch.tensor(X_te, dtype=torch.float32).to(device)
    with torch.no_grad():
        probs = torch.sigmoid(model(X_t)).cpu().numpy()
    preds = (probs > 0.5).astype(float)
    f1  = float(f1_score(y_te, preds, average="macro", zero_division=0))
    acc = float(accuracy_score(y_te.argmax(1), preds.argmax(1)))
    return f1, acc


def multi_label_predict(clf, X):
    """Return binary predictions from a MultiOutputClassifier."""
    preds = np.column_stack([e.predict(X) for e in clf.estimators_])
    return preds


def eval_sklearn(clf, X_te, y_te):
    preds = clf.predict(X_te)
    f1  = float(f1_score(y_te, preds, average="macro", zero_division=0))
    acc = float(accuracy_score(y_te.argmax(1), np.array(preds).argmax(1)))
    return f1, acc


def main():
    print("Loading expanded data and building same splits as training...")

    # Load data (same as train_toxigene_expanded.py)
    expression = np.load(DATA_DIR / "expression_matrix_v2.npy")
    outcomes   = np.load(DATA_DIR / "outcome_labels_v2.npy")
    pathways   = np.load(DATA_DIR / "pathway_labels_v2.npy")
    gene_names = json.load(open(DATA_DIR / "gene_names.json"))

    # Normalize (same as training)
    expr_mean = expression.mean(axis=0)
    expr_std  = expression.std(axis=0)
    expr_std[expr_std < 1e-6] = 1.0
    expression = (expression - expr_mean) / expr_std

    n = len(expression)
    n_tr = int(0.70 * n)
    n_va = int(0.15 * n)
    n_te = n - n_tr - n_va

    # Replicate the same random_split indices
    gen = torch.Generator().manual_seed(42)
    ds  = MolecularDataset(expression, outcomes, pathways)
    tr_ds, va_ds, te_ds = random_split(ds, [n_tr, n_va, n_te], generator=gen)

    tr_idx = list(tr_ds.indices)
    va_idx = list(va_ds.indices)
    te_idx = list(te_ds.indices)

    X_tr, y_tr = expression[tr_idx], outcomes[tr_idx]
    X_va, y_va = expression[va_idx], outcomes[va_idx]
    X_te, y_te = expression[te_idx], outcomes[te_idx]

    n_outcomes = outcomes.shape[1]
    print(f"Train: {len(X_tr)}, Val: {len(X_va)}, Test: {len(X_te)}")

    results = {}

    # ── 1. SENTINEL ToxiGene (load best expanded checkpoint) ───────────────
    print("\n[1/6] Loading SENTINEL ToxiGene expanded checkpoint...")
    pathway_adj = load_sparse_adj(DATA_DIR / "hierarchy_layer0_gene_to_pathway.npz")
    process_adj = load_sparse_adj(DATA_DIR / "hierarchy_layer1_pathway_to_process.npz")
    outcome_adj = load_sparse_adj(DATA_DIR / "hierarchy_layer2_process_to_outcome.npz")

    toxigene = MolecularEncoder(
        gene_names=gene_names,
        pathway_adj=pathway_adj,
        process_adj=process_adj,
        outcome_adj=outcome_adj,
        num_chem_classes=50,
        lambda_l1=0.01,
        dropout=0.2,
    ).to(DEVICE)
    ckpt_path = CKPT / "toxigene_expanded_best.pt"
    toxigene.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
    toxigene.eval()

    te_dl = DataLoader(te_ds, batch_size=BATCH_SIZE, num_workers=2)
    tg_preds, tg_labels = [], []
    with torch.no_grad():
        for batch in te_dl:
            expr    = batch["expression"].to(DEVICE)
            outputs = toxigene(gene_expression=expr)
            preds   = (torch.sigmoid(outputs["outcome_logits"]) > 0.5).float().cpu()
            tg_preds.append(preds)
            tg_labels.append(batch["outcomes"])

    tg_preds  = torch.cat(tg_preds).numpy()
    tg_labels = torch.cat(tg_labels).numpy()
    tg_f1  = float(f1_score(tg_labels, tg_preds, average="macro", zero_division=0))
    tg_acc = float((tg_preds == tg_labels).mean())
    results["SENTINEL_ToxiGene"] = {"f1_macro": tg_f1, "accuracy": tg_acc}
    print(f"    SENTINEL ToxiGene  — F1={tg_f1:.4f}  Acc={tg_acc:.4f}")

    # ── 2. Simple MLP (no pathway constraints) ──────────────────────────────
    print("\n[2/6] Training Simple MLP...")
    t1 = time.time()
    mlp = train_simple_mlp(X_tr, y_tr, X_va, y_va, n_outcomes, DEVICE)
    mlp_f1, mlp_acc = eval_mlp(mlp, X_te, y_te, DEVICE)
    results["SimpleMLP"] = {"f1_macro": mlp_f1, "accuracy": mlp_acc}
    print(f"    Simple MLP         — F1={mlp_f1:.4f}  Acc={mlp_acc:.4f}  ({time.time()-t1:.1f}s)")

    # ── 3. Random Forest ────────────────────────────────────────────────────
    print("\n[3/6] Training Random Forest...")
    t1 = time.time()
    rf = MultiOutputClassifier(
        RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        n_jobs=-1,
    )
    rf.fit(X_tr, y_tr)
    rf_f1, rf_acc = eval_sklearn(rf, X_te, y_te)
    results["RandomForest"] = {"f1_macro": rf_f1, "accuracy": rf_acc}
    print(f"    Random Forest      — F1={rf_f1:.4f}  Acc={rf_acc:.4f}  ({time.time()-t1:.1f}s)")

    # ── 4. XGBoost (fall back to GradientBoosting if not available) ─────────
    print("\n[4/6] Training XGBoost / GBT...")
    t1 = time.time()
    try:
        import xgboost as xgb
        base_xgb = xgb.XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, n_jobs=-1, tree_method="hist",
        )
        xgb_clf = MultiOutputClassifier(base_xgb, n_jobs=-1)
        model_name_xgb = "XGBoost"
    except ImportError:
        print("    xgboost not found — using GradientBoostingClassifier")
        base_xgb = GradientBoostingClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
        )
        xgb_clf = MultiOutputClassifier(base_xgb, n_jobs=-1)
        model_name_xgb = "XGBoost(GBT)"

    xgb_clf.fit(X_tr, y_tr)
    xgb_f1, xgb_acc = eval_sklearn(xgb_clf, X_te, y_te)
    results["XGBoost"] = {"f1_macro": xgb_f1, "accuracy": xgb_acc, "model": model_name_xgb}
    print(f"    {model_name_xgb:<18s} — F1={xgb_f1:.4f}  Acc={xgb_acc:.4f}  ({time.time()-t1:.1f}s)")

    # ── 5. Logistic Regression ──────────────────────────────────────────────
    print("\n[5/6] Training Logistic Regression...")
    t1 = time.time()
    lr_clf = MultiOutputClassifier(
        LogisticRegression(max_iter=1000, C=0.1, random_state=42, n_jobs=-1),
        n_jobs=-1,
    )
    lr_clf.fit(X_tr, y_tr)
    lr_f1, lr_acc = eval_sklearn(lr_clf, X_te, y_te)
    results["LogisticRegression"] = {"f1_macro": lr_f1, "accuracy": lr_acc}
    print(f"    Logistic Regression— F1={lr_f1:.4f}  Acc={lr_acc:.4f}  ({time.time()-t1:.1f}s)")

    # ── 6. PCA + LR ─────────────────────────────────────────────────────────
    print("\n[6/6] Training PCA + LR...")
    t1 = time.time()
    pca = PCA(n_components=50, random_state=42)
    X_tr_pca = pca.fit_transform(X_tr)
    X_te_pca = pca.transform(X_te)
    pca_lr_clf = MultiOutputClassifier(
        LogisticRegression(max_iter=1000, C=0.1, random_state=42, n_jobs=-1),
        n_jobs=-1,
    )
    pca_lr_clf.fit(X_tr_pca, y_tr)
    pca_preds = pca_lr_clf.predict(X_te_pca)
    pca_f1  = float(f1_score(y_te, pca_preds, average="macro", zero_division=0))
    pca_acc = float(accuracy_score(y_te.argmax(1), np.array(pca_preds).argmax(1)))
    results["PCA_LR"] = {"f1_macro": pca_f1, "accuracy": pca_acc}
    print(f"    PCA + LR           — F1={pca_f1:.4f}  Acc={pca_acc:.4f}  ({time.time()-t1:.1f}s)")

    # ── Save results ────────────────────────────────────────────────────────
    out_path = RESULTS / "toxigene_benchmark.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nBenchmark saved to {out_path}")

    # ── Final report ─────────────────────────────────────────────────────────
    print()
    print("=== ToxiGene Data Expansion + Benchmark Results ===")
    print(f"Data: 1000 → 1800 samples")
    print(f"{'Model':<28} {'Macro-F1':>10} {'Accuracy':>10}")
    print("-" * 52)

    def row(name, r):
        f1  = r.get("f1_macro", float("nan"))
        acc = r.get("accuracy", float("nan"))
        return f"{name:<28} {f1:>10.3f} {acc:>10.3f}"

    print(row("SENTINEL ToxiGene",      results["SENTINEL_ToxiGene"]) + "  <- retrained")
    print(row("Simple MLP (no pathway)", results["SimpleMLP"]))
    print(row("Random Forest",           results["RandomForest"]))
    print(row(model_name_xgb,            results["XGBoost"]))
    print(row("Logistic Regression",     results["LogisticRegression"]))
    print(row("PCA + LR",                results["PCA_LR"]))

    # Insight gaps
    gap_mlp = results["SENTINEL_ToxiGene"]["f1_macro"] - results["SimpleMLP"]["f1_macro"]
    gap_rf  = results["SENTINEL_ToxiGene"]["f1_macro"] - results["RandomForest"]["f1_macro"]
    gap_lr  = results["SENTINEL_ToxiGene"]["f1_macro"] - results["LogisticRegression"]["f1_macro"]
    print()
    print(f"Pathway constraint benefit (ToxiGene vs SimpleMLP): ΔF1 = {gap_mlp:+.3f}")
    print(f"Deep learning benefit (ToxiGene vs RandomForest):   ΔF1 = {gap_rf:+.3f}")
    print(f"Deep learning benefit (ToxiGene vs LogReg):         ΔF1 = {gap_lr:+.3f}")


if __name__ == "__main__":
    main()
