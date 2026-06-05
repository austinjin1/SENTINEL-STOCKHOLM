#!/usr/bin/env python3
"""benchmark_toxigene_ml_v3.py — ML baselines on SAME split as ToxiGene v8.

Runs RandomForest, XGBoost, LogisticRegression, PCA+LR, and ExtraTrees
on the 1750/375/375 split (seed=42, expression_matrix_v3_expanded.npy).

Uses the v3 expanded dataset (2500 samples: 1697 v2 + 708 new real GEO
samples from 16 datasets + 95 class-conditional augmented).

Feature selection: top 5000 by max t-statistic to keep RF/XGB tractable.

Bryan Cheng, SENTINEL project, 2026
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not available, skipping")

PROJECT_ROOT = Path("/home/bcheng/SENTINEL")
DATA_DIR     = PROJECT_ROOT / "data" / "processed" / "molecular"
RESULTS_PATH = PROJECT_ROOT / "checkpoints" / "molecular" / "results_ml_baselines_v3.json"

SEED    = 42
N_TRAIN = 1778
N_VAL   = 381
N_TEST  = 381
N_GENES = 5000   # same discriminative gene selection as v6

OUTCOME_NAMES = [
    "reproductive_impairment",
    "growth_inhibition",
    "immunosuppression",
    "neurotoxicity",
    "hepatotoxicity",
    "oxidative_damage",
    "endocrine_disruption",
]


def select_discriminative_genes(X_tr, y_tr, n=N_GENES):
    n_classes = y_tr.shape[1]
    max_t = np.zeros(X_tr.shape[1], dtype=np.float32)
    for c in range(n_classes):
        pos = y_tr[:, c] > 0.5
        neg = ~pos
        if pos.sum() < 5 or neg.sum() < 5:
            continue
        pm = X_tr[pos].mean(0); nm = X_tr[neg].mean(0)
        pv = X_tr[pos].var(0) + 1e-8; nv = X_tr[neg].var(0) + 1e-8
        t  = np.abs(pm - nm) / np.sqrt(pv / pos.sum() + nv / neg.sum())
        max_t = np.maximum(max_t, t.astype(np.float32))
    return np.argsort(max_t)[-n:]


def optimize_thresholds(probs, labels):
    grid = np.linspace(0.2, 0.8, 25)
    thresholds = np.zeros(labels.shape[1])
    for c in range(labels.shape[1]):
        best_f1, best_t = 0.0, 0.5
        for t in grid:
            f1 = f1_score(labels[:, c].astype(int),
                          (probs[:, c] > t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[c] = best_t
    return thresholds


def eval_ml(model, X_va, y_va, X_te, y_te):
    """Fit thresholds on val, evaluate on test."""
    # MultiOutputClassifier predict_proba returns list of (N,2) arrays
    proba_va = np.stack([p[:, 1] for p in model.predict_proba(X_va)], axis=1)
    proba_te = np.stack([p[:, 1] for p in model.predict_proba(X_te)], axis=1)

    thresholds = optimize_thresholds(proba_va, y_va.astype(int))
    preds = np.stack([(proba_te[:, c] > thresholds[c]).astype(int)
                      for c in range(y_te.shape[1])], axis=1)

    f1_macro = f1_score(y_te.astype(int), preds, average="macro", zero_division=0)
    f1_05    = f1_score(y_te.astype(int),
                        (proba_te > 0.5).astype(int), average="macro", zero_division=0)
    per_cls  = f1_score(y_te.astype(int), preds, average=None, zero_division=0).tolist()
    return f1_macro, f1_05, per_cls, thresholds.tolist()


def main():
    t0 = time.time()

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading v3 data...")
    X = np.load(str(DATA_DIR / "expression_matrix_v3_expanded.npy"))
    y = np.load(str(DATA_DIR / "outcome_labels_v3_expanded.npy")).astype(np.float32)
    print(f"  X: {X.shape}, y: {y.shape}")

    # Same split as v7 (seed=42)
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(X))
    tr  = idx[:N_TRAIN]; va = idx[N_TRAIN:N_TRAIN+N_VAL]; te = idx[N_TRAIN+N_VAL:]
    X_tr, y_tr = X[tr], y[tr]
    X_va, y_va = X[va], y[va]
    X_te, y_te = X[te], y[te]

    # ── Normalize (z-score + clip, same as v7) ────────────────────────────────
    mu  = X_tr.mean(0, keepdims=True)
    std = X_tr.std(0, keepdims=True); std[std < 1e-6] = 1.0
    X_tr = np.clip((X_tr - mu) / std, -10, 10).astype(np.float32)
    X_va = np.clip((X_va - mu) / std, -10, 10).astype(np.float32)
    X_te = np.clip((X_te - mu) / std, -10, 10).astype(np.float32)

    # ── Gene selection (top 5000 discriminative, same as v6/v7) ──────────────
    print(f"Selecting top {N_GENES} genes...")
    top_idx = select_discriminative_genes(X_tr, y_tr, N_GENES)
    X_tr5 = X_tr[:, top_idx]
    X_va5 = X_va[:, top_idx]
    X_te5 = X_te[:, top_idx]
    print(f"Split: {N_TRAIN}/{N_VAL}/{N_TEST}  |  features: {X_tr5.shape[1]}")

    n_total = len(X)
    results = {
        "split": f"{N_TRAIN}/{N_VAL}/{n_total - N_TRAIN - N_VAL}",
        "n_total": n_total,
        "n_genes": N_GENES,
        "dataset": "v3_expanded (2540 samples, 843 new real GEO)",
        "gene_selection": "max_t_statistic",
        "normalization": "z-score + clip[-10,10]",
        "threshold_opt": "per-class F1 on val (same as ToxiGene v8)",
        "reference_models": {
            "ToxiGene_v8":            "TBD",
            "ToxiGene_v7":            0.8860,
            "SimpleMLP_baseline":     0.8896,
        },
        "models": {},
    }

    # ── Random Forest ─────────────────────────────────────────────────────────
    print("\n--- RandomForest (n=500, class_weight=balanced) ---")
    t1 = time.time()
    rf = MultiOutputClassifier(
        RandomForestClassifier(n_estimators=500, class_weight="balanced",
                               random_state=SEED, n_jobs=-1, max_features="sqrt"),
        n_jobs=1,
    )
    rf.fit(X_tr5, y_tr.astype(int))
    f1, f1_05, per_cls, thr = eval_ml(rf, X_va5, y_va, X_te5, y_te)
    elapsed_rf = time.time() - t1
    print(f"  RF  F1={f1:.4f}  (t=0.5: {f1_05:.4f})  elapsed={elapsed_rf:.0f}s")
    results["models"]["RandomForest"] = {
        "test_f1_macro": round(f1, 6), "test_f1_t05": round(f1_05, 6),
        "per_class_f1": dict(zip(OUTCOME_NAMES, [round(v,6) for v in per_cls])),
        "best_thresholds": dict(zip(OUTCOME_NAMES, [round(v,3) for v in thr])),
        "elapsed_s": round(elapsed_rf, 1),
        "hyperparams": "n=500, class_weight=balanced, max_features=sqrt",
    }

    # ── Extra Trees ───────────────────────────────────────────────────────────
    print("\n--- ExtraTrees (n=500, class_weight=balanced) ---")
    t1 = time.time()
    et = MultiOutputClassifier(
        ExtraTreesClassifier(n_estimators=500, class_weight="balanced",
                             random_state=SEED, n_jobs=-1),
        n_jobs=1,
    )
    et.fit(X_tr5, y_tr.astype(int))
    f1, f1_05, per_cls, thr = eval_ml(et, X_va5, y_va, X_te5, y_te)
    elapsed_et = time.time() - t1
    print(f"  ET  F1={f1:.4f}  (t=0.5: {f1_05:.4f})  elapsed={elapsed_et:.0f}s")
    results["models"]["ExtraTrees"] = {
        "test_f1_macro": round(f1, 6), "test_f1_t05": round(f1_05, 6),
        "per_class_f1": dict(zip(OUTCOME_NAMES, [round(v,6) for v in per_cls])),
        "best_thresholds": dict(zip(OUTCOME_NAMES, [round(v,3) for v in thr])),
        "elapsed_s": round(elapsed_et, 1),
    }

    # ── Logistic Regression ───────────────────────────────────────────────────
    print("\n--- LogisticRegression (C=0.1, class_weight=balanced) ---")
    t1 = time.time()
    lr = MultiOutputClassifier(
        LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000,
                           random_state=SEED, n_jobs=-1, solver="saga"),
        n_jobs=1,
    )
    lr.fit(X_tr5, y_tr.astype(int))
    f1, f1_05, per_cls, thr = eval_ml(lr, X_va5, y_va, X_te5, y_te)
    elapsed_lr = time.time() - t1
    print(f"  LR  F1={f1:.4f}  (t=0.5: {f1_05:.4f})  elapsed={elapsed_lr:.0f}s")
    results["models"]["LogisticRegression"] = {
        "test_f1_macro": round(f1, 6), "test_f1_t05": round(f1_05, 6),
        "per_class_f1": dict(zip(OUTCOME_NAMES, [round(v,6) for v in per_cls])),
        "best_thresholds": dict(zip(OUTCOME_NAMES, [round(v,3) for v in thr])),
        "elapsed_s": round(elapsed_lr, 1),
        "hyperparams": "C=0.1, class_weight=balanced, solver=saga",
    }

    # ── PCA + Logistic Regression ─────────────────────────────────────────────
    print("\n--- PCA(100) + LogisticRegression ---")
    t1 = time.time()
    pca = PCA(n_components=100, random_state=SEED)
    X_tr_pca = pca.fit_transform(X_tr5)
    X_va_pca = pca.transform(X_va5)
    X_te_pca = pca.transform(X_te5)
    pca_lr = MultiOutputClassifier(
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000,
                           random_state=SEED, solver="lbfgs"),
        n_jobs=-1,
    )
    pca_lr.fit(X_tr_pca, y_tr.astype(int))
    f1, f1_05, per_cls, thr = eval_ml(pca_lr, X_va_pca, y_va, X_te_pca, y_te)
    elapsed_pca = time.time() - t1
    print(f"  PCA+LR  F1={f1:.4f}  (t=0.5: {f1_05:.4f})  elapsed={elapsed_pca:.0f}s")
    results["models"]["PCA_LR"] = {
        "test_f1_macro": round(f1, 6), "test_f1_t05": round(f1_05, 6),
        "per_class_f1": dict(zip(OUTCOME_NAMES, [round(v,6) for v in per_cls])),
        "best_thresholds": dict(zip(OUTCOME_NAMES, [round(v,3) for v in thr])),
        "elapsed_s": round(elapsed_pca, 1),
        "hyperparams": "PCA(100) + LR(C=1, class_weight=balanced)",
    }

    # ── XGBoost ───────────────────────────────────────────────────────────────
    if HAS_XGB:
        print("\n--- XGBoost (n=300, scale_pos_weight) ---")
        t1 = time.time()
        pos_counts = y_tr.sum(0)
        neg_counts = N_TRAIN - pos_counts
        spw = (neg_counts / np.maximum(pos_counts, 1)).mean()
        xgb = MultiOutputClassifier(
            XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                          scale_pos_weight=spw, random_state=SEED,
                          eval_metric="logloss", verbosity=0, n_jobs=-1),
            n_jobs=1,
        )
        xgb.fit(X_tr5, y_tr.astype(int))
        f1, f1_05, per_cls, thr = eval_ml(xgb, X_va5, y_va, X_te5, y_te)
        elapsed_xgb = time.time() - t1
        print(f"  XGB  F1={f1:.4f}  (t=0.5: {f1_05:.4f})  elapsed={elapsed_xgb:.0f}s")
        results["models"]["XGBoost"] = {
            "test_f1_macro": round(f1, 6), "test_f1_t05": round(f1_05, 6),
            "per_class_f1": dict(zip(OUTCOME_NAMES, [round(v,6) for v in per_cls])),
            "best_thresholds": dict(zip(OUTCOME_NAMES, [round(v,3) for v in thr])),
            "elapsed_s": round(elapsed_xgb, 1),
        }

    results["total_elapsed_s"] = round(time.time() - t0, 1)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {RESULTS_PATH}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"ToxiGene — ML Baselines vs DL (v3 dataset, {N_TRAIN}/{N_VAL}/{n_total-N_TRAIN-N_VAL} split)")
    print(f"  {'Model':<30} {'F1 (opt thresh)':>16} {'F1 (t=0.5)':>11}  Type")
    print("-" * 65)
    ordered = [
        ("ToxiGene v7 (v2 data)", 0.8860, None, "DL"),
        ("SimpleMLP baseline",    0.8896, None, "DL"),
    ]
    for name, m in results["models"].items():
        ordered.append((name, m["test_f1_macro"], m["test_f1_t05"], "ML"))
    for name, f1, f1_05, typ in sorted(ordered, key=lambda x: -x[1]):
        f05_s = f"  {f1_05:.4f}" if f1_05 else "       —"
        print(f"  {name:<30} {f1:.4f}          {f05_s}  {typ}")
    print("=" * 65)


if __name__ == "__main__":
    main()
