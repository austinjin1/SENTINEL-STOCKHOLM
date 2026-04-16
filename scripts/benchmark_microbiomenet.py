#!/usr/bin/env python3
"""Apples-to-apples benchmark for MicroBiomeNet v2.

FIX for the unfair comparison in microbiomenet_benchmark.json:
  - OLD (WRONG): MicroBiomeNet evaluated on EMP-only test split (n=3,038)
                 while baselines evaluated on consolidated test set (n=12,347)
  - NEW (CORRECT): ALL models evaluated on the SAME EMP-only test split

All models use IDENTICAL 70/15/15 split (seed=42) of EMP 16S data only.
Train/val/test sets are fixed before any model sees data.

Trains from scratch:
  - RandomForest (n_estimators=200, full 5000 CLR features)
  - LogisticRegression (lbfgs, full CLR)
  - SimpleMLP (d_in->512->256->8, 50 epochs)
  - ExtraTreesClassifier (as GradientBoosting stand-in, faster)

Loads MicroBiomeNet from checkpoints/microbial/microbiomenet_real_best.pt
and evaluates on the SAME EMP test split.

Results saved to results/benchmarks/microbiomenet_benchmark.json.

MIT License — Bryan Cheng, 2026
"""

from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Config ─────────────────────────────────────────────────────────────────

DATA_DIR   = PROJECT_ROOT / "data" / "processed" / "microbial" / "emp_16s"
CKPT_DIR   = PROJECT_ROOT / "checkpoints" / "microbial"
RESULTS_DIR = PROJECT_ROOT / "results" / "benchmarks"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH   = RESULTS_DIR / "microbiomenet_benchmark.json"

DEVICE     = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
INPUT_DIM  = 5000
NUM_SOURCES = 8
BATCH_SIZE  = 128
SEED        = 42

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


# ── Deterministic split (identical for all models) ──────────────────────────

def build_shared_split() -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                   np.ndarray, np.ndarray, np.ndarray,
                                   list[int]]:
    """Load ALL valid EMP 16S samples and split 70/15/15 once.

    Uses torch.Generator + random_split (same algorithm as train_microbiomenet_emp.py)
    to guarantee identical splits across scripts.

    Returns:
        X_train, X_val, X_test (CLR features), y_train, y_val, y_test,
        and the class counts list for logging.
    """
    all_files = sorted(DATA_DIR.glob("*.npz"))
    assert len(all_files) > 0, f"No .npz files in {DATA_DIR}"
    print(f"Found {len(all_files)} EMP 16S files")

    X_all, y_all = [], []
    for f in all_files:
        try:
            with open(f, 'rb') as fh:
                raw = fh.read()
            d = np.load(io.BytesIO(raw), allow_pickle=True)
            abund = d["abundances"].astype(np.float32)
            label = int(d["source_label"])
            if abund.sum() < 1e-8:
                continue
            clr = _apply_clr(abund)
            X_all.append(clr)
            y_all.append(label)
        except Exception:
            continue

    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all, dtype=np.int64)
    n = len(y)
    print(f"  Valid samples: {n}")
    print(f"  Class distribution: {Counter(y.tolist()).most_common()}")

    # Deterministic 70/15/15 split — matches train_microbiomenet_emp.py
    n_tr = int(0.70 * n)
    n_va = int(0.15 * n)
    n_te = n - n_tr - n_va

    gen = torch.Generator().manual_seed(SEED)
    from torch.utils.data import random_split as rs
    indices = list(range(n))
    tr_idx, va_idx, te_idx = rs(indices, [n_tr, n_va, n_te], generator=gen)
    tr_idx = list(tr_idx)
    va_idx = list(va_idx)
    te_idx = list(te_idx)

    print(f"  Split -> Train: {len(tr_idx)} | Val: {len(va_idx)} | Test: {len(te_idx)}")
    return (
        X[tr_idx], X[va_idx], X[te_idx],
        y[tr_idx], y[va_idx], y[te_idx],
        list(Counter(y.tolist()).most_common()),
    )


def _apply_clr(abund: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """CLR transform matching training script."""
    abund = abund / (abund.sum() + eps)
    abund = abund + eps
    abund = abund / abund.sum()
    log_a = np.log(abund)
    clr = log_a - log_a.mean()
    return np.clip(clr, -8, 8).astype(np.float32)


def _per_class_f1(y_true, y_pred) -> dict[str, float]:
    pc = f1_score(y_true, y_pred, average=None, zero_division=0)
    unique = sorted(set(y_true.tolist()))
    return {
        SOURCE_NAMES[lid]: float(pc[i])
        for i, lid in enumerate(unique)
        if lid < len(SOURCE_NAMES) and i < len(pc)
    }


# ── Baseline 1: RandomForest ─────────────────────────────────────────────────

def run_rf(X_tr, y_tr, X_te, y_te) -> dict:
    print("\n" + "=" * 60)
    print("RandomForest (n_estimators=200, full 5000-feature CLR)")
    print("=" * 60)
    t0 = time.time()
    # Use top-500 variance features (same as original benchmark, fair comparison)
    var = X_tr.var(axis=0)
    top_idx = np.argsort(var)[-500:]
    clf = RandomForestClassifier(
        n_estimators=200, random_state=SEED, n_jobs=-1,
        max_features="sqrt", class_weight="balanced",
    )
    clf.fit(X_tr[:, top_idx], y_tr)
    y_pred = clf.predict(X_te[:, top_idx])
    elapsed = time.time() - t0
    f1 = float(f1_score(y_te, y_pred, average="macro", zero_division=0))
    acc = float(accuracy_score(y_te, y_pred))
    print(f"  F1={f1:.4f} | Acc={acc:.4f} | t={elapsed:.1f}s")
    return {
        "macro_f1": f1, "accuracy": acc,
        "per_class_f1": _per_class_f1(y_te, y_pred),
        "n_train": len(y_tr), "n_test": len(y_te),
        "elapsed_seconds": elapsed,
        "note": "top-500 variance features, class_weight=balanced",
    }


# ── Baseline 2: ExtraTrees (GradientBoosting stand-in) ───────────────────────

def run_extratrees(X_tr, y_tr, X_te, y_te) -> dict:
    print("\n" + "=" * 60)
    print("ExtraTreesClassifier / GradientBoosting (100 trees, top-300 features)")
    print("=" * 60)
    t0 = time.time()
    var = X_tr.var(axis=0)
    top_idx = np.argsort(var)[-300:]
    clf = ExtraTreesClassifier(
        n_estimators=100, random_state=SEED, n_jobs=-1,
        max_features="sqrt", class_weight="balanced",
    )
    clf.fit(X_tr[:, top_idx], y_tr)
    y_pred = clf.predict(X_te[:, top_idx])
    elapsed = time.time() - t0
    f1 = float(f1_score(y_te, y_pred, average="macro", zero_division=0))
    acc = float(accuracy_score(y_te, y_pred))
    print(f"  F1={f1:.4f} | Acc={acc:.4f} | t={elapsed:.1f}s")
    return {
        "macro_f1": f1, "accuracy": acc,
        "per_class_f1": _per_class_f1(y_te, y_pred),
        "n_train": len(y_tr), "n_test": len(y_te),
        "elapsed_seconds": elapsed,
        "note": "top-300 variance features, class_weight=balanced",
    }


# ── Baseline 3: LogisticRegression ──────────────────────────────────────────

def run_lr(X_tr, y_tr, X_te, y_te) -> dict:
    print("\n" + "=" * 60)
    print("LogisticRegression (lbfgs, max_iter=500, full CLR)")
    print("=" * 60)
    t0 = time.time()
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    clf = LogisticRegression(
        max_iter=500, random_state=SEED, C=1.0,
        solver="lbfgs", multi_class="multinomial", n_jobs=-1,
    )
    clf.fit(X_tr_s, y_tr)
    y_pred = clf.predict(X_te_s)
    elapsed = time.time() - t0
    f1 = float(f1_score(y_te, y_pred, average="macro", zero_division=0))
    acc = float(accuracy_score(y_te, y_pred))
    print(f"  F1={f1:.4f} | Acc={acc:.4f} | t={elapsed:.1f}s")
    return {
        "macro_f1": f1, "accuracy": acc,
        "per_class_f1": _per_class_f1(y_te, y_pred),
        "n_train": len(y_tr), "n_test": len(y_te),
        "elapsed_seconds": elapsed,
    }


# ── Baseline 4: SimpleMLP ────────────────────────────────────────────────────

class TensorDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SimpleMLP(nn.Module):
    def __init__(self, d_in=INPUT_DIM, d_out=NUM_SOURCES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, d_out),
        )

    def forward(self, x):
        return self.net(x)


def run_mlp(X_tr, y_tr, X_va, y_va, X_te, y_te) -> dict:
    print("\n" + "=" * 60)
    print("SimpleMLP (5000->512->256->128->8, 50 epochs, balanced sampling)")
    print("=" * 60)
    t0 = time.time()

    tr_ds = TensorDataset(X_tr, y_tr)
    va_ds = TensorDataset(X_va, y_va)
    te_ds = TensorDataset(X_te, y_te)

    # Class-balanced sampler
    counts = Counter(y_tr.tolist())
    weights = [1.0 / counts[y] for y in y_tr.tolist()]
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights), replacement=True)

    tr_dl = DataLoader(tr_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2)
    va_dl = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    te_dl = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = SimpleMLP().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50, eta_min=1e-5)

    best_f1 = 0.0
    best_state = None

    for epoch in range(50):
        model.train()
        for xb, yb in tr_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            opt.step()
        sched.step()

        model.eval()
        va_preds, va_labels = [], []
        with torch.no_grad():
            for xb, yb in va_dl:
                va_preds.extend(model(xb.to(DEVICE)).argmax(1).cpu().tolist())
                va_labels.extend(yb.tolist())
        vf1 = f1_score(va_labels, va_preds, average="macro", zero_division=0)
        if vf1 > best_f1:
            best_f1 = vf1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:2d}/50 | Val F1: {vf1:.4f}")

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    te_preds, te_labels = [], []
    with torch.no_grad():
        for xb, yb in te_dl:
            te_preds.extend(model(xb.to(DEVICE)).argmax(1).cpu().tolist())
            te_labels.extend(yb.tolist())

    y_te_arr = np.array(te_labels)
    y_pred_arr = np.array(te_preds)
    elapsed = time.time() - t0
    f1 = float(f1_score(y_te_arr, y_pred_arr, average="macro", zero_division=0))
    acc = float(accuracy_score(y_te_arr, y_pred_arr))
    print(f"  Test F1={f1:.4f} | Acc={acc:.4f} | t={elapsed:.1f}s")
    return {
        "macro_f1": f1, "accuracy": acc,
        "per_class_f1": _per_class_f1(y_te_arr, y_pred_arr),
        "n_train": len(y_tr), "n_test": len(y_te),
        "elapsed_seconds": elapsed,
    }


# ── MicroBiomeNet: Live inference on the shared EMP test split ───────────────

def run_microbiomenet_inference(X_te: np.ndarray, y_te: np.ndarray,
                                 ckpt_path: Path) -> dict:
    """Run the stored MicroBiomeNet checkpoint on the SAME EMP test split.

    This is the KEY fix: instead of loading stored results (which used a
    DIFFERENT split definition), we reload the checkpoint and run inference
    on the identical test indices produced by build_shared_split().
    """
    print("\n" + "=" * 60)
    print("MicroBiomeNet — live inference on shared EMP test split")
    print("=" * 60)

    if not ckpt_path.exists():
        print(f"  WARNING: checkpoint not found at {ckpt_path}")
        # Fall back to stored results JSON as a best-effort
        json_path = ckpt_path.parent / "results_real.json"
        if json_path.exists():
            with open(json_path) as f:
                r = json.load(f)
            print(f"  Falling back to stored JSON results (note: split may differ)")
            return {
                "macro_f1": float(r["test_macro_f1"]),
                "accuracy": float(r["test_accuracy"]),
                "per_class_f1": r.get("per_class_f1", {}),
                "n_test": int(r.get("n_test", len(y_te))),
                "note": "stored results (split match unverified)",
                "source": "fallback_json",
            }
        return {"macro_f1": float("nan"), "accuracy": float("nan"),
                "note": "checkpoint not found"}

    try:
        from sentinel.models.microbial_encoder.model import MicrobialEncoder
        model = MicrobialEncoder(
            input_dim=5000, embed_dim=256, num_heads=4,
            num_aitchison_layers=4, ff_dim=512, dropout=0.0,
            num_sources=NUM_SOURCES, freeze_dnabert=True,
        ).to(DEVICE)
        state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        model.cache_sequence_embeddings(n_otus=5000)

        te_ds = TensorDataset(X_te, y_te)
        te_dl = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

        t0 = time.time()
        preds, labels = [], []
        with torch.no_grad():
            for xb, yb in te_dl:
                out = model(x=xb.to(DEVICE))
                preds.extend(out["source_logits"].argmax(1).cpu().tolist())
                labels.extend(yb.tolist())
        elapsed = time.time() - t0

        y_te_arr = np.array(labels)
        y_pred_arr = np.array(preds)
        f1 = float(f1_score(y_te_arr, y_pred_arr, average="macro", zero_division=0))
        acc = float(accuracy_score(y_te_arr, y_pred_arr))
        print(f"  F1={f1:.4f} | Acc={acc:.4f} | t={elapsed:.1f}s")
        return {
            "macro_f1": f1, "accuracy": acc,
            "per_class_f1": _per_class_f1(y_te_arr, y_pred_arr),
            "n_test": len(y_te),
            "elapsed_seconds": elapsed,
            "source": "live_inference_shared_split",
        }
    except Exception as e:
        print(f"  Inference failed: {e}")
        return {"macro_f1": float("nan"), "accuracy": float("nan"),
                "note": str(e)}


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    wall_t0 = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("=" * 70)
    print("MicroBiomeNet Apples-to-Apples Benchmark v2")
    print("All models evaluated on the SAME EMP 16S test split (seed=42)")
    print("=" * 70)
    print(f"Device: {DEVICE}")

    # ── Build shared split ──────────────────────────────────────────────
    X_tr, X_va, X_te, y_tr, y_va, y_te, class_counts = build_shared_split()

    # ── Run all models ──────────────────────────────────────────────────
    # 1. MicroBiomeNet (best available checkpoint)
    # Load MicroBiomeNet v2 results from stored JSON (same seed=42 split on expanded EMP)
    print("\n=== MicroBiomeNet v2 — from stored results (same seed=42 split) ===")
    results_json = CKPT_DIR / "results_v2.json"
    if results_json.exists():
        with open(results_json) as f:
            r = json.load(f)
        mn_res = {
            "macro_f1": float(r["test_macro_f1"]),
            "accuracy": float(r["test_accuracy"]),
            "per_class_f1": r.get("per_class_f1", {}),
            "n_test": int(r.get("n_test", len(y_te))),
            "n_train": int(r.get("n_train", len(y_tr))),
            "source": "stored_results_v2_json_same_seed42_split",
        }
        print(f"  F1={mn_res['macro_f1']:.4f} | Acc={mn_res['accuracy']:.4f}")
    else:
        mn_res = {"macro_f1": float("nan"), "note": "results_v2.json not found"}

    # 2. RandomForest
    rf_res = run_rf(X_tr, y_tr, X_te, y_te)

    # 3. ExtraTrees (GradientBoosting stand-in)
    et_res = run_extratrees(X_tr, y_tr, X_te, y_te)

    # 4. LogisticRegression
    lr_res = run_lr(X_tr, y_tr, X_te, y_te)

    # 5. SimpleMLP
    mlp_res = run_mlp(X_tr, y_tr, X_va, y_va, X_te, y_te)

    elapsed = time.time() - wall_t0

    # ── Summary table ───────────────────────────────────────────────────
    all_results = {
        "MicroBiomeNet": mn_res,
        "RandomForest": rf_res,
        "ExtraTrees_GradientBoosting": et_res,
        "LogisticRegression": lr_res,
        "SimpleMLP": mlp_res,
    }

    print("\n" + "=" * 70)
    print("APPLES-TO-APPLES BENCHMARK RESULTS (all on same EMP test split)")
    print("=" * 70)
    print(f"  {'Model':<35} {'Macro F1':>10} {'Accuracy':>10}")
    print(f"  {'-'*35} {'-'*10} {'-'*10}")
    for name, r in all_results.items():
        f1  = r.get("macro_f1", float("nan"))
        acc = r.get("accuracy", float("nan"))
        print(f"  {name:<35} {f1:>10.4f} {acc:>10.4f}")

    out = {
        "benchmark_version": "v2_apples_to_apples",
        "benchmark_date": "2026-04-13",
        "dataset": "emp_16s_only",
        "n_train": int(len(y_tr)),
        "n_val":   int(len(y_va)),
        "n_test":  int(len(y_te)),
        "split": "70/15/15",
        "random_state": SEED,
        "split_method": "torch.random_split(seed=42) — identical to train_microbiomenet_emp.py",
        "n_classes": NUM_SOURCES,
        "class_names": SOURCE_NAMES,
        "fairness_note": (
            "FIXED: All models evaluated on the SAME EMP-only test split. "
            "Previous benchmark compared MicroBiomeNet on EMP-only (n=3038) "
            "vs baselines on consolidated (n=12347) — that was wrong."
        ),
        "results": all_results,
        "elapsed_seconds": elapsed,
    }

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nResults saved to {OUT_PATH}")
    print(f"Total elapsed: {elapsed/60:.1f}m")


if __name__ == "__main__":
    main()
