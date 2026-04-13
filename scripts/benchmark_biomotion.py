#!/usr/bin/env python3
"""
BioMotion Benchmark — compare against classical and DL baselines.

Uses data/processed/behavioral_fullreal/ (expanded dataset).
SAME split as train_biomotion_expanded.py: seed=42, TEST_FRAC=0.15, VAL_FRAC=0.15,
stratified by class (separate shuffle of normal and anomalous, then split each).

Baselines:
  1. Statistical threshold (DaphTox-style): flag if mean_speed < 0.3 * pop_mean_speed
  2. LSTM on features sequence (200, 16) — bidirectional, hidden=128
  3. Transformer on features sequence (200, 16) — 2-layer with CLS token
  4. Isolation Forest on per-trajectory summary statistics (mean/std/max + keypoint stats)
  5. VAE reconstruction error (train on normal only; anomaly = high reconstruction error)
  6. BioMotion (original 17k): loaded from checkpoints/biomotion/phase2_best.pt
  7. BioMotion (expanded):     loaded from checkpoints/biomotion/biomotion_expanded_best.pt

Saves: results/benchmarks/biomotion_benchmark.json

Usage:
    CUDA_VISIBLE_DEVICES=3 python scripts/benchmark_biomotion.py
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sentinel.models.biomotion.trajectory_encoder import TrajectoryDiffusionEncoder, EMBED_DIM
from sentinel.models.biomotion.multi_organism import SPECIES_FEATURE_DIM

# ── Config ─────────────────────────────────────────────────────────────────
# Use expanded fullreal dir if available, fall back to behavioral_real
_fullreal = PROJECT_ROOT / "data" / "processed" / "behavioral_fullreal"
_realdir  = PROJECT_ROOT / "data" / "processed" / "behavioral_real"
DATA_DIR  = _fullreal if _fullreal.exists() and any(_fullreal.glob("traj_*.npz")) else _realdir

CKPT_DIR    = PROJECT_ROOT / "checkpoints" / "biomotion"
RESULTS_DIR = PROJECT_ROOT / "results" / "benchmarks"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = RESULTS_DIR / "biomotion_benchmark.json"

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEATURE_DIM = SPECIES_FEATURE_DIM["daphnia"]   # 16
T           = 200
BATCH_SIZE  = 64
SEED        = 42
TEST_FRAC   = 0.15
VAL_FRAC    = 0.15


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── Dataset ────────────────────────────────────────────────────────────────

class BehavioralDataset(Dataset):
    def __init__(self, file_paths: list[Path]) -> None:
        self.fps = file_paths
        self._cache: dict[int, dict] = {}

    def __len__(self) -> int:
        return len(self.fps)

    def __getitem__(self, idx: int) -> dict:
        if idx not in self._cache:
            d = np.load(self.fps[idx])
            self._cache[idx] = {
                "keypoints":  d["keypoints"].astype(np.float32),
                "features":   d["features"].astype(np.float32),
                "timestamps": d["timestamps"].astype(np.float32),
                "is_anomaly": bool(d["is_anomaly"]),
            }
        return self._cache[idx]

    @staticmethod
    def collate(batch):
        return {
            "keypoints":  torch.from_numpy(np.stack([s["keypoints"]  for s in batch])),
            "features":   torch.from_numpy(np.stack([s["features"]   for s in batch])),
            "timestamps": torch.from_numpy(np.stack([s["timestamps"] for s in batch])),
            "labels":     torch.tensor([float(s["is_anomaly"]) for s in batch], dtype=torch.float32),
        }


# ── Data splitting — identical to train_biomotion_expanded.py ──────────────

def load_split() -> tuple[list, list, list, list]:
    """Return (train_normal_fps, train_all_fps, val_fps, test_fps)."""
    all_files = sorted(DATA_DIR.glob("traj_*.npz"))
    assert len(all_files), f"No files in {DATA_DIR}"
    log(f"Dataset: {DATA_DIR.name}  ({len(all_files)} files)")

    normal, anomaly = [], []
    for f in all_files:
        d = np.load(f)
        (anomaly if bool(d["is_anomaly"]) else normal).append(f)
    log(f"  Normal: {len(normal)}, Anomalous: {len(anomaly)}")

    rng = np.random.default_rng(SEED)
    rng.shuffle(normal); rng.shuffle(anomaly)

    def split(lst):
        n = len(lst)
        n_te = max(1, int(n * TEST_FRAC))
        n_va = max(1, int(n * VAL_FRAC))
        return lst[:n - n_te - n_va], lst[n - n_te - n_va:n - n_te], lst[n - n_te:]

    n_tr, n_va, n_te = split(normal)
    a_tr, a_va, a_te = split(anomaly)
    log(f"  Train: {len(n_tr)+len(a_tr)}, Val: {len(n_va)+len(a_va)}, Test: {len(n_te)+len(a_te)}")
    return n_tr, n_tr + a_tr, n_va + a_va, n_te + a_te


def make_loader(fps, shuffle: bool, bs: int = BATCH_SIZE) -> DataLoader:
    ds = BehavioralDataset(fps)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle,
                      collate_fn=BehavioralDataset.collate,
                      num_workers=4, pin_memory=True)


def load_arrays(fps) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feats, labels, kps = [], [], []
    for f in fps:
        d = np.load(f)
        feats.append(d["features"].astype(np.float32))
        labels.append(float(d["is_anomaly"]))
        kps.append(d["keypoints"].astype(np.float32))
    return np.array(feats), np.array(labels), np.array(kps)


# ── Metrics ────────────────────────────────────────────────────────────────

def metrics(y_true: np.ndarray, y_scores: np.ndarray) -> dict[str, float]:
    s_min, s_max = y_scores.min(), y_scores.max()
    y_prob = (y_scores - s_min) / (s_max - s_min + 1e-8)
    y_pred = (y_prob >= 0.5).astype(float)
    return {
        "auroc":     float(roc_auc_score(y_true, y_scores)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "accuracy":  float(accuracy_score(y_true, y_pred)),
    }


# ══════════════════════════════════════════════════════════════════════════
# BASELINE 1 — Statistical Threshold (DaphTox-style)
# ══════════════════════════════════════════════════════════════════════════

def baseline_statistical(train_fps, test_fps) -> dict:
    """Flag trajectory as anomalous if mean_speed < 0.3 * population mean speed.

    Speed proxy = mean of features[:, 0] (locomotion channel, index 0 = LOCO).
    """
    log("Baseline 1: Statistical threshold (DaphTox-style)...")
    tr_feats, tr_labels, _ = load_arrays(train_fps)
    normal_speeds = tr_feats[tr_labels == 0, :, 0].mean(axis=1)
    pop_mean = float(normal_speeds.mean())

    te_feats, te_labels, _ = load_arrays(test_fps)
    te_speeds = te_feats[:, :, 0].mean(axis=1)
    # Score: more negative = faster (more normal). Anomaly = score closer to 0/positive.
    anomaly_scores = -(te_speeds / (pop_mean + 1e-8))
    m = metrics(te_labels, anomaly_scores)
    log(f"  AUROC={m['auroc']:.4f}  F1={m['f1']:.4f}")
    return {"method": "StatisticalThreshold (DaphTox-style)",
            "pop_mean_speed": pop_mean, "threshold_30pct": 0.3 * pop_mean, **m}


# ══════════════════════════════════════════════════════════════════════════
# Sequence model training helper
# ══════════════════════════════════════════════════════════════════════════

def train_seq(model, train_fps, val_fps, n_epochs=30, lr=1e-3, patience=7, tag="") -> None:
    model.to(DEVICE)
    tr_ld = make_loader(train_fps, shuffle=True)
    va_ld = make_loader(val_fps,   shuffle=False)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_val = float("inf"); no_improve = 0; best_state = None

    for epoch in range(n_epochs):
        model.train()
        for batch in tr_ld:
            feats  = batch["features"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            loss   = F.binary_cross_entropy_with_logits(model(feats), labels)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

        model.eval(); va_losses = []
        with torch.no_grad():
            for batch in va_ld:
                feats  = batch["features"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)
                va_losses.append(F.binary_cross_entropy_with_logits(model(feats), labels).item())
        val_loss = float(np.mean(va_losses))
        if val_loss < best_val:
            best_val = val_loss; no_improve = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
        if no_improve >= patience:
            log(f"    {tag}: early stop epoch {epoch+1}")
            break

    if best_state:
        model.load_state_dict(best_state)


def eval_seq(model, test_fps) -> tuple[np.ndarray, np.ndarray]:
    model.to(DEVICE).eval()
    ld = make_loader(test_fps, shuffle=False)
    scores, labels = [], []
    with torch.no_grad():
        for batch in ld:
            scores.append(torch.sigmoid(model(batch["features"].to(DEVICE))).cpu().numpy())
            labels.append(batch["labels"].numpy())
    return np.concatenate(scores), np.concatenate(labels)


# ══════════════════════════════════════════════════════════════════════════
# BASELINE 2 — Bidirectional LSTM
# ══════════════════════════════════════════════════════════════════════════

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim=16, hidden=128, n_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, n_layers,
                            batch_first=True, dropout=dropout, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 1))
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.head(torch.cat([h[-2], h[-1]], dim=-1)).squeeze(-1)


def baseline_lstm(train_fps, val_fps, test_fps) -> dict:
    log("Baseline 2: Bidirectional LSTM classifier (30 epochs, patience=7)...")
    model = LSTMClassifier()
    n_p = sum(p.numel() for p in model.parameters())
    log(f"  Params: {n_p:,}")
    train_seq(model, train_fps, val_fps, n_epochs=30, tag="LSTM")
    scores, labels = eval_seq(model, test_fps)
    m = metrics(labels, scores)
    log(f"  AUROC={m['auroc']:.4f}  F1={m['f1']:.4f}")
    return {"method": "LSTM (BiLSTM, hidden=128)", "n_params": n_p, **m}


# ══════════════════════════════════════════════════════════════════════════
# BASELINE 3 — Transformer
# ══════════════════════════════════════════════════════════════════════════

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim=16, d_model=128, nhead=4, n_layers=2,
                 dim_ff=256, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout,
                                                batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, n_layers)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.head = nn.Sequential(
            nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 1))
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = torch.cat([self.cls.expand(B, -1, -1), self.proj(x)], dim=1)
        return self.head(self.enc(x)[:, 0]).squeeze(-1)


def baseline_transformer(train_fps, val_fps, test_fps) -> dict:
    log("Baseline 3: Transformer classifier (30 epochs, patience=7)...")
    model = TransformerClassifier()
    n_p = sum(p.numel() for p in model.parameters())
    log(f"  Params: {n_p:,}")
    train_seq(model, train_fps, val_fps, n_epochs=30, tag="Transformer")
    scores, labels = eval_seq(model, test_fps)
    m = metrics(labels, scores)
    log(f"  AUROC={m['auroc']:.4f}  F1={m['f1']:.4f}")
    return {"method": "Transformer (2-layer, CLS token)", "n_params": n_p, **m}


# ══════════════════════════════════════════════════════════════════════════
# BASELINE 4 — Isolation Forest on summary statistics
# ══════════════════════════════════════════════════════════════════════════

def extract_summary(feats: np.ndarray, kps: np.ndarray) -> np.ndarray:
    """Mean/std/max of 16 feature channels + keypoint-derived stats → (N, 52)."""
    rows = []
    for i in range(feats.shape[0]):
        f = feats[i]; k = kps[i]
        fstats = np.concatenate([f.mean(0), f.std(0), f.max(0)])   # 48
        kdisp  = np.linalg.norm(np.diff(k, axis=0), axis=-1).mean()
        kstd   = float(k.std())
        krange = float(k.max() - k.min())
        loco   = float(f[:, 0].mean())
        rows.append(np.concatenate([fstats, [kdisp, kstd, krange, loco]]))
    return np.array(rows, dtype=np.float32)


def baseline_isolation_forest(train_fps, test_fps) -> dict:
    log("Baseline 4: Isolation Forest on summary statistics...")
    tr_feats, _, tr_kps = load_arrays(train_fps)
    te_feats, te_labels, te_kps = load_arrays(test_fps)
    X_tr = extract_summary(tr_feats, tr_kps)
    X_te = extract_summary(te_feats, te_kps)
    clf = IsolationForest(n_estimators=200, contamination="auto",
                          random_state=SEED, n_jobs=-1)
    clf.fit(X_tr)
    anomaly_scores = -clf.score_samples(X_te)
    m = metrics(te_labels, anomaly_scores)
    log(f"  AUROC={m['auroc']:.4f}  F1={m['f1']:.4f}")
    return {"method": "Isolation Forest (mean/std/max + keypoint stats)", **m}


# ══════════════════════════════════════════════════════════════════════════
# BASELINE 5 — VAE Reconstruction Error
# ══════════════════════════════════════════════════════════════════════════

class SequenceVAE(nn.Module):
    def __init__(self, input_dim=16, latent=32, hidden=128):
        super().__init__()
        flat = input_dim * T
        self.enc  = nn.Sequential(nn.Linear(flat, hidden*2), nn.GELU(),
                                   nn.Linear(hidden*2, hidden), nn.GELU())
        self.mu   = nn.Linear(hidden, latent)
        self.lv   = nn.Linear(hidden, latent)
        self.dec  = nn.Sequential(nn.Linear(latent, hidden), nn.GELU(),
                                   nn.Linear(hidden, hidden*2), nn.GELU(),
                                   nn.Linear(hidden*2, flat))

    def forward(self, x):
        h  = self.enc(x.reshape(x.size(0), -1))
        mu, lv = self.mu(h), self.lv(h)
        z  = mu + (torch.randn_like(mu) * torch.exp(0.5 * lv) if self.training else 0)
        return self.dec(z).reshape_as(x), mu, lv

    @torch.no_grad()
    def recon_error(self, x):
        rec, _, _ = self.forward(x)
        return F.mse_loss(rec, x, reduction="none").mean(dim=[1, 2])


def train_vae(model, normal_fps, val_fps, n_epochs=30, lr=1e-3, patience=7) -> None:
    model.to(DEVICE)
    tr_ld = make_loader(normal_fps, shuffle=True)
    va_ld = make_loader(val_fps,    shuffle=False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_val = float("inf"); no_improve = 0

    for epoch in range(n_epochs):
        model.train()
        for batch in tr_ld:
            feats = batch["features"].to(DEVICE)
            rec, mu, lv = model(feats)
            loss = F.mse_loss(rec, feats) - 5e-4 * (1 + lv - mu.pow(2) - lv.exp()).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

        model.eval(); va_l = []
        with torch.no_grad():
            for batch in va_ld:
                feats = batch["features"].to(DEVICE)
                rec, _, _ = model(feats)
                va_l.append(F.mse_loss(rec, feats).item())
        val_loss = float(np.mean(va_l))
        if val_loss < best_val:
            best_val = val_loss; no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            log(f"    VAE: early stop epoch {epoch+1}")
            break


def baseline_vae(normal_fps, val_fps, test_fps) -> dict:
    log("Baseline 5: VAE Reconstruction Error (normal-only training)...")
    model = SequenceVAE()
    n_p = sum(p.numel() for p in model.parameters())
    log(f"  VAE params: {n_p:,}")
    train_vae(model, normal_fps, val_fps, n_epochs=30)
    model.eval(); scores, labels = [], []
    ld = make_loader(test_fps, shuffle=False)
    with torch.no_grad():
        for batch in ld:
            scores.append(model.recon_error(batch["features"].to(DEVICE)).cpu().numpy())
            labels.append(batch["labels"].numpy())
    y_scores = np.concatenate(scores); y_true = np.concatenate(labels)
    m = metrics(y_true, y_scores)
    log(f"  AUROC={m['auroc']:.4f}  F1={m['f1']:.4f}")
    return {"method": "VAE Reconstruction Error (normal-only)", "n_params": n_p, **m}


# ══════════════════════════════════════════════════════════════════════════
# BioMotion checkpoints
# ══════════════════════════════════════════════════════════════════════════

class AnomalyClassifier(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(EMBED_DIM, EMBED_DIM // 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(EMBED_DIM // 2, 1))
    def forward(self, x):
        return self.classifier(self.encoder.forward_encode(x)).squeeze(-1)


def load_biomotion(ckpt_path: Path) -> AnomalyClassifier:
    enc = TrajectoryDiffusionEncoder(feature_dim=FEATURE_DIM, embed_dim=EMBED_DIM,
                                      nhead=4, num_layers=4, dim_feedforward=512, dropout=0.1)
    model = AnomalyClassifier(enc)
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    return model.to(DEVICE).eval()


def eval_biomotion(model, test_fps) -> dict:
    ld = make_loader(test_fps, shuffle=False)
    scores, labels = [], []
    with torch.no_grad():
        for batch in ld:
            scores.append(torch.sigmoid(model(batch["features"].to(DEVICE))).cpu().numpy())
            labels.append(batch["labels"].numpy())
    return metrics(np.concatenate(labels), np.concatenate(scores))


def baseline_biomotion_original(test_fps) -> dict:
    ckpt = CKPT_DIR / "phase2_best.pt"
    if not ckpt.exists():
        return {"method": "BioMotion (original 17k)", "error": "checkpoint not found"}
    log("BioMotion (original 17k, phase2_best.pt)...")
    m = eval_biomotion(load_biomotion(ckpt), test_fps)
    log(f"  AUROC={m['auroc']:.4f}  F1={m['f1']:.4f}")
    return {"method": "BioMotion (original 17k)", **m}


def baseline_biomotion_expanded(test_fps) -> dict:
    ckpt = CKPT_DIR / "biomotion_expanded_best.pt"
    if not ckpt.exists():
        return {"method": "BioMotion (expanded)", "error": "checkpoint not found"}
    log("BioMotion (expanded, biomotion_expanded_best.pt)...")
    m = eval_biomotion(load_biomotion(ckpt), test_fps)
    log(f"  AUROC={m['auroc']:.4f}  F1={m['f1']:.4f}")
    return {"method": "BioMotion (expanded)", **m}


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def print_table(results: list[dict]) -> None:
    print("\n" + "=" * 82)
    print(f"{'Method':<42} {'AUROC':>7} {'F1':>7} {'Prec':>7} {'Recall':>7} {'Acc':>7}")
    print("-" * 82)
    for r in results:
        name = r.get("method", "?")[:41]
        if "error" in r:
            print(f"{name:<42}  [ERROR: {r['error']}]"); continue
        print(f"{name:<42} "
              f"{r.get('auroc', 0):>7.4f} {r.get('f1', 0):>7.4f} "
              f"{r.get('precision', 0):>7.4f} {r.get('recall', 0):>7.4f} "
              f"{r.get('accuracy', 0):>7.4f}")
    print("=" * 82)


def main() -> None:
    t0 = time.time()
    log(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        log(f"  GPU: {torch.cuda.get_device_name()}")
    torch.manual_seed(SEED); np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    train_normal_fps, train_all_fps, val_fps, test_fps = load_split()

    results = []
    results.append(baseline_statistical(train_all_fps, test_fps))
    results.append(baseline_lstm(train_all_fps, val_fps, test_fps))
    results.append(baseline_transformer(train_all_fps, val_fps, test_fps))
    results.append(baseline_isolation_forest(train_all_fps, test_fps))
    results.append(baseline_vae(train_normal_fps, val_fps, test_fps))
    results.append(baseline_biomotion_original(test_fps))
    results.append(baseline_biomotion_expanded(test_fps))

    print_table(results)

    output = {
        "benchmark": "BioMotion vs Baselines",
        "data_dir": str(DATA_DIR),
        "n_train": len(train_all_fps),
        "n_val":   len(val_fps),
        "n_test":  len(test_fps),
        "seed": SEED,
        "split": f"test={TEST_FRAC}, val={VAL_FRAC}, stratified by class",
        "baseline_auroc_original_17k": 0.9621,
        "results": results,
        "elapsed_seconds": time.time() - t0,
    }
    with open(RESULTS_PATH, "w") as fh:
        json.dump(output, fh, indent=2, default=str)
    log(f"Saved → {RESULTS_PATH}")
    log(f"Total time: {(time.time()-t0)/60:.1f}m")


if __name__ == "__main__":
    main()
