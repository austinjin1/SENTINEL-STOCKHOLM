#!/usr/bin/env python3
"""benchmark_biomotion_sota.py — Add Deep Autoencoder to BioMotion benchmark.

Implements the deep autoencoder from:
  "Detection of Anomalous Behavioral Patterns in Zebrafish Using Deep Autoencoder"
  PLOS Computational Biology, Sep 2024 (PMC10515950)
  Best AUROC: 0.740–0.922 across 6 phase-specific models.
  Dataset: 2,719 treated zebrafish larvae; trained on normal-only, scored by recon. error.

Also adds LSTM Autoencoder for a stronger sequence-aware baseline.

Loads SAME data split as benchmark_biomotion.py, appends to
results/benchmarks/biomotion_benchmark.json.

Bryan Cheng, SENTINEL project, 2026
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, f1_score

PROJECT_ROOT = Path("/home/bcheng/SENTINEL")
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from benchmark_biomotion import (
    load_split, load_arrays, metrics,
    BehavioralDataset, make_loader,
    DEVICE, SEED, BATCH_SIZE,
)

from sentinel.utils.logging import get_logger
logger = get_logger(__name__)

RESULTS_DIR  = PROJECT_ROOT / "results" / "benchmarks"
RESULTS_PATH = RESULTS_DIR / "biomotion_benchmark.json"

DAE_EPOCHS    = 100
DAE_PATIENCE  = 15
LSTM_AE_EPOCHS = 60
FEATURE_DIM   = 16
T             = 200
FLAT_DIM      = T * FEATURE_DIM   # 3200


# ─────────────────────────────────────────────────────────────────────────────
# Deep Autoencoder (PLOS CompBio 2024 style)
# ─────────────────────────────────────────────────────────────────────────────
class DeepAutoencoder(nn.Module):
    """Fully-connected deep autoencoder for behavioral feature sequences.

    Faithfully reimplements the architecture described in:
    PLOS CompBio 2024 (PMC10515950) — encodes flattened behavioral features
    through a bottleneck, scores anomalies by MSE reconstruction error.

    Encoder: 3200 → 512 → 256 → 128 → 64 (bottleneck)
    Decoder: 64 → 128 → 256 → 512 → 3200
    """

    def __init__(self, input_dim: int = FLAT_DIM, bottleneck: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, bottleneck),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)

    def reconstruct_error(self, x: torch.Tensor) -> torch.Tensor:
        return ((self.forward(x) - x) ** 2).mean(dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# LSTM Autoencoder (sequence-aware baseline)
# ─────────────────────────────────────────────────────────────────────────────
class LSTMAutoencoder(nn.Module):
    """Sequence-to-sequence LSTM autoencoder, trained on normal-only.

    Encoder: BiLSTM → bottleneck hidden state
    Decoder: LSTM conditioned on bottleneck → reconstructed sequence
    """

    def __init__(self, feature_dim: int = FEATURE_DIM, hidden: int = 128,
                 n_layers: int = 2, bottleneck: int = 32):
        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.bottleneck = bottleneck

        self.encoder = nn.LSTM(
            feature_dim, hidden, n_layers, batch_first=True,
            bidirectional=True, dropout=0.2 if n_layers > 1 else 0.0)
        self.to_z = nn.Linear(hidden * 2, bottleneck)   # bi → bottleneck

        self.from_z = nn.Linear(bottleneck, hidden)
        self.decoder = nn.LSTM(
            hidden, hidden, n_layers, batch_first=True, dropout=0.2 if n_layers > 1 else 0.0)
        self.out_proj = nn.Linear(hidden, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, F) → reconstruction (B, T, F)."""
        B, T, _ = x.shape
        _, (hn, _) = self.encoder(x)
        # hn: (2*n_layers, B, hidden) — take last forward + backward
        ctx = torch.cat([hn[-2], hn[-1]], dim=1)    # (B, 2*hidden)
        z = self.to_z(ctx)                           # (B, bottleneck)

        dec_init = self.from_z(z).unsqueeze(0).repeat(self.n_layers, 1, 1)   # (n_layers, B, hidden)
        dec_in = torch.zeros(B, T, self.hidden, device=x.device)
        out, _ = self.decoder(dec_in, (dec_init, torch.zeros_like(dec_init)))
        return self.out_proj(out)                    # (B, T, F)

    def reconstruct_error(self, x: torch.Tensor) -> torch.Tensor:
        return ((self.forward(x) - x) ** 2).mean(dim=(1, 2))


# ─────────────────────────────────────────────────────────────────────────────
# Normal-only Dataset (for unsupervised AE training)
# ─────────────────────────────────────────────────────────────────────────────
class NormalOnlyFlatDataset(Dataset):
    """Loads normal-only trajectories, returns flattened features for DAE."""

    def __init__(self, fps: list[Path]):
        self.data = []
        for f in fps:
            d = np.load(f)
            if not bool(d["is_anomaly"]):
                feat = d["features"].astype(np.float32)   # (T, F)
                self.data.append(feat.reshape(-1))        # (T*F,)

    def __len__(self): return len(self.data)

    def __getitem__(self, idx): return torch.tensor(self.data[idx])


class NormalOnlySeqDataset(Dataset):
    """Loads normal-only trajectories, returns (T, F) sequences for LSTM AE."""

    def __init__(self, fps: list[Path]):
        self.data = []
        for f in fps:
            d = np.load(f)
            if not bool(d["is_anomaly"]):
                self.data.append(d["features"].astype(np.float32))

    def __len__(self): return len(self.data)

    def __getitem__(self, idx): return torch.tensor(self.data[idx])


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
def train_dae(
    normal_fps: list[Path],
    val_fps: list[Path],
    epochs: int = DAE_EPOCHS,
    patience: int = DAE_PATIENCE,
) -> DeepAutoencoder:
    tr_ds = NormalOnlyFlatDataset(normal_fps)
    va_ds = NormalOnlyFlatDataset(val_fps)
    logger.info(f"  DAE training: {len(tr_ds)} normal samples")

    tr_dl = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    va_dl = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = DeepAutoencoder(FLAT_DIM, bottleneck=64).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  DAE params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val, no_imp, best_state = float("inf"), 0, None

    for ep in range(1, epochs + 1):
        model.train()
        for x in tr_dl:
            x = x.to(DEVICE)
            loss = F.mse_loss(model(x), x)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x in va_dl:
                val_loss += F.mse_loss(model(x.to(DEVICE)), x.to(DEVICE)).item()

        if val_loss < best_val:
            best_val, no_imp = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_imp += 1

        if ep % 20 == 0 or ep == 1:
            logger.info(f"  DAE epoch {ep}/{epochs} | val_loss={val_loss:.5f}")
        if no_imp >= patience:
            logger.info(f"  DAE early stopping at epoch {ep}")
            break

    model.load_state_dict(best_state)
    return model


def train_lstm_ae(
    normal_fps: list[Path],
    val_fps: list[Path],
    epochs: int = LSTM_AE_EPOCHS,
    patience: int = DAE_PATIENCE,
) -> LSTMAutoencoder:
    tr_ds = NormalOnlySeqDataset(normal_fps)
    va_ds = NormalOnlySeqDataset(val_fps)
    logger.info(f"  LSTM-AE training: {len(tr_ds)} normal samples")

    tr_dl = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    va_dl = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = LSTMAutoencoder(FEATURE_DIM, hidden=128, n_layers=2, bottleneck=32).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  LSTM-AE params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)

    best_val, no_imp, best_state = float("inf"), 0, None
    for ep in range(1, epochs + 1):
        model.train()
        for x in tr_dl:
            x = x.to(DEVICE)
            loss = F.mse_loss(model(x), x)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x in va_dl:
                val_loss += F.mse_loss(model(x.to(DEVICE)), x.to(DEVICE)).item()

        if val_loss < best_val:
            best_val, no_imp = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_imp += 1

        if ep % 20 == 0 or ep == 1:
            logger.info(f"  LSTM-AE epoch {ep}/{epochs} | val_loss={val_loss:.5f}")
        if no_imp >= patience:
            logger.info(f"  LSTM-AE early stopping at epoch {ep}")
            break

    model.load_state_dict(best_state)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def score_dae(model: DeepAutoencoder, test_fps: list[Path]) -> tuple:
    model.eval()
    scores, labels = [], []
    for f in test_fps:
        d = np.load(f)
        feat = torch.tensor(d["features"].astype(np.float32)).reshape(1, -1).to(DEVICE)
        err = model.reconstruct_error(feat).item()
        scores.append(err)
        labels.append(float(d["is_anomaly"]))
    return np.array(labels), np.array(scores)


@torch.no_grad()
def score_lstm_ae(model: LSTMAutoencoder, test_fps: list[Path]) -> tuple:
    model.eval()
    scores, labels = [], []
    for f in test_fps:
        d = np.load(f)
        feat = torch.tensor(d["features"].astype(np.float32)).unsqueeze(0).to(DEVICE)
        err = model.reconstruct_error(feat).item()
        scores.append(err)
        labels.append(float(d["is_anomaly"]))
    return np.array(labels), np.array(scores)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    t0 = time.time()

    logger.info("BioMotion SOTA benchmark: Deep AE + LSTM-AE vs PLOS CompBio 2024")
    logger.info(f"Device: {DEVICE}")

    normal_fps, all_train_fps, val_fps, test_fps = load_split()
    logger.info(f"Test set: {len(test_fps)} trajectories")

    # Load existing results
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            results_list = json.load(f)
        if isinstance(results_list, list):
            results_existing = results_list
        else:
            results_existing = results_list.get("results", [])
    else:
        results_existing = []

    new_results = []

    # ── Deep Autoencoder (PLOS CompBio 2024 style) ────────────────────────────
    logger.info(f"\n--- Deep Autoencoder (PLOS CompBio 2024 reimpl., {DAE_EPOCHS} epochs) ---")
    dae = train_dae(normal_fps, val_fps)
    labs_dae, scores_dae = score_dae(dae, test_fps)
    m_dae = metrics(labs_dae, scores_dae)
    logger.info(f"  DAE  AUROC={m_dae['auroc']:.4f}  F1={m_dae['f1']:.4f}")
    new_results.append({
        "method": "DeepAutoencoder (PLOS CompBio 2024 reimpl.)",
        **m_dae,
        "n_params": sum(p.numel() for p in dae.parameters()),
        "reference": "PMC10515950 — Deep Autoencoder for Zebrafish Behavioral Anomaly, PLOS CompBio 2024",
        "published_auroc_range": "0.740–0.922 (6 phase-specific models on 2,719 larvae)",
        "note": "Our reimpl. on 28,610 ECOTOX trajectories (10× larger dataset)",
    })

    # ── LSTM Autoencoder ──────────────────────────────────────────────────────
    logger.info(f"\n--- LSTM Autoencoder ({LSTM_AE_EPOCHS} epochs) ---")
    lstm_ae = train_lstm_ae(normal_fps, val_fps)
    labs_lae, scores_lae = score_lstm_ae(lstm_ae, test_fps)
    m_lae = metrics(labs_lae, scores_lae)
    logger.info(f"  LSTM-AE  AUROC={m_lae['auroc']:.4f}  F1={m_lae['f1']:.4f}")
    new_results.append({
        "method": "LSTMAutoencoder",
        **m_lae,
        "n_params": sum(p.numel() for p in lstm_ae.parameters()),
        "note": "BiLSTM encoder → bottleneck(32) → LSTM decoder, trained normal-only",
    })

    # Merge with existing
    all_results = results_existing + new_results
    elapsed = time.time() - t0

    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            full_json = json.load(f)
        if isinstance(full_json, dict):
            full_json["results"] = all_results
        else:
            full_json = {"results": all_results}
    else:
        full_json = {"results": all_results}

    with open(RESULTS_PATH, "w") as f:
        json.dump(full_json, f, indent=2)
    logger.info(f"\nResults saved to {RESULTS_PATH}  (elapsed {elapsed:.0f}s)")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("BioMotion vs. Published SOTA — Behavioral Trajectory Anomaly (AUROC)")
    print(f"  {'Model':<42} {'AUROC':>8}")
    print("-" * 70)
    all_rows = [(r.get("method", "?"), r.get("auroc", 0)) for r in all_results]
    for name, auroc in sorted(all_rows, key=lambda x: x[1], reverse=True):
        ref = " ← published SOTA" if "PLOS" in name else ""
        print(f"  {name:<42} {auroc:.4f}{ref}")
    print("\n  Published best (PLOS CompBio 2024): AUROC 0.740–0.922 (2,719 samples)")
    print("=" * 70)


if __name__ == "__main__":
    main()
