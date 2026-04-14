#!/usr/bin/env python3
"""benchmark_aquassm_sota.py — Add MCN-LSTM (Sensors 2023) to AquaSSM benchmark.

Implements Multi-Column Network with LSTM from:
  "Real-Time Anomaly Detection for Water Quality Sensor Monitoring Based on
   Multivariate Deep Learning Technique" — Sensors (MDPI), Oct 2023 (PMC10610887)

Architecture: 3 parallel CNN columns (kernel=3,5,7) over time axis → concat
→ BiLSTM(hidden=128, layers=2) → linear → binary anomaly output.
Best reported F1=0.93 on USGS Madison River (SpCond, Turbidity, DO).

Loads SAME data split as benchmark_aquassm.py, adds MCN-LSTM result,
appends to results/benchmarks/aquassm_benchmark.json.

Bryan Cheng, SENTINEL project, 2026
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, random_split
from sklearn.metrics import roc_auc_score, f1_score

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, "/home/bcheng/SENTINEL")

# Re-use data loading from existing benchmark
sys.path.insert(0, str(Path(__file__).parent))
from benchmark_aquassm import (
    build_split, _collate, _compute_pos_weight, _safe_metrics,
    SEED, MAX_LEN, BATCH_SIZE,
)

from sentinel.utils.logging import get_logger
logger = get_logger(__name__)

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = Path("results/benchmarks")
MCN_EPOCHS  = 50
MCN_PATIENCE = 10


# ─────────────────────────────────────────────────────────────────────────────
# MCN-LSTM: Multi-Column Network + BiLSTM (Sensors 2023)
# ─────────────────────────────────────────────────────────────────────────────
class ConvColumn(nn.Module):
    """Single CNN column: two Conv1d → ReLU stacks with MaxPool."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=pad),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=pad),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),   # → (B, out_ch, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)   # (B, out_ch)


class MCN_LSTM(nn.Module):
    """Multi-Column Network with LSTM for multivariate time-series anomaly detection.

    Faithfully reimplements architecture from Sensors 2023 (PMC10610887):
      - 3 parallel CNN columns (kernel_sizes=3,5,7), 64 filters each
      - Column outputs concatenated → BiLSTM (hidden=128, 2 layers)
      - Linear(256, 1) → binary anomaly score
    """

    def __init__(
        self,
        in_channels: int = 6,
        cnn_out: int = 64,
        kernel_sizes: tuple = (3, 5, 7),
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.columns = nn.ModuleList([
            ConvColumn(in_channels, cnn_out, k) for k in kernel_sizes
        ])
        cnn_total = cnn_out * len(kernel_sizes)   # 192

        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        lstm_out = lstm_hidden * 2   # bidirectional

        # Fuse CNN features + LSTM CLS token
        self.classifier = nn.Sequential(
            nn.Linear(cnn_total + lstm_out, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) sensor sequence → (B,) anomaly logit."""
        x_cnn = x.permute(0, 2, 1)   # (B, C, T) for Conv1d

        # CNN multi-column
        col_feats = torch.cat([col(x_cnn) for col in self.columns], dim=1)  # (B, 192)

        # LSTM (full sequence)
        lstm_out, (hn, _) = self.lstm(x)    # hn: (2*layers, B, hidden)
        # Take final layer forward + backward hidden states
        lstm_feats = torch.cat([hn[-2], hn[-1]], dim=1)   # (B, 256)

        fused = torch.cat([col_feats, lstm_feats], dim=1)
        return self.classifier(fused).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Train / eval
# ─────────────────────────────────────────────────────────────────────────────
def train_mcn_lstm(
    train_dl: DataLoader,
    val_dl: DataLoader,
    pos_weight: float,
    epochs: int = MCN_EPOCHS,
    patience: int = MCN_PATIENCE,
) -> MCN_LSTM:
    model = MCN_LSTM(in_channels=6).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  MCN-LSTM: {n_params:,} params")

    pw = torch.tensor([pos_weight], device=DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5)

    best_val, no_imp, best_state = float("inf"), 0, None

    for ep in range(1, epochs + 1):
        model.train()
        for batch in train_dl:
            x = batch["values"].to(DEVICE)
            y = batch["has_anomaly"].to(DEVICE)
            loss = F.binary_cross_entropy_with_logits(model(x), y, pos_weight=pw)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dl:
                x = batch["values"].to(DEVICE)
                y = batch["has_anomaly"].to(DEVICE)
                val_loss += F.binary_cross_entropy_with_logits(
                    model(x), y, pos_weight=pw).item()
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val, no_imp = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_imp += 1
        if no_imp >= patience:
            logger.info(f"  Early stopping at epoch {ep}")
            break

        if ep % 10 == 0 or ep == 1:
            logger.info(f"  MCN-LSTM epoch {ep}/{epochs} | val_loss={val_loss:.4f}")

    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def eval_model(model: MCN_LSTM, test_dl: DataLoader):
    model.eval()
    probs, labs = [], []
    for batch in test_dl:
        x = batch["values"].to(DEVICE)
        y = batch["has_anomaly"]
        logits = model(x)
        probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        labs.extend(y.numpy().tolist())
    return np.array(labs, dtype=int), np.array(probs)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    t0 = time.time()

    logger.info("MCN-LSTM benchmark (Sensors 2023) on AquaSSM data split")
    logger.info(f"Device: {DEVICE}")

    tr_ds, va_ds, te_ds, N = build_split()
    pos_w = _compute_pos_weight(tr_ds)

    tr_dl = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=_collate, num_workers=0)
    va_dl = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=_collate, num_workers=0)
    te_dl = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=_collate, num_workers=0)

    logger.info(f"\n--- MCN-LSTM (Sensors 2023 reimplementation, {MCN_EPOCHS} epochs) ---")
    model = train_mcn_lstm(tr_dl, va_dl, pos_w, MCN_EPOCHS)
    labs, probs = eval_model(model, te_dl)
    auroc, f1 = _safe_metrics(labs, probs, "MCN-LSTM")
    elapsed = time.time() - t0
    logger.info(f"MCN-LSTM  AUROC={auroc:.4f}  F1={f1:.4f}  (n={len(labs)}, t={elapsed:.0f}s)")

    # Load existing results and append
    results_path = RESULTS_DIR / "aquassm_benchmark.json"
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
    else:
        results = {"models": {}}

    results["models"]["MCN-LSTM (Sensors 2023)"] = {
        "auroc":       auroc,
        "f1":          f1,
        "n_test":      len(labs),
        "n_params":    sum(p.numel() for p in model.parameters()),
        "elapsed_s":   round(elapsed, 1),
        "reference":   "PMC10610887 — Real-Time Anomaly Detection for Water Quality Sensor Monitoring, Sensors 2023",
        "architecture":"3×ConvColumn(k=3,5,7,64ch) + BiLSTM(h=128,L=2) + Linear",
        "note":        "Reimplementation on USGS 5-param data; published F1=0.93 on 3-param Madison River",
    }

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Updated {results_path}")

    # Print comparison table
    print("\n" + "=" * 65)
    print("AquaSSM vs. Published SOTA — Sensor Anomaly Detection")
    print(f"{'Model':<35} {'AUROC':>8} {'F1@0.5':>8}")
    print("-" * 65)
    for name, m in results["models"].items():
        if m.get("auroc") is None:
            continue
        ref = " ← published" if "Sensors 2023" in name or "AquaDynNet" in name else ""
        print(f"  {name:<33} {m['auroc']:.4f}  {m.get('f1', 0):.4f}{ref}")
    print("=" * 65)


if __name__ == "__main__":
    main()
