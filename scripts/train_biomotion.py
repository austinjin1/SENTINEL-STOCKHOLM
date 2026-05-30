#!/usr/bin/env python3
"""
BioMotion v2: Spatial-holdout training on REAL ECOTOX behavioral data only.

Key differences from train_biomotion.py:
  - Strict spatial holdout splits via sentinel.data.splits (no random splits)
  - Hash-based test_id assignment: folds 0-2 train, fold 3 val, fold 4 test
  - Zero test_id leakage verified at startup
  - NO synthetic data — only real EPA ECOTOX behavioral trajectories
  - Results saved to results/benchmarks/biomotion_v2_holdout.json

Architecture (unchanged):
  Phase 1: Diffusion pretraining on NORMAL trajectories from train split
  Phase 2: Anomaly classification on ALL trajectories from train split
  Evaluation: AUROC, F1, precision, recall + diffusion anomaly score

Usage:
    CUDA_VISIBLE_DEVICES=3 python scripts/train_biomotion_v2.py
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sentinel.data.splits import split_indices_spatial_only, assign_spatial_fold, SplitConfig
from sentinel.models.biomotion.trajectory_encoder import TrajectoryDiffusionEncoder, EMBED_DIM
from sentinel.models.biomotion.multi_organism import SPECIES_FEATURE_DIM

# ── Config ─────────────────────────────────────────────────────────────────
DATA_DIR_PRIMARY = PROJECT_ROOT / "data" / "processed" / "behavioral_fullreal"
DATA_DIR_FALLBACK = PROJECT_ROOT / "data" / "processed" / "behavioral_real"
CKPT_DIR = PROJECT_ROOT / "checkpoints" / "biomotion"
RESULTS_DIR = PROJECT_ROOT / "results" / "benchmarks"

FEATURE_DIM = SPECIES_FEATURE_DIM["daphnia"]   # 16
N_KEYPOINTS = 12
BATCH_SIZE = 64
SEED = 42

# Phase 1: diffusion pretraining on normal trajectories
P1_EPOCHS = 50
P1_LR = 2e-4
P1_WARMUP = 400

# Phase 2: supervised classification fine-tuning
P2_EPOCHS = 50
P2_LR = 5e-5
P2_HEAD_LR_MULT = 5       # head lr = P2_LR * 5 = 2.5e-4
P2_WARMUP = 200
EARLY_STOP_PATIENCE = 10

# Filename pattern: traj_{test_id}_{idx}.npz
FILENAME_PATTERN = re.compile(r"traj_(.+?)_(\d+)\.npz$")
FILENAME_SIMPLE = re.compile(r"traj_(\d+)\.npz$")


# ── Dataset ────────────────────────────────────────────────────────────────

class BehavioralDataset(Dataset):
    """In-memory cached dataset for ECOTOX behavioral trajectories."""

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
    def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
        return {
            "keypoints":  torch.from_numpy(np.stack([s["keypoints"]  for s in batch])),
            "features":   torch.from_numpy(np.stack([s["features"]   for s in batch])),
            "timestamps": torch.from_numpy(np.stack([s["timestamps"] for s in batch])),
            "labels":     torch.tensor([float(s["is_anomaly"]) for s in batch],
                                       dtype=torch.float32),
        }


# ── Data loading with spatial holdout ──────────────────────────────────────

def extract_test_id(filepath: Path) -> str:
    """Extract test_id from filename pattern traj_{test_id}_{idx}.npz or traj_{idx}.npz."""
    m = FILENAME_PATTERN.search(filepath.name)
    if m is not None:
        return m.group(1)
    m = FILENAME_SIMPLE.search(filepath.name)
    if m is not None:
        return m.group(1)
    # Last resort: use stem as identifier
    return filepath.stem


def resolve_data_dir() -> Path:
    """Find the real data directory, failing clearly if unavailable."""
    if DATA_DIR_PRIMARY.exists() and any(DATA_DIR_PRIMARY.glob("traj_*.npz")):
        return DATA_DIR_PRIMARY
    if DATA_DIR_FALLBACK.exists() and any(DATA_DIR_FALLBACK.glob("traj_*.npz")):
        print(f"[INFO] Primary data dir not found, using fallback: {DATA_DIR_FALLBACK}")
        return DATA_DIR_FALLBACK
    print(
        f"\n[ERROR] No real behavioral data found.\n"
        f"  Checked:\n"
        f"    {DATA_DIR_PRIMARY}\n"
        f"    {DATA_DIR_FALLBACK}\n\n"
        f"  Run the data preparation pipeline first:\n"
        f"    python scripts/expand_biomotion_data.py\n",
        file=sys.stderr,
    )
    sys.exit(1)


def verify_no_leakage(
    split_files: dict[str, list[Path]],
) -> None:
    """Assert zero test_id overlap between train/val/test splits."""
    split_ids: dict[str, set[str]] = {}
    for name, files in split_files.items():
        split_ids[name] = {extract_test_id(f) for f in files}

    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    for a, b in pairs:
        overlap = split_ids[a] & split_ids[b]
        if overlap:
            print(
                f"\n[FATAL] test_id leakage between {a} and {b}: "
                f"{sorted(overlap)[:10]}... ({len(overlap)} total)",
                file=sys.stderr,
            )
            sys.exit(1)

    print(f"  Leakage check PASSED: 0 test_id overlap across splits")
    for name, ids in split_ids.items():
        print(f"    {name}: {len(ids)} unique test_ids")


def load_data() -> tuple[
    BehavioralDataset, BehavioralDataset, BehavioralDataset, BehavioralDataset, Path
]:
    """Load and split real ECOTOX behavioral data using spatial holdout.

    Returns:
        (train_normal, train_all, val, test, data_dir)
    """
    data_dir = resolve_data_dir()
    all_files = sorted(data_dir.glob("traj_*.npz"))
    assert len(all_files) > 0, f"No trajectory files found in {data_dir}"
    print(f"\nFound {len(all_files)} trajectories in {data_dir}")

    # Extract test_id for each file (used as the "site" for spatial splitting)
    test_ids = [extract_test_id(f) for f in all_files]

    # Use split_indices_spatial_only with hash-based fold assignment
    # Folds 0,1,2 -> train | Fold 3 -> val | Fold 4 -> test
    config = SplitConfig(seed=SEED)
    idx_map = split_indices_spatial_only(test_ids, config=config)

    train_indices = idx_map["train"]
    val_indices = idx_map["val"]
    test_indices = idx_map["test"]

    train_files = [all_files[i] for i in train_indices]
    val_files = [all_files[i] for i in val_indices]
    test_files = [all_files[i] for i in test_indices]

    # Log fold distribution
    fold_counts: Counter[int] = Counter()
    for tid in test_ids:
        fold_counts[assign_spatial_fold(tid, config.num_spatial_folds, config.seed)] += 1
    print(f"\n  Fold distribution:")
    for fold in sorted(fold_counts):
        print(f"    Fold {fold}: {fold_counts[fold]} samples")

    # Verify zero leakage
    verify_no_leakage({"train": train_files, "val": val_files, "test": test_files})

    # Pre-scan to separate normal/anomaly within train split and log anomaly rates
    print(f"\n  Split sizes:")
    split_info: dict[str, dict[str, int]] = {}
    for name, files in [("train", train_files), ("val", val_files), ("test", test_files)]:
        n_anom = 0
        for f in files:
            d = np.load(f)
            if bool(d["is_anomaly"]):
                n_anom += 1
        n_norm = len(files) - n_anom
        rate = n_anom / max(len(files), 1)
        split_info[name] = {"normal": n_norm, "anomaly": n_anom, "total": len(files)}
        print(f"    {name:5s}: {len(files):5d} total | "
              f"{n_norm:5d} normal, {n_anom:5d} anomaly | "
              f"anomaly rate = {rate:.3f}")

    # Build train normal-only file list
    train_normal_files = []
    for f in train_files:
        d = np.load(f)
        if not bool(d["is_anomaly"]):
            train_normal_files.append(f)

    print(f"\n  Phase 1 (normal-only train): {len(train_normal_files)} trajectories")
    print(f"  Phase 2 (all train):         {len(train_files)} trajectories")

    return (
        BehavioralDataset(train_normal_files),    # phase-1: normal only
        BehavioralDataset(train_files),            # phase-2: all train
        BehavioralDataset(val_files),              # validation
        BehavioralDataset(test_files),             # held-out test
        data_dir,
    )


def make_loader(ds: BehavioralDataset, shuffle: bool, bs: int = BATCH_SIZE) -> DataLoader:
    return DataLoader(
        ds, batch_size=bs, shuffle=shuffle,
        collate_fn=BehavioralDataset.collate,
        num_workers=4, pin_memory=True,
        drop_last=shuffle,
    )


# ── LR schedule ───────────────────────────────────────────────────────────

def cosine_schedule(optimizer, total_steps: int, warmup: int):
    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        p = (step - warmup) / max(1, total_steps - warmup)
        return 0.01 + 0.5 * 0.99 * (1 + math.cos(math.pi * p))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Model ──────────────────────────────────────────────────────────────────

class AnomalyClassifier(nn.Module):
    """Encoder + classification head for anomaly detection."""

    def __init__(self, encoder: TrajectoryDiffusionEncoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(EMBED_DIM, EMBED_DIM // 2),      # 256 -> 128 -> but spec says Linear(256)
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(EMBED_DIM // 2, 1),
        )
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        emb = self.encoder.forward_encode(features)
        return self.classifier(emb).squeeze(-1)


# ── Phase 1: Diffusion pretraining ────────────────────────────────────────

def train_phase1(
    encoder: TrajectoryDiffusionEncoder,
    train_ds: BehavioralDataset,
    val_ds: BehavioralDataset,
    device: torch.device,
) -> dict:
    print("\n" + "=" * 70)
    print("PHASE 1: Diffusion Pretraining (normal train trajectories, 50 epochs)")
    print("=" * 70)

    encoder.to(device)
    tr_ld = make_loader(train_ds, shuffle=True)
    va_ld = make_loader(val_ds, shuffle=False)

    opt = torch.optim.AdamW(encoder.parameters(), lr=P1_LR, weight_decay=0.01)
    sch = cosine_schedule(opt, len(tr_ld) * P1_EPOCHS, P1_WARMUP)

    best_val = float("inf")
    hist: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    t0 = time.time()

    p1_ckpt = CKPT_DIR / "biomotion_v2_phase1_best.pt"

    for epoch in range(P1_EPOCHS):
        encoder.train()
        ep_loss = []
        for batch in tr_ld:
            feats = batch["features"].to(device)
            loss = encoder.compute_training_loss(feats)["loss"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            opt.step()
            sch.step()
            ep_loss.append(loss.item())
        tr_loss = float(np.mean(ep_loss))

        encoder.eval()
        va_loss = []
        with torch.no_grad():
            for batch in va_ld:
                feats = batch["features"].to(device)
                va_loss.append(encoder.compute_training_loss(feats)["loss"].item())
        val_loss = float(np.mean(va_loss)) if va_loss else float("inf")

        hist["train_loss"].append(tr_loss)
        hist["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {"epoch": epoch, "model_state_dict": encoder.state_dict(),
                 "val_loss": val_loss},
                p1_ckpt,
            )

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{P1_EPOCHS}  "
                  f"tr={tr_loss:.6f}  va={val_loss:.6f}  "
                  f"lr={opt.param_groups[0]['lr']:.2e}  "
                  f"time={time.time()-t0:.0f}s")

    print(f"Phase 1 done in {time.time()-t0:.0f}s | best_val_loss={best_val:.6f}")

    # Reload best
    state = torch.load(p1_ckpt, map_location=device, weights_only=False)
    encoder.load_state_dict(state["model_state_dict"])
    return {"best_val_loss": best_val, "best_epoch": state["epoch"], "history": hist}


# ── Phase 2: Anomaly classification ──────────────────────────────────────

def train_phase2(
    encoder: TrajectoryDiffusionEncoder,
    train_ds: BehavioralDataset,
    val_ds: BehavioralDataset,
    device: torch.device,
) -> tuple[AnomalyClassifier, dict]:
    print("\n" + "=" * 70)
    print("PHASE 2: Anomaly Classification (50 epochs, patience=10)")
    print("=" * 70)

    model = AnomalyClassifier(encoder)
    model.to(device)
    tr_ld = make_loader(train_ds, shuffle=True)
    va_ld = make_loader(val_ds, shuffle=False)

    param_groups = [
        {"params": model.encoder.parameters(), "lr": P2_LR},
        {"params": model.classifier.parameters(), "lr": P2_LR * P2_HEAD_LR_MULT},
    ]
    opt = torch.optim.AdamW(param_groups, weight_decay=0.01)
    sch = cosine_schedule(opt, len(tr_ld) * P2_EPOCHS, P2_WARMUP)

    best_val = float("inf")
    best_acc = 0.0
    no_improve = 0
    best_epoch = 0
    hist: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_acc": []}
    t0 = time.time()

    p2_ckpt = CKPT_DIR / "biomotion_v2_best.pt"

    for epoch in range(P2_EPOCHS):
        model.train()
        ep_loss = []
        for batch in tr_ld:
            feats = batch["features"].to(device)
            labels = batch["labels"].to(device)
            logits = model(feats)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            ep_loss.append(loss.item())
        tr_loss = float(np.mean(ep_loss))

        model.eval()
        va_losses, correct, total = [], 0, 0
        with torch.no_grad():
            for batch in va_ld:
                feats = batch["features"].to(device)
                labels = batch["labels"].to(device)
                logits = model(feats)
                va_losses.append(
                    F.binary_cross_entropy_with_logits(logits, labels).item()
                )
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct += (preds == labels).sum().item()
                total += len(labels)
        val_loss = float(np.mean(va_losses)) if va_losses else float("inf")
        val_acc = correct / max(total, 1)

        hist["train_loss"].append(tr_loss)
        hist["val_loss"].append(val_loss)
        hist["val_acc"].append(val_acc)

        if val_loss < best_val:
            best_val = val_loss
            best_acc = val_acc
            no_improve = 0
            best_epoch = epoch
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "val_loss": val_loss, "val_acc": val_acc},
                p2_ckpt,
            )
        else:
            no_improve += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{P2_EPOCHS}  "
                  f"tr={tr_loss:.4f}  va={val_loss:.4f}  acc={val_acc:.4f}  "
                  f"lr={opt.param_groups[0]['lr']:.2e}  "
                  f"patience={no_improve}/{EARLY_STOP_PATIENCE}")

        if no_improve >= EARLY_STOP_PATIENCE:
            print(f"\n  Early stopping at epoch {epoch+1} (best epoch {best_epoch+1})")
            break

    print(f"Phase 2 done in {time.time()-t0:.0f}s | "
          f"best_val={best_val:.4f}  acc={best_acc:.4f}")

    # Reload best
    state = torch.load(p2_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    return model, {
        "best_val_loss": best_val,
        "best_val_acc": best_acc,
        "best_epoch": best_epoch,
        "history": hist,
    }


# ── Evaluation ─────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: AnomalyClassifier,
    test_ds: BehavioralDataset,
    device: torch.device,
) -> dict:
    from sklearn.metrics import (
        roc_auc_score, f1_score, precision_score, recall_score, accuracy_score,
    )

    print("\n" + "=" * 70)
    print("TEST SET EVALUATION (held-out fold 4)")
    print("=" * 70)

    model.to(device)
    model.eval()
    ld = make_loader(test_ds, shuffle=False)

    y_true, y_scores, y_pred = [], [], []
    for batch in ld:
        feats = batch["features"].to(device)
        labels = batch["labels"]
        probs = torch.sigmoid(model(feats)).cpu().numpy()
        y_true.append(labels.numpy())
        y_scores.append(probs)
        y_pred.append((probs > 0.5).astype(float))

    y_true = np.concatenate(y_true)
    y_scores = np.concatenate(y_scores)
    y_pred = np.concatenate(y_pred)

    n_classes = len(np.unique(y_true))
    if n_classes < 2:
        print(f"\n  WARNING: Only {n_classes} class(es) in test set. "
              f"AUROC undefined. Using accuracy only.")
        auroc = float("nan")
        f1 = float("nan")
        precision = float("nan")
        recall = float("nan")
    else:
        auroc = float(roc_auc_score(y_true, y_scores))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall = float(recall_score(y_true, y_pred, zero_division=0))

    accuracy = float(accuracy_score(y_true, y_pred))

    print(f"\n  Samples:   {len(y_true)}")
    print(f"  Normal:    {int((y_true == 0).sum())}")
    print(f"  Anomalous: {int((y_true == 1).sum())}")
    print(f"\n  AUROC:     {auroc:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  Accuracy:  {accuracy:.4f}")

    # Diffusion-based anomaly score
    diff_scores_list = []
    for batch in ld:
        feats = batch["features"].to(device)
        ds_ = model.encoder.compute_anomaly_score(feats, num_noise_levels=5)
        diff_scores_list.append(ds_.cpu().numpy())
    diff_scores = np.concatenate(diff_scores_list)

    if n_classes >= 2:
        diff_auroc = float(roc_auc_score(y_true, diff_scores))
    else:
        diff_auroc = float("nan")
    print(f"  Diffusion AUROC: {diff_auroc:.4f}")

    return {
        "n_test": int(len(y_true)),
        "n_normal": int((y_true == 0).sum()),
        "n_anomalous": int((y_true == 1).sum()),
        "auroc": auroc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "diffusion_auroc": diff_auroc,
    }


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # Ensure output dirs exist
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data with spatial holdout
    train_normal, train_all, val_ds, test_ds, data_dir = load_data()

    # Build encoder
    encoder = TrajectoryDiffusionEncoder(
        feature_dim=FEATURE_DIM,
        embed_dim=EMBED_DIM,
        nhead=4,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.1,
    )
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"\nEncoder parameters: {n_params:,}")

    # Phase 1: diffusion pretraining on normal train trajectories
    p1_res = train_phase1(encoder, train_normal, val_ds, device)

    # Phase 2: supervised anomaly classification on all train trajectories
    model, p2_res = train_phase2(encoder, train_all, val_ds, device)

    # Evaluate on held-out test split (fold 4)
    test_res = evaluate(model, test_ds, device)

    # Save results
    results = {
        "model": "BioMotion v2 (TrajectoryDiffusionEncoder + AnomalyClassifier)",
        "split_protocol": "spatial_holdout",
        "split_method": "hash-based (sentinel.data.splits.split_indices_spatial_only)",
        "split_folds": {"train": [0, 1, 2], "val": [3], "test": [4]},
        "data_source": "REAL EPA ECOTOX behavioral data only (no synthetic)",
        "data_dir": str(data_dir),
        "species": "daphnia",
        "feature_dim": FEATURE_DIM,
        "embed_dim": EMBED_DIM,
        "n_parameters": n_params,
        "n_train_normal": len(train_normal),
        "n_train_all": len(train_all),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
        "phase1": p1_res,
        "phase2": p2_res,
        "test": test_res,
        "test_auroc": test_res["auroc"],
        "test_f1": test_res["f1"],
        "test_diffusion_auroc": test_res["diffusion_auroc"],
        "checkpoint": str(CKPT_DIR / "biomotion_v2_best.pt"),
    }

    results_path = RESULTS_DIR / "biomotion_v2_holdout.json"
    with open(results_path, "w") as fh:
        json.dump(results, fh, indent=2, default=str)

    print(f"\n{'=' * 70}")
    print(f"Results saved to {results_path}")
    print(f"Checkpoint at   {CKPT_DIR / 'biomotion_v2_best.pt'}")
    print(f"{'=' * 70}")
    print(f"  Split:           spatial holdout (folds 0-2 train / 3 val / 4 test)")
    print(f"  Data:            REAL only ({data_dir})")
    print(f"  AUROC:           {test_res['auroc']:.4f}")
    print(f"  F1:              {test_res['f1']:.4f}")
    print(f"  Precision:       {test_res['precision']:.4f}")
    print(f"  Recall:          {test_res['recall']:.4f}")
    print(f"  Accuracy:        {test_res['accuracy']:.4f}")
    print(f"  Diffusion AUROC: {test_res['diffusion_auroc']:.4f}")
    print(f"  n_train_normal:  {len(train_normal)}")
    print(f"  n_train_all:     {len(train_all)}")
    print(f"  n_val:           {len(val_ds)}")
    print(f"  n_test:          {len(test_ds)}")


if __name__ == "__main__":
    main()
