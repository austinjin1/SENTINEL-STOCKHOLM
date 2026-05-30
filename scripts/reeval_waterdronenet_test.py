#!/usr/bin/env python3
"""Re-evaluate WaterDroneNet on the test set using the saved best checkpoint.

This script loads the best checkpoint and re-runs ONLY the test evaluation
(no retraining) with the fixed evaluate() function that filters sentinel
values and extreme outliers from USGS NWIS data before computing metrics.

Usage:
    PYTHONNOUSERSITE=1 conda run -n physiformer python scripts/reeval_waterdronenet_test.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Reuse everything from the training script
# ---------------------------------------------------------------------------
PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))

from train_waterdronenet import (
    ALL_PREDICT_COLS,
    CKPT_DIR,
    PARAM_RANGES,
    RESULTS_DIR,
    SCALAR_COLS,
    WaterDroneNetDataset,
    _load_waterdronenet_model,
    evaluate,
    log,
    setup_logging,
)

LOG_DIR = PROJECT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    log_path = LOG_DIR / "reeval_waterdronenet_test.log"
    setup_logging(log_path)

    # Device
    gpu = 1
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
        log("WARNING: CUDA not available; running on CPU.")

    torch.manual_seed(42)
    np.random.seed(42)

    log("=" * 65)
    log("WaterDroneNet — Test Set Re-evaluation (fixed sentinel filtering)")
    log("=" * 65)
    log(f"Device: {device}")

    # ------------------------------------------------------------------
    # Load test dataset
    # ------------------------------------------------------------------
    log("\n--- Loading test dataset ---")
    test_ds = WaterDroneNetDataset(split="test", seed=42)
    log(f"Test: {len(test_ds)} samples")

    # Also load train/val sizes for the results JSON
    train_ds = WaterDroneNetDataset(split="train", seed=42)
    val_ds = WaterDroneNetDataset(split="val", seed=42)
    log(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    batch_size = 32
    num_workers = min(4, os.cpu_count() or 1)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    # ------------------------------------------------------------------
    # Model + checkpoint
    # ------------------------------------------------------------------
    log("\n--- Loading model and checkpoint ---")
    model = _load_waterdronenet_model().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"WaterDroneNet parameters: {n_params:,}")

    best_ckpt_path = CKPT_DIR / "waterdronenet_best.pt"
    if not best_ckpt_path.exists():
        log(f"ERROR: Best checkpoint not found at {best_ckpt_path}")
        sys.exit(1)

    ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    best_epoch = ckpt["epoch"]
    best_val_score = float(ckpt.get("val_scalar",
                          ckpt["metrics"].get("best_val_score",
                          ckpt["metrics"].get("val_score", 0.0))))
    log(f"  Loaded best checkpoint from epoch {best_epoch}")

    # ------------------------------------------------------------------
    # Test evaluation with fixed filtering
    # ------------------------------------------------------------------
    log("\n" + "=" * 65)
    log("Test Set Evaluation (with sentinel/outlier filtering)")
    log("=" * 65)

    test_metrics = evaluate(model, test_loader, device)

    log(f"\nTest NLL:   {test_metrics['nll']:.4f}")
    log(f"Trust flag accuracy (red flags): {test_metrics.get('trust_flag_accuracy', float('nan')):.3f}")
    log("\nPer-target metrics:")
    for col_name, m in test_metrics.get("per_target", {}).items():
        n_filt = m.get('n_filtered', 0)
        log(
            f"  {col_name:8s}: R2={m['r2']:.4f}  MAE={m['mae']:.4f}  "
            f"RMSE={m['rmse']:.4f}  Coverage90={m['coverage_90pct']:.3f}  "
            f"(n={m['n_valid']}, filtered={n_filt})"
        )

    # ------------------------------------------------------------------
    # Physics residual analysis (with clamping for numerical stability)
    # ------------------------------------------------------------------
    log("\n--- Physics Residual Analysis ---")
    model.eval()
    residual_contributions = {col: [] for col in ALL_PREDICT_COLS}
    prior_contributions = {col: [] for col in ALL_PREDICT_COLS}

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            scalars = batch["scalar"].to(device)
            with autocast("cuda"):
                out = model(images, scalars)
            mu_b = out["mu"].cpu().float().numpy()
            prior_b = out["physics_prior"].cpu().float().numpy()
            for t_idx, col_name in enumerate(ALL_PREDICT_COLS):
                mu_col = mu_b[:, t_idx]
                prior_col = prior_b[:, t_idx]
                # Filter out NaN/Inf predictions (from OOD scalar inputs)
                finite = np.isfinite(mu_col) & np.isfinite(prior_col)
                if finite.sum() == 0:
                    continue
                residual_contributions[col_name].append(
                    float(np.abs(mu_col[finite] - prior_col[finite]).mean())
                )
                prior_contributions[col_name].append(
                    float(np.abs(prior_col[finite]).mean())
                )

    log("  Mean |residual| vs |prior| per target:")
    residual_analysis = {}
    for col_name in ALL_PREDICT_COLS:
        mean_res = float(np.mean(residual_contributions[col_name]))
        mean_prior = float(np.mean(prior_contributions[col_name]))
        ratio = mean_res / (mean_prior + 1e-8)
        log(f"  {col_name:8s}: |residual|={mean_res:.4f}  |prior|={mean_prior:.4f}  ratio={ratio:.3f}")
        residual_analysis[col_name] = {
            "mean_residual": mean_res,
            "mean_prior": mean_prior,
            "residual_ratio": ratio,
        }

    # ------------------------------------------------------------------
    # Read previous results to preserve training_config
    # ------------------------------------------------------------------
    prev_results_path = RESULTS_DIR / "waterdronenet_holdout.json"
    prev_config = {
        "epochs": 100,
        "batch_size": 32,
        "lr": 5e-05,
        "weight_decay": 0.01,
        "scheduler": "cosine_annealing",
        "pretrain_epochs": 0,
        "loss": "masked_gaussian_nll + physics_constraint + calibration + trust",
    }
    if prev_results_path.exists():
        with open(prev_results_path) as fh:
            prev = json.load(fh)
            if "training_config" in prev:
                prev_config = prev["training_config"]
            if "best_epoch" in prev:
                best_epoch = prev["best_epoch"]
            if "best_val_score" in prev and prev["best_val_score"] is not None:
                best_val_score = prev["best_val_score"]

    # ------------------------------------------------------------------
    # Persist results
    # ------------------------------------------------------------------
    results = {
        "model": "WaterDroneNet",
        "n_params": int(n_params),
        "train_size": int(len(train_ds)),
        "val_size": int(len(val_ds)),
        "test_size": int(len(test_ds)),
        "best_epoch": int(best_epoch),
        "best_val_score": float(best_val_score),
        "targets": ALL_PREDICT_COLS,
        "scalar_inputs": SCALAR_COLS,
        "temporal_holdout": {
            "train": "< 2023-01-01",
            "val": "2023-01-01 \u2013 2024-01-01",
            "test": ">= 2024-01-01",
        },
        "test_metrics": test_metrics,
        "physics_residual_analysis": residual_analysis,
        "training_config": prev_config,
    }

    # Convert any remaining numpy scalars to Python floats for JSON
    def _to_serializable(obj):
        if isinstance(obj, dict):
            return {k: _to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_serializable(x) for x in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None  # JSON doesn't support NaN/Inf
        return obj

    results = _to_serializable(results)

    out_path = RESULTS_DIR / "waterdronenet_holdout.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)

    log(f"\nResults saved to {out_path}")
    log(f"Checkpoint used: {best_ckpt_path}")
    log("DONE")


if __name__ == "__main__":
    main()
