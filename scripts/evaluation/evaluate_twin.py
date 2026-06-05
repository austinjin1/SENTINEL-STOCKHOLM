#!/usr/bin/env python3
"""Comprehensive evaluation of the Digital Twin model.

Computes per-variable and per-horizon metrics beyond MSE:
  - R² score (coefficient of determination)
  - MAE (mean absolute error in original units)
  - RMSE (root mean squared error)
  - Direction accuracy (does predicted change match actual change?)

Loads test data identically to train_twin.py and the best checkpoint.

Usage:
    CUDA_VISIBLE_DEVICES=2 conda run -n physiformer python scripts/evaluate_twin.py

MIT License — Bryan Cheng, 2026
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader

PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT))

# Reuse TwinDataset from train_twin — it handles both precomputed and raw data
# Import from train_twin (which lives in the same directory)
import importlib.util
_spec = importlib.util.spec_from_file_location("train_twin", PROJECT / "scripts" / "train_twin.py")
_train_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_train_mod)
TwinDataset = _train_mod.TwinDataset
STATE_VARS = _train_mod.STATE_VARS
FORECAST_HORIZONS = _train_mod.FORECAST_HORIZONS
NUM_STATES = _train_mod.NUM_STATES
log = _train_mod.log

CKPT_PATH = PROJECT / "checkpoints" / "twin" / "twin_best.pt"
RESULTS_DIR = PROJECT / "results" / "benchmarks"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# Physical units for each state variable (for interpretable reporting)
UNITS = {
    "dissolved_oxygen": "mg/L",
    "bod": "mg/L",
    "total_nitrogen": "mg/L",
    "total_phosphorus": "mg/L",
    "chlorophyll_a": "ug/L",
    "temperature": "°C",
    "ph": "pH",
    "turbidity": "NTU",
    "doc": "mg/L",
    "sediment": "mg/L",
}


def _valid_mask(predictions: np.ndarray, targets: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Build combined validity mask: observed AND finite in both pred and target."""
    return (mask > 0.5) & np.isfinite(predictions) & np.isfinite(targets)


def compute_r2(predictions: np.ndarray, targets: np.ndarray, mask: np.ndarray) -> float:
    """Compute R² score on masked entries (skipping NaN/inf)."""
    valid = _valid_mask(predictions, targets, mask)
    if valid.sum() < 2:
        return float("nan")
    y_true = targets[valid]
    y_pred = predictions[valid]
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-12:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def compute_mae(predictions: np.ndarray, targets: np.ndarray, mask: np.ndarray) -> float:
    """Compute MAE on masked entries (skipping NaN/inf)."""
    valid = _valid_mask(predictions, targets, mask)
    if valid.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(predictions[valid] - targets[valid])))


def compute_rmse(predictions: np.ndarray, targets: np.ndarray, mask: np.ndarray) -> float:
    """Compute RMSE on masked entries (skipping NaN/inf)."""
    valid = _valid_mask(predictions, targets, mask)
    if valid.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((predictions[valid] - targets[valid]) ** 2)))


def compute_direction_accuracy(
    predictions: np.ndarray,
    targets: np.ndarray,
    initial_states: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Compute direction accuracy: does predicted change direction match actual?

    For each valid observation, we check whether the predicted direction of
    change from the initial state (increase vs. decrease) matches the actual
    direction. Ties (no change) are counted as correct if both agree.
    """
    valid = _valid_mask(predictions, targets, mask) & np.isfinite(initial_states)
    if valid.sum() == 0:
        return float("nan")

    pred_delta = predictions[valid] - initial_states[valid]
    true_delta = targets[valid] - initial_states[valid]

    # Direction matches if signs agree (or both are zero)
    pred_sign = np.sign(pred_delta)
    true_sign = np.sign(true_delta)
    matches = (pred_sign == true_sign).astype(np.float32)

    return float(np.mean(matches))


def compute_coverage_90(
    lower: np.ndarray, upper: np.ndarray,
    targets: np.ndarray, mask: np.ndarray,
) -> float:
    """Compute 90% CI coverage: fraction of true values within predicted interval."""
    valid = (mask > 0.5) & np.isfinite(lower) & np.isfinite(upper) & np.isfinite(targets)
    if valid.sum() == 0:
        return float("nan")
    in_interval = (targets[valid] >= lower[valid]) & (targets[valid] <= upper[valid])
    return float(np.mean(in_interval))


def main():
    device_id = 0  # CUDA_VISIBLE_DEVICES remaps to device 0
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)

    log("=" * 70)
    log("Digital Twin — Comprehensive Evaluation")
    log("=" * 70)
    log(f"Device: {device}")
    log(f"Checkpoint: {CKPT_PATH}")

    # -----------------------------------------------------------------------
    # 1. Load model
    # -----------------------------------------------------------------------
    from sentinel.models.twin.twin_engine import DigitalTwinEngine

    model = DigitalTwinEngine().to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    log(f"Model parameters: {n_params:,}")
    log(f"Checkpoint epoch: {ckpt['epoch']}, phase: {ckpt['phase']}")

    # -----------------------------------------------------------------------
    # 2. Load test data (same split logic as train_twin.py)
    # -----------------------------------------------------------------------
    log("\nLoading test dataset...")
    test_ds = TwinDataset(split="test", seed=42)
    test_loader = DataLoader(
        test_ds, batch_size=32, shuffle=False,
        num_workers=2, pin_memory=True,
    )
    log(f"Test samples: {len(test_ds)}")

    horizons = FORECAST_HORIZONS
    num_horizons = len(horizons)

    # -----------------------------------------------------------------------
    # 3. Collect all predictions, targets, masks (with caching)
    # -----------------------------------------------------------------------
    cache_path = RESULTS_DIR / "twin_eval_cache.npz"

    if cache_path.exists():
        log(f"\nLoading cached predictions from {cache_path}")
        cache = np.load(cache_path)
        preds_all = cache["preds"]
        lower_all = cache["lower"]
        upper_all = cache["upper"]
        targets_all = cache["targets"]
        masks_all = cache["masks"]
        init_all = cache["init_states"]
        physics_all = cache["physics"]
        inference_time = float(cache.get("inference_time", 0.0))
        N = preds_all.shape[1]
        log(f"Cached: {N} samples, inference took {inference_time:.1f}s originally")
    else:
        log("\nRunning inference on test set...")
        all_preds = []       # [H, B, D]
        all_lower = []       # [H, B, D]
        all_upper = []       # [H, B, D]
        all_targets = []     # [H, B, D]
        all_masks = []       # [H, B, D]
        all_init_states = [] # [B, D]
        all_physics = []     # [H, B, D]
        n_batches = 0

        t_start = time.time()
        for batch in test_loader:
            emb = batch["embedding"].to(device)
            init_state = batch["initial_state"].to(device)
            future = batch["future_states"].to(device)      # [B, H, D]
            mask = batch["future_mask"].to(device)           # [B, H, D]

            try:
                with torch.no_grad():
                    output = model(emb.float(), horizons=horizons, state_override=init_state.float())

                preds = output.predictions.float()    # [H, B, D]
                lower_90 = output.lower_90.float()
                upper_90 = output.upper_90.float()
                future_t = future.permute(1, 0, 2).float()    # [H, B, D]
                mask_t = mask.permute(1, 0, 2).float()         # [H, B, D]

                # Physics-only predictions extracted from the same forward pass:
                # physics_trajectory is the raw ODE output [T, B, D] including t=0.
                # We skip t=0 to get predictions at horizon time-points.
                physics_at_horizons = output.physics_trajectory[1:].float()  # [H, B, D]

                all_preds.append(preds.cpu().numpy())
                all_lower.append(lower_90.cpu().numpy())
                all_upper.append(upper_90.cpu().numpy())
                all_targets.append(future_t.cpu().numpy())
                all_masks.append(mask_t.cpu().numpy())
                all_init_states.append(init_state.cpu().numpy())
                all_physics.append(physics_at_horizons.cpu().numpy())
                n_batches += 1

                if n_batches % 100 == 0:
                    log(f"  ... processed {n_batches} batches ({n_batches * 32} samples)")

            except (RuntimeError, AssertionError) as e:
                log(f"  WARNING: batch failed: {e}")
                continue

        inference_time = time.time() - t_start
        log(f"Inference complete: {n_batches} batches, {inference_time:.1f}s")

        # Concatenate along batch dimension
        preds_all = np.concatenate(all_preds, axis=1)       # [H, N, D]
        lower_all = np.concatenate(all_lower, axis=1)
        upper_all = np.concatenate(all_upper, axis=1)
        targets_all = np.concatenate(all_targets, axis=1)   # [H, N, D]
        masks_all = np.concatenate(all_masks, axis=1)       # [H, N, D]
        init_all = np.concatenate(all_init_states, axis=0)  # [N, D]
        physics_all = np.concatenate(all_physics, axis=1)   # [H, N, D]

        # Save cache
        np.savez_compressed(
            cache_path,
            preds=preds_all, lower=lower_all, upper=upper_all,
            targets=targets_all, masks=masks_all,
            init_states=init_all, physics=physics_all,
            inference_time=np.array(inference_time),
        )
        log(f"Cached predictions to {cache_path}")

    N = preds_all.shape[1]
    log(f"Total test samples: {N}")
    log(f"Observed entries: {int(masks_all.sum())} / {masks_all.size} "
        f"({100 * masks_all.sum() / masks_all.size:.1f}%)")

    # Diagnostics: check for NaN/inf in predictions
    observed_mask = masks_all > 0.5
    n_observed = observed_mask.sum()
    n_pred_finite = np.isfinite(preds_all[observed_mask]).sum()
    n_pred_nan = np.isnan(preds_all[observed_mask]).sum()
    n_pred_inf = np.isinf(preds_all[observed_mask]).sum()
    log(f"Prediction quality on observed entries:")
    log(f"  Finite: {n_pred_finite}/{n_observed} ({100*n_pred_finite/n_observed:.1f}%)")
    log(f"  NaN: {n_pred_nan}/{n_observed} ({100*n_pred_nan/n_observed:.1f}%)")
    log(f"  Inf: {n_pred_inf}/{n_observed} ({100*n_pred_inf/n_observed:.1f}%)")

    n_phys_finite = np.isfinite(physics_all[observed_mask]).sum()
    log(f"Physics-only finite: {n_phys_finite}/{n_observed} ({100*n_phys_finite/n_observed:.1f}%)")

    # Prediction range statistics (finite values only)
    finite_preds = preds_all[observed_mask & np.isfinite(preds_all)]
    if len(finite_preds) > 0:
        log(f"Prediction range (finite): [{np.min(finite_preds):.2f}, {np.max(finite_preds):.2f}]")
        log(f"  Mean: {np.mean(finite_preds):.4f}, Std: {np.std(finite_preds):.4f}")

    finite_targets = targets_all[observed_mask & np.isfinite(targets_all)]
    if len(finite_targets) > 0:
        log(f"Target range: [{np.min(finite_targets):.2f}, {np.max(finite_targets):.2f}]")
        log(f"  Mean: {np.mean(finite_targets):.4f}, Std: {np.std(finite_targets):.4f}")

    # -----------------------------------------------------------------------
    # 4. Compute metrics
    # -----------------------------------------------------------------------
    log("\n" + "=" * 70)
    log("Computing metrics...")
    log("=" * 70)

    # Expand init_all to match [H, N, D] for direction accuracy
    init_expanded = np.broadcast_to(init_all[np.newaxis, :, :], preds_all.shape)

    # --- Overall metrics ---
    overall = {
        "r2": compute_r2(preds_all, targets_all, masks_all),
        "mae": compute_mae(preds_all, targets_all, masks_all),
        "rmse": compute_rmse(preds_all, targets_all, masks_all),
        "direction_accuracy": compute_direction_accuracy(
            preds_all, targets_all, init_expanded, masks_all
        ),
        "coverage_90": compute_coverage_90(lower_all, upper_all, targets_all, masks_all),
    }

    # Physics-only baseline overall
    physics_overall = {
        "r2": compute_r2(physics_all, targets_all, masks_all),
        "mae": compute_mae(physics_all, targets_all, masks_all),
        "rmse": compute_rmse(physics_all, targets_all, masks_all),
        "direction_accuracy": compute_direction_accuracy(
            physics_all, targets_all, init_expanded, masks_all
        ),
    }

    # --- Per-variable metrics ---
    per_variable = {}
    for v_idx, var_name in enumerate(STATE_VARS):
        p = preds_all[:, :, v_idx]
        t = targets_all[:, :, v_idx]
        m = masks_all[:, :, v_idx]
        ini = init_expanded[:, :, v_idx]
        lo = lower_all[:, :, v_idx]
        up = upper_all[:, :, v_idx]
        phys = physics_all[:, :, v_idx]

        n_obs = int(m.sum())
        n_finite = int(_valid_mask(p, t, m).sum())

        per_variable[var_name] = {
            "r2": compute_r2(p, t, m),
            "mae": compute_mae(p, t, m),
            "rmse": compute_rmse(p, t, m),
            "direction_accuracy": compute_direction_accuracy(p, t, ini, m),
            "coverage_90": compute_coverage_90(lo, up, t, m),
            "n_observations": n_obs,
            "n_finite_predictions": n_finite,
            "unit": UNITS[var_name],
            # Physics-only baseline for comparison
            "physics_r2": compute_r2(phys, t, m),
            "physics_mae": compute_mae(phys, t, m),
            "physics_rmse": compute_rmse(phys, t, m),
        }

    # --- Per-horizon metrics ---
    per_horizon = {}
    for h_idx, h in enumerate(horizons):
        p = preds_all[h_idx]
        t = targets_all[h_idx]
        m = masks_all[h_idx]
        ini = init_all  # [N, D]
        lo = lower_all[h_idx]
        up = upper_all[h_idx]
        phys = physics_all[h_idx]

        per_horizon[f"{h}d"] = {
            "r2": compute_r2(p, t, m),
            "mae": compute_mae(p, t, m),
            "rmse": compute_rmse(p, t, m),
            "direction_accuracy": compute_direction_accuracy(p, t, ini, m),
            "coverage_90": compute_coverage_90(lo, up, t, m),
            "n_observations": int(m.sum()),
            # Physics baseline
            "physics_r2": compute_r2(phys, t, m),
            "physics_mae": compute_mae(phys, t, m),
            "physics_rmse": compute_rmse(phys, t, m),
        }

    # --- Per-variable per-horizon breakdown ---
    var_horizon = {}
    for v_idx, var_name in enumerate(STATE_VARS):
        var_horizon[var_name] = {}
        for h_idx, h in enumerate(horizons):
            p = preds_all[h_idx, :, v_idx]
            t = targets_all[h_idx, :, v_idx]
            m = masks_all[h_idx, :, v_idx]
            ini = init_all[:, v_idx]
            phys = physics_all[h_idx, :, v_idx]

            n_obs = int(m.sum())
            if n_obs < 2:
                continue

            var_horizon[var_name][f"{h}d"] = {
                "r2": compute_r2(p, t, m),
                "mae": compute_mae(p, t, m),
                "rmse": compute_rmse(p, t, m),
                "direction_accuracy": compute_direction_accuracy(p, t, ini, m),
                "n_observations": n_obs,
                "physics_r2": compute_r2(phys, t, m),
            }

    # --- Naive baseline: always predict initial state ---
    naive_r2 = compute_r2(init_expanded, targets_all, masks_all)
    naive_mae = compute_mae(init_expanded, targets_all, masks_all)
    naive_rmse = compute_rmse(init_expanded, targets_all, masks_all)

    # -----------------------------------------------------------------------
    # 5. Print summary tables
    # -----------------------------------------------------------------------
    log("\n" + "=" * 70)
    log("OVERALL METRICS")
    log("=" * 70)
    log(f"  {'Metric':<25s} {'Hybrid':<12s} {'Physics-Only':<12s} {'Naive (t=0)':<12s}")
    log(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12}")
    log(f"  {'R²':<25s} {overall['r2']:>10.4f}   {physics_overall['r2']:>10.4f}   {naive_r2:>10.4f}")
    log(f"  {'MAE':<25s} {overall['mae']:>10.4f}   {physics_overall['mae']:>10.4f}   {naive_mae:>10.4f}")
    log(f"  {'RMSE':<25s} {overall['rmse']:>10.4f}   {physics_overall['rmse']:>10.4f}   {naive_rmse:>10.4f}")
    log(f"  {'Direction Accuracy':<25s} {overall['direction_accuracy']:>9.1%}   {physics_overall['direction_accuracy']:>9.1%}   {'N/A':>11s}")
    log(f"  {'90% CI Coverage':<25s} {overall['coverage_90']:>9.1%}   {'N/A':>11s}   {'N/A':>11s}")

    log(f"\n  Test samples: {N}")
    log(f"  Valid observations: {int(masks_all.sum())}")

    log("\n" + "=" * 70)
    log("PER-VARIABLE METRICS (Hybrid model)")
    log("=" * 70)
    log(f"  {'Variable':<20s} {'Unit':<7s} {'R²':>8s} {'MAE':>10s} {'RMSE':>10s} {'DirAcc':>8s} {'Cov90':>7s} {'N':>8s} {'PhysR²':>8s}")
    log(f"  {'-'*20} {'-'*7} {'-'*8} {'-'*10} {'-'*10} {'-'*8} {'-'*7} {'-'*8} {'-'*8}")

    # Sort variables by R² to highlight best vs worst
    sorted_vars = sorted(per_variable.items(), key=lambda x: x[1]["r2"] if not np.isnan(x[1]["r2"]) else -999, reverse=True)

    for var_name, m in sorted_vars:
        r2_str = f"{m['r2']:.4f}" if not np.isnan(m['r2']) else "N/A"
        mae_str = f"{m['mae']:.4f}" if not np.isnan(m['mae']) else "N/A"
        rmse_str = f"{m['rmse']:.4f}" if not np.isnan(m['rmse']) else "N/A"
        da_str = f"{m['direction_accuracy']:.1%}" if not np.isnan(m['direction_accuracy']) else "N/A"
        cov_str = f"{m['coverage_90']:.1%}" if not np.isnan(m['coverage_90']) else "N/A"
        pr2_str = f"{m['physics_r2']:.4f}" if not np.isnan(m['physics_r2']) else "N/A"
        log(f"  {var_name:<20s} {m['unit']:<7s} {r2_str:>8s} {mae_str:>10s} {rmse_str:>10s} {da_str:>8s} {cov_str:>7s} {m['n_observations']:>8d} {pr2_str:>8s}")

    log("\n" + "=" * 70)
    log("PER-HORIZON METRICS (Hybrid model)")
    log("=" * 70)
    log(f"  {'Horizon':<10s} {'R²':>8s} {'MAE':>10s} {'RMSE':>10s} {'DirAcc':>8s} {'Cov90':>7s} {'N':>8s} {'PhysR²':>8s}")
    log(f"  {'-'*10} {'-'*8} {'-'*10} {'-'*10} {'-'*8} {'-'*7} {'-'*8} {'-'*8}")

    for h in horizons:
        m = per_horizon[f"{h}d"]
        r2_str = f"{m['r2']:.4f}" if not np.isnan(m['r2']) else "N/A"
        mae_str = f"{m['mae']:.4f}" if not np.isnan(m['mae']) else "N/A"
        rmse_str = f"{m['rmse']:.4f}" if not np.isnan(m['rmse']) else "N/A"
        da_str = f"{m['direction_accuracy']:.1%}" if not np.isnan(m['direction_accuracy']) else "N/A"
        cov_str = f"{m['coverage_90']:.1%}" if not np.isnan(m['coverage_90']) else "N/A"
        pr2_str = f"{m['physics_r2']:.4f}" if not np.isnan(m['physics_r2']) else "N/A"
        log(f"  {h:>4d}d     {r2_str:>8s} {mae_str:>10s} {rmse_str:>10s} {da_str:>8s} {cov_str:>7s} {m['n_observations']:>8d} {pr2_str:>8s}")

    log("\n" + "=" * 70)
    log("PER-VARIABLE PER-HORIZON R² (Hybrid model)")
    log("=" * 70)
    header = f"  {'Variable':<20s}" + "".join(f" {h:>6d}d" for h in horizons)
    log(header)
    log("  " + "-" * (20 + 7 * len(horizons)))
    for var_name in STATE_VARS:
        if var_name not in var_horizon or not var_horizon[var_name]:
            continue
        row = f"  {var_name:<20s}"
        for h in horizons:
            key = f"{h}d"
            if key in var_horizon[var_name]:
                r2 = var_horizon[var_name][key]["r2"]
                r2_str = f"{r2:.3f}" if not np.isnan(r2) else "  N/A"
            else:
                r2_str = "  N/A"
            row += f" {r2_str:>7s}"
        log(row)

    # -----------------------------------------------------------------------
    # 6. Qualitative summary
    # -----------------------------------------------------------------------
    log("\n" + "=" * 70)
    log("QUALITATIVE SUMMARY")
    log("=" * 70)

    # Variables model predicts well (R² > 0.5)
    good_vars = [(v, m["r2"]) for v, m in per_variable.items() if not np.isnan(m["r2"]) and m["r2"] > 0.5]
    mediocre_vars = [(v, m["r2"]) for v, m in per_variable.items() if not np.isnan(m["r2"]) and 0.0 <= m["r2"] <= 0.5]
    poor_vars = [(v, m["r2"]) for v, m in per_variable.items() if not np.isnan(m["r2"]) and m["r2"] < 0.0]
    no_data = [(v, m["r2"]) for v, m in per_variable.items() if np.isnan(m["r2"])]

    if good_vars:
        log(f"\n  GOOD (R² > 0.5):")
        for v, r2 in sorted(good_vars, key=lambda x: x[1], reverse=True):
            log(f"    {v}: R²={r2:.4f}")

    if mediocre_vars:
        log(f"\n  MEDIOCRE (0 < R² < 0.5):")
        for v, r2 in sorted(mediocre_vars, key=lambda x: x[1], reverse=True):
            log(f"    {v}: R²={r2:.4f}")

    if poor_vars:
        log(f"\n  POOR (R² < 0, worse than predicting the mean):")
        for v, r2 in sorted(poor_vars, key=lambda x: x[1], reverse=True):
            log(f"    {v}: R²={r2:.4f}")

    if no_data:
        log(f"\n  NO DATA:")
        for v, _ in no_data:
            log(f"    {v}")

    # Neural corrector improvement
    hybrid_r2 = overall["r2"]
    physics_r2 = physics_overall["r2"]
    log(f"\n  Neural corrector: R² {physics_r2:.4f} -> {hybrid_r2:.4f} "
        f"(improvement: {hybrid_r2 - physics_r2:+.4f})")

    # Horizon degradation
    log(f"\n  Forecast degradation:")
    for h in horizons:
        m = per_horizon[f"{h}d"]
        log(f"    {h:>4d}d: R²={m['r2']:.4f}, MAE={m['mae']:.4f}")

    # -----------------------------------------------------------------------
    # 7. Save results
    # -----------------------------------------------------------------------
    results = {
        "model": "DigitalTwinEngine",
        "checkpoint": str(CKPT_PATH),
        "checkpoint_epoch": ckpt["epoch"],
        "checkpoint_phase": ckpt["phase"],
        "n_params": n_params,
        "test_samples": N,
        "total_observations": int(masks_all.sum()),
        "inference_time_seconds": round(inference_time, 2),
        "overall": overall,
        "physics_only_baseline": physics_overall,
        "naive_baseline": {
            "r2": naive_r2,
            "mae": naive_mae,
            "rmse": naive_rmse,
        },
        "per_variable": per_variable,
        "per_horizon": per_horizon,
        "per_variable_per_horizon": var_horizon,
        "state_variables": list(STATE_VARS),
        "forecast_horizons": list(horizons),
    }

    out_path = RESULTS_DIR / "twin_evaluation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    log(f"\nResults saved to {out_path}")
    log("DONE")


if __name__ == "__main__":
    main()
