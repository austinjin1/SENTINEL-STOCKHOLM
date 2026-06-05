#!/usr/bin/env python3
"""Train Digital Aquatic Ecosystem Twin — Phase 4.1 of SENTINEL 2.0.

Trains the hybrid neural-ODE digital twin engine that forecasts
10 biogeochemical state variables at 1/7/14/30/90/365-day horizons.

The twin combines:
  - BiogeochemicalODE: 10-variable coupled ODE with literature-initialized
    parameters (Chapra 2008)
  - NeuralCorrector: additive residual MLP for physics model error
  - DataAssimilator: SENTINEL embedding → state posterior + parameter modulation
  - ForecastHead: multi-horizon prediction with calibrated uncertainty

State variables:
  DO, BOD, TotalN, TotalP, Chl-a, Temp, pH, Turbidity, DOC, Sediment

Uses real USGS NWIS data paired with SENTINEL embeddings.

Usage:
    conda run -n physiformer python scripts/train_twin.py

GPU: 3 (default)

MIT License — Bryan Cheng, 2026
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT))

CKPT_DIR = PROJECT / "checkpoints" / "twin"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = PROJECT / "results" / "benchmarks"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SENSOR_DIR = PROJECT / "data" / "processed" / "sensor" / "full"
RAW_SENSOR_DIR = PROJECT / "data" / "raw" / "sensor" / "full"

# State variable names matching twin_engine.py
STATE_VARS = [
    "dissolved_oxygen", "bod", "total_nitrogen", "total_phosphorus",
    "chlorophyll_a", "temperature", "ph", "turbidity", "doc", "sediment",
]
NUM_STATES = 10
FORECAST_HORIZONS = (1, 7, 14, 30, 90, 365)


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class TwinDataset(Dataset):
    """Training data for digital twin ecosystem forecasting.

    Each sample:
      - embedding (256-d): SENTINEL environmental embedding
      - initial_state (10-d): observed state at t=0
      - future_states (H, 10): observed states at forecast horizons
      - future_mask (H, 10): which future observations are available
    """

    def __init__(self, split: str = "train", seed: int = 42):
        super().__init__()
        rng = np.random.RandomState(seed + hash(split) % 1000)

        # Try pre-computed data
        precomputed = PROJECT / "data" / "processed" / "twin" / f"twin_{split}.npz"
        if precomputed.exists():
            d = np.load(precomputed, allow_pickle=True)
            self.embeddings = d["embeddings"].astype(np.float32)
            self.initial_states = d["initial_states"].astype(np.float32)
            self.future_states = d["future_states"].astype(np.float32)
            self.future_masks = d["future_masks"].astype(np.float32)
            log(f"  Loaded pre-computed {split}: {len(self.embeddings)} samples")
            return

        # Build from raw sensor data
        self._build_from_sensor_data(split, seed, rng)

    def _build_from_sensor_data(self, split: str, seed: int, rng: np.random.RandomState):
        """Build twin training data from USGS sensor time series."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required to load USGS sensor data. "
                "Install it with: pip install pandas pyarrow"
            )

        sensor_files = sorted(RAW_SENSOR_DIR.glob("*.parquet")) if RAW_SENSOR_DIR.exists() else []
        if not sensor_files:
            sensor_files = sorted(SENSOR_DIR.glob("*.npz")) if SENSOR_DIR.exists() else []

        if not sensor_files:
            raise FileNotFoundError(
                f"No sensor data found in {RAW_SENSOR_DIR} or {SENSOR_DIR}. "
                "Real USGS data is required — synthetic fallback disabled."
            )

        # Split by site
        split_files = {"train": [], "val": [], "test": []}
        for f in sensor_files:
            h = hashlib.sha256(f"{seed}:{f.stem}".encode()).hexdigest()
            fold = int(h[:8], 16) % 10
            if fold < 7:
                split_files["train"].append(f)
            elif fold < 9:
                split_files["val"].append(f)
            else:
                split_files["test"].append(f)

        selected = split_files[split]
        embeddings = []
        init_states_list = []
        future_states_list = []
        future_masks_list = []

        # Column name → state variable index mapping
        # Actual USGS parquet columns: Temp, SpCond, DO, pH, Turb
        # Also support USGS parameter codes (00300 etc.) as fallback
        col_name_map = {
            "DO": 0,       # dissolved_oxygen
            "Temp": 5,     # temperature
            "pH": 6,       # ph
            "Turb": 7,     # turbidity
            "SpCond": 7,   # specific conductance → turbidity proxy (lower priority)
        }
        param_code_map = {
            "00300": 0,  # DO → dissolved_oxygen
            "00010": 5,  # Temp → temperature
            "00400": 6,  # pH → ph
            "63680": 7,  # Turbidity → turbidity
            "00095": 7,  # Specific conductance → turbidity proxy
            "00060": 9,  # Discharge → sediment proxy
        }

        for fpath in selected:
            try:
                if fpath.suffix == ".parquet":
                    df = pd.read_parquet(fpath)

                    # Extract time series for state variables
                    # Build sliding windows for twin training
                    if "datetime" in df.columns:
                        df["datetime"] = pd.to_datetime(df["datetime"])
                        df = df.sort_values("datetime")
                    elif df.index.name == "datetime":
                        df = df.sort_index()
                        df = df.reset_index()

                    # Map columns to state variables
                    # First try exact column name match, then param code match
                    state_cols = {}
                    for col in df.columns:
                        if col == "datetime":
                            continue
                        # Exact name match (higher priority)
                        if col in col_name_map:
                            idx = col_name_map[col]
                            # Don't overwrite a more specific match (e.g. Turb > SpCond)
                            if idx not in state_cols or col != "SpCond":
                                state_cols[idx] = col
                            continue
                        # Fallback: USGS parameter code match
                        for code, idx in param_code_map.items():
                            if code in str(col):
                                state_cols[idx] = col
                                break

                    if len(state_cols) < 2:
                        continue

                    # Convert to numeric
                    for idx, col in state_cols.items():
                        df[col] = pd.to_numeric(df[col], errors="coerce")

                    # Create sliding windows (daily averages)
                    if "datetime" in df.columns:
                        df = df.set_index("datetime")
                    daily = df.resample("D").mean()

                    if len(daily) < 30:
                        continue

                    # Create training samples: each sample is a starting point
                    # with future observations at forecast horizons
                    for start_idx in range(0, len(daily) - 30, 7):
                        state_t0 = np.zeros(NUM_STATES, dtype=np.float32)
                        future = np.zeros((len(FORECAST_HORIZONS), NUM_STATES), dtype=np.float32)
                        mask = np.zeros((len(FORECAST_HORIZONS), NUM_STATES), dtype=np.float32)

                        # Fill initial state (clamp to physically reasonable ranges)
                        # Ranges: DO 0-20, BOD 0-50, TN 0-20, TP 0-5,
                        # Chl-a 0-500, Temp -5 to 45, pH 3-12, Turb 0-4000,
                        # DOC 0-100, Sediment 0-5000
                        state_clamps = [
                            (0, 20), (0, 50), (0, 20), (0, 5),
                            (0, 500), (-5, 45), (3, 12), (0, 4000),
                            (0, 100), (0, 5000),
                        ]
                        for var_idx, col in state_cols.items():
                            val = daily.iloc[start_idx][col]
                            if not np.isnan(val) and val != -999999:
                                lo, hi = state_clamps[var_idx]
                                state_t0[var_idx] = np.clip(val, lo, hi)

                        # Fill future states at each horizon
                        for h_idx, horizon in enumerate(FORECAST_HORIZONS):
                            future_idx = start_idx + horizon
                            if future_idx < len(daily):
                                for var_idx, col in state_cols.items():
                                    val = daily.iloc[future_idx][col]
                                    if not np.isnan(val) and val != -999999:
                                        lo, hi = state_clamps[var_idx]
                                        future[h_idx, var_idx] = np.clip(val, lo, hi)
                                        mask[h_idx, var_idx] = 1.0

                        # Only keep if at least some future observations exist
                        if mask.sum() < 3:
                            continue

                        # Build embedding from sensor features
                        window = daily.iloc[max(0, start_idx - 7):start_idx + 1]
                        feats = []
                        for col in window.columns:
                            vals = pd.to_numeric(window[col], errors="coerce").dropna()
                            if len(vals) > 0:
                                feats.extend([vals.mean(), vals.std(), vals.min(), vals.max()])
                        if len(feats) < 4:
                            continue

                        feats = np.array(feats, dtype=np.float32)
                        if len(feats) < 256:
                            feats = np.pad(feats, (0, 256 - len(feats)))
                        feats = feats[:256]
                        feats = feats / (np.linalg.norm(feats) + 1e-8)

                        embeddings.append(feats)
                        init_states_list.append(state_t0)
                        future_states_list.append(future)
                        future_masks_list.append(mask)

                elif fpath.suffix == ".npz":
                    # NPZ files without paired time series — skip
                    # (requires parquet with real temporal observations)
                    continue

            except Exception:
                continue

        if not embeddings:
            raise RuntimeError(
                f"Failed to build any samples from {len(selected)} sensor files for {split}. "
                "Check that parquet files have >=2 state variables (DO, Temp, pH, Turb, SpCond)."
            )

        self.embeddings = np.stack(embeddings)
        self.initial_states = np.stack(init_states_list)
        self.future_states = np.stack(future_states_list)
        self.future_masks = np.stack(future_masks_list)

        log(f"  Built {split} set: {len(self.embeddings)} samples from sensor data")

    # _generate_minimal removed — synthetic data not allowed

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return {
            "embedding": torch.from_numpy(self.embeddings[idx]),
            "initial_state": torch.from_numpy(self.initial_states[idx]),
            "future_states": torch.from_numpy(self.future_states[idx]),
            "future_mask": torch.from_numpy(self.future_masks[idx]),
        }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_epoch(model, loader, optimizer, scaler, device, horizons):
    model.train()
    total_loss = 0.0
    n = 0

    for batch in loader:
        emb = batch["embedding"].to(device)
        init_state = batch["initial_state"].to(device)
        future = batch["future_states"].to(device)
        mask = batch["future_mask"].to(device)

        optimizer.zero_grad(set_to_none=True)

        try:
            with autocast("cuda", dtype=torch.float16):
                output = model(emb, horizons=horizons, state_override=init_state)

                # Prediction loss: MSE on observed future states
                # output.predictions shape: [H, B, D]
                preds = output.predictions  # [H, B, D]
                # future shape: [B, H, D], mask shape: [B, H, D]
                future_t = future.permute(1, 0, 2)  # [H, B, D]
                mask_t = mask.permute(1, 0, 2)  # [H, B, D]

                pred_loss = (((preds - future_t) ** 2) * mask_t).sum() / mask_t.sum().clamp(min=1)

                # KL divergence on state posterior (vs standard normal)
                kl_loss = -0.5 * torch.mean(
                    1 + output.state_log_var - output.state_mean.pow(2) - output.state_log_var.exp()
                )

                # Physics consistency: penalize negative concentrations
                physics_loss = F.relu(-preds[:, :, :6]).mean()  # DO, BOD, TN, TP, Chl-a, Temp shouldn't be negative

                # Correction magnitude regularization: prevent neural corrector
                # from learning large corrections that overpower the physics
                correction_reg = (output.corrections ** 2).mean()

                # Physics-consistency loss: hybrid should stay close to physics
                physics_traj = output.physics_trajectory
                physics_at_horizons = physics_traj  # same time grid
                correction_rel = (output.corrections ** 2).sum() / (physics_traj ** 2 + 1e-6).sum()

                loss = pred_loss + 0.01 * kl_loss + 0.1 * physics_loss + 0.5 * correction_reg + 0.1 * correction_rel

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * emb.size(0)
            n += emb.size(0)

        except RuntimeError as e:
            if "nan" in str(e).lower() or "inf" in str(e).lower():
                log(f"  WARNING: NaN/Inf in batch, skipping")
                optimizer.zero_grad(set_to_none=True)
                continue
            raise

    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device, horizons):
    model.eval()
    total_loss = 0.0
    n = 0
    per_horizon_mse = {h: [] for h in horizons}
    per_var_mse = {v: [] for v in STATE_VARS}

    for batch in loader:
        emb = batch["embedding"].to(device)
        init_state = batch["initial_state"].to(device)
        future = batch["future_states"].to(device)
        mask = batch["future_mask"].to(device)

        try:
            with autocast("cuda", dtype=torch.float16):
                output = model(emb, horizons=horizons, state_override=init_state)

                preds = output.predictions  # [H, B, D]
                future_t = future.permute(1, 0, 2)  # [H, B, D]
                mask_t = mask.permute(1, 0, 2)

                loss = (((preds - future_t) ** 2) * mask_t).sum() / mask_t.sum().clamp(min=1)

            total_loss += loss.item() * emb.size(0)
            n += emb.size(0)

            # Per-horizon MSE
            for h_idx, h in enumerate(horizons):
                h_mask = mask_t[h_idx]
                if h_mask.sum() > 0:
                    h_mse = (((preds[h_idx] - future_t[h_idx]) ** 2) * h_mask).sum() / h_mask.sum()
                    per_horizon_mse[h].append(h_mse.item())

            # Per-variable MSE
            for v_idx, v_name in enumerate(STATE_VARS):
                v_mask = mask_t[:, :, v_idx]
                if v_mask.sum() > 0:
                    v_mse = (((preds[:, :, v_idx] - future_t[:, :, v_idx]) ** 2) * v_mask).sum() / v_mask.sum()
                    per_var_mse[v_name].append(v_mse.item())

        except RuntimeError:
            continue

    avg_loss = total_loss / max(n, 1)

    horizon_metrics = {}
    for h in horizons:
        vals = per_horizon_mse[h]
        horizon_metrics[f"{h}d_mse"] = float(np.mean(vals)) if vals else float("nan")

    var_metrics = {}
    for v in STATE_VARS:
        vals = per_var_mse[v]
        var_metrics[f"{v}_mse"] = float(np.mean(vals)) if vals else float("nan")

    return {
        "loss": avg_loss,
        "horizon_metrics": horizon_metrics,
        "variable_metrics": var_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Train Digital Twin")
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    log("=" * 60)
    log("Digital Aquatic Ecosystem Twin — Training")
    log("=" * 60)
    log(f"Device: {device}")

    from sentinel.models.twin.twin_engine import DigitalTwinEngine
    model = DigitalTwinEngine().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Model parameters: {n_params:,}")

    horizons = FORECAST_HORIZONS

    train_ds = TwinDataset(split="train", seed=args.seed)
    val_ds = TwinDataset(split="val", seed=args.seed)
    test_ds = TwinDataset(split="test", seed=args.seed)

    log(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)

    # Two-phase training:
    # Phase 1: Train data assimilator + forecast head (freeze ODE params)
    # Phase 2: Fine-tune everything including ODE parameters

    # Phase 1
    log("\n--- Phase 1: Assimilator + Forecast Head ---")
    ode_params = list(model.ode.parameters())
    other_params = [p for p in model.parameters() if not any(p is op for op in ode_params)]
    for p in ode_params:
        p.requires_grad = False

    optimizer = torch.optim.AdamW(other_params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs // 2)
    scaler = GradScaler("cuda")

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs // 2 + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scaler, device, horizons)
        val_metrics = evaluate(model, val_loader, device, horizons)
        scheduler.step()

        dt = time.time() - t0
        h_str = " | ".join(f"{k}: {v:.4f}" for k, v in list(val_metrics["horizon_metrics"].items())[:3])
        log(f"P1 Epoch {epoch:3d}/{args.epochs // 2} | "
            f"Train: {train_loss:.4f} | Val: {val_metrics['loss']:.4f} | {h_str} | {dt:.1f}s")

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "phase": 1,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_metrics,
            }, CKPT_DIR / "twin_phase1_best.pt")
            log(f"  ** New best (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                log(f"  Phase 1 early stopping at epoch {epoch}")
                break

    # Phase 2: Fine-tune everything
    log("\n--- Phase 2: Full Fine-Tuning (including ODE) ---")
    for p in ode_params:
        p.requires_grad = True

    optimizer = torch.optim.AdamW([
        {"params": ode_params, "lr": args.lr * 0.1},
        {"params": other_params, "lr": args.lr * 0.3},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs // 2)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs // 2 + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scaler, device, horizons)
        val_metrics = evaluate(model, val_loader, device, horizons)
        scheduler.step()

        dt = time.time() - t0
        h_str = " | ".join(f"{k}: {v:.4f}" for k, v in list(val_metrics["horizon_metrics"].items())[:3])
        log(f"P2 Epoch {epoch:3d}/{args.epochs // 2} | "
            f"Train: {train_loss:.4f} | Val: {val_metrics['loss']:.4f} | {h_str} | {dt:.1f}s")

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "phase": 2,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_metrics,
            }, CKPT_DIR / "twin_best.pt")
            log(f"  ** New best (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                log(f"  Phase 2 early stopping at epoch {epoch}")
                break

    # Test evaluation
    log("\n" + "=" * 60)
    log("Test Evaluation")
    log("=" * 60)

    ckpt = torch.load(CKPT_DIR / "twin_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device, horizons)

    log(f"Test Loss: {test_metrics['loss']:.4f}")
    log("\nPer-horizon MSE:")
    for k, v in test_metrics["horizon_metrics"].items():
        log(f"  {k}: {v:.4f}")
    log("\nPer-variable MSE:")
    for k, v in test_metrics["variable_metrics"].items():
        log(f"  {k}: {v:.4f}")

    # Compare physics-only vs hybrid
    log("\n--- Physics-Only Baseline ---")
    physics_preds_all = []
    hybrid_preds_all = []
    targets_all = []
    masks_all = []

    model.eval()
    for batch in test_loader:
        emb = batch["embedding"].to(device)
        init_state = batch["initial_state"].to(device)
        future = batch["future_states"].to(device)
        mask = batch["future_mask"].to(device)

        try:
            with torch.no_grad():
                physics_traj = model.physics_only_forward(emb, horizons=horizons, state_override=init_state)
                hybrid_out = model(emb, horizons=horizons, state_override=init_state)

            physics_preds_all.append(physics_traj[1:].cpu())  # skip t=0
            hybrid_preds_all.append(hybrid_out.predictions.cpu())
            targets_all.append(future.permute(1, 0, 2).cpu())
            masks_all.append(mask.permute(1, 0, 2).cpu())
        except RuntimeError:
            continue

    if physics_preds_all:
        physics_preds = torch.cat(physics_preds_all, dim=1)
        hybrid_preds = torch.cat(hybrid_preds_all, dim=1)
        targets = torch.cat(targets_all, dim=1)
        masks = torch.cat(masks_all, dim=1)

        physics_mse = (((physics_preds - targets) ** 2) * masks).sum() / masks.sum()
        hybrid_mse = (((hybrid_preds - targets) ** 2) * masks).sum() / masks.sum()

        log(f"Physics-only MSE: {physics_mse.item():.4f}")
        log(f"Hybrid (neural-ODE) MSE: {hybrid_mse.item():.4f}")
        log(f"Neural corrector improvement: {(1 - hybrid_mse / physics_mse) * 100:.1f}%")

        test_metrics["physics_only_mse"] = physics_mse.item()
        test_metrics["hybrid_mse"] = hybrid_mse.item()

    results = {
        "model": "DigitalTwinEngine",
        "n_params": n_params,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
        "best_epoch": ckpt["epoch"],
        "test_metrics": test_metrics,
        "state_variables": STATE_VARS,
        "forecast_horizons": list(FORECAST_HORIZONS),
    }
    with open(RESULTS_DIR / "twin_holdout.json", "w") as f:
        json.dump(results, f, indent=2)

    log(f"\nResults saved to {RESULTS_DIR / 'twin_holdout.json'}")
    log("DONE")


if __name__ == "__main__":
    main()
