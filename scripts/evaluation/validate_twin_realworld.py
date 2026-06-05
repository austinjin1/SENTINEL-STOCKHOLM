#!/usr/bin/env python3
"""Validate Digital Twin with Real-World Case Studies.

Loads the trained digital twin and applies it to specific, well-documented
water quality events using real USGS data. For each event, we:
  1. Extract the pre-event sensor window as initial conditions
  2. Run the twin forward at all forecast horizons
  3. Compare predictions against actual observed outcomes
  4. Compute per-variable and per-horizon accuracy metrics

Case studies use the same real USGS parquet data from NWIS stations
that observed the events.

MIT License — Bryan Cheng, 2026
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT))

RESULTS_DIR = PROJECT / "results" / "benchmarks"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RAW_SENSOR_DIR = PROJECT / "data" / "raw" / "sensor" / "full"

STATE_VARS = [
    "dissolved_oxygen", "bod", "total_nitrogen", "total_phosphorus",
    "chlorophyll_a", "temperature", "ph", "turbidity", "doc", "sediment",
]
NUM_STATES = 10
FORECAST_HORIZONS = (1, 7, 14, 30, 90, 365)

# Column → state variable index mapping (same as train_twin.py)
COL_NAME_MAP = {
    "DO": 0, "Temp": 5, "pH": 6, "Turb": 7, "SpCond": 7,
}
STATE_CLAMPS = [
    (0, 20), (0, 50), (0, 20), (0, 5),
    (0, 500), (-5, 45), (3, 12), (0, 4000),
    (0, 100), (0, 5000),
]

# Real-world case studies with USGS station IDs and event dates
CASE_STUDIES = [
    {
        "name": "Lake Erie HAB 2023",
        "station_id": "04199500",  # Huron River at Milan, OH
        "event_date": "2023-08-15",
        "description": "Summer harmful algal bloom in western Lake Erie basin",
        "expected_changes": {"dissolved_oxygen": "decrease", "ph": "increase", "turbidity": "increase"},
    },
    {
        "name": "Gulf of Mexico Dead Zone 2023",
        "station_id": "07374000",  # Mississippi River at Baton Rouge
        "event_date": "2023-07-01",
        "description": "Hypoxic dead zone from nutrient loading via Mississippi",
        "expected_changes": {"dissolved_oxygen": "decrease", "temperature": "increase"},
    },
    {
        "name": "Chesapeake Bay Hypoxia 2018",
        "station_id": "01570500",  # Susquehanna River at Harrisburg, PA
        "event_date": "2018-08-01",
        "description": "Summer stratification and nutrient-driven hypoxia",
        "expected_changes": {"dissolved_oxygen": "decrease", "temperature": "increase"},
    },
    {
        "name": "Iowa Nitrate Crisis 2015",
        "station_id": "05420500",  # Mississippi River at Clinton, IA
        "event_date": "2015-06-01",
        "description": "Spring agricultural runoff nitrate contamination",
        "expected_changes": {"dissolved_oxygen": "decrease"},
    },
    {
        "name": "Klamath River HAB 2021",
        "station_id": "11510700",  # Klamath River near Klamath, CA
        "event_date": "2021-08-01",
        "description": "Cyanobacterial HAB in Klamath River system",
        "expected_changes": {"dissolved_oxygen": "decrease", "ph": "increase"},
    },
    {
        "name": "Chattahoochee Summer Low-Flow 2023",
        "station_id": "02336000",  # Chattahoochee River at Atlanta, GA
        "event_date": "2023-08-01",
        "description": "Summer low-flow stress on urban river system",
        "expected_changes": {"dissolved_oxygen": "decrease", "temperature": "increase"},
    },
    {
        "name": "Delaware River Seasonal Transition 2023",
        "station_id": "01463500",  # Delaware River at Trenton, NJ
        "event_date": "2023-10-01",
        "description": "Fall cooling transition — DO recovery expected",
        "expected_changes": {"dissolved_oxygen": "increase", "temperature": "decrease"},
    },
    {
        "name": "Columbia River Spring Melt 2023",
        "station_id": "14211720",  # Willamette River at Portland, OR
        "event_date": "2023-05-01",
        "description": "Spring snowmelt pulse — increased turbidity and flow",
        "expected_changes": {"turbidity": "increase", "temperature": "increase"},
    },
]


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_station_data(station_id: str):
    """Load USGS station parquet data into daily time series."""
    import pandas as pd

    fpath = RAW_SENSOR_DIR / f"{station_id}.parquet"
    if not fpath.exists():
        # Try with leading zeros variants
        for f in RAW_SENSOR_DIR.glob(f"*{station_id}*.parquet"):
            fpath = f
            break

    if not fpath.exists():
        return None

    df = pd.read_parquet(fpath)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").set_index("datetime")
    elif df.index.name == "datetime":
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

    # Map columns to state variables
    state_cols = {}
    for col in df.columns:
        if col in COL_NAME_MAP:
            idx = COL_NAME_MAP[col]
            if idx not in state_cols or col != "SpCond":
                state_cols[idx] = col

    if len(state_cols) < 2:
        return None

    for idx, col in state_cols.items():
        df[col] = pd.to_numeric(df[col], errors="coerce")

    daily = df.resample("D").mean()
    return daily, state_cols


def extract_state(daily, state_cols, date_str, window_days=7):
    """Extract state vector and embedding from daily data around a date."""
    import pandas as pd

    target_date = pd.Timestamp(date_str)
    # Match timezone if the index is tz-aware
    if daily.index.tz is not None:
        target_date = target_date.tz_localize(daily.index.tz)

    # Find nearest available date
    if target_date not in daily.index:
        nearest_idx = daily.index.get_indexer([target_date], method="nearest")[0]
        if nearest_idx < 0 or nearest_idx >= len(daily):
            return None, None
        target_date = daily.index[nearest_idx]

    loc = daily.index.get_loc(target_date)
    if isinstance(loc, slice):
        loc = loc.start

    # Initial state
    state = np.zeros(NUM_STATES, dtype=np.float32)
    for var_idx, col in state_cols.items():
        val = daily.iloc[loc][col]
        if not np.isnan(val) and val != -999999:
            lo, hi = STATE_CLAMPS[var_idx]
            state[var_idx] = np.clip(val, lo, hi)

    # Embedding from pre-event window
    window_start = max(0, loc - window_days)
    window = daily.iloc[window_start:loc + 1]
    feats = []
    for col in window.columns:
        vals = pd.to_numeric(window[col], errors="coerce").dropna()
        if len(vals) > 0:
            feats.extend([vals.mean(), vals.std(), vals.min(), vals.max()])
    if len(feats) < 4:
        return None, None

    feats = np.array(feats, dtype=np.float32)
    if len(feats) < 256:
        feats = np.pad(feats, (0, 256 - len(feats)))
    feats = feats[:256]
    feats = feats / (np.linalg.norm(feats) + 1e-8)

    return state, feats


def extract_future_observations(daily, state_cols, date_str):
    """Extract actual observations at forecast horizon dates."""
    import pandas as pd

    target_date = pd.Timestamp(date_str)
    if daily.index.tz is not None:
        target_date = target_date.tz_localize(daily.index.tz)
    if target_date not in daily.index:
        nearest_idx = daily.index.get_indexer([target_date], method="nearest")[0]
        if nearest_idx >= 0 and nearest_idx < len(daily):
            target_date = daily.index[nearest_idx]

    loc = daily.index.get_loc(target_date)
    if isinstance(loc, slice):
        loc = loc.start

    future_obs = np.zeros((len(FORECAST_HORIZONS), NUM_STATES), dtype=np.float32)
    future_mask = np.zeros((len(FORECAST_HORIZONS), NUM_STATES), dtype=np.float32)

    for h_idx, horizon in enumerate(FORECAST_HORIZONS):
        future_loc = loc + horizon
        if future_loc < len(daily):
            for var_idx, col in state_cols.items():
                val = daily.iloc[future_loc][col]
                if not np.isnan(val) and val != -999999:
                    lo, hi = STATE_CLAMPS[var_idx]
                    future_obs[h_idx, var_idx] = np.clip(val, lo, hi)
                    future_mask[h_idx, var_idx] = 1.0

    return future_obs, future_mask


def main():
    log("=" * 60)
    log("Digital Twin — Real-World Case Study Validation")
    log("=" * 60)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    # Load model
    from sentinel.models.twin.twin_engine import DigitalTwinEngine

    model = DigitalTwinEngine().to(device)

    # Load best checkpoint
    ckpt_dir = PROJECT / "checkpoints" / "twin"
    ckpt_paths = [
        ckpt_dir / "twin_best.pt",
        ckpt_dir / "twin_phase2_best.pt",
        ckpt_dir / "twin_phase1_best.pt",
    ]
    loaded = False
    for ckpt_path in ckpt_paths:
        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location=device, weights_only=True)
            model.load_state_dict(state, strict=False)
            log(f"Loaded checkpoint: {ckpt_path.name}")
            loaded = True
            break

    if not loaded:
        log("WARNING: No checkpoint found, using random weights")

    model.eval()

    results = []
    total_mse = 0.0
    total_count = 0

    for case in CASE_STUDIES:
        log(f"\n--- {case['name']} ---")
        log(f"  Station: {case['station_id']}, Event: {case['event_date']}")

        data = load_station_data(case["station_id"])
        if data is None:
            log(f"  SKIPPED: No data for station {case['station_id']}")
            results.append({
                "name": case["name"],
                "status": "NO_DATA",
                "station_id": case["station_id"],
            })
            continue

        daily, state_cols = data
        log(f"  Data: {len(daily)} days, {len(state_cols)} variables")

        # Extract pre-event state
        state, emb = extract_state(daily, state_cols, case["event_date"])
        if state is None:
            log(f"  SKIPPED: Cannot extract state at {case['event_date']}")
            results.append({
                "name": case["name"],
                "status": "NO_STATE",
                "station_id": case["station_id"],
            })
            continue

        # Get actual future observations
        future_obs, future_mask = extract_future_observations(
            daily, state_cols, case["event_date"]
        )

        n_future = int(future_mask.sum())
        log(f"  Initial state: {', '.join(f'{STATE_VARS[i]}={state[i]:.2f}' for i in state_cols.keys())}")
        log(f"  Future observations available: {n_future}")

        if n_future < 3:
            log(f"  SKIPPED: Insufficient future observations")
            results.append({
                "name": case["name"],
                "status": "INSUFFICIENT_FUTURE",
                "station_id": case["station_id"],
            })
            continue

        # Run twin prediction
        with torch.no_grad():
            emb_t = torch.from_numpy(emb).unsqueeze(0).to(device)
            state_t = torch.from_numpy(state).unsqueeze(0).to(device)

            # Forward pass through twin
            try:
                pred = model(emb_t)
            except Exception as e:
                log(f"  ERROR in model forward: {e}")
                results.append({
                    "name": case["name"],
                    "status": "MODEL_ERROR",
                    "station_id": case["station_id"],
                    "error": str(e),
                })
                continue

        # Extract predictions — shape [H, B, D], take batch index 0
        pred_states = pred.predictions.cpu().numpy()[:, 0, :]  # (H, 10)

        # Compute per-horizon errors (only where observations exist)
        case_result = {
            "name": case["name"],
            "status": "EVALUATED",
            "station_id": case["station_id"],
            "event_date": case["event_date"],
            "description": case["description"],
            "initial_state": {STATE_VARS[i]: float(state[i]) for i in state_cols.keys()},
            "horizon_metrics": {},
            "variable_metrics": {},
            "direction_accuracy": {},
        }

        for h_idx, horizon in enumerate(FORECAST_HORIZONS):
            h_mask = future_mask[h_idx]
            if h_mask.sum() == 0:
                continue

            obs = future_obs[h_idx]
            pred_h = pred_states[h_idx]

            # MSE only on observed variables
            masked_mse = ((pred_h - obs) ** 2 * h_mask).sum() / max(h_mask.sum(), 1)
            case_result["horizon_metrics"][f"{horizon}d_mse"] = float(masked_mse)
            total_mse += float(masked_mse)
            total_count += 1

        # Per-variable accuracy across all horizons
        for var_idx in state_cols.keys():
            var_name = STATE_VARS[var_idx]
            var_mask = future_mask[:, var_idx]
            if var_mask.sum() == 0:
                continue

            obs_vals = future_obs[:, var_idx][var_mask > 0]
            pred_vals = pred_states[:, var_idx][var_mask > 0]

            var_mse = float(((pred_vals - obs_vals) ** 2).mean())
            var_mae = float(np.abs(pred_vals - obs_vals).mean())
            case_result["variable_metrics"][var_name] = {
                "mse": var_mse,
                "mae": var_mae,
                "n_obs": int(var_mask.sum()),
            }

        # Direction accuracy: did the twin predict the right direction of change?
        for var_name, expected_dir in case.get("expected_changes", {}).items():
            var_idx = STATE_VARS.index(var_name)
            if var_idx not in state_cols:
                continue

            init_val = state[var_idx]
            # Check 7-day and 30-day predictions
            for h_idx, horizon in enumerate(FORECAST_HORIZONS):
                if horizon in (7, 30) and future_mask[h_idx, var_idx] > 0:
                    pred_val = pred_states[h_idx, var_idx]
                    obs_val = future_obs[h_idx, var_idx]
                    pred_change = pred_val - init_val
                    obs_change = obs_val - init_val

                    if expected_dir == "decrease":
                        correct = pred_change < 0
                    else:
                        correct = pred_change > 0

                    key = f"{var_name}_{horizon}d"
                    case_result["direction_accuracy"][key] = {
                        "predicted_change": float(pred_change),
                        "observed_change": float(obs_change),
                        "expected_direction": expected_dir,
                        "correct": bool(correct),
                    }

        # Print per-variable results
        for var_name, metrics in case_result["variable_metrics"].items():
            log(f"  {var_name}: MAE={metrics['mae']:.3f}, MSE={metrics['mse']:.3f} (n={metrics['n_obs']})")

        # Print direction accuracy
        for key, acc in case_result["direction_accuracy"].items():
            mark = "CORRECT" if acc["correct"] else "WRONG"
            log(f"  Direction {key}: pred={acc['predicted_change']:+.3f}, "
                f"obs={acc['observed_change']:+.3f} → {mark}")

        results.append(case_result)

    # Summary statistics
    log("\n" + "=" * 60)
    log("Summary")
    log("=" * 60)

    evaluated = [r for r in results if r["status"] == "EVALUATED"]
    skipped = [r for r in results if r["status"] != "EVALUATED"]

    log(f"Evaluated: {len(evaluated)} / {len(CASE_STUDIES)} case studies")
    log(f"Skipped: {len(skipped)} ({', '.join(r['name'] for r in skipped) if skipped else 'none'})")

    if total_count > 0:
        log(f"Mean MSE across all horizons: {total_mse / total_count:.3f}")

    # Direction accuracy summary (vs expected)
    total_dir = 0
    correct_dir = 0
    # Direction accuracy vs observed (does prediction change match observation change?)
    total_obs_dir = 0
    correct_obs_dir = 0
    for r in evaluated:
        for key, acc in r.get("direction_accuracy", {}).items():
            total_dir += 1
            if acc["correct"]:
                correct_dir += 1
            # Also check against observed direction
            total_obs_dir += 1
            pred_sign = 1 if acc["predicted_change"] > 0 else -1
            obs_sign = 1 if acc["observed_change"] > 0 else -1
            if pred_sign == obs_sign:
                correct_obs_dir += 1

    if total_dir > 0:
        log(f"Direction vs expected: {correct_dir}/{total_dir} ({100*correct_dir/total_dir:.1f}%)")
        log(f"Direction vs observed: {correct_obs_dir}/{total_obs_dir} ({100*correct_obs_dir/total_obs_dir:.1f}%)")

    # Save results
    out = {
        "model": "DigitalTwinEngine",
        "validation_type": "real_world_case_studies",
        "n_case_studies": len(CASE_STUDIES),
        "n_evaluated": len(evaluated),
        "mean_mse": total_mse / max(total_count, 1),
        "direction_accuracy": f"{correct_dir}/{total_dir}" if total_dir > 0 else "N/A",
        "case_studies": results,
    }

    out_path = RESULTS_DIR / "twin_realworld_validation.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    log(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
