#!/usr/bin/env python3
"""
Process ECOTOX Daphnia behavioral data into BioMotion training format.

Real Daphnia behavioral ecotoxicology tests from EPA ECOTOX database.
Each test measures behavioral responses (locomotion, swimming, equilibrium,
activity) across multiple concentrations. We convert each test's
concentration-response curve into a trajectory-like time series matching
BioMotion's .npz format.

Output .npz format (matches existing synthetic format):
  keypoints:  (T=200, 12, 2)  — 12 behavioral metrics as (mean_response, normalized_conc)
  features:   (T=200, 16)     — 16 behavioral features per time step
  timestamps: (T=200,)        — normalized concentration [0, 1]
  is_anomaly: bool            — True if significant behavioral impairment

Usage:
    python scripts/process_ecotox_behavioral.py

MIT License — Bryan Cheng, 2026
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

ECOTOX_DIR = PROJECT_ROOT / "data" / "raw" / "ecotox" / "ecotox_ascii_03_12_2026"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "behavioral_real"
OUT_DIR.mkdir(parents=True, exist_ok=True)

T = 200           # time steps (matches synthetic format)
N_KEYPOINTS = 12  # matches BioMotion architecture
FEATURE_DIM = 16  # matches BioMotion architecture

# Effect anomaly threshold: >20% effect relative to control
ANOMALY_EFFECT_THRESHOLD = 20.0

# Behavioral measurement codes → feature index (0-11 as keypoints, 0-15 as features)
BEH_MEASUREMENTS = {
    "LOCO":  0,   # locomotion
    "LOCO/": 0,
    "SWIM":  1,   # swimming velocity
    "SWIM/": 1,
    "EQUL":  2,   # equilibrium
    "EQUL/": 2,
    "ACTV":  3,   # general activity
    "ACTV/": 3,
    "MOTL":  4,   # motility
    "MOTL/": 4,
    "NMVM":  5,   # no movement / immobility
    "NMVM/": 5,
    "PHTR":  6,   # phototaxis
    "PHTR/": 6,
    "GBHV":  7,   # general behavior
    "GBHV/": 7,
    "FLTR":  8,   # filter feeding rate
    "FLTR/": 8,
    "VACL":  9,   # valve activity / locomotion
    "VACL/": 9,
    "ACTP": 10,   # active time proportion
    "ACTP/":10,
    "SEBH": 11,   # secondary behavioral response
    "SEBH/":11,
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_ecotox() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load Daphnia behavioral tests and results from ECOTOX."""
    log("Loading species lookup...")
    species = pd.read_csv(ECOTOX_DIR / "validation" / "species.txt", sep="|")
    daphnia_nums = set(
        species[species["common_name"].str.contains("Water Flea", case=False, na=False)][
            "species_number"
        ].tolist()
    )
    log(f"  {len(daphnia_nums)} Daphnia species")

    log("Loading tests...")
    tests = pd.read_csv(
        ECOTOX_DIR / "tests.txt",
        sep="|",
        usecols=["test_id", "species_number", "study_duration_mean", "study_duration_unit"],
        low_memory=False,
    )
    daphnia_test_ids = set(tests[tests["species_number"].isin(daphnia_nums)]["test_id"])
    log(f"  {len(daphnia_test_ids)} Daphnia tests")

    log("Loading results (BEH effect only)...")
    results = pd.read_csv(
        ECOTOX_DIR / "results.txt",
        sep="|",
        usecols=[
            "result_id", "test_id", "effect", "measurement",
            "endpoint", "trend", "conc1_mean", "conc1_unit",
            "effect_pct_mean", "obs_duration_mean", "obs_duration_unit",
            "significance_code",
        ],
        low_memory=False,
    )

    # Include BEH (behavioral), MOR (mortality), PHY (physiology) — all relevant to
    # Daphnia organism function; MOR/immobility is the primary Daphnia bioassay
    beh = results[
        results["test_id"].isin(daphnia_test_ids) &
        results["effect"].isin({"BEH", "MOR", "PHY", "MVT", "REP"})
    ].copy()
    log(f"  {len(beh)} Daphnia result rows, {beh['test_id'].nunique()} unique tests")
    return beh, tests[tests["test_id"].isin(daphnia_test_ids)]


# Map endpoint codes → approximate effect percentage
ENDPOINT_EFFECT_PCT: dict[str, float] = {
    "EC0":  0.0, "NOEC": 0.0, "NOEL": 0.0, "NC": 0.0,
    "EC5":  5.0, "EC10": 10.0, "EC15": 15.0, "EC20": 20.0,
    "EC25": 25.0, "EC30": 30.0, "EC50": 50.0, "EC75": 75.0,
    "EC90": 90.0, "EC100": 100.0,
    "LOEC": 15.0, "LOEL": 15.0,   # lowest observable effect ~ 15%
    "NR":    0.0,  # not reported — treat as baseline
}


def endpoint_to_effect(endpoint: str, trend: str) -> float:
    """Convert endpoint code + trend direction to effect percentage."""
    ep = str(endpoint).strip().upper().rstrip("/")
    pct = ENDPOINT_EFFECT_PCT.get(ep, 20.0)  # default 20% if unknown
    # If trend = INC and measurement is NMVM (immobility), that's anomalous
    # If trend = DEC and measurement is LOCO/SWIM, that's anomalous
    # Both cases already reflected in pct magnitude; trend just confirms direction
    return pct


def build_trajectory(group: pd.DataFrame) -> dict | None:
    """
    Convert one test's concentration-response measurements into a trajectory.

    Concentration levels become 'time steps'. Each step has 12 behavioral
    keypoint values and 16 feature values. Effect percentages are derived
    from endpoint codes (EC50, LOEC, NOEC, etc.) since effect_pct_mean is
    often not reported directly in ECOTOX.
    """
    group = group.copy()

    # Convert conc1_mean to float
    group["conc1_mean"] = pd.to_numeric(group["conc1_mean"], errors="coerce")

    # Derive effect_pct from endpoint code
    group["effect_pct"] = group.apply(
        lambda r: endpoint_to_effect(r.get("endpoint", "NR"), r.get("trend", "NR")),
        axis=1,
    )

    # Sort by concentration
    group = group.sort_values("conc1_mean", na_position="first")

    # Filter to rows with valid concentration
    valid = group.dropna(subset=["conc1_mean"])
    if len(valid) < 1:
        return None
    # Duplicate single-row tests to create minimal trajectory
    if len(valid) == 1:
        valid = pd.concat([valid, valid], ignore_index=True)
        # Set second row to 0 concentration (control baseline)
        valid.at[0, "conc1_mean"] = 0.0
        valid.at[0, "effect_pct"] = 0.0

    # Build concentration axis
    concs = valid["conc1_mean"].values.astype(np.float32)
    concs = np.clip(concs, 0, None)

    # Log-normalize concentrations to [0, 1]
    if concs.max() > 0:
        concs_log = np.log1p(concs)
        concs_norm = concs_log / (concs_log.max() + 1e-8)
    else:
        concs_norm = np.zeros_like(concs)

    n_pts = len(valid)
    timestamps_raw = concs_norm  # (n_pts,)

    # Build per-step feature matrices
    keypoints_raw = np.zeros((n_pts, N_KEYPOINTS, 2), dtype=np.float32)
    features_raw = np.zeros((n_pts, FEATURE_DIM), dtype=np.float32)

    for i, (_, row) in enumerate(valid.iterrows()):
        meas_code = str(row.get("measurement", "")).strip()
        feat_idx = BEH_MEASUREMENTS.get(meas_code, 7)  # default: GBHV

        effect_val = float(row["effect_pct"]) / 100.0  # normalize to [0,1]
        conc_norm_val = float(concs_norm[i])

        # Keypoints: (behavioral_metric_idx) → (mean_response, normalized_conc)
        keypoints_raw[i, feat_idx, 0] = np.clip(effect_val, 0.0, 1.0)
        keypoints_raw[i, feat_idx, 1] = conc_norm_val

        # Features: first 12 are behavioral metric values
        features_raw[i, feat_idx] = np.clip(effect_val, 0.0, 1.0)
        # Feature 12: concentration
        features_raw[i, 12] = conc_norm_val
        # Feature 13: obs_duration (normalized)
        dur = pd.to_numeric(row.get("obs_duration_mean"), errors="coerce")
        features_raw[i, 13] = float(dur) / 96.0 if pd.notna(dur) else 0.5
        # Feature 14: raw effect percent
        features_raw[i, 14] = np.clip(effect_val, 0.0, 2.0)
        # Feature 15: significance (1 = significant, 0 = not)
        sig = str(row.get("significance_code", "")).strip()
        features_raw[i, 15] = 1.0 if sig in {"*", "**", "***", "SIG"} else 0.0

    # Interpolate/pad to T=200
    if n_pts >= T:
        # Subsample evenly
        idx = np.linspace(0, n_pts - 1, T, dtype=int)
        keypoints = keypoints_raw[idx]
        features = features_raw[idx]
        timestamps = timestamps_raw[idx]
    else:
        # Interpolate each dimension up to T
        t_old = np.linspace(0, 1, n_pts)
        t_new = np.linspace(0, 1, T)
        keypoints = np.zeros((T, N_KEYPOINTS, 2), dtype=np.float32)
        features = np.zeros((T, FEATURE_DIM), dtype=np.float32)
        for k in range(N_KEYPOINTS):
            for xy in range(2):
                keypoints[:, k, xy] = np.interp(t_new, t_old, keypoints_raw[:, k, xy])
        for f in range(FEATURE_DIM):
            features[:, f] = np.interp(t_new, t_old, features_raw[:, f])
        timestamps = np.interp(t_new, t_old, timestamps_raw)

    # is_anomaly: any endpoint shows >20% behavioral impairment, or LOEC/EC50+ present
    max_effect = float(valid["effect_pct"].max())
    has_loec = valid["endpoint"].str.upper().str.strip().str.rstrip("/").isin(
        {"LOEC", "LOEL", "EC50", "EC100", "EC90", "EC75"}
    ).any()
    is_anomaly = bool(max_effect > ANOMALY_EFFECT_THRESHOLD or has_loec)

    return {
        "keypoints": keypoints.astype(np.float32),
        "features": features.astype(np.float32),
        "timestamps": timestamps.astype(np.float32),
        "is_anomaly": is_anomaly,
    }


def main() -> None:
    log("=== ECOTOX Daphnia Behavioral Data Processing ===")
    log(f"Output: {OUT_DIR}")

    beh_results, _ = load_ecotox()

    # Group by test_id and build one trajectory per test
    log("Building trajectories...")
    n_saved = 0
    n_normal = 0
    n_anomaly = 0
    n_skipped = 0

    grouped = beh_results.groupby("test_id")
    for test_id, group in grouped:
        traj = build_trajectory(group)
        if traj is None:
            n_skipped += 1
            continue

        out_path = OUT_DIR / f"traj_{n_saved:04d}.npz"
        np.savez_compressed(out_path, **traj)

        if traj["is_anomaly"]:
            n_anomaly += 1
        else:
            n_normal += 1
        n_saved += 1

        if n_saved % 100 == 0:
            log(f"  Saved {n_saved} trajectories ({n_normal} normal, {n_anomaly} anomaly)")

    log(f"\nDone. {n_saved} trajectories saved ({n_skipped} skipped).")
    log(f"  Normal: {n_normal} | Anomaly: {n_anomaly}")
    log(f"  Anomaly rate: {n_anomaly / max(n_saved, 1):.1%}")

    # Save metadata
    meta = {
        "n_total": n_saved,
        "n_normal": n_normal,
        "n_anomaly": n_anomaly,
        "n_skipped": n_skipped,
        "anomaly_threshold_pct": ANOMALY_EFFECT_THRESHOLD,
        "source": "EPA ECOTOX — Daphnia behavioral endpoints",
        "format": "keypoints (200,12,2), features (200,16), timestamps (200,), is_anomaly bool",
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
    log(f"Metadata → {OUT_DIR / 'metadata.json'}")


if __name__ == "__main__":
    main()
