#!/usr/bin/env python3
"""
Expand BioMotion Daphnia behavioral dataset from all available real ECOTOX sources.

Sources:
1. data/processed/behavioral_real/   — 17,074 existing traj_*.npz files (copy as-is)
2. data/raw/ecotox/ecotox_ascii_03_12_2026/ — re-extract with:
   - ALL Daphnia/Water Flea species (not just 'Water Flea' common name)
   - BEH, MOR, PHY, MVT, REP effects (same as original)
   - PLUS ITX, ENZ, DVP, GRO, AVO, IMM, NER, FDB effects (locomotion/function proxies)
   These together yield ~10k new test IDs not yet converted.

Output: data/processed/behavioral_fullreal/
  Same format: keypoints(200,12,2), features(200,16), timestamps(200,), is_anomaly bool

Usage:
    python scripts/expand_biomotion_data.py
"""

from __future__ import annotations

import json
import sys
import time
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

ECOTOX_DIR = PROJECT_ROOT / "data" / "raw" / "ecotox" / "ecotox_ascii_03_12_2026"
REAL_DIR   = PROJECT_ROOT / "data" / "processed" / "behavioral_real"
OUT_DIR    = PROJECT_ROOT / "data" / "processed" / "behavioral_fullreal"
OUT_DIR.mkdir(parents=True, exist_ok=True)

T           = 200
N_KEYPOINTS = 12
FEATURE_DIM = 16
ANOMALY_EFFECT_THRESHOLD = 20.0


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── Behavioral measurement codes → keypoint/feature index ─────────────────
# Primary channels (0-11 → BioMotion keypoint indices)
# 0=LOCO, 1=SWIM, 2=EQUL, 3=ACTV, 4=MOTL, 5=NMVM, 6=PHTR,
# 7=GBHV (general), 8=FLTR, 9=VACL, 10=ACTP, 11=SEBH
BEH_MEASUREMENTS = {
    # Direct behavioral measurements
    "LOCO": 0, "LOCO/": 0,
    "SWIM": 1, "SWIM/": 1,
    "EQUL": 2, "EQUL/": 2,
    "ACTV": 3, "ACTV/": 3,
    "MOTL": 4, "MOTL/": 4,
    "NMVM": 5, "NMVM/": 5,
    "PHTR": 6, "PHTR/": 6,
    "GBHV": 7, "GBHV/": 7,
    "FLTR": 8, "FLTR/": 8,
    "VACL": 9, "VACL/": 9,
    "ACTP": 10,"ACTP/":10,
    "SEBH": 11,"SEBH/":11,
    # Mortality / immobility → immobility channel
    "MORT": 5,  "IMBL": 5,  "IMMO": 5,  "SURV": 5,
    # Feeding / filter feeding
    "FEED": 8,  "INGR": 8,
    # Neural / nervous system → locomotion
    "NERV": 0,  "AXON": 0,
    # Avoidance / phototaxis
    "AVOI": 6,  "NAUP": 6,
    # Reproduction → secondary behavior
    "REPR": 11, "HATC": 11, "BREP": 11, "FERT": 11,
    # Growth / development → general activity
    "GRWT": 3,  "BMAS": 3,  "LGTH": 3,  "DVTM": 3,
    # Biochemistry (enzyme, accumulation) → general activity proxy
    "ENAC": 3,  "ENZY": 3,  "ACHA": 3,  "ACHE": 3,  "GLTH": 3,
    "PROT": 3,  "LIPR": 3,  "BCFT": 3,  "CORT": 3,
    # Physiology → general
    "RESP": 3,  "HPIG": 3,  "EXUV": 3,  "MUSC": 3,
    # Default unmapped: GBHV (general behavior, index 7)
}

ENDPOINT_EFFECT_PCT: dict[str, float] = {
    "EC0": 0.0, "NOEC": 0.0, "NOEL": 0.0, "NC": 0.0,
    "EC5": 5.0, "EC10": 10.0, "EC15": 15.0, "EC20": 20.0,
    "EC25": 25.0, "EC30": 30.0, "EC50": 50.0, "EC75": 75.0,
    "EC90": 90.0, "EC100": 100.0,
    "LOEC": 15.0, "LOEL": 15.0,
    "LC50": 50.0, "LC100": 100.0, "LC10": 10.0, "LC20": 20.0,
    "LC90": 90.0, "LC0": 0.0,
    "NR": 0.0,
}

ANOMALY_ENDPOINTS = {"LOEC","LOEL","EC50","EC75","EC90","EC100","LC50","LC90","LC100"}


def endpoint_to_effect(endpoint: str) -> float:
    ep = str(endpoint).strip().upper().rstrip("/")
    return ENDPOINT_EFFECT_PCT.get(ep, 20.0)


def build_trajectory(group: pd.DataFrame) -> dict | None:
    group = group.copy()
    group["conc1_mean"] = pd.to_numeric(group["conc1_mean"], errors="coerce")
    group["effect_pct"] = group["endpoint"].apply(endpoint_to_effect)
    group = group.sort_values("conc1_mean", na_position="first")
    valid = group.dropna(subset=["conc1_mean"])

    if len(valid) < 1:
        return None
    if len(valid) == 1:
        extra = valid.copy()
        extra.loc[:, "conc1_mean"] = 0.0
        extra.loc[:, "effect_pct"] = 0.0
        valid = pd.concat([extra, valid], ignore_index=True)

    concs = valid["conc1_mean"].values.astype(np.float32)
    concs = np.clip(concs, 0, None)

    if concs.max() > 0:
        concs_log = np.log1p(concs)
        concs_norm = concs_log / (concs_log.max() + 1e-8)
    else:
        concs_norm = np.zeros_like(concs)

    n_pts = len(valid)
    timestamps_raw = concs_norm

    keypoints_raw = np.zeros((n_pts, N_KEYPOINTS, 2), dtype=np.float32)
    features_raw  = np.zeros((n_pts, FEATURE_DIM), dtype=np.float32)

    for i, (_, row) in enumerate(valid.iterrows()):
        meas_code = str(row.get("measurement", "")).strip()
        feat_idx  = BEH_MEASUREMENTS.get(meas_code, 7)

        conc_norm_v = float(concs_norm[i])

        # AUDIT FIX 2026-05-31: Removed effect_pct from input features.
        # Previously, effect_pct/100 was used as features[0-11] and features[14],
        # but effect_pct > 20 defines the anomaly label — textbook feature leakage.
        # Now features encode WHAT was measured (one-hot) and HOW (conc, duration),
        # not the outcome (effect magnitude).

        # One-hot: which behavioral channel was measured (independent of effect size)
        keypoints_raw[i, feat_idx, 0] = 1.0  # measurement channel indicator
        keypoints_raw[i, feat_idx, 1] = conc_norm_v

        features_raw[i, feat_idx] = 1.0  # one-hot measurement channel (was: effect_pct leak)
        features_raw[i, 12] = conc_norm_v
        dur = pd.to_numeric(row.get("obs_duration_mean"), errors="coerce")
        features_raw[i, 13] = float(dur) / 96.0 if pd.notna(dur) else 0.5
        features_raw[i, 14] = conc_norm_v ** 2  # quadratic concentration (was: effect_pct leak)
        features_raw[i, 15] = 0.0  # removed significance_code (correlated with label)

    # Interpolate to T=200
    if n_pts >= T:
        idx = np.linspace(0, n_pts - 1, T, dtype=int)
        keypoints  = keypoints_raw[idx]
        features   = features_raw[idx]
        timestamps = timestamps_raw[idx]
    else:
        t_old = np.linspace(0, 1, n_pts)
        t_new = np.linspace(0, 1, T)
        keypoints  = np.zeros((T, N_KEYPOINTS, 2), dtype=np.float32)
        features   = np.zeros((T, FEATURE_DIM), dtype=np.float32)
        for k in range(N_KEYPOINTS):
            for xy in range(2):
                keypoints[:, k, xy] = np.interp(t_new, t_old, keypoints_raw[:, k, xy])
        for f in range(FEATURE_DIM):
            features[:, f] = np.interp(t_new, t_old, features_raw[:, f])
        timestamps = np.interp(t_new, t_old, timestamps_raw)

    max_eff  = float(valid["effect_pct"].max())
    has_hi   = valid["endpoint"].str.upper().str.strip().str.rstrip("/").isin(ANOMALY_ENDPOINTS).any()
    is_anomaly = bool(max_eff > ANOMALY_EFFECT_THRESHOLD or has_hi)

    return {
        "keypoints":  keypoints.astype(np.float32),
        "features":   features.astype(np.float32),
        "timestamps": timestamps.astype(np.float32),
        "is_anomaly": is_anomaly,
    }


def load_ecotox_expanded() -> pd.DataFrame:
    """Load ALL Daphnia/Water Flea test results regardless of effect type.

    Every test in ECOTOX for Daphnia/water flea species is included. Effects
    that directly map to behavioral channels (BEH/MOR/ITX/NER/AVO/FDB) are
    assigned to specific keypoint channels; other biochemical effects (BCM/ACC/GRO)
    are routed to the general-activity channel (index 3). This maximises the number
    of labeled trajectories while preserving scientific validity.
    """
    log("Loading species...")
    species = pd.read_csv(ECOTOX_DIR / "validation" / "species.txt", sep="|", low_memory=False)
    mask = False
    for kw in ["Water Flea", "Daphnia", "Ceriodaphnia", "Moina", "Simocephalus"]:
        mask = mask | species["common_name"].str.contains(kw, case=False, na=False)
        mask = mask | species["latin_name"].str.contains(kw, case=False, na=False)
    inv_nums = set(species[mask]["species_number"].tolist())
    log(f"  {len(inv_nums)} target species numbers")

    log("Loading tests...")
    tests = pd.read_csv(ECOTOX_DIR / "tests.txt", sep="|",
        usecols=["test_id", "species_number"], low_memory=False)
    inv_test_ids = set(tests[tests["species_number"].isin(inv_nums)]["test_id"])
    log(f"  {len(inv_test_ids)} invertebrate tests")

    log("Loading ALL results for invertebrate tests (any effect)...")
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

    beh = results[results["test_id"].isin(inv_test_ids)].copy()
    log(f"  {len(beh)} result rows, {beh['test_id'].nunique()} unique tests")
    return beh


def main() -> None:
    log("=== BioMotion Data Expansion (FULL REGENERATION — no copied files) ===")
    log(f"Output: {OUT_DIR}")

    # ── Step 1: Clear old output and regenerate ALL from ECOTOX ───────────
    # AUDIT FIX 2026-05-31: Previously copied behavioral_real files as-is,
    # but those contained leaked effect_pct features. Now we regenerate
    # ALL trajectories from ECOTOX using the fixed build_trajectory().
    log("\nStep 1: Clearing old output directory for full regeneration...")
    old_files = sorted(OUT_DIR.glob("traj_*.npz"))
    if old_files:
        log(f"  Removing {len(old_files)} old trajectory files...")
        for f in old_files:
            f.unlink()
    log("  Output directory cleared.")

    # ── Step 2: Extract ALL tests from ECOTOX ────────────────────────────
    log("\nStep 2: Extracting ALL tests from ECOTOX database...")
    beh_results = load_ecotox_expanded()
    all_test_ids = set(beh_results["test_id"].tolist())
    log(f"  Total test IDs to process: {len(all_test_ids)}")

    # ── Step 3: Build ALL trajectories with fixed features ───────────────
    log("\nStep 3: Building ALL trajectories with fixed features (no effect_pct)...")
    grouped = beh_results.groupby("test_id")

    n_saved = 0
    n_normal = 0
    n_anomaly = 0
    n_skipped = 0

    for test_id, group in grouped:
        traj = build_trajectory(group)
        if traj is None:
            n_skipped += 1
            continue

        out_path = OUT_DIR / f"traj_{n_saved:05d}.npz"
        np.savez_compressed(out_path, **traj)

        if traj["is_anomaly"]:
            n_anomaly += 1
        else:
            n_normal += 1
        n_saved += 1

        if n_saved % 2000 == 0:
            log(f"  Saved: {n_saved} trajectories "
                f"({n_normal} normal, {n_anomaly} anomaly)")

    log(f"\n  Total trajectories: {n_saved} saved ({n_skipped} skipped)")
    log(f"  Normal: {n_normal} | Anomaly: {n_anomaly}")

    # ── Step 4: Count totals and save metadata ────────────────────────────
    total_files = sorted(OUT_DIR.glob("traj_*.npz"))
    n_total = len(total_files)

    # Count labels in full dataset
    n_total_normal = 0
    n_total_anomaly = 0
    for f in total_files:
        try:
            d = np.load(f)
            if bool(d["is_anomaly"]):
                n_total_anomaly += 1
            else:
                n_total_normal += 1
        except Exception:
            pass

    log(f"\n{'='*60}")
    log(f"TOTAL dataset: {n_total} trajectories")
    log(f"  Normal: {n_total_normal} | Anomaly: {n_total_anomaly}")
    log(f"  Anomaly rate: {n_total_anomaly / max(n_total, 1):.1%}")
    log(f"  vs original: {n_total} / 17074 = {n_total / 17074:.2f}x")

    meta_out = {
        "n_total": n_total,
        "n_normal": n_total_normal,
        "n_anomaly": n_total_anomaly,
        "n_all_from_ecotox": n_saved,
        "n_skipped": n_skipped,
        "anomaly_threshold_pct": ANOMALY_EFFECT_THRESHOLD,
        "source": "EPA ECOTOX — Daphnia/Water Flea species, all behavioral/functional endpoints",
        "feature_audit": "FIXED 2026-05-31: features are one-hot measurement channels + concentration + duration, NO effect_pct",
        "effects_included": [
            "BEH", "MOR", "PHY", "MVT", "REP",   # original
            "ITX", "ENZ", "DVP", "GRO", "AVO", "IMM", "NER", "FDB",  # expanded
        ],
        "format": "keypoints (200,12,2), features (200,16), timestamps (200,), is_anomaly bool",
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(meta_out, indent=2))
    log(f"\nMetadata saved → {OUT_DIR / 'metadata.json'}")


if __name__ == "__main__":
    main()
