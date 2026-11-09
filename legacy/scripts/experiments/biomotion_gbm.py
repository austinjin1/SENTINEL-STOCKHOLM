#!/usr/bin/env python3
"""biomotion_gbm.py — Tabular Gradient-Boosted-Trees replacement for the
BioMotion diffusion-transformer anomaly classifier.

Motivation
----------
The current BioMotion pipeline converts every ECOTOX test's concentration-
response curve into a (200, 16) "trajectory" and feeds it to a
TrajectoryDiffusionEncoder + AnomalyClassifier (~3M params), reaching a
random-split AUROC of 0.8067 (results/benchmarks/biomotion_holdout.json).

A behavioral ecotoxicology test is fundamentally a small table of facts
(what was measured, at what concentrations, for how long, on which species),
not a 200-step trajectory. This script fits a GradientBoostingClassifier on
exactly those tabular facts and measures whether it beats the diffusion model.

Label
-----
We reuse the EXACT BioMotion v2 label from
scripts/preprocessing/expand_biomotion_data.py::build_trajectory:

    effect_pct(row) = ENDPOINT_EFFECT_PCT[endpoint]          # per result row
    max_eff         = max(effect_pct over rows w/ valid conc1_mean)
    has_hi          = any endpoint in ANOMALY_ENDPOINTS
    is_anomaly      = (max_eff > 20.0) OR has_hi

Data source = the same set the baseline used: ALL Daphnia / Water-Flea /
Ceriodaphnia / Moina / Simocephalus species tests, every effect type, keeping
tests with at least one valid conc1_mean (matches build_trajectory's gate).

HARD GUARDRAIL (label-leakage prevention)
-----------------------------------------
The label is computed entirely from the `endpoint` code. Therefore we NEVER use
`endpoint`, `effect_pct(_mean)`, or `significance_code` as model features — those
trivially reconstruct the label and produced the prior bogus 0.9999. Features
encode only WHAT was measured and HOW (measurement channel, effect group,
concentrations, exposure duration, species), never the OUTCOME.

Outputs
-------
results/benchmarks/biomotion_v2_holdout.json  (this script's GBM block)
  - random_split AUROC  (baseline to beat: BioMotion diffusion 0.807)
  - chemical_grouped split AUROC (GroupShuffleSplit on test_cas) =
    honest novel-chemical generalization number.

MIT License — Bryan Cheng, 2026
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

ECOTOX_DIR = PROJECT_ROOT / "data" / "raw" / "ecotox" / "ecotox_ascii_03_12_2026"
RESULTS_DIR = PROJECT_ROOT / "results" / "benchmarks"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = RESULTS_DIR / "biomotion_v2_holdout.json"

SEED = 42
BASELINE_DIFFUSION_AUROC = 0.8067  # BioMotion v2, results/benchmarks/biomotion_holdout.json

# ─────────────────────────────────────────────────────────────────────────────
# Label definition — copied verbatim from expand_biomotion_data.py
# (these drive the LABEL only; endpoint/effect_pct are NEVER used as features)
# ─────────────────────────────────────────────────────────────────────────────
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
ANOMALY_ENDPOINTS = {"LOEC", "LOEL", "EC50", "EC75", "EC90", "EC100",
                     "LC50", "LC90", "LC100"}
ANOMALY_EFFECT_THRESHOLD = 20.0

# Measurement code → BioMotion behavioral channel (0-11). Copied from
# expand_biomotion_data.py::BEH_MEASUREMENTS. Used ONLY as a one-hot feature
# describing WHICH channel was measured — independent of the outcome.
BEH_MEASUREMENTS = {
    "LOCO": 0, "LOCO/": 0, "SWIM": 1, "SWIM/": 1, "EQUL": 2, "EQUL/": 2,
    "ACTV": 3, "ACTV/": 3, "MOTL": 4, "MOTL/": 4, "NMVM": 5, "NMVM/": 5,
    "PHTR": 6, "PHTR/": 6, "GBHV": 7, "GBHV/": 7, "FLTR": 8, "FLTR/": 8,
    "VACL": 9, "VACL/": 9, "ACTP": 10, "ACTP/": 10, "SEBH": 11, "SEBH/": 11,
    "MORT": 5, "IMBL": 5, "IMMO": 5, "SURV": 5,
    "FEED": 8, "INGR": 8,
    "NERV": 0, "AXON": 0,
    "AVOI": 6, "NAUP": 6,
    "REPR": 11, "HATC": 11, "BREP": 11, "FERT": 11,
    "GRWT": 3, "BMAS": 3, "LGTH": 3, "DVTM": 3,
    "ENAC": 3, "ENZY": 3, "ACHA": 3, "ACHE": 3, "GLTH": 3,
    "PROT": 3, "LIPR": 3, "BCFT": 3, "CORT": 3,
    "RESP": 3, "HPIG": 3, "EXUV": 3, "MUSC": 3,
}
N_CHANNELS = 12

# Effect group → one-hot index. The `effect` column is the
# measurement-channel / endpoint GROUP (BEH=behavior, MOR=mortality,
# ITX=intoxication, REP=reproduction, ...). It says WHAT category was studied,
# not the magnitude of harm, so it is guardrail-safe.
EFFECT_GROUPS = ["MOR", "ITX", "REP", "POP", "GRO", "BCM", "BEH",
                 "ACC", "GEN", "ENZ", "PHY", "MVT", "DVP", "IMM", "FDB"]

# Duration unit → hours, to put exposure_duration on one scale.
DURATION_UNIT_HOURS = {
    "h": 1.0, "hr": 1.0, "hours": 1.0,
    "d": 24.0, "dph": 24.0, "dpf": 24.0,
    "wk": 24.0 * 7, "mo": 24.0 * 30, "yr": 24.0 * 365,
    "mi": 1.0 / 60.0, "min": 1.0 / 60.0, "s": 1.0 / 3600.0,
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def endpoint_to_effect(endpoint: str) -> float:
    ep = str(endpoint).strip().upper().rstrip("/")
    return ENDPOINT_EFFECT_PCT.get(ep, 20.0)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading — reproduce the exact baseline test set
# ─────────────────────────────────────────────────────────────────────────────
def load_results_and_tests() -> tuple[pd.DataFrame, pd.DataFrame]:
    log("Loading species (Daphnia / Water-Flea group)...")
    species = pd.read_csv(ECOTOX_DIR / "validation" / "species.txt",
                          sep="|", low_memory=False)
    mask = pd.Series(False, index=species.index)
    for kw in ["Water Flea", "Daphnia", "Ceriodaphnia", "Moina", "Simocephalus"]:
        mask = mask | species["common_name"].str.contains(kw, case=False, na=False)
        mask = mask | species["latin_name"].str.contains(kw, case=False, na=False)
    inv_nums = set(species[mask]["species_number"].tolist())
    log(f"  {len(inv_nums)} target species numbers")

    log("Loading tests.txt...")
    tests = pd.read_csv(
        ECOTOX_DIR / "tests.txt", sep="|",
        usecols=["test_id", "species_number", "test_cas",
                 "exposure_duration_mean", "exposure_duration_unit"],
        low_memory=False,
    )
    tests = tests[tests["species_number"].isin(inv_nums)].copy()
    inv_test_ids = set(tests["test_id"])
    log(f"  {len(inv_test_ids)} invertebrate tests")

    log("Loading results.txt (all effects for these tests)...")
    results = pd.read_csv(
        ECOTOX_DIR / "results.txt", sep="|",
        usecols=["result_id", "test_id", "effect", "measurement",
                 "endpoint", "conc1_mean", "obs_duration_mean"],
        low_memory=False,
    )
    results = results[results["test_id"].isin(inv_test_ids)].copy()
    log(f"  {len(results)} result rows, {results['test_id'].nunique()} unique tests")
    return results, tests


# ─────────────────────────────────────────────────────────────────────────────
# Per-test feature + label construction
# ─────────────────────────────────────────────────────────────────────────────
def build_table(results: pd.DataFrame, tests: pd.DataFrame) -> pd.DataFrame:
    results = results.copy()
    results["conc1_mean"] = pd.to_numeric(results["conc1_mean"], errors="coerce")
    results["effect_pct"] = results["endpoint"].apply(endpoint_to_effect)
    results["endpoint_norm"] = (
        results["endpoint"].astype(str).str.upper().str.strip().str.rstrip("/")
    )

    tests = tests.copy()
    tests["exposure_duration_mean"] = pd.to_numeric(
        tests["exposure_duration_mean"], errors="coerce")
    cas_map = dict(zip(tests["test_id"], tests["test_cas"]))
    species_map = dict(zip(tests["test_id"], tests["species_number"]))
    dur_val_map = dict(zip(tests["test_id"], tests["exposure_duration_mean"]))
    dur_unit_map = dict(zip(tests["test_id"], tests["exposure_duration_unit"]))

    rows = []
    for test_id, g in results.groupby("test_id"):
        g_valid = g.dropna(subset=["conc1_mean"])
        # Gate matches build_trajectory: skip tests with no valid concentration.
        if len(g_valid) < 1:
            continue

        # ── LABEL (BioMotion v2, verbatim) ───────────────────────────────────
        max_eff = float(g_valid["effect_pct"].max())
        has_hi = g_valid["endpoint_norm"].isin(ANOMALY_ENDPOINTS).any()
        is_anomaly = bool(max_eff > ANOMALY_EFFECT_THRESHOLD or has_hi)

        # ── FEATURES (guardrail-safe: no endpoint / effect_pct / significance) ─
        concs = np.clip(g_valid["conc1_mean"].values.astype(np.float64), 0, None)
        log_conc = np.log1p(concs)
        unique_concs = np.unique(concs)

        feat = {
            "test_id": test_id,
            "test_cas": cas_map.get(test_id, -1),
            "is_anomaly": int(is_anomaly),
            # log-concentration summary stats
            "logconc_median": float(np.median(log_conc)),
            "logconc_max": float(log_conc.max()),
            "logconc_min": float(log_conc.min()),
            "logconc_mean": float(log_conc.mean()),
            "logconc_std": float(log_conc.std()) if len(log_conc) > 1 else 0.0,
            "logconc_range": float(log_conc.max() - log_conc.min()),
            "n_levels": int(len(unique_concs)),       # distinct concentrations tested
            "n_rows": int(len(g_valid)),              # result rows for this test
        }

        # exposure duration (tests.txt), normalized to hours then log
        dur_val = dur_val_map.get(test_id, np.nan)
        dur_unit = str(dur_unit_map.get(test_id, "")).strip().lower()
        if pd.notna(dur_val):
            hours = float(dur_val) * DURATION_UNIT_HOURS.get(dur_unit, 24.0)
            feat["exposure_hours_log"] = float(np.log1p(max(hours, 0.0)))
            feat["exposure_missing"] = 0
        else:
            feat["exposure_hours_log"] = 0.0
            feat["exposure_missing"] = 1

        # obs_duration (results.txt) summary — distinct from exposure duration
        obs = pd.to_numeric(g_valid["obs_duration_mean"], errors="coerce").dropna()
        feat["obs_duration_log"] = float(np.log1p(obs.max())) if len(obs) else 0.0

        # species id (as an integer category; trees can split on it)
        feat["species_number"] = int(species_map.get(test_id, -1))

        # measurement-channel one-hot (count of rows mapped to each channel,
        # normalized) — "what was measured", never the outcome
        chan_counts = np.zeros(N_CHANNELS, dtype=np.float64)
        for m in g_valid["measurement"].astype(str).str.strip():
            idx = BEH_MEASUREMENTS.get(m, 7)  # default GBHV
            chan_counts[idx] += 1.0
        total = chan_counts.sum()
        if total > 0:
            chan_counts /= total
        for c in range(N_CHANNELS):
            feat[f"chan_{c}"] = float(chan_counts[c])

        # effect-group one-hot (presence of each endpoint GROUP among rows)
        eff_present = set(g_valid["effect"].astype(str).str.strip().str.upper())
        for eg in EFFECT_GROUPS:
            feat[f"effgrp_{eg}"] = 1.0 if eg in eff_present else 0.0
        feat["effgrp_n"] = float(len(eff_present))

        rows.append(feat)

    df = pd.DataFrame(rows)
    log(f"Built table: {len(df)} tests, anomaly rate {df['is_anomaly'].mean():.3f}")
    return df


FEATURE_COLS_EXCLUDE = {"test_id", "test_cas", "is_anomaly"}


def feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    cols = [c for c in df.columns if c not in FEATURE_COLS_EXCLUDE]
    X = df[cols].to_numpy(dtype=np.float64)
    y = df["is_anomaly"].to_numpy(dtype=np.int64)
    return X, y, cols


def fit_eval(X_tr, y_tr, X_te, y_te) -> dict:
    clf = GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=3,
        subsample=0.8, random_state=SEED,
    )
    clf.fit(X_tr, y_tr)
    p = clf.predict_proba(X_te)[:, 1]
    return {
        "auroc": float(roc_auc_score(y_te, p)),
        "auprc": float(average_precision_score(y_te, p)),
        "f1": float(f1_score(y_te, (p >= 0.5).astype(int))),
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "test_pos_rate": float(y_te.mean()),
    }, clf


def main() -> None:
    t0 = time.time()
    log("=== BioMotion GBM (tabular ECOTOX) ===")

    results, tests = load_results_and_tests()
    df = build_table(results, tests)

    # Guardrail self-check: ensure no banned columns leaked into features
    banned = {"endpoint", "effect_pct", "effect_pct_mean", "significance_code",
              "endpoint_norm", "max_eff"}
    leaked = banned.intersection(df.columns)
    assert not leaked, f"GUARDRAIL VIOLATION: banned feature(s) present: {leaked}"

    X, y, cols = feature_matrix(df)
    groups = df["test_cas"].to_numpy()
    log(f"Feature matrix: X={X.shape}, {len(cols)} features, "
        f"{df['test_cas'].nunique()} distinct chemicals (test_cas)")

    # ── (a) Random split ─────────────────────────────────────────────────────
    log("\n--- (a) Random split (stratified 80/20) ---")
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y)
    rand_m, _ = fit_eval(Xtr, ytr, Xte, yte)
    log(f"  random-split AUROC = {rand_m['auroc']:.4f}  "
        f"(AUPRC {rand_m['auprc']:.4f}, F1 {rand_m['f1']:.4f})  "
        f"vs diffusion baseline {BASELINE_DIFFUSION_AUROC:.4f}")

    # ── (b) Chemical-grouped split (novel-chemical generalization) ───────────
    log("\n--- (b) Chemical-grouped split (GroupShuffleSplit on test_cas) ---")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    tr_idx, te_idx = next(gss.split(X, y, groups))
    # sanity: zero chemical overlap between train and test
    overlap = set(groups[tr_idx]) & set(groups[te_idx])
    assert not overlap, f"chemical leakage across grouped split: {len(overlap)} CAS"
    grp_m, grp_clf = fit_eval(X[tr_idx], y[tr_idx], X[te_idx], y[te_idx])
    log(f"  grouped-split AUROC = {grp_m['auroc']:.4f}  "
        f"(AUPRC {grp_m['auprc']:.4f}, F1 {grp_m['f1']:.4f})")
    log(f"  train chemicals={len(set(groups[tr_idx]))}, "
        f"test chemicals={len(set(groups[te_idx]))}")

    # Multi-seed robustness for the grouped split (so the headline novel-chemical
    # number is not a single lucky fold). Reported as mean ± std over 10 seeds.
    log("  computing grouped-split robustness over 10 seeds...")
    grp_aurocs = []
    for s in range(10):
        gss_s = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=s)
        tr_s, te_s = next(gss_s.split(X, y, groups))
        m_s, _ = fit_eval(X[tr_s], y[tr_s], X[te_s], y[te_s])
        grp_aurocs.append(m_s["auroc"])
    grp_aurocs = np.array(grp_aurocs)
    grouped_robustness = {
        "n_seeds": 10,
        "auroc_mean": float(grp_aurocs.mean()),
        "auroc_std": float(grp_aurocs.std()),
        "auroc_min": float(grp_aurocs.min()),
        "auroc_max": float(grp_aurocs.max()),
        "auroc_per_seed": [float(a) for a in grp_aurocs],
    }
    log(f"  grouped AUROC over 10 seeds: "
        f"{grouped_robustness['auroc_mean']:.4f} ± {grouped_robustness['auroc_std']:.4f}")

    # top feature importances from the grouped model (honest generalization model)
    imp = sorted(zip(cols, grp_clf.feature_importances_),
                 key=lambda z: z[1], reverse=True)[:12]

    out = {
        "model": "BioMotion-GBM (GradientBoostingClassifier on tabular ECOTOX)",
        "replaces": "BioMotion v2 TrajectoryDiffusionEncoder + AnomalyClassifier",
        "data_source": "REAL EPA ECOTOX (no synthetic). Daphnia/Water-Flea group, "
                       "all effects, tests with >=1 valid conc1_mean.",
        "label_definition": "BioMotion v2 (expand_biomotion_data.py): "
                            "is_anomaly = (max endpoint-derived effect_pct > 20) OR "
                            "(any endpoint in {LOEC,LOEL,EC50,EC75,EC90,EC100,LC50,LC90,LC100})",
        "guardrail": "Features exclude endpoint, effect_pct, significance_code "
                     "(label-leaking). Features = measurement-channel one-hot, "
                     "effect-group one-hot, log-concentration median/max/min/std/range/"
                     "n_levels, exposure duration, obs duration, species id.",
        "n_tests": int(len(df)),
        "n_chemicals": int(df["test_cas"].nunique()),
        "anomaly_rate": float(df["is_anomaly"].mean()),
        "n_features": len(cols),
        "feature_names": cols,
        "classifier": {
            "type": "GradientBoostingClassifier",
            "n_estimators": 300, "learning_rate": 0.05, "max_depth": 3,
            "subsample": 0.8, "random_state": SEED,
        },
        "baseline_diffusion_auroc": BASELINE_DIFFUSION_AUROC,
        "random_split": rand_m,
        "chemical_grouped_split": grp_m,
        "chemical_grouped_split_robustness": grouped_robustness,
        "top_feature_importances_grouped": [
            {"feature": f, "importance": float(i)} for f, i in imp
        ],
        "elapsed_sec": round(time.time() - t0, 1),
    }
    OUT_PATH.write_text(json.dumps(out, indent=2))
    log(f"\nResults → {OUT_PATH}")

    print("\n" + "=" * 70)
    print("BioMotion-GBM (tabular ECOTOX) — honest AUROC")
    print("-" * 70)
    print(f"  Diffusion baseline (random split)     : {BASELINE_DIFFUSION_AUROC:.4f}")
    print(f"  GBM random split                      : {rand_m['auroc']:.4f}")
    print(f"  GBM chemical-grouped (novel chemicals): {grp_m['auroc']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
