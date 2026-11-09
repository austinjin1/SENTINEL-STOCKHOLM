#!/usr/bin/env python3
"""biomotion_transfer.py — Honest out-of-distribution transfer tests for the
BioMotion tabular ECOTOX anomaly classifier.

The baseline (biomotion_gbm.py) reports random-split AUROC 0.872 and a
"chemical-grouped" split (random CAS held out) AUROC ~0.85. Random-CAS grouping
is a WEAK generalization test: a held-out chemical is almost always surrounded
in feature space by chemically-similar, mechanistically-similar training
chemicals. This script asks the three harder questions:

  (1) NOVEL CHEMICAL CLASS — leave-one-chemical-class-out (LOCO-class).
      Group chemicals by ECOTOX `ecotox_group` (mode-of-action / class:
      PFAS, PPCPs, EDCs, PCBs, organotin, conazoles, PAHs, phthalates,
      metals...). Train on all-but-one class, test on the held-out class.
      A model that has learned a transferable dose-response signature should
      still rank harm on a chemistry it has never seen.

  (2) CROSS-SPECIES / CROSS-TAXON — train on Daphnia (the standard test
      genus), test on a genuinely different invertebrate genus:
      Ceriodaphnia (sister cladoceran), Chironomus/Aedes/Culex (insects),
      Gammarus/Hyalella/Americamysis (malacostracan crustaceans),
      Caenorhabditis (nematode), Lymnaea/Physa (molluscs),
      Tubifex/Lumbriculus (annelids). This crosses phylum boundaries, not
      just genus, so it is a real physiology-transfer test.

  (3) WHICH FEATURES TRANSFER — the baseline's top feature is
      `obs_duration_log` (0.32 importance), a STUDY-DESIGN variable, not a
      biological response. We measure how much of the transfer signal is a
      design artifact by ablating design-only features and re-running (1)+(2).

Everything reuses the EXACT BioMotion v2 label and guardrail-safe feature
construction from biomotion_gbm.py — no new leakage surface is introduced.
The label is derived only from `endpoint`; `endpoint`, `effect_pct`, and
`significance_code` are NEVER features.

Data = REAL EPA ECOTOX (data/raw/ecotox/ecotox_ascii_03_12_2026). No synthetic.

Outputs results/benchmarks/biomotion_transfer.json.

MIT License — Bryan Cheng, 2026
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ECOTOX_DIR = PROJECT_ROOT / "data" / "raw" / "ecotox" / "ecotox_ascii_03_12_2026"
RESULTS_DIR = PROJECT_ROOT / "results" / "benchmarks"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = RESULTS_DIR / "biomotion_transfer.json"

SEED = 42

# ── Label + feature constants (verbatim from biomotion_gbm.py) ───────────────
ENDPOINT_EFFECT_PCT = {
    "EC0": 0.0, "NOEC": 0.0, "NOEL": 0.0, "NC": 0.0,
    "EC5": 5.0, "EC10": 10.0, "EC15": 15.0, "EC20": 20.0,
    "EC25": 25.0, "EC30": 30.0, "EC50": 50.0, "EC75": 75.0,
    "EC90": 90.0, "EC100": 100.0, "LOEC": 15.0, "LOEL": 15.0,
    "LC50": 50.0, "LC100": 100.0, "LC10": 10.0, "LC20": 20.0,
    "LC90": 90.0, "LC0": 0.0, "NR": 0.0,
}
ANOMALY_ENDPOINTS = {"LOEC", "LOEL", "EC50", "EC75", "EC90", "EC100",
                     "LC50", "LC90", "LC100"}
ANOMALY_EFFECT_THRESHOLD = 20.0

BEH_MEASUREMENTS = {
    "LOCO": 0, "LOCO/": 0, "SWIM": 1, "SWIM/": 1, "EQUL": 2, "EQUL/": 2,
    "ACTV": 3, "ACTV/": 3, "MOTL": 4, "MOTL/": 4, "NMVM": 5, "NMVM/": 5,
    "PHTR": 6, "PHTR/": 6, "GBHV": 7, "GBHV/": 7, "FLTR": 8, "FLTR/": 8,
    "VACL": 9, "VACL/": 9, "ACTP": 10, "ACTP/": 10, "SEBH": 11, "SEBH/": 11,
    "MORT": 5, "IMBL": 5, "IMMO": 5, "SURV": 5,
    "FEED": 8, "INGR": 8, "NERV": 0, "AXON": 0, "AVOI": 6, "NAUP": 6,
    "REPR": 11, "HATC": 11, "BREP": 11, "FERT": 11,
    "GRWT": 3, "BMAS": 3, "LGTH": 3, "DVTM": 3, "ENAC": 3, "ENZY": 3,
    "ACHA": 3, "ACHE": 3, "GLTH": 3, "PROT": 3, "LIPR": 3, "BCFT": 3,
    "CORT": 3, "RESP": 3, "HPIG": 3, "EXUV": 3, "MUSC": 3,
}
N_CHANNELS = 12
EFFECT_GROUPS = ["MOR", "ITX", "REP", "POP", "GRO", "BCM", "BEH",
                 "ACC", "GEN", "ENZ", "PHY", "MVT", "DVP", "IMM", "FDB"]
DURATION_UNIT_HOURS = {
    "h": 1.0, "hr": 1.0, "hours": 1.0,
    "d": 24.0, "dph": 24.0, "dpf": 24.0,
    "wk": 24.0 * 7, "mo": 24.0 * 30, "yr": 24.0 * 365,
    "mi": 1.0 / 60.0, "min": 1.0 / 60.0, "s": 1.0 / 3600.0,
}

# Design-only features (study-design variables, NOT biological response).
# These describe HOW the experiment was run, not what the organism did.
DESIGN_ONLY_FEATURES = {
    "obs_duration_log", "exposure_hours_log", "exposure_missing",
    "n_levels", "n_rows",
    "logconc_median", "logconc_max", "logconc_min", "logconc_mean",
    "logconc_std", "logconc_range",
}

# Genera for the cross-species transfer test. Daphnia is the train genus.
CROSS_SPECIES_TARGETS = {
    "Daphnia":        ("Arthropoda", "Branchiopoda", "cladoceran (train genus)"),
    "Ceriodaphnia":   ("Arthropoda", "Branchiopoda", "sister cladoceran"),
    "Moina":          ("Arthropoda", "Branchiopoda", "cladoceran"),
    "Chironomus":     ("Arthropoda", "Insecta",      "insect (midge)"),
    "Aedes":          ("Arthropoda", "Insecta",      "insect (mosquito)"),
    "Culex":          ("Arthropoda", "Insecta",      "insect (mosquito)"),
    "Gammarus":       ("Arthropoda", "Malacostraca", "amphipod crustacean"),
    "Hyalella":       ("Arthropoda", "Malacostraca", "amphipod crustacean"),
    "Americamysis":   ("Arthropoda", "Malacostraca", "mysid crustacean"),
    "Artemia":        ("Arthropoda", "Branchiopoda", "brine shrimp"),
    "Caenorhabditis": ("Nematoda",   "Secernentea",  "nematode"),
    "Lymnaea":        ("Mollusca",   "Gastropoda",   "gastropod mollusc"),
    "Physa":          ("Mollusca",   "Gastropoda",   "gastropod mollusc"),
    "Tubifex":        ("Annelida",   "Oligochaeta",  "annelid worm"),
    "Lumbriculus":    ("Annelida",   "Clitellata",   "annelid worm"),
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def endpoint_to_effect(endpoint: str) -> float:
    ep = str(endpoint).strip().upper().rstrip("/")
    return ENDPOINT_EFFECT_PCT.get(ep, 20.0)


# ── Chemical-class canonicalization ──────────────────────────────────────────
# Many ECOTOX ecotox_group values are single metals with tiny n. Collapse all
# elemental metals into one "Metals" superclass so the held-out class has enough
# tests to give a meaningful AUROC. Mode-of-action classes are kept as-is.
METAL_GROUPS = {
    "arsenic", "mercury", "copper", "zinc", "chromium", "aluminum", "iron",
    "selenium", "cadmium", "nickel", "lead", "thallium", "silver", "cobalt",
    "beryllium", "barium", "antimony", "manganese", "vanadium", "tin",
    "chromium(vi)", "chromium(iii)", "major ions",
}


def canonical_class(group: str) -> str | None:
    """Map an ecotox_group string to a canonical chemical class, or None."""
    if not isinstance(group, str):
        return None
    g = group.strip()
    if not g:
        return None
    # a chemical may carry multiple ';'-joined groups; take the first
    first = g.split(";")[0].strip()
    low = first.lower()
    if low in METAL_GROUPS or low.rstrip("s") in {m.rstrip("s") for m in METAL_GROUPS}:
        return "Metals"
    return first


# ── Data loading ─────────────────────────────────────────────────────────────
def load_all():
    """Load species+tests+results for ALL invertebrate transfer genera at once,
    plus chemical-class labels. Returns (results, tests, cas_to_class)."""
    log("Loading species.txt ...")
    species = pd.read_csv(ECOTOX_DIR / "validation" / "species.txt",
                          sep="|", low_memory=False)
    sub = species[species["genus"].isin(CROSS_SPECIES_TARGETS.keys())].copy()
    # also keep the exact baseline Daphnia-group (Water Flea common names etc.)
    kw = "Water Flea|Daphnia|Ceriodaphnia|Moina|Simocephalus"
    extra = species[
        species["common_name"].str.contains(kw, case=False, na=False)
        | species["latin_name"].str.contains(kw, case=False, na=False)
    ]
    keep = pd.concat([sub, extra]).drop_duplicates("species_number")
    sp_num_to_genus = dict(zip(keep["species_number"], keep["genus"]))
    sp_num_to_phylum = dict(zip(keep["species_number"], keep["phylum_division"]))
    sp_num_to_class = dict(zip(keep["species_number"], keep["class"]))
    keep_nums = set(keep["species_number"])
    log(f"  {len(keep_nums)} species across {keep['genus'].nunique()} genera")

    log("Loading chemicals.txt (class labels) ...")
    chem = pd.read_csv(ECOTOX_DIR / "validation" / "chemicals.txt",
                       sep="|", low_memory=False, dtype={"cas_number": str})
    cas_to_class = {}
    for cas, grp in zip(chem["cas_number"], chem["ecotox_group"]):
        c = canonical_class(grp)
        if c:
            cas_to_class[str(cas)] = c

    log("Loading tests.txt ...")
    tests = pd.read_csv(
        ECOTOX_DIR / "tests.txt", sep="|",
        usecols=["test_id", "species_number", "test_cas",
                 "exposure_duration_mean", "exposure_duration_unit"],
        low_memory=False, dtype={"test_cas": str})
    tests = tests[tests["species_number"].isin(keep_nums)].copy()
    tests["genus"] = tests["species_number"].map(sp_num_to_genus)
    tests["phylum"] = tests["species_number"].map(sp_num_to_phylum)
    tests["sp_class"] = tests["species_number"].map(sp_num_to_class)
    test_ids = set(tests["test_id"])
    log(f"  {len(test_ids)} tests across transfer genera")

    log("Loading results.txt ...")
    results = pd.read_csv(
        ECOTOX_DIR / "results.txt", sep="|",
        usecols=["result_id", "test_id", "effect", "measurement",
                 "endpoint", "conc1_mean", "obs_duration_mean"],
        low_memory=False)
    results = results[results["test_id"].isin(test_ids)].copy()
    log(f"  {len(results)} result rows, {results['test_id'].nunique()} tests")
    return results, tests, cas_to_class


# ── Per-test feature + label table (identical logic to biomotion_gbm.py) ─────
def build_table(results: pd.DataFrame, tests: pd.DataFrame,
                cas_to_class: dict) -> pd.DataFrame:
    results = results.copy()
    results["conc1_mean"] = pd.to_numeric(results["conc1_mean"], errors="coerce")
    results["effect_pct"] = results["endpoint"].apply(endpoint_to_effect)
    results["endpoint_norm"] = (
        results["endpoint"].astype(str).str.upper().str.strip().str.rstrip("/"))

    tests = tests.copy()
    tests["exposure_duration_mean"] = pd.to_numeric(
        tests["exposure_duration_mean"], errors="coerce")
    cas_map = dict(zip(tests["test_id"], tests["test_cas"]))
    species_map = dict(zip(tests["test_id"], tests["species_number"]))
    genus_map = dict(zip(tests["test_id"], tests["genus"]))
    phylum_map = dict(zip(tests["test_id"], tests["phylum"]))
    dur_val_map = dict(zip(tests["test_id"], tests["exposure_duration_mean"]))
    dur_unit_map = dict(zip(tests["test_id"], tests["exposure_duration_unit"]))

    rows = []
    for test_id, g in results.groupby("test_id"):
        g_valid = g.dropna(subset=["conc1_mean"])
        if len(g_valid) < 1:
            continue

        max_eff = float(g_valid["effect_pct"].max())
        has_hi = g_valid["endpoint_norm"].isin(ANOMALY_ENDPOINTS).any()
        is_anomaly = bool(max_eff > ANOMALY_EFFECT_THRESHOLD or has_hi)

        concs = np.clip(g_valid["conc1_mean"].values.astype(np.float64), 0, None)
        log_conc = np.log1p(concs)
        unique_concs = np.unique(concs)

        cas = cas_map.get(test_id, "")
        feat = {
            "test_id": test_id,
            "test_cas": cas,
            "chem_class": cas_to_class.get(str(cas)),
            "genus": genus_map.get(test_id),
            "phylum": phylum_map.get(test_id),
            "is_anomaly": int(is_anomaly),
            "logconc_median": float(np.median(log_conc)),
            "logconc_max": float(log_conc.max()),
            "logconc_min": float(log_conc.min()),
            "logconc_mean": float(log_conc.mean()),
            "logconc_std": float(log_conc.std()) if len(log_conc) > 1 else 0.0,
            "logconc_range": float(log_conc.max() - log_conc.min()),
            "n_levels": int(len(unique_concs)),
            "n_rows": int(len(g_valid)),
        }
        dur_val = dur_val_map.get(test_id, np.nan)
        dur_unit = str(dur_unit_map.get(test_id, "")).strip().lower()
        if pd.notna(dur_val):
            hours = float(dur_val) * DURATION_UNIT_HOURS.get(dur_unit, 24.0)
            feat["exposure_hours_log"] = float(np.log1p(max(hours, 0.0)))
            feat["exposure_missing"] = 0
        else:
            feat["exposure_hours_log"] = 0.0
            feat["exposure_missing"] = 1

        obs = pd.to_numeric(g_valid["obs_duration_mean"], errors="coerce").dropna()
        feat["obs_duration_log"] = float(np.log1p(obs.max())) if len(obs) else 0.0
        feat["species_number"] = int(species_map.get(test_id, -1))

        chan_counts = np.zeros(N_CHANNELS, dtype=np.float64)
        for m in g_valid["measurement"].astype(str).str.strip():
            chan_counts[BEH_MEASUREMENTS.get(m, 7)] += 1.0
        total = chan_counts.sum()
        if total > 0:
            chan_counts /= total
        for c in range(N_CHANNELS):
            feat[f"chan_{c}"] = float(chan_counts[c])

        eff_present = set(g_valid["effect"].astype(str).str.strip().str.upper())
        for eg in EFFECT_GROUPS:
            feat[f"effgrp_{eg}"] = 1.0 if eg in eff_present else 0.0
        feat["effgrp_n"] = float(len(eff_present))
        rows.append(feat)

    df = pd.DataFrame(rows)
    log(f"Built table: {len(df)} tests, overall anomaly rate {df['is_anomaly'].mean():.3f}")
    return df


META_COLS = {"test_id", "test_cas", "chem_class", "genus", "phylum", "is_anomaly"}


def feature_cols(df: pd.DataFrame, exclude_design: bool = False) -> list[str]:
    cols = [c for c in df.columns if c not in META_COLS]
    if exclude_design:
        cols = [c for c in cols if c not in DESIGN_ONLY_FEATURES]
    return cols


def new_clf() -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=3,
        subsample=0.8, random_state=SEED)


def fit_eval(df_tr: pd.DataFrame, df_te: pd.DataFrame,
             cols: list[str]) -> dict | None:
    ytr = df_tr["is_anomaly"].to_numpy()
    yte = df_te["is_anomaly"].to_numpy()
    # need both classes in train and ≥1 positive+negative in test for AUROC
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        return None
    Xtr = df_tr[cols].to_numpy(dtype=np.float64)
    Xte = df_te[cols].to_numpy(dtype=np.float64)
    clf = new_clf()
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    return {
        "auroc": float(roc_auc_score(yte, p)),
        "auprc": float(average_precision_score(yte, p)),
        "f1": float(f1_score(yte, (p >= 0.5).astype(int))),
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
        "test_pos_rate": float(yte.mean()),
        "train_pos_rate": float(ytr.mean()),
    }


# ── (1) Leave-one-chemical-class-out ─────────────────────────────────────────
def loco_class(df: pd.DataFrame, cols: list[str], min_test: int = 60) -> dict:
    """For each chemical class with >= min_test labeled tests, train on ALL
    other labeled+unlabeled chemistries EXCEPT that class, test on the held-out
    class. Guarantees zero CAS overlap (a CAS belongs to exactly one class)."""
    labeled = df[df["chem_class"].notna()].copy()
    class_counts = labeled["chem_class"].value_counts()
    eligible = [c for c, n in class_counts.items() if n >= min_test]
    per_class = {}
    aurocs = []
    for cls in eligible:
        te = df[df["chem_class"] == cls]
        # train = everything whose chem_class is NOT this class (incl. unlabeled)
        tr = df[df["chem_class"] != cls]
        # zero-CAS-overlap sanity (a CAS is in exactly one class)
        overlap = set(tr["test_cas"]) & set(te["test_cas"])
        assert not overlap, f"CAS leak into held-out class {cls}: {len(overlap)}"
        m = fit_eval(tr, te, cols)
        if m is None:
            per_class[cls] = {"skipped": "single-class in train or test"}
            continue
        m["n_class_tests"] = int(len(te))
        per_class[cls] = m
        aurocs.append(m["auroc"])
    return {
        "min_test_per_class": min_test,
        "n_classes_evaluated": len(aurocs),
        "classes": eligible,
        "per_class": per_class,
        "macro_auroc_mean": float(np.mean(aurocs)) if aurocs else None,
        "macro_auroc_std": float(np.std(aurocs)) if aurocs else None,
        "auroc_min": float(np.min(aurocs)) if aurocs else None,
        "auroc_max": float(np.max(aurocs)) if aurocs else None,
    }


# ── (2) Cross-species / cross-taxon ──────────────────────────────────────────
def cross_species(df: pd.DataFrame, cols: list[str],
                  min_test: int = 80) -> dict:
    """Train on Daphnia only; test on each OTHER genus with >= min_test tests.
    Also an all-non-Daphnia pooled test."""
    train = df[df["genus"] == "Daphnia"]
    per_genus = {}
    aurocs = []
    for genus, (phylum, cls_, desc) in CROSS_SPECIES_TARGETS.items():
        if genus == "Daphnia":
            continue
        te = df[df["genus"] == genus]
        if len(te) < min_test:
            continue
        m = fit_eval(train, te, cols)
        if m is None:
            per_genus[genus] = {"skipped": "single-class test set",
                                "n_test": int(len(te)), "desc": desc}
            continue
        m["phylum"] = phylum
        m["taxon_class"] = cls_
        m["desc"] = desc
        # taxonomic distance flag: same class as Daphnia (Branchiopoda) or not
        m["same_class_as_daphnia"] = (cls_ == "Branchiopoda")
        per_genus[genus] = m
        aurocs.append(m["auroc"])
    # pooled non-Daphnia (excluding the other two cladocerans to force cross-taxon)
    cross_taxa = [g for g, (p, c, d) in CROSS_SPECIES_TARGETS.items()
                  if c != "Branchiopoda"]
    te_pool = df[df["genus"].isin(cross_taxa)]
    pooled = fit_eval(train, te_pool, cols)
    return {
        "train_genus": "Daphnia",
        "train_n": int(len(train)),
        "train_pos_rate": float(train["is_anomaly"].mean()),
        "min_test_per_genus": min_test,
        "per_genus": per_genus,
        "n_genera_evaluated": len(aurocs),
        "macro_auroc_mean": float(np.mean(aurocs)) if aurocs else None,
        "macro_auroc_std": float(np.std(aurocs)) if aurocs else None,
        "auroc_min": float(np.min(aurocs)) if aurocs else None,
        "auroc_max": float(np.max(aurocs)) if aurocs else None,
        "pooled_cross_taxon": pooled,
        "pooled_cross_taxon_genera": cross_taxa,
    }


# ── (3) Feature attribution: signal vs study-design artifact ─────────────────
def within_daphnia_grouped_auroc(df: pd.DataFrame, cols: list[str],
                                 n_seeds: int = 5) -> dict:
    """Chemical-grouped split WITHIN Daphnia (matches baseline scope) for a
    fair full-vs-ablated comparison."""
    from sklearn.model_selection import GroupShuffleSplit
    d = df[df["genus"] == "Daphnia"].reset_index(drop=True)
    groups = d["test_cas"].to_numpy()
    aurocs = []
    for s in range(n_seeds):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=s)
        tr, te = next(gss.split(d, d["is_anomaly"], groups))
        m = fit_eval(d.iloc[tr], d.iloc[te], cols)
        if m:
            aurocs.append(m["auroc"])
    return {"auroc_mean": float(np.mean(aurocs)),
            "auroc_std": float(np.std(aurocs)), "n_seeds": n_seeds}


def main() -> None:
    t0 = time.time()
    log("=== BioMotion transfer battery ===")
    results, tests, cas_to_class = load_all()
    df = build_table(results, tests, cas_to_class)

    # guardrail: no banned feature columns
    banned = {"endpoint", "effect_pct", "effect_pct_mean", "significance_code",
              "endpoint_norm", "max_eff"}
    assert not banned.intersection(df.columns), "GUARDRAIL VIOLATION"

    full_cols = feature_cols(df, exclude_design=False)
    bio_cols = feature_cols(df, exclude_design=True)  # drop study-design features
    log(f"Full feature set: {len(full_cols)} | biology-only set: {len(bio_cols)}")

    genus_counts = df["genus"].value_counts().to_dict()
    class_counts = df[df["chem_class"].notna()]["chem_class"].value_counts().to_dict()

    # ── (1) LOCO-class ───────────────────────────────────────────────────────
    log("\n--- (1) Leave-one-chemical-class-out (FULL features) ---")
    loco_full = loco_class(df, full_cols)
    log(f"  macro AUROC = {loco_full['macro_auroc_mean']} "
        f"over {loco_full['n_classes_evaluated']} classes")
    log("--- (1b) Leave-one-chemical-class-out (BIOLOGY-only features) ---")
    loco_bio = loco_class(df, bio_cols)
    log(f"  macro AUROC = {loco_bio['macro_auroc_mean']}")

    # ── (2) Cross-species ────────────────────────────────────────────────────
    log("\n--- (2) Cross-species: train Daphnia (FULL features) ---")
    xsp_full = cross_species(df, full_cols)
    log(f"  macro AUROC = {xsp_full['macro_auroc_mean']} "
        f"over {xsp_full['n_genera_evaluated']} genera; "
        f"pooled cross-taxon AUROC = "
        f"{xsp_full['pooled_cross_taxon']['auroc'] if xsp_full['pooled_cross_taxon'] else None}")
    log("--- (2b) Cross-species (BIOLOGY-only features) ---")
    xsp_bio = cross_species(df, bio_cols)
    log(f"  macro AUROC = {xsp_bio['macro_auroc_mean']}")

    # ── (3) Attribution: full vs biology-only within-Daphnia grouped ─────────
    log("\n--- (3) Feature attribution (within-Daphnia grouped) ---")
    within_full = within_daphnia_grouped_auroc(df, full_cols)
    within_bio = within_daphnia_grouped_auroc(df, bio_cols)
    log(f"  within-Daphnia grouped AUROC: full={within_full['auroc_mean']:.4f} "
        f"biology-only={within_bio['auroc_mean']:.4f}")

    # importance of design vs biology from a within-Daphnia model
    d = df[df["genus"] == "Daphnia"]
    clf = new_clf()
    clf.fit(d[full_cols].to_numpy(dtype=np.float64), d["is_anomaly"].to_numpy())
    imp = dict(zip(full_cols, clf.feature_importances_))
    design_imp = sum(v for k, v in imp.items() if k in DESIGN_ONLY_FEATURES)
    bio_imp = sum(v for k, v in imp.items() if k not in DESIGN_ONLY_FEATURES)
    top_imp = sorted(imp.items(), key=lambda z: z[1], reverse=True)[:12]

    # single-feature obs_duration_log AUROC (how much a pure design artifact
    # alone explains the label) — within-Daphnia random hold-out
    from sklearn.model_selection import train_test_split
    dd = d.reset_index(drop=True)
    ytr_idx, yte_idx = train_test_split(
        np.arange(len(dd)), test_size=0.2, random_state=SEED,
        stratify=dd["is_anomaly"])
    obs_only = fit_eval(dd.iloc[ytr_idx], dd.iloc[yte_idx], ["obs_duration_log"])

    out = {
        "title": "BioMotion out-of-distribution transfer battery",
        "data_source": "REAL EPA ECOTOX (ecotox_ascii_03_12_2026), no synthetic.",
        "label_definition": "BioMotion v2: is_anomaly = (max endpoint-derived "
                            "effect_pct > 20) OR (endpoint in {LOEC,LOEL,EC50,"
                            "EC75,EC90,EC100,LC50,LC90,LC100}). Label from "
                            "`endpoint` only.",
        "guardrail": "endpoint / effect_pct / significance_code are NEVER "
                     "features. Features = concentration summary, exposure/obs "
                     "duration, measurement-channel & effect-group one-hots, "
                     "species id.",
        "baselines": {
            "diffusion_random_split_auroc": 0.8067,
            "gbm_random_split_auroc": 0.8717,
            "gbm_random_CAS_grouped_auroc": 0.8528,
            "note": "The 0.85 baseline holds out RANDOM chemicals. The tests "
                    "below hold out whole chemical CLASSES and whole TAXA.",
        },
        "dataset": {
            "n_tests_total": int(len(df)),
            "overall_anomaly_rate": float(df["is_anomaly"].mean()),
            "n_features_full": len(full_cols),
            "n_features_biology_only": len(bio_cols),
            "design_only_features": sorted(DESIGN_ONLY_FEATURES),
            "genus_test_counts": genus_counts,
            "chemical_class_test_counts": class_counts,
        },
        "test_1_leave_one_chemical_class_out": {
            "full_features": loco_full,
            "biology_only_features": loco_bio,
        },
        "test_2_cross_species_transfer": {
            "full_features": xsp_full,
            "biology_only_features": xsp_bio,
        },
        "test_3_feature_attribution": {
            "within_daphnia_grouped_full_auroc": within_full,
            "within_daphnia_grouped_biology_only_auroc": within_bio,
            "design_feature_total_importance": float(design_imp),
            "biology_feature_total_importance": float(bio_imp),
            "top_feature_importances": [
                {"feature": k, "importance": float(v)} for k, v in top_imp],
            "obs_duration_log_alone_auroc": (
                obs_only["auroc"] if obs_only else None),
        },
        "elapsed_sec": round(time.time() - t0, 1),
    }
    OUT_PATH.write_text(json.dumps(out, indent=2, default=str))
    log(f"\nResults -> {OUT_PATH}")

    print("\n" + "=" * 74)
    print("BioMotion transfer — honest OOD AUROC (baseline random-CAS = 0.853)")
    print("-" * 74)
    print(f"  (1) leave-one-chemical-class-out  full    : "
          f"{loco_full['macro_auroc_mean']:.4f}  "
          f"[{loco_full['auroc_min']:.3f}, {loco_full['auroc_max']:.3f}]")
    print(f"      leave-one-chemical-class-out  bio-only : "
          f"{loco_bio['macro_auroc_mean']:.4f}")
    print(f"  (2) cross-species (train Daphnia) full     : "
          f"{xsp_full['macro_auroc_mean']:.4f}  "
          f"[{xsp_full['auroc_min']:.3f}, {xsp_full['auroc_max']:.3f}]")
    print(f"      pooled cross-taxon AUROC               : "
          f"{xsp_full['pooled_cross_taxon']['auroc']:.4f}")
    print(f"  (3) design-feature importance share        : "
          f"{design_imp:.3f}  (obs_duration alone AUROC "
          f"{obs_only['auroc']:.3f})")
    print(f"      within-Daphnia grouped: full={within_full['auroc_mean']:.3f}"
          f"  bio-only={within_bio['auroc_mean']:.3f}")
    print("=" * 74)


if __name__ == "__main__":
    main()
