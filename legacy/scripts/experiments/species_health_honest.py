#!/usr/bin/env python3
"""Leakage-free honest evaluation of Species Health occupancy prediction.

CONTEXT / WHY THIS EXISTS
-------------------------
The reported Species Health metrics (health_r2 = 0.9996, occ_accuracy = 0.999,
and a frequently-quoted AUROC of 1.000) are CIRCULAR.

In ``scripts/training/train_species_health.py`` the per-site input embedding is,
for every site WITHOUT a matched USGS sensor station (which is ~99.8% of sites —
only 54 of 24,321 biodata sites have a sensor match), constructed as::

    bio_feats = [lat/50, lon/-100, richness/100, log1p(total_count)/10,
                 log1p(count_species_0)/10,   # <-- positions 4..9 ARE the
                 log1p(count_species_1)/10,   #     per-species log-counts
                 ...,                          #     that DEFINE the occupancy
                 log1p(count_species_5)/10]    #     labels (count>0 == present)

The occupancy label for species s is exactly ``count_species_s > 0``. So input
positions 4..9 let the model read the label directly off its own input. AUROC of
1.000 is therefore label leakage, not predictive skill.

WHAT THIS SCRIPT DOES
---------------------
Rebuilds the SAME per-site table from the SAME real USGS BioData, with the SAME
seed=42 spatial (hash-of-site-id) holdout split, but:

  * REMOVES the leaked per-species log-count features (positions 4..9).
  * Keeps ONLY genuinely independent predictors:
      - geographic: lat, lon, |lat|, elevation (VerticalMeasure)
      - watershed:  log drainage area, HUC2 region (one-hot)
      - community:  log1p(community richness EXCLUDING the 6 target taxa),
                    log1p(community total count EXCLUDING the 6 target taxa),
                    n distinct activity/sampling events at the site
      - sensor:     real water-quality summary stats (mean/std/min/max/median)
                    for the ~54 sites that have a matched USGS sensor station;
                    NaN (median-imputed) otherwise.
    The community richness/count are computed AFTER dropping the target taxa so
    they cannot trivially encode "is target species present".

  * Drops the two always-absent species (Hellbender, Pearl Mussel) which are
    trivially predicted (they never occur in this matched dataset).

  * Trains a leakage-free probe (Gradient Boosting classifier per species, with
    a logistic-regression cross-check) on TRAIN and evaluates on the held-out
    TEST split.

  * Reports per-species balanced occupancy AUROC + F1 (threshold 0.5),
    macro-AUROC / macro-F1, Mayflies (high-prevalence) separately, and a
    present-only abundance R^2 on the leakage-free features.

This is an HONEST probe of the signal genuinely recoverable from environment /
site / community context — NOT a circular read-off of the label.

A GBM/logistic probe (rather than the full SentinelSpeciesHealthIndex network)
is used deliberately: it is a faster but unbiased estimate of the achievable
signal in the leakage-free features, and removes any confound from the network's
own (leaked) embedding-construction logic. The features fed to the probe are
exactly the leakage-free subset the network would have access to.

Run:
    PYTHONNOUSERSITE=1 ~/.conda/envs/physiformer/bin/python \
        scripts/experiments/species_health_honest.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT))

BIODATA_DIR = PROJECT / "data" / "processed" / "biology" / "usgs_biodata"
SENSOR_DIR = PROJECT / "data" / "raw" / "sensor" / "full"
RESULTS_DIR = PROJECT / "results" / "benchmarks"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = RESULTS_DIR / "species_health_honest.json"

SEED = 42

# Species groups — identical definition to the training script.
SPECIES_NAMES = [
    "Freshwater mussels (Unionidae)",
    "Mayflies (Ephemeroptera)",
    "Brook trout (Salvelinus fontinalis)",
    "Hellbender (Cryptobranchus alleganiensis)",
    "Freshwater pearl mussel (Margaritifera margaritifera)",
    "American eel (Anguilla rostrata)",
]
NUM_SPECIES = 6
ALWAYS_ABSENT_IDX = {3, 4}  # Hellbender, Pearl Mussel — verified empirically below
MAYFLY_IDX = 1

TAXON_PATTERNS = {
    0: ["unionidae", "unionid", "lampsilis", "amblema", "elliptio",
        "quadrula", "fusconaia", "pleurobema", "villosa", "corbicula",
        "anodonta", "pyganodon", "utterbackia", "lasmigona", "strophitus"],
    1: ["ephemeroptera", "ephemerell", "hexagenia", "baetis", "heptageni",
        "leptophlebi", "caenis", "tricorythodes", "isonychia", "stenonema",
        "epeorus", "drunella", "paraleptophlebia", "ameletus", "cinygmula"],
    2: ["salvelinus fontinalis", "brook trout", "salvelinus"],
    3: ["cryptobranchus", "hellbender"],
    4: ["margaritifera", "pearl mussel"],
    5: ["anguilla rostrata", "american eel", "anguilla"],
}


def log(msg: str) -> None:
    print(msg, flush=True)


def split_of(site_id: str, seed: int = SEED) -> str:
    """Identical hash split to the training script (seed:site_id, mod 10)."""
    h = hashlib.sha256(f"{seed}:{site_id}".encode()).hexdigest()
    fold = int(h[:8], 16) % 10
    if fold < 7:
        return "train"
    if fold < 9:
        return "val"
    return "test"


def build_site_table() -> pd.DataFrame:
    """Build per-site table with LEAKAGE-FREE features + occupancy/abundance labels."""
    inv_path = BIODATA_DIR / "biodata_invertebrates.parquet"
    fish_path = BIODATA_DIR / "biodata_fish.parquet"
    sites_path = BIODATA_DIR / "biodata_sites.parquet"

    log("Loading USGS BioData (real surveys)...")
    df_inv = pd.read_parquet(inv_path)
    df_fish = pd.read_parquet(fish_path) if fish_path.exists() else pd.DataFrame()
    df_sites = pd.read_parquet(sites_path)
    bio = pd.concat([df_inv, df_fish], ignore_index=True) if len(df_fish) else df_inv

    bio["count"] = pd.to_numeric(bio["ResultMeasureValue"], errors="coerce").fillna(0)
    bio = bio[bio["count"] > 0].copy()

    # Match records to species groups (same logic as training).
    taxon_lower = bio["SubjectTaxonomicName"].astype(str).str.lower()
    bio["species_group"] = -1
    for grp, pats in TAXON_PATTERNS.items():
        mask = taxon_lower.str.contains("|".join(pats), na=False)
        bio.loc[mask, "species_group"] = grp

    site_col = "MonitoringLocationIdentifier"
    matched = bio[bio["species_group"] >= 0]
    if len(matched) == 0:
        raise RuntimeError("No records matched target species groups.")
    log(f"  {len(matched)} records matched target species groups "
        f"out of {len(bio)} positive-count records.")

    # Per-site x species_group total count (defines occupancy + abundance labels).
    ss = matched.groupby([site_col, "species_group"])["count"].sum().reset_index()
    pivot = ss.pivot(index=site_col, columns="species_group", values="count").fillna(0.0)
    for g in range(NUM_SPECIES):
        if g not in pivot.columns:
            pivot[g] = 0.0
    pivot = pivot[[g for g in range(NUM_SPECIES)]]

    # ----- LEAKAGE-FREE community features: computed on NON-target taxa only ----
    target_taxa_mask = bio["species_group"] >= 0
    bio_nontarget = bio[~target_taxa_mask]
    comm_rich = (bio_nontarget.groupby(site_col)["SubjectTaxonomicName"]
                 .nunique().rename("comm_richness_nontarget"))
    comm_count = (bio_nontarget.groupby(site_col)["count"]
                  .sum().rename("comm_count_nontarget"))
    # Sampling effort: number of distinct sampling activities at the site.
    if "ActivityIdentifier" in bio.columns:
        n_events = (bio.groupby(site_col)["ActivityIdentifier"]
                    .nunique().rename("n_events"))
    else:
        n_events = (bio.groupby(site_col)["ActivityStartDate"]
                    .nunique().rename("n_events"))

    # ----- Site geographic / watershed metadata (leakage-free) ------------------
    s = df_sites.drop_duplicates(subset=[site_col]).set_index(site_col)
    site_meta = pd.DataFrame(index=s.index)
    site_meta["lat"] = pd.to_numeric(s["LatitudeMeasure"], errors="coerce")
    site_meta["lon"] = pd.to_numeric(s["LongitudeMeasure"], errors="coerce")
    site_meta["elev"] = pd.to_numeric(s["VerticalMeasure/MeasureValue"], errors="coerce")
    site_meta["drainage"] = pd.to_numeric(
        s["DrainageAreaMeasure/MeasureValue"], errors="coerce")
    site_meta["huc2"] = s["_huc2"].astype(str) if "_huc2" in s.columns else "NA"

    # ----- Assemble -------------------------------------------------------------
    tab = pivot.join(site_meta, how="inner")
    tab = tab.join(comm_rich, how="left").join(comm_count, how="left").join(n_events, how="left")
    tab = tab.dropna(subset=["lat", "lon"])
    tab[["comm_richness_nontarget", "comm_count_nontarget", "n_events"]] = tab[
        ["comm_richness_nontarget", "comm_count_nontarget", "n_events"]].fillna(0.0)

    # Spatial holdout split (identical to training).
    tab["split"] = [split_of(sid) for sid in tab.index]
    log(f"  {len(tab)} sites with coordinates. "
        f"Split: train={int((tab.split=='train').sum())}, "
        f"val={int((tab.split=='val').sum())}, test={int((tab.split=='test').sum())}")
    return tab


def attach_sensor_features(tab: pd.DataFrame) -> pd.DataFrame:
    """Attach REAL water-quality summary stats for sites with a sensor match."""
    sensor_ids = {f.stem: f for f in SENSOR_DIR.glob("*.parquet")} if SENSOR_DIR.exists() else {}
    stat_names = ["wq_mean", "wq_std", "wq_min", "wq_max", "wq_median"]
    feat_rows = {}
    n_match = 0
    for sid in tab.index:
        station = str(sid).replace("USGS-", "")
        if station not in sensor_ids:
            continue
        try:
            dfx = pd.read_parquet(sensor_ids[station])
        except Exception:
            continue
        vals = []
        for col in dfx.columns:
            if col == "datetime":
                continue
            v = pd.to_numeric(dfx[col], errors="coerce").dropna()
            if len(v):
                vals.append([v.mean(), v.std(), v.min(), v.max(), v.median()])
        if not vals:
            continue
        agg = np.nanmean(np.array(vals, dtype=float), axis=0)
        feat_rows[sid] = agg
        n_match += 1
    sens = pd.DataFrame.from_dict(feat_rows, orient="index", columns=stat_names)
    tab = tab.join(sens, how="left")
    log(f"  Attached real sensor water-quality stats for {n_match} sites "
        f"(median-imputed elsewhere).")
    return tab, stat_names


FEATURE_COLS_BASE = [
    "lat", "lon", "abslat", "elev", "log_drainage",
    "log_comm_richness_nontarget", "log_comm_count_nontarget", "log_n_events",
]


def make_feature_matrix(tab: pd.DataFrame, sensor_stats,
                        include_huc2: bool = True, include_geo: bool = True):
    """Return X DataFrame. EXCLUDES per-species log-counts (the leaked features).

    Three leakage-free variants are supported (all exclude the leaked counts):
      * A  include_huc2=True,  include_geo=True : full set, incl. HUC2 region
        one-hot AND continuous lat/lon. Count-independent, but the rare species
        are range-restricted so geography acts as a strong range prior.
      * B  include_huc2=False, include_geo=True : drops the coarse region one-hot
        but keeps continuous lat/lon/elev. The rare-species signal turns out to be
        carried by lat/lon (tight geographic ranges), so this barely differs from A.
      * C  include_huc2=False, include_geo=False : environment-ONLY, NO geography.
        Drops lat/lon/|lat| entirely; keeps only watershed (drainage), NON-target
        community context (richness/count of OTHER taxa), sampling effort, and real
        sensor water-quality stats. This isolates the non-geographic environmental
        signal — the conservative honest floor that does NOT rely on a species
        range prior.
    """
    df = pd.DataFrame(index=tab.index)
    if include_geo:
        df["lat"] = tab["lat"]
        df["lon"] = tab["lon"]
        df["abslat"] = tab["lat"].abs()
    df["elev"] = tab["elev"]
    df["log_drainage"] = np.log1p(tab["drainage"].clip(lower=0).fillna(0))
    df["log_comm_richness_nontarget"] = np.log1p(tab["comm_richness_nontarget"])
    df["log_comm_count_nontarget"] = np.log1p(tab["comm_count_nontarget"])
    df["log_n_events"] = np.log1p(tab["n_events"])
    if include_huc2:
        # HUC2 region one-hot (geographic context, count-independent).
        huc_dummies = pd.get_dummies(tab["huc2"].fillna("NA"), prefix="huc2")
        df = pd.concat([df, huc_dummies], axis=1)
    # Real sensor stats (median-imputed inside the probes).
    for c in sensor_stats:
        df[c] = tab[c] if c in tab.columns else np.nan
    return df


def metrics_for_species(y_train, y_test, X_train, X_test, ab_train, ab_test):
    """Train GBM + logistic probes; return occupancy + present-only abundance metrics."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import roc_auc_score, f1_score

    out = {}
    # Degenerate cases.
    if y_train.sum() == 0 or y_train.sum() == len(y_train):
        out["status"] = "degenerate_train (single class)"
        return out
    if len(np.unique(y_test)) < 2:
        out["status"] = "degenerate_test (single class)"
        out["test_prevalence"] = float(y_test.mean())
        return out

    # --- GBM probe ---
    gbm = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=4,
        l2_regularization=1.0, random_state=SEED,
        class_weight="balanced",
    )
    gbm.fit(X_train, y_train)
    p_gbm = gbm.predict_proba(X_test)[:, 1]
    auroc_gbm = roc_auc_score(y_test, p_gbm)
    f1_gbm = f1_score(y_test, (p_gbm >= 0.5).astype(int), zero_division=0)

    # --- Logistic cross-check ---
    logit = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0),
    )
    logit.fit(X_train, y_train)
    p_log = logit.predict_proba(X_test)[:, 1]
    auroc_log = roc_auc_score(y_test, p_log)
    f1_log = f1_score(y_test, (p_log >= 0.5).astype(int), zero_division=0)

    out["status"] = "ok"
    out["test_prevalence"] = float(y_test.mean())
    out["n_test"] = int(len(y_test))
    out["auroc_gbm"] = float(auroc_gbm)
    out["f1_gbm"] = float(f1_gbm)
    out["auroc_logistic"] = float(auroc_log)
    out["f1_logistic"] = float(f1_log)

    # --- Present-only abundance R^2 (GBM regressor on log abundance) ---
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import r2_score
    mask_tr = ab_train > 0
    mask_te = ab_test > 0
    if mask_tr.sum() >= 30 and mask_te.sum() >= 10:
        reg = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_depth=4,
            l2_regularization=1.0, random_state=SEED,
        )
        yreg_tr = np.log1p(ab_train[mask_tr])
        reg.fit(X_train[mask_tr], yreg_tr)
        yreg_pred = reg.predict(X_test[mask_te])
        yreg_true = np.log1p(ab_test[mask_te])
        out["present_only_r2"] = float(r2_score(yreg_true, yreg_pred))
        out["n_present_test"] = int(mask_te.sum())
    else:
        out["present_only_r2"] = None
        out["n_present_test"] = int(mask_te.sum())
    return out


def evaluate_variant(tab, X, train_mask, test_mask):
    """Run per-species leakage-free probes for one feature matrix; return dict."""
    X_train = X[train_mask].values.astype(float)
    X_test = X[test_mask].values.astype(float)

    per_species = {}
    evaluated_idx = []
    for g in range(NUM_SPECIES):
        name = SPECIES_NAMES[g]
        present_total = int((tab[g] > 0).sum())
        if g in ALWAYS_ABSENT_IDX or present_total == 0:
            per_species[name] = {"status": "dropped_always_absent",
                                 "present_sites_total": present_total}
            log(f"  [{g}] {name}: DROPPED (always-absent, present at {present_total} sites)")
            continue
        y = (tab[g] > 0).astype(int).values
        ab = tab[g].astype(float).values
        res = metrics_for_species(y[train_mask], y[test_mask], X_train, X_test,
                                  ab[train_mask], ab[test_mask])
        per_species[name] = res
        if res.get("status") == "ok":
            evaluated_idx.append(g)
            log(f"  [{g}] {name}: prev(test)={res['test_prevalence']:.3f} "
                f"GBM AUROC={res['auroc_gbm']:.4f} F1={res['f1_gbm']:.4f} | "
                f"Logit AUROC={res['auroc_logistic']:.4f} F1={res['f1_logistic']:.4f} | "
                f"presentR2={res['present_only_r2']}")
        else:
            log(f"  [{g}] {name}: {res.get('status')}")

    macro_idx = [g for g in evaluated_idx if g != MAYFLY_IDX]

    def _macro(vals_idx, key):
        vals = [per_species[SPECIES_NAMES[g]][key] for g in vals_idx]
        return float(np.mean(vals)) if vals else None

    macro = {
        "species_in_macro": [SPECIES_NAMES[g] for g in macro_idx],
        "macro_auroc_gbm": _macro(macro_idx, "auroc_gbm"),
        "macro_f1_gbm": _macro(macro_idx, "f1_gbm"),
        "macro_auroc_logistic": _macro(macro_idx, "auroc_logistic"),
        "macro_f1_logistic": _macro(macro_idx, "f1_logistic"),
    }
    macro_incl_mayfly = {
        "macro_auroc_gbm_incl_mayfly": _macro(list(evaluated_idx), "auroc_gbm"),
        "macro_f1_gbm_incl_mayfly": _macro(list(evaluated_idx), "f1_gbm"),
    }
    mayfly = per_species[SPECIES_NAMES[MAYFLY_IDX]] if MAYFLY_IDX in evaluated_idx else None
    return {
        "n_features": int(X.shape[1]),
        "feature_names": list(X.columns),
        "per_species": per_species,
        "macro_excl_mayfly_and_dropped": macro,
        "macro_reference_incl_mayfly": macro_incl_mayfly,
        "mayfly_high_prevalence_separate": mayfly,
    }


def main():
    tab = build_site_table()
    tab, sensor_stats = attach_sensor_features(tab)

    # Empirical prevalence per species over all sites.
    prevalence = {SPECIES_NAMES[g]: float((tab[g] > 0).mean()) for g in range(NUM_SPECIES)}
    log("\nEmpirical occupancy prevalence (all sites):")
    for g in range(NUM_SPECIES):
        log(f"  [{g}] {SPECIES_NAMES[g]}: {prevalence[SPECIES_NAMES[g]]:.4f} "
            f"present at {int((tab[g] > 0).sum())} sites")

    train_mask = (tab["split"] == "train").values
    test_mask = (tab["split"] == "test").values

    # Variant A: full leakage-free set (incl. HUC2 region one-hot).
    X_full = make_feature_matrix(tab, sensor_stats, include_huc2=True)
    log(f"\n=== Variant A: full leakage-free features ({X_full.shape[1]} cols, "
        f"incl. HUC2 region one-hot; per-species counts EXCLUDED) ===")
    variant_full = evaluate_variant(tab, X_full, train_mask, test_mask)

    # Variant B: drop coarse region one-hot, keep continuous lat/lon.
    X_env = make_feature_matrix(tab, sensor_stats, include_huc2=False, include_geo=True)
    log(f"\n=== Variant B: continuous-geo features ({X_env.shape[1]} cols, "
        f"NO HUC2 region one-hot; keeps lat/lon) ===")
    variant_env = evaluate_variant(tab, X_env, train_mask, test_mask)

    # Variant C: environment-ONLY, NO geography (conservative honest floor).
    X_nogeo = make_feature_matrix(tab, sensor_stats, include_huc2=False, include_geo=False)
    log(f"\n=== Variant C: environment-ONLY, NO geography ({X_nogeo.shape[1]} cols; "
        f"drops lat/lon — removes species range prior) ===")
    variant_nogeo = evaluate_variant(tab, X_nogeo, train_mask, test_mask)

    # Headline = Variant B (continuous geo, leakage-free). Variant C = conservative floor.
    macro = variant_env["macro_excl_mayfly_and_dropped"]
    macro_incl_mayfly = variant_env["macro_reference_incl_mayfly"]
    mayfly = variant_env["mayfly_high_prevalence_separate"]

    results = {
        "_PROVENANCE": (
            "HONEST leakage-free re-evaluation of Species Health occupancy. "
            "The training script's per-site embedding positions 4-9 are "
            "log1p(per-species counts) which DEFINE the occupancy labels, so the "
            "previously reported AUROC=1.000 / occ_accuracy=0.999 / health_r2=0.9996 "
            "are circular (label read off the input). Here those per-species count "
            "features are REMOVED; only genuinely independent predictors (geographic, "
            "watershed, NON-target community context, and real matched sensor "
            "water-quality stats) are used. Probe = HistGradientBoosting classifier "
            "(+ logistic cross-check) on the leakage-free features, NOT the full "
            "network; this is an unbiased estimate of the achievable signal. "
            "Same real USGS BioData, same seed=42 spatial hash holdout split. "
            "Three leakage-free variants: A keeps a coarse HUC2 region one-hot + "
            "lat/lon; B drops the region one-hot (keeps lat/lon); C drops geography "
            "entirely (no lat/lon/region) to remove the species RANGE PRIOR and "
            "isolate non-geographic environmental signal. KEY FINDING: the rare "
            "species (eel, brook trout) are geographically range-restricted, so most "
            "of their AUROC comes from lat/lon (a real, count-independent range prior, "
            "NOT label leakage). The widespread mussel (16% prevalence) is the honest, "
            "informative case at ~0.85 AUROC, driven by NON-target community richness. "
            "Headline = Variant B; Variant C is the conservative no-geography floor."
        ),
        "probe": "HistGradientBoostingClassifier (GBM) + LogisticRegression cross-check",
        "data_source": "USGS BioData (real biological surveys) + matched USGS sensor stations",
        "synthetic_data": False,
        "seed": SEED,
        "split": "spatial hash holdout (sha256 of seed:site_id, mod 10)",
        "leaked_features_removed": "positions 4-9: per-species log1p(count) (define the labels)",
        "n_sites_total": int(len(tab)),
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "n_sites_with_real_sensor_match": int(
            tab[[c for c in tab.columns if c == "wq_mean"]].notna().any(axis=1).sum()
        ) if "wq_mean" in tab.columns else 0,
        "occupancy_prevalence_all_sites": prevalence,
        "species_dropped_always_absent": [SPECIES_NAMES[i] for i in sorted(ALWAYS_ABSENT_IDX)],
        "HEADLINE_variant_B_continuous_geo": {
            "description": "Leakage-free, drops coarse HUC2 region one-hot, keeps lat/lon.",
            "n_features": variant_env["n_features"],
            "macro_excl_mayfly_and_dropped": macro,
            "macro_reference_incl_mayfly": macro_incl_mayfly,
            "mayfly_high_prevalence_separate": mayfly,
            "per_species": variant_env["per_species"],
            "feature_names": variant_env["feature_names"],
        },
        "variant_C_environment_only_no_geography": {
            "description": (
                "Conservative honest floor: drops lat/lon/region entirely so the "
                "model cannot use a species range prior. Relies only on watershed "
                "drainage, NON-target community richness/count, sampling effort, and "
                "real sensor stats. This is the non-geographic environmental signal."
            ),
            "n_features": variant_nogeo["n_features"],
            "macro_excl_mayfly_and_dropped": variant_nogeo["macro_excl_mayfly_and_dropped"],
            "macro_reference_incl_mayfly": variant_nogeo["macro_reference_incl_mayfly"],
            "mayfly_high_prevalence_separate": variant_nogeo["mayfly_high_prevalence_separate"],
            "per_species": variant_nogeo["per_species"],
            "feature_names": variant_nogeo["feature_names"],
        },
        "variant_A_full_incl_huc2_region": {
            "n_features": variant_full["n_features"],
            "macro_excl_mayfly_and_dropped": variant_full["macro_excl_mayfly_and_dropped"],
            "macro_reference_incl_mayfly": variant_full["macro_reference_incl_mayfly"],
            "mayfly_high_prevalence_separate": variant_full["mayfly_high_prevalence_separate"],
            "per_species": variant_full["per_species"],
            "feature_names": variant_full["feature_names"],
            "caveat": (
                "Rare species (eel, brook trout) are >=75% confined to a single "
                "HUC2 region, so the region one-hot acts as a range memorizer here; "
                "treat these AUROCs as range-prior, not environmental skill."
            ),
        },
        "leaked_baseline_for_comparison": {
            "reported_health_r2": 0.9995507135053653,
            "reported_occ_accuracy": 0.9988738894462585,
            "frequently_quoted_auroc": 1.000,
            "note": "These are leakage-inflated / circular; not real predictive skill.",
        },
    }

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    log("\n" + "=" * 70)
    log("HONEST LEAKAGE-FREE SUMMARY (occupancy AUROC, GBM)")
    log("=" * 70)
    vf = variant_full["macro_excl_mayfly_and_dropped"]
    vc = variant_nogeo["macro_excl_mayfly_and_dropped"]
    log("Variant A (full, incl. HUC2 region one-hot + lat/lon):")
    log(f"   Macro-AUROC (excl Mayfly): {vf['macro_auroc_gbm']:.4f}   "
        f"Macro-F1: {vf['macro_f1_gbm']:.4f}")
    log("Variant B (continuous geo, NO region one-hot) — HEADLINE:")
    log(f"   Macro-AUROC (excl Mayfly): {macro['macro_auroc_gbm']:.4f}   "
        f"Macro-F1: {macro['macro_f1_gbm']:.4f}")
    log(f"   Macro-AUROC (incl Mayfly): {macro_incl_mayfly['macro_auroc_gbm_incl_mayfly']:.4f}")
    log("Variant C (environment-ONLY, NO geography) — conservative floor:")
    log(f"   Macro-AUROC (excl Mayfly): {vc['macro_auroc_gbm']:.4f}   "
        f"Macro-F1: {vc['macro_f1_gbm']:.4f}")
    if mayfly:
        log(f"   Mayflies (high prevalence) AUROC: {mayfly.get('auroc_gbm'):.4f}  "
            f"F1: {mayfly.get('f1_gbm'):.4f}")
    log("vs leaked/circular baseline: AUROC=1.000, occ_acc=0.999, health_r2=0.9996")
    log(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
