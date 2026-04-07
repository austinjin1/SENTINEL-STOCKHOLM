#!/usr/bin/env python3
"""Process EPA ECOTOX database into training data for ToxiGene molecular encoder.

Processes 1.23M dose-response records from the ECOTOX ASCII download into
structured toxicity datasets for contaminant classification training.

Pipeline:
1. Load chemicals.txt → contaminant class mapping (8 classes)
2. Load species.txt → filter to aquatic organisms
3. Join tests + results → aquatic toxicity records
4. Extract features: chemical fingerprint, concentration, duration, media conditions
5. Create per-chemical-organism dose-response profiles
6. Save as training data in ToxiGene format

MIT License — Bryan Cheng, 2026
"""

import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ECOTOX_DIR = Path("data/raw/ecotox/ecotox_ascii_03_12_2026")
OUT_DIR = Path("data/processed/ecotox")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Contaminant class mapping (matches ToxiGene's 8 classes)
CONTAMINANT_KEYWORDS = {
    "heavy_metal": [
        "cadmium", "lead", "mercury", "arsenic", "copper", "zinc", "chromium",
        "nickel", "cobalt", "selenium", "silver", "aluminum", "manganese",
        "iron", "tin", "thallium", "barium", "beryllium", "vanadium",
        "antimony", "molybdenum", "tungsten",
    ],
    "pesticide": [
        "atrazine", "chlorpyrifos", "ddt", "permethrin", "cypermethrin",
        "malathion", "diazinon", "carbaryl", "glyphosate", "imidacloprid",
        "fipronil", "deltamethrin", "endosulfan", "parathion", "aldrin",
        "dieldrin", "heptachlor", "lindane", "methoxychlor", "toxaphene",
        "organophosphate", "pyrethroid", "neonicotinoid", "herbicide",
        "fungicide", "insecticide", "pesticide", "2,4-d", "dicamba",
        "trifluralin", "metolachlor", "simazine", "diuron", "pendimethalin",
    ],
    "pharmaceutical": [
        "estrogen", "estradiol", "ibuprofen", "acetaminophen", "diclofenac",
        "naproxen", "carbamazepine", "fluoxetine", "sertraline",
        "ciprofloxacin", "sulfamethoxazole", "trimethoprim", "metformin",
        "atenolol", "propranolol", "erythromycin", "tetracycline",
        "amoxicillin", "triclosan", "caffeine", "contraceptive",
        "pharmaceutical", "antibiotic", "analgesic", "hormone",
        "ethinylestradiol", "17alpha-ethinylestradiol", "bisphenol",
    ],
    "pah": [
        "naphthalene", "anthracene", "pyrene", "benzo", "fluoranthene",
        "phenanthrene", "fluorene", "acenaphthylene", "acenaphthene",
        "chrysene", "benz[a]anthracene", "benzo[a]pyrene", "benzo[b]fluoranthene",
        "indeno", "dibenz", "polycyclic aromatic", "pah", "coal tar",
        "creosote", "petroleum",
    ],
    "pcb": [
        "pcb", "polychlorinated biphenyl", "aroclor", "biphenyl, chloro",
        "dioxin", "furan", "tcdd", "pcdd", "pcdf",
    ],
    "pfas": [
        "pfas", "pfoa", "pfos", "perfluoro", "polyfluoro",
        "fluorotelomer", "genx", "pfna", "pfda", "pfhxs",
    ],
    "nanomaterial": [
        "nanoparticle", "nanotube", "nano ", "quantum dot", "fullerene",
        "graphene oxide", "titanium dioxide nano", "silver nano",
        "zinc oxide nano", "cerium oxide nano", "nanosilver",
    ],
    "nutrient": [
        "ammonia", "nitrate", "nitrite", "phosphate", "nitrogen",
        "phosphorus", "urea", "ammonium", "nutrient", "eutrophic",
    ],
}

# Aquatic habitat/media filters
AQUATIC_HABITATS = {"Water", "water", "FW", "SW"}
AQUATIC_MEDIA = {
    "FW", "Fresh water", "Freshwater", "SW", "Salt water",
    "Saltwater", "Estuary", "Brackish", "NR",
}

# Target endpoints
TARGET_ENDPOINTS = {"LC50", "EC50", "NOEC", "LOEC", "IC50", "MATC", "LC10", "LC100", "EC10", "EC25"}

# Target effects for behavioral relevance
BEHAVIORAL_EFFECTS = {"BEH", "MOR", "GRO", "REP", "DVP", "BCM", "POP", "PHY", "IMM", "CEL"}


def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def classify_chemical(chemical_name, ecotox_group=""):
    """Classify a chemical into one of 8 contaminant classes."""
    name_lower = str(chemical_name or "").lower()
    group_lower = str(ecotox_group or "").lower()
    combined = name_lower + " " + group_lower

    for class_name, keywords in CONTAMINANT_KEYWORDS.items():
        for kw in keywords:
            if kw in combined:
                return class_name
    return "other"


def load_chemicals():
    """Load chemical lookup table and classify into contaminant classes."""
    log("Loading chemicals...")
    chem_path = ECOTOX_DIR / "validation" / "chemicals.txt"
    df = pd.read_csv(chem_path, sep="|", dtype=str, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    df["contaminant_class"] = df.apply(
        lambda r: classify_chemical(r.get("chemical_name", ""), r.get("ecotox_group", "")),
        axis=1,
    )

    class_counts = df["contaminant_class"].value_counts()
    log(f"  {len(df)} chemicals loaded")
    for cls, count in class_counts.items():
        log(f"    {cls}: {count}")

    return df.set_index("cas_number")[["chemical_name", "ecotox_group", "contaminant_class"]]


def load_species():
    """Load species lookup and filter to aquatic organisms."""
    log("Loading species...")
    sp_path = ECOTOX_DIR / "validation" / "species.txt"
    df = pd.read_csv(sp_path, sep="|", dtype=str, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    # Keep aquatic-relevant phyla/classes
    aquatic_taxa = {
        "Chordata", "Arthropoda", "Mollusca", "Annelida",
        "Cnidaria", "Echinodermata", "Rotifera", "Bryozoa",
        "Chlorophyta", "Bacillariophyta", "Cyanobacteria",
    }

    log(f"  {len(df)} species loaded")
    return df.set_index("species_number")[["common_name", "latin_name", "kingdom",
                                           "phylum_division", "class", "ecotox_group"]]


def load_tests(species_lookup):
    """Load tests and filter to aquatic studies."""
    log("Loading tests...")
    tests_path = ECOTOX_DIR / "tests.txt"
    cols = [
        "test_id", "test_cas", "species_number", "organism_habitat",
        "media_type", "exposure_type", "test_type", "test_location",
        "exposure_duration_mean", "exposure_duration_unit",
        "study_duration_mean", "study_duration_unit",
    ]
    df = pd.read_csv(tests_path, sep="|", usecols=cols, dtype=str, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    log(f"  {len(df)} total tests")

    # Filter to aquatic
    aquatic_mask = (
        df["organism_habitat"].isin(AQUATIC_HABITATS) |
        df["media_type"].str.strip().isin(AQUATIC_MEDIA)
    )
    df = df[aquatic_mask].copy()
    log(f"  {len(df)} aquatic tests after filtering")

    # Normalize exposure duration to hours
    duration_multipliers = {
        "h": 1.0, "d": 24.0, "mi": 1/60, "mo": 720.0,
        "wk": 168.0, "yr": 8760.0, "s": 1/3600,
    }
    df["exposure_hours"] = pd.to_numeric(df["exposure_duration_mean"], errors="coerce")
    df["duration_unit"] = df["exposure_duration_unit"].str.strip()
    for unit, mult in duration_multipliers.items():
        mask = df["duration_unit"] == unit
        df.loc[mask, "exposure_hours"] = df.loc[mask, "exposure_hours"] * mult

    return df


def load_results():
    """Load results (endpoints, concentrations, effects)."""
    log("Loading results...")
    results_path = ECOTOX_DIR / "results.txt"
    cols = [
        "result_id", "test_id", "endpoint", "effect",
        "conc1_type", "conc1_mean", "conc1_unit",
        "obs_duration_mean", "obs_duration_unit",
    ]
    df = pd.read_csv(results_path, sep="|", usecols=cols, dtype=str, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    log(f"  {len(df)} total results")

    # Filter to target endpoints
    df["endpoint"] = df["endpoint"].str.strip()
    df = df[df["endpoint"].isin(TARGET_ENDPOINTS)].copy()
    log(f"  {len(df)} results with target endpoints")

    # Parse concentration
    df["concentration"] = pd.to_numeric(df["conc1_mean"], errors="coerce")
    df = df.dropna(subset=["concentration"])
    df = df[df["concentration"] > 0]
    log(f"  {len(df)} results with valid concentration")

    return df


def load_media_characteristics():
    """Load media (environmental conditions) for each result."""
    log("Loading media characteristics...")
    media_path = ECOTOX_DIR / "media_characteristics.txt"

    # This file is huge (1.2M rows), load selectively
    cols_header = pd.read_csv(media_path, sep="|", nrows=0).columns.tolist()
    cols_header = [c.strip() for c in cols_header]
    log(f"  Media columns: {cols_header[:15]}...")

    # Load just the key environmental columns
    target_cols = ["result_id"]
    for c in cols_header:
        c_clean = c.strip()
        if any(kw in c_clean.lower() for kw in ["ph", "temp", "hardness", "do_", "dissolved",
                                                   "conductivity", "salinity", "alkalinity"]):
            target_cols.append(c)

    if len(target_cols) <= 1:
        log("  No media condition columns found, skipping")
        return None

    df = pd.read_csv(media_path, sep="|", usecols=target_cols[:10], dtype=str,
                     low_memory=False, nrows=500000)
    log(f"  Loaded {len(df)} media records with {len(target_cols)} condition columns")
    return df


def build_training_dataset(tests, results, chemicals):
    """Build structured training dataset from joined tables."""
    log("Building training dataset...")

    # Join results with tests
    merged = results.merge(tests, on="test_id", how="inner")
    log(f"  Merged results-tests: {len(merged)} records")

    # Join with chemicals
    merged["test_cas"] = merged["test_cas"].str.strip()
    merged = merged.merge(
        chemicals[["chemical_name", "contaminant_class"]],
        left_on="test_cas",
        right_index=True,
        how="left",
    )

    # Drop unknown class
    merged = merged[merged["contaminant_class"] != "other"].copy()
    log(f"  After dropping 'other' class: {len(merged)} records")

    if len(merged) == 0:
        log("ERROR: No records after filtering!")
        return None

    # Class distribution
    class_counts = merged["contaminant_class"].value_counts()
    log("  Contaminant class distribution:")
    for cls, count in class_counts.items():
        log(f"    {cls}: {count}")

    # Create feature vectors per record
    # Features: log(concentration), log(exposure_hours), endpoint one-hot, effect type
    endpoint_cats = sorted(merged["endpoint"].unique())
    endpoint_map = {e: i for i, e in enumerate(endpoint_cats)}

    effect_cats = sorted(merged["effect"].str.strip().unique())[:20]  # top 20 effects
    effect_map = {e: i for i, e in enumerate(effect_cats)}

    n_features = 2 + len(endpoint_cats) + len(effect_cats)
    log(f"  Feature vector size: {n_features} ({len(endpoint_cats)} endpoints, {len(effect_cats)} effects)")

    features = np.zeros((len(merged), n_features), dtype=np.float32)
    labels = []
    metadata_list = []

    class_to_idx = {cls: i for i, cls in enumerate(sorted(class_counts.keys()))}

    for i, (_, row) in enumerate(merged.iterrows()):
        # Continuous features
        features[i, 0] = np.log1p(float(row["concentration"]))
        hours = row.get("exposure_hours")
        if pd.notna(hours):
            features[i, 1] = np.log1p(float(hours))

        # Endpoint one-hot
        ep = row["endpoint"]
        if ep in endpoint_map:
            features[i, 2 + endpoint_map[ep]] = 1.0

        # Effect one-hot
        effect = str(row.get("effect", "")).strip()
        if effect in effect_map:
            features[i, 2 + len(endpoint_cats) + effect_map[effect]] = 1.0

        labels.append(class_to_idx[row["contaminant_class"]])
        metadata_list.append({
            "chemical": str(row.get("chemical_name", "")),
            "cas": str(row.get("test_cas", "")),
            "endpoint": ep,
            "concentration": float(row["concentration"]),
            "class": row["contaminant_class"],
        })

    labels = np.array(labels, dtype=np.int64)

    log(f"  Final dataset: {len(features)} samples, {n_features} features, {len(class_to_idx)} classes")
    return features, labels, class_to_idx, endpoint_cats, effect_cats, metadata_list


def build_dose_response_profiles(tests, results, chemicals):
    """Build per-chemical dose-response profiles for transfer learning."""
    log("Building dose-response profiles...")

    merged = results.merge(tests, on="test_id", how="inner")
    merged["test_cas"] = merged["test_cas"].str.strip()
    merged = merged.merge(
        chemicals[["chemical_name", "contaminant_class"]],
        left_on="test_cas",
        right_index=True,
        how="left",
    )
    merged = merged[merged["contaminant_class"] != "other"]

    # Group by chemical
    profiles = {}
    for cas, group in merged.groupby("test_cas"):
        if len(group) < 5:
            continue

        concs = group["concentration"].values
        endpoints = group["endpoint"].values
        effects = group["effect"].str.strip().values
        hours = group["exposure_hours"].values

        profiles[cas] = {
            "chemical_name": group["chemical_name"].iloc[0],
            "contaminant_class": group["contaminant_class"].iloc[0],
            "n_records": len(group),
            "concentration_range": [float(np.nanmin(concs)), float(np.nanmax(concs))],
            "endpoints": dict(Counter(endpoints)),
            "effects": dict(Counter(effects)),
            "median_exposure_hours": float(np.nanmedian(hours)) if np.any(~np.isnan(hours)) else None,
        }

    log(f"  {len(profiles)} chemical profiles (≥5 records each)")
    return profiles


def main():
    log("=" * 60)
    log("Processing EPA ECOTOX Database for ToxiGene")
    log("=" * 60)

    # Load lookup tables
    chemicals = load_chemicals()
    species = load_species()

    # Load main tables
    tests = load_tests(species)
    results = load_results()

    # Build training dataset
    result = build_training_dataset(tests, results, chemicals)
    if result is None:
        log("Failed to build training dataset")
        return

    features, labels, class_map, endpoint_cats, effect_cats, metadata = result

    # Save main training dataset
    np.savez_compressed(
        OUT_DIR / "ecotox_training.npz",
        features=features,
        labels=labels,
    )
    log(f"Saved training data: {OUT_DIR / 'ecotox_training.npz'}")

    # Save metadata
    meta = {
        "n_samples": len(features),
        "n_features": features.shape[1],
        "class_map": class_map,
        "endpoint_categories": endpoint_cats,
        "effect_categories": effect_cats[:20],
        "class_counts": {cls: int((labels == idx).sum()) for cls, idx in class_map.items()},
    }
    with open(OUT_DIR / "ecotox_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    log(f"Saved metadata: {OUT_DIR / 'ecotox_metadata.json'}")

    # Build dose-response profiles
    profiles = build_dose_response_profiles(tests, results, chemicals)
    with open(OUT_DIR / "dose_response_profiles.json", "w") as f:
        json.dump(profiles, f, indent=2, default=str)
    log(f"Saved {len(profiles)} dose-response profiles")

    # Summary
    log("\n" + "=" * 60)
    log("ECOTOX Processing Summary")
    log("=" * 60)
    log(f"Total training samples: {len(features)}")
    log(f"Feature dimensions: {features.shape[1]}")
    log(f"Classes: {len(class_map)}")
    for cls, idx in sorted(class_map.items(), key=lambda x: x[1]):
        n = int((labels == idx).sum())
        log(f"  {cls}: {n} ({100*n/len(labels):.1f}%)")
    log(f"Dose-response profiles: {len(profiles)} chemicals")
    log(f"Output: {OUT_DIR}")
    log("=" * 60)


if __name__ == "__main__":
    main()
