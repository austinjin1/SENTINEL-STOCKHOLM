#!/usr/bin/env python3
"""Consolidate all real data sources for MicroBiomeNet expanded training.

Sources:
1. data/processed/microbial/emp_16s/    — 20,288 EMP 16S samples (habitat labels)
2. data/processed/microbial/nars/       — 175,846 EPA NARS samples (pollution labels)
3. data/processed/microbial/real/       — 2,111 NRSA WQ samples (subset, 25 features)
4. data/raw/microbial/                  — raw CSVs (already incorporated into NARS)

Label unification strategy:
- The best existing model uses EMP 16S habitat labels (8 classes, F1=0.913)
- NARS uses pollution source labels (8 different classes, F1=0.207)
- Two source datasets use incompatible label schemas; we keep them SEPARATE
  and create a unified 9-class schema that covers both:
    0: freshwater_natural       (EMP only)
    1: freshwater_impacted      (EMP only)
    2: saline_water             (EMP only)
    3: freshwater_sediment      (EMP, maps to NARS sediment)
    4: saline_sediment          (EMP only)
    5: soil_runoff              (EMP only)
    6: animal_fecal             (EMP, maps to NARS sewage)
    7: plant_associated         (EMP only)
    8: nutrient_pollution       (NARS: nutrient)
    9: heavy_metals             (NARS)
   10: thermal_pollution        (NARS)
   11: pharmaceutical           (NARS)
   12: oil_petrochemical        (NARS)
   13: acid_mine_drainage       (NARS)

Actually — to maximize compatibility with the trained model architecture (num_sources=8),
we use a DUAL approach:
  - Primary task: use EMP labels (8 classes) as ground truth
  - NARS samples: map NARS pollution labels to nearest EMP habitat labels via lookup
  - real/nrsa: map 5 classes to EMP schema

NARS->EMP mapping:
  nutrient         -> freshwater_impacted (1)   [nutrient runoff -> impacted freshwater]
  heavy_metals     -> freshwater_impacted (1)   [heavy metal contamination]
  sewage           -> animal_fecal (6)           [sewage/fecal indicator]
  oil_petrochemical -> freshwater_impacted (1)  [industrial pollution]
  thermal          -> freshwater_impacted (1)   [thermal pollution]
  sediment         -> freshwater_sediment (3)   [sediment/erosion]
  pharmaceutical   -> freshwater_impacted (1)   [pharma contamination]
  acid_mine        -> freshwater_impacted (1)   [acid mine drainage -> impacted]

real/nrsa labels: acid_mine->1, heavy_metals->1, nutrient->1, reference->0, sewage->6

Feature alignment:
- EMP: 5000 OTU relative abundances [0,1] -> CLR applied at training time
- NARS: 5000 features (610 real CLR + 4390 zeros) — already CLR-like
  We store them as-is in 'abundances'; training script re-applies CLR which is fine
  since the values are already approximately CLR-scaled
- real/nrsa: 25 features -> pad to 5000 (non-zero in first 25 positions)

MIT License — Bryan Cheng, 2026
"""

import json
import pathlib
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, "/home/bcheng/SENTINEL")

DATA_EMP = pathlib.Path("data/processed/microbial/emp_16s")
DATA_NARS = pathlib.Path("data/processed/microbial/nars")
DATA_REAL = pathlib.Path("data/processed/microbial/real")
DATA_RAW_MIC = pathlib.Path("data/raw/microbial")
OUT_DIR = pathlib.Path("data/processed/microbial/consolidated_real")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Canonical 8-class EMP label schema (the better-performing schema)
CLASS_NAMES = [
    "freshwater_natural",    # 0
    "freshwater_impacted",   # 1
    "saline_water",          # 2
    "freshwater_sediment",   # 3
    "saline_sediment",       # 4
    "soil_runoff",           # 5
    "animal_fecal",          # 6
    "plant_associated",      # 7
]
NUM_CLASSES = len(CLASS_NAMES)

# NARS pollution label -> EMP habitat label mapping
NARS_TO_EMP = {
    0: 1,  # nutrient -> freshwater_impacted
    1: 1,  # heavy_metals -> freshwater_impacted
    2: 6,  # sewage -> animal_fecal
    3: 1,  # oil_petrochemical -> freshwater_impacted
    4: 1,  # thermal -> freshwater_impacted
    5: 3,  # sediment -> freshwater_sediment
    6: 1,  # pharmaceutical -> freshwater_impacted
    7: 1,  # acid_mine -> freshwater_impacted
}

# NARS name -> EMP label
NARS_NAME_TO_EMP = {
    "nutrient": 1,
    "heavy_metals": 1,
    "sewage": 6,
    "oil_petrochemical": 1,
    "thermal": 1,
    "sediment": 3,
    "pharmaceutical": 1,
    "acid_mine": 1,
}

# real/nrsa label names
NRSA_NAME_TO_EMP = {
    "acid_mine": 1,      # freshwater_impacted
    "heavy_metals": 1,   # freshwater_impacted
    "nutrient": 1,       # freshwater_impacted
    "reference": 0,      # freshwater_natural
    "sewage": 6,         # animal_fecal
}

FEATURES = 5000


def load_emp_16s():
    """Load EMP 16S data. Returns (X, y, sources) arrays."""
    print(f"\n[1/4] Loading EMP 16S from {DATA_EMP}...")
    files = sorted(DATA_EMP.glob("emp16s_*.npz"))
    X, y, sources = [], [], []
    skipped = 0
    for f in files:
        try:
            d = np.load(f, allow_pickle=True)
            abund = d["abundances"].astype(np.float32)
            if abund.sum() < 1e-8:
                skipped += 1
                continue
            label = int(d["source_label"])
            X.append(abund)
            y.append(label)
            sources.append("emp_16s")
        except Exception as e:
            skipped += 1
    X = np.stack(X)
    y = np.array(y, dtype=np.int64)
    print(f"  Loaded {len(X)} samples ({skipped} skipped), shape={X.shape}")
    print(f"  Label dist: {dict(sorted(Counter(y.tolist()).items()))}")
    return X, y, sources


def load_nars(max_samples=60000):
    """Load NARS data and remap labels to EMP schema.

    Caps at max_samples to avoid memory issues (175K files).
    NARS abundances are CLR-like measurements; they will be treated
    as pre-processed abundance vectors.
    """
    print(f"\n[2/4] Loading NARS from {DATA_NARS} (max={max_samples})...")
    files = sorted([f for f in DATA_NARS.glob("nars_*.npz") if "matrix" not in f.name])

    # Sample evenly across the full range to avoid label bias
    if len(files) > max_samples:
        step = len(files) / max_samples
        indices = [int(i * step) for i in range(max_samples)]
        files = [files[i] for i in indices]

    X, y, sources = [], [], []
    skipped = 0
    remap_counts = Counter()

    for f in files:
        try:
            d = np.load(f, allow_pickle=True)
            abund = d["abundances"].astype(np.float32)
            nars_label = int(d["source_label"])

            # Remap to EMP label schema
            emp_label = NARS_TO_EMP.get(nars_label, 1)
            remap_counts[emp_label] += 1

            # NARS features: CLR-like values in first 610 positions, zeros after
            # Normalize to [0, 1] range so CLR transform at train time works correctly
            # Shift to non-negative and normalize (approximate relative abundance)
            abund_shifted = abund - abund.min()  # all >= 0
            total = abund_shifted.sum()
            if total < 1e-10:
                skipped += 1
                continue
            abund_normalized = abund_shifted / total  # sums to 1, like relative abundance

            X.append(abund_normalized)
            y.append(emp_label)
            sources.append("nars")
        except Exception:
            skipped += 1

    X = np.stack(X)
    y = np.array(y, dtype=np.int64)
    print(f"  Loaded {len(X)} samples ({skipped} skipped), shape={X.shape}")
    print(f"  Label dist after remapping: {dict(sorted(Counter(y.tolist()).items()))}")
    print(f"  Remap counts: {dict(sorted(remap_counts.items()))}")
    return X, y, sources


def load_real_nrsa():
    """Load real/nrsa_wq_training.npz — 2111 samples, 25 features -> pad to 5000."""
    f = DATA_REAL / "nrsa_wq_training.npz"
    if not f.exists():
        print(f"\n[3/4] real/nrsa_wq_training.npz not found, skipping.")
        return np.empty((0, FEATURES), dtype=np.float32), np.empty(0, dtype=np.int64), []

    print(f"\n[3/4] Loading real NRSA WQ from {f}...")
    d = np.load(f, allow_pickle=True)
    features_25 = d["features"].astype(np.float32)  # (2111, 25)
    label_ids = d["labels"].astype(np.int64)          # (2111,)
    label_names = d["label_names"]                     # ['acid_mine', 'heavy_metals', ...]

    # Map label IDs -> names -> EMP labels
    y_emp = np.array([
        NRSA_NAME_TO_EMP.get(str(label_names[lid]), 1)
        for lid in label_ids
    ], dtype=np.int64)

    # Pad features from 25 to 5000
    # Normalize features to [0,1] first (they are CLR-scaled -5 to 5)
    # Shift and normalize to approximate relative abundance
    feat_shifted = features_25 - features_25.min(axis=1, keepdims=True)
    row_sums = feat_shifted.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < 1e-10, 1.0, row_sums)
    feat_norm = feat_shifted / row_sums

    X_padded = np.zeros((len(feat_norm), FEATURES), dtype=np.float32)
    X_padded[:, :25] = feat_norm

    print(f"  Loaded {len(X_padded)} samples, padded from 25 -> {FEATURES} features")
    print(f"  Label dist: {dict(sorted(Counter(y_emp.tolist()).items()))}")
    return X_padded, y_emp, ["real_nrsa"] * len(X_padded)


def load_raw_microbial():
    """Check raw/microbial/ for usable CSV data."""
    print(f"\n[4/4] Checking raw/microbial CSVs in {DATA_RAW_MIC}...")
    csv_files = list(DATA_RAW_MIC.glob("*.csv"))
    print(f"  Found CSVs: {[f.name for f in csv_files]}")
    print("  These are source CSVs already incorporated into NARS processed data.")
    print("  Skipping to avoid double-counting.")
    return np.empty((0, FEATURES), dtype=np.float32), np.empty(0, dtype=np.int64), []


def save_consolidated(X, y, sources):
    """Save consolidated dataset in sharded format."""
    print(f"\nSaving consolidated dataset to {OUT_DIR}...")
    n = len(X)
    sources_arr = np.array(sources)

    # Save in chunks of 10,000 for memory efficiency
    chunk_size = 10000
    n_chunks = (n + chunk_size - 1) // chunk_size
    shard_info = []

    for i in range(n_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, n)
        chunk_X = X[start:end]
        chunk_y = y[start:end]
        chunk_src = sources_arr[start:end]

        fname = OUT_DIR / f"consolidated_{i:03d}.npz"
        np.savez_compressed(fname, X=chunk_X, y=chunk_y, sources=chunk_src)
        shard_info.append({
            "file": fname.name,
            "n_samples": int(end - start),
            "start": int(start),
            "end": int(end),
        })

    # Save single large npz for fast loading during training
    print("  Saving single consolidated.npz for training...")
    np.savez_compressed(
        OUT_DIR / "consolidated.npz",
        X=X, y=y, sources=sources_arr
    )

    # Compute stats
    label_counts = Counter(y.tolist())
    source_counts = Counter(sources)

    meta = {
        "n_total": int(n),
        "n_features": FEATURES,
        "n_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "label_counts": {CLASS_NAMES[k]: int(v) for k, v in sorted(label_counts.items())},
        "source_counts": dict(sorted(source_counts.items())),
        "shards": shard_info,
        "label_schema": "EMP_habitat_8class",
        "nars_to_emp_mapping": {v: k for k, v in NARS_TO_EMP.items()},
    }
    with open(OUT_DIR / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nConsolidation complete:")
    print(f"  Total samples: {n:,}")
    print(f"  Features: {FEATURES}")
    print(f"  Classes: {NUM_CLASSES}")
    print(f"\nSources:")
    for src, cnt in sorted(source_counts.items()):
        print(f"  {src:>20}: {cnt:>8,} samples")
    print(f"\nLabel distribution (EMP schema):")
    for k, v in sorted(label_counts.items()):
        pct = 100 * v / n
        print(f"  {CLASS_NAMES[k]:>25}: {v:>8,} ({pct:.1f}%)")

    return meta


def main():
    t0 = time.time()
    print("=" * 60)
    print("MicroBiomeNet Data Consolidation")
    print("=" * 60)

    all_X, all_y, all_sources = [], [], []

    # Load all sources
    X1, y1, src1 = load_emp_16s()
    all_X.append(X1); all_y.append(y1); all_sources.extend(src1)

    # NARS: cap at 60,000 for balance (otherwise 175K NARS overwhelms 20K EMP)
    # This gives ~3x expansion: 20K + 60K = 80K total
    X2, y2, src2 = load_nars(max_samples=60000)
    all_X.append(X2); all_y.append(y2); all_sources.extend(src2)

    X3, y3, src3 = load_real_nrsa()
    if len(X3) > 0:
        all_X.append(X3); all_y.append(y3); all_sources.extend(src3)

    X4, y4, src4 = load_raw_microbial()  # returns empty

    # Concatenate
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    # Shuffle
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(X))
    X = X[perm]
    y = y[perm]
    all_sources = [all_sources[i] for i in perm]

    meta = save_consolidated(X, y, all_sources)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Output: {OUT_DIR}")

    return meta


if __name__ == "__main__":
    main()
