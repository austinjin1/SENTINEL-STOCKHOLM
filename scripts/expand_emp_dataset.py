#!/usr/bin/env python3
"""Expand EMP 16S dataset by extracting additional samples from full release1 biom.

The current 20,288 processed samples used these empo_3 categories:
  Water (non-saline)      -> freshwater_natural / freshwater_impacted
  Water (saline)          -> saline_water
  Sediment (non-saline)   -> freshwater_sediment
  Sediment (saline)       -> saline_sediment
  Soil (non-saline)       -> soil_runoff
  Animal distal gut       -> animal_fecal
  Animal secretion        -> animal_fecal
  Animal proximal gut     -> animal_fecal
  Plant surface           -> plant_associated
  Plant rhizosphere       -> plant_associated
  Plant corpus            -> plant_associated

Remaining unprocessed samples in the full biom that CAN be mapped:
  Surface (non-saline) 1308 -> soil_runoff  (urban/environmental surfaces)
  Aerosol (non-saline)   88 -> soil_runoff  (soil-derived aerosols)
  Animal corpus          580 -> animal_fecal (body tissue samples)
  Animal surface        3632 -> animal_fecal (skin/surface microbiome)

Total new samples: ~5608 (pending rarefaction filter)
Total after expansion: ~25,896

NOTE: EMP release1 biom uses rarefied data at 5,000 sequences per sample.
      Top-5000 ASVs by prevalence are selected (same as original processing).
      This script uses the SAME feature space as the existing processed files.

REAL DATA ONLY — no synthetic augmentation.

MIT License — Bryan Cheng, 2026
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from collections import Counter

import h5py
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_RAW = PROJECT_ROOT / "data" / "raw" / "emp"
DATA_OUT = PROJECT_ROOT / "data" / "processed" / "microbial" / "emp_16s"
DATA_OUT.mkdir(parents=True, exist_ok=True)

BIOM_PATH    = DATA_RAW / "emp_deblur_90bp.release1.biom"
MAPPING_PATH = DATA_RAW / "emp_qiime_mapping_release1.tsv"

# Label mapping — must match train_microbiomenet_emp.py
SOURCE_NAMES = [
    "freshwater_natural",    # 0
    "freshwater_impacted",   # 1
    "saline_water",          # 2
    "freshwater_sediment",   # 3
    "saline_sediment",       # 4
    "soil_runoff",           # 5
    "animal_fecal",          # 6
    "plant_associated",      # 7
]

# Expanded empo_3 -> class label mapping (adds Surface, Aerosol, Animal surface/corpus)
EMPO3_TO_LABEL: dict[str, int] = {
    # Already processed
    "Water (non-saline)":    -1,   # handled by freshwater_natural/impacted logic
    "Water (saline)":         2,
    "Sediment (non-saline)":  3,
    "Sediment (saline)":      4,
    "Soil (non-saline)":      5,
    "Animal distal gut":      6,
    "Animal secretion":       6,
    "Animal proximal gut":    6,
    "Plant surface":          7,
    "Plant rhizosphere":      7,
    "Plant corpus":           7,
    # NEW: expanded categories
    "Surface (non-saline)":   5,   # soil_runoff (urban/environmental surface)
    "Aerosol (non-saline)":   5,   # soil_runoff (soil-derived aerosol)
    "Animal corpus":          6,   # animal_fecal (body tissue microbiome)
    "Animal surface":         6,   # animal_fecal (skin microbiome)
}

# Keywords used to split Water (non-saline) into natural vs impacted
IMPACTED_KEYWORDS = [
    "wastewater", "sewage", "effluent", "pollut", "contamina",
    "river", "stream", "canal", "ditch",
]
NATURAL_KEYWORDS = [
    "pristine", "alpine", "lake", "reservoir", "drinking",
    "groundwater", "spring",
]


def classify_nonsaline_water(row: pd.Series) -> int:
    """Distinguish freshwater_natural (0) from freshwater_impacted (1)."""
    text = " ".join([
        str(row.get("env_biome", "")),
        str(row.get("env_feature", "")),
        str(row.get("env_material", "")),
        str(row.get("title", "")),
        str(row.get("description", "")),
    ]).lower()

    is_impacted = any(kw in text for kw in IMPACTED_KEYWORDS)
    is_natural  = any(kw in text for kw in NATURAL_KEYWORDS)

    if is_impacted and not is_natural:
        return 1   # freshwater_impacted
    elif is_natural and not is_impacted:
        return 0   # freshwater_natural
    else:
        # Weighted default toward impacted (most rivers/streams are impacted)
        return 1 if hash(str(row.name)) % 3 != 0 else 0


def load_biom_as_csr(biom_path: Path):
    """Load HDF5 BIOM file as sparse CSR matrix (samples x OTUs).

    BIOM v2 HDF5 format stores TWO sparse matrices:
      - observation/matrix: OTU-indexed CSR (n_otus x n_samples)
      - sample/matrix:      Sample-indexed CSR (n_samples x n_otus)

    We use sample/matrix (already in sample x OTU orientation).

    Returns:
        data_csr: scipy.sparse.csr_matrix [n_samples, n_otus]
        sample_ids: list of sample IDs
        otu_ids: list of OTU IDs
    """
    import scipy.sparse as sp

    with h5py.File(biom_path, "r") as f:
        otu_ids    = [x.decode() if isinstance(x, bytes) else str(x)
                      for x in f["observation"]["ids"][:]]
        sample_ids = [x.decode() if isinstance(x, bytes) else str(x)
                      for x in f["sample"]["ids"][:]]

        n_otus    = len(otu_ids)
        n_samples = len(sample_ids)
        print(f"  BIOM dimensions: {n_otus} OTUs x {n_samples} samples")

        # sample/matrix is in sample x OTU orientation (CSR)
        data    = f["sample"]["matrix"]["data"][:]
        indices = f["sample"]["matrix"]["indices"][:]   # OTU indices
        indptr  = f["sample"]["matrix"]["indptr"][:]    # n_samples+1 entries

        sample_x_otu = sp.csr_matrix(
            (data, indices, indptr), shape=(n_samples, n_otus)
        ).astype(np.float32)

    return sample_x_otu, sample_ids, otu_ids


def get_top5000_otu_indices(biom_otu_ids: list[str]) -> np.ndarray:
    """Get indices into the full biom OTU list that match the saved selected_otu_ids.

    Loads the OTU IDs saved during the original processing run
    (data/processed/microbial/emp_16s/selected_otu_ids.npy) and maps them
    to indices in the full biom's OTU list.

    Returns:
        np.ndarray of shape (5000,) with indices into biom_otu_ids
    """
    otu_id_file = DATA_OUT / "selected_otu_ids.npy"
    if not otu_id_file.exists():
        raise FileNotFoundError(
            f"selected_otu_ids.npy not found at {otu_id_file}. "
            "Run download_emp_microbiome.py first."
        )

    selected = np.load(otu_id_file, allow_pickle=True)
    selected_set = {s: i for i, s in enumerate(selected)}

    biom_id_to_idx = {sid: i for i, sid in enumerate(biom_otu_ids)}

    indices = []
    for sel_id in selected:
        if sel_id in biom_id_to_idx:
            indices.append(biom_id_to_idx[sel_id])
        else:
            indices.append(-1)   # OTU not in full biom (will be zeroed out)

    indices = np.array(indices, dtype=np.int64)
    n_found = (indices >= 0).sum()
    print(f"  Matched {n_found}/5000 OTUs from original selection to full biom")
    return indices


def main() -> None:
    t0 = time.time()
    print("=" * 60)
    print("EMP 16S Dataset Expansion")
    print("=" * 60)

    assert BIOM_PATH.exists(), f"Biom file not found: {BIOM_PATH}"
    assert MAPPING_PATH.exists(), f"Mapping file not found: {MAPPING_PATH}"

    # ── Load mapping metadata ─────────────────────────────────────────
    print("\nLoading mapping file...")
    df = pd.read_csv(MAPPING_PATH, sep="\t", low_memory=False)
    df = df.set_index("#SampleID")
    print(f"  Mapping entries: {len(df)}")

    # ── Find already-processed samples ────────────────────────────────
    existing_files = sorted(DATA_OUT.glob("*.npz"))
    processed_ids  = set()
    for f in existing_files:
        try:
            d = np.load(f, allow_pickle=True)
            processed_ids.add(str(d["site_id"]))
        except Exception:
            continue
    print(f"  Already processed: {len(processed_ids)}")

    # ── Load BIOM matrix ──────────────────────────────────────────────
    print("\nLoading full EMP release1 BIOM matrix...")
    sample_x_otu, biom_sample_ids, otu_ids = load_biom_as_csr(BIOM_PATH)

    # ── Get top-5000 OTU indices (aligned to original processing) ────
    top5000_idx = get_top5000_otu_indices(otu_ids)
    print(f"  OTU index mapping complete")

    # ── Identify samples to process ───────────────────────────────────
    to_process = []
    for i, sid in enumerate(biom_sample_ids):
        if sid in processed_ids:
            continue
        if sid not in df.index:
            continue
        row = df.loc[sid]
        empo3 = str(row.get("empo_3", ""))
        if empo3 not in EMPO3_TO_LABEL:
            continue
        label = EMPO3_TO_LABEL[empo3]
        if label == -1:
            # Water (non-saline) — classify as natural or impacted
            label = classify_nonsaline_water(row)
        to_process.append((i, sid, label, empo3))

    print(f"\n  Samples to add: {len(to_process)}")
    label_dist = Counter(label for _, _, label, _ in to_process)
    for lid, cnt in sorted(label_dist.items()):
        print(f"    {SOURCE_NAMES[lid]:>25}: {cnt:,}")

    if not to_process:
        print("No new samples to process. Dataset already complete.")
        return

    # ── Extract features and save ──────────────────────────────────────
    start_idx = len(existing_files)
    saved = 0
    failed = 0

    print(f"\nExtracting and saving {len(to_process)} new samples...")
    for batch_start in range(0, len(to_process), 500):
        batch = to_process[batch_start: batch_start + 500]
        if batch_start % 2000 == 0:
            print(f"  Progress: {batch_start}/{len(to_process)} ({saved} saved, {failed} failed)")

        for matrix_idx, sid, label, empo3 in batch:
            try:
                # Extract top-5000 features for this sample using pre-computed indices
                # CSR row access: sample_x_otu[i] gives the i-th sample's OTU vector
                row_data = np.asarray(sample_x_otu[matrix_idx, :].todense()).flatten()
                # Map to the 5000-feature space using saved OTU index alignment
                abundances = np.zeros(5000, dtype=np.float32)
                valid_mask = top5000_idx >= 0
                abundances[valid_mask] = row_data[top5000_idx[valid_mask]]

                total = abundances.sum()
                if total < 1e-8:
                    failed += 1
                    continue

                # Normalize to relative abundance (simplex)
                abundances = abundances / total

                # Save as npz with same format as existing files
                out_path = DATA_OUT / f"emp16s_{start_idx + saved:05d}.npz"
                np.savez_compressed(
                    out_path,
                    abundances=abundances,
                    source_label=np.array(label, dtype=np.int64),
                    source_name=np.array(SOURCE_NAMES[label]),
                    site_id=np.array(sid),
                )
                saved += 1
            except Exception as e:
                failed += 1
                continue

    elapsed = time.time() - t0

    # ── Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EXPANSION COMPLETE")
    print("=" * 60)
    total_now = len(existing_files) + saved
    print(f"  New samples saved: {saved}")
    print(f"  Failed/skipped:    {failed}")
    print(f"  Previous total:    {len(existing_files)}")
    print(f"  New total:         {total_now}")
    print(f"  Elapsed:           {elapsed/60:.1f}m")

    # Save expansion log
    log = {
        "expansion_date": "2026-04-13",
        "previous_n": len(existing_files),
        "new_samples_added": saved,
        "failed_skipped": failed,
        "new_total": total_now,
        "new_empo3_categories": ["Surface (non-saline)", "Aerosol (non-saline)",
                                  "Animal corpus", "Animal surface"],
        "label_dist_new": {SOURCE_NAMES[k]: v for k, v in label_dist.items()},
        "elapsed_seconds": elapsed,
    }
    log_path = PROJECT_ROOT / "data" / "processed" / "microbial" / "expansion_log.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nExpansion log saved to {log_path}")


if __name__ == "__main__":
    main()
