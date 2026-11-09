#!/usr/bin/env python3
"""Process the FULL EMP Release-1 90bp BIOM table into MicroBiomeNet npz files.

Memory-safe replacement for download_emp_microbiome.py when operating on the
full release (317,314 OTUs x 26,181 samples). The original script densified the
ENTIRE filtered OTU x sample matrix, which for the full release would require
tens of terabytes of RAM. This version:

  1. Loads the BIOM sparse (CSR) matrix and the QIIME mapping file.
  2. Assigns the SAME source labels as download_emp_microbiome.py
     (EMPO_3 -> 8 pollution-source classes; the MD5%3 coin-flip freshwater
     labels are KEPT here so the geographic loader can exclude them later,
     exactly as it does for the subset — keeping the pipeline identical).
  3. Filters to labeled samples (columns) only.
  4. Computes per-OTU prevalence using sparse ops (no densification).
  5. Selects the top-5000 OTUs by prevalence and densifies ONLY the
     5000 x N_labeled sub-matrix (~0.4 GB), normalizes to relative abundance.
  6. Writes one .npz per sample with the schema MicroBiomeNet expects:
       abundances (5000,), source_label int, source_name str, site_id str.

Output goes to a SEPARATE directory (data/processed/microbial/emp_16s_full)
so the existing 2k-subset processed data (the 0.304 baseline) is untouched.

REAL data only. No synthetic samples, no augmentation, no label leakage.

MIT License — Bryan Cheng, 2026
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.sparse as sp

try:
    from biom import load_table
except ImportError:
    print("biom-format not installed. Run: pip install biom-format")
    sys.exit(1)

RAW_DIR = Path("data/raw/emp")
DEFAULT_BIOM = RAW_DIR / "emp_deblur_90bp.release1.biom"
DEFAULT_MAPPING = RAW_DIR / "emp_qiime_mapping_release1.tsv"
DEFAULT_OUT = Path("data/processed/microbial/emp_16s_full")

N_FEATURES = 5000  # must match MicroBiomeNet input_dim

SOURCE_CLASSES = {
    0: "freshwater_natural",
    1: "freshwater_impacted",
    2: "saline_water",
    3: "freshwater_sediment",
    4: "saline_sediment",
    5: "soil_runoff",
    6: "animal_fecal",
    7: "plant_associated",
}

IMPACTED_KW = [
    "wastewater", "sewage", "urban", "pollut", "eutrophic", "treatment",
    "industrial", "runoff", "effluent", "contaminat", "river", "stream",
    "canal", "ditch",
]
NATURAL_KW = [
    "lake", "pond", "spring", "pristine", "glacier", "mountain",
    "oligotrophic", "alpine", "reservoir", "groundwater",
]


def assign_source_label(row: dict) -> int | None:
    """Identical mapping to download_emp_microbiome.py.assign_source_label."""
    empo3 = row.get("empo_3", "")
    env_biome = row.get("env_biome", "").lower()
    env_feature = row.get("env_feature", "").lower()

    if empo3 == "Water (non-saline)":
        text = f"{env_biome} {env_feature}".lower()
        is_impacted = any(kw in text for kw in IMPACTED_KW)
        is_natural = any(kw in text for kw in NATURAL_KW)
        if is_impacted and not is_natural:
            return 1
        elif is_natural and not is_impacted:
            return 0
        else:
            h = int(hashlib.md5(row.get("#SampleID", "").encode()).hexdigest(), 16)
            return 1 if h % 3 != 0 else 0  # coin-flip (excluded later in geo mode)
    elif empo3 == "Water (saline)":
        return 2
    elif empo3 == "Sediment (non-saline)":
        return 3
    elif empo3 == "Sediment (saline)":
        return 4
    elif empo3 == "Soil (non-saline)":
        return 5
    elif empo3 in ("Animal distal gut", "Animal proximal gut", "Animal secretion"):
        return 6
    elif empo3 in ("Plant surface", "Plant rhizosphere", "Plant corpus"):
        return 7
    return None


def process(biom_path: Path, mapping_path: Path, out_dir: Path,
            max_otus: int = N_FEATURES, min_samples_per_class: int = 20) -> dict:
    print(f"Loading BIOM table from {biom_path} ...")
    table = load_table(str(biom_path))
    print(f"  Loaded: {table.shape[0]:,} OTUs x {table.shape[1]:,} samples")

    print(f"Loading mapping file from {mapping_path} ...")
    with open(mapping_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        mapping = {row["#SampleID"]: row for row in reader}
    print(f"  Loaded metadata for {len(mapping):,} samples")

    sample_ids = [str(s) for s in table.ids(axis="sample")]
    labeled = {}
    for sid in sample_ids:
        row = mapping.get(sid)
        if row is None:
            continue
        lab = assign_source_label(row)
        if lab is not None:
            labeled[sid] = lab
    print(f"  Labeled {len(labeled):,} samples out of {len(sample_ids):,}")

    counts = Counter(labeled.values())
    print("\n  Class distribution (pre-filter):")
    for lid in sorted(counts):
        print(f"    {lid}: {SOURCE_CLASSES[lid]:>22} — {counts[lid]:,}")

    valid_labels = {k for k, v in counts.items() if v >= min_samples_per_class}
    labeled = {k: v for k, v in labeled.items() if v in valid_labels}
    print(f"\n  After filtering (min {min_samples_per_class}/class): {len(labeled):,} samples")
    if not labeled:
        print("ERROR: no samples passed filtering")
        return {"n_samples": 0}

    # Column indices of labeled samples (preserve BIOM column order)
    sid_to_col = {sid: i for i, sid in enumerate(sample_ids)}
    kept_sids = [s for s in sample_ids if s in labeled]  # BIOM order
    kept_cols = np.array([sid_to_col[s] for s in kept_sids], dtype=np.int64)

    # Sparse matrix (OTUs x samples), CSR
    M = table.matrix_data.tocsr()
    print(f"\n  Full sparse matrix: {M.shape}, nnz={M.nnz:,}")

    # Slice labeled columns (still sparse). CSC is efficient for column slicing.
    Msub = M.tocsc()[:, kept_cols].tocsr()  # OTUs x N_labeled
    n_otus, n_samp = Msub.shape
    print(f"  Labeled sub-matrix: {n_otus:,} OTUs x {n_samp:,} samples, nnz={Msub.nnz:,}")

    # Per-OTU prevalence = fraction of labeled samples with count > 0 (sparse).
    binar = (Msub > 0).astype(np.int32)
    prevalence = np.asarray(binar.sum(axis=1)).ravel() / float(n_samp)
    top_idx = np.argsort(-prevalence)[:max_otus]
    print(f"  Selecting top {max_otus} OTUs by prevalence "
          f"(mean prevalence of selected: {prevalence[top_idx].mean():.4f})")

    # Densify ONLY the selected rows: 5000 x N_labeled (~0.4 GB float32).
    dense_sel = np.asarray(Msub[top_idx, :].todense(), dtype=np.float32)  # 5000 x N
    if dense_sel.shape[0] < max_otus:
        pad = np.zeros((max_otus - dense_sel.shape[0], n_samp), dtype=np.float32)
        dense_sel = np.concatenate([dense_sel, pad], axis=0)

    sample_matrix = dense_sel.T  # N x 5000
    row_sums = sample_matrix.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    rel = (sample_matrix / row_sums).astype(np.float32)
    chk = rel.sum(axis=1)
    print(f"  Row sums after normalization: min={chk.min():.6f} max={chk.max():.6f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    # Clear any stale npz in the target dir to avoid mixing runs.
    for old in out_dir.glob("*.npz"):
        old.unlink()

    saved = 0
    for j, sid in enumerate(kept_sids):
        lab = labeled[sid]
        np.savez_compressed(
            out_dir / f"emp16s_{saved:05d}.npz",
            abundances=rel[j],
            source_label=lab,
            source_name=SOURCE_CLASSES[lab],
            site_id=sid,
        )
        saved += 1
        if saved % 2000 == 0:
            print(f"    ...saved {saved:,}")

    print(f"\nSaved {saved:,} samples to {out_dir}")

    obs_ids = table.ids(axis="observation")
    selected_otu_ids = [str(obs_ids[i]) for i in top_idx[:max_otus]]
    np.save(out_dir / "selected_otu_ids.npy", np.array(selected_otu_ids))

    meta = {
        "source": "Earth Microbiome Project Release 1 (FULL deblur 90bp)",
        "biom_file": str(biom_path),
        "n_samples": saved,
        "n_otus_original": int(table.shape[0]),
        "n_otus_selected": max_otus,
        "selection_method": "top prevalence (sparse, full release)",
        "normalization": "relative abundance (simplex)",
        "n_classes": len(valid_labels),
        "class_names": {int(k): v for k, v in SOURCE_CLASSES.items() if k in valid_labels},
        "class_counts": {int(k): int(counts[k]) for k in valid_labels},
        "note": "Coin-flip freshwater labels are KEPT in npz (same as subset); "
                "geographic loader excludes them at split time.",
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    return meta


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--biom", type=Path, default=DEFAULT_BIOM)
    ap.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    if not args.biom.exists() or args.biom.stat().st_size < 100_000_000:
        print(f"ERROR: full BIOM not found / too small at {args.biom}")
        sys.exit(1)

    stats = process(args.biom, args.mapping, args.out)
    print("\n" + "=" * 60)
    print("FULL-RELEASE PROCESSING COMPLETE")
    print("=" * 60)
    print(f"  Samples: {stats.get('n_samples', 0):,}")
    print(f"  Classes: {stats.get('n_classes', 0)}")
    print(f"  Output:  {args.out}")


if __name__ == "__main__":
    main()
