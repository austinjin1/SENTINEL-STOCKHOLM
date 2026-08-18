#!/usr/bin/env python3
"""Extract TRANSFERABLE taxonomic features from the full EMP Release-1 BIOM.

DIAGNOSIS
---------
The 0.405 geographic-holdout macro-F1 baseline (MicroBiomeNetV5) is trained on
the top-5000 *raw OTUs*. EMP deblur OTUs are exact 90bp amplicon sequence
variants (see BIOM observation ids: literal 90-nt DNA strings). Exact sequence
variants are highly region-specific: the same environment type in two different
continents is dominated by *different* ASVs of the *same* higher taxa. A model
keyed on raw ASVs therefore learns region-specific fingerprints that do NOT
transfer to held-out geographic regions -> poor cross-region macro-F1.

INNOVATION (real, no leakage)
-----------------------------
Replace raw-ASV features with COARSER, more transferable representations built
from the REAL Greengenes taxonomy strings shipped inside the BIOM
(k__;p__;c__;o__;f__;g__;s__ per OTU). We aggregate per-sample OTU abundances
up to phylum / class / order / family / genus level. Higher taxa are shared
across continents, so a "marine" vs "gut" vs "soil" signal expressed at the
family/genus level should generalise across regions far better than ASVs.

We produce, per sample, concatenated CLR-transformed relative-abundance vectors
at multiple taxonomic ranks. NO EMPO_3 / env_biome / env_feature is ever used
as a feature (label source only). NO synthetic data. Everything comes from the
real BIOM counts + real taxonomy.

OUTPUT
------
A single compressed .npz with:
  sample_ids   (N,)         str   — EMP #SampleID, matches processed npz site_id
  X_phylum     (N, Pp)      f32   — relative abundance per phylum
  X_class      (N, Pc)
  X_order      (N, Po)
  X_family     (N, Pf)
  X_genus      (N, Pg)
  names_*      lists of taxon names per rank
Saved to data/processed/microbial/emp_taxonomic_features.npz

REAL data only. MIT License — Bryan Cheng, 2026.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

try:
    from biom import load_table
except ImportError:
    print("biom-format not installed", file=sys.stderr)
    sys.exit(1)

BIOM = Path("data/raw/emp/emp_deblur_90bp.release1.biom")
OUT = Path("data/processed/microbial/emp_taxonomic_features.npz")

# taxonomy rank prefixes in Greengenes strings
RANK_PREFIX = {
    "phylum": "p__",
    "class": "c__",
    "order": "o__",
    "family": "f__",
    "genus": "g__",
}
# index of each rank in the 7-element taxonomy list [k,p,c,o,f,g,s]
RANK_IDX = {"phylum": 1, "class": 2, "order": 3, "family": 4, "genus": 5}


def build_group_matrix(taxa_labels: list[str], n_otus: int, rank: str):
    """Build a sparse (n_groups x n_otus) 0/1 aggregation matrix for a rank.

    Returns (G, group_names). G @ counts_matrix (otu x samp) sums OTU counts
    into taxon rows. Unclassified taxa at this rank map to a shared
    "<rank>__unclassified" bucket so nothing is silently dropped.
    """
    names: dict[str, int] = {}
    rows = []
    cols = []
    for j, lab in enumerate(taxa_labels):
        if lab not in names:
            names[lab] = len(names)
        rows.append(names[lab])
        cols.append(j)
    data = np.ones(len(rows), dtype=np.float32)
    G = sp.csr_matrix((data, (rows, cols)), shape=(len(names), n_otus))
    inv = [None] * len(names)
    for k, v in names.items():
        inv[v] = k
    return G, inv


def taxon_label(tax_list, rank: str) -> str:
    """Map a per-OTU taxonomy list to its label at `rank`.

    A taxon is 'defined' only if every ancestor rank down to `rank` is non-empty
    (e.g. p__Firmicutes;...;g__ is genus-unclassified). Otherwise pooled into a
    single '<rank>__unclassified' bucket. Uses the full lineage prefix so that
    e.g. two families named 'f__' under different orders don't collide.
    """
    idx = RANK_IDX[rank]
    if tax_list is None or len(tax_list) <= idx:
        return f"{rank}__unclassified"
    parts = []
    for i in range(1, idx + 1):  # start at phylum (skip kingdom for compactness)
        t = tax_list[i].strip() if i < len(tax_list) else ""
        # strip the "x__" prefix content check
        name = t.split("__", 1)[1] if "__" in t else t
        if name == "" or name.lower() in ("unclassified", "unidentified"):
            return f"{rank}__unclassified"
        parts.append(t)
    return ";".join(parts)


def main():
    t0 = time.time()
    print(f"Loading BIOM {BIOM} ...", flush=True)
    tbl = load_table(str(BIOM))
    n_otu, n_samp = tbl.shape
    print(f"  {n_otu:,} OTUs x {n_samp:,} samples  ({time.time()-t0:.1f}s)", flush=True)

    sample_ids = np.array([str(s) for s in tbl.ids(axis="sample")])

    # observation (OTU) taxonomy metadata
    obs_md = tbl.metadata(axis="observation")
    taxonomies = []
    for md in obs_md:
        tax = md.get("taxonomy") if md is not None else None
        taxonomies.append(tax)
    print(f"  parsed taxonomy for {len(taxonomies):,} OTUs ({time.time()-t0:.1f}s)", flush=True)

    # sparse OTU x sample count matrix (CSC for fast column ops; but we need G @ M)
    M = tbl.matrix_data.tocsr()  # otu x samp, sparse
    print(f"  count matrix nnz={M.nnz:,}  ({time.time()-t0:.1f}s)", flush=True)

    out = {"sample_ids": sample_ids}
    # per-sample total counts for normalisation (sum over OTUs)
    col_tot = np.asarray(M.sum(axis=0)).ravel().astype(np.float64)  # (n_samp,)
    col_tot_safe = np.where(col_tot > 0, col_tot, 1.0)

    for rank in ["phylum", "class", "order", "family", "genus"]:
        tr = time.time()
        labels = [taxon_label(t, rank) for t in taxonomies]
        G, names = build_group_matrix(labels, n_otu, rank)  # (n_groups x n_otu)
        # aggregate counts: (n_groups x n_samp)
        agg = G @ M  # sparse
        agg = np.asarray(agg.todense(), dtype=np.float64)  # groups are few -> dense OK
        # relative abundance per sample (columns), then transpose to (n_samp x groups)
        rel = (agg / col_tot_safe[None, :]).T.astype(np.float32)
        out[f"X_{rank}"] = rel
        out[f"names_{rank}"] = np.array(names, dtype=object)
        print(f"  [{rank}] {len(names)} taxa -> X shape {rel.shape}  ({time.time()-tr:.1f}s)", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **out)
    print(f"Wrote {OUT}  ({time.time()-t0:.1f}s total)", flush=True)


if __name__ == "__main__":
    main()
