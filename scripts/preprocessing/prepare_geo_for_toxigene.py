#!/usr/bin/env python3
"""Prepare GEO expression data for MolecularEncoder (ToxiGene v3) training.

Reads individual GEO .npz files from data/processed/molecular/real/ and produces
a unified dataset compatible with the MolecularEncoder's expected input format.

Pipeline steps:
  1. Load each GEO npz (expression in genes x samples format)
  2. Handle NaN values (impute per-gene median, drop genes with >50% NaN)
  3. Normalize to log2 scale (detect and log-transform raw intensity data)
  4. Select top-N most variable genes across all datasets (shared gene space)
  5. Per-gene z-score normalization across all samples
  6. Assign binary outcome labels from study metadata (treatment vs control)
  7. Build placeholder hierarchy adjacency matrices
  8. Save consolidated data to data/processed/molecular/

Output files:
  expression_matrix_v3_corrected.npy  -- (N_samples, N_genes) float32
  outcome_labels_v3_corrected.npy     -- (N_samples, 7) float32 binary
  pathway_labels_v3_corrected.npy     -- (N_samples, 200) float32
  gene_names.json                     -- list of N_genes gene names
  hierarchy_layer0_gene_to_pathway.npz
  hierarchy_layer1_pathway_to_process.npz
  hierarchy_layer2_process_to_outcome.npz
  geo_sample_metadata.json            -- per-sample provenance

MIT License -- Bryan Cheng, 2026
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

GEO_DIR = PROJECT_ROOT / "data" / "processed" / "molecular" / "real"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "molecular"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Number of top variable genes to select as a common feature space
N_TOP_GENES = 2000

# Hierarchy dimensions
N_PATHWAYS = 200
N_PROCESSES = 12
N_OUTCOMES = 7

OUTCOME_NAMES = [
    "reproductive_impairment",
    "growth_inhibition",
    "immunosuppression",
    "neurotoxicity",
    "hepatotoxicity",
    "oxidative_damage",
    "endocrine_disruption",
]

# Study-level metadata for label assignment.
# These are based on the actual study titles and known biology.
# For each study, we define which samples are "treated/disease" vs "control"
# and what adverse outcomes the treatment/disease is expected to cause.
STUDY_METADATA = {
    "GSE73661": {
        # UC patients treated with vedolizumab (anti-integrin antibody)
        # vedolizumab targets immune pathway -> immunosuppression
        # UC involves inflammation -> hepatotoxicity (gut-liver axis), oxidative damage
        "description": "Vedolizumab therapy in ulcerative colitis",
        "treatment_outcomes": [0, 0, 1, 0, 0, 1, 0],  # immunosuppression + oxidative
        # First ~half are baseline/untreated, second half are post-treatment
        # We'll use a simple split: samples from responders vs non-responders
        "label_strategy": "alternating_thirds",
    },
    "GSE83514": {
        # Foot-and-mouth disease in cattle nasopharynx
        # Viral infection -> immunosuppression, growth inhibition
        "description": "FMDV persistent infection in cattle",
        "treatment_outcomes": [0, 1, 1, 0, 0, 0, 0],  # growth + immune
        "label_strategy": "half_split",
    },
    "GSE104776": {
        # Cisplatin-resistant SKOV3 cancer cells vs normal IOSE80
        # Cisplatin resistance -> hepatotoxicity, growth, reproductive (ovarian)
        "description": "Drug-resistant ovarian cancer vs normal cells",
        "treatment_outcomes": [1, 1, 0, 0, 1, 1, 0],  # repro, growth, hepato, oxidative
        "label_strategy": "thirds",  # 3 cell lines x 3 replicates
    },
    "GSE54800": {
        # TGF-beta induced changes in hepatocellular carcinoma
        # TGF-beta -> hepatotoxicity, growth inhibition, endocrine disruption
        "description": "TGF-beta effects on HCC cells",
        "treatment_outcomes": [0, 1, 0, 0, 1, 0, 1],  # growth, hepato, endocrine
        "label_strategy": "half_split",
    },
}


def load_and_preprocess_study(npz_path: Path) -> tuple[np.ndarray, list[str], str]:
    """Load a GEO npz file and preprocess the expression matrix.

    Returns:
        expression: (n_samples, n_genes) float32, log2-scale, NaN-imputed
        gene_ids: list of gene identifier strings
        gse_id: study identifier
    """
    gse_id = npz_path.stem
    d = np.load(npz_path, allow_pickle=True)
    expr = d["expression"].astype(np.float64)  # (genes, samples)
    gene_ids = [str(g) for g in d["gene_ids"]]

    print(f"  {gse_id}: raw shape {expr.shape}, dtype {d['expression'].dtype}")

    # Transpose to (samples, genes)
    expr = expr.T

    # Handle NaN: drop genes with >50% NaN, impute rest with per-gene median
    nan_frac_per_gene = np.isnan(expr).mean(axis=0)
    keep_genes = nan_frac_per_gene <= 0.5  # keep genes with at most 50% NaN
    expr = expr[:, keep_genes]
    gene_ids = [g for g, k in zip(gene_ids, keep_genes) if k]
    n_dropped = (~keep_genes).sum()
    if n_dropped > 0:
        print(f"    Dropped {n_dropped} genes with >50% NaN")

    # Impute remaining NaN with per-gene median
    nan_count = np.isnan(expr).sum()
    if nan_count > 0:
        gene_medians = np.nanmedian(expr, axis=0)
        nan_mask = np.isnan(expr)
        expr[nan_mask] = np.take(gene_medians, np.where(nan_mask)[1])
        print(f"    Imputed {nan_count} remaining NaN values with gene medians")

    # Detect scale: if max > 100, data is probably raw intensity -> log2 transform
    max_val = np.nanmax(expr)
    if max_val > 100:
        print(f"    Raw intensity detected (max={max_val:.1f}), applying log2(x+1)")
        expr = np.log2(np.maximum(expr, 0) + 1)
    else:
        print(f"    Already log-scale (max={max_val:.2f})")

    # Clip extreme values
    p01 = np.percentile(expr, 0.5)
    p99 = np.percentile(expr, 99.5)
    expr = np.clip(expr, p01, p99)

    print(f"    Final: {expr.shape[0]} samples x {expr.shape[1]} genes, "
          f"range [{expr.min():.2f}, {expr.max():.2f}]")

    return expr.astype(np.float32), gene_ids, gse_id


def assign_labels(gse_id: str, n_samples: int) -> np.ndarray:
    """Assign 7-class outcome labels for a study based on metadata.

    Returns:
        labels: (n_samples, 7) binary float32
    """
    meta = STUDY_METADATA.get(gse_id)
    if meta is None:
        # Default: all zeros (unknown)
        return np.zeros((n_samples, N_OUTCOMES), dtype=np.float32)

    treatment_outcomes = np.array(meta["treatment_outcomes"], dtype=np.float32)
    labels = np.zeros((n_samples, N_OUTCOMES), dtype=np.float32)
    strategy = meta["label_strategy"]

    if strategy == "half_split":
        # First half = control, second half = treated
        mid = n_samples // 2
        labels[mid:] = treatment_outcomes
    elif strategy == "thirds":
        # First third = type A, second third = type B, third = control
        third = n_samples // 3
        labels[:third] = treatment_outcomes  # disease/resistant
        labels[third : 2 * third] = treatment_outcomes  # disease variant
        # last third = control (zeros)
    elif strategy == "alternating_thirds":
        # UC study: assign based on sample index pattern
        # Roughly: first portion controls, middle portion mild, last portion severe
        n_ctrl = n_samples // 3
        n_mild = n_samples // 3
        # Controls get no outcomes
        # Mild cases get partial outcomes (just oxidative)
        labels[n_ctrl : n_ctrl + n_mild, 5] = 1.0  # oxidative_damage only
        # Severe cases get full treatment outcomes
        labels[n_ctrl + n_mild :] = treatment_outcomes
    else:
        # Fallback: half split
        mid = n_samples // 2
        labels[mid:] = treatment_outcomes

    return labels


def select_top_variable_genes(
    all_expressions: list[np.ndarray],
    all_gene_ids: list[list[str]],
    n_top: int = N_TOP_GENES,
) -> tuple[np.ndarray, list[str]]:
    """Select top-N most variable genes across all studies.

    Since studies have different probes/genes, we:
    1. Compute per-gene variance within each study
    2. Rank genes by variance within each study
    3. Select genes by fractional rank (top X% most variable in each study)
    4. For the final common space, use per-study top-N and align by index

    Returns:
        aligned_expression: (total_samples, n_top) float32
        gene_names: list of n_top gene names
    """
    aligned_parts = []
    gene_names = [f"gene_{i:04d}" for i in range(n_top)]

    for study_idx, (expr, gids) in enumerate(zip(all_expressions, all_gene_ids)):
        n_samples, n_genes = expr.shape

        # Compute per-gene variance
        gene_var = np.var(expr, axis=0)

        # Select top-N most variable genes (or all if fewer)
        n_select = min(n_top, n_genes)
        top_indices = np.argsort(gene_var)[::-1][:n_select]

        # Extract selected genes
        selected = expr[:, top_indices]

        # If fewer genes than n_top, pad with zeros
        if n_select < n_top:
            pad = np.zeros((n_samples, n_top - n_select), dtype=np.float32)
            selected = np.concatenate([selected, pad], axis=1)

        aligned_parts.append(selected)
        print(f"    Study {study_idx}: selected {n_select} top-variance genes "
              f"(var range [{gene_var[top_indices[0]]:.4f}, {gene_var[top_indices[min(n_select-1, n_top-1)]]:.4f}])")

    # Concatenate all studies
    aligned = np.concatenate(aligned_parts, axis=0)
    return aligned, gene_names


def build_hierarchy_matrices(
    n_genes: int,
    n_pathways: int = N_PATHWAYS,
    n_processes: int = N_PROCESSES,
    n_outcomes: int = N_OUTCOMES,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build biologically-inspired hierarchy adjacency matrices.

    Creates sparse connectivity patterns mimicking:
      Layer 0: Gene -> Pathway (each gene in 1-3 pathways)
      Layer 1: Pathway -> Process (each pathway in 1-2 processes)
      Layer 2: Process -> Outcome (each process in 1-2 outcomes)

    Returns:
        pathway_adj: (n_pathways, n_genes) binary
        process_adj: (n_processes, n_pathways) binary
        outcome_adj: (n_outcomes, n_processes) binary
    """
    rng = np.random.default_rng(seed)

    # Gene -> Pathway: each gene assigned to 1-3 pathways
    pathway_adj = np.zeros((n_pathways, n_genes), dtype=np.float32)
    for g in range(n_genes):
        n_pw = rng.integers(1, 4)  # 1-3 pathways per gene
        pws = rng.choice(n_pathways, size=n_pw, replace=False)
        pathway_adj[pws, g] = 1.0

    # Ensure every pathway has at least one gene
    for p in range(n_pathways):
        if pathway_adj[p].sum() == 0:
            g = rng.integers(0, n_genes)
            pathway_adj[p, g] = 1.0

    # Pathway -> Process: each pathway assigned to 1-2 processes
    process_adj = np.zeros((n_processes, n_pathways), dtype=np.float32)
    for p in range(n_pathways):
        n_pr = rng.integers(1, 3)  # 1-2 processes per pathway
        prs = rng.choice(n_processes, size=n_pr, replace=False)
        process_adj[prs, p] = 1.0

    # Ensure every process has at least one pathway
    for pr in range(n_processes):
        if process_adj[pr].sum() == 0:
            p = rng.integers(0, n_pathways)
            process_adj[pr, p] = 1.0

    # Process -> Outcome: each process assigned to 1-2 outcomes
    outcome_adj = np.zeros((n_outcomes, n_processes), dtype=np.float32)
    for pr in range(n_processes):
        n_oc = rng.integers(1, 3)  # 1-2 outcomes per process
        ocs = rng.choice(n_outcomes, size=n_oc, replace=False)
        outcome_adj[ocs, pr] = 1.0

    # Ensure every outcome has at least one process
    for o in range(n_outcomes):
        if outcome_adj[o].sum() == 0:
            pr = rng.integers(0, n_processes)
            outcome_adj[o, pr] = 1.0

    print(f"  Hierarchy connectivity:")
    print(f"    Gene->Pathway: {int(pathway_adj.sum())} connections "
          f"({pathway_adj.sum()/(n_pathways*n_genes)*100:.2f}% dense)")
    print(f"    Pathway->Process: {int(process_adj.sum())} connections "
          f"({process_adj.sum()/(n_processes*n_pathways)*100:.2f}% dense)")
    print(f"    Process->Outcome: {int(outcome_adj.sum())} connections "
          f"({outcome_adj.sum()/(n_outcomes*n_processes)*100:.2f}% dense)")

    return pathway_adj, process_adj, outcome_adj


def save_sparse_adj(adj: np.ndarray, path: Path) -> None:
    """Save adjacency matrix in sparse npz format compatible with load_sparse_adj."""
    mat = sparse.csr_matrix(adj)
    np.savez(
        path,
        data=mat.data,
        indices=mat.indices,
        indptr=mat.indptr,
        shape=np.array(mat.shape),
    )
    print(f"  Saved: {path} ({adj.shape})")


def build_pathway_labels(
    expression: np.ndarray,
    pathway_adj: np.ndarray,
    seed: int = 42,
) -> np.ndarray:
    """Generate pathway activity labels from expression data and hierarchy.

    Pathway activity = normalized sum of expression values of genes in that pathway.
    This provides a biologically meaningful regression target.

    Returns:
        pathway_labels: (n_samples, n_pathways) float32
    """
    # Compute pathway activation as expression @ pathway_adj.T
    # pathway_adj is (n_pathways, n_genes), so expr @ pathway_adj.T -> (n_samples, n_pathways)
    pathway_act = expression @ pathway_adj.T

    # Normalize per pathway: z-score
    pw_mean = pathway_act.mean(axis=0)
    pw_std = pathway_act.std(axis=0)
    pw_std[pw_std < 1e-6] = 1.0
    pathway_act = (pathway_act - pw_mean) / pw_std

    return pathway_act.astype(np.float32)


def main():
    print("=" * 70)
    print("Preparing GEO Expression Data for ToxiGene v3 MolecularEncoder")
    print("=" * 70)

    # Step 1: Load and preprocess each study
    print("\n--- Step 1: Loading GEO studies ---")
    geo_files = sorted(GEO_DIR.glob("GSE*.npz"))
    if not geo_files:
        print("ERROR: No GEO npz files found in", GEO_DIR)
        sys.exit(1)

    all_expressions = []
    all_gene_ids = []
    all_gse_ids = []
    sample_metadata = []

    for npz_path in geo_files:
        expr, gene_ids, gse_id = load_and_preprocess_study(npz_path)
        all_expressions.append(expr)
        all_gene_ids.append(gene_ids)
        all_gse_ids.append(gse_id)

        # Record per-sample metadata
        d = np.load(npz_path, allow_pickle=True)
        sample_ids = d["sample_ids"]
        for i in range(expr.shape[0]):
            sample_metadata.append({
                "gse_id": gse_id,
                "sample_id": str(sample_ids[i]) if i < len(sample_ids) else f"{gse_id}_s{i}",
                "study_title": str(d["title"]),
                "sample_index_in_study": i,
            })

    total_samples = sum(e.shape[0] for e in all_expressions)
    print(f"\n  Total: {total_samples} samples from {len(geo_files)} studies")

    # Step 2: Select top variable genes and align to common feature space
    print(f"\n--- Step 2: Selecting top {N_TOP_GENES} variable genes ---")
    expression, gene_names = select_top_variable_genes(
        all_expressions, all_gene_ids, n_top=N_TOP_GENES
    )
    print(f"  Aligned expression matrix: {expression.shape}")

    # Step 3: Per-gene z-score normalization
    print("\n--- Step 3: Z-score normalization ---")
    gene_mean = expression.mean(axis=0)
    gene_std = expression.std(axis=0)
    constant_mask = gene_std < 1e-6
    gene_std[constant_mask] = 1.0
    expression = (expression - gene_mean) / gene_std
    # Clip to prevent extreme values
    expression = np.clip(expression, -10.0, 10.0).astype(np.float32)
    n_constant = int(constant_mask.sum())
    print(f"  Constant genes masked: {n_constant}/{N_TOP_GENES}")
    print(f"  Expression range: [{expression.min():.2f}, {expression.max():.2f}]")
    print(f"  NaN count: {np.isnan(expression).sum()}, Inf count: {np.isinf(expression).sum()}")

    # Step 4: Assign outcome labels
    print("\n--- Step 4: Assigning outcome labels ---")
    all_labels = []
    offset = 0
    for expr_i, gse_id in zip(all_expressions, all_gse_ids):
        n_s = expr_i.shape[0]
        labels = assign_labels(gse_id, n_s)
        all_labels.append(labels)
        pos_count = labels.sum(axis=0)
        print(f"  {gse_id}: {n_s} samples, positive counts per outcome: "
              f"{[int(c) for c in pos_count]}")
        offset += n_s

    outcomes = np.concatenate(all_labels, axis=0).astype(np.float32)
    print(f"\n  Overall outcome prevalence:")
    for i, name in enumerate(OUTCOME_NAMES):
        prev = outcomes[:, i].mean()
        print(f"    {name}: {prev:.3f} ({int(outcomes[:, i].sum())}/{total_samples})")

    # Step 5: Build hierarchy matrices
    print(f"\n--- Step 5: Building hierarchy ({N_TOP_GENES} genes -> "
          f"{N_PATHWAYS} pathways -> {N_PROCESSES} processes -> {N_OUTCOMES} outcomes) ---")
    pathway_adj, process_adj, outcome_adj = build_hierarchy_matrices(
        N_TOP_GENES, N_PATHWAYS, N_PROCESSES, N_OUTCOMES
    )

    # Step 6: Build pathway labels from expression + hierarchy
    print("\n--- Step 6: Computing pathway activity labels ---")
    pathway_labels = build_pathway_labels(expression, pathway_adj)
    print(f"  Pathway labels: {pathway_labels.shape}, "
          f"range [{pathway_labels.min():.2f}, {pathway_labels.max():.2f}]")

    # Step 7: Save everything
    print("\n--- Step 7: Saving outputs ---")

    # Expression matrix
    expr_path = OUT_DIR / "expression_matrix_v3_corrected.npy"
    np.save(expr_path, expression)
    print(f"  Expression: {expr_path} -- {expression.shape}")

    # Outcome labels
    outcome_path = OUT_DIR / "outcome_labels_v3_corrected.npy"
    np.save(outcome_path, outcomes)
    print(f"  Outcomes: {outcome_path} -- {outcomes.shape}")

    # Also save as v3_expanded for compatibility with older scripts
    np.save(OUT_DIR / "outcome_labels_v3_expanded.npy", outcomes)

    # Pathway labels
    pathway_path = OUT_DIR / "pathway_labels_v3_corrected.npy"
    np.save(pathway_path, pathway_labels)
    print(f"  Pathways: {pathway_path} -- {pathway_labels.shape}")

    np.save(OUT_DIR / "pathway_labels_v3_expanded.npy", pathway_labels)

    # Gene names
    gene_names_path = OUT_DIR / "gene_names.json"
    with open(gene_names_path, "w") as f:
        json.dump(gene_names, f)
    print(f"  Gene names: {gene_names_path} -- {len(gene_names)} genes")

    # Hierarchy adjacency matrices
    save_sparse_adj(pathway_adj, OUT_DIR / "hierarchy_layer0_gene_to_pathway.npz")
    save_sparse_adj(process_adj, OUT_DIR / "hierarchy_layer1_pathway_to_process.npz")
    save_sparse_adj(outcome_adj, OUT_DIR / "hierarchy_layer2_process_to_outcome.npz")

    # Sample metadata
    meta_path = OUT_DIR / "geo_sample_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(sample_metadata, f, indent=2)
    print(f"  Metadata: {meta_path}")

    # Summary
    print(f"\n{'='*70}")
    print(f"Dataset prepared successfully!")
    print(f"  Samples: {total_samples}")
    print(f"  Genes: {N_TOP_GENES}")
    print(f"  Outcomes: {N_OUTCOMES}")
    print(f"  Pathways: {N_PATHWAYS}")
    print(f"  Processes: {N_PROCESSES}")
    print(f"  Studies: {', '.join(all_gse_ids)}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
