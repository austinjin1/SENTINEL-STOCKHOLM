#!/usr/bin/env python
"""Generate synthetic transcriptomic training data for ToxiGene module.

Creates realistic gene expression profiles with biologically-informed pathway
activation patterns, using downloaded Reactome data to build sparse adjacency
matrices for the gene -> pathway -> process -> outcome hierarchy.

Output files (in data/processed/molecular/):
  - expression_data.npz: expression matrix [n_samples, n_genes]
  - labels.npz: contaminant class labels
  - hierarchy.npz: pathway_adj, process_adj, outcome_adj matrices
  - gene_names.json: list of gene names
  - expression_matrix.npy: same expression data for train_molecular.py compat
  - expression_metadata.json: metadata for train_molecular.py compat
  - outcome_labels.npy: outcome labels for train_molecular.py compat
  - pathway_labels.npy: pathway activation targets
  - chemical_ids.json: chemical class IDs per sample
  - hierarchy_metadata.json: hierarchy dimension metadata

Usage:
    python scripts/generate_molecular_data.py [--n-samples 1000] [--n-genes 1000]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONTAMINANT_CLASSES = [
    "PAH",
    "heavy_metal",
    "endocrine",
    "pesticide",
    "pharmaceutical",
    "solvent",
    "nutrient",
]

# Biologically-informed pathway groupings for each contaminant class.
# Each class preferentially activates specific Reactome top-level pathways.
# These are Reactome top-level pathway keywords that we map to our gene sets.
CONTAMINANT_PATHWAY_SIGNATURES = {
    "PAH": {
        # PAHs: AhR activation, xenobiotic metabolism, DNA damage response
        "activated_keywords": [
            "metabolism of xenobiotics",
            "cytochrome p450",
            "dna repair",
            "dna damage",
            "apoptosis",
            "cellular response to stress",
        ],
        "base_activation": 0.3,
        "specific_activation": 2.5,
        "noise_scale": 0.4,
    },
    "heavy_metal": {
        # Heavy metals: oxidative stress, metallothioneins, protein misfolding
        "activated_keywords": [
            "cellular response to stress",
            "unfolded protein response",
            "metabolism of proteins",
            "apoptosis",
            "ion transport",
            "detoxification",
        ],
        "base_activation": 0.25,
        "specific_activation": 2.8,
        "noise_scale": 0.5,
    },
    "endocrine": {
        # EDCs: nuclear receptor signaling, steroid metabolism, reproduction
        "activated_keywords": [
            "nuclear receptor",
            "signaling by nuclear receptors",
            "estrogen",
            "metabolism of steroid",
            "reproductive",
            "gene expression",
        ],
        "base_activation": 0.2,
        "specific_activation": 3.0,
        "noise_scale": 0.35,
    },
    "pesticide": {
        # Pesticides: cholinesterase inhibition, neurotoxicity, immune effects
        "activated_keywords": [
            "neurotransmitter",
            "neuronal system",
            "immune system",
            "signal transduction",
            "metabolism of xenobiotics",
            "apoptosis",
        ],
        "base_activation": 0.3,
        "specific_activation": 2.2,
        "noise_scale": 0.45,
    },
    "pharmaceutical": {
        # Pharmaceuticals: target-specific pathways, metabolism
        "activated_keywords": [
            "metabolism",
            "signal transduction",
            "gene expression",
            "transport of small molecules",
            "programmed cell death",
            "cell cycle",
        ],
        "base_activation": 0.2,
        "specific_activation": 2.0,
        "noise_scale": 0.3,
    },
    "solvent": {
        # Solvents: membrane disruption, narcosis, general stress
        "activated_keywords": [
            "cellular response to stress",
            "membrane trafficking",
            "metabolism",
            "transport",
            "apoptosis",
            "lipid metabolism",
        ],
        "base_activation": 0.35,
        "specific_activation": 1.8,
        "noise_scale": 0.5,
    },
    "nutrient": {
        # Nutrient enrichment: growth signaling, metabolism upregulation
        "activated_keywords": [
            "metabolism",
            "cell cycle",
            "gene expression",
            "growth factor",
            "mtor",
            "pi3k",
        ],
        "base_activation": 0.15,
        "specific_activation": 2.0,
        "noise_scale": 0.25,
    },
}

# Adverse outcome categories (aligned with AOP-Wiki common outcomes)
ADVERSE_OUTCOMES = [
    "reproductive_impairment",
    "growth_inhibition",
    "immunosuppression",
    "neurotoxicity",
    "hepatotoxicity",
    "oxidative_damage",
    "endocrine_disruption",
]

# Mapping: which contaminant classes trigger which adverse outcomes
CONTAMINANT_TO_OUTCOMES = {
    "PAH": [0, 4, 5],              # reproductive, hepatotoxicity, oxidative
    "heavy_metal": [1, 2, 5],       # growth, immunosuppression, oxidative
    "endocrine": [0, 1, 6],         # reproductive, growth, endocrine
    "pesticide": [2, 3, 4],         # immunosuppression, neurotoxicity, hepato
    "pharmaceutical": [1, 4, 6],     # growth, hepatotoxicity, endocrine
    "solvent": [3, 4, 5],           # neurotoxicity, hepatotoxicity, oxidative
    "nutrient": [0, 1, 6],          # reproductive, growth, endocrine
}

# Biological process groupings (intermediate between pathways and outcomes)
PROCESS_KEYWORDS = [
    "metabolism",
    "signal transduction",
    "immune system",
    "cell cycle",
    "apoptosis",
    "gene expression",
    "transport",
    "cellular stress",
    "development",
    "neuronal",
    "reproduction",
    "dna repair",
]


def load_reactome_gene_pathway_mappings(
    reactome_dir: Path,
    n_genes: int,
    rng: np.random.Generator,
) -> tuple[list[str], list[str], np.ndarray, dict[str, list[int]]]:
    """Load Reactome gene-pathway associations for Danio rerio.

    Returns:
        gene_names: List of selected gene names.
        pathway_names: List of pathway names.
        pathway_adj: Binary adjacency [n_pathways, n_genes].
        pathway_keyword_map: Maps keywords to pathway indices.
    """
    assoc_file = reactome_dir / "Ensembl2Reactome_Danio_rerio.txt"

    if not assoc_file.exists():
        print(f"  Reactome association file not found at {assoc_file}")
        print("  Generating fully synthetic pathway structure instead.")
        return _generate_synthetic_pathway_structure(n_genes, rng)

    # Parse gene-pathway associations
    gene_to_pathways: dict[str, set[str]] = defaultdict(set)
    pathway_names_set: dict[str, str] = {}  # pathway_id -> pathway_name

    with open(assoc_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            gene_id = parts[0]
            pathway_id = parts[1]
            pathway_name = parts[3]
            gene_to_pathways[gene_id].add(pathway_id)
            pathway_names_set[pathway_id] = pathway_name

    print(f"  Reactome: {len(gene_to_pathways)} genes, {len(pathway_names_set)} pathways")

    # Select top genes by pathway connectivity (most informative)
    gene_list = sorted(
        gene_to_pathways.keys(),
        key=lambda g: len(gene_to_pathways[g]),
        reverse=True,
    )

    # Take top n_genes, but ensure diversity by also sampling some less-connected
    n_top = min(int(n_genes * 0.7), len(gene_list))
    n_random = min(n_genes - n_top, len(gene_list) - n_top)
    top_genes = gene_list[:n_top]
    if n_random > 0 and len(gene_list) > n_top:
        remaining = gene_list[n_top:]
        random_genes = list(rng.choice(remaining, size=n_random, replace=False))
        selected_genes = top_genes + random_genes
    else:
        selected_genes = top_genes[:n_genes]

    selected_genes = selected_genes[:n_genes]
    gene_set = set(selected_genes)

    # Filter pathways to those with at least 3 selected genes
    pathway_gene_counts: dict[str, int] = defaultdict(int)
    for gene in selected_genes:
        for pw in gene_to_pathways[gene]:
            pathway_gene_counts[pw] += 1

    valid_pathways = [
        pw for pw, count in pathway_gene_counts.items()
        if count >= 3
    ]
    # Limit to manageable number of pathways (50-200)
    valid_pathways.sort(key=lambda pw: pathway_gene_counts[pw], reverse=True)
    valid_pathways = valid_pathways[:200]

    print(f"  Selected {len(selected_genes)} genes, {len(valid_pathways)} pathways")

    # Build pathway adjacency matrix [n_pathways, n_genes]
    gene_idx = {g: i for i, g in enumerate(selected_genes)}
    pw_idx = {pw: i for i, pw in enumerate(valid_pathways)}

    pathway_adj = np.zeros((len(valid_pathways), len(selected_genes)), dtype=np.float32)
    for gene in selected_genes:
        gi = gene_idx[gene]
        for pw in gene_to_pathways[gene]:
            if pw in pw_idx:
                pathway_adj[pw_idx[pw], gi] = 1.0

    # Build keyword -> pathway index mapping
    pathway_keyword_map: dict[str, list[int]] = defaultdict(list)
    for pw_id, pi in pw_idx.items():
        pw_name = pathway_names_set.get(pw_id, "").lower()
        for keyword in (
            list(PROCESS_KEYWORDS)
            + [kw for sig in CONTAMINANT_PATHWAY_SIGNATURES.values()
               for kw in sig["activated_keywords"]]
        ):
            if keyword.lower() in pw_name:
                pathway_keyword_map[keyword.lower()].append(pi)

    pathway_names = [pathway_names_set.get(pw, pw) for pw in valid_pathways]

    return selected_genes, pathway_names, pathway_adj, dict(pathway_keyword_map)


def _generate_synthetic_pathway_structure(
    n_genes: int,
    rng: np.random.Generator,
) -> tuple[list[str], list[str], np.ndarray, dict[str, list[int]]]:
    """Fallback: generate fully synthetic pathway structure."""
    n_pathways = 150
    gene_names = [f"GENE_{i:04d}" for i in range(n_genes)]
    pathway_names = [f"PATHWAY_{i:03d}" for i in range(n_pathways)]

    # Sparse random adjacency (~5% density)
    pathway_adj = np.zeros((n_pathways, n_genes), dtype=np.float32)
    for i in range(n_pathways):
        n_genes_in_pw = rng.integers(10, 80)
        gene_indices = rng.choice(n_genes, size=n_genes_in_pw, replace=False)
        pathway_adj[i, gene_indices] = 1.0

    # Even distribution of keywords
    pathway_keyword_map: dict[str, list[int]] = {}
    all_keywords = set()
    for sig in CONTAMINANT_PATHWAY_SIGNATURES.values():
        all_keywords.update(kw.lower() for kw in sig["activated_keywords"])
    for i, kw in enumerate(sorted(all_keywords)):
        indices = list(range(i * 5, min((i + 1) * 5, n_pathways)))
        pathway_keyword_map[kw] = indices

    return gene_names, pathway_names, pathway_adj, pathway_keyword_map


def build_process_adjacency(
    pathway_names: list[str],
    n_pathways: int,
    rng: np.random.Generator,
) -> tuple[list[str], np.ndarray]:
    """Build pathway -> biological process adjacency matrix.

    Groups pathways into biological processes based on keyword matching
    and random assignment for unmatched pathways.

    Returns:
        process_names: List of process names.
        process_adj: Binary adjacency [n_processes, n_pathways].
    """
    n_processes = len(PROCESS_KEYWORDS)
    process_adj = np.zeros((n_processes, n_pathways), dtype=np.float32)

    # Keyword-based assignment
    assigned = set()
    for pi, pw_name in enumerate(pathway_names):
        pw_lower = pw_name.lower()
        for proc_i, keyword in enumerate(PROCESS_KEYWORDS):
            if keyword.lower() in pw_lower:
                process_adj[proc_i, pi] = 1.0
                assigned.add(pi)

    # Assign unmatched pathways randomly (each to 1-2 processes)
    unassigned = [i for i in range(n_pathways) if i not in assigned]
    for pi in unassigned:
        n_procs = rng.integers(1, 3)
        proc_indices = rng.choice(n_processes, size=n_procs, replace=False)
        for proc_i in proc_indices:
            process_adj[proc_i, pi] = 1.0

    # Ensure each process has at least some pathways
    for proc_i in range(n_processes):
        if process_adj[proc_i].sum() < 3:
            extra = rng.choice(n_pathways, size=5, replace=False)
            for pi in extra:
                process_adj[proc_i, pi] = 1.0

    return PROCESS_KEYWORDS.copy(), process_adj


def build_outcome_adjacency(
    n_processes: int,
    rng: np.random.Generator,
) -> tuple[list[str], np.ndarray]:
    """Build biological process -> adverse outcome adjacency matrix.

    Returns:
        outcome_names: List of adverse outcome names.
        outcome_adj: Binary adjacency [n_outcomes, n_processes].
    """
    n_outcomes = len(ADVERSE_OUTCOMES)
    outcome_adj = np.zeros((n_outcomes, n_processes), dtype=np.float32)

    # Biologically-informed mappings:
    # Each outcome is driven by specific biological processes
    outcome_process_map = {
        0: [0, 5, 10],       # reproductive_impairment: metabolism, gene expr, reproduction
        1: [3, 5, 8],        # growth_inhibition: cell cycle, gene expr, development
        2: [2, 4, 7],        # immunosuppression: immune, apoptosis, stress
        3: [1, 6, 9],        # neurotoxicity: signal transduction, transport, neuronal
        4: [0, 4, 7],        # hepatotoxicity: metabolism, apoptosis, stress
        5: [4, 7, 11],       # oxidative_damage: apoptosis, stress, dna repair
        6: [0, 1, 5, 10],    # endocrine_disruption: metabolism, signaling, gene expr, repro
    }

    for outcome_i, process_indices in outcome_process_map.items():
        for proc_i in process_indices:
            if proc_i < n_processes:
                outcome_adj[outcome_i, proc_i] = 1.0

    # Add some noise connections (biology is messy)
    for outcome_i in range(n_outcomes):
        n_extra = rng.integers(1, 3)
        extra_procs = rng.choice(n_processes, size=n_extra, replace=False)
        for proc_i in extra_procs:
            outcome_adj[outcome_i, proc_i] = 1.0

    return ADVERSE_OUTCOMES.copy(), outcome_adj


def generate_expression_profiles(
    n_samples: int,
    n_genes: int,
    n_pathways: int,
    pathway_adj: np.ndarray,
    pathway_keyword_map: dict[str, list[int]],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic gene expression profiles with class-specific patterns.

    Returns:
        expression: Gene expression matrix [n_samples, n_genes].
        labels: Contaminant class indices [n_samples].
        pathway_activations: Per-sample pathway activation levels [n_samples, n_pathways].
        outcome_labels: Binary adverse outcome labels [n_samples, n_outcomes].
    """
    n_classes = len(CONTAMINANT_CLASSES)
    n_outcomes = len(ADVERSE_OUTCOMES)
    samples_per_class = n_samples // n_classes
    remainder = n_samples % n_classes

    expression = np.zeros((n_samples, n_genes), dtype=np.float32)
    labels = np.zeros(n_samples, dtype=np.int64)
    pathway_activations = np.zeros((n_samples, n_pathways), dtype=np.float32)
    outcome_labels = np.zeros((n_samples, n_outcomes), dtype=np.float32)

    sample_idx = 0
    for class_idx, class_name in enumerate(CONTAMINANT_CLASSES):
        n_class = samples_per_class + (1 if class_idx < remainder else 0)
        sig = CONTAMINANT_PATHWAY_SIGNATURES[class_name]

        # Determine which pathways are activated for this class
        activated_pathway_indices = set()
        for kw in sig["activated_keywords"]:
            kw_lower = kw.lower()
            if kw_lower in pathway_keyword_map:
                activated_pathway_indices.update(pathway_keyword_map[kw_lower])

        # If keyword matching is sparse, also activate random pathways
        if len(activated_pathway_indices) < 10:
            extra = rng.choice(n_pathways, size=15, replace=False)
            activated_pathway_indices.update(extra.tolist())

        activated_list = sorted(activated_pathway_indices)

        for i in range(n_class):
            # Simulate dose-response variation (concentration effect)
            dose_factor = rng.uniform(0.3, 1.0)

            # Base pathway activations (low-level background)
            pw_act = np.full(n_pathways, sig["base_activation"], dtype=np.float32)

            # Class-specific pathway activation
            for pw_i in activated_list:
                pw_act[pw_i] = sig["specific_activation"] * dose_factor
                pw_act[pw_i] += rng.normal(0, sig["noise_scale"])

            # Add biological noise to all pathways
            pw_act += rng.normal(0, 0.1, size=n_pathways).astype(np.float32)
            pw_act = np.clip(pw_act, 0, None)
            pathway_activations[sample_idx] = pw_act

            # Generate gene expression from pathway activations
            # Each gene's expression is a weighted sum of its pathway memberships
            gene_expr = pathway_adj.T @ pw_act  # [n_genes]

            # Add gene-level noise (biological + technical)
            gene_expr += rng.normal(0, 0.3, size=n_genes).astype(np.float32)

            # Simulate log-normal expression distribution (typical of real RNA-seq)
            gene_expr = np.abs(gene_expr) + rng.exponential(0.5, size=n_genes).astype(np.float32)

            # Apply sample-specific scaling (library size variation)
            lib_size_factor = rng.lognormal(0, 0.3)
            gene_expr *= lib_size_factor

            expression[sample_idx] = gene_expr
            labels[sample_idx] = class_idx

            # Generate outcome labels based on contaminant type
            outcome_indices = CONTAMINANT_TO_OUTCOMES[class_name]
            for oi in outcome_indices:
                # Not all samples show all outcomes (dose-dependent)
                if dose_factor > 0.4 or rng.random() < 0.3:
                    outcome_labels[sample_idx, oi] = 1.0

            sample_idx += 1

    # Shuffle samples
    perm = rng.permutation(n_samples)
    expression = expression[perm]
    labels = labels[perm]
    pathway_activations = pathway_activations[perm]
    outcome_labels = outcome_labels[perm]

    # Normalize expression: log2(x + 1) transform (standard for RNA-seq)
    expression = np.log2(expression + 1)

    # Quantile normalization (per-gene z-score for cross-sample comparability)
    gene_means = expression.mean(axis=0, keepdims=True)
    gene_stds = expression.std(axis=0, keepdims=True) + 1e-8
    expression = (expression - gene_means) / gene_stds

    return expression, labels, pathway_activations, outcome_labels


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic transcriptomic training data for ToxiGene"
    )
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of samples")
    parser.add_argument("--n-genes", type=int, default=1000, help="Number of genes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--reactome-dir",
        type=str,
        default="data/raw/molecular/reactome",
        help="Path to downloaded Reactome data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed/molecular",
        help="Output directory",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    reactome_dir = Path(args.reactome_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating synthetic transcriptomic data:")
    print(f"  Samples: {args.n_samples}")
    print(f"  Genes: {args.n_genes}")
    print(f"  Classes: {len(CONTAMINANT_CLASSES)}")
    print(f"  Seed: {args.seed}")
    print()

    # Step 1: Load/build pathway structure from Reactome
    print("Step 1: Building gene-pathway hierarchy from Reactome data...")
    gene_names, pathway_names, pathway_adj, pathway_keyword_map = (
        load_reactome_gene_pathway_mappings(reactome_dir, args.n_genes, rng)
    )
    n_genes = len(gene_names)
    n_pathways = len(pathway_names)
    print(f"  Gene-pathway layer: {n_genes} genes x {n_pathways} pathways")
    print(f"  Pathway adjacency density: {pathway_adj.sum() / pathway_adj.size:.4f}")
    print()

    # Step 2: Build pathway -> process adjacency
    print("Step 2: Building pathway-to-process hierarchy...")
    process_names, process_adj = build_process_adjacency(pathway_names, n_pathways, rng)
    n_processes = len(process_names)
    print(f"  Process layer: {n_pathways} pathways x {n_processes} processes")
    print(f"  Process adjacency density: {process_adj.sum() / process_adj.size:.4f}")
    print()

    # Step 3: Build process -> outcome adjacency
    print("Step 3: Building process-to-outcome hierarchy...")
    outcome_names, outcome_adj = build_outcome_adjacency(n_processes, rng)
    n_outcomes = len(outcome_names)
    print(f"  Outcome layer: {n_processes} processes x {n_outcomes} outcomes")
    print(f"  Outcome adjacency density: {outcome_adj.sum() / outcome_adj.size:.4f}")
    print()

    # Step 4: Generate expression profiles
    print("Step 4: Generating gene expression profiles...")
    expression, labels, pathway_activations, outcome_labels = generate_expression_profiles(
        n_samples=args.n_samples,
        n_genes=n_genes,
        n_pathways=n_pathways,
        pathway_adj=pathway_adj,
        pathway_keyword_map=pathway_keyword_map,
        rng=rng,
    )
    print(f"  Expression matrix: {expression.shape}")
    print(f"  Labels: {labels.shape}, classes: {np.unique(labels)}")
    print(f"  Outcome labels: {outcome_labels.shape}, positives per outcome: {outcome_labels.sum(axis=0).astype(int)}")
    print()

    # Step 5: Save everything
    print("Step 5: Saving to", output_dir)

    # Primary outputs (compact format)
    np.savez_compressed(
        output_dir / "expression_data.npz",
        expression=expression,
    )
    np.savez_compressed(
        output_dir / "labels.npz",
        labels=labels,
        class_names=np.array(CONTAMINANT_CLASSES),
    )
    np.savez_compressed(
        output_dir / "hierarchy.npz",
        pathway_adj=pathway_adj,
        process_adj=process_adj,
        outcome_adj=outcome_adj,
    )
    with open(output_dir / "gene_names.json", "w", encoding="utf-8") as f:
        json.dump(gene_names, f, indent=2)

    # Compatibility outputs for train_molecular.py (ToxiGeneDataset format)
    np.save(output_dir / "expression_matrix.npy", expression)
    np.save(output_dir / "outcome_labels.npy", outcome_labels)
    np.save(output_dir / "pathway_labels.npy", pathway_activations)

    # Save hierarchy as individual scipy sparse matrices for load_hierarchy_adjacency
    scipy.sparse.save_npz(
        output_dir / "hierarchy_layer0_gene_to_pathway.npz",
        scipy.sparse.csr_matrix(pathway_adj),
    )
    scipy.sparse.save_npz(
        output_dir / "hierarchy_layer1_pathway_to_process.npz",
        scipy.sparse.csr_matrix(process_adj),
    )
    scipy.sparse.save_npz(
        output_dir / "hierarchy_layer2_process_to_outcome.npz",
        scipy.sparse.csr_matrix(outcome_adj),
    )

    # Metadata JSON files for train_molecular.py
    metadata = {
        "gene_names": gene_names,
        "sample_names": [f"sample_{i:04d}" for i in range(args.n_samples)],
        "shape": [args.n_samples, n_genes],
        "n_samples": args.n_samples,
        "n_genes": n_genes,
        "contaminant_classes": CONTAMINANT_CLASSES,
        "adverse_outcomes": ADVERSE_OUTCOMES,
    }
    with open(output_dir / "expression_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    hierarchy_metadata = {
        "n_pathways": n_pathways,
        "n_processes": n_processes,
        "n_outcomes": n_outcomes,
        "pathway_names": pathway_names,
        "process_names": process_names,
        "outcome_names": outcome_names,
    }
    with open(output_dir / "hierarchy_metadata.json", "w", encoding="utf-8") as f:
        json.dump(hierarchy_metadata, f, indent=2)

    # Chemical IDs (one per sample, mapping to class index)
    chemical_ids = {
        "chemical_ids": labels.tolist(),
        "chemical_names": CONTAMINANT_CLASSES,
    }
    with open(output_dir / "chemical_ids.json", "w", encoding="utf-8") as f:
        json.dump(chemical_ids, f, indent=2)

    # Summary
    print()
    print("=" * 60)
    print("Synthetic ToxiGene data generated successfully!")
    print("=" * 60)
    print(f"  Output directory: {output_dir}")
    print(f"  Expression: {expression.shape[0]} samples x {expression.shape[1]} genes")
    print(f"  Hierarchy: {n_genes} genes -> {n_pathways} pathways -> {n_processes} processes -> {n_outcomes} outcomes")
    print(f"  Pathway adj density: {pathway_adj.sum() / pathway_adj.size:.4f}")
    print(f"  Process adj density: {process_adj.sum() / process_adj.size:.4f}")
    print(f"  Outcome adj density: {outcome_adj.sum() / outcome_adj.size:.4f}")
    print()
    print("Files:")
    for p in sorted(output_dir.iterdir()):
        size_kb = p.stat().st_size / 1024
        print(f"  {p.name:45s} {size_kb:8.1f} KB")


if __name__ == "__main__":
    main()
