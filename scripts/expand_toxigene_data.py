#!/usr/bin/env python3
"""Expand ToxiGene training data by ~1.5-2x using class-conditional Gaussian augmentation.

Generates synthetic expression profiles by adding scaled Gaussian noise to
class-conditional mean vectors. Targets 1800 total samples (800 new synthetic).

MIT License — Bryan Cheng, 2026
"""

import numpy as np
from pathlib import Path

DATA_DIR = Path("data/processed/molecular")
RNG_SEED = 42


def main():
    rng = np.random.default_rng(RNG_SEED)

    # Load originals
    expression = np.load(DATA_DIR / "expression_matrix.npy")    # (1000, 1000)
    outcomes   = np.load(DATA_DIR / "outcome_labels.npy")       # (1000, 7)
    pathways   = np.load(DATA_DIR / "pathway_labels.npy")       # (1000, 200)

    n_orig, n_genes  = expression.shape
    n_classes = outcomes.shape[1]
    n_pathways = pathways.shape[1]

    print(f"Original data: {n_orig} samples, {n_genes} genes, {n_classes} classes, {n_pathways} pathways")

    # Multi-label: each sample may belong to >1 class.
    # class_freq[c] = fraction of samples positive for class c
    class_freq = outcomes.sum(0)          # (7,)
    total_pos  = class_freq.sum()         # total positives across all classes
    n_new_target = 800

    # Allocate new samples proportionally to class frequency
    class_alloc = np.round(class_freq / total_pos * n_new_target).astype(int)
    # Adjust for rounding to hit exactly n_new_target
    diff = n_new_target - class_alloc.sum()
    if diff != 0:
        # Add/subtract from the class with highest frequency
        idx = int(np.argmax(class_freq))
        class_alloc[idx] += diff

    print(f"New samples per class: {class_alloc} (total new: {class_alloc.sum()})")

    syn_expr     = []
    syn_outcomes = []
    syn_pathways = []

    for c in range(n_classes):
        # Samples that are positive for class c
        mask = outcomes[:, c] > 0.5
        if mask.sum() < 2:
            print(f"  Class {c}: too few samples ({mask.sum()}), skipping")
            continue

        expr_c = expression[mask]      # (n_c, 1000)
        path_c = pathways[mask]        # (n_c, 200)
        out_c  = outcomes[mask]        # (n_c, 7)  — multi-label, use as template

        expr_mean = expr_c.mean(axis=0)   # (1000,)
        expr_std  = expr_c.std(axis=0)    # (1000,)
        expr_std  = np.clip(expr_std, 1e-6, None)

        path_mean = path_c.mean(axis=0)   # (200,)
        path_std  = path_c.std(axis=0)
        path_std  = np.clip(path_std, 1e-6, None)

        n_c_new = class_alloc[c]
        if n_c_new == 0:
            continue

        # Generate synthetic expression: class_mean + N(0, 0.3 * class_std)
        noise_expr = rng.normal(0.0, 0.3 * expr_std, size=(n_c_new, n_genes)).astype(np.float32)
        new_expr = (expr_mean + noise_expr).astype(np.float32)

        # Generate synthetic pathways: path_mean + N(0, 0.1 * path_std), clipped ≥ 0
        noise_path = rng.normal(0.0, 0.1 * path_std, size=(n_c_new, n_pathways)).astype(np.float32)
        new_path = np.clip(path_mean + noise_path, 0.0, None).astype(np.float32)

        # Build outcome labels for new samples:
        # Use class mean of multi-hot labels (rounded to binary), then ensure class c is set
        mean_out = (out_c.mean(axis=0) > 0.5).astype(np.float32)
        mean_out[c] = 1.0
        new_out = np.tile(mean_out, (n_c_new, 1)).astype(np.float32)

        syn_expr.append(new_expr)
        syn_outcomes.append(new_out)
        syn_pathways.append(new_path)

        print(f"  Class {c}: {mask.sum()} orig → +{n_c_new} synthetic")

    # Concatenate
    syn_expr     = np.vstack(syn_expr)
    syn_outcomes = np.vstack(syn_outcomes)
    syn_pathways = np.vstack(syn_pathways)

    expr_v2     = np.vstack([expression, syn_expr])
    outcomes_v2 = np.vstack([outcomes, syn_outcomes])
    pathways_v2 = np.vstack([pathways, syn_pathways])

    n_new   = len(syn_expr)
    n_total = len(expr_v2)

    print()
    print(f"Generated {n_new} new samples, total: {n_orig} + {n_new} = {n_total}")
    print(f"  expression_matrix_v2: {expr_v2.shape}")
    print(f"  outcome_labels_v2:    {outcomes_v2.shape}")
    print(f"  pathway_labels_v2:    {pathways_v2.shape}")
    print(f"  New label dist:       {outcomes_v2.sum(0)}")

    np.save(DATA_DIR / "expression_matrix_v2.npy", expr_v2)
    np.save(DATA_DIR / "outcome_labels_v2.npy",    outcomes_v2)
    np.save(DATA_DIR / "pathway_labels_v2.npy",    pathways_v2)
    print("Saved v2 datasets.")


if __name__ == "__main__":
    main()
