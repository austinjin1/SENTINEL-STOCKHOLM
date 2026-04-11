#!/usr/bin/env python3
"""Experiment 7: Cross-modal embedding alignment via CKA and mutual nearest neighbors.

Tests whether the four real-data modality embeddings share a common latent
structure, using two complementary metrics:

1. **Centered Kernel Alignment (CKA)** — measures representational similarity
   between all pairs of modalities independent of rotation or scaling.
   CKA = 1.0 means identical geometry; 0.0 means orthogonal.

2. **Mutual Nearest Neighbor rate (mNN)** — for each embedding in modality A,
   finds its cosine nearest neighbor in modality B, then checks if the reverse
   holds. High mNN = the two spaces share local neighborhood structure.

3. **Embedding space statistics** — norm distributions, inter-modal cosine
   similarity means, and within-vs-between modality similarity ratios.

Real embeddings used:
  - satellite:  2861 × 256  (HydroViT v6 on Sentinel-2 tiles)
  - sensor:     2000 × 256  (AquaSSM on real USGS sequences)
  - microbial:  5000 × 256  (MicroBiomeNet on EMP 16S rDNA)
  - behavioral: 3000 × 256  (BioMotion on real ECOTOX Daphnia)

Outputs:
  - results/exp7_crossmodal/alignment_results.json
  - paper/figures/fig_exp7_cka_matrix.jpg
  - paper/figures/fig_exp7_embedding_norms.jpg

MIT License — Bryan Cheng, 2026
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sentinel.utils.logging import get_logger

logger = get_logger(__name__)

EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "real_embeddings"
RESULTS_DIR = PROJECT_ROOT / "results" / "exp7_crossmodal"
FIGURES_DIR = PROJECT_ROOT / "paper" / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Subsample size for CKA (O(n²) memory — keep ≤ 1500)
CKA_SUBSAMPLE = 1000
# Subsample for mNN (cosine sim matrix is n×m)
MNN_SUBSAMPLE = 500


# ---------------------------------------------------------------------------
# CKA implementation
# ---------------------------------------------------------------------------

def centering(K: np.ndarray) -> np.ndarray:
    """Double-center a kernel (Gram) matrix."""
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute linear CKA between two feature matrices X (n×d1) and Y (n×d2).

    CKA(X, Y) = HSIC(X X^T, Y Y^T) / sqrt(HSIC(X X^T, X X^T) * HSIC(Y Y^T, Y Y^T))
    """
    # Gram matrices
    Kx = X @ X.T  # (n, n)
    Ky = Y @ Y.T  # (n, n)

    # Center
    Kxc = centering(Kx)
    Kyc = centering(Ky)

    # HSIC estimates (unbiased Frobenius inner product)
    hsic_xy = np.sum(Kxc * Kyc) / ((X.shape[0] - 1) ** 2)
    hsic_xx = np.sum(Kxc * Kxc) / ((X.shape[0] - 1) ** 2)
    hsic_yy = np.sum(Kyc * Kyc) / ((X.shape[0] - 1) ** 2)

    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-10:
        return 0.0
    return float(hsic_xy / denom)


# ---------------------------------------------------------------------------
# mNN implementation
# ---------------------------------------------------------------------------

def mutual_nearest_neighbors(X: np.ndarray, Y: np.ndarray, k: int = 1) -> float:
    """Compute mutual nearest neighbor rate between X and Y.

    For each x in X, find its k nearest neighbors in Y (by cosine similarity).
    For each of those y neighbors, check if x is among their k nearest in X.
    mNN rate = fraction of (x, y) pairs that are mutually nearest.

    Args:
        X: (n, d) float32 array, L2-normalized
        Y: (m, d) float32 array, L2-normalized
        k: number of neighbors

    Returns:
        mNN rate in [0, 1].
    """
    # Cosine similarity matrix (n, m) — both already L2-normalized
    sim_xy = X @ Y.T  # (n, m)
    sim_yx = sim_xy.T  # (m, n)

    # For each x, find top-k neighbors in Y
    top_k_xy = np.argsort(-sim_xy, axis=1)[:, :k]  # (n, k) indices in Y
    # For each y, find top-k neighbors in X
    top_k_yx = np.argsort(-sim_yx, axis=1)[:, :k]  # (m, k) indices in X

    mutual = 0
    total = X.shape[0]
    for i in range(total):
        for j in top_k_xy[i]:
            if i in top_k_yx[j]:
                mutual += 1
                break  # count each x at most once

    return mutual / total


# ---------------------------------------------------------------------------
# Load and preprocess embeddings
# ---------------------------------------------------------------------------

def load_all_embeddings() -> dict[str, np.ndarray]:
    """Load all real embedding files and L2-normalize."""
    files = {
        "satellite":  EMBEDDINGS_DIR / "satellite_embeddings.pt",
        "sensor":     EMBEDDINGS_DIR / "sensor_embeddings.pt",
        "microbial":  EMBEDDINGS_DIR / "microbial_embeddings.pt",
        "behavioral": EMBEDDINGS_DIR / "behavioral_embeddings.pt",
    }
    embs = {}
    for name, path in files.items():
        if not path.exists():
            logger.warning(f"  Missing: {path}")
            continue
        t = torch.load(str(path), map_location="cpu", weights_only=True)
        arr = t.float().numpy()  # (N, 256)
        # L2 normalize
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms > 1e-8, norms, 1.0)
        arr = arr / norms
        embs[name] = arr
        logger.info(f"  {name:12s}: {arr.shape}  norm_mean={np.linalg.norm(arr, axis=1).mean():.4f}")
    return embs


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def subsample(arr: np.ndarray, n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(arr), size=min(n, len(arr)), replace=False)
    return arr[idx]


def run_alignment_analysis(embs: dict[str, np.ndarray]) -> dict:
    modalities = list(embs.keys())
    n_mod = len(modalities)

    # ------------------------------------------------------------------
    # 1. CKA matrix
    # ------------------------------------------------------------------
    logger.info("Computing linear CKA matrix...")
    cka_matrix = np.zeros((n_mod, n_mod))
    for i, mi in enumerate(modalities):
        for j, mj in enumerate(modalities):
            Xi = subsample(embs[mi], CKA_SUBSAMPLE)
            Xj = subsample(embs[mj], CKA_SUBSAMPLE)
            # CKA requires same number of samples → use min
            n = min(len(Xi), len(Xj))
            cka_val = linear_cka(Xi[:n], Xj[:n])
            cka_matrix[i, j] = cka_val
            logger.info(f"  CKA({mi:<12s}, {mj:<12s}) = {cka_val:.4f}")

    # ------------------------------------------------------------------
    # 2. mNN matrix
    # ------------------------------------------------------------------
    logger.info("Computing mutual nearest neighbor rates...")
    mnn_matrix = np.zeros((n_mod, n_mod))
    for i, mi in enumerate(modalities):
        for j, mj in enumerate(modalities):
            if i == j:
                mnn_matrix[i, j] = 1.0
                continue
            Xi = subsample(embs[mi], MNN_SUBSAMPLE)
            Xj = subsample(embs[mj], MNN_SUBSAMPLE)
            rate = mutual_nearest_neighbors(Xi, Xj, k=1)
            mnn_matrix[i, j] = rate
            logger.info(f"  mNN({mi:<12s}, {mj:<12s}) = {rate:.4f}")

    # ------------------------------------------------------------------
    # 3. Within- vs. between-modality cosine similarity
    # ------------------------------------------------------------------
    logger.info("Computing within/between cosine similarity stats...")
    within_sim = {}
    for mi in modalities:
        Xi = subsample(embs[mi], 200)
        # Upper triangle cosine sim
        sim = Xi @ Xi.T  # (200, 200)
        upper = sim[np.triu_indices(len(Xi), k=1)]
        within_sim[mi] = {"mean": float(upper.mean()), "std": float(upper.std())}

    between_sim = {}
    for i, mi in enumerate(modalities):
        for j, mj in enumerate(modalities):
            if j <= i:
                continue
            Xi = subsample(embs[mi], 200)
            Xj = subsample(embs[mj], 200)
            sim = Xi @ Xj.T  # (200, 200)
            key = f"{mi}_vs_{mj}"
            between_sim[key] = {"mean": float(sim.mean()), "std": float(sim.std())}

    # ------------------------------------------------------------------
    # 4. Embedding norm statistics
    # ------------------------------------------------------------------
    norm_stats = {}
    for mi, arr in embs.items():
        norms = np.linalg.norm(arr, axis=1)
        norm_stats[mi] = {
            "mean": float(norms.mean()),
            "std": float(norms.std()),
            "min": float(norms.min()),
            "max": float(norms.max()),
            "n_embeddings": int(len(arr)),
            "embedding_dim": int(arr.shape[1]),
        }

    return {
        "modalities": modalities,
        "cka_matrix": cka_matrix.tolist(),
        "mnn_matrix": mnn_matrix.tolist(),
        "within_cosine_similarity": within_sim,
        "between_cosine_similarity": between_sim,
        "norm_stats": norm_stats,
    }


# ---------------------------------------------------------------------------
# Figure: CKA heatmap
# ---------------------------------------------------------------------------

def plot_cka_heatmap(results: dict):
    modalities = results["modalities"]
    cka_mat = np.array(results["cka_matrix"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # CKA
    ax = axes[0]
    labels = [m.capitalize() for m in modalities]
    mask = np.eye(len(modalities), dtype=bool)
    sns.heatmap(
        cka_mat, ax=ax, annot=True, fmt=".3f", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        vmin=0, vmax=1, linewidths=0.5,
        annot_kws={"size": 11, "weight": "bold"},
    )
    ax.set_title("Cross-Modal CKA (Linear)\n(1.0 = identical geometry)", fontsize=12)

    # mNN
    ax = axes[1]
    mnn_mat = np.array(results["mnn_matrix"])
    sns.heatmap(
        mnn_mat, ax=ax, annot=True, fmt=".3f", cmap="Greens",
        xticklabels=labels, yticklabels=labels,
        vmin=0, vmax=1, linewidths=0.5,
        annot_kws={"size": 11, "weight": "bold"},
    )
    ax.set_title("Mutual Nearest Neighbor Rate\n(1.0 = perfect alignment)", fontsize=12)

    plt.suptitle("SENTINEL Cross-Modal Embedding Alignment", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    path = FIGURES_DIR / "fig_exp7_cka_matrix.jpg"
    fig.savefig(str(path), dpi=150, bbox_inches="tight", pil_kwargs={"quality": 85})
    plt.close(fig)
    logger.info(f"Saved: {path}")


def plot_norm_distributions(results: dict, embs: dict):
    norm_stats = results["norm_stats"]
    modalities = results["modalities"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Norm distribution (violin)
    ax = axes[0]
    data = []
    labels_v = []
    for mi in modalities:
        arr = embs[mi]
        norms = np.linalg.norm(arr, axis=1)
        data.append(norms)
        labels_v.append(mi.capitalize())

    ax.violinplot(data, positions=range(len(modalities)), showmedians=True)
    ax.set_xticks(range(len(modalities)))
    ax.set_xticklabels(labels_v)
    ax.set_ylabel("L2 Norm (post-normalization ≈ 1.0)")
    ax.set_title("Embedding Norm Distribution")

    # Between-modality cosine similarity
    ax = axes[1]
    between = results["between_cosine_similarity"]
    pair_labels = [k.replace("_vs_", "\nvs\n").replace("_", " ").title() for k in between]
    means = [v["mean"] for v in between.values()]
    stds  = [v["std"]  for v in between.values()]

    colors = plt.cm.tab10(np.linspace(0, 0.6, len(pair_labels)))
    bars = ax.barh(pair_labels, means, xerr=stds, color=colors,
                   edgecolor="black", linewidth=0.5, capsize=3)
    ax.set_xlabel("Mean Cosine Similarity")
    ax.set_title("Between-Modality Cosine Similarity\n(L2-normalized embeddings)")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")

    plt.tight_layout()
    path = FIGURES_DIR / "fig_exp7_embedding_norms.jpg"
    fig.savefig(str(path), dpi=150, bbox_inches="tight", pil_kwargs={"quality": 85})
    plt.close(fig)
    logger.info(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    logger.info("=" * 65)
    logger.info("EXPERIMENT 7: Cross-Modal Embedding Alignment")
    logger.info("=" * 65)

    logger.info("Loading embeddings...")
    embs = load_all_embeddings()
    if len(embs) < 2:
        logger.error("Need at least 2 modalities; aborting.")
        return

    results = run_alignment_analysis(embs)
    results["elapsed_s"] = round(time.time() - t0, 1)

    # Print summary
    modalities = results["modalities"]
    logger.info("\n=== CKA SUMMARY ===")
    cka_mat = np.array(results["cka_matrix"])
    for i, mi in enumerate(modalities):
        for j, mj in enumerate(modalities):
            if j > i:
                logger.info(f"  CKA({mi}, {mj}) = {cka_mat[i,j]:.4f}")

    logger.info("\n=== mNN SUMMARY ===")
    mnn_mat = np.array(results["mnn_matrix"])
    for i, mi in enumerate(modalities):
        for j, mj in enumerate(modalities):
            if j > i:
                logger.info(f"  mNN({mi}, {mj}) = {mnn_mat[i,j]:.4f}")

    logger.info("\n=== COSINE SIMILARITY ===")
    for mod, stats in results["within_cosine_similarity"].items():
        logger.info(f"  Within-{mod}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    for pair, stats in results["between_cosine_similarity"].items():
        logger.info(f"  Between-{pair}: {stats['mean']:.4f} ± {stats['std']:.4f}")

    # Save results
    out_path = RESULTS_DIR / "alignment_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved: {out_path}")

    # Figures
    plot_cka_heatmap(results)
    plot_norm_distributions(results, embs)

    logger.info(f"\nExperiment 7 complete in {results['elapsed_s']:.1f}s")


if __name__ == "__main__":
    main()
