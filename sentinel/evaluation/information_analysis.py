"""Cross-modal mutual information estimation for SENTINEL.

Quantifies the unique, redundant, and synergistic information across
the five SENTINEL modalities using neural MI estimation (MINE) and
KSG estimators, answering: "Are we measuring the same thing five
ways, or genuinely complementary signals?"

Usage::

    python -m sentinel.evaluation.information_analysis \\
        --embeddings-dir data/embeddings \\
        --output-dir results/information
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sentinel.evaluation.ablation import MODALITIES
from sentinel.evaluation.case_study import (
    HISTORICAL_EVENTS,
    build_timeline,
    generate_simulated_stream,
)
from sentinel.models.fusion.embedding_registry import SHARED_EMBEDDING_DIM
from sentinel.utils.logging import get_logger, make_progress

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# MINE (Mutual Information Neural Estimation)
# ---------------------------------------------------------------------------

class MINENetwork(nn.Module):
    """Statistics network for MINE estimator.

    Implements the T(x, y) function in the Donsker-Varadhan representation
    of KL divergence, used to lower-bound mutual information.

    Args:
        dim_x: Dimensionality of first variable.
        dim_y: Dimensionality of second variable.
        hidden_dim: Hidden layer width.
    """

    def __init__(self, dim_x: int, dim_y: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_x + dim_y, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute T(x, y).

        Args:
            x: Batch of first variable, shape ``(batch, dim_x)``.
            y: Batch of second variable, shape ``(batch, dim_y)``.

        Returns:
            Scalar statistics, shape ``(batch, 1)``.
        """
        return self.net(torch.cat([x, y], dim=-1))


def estimate_mi_mine(
    embeddings_a: np.ndarray,
    embeddings_b: np.ndarray,
    hidden_dim: int = 128,
    n_epochs: int = 200,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    seed: int = 42,
) -> float:
    """Estimate mutual information using MINE.

    Implements the Mutual Information Neural Estimation (Belghazi et al.,
    2018) with exponential moving average bias correction.

    Args:
        embeddings_a: First modality embeddings, shape ``(n, dim_a)``.
        embeddings_b: Second modality embeddings, shape ``(n, dim_b)``.
        hidden_dim: Hidden dimension for the statistics network.
        n_epochs: Number of training epochs.
        batch_size: Mini-batch size.
        learning_rate: Adam learning rate.
        seed: Random seed.

    Returns:
        Estimated mutual information in nats.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    n = min(len(embeddings_a), len(embeddings_b))
    a = torch.from_numpy(embeddings_a[:n].astype(np.float32))
    b = torch.from_numpy(embeddings_b[:n].astype(np.float32))

    dim_a = a.shape[1]
    dim_b = b.shape[1]

    net = MINENetwork(dim_a, dim_b, hidden_dim)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    # Exponential moving average for bias correction
    ema_weight = 0.01
    running_mean = 1.0

    mi_estimates: List[float] = []

    for epoch in range(n_epochs):
        # Sample joint (x, y) and marginal (x, y_shuffled)
        idx = rng.choice(n, size=min(batch_size, n), replace=False)
        idx_marginal = rng.permutation(idx)

        x_joint = a[idx]
        y_joint = b[idx]
        y_marginal = b[idx_marginal]

        # T(x, y) for joint
        t_joint = net(x_joint, y_joint)
        # T(x, y') for marginal
        t_marginal = net(x_joint, y_marginal)

        # MINE objective with EMA bias correction
        joint_mean = t_joint.mean()
        exp_marginal = torch.exp(t_marginal)
        marginal_log_mean = torch.log(exp_marginal.mean() + 1e-8)

        # Bias-corrected gradient
        running_mean = (
            (1 - ema_weight) * running_mean
            + ema_weight * exp_marginal.mean().item()
        )
        corrected = exp_marginal / max(running_mean, 1e-8)
        loss = -(joint_mean - torch.log(corrected.mean() + 1e-8))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()

        mi_est = float(joint_mean.item() - marginal_log_mean.item())
        mi_estimates.append(mi_est)

    # Use the mean of the last 20% of estimates as the final value
    tail = mi_estimates[int(0.8 * len(mi_estimates)):]
    return max(float(np.mean(tail)), 0.0)


# ---------------------------------------------------------------------------
# KSG estimator (Kraskov-Stoegbauer-Grassberger)
# ---------------------------------------------------------------------------

def estimate_mi_ksg(
    embeddings_a: np.ndarray,
    embeddings_b: np.ndarray,
    k: int = 5,
) -> float:
    """Estimate mutual information using the KSG estimator.

    Implements the first KSG estimator (Kraskov et al., 2004) based on
    k-nearest-neighbor distances in joint and marginal spaces.

    Args:
        embeddings_a: First modality embeddings, shape ``(n, dim_a)``.
        embeddings_b: Second modality embeddings, shape ``(n, dim_b)``.
        k: Number of nearest neighbors.

    Returns:
        Estimated mutual information in nats.
    """
    from scipy.special import digamma

    n = min(len(embeddings_a), len(embeddings_b))
    a = embeddings_a[:n].astype(np.float64)
    b = embeddings_b[:n].astype(np.float64)

    # Joint space
    joint = np.hstack([a, b])

    # Chebyshev distances to k-th neighbor in joint space
    # For efficiency, use a simple brute-force approach for moderate n
    if n > 5000:
        # Subsample for tractability
        idx = np.random.default_rng(42).choice(n, size=5000, replace=False)
        a = a[idx]
        b = b[idx]
        joint = np.hstack([a, b])
        n = 5000

    # Pairwise Chebyshev distances in joint space
    eps = np.zeros(n)
    for i in range(n):
        dists = np.max(np.abs(joint - joint[i]), axis=1)
        dists[i] = np.inf  # exclude self
        sorted_dists = np.sort(dists)
        eps[i] = sorted_dists[k - 1]  # k-th nearest neighbor distance

    # Count neighbors within eps in marginal spaces
    n_x = np.zeros(n, dtype=np.int64)
    n_y = np.zeros(n, dtype=np.int64)

    for i in range(n):
        dx = np.max(np.abs(a - a[i]), axis=1)
        dy = np.max(np.abs(b - b[i]), axis=1)
        n_x[i] = np.sum(dx < eps[i]) - 1  # exclude self
        n_y[i] = np.sum(dy < eps[i]) - 1

    # KSG formula
    mi = float(
        digamma(k)
        - np.mean(digamma(n_x + 1) + digamma(n_y + 1))
        + digamma(n)
    )
    return max(mi, 0.0)


# ---------------------------------------------------------------------------
# Unified MI interface
# ---------------------------------------------------------------------------

def estimate_mutual_information(
    embeddings_a: np.ndarray,
    embeddings_b: np.ndarray,
    method: str = "mine",
    **kwargs: Any,
) -> float:
    """Estimate mutual information between two sets of modality embeddings.

    Args:
        embeddings_a: First modality embeddings, shape ``(n, dim)``.
        embeddings_b: Second modality embeddings, shape ``(n, dim)``.
        method: Estimation method, one of ``"mine"`` or ``"ksg"``.
        **kwargs: Forwarded to the chosen estimator.

    Returns:
        Estimated mutual information in nats (non-negative).

    Raises:
        ValueError: If method is unknown.
    """
    if method == "mine":
        return estimate_mi_mine(embeddings_a, embeddings_b, **kwargs)
    elif method == "ksg":
        return estimate_mi_ksg(embeddings_a, embeddings_b, **kwargs)
    else:
        raise ValueError(f"Unknown MI method '{method}'. Use 'mine' or 'ksg'.")


# ---------------------------------------------------------------------------
# Information matrix
# ---------------------------------------------------------------------------

def compute_information_matrix(
    all_embeddings: Dict[str, np.ndarray],
    method: str = "mine",
    **kwargs: Any,
) -> np.ndarray:
    """Compute the 5x5 pairwise mutual information matrix.

    Diagonal entries are self-information estimates (entropy proxy via
    MI between two halves of the same embedding via a split-half approach).

    Off-diagonal entries are MI between modality pairs.

    Args:
        all_embeddings: Dict mapping modality name to embeddings array
            of shape ``(n, dim)``.
        method: MI estimation method.
        **kwargs: Forwarded to the estimator.

    Returns:
        5x5 numpy array of MI values in nats, indexed by MODALITIES order.
    """
    n_mod = len(MODALITIES)
    matrix = np.zeros((n_mod, n_mod))

    pairs_total = n_mod * (n_mod + 1) // 2
    progress = make_progress()

    with progress:
        task = progress.add_task("Computing MI matrix", total=pairs_total)

        for i, mod_i in enumerate(MODALITIES):
            for j, mod_j in enumerate(MODALITIES):
                if j < i:
                    # Symmetric: fill from upper triangle
                    matrix[i, j] = matrix[j, i]
                    continue

                emb_i = all_embeddings.get(mod_i)
                emb_j = all_embeddings.get(mod_j)

                if emb_i is None or emb_j is None:
                    logger.warning(f"Missing embeddings for {mod_i} or {mod_j}")
                    progress.advance(task)
                    continue

                if i == j:
                    # Self-information: split-half MI as entropy proxy
                    n = emb_i.shape[0]
                    half = n // 2
                    if half < 10:
                        matrix[i, j] = 0.0
                    else:
                        matrix[i, j] = estimate_mutual_information(
                            emb_i[:half], emb_i[half:2 * half],
                            method=method, **kwargs,
                        )
                else:
                    matrix[i, j] = estimate_mutual_information(
                        emb_i, emb_j, method=method, **kwargs,
                    )

                logger.info(
                    f"  MI({mod_i}, {mod_j}) = {matrix[i, j]:.4f} nats"
                )
                progress.advance(task)

    return matrix


# ---------------------------------------------------------------------------
# Unique information estimation
# ---------------------------------------------------------------------------

def compute_unique_information(
    all_embeddings: Dict[str, np.ndarray],
    mi_matrix: Optional[np.ndarray] = None,
    method: str = "mine",
    **kwargs: Any,
) -> Dict[str, float]:
    """Estimate information unique to each modality.

    Uses a practical approximation of partial information decomposition
    (PID): for each modality, the unique information is estimated as the
    self-information minus the maximum pairwise MI with any other modality.

    This approximation captures information that modality X provides but
    no single other modality can replicate.

    Args:
        all_embeddings: Dict mapping modality name to embeddings.
        mi_matrix: Pre-computed MI matrix (optional; computed if None).
        method: MI estimation method.
        **kwargs: Forwarded to the estimator.

    Returns:
        Dict mapping modality name to estimated unique information (nats).
    """
    if mi_matrix is None:
        mi_matrix = compute_information_matrix(all_embeddings, method=method, **kwargs)

    unique_info: Dict[str, float] = {}

    for i, mod in enumerate(MODALITIES):
        self_info = mi_matrix[i, i]
        # Max MI with any other modality
        off_diag = [mi_matrix[i, j] for j in range(len(MODALITIES)) if j != i]
        max_shared = max(off_diag) if off_diag else 0.0
        unique_info[mod] = max(self_info - max_shared, 0.0)

    return unique_info


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_information_report(
    mi_matrix: np.ndarray,
    unique_info: Dict[str, float],
) -> Dict[str, Any]:
    """Generate a comprehensive cross-modal information analysis report.

    Answers the question: "Are we measuring the same thing five ways,
    or genuinely complementary signals?"

    Args:
        mi_matrix: 5x5 pairwise MI matrix.
        unique_info: Per-modality unique information estimates.

    Returns:
        Dict with redundancy ratio, synergy estimates, and per-modality
        breakdown.
    """
    n_mod = len(MODALITIES)

    # Total pairwise MI (off-diagonal, upper triangle)
    total_pairwise_mi = 0.0
    pair_count = 0
    for i in range(n_mod):
        for j in range(i + 1, n_mod):
            total_pairwise_mi += mi_matrix[i, j]
            pair_count += 1
    mean_pairwise_mi = total_pairwise_mi / max(pair_count, 1)

    # Total self-information
    total_self_info = sum(mi_matrix[i, i] for i in range(n_mod))
    total_unique = sum(unique_info.values())

    # Redundancy ratio: fraction of total information that is shared
    # Higher = more redundant, lower = more complementary
    if total_self_info > 0:
        redundancy_ratio = 1.0 - (total_unique / total_self_info)
    else:
        redundancy_ratio = 0.0
    redundancy_ratio = float(np.clip(redundancy_ratio, 0.0, 1.0))

    # Complementarity score: inverse of redundancy
    complementarity = 1.0 - redundancy_ratio

    # Most redundant pair
    max_mi_val = 0.0
    most_redundant_pair = ("", "")
    for i in range(n_mod):
        for j in range(i + 1, n_mod):
            if mi_matrix[i, j] > max_mi_val:
                max_mi_val = mi_matrix[i, j]
                most_redundant_pair = (MODALITIES[i], MODALITIES[j])

    # Most independent pair
    min_mi_val = float("inf")
    most_independent_pair = ("", "")
    for i in range(n_mod):
        for j in range(i + 1, n_mod):
            if mi_matrix[i, j] < min_mi_val:
                min_mi_val = mi_matrix[i, j]
                most_independent_pair = (MODALITIES[i], MODALITIES[j])

    # Per-modality summary
    per_modality: List[Dict[str, Any]] = []
    for i, mod in enumerate(MODALITIES):
        per_modality.append({
            "modality": mod,
            "self_information": float(mi_matrix[i, i]),
            "unique_information": unique_info.get(mod, 0.0),
            "mean_mi_with_others": float(np.mean([
                mi_matrix[i, j] for j in range(n_mod) if j != i
            ])),
            "max_mi_with_other": float(max(
                mi_matrix[i, j] for j in range(n_mod) if j != i
            )),
        })

    report = {
        "summary": {
            "redundancy_ratio": redundancy_ratio,
            "complementarity_score": complementarity,
            "total_self_information": float(total_self_info),
            "total_unique_information": float(total_unique),
            "mean_pairwise_mi": float(mean_pairwise_mi),
            "interpretation": (
                "Highly complementary: modalities capture distinct signals."
                if complementarity > 0.6
                else "Moderately complementary: some shared information."
                if complementarity > 0.3
                else "Highly redundant: modalities largely overlap."
            ),
        },
        "mi_matrix": mi_matrix.tolist(),
        "modality_labels": list(MODALITIES),
        "most_redundant_pair": {
            "modalities": list(most_redundant_pair),
            "mi": float(max_mi_val),
        },
        "most_independent_pair": {
            "modalities": list(most_independent_pair),
            "mi": float(min_mi_val),
        },
        "per_modality": per_modality,
        "unique_information": {k: float(v) for k, v in unique_info.items()},
    }

    return report


# ---------------------------------------------------------------------------
# Embedding extraction from case studies
# ---------------------------------------------------------------------------

def extract_embeddings_from_case_studies(
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Extract per-modality embeddings from simulated case study streams.

    Runs the case study simulation and collects embeddings grouped by
    modality, providing data for information analysis.

    Args:
        seed: Random seed.

    Returns:
        Dict mapping modality name to stacked embeddings array.
    """
    embeddings: Dict[str, List[np.ndarray]] = {m: [] for m in MODALITIES}

    for i, (event_id, event) in enumerate(HISTORICAL_EVENTS.items()):
        rng = np.random.default_rng(seed + i)
        timeline = build_timeline(event)
        stream = generate_simulated_stream(timeline, rng=rng)

        for obs in stream:
            if obs.modality in embeddings:
                embeddings[obs.modality].append(obs.embedding)

    # Stack into arrays
    result: Dict[str, np.ndarray] = {}
    for mod, emb_list in embeddings.items():
        if emb_list:
            result[mod] = np.stack(emb_list)
            logger.info(f"  {mod}: {result[mod].shape[0]} embeddings")
        else:
            logger.warning(f"  {mod}: no embeddings collected")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for information analysis."""
    parser = argparse.ArgumentParser(
        description="SENTINEL Cross-Modal Mutual Information Analysis",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/information"),
        help="Output directory for results (default: results/information).",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["mine", "ksg"],
        default="mine",
        help="MI estimation method (default: mine).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--mine-epochs",
        type=int,
        default=200,
        help="MINE training epochs (default: 200).",
    )
    parser.add_argument(
        "--ksg-k",
        type=int,
        default=5,
        help="KSG number of neighbors (default: 5).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for cross-modal information analysis."""
    parser = build_parser()
    args = parser.parse_args(argv)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract embeddings from case studies
    logger.info("Extracting embeddings from case study simulations...")
    all_embeddings = extract_embeddings_from_case_studies(seed=args.seed)

    # Compute MI matrix
    estimator_kwargs: Dict[str, Any] = {}
    if args.method == "mine":
        estimator_kwargs["n_epochs"] = args.mine_epochs
    elif args.method == "ksg":
        estimator_kwargs["k"] = args.ksg_k

    logger.info(f"Computing MI matrix using {args.method} estimator...")
    mi_matrix = compute_information_matrix(
        all_embeddings, method=args.method, **estimator_kwargs,
    )

    # Compute unique information
    logger.info("Estimating unique information per modality...")
    unique_info = compute_unique_information(
        all_embeddings, mi_matrix=mi_matrix, method=args.method, **estimator_kwargs,
    )

    # Generate report
    report = generate_information_report(mi_matrix, unique_info)

    # Save outputs
    with open(output_dir / "information_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    np.save(output_dir / "mi_matrix.npy", mi_matrix)

    logger.info(f"Information analysis saved to {output_dir}")

    # Print summary
    print("\n" + "=" * 65)
    print("SENTINEL Cross-Modal Information Analysis")
    print("=" * 65)

    s = report["summary"]
    print(f"\nRedundancy ratio:     {s['redundancy_ratio']:.4f}")
    print(f"Complementarity:      {s['complementarity_score']:.4f}")
    print(f"Mean pairwise MI:     {s['mean_pairwise_mi']:.4f} nats")
    print(f"Interpretation:       {s['interpretation']}")

    print(f"\nMost redundant pair:  {report['most_redundant_pair']['modalities']} "
          f"(MI={report['most_redundant_pair']['mi']:.4f})")
    print(f"Most independent pair: {report['most_independent_pair']['modalities']} "
          f"(MI={report['most_independent_pair']['mi']:.4f})")

    print(f"\n{'Modality':<15} {'Self-Info':>10} {'Unique':>10} {'Mean MI w/ Others':>18}")
    print("-" * 58)
    for pm in report["per_modality"]:
        print(
            f"{pm['modality']:<15} {pm['self_information']:>10.4f} "
            f"{pm['unique_information']:>10.4f} {pm['mean_mi_with_others']:>18.4f}"
        )
    print("=" * 65)


if __name__ == "__main__":
    main()
