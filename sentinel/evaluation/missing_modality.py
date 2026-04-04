"""Missing-modality robustness analysis for SENTINEL.

Evaluates how gracefully SENTINEL degrades when modalities are
randomly dropped, quantifying the system's robustness to partial
data availability.

Usage::

    python -m sentinel.evaluation.missing_modality \\
        --output-dir results/robustness \\
        --n-trials 100
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from sentinel.evaluation.ablation import MODALITIES, AblationCondition, run_ablation_condition
from sentinel.evaluation.metrics import compute_auc, bootstrap_ci
from sentinel.utils.logging import get_logger, make_progress

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DropTrialResult:
    """Result from a single random-drop trial."""

    trial_id: int
    num_dropped: int
    dropped_modalities: Tuple[str, ...]
    active_modalities: Tuple[str, ...]
    detection_auc: float
    mean_lead_time_hours: float
    source_attribution_top1: float


@dataclass
class DegradationCurve:
    """Aggregated performance at each drop level (0 through 4)."""

    num_dropped: int
    num_available: int
    mean_auc: float
    std_auc: float
    ci_lower: float
    ci_upper: float
    mean_lead_time: float
    std_lead_time: float
    num_trials: int


@dataclass
class RobustnessReport:
    """Complete robustness analysis report."""

    degradation_curves: List[DegradationCurve]
    graceful_degradation_score: float
    per_trial_results: List[DropTrialResult]
    modality_criticality: Dict[str, float]


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def run_missing_modality_analysis(
    n_trials: int = 100,
    seed: int = 42,
    device: torch.device = torch.device("cpu"),
) -> RobustnessReport:
    """Evaluate performance degradation under random modality dropout.

    For each trial, a random number of modalities (0 to 4) are dropped,
    and SENTINEL is evaluated on the remaining modalities across all
    case study events.

    Args:
        n_trials: Number of random-drop trials.
        seed: Random seed for reproducibility.
        device: Torch device for model inference.

    Returns:
        RobustnessReport with degradation curves and per-trial results.
    """
    rng = np.random.default_rng(seed)
    all_mods = list(MODALITIES)

    trial_results: List[DropTrialResult] = []
    progress = make_progress()

    logger.info(f"Running missing-modality analysis: {n_trials} trials")

    with progress:
        task = progress.add_task("Robustness trials", total=n_trials)

        for trial_id in range(n_trials):
            # Randomly choose how many to drop (0 to 4; must keep at least 1)
            num_drop = rng.integers(0, len(all_mods))  # 0..4
            drop_indices = rng.choice(len(all_mods), size=num_drop, replace=False)
            dropped = tuple(sorted(all_mods[i] for i in drop_indices))
            active = tuple(m for m in all_mods if m not in dropped)

            condition = AblationCondition.from_modalities(active)
            trial_seed = seed + trial_id * 7  # vary seed per trial

            result = run_ablation_condition(
                condition,
                seed=trial_seed,
                device=device,
            )

            trial_results.append(DropTrialResult(
                trial_id=trial_id,
                num_dropped=num_drop,
                dropped_modalities=dropped,
                active_modalities=active,
                detection_auc=result.detection_auc,
                mean_lead_time_hours=result.mean_lead_time_hours,
                source_attribution_top1=result.source_attribution_top1,
            ))
            progress.advance(task)

    # Build degradation curves
    degradation_curves = _build_degradation_curves(trial_results)

    # Graceful degradation score
    gd_score = compute_graceful_degradation_score(degradation_curves)

    # Modality criticality: average AUC drop when each modality is absent
    criticality = _compute_modality_criticality(trial_results)

    report = RobustnessReport(
        degradation_curves=degradation_curves,
        graceful_degradation_score=gd_score,
        per_trial_results=trial_results,
        modality_criticality=criticality,
    )

    logger.info(f"Graceful degradation score: {gd_score:.4f}")
    for mod, crit in sorted(criticality.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  Criticality of {mod}: {crit:.4f}")

    return report


def _build_degradation_curves(
    trial_results: List[DropTrialResult],
) -> List[DegradationCurve]:
    """Aggregate per-trial results into degradation curves by drop count.

    Args:
        trial_results: All trial results.

    Returns:
        List of DegradationCurve, one per drop level (0 through 4).
    """
    curves: List[DegradationCurve] = []

    for num_dropped in range(5):  # 0, 1, 2, 3, 4
        trials = [t for t in trial_results if t.num_dropped == num_dropped]
        if not trials:
            curves.append(DegradationCurve(
                num_dropped=num_dropped,
                num_available=5 - num_dropped,
                mean_auc=0.0,
                std_auc=0.0,
                ci_lower=0.0,
                ci_upper=0.0,
                mean_lead_time=0.0,
                std_lead_time=0.0,
                num_trials=0,
            ))
            continue

        aucs = np.array([t.detection_auc for t in trials])
        lead_times = np.array([t.mean_lead_time_hours for t in trials])

        mean_auc = float(np.mean(aucs))
        std_auc = float(np.std(aucs))

        # Bootstrap CI for mean AUC
        if len(aucs) >= 3:
            _, ci_lo, ci_hi = bootstrap_ci(aucs, n_bootstrap=2000)
        else:
            ci_lo, ci_hi = mean_auc - 2 * std_auc, mean_auc + 2 * std_auc

        curves.append(DegradationCurve(
            num_dropped=num_dropped,
            num_available=5 - num_dropped,
            mean_auc=mean_auc,
            std_auc=std_auc,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            mean_lead_time=float(np.mean(lead_times)),
            std_lead_time=float(np.std(lead_times)),
            num_trials=len(trials),
        ))

    return curves


def compute_graceful_degradation_score(
    curves: List[DegradationCurve],
) -> float:
    """Quantify how gracefully performance degrades with fewer modalities.

    Score interpretation:
      - 1.0 = no degradation (AUC constant regardless of drops)
      - 0.0 = linear degradation (AUC drops proportionally)
      - <0  = catastrophic degradation (worse than linear)

    The score is computed as 1 minus the normalized area between the
    actual degradation curve and a flat line at the full-fusion AUC,
    relative to the area under linear degradation.

    Args:
        curves: Degradation curves sorted by num_dropped.

    Returns:
        Graceful degradation score in approximately [-1, 1].
    """
    # Filter to curves with data
    valid = [c for c in curves if c.num_trials > 0]
    if len(valid) < 2:
        return 0.0

    # Sort by number available (descending)
    valid.sort(key=lambda c: c.num_available, reverse=True)

    full_auc = valid[0].mean_auc  # 5-modality AUC (or closest)
    if full_auc <= 0:
        return 0.0

    # Actual area under degradation curve (normalized by full AUC)
    x = np.array([c.num_available for c in valid], dtype=np.float64)
    y = np.array([c.mean_auc / full_auc for c in valid])

    # Normalize x to [0, 1] range
    x_norm = (x - x.min()) / max(x.max() - x.min(), 1e-8)

    # Actual area (trapezoidal)
    actual_area = float(np.trapz(y, x_norm))

    # Perfect area (no degradation) = 1.0
    perfect_area = 1.0

    # Linear degradation area: from 1.0 at full to some minimum at 1-modality
    min_y = y[-1] if len(y) > 1 else 0.0
    linear_area = (1.0 + min_y) / 2.0

    # Score: how much better than linear (0 = linear, 1 = perfect)
    if perfect_area - linear_area < 1e-8:
        return 1.0 if actual_area >= perfect_area - 1e-8 else 0.0

    score = (actual_area - linear_area) / (perfect_area - linear_area)
    return float(np.clip(score, -1.0, 1.0))


def _compute_modality_criticality(
    trial_results: List[DropTrialResult],
) -> Dict[str, float]:
    """Compute per-modality criticality scores.

    Criticality is the average AUC drop when a modality is absent vs
    when it is present, across all trials.

    Args:
        trial_results: All trial results.

    Returns:
        Dict mapping modality name to criticality score (higher = more critical).
    """
    criticality: Dict[str, float] = {}

    for mod in MODALITIES:
        present_aucs = [t.detection_auc for t in trial_results if mod in t.active_modalities]
        absent_aucs = [t.detection_auc for t in trial_results if mod not in t.active_modalities]

        if present_aucs and absent_aucs:
            criticality[mod] = float(np.mean(present_aucs) - np.mean(absent_aucs))
        else:
            criticality[mod] = 0.0

    return criticality


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def save_robustness_report(report: RobustnessReport, output_dir: Path) -> None:
    """Save the robustness report to JSON files.

    Args:
        report: The complete robustness report.
        output_dir: Directory to write output files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Degradation curves
    curves_data = []
    for c in report.degradation_curves:
        curves_data.append({
            "num_dropped": c.num_dropped,
            "num_available": c.num_available,
            "mean_auc": c.mean_auc,
            "std_auc": c.std_auc,
            "ci_lower": c.ci_lower,
            "ci_upper": c.ci_upper,
            "mean_lead_time": c.mean_lead_time,
            "std_lead_time": c.std_lead_time,
            "num_trials": c.num_trials,
        })
    with open(output_dir / "degradation_curves.json", "w", encoding="utf-8") as f:
        json.dump(curves_data, f, indent=2)

    # Per-trial results
    trials_data = []
    for t in report.per_trial_results:
        trials_data.append({
            "trial_id": t.trial_id,
            "num_dropped": t.num_dropped,
            "dropped_modalities": list(t.dropped_modalities),
            "active_modalities": list(t.active_modalities),
            "detection_auc": t.detection_auc,
            "mean_lead_time_hours": t.mean_lead_time_hours,
            "source_attribution_top1": t.source_attribution_top1,
        })
    with open(output_dir / "trial_results.json", "w", encoding="utf-8") as f:
        json.dump(trials_data, f, indent=2)

    # Summary
    summary = {
        "graceful_degradation_score": report.graceful_degradation_score,
        "modality_criticality": report.modality_criticality,
        "num_trials": len(report.per_trial_results),
    }
    with open(output_dir / "robustness_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Robustness report saved to {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for missing-modality analysis."""
    parser = argparse.ArgumentParser(
        description="SENTINEL Missing-Modality Robustness Analysis",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/robustness"),
        help="Output directory for results (default: results/robustness).",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of random-drop trials (default: 100).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device (default: cpu).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for missing-modality robustness analysis."""
    parser = build_parser()
    args = parser.parse_args(argv)

    device = torch.device(args.device)

    report = run_missing_modality_analysis(
        n_trials=args.n_trials,
        seed=args.seed,
        device=device,
    )

    save_robustness_report(report, args.output_dir)

    # Print summary table
    print("\n" + "=" * 65)
    print("SENTINEL Missing-Modality Robustness Analysis")
    print("=" * 65)
    print(f"\n{'Modalities Available':<25} {'Mean AUC':>10} {'Std':>8} {'95% CI':>20} {'Trials':>8}")
    print("-" * 75)
    for c in report.degradation_curves:
        print(
            f"{c.num_available:<25} {c.mean_auc:>10.4f} {c.std_auc:>8.4f} "
            f"[{c.ci_lower:.4f}, {c.ci_upper:.4f}] {c.num_trials:>8}"
        )
    print(f"\nGraceful Degradation Score: {report.graceful_degradation_score:.4f}")
    print(f"  (1.0 = no degradation, 0.0 = linear, <0 = catastrophic)\n")

    print("Modality Criticality (avg AUC drop when absent):")
    for mod, crit in sorted(report.modality_criticality.items(), key=lambda x: x[1], reverse=True):
        print(f"  {mod:<15} {crit:+.4f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
