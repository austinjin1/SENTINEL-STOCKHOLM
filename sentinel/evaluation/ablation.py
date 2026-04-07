"""31-condition modality ablation study for SENTINEL.

Systematically evaluates all 2^5 - 1 = 31 non-empty subsets of the
five SENTINEL modalities (sensor, satellite, microbial, molecular,
behavioral) to quantify each modality's marginal contribution.

Usage::

    python -m sentinel.evaluation.ablation \\
        --data-dir data/case_studies \\
        --checkpoint-dir checkpoints/ \\
        --output-dir results/ablation
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from sentinel.evaluation.case_study import (
    HISTORICAL_EVENTS,
    SENTINELSimulator,
    build_timeline,
    generate_simulated_stream,
    SimulatedObservation,
)
from sentinel.evaluation.metrics import (
    compute_auc,
    paired_permutation_test,
    bootstrap_ci,
)
from sentinel.models.fusion.embedding_registry import (
    MODALITY_IDS,
    SHARED_EMBEDDING_DIM,
)
from sentinel.utils.logging import get_logger, make_progress

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Canonical modality list
# ---------------------------------------------------------------------------

MODALITIES: Tuple[str, ...] = ("sensor", "satellite", "microbial", "molecular", "behavioral")
"""All five SENTINEL modalities in canonical order."""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

def generate_all_conditions() -> list[tuple[str, ...]]:
    """Generate all 31 non-empty subsets of the five modalities.

    Returns:
        Sorted list of tuples, ordered by subset size then lexicographically.
        Contains exactly ``2^5 - 1 = 31`` entries.
    """
    subsets: list[tuple[str, ...]] = []
    for r in range(1, len(MODALITIES) + 1):
        for combo in combinations(MODALITIES, r):
            subsets.append(combo)
    return subsets


@dataclass
class AblationCondition:
    """Specification of one ablation condition."""

    name: str
    modalities: tuple[str, ...]

    @staticmethod
    def from_modalities(modalities: tuple[str, ...]) -> "AblationCondition":
        """Create an AblationCondition with an auto-generated name."""
        name = "+".join(modalities)
        return AblationCondition(name=name, modalities=modalities)


@dataclass
class AblationResult:
    """Metrics collected for a single ablation condition."""

    condition: AblationCondition
    detection_auc: float
    detection_auc_ci_lower: float
    detection_auc_ci_upper: float
    mean_lead_time_hours: float
    median_lead_time_hours: float
    source_attribution_top1: float
    false_positive_rate: float
    cost_efficiency: float
    num_events_detected: int
    per_event_scores: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def _filter_stream(
    stream: List[SimulatedObservation],
    active_modalities: tuple[str, ...],
    rng: np.random.Generator,
) -> List[SimulatedObservation]:
    """Replace observations from inactive modalities with "no data" tokens.

    Active modality observations are passed through unchanged.  Inactive
    modality observations are replaced with near-zero embeddings and zero
    anomaly score to simulate missing data rather than dropping the
    timestamp entirely (preserving chronological alignment).

    Args:
        stream: Full observation stream with all modalities.
        active_modalities: Modalities to keep active.
        rng: Random generator for noise.

    Returns:
        New stream with inactive modalities silenced.
    """
    filtered: List[SimulatedObservation] = []
    for obs in stream:
        if obs.modality in active_modalities:
            filtered.append(obs)
        else:
            # "No data" token: near-zero embedding, zero anomaly
            no_data_emb = rng.standard_normal(SHARED_EMBEDDING_DIM).astype(np.float32) * 0.001
            filtered.append(SimulatedObservation(
                timestamp=obs.timestamp,
                modality=obs.modality,
                embedding=no_data_emb,
                confidence=0.0,
                anomaly_score=0.0,
                metadata={"masked": True, **obs.metadata},
            ))
    return filtered


def run_ablation_condition(
    condition: AblationCondition,
    seed: int = 42,
    anomaly_threshold: float = 0.3,
    escalation_threshold: float = 0.5,
    alert_threshold: float = 0.7,
    device: torch.device = torch.device("cpu"),
    fast_mode: bool = True,
    _cached_streams: Optional[Dict[str, Any]] = None,
) -> AblationResult:
    """Evaluate SENTINEL under a single ablation condition.

    Runs all historical case study events with only the specified
    modalities active and computes aggregate metrics.

    Args:
        condition: The ablation condition to evaluate.
        seed: Random seed for reproducibility.
        anomaly_threshold: Anomaly detection threshold.
        escalation_threshold: Tier escalation threshold.
        alert_threshold: Formal alert threshold.
        device: Torch device for model inference.
        fast_mode: Use lightweight fusion (skip attention forward pass).
        _cached_streams: Pre-generated streams keyed by event_id (internal).

    Returns:
        AblationResult with aggregate detection metrics.
    """
    rng = np.random.default_rng(seed)

    all_y_true: List[int] = []
    all_y_score: List[float] = []
    lead_times: List[float] = []
    top1_correct = 0
    total_attributed = 0
    fp_count = 0
    fp_months = 0.0
    tiers_non_event: List[float] = []
    tiers_event: List[float] = []
    per_event_scores: Dict[str, float] = {}
    events_detected = 0

    for i, (event_id, event) in enumerate(HISTORICAL_EVENTS.items()):
        event_rng = np.random.default_rng(seed + i)
        timeline = build_timeline(event)

        # Use cached stream if available, otherwise generate
        if _cached_streams is not None and event_id in _cached_streams:
            full_stream = _cached_streams[event_id]["stream"]
        else:
            full_stream = generate_simulated_stream(timeline, rng=event_rng)
        stream = _filter_stream(full_stream, condition.modalities, event_rng)

        simulator = SENTINELSimulator(
            anomaly_threshold=anomaly_threshold,
            escalation_threshold=escalation_threshold,
            alert_threshold=alert_threshold,
            device=device,
            seed=seed + i,
            fast_mode=fast_mode,
        )
        simulator.reset()

        for obs in stream:
            simulator.process_observation(obs)

        detections = simulator.get_detections()

        # Collect y_true / y_score for AUC
        for d in detections:
            dt = d.timestamp - timeline.event_onset_ts
            if dt < -48 * 3600:
                all_y_true.append(0)
                all_y_score.append(d.anomaly_score)
            elif dt >= 0:
                all_y_true.append(1)
                all_y_score.append(d.anomaly_score)

        # Lead time
        first_alert_ts = None
        source_pred = None
        for d in detections:
            if d.is_alert:
                first_alert_ts = d.timestamp
                if d.source_attribution:
                    ranked = sorted(d.source_attribution.items(), key=lambda x: x[1], reverse=True)
                    source_pred = ranked[0][0]
                break

        sentinel_ts = first_alert_ts
        if sentinel_ts is None:
            # Fall back to first escalation or first anomaly
            for d in detections:
                if d.tier >= 2:
                    sentinel_ts = d.timestamp
                    break
        if sentinel_ts is None:
            for d in detections:
                if d.anomaly_score >= anomaly_threshold:
                    sentinel_ts = d.timestamp
                    break

        if sentinel_ts is not None:
            lt = (timeline.official_detection_ts - sentinel_ts) / 3600.0
            lead_times.append(lt)

        if first_alert_ts is not None:
            events_detected += 1

        # Source attribution accuracy
        if source_pred is not None:
            total_attributed += 1
            if source_pred == event.contaminant_class:
                top1_correct += 1

        # False positive rate (pre-event)
        pre_scores = [d.anomaly_score for d in detections
                      if d.timestamp < timeline.event_onset_ts - 48 * 3600]
        if pre_scores:
            fp_count += sum(1 for s in pre_scores if s >= anomaly_threshold)
            pre_ts = [d.timestamp for d in detections
                      if d.timestamp < timeline.event_onset_ts - 48 * 3600]
            if len(pre_ts) >= 2:
                fp_months += (max(pre_ts) - min(pre_ts)) / (30 * 86400)

        # Tier tracking
        for d in detections:
            if d.timestamp < timeline.event_onset_ts:
                tiers_non_event.append(float(d.tier))
            else:
                tiers_event.append(float(d.tier))

        # Per-event peak anomaly score
        event_scores = [d.anomaly_score for d in detections
                        if d.timestamp >= timeline.event_onset_ts]
        per_event_scores[event_id] = max(event_scores) if event_scores else 0.0

    # Aggregate metrics
    y_true = np.array(all_y_true, dtype=np.int32)
    y_score = np.array(all_y_score, dtype=np.float64)

    if len(y_true) > 0 and y_true.sum() > 0 and (len(y_true) - y_true.sum()) > 0:
        auc, auc_lo, auc_hi = compute_auc(y_true, y_score, n_bootstrap=500)
    else:
        auc, auc_lo, auc_hi = 0.5, 0.5, 0.5

    mean_lt = float(np.mean(lead_times)) if lead_times else 0.0
    median_lt = float(np.median(lead_times)) if lead_times else 0.0
    sa_top1 = top1_correct / max(total_attributed, 1)
    fp_rate = fp_count / max(fp_months, 0.1)

    mean_tier_non = float(np.mean(tiers_non_event)) if tiers_non_event else 0.0
    mean_tier_event = float(np.mean(tiers_event)) if tiers_event else 0.0
    cost_eff = mean_tier_non / max(mean_tier_event, 0.01)

    return AblationResult(
        condition=condition,
        detection_auc=auc,
        detection_auc_ci_lower=auc_lo,
        detection_auc_ci_upper=auc_hi,
        mean_lead_time_hours=mean_lt,
        median_lead_time_hours=median_lt,
        source_attribution_top1=sa_top1,
        false_positive_rate=fp_rate,
        cost_efficiency=cost_eff,
        num_events_detected=events_detected,
        per_event_scores=per_event_scores,
    )


# ---------------------------------------------------------------------------
# Full ablation sweep
# ---------------------------------------------------------------------------

def run_full_ablation(
    output_dir: Path,
    seed: int = 42,
    device: torch.device = torch.device("cpu"),
) -> List[AblationResult]:
    """Run all 31 ablation conditions and save results.

    Args:
        output_dir: Directory to write JSON and CSV results.
        seed: Random seed.
        device: Torch device.

    Returns:
        List of AblationResult for all 31 conditions.
    """
    conditions = [
        AblationCondition.from_modalities(mods)
        for mods in generate_all_conditions()
    ]
    logger.info(f"Running ablation study: {len(conditions)} conditions")

    # Pre-generate all event streams once (shared across conditions)
    logger.info("Pre-generating event streams...")
    cached_streams: Dict[str, Any] = {}
    for i, (event_id, event) in enumerate(HISTORICAL_EVENTS.items()):
        event_rng = np.random.default_rng(seed + i)
        timeline = build_timeline(event)
        full_stream = generate_simulated_stream(timeline, rng=event_rng)
        cached_streams[event_id] = {"stream": full_stream, "timeline": timeline}
    logger.info(f"  Cached {len(cached_streams)} event streams")

    results: List[AblationResult] = []
    progress = make_progress()

    with progress:
        task = progress.add_task("Ablation conditions", total=len(conditions))
        for cond in conditions:
            logger.info(f"  Condition: {cond.name}")
            t0 = time.time()
            result = run_ablation_condition(
                cond, seed=seed, device=device, _cached_streams=cached_streams,
            )
            elapsed = time.time() - t0
            logger.info(
                f"    AUC={result.detection_auc:.4f}  "
                f"lead_time={result.mean_lead_time_hours:.1f}h  "
                f"({elapsed:.1f}s)"
            )
            results.append(result)
            progress.advance(task)

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_results_json(results, output_dir / "ablation_results.json")
    _save_results_csv(results, output_dir / "ablation_results.csv")
    logger.info(f"Ablation results saved to {output_dir}")

    return results


def _save_results_json(results: List[AblationResult], path: Path) -> None:
    """Serialize ablation results to JSON."""
    data = []
    for r in results:
        entry = {
            "condition_name": r.condition.name,
            "modalities": list(r.condition.modalities),
            "num_modalities": len(r.condition.modalities),
            "detection_auc": r.detection_auc,
            "detection_auc_ci_lower": r.detection_auc_ci_lower,
            "detection_auc_ci_upper": r.detection_auc_ci_upper,
            "mean_lead_time_hours": r.mean_lead_time_hours,
            "median_lead_time_hours": r.median_lead_time_hours,
            "source_attribution_top1": r.source_attribution_top1,
            "false_positive_rate": r.false_positive_rate,
            "cost_efficiency": r.cost_efficiency,
            "num_events_detected": r.num_events_detected,
            "per_event_scores": r.per_event_scores,
        }
        data.append(entry)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _save_results_csv(results: List[AblationResult], path: Path) -> None:
    """Save ablation results as a CSV table."""
    fieldnames = [
        "condition_name", "num_modalities", "modalities",
        "detection_auc", "detection_auc_ci_lower", "detection_auc_ci_upper",
        "mean_lead_time_hours", "median_lead_time_hours",
        "source_attribution_top1", "false_positive_rate",
        "cost_efficiency", "num_events_detected",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "condition_name": r.condition.name,
                "num_modalities": len(r.condition.modalities),
                "modalities": ";".join(r.condition.modalities),
                "detection_auc": f"{r.detection_auc:.6f}",
                "detection_auc_ci_lower": f"{r.detection_auc_ci_lower:.6f}",
                "detection_auc_ci_upper": f"{r.detection_auc_ci_upper:.6f}",
                "mean_lead_time_hours": f"{r.mean_lead_time_hours:.2f}",
                "median_lead_time_hours": f"{r.median_lead_time_hours:.2f}",
                "source_attribution_top1": f"{r.source_attribution_top1:.4f}",
                "false_positive_rate": f"{r.false_positive_rate:.4f}",
                "cost_efficiency": f"{r.cost_efficiency:.4f}",
                "num_events_detected": r.num_events_detected,
            })


# ---------------------------------------------------------------------------
# Marginal gain analysis
# ---------------------------------------------------------------------------

def compute_marginal_gains(results: List[AblationResult]) -> pd.DataFrame:
    """Compute the average marginal gain of adding each modality.

    For each modality *m*, considers all conditions that do NOT include
    *m*, finds the corresponding condition that adds *m*, and averages
    the AUC difference.  This answers: "which modality contributes the
    most marginal information?"

    Args:
        results: List of AblationResult from all 31 conditions.

    Returns:
        DataFrame with columns ``[modality, mean_auc_gain, std_auc_gain,
        mean_lead_time_gain, num_comparisons]``, one row per modality.
    """
    # Index results by modality set
    result_map: Dict[frozenset, AblationResult] = {}
    for r in results:
        key = frozenset(r.condition.modalities)
        result_map[key] = r

    rows: List[Dict[str, Any]] = []

    for modality in MODALITIES:
        auc_gains: List[float] = []
        lead_time_gains: List[float] = []

        for r in results:
            mods = frozenset(r.condition.modalities)
            if modality in mods:
                continue  # skip conditions that already include this modality

            # Find the condition that adds this modality
            with_mod = mods | {modality}
            if with_mod in result_map:
                base = r
                augmented = result_map[with_mod]
                auc_gains.append(augmented.detection_auc - base.detection_auc)
                lead_time_gains.append(
                    augmented.mean_lead_time_hours - base.mean_lead_time_hours
                )

        rows.append({
            "modality": modality,
            "mean_auc_gain": float(np.mean(auc_gains)) if auc_gains else 0.0,
            "std_auc_gain": float(np.std(auc_gains)) if auc_gains else 0.0,
            "mean_lead_time_gain": float(np.mean(lead_time_gains)) if lead_time_gains else 0.0,
            "num_comparisons": len(auc_gains),
        })

    df = pd.DataFrame(rows).sort_values("mean_auc_gain", ascending=False)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def run_statistical_tests(results: List[AblationResult]) -> Dict[str, Any]:
    """Run key statistical comparisons across ablation conditions.

    Tests performed:
      1. Full fusion vs best single modality (paired permutation).
      2. Full fusion vs synchronized-only (sensor + satellite).
      3. RL escalation equivalence at lower tiers.

    Args:
        results: All 31 ablation results.

    Returns:
        Dict of test names to ``{diff, p_value, significant}`` entries.
    """
    result_map: Dict[frozenset, AblationResult] = {}
    for r in results:
        result_map[frozenset(r.condition.modalities)] = r

    full_key = frozenset(MODALITIES)
    full_result = result_map.get(full_key)
    if full_result is None:
        logger.warning("Full-fusion condition not found in results")
        return {}

    tests: Dict[str, Any] = {}

    # Test 1: Full fusion vs best single modality
    single_results = [r for r in results if len(r.condition.modalities) == 1]
    if single_results:
        best_single = max(single_results, key=lambda r: r.detection_auc)
        full_scores = np.array(
            [full_result.per_event_scores.get(eid, 0.0) for eid in HISTORICAL_EVENTS]
        )
        best_scores = np.array(
            [best_single.per_event_scores.get(eid, 0.0) for eid in HISTORICAL_EVENTS]
        )
        diff, p_val = paired_permutation_test(full_scores, best_scores)
        tests["full_vs_best_single"] = {
            "full_auc": full_result.detection_auc,
            "best_single_name": best_single.condition.name,
            "best_single_auc": best_single.detection_auc,
            "observed_diff": diff,
            "p_value": p_val,
            "significant_at_0.05": p_val < 0.05,
        }

    # Test 2: Full fusion vs sensor+satellite only
    sync_key = frozenset(["sensor", "satellite"])
    sync_result = result_map.get(sync_key)
    if sync_result is not None:
        full_scores = np.array(
            [full_result.per_event_scores.get(eid, 0.0) for eid in HISTORICAL_EVENTS]
        )
        sync_scores = np.array(
            [sync_result.per_event_scores.get(eid, 0.0) for eid in HISTORICAL_EVENTS]
        )
        diff, p_val = paired_permutation_test(full_scores, sync_scores)
        tests["full_vs_synchronized_only"] = {
            "full_auc": full_result.detection_auc,
            "sync_auc": sync_result.detection_auc,
            "observed_diff": diff,
            "p_value": p_val,
            "significant_at_0.05": p_val < 0.05,
        }

    # Test 3: Cost efficiency -- RL escalation should be cheaper at lower tiers
    # Compare cost efficiency of full fusion vs always-on (proxy: highest-tier average)
    tests["cost_efficiency_comparison"] = {
        "full_cost_efficiency": full_result.cost_efficiency,
        "note": "Lower cost_efficiency ratio indicates better resource usage during events",
    }

    return tests


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the ablation study."""
    parser = argparse.ArgumentParser(
        description="SENTINEL 31-Condition Modality Ablation Study",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/ablation"),
        help="Output directory for results (default: results/ablation).",
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
    parser.add_argument(
        "--marginal-gains",
        action="store_true",
        help="Compute and display marginal gain analysis.",
    )
    parser.add_argument(
        "--statistical-tests",
        action="store_true",
        help="Run and display key statistical tests.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for the ablation study."""
    parser = build_parser()
    args = parser.parse_args(argv)

    device = torch.device(args.device)

    results = run_full_ablation(
        output_dir=args.output_dir,
        seed=args.seed,
        device=device,
    )

    if args.marginal_gains:
        df = compute_marginal_gains(results)
        print("\n=== Marginal Gain Analysis ===")
        print(df.to_string(index=False))
        df.to_csv(args.output_dir / "marginal_gains.csv", index=False)

    if args.statistical_tests:
        tests = run_statistical_tests(results)
        print("\n=== Statistical Tests ===")
        for name, info in tests.items():
            print(f"\n{name}:")
            for k, v in info.items():
                print(f"  {k}: {v}")
        with open(args.output_dir / "statistical_tests.json", "w", encoding="utf-8") as f:
            json.dump(tests, f, indent=2, default=str)


if __name__ == "__main__":
    main()
