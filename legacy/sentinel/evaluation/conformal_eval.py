#!/usr/bin/env python3
"""Conformal anomaly detection evaluation.

Calibrates and evaluates conformal prediction on case study simulated
streams, demonstrating distribution-free coverage guarantees.

Usage::

    python -m sentinel.evaluation.conformal_eval --output-dir results/conformal
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

from sentinel.evaluation.case_study import (
    HISTORICAL_EVENTS,
    build_timeline,
    generate_simulated_stream,
)
from sentinel.models.fusion.embedding_registry import SHARED_EMBEDDING_DIM
from sentinel.models.theory.conformal import (
    ConformalAnomalyDetector,
    GeometryAwareNonconformityScore,
    MultimodalConformalEnsemble,
    SpaceType,
)
from sentinel.utils.logging import get_logger

logger = get_logger(__name__)

MODALITY_SPACE_TYPES: Dict[str, SpaceType] = {
    "sensor": SpaceType.EUCLIDEAN,
    "satellite": SpaceType.IMAGE_FEATURE,
    "microbial": SpaceType.SIMPLEX,
    "molecular": SpaceType.EUCLIDEAN,
    "behavioral": SpaceType.EUCLIDEAN,
}


def extract_embeddings_from_events(
    seed: int = 42,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Extract per-modality embeddings from all case study simulated streams.

    Returns:
        Dict with keys "normal" and "anomalous", each mapping modality
        name to a tensor of embeddings.
    """
    normal: Dict[str, List[np.ndarray]] = {}
    anomalous: Dict[str, List[np.ndarray]] = {}

    for i, (event_id, event) in enumerate(HISTORICAL_EVENTS.items()):
        timeline = build_timeline(event)
        rng = np.random.default_rng(seed + i)
        stream = generate_simulated_stream(timeline, rng=rng)

        for obs in stream:
            bucket = anomalous if obs.anomaly_score > 0.3 else normal
            bucket.setdefault(obs.modality, []).append(obs.embedding)

    # Convert to tensors
    result: Dict[str, Dict[str, torch.Tensor]] = {"normal": {}, "anomalous": {}}
    for label, data in [("normal", normal), ("anomalous", anomalous)]:
        for mod, embs in data.items():
            result[label][mod] = torch.from_numpy(np.stack(embs))

    return result


def evaluate_conformal(
    alpha_levels: List[float] = [0.05, 0.10],
    seed: int = 42,
    output_dir: Path | None = None,
) -> Dict:
    """Run conformal anomaly detection evaluation.

    1. Extract normal/anomalous embeddings from case study streams.
    2. Split normal data into calibration (70%) and test (30%).
    3. Calibrate per-modality ConformalAnomalyDetectors.
    4. Evaluate coverage on test normal + anomalous data.

    Args:
        alpha_levels: Miscoverage rates to evaluate.
        seed: Random seed.
        output_dir: Directory for results JSON.

    Returns:
        Results dict with coverage rates, detection metrics, etc.
    """
    logger.info("Extracting embeddings from 10 case study events...")
    data = extract_embeddings_from_events(seed=seed)

    results = {"alpha_levels": alpha_levels, "per_alpha": {}, "per_modality": {}}

    for alpha in alpha_levels:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating at alpha = {alpha}")
        logger.info(f"{'='*60}")

        alpha_results: Dict = {"modalities": {}}
        all_normal_correct = 0
        all_normal_total = 0
        all_anomalous_detected = 0
        all_anomalous_total = 0

        for modality in sorted(data["normal"].keys()):
            normal_embs = data["normal"][modality]
            anomalous_embs = data["anomalous"].get(modality)

            if normal_embs.size(0) < 20:
                logger.warning(f"  {modality}: too few normal samples ({normal_embs.size(0)}), skipping")
                continue

            # Split normal: 70% calibration, 30% test
            n = normal_embs.size(0)
            perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
            n_cal = int(0.7 * n)
            cal_data = normal_embs[perm[:n_cal]]
            test_normal = normal_embs[perm[n_cal:]]

            # Calibrate detector
            space_type = MODALITY_SPACE_TYPES.get(modality, SpaceType.EUCLIDEAN)

            # For simplex, ensure data is positive and sums to ~1
            if space_type == SpaceType.SIMPLEX:
                cal_data = torch.softmax(cal_data, dim=-1)
                test_normal = torch.softmax(test_normal, dim=-1)
                if anomalous_embs is not None:
                    anomalous_embs = torch.softmax(anomalous_embs, dim=-1)

            detector = ConformalAnomalyDetector(alpha=alpha)
            threshold = detector.calibrate(cal_data, alpha=alpha, space_type=space_type)

            # Evaluate on test normal (should NOT be flagged as anomalous)
            is_anom_normal, pvals_normal = detector(test_normal)
            n_test = test_normal.size(0)
            n_correct = int((~is_anom_normal).sum().item())
            empirical_coverage = n_correct / max(n_test, 1)

            all_normal_correct += n_correct
            all_normal_total += n_test

            mod_result = {
                "n_calibration": n_cal,
                "n_test_normal": n_test,
                "threshold": threshold,
                "empirical_coverage": empirical_coverage,
                "target_coverage": 1 - alpha,
                "coverage_met": empirical_coverage >= (1 - alpha - 0.02),  # small tolerance
                "mean_pvalue_normal": float(pvals_normal.mean().item()),
            }

            # Evaluate on anomalous data (should be detected)
            if anomalous_embs is not None and anomalous_embs.size(0) > 0:
                is_anom_anom, pvals_anom = detector(anomalous_embs)
                n_anom = anomalous_embs.size(0)
                n_detected = int(is_anom_anom.sum().item())
                detection_rate = n_detected / max(n_anom, 1)

                all_anomalous_detected += n_detected
                all_anomalous_total += n_anom

                mod_result.update({
                    "n_anomalous": n_anom,
                    "anomaly_detection_rate": detection_rate,
                    "mean_pvalue_anomalous": float(pvals_anom.mean().item()),
                })

            alpha_results["modalities"][modality] = mod_result

            logger.info(
                f"  {modality:>12s}: coverage={empirical_coverage:.3f} "
                f"(target={1-alpha:.2f}), "
                f"detection={mod_result.get('anomaly_detection_rate', 'N/A')}"
            )

        # Aggregate
        overall_coverage = all_normal_correct / max(all_normal_total, 1)
        overall_detection = all_anomalous_detected / max(all_anomalous_total, 1)
        alpha_results["overall_coverage"] = overall_coverage
        alpha_results["overall_detection_rate"] = overall_detection
        alpha_results["coverage_guarantee_met"] = overall_coverage >= (1 - alpha)

        logger.info(f"\n  Overall coverage: {overall_coverage:.4f} (target: {1-alpha:.2f})")
        logger.info(f"  Overall detection rate: {overall_detection:.4f}")
        logger.info(f"  Coverage guarantee met: {alpha_results['coverage_guarantee_met']}")

        results["per_alpha"][str(alpha)] = alpha_results

    # Multimodal ensemble evaluation
    logger.info(f"\n{'='*60}")
    logger.info("Multimodal Conformal Ensemble (Benjamini-Hochberg)")
    logger.info(f"{'='*60}")

    modalities = sorted(data["normal"].keys())
    ensemble = MultimodalConformalEnsemble(
        modality_names=modalities, alpha=0.05, correction="bh"
    )

    # Calibrate ensemble
    cal_data_dict: Dict[str, torch.Tensor] = {}
    space_types_dict: Dict[str, SpaceType] = {}
    for mod in modalities:
        embs = data["normal"][mod]
        n = embs.size(0)
        n_cal = int(0.7 * n)
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
        cal_embs = embs[perm[:n_cal]]
        st = MODALITY_SPACE_TYPES.get(mod, SpaceType.EUCLIDEAN)
        if st == SpaceType.SIMPLEX:
            cal_embs = torch.softmax(cal_embs, dim=-1)
        cal_data_dict[mod] = cal_embs
        space_types_dict[mod] = st

    ensemble.calibrate_all(cal_data_dict, space_types_dict)
    results["ensemble"] = {"correction": "benjamini-hochberg", "alpha": 0.05}

    logger.info("  Ensemble calibrated successfully")

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "conformal_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {out_path}")

    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Conformal anomaly detection evaluation"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/conformal"),
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    evaluate_conformal(output_dir=args.output_dir, seed=args.seed)


if __name__ == "__main__":
    main()
