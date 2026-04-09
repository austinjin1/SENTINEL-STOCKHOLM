#!/usr/bin/env python3
"""Recalibrate conformal anomaly detection on real encoder embeddings.

Uses real embeddings from extract_real_embeddings.py instead of
synthetic ones, giving meaningful coverage guarantees.

Usage::

    python scripts/conformal_real_eval.py
"""

import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sentinel.models.theory.conformal import (
    ConformalAnomalyDetector,
    MultimodalConformalEnsemble,
    SpaceType,
)
from sentinel.utils.logging import get_logger

logger = get_logger(__name__)

EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "real_embeddings"
OUTPUT_DIR = PROJECT_ROOT / "results" / "conformal"

MODALITY_SPACE_TYPES = {
    "satellite": SpaceType.IMAGE_FEATURE,
    "sensor": SpaceType.EUCLIDEAN,
    "microbial": SpaceType.SIMPLEX,
    "molecular": SpaceType.EUCLIDEAN,
    "behavioral": SpaceType.EUCLIDEAN,
}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load all available real embeddings
    available = {}
    for mod in ["satellite", "sensor", "microbial", "molecular", "behavioral"]:
        path = EMBEDDINGS_DIR / f"{mod}_embeddings.pt"
        if path.exists():
            emb = torch.load(path, weights_only=True)
            available[mod] = emb
            logger.info(f"Loaded {mod}: {emb.shape}")
        else:
            logger.warning(f"No embeddings for {mod} at {path}")

    if not available:
        logger.error("No embeddings found. Run extract_real_embeddings.py first.")
        return

    alpha_levels = [0.05, 0.10]
    results = {"source": "real_embeddings", "alpha_levels": alpha_levels, "per_alpha": {}}

    for alpha in alpha_levels:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating at alpha = {alpha}")
        logger.info(f"{'='*60}")

        alpha_results = {"modalities": {}}
        all_correct = 0
        all_total = 0

        for mod, emb in available.items():
            n = emb.size(0)
            if n < 20:
                logger.warning(f"  {mod}: too few samples ({n}), skipping")
                continue

            # Split 70/30
            perm = torch.randperm(n, generator=torch.Generator().manual_seed(42))
            n_cal = int(0.7 * n)
            cal_data = emb[perm[:n_cal]]
            test_data = emb[perm[n_cal:]]
            n_test = test_data.size(0)

            space_type = MODALITY_SPACE_TYPES.get(mod, SpaceType.EUCLIDEAN)

            # For simplex: softmax to ensure positive values summing to ~1
            if space_type == SpaceType.SIMPLEX:
                cal_data = torch.softmax(cal_data, dim=-1)
                test_data = torch.softmax(test_data, dim=-1)

            detector = ConformalAnomalyDetector(alpha=alpha)
            threshold = detector.calibrate(cal_data, alpha=alpha, space_type=space_type)

            # Evaluate coverage on test set (should NOT flag normal data)
            is_anom, pvals = detector(test_data)
            n_correct = int((~is_anom).sum().item())
            coverage = n_correct / max(n_test, 1)
            all_correct += n_correct
            all_total += n_test

            mod_result = {
                "n_calibration": n_cal,
                "n_test": n_test,
                "threshold": threshold,
                "empirical_coverage": coverage,
                "target_coverage": 1 - alpha,
                "coverage_met": coverage >= (1 - alpha - 0.01),
                "mean_pvalue": float(pvals.mean().item()),
                "median_pvalue": float(pvals.median().item()),
            }
            alpha_results["modalities"][mod] = mod_result

            status = "MET" if mod_result["coverage_met"] else "MISSED"
            logger.info(
                f"  {mod:>12s}: coverage={coverage:.4f} "
                f"(target={1-alpha:.2f}) [{status}] "
                f"n_cal={n_cal}, n_test={n_test}"
            )

        overall_coverage = all_correct / max(all_total, 1)
        alpha_results["overall_coverage"] = overall_coverage
        alpha_results["coverage_guarantee_met"] = overall_coverage >= (1 - alpha)
        results["per_alpha"][str(alpha)] = alpha_results

        logger.info(f"\n  Overall coverage: {overall_coverage:.4f} (target: {1-alpha:.2f})")
        logger.info(f"  Coverage met: {alpha_results['coverage_guarantee_met']}")

    # Multimodal ensemble
    modalities = sorted(available.keys())
    if len(modalities) >= 2:
        logger.info(f"\n{'='*60}")
        logger.info("Multimodal Conformal Ensemble (Bonferroni)")
        logger.info(f"{'='*60}")

        ensemble = MultimodalConformalEnsemble(
            modality_names=modalities, alpha=0.05, correction="bonferroni"
        )
        cal_dict = {}
        space_dict = {}
        for mod in modalities:
            emb = available[mod]
            n = emb.size(0)
            perm = torch.randperm(n, generator=torch.Generator().manual_seed(42))
            cal = emb[perm[:int(0.7 * n)]]
            st = MODALITY_SPACE_TYPES.get(mod, SpaceType.EUCLIDEAN)
            if st == SpaceType.SIMPLEX:
                cal = torch.softmax(cal, dim=-1)
            cal_dict[mod] = cal
            space_dict[mod] = st

        ensemble.calibrate_all(cal_dict, space_dict)
        results["ensemble"] = {"modalities": modalities, "correction": "bonferroni", "alpha": 0.05}
        logger.info("  Ensemble calibrated successfully")

    # Save
    out_path = OUTPUT_DIR / "real_conformal_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
