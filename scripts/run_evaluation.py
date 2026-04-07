#!/usr/bin/env python3
"""SENTINEL comprehensive evaluation suite runner.

Executes the full evaluation pipeline:
  1. 31-condition modality ablation study (all 2^5-1 subsets)
  2. Missing-modality robustness analysis (100 random-drop trials)
  3. Cross-modal mutual information analysis (MINE-based)
  4. Publication-quality figure generation (10 figures)

All evaluation uses simulated observation streams (SENTINELSimulator)
and runs on CPU -- no GPU or model checkpoints required.

Usage::

    cd /home/bcheng/SENTINEL && python scripts/run_evaluation.py
"""

from __future__ import annotations

import json
import sys
import time
import types
from pathlib import Path

# Workaround: stable_baselines3 -> torch.utils.tensorboard tries to
# import tensorflow, which fails in this environment.  Mock the module
# so the import chain completes without error.  Also set __spec__ to
# avoid torch._dynamo.trace_rules errors.
if "tensorflow" not in sys.modules:
    import importlib
    _tf = types.ModuleType("tensorflow")
    _tf_io = types.ModuleType("tensorflow.io")
    _tf_gfile = types.ModuleType("tensorflow.io.gfile")
    _tf.io = _tf_io
    _tf_io.gfile = _tf_gfile
    # Set __spec__ so torch._dynamo doesn't crash
    _tf.__spec__ = importlib.machinery.ModuleSpec("tensorflow", None)
    _tf_io.__spec__ = importlib.machinery.ModuleSpec("tensorflow.io", None)
    _tf_gfile.__spec__ = importlib.machinery.ModuleSpec("tensorflow.io.gfile", None)
    sys.modules["tensorflow"] = _tf
    sys.modules["tensorflow.io"] = _tf_io
    sys.modules["tensorflow.io.gfile"] = _tf_gfile

import torch

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sentinel.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------

ABLATION_DIR = PROJECT_ROOT / "results" / "ablation"
ROBUSTNESS_DIR = PROJECT_ROOT / "results" / "robustness"
INFORMATION_DIR = PROJECT_ROOT / "results" / "information"
FIGURES_DIR = PROJECT_ROOT / "figures"

DEVICE = torch.device("cpu")
SEED = 42


# ---------------------------------------------------------------------------
# 1. Modality ablation study
# ---------------------------------------------------------------------------

def run_ablation_study() -> None:
    """Run the 31-condition modality ablation study and save results."""
    logger.info("=" * 65)
    logger.info("PHASE 1: 31-Condition Modality Ablation Study")
    logger.info("=" * 65)

    from sentinel.evaluation.ablation import (
        run_full_ablation,
        compute_marginal_gains,
        run_statistical_tests,
    )

    t0 = time.time()

    # Run all 31 conditions
    results = run_full_ablation(
        output_dir=ABLATION_DIR,
        seed=SEED,
        device=DEVICE,
    )

    # Compute marginal gains
    marginal_df = compute_marginal_gains(results)
    marginal_df.to_csv(ABLATION_DIR / "marginal_gains.csv", index=False)
    logger.info("Marginal gains saved to results/ablation/marginal_gains.csv")

    # Run statistical tests
    stat_tests = run_statistical_tests(results)
    with open(ABLATION_DIR / "statistical_tests.json", "w", encoding="utf-8") as f:
        json.dump(stat_tests, f, indent=2, default=str)
    logger.info("Statistical tests saved to results/ablation/statistical_tests.json")

    # Also save ablation results in the dict-of-dicts format that
    # figure_ablation_bar_chart expects
    ablation_chart_data = {}
    for r in results:
        ablation_chart_data[r.condition.name] = {
            "auc": r.detection_auc,
            "n_modalities": len(r.condition.modalities),
            "ci_lower": r.detection_auc_ci_lower,
            "ci_upper": r.detection_auc_ci_upper,
        }
    with open(ABLATION_DIR / "ablation_chart_data.json", "w", encoding="utf-8") as f:
        json.dump(ablation_chart_data, f, indent=2)

    elapsed = time.time() - t0
    logger.info(f"Ablation study completed in {elapsed:.1f}s")
    logger.info(f"  Conditions evaluated: {len(results)}")
    full_result = next(
        (r for r in results if len(r.condition.modalities) == 5), None
    )
    if full_result:
        logger.info(
            f"  Full fusion AUC: {full_result.detection_auc:.4f} "
            f"[{full_result.detection_auc_ci_lower:.4f}, "
            f"{full_result.detection_auc_ci_upper:.4f}]"
        )

    # Print summary table
    print("\n" + "=" * 75)
    print("ABLATION STUDY SUMMARY")
    print("=" * 75)
    print(f"{'Condition':<45} {'AUC':>8} {'Lead Time (h)':>14} {'Detected':>10}")
    print("-" * 75)
    for r in sorted(results, key=lambda x: x.detection_auc, reverse=True)[:10]:
        print(
            f"{r.condition.name:<45} {r.detection_auc:>8.4f} "
            f"{r.mean_lead_time_hours:>14.1f} {r.num_events_detected:>10}"
        )
    print(f"  ... ({len(results)} total conditions)")

    print("\nMarginal Gain Analysis:")
    print(marginal_df.to_string(index=False))
    print("=" * 75)


# ---------------------------------------------------------------------------
# 2. Missing-modality robustness analysis
# ---------------------------------------------------------------------------

def run_robustness_analysis() -> None:
    """Run the missing-modality robustness analysis (100 trials)."""
    logger.info("=" * 65)
    logger.info("PHASE 2: Missing-Modality Robustness Analysis (100 trials)")
    logger.info("=" * 65)

    from sentinel.evaluation.missing_modality import (
        run_missing_modality_analysis,
        save_robustness_report,
    )

    t0 = time.time()

    report = run_missing_modality_analysis(
        n_trials=100,
        seed=SEED,
        device=DEVICE,
    )

    save_robustness_report(report, ROBUSTNESS_DIR)

    elapsed = time.time() - t0
    logger.info(f"Robustness analysis completed in {elapsed:.1f}s")

    # Print summary
    print("\n" + "=" * 75)
    print("ROBUSTNESS ANALYSIS SUMMARY")
    print("=" * 75)
    print(
        f"{'Modalities Available':<25} {'Mean AUC':>10} {'Std':>8} "
        f"{'95% CI':>20} {'Trials':>8}"
    )
    print("-" * 75)
    for c in report.degradation_curves:
        print(
            f"{c.num_available:<25} {c.mean_auc:>10.4f} {c.std_auc:>8.4f} "
            f"[{c.ci_lower:.4f}, {c.ci_upper:.4f}] {c.num_trials:>8}"
        )
    print(f"\nGraceful Degradation Score: {report.graceful_degradation_score:.4f}")
    print("  (1.0 = no degradation, 0.0 = linear, <0 = catastrophic)")

    print("\nModality Criticality (avg AUC drop when absent):")
    for mod, crit in sorted(
        report.modality_criticality.items(), key=lambda x: x[1], reverse=True
    ):
        print(f"  {mod:<15} {crit:+.4f}")
    print("=" * 75)


# ---------------------------------------------------------------------------
# 3. Cross-modal information analysis
# ---------------------------------------------------------------------------

def run_information_analysis() -> None:
    """Run MINE-based cross-modal mutual information analysis."""
    logger.info("=" * 65)
    logger.info("PHASE 3: Cross-Modal Information Analysis (MINE)")
    logger.info("=" * 65)

    from sentinel.evaluation.information_analysis import (
        extract_embeddings_from_case_studies,
        compute_information_matrix,
        compute_unique_information,
        generate_information_report,
    )
    import numpy as np

    t0 = time.time()

    INFORMATION_DIR.mkdir(parents=True, exist_ok=True)

    # Extract embeddings from simulated case study streams
    logger.info("Extracting embeddings from case study simulations...")
    all_embeddings = extract_embeddings_from_case_studies(seed=SEED)

    # Compute the 5x5 pairwise MI matrix using MINE
    logger.info("Computing MI matrix using MINE estimator...")
    mi_matrix = compute_information_matrix(
        all_embeddings,
        method="mine",
        n_epochs=200,
    )

    # Compute unique information per modality
    logger.info("Estimating unique information per modality...")
    unique_info = compute_unique_information(
        all_embeddings,
        mi_matrix=mi_matrix,
        method="mine",
        n_epochs=200,
    )

    # Generate report
    report = generate_information_report(mi_matrix, unique_info)

    # Save outputs
    with open(INFORMATION_DIR / "information_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    np.save(INFORMATION_DIR / "mi_matrix.npy", mi_matrix)

    elapsed = time.time() - t0
    logger.info(f"Information analysis completed in {elapsed:.1f}s")

    # Print summary
    print("\n" + "=" * 65)
    print("CROSS-MODAL INFORMATION ANALYSIS")
    print("=" * 65)

    s = report["summary"]
    print(f"\nRedundancy ratio:     {s['redundancy_ratio']:.4f}")
    print(f"Complementarity:      {s['complementarity_score']:.4f}")
    print(f"Mean pairwise MI:     {s['mean_pairwise_mi']:.4f} nats")
    print(f"Interpretation:       {s['interpretation']}")

    print(
        f"\nMost redundant pair:  {report['most_redundant_pair']['modalities']} "
        f"(MI={report['most_redundant_pair']['mi']:.4f})"
    )
    print(
        f"Most independent pair: {report['most_independent_pair']['modalities']} "
        f"(MI={report['most_independent_pair']['mi']:.4f})"
    )

    from sentinel.evaluation.ablation import MODALITIES

    print(
        f"\n{'Modality':<15} {'Self-Info':>10} {'Unique':>10} "
        f"{'Mean MI w/ Others':>18}"
    )
    print("-" * 58)
    for pm in report["per_modality"]:
        print(
            f"{pm['modality']:<15} {pm['self_information']:>10.4f} "
            f"{pm['unique_information']:>10.4f} {pm['mean_mi_with_others']:>18.4f}"
        )
    print("=" * 65)


# ---------------------------------------------------------------------------
# 4. Publication-quality figure generation
# ---------------------------------------------------------------------------

def run_figure_generation() -> None:
    """Generate all 10 publication-quality figures."""
    logger.info("=" * 65)
    logger.info("PHASE 4: Publication-Quality Figure Generation")
    logger.info("=" * 65)

    from sentinel.evaluation.figures import generate_all_figures

    t0 = time.time()

    # The figures module looks for ablation_results.json in results_dir.
    # We need to point it at the ablation chart data we saved.
    # Copy ablation chart data to results/ root so figures can find it.
    results_root = PROJECT_ROOT / "results"
    results_root.mkdir(parents=True, exist_ok=True)

    ablation_chart_src = ABLATION_DIR / "ablation_chart_data.json"
    ablation_chart_dst = results_root / "ablation_results.json"
    if ablation_chart_src.exists() and not ablation_chart_dst.exists():
        import shutil
        shutil.copy2(ablation_chart_src, ablation_chart_dst)
        logger.info(f"Copied ablation chart data to {ablation_chart_dst}")

    paths = generate_all_figures(
        results_dir=results_root,
        output_dir=FIGURES_DIR,
    )

    elapsed = time.time() - t0
    logger.info(f"Figure generation completed in {elapsed:.1f}s")

    print("\n" + "=" * 65)
    print("GENERATED FIGURES")
    print("=" * 65)
    for p in paths:
        print(f"  {p}")
    print(f"\nTotal: {len(paths)} figures in {FIGURES_DIR}")
    print("=" * 65)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the complete SENTINEL evaluation suite."""
    import argparse

    parser = argparse.ArgumentParser(description="SENTINEL Evaluation Suite")
    parser.add_argument(
        "--skip-ablation", action="store_true",
        help="Skip ablation study if results already exist.",
    )
    parser.add_argument(
        "--skip-robustness", action="store_true",
        help="Skip robustness analysis if results already exist.",
    )
    parser.add_argument(
        "--skip-information", action="store_true",
        help="Skip information analysis if results already exist.",
    )
    parser.add_argument(
        "--skip-figures", action="store_true",
        help="Skip figure generation.",
    )
    args = parser.parse_args()

    total_t0 = time.time()

    print("\n" + "#" * 70)
    print("#  SENTINEL Comprehensive Evaluation Suite")
    print("#  Running on CPU with simulated data")
    print("#" * 70 + "\n")

    # Phase 1: Ablation study
    ablation_done = (ABLATION_DIR / "ablation_results.json").exists()
    if args.skip_ablation and ablation_done:
        logger.info("Skipping Phase 1 (ablation) -- results already exist")
    else:
        run_ablation_study()

    # Phase 2: Robustness analysis
    robustness_done = (ROBUSTNESS_DIR / "robustness_summary.json").exists()
    if args.skip_robustness and robustness_done:
        logger.info("Skipping Phase 2 (robustness) -- results already exist")
    else:
        run_robustness_analysis()

    # Phase 3: Information analysis
    info_done = (INFORMATION_DIR / "information_report.json").exists()
    if args.skip_information and info_done:
        logger.info("Skipping Phase 3 (information) -- results already exist")
    else:
        run_information_analysis()

    # Phase 4: Figure generation
    if not args.skip_figures:
        run_figure_generation()
    else:
        logger.info("Skipping Phase 4 (figures)")

    total_elapsed = time.time() - total_t0

    print("\n" + "#" * 70)
    print("#  EVALUATION COMPLETE")
    print(f"#  Total time: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")
    print("#")
    print("#  Output locations:")
    print(f"#    Ablation results:   {ABLATION_DIR}")
    print(f"#    Robustness results: {ROBUSTNESS_DIR}")
    print(f"#    Information results:{INFORMATION_DIR}")
    print(f"#    Figures:            {FIGURES_DIR}")
    print("#" * 70)


if __name__ == "__main__":
    main()
