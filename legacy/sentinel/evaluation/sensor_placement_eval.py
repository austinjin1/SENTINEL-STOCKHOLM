#!/usr/bin/env python3
"""Sensor placement optimization evaluation.

Demonstrates information-theoretic sensor network design using
submodular optimization on real EPA WQP station locations.

Usage::

    python -m sentinel.evaluation.sensor_placement_eval --output-dir results/sensor_placement
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

from sentinel.models.theory.sensor_placement import (
    CandidateSensor,
    SubmodularObjective,
    optimize_placement,
)
from sentinel.utils.logging import get_logger

logger = get_logger(__name__)

# Modality costs (relative, arbitrary units — e.g., thousands USD per year)
MODALITY_COSTS: Dict[str, float] = {
    "sensor": 5.0,       # IoT sonde: cheap, continuous
    "satellite": 0.5,    # Free Sentinel-2 data, just processing cost
    "microbial": 15.0,   # eDNA sampling + sequencing
    "molecular": 25.0,   # Transcriptomic/biomarker assays
    "behavioral": 10.0,  # Biomonitor (Daphnia/mussel chamber)
}

MODALITIES = list(MODALITY_COSTS.keys())

# Representative US watershed monitoring stations (synthetic candidates
# based on real USGS HUC2 basin centroids and major water bodies)
CANDIDATE_STATIONS = [
    # (id, lat, lon, name)
    (1, 42.45, -72.60, "Connecticut River, MA"),
    (2, 41.18, -73.20, "Housatonic River, CT"),
    (3, 40.73, -74.17, "Passaic River, NJ"),
    (4, 39.96, -75.15, "Schuylkill River, PA"),
    (5, 38.90, -77.04, "Potomac River, DC"),
    (6, 37.53, -77.43, "James River, VA"),
    (7, 35.77, -78.64, "Neuse River, NC"),
    (8, 33.95, -83.37, "Oconee River, GA"),
    (9, 30.33, -81.66, "St. Johns River, FL"),
    (10, 35.05, -85.31, "Tennessee River, TN"),
    (11, 38.25, -85.76, "Ohio River, KY"),
    (12, 41.50, -81.69, "Cuyahoga River, OH"),
    (13, 43.05, -87.92, "Milwaukee River, WI"),
    (14, 41.88, -87.64, "Chicago River, IL"),
    (15, 38.63, -90.20, "Mississippi River, MO"),
    (16, 29.95, -90.07, "Mississippi Delta, LA"),
    (17, 32.32, -90.18, "Pearl River, MS"),
    (18, 30.69, -88.04, "Mobile River, AL"),
    (19, 29.76, -95.36, "Buffalo Bayou, TX"),
    (20, 35.47, -97.52, "Oklahoma River, OK"),
    (21, 39.10, -94.58, "Missouri River, KS"),
    (22, 41.26, -95.94, "Missouri River, NE"),
    (23, 44.98, -93.27, "Mississippi River, MN"),
    (24, 46.88, -96.79, "Red River, ND"),
    (25, 40.82, -96.70, "Platte River, NE"),
    (26, 39.74, -104.99, "South Platte River, CO"),
    (27, 40.76, -111.89, "Jordan River, UT"),
    (28, 43.61, -116.21, "Boise River, ID"),
    (29, 45.52, -122.68, "Willamette River, OR"),
    (30, 47.61, -122.34, "Duwamish River, WA"),
]


def build_candidates() -> List[CandidateSensor]:
    """Build candidate sensors: each station × each modality = 150 candidates."""
    candidates = []
    for station_id, lat, lon, name in CANDIDATE_STATIONS:
        for modality in MODALITIES:
            candidates.append(CandidateSensor(
                location_id=station_id * 100 + MODALITIES.index(modality),
                modality=modality,
                cost=MODALITY_COSTS[modality],
                latitude=lat,
                longitude=lon,
            ))
    return candidates


def build_features(candidates: List[CandidateSensor]) -> torch.Tensor:
    """Build feature matrix from candidate locations.

    Features: [lat_norm, lon_norm, modality_onehot (5), cos_lat, sin_lon]
    """
    n = len(candidates)
    features = torch.zeros(n, 9)  # 2 + 5 + 2
    for i, c in enumerate(candidates):
        features[i, 0] = (c.latitude - 38.0) / 10.0  # normalize
        features[i, 1] = (c.longitude + 95.0) / 20.0  # normalize
        mod_idx = MODALITIES.index(c.modality)
        features[i, 2 + mod_idx] = 1.0  # one-hot
        features[i, 7] = np.cos(np.radians(c.latitude))
        features[i, 8] = np.sin(np.radians(c.longitude))
    return features


def run_sensor_placement(
    budget_levels: List[float] = [50, 100, 200, 500],
    output_dir: Path | None = None,
) -> Dict:
    """Run sensor placement optimization at multiple budget levels.

    Args:
        budget_levels: Budget levels to evaluate.
        output_dir: Output directory for results.

    Returns:
        Results dict.
    """
    candidates = build_candidates()
    features = build_features(candidates)

    logger.info(f"Built {len(candidates)} candidate sensors "
                f"({len(CANDIDATE_STATIONS)} stations × {len(MODALITIES)} modalities)")
    logger.info(f"Feature dim: {features.size(1)}")

    objective = SubmodularObjective(
        n_candidates=len(candidates),
        feature_dim=features.size(1),
        kernel_type="rbf",
        noise_variance=0.1,
        length_scale=1.0,
    )

    results = {
        "n_candidates": len(candidates),
        "n_stations": len(CANDIDATE_STATIONS),
        "n_modalities": len(MODALITIES),
        "modality_costs": MODALITY_COSTS,
        "budget_levels": {},
    }

    for budget in budget_levels:
        logger.info(f"\n{'='*60}")
        logger.info(f"Optimizing for budget = {budget}")
        logger.info(f"{'='*60}")

        placements = optimize_placement(
            features=features,
            adjacency=None,
            candidates=candidates,
            budget=budget,
            modality_costs=MODALITY_COSTS,
            objective=objective,
        )

        # Summarize
        modality_counts = {}
        modality_spend = {}
        for p in placements:
            modality_counts[p.modality] = modality_counts.get(p.modality, 0) + 1
            modality_spend[p.modality] = modality_spend.get(p.modality, 0) + p.cost

        total_gain = placements[-1].cumulative_gain if placements else 0.0
        total_cost = sum(p.cost for p in placements)

        budget_result = {
            "n_sensors": len(placements),
            "total_gain": total_gain,
            "total_cost": total_cost,
            "cost_efficiency": total_gain / max(total_cost, 1e-8),
            "modality_counts": modality_counts,
            "modality_spend": modality_spend,
            "placements": [
                {
                    "location_id": p.location_id,
                    "modality": p.modality,
                    "cost": p.cost,
                    "marginal_gain": p.marginal_gain,
                    "cumulative_gain": p.cumulative_gain,
                }
                for p in placements
            ],
            "marginal_gain_curve": [p.marginal_gain for p in placements],
        }

        results["budget_levels"][str(budget)] = budget_result

        logger.info(f"\n  Budget={budget}: {len(placements)} sensors, "
                    f"total_gain={total_gain:.4f}")
        logger.info(f"  Modality allocation: {modality_counts}")
        logger.info(f"  Cost efficiency: {budget_result['cost_efficiency']:.4f} gain/cost")

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "sensor_placement_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"\nResults saved to {out_path}")

    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sensor placement optimization evaluation"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/sensor_placement"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_sensor_placement(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
