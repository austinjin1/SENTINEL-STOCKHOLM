#!/usr/bin/env python3
"""Run causal chain discovery on case study simulated streams.

Generates site-level multimodal time series from the 10 historical
case study events, then runs PCMCI-style causal discovery and validates
against 14 known environmental causal chains.

Usage::

    python -m sentinel.evaluation.run_causal_discovery --output-dir results/causal
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from sentinel.evaluation.case_study import (
    HISTORICAL_EVENTS,
    build_timeline,
    generate_simulated_stream,
)
from sentinel.evaluation.causal_chains import (
    CausalChainDiscovery,
    KNOWN_CAUSAL_CHAINS,
)
from sentinel.utils.logging import get_logger

logger = get_logger(__name__)

# Per-modality variable names (simulated)
MODALITY_VARIABLES = {
    "sensor": [
        "dissolved_oxygen", "turbidity", "conductivity",
        "water_temperature", "pH", "total_phosphorus",
    ],
    "satellite": [
        "chlorophyll_a", "secchi_depth", "phycocyanin", "TSS",
    ],
    "microbial": [
        "alpha_diversity", "anaerobe_fraction", "community_turnover_rate",
    ],
    "molecular": [
        "metallothionein_expression", "cyp1a_expression",
        "microcystin_concentration", "acetylcholinesterase_activity",
    ],
    "behavioral": [
        "activity_index", "avoidance_index",
    ],
}


def generate_site_data(event_id: str, seed: int = 42) -> Dict[str, Any]:
    """Generate multimodal time series data for a case study event.

    Converts the simulated observation stream into the site data format
    expected by CausalChainDiscovery.prepare_multimodal_timeseries().

    Each modality's embedding is projected into named variables using
    deterministic pseudo-projections.
    """
    event = HISTORICAL_EVENTS[event_id]
    timeline = build_timeline(event)
    rng = np.random.default_rng(seed)
    stream = generate_simulated_stream(timeline, rng=rng)

    site_data: Dict[str, Any] = {}

    for obs in stream:
        mod = obs.modality
        if mod not in site_data:
            var_names = MODALITY_VARIABLES.get(mod, ["var_0"])
            site_data[mod] = {
                "timestamps": [],
                "variables": {v: [] for v in var_names},
            }

        site_data[mod]["timestamps"].append(obs.timestamp)

        # Project embedding into named variables using deterministic directions
        var_names = list(site_data[mod]["variables"].keys())
        for k, var_name in enumerate(var_names):
            # Deterministic pseudo-random projection
            proj_rng = np.random.default_rng(hash(f"{mod}/{var_name}") % (2**31))
            direction = proj_rng.standard_normal(len(obs.embedding)).astype(np.float32)
            direction /= np.linalg.norm(direction) + 1e-8
            value = float(np.dot(obs.embedding, direction))

            # Add anomaly-correlated signal for environmental realism
            if obs.anomaly_score > 0.1:
                value += obs.anomaly_score * (1.0 if k % 2 == 0 else -0.5)

            site_data[mod]["variables"][var_name].append(value)

    return site_data


def run_causal_discovery(
    max_lag: int = 168,
    significance: float = 0.05,
    output_dir: Path | None = None,
    seed: int = 42,
) -> Dict:
    """Run causal discovery on all 10 case study events."""
    discovery = CausalChainDiscovery(
        max_lag_hours=max_lag,
        significance_level=significance,
        min_observations=30,
    )

    all_chains: Dict[str, list] = {}
    all_results = {
        "n_events": len(HISTORICAL_EVENTS),
        "max_lag_hours": max_lag,
        "significance_level": significance,
        "n_known_chains": len(KNOWN_CAUSAL_CHAINS),
        "per_event": {},
    }

    for i, (event_id, event) in enumerate(HISTORICAL_EVENTS.items()):
        logger.info(f"\n{'='*60}")
        logger.info(f"Event: {event.name} ({event.year})")
        logger.info(f"{'='*60}")

        # Generate site data
        site_data = generate_site_data(event_id, seed=seed + i)

        logger.info(f"  Modalities: {list(site_data.keys())}")
        for mod, data in site_data.items():
            n_obs = len(data["timestamps"])
            n_vars = len(data["variables"])
            logger.info(f"    {mod}: {n_obs} observations, {n_vars} variables")

        # Prepare and align time series
        aligned = discovery.prepare_multimodal_timeseries(site_data)

        if len(aligned) < 2:
            logger.warning(f"  Insufficient variables for {event_id}, skipping")
            continue

        # Run causal discovery
        chains = discovery.discover_chains(aligned)
        all_chains[event_id] = chains

        n_validated = sum(1 for c in chains if c.validated)
        event_result = {
            "event_name": event.name,
            "n_chains": len(chains),
            "n_validated": n_validated,
            "chains": [asdict(c) for c in chains[:20]],  # top 20
        }
        all_results["per_event"][event_id] = event_result

        logger.info(f"  Found {len(chains)} chains, {n_validated} validated")

        # Save per-event results
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            event_path = output_dir / f"{event_id}_chains.json"
            with open(event_path, "w", encoding="utf-8") as f:
                json.dump([asdict(c) for c in chains], f, indent=2)

    # Aggregate
    if all_chains:
        aggregated = discovery.aggregate_across_sites(all_chains)
        all_results["aggregated"] = aggregated

        logger.info(f"\n{'='*60}")
        logger.info("AGGREGATED RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"Total chains: {aggregated['total_chains_discovered']}")
        logger.info(f"Validated: {aggregated['total_validated']} "
                    f"({aggregated['validation_rate']:.1%})")
        if aggregated.get("novel_chains"):
            logger.info(f"Novel (unvalidated but frequent): {len(aggregated['novel_chains'])}")

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "causal_discovery_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, default=str)
        logger.info(f"\nResults saved to {out_path}")

    return all_results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SENTINEL Causal Chain Discovery (from case study streams)"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/causal"))
    parser.add_argument("--max-lag", type=int, default=168)
    parser.add_argument("--significance", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_causal_discovery(
        max_lag=args.max_lag,
        significance=args.significance,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
