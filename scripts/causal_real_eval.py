#!/usr/bin/env python3
"""Causal discovery on real GRQA environmental data.

Uses temporally-aligned water quality measurements from co-located
monitoring sites to discover cross-parameter causal chains via PCMCI.

Usage::

    python scripts/causal_real_eval.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sentinel.evaluation.causal_chains import (
    CausalChainDiscovery,
    KNOWN_CAUSAL_CHAINS,
)
from sentinel.utils.logging import get_logger

logger = get_logger(__name__)

GRQA_DIR = PROJECT_ROOT / "data" / "sentinel_db" / "grqa"
OUTPUT_DIR = PROJECT_ROOT / "results" / "causal"

# Parameters to load (sensor-observable + satellite-observable)
PARAMS_TO_LOAD = {
    "DO": "dissolved_oxygen",
    "TEMP": "water_temperature",
    "pH": "pH",
    "TSS": "total_suspended_solids",
    "TN": "total_nitrogen",
    "TP": "total_phosphorus",
    "NH4N": "ammonia",
    "NO3N": "nitrate",
    "BOD": "biochemical_oxygen_demand",
    "COD": "chemical_oxygen_demand",
    "DOC": "dissolved_organic_carbon",
}


def load_grqa_params() -> pd.DataFrame:
    """Load and merge multiple GRQA parameter parquets."""
    frames = []
    for param_code, param_name in PARAMS_TO_LOAD.items():
        path = GRQA_DIR / f"{param_code}.parquet"
        if not path.exists():
            logger.warning(f"Missing {path}")
            continue
        df = pd.read_parquet(path, columns=["value", "site_id", "timestamp"])
        df["parameter"] = param_name
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        frames.append(df)
        logger.info(f"  {param_code}: {len(df):,} records")

    all_data = pd.concat(frames, ignore_index=True)
    logger.info(f"Total: {len(all_data):,} records, {all_data['site_id'].nunique()} sites")
    return all_data


def find_rich_sites(data: pd.DataFrame, min_params: int = 5, min_records_per_param: int = 20) -> list:
    """Find sites with sufficient multi-parameter temporal coverage."""
    site_param_counts = data.groupby(["site_id", "parameter"]).size().reset_index(name="count")
    site_param_counts = site_param_counts[site_param_counts["count"] >= min_records_per_param]

    sites_with_enough_params = (
        site_param_counts.groupby("site_id")["parameter"]
        .nunique()
        .reset_index(name="n_params")
    )
    rich_sites = sites_with_enough_params[sites_with_enough_params["n_params"] >= min_params]
    rich_sites = rich_sites.sort_values("n_params", ascending=False)

    logger.info(f"Found {len(rich_sites)} sites with >= {min_params} params "
                f"(>= {min_records_per_param} records each)")
    return rich_sites["site_id"].tolist()


def build_site_timeseries(data: pd.DataFrame, site_id: str) -> dict:
    """Build multivariate time series for one site in causal discovery format."""
    site_data = data[data["site_id"] == site_id].copy()
    site_data = site_data.sort_values("timestamp")

    result = {}
    for param in site_data["parameter"].unique():
        param_data = site_data[site_data["parameter"] == param]
        timestamps = param_data["timestamp"].values.astype(np.int64) / 1e9  # to seconds
        values = param_data["value"].values.astype(np.float64)

        # Map to modality for causal chain validation
        modality = "sensor"  # Most GRQA params are sensor-observable

        result[f"{modality}/{param}"] = {
            "timestamps": timestamps.tolist(),
            "values": values.tolist(),
        }

    return result


def run_causal_on_real_data(max_sites: int = 20, max_lag: int = 168):
    """Run causal discovery on real GRQA environmental data."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading GRQA parameters...")
    data = load_grqa_params()

    logger.info("\nFinding sites with rich multi-parameter coverage...")
    rich_sites = find_rich_sites(data, min_params=5, min_records_per_param=30)

    if not rich_sites:
        logger.error("No sites with sufficient coverage found")
        return

    n_sites = min(len(rich_sites), max_sites)
    logger.info(f"\nAnalyzing top {n_sites} sites...")

    discovery = CausalChainDiscovery(
        max_lag_hours=max_lag,
        significance_level=0.05,
        min_observations=20,
    )

    all_chains = {}
    results = {
        "source": "real_grqa_data",
        "n_sites_analyzed": n_sites,
        "max_lag_hours": max_lag,
        "n_known_chains": len(KNOWN_CAUSAL_CHAINS),
        "per_site": {},
    }

    for i, site_id in enumerate(rich_sites[:n_sites]):
        site_data = data[data["site_id"] == site_id]
        n_params = site_data["parameter"].nunique()
        n_records = len(site_data)

        logger.info(f"\n{'='*60}")
        logger.info(f"Site {i+1}/{n_sites}: {site_id} ({n_params} params, {n_records} records)")
        logger.info(f"{'='*60}")

        # Build site-level multivariate timeseries
        site_ts_raw = build_site_timeseries(data, site_id)

        # Convert to format expected by CausalChainDiscovery
        # Each variable gets its own modality entry to avoid timestamp mismatches
        site_data_dict = {}
        for key, ts_data in site_ts_raw.items():
            modality, variable = key.split("/", 1)
            # Use unique key per variable to avoid shared timestamps
            mod_key = f"{modality}_{variable}"
            site_data_dict[mod_key] = {
                "timestamps": ts_data["timestamps"],
                "variables": {variable: ts_data["values"]},
            }

        # Prepare aligned timeseries
        aligned = discovery.prepare_multimodal_timeseries(site_data_dict)

        if len(aligned) < 2:
            logger.warning(f"  Insufficient aligned variables, skipping")
            continue

        # Run discovery
        chains = discovery.discover_chains(aligned)
        all_chains[site_id] = chains

        n_validated = sum(1 for c in chains if c.validated)
        results["per_site"][site_id] = {
            "n_params": n_params,
            "n_records": n_records,
            "n_chains": len(chains),
            "n_validated": n_validated,
            "top_chains": [
                {
                    "cause": f"{c.cause_modality}/{c.cause_variable}",
                    "effect": f"{c.effect_modality}/{c.effect_variable}",
                    "lag_hours": c.lag_hours,
                    "strength": c.strength,
                    "p_value": c.p_value,
                    "direction": c.direction,
                    "validated": c.validated,
                    "note": c.validation_note,
                }
                for c in chains[:10]
            ],
        }

        logger.info(f"  Found {len(chains)} chains, {n_validated} validated")

    # Aggregate
    if all_chains:
        aggregated = discovery.aggregate_across_sites(all_chains)
        results["aggregated"] = aggregated

        total = aggregated["total_chains_discovered"]
        validated = aggregated["total_validated"]
        rate = aggregated["validation_rate"]
        novel = len(aggregated.get("novel_chains", []))

        logger.info(f"\n{'='*60}")
        logger.info("AGGREGATED RESULTS (REAL DATA)")
        logger.info(f"{'='*60}")
        logger.info(f"Total chains: {total}")
        logger.info(f"Validated: {validated} ({rate:.1%})")
        logger.info(f"Novel: {novel}")

        # Show top validated chains
        for chain_key, info in list(aggregated.get("chain_types", {}).items())[:10]:
            if info.get("validated_fraction", 0) > 0:
                logger.info(f"  VALIDATED: {chain_key} "
                            f"(freq={info['frequency']}, lag={info['mean_lag_hours']:.0f}h, "
                            f"strength={info['mean_strength']:.3f})")

    # Save
    out_path = OUTPUT_DIR / "real_causal_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    run_causal_on_real_data()
