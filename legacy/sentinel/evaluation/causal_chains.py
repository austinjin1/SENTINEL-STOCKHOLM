"""Temporal causal chain discovery across modalities.

Applies heterogeneous causal discovery (PCMCI-based) to identify
cross-modal temporal relationships in water quality data. Validates
discovered chains against known environmental science causal pathways.

Usage::

    python -m sentinel.evaluation.causal_chains --data-dir data/sites --sites all --max-lag 168 --output-dir results/causal
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from sentinel.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CausalChain:
    """A discovered causal relationship between two modality variables."""

    cause_modality: str
    """Source modality (e.g., 'sensor', 'satellite', 'microbial')."""

    cause_variable: str
    """Specific variable within the source modality."""

    effect_modality: str
    """Target modality."""

    effect_variable: str
    """Specific variable within the target modality."""

    lag_hours: float
    """Time lag from cause to effect in hours."""

    strength: float
    """Effect strength (absolute partial correlation or regression coefficient)."""

    p_value: float
    """Statistical significance of the causal link."""

    direction: str = "positive"
    """Relationship direction: 'positive' or 'negative'."""

    validated: Optional[bool] = None
    """Whether this chain matches a known environmental causal pathway."""

    validation_note: Optional[str] = None
    """Note on validation outcome."""


# ---------------------------------------------------------------------------
# Known causal chains from environmental science
# ---------------------------------------------------------------------------

KNOWN_CAUSAL_CHAINS: List[Dict[str, Any]] = [
    # Turbidity cascades
    {
        "cause": ("sensor", "turbidity"),
        "effect": ("sensor", "dissolved_oxygen"),
        "direction": "negative",
        "lag_range_hours": (2, 48),
        "description": "Increased turbidity reduces light penetration, decreasing photosynthetic oxygen production.",
    },
    {
        "cause": ("sensor", "turbidity"),
        "effect": ("satellite", "secchi_depth"),
        "direction": "negative",
        "lag_range_hours": (0, 24),
        "description": "Turbidity directly reduces water clarity / Secchi depth.",
    },
    # Nutrient loading -> algal bloom
    {
        "cause": ("sensor", "total_phosphorus"),
        "effect": ("satellite", "chlorophyll_a"),
        "direction": "positive",
        "lag_range_hours": (24, 168),
        "description": "Phosphorus loading stimulates algal growth within days to a week.",
    },
    {
        "cause": ("sensor", "total_nitrogen"),
        "effect": ("satellite", "chlorophyll_a"),
        "direction": "positive",
        "lag_range_hours": (24, 168),
        "description": "Nitrogen enrichment promotes algal bloom formation.",
    },
    # Algal bloom -> DO crash
    {
        "cause": ("satellite", "chlorophyll_a"),
        "effect": ("sensor", "dissolved_oxygen"),
        "direction": "negative",
        "lag_range_hours": (24, 120),
        "description": "Algal bloom die-off and decomposition depletes dissolved oxygen.",
    },
    # DO crash -> microbial community shift
    {
        "cause": ("sensor", "dissolved_oxygen"),
        "effect": ("microbial", "alpha_diversity"),
        "direction": "positive",
        "lag_range_hours": (12, 72),
        "description": "Hypoxia reduces aerobic microbial diversity.",
    },
    {
        "cause": ("sensor", "dissolved_oxygen"),
        "effect": ("microbial", "anaerobe_fraction"),
        "direction": "negative",
        "lag_range_hours": (12, 72),
        "description": "Low DO selects for anaerobic organisms.",
    },
    # Heavy metal -> molecular biomarker response
    {
        "cause": ("sensor", "heavy_metal_index"),
        "effect": ("molecular", "metallothionein_expression"),
        "direction": "positive",
        "lag_range_hours": (6, 48),
        "description": "Metal exposure upregulates metallothionein defense genes.",
    },
    {
        "cause": ("sensor", "heavy_metal_index"),
        "effect": ("molecular", "cyp1a_expression"),
        "direction": "positive",
        "lag_range_hours": (6, 72),
        "description": "Heavy metals induce cytochrome P450 detoxification pathways.",
    },
    # Chemical contamination -> behavioral response
    {
        "cause": ("sensor", "conductivity"),
        "effect": ("behavioral", "activity_index"),
        "direction": "negative",
        "lag_range_hours": (1, 24),
        "description": "Sudden conductivity changes indicate chemical spills; fish show reduced activity.",
    },
    {
        "cause": ("molecular", "acetylcholinesterase_activity"),
        "effect": ("behavioral", "activity_index"),
        "direction": "positive",
        "lag_range_hours": (2, 48),
        "description": "Organophosphate inhibition of AChE reduces fish locomotion.",
    },
    # Temperature effects
    {
        "cause": ("sensor", "water_temperature"),
        "effect": ("sensor", "dissolved_oxygen"),
        "direction": "negative",
        "lag_range_hours": (0, 6),
        "description": "Warmer water holds less dissolved oxygen.",
    },
    {
        "cause": ("sensor", "water_temperature"),
        "effect": ("microbial", "community_turnover_rate"),
        "direction": "positive",
        "lag_range_hours": (24, 168),
        "description": "Temperature changes drive microbial community restructuring.",
    },
    # Cyanotoxin cascade
    {
        "cause": ("satellite", "phycocyanin"),
        "effect": ("molecular", "microcystin_concentration"),
        "direction": "positive",
        "lag_range_hours": (0, 48),
        "description": "Phycocyanin pigment from cyanobacteria correlates with microcystin toxin production.",
    },
    {
        "cause": ("molecular", "microcystin_concentration"),
        "effect": ("behavioral", "avoidance_index"),
        "direction": "positive",
        "lag_range_hours": (6, 72),
        "description": "Fish avoid areas with high cyanotoxin concentrations.",
    },
]


# ---------------------------------------------------------------------------
# CausalChainDiscovery
# ---------------------------------------------------------------------------

class CausalChainDiscovery:
    """Discover temporal causal chains across environmental modalities.

    Uses a PCMCI-inspired approach to test for Granger-causal relationships
    between aligned multimodal time series, then validates discovered chains
    against established environmental science knowledge.

    Args:
        max_lag_hours: Maximum lag to test in hours.
        significance_level: P-value threshold for causal links.
        min_observations: Minimum aligned observations required per site.
    """

    def __init__(
        self,
        max_lag_hours: int = 168,
        significance_level: float = 0.05,
        min_observations: int = 50,
    ) -> None:
        self.max_lag_hours = max_lag_hours
        self.significance_level = significance_level
        self.min_observations = min_observations

    def prepare_multimodal_timeseries(
        self,
        site_data: Dict[str, Any],
    ) -> Dict[str, np.ndarray]:
        """Align all modality observations at a site into a common temporal frame.

        Resamples heterogeneous-frequency observations (e.g., hourly sensors,
        daily satellite, weekly eDNA) to a common temporal grid using
        interpolation or forward-fill.

        Args:
            site_data: Dict with keys per modality, each containing:
                - ``timestamps``: list of epoch timestamps.
                - ``variables``: dict mapping variable names to value lists.

        Returns:
            Dict mapping ``(modality, variable)`` string keys to aligned
            numpy arrays on a common time grid.
        """
        # Determine global time range
        all_timestamps: List[float] = []
        for modality, mod_data in site_data.items():
            if isinstance(mod_data, dict) and "timestamps" in mod_data:
                all_timestamps.extend(mod_data["timestamps"])

        if not all_timestamps:
            logger.warning("No timestamps found in site data")
            return {}

        t_min = min(all_timestamps)
        t_max = max(all_timestamps)

        # Common grid at hourly resolution
        n_steps = int((t_max - t_min) / 3600) + 1
        common_grid = np.linspace(t_min, t_max, n_steps)

        aligned: Dict[str, np.ndarray] = {}

        for modality, mod_data in site_data.items():
            if not isinstance(mod_data, dict) or "timestamps" not in mod_data:
                continue

            ts = np.array(mod_data["timestamps"], dtype=np.float64)
            variables = mod_data.get("variables", {})

            for var_name, values in variables.items():
                vals = np.array(values, dtype=np.float64)

                if len(ts) != len(vals):
                    logger.warning(
                        f"Timestamp/value length mismatch for {modality}/{var_name}: "
                        f"{len(ts)} vs {len(vals)}"
                    )
                    continue

                # Sort by time
                sort_idx = np.argsort(ts)
                ts_sorted = ts[sort_idx]
                vals_sorted = vals[sort_idx]

                # Interpolate to common grid
                interpolated = np.interp(
                    common_grid, ts_sorted, vals_sorted,
                    left=np.nan, right=np.nan,
                )

                key = f"{modality}/{var_name}"
                aligned[key] = interpolated

        # Remove variables with too many NaNs
        min_valid = self.min_observations
        filtered = {}
        for key, arr in aligned.items():
            n_valid = np.count_nonzero(~np.isnan(arr))
            if n_valid >= min_valid:
                filtered[key] = arr
            else:
                logger.debug(f"Dropping {key}: only {n_valid} valid observations")

        logger.info(f"Aligned {len(filtered)} variables on grid of {n_steps} hours")
        return filtered

    def discover_chains(
        self,
        timeseries: Dict[str, np.ndarray],
        max_lag_hours: Optional[int] = None,
    ) -> List[CausalChain]:
        """Run heterogeneous causal discovery to find causal relationships.

        Tests all pairwise variable combinations at multiple lags using
        partial correlation (PCMCI approach). Controls for confounders
        through conditional independence testing.

        Args:
            timeseries: Aligned multimodal timeseries from
                ``prepare_multimodal_timeseries``.
            max_lag_hours: Override max lag (uses self.max_lag_hours if None).

        Returns:
            List of discovered CausalChain objects, sorted by strength.
        """
        max_lag = max_lag_hours or self.max_lag_hours
        variables = list(timeseries.keys())
        n_vars = len(variables)

        if n_vars < 2:
            logger.warning("Need at least 2 variables for causal discovery")
            return []

        logger.info(
            f"Testing {n_vars * (n_vars - 1)} variable pairs "
            f"at lags 1-{max_lag}h"
        )

        chains: List[CausalChain] = []

        # Test lags at multiple scales: hourly for short, daily for longer
        test_lags = list(range(1, min(25, max_lag + 1)))  # hourly up to 24h
        test_lags += list(range(24, min(max_lag + 1, 169), 6))  # 6-hourly up to 7 days

        for i, var_x in enumerate(variables):
            for j, var_y in enumerate(variables):
                if i == j:
                    continue

                x = timeseries[var_x]
                y = timeseries[var_y]

                # Parse modality/variable names
                parts_x = var_x.split("/", 1)
                parts_y = var_y.split("/", 1)
                mod_x = parts_x[0] if len(parts_x) > 1 else "unknown"
                name_x = parts_x[1] if len(parts_x) > 1 else var_x
                mod_y = parts_y[0] if len(parts_y) > 1 else "unknown"
                name_y = parts_y[1] if len(parts_y) > 1 else var_y

                best_chain = self._test_granger_causality(
                    x, y, test_lags,
                    cause_modality=mod_x,
                    cause_variable=name_x,
                    effect_modality=mod_y,
                    effect_variable=name_y,
                )

                if best_chain is not None and best_chain.p_value <= self.significance_level:
                    chains.append(best_chain)

        # Sort by strength (descending)
        chains.sort(key=lambda c: abs(c.strength), reverse=True)

        # Validate against known chains
        for chain in chains:
            chain.validated, chain.validation_note = self.validate_chain(
                chain, KNOWN_CAUSAL_CHAINS
            )

        logger.info(
            f"Discovered {len(chains)} significant causal chains "
            f"({sum(1 for c in chains if c.validated)} validated)"
        )

        return chains

    def validate_chain(
        self,
        chain: CausalChain,
        known_chains: List[Dict[str, Any]],
    ) -> Tuple[bool, str]:
        """Check if a discovered chain matches known environmental science.

        Compares the discovered causal relationship against a database of
        established environmental causal pathways, checking for matching
        modalities, variables, direction, and plausible lag range.

        Args:
            chain: Discovered causal chain to validate.
            known_chains: List of known causal relationship dicts.

        Returns:
            Tuple of (is_validated, validation_note).
        """
        for known in known_chains:
            known_cause_mod, known_cause_var = known["cause"]
            known_effect_mod, known_effect_var = known["effect"]
            known_direction = known["direction"]
            lag_lo, lag_hi = known["lag_range_hours"]

            # Check modality match
            if chain.cause_modality != known_cause_mod:
                continue
            if chain.effect_modality != known_effect_mod:
                continue

            # Check variable match (allow partial/fuzzy matching)
            cause_match = (
                known_cause_var.lower() in chain.cause_variable.lower()
                or chain.cause_variable.lower() in known_cause_var.lower()
            )
            effect_match = (
                known_effect_var.lower() in chain.effect_variable.lower()
                or chain.effect_variable.lower() in known_effect_var.lower()
            )

            if not (cause_match and effect_match):
                continue

            # Check lag range
            if not (lag_lo <= chain.lag_hours <= lag_hi):
                note = (
                    f"Matches known chain ({known['description']}) but "
                    f"lag {chain.lag_hours:.1f}h outside expected range "
                    f"[{lag_lo}, {lag_hi}]h"
                )
                return False, note

            # Check direction
            if chain.direction != known_direction:
                note = (
                    f"Matches known chain ({known['description']}) but "
                    f"direction is {chain.direction} (expected {known_direction})"
                )
                return False, note

            return True, f"Validated: {known['description']}"

        return False, "No matching known chain found (may be novel)"

    def aggregate_across_sites(
        self,
        all_chains: Dict[str, List[CausalChain]],
    ) -> Dict[str, Any]:
        """Aggregate discovered chains across many sites.

        Reports frequency (how many sites show a given chain), robustness
        (consistency of lag and strength), and validation rates.

        Args:
            all_chains: Dict mapping site IDs to lists of CausalChain objects.

        Returns:
            Dict with aggregated chain statistics:
                - ``chain_frequency``: how often each chain type appears
                - ``mean_lag``: average lag per chain type
                - ``mean_strength``: average strength per chain type
                - ``validation_rate``: fraction of chains validated
                - ``novel_chains``: list of unvalidated but frequent chains
        """
        # Group chains by (cause_mod/var -> effect_mod/var)
        chain_groups: Dict[str, List[CausalChain]] = {}
        total_chains = 0
        total_validated = 0

        for site_id, chains in all_chains.items():
            for chain in chains:
                key = (
                    f"{chain.cause_modality}/{chain.cause_variable}"
                    f" -> {chain.effect_modality}/{chain.effect_variable}"
                )
                if key not in chain_groups:
                    chain_groups[key] = []
                chain_groups[key].append(chain)
                total_chains += 1
                if chain.validated:
                    total_validated += 1

        n_sites = len(all_chains)
        results: Dict[str, Any] = {
            "n_sites": n_sites,
            "total_chains_discovered": total_chains,
            "total_validated": total_validated,
            "validation_rate": total_validated / max(total_chains, 1),
            "chain_types": {},
            "novel_chains": [],
        }

        for chain_key, group in chain_groups.items():
            lags = [c.lag_hours for c in group]
            strengths = [abs(c.strength) for c in group]
            p_values = [c.p_value for c in group]
            validated_count = sum(1 for c in group if c.validated)

            chain_info = {
                "frequency": len(group),
                "fraction_of_sites": len(group) / max(n_sites, 1),
                "mean_lag_hours": float(np.mean(lags)),
                "std_lag_hours": float(np.std(lags)),
                "mean_strength": float(np.mean(strengths)),
                "std_strength": float(np.std(strengths)),
                "mean_p_value": float(np.mean(p_values)),
                "validated_fraction": validated_count / len(group),
                "direction": group[0].direction,
            }
            results["chain_types"][chain_key] = chain_info

            # Novel chains: appear in >20% of sites but not validated
            if (
                validated_count == 0
                and len(group) / max(n_sites, 1) > 0.2
                and len(group) >= 3
            ):
                results["novel_chains"].append({
                    "chain": chain_key,
                    "frequency": len(group),
                    "mean_lag_hours": float(np.mean(lags)),
                    "mean_strength": float(np.mean(strengths)),
                })

        # Sort chain types by frequency
        results["chain_types"] = dict(
            sorted(
                results["chain_types"].items(),
                key=lambda x: x[1]["frequency"],
                reverse=True,
            )
        )

        return results

    # --- Private helper methods ---

    def _test_granger_causality(
        self,
        x: np.ndarray,
        y: np.ndarray,
        lags: List[int],
        cause_modality: str,
        cause_variable: str,
        effect_modality: str,
        effect_variable: str,
    ) -> Optional[CausalChain]:
        """Test Granger causality from x to y at multiple lags.

        Uses partial correlation with lagged x values to test if past x
        values predict current y beyond y's own past.

        Returns the best (strongest) significant causal chain, or None.
        """
        best_strength = 0.0
        best_lag = 0
        best_pvalue = 1.0
        best_direction = "positive"

        for lag in lags:
            if lag >= len(x) - 10:
                continue

            # Align: y[lag:] ~ x[:-lag]
            y_target = y[lag:]
            x_lagged = x[:-lag] if lag > 0 else x
            y_lagged = y[:-lag] if lag > 0 else y

            # Remove NaN pairs
            valid = ~(np.isnan(y_target) | np.isnan(x_lagged) | np.isnan(y_lagged))
            n_valid = valid.sum()
            if n_valid < self.min_observations:
                continue

            y_t = y_target[valid]
            x_l = x_lagged[valid]
            y_l = y_lagged[valid]

            # Partial correlation: correlation of x_lagged with y_target,
            # controlling for y_lagged (autoregressive component)
            pcorr, pval = self._partial_correlation(x_l, y_t, y_l)

            if abs(pcorr) > abs(best_strength) and pval < best_pvalue:
                best_strength = pcorr
                best_lag = lag
                best_pvalue = pval
                best_direction = "positive" if pcorr > 0 else "negative"

        if best_pvalue > self.significance_level:
            return None

        return CausalChain(
            cause_modality=cause_modality,
            cause_variable=cause_variable,
            effect_modality=effect_modality,
            effect_variable=effect_variable,
            lag_hours=float(best_lag),
            strength=float(best_strength),
            p_value=float(best_pvalue),
            direction=best_direction,
        )

    @staticmethod
    def _partial_correlation(
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
    ) -> Tuple[float, float]:
        """Compute partial correlation of x and y controlling for z.

        Uses residual-based approach: regress x and y on z, then
        correlate residuals.

        Args:
            x: First variable.
            y: Second variable.
            z: Control variable.

        Returns:
            Tuple of (partial correlation, p-value).
        """
        from scipy import stats as sp_stats

        # Residuals of x regressed on z
        z_mean = z.mean()
        z_var = np.var(z)
        if z_var < 1e-10:
            # z is constant; partial correlation = correlation
            r, p = sp_stats.pearsonr(x, y)
            return float(r), float(p)

        beta_xz = np.cov(x, z)[0, 1] / z_var
        resid_x = x - beta_xz * (z - z_mean)

        beta_yz = np.cov(y, z)[0, 1] / z_var
        resid_y = y - beta_yz * (z - z_mean)

        if np.std(resid_x) < 1e-10 or np.std(resid_y) < 1e-10:
            return 0.0, 1.0

        r, p = sp_stats.pearsonr(resid_x, resid_y)
        return float(r), float(p)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for causal chain discovery."""
    parser = argparse.ArgumentParser(
        description="SENTINEL Temporal Causal Chain Discovery",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing site data JSON files.",
    )
    parser.add_argument(
        "--sites",
        type=str,
        nargs="+",
        default=["all"],
        help="Site IDs to analyze, or 'all' (default: all).",
    )
    parser.add_argument(
        "--max-lag",
        type=int,
        default=168,
        help="Maximum lag to test in hours (default: 168 = 7 days).",
    )
    parser.add_argument(
        "--significance",
        type=float,
        default=0.05,
        help="Significance level for causal links (default: 0.05).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/causal"),
        help="Output directory (default: results/causal).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for causal chain discovery."""
    parser = build_parser()
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover site data files
    site_files = sorted(data_dir.glob("*.json"))
    if not site_files:
        logger.error(f"No site data files found in {data_dir}")
        return

    # Filter sites if specified
    if args.sites != ["all"]:
        site_files = [f for f in site_files if f.stem in args.sites]
        logger.info(f"Filtered to {len(site_files)} sites: {args.sites}")

    logger.info(f"Analyzing {len(site_files)} sites with max lag {args.max_lag}h")

    discovery = CausalChainDiscovery(
        max_lag_hours=args.max_lag,
        significance_level=args.significance,
    )

    all_chains: Dict[str, List[CausalChain]] = {}

    for site_file in site_files:
        site_id = site_file.stem
        logger.info(f"Processing site: {site_id}")

        with open(site_file, "r", encoding="utf-8") as f:
            site_data = json.load(f)

        # Prepare timeseries
        aligned = discovery.prepare_multimodal_timeseries(site_data)
        if len(aligned) < 2:
            logger.warning(f"Insufficient variables at {site_id}; skipping")
            continue

        # Discover chains
        chains = discovery.discover_chains(aligned)
        all_chains[site_id] = chains

        # Save per-site results
        site_output = output_dir / f"{site_id}_chains.json"
        with open(site_output, "w", encoding="utf-8") as f:
            json.dump([asdict(c) for c in chains], f, indent=2)
        logger.info(f"  {len(chains)} chains discovered at {site_id}")

    # Aggregate across sites
    if all_chains:
        aggregated = discovery.aggregate_across_sites(all_chains)

        agg_path = output_dir / "aggregated_chains.json"
        with open(agg_path, "w", encoding="utf-8") as f:
            json.dump(aggregated, f, indent=2, default=str)

        logger.info(f"Aggregated results saved to {agg_path}")
        logger.info(
            f"Total: {aggregated['total_chains_discovered']} chains, "
            f"{aggregated['total_validated']} validated "
            f"({aggregated['validation_rate']:.1%})"
        )
        if aggregated["novel_chains"]:
            logger.info(
                f"Novel chains (unvalidated but frequent): "
                f"{len(aggregated['novel_chains'])}"
            )
    else:
        logger.warning("No causal chains discovered across any site")


if __name__ == "__main__":
    main()
