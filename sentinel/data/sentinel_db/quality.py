"""
SENTINEL-DB quality tier assignment and validation.

Implements a multi-stage quality assurance pipeline:
1. Source-based tier assignment (lab, sensor, citizen science, derived).
2. Physical plausibility checks against known parameter ranges.
3. Spatial consistency scoring against neighboring observations.
4. Temporal consistency scoring for repeat contributors.
5. Tier promotion (Q3 -> Q2) when spatial and temporal scores are high.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Sequence

from sentinel.data.sentinel_db.schema import QualityTier, WaterQualityRecord
from sentinel.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Physical plausibility ranges
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PlausibleRange:
    """Physically plausible range for a water quality parameter."""

    param: str
    min_val: float
    max_val: float
    unit: str


# Ranges for 25+ common parameters.  Values based on environmental extremes.
PLAUSIBLE_RANGES: dict[str, PlausibleRange] = {
    "ph": PlausibleRange("ph", 0.0, 14.0, "pH units"),
    "water_temperature": PlausibleRange("water_temperature", -5.0, 45.0, "degC"),
    "dissolved_oxygen": PlausibleRange("dissolved_oxygen", 0.0, 25.0, "mg/L"),
    "dissolved_oxygen_saturation": PlausibleRange(
        "dissolved_oxygen_saturation", 0.0, 300.0, "% sat"
    ),
    "specific_conductance": PlausibleRange(
        "specific_conductance", 0.0, 100_000.0, "uS/cm"
    ),
    "electrical_conductivity": PlausibleRange(
        "electrical_conductivity", 0.0, 100_000.0, "uS/cm"
    ),
    "turbidity": PlausibleRange("turbidity", 0.0, 10_000.0, "NTU"),
    "total_dissolved_solids": PlausibleRange(
        "total_dissolved_solids", 0.0, 100_000.0, "mg/L"
    ),
    "total_suspended_solids": PlausibleRange(
        "total_suspended_solids", 0.0, 50_000.0, "mg/L"
    ),
    "oxidation_reduction_potential": PlausibleRange(
        "oxidation_reduction_potential", -500.0, 1000.0, "mV"
    ),
    "total_nitrogen": PlausibleRange("total_nitrogen", 0.0, 200.0, "mg/L"),
    "nitrate": PlausibleRange("nitrate", 0.0, 100.0, "mg/L"),
    "nitrite": PlausibleRange("nitrite", 0.0, 10.0, "mg/L"),
    "nitrate_nitrite": PlausibleRange("nitrate_nitrite", 0.0, 100.0, "mg/L"),
    "ammonia": PlausibleRange("ammonia", 0.0, 50.0, "mg/L"),
    "ammonium": PlausibleRange("ammonium", 0.0, 50.0, "mg/L"),
    "total_phosphorus": PlausibleRange("total_phosphorus", 0.0, 50.0, "mg/L"),
    "orthophosphate": PlausibleRange("orthophosphate", 0.0, 30.0, "mg/L"),
    "biological_oxygen_demand": PlausibleRange(
        "biological_oxygen_demand", 0.0, 500.0, "mg/L"
    ),
    "chemical_oxygen_demand": PlausibleRange(
        "chemical_oxygen_demand", 0.0, 1000.0, "mg/L"
    ),
    "total_organic_carbon": PlausibleRange("total_organic_carbon", 0.0, 200.0, "mg/L"),
    "dissolved_organic_carbon": PlausibleRange(
        "dissolved_organic_carbon", 0.0, 200.0, "mg/L"
    ),
    "chlorophyll_a": PlausibleRange("chlorophyll_a", 0.0, 1000.0, "ug/L"),
    "salinity": PlausibleRange("salinity", 0.0, 45.0, "PSU"),
    "hardness": PlausibleRange("hardness", 0.0, 5000.0, "mg/L CaCO3"),
    "alkalinity": PlausibleRange("alkalinity", 0.0, 5000.0, "mg/L CaCO3"),
    "secchi_depth": PlausibleRange("secchi_depth", 0.0, 60.0, "m"),
    "chloride": PlausibleRange("chloride", 0.0, 30_000.0, "mg/L"),
    "sulfate": PlausibleRange("sulfate", 0.0, 5000.0, "mg/L"),
    "fluoride": PlausibleRange("fluoride", 0.0, 30.0, "mg/L"),
    "calcium": PlausibleRange("calcium", 0.0, 2000.0, "mg/L"),
    "iron": PlausibleRange("iron", 0.0, 50_000.0, "ug/L"),
    "lead": PlausibleRange("lead", 0.0, 10_000.0, "ug/L"),
    "copper": PlausibleRange("copper", 0.0, 10_000.0, "ug/L"),
    "zinc": PlausibleRange("zinc", 0.0, 50_000.0, "ug/L"),
    "mercury": PlausibleRange("mercury", 0.0, 100.0, "ug/L"),
    "arsenic": PlausibleRange("arsenic", 0.0, 5000.0, "ug/L"),
    "discharge": PlausibleRange("discharge", 0.0, 500_000.0, "m3/s"),
    "total_coliform": PlausibleRange("total_coliform", 0.0, 1e8, "CFU/100mL"),
    "fecal_coliform": PlausibleRange("fecal_coliform", 0.0, 1e7, "CFU/100mL"),
    "e_coli": PlausibleRange("e_coli", 0.0, 1e7, "CFU/100mL"),
}


# ---------------------------------------------------------------------------
# Source -> quality tier mapping
# ---------------------------------------------------------------------------

_SOURCE_TIER_MAP: dict[str, QualityTier] = {
    # Q1: Certified lab / official monitoring
    "epa_wqp": QualityTier.Q1,
    "epa_storet": QualityTier.Q1,
    "eu_waterbase": QualityTier.Q1,
    "gemstat": QualityTier.Q1,
    "grqa": QualityTier.Q1,
    "usgs_nwis_lab": QualityTier.Q1,
    # Q2: Calibrated sensors
    "usgs_nwis": QualityTier.Q2,
    "usgs_iv": QualityTier.Q2,
    "sensor": QualityTier.Q2,
    "continuous_monitor": QualityTier.Q2,
    "buoy": QualityTier.Q2,
    # Q3: Citizen science
    "freshwater_watch": QualityTier.Q3,
    "citizen_science": QualityTier.Q3,
    "globe_observer": QualityTier.Q3,
    "inaturalist": QualityTier.Q3,
    # Q4: Derived / modelled
    "satellite_derived": QualityTier.Q4,
    "model": QualityTier.Q4,
    "interpolated": QualityTier.Q4,
    "estimated": QualityTier.Q4,
}


# ---------------------------------------------------------------------------
# Tier assignment
# ---------------------------------------------------------------------------


def assign_quality_tier(
    record: WaterQualityRecord,
    source: str,
) -> QualityTier:
    """Assign a quality tier based on data source provenance.

    Parameters
    ----------
    record:
        The water quality record (used for future context-aware assignment).
    source:
        Data source identifier string.

    Returns
    -------
    The assigned :class:`QualityTier`.
    """
    source_lower = source.strip().lower()

    # Exact match
    if source_lower in _SOURCE_TIER_MAP:
        return _SOURCE_TIER_MAP[source_lower]

    # Partial match
    for key, tier in _SOURCE_TIER_MAP.items():
        if key in source_lower or source_lower in key:
            return tier

    # Default: Q3 (unverified)
    logger.debug(f"Unknown source {source!r}, defaulting to Q3")
    return QualityTier.Q3


# ---------------------------------------------------------------------------
# Physical plausibility validation
# ---------------------------------------------------------------------------


def validate_physical_plausibility(
    record: WaterQualityRecord,
) -> tuple[bool, str]:
    """Check whether a record's value is physically plausible.

    Parameters
    ----------
    record:
        Water quality record to validate.

    Returns
    -------
    Tuple of ``(is_plausible, reason)``.  If plausible, reason is empty.
    """
    param = record.canonical_param
    value = record.value

    if param not in PLAUSIBLE_RANGES:
        return True, ""

    pr = PLAUSIBLE_RANGES[param]
    if value < pr.min_val:
        return False, (
            f"{param} value {value} {record.unit} below physical minimum "
            f"{pr.min_val} {pr.unit}"
        )
    if value > pr.max_val:
        return False, (
            f"{param} value {value} {record.unit} above physical maximum "
            f"{pr.max_val} {pr.unit}"
        )

    return True, ""


# ---------------------------------------------------------------------------
# Spatial consistency
# ---------------------------------------------------------------------------


def check_spatial_consistency(
    record: WaterQualityRecord,
    nearby_records: Sequence[WaterQualityRecord],
    *,
    satellite_estimate: float | None = None,
) -> float:
    """Score spatial consistency of a record against its neighbors.

    The score is computed as the fraction of nearby observations that agree
    with the record within parameter-dependent tolerance bands.  A satellite
    estimate, if available, is weighted 2x.

    Parameters
    ----------
    record:
        The target record to score.
    nearby_records:
        Neighboring records (same parameter, recent time window).
    satellite_estimate:
        Optional satellite-derived estimate for comparison.

    Returns
    -------
    Score in [0.0, 1.0].  Higher means more spatially consistent.
    """
    if not nearby_records and satellite_estimate is None:
        return 0.5  # No neighbors: neutral score

    # Parameter-dependent tolerance (fraction of value range or fixed)
    param = record.canonical_param
    pr = PLAUSIBLE_RANGES.get(param)
    if pr is not None:
        # Tolerance = 20% of the plausible range
        tolerance = (pr.max_val - pr.min_val) * 0.20
    else:
        # Fallback: 30% of the record value
        tolerance = abs(record.value) * 0.30 if record.value != 0 else 1.0

    agreements = 0.0
    total_weight = 0.0

    # Compare with neighbors
    for neighbor in nearby_records:
        if neighbor.canonical_param != record.canonical_param:
            continue
        weight = 1.0
        if abs(record.value - neighbor.value) <= tolerance:
            agreements += weight
        total_weight += weight

    # Compare with satellite estimate
    if satellite_estimate is not None:
        weight = 2.0
        if abs(record.value - satellite_estimate) <= tolerance:
            agreements += weight
        total_weight += weight

    if total_weight == 0.0:
        return 0.5

    return min(1.0, agreements / total_weight)


# ---------------------------------------------------------------------------
# Temporal consistency
# ---------------------------------------------------------------------------


def check_temporal_consistency(
    record: WaterQualityRecord,
    contributor_history: Sequence[WaterQualityRecord],
    *,
    max_lookback_days: int = 365,
) -> float:
    """Score temporal consistency for a repeat contributor.

    Evaluates how consistent the contributor's measurements are over time
    at the same location and parameter.

    Parameters
    ----------
    record:
        The current record.
    contributor_history:
        Previous records from the same contributor at the same site.
    max_lookback_days:
        How far back to look in the contributor's history.

    Returns
    -------
    Score in [0.0, 1.0].  Higher means more temporally consistent.
    """
    if not contributor_history:
        return 0.5  # No history: neutral

    cutoff = record.timestamp - timedelta(days=max_lookback_days)

    # Filter to same parameter, same H3 cell, within lookback window
    relevant = [
        r
        for r in contributor_history
        if (
            r.canonical_param == record.canonical_param
            and r.h3_index == record.h3_index
            and r.timestamp >= cutoff
            and r.timestamp < record.timestamp
        )
    ]

    if not relevant:
        return 0.5

    # Compute mean and standard deviation of historical values
    values = [r.value for r in relevant]
    n = len(values)
    mean_val = sum(values) / n
    if n >= 2:
        variance = sum((v - mean_val) ** 2 for v in values) / (n - 1)
        std_val = variance ** 0.5
    else:
        std_val = abs(mean_val) * 0.2 if mean_val != 0 else 1.0

    # Score: how many standard deviations away is the current value?
    if std_val == 0:
        std_val = abs(mean_val) * 0.1 if mean_val != 0 else 1.0

    z_score = abs(record.value - mean_val) / std_val

    # Map z-score to [0, 1]: z=0 -> 1.0, z=3 -> 0.0
    score = max(0.0, 1.0 - z_score / 3.0)

    return score


# ---------------------------------------------------------------------------
# Tier promotion
# ---------------------------------------------------------------------------


def promote_tier(
    record: WaterQualityRecord,
    spatial_score: float,
    temporal_score: float,
    *,
    threshold: float = 0.7,
) -> QualityTier:
    """Potentially promote a record's quality tier based on consistency scores.

    Promotion rules:
    - Q3 -> Q2 if both spatial and temporal scores exceed the threshold.
    - Q4 -> Q3 if both scores exceed the threshold.
    - Q1 and Q2 are never demoted.

    Parameters
    ----------
    record:
        The record to evaluate.
    spatial_score:
        Spatial consistency score [0, 1].
    temporal_score:
        Temporal consistency score [0, 1].
    threshold:
        Minimum score for promotion (default 0.7).

    Returns
    -------
    The (potentially promoted) quality tier.
    """
    current = record.quality_tier

    if spatial_score >= threshold and temporal_score >= threshold:
        if current == QualityTier.Q3:
            logger.debug(
                f"Promoting {record.site_id}/{record.canonical_param} "
                f"Q3 -> Q2 (spatial={spatial_score:.2f}, temporal={temporal_score:.2f})"
            )
            return QualityTier.Q2
        if current == QualityTier.Q4:
            logger.debug(
                f"Promoting {record.site_id}/{record.canonical_param} "
                f"Q4 -> Q3 (spatial={spatial_score:.2f}, temporal={temporal_score:.2f})"
            )
            return QualityTier.Q3

    return current
