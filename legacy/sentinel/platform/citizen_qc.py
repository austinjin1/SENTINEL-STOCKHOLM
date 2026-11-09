"""
SENTINEL citizen science data quality control.

Three-stage automated QC pipeline for citizen-contributed water quality
observations.  Designed to maximize data inclusion while flagging suspect
readings --- large deviations are flagged but NOT rejected, because they
may indicate genuine local variation.

Stages
------
1. **Physical plausibility** -- season- and altitude-adjusted range checks.
2. **Spatial consistency** -- comparison with nearby sensors and satellite
   estimates.
3. **Temporal consistency** -- drift and bias detection for repeat
   contributors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Sequence

from sentinel.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Season enumeration
# ---------------------------------------------------------------------------


class Season(str, Enum):
    """Northern-hemisphere meteorological seasons."""

    SPRING = "spring"
    SUMMER = "summer"
    AUTUMN = "autumn"
    WINTER = "winter"


def _month_to_season(month: int, latitude: float = 50.0) -> Season:
    """Map calendar month to meteorological season.

    Flips for southern hemisphere (latitude < 0).
    """
    northern = {
        12: Season.WINTER, 1: Season.WINTER, 2: Season.WINTER,
        3: Season.SPRING, 4: Season.SPRING, 5: Season.SPRING,
        6: Season.SUMMER, 7: Season.SUMMER, 8: Season.SUMMER,
        9: Season.AUTUMN, 10: Season.AUTUMN, 11: Season.AUTUMN,
    }
    season = northern[month]
    if latitude < 0:
        flip = {
            Season.WINTER: Season.SUMMER,
            Season.SUMMER: Season.WINTER,
            Season.SPRING: Season.AUTUMN,
            Season.AUTUMN: Season.SPRING,
        }
        season = flip[season]
    return season


# ---------------------------------------------------------------------------
# Plausibility ranges (season-adjusted)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParameterRange:
    """Season-aware plausibility range for a water quality parameter."""

    param: str
    abs_min: float
    abs_max: float
    unit: str
    # Seasonal adjustments: (summer_min, summer_max, winter_min, winter_max)
    # None means no seasonal adjustment.
    seasonal: tuple[float, float, float, float] | None = None


# fmt: off
PARAMETER_RANGES: dict[str, ParameterRange] = {
    "ph":                        ParameterRange("ph", 0.0, 14.0, "pH units"),
    "water_temperature":         ParameterRange("water_temperature", -5.0, 45.0, "degC",
                                                seasonal=(5.0, 40.0, -5.0, 15.0)),
    "dissolved_oxygen":          ParameterRange("dissolved_oxygen", 0.0, 20.0, "mg/L",
                                                seasonal=(0.0, 15.0, 2.0, 20.0)),
    "dissolved_oxygen_saturation": ParameterRange("dissolved_oxygen_saturation", 0.0, 200.0, "% sat"),
    "specific_conductance":      ParameterRange("specific_conductance", 0.0, 80_000.0, "uS/cm"),
    "turbidity":                 ParameterRange("turbidity", 0.0, 5_000.0, "NTU"),
    "nitrate":                   ParameterRange("nitrate", 0.0, 100.0, "mg/L"),
    "nitrite":                   ParameterRange("nitrite", 0.0, 10.0, "mg/L"),
    "ammonia":                   ParameterRange("ammonia", 0.0, 50.0, "mg/L"),
    "total_phosphorus":          ParameterRange("total_phosphorus", 0.0, 50.0, "mg/L"),
    "orthophosphate":            ParameterRange("orthophosphate", 0.0, 30.0, "mg/L"),
    "chlorine":                  ParameterRange("chlorine", 0.0, 10.0, "mg/L"),
    "hardness":                  ParameterRange("hardness", 0.0, 5_000.0, "mg/L CaCO3"),
    "alkalinity":                ParameterRange("alkalinity", 0.0, 5_000.0, "mg/L CaCO3"),
    "copper":                    ParameterRange("copper", 0.0, 10_000.0, "ug/L"),
    "iron":                      ParameterRange("iron", 0.0, 50_000.0, "ug/L"),
    "lead":                      ParameterRange("lead", 0.0, 10_000.0, "ug/L"),
    "chlorophyll_a":             ParameterRange("chlorophyll_a", 0.0, 500.0, "ug/L",
                                                seasonal=(0.0, 500.0, 0.0, 100.0)),
    "secchi_depth":              ParameterRange("secchi_depth", 0.0, 40.0, "m"),
    "total_dissolved_solids":    ParameterRange("total_dissolved_solids", 0.0, 60_000.0, "mg/L"),
    "salinity":                  ParameterRange("salinity", 0.0, 45.0, "PSU"),
    "e_coli":                    ParameterRange("e_coli", 0.0, 1e7, "CFU/100mL"),
    "total_coliform":            ParameterRange("total_coliform", 0.0, 1e8, "CFU/100mL"),
    "fluoride":                  ParameterRange("fluoride", 0.0, 30.0, "mg/L"),
    "sulfate":                   ParameterRange("sulfate", 0.0, 5_000.0, "mg/L"),
}
# fmt: on


# Altitude correction for dissolved oxygen saturation.
# At sea level max DO ~14.6 mg/L at 0 degC; decreases ~11.3% per 1000 m.
_DO_SEA_LEVEL_MAX = 14.6  # mg/L at 0 degC


def _altitude_do_correction(altitude_m: float) -> float:
    """Return a multiplicative correction factor for DO at a given altitude.

    Based on barometric pressure decrease with elevation.
    """
    if altitude_m <= 0:
        return 1.0
    # Barometric formula approximation
    return math.exp(-altitude_m / 8500.0)


# ---------------------------------------------------------------------------
# Stage 1: Physical plausibility
# ---------------------------------------------------------------------------


def check_plausibility(
    parameter: str,
    value: float,
    location: dict[str, Any] | None = None,
    season: Season | str | None = None,
) -> tuple[bool, str]:
    """Check whether a value is physically plausible.

    Parameters
    ----------
    parameter:
        Canonical parameter name (e.g. ``"ph"``, ``"dissolved_oxygen"``).
    value:
        The measured value.
    location:
        Optional dict with ``"latitude"``, ``"longitude"``, ``"altitude_m"``
        keys for location-aware checks.
    season:
        Season string or :class:`Season` enum.  If *None* and *location*
        includes latitude, season is inferred from the current month.

    Returns
    -------
    Tuple of ``(is_valid, reason_if_invalid)``.
    """
    param_lower = parameter.strip().lower()
    if param_lower not in PARAMETER_RANGES:
        return True, ""

    pr = PARAMETER_RANGES[param_lower]

    # Determine effective season
    effective_season: Season | None = None
    if season is not None:
        effective_season = Season(season) if isinstance(season, str) else season
    elif location and "latitude" in location:
        effective_season = _month_to_season(
            datetime.utcnow().month, location["latitude"]
        )

    # Determine range
    lo, hi = pr.abs_min, pr.abs_max
    if effective_season is not None and pr.seasonal is not None:
        s_lo_sum, s_hi_sum, s_lo_win, s_hi_win = pr.seasonal
        if effective_season == Season.SUMMER:
            lo, hi = s_lo_sum, s_hi_sum
        elif effective_season == Season.WINTER:
            lo, hi = s_lo_win, s_hi_win
        # Spring/autumn: use absolute range (transitional)

    # Altitude correction for dissolved oxygen
    if param_lower == "dissolved_oxygen" and location and "altitude_m" in location:
        correction = _altitude_do_correction(location["altitude_m"])
        hi *= correction

    if value < lo:
        return False, (
            f"{param_lower} value {value} {pr.unit} below minimum {lo} {pr.unit}"
        )
    if value > hi:
        return False, (
            f"{param_lower} value {value} {pr.unit} above maximum {hi} {pr.unit}"
        )

    return True, ""


# ---------------------------------------------------------------------------
# Citizen record type (lightweight, for QC purposes)
# ---------------------------------------------------------------------------


@dataclass
class CitizenRecord:
    """A single citizen-submitted water quality observation."""

    parameter: str
    value: float
    unit: str
    latitude: float
    longitude: float
    timestamp: datetime
    contributor_id: str = ""
    altitude_m: float = 0.0
    site_id: str = ""


# ---------------------------------------------------------------------------
# Stage 2: Spatial consistency
# ---------------------------------------------------------------------------


def check_spatial_consistency(
    record: CitizenRecord,
    nearby_records: Sequence[CitizenRecord],
    satellite_estimate: float | None = None,
) -> float:
    """Score spatial consistency against nearby observations.

    Parameters
    ----------
    record:
        The citizen record to evaluate.
    nearby_records:
        Recent records from nearby locations for the same parameter.
    satellite_estimate:
        Optional satellite-derived estimate for the same parameter and
        location.

    Returns
    -------
    Consistency score in ``[0.0, 1.0]``.  1.0 = perfectly consistent.
    """
    if not nearby_records and satellite_estimate is None:
        return 0.5  # No comparison data: neutral

    param = record.parameter.strip().lower()
    pr = PARAMETER_RANGES.get(param)
    if pr is not None:
        tolerance = (pr.abs_max - pr.abs_min) * 0.15
    else:
        tolerance = abs(record.value) * 0.30 if record.value != 0 else 1.0

    agreements = 0.0
    total_weight = 0.0

    for neighbor in nearby_records:
        if neighbor.parameter.strip().lower() != param:
            continue
        weight = 1.0
        deviation = abs(record.value - neighbor.value)
        # Smooth scoring: 1.0 at zero deviation, 0.0 at 2*tolerance
        score = max(0.0, 1.0 - deviation / (2.0 * tolerance))
        agreements += weight * score
        total_weight += weight

    if satellite_estimate is not None:
        weight = 2.0  # satellite weighted higher
        deviation = abs(record.value - satellite_estimate)
        score = max(0.0, 1.0 - deviation / (2.0 * tolerance))
        agreements += weight * score
        total_weight += weight

    if total_weight == 0.0:
        return 0.5

    return min(1.0, agreements / total_weight)


# ---------------------------------------------------------------------------
# Stage 3: Temporal consistency
# ---------------------------------------------------------------------------


def check_temporal_consistency(
    record: CitizenRecord,
    contributor_history: Sequence[CitizenRecord],
) -> tuple[float, float | None]:
    """Score temporal consistency and detect systematic kit bias.

    Parameters
    ----------
    record:
        The current record.
    contributor_history:
        Previous records from the same contributor at the same (or nearby)
        location.

    Returns
    -------
    ``(consistency_score, estimated_bias)`` where *estimated_bias* is
    *None* if insufficient history for bias estimation.
    """
    if not contributor_history:
        return 0.5, None

    param = record.parameter.strip().lower()
    relevant = [
        r
        for r in contributor_history
        if (
            r.parameter.strip().lower() == param
            and r.timestamp < record.timestamp
        )
    ]

    if not relevant:
        return 0.5, None

    values = [r.value for r in relevant]
    n = len(values)
    mean_val = sum(values) / n

    if n >= 2:
        variance = sum((v - mean_val) ** 2 for v in values) / (n - 1)
        std_val = max(variance ** 0.5, abs(mean_val) * 0.05, 0.1)
    else:
        std_val = abs(mean_val) * 0.2 if mean_val != 0 else 1.0

    z_score = abs(record.value - mean_val) / std_val
    consistency = max(0.0, 1.0 - z_score / 3.0)

    # Bias detection: if >= 5 historical records, check for systematic offset
    estimated_bias: float | None = None
    if n >= 5:
        # Use the median offset between contributor values and the overall
        # mean as the bias estimate.
        offsets = sorted(v - mean_val for v in values)
        mid = n // 2
        median_offset = (
            offsets[mid]
            if n % 2 == 1
            else (offsets[mid - 1] + offsets[mid]) / 2.0
        )
        # Only report bias if it exceeds 5% of the mean or 0.1 absolute
        threshold = max(abs(mean_val) * 0.05, 0.1)
        if abs(median_offset) > threshold:
            estimated_bias = median_offset

    return consistency, estimated_bias


# ---------------------------------------------------------------------------
# QC report
# ---------------------------------------------------------------------------


@dataclass
class QCReport:
    """Detailed quality control report for a citizen submission."""

    record: CitizenRecord
    plausibility_passed: bool = True
    plausibility_reason: str = ""
    spatial_score: float = 0.5
    temporal_score: float = 0.5
    estimated_bias: float | None = None
    quality_tier: str = "Q3"
    recommendations: list[str] = field(default_factory=list)

    @property
    def overall_score(self) -> float:
        """Weighted overall QC score."""
        if not self.plausibility_passed:
            return 0.0
        return 0.5 * self.spatial_score + 0.5 * self.temporal_score


# ---------------------------------------------------------------------------
# Full QC pipeline
# ---------------------------------------------------------------------------


class CitizenQCPipeline:
    """Three-stage QC pipeline for citizen science submissions.

    Usage::

        pipeline = CitizenQCPipeline()
        report = pipeline.process_submission(
            record=citizen_record,
            context={
                "nearby_records": [...],
                "satellite_estimate": 7.2,
                "contributor_history": [...],
            },
        )
    """

    def __init__(self, *, promotion_threshold: float = 0.7) -> None:
        self.promotion_threshold = promotion_threshold

    def process_submission(
        self,
        record: CitizenRecord,
        context: dict[str, Any],
    ) -> QCReport:
        """Run all three QC stages and return a detailed report.

        Parameters
        ----------
        record:
            The citizen record to evaluate.
        context:
            Dictionary with optional keys:

            - ``"nearby_records"`` -- sequence of :class:`CitizenRecord`
            - ``"satellite_estimate"`` -- float
            - ``"contributor_history"`` -- sequence of :class:`CitizenRecord`
            - ``"location"`` -- dict with ``altitude_m``, etc.
            - ``"season"`` -- :class:`Season` or string

        Returns
        -------
        A :class:`QCReport` with per-stage scores and recommendations.
        """
        report = QCReport(record=record)

        # Stage 1: Physical plausibility
        location = context.get("location", {
            "latitude": record.latitude,
            "longitude": record.longitude,
            "altitude_m": record.altitude_m,
        })
        season = context.get("season")

        plausible, reason = check_plausibility(
            record.parameter, record.value,
            location=location, season=season,
        )
        report.plausibility_passed = plausible
        report.plausibility_reason = reason

        if not plausible:
            report.quality_tier = "Q3"
            report.recommendations.append(
                f"Value failed plausibility check: {reason}. "
                "Please verify your measurement."
            )
            logger.info(
                f"Plausibility FAIL for {record.parameter}={record.value} "
                f"from contributor {record.contributor_id!r}: {reason}"
            )
            return report

        # Stage 2: Spatial consistency
        nearby = context.get("nearby_records", [])
        sat_est = context.get("satellite_estimate")
        report.spatial_score = check_spatial_consistency(
            record, nearby, satellite_estimate=sat_est,
        )

        if report.spatial_score < 0.3:
            report.recommendations.append(
                "Value deviates significantly from nearby observations. "
                "This may indicate a genuine local anomaly or a measurement "
                "error -- consider retesting."
            )

        # Stage 3: Temporal consistency
        history = context.get("contributor_history", [])
        report.temporal_score, report.estimated_bias = (
            check_temporal_consistency(record, history)
        )

        if report.estimated_bias is not None:
            report.recommendations.append(
                f"Systematic bias of {report.estimated_bias:+.2f} detected "
                f"for your {record.parameter} readings. Consider recalibrating "
                "your test kit."
            )

        # Tier assignment
        if (
            report.spatial_score >= self.promotion_threshold
            and report.temporal_score >= self.promotion_threshold
        ):
            report.quality_tier = "Q2"
            logger.debug(
                f"Promoting {record.contributor_id}/{record.parameter} "
                f"Q3 -> Q2 (spatial={report.spatial_score:.2f}, "
                f"temporal={report.temporal_score:.2f})"
            )
        else:
            report.quality_tier = "Q3"

        return report
