"""
SENTINEL test kit data input and validation.

Handles home water test kit readings: validation against physical ranges
and sensor/satellite baselines, per-kit bias calibration for repeat users,
and submission into SENTINEL-DB format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Sequence

from sentinel.data.sentinel_db.schema import QualityTier, WaterQualityRecord
from sentinel.platform.citizen_qc import (
    CitizenQCPipeline,
    CitizenRecord,
    check_plausibility,
)
from sentinel.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Supported test kit parameters
# ---------------------------------------------------------------------------

TEST_KIT_PARAMETERS: list[str] = [
    "ph",
    "nitrate",
    "nitrite",
    "orthophosphate",
    "ammonia",
    "dissolved_oxygen",
    "chlorine",
    "hardness",
    "alkalinity",
    "copper",
    "iron",
    "lead",
]
"""Canonical parameter names commonly measured by home water test kits."""


# Default units for each parameter
_DEFAULT_UNITS: dict[str, str] = {
    "ph": "pH units",
    "nitrate": "mg/L",
    "nitrite": "mg/L",
    "orthophosphate": "mg/L",
    "ammonia": "mg/L",
    "dissolved_oxygen": "mg/L",
    "chlorine": "mg/L",
    "hardness": "mg/L CaCO3",
    "alkalinity": "mg/L CaCO3",
    "copper": "ug/L",
    "iron": "ug/L",
    "lead": "ug/L",
}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class TestKitReading:
    """A single reading from a home water test kit."""

    parameter: str
    value: float
    unit: str = ""
    kit_brand: str = ""
    test_method: str = ""

    def __post_init__(self) -> None:
        self.parameter = self.parameter.strip().lower()
        if not self.unit:
            self.unit = _DEFAULT_UNITS.get(self.parameter, "")


@dataclass
class ValidationResult:
    """Result of validating a single test kit reading."""

    is_valid: bool = True
    confidence: float = 1.0
    deviation_from_baseline: float | None = None
    suggested_correction: float | None = None
    messages: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_test_kit_reading(
    reading: TestKitReading,
    lat: float,
    lon: float,
    *,
    satellite_baseline: float | None = None,
    sensor_baseline: float | None = None,
) -> ValidationResult:
    """Validate a test kit reading against plausibility and baselines.

    Parameters
    ----------
    reading:
        The test kit reading to validate.
    lat, lon:
        GPS coordinates of the measurement location.
    satellite_baseline:
        Satellite-derived estimate for the same parameter, if available.
    sensor_baseline:
        Nearest sensor station value for the same parameter, if available.

    Returns
    -------
    :class:`ValidationResult` with validity flag, confidence, and
    deviation metrics.
    """
    result = ValidationResult()

    # Step 1: Physical plausibility
    plausible, reason = check_plausibility(
        reading.parameter,
        reading.value,
        location={"latitude": lat, "longitude": lon},
    )
    if not plausible:
        result.is_valid = False
        result.confidence = 0.0
        result.messages.append(f"Implausible value: {reason}")
        return result

    # Step 2: Compare with baselines
    baselines: list[tuple[str, float]] = []
    if sensor_baseline is not None:
        baselines.append(("sensor", sensor_baseline))
    if satellite_baseline is not None:
        baselines.append(("satellite", satellite_baseline))

    if baselines:
        deviations: list[float] = []
        for source, baseline in baselines:
            dev = reading.value - baseline
            deviations.append(dev)

            # Flag large deviations
            rel_dev = abs(dev) / max(abs(baseline), 0.01)
            if rel_dev > 0.5:
                result.messages.append(
                    f"Large deviation from {source} baseline: "
                    f"{reading.value} vs {baseline} ({rel_dev:.0%} difference)"
                )

        # Mean deviation across available baselines
        mean_dev = sum(deviations) / len(deviations)
        result.deviation_from_baseline = mean_dev

        # Confidence decreases with deviation
        max_rel_dev = max(
            abs(d) / max(abs(b), 0.01) for d, (_, b) in zip(deviations, baselines)
        )
        result.confidence = max(0.1, 1.0 - max_rel_dev)

        # Suggest correction only for severe deviations
        if max_rel_dev > 0.5:
            avg_baseline = sum(b for _, b in baselines) / len(baselines)
            result.suggested_correction = avg_baseline
            result.messages.append(
                f"Suggested correction: {avg_baseline:.2f} "
                f"(average of available baselines)"
            )
    else:
        # No baselines: moderate confidence based solely on plausibility
        result.confidence = 0.6
        result.messages.append(
            "No baseline data available for comparison; "
            "confidence based on plausibility only."
        )

    return result


# ---------------------------------------------------------------------------
# Kit bias calibration
# ---------------------------------------------------------------------------


def calibrate_kit_readings(
    readings_history: Sequence[TestKitReading],
    reference_values: Sequence[float],
) -> dict[str, float]:
    """Learn systematic bias correction factors for a user's test kit.

    Parameters
    ----------
    readings_history:
        Historical readings from the contributor's kit.
    reference_values:
        Corresponding reference values (same length/order as
        *readings_history*), from sensor stations or lab measurements.

    Returns
    -------
    Dictionary mapping parameter names to additive bias corrections.
    Apply as: ``corrected = reading.value - bias[parameter]``.

    Raises
    ------
    ValueError
        If lengths of *readings_history* and *reference_values* differ.
    """
    if len(readings_history) != len(reference_values):
        raise ValueError(
            f"Length mismatch: {len(readings_history)} readings vs "
            f"{len(reference_values)} reference values"
        )

    # Group by parameter
    offsets_by_param: dict[str, list[float]] = {}
    for reading, ref in zip(readings_history, reference_values):
        param = reading.parameter
        offsets_by_param.setdefault(param, []).append(reading.value - ref)

    corrections: dict[str, float] = {}
    for param, offsets in offsets_by_param.items():
        if len(offsets) < 3:
            # Need at least 3 paired observations
            logger.debug(
                f"Skipping bias calibration for {param}: "
                f"only {len(offsets)} paired observations"
            )
            continue

        # Use median offset (robust to outliers)
        sorted_offsets = sorted(offsets)
        n = len(sorted_offsets)
        mid = n // 2
        median_bias = (
            sorted_offsets[mid]
            if n % 2 == 1
            else (sorted_offsets[mid - 1] + sorted_offsets[mid]) / 2.0
        )

        # Only apply correction if bias is meaningful
        mean_ref = sum(
            ref
            for reading, ref in zip(readings_history, reference_values)
            if reading.parameter == param
        ) / len(offsets)
        threshold = max(abs(mean_ref) * 0.02, 0.05)

        if abs(median_bias) > threshold:
            corrections[param] = round(median_bias, 4)
            logger.info(
                f"Kit calibration: {param} bias = {median_bias:+.4f} "
                f"(based on {n} paired observations)"
            )

    return corrections


# ---------------------------------------------------------------------------
# Full submission pipeline
# ---------------------------------------------------------------------------


class TestKitSubmission:
    """End-to-end test kit submission pipeline.

    Validates readings, applies QC, corrects for known kit bias, and
    produces SENTINEL-DB compatible :class:`WaterQualityRecord` objects.

    Parameters
    ----------
    qc_pipeline:
        An optional pre-configured :class:`CitizenQCPipeline`.
        If *None*, a default pipeline is created.
    kit_corrections:
        Optional per-parameter bias corrections from
        :func:`calibrate_kit_readings`.
    """

    def __init__(
        self,
        qc_pipeline: CitizenQCPipeline | None = None,
        kit_corrections: dict[str, float] | None = None,
    ) -> None:
        self.qc_pipeline = qc_pipeline or CitizenQCPipeline()
        self.kit_corrections = kit_corrections or {}

    def submit(
        self,
        readings: Sequence[TestKitReading],
        lat: float,
        lon: float,
        contributor_id: str,
        *,
        timestamp: datetime | None = None,
        satellite_baselines: dict[str, float] | None = None,
        sensor_baselines: dict[str, float] | None = None,
        nearby_records: Sequence[CitizenRecord] | None = None,
        contributor_history: Sequence[CitizenRecord] | None = None,
    ) -> list[WaterQualityRecord]:
        """Submit a batch of test kit readings.

        Parameters
        ----------
        readings:
            List of :class:`TestKitReading` from a single test session.
        lat, lon:
            GPS coordinates of the test location.
        contributor_id:
            Unique contributor identifier.
        timestamp:
            Measurement timestamp (defaults to UTC now).
        satellite_baselines:
            Dict mapping parameter names to satellite-derived values.
        sensor_baselines:
            Dict mapping parameter names to nearest sensor values.
        nearby_records:
            Recent nearby records for spatial QC.
        contributor_history:
            Historical records from this contributor for temporal QC.

        Returns
        -------
        List of :class:`WaterQualityRecord` ready for SENTINEL-DB ingestion.
        """
        ts = timestamp or datetime.utcnow()
        sat_baselines = satellite_baselines or {}
        sen_baselines = sensor_baselines or {}
        nearby = list(nearby_records or [])
        history = list(contributor_history or [])

        records: list[WaterQualityRecord] = []

        for reading in readings:
            param = reading.parameter

            # Step 1: Validate
            validation = validate_test_kit_reading(
                reading, lat, lon,
                satellite_baseline=sat_baselines.get(param),
                sensor_baseline=sen_baselines.get(param),
            )

            if not validation.is_valid:
                logger.warning(
                    f"Rejecting {param}={reading.value} from "
                    f"contributor {contributor_id}: "
                    f"{'; '.join(validation.messages)}"
                )
                continue

            # Step 2: Apply bias correction
            corrected_value = reading.value
            if param in self.kit_corrections:
                correction = self.kit_corrections[param]
                corrected_value = reading.value - correction
                logger.debug(
                    f"Applied bias correction for {param}: "
                    f"{reading.value} -> {corrected_value} "
                    f"(correction={correction:+.4f})"
                )

            # Step 3: QC pipeline
            citizen_rec = CitizenRecord(
                parameter=param,
                value=corrected_value,
                unit=reading.unit,
                latitude=lat,
                longitude=lon,
                timestamp=ts,
                contributor_id=contributor_id,
            )

            qc_report = self.qc_pipeline.process_submission(
                record=citizen_rec,
                context={
                    "nearby_records": nearby,
                    "satellite_estimate": sat_baselines.get(param),
                    "contributor_history": history,
                },
            )

            # Step 4: Build SENTINEL-DB record
            quality_tier = QualityTier(qc_report.quality_tier)

            wqr = WaterQualityRecord(
                canonical_param=param,
                value=corrected_value,
                unit=reading.unit or _DEFAULT_UNITS.get(param, ""),
                timestamp=ts,
                latitude=lat,
                longitude=lon,
                h3_index="",  # Populated downstream by SENTINEL-DB
                source=f"testkit:{reading.kit_brand}" if reading.kit_brand else "testkit",
                quality_tier=quality_tier,
                raw_param_name=reading.parameter,
                raw_unit=reading.unit,
                site_id=f"citizen_{contributor_id}",
            )

            records.append(wqr)
            logger.info(
                f"Accepted {param}={corrected_value:.3f} "
                f"({quality_tier.value}) from {contributor_id}"
            )

        return records
