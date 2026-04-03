"""
SENTINEL-DB cross-modality linking engine.

Spatiotemporally links water quality records with satellite observations,
microbial community data, and ECOTOX toxicity records to build unified
multimodal records for downstream ML.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Sequence

from sentinel.data.sentinel_db.schema import (
    LinkedMultimodalRecord,
    MicrobialSample,
    SatelliteObservation,
    WaterQualityRecord,
)
from sentinel.data.sentinel_db.spatial import (
    _haversine_m,
    find_nearby_records,
    h3_to_latlon,
)
from sentinel.utils.logging import get_logger, make_progress

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Single-modality matching
# ---------------------------------------------------------------------------


def find_satellite_match(
    record: WaterQualityRecord,
    satellite_catalog: Sequence[SatelliteObservation],
    *,
    spatial_tol_m: float = 500.0,
    temporal_tol_hours: float = 3.0,
) -> SatelliteObservation | None:
    """Find the best-matching satellite observation for a WQ record.

    The match must be within *spatial_tol_m* meters and *temporal_tol_hours*
    hours of the record.  When multiple candidates qualify, the closest in
    time (and lowest cloud cover) is preferred.

    Parameters
    ----------
    record:
        Water quality record to match.
    satellite_catalog:
        Available satellite observations.
    spatial_tol_m:
        Maximum spatial distance in meters.
    temporal_tol_hours:
        Maximum temporal distance in hours.

    Returns
    -------
    Best matching :class:`SatelliteObservation`, or ``None``.
    """
    time_window = timedelta(hours=temporal_tol_hours)
    rec_lat, rec_lon = record.latitude, record.longitude

    best: SatelliteObservation | None = None
    best_score = float("inf")

    for sat in satellite_catalog:
        # Temporal check
        dt = abs((sat.timestamp - record.timestamp).total_seconds()) / 3600.0
        if dt > temporal_tol_hours:
            continue

        # Spatial check
        sat_lat, sat_lon = h3_to_latlon(sat.h3_index)
        dist = _haversine_m(rec_lat, rec_lon, sat_lat, sat_lon)
        if dist > spatial_tol_m:
            continue

        # Score: weighted combination of time gap and cloud cover
        score = dt + sat.cloud_pct / 100.0
        if score < best_score:
            best_score = score
            best = sat

    return best


def find_microbial_match(
    record: WaterQualityRecord,
    microbial_catalog: Sequence[MicrobialSample],
    *,
    spatial_tol_km: float = 10.0,
    temporal_tol_days: float = 30.0,
) -> MicrobialSample | None:
    """Find the best-matching microbial community sample for a WQ record.

    Microbial samples are typically collected less frequently and with
    coarser spatial alignment, so tolerances are larger than for satellite.

    Parameters
    ----------
    record:
        Water quality record to match.
    microbial_catalog:
        Available microbial samples.
    spatial_tol_km:
        Maximum spatial distance in kilometers.
    temporal_tol_days:
        Maximum temporal distance in days.

    Returns
    -------
    Best matching :class:`MicrobialSample`, or ``None``.
    """
    spatial_tol_m = spatial_tol_km * 1000.0
    rec_lat, rec_lon = record.latitude, record.longitude

    best: MicrobialSample | None = None
    best_dt = float("inf")

    for mic in microbial_catalog:
        # Temporal check
        dt_days = abs((mic.timestamp - record.timestamp).total_seconds()) / 86400.0
        if dt_days > temporal_tol_days:
            continue

        # Spatial check
        dist = _haversine_m(rec_lat, rec_lon, mic.latitude, mic.longitude)
        if dist > spatial_tol_m:
            continue

        if dt_days < best_dt:
            best_dt = dt_days
            best = mic

    return best


def find_ecotox_matches(
    record: WaterQualityRecord,
    ecotox_db: Sequence[dict[str, Any]],
    *,
    species_list: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Find ECOTOX toxicity records for chemicals detected at this location.

    Matches are based on the canonical parameter name (e.g., a metal or
    pesticide measurement) against ECOTOX chemical endpoints.

    Parameters
    ----------
    record:
        Water quality record (typically a contaminant measurement).
    ecotox_db:
        ECOTOX records as dicts with keys like ``cas_number``, ``endpoint``,
        ``concentration``, ``species_id``, etc.
    species_list:
        Optional filter for target species.

    Returns
    -------
    List of matching ECOTOX records.
    """
    param = record.canonical_param.lower()
    matches: list[dict[str, Any]] = []

    for entry in ecotox_db:
        # Match by chemical name (simplified: check if param appears in the
        # ECOTOX chemical name or CAS mapping)
        chemical = str(entry.get("chemical_name", "")).lower()
        cas = str(entry.get("cas_number", ""))

        if param not in chemical and param not in cas:
            continue

        # Optional species filter
        if species_list:
            species_id = str(entry.get("species_id", ""))
            species_name = str(entry.get("species_name", "")).lower()
            if not any(
                s.lower() in species_name or s == species_id for s in species_list
            ):
                continue

        matches.append(entry)

    return matches


# ---------------------------------------------------------------------------
# Multimodal record builder
# ---------------------------------------------------------------------------


def build_linked_record(
    wq_records: Sequence[WaterQualityRecord],
    satellite_catalog: Sequence[SatelliteObservation] | None = None,
    microbial_catalog: Sequence[MicrobialSample] | None = None,
    ecotox_db: Sequence[dict[str, Any]] | None = None,
) -> LinkedMultimodalRecord:
    """Build a linked multimodal record from co-located WQ measurements.

    Uses the first WQ record's location and timestamp as the spatial and
    temporal anchor, then searches each modality catalog for matches.

    Parameters
    ----------
    wq_records:
        Water quality records at this location/time (may be multiple params).
    satellite_catalog:
        Available satellite observations.
    microbial_catalog:
        Available microbial samples.
    ecotox_db:
        ECOTOX toxicity database records.

    Returns
    -------
    A :class:`LinkedMultimodalRecord` combining all matched modalities.
    """
    if not wq_records:
        raise ValueError("At least one WaterQualityRecord is required.")

    anchor = wq_records[0]

    # Find satellite match
    satellite: SatelliteObservation | None = None
    if satellite_catalog:
        satellite = find_satellite_match(anchor, satellite_catalog)

    # Find microbial match
    microbial: MicrobialSample | None = None
    if microbial_catalog:
        microbial = find_microbial_match(anchor, microbial_catalog)

    return LinkedMultimodalRecord(
        location_h3=anchor.h3_index,
        timestamp=anchor.timestamp,
        water_quality=list(wq_records),
        satellite=satellite,
        microbial=microbial,
        transcriptomic=None,
        behavioral=None,
    )


# ---------------------------------------------------------------------------
# Batch linking
# ---------------------------------------------------------------------------


def _group_by_h3_time(
    records: Sequence[WaterQualityRecord],
    temporal_window_hours: float = 3.0,
) -> list[list[WaterQualityRecord]]:
    """Group records by H3 cell and temporal proximity."""
    if not records:
        return []

    sorted_recs = sorted(records, key=lambda r: (r.h3_index, r.timestamp))
    groups: list[list[WaterQualityRecord]] = []
    current_group: list[WaterQualityRecord] = [sorted_recs[0]]

    window = timedelta(hours=temporal_window_hours)

    for rec in sorted_recs[1:]:
        last = current_group[-1]
        if rec.h3_index == last.h3_index and abs(rec.timestamp - last.timestamp) <= window:
            current_group.append(rec)
        else:
            groups.append(current_group)
            current_group = [rec]

    groups.append(current_group)
    return groups


def link_all(
    wq_records: Sequence[WaterQualityRecord],
    catalogs: dict[str, Any] | None = None,
    *,
    temporal_window_hours: float = 3.0,
) -> list[LinkedMultimodalRecord]:
    """Batch-link water quality records with all available modalities.

    Records are grouped by H3 cell and time window, then each group is
    linked to satellite, microbial, and ECOTOX data.

    Parameters
    ----------
    wq_records:
        All water quality records to link.
    catalogs:
        Dictionary with optional keys ``satellite``, ``microbial``,
        ``ecotox`` containing the respective data catalogs.
    temporal_window_hours:
        Time window for grouping co-located records.

    Returns
    -------
    List of :class:`LinkedMultimodalRecord` instances.
    """
    catalogs = catalogs or {}
    satellite_catalog = catalogs.get("satellite", [])
    microbial_catalog = catalogs.get("microbial", [])
    ecotox_db = catalogs.get("ecotox", [])

    groups = _group_by_h3_time(wq_records, temporal_window_hours)

    linked: list[LinkedMultimodalRecord] = []
    progress = make_progress()
    with progress:
        task = progress.add_task("Linking multimodal records", total=len(groups))
        for group in groups:
            rec = build_linked_record(
                group,
                satellite_catalog=satellite_catalog or None,
                microbial_catalog=microbial_catalog or None,
                ecotox_db=ecotox_db or None,
            )
            linked.append(rec)
            progress.advance(task)

    n_multi = sum(1 for r in linked if r.n_modalities > 1)
    logger.info(
        f"Linking complete: {len(linked)} records, "
        f"{n_multi} with multiple modalities"
    )
    return linked
