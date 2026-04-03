"""
SENTINEL-DB H3 hexagonal spatial indexing.

Provides spatial operations for the unified database using Uber's H3
hierarchical hexagonal grid system.  Default resolution 8 yields cells
of ~0.74 km² — a good match for co-registering satellite pixels to
in-situ monitoring stations.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any, Sequence, TypeVar

import h3

from sentinel.data.sentinel_db.schema import (
    QualityTier,
    SatelliteObservation,
    WaterQualityRecord,
)
from sentinel.utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")

# Default H3 resolution: ~0.74 km² hexagons
DEFAULT_RESOLUTION = 8


# ---------------------------------------------------------------------------
# Core H3 wrappers
# ---------------------------------------------------------------------------


def latlon_to_h3(lat: float, lon: float, resolution: int = DEFAULT_RESOLUTION) -> str:
    """Convert latitude/longitude to an H3 cell index.

    Parameters
    ----------
    lat:
        Latitude in decimal degrees [-90, 90].
    lon:
        Longitude in decimal degrees [-180, 180].
    resolution:
        H3 resolution level (0–15).  Default 8 (~0.74 km²).

    Returns
    -------
    H3 cell index string (e.g. ``'882a100d63fffff'``).
    """
    return h3.latlng_to_cell(lat, lon, resolution)


def h3_to_latlon(h3_index: str) -> tuple[float, float]:
    """Return the center (lat, lon) of an H3 cell.

    Parameters
    ----------
    h3_index:
        H3 cell index string.

    Returns
    -------
    Tuple of (latitude, longitude).
    """
    lat, lon = h3.cell_to_latlng(h3_index)
    return lat, lon


# ---------------------------------------------------------------------------
# Spatial search
# ---------------------------------------------------------------------------


def find_nearby_records(
    target_h3: str,
    records: Sequence[Any],
    *,
    max_ring: int = 2,
    h3_attr: str = "h3_index",
) -> list[Any]:
    """Find records within *max_ring* H3 k-rings of *target_h3*.

    Parameters
    ----------
    target_h3:
        Center H3 cell index.
    records:
        Sequence of objects with an ``h3_index`` attribute (or dict key).
    max_ring:
        Maximum k-ring distance (0 = same cell only, 1 = immediate
        neighbors, 2 = two rings out, etc.).
    h3_attr:
        Attribute name (or dict key) holding the H3 index.

    Returns
    -------
    List of matching records.
    """
    # Build the set of H3 cells in the k-ring neighborhood
    neighborhood = h3.grid_disk(target_h3, max_ring)
    neighborhood_set = set(neighborhood)

    matches: list[Any] = []
    for rec in records:
        rec_h3 = getattr(rec, h3_attr, None)
        if rec_h3 is None and isinstance(rec, dict):
            rec_h3 = rec.get(h3_attr)
        if rec_h3 in neighborhood_set:
            matches.append(rec)

    return matches


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def _get_timestamp(rec: Any) -> datetime:
    """Extract timestamp from a record (Pydantic model or dict)."""
    if hasattr(rec, "timestamp"):
        return rec.timestamp
    if isinstance(rec, dict):
        return rec["timestamp"]
    raise TypeError(f"Cannot extract timestamp from {type(rec)}")


def _get_h3(rec: Any) -> str:
    """Extract H3 index from a record."""
    if hasattr(rec, "h3_index"):
        return rec.h3_index
    if isinstance(rec, dict):
        return rec["h3_index"]
    raise TypeError(f"Cannot extract h3_index from {type(rec)}")


def _get_quality_tier(rec: Any) -> QualityTier:
    """Extract quality tier from a record."""
    if hasattr(rec, "quality_tier"):
        return rec.quality_tier
    if isinstance(rec, dict):
        return QualityTier(rec["quality_tier"])
    return QualityTier.Q4


def _get_canonical_param(rec: Any) -> str:
    """Extract canonical parameter name from a record."""
    if hasattr(rec, "canonical_param"):
        return rec.canonical_param
    if isinstance(rec, dict):
        return rec.get("canonical_param", "")
    return ""


_TIER_RANK = {QualityTier.Q1: 0, QualityTier.Q2: 1, QualityTier.Q3: 2, QualityTier.Q4: 3}


def deduplicate_records(
    records: Sequence[WaterQualityRecord],
    *,
    temporal_window_hours: float = 3.0,
) -> list[WaterQualityRecord]:
    """Remove duplicate records within the same H3 cell and time window.

    When duplicates are found for the same parameter, the record with the
    highest quality tier (lowest Q number) is kept.

    Parameters
    ----------
    records:
        Input records to deduplicate.
    temporal_window_hours:
        Time window within which records at the same H3 cell and parameter
        are considered duplicates.

    Returns
    -------
    Deduplicated list of records.
    """
    if not records:
        return []

    # Sort by (h3, param, timestamp)
    sorted_recs = sorted(
        records,
        key=lambda r: (_get_h3(r), _get_canonical_param(r), _get_timestamp(r)),
    )

    window = timedelta(hours=temporal_window_hours)
    kept: list[WaterQualityRecord] = []

    for rec in sorted_recs:
        h3_idx = _get_h3(rec)
        ts = _get_timestamp(rec)
        param = _get_canonical_param(rec)
        tier = _get_quality_tier(rec)

        # Check if the last kept record is a duplicate
        if kept:
            last = kept[-1]
            last_h3 = _get_h3(last)
            last_ts = _get_timestamp(last)
            last_param = _get_canonical_param(last)

            if last_h3 == h3_idx and last_param == param and abs(ts - last_ts) <= window:
                # Duplicate: keep the higher quality one
                last_tier = _get_quality_tier(last)
                if _TIER_RANK.get(tier, 3) < _TIER_RANK.get(last_tier, 3):
                    kept[-1] = rec
                continue

        kept.append(rec)

    n_removed = len(records) - len(kept)
    if n_removed > 0:
        logger.info(
            f"Deduplication: {n_removed} duplicates removed, "
            f"{len(kept)}/{len(records)} records retained"
        )
    return kept


# ---------------------------------------------------------------------------
# Satellite co-registration
# ---------------------------------------------------------------------------


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance between two points in meters."""
    R = 6_371_000.0  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def coregister_satellite_to_station(
    satellite_obs: Sequence[SatelliteObservation],
    stations: Sequence[WaterQualityRecord],
    *,
    max_distance_m: float = 500.0,
) -> list[tuple[SatelliteObservation, WaterQualityRecord, float]]:
    """Match satellite observations to nearby in-situ stations.

    For each satellite observation, find stations within *max_distance_m*
    meters.  Returns tuples of ``(satellite_obs, station_record, distance_m)``.

    Parameters
    ----------
    satellite_obs:
        Satellite observations with H3 indices.
    stations:
        In-situ station records with lat/lon coordinates.
    max_distance_m:
        Maximum matching distance in meters.

    Returns
    -------
    List of (satellite, station, distance) tuples.
    """
    matches: list[tuple[SatelliteObservation, WaterQualityRecord, float]] = []

    for sat in satellite_obs:
        sat_lat, sat_lon = h3_to_latlon(sat.h3_index)

        for station in stations:
            dist = _haversine_m(sat_lat, sat_lon, station.latitude, station.longitude)
            if dist <= max_distance_m:
                matches.append((sat, station, dist))

    logger.info(
        f"Co-registration: {len(matches)} satellite-station pairs "
        f"within {max_distance_m}m"
    )
    return matches
