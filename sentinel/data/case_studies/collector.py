"""Collect multi-modal data packages for historical contamination events.

Orchestrates data retrieval from USGS NWIS sensors, Sentinel-2 satellite
imagery, and EPA emergency response records for each validation event
defined in :mod:`sentinel.data.case_studies.events`.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from sentinel.data.case_studies.events import (
    HISTORICAL_EVENTS,
    ContaminationEvent,
)
from sentinel.utils.logging import get_logger, make_progress

logger = get_logger(__name__)

# Earth radius for bounding-box expansion
_EARTH_RADIUS_KM = 6371.0


# ---------------------------------------------------------------------------
# Data package dataclass
# ---------------------------------------------------------------------------


@dataclass
class EventDataPackage:
    """Collected multi-modal data for one contamination event."""

    event: ContaminationEvent
    sensor_data: dict[str, Path] = field(default_factory=dict)   # station_id -> parquet
    satellite_tiles: list[Path] = field(default_factory=list)
    microbial_samples: list[Path] = field(default_factory=list)
    epa_records: list[dict[str, Any]] = field(default_factory=list)
    collection_window: tuple[str, str] = ("", "")  # (start_date, end_date)
    stations_found: int = 0
    tiles_found: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_collection_window(
    event: ContaminationEvent,
    window_days_before: int = 30,
    window_days_after: int = 60,
) -> tuple[date, date]:
    """Compute the data collection window around an event.

    For recurring events, uses the most recent year in the ``year`` field
    as the reference.
    """
    if isinstance(event.year, list):
        ref_year = max(event.year)
    else:
        ref_year = event.year

    if event.documented_onset != "recurring":
        onset = date.fromisoformat(event.documented_onset)
    else:
        # For recurring events, use June 1 of the reference year as
        # a reasonable midpoint of the bloom / contamination season
        onset = date(ref_year, 6, 1)

    start = onset - timedelta(days=window_days_before)
    end = onset + timedelta(days=window_days_after)
    return start, end


def _radius_to_bbox(
    center_lat: float,
    center_lon: float,
    radius_km: float,
) -> tuple[float, float, float, float]:
    """Convert center + radius to a bounding box (west, south, east, north)."""
    lat_delta = math.degrees(radius_km / _EARTH_RADIUS_KM)
    lon_delta = math.degrees(
        radius_km / (_EARTH_RADIUS_KM * math.cos(math.radians(center_lat)))
    )
    return (
        center_lon - lon_delta,
        center_lat - lat_delta,
        center_lon + lon_delta,
        center_lat + lat_delta,
    )


# ---------------------------------------------------------------------------
# Sensor data collection
# ---------------------------------------------------------------------------


def collect_sensor_data(
    event: ContaminationEvent,
    radius_km: float = 50.0,
    window_days_before: int = 30,
    window_days_after: int = 60,
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Collect USGS NWIS sensor data around a contamination event.

    Finds all USGS stations within *radius_km* of the event center and
    downloads instantaneous values for the time window spanning
    *window_days_before* through *window_days_after* relative to the
    documented onset date.

    Parameters
    ----------
    event:
        The contamination event to collect data for.
    radius_km:
        Search radius in km from the event center.
    window_days_before:
        Days before onset to begin data collection.
    window_days_after:
        Days after onset to end data collection.
    output_dir:
        Override output directory.  Default: ``data/case_studies/<event_id>/sensor/``.

    Returns
    -------
    Mapping of ``site_no`` -> Parquet file path.
    """
    from sentinel.data.sensor.download import (
        StationInfo,
        discover_stations,
        download_station_data,
    )

    start, end = _compute_collection_window(
        event, window_days_before, window_days_after
    )

    if output_dir is None:
        output_dir = Path("data") / "case_studies" / event.event_id / "sensor"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find stations near the event using NWIS bounding-box query
    bbox = _radius_to_bbox(event.center_lat, event.center_lon, radius_km)
    logger.info(
        f"[{event.event_id}] Searching USGS stations within {radius_km} km "
        f"of ({event.center_lat:.4f}, {event.center_lon:.4f})"
    )

    try:
        import dataretrieval.nwis as nwis

        df, _ = nwis.get_info(
            bBox=f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
            siteType="ST",
            siteStatus="active",
            hasDataTypeCd="iv",
        )
    except Exception as exc:
        logger.warning(f"[{event.event_id}] Station discovery failed: {exc}")
        return {}

    if df is None or df.empty:
        logger.warning(f"[{event.event_id}] No USGS stations found within radius")
        return {}

    # Deduplicate by site number
    site_numbers = df["site_no"].unique().tolist() if "site_no" in df.columns else []
    logger.info(f"[{event.event_id}] Found {len(site_numbers)} USGS stations")

    results: dict[str, Path] = {}
    progress = make_progress()
    with progress:
        task = progress.add_task(
            f"Downloading sensor data ({event.event_id})",
            total=len(site_numbers),
        )
        for site_no in site_numbers:
            site_no = str(site_no).strip()
            try:
                df_iv = download_station_data(
                    site_no,
                    start.isoformat(),
                    end.isoformat(),
                )
                if not df_iv.empty:
                    out_path = output_dir / f"{site_no}.parquet"
                    df_iv.to_parquet(out_path)
                    results[site_no] = out_path
            except Exception as exc:
                logger.warning(
                    f"[{event.event_id}] Failed to download {site_no}: {exc}"
                )
            progress.advance(task)

    logger.info(
        f"[{event.event_id}] Sensor data: {len(results)} stations "
        f"({start} to {end})"
    )
    return results


# ---------------------------------------------------------------------------
# Satellite data collection
# ---------------------------------------------------------------------------


def collect_satellite_data(
    event: ContaminationEvent,
    window_days_before: int = 30,
    window_days_after: int = 60,
    cloud_max: float = 30.0,
    output_dir: str | Path | None = None,
    backend: str = "planetary_computer",
) -> list[Path]:
    """Collect Sentinel-2 imagery around a contamination event.

    Parameters
    ----------
    event:
        The contamination event.
    window_days_before, window_days_after:
        Time window relative to documented onset.
    cloud_max:
        Maximum cloud cover percentage.
    output_dir:
        Override output directory.  Default: ``data/case_studies/<event_id>/satellite/``.
    backend:
        Satellite download backend (``"gee"`` or ``"planetary_computer"``).

    Returns
    -------
    List of paths to downloaded tile files.
    """
    if not event.sentinel2_available:
        logger.info(
            f"[{event.event_id}] Sentinel-2 not available (pre-June 2015 event)"
        )
        return []

    from sentinel.data.satellite.download import AOI, download_satellite

    start, end = _compute_collection_window(
        event, window_days_before, window_days_after
    )

    if output_dir is None:
        output_dir = Path("data") / "case_studies" / event.event_id / "satellite"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    aoi = AOI(
        west=event.location_bbox[0],
        south=event.location_bbox[1],
        east=event.location_bbox[2],
        north=event.location_bbox[3],
    )

    logger.info(
        f"[{event.event_id}] Downloading Sentinel-2 imagery: "
        f"bbox={event.location_bbox}, {start} to {end}, cloud<{cloud_max}%"
    )

    try:
        result = download_satellite(
            aoi=aoi,
            start_date=start,
            end_date=end,
            backend=backend,
            output_dir=str(output_dir),
            cloud_max_pct=cloud_max,
            include_thermal=True,
        )
        paths = [Path(p) for p in result.get("paths", [])]
        logger.info(f"[{event.event_id}] Satellite tiles: {len(paths)}")
        return paths
    except Exception as exc:
        logger.warning(f"[{event.event_id}] Satellite download failed: {exc}")
        return []


# ---------------------------------------------------------------------------
# EPA emergency response records
# ---------------------------------------------------------------------------


def collect_epa_records(event: ContaminationEvent) -> list[dict[str, Any]]:
    """Search EPA Emergency Response for event records.

    Queries the EPA Emergency Response system
    (https://response.epa.gov/) for incident reports and violation records
    related to the event.

    Parameters
    ----------
    event:
        The contamination event.

    Returns
    -------
    List of record dicts, each with at minimum ``"source"``,
    ``"description"``, ``"date"``, and ``"url"`` keys.
    """
    import urllib.request
    import urllib.error

    records: list[dict[str, Any]] = []

    # Record the known EPA response URL
    if event.epa_response_url:
        records.append(
            {
                "source": "epa_response",
                "description": f"EPA Emergency Response page for {event.name}",
                "date": event.documented_onset,
                "url": event.epa_response_url,
                "event_id": event.event_id,
            }
        )

    # Attempt to query EPA ECHO (Enforcement and Compliance History Online)
    # for facilities in the event's bounding box
    echo_base_url = "https://echodata.epa.gov/echo/echo_rest_services.get_facilities"
    bbox = event.location_bbox
    echo_params = (
        f"?output=JSON"
        f"&p_c1lat={bbox[1]}&p_c1lon={bbox[0]}"
        f"&p_c2lat={bbox[3]}&p_c2lon={bbox[2]}"
        f"&p_act=Y"  # active facilities only
    )
    echo_url = echo_base_url + echo_params

    try:
        req = urllib.request.Request(
            echo_url,
            headers={"User-Agent": "SENTINEL/1.0 (research)"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        facilities = data.get("Results", {}).get("Facilities", [])
        for fac in facilities[:50]:  # Cap at 50 facilities
            records.append(
                {
                    "source": "epa_echo",
                    "description": (
                        f"ECHO facility: {fac.get('FacName', 'Unknown')} "
                        f"({fac.get('FacCity', '')}, {fac.get('FacState', '')})"
                    ),
                    "date": event.documented_onset,
                    "url": (
                        f"https://echo.epa.gov/detailed-facility-report?"
                        f"fid={fac.get('RegistryID', '')}"
                    ),
                    "event_id": event.event_id,
                    "facility_name": fac.get("FacName", ""),
                    "registry_id": fac.get("RegistryID", ""),
                    "violations": fac.get("CWAViolStatus", ""),
                }
            )
        logger.info(
            f"[{event.event_id}] EPA ECHO: {len(facilities)} facilities in bbox"
        )
    except (urllib.error.URLError, json.JSONDecodeError, Exception) as exc:
        logger.warning(f"[{event.event_id}] EPA ECHO query failed: {exc}")

    logger.info(f"[{event.event_id}] Collected {len(records)} EPA records")
    return records


# ---------------------------------------------------------------------------
# Full event package collection
# ---------------------------------------------------------------------------


def collect_event_package(
    event: ContaminationEvent,
    output_dir: str | Path = "data/case_studies",
    *,
    radius_km: float = 50.0,
    window_days_before: int = 30,
    window_days_after: int = 60,
    cloud_max: float = 30.0,
    satellite_backend: str = "planetary_computer",
    skip_satellite: bool = False,
    skip_sensor: bool = False,
    skip_epa: bool = False,
) -> EventDataPackage:
    """Collect the full multi-modal data package for a single event.

    Creates an organized directory under
    ``<output_dir>/<event_id>/`` containing sensor data, satellite
    tiles, EPA records, and a ``manifest.json`` summarizing the
    collection.

    Parameters
    ----------
    event:
        The contamination event to collect data for.
    output_dir:
        Root output directory for case study data.
    radius_km:
        Search radius for USGS stations.
    window_days_before, window_days_after:
        Time window relative to onset.
    cloud_max:
        Maximum cloud cover for satellite imagery.
    satellite_backend:
        Backend for satellite downloads.
    skip_satellite, skip_sensor, skip_epa:
        Flags to skip individual data collection steps.

    Returns
    -------
    :class:`EventDataPackage` with paths to all collected data.
    """
    output_dir = Path(output_dir)
    event_dir = output_dir / event.event_id
    event_dir.mkdir(parents=True, exist_ok=True)

    start, end = _compute_collection_window(
        event, window_days_before, window_days_after
    )

    logger.info(
        f"Collecting data package for '{event.name}' "
        f"({start.isoformat()} to {end.isoformat()})"
    )

    package = EventDataPackage(
        event=event,
        collection_window=(start.isoformat(), end.isoformat()),
    )

    # 1. Sensor data
    if not skip_sensor:
        sensor_dir = event_dir / "sensor"
        package.sensor_data = collect_sensor_data(
            event,
            radius_km=radius_km,
            window_days_before=window_days_before,
            window_days_after=window_days_after,
            output_dir=sensor_dir,
        )
        package.stations_found = len(package.sensor_data)

    # 2. Satellite data
    if not skip_satellite:
        satellite_dir = event_dir / "satellite"
        package.satellite_tiles = collect_satellite_data(
            event,
            window_days_before=window_days_before,
            window_days_after=window_days_after,
            cloud_max=cloud_max,
            output_dir=satellite_dir,
            backend=satellite_backend,
        )
        package.tiles_found = len(package.satellite_tiles)

    # 3. EPA records
    if not skip_epa:
        package.epa_records = collect_epa_records(event)
        epa_path = event_dir / "epa_records.json"
        with open(epa_path, "w", encoding="utf-8") as f:
            json.dump(package.epa_records, f, indent=2, default=str)

    # 4. Save manifest
    manifest = {
        "event_id": event.event_id,
        "event_name": event.name,
        "collection_window": list(package.collection_window),
        "stations_found": package.stations_found,
        "tiles_found": package.tiles_found,
        "epa_records_count": len(package.epa_records),
        "sensor_files": {
            k: str(v) for k, v in package.sensor_data.items()
        },
        "satellite_files": [str(p) for p in package.satellite_tiles],
        "microbial_files": [str(p) for p in package.microbial_samples],
    }
    manifest_path = event_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"[{event.event_id}] Manifest saved: {manifest_path}")

    return package


# ---------------------------------------------------------------------------
# Batch collection
# ---------------------------------------------------------------------------


def collect_all_events(
    output_dir: str | Path = "data/case_studies",
    events: Sequence[ContaminationEvent] | None = None,
    *,
    skip_existing: bool = True,
    radius_km: float = 50.0,
    window_days_before: int = 30,
    window_days_after: int = 60,
    cloud_max: float = 30.0,
    satellite_backend: str = "planetary_computer",
    skip_satellite: bool = False,
    skip_sensor: bool = False,
    skip_epa: bool = False,
) -> list[EventDataPackage]:
    """Batch-collect data packages for multiple contamination events.

    Parameters
    ----------
    output_dir:
        Root output directory.
    events:
        List of events to collect.  Default: :data:`HISTORICAL_EVENTS`.
    skip_existing:
        If *True*, skip events that already have a ``manifest.json``.
    radius_km:
        Search radius for USGS stations.
    window_days_before, window_days_after:
        Time window relative to onset.
    cloud_max:
        Maximum cloud cover for satellite imagery.
    satellite_backend:
        Backend for satellite downloads.
    skip_satellite, skip_sensor, skip_epa:
        Flags to skip individual data collection steps.

    Returns
    -------
    List of :class:`EventDataPackage` for all successfully collected events.
    """
    events = events or HISTORICAL_EVENTS
    output_dir = Path(output_dir)
    packages: list[EventDataPackage] = []

    progress = make_progress()
    with progress:
        task = progress.add_task(
            "Collecting event packages", total=len(events)
        )
        for event in events:
            event_dir = output_dir / event.event_id
            manifest_path = event_dir / "manifest.json"

            if skip_existing and manifest_path.exists():
                logger.info(
                    f"[{event.event_id}] Skipping (manifest already exists)"
                )
                progress.advance(task)
                continue

            try:
                pkg = collect_event_package(
                    event,
                    output_dir=output_dir,
                    radius_km=radius_km,
                    window_days_before=window_days_before,
                    window_days_after=window_days_after,
                    cloud_max=cloud_max,
                    satellite_backend=satellite_backend,
                    skip_satellite=skip_satellite,
                    skip_sensor=skip_sensor,
                    skip_epa=skip_epa,
                )
                packages.append(pkg)
            except Exception as exc:
                logger.error(f"[{event.event_id}] Collection failed: {exc}")

            progress.advance(task)

    logger.info(
        f"Batch collection complete: {len(packages)}/{len(events)} events "
        f"collected -> {output_dir}"
    )
    return packages
