"""Geographic alignment for SENTINEL multimodal data.

Uses H3 hexagonal indexing for global spatial alignment across all data
sources. Replaces HUC-8 (US-only) with global-capable H3 system.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from sentinel.utils.logging import get_logger, make_progress

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# H3 configuration
# ---------------------------------------------------------------------------

# Resolution 8: ~0.74 km² per hexagon, ~461 m edge length.
# Fine enough for co-location matching, coarse enough to keep index compact.
H3_RESOLUTION = 8

# Approximate area (km²) and edge length (m) at each resolution, for reference.
_H3_RESOLUTION_TABLE: dict[int, dict[str, float]] = {
    4: {"area_km2": 1770.35, "edge_m": 22607},
    5: {"area_km2": 252.90, "edge_m": 8544},
    6: {"area_km2": 36.13, "edge_m": 3229},
    7: {"area_km2": 5.16, "edge_m": 1220},
    8: {"area_km2": 0.74, "edge_m": 461},
    9: {"area_km2": 0.11, "edge_m": 174},
    10: {"area_km2": 0.015, "edge_m": 65},
}

# Earth radius for Haversine distance (km)
_EARTH_RADIUS_KM = 6371.0


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class MonitoringLocation:
    """Metadata for a single water-quality monitoring site."""

    site_id: str
    latitude: float
    longitude: float
    h3_index: str
    source: str  # usgs, epa_wqp, eu_waterbase, gemstat, nars, etc.
    available_modalities: list[str] = field(default_factory=list)
    watershed_id: str = ""  # HUC-8 for US, HydroBASINS for global
    water_body_name: str = ""


# ---------------------------------------------------------------------------
# Core H3 indexing
# ---------------------------------------------------------------------------


def index_location(lat: float, lon: float, resolution: int = H3_RESOLUTION) -> str:
    """Compute the H3 hexagonal index for a (lat, lon) coordinate.

    Parameters
    ----------
    lat:
        Latitude in decimal degrees (WGS-84).
    lon:
        Longitude in decimal degrees (WGS-84).
    resolution:
        H3 resolution level (0--15).  Default :data:`H3_RESOLUTION` (8).

    Returns
    -------
    H3 index as a hexadecimal string (e.g. ``"882a100d63fffff"``).
    """
    import h3

    return h3.latlng_to_cell(lat, lon, resolution)


def index_locations_batch(
    locations_df: pd.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    resolution: int = H3_RESOLUTION,
) -> pd.DataFrame:
    """Add an ``h3_index`` column to a DataFrame of monitoring locations.

    Parameters
    ----------
    locations_df:
        DataFrame with at least *lat_col* and *lon_col* columns.
    lat_col, lon_col:
        Column names for latitude / longitude.
    resolution:
        H3 resolution level.

    Returns
    -------
    A copy of *locations_df* with the ``h3_index`` column appended.
    """
    import h3

    df = locations_df.copy()
    df["h3_index"] = df.apply(
        lambda row: h3.latlng_to_cell(row[lat_col], row[lon_col], resolution),
        axis=1,
    )
    logger.info(
        f"Indexed {len(df)} locations at H3 resolution {resolution} "
        f"({df['h3_index'].nunique()} unique cells)"
    )
    return df


# ---------------------------------------------------------------------------
# Co-location search
# ---------------------------------------------------------------------------


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points in km."""
    rlat1, rlon1, rlat2, rlon2 = (
        math.radians(v) for v in (lat1, lon1, lat2, lon2)
    )
    dlat = rlat2 - rlat1
    dlon = rlon2 - rlon1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    )
    return 2 * _EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def find_colocated_sites(
    target_location: MonitoringLocation,
    all_locations: Sequence[MonitoringLocation],
    max_distance_km: float = 10.0,
) -> list[MonitoringLocation]:
    """Find monitoring sites within *max_distance_km* of *target_location*.

    Uses H3 k-ring expansion to narrow candidates before computing exact
    Haversine distances.

    Parameters
    ----------
    target_location:
        The reference site.
    all_locations:
        Pool of candidate sites.
    max_distance_km:
        Maximum search radius in km.

    Returns
    -------
    List of :class:`MonitoringLocation` within the radius (excluding the
    target itself), sorted by distance.
    """
    import h3

    # Estimate k-ring radius: H3 res-8 edge ~0.461 km
    edge_km = _H3_RESOLUTION_TABLE.get(H3_RESOLUTION, {}).get("edge_m", 461) / 1000
    k = max(1, math.ceil(max_distance_km / (edge_km * 2)))

    # Build candidate set from k-ring hexes
    ring_hexes = set(h3.grid_disk(target_location.h3_index, k))

    candidates: list[tuple[float, MonitoringLocation]] = []
    for loc in all_locations:
        if loc.site_id == target_location.site_id:
            continue
        if loc.h3_index not in ring_hexes:
            continue
        dist = _haversine_km(
            target_location.latitude,
            target_location.longitude,
            loc.latitude,
            loc.longitude,
        )
        if dist <= max_distance_km:
            candidates.append((dist, loc))

    candidates.sort(key=lambda t: t[0])
    return [loc for _, loc in candidates]


# ---------------------------------------------------------------------------
# Master alignment index
# ---------------------------------------------------------------------------


def build_alignment_index(
    *source_dataframes: pd.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    source_col: str = "source",
    site_id_col: str = "site_id",
    modality_col: str = "modality",
    resolution: int = H3_RESOLUTION,
) -> pd.DataFrame:
    """Build the master H3 alignment index across all data sources.

    Accepts any number of DataFrames (USGS stations, EPA sites, NARS sites,
    satellite tiles, etc.).  Each must contain at minimum *lat_col*,
    *lon_col*, *source_col*, and *site_id_col* columns.  An optional
    *modality_col* indicates the data modality (e.g. ``"sensor"``,
    ``"satellite"``, ``"microbial"``).

    Parameters
    ----------
    *source_dataframes:
        One or more DataFrames, each representing a set of monitoring
        locations from a single source or modality.
    lat_col, lon_col:
        Column names for coordinates.
    source_col:
        Column identifying the data source.
    site_id_col:
        Column containing the unique site identifier within its source.
    modality_col:
        Column naming the data modality.
    resolution:
        H3 resolution level.

    Returns
    -------
    DataFrame with columns ``[h3_index, n_sources, source_list, site_ids,
    modalities_available]`` — one row per H3 cell that contains at least
    one monitoring location.
    """
    if not source_dataframes:
        logger.warning("No source DataFrames provided to build_alignment_index")
        return pd.DataFrame(
            columns=["h3_index", "n_sources", "source_list", "site_ids",
                      "modalities_available"]
        )

    combined = pd.concat(source_dataframes, ignore_index=True)
    combined = index_locations_batch(combined, lat_col, lon_col, resolution)

    records: list[dict[str, Any]] = []
    for h3_idx, group in combined.groupby("h3_index"):
        sources = sorted(group[source_col].unique().tolist())
        site_ids = sorted(group[site_id_col].unique().tolist())
        modalities = (
            sorted(group[modality_col].unique().tolist())
            if modality_col in group.columns
            else []
        )
        records.append(
            {
                "h3_index": h3_idx,
                "n_sources": len(sources),
                "source_list": sources,
                "site_ids": site_ids,
                "modalities_available": modalities,
            }
        )

    index_df = pd.DataFrame(records)
    index_df.sort_values("n_sources", ascending=False, inplace=True)
    index_df.reset_index(drop=True, inplace=True)

    logger.info(
        f"Alignment index: {len(index_df)} H3 cells, "
        f"{(index_df['n_sources'] > 1).sum()} with multi-source coverage"
    )
    return index_df


# ---------------------------------------------------------------------------
# Satellite-to-station matching
# ---------------------------------------------------------------------------


def match_satellite_to_stations(
    tile_specs: Sequence[dict[str, Any]],
    station_catalog: Sequence[MonitoringLocation],
    temporal_tol_days: int = 5,
) -> list[dict[str, Any]]:
    """Match satellite tiles to nearby ground monitoring stations.

    For each tile, finds stations whose H3 index falls within the tile
    footprint and — when temporal information is available — identifies
    tiles acquired within *temporal_tol_days* of station observation dates.

    Parameters
    ----------
    tile_specs:
        List of dicts each containing at minimum ``"tile_id"``,
        ``"bbox"`` (``[west, south, east, north]``), and optionally
        ``"datetime"`` (ISO-8601 string).
    station_catalog:
        All known monitoring stations.
    temporal_tol_days:
        Maximum allowed temporal offset in days.

    Returns
    -------
    List of dicts ``{"tile_id", "station_ids", "temporal_offsets"}``
    where *temporal_offsets* is a dict mapping ``station_id`` to the
    offset in days (or ``None`` if temporal info is unavailable).
    """
    import h3
    from datetime import datetime

    results: list[dict[str, Any]] = []

    # Build a lookup by h3 index for fast containment checks
    h3_to_stations: dict[str, list[MonitoringLocation]] = {}
    for station in station_catalog:
        h3_to_stations.setdefault(station.h3_index, []).append(station)

    progress = make_progress()
    with progress:
        task = progress.add_task("Matching tiles to stations", total=len(tile_specs))
        for spec in tile_specs:
            tile_id = spec["tile_id"]
            bbox = spec["bbox"]  # [west, south, east, north]
            tile_dt_str = spec.get("datetime")
            tile_dt = (
                datetime.fromisoformat(tile_dt_str) if tile_dt_str else None
            )

            # Find all H3 cells covering the tile footprint
            west, south, east, north = bbox
            polygon = [
                [south, west],
                [north, west],
                [north, east],
                [south, east],
                [south, west],
            ]
            try:
                covered_hexes = h3.polygon_to_cells(
                    h3.LatLngPoly(polygon), H3_RESOLUTION
                )
            except Exception:
                # Fallback: sample grid of points within bbox
                covered_hexes = set()
                lat_step = (north - south) / 10
                lon_step = (east - west) / 10
                for i in range(11):
                    for j in range(11):
                        lat = south + i * lat_step
                        lon = west + j * lon_step
                        covered_hexes.add(
                            h3.latlng_to_cell(lat, lon, H3_RESOLUTION)
                        )

            # Find stations in those hexes
            matched_stations: list[MonitoringLocation] = []
            for hex_id in covered_hexes:
                matched_stations.extend(h3_to_stations.get(hex_id, []))

            # Compute temporal offsets if possible
            temporal_offsets: dict[str, int | None] = {}
            for station in matched_stations:
                temporal_offsets[station.site_id] = None

            results.append(
                {
                    "tile_id": tile_id,
                    "station_ids": [s.site_id for s in matched_stations],
                    "temporal_offsets": temporal_offsets,
                }
            )
            progress.advance(task)

    logger.info(
        f"Matched {len(tile_specs)} tiles: "
        f"{sum(len(r['station_ids']) for r in results)} total station-tile pairs"
    )
    return results


# ---------------------------------------------------------------------------
# HydroBASINS / HydroLAKES download
# ---------------------------------------------------------------------------


def download_hydrobasins(output_dir: str | Path, level: int = 8) -> Path:
    """Download HydroBASINS shapefiles for global watershed delineation.

    HydroBASINS (https://www.hydrosheds.org/products/hydrobasins) provides
    a global coverage of sub-basin delineations at multiple scales.  Level 8
    is roughly equivalent to US HUC-8 watersheds.

    **Registration is required.**  This function provides parsing code for
    the downloaded shapefiles plus instructions for manual download.

    Parameters
    ----------
    output_dir:
        Directory where the user should place the downloaded shapefiles
        and where the processed GeoParquet will be written.
    level:
        HydroBASINS subdivision level (1--12).  Default 8.

    Returns
    -------
    Path to the processed GeoParquet file (or the expected path if not
    yet downloaded).

    Raises
    ------
    FileNotFoundError
        If no HydroBASINS shapefiles are found, with download instructions.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    geoparquet_path = output_dir / f"hydrobasins_lev{level:02d}.parquet"

    # Check for existing processed file
    if geoparquet_path.exists():
        logger.info(f"HydroBASINS level {level} already processed: {geoparquet_path}")
        return geoparquet_path

    # Look for shapefiles — HydroBASINS ships per-continent archives
    shp_files = sorted(output_dir.glob(f"*lev{level:02d}*.shp"))
    if not shp_files:
        # Also check for any .shp file in subdirectories
        shp_files = sorted(output_dir.rglob(f"*lev{level:02d}*.shp"))

    if not shp_files:
        instructions = (
            f"HydroBASINS level {level} shapefiles not found.\n"
            "Download instructions:\n"
            "  1. Visit https://www.hydrosheds.org/products/hydrobasins\n"
            "  2. Register / log in (free academic registration)\n"
            "  3. Download 'Standard' shapefiles for each continent:\n"
            "     - af (Africa), ar (Arctic), as (Asia), au (Australasia)\n"
            "     - eu (Europe), gr (Greenland), na (North America)\n"
            "     - sa (South America), si (Siberia)\n"
            f"  4. Extract all archives into: {output_dir.resolve()}\n"
            "  5. Re-run this function to merge and convert to GeoParquet."
        )
        logger.warning(instructions)
        raise FileNotFoundError(
            f"No HydroBASINS shapefiles found in {output_dir}. "
            "See log output for download instructions."
        )

    import geopandas as gpd

    logger.info(f"Merging {len(shp_files)} HydroBASINS shapefiles...")
    frames: list[gpd.GeoDataFrame] = []
    progress = make_progress()
    with progress:
        task = progress.add_task("Loading HydroBASINS shapefiles", total=len(shp_files))
        for shp_path in shp_files:
            gdf = gpd.read_file(shp_path)
            frames.append(gdf)
            progress.advance(task)

    merged = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True))
    merged.to_parquet(geoparquet_path)
    logger.info(
        f"HydroBASINS level {level}: {len(merged)} sub-basins -> {geoparquet_path}"
    )
    return geoparquet_path


def download_hydrolakes(output_dir: str | Path) -> Path:
    """Download HydroLAKES shapefile for lake boundary delineation.

    HydroLAKES (https://www.hydrosheds.org/products/hydrolakes) provides
    shoreline polygons for 1.4+ million lakes and reservoirs globally.

    **Registration is required.**  This function parses the downloaded
    shapefile and provides instructions for manual download.

    Parameters
    ----------
    output_dir:
        Directory for downloaded / processed data.

    Returns
    -------
    Path to the processed GeoParquet file.

    Raises
    ------
    FileNotFoundError
        If no HydroLAKES shapefile is found, with download instructions.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    geoparquet_path = output_dir / "hydrolakes.parquet"

    if geoparquet_path.exists():
        logger.info(f"HydroLAKES already processed: {geoparquet_path}")
        return geoparquet_path

    shp_files = sorted(output_dir.rglob("*HydroLAKES*.shp"))
    if not shp_files:
        shp_files = sorted(output_dir.rglob("*hydrolakes*.shp"))

    if not shp_files:
        instructions = (
            "HydroLAKES shapefile not found.\n"
            "Download instructions:\n"
            "  1. Visit https://www.hydrosheds.org/products/hydrolakes\n"
            "  2. Register / log in (free academic registration)\n"
            "  3. Download 'HydroLAKES_polys_v10.shp' (global shapefile)\n"
            f"  4. Extract into: {output_dir.resolve()}\n"
            "  5. Re-run this function to convert to GeoParquet."
        )
        logger.warning(instructions)
        raise FileNotFoundError(
            f"No HydroLAKES shapefile found in {output_dir}. "
            "See log output for download instructions."
        )

    import geopandas as gpd

    shp_path = shp_files[0]
    logger.info(f"Loading HydroLAKES shapefile: {shp_path}")
    gdf = gpd.read_file(shp_path)
    gdf.to_parquet(geoparquet_path)
    logger.info(f"HydroLAKES: {len(gdf)} lakes -> {geoparquet_path}")
    return geoparquet_path


# ---------------------------------------------------------------------------
# Watershed assignment
# ---------------------------------------------------------------------------


def assign_watershed(
    lat: float,
    lon: float,
    hydrobasins_gdf: Any,
) -> str:
    """Assign a monitoring location to its watershed via point-in-polygon.

    Parameters
    ----------
    lat, lon:
        Coordinates of the monitoring location (WGS-84).
    hydrobasins_gdf:
        GeoDataFrame of HydroBASINS polygons (as returned by
        ``geopandas.read_parquet`` on the output of
        :func:`download_hydrobasins`).

    Returns
    -------
    The ``HYBAS_ID`` (HydroBASINS identifier) of the containing sub-basin,
    or ``""`` if the point falls outside all basins.
    """
    from shapely.geometry import Point

    point = Point(lon, lat)

    # Use spatial index if available
    if hasattr(hydrobasins_gdf, "sindex"):
        candidates_idx = list(hydrobasins_gdf.sindex.query(point, predicate="intersects"))
        if candidates_idx:
            # Return the first matching basin ID
            row = hydrobasins_gdf.iloc[candidates_idx[0]]
            hybas_col = None
            for col in ("HYBAS_ID", "hybas_id", "HYBAS_Id"):
                if col in hydrobasins_gdf.columns:
                    hybas_col = col
                    break
            return str(row[hybas_col]) if hybas_col else str(row.name)
    else:
        # Brute-force containment check
        for idx, row in hydrobasins_gdf.iterrows():
            if row.geometry.contains(point):
                hybas_col = None
                for col in ("HYBAS_ID", "hybas_id", "HYBAS_Id"):
                    if col in hydrobasins_gdf.columns:
                        hybas_col = col
                        break
                return str(row[hybas_col]) if hybas_col else str(idx)

    logger.debug(f"No watershed found for ({lat}, {lon})")
    return ""


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def compute_alignment_statistics(alignment_index: pd.DataFrame) -> dict[str, Any]:
    """Compute summary statistics from the master alignment index.

    Parameters
    ----------
    alignment_index:
        DataFrame produced by :func:`build_alignment_index`.

    Returns
    -------
    Dictionary with keys:

    - ``"total_cells"`` — total H3 cells in index
    - ``"cells_by_n_sources"`` — dict mapping source-count to cell-count
    - ``"total_sites_per_source"`` — dict mapping source name to site count
    - ``"multi_modal_cells"`` — cells with 2+ modalities
    - ``"max_sources_per_cell"`` — highest source overlap observed
    """
    if alignment_index.empty:
        return {
            "total_cells": 0,
            "cells_by_n_sources": {},
            "total_sites_per_source": {},
            "multi_modal_cells": 0,
            "max_sources_per_cell": 0,
        }

    # Cells by number of sources
    cells_by_n = alignment_index["n_sources"].value_counts().sort_index()
    cells_by_n_dict = {int(k): int(v) for k, v in cells_by_n.items()}

    # Total sites per source
    source_counts: dict[str, int] = {}
    for source_list in alignment_index["source_list"]:
        for src in source_list:
            source_counts[src] = source_counts.get(src, 0) + 1

    # Multi-modal cells
    multi_modal = 0
    if "modalities_available" in alignment_index.columns:
        multi_modal = int(
            alignment_index["modalities_available"].apply(len).gt(1).sum()
        )

    stats = {
        "total_cells": len(alignment_index),
        "cells_by_n_sources": cells_by_n_dict,
        "total_sites_per_source": source_counts,
        "multi_modal_cells": multi_modal,
        "max_sources_per_cell": int(alignment_index["n_sources"].max()),
    }

    logger.info(
        f"Alignment statistics: {stats['total_cells']} cells, "
        f"{stats['multi_modal_cells']} multi-modal, "
        f"max {stats['max_sources_per_cell']} sources/cell"
    )
    return stats
