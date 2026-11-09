"""
SENTINEL Environmental Justice dashboard overlay.

Overlays SENTINEL water quality anomaly alerts with EPA EJScreen
demographic data to quantify which detected anomalies affect vulnerable
communities.  Computes a *Detection Equity Index* measuring whether EJ
communities receive equitable monitoring coverage, and identifies
undermonitored communities where monitoring expansion is most needed.

This is Phase 5.3 of the SENTINEL 2.0 improvement plan.

Key Concepts
------------
- **EJ community**: census block group with EPA EJ index at or above
  the 80th percentile nationally.
- **Detection Equity Index**: ratio of the alert rate in EJ communities
  to the overall alert rate.  A value of 1.0 means perfect equity;
  values below 1.0 indicate that EJ communities are under-detected.
- **Monitoring gap**: an EJ community with no SENTINEL monitoring site
  within the default search radius (10 km).
"""

from __future__ import annotations

import csv
import json
import math
import os
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from sentinel.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EARTH_RADIUS_KM = 6_371.0
DEFAULT_SEARCH_RADIUS_KM = 10.0
EJ_PERCENTILE_THRESHOLD = 80.0
EJSCREEN_API_BASE = "https://ejscreen.epa.gov/mapper/"


# ---------------------------------------------------------------------------
# Haversine distance
# ---------------------------------------------------------------------------


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres between two lat/lon points."""
    rlat1, rlon1 = math.radians(lat1), math.radians(lon1)
    rlat2, rlon2 = math.radians(lat2), math.radians(lon2)
    dlat = rlat2 - rlat1
    dlon = rlon2 - rlon1
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# EJScreen data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EJScreenData:
    """Demographic profile for a single census block group.

    Fields mirror key columns from EPA's EJScreen dataset.  Percentages
    are expressed on a 0--100 scale.
    """

    census_block_group_id: str
    total_population: int
    pct_minority: float
    pct_low_income: float
    pct_linguistically_isolated: float
    pct_under_5: float
    pct_over_64: float
    ej_index: float
    supplemental_indices: dict[str, float] = field(default_factory=dict)
    # Centroid coordinates for spatial matching
    latitude: float = 0.0
    longitude: float = 0.0


# ---------------------------------------------------------------------------
# Annotated alert
# ---------------------------------------------------------------------------


@dataclass
class EJAnnotatedAlert:
    """A SENTINEL anomaly alert annotated with EJ demographic context."""

    site_id: str
    latitude: float
    longitude: float
    alert_info: dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    # EJ overlay fields -- None when no EJScreen data is available
    ej_data: EJScreenData | None = None
    is_ej_community: bool = False
    distance_to_block_group_km: float | None = None
    ej_warning: str = ""


# ---------------------------------------------------------------------------
# Equity report
# ---------------------------------------------------------------------------


@dataclass
class EquityReport:
    """Aggregate equity metrics across a collection of alerts."""

    total_alerts: int = 0
    alerts_in_ej_communities: int = 0
    detection_equity_index: float = 0.0
    pct_minority_affected: float = 0.0
    pct_low_income_affected: float = 0.0
    most_affected_communities: list[dict[str, Any]] = field(default_factory=list)
    monitoring_gap_communities: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Community gap
# ---------------------------------------------------------------------------


@dataclass
class CommunityGap:
    """An undermonitored community identified by the gap analysis."""

    census_block_group_id: str
    ej_index: float
    total_population: int
    pct_minority: float
    pct_low_income: float
    latitude: float
    longitude: float
    nearest_site_distance_km: float | None = None
    nearest_site_id: str = ""
    known_pollution_sources: list[str] = field(default_factory=list)
    recommendation: str = ""


# ---------------------------------------------------------------------------
# EJScreen data fetcher
# ---------------------------------------------------------------------------


class EJScreenFetcher:
    """Download and cache EPA EJScreen data.

    Supports two modes of operation:

    1. **Local cache** -- load from a CSV or JSON file previously
       downloaded from EPA.
    2. **API fetch** -- attempt to query EPA's EJScreen API (best-effort;
       the module degrades gracefully if the API is unavailable).

    Parameters
    ----------
    cache_dir:
        Directory for cached EJScreen data files.
    """

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        if cache_dir is not None:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = Path.home() / ".sentinel" / "ejscreen_cache"
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # In-memory index: block group ID -> EJScreenData
        self._data: dict[str, EJScreenData] = {}
        # Spatial index: list of (lat, lon, block_group_id)
        self._spatial_index: list[tuple[float, float, str]] = []

    # -- loading ------------------------------------------------------------

    def load_csv(self, csv_path: str | Path) -> int:
        """Load EJScreen data from a CSV file.

        Expected columns (case-insensitive): ``ID``, ``ACSTOTPOP``,
        ``MINORPCT``, ``LOWINCPCT``, ``LINGISOPCT``, ``UNDER5PCT``,
        ``OVER64PCT``, ``P_EJ_SUMM`` (or ``EJ_INDEX``), ``LAT``,
        ``LON``.

        Returns the number of records loaded.
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            logger.warning(f"EJScreen CSV not found: {csv_path}")
            return 0

        count = 0
        with open(csv_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            # Normalise header names to uppercase
            if reader.fieldnames is None:
                return 0
            header_map = {h.strip().upper(): h for h in reader.fieldnames}

            for row in reader:
                try:
                    rec = self._parse_csv_row(row, header_map)
                    self._data[rec.census_block_group_id] = rec
                    self._spatial_index.append(
                        (rec.latitude, rec.longitude, rec.census_block_group_id)
                    )
                    count += 1
                except (KeyError, ValueError) as exc:
                    logger.debug(f"Skipping malformed EJScreen row: {exc}")

        logger.info(f"Loaded {count} EJScreen records from {csv_path}")
        return count

    @staticmethod
    def _parse_csv_row(
        row: dict[str, str],
        header_map: dict[str, str],
    ) -> EJScreenData:
        """Parse a single CSV row into an :class:`EJScreenData`."""

        def _get(key: str) -> str:
            """Case-insensitive column lookup."""
            col = header_map.get(key.upper())
            if col is None:
                raise KeyError(f"Missing column: {key}")
            return row[col].strip()

        def _float(key: str, default: float = 0.0) -> float:
            try:
                return float(_get(key))
            except (KeyError, ValueError):
                return default

        def _int(key: str, default: int = 0) -> int:
            try:
                return int(float(_get(key)))
            except (KeyError, ValueError):
                return default

        # EJ index may be in different columns depending on vintage
        ej_index = _float("P_EJ_SUMM", default=-1.0)
        if ej_index < 0:
            ej_index = _float("EJ_INDEX", default=0.0)

        # Supplemental indices -- best-effort grab
        suppl: dict[str, float] = {}
        for key in ("P_PM25", "P_OZONE", "P_DSLPM", "P_TRAFFIC", "P_LEAD",
                     "P_RSEI_AIR", "P_PTRAF", "P_PWDIS", "P_PNPL", "P_PRMP"):
            val = _float(key, default=-1.0)
            if val >= 0:
                suppl[key] = val

        return EJScreenData(
            census_block_group_id=_get("ID"),
            total_population=_int("ACSTOTPOP"),
            pct_minority=_float("MINORPCT") * 100.0
            if _float("MINORPCT") <= 1.0
            else _float("MINORPCT"),
            pct_low_income=_float("LOWINCPCT") * 100.0
            if _float("LOWINCPCT") <= 1.0
            else _float("LOWINCPCT"),
            pct_linguistically_isolated=_float("LINGISOPCT") * 100.0
            if _float("LINGISOPCT") <= 1.0
            else _float("LINGISOPCT"),
            pct_under_5=_float("UNDER5PCT") * 100.0
            if _float("UNDER5PCT") <= 1.0
            else _float("UNDER5PCT"),
            pct_over_64=_float("OVER64PCT") * 100.0
            if _float("OVER64PCT") <= 1.0
            else _float("OVER64PCT"),
            ej_index=ej_index,
            supplemental_indices=suppl,
            latitude=_float("LAT"),
            longitude=_float("LON"),
        )

    def load_json(self, json_path: str | Path) -> int:
        """Load EJScreen data from a JSON file.

        Expected format: a list of objects with the same keys as the CSV
        columns (case-insensitive).

        Returns the number of records loaded.
        """
        json_path = Path(json_path)
        if not json_path.exists():
            logger.warning(f"EJScreen JSON not found: {json_path}")
            return 0

        with open(json_path, encoding="utf-8") as fh:
            data = json.load(fh)

        if isinstance(data, dict) and "features" in data:
            # GeoJSON-style
            records = data["features"]
        elif isinstance(data, list):
            records = data
        else:
            logger.warning(f"Unexpected JSON structure in {json_path}")
            return 0

        count = 0
        for item in records:
            props = item.get("properties", item) if isinstance(item, dict) else item
            try:
                rec = EJScreenData(
                    census_block_group_id=str(props["ID"]),
                    total_population=int(float(props.get("ACSTOTPOP", 0))),
                    pct_minority=float(props.get("MINORPCT", 0)),
                    pct_low_income=float(props.get("LOWINCPCT", 0)),
                    pct_linguistically_isolated=float(
                        props.get("LINGISOPCT", 0)
                    ),
                    pct_under_5=float(props.get("UNDER5PCT", 0)),
                    pct_over_64=float(props.get("OVER64PCT", 0)),
                    ej_index=float(
                        props.get("P_EJ_SUMM", props.get("EJ_INDEX", 0))
                    ),
                    supplemental_indices={
                        k: float(v)
                        for k, v in props.items()
                        if k.startswith("P_") and k != "P_EJ_SUMM"
                    },
                    latitude=float(props.get("LAT", 0)),
                    longitude=float(props.get("LON", 0)),
                )
                self._data[rec.census_block_group_id] = rec
                self._spatial_index.append(
                    (rec.latitude, rec.longitude, rec.census_block_group_id)
                )
                count += 1
            except (KeyError, ValueError, TypeError) as exc:
                logger.debug(f"Skipping malformed EJScreen JSON record: {exc}")

        logger.info(f"Loaded {count} EJScreen records from {json_path}")
        return count

    def load_cached(self) -> int:
        """Attempt to load any cached EJScreen files from the cache dir.

        Returns the total number of records loaded.
        """
        total = 0
        for fp in sorted(self.cache_dir.iterdir()):
            if fp.suffix.lower() == ".csv":
                total += self.load_csv(fp)
            elif fp.suffix.lower() == ".json":
                total += self.load_json(fp)
        if total == 0:
            logger.info(
                f"No cached EJScreen data found in {self.cache_dir}. "
                "The overlay engine will return empty annotations with "
                "warnings until EJScreen data is loaded."
            )
        return total

    # -- queries ------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        """Whether any EJScreen data has been loaded."""
        return len(self._data) > 0

    def __len__(self) -> int:
        return len(self._data)

    def get_by_id(self, block_group_id: str) -> EJScreenData | None:
        """Look up a census block group by its FIPS ID."""
        return self._data.get(block_group_id)

    def find_nearest(
        self,
        lat: float,
        lon: float,
        radius_km: float = DEFAULT_SEARCH_RADIUS_KM,
    ) -> tuple[EJScreenData | None, float | None]:
        """Find the nearest census block group centroid to a point.

        Parameters
        ----------
        lat, lon:
            Query coordinates.
        radius_km:
            Maximum search radius in kilometres.

        Returns
        -------
        ``(EJScreenData, distance_km)`` for the nearest block group
        within the search radius, or ``(None, None)`` if none found.
        """
        if not self._spatial_index:
            return None, None

        best_id: str | None = None
        best_dist = float("inf")

        for blat, blon, bid in self._spatial_index:
            d = _haversine_km(lat, lon, blat, blon)
            if d < best_dist:
                best_dist = d
                best_id = bid

        if best_id is not None and best_dist <= radius_km:
            return self._data[best_id], best_dist

        return None, None

    def find_all_within(
        self,
        lat: float,
        lon: float,
        radius_km: float = DEFAULT_SEARCH_RADIUS_KM,
    ) -> list[tuple[EJScreenData, float]]:
        """Find all census block groups within *radius_km* of a point.

        Returns a list of ``(EJScreenData, distance_km)`` sorted by
        distance.
        """
        results: list[tuple[EJScreenData, float]] = []
        for blat, blon, bid in self._spatial_index:
            d = _haversine_km(lat, lon, blat, blon)
            if d <= radius_km:
                results.append((self._data[bid], d))
        results.sort(key=lambda pair: pair[1])
        return results

    def all_records(self) -> list[EJScreenData]:
        """Return all loaded EJScreen records."""
        return list(self._data.values())


# ---------------------------------------------------------------------------
# Monitoring site descriptor (lightweight)
# ---------------------------------------------------------------------------


@dataclass
class MonitoringSite:
    """Minimal descriptor for a SENTINEL monitoring site."""

    site_id: str
    latitude: float
    longitude: float
    parameters: list[str] = field(default_factory=list)
    is_active: bool = True


# ---------------------------------------------------------------------------
# Undermonitored communities finder
# ---------------------------------------------------------------------------


class UndermonitoredCommunitiesFinder:
    """Identify EJ communities with insufficient SENTINEL monitoring.

    Criteria for flagging a community as undermonitored:

    1. High EJ index (>= ``ej_threshold`` percentile) but **no**
       SENTINEL monitoring site within ``radius_km``.
    2. Known pollution sources nearby but insufficient monitoring
       coverage.

    Parameters
    ----------
    fetcher:
        An :class:`EJScreenFetcher` with loaded data.
    monitoring_sites:
        Currently deployed SENTINEL monitoring sites.
    radius_km:
        Maximum distance to consider a site as "covering" a community.
    ej_threshold:
        Percentile threshold for identifying EJ communities.
    """

    def __init__(
        self,
        fetcher: EJScreenFetcher,
        monitoring_sites: Sequence[MonitoringSite],
        radius_km: float = DEFAULT_SEARCH_RADIUS_KM,
        ej_threshold: float = EJ_PERCENTILE_THRESHOLD,
        pollution_sources: list[dict[str, Any]] | None = None,
    ) -> None:
        self.fetcher = fetcher
        self.sites = list(monitoring_sites)
        self.radius_km = radius_km
        self.ej_threshold = ej_threshold
        self.pollution_sources = pollution_sources or []

    def find_gaps(self) -> list[CommunityGap]:
        """Identify undermonitored EJ communities.

        Returns a list of :class:`CommunityGap` records sorted by
        EJ index descending (most vulnerable first).
        """
        if not self.fetcher.is_loaded:
            logger.warning(
                "No EJScreen data loaded -- cannot identify undermonitored "
                "communities."
            )
            return []

        gaps: list[CommunityGap] = []

        for rec in self.fetcher.all_records():
            if rec.ej_index < self.ej_threshold:
                continue

            # Find nearest monitoring site
            nearest_dist: float | None = None
            nearest_id = ""
            for site in self.sites:
                d = _haversine_km(
                    rec.latitude, rec.longitude,
                    site.latitude, site.longitude,
                )
                if nearest_dist is None or d < nearest_dist:
                    nearest_dist = d
                    nearest_id = site.site_id

            is_covered = (
                nearest_dist is not None and nearest_dist <= self.radius_km
            )
            if is_covered:
                continue

            # Check for known pollution sources nearby
            nearby_sources: list[str] = []
            for src in self.pollution_sources:
                src_lat = src.get("latitude", 0.0)
                src_lon = src.get("longitude", 0.0)
                src_name = src.get("name", "Unknown source")
                d = _haversine_km(rec.latitude, rec.longitude, src_lat, src_lon)
                if d <= self.radius_km:
                    nearby_sources.append(f"{src_name} ({d:.1f} km)")

            # Build recommendation
            parts: list[str] = [
                f"EJ community {rec.census_block_group_id} "
                f"(pop. {rec.total_population:,}, "
                f"EJ index {rec.ej_index:.1f}) "
                f"has no SENTINEL monitoring within {self.radius_km} km."
            ]
            if nearest_dist is not None:
                parts.append(
                    f"Nearest site is {nearest_id} at {nearest_dist:.1f} km."
                )
            if nearby_sources:
                parts.append(
                    f"Known pollution sources nearby: {'; '.join(nearby_sources)}."
                )
            parts.append(
                "Recommend deploying a monitoring station in or near this "
                "community."
            )

            gaps.append(
                CommunityGap(
                    census_block_group_id=rec.census_block_group_id,
                    ej_index=rec.ej_index,
                    total_population=rec.total_population,
                    pct_minority=rec.pct_minority,
                    pct_low_income=rec.pct_low_income,
                    latitude=rec.latitude,
                    longitude=rec.longitude,
                    nearest_site_distance_km=nearest_dist,
                    nearest_site_id=nearest_id,
                    known_pollution_sources=nearby_sources,
                    recommendation=" ".join(parts),
                )
            )

        # Sort by EJ index descending (most vulnerable first)
        gaps.sort(key=lambda g: g.ej_index, reverse=True)

        logger.info(
            f"Identified {len(gaps)} undermonitored EJ communities "
            f"(threshold={self.ej_threshold}, radius={self.radius_km} km)"
        )
        return gaps


# ---------------------------------------------------------------------------
# EJ overlay engine
# ---------------------------------------------------------------------------


class EJOverlayEngine:
    """Core engine linking SENTINEL alerts to EJ demographics.

    Parameters
    ----------
    fetcher:
        An :class:`EJScreenFetcher` for demographic lookups.  May be
        empty -- the engine will return annotations with warnings.
    monitoring_sites:
        Known SENTINEL monitoring sites (used for equity calculations
        and gap analysis).
    search_radius_km:
        Maximum distance to match a monitoring site to a census block
        group centroid.
    ej_threshold:
        Percentile threshold for classifying a community as EJ.
    pollution_sources:
        Optional list of known pollution sources for gap analysis.
    """

    def __init__(
        self,
        fetcher: EJScreenFetcher | None = None,
        monitoring_sites: Sequence[MonitoringSite] | None = None,
        search_radius_km: float = DEFAULT_SEARCH_RADIUS_KM,
        ej_threshold: float = EJ_PERCENTILE_THRESHOLD,
        pollution_sources: list[dict[str, Any]] | None = None,
    ) -> None:
        self.fetcher = fetcher or EJScreenFetcher()
        self.sites = list(monitoring_sites or [])
        self.search_radius_km = search_radius_km
        self.ej_threshold = ej_threshold
        self.pollution_sources = pollution_sources or []

    # -- single alert annotation --------------------------------------------

    def overlay_alert(
        self,
        site_id: str,
        lat: float,
        lon: float,
        alert_info: dict[str, Any],
    ) -> EJAnnotatedAlert:
        """Annotate a single SENTINEL alert with EJ demographic context.

        Parameters
        ----------
        site_id:
            SENTINEL monitoring site identifier.
        lat, lon:
            Coordinates of the monitoring site.
        alert_info:
            Arbitrary alert payload (anomaly type, severity, parameters,
            etc.).

        Returns
        -------
        An :class:`EJAnnotatedAlert` with demographic overlay.  If
        EJScreen data is not available the ``ej_warning`` field is set
        and ``ej_data`` is *None*.
        """
        annotated = EJAnnotatedAlert(
            site_id=site_id,
            latitude=lat,
            longitude=lon,
            alert_info=alert_info,
        )

        if not self.fetcher.is_loaded:
            annotated.ej_warning = (
                "EJScreen data not loaded. Load data via "
                "EJScreenFetcher.load_csv() or load_json() to enable "
                "demographic overlay."
            )
            logger.debug(
                f"No EJScreen data for alert at site {site_id} "
                f"({lat}, {lon})"
            )
            return annotated

        ej_rec, dist = self.fetcher.find_nearest(
            lat, lon, radius_km=self.search_radius_km,
        )

        if ej_rec is None:
            annotated.ej_warning = (
                f"No census block group found within {self.search_radius_km} "
                f"km of site {site_id} ({lat}, {lon})."
            )
            return annotated

        annotated.ej_data = ej_rec
        annotated.distance_to_block_group_km = dist
        annotated.is_ej_community = ej_rec.ej_index >= self.ej_threshold

        return annotated

    # -- aggregate equity metrics -------------------------------------------

    def compute_equity_metrics(
        self,
        alerts: Sequence[EJAnnotatedAlert],
    ) -> EquityReport:
        """Compute aggregate equity metrics across a set of alerts.

        Parameters
        ----------
        alerts:
            A sequence of :class:`EJAnnotatedAlert` (typically the output
            of :meth:`overlay_alert`).

        Returns
        -------
        An :class:`EquityReport` summarising detection equity.
        """
        report = EquityReport()
        report.total_alerts = len(alerts)

        if not alerts:
            report.warnings.append("No alerts provided for equity analysis.")
            return report

        # Count alerts by EJ status and accumulate demographics
        total_minority_pop = 0
        total_low_income_pop = 0
        total_pop = 0
        community_alert_counts: dict[str, dict[str, Any]] = {}

        alerts_without_data = 0

        for alert in alerts:
            if alert.ej_data is None:
                alerts_without_data += 1
                continue

            ej = alert.ej_data
            bg_id = ej.census_block_group_id

            if alert.is_ej_community:
                report.alerts_in_ej_communities += 1

            total_pop += ej.total_population
            total_minority_pop += int(
                ej.total_population * ej.pct_minority / 100.0
            )
            total_low_income_pop += int(
                ej.total_population * ej.pct_low_income / 100.0
            )

            # Track per-community counts
            if bg_id not in community_alert_counts:
                community_alert_counts[bg_id] = {
                    "census_block_group_id": bg_id,
                    "total_population": ej.total_population,
                    "pct_minority": ej.pct_minority,
                    "pct_low_income": ej.pct_low_income,
                    "ej_index": ej.ej_index,
                    "is_ej_community": alert.is_ej_community,
                    "alert_count": 0,
                    "latitude": ej.latitude,
                    "longitude": ej.longitude,
                }
            community_alert_counts[bg_id]["alert_count"] += 1

        if alerts_without_data > 0:
            report.warnings.append(
                f"{alerts_without_data} of {len(alerts)} alerts lack "
                "EJScreen data and were excluded from equity calculations."
            )

        alerts_with_data = len(alerts) - alerts_without_data
        if alerts_with_data == 0:
            report.warnings.append(
                "No alerts had EJScreen data. Cannot compute equity metrics."
            )
            return report

        # Percentage of affected population that is minority / low-income
        if total_pop > 0:
            report.pct_minority_affected = (
                total_minority_pop / total_pop * 100.0
            )
            report.pct_low_income_affected = (
                total_low_income_pop / total_pop * 100.0
            )

        # Detection Equity Index
        #   DEI = (alerts in EJ communities / total alerts)
        #       / (EJ communities fraction in loaded data)
        # A value of 1.0 = perfectly proportional.
        # > 1.0 = EJ communities are over-represented in alerts (could
        #   indicate higher exposure *or* better coverage).
        # < 1.0 = EJ communities are under-detected.
        ej_fraction_in_data = self._ej_community_fraction()
        if ej_fraction_in_data > 0 and alerts_with_data > 0:
            ej_alert_fraction = (
                report.alerts_in_ej_communities / alerts_with_data
            )
            report.detection_equity_index = (
                ej_alert_fraction / ej_fraction_in_data
            )
        else:
            report.detection_equity_index = 0.0
            report.warnings.append(
                "Could not compute Detection Equity Index: insufficient "
                "baseline EJ data."
            )

        # Most-affected communities (top 10 by alert count)
        sorted_communities = sorted(
            community_alert_counts.values(),
            key=lambda c: c["alert_count"],
            reverse=True,
        )
        report.most_affected_communities = sorted_communities[:10]

        # Monitoring gaps
        gap_finder = UndermonitoredCommunitiesFinder(
            fetcher=self.fetcher,
            monitoring_sites=self.sites,
            radius_km=self.search_radius_km,
            ej_threshold=self.ej_threshold,
            pollution_sources=self.pollution_sources,
        )
        gaps = gap_finder.find_gaps()
        report.monitoring_gap_communities = [
            {
                "census_block_group_id": g.census_block_group_id,
                "ej_index": g.ej_index,
                "total_population": g.total_population,
                "pct_minority": g.pct_minority,
                "pct_low_income": g.pct_low_income,
                "nearest_site_distance_km": g.nearest_site_distance_km,
                "nearest_site_id": g.nearest_site_id,
                "recommendation": g.recommendation,
            }
            for g in gaps[:20]  # top 20 gaps
        ]

        return report

    # -- gap analysis -------------------------------------------------------

    def identify_undermonitored_communities(self) -> list[CommunityGap]:
        """Identify EJ communities with insufficient monitoring coverage.

        Delegates to :class:`UndermonitoredCommunitiesFinder`.

        Returns
        -------
        List of :class:`CommunityGap` sorted by EJ index descending.
        """
        finder = UndermonitoredCommunitiesFinder(
            fetcher=self.fetcher,
            monitoring_sites=self.sites,
            radius_km=self.search_radius_km,
            ej_threshold=self.ej_threshold,
            pollution_sources=self.pollution_sources,
        )
        return finder.find_gaps()

    # -- helpers ------------------------------------------------------------

    def _ej_community_fraction(self) -> float:
        """Fraction of loaded block groups classified as EJ communities."""
        if not self.fetcher.is_loaded:
            return 0.0
        records = self.fetcher.all_records()
        if not records:
            return 0.0
        ej_count = sum(
            1 for r in records if r.ej_index >= self.ej_threshold
        )
        return ej_count / len(records)

    def add_monitoring_site(self, site: MonitoringSite) -> None:
        """Register an additional monitoring site."""
        self.sites.append(site)

    def set_monitoring_sites(self, sites: Sequence[MonitoringSite]) -> None:
        """Replace the list of monitoring sites."""
        self.sites = list(sites)

    def summary(self) -> dict[str, Any]:
        """Return a summary of the engine's current state."""
        return {
            "ejscreen_records_loaded": len(self.fetcher),
            "monitoring_sites": len(self.sites),
            "search_radius_km": self.search_radius_km,
            "ej_threshold_percentile": self.ej_threshold,
            "ej_community_fraction": round(
                self._ej_community_fraction(), 4
            ),
            "pollution_sources": len(self.pollution_sources),
        }
