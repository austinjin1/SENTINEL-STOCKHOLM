"""
USGS NWIS sensor data download for SENTINEL.

Uses the ``dataretrieval`` package to pull instantaneous water-quality
measurements from the USGS National Water Information System (NWIS).

Supports both the legacy ``nwis`` module and the newer ``waterdata`` module.
Batches requests by state to manage API rate limits.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from sentinel.utils.config import load_config
from sentinel.utils.logging import get_logger, make_progress

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# USGS parameter codes
# ---------------------------------------------------------------------------

PARAMETER_CODES: dict[str, str] = {
    "00300": "DO",       # Dissolved Oxygen (mg/L)
    "00400": "pH",       # pH
    "00095": "SpCond",   # Specific Conductance (uS/cm)
    "00010": "Temp",     # Water Temperature (degC)
    "63680": "Turb",     # Turbidity (FNU)
    "00090": "ORP",      # Oxidation-Reduction Potential (mV)
}

US_STATES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
    "DC", "PR", "VI",
]

# Maximum days per single NWIS instantaneous-values request
_MAX_DAYS_PER_REQUEST = 120


# ---------------------------------------------------------------------------
# Station discovery
# ---------------------------------------------------------------------------


@dataclass
class StationInfo:
    """Metadata for a USGS monitoring station."""

    site_no: str
    station_name: str
    state: str
    latitude: float
    longitude: float
    huc8: str = ""
    available_params: list[str] = field(default_factory=list)


def discover_stations(
    states: Sequence[str] | None = None,
    parameter_codes: Sequence[str] | None = None,
    *,
    site_type: str = "ST",
    min_params: int = 3,
    delay_between_states: float = 1.0,
) -> list[StationInfo]:
    """Find all continuous monitoring stations with target parameters.

    Parameters
    ----------
    states:
        State codes to query (default: all US states).
    parameter_codes:
        USGS parameter codes to require (default: all 6 SENTINEL params).
    site_type:
        NWIS site type code (``"ST"`` = stream).
    min_params:
        Minimum number of target parameters a station must monitor.
    delay_between_states:
        Seconds to wait between state queries (rate limiting).

    Returns
    -------
    List of StationInfo for qualifying stations.
    """
    import dataretrieval.nwis as nwis

    states = states or US_STATES
    parameter_codes = parameter_codes or list(PARAMETER_CODES.keys())
    all_stations: dict[str, StationInfo] = {}

    progress = make_progress()
    with progress:
        task = progress.add_task("Discovering stations", total=len(states))
        for state in states:
            try:
                df, _ = nwis.get_info(
                    stateCd=state,
                    parameterCd=parameter_codes,
                    siteType=site_type,
                    siteStatus="active",
                    hasDataTypeCd="iv",
                )
                if df.empty:
                    progress.advance(task)
                    continue

                for site_no, group in df.groupby("site_no"):
                    site_no = str(site_no).strip()
                    if site_no in all_stations:
                        continue

                    row = group.iloc[0]
                    params = [
                        str(p).strip()
                        for p in group["parm_cd"].unique()
                        if str(p).strip() in parameter_codes
                    ]

                    if len(params) >= min_params:
                        all_stations[site_no] = StationInfo(
                            site_no=site_no,
                            station_name=str(row.get("station_nm", "")),
                            state=state,
                            latitude=float(row.get("dec_lat_va", 0)),
                            longitude=float(row.get("dec_long_va", 0)),
                            huc8=str(row.get("huc_cd", ""))[:8],
                            available_params=params,
                        )
            except Exception as exc:
                logger.warning(f"Station discovery failed for {state}: {exc}")

            progress.advance(task)
            if delay_between_states > 0:
                time.sleep(delay_between_states)

    stations = list(all_stations.values())
    logger.info(
        f"Discovered {len(stations)} stations with >= {min_params} target parameters"
    )
    return stations


# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------


def _download_iv_legacy(
    site_no: str,
    parameter_codes: Sequence[str],
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """Download instantaneous values using the legacy nwis module."""
    import dataretrieval.nwis as nwis

    frames: list[pd.DataFrame] = []
    current = start_date

    while current < end_date:
        chunk_end = min(current + timedelta(days=_MAX_DAYS_PER_REQUEST), end_date)
        try:
            df, _ = nwis.get_iv(
                sites=site_no,
                parameterCd=list(parameter_codes),
                start=current.isoformat(),
                end=chunk_end.isoformat(),
            )
            if not df.empty:
                frames.append(df)
        except Exception as exc:
            logger.warning(
                f"Legacy IV download failed for {site_no} "
                f"({current} - {chunk_end}): {exc}"
            )
        current = chunk_end

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames).sort_index()


def _download_iv_waterdata(
    site_no: str,
    parameter_codes: Sequence[str],
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """Download instantaneous values using the newer waterdata module."""
    try:
        from dataretrieval import waterdata
    except ImportError:
        logger.warning("waterdata module not available, falling back to nwis")
        return _download_iv_legacy(site_no, parameter_codes, start_date, end_date)

    frames: list[pd.DataFrame] = []
    current = start_date

    while current < end_date:
        chunk_end = min(current + timedelta(days=_MAX_DAYS_PER_REQUEST), end_date)
        try:
            df, _ = waterdata.get_iv(
                sites=site_no,
                parameterCd=list(parameter_codes),
                start=current.isoformat(),
                end=chunk_end.isoformat(),
            )
            if not df.empty:
                frames.append(df)
        except Exception as exc:
            logger.warning(
                f"waterdata IV download failed for {site_no} "
                f"({current} - {chunk_end}): {exc}"
            )
        current = chunk_end

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames).sort_index()


def download_station_data(
    station: StationInfo | str,
    start_date: str | date,
    end_date: str | date,
    *,
    parameter_codes: Sequence[str] | None = None,
    use_waterdata: bool = False,
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> pd.DataFrame:
    """Download instantaneous values for a single station.

    Parameters
    ----------
    station:
        StationInfo or site number string.
    start_date, end_date:
        Date range for the query.
    parameter_codes:
        Parameter codes to request (default: all 6).
    use_waterdata:
        If True, prefer the ``waterdata`` module over legacy ``nwis``.
    max_retries:
        Number of retry attempts on failure.
    retry_delay:
        Seconds between retries.
    """
    site_no = station.site_no if isinstance(station, StationInfo) else station
    if isinstance(start_date, str):
        start_date = date.fromisoformat(start_date)
    if isinstance(end_date, str):
        end_date = date.fromisoformat(end_date)
    parameter_codes = parameter_codes or list(PARAMETER_CODES.keys())

    download_fn = _download_iv_waterdata if use_waterdata else _download_iv_legacy

    for attempt in range(1, max_retries + 1):
        try:
            df = download_fn(site_no, parameter_codes, start_date, end_date)
            if not df.empty:
                logger.info(
                    f"Downloaded {len(df)} records for {site_no} "
                    f"({start_date} to {end_date})"
                )
            return df
        except Exception as exc:
            logger.warning(f"Attempt {attempt}/{max_retries} for {site_no}: {exc}")
            if attempt < max_retries:
                time.sleep(retry_delay * attempt)

    logger.error(f"All retries exhausted for station {site_no}")
    return pd.DataFrame()


def download_bulk(
    stations: Sequence[StationInfo],
    start_date: str | date,
    end_date: str | date,
    *,
    output_dir: str | Path = "data/sensor/raw",
    parameter_codes: Sequence[str] | None = None,
    use_waterdata: bool = False,
    delay_between_stations: float = 0.5,
) -> dict[str, Path]:
    """Download data for multiple stations, saving each as a Parquet file.

    Parameters
    ----------
    stations:
        List of stations to download.
    start_date, end_date:
        Date range.
    output_dir:
        Directory for output Parquet files.
    delay_between_stations:
        Seconds to wait between station queries.

    Returns
    -------
    Mapping of site_no -> output file path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Path] = {}

    progress = make_progress()
    with progress:
        task = progress.add_task("Downloading sensor data", total=len(stations))
        for station in stations:
            df = download_station_data(
                station,
                start_date,
                end_date,
                parameter_codes=parameter_codes,
                use_waterdata=use_waterdata,
            )
            if not df.empty:
                out_path = output_dir / f"{station.site_no}.parquet"
                df.to_parquet(out_path)
                results[station.site_no] = out_path
            progress.advance(task)
            if delay_between_stations > 0:
                time.sleep(delay_between_stations)

    logger.info(
        f"Downloaded data for {len(results)}/{len(stations)} stations -> {output_dir}"
    )
    return results


# ---------------------------------------------------------------------------
# Station catalog persistence
# ---------------------------------------------------------------------------


def save_station_catalog(
    stations: Sequence[StationInfo],
    path: str | Path = "data/sensor/station_catalog.json",
) -> Path:
    """Save station metadata to a JSON catalog."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for s in stations:
        records.append(
            {
                "site_no": s.site_no,
                "station_name": s.station_name,
                "state": s.state,
                "latitude": s.latitude,
                "longitude": s.longitude,
                "huc8": s.huc8,
                "available_params": s.available_params,
            }
        )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    logger.info(f"Station catalog saved: {len(records)} stations -> {path}")
    return path


def load_station_catalog(path: str | Path = "data/sensor/station_catalog.json") -> list[StationInfo]:
    """Load station metadata from a JSON catalog."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    return [StationInfo(**r) for r in records]
