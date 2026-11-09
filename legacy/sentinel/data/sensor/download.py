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


# ---------------------------------------------------------------------------
# EPA Water Quality Portal (WQP) — discrete water quality samples
# ---------------------------------------------------------------------------

# Mapping from SENTINEL canonical parameter names to WQP characteristic names.
# WQP uses human-readable characteristic names rather than numeric codes.
WQP_CHARACTERISTIC_NAMES: dict[str, str] = {
    "DO": "Dissolved oxygen (DO)",
    "pH": "pH",
    "SpCond": "Specific conductance",
    "Temp": "Temperature, water",
    "Turb": "Turbidity",
    "ORP": "Oxidation reduction potential (ORP)",
}

# Reverse lookup: WQP characteristic name -> SENTINEL parameter name
_WQP_TO_SENTINEL: dict[str, str] = {v: k for k, v in WQP_CHARACTERISTIC_NAMES.items()}


def download_wqp(
    states: Sequence[str] | None = None,
    parameter_codes: Sequence[str] | None = None,
    start_date: str | date = "2020-01-01",
    end_date: str | date = "2024-12-31",
    output_dir: str | Path = "data/sensor/wqp",
    *,
    delay_between_states: float = 2.0,
) -> dict[str, Path]:
    """Download discrete water quality samples from the EPA Water Quality Portal.

    Uses ``dataretrieval.wqp.get_results()`` to fetch grab-sample (discrete)
    lab-analyzed measurements.  These complement USGS continuous IV data with
    lab-verified values that are typically collected weekly to monthly.

    Parameters
    ----------
    states:
        Two-letter state codes to query (default: all US states).
    parameter_codes:
        SENTINEL canonical parameter names to request (e.g. ``["DO", "pH"]``).
        Default: all 6 SENTINEL parameters.
    start_date, end_date:
        Date range for the query.
    output_dir:
        Directory for output Parquet files (one per state).
    delay_between_states:
        Seconds to wait between state queries (rate limiting).

    Returns
    -------
    Mapping of state code -> output Parquet path for states with data.
    """
    from dataretrieval import wqp

    states = states or US_STATES
    if isinstance(start_date, str):
        start_date = date.fromisoformat(start_date)
    if isinstance(end_date, str):
        end_date = date.fromisoformat(end_date)

    # Resolve which WQP characteristic names to request
    param_names = parameter_codes or list(WQP_CHARACTERISTIC_NAMES.keys())
    characteristics = [
        WQP_CHARACTERISTIC_NAMES[p]
        for p in param_names
        if p in WQP_CHARACTERISTIC_NAMES
    ]
    if not characteristics:
        raise ValueError(
            f"No valid WQP characteristic names for parameters: {param_names}"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Path] = {}

    progress = make_progress()
    with progress:
        task = progress.add_task("Downloading WQP data", total=len(states))
        for state in states:
            try:
                df, _ = wqp.get_results(
                    statecode=f"US:{state}",
                    characteristicName=characteristics,
                    startDateLo=start_date.strftime("%m-%d-%Y"),
                    startDateHi=end_date.strftime("%m-%d-%Y"),
                )
                if df is not None and not df.empty:
                    # Add a SENTINEL canonical parameter column
                    if "CharacteristicName" in df.columns:
                        df["sentinel_param"] = (
                            df["CharacteristicName"]
                            .map(_WQP_TO_SENTINEL)
                        )
                    out_path = output_dir / f"wqp_{state}.parquet"
                    df.to_parquet(out_path)
                    results[state] = out_path
                    logger.info(
                        f"WQP {state}: {len(df)} discrete samples -> {out_path}"
                    )
            except Exception as exc:
                logger.warning(f"WQP download failed for {state}: {exc}")

            progress.advance(task)
            if delay_between_states > 0:
                time.sleep(delay_between_states)

    logger.info(
        f"WQP download complete: {sum(1 for _ in results)} states with data "
        f"-> {output_dir}"
    )
    return results


# ---------------------------------------------------------------------------
# EU Waterbase — EEA water quality bulk download
# ---------------------------------------------------------------------------

# Canonical URL for the Waterbase disaggregated data CSV (rivers).
_EU_WATERBASE_URL = (
    "https://discodata.eea.europa.eu/sql?query="
    "SELECT%20*%20FROM%20%5BWISE_SOE%5D.%5Blatest%5D.%5BSOE_WISE6_WaterQuality%5D"
    "&p=1&nrOfHits=100000&mail=null&schema=null"
)

# Fallback: direct bulk CSV download from EEA data catalogue.
_EU_WATERBASE_BULK_URL = (
    "https://cmshare.eea.europa.eu/s/YbfYWHoXxXLSm6n/download"
)

# Mapping of EU Waterbase determinand codes to SENTINEL canonical names.
EU_DETERMINAND_TO_SENTINEL: dict[str, str] = {
    "EEA_3131-01-2": "DO",     # Dissolved oxygen
    "EEA_3121-01-2": "pH",     # pH
    "EEA_3123-01-5": "SpCond", # Electrical conductivity (proxy for SpCond)
    "EEA_3101-01-9": "Temp",   # Water temperature
    "EEA_3126-01-2": "Turb",   # Turbidity
    "EEA_3145-01-2": "ORP",    # Oxidation-reduction potential
}


def download_eu_waterbase(
    output_dir: str | Path = "data/sensor/eu_waterbase",
    countries: Sequence[str] | None = None,
) -> Path:
    """Download EU Waterbase water quality data from the EEA.

    Downloads the Waterbase bulk CSV and optionally filters to specific
    countries.  The EU Waterbase contains discrete water quality monitoring
    data from the European Environment Agency (EEA) covering EU member states.

    Parameters
    ----------
    output_dir:
        Directory for the downloaded / processed file.
    countries:
        Optional ISO-2 country codes to keep (e.g. ``["SE", "DE", "NL"]``).
        If *None*, all countries are retained.

    Returns
    -------
    Path to the saved Parquet file.
    """
    import urllib.request

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "waterbase_raw.csv"
    parquet_path = output_dir / "waterbase.parquet"

    # Download bulk CSV
    logger.info("Downloading EU Waterbase bulk CSV (this may take several minutes)...")
    try:
        urllib.request.urlretrieve(_EU_WATERBASE_BULK_URL, str(csv_path))
    except Exception as exc:
        logger.warning(
            f"Bulk download failed ({exc}); trying DiscoData SQL endpoint..."
        )
        urllib.request.urlretrieve(_EU_WATERBASE_URL, str(csv_path))

    logger.info(f"Parsing EU Waterbase CSV: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)

    # Filter to target countries if specified
    country_col = None
    for candidate in ("countryCode", "CountryCode", "country_code", "countrycode"):
        if candidate in df.columns:
            country_col = candidate
            break
    if countries and country_col:
        df = df[df[country_col].isin(countries)]
        logger.info(f"Filtered to {len(df)} rows for countries: {countries}")

    # Map determinand codes to SENTINEL parameter names
    determinand_col = None
    for candidate in (
        "observedPropertyDeterminandCode",
        "Determinand",
        "determinand_code",
    ):
        if candidate in df.columns:
            determinand_col = candidate
            break
    if determinand_col:
        df["sentinel_param"] = df[determinand_col].map(EU_DETERMINAND_TO_SENTINEL)

    df.to_parquet(parquet_path)
    logger.info(f"EU Waterbase saved: {len(df)} records -> {parquet_path}")

    # Clean up raw CSV to save disk space
    try:
        csv_path.unlink()
    except OSError:
        pass

    return parquet_path


# ---------------------------------------------------------------------------
# GEMS/Water (GEMStat) — global water quality data
# ---------------------------------------------------------------------------

GEMSTAT_SENTINEL_PARAMS: dict[str, str] = {
    "Dissolved Oxygen": "DO",
    "pH": "pH",
    "Electrical Conductivity": "SpCond",
    "Water Temperature": "Temp",
    "Turbidity": "Turb",
    "Oxidation Reduction Potential": "ORP",
}


def download_gemstat(
    output_dir: str | Path = "data/sensor/gemstat",
) -> Path:
    """Download GEMStat global water quality data.

    GEMStat (https://gemstat.org/) is the Global Water Quality database
    maintained by UNEP.  **Registration is required** to access the data
    downloads — the data cannot be fetched programmatically without an
    authenticated session.

    This function provides:
      1. Instructions for manual download and expected file placement.
      2. Parsing of the downloaded CSV into a SENTINEL-compatible Parquet
         file with canonical parameter names.

    Parameters
    ----------
    output_dir:
        Directory where the user should place the downloaded GEMStat CSV
        and where the processed Parquet will be written.

    Returns
    -------
    Path to the processed Parquet file (or the expected path if not yet
    downloaded).

    Raises
    ------
    FileNotFoundError
        If no GEMStat CSV is found in *output_dir* — with instructions on
        how to obtain the data.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = output_dir / "gemstat.parquet"

    # Look for an existing download (any CSV in the output directory)
    csv_files = sorted(output_dir.glob("*.csv"))
    if not csv_files:
        instructions = (
            "GEMStat data requires manual download from https://gemstat.org/.\n"
            "Steps:\n"
            "  1. Register / log in at https://gemstat.org/\n"
            "  2. Navigate to Data > Data Download\n"
            "  3. Select parameters: Dissolved Oxygen, pH, Electrical "
            "Conductivity, Water Temperature, Turbidity, ORP\n"
            "  4. Select desired countries / river basins\n"
            "  5. Download the CSV export\n"
            f"  6. Place the CSV file in: {output_dir.resolve()}\n"
            "  7. Re-run this function to parse the CSV into Parquet."
        )
        logger.warning(instructions)
        raise FileNotFoundError(
            f"No GEMStat CSV found in {output_dir}. "
            "See log output for download instructions."
        )

    # Parse the first CSV found
    csv_path = csv_files[0]
    logger.info(f"Parsing GEMStat CSV: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False, encoding="utf-8-sig")

    # Map parameter names to SENTINEL canonical names
    param_col = None
    for candidate in (
        "Parameter",
        "parameter",
        "ParameterDescription",
        "parameter_description",
    ):
        if candidate in df.columns:
            param_col = candidate
            break
    if param_col:
        df["sentinel_param"] = df[param_col].map(GEMSTAT_SENTINEL_PARAMS)

    df.to_parquet(parquet_path)
    logger.info(f"GEMStat saved: {len(df)} records -> {parquet_path}")
    return parquet_path
