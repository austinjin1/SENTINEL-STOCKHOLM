"""
SENTINEL-DB multi-source data ingest.

Orchestrates download and harmonization from major water quality databases:
- EPA Water Quality Portal (WQP) via ``dataretrieval``
- EU Waterbase (EEA WISE / DiscoData)
- GEMStat (UN GEMS/Water)
- GRQA harmonized dataset (Zenodo)
- FreshWater Watch (citizen science)

Each ingest function: downloads raw data, applies ontology mapping, assigns
quality tiers, computes H3 indices, and returns standardized records.
"""

from __future__ import annotations

import io
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import requests

from sentinel.data.sentinel_db.ontology import harmonize_unit, resolve_parameter
from sentinel.data.sentinel_db.quality import assign_quality_tier
from sentinel.data.sentinel_db.schema import QualityTier, WaterQualityRecord
from sentinel.data.sentinel_db.spatial import latlon_to_h3
from sentinel.utils.logging import get_logger, make_progress

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(val: Any) -> float | None:
    """Convert *val* to float, returning None on failure."""
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _safe_datetime(val: Any) -> datetime | None:
    """Parse a datetime value, returning None on failure."""
    if val is None:
        return None
    if isinstance(val, datetime):
        return val
    if isinstance(val, pd.Timestamp):
        return val.to_pydatetime()
    try:
        return pd.to_datetime(val).to_pydatetime()
    except Exception:
        return None


def _row_to_wq_record(
    row: dict[str, Any],
    *,
    param_col: str,
    value_col: str,
    unit_col: str,
    time_col: str,
    lat_col: str,
    lon_col: str,
    site_col: str,
    source: str,
) -> WaterQualityRecord | None:
    """Convert a raw data row to a WaterQualityRecord.

    Returns None if the row cannot be converted (missing value, unknown
    parameter, bad coordinates, etc.).
    """
    raw_param = str(row.get(param_col, "")).strip()
    if not raw_param:
        return None

    canonical = resolve_parameter(raw_param)
    if canonical is None:
        return None

    value = _safe_float(row.get(value_col))
    if value is None:
        return None

    raw_unit = str(row.get(unit_col, "")).strip()
    timestamp = _safe_datetime(row.get(time_col))
    if timestamp is None:
        return None

    lat = _safe_float(row.get(lat_col))
    lon = _safe_float(row.get(lon_col))
    if lat is None or lon is None or not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        return None

    # Unit harmonization
    try:
        harmonized_value = harmonize_unit(value, raw_unit, canonical.canonical_unit)
    except ValueError:
        harmonized_value = value  # Keep original if conversion fails

    site_id = str(row.get(site_col, "")).strip()
    h3_index = latlon_to_h3(lat, lon)

    rec = WaterQualityRecord(
        canonical_param=canonical.canonical_name,
        value=harmonized_value,
        unit=canonical.canonical_unit,
        timestamp=timestamp,
        latitude=lat,
        longitude=lon,
        h3_index=h3_index,
        source=source,
        raw_param_name=raw_param,
        raw_unit=raw_unit,
        site_id=site_id,
    )

    # Assign quality tier
    rec.quality_tier = assign_quality_tier(rec, source)
    return rec


# ---------------------------------------------------------------------------
# EPA Water Quality Portal (WQP)
# ---------------------------------------------------------------------------


def ingest_epa_wqp(
    states: Sequence[str],
    params: Sequence[str] | None = None,
    start_date: str = "2020-01-01",
    end_date: str = "2024-12-31",
    output_dir: str | Path = "data/sentinel_db/epa_wqp",
) -> list[WaterQualityRecord]:
    """Ingest data from the EPA Water Quality Portal.

    Uses the ``dataretrieval.wqp`` module to pull results from the WQP
    REST API, then harmonizes to canonical schema.

    Parameters
    ----------
    states:
        US state codes to query (e.g. ``["CA", "OR", "WA"]``).
    params:
        WQP characteristic names (default: core parameters).
    start_date, end_date:
        Date range as ISO strings.
    output_dir:
        Directory to cache raw downloads.

    Returns
    -------
    List of harmonized :class:`WaterQualityRecord` instances.
    """
    import dataretrieval.wqp as wqp

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if params is None:
        params = [
            "Dissolved oxygen (DO)",
            "pH",
            "Specific conductance",
            "Temperature, water",
            "Turbidity",
            "Nitrogen, total",
            "Phosphorus, total",
            "Chlorophyll a",
        ]

    all_records: list[WaterQualityRecord] = []

    progress = make_progress()
    with progress:
        task = progress.add_task("Ingesting EPA WQP", total=len(states))
        for state in states:
            try:
                df, _ = wqp.get_results(
                    statecode=f"US:{state}",
                    characteristicName=list(params),
                    startDateLo=start_date,
                    startDateHi=end_date,
                    dataProfile="resultPhysChem",
                )

                if df.empty:
                    progress.advance(task)
                    continue

                # Cache raw data
                cache_path = output_dir / f"wqp_{state}.parquet"
                df.to_parquet(cache_path)

                # Convert rows
                for _, row in df.iterrows():
                    rec = _row_to_wq_record(
                        row.to_dict(),
                        param_col="CharacteristicName",
                        value_col="ResultMeasureValue",
                        unit_col="ResultMeasure/MeasureUnitCode",
                        time_col="ActivityStartDate",
                        lat_col="ActivityLocation/LatitudeMeasure",
                        lon_col="ActivityLocation/LongitudeMeasure",
                        site_col="MonitoringLocationIdentifier",
                        source="EPA_WQP",
                    )
                    if rec is not None:
                        all_records.append(rec)

            except Exception as exc:
                logger.warning(f"WQP ingest failed for {state}: {exc}")

            progress.advance(task)
            time.sleep(1.0)  # Rate limiting

    logger.info(f"EPA WQP ingest: {len(all_records)} records from {len(states)} states")
    return all_records


# ---------------------------------------------------------------------------
# EU Waterbase (EEA DiscoData)
# ---------------------------------------------------------------------------

_EU_WATERBASE_URL = (
    "https://discodata.eea.europa.eu/sql?"
    "query=SELECT+*+FROM+%5BWISE_SOE%5D.%5Bv_WISE_SOE_WaterBody%5D"
    "&p_CountryCode={country}"
    "&f=csv"
)

_EU_WATERBASE_BULK_URL = (
    "https://cmshare.eea.europa.eu/s/HyFo4Wfqo6SXk4c/download"
)


def ingest_eu_waterbase(
    output_dir: str | Path = "data/sentinel_db/eu_waterbase",
    *,
    url: str = _EU_WATERBASE_BULK_URL,
    timeout: int = 600,
) -> list[WaterQualityRecord]:
    """Ingest EU Waterbase data from EEA bulk CSV download.

    Downloads the WISE SOE water quality dataset and harmonizes to
    canonical schema.

    Parameters
    ----------
    output_dir:
        Directory for cached downloads.
    url:
        Download URL for the bulk CSV ZIP.
    timeout:
        HTTP timeout in seconds.

    Returns
    -------
    List of harmonized :class:`WaterQualityRecord` instances.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_records: list[WaterQualityRecord] = []

    try:
        logger.info("Downloading EU Waterbase bulk data...")
        resp = requests.get(url, timeout=timeout, stream=True)
        resp.raise_for_status()

        zip_path = output_dir / "waterbase_bulk.zip"
        total_size = int(resp.headers.get("content-length", 0))
        progress = make_progress()
        with progress:
            dl_task = progress.add_task(
                "Downloading EU Waterbase",
                total=total_size if total_size else None,
            )
            with open(zip_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)
                    progress.advance(dl_task, advance=len(chunk))

        # Extract and process CSV files
        with zipfile.ZipFile(zip_path, "r") as zf:
            csv_files = [n for n in zf.namelist() if n.endswith(".csv")]
            logger.info(f"Found {len(csv_files)} CSV files in archive")

            prog2 = make_progress()
            with prog2:
                task2 = prog2.add_task("Processing EU Waterbase", total=len(csv_files))
                for csv_name in csv_files:
                    try:
                        with zf.open(csv_name) as csv_fh:
                            df = pd.read_csv(
                                io.TextIOWrapper(csv_fh, encoding="utf-8"),
                                low_memory=False,
                                on_bad_lines="skip",
                            )

                        # EU Waterbase column mapping (typical WISE SOE schema)
                        param_col = _find_col(df, [
                            "observedPropertyDeterminandLabel",
                            "determinandLabel", "parameterLabel", "Determinand",
                        ])
                        value_col = _find_col(df, [
                            "resultObservedValue", "resultMeanValue",
                            "resultNumberValue", "Value",
                        ])
                        unit_col = _find_col(df, [
                            "resultUom", "resultUnit", "Unit",
                        ])
                        time_col = _find_col(df, [
                            "phenomenonTimeSamplingDate", "sampleDate",
                            "phenomenonTimeReferenceYear", "Year",
                        ])
                        lat_col = _find_col(df, [
                            "lat", "latitude", "Latitude",
                            "monitoringSiteLatitude",
                        ])
                        lon_col = _find_col(df, [
                            "lon", "longitude", "Longitude",
                            "monitoringSiteLongitude",
                        ])
                        site_col = _find_col(df, [
                            "monitoringSiteIdentifier", "stationId",
                            "monitoringSiteCode",
                        ])

                        if not all([param_col, value_col, lat_col, lon_col]):
                            prog2.advance(task2)
                            continue

                        for _, row in df.iterrows():
                            rec = _row_to_wq_record(
                                row.to_dict(),
                                param_col=param_col,
                                value_col=value_col,
                                unit_col=unit_col or "unit",
                                time_col=time_col or "date",
                                lat_col=lat_col,
                                lon_col=lon_col,
                                site_col=site_col or "site_id",
                                source="EU_Waterbase",
                            )
                            if rec is not None:
                                all_records.append(rec)

                    except Exception as exc:
                        logger.warning(f"Failed to process {csv_name}: {exc}")

                    prog2.advance(task2)

    except Exception as exc:
        logger.error(f"EU Waterbase ingest failed: {exc}")

    logger.info(f"EU Waterbase ingest: {len(all_records)} records")
    return all_records


# ---------------------------------------------------------------------------
# GEMStat
# ---------------------------------------------------------------------------

_GEMSTAT_URL = "https://gemstat.org/wp-content/uploads/2024/01/GEMStat_Data_Export.zip"


def ingest_gemstat(
    output_dir: str | Path = "data/sentinel_db/gemstat",
    *,
    url: str = _GEMSTAT_URL,
    timeout: int = 600,
) -> list[WaterQualityRecord]:
    """Ingest data from GEMStat (UN GEMS/Water).

    Downloads the GEMStat bulk export and harmonizes parameter names.

    Parameters
    ----------
    output_dir:
        Directory for cached downloads.
    url:
        Download URL for the GEMStat ZIP archive.
    timeout:
        HTTP timeout in seconds.

    Returns
    -------
    List of harmonized :class:`WaterQualityRecord` instances.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_records: list[WaterQualityRecord] = []

    try:
        logger.info("Downloading GEMStat data...")
        resp = requests.get(url, timeout=timeout, stream=True)
        resp.raise_for_status()

        zip_path = output_dir / "gemstat_export.zip"
        with open(zip_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)

        # Extract and process
        with zipfile.ZipFile(zip_path, "r") as zf:
            csv_files = [n for n in zf.namelist() if n.endswith(".csv")]

            progress = make_progress()
            with progress:
                task = progress.add_task("Processing GEMStat", total=len(csv_files))
                for csv_name in csv_files:
                    try:
                        with zf.open(csv_name) as csv_fh:
                            df = pd.read_csv(
                                io.TextIOWrapper(csv_fh, encoding="utf-8"),
                                low_memory=False,
                                on_bad_lines="skip",
                            )

                        param_col = _find_col(df, [
                            "Parameter", "parameter", "ParameterDescription",
                        ])
                        value_col = _find_col(df, [
                            "Value", "value", "DataValue", "Result",
                        ])
                        unit_col = _find_col(df, ["Unit", "unit", "Units"])
                        time_col = _find_col(df, [
                            "Sample Date", "SampleDate", "Date", "date",
                        ])
                        lat_col = _find_col(df, [
                            "Latitude", "latitude", "lat", "GEMS Station Latitude",
                        ])
                        lon_col = _find_col(df, [
                            "Longitude", "longitude", "lon", "GEMS Station Longitude",
                        ])
                        site_col = _find_col(df, [
                            "GEMS Station Number", "StationID", "station_id",
                        ])

                        if not all([param_col, value_col, lat_col, lon_col]):
                            progress.advance(task)
                            continue

                        for _, row in df.iterrows():
                            rec = _row_to_wq_record(
                                row.to_dict(),
                                param_col=param_col,
                                value_col=value_col,
                                unit_col=unit_col or "Unit",
                                time_col=time_col or "Date",
                                lat_col=lat_col,
                                lon_col=lon_col,
                                site_col=site_col or "station_id",
                                source="GEMStat",
                            )
                            if rec is not None:
                                all_records.append(rec)

                    except Exception as exc:
                        logger.warning(f"Failed to process {csv_name}: {exc}")

                    progress.advance(task)

    except Exception as exc:
        logger.error(f"GEMStat ingest failed: {exc}")

    logger.info(f"GEMStat ingest: {len(all_records)} records")
    return all_records


# ---------------------------------------------------------------------------
# GRQA (Global River Water Quality Archive — Zenodo)
# ---------------------------------------------------------------------------

_GRQA_ZENODO_URL = "https://zenodo.org/records/7056647/files/GRQA_data_v1.3.zip"


def ingest_grqa(
    output_dir: str | Path = "data/sentinel_db/grqa",
    *,
    url: str = _GRQA_ZENODO_URL,
    timeout: int = 1800,
) -> list[WaterQualityRecord]:
    """Ingest the GRQA harmonized global river water quality dataset.

    GRQA (Zenodo) provides pre-harmonized data from multiple sources.  We
    still map through our ontology for consistency.

    Parameters
    ----------
    output_dir:
        Directory for cached downloads.
    url:
        Zenodo download URL.
    timeout:
        HTTP timeout in seconds.

    Returns
    -------
    List of harmonized :class:`WaterQualityRecord` instances.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_records: list[WaterQualityRecord] = []

    try:
        logger.info("Downloading GRQA dataset from Zenodo...")
        resp = requests.get(url, timeout=timeout, stream=True)
        resp.raise_for_status()

        zip_path = output_dir / "grqa_data.zip"
        total_size = int(resp.headers.get("content-length", 0))
        progress = make_progress()
        with progress:
            dl_task = progress.add_task(
                "Downloading GRQA",
                total=total_size if total_size else None,
            )
            with open(zip_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)
                    progress.advance(dl_task, advance=len(chunk))

        # Extract and process
        with zipfile.ZipFile(zip_path, "r") as zf:
            csv_files = [n for n in zf.namelist() if n.endswith(".csv")]

            prog2 = make_progress()
            with prog2:
                task2 = prog2.add_task("Processing GRQA", total=len(csv_files))
                for csv_name in csv_files:
                    try:
                        with zf.open(csv_name) as csv_fh:
                            df = pd.read_csv(
                                io.TextIOWrapper(csv_fh, encoding="utf-8"),
                                low_memory=False,
                                on_bad_lines="skip",
                            )

                        param_col = _find_col(df, [
                            "obs_variable", "parameter", "Parameter",
                        ])
                        value_col = _find_col(df, [
                            "obs_value", "value", "Value",
                        ])
                        unit_col = _find_col(df, [
                            "obs_unit", "unit", "Unit",
                        ])
                        time_col = _find_col(df, [
                            "obs_date", "date", "Date",
                        ])
                        lat_col = _find_col(df, [
                            "lat", "latitude", "Latitude",
                        ])
                        lon_col = _find_col(df, [
                            "lon", "longitude", "Longitude",
                        ])
                        site_col = _find_col(df, [
                            "site_id", "station_id", "StationID",
                        ])

                        if not all([param_col, value_col, lat_col, lon_col]):
                            prog2.advance(task2)
                            continue

                        for _, row in df.iterrows():
                            rec = _row_to_wq_record(
                                row.to_dict(),
                                param_col=param_col,
                                value_col=value_col,
                                unit_col=unit_col or "unit",
                                time_col=time_col or "date",
                                lat_col=lat_col,
                                lon_col=lon_col,
                                site_col=site_col or "site_id",
                                source="GRQA",
                            )
                            if rec is not None:
                                all_records.append(rec)

                    except Exception as exc:
                        logger.warning(f"Failed to process {csv_name}: {exc}")

                    prog2.advance(task2)

    except Exception as exc:
        logger.error(f"GRQA ingest failed: {exc}")

    logger.info(f"GRQA ingest: {len(all_records)} records")
    return all_records


# ---------------------------------------------------------------------------
# FreshWater Watch (citizen science)
# ---------------------------------------------------------------------------

_FWW_API_URL = "https://freshwaterwatch.thewaterhub.org/api/v1/observations"


def ingest_freshwater_watch(
    output_dir: str | Path = "data/sentinel_db/freshwater_watch",
    *,
    api_url: str = _FWW_API_URL,
    timeout: int = 300,
    max_pages: int = 100,
) -> list[WaterQualityRecord]:
    """Ingest citizen science data from the FreshWater Watch platform.

    FreshWater Watch typically provides turbidity (NTU) and nitrate
    measurements collected by volunteers.

    Parameters
    ----------
    output_dir:
        Directory for cached downloads.
    api_url:
        FreshWater Watch API endpoint.
    timeout:
        HTTP timeout in seconds.
    max_pages:
        Maximum API pages to fetch.

    Returns
    -------
    List of harmonized :class:`WaterQualityRecord` instances.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_records: list[WaterQualityRecord] = []

    try:
        page = 1
        progress = make_progress()
        with progress:
            task = progress.add_task(
                "Ingesting FreshWater Watch", total=max_pages
            )
            while page <= max_pages:
                try:
                    resp = requests.get(
                        api_url,
                        params={"page": page, "per_page": 500},
                        timeout=timeout,
                    )
                    resp.raise_for_status()
                    data = resp.json()

                    # Handle both list and paginated dict responses
                    if isinstance(data, dict):
                        observations = data.get("data", data.get("results", []))
                    elif isinstance(data, list):
                        observations = data
                    else:
                        break

                    if not observations:
                        break

                    for obs in observations:
                        # FreshWater Watch observations typically include:
                        # latitude, longitude, date, turbidity, nitrate, phosphate
                        lat = _safe_float(obs.get("latitude", obs.get("lat")))
                        lon = _safe_float(obs.get("longitude", obs.get("lon")))
                        ts = _safe_datetime(
                            obs.get("date", obs.get("observation_date"))
                        )
                        site = str(obs.get("site_id", obs.get("location_id", "")))

                        if lat is None or lon is None or ts is None:
                            continue

                        h3_idx = latlon_to_h3(lat, lon)

                        # Extract measurements
                        measurement_keys = {
                            "turbidity": ("turbidity", "NTU"),
                            "nitrate": ("nitrate", "mg/L"),
                            "phosphate": ("orthophosphate", "mg/L"),
                            "dissolved_oxygen": ("dissolved_oxygen", "mg/L"),
                            "ph": ("ph", "pH units"),
                        }

                        for raw_key, (canon, unit) in measurement_keys.items():
                            val = _safe_float(obs.get(raw_key))
                            if val is None:
                                continue

                            rec = WaterQualityRecord(
                                canonical_param=canon,
                                value=val,
                                unit=unit,
                                timestamp=ts,
                                latitude=lat,
                                longitude=lon,
                                h3_index=h3_idx,
                                source="FreshWater_Watch",
                                quality_tier=QualityTier.Q3,
                                raw_param_name=raw_key,
                                raw_unit=unit,
                                site_id=site,
                            )
                            all_records.append(rec)

                except requests.exceptions.HTTPError as exc:
                    if exc.response is not None and exc.response.status_code == 404:
                        break
                    logger.warning(f"FWW API page {page} failed: {exc}")
                except Exception as exc:
                    logger.warning(f"FWW API page {page} failed: {exc}")

                progress.advance(task)
                page += 1
                time.sleep(0.5)  # Rate limiting

    except Exception as exc:
        logger.error(f"FreshWater Watch ingest failed: {exc}")

    # Cache results
    if all_records:
        cache_path = output_dir / "freshwater_watch.parquet"
        df = pd.DataFrame([r.model_dump() for r in all_records])
        df.to_parquet(cache_path)
        logger.info(f"Cached {len(all_records)} records to {cache_path}")

    logger.info(f"FreshWater Watch ingest: {len(all_records)} records")
    return all_records


# ---------------------------------------------------------------------------
# Utility: flexible column finder
# ---------------------------------------------------------------------------


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Find the first matching column name from a list of candidates.

    Performs case-insensitive matching and returns the actual column name
    from the DataFrame.
    """
    col_lower_map = {c.lower().strip(): c for c in df.columns}
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
        actual = col_lower_map.get(candidate.lower().strip())
        if actual is not None:
            return actual
    return None
