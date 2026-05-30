#!/usr/bin/env python3
"""Download real HAB (Harmful Algal Bloom) data from NOAA sources.

Sources:
1. NOAA CoastWatch ERDDAP — Satellite-derived cyanobacteria/chlorophyll
   products for Great Lakes and major US water bodies.
   Base: https://coastwatch.pfeg.noaa.gov/erddap/
2. NOAA HABSOS (Harmful Algal BloomS Observing System) — In-situ HAB
   observations for the Gulf of Mexico.
   Base: https://habsos.coastalscience.noaa.gov/

Downloads tabular CSV data and saves as parquet with metadata.

Usage:
    python scripts/download_noaa_habs.py
    python scripts/download_noaa_habs.py --regions great_lakes gulf_of_mexico
    python scripts/download_noaa_habs.py --max-records 50000

MIT License -- Bryan Cheng, 2026
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("download_noaa_habs")

# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
ERDDAP_BASE = "https://coastwatch.pfeg.noaa.gov/erddap"
HABSOS_BASE = "https://habsos.coastalscience.noaa.gov"

# Rate limiting
REQUEST_DELAY = 1.5  # seconds between API calls
MAX_RETRIES = 3
RETRY_BACKOFF = 5.0  # seconds base backoff on failure
TIMEOUT = 60

# ---------------------------------------------------------------------------
# ERDDAP dataset IDs — satellite-derived water quality products
# ---------------------------------------------------------------------------
ERDDAP_DATASETS = {
    # Great Lakes chlorophyll and cyanobacteria products
    "nesdisVHNSQchlaWeekly": {
        "description": "VIIRS chlorophyll-a weekly composite",
        "variables": ["chlor_a"],
        "region": "great_lakes",
        "has_altitude": True,
        "max_end": "2026-12-31T00:00:00Z",
    },
    "erdMH1chla8day": {
        "description": "MODIS Aqua chlorophyll-a 8-day composite",
        "variables": ["chlorophyll"],
        "region": "great_lakes",
        "has_altitude": False,
        "max_end": "2022-06-01T00:00:00Z",  # dataset ends ~mid-2022
    },
    "erdMH1chlamday": {
        "description": "MODIS Aqua chlorophyll-a monthly composite",
        "variables": ["chlorophyll"],
        "region": "great_lakes",
        "has_altitude": False,
        "max_end": "2022-06-01T00:00:00Z",  # dataset ends ~mid-2022
    },
    "nesdisVHNSQchlaMonthly": {
        "description": "VIIRS chlorophyll-a monthly composite",
        "variables": ["chlor_a"],
        "region": "great_lakes",
        "has_altitude": True,
        "max_end": "2026-12-31T00:00:00Z",
    },
}

# Bounding boxes for regions of interest (min_lon, min_lat, max_lon, max_lat)
REGION_BOUNDS = {
    "great_lakes": (-92.5, 41.0, -76.0, 49.5),
    "chesapeake_bay": (-77.5, 36.5, -75.5, 39.8),
    "gulf_of_mexico": (-98.0, 24.0, -81.0, 31.0),
    "florida_coast": (-83.0, 24.0, -79.5, 31.0),
    "lake_okeechobee": (-81.5, 26.5, -80.2, 27.5),
}

# HABSOS species of interest
HABSOS_SPECIES = [
    "Karenia brevis",
    "Pseudo-nitzschia",
    "Alexandrium",
    "Dinophysis",
    "Prorocentrum",
]


def _rate_limited_get(
    url: str,
    params: Optional[dict] = None,
    timeout: int = TIMEOUT,
    retries: int = MAX_RETRIES,
) -> Optional[requests.Response]:
    """Make a rate-limited GET request with retries."""
    for attempt in range(retries):
        try:
            time.sleep(REQUEST_DELAY)
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code == 200:
                return resp
            elif resp.status_code == 429:
                wait = RETRY_BACKOFF * (attempt + 2)
                log.warning(f"Rate limited (429), waiting {wait:.0f}s...")
                time.sleep(wait)
            elif resp.status_code == 404:
                log.warning(f"Not found (404): {url}")
                return None
            else:
                log.warning(
                    f"HTTP {resp.status_code} for {url} "
                    f"(attempt {attempt + 1}/{retries})"
                )
                time.sleep(RETRY_BACKOFF * (attempt + 1))
        except requests.exceptions.Timeout:
            log.warning(f"Timeout for {url} (attempt {attempt + 1}/{retries})")
            time.sleep(RETRY_BACKOFF * (attempt + 1))
        except requests.exceptions.ConnectionError as e:
            log.warning(f"Connection error: {e} (attempt {attempt + 1}/{retries})")
            time.sleep(RETRY_BACKOFF * (attempt + 1))
        except Exception as e:
            log.error(f"Unexpected error: {e}")
            return None
    log.error(f"All {retries} attempts failed for {url}")
    return None


# ---------------------------------------------------------------------------
# ERDDAP Functions
# ---------------------------------------------------------------------------

def search_erddap_datasets(keyword: str = "chlorophyll") -> List[Dict]:
    """Search ERDDAP for available datasets matching a keyword."""
    url = f"{ERDDAP_BASE}/search/index.json"
    params = {
        "searchFor": keyword,
        "page": 1,
        "itemsPerPage": 50,
    }
    resp = _rate_limited_get(url, params=params)
    if resp is None:
        return []
    try:
        data = resp.json()
        rows = data.get("table", {}).get("rows", [])
        col_names = data.get("table", {}).get("columnNames", [])
        results = []
        for row in rows:
            entry = dict(zip(col_names, row))
            results.append(entry)
        return results
    except Exception as e:
        log.error(f"Failed to parse ERDDAP search results: {e}")
        return []


def get_erddap_dataset_info(dataset_id: str) -> Optional[Dict]:
    """Get metadata for an ERDDAP dataset."""
    url = f"{ERDDAP_BASE}/info/{dataset_id}/index.json"
    resp = _rate_limited_get(url)
    if resp is None:
        return None
    try:
        data = resp.json()
        rows = data.get("table", {}).get("rows", [])
        info = {"variables": [], "attributes": {}}
        for row in rows:
            row_type = row[0] if len(row) > 0 else ""
            var_name = row[1] if len(row) > 1 else ""
            attr_name = row[2] if len(row) > 2 else ""
            data_type = row[3] if len(row) > 3 else ""
            value = row[4] if len(row) > 4 else ""
            if row_type == "variable":
                info["variables"].append(var_name)
            if attr_name:
                info["attributes"][f"{var_name}.{attr_name}"] = value
        return info
    except Exception as e:
        log.error(f"Failed to parse dataset info for {dataset_id}: {e}")
        return None


def _generate_year_chunks(
    time_start: str, time_end: str
) -> List[Tuple[str, str]]:
    """Split a time range into per-year chunks for manageable ERDDAP queries."""
    from datetime import datetime

    fmt = "%Y-%m-%dT%H:%M:%SZ"
    t0 = datetime.strptime(time_start, fmt)
    t1 = datetime.strptime(time_end, fmt)

    chunks = []
    current = t0
    while current < t1:
        year_end = datetime(current.year, 12, 31, 23, 59, 59)
        chunk_end = min(year_end, t1)
        chunks.append(
            (current.strftime(fmt), chunk_end.strftime(fmt))
        )
        current = datetime(current.year + 1, 1, 1, 0, 0, 0)
    return chunks


def download_erddap_griddap(
    dataset_id: str,
    variables: List[str],
    time_range: Tuple[str, str],
    lat_range: Tuple[float, float],
    lon_range: Tuple[float, float],
    output_path: Path,
    stride: int = 1,
    has_altitude: bool = False,
) -> Optional[pd.DataFrame]:
    """Download gridded data from ERDDAP griddap service as CSV.

    Builds a constraint URL for griddap and downloads as CSV.
    Chunks by year to avoid server-side timeouts on large requests.
    """
    if output_path.exists() and output_path.stat().st_size > 500:
        log.info(f"  Already exists: {output_path.name}")
        try:
            return pd.read_parquet(output_path)
        except Exception:
            pass

    lat_min, lat_max = lat_range
    lon_min, lon_max = lon_range

    # Build altitude part of constraint if needed
    alt_constraint = "[(0.0)]" if has_altitude else ""

    # Request one variable at a time to keep responses manageable
    var_list = ",".join(variables[:1])

    # Chunk by year to avoid 500 errors from too-large requests
    year_chunks = _generate_year_chunks(time_range[0], time_range[1])
    all_dfs = []

    for chunk_start, chunk_end in year_chunks:
        constraint = (
            f"[({chunk_start}):({chunk_end})]"
            f"{alt_constraint}"
            f"[({lat_min}):{stride}:({lat_max})]"
            f"[({lon_min}):{stride}:({lon_max})]"
        )

        url = f"{ERDDAP_BASE}/griddap/{dataset_id}.csv?{var_list}{constraint}"
        log.info(f"  Requesting: {dataset_id} [{chunk_start} to {chunk_end}]")

        resp = _rate_limited_get(url, timeout=180)
        if resp is None:
            log.warning(f"  No data for chunk {chunk_start} to {chunk_end}")
            continue

        try:
            from io import StringIO
            lines = resp.text.split("\n")
            if len(lines) < 3:
                log.warning(f"  Empty response for {dataset_id} chunk")
                continue
            # Remove the units row (second line)
            clean_lines = [lines[0]] + lines[2:]
            df_chunk = pd.read_csv(StringIO("\n".join(clean_lines)))
            if len(df_chunk) > 0:
                all_dfs.append(df_chunk)
                log.info(f"    Got {len(df_chunk):,} rows for {chunk_start[:4]}")
        except Exception as e:
            log.error(f"  Failed to parse CSV chunk: {e}")
            continue

    if not all_dfs:
        log.warning(f"  No data retrieved for {dataset_id}")
        return None

    df = pd.concat(all_dfs, ignore_index=True)
    df.to_parquet(output_path, index=False)
    log.info(f"  Saved {len(df):,} rows to {output_path.name}")
    return df


def download_erddap_tabledap(
    dataset_id: str,
    variables: List[str],
    constraints: Dict[str, str],
    output_path: Path,
    max_rows: int = 100000,
) -> Optional[pd.DataFrame]:
    """Download tabular data from ERDDAP tabledap service."""
    if output_path.exists() and output_path.stat().st_size > 500:
        log.info(f"  Already exists: {output_path.name}")
        try:
            return pd.read_parquet(output_path)
        except Exception:
            pass

    var_str = ",".join(variables)
    constraint_parts = []
    for key, val in constraints.items():
        constraint_parts.append(f"&{key}{val}")
    constraint_str = "".join(constraint_parts)

    url = (
        f"{ERDDAP_BASE}/tabledap/{dataset_id}.csv"
        f"?{var_str}{constraint_str}&orderByLimit(\"{max_rows}\")"
    )
    log.info(f"  Requesting tabledap: {dataset_id}")

    resp = _rate_limited_get(url, timeout=120)
    if resp is None:
        return None

    try:
        from io import StringIO
        lines = resp.text.split("\n")
        if len(lines) < 3:
            return None
        clean_lines = [lines[0]] + lines[2:]
        df = pd.read_csv(StringIO("\n".join(clean_lines)))
        if len(df) == 0:
            return None
        df.to_parquet(output_path, index=False)
        log.info(f"  Saved {len(df):,} rows to {output_path.name}")
        return df
    except Exception as e:
        log.error(f"  Failed to parse tabledap CSV: {e}")
        return None


# ---------------------------------------------------------------------------
# HABSOS Functions
# ---------------------------------------------------------------------------

def download_habsos_data(
    output_dir: Path,
    start_year: int = 2015,
    end_year: int = 2025,
    max_records: int = 100000,
) -> Optional[pd.DataFrame]:
    """Download HAB observation data from NOAA HABSOS.

    HABSOS provides cell count data for harmful algal species,
    primarily from the Gulf of Mexico monitoring network.
    """
    output_path = output_dir / "habsos_observations.parquet"
    if output_path.exists() and output_path.stat().st_size > 500:
        log.info(f"Already exists: {output_path.name}")
        try:
            return pd.read_parquet(output_path)
        except Exception:
            pass

    # HABSOS API endpoint for data query
    url = f"{HABSOS_BASE}/api/public/data"
    all_records = []

    for year in range(start_year, end_year + 1):
        params = {
            "startDate": f"{year}-01-01",
            "endDate": f"{year}-12-31",
            "format": "json",
        }
        log.info(f"  Querying HABSOS for {year}...")
        resp = _rate_limited_get(url, params=params, timeout=90)
        if resp is None:
            # Try alternate endpoint
            alt_url = f"{HABSOS_BASE}/maps/data"
            resp = _rate_limited_get(alt_url, params=params, timeout=90)

        if resp is not None:
            try:
                data = resp.json()
                if isinstance(data, list):
                    all_records.extend(data)
                    log.info(f"    {year}: {len(data)} records")
                elif isinstance(data, dict) and "data" in data:
                    all_records.extend(data["data"])
                    log.info(f"    {year}: {len(data['data'])} records")
                else:
                    log.info(f"    {year}: unexpected format, keys={list(data.keys()) if isinstance(data, dict) else type(data)}")
            except Exception as e:
                log.warning(f"    {year}: JSON parse error — {e}")
        else:
            log.warning(f"    {year}: no response")

        if len(all_records) >= max_records:
            log.info(f"  Reached max records ({max_records}), stopping.")
            break

    if not all_records:
        log.warning("No HABSOS records retrieved. Trying CSV fallback...")
        return _habsos_csv_fallback(output_dir)

    df = pd.DataFrame(all_records)
    df.to_parquet(output_path, index=False)
    log.info(f"Saved {len(df):,} HABSOS records to {output_path.name}")
    return df


def _habsos_csv_fallback(output_dir: Path) -> Optional[pd.DataFrame]:
    """Try downloading HABSOS data as CSV export."""
    output_path = output_dir / "habsos_observations.parquet"

    # Try the HABSOS download CSV endpoint
    csv_url = f"{HABSOS_BASE}/api/public/data/csv"
    params = {
        "startDate": "2015-01-01",
        "endDate": "2025-12-31",
    }
    resp = _rate_limited_get(csv_url, params=params, timeout=120)
    if resp is None:
        log.warning("HABSOS CSV fallback also failed.")
        # Save an empty metadata file to indicate we tried
        meta = {
            "status": "api_unavailable",
            "message": (
                "HABSOS API was not reachable. To manually download data, "
                "visit https://habsos.coastalscience.noaa.gov/ and use the "
                "interactive map to export CSV data."
            ),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(output_dir / "habsos_status.json", "w") as f:
            json.dump(meta, f, indent=2)
        return None

    try:
        from io import StringIO
        df = pd.read_csv(StringIO(resp.text))
        df.to_parquet(output_path, index=False)
        log.info(f"Saved {len(df):,} HABSOS records (CSV fallback)")
        return df
    except Exception as e:
        log.error(f"Failed to parse HABSOS CSV: {e}")
        return None


# ---------------------------------------------------------------------------
# ERDDAP Dataset Discovery
# ---------------------------------------------------------------------------

def discover_hab_datasets(output_dir: Path) -> List[Dict]:
    """Search ERDDAP for all HAB-related datasets and save catalog."""
    catalog_path = output_dir / "erddap_hab_catalog.json"
    if catalog_path.exists():
        with open(catalog_path) as f:
            return json.load(f)

    log.info("Discovering HAB-related datasets on ERDDAP...")
    search_terms = [
        "chlorophyll",
        "cyanobacteria",
        "harmful algal bloom",
        "phycocyanin",
        "chl_a",
        "fluorescence",
    ]
    all_datasets = {}
    for term in search_terms:
        results = search_erddap_datasets(term)
        for ds in results:
            ds_id = ds.get("Dataset ID", ds.get("datasetID", ""))
            if ds_id and ds_id not in all_datasets:
                all_datasets[ds_id] = {
                    "dataset_id": ds_id,
                    "title": ds.get("Title", ds.get("title", "")),
                    "summary": ds.get("Summary", ds.get("summary", ""))[:200],
                    "search_term": term,
                }

    catalog = list(all_datasets.values())
    with open(catalog_path, "w") as f:
        json.dump(catalog, f, indent=2)
    log.info(f"Found {len(catalog)} unique HAB-related datasets on ERDDAP")
    return catalog


# ---------------------------------------------------------------------------
# Main download pipeline
# ---------------------------------------------------------------------------

def download_all_erddap(
    output_dir: Path,
    regions: List[str],
    time_range: Tuple[str, str],
    max_records: int = 100000,
) -> Dict[str, Any]:
    """Download ERDDAP satellite chlorophyll/cyanobacteria data."""
    stats = {"datasets_attempted": 0, "datasets_downloaded": 0, "total_rows": 0}

    for ds_id, ds_info in ERDDAP_DATASETS.items():
        ds_region = ds_info["region"]
        if ds_region not in regions:
            continue

        stats["datasets_attempted"] += 1
        bounds = REGION_BOUNDS.get(ds_region)
        if bounds is None:
            continue

        min_lon, min_lat, max_lon, max_lat = bounds

        out_path = output_dir / f"erddap_{ds_id}.parquet"

        # Clamp end date to dataset's known maximum
        effective_end = time_range[1]
        max_end = ds_info.get("max_end")
        if max_end and max_end < effective_end:
            effective_end = max_end
            log.info(f"  Clamping end date for {ds_id} to {effective_end}")

        effective_time_range = (time_range[0], effective_end)

        # Try griddap first (for gridded satellite products)
        # Use stride=20 (~0.75-0.8 degree spacing) to keep CSV responses manageable
        df = download_erddap_griddap(
            dataset_id=ds_id,
            variables=ds_info["variables"],
            time_range=effective_time_range,
            lat_range=(min_lat, max_lat),
            lon_range=(min_lon, max_lon),
            output_path=out_path,
            stride=20,
            has_altitude=ds_info.get("has_altitude", False),
        )

        if df is not None:
            stats["datasets_downloaded"] += 1
            stats["total_rows"] += len(df)

    return stats


def compute_summary_statistics(output_dir: Path) -> Dict[str, Any]:
    """Compute summary statistics over all downloaded data."""
    summary = {
        "total_files": 0,
        "total_rows": 0,
        "datasets": {},
    }

    for pq_file in sorted(output_dir.glob("*.parquet")):
        try:
            df = pd.read_parquet(pq_file)
            ds_stats = {
                "file": pq_file.name,
                "rows": len(df),
                "columns": list(df.columns),
                "dtypes": {c: str(df[c].dtype) for c in df.columns},
            }
            # Basic stats for numeric columns
            for col in df.select_dtypes(include=[np.number]).columns:
                ds_stats[f"{col}_min"] = float(df[col].min()) if not df[col].isna().all() else None
                ds_stats[f"{col}_max"] = float(df[col].max()) if not df[col].isna().all() else None
                ds_stats[f"{col}_mean"] = float(df[col].mean()) if not df[col].isna().all() else None
                ds_stats[f"{col}_null_pct"] = float(df[col].isna().mean() * 100)

            summary["datasets"][pq_file.stem] = ds_stats
            summary["total_files"] += 1
            summary["total_rows"] += len(df)
        except Exception as e:
            log.warning(f"Could not read {pq_file.name}: {e}")

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download NOAA HAB data (ERDDAP + HABSOS)"
    )
    parser.add_argument(
        "--data-dir",
        default="/home/bcheng/SENTINEL/data/processed/biology/noaa_habs",
        help="Output directory for downloaded data",
    )
    parser.add_argument(
        "--regions",
        nargs="+",
        default=["great_lakes", "gulf_of_mexico", "chesapeake_bay"],
        choices=list(REGION_BOUNDS.keys()),
        help="Regions to download data for",
    )
    parser.add_argument(
        "--start-date",
        default="2020-01-01T00:00:00Z",
        help="Start date for ERDDAP queries (ISO format)",
    )
    parser.add_argument(
        "--end-date",
        default="2025-12-31T00:00:00Z",
        help="End date for ERDDAP queries (ISO format)",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=100000,
        help="Maximum records per dataset",
    )
    parser.add_argument(
        "--skip-erddap",
        action="store_true",
        help="Skip ERDDAP satellite products",
    )
    parser.add_argument(
        "--skip-habsos",
        action="store_true",
        help="Skip HABSOS in-situ data",
    )
    parser.add_argument(
        "--discover-only",
        action="store_true",
        help="Only discover available datasets, do not download",
    )
    args = parser.parse_args()

    output_dir = Path(args.data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 70)
    log.info("NOAA HAB Data Download Pipeline")
    log.info(f"  Output: {output_dir}")
    log.info(f"  Regions: {args.regions}")
    log.info(f"  Time range: {args.start_date} to {args.end_date}")
    log.info("=" * 70)

    # Step 1: Discover available datasets
    log.info("STEP 1: Discovering HAB-related datasets on ERDDAP")
    catalog = discover_hab_datasets(output_dir)
    log.info(f"  Found {len(catalog)} HAB-related datasets")

    if args.discover_only:
        log.info("Discovery-only mode. Exiting.")
        for ds in catalog[:20]:
            log.info(f"  [{ds['dataset_id']}] {ds['title'][:70]}")
        return

    # Step 2: Download ERDDAP satellite products
    erddap_stats = {"datasets_attempted": 0, "datasets_downloaded": 0, "total_rows": 0}
    if not args.skip_erddap:
        log.info("=" * 70)
        log.info("STEP 2: Downloading ERDDAP satellite chlorophyll/HAB products")
        log.info("=" * 70)
        erddap_stats = download_all_erddap(
            output_dir=output_dir,
            regions=args.regions,
            time_range=(args.start_date, args.end_date),
            max_records=args.max_records,
        )
        log.info(
            f"  ERDDAP: {erddap_stats['datasets_downloaded']}/{erddap_stats['datasets_attempted']} "
            f"datasets downloaded, {erddap_stats['total_rows']:,} total rows"
        )
    else:
        log.info("Skipping ERDDAP downloads.")

    # Step 3: Download HABSOS in-situ observations
    habsos_df = None
    if not args.skip_habsos:
        log.info("=" * 70)
        log.info("STEP 3: Downloading HABSOS in-situ HAB observations")
        log.info("=" * 70)
        habsos_df = download_habsos_data(
            output_dir=output_dir,
            start_year=2015,
            end_year=2025,
            max_records=args.max_records,
        )
        if habsos_df is not None:
            log.info(f"  HABSOS: {len(habsos_df):,} observations")
        else:
            log.info("  HABSOS: no data retrieved (see status file)")
    else:
        log.info("Skipping HABSOS downloads.")

    # Step 4: Summary statistics
    log.info("=" * 70)
    log.info("STEP 4: Computing summary statistics")
    log.info("=" * 70)
    summary = compute_summary_statistics(output_dir)

    # Save metadata
    metadata = {
        "download_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "regions": args.regions,
        "time_range": [args.start_date, args.end_date],
        "erddap_stats": erddap_stats,
        "habsos_records": len(habsos_df) if habsos_df is not None else 0,
        "summary": {
            "total_files": summary["total_files"],
            "total_rows": summary["total_rows"],
        },
        "erddap_datasets_configured": list(ERDDAP_DATASETS.keys()),
        "region_bounds": {k: list(v) for k, v in REGION_BOUNDS.items()},
    }
    with open(output_dir / "download_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Print final summary
    log.info("=" * 70)
    log.info("DOWNLOAD COMPLETE")
    log.info(f"  Total files: {summary['total_files']}")
    log.info(f"  Total rows:  {summary['total_rows']:,}")
    log.info(f"  Output dir:  {output_dir}")
    for ds_name, ds_stats in summary["datasets"].items():
        log.info(f"    {ds_name}: {ds_stats['rows']:,} rows, cols={ds_stats['columns']}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
