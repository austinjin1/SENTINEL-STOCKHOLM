#!/usr/bin/env python3
"""Download Sentinel-3 OLCI water quality products for inland waters.

Sources:
1. Copernicus Marine Service (CMEMS) — chlorophyll-a, turbidity, suspended
   matter products via the CMEMS API (requires free registration).
   https://marine.copernicus.eu/
2. EUMETSAT Data Store — Sentinel-3 OLCI Level-2 products.
   https://data.eumetsat.int/
3. Copernicus Data Space Ecosystem (CDSE) — the new unified access point
   for all Copernicus satellite data (replaces SciHub).
   https://dataspace.copernicus.eu/

For CONUS inland waters. Authentication is required for actual data
download; this script handles discovery and downloads what is publicly
accessible, with clear documentation for authenticated access.

Usage:
    python scripts/download_sentinel3.py
    python scripts/download_sentinel3.py --discover-only
    python scripts/download_sentinel3.py --region great_lakes --start 2023-01-01

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
log = logging.getLogger("download_sentinel3")

# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------
# Copernicus Data Space Ecosystem (CDSE) — public catalog search
CDSE_CATALOG = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
CDSE_TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"

# CMEMS STAC catalog (public discovery)
CMEMS_STAC = "https://stac.marine.copernicus.eu/metadata"

# Copernicus Global Land Service — inland water quality (public)
CGLS_BASE = "https://land.copernicus.eu/en/products/water"

# Rate limiting
REQUEST_DELAY = 1.5
MAX_RETRIES = 3
RETRY_BACKOFF = 5.0
TIMEOUT = 60

# ---------------------------------------------------------------------------
# Regions of interest (CONUS inland waters)
# ---------------------------------------------------------------------------
REGIONS = {
    "great_lakes": {
        "name": "Great Lakes",
        "bbox": [-92.5, 41.0, -76.0, 49.5],
        "description": "Lakes Superior, Michigan, Huron, Erie, Ontario",
    },
    "chesapeake": {
        "name": "Chesapeake Bay",
        "bbox": [-77.5, 36.5, -75.5, 39.8],
        "description": "Chesapeake Bay estuary",
    },
    "florida_lakes": {
        "name": "Florida Lakes",
        "bbox": [-82.5, 26.0, -80.0, 29.0],
        "description": "Lake Okeechobee and central Florida lakes",
    },
    "upper_midwest": {
        "name": "Upper Midwest Lakes",
        "bbox": [-95.0, 43.0, -86.0, 49.0],
        "description": "Minnesota/Wisconsin lake country",
    },
    "california": {
        "name": "California Water Bodies",
        "bbox": [-123.0, 35.0, -118.0, 42.0],
        "description": "Clear Lake, Sacramento Delta, reservoirs",
    },
}

# Sentinel-3 OLCI product types of interest
OLCI_PRODUCTS = {
    "OL_2_WFR___": "OLCI Level-2 Water Full Resolution (300m)",
    "OL_2_WRR___": "OLCI Level-2 Water Reduced Resolution (1.2km)",
    "OL_1_EFR___": "OLCI Level-1 Full Resolution",
}

# Water quality parameters available in OLCI Level-2
WQ_PARAMETERS = [
    "chlorophyll_a",      # CHL_OC4ME, CHL_NN
    "total_suspended_matter",  # TSM_NN
    "turbidity",          # Derived from TSM
    "cdom",               # Colored Dissolved Organic Matter
    "water_reflectance",  # Rw at multiple bands
]


def _rate_limited_get(
    url: str,
    params: Optional[dict] = None,
    timeout: int = TIMEOUT,
    retries: int = MAX_RETRIES,
    headers: Optional[dict] = None,
) -> Optional[requests.Response]:
    """Make a rate-limited GET request with retries."""
    for attempt in range(retries):
        try:
            time.sleep(REQUEST_DELAY)
            resp = requests.get(
                url, params=params, timeout=timeout, headers=headers
            )
            if resp.status_code == 200:
                return resp
            elif resp.status_code == 401:
                log.warning("Authentication required (401). See auth docs below.")
                return None
            elif resp.status_code == 429:
                wait = RETRY_BACKOFF * (attempt + 2)
                log.warning(f"Rate limited (429), waiting {wait:.0f}s...")
                time.sleep(wait)
            elif resp.status_code == 404:
                log.debug(f"Not found (404): {url}")
                return None
            else:
                log.warning(
                    f"HTTP {resp.status_code} for {url} "
                    f"(attempt {attempt + 1}/{retries})"
                )
                time.sleep(RETRY_BACKOFF * (attempt + 1))
        except requests.exceptions.Timeout:
            log.warning(f"Timeout (attempt {attempt + 1}/{retries})")
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
# CDSE Catalog Search (public, no auth needed)
# ---------------------------------------------------------------------------

def search_cdse_catalog(
    product_type: str,
    bbox: List[float],
    start_date: str,
    end_date: str,
    max_results: int = 100,
) -> List[Dict]:
    """Search Copernicus Data Space Ecosystem catalog for Sentinel-3 products.

    The CDSE OData catalog is publicly queryable without authentication.
    Actual download of products requires a free CDSE account.
    """
    min_lon, min_lat, max_lon, max_lat = bbox

    # OData filter for CDSE catalog
    filter_str = (
        f"Collection/Name eq 'SENTINEL-3' "
        f"and Attributes/OData.CSC.StringAttribute/any("
        f"att:att/Name eq 'productType' and att/OData.CSC.StringAttribute/Value eq '{product_type}') "
        f"and ContentDate/Start gt {start_date}T00:00:00.000Z "
        f"and ContentDate/Start lt {end_date}T23:59:59.999Z "
        f"and OData.CSC.Intersects(area=geography'SRID=4326;POLYGON(("
        f"{min_lon} {min_lat},{max_lon} {min_lat},{max_lon} {max_lat},"
        f"{min_lon} {max_lat},{min_lon} {min_lat}))')"
    )

    params = {
        "$filter": filter_str,
        "$top": max_results,
        "$orderby": "ContentDate/Start desc",
    }

    log.info(f"  Searching CDSE for {product_type} over [{min_lon},{min_lat},{max_lon},{max_lat}]")
    resp = _rate_limited_get(CDSE_CATALOG, params=params)

    if resp is None:
        return []

    try:
        data = resp.json()
        products = data.get("value", [])
        results = []
        for p in products:
            results.append({
                "id": p.get("Id", ""),
                "name": p.get("Name", ""),
                "start_date": p.get("ContentDate", {}).get("Start", ""),
                "end_date": p.get("ContentDate", {}).get("End", ""),
                "size_mb": round(p.get("ContentLength", 0) / 1e6, 1),
                "online": p.get("Online", False),
                "product_type": product_type,
            })
        return results
    except Exception as e:
        log.error(f"Failed to parse CDSE response: {e}")
        return []


def discover_sentinel3_products(
    output_dir: Path,
    regions: List[str],
    product_types: List[str],
    start_date: str,
    end_date: str,
    max_results_per_query: int = 50,
) -> Dict[str, List[Dict]]:
    """Discover available Sentinel-3 products across all regions."""
    catalog_path = output_dir / "sentinel3_product_catalog.json"
    if catalog_path.exists():
        with open(catalog_path) as f:
            return json.load(f)

    all_products = {}

    for region_key in regions:
        region = REGIONS.get(region_key)
        if region is None:
            log.warning(f"Unknown region: {region_key}")
            continue

        region_products = []
        for pt in product_types:
            products = search_cdse_catalog(
                product_type=pt,
                bbox=region["bbox"],
                start_date=start_date,
                end_date=end_date,
                max_results=max_results_per_query,
            )
            region_products.extend(products)
            log.info(
                f"    {region['name']} / {pt}: {len(products)} products found"
            )

        all_products[region_key] = region_products

    with open(catalog_path, "w") as f:
        json.dump(all_products, f, indent=2)
    total = sum(len(v) for v in all_products.values())
    log.info(f"Saved catalog with {total} total products to {catalog_path.name}")
    return all_products


# ---------------------------------------------------------------------------
# CMEMS Water Quality Data (public metadata)
# ---------------------------------------------------------------------------

def search_cmems_datasets(output_dir: Path) -> List[Dict]:
    """Search Copernicus Marine Service for water quality datasets.

    CMEMS provides analysis and reanalysis products for ocean and
    coastal water quality including chlorophyll, turbidity, and more.
    """
    catalog_path = output_dir / "cmems_wq_catalog.json"
    if catalog_path.exists():
        with open(catalog_path) as f:
            return json.load(f)

    log.info("Searching CMEMS for water quality datasets...")

    # Known CMEMS dataset IDs for water quality
    cmems_datasets = [
        {
            "id": "OCEANCOLOUR_GLO_BGC_L3_MY_009_103",
            "title": "Global Ocean Colour (CHL, SPM, KD, PP) L3 Multi-Sensor",
            "variables": ["chlorophyll_a", "suspended_matter", "kd490"],
            "resolution": "4km daily/monthly",
        },
        {
            "id": "OCEANCOLOUR_GLO_BGC_L4_MY_009_104",
            "title": "Global Ocean Colour (CHL, PP) L4 Gap-Free",
            "variables": ["chlorophyll_a", "primary_production"],
            "resolution": "4km daily/monthly",
        },
        {
            "id": "OCEANCOLOUR_ATL_BGC_L3_MY_009_111",
            "title": "Atlantic Ocean Colour (CHL, SPM, KD) L3",
            "variables": ["chlorophyll_a", "suspended_matter"],
            "resolution": "1km daily",
        },
        {
            "id": "OCEANCOLOUR_GLO_BGC_L3_NRT_009_101",
            "title": "Global Ocean Colour (CHL, SPM) L3 Near-Real-Time",
            "variables": ["chlorophyll_a", "suspended_matter", "turbidity"],
            "resolution": "4km daily",
        },
    ]

    # Try to discover more via CMEMS STAC catalog
    stac_url = f"{CMEMS_STAC}/search"
    params = {"q": "chlorophyll water quality", "limit": 20}
    resp = _rate_limited_get(stac_url, params=params)
    if resp is not None:
        try:
            data = resp.json()
            features = data.get("features", data.get("collections", []))
            for feat in features:
                ds_id = feat.get("id", "")
                if ds_id and not any(d["id"] == ds_id for d in cmems_datasets):
                    cmems_datasets.append({
                        "id": ds_id,
                        "title": feat.get("title", feat.get("description", ""))[:100],
                        "variables": [],
                        "resolution": "",
                        "source": "stac_discovery",
                    })
        except Exception as e:
            log.warning(f"CMEMS STAC parse error: {e}")

    with open(catalog_path, "w") as f:
        json.dump(cmems_datasets, f, indent=2)
    log.info(f"Found {len(cmems_datasets)} CMEMS water quality datasets")
    return cmems_datasets


# ---------------------------------------------------------------------------
# Authentication helper
# ---------------------------------------------------------------------------

def get_cdse_token(username: str, password: str) -> Optional[str]:
    """Get access token from CDSE identity service.

    Requires free registration at https://dataspace.copernicus.eu/
    """
    try:
        resp = requests.post(
            CDSE_TOKEN_URL,
            data={
                "client_id": "cdse-public",
                "grant_type": "password",
                "username": username,
                "password": password,
            },
            timeout=30,
        )
        if resp.status_code == 200:
            token = resp.json().get("access_token")
            log.info("CDSE authentication successful")
            return token
        else:
            log.error(f"CDSE auth failed: HTTP {resp.status_code}")
            return None
    except Exception as e:
        log.error(f"CDSE auth error: {e}")
        return None


def download_product(
    product_id: str,
    product_name: str,
    output_dir: Path,
    token: str,
) -> bool:
    """Download a single Sentinel-3 product from CDSE (requires auth token)."""
    output_path = output_dir / f"{product_name}.zip"
    if output_path.exists() and output_path.stat().st_size > 1000:
        log.info(f"  Already exists: {product_name}")
        return True

    url = f"https://zipper.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"
    headers = {"Authorization": f"Bearer {token}"}

    try:
        time.sleep(REQUEST_DELAY)
        resp = requests.get(url, headers=headers, stream=True, timeout=300)
        if resp.status_code == 200:
            with open(output_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            size_mb = output_path.stat().st_size / 1e6
            log.info(f"  Downloaded {product_name} ({size_mb:.1f} MB)")
            return True
        else:
            log.warning(f"  Download failed: HTTP {resp.status_code}")
            return False
    except Exception as e:
        log.error(f"  Download error: {e}")
        return False


# ---------------------------------------------------------------------------
# Save authentication documentation
# ---------------------------------------------------------------------------

def save_auth_docs(output_dir: Path):
    """Save documentation for how to set up authenticated data access."""
    doc = {
        "sentinel3_data_access": {
            "overview": (
                "Sentinel-3 OLCI water quality data requires free registration "
                "with the Copernicus Data Space Ecosystem (CDSE). The catalog "
                "search is public, but downloading actual products requires "
                "authentication."
            ),
            "steps": [
                "1. Register at https://dataspace.copernicus.eu/ (free account)",
                "2. Set environment variables: CDSE_USERNAME and CDSE_PASSWORD",
                "3. Re-run this script: python scripts/download_sentinel3.py",
                "4. The script will authenticate and download products",
            ],
            "env_vars": {
                "CDSE_USERNAME": "Your CDSE account email",
                "CDSE_PASSWORD": "Your CDSE account password",
            },
            "alternative_access": {
                "copernicus_browser": (
                    "Browse and manually download products at "
                    "https://browser.dataspace.copernicus.eu/"
                ),
                "sentinelsat": (
                    "Use the sentinelsat Python package for programmatic access: "
                    "pip install sentinelsat"
                ),
                "cdsetool": (
                    "Use the cdsetool package: pip install cdsetool"
                ),
            },
            "products_of_interest": {
                "OL_2_WFR": "Water Full Resolution (300m) — chlorophyll, TSM, turbidity",
                "OL_2_WRR": "Water Reduced Resolution (1.2km) — same variables, lower res",
            },
            "water_quality_variables": {
                "CHL_OC4ME": "Chlorophyll-a concentration (OC4ME algorithm)",
                "CHL_NN": "Chlorophyll-a concentration (neural network algorithm)",
                "TSM_NN": "Total Suspended Matter (neural network algorithm)",
                "ADG443_NN": "Absorption by CDOM and detritus at 443nm",
                "KD490_M07": "Diffuse attenuation coefficient at 490nm",
            },
        },
        "cmems_access": {
            "overview": (
                "Copernicus Marine Service provides gridded analysis products. "
                "Free registration required at https://marine.copernicus.eu/"
            ),
            "python_package": "pip install copernicusmarine",
            "example": (
                "import copernicusmarine; "
                "copernicusmarine.subset("
                "dataset_id='OCEANCOLOUR_GLO_BGC_L3_MY_009_103', "
                "variables=['CHL'], "
                "minimum_longitude=-92.5, maximum_longitude=-76.0, "
                "minimum_latitude=41.0, maximum_latitude=49.5, "
                "start_datetime='2023-01-01', end_datetime='2023-12-31')"
            ),
        },
    }
    with open(output_dir / "authentication_guide.json", "w") as f:
        json.dump(doc, f, indent=2)
    log.info("Saved authentication guide to authentication_guide.json")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download Sentinel-3 OLCI water quality data"
    )
    parser.add_argument(
        "--data-dir",
        default="/home/bcheng/SENTINEL/data/processed/satellite/sentinel3",
        help="Output directory for downloaded data",
    )
    parser.add_argument(
        "--regions",
        nargs="+",
        default=["great_lakes", "chesapeake", "florida_lakes"],
        choices=list(REGIONS.keys()),
        help="Regions to search for products",
    )
    parser.add_argument(
        "--start-date",
        default="2023-01-01",
        help="Start date for product search (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        default="2025-12-31",
        help="End date for product search (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--product-types",
        nargs="+",
        default=["OL_2_WFR___", "OL_2_WRR___"],
        help="Sentinel-3 product types to search",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=50,
        help="Maximum results per search query",
    )
    parser.add_argument(
        "--discover-only",
        action="store_true",
        help="Only discover products, do not attempt download",
    )
    parser.add_argument(
        "--max-download",
        type=int,
        default=10,
        help="Maximum number of products to download (requires auth)",
    )
    args = parser.parse_args()

    output_dir = Path(args.data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 70)
    log.info("Sentinel-3 OLCI Water Quality Data Pipeline")
    log.info(f"  Output: {output_dir}")
    log.info(f"  Regions: {args.regions}")
    log.info(f"  Date range: {args.start_date} to {args.end_date}")
    log.info(f"  Products: {args.product_types}")
    log.info("=" * 70)

    # Step 1: Discover Sentinel-3 products via CDSE catalog (public)
    log.info("STEP 1: Discovering Sentinel-3 products (CDSE catalog)")
    log.info("-" * 70)
    catalog = discover_sentinel3_products(
        output_dir=output_dir,
        regions=args.regions,
        product_types=args.product_types,
        start_date=args.start_date,
        end_date=args.end_date,
        max_results_per_query=args.max_results,
    )
    total_products = sum(len(v) for v in catalog.values())
    for region_key, products in catalog.items():
        region_name = REGIONS.get(region_key, {}).get("name", region_key)
        log.info(f"  {region_name}: {len(products)} products found")
        for p in products[:3]:
            log.info(f"    {p['name'][:60]}  ({p['size_mb']:.0f} MB)")

    # Step 2: Search CMEMS water quality datasets
    log.info("=" * 70)
    log.info("STEP 2: CMEMS water quality dataset catalog")
    log.info("-" * 70)
    cmems_datasets = search_cmems_datasets(output_dir)
    for ds in cmems_datasets:
        log.info(f"  [{ds['id']}] {ds['title'][:60]}")

    # Step 3: Save authentication documentation
    log.info("=" * 70)
    log.info("STEP 3: Authentication setup")
    log.info("-" * 70)
    save_auth_docs(output_dir)

    # Step 4: Attempt download if credentials are available
    downloaded = 0
    if not args.discover_only:
        cdse_user = os.environ.get("CDSE_USERNAME")
        cdse_pass = os.environ.get("CDSE_PASSWORD")

        if cdse_user and cdse_pass:
            log.info("=" * 70)
            log.info("STEP 4: Downloading products (authenticated)")
            log.info("-" * 70)
            token = get_cdse_token(cdse_user, cdse_pass)
            if token:
                products_to_download = []
                for region_products in catalog.values():
                    products_to_download.extend(region_products)
                products_to_download = products_to_download[: args.max_download]

                raw_dir = output_dir / "raw"
                raw_dir.mkdir(exist_ok=True)

                for p in products_to_download:
                    if download_product(p["id"], p["name"], raw_dir, token):
                        downloaded += 1
                log.info(f"Downloaded {downloaded}/{len(products_to_download)} products")
            else:
                log.warning("Authentication failed. Set CDSE_USERNAME/CDSE_PASSWORD.")
        else:
            log.info("=" * 70)
            log.info("STEP 4: Skipping download (no CDSE credentials)")
            log.info(
                "  Set CDSE_USERNAME and CDSE_PASSWORD environment variables "
                "to enable downloads."
            )
            log.info("  See authentication_guide.json for setup instructions.")
            log.info("-" * 70)

    # Step 5: Summary
    log.info("=" * 70)
    log.info("PIPELINE COMPLETE")
    log.info(f"  Products discovered: {total_products}")
    log.info(f"  Products downloaded: {downloaded}")
    log.info(f"  CMEMS datasets cataloged: {len(cmems_datasets)}")
    log.info(f"  Output dir: {output_dir}")
    log.info(f"  Catalog: sentinel3_product_catalog.json")
    log.info(f"  CMEMS catalog: cmems_wq_catalog.json")
    log.info(f"  Auth guide: authentication_guide.json")
    for region_key in args.regions:
        region = REGIONS.get(region_key, {})
        products = catalog.get(region_key, [])
        total_size = sum(p.get("size_mb", 0) for p in products)
        log.info(
            f"  {region.get('name', region_key)}: "
            f"{len(products)} products, {total_size:.0f} MB total"
        )
    log.info("=" * 70)

    # Save final metadata
    metadata = {
        "download_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "regions": args.regions,
        "date_range": [args.start_date, args.end_date],
        "product_types": args.product_types,
        "products_discovered": total_products,
        "products_downloaded": downloaded,
        "cmems_datasets": len(cmems_datasets),
        "regions_info": {
            k: {
                "name": v["name"],
                "bbox": v["bbox"],
                "products_found": len(catalog.get(k, [])),
            }
            for k, v in REGIONS.items()
            if k in args.regions
        },
        "wq_parameters": WQ_PARAMETERS,
        "authentication_required": True,
        "auth_guide": "authentication_guide.json",
    }
    with open(output_dir / "download_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    main()
