#!/usr/bin/env python3
"""Download real biological/invertebrate data from USGS BioData and NWIS.

Sources:
1. USGS BioData (aquatic bioassessment) — macroinvertebrate community samples
   and fish assemblage data via https://aquatic.biodata.usgs.gov/
2. USGS NWIS (via dataretrieval package) — biological water quality parameters
   including chlorophyll, algae, and related measurements.
3. EPA Water Quality Portal (WQP) — biological results from the national
   WQP (https://www.waterqualitydata.us/).

Targets: Great Lakes region, Ohio River basin, Chesapeake Bay watershed.

Usage:
    python scripts/download_usgs_biodata.py
    python scripts/download_usgs_biodata.py --huc-regions 04 05 02
    python scripts/download_usgs_biodata.py --max-sites 200 --skip-wqp

MIT License -- Bryan Cheng, 2026
"""

import argparse
import json
import logging
import os
import sys
import time
from io import StringIO
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
log = logging.getLogger("download_usgs_biodata")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REQUEST_DELAY = 1.0
MAX_RETRIES = 3
RETRY_BACKOFF = 5.0
TIMEOUT = 90

# USGS BioData web service endpoints
BIODATA_BASE = "https://aquatic.biodata.usgs.gov"

# WQP endpoints
WQP_BASE = "https://www.waterqualitydata.us"

# HUC2 regions of interest
TARGET_HUC_REGIONS = {
    "04": "Great Lakes",
    "05": "Ohio River Basin",
    "02": "Mid-Atlantic / Chesapeake Bay",
    "03": "South Atlantic-Gulf",
    "07": "Upper Mississippi",
    "06": "Tennessee",
}

# Biological characteristic names for WQP queries
BIO_CHARACTERISTICS = [
    "Chlorophyll a",
    "Chlorophyll a, corrected for pheophytin",
    "Chlorophyll a, free of pheophytin",
    "Phycocyanin",
    "Pheophytin a",
    "Benthic Macroinvertebrates",
    "Fish Species Richness",
    "Macroinvertebrate, Species Richness",
    "Total Coliform",
    "Enterococcus",
    "Escherichia coli",
    "Chlorophyll",
    "Periphyton",
]

# Biological parameter codes for NWIS
NWIS_BIO_PARAMS = {
    "70953": "Chlorophyll a",
    "70954": "Chlorophyll b",
    "62360": "Pheophytin a",
    "70969": "Phycocyanin",
    "31501": "Total Coliform",
    "31648": "E. coli",
    "61028": "Enterococcus",
    "32211": "Chlorophyll a, phytoplankton",
}

# State codes per HUC2 region (for NWIS queries)
HUC_STATES = {
    "04": ["MI", "WI", "MN", "OH", "IN", "IL", "PA", "NY"],
    "05": ["OH", "PA", "WV", "KY", "IN", "VA"],
    "02": ["PA", "NY", "NJ", "DE", "MD", "VA", "DC", "CT"],
    "03": ["GA", "FL", "AL", "SC", "NC", "VA", "MS"],
    "07": ["MN", "WI", "IA", "IL", "MO", "IN"],
    "06": ["TN", "AL", "KY", "GA", "NC", "VA", "MS"],
}


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
# BioData web service queries
# ---------------------------------------------------------------------------

def download_biodata_invertebrates(
    output_dir: Path,
    huc_regions: List[str],
) -> Optional[pd.DataFrame]:
    """Download macroinvertebrate community data via the Water Quality Portal.

    BioData content has been migrated to the WQP.  We query for biological
    community count data and filter to invertebrate taxa.
    """
    output_path = output_dir / "biodata_invertebrates.parquet"
    if output_path.exists() and output_path.stat().st_size > 500:
        log.info(f"Already exists: {output_path.name}")
        try:
            return pd.read_parquet(output_path)
        except Exception:
            pass

    all_records = []

    # Use dataretrieval to query WQP for biological community data
    try:
        import dataretrieval.wqp as wqp
        use_dataretrieval = True
    except ImportError:
        use_dataretrieval = False

    for huc in huc_regions:
        huc_name = TARGET_HUC_REGIONS.get(huc, huc)
        log.info(f"  Querying WQP biological community data for HUC {huc} ({huc_name})...")

        if use_dataretrieval:
            try:
                df, _ = wqp.get_results(
                    huc=huc,
                    characteristicName="Count",
                    startDateLo="01-01-2015",
                    startDateHi="12-31-2025",
                )
                if df is not None and len(df) > 0:
                    # Keep only Biological media (invertebrate/fish community data)
                    if "ActivityMediaName" in df.columns:
                        df = df[df["ActivityMediaName"] == "Biological"].copy()
                    df["_huc2"] = huc
                    df["_source"] = "WQP_BioData"
                    all_records.append(df)
                    log.info(f"    HUC {huc}: {len(df):,} biological community records")
                else:
                    log.warning(f"    HUC {huc}: no community data returned")
                time.sleep(REQUEST_DELAY)
            except Exception as e:
                log.warning(f"    HUC {huc}: dataretrieval error — {e}")
                time.sleep(REQUEST_DELAY)
        else:
            # Direct HTTP fallback
            url = f"{WQP_BASE}/data/Result/search"
            params = {
                "huc": huc,
                "characteristicName": "Count",
                "startDateLo": "01-01-2015",
                "startDateHi": "12-31-2025",
                "mimeType": "csv",
                "sorted": "no",
                "zip": "no",
            }
            resp = _rate_limited_get(url, params=params, headers={"Accept": "text/csv"}, timeout=180)
            if resp is not None and len(resp.text) > 100:
                try:
                    df = pd.read_csv(StringIO(resp.text))
                    if "ActivityMediaName" in df.columns:
                        df = df[df["ActivityMediaName"] == "Biological"].copy()
                    df["_huc2"] = huc
                    df["_source"] = "WQP_BioData"
                    all_records.append(df)
                    log.info(f"    HUC {huc}: {len(df):,} biological community records")
                except Exception as e:
                    log.warning(f"    HUC {huc}: CSV parse failed — {e}")
            else:
                log.warning(f"    HUC {huc}: no community data retrieved via HTTP")

    if all_records:
        df = pd.concat(all_records, ignore_index=True)

        # Separate invertebrates from fish using SubjectTaxonomicName
        # Common invertebrate orders/families for filtering
        invertebrate_keywords = [
            "Ephemeroptera", "Plecoptera", "Trichoptera", "Diptera",
            "Coleoptera", "Odonata", "Megaloptera", "Hemiptera",
            "Chironomidae", "Chironominae", "Oligochaeta", "Gastropoda",
            "Pelecypoda", "Amphipoda", "Isopoda", "Decapoda",
            "Turbellaria", "Hirudinea", "Hydracarina", "Nematoda",
            "Annelida", "Mollusca", "Crustacea", "Insecta",
            "Cheumatopsyche", "Polypedilum", "Hydropsychidae",
            "Baetidae", "Heptageniidae", "Simuliidae", "Tipulidae",
            "Lumbriculata", "Haplotaxida", "Tubificidae",
        ]
        fish_families = [
            "Cyprinidae", "Centrarchidae", "Percidae", "Catostomidae",
            "Ictaluridae", "Salmonidae", "Cottidae", "Petromyzontidae",
            "Semotilus", "Rhinichthys", "Lepomis", "Micropterus",
            "Etheostoma", "Catostomus", "Notropis", "Pimephales",
        ]

        if "SubjectTaxonomicName" in df.columns:
            # Build a boolean mask for invertebrates
            taxon_col = df["SubjectTaxonomicName"].fillna("").str.lower()
            invert_mask = pd.Series(False, index=df.index)
            for kw in invertebrate_keywords:
                invert_mask |= taxon_col.str.contains(kw.lower(), na=False)

            fish_mask = pd.Series(False, index=df.index)
            for kw in fish_families:
                fish_mask |= taxon_col.str.contains(kw.lower(), na=False)

            # Invertebrates: matched invertebrate keywords OR not matched as fish
            invert_df = df[invert_mask | (~fish_mask & ~invert_mask)].copy()
            fish_df = df[fish_mask & ~invert_mask].copy()

            # Save fish separately for the fish function to pick up
            if len(fish_df) > 0:
                fish_path = output_dir / "biodata_fish.parquet"
                if not fish_path.exists():
                    for col in fish_df.columns:
                        if fish_df[col].dtype == object:
                            numeric = pd.to_numeric(fish_df[col], errors="coerce")
                            non_null = fish_df[col].dropna()
                            if len(non_null) > 0 and numeric.notna().sum() / max(len(non_null), 1) > 0.5:
                                fish_df[col] = numeric
                            else:
                                fish_df[col] = fish_df[col].astype(str).replace("nan", pd.NA)
                    fish_df.to_parquet(fish_path, index=False)
                    log.info(f"  Also saved {len(fish_df):,} fish records")

            df = invert_df

        # Fix mixed-type columns for parquet compatibility
        for col in df.columns:
            if df[col].dtype == object:
                numeric = pd.to_numeric(df[col], errors="coerce")
                non_null = df[col].dropna()
                if len(non_null) > 0 and numeric.notna().sum() / max(len(non_null), 1) > 0.5:
                    df[col] = numeric
                else:
                    df[col] = df[col].astype(str).replace("nan", pd.NA)
        df.to_parquet(output_path, index=False)
        log.info(f"Saved {len(df):,} total invertebrate/community records")
        return df
    else:
        log.warning("No biological community records retrieved.")
        return None


def download_biodata_fish(
    output_dir: Path,
    huc_regions: List[str],
) -> Optional[pd.DataFrame]:
    """Download fish assemblage data via WQP (formerly USGS BioData).

    If the invertebrate download already separated and saved fish records,
    this will pick up that file.  Otherwise it queries WQP directly for
    fish-specific biological community data.
    """
    output_path = output_dir / "biodata_fish.parquet"
    if output_path.exists() and output_path.stat().st_size > 500:
        log.info(f"Already exists: {output_path.name}")
        try:
            return pd.read_parquet(output_path)
        except Exception:
            pass

    all_records = []

    # Fish-specific characteristic names in WQP
    fish_char_names = [
        "Count",
        "Fish abundance, number",
        "Fish Species Richness",
    ]

    fish_keywords = [
        "cyprinidae", "centrarchidae", "percidae", "catostomidae",
        "ictaluridae", "salmonidae", "cottidae", "petromyzontidae",
        "semotilus", "rhinichthys", "lepomis", "micropterus",
        "etheostoma", "catostomus", "notropis", "pimephales",
        "campostoma", "luxilus", "ambloplites", "sander",
        "oncorhynchus", "salvelinus", "esox", "perca",
        "fundulus", "gambusia", "moxostoma", "hypentelium",
    ]

    try:
        import dataretrieval.wqp as wqp
        use_dataretrieval = True
    except ImportError:
        use_dataretrieval = False

    for huc in huc_regions:
        huc_name = TARGET_HUC_REGIONS.get(huc, huc)
        log.info(f"  Querying WQP fish data for HUC {huc} ({huc_name})...")

        for char_name in fish_char_names:
            if use_dataretrieval:
                try:
                    df, _ = wqp.get_results(
                        huc=huc,
                        characteristicName=char_name,
                        startDateLo="01-01-2015",
                        startDateHi="12-31-2025",
                    )
                    if df is not None and len(df) > 0:
                        # Filter to biological media
                        if "ActivityMediaName" in df.columns:
                            df = df[df["ActivityMediaName"] == "Biological"].copy()
                        # Filter to fish taxa
                        if "SubjectTaxonomicName" in df.columns and len(df) > 0:
                            taxon_col = df["SubjectTaxonomicName"].fillna("").str.lower()
                            fish_mask = pd.Series(False, index=df.index)
                            for kw in fish_keywords:
                                fish_mask |= taxon_col.str.contains(kw, na=False)
                            df = df[fish_mask].copy()
                        if len(df) > 0:
                            df["_huc2"] = huc
                            df["_source"] = "WQP_BioData"
                            df["_characteristic_query"] = char_name
                            all_records.append(df)
                            log.info(f"    HUC {huc}/{char_name}: {len(df):,} fish records")
                    time.sleep(REQUEST_DELAY)
                except Exception as e:
                    log.warning(f"    HUC {huc}/{char_name}: error — {e}")
                    time.sleep(REQUEST_DELAY)
            else:
                url = f"{WQP_BASE}/data/Result/search"
                params = {
                    "huc": huc,
                    "characteristicName": char_name,
                    "startDateLo": "01-01-2015",
                    "startDateHi": "12-31-2025",
                    "mimeType": "csv",
                    "sorted": "no",
                    "zip": "no",
                }
                resp = _rate_limited_get(url, params=params, headers={"Accept": "text/csv"}, timeout=180)
                if resp is not None and len(resp.text) > 100:
                    try:
                        df = pd.read_csv(StringIO(resp.text))
                        if "ActivityMediaName" in df.columns:
                            df = df[df["ActivityMediaName"] == "Biological"].copy()
                        if "SubjectTaxonomicName" in df.columns and len(df) > 0:
                            taxon_col = df["SubjectTaxonomicName"].fillna("").str.lower()
                            fish_mask = pd.Series(False, index=df.index)
                            for kw in fish_keywords:
                                fish_mask |= taxon_col.str.contains(kw, na=False)
                            df = df[fish_mask].copy()
                        if len(df) > 0:
                            df["_huc2"] = huc
                            df["_source"] = "WQP_BioData"
                            df["_characteristic_query"] = char_name
                            all_records.append(df)
                            log.info(f"    HUC {huc}/{char_name}: {len(df):,} fish records")
                    except Exception as e:
                        log.warning(f"    HUC {huc}/{char_name}: CSV parse failed — {e}")

    if all_records:
        df = pd.concat(all_records, ignore_index=True)
        # Deduplicate by ActivityIdentifier if available
        if "ActivityIdentifier" in df.columns and "SubjectTaxonomicName" in df.columns:
            df = df.drop_duplicates(
                subset=["ActivityIdentifier", "SubjectTaxonomicName"]
            )
        # Fix mixed-type columns for parquet compatibility
        for col in df.columns:
            if df[col].dtype == object:
                numeric = pd.to_numeric(df[col], errors="coerce")
                non_null = df[col].dropna()
                if len(non_null) > 0 and numeric.notna().sum() / max(len(non_null), 1) > 0.5:
                    df[col] = numeric
                else:
                    df[col] = df[col].astype(str).replace("nan", pd.NA)
        df.to_parquet(output_path, index=False)
        log.info(f"Saved {len(df):,} total fish records")
        return df
    else:
        log.warning("No fish records retrieved from WQP.")
        return None


def download_biodata_sites(
    output_dir: Path,
    huc_regions: List[str],
) -> Optional[pd.DataFrame]:
    """Download biological monitoring site metadata via WQP.

    Queries the WQP for sites that have biological community count data
    (the data formerly served by USGS BioData at aquatic.biodata.usgs.gov).
    """
    output_path = output_dir / "biodata_sites.parquet"
    if output_path.exists() and output_path.stat().st_size > 500:
        log.info(f"Already exists: {output_path.name}")
        try:
            return pd.read_parquet(output_path)
        except Exception:
            pass

    all_sites = []

    try:
        import dataretrieval.wqp as wqp
        use_dataretrieval = True
    except ImportError:
        use_dataretrieval = False

    for huc in huc_regions:
        huc_name = TARGET_HUC_REGIONS.get(huc, huc)
        log.info(f"  Querying WQP biological sites for HUC {huc} ({huc_name})...")

        if use_dataretrieval:
            try:
                df, _ = wqp.what_sites(
                    huc=huc,
                    characteristicName="Count",
                    siteType="Stream",
                )
                if df is not None and len(df) > 0:
                    df["_huc2"] = huc
                    all_sites.append(df)
                    log.info(f"    HUC {huc}: {len(df):,} biological sites")
                else:
                    log.warning(f"    HUC {huc}: no sites returned")
                time.sleep(REQUEST_DELAY)
            except Exception as e:
                log.warning(f"    HUC {huc}: site query error — {e}")
                time.sleep(REQUEST_DELAY)
        else:
            url = f"{WQP_BASE}/data/Station/search"
            params = {
                "huc": huc,
                "characteristicName": "Count",
                "siteType": "Stream",
                "mimeType": "csv",
                "sorted": "no",
                "zip": "no",
            }
            resp = _rate_limited_get(
                url, params=params, headers={"Accept": "text/csv"}, timeout=120
            )
            if resp is not None and len(resp.text) > 100:
                try:
                    df = pd.read_csv(StringIO(resp.text))
                    df["_huc2"] = huc
                    all_sites.append(df)
                    log.info(f"    HUC {huc}: {len(df):,} biological sites")
                except Exception:
                    pass

    if all_sites:
        df = pd.concat(all_sites, ignore_index=True)
        # Deduplicate by site identifier
        for col in ["MonitoringLocationIdentifier", "SiteNumber", "site_no"]:
            if col in df.columns:
                df = df.drop_duplicates(subset=[col])
                break
        # Fix mixed-type columns for parquet compatibility
        for col in df.columns:
            if df[col].dtype == object:
                numeric = pd.to_numeric(df[col], errors="coerce")
                non_null = df[col].dropna()
                if len(non_null) > 0 and numeric.notna().sum() / max(len(non_null), 1) > 0.5:
                    df[col] = numeric
                else:
                    df[col] = df[col].astype(str).replace("nan", pd.NA)
        df.to_parquet(output_path, index=False)
        log.info(f"Saved {len(df):,} unique biological monitoring sites")
        return df
    return None


# ---------------------------------------------------------------------------
# WQP Biological Data
# ---------------------------------------------------------------------------

def download_wqp_biological(
    output_dir: Path,
    huc_regions: List[str],
    characteristics: List[str],
    start_date: str = "01-01-2015",
    end_date: str = "12-31-2025",
    max_records_per_huc: int = 100000,
) -> Optional[pd.DataFrame]:
    """Download biological results from EPA Water Quality Portal.

    Uses the dataretrieval package when available, with a direct HTTP
    fallback to the WQP REST API.
    """
    output_path = output_dir / "wqp_biological.parquet"
    if output_path.exists() and output_path.stat().st_size > 500:
        log.info(f"Already exists: {output_path.name}")
        try:
            return pd.read_parquet(output_path)
        except Exception:
            pass

    all_dfs = []

    # Try using dataretrieval package first
    try:
        import dataretrieval.wqp as wqp
        use_dataretrieval = True
        log.info("Using dataretrieval package for WQP queries")
    except ImportError:
        use_dataretrieval = False
        log.info("dataretrieval not available; using direct HTTP")

    for huc in huc_regions:
        huc_name = TARGET_HUC_REGIONS.get(huc, huc)
        log.info(f"  Querying WQP biological data for HUC {huc} ({huc_name})...")

        for char in characteristics:
            if use_dataretrieval:
                try:
                    df, _ = wqp.get_results(
                        huc=huc,
                        characteristicName=char,
                        startDateLo=start_date,
                        startDateHi=end_date,
                    )
                    if df is not None and len(df) > 0:
                        df["_huc2"] = huc
                        df["_characteristic_query"] = char
                        all_dfs.append(df)
                        log.info(f"    {char}: {len(df):,} records")
                    time.sleep(REQUEST_DELAY)
                except Exception as e:
                    log.warning(f"    {char}: dataretrieval error — {e}")
                    time.sleep(REQUEST_DELAY)
            else:
                # Direct HTTP to WQP API
                url = f"{WQP_BASE}/data/Result/search"
                params = {
                    "huc": huc,
                    "characteristicName": char,
                    "startDateLo": start_date,
                    "startDateHi": end_date,
                    "mimeType": "csv",
                    "sorted": "no",
                    "zip": "no",
                }
                headers = {"Accept": "text/csv"}
                resp = _rate_limited_get(
                    url, params=params, headers=headers, timeout=120
                )
                if resp is not None and len(resp.text) > 100:
                    try:
                        df = pd.read_csv(StringIO(resp.text))
                        df["_huc2"] = huc
                        df["_characteristic_query"] = char
                        all_dfs.append(df)
                        log.info(f"    {char}: {len(df):,} records")
                    except Exception as e:
                        log.warning(f"    {char}: CSV parse error — {e}")

    if all_dfs:
        df = pd.concat(all_dfs, ignore_index=True)
        # Trim to max records if needed
        if len(df) > max_records_per_huc * len(huc_regions):
            log.info(f"  Trimming from {len(df):,} to {max_records_per_huc * len(huc_regions):,}")
            df = df.head(max_records_per_huc * len(huc_regions))
        # Fix mixed-type columns for parquet compatibility
        for col in df.columns:
            if df[col].dtype == object:
                numeric = pd.to_numeric(df[col], errors="coerce")
                non_null = df[col].dropna()
                if len(non_null) > 0 and numeric.notna().sum() / max(len(non_null), 1) > 0.5:
                    df[col] = numeric
                else:
                    df[col] = df[col].astype(str).replace("nan", pd.NA)
        df.to_parquet(output_path, index=False)
        log.info(f"Saved {len(df):,} WQP biological records")
        return df
    else:
        log.warning("No WQP biological records retrieved.")
        return None


# ---------------------------------------------------------------------------
# NWIS biological parameter data
# ---------------------------------------------------------------------------

def download_nwis_bio_sites(
    output_dir: Path,
    huc_regions: List[str],
    max_sites: int = 500,
) -> Optional[pd.DataFrame]:
    """Discover and download NWIS sites with biological parameter data.

    Uses the dataretrieval package to find sites with biological
    water quality parameters (chlorophyll, bacteria counts, etc.)
    """
    output_path = output_dir / "nwis_bio_sites.parquet"
    if output_path.exists() and output_path.stat().st_size > 500:
        log.info(f"Already exists: {output_path.name}")
        try:
            return pd.read_parquet(output_path)
        except Exception:
            pass

    try:
        import dataretrieval.nwis as nwis
    except ImportError:
        log.warning("dataretrieval package not available. Skipping NWIS bio sites.")
        return None

    all_sites = []
    param_codes = list(NWIS_BIO_PARAMS.keys())

    for huc in huc_regions:
        states = HUC_STATES.get(huc, [])
        if not states:
            continue

        for state in states[:3]:  # Limit to first 3 states per HUC
            for pcode in param_codes[:4]:  # Top 4 params
                try:
                    result = nwis.what_sites(
                        stateCd=state,
                        parameterCd=pcode,
                    )
                    df = result[0] if isinstance(result, tuple) else result
                    if df is not None and len(df) > 0:
                        if hasattr(df, "reset_index"):
                            df = df.reset_index()
                        df["_param_code"] = pcode
                        df["_param_name"] = NWIS_BIO_PARAMS[pcode]
                        df["_state"] = state
                        df["_huc2"] = huc
                        all_sites.append(df)
                        log.info(
                            f"    {state}/{NWIS_BIO_PARAMS[pcode]}: "
                            f"{len(df)} sites"
                        )
                except Exception as e:
                    log.warning(f"    {state}/{pcode}: error — {e}")
                time.sleep(0.5)

        if sum(len(d) for d in all_sites) >= max_sites:
            log.info(f"  Reached {max_sites} site limit")
            break

    if all_sites:
        df = pd.concat(all_sites, ignore_index=True)
        # Deduplicate
        if "site_no" in df.columns:
            df = df.drop_duplicates(subset=["site_no", "_param_code"])
        # Fix mixed-type columns for parquet compatibility
        for col in df.columns:
            if df[col].dtype == object:
                numeric = pd.to_numeric(df[col], errors="coerce")
                non_null = df[col].dropna()
                if len(non_null) > 0 and numeric.notna().sum() / max(len(non_null), 1) > 0.5:
                    df[col] = numeric
                else:
                    df[col] = df[col].astype(str).replace("nan", pd.NA)
        df.to_parquet(output_path, index=False)
        log.info(f"Saved {len(df):,} NWIS bio site records")
        return df
    return None


def download_nwis_bio_data(
    output_dir: Path,
    sites_df: Optional[pd.DataFrame],
    max_sites: int = 100,
) -> Optional[pd.DataFrame]:
    """Download actual bio measurement data from NWIS for discovered sites."""
    output_path = output_dir / "nwis_bio_data.parquet"
    if output_path.exists() and output_path.stat().st_size > 500:
        log.info(f"Already exists: {output_path.name}")
        try:
            return pd.read_parquet(output_path)
        except Exception:
            pass

    if sites_df is None or len(sites_df) == 0:
        log.warning("No sites available for NWIS bio data download.")
        return None

    try:
        import dataretrieval.nwis as nwis
    except ImportError:
        log.warning("dataretrieval not available.")
        return None

    # Get unique site numbers
    if "site_no" not in sites_df.columns:
        log.warning("No site_no column in sites data.")
        return None

    unique_sites = sites_df["site_no"].unique()[:max_sites]
    log.info(f"Downloading bio data for {len(unique_sites)} NWIS sites...")

    all_data = []
    param_codes = list(NWIS_BIO_PARAMS.keys())

    for i, site_no in enumerate(unique_sites):
        try:
            df, _ = nwis.get_qwdata(
                sites=str(site_no),
                parameterCd=param_codes,
                start="2015-01-01",
                end="2025-12-31",
            )
            if df is not None and len(df) > 0:
                if hasattr(df, "reset_index"):
                    df = df.reset_index()
                df["_site_no"] = str(site_no)
                all_data.append(df)
        except Exception as e:
            # get_qwdata might not exist; try get_dv as fallback
            try:
                df, _ = nwis.get_dv(
                    sites=str(site_no),
                    parameterCd=param_codes[:3],
                    start="2015-01-01",
                    end="2025-12-31",
                )
                if df is not None and len(df) > 0:
                    if hasattr(df, "reset_index"):
                        df = df.reset_index()
                    df["_site_no"] = str(site_no)
                    all_data.append(df)
            except Exception:
                pass

        if (i + 1) % 20 == 0:
            log.info(f"  Progress: {i + 1}/{len(unique_sites)} sites, {len(all_data)} with data")
        time.sleep(0.5)

    if all_data:
        df = pd.concat(all_data, ignore_index=True)
        # Fix mixed-type columns for parquet compatibility
        for col in df.columns:
            if df[col].dtype == object:
                numeric = pd.to_numeric(df[col], errors="coerce")
                non_null = df[col].dropna()
                if len(non_null) > 0 and numeric.notna().sum() / max(len(non_null), 1) > 0.5:
                    df[col] = numeric
                else:
                    df[col] = df[col].astype(str).replace("nan", pd.NA)
        df.to_parquet(output_path, index=False)
        log.info(f"Saved {len(df):,} NWIS bio data records from {len(all_data)} sites")
        return df
    else:
        log.warning("No NWIS bio data retrieved.")
        return None


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def compute_summary(output_dir: Path) -> Dict[str, Any]:
    """Compute summary statistics for all downloaded data."""
    summary = {"total_files": 0, "total_rows": 0, "datasets": {}}

    for pq_file in sorted(output_dir.glob("*.parquet")):
        try:
            df = pd.read_parquet(pq_file)
            ds_info = {
                "file": pq_file.name,
                "rows": len(df),
                "columns": list(df.columns[:20]),  # First 20 columns
                "size_mb": round(pq_file.stat().st_size / 1e6, 2),
            }

            # Site counts if available
            for site_col in ["site_no", "SiteNumber", "MonitoringLocationIdentifier"]:
                if site_col in df.columns:
                    ds_info["unique_sites"] = int(df[site_col].nunique())
                    break

            # Lat/lon coverage
            for lat_col in ["dec_lat_va", "latitude", "LatitudeMeasure"]:
                if lat_col in df.columns:
                    ds_info["lat_range"] = [
                        float(df[lat_col].min()),
                        float(df[lat_col].max()),
                    ]
                    break
            for lon_col in ["dec_long_va", "longitude", "LongitudeMeasure"]:
                if lon_col in df.columns:
                    ds_info["lon_range"] = [
                        float(df[lon_col].min()),
                        float(df[lon_col].max()),
                    ]
                    break

            summary["datasets"][pq_file.stem] = ds_info
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
        description="Download USGS BioData and biological water quality data"
    )
    parser.add_argument(
        "--data-dir",
        default="/home/bcheng/SENTINEL/data/processed/biology/usgs_biodata",
        help="Output directory for downloaded data",
    )
    parser.add_argument(
        "--huc-regions",
        nargs="+",
        default=["04", "05", "02"],
        help="HUC2 region codes to query (default: Great Lakes, Ohio, Mid-Atlantic)",
    )
    parser.add_argument(
        "--max-sites",
        type=int,
        default=500,
        help="Maximum number of NWIS sites to discover",
    )
    parser.add_argument(
        "--max-nwis-download",
        type=int,
        default=100,
        help="Maximum NWIS sites to download bio data for",
    )
    parser.add_argument(
        "--start-date",
        default="01-01-2015",
        help="Start date for WQP queries (MM-DD-YYYY)",
    )
    parser.add_argument(
        "--end-date",
        default="12-31-2025",
        help="End date for WQP queries (MM-DD-YYYY)",
    )
    parser.add_argument(
        "--skip-biodata",
        action="store_true",
        help="Skip USGS BioData downloads",
    )
    parser.add_argument(
        "--skip-wqp",
        action="store_true",
        help="Skip WQP biological downloads",
    )
    parser.add_argument(
        "--skip-nwis",
        action="store_true",
        help="Skip NWIS biological parameter downloads",
    )
    args = parser.parse_args()

    output_dir = Path(args.data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 70)
    log.info("USGS BioData & Biological Water Quality Download Pipeline")
    log.info(f"  Output: {output_dir}")
    log.info(f"  HUC regions: {args.huc_regions}")
    log.info(f"  Date range: {args.start_date} to {args.end_date}")
    log.info("=" * 70)

    results = {}

    # Step 1: USGS BioData — invertebrates and fish
    if not args.skip_biodata:
        log.info("=" * 70)
        log.info("STEP 1: USGS BioData — Invertebrate & Fish Assemblage Data")
        log.info("=" * 70)

        # Download sites
        sites_df = download_biodata_sites(output_dir, args.huc_regions)
        results["biodata_sites"] = len(sites_df) if sites_df is not None else 0

        # Download invertebrate data
        invert_df = download_biodata_invertebrates(output_dir, args.huc_regions)
        results["biodata_invertebrates"] = len(invert_df) if invert_df is not None else 0

        # Download fish data
        fish_df = download_biodata_fish(output_dir, args.huc_regions)
        results["biodata_fish"] = len(fish_df) if fish_df is not None else 0
    else:
        log.info("Skipping BioData downloads.")

    # Step 2: WQP biological data
    if not args.skip_wqp:
        log.info("=" * 70)
        log.info("STEP 2: EPA Water Quality Portal — Biological Results")
        log.info("=" * 70)
        wqp_df = download_wqp_biological(
            output_dir=output_dir,
            huc_regions=args.huc_regions,
            characteristics=BIO_CHARACTERISTICS,
            start_date=args.start_date,
            end_date=args.end_date,
        )
        results["wqp_biological"] = len(wqp_df) if wqp_df is not None else 0
    else:
        log.info("Skipping WQP biological downloads.")

    # Step 3: NWIS biological parameters
    if not args.skip_nwis:
        log.info("=" * 70)
        log.info("STEP 3: USGS NWIS — Biological Parameter Sites & Data")
        log.info("=" * 70)

        nwis_sites = download_nwis_bio_sites(
            output_dir, args.huc_regions, max_sites=args.max_sites
        )
        results["nwis_bio_sites"] = len(nwis_sites) if nwis_sites is not None else 0

        nwis_data = download_nwis_bio_data(
            output_dir, nwis_sites, max_sites=args.max_nwis_download
        )
        results["nwis_bio_data"] = len(nwis_data) if nwis_data is not None else 0
    else:
        log.info("Skipping NWIS biological downloads.")

    # Step 4: Summary
    log.info("=" * 70)
    log.info("STEP 4: Computing summary statistics")
    log.info("=" * 70)
    summary = compute_summary(output_dir)

    # Save metadata
    metadata = {
        "download_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "huc_regions": args.huc_regions,
        "huc_region_names": {h: TARGET_HUC_REGIONS.get(h, "") for h in args.huc_regions},
        "date_range": [args.start_date, args.end_date],
        "results": results,
        "summary": {
            "total_files": summary["total_files"],
            "total_rows": summary["total_rows"],
        },
        "datasets": summary["datasets"],
        "bio_characteristics_queried": BIO_CHARACTERISTICS,
        "nwis_bio_params": NWIS_BIO_PARAMS,
    }
    with open(output_dir / "download_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    # Print summary
    log.info("=" * 70)
    log.info("DOWNLOAD COMPLETE")
    log.info(f"  Total files: {summary['total_files']}")
    log.info(f"  Total rows:  {summary['total_rows']:,}")
    log.info(f"  Output dir:  {output_dir}")
    for ds_name, ds_info in summary["datasets"].items():
        log.info(
            f"    {ds_name}: {ds_info['rows']:,} rows, "
            f"{ds_info['size_mb']:.1f} MB"
        )
    for key, val in results.items():
        log.info(f"  {key}: {val:,} records")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
