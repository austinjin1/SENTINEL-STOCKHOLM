"""
EPA ECOTOX database download for SENTINEL.

Downloads the EPA ECOTOX knowledgebase bulk data and filters for
freshwater aquatic organisms.
"""

from __future__ import annotations

import json
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import requests

from sentinel.utils.logging import get_logger, make_progress

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# ECOTOX download URLs and key fields
# ---------------------------------------------------------------------------

ECOTOX_BULK_URL = "https://cfpub.epa.gov/ecotox/files/ecotox_ascii_03_15_2024.zip"
# Note: The URL is updated quarterly by EPA. The date suffix changes.
# Users should check https://cfpub.epa.gov/ecotox/ for the latest version.

ECOTOX_KEY_FILES = [
    "tests.txt",
    "results.txt",
    "species.txt",
    "chemicals.txt",
    "doses.txt",
    "references.txt",
]

# Fields of interest
KEY_FIELDS_TESTS = [
    "test_id", "test_cas", "species_number", "organism_habitat",
    "organism_lifestage", "exposure_type", "exposure_duration_value",
    "exposure_duration_unit", "media_type", "test_location",
    "endpoint", "effect", "observed_duration_value", "observed_duration_unit",
    "conc1_mean", "conc1_unit", "conc1_type",
    "study_number", "reference_number",
]

# Freshwater-related media types and habitats
FRESHWATER_MEDIA = {"FW", "Fresh water", "Freshwater"}
AQUATIC_HABITATS = {"Water", "Freshwater", "FW"}

# Target endpoint types
TARGET_ENDPOINTS = {"LC50", "EC50", "NOEC", "LOEC", "IC50", "MATC"}


@dataclass
class ECOTOXDownloadResult:
    """Result of an ECOTOX bulk download."""

    output_dir: Path
    zip_path: Path
    extracted_files: list[str]
    n_bytes: int
    success: bool
    error: str = ""


def download_ecotox_bulk(
    output_dir: str | Path = "data/ecotox/raw",
    *,
    url: str = ECOTOX_BULK_URL,
    timeout: int = 1800,
    max_retries: int = 3,
) -> ECOTOXDownloadResult:
    """Download the EPA ECOTOX bulk ASCII data archive.

    Parameters
    ----------
    output_dir:
        Directory for the downloaded archive and extracted files.
    url:
        ECOTOX download URL (updated quarterly by EPA).
    timeout:
        HTTP timeout in seconds.
    max_retries:
        Number of retry attempts.

    Returns
    -------
    Download result with file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / "ecotox_ascii.zip"

    result = ECOTOXDownloadResult(
        output_dir=output_dir,
        zip_path=zip_path,
        extracted_files=[],
        n_bytes=0,
        success=False,
    )

    # Download archive
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(
                f"Downloading ECOTOX bulk data (attempt {attempt}/{max_retries})"
            )
            resp = requests.get(url, timeout=timeout, stream=True)
            resp.raise_for_status()

            total_size = int(resp.headers.get("content-length", 0))
            progress = make_progress()
            with progress:
                task = progress.add_task(
                    "Downloading ECOTOX",
                    total=total_size if total_size else None,
                )
                with open(zip_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=65536):
                        f.write(chunk)
                        progress.advance(task, advance=len(chunk))

            result.n_bytes = zip_path.stat().st_size
            logger.info(f"Downloaded {result.n_bytes / 1e9:.2f} GB")

            # Extract
            logger.info("Extracting ECOTOX archive")
            with zipfile.ZipFile(zip_path, "r") as zf:
                members = zf.namelist()
                for member in members:
                    basename = Path(member).name
                    if basename in ECOTOX_KEY_FILES or member.endswith(".txt"):
                        zf.extract(member, output_dir)
                        result.extracted_files.append(member)

            result.success = True
            logger.info(
                f"Extracted {len(result.extracted_files)} files to {output_dir}"
            )
            break

        except Exception as exc:
            result.error = str(exc)
            logger.warning(f"Download attempt {attempt} failed: {exc}")
            if attempt < max_retries:
                time.sleep(10 * attempt)

    return result


# ---------------------------------------------------------------------------
# Filtering and parsing
# ---------------------------------------------------------------------------


def load_ecotox_table(
    file_path: str | Path,
    *,
    usecols: Sequence[str] | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Load an ECOTOX pipe-delimited text file.

    ECOTOX ASCII files use ``|`` as delimiter and may have encoding issues.
    """
    file_path = Path(file_path)
    try:
        df = pd.read_csv(
            file_path,
            sep="|",
            encoding="latin-1",
            low_memory=False,
            usecols=usecols,
            nrows=nrows,
            on_bad_lines="skip",
        )
    except Exception:
        # Fallback: try utf-8
        df = pd.read_csv(
            file_path,
            sep="|",
            encoding="utf-8",
            low_memory=False,
            usecols=usecols,
            nrows=nrows,
            on_bad_lines="skip",
        )
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    return df


def filter_freshwater_aquatic(
    tests_df: pd.DataFrame,
    species_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Filter ECOTOX test records for freshwater aquatic organisms.

    Parameters
    ----------
    tests_df:
        Tests table from ECOTOX.
    species_df:
        Species table (optional; used for habitat filtering if available).

    Returns
    -------
    Filtered DataFrame.
    """
    original_count = len(tests_df)

    # Filter by media type
    if "media_type" in tests_df.columns:
        media_mask = tests_df["media_type"].astype(str).str.strip().isin(FRESHWATER_MEDIA)
    else:
        media_mask = pd.Series(True, index=tests_df.index)

    # Filter by habitat
    if "organism_habitat" in tests_df.columns:
        habitat_mask = (
            tests_df["organism_habitat"]
            .astype(str)
            .str.strip()
            .str.lower()
            .str.contains("water|aquatic|fresh", na=False)
        )
    else:
        habitat_mask = pd.Series(True, index=tests_df.index)

    filtered = tests_df[media_mask | habitat_mask].copy()

    # If species table available, further filter by aquatic species
    if species_df is not None and "species_number" in filtered.columns:
        if "species_number" in species_df.columns and "habitat" in species_df.columns:
            aquatic_species = species_df[
                species_df["habitat"]
                .astype(str)
                .str.lower()
                .str.contains("water|aquatic|fresh", na=False)
            ]["species_number"].unique()
            filtered = filtered[
                filtered["species_number"].isin(aquatic_species)
            ]

    logger.info(
        f"Freshwater filter: {len(filtered)}/{original_count} records retained"
    )
    return filtered


def filter_target_endpoints(
    df: pd.DataFrame,
    endpoints: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Filter for target toxicity endpoints (LC50, EC50, NOEC, LOEC).

    Parameters
    ----------
    df:
        Filtered ECOTOX tests table.
    endpoints:
        Target endpoint types.

    Returns
    -------
    Filtered DataFrame with only target endpoints.
    """
    endpoints = endpoints or list(TARGET_ENDPOINTS)
    if "endpoint" not in df.columns:
        logger.warning("No 'endpoint' column found")
        return df

    mask = df["endpoint"].astype(str).str.strip().str.upper().isin(
        {e.upper() for e in endpoints}
    )
    filtered = df[mask].copy()
    logger.info(f"Endpoint filter: {len(filtered)}/{len(df)} records retained")
    return filtered


def extract_key_fields(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Extract and standardize key fields for downstream processing.

    Output columns: cas_number, species_id, endpoint, concentration,
    concentration_unit, duration_value, duration_unit, effect.
    """
    result = pd.DataFrame()

    col_map = {
        "test_cas": "cas_number",
        "species_number": "species_id",
        "endpoint": "endpoint",
        "conc1_mean": "concentration",
        "conc1_unit": "concentration_unit",
        "exposure_duration_value": "duration_value",
        "exposure_duration_unit": "duration_unit",
        "effect": "effect",
        "test_id": "test_id",
    }

    for src, dst in col_map.items():
        if src in df.columns:
            result[dst] = df[src]
        else:
            result[dst] = None

    return result


# ---------------------------------------------------------------------------
# Unified download and parse
# ---------------------------------------------------------------------------


def download_and_parse(
    output_dir: str | Path = "data/ecotox",
    *,
    url: str = ECOTOX_BULK_URL,
    filter_freshwater: bool = True,
    filter_endpoints: bool = True,
) -> dict[str, Any]:
    """Download ECOTOX bulk data, filter, and save processed results.

    Returns a summary dict.
    """
    output_dir = Path(output_dir)
    raw_dir = output_dir / "raw"

    # Download
    dl_result = download_ecotox_bulk(raw_dir, url=url)
    if not dl_result.success:
        logger.error(f"ECOTOX download failed: {dl_result.error}")
        return {"success": False, "error": dl_result.error}

    # Find extracted tests file
    tests_file = None
    species_file = None
    for f in dl_result.extracted_files:
        fpath = raw_dir / f
        if "tests" in f.lower() and fpath.exists():
            tests_file = fpath
        if "species" in f.lower() and fpath.exists():
            species_file = fpath

    if tests_file is None:
        # Try finding it in subdirectories
        candidates = list(raw_dir.rglob("tests.txt"))
        if candidates:
            tests_file = candidates[0]
        else:
            return {"success": False, "error": "tests.txt not found in archive"}

    # Load and filter
    logger.info(f"Loading tests from {tests_file}")
    tests = load_ecotox_table(tests_file)
    species_df = None
    if species_file and Path(species_file).exists():
        species_df = load_ecotox_table(species_file)

    if filter_freshwater:
        tests = filter_freshwater_aquatic(tests, species_df)

    if filter_endpoints:
        tests = filter_target_endpoints(tests)

    # Extract key fields
    processed = extract_key_fields(tests)

    # Save
    processed_dir = output_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed.to_parquet(processed_dir / "ecotox_freshwater.parquet")

    summary = {
        "success": True,
        "n_records": len(processed),
        "endpoints": processed["endpoint"].value_counts().to_dict()
        if "endpoint" in processed.columns
        else {},
        "output_path": str(processed_dir / "ecotox_freshwater.parquet"),
    }

    with open(output_dir / "download_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"ECOTOX processing complete: {len(processed)} records")
    return summary
