"""
Microbial community data download for SENTINEL.

Data sources:
  1. EPA National Aquatic Resource Surveys (NARS) — microbial indicators
  2. Earth Microbiome Project (EMP) via Qiita/Redbiom — aquatic subset
  3. NCBI SRA — targeted freshwater microbiome studies
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urljoin

import requests

from sentinel.utils.logging import get_logger, make_progress

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# EPA NARS download
# ---------------------------------------------------------------------------

NARS_BASE_URL = "https://www.epa.gov/national-aquatic-resource-surveys/data-national-aquatic-resource-surveys"

# Direct download URLs for NARS datasets (updated periodically by EPA)
NARS_DATASETS: dict[str, str] = {
    "nla_2017_microbial": (
        "https://www.epa.gov/system/files/other-files/2024-03/"
        "nla22_microbial_indicator_data_csv.zip"
    ),
    "nrsa_2018_microbial": (
        "https://www.epa.gov/system/files/other-files/2024-03/"
        "nrsa1819_microbial_indicator_data_csv.zip"
    ),
    "ncca_2020_microbial": (
        "https://www.epa.gov/system/files/other-files/2024-03/"
        "ncca20_microbial_indicator_data_csv.zip"
    ),
}


@dataclass
class NARSDownloadResult:
    """Result of a NARS dataset download."""

    dataset_name: str
    output_path: Path
    n_bytes: int
    success: bool
    error: str = ""


def download_nars(
    output_dir: str | Path = "data/microbial/nars",
    *,
    datasets: Sequence[str] | None = None,
    timeout: int = 300,
    max_retries: int = 3,
) -> list[NARSDownloadResult]:
    """Download EPA NARS microbial indicator datasets.

    Parameters
    ----------
    output_dir:
        Directory for downloaded ZIP archives.
    datasets:
        Subset of dataset keys to download (default: all).
    timeout:
        HTTP request timeout in seconds.
    max_retries:
        Number of retry attempts per dataset.

    Returns
    -------
    List of download results.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    targets = datasets or list(NARS_DATASETS.keys())
    results: list[NARSDownloadResult] = []

    progress = make_progress()
    with progress:
        task = progress.add_task("Downloading NARS data", total=len(targets))
        for name in targets:
            url = NARS_DATASETS.get(name)
            if url is None:
                logger.warning(f"Unknown NARS dataset: {name}")
                progress.advance(task)
                continue

            out_path = output_dir / f"{name}.zip"
            result = NARSDownloadResult(
                dataset_name=name, output_path=out_path, n_bytes=0, success=False
            )

            for attempt in range(1, max_retries + 1):
                try:
                    resp = requests.get(url, timeout=timeout, stream=True)
                    resp.raise_for_status()
                    with open(out_path, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=8192):
                            f.write(chunk)
                    result.n_bytes = out_path.stat().st_size
                    result.success = True
                    logger.info(
                        f"Downloaded {name}: {result.n_bytes / 1e6:.1f} MB"
                    )
                    break
                except Exception as exc:
                    result.error = str(exc)
                    logger.warning(f"Attempt {attempt}/{max_retries} for {name}: {exc}")
                    if attempt < max_retries:
                        time.sleep(5 * attempt)

            results.append(result)
            progress.advance(task)

    return results


# ---------------------------------------------------------------------------
# Earth Microbiome Project (Qiita / Redbiom)
# ---------------------------------------------------------------------------

EMP_STUDY_ID = 10317  # Earth Microbiome Project on Qiita


def download_emp_aquatic(
    output_dir: str | Path = "data/microbial/emp",
    *,
    study_id: int = EMP_STUDY_ID,
    environment_filter: str = "aquatic",
    redbiom_context: str = "Deblur-Illumina-16S-V4-150nt-780653",
) -> Path:
    """Download the aquatic subset of the Earth Microbiome Project via Redbiom.

    Requires ``redbiom`` to be installed (``pip install redbiom``).

    Parameters
    ----------
    output_dir:
        Directory for output BIOM file.
    study_id:
        Qiita study ID for EMP.
    environment_filter:
        Metadata keyword to filter aquatic samples.
    redbiom_context:
        Redbiom processing context.

    Returns
    -------
    Path to the output BIOM file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    biom_path = output_dir / f"emp_{study_id}_aquatic.biom"
    metadata_path = output_dir / f"emp_{study_id}_aquatic_metadata.txt"

    # Step 1: Fetch sample IDs matching aquatic environments
    logger.info(f"Querying Redbiom for aquatic samples in study {study_id}")
    try:
        import redbiom.fetch
        import redbiom.search
        import redbiom.util

        # Search for samples with aquatic environment metadata
        sample_ids = redbiom.search.metadata_full(
            f"where qiita_study_id == '{study_id}' and "
            f"env_material like '%{environment_filter}%'"
        )
        if not sample_ids:
            # Broader search
            sample_ids = redbiom.search.metadata_full(
                f"where qiita_study_id == '{study_id}' and "
                f"env_biome like '%{environment_filter}%'"
            )

        logger.info(f"Found {len(sample_ids)} aquatic samples")

        if not sample_ids:
            logger.warning("No aquatic samples found via Redbiom")
            return biom_path

        # Step 2: Fetch the BIOM table for these samples
        table, ambiguous = redbiom.fetch.data_from_samples(
            redbiom_context, sample_ids
        )
        if table is not None:
            with open(biom_path, "w") as f:
                table.to_json("SENTINEL-download", f)
            logger.info(
                f"Saved BIOM table: {table.shape[0]} features x "
                f"{table.shape[1]} samples -> {biom_path}"
            )

        # Step 3: Fetch metadata
        metadata = redbiom.fetch.sample_metadata(sample_ids)
        if metadata is not None:
            metadata.to_csv(metadata_path, sep="\t")
            logger.info(f"Saved metadata -> {metadata_path}")

    except ImportError:
        logger.warning(
            "redbiom not installed. Falling back to Qiita REST API download."
        )
        _download_emp_via_qiita(study_id, output_dir)
    except Exception as exc:
        logger.error(f"Redbiom download failed: {exc}")
        _download_emp_via_qiita(study_id, output_dir)

    return biom_path


def _download_emp_via_qiita(study_id: int, output_dir: Path) -> None:
    """Fallback: download EMP data via Qiita REST API."""
    api_url = f"https://qiita.ucsd.edu/api/v1/study/{study_id}/data"
    try:
        resp = requests.get(api_url, timeout=60)
        resp.raise_for_status()
        artifacts = resp.json()

        for artifact in artifacts:
            artifact_id = artifact.get("id")
            download_url = (
                f"https://qiita.ucsd.edu/public_artifact_download/"
                f"?artifact_id={artifact_id}"
            )
            out_path = output_dir / f"qiita_{study_id}_{artifact_id}.zip"
            logger.info(f"Downloading Qiita artifact {artifact_id}")
            r = requests.get(download_url, timeout=300, stream=True)
            r.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info(f"Saved {out_path.name}")
    except Exception as exc:
        logger.error(f"Qiita download failed: {exc}")


# ---------------------------------------------------------------------------
# NCBI SRA targeted studies
# ---------------------------------------------------------------------------

SRA_SEARCH_KEYWORDS = [
    "freshwater microbiome",
    "water quality microbiome",
    "river microbiome",
    "lake microbiome 16S",
    "drinking water microbiome",
]

ENTREZ_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


@dataclass
class SRAStudy:
    """Metadata for an NCBI SRA study."""

    accession: str
    title: str
    organism: str = ""
    n_runs: int = 0
    bioproject: str = ""


def search_sra(
    keywords: Sequence[str] | None = None,
    *,
    max_results_per_keyword: int = 50,
    api_key: str | None = None,
    delay: float = 0.4,
) -> list[SRAStudy]:
    """Search NCBI SRA for freshwater microbiome studies.

    Parameters
    ----------
    keywords:
        Search terms (default: predefined freshwater microbiome terms).
    max_results_per_keyword:
        Cap on results per search term.
    api_key:
        NCBI API key for higher rate limits.
    delay:
        Seconds between API calls.

    Returns
    -------
    Deduplicated list of SRA studies.
    """
    keywords = keywords or SRA_SEARCH_KEYWORDS
    seen_accessions: set[str] = set()
    studies: list[SRAStudy] = []
    params_base: dict[str, str] = {"db": "sra", "retmode": "json"}
    if api_key:
        params_base["api_key"] = api_key

    for kw in keywords:
        try:
            # Search
            search_params = {
                **params_base,
                "term": f'"{kw}"[All Fields] AND "AMPLICON"[Strategy]',
                "retmax": str(max_results_per_keyword),
            }
            resp = requests.get(f"{ENTREZ_BASE}/esearch.fcgi", params=search_params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            id_list = data.get("esearchresult", {}).get("idlist", [])

            if not id_list:
                continue

            time.sleep(delay)

            # Fetch summaries
            summary_params = {
                **params_base,
                "id": ",".join(id_list),
            }
            resp = requests.get(
                f"{ENTREZ_BASE}/esummary.fcgi", params=summary_params, timeout=30
            )
            resp.raise_for_status()
            summary_data = resp.json()

            results = summary_data.get("result", {})
            for uid in id_list:
                entry = results.get(uid, {})
                accession = entry.get("expxml", {}).get("Study", {}).get("acc", "")
                if not accession:
                    # Try alternative field
                    accession = entry.get("accession", uid)
                if accession in seen_accessions:
                    continue
                seen_accessions.add(accession)

                studies.append(
                    SRAStudy(
                        accession=accession,
                        title=entry.get("title", ""),
                        organism=entry.get("organism", ""),
                        n_runs=int(entry.get("runs", {}).get("Run", [{}]).__len__())
                        if isinstance(entry.get("runs"), dict)
                        else 0,
                        bioproject=entry.get("bioproject", ""),
                    )
                )

            time.sleep(delay)

        except Exception as exc:
            logger.warning(f"SRA search failed for '{kw}': {exc}")

    logger.info(f"Found {len(studies)} unique SRA studies")
    return studies


def download_sra_metadata(
    studies: Sequence[SRAStudy],
    output_dir: str | Path = "data/microbial/sra",
) -> Path:
    """Save SRA study metadata to JSON for later download with sra-tools.

    Actual FASTQ download is deferred to ``fasterq-dump`` because the files
    are large. This function creates a manifest for batch download.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "sra_manifest.json"

    records = [
        {
            "accession": s.accession,
            "title": s.title,
            "organism": s.organism,
            "n_runs": s.n_runs,
            "bioproject": s.bioproject,
        }
        for s in studies
    ]

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    logger.info(f"SRA manifest saved: {len(records)} studies -> {manifest_path}")

    # Also write a shell script for fasterq-dump
    script_path = output_dir / "download_fastq.sh"
    with open(script_path, "w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\n")
        f.write("# Auto-generated by SENTINEL microbial download\n")
        f.write(f"# {len(records)} studies\n\n")
        f.write("set -euo pipefail\n\n")
        for rec in records:
            f.write(f'echo "Downloading {rec["accession"]}"\n')
            f.write(
                f'fasterq-dump --split-files --outdir "{output_dir}" '
                f'{rec["accession"]} || true\n'
            )
    logger.info(f"Download script written -> {script_path}")

    return manifest_path


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------


def download_all_microbial(
    output_base: str | Path = "data/microbial",
    *,
    skip_nars: bool = False,
    skip_emp: bool = False,
    skip_sra: bool = False,
    sra_api_key: str | None = None,
) -> dict[str, Any]:
    """Download all microbial data sources.

    Returns a summary dict with paths and counts.
    """
    output_base = Path(output_base)
    summary: dict[str, Any] = {}

    if not skip_nars:
        logger.info("--- Downloading EPA NARS data ---")
        nars_results = download_nars(output_base / "nars")
        summary["nars"] = {
            "datasets": len(nars_results),
            "successful": sum(1 for r in nars_results if r.success),
        }

    if not skip_emp:
        logger.info("--- Downloading Earth Microbiome Project aquatic subset ---")
        biom_path = download_emp_aquatic(output_base / "emp")
        summary["emp"] = {"biom_path": str(biom_path)}

    if not skip_sra:
        logger.info("--- Searching NCBI SRA for freshwater microbiome studies ---")
        studies = search_sra(api_key=sra_api_key)
        manifest = download_sra_metadata(studies, output_base / "sra")
        summary["sra"] = {"studies": len(studies), "manifest": str(manifest)}

    # Save summary
    summary_path = output_base / "download_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Download summary -> {summary_path}")

    return summary


# ---------------------------------------------------------------------------
# MGnify aquatic metagenome download
# ---------------------------------------------------------------------------

MGNIFY_API_BASE = "https://www.ebi.ac.uk/metagenomics/api/v1"


def search_mgnify_aquatic(
    biome: str = "root:Environmental:Aquatic",
    max_studies: int = 100,
) -> list[dict]:
    """Search MGnify for aquatic metagenome studies.

    Parameters
    ----------
    biome:
        MGnify biome lineage filter (default: aquatic environments).
    max_studies:
        Maximum number of studies to return.

    Returns
    -------
    List of study dicts with keys: accession, bioproject, name, description,
    biomes, n_samples.
    """
    studies: list[dict] = []
    url = f"{MGNIFY_API_BASE}/studies"
    params: dict[str, Any] = {
        "lineage": biome,
        "ordering": "-samples_count",
        "page_size": min(max_studies, 100),
    }

    logger.info(f"Searching MGnify for biome: {biome}")

    while url and len(studies) < max_studies:
        try:
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:
            logger.warning(f"MGnify API request failed: {exc}")
            break

        for item in payload.get("data", []):
            if len(studies) >= max_studies:
                break
            attrs = item.get("attributes", {})
            studies.append({
                "accession": item.get("id", ""),
                "bioproject": attrs.get("bioproject", ""),
                "name": attrs.get("study-name", ""),
                "description": attrs.get("study-abstract", ""),
                "biomes": [
                    b.get("id", "")
                    for b in item.get("relationships", {})
                    .get("biomes", {})
                    .get("data", [])
                ],
                "n_samples": attrs.get("samples-count", 0),
            })

        # Follow pagination link; clear params so they aren't doubled
        next_link = payload.get("links", {}).get("next")
        url = next_link
        params = {}

    logger.info(f"MGnify search returned {len(studies)} aquatic studies")
    return studies


def download_mgnify_study(
    accession: str,
    output_dir: str | Path,
    *,
    timeout: int = 300,
) -> Path:
    """Download a MGnify study's OTU/ASV tables and metadata.

    Parameters
    ----------
    accession:
        MGnify study accession (e.g., ``MGYS00001234``).
    output_dir:
        Directory for downloaded files.
    timeout:
        HTTP request timeout in seconds.

    Returns
    -------
    Path to the study output directory containing downloaded files.
    """
    output_dir = Path(output_dir) / accession
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download study metadata
    study_url = f"{MGNIFY_API_BASE}/studies/{accession}"
    try:
        resp = requests.get(study_url, timeout=timeout)
        resp.raise_for_status()
        study_meta = resp.json()
        with open(output_dir / "study_metadata.json", "w", encoding="utf-8") as f:
            json.dump(study_meta, f, indent=2)
        logger.info(f"Saved study metadata for {accession}")
    except Exception as exc:
        logger.error(f"Failed to download metadata for {accession}: {exc}")

    # Download analyses (OTU/ASV taxonomic abundance tables)
    analyses_url = f"{MGNIFY_API_BASE}/studies/{accession}/analyses"
    analysis_params: dict[str, Any] = {"page_size": 100}
    analysis_ids: list[str] = []

    try:
        while analyses_url:
            resp = requests.get(analyses_url, params=analysis_params, timeout=timeout)
            resp.raise_for_status()
            payload = resp.json()
            for item in payload.get("data", []):
                analysis_ids.append(item.get("id", ""))
            analyses_url = payload.get("links", {}).get("next")
            analysis_params = {}
    except Exception as exc:
        logger.warning(f"Failed to list analyses for {accession}: {exc}")

    # For each analysis, attempt to download the taxonomic abundance TSV
    for analysis_id in analysis_ids:
        dl_url = (
            f"{MGNIFY_API_BASE}/analyses/{analysis_id}/file/"
            f"OTU_abundances_v2.tsv"
        )
        try:
            resp = requests.get(dl_url, timeout=timeout, stream=True)
            if resp.status_code == 200:
                out_file = output_dir / f"{analysis_id}_otu_abundances.tsv"
                with open(out_file, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                logger.info(f"Downloaded OTU table for analysis {analysis_id}")
            else:
                # Try BIOM format
                biom_url = (
                    f"{MGNIFY_API_BASE}/analyses/{analysis_id}/file/"
                    f"OTU_abundances.biom"
                )
                resp2 = requests.get(biom_url, timeout=timeout, stream=True)
                if resp2.status_code == 200:
                    out_file = output_dir / f"{analysis_id}_otu_abundances.biom"
                    with open(out_file, "wb") as f:
                        for chunk in resp2.iter_content(chunk_size=8192):
                            f.write(chunk)
                    logger.info(
                        f"Downloaded OTU BIOM for analysis {analysis_id}"
                    )
        except Exception as exc:
            logger.warning(
                f"Failed to download OTU table for {analysis_id}: {exc}"
            )

    logger.info(
        f"Study {accession}: downloaded metadata + "
        f"{len(analysis_ids)} analysis files -> {output_dir}"
    )
    return output_dir


def download_mgnify_batch(
    studies: list[dict],
    output_dir: str | Path,
    max_studies: int = 50,
) -> list[Path]:
    """Batch download MGnify studies with progress bar.

    Parameters
    ----------
    studies:
        List of study dicts as returned by :func:`search_mgnify_aquatic`.
    output_dir:
        Root output directory.
    max_studies:
        Maximum number of studies to download.

    Returns
    -------
    List of paths to downloaded study directories.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    targets = studies[:max_studies]
    results: list[Path] = []

    progress = make_progress()
    with progress:
        task = progress.add_task("Downloading MGnify studies", total=len(targets))
        for study in targets:
            accession = study.get("accession", "")
            if not accession:
                progress.advance(task)
                continue
            try:
                path = download_mgnify_study(accession, output_dir)
                results.append(path)
            except Exception as exc:
                logger.warning(f"Failed to download {accession}: {exc}")
            progress.advance(task)

    logger.info(
        f"MGnify batch download: {len(results)}/{len(targets)} studies succeeded"
    )
    return results


# ---------------------------------------------------------------------------
# FreshWater Watch citizen science data
# ---------------------------------------------------------------------------


def download_freshwater_watch(output_dir: str | Path) -> Path:
    """Download FreshWater Watch citizen science data.

    FreshWater Watch data requires registration at
    https://freshwaterwatch.thewaterhub.org/

    This function processes pre-downloaded CSV exports. Users must:
      1. Register at https://freshwaterwatch.thewaterhub.org/
      2. Navigate to the data download portal
      3. Export data as CSV and place files in ``output_dir``

    The expected CSV columns include: latitude, longitude, date, nitrate,
    phosphate, turbidity, and other water quality indicators.

    Parameters
    ----------
    output_dir:
        Directory containing pre-downloaded FreshWater Watch CSV exports.
        Processed outputs will also be written here.

    Returns
    -------
    Path to the processed/consolidated output file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = list(output_dir.glob("*.csv"))
    if not csv_files:
        logger.warning(
            "No FreshWater Watch CSV files found in %s. "
            "Please download data from https://freshwaterwatch.thewaterhub.org/ "
            "and place CSV exports in this directory.",
            output_dir,
        )
        # Write instructions file
        instructions_path = output_dir / "DOWNLOAD_INSTRUCTIONS.txt"
        instructions_path.write_text(
            "FreshWater Watch Data Download Instructions\n"
            "============================================\n\n"
            "1. Register at https://freshwaterwatch.thewaterhub.org/\n"
            "2. Log in and navigate to the data download section\n"
            "3. Select your region/time range of interest\n"
            "4. Export data as CSV\n"
            "5. Place the CSV files in this directory\n"
            "6. Re-run this function to process the data\n",
            encoding="utf-8",
        )
        return instructions_path

    # Consolidate all CSV exports into a single DataFrame
    import pandas as pd

    frames: list[pd.DataFrame] = []
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path, encoding="utf-8")
            frames.append(df)
            logger.info(f"Loaded {csv_path.name}: {len(df)} records")
        except Exception as exc:
            logger.warning(f"Failed to read {csv_path.name}: {exc}")

    if not frames:
        logger.error("No valid CSV files could be loaded")
        return output_dir

    combined = pd.concat(frames, ignore_index=True)

    # Standardize column names (lowercase, underscores)
    combined.columns = [
        c.strip().lower().replace(" ", "_") for c in combined.columns
    ]

    output_path = output_dir / "freshwater_watch_consolidated.parquet"
    combined.to_parquet(output_path, index=False)
    logger.info(
        f"FreshWater Watch: consolidated {len(combined)} records "
        f"from {len(csv_files)} files -> {output_path}"
    )

    return output_path
