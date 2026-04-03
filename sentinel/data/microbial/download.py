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
