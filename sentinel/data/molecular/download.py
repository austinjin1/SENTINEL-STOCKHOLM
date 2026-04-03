"""
Molecular/toxicogenomics data download for SENTINEL.

Data sources:
  1. GEO (Gene Expression Omnibus) via GEOparse — toxicogenomics studies
  2. CTD (Comparative Toxicogenomics Database) — chemical-gene interactions
  3. ArrayExpress (EBI) — supplementary gene expression studies
"""

from __future__ import annotations

import gzip
import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import requests

from sentinel.utils.logging import get_logger, make_progress

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Target species for aquatic toxicogenomics
# ---------------------------------------------------------------------------

TARGET_SPECIES = [
    "Danio rerio",           # Zebrafish
    "Pimephales promelas",   # Fathead minnow
    "Daphnia magna",         # Water flea
    "Oncorhynchus mykiss",   # Rainbow trout
]

# GEO search terms for chemical exposure studies
GEO_SEARCH_TERMS = [
    '"{species}" AND "chemical exposure"',
    '"{species}" AND "toxicogenomics"',
    '"{species}" AND "contaminant"',
    '"{species}" AND ("pesticide" OR "heavy metal" OR "pharmaceutical")',
]


# ---------------------------------------------------------------------------
# GEO download via GEOparse
# ---------------------------------------------------------------------------


@dataclass
class GEOStudy:
    """Metadata for a GEO dataset."""

    accession: str  # GSE accession
    title: str
    organism: str
    n_samples: int = 0
    platform: str = ""
    study_type: str = ""  # microarray or RNA-seq
    chemicals: list[str] = field(default_factory=list)


def search_geo(
    species: Sequence[str] | None = None,
    *,
    max_results_per_query: int = 50,
    api_key: str | None = None,
    delay: float = 0.5,
) -> list[GEOStudy]:
    """Search GEO for toxicogenomics studies on target aquatic species.

    Parameters
    ----------
    species:
        Species names to search (default: TARGET_SPECIES).
    max_results_per_query:
        Maximum results per search query.
    api_key:
        NCBI API key.
    delay:
        Seconds between API requests.

    Returns
    -------
    Deduplicated list of GEO studies.
    """
    species = species or TARGET_SPECIES
    entrez_base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    base_params: dict[str, str] = {"db": "gds", "retmode": "json"}
    if api_key:
        base_params["api_key"] = api_key

    seen: set[str] = set()
    studies: list[GEOStudy] = []

    for sp in species:
        for template in GEO_SEARCH_TERMS:
            query = template.format(species=sp)
            try:
                # Search
                resp = requests.get(
                    f"{entrez_base}/esearch.fcgi",
                    params={**base_params, "term": query, "retmax": str(max_results_per_query)},
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()
                id_list = data.get("esearchresult", {}).get("idlist", [])

                if not id_list:
                    continue

                time.sleep(delay)

                # Fetch summaries
                resp = requests.get(
                    f"{entrez_base}/esummary.fcgi",
                    params={**base_params, "id": ",".join(id_list)},
                    timeout=30,
                )
                resp.raise_for_status()
                results = resp.json().get("result", {})

                for uid in id_list:
                    entry = results.get(uid, {})
                    accession = entry.get("accession", "")
                    if not accession or not accession.startswith("GSE"):
                        continue
                    if accession in seen:
                        continue
                    seen.add(accession)

                    studies.append(
                        GEOStudy(
                            accession=accession,
                            title=entry.get("title", ""),
                            organism=sp,
                            n_samples=int(entry.get("n_samples", 0)),
                            platform=entry.get("gpl", ""),
                            study_type=(
                                "RNA-seq"
                                if "high throughput sequencing" in entry.get("gdstype", "").lower()
                                else "microarray"
                            ),
                        )
                    )

                time.sleep(delay)

            except Exception as exc:
                logger.warning(f"GEO search failed for '{query}': {exc}")

    logger.info(f"Found {len(studies)} GEO studies across {len(species)} species")
    return studies


def download_geo_study(
    accession: str,
    output_dir: str | Path,
    *,
    soft_only: bool = False,
) -> Path:
    """Download a GEO dataset using GEOparse.

    Parameters
    ----------
    accession:
        GEO series accession (e.g., ``GSE12345``).
    output_dir:
        Directory for downloaded files.
    soft_only:
        If True, download only the SOFT file (metadata + expression matrix).
        If False, also attempt to download supplementary files.

    Returns
    -------
    Path to the downloaded SOFT file directory.
    """
    import GEOparse

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    study_dir = output_dir / accession

    logger.info(f"Downloading GEO dataset {accession}")
    try:
        gse = GEOparse.get_GEO(
            geo=accession,
            destdir=str(study_dir),
            silent=True,
        )
        logger.info(
            f"Downloaded {accession}: "
            f"{len(gse.gsms)} samples, {len(gse.gpls)} platforms"
        )

        # Save metadata summary
        meta = {
            "accession": accession,
            "title": gse.metadata.get("title", [""])[0],
            "summary": gse.metadata.get("summary", [""])[0],
            "organism": gse.metadata.get("organism_ch1", [""])[0]
            if hasattr(gse, "metadata")
            else "",
            "n_samples": len(gse.gsms),
            "platforms": list(gse.gpls.keys()),
            "sample_ids": list(gse.gsms.keys()),
        }
        with open(study_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        if not soft_only:
            # Download supplementary files (count matrices, etc.)
            for gsm_name, gsm in gse.gsms.items():
                for url in gsm.metadata.get("supplementary_file", []):
                    if url.startswith("ftp://") or url.startswith("http"):
                        _download_file(url, study_dir / "supplementary", timeout=120)

    except Exception as exc:
        logger.error(f"Failed to download {accession}: {exc}")
        raise

    return study_dir


def download_geo_batch(
    studies: Sequence[GEOStudy],
    output_dir: str | Path = "data/molecular/geo",
    *,
    max_studies: int | None = None,
    delay: float = 2.0,
) -> list[Path]:
    """Download multiple GEO datasets.

    Returns list of output directories.
    """
    output_dir = Path(output_dir)
    targets = studies[:max_studies] if max_studies else studies
    paths: list[Path] = []

    progress = make_progress()
    with progress:
        task = progress.add_task("Downloading GEO studies", total=len(targets))
        for study in targets:
            try:
                p = download_geo_study(study.accession, output_dir)
                paths.append(p)
            except Exception as exc:
                logger.warning(f"Skipping {study.accession}: {exc}")
            progress.advance(task)
            time.sleep(delay)

    logger.info(f"Downloaded {len(paths)}/{len(targets)} GEO studies")
    return paths


# ---------------------------------------------------------------------------
# CTD (Comparative Toxicogenomics Database) bulk download
# ---------------------------------------------------------------------------

CTD_DOWNLOAD_URLS = {
    "chem_gene_ixns": "https://ctdbase.org/reports/CTD_chem_gene_ixns.tsv.gz",
    "chem_diseases": "https://ctdbase.org/reports/CTD_chemicals_diseases.tsv.gz",
    "chem_pathways": "https://ctdbase.org/reports/CTD_chem_pathways_enriched.tsv.gz",
    "gene_diseases": "https://ctdbase.org/reports/CTD_genes_diseases.tsv.gz",
}


def download_ctd(
    output_dir: str | Path = "data/molecular/ctd",
    *,
    datasets: Sequence[str] | None = None,
    timeout: int = 600,
    max_retries: int = 3,
) -> dict[str, Path]:
    """Download CTD bulk data files.

    Parameters
    ----------
    output_dir:
        Output directory.
    datasets:
        Subset of CTD datasets to download (default: all).
    timeout:
        HTTP timeout in seconds.
    max_retries:
        Retry attempts per file.

    Returns
    -------
    Mapping of dataset name -> downloaded file path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = datasets or list(CTD_DOWNLOAD_URLS.keys())
    results: dict[str, Path] = {}

    progress = make_progress()
    with progress:
        task = progress.add_task("Downloading CTD data", total=len(targets))
        for name in targets:
            url = CTD_DOWNLOAD_URLS.get(name)
            if not url:
                logger.warning(f"Unknown CTD dataset: {name}")
                progress.advance(task)
                continue

            gz_path = output_dir / f"{name}.tsv.gz"
            tsv_path = output_dir / f"{name}.tsv"

            for attempt in range(1, max_retries + 1):
                try:
                    resp = requests.get(url, timeout=timeout, stream=True)
                    resp.raise_for_status()
                    with open(gz_path, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=8192):
                            f.write(chunk)

                    # Decompress
                    with gzip.open(gz_path, "rb") as f_in:
                        with open(tsv_path, "wb") as f_out:
                            shutil.copyfileobj(f_in, f_out)

                    results[name] = tsv_path
                    logger.info(
                        f"Downloaded CTD {name}: "
                        f"{tsv_path.stat().st_size / 1e6:.1f} MB"
                    )
                    break
                except Exception as exc:
                    logger.warning(
                        f"CTD download attempt {attempt}/{max_retries} "
                        f"for {name}: {exc}"
                    )
                    if attempt < max_retries:
                        time.sleep(5 * attempt)

            progress.advance(task)

    return results


# ---------------------------------------------------------------------------
# ArrayExpress supplementary download
# ---------------------------------------------------------------------------


def search_arrayexpress(
    species: Sequence[str] | None = None,
    *,
    max_results: int = 50,
) -> list[dict[str, Any]]:
    """Search ArrayExpress (EBI) for toxicogenomics experiments.

    Parameters
    ----------
    species:
        Species to search.
    max_results:
        Maximum results per species.

    Returns
    -------
    List of experiment metadata dicts.
    """
    species = species or TARGET_SPECIES
    api_base = "https://www.ebi.ac.uk/biostudies/api/v1/search"
    results: list[dict[str, Any]] = []
    seen: set[str] = set()

    for sp in species:
        query = f'"{sp}" AND (toxicogenomics OR "chemical exposure")'
        try:
            resp = requests.get(
                api_base,
                params={
                    "query": query,
                    "type": "study",
                    "pageSize": str(max_results),
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

            for hit in data.get("hits", []):
                accession = hit.get("accession", "")
                if accession in seen:
                    continue
                seen.add(accession)
                results.append(
                    {
                        "accession": accession,
                        "title": hit.get("title", ""),
                        "organism": sp,
                        "type": hit.get("type", ""),
                    }
                )
        except Exception as exc:
            logger.warning(f"ArrayExpress search failed for '{sp}': {exc}")

    logger.info(f"Found {len(results)} ArrayExpress experiments")
    return results


def download_arrayexpress_study(
    accession: str,
    output_dir: str | Path = "data/molecular/arrayexpress",
    *,
    timeout: int = 300,
) -> Path:
    """Download files for a single ArrayExpress/BioStudies accession.

    Returns the study output directory.
    """
    output_dir = Path(output_dir) / accession
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fetch file list from BioStudies API
    files_url = f"https://www.ebi.ac.uk/biostudies/api/v1/studies/{accession}/info"
    try:
        resp = requests.get(files_url, timeout=30)
        resp.raise_for_status()
        info = resp.json()

        for section in info.get("section", {}).get("files", []):
            for file_entry in section if isinstance(section, list) else [section]:
                file_path = file_entry.get("path", "")
                if not file_path:
                    continue
                dl_url = (
                    f"https://www.ebi.ac.uk/biostudies/files/{accession}/{file_path}"
                )
                _download_file(dl_url, output_dir, timeout=timeout)

    except Exception as exc:
        logger.warning(f"ArrayExpress download failed for {accession}: {exc}")

    return output_dir


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------


def download_all_molecular(
    output_base: str | Path = "data/molecular",
    *,
    skip_geo: bool = False,
    skip_ctd: bool = False,
    skip_arrayexpress: bool = False,
    geo_max_studies: int = 20,
    ncbi_api_key: str | None = None,
) -> dict[str, Any]:
    """Download all molecular/toxicogenomics data sources.

    Returns a summary dict.
    """
    output_base = Path(output_base)
    summary: dict[str, Any] = {}

    if not skip_geo:
        logger.info("--- Searching GEO for toxicogenomics studies ---")
        studies = search_geo(api_key=ncbi_api_key)
        paths = download_geo_batch(
            studies, output_base / "geo", max_studies=geo_max_studies
        )
        summary["geo"] = {
            "studies_found": len(studies),
            "downloaded": len(paths),
        }

        # Save study catalog
        catalog = [
            {
                "accession": s.accession,
                "title": s.title,
                "organism": s.organism,
                "n_samples": s.n_samples,
                "platform": s.platform,
                "study_type": s.study_type,
            }
            for s in studies
        ]
        catalog_path = output_base / "geo" / "study_catalog.json"
        catalog_path.parent.mkdir(parents=True, exist_ok=True)
        with open(catalog_path, "w", encoding="utf-8") as f:
            json.dump(catalog, f, indent=2)

    if not skip_ctd:
        logger.info("--- Downloading CTD bulk data ---")
        ctd_paths = download_ctd(output_base / "ctd")
        summary["ctd"] = {k: str(v) for k, v in ctd_paths.items()}

    if not skip_arrayexpress:
        logger.info("--- Searching ArrayExpress ---")
        ae_results = search_arrayexpress()
        summary["arrayexpress"] = {"experiments_found": len(ae_results)}

        # Save catalog (actual download of large files is deferred)
        ae_catalog_path = output_base / "arrayexpress" / "experiment_catalog.json"
        ae_catalog_path.parent.mkdir(parents=True, exist_ok=True)
        with open(ae_catalog_path, "w", encoding="utf-8") as f:
            json.dump(ae_results, f, indent=2)

    # Save summary
    summary_path = output_base / "download_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _download_file(
    url: str,
    output_dir: Path,
    *,
    timeout: int = 120,
    max_retries: int = 2,
) -> Path | None:
    """Download a single file with retries."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = url.split("/")[-1].split("?")[0]
    out_path = output_dir / filename

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, timeout=timeout, stream=True)
            resp.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            return out_path
        except Exception as exc:
            if attempt == max_retries:
                logger.debug(f"Failed to download {url}: {exc}")
                return None
            time.sleep(2 * attempt)
    return None
