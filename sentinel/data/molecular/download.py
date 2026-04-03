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
    skip_aop: bool = False,
    skip_reactome: bool = False,
    skip_orthologs: bool = False,
    geo_max_studies: int = 20,
    ncbi_api_key: str | None = None,
    reactome_species: str | None = None,
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

    if not skip_aop:
        logger.info("--- Downloading AOP-Wiki hierarchy ---")
        try:
            aop_path = download_aop_wiki(output_base / "aop_wiki")
            summary["aop_wiki"] = {"hierarchy_path": str(aop_path)}
        except Exception as exc:
            logger.warning(f"AOP-Wiki download failed: {exc}")
            summary["aop_wiki"] = {"error": str(exc)}

    if not skip_reactome:
        logger.info("--- Downloading Reactome hierarchy ---")
        try:
            reactome_paths = download_reactome_hierarchy(
                output_base / "reactome", species=reactome_species
            )
            summary["reactome"] = {k: str(v) for k, v in reactome_paths.items()}
        except Exception as exc:
            logger.warning(f"Reactome download failed: {exc}")
            summary["reactome"] = {"error": str(exc)}

    if not skip_orthologs:
        logger.info("--- Downloading ortholog mappings ---")
        try:
            orth_path = download_ortholog_mappings(output_base / "orthologs")
            summary["orthologs"] = {"mapping_path": str(orth_path)}
        except Exception as exc:
            logger.warning(f"Ortholog download failed: {exc}")
            summary["orthologs"] = {"error": str(exc)}

    # Save summary
    summary_path = output_base / "download_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# AOP-Wiki hierarchy download
# ---------------------------------------------------------------------------

AOP_WIKI_API = "https://aopwiki.org/aops.json"


def download_aop_wiki(output_dir: str | Path) -> Path:
    """Download Adverse Outcome Pathway hierarchy from AOP-Wiki.

    Extracts:
    - Key Events (molecular initiating events, key events, adverse outcomes)
    - Key Event Relationships (causal links between events)
    - AOP-gene associations

    This provides the biological hierarchy for ToxiGene:
    Gene -> Molecular Initiating Event -> Key Event -> Adverse Outcome

    Parameters
    ----------
    output_dir:
        Directory for downloaded files.

    Returns
    -------
    Path to the structured AOP JSON output.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading AOP-Wiki hierarchy")

    # Fetch full AOP list
    try:
        resp = requests.get(AOP_WIKI_API, timeout=60)
        resp.raise_for_status()
        aop_list = resp.json()
    except Exception as exc:
        logger.error(f"Failed to fetch AOP-Wiki AOP list: {exc}")
        raise

    # Save raw response
    raw_path = output_dir / "aops_raw.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(aop_list, f, indent=2)
    logger.info(f"Saved raw AOP-Wiki data: {raw_path}")

    # Parse AOP structures into hierarchy
    aop_hierarchy: dict[str, Any] = {
        "aops": [],
        "key_events": {},
        "key_event_relationships": [],
        "gene_to_mie": {},
        "mie_to_ke": {},
        "ke_to_ao": {},
    }

    all_key_events: dict[str, dict[str, Any]] = {}
    all_kers: list[dict[str, Any]] = []
    gene_to_mie: dict[str, list[str]] = {}
    mie_to_ke: dict[str, list[str]] = {}
    ke_to_ao: dict[str, list[str]] = {}

    progress = make_progress()
    aops = aop_list if isinstance(aop_list, list) else aop_list.get("aops", [])
    with progress:
        task = progress.add_task("Parsing AOP-Wiki entries", total=len(aops))
        for aop_entry in aops:
            aop_id = str(aop_entry.get("id", ""))
            title = aop_entry.get("title", "")

            # Extract key events by type
            mies: list[str] = []
            kes: list[str] = []
            aos: list[str] = []

            for ke in aop_entry.get("key_events", []):
                ke_id = str(ke.get("id", ""))
                ke_title = ke.get("title", "")
                ke_type = ke.get("short_name", ke.get("type", "")).lower()

                all_key_events[ke_id] = {
                    "id": ke_id,
                    "title": ke_title,
                    "type": ke_type,
                    "aop_id": aop_id,
                }

                if "molecular initiating" in ke_type or "mie" in ke_type:
                    mies.append(ke_id)
                elif "adverse outcome" in ke_type or ke_type == "ao":
                    aos.append(ke_id)
                else:
                    kes.append(ke_id)

            # Extract key event relationships
            for ker in aop_entry.get("key_event_relationships", []):
                upstream = str(ker.get("upstream_event_id", ker.get("upstream", "")))
                downstream = str(ker.get("downstream_event_id", ker.get("downstream", "")))
                if upstream and downstream:
                    all_kers.append({
                        "upstream": upstream,
                        "downstream": downstream,
                        "aop_id": aop_id,
                    })

            # Extract gene associations (from stressors or biological context)
            genes: list[str] = []
            for stressor in aop_entry.get("stressors", []):
                stressor_name = stressor.get("name", "")
                if stressor_name:
                    genes.append(stressor_name)

            # Build gene->MIE mappings
            for gene in genes:
                gene_to_mie.setdefault(gene, []).extend(mies)

            # Build MIE->KE and KE->AO mappings from relationships
            for ker in aop_entry.get("key_event_relationships", []):
                up_id = str(ker.get("upstream_event_id", ker.get("upstream", "")))
                down_id = str(ker.get("downstream_event_id", ker.get("downstream", "")))
                if up_id in mies and down_id in (kes + aos):
                    mie_to_ke.setdefault(up_id, []).append(down_id)
                if down_id in aos and up_id in kes:
                    ke_to_ao.setdefault(up_id, []).append(down_id)

            aop_hierarchy["aops"].append({
                "id": aop_id,
                "title": title,
                "mies": mies,
                "key_events": kes,
                "adverse_outcomes": aos,
                "genes": genes,
            })

            progress.advance(task)

    aop_hierarchy["key_events"] = all_key_events
    aop_hierarchy["key_event_relationships"] = all_kers
    aop_hierarchy["gene_to_mie"] = {k: list(set(v)) for k, v in gene_to_mie.items()}
    aop_hierarchy["mie_to_ke"] = {k: list(set(v)) for k, v in mie_to_ke.items()}
    aop_hierarchy["ke_to_ao"] = {k: list(set(v)) for k, v in ke_to_ao.items()}

    # Save structured hierarchy
    hierarchy_path = output_dir / "aop_hierarchy.json"
    with open(hierarchy_path, "w", encoding="utf-8") as f:
        json.dump(aop_hierarchy, f, indent=2)

    logger.info(
        f"AOP-Wiki hierarchy: {len(aop_hierarchy['aops'])} AOPs, "
        f"{len(all_key_events)} key events, "
        f"{len(all_kers)} relationships"
    )
    return hierarchy_path


# ---------------------------------------------------------------------------
# Reactome pathway hierarchy download
# ---------------------------------------------------------------------------

REACTOME_DOWNLOAD = "https://reactome.org/download/current/"


def download_reactome_hierarchy(
    output_dir: str | Path,
    species: str | None = None,
) -> dict[str, Path]:
    """Download Reactome pathway hierarchy and gene-pathway mappings.

    Files:
    - ReactomePathwaysRelation.txt (parent-child pathway relationships)
    - ReactomePathways.gmt (gene sets per pathway)
    - Ensembl2Reactome.txt (gene-pathway associations)

    Parameters
    ----------
    output_dir:
        Directory for downloaded files.
    species:
        Species filter (e.g., "Homo sapiens", "Danio rerio").
        If None, downloads all species.

    Returns
    -------
    Dict mapping file type to downloaded path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files_to_download = {
        "pathway_relations": "ReactomePathwaysRelation.txt",
        "pathway_gmt": "ReactomePathways.gmt.zip",
        "gene_associations": "Ensembl2Reactome.txt",
    }

    results: dict[str, Path] = {}

    progress = make_progress()
    with progress:
        task = progress.add_task("Downloading Reactome data", total=len(files_to_download))
        for key, filename in files_to_download.items():
            url = f"{REACTOME_DOWNLOAD}{filename}"
            out_path = output_dir / filename

            try:
                resp = requests.get(url, timeout=120, stream=True)
                resp.raise_for_status()
                with open(out_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)

                # Unzip if needed
                if filename.endswith(".zip"):
                    import zipfile

                    with zipfile.ZipFile(out_path, "r") as zf:
                        zf.extractall(output_dir)
                    # Point to the extracted file
                    extracted = out_path.with_suffix("")  # remove .zip
                    if extracted.exists():
                        out_path = extracted

                results[key] = out_path
                logger.info(f"Downloaded Reactome {key}: {out_path}")

            except Exception as exc:
                logger.warning(f"Failed to download Reactome {key}: {exc}")

            progress.advance(task)

    # Filter by species if requested
    if species:
        logger.info(f"Filtering Reactome data for species: {species}")

        # Filter gene associations
        if "gene_associations" in results:
            assoc_path = results["gene_associations"]
            filtered_path = output_dir / f"Ensembl2Reactome_{species.replace(' ', '_')}.txt"
            with open(assoc_path, "r", encoding="utf-8") as f_in, \
                 open(filtered_path, "w", encoding="utf-8") as f_out:
                for line in f_in:
                    if species in line:
                        f_out.write(line)
            results["gene_associations_filtered"] = filtered_path
            logger.info(f"Filtered gene associations for {species}")

        # Filter pathway relations (keep pathways that appear in filtered associations)
        if "gene_associations_filtered" in results and "pathway_relations" in results:
            # Collect pathway IDs for this species
            species_pathways: set[str] = set()
            with open(results["gene_associations_filtered"], "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        species_pathways.add(parts[1])

            rel_path = results["pathway_relations"]
            filtered_rel_path = output_dir / f"PathwayRelations_{species.replace(' ', '_')}.txt"
            with open(rel_path, "r", encoding="utf-8") as f_in, \
                 open(filtered_rel_path, "w", encoding="utf-8") as f_out:
                for line in f_in:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        if parts[0] in species_pathways or parts[1] in species_pathways:
                            f_out.write(line)
            results["pathway_relations_filtered"] = filtered_rel_path

    return results


# ---------------------------------------------------------------------------
# Ortholog mapping tables download
# ---------------------------------------------------------------------------

ENSEMBL_BIOMART = "http://www.ensembl.org/biomart/martservice"

# Default species pairs for aquatic toxicogenomics
DEFAULT_ORTHOLOG_PAIRS = [
    ("drerio", "hsapiens"),         # Danio rerio <-> Homo sapiens
    ("drerio", "pmelas"),           # Danio rerio <-> Pimephales promelas
    ("drerio", "omykiss"),          # Danio rerio <-> Oncorhynchus mykiss
]

# BioMart dataset names for species
BIOMART_DATASETS = {
    "hsapiens": "hsapiens_gene_ensembl",
    "drerio": "drerio_gene_ensembl",
    "omykiss": "omykiss_gene_ensembl",
    "pmelas": "ppromelas_gene_ensembl",
}

# NCBI HomoloGene for species lacking Ensembl BioMart coverage
HOMOLOGENE_URL = "https://ftp.ncbi.nih.gov/pub/HomoloGene/current/homologene.data"

# Taxonomy IDs for our target species
SPECIES_TAXIDS = {
    "Homo sapiens": 9606,
    "Danio rerio": 7955,
    "Daphnia magna": 35525,
    "Pimephales promelas": 90988,
    "Oncorhynchus mykiss": 8022,
}


def download_ortholog_mappings(
    output_dir: str | Path,
    species_pairs: list[tuple[str, str]] | None = None,
) -> Path:
    """Download ortholog gene mappings across aquatic model organisms.

    Default species pairs:
    - Danio rerio (zebrafish) <-> Homo sapiens (for gene name standardization)
    - Danio rerio <-> Daphnia magna (limited orthologs, use NCBI HomoloGene)
    - Danio rerio <-> Pimephales promelas (fathead minnow)
    - Danio rerio <-> Oncorhynchus mykiss (rainbow trout)

    Uses Ensembl BioMart REST API or NCBI HomoloGene for mapping.

    Parameters
    ----------
    output_dir:
        Directory for downloaded files.
    species_pairs:
        List of (source, target) BioMart species short names.
        Default: DEFAULT_ORTHOLOG_PAIRS.

    Returns
    -------
    Path to the unified ortholog mapping JSON file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    species_pairs = species_pairs or DEFAULT_ORTHOLOG_PAIRS

    all_orthologs: dict[str, dict[str, list[str]]] = {}
    group_counter = 0

    # --- Ensembl BioMart orthologs ---
    progress = make_progress()
    with progress:
        task = progress.add_task("Downloading ortholog mappings", total=len(species_pairs) + 1)

        for source, target in species_pairs:
            source_dataset = BIOMART_DATASETS.get(source)
            target_short = target

            if not source_dataset:
                logger.warning(f"No BioMart dataset for {source}, skipping")
                progress.advance(task)
                continue

            # Build BioMart XML query
            xml_query = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE Query>
<Query virtualSchemaName="default" formatter="TSV" header="1" uniqueRows="1">
    <Dataset name="{source_dataset}" interface="default">
        <Attribute name="ensembl_gene_id"/>
        <Attribute name="external_gene_name"/>
        <Attribute name="{target_short}_homolog_ensembl_gene"/>
        <Attribute name="{target_short}_homolog_associated_gene_name"/>
        <Attribute name="{target_short}_homolog_orthology_type"/>
    </Dataset>
</Query>"""

            try:
                resp = requests.get(
                    ENSEMBL_BIOMART,
                    params={"query": xml_query},
                    timeout=120,
                )
                resp.raise_for_status()

                # Parse TSV response
                lines = resp.text.strip().split("\n")
                if len(lines) < 2:
                    logger.warning(f"Empty BioMart response for {source}->{target}")
                    progress.advance(task)
                    continue

                header = lines[0].split("\t")
                pair_key = f"{source}__{target}"
                pair_orthologs: list[dict[str, str]] = []

                for line in lines[1:]:
                    fields = line.split("\t")
                    if len(fields) < 4:
                        continue
                    source_gene = fields[1] or fields[0]  # prefer symbol, fall back to ensembl
                    target_gene = fields[3] or fields[2]
                    orth_type = fields[4] if len(fields) > 4 else ""

                    if source_gene and target_gene:
                        pair_orthologs.append({
                            "source_gene": source_gene,
                            "target_gene": target_gene,
                            "type": orth_type,
                        })

                # Group into ortholog groups
                for orth in pair_orthologs:
                    # Check if either gene already in an existing group
                    found_group = None
                    for gid, members in all_orthologs.items():
                        if (orth["source_gene"] in members.get(source, [])
                                or orth["target_gene"] in members.get(target, [])):
                            found_group = gid
                            break

                    if found_group:
                        all_orthologs[found_group].setdefault(source, [])
                        all_orthologs[found_group].setdefault(target, [])
                        if orth["source_gene"] not in all_orthologs[found_group][source]:
                            all_orthologs[found_group][source].append(orth["source_gene"])
                        if orth["target_gene"] not in all_orthologs[found_group][target]:
                            all_orthologs[found_group][target].append(orth["target_gene"])
                    else:
                        gid = f"OG_{group_counter:06d}"
                        group_counter += 1
                        all_orthologs[gid] = {
                            source: [orth["source_gene"]],
                            target: [orth["target_gene"]],
                        }

                logger.info(
                    f"BioMart orthologs {source}<->{target}: "
                    f"{len(pair_orthologs)} pairs"
                )

            except Exception as exc:
                logger.warning(f"BioMart query failed for {source}->{target}: {exc}")

            progress.advance(task)

        # --- NCBI HomoloGene as fallback (especially for Daphnia) ---
        logger.info("Downloading NCBI HomoloGene for additional ortholog coverage")
        try:
            homologene_path = output_dir / "homologene.data"
            resp = requests.get(HOMOLOGENE_URL, timeout=120, stream=True)
            resp.raise_for_status()
            with open(homologene_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Parse HomoloGene: columns are HID, TaxID, GeneID, Symbol, ...
            target_taxids = set(SPECIES_TAXIDS.values())
            homolog_groups: dict[str, dict[int, list[str]]] = {}

            with open(homologene_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) < 4:
                        continue
                    hid = parts[0]
                    tax_id = int(parts[1])
                    symbol = parts[3]

                    if tax_id in target_taxids:
                        homolog_groups.setdefault(hid, {}).setdefault(tax_id, []).append(symbol)

            # Merge HomoloGene groups into unified orthologs
            taxid_to_species = {v: k for k, v in SPECIES_TAXIDS.items()}
            for hid, species_genes in homolog_groups.items():
                if len(species_genes) < 2:
                    continue
                # Only add groups not already covered by BioMart
                gid = f"HG_{hid}"
                all_orthologs[gid] = {}
                for tax_id, genes in species_genes.items():
                    species_name = taxid_to_species.get(tax_id, str(tax_id))
                    all_orthologs[gid][species_name] = genes

            logger.info(
                f"HomoloGene: {len(homolog_groups)} groups with target species"
            )

        except Exception as exc:
            logger.warning(f"HomoloGene download failed: {exc}")

        progress.advance(task)

    # Save unified ortholog mapping
    ortholog_path = output_dir / "ortholog_mappings.json"
    with open(ortholog_path, "w", encoding="utf-8") as f:
        json.dump(all_orthologs, f, indent=2)

    logger.info(
        f"Unified ortholog mapping: {len(all_orthologs)} groups -> {ortholog_path}"
    )
    return ortholog_path


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
