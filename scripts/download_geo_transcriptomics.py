#!/usr/bin/env python3
"""Download real aquatic toxicogenomics data from NCBI GEO.

Searches for gene expression datasets from aquatic model organisms
(zebrafish, Daphnia, fathead minnow) exposed to contaminants.
Downloads expression matrices and metadata for ToxiGene training.

MIT License — Bryan Cheng, 2026
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data/raw/molecular/geo")
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = Path("data/processed/molecular/real")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# Key search terms for aquatic toxicogenomics
SEARCH_QUERIES = [
    '"Danio rerio"[Organism] AND "toxicology"[MeSH] AND "expression profiling"[Filter]',
    '"Danio rerio"[Organism] AND "chemical exposure" AND "expression profiling"[Filter]',
    '"Daphnia"[Organism] AND "expression profiling"[Filter]',
    '"Pimephales promelas"[Organism] AND "expression profiling"[Filter]',
    '"Oncorhynchus mykiss"[Organism] AND "toxicology" AND "expression profiling"[Filter]',
]

# Contaminant class keywords for labeling
CONTAMINANT_KEYWORDS = {
    "heavy_metal": ["cadmium", "lead", "mercury", "arsenic", "copper", "zinc", "chromium", "nickel", "metal"],
    "pesticide": ["atrazine", "chlorpyrifos", "DDT", "pesticide", "insecticide", "herbicide", "organophosphate"],
    "pharmaceutical": ["estrogen", "estradiol", "ibuprofen", "pharmaceutical", "drug", "endocrine"],
    "pah": ["benzo", "pyrene", "PAH", "polycyclic aromatic"],
    "pcb": ["PCB", "polychlorinated biphenyl", "dioxin"],
    "pfas": ["PFOS", "PFOA", "PFAS", "perfluoro"],
    "nanomaterial": ["nanoparticle", "nano", "TiO2", "ZnO"],
    "nutrient": ["nitrogen", "phosphorus", "eutrophication", "ammonia", "nitrate"],
}


def search_geo(query, max_results=100):
    """Search GEO for datasets matching query."""
    from Bio import Entrez
    Entrez.email = "sentinel@example.com"

    try:
        handle = Entrez.esearch(db="gds", term=query, retmax=max_results)
        record = Entrez.read(handle)
        handle.close()
        ids = record.get("IdList", [])
        log(f"  Found {len(ids)} GEO datasets for: {query[:60]}...")
        return ids
    except Exception as e:
        log(f"  GEO search failed: {e}")
        return []


def download_geo_dataset(gse_id):
    """Download a GEO dataset using GEOparse."""
    import GEOparse

    cache_path = DATA_DIR / f"{gse_id}_family.soft.gz"
    if cache_path.exists():
        return GEOparse.get_GEO(filepath=str(cache_path), silent=True)

    try:
        gse = GEOparse.get_GEO(geo=gse_id, destdir=str(DATA_DIR), silent=True)
        return gse
    except Exception as e:
        log(f"    Failed to download {gse_id}: {e}")
        return None


def classify_contaminant(text):
    """Classify contaminant type from dataset description."""
    text_lower = text.lower()
    for cls, keywords in CONTAMINANT_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in text_lower:
                return cls
    return "unknown"


def extract_expression_data(gse, gse_id):
    """Extract expression matrix and metadata from a GEO dataset."""
    if gse is None:
        return None

    try:
        # Get metadata
        title = gse.metadata.get("title", [""])[0]
        summary = gse.metadata.get("summary", [""])[0]
        organism = gse.metadata.get("platform_organism", [""])[0]
        if not organism:
            organism = gse.metadata.get("sample_organism", ["unknown"])[0]

        # Classify contaminant
        contaminant_class = classify_contaminant(f"{title} {summary}")

        # Try to get expression table
        if hasattr(gse, "pivot_samples") and callable(gse.pivot_samples):
            expr_df = gse.pivot_samples("VALUE")
        elif gse.gsms:
            # Build expression matrix from individual samples
            sample_data = {}
            for gsm_name, gsm in gse.gsms.items():
                if hasattr(gsm, "table") and gsm.table is not None and len(gsm.table) > 0:
                    if "VALUE" in gsm.table.columns:
                        sample_data[gsm_name] = gsm.table.set_index("ID_REF")["VALUE"]
            if sample_data:
                expr_df = pd.DataFrame(sample_data)
            else:
                return None
        else:
            return None

        if expr_df.empty or expr_df.shape[1] < 3:
            return None

        # Convert to numeric
        expr_df = expr_df.apply(pd.to_numeric, errors="coerce")
        expr_df = expr_df.dropna(how="all")

        if len(expr_df) < 100:
            return None

        return {
            "gse_id": gse_id,
            "title": title,
            "organism": organism,
            "contaminant_class": contaminant_class,
            "n_genes": len(expr_df),
            "n_samples": expr_df.shape[1],
            "expression": expr_df.values.astype(np.float32),
            "gene_ids": list(expr_df.index),
            "sample_ids": list(expr_df.columns),
        }
    except Exception as e:
        log(f"    Expression extraction failed for {gse_id}: {e}")
        return None


def main():
    log("=" * 60)
    log("GEO Transcriptomics Download for ToxiGene")
    log("=" * 60)

    # Check if BioPython is available
    try:
        from Bio import Entrez
    except ImportError:
        log("BioPython not installed. Using direct GEOparse search instead.")
        # Fallback: search using GEOparse directly
        import GEOparse
        # Use known zebrafish toxicogenomics GSE IDs
        known_gses = [
            "GSE153551",  # Zebrafish PFOS exposure
            "GSE164437",  # Zebrafish cadmium
            "GSE122556",  # Zebrafish atrazine
            "GSE133548",  # Zebrafish estradiol
            "GSE97434",   # Zebrafish BPA
            "GSE73661",   # Zebrafish PAH
            "GSE59768",   # Zebrafish arsenic
            "GSE143795",  # Daphnia magna contaminants
            "GSE104776",  # Zebrafish lead
            "GSE83514",   # Zebrafish copper
            "GSE114036",  # Fathead minnow estrogen
            "GSE54800",   # Daphnia magna metals
            "GSE130306",  # Zebrafish microplastics
            "GSE108634",  # Zebrafish organophosphate
            "GSE126666",  # Zebrafish pharmaceutical mixture
        ]

        all_datasets = []
        for gse_id in known_gses:
            log(f"  Downloading {gse_id}...")
            gse = download_geo_dataset(gse_id)
            if gse is not None:
                data = extract_expression_data(gse, gse_id)
                if data is not None:
                    out_path = OUT_DIR / f"{gse_id}.npz"
                    np.savez_compressed(
                        out_path,
                        expression=data["expression"],
                        gene_ids=data["gene_ids"],
                        sample_ids=data["sample_ids"],
                        contaminant_class=data["contaminant_class"],
                        organism=data["organism"],
                        title=data["title"],
                    )
                    all_datasets.append({
                        "gse_id": gse_id,
                        "organism": data["organism"],
                        "contaminant": data["contaminant_class"],
                        "n_genes": data["n_genes"],
                        "n_samples": data["n_samples"],
                    })
                    log(f"    Saved: {data['n_genes']} genes x {data['n_samples']} samples ({data['organism']}, {data['contaminant_class']})")
                else:
                    log(f"    No expression data extracted")
            time.sleep(1)

        log(f"\nTotal datasets: {len(all_datasets)}")
        with open(OUT_DIR / "geo_summary.json", "w") as f:
            json.dump(all_datasets, f, indent=2)
        log("DONE")
        return

    # Full BioPython path
    all_geo_ids = set()
    for query in SEARCH_QUERIES:
        ids = search_geo(query, max_results=200)
        all_geo_ids.update(ids)
        time.sleep(1)

    log(f"Total unique GEO dataset IDs: {len(all_geo_ids)}")

    # Convert GDS IDs to GSE IDs and download
    all_datasets = []
    for gds_id in list(all_geo_ids)[:50]:  # Limit to 50 for initial download
        log(f"  Processing GDS {gds_id}...")
        try:
            handle = Entrez.esummary(db="gds", id=gds_id)
            summary = Entrez.read(handle)
            handle.close()
            if summary:
                accession = summary[0].get("Accession", "")
                if accession.startswith("GSE"):
                    gse = download_geo_dataset(accession)
                    if gse:
                        data = extract_expression_data(gse, accession)
                        if data:
                            out_path = OUT_DIR / f"{accession}.npz"
                            np.savez_compressed(
                                out_path,
                                expression=data["expression"],
                                gene_ids=data["gene_ids"],
                                sample_ids=data["sample_ids"],
                                contaminant_class=data["contaminant_class"],
                                organism=data["organism"],
                            )
                            all_datasets.append({
                                "gse_id": accession,
                                "organism": data["organism"],
                                "contaminant": data["contaminant_class"],
                                "n_genes": data["n_genes"],
                                "n_samples": data["n_samples"],
                            })
                            log(f"    Saved: {data['n_genes']} genes x {data['n_samples']} samples")
        except Exception as e:
            log(f"    Failed: {e}")
        time.sleep(2)

    log(f"\nTotal datasets downloaded: {len(all_datasets)}")
    with open(OUT_DIR / "geo_summary.json", "w") as f:
        json.dump(all_datasets, f, indent=2)
    log("DONE")


if __name__ == "__main__":
    main()
