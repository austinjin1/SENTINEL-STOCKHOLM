"""
Molecular/toxicogenomics preprocessing for SENTINEL.

Pipeline:
  1. RNA-seq: TPM normalization -> log2 transform
  2. Microarray: RMA normalization -> quantile normalization
  3. Cross-platform batch correction via ComBat
  4. Curated ~200 stress-response gene panel (7 pathways)
  5. Binary pathway activation labelling
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd
import scipy.sparse

from sentinel.utils.config import load_config
from sentinel.utils.logging import get_logger, make_progress

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Stress-response gene panel (curated ~200 genes across 7 pathways)
# ---------------------------------------------------------------------------

PATHWAY_NAMES = [
    "AHR/CYP1A",
    "Metallothionein",
    "Estrogen/Endocrine",
    "Cholinesterase",
    "Oxidative Stress",
    "Heat Shock",
    "DNA Damage",
]

# Representative genes per pathway (gene symbols).
# In production, this would be loaded from a curated TSV; here we define
# the core markers that anchor each pathway signature.
STRESS_GENE_PANEL: dict[str, list[str]] = {
    "AHR/CYP1A": [
        "AHR", "ARNT", "CYP1A1", "CYP1A2", "CYP1B1", "AHRR", "TIPARP",
        "NQO1", "ALDH3A1", "UGT1A1", "UGT1A6", "GSTA1", "GSTA2",
        "EPHX1", "CYP2B6", "CYP3A4", "ABCB1", "ABCC2", "ABCG2",
        "SLC22A1", "SLC7A11", "AKR1B10", "AKR1C1", "AKR1C2", "AKR1C3",
        "CBR1", "CBR3", "FMO1", "FMO3",
    ],
    "Metallothionein": [
        "MT1A", "MT1B", "MT1E", "MT1F", "MT1G", "MT1H", "MT1M", "MT1X",
        "MT2A", "MT3", "MT4", "SLC30A1", "SLC30A2", "SLC39A1", "SLC39A4",
        "SLC39A14", "DMT1", "ATP7A", "ATP7B", "HMOX1", "HMOX2",
        "ZIP8", "ZIP14", "ZNT1", "ZNT2", "PCNA", "FTH1", "FTL",
    ],
    "Estrogen/Endocrine": [
        "ESR1", "ESR2", "AR", "PGR", "GPER1", "CYP19A1", "CYP17A1",
        "HSD17B1", "HSD17B2", "HSD3B1", "HSD3B2", "SRD5A1", "SRD5A2",
        "SULT1E1", "SHBG", "VTG1", "VTG2", "STAR", "NR5A1", "NR0B1",
        "FSHR", "LHCGR", "GnRHR", "KISS1", "KISS1R", "SPP1",
        "TFF1", "GREB1", "PGR", "MYC",
    ],
    "Cholinesterase": [
        "ACHE", "BCHE", "CHAT", "SLC18A3", "CHRNA1", "CHRNA7", "CHRNB1",
        "CHRM1", "CHRM2", "CHRM3", "SLC5A7", "COLQ", "PRIMA1",
        "NKAIN1", "PRSS12", "NLGN1", "NLGN2", "NRXN1",
        "SYN1", "SYP", "SNAP25", "SYT1", "VAMP2", "STX1A",
        "GRIA1", "GRIA2", "GRIN1", "GRIN2A", "GABRA1",
    ],
    "Oxidative Stress": [
        "SOD1", "SOD2", "SOD3", "CAT", "GPX1", "GPX2", "GPX3", "GPX4",
        "GSR", "GCLC", "GCLM", "GSS", "GSTP1", "GSTM1", "GSTT1",
        "TXNRD1", "TXN", "PRDX1", "PRDX2", "PRDX3", "PRDX4", "PRDX5",
        "PRDX6", "NFE2L2", "KEAP1", "MAFK", "SQSTM1", "HMOX1",
        "NQO1", "FTH1",
    ],
    "Heat Shock": [
        "HSPA1A", "HSPA1B", "HSPA2", "HSPA4", "HSPA5", "HSPA6", "HSPA8",
        "HSP90AA1", "HSP90AB1", "HSP90B1", "HSPB1", "HSPB5", "HSPB8",
        "HSPD1", "HSPE1", "HSPH1", "HSF1", "HSF2", "DNAJB1", "DNAJB4",
        "DNAJB6", "DNAJC3", "BAG3", "SERPINH1", "CCT2", "CCT5",
        "TCP1", "STIP1", "FKBP4", "PPID",
    ],
    "DNA Damage": [
        "TP53", "MDM2", "CDKN1A", "GADD45A", "GADD45B", "GADD45G",
        "ATM", "ATR", "CHEK1", "CHEK2", "BRCA1", "BRCA2", "RAD51",
        "RAD50", "MRE11", "NBN", "H2AX", "RPA1", "XPC", "XPA",
        "ERCC1", "ERCC2", "MSH2", "MSH6", "MLH1", "PMS2",
        "PARP1", "XRCC1", "OGG1", "APEX1",
    ],
}

# Flatten for lookup
ALL_PANEL_GENES: set[str] = set()
GENE_TO_PATHWAYS: dict[str, list[str]] = {}
for _pw, _genes in STRESS_GENE_PANEL.items():
    for _g in _genes:
        ALL_PANEL_GENES.add(_g)
        GENE_TO_PATHWAYS.setdefault(_g, []).append(_pw)


# ---------------------------------------------------------------------------
# RNA-seq normalization: TPM -> log2
# ---------------------------------------------------------------------------


def tpm_normalize(counts: pd.DataFrame, gene_lengths: pd.Series) -> pd.DataFrame:
    """Compute Transcripts Per Million (TPM) from raw counts.

    Parameters
    ----------
    counts:
        Raw count matrix (genes x samples).
    gene_lengths:
        Gene lengths in base pairs, indexed by gene name/ID.

    Returns
    -------
    TPM-normalized DataFrame (genes x samples).
    """
    # Align gene lengths with count matrix rows
    lengths = gene_lengths.reindex(counts.index)
    if lengths.isna().any():
        logger.warning(
            f"{lengths.isna().sum()} genes missing length info; using median length"
        )
        lengths = lengths.fillna(lengths.median())

    # Rate = counts / length (in kb)
    rate = counts.div(lengths / 1000.0, axis=0)

    # TPM = rate / sum(rate) * 1e6
    tpm = rate.div(rate.sum(axis=0), axis=1) * 1e6
    return tpm


def log2_transform(
    tpm: pd.DataFrame,
    pseudocount: float = 1.0,
) -> pd.DataFrame:
    """log2(TPM + pseudocount) transformation."""
    return np.log2(tpm + pseudocount)


# ---------------------------------------------------------------------------
# Microarray normalization
# ---------------------------------------------------------------------------


def quantile_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Quantile normalization of a matrix (genes x samples).

    Each column's distribution is forced to match the mean rank distribution.
    """
    rank_mean = df.stack().groupby(df.rank(method="first").stack().astype(int)).mean()

    result = df.rank(method="min").astype(int)
    for col in result.columns:
        result[col] = result[col].map(rank_mean)
    return result


def rma_normalize(raw_intensities: pd.DataFrame) -> pd.DataFrame:
    """Simplified RMA-like normalization for microarray data.

    Steps: background correction (log2) -> quantile normalization.
    In production, use ``rpy2`` with the ``affy`` or ``oligo`` R package
    for full RMA. This implementation serves as a pure-Python fallback.
    """
    # Background correction: log2 of positive values
    corrected = raw_intensities.clip(lower=1.0)
    log_data = np.log2(corrected)

    # Quantile normalization
    normalized = quantile_normalize(log_data)
    return normalized


# ---------------------------------------------------------------------------
# Cross-platform batch correction (ComBat)
# ---------------------------------------------------------------------------


def combat_batch_correction(
    expression: pd.DataFrame,
    batch_labels: pd.Series,
    *,
    covariates: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Batch correction using ComBat (parametric empirical Bayes).

    Wraps ``pycombat`` from the ``combat`` package. Falls back to a
    simple mean-centering approach if the package is unavailable.

    Parameters
    ----------
    expression:
        Gene expression matrix (genes x samples).
    batch_labels:
        Series mapping sample names to batch identifiers.
    covariates:
        Optional biological covariates to preserve.

    Returns
    -------
    Batch-corrected expression matrix.
    """
    try:
        from combat.pycombat import pycombat

        corrected = pycombat(expression, batch_labels)
        logger.info("ComBat batch correction applied")
        return corrected
    except ImportError:
        logger.warning(
            "pycombat not installed; falling back to mean-centering per batch"
        )
        return _mean_center_batch_correction(expression, batch_labels)


def _mean_center_batch_correction(
    expression: pd.DataFrame,
    batch_labels: pd.Series,
) -> pd.DataFrame:
    """Simple mean-centering batch correction as fallback."""
    corrected = expression.copy()
    grand_mean = expression.mean(axis=1)

    for batch_id in batch_labels.unique():
        batch_samples = batch_labels[batch_labels == batch_id].index
        batch_cols = [c for c in batch_samples if c in corrected.columns]
        if not batch_cols:
            continue
        batch_mean = corrected[batch_cols].mean(axis=1)
        corrected[batch_cols] = corrected[batch_cols].sub(batch_mean - grand_mean, axis=0)

    return corrected


# ---------------------------------------------------------------------------
# Pathway activation labelling
# ---------------------------------------------------------------------------


@dataclass
class PathwayActivation:
    """Binary pathway activation labels for a sample."""

    sample_id: str
    labels: dict[str, int] = field(default_factory=dict)  # pathway -> 0/1

    def to_array(self) -> np.ndarray:
        """Return labels as a (7,) binary array in PATHWAY_NAMES order."""
        return np.array([self.labels.get(pw, 0) for pw in PATHWAY_NAMES], dtype=np.int8)


# Default fold-change thresholds per pathway
DEFAULT_FC_THRESHOLDS: dict[str, float] = {
    "AHR/CYP1A": 1.5,
    "Metallothionein": 2.0,
    "Estrogen/Endocrine": 1.5,
    "Cholinesterase": -1.5,  # negative = suppression
    "Oxidative Stress": 1.5,
    "Heat Shock": 2.0,
    "DNA Damage": 1.5,
}


def label_pathway_activation(
    expression: pd.DataFrame,
    control_samples: Sequence[str],
    treated_samples: Sequence[str],
    *,
    fc_thresholds: dict[str, float] | None = None,
    min_genes_activated: int = 3,
) -> list[PathwayActivation]:
    """Label pathway activation for each treated sample.

    A pathway is considered activated if at least ``min_genes_activated``
    genes in its panel exceed the fold-change threshold relative to
    control mean.

    Parameters
    ----------
    expression:
        Log2-transformed expression matrix (genes x samples).
    control_samples:
        Sample IDs for the control group.
    treated_samples:
        Sample IDs for treated/exposed samples.
    fc_thresholds:
        Per-pathway fold-change threshold (in log2 space).
    min_genes_activated:
        Minimum pathway genes exceeding threshold to call activation.

    Returns
    -------
    List of PathwayActivation, one per treated sample.
    """
    fc_thresholds = fc_thresholds or DEFAULT_FC_THRESHOLDS

    # Compute control means
    ctrl_cols = [c for c in control_samples if c in expression.columns]
    if not ctrl_cols:
        raise ValueError("No control samples found in expression matrix")
    control_mean = expression[ctrl_cols].mean(axis=1)

    results: list[PathwayActivation] = []

    for sample in treated_samples:
        if sample not in expression.columns:
            continue

        log2fc = expression[sample] - control_mean
        labels: dict[str, int] = {}

        for pathway in PATHWAY_NAMES:
            genes = STRESS_GENE_PANEL[pathway]
            present_genes = [g for g in genes if g in log2fc.index]

            if not present_genes:
                labels[pathway] = 0
                continue

            threshold = fc_thresholds.get(pathway, 1.5)
            if threshold < 0:
                # Suppression pathway (e.g., cholinesterase)
                activated_count = sum(
                    1 for g in present_genes if log2fc[g] <= threshold
                )
            else:
                activated_count = sum(
                    1 for g in present_genes if log2fc[g] >= threshold
                )

            labels[pathway] = 1 if activated_count >= min_genes_activated else 0

        results.append(PathwayActivation(sample_id=sample, labels=labels))

    n_active = sum(sum(r.labels.values()) for r in results)
    logger.info(
        f"Pathway labelling: {len(results)} samples, "
        f"{n_active} total activations across all pathways"
    )
    return results


# ---------------------------------------------------------------------------
# Gene panel subsetting
# ---------------------------------------------------------------------------


def subset_to_panel(
    expression: pd.DataFrame,
    panel: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Subset expression matrix to curated stress-response gene panel.

    Parameters
    ----------
    expression:
        Full expression matrix (genes x samples).
    panel:
        Gene panel dict (default: STRESS_GENE_PANEL).

    Returns
    -------
    Subsetted expression matrix with only panel genes.
    """
    panel = panel or STRESS_GENE_PANEL
    all_genes = set()
    for genes in panel.values():
        all_genes.update(genes)

    present = [g for g in all_genes if g in expression.index]
    missing = all_genes - set(present)
    if missing:
        logger.info(
            f"Panel subsetting: {len(present)}/{len(all_genes)} genes present "
            f"({len(missing)} missing)"
        )
    return expression.loc[present]


# ---------------------------------------------------------------------------
# Full preprocessing pipeline
# ---------------------------------------------------------------------------


def preprocess_rnaseq(
    counts_path: str | Path,
    gene_lengths_path: str | Path,
    output_dir: str | Path,
    *,
    sample_metadata_path: str | Path | None = None,
) -> dict[str, Any]:
    """Full RNA-seq preprocessing pipeline.

    Steps: load counts -> TPM -> log2 -> panel subset -> save.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading RNA-seq count data")
    counts = pd.read_csv(counts_path, index_col=0)
    gene_lengths = pd.read_csv(gene_lengths_path, index_col=0, squeeze=True)

    # TPM normalization
    tpm = tpm_normalize(counts, gene_lengths)
    log2_tpm = log2_transform(tpm)

    # Subset to panel
    panel_expr = subset_to_panel(log2_tpm)

    # Save
    log2_tpm.to_parquet(output_dir / "log2_tpm_full.parquet")
    panel_expr.to_parquet(output_dir / "log2_tpm_panel.parquet")

    return {
        "full_genes": log2_tpm.shape[0],
        "panel_genes": panel_expr.shape[0],
        "samples": log2_tpm.shape[1],
    }


def preprocess_microarray(
    raw_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Full microarray preprocessing pipeline.

    Steps: load -> RMA -> quantile normalization -> panel subset -> save.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading microarray data")
    raw = pd.read_csv(raw_path, index_col=0)

    # RMA-like normalization
    normalized = rma_normalize(raw)

    # Subset to panel
    panel_expr = subset_to_panel(normalized)

    # Save
    normalized.to_parquet(output_dir / "rma_normalized_full.parquet")
    panel_expr.to_parquet(output_dir / "rma_normalized_panel.parquet")

    return {
        "full_genes": normalized.shape[0],
        "panel_genes": panel_expr.shape[0],
        "samples": normalized.shape[1],
    }


def preprocess_and_integrate(
    expression_paths: Sequence[str | Path],
    platform_labels: Sequence[str],
    batch_ids: Sequence[str],
    output_dir: str | Path,
) -> pd.DataFrame:
    """Integrate multiple expression datasets with batch correction.

    Parameters
    ----------
    expression_paths:
        Paths to normalized expression Parquet files (genes x samples).
    platform_labels:
        Platform type per dataset (``"rnaseq"`` or ``"microarray"``).
    batch_ids:
        Batch identifier per dataset.
    output_dir:
        Output directory.

    Returns
    -------
    Integrated, batch-corrected expression matrix.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and merge all datasets
    all_dfs: list[pd.DataFrame] = []
    batch_series_parts: list[pd.Series] = []

    for path, batch_id in zip(expression_paths, batch_ids):
        df = pd.read_parquet(path)
        all_dfs.append(df)
        batch_series_parts.append(
            pd.Series(batch_id, index=df.columns, name="batch")
        )

    # Find common genes
    common_genes = set(all_dfs[0].index)
    for df in all_dfs[1:]:
        common_genes &= set(df.index)
    common_genes_list = sorted(common_genes)

    logger.info(f"Integrating {len(all_dfs)} datasets, {len(common_genes_list)} common genes")

    # Merge on common genes
    merged = pd.concat([df.loc[common_genes_list] for df in all_dfs], axis=1)
    batch_series = pd.concat(batch_series_parts)

    # ComBat batch correction
    corrected = combat_batch_correction(merged, batch_series)

    # Subset to panel
    panel = subset_to_panel(corrected)

    # Save
    corrected.to_parquet(output_dir / "integrated_full.parquet")
    panel.to_parquet(output_dir / "integrated_panel.parquet")

    logger.info(
        f"Integration complete: {corrected.shape[0]} genes x "
        f"{corrected.shape[1]} samples -> {output_dir}"
    )
    return panel


# ---------------------------------------------------------------------------
# AOP-Wiki JSON parsing
# ---------------------------------------------------------------------------


def parse_aop_wiki_json(aop_json_path: str | Path) -> dict:
    """Parse downloaded AOP-Wiki JSON into structured hierarchy.

    Parameters
    ----------
    aop_json_path:
        Path to the aop_hierarchy.json file produced by ``download_aop_wiki``.

    Returns
    -------
    Dict with keys:
    - gene_to_mie: gene symbol -> list of MIE IDs
    - mie_to_ke: MIE ID -> list of KE IDs
    - ke_to_ao: KE ID -> list of AO IDs
    - full_aop_chains: list of complete AOP chains (gene->MIE->KE->AO)
    - key_events: dict of all key events with metadata
    - panel_gene_aop_links: subset mapped to STRESS_GENE_PANEL genes
    """
    aop_json_path = Path(aop_json_path)
    with open(aop_json_path, "r", encoding="utf-8") as f:
        hierarchy = json.load(f)

    gene_to_mie: dict[str, list[str]] = hierarchy.get("gene_to_mie", {})
    mie_to_ke: dict[str, list[str]] = hierarchy.get("mie_to_ke", {})
    ke_to_ao: dict[str, list[str]] = hierarchy.get("ke_to_ao", {})
    key_events: dict[str, Any] = hierarchy.get("key_events", {})

    # Build full AOP chains: Gene -> MIE -> KE -> AO
    full_chains: list[dict[str, str]] = []
    for gene, mie_ids in gene_to_mie.items():
        for mie_id in mie_ids:
            ke_ids = mie_to_ke.get(mie_id, [])
            if not ke_ids:
                # Direct MIE -> AO (skip KE)
                ao_ids = ke_to_ao.get(mie_id, [])
                for ao_id in ao_ids:
                    full_chains.append({
                        "gene": gene,
                        "mie": mie_id,
                        "ke": None,
                        "ao": ao_id,
                    })
            for ke_id in ke_ids:
                ao_ids = ke_to_ao.get(ke_id, [])
                for ao_id in ao_ids:
                    full_chains.append({
                        "gene": gene,
                        "mie": mie_id,
                        "ke": ke_id,
                        "ao": ao_id,
                    })
                if not ao_ids:
                    full_chains.append({
                        "gene": gene,
                        "mie": mie_id,
                        "ke": ke_id,
                        "ao": None,
                    })

    # Map to STRESS_GENE_PANEL where possible
    panel_gene_aop_links: dict[str, list[str]] = {}
    for gene in gene_to_mie:
        gene_upper = gene.upper()
        if gene_upper in ALL_PANEL_GENES:
            panel_gene_aop_links[gene_upper] = gene_to_mie[gene]

    logger.info(
        f"AOP-Wiki parsed: {len(gene_to_mie)} gene->MIE links, "
        f"{len(full_chains)} full chains, "
        f"{len(panel_gene_aop_links)} panel gene matches"
    )

    return {
        "gene_to_mie": gene_to_mie,
        "mie_to_ke": mie_to_ke,
        "ke_to_ao": ke_to_ao,
        "full_aop_chains": full_chains,
        "key_events": key_events,
        "panel_gene_aop_links": panel_gene_aop_links,
    }


# ---------------------------------------------------------------------------
# Reactome GMT parsing
# ---------------------------------------------------------------------------


def parse_reactome_gmt(
    gmt_path: str | Path,
    species_filter: str | None = None,
) -> dict[str, list[str]]:
    """Parse Reactome GMT file (pathway -> gene set).

    GMT format: each line is ``pathway_name<TAB>url<TAB>gene1<TAB>gene2<TAB>...``

    Parameters
    ----------
    gmt_path:
        Path to the GMT file.
    species_filter:
        If provided, only keep pathways whose name contains this species
        string (e.g., "Homo sapiens", "Danio rerio").

    Returns
    -------
    Dict mapping pathway name to list of gene symbols.
    """
    gmt_path = Path(gmt_path)
    pathways: dict[str, list[str]] = {}

    with open(gmt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue

            pathway_name = parts[0]

            # Species filter: Reactome GMT pathway names often include species
            if species_filter and species_filter not in pathway_name:
                continue

            # parts[1] is typically a URL or description; genes start at parts[2]
            genes = [g.strip() for g in parts[2:] if g.strip()]
            if genes:
                pathways[pathway_name] = genes

    logger.info(
        f"Reactome GMT: {len(pathways)} pathways"
        + (f" (filtered: {species_filter})" if species_filter else "")
    )
    return pathways


# ---------------------------------------------------------------------------
# ToxiGene hierarchy graph (P-NET-style sparse adjacency matrices)
# ---------------------------------------------------------------------------


def build_hierarchy_graph(
    reactome_dir: str | Path,
    aop_dir: str | Path,
    gene_panel: dict[str, list[str]] | None = None,
) -> dict:
    """Construct ToxiGene's P-NET-style sparse adjacency matrices.

    Builds a 3-layer biological hierarchy:
      - Layer 1: Gene -> Pathway (from Reactome GMT + STRESS_GENE_PANEL)
      - Layer 2: Pathway -> Biological Process (from Reactome hierarchy)
      - Layer 3: Biological Process -> Adverse Outcome (from AOP-Wiki)

    Each layer is a sparse binary matrix (scipy.sparse.csr_matrix).
    Connections are constrained to known biological relationships only.

    Parameters
    ----------
    reactome_dir:
        Directory containing downloaded Reactome files.
    aop_dir:
        Directory containing downloaded AOP-Wiki files.
    gene_panel:
        Gene panel dict (default: STRESS_GENE_PANEL).

    Returns
    -------
    Dict with:
    - adjacency_matrices: list of 3 sparse CSR matrices
    - layer_names: ["Gene->Pathway", "Pathway->Process", "Process->Outcome"]
    - gene_names: list of gene names (rows of layer 1)
    - pathway_names: list of pathway names (cols of layer 1 / rows of layer 2)
    - process_names: list of biological process names (cols of layer 2 / rows of layer 3)
    - outcome_names: list of adverse outcome names (cols of layer 3)
    """
    reactome_dir = Path(reactome_dir)
    aop_dir = Path(aop_dir)
    gene_panel = gene_panel or STRESS_GENE_PANEL

    # --- Collect gene names from panel ---
    all_genes = sorted({g for genes in gene_panel.values() for g in genes})
    gene_idx = {g: i for i, g in enumerate(all_genes)}

    # --- Layer 1: Gene -> Pathway ---
    # Start with STRESS_GENE_PANEL pathways
    pathway_set: set[str] = set(gene_panel.keys())

    # Augment with Reactome GMT if available
    gmt_files = list(reactome_dir.glob("*.gmt"))
    reactome_pathways: dict[str, list[str]] = {}
    if gmt_files:
        reactome_pathways = parse_reactome_gmt(gmt_files[0])
        # Only keep pathways that contain at least one panel gene
        for pw_name, pw_genes in reactome_pathways.items():
            if any(g in gene_idx for g in pw_genes):
                pathway_set.add(pw_name)

    pathway_names = sorted(pathway_set)
    pathway_idx = {pw: i for i, pw in enumerate(pathway_names)}

    # Build sparse matrix: genes x pathways
    rows_l1: list[int] = []
    cols_l1: list[int] = []

    # From curated panel
    for pw_name, pw_genes in gene_panel.items():
        pw_i = pathway_idx[pw_name]
        for g in pw_genes:
            if g in gene_idx:
                rows_l1.append(gene_idx[g])
                cols_l1.append(pw_i)

    # From Reactome GMT
    for pw_name, pw_genes in reactome_pathways.items():
        if pw_name not in pathway_idx:
            continue
        pw_i = pathway_idx[pw_name]
        for g in pw_genes:
            if g in gene_idx:
                rows_l1.append(gene_idx[g])
                cols_l1.append(pw_i)

    layer1 = scipy.sparse.csr_matrix(
        (np.ones(len(rows_l1), dtype=np.float32), (rows_l1, cols_l1)),
        shape=(len(all_genes), len(pathway_names)),
    )
    # Binarize (in case of duplicates)
    layer1.data[:] = 1.0

    # --- Layer 2: Pathway -> Biological Process ---
    # Use Reactome parent-child relationships
    process_set: set[str] = set()
    pathway_to_process: dict[str, list[str]] = {}

    rel_files = list(reactome_dir.glob("*PathwaysRelation*"))
    if rel_files:
        with open(rel_files[0], "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    parent, child = parts[0], parts[1]
                    # If child is a pathway we have, parent is the process
                    # Reactome uses IDs, but we may have names; use IDs as process names
                    process_set.add(parent)
                    pathway_to_process.setdefault(child, []).append(parent)

    # If no Reactome relations found, use PATHWAY_NAMES as processes too
    if not process_set:
        logger.warning(
            "No Reactome pathway relations found; using panel pathways as processes"
        )
        process_set = set(PATHWAY_NAMES)
        for pw in PATHWAY_NAMES:
            pathway_to_process[pw] = [pw]

    process_names = sorted(process_set)
    process_idx = {p: i for i, p in enumerate(process_names)}

    rows_l2: list[int] = []
    cols_l2: list[int] = []

    for pw_name, processes in pathway_to_process.items():
        if pw_name not in pathway_idx:
            continue
        pw_i = pathway_idx[pw_name]
        for proc in processes:
            if proc in process_idx:
                rows_l2.append(pw_i)
                cols_l2.append(process_idx[proc])

    layer2 = scipy.sparse.csr_matrix(
        (np.ones(len(rows_l2), dtype=np.float32), (rows_l2, cols_l2)),
        shape=(len(pathway_names), len(process_names)),
    )
    layer2.data[:] = 1.0

    # --- Layer 3: Biological Process -> Adverse Outcome ---
    aop_hierarchy_path = aop_dir / "aop_hierarchy.json"
    outcome_set: set[str] = set()
    process_to_outcome: dict[str, list[str]] = {}

    if aop_hierarchy_path.exists():
        aop_data = parse_aop_wiki_json(aop_hierarchy_path)
        ke_to_ao = aop_data["ke_to_ao"]
        key_events = aop_data["key_events"]

        for ke_id, ao_ids in ke_to_ao.items():
            ke_info = key_events.get(ke_id, {})
            ke_title = ke_info.get("title", ke_id)
            for ao_id in ao_ids:
                ao_info = key_events.get(ao_id, {})
                ao_title = ao_info.get("title", ao_id)
                outcome_set.add(ao_title)
                process_to_outcome.setdefault(ke_title, []).append(ao_title)
                # Also try mapping process IDs (Reactome IDs or KE IDs)
                process_to_outcome.setdefault(ke_id, []).append(ao_title)
    else:
        logger.warning(
            "AOP-Wiki hierarchy not found; Layer 3 will be empty"
        )

    outcome_names = sorted(outcome_set) if outcome_set else ["Unknown_AO"]
    outcome_idx = {o: i for i, o in enumerate(outcome_names)}

    rows_l3: list[int] = []
    cols_l3: list[int] = []

    for proc, outcomes in process_to_outcome.items():
        if proc not in process_idx:
            continue
        proc_i = process_idx[proc]
        for outcome in outcomes:
            if outcome in outcome_idx:
                rows_l3.append(proc_i)
                cols_l3.append(outcome_idx[outcome])

    layer3 = scipy.sparse.csr_matrix(
        (
            np.ones(len(rows_l3), dtype=np.float32) if rows_l3 else np.array([], dtype=np.float32),
            (rows_l3 if rows_l3 else [], cols_l3 if cols_l3 else []),
        ),
        shape=(len(process_names), len(outcome_names)),
    )
    if layer3.data.size > 0:
        layer3.data[:] = 1.0

    logger.info(
        f"Hierarchy graph built: "
        f"L1 {layer1.shape} ({layer1.nnz} edges), "
        f"L2 {layer2.shape} ({layer2.nnz} edges), "
        f"L3 {layer3.shape} ({layer3.nnz} edges)"
    )

    return {
        "adjacency_matrices": [layer1, layer2, layer3],
        "layer_names": ["Gene->Pathway", "Pathway->Process", "Process->Outcome"],
        "gene_names": all_genes,
        "pathway_names": pathway_names,
        "process_names": process_names,
        "outcome_names": outcome_names,
    }


# ---------------------------------------------------------------------------
# Ortholog alignment for cross-species transfer learning
# ---------------------------------------------------------------------------


def build_ortholog_alignment(
    ortholog_path: str | Path,
    expression_datasets: dict[str, pd.DataFrame],
) -> dict:
    """Build ortholog-aligned gene indices for cross-species transfer learning.

    Given ortholog mappings and multiple per-species expression datasets,
    maps genes across species to shared ortholog groups and creates aligned
    gene embedding indices so ToxiGene can transfer-learn across species.

    Parameters
    ----------
    ortholog_path:
        Path to ortholog_mappings.json produced by ``download_ortholog_mappings``.
    expression_datasets:
        Dict mapping species/dataset name to expression DataFrame (genes x samples).

    Returns
    -------
    Dict with:
    - ortholog_to_index: {ortholog_group_id: int}
    - per_species_gene_to_ortholog: {species: {gene_symbol: ortholog_group_id}}
    - alignment_matrices: {species: sparse matrix mapping species genes to ortholog space}
    - coverage_stats: per-species coverage statistics
    """
    ortholog_path = Path(ortholog_path)
    with open(ortholog_path, "r", encoding="utf-8") as f:
        ortholog_groups = json.load(f)

    # Collect all species present in expression datasets
    dataset_species = set(expression_datasets.keys())

    # Build per-species gene -> ortholog group mappings
    per_species_gene_to_orth: dict[str, dict[str, str]] = {sp: {} for sp in dataset_species}

    for group_id, members in ortholog_groups.items():
        for species_key, genes in members.items():
            # Match species key to dataset keys (flexible matching)
            matched_dataset = None
            for ds_name in dataset_species:
                if (species_key.lower() in ds_name.lower()
                        or ds_name.lower() in species_key.lower()):
                    matched_dataset = ds_name
                    break

            if matched_dataset is None:
                continue

            for gene in genes:
                # Map gene symbol to ortholog group
                per_species_gene_to_orth[matched_dataset][gene] = group_id
                # Also try uppercase for matching
                per_species_gene_to_orth[matched_dataset][gene.upper()] = group_id

    # Build ortholog index (only groups that appear in at least one dataset)
    used_groups: set[str] = set()
    for sp_mapping in per_species_gene_to_orth.values():
        used_groups.update(sp_mapping.values())

    ortholog_to_index = {gid: i for i, gid in enumerate(sorted(used_groups))}
    n_orthologs = len(ortholog_to_index)

    # Build per-species alignment matrices (species_genes x ortholog_space)
    alignment_matrices: dict[str, scipy.sparse.csr_matrix] = {}
    coverage_stats: dict[str, dict[str, Any]] = {}

    for sp_name, expr_df in expression_datasets.items():
        sp_genes = list(expr_df.index)
        sp_mapping = per_species_gene_to_orth.get(sp_name, {})

        rows: list[int] = []
        cols: list[int] = []
        mapped_count = 0

        for gene_i, gene in enumerate(sp_genes):
            orth_group = sp_mapping.get(gene) or sp_mapping.get(gene.upper())
            if orth_group and orth_group in ortholog_to_index:
                rows.append(gene_i)
                cols.append(ortholog_to_index[orth_group])
                mapped_count += 1

        mat = scipy.sparse.csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)),
            shape=(len(sp_genes), n_orthologs),
        )
        alignment_matrices[sp_name] = mat

        coverage_stats[sp_name] = {
            "total_genes": len(sp_genes),
            "mapped_genes": mapped_count,
            "coverage_pct": round(100.0 * mapped_count / max(len(sp_genes), 1), 1),
        }

        logger.info(
            f"Ortholog alignment [{sp_name}]: "
            f"{mapped_count}/{len(sp_genes)} genes mapped "
            f"({coverage_stats[sp_name]['coverage_pct']}%)"
        )

    return {
        "ortholog_to_index": ortholog_to_index,
        "per_species_gene_to_ortholog": per_species_gene_to_orth,
        "alignment_matrices": alignment_matrices,
        "coverage_stats": coverage_stats,
    }


# ---------------------------------------------------------------------------
# Full ToxiGene input preparation pipeline
# ---------------------------------------------------------------------------


def prepare_toxigene_inputs(
    expression_dir: str | Path,
    hierarchy_dir: str | Path,
    ortholog_path: str | Path,
    output_dir: str | Path,
) -> dict:
    """Full pipeline to prepare ToxiGene-ready training data.

    Combines:
    1. Batch-corrected expression data (from ``preprocess_and_integrate``)
    2. Hierarchy graph (from ``build_hierarchy_graph``)
    3. Ortholog alignment (from ``build_ortholog_alignment``)

    Parameters
    ----------
    expression_dir:
        Directory containing integrated expression parquet files
        (output of ``preprocess_and_integrate``).
    hierarchy_dir:
        Directory containing Reactome and AOP-Wiki downloaded data.
        Expected subdirectories: ``reactome/`` and ``aop_wiki/``.
    ortholog_path:
        Path to ortholog_mappings.json.
    output_dir:
        Output directory for ToxiGene-ready files.

    Returns
    -------
    Dict with paths to all output files.
    """
    expression_dir = Path(expression_dir)
    hierarchy_dir = Path(hierarchy_dir)
    ortholog_path = Path(ortholog_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths: dict[str, str] = {}

    # 1. Load batch-corrected expression data
    logger.info("Loading integrated expression data")
    panel_path = expression_dir / "integrated_panel.parquet"
    full_path = expression_dir / "integrated_full.parquet"

    if panel_path.exists():
        expression = pd.read_parquet(panel_path)
        logger.info(f"Loaded panel expression: {expression.shape}")
    elif full_path.exists():
        expression = pd.read_parquet(full_path)
        expression = subset_to_panel(expression)
        logger.info(f"Loaded full expression, subsetted to panel: {expression.shape}")
    else:
        raise FileNotFoundError(
            f"No integrated expression data found in {expression_dir}. "
            "Run preprocess_and_integrate first."
        )

    # Save expression matrix as numpy for ToxiGene
    expr_np_path = output_dir / "expression_matrix.npy"
    np.save(expr_np_path, expression.values)
    output_paths["expression_matrix"] = str(expr_np_path)

    # Save gene and sample names
    names_path = output_dir / "expression_metadata.json"
    with open(names_path, "w", encoding="utf-8") as f:
        json.dump({
            "gene_names": list(expression.index),
            "sample_names": list(expression.columns),
            "shape": list(expression.shape),
        }, f, indent=2)
    output_paths["expression_metadata"] = str(names_path)

    # 2. Build hierarchy graph
    logger.info("Building hierarchy graph")
    reactome_dir = hierarchy_dir / "reactome"
    aop_dir = hierarchy_dir / "aop_wiki"

    if not reactome_dir.exists():
        logger.warning(f"Reactome directory not found: {reactome_dir}")
        reactome_dir.mkdir(parents=True, exist_ok=True)
    if not aop_dir.exists():
        logger.warning(f"AOP-Wiki directory not found: {aop_dir}")
        aop_dir.mkdir(parents=True, exist_ok=True)

    hierarchy = build_hierarchy_graph(reactome_dir, aop_dir)

    # Save adjacency matrices as .npz
    for i, (matrix, layer_name) in enumerate(
        zip(hierarchy["adjacency_matrices"], hierarchy["layer_names"])
    ):
        npz_path = output_dir / f"hierarchy_layer{i}_{layer_name.replace('->', '_to_').replace(' ', '_')}.npz"
        scipy.sparse.save_npz(npz_path, matrix)
        output_paths[f"hierarchy_layer{i}"] = str(npz_path)

    # Save hierarchy metadata
    hierarchy_meta_path = output_dir / "hierarchy_metadata.json"
    with open(hierarchy_meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "layer_names": hierarchy["layer_names"],
            "gene_names": hierarchy["gene_names"],
            "pathway_names": hierarchy["pathway_names"],
            "process_names": hierarchy["process_names"],
            "outcome_names": hierarchy["outcome_names"],
            "layer_shapes": [
                list(m.shape) for m in hierarchy["adjacency_matrices"]
            ],
            "layer_nnz": [
                int(m.nnz) for m in hierarchy["adjacency_matrices"]
            ],
        }, f, indent=2)
    output_paths["hierarchy_metadata"] = str(hierarchy_meta_path)

    # 3. Build ortholog alignment
    logger.info("Building ortholog alignment")
    if ortholog_path.exists():
        # For alignment, we need per-species expression datasets
        # Look for species-specific parquet files in expression_dir
        species_datasets: dict[str, pd.DataFrame] = {}

        for parquet_file in expression_dir.glob("*.parquet"):
            ds_name = parquet_file.stem
            if ds_name in ("integrated_full", "integrated_panel"):
                continue
            try:
                ds = pd.read_parquet(parquet_file)
                species_datasets[ds_name] = ds
            except Exception as exc:
                logger.debug(f"Could not load {parquet_file}: {exc}")

        # If no species-specific datasets, use the integrated one as a single dataset
        if not species_datasets:
            species_datasets["integrated"] = expression

        alignment = build_ortholog_alignment(ortholog_path, species_datasets)

        # Save alignment matrices
        for sp_name, mat in alignment["alignment_matrices"].items():
            safe_name = sp_name.replace(" ", "_").replace("/", "_")
            align_path = output_dir / f"ortholog_alignment_{safe_name}.npz"
            scipy.sparse.save_npz(align_path, mat)
            output_paths[f"ortholog_alignment_{safe_name}"] = str(align_path)

        # Save alignment metadata
        align_meta_path = output_dir / "ortholog_metadata.json"
        with open(align_meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "ortholog_to_index": alignment["ortholog_to_index"],
                "coverage_stats": alignment["coverage_stats"],
                "per_species_gene_to_ortholog": alignment["per_species_gene_to_ortholog"],
            }, f, indent=2)
        output_paths["ortholog_metadata"] = str(align_meta_path)

    else:
        logger.warning(
            f"Ortholog mappings not found at {ortholog_path}; "
            "skipping alignment step"
        )

    # Save master output manifest
    manifest_path = output_dir / "toxigene_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(output_paths, f, indent=2)
    output_paths["manifest"] = str(manifest_path)

    logger.info(
        f"ToxiGene inputs prepared: {len(output_paths)} files -> {output_dir}"
    )
    return output_paths
