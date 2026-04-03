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
