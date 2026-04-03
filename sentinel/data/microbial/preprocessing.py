"""
Microbial community data preprocessing for SENTINEL.

Pipeline stages:
  1. Quality filtering (DADA2/Deblur-compatible ASV inference)
  2. Chimera removal (UCHIME reference-based)
  3. Taxonomic classification (SILVA 138 via naive Bayes)
  4. CLR (Centered Log-Ratio) transformation
  5. Environmental metadata linkage
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from sentinel.utils.config import load_config
from sentinel.utils.logging import get_logger, make_progress

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_PSEUDOCOUNT = 0.5
DEFAULT_MIN_ABUNDANCE = 0.001
DEFAULT_MAX_FEATURES = 5000

# SILVA 138 classifier paths (user must supply these)
SILVA_CLASSIFIER_PATH = "data/microbial/classifiers/silva-138-99-nb-classifier.qza"
UCHIME_REF_DB = "data/microbial/classifiers/silva-138-99-seqs.qza"


# ---------------------------------------------------------------------------
# Quality filtering
# ---------------------------------------------------------------------------


@dataclass
class QualityFilterParams:
    """Parameters for amplicon sequence quality filtering."""

    min_quality: int = 20
    min_length: int = 100
    max_ee: float = 2.0  # maximum expected errors (DADA2)
    trunc_len_f: int = 150  # forward read truncation length
    trunc_len_r: int = 150  # reverse read truncation length
    chimera_method: str = "consensus"  # DADA2 chimera method
    n_threads: int = 4


def run_dada2_denoise(
    input_dir: str | Path,
    output_dir: str | Path,
    params: QualityFilterParams | None = None,
) -> Path:
    """Run DADA2 denoising via QIIME 2 CLI.

    Expects demultiplexed paired-end reads imported as a QIIME 2 artifact.

    Parameters
    ----------
    input_dir:
        Directory containing the imported QIIME 2 artifact (.qza).
    output_dir:
        Directory for output artifacts (table.qza, rep-seqs.qza, stats.qza).
    params:
        Quality filtering parameters.

    Returns
    -------
    Path to the output ASV table artifact.
    """
    params = params or QualityFilterParams()
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find the demultiplexed reads artifact
    input_artifact = _find_artifact(input_dir, suffix="-demux")

    table_path = output_dir / "table.qza"
    rep_seqs_path = output_dir / "rep-seqs.qza"
    stats_path = output_dir / "stats.qza"

    cmd = [
        "qiime", "dada2", "denoise-paired",
        "--i-demultiplexed-seqs", str(input_artifact),
        "--p-trunc-len-f", str(params.trunc_len_f),
        "--p-trunc-len-r", str(params.trunc_len_r),
        "--p-max-ee-f", str(params.max_ee),
        "--p-max-ee-r", str(params.max_ee),
        "--p-chimera-method", params.chimera_method,
        "--p-n-threads", str(params.n_threads),
        "--o-table", str(table_path),
        "--o-representative-sequences", str(rep_seqs_path),
        "--o-denoising-stats", str(stats_path),
        "--verbose",
    ]

    logger.info(f"Running DADA2 denoising: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info("DADA2 denoising completed successfully")
    except FileNotFoundError:
        logger.error("QIIME 2 not found. Install via: conda install -c qiime2 qiime2")
        raise
    except subprocess.CalledProcessError as exc:
        logger.error(f"DADA2 denoising failed: {exc.stderr}")
        raise

    return table_path


def run_deblur_denoise(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    trim_length: int = 150,
) -> Path:
    """Run Deblur denoising via QIIME 2 CLI (alternative to DADA2).

    Expects quality-filtered single-end reads.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_artifact = _find_artifact(input_dir, suffix="-filtered")
    table_path = output_dir / "table-deblur.qza"
    rep_seqs_path = output_dir / "rep-seqs-deblur.qza"
    stats_path = output_dir / "stats-deblur.qza"

    cmd = [
        "qiime", "deblur", "denoise-16S",
        "--i-demultiplexed-seqs", str(input_artifact),
        "--p-trim-length", str(trim_length),
        "--o-table", str(table_path),
        "--o-representative-sequences", str(rep_seqs_path),
        "--o-stats", str(stats_path),
        "--verbose",
    ]

    logger.info(f"Running Deblur denoising: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info("Deblur denoising completed successfully")
    except subprocess.CalledProcessError as exc:
        logger.error(f"Deblur denoising failed: {exc.stderr}")
        raise

    return table_path


# ---------------------------------------------------------------------------
# Chimera removal (standalone UCHIME)
# ---------------------------------------------------------------------------


def remove_chimeras_uchime(
    rep_seqs_path: str | Path,
    output_dir: str | Path,
    *,
    ref_db: str | Path = UCHIME_REF_DB,
) -> Path:
    """Reference-based chimera removal using vsearch (UCHIME algorithm).

    Parameters
    ----------
    rep_seqs_path:
        Path to representative sequences QIIME 2 artifact.
    output_dir:
        Output directory for filtered sequences.
    ref_db:
        Path to reference database for chimera detection.

    Returns
    -------
    Path to chimera-filtered sequences artifact.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nonchimeras_path = output_dir / "rep-seqs-nonchimeric.qza"
    chimeras_path = output_dir / "chimeras.qza"
    stats_path = output_dir / "chimera-stats.qza"

    cmd = [
        "qiime", "vsearch", "uchime-ref",
        "--i-sequences", str(rep_seqs_path),
        "--i-table", str(output_dir.parent / "table.qza"),
        "--i-reference-sequences", str(ref_db),
        "--o-nonchimeras", str(nonchimeras_path),
        "--o-chimeras", str(chimeras_path),
        "--o-stats", str(stats_path),
        "--verbose",
    ]

    logger.info("Running UCHIME reference-based chimera removal")
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info("Chimera removal completed")
    except subprocess.CalledProcessError as exc:
        logger.error(f"Chimera removal failed: {exc.stderr}")
        raise

    return nonchimeras_path


# ---------------------------------------------------------------------------
# Taxonomic classification
# ---------------------------------------------------------------------------


def classify_taxonomy(
    rep_seqs_path: str | Path,
    output_dir: str | Path,
    *,
    classifier_path: str | Path = SILVA_CLASSIFIER_PATH,
    n_jobs: int = 4,
) -> Path:
    """Classify ASV representative sequences using SILVA 138 naive Bayes.

    Parameters
    ----------
    rep_seqs_path:
        Path to representative sequences artifact.
    output_dir:
        Output directory.
    classifier_path:
        Path to pre-trained SILVA 138 classifier.
    n_jobs:
        Number of parallel jobs.

    Returns
    -------
    Path to taxonomy artifact.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    taxonomy_path = output_dir / "taxonomy.qza"

    cmd = [
        "qiime", "feature-classifier", "classify-sklearn",
        "--i-classifier", str(classifier_path),
        "--i-reads", str(rep_seqs_path),
        "--p-n-jobs", str(n_jobs),
        "--o-classification", str(taxonomy_path),
        "--verbose",
    ]

    logger.info("Running taxonomic classification with SILVA 138")
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info("Taxonomic classification completed")
    except subprocess.CalledProcessError as exc:
        logger.error(f"Taxonomy classification failed: {exc.stderr}")
        raise

    return taxonomy_path


# ---------------------------------------------------------------------------
# CLR transformation
# ---------------------------------------------------------------------------


def clr_transform(
    abundance_matrix: np.ndarray | pd.DataFrame,
    *,
    pseudocount: float = DEFAULT_PSEUDOCOUNT,
) -> np.ndarray:
    """Centered Log-Ratio (CLR) transformation.

    clr(x_i) = log(x_i / geometric_mean(x))

    A pseudocount is added to handle zeros before log transformation.

    Parameters
    ----------
    abundance_matrix:
        Array of shape ``(n_samples, n_features)`` with raw counts or
        relative abundances.
    pseudocount:
        Value added to all entries before log transform.

    Returns
    -------
    CLR-transformed array of the same shape.
    """
    if isinstance(abundance_matrix, pd.DataFrame):
        abundance_matrix = abundance_matrix.values

    x = abundance_matrix.astype(np.float64) + pseudocount

    # Geometric mean per sample (row)
    log_x = np.log(x)
    log_geom_mean = log_x.mean(axis=1, keepdims=True)

    clr = log_x - log_geom_mean
    return clr.astype(np.float32)


# ---------------------------------------------------------------------------
# Feature filtering
# ---------------------------------------------------------------------------


def filter_low_abundance(
    table: pd.DataFrame,
    *,
    min_relative_abundance: float = DEFAULT_MIN_ABUNDANCE,
    max_features: int = DEFAULT_MAX_FEATURES,
) -> pd.DataFrame:
    """Filter ASVs by minimum relative abundance and cap total features.

    Parameters
    ----------
    table:
        ASV table (samples x features).
    min_relative_abundance:
        Minimum mean relative abundance across samples.
    max_features:
        Maximum number of features to retain (top by mean abundance).

    Returns
    -------
    Filtered table.
    """
    # Convert to relative abundance
    row_sums = table.sum(axis=1).replace(0, 1)
    rel_table = table.div(row_sums, axis=0)

    # Filter by minimum mean abundance
    mean_abundance = rel_table.mean(axis=0)
    keep = mean_abundance[mean_abundance >= min_relative_abundance].index
    table = table[keep]
    logger.info(
        f"Abundance filter: {len(keep)} features retained "
        f"(min rel. abundance = {min_relative_abundance})"
    )

    # Cap at max features
    if table.shape[1] > max_features:
        top_features = mean_abundance[keep].nlargest(max_features).index
        table = table[top_features]
        logger.info(f"Capped to top {max_features} features")

    return table


# ---------------------------------------------------------------------------
# Metadata linkage
# ---------------------------------------------------------------------------


def link_environmental_metadata(
    asv_table: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    sample_id_col: str = "sample_id",
) -> pd.DataFrame:
    """Join ASV table with environmental metadata on sample IDs.

    Parameters
    ----------
    asv_table:
        CLR-transformed ASV table (samples x features). Index = sample IDs.
    metadata:
        Environmental metadata DataFrame with a sample ID column.
    sample_id_col:
        Column name in metadata containing sample IDs.

    Returns
    -------
    Merged DataFrame with both ASV features and environmental variables.
    """
    if sample_id_col in metadata.columns:
        metadata = metadata.set_index(sample_id_col)

    common = asv_table.index.intersection(metadata.index)
    logger.info(
        f"Linking metadata: {len(common)} samples matched "
        f"(of {len(asv_table)} ASV samples, {len(metadata)} metadata records)"
    )
    merged = asv_table.loc[common].join(metadata.loc[common], rsuffix="_meta")
    return merged


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def preprocess_biom(
    biom_path: str | Path,
    output_dir: str | Path,
    *,
    metadata_path: str | Path | None = None,
    pseudocount: float = DEFAULT_PSEUDOCOUNT,
    min_abundance: float = DEFAULT_MIN_ABUNDANCE,
    max_features: int = DEFAULT_MAX_FEATURES,
) -> dict[str, Any]:
    """Preprocess a BIOM table end-to-end (post-denoising).

    Steps: load -> filter -> CLR -> metadata linkage -> save.

    Parameters
    ----------
    biom_path:
        Path to BIOM file (JSON or HDF5 format).
    output_dir:
        Output directory for processed files.
    metadata_path:
        Optional path to sample metadata TSV.
    pseudocount:
        CLR pseudocount.
    min_abundance:
        Minimum relative abundance threshold.
    max_features:
        Maximum ASV features.

    Returns
    -------
    Dict with ``"clr_matrix"``, ``"feature_names"``, ``"sample_ids"``.
    """
    from biom import load_table

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading BIOM table from {biom_path}")
    table = load_table(str(biom_path))
    df = pd.DataFrame(
        table.to_dataframe().T.values,
        index=table.ids(axis="sample"),
        columns=table.ids(axis="observation"),
    )
    logger.info(f"Raw table: {df.shape[0]} samples x {df.shape[1]} features")

    # Filter
    df = filter_low_abundance(
        df, min_relative_abundance=min_abundance, max_features=max_features
    )

    # CLR transform
    clr_matrix = clr_transform(df, pseudocount=pseudocount)
    clr_df = pd.DataFrame(clr_matrix, index=df.index, columns=df.columns)

    # Metadata linkage
    if metadata_path:
        metadata = pd.read_csv(metadata_path, sep="\t", index_col=0)
        clr_df = link_environmental_metadata(clr_df, metadata)

    # Save
    clr_df.to_parquet(output_dir / "clr_table.parquet")
    np.save(output_dir / "clr_matrix.npy", clr_matrix)

    feature_info = {
        "feature_names": list(df.columns),
        "sample_ids": list(df.index),
        "n_samples": int(df.shape[0]),
        "n_features": int(df.shape[1]),
        "pseudocount": pseudocount,
    }
    with open(output_dir / "feature_info.json", "w", encoding="utf-8") as f:
        json.dump(feature_info, f, indent=2)

    logger.info(
        f"Preprocessing complete: {clr_matrix.shape[0]} samples x "
        f"{clr_matrix.shape[1]} features -> {output_dir}"
    )

    return {
        "clr_matrix": clr_matrix,
        "feature_names": list(df.columns),
        "sample_ids": list(df.index),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_artifact(directory: Path, suffix: str = "") -> Path:
    """Find a .qza artifact in a directory, optionally matching a suffix."""
    candidates = list(directory.glob("*.qza"))
    if suffix:
        candidates = [c for c in candidates if suffix in c.stem]
    if not candidates:
        raise FileNotFoundError(
            f"No .qza artifact found in {directory} (suffix={suffix!r})"
        )
    return candidates[0]
