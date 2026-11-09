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


# ---------------------------------------------------------------------------
# MicroBiomeNet input preparation
# ---------------------------------------------------------------------------


def extract_representative_sequences(
    rep_seqs_artifact_or_fasta: str | Path,
    output_dir: str | Path,
) -> dict[str, str]:
    """Extract DNA sequences for each ASV from a QIIME 2 artifact or FASTA.

    These sequences are intended for DNABERT-S phylogenetic-aware encoding
    in MicroBiomeNet.

    Parameters
    ----------
    rep_seqs_artifact_or_fasta:
        Path to a QIIME 2 rep-seqs artifact (``.qza``) or a plain FASTA file.
    output_dir:
        Directory where the output JSON will be written.

    Returns
    -------
    Dict mapping ASV ID to its DNA sequence string.
    """
    rep_seqs_artifact_or_fasta = Path(rep_seqs_artifact_or_fasta)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fasta_path: Path

    # If .qza, export to FASTA first
    if rep_seqs_artifact_or_fasta.suffix == ".qza":
        export_dir = output_dir / "_qza_export"
        export_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            "qiime", "tools", "export",
            "--input-path", str(rep_seqs_artifact_or_fasta),
            "--output-path", str(export_dir),
        ]
        logger.info(f"Exporting .qza to FASTA: {rep_seqs_artifact_or_fasta.name}")
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
        except FileNotFoundError:
            logger.error(
                "QIIME 2 not found. Install via: conda install -c qiime2 qiime2"
            )
            raise
        except subprocess.CalledProcessError as exc:
            logger.error(f"QIIME 2 export failed: {exc.stderr}")
            raise

        # qiime tools export writes dna-sequences.fasta
        fasta_path = export_dir / "dna-sequences.fasta"
        if not fasta_path.exists():
            # Try alternative naming
            fasta_candidates = list(export_dir.glob("*.fasta")) + list(
                export_dir.glob("*.fa")
            )
            if not fasta_candidates:
                raise FileNotFoundError(
                    f"No FASTA file found after exporting {rep_seqs_artifact_or_fasta}"
                )
            fasta_path = fasta_candidates[0]
    else:
        fasta_path = rep_seqs_artifact_or_fasta

    # Parse FASTA
    sequences: dict[str, str] = {}
    current_id: str | None = None
    current_seq: list[str] = []

    with open(fasta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id is not None:
                    sequences[current_id] = "".join(current_seq)
                # Take the first whitespace-delimited token as the ID
                current_id = line[1:].split()[0]
                current_seq = []
            elif current_id is not None:
                current_seq.append(line)
        # Last record
        if current_id is not None:
            sequences[current_id] = "".join(current_seq)

    # Save as JSON
    json_path = output_dir / "representative_sequences.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sequences, f, indent=2)

    logger.info(
        f"Extracted {len(sequences)} representative sequences -> {json_path}"
    )
    return sequences


def annotate_zero_inflation(
    abundance_table: pd.DataFrame,
    min_prevalence: float = 0.1,
) -> pd.DataFrame:
    """Classify zeros in an ASV abundance table as structural or sampling.

    For each zero entry, determines whether the zero is:
    - ``structural``: the taxon is truly absent (prevalence < *min_prevalence*
      across all samples)
    - ``sampling``: the taxon is present in enough samples to expect detection,
      but was below detection limit in this particular sample

    This annotation is critical for MicroBiomeNet's zero-inflation gating
    mechanism.

    Parameters
    ----------
    abundance_table:
        ASV table of shape ``(n_samples, n_features)`` with raw counts.
    min_prevalence:
        Prevalence threshold (fraction of samples a taxon must appear in
        to be considered truly present in the community).

    Returns
    -------
    DataFrame of the same shape with string values: ``"observed"``,
    ``"structural_zero"``, or ``"sampling_zero"``.
    """
    n_samples = abundance_table.shape[0]

    # Prevalence: fraction of samples where each taxon has count > 0
    prevalence = (abundance_table > 0).sum(axis=0) / n_samples

    # Build annotation matrix
    annotations = pd.DataFrame(
        "observed",
        index=abundance_table.index,
        columns=abundance_table.columns,
    )

    # Mask of zeros
    is_zero = abundance_table == 0

    # Structural zeros: taxon prevalence < min_prevalence
    structural_taxa = prevalence[prevalence < min_prevalence].index
    annotations.loc[:, structural_taxa] = annotations.loc[:, structural_taxa].where(
        ~is_zero[structural_taxa], other="structural_zero"
    )

    # Sampling zeros: taxon prevalence >= min_prevalence but zero in this sample
    sampling_taxa = prevalence[prevalence >= min_prevalence].index
    annotations.loc[:, sampling_taxa] = annotations.loc[:, sampling_taxa].where(
        ~is_zero[sampling_taxa], other="sampling_zero"
    )

    n_structural = (annotations == "structural_zero").sum().sum()
    n_sampling = (annotations == "sampling_zero").sum().sum()
    n_observed = (annotations == "observed").sum().sum()

    logger.info(
        f"Zero-inflation annotation: {n_observed} observed, "
        f"{n_structural} structural zeros, {n_sampling} sampling zeros"
    )

    return annotations


def export_simplex_format(
    abundance_table: pd.DataFrame | np.ndarray,
    output_path: str | Path,
) -> np.ndarray:
    """Convert raw counts to proportions on the simplex.

    Unlike CLR, this keeps data as relative abundances summing to 1.0 per
    sample, which is the native format for MicroBiomeNet's neural ODE
    simplex operations.

    Parameters
    ----------
    abundance_table:
        Raw count matrix of shape ``(n_samples, n_features)``.
    output_path:
        Path (without extension) for output files. Writes both a ``.npy``
        array and a ``_features.json`` with feature names.

    Returns
    -------
    Proportions array of shape ``(n_samples, n_features)``.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(abundance_table, pd.DataFrame):
        feature_names = list(abundance_table.columns)
        values = abundance_table.values.astype(np.float64)
    else:
        feature_names = [f"feature_{i}" for i in range(abundance_table.shape[1])]
        values = abundance_table.astype(np.float64)

    # Compute per-sample proportions (avoid division by zero)
    row_sums = values.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    proportions = values / row_sums

    # Verify simplex constraint
    assert np.allclose(
        proportions.sum(axis=1), 1.0, atol=1e-10
    ), "Proportions do not sum to 1.0 for all samples"

    proportions = proportions.astype(np.float32)

    # Save array
    npy_path = output_path.with_suffix(".npy")
    np.save(npy_path, proportions)

    # Save feature names
    features_path = output_path.with_name(output_path.stem + "_features.json")
    with open(features_path, "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)

    logger.info(
        f"Simplex export: {proportions.shape[0]} samples x "
        f"{proportions.shape[1]} features -> {npy_path}"
    )

    return proportions


def extract_temporal_metadata(
    metadata: pd.DataFrame,
    timestamp_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Parse and standardize temporal metadata for MicroBiomeNet's ODE modeling.

    Scans metadata columns for date/time information and converts to Unix
    timestamps, which are required for MicroBiomeNet's simplex ODE temporal
    trajectory modeling.

    Parameters
    ----------
    metadata:
        Sample metadata DataFrame (index = sample IDs).
    timestamp_cols:
        Explicit column names containing timestamps. If *None*, the function
        auto-detects columns by searching for common names like
        ``collection_date``, ``collection_timestamp``, ``date``, etc.

    Returns
    -------
    DataFrame indexed by sample ID with columns:
    - ``timestamp_unix``: Unix timestamp (seconds since epoch)
    - ``timestamp_iso``: ISO 8601 formatted string
    - ``source_column``: which metadata column the timestamp came from
    """
    CANDIDATE_COLS = [
        "collection_date",
        "collection_timestamp",
        "date",
        "sample_date",
        "sampling_date",
        "datetime",
        "collection_time",
        "date_collected",
        "event_date",
        "eventDate",
        "collectionDate",
    ]

    if timestamp_cols is None:
        # Auto-detect: check for known column names (case-insensitive)
        col_lower_map = {c.lower(): c for c in metadata.columns}
        timestamp_cols = []
        for candidate in CANDIDATE_COLS:
            if candidate.lower() in col_lower_map:
                timestamp_cols.append(col_lower_map[candidate.lower()])
        if not timestamp_cols:
            # Fallback: try columns with "date" or "time" in the name
            timestamp_cols = [
                c for c in metadata.columns
                if "date" in c.lower() or "time" in c.lower()
            ]

    if not timestamp_cols:
        logger.warning(
            "No timestamp columns found in metadata. "
            "Provide explicit column names via timestamp_cols parameter."
        )
        return pd.DataFrame(
            columns=["timestamp_unix", "timestamp_iso", "source_column"],
            index=metadata.index,
        )

    logger.info(f"Detected timestamp columns: {timestamp_cols}")

    # Try each candidate column until we get valid parses
    best_col: str | None = None
    best_parsed: pd.Series | None = None
    best_valid_count = 0

    for col in timestamp_cols:
        try:
            parsed = pd.to_datetime(metadata[col], errors="coerce", utc=True)
            valid_count = parsed.notna().sum()
            if valid_count > best_valid_count:
                best_col = col
                best_parsed = parsed
                best_valid_count = valid_count
        except Exception:
            continue

    if best_parsed is None or best_valid_count == 0:
        logger.warning("Could not parse any timestamp columns")
        return pd.DataFrame(
            columns=["timestamp_unix", "timestamp_iso", "source_column"],
            index=metadata.index,
        )

    logger.info(
        f"Using column '{best_col}': {best_valid_count}/{len(metadata)} "
        f"valid timestamps"
    )

    result = pd.DataFrame(index=metadata.index)
    result["timestamp_unix"] = (
        best_parsed.astype("int64") // 10**9
    ).astype("Int64")  # nullable integer for NaT
    result["timestamp_iso"] = best_parsed.dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    result["timestamp_iso"] = result["timestamp_iso"].where(best_parsed.notna(), other=None)
    result["source_column"] = best_col

    return result


def prepare_microbiomenet_inputs(
    biom_path: str | Path,
    metadata_path: str | Path,
    output_dir: str | Path,
    *,
    rep_seqs_path: str | Path | None = None,
    pseudocount: float = DEFAULT_PSEUDOCOUNT,
    min_abundance: float = DEFAULT_MIN_ABUNDANCE,
    max_features: int = DEFAULT_MAX_FEATURES,
    min_prevalence: float = 0.1,
) -> dict:
    """Full pipeline producing all input formats needed by MicroBiomeNet.

    Generates:
      - CLR-transformed matrix (via existing :func:`preprocess_biom`)
      - Raw simplex proportions (:func:`export_simplex_format`)
      - Representative DNA sequences (:func:`extract_representative_sequences`)
      - Zero-inflation annotations (:func:`annotate_zero_inflation`)
      - Temporal metadata (:func:`extract_temporal_metadata`)

    Parameters
    ----------
    biom_path:
        Path to BIOM file (JSON or HDF5 format).
    metadata_path:
        Path to sample metadata TSV.
    output_dir:
        Root output directory for all generated files.
    rep_seqs_path:
        Path to representative sequences (``.qza`` or FASTA). If *None*,
        the DNA sequence extraction step is skipped.
    pseudocount:
        CLR pseudocount.
    min_abundance:
        Minimum relative abundance threshold.
    max_features:
        Maximum ASV features.
    min_prevalence:
        Prevalence threshold for zero-inflation annotation.

    Returns
    -------
    Dict with paths to all output files:
      - ``clr_matrix``: path to CLR .npy file
      - ``simplex``: path to simplex proportions .npy file
      - ``rep_seqs``: path to representative sequences JSON (or None)
      - ``zero_inflation``: path to zero-inflation annotation parquet
      - ``temporal_metadata``: path to temporal metadata parquet
      - ``feature_info``: path to feature info JSON
    """
    from biom import load_table

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Run existing CLR pipeline ---
    logger.info("Step 1/5: CLR transformation via preprocess_biom")
    clr_result = preprocess_biom(
        biom_path,
        output_dir / "clr",
        metadata_path=metadata_path,
        pseudocount=pseudocount,
        min_abundance=min_abundance,
        max_features=max_features,
    )

    # --- 2. Load raw table for simplex + zero-inflation ---
    logger.info("Step 2/5: Simplex proportions export")
    table = load_table(str(biom_path))
    raw_df = pd.DataFrame(
        table.to_dataframe().T.values,
        index=table.ids(axis="sample"),
        columns=table.ids(axis="observation"),
    )
    # Filter to same features as CLR output
    feature_names = clr_result["feature_names"]
    available_features = [f for f in feature_names if f in raw_df.columns]
    raw_filtered = raw_df[available_features]

    simplex_path = output_dir / "simplex" / "proportions"
    export_simplex_format(raw_filtered, simplex_path)

    # --- 3. Zero-inflation annotation ---
    logger.info("Step 3/5: Zero-inflation annotation")
    zero_annot = annotate_zero_inflation(raw_filtered, min_prevalence=min_prevalence)
    zero_annot_path = output_dir / "zero_inflation_annotations.parquet"
    zero_annot.to_parquet(zero_annot_path)

    # --- 4. Representative sequences ---
    rep_seqs_output: str | None = None
    if rep_seqs_path is not None:
        logger.info("Step 4/5: Representative sequence extraction")
        seqs = extract_representative_sequences(
            rep_seqs_path, output_dir / "rep_seqs"
        )
        rep_seqs_output = str(output_dir / "rep_seqs" / "representative_sequences.json")
    else:
        logger.info("Step 4/5: Skipping representative sequences (no path provided)")

    # --- 5. Temporal metadata ---
    logger.info("Step 5/5: Temporal metadata extraction")
    metadata = pd.read_csv(metadata_path, sep="\t", index_col=0)
    temporal = extract_temporal_metadata(metadata)
    temporal_path = output_dir / "temporal_metadata.parquet"
    temporal.to_parquet(temporal_path)

    outputs = {
        "clr_matrix": str(output_dir / "clr" / "clr_matrix.npy"),
        "simplex": str(simplex_path.with_suffix(".npy")),
        "rep_seqs": rep_seqs_output,
        "zero_inflation": str(zero_annot_path),
        "temporal_metadata": str(temporal_path),
        "feature_info": str(output_dir / "clr" / "feature_info.json"),
    }

    # Save manifest
    manifest_path = output_dir / "microbiomenet_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2)

    logger.info(f"MicroBiomeNet inputs ready -> {manifest_path}")
    return outputs
