"""
Sensor time-series preprocessing for SENTINEL.

Transforms raw USGS NWIS instantaneous-value data into model-ready tensors:
  - [T, P] where T=672 (7 days at 15-min intervals), P=6 parameters
  - Per-station, per-parameter z-score normalization (rolling 90-day window)
  - Gap filling (forward fill + linear interpolation for short gaps)
  - Quality flag filtering
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
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

PARAMETER_CODES = ["00300", "00400", "00095", "00010", "63680", "00090"]
PARAMETER_NAMES = ["DO", "pH", "SpCond", "Temp", "Turb", "ORP"]
NUM_PARAMS = len(PARAMETER_CODES)

INTERVAL_MINUTES = 15
LOOKBACK_STEPS = 672  # 7 days * 24 h * 4 (15-min intervals)
ROLLING_WINDOW_DAYS = 90

# NWIS qualification codes indicating suspect or rejected data
SUSPECT_QUAL_CODES = {"e", "E", "P", "X", "<", ">", "~", "R"}

# Maximum gap length (in steps) eligible for interpolation
MAX_INTERP_GAP = 12  # 3 hours at 15-min intervals


# ---------------------------------------------------------------------------
# Quality filtering
# ---------------------------------------------------------------------------


def filter_quality_flags(
    df: pd.DataFrame,
    *,
    drop_suspect: bool = True,
) -> pd.DataFrame:
    """Remove or flag rows with NWIS qualification codes indicating
    suspect data.

    NWIS columns ending in ``_cd`` contain qualification codes. A value of
    ``"A"`` (approved) or ``""`` is considered clean. Codes in
    ``SUSPECT_QUAL_CODES`` trigger removal when *drop_suspect* is True.

    Parameters
    ----------
    df:
        DataFrame with NWIS-style columns (e.g., ``00300``, ``00300_cd``).
    drop_suspect:
        If True, set suspect values to NaN rather than removing entire rows.

    Returns
    -------
    Cleaned DataFrame.
    """
    df = df.copy()
    for col in df.columns:
        if not col.endswith("_cd"):
            continue
        value_col = col.replace("_cd", "")
        if value_col not in df.columns:
            continue
        if drop_suspect:
            mask = df[col].astype(str).str.strip().isin(SUSPECT_QUAL_CODES)
            df.loc[mask, value_col] = np.nan
            n_flagged = mask.sum()
            if n_flagged > 0:
                logger.debug(f"Flagged {n_flagged} suspect values in {value_col}")
    return df


# ---------------------------------------------------------------------------
# Resampling and gap filling
# ---------------------------------------------------------------------------


def resample_to_regular(
    df: pd.DataFrame,
    interval: str = "15min",
) -> pd.DataFrame:
    """Resample an irregularly-sampled DataFrame to a fixed interval.

    The index must be a DatetimeIndex.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")

    # Keep only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df = df[numeric_cols]

    # Resample: take the mean of values falling within each bin
    resampled = df.resample(interval).mean()
    return resampled


def fill_gaps(
    series: pd.Series,
    max_gap: int = MAX_INTERP_GAP,
) -> pd.Series:
    """Fill gaps in a single time series.

    Strategy:
      1. Forward-fill up to ``max_gap`` consecutive NaN values.
      2. Linear interpolation for remaining short gaps (<= ``max_gap``).
      3. Longer gaps remain NaN (to be masked during training).
    """
    filled = series.ffill(limit=max_gap)
    filled = filled.interpolate(method="linear", limit=max_gap, limit_direction="both")
    return filled


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


@dataclass
class NormalizationStats:
    """Per-parameter rolling normalization statistics."""

    mean: pd.Series
    std: pd.Series

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean.to_dict(),
            "std": self.std.to_dict(),
        }


def compute_rolling_stats(
    df: pd.DataFrame,
    window_days: int = ROLLING_WINDOW_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute rolling mean and std for each column.

    Parameters
    ----------
    df:
        Resampled DataFrame at 15-min intervals.
    window_days:
        Window size in days for rolling statistics.

    Returns
    -------
    Tuple of (rolling_mean, rolling_std) DataFrames.
    """
    window_steps = window_days * 24 * 4  # 15-min intervals
    rolling_mean = df.rolling(window=window_steps, min_periods=window_steps // 4).mean()
    rolling_std = df.rolling(window=window_steps, min_periods=window_steps // 4).std()
    # Prevent division by zero
    rolling_std = rolling_std.clip(lower=1e-6)
    return rolling_mean, rolling_std


def normalize_zscore(
    df: pd.DataFrame,
    rolling_mean: pd.DataFrame,
    rolling_std: pd.DataFrame,
) -> pd.DataFrame:
    """Apply per-parameter z-score normalization using rolling statistics."""
    return (df - rolling_mean) / rolling_std


# ---------------------------------------------------------------------------
# Tensor extraction
# ---------------------------------------------------------------------------


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename NWIS-style columns to canonical parameter codes.

    NWIS returns columns like ``00300_00003`` (parameter + statistic code).
    We strip the statistic suffix and keep the first match per parameter.
    """
    rename_map: dict[str, str] = {}
    seen: set[str] = set()
    for col in df.columns:
        base = str(col).split("_")[0]
        if base in PARAMETER_CODES and base not in seen:
            rename_map[col] = base
            seen.add(base)
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def extract_windows(
    df: pd.DataFrame,
    window_size: int = LOOKBACK_STEPS,
    stride: int | None = None,
) -> tuple[np.ndarray, list[pd.Timestamp]]:
    """Extract sliding windows from a preprocessed DataFrame.

    Parameters
    ----------
    df:
        Normalized DataFrame with columns matching ``PARAMETER_CODES``.
    window_size:
        Number of timesteps per window.
    stride:
        Step between windows (default: ``window_size // 2``).

    Returns
    -------
    Tuple of:
      - ``windows``: array of shape ``(N, T, P)``
      - ``timestamps``: list of end-of-window timestamps
    """
    stride = stride or window_size // 2
    param_cols = [c for c in PARAMETER_CODES if c in df.columns]
    if not param_cols:
        raise ValueError(f"No parameter columns found in DataFrame. Columns: {list(df.columns)}")

    data = df[param_cols].values.astype(np.float32)
    n_steps = data.shape[0]
    windows: list[np.ndarray] = []
    timestamps: list[pd.Timestamp] = []

    for end in range(window_size, n_steps + 1, stride):
        start = end - window_size
        window = data[start:end]
        # Skip windows with too many NaNs (>20%)
        nan_ratio = np.isnan(window).mean()
        if nan_ratio > 0.20:
            continue
        # Replace remaining NaN with 0 (masked during training)
        window = np.nan_to_num(window, nan=0.0)
        windows.append(window)
        timestamps.append(df.index[end - 1])

    if not windows:
        return np.empty((0, window_size, len(param_cols)), dtype=np.float32), []

    return np.stack(windows), timestamps


# ---------------------------------------------------------------------------
# Full preprocessing pipeline
# ---------------------------------------------------------------------------


def preprocess_station(
    input_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    window_size: int = LOOKBACK_STEPS,
    rolling_window_days: int = ROLLING_WINDOW_DAYS,
) -> dict[str, Any]:
    """Full preprocessing pipeline for a single station.

    Parameters
    ----------
    input_path:
        Path to a Parquet file with raw NWIS data.
    output_dir:
        Directory for output files (defaults to input's parent / processed).
    window_size:
        Number of 15-min timesteps per window.
    rolling_window_days:
        Days for rolling normalization statistics.

    Returns
    -------
    Dict with ``"windows"`` (ndarray), ``"timestamps"``, ``"stats"``,
    ``"n_windows"``, ``"site_no"``.
    """
    input_path = Path(input_path)
    site_no = input_path.stem

    if output_dir is None:
        output_dir = input_path.parent.parent / "processed"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Preprocessing station {site_no}")

    # Load raw data
    df = pd.read_parquet(input_path)

    # Standardize column names
    df = _standardize_columns(df)

    # Quality filtering
    df = filter_quality_flags(df)

    # Resample to regular 15-min intervals
    df = resample_to_regular(df, interval="15min")

    # Gap filling per parameter
    for col in df.columns:
        if col in PARAMETER_CODES:
            df[col] = fill_gaps(df[col])

    # Rolling normalization
    rolling_mean, rolling_std = compute_rolling_stats(df, rolling_window_days)
    df_norm = normalize_zscore(df, rolling_mean, rolling_std)

    # Extract windows
    windows, timestamps = extract_windows(df_norm, window_size=window_size)

    # Save outputs
    if windows.shape[0] > 0:
        np.save(output_dir / f"{site_no}_windows.npy", windows)
        ts_df = pd.DataFrame({"timestamp": timestamps})
        ts_df.to_parquet(output_dir / f"{site_no}_timestamps.parquet")

    logger.info(f"Station {site_no}: {windows.shape[0]} windows of shape {windows.shape[1:]}")

    return {
        "site_no": site_no,
        "windows": windows,
        "timestamps": timestamps,
        "n_windows": windows.shape[0],
        "stats": NormalizationStats(
            mean=rolling_mean.iloc[-1] if len(rolling_mean) > 0 else pd.Series(),
            std=rolling_std.iloc[-1] if len(rolling_std) > 0 else pd.Series(),
        ),
    }


def preprocess_all_stations(
    input_dir: str | Path,
    output_dir: str | Path = "data/sensor/processed",
    *,
    window_size: int = LOOKBACK_STEPS,
) -> dict[str, dict[str, Any]]:
    """Batch preprocess all station Parquet files in a directory.

    Returns a dict mapping site_no -> preprocessing results.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    files = sorted(input_dir.glob("*.parquet"))

    if not files:
        logger.warning(f"No Parquet files found in {input_dir}")
        return {}

    results: dict[str, dict[str, Any]] = {}
    progress = make_progress()
    with progress:
        task = progress.add_task("Preprocessing stations", total=len(files))
        for f in files:
            try:
                result = preprocess_station(
                    f, output_dir=output_dir, window_size=window_size
                )
                results[result["site_no"]] = result
            except Exception as exc:
                logger.warning(f"Failed to preprocess {f.stem}: {exc}")
            progress.advance(task)

    total_windows = sum(r["n_windows"] for r in results.values())
    logger.info(
        f"Preprocessed {len(results)} stations, {total_windows} total windows"
    )
    return results


# ===========================================================================
# Irregular-time preprocessing for AquaSSM (continuous-time state space model)
# ===========================================================================


@dataclass
class IrregularTimeSample:
    """A single observation at an irregular timestamp for AquaSSM.

    AquaSSM handles variable-interval observations natively via a learned
    step-size mechanism.  This dataclass stores the raw observation along
    with the time gap since the previous sample and a quality mask.
    """

    timestamp_seconds: float
    """Unix timestamp (seconds since epoch) of this observation."""

    parameter_values: np.ndarray
    """Shape ``(6,)`` — raw values for DO, pH, SpCond, Temp, Turb, ORP."""

    delta_t_seconds: float
    """Time gap (seconds) since the previous observation."""

    quality_mask: np.ndarray
    """Shape ``(6,)`` — ``1.0`` for valid, ``0.0`` for missing/suspect."""

    normalized_values: np.ndarray
    """Shape ``(6,)`` — rolling z-score normalized values."""


# ---------------------------------------------------------------------------
# Irregular preprocessing pipeline (single station)
# ---------------------------------------------------------------------------


def _rolling_zscore_irregular(
    values: np.ndarray,
    timestamps: np.ndarray,
    window_seconds: float,
    min_samples: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute rolling z-score normalization on irregular observations.

    For each observation *i* the window covers all prior observations whose
    timestamp falls within ``[t_i - window_seconds, t_i]``.

    Parameters
    ----------
    values:
        Shape ``(N, P)`` — raw parameter values (may contain NaN).
    timestamps:
        Shape ``(N,)`` — Unix timestamps in seconds.
    window_seconds:
        Rolling window width in seconds.
    min_samples:
        Minimum number of valid (non-NaN) samples in the window before
        statistics are considered reliable.  If fewer, the expanding
        (all-prior) mean/std is used instead.

    Returns
    -------
    Tuple of ``(normalized, rolling_mean, rolling_std)`` each ``(N, P)``.
    """
    n, p = values.shape
    rolling_mean = np.full_like(values, np.nan)
    rolling_std = np.full_like(values, np.nan)

    for i in range(n):
        t_i = timestamps[i]
        # All observations in the window [t_i - window_seconds, t_i]
        mask = (timestamps[:i + 1] >= t_i - window_seconds)
        window_vals = values[:i + 1][mask]

        for j in range(p):
            col = window_vals[:, j]
            valid = col[~np.isnan(col)]
            if len(valid) >= min_samples:
                rolling_mean[i, j] = np.mean(valid)
                rolling_std[i, j] = max(np.std(valid, ddof=1), 1e-6)
            elif i > 0:
                # Fall back to expanding stats over all prior observations
                all_prior = values[:i + 1, j]
                valid_all = all_prior[~np.isnan(all_prior)]
                if len(valid_all) >= 2:
                    rolling_mean[i, j] = np.mean(valid_all)
                    rolling_std[i, j] = max(np.std(valid_all, ddof=1), 1e-6)

    # Normalize
    normalized = (values - rolling_mean) / np.where(
        np.isnan(rolling_std), 1.0, rolling_std
    )
    # NaN in original values stays NaN in normalized
    normalized = np.where(np.isnan(values), np.nan, normalized)

    return normalized, rolling_mean, rolling_std


def preprocess_station_irregular(
    input_path: str | Path,
    *,
    rolling_window_days: int = ROLLING_WINDOW_DAYS,
    drop_suspect: bool = True,
) -> list[IrregularTimeSample]:
    """Preprocess a station's raw data for AquaSSM irregular-time input.

    Unlike :func:`preprocess_station`, this function does **not** resample
    to regular 15-min intervals.  AquaSSM handles irregular intervals
    natively via a learned step-size mechanism, so we preserve the actual
    observation timestamps and compute :math:`\\Delta t` between consecutive
    observations as an explicit feature.

    Parameters
    ----------
    input_path:
        Path to a Parquet file with raw NWIS data.
    rolling_window_days:
        Days for rolling z-score normalization window.
    drop_suspect:
        If True, set suspect-quality values to NaN (reuses
        :func:`filter_quality_flags`).

    Returns
    -------
    List of :class:`IrregularTimeSample` with original timestamps preserved,
    sorted chronologically.
    """
    input_path = Path(input_path)
    site_no = input_path.stem
    logger.info(f"Irregular preprocessing: station {site_no}")

    # Load and standardize columns
    df = pd.read_parquet(input_path)
    df = _standardize_columns(df)

    # Quality filtering (reuse existing function)
    if drop_suspect:
        df = filter_quality_flags(df, drop_suspect=True)

    # Ensure DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        # Try to parse a datetime column
        for col_name in ("datetime", "dateTime", "Datetime"):
            if col_name in df.columns:
                df.index = pd.to_datetime(df[col_name])
                break
        else:
            raise ValueError(
                f"Station {site_no}: DataFrame must have a DatetimeIndex "
                "or a recognized datetime column."
            )

    df = df.sort_index()

    # Keep only the 6 SENTINEL parameter columns that are present
    param_cols = [c for c in PARAMETER_CODES if c in df.columns]
    if not param_cols:
        logger.warning(
            f"Station {site_no}: no recognized parameter columns found."
        )
        return []

    # Build values array (N, 6) — missing params filled with NaN
    n = len(df)
    values = np.full((n, NUM_PARAMS), np.nan, dtype=np.float64)
    for idx, code in enumerate(PARAMETER_CODES):
        if code in df.columns:
            values[:, idx] = df[code].values.astype(np.float64)

    # Unix timestamps
    timestamps = df.index.astype(np.int64) / 1e9  # nanoseconds -> seconds

    # Delta-t between consecutive observations
    delta_ts = np.zeros(n, dtype=np.float64)
    delta_ts[1:] = np.diff(timestamps)
    delta_ts[0] = 0.0  # No predecessor for first observation

    # Quality mask: 1.0 = valid, 0.0 = NaN/missing
    quality_mask = (~np.isnan(values)).astype(np.float64)

    # Rolling z-score normalization on irregular observations
    window_seconds = rolling_window_days * 86400.0
    normalized, _, _ = _rolling_zscore_irregular(
        values, timestamps, window_seconds
    )
    # Replace NaN in normalized with 0.0 (masked during training)
    normalized_clean = np.nan_to_num(normalized, nan=0.0)

    # Build sample list
    samples: list[IrregularTimeSample] = []
    for i in range(n):
        samples.append(
            IrregularTimeSample(
                timestamp_seconds=float(timestamps[i]),
                parameter_values=values[i].astype(np.float32),
                delta_t_seconds=float(delta_ts[i]),
                quality_mask=quality_mask[i].astype(np.float32),
                normalized_values=normalized_clean[i].astype(np.float32),
            )
        )

    logger.info(
        f"Station {site_no}: {len(samples)} irregular samples, "
        f"{len(param_cols)}/{NUM_PARAMS} parameters present"
    )
    return samples


# ---------------------------------------------------------------------------
# Sequence extraction for AquaSSM batching
# ---------------------------------------------------------------------------


def extract_irregular_sequences(
    samples: list[IrregularTimeSample],
    max_sequence_length: int = 2000,
    overlap_fraction: float = 0.10,
) -> list[dict]:
    """Split a station's irregular samples into fixed-length sequences.

    AquaSSM processes sequences of bounded length.  Long station records
    are split into overlapping chunks.

    Parameters
    ----------
    samples:
        Chronologically sorted list of :class:`IrregularTimeSample`.
    max_sequence_length:
        Maximum number of observations per sequence.
    overlap_fraction:
        Fraction of ``max_sequence_length`` to overlap between consecutive
        sequences (default 10%).

    Returns
    -------
    List of dicts, each containing:
      - ``"timestamps"``: float array of shape ``(T,)``
      - ``"values"``: float array of shape ``(T, 6)``
      - ``"delta_ts"``: float array of shape ``(T,)``
      - ``"masks"``: float array of shape ``(T, 6)``
      - ``"normalized"``: float array of shape ``(T, 6)``
    """
    if not samples:
        return []

    n = len(samples)
    overlap = max(1, int(max_sequence_length * overlap_fraction))
    stride = max_sequence_length - overlap

    sequences: list[dict] = []
    start = 0
    while start < n:
        end = min(start + max_sequence_length, n)
        chunk = samples[start:end]

        seq = {
            "timestamps": np.array(
                [s.timestamp_seconds for s in chunk], dtype=np.float64
            ),
            "values": np.stack(
                [s.parameter_values for s in chunk]
            ).astype(np.float32),
            "delta_ts": np.array(
                [s.delta_t_seconds for s in chunk], dtype=np.float32
            ),
            "masks": np.stack(
                [s.quality_mask for s in chunk]
            ).astype(np.float32),
            "normalized": np.stack(
                [s.normalized_values for s in chunk]
            ).astype(np.float32),
        }
        sequences.append(seq)

        if end >= n:
            break
        start += stride

    logger.debug(
        f"Extracted {len(sequences)} irregular sequences "
        f"(max_len={max_sequence_length}, overlap={overlap})"
    )
    return sequences


# ---------------------------------------------------------------------------
# Gap statistics for AquaSSM temporal decomposition
# ---------------------------------------------------------------------------


def compute_gap_statistics(samples: list[IrregularTimeSample]) -> dict:
    """Compute distribution of inter-observation time gaps for a station.

    These statistics inform AquaSSM's multi-scale temporal decomposition
    initialization — the model needs to know the characteristic gap
    structure of the input data.

    Parameters
    ----------
    samples:
        Chronologically sorted list of :class:`IrregularTimeSample`.

    Returns
    -------
    Dict with keys:
      - ``"n_observations"``: total number of observations
      - ``"median_gap_seconds"``: median inter-observation gap
      - ``"mean_gap_seconds"``: mean inter-observation gap
      - ``"max_gap_seconds"``: largest gap
      - ``"min_gap_seconds"``: smallest non-zero gap
      - ``"frac_gt_1h"``: fraction of gaps > 1 hour
      - ``"frac_gt_6h"``: fraction of gaps > 6 hours
      - ``"frac_gt_24h"``: fraction of gaps > 24 hours
      - ``"total_span_days"``: total time span covered
    """
    if len(samples) < 2:
        return {
            "n_observations": len(samples),
            "median_gap_seconds": 0.0,
            "mean_gap_seconds": 0.0,
            "max_gap_seconds": 0.0,
            "min_gap_seconds": 0.0,
            "frac_gt_1h": 0.0,
            "frac_gt_6h": 0.0,
            "frac_gt_24h": 0.0,
            "total_span_days": 0.0,
        }

    # Skip the first sample (delta_t=0) — use gaps from index 1 onward
    gaps = np.array([s.delta_t_seconds for s in samples[1:]], dtype=np.float64)
    nonzero_gaps = gaps[gaps > 0]

    total_span = samples[-1].timestamp_seconds - samples[0].timestamp_seconds

    return {
        "n_observations": len(samples),
        "median_gap_seconds": float(np.median(gaps)) if len(gaps) > 0 else 0.0,
        "mean_gap_seconds": float(np.mean(gaps)) if len(gaps) > 0 else 0.0,
        "max_gap_seconds": float(np.max(gaps)) if len(gaps) > 0 else 0.0,
        "min_gap_seconds": float(np.min(nonzero_gaps)) if len(nonzero_gaps) > 0 else 0.0,
        "frac_gt_1h": float(np.mean(gaps > 3600)) if len(gaps) > 0 else 0.0,
        "frac_gt_6h": float(np.mean(gaps > 21600)) if len(gaps) > 0 else 0.0,
        "frac_gt_24h": float(np.mean(gaps > 86400)) if len(gaps) > 0 else 0.0,
        "total_span_days": total_span / 86400.0,
    }


# ---------------------------------------------------------------------------
# Batch irregular preprocessing
# ---------------------------------------------------------------------------


def preprocess_all_stations_irregular(
    input_dir: str | Path,
    output_dir: str | Path = "data/sensor/irregular",
    *,
    max_sequence_length: int = 2000,
    rolling_window_days: int = ROLLING_WINDOW_DAYS,
) -> dict:
    """Batch irregular preprocessing for all station Parquet files.

    Processes each station with :func:`preprocess_station_irregular`, splits
    into sequences via :func:`extract_irregular_sequences`, and saves each
    station's output as an ``.npz`` file.

    Parameters
    ----------
    input_dir:
        Directory containing raw station Parquet files.
    output_dir:
        Directory for output ``.npz`` files.
    max_sequence_length:
        Maximum observations per sequence chunk.
    rolling_window_days:
        Days for rolling z-score normalization.

    Returns
    -------
    Dict mapping ``site_no`` -> dict with keys:
      - ``"path"``: Path to the saved ``.npz`` file
      - ``"n_samples"``: total irregular samples
      - ``"n_sequences"``: number of sequence chunks
      - ``"gap_stats"``: gap statistics dict
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.parquet"))
    if not files:
        logger.warning(f"No Parquet files found in {input_dir}")
        return {}

    results: dict[str, dict] = {}
    progress = make_progress()

    with progress:
        task = progress.add_task(
            "Irregular preprocessing", total=len(files)
        )
        for f in files:
            site_no = f.stem
            try:
                samples = preprocess_station_irregular(
                    f, rolling_window_days=rolling_window_days
                )
                if not samples:
                    progress.advance(task)
                    continue

                sequences = extract_irregular_sequences(
                    samples, max_sequence_length=max_sequence_length
                )
                gap_stats = compute_gap_statistics(samples)

                # Save as .npz — one array per key, sequences stacked
                npz_path = output_dir / f"{site_no}.npz"
                npz_data: dict[str, np.ndarray] = {}
                for i, seq in enumerate(sequences):
                    for key, arr in seq.items():
                        npz_data[f"seq{i}_{key}"] = arr
                npz_data["n_sequences"] = np.array(len(sequences))
                np.savez_compressed(str(npz_path), **npz_data)

                results[site_no] = {
                    "path": npz_path,
                    "n_samples": len(samples),
                    "n_sequences": len(sequences),
                    "gap_stats": gap_stats,
                }
                logger.info(
                    f"Station {site_no}: {len(samples)} samples -> "
                    f"{len(sequences)} sequences -> {npz_path}"
                )
            except Exception as exc:
                logger.warning(
                    f"Irregular preprocessing failed for {site_no}: {exc}"
                )
            progress.advance(task)

    total_samples = sum(r["n_samples"] for r in results.values())
    total_seqs = sum(r["n_sequences"] for r in results.values())
    logger.info(
        f"Irregular preprocessing complete: {len(results)} stations, "
        f"{total_samples} samples, {total_seqs} sequences -> {output_dir}"
    )
    return results
