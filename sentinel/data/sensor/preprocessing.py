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
