"""Temporal-spatial holdout split utilities for SENTINEL.

Implements strict evaluation protocol:
  - Train:  2015-2022, spatial folds A-C
  - Val:    2023,       spatial fold D
  - Test:   2024-2026,  spatial fold E

Sites are grouped into 5 spatial folds via k-means clustering on
(lat, lon) coordinates, ensuring geographically co-located sites
stay in the same fold and are never seen across train/val/test.

Every modality-specific training script should call `get_split_assignment()`
to obtain per-sample train/val/test labels.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional, Sequence

import numpy as np

from sentinel.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SPLIT_SEED = 42

# Temporal boundaries (inclusive start, exclusive end)
TEMPORAL_RANGES = {
    "train": ("2015-01-01", "2023-01-01"),  # 2015-2022
    "val":   ("2023-01-01", "2024-01-01"),  # 2023
    "test":  ("2024-01-01", "2027-01-01"),  # 2024-2026
}

# Spatial fold assignment: 5 folds, A-E
NUM_SPATIAL_FOLDS = 5
FOLD_ASSIGNMENT = {
    "train": [0, 1, 2],  # Folds A, B, C
    "val":   [3],         # Fold D
    "test":  [4],         # Fold E
}

SplitName = Literal["train", "val", "test"]


@dataclass
class SplitConfig:
    """Configuration for temporal-spatial holdout splits."""

    temporal_ranges: dict[str, tuple[str, str]] = field(
        default_factory=lambda: dict(TEMPORAL_RANGES)
    )
    num_spatial_folds: int = NUM_SPATIAL_FOLDS
    fold_assignment: dict[str, list[int]] = field(
        default_factory=lambda: dict(FOLD_ASSIGNMENT)
    )
    seed: int = SPLIT_SEED

    @classmethod
    def from_yaml(cls, cfg: dict) -> "SplitConfig":
        """Create from parsed YAML config dict (under 'evaluation.splits')."""
        splits_cfg = cfg.get("evaluation", {}).get("splits", {})
        if not splits_cfg:
            return cls()  # defaults
        return cls(
            temporal_ranges=splits_cfg.get("temporal_ranges", dict(TEMPORAL_RANGES)),
            num_spatial_folds=splits_cfg.get("num_spatial_folds", NUM_SPATIAL_FOLDS),
            fold_assignment=splits_cfg.get("fold_assignment", dict(FOLD_ASSIGNMENT)),
            seed=splits_cfg.get("seed", SPLIT_SEED),
        )


# ---------------------------------------------------------------------------
# Spatial fold assignment via deterministic hashing
# ---------------------------------------------------------------------------

def assign_spatial_fold(
    site_id: str,
    num_folds: int = NUM_SPATIAL_FOLDS,
    seed: int = SPLIT_SEED,
) -> int:
    """Assign a site to a spatial fold using deterministic hashing.

    This ensures the same site always lands in the same fold, regardless
    of the order sites are processed.  Uses SHA-256 for uniform distribution.

    Parameters
    ----------
    site_id:
        Unique site identifier (USGS site_no, NEON code, EMP sample_id, etc.).
    num_folds:
        Number of spatial folds (default 5).
    seed:
        Seed mixed into the hash for reproducibility.

    Returns
    -------
    Fold index in [0, num_folds).
    """
    key = f"{seed}:{site_id}".encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()
    return int(digest, 16) % num_folds


def assign_spatial_fold_geographic(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    site_ids: Sequence[str],
    num_folds: int = NUM_SPATIAL_FOLDS,
    seed: int = SPLIT_SEED,
) -> dict[str, int]:
    """Assign sites to spatial folds using k-means on (lat, lon).

    Groups geographically close sites into the same fold so that
    spatial autocorrelation doesn't leak across splits.

    Parameters
    ----------
    latitudes, longitudes:
        Arrays of site coordinates (WGS-84).
    site_ids:
        Corresponding site identifiers.
    num_folds:
        Number of folds.
    seed:
        Random seed for k-means.

    Returns
    -------
    Dict mapping site_id → fold index.
    """
    from sklearn.cluster import KMeans

    coords = np.column_stack([latitudes, longitudes])

    # Use k-means to cluster sites geographically
    kmeans = KMeans(n_clusters=num_folds, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(coords)

    # Deterministic fold permutation so fold indices are reproducible
    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_folds)
    remapped = perm[labels]

    return {sid: int(remapped[i]) for i, sid in enumerate(site_ids)}


# ---------------------------------------------------------------------------
# Temporal assignment
# ---------------------------------------------------------------------------

def get_temporal_split(
    timestamp: datetime | str | float,
    config: SplitConfig | None = None,
) -> SplitName | None:
    """Determine which temporal split a timestamp belongs to.

    Parameters
    ----------
    timestamp:
        A datetime, ISO-format string, or Unix timestamp (seconds).
    config:
        Split configuration. Uses defaults if None.

    Returns
    -------
    "train", "val", "test", or None if outside all ranges.
    """
    if config is None:
        config = SplitConfig()

    if isinstance(timestamp, (int, float)):
        dt = datetime.utcfromtimestamp(timestamp)
    elif isinstance(timestamp, str):
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00").replace("+00:00", ""))
    else:
        dt = timestamp

    for split_name, (start_str, end_str) in config.temporal_ranges.items():
        start = datetime.fromisoformat(start_str)
        end = datetime.fromisoformat(end_str)
        if start <= dt < end:
            return split_name

    return None


# ---------------------------------------------------------------------------
# Combined temporal-spatial assignment
# ---------------------------------------------------------------------------

def get_split_assignment(
    site_id: str,
    timestamp: datetime | str | float | None = None,
    spatial_fold: int | None = None,
    config: SplitConfig | None = None,
) -> SplitName | None:
    """Determine the split for a (site, timestamp) pair.

    A sample is assigned to a split only if BOTH its spatial fold AND
    temporal range match the same split.  Samples that fall in a
    spatial fold assigned to "train" but a temporal range assigned to
    "val" are EXCLUDED (returned as None) to prevent leakage.

    Parameters
    ----------
    site_id:
        Unique site identifier.
    timestamp:
        Observation timestamp (optional for non-temporal modalities).
    spatial_fold:
        Pre-computed fold index. If None, computed from site_id via hashing.
    config:
        Split configuration.

    Returns
    -------
    "train", "val", "test", or None (excluded).
    """
    if config is None:
        config = SplitConfig()

    # Spatial assignment
    if spatial_fold is None:
        spatial_fold = assign_spatial_fold(site_id, config.num_spatial_folds, config.seed)

    spatial_split = None
    for split_name, folds in config.fold_assignment.items():
        if spatial_fold in folds:
            spatial_split = split_name
            break

    if spatial_split is None:
        return None

    # For non-temporal data (molecular, ecotox lab data), use spatial only
    if timestamp is None:
        return spatial_split

    # Temporal assignment
    temporal_split = get_temporal_split(timestamp, config)
    if temporal_split is None:
        return None

    # Both must agree — strict intersection
    if spatial_split == temporal_split:
        return spatial_split

    # For training data, allow temporal-train + spatial-train only
    # For val/test, require exact match to prevent any leakage
    # However, we can be slightly lenient: train sites in train period → train
    # Train sites in val period → exclude (temporal leakage)
    # Val sites in train period → exclude (spatial leakage)
    return None


# ---------------------------------------------------------------------------
# Batch splitting utilities
# ---------------------------------------------------------------------------

def split_indices(
    site_ids: Sequence[str],
    timestamps: Sequence[datetime | str | float] | None = None,
    config: SplitConfig | None = None,
    spatial_folds: dict[str, int] | None = None,
) -> dict[SplitName, list[int]]:
    """Split a list of samples into train/val/test index lists.

    Parameters
    ----------
    site_ids:
        Per-sample site identifiers.
    timestamps:
        Per-sample timestamps (optional).
    config:
        Split configuration.
    spatial_folds:
        Pre-computed site_id → fold mapping (e.g., from k-means).
        If None, uses hash-based assignment.

    Returns
    -------
    Dict with keys "train", "val", "test" mapping to lists of indices.
    Samples that don't fit any split are excluded.
    """
    if config is None:
        config = SplitConfig()

    result: dict[SplitName, list[int]] = {"train": [], "val": [], "test": []}

    for i, sid in enumerate(site_ids):
        ts = timestamps[i] if timestamps is not None else None
        fold = spatial_folds.get(sid) if spatial_folds else None
        split = get_split_assignment(sid, ts, spatial_fold=fold, config=config)
        if split is not None:
            result[split].append(i)

    total = sum(len(v) for v in result.values())
    excluded = len(site_ids) - total
    logger.info(
        f"Temporal-spatial split: train={len(result['train'])}, "
        f"val={len(result['val'])}, test={len(result['test'])}, "
        f"excluded={excluded}"
    )
    return result


def split_indices_spatial_only(
    site_ids: Sequence[str],
    config: SplitConfig | None = None,
    spatial_folds: dict[str, int] | None = None,
) -> dict[SplitName, list[int]]:
    """Split by spatial fold only (for non-temporal data like lab assays).

    Parameters
    ----------
    site_ids:
        Per-sample identifiers (chemical IDs, sample IDs, etc.).
    config:
        Split configuration.
    spatial_folds:
        Pre-computed mapping. If None, uses hash-based assignment.

    Returns
    -------
    Dict mapping split name → index list.
    """
    return split_indices(site_ids, timestamps=None, config=config,
                         spatial_folds=spatial_folds)


# ---------------------------------------------------------------------------
# Save / load fold assignments for reproducibility
# ---------------------------------------------------------------------------

def save_fold_assignments(
    assignments: dict[str, int],
    path: str | Path,
) -> None:
    """Save site → fold assignments to JSON for reproducibility."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(assignments, f, indent=2, sort_keys=True)
    logger.info(f"Saved {len(assignments)} fold assignments to {path}")


def load_fold_assignments(path: str | Path) -> dict[str, int]:
    """Load site → fold assignments from JSON."""
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def summarize_split(
    split_indices: dict[SplitName, list[int]],
    site_ids: Sequence[str] | None = None,
    timestamps: Sequence[datetime | str | float] | None = None,
) -> dict[str, Any]:
    """Generate summary statistics for a split."""
    summary = {}
    for split_name, indices in split_indices.items():
        info: dict[str, Any] = {"n_samples": len(indices)}

        if site_ids is not None:
            unique_sites = set(site_ids[i] for i in indices)
            info["n_sites"] = len(unique_sites)

        if timestamps is not None:
            ts_list = []
            for i in indices:
                t = timestamps[i]
                if isinstance(t, (int, float)):
                    ts_list.append(datetime.utcfromtimestamp(t))
                elif isinstance(t, str):
                    ts_list.append(datetime.fromisoformat(t))
                else:
                    ts_list.append(t)
            if ts_list:
                info["date_range"] = [
                    min(ts_list).isoformat(),
                    max(ts_list).isoformat(),
                ]

        summary[split_name] = info

    return summary
