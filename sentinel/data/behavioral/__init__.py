"""
Behavioral tracking data pipeline for SENTINEL BioMotion encoder.

Submodules:
  - download: data acquisition and synthetic trajectory generation
  - preprocessing: pose estimation, feature extraction, normalization
  - trajectory: data structures, augmentation, and diffusion corruption
"""

from __future__ import annotations

from sentinel.data.behavioral.download import (
    download_all_behavioral,
    download_daphnia_datasets,
    download_fish_tracking,
    download_mussel_valve_data,
    generate_synthetic_trajectories,
)
from sentinel.data.behavioral.preprocessing import (
    ORGANISM_KEYPOINTS,
    compute_behavioral_features,
    downsample_features,
    load_keypoints,
    normalize_trajectory,
    preprocess_all_recordings,
    preprocess_recording,
    run_sleap_inference,
)
from sentinel.data.behavioral.trajectory import (
    TrajectoryDataset,
    TrajectoryRecord,
    augment_trajectory,
    corrupt_batch,
    corrupt_trajectory,
    split_trajectories,
)

__all__ = [
    # download
    "download_all_behavioral",
    "download_daphnia_datasets",
    "download_fish_tracking",
    "download_mussel_valve_data",
    "generate_synthetic_trajectories",
    # preprocessing
    "ORGANISM_KEYPOINTS",
    "compute_behavioral_features",
    "downsample_features",
    "load_keypoints",
    "normalize_trajectory",
    "preprocess_all_recordings",
    "preprocess_recording",
    "run_sleap_inference",
    # trajectory
    "TrajectoryDataset",
    "TrajectoryRecord",
    "augment_trajectory",
    "corrupt_batch",
    "corrupt_trajectory",
    "split_trajectories",
]
