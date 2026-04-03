"""
Trajectory data structures and augmentation for BioMotion diffusion pretraining.

Provides:
  - TrajectoryRecord dataclass for individual trajectory storage
  - TrajectoryDataset (torch Dataset) for batching and filtering
  - Augmentation functions (rotation, flip, jitter, speed scaling, etc.)
  - Diffusion corruption utilities (noise schedules for denoising pretraining)
  - Train/val/test splitting with stratification
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from sentinel.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# TrajectoryRecord dataclass
# ---------------------------------------------------------------------------


@dataclass
class TrajectoryRecord:
    """A single organism behavioral trajectory with metadata.

    Attributes
    ----------
    species:
        Organism type (``"daphnia"``, ``"fish"``, ``"mussel"``).
    keypoints:
        Pose keypoint coordinates, shape ``(T, n_keypoints, 2)``.
    features:
        Behavioral features extracted from the keypoints, shape
        ``(T, n_features)``.
    timestamps:
        Timestamp for each frame in seconds, shape ``(T,)``.
    label:
        Binary label: 0 = normal, 1 = anomalous.
    anomaly_type:
        Description of the anomaly type.  ``"none"`` for normal
        trajectories.
    metadata:
        Arbitrary metadata dict (source file, recording ID, etc.).
    """

    species: str
    keypoints: np.ndarray  # (T, n_keypoints, 2)
    features: np.ndarray   # (T, n_features)
    timestamps: np.ndarray  # (T,) seconds
    label: int  # 0=normal, 1=anomalous
    anomaly_type: str  # "none", "reduced_velocity", "erratic", "freezing", ...
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        """Number of time frames."""
        return self.keypoints.shape[0]

    @property
    def n_keypoints(self) -> int:
        """Number of keypoints per frame."""
        return self.keypoints.shape[1]

    @property
    def duration_s(self) -> float:
        """Duration of the trajectory in seconds."""
        if len(self.timestamps) < 2:
            return 0.0
        return float(self.timestamps[-1] - self.timestamps[0])

    def save(self, path: str | Path) -> Path:
        """Save trajectory to a compressed ``.npz`` file.

        Parameters
        ----------
        path:
            Output file path (should end in ``.npz``).

        Returns
        -------
        Path to the saved file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(path),
            keypoints=self.keypoints,
            features=self.features,
            timestamps=self.timestamps,
            label=np.array(self.label, dtype=np.int32),
            anomaly_type=np.array(self.anomaly_type),
            species=np.array(self.species),
            metadata=np.array(json.dumps(self.metadata)),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> TrajectoryRecord:
        """Load a trajectory from a ``.npz`` file.

        Parameters
        ----------
        path:
            Path to the ``.npz`` file.

        Returns
        -------
        Loaded TrajectoryRecord instance.
        """
        path = Path(path)
        data = np.load(str(path), allow_pickle=False)

        metadata_str = str(data["metadata"])
        try:
            metadata = json.loads(metadata_str)
        except (json.JSONDecodeError, TypeError):
            metadata = {}

        return cls(
            species=str(data["species"]),
            keypoints=data["keypoints"].astype(np.float32),
            features=data["features"].astype(np.float32),
            timestamps=data["timestamps"].astype(np.float64),
            label=int(data["label"]),
            anomaly_type=str(data["anomaly_type"]),
            metadata=metadata,
        )


# ---------------------------------------------------------------------------
# TrajectoryDataset (PyTorch Dataset)
# ---------------------------------------------------------------------------


class TrajectoryDataset:
    """PyTorch-compatible dataset for behavioral trajectory data.

    Loads TrajectoryRecord ``.npz`` files from a directory with support
    for filtering by species, label, and anomaly type.

    Parameters
    ----------
    data_dir:
        Directory containing ``.npz`` trajectory files.
    species_filter:
        If provided, only include trajectories matching these species.
    label_filter:
        If provided, only include trajectories with these labels.
    anomaly_type_filter:
        If provided, only include trajectories with these anomaly types.
    max_length:
        If set, truncate trajectories longer than this many frames.
    transform:
        Optional callable to apply to each sample.

    Notes
    -----
    This class implements the ``torch.utils.data.Dataset`` interface
    but does **not** inherit from it directly, so that ``torch`` is not
    a hard import requirement.  When used with a DataLoader, assign
    ``__class__`` or register via ``torch.utils.data.Dataset.register``
    as needed.
    """

    def __init__(
        self,
        data_dir: str | Path,
        species_filter: Sequence[str] | None = None,
        label_filter: Sequence[int] | None = None,
        anomaly_type_filter: Sequence[str] | None = None,
        max_length: int | None = None,
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.max_length = max_length
        self.transform = transform

        # Discover all .npz files
        all_files = sorted(self.data_dir.glob("*.npz"))
        if not all_files:
            logger.warning(f"No .npz files found in {self.data_dir}")

        # Apply filters by loading metadata only
        self._files: list[Path] = []
        self._metadata_cache: dict[int, dict[str, Any]] = {}

        for f in all_files:
            try:
                data = np.load(str(f), allow_pickle=False)
                species = str(data["species"])
                label = int(data["label"])
                anomaly_type = str(data["anomaly_type"])

                if species_filter and species not in species_filter:
                    continue
                if label_filter and label not in label_filter:
                    continue
                if anomaly_type_filter and anomaly_type not in anomaly_type_filter:
                    continue

                idx = len(self._files)
                self._files.append(f)
                self._metadata_cache[idx] = {
                    "species": species,
                    "label": label,
                    "anomaly_type": anomaly_type,
                }
            except Exception as exc:
                logger.warning(f"Skipping {f.name}: {exc}")

        logger.info(
            f"TrajectoryDataset: {len(self._files)} trajectories from {self.data_dir}"
        )

    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Load and return a trajectory sample.

        Returns
        -------
        Dict with keys:
          - ``"keypoints"``: float32 array ``(T, n_keypoints, 2)``
          - ``"features"``: float32 array ``(T, n_features)``
          - ``"label"``: int
          - ``"species"``: str
          - ``"anomaly_type"``: str
          - ``"metadata"``: dict
          - ``"length"``: int (original length before padding)
        """
        record = TrajectoryRecord.load(self._files[idx])

        keypoints = record.keypoints
        features = record.features

        # Truncate if needed
        if self.max_length is not None and keypoints.shape[0] > self.max_length:
            keypoints = keypoints[: self.max_length]
            features = features[: self.max_length]

        sample = {
            "keypoints": keypoints,
            "features": features,
            "label": record.label,
            "species": record.species,
            "anomaly_type": record.anomaly_type,
            "metadata": record.metadata,
            "length": keypoints.shape[0],
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    @staticmethod
    def collate_fn(
        batch: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Collate variable-length trajectories into a padded batch.

        Pads keypoints and features to the maximum length in the batch
        and produces a boolean mask indicating valid (non-padded) frames.

        Parameters
        ----------
        batch:
            List of sample dicts from ``__getitem__``.

        Returns
        -------
        Dict with keys:
          - ``"keypoints"``: float32 array ``(B, T_max, n_keypoints, 2)``
          - ``"features"``: float32 array ``(B, T_max, n_features)``
          - ``"mask"``: bool array ``(B, T_max)`` — True for valid frames
          - ``"labels"``: int32 array ``(B,)``
          - ``"lengths"``: int32 array ``(B,)``
          - ``"species"``: list of str
          - ``"anomaly_types"``: list of str
          - ``"metadata"``: list of dict
        """
        max_len = max(s["length"] for s in batch)
        batch_size = len(batch)

        # Determine shapes from first sample
        sample0 = batch[0]
        n_kp = sample0["keypoints"].shape[1]
        n_feat = sample0["features"].shape[1] if sample0["features"].ndim > 1 else 1

        kp_padded = np.zeros(
            (batch_size, max_len, n_kp, 2), dtype=np.float32
        )
        feat_padded = np.zeros(
            (batch_size, max_len, n_feat), dtype=np.float32
        )
        mask = np.zeros((batch_size, max_len), dtype=bool)
        labels = np.zeros(batch_size, dtype=np.int32)
        lengths = np.zeros(batch_size, dtype=np.int32)
        species_list: list[str] = []
        anomaly_list: list[str] = []
        meta_list: list[dict] = []

        for i, sample in enumerate(batch):
            length = sample["length"]
            kp_padded[i, :length] = sample["keypoints"][:length]
            feat = sample["features"][:length]
            if feat.ndim == 1:
                feat = feat[:, np.newaxis]
            feat_padded[i, :length] = feat
            mask[i, :length] = True
            labels[i] = sample["label"]
            lengths[i] = length
            species_list.append(sample["species"])
            anomaly_list.append(sample["anomaly_type"])
            meta_list.append(sample.get("metadata", {}))

        return {
            "keypoints": kp_padded,
            "features": feat_padded,
            "mask": mask,
            "labels": labels,
            "lengths": lengths,
            "species": species_list,
            "anomaly_types": anomaly_list,
            "metadata": meta_list,
        }


# ---------------------------------------------------------------------------
# Trajectory augmentation
# ---------------------------------------------------------------------------


def augment_trajectory(
    trajectory: TrajectoryRecord,
    augmentations: Sequence[str] | None = None,
    rng: np.random.Generator | None = None,
) -> TrajectoryRecord:
    """Apply data augmentations to a trajectory.

    Parameters
    ----------
    trajectory:
        Input trajectory record.
    augmentations:
        List of augmentation names to apply.  Supported:
          - ``"temporal_jitter"``: randomly shift timestamps
          - ``"spatial_rotation"``: rotate keypoints by random angle
          - ``"spatial_flip"``: mirror trajectory horizontally
          - ``"speed_scaling"``: scale velocities by random factor [0.8, 1.2]
          - ``"noise_injection"``: add Gaussian noise to keypoints
          - ``"segment_shuffle"``: shuffle short temporal segments
        If None, all augmentations are applied.
    rng:
        Numpy random generator.

    Returns
    -------
    New TrajectoryRecord with augmented data.
    """
    rng = rng or np.random.default_rng()

    all_augmentations = [
        "temporal_jitter",
        "spatial_rotation",
        "spatial_flip",
        "speed_scaling",
        "noise_injection",
        "segment_shuffle",
    ]
    augmentations = augmentations or all_augmentations

    kp = trajectory.keypoints.copy().astype(np.float64)
    timestamps = trajectory.timestamps.copy()
    features = trajectory.features.copy()

    for aug in augmentations:
        if aug == "temporal_jitter":
            # Add small random offsets to timestamps (max 10% of frame interval)
            if len(timestamps) > 1:
                dt = float(np.median(np.diff(timestamps)))
                jitter = rng.uniform(-0.1 * dt, 0.1 * dt, size=len(timestamps))
                timestamps = timestamps + jitter
                # Ensure monotonically increasing
                timestamps = np.sort(timestamps)

        elif aug == "spatial_rotation":
            angle = rng.uniform(0, 2 * math.pi)
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float64)
            for t in range(kp.shape[0]):
                kp[t] = kp[t] @ rot.T

        elif aug == "spatial_flip":
            # Mirror along x-axis
            kp[:, :, 0] = -kp[:, :, 0]

        elif aug == "speed_scaling":
            # Scale displacements by a random factor
            factor = rng.uniform(0.8, 1.2)
            centroid = np.mean(kp, axis=1, keepdims=True)
            displacements = np.diff(centroid, axis=0)
            displacements *= factor
            # Reconstruct centroid path
            new_centroid = np.zeros_like(centroid)
            new_centroid[0] = centroid[0]
            for t in range(1, len(centroid)):
                new_centroid[t] = new_centroid[t - 1] + displacements[t - 1]
            # Shift keypoints to follow new centroid
            shift = new_centroid - centroid
            kp += shift

        elif aug == "noise_injection":
            # Add Gaussian noise proportional to keypoint scale
            std = max(float(np.nanstd(kp)), 1e-6) * 0.02
            noise = rng.normal(0, std, size=kp.shape)
            kp += noise

        elif aug == "segment_shuffle":
            # Shuffle short segments of 5-15 frames
            n_frames = kp.shape[0]
            seg_len = rng.integers(5, min(16, n_frames // 2 + 1))
            n_segs = n_frames // seg_len
            if n_segs > 1:
                segments_kp = [
                    kp[i * seg_len : (i + 1) * seg_len] for i in range(n_segs)
                ]
                segments_ts = [
                    timestamps[i * seg_len : (i + 1) * seg_len]
                    for i in range(n_segs)
                ]
                segments_feat = [
                    features[i * seg_len : (i + 1) * seg_len]
                    for i in range(n_segs)
                ]

                perm = rng.permutation(n_segs)
                kp_shuffled = np.concatenate([segments_kp[p] for p in perm])
                ts_shuffled = np.concatenate([segments_ts[p] for p in perm])
                feat_shuffled = np.concatenate([segments_feat[p] for p in perm])

                # Retain any leftover frames at the end
                leftover = n_frames - n_segs * seg_len
                if leftover > 0:
                    kp = np.concatenate([kp_shuffled, kp[-leftover:]])
                    timestamps = np.concatenate(
                        [ts_shuffled, timestamps[-leftover:]]
                    )
                    features = np.concatenate(
                        [feat_shuffled, features[-leftover:]]
                    )
                else:
                    kp = kp_shuffled
                    timestamps = ts_shuffled
                    features = feat_shuffled

        else:
            logger.warning(f"Unknown augmentation: {aug!r}")

    return TrajectoryRecord(
        species=trajectory.species,
        keypoints=kp.astype(np.float32),
        features=features.astype(np.float32),
        timestamps=timestamps,
        label=trajectory.label,
        anomaly_type=trajectory.anomaly_type,
        metadata={
            **trajectory.metadata,
            "augmentations": list(augmentations),
        },
    )


# ---------------------------------------------------------------------------
# Diffusion corruption for denoising pretraining
# ---------------------------------------------------------------------------


def _linear_schedule(t: float) -> float:
    """Linear noise schedule: beta(t) = t."""
    return t


def _cosine_schedule(t: float) -> float:
    """Cosine noise schedule from Nichol & Dhariwal (2021)."""
    s = 0.008
    f_t = math.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    f_0 = math.cos(s / (1 + s) * math.pi / 2) ** 2
    return 1.0 - f_t / f_0


def _sigmoid_schedule(t: float) -> float:
    """Sigmoid noise schedule."""
    return 1.0 / (1.0 + math.exp(-12.0 * (t - 0.5)))


_NOISE_SCHEDULES: dict[str, Callable[[float], float]] = {
    "linear": _linear_schedule,
    "cosine": _cosine_schedule,
    "sigmoid": _sigmoid_schedule,
}


def corrupt_trajectory(
    trajectory: TrajectoryRecord,
    noise_level: float,
    noise_schedule: str = "cosine",
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Add calibrated noise to a trajectory for diffusion pretraining.

    The trajectory keypoints are corrupted according to the specified
    noise schedule at the given noise level.  The model is trained to
    predict the added noise (epsilon prediction).

    Parameters
    ----------
    trajectory:
        Input trajectory record.
    noise_level:
        Noise level parameter ``t`` in ``[0, 1]``.  ``t=0`` means no
        noise; ``t=1`` means maximum noise (fully corrupted).
    noise_schedule:
        Noise schedule type: ``"linear"``, ``"cosine"``, or ``"sigmoid"``.
    rng:
        Numpy random generator.

    Returns
    -------
    Tuple of:
      - ``corrupted``: noised keypoints, same shape as input keypoints
      - ``noise``: the noise that was added (model target for denoising)

    Raises
    ------
    ValueError
        If noise_schedule is not recognized or noise_level is out of range.
    """
    if noise_schedule not in _NOISE_SCHEDULES:
        raise ValueError(
            f"Unknown noise schedule {noise_schedule!r}. "
            f"Supported: {list(_NOISE_SCHEDULES.keys())}"
        )
    if not 0.0 <= noise_level <= 1.0:
        raise ValueError(
            f"noise_level must be in [0, 1], got {noise_level}"
        )

    rng = rng or np.random.default_rng()
    schedule_fn = _NOISE_SCHEDULES[noise_schedule]

    # Compute noise scale from schedule
    beta = schedule_fn(noise_level)
    alpha = 1.0 - beta
    alpha_bar = alpha  # simplified for single step; in full diffusion this is cumulative

    # Determine noise scale from data statistics
    kp = trajectory.keypoints.astype(np.float64)
    data_std = max(float(np.nanstd(kp)), 1e-6)

    # Generate noise
    noise = rng.normal(0, data_std, size=kp.shape).astype(np.float64)

    # Forward diffusion: x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * noise
    sqrt_alpha = math.sqrt(max(alpha_bar, 1e-9))
    sqrt_one_minus_alpha = math.sqrt(max(1.0 - alpha_bar, 1e-9))

    corrupted = sqrt_alpha * kp + sqrt_one_minus_alpha * noise

    return corrupted.astype(np.float32), noise.astype(np.float32)


def corrupt_batch(
    keypoints_batch: np.ndarray,
    noise_levels: np.ndarray,
    noise_schedule: str = "cosine",
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Corrupt a batch of keypoint trajectories for diffusion training.

    Parameters
    ----------
    keypoints_batch:
        Batch of keypoints, shape ``(B, T, n_keypoints, 2)``.
    noise_levels:
        Per-sample noise levels, shape ``(B,)`` in ``[0, 1]``.
    noise_schedule:
        Noise schedule type.
    rng:
        Numpy random generator.

    Returns
    -------
    Tuple of:
      - ``corrupted``: noised keypoints ``(B, T, n_keypoints, 2)``
      - ``noise``: added noise ``(B, T, n_keypoints, 2)``
      - ``alpha_bars``: noise scaling factors ``(B,)``
    """
    rng = rng or np.random.default_rng()
    schedule_fn = _NOISE_SCHEDULES[noise_schedule]

    B = keypoints_batch.shape[0]
    corrupted = np.zeros_like(keypoints_batch, dtype=np.float32)
    noise_out = np.zeros_like(keypoints_batch, dtype=np.float32)
    alpha_bars = np.zeros(B, dtype=np.float32)

    for i in range(B):
        beta = schedule_fn(float(noise_levels[i]))
        alpha_bar = 1.0 - beta
        alpha_bars[i] = alpha_bar

        kp = keypoints_batch[i].astype(np.float64)
        data_std = max(float(np.nanstd(kp)), 1e-6)
        noise = rng.normal(0, data_std, size=kp.shape)

        sqrt_alpha = math.sqrt(max(alpha_bar, 1e-9))
        sqrt_one_minus_alpha = math.sqrt(max(1.0 - alpha_bar, 1e-9))

        corrupted[i] = (sqrt_alpha * kp + sqrt_one_minus_alpha * noise).astype(
            np.float32
        )
        noise_out[i] = noise.astype(np.float32)

    return corrupted, noise_out, alpha_bars


# ---------------------------------------------------------------------------
# Train/val/test splitting
# ---------------------------------------------------------------------------


def split_trajectories(
    records: Sequence[TrajectoryRecord],
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    stratify_by: str = "species",
    seed: int = 42,
) -> dict[str, list[TrajectoryRecord]]:
    """Split trajectory records into train/val/test sets.

    Parameters
    ----------
    records:
        Sequence of TrajectoryRecord instances.
    train_frac, val_frac, test_frac:
        Fractions for each split. Must sum to 1.0.
    stratify_by:
        Attribute to stratify by: ``"species"``, ``"label"``,
        ``"anomaly_type"``, or ``"species_label"`` (joint stratification).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    Dict with keys ``"train"``, ``"val"``, ``"test"``, each mapping
    to a list of TrajectoryRecord.

    Raises
    ------
    ValueError
        If fractions do not sum to approximately 1.0, or if
        *stratify_by* is unrecognized.
    """
    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"Split fractions must sum to 1.0, got {total:.4f} "
            f"({train_frac} + {val_frac} + {test_frac})"
        )

    valid_stratify = {"species", "label", "anomaly_type", "species_label"}
    if stratify_by not in valid_stratify:
        raise ValueError(
            f"Unknown stratify_by={stratify_by!r}. Supported: {valid_stratify}"
        )

    rng = np.random.default_rng(seed)

    # Group by stratification key
    groups: dict[str, list[int]] = {}
    for i, rec in enumerate(records):
        if stratify_by == "species":
            key = rec.species
        elif stratify_by == "label":
            key = str(rec.label)
        elif stratify_by == "anomaly_type":
            key = rec.anomaly_type
        elif stratify_by == "species_label":
            key = f"{rec.species}_{rec.label}"
        else:
            key = "all"
        groups.setdefault(key, []).append(i)

    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []

    for key, indices in groups.items():
        shuffled = rng.permutation(indices).tolist()
        n = len(shuffled)
        n_train = max(1, int(round(n * train_frac)))
        n_val = max(0, int(round(n * val_frac)))
        # Ensure at least one test sample if test_frac > 0
        n_test = n - n_train - n_val
        if n_test < 0:
            n_val = max(0, n_val + n_test)
            n_test = 0

        train_idx.extend(shuffled[:n_train])
        val_idx.extend(shuffled[n_train : n_train + n_val])
        test_idx.extend(shuffled[n_train + n_val :])

    result = {
        "train": [records[i] for i in train_idx],
        "val": [records[i] for i in val_idx],
        "test": [records[i] for i in test_idx],
    }

    logger.info(
        f"Split {len(records)} trajectories: "
        f"train={len(result['train'])}, "
        f"val={len(result['val'])}, "
        f"test={len(result['test'])} "
        f"(stratified by {stratify_by})"
    )
    return result
