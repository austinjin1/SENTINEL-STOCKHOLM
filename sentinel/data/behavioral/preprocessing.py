"""
Behavioral data preprocessing for SENTINEL BioMotion encoder.

Pipeline:
  1. Pose estimation via SLEAP (or load pre-tracked keypoints)
  2. Behavioral feature extraction from pose sequences
  3. Temporal downsampling (30 Hz -> 1 Hz summary statistics)
  4. Trajectory normalization (center, scale, heading alignment)
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from sentinel.utils.logging import get_logger, make_progress

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Organism keypoint definitions
# ---------------------------------------------------------------------------

ORGANISM_KEYPOINTS: dict[str, list[str]] = {
    "daphnia": [
        "head",
        "eye",
        "antenna_l",
        "antenna_r",
        "carapace_anterior",
        "carapace_posterior",
        "thoracopod_1",
        "thoracopod_2",
        "thoracopod_3",
        "thoracopod_4",
        "tail_spine",
        "heart",
    ],
    "mussel": [
        "hinge",
        "valve_left_anterior",
        "valve_left_posterior",
        "valve_right_anterior",
        "valve_right_posterior",
        "siphon_inhalant",
        "siphon_exhalant",
        "foot",
    ],
    "fish": [
        "snout",
        "left_eye",
        "right_eye",
        "operculum_l",
        "operculum_r",
        "dorsal_fin_anterior",
        "dorsal_fin_posterior",
        "pectoral_l",
        "pectoral_r",
        "pelvic_l",
        "pelvic_r",
        "anal_fin",
        "caudal_peduncle",
        "caudal_fin_upper",
        "caudal_fin_lower",
        "lateral_line_1",
        "lateral_line_2",
        "lateral_line_3",
        "lateral_line_4",
        "lateral_line_5",
        "swim_bladder",
        "tail_tip",
    ],
}

# Feature velocity thresholds (mm/s) per species
_VELOCITY_THRESHOLDS: dict[str, dict[str, float]] = {
    "daphnia": {"active": 0.5, "freezing": 0.2},
    "fish": {"active": 1.0, "freezing": 0.3},
    "mussel": {"active": 0.1, "freezing": 0.02},
}


# ---------------------------------------------------------------------------
# SLEAP inference
# ---------------------------------------------------------------------------


def run_sleap_inference(
    video_path: str | Path,
    model_path: str | Path,
    species: str,
    output_dir: str | Path,
) -> Path:
    """Invoke SLEAP CLI to run pose estimation on a video.

    Calls ``sleap-track`` with the specified model to produce an H5
    predictions file.  If SLEAP is not installed, raises a helpful error
    with installation instructions.

    Parameters
    ----------
    video_path:
        Path to input video file.
    model_path:
        Path to a trained SLEAP model directory.
    species:
        Organism type (used for output naming).
    output_dir:
        Directory for the output predictions file.

    Returns
    -------
    Path to the SLEAP H5 predictions file.

    Raises
    ------
    FileNotFoundError
        If the video file does not exist.
    RuntimeError
        If SLEAP is not installed or the inference command fails.
    """
    video_path = Path(video_path)
    model_path = Path(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"SLEAP model not found: {model_path}")

    # Check that SLEAP CLI is available
    if shutil.which("sleap-track") is None:
        raise RuntimeError(
            "SLEAP is not installed or not on PATH. "
            "Install with: pip install sleap  (or conda install sleap). "
            "See https://sleap.ai/installation.html for details."
        )

    output_name = f"{video_path.stem}_{species}.predictions.slp"
    output_path = output_dir / output_name
    h5_path = output_path.with_suffix(".h5")

    # Run sleap-track
    cmd = [
        "sleap-track",
        str(video_path),
        "--model", str(model_path),
        "--output", str(output_path),
        "--verbosity", "rich",
    ]

    logger.info(f"Running SLEAP inference: {video_path.name} with {model_path.name}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
        )
        if result.returncode != 0:
            logger.error(f"SLEAP inference failed:\n{result.stderr}")
            raise RuntimeError(
                f"sleap-track exited with code {result.returncode}: {result.stderr}"
            )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"SLEAP inference timed out after 3600 seconds for {video_path}"
        )

    # Convert .slp to .h5 if needed
    if output_path.exists() and not h5_path.exists():
        convert_cmd = [
            "sleap-convert",
            str(output_path),
            "--format", "analysis",
            "--output", str(h5_path),
        ]
        try:
            subprocess.run(convert_cmd, capture_output=True, text=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning(
                "sleap-convert failed; returning .slp file. "
                "Convert manually with: sleap-convert --format analysis"
            )
            return output_path

    result_path = h5_path if h5_path.exists() else output_path
    logger.info(f"SLEAP predictions saved: {result_path}")
    return result_path


# ---------------------------------------------------------------------------
# Keypoint loading (multi-format)
# ---------------------------------------------------------------------------


def load_keypoints(
    path: str | Path,
    format: str = "sleap",
) -> np.ndarray:
    """Load keypoints from various tracking output formats.

    Parameters
    ----------
    path:
        Path to the tracking data file.
    format:
        One of ``"sleap"``, ``"deeplabcut"``, ``"csv"``, ``"numpy"``.

    Returns
    -------
    Array of shape ``(T, n_keypoints, 2)`` for XY coordinates, or
    ``(T, n_keypoints, 3)`` if confidence scores are included.

    Raises
    ------
    ValueError
        If the format is unrecognized or the file cannot be parsed.
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Keypoints file not found: {path}")

    if format == "sleap":
        return _load_sleap_h5(path)
    elif format == "deeplabcut":
        return _load_deeplabcut_h5(path)
    elif format == "csv":
        return _load_csv_keypoints(path)
    elif format == "numpy":
        return _load_numpy_keypoints(path)
    else:
        raise ValueError(
            f"Unknown keypoint format {format!r}. "
            "Supported: 'sleap', 'deeplabcut', 'csv', 'numpy'"
        )


def _load_sleap_h5(path: Path) -> np.ndarray:
    """Load keypoints from a SLEAP analysis H5 file.

    SLEAP H5 files store data under:
      - ``tracks``: shape ``(n_frames, n_tracks, n_nodes, 2)``
      - ``node_names``: keypoint names
      - ``track_names``: track identifiers

    We take the first track (single-organism assumption) or stack all
    tracks along the keypoint dimension for multi-organism recordings.
    """
    import h5py

    with h5py.File(str(path), "r") as f:
        if "tracks" in f:
            # SLEAP analysis format: (n_frames, n_tracks, n_nodes, 2)
            tracks = f["tracks"][:]
            # Transpose if stored as (2, n_nodes, n_tracks, n_frames) — old format
            if tracks.shape[0] == 2 and tracks.ndim == 4:
                tracks = np.transpose(tracks, (3, 2, 1, 0))
            if tracks.ndim == 4 and tracks.shape[1] == 1:
                # Single organism: squeeze track dimension
                return tracks[:, 0, :, :]
            elif tracks.ndim == 4:
                # Multi-organism: take first track
                logger.warning(
                    f"SLEAP file has {tracks.shape[1]} tracks; using first track only"
                )
                return tracks[:, 0, :, :]
            return tracks
        elif "pred_points" in f or "instance_peaks" in f:
            key = "pred_points" if "pred_points" in f else "instance_peaks"
            points = f[key][:]
            n_frames_key = "pred_point_frames" if "pred_point_frames" in f else None
            if points.ndim == 3:
                return points  # (T, n_keypoints, 2)
            raise ValueError(
                f"Unexpected shape for {key} in SLEAP H5: {points.shape}"
            )
        else:
            raise ValueError(
                f"Unrecognized SLEAP H5 structure. Available keys: {list(f.keys())}"
            )


def _load_deeplabcut_h5(path: Path) -> np.ndarray:
    """Load keypoints from a DeepLabCut H5 file.

    DLC H5 files use a MultiIndex DataFrame with levels
    (scorer, bodyparts, coords) where coords are ``x``, ``y``,
    ``likelihood``.
    """
    df = pd.read_hdf(str(path))

    if isinstance(df.columns, pd.MultiIndex):
        # Standard DLC format: (scorer, bodyparts, coords)
        # Get x, y, likelihood for each bodypart
        scorer = df.columns.get_level_values(0)[0]
        bodyparts = df.columns.get_level_values(1).unique()
        n_keypoints = len(bodyparts)
        n_frames = len(df)

        # Check for likelihood column
        coords = df.columns.get_level_values(-1).unique()
        has_likelihood = "likelihood" in coords
        n_coords = 3 if has_likelihood else 2

        keypoints = np.zeros((n_frames, n_keypoints, n_coords), dtype=np.float64)
        for j, bp in enumerate(bodyparts):
            keypoints[:, j, 0] = df[(scorer, bp, "x")].values
            keypoints[:, j, 1] = df[(scorer, bp, "y")].values
            if has_likelihood:
                keypoints[:, j, 2] = df[(scorer, bp, "likelihood")].values

        return keypoints.astype(np.float32)
    else:
        # Flat column format
        raise ValueError(
            "DeepLabCut H5 file does not have expected MultiIndex columns. "
            f"Columns: {list(df.columns[:10])}"
        )


def _load_csv_keypoints(path: Path) -> np.ndarray:
    """Load keypoints from a CSV file with frame, x, y columns.

    Supports two formats:
      1. Simple: columns ``(frame, x, y)`` — single keypoint (centroid)
      2. Multi-keypoint: columns ``(frame, kp0_x, kp0_y, kp1_x, kp1_y, ...)``
    """
    df = pd.read_csv(path)
    cols = [c.lower().strip() for c in df.columns]
    df.columns = cols

    # Detect format
    if "x" in cols and "y" in cols:
        # Simple single-keypoint format
        x = df["x"].values.astype(np.float32)
        y = df["y"].values.astype(np.float32)
        return np.stack([x, y], axis=-1)[:, np.newaxis, :]  # (T, 1, 2)

    # Multi-keypoint: look for *_x, *_y pairs
    x_cols = sorted([c for c in cols if c.endswith("_x")])
    y_cols = sorted([c for c in cols if c.endswith("_y")])

    if x_cols and len(x_cols) == len(y_cols):
        n_keypoints = len(x_cols)
        n_frames = len(df)
        keypoints = np.zeros((n_frames, n_keypoints, 2), dtype=np.float32)
        for j, (xc, yc) in enumerate(zip(x_cols, y_cols)):
            keypoints[:, j, 0] = df[xc].values
            keypoints[:, j, 1] = df[yc].values
        return keypoints

    # DaphTox format: columns might include distance, velocity, etc.
    # Try x_mm, y_mm
    if "x_mm" in cols and "y_mm" in cols:
        x = df["x_mm"].values.astype(np.float32)
        y = df["y_mm"].values.astype(np.float32)
        return np.stack([x, y], axis=-1)[:, np.newaxis, :]

    raise ValueError(
        f"Cannot parse CSV keypoint format. Columns: {list(df.columns[:20])}"
    )


def _load_numpy_keypoints(path: Path) -> np.ndarray:
    """Load keypoints from a numpy ``.npy`` or ``.npz`` file."""
    if path.suffix == ".npz":
        data = np.load(str(path))
        # Look for a keypoints-like array
        for key in ("keypoints", "tracks", "poses", "data"):
            if key in data:
                return data[key].astype(np.float32)
        # Fall back to the first array
        first_key = list(data.keys())[0]
        logger.warning(
            f"No standard key found in .npz; using first key: {first_key}"
        )
        return data[first_key].astype(np.float32)
    else:
        return np.load(str(path)).astype(np.float32)


# ---------------------------------------------------------------------------
# Behavioral feature extraction
# ---------------------------------------------------------------------------


def compute_behavioral_features(
    keypoints: np.ndarray,
    fps: float,
    species: str,
) -> pd.DataFrame:
    """Extract behavioral features from raw keypoint trajectories.

    Parameters
    ----------
    keypoints:
        Array of shape ``(T, n_keypoints, 2)`` or ``(T, n_keypoints, 3)``
        (with confidence as third coordinate).
    fps:
        Frames per second of the recording.
    species:
        Organism type — determines which features to compute.

    Returns
    -------
    DataFrame with one row per frame and columns for each computed
    behavioral feature:
      - ``velocity``: instantaneous centroid speed (mm/s)
      - ``acceleration``: change in velocity (mm/s^2)
      - ``turning_angle``: angular heading change per frame (degrees)
      - ``body_curvature``: body-axis curvature (fish only)
      - ``phototactic_index``: phototaxis correlation (daphnia only)
      - ``valve_gape``: valve opening distance (mussel only)
      - ``activity_level``: rolling fraction of time with velocity > threshold
      - ``freezing_episodes``: cumulative count of freezing episodes

    Raises
    ------
    ValueError
        If keypoints array has unexpected shape.
    """
    if keypoints.ndim < 2:
        raise ValueError(
            f"Expected keypoints of shape (T, n_keypoints, 2+), got {keypoints.shape}"
        )

    # Handle (T, 2) single-keypoint case
    if keypoints.ndim == 2:
        keypoints = keypoints[:, np.newaxis, :]

    # Strip confidence if present (T, K, 3) -> (T, K, 2)
    xy = keypoints[:, :, :2].astype(np.float64)
    n_frames = xy.shape[0]
    dt = 1.0 / fps

    thresholds = _VELOCITY_THRESHOLDS.get(species, _VELOCITY_THRESHOLDS["fish"])

    # Centroid
    centroid = np.nanmean(xy, axis=1)  # (T, 2)

    # Velocity (instantaneous centroid speed in mm/s)
    displacement = np.diff(centroid, axis=0)  # (T-1, 2)
    speed = np.linalg.norm(displacement, axis=1) / dt  # (T-1,)
    velocity = np.concatenate([[0.0], speed])

    # Acceleration (mm/s^2)
    accel = np.diff(velocity) / dt
    acceleration = np.concatenate([[0.0], accel])

    # Turning angle (degrees)
    heading = np.arctan2(displacement[:, 1], displacement[:, 0])
    turn = np.diff(heading)
    # Wrap to [-pi, pi]
    turn = (turn + np.pi) % (2 * np.pi) - np.pi
    turning_angle_deg = np.degrees(turn)
    turning_angle = np.concatenate([[0.0], [0.0], turning_angle_deg])

    # Activity level (rolling 1-second window)
    window = max(1, int(fps))
    active_mask = (velocity > thresholds["active"]).astype(np.float64)
    activity_level = pd.Series(active_mask).rolling(
        window=window, min_periods=1
    ).mean().values

    # Freezing episodes (velocity < threshold for > 2 seconds)
    freeze_threshold = thresholds["freezing"]
    min_freeze_frames = int(2.0 * fps)
    frozen = velocity < freeze_threshold
    freezing_count = np.zeros(n_frames, dtype=np.int32)
    episode_count = 0
    consecutive = 0
    for i in range(n_frames):
        if frozen[i]:
            consecutive += 1
            if consecutive == min_freeze_frames:
                episode_count += 1
        else:
            consecutive = 0
        freezing_count[i] = episode_count

    features: dict[str, np.ndarray] = {
        "velocity": velocity,
        "acceleration": acceleration,
        "turning_angle": turning_angle,
        "activity_level": activity_level,
        "freezing_episodes": freezing_count.astype(np.float64),
    }

    # Species-specific features
    if species == "fish":
        features["body_curvature"] = _compute_body_curvature(xy, species)

    if species == "daphnia":
        features["phototactic_index"] = _compute_phototactic_index(
            displacement, fps
        )

    if species == "mussel":
        features["valve_gape"] = _compute_valve_gape(xy, species)

    return pd.DataFrame(features)


def _compute_body_curvature(
    xy: np.ndarray,
    species: str,
) -> np.ndarray:
    """Compute body-axis curvature for fish from keypoints along the spine.

    Uses the keypoints along the body midline (snout -> caudal_peduncle ->
    tail_tip) to fit curvature as the reciprocal of the radius of the
    osculating circle at the body center.

    Parameters
    ----------
    xy:
        Keypoints array of shape ``(T, n_keypoints, 2)``.
    species:
        Organism type.

    Returns
    -------
    Array of shape ``(T,)`` with curvature values (1/mm).
    """
    n_frames = xy.shape[0]
    curvature = np.zeros(n_frames, dtype=np.float64)

    kp_names = ORGANISM_KEYPOINTS.get(species, [])
    if species == "fish" and len(kp_names) == xy.shape[1]:
        # Use snout (0), caudal_peduncle (12), tail_tip (21) for curvature
        idx_snout = 0
        idx_mid = 12  # caudal_peduncle
        idx_tail = 21  # tail_tip

        for i in range(n_frames):
            p1 = xy[i, idx_snout]
            p2 = xy[i, idx_mid]
            p3 = xy[i, idx_tail]

            # Menger curvature: 4 * area(triangle) / (|p1-p2| * |p2-p3| * |p3-p1|)
            area = 0.5 * abs(
                (p2[0] - p1[0]) * (p3[1] - p1[1])
                - (p3[0] - p1[0]) * (p2[1] - p1[1])
            )
            d12 = np.linalg.norm(p2 - p1)
            d23 = np.linalg.norm(p3 - p2)
            d31 = np.linalg.norm(p1 - p3)
            denom = d12 * d23 * d31
            if denom > 1e-9:
                curvature[i] = 4.0 * area / denom

    return curvature


def _compute_phototactic_index(
    displacement: np.ndarray,
    fps: float,
) -> np.ndarray:
    """Compute phototactic index for Daphnia.

    Measures the correlation between movement direction and the
    vertical axis (positive y = toward typical overhead light source).
    A rolling window of 1 second is used.

    Parameters
    ----------
    displacement:
        Frame-to-frame centroid displacement, shape ``(T-1, 2)``.
    fps:
        Frames per second.

    Returns
    -------
    Array of shape ``(T,)`` with phototactic index in [-1, 1].
    """
    # Normalize displacement to unit vectors
    norms = np.linalg.norm(displacement, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1.0, norms)
    unit_disp = displacement / norms

    # Dot product with upward direction [0, 1] = y-component
    phototaxis_raw = unit_disp[:, 1]
    phototaxis = np.concatenate([[0.0], phototaxis_raw])

    # Rolling mean over 1 second
    window = max(1, int(fps))
    phototaxis_smooth = (
        pd.Series(phototaxis).rolling(window=window, min_periods=1).mean().values
    )
    return phototaxis_smooth


def _compute_valve_gape(
    xy: np.ndarray,
    species: str,
) -> np.ndarray:
    """Compute valve gape distance for mussels.

    Distance between left and right valve anterior keypoints.

    Parameters
    ----------
    xy:
        Keypoints array of shape ``(T, n_keypoints, 2)``.
    species:
        Organism type.

    Returns
    -------
    Array of shape ``(T,)`` with gape distance (mm).
    """
    n_frames = xy.shape[0]
    gape = np.zeros(n_frames, dtype=np.float64)

    kp_names = ORGANISM_KEYPOINTS.get(species, [])
    if species == "mussel" and len(kp_names) == xy.shape[1]:
        # valve_left_anterior (1) and valve_right_anterior (3)
        idx_left = 1
        idx_right = 3
        for i in range(n_frames):
            gape[i] = np.linalg.norm(xy[i, idx_left] - xy[i, idx_right])

    return gape


# ---------------------------------------------------------------------------
# Temporal downsampling
# ---------------------------------------------------------------------------


def downsample_features(
    features_df: pd.DataFrame,
    target_fps: float = 1.0,
    source_fps: float = 30.0,
) -> pd.DataFrame:
    """Downsample behavioral features using summary statistics.

    For each non-overlapping window of ``source_fps / target_fps``
    frames, computes mean, std, max, min, and 95th percentile for
    every feature column.

    Parameters
    ----------
    features_df:
        DataFrame with one row per source frame.
    target_fps:
        Target frame rate (Hz).
    source_fps:
        Source frame rate (Hz).

    Returns
    -------
    DataFrame with one row per target-fps time bin, columns named
    ``{feature}_{stat}`` (e.g. ``velocity_mean``, ``velocity_std``).
    """
    window_size = max(1, int(round(source_fps / target_fps)))
    n_frames = len(features_df)
    n_windows = n_frames // window_size

    if n_windows == 0:
        logger.warning(
            f"Not enough frames ({n_frames}) for even one window "
            f"(size={window_size}). Returning single-row summary."
        )
        n_windows = 1
        window_size = n_frames

    records: list[dict[str, float]] = []
    for i in range(n_windows):
        start = i * window_size
        end = start + window_size
        chunk = features_df.iloc[start:end]
        row: dict[str, float] = {}
        for col in chunk.columns:
            vals = chunk[col].values.astype(np.float64)
            valid = vals[~np.isnan(vals)]
            if len(valid) == 0:
                row[f"{col}_mean"] = np.nan
                row[f"{col}_std"] = np.nan
                row[f"{col}_max"] = np.nan
                row[f"{col}_min"] = np.nan
                row[f"{col}_p95"] = np.nan
            else:
                row[f"{col}_mean"] = float(np.mean(valid))
                row[f"{col}_std"] = float(np.std(valid))
                row[f"{col}_max"] = float(np.max(valid))
                row[f"{col}_min"] = float(np.min(valid))
                row[f"{col}_p95"] = float(np.percentile(valid, 95))
        records.append(row)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Trajectory normalization
# ---------------------------------------------------------------------------


def normalize_trajectory(
    keypoints: np.ndarray,
    method: str = "center_scale",
) -> np.ndarray:
    """Normalize a keypoint trajectory.

    Parameters
    ----------
    keypoints:
        Array of shape ``(T, n_keypoints, 2)``.
    method:
        Normalization method:
          - ``"center_scale"``: subtract centroid and scale to unit variance
          - ``"center"``: subtract centroid only
          - ``"heading_align"``: center, scale, and rotate so that initial
            heading is along the positive x-axis

    Returns
    -------
    Normalized keypoints array with the same shape.

    Raises
    ------
    ValueError
        If the method is unrecognized.
    """
    if method not in ("center_scale", "center", "heading_align"):
        raise ValueError(
            f"Unknown normalization method {method!r}. "
            "Supported: 'center_scale', 'center', 'heading_align'"
        )

    kp = keypoints.astype(np.float64).copy()

    # Center: subtract per-frame centroid
    centroid = np.nanmean(kp, axis=1, keepdims=True)  # (T, 1, 2)
    kp -= centroid

    if method == "center":
        return kp.astype(np.float32)

    # Scale to unit variance across all coordinates
    flat = kp.reshape(-1, 2)
    valid = flat[~np.isnan(flat).any(axis=1)]
    if len(valid) > 0:
        std = np.std(valid)
        if std > 1e-9:
            kp /= std

    if method == "center_scale":
        return kp.astype(np.float32)

    # Heading alignment: rotate so initial heading is along +x
    if method == "heading_align":
        # Compute initial heading from first few frames
        centroid_flat = np.nanmean(keypoints[:, :, :2], axis=1)
        # Find first non-zero displacement
        for i in range(min(10, len(centroid_flat) - 1)):
            disp = centroid_flat[i + 1] - centroid_flat[i]
            if np.linalg.norm(disp) > 1e-9:
                angle = -np.arctan2(disp[1], disp[0])
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                for t in range(kp.shape[0]):
                    for k in range(kp.shape[1]):
                        if not np.any(np.isnan(kp[t, k])):
                            kp[t, k] = rot @ kp[t, k]
                break

        return kp.astype(np.float32)

    return kp.astype(np.float32)


# ---------------------------------------------------------------------------
# Single-recording pipeline
# ---------------------------------------------------------------------------


def preprocess_recording(
    video_or_keypoints_path: str | Path,
    species: str,
    output_dir: str | Path,
    sleap_model: str | Path | None = None,
    fps: float = 30.0,
    keypoint_format: str = "sleap",
    normalize_method: str = "center_scale",
    target_fps: float = 1.0,
) -> dict[str, Any]:
    """Full preprocessing pipeline for a single recording.

    Steps:
      1. Load keypoints (or run SLEAP inference if a video is provided)
      2. Compute behavioral features
      3. Downsample features to target fps
      4. Normalize trajectory
      5. Save outputs

    Parameters
    ----------
    video_or_keypoints_path:
        Path to a video file (requires *sleap_model*) or a pre-tracked
        keypoints file (H5, CSV, NPY/NPZ).
    species:
        Organism type.
    output_dir:
        Directory for output files.
    sleap_model:
        Path to a SLEAP model directory.  Required if *video_or_keypoints_path*
        is a video file.
    fps:
        Frames per second of the recording.
    keypoint_format:
        Format of the keypoints file (``"sleap"``, ``"deeplabcut"``,
        ``"csv"``, ``"numpy"``).  Ignored if a video file is provided.
    normalize_method:
        Trajectory normalization method.
    target_fps:
        Target frame rate for downsampled features.

    Returns
    -------
    Dict with keys:
      - ``"keypoints_path"``: path to saved normalized keypoints
      - ``"features_path"``: path to saved full-rate features
      - ``"downsampled_path"``: path to saved downsampled features
      - ``"n_frames"``: number of frames
      - ``"n_keypoints"``: number of keypoints
      - ``"species"``: organism type
    """
    input_path = Path(video_or_keypoints_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem

    # Step 1: Load or estimate keypoints
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    if input_path.suffix.lower() in video_extensions:
        if sleap_model is None:
            raise ValueError(
                "sleap_model must be provided when input is a video file"
            )
        kp_path = run_sleap_inference(
            input_path, sleap_model, species, output_dir
        )
        keypoints = load_keypoints(kp_path, format="sleap")
    else:
        keypoints = load_keypoints(input_path, format=keypoint_format)

    logger.info(
        f"Loaded keypoints: {keypoints.shape} from {input_path.name}"
    )

    # Step 2: Compute behavioral features
    features_df = compute_behavioral_features(keypoints, fps, species)

    # Step 3: Downsample features
    downsampled_df = downsample_features(features_df, target_fps, fps)

    # Step 4: Normalize trajectory
    kp_norm = normalize_trajectory(keypoints, method=normalize_method)

    # Step 5: Save outputs
    kp_out = output_dir / f"{stem}_keypoints.npy"
    np.save(str(kp_out), kp_norm)

    feat_out = output_dir / f"{stem}_features.parquet"
    features_df.to_parquet(str(feat_out))

    ds_out = output_dir / f"{stem}_features_downsampled.parquet"
    downsampled_df.to_parquet(str(ds_out))

    logger.info(
        f"Preprocessed {stem}: {keypoints.shape[0]} frames, "
        f"{keypoints.shape[1]} keypoints, "
        f"{len(downsampled_df)} downsampled rows"
    )

    return {
        "keypoints_path": kp_out,
        "features_path": feat_out,
        "downsampled_path": ds_out,
        "n_frames": keypoints.shape[0],
        "n_keypoints": keypoints.shape[1],
        "species": species,
    }


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------


def preprocess_all_recordings(
    input_dir: str | Path,
    species: str,
    output_dir: str | Path,
    fps: float = 30.0,
    keypoint_format: str = "sleap",
    sleap_model: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Batch preprocess all recordings in a directory.

    Processes all recognized files (H5, CSV, NPY/NPZ, video) in the
    input directory using :func:`preprocess_recording`.

    Parameters
    ----------
    input_dir:
        Directory containing recording files.
    species:
        Organism type for all recordings in this directory.
    output_dir:
        Directory for output files.
    fps:
        Frames per second for all recordings.
    keypoint_format:
        Default keypoint format for non-video files.
    sleap_model:
        Optional SLEAP model path for video files.

    Returns
    -------
    Dict mapping filename stem -> preprocessing result dict.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    extensions = {".h5", ".hdf5", ".csv", ".npy", ".npz", ".mp4", ".avi", ".mov"}
    files = sorted(
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in extensions
    )

    if not files:
        logger.warning(f"No recognized files found in {input_dir}")
        return {}

    results: dict[str, dict[str, Any]] = {}
    progress = make_progress()
    with progress:
        task = progress.add_task("Preprocessing recordings", total=len(files))
        for f in files:
            try:
                fmt = keypoint_format
                if f.suffix.lower() in (".csv",):
                    fmt = "csv"
                elif f.suffix.lower() in (".npy", ".npz"):
                    fmt = "numpy"
                elif f.suffix.lower() in (".h5", ".hdf5"):
                    # Attempt auto-detection between SLEAP and DLC
                    fmt = keypoint_format

                result = preprocess_recording(
                    f,
                    species=species,
                    output_dir=output_dir,
                    sleap_model=sleap_model,
                    fps=fps,
                    keypoint_format=fmt,
                )
                results[f.stem] = result
            except Exception as exc:
                logger.warning(f"Failed to preprocess {f.name}: {exc}")
            progress.advance(task)

    total_frames = sum(r["n_frames"] for r in results.values())
    logger.info(
        f"Preprocessed {len(results)}/{len(files)} recordings, "
        f"{total_frames} total frames -> {output_dir}"
    )
    return results
