"""
Behavioral tracking data download for SENTINEL BioMotion encoder.

Data sources:
  1. Published Daphnia behavioral tracking datasets
  2. Fish tracking datasets (zebrafish, fathead minnow)
  3. Mussel valve-gaping datasets
  4. Simulated trajectory generation for pretraining
"""

from __future__ import annotations

import json
import math
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from sentinel.utils.logging import get_logger, make_progress

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Known data sources (Zenodo / figshare DOIs)
# ---------------------------------------------------------------------------

# Zenodo record for Daphnia magna swimming trajectories from Simao et al.
# (ecotoxicology behavioral assay, 25 fps XY tracking).
_DAPHNIA_ZENODO_RECORDS: list[dict[str, str]] = [
    {
        "name": "daphnia_simao_2019",
        "doi": "10.5281/zenodo.3247400",
        "url": "https://zenodo.org/api/records/3247400/files",
        "description": (
            "Daphnia magna swimming trajectories under control and "
            "contaminant-exposed conditions (Simao et al. 2019). "
            "XY coordinates at 25 fps."
        ),
    },
    {
        "name": "daphnia_noldus_example",
        "doi": "10.5281/zenodo.7675234",
        "url": "https://zenodo.org/api/records/7675234/files",
        "description": (
            "DaphTox-format Daphnia tracking export with distance, velocity, "
            "and turn angle columns."
        ),
    },
]

_FISH_ZENODO_RECORDS: list[dict[str, str]] = [
    {
        "name": "zebrafish_dlc_locomotion",
        "doi": "10.5281/zenodo.4088310",
        "url": "https://zenodo.org/api/records/4088310/files",
        "description": (
            "Zebrafish larvae locomotion tracked with DeepLabCut. "
            "H5 files with multi-point body keypoints at 30 fps."
        ),
    },
    {
        "name": "zebrafish_openfield",
        "doi": "10.5281/zenodo.6539987",
        "url": "https://zenodo.org/api/records/6539987/files",
        "description": (
            "Adult zebrafish open-field behavioral assay. CSV with "
            "(frame, x, y, angle) columns at 30 fps."
        ),
    },
]

_MUSSEL_DATA_SOURCES: list[dict[str, str]] = [
    {
        "name": "mosselmonitor_reference",
        "description": (
            "Mussel valve-gaping data from Mosselmonitor-type biosensors. "
            "Binary open/close or analog gap measurements at ~1 Hz. "
            "Data must be obtained from instrument operators or published "
            "supplementary materials. See: de Zwart et al. (2018) "
            "doi: 10.1016/j.ecolind.2018.02.022."
        ),
        "manual_download": True,
    },
]

# ---------------------------------------------------------------------------
# Daphnia trajectory parameters (for synthetic generation)
# ---------------------------------------------------------------------------

_DAPHNIA_PARAMS: dict[str, Any] = {
    "speed_mean_mm_s": 3.5,
    "speed_std_mm_s": 1.2,
    "turn_std_deg": 25.0,
    "phototaxis_strength": 0.15,
    "arena_radius_mm": 35.0,
}

_FISH_PARAMS: dict[str, Any] = {
    "speed_mean_mm_s": 8.0,
    "speed_std_mm_s": 3.0,
    "turn_std_deg": 15.0,
    "burst_probability": 0.02,
    "burst_speed_factor": 3.0,
    "schooling_attraction": 0.05,
    "arena_length_mm": 200.0,
    "arena_width_mm": 100.0,
}

_MUSSEL_PARAMS: dict[str, Any] = {
    "gape_mean_mm": 2.5,
    "gape_std_mm": 0.8,
    "open_fraction": 0.85,
    "cycle_period_s": 600.0,
}

# Anomaly definitions for each species
_ANOMALY_TYPES: dict[str, list[str]] = {
    "daphnia": [
        "reduced_velocity",
        "erratic",
        "freezing",
        "hyperactive",
        "loss_of_phototaxis",
    ],
    "fish": [
        "reduced_velocity",
        "erratic",
        "freezing",
        "hyperactive",
        "loss_of_schooling",
    ],
    "mussel": [
        "prolonged_closure",
        "erratic_gaping",
        "reduced_gape",
        "hyperactive_gaping",
    ],
}


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def _download_zenodo_record(
    record_url: str,
    output_dir: Path,
    record_name: str,
) -> list[Path]:
    """Download all files from a Zenodo record via its API.

    Parameters
    ----------
    record_url:
        Zenodo files API endpoint (e.g. ``https://zenodo.org/api/records/<id>/files``).
    output_dir:
        Local directory to save files into.
    record_name:
        Human-readable name for progress display.

    Returns
    -------
    List of paths to downloaded files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []

    try:
        with urllib.request.urlopen(record_url, timeout=30) as resp:
            file_list = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        logger.warning(
            f"Could not fetch file list for {record_name} from {record_url}: {exc}. "
            "The dataset may require manual download."
        )
        return downloaded

    entries = file_list.get("entries", file_list) if isinstance(file_list, dict) else file_list
    if not isinstance(entries, list):
        logger.warning(f"Unexpected Zenodo API response for {record_name}")
        return downloaded

    progress = make_progress()
    with progress:
        task = progress.add_task(
            f"Downloading {record_name}", total=len(entries)
        )
        for entry in entries:
            filename = entry.get("key") or entry.get("filename", "unknown")
            download_link = entry.get("links", {}).get("content") or entry.get("links", {}).get("self")
            if not download_link:
                progress.advance(task)
                continue

            dest = output_dir / filename
            if dest.exists():
                logger.info(f"Already downloaded: {dest}")
                downloaded.append(dest)
                progress.advance(task)
                continue

            try:
                urllib.request.urlretrieve(download_link, str(dest))
                downloaded.append(dest)
                logger.info(f"Downloaded {filename} -> {dest}")
            except Exception as exc:
                logger.warning(f"Failed to download {filename}: {exc}")
            progress.advance(task)

    return downloaded


# ---------------------------------------------------------------------------
# Public download functions
# ---------------------------------------------------------------------------


def download_daphnia_datasets(
    output_dir: str | Path = "data/behavioral/daphnia",
) -> list[Path]:
    """Download published Daphnia magna swimming trajectory datasets.

    Attempts to download from Zenodo records containing XY coordinate
    time-series data at 25-30 fps.  If Zenodo is unreachable, logs
    instructions for manual download.

    Parameters
    ----------
    output_dir:
        Directory for downloaded files.

    Returns
    -------
    List of paths to downloaded data files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_files: list[Path] = []

    for record in _DAPHNIA_ZENODO_RECORDS:
        logger.info(f"Fetching Daphnia dataset: {record['name']} (DOI: {record['doi']})")
        record_dir = output_dir / record["name"]
        files = _download_zenodo_record(
            record["url"], record_dir, record["name"]
        )
        all_files.extend(files)

    if not all_files:
        logger.warning(
            "No Daphnia datasets could be downloaded automatically. "
            "Manual download instructions:\n"
            "  1. Visit https://zenodo.org and search for 'Daphnia behavior tracking'\n"
            "  2. Download CSV/XLS files with XY trajectory coordinates\n"
            f"  3. Place files in: {output_dir.resolve()}\n"
            "  Expected format: CSV with columns (frame, x_mm, y_mm) or "
            "DaphTox export format."
        )

    logger.info(f"Daphnia datasets: {len(all_files)} files -> {output_dir}")
    return all_files


def download_fish_tracking(
    output_dir: str | Path = "data/behavioral/fish",
) -> list[Path]:
    """Download published zebrafish/fish behavioral tracking datasets.

    Targets DeepLabCut H5, SLEAP H5, or CSV trajectory files from
    Zenodo.  Common formats include columns (frame, x, y, ...) at
    30 fps.

    Parameters
    ----------
    output_dir:
        Directory for downloaded files.

    Returns
    -------
    List of paths to downloaded data files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_files: list[Path] = []

    for record in _FISH_ZENODO_RECORDS:
        logger.info(f"Fetching fish dataset: {record['name']} (DOI: {record['doi']})")
        record_dir = output_dir / record["name"]
        files = _download_zenodo_record(
            record["url"], record_dir, record["name"]
        )
        all_files.extend(files)

    if not all_files:
        logger.warning(
            "No fish tracking datasets could be downloaded automatically. "
            "Manual download instructions:\n"
            "  1. Visit https://zenodo.org and search for "
            "'zebrafish behavior tracking DeepLabCut'\n"
            "  2. Download H5 or CSV tracking files\n"
            f"  3. Place files in: {output_dir.resolve()}\n"
            "  Expected formats: DeepLabCut H5, SLEAP H5, or CSV with "
            "(frame, x, y) columns."
        )

    logger.info(f"Fish tracking datasets: {len(all_files)} files -> {output_dir}")
    return all_files


def download_mussel_valve_data(
    output_dir: str | Path = "data/behavioral/mussel",
) -> list[Path]:
    """Download or describe mussel valve-gaping sensor data.

    Mussel valve-gaping datasets (Mosselmonitor format or similar) are
    typically not available for direct download.  This function checks
    for locally placed files and provides acquisition instructions.

    The expected data format is a CSV or binary file with columns:
      - timestamp (ISO 8601 or Unix seconds)
      - valve_gape_mm (analog gap measurement) or valve_state (0/1 binary)
      - Sampling rate: ~1 Hz

    Parameters
    ----------
    output_dir:
        Directory where the user should place mussel data files and
        where processed outputs will be written.

    Returns
    -------
    List of paths to any existing data files found.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(
        list(output_dir.glob("*.csv"))
        + list(output_dir.glob("*.tsv"))
        + list(output_dir.glob("*.h5"))
        + list(output_dir.glob("*.parquet"))
    )

    if existing:
        logger.info(
            f"Found {len(existing)} existing mussel data files in {output_dir}"
        )
        return existing

    logger.warning(
        "Mussel valve-gaping data requires manual acquisition.\n"
        "Sources:\n"
        "  - Mosselmonitor / Delta Consult instrument exports\n"
        "  - Published supplementary data (see de Zwart et al. 2018, "
        "doi:10.1016/j.ecolind.2018.02.022)\n"
        "  - WRRL biomonitoring station data exports\n"
        f"Place CSV/H5 files in: {output_dir.resolve()}\n"
        "Expected format: CSV with columns (timestamp, valve_gape_mm) "
        "or (timestamp, valve_state) at ~1 Hz."
    )
    return []


# ---------------------------------------------------------------------------
# Synthetic trajectory generation
# ---------------------------------------------------------------------------


def _generate_daphnia_trajectory(
    duration_s: float,
    fps: float,
    anomalous: bool = False,
    anomaly_type: str = "none",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a single synthetic Daphnia trajectory.

    Returns array of shape ``(T, 12, 2)`` representing 12 keypoints
    over T frames.  Only the centroid is independently simulated; other
    keypoints are offset from the centroid with small jitter.

    Parameters
    ----------
    duration_s:
        Duration in seconds.
    fps:
        Frames per second.
    anomalous:
        Whether to inject anomalous behavior.
    anomaly_type:
        Type of anomaly to inject if *anomalous* is True.
    rng:
        Numpy random generator instance.

    Returns
    -------
    Keypoints array of shape ``(T, 12, 2)``.
    """
    rng = rng or np.random.default_rng()
    params = _DAPHNIA_PARAMS
    n_frames = int(duration_s * fps)
    dt = 1.0 / fps

    # Centroid random walk
    speed_mean = params["speed_mean_mm_s"]
    speed_std = params["speed_std_mm_s"]
    turn_std = np.radians(params["turn_std_deg"])
    arena_r = params["arena_radius_mm"]
    phototaxis = params["phototaxis_strength"]

    # Apply anomaly modifications
    if anomalous:
        if anomaly_type == "reduced_velocity":
            speed_mean *= 0.2
            speed_std *= 0.3
        elif anomaly_type == "erratic":
            turn_std *= 3.0
            speed_std *= 2.0
        elif anomaly_type == "freezing":
            pass  # handled per-frame below
        elif anomaly_type == "hyperactive":
            speed_mean *= 2.5
            speed_std *= 1.5
        elif anomaly_type == "loss_of_phototaxis":
            phototaxis = 0.0

    # Simulate centroid path
    positions = np.zeros((n_frames, 2))
    heading = rng.uniform(0, 2 * math.pi)
    pos = np.array([0.0, 0.0])

    frozen = False
    freeze_counter = 0

    for i in range(n_frames):
        # Freezing episodes for anomalous trajectories
        if anomalous and anomaly_type == "freezing":
            if not frozen and rng.random() < 0.005:
                frozen = True
                freeze_counter = int(rng.uniform(2.0, 8.0) * fps)
            if frozen:
                freeze_counter -= 1
                if freeze_counter <= 0:
                    frozen = False
                positions[i] = pos + rng.normal(0, 0.01, size=2)
                continue

        speed = max(0.0, rng.normal(speed_mean, speed_std))
        turn = rng.normal(0, turn_std)

        # Phototaxis: bias heading upward (positive y = toward light)
        if phototaxis > 0:
            to_light = math.atan2(1.0, 0.0) - heading
            # Wrap to [-pi, pi]
            to_light = (to_light + math.pi) % (2 * math.pi) - math.pi
            turn += phototaxis * to_light

        heading += turn

        dx = speed * dt * math.cos(heading)
        dy = speed * dt * math.sin(heading)
        pos = pos + np.array([dx, dy])

        # Arena boundary reflection
        dist = np.linalg.norm(pos)
        if dist > arena_r:
            pos = pos * (arena_r / dist) * 0.95
            heading += math.pi + rng.normal(0, 0.3)

        positions[i] = pos

    # Generate 12 keypoints as offsets from centroid
    n_keypoints = 12
    keypoints = np.zeros((n_frames, n_keypoints, 2))

    # Body-relative offsets in mm (approximate Daphnia anatomy ~2mm body)
    offsets = np.array([
        [0.8, 0.0],    # head
        [0.6, 0.15],   # eye
        [0.7, 0.3],    # antenna_l
        [0.7, -0.3],   # antenna_r
        [0.3, 0.0],    # carapace_anterior
        [-0.3, 0.0],   # carapace_posterior
        [0.0, 0.2],    # thoracopod_1
        [-0.05, 0.2],  # thoracopod_2
        [-0.10, 0.2],  # thoracopod_3
        [-0.15, 0.2],  # thoracopod_4
        [-0.7, 0.0],   # tail_spine
        [0.2, 0.05],   # heart
    ])

    for i in range(n_frames):
        heading_i = math.atan2(
            positions[min(i + 1, n_frames - 1), 1] - positions[max(i - 1, 0), 1],
            positions[min(i + 1, n_frames - 1), 0] - positions[max(i - 1, 0), 0],
        )
        cos_h, sin_h = math.cos(heading_i), math.sin(heading_i)
        rot = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
        rotated_offsets = offsets @ rot.T
        jitter = rng.normal(0, 0.02, size=(n_keypoints, 2))
        keypoints[i] = positions[i] + rotated_offsets + jitter

    return keypoints


def _generate_fish_trajectory(
    duration_s: float,
    fps: float,
    anomalous: bool = False,
    anomaly_type: str = "none",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a single synthetic fish trajectory.

    Returns array of shape ``(T, 22, 2)`` representing 22 keypoints.

    Parameters
    ----------
    duration_s:
        Duration in seconds.
    fps:
        Frames per second.
    anomalous:
        Whether to inject anomalous behavior.
    anomaly_type:
        Type of anomaly to inject.
    rng:
        Numpy random generator instance.

    Returns
    -------
    Keypoints array of shape ``(T, 22, 2)``.
    """
    rng = rng or np.random.default_rng()
    params = _FISH_PARAMS
    n_frames = int(duration_s * fps)
    dt = 1.0 / fps

    speed_mean = params["speed_mean_mm_s"]
    speed_std = params["speed_std_mm_s"]
    turn_std = np.radians(params["turn_std_deg"])
    burst_prob = params["burst_probability"]
    burst_factor = params["burst_speed_factor"]
    arena_l = params["arena_length_mm"]
    arena_w = params["arena_width_mm"]

    if anomalous:
        if anomaly_type == "reduced_velocity":
            speed_mean *= 0.15
            speed_std *= 0.2
        elif anomaly_type == "erratic":
            turn_std *= 4.0
            speed_std *= 2.5
        elif anomaly_type == "freezing":
            pass  # handled per-frame
        elif anomaly_type == "hyperactive":
            speed_mean *= 2.5
            burst_prob *= 3.0

    positions = np.zeros((n_frames, 2))
    heading = rng.uniform(0, 2 * math.pi)
    pos = np.array([arena_l / 2, arena_w / 2])

    frozen = False
    freeze_counter = 0

    for i in range(n_frames):
        if anomalous and anomaly_type == "freezing":
            if not frozen and rng.random() < 0.003:
                frozen = True
                freeze_counter = int(rng.uniform(3.0, 15.0) * fps)
            if frozen:
                freeze_counter -= 1
                if freeze_counter <= 0:
                    frozen = False
                positions[i] = pos + rng.normal(0, 0.02, size=2)
                continue

        speed = max(0.0, rng.normal(speed_mean, speed_std))
        if rng.random() < burst_prob:
            speed *= burst_factor

        turn = rng.normal(0, turn_std)
        heading += turn

        dx = speed * dt * math.cos(heading)
        dy = speed * dt * math.sin(heading)
        pos = pos + np.array([dx, dy])

        # Rectangular arena boundary reflection
        if pos[0] < 0 or pos[0] > arena_l:
            pos[0] = np.clip(pos[0], 0, arena_l)
            heading = math.pi - heading
        if pos[1] < 0 or pos[1] > arena_w:
            pos[1] = np.clip(pos[1], 0, arena_w)
            heading = -heading

        positions[i] = pos

    # Generate 22 keypoints along fish body axis
    n_keypoints = 22
    keypoints = np.zeros((n_frames, n_keypoints, 2))

    # Body-relative offsets in mm (approximate adult zebrafish ~30mm)
    # Arranged from snout to tail along the body axis
    offsets = np.array([
        [15.0, 0.0],    # snout
        [12.0, 2.0],    # left_eye
        [12.0, -2.0],   # right_eye
        [9.0, 3.0],     # operculum_l
        [9.0, -3.0],    # operculum_r
        [5.0, 4.0],     # dorsal_fin_anterior
        [-2.0, 3.5],    # dorsal_fin_posterior
        [7.0, 5.0],     # pectoral_l
        [7.0, -5.0],    # pectoral_r
        [2.0, 3.0],     # pelvic_l
        [2.0, -3.0],    # pelvic_r
        [-5.0, 2.5],    # anal_fin
        [-8.0, 0.0],    # caudal_peduncle
        [-12.0, 2.5],   # caudal_fin_upper
        [-12.0, -2.5],  # caudal_fin_lower
        [6.0, 1.0],     # lateral_line_1
        [3.0, 1.0],     # lateral_line_2
        [0.0, 1.0],     # lateral_line_3
        [-3.0, 1.0],    # lateral_line_4
        [-6.0, 1.0],    # lateral_line_5
        [4.0, 0.0],     # swim_bladder
        [-13.0, 0.0],   # tail_tip
    ])

    for i in range(n_frames):
        heading_i = math.atan2(
            positions[min(i + 1, n_frames - 1), 1] - positions[max(i - 1, 0), 1],
            positions[min(i + 1, n_frames - 1), 0] - positions[max(i - 1, 0), 0],
        )
        cos_h, sin_h = math.cos(heading_i), math.sin(heading_i)
        rot = np.array([[cos_h, -sin_h], [sin_h, cos_h]])

        # Add body curvature wave for fish
        body_wave = 0.5 * math.sin(2 * math.pi * i / (fps * 0.5))
        wave_offsets = offsets.copy()
        for j in range(n_keypoints):
            progress_along_body = (offsets[j, 0] - offsets[:, 0].min()) / max(
                offsets[:, 0].max() - offsets[:, 0].min(), 1e-6
            )
            wave_offsets[j, 1] += body_wave * (1.0 - progress_along_body) * 1.5

        rotated = wave_offsets @ rot.T
        jitter = rng.normal(0, 0.1, size=(n_keypoints, 2))
        keypoints[i] = positions[i] + rotated + jitter

    return keypoints


def _generate_mussel_trajectory(
    duration_s: float,
    fps: float,
    anomalous: bool = False,
    anomaly_type: str = "none",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a single synthetic mussel valve-gaping trajectory.

    Returns array of shape ``(T, 8, 2)`` representing 8 keypoints.
    The primary dynamic is valve opening/closing.

    Parameters
    ----------
    duration_s:
        Duration in seconds.
    fps:
        Frames per second.
    anomalous:
        Whether to inject anomalous behavior.
    anomaly_type:
        Type of anomaly to inject.
    rng:
        Numpy random generator instance.

    Returns
    -------
    Keypoints array of shape ``(T, 8, 2)``.
    """
    rng = rng or np.random.default_rng()
    params = _MUSSEL_PARAMS
    n_frames = int(duration_s * fps)

    gape_mean = params["gape_mean_mm"]
    gape_std = params["gape_std_mm"]
    open_frac = params["open_fraction"]

    if anomalous:
        if anomaly_type == "prolonged_closure":
            open_frac = 0.2
        elif anomaly_type == "erratic_gaping":
            gape_std *= 3.0
        elif anomaly_type == "reduced_gape":
            gape_mean *= 0.3
        elif anomaly_type == "hyperactive_gaping":
            gape_std *= 2.0

    # Simulate valve gape over time
    gape = np.zeros(n_frames)
    is_open = rng.random() < open_frac

    for i in range(n_frames):
        if is_open:
            gape[i] = max(0.0, rng.normal(gape_mean, gape_std * 0.3))
            if rng.random() < (1.0 - open_frac) / (fps * 30):
                is_open = False
        else:
            gape[i] = max(0.0, rng.normal(0.1, 0.05))
            if rng.random() < open_frac / (fps * 30):
                is_open = True

    # Generate 8 keypoints around mussel body (~50mm shell)
    n_keypoints = 8
    keypoints = np.zeros((n_frames, n_keypoints, 2))

    # Static base positions (hinge at origin, shell opens to the right)
    base_positions = np.array([
        [0.0, 0.0],     # hinge
        [20.0, 0.0],    # valve_left_anterior (will be displaced by gape)
        [-10.0, 0.0],   # valve_left_posterior
        [20.0, 0.0],    # valve_right_anterior
        [-10.0, 0.0],   # valve_right_posterior
        [25.0, 0.0],    # siphon_inhalant
        [23.0, 0.0],    # siphon_exhalant
        [-5.0, 0.0],    # foot
    ])

    for i in range(n_frames):
        g = gape[i]
        half_g = g / 2.0

        kp = base_positions.copy()
        # Displace valve keypoints by gape
        kp[1, 1] = half_g       # left anterior up
        kp[2, 1] = half_g * 0.5  # left posterior up (less)
        kp[3, 1] = -half_g      # right anterior down
        kp[4, 1] = -half_g * 0.5  # right posterior down
        kp[5, 1] = half_g * 0.8   # siphon inhalant
        kp[6, 1] = half_g * 0.6   # siphon exhalant
        kp[7, 1] = rng.normal(0, 0.1)  # foot small movement

        # Add small jitter
        kp += rng.normal(0, 0.02, size=(n_keypoints, 2))
        keypoints[i] = kp

    return keypoints


_SPECIES_GENERATORS = {
    "daphnia": _generate_daphnia_trajectory,
    "fish": _generate_fish_trajectory,
    "mussel": _generate_mussel_trajectory,
}

_SPECIES_N_KEYPOINTS = {
    "daphnia": 12,
    "fish": 22,
    "mussel": 8,
}


def generate_synthetic_trajectories(
    n_trajectories: int = 1000,
    duration_s: float = 60.0,
    fps: float = 30.0,
    species: str = "daphnia",
    output_dir: str | Path = "data/behavioral/synthetic",
    anomaly_fraction: float = 0.1,
    seed: int = 42,
) -> Path:
    """Generate synthetic normal and anomalous behavioral trajectories.

    Produces a dataset of simulated organism trajectories suitable for
    pretraining the BioMotion diffusion encoder.  Normal trajectories
    follow species-specific random-walk models; anomalous trajectories
    exhibit characteristic stress responses.

    Parameters
    ----------
    n_trajectories:
        Total number of trajectories to generate.
    duration_s:
        Duration of each trajectory in seconds.
    fps:
        Frames per second.
    species:
        Organism type (``"daphnia"``, ``"fish"``, or ``"mussel"``).
    output_dir:
        Directory for output files.
    anomaly_fraction:
        Fraction of trajectories that should be anomalous (0.0 to 1.0).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    Path to the saved ``.npz`` file containing:
      - ``keypoints``: ``(n_trajectories, T, n_keypoints, 2)``
      - ``labels``: ``(n_trajectories,)`` — 0=normal, 1=anomalous
      - ``anomaly_types``: ``(n_trajectories,)`` — string anomaly type
      - ``metadata``: JSON string with generation parameters

    Raises
    ------
    ValueError
        If *species* is not recognized.
    """
    if species not in _SPECIES_GENERATORS:
        raise ValueError(
            f"Unknown species {species!r}. "
            f"Supported: {list(_SPECIES_GENERATORS.keys())}"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    generator = _SPECIES_GENERATORS[species]
    n_keypoints = _SPECIES_N_KEYPOINTS[species]
    n_frames = int(duration_s * fps)
    n_anomalous = int(n_trajectories * anomaly_fraction)
    n_normal = n_trajectories - n_anomalous

    anomaly_types_list = _ANOMALY_TYPES.get(species, ["reduced_velocity"])

    all_keypoints = np.zeros(
        (n_trajectories, n_frames, n_keypoints, 2), dtype=np.float32
    )
    labels = np.zeros(n_trajectories, dtype=np.int32)
    anomaly_type_arr = np.array(["none"] * n_trajectories, dtype=object)

    progress = make_progress()
    with progress:
        task = progress.add_task(
            f"Generating {species} trajectories", total=n_trajectories
        )

        # Normal trajectories
        for i in range(n_normal):
            all_keypoints[i] = generator(
                duration_s, fps, anomalous=False, rng=rng
            )
            progress.advance(task)

        # Anomalous trajectories
        for i in range(n_anomalous):
            idx = n_normal + i
            atype = anomaly_types_list[i % len(anomaly_types_list)]
            all_keypoints[idx] = generator(
                duration_s, fps, anomalous=True, anomaly_type=atype, rng=rng
            )
            labels[idx] = 1
            anomaly_type_arr[idx] = atype
            progress.advance(task)

    # Shuffle trajectories
    perm = rng.permutation(n_trajectories)
    all_keypoints = all_keypoints[perm]
    labels = labels[perm]
    anomaly_type_arr = anomaly_type_arr[perm]

    # Save
    out_path = output_dir / f"synthetic_{species}_{n_trajectories}.npz"
    metadata = json.dumps({
        "species": species,
        "n_trajectories": n_trajectories,
        "duration_s": duration_s,
        "fps": fps,
        "n_keypoints": n_keypoints,
        "anomaly_fraction": anomaly_fraction,
        "seed": seed,
        "anomaly_types": anomaly_types_list,
    })

    np.savez_compressed(
        str(out_path),
        keypoints=all_keypoints,
        labels=labels,
        anomaly_types=anomaly_type_arr,
        metadata=np.array(metadata),
    )

    logger.info(
        f"Generated {n_trajectories} synthetic {species} trajectories "
        f"({n_normal} normal, {n_anomalous} anomalous) -> {out_path}"
    )
    return out_path


# ---------------------------------------------------------------------------
# Convenience: download everything
# ---------------------------------------------------------------------------


def download_all_behavioral(
    output_dir: str | Path = "data/behavioral",
    generate_synthetic: bool = True,
    n_synthetic: int = 10000,
) -> dict[str, Any]:
    """Download all behavioral tracking datasets and optionally generate
    synthetic pretraining data.

    Parameters
    ----------
    output_dir:
        Root directory for all behavioral data.
    generate_synthetic:
        If True, generate synthetic trajectories for each species.
    n_synthetic:
        Number of synthetic trajectories per species.

    Returns
    -------
    Dict with keys ``"daphnia"``, ``"fish"``, ``"mussel"``, and
    optionally ``"synthetic"`` mapping to lists of output paths.
    """
    output_dir = Path(output_dir)
    results: dict[str, Any] = {}

    logger.info("Downloading all behavioral tracking datasets...")

    results["daphnia"] = download_daphnia_datasets(output_dir / "daphnia")
    results["fish"] = download_fish_tracking(output_dir / "fish")
    results["mussel"] = download_mussel_valve_data(output_dir / "mussel")

    if generate_synthetic:
        results["synthetic"] = {}
        for species in ("daphnia", "fish", "mussel"):
            logger.info(f"Generating {n_synthetic} synthetic {species} trajectories...")
            path = generate_synthetic_trajectories(
                n_trajectories=n_synthetic,
                species=species,
                output_dir=output_dir / "synthetic",
            )
            results["synthetic"][species] = path

    total_files = (
        len(results.get("daphnia", []))
        + len(results.get("fish", []))
        + len(results.get("mussel", []))
    )
    logger.info(
        f"Behavioral data acquisition complete: {total_files} downloaded files, "
        f"synthetic={'enabled' if generate_synthetic else 'disabled'}"
    )
    return results
