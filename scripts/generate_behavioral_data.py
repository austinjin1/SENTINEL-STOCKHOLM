#!/usr/bin/env python3
"""Generate realistic synthetic Daphnia behavioral trajectory data.

Creates 500 trajectories (250 normal, 250 anomalous) for training the
BioMotion encoder.  Each trajectory contains:
  - keypoints: (T, 12, 2) -- 12 Daphnia body keypoints in 2D
  - features:  (T, 16)    -- behavioral features (velocity, acceleration,
                              turning angle, inter-keypoint distances)
  - timestamps: (T,)      -- frame timestamps in seconds at 30 fps
  - is_anomaly: bool       -- whether the trajectory is anomalous

Normal behavior: smooth random walk with characteristic Daphnia velocity
  and turning distributions (hop-and-sink locomotion).

Anomalous behavior (4 types, evenly split):
  1. Erratic -- sudden direction changes, high jitter
  2. Freezing -- prolonged immobility periods
  3. Spinning -- circular/rotational motion patterns
  4. Altered velocity -- abnormally fast or slow movement

Output: data/processed/behavioral/traj_{idx:04d}.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


# ── Constants ───────────────────────────────────────────────────────────────

N_KEYPOINTS = 12
N_FEATURES = 16
T = 200          # frames per trajectory
FPS = 30.0
DT = 1.0 / FPS

# Daphnia body plan: approximate relative positions of 12 keypoints
# (head, antennae x2, eye, carapace x3, gut, brood pouch, post-abdomen,
#  tail spine, swimming appendage)
DAPHNIA_REST_POSE = np.array([
    [0.0, 1.5],    # head (dorsal)
    [-0.4, 1.8],   # left antenna base
    [0.4, 1.8],    # right antenna base
    [0.0, 1.3],    # eye
    [-0.3, 0.8],   # left carapace
    [0.3, 0.8],    # right carapace
    [0.0, 0.5],    # mid carapace
    [0.0, 0.3],    # gut
    [0.0, 0.0],    # brood pouch
    [0.0, -0.4],   # post-abdomen
    [0.0, -0.8],   # tail spine
    [-0.2, -0.2],  # swimming appendage
], dtype=np.float32)

# Pre-compute inter-keypoint distance pairs for feature extraction
# Use a subset of 10 canonical distances + 6 extra = 16 features total
# (velocity=1, acceleration=1, turning_angle=1, speed_std=1,
#  angular_velocity=1, path_curvature=1, inter-kp distances x10)
N_IKD = 10  # inter-keypoint distances
IKD_PAIRS = [
    (0, 10),  # head to tail
    (0, 3),   # head to eye
    (1, 2),   # left to right antenna
    (4, 5),   # left to right carapace
    (0, 8),   # head to brood pouch
    (3, 7),   # eye to gut
    (6, 9),   # mid carapace to post-abdomen
    (8, 10),  # brood pouch to tail
    (9, 11),  # post-abdomen to swimming appendage
    (7, 10),  # gut to tail spine
]

assert len(IKD_PAIRS) == N_IKD


def rotation_matrix(angle: float) -> np.ndarray:
    """2D rotation matrix."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]], dtype=np.float32)


# ── Trajectory generators ──────────────────────────────────────────────────

def generate_normal_trajectory(rng: np.random.Generator) -> np.ndarray:
    """Generate a normal Daphnia trajectory (centroid path).

    Daphnia exhibit hop-and-sink locomotion: short bursts of upward
    swimming followed by passive sinking.  Normal trajectories have
    moderate velocity with smooth directional changes.

    Returns:
        positions: (T, 2) centroid trajectory
    """
    positions = np.zeros((T, 2), dtype=np.float32)
    # Initial position
    positions[0] = rng.uniform(-5, 5, size=2).astype(np.float32)

    # Parameters for normal Daphnia movement
    base_speed = rng.uniform(0.8, 1.5)     # body lengths/sec
    speed_per_frame = base_speed * DT
    direction = rng.uniform(0, 2 * np.pi)
    turning_rate = 0.15  # radians per frame (smooth turns)

    for t in range(1, T):
        # Hop-and-sink: occasional speed bursts
        if rng.random() < 0.08:
            speed = speed_per_frame * rng.uniform(2.0, 4.0)
            direction += rng.uniform(-0.3, 0.3)
        else:
            speed = speed_per_frame * rng.uniform(0.3, 1.2)
            direction += rng.normal(0, turning_rate)

        dx = speed * np.cos(direction)
        dy = speed * np.sin(direction)
        positions[t] = positions[t - 1] + np.array([dx, dy], dtype=np.float32)

    return positions


def generate_erratic_trajectory(rng: np.random.Generator) -> np.ndarray:
    """Anomalous: erratic movement with sudden direction changes."""
    positions = np.zeros((T, 2), dtype=np.float32)
    positions[0] = rng.uniform(-5, 5, size=2).astype(np.float32)

    base_speed = rng.uniform(1.5, 3.0) * DT

    for t in range(1, T):
        # Frequent, large direction changes
        direction = rng.uniform(0, 2 * np.pi)
        speed = base_speed * rng.uniform(0.5, 5.0)

        # Add high-frequency jitter
        jitter = rng.normal(0, 0.03, size=2).astype(np.float32)

        dx = speed * np.cos(direction) + jitter[0]
        dy = speed * np.sin(direction) + jitter[1]
        positions[t] = positions[t - 1] + np.array([dx, dy], dtype=np.float32)

    return positions


def generate_freezing_trajectory(rng: np.random.Generator) -> np.ndarray:
    """Anomalous: prolonged freezing periods interspersed with movement."""
    positions = np.zeros((T, 2), dtype=np.float32)
    positions[0] = rng.uniform(-5, 5, size=2).astype(np.float32)

    base_speed = rng.uniform(0.8, 1.5) * DT
    direction = rng.uniform(0, 2 * np.pi)

    # Generate freezing intervals (30-70% of trajectory is frozen)
    freeze_frac = rng.uniform(0.3, 0.7)
    is_frozen = np.zeros(T, dtype=bool)
    n_freeze_bouts = rng.integers(2, 6)
    freeze_starts = sorted(rng.choice(range(10, T - 20), size=n_freeze_bouts, replace=False))
    for start in freeze_starts:
        length = int(T * freeze_frac / n_freeze_bouts)
        is_frozen[start:min(start + length, T)] = True

    for t in range(1, T):
        if is_frozen[t]:
            # Frozen: only tiny vibration noise
            noise = rng.normal(0, 0.002, size=2).astype(np.float32)
            positions[t] = positions[t - 1] + noise
        else:
            speed = base_speed * rng.uniform(0.5, 1.5)
            direction += rng.normal(0, 0.15)
            dx = speed * np.cos(direction)
            dy = speed * np.sin(direction)
            positions[t] = positions[t - 1] + np.array([dx, dy], dtype=np.float32)

    return positions


def generate_spinning_trajectory(rng: np.random.Generator) -> np.ndarray:
    """Anomalous: circular/rotational movement patterns."""
    positions = np.zeros((T, 2), dtype=np.float32)
    center = rng.uniform(-5, 5, size=2).astype(np.float32)
    positions[0] = center + rng.uniform(-0.5, 0.5, size=2).astype(np.float32)

    # Spinning parameters
    angular_speed = rng.uniform(0.15, 0.4) * rng.choice([-1, 1])
    radius = rng.uniform(0.3, 1.0)
    angle = rng.uniform(0, 2 * np.pi)

    # Mix of spinning bouts and brief normal movement
    spin_active = rng.random(T) < rng.uniform(0.5, 0.9)
    direction = rng.uniform(0, 2 * np.pi)

    for t in range(1, T):
        if spin_active[t]:
            angle += angular_speed
            target = center + radius * np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
            # Drift toward circle path
            positions[t] = positions[t - 1] * 0.7 + target * 0.3
            positions[t] += rng.normal(0, 0.01, size=2).astype(np.float32)
        else:
            speed = rng.uniform(0.8, 1.5) * DT
            direction += rng.normal(0, 0.15)
            dx = speed * np.cos(direction)
            dy = speed * np.sin(direction)
            positions[t] = positions[t - 1] + np.array([dx, dy], dtype=np.float32)

    return positions


def generate_altered_velocity_trajectory(rng: np.random.Generator) -> np.ndarray:
    """Anomalous: abnormally fast or slow sustained movement."""
    positions = np.zeros((T, 2), dtype=np.float32)
    positions[0] = rng.uniform(-5, 5, size=2).astype(np.float32)

    # Either very fast or very slow (not normal range)
    if rng.random() < 0.5:
        base_speed = rng.uniform(3.0, 6.0) * DT   # abnormally fast
    else:
        base_speed = rng.uniform(0.05, 0.2) * DT   # abnormally slow

    direction = rng.uniform(0, 2 * np.pi)

    for t in range(1, T):
        speed = base_speed * rng.uniform(0.8, 1.2)
        direction += rng.normal(0, 0.12)
        dx = speed * np.cos(direction)
        dy = speed * np.sin(direction)
        positions[t] = positions[t - 1] + np.array([dx, dy], dtype=np.float32)

    return positions


# ── Keypoint generation from centroid ──────────────────────────────────────

def generate_keypoints_from_centroid(
    positions: np.ndarray,
    rng: np.random.Generator,
    anomaly_type: str | None = None,
) -> np.ndarray:
    """Generate 12 keypoints per frame from centroid trajectory.

    Applies the rest-pose template with heading-based rotation and
    biologically-motivated deformation noise.

    Args:
        positions: (T, 2) centroid path
        rng: random generator
        anomaly_type: if not None, adds type-specific keypoint perturbations

    Returns:
        keypoints: (T, 12, 2)
    """
    keypoints = np.zeros((T, N_KEYPOINTS, 2), dtype=np.float32)

    # Compute heading from velocity
    velocities = np.diff(positions, axis=0, prepend=positions[:1])
    headings = np.arctan2(velocities[:, 1], velocities[:, 0])

    # Smooth headings
    for t in range(1, T):
        diff = headings[t] - headings[t - 1]
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        headings[t] = headings[t - 1] + 0.3 * diff

    # Body deformation: sinusoidal carapace flex
    flex_freq = rng.uniform(0.5, 2.0)
    flex_amp = rng.uniform(0.02, 0.08)

    for t in range(T):
        R = rotation_matrix(headings[t])
        # Apply rest pose with rotation
        pose = (R @ DAPHNIA_REST_POSE.T).T  # (12, 2)

        # Add body flex (primarily anterior-posterior)
        flex = flex_amp * np.sin(2 * np.pi * flex_freq * t / FPS)
        pose[:, 1] *= 1.0 + flex * 0.1

        # Antenna waving (keypoints 1, 2)
        antenna_wave = 0.05 * np.sin(2 * np.pi * 3.0 * t / FPS + rng.uniform(0, np.pi))
        pose[1, 0] += antenna_wave
        pose[2, 0] -= antenna_wave

        # Swimming appendage beat (keypoint 11)
        beat = 0.08 * np.sin(2 * np.pi * 5.0 * t / FPS)
        pose[11, 0] += beat

        # Add measurement noise (simulates tracking uncertainty)
        noise = rng.normal(0, 0.015, size=(N_KEYPOINTS, 2)).astype(np.float32)

        # Type-specific keypoint perturbations
        if anomaly_type == "spinning":
            noise *= 2.0  # higher tracking noise during spinning
        elif anomaly_type == "erratic":
            noise *= 3.0  # very noisy tracking
        elif anomaly_type == "freezing":
            noise *= 0.3  # very low noise when frozen

        keypoints[t] = positions[t] + pose + noise

    return keypoints


# ── Feature extraction ─────────────────────────────────────────────────────

def extract_features(
    keypoints: np.ndarray,
    timestamps: np.ndarray,
) -> np.ndarray:
    """Extract 16 behavioral features from keypoint trajectories.

    Features (16 total):
      0: instantaneous speed (centroid)
      1: acceleration magnitude
      2: turning angle (radians)
      3: speed standard deviation (rolling window)
      4: angular velocity
      5: path curvature
      6-15: 10 inter-keypoint distances

    Args:
        keypoints: (T, 12, 2) keypoint positions
        timestamps: (T,) frame timestamps in seconds

    Returns:
        features: (T, 16)
    """
    features = np.zeros((T, N_FEATURES), dtype=np.float32)

    # Centroid
    centroid = keypoints.mean(axis=1)  # (T, 2)

    # Velocity vectors
    dt = np.diff(timestamps, prepend=timestamps[0] - DT)
    dt = np.maximum(dt, 1e-6)
    vel = np.diff(centroid, axis=0, prepend=centroid[:1]) / dt[:, None]

    # Feature 0: instantaneous speed
    speed = np.linalg.norm(vel, axis=1)
    features[:, 0] = speed

    # Feature 1: acceleration magnitude
    accel = np.diff(vel, axis=0, prepend=vel[:1]) / dt[:, None]
    features[:, 1] = np.linalg.norm(accel, axis=1)

    # Feature 2: turning angle
    for t in range(2, T):
        v1 = vel[t - 1]
        v2 = vel[t]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 > 1e-8 and n2 > 1e-8:
            cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
            features[t, 2] = np.arccos(cos_a)

    # Feature 3: speed standard deviation (rolling window of 10 frames)
    window = 10
    for t in range(T):
        start = max(0, t - window + 1)
        features[t, 3] = np.std(speed[start:t + 1])

    # Feature 4: angular velocity
    headings = np.arctan2(vel[:, 1], vel[:, 0])
    angular_vel = np.diff(headings, prepend=headings[0])
    angular_vel = (angular_vel + np.pi) % (2 * np.pi) - np.pi
    features[:, 4] = np.abs(angular_vel) / dt

    # Feature 5: path curvature (turning angle / distance)
    dist = speed * dt
    curvature = np.where(dist > 1e-6, np.abs(angular_vel) / dist, 0.0)
    features[:, 5] = curvature

    # Features 6-15: inter-keypoint distances
    for i, (k1, k2) in enumerate(IKD_PAIRS):
        features[:, 6 + i] = np.linalg.norm(
            keypoints[:, k1] - keypoints[:, k2], axis=1
        )

    return features


# ── Main generation loop ───────────────────────────────────────────────────

ANOMALY_GENERATORS = {
    "erratic": generate_erratic_trajectory,
    "freezing": generate_freezing_trajectory,
    "spinning": generate_spinning_trajectory,
    "altered_velocity": generate_altered_velocity_trajectory,
}


def generate_dataset(
    output_dir: Path,
    n_total: int = 500,
    n_anomalous: int = 250,
    seed: int = 42,
) -> None:
    """Generate the full synthetic behavioral dataset.

    Args:
        output_dir: Directory to save .npz files.
        n_total: Total number of trajectories.
        n_anomalous: Number of anomalous trajectories.
        seed: Random seed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    n_normal = n_total - n_anomalous
    anomaly_types = list(ANOMALY_GENERATORS.keys())
    per_type = n_anomalous // len(anomaly_types)
    remainder = n_anomalous - per_type * len(anomaly_types)

    timestamps = np.arange(T, dtype=np.float32) / FPS

    print(f"Generating {n_total} trajectories ({n_normal} normal, {n_anomalous} anomalous)")
    print(f"  Anomaly types: {anomaly_types} ({per_type} each + {remainder} extra)")
    print(f"  Output: {output_dir}")

    idx = 0

    # Normal trajectories
    for i in range(n_normal):
        positions = generate_normal_trajectory(rng)
        kp = generate_keypoints_from_centroid(positions, rng, anomaly_type=None)
        feats = extract_features(kp, timestamps)

        np.savez_compressed(
            output_dir / f"traj_{idx:04d}.npz",
            keypoints=kp,
            features=feats,
            timestamps=timestamps,
            is_anomaly=False,
        )
        idx += 1
        if (i + 1) % 50 == 0:
            print(f"  Normal: {i + 1}/{n_normal}")

    # Anomalous trajectories
    for type_idx, atype in enumerate(anomaly_types):
        gen_fn = ANOMALY_GENERATORS[atype]
        n_this_type = per_type + (1 if type_idx < remainder else 0)

        for i in range(n_this_type):
            positions = gen_fn(rng)
            kp = generate_keypoints_from_centroid(positions, rng, anomaly_type=atype)
            feats = extract_features(kp, timestamps)

            np.savez_compressed(
                output_dir / f"traj_{idx:04d}.npz",
                keypoints=kp,
                features=feats,
                timestamps=timestamps,
                is_anomaly=True,
            )
            idx += 1

        print(f"  Anomalous ({atype}): {n_this_type} trajectories")

    print(f"\nDone. Generated {idx} trajectories in {output_dir}")

    # Verify shapes on one file
    sample = np.load(output_dir / "traj_0000.npz")
    print(f"\nSample file shapes:")
    print(f"  keypoints:  {sample['keypoints'].shape}   (expected ({T}, {N_KEYPOINTS}, 2))")
    print(f"  features:   {sample['features'].shape}   (expected ({T}, {N_FEATURES}))")
    print(f"  timestamps: {sample['timestamps'].shape}   (expected ({T},))")
    print(f"  is_anomaly: {sample['is_anomaly']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic Daphnia behavioral data")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed/behavioral",
        help="Output directory for .npz files",
    )
    parser.add_argument("--n-total", type=int, default=500)
    parser.add_argument("--n-anomalous", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Resolve relative to project root
    output = Path(args.output_dir)
    if not output.is_absolute():
        project_root = Path(__file__).resolve().parent.parent
        output = project_root / output

    generate_dataset(output, args.n_total, args.n_anomalous, args.seed)


if __name__ == "__main__":
    main()
