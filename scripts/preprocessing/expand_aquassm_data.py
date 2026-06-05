#!/usr/bin/env python3
"""
Expand AquaSSM training data by generating 600 synthetic sensor sequences.
Target: 300 normal + 300 anomaly → save to data/processed/sensor/expanded_v2/

Each file saves: values (T, 6), delta_ts (T,), labels (T,), has_anomaly (scalar)
Format matches clean_synthetic/ exactly.

Bryan Cheng, SENTINEL project, 2026
"""

import numpy as np
from pathlib import Path

SEED = 42
rng = np.random.default_rng(SEED)

OUTPUT_DIR = Path("data/processed/sensor/expanded_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_NORMAL = 300
N_ANOMALY = 300
T_MIN = 128
T_MAX = 512

INTERVAL_SECONDS = 900.0  # 15-min USGS standard


def generate_ar1_noise(T: int, sigma: float, phi: float = 0.85) -> np.ndarray:
    """Generate AR(1) noise with given phi and innovation std."""
    innovation_std = sigma * np.sqrt(1 - phi**2)
    noise = np.zeros(T)
    noise[0] = rng.normal(0, sigma)
    for t in range(1, T):
        noise[t] = phi * noise[t - 1] + rng.normal(0, innovation_std)
    return noise.astype(np.float32)


def generate_normal_sequence(T: int) -> np.ndarray:
    """
    Generate realistic normal water quality sequence.
    Returns array of shape (T, 6) with channels:
      0: pH, 1: DO, 2: turbidity, 3: SpCond, 4: Temp, 5: ORP
    Values are RAW (not normalized) — will be z-scored to [-5,5] range for storage.
    """
    t = np.arange(T, dtype=np.float32)
    season = np.sin(2 * np.pi * t / (365 * 96))  # ~annual cycle at 15-min intervals

    # pH: N(7.2, 0.4) with AR(1) noise
    ph_base = rng.normal(7.2, 0.2)
    ph = ph_base + generate_ar1_noise(T, 0.3) + 0.05 * season
    ph = np.clip(ph, 6.0, 9.0)

    # DO: N(8.5, 1.5) with AR(1) noise, slight anti-correlation with temp
    do_base = rng.normal(8.5, 1.0)
    do_ = do_base + generate_ar1_noise(T, 1.0) - 0.5 * season
    do_ = np.clip(do_, 2.0, 14.0)

    # Turbidity: lognormal(2.0, 0.8) — always positive
    turb_log_mean = rng.normal(2.0, 0.5)
    turb_log_noise = generate_ar1_noise(T, 0.6)
    turb = np.exp(turb_log_mean + turb_log_noise)
    turb = np.clip(turb, 0.0, 300.0)

    # SpCond: N(400, 150) with AR(1) noise
    spcond_base = rng.normal(400, 100)
    spcond = spcond_base + generate_ar1_noise(T, 80)
    spcond = np.clip(spcond, 10.0, 2000.0)

    # Temp: N(15, 5) + seasonal component + AR(1)
    temp_base = rng.normal(15, 3)
    temp = temp_base + 8 * season + generate_ar1_noise(T, 2.0)
    temp = np.clip(temp, -2.0, 40.0)

    # ORP: N(200, 50) with AR(1)
    orp_base = rng.normal(200, 30)
    orp = orp_base + generate_ar1_noise(T, 35)
    # No clip needed for ORP

    seq = np.stack([ph, do_, turb, spcond, temp, orp], axis=1).astype(np.float32)
    return seq


def normalize_sequence(seq: np.ndarray) -> np.ndarray:
    """
    Z-score normalize per channel, clipped to [-5, 5].
    Matches how clean_synthetic was created.
    """
    # Use approximate population stats per channel
    means = np.array([7.2, 8.5, 10.0, 400.0, 15.0, 200.0], dtype=np.float32)
    stds = np.array([0.5, 1.5, 15.0, 150.0, 8.0, 60.0], dtype=np.float32)
    normalized = (seq - means) / (stds + 1e-8)
    return np.clip(normalized, -5.0, 5.0)


def generate_normal_file(T: int):
    """Generate a normal sequence file."""
    seq_raw = generate_normal_sequence(T)
    values = normalize_sequence(seq_raw)
    delta_ts = np.full(T, INTERVAL_SECONDS, dtype=np.float32)
    delta_ts[0] = 0.0
    labels = np.zeros(T, dtype=np.int64)
    has_anomaly = np.int64(0)
    return values, delta_ts, labels, has_anomaly


def generate_anomaly_file(T: int):
    """Generate anomaly sequence: normal + injected anomaly at random position."""
    seq_raw = generate_normal_sequence(T).copy()

    # Anomaly position: 30–70% into sequence
    start_frac = rng.uniform(0.30, 0.70)
    anom_start = int(start_frac * T)

    # Choose anomaly type
    anom_type = rng.integers(0, 4)

    labels = np.zeros(T, dtype=np.int64)

    if anom_type == 0:
        # pH spike: +2.5 or -2.0 for 15-40 steps
        dur = int(rng.integers(15, 41))
        dur = min(dur, T - anom_start)
        delta = rng.choice([2.5, -2.0])
        seq_raw[anom_start:anom_start + dur, 0] += delta
        seq_raw[:, 0] = np.clip(seq_raw[:, 0], 6.0, 9.0)
        labels[anom_start:anom_start + dur] = 1

    elif anom_type == 1:
        # DO crash: multiply by 0.2 for 20-50 steps
        dur = int(rng.integers(20, 51))
        dur = min(dur, T - anom_start)
        seq_raw[anom_start:anom_start + dur, 1] *= 0.2
        seq_raw[:, 1] = np.clip(seq_raw[:, 1], 2.0, 14.0)
        labels[anom_start:anom_start + dur] = 1

    elif anom_type == 2:
        # Turbidity surge: multiply by 8 for 15-30 steps
        dur = int(rng.integers(15, 31))
        dur = min(dur, T - anom_start)
        seq_raw[anom_start:anom_start + dur, 2] *= 8.0
        seq_raw[:, 2] = np.clip(seq_raw[:, 2], 0.0, 300.0)
        labels[anom_start:anom_start + dur] = 1

    else:
        # SpCond jump: multiply by 3 for 25-60 steps
        dur = int(rng.integers(25, 61))
        dur = min(dur, T - anom_start)
        seq_raw[anom_start:anom_start + dur, 3] *= 3.0
        seq_raw[:, 3] = np.clip(seq_raw[:, 3], 10.0, 2000.0)
        labels[anom_start:anom_start + dur] = 1

    # Add Gaussian noise throughout
    seq_raw += rng.normal(0, 0.05, seq_raw.shape).astype(np.float32)

    values = normalize_sequence(seq_raw)
    delta_ts = np.full(T, INTERVAL_SECONDS, dtype=np.float32)
    delta_ts[0] = 0.0
    has_anomaly = np.int64(1)
    return values, delta_ts, labels, has_anomaly


def main():
    print(f"Generating {N_NORMAL} normal + {N_ANOMALY} anomaly sequences...")
    print(f"Output directory: {OUTPUT_DIR}")

    file_idx = 0
    # --- Normal sequences ---
    for i in range(N_NORMAL):
        T = int(rng.integers(T_MIN, T_MAX + 1))
        values, delta_ts, labels, has_anomaly = generate_normal_file(T)
        fname = OUTPUT_DIR / f"sample_{file_idx:04d}.npz"
        np.savez(fname, values=values, delta_ts=delta_ts, labels=labels, has_anomaly=has_anomaly)
        file_idx += 1
        if (i + 1) % 100 == 0:
            print(f"  Normal: {i + 1}/{N_NORMAL}")

    print(f"  Normal done: {N_NORMAL} files")

    # --- Anomaly sequences ---
    for i in range(N_ANOMALY):
        T = int(rng.integers(T_MIN, T_MAX + 1))
        values, delta_ts, labels, has_anomaly = generate_anomaly_file(T)
        fname = OUTPUT_DIR / f"sample_{file_idx:04d}.npz"
        np.savez(fname, values=values, delta_ts=delta_ts, labels=labels, has_anomaly=has_anomaly)
        file_idx += 1
        if (i + 1) % 100 == 0:
            print(f"  Anomaly: {i + 1}/{N_ANOMALY}")

    print(f"  Anomaly done: {N_ANOMALY} files")
    print(f"\nTotal files generated: {file_idx}")
    print(f"Saved to: {OUTPUT_DIR.resolve()}")

    # Verify one file
    test_f = OUTPUT_DIR / "sample_0000.npz"
    d = np.load(test_f)
    print(f"\nVerification of sample_0000.npz:")
    print(f"  keys: {d.files}")
    print(f"  values shape: {d['values'].shape}, dtype: {d['values'].dtype}")
    print(f"  delta_ts shape: {d['delta_ts'].shape}")
    print(f"  labels shape: {d['labels'].shape}")
    print(f"  has_anomaly: {d['has_anomaly']}")
    print(f"  values range: [{d['values'].min():.3f}, {d['values'].max():.3f}]")

    # Final counts summary
    all_anomaly = sum(
        int(np.load(f)['has_anomaly'])
        for f in sorted(OUTPUT_DIR.glob("*.npz"))
    )
    all_total = len(list(OUTPUT_DIR.glob("*.npz")))
    print(f"\nFinal count: {all_total} total ({all_total - all_anomaly} normal, {all_anomaly} anomaly)")

    # Overall data summary
    print("\n=== Overall Data Summary ===")
    clean_n = len(list(Path("data/processed/sensor/clean_synthetic").glob("*.npz")))
    pretrain_n = len(list(Path("data/processed/sensor/pretrain").glob("*.npz")))
    expanded_n = all_total
    total = clean_n + pretrain_n + expanded_n
    print(f"  clean_synthetic: {clean_n}")
    print(f"  pretrain:        {pretrain_n}")
    print(f"  expanded_v2:     {expanded_n}")
    print(f"  TOTAL:           {total} ({total/262:.1f}x original)")


if __name__ == "__main__":
    main()
