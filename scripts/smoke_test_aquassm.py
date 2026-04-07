#!/usr/bin/env python3
"""Smoke test: Verify AquaSSM trains on synthetic data.

Quick validation that the full training pipeline works end-to-end
before committing to the real data training run.

MIT License — Bryan Cheng, 2026
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from sentinel.models.sensor_encoder import SensorEncoder
from sentinel.models.sensor_encoder.physics_constraints import PhysicsConstraintLoss

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path("data/processed/sensor/synthetic")


class SyntheticSensorDataset(Dataset):
    """Load synthetic .npz files for AquaSSM training."""

    def __init__(self, data_dir: Path, max_len: int = 512):
        self.files = sorted(data_dir.glob("*.npz"))
        self.max_len = max_len

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        values = data["values"][:self.max_len].astype(np.float32)
        delta_ts = data["delta_ts"][:self.max_len].astype(np.float32)
        labels = data["labels"][:self.max_len].astype(np.int64)

        return {
            "values": torch.tensor(values),
            "delta_ts": torch.tensor(delta_ts),
            "labels": torch.tensor(labels),
        }


def run_smoke_test():
    print(f"Device: {DEVICE}")
    print(f"Data dir: {DATA_DIR}")

    # Load dataset
    ds = SyntheticSensorDataset(DATA_DIR, max_len=512)
    print(f"Dataset size: {len(ds)} sequences")

    dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)

    # Build model
    model = SensorEncoder().to(DEVICE)
    physics_loss_fn = PhysicsConstraintLoss().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Phase 1: MPP Pretraining (5 epochs)
    print("\n--- Phase 1: MPP Pretraining ---")
    model.train()
    for epoch in range(5):
        total_loss = 0
        n_batches = 0
        for batch in dl:
            values = batch["values"].to(DEVICE)
            delta_ts = batch["delta_ts"].to(DEVICE)

            mpp_out = model.forward_pretrain(x=values, delta_ts=delta_ts)
            mpp_loss = mpp_out["loss"]

            # Skip batch if MPP loss is NaN (rare masking edge case)
            if torch.isnan(mpp_loss):
                optimizer.zero_grad()
                continue

            # Physics constraints — convert tensor [B,T,6] to named dict
            pred_tensor = mpp_out["predictions"]  # [B, T, 6]
            param_names = ["do", "ph", "conductivity", "temperature", "turb", "orp"]
            pred_dict = {name: pred_tensor[..., i] for i, name in enumerate(param_names)}
            phys_out = physics_loss_fn(pred_dict)
            phys_loss = phys_out["total_loss"].clamp(max=10.0)

            loss = mpp_loss + 0.1 * phys_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"  Epoch {epoch+1}/5 | Loss: {avg_loss:.4f} ({n_batches} batches)")

    # Phase 2: Full forward pass (anomaly detection)
    print("\n--- Phase 2: Full Forward Pass ---")
    model.eval()
    with torch.no_grad():
        batch = next(iter(dl))
        values = batch["values"].to(DEVICE)
        delta_ts = batch["delta_ts"].to(DEVICE)

        output = model(values, delta_ts, compute_anomaly=True)

        print(f"  Embedding shape: {output['embedding'].shape}")
        print(f"  Fusion embedding shape: {output['fusion_embedding'].shape}")
        print(f"  Anomaly scores keys: {list(output['anomaly_scores'].keys())}")
        print(f"  Sensor health keys: {list(output['sensor_health'].keys())}")

        if "mean_errors" in output["anomaly_scores"]:
            mean_err = output["anomaly_scores"]["mean_errors"]
            print(f"  Mean reconstruction error: {mean_err.mean():.4f}")

    print("\n✓ AquaSSM smoke test PASSED!")
    return True


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
