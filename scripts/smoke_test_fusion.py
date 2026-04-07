#!/usr/bin/env python3
"""End-to-end fusion smoke test: all 5 encoders → fusion → output heads.

Verifies the complete SENTINEL pipeline works with synthetic data.

MIT License — Bryan Cheng, 2026
"""

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from sentinel.models.sensor_encoder import SensorEncoder
from sentinel.models.satellite_encoder.model import SatelliteEncoder
from sentinel.models.microbial_encoder.model import MicrobialEncoder
from sentinel.models.molecular_encoder.model import MolecularEncoder
from sentinel.models.biomotion.model import BioMotionEncoder
from sentinel.models.fusion.model import PerceiverIOFusion as SentinelFusion

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path("data/processed/synthetic_multimodal")


def load_sample(modality, idx):
    """Load a single synthetic sample."""
    f = DATA_DIR / modality / f"{modality}_{idx:04d}.npz"
    return dict(np.load(f, allow_pickle=True))


def main():
    print(f"Device: {DEVICE}")
    print("=" * 60)
    print("SENTINEL End-to-End Fusion Smoke Test")
    print("=" * 60)

    B = 4  # batch size

    # Initialize all encoders
    print("\n--- Initializing Encoders ---")
    sensor_enc = SensorEncoder().to(DEVICE)
    satellite_enc = SatelliteEncoder().to(DEVICE)
    microbial_enc = MicrobialEncoder().to(DEVICE)

    # Molecular needs hierarchy
    n_genes, n_pw, n_pr, n_out = 1000, 50, 20, 7
    pw_adj = (torch.rand(n_pw, n_genes) > 0.95).float()
    pr_adj = (torch.rand(n_pr, n_pw) > 0.7).float()
    out_adj = (torch.rand(n_out, n_pr) > 0.5).float()
    molecular_enc = MolecularEncoder(
        [f"g{i}" for i in range(n_genes)], pw_adj, pr_adj, out_adj
    ).to(DEVICE)

    biomotion_enc = BioMotionEncoder().to(DEVICE)
    # Fusion native dims must match actual encoder output dimensions
    actual_native_dims = {
        "satellite": 256,   # SatelliteEncoder projects to 256 internally
        "sensor": 256,
        "microbial": 256,
        "molecular": 256,   # MolecularEncoder projects to 256 internally
        "behavioral": 256,
    }
    fusion = SentinelFusion(native_dims=actual_native_dims).to(DEVICE)

    total_params = sum(
        sum(p.numel() for p in m.parameters())
        for m in [sensor_enc, satellite_enc, microbial_enc, molecular_enc, biomotion_enc, fusion]
    )
    print(f"Total parameters: {total_params:,}")

    # Encode each modality
    print("\n--- Encoding Modalities ---")

    # 1. Sensor
    sensor_data = [load_sample("sensor", i) for i in range(B)]
    max_t = max(len(d["values"]) for d in sensor_data)
    sensor_values = torch.zeros(B, max_t, 6)
    sensor_dt = torch.zeros(B, max_t)
    for i, d in enumerate(sensor_data):
        T = len(d["values"])
        sensor_values[i, :T] = torch.tensor(d["values"])
        sensor_dt[i, :T] = torch.tensor(d["delta_ts"])
    with torch.no_grad():
        sensor_out = sensor_enc(sensor_values.to(DEVICE), sensor_dt.to(DEVICE))
    print(f"  Sensor:     {sensor_out['embedding'].shape} ✓")

    # 2. Satellite (resize to 224x224 for ViT)
    sat_data = [load_sample("satellite", i) for i in range(B)]
    sat_images = torch.stack([
        torch.nn.functional.interpolate(
            torch.tensor(d["image"]).unsqueeze(0), size=(224, 224), mode="bilinear"
        ).squeeze(0)
        for d in sat_data
    ])
    with torch.no_grad():
        sat_out = satellite_enc(sat_images.to(DEVICE))
    print(f"  Satellite:  {sat_out['embedding'].shape} ✓")

    # 3. Microbial
    micro_data = [load_sample("microbial", i) for i in range(B)]
    micro_abund = torch.stack([torch.tensor(d["abundances"]) for d in micro_data])
    with torch.no_grad():
        micro_out = microbial_enc(micro_abund.to(DEVICE))
    print(f"  Microbial:  {micro_out['embedding'].shape} ✓")

    # 4. Molecular
    mol_data = [load_sample("molecular", i) for i in range(B)]
    mol_expr = torch.stack([torch.tensor(d["expression"]) for d in mol_data])
    with torch.no_grad():
        mol_out = molecular_enc(mol_expr.to(DEVICE))
    print(f"  Molecular:  {mol_out['embedding'].shape} ✓")

    # 5. Behavioral
    bio_data = [load_sample("behavioral", i) for i in range(B)]
    max_t_bio = max(len(d["keypoints"]) for d in bio_data)
    bio_kp = torch.zeros(B, max_t_bio, 12, 2)
    bio_feat = torch.zeros(B, max_t_bio, 16)
    bio_ts = torch.zeros(B, max_t_bio)
    for i, d in enumerate(bio_data):
        T = len(d["keypoints"])
        bio_kp[i, :T] = torch.tensor(d["keypoints"])
        bio_feat[i, :T] = torch.tensor(d["features"])
        bio_ts[i, :T] = torch.tensor(d["timestamps"])
    bio_input = {
        "daphnia": {
            "keypoints": bio_kp.to(DEVICE),
            "features": bio_feat.to(DEVICE),
            "timestamps": bio_ts.to(DEVICE),
        }
    }
    with torch.no_grad():
        bio_out = biomotion_enc(bio_input)
    print(f"  Behavioral: {bio_out['embedding'].shape} ✓")

    # Fusion — event-driven: process one modality at a time
    print("\n--- Running Fusion ---")
    now = time.time()
    latent_state = None  # initial state

    # Fusion expects native-dim embeddings (not pre-projected fusion_embedding)
    # satellite: 384, sensor: 256, microbial: 256, molecular: 128, behavioral: 256
    modality_data = [
        ("behavioral", bio_out["embedding"], now - 60, 0.90),
        ("sensor", sensor_out["embedding"], now, 0.95),
        ("satellite", sat_out["embedding"], now - 3600, 0.85),  # 384-dim
        ("microbial", micro_out["embedding"], now - 86400, 0.70),
        ("molecular", mol_out["embedding"], now - 172800, 0.65),  # 128-dim
    ]

    with torch.no_grad():
        for mod_id, emb, ts, conf in modality_data:
            fusion_out = fusion(
                modality_id=mod_id,
                raw_embedding=emb,
                timestamp=ts,
                confidence=conf,
                latent_state=latent_state,
            )
            latent_state = fusion_out.latent_state
            print(f"  After {mod_id:>12}: fused_state={fusion_out.fused_state.shape}")

    # Final output
    print(f"\n  Final fusion output:")
    for attr in ['fused_state', 'latent_state', 'attn_weights', 'decay_weights']:
        if hasattr(fusion_out, attr):
            val = getattr(fusion_out, attr)
            if isinstance(val, torch.Tensor):
                print(f"    {attr}: {val.shape}")
            elif val is not None:
                print(f"    {attr}: {val}")

    print("\n" + "=" * 60)
    print("✓ FULL SENTINEL PIPELINE SMOKE TEST PASSED!")
    print("  All 5 modalities → 256-dim embeddings → Perceiver IO Fusion")
    print("=" * 60)


if __name__ == "__main__":
    main()
