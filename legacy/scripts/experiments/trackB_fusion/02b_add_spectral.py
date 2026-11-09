#!/usr/bin/env python
"""Track B 2b — cache chl-sensitive spectral features for each embedded tile."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import _fusion as F

ROOT = Path(__file__).resolve().parents[3]
TB = ROOT / "data" / "processed" / "trackB"
PAIRS = ROOT / "data" / "processed" / "satellite" / "drone_wq_pairs.npz"


def main():
    emb = np.load(TB / "embeddings.npz", allow_pickle=True)
    tile_idx = emb["tile_idx"]
    images = np.load(PAIRS, allow_pickle=True)["images"]
    spec = np.array([F.spectral_feats(images[int(i)]) for i in tile_idx], dtype="float32")
    spec = np.nan_to_num(spec, nan=0.0, posinf=0.0, neginf=0.0)
    np.savez(TB / "embeddings_v2.npz", tile_idx=tile_idx,
             sat=emb["sat"], sensor=emb["sensor"], spectral=spec)
    print(f"WROTE embeddings_v2.npz  spectral {spec.shape}")


if __name__ == "__main__":
    main()
