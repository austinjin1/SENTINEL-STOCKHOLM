#!/usr/bin/env python
"""Track B, step 2 — cache real encoder embeddings for every labeled co-located tile.

WaterDroneNet cls_feat (384) from the real Sentinel-2 tile + AquaSSM embedding
(256) from the date-aligned in-situ sensor window. Only samples that have BOTH a
tile and a real sensor window (enough station history before the tile date) are
kept; the rest are dropped and counted (no zero-filling that would fake a modality).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import _fusion as F

ROOT = Path(__file__).resolve().parents[3]
PAIRS_NPZ = ROOT / "data" / "processed" / "satellite" / "drone_wq_pairs.npz"
TB = ROOT / "data" / "processed" / "trackB"


def main():
    z = np.load(PAIRS_NPZ, allow_pickle=True)
    md = json.loads(str(z["metadata"]))
    F.set_sensor_stats_from_targets(z["targets"])

    # union of labeled tile indices
    idxs = set()
    for f in ["pairs_chl.npz", "pairs_forecast.npz"]:
        p = TB / f
        if p.exists():
            idxs.update(np.load(p, allow_pickle=True)["tile_idx"].tolist())
    union = np.array(sorted(idxs))
    print(f"labeled tiles (union): {len(union)}", flush=True)

    # build sensor windows; keep only those with real history
    X, DT, kept = [], [], []
    for i in union:
        m = md[int(i)]
        w = F.build_sensor_window(m["site_id"], m["date"])
        if w is None:
            continue
        X.append(w[0]); DT.append(w[1]); kept.append(int(i))
    kept = np.array(kept)
    print(f"with real sensor window: {len(kept)} (dropped {len(union)-len(kept)})", flush=True)
    X = np.stack(X); DT = np.stack(DT)

    images = z["images"]                      # (12000,4,224,224) memmap-ish
    imgs = images[kept]                       # gather labeled subset

    sat_enc = F.load_satellite_encoder()
    sen_enc = F.load_sensor_encoder()
    print("encoders loaded; embedding...", flush=True)
    sat_emb = F.embed_satellite(sat_enc, imgs)
    sen_emb = F.embed_sensor(sen_enc, X, DT)
    print("sat_emb", sat_emb.shape, "sen_emb", sen_emb.shape, flush=True)

    np.savez(TB / "embeddings.npz",
             tile_idx=kept, sat=sat_emb.astype("float32"),
             sensor=sen_emb.astype("float32"))
    print(f"WROTE {TB/'embeddings.npz'}")


if __name__ == "__main__":
    main()
