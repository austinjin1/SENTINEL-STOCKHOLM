#!/usr/bin/env python
"""Track B, step 4 — train a FINAL fusion model on all co-located data and save it.

The CV in 03 measures generalization; this trains one deployable FusionClassifier
on every labeled co-located sample (chl-a held-out task) and saves the weights +
the input standardization, so the cohesive SENTINEL pipeline has a real fusion
artifact to load. The honest performance is the CV number from 03 (fusion ~0.65,
not better than the best single encoder) — this checkpoint does not change that.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

import _fusion as F

ROOT = Path(__file__).resolve().parents[3]
TB = ROOT / "data" / "processed" / "trackB"
CKDIR = ROOT / "checkpoints" / "fusion"
CKDIR.mkdir(parents=True, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS, BATCH = 120, 256


def main():
    emb = np.load(TB / "embeddings.npz", allow_pickle=True)
    row = {int(t): i for i, t in enumerate(emb["tile_idx"])}
    d = np.load(TB / "pairs_chl.npz", allow_pickle=True)
    keep = [(row[int(t)], int(t)) for t in d["tile_idx"] if int(t) in row]
    ridx = np.array([r[0] for r in keep])
    y = d["label"][[i for i, _ in enumerate(d["tile_idx"]) if int(d["tile_idx"][i]) in row]].astype("float32")
    sat, sen = emb["sat"][ridx], emb["sensor"][ridx]

    sc_s, sc_n = StandardScaler().fit(sat), StandardScaler().fit(sen)
    Xs = torch.tensor(sc_s.transform(sat).astype("float32"), device=DEV)
    Xn = torch.tensor(sc_n.transform(sen).astype("float32"), device=DEV)
    yt = torch.tensor(y, device=DEV)

    torch.manual_seed(0)
    model = F.FusionClassifier(num_latents=64).to(DEV)
    pos_w = torch.tensor([(len(y) - y.sum()) / max(1, y.sum())], device=DEV)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    n = len(y)
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            loss = bce(model(sat=Xs[b], sensor=Xn[b]), yt[b])
            loss.backward(); opt.step()

    torch.save({"model_state_dict": model.state_dict(),
                "sat_mean": sc_s.mean_.tolist(), "sat_scale": sc_s.scale_.tolist(),
                "sen_mean": sc_n.mean_.tolist(), "sen_scale": sc_n.scale_.tolist(),
                "task": "chl_gt8_ugL", "num_latents": 64,
                "cv_auroc_note": "fusion ~0.65, does NOT beat sensor-only ~0.64 (see results/trackB_fusion)"},
               CKDIR / "fusion_real_chl.pt")
    print(f"WROTE {CKDIR/'fusion_real_chl.pt'} (n={n}, pos={int(y.sum())})")


if __name__ == "__main__":
    main()
