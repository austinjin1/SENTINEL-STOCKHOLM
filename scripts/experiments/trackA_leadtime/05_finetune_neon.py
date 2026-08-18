#!/usr/bin/env python
"""Track A 5 — self-supervised fine-tune of the AquaSSM reconstruction backbone on NEON.

The shipped backbone was MPP-pretrained on USGS; on NEON its reconstruction
residuals separate hypoxia from clean water only moderately (3/6 sites over-fire).
Here we continue masked-parameter-prediction (MPP) — fully self-supervised, NO
anomaly labels — on NEON *clean* windows only, so the model learns NEON "normal"
dynamics. A better normal model => lower residual on clean water (lower FPR) and
higher residual on genuine anomalies (better separation). Hypoxic/near-event
windows are EXCLUDED from fine-tuning so we never teach the model the anomalies.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import _detector as D

ROOT = Path(__file__).resolve().parents[3]
INDIR = ROOT / "data" / "processed" / "trackA"
SRC_CKPT = ROOT / "checkpoints" / "sensor" / "aquassm_v2_mpp_best.pt"
OUT_CKPT = ROOT / "checkpoints" / "sensor" / "aquassm_neon_ft.pt"

DO_THR, BUFFER_DAYS, MIN_COV, STRIDE = 4.0, 7, 0.30, 16
PER_SITE_CAP, EPOCHS, BATCH, DAY = 1500, 8, 128, 86400


def clean_windows(site):
    npz = INDIR / f"{site}.npz"
    if not npz.exists():
        return np.zeros((0, D.WIN, 6), "float32")
    s = {k: v for k, v in np.load(npz, allow_pickle=True).items()}
    do, dom, ts = s["DO"], s["mask_DO"], s["ts"]
    hyp = dom & (do < DO_THR)
    near = np.zeros(ts.size, bool)
    for o in np.where(hyp)[0]:
        near |= (ts >= ts[o] - BUFFER_DAYS * DAY) & (ts <= ts[o] + BUFFER_DAYS * DAY)
    in_stats = {p: D.robust_stats(s[p], s[f"mask_{p}"] & ~near) for p in D.REAL_PARAMS}
    chans = {p: D.zscore_channel(s[p], s[f"mask_{p}"], in_stats[p][0], in_stats[p][1])
             for p in D.REAL_PARAMS}
    ends = []
    for e in range(D.WIN, ts.size + 1, STRIDE):
        sl = slice(e - D.WIN, e)
        if (not near[sl].any()) and dom[sl].mean() >= MIN_COV:
            ends.append(e)
    if len(ends) > PER_SITE_CAP:
        ends = [ends[i] for i in np.linspace(0, len(ends) - 1, PER_SITE_CAP).astype(int)]
    X = np.zeros((len(ends), D.WIN, 6), "float32")
    for n, e in enumerate(ends):
        for j, p in zip(D.REAL_IDX, D.REAL_PARAMS):
            X[n, :, j] = chans[p][e - D.WIN:e]
    return X


def main():
    import json
    sites = list(json.load(open(INDIR / "extract_summary.json"))["sites"].keys())
    Xs = [clean_windows(s) for s in sites]
    X = np.concatenate([x for x in Xs if len(x)], 0)
    print(f"NEON clean windows for fine-tuning: {len(X)} from {sum(1 for x in Xs if len(x))} sites", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = D.load_detector(SRC_CKPT, dev)            # SensorEncoder w/ MPP weights
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-5)
    dt = torch.full((BATCH, D.WIN), D.DELTA_T, device=dev)
    n = len(X)
    for ep in range(EPOCHS):
        perm = np.random.permutation(n)
        tot = 0.0
        for i in range(0, n, BATCH):
            b = perm[i:i + BATCH]
            xb = torch.from_numpy(X[b]).to(dev)
            out = model.forward_pretrain(xb, delta_ts=dt[:len(b)])  # MPP self-supervised
            loss = out["loss"]
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)
        print(f"  epoch {ep+1}/{EPOCHS}  MPP loss {tot/n:.4f}", flush=True)

    torch.save(model.state_dict(), OUT_CKPT)
    print(f"WROTE {OUT_CKPT}")


if __name__ == "__main__":
    main()
