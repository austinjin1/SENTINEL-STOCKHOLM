#!/usr/bin/env python
"""Cache AquaSSM reconstruction residuals for the NEW (expanded) Track A sites.

Identical residual pipeline to cache_residuals.py (same model checkpoint, same
per-site robust input standardization on clean/non-near bins, same STRIDE=8 2 h
grid, same DO<4 +/-7 d `near` band). Writes a SEPARATE cache so the original
audited residual_cache.npz / cache_meta.json stay frozen. Downstream, the merged
matched-FPR comparison loads BOTH caches.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LEAD_DIR = HERE.parent
sys.path.insert(0, str(LEAD_DIR))
import _detector as D  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
INDIR = ROOT / "data" / "processed" / "trackA"
OUTDIR = ROOT / "results" / "trackA_leadtime" / "aquassm_specificity"
CACHE = OUTDIR / "residual_cache_expanded.npz"

DO_THR = 4.0
BUFFER_DAYS = 7
STRIDE = 8
DAY = 86400
CKPT = ROOT / "checkpoints" / "sensor" / "aquassm_neon_ft.pt"

NEW_SITES = ["BLWA", "PRLA", "SUGG", "POSE", "PRPO", "REDB",
             "TECR", "TOMB", "WALK", "WLOU"]


def hypoxia_near(series):
    do, dom, ts = series["DO"], series["mask_DO"], series["ts"]
    hyp = dom & (do < DO_THR)
    near = np.zeros(ts.size, dtype=bool)
    for o in np.where(hyp)[0]:
        t0 = ts[o]
        near |= (ts >= t0 - BUFFER_DAYS * DAY) & (ts <= t0 + BUFFER_DAYS * DAY)
    return near


def main():
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}", flush=True)
    model = D.load_detector(CKPT, device=dev)
    manifest = json.load(open(INDIR / "expanded_manifest.json"))

    blob, meta = {}, {}
    for site in NEW_SITES:
        npz = INDIR / f"{site}.npz"
        if not npz.exists():
            print(f"[{site}] SKIP no npz", flush=True)
            continue
        series = {k: v for k, v in np.load(npz, allow_pickle=True).items()}
        ts = series["ts"]
        role = str(series["role"])
        onset_i = int(series["onset_index"])
        near = hypoxia_near(series)

        in_stats = {p: D.robust_stats(series[p], series[f"mask_{p}"] & ~near)
                    for p in D.REAL_PARAMS}
        chans = {p: D.zscore_channel(series[p], series[f"mask_{p}"],
                                     in_stats[p][0], in_stats[p][1]) for p in D.REAL_PARAMS}

        ends = np.arange(D.WIN, ts.size + 1, STRIDE)
        X = np.zeros((len(ends), D.WIN, 6), "float32")
        cov = np.zeros(len(ends), "float32")
        near_w = np.zeros(len(ends), bool)
        for n, end in enumerate(ends):
            sl = slice(end - D.WIN, end)
            for j, p in zip(D.REAL_IDX, D.REAL_PARAMS):
                X[n, :, j] = chans[p][sl]
            cov[n] = series["mask_DO"][sl].mean()
            near_w[n] = near[sl].any()
        err = D.recon_errors(model, X, device=dev, batch=256)[:, D.REAL_IDX]
        end_ts = ts[np.clip(ends - 1, 0, ts.size - 1)]

        blob[f"{site}__ends"] = ends.astype("int64")
        blob[f"{site}__end_ts"] = end_ts.astype("int64")
        blob[f"{site}__cov"] = cov.astype("float32")
        blob[f"{site}__near"] = near_w
        blob[f"{site}__err"] = err.astype("float32")
        meta[site] = {"role": role, "onset_index": onset_i,
                      "onset_ts": int(ts[onset_i]) if onset_i >= 0 else None,
                      "onset_utc": manifest["sites"][site].get("onset_utc"),
                      "n_windows": int(len(ends))}
        print(f"[{site}] {role} n_win={len(ends)} near={int(near_w.sum())} onset_i={onset_i}", flush=True)

    np.savez_compressed(CACHE, **blob)
    (OUTDIR / "cache_meta_expanded.json").write_text(json.dumps(meta, indent=2))
    print(f"\nWROTE {CACHE} and cache_meta_expanded.json ({len(meta)} sites)")


if __name__ == "__main__":
    main()
