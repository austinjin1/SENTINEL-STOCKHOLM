#!/usr/bin/env python
"""ZERO-SHOT USGS residual cache — mirrors cache_residuals_expanded.py exactly.

Feeds the FROZEN NEON-fine-tuned AquaSSM (checkpoints/sensor/aquassm_neon_ft.pt)
over the USGS NWIS .npz sites with NO retraining. Identical residual recipe to the
NEON cache: per-site robust input standardization on clean/non-near bins, STRIDE=8
(2 h) window grid, DO<4 +/-7 d `near` band, err[N,4] for pH,DO,turbidity,conductivity,
end_ts, cov (DO coverage), near, plus role/onset for EVALUATION-only gating.

Writes a SEPARATE cache so the NEON caches stay frozen:
  results/trackA_leadtime/usgs/residual_cache_usgs.npz
  results/trackA_leadtime/usgs/cache_meta_usgs.json
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
INDIR = ROOT / "data" / "processed" / "trackA_usgs"
OUTDIR = ROOT / "results" / "trackA_leadtime" / "usgs"
CACHE = OUTDIR / "residual_cache_usgs.npz"
MANIFEST = INDIR / "usgs_manifest.json"

DO_THR = 4.0
BUFFER_DAYS = 7
STRIDE = 8
DAY = 86400
CKPT = ROOT / "checkpoints" / "sensor" / "aquassm_neon_ft.pt"


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
    manifest = json.load(open(MANIFEST))
    sites = manifest["episode_sites"] + manifest["control_sites"]

    blob, meta = {}, {}
    for site in sites:
        npz = INDIR / f"{site}.npz"
        if not npz.exists():
            print(f"[{site}] SKIP no npz", flush=True)
            continue
        series = {k: v for k, v in np.load(npz, allow_pickle=True).items()}
        ts = series["ts"]
        role = str(series["role"])
        onset_i = int(series["onset_index"])
        near = hypoxia_near(series)

        # per-site robust input standardization on CLEAN, non-near bins only
        in_stats = {p: D.robust_stats(series[p], series[f"mask_{p}"] & ~near)
                    for p in D.REAL_PARAMS}
        chans = {p: D.zscore_channel(series[p], series[f"mask_{p}"],
                                     in_stats[p][0], in_stats[p][1]) for p in D.REAL_PARAMS}

        ends = np.arange(D.WIN, ts.size + 1, STRIDE)
        # VECTORIZED window construction (identical output to the per-window Python
        # loop: window n spans chans[p][end-WIN:end] for end=ends[n]). We build all
        # sliding windows once and stride them, avoiding a ~13K-iteration Python loop.
        from numpy.lib.stride_tricks import sliding_window_view
        starts = ends - D.WIN                                  # left edge of each window
        X = np.zeros((len(ends), D.WIN, 6), "float32")
        for j, p in zip(D.REAL_IDX, D.REAL_PARAMS):
            sw = sliding_window_view(chans[p], D.WIN)          # [ts.size-WIN+1, WIN]
            X[:, :, j] = sw[starts]                            # starts index left edges
        do_mask = series["mask_DO"].astype("float32")
        cov = sliding_window_view(do_mask, D.WIN)[starts].mean(axis=1).astype("float32")
        near_sw = sliding_window_view(near, D.WIN)[starts]
        near_w = near_sw.any(axis=1)
        err = D.recon_errors(model, X, device=dev, batch=1024)[:, D.REAL_IDX]
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
        print(f"[{site}] {role} n_win={len(ends)} near={int(near_w.sum())} "
              f"onset_i={onset_i}", flush=True)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, **blob)
    (OUTDIR / "cache_meta_usgs.json").write_text(json.dumps(meta, indent=2))
    print(f"\nWROTE {CACHE} and cache_meta_usgs.json ({len(meta)} sites)")


if __name__ == "__main__":
    main()
