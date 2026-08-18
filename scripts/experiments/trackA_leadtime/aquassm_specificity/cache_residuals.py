#!/usr/bin/env python
"""Cache raw per-window per-param reconstruction residuals for every Track A site.

Runs the AquaSSM backbone ONCE over the full contiguous window-end grid of each
site and stores, per window:
  * end_index  (grid position; last bin index of the window)
  * end_ts     (UTC seconds of the last bin)
  * do_cov     (fraction of DO bins QC-observed in the window)
  * near       (bool: window overlaps the DO<4 +/- buffer_days band = hypoxia-adjacent)
  * err[4]     (mean reconstruction error for pH,DO,turbidity,conductivity)

All downstream scoring/specificity experiments then run purely on CPU from this
cache, so we can iterate on the detector rule without re-running the model.

No anomaly labels are used. `near` is derived only from the DO<4 physical
definition (already the event definition), and is used ONLY to exclude
hypoxia-adjacent windows from clean-FPR evaluation -- never fed to the score.
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
OUTDIR.mkdir(parents=True, exist_ok=True)
CACHE = OUTDIR / "residual_cache.npz"

DO_THR = 4.0
BUFFER_DAYS = 7
MIN_COV = 0.30
STRIDE = 8                # 2 h grid (matches baseline)
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
    model = D.load_detector(CKPT, device=dev)
    summary = json.load(open(INDIR / "extract_summary.json"))

    blob = {}
    meta = {}
    for site in summary["sites"]:
        npz = INDIR / f"{site}.npz"
        if not npz.exists():
            continue
        series = {k: v for k, v in np.load(npz, allow_pickle=True).items()}
        ts = series["ts"]
        role = str(series["role"])
        onset_i = int(series["onset_index"])
        near = hypoxia_near(series)

        # robust per-site input standardization on clean, non-near values (baseline)
        in_stats = {p: D.robust_stats(series[p], series[f"mask_{p}"] & ~near)
                    for p in D.REAL_PARAMS}
        chans = {p: D.zscore_channel(series[p], series[f"mask_{p}"],
                                     in_stats[p][0], in_stats[p][1]) for p in D.REAL_PARAMS}

        ends = np.arange(D.WIN, ts.size + 1, STRIDE)
        # build X windows
        X = np.zeros((len(ends), D.WIN, 6), "float32")
        cov = np.zeros(len(ends), "float32")
        near_w = np.zeros(len(ends), bool)
        for n, end in enumerate(ends):
            sl = slice(end - D.WIN, end)
            for j, p in zip(D.REAL_IDX, D.REAL_PARAMS):
                X[n, :, j] = chans[p][sl]
            cov[n] = series["mask_DO"][sl].mean()
            near_w[n] = near[sl].any()
        err = D.recon_errors(model, X, device=dev, batch=256)[:, D.REAL_IDX]  # [N,4]
        end_ts = ts[np.clip(ends - 1, 0, ts.size - 1)]

        blob[f"{site}__ends"] = ends.astype("int64")
        blob[f"{site}__end_ts"] = end_ts.astype("int64")
        blob[f"{site}__cov"] = cov.astype("float32")
        blob[f"{site}__near"] = near_w
        blob[f"{site}__err"] = err.astype("float32")
        meta[site] = {"role": role,
                      "onset_index": onset_i,
                      "onset_ts": int(ts[onset_i]) if onset_i >= 0 else None,
                      "onset_utc": summary["sites"][site].get("onset_utc"),
                      "n_windows": int(len(ends))}
        print(f"[{site}] {role} n_win={len(ends)} near={int(near_w.sum())} "
              f"onset_i={onset_i}", flush=True)

    np.savez_compressed(CACHE, **blob)
    (OUTDIR / "cache_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nWROTE {CACHE} and cache_meta.json ({len(meta)} sites)")


if __name__ == "__main__":
    main()
