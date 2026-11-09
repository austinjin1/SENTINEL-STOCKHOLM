#!/usr/bin/env python
"""ZERO-SHOT salinization (specific-conductance excursion) residual cache.

Adapts cache_residuals_usgs.py to the SALINIZATION anomaly CLASS. The FROZEN
NEON-fine-tuned AquaSSM (checkpoints/sensor/aquassm_neon_ft.pt) is fed over the
USGS road-salt / urban stream .npz sites with NO retraining (zero-shot). The
ONLY things retargeted from DO-hypoxia to conductance are the EVALUATION-only
event structure:

  * `near` band is CONDUCTANCE-based: +/-7 d of the conductance EVENT ONSET
    (role==episode), NOT DO<4. On controls (no qualifying excursion) near is all
    False. This is what the clean-baseline standardization and the precursor
    guardrail use.
  * `cov` is CONDUCTIVITY coverage (fraction of the window's conductance bins that
    are QC-observed), NOT DO coverage -- 14/41 of these sites have NO DO channel,
    and the event is a conductance event, so coverage must be measured on the
    channel that carries the label-free precursor.

The residual recipe is otherwise IDENTICAL to the frozen NEON/USGS cache:
per-site robust input standardization on clean (QC & finite) non-near bins,
STRIDE=8 (2 h) window grid, err[N,4] for pH,DO,turbidity,conductivity, end_ts.

The detector never saw any conductance label; the conductance EVENT threshold is
on the RAW conductance value (Phase 1), independent of the residual (non-circular,
Guardrail A). `near`/onset gate EVALUATION only (Guardrail B).

Writes a SEPARATE cache so the NEON/USGS caches stay frozen:
  results/trackA_leadtime/salinization/residual_cache_salt.npz
  results/trackA_leadtime/salinization/cache_meta_salt.json
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
INDIR = ROOT / "data" / "processed" / "trackA_salt"
OUTDIR = ROOT / "results" / "trackA_leadtime" / "salinization"
CACHE = OUTDIR / "residual_cache_salt.npz"
MANIFEST = OUTDIR / "conductance_events.json"

BUFFER_DAYS = 7
STRIDE = 8
DAY = 86400
CKPT = ROOT / "checkpoints" / "sensor" / "aquassm_neon_ft.pt"


def conductance_near(ts, onset_i):
    """+/-7 d around the conductance EVENT ONSET (episodes only; else all-False)."""
    near = np.zeros(ts.size, dtype=bool)
    if onset_i is not None and onset_i >= 0:
        t0 = ts[onset_i]
        near |= (ts >= t0 - BUFFER_DAYS * DAY) & (ts <= t0 + BUFFER_DAYS * DAY)
    return near


def main():
    import torch
    from numpy.lib.stride_tricks import sliding_window_view
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}", flush=True)
    model = D.load_detector(CKPT, device=dev)
    manifest = json.load(open(MANIFEST))
    sites = manifest["event_sites"] + manifest["control_sites"]

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
        near = conductance_near(ts, onset_i)

        # per-site robust input standardization on CLEAN, non-near bins only
        in_stats = {p: D.robust_stats(series[p], series[f"mask_{p}"] & ~near)
                    for p in D.REAL_PARAMS}
        chans = {p: D.zscore_channel(series[p], series[f"mask_{p}"],
                                     in_stats[p][0], in_stats[p][1]) for p in D.REAL_PARAMS}

        ends = np.arange(D.WIN, ts.size + 1, STRIDE)
        starts = ends - D.WIN
        X = np.zeros((len(ends), D.WIN, 6), "float32")
        for j, p in zip(D.REAL_IDX, D.REAL_PARAMS):
            sw = sliding_window_view(chans[p], D.WIN)
            X[:, :, j] = sw[starts]

        # COVERAGE on the CONDUCTIVITY channel (the event channel), not DO.
        cond_mask = series["mask_conductivity"].astype("float32")
        cov = sliding_window_view(cond_mask, D.WIN)[starts].mean(axis=1).astype("float32")
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
              f"cond_cov_med={np.median(cov):.2f} onset_i={onset_i}", flush=True)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, **blob)
    (OUTDIR / "cache_meta_salt.json").write_text(json.dumps(meta, indent=2))
    print(f"\nWROTE {CACHE} and cache_meta_salt.json ({len(meta)} sites)")


if __name__ == "__main__":
    main()
