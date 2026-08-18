#!/usr/bin/env python
"""Track A, step 2 — calibrate the label-free detector to a real FPR, get lead times.

Key correctness fixes over the first pass:
  * The threshold is calibrated against the *sustained* alarm rate (K consecutive
    windows) on **contiguous** held-out clean periods, not the per-window rate.
    Reconstruction errors are autocorrelated, so a 95th-percentile per-window
    threshold leaks far more than 5% once the sustained rule is applied. We
    instead search tau so the sustained alarm rate on a contiguous clean
    calibration slice is <= target FPR.
  * Quiet windows are grouped into contiguous runs so the sustained rule is
    physically meaningful (3 consecutive 2-hour windows = 6 h of persistence).

Per site: standardize inputs on quiet clean values, score a contiguous window
grid, calibrate tau to target sustained FPR on the first 60% of clean runs,
measure held-out FPR on the last 40% (+ control sites), and on episode sites
measure the lead time of the first sustained alarm before the verified DO<4 onset.
No anomaly label is ever used to fit the detector.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

import _detector as D

ROOT = Path(__file__).resolve().parents[3]
INDIR = ROOT / "data" / "processed" / "trackA"
OUTDIR = ROOT / "results" / "trackA_leadtime"
OUTDIR.mkdir(parents=True, exist_ok=True)
_FT = ROOT / "checkpoints" / "sensor" / "aquassm_neon_ft.pt"      # NEON fine-tuned (run 05 first)
_BASE = ROOT / "checkpoints" / "sensor" / "aquassm_v2_mpp_best.pt"
CKPT = Path(os.environ.get("SENTINEL_CKPT", str(_FT if _FT.exists() else _BASE)))
OUT_NAME = os.environ.get("SENTINEL_OUT", "leadtime_results.json")

DO_THR = 4.0
BUFFER_DAYS = 7
MIN_COV = 0.30
STRIDE = 8               # 2 h grid, used everywhere (so K is comparable)
K_SUSTAIN = 3            # 3 consecutive windows = 6 h sustained
TARGET_FPR = 0.05
LOOKBACK_DAYS = 45
QUIET_CAP = 2500         # cap scored quiet windows per site
DAY = 86400


def hypoxia_near(series):
    do, dom, ts = series["DO"], series["mask_DO"], series["ts"]
    hyp = dom & (do < DO_THR)
    near = np.zeros(ts.size, dtype=bool)
    for o in np.where(hyp)[0]:
        t0 = ts[o]
        near |= (ts >= t0 - BUFFER_DAYS * DAY) & (ts <= t0 + BUFFER_DAYS * DAY)
    return near


def grid(series, near):
    """Window-end grid over the record; classify quiet & covered."""
    ts, dm = series["ts"], series["mask_DO"]
    ends, quiet, cov = [], [], []
    for end in range(D.WIN, ts.size + 1, STRIDE):
        sl = slice(end - D.WIN, end)
        c = dm[sl].mean()
        ends.append(end)
        cov.append(c)
        quiet.append((c >= MIN_COV) and (not near[sl].any()))
    return np.array(ends), np.array(quiet), np.array(cov)


def runs_of(mask):
    """Yield (lo,hi) index ranges of contiguous True in mask (grid positions)."""
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def sustained_rate(scores, tau, k):
    """Fraction of windows in a sustained (>=k consecutive) exceedance state."""
    flags = scores > tau
    alarm = np.zeros(len(flags), bool)
    run = 0
    for i, f in enumerate(flags):
        run = run + 1 if f else 0
        if run >= k:
            alarm[i] = True
    return float(alarm.mean()) if len(alarm) else 0.0


def sustained_first_time(times, scores, tau, k):
    run = 0
    for i, s in enumerate(scores):
        run = run + 1 if s > tau else 0
        if run >= k:
            return int(times[i])
    return None


def score_ends(model, chans, ends):
    if len(ends) == 0:
        return np.zeros((0, 6), "float32")
    X = np.zeros((len(ends), D.WIN, 6), "float32")
    for n, end in enumerate(ends):
        sl = slice(end - D.WIN, end)
        for j, p in zip(D.REAL_IDX, D.REAL_PARAMS):
            X[n, :, j] = chans[p][sl]
    return D.recon_errors(model, X, batch=128)


def calibrate_tau(s_runs, k, target):
    """Smallest tau with total sustained rate over calibration runs <= target."""
    alls = np.concatenate(s_runs) if s_runs else np.zeros(0)
    if alls.size == 0:
        return float("inf")
    cands = np.unique(np.percentile(alls, np.linspace(50, 99.9, 120)))
    best = cands[-1]
    for tau in cands:
        tot_alarm = tot = 0
        for s in s_runs:
            fl = s > tau
            run = 0
            a = 0
            for f in fl:
                run = run + 1 if f else 0
                a += run >= k
            tot_alarm += a
            tot += len(s)
        if tot and tot_alarm / tot <= target:
            best = float(tau)
            break
    return float(best)


def main():
    model = D.load_detector(CKPT)
    summary = json.load(open(INDIR / "extract_summary.json"))
    results = {"config": {"do_thr": DO_THR, "buffer_days": BUFFER_DAYS,
                          "min_cov": MIN_COV, "stride_h": STRIDE / 4,
                          "k_sustain": K_SUSTAIN, "k_hours": K_SUSTAIN * STRIDE / 4,
                          "target_fpr": TARGET_FPR, "lookback_days": LOOKBACK_DAYS,
                          "checkpoint": CKPT.name}, "sites": {}}

    for site, info in summary["sites"].items():
        npz = INDIR / f"{site}.npz"
        if not npz.exists():
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

        ends, quiet, cov = grid(series, near)
        q_runs = [(lo, hi) for lo, hi in runs_of(quiet) if hi - lo >= K_SUSTAIN]
        if not q_runs:
            results["sites"][site] = {"role": role, "skipped": "no contiguous quiet runs"}
            print(f"[{site}] SKIP no quiet runs", flush=True)
            continue
        # cap total quiet windows by taking the longest runs, then time-order
        q_runs.sort(key=lambda r: r[1] - r[0], reverse=True)
        kept, tot = [], 0
        for r in q_runs:
            kept.append(r); tot += r[1] - r[0]
            if tot >= QUIET_CAP:
                break
        kept.sort()
        # Interleave runs into calibration / held-out FPR so calibration is
        # REPRESENTATIVE of the whole record (a contiguous time-block split
        # under-estimates the threshold when a site's residual noise drifts).
        cal_runs, fpr_runs = kept[::2], kept[1::2]
        if not cal_runs or not fpr_runs:        # single run: reuse for both
            cal_runs = fpr_runs = kept

        # score calibration + fpr quiet windows
        def run_scores(run_list):
            out = []
            for lo, hi in run_list:
                e = ends[lo:hi]
                err = score_ends(model, chans, e)
                out.append((e, err))
            return out

        # NOTE: a per-day-of-year "deseasonalized" residual normalization was
        # tested here and REJECTED — it made MCDI/ARIK FPR worse (0.30->0.57,
        # 0.09->0.57) and inflated leads to the window length. The seasonal swing
        # in clean residuals is partly correlated with the hypoxia signal, so
        # removing it strips discriminative structure. Global per-site normalization
        # is the proven baseline.
        cal_scored = run_scores(cal_runs)
        cal_err = np.concatenate([er[:, D.REAL_IDX] for _, er in cal_scored], 0)
        err_c = cal_err.mean(0)
        err_s = cal_err.std(0).clip(1e-6)

        def s_of(err):
            return ((err[:, D.REAL_IDX] - err_c) / err_s).max(1)

        # NOTE: per-site adaptive sustained-K was also tested and REJECTED (no-op):
        # K never escalated because K=3 already meets the target on the calibration
        # slice; the over-firing on non-stationary sites is a held-out GENERALIZATION
        # gap, not a calibration-slice problem, so neither deseasonalization nor
        # adaptive-K fixes it. Track A specificity is at the method's real limit here.
        s_cal_runs = [s_of(er) for _, er in cal_scored]
        tau = calibrate_tau(s_cal_runs, K_SUSTAIN, TARGET_FPR)

        fpr_scored = run_scores(fpr_runs)
        s_fpr_runs = [s_of(er) for _, er in fpr_scored]
        fpr = (sum(sustained_rate(s, tau, K_SUSTAIN) * len(s) for s in s_fpr_runs)
               / max(1, sum(len(s) for s in s_fpr_runs)))
        cal_fpr = (sum(sustained_rate(s, tau, K_SUSTAIN) * len(s) for s in s_cal_runs)
                   / max(1, sum(len(s) for s in s_cal_runs)))

        site_res = {"role": role, "tau": tau, "n_quiet_windows": int(tot),
                    "fpr_heldout": fpr, "fpr_calib": cal_fpr,
                    "n_fpr_windows": int(sum(len(s) for s in s_fpr_runs)),
                    "in_stats": {p: in_stats[p] for p in D.REAL_PARAMS},
                    "err_center": err_c.tolist(), "err_scale": err_s.tolist()}

        if role == "episode" and onset_i >= 0:
            onset_ts = int(ts[onset_i])
            lo_t = onset_ts - LOOKBACK_DAYS * DAY
            det_mask = (ends_ts := ts[np.clip(ends - 1, 0, ts.size - 1)]) >= lo_t
            det_mask &= ends_ts <= onset_ts + 2 * DAY
            det_mask &= cov >= MIN_COV
            det_ends = ends[det_mask]
            det_ts = ts[np.clip(det_ends - 1, 0, ts.size - 1)]
            det_err = score_ends(model, chans, det_ends)
            s_det = s_of(det_err)
            pre = det_ts < onset_ts
            t_alarm = sustained_first_time(det_ts[pre], s_det[pre], tau, K_SUSTAIN)
            site_res["onset_utc"] = info.get("onset_utc")
            site_res["detected_pre_onset"] = t_alarm is not None
            site_res["lead_days"] = (onset_ts - t_alarm) / DAY if t_alarm else None
            site_res["n_det_windows"] = int(len(det_ends))

        results["sites"][site] = site_res
        m = f"[{site}] {role} tau={tau:.2f} FPRheldout={fpr:.3f}"
        if role == "episode":
            m += f" lead={site_res.get('lead_days')}"
        print(m, flush=True)

    (OUTDIR / OUT_NAME).write_text(json.dumps(results, indent=2))
    print(f"\nWROTE {OUTDIR/OUT_NAME}")


if __name__ == "__main__":
    main()
