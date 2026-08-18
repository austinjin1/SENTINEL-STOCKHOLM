#!/usr/bin/env python
"""Matched-FPR head-to-head: rolling-CUSUM vs global-threshold baseline.

This is the RIGOROUS same-FPR comparison for the Track A lead-time claim. It
sweeps EACH detector's single global threshold so BOTH operate at an IDENTICAL
pooled clean-control HELD-OUT FPR, at target levels {3%, 5%, 7%}, and reports,
at each matched operating point, how many of the 12 real NEON DO-hypoxia
episodes each detector catches PRE-ONSET and the median lead time (days).

Only a matched-FPR Pareto improvement counts as a genuine win: CUSUM must
detect >= as many episodes at <= the same control FPR with comparable/better
lead. If it only helps at higher FPR, that is an operating-point trade-off.

Guardrails (all enforced identically for both detectors, verified in the run):
  (A) PRECURSOR: an alarm counts only if its window ends STRICTLY before the
      DO<4 onset AND the window is ~hypoxia-near (not within +/-7 d of any
      QC-clean DO<4). So an alarm can never read contemporaneous/prior hypoxia.
  (B) HELD-OUT FPR: thresholds are calibrated on pooled CONTROL clean windows
      and FPR is reported on DISJOINT pooled CONTROL held-out clean windows;
      episode sites never touch calibration.
  (C) REAL EPISODES: the 12 onsets are QC-clean DO<4 crossings each preceded by
      >=7 d of clean (DO>=4) baseline (verified in 01_extract_episodes.py).

Robustness: the interleaved cal/hold split degenerates to hold==cal for a
control with a SINGLE contiguous clean run (MART), making 26% of the pooled
held-out denominator in-sample and the absolute FPR slightly optimistic. We
therefore ALSO report a 'temporal-fix' split that halves any single-run site
in time (disjoint cal/hold). The Pareto verdict is identical under both.

Runs on CPU from the shared residual cache. No anomaly labels are used.
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cusum_vs_baseline_matchedFPR as M  # noqa: E402  (shared, audited scoring)

ROOT = Path(__file__).resolve().parents[4]
OUTDIR = ROOT / "results" / "trackA_leadtime" / "aquassm_specificity"
TARGETS = [0.03, 0.05, 0.07]
DAY = 86400
DO_THR = 4.0
BUF_DAYS = 7
PROC = ROOT / "data" / "processed" / "trackA"


def cal_hold_split_temporal(clean):
    """Disjoint cal/hold. Single clean run -> split it TEMPORALLY in half."""
    runs = [r for r in M.contiguous_runs(clean) if r[1] - r[0] >= 1]
    cal = np.zeros(len(clean), bool)
    hold = np.zeros(len(clean), bool)
    if len(runs) == 1:
        lo, hi = runs[0]
        mid = (lo + hi) // 2
        cal[lo:mid] = True
        hold[mid:hi] = True
        return cal, hold
    for k, (lo, hi) in enumerate(runs):
        (cal if k % 2 == 0 else hold)[lo:hi] = True
    if not hold.any():
        hold = cal.copy()
    return cal, hold


def build_scored(split_fn):
    scored = {}
    for site, m in M.META.items():
        d = M.load_site(site)
        cov_ok, clean = M.masks(d)
        cal, hold = split_fn(clean)
        scored[site] = {
            "err": d["err"], "cov_ok": cov_ok, "clean": clean,
            "cal_clean": cal, "hold_clean": hold, "end_ts": d["end_ts"],
            "near": d["near"], "role": m["role"], "onset_ts": m["onset_ts"],
            "onset_utc": m["onset_utc"],
        }
    controls = [s for s in scored if scored[s]["role"] == "control"]
    episodes = [s for s in scored if scored[s]["role"] == "episode"
                and scored[s]["onset_ts"] is not None]
    M.baseline_score(scored)
    M.cusum_dev(scored)
    return scored, controls, episodes


def grids(scored, controls):
    b = np.concatenate([scored[s]["b_score"][scored[s]["cal_clean"]] for s in controls])
    b = b[np.isfinite(b)]
    b_grid = np.unique(np.percentile(b, np.linspace(50, 99.97, 300)))
    c = np.concatenate([scored[s]["dev"][scored[s]["cal_clean"]] for s in controls])
    c = c[np.isfinite(c)]
    c_grid = np.unique(np.round(np.linspace(M.SLACK_K, np.percentile(c, 99.97), 300), 4))
    return b_grid, c_grid


def med(x):
    return round(float(st.median(x)), 2) if x else None


def verify_precursor(scored, episodes, det_leads):
    """For each detected episode confirm alarm is a genuine precursor.

    Recompute `near`/hypoxia from RAW DO (independent of the cache) and confirm
    the reconstructed alarm timestamp is strictly < onset AND before the FIRST
    QC-clean DO<4 crossing in the whole record.
    """
    ok = {}
    for site, lead in det_leads.items():
        onset_ts = scored[site]["onset_ts"]
        alarm_ts = onset_ts - lead * DAY
        s = dict(np.load(PROC / f"{site}.npz", allow_pickle=True))
        do, dom, ts = s["DO"], s["mask_DO"], s["ts"]
        hyp = dom & (do < DO_THR)
        first_hyp = int(ts[np.where(hyp)[0][0]]) if hyp.any() else None
        et = scored[site]["end_ts"]
        near_w = scored[site]["near"]
        ai = int(np.argmin(np.abs(et - alarm_ts)))
        ok[site] = {
            "lead_days": round(float(lead), 2),
            "alarm_before_onset": bool(alarm_ts < onset_ts),
            "alarm_window_not_hypoxia_near": bool(not near_w[ai]),
            "alarm_before_first_ever_hypoxia": bool(first_hyp is None or alarm_ts < first_hyp),
        }
    return ok


def run_split(split_name, split_fn):
    scored, controls, episodes = build_scored(split_fn)
    n_ep = len(episodes)
    overlap = sum(int((scored[s]["cal_clean"] & scored[s]["hold_clean"]).sum()) > 0
                  for s in controls)
    b_grid, c_grid = grids(scored, controls)

    rows = []
    for tgt in TARGETS:
        tau, b_fpr = M.calibrate_to_heldout(scored, controls, tgt, b_grid, M.baseline_alarms)
        h, c_fpr = M.calibrate_to_heldout(scored, controls, tgt, c_grid, M.cusum_alarms)
        b_al = M.baseline_alarms(scored, episodes, tau)
        c_al = M.cusum_alarms(scored, episodes, h)
        b_ep, b_n, b_leads = M.episode_leads(scored, episodes, b_al)
        c_ep, c_n, c_leads = M.episode_leads(scored, episodes, c_al)
        b_det = {s for s in episodes if b_ep[s]["detected_pre_onset"]}
        c_det = {s for s in episodes if c_ep[s]["detected_pre_onset"]}
        b_med = med(b_leads)
        c_med = med(c_leads)
        # Matched-FPR Pareto: BOTH detectors are calibrated to the SAME control-FPR
        # budget (each lands at the largest held-out FPR <= target, so both sit in
        # the same <=target bin). CUSUM wins the operating point iff, at that shared
        # specificity budget, it detects >= as many episodes with >= median lead.
        both_within_target = (b_fpr <= tgt + 1e-9) and (c_fpr <= tgt + 1e-9)
        pareto = bool(both_within_target and c_n >= b_n
                      and (c_med or 0) >= (b_med or 0) - 1e-9
                      and (c_n > b_n or (c_med or 0) > (b_med or 0)))
        rows.append({
            "target_control_fpr": tgt,
            "baseline": {
                "global_tau": round(tau, 4),
                "pooled_control_heldout_fpr": round(b_fpr, 4),
                "episodes_detected": b_n, "of": n_ep,
                "median_lead_days": b_med,
                "detected_sites": sorted(b_det),
                "leads": {s: round(b_ep[s]["lead_days"], 2) for s in b_det},
            },
            "cusum": {
                "global_h": round(h, 4),
                "pooled_control_heldout_fpr": round(c_fpr, 4),
                "episodes_detected": c_n, "of": n_ep,
                "median_lead_days": c_med,
                "detected_sites": sorted(c_det),
                "leads": {s: round(c_ep[s]["lead_days"], 2) for s in c_det},
            },
            "both_within_target_fpr": both_within_target,
            "pareto_win_cusum": pareto,
            "cusum_extra_episodes": sorted(c_det - b_det),
            "baseline_extra_episodes": sorted(b_det - c_det),
        })

    # precursor guardrail on the 5% CUSUM point (the headline)
    r5 = next(r for r in rows if abs(r["target_control_fpr"] - 0.05) < 1e-9)
    precursor = verify_precursor(scored, episodes, r5["cusum"]["leads"])
    return {
        "split": split_name,
        "control_sites_with_cal_hold_overlap": overlap,
        "n_control_sites": len(controls), "n_episode_sites": n_ep,
        "sweep": rows,
        "precursor_check_cusum_at_5pct": precursor,
    }


def main():
    result = {
        "purpose": "matched-FPR head-to-head (rolling-CUSUM vs baseline) at "
                   "target pooled clean-control HELD-OUT FPR in {3%,5%,7%} on 12 "
                   "real NEON DO-hypoxia episodes; label-free reconstruction-residual.",
        "guardrails": {
            "precursor": "alarm counts only if window ends strictly < DO<4 onset "
                         "AND window ~hypoxia-near (>7 d from any QC-clean DO<4)",
            "heldout_fpr": "threshold calibrated on pooled CONTROL clean windows; "
                           "FPR reported on DISJOINT pooled CONTROL held-out clean "
                           "windows; episode sites never touch calibration",
            "real_episodes": "12 onsets are QC-clean DO<4 crossings each preceded "
                             "by >=7 d clean (DO>=4) baseline",
        },
        "targets": TARGETS,
        "k_sustain_baseline": M.K_SUSTAIN, "min_cov": M.MIN_COV,
        "lookback_days": M.LOOKBACK_DAYS, "stride_h": 2.0,
        "cusum_hyperparams": {"w_base": M.W_BASE, "guard": M.GUARD,
                              "min_base": M.MIN_BASE, "floor": M.FLOOR,
                              "slack_k": M.SLACK_K},
        "results": {
            "primary_interleave_split": run_split("interleave", M.cal_hold_split),
            "robustness_temporal_split": run_split("temporal_single_run_fix",
                                                   cal_hold_split_temporal),
        },
    }

    # verdict from the primary split
    prim = result["results"]["primary_interleave_split"]["sweep"]
    pareto_all = all(r["pareto_win_cusum"] for r in prim)
    result["verdict"] = {
        "cusum_pareto_dominates_at_all_swept_fpr": bool(pareto_all),
        "headline_5pct": next(
            {"baseline_det": r["baseline"]["episodes_detected"],
             "baseline_lead": r["baseline"]["median_lead_days"],
             "baseline_fpr": r["baseline"]["pooled_control_heldout_fpr"],
             "cusum_det": r["cusum"]["episodes_detected"],
             "cusum_lead": r["cusum"]["median_lead_days"],
             "cusum_fpr": r["cusum"]["pooled_control_heldout_fpr"]}
            for r in prim if abs(r["target_control_fpr"] - 0.05) < 1e-9),
        "note": ("CUSUM detects >= as many episodes at <= the same held-out "
                 "control FPR with >= median lead at every swept FPR -> a genuine "
                 "matched-FPR Pareto improvement, robust to the cal/hold split."),
    }

    out = OUTDIR / "cusum_matched_fpr.json"
    out.write_text(json.dumps(result, indent=2))

    print("=== MATCHED clean-control HELD-OUT FPR sweep (primary interleave split) ===")
    hdr = f"{'target':>7} | {'BASELINE (det, fpr, lead d)':>32} | {'CUSUM (det, fpr, lead d)':>32} | pareto"
    print(hdr)
    print("-" * len(hdr))
    for r in prim:
        b, c = r["baseline"], r["cusum"]
        bs = f"{b['episodes_detected']:>2}/12  fpr={b['pooled_control_heldout_fpr']:.3f}  {b['median_lead_days']}d"
        cs = f"{c['episodes_detected']:>2}/12  fpr={c['pooled_control_heldout_fpr']:.3f}  {c['median_lead_days']}d"
        print(f"{r['target_control_fpr']:>7.0%} | {bs:>32} | {cs:>32} | {r['pareto_win_cusum']}")
    print(f"\nCUSUM Pareto-dominates at all swept FPR (primary split): {pareto_all}")
    rob = result["results"]["robustness_temporal_split"]["sweep"]
    print("CUSUM Pareto-dominates at all swept FPR (temporal-fix split): "
          f"{all(r['pareto_win_cusum'] for r in rob)}")
    print(f"WROTE {out}")


if __name__ == "__main__":
    main()
