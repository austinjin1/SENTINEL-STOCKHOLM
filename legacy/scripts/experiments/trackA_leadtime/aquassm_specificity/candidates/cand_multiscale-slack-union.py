#!/usr/bin/env python
"""CANDIDATE: Multi-scale slack union CUSUM (key=multiscale-slack-union).

IDEA
----
OR alarms from a-priori CUSUM channels at different rolling-baseline scales
(w_base) and different slack (slack_k): a FAST channel (short w_base, small
slack) catches abrupt shifts, a SLOW channel (long w_base) catches sustained
drift. The union first-alarm time is the MIN over channels, so lead can only be
>= the single-channel current headline (union only ADDS alarms).

A-PRIORI CHANNEL SET (fixed; NOT tuned on leads):
  channel 1: w_base=120, slack_k=0.5   (fast)
  channel 2: w_base=240, slack_k=1.0   (mid == current headline single channel)
  channel 3: w_base=480, slack_k=1.0   (slow drift)
Shared fixed: guard=12, min_base=20, floor=1.0.
A SINGLE shared threshold `thr` (the CUSUM alarm level h, common to all channels)
is calibrated by M.calibrate_to_heldout so the UNION pooled control HELD-OUT FPR
is <= 5%. Fallback: shared quantile of pooled control cal-clean dev if the grid
degenerates. Eval via M.episode_leads.

GUARDRAILS honored:
  (A) label-free: channels a priori; onset/near gate evaluation only.
  (B) precursor-only: enforced by M.episode_leads.
  (C) matched held-out FPR: single shared thr calibrated on pooled CONTROL
      held-out clean windows; episodes never touch calibration.
  (D) never tune on leads: all hyperparameters fixed above.
  (E) reported both episodes-detected AND median lead.

Runs on CPU from the shared residual caches (E.build_merged_scored).
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SPEC_DIR = HERE.parent
sys.path.insert(0, str(SPEC_DIR))

import cusum_vs_baseline_matchedFPR as M  # noqa: E402
import expanded_matched_fpr as E  # noqa: E402
import rolling_cusum as RC  # noqa: E402
from matched_fpr_sweep import cal_hold_split_temporal  # noqa: E402

ROOT = Path(__file__).resolve().parents[5]
OUTDIR = ROOT / "results" / "trackA_leadtime" / "aquassm_specificity" / "candidates"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ------------------------- A-PRIORI HYPERPARAMETERS -------------------------
# (w_base, slack_k) per channel; shared guard/min_base/floor.
CHANNELS = [
    {"w_base": 120, "slack_k": 0.5},   # fast: abrupt shifts
    {"w_base": 240, "slack_k": 1.0},   # mid: current headline channel
    {"w_base": 480, "slack_k": 1.0},   # slow: sustained drift
]
GUARD = 12
MIN_BASE = 20
FLOOR = 1.0
TARGET_FPR = 0.05


def precompute_channels(scored):
    """For each site, precompute the per-channel aggregate deviation `dev`.

    Uses RC.score_site with h huge (no alarms yet) so only the dev signal is
    produced; each channel uses its OWN w_base/slack_k/min_base. The rolling
    baseline is built from CLEAN windows only (label-free QC gate), identical to
    the headline detector. Stored under scored[s]['dev_ms'] = list of dev arrays.
    """
    for s, sc in scored.items():
        devs = []
        for ch in CHANNELS:
            res = RC.score_site(
                sc["err"], sc["clean"],
                w_base=ch["w_base"], guard=GUARD, min_base=MIN_BASE,
                floor=FLOOR, slack_k=ch["slack_k"], h=1e9,
            )
            devs.append(res["dev"])
        sc["dev_ms"] = devs


def alarm_fn(scored, sites, thr):
    """Union CUSUM alarms across channels at shared alarm level thr.

    For each channel run RC.cusum_alarms on that channel's dev with its OWN
    slack_k and the SHARED h=thr, then OR across channels. Union only adds
    alarms.
    """
    out = {}
    for s in sites:
        devs = scored[s]["dev_ms"]
        al = np.zeros(len(devs[0]), bool)
        for ch, dev in zip(CHANNELS, devs):
            al |= RC.cusum_alarms(dev, ch["slack_k"], thr)
        out[s] = al
    return out


def build_grid(scored, controls):
    """Ascending shared-threshold grid from pooled control cal-clean dev.

    Pool the per-channel cal-clean dev across ALL channels and controls (the
    shared threshold is common to all channels), then take an ascending grid of
    quantiles. Also return a fallback shared quantile q.
    """
    pool = []
    for s in controls:
        for dev in scored[s]["dev_ms"]:
            v = dev[scored[s]["cal_clean"]]
            pool.append(v[np.isfinite(v)])
    pool = np.concatenate(pool) if pool else np.array([1.0])
    lo = float(min(ch["slack_k"] for ch in CHANNELS))
    hi = float(np.percentile(pool, 99.97)) if pool.size else 10.0
    grid = np.unique(np.round(np.linspace(lo, max(hi, lo + 1.0), 400), 4))
    q_fallback = float(np.percentile(pool, 99.0)) if pool.size else hi
    return grid, q_fallback


def med(x):
    return round(float(st.median(x)), 4) if x else None


def evaluate(split_name, split_fn):
    scored, controls, episodes = E.build_merged_scored(split_fn)
    precompute_channels(scored)
    n_ep = len(episodes)

    grid, q_fallback = build_grid(scored, controls)
    thr, ctrl_fpr = M.calibrate_to_heldout(scored, controls, TARGET_FPR, grid, alarm_fn)

    # fallback: if calibration failed to reach <= target (fpr > target), use
    # shared quantile q as thr (still a label-free control-only choice).
    used_fallback = False
    if ctrl_fpr > TARGET_FPR + 1e-9:
        thr = q_fallback
        al_ctrl = alarm_fn(scored, controls, thr)
        ctrl_fpr = M.pooled_fpr(al_ctrl, "hold_clean", controls, scored)
        used_fallback = True

    al = alarm_fn(scored, episodes, thr)
    per_ep, n_detected, leads = M.episode_leads(scored, episodes, al)
    detected = sorted(s for s in episodes if per_ep[s]["detected_pre_onset"])
    leads_by_site = {s: round(per_ep[s]["lead_days"], 4) for s in detected}
    median_lead = med(leads)

    return {
        "split": split_name,
        "thr": round(float(thr), 6),
        "used_fallback_quantile": used_fallback,
        "control_heldout_fpr": round(float(ctrl_fpr), 6),
        "n_episode_sites": n_ep,
        "n_control_sites": len(controls),
        "episodes_detected": int(n_detected),
        "median_lead_days": median_lead,
        "detected_sites": detected,
        "leads_by_site": leads_by_site,
        "is_pareto_win": bool(n_detected >= 11 and (median_lead or 0) >= 36.1
                              and ctrl_fpr <= TARGET_FPR + 1e-9),
    }


def main():
    channels_desc = [{"w_base": c["w_base"], "slack_k": c["slack_k"],
                      "guard": GUARD, "min_base": MIN_BASE, "floor": FLOOR}
                     for c in CHANNELS]

    primary = evaluate("interleave", M.cal_hold_split)
    temporal = evaluate("temporal", cal_hold_split_temporal)

    result = {
        "candidate_key": "multiscale-slack-union",
        "name": "Multi-scale slack union CUSUM",
        "idea": "OR CUSUM alarms across a-priori (w_base,slack_k) channels; union "
                "first-alarm = min over channels. Fast catches abrupt shifts, slow "
                "catches drift. One shared threshold on union control held-out FPR.",
        "channels": channels_desc,
        "shared": {"guard": GUARD, "min_base": MIN_BASE, "floor": FLOOR,
                   "target_control_heldout_fpr": TARGET_FPR},
        "hyperparameters_fixed_a_priori": True,
        "primary_interleave_split": primary,
        "robustness_temporal_split": temporal,
        "headline_baseline_to_beat": {"episodes_detected": 11, "median_lead_days": 36.1,
                                      "at_matched_control_heldout_fpr": 0.05},
    }

    out = OUTDIR / "multiscale-slack-union.json"
    out.write_text(json.dumps(result, indent=2))

    for tag, r in [("PRIMARY (interleave)", primary), ("ROBUSTNESS (temporal)", temporal)]:
        print(f"=== {tag} ===")
        print(f"  thr={r['thr']}  ctrl-heldout-FPR={r['control_heldout_fpr']:.4f}  "
              f"fallback={r['used_fallback_quantile']}")
        print(f"  episodes_detected={r['episodes_detected']}/{r['n_episode_sites']}  "
              f"median_lead={r['median_lead_days']} d")
        print(f"  detected={r['detected_sites']}")
        print(f"  leads={r['leads_by_site']}")
        print(f"  is_pareto_win={r['is_pareto_win']}")
    print(f"WROTE {out}")


if __name__ == "__main__":
    main()
