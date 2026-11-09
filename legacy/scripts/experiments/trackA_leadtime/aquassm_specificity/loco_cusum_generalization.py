#!/usr/bin/env python
"""Leave-one-control-out stability of the CUSUM headline operating point.

The expanded matched-FPR headline (expanded_matched_fpr.json) already transfers a
SINGLE global CUSUM threshold h, calibrated purely on the pooled CONTROL sites, to
the 15 fully-unseen EPISODE sites (episodes never touch calibration) — so 11/15 at
5% FPR is already a cross-site-transferred operating point. What it does not yet
show is whether that h is STABLE: does dropping any single control move the
operating point or change which episodes are caught?

This is the CUSUM analog of the baseline leave-one-site-out proof
(08_loso_generalization.py, which found the baseline tau is 2.79 +/- 0.02 sigma and
detection is unchanged under transfer). Here we leave out each control site in
turn, re-calibrate h on the OTHER controls to the 5% pooled held-out FPR target,
and record:

  * the calibrated h (its spread across the 17 leave-one-out fits);
  * the number/identity of episodes detected pre-onset under that transferred h;
  * the LEFT-OUT control's own held-out FPR under an h it did not help set
    (true cross-site specificity).

Runs on CPU from the shared residual caches. Label-free; identical guardrails.
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cusum_vs_baseline_matchedFPR as M  # noqa: E402
import expanded_matched_fpr as E  # noqa: E402

OUTDIR = E.OUTDIR
TARGET = 0.05


def c_grid_from(scored, controls):
    """CUSUM candidate-h grid from the given controls' cal-clean deviations."""
    c = np.concatenate([scored[s]["dev"][scored[s]["cal_clean"]] for s in controls])
    c = c[np.isfinite(c)]
    return np.unique(np.round(np.linspace(M.SLACK_K, np.percentile(c, 99.97), 300), 4))


def pooled_control_fpr(scored, controls, h):
    """Pooled held-out clean FPR over the given controls under threshold h."""
    al = M.cusum_alarms(scored, controls, h)
    return M.pooled_fpr(al, "hold_clean", controls, scored)


def leftout_fpr(scored, site, h):
    al = M.cusum_alarms(scored, [site], h)
    return M.pooled_fpr(al, "hold_clean", [site], scored)


def episodes_detected(scored, episodes, h):
    al = M.cusum_alarms(scored, episodes, h)
    ep, n, leads = M.episode_leads(scored, episodes, al)
    det = {s for s in episodes if ep[s]["detected_pre_onset"]}
    return det, leads


def main():
    scored, controls, episodes = E.build_merged_scored(M.cal_hold_split)
    controls = sorted(controls)
    episodes = sorted(episodes)
    n_ep = len(episodes)

    # --- full-fleet reference (all controls -> the published 11/15 headline) ---
    full_grid = c_grid_from(scored, controls)
    h_full, fpr_full = M.calibrate_to_heldout(scored, controls, TARGET, full_grid, M.cusum_alarms)
    det_full, leads_full = episodes_detected(scored, episodes, h_full)

    # --- leave-one-control-out ---
    loco = []
    for c in controls:
        others = [s for s in controls if s != c]
        grid = c_grid_from(scored, others)
        h, fpr_train = M.calibrate_to_heldout(scored, others, TARGET, grid, M.cusum_alarms)
        det, leads = episodes_detected(scored, episodes, h)
        loco.append({
            "left_out_control": c,
            "h": round(float(h), 4),
            "train_pooled_control_heldout_fpr": round(float(fpr_train), 4),
            "leftout_control_heldout_fpr": round(float(leftout_fpr(scored, c, h)), 4),
            "episodes_detected": len(det),
            "median_lead_days": round(float(st.median(leads)), 2) if leads else None,
            "detected_sites": sorted(det),
            "same_set_as_full": sorted(det) == sorted(det_full),
        })

    hs = [r["h"] for r in loco]
    ndets = [r["episodes_detected"] for r in loco]
    leftout_fprs = [r["leftout_control_heldout_fpr"] for r in loco]
    # union / intersection of detected episode sets across all leave-one-out fits
    det_sets = [set(r["detected_sites"]) for r in loco]
    inter = set.intersection(*det_sets) if det_sets else set()
    union = set.union(*det_sets) if det_sets else set()

    result = {
        "purpose": "Leave-one-control-out stability of the CUSUM headline operating "
                   "point h. h is re-calibrated on the OTHER controls to 5% pooled "
                   "held-out FPR and transferred to all 15 unseen episode sites + the "
                   "left-out control. Tests that the transferred operating point is "
                   "universal (not dependent on any single control).",
        "target_control_fpr": TARGET,
        "n_controls": len(controls), "n_episodes": n_ep,
        "full_fleet_reference": {
            "h": round(float(h_full), 4),
            "pooled_control_heldout_fpr": round(float(fpr_full), 4),
            "episodes_detected": len(det_full),
            "median_lead_days": round(float(st.median(leads_full)), 2) if leads_full else None,
            "detected_sites": sorted(det_full),
        },
        "loco_summary": {
            "h_mean": round(float(np.mean(hs)), 3),
            "h_sd": round(float(np.std(hs)), 3),
            "h_range": [float(min(hs)), float(max(hs))],
            "h_cv_pct": round(100 * float(np.std(hs)) / float(np.mean(hs)), 2),
            "episodes_detected_min": int(min(ndets)),
            "episodes_detected_max": int(max(ndets)),
            "all_fits_same_episode_set_as_full": all(r["same_set_as_full"] for r in loco),
            "episode_set_intersection": sorted(inter),
            "episode_set_union": sorted(union),
            "n_always_detected": len(inter),
            "leftout_control_fpr_mean": round(float(np.mean(leftout_fprs)), 4),
            "leftout_control_fpr_median": round(float(np.median(leftout_fprs)), 4),
            "leftout_control_fpr_max": round(float(max(leftout_fprs)), 4),
            "n_leftout_controls_within_5pct": int(sum(f <= TARGET + 1e-9 for f in leftout_fprs)),
        },
        "loco": loco,
    }

    out = OUTDIR / "loco_cusum_generalization.json"
    out.write_text(json.dumps(result, indent=2))

    s = result["loco_summary"]
    print(f"=== CUSUM leave-one-control-out ({len(controls)} controls, {n_ep} episodes) ===")
    print(f"FULL fleet: h={h_full:.2f}  ctrl-heldout-FPR={fpr_full:.3f}  "
          f"episodes={len(det_full)}/{n_ep}  lead={result['full_fleet_reference']['median_lead_days']}d")
    print(f"LOCO h:  mean={s['h_mean']}  sd={s['h_sd']}  range={s['h_range']}  CV={s['h_cv_pct']}%")
    print(f"LOCO episodes detected:  {s['episodes_detected_min']}..{s['episodes_detected_max']}  "
          f"(same set as full every time: {s['all_fits_same_episode_set_as_full']})")
    print(f"Always-detected episodes (intersection of all fits): {s['n_always_detected']} -> {s['episode_set_intersection']}")
    print(f"Left-out control held-out FPR: mean={s['leftout_control_fpr_mean']}  "
          f"median={s['leftout_control_fpr_median']}  max={s['leftout_control_fpr_max']}  "
          f"({s['n_leftout_controls_within_5pct']}/{len(controls)} within 5%)")
    print(f"WROTE {out}")


if __name__ == "__main__":
    main()
