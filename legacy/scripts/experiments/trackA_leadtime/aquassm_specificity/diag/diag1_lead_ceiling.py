#!/usr/bin/env python
"""Diagnostic 1 -- the lead CEILING per episode.

For each of the 15 episode sites, quantify:
  * runway_ceiling_days: the earliest a precursor alarm could physically fire,
    = (onset_ts - end_ts of the FIRST covered, non-hypoxia-near, pre-onset window)/86400.
  * current CUSUM lead at the headline operating point (global h calibrated on the
    17 controls to pooled HELD-OUT 5% FPR, applied to the episode).
  * pre-onset dev[] trajectory: day-before-onset at which dev first exceeds
    {1,2,5,10} x slack_k, and where dev first crosses slack_k and h.

Then classify each detected episode as lead-LIMITED-BY-RUNWAY (current lead ~ ceiling)
vs lead-LIMITED-BY-DETECTOR (dev rose earlier than the alarm), and rank the detected
episodes by exploitable headroom = ceiling - current_lead.

Label-free. onset/near gate EVALUATION only. No hyperparameter tuned on leads.
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import cusum_vs_baseline_matchedFPR as M  # noqa: E402
import rolling_cusum as RC  # noqa: E402
import expanded_matched_fpr as E  # noqa: E402

DAY = 86400.0
TARGET = 0.05
LOOKBACK_DAYS = M.LOOKBACK_DAYS  # 45
SLACK_K = M.SLACK_K              # 1.0


def main():
    scored, controls, episodes = E.build_merged_scored(M.cal_hold_split)

    # --- global h at pooled control HELD-OUT 5% FPR (headline operating point) ---
    c_dev = np.concatenate([scored[s]["dev"][scored[s]["cal_clean"]] for s in controls])
    c_dev = c_dev[np.isfinite(c_dev)]
    c_grid = np.unique(np.round(np.linspace(SLACK_K, np.percentile(c_dev, 99.97), 400), 4))
    h, c_fpr = M.calibrate_to_heldout(scored, controls, TARGET, c_grid, M.cusum_alarms)

    # apply that single global h to every episode
    c_alarm_ep = M.cusum_alarms(scored, episodes, h)
    c_ep, c_nval, c_leads = M.episode_leads(scored, episodes, c_alarm_ep)

    rows = {}
    for s in sorted(episodes):
        sc = scored[s]
        onset = sc["onset_ts"]
        et = sc["end_ts"].astype(np.float64)
        near = sc["near"]
        cov_ok = sc["cov_ok"]
        dev = sc["dev"]

        # precursor-eligible windows: strictly pre-onset, covered, not hypoxia-near
        # (NO lookback clip here for the CEILING -- ceiling is physical earliest)
        pre_cov = (et < onset) & cov_ok & (~near)
        n_pre_cov = int(pre_cov.sum())

        if n_pre_cov > 0:
            idx_pre = np.where(pre_cov)[0]
            # earliest covered pre-onset window end
            first_end = et[idx_pre].min()
            runway_ceiling = (onset - first_end) / DAY
        else:
            runway_ceiling = None

        # PROTOCOL-CAPPED ceiling: the current evaluation harness (episode_leads)
        # only counts alarms whose window end is within LOOKBACK_DAYS of onset, so
        # NO lead > LOOKBACK_DAYS can ever be scored. The achievable-lead ceiling
        # under the current protocol is min(physical runway, LOOKBACK_DAYS), further
        # limited to the earliest covered non-near window that ALSO lies in the
        # lookback window.
        in_lb = pre_cov & (et >= onset - LOOKBACK_DAYS * DAY)
        if in_lb.any():
            first_end_lb = et[np.where(in_lb)[0]].min()
            protocol_ceiling = (onset - first_end_lb) / DAY
        else:
            protocol_ceiling = None

        # dev trajectory within the SAME precursor-eligible windows.
        # For dev to be meaningful the window must have a finite dev.
        finite = np.isfinite(dev)
        traj_mask = pre_cov & finite
        idx_t = np.where(traj_mask)[0]
        # days-before-onset for each such window
        dbo = (onset - et) / DAY

        def first_cross_day(mult):
            """Earliest (largest days-before-onset) precursor window with dev > mult*slack_k."""
            hit = idx_t[dev[idx_t] > mult * SLACK_K]
            if hit.size == 0:
                return None
            return float(dbo[hit].max())

        dev_first_gt = {m: first_cross_day(m) for m in (1, 2, 5, 10)}

        # when does dev first exceed slack_k and h in covered pre-onset windows
        dev_first_gt_slack = dev_first_gt[1]
        hit_h = idx_t[dev[idx_t] > h]
        dev_first_gt_h_day = float(dbo[hit_h].max()) if hit_h.size else None

        # peak dev pre-onset + its day
        if idx_t.size:
            peak_i = idx_t[np.argmax(dev[idx_t])]
            peak_dev = float(dev[peak_i]); peak_day = float(dbo[peak_i])
            # median/max dev in precursor windows
            dev_pre_max = float(np.nanmax(dev[idx_t]))
        else:
            peak_dev = peak_day = dev_pre_max = None

        # slowly-building vs late-spike heuristic:
        # slope of dev over days-before-onset among precursor windows where dev>slack_k.
        rise = idx_t[dev[idx_t] > SLACK_K]
        if rise.size >= 3:
            x = dbo[rise]  # days before onset (decreasing toward 0)
            y = dev[rise]
            # correlation of dev with proximity-to-onset (small dbo). Positive slope
            # of dev vs (-dbo) => builds as onset nears.
            slope = float(np.polyfit(-x, y, 1)[0])
        else:
            slope = None

        # dev signal availability WITHIN the lookback window (the region the current
        # protocol actually scores). Earliest day-before-onset that dev>slack / dev>h.
        idx_lb = np.where(in_lb & finite)[0] if isinstance(in_lb, np.ndarray) else np.array([], int)
        if idx_lb.size:
            hit_s_lb = idx_lb[dev[idx_lb] > SLACK_K]
            hit_h_lb = idx_lb[dev[idx_lb] > h]
            dev_gt_slack_lb = float(dbo[hit_s_lb].max()) if hit_s_lb.size else None
            dev_gt_h_lb = float(dbo[hit_h_lb].max()) if hit_h_lb.size else None
        else:
            dev_gt_slack_lb = dev_gt_h_lb = None

        # current CUSUM lead
        ci = c_ep[s]
        cur_lead = ci["lead_days"]
        detected = ci["detected_pre_onset"]

        # exploitable headroom UNDER THE CURRENT PROTOCOL (lookback-capped): this is
        # the max extra lead a better detector could actually be CREDITED with today.
        if detected and protocol_ceiling is not None and cur_lead is not None:
            headroom = protocol_ceiling - cur_lead
        else:
            headroom = None
        # headroom if the LOOKBACK were removed (physical ceiling)
        if detected and runway_ceiling is not None and cur_lead is not None:
            headroom_physical = runway_ceiling - cur_lead
        else:
            headroom_physical = None

        # detector-limited?  dev crossed h earlier than the alarm actually fired.
        # (CUSUM accumulates, so alarm day <= dev-cross-h day in days-before-onset;
        #  a big gap means signal existed well before the accumulator tripped.)
        if detected and dev_first_gt_h_day is not None and cur_lead is not None:
            detector_gap = dev_first_gt_h_day - cur_lead
        else:
            detector_gap = None

        rows[s] = {
            "onset_utc": ci["onset_utc"],
            "detected": bool(detected),
            "current_cusum_lead_days": round(cur_lead, 2) if cur_lead is not None else None,
            "runway_ceiling_days": round(runway_ceiling, 2) if runway_ceiling is not None else None,
            "protocol_ceiling_days": round(protocol_ceiling, 2) if protocol_ceiling is not None else None,
            "n_pre_onset_covered_windows": n_pre_cov,
            "exploitable_headroom_days": round(headroom, 2) if headroom is not None else None,
            "exploitable_headroom_physical_days": round(headroom_physical, 2) if headroom_physical is not None else None,
            "dev_gt_slack_in_lookback_dbo": round(dev_gt_slack_lb, 2) if dev_gt_slack_lb is not None else None,
            "dev_gt_h_in_lookback_dbo": round(dev_gt_h_lb, 2) if dev_gt_h_lb is not None else None,
            "dev_first_exceeds_x_slack_days_before_onset": {
                str(m): (round(v, 2) if v is not None else None)
                for m, v in dev_first_gt.items()},
            "dev_first_gt_slack_days_before_onset": (
                round(dev_first_gt_slack, 2) if dev_first_gt_slack is not None else None),
            "dev_first_gt_h_days_before_onset": (
                round(dev_first_gt_h_day, 2) if dev_first_gt_h_day is not None else None),
            "dev_alarm_gap_days": round(detector_gap, 2) if detector_gap is not None else None,
            "peak_dev_pre_onset": round(peak_dev, 3) if peak_dev is not None else None,
            "peak_dev_days_before_onset": round(peak_day, 2) if peak_day is not None else None,
            "dev_rise_slope_per_day": round(slope, 4) if slope is not None else None,
        }

    # --- classification & ranking of the DETECTED episodes ---
    detected = [s for s in rows if rows[s]["detected"]]
    RUNWAY_TOL = 5.0  # days: current lead within TOL of ceiling => runway-limited
    classify = {}
    for s in detected:
        r = rows[s]
        hr = r["exploitable_headroom_days"]
        if hr is None:
            classify[s] = "unknown"
        elif hr <= RUNWAY_TOL:
            classify[s] = "runway_limited"
        else:
            classify[s] = "detector_limited"

    ranked = sorted(
        [s for s in detected if rows[s]["exploitable_headroom_days"] is not None],
        key=lambda s: rows[s]["exploitable_headroom_days"], reverse=True)

    # max achievable median lead if every detected episode fired at its ceiling
    ceil_leads = [rows[s]["runway_ceiling_days"] for s in detected
                  if rows[s]["runway_ceiling_days"] is not None]
    proto_leads = [rows[s]["protocol_ceiling_days"] for s in detected
                   if rows[s]["protocol_ceiling_days"] is not None]
    cur_leads = [rows[s]["current_cusum_lead_days"] for s in detected
                 if rows[s]["current_cusum_lead_days"] is not None]
    max_median_if_ceiling = round(float(st.median(ceil_leads)), 2) if ceil_leads else None
    max_median_if_protocol_ceiling = round(float(st.median(proto_leads)), 2) if proto_leads else None
    cur_median = round(float(st.median(cur_leads)), 2) if cur_leads else None

    summary = {
        "global_h": round(h, 4),
        "control_heldout_fpr": round(c_fpr, 4),
        "n_detected": len(detected),
        "detected_sites": sorted(detected),
        "lookback_days_cap": LOOKBACK_DAYS,
        "current_median_lead_days": cur_median,
        "max_median_lead_if_all_at_protocol_ceiling_days": max_median_if_protocol_ceiling,
        "max_median_lead_if_all_at_physical_ceiling_days": max_median_if_ceiling,
        "classification": classify,
        "ranked_by_headroom_desc": [
            {"site": s, "headroom_days": rows[s]["exploitable_headroom_days"],
             "headroom_physical_days": rows[s]["exploitable_headroom_physical_days"],
             "current_lead": rows[s]["current_cusum_lead_days"],
             "protocol_ceiling": rows[s]["protocol_ceiling_days"],
             "physical_ceiling": rows[s]["runway_ceiling_days"],
             "dev_gt_h_in_lookback_dbo": rows[s]["dev_gt_h_in_lookback_dbo"],
             "class": classify[s]}
            for s in ranked],
        "runway_limited": sorted([s for s in detected if classify.get(s) == "runway_limited"]),
        "detector_limited": sorted([s for s in detected if classify.get(s) == "detector_limited"]),
    }

    out = {"summary": summary, "per_episode": rows}
    outpath = HERE / "diag1_lead_ceiling.json"
    outpath.write_text(json.dumps(out, indent=2))

    # ---- readable print ----
    print(f"global h={h:.4f}  control held-out FPR={c_fpr:.4f}  detected={len(detected)}/15")
    print(f"LOOKBACK cap={LOOKBACK_DAYS}d  current median lead={cur_median}d")
    print(f"median lead if all at PROTOCOL ceiling={max_median_if_protocol_ceiling}d "
          f"| if lookback removed (physical)={max_median_if_ceiling}d")
    print()
    hdr = (f"{'site':<6}{'det':>4}{'lead':>7}{'protoC':>8}{'physC':>8}"
           f"{'head':>7}{'devH_lb':>9}  class")
    print(hdr); print("-" * len(hdr))
    def f(v, w=7):
        return f"{v:>{w}.1f}" if isinstance(v, (int, float)) else f"{'--':>{w}}"
    for s in sorted(rows):
        r = rows[s]
        cls = classify.get(s, "-" if not r["detected"] else "?")
        print(f"{s:<6}{('Y' if r['detected'] else 'n'):>4}"
              f"{f(r['current_cusum_lead_days'])}{f(r['protocol_ceiling_days'], 8)}"
              f"{f(r['runway_ceiling_days'], 8)}{f(r['exploitable_headroom_days'])}"
              f"{f(r['dev_gt_h_in_lookback_dbo'], 9)}  {cls}")
    print()
    print("RANK by exploitable headroom UNDER CURRENT PROTOCOL (detected):")
    for e in summary["ranked_by_headroom_desc"]:
        print(f"  {e['site']:<6} headroom={e['headroom_days']:>6.1f}d "
              f"(lead={e['current_lead']}d protoC={e['protocol_ceiling']}d "
              f"physC={e['physical_ceiling']}d devH_lb={e['dev_gt_h_in_lookback_dbo']}) {e['class']}")
    print(f"\nWROTE {outpath}")


if __name__ == "__main__":
    main()
