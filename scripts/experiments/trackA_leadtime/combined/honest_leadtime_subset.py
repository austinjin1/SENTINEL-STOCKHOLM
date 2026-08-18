#!/usr/bin/env python
"""SENTINEL Track A -- Phase 1: the FINAL honest lead-time question.

Two prior rigorous tests closed the ABSOLUTE lead: (cycle 8) the combined
median lead tracks the arbitrary lookback cap almost perfectly (slope~1,
corr~0.997) and only ~1/4-1/3 of episodes' dev_plus BUILDS toward onset; (cycle
9) versus 7 real events' OFFICIAL detection dates, all null (chronic sites /
window-sensitive artifacts).

THE ONE REMAINING HONEST QUESTION addressed here: on the A-PRIORI SUBSET of the
32 benchmark episodes that BOTH

  (i)  show a GENUINE BUILDING PRECURSOR over [first-alarm, onset): dev_plus
       late-third median > early-third median AND Theil-Sen slope > 0 (a
       connected incipient buildup, not chronic/flat/decaying); AND

  (ii) have a genuinely QUIET WITHIN-SITE baseline: calibrate the dev_plus
       CUSUM threshold on THAT episode-site's OWN clean (non-hypoxia-near)
       windows to a <=5% sustained false-alarm rate, and REQUIRE that the <=5%
       bound also holds on HELD-OUT clean windows of the SAME site (so the
       pre-onset alarm is a real quiet->alarm TRANSITION, not chronic firing),

is there a WINDOW-INSENSITIVE lead from that within-site quiet->alarm
transition to onset?

DECISIVE GATE = WINDOW-INSENSITIVITY. For each episode that meets (i)+(ii) we
recompute its lead under a grid of reasonable clean-calibration windows / FPR
points (FPR in {3%,5%,7%} x {full-clean, first-half clean, second-half clean}
= 9 windows). The lead COUNTS only if it is STABLE: the episode still detects
pre-onset in >=3 of the windows AND the spread of the lead is bounded
(max/min <= 2, i.e. it does not vanish or halve across windows). This is
EXACTLY the test that refuted the cycle-9 Okeechobee/Delaware "leads".

any_defensible_lead = at least one episode passes ALL THREE (building precursor
+ quiet within-site baseline + window-insensitive lead).

All guardrails preserved: label-free; alarm strictly pre-onset AND
~hypoxia-near; threshold from CLEAN (non-near) data only, no peeking at onset;
a-priori criteria FIXED before looking at any lead; NEVER select to maximize
lead. Detector frozen (dev_plus = dev + BETA*dev_ews, headline params).
"""
from __future__ import annotations

import importlib.util as ilu
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SPEC_DIR = HERE.parent / "aquassm_specificity"
USGS_DIR = HERE.parent / "usgs"
sys.path.insert(0, str(SPEC_DIR))
sys.path.insert(0, str(SPEC_DIR / "candidates"))
sys.path.insert(0, str(USGS_DIR))

import cusum_vs_baseline_matchedFPR as M   # noqa: E402
import rolling_cusum as RC                 # noqa: E402

_ce = SPEC_DIR / "candidates" / "cand_csd-ews-channel.py"
_sp = ilu.spec_from_file_location("cand_csd_ews_channel", _ce)
CE = ilu.module_from_spec(_sp)
_sp.loader.exec_module(CE)

_ug = USGS_DIR / "detect_csd_ews_usgs.py"
_sp2 = ilu.spec_from_file_location("detect_csd_ews_usgs", _ug)
UG = ilu.module_from_spec(_sp2)
_sp2.loader.exec_module(UG)

ROOT = Path(__file__).resolve().parents[4]
OUTDIR = ROOT / "results" / "trackA_leadtime" / "combined"

DAY = 86400.0
SLACK_K = CE.SLACK_K

# ---- a-priori fixed constants (set BEFORE looking at any lead) ----
PRIMARY_FPR = 0.05                 # the headline within-site false-alarm bar
FPR_GRID = [0.03, 0.05, 0.07]      # window-insensitivity FPR points
CLEAN_SUBPERIODS = ["full", "first_half", "second_half"]   # clean sub-periods
# window-insensitivity gate: lead must survive in >=MIN_STABLE_WINDOWS of the
# FPR x sub-period grid AND the surviving leads' max/min must be <= MAX_SPREAD.
MIN_STABLE_WINDOWS = 3
MAX_SPREAD = 2.0
# quiet-baseline bar: held-out same-site clean FPR must be <= this (allow tiny
# slack for grid granularity) for the site to count as genuinely quiet.
QUIET_FPR_BAR = PRIMARY_FPR + 1e-9


def med(x):
    return round(float(st.median(x)), 4) if x else None


# --------------------------------------------------------------------------
# within-site CUSUM threshold calibrated on a chosen clean sub-period at FPR
# --------------------------------------------------------------------------
def clean_subperiod_mask(clean_idx, which):
    """Split a site's ordered clean-window indices into full/first/second half."""
    n = clean_idx.size
    if which == "full" or n < 4:
        return clean_idx
    h = n // 2
    return clean_idx[:h] if which == "first_half" else clean_idx[h:]


def calibrate_site_thr(dev, cal_mask, target):
    """Largest dev_plus CUSUM h with SUSTAINED FPR on cal_mask windows <= target.

    Uses the SAME sustained-CUSUM alarm as detection (RC.cusum_alarms). Returns
    (thr, cal_side_fpr). Purely from clean (non-near) calibration windows: no
    onset peeking.
    """
    vals = dev[cal_mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None, None
    hi = np.percentile(vals, 99.99)
    grid = np.unique(np.round(np.linspace(SLACK_K, max(hi, SLACK_K + 1), 400), 4))
    # ascending threshold => descending FPR; take smallest thr (most sensitive)
    # whose cal-side sustained FPR <= target.
    best_thr, best_fpr = float(grid[-1]), None
    denom = int(cal_mask.sum())
    for h in grid:
        al = RC.cusum_alarms(dev, SLACK_K, h)
        num = int(al[cal_mask].sum())
        fpr = (num / denom) if denom else 0.0
        if fpr <= target:
            best_thr, best_fpr = float(h), float(fpr)
            break
    if best_fpr is None:
        al = RC.cusum_alarms(dev, SLACK_K, best_thr)
        best_fpr = (int(al[cal_mask].sum()) / denom) if denom else 0.0
    return best_thr, best_fpr


def heldout_site_fpr(dev, hold_mask, thr):
    denom = int(hold_mask.sum())
    if denom == 0:
        return None
    al = RC.cusum_alarms(dev, SLACK_K, thr)
    return float(int(al[hold_mask].sum()) / denom)


def pre_onset_first_alarm(sc, thr):
    """First alarm ts in the strictly-pre-onset, covered, ~near window."""
    dev = sc["dev_plus"]
    et = sc["end_ts"]
    onset = sc["onset_ts"]
    keep = (et < onset) & sc["cov_ok"] & (~sc["near"])
    al = RC.cusum_alarms(dev, SLACK_K, thr)
    return M.first_time(al, et, keep)


def building_precursor(sc, first_alarm_ts):
    """Criterion (i): dev_plus BUILDS toward onset over [first-alarm, onset).

    late-third median > early-third median AND Theil-Sen slope > 0. Identical
    logic to the cycle-8 builds_toward_onset flag.
    """
    onset = sc["onset_ts"]
    et = sc["end_ts"]
    dev = sc["dev_plus"]
    win = (et >= first_alarm_ts) & (et < onset) & sc["cov_ok"] & (~sc["near"])
    idx = np.where(win)[0]
    idx = idx[np.isfinite(dev[idx])]
    seg = dev[idx]
    n = seg.size
    if n == 0:
        return {"n_windows": 0, "buildup_ratio": None,
                "theilsen_slope_per_day": None, "builds_toward_onset": False}
    third = max(1, n // 3)
    med_early = float(np.median(seg[:third]))
    med_late = float(np.median(seg[-third:]))
    ratio = (med_late / med_early) if med_early > 1e-9 else None
    tt = (et[idx] - first_alarm_ts) / DAY
    slope = None
    if n >= 5 and np.ptp(tt) > 0:
        m = min(n, 400)
        sel = np.linspace(0, n - 1, m).astype(int)
        ts, ds = tt[sel], seg[sel]
        sl = []
        for i in range(m):
            dj = ts[i + 1:] - ts[i]
            ok = dj != 0
            if ok.any():
                sl.append((ds[i + 1:][ok] - ds[i]) / dj[ok])
        if sl:
            slope = float(np.median(np.concatenate(sl)))
    builds = bool(ratio is not None and ratio > 1.0 and slope is not None and slope > 0)
    return {"n_windows": int(n),
            "buildup_ratio": (round(ratio, 4) if ratio is not None else None),
            "theilsen_slope_per_day": (round(slope, 6) if slope is not None else None),
            "builds_toward_onset": builds}


def analyze_episode(sc):
    """Full a-priori analysis for one episode site."""
    dev = sc["dev_plus"]
    clean = sc["clean"]
    clean_idx = np.where(clean & np.isfinite(dev))[0]

    # ---- (ii) QUIET within-site baseline: primary 5% cal on FULL clean,
    #           tested on the SITE's OWN held-out clean windows. ----
    cal_mask = sc["cal_clean"] & np.isfinite(dev)
    hold_mask = sc["hold_clean"] & np.isfinite(dev)
    thr5, cal5_fpr = calibrate_site_thr(dev, cal_mask, PRIMARY_FPR)
    held5 = heldout_site_fpr(dev, hold_mask, thr5) if thr5 is not None else None
    # site is genuinely quiet iff the <=5% bound survives on held-out same-site clean
    quiet = bool(thr5 is not None and held5 is not None and held5 <= QUIET_FPR_BAR)

    # ---- first alarm + (i) building precursor at the primary threshold ----
    fa5 = pre_onset_first_alarm(sc, thr5) if thr5 is not None else None
    lead5 = (sc["onset_ts"] - fa5) / DAY if fa5 is not None else None
    bp = building_precursor(sc, fa5) if fa5 is not None else \
        {"n_windows": 0, "buildup_ratio": None, "theilsen_slope_per_day": None,
         "builds_toward_onset": False}
    building = bool(bp["builds_toward_onset"])

    # ---- DECISIVE window-insensitivity grid (FPR x clean sub-period) ----
    grid_leads = {}
    for sub in CLEAN_SUBPERIODS:
        sub_idx = clean_subperiod_mask(clean_idx, sub)
        sub_mask = np.zeros_like(clean, bool)
        sub_mask[sub_idx] = True
        for fpr in FPR_GRID:
            thr, cf = calibrate_site_thr(dev, sub_mask, fpr)
            if thr is None:
                grid_leads[f"{sub}@{int(fpr*100)}"] = None
                continue
            fa = pre_onset_first_alarm(sc, thr)
            ld = (sc["onset_ts"] - fa) / DAY if fa is not None else None
            grid_leads[f"{sub}@{int(fpr*100)}"] = (round(ld, 3) if ld is not None else None)
    detected_leads = [v for v in grid_leads.values() if v is not None and v > 0]
    n_detected_windows = len(detected_leads)
    if detected_leads:
        lo, hi = min(detected_leads), max(detected_leads)
        spread = (hi / lo) if lo > 1e-9 else float("inf")
    else:
        lo = hi = spread = None
    window_insensitive = bool(n_detected_windows >= MIN_STABLE_WINDOWS
                              and spread is not None and spread <= MAX_SPREAD)

    return {
        "onset_utc": sc.get("onset_utc"),
        # (ii) quiet baseline
        "primary_thr_5pct": (round(thr5, 4) if thr5 is not None else None),
        "cal_side_fpr_5pct": (round(cal5_fpr, 4) if cal5_fpr is not None else None),
        "heldout_samesite_fpr_5pct": (round(held5, 4) if held5 is not None else None),
        "quiet_within_site_baseline": quiet,
        "n_clean": int(clean_idx.size),
        "n_cal_clean": int(cal_mask.sum()),
        "n_hold_clean": int(hold_mask.sum()),
        # first alarm / lead at primary
        "first_alarm_pre_onset": fa5 is not None,
        "primary_lead_days": (round(lead5, 3) if lead5 is not None else None),
        # (i) building precursor
        "building_precursor": building,
        "buildup_ratio": bp["buildup_ratio"],
        "theilsen_slope_per_day": bp["theilsen_slope_per_day"],
        "n_precursor_windows": bp["n_windows"],
        # decisive window-insensitivity
        "window_grid_leads": grid_leads,
        "n_windows_detected": n_detected_windows,
        "lead_min_days": (round(lo, 3) if lo is not None else None),
        "lead_max_days": (round(hi, 3) if hi is not None else None),
        "lead_spread_maxmin": (round(spread, 3) if spread not in (None, float("inf")) else None),
        "window_insensitive_lead": window_insensitive,
        # combined gates
        "meets_i_building": building,
        "meets_ii_quiet": quiet,
        "meets_both": bool(building and quiet),
        "qualifies_defensible": bool(building and quiet and window_insensitive),
    }


def main():
    n_scored, n_ctrl, n_ep = UG.build_neon_scored(M.cal_hold_split)
    u_scored, u_ctrl, u_ep = UG.build_usgs_scored(M.cal_hold_split)
    print(f"NEON: {len(n_ep)} ep | USGS: {len(u_ep)} ep", flush=True)

    per = {}
    for tag, sc_, ep in (("NEON", n_scored, n_ep), ("USGS", u_scored, u_ep)):
        for s in ep:
            per[f"{tag}:{s}"] = analyze_episode(sc_[s])
            r = per[f"{tag}:{s}"]
            print(f"  {tag}:{s:>16}  build={int(r['meets_i_building'])} "
                  f"quiet={int(r['meets_ii_quiet'])} both={int(r['meets_both'])} "
                  f"win_insens={int(r['window_insensitive_lead'])} "
                  f"lead5={r['primary_lead_days']} "
                  f"grid[det/9]={r['n_windows_detected']} spread={r['lead_spread_maxmin']} "
                  f"QUALIFIES={int(r['qualifies_defensible'])}", flush=True)

    n_total = len(per)
    n_build = sum(1 for v in per.values() if v["meets_i_building"])
    n_quiet = sum(1 for v in per.values() if v["meets_ii_quiet"])
    n_both = sum(1 for v in per.values() if v["meets_both"])
    qual = {k: v for k, v in per.items() if v["qualifies_defensible"]}
    n_qual = len(qual)
    stable_leads = [v["primary_lead_days"] for v in qual.values()
                    if v["primary_lead_days"] is not None]
    stable_median = med(stable_leads)
    any_defensible = bool(n_qual >= 1)

    qualifying_episodes = {k: v["primary_lead_days"] for k, v in qual.items()}
    window_insensitive_leads = {
        k: {"detected_windows": v["n_windows_detected"],
            "lead_min": v["lead_min_days"], "lead_max": v["lead_max_days"],
            "spread_maxmin": v["lead_spread_maxmin"],
            "window_insensitive": v["window_insensitive_lead"]}
        for k, v in per.items() if v["meets_both"]
    }

    result = {
        "purpose": "FINAL honest lead-time question: on the a-priori subset of the 32 "
                   "NEON+USGS episodes that BOTH (i) show a genuine BUILDING precursor "
                   "AND (ii) have a genuinely QUIET within-site baseline, is there a "
                   "WINDOW-INSENSITIVE lead from a real quiet->alarm transition to onset?",
        "detector": {
            "channel": "dev_plus = dev + BETA*dev_ews (frozen CSD-EWS)",
            "hyperparameters": {"W_BASE": CE.W_BASE, "GUARD": CE.GUARD,
                                "MIN_BASE": M.MIN_BASE, "SLACK_K": SLACK_K,
                                "EWS_WIN": CE.EWS_WIN, "EWS_MIN_BASE": CE.EWS_MIN_BASE,
                                "BETA": CE.BETA},
            "split": "interleave (M.cal_hold_split)",
        },
        "a_priori_criteria": {
            "i_building_precursor": "over [first-alarm, onset): dev_plus late-third median "
                                    "> early-third median AND Theil-Sen slope > 0.",
            "ii_quiet_within_site_baseline": (
                f"within-site CUSUM thr calibrated on that site's OWN cal-clean "
                f"(non-near) windows to <= {PRIMARY_FPR:.0%} sustained FPR, AND the "
                f"<= {PRIMARY_FPR:.0%} bound also holds on HELD-OUT clean windows of the "
                "SAME site (genuine quiet->alarm transition, not chronic firing)."),
            "decisive_window_insensitivity": (
                f"recompute lead over FPR x clean sub-period grid "
                f"({FPR_GRID} x {CLEAN_SUBPERIODS} = 9 windows); the lead counts only if "
                f"detected pre-onset in >= {MIN_STABLE_WINDOWS} windows AND max/min lead "
                f"spread <= {MAX_SPREAD} (does not vanish or halve)."),
            "fixed_before_looking_at_leads": True,
            "never_selected_to_maximize_lead": True,
        },
        "counts": {
            "n_episodes_total": n_total,
            "n_building_precursor_i": n_build,
            "n_quiet_baseline_ii": n_quiet,
            "n_both_i_and_ii": n_both,
            "n_qualifying_defensible_all_three": n_qual,
        },
        "stable_median_lead_days": stable_median,
        "any_defensible_lead": any_defensible,
        "qualifying_episodes_lead_days": qualifying_episodes,
        "window_insensitivity_by_both_ii_i_episode": window_insensitive_leads,
        "per_episode": per,
        "verdict": (
            f"{n_qual}/{n_total} episodes yield a DEFENSIBLE lead (building precursor + "
            f"quiet within-site baseline + window-insensitive)."
            + (f" Stable median lead = {stable_median} d over the surviving subset "
               f"{sorted(qual)}." if any_defensible else
               " NONE survive all three gates -> lead-time is definitively closed: there "
               "is no honest, window-insensitive, quiet->alarm lead in the benchmark.")),
        "guardrails": {
            "label_free": True,
            "alarm_strictly_pre_onset_and_not_hypoxia_near": True,
            "threshold_from_clean_non_near_data_only_no_onset_peeking": True,
            "within_site_calibration": True,
            "frozen_detector": True,
            "a_priori_subset_no_cherry_picking": True,
        },
    }

    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / "honest_leadtime_subset.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"\n=== SUMMARY ===", flush=True)
    print(f"total={n_total} building(i)={n_build} quiet(ii)={n_quiet} "
          f"both={n_both} qualifying(all3)={n_qual}", flush=True)
    print(f"any_defensible_lead={any_defensible} stable_median_lead={stable_median}", flush=True)
    print(f"WROTE {out}", flush=True)


if __name__ == "__main__":
    main()
