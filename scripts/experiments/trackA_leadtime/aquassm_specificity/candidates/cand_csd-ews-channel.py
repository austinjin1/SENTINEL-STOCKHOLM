#!/usr/bin/env python
"""Candidate: csd-ews-channel — Critical-slowing-down EWS channel.

IDEA: Add rising trailing VARIANCE and lag-1 AUTOCORRELATION (AC1) of the causal
residuals as an ORTHOGONAL early-warning channel that can precede the mean shift.

Rising variance / AC1 ("critical slowing down") is a classic precursor of a regime
transition and is statistically independent of the mean-level shift that the existing
CUSUM channel accumulates. We fuse it additively.

SPEC (all hyperparameters fixed a priori; NEVER tuned on leads):
  * Per param (pH, DO, turbidity, conductivity) compute STRICTLY-CAUSAL trailing
    rolling variance and lag-1 AC over WIN=180 with GUARD=12 (the current window's
    own residual excluded via the guard gap; only PAST residuals used).
  * z-score EACH ews feature series (variance-series and AC1-series, 8 series total)
    with RC.rolling_causal_z(w_base=W_BASE, guard=GUARD, min_base=EWS_MIN_BASE=20).
  * dev_ews = sum of positive parts over the 8 ews-z (RC.aggregate_deviation-style).
  * dev_plus = dev  +  BETA * dev_ews    (BETA = 1.0), where dev is the AUDITED
    mean-shift CUSUM deviation from RC.score_site (identical to the headline detector).
  * alarm_fn = RC.cusum_alarms(dev_plus, SLACK_K, thr).
  * thr calibrated via M.calibrate_to_heldout at 5% pooled-control HELD-OUT FPR.

Guardrails A-E respected: label-free (onset/near gate EVALUATION only via
M.episode_leads); strictly causal; matched held-out control FPR; nothing tuned on
leads. The extra channel can inflate control FPR, so episodes-detected is reported.
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
OUTDIR = ROOT / "results" / "trackA_leadtime" / "aquassm_specificity"
OUTJSON = OUTDIR / "candidates" / "csd-ews-channel.json"

# ---- hyperparameters (FIXED A PRIORI; never tuned on leads) ----
EWS_WIN = 180        # trailing window for variance / AC1
EWS_GUARD = 12       # guard gap (matches CUSUM guard)
W_BASE = M.W_BASE    # 240; baseline window for z-scoring the ews feature series
GUARD = M.GUARD      # 12
EWS_MIN_BASE = 20    # per spec: Pair min_base 20 for the ews-z
FLOOR = 0.0          # ews-z positive-part floor (rising var/AC1 -> positive z)
BETA = 1.0           # ews channel weight
SLACK_K = M.SLACK_K  # 1.0
TARGET_FPR = 0.05


def _causal_trailing_var_ac1(err_col: np.ndarray, valid: np.ndarray,
                             win: int, guard: int):
    """Strictly-causal trailing rolling variance and lag-1 autocorrelation.

    For window index i, use residuals from the trailing valid window
    [i-guard-win, i-guard) (the current window and the `guard` most recent windows
    are EXCLUDED so an incipient shift does not contaminate its own EWS estimate,
    and NO future information is used). Returns (var[N], ac1[N]) with NaN where
    fewer than a minimal count of valid trailing residuals exist.
    """
    n = len(err_col)
    var = np.full(n, np.nan, np.float64)
    ac1 = np.full(n, np.nan, np.float64)
    src = np.where(valid & np.isfinite(err_col), err_col.astype(np.float64), np.nan)
    min_pts = max(8, win // 6)   # need enough points for a stable var/ac1
    for i in range(n):
        hi = i - guard
        lo = hi - win
        if lo < 0:
            lo = 0
        if hi <= lo:
            continue
        seg = src[lo:hi]
        seg = seg[np.isfinite(seg)]
        if seg.size < min_pts:
            continue
        m = seg.mean()
        d = seg - m
        v = float(np.mean(d * d))
        var[i] = v
        if v > 1e-12 and seg.size >= 2:
            # lag-1 autocorrelation on the trailing segment
            a = d[:-1]
            b = d[1:]
            denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
            if denom > 1e-12:
                ac1[i] = float(np.sum(a * b) / denom)
    return var, ac1


def compute_ews_dev(scored):
    """Add sc['dev_ews'] and sc['dev_plus'] to every site in `scored`.

    sc already has 'err'[N,4], 'clean'[N] (baseline-usable mask), and (from
    E.build_merged_scored -> M.cusum_dev) 'dev'[N] = audited mean-shift deviation.
    """
    for s, sc in scored.items():
        err = sc["err"]
        valid = sc["clean"]
        n, p = err.shape
        # collect ews-z for each param's variance-series and ac1-series
        ews_z_cols = []
        for j in range(p):
            var_ser, ac1_ser = _causal_trailing_var_ac1(
                err[:, j], valid, EWS_WIN, EWS_GUARD)
            # z-score each ews feature against its OWN strictly-causal trailing
            # baseline. valid mask = finite feature values (label-free).
            for feat in (var_ser, ac1_ser):
                fvalid = np.isfinite(feat)
                z = RC.rolling_causal_z(feat, fvalid, W_BASE, GUARD, EWS_MIN_BASE)
                ews_z_cols.append(z)
        ews_z = np.column_stack(ews_z_cols)               # [N, 8]
        dev_ews = RC.aggregate_deviation(ews_z, floor=FLOOR)  # sum positive parts
        # a window with no valid ews feature -> dev_ews NaN; treat as 0 contribution
        dev_ews_filled = np.where(np.isfinite(dev_ews), dev_ews, 0.0)
        dev = sc["dev"]
        dev_plus = np.where(np.isfinite(dev), dev + BETA * dev_ews_filled, np.nan)
        sc["dev_ews"] = dev_ews
        sc["dev_plus"] = dev_plus


def alarm_fn(scored, sites, thr):
    return {s: RC.cusum_alarms(scored[s]["dev_plus"], SLACK_K, thr) for s in sites}


def med(x):
    return round(float(st.median(x)), 4) if x else None


def run_split(split_name, split_fn):
    scored, controls, episodes = E.build_merged_scored(split_fn)
    compute_ews_dev(scored)
    n_ep = len(episodes)

    # ascending threshold grid from pooled control cal-clean dev_plus scores
    cal_scores = np.concatenate([scored[s]["dev_plus"][scored[s]["cal_clean"]]
                                 for s in controls])
    cal_scores = cal_scores[np.isfinite(cal_scores)]
    hi = np.percentile(cal_scores, 99.97) if cal_scores.size else (SLACK_K + 10)
    grid = np.unique(np.round(np.linspace(SLACK_K, max(hi, SLACK_K + 1), 300), 4))

    thr, ctrl_heldout_fpr = M.calibrate_to_heldout(
        scored, controls, TARGET_FPR, grid, alarm_fn)
    al = alarm_fn(scored, episodes, thr)
    per_ep, n_detected, leads = M.episode_leads(scored, episodes, al)
    median_lead = med(leads)
    detected = sorted(s for s in episodes if per_ep[s]["detected_pre_onset"])
    leads_by_site = {s: round(per_ep[s]["lead_days"], 2) for s in detected}

    is_pareto = bool(n_detected >= 11 and (median_lead or 0) >= 36.1
                     and ctrl_heldout_fpr <= 0.05 + 1e-9)

    return {
        "split": split_name,
        "n_episodes": n_ep, "n_controls": len(controls),
        "global_thr": round(float(thr), 4),
        "control_heldout_fpr": round(float(ctrl_heldout_fpr), 4),
        "episodes_detected": int(n_detected),
        "median_lead_days": median_lead,
        "detected_sites": detected,
        "leads_by_site": leads_by_site,
        "is_pareto_win": is_pareto,
    }


def main():
    prim = run_split("interleave", M.cal_hold_split)
    temp = run_split("temporal_single_run_fix", cal_hold_split_temporal)

    result = {
        "candidate": "csd-ews-channel",
        "idea": "Add rising trailing variance + lag-1 autocorrelation of causal "
                "residuals as an orthogonal EWS channel fused additively with the "
                "audited mean-shift CUSUM deviation (dev_plus = dev + BETA*dev_ews).",
        "hyperparameters": {
            "EWS_WIN": EWS_WIN, "EWS_GUARD": EWS_GUARD, "W_BASE": W_BASE,
            "GUARD": GUARD, "EWS_MIN_BASE": EWS_MIN_BASE, "FLOOR": FLOOR,
            "BETA": BETA, "SLACK_K": SLACK_K, "TARGET_FPR": TARGET_FPR,
            "note": "all fixed a priori; none tuned on leads",
        },
        "headline_5pct": {"interleave": prim},
        "robustness_temporal": temp,
        "baseline_to_beat": {"episodes_detected": 11, "median_lead_days": 36.1,
                             "at_fpr": 0.05},
    }
    OUTJSON.parent.mkdir(parents=True, exist_ok=True)
    OUTJSON.write_text(json.dumps(result, indent=2))

    for tag, r in (("INTERLEAVE", prim), ("TEMPORAL", temp)):
        print(f"=== {tag} ({r['n_episodes']} ep / {r['n_controls']} ctrl) ===")
        print(f"  thr={r['global_thr']}  ctrl-heldout-FPR={r['control_heldout_fpr']}")
        print(f"  episodes_detected={r['episodes_detected']}/{r['n_episodes']}  "
              f"median_lead={r['median_lead_days']} d")
        print(f"  detected={r['detected_sites']}")
        print(f"  leads={r['leads_by_site']}")
        print(f"  is_pareto_win={r['is_pareto_win']}")
    print(f"WROTE {OUTJSON}")


if __name__ == "__main__":
    main()
