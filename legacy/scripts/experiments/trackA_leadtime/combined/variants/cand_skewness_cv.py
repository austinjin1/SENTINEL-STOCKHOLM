#!/usr/bin/env python
"""Augmented CSD-EWS variant  key=skewness_cv  name="Rising skewness + coefficient of variation".

Extends the VERIFIED critical-slowing-down EWS channel (rolling residual VARIANCE +
lag-1 AUTOCORRELATION) with TWO additional strictly-causal trailing indicators per
parameter:

  * SKEW: trailing standardized skewness (third standardized moment) of the residuals,
    skew = mean(d^3) / v^1.5  (d = seg - mean(seg), v = trailing variance already
    computed). SIGNED skew is used directly (positive skew = growing heavy upper tail
    preceding an upward residual excursion), one-sided via the aggregate positive-part
    floor=0 so only ABOVE-baseline skew contributes.
  * CV:   coefficient of variation, cv = sqrt(v) / max(|mean(seg)|, CV_EPS), with a
    guarded denominator (residuals are near-zero-mean reconstruction errors).

Both features are z-scored per site against their OWN strictly-causal trailing baseline
(RC.rolling_causal_z), so a control with intrinsically skewed-but-stationary residuals
sits at its own baseline z~0 and does not add FPR; only a per-site RISE contributes.

The new skew + cv z-series are STACKED ONTO variant-1 (var + ac1) -> 4 params x 4 feats
= 16 ews-z series. Aggregation (positive-part sum, floor=0), additive fusion
(dev_plus = dev + BETA*dev_ews), thresholding and matched-held-out-FPR calibration are
UNCHANGED from the headline CSD-EWS detector.

A-PRIORI hyperparameters (NEVER tuned on leads):
  SKEW_MINPTS = 24, CV_EPS = 1e-3.
  Everything else inherited from the headline: EWS_WIN=180, EWS_GUARD=12, W_BASE=240,
  GUARD=12, EWS_MIN_BASE=20, FLOOR=0.0, BETA=1.0, SLACK_K=1.0, TARGET_FPR=0.05.

Evaluated at matched <=5% held-out CONTROL FPR on BOTH networks (NEON self-cal, USGS
self-cal) PLUS leave-one-network-out (calibrate threshold on the OTHER network's
controls). Guardrails A-E respected: label-free (onset/near gate EVALUATION only via
M.episode_leads); strictly causal; matched held-out control FPR (episodes never
calibrate); fixed a-priori hyperparams; honest Pareto (report both count and lead).
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[5]
SPEC_DIR = ROOT / "scripts" / "experiments" / "trackA_leadtime" / "aquassm_specificity"
USGS_DIR = ROOT / "scripts" / "experiments" / "trackA_leadtime" / "usgs"
sys.path.insert(0, str(SPEC_DIR))

import cusum_vs_baseline_matchedFPR as M  # noqa: E402
import expanded_matched_fpr as E  # noqa: E402
import rolling_cusum as RC  # noqa: E402

# ---- output ----
OUTJSON = ROOT / "results" / "trackA_leadtime" / "combined" / "variants" / "skewness_cv.json"

# ---- USGS cache locations (mirror of detect_csd_ews_usgs.py) ----
USGS_OUT = ROOT / "results" / "trackA_leadtime" / "usgs"
USGS_CACHE = USGS_OUT / "residual_cache_usgs.npz"
USGS_META = json.load(open(USGS_OUT / "cache_meta_usgs.json"))

# ---- hyperparameters (FIXED A PRIORI; never tuned on leads) ----
EWS_WIN = 180        # trailing window for variance / AC1 / skew / cv  (inherited)
EWS_GUARD = 12       # guard gap (matches CUSUM guard)                 (inherited)
W_BASE = M.W_BASE    # 240; baseline window for z-scoring the ews feature series
GUARD = M.GUARD      # 12
EWS_MIN_BASE = 20    # min_base for the ews-z                          (inherited)
FLOOR = 0.0          # ews-z positive-part floor                       (inherited)
BETA = 1.0           # ews channel weight                             (inherited)
SLACK_K = M.SLACK_K  # 1.0
TARGET_FPR = 0.05

# NEW a-priori constants for this variant
SKEW_MINPTS = 24     # min trailing valid pts before a skew estimate is trusted
CV_EPS = 1e-3        # denominator floor for CV (near-zero-mean residuals)


def _causal_trailing_var_ac1_skew_cv(err_col: np.ndarray, valid: np.ndarray,
                                     win: int, guard: int):
    """Strictly-causal trailing variance, lag-1 AC, standardized skewness and CV.

    For window index i, use residuals from the trailing valid window
    [i-guard-win, i-guard) (the current window and the `guard` most recent windows
    are EXCLUDED so an incipient shift does not contaminate its own EWS estimate,
    and NO future information is used).

    Returns (var[N], ac1[N], skew[N], cv[N]) with NaN where fewer than the relevant
    minimal count of valid trailing residuals exist. var/ac1 use the SAME logic as
    the headline _causal_trailing_var_ac1; skew/cv reuse the same trailing segment.
    """
    n = len(err_col)
    var = np.full(n, np.nan, np.float64)
    ac1 = np.full(n, np.nan, np.float64)
    skew = np.full(n, np.nan, np.float64)
    cv = np.full(n, np.nan, np.float64)
    src = np.where(valid & np.isfinite(err_col), err_col.astype(np.float64), np.nan)
    min_pts = max(8, win // 6)   # var/ac1 need enough points (headline value)
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
            a = d[:-1]
            b = d[1:]
            denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
            if denom > 1e-12:
                ac1[i] = float(np.sum(a * b) / denom)
        # ---- NEW: standardized skewness (third standardized moment) ----
        # skew = mean(d^3) / v^1.5 ; NaN if v<=1e-12 or seg.size < SKEW_MINPTS
        if v > 1e-12 and seg.size >= SKEW_MINPTS:
            skew[i] = float(np.mean(d * d * d) / (v ** 1.5))
        # ---- NEW: coefficient of variation (guarded denominator) ----
        # cv = sqrt(v) / max(|mean(seg)|, CV_EPS)
        cv[i] = float(np.sqrt(v) / max(abs(m), CV_EPS))
    return var, ac1, skew, cv


def compute_ews_dev(scored):
    """Add sc['dev_ews'] and sc['dev_plus'] to every site in `scored`.

    ews-z columns per param: [var, ac1, skew, cv] (4 params x 4 feats = 16 series),
    stacked onto the headline variant-1 (var + ac1). Aggregation = sum of positive
    parts (floor=0). dev_plus = dev + BETA * dev_ews (BETA=1.0).
    """
    for s, sc in scored.items():
        err = sc["err"]
        valid = sc["clean"]
        n, p = err.shape
        ews_z_cols = []
        for j in range(p):
            var_ser, ac1_ser, skew_ser, cv_ser = _causal_trailing_var_ac1_skew_cv(
                err[:, j], valid, EWS_WIN, EWS_GUARD)
            # z-score EACH ews feature against its OWN strictly-causal trailing
            # baseline. valid mask = finite feature values (label-free).
            for feat in (var_ser, ac1_ser, skew_ser, cv_ser):
                fvalid = np.isfinite(feat)
                z = RC.rolling_causal_z(feat, fvalid, W_BASE, GUARD, EWS_MIN_BASE)
                ews_z_cols.append(z)
        ews_z = np.column_stack(ews_z_cols)                   # [N, 16]
        dev_ews = RC.aggregate_deviation(ews_z, floor=FLOOR)  # sum positive parts
        dev_ews_filled = np.where(np.isfinite(dev_ews), dev_ews, 0.0)
        dev = sc["dev"]
        dev_plus = np.where(np.isfinite(dev), dev + BETA * dev_ews_filled, np.nan)
        sc["dev_ews"] = dev_ews
        sc["dev_plus"] = dev_plus


def alarm_fn(scored, sites, thr):
    return {s: RC.cusum_alarms(scored[s]["dev_plus"], SLACK_K, thr) for s in sites}


def med(x):
    return round(float(st.median(x)), 4) if x else None


def calibrate_thr(scored, controls, target):
    """Largest dev_plus thr with pooled control HELD-OUT FPR <= target."""
    cal_scores = np.concatenate([scored[s]["dev_plus"][scored[s]["cal_clean"]]
                                 for s in controls])
    cal_scores = cal_scores[np.isfinite(cal_scores)]
    hi = np.percentile(cal_scores, 99.97) if cal_scores.size else (SLACK_K + 10)
    grid = np.unique(np.round(np.linspace(SLACK_K, max(hi, SLACK_K + 1), 300), 4))
    thr, fpr = M.calibrate_to_heldout(scored, controls, target, grid, alarm_fn)
    return float(thr), float(fpr)


def evaluate_at(scored, controls, episodes, thr):
    """Apply thr: pooled control HELD-OUT FPR + episode detection/leads."""
    al_ctrl = alarm_fn(scored, controls, thr)
    ctrl_fpr = M.pooled_fpr(al_ctrl, "hold_clean", controls, scored)
    al_ep = alarm_fn(scored, episodes, thr)
    per_ep, n_det, leads = M.episode_leads(scored, episodes, al_ep)
    detected = sorted(s for s in episodes if per_ep[s]["detected_pre_onset"])
    leads_by_site = {s: round(per_ep[s]["lead_days"], 2) for s in detected}
    return {
        "global_thr": round(float(thr), 4),
        "control_heldout_fpr": round(float(ctrl_fpr), 4),
        "episodes_detected": int(n_det),
        "n_episodes": len(episodes),
        "median_lead_days": med(leads),
        "detected_sites": detected,
        "leads_by_site": leads_by_site,
    }


# --------------------- network scored builders ---------------------
def build_neon():
    scored, controls, episodes = E.build_merged_scored(M.cal_hold_split)
    compute_ews_dev(scored)
    return scored, controls, episodes


def load_usgs_site(site):
    z = np.load(USGS_CACHE, allow_pickle=True)
    return {
        "end_ts": z[f"{site}__end_ts"],
        "cov": z[f"{site}__cov"],
        "near": z[f"{site}__near"],
        "err": z[f"{site}__err"].astype(np.float64),
    }


def build_usgs():
    """Mirror E.build_merged_scored on the USGS cache (from detect_csd_ews_usgs.py)."""
    scored = {}
    for site, m in USGS_META.items():
        d = load_usgs_site(site)
        cov_ok, clean = M.masks(d)
        cal, hold = M.cal_hold_split(clean)
        scored[site] = {
            "err": d["err"], "cov_ok": cov_ok, "clean": clean,
            "cal_clean": cal, "hold_clean": hold, "end_ts": d["end_ts"],
            "near": d["near"], "role": m["role"], "onset_ts": m["onset_ts"],
            "onset_utc": m["onset_utc"],
        }
    controls = [s for s in scored if scored[s]["role"] == "control"]
    episodes = [s for s in scored if scored[s]["role"] == "episode"
                and scored[s]["onset_ts"] is not None]
    M.cusum_dev(scored)
    compute_ews_dev(scored)
    return scored, controls, episodes


def main():
    # ---- build both networks (variant channel) ----
    n_scored, n_controls, n_episodes = build_neon()
    u_scored, u_controls, u_episodes = build_usgs()
    print(f"NEON: {len(n_episodes)} ep / {len(n_controls)} ctrl", flush=True)
    print(f"USGS: {len(u_episodes)} ep / {len(u_controls)} ctrl", flush=True)

    # ---- (a) NEON self-calibrated ----
    n_thr, n_calfpr = calibrate_thr(n_scored, n_controls, TARGET_FPR)
    neon_self = evaluate_at(n_scored, n_controls, n_episodes, n_thr)
    print(f"[NEON self] thr={n_thr:.4f} fpr={neon_self['control_heldout_fpr']} "
          f"det={neon_self['episodes_detected']}/{neon_self['n_episodes']} "
          f"lead={neon_self['median_lead_days']}", flush=True)

    # ---- (b) USGS self-calibrated ----
    u_thr, u_calfpr = calibrate_thr(u_scored, u_controls, TARGET_FPR)
    usgs_self = evaluate_at(u_scored, u_controls, u_episodes, u_thr)
    print(f"[USGS self] thr={u_thr:.4f} fpr={usgs_self['control_heldout_fpr']} "
          f"det={usgs_self['episodes_detected']}/{usgs_self['n_episodes']} "
          f"lead={usgs_self['median_lead_days']}", flush=True)

    # ---- LEAVE-ONE-NETWORK-OUT: calibrate on the OTHER network's controls ----
    # NEON thr applied to USGS episodes (USGS control FPR measured with NEON thr)
    neon_thr_on_usgs = evaluate_at(u_scored, u_controls, u_episodes, n_thr)
    # USGS thr applied to NEON episodes
    usgs_thr_on_neon = evaluate_at(n_scored, n_controls, n_episodes, u_thr)
    print(f"[NEON->USGS] thr={n_thr:.4f} usgs-ctrl-fpr={neon_thr_on_usgs['control_heldout_fpr']} "
          f"det={neon_thr_on_usgs['episodes_detected']}/{neon_thr_on_usgs['n_episodes']} "
          f"lead={neon_thr_on_usgs['median_lead_days']}", flush=True)
    print(f"[USGS->NEON] thr={u_thr:.4f} neon-ctrl-fpr={usgs_thr_on_neon['control_heldout_fpr']} "
          f"det={usgs_thr_on_neon['episodes_detected']}/{usgs_thr_on_neon['n_episodes']} "
          f"lead={usgs_thr_on_neon['median_lead_days']}", flush=True)

    # ---- verdicts (honest Pareto) ----
    NEON_HEADLINE_LEAD = 41.3
    USGS_HEADLINE_LEAD = 41.4
    NEON_HEADLINE_DET = 11   # of 15
    USGS_HEADLINE_DET = 14   # of 17 (self-calibrated headline)

    def _ge_lead(v, base):
        return v is not None and v >= base - 1e-9

    # cross-network win: at matched <=5% FPR, median lead exceeds current headline
    # on BOTH networks WITH >= headline detections on both (never drop detections).
    neon_win = bool(neon_self["control_heldout_fpr"] <= TARGET_FPR + 1e-9
                    and neon_self["episodes_detected"] >= NEON_HEADLINE_DET
                    and _ge_lead(neon_self["median_lead_days"], NEON_HEADLINE_LEAD)
                    and (neon_self["median_lead_days"] or 0) > NEON_HEADLINE_LEAD + 1e-9)
    usgs_win = bool(usgs_self["control_heldout_fpr"] <= TARGET_FPR + 1e-9
                    and usgs_self["episodes_detected"] >= USGS_HEADLINE_DET
                    and _ge_lead(usgs_self["median_lead_days"], USGS_HEADLINE_LEAD)
                    and (usgs_self["median_lead_days"] or 0) > USGS_HEADLINE_LEAD + 1e-9)
    is_cross_network_win = bool(neon_win and usgs_win)

    # leave-one-network-out holds: cross-calibrated threshold still yields a lead
    # gain over the headline on the applied network (gain not overfit to self-cal).
    loco_usgs_ok = bool(neon_thr_on_usgs["control_heldout_fpr"] <= TARGET_FPR + 1e-2
                        and neon_thr_on_usgs["episodes_detected"] >= USGS_HEADLINE_DET
                        and (neon_thr_on_usgs["median_lead_days"] or 0) > USGS_HEADLINE_LEAD + 1e-9)
    loco_neon_ok = bool(usgs_thr_on_neon["control_heldout_fpr"] <= TARGET_FPR + 1e-2
                        and usgs_thr_on_neon["episodes_detected"] >= NEON_HEADLINE_DET
                        and (usgs_thr_on_neon["median_lead_days"] or 0) > NEON_HEADLINE_LEAD + 1e-9)
    leave_one_network_out_holds = bool(loco_usgs_ok and loco_neon_ok)

    result = {
        "variant_key": "skewness_cv",
        "variant_name": "Rising skewness + coefficient of variation",
        "idea": "Extend the verified CSD-EWS channel (rolling variance + lag-1 AC1) "
                "with strictly-causal trailing standardized skewness (signed third "
                "moment; positive = heavy upper tail preceding an upward excursion) and "
                "coefficient of variation (sqrt(var)/max(|mean|,CV_EPS)), stacked onto "
                "variant-1 -> 4 params x 4 feats = 16 self-normalized ews-z series. "
                "Aggregation/fusion/threshold/calibration unchanged; dev_plus=dev+BETA*dev_ews.",
        "hyperparameters": {
            "EWS_WIN": EWS_WIN, "EWS_GUARD": EWS_GUARD, "W_BASE": W_BASE,
            "GUARD": GUARD, "EWS_MIN_BASE": EWS_MIN_BASE, "FLOOR": FLOOR,
            "BETA": BETA, "SLACK_K": SLACK_K, "TARGET_FPR": TARGET_FPR,
            "SKEW_MINPTS": SKEW_MINPTS, "CV_EPS": CV_EPS,
            "note": "all fixed a priori; SKEW_MINPTS/CV_EPS new, rest inherited; none tuned on leads",
        },
        "headline_to_beat": {
            "neon_median_lead": NEON_HEADLINE_LEAD, "neon_detected": f"{NEON_HEADLINE_DET}/15",
            "usgs_median_lead": USGS_HEADLINE_LEAD, "usgs_detected": f"{USGS_HEADLINE_DET}/17",
            "at_fpr": TARGET_FPR,
        },
        "neon_self_calibrated": neon_self,
        "usgs_self_calibrated": usgs_self,
        "leave_one_network_out": {
            "neon_thr_applied_to_usgs": neon_thr_on_usgs,
            "usgs_thr_applied_to_neon": usgs_thr_on_neon,
        },
        "verdict": {
            "neon_beats_headline": neon_win,
            "usgs_beats_headline": usgs_win,
            "is_cross_network_win": is_cross_network_win,
            "leave_one_network_out_holds": leave_one_network_out_holds,
            "criterion_cross_network_win": "at matched <=5% held-out FPR, median lead "
                "> headline (NEON 41.3 / USGS 41.4) with >= headline detections on BOTH networks",
            "criterion_loco": "cross-network-calibrated threshold still yields lead gain "
                "on the applied network (>=~5% FPR, >= headline detections)",
        },
    }

    OUTJSON.parent.mkdir(parents=True, exist_ok=True)
    OUTJSON.write_text(json.dumps(result, indent=2))
    print(f"\nis_cross_network_win={is_cross_network_win}  "
          f"leave_one_network_out_holds={leave_one_network_out_holds}")
    print(f"WROTE {OUTJSON}")


if __name__ == "__main__":
    main()
