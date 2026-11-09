#!/usr/bin/env python
"""VARIANT return_rate: Decreasing return rate (recovery-rate proxy) EWS channel.

Extends the VERIFIED CSD-EWS channel (rolling trailing VARIANCE + lag-1
AUTOCORRELATION, dev_plus = dev + BETA*dev_ews) with a THIRD, dynamically most
direct critical-slowing-down indicator: the trailing per-window RETURN RATE

    lambda_hat = -ln( clip(AC1, AC_EPS, 1-AC_EPS) )

which is the discrete-time linearized recovery rate of the system. Critical
slowing down IS the decline of lambda toward zero as the system approaches a
bifurcation; 1/lambda (the characteristic return time) diverges. Because AC1 is
bounded in [0,1] and saturates near the transition, AC1 alone compresses, whereas
working in log/return-time space linearizes the approach so the z-excursion can
grow steadily (and hence fire the CUSUM EARLIER) rather than plateauing.

The fusion aggregates POSITIVE deviations of z, and a CSD precursor is a FALLING
return rate, so we feed the NEGATED per-site series rt_feat = -lambda_hat (a
RISING series exactly when recovery slows). RC.rolling_causal_z then yields
positive z precisely when the current recovery rate drops BELOW the site's own
trailing-baseline recovery rate. Per-site self-normalization means a control with
an intrinsically slow but STATIONARY recovery rate sits at z~0 and adds no FPR;
only a per-site DROP below trailing recovery rate contributes.

lambda_hat is a MONOTONE transform of the SAME trailing AC1 already estimated
(strictly causal), so it has limited orthogonality to variant 1's AC panel and the
log amplifies AC1 estimator noise; it is therefore placed LAST as a small
complement. This script reports honest numbers whether or not it beats the current
detector (var+AC1 may already be optimal -- a null is valuable).

Hyperparameters ALL FIXED A PRIORI (inherited from the headline CSD-EWS channel):
  EWS_WIN=180, EWS_GUARD=12, W_BASE=240, GUARD=12, EWS_MIN_BASE=20, FLOOR=0.0,
  BETA=1.0, SLACK_K=1.0, TARGET_FPR=0.05, and the NEW clip AC_EPS=1e-3.
NOTHING is tuned on leads. Guardrails A-E respected (label-free; onset/near gate
EVALUATION only via M.episode_leads; matched CONTROL held-out FPR; episodes never
calibrate; strictly causal).

Evaluation on BOTH networks at matched <=5% held-out CONTROL FPR:
  (a) NEON self-calibrated, (b) USGS self-calibrated, plus LEAVE-ONE-NETWORK-OUT
  (calibrate threshold on the OTHER network's controls, apply cross-network) to
  confirm any lead gain is NOT overfit.
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
SPEC_DIR = ROOT / "scripts" / "experiments" / "trackA_leadtime" / "aquassm_specificity"
sys.path.insert(0, str(SPEC_DIR))
sys.path.insert(0, str(SPEC_DIR / "candidates"))

import cusum_vs_baseline_matchedFPR as M  # noqa: E402
import expanded_matched_fpr as E  # noqa: E402
import rolling_cusum as RC  # noqa: E402

# ---- hyperparameters (FIXED A PRIORI; never tuned on leads) ----
EWS_WIN = 180        # trailing window for variance / AC1 / lambda_hat
EWS_GUARD = 12       # guard gap (matches CUSUM guard)
W_BASE = M.W_BASE    # 240; baseline window for z-scoring the ews feature series
GUARD = M.GUARD      # 12
EWS_MIN_BASE = 20    # ews-z min baseline count
FLOOR = 0.0          # ews-z positive-part floor
BETA = 1.0           # ews channel weight
SLACK_K = M.SLACK_K  # 1.0
TARGET_FPR = 0.05
AC_EPS = 1e-3        # a-priori clip for AC1 before -ln (return-rate channel)

USGS_OUT = ROOT / "results" / "trackA_leadtime" / "usgs"
USGS_CACHE = USGS_OUT / "residual_cache_usgs.npz"
USGS_META = json.load(open(USGS_OUT / "cache_meta_usgs.json"))

OUTJSON = ROOT / "results" / "trackA_leadtime" / "combined" / "variants" / "return_rate.json"


# ------------------------- EXTENDED EWS feature extractor -------------------------
def _causal_trailing_var_ac1_rr(err_col: np.ndarray, valid: np.ndarray,
                                win: int, guard: int):
    """Strictly-causal trailing rolling variance, lag-1 AC1, AND return-rate lambda_hat.

    Extends aquassm_specificity/candidates/cand_csd-ews-channel.py
    _causal_trailing_var_ac1: for window i the trailing valid segment
    [i-guard-win, i-guard) is used (current window + `guard` most recent EXCLUDED,
    NO future info). From the SAME trailing lag-1 AC1 already estimated we compute

        lambda_hat = -ln( min(max(ac1, AC_EPS), 1-AC_EPS) )

    the discrete-time linearized recovery rate. Returns (var[N], ac1[N], lam[N])
    with NaN where fewer than a minimal count of valid trailing residuals exist or
    AC1 is undefined.
    """
    n = len(err_col)
    var = np.full(n, np.nan, np.float64)
    ac1 = np.full(n, np.nan, np.float64)
    lam = np.full(n, np.nan, np.float64)
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
            a = d[:-1]
            b = d[1:]
            denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
            if denom > 1e-12:
                r = float(np.sum(a * b) / denom)
                ac1[i] = r
                # discrete-time linearized recovery rate lambda = -ln AC1.
                # clip AC1 into (AC_EPS, 1-AC_EPS) so the log is finite/stable.
                rc = min(max(r, AC_EPS), 1.0 - AC_EPS)
                lam[i] = -float(np.log(rc))
    return var, ac1, lam


def compute_ews_dev_rr(scored):
    """Add sc['dev_ews'] and sc['dev_plus'] using var + AC1 + RETURN-RATE panels.

    Per param (pH, DO, turbidity, conductivity): z-score the variance-series and
    AC1-series (as in variant 1, 8 series) PLUS one RETURN-RATE series fed as the
    NEGATED per-site lambda_hat, rt_feat = -lambda_hat, so a FALLING return rate is
    a RISING series and RC.rolling_causal_z yields positive z exactly when the
    current recovery rate is below the site's trailing-baseline recovery rate.
    Appends 4 rt columns (one per param) LAST -> 12 ews-z columns. Fuse via
    RC.aggregate_deviation (sum of positive parts). dev_plus = dev + BETA*dev_ews.
    Fusion/threshold/calibration UNCHANGED from the headline detector.
    """
    for s, sc in scored.items():
        err = sc["err"]
        valid = sc["clean"]
        n, p = err.shape
        var_ac_cols = []   # variant-1 panel (var + AC1), placed first
        rt_cols = []       # return-rate panel, appended LAST
        for j in range(p):
            var_ser, ac1_ser, lam_ser = _causal_trailing_var_ac1_rr(
                err[:, j], valid, EWS_WIN, EWS_GUARD)
            for feat in (var_ser, ac1_ser):
                fvalid = np.isfinite(feat)
                z = RC.rolling_causal_z(feat, fvalid, W_BASE, GUARD, EWS_MIN_BASE)
                var_ac_cols.append(z)
            # RETURN-RATE: feed the NEGATED lambda_hat so a decreasing recovery
            # rate (slower recovery) produces a POSITIVE deviation.
            rt_feat = -lam_ser
            rt_valid = np.isfinite(rt_feat)
            z_rt = RC.rolling_causal_z(rt_feat, rt_valid, W_BASE, GUARD, EWS_MIN_BASE)
            rt_cols.append(z_rt)
        ews_z = np.column_stack(var_ac_cols + rt_cols)     # [N, 12]  rt LAST
        dev_ews = RC.aggregate_deviation(ews_z, floor=FLOOR)
        dev_ews_filled = np.where(np.isfinite(dev_ews), dev_ews, 0.0)
        dev = sc["dev"]
        dev_plus = np.where(np.isfinite(dev), dev + BETA * dev_ews_filled, np.nan)
        sc["dev_ews"] = dev_ews
        sc["dev_plus"] = dev_plus


# ------------------------- alarms / calibration (UNCHANGED fusion) -------------------------
def alarm_fn(scored, sites, thr):
    return {s: RC.cusum_alarms(scored[s]["dev_plus"], SLACK_K, thr) for s in sites}


def med(x):
    return round(float(st.median(x)), 4) if x else None


def calibrate_thr(scored, controls, target):
    """Largest dev_plus thr with pooled control HELD-OUT FPR <= target (M.calibrate_to_heldout)."""
    cal_scores = np.concatenate([scored[s]["dev_plus"][scored[s]["cal_clean"]]
                                 for s in controls])
    cal_scores = cal_scores[np.isfinite(cal_scores)]
    hi = np.percentile(cal_scores, 99.97) if cal_scores.size else (SLACK_K + 10)
    grid = np.unique(np.round(np.linspace(SLACK_K, max(hi, SLACK_K + 1), 300), 4))
    thr, fpr = M.calibrate_to_heldout(scored, controls, target, grid, alarm_fn)
    return float(thr), float(fpr)


def evaluate(scored, controls, episodes, thr):
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


# ------------------------- network scored builders -------------------------
def build_neon_scored():
    scored, controls, episodes = E.build_merged_scored(M.cal_hold_split)
    compute_ews_dev_rr(scored)
    return scored, controls, episodes


def load_usgs_site(site):
    z = np.load(USGS_CACHE, allow_pickle=True)
    return {
        "end_ts": z[f"{site}__end_ts"],
        "cov": z[f"{site}__cov"],
        "near": z[f"{site}__near"],
        "err": z[f"{site}__err"].astype(np.float64),
    }


def build_usgs_scored():
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
    M.cusum_dev(scored)          # audited mean-shift CUSUM deviation 'dev'
    compute_ews_dev_rr(scored)   # adds 'dev_ews','dev_plus'
    return scored, controls, episodes


def main():
    # ---------- build both networks ----------
    n_scored, n_ctrl, n_ep = build_neon_scored()
    u_scored, u_ctrl, u_ep = build_usgs_scored()
    print(f"NEON: {len(n_ep)} ep / {len(n_ctrl)} ctrl | USGS: {len(u_ep)} ep / {len(u_ctrl)} ctrl",
          flush=True)

    # ---------- (a) NEON self-calibrated ----------
    n_thr, n_calfpr = calibrate_thr(n_scored, n_ctrl, TARGET_FPR)
    neon_self = evaluate(n_scored, n_ctrl, n_ep, n_thr)
    print(f"[NEON self] thr={n_thr:.4f} ctrlFPR={neon_self['control_heldout_fpr']} "
          f"det={neon_self['episodes_detected']}/{len(n_ep)} lead={neon_self['median_lead_days']}",
          flush=True)

    # ---------- (b) USGS self-calibrated ----------
    u_thr, u_calfpr = calibrate_thr(u_scored, u_ctrl, TARGET_FPR)
    usgs_self = evaluate(u_scored, u_ctrl, u_ep, u_thr)
    print(f"[USGS self] thr={u_thr:.4f} ctrlFPR={usgs_self['control_heldout_fpr']} "
          f"det={usgs_self['episodes_detected']}/{len(u_ep)} lead={usgs_self['median_lead_days']}",
          flush=True)

    # ---------- LEAVE-ONE-NETWORK-OUT: calibrate on OTHER network's controls ----------
    # NEON threshold applied to USGS episodes (+ USGS held-out control FPR)
    neon_thr_on_usgs = evaluate(u_scored, u_ctrl, u_ep, n_thr)
    # USGS threshold applied to NEON episodes (+ NEON held-out control FPR)
    usgs_thr_on_neon = evaluate(n_scored, n_ctrl, n_ep, u_thr)
    print(f"[NEON->USGS] thr={n_thr:.4f} USGS-ctrlFPR={neon_thr_on_usgs['control_heldout_fpr']} "
          f"det={neon_thr_on_usgs['episodes_detected']}/{len(u_ep)} "
          f"lead={neon_thr_on_usgs['median_lead_days']}", flush=True)
    print(f"[USGS->NEON] thr={u_thr:.4f} NEON-ctrlFPR={usgs_thr_on_neon['control_heldout_fpr']} "
          f"det={usgs_thr_on_neon['episodes_detected']}/{len(n_ep)} "
          f"lead={usgs_thr_on_neon['median_lead_days']}", flush=True)

    # ---------- verdicts (HONEST PARETO) ----------
    NEON_HEAD_LEAD = 41.3
    USGS_HEAD_LEAD = 41.4
    NEON_HEAD_DET = 11
    USGS_HEAD_DET = 14

    def within(fpr):
        return fpr <= TARGET_FPR + 1e-9

    neon_lead = neon_self["median_lead_days"] or 0.0
    usgs_lead = usgs_self["median_lead_days"] or 0.0
    # cross-network win: at matched <=5% FPR on BOTH, median lead exceeds current
    # headline with >= headline detections on BOTH networks (never drop detections
    # or inflate FPR to game the median).
    neon_win = bool(within(neon_self["control_heldout_fpr"])
                    and neon_self["episodes_detected"] >= NEON_HEAD_DET
                    and neon_lead > NEON_HEAD_LEAD + 1e-9)
    usgs_win = bool(within(usgs_self["control_heldout_fpr"])
                    and usgs_self["episodes_detected"] >= USGS_HEAD_DET
                    and usgs_lead > USGS_HEAD_LEAD + 1e-9)
    is_cross_network_win = bool(neon_win and usgs_win)

    # leave-one-network-out holds: the gain survives cross-network calibration, i.e.
    # applying the OTHER network's threshold still yields median lead exceeding the
    # headline on the target network (with real detections at controlled FPR).
    # LOCO check (FPR measured on target network's own held-out controls):
    loco_usgs_ok = bool((neon_thr_on_usgs["median_lead_days"] or 0.0) > USGS_HEAD_LEAD + 1e-9
                        and neon_thr_on_usgs["episodes_detected"] >= USGS_HEAD_DET)
    loco_neon_ok = bool((usgs_thr_on_neon["median_lead_days"] or 0.0) > NEON_HEAD_LEAD + 1e-9
                        and usgs_thr_on_neon["episodes_detected"] >= NEON_HEAD_DET)
    leave_one_network_out_holds = bool(loco_usgs_ok and loco_neon_ok)

    result = {
        "variant_key": "return_rate",
        "variant_name": "Decreasing return rate (recovery-rate proxy)",
        "idea": ("Extend the verified CSD-EWS channel (trailing variance + lag-1 AC1) "
                 "with a THIRD indicator: trailing return-rate lambda_hat = -ln(clip(AC1)), "
                 "the discrete-time linearized recovery rate. Fed as -lambda_hat so a "
                 "FALLING return rate (slower recovery, the CSD precursor) yields a "
                 "POSITIVE per-site z; per-site self-normalized so stationary-slow "
                 "controls add no FPR. Placed LAST (monotone transform of AC1)."),
        "hyperparameters": {
            "EWS_WIN": EWS_WIN, "EWS_GUARD": EWS_GUARD, "W_BASE": W_BASE, "GUARD": GUARD,
            "EWS_MIN_BASE": EWS_MIN_BASE, "FLOOR": FLOOR, "BETA": BETA, "SLACK_K": SLACK_K,
            "TARGET_FPR": TARGET_FPR, "AC_EPS": AC_EPS,
            "note": "all fixed a priori; AC_EPS=1e-3 a-priori; nothing tuned on leads",
        },
        "panels": "var(4) + AC1(4) + return_rate(4)  [return_rate columns appended LAST]",
        "headline_to_beat": {"NEON_median_lead": NEON_HEAD_LEAD, "NEON_detected": NEON_HEAD_DET,
                             "USGS_median_lead": USGS_HEAD_LEAD, "USGS_detected": USGS_HEAD_DET,
                             "at_matched_fpr": TARGET_FPR},
        "neon_self_calibrated": neon_self,
        "usgs_self_calibrated": usgs_self,
        "leave_one_network_out": {
            "neon_thr_applied_to_usgs": {"thr": round(n_thr, 4), **neon_thr_on_usgs},
            "usgs_thr_applied_to_neon": {"thr": round(u_thr, 4), **usgs_thr_on_neon},
            "note": ("Threshold calibrated ONLY on the other network's controls, then "
                     "applied to this network's episodes; control FPR reported on THIS "
                     "network's held-out controls. Confirms any lead gain is not overfit."),
        },
        "verdict": {
            "neon_win_vs_headline": neon_win,
            "usgs_win_vs_headline": usgs_win,
            "is_cross_network_win": is_cross_network_win,
            "leave_one_network_out_holds": leave_one_network_out_holds,
            "criterion": ("cross-network win = at <=5% held-out FPR, median lead exceeds "
                          "headline (NEON 41.3 / USGS 41.4) with >= headline detections "
                          "(NEON 11 / USGS 14) on BOTH networks; never drop detections or "
                          "inflate FPR to game the median."),
        },
    }
    OUTJSON.parent.mkdir(parents=True, exist_ok=True)
    OUTJSON.write_text(json.dumps(result, indent=2))

    print("\n=== return_rate variant summary ===")
    print(f"NEON self : det={neon_self['episodes_detected']}/{len(n_ep)} "
          f"lead={neon_self['median_lead_days']} fpr={neon_self['control_heldout_fpr']} "
          f"(headline 11 @ 41.3)")
    print(f"USGS self : det={usgs_self['episodes_detected']}/{len(u_ep)} "
          f"lead={usgs_self['median_lead_days']} fpr={usgs_self['control_heldout_fpr']} "
          f"(headline 14 @ 41.4)")
    print(f"is_cross_network_win={is_cross_network_win}  "
          f"leave_one_network_out_holds={leave_one_network_out_holds}")
    print(f"WROTE {OUTJSON}")


if __name__ == "__main__":
    main()
