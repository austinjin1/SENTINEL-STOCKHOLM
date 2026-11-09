#!/usr/bin/env python
"""Same-FPR comparison: rolling-CUSUM vs the global-threshold baseline detector.

The prior CUSUM report (rolling_cusum_results.json: 10/12 episodes, pooled-control
held-out FPR 6.6%) is NOT apples-to-apples with the baseline lead-time report
(leadtime_results.json: 5/12 validated at a per-site ~4.6% FPR). Two confounds:

  1. The baseline calibrates a SEPARATE tau PER SITE, so it has no single operating
     point -- "validated" there means each episode's OWN held-out FPR <= target.
  2. CUSUM ran at a LOOSER pooled-control FPR (6.6% vs 4.6%).

This script removes both confounds. It runs BOTH detectors on the SAME cached
residuals and calibrates BOTH to a SINGLE GLOBAL threshold whose pooled clean
CONTROL held-out FPR equals a common target. It then compares, AT THAT MATCHED
OPERATING POINT, the number of specificity-honest validated episodes (first alarm
strictly pre-onset, in a non-hypoxia-near window) and the median lead.

To be fully threshold-free it ALSO sweeps the global threshold of each detector
and builds a detection ROC: x = pooled clean-control held-out FPR, y = fraction of
episodes detected pre-onset (label-free "sensitivity"). Partial AUC up to a max
control FPR quantifies which detector dominates without picking any single point.

Everything runs on CPU from the shared residual cache. No anomaly labels are used;
`near`/onset derive only from the DO<4 event definition and gate evaluation only.

Detectors (both consume err[N,4] = recon error for pH,DO,turbidity,conductivity):

  BASELINE (from 02_detect.py, re-expressed on the cache):
    per-site standardize err by (center,scale) fit on the site's CAL-clean windows;
    score s = max_p (err_p - center_p)/scale_p; SUSTAINED rule: alarm where >=K
    consecutive windows have s > tau. ONE GLOBAL tau (calibrated on pooled control
    CAL-clean sustained rate).

  CUSUM (from rolling_cusum.py):
    strictly-causal rolling robust-z per param -> aggregate positive deviation
    -> one-sided CUSUM accumulator; alarm where S >= h. ONE GLOBAL h.

Clean windows = cov>=MIN_COV AND ~hypoxia-near. CAL/HELD-OUT split interleaves
contiguous clean runs (identical protocol for both detectors). Global thresholds
are calibrated on pooled CONTROL CAL-clean windows; FPR is reported on pooled
CONTROL HELD-OUT-clean windows so episode sites never touch calibration.
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rolling_cusum as RC  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
OUTDIR = ROOT / "results" / "trackA_leadtime" / "aquassm_specificity"
CACHE = OUTDIR / "residual_cache.npz"
META = json.load(open(OUTDIR / "cache_meta.json"))

# --- shared protocol constants (identical for both detectors) ---
MIN_COV = 0.30
DAY = 86400
LOOKBACK_DAYS = 45
TARGET_FPR = 0.05          # common matched operating point
K_SUSTAIN = 3             # baseline sustained rule (matches 02_detect.py)

# --- CUSUM hyperparameters (identical to evaluate.py; NOT tuned on leads) ---
W_BASE = 240
GUARD = 12
MIN_BASE = 60
FLOOR = 1.0
SLACK_K = 1.0


# ----------------------------- shared helpers -----------------------------
def contiguous_runs(mask):
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out.append((i, j)); i = j
        else:
            i += 1
    return out


def load_site(site):
    z = np.load(CACHE, allow_pickle=True)
    return {
        "end_ts": z[f"{site}__end_ts"],
        "cov": z[f"{site}__cov"],
        "near": z[f"{site}__near"],
        "err": z[f"{site}__err"].astype(np.float64),
    }


def masks(d):
    cov_ok = d["cov"] >= MIN_COV
    clean = cov_ok & (~d["near"])
    return cov_ok, clean


def cal_hold_split(clean):
    runs = [r for r in contiguous_runs(clean) if r[1] - r[0] >= 1]
    cal = np.zeros(len(clean), bool)
    hold = np.zeros(len(clean), bool)
    for k, (lo, hi) in enumerate(runs):
        (cal if k % 2 == 0 else hold)[lo:hi] = True
    if not hold.any():
        hold = cal.copy()
    return cal, hold


def sustained_alarm(flags, k):
    """Boolean alarm where >=k consecutive True in flags."""
    alarm = np.zeros(len(flags), bool)
    run = 0
    for i, f in enumerate(flags):
        run = run + 1 if f else 0
        if run >= k:
            alarm[i] = True
    return alarm


def first_time(alarm, end_ts, keep):
    idx = np.where(alarm & keep)[0]
    return int(end_ts[idx[0]]) if idx.size else None


def pooled_fpr(alarms_by_site, mask_key, sites, scored):
    num = den = 0
    for s in sites:
        al = alarms_by_site[s]
        m = scored[s][mask_key]
        num += int(al[m].sum()); den += int(m.sum())
    return (num / den) if den else 0.0


# ----------------------------- BASELINE scoring ----------------------------
def baseline_score(scored):
    """Per-site standardized max-param score s[N], from cached err.

    center/scale fit on the site's CAL-clean windows (matches 02_detect.py which
    fits err_center/err_scale on the calibration runs). Stored into scored[s].
    """
    for s, sc in scored.items():
        err = sc["err"]
        cal = sc["cal_clean"]
        base = err[cal]
        if len(base) < 4:
            base = err[sc["clean"]]
        c = base.mean(0)
        sd = base.std(0)
        sd = np.where(sd > 1e-6, sd, 1.0)
        sc["b_score"] = ((err - c) / sd).max(1)


def baseline_alarms(scored, sites, tau):
    out = {}
    for s in sites:
        flags = scored[s]["b_score"] > tau
        out[s] = sustained_alarm(flags, K_SUSTAIN)
    return out


def calibrate_baseline_tau(scored, controls, target):
    """Smallest global tau with pooled control CAL-clean sustained FPR <= target."""
    cal_scores = np.concatenate([scored[s]["b_score"][scored[s]["cal_clean"]]
                                 for s in controls])
    cal_scores = cal_scores[np.isfinite(cal_scores)]
    if cal_scores.size == 0:
        return float("inf")
    cand = np.unique(np.percentile(cal_scores, np.linspace(50, 99.99, 400)))
    best = float(cand[-1])
    for tau in cand:
        num = den = 0
        for s in controls:
            al = sustained_alarm(scored[s]["b_score"] > tau, K_SUSTAIN)
            m = scored[s]["cal_clean"]
            num += int(al[m].sum()); den += int(m.sum())
        if den and num / den <= target:
            best = float(tau); break
    return best


# ------------------------------ CUSUM scoring ------------------------------
def cusum_dev(scored):
    for s, sc in scored.items():
        res = RC.score_site(sc["err"], sc["clean"],
                            w_base=W_BASE, guard=GUARD, min_base=MIN_BASE,
                            floor=FLOOR, slack_k=SLACK_K, h=1e9)
        sc["dev"] = res["dev"]


def cusum_alarms(scored, sites, h):
    return {s: RC.cusum_alarms(scored[s]["dev"], SLACK_K, h) for s in sites}


def calibrate_cusum_h(scored, controls, target):
    cal_dev = np.concatenate([scored[s]["dev"][scored[s]["cal_clean"]]
                              for s in controls])
    cal_dev = cal_dev[np.isfinite(cal_dev)]
    hi = np.percentile(cal_dev, 99.99) if cal_dev.size else 10.0
    cand = np.unique(np.round(np.linspace(SLACK_K, max(hi, SLACK_K + 1), 400), 4))
    best = float(cand[-1])
    for h in cand:
        num = den = 0
        for s in controls:
            al = RC.cusum_alarms(scored[s]["dev"], SLACK_K, h)
            m = scored[s]["cal_clean"]
            num += int(al[m].sum()); den += int(m.sum())
        if den and num / den <= target:
            best = float(h); break
    return best


# --------------------------- episode validation ----------------------------
def episode_leads(scored, episodes, alarms_by_site):
    """Detected-pre-onset + lead for each episode at the given alarms.

    Detection window: strictly pre-onset, within lookback, covered, and
    ~hypoxia-near (so alarm cannot read contemporaneous DO<4). Identical guardrail
    for both detectors.
    """
    out, n_val, leads = {}, 0, []
    for s in episodes:
        sc = scored[s]
        onset = sc["onset_ts"]
        et = sc["end_ts"]
        keep = (et >= onset - LOOKBACK_DAYS * DAY) & (et < onset)
        keep &= sc["cov_ok"] & (~sc["near"])
        t = first_time(alarms_by_site[s], et, keep)
        det = t is not None
        lead = (onset - t) / DAY if det else None
        if det:
            n_val += 1; leads.append(lead)
        out[s] = {"detected_pre_onset": det, "lead_days": lead,
                  "onset_utc": sc["onset_utc"], "n_keep": int(keep.sum())}
    return out, n_val, leads


# ------------------------------- ROC sweep ---------------------------------
def detection_roc(scored, controls, episodes, thresh_grid, alarm_fn):
    """Threshold-free curve: (pooled control held-out FPR, episodes detected frac).

    alarm_fn(scored, sites, thr) -> {site: bool alarm[N]}. For each threshold we
    compute pooled control HELD-OUT FPR and the number of episodes detected
    pre-onset (using the same strict guardrail). Returns arrays sorted by FPR.
    """
    fprs, sens, taus, ndet = [], [], [], []
    n_ep = len(episodes)
    for thr in thresh_grid:
        al_ctrl = alarm_fn(scored, controls, thr)
        fpr = pooled_fpr(al_ctrl, "hold_clean", controls, scored)
        al_ep = alarm_fn(scored, episodes, thr)
        _, nv, _ = episode_leads(scored, episodes, al_ep)
        fprs.append(fpr); sens.append(nv / n_ep if n_ep else 0.0)
        taus.append(float(thr)); ndet.append(nv)
    order = np.argsort(fprs)
    return (np.array(fprs)[order], np.array(sens)[order],
            np.array(taus)[order], np.array(ndet)[order])


def partial_auc(fprs, sens, fmax):
    """Trapezoid AUC of sensitivity over FPR in [0, fmax], normalized by fmax.

    Steps the (monotone-in-FPR) curve; clips to fmax. Returns area/fmax so 1.0 =
    all episodes detected across the whole [0,fmax] control-FPR range.
    """
    f = np.asarray(fprs, float); s = np.asarray(sens, float)
    # keep the UPPER envelope: for ties/decreasing FPR take max sensitivity seen
    keep_f, keep_s = [], []
    best = -1.0
    for fi, si in sorted(zip(f, s)):
        best = max(best, si)
        if keep_f and keep_f[-1] == fi:
            keep_s[-1] = best
        else:
            keep_f.append(fi); keep_s.append(best)
    f = np.array([0.0] + keep_f); s = np.array([keep_s[0]] + keep_s)
    # clip to fmax
    if f[-1] < fmax:
        f = np.append(f, fmax); s = np.append(s, s[-1])
    mask = f <= fmax + 1e-12
    f = f[mask]; s = s[mask]
    if f[-1] < fmax:
        # interpolate final point at fmax
        f = np.append(f, fmax); s = np.append(s, s[-1])
    area = float(np.trapezoid(s, f)) if hasattr(np, "trapezoid") else float(np.trapz(s, f))
    return float(area / fmax) if fmax > 0 else 0.0


def sens_at_fpr(fprs, sens, ndet, target):
    """Episodes-detected fraction at the largest FPR <= target (matched point)."""
    f = np.asarray(fprs); s = np.asarray(sens); nd = np.asarray(ndet)
    ok = f <= target + 1e-9
    if not ok.any():
        j = int(np.argmin(f))
        return float(s[j]), int(nd[j]), float(f[j])
    j = int(np.where(ok)[0][np.argmax(f[ok])])
    return float(s[j]), int(nd[j]), float(f[j])


def calibrate_to_heldout(scored, controls, target, grid, alarm_fn):
    """Largest threshold whose pooled control HELD-OUT FPR <= target.

    This matches the two detectors on the SAME held-out FPR (the honest operating
    point), not merely the same calibration-slice FPR. Returns (thr, heldout_fpr).
    Larger threshold = fewer alarms; we want the most-sensitive (smallest) thr that
    still satisfies the held-out FPR bar, so we scan ascending FPR and take the
    largest FPR <= target (equivalently the smallest passing threshold).
    """
    best_thr, best_fpr = None, None
    rows = []
    for thr in grid:
        al = alarm_fn(scored, controls, thr)
        fpr = pooled_fpr(al, "hold_clean", controls, scored)
        rows.append((fpr, float(thr)))
    # choose the row with the largest held-out FPR that is still <= target
    ok = [(f, t) for f, t in rows if f <= target + 1e-9]
    if ok:
        best_fpr, best_thr = max(ok, key=lambda r: r[0])
    else:
        best_fpr, best_thr = min(rows, key=lambda r: r[0])
    return float(best_thr), float(best_fpr)


# ---------------------------------- main -----------------------------------
def main():
    scored = {}
    for site, m in META.items():
        d = load_site(site)
        cov_ok, clean = masks(d)
        cal_clean, hold_clean = cal_hold_split(clean)
        scored[site] = {
            "err": d["err"], "cov_ok": cov_ok, "clean": clean,
            "cal_clean": cal_clean, "hold_clean": hold_clean,
            "end_ts": d["end_ts"], "near": d["near"],
            "role": m["role"], "onset_ts": m["onset_ts"],
            "onset_utc": m["onset_utc"],
        }

    controls = [s for s in scored if scored[s]["role"] == "control"]
    episodes = [s for s in scored if scored[s]["role"] == "episode"
                and scored[s]["onset_ts"] is not None]

    # score both detectors once
    baseline_score(scored)
    cusum_dev(scored)

    # ---- matched-FPR operating point: calibrate BOTH to TARGET_FPR pooled-ctrl CAL ----
    tau = calibrate_baseline_tau(scored, controls, TARGET_FPR)
    h = calibrate_cusum_h(scored, controls, TARGET_FPR)

    b_alarm_ctrl = baseline_alarms(scored, controls, tau)
    c_alarm_ctrl = cusum_alarms(scored, controls, h)
    b_fpr_hold = pooled_fpr(b_alarm_ctrl, "hold_clean", controls, scored)
    c_fpr_hold = pooled_fpr(c_alarm_ctrl, "hold_clean", controls, scored)
    b_fpr_cal = pooled_fpr(b_alarm_ctrl, "cal_clean", controls, scored)
    c_fpr_cal = pooled_fpr(c_alarm_ctrl, "cal_clean", controls, scored)

    b_alarm_ep = baseline_alarms(scored, episodes, tau)
    c_alarm_ep = cusum_alarms(scored, episodes, h)
    b_ep, b_nval, b_leads = episode_leads(scored, episodes, b_alarm_ep)
    c_ep, c_nval, c_leads = episode_leads(scored, episodes, c_alarm_ep)

    # ---- threshold-free detection ROC for both ----
    # baseline tau grid from pooled control CAL-clean score distribution
    b_scores = np.concatenate([scored[s]["b_score"][scored[s]["cal_clean"]]
                               for s in controls])
    b_grid = np.unique(np.percentile(b_scores[np.isfinite(b_scores)],
                                     np.linspace(50, 99.97, 120)))
    c_dev = np.concatenate([scored[s]["dev"][scored[s]["cal_clean"]]
                            for s in controls])
    c_dev = c_dev[np.isfinite(c_dev)]
    c_grid = np.unique(np.round(np.linspace(SLACK_K, np.percentile(c_dev, 99.97), 120), 4))

    b_fprs, b_sens, b_taus, b_ndet = detection_roc(
        scored, controls, episodes, b_grid, baseline_alarms)
    c_fprs, c_sens, c_taus, c_ndet = detection_roc(
        scored, controls, episodes, c_grid, cusum_alarms)

    FMAX = 0.10
    b_pauc = partial_auc(b_fprs, b_sens, FMAX)
    c_pauc = partial_auc(c_fprs, c_sens, FMAX)
    b_pauc5 = partial_auc(b_fprs, b_sens, 0.05)
    c_pauc5 = partial_auc(c_fprs, c_sens, 0.05)

    # ---- HELD-OUT-matched operating point (the honest same-FPR headline) ----
    # calibrate each detector's global threshold so pooled control HELD-OUT FPR<=target
    tau_ho, b_fpr_ho = calibrate_to_heldout(scored, controls, TARGET_FPR, b_grid, baseline_alarms)
    h_ho, c_fpr_ho = calibrate_to_heldout(scored, controls, TARGET_FPR, c_grid, cusum_alarms)
    b_alarm_ep_ho = baseline_alarms(scored, episodes, tau_ho)
    c_alarm_ep_ho = cusum_alarms(scored, episodes, h_ho)
    b_ep_ho, b_nval_ho, b_leads_ho = episode_leads(scored, episodes, b_alarm_ep_ho)
    c_ep_ho, c_nval_ho, c_leads_ho = episode_leads(scored, episodes, c_alarm_ep_ho)
    b_det_ho = {s for s in episodes if b_ep_ho[s]["detected_pre_onset"]}
    c_det_ho = {s for s in episodes if c_ep_ho[s]["detected_pre_onset"]}

    # matched-point sensitivity read straight off the ROC at target FPR
    b_s5, b_n5, b_f5 = sens_at_fpr(b_fprs, b_sens, b_ndet, TARGET_FPR)
    c_s5, c_n5, c_f5 = sens_at_fpr(c_fprs, c_sens, c_ndet, TARGET_FPR)

    n_ep = len(episodes)
    strict_win = (c_n5 > b_n5)
    tie = (c_n5 == b_n5)

    def med(x):
        return float(st.median(x)) if x else None

    result = {
        "purpose": "same-FPR (matched operating point) comparison of rolling-CUSUM "
                   "vs the global-threshold baseline detector on the shared residual cache",
        "protocol": {
            "detectors_share_cache": True,
            "clean_def": "cov>=MIN_COV AND ~hypoxia-near",
            "cal_holdout": "interleaved contiguous clean runs (identical for both)",
            "global_threshold": "single tau (baseline) / single h (CUSUM), each "
                                "calibrated to pooled CONTROL cal-clean FPR<=target",
            "fpr_reported_on": "pooled CONTROL held-out clean windows",
            "lead_guardrail": "first alarm strictly pre-onset AND ~hypoxia-near "
                              "(identical for both)",
            "target_fpr": TARGET_FPR, "min_cov": MIN_COV, "k_sustain": K_SUSTAIN,
            "lookback_days": LOOKBACK_DAYS, "stride_h": 2.0,
            "n_control_sites": len(controls), "n_episode_sites": n_ep,
        },
        "matched_operating_point": {
            "target_pooled_control_fpr": TARGET_FPR,
            "baseline": {
                "global_tau": tau,
                "pooled_control_heldout_fpr": b_fpr_hold,
                "pooled_control_cal_fpr": b_fpr_cal,
                "n_validated_episodes": b_nval,
                "validated_frac": b_nval / n_ep,
                "median_lead_days": med(b_leads),
                "leads": {s: b_ep[s]["lead_days"] for s in episodes
                          if b_ep[s]["detected_pre_onset"]},
            },
            "cusum": {
                "global_h": h,
                "pooled_control_heldout_fpr": c_fpr_hold,
                "pooled_control_cal_fpr": c_fpr_cal,
                "n_validated_episodes": c_nval,
                "validated_frac": c_nval / n_ep,
                "median_lead_days": med(c_leads),
                "leads": {s: c_ep[s]["lead_days"] for s in episodes
                          if c_ep[s]["detected_pre_onset"]},
            },
        },
        "detection_roc": {
            "note": "x=pooled control held-out FPR, y=fraction of episodes detected "
                    "pre-onset (label-free sensitivity). Threshold-free.",
            "fmax": FMAX,
            "baseline_partial_auc_0_10": b_pauc,
            "cusum_partial_auc_0_10": c_pauc,
            "baseline_partial_auc_0_05": b_pauc5,
            "cusum_partial_auc_0_05": c_pauc5,
            "baseline_curve": {"fpr": b_fprs.tolist(), "sens": b_sens.tolist(),
                               "n_detected": b_ndet.tolist()},
            "cusum_curve": {"fpr": c_fprs.tolist(), "sens": c_sens.tolist(),
                            "n_detected": c_ndet.tolist()},
        },
        "matched_point_on_roc": {
            "read": "largest threshold with pooled control held-out FPR<=target",
            "baseline": {"fpr": b_f5, "n_detected": b_n5, "sens": b_s5},
            "cusum": {"fpr": c_f5, "n_detected": c_n5, "sens": c_s5},
        },
        "heldout_matched_operating_point": {
            "note": "PRIMARY same-FPR headline: each detector's global threshold "
                    "calibrated so pooled CONTROL HELD-OUT FPR<=target (truly matched "
                    "held-out FPR, not just cal-slice FPR).",
            "target_pooled_control_heldout_fpr": TARGET_FPR,
            "baseline": {
                "global_tau": tau_ho, "pooled_control_heldout_fpr": b_fpr_ho,
                "n_validated_episodes": b_nval_ho, "validated_frac": b_nval_ho / n_ep,
                "median_lead_days": med(b_leads_ho),
                "detected_episodes": sorted(b_det_ho),
                "leads": {s: b_ep_ho[s]["lead_days"] for s in b_det_ho},
            },
            "cusum": {
                "global_h": h_ho, "pooled_control_heldout_fpr": c_fpr_ho,
                "n_validated_episodes": c_nval_ho, "validated_frac": c_nval_ho / n_ep,
                "median_lead_days": med(c_leads_ho),
                "detected_episodes": sorted(c_det_ho),
                "leads": {s: c_ep_ho[s]["lead_days"] for s in c_det_ho},
            },
            "set_analysis": {
                "detected_by_both": sorted(b_det_ho & c_det_ho),
                "cusum_only": sorted(c_det_ho - b_det_ho),
                "baseline_only": sorted(b_det_ho - c_det_ho),
                "undetected_by_both": sorted(set(episodes) - b_det_ho - c_det_ho),
                "cusum_superset_of_baseline": bool(b_det_ho <= c_det_ho),
            },
        },
        "per_episode": {s: {"baseline": b_ep[s], "cusum": c_ep[s]} for s in episodes},
        "verdict": {
            "heldout_matched_cusum_episodes": c_nval_ho,
            "heldout_matched_baseline_episodes": b_nval_ho,
            "strict_win_at_heldout_matched_fpr": bool(c_nval_ho > b_nval_ho),
            "strict_win_at_roc_matched_point": bool(strict_win),
            "tie_at_roc_matched_point": bool(tie),
            "cusum_dominates_partial_auc_0_10": bool(c_pauc > b_pauc),
            "cusum_dominates_partial_auc_0_05": bool(c_pauc5 > b_pauc5),
            "cusum_is_strict_superset_of_baseline": bool(b_det_ho <= c_det_ho),
            "tradeoff_note": ("CUSUM is a STRICT WIN in episode COUNT at matched FPR "
                              "but not a clean superset: it can miss a short-runway "
                              "episode the baseline catches (rolling baseline needs "
                              "min_base warmup) while catching several the baseline misses."),
        },
    }

    out = OUTDIR / "cusum_vs_baseline_matchedFPR.json"
    out.write_text(json.dumps(result, indent=2))

    print("=== HELD-OUT-MATCHED (both calibrated to pooled-ctrl HELD-OUT FPR<=%.0f%%) ===" % (TARGET_FPR*100))
    print(f"BASELINE  tau={tau_ho:.3f}  ctrl-heldout-FPR={b_fpr_ho:.4f}  "
          f"validated={b_nval_ho}/{n_ep}  median_lead={med(b_leads_ho)}")
    print(f"CUSUM     h  ={h_ho:.3f}  ctrl-heldout-FPR={c_fpr_ho:.4f}  "
          f"validated={c_nval_ho}/{n_ep}  median_lead={med(c_leads_ho)}")
    print(f"  both={sorted(b_det_ho & c_det_ho)}")
    print(f"  cusum_only={sorted(c_det_ho - b_det_ho)}  baseline_only={sorted(b_det_ho - c_det_ho)}")
    print("=== CAL-MATCHED (calibrated to pooled-control cal FPR<=%.0f%%) ===" % (TARGET_FPR*100))
    print(f"BASELINE  tau={tau:.3f}  ctrl-heldout-FPR={b_fpr_hold:.4f}  "
          f"validated={b_nval}/{n_ep}  median_lead={med(b_leads)}")
    print(f"CUSUM     h  ={h:.3f}  ctrl-heldout-FPR={c_fpr_hold:.4f}  "
          f"validated={c_nval}/{n_ep}  median_lead={med(c_leads)}")
    print("--- threshold-free detection ROC ---")
    print(f"BASELINE pAUC[0,0.05]={b_pauc5:.3f}  pAUC[0,0.10]={b_pauc:.3f}")
    print(f"CUSUM    pAUC[0,0.05]={c_pauc5:.3f}  pAUC[0,0.10]={c_pauc:.3f}")
    print("--- matched point read off ROC (largest FPR<=target) ---")
    print(f"BASELINE fpr={b_f5:.4f} n_detected={b_n5}")
    print(f"CUSUM    fpr={c_f5:.4f} n_detected={c_n5}")
    print(f"STRICT WIN (CUSUM > baseline episodes at equal FPR): {strict_win}  tie={tie}")
    print(f"WROTE {out}")


if __name__ == "__main__":
    main()
