#!/usr/bin/env python
"""VARIANT ac_multilag_spectral — Higher-lag autocorrelation + spectral reddening.

Augments the VERIFIED CSD-EWS channel (trailing rolling VARIANCE + lag-1 AC of the
strictly-causal AquaSSM reconstruction residuals) with THREE new critical-slowing
indicators, all computed over the SAME strictly-causal trailing segment:

  * AC2, AC3 : lag-2 and lag-3 autocorrelation of the demeaned trailing residuals.
  * redratio : spectral reddening = (power in the lowest-frequency HALF of the
    Hann-windowed trailing segment's periodogram) / (total positive-freq power).

WHY EARLIER (a-priori mechanism, NOT tuned on leads): near a bifurcation the
dominant eigenvalue -> 1, so the autocorrelation function decays more slowly with
lag; AC2/AC3 integrate the slowing over a longer memory horizon and rise more &
sooner than AC1. Spectral reddening is the frequency-domain dual of critical
slowing: low-frequency variance grows first as the recovery rate collapses, so the
low/total power ratio climbs before the raw variance (which needs a full-band
amplitude increase) and before AC1 saturates. Each series is self-normalized per
site against its OWN trailing baseline (RC.rolling_causal_z), so a stationary
red-but-clean control (biofouling drift, seasonality) sits at z~0 and does not
raise held-out FPR; only a genuine per-site reddening ABOVE its own history fires.

CHANNEL: for each of the 4 params (pH, DO, turbidity, conductivity) we append 5
ews-z series (var, ac1, ac2, ac3, redratio) -> 4x5 = 20 series, each z-scored with
the identical RC.rolling_causal_z(W_BASE=240, GUARD=12, EWS_MIN_BASE=20). Then
dev_ews = RC.aggregate_deviation(all_z, floor=0.0); dev_plus = dev + BETA*dev_ews.
The alarm_fn / CUSUM / threshold / calibration are UNCHANGED from the headline.

HYPERPARAMETERS (ALL FIXED A PRIORI; NONE tuned on leads):
  LAGS = {2,3}; SPEC_MINPTS = 32; spectral split = lowest half of positive
  frequencies (Nyquist/2 cut); Hann window. Inherited: EWS_WIN=180 (15 d),
  EWS_GUARD=12, W_BASE=240, GUARD=12, EWS_MIN_BASE=20, FLOOR=0.0, BETA=1.0,
  SLACK_K=1.0, TARGET_FPR=0.05.

EVALUATION (guardrails A-E respected): matched <=5% held-out CONTROL FPR on
  (a) NEON self-calibrated, (b) USGS self-calibrated, plus LEAVE-ONE-NETWORK-OUT
  (calibrate threshold on the OTHER network's controls, apply to this network's
  episodes) to confirm any lead gain is not overfit. Episodes never calibrate.
  Reports detected + median lead + control held-out FPR for every mode. Honest
  Pareto: a lead improvement counts only if at matched <=5% FPR it detects >= as
  many episodes with >= median lead; BOTH numbers reported regardless.
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
sys.path.insert(0, str(SPEC_DIR / "candidates"))

import cusum_vs_baseline_matchedFPR as M  # noqa: E402
import expanded_matched_fpr as E  # noqa: E402
import rolling_cusum as RC  # noqa: E402

# ---- hyperparameters (FIXED A PRIORI; NEVER tuned on leads) ----
EWS_WIN = 180        # trailing window for var/AC/spectral (15 d @ 2 h stride)
EWS_GUARD = 12       # guard gap (matches CUSUM guard)
W_BASE = M.W_BASE    # 240; baseline window for z-scoring the ews feature series
GUARD = M.GUARD      # 12
EWS_MIN_BASE = 20    # per spec: min_base 20 for the ews-z
FLOOR = 0.0          # ews-z positive-part floor
BETA = 1.0           # ews channel weight
SLACK_K = M.SLACK_K  # 1.0
TARGET_FPR = 0.05
LAGS = (2, 3)        # higher-lag autocorrelation orders
SPEC_MINPTS = 32     # min trailing points required for a spectral estimate
AC_MIN_EXTRA = 8     # AC_k requires seg.size >= k + AC_MIN_EXTRA

# USGS cache locations
USGS_OUT = ROOT / "results" / "trackA_leadtime" / "usgs"
USGS_CACHE = USGS_OUT / "residual_cache_usgs.npz"
USGS_META = json.load(open(USGS_OUT / "cache_meta_usgs.json"))

OUTJSON = ROOT / "results" / "trackA_leadtime" / "combined" / "variants" / "ac_multilag_spectral.json"


def _causal_trailing_features(err_col: np.ndarray, valid: np.ndarray,
                              win: int, guard: int):
    """Strictly-causal trailing var, ac1, ac2, ac3, redratio.

    For window index i the trailing segment is seg = src[lo:hi] with
    hi = i - guard, lo = hi - win (current window + guard EXCLUDED; NO future
    data). All indicators computed on the demeaned finite trailing residuals d.

      AC_k = sum(d[:-k]*d[k:]) / (||d[:-k]|| * ||d[k:]||), only if v>1e-12 and
             seg.size >= k + AC_MIN_EXTRA(=8).
      redratio = (sum of periodogram power over the lowest-frequency HALF of the
             positive freqs) / (total positive-freq power), periodogram =
             |rFFT(d_hann)|^2 on the Hann-windowed demeaned segment; requires
             seg.size >= SPEC_MINPTS(=32) else NaN.

    Returns (var, ac1, ac2, ac3, redratio), each float64[N] with NaN where the
    minimal count is not met.
    """
    n = len(err_col)
    var = np.full(n, np.nan, np.float64)
    ac1 = np.full(n, np.nan, np.float64)
    ac2 = np.full(n, np.nan, np.float64)
    ac3 = np.full(n, np.nan, np.float64)
    red = np.full(n, np.nan, np.float64)
    src = np.where(valid & np.isfinite(err_col), err_col.astype(np.float64), np.nan)
    min_pts = max(8, win // 6)   # need enough points for a stable var/ac (headline)
    ac_store = {1: ac1, 2: ac2, 3: ac3}
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
        if v > 1e-12:
            # autocorrelation at lags 1,2,3 (lag-1 identical to headline formula)
            for k in (1,) + tuple(LAGS):
                if seg.size >= k + AC_MIN_EXTRA:
                    a = d[:-k]
                    b = d[k:]
                    denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
                    if denom > 1e-12:
                        ac_store[k][i] = float(np.sum(a * b) / denom)
            # spectral reddening on Hann-windowed demeaned segment
            if seg.size >= SPEC_MINPTS:
                w = np.hanning(d.size)
                dw = d * w
                # demean again after windowing not needed; Hann already zero-ends.
                spec = np.abs(np.fft.rfft(dw)) ** 2
                # positive frequencies only: drop the DC (0-frequency) bin
                pos = spec[1:]
                tot = float(np.sum(pos))
                if tot > 1e-18 and pos.size >= 2:
                    half = pos.size // 2          # lowest-frequency half (Nyquist/2 cut)
                    if half >= 1:
                        low = float(np.sum(pos[:half]))
                        red[i] = low / tot
    return var, ac1, ac2, ac3, red


def compute_ews_dev(scored):
    """Add sc['dev_ews'] and sc['dev_plus'] using the 20-series augmented channel."""
    for s, sc in scored.items():
        err = sc["err"]
        valid = sc["clean"]
        n, p = err.shape
        ews_z_cols = []
        for j in range(p):
            var_ser, ac1_ser, ac2_ser, ac3_ser, red_ser = _causal_trailing_features(
                err[:, j], valid, EWS_WIN, EWS_GUARD)
            for feat in (var_ser, ac1_ser, ac2_ser, ac3_ser, red_ser):
                fvalid = np.isfinite(feat)
                z = RC.rolling_causal_z(feat, fvalid, W_BASE, GUARD, EWS_MIN_BASE)
                ews_z_cols.append(z)
        ews_z = np.column_stack(ews_z_cols)                    # [N, 20]
        dev_ews = RC.aggregate_deviation(ews_z, floor=FLOOR)
        dev_ews_filled = np.where(np.isfinite(dev_ews), dev_ews, 0.0)
        dev = sc["dev"]
        dev_plus = np.where(np.isfinite(dev), dev + BETA * dev_ews_filled, np.nan)
        sc["dev_ews"] = dev_ews
        sc["dev_plus"] = dev_plus


def alarm_fn(scored, sites, thr):
    return {s: RC.cusum_alarms(scored[s]["dev_plus"], SLACK_K, thr) for s in sites}


def med(x):
    return round(float(st.median(x)), 4) if x else None


# ------------------------- scored builders -------------------------
def build_neon_scored(split_fn):
    """NEON merged scored via E.build_merged_scored (adds b_score, dev), then EWS."""
    scored, controls, episodes = E.build_merged_scored(split_fn)
    compute_ews_dev(scored)
    return scored, controls, episodes


def _load_usgs_site(site):
    z = np.load(USGS_CACHE, allow_pickle=True)
    return {
        "end_ts": z[f"{site}__end_ts"],
        "cov": z[f"{site}__cov"],
        "near": z[f"{site}__near"],
        "err": z[f"{site}__err"].astype(np.float64),
    }


def build_usgs_scored(split_fn):
    scored = {}
    for site, m in USGS_META.items():
        d = _load_usgs_site(site)
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
    M.cusum_dev(scored)        # adds sc['dev']
    compute_ews_dev(scored)    # adds sc['dev_ews'], sc['dev_plus']
    return scored, controls, episodes


# ------------------------- calibration / evaluation -------------------------
def _grid(scored, controls):
    cal_scores = np.concatenate([scored[s]["dev_plus"][scored[s]["cal_clean"]]
                                 for s in controls])
    cal_scores = cal_scores[np.isfinite(cal_scores)]
    hi = np.percentile(cal_scores, 99.97) if cal_scores.size else (SLACK_K + 10)
    return np.unique(np.round(np.linspace(SLACK_K, max(hi, SLACK_K + 1), 300), 4))


def calibrate_thr(scored, controls, target):
    """Largest dev_plus thr with pooled control HELD-OUT FPR <= target."""
    grid = _grid(scored, controls)
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


def main():
    split_fn = M.cal_hold_split

    # ---- build both networks once ----
    n_scored, n_controls, n_episodes = build_neon_scored(split_fn)
    u_scored, u_controls, u_episodes = build_usgs_scored(split_fn)
    print(f"NEON: {len(n_episodes)} ep / {len(n_controls)} ctrl", flush=True)
    print(f"USGS: {len(u_episodes)} ep / {len(u_controls)} ctrl", flush=True)

    # ---- (a) NEON self-calibrated ----
    thr_neon, neon_cal_fpr = calibrate_thr(n_scored, n_controls, TARGET_FPR)
    neon_self = evaluate(n_scored, n_controls, n_episodes, thr_neon)
    print(f"[NEON self] thr={thr_neon:.4f} fpr={neon_self['control_heldout_fpr']} "
          f"det={neon_self['episodes_detected']}/{neon_self['n_episodes']} "
          f"lead={neon_self['median_lead_days']}", flush=True)

    # ---- (b) USGS self-calibrated ----
    thr_usgs, usgs_cal_fpr = calibrate_thr(u_scored, u_controls, TARGET_FPR)
    usgs_self = evaluate(u_scored, u_controls, u_episodes, thr_usgs)
    print(f"[USGS self] thr={thr_usgs:.4f} fpr={usgs_self['control_heldout_fpr']} "
          f"det={usgs_self['episodes_detected']}/{usgs_self['n_episodes']} "
          f"lead={usgs_self['median_lead_days']}", flush=True)

    # ---- LEAVE-ONE-NETWORK-OUT: calibrate on OTHER network's controls ----
    # NEON threshold applied to USGS episodes (+ USGS held-out control FPR)
    neon_thr_on_usgs = evaluate(u_scored, u_controls, u_episodes, thr_neon)
    # USGS threshold applied to NEON episodes (+ NEON held-out control FPR)
    usgs_thr_on_neon = evaluate(n_scored, n_controls, n_episodes, thr_usgs)
    print(f"[NEON thr -> USGS] thr={thr_neon:.4f} usgs-fpr={neon_thr_on_usgs['control_heldout_fpr']} "
          f"det={neon_thr_on_usgs['episodes_detected']}/{neon_thr_on_usgs['n_episodes']} "
          f"lead={neon_thr_on_usgs['median_lead_days']}", flush=True)
    print(f"[USGS thr -> NEON] thr={thr_usgs:.4f} neon-fpr={usgs_thr_on_neon['control_heldout_fpr']} "
          f"det={usgs_thr_on_neon['episodes_detected']}/{usgs_thr_on_neon['n_episodes']} "
          f"lead={usgs_thr_on_neon['median_lead_days']}", flush=True)

    # ---- verdicts (HONEST PARETO) ----
    HEAD_NEON_LEAD = 41.3
    HEAD_USGS_LEAD = 41.4
    HEAD_NEON_DET = 11
    HEAD_USGS_DET = 14

    neon_ml = neon_self["median_lead_days"]
    usgs_ml = usgs_self["median_lead_days"]

    neon_win = bool(
        neon_self["control_heldout_fpr"] <= TARGET_FPR + 1e-9
        and neon_self["episodes_detected"] >= HEAD_NEON_DET
        and neon_ml is not None and neon_ml > HEAD_NEON_LEAD)
    usgs_win = bool(
        usgs_self["control_heldout_fpr"] <= TARGET_FPR + 1e-9
        and usgs_self["episodes_detected"] >= HEAD_USGS_DET
        and usgs_ml is not None and usgs_ml > HEAD_USGS_LEAD)
    is_cross_network_win = bool(neon_win and usgs_win)

    # leave-one-network-out holds: the per-network gain survives OTHER-network
    # calibration (no overfit) — require the cross-calibrated lead to still exceed
    # the headline on BOTH networks with detections on both, at controlled FPR.
    lono_neon_ml = usgs_thr_on_neon["median_lead_days"]
    lono_usgs_ml = neon_thr_on_usgs["median_lead_days"]
    lono_neon_ok = bool(
        usgs_thr_on_neon["control_heldout_fpr"] <= TARGET_FPR + 1e-2
        and usgs_thr_on_neon["episodes_detected"] > 0
        and lono_neon_ml is not None and lono_neon_ml >= HEAD_NEON_LEAD)
    lono_usgs_ok = bool(
        neon_thr_on_usgs["control_heldout_fpr"] <= TARGET_FPR + 1e-2
        and neon_thr_on_usgs["episodes_detected"] > 0
        and lono_usgs_ml is not None and lono_usgs_ml >= HEAD_USGS_LEAD)
    leave_one_network_out_holds = bool(is_cross_network_win and lono_neon_ok and lono_usgs_ok)

    result = {
        "variant_key": "ac_multilag_spectral",
        "variant_name": "Higher-lag autocorrelation + spectral reddening",
        "purpose": "Augment the verified CSD-EWS channel (trailing var + AC1) with "
                   "lag-2/lag-3 autocorrelation and low-frequency spectral reddening "
                   "(4 params x 5 feats = 20 self-normalized ews-z series), fused as "
                   "dev_plus = dev + BETA*dev_ews. Evaluate at matched <=5% held-out "
                   "control FPR on NEON + USGS (self-calibrated) and leave-one-network-out.",
        "hyperparameters": {
            "EWS_WIN": EWS_WIN, "EWS_GUARD": EWS_GUARD, "W_BASE": W_BASE, "GUARD": GUARD,
            "EWS_MIN_BASE": EWS_MIN_BASE, "FLOOR": FLOOR, "BETA": BETA, "SLACK_K": SLACK_K,
            "TARGET_FPR": TARGET_FPR, "LAGS": list(LAGS), "SPEC_MINPTS": SPEC_MINPTS,
            "AC_MIN_EXTRA": AC_MIN_EXTRA, "spectral_split": "lowest half of positive freqs (Nyquist/2 cut)",
            "spectral_window": "Hann", "n_ews_series": 20,
            "note": "all fixed a priori; NONE tuned on leads",
        },
        "headline_to_beat": {"NEON_median_lead": HEAD_NEON_LEAD, "NEON_detected": HEAD_NEON_DET,
                             "USGS_median_lead": HEAD_USGS_LEAD, "USGS_detected": HEAD_USGS_DET,
                             "at_fpr": TARGET_FPR},
        "neon_self_calibrated": neon_self,
        "usgs_self_calibrated": usgs_self,
        "leave_one_network_out": {
            "neon_thr_applied_to_usgs": neon_thr_on_usgs,
            "usgs_thr_applied_to_neon": usgs_thr_on_neon,
            "neon_calib_heldout_fpr": round(float(neon_cal_fpr), 4),
            "usgs_calib_heldout_fpr": round(float(usgs_cal_fpr), 4),
        },
        "verdict": {
            "neon_beats_headline": neon_win,
            "usgs_beats_headline": usgs_win,
            "is_cross_network_win": is_cross_network_win,
            "leave_one_network_out_holds": leave_one_network_out_holds,
            "criteria": {
                "is_cross_network_win": "median lead > headline (NEON 41.3 / USGS 41.4) "
                    "with >= headline detections on BOTH networks at <=5% held-out FPR",
                "leave_one_network_out_holds": "cross-network-calibrated lead still >= "
                    "headline on both networks with detections and controlled FPR",
            },
        },
    }

    OUTJSON.parent.mkdir(parents=True, exist_ok=True)
    OUTJSON.write_text(json.dumps(result, indent=2))
    print(f"\nWROTE {OUTJSON}")
    print(f"is_cross_network_win={is_cross_network_win} "
          f"leave_one_network_out_holds={leave_one_network_out_holds}")


if __name__ == "__main__":
    main()
