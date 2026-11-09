#!/usr/bin/env python
"""Cross-network CSD-EWS detection on the ZERO-SHOT USGS residual cache.

Uses the FROZEN NEON detector (AquaSSM encoder + CSD-EWS fused CUSUM channel,
dev_plus = dev + BETA*dev_ews, headline params EWS_WIN=180, EWS_MIN_BASE=20,
BETA=1.0) applied to USGS residuals with NO retraining.

Two modes, each reporting detected-pre-onset / median lead / control HELD-OUT FPR:

  (a) USGS-SELF-CALIBRATED: calibrate global thr on USGS CONTROL clean windows at
      5% held-out FPR, apply to USGS episodes. Tests whether the zero-shot NEON
      encoder yields a usable USGS precursor.

  (b) NEON-TRANSFERRED: take the CSD-EWS global thr calibrated on NEON controls at
      5% held-out FPR (recomputed from the NEON caches), and apply that SAME thr to
      USGS episodes + measure USGS control held-out FPR. Tests cross-network
      operating-point transfer.

Why the raw dev_plus threshold transfers fairly: dev_plus is built ENTIRELY from
per-site STRICTLY-CAUSAL rolling robust-z (RC.rolling_causal_z) of the reconstruction
residuals and their EWS features. Every component is self-normalized against that
site's own trailing clean baseline (median/MAD), so the dev_plus scale is intrinsic
to the detector, not to the network's absolute residual magnitude. A CUSUM threshold
h in dev_plus units therefore denotes the same statistical operating point on either
network. We ALSO report the transfer explicitly and, as a robustness cross-check, the
per-site self-normalized-score variant is exactly what dev_plus already is (there is
no additional site-specific rescaling to undo), so the raw-threshold transfer IS the
fair comparison. We note this in the JSON.

All guardrails A-E respected: label-free detector; onset/near gate EVALUATION only;
matched CONTROL held-out FPR (episodes never calibrate); a-priori fixed hyperparams;
real QC-clean drift-screened episodes only.
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SPEC_DIR = HERE.parent / "aquassm_specificity"
sys.path.insert(0, str(HERE.parent))          # for _detector if needed
sys.path.insert(0, str(SPEC_DIR))
sys.path.insert(0, str(SPEC_DIR / "candidates"))

import importlib.util as _ilu

import cusum_vs_baseline_matchedFPR as M  # noqa: E402
import rolling_cusum as RC  # noqa: E402

# candidate module file has a hyphen in its name -> load by path
_ce_path = SPEC_DIR / "candidates" / "cand_csd-ews-channel.py"
_spec = _ilu.spec_from_file_location("cand_csd_ews_channel", _ce_path)
CE = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(CE)


ROOT = Path(__file__).resolve().parents[4]
NEON_OUT = ROOT / "results" / "trackA_leadtime" / "aquassm_specificity"
USGS_OUT = ROOT / "results" / "trackA_leadtime" / "usgs"
USGS_CACHE = USGS_OUT / "residual_cache_usgs.npz"
USGS_META = json.load(open(USGS_OUT / "cache_meta_usgs.json"))

NEON_CACHE0 = NEON_OUT / "residual_cache.npz"
NEON_CACHE1 = NEON_OUT / "residual_cache_expanded.npz"
NEON_META0 = json.load(open(NEON_OUT / "cache_meta.json"))
NEON_META1 = json.load(open(NEON_OUT / "cache_meta_expanded.json"))

TARGET_FPR = 0.05
SLACK_K = CE.SLACK_K


def med(x):
    return round(float(st.median(x)), 4) if x else None


# ------------------------- USGS scored builder -------------------------
def load_usgs_site(site):
    z = np.load(USGS_CACHE, allow_pickle=True)
    return {
        "end_ts": z[f"{site}__end_ts"],
        "cov": z[f"{site}__cov"],
        "near": z[f"{site}__near"],
        "err": z[f"{site}__err"].astype(np.float64),
    }


def build_usgs_scored(split_fn):
    """Mirror E.build_merged_scored but on the USGS cache."""
    scored = {}
    for site, m in USGS_META.items():
        d = load_usgs_site(site)
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
    M.cusum_dev(scored)          # adds sc['dev'] (audited mean-shift CUSUM)
    CE.compute_ews_dev(scored)   # adds sc['dev_ews'], sc['dev_plus']
    return scored, controls, episodes


# ------------------------- NEON scored builder (for transfer thr) -------------------------
def build_neon_scored(split_fn):
    scored = {}
    for meta, cache in [(NEON_META0, NEON_CACHE0), (NEON_META1, NEON_CACHE1)]:
        for site, m in meta.items():
            z = np.load(cache, allow_pickle=True)
            d = {
                "end_ts": z[f"{site}__end_ts"],
                "cov": z[f"{site}__cov"],
                "near": z[f"{site}__near"],
                "err": z[f"{site}__err"].astype(np.float64),
            }
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
    M.cusum_dev(scored)
    CE.compute_ews_dev(scored)
    return scored, controls, episodes


def alarm_fn(scored, sites, thr):
    return {s: RC.cusum_alarms(scored[s]["dev_plus"], SLACK_K, thr) for s in sites}


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
    """Apply thr: control held-out FPR + episode detection/leads."""
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
    split_fn = M.cal_hold_split   # primary interleave split (matches headline)

    # ---- USGS scored (zero-shot cache) ----
    u_scored, u_controls, u_episodes = build_usgs_scored(split_fn)
    n_ep = len(u_episodes)
    print(f"USGS: {n_ep} episodes / {len(u_controls)} controls", flush=True)

    # ---- (a) USGS-SELF-CALIBRATED ----
    thr_self, self_cal_fpr = calibrate_thr(u_scored, u_controls, TARGET_FPR)
    self_res = evaluate(u_scored, u_controls, u_episodes, thr_self)
    print(f"[self-cal] thr={thr_self:.4f} ctrl-heldout-FPR={self_res['control_heldout_fpr']} "
          f"det={self_res['episodes_detected']}/{n_ep} lead={self_res['median_lead_days']}",
          flush=True)

    # ---- (b) NEON-TRANSFERRED threshold ----
    n_scored, n_controls, n_episodes = build_neon_scored(split_fn)
    thr_neon, neon_cal_fpr = calibrate_thr(n_scored, n_controls, TARGET_FPR)
    # sanity: reproduce NEON headline on NEON with this thr
    neon_self = evaluate(n_scored, n_controls, n_episodes, thr_neon)
    print(f"[NEON recomputed] thr={thr_neon:.4f} NEON-ctrl-heldout-FPR={neon_self['control_heldout_fpr']} "
          f"NEON det={neon_self['episodes_detected']}/{len(n_episodes)} "
          f"lead={neon_self['median_lead_days']}", flush=True)

    # apply the SAME NEON thr to USGS
    transfer_res = evaluate(u_scored, u_controls, u_episodes, thr_neon)
    print(f"[NEON->USGS] thr={thr_neon:.4f} USGS-ctrl-heldout-FPR={transfer_res['control_heldout_fpr']} "
          f"det={transfer_res['episodes_detected']}/{n_ep} lead={transfer_res['median_lead_days']}",
          flush=True)

    # ---- verdicts ----
    encoder_generalizes = bool(
        self_res["episodes_detected"] >= int(np.ceil(0.5 * n_ep))
        and self_res["control_heldout_fpr"] <= TARGET_FPR + 1e-9
        and self_res["median_lead_days"] is not None
        and self_res["median_lead_days"] > 0)
    operating_point_transfers = bool(
        transfer_res["control_heldout_fpr"] <= TARGET_FPR + 1e-2  # ~5%
        and transfer_res["episodes_detected"] > 0)

    result = {
        "purpose": "Cross-network zero-shot generalization: NEON-fine-tuned AquaSSM "
                   "encoder + CSD-EWS CUSUM detector applied ZERO-SHOT to USGS NWIS. "
                   "Two operating-point modes at 5% held-out control FPR.",
        "detector": {
            "checkpoint": "checkpoints/sensor/aquassm_neon_ft.pt (frozen; zero-shot on USGS)",
            "channel": "dev_plus = dev + BETA*dev_ews (audited mean-shift CUSUM + CSD-EWS)",
            "hyperparameters": {
                "EWS_WIN": CE.EWS_WIN, "EWS_MIN_BASE": CE.EWS_MIN_BASE, "BETA": CE.BETA,
                "W_BASE": CE.W_BASE, "GUARD": CE.GUARD, "SLACK_K": SLACK_K,
                "STRIDE": 8, "TARGET_FPR": TARGET_FPR,
                "note": "all fixed a priori from the NEON headline; nothing tuned on USGS leads",
            },
            "split": "interleave (M.cal_hold_split)",
        },
        "transfer_fairness_note": (
            "dev_plus is composed entirely of per-site strictly-causal rolling robust-z "
            "(median/MAD) of residuals and EWS features, so its scale is self-normalized "
            "and network-independent. The raw dev_plus threshold is therefore the fair "
            "operating-point-transfer statistic; no additional per-site rescaling exists "
            "to undo, so no separate self-normalized-score formulation is needed."),
        "n_usgs_episodes": n_ep,
        "n_usgs_controls": len(u_controls),
        "usgs_episode_sites": sorted(u_episodes),
        "usgs_control_sites": sorted(u_controls),
        "self_calibrated": self_res,
        "neon_transferred": {
            "neon_global_thr": round(float(thr_neon), 4),
            "neon_control_heldout_fpr_at_calib": round(float(neon_self["control_heldout_fpr"]), 4),
            "neon_self_detected": f"{neon_self['episodes_detected']}/{len(n_episodes)}",
            "neon_self_median_lead_days": neon_self["median_lead_days"],
            "applied_to_usgs": transfer_res,
        },
        "verdict": {
            "encoder_generalizes": encoder_generalizes,
            "operating_point_transfers": operating_point_transfers,
            "encoder_generalizes_criterion": ">=50% USGS episodes at <=5% held-out FPR with real lead (self-cal)",
            "operating_point_transfers_criterion": "NEON thr gives ~<=5% USGS held-out FPR with real detection",
        },
    }

    USGS_OUT.mkdir(parents=True, exist_ok=True)
    out = USGS_OUT / "cross_network_csd_ews.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"\nWROTE {out}")
    print(f"encoder_generalizes={encoder_generalizes}  "
          f"operating_point_transfers={operating_point_transfers}")


if __name__ == "__main__":
    main()
