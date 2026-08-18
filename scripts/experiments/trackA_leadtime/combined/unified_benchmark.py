#!/usr/bin/env python
"""SENTINEL Track A -- UNIFIED cross-network lead-time benchmark (Phase 1 consolidate).

Pools the FROZEN CSD-EWS detector (dev_plus = dev + BETA*dev_ews; audited mean-shift
CUSUM + verified critical-slowing-down EWS channel; ALL hyperparameters fixed a priori,
NEVER tuned on leads) across BOTH real sensor networks:

  * NEON  (15 DO-hypoxia episodes / 17 clean controls)  -- built via
    E.build_merged_scored(M.cal_hold_split) over residual_cache.npz + _expanded.npz
  * USGS NWIS (17 episodes / 22 controls, ZERO-SHOT second network) -- built by reusing
    the loader in usgs/detect_csd_ews_usgs.py over residual_cache_usgs.npz

Reports three rigorous consolidations, ALL at matched <=5% pooled-control HELD-OUT FPR:

  (2a) POOLED benchmark: pool 15+17=32 episodes; per-network self-calibrated thresholds
       (each network's controls set that network's operating point), combined detection
       count + MEDIAN LEAD with an EPISODE-level bootstrap 95% CI (resample the 32
       episodes' leads).  We ALSO report a SINGLE GLOBAL threshold calibrated on ALL
       39 controls (both networks pooled) at 5% held-out FPR, applied to all 32 episodes.

  (3) LEAVE-ONE-NETWORK-OUT generalization (the clean cross-network test):
       calibrate the threshold ONLY on NEON controls (5% held-out) -> apply to USGS
       episodes (report detection / lead / USGS-control held-out FPR); and the reverse
       (calibrate on USGS controls -> apply to NEON episodes). Both directions.

dev_plus is composed entirely of per-site strictly-causal rolling robust-z of residuals
and their EWS features, so its scale is self-normalized and network-independent -- a raw
dev_plus threshold denotes the same statistical operating point on either network, which
is what makes the pooled-global and leave-one-network-out thresholds fair.

Guardrails A-E respected: label-free detector; onset/near gate EVALUATION only; matched
CONTROL held-out FPR (episodes NEVER calibrate); fixed a-priori hyperparameters; honest
Pareto (report BOTH detection count and median lead everywhere). Runs on CPU from caches.
"""
from __future__ import annotations

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

import importlib.util as _ilu

import cusum_vs_baseline_matchedFPR as M  # noqa: E402
import expanded_matched_fpr as E  # noqa: E402
import rolling_cusum as RC  # noqa: E402

# candidate module file has a hyphen in its name -> load by path
_ce_path = SPEC_DIR / "candidates" / "cand_csd-ews-channel.py"
_spec = _ilu.spec_from_file_location("cand_csd_ews_channel", _ce_path)
CE = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(CE)

# USGS scored builder (reuse the audited loader)
_ug_path = USGS_DIR / "detect_csd_ews_usgs.py"
_ugspec = _ilu.spec_from_file_location("detect_csd_ews_usgs", _ug_path)
UG = _ilu.module_from_spec(_ugspec)
_ugspec.loader.exec_module(UG)

ROOT = Path(__file__).resolve().parents[4]
OUTDIR = ROOT / "results" / "trackA_leadtime" / "combined"

TARGET_FPR = 0.05
SLACK_K = CE.SLACK_K
SPLIT_FN = M.cal_hold_split          # primary interleave split (matches headline)
RNG_SEED = 12345
N_BOOT = 10000


def med(x):
    return round(float(st.median(x)), 4) if x else None


# ---- dev_plus alarm_fn compatible with M.calibrate_to_heldout / M.pooled_fpr ----
def alarm_fn(scored, sites, thr):
    return {s: RC.cusum_alarms(scored[s]["dev_plus"], SLACK_K, thr) for s in sites}


def calibrate_thr(scored, controls, target):
    """Largest dev_plus thr with pooled control HELD-OUT FPR <= target."""
    cal_scores = np.concatenate([scored[s]["dev_plus"][scored[s]["cal_clean"]]
                                 for s in controls])
    cal_scores = cal_scores[np.isfinite(cal_scores)]
    hi = np.percentile(cal_scores, 99.97) if cal_scores.size else (SLACK_K + 10)
    grid = np.unique(np.round(np.linspace(SLACK_K, max(hi, SLACK_K + 1), 300), 4))
    thr, fpr = M.calibrate_to_heldout(scored, controls, target, grid, alarm_fn)
    return float(thr), float(fpr)


def evaluate(scored, controls, episodes, thr):
    """Apply thr: pooled control held-out FPR + episode detection / leads (per-site)."""
    al_ctrl = alarm_fn(scored, controls, thr)
    ctrl_fpr = M.pooled_fpr(al_ctrl, "hold_clean", controls, scored)
    al_ep = alarm_fn(scored, episodes, thr)
    per_ep, n_det, leads = M.episode_leads(scored, episodes, al_ep)
    detected = sorted(s for s in episodes if per_ep[s]["detected_pre_onset"])
    leads_by_site = {s: round(per_ep[s]["lead_days"], 2) for s in detected}
    return {
        "global_thr": round(float(thr), 4),
        "control_heldout_fpr": round(float(ctrl_fpr), 4),
        "n_controls": len(controls),
        "episodes_detected": int(n_det),
        "n_episodes": len(episodes),
        "median_lead_days": med(leads),
        "detected_sites": detected,
        "leads_by_site": leads_by_site,
        "_leads": [float(x) for x in leads],   # internal, for bootstrap
    }


def boot_median_ci(leads, n_boot=N_BOOT, seed=RNG_SEED):
    """Episode-level bootstrap 95% CI on the median lead (resample episodes)."""
    if not leads:
        return {"median": None, "ci95_low": None, "ci95_high": None, "n": 0}
    rng = np.random.default_rng(seed)
    arr = np.asarray(leads, float)
    meds = np.empty(n_boot)
    n = arr.size
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        meds[b] = np.median(arr[idx])
    lo, hi = np.percentile(meds, [2.5, 97.5])
    return {"median": round(float(np.median(arr)), 4),
            "ci95_low": round(float(lo), 4), "ci95_high": round(float(hi), 4),
            "n": int(n)}


def prefix(scored, tag):
    """Return a renamed copy of scored so NEON/USGS site keys never collide when pooled."""
    return {f"{tag}:{s}": v for s, v in scored.items()}


def main():
    # ---------------- build both networks' scored dicts ----------------
    n_scored, n_controls, n_episodes = UG.build_neon_scored(SPLIT_FN)
    # (UG.build_neon_scored mirrors E.build_merged_scored + M.cusum_dev + CE.compute_ews_dev)
    u_scored, u_controls, u_episodes = UG.build_usgs_scored(SPLIT_FN)

    print(f"NEON: {len(n_episodes)} ep / {len(n_controls)} ctrl", flush=True)
    print(f"USGS: {len(u_episodes)} ep / {len(u_controls)} ctrl", flush=True)

    # ---------------- per-network self-calibrated operating points ----------------
    thr_neon, _ = calibrate_thr(n_scored, n_controls, TARGET_FPR)
    neon_self = evaluate(n_scored, n_controls, n_episodes, thr_neon)
    thr_usgs, _ = calibrate_thr(u_scored, u_controls, TARGET_FPR)
    usgs_self = evaluate(u_scored, u_controls, u_episodes, thr_usgs)

    print(f"[NEON self] thr={thr_neon:.4f} fpr={neon_self['control_heldout_fpr']} "
          f"det={neon_self['episodes_detected']}/{neon_self['n_episodes']} "
          f"lead={neon_self['median_lead_days']}", flush=True)
    print(f"[USGS self] thr={thr_usgs:.4f} fpr={usgs_self['control_heldout_fpr']} "
          f"det={usgs_self['episodes_detected']}/{usgs_self['n_episodes']} "
          f"lead={usgs_self['median_lead_days']}", flush=True)

    # ---------------- (2a) POOLED benchmark (per-network self-calibrated thresholds) ----------------
    pooled_leads = neon_self["_leads"] + usgs_self["_leads"]
    pooled_detected = neon_self["episodes_detected"] + usgs_self["episodes_detected"]
    pooled_n_ep = neon_self["n_episodes"] + usgs_self["n_episodes"]
    pooled_ci = boot_median_ci(pooled_leads)

    pooled_self = {
        "note": "each network's threshold self-calibrated on that network's controls at "
                "5% held-out FPR; episodes pooled for the combined lead distribution.",
        "neon_thr": round(thr_neon, 4), "usgs_thr": round(thr_usgs, 4),
        "combined_detected": pooled_detected, "combined_n_episodes": pooled_n_ep,
        "combined_detected_str": f"{pooled_detected}/{pooled_n_ep}",
        "neon_control_heldout_fpr": neon_self["control_heldout_fpr"],
        "usgs_control_heldout_fpr": usgs_self["control_heldout_fpr"],
        "combined_median_lead_days": pooled_ci["median"],
        "combined_median_lead_ci95": [pooled_ci["ci95_low"], pooled_ci["ci95_high"]],
        "bootstrap": {"n_boot": N_BOOT, "seed": RNG_SEED, "n_detected_episodes": pooled_ci["n"]},
    }

    # ---------------- (2b) SINGLE GLOBAL threshold on ALL controls (both networks) ----------------
    all_scored = {**prefix(n_scored, "NEON"), **prefix(u_scored, "USGS")}
    all_controls = [f"NEON:{s}" for s in n_controls] + [f"USGS:{s}" for s in u_controls]
    all_episodes = [f"NEON:{s}" for s in n_episodes] + [f"USGS:{s}" for s in u_episodes]

    thr_global, glob_cal_fpr = calibrate_thr(all_scored, all_controls, TARGET_FPR)
    glob = evaluate(all_scored, all_controls, all_episodes, thr_global)
    glob_ci = boot_median_ci(glob["_leads"])
    # per-network breakdown at the SAME global threshold
    glob_neon = evaluate(n_scored, n_controls, n_episodes, thr_global)
    glob_usgs = evaluate(u_scored, u_controls, u_episodes, thr_global)

    pooled_global = {
        "note": "ONE threshold calibrated on ALL 39 controls (NEON+USGS pooled) at 5% "
                "pooled held-out FPR, applied to all 32 episodes. dev_plus is "
                "self-normalized so a single threshold is a fair shared operating point.",
        "global_thr": round(thr_global, 4),
        "pooled_control_heldout_fpr": glob["control_heldout_fpr"],
        "combined_detected": glob["episodes_detected"],
        "combined_n_episodes": glob["n_episodes"],
        "combined_detected_str": f"{glob['episodes_detected']}/{glob['n_episodes']}",
        "combined_median_lead_days": glob_ci["median"],
        "combined_median_lead_ci95": [glob_ci["ci95_low"], glob_ci["ci95_high"]],
        "per_network": {
            "neon": {"detected": f"{glob_neon['episodes_detected']}/{glob_neon['n_episodes']}",
                     "median_lead_days": glob_neon["median_lead_days"],
                     "control_heldout_fpr": glob_neon["control_heldout_fpr"]},
            "usgs": {"detected": f"{glob_usgs['episodes_detected']}/{glob_usgs['n_episodes']}",
                     "median_lead_days": glob_usgs["median_lead_days"],
                     "control_heldout_fpr": glob_usgs["control_heldout_fpr"]},
        },
        "detected_sites": glob["detected_sites"],
    }
    print(f"[POOLED GLOBAL] thr={thr_global:.4f} pooled-ctrl-heldout-FPR={glob['control_heldout_fpr']} "
          f"det={glob['episodes_detected']}/{glob['n_episodes']} lead={glob_ci['median']}", flush=True)

    # ---------------- (3) LEAVE-ONE-NETWORK-OUT ----------------
    # calibrate on NEON controls -> apply to USGS episodes; report USGS-control held-out FPR
    neon2usgs = evaluate(u_scored, u_controls, u_episodes, thr_neon)
    # calibrate on USGS controls -> apply to NEON episodes; report NEON-control held-out FPR
    usgs2neon = evaluate(n_scored, n_controls, n_episodes, thr_usgs)

    loso = {
        "note": "cross-network generalization: threshold calibrated ONLY on the source "
                "network's controls (5% held-out) then applied to the OTHER network's "
                "episodes; held-out FPR reported on the TARGET network's controls.",
        "calibrate_NEON_apply_USGS": {
            "source_thr": round(thr_neon, 4),
            "target_control_heldout_fpr": neon2usgs["control_heldout_fpr"],
            "episodes_detected": f"{neon2usgs['episodes_detected']}/{neon2usgs['n_episodes']}",
            "median_lead_days": neon2usgs["median_lead_days"],
            "median_lead_ci95": list(boot_median_ci(neon2usgs["_leads"]).values())[1:3],
            "detected_sites": neon2usgs["detected_sites"],
        },
        "calibrate_USGS_apply_NEON": {
            "source_thr": round(thr_usgs, 4),
            "target_control_heldout_fpr": usgs2neon["control_heldout_fpr"],
            "episodes_detected": f"{usgs2neon['episodes_detected']}/{usgs2neon['n_episodes']}",
            "median_lead_days": usgs2neon["median_lead_days"],
            "median_lead_ci95": list(boot_median_ci(usgs2neon["_leads"]).values())[1:3],
            "detected_sites": usgs2neon["detected_sites"],
        },
    }
    print(f"[NEON->USGS] thr={thr_neon:.4f} USGS-ctrl-FPR={neon2usgs['control_heldout_fpr']} "
          f"det={neon2usgs['episodes_detected']}/{neon2usgs['n_episodes']} "
          f"lead={neon2usgs['median_lead_days']}", flush=True)
    print(f"[USGS->NEON] thr={thr_usgs:.4f} NEON-ctrl-FPR={usgs2neon['control_heldout_fpr']} "
          f"det={usgs2neon['episodes_detected']}/{usgs2neon['n_episodes']} "
          f"lead={usgs2neon['median_lead_days']}", flush=True)

    # strip internal keys
    for d in (neon_self, usgs_self, glob, glob_neon, glob_usgs, neon2usgs, usgs2neon):
        d.pop("_leads", None)

    result = {
        "purpose": "UNIFIED cross-network Track A lead-time benchmark: FROZEN CSD-EWS "
                   "detector (dev_plus = dev + BETA*dev_ews) pooled over NEON + USGS NWIS. "
                   "Pooled median lead with episode-bootstrap 95% CI, single global "
                   "threshold, and leave-one-network-out generalization. Matched <=5% "
                   "held-out control FPR everywhere; label-free; a-priori hyperparameters.",
        "detector": {
            "channel": "dev_plus = dev + BETA*dev_ews (audited mean-shift CUSUM + CSD-EWS)",
            "hyperparameters": {
                "W_BASE": CE.W_BASE, "GUARD": CE.GUARD, "SLACK_K": SLACK_K,
                "EWS_WIN": CE.EWS_WIN, "EWS_MIN_BASE": CE.EWS_MIN_BASE, "BETA": CE.BETA,
                "TARGET_FPR": TARGET_FPR, "split": "interleave (M.cal_hold_split)",
                "note": "all fixed a priori from the NEON headline; nothing tuned on leads",
            },
            "self_normalization_note": (
                "dev_plus = per-site strictly-causal rolling robust-z (median/MAD) of "
                "residuals and EWS features; scale is intrinsic to the detector, not the "
                "network -> a raw dev_plus threshold is a fair shared operating point."),
        },
        "networks": {
            "neon": {"n_episodes": neon_self["n_episodes"], "n_controls": len(n_controls),
                     "self_calibrated": {k: neon_self[k] for k in
                                         ("global_thr", "control_heldout_fpr",
                                          "episodes_detected", "n_episodes",
                                          "median_lead_days", "detected_sites",
                                          "leads_by_site")}},
            "usgs": {"n_episodes": usgs_self["n_episodes"], "n_controls": len(u_controls),
                     "self_calibrated": {k: usgs_self[k] for k in
                                         ("global_thr", "control_heldout_fpr",
                                          "episodes_detected", "n_episodes",
                                          "median_lead_days", "detected_sites",
                                          "leads_by_site")}},
        },
        "pooled_self_calibrated": pooled_self,
        "pooled_global_threshold": pooled_global,
        "leave_one_network_out": loso,
        "guardrails": {
            "label_free": True,
            "precursor_only": "alarm strictly pre-onset AND not hypoxia-near (M.episode_leads)",
            "matched_heldout_fpr": "thresholds on CONTROL clean windows; pooled control "
                                   "HELD-OUT FPR <=5%; episodes never calibrate",
            "fixed_hyperparameters": True,
            "honest_pareto": "report BOTH detection count and median lead everywhere",
        },
    }

    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / "unified_benchmark.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"\nWROTE {out}")


if __name__ == "__main__":
    main()
