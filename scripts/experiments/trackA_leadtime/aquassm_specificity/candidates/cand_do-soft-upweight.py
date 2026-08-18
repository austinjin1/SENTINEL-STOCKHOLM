#!/usr/bin/env python
"""CANDIDATE: do-soft-upweight -- DO soft up-weight.

IDEA: A mild FIXED up-weight on the DO positive-part deviation trips the CUSUM
accumulator earlier for DO-dominated episodes without regressing turbidity- or
conductivity-dominated ones.

SPEC (all hyperparameters FIXED A PRIORI -- never tuned on leads):
  Per-param causal robust z is identical to the audited headline (rolling_cusum).
  The aggregate deviation is a WEIGHTED sum of positive parts:
      dev_i = sum_p  W_p * max(0, z_{i,p} - floor)
  with fixed physical-prior weights (err column order = pH,DO,turbidity,conductivity):
      W = [1.0(pH), 1.5(DO), 1.0(turb), 1.0(cond)]     floor = 1.0
  DO (col 1) is up-weighted 1.5x; everything else stays 1.0 so multivariate
  (turbidity/conductivity dominated) episodes are unaffected.
  Rolling-baseline hyperparams: w_base=240, guard=12, min_base=20 (paired per SPEC),
  slack_k=1.0. The single global threshold h is calibrated via
  M.calibrate_to_heldout so pooled CONTROL HELD-OUT FPR <= 5%. Episodes never
  touch calibration.

GUARDRAILS: label-free (weights are a fixed prior, only h fits control FPR),
precursor-only + matched held-out FPR enforced by the audited M.episode_leads /
M.calibrate_to_heldout. Reports BOTH episodes-detected and median lead.
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

ROOT = SPEC_DIR.resolve().parents[3]
OUTDIR = ROOT / "results" / "trackA_leadtime" / "aquassm_specificity"

# ---- FIXED A PRIORI hyperparameters (NEVER tuned on leads) ----
W_BASE = 240
GUARD = 12
MIN_BASE = 20            # paired per SPEC
FLOOR = 1.0
SLACK_K = M.SLACK_K      # 1.0
# err column order: pH, DO, turbidity, conductivity
WEIGHTS = np.array([1.0, 1.5, 1.0, 1.0], dtype=np.float64)
TARGET = 0.05


def weighted_dev(z: np.ndarray, weights: np.ndarray, floor: float) -> np.ndarray:
    """Weighted sum of positive parts across params: sum_p W_p max(0, z_p - floor)."""
    pos = np.clip(z - floor, 0.0, None)
    any_valid = np.isfinite(z).any(axis=1)
    agg = np.nansum(pos * weights[None, :], axis=1)
    agg[~any_valid] = np.nan
    return agg


def precompute_dev_do(scored):
    """Recompute per-site DO-up-weighted dev in place (key 'dev_do')."""
    for s, sc in scored.items():
        err = sc["err"]
        n, p = err.shape
        z = np.full((n, p), np.nan, np.float64)
        for j in range(p):
            z[:, j] = RC.rolling_causal_z(err[:, j], sc["clean"],
                                          W_BASE, GUARD, MIN_BASE)
        sc["dev_do"] = weighted_dev(z, WEIGHTS, FLOOR)


def alarm_fn(scored, sites, thr):
    """alarm_fn(scored, sites, thr) -> {site: bool alarm[N]} using DO-weighted dev."""
    return {s: RC.cusum_alarms(scored[s]["dev_do"], SLACK_K, thr) for s in sites}


def build_grid(scored, controls):
    dev = np.concatenate([scored[s]["dev_do"][scored[s]["cal_clean"]]
                          for s in controls])
    dev = dev[np.isfinite(dev)]
    hi = np.percentile(dev, 99.97) if dev.size else (SLACK_K + 1.0)
    return np.unique(np.round(np.linspace(SLACK_K, max(hi, SLACK_K + 1.0), 300), 4))


def med(x):
    return round(float(st.median(x)), 4) if x else None


def run_split(split_name, split_fn):
    scored, controls, episodes = E.build_merged_scored(split_fn)
    precompute_dev_do(scored)
    n_ep = len(episodes)
    grid = build_grid(scored, controls)

    thr, ctrl_fpr = M.calibrate_to_heldout(scored, controls, TARGET, grid, alarm_fn)
    al = alarm_fn(scored, episodes, thr)
    per_ep, n_det, leads = M.episode_leads(scored, episodes, al)
    detected = sorted(s for s in episodes if per_ep[s]["detected_pre_onset"])
    leads_by_site = {s: round(per_ep[s]["lead_days"], 4) for s in detected}
    median_lead = med(leads)

    return {
        "split": split_name,
        "n_episode_sites": n_ep,
        "n_control_sites": len(controls),
        "global_h": round(float(thr), 4),
        "control_heldout_fpr": round(float(ctrl_fpr), 4),
        "episodes_detected": int(n_det),
        "median_lead_days": median_lead,
        "detected_sites": detected,
        "leads_by_site": leads_by_site,
    }


def main():
    primary = run_split("interleave", M.cal_hold_split)
    temporal = run_split("temporal", cal_hold_split_temporal)

    is_pareto = bool(primary["episodes_detected"] >= 11
                     and (primary["median_lead_days"] or 0) >= 36.1
                     and primary["control_heldout_fpr"] <= 0.05 + 1e-9)

    result = {
        "candidate_key": "do-soft-upweight",
        "candidate_name": "DO soft up-weight",
        "spec": {
            "weights_pH_DO_turb_cond": WEIGHTS.tolist(),
            "floor": FLOOR, "w_base": W_BASE, "guard": GUARD,
            "min_base": MIN_BASE, "slack_k": SLACK_K, "target_fpr": TARGET,
        },
        "headline_bar": {"episodes_detected>=": 11, "median_lead_days>=": 36.1,
                         "control_heldout_fpr<=": 0.05},
        "primary_interleave": primary,
        "robustness_temporal": temporal,
        "is_pareto_win": is_pareto,
    }

    out = OUTDIR / "candidates" / "do-soft-upweight.json"
    out.write_text(json.dumps(result, indent=2))

    for r in (primary, temporal):
        print(f"=== {r['split']} ({r['n_episode_sites']} ep / {r['n_control_sites']} ctrl) ===")
        print(f"  h={r['global_h']}  ctrl-heldout-FPR={r['control_heldout_fpr']}")
        print(f"  detected={r['episodes_detected']}/{r['n_episode_sites']}  "
              f"median_lead={r['median_lead_days']}d")
        print(f"  detected_sites={r['detected_sites']}")
        print(f"  leads={r['leads_by_site']}")
    print(f"IS_PARETO_WIN (>=11 det, >=36.1d, fpr<=5%): {is_pareto}")
    print(f"WROTE {out}")


if __name__ == "__main__":
    main()
