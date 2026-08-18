#!/usr/bin/env python
"""CANDIDATE: leaky-ewma-integrator -- leaky non-resetting EWMA accumulator.

Swaps the resetting one-sided CUSUM for a leaky, NON-resetting integrator:

    G_i = 0                          if dev_i is not finite (missing context)
        = max(0, LAMBDA*G_{i-1} + dev_i - SLACK_K)   otherwise
    alarm_i = (G_i >= thr)           # NO reset after an alarm

Rationale: the classic CUSUM resets S->0 the instant it fires, and its flat slack
makes dev exceed slack early yet the accumulator trips late. A leaky integrator
never resets on alarm, so early precursors keep accumulating; the leak (LAMBDA<1)
bounds the memory so a persistently-slightly-elevated clean control cannot ramp
without bound -> FPR stays controllable. Targets gradual episodes (COMO/LIRO/MAYF).

GUARDRAILS honored:
  (A) LABEL-FREE: LAMBDA=0.99, SLACK_K=1.0 fixed a priori; thr chosen ONLY to hit
      pooled CONTROL held-out FPR<=5%. onset/near gate EVALUATION only.
  (B) PRECURSOR: M.episode_leads enforces alarm ends strictly pre-onset & ~near.
  (C) MATCHED HELD-OUT FPR: M.calibrate_to_heldout on pooled control cal->hold.
  (D) NO TUNING ON LEADS. dev reuses the frozen CUSUM rolling baseline
      (w_base=240, guard=12, floor=1.0). We evaluate BOTH min_base=60 (headline
      pairing) and min_base=20 (spec "Pair min_base 20"), a priori choices.

Reuses the AUDITED protocol: cusum_vs_baseline_matchedFPR (M) + expanded_matched_fpr (E).
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
OUTDIR = ROOT / "results" / "trackA_leadtime" / "aquassm_specificity" / "candidates"
OUTDIR.mkdir(parents=True, exist_ok=True)

# --- a priori hyperparameters (NEVER tuned on leads) ---
LAMBDA = 0.99
SLACK_K = 1.0            # same slack as the frozen CUSUM
FLOOR = 1.0
W_BASE = 240
GUARD = 12
TARGET = 0.05


def recompute_dev(scored, min_base):
    """Recompute the aggregate deviation `dev` with a given rolling min_base.

    Overwrites scored[s]['dev'] in place (label-free; identical rolling baseline
    machinery as the frozen CUSUM, only min_base varies -- an a priori choice).
    """
    for s, sc in scored.items():
        res = RC.score_site(sc["err"], sc["clean"],
                            w_base=W_BASE, guard=GUARD, min_base=min_base,
                            floor=FLOOR, slack_k=SLACK_K, h=1e9)
        sc["dev"] = res["dev"]


def leaky_integrator(dev, slack_k, lam, thr):
    """Leaky non-resetting integrator alarms on an aggregate deviation signal."""
    n = len(dev)
    alarm = np.zeros(n, bool)
    G = 0.0
    for i in range(n):
        d = dev[i]
        if not np.isfinite(d):
            G = 0.0
            continue
        G = max(0.0, lam * G + d - slack_k)
        if G >= thr:
            alarm[i] = True
        # NO reset after alarm
    return alarm


def make_alarm_fn(lam):
    def alarm_fn(scored, sites, thr):
        return {s: leaky_integrator(scored[s]["dev"], SLACK_K, lam, thr)
                for s in sites}
    return alarm_fn


def control_cal_grid(scored, controls):
    """Ascending threshold grid from pooled control cal-clean integrator G values.

    We drive the integrator over each control's cal-clean-covered stream to see
    the reachable G distribution, then build a percentile grid. Grid choice is a
    label-free property of controls only.
    """
    gvals = []
    afn = make_alarm_fn(LAMBDA)  # unused; we replicate G directly below
    for s in controls:
        dev = scored[s]["dev"]
        cal = scored[s]["cal_clean"]
        G = 0.0
        gser = np.zeros(len(dev))
        for i in range(len(dev)):
            d = dev[i]
            if not np.isfinite(d):
                G = 0.0
            else:
                G = max(0.0, LAMBDA * G + d - SLACK_K)
            gser[i] = G
        gvals.append(gser[cal])
    g = np.concatenate(gvals)
    g = g[np.isfinite(g)]
    if g.size == 0:
        return np.array([SLACK_K, 10.0])
    lo = max(SLACK_K, float(np.percentile(g, 50)))
    hi = float(np.percentile(g, 99.99))
    grid = np.unique(np.round(np.linspace(lo, max(hi, lo + 1.0), 300), 4))
    return grid


def med(x):
    return round(float(st.median(x)), 3) if x else None


def evaluate(split_name, split_fn, min_base):
    scored, controls, episodes = E.build_merged_scored(split_fn)
    recompute_dev(scored, min_base)
    n_ep = len(episodes)

    alarm_fn = make_alarm_fn(LAMBDA)
    grid = control_cal_grid(scored, controls)
    thr, ctrl_fpr = M.calibrate_to_heldout(scored, controls, TARGET, grid, alarm_fn)

    al = alarm_fn(scored, episodes, thr)
    per_ep, n_det, leads = M.episode_leads(scored, episodes, al)
    det = sorted(s for s in episodes if per_ep[s]["detected_pre_onset"])
    leads_by_site = {s: round(per_ep[s]["lead_days"], 3) for s in det}
    m_lead = med(leads)

    pareto = bool(n_det >= 11 and (m_lead or 0) >= 36.1 - 1e-9 and ctrl_fpr <= TARGET + 1e-9)
    return {
        "split": split_name,
        "min_base": min_base,
        "lambda": LAMBDA,
        "slack_k": SLACK_K,
        "thr": round(thr, 4),
        "control_heldout_fpr": round(float(ctrl_fpr), 4),
        "n_episodes": n_ep,
        "episodes_detected": int(n_det),
        "median_lead_days": m_lead,
        "detected_sites": det,
        "leads_by_site": leads_by_site,
        "is_pareto_win": pareto,
    }


def main():
    results = []
    for min_base in (60, 20):
        for split_name, split_fn in (("interleave", M.cal_hold_split),
                                     ("temporal", cal_hold_split_temporal)):
            r = evaluate(split_name, split_fn, min_base)
            results.append(r)
            print(f"[{split_name:>10} mb={min_base:>2}] thr={r['thr']:>8.3f} "
                  f"ctrl-heldout-FPR={r['control_heldout_fpr']:.4f} "
                  f"detected={r['episodes_detected']}/{r['n_episodes']} "
                  f"median_lead={r['median_lead_days']} pareto={r['is_pareto_win']}")
            print(f"             detected={r['detected_sites']}")
            print(f"             leads={r['leads_by_site']}")

    # headline = interleave, min_base=60 (the frozen pairing)
    headline = next(r for r in results
                    if r["split"] == "interleave" and r["min_base"] == 60)
    temporal60 = next(r for r in results
                      if r["split"] == "temporal" and r["min_base"] == 60)

    out = {
        "candidate": "leaky-ewma-integrator",
        "name": "Leaky non-resetting EWMA accumulator",
        "idea": "G=max(0, LAMBDA*G_prev + dev - SLACK_K); alarm G>=thr; no reset on "
                "alarm, reset only on NaN dev. Early precursors accumulate; leak bounds "
                "memory for FPR control.",
        "hyperparameters_a_priori": {"lambda": LAMBDA, "slack_k": SLACK_K,
                                     "floor": FLOOR, "w_base": W_BASE, "guard": GUARD},
        "target_control_heldout_fpr": TARGET,
        "baseline_to_beat": {"episodes_detected": 11, "median_lead_days": 36.1,
                             "at_control_heldout_fpr": 0.05},
        "headline_interleave_min_base_60": headline,
        "robustness_temporal_min_base_60": temporal60,
        "all_runs": results,
        "verdict": {
            "is_pareto_win_headline": headline["is_pareto_win"],
            "is_pareto_win_temporal": temporal60["is_pareto_win"],
            "holds_under_temporal": bool(headline["is_pareto_win"]
                                         and temporal60["is_pareto_win"]),
        },
    }
    outpath = OUTDIR / "leaky-ewma-integrator.json"
    outpath.write_text(json.dumps(out, indent=2))
    print(f"\nWROTE {outpath}")
    print(f"HEADLINE (interleave, min_base=60): detected={headline['episodes_detected']}"
          f"/{headline['n_episodes']} median_lead={headline['median_lead_days']} "
          f"ctrl-FPR={headline['control_heldout_fpr']} pareto={headline['is_pareto_win']}")


if __name__ == "__main__":
    main()
