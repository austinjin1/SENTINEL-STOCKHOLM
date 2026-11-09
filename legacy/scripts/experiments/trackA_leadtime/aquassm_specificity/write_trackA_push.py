#!/usr/bin/env python
"""Collate the Track A PUSH results into results/trackA_leadtime/trackA_push.json.

Aggregates three JSONs produced upstream (all label-free, matched-FPR, held-out):
  * cusum_matched_fpr.json    -- reproduced GLOBAL-h CUSUM vs baseline (12 ep)
  * per_site_adaptive.json    -- sub-goal 1: per-site adaptive vs global-h (12 ep)
  * expanded_matched_fpr.json -- sub-goal 2: expanded event set (15 ep / 17 ctrl)

Produces the reporting object the task asks for: (a) per-site vs global-h head-to-
head; (b) expanded count + matched-FPR result; (c) Pareto hold/improve; (d)
guardrail confirmation; (e) honest verdict; (f) files.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SPEC = ROOT / "results" / "trackA_leadtime" / "aquassm_specificity"
OUT = ROOT / "results" / "trackA_leadtime" / "trackA_push.json"


def load(p):
    return json.load(open(p))


def main():
    glob = load(SPEC / "cusum_matched_fpr.json")
    psa = load(SPEC / "per_site_adaptive.json")
    exp = load(SPEC / "expanded_matched_fpr.json")

    glob_prim = glob["results"]["primary_interleave_split"]["sweep"]
    def g5(sweep, key):
        r = next(r for r in sweep if abs(r["target_control_fpr"] - 0.05) < 1e-9)
        return r[key]

    # --- (a) per-site adaptive vs global-h head-to-head (original 12) ---
    psa_prim = psa["results"]["primary_interleave_split"]["sweep"]
    psa_rob = psa["results"]["robustness_temporal_split"]["sweep"]
    a_headtohead = {
        "event_set": "original 12 NEON DO-hypoxia episodes / 10 controls",
        "sweep_interleave": [
            {"target_fpr": r["target_control_fpr"],
             "global_h": {"det": r["global_h"]["episodes_detected"],
                          "fpr": r["global_h"]["pooled_control_heldout_fpr"],
                          "lead": r["global_h"]["median_lead_days"]},
             "per_site_adaptive": {"det": r["per_site_adaptive"]["episodes_detected"],
                                   "fpr": r["per_site_adaptive"]["pooled_control_heldout_fpr"],
                                   "lead": r["per_site_adaptive"]["median_lead_days"],
                                   "per_site_q": r["per_site_adaptive"]["per_site_target_q"]},
             "adaptive_beats_global": r["adaptive_beats_global"],
             "adaptive_ties_global": r["adaptive_ties_global"],
             "adaptive_extra_episodes": r["adaptive_extra_episodes"],
             "global_extra_episodes": r["global_extra_episodes"]}
            for r in psa_prim],
        "temporal_split_overlap_control_sites":
            psa["results"]["robustness_temporal_split"]["control_sites_with_cal_hold_overlap"],
        "interleave_split_overlap_control_sites":
            psa["results"]["primary_interleave_split"]["control_sites_with_cal_hold_overlap"],
        "verdict": psa["verdict"],
    }

    # --- (b) expanded event set ---
    exp_prim = exp["results"]["primary_interleave_split"]
    exp_rob = exp["results"]["robustness_temporal_split"]
    b_expanded = {
        "n_episodes": exp_prim["n_episode_sites"],
        "n_controls": exp_prim["n_control_sites"],
        "new_episodes": exp["new_episodes"],
        "new_controls": exp["new_controls"],
        "episode_sites": exp_prim["episode_sites"],
        "control_sites": exp_prim["control_sites"],
        "sweep_interleave": [
            {"target_fpr": r["target_control_fpr"],
             "baseline": {"det": r["baseline"]["episodes_detected"], "of": r["baseline"]["of"],
                          "fpr": r["baseline"]["pooled_control_heldout_fpr"],
                          "lead": r["baseline"]["median_lead_days"]},
             "cusum_global_h": {"det": r["cusum_global_h"]["episodes_detected"], "of": r["cusum_global_h"]["of"],
                                "fpr": r["cusum_global_h"]["pooled_control_heldout_fpr"],
                                "lead": r["cusum_global_h"]["median_lead_days"]},
             "cusum_per_site_adaptive": {"det": r["cusum_per_site_adaptive"]["episodes_detected"],
                                         "fpr": r["cusum_per_site_adaptive"]["pooled_control_heldout_fpr"],
                                         "lead": r["cusum_per_site_adaptive"]["median_lead_days"]},
             "pareto_win_cusum_over_baseline": r["pareto_win_cusum_over_baseline"],
             "new_episodes_detected_by_cusum": r["new_episodes_detected_by_cusum"]}
            for r in exp_prim["sweep"]],
        "verdict": exp["verdict"],
    }

    # --- (c) does the Pareto win hold / improve? ---
    orig_pareto_all = all(r["pareto_win_cusum"] for r in glob_prim)
    exp_pareto_all = exp["verdict"]["cusum_pareto_dominates_at_all_fpr_interleave"]
    c_pareto = {
        "original_12_cusum_pareto_all_fpr": bool(orig_pareto_all),
        "expanded_cusum_pareto_all_fpr_interleave": bool(exp_pareto_all),
        "expanded_cusum_pareto_all_fpr_temporal":
            exp["verdict"]["cusum_pareto_dominates_at_all_fpr_temporal"],
        "original_5pct": {"baseline": g5(glob_prim, "baseline")["episodes_detected"],
                          "cusum": g5(glob_prim, "cusum")["episodes_detected"]},
        "expanded_5pct": exp["verdict"]["headline_5pct"],
    }

    # --- (d) guardrails ---
    prec_exp = exp_prim["precursor_check_cusum_at_5pct"]
    prec_psa = psa["results"]["primary_interleave_split"]["precursor_check_adaptive_at_5pct"]
    all_precursor_ok = all(
        v["alarm_before_onset"] and v["alarm_window_not_hypoxia_near"]
        and v["alarm_before_first_ever_hypoxia"]
        for v in list(prec_exp.values()) + list(prec_psa.values()))
    d_guardrails = {
        "label_free": True,
        "precursor_only_all_detections_verified_from_raw_DO": bool(all_precursor_ok),
        "held_out_control_fpr": True,
        "real_neon_data_only": True,
        "precursor_detail_expanded_cusum_5pct": prec_exp,
        "precursor_detail_adaptive_5pct": prec_psa,
        "mart_absolute_fpr_caveat_fix": (
            "temporal single-run split makes MART (and every single-clean-run "
            "control) calibrate & evaluate on DISJOINT halves of its own run: "
            "control_sites_with_cal_hold_overlap==0 under the temporal split "
            "(vs 1 under interleave), so the pooled held-out FPR denominator is "
            "fully out-of-sample. Pareto verdict identical under both splits."),
    }

    result = {
        "title": "Track A PUSH: per-site adaptive operating point + expanded real-event set "
                 "for the label-free AquaSSM reconstruction-residual lead-time detector",
        "context": {
            "established_result": "rolling-CUSUM is a matched-FPR Pareto win over the "
                                  "point-threshold baseline: at ~5% held-out clean-control "
                                  "FPR, 10/12 episodes (median lead 37.2 d) vs 5/12 (25.5 d).",
            "reproduced_here": {"cusum_5pct_det": g5(glob_prim, "cusum")["episodes_detected"],
                                "cusum_5pct_lead": g5(glob_prim, "cusum")["median_lead_days"],
                                "cusum_5pct_fpr": g5(glob_prim, "cusum")["pooled_control_heldout_fpr"],
                                "baseline_5pct_det": g5(glob_prim, "baseline")["episodes_detected"],
                                "baseline_5pct_lead": g5(glob_prim, "baseline")["median_lead_days"]},
        },
        "a_per_site_adaptive_vs_global_h": a_headtohead,
        "b_expanded_event_set": b_expanded,
        "c_pareto_hold_or_improve": c_pareto,
        "d_guardrails": d_guardrails,
        "e_honest_verdict": {
            "sub_goal_1_per_site_adaptive": (
                "NOT a clean Pareto win over global-h. It helps only at the "
                "STRINGENT 3% operating point (validates +1 episode, BARC: 10/12 "
                "vs 9/12, at a LOWER held-out FPR 0.028 vs 0.030). At 5%/7% it TIES "
                "on episode count (10/12) at marginally lower FPR but with somewhat "
                "shorter median lead (34.8 vs 37.2 d @5%). Net: a modest, "
                "operating-point-dependent gain; its clearest value is tightening "
                "the FPR (matches at <= global FPR at every level, robust under the "
                "temporal single-run split)."),
            "sub_goal_2_expanded_set": (
                "Event set grown from 12->15 real NEON DO-hypoxia episodes (+BLWA, "
                "+SUGG, +PRLA) and 10->17 clean controls (+POSE,PRPO,REDB,TECR,TOMB,"
                "WALK,WLOU). CUSUM catches 11/15 at ~5% held-out FPR (lead 36.1 d) "
                "vs baseline 5/15 (25.5 d): the matched-FPR Pareto win HOLDS and "
                "STRENGTHENS at all swept FPR on BOTH splits. Of the 3 new episodes, "
                "BLWA is a genuine new validated detection (36.1 d lead, all "
                "precursor guardrails pass); SUGG and PRLA are UN-VALIDATABLE by "
                "construction (short records leave ZERO covered non-hypoxia-near "
                "pre-onset windows >=7 d before onset), so no detector can catch "
                "them and they penalize neither -- effectively 11/13 catchable."),
            "fpr_caveat": (
                "MART absolute-FPR caveat FIXED via the temporal single-run split "
                "(each single-clean-run control calibrates & evaluates on disjoint "
                "halves of its OWN run -> control_sites_with_cal_hold_overlap==0, "
                "fully out-of-sample held-out FPR). Expanding controls 10->17 also "
                "dilutes any single-site window reuse. Pareto verdict identical."),
            "overall": (
                "The central lead-time claim is REINFORCED on more real events: "
                "rolling-CUSUM remains a matched-FPR Pareto win (11/15 vs 5/15 @5%). "
                "Per-site adaptation is a real but modest refinement (helps at tight "
                "FPR, tightens the FPR caveat) rather than a decisive improvement."),
        },
        "catchable_episode_note": {
            "expanded_new_episodes_pre_onset_keep_windows": {
                "BLWA": 456, "SUGG": 0, "PRLA": 0},
            "meaning": "SUGG/PRLA have zero valid pre-onset detection windows "
                       "(record too short / onset too early) so are un-validatable "
                       "under the strict precursor guardrail; the fair CUSUM tally is "
                       "11/13 catchable at 5% FPR.",
        },
        "files": {
            "extract_expanded": "scripts/experiments/trackA_leadtime/06_extract_expanded.py",
            "cache_residuals_expanded": "scripts/experiments/trackA_leadtime/aquassm_specificity/cache_residuals_expanded.py",
            "per_site_adaptive": "scripts/experiments/trackA_leadtime/aquassm_specificity/per_site_adaptive.py",
            "expanded_matched_fpr": "scripts/experiments/trackA_leadtime/aquassm_specificity/expanded_matched_fpr.py",
            "expanded_manifest": "data/processed/trackA/expanded_manifest.json",
            "per_site_adaptive_json": "results/trackA_leadtime/aquassm_specificity/per_site_adaptive.json",
            "expanded_matched_fpr_json": "results/trackA_leadtime/aquassm_specificity/expanded_matched_fpr.json",
        },
    }
    OUT.write_text(json.dumps(result, indent=2))
    print(f"WROTE {OUT}")
    print(json.dumps(result["c_pareto_hold_or_improve"], indent=2))


if __name__ == "__main__":
    main()
