#!/usr/bin/env python
"""Sub-goal 2: EXPANDED real-event matched-FPR head-to-head.

Grows the real-event set with ADDITIONAL NEON DO-hypoxia episodes and clean
controls extracted by 06_extract_expanded.py (residuals in
residual_cache_expanded.npz), then re-runs the SAME rigorous matched-FPR
comparison used for the frozen 12-episode result:

  * CUSUM (global-h) vs baseline (global-tau), both calibrated to a common pooled
    clean-control HELD-OUT FPR in {3%,5%,7%};
  * PER-SITE ADAPTIVE CUSUM vs global-h CUSUM at matched held-out FPR (sub-goal 1
    re-checked on the larger set).

Everything reuses the audited scoring in cusum_vs_baseline_matchedFPR.py; the only
change is the site universe (original 12 ep + 10 ctrl  ->  +3 ep +7 ctrl). All
guardrails identical: label-free; precursor-only (re-verified from RAW DO);
held-out control FPR; real DO<4 onsets w/ >=7 d clean runway.

Runs on CPU from the two residual caches. No anomaly labels.
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cusum_vs_baseline_matchedFPR as M  # noqa: E402
import rolling_cusum as RC  # noqa: E402
from matched_fpr_sweep import cal_hold_split_temporal, verify_precursor  # noqa: E402
import per_site_adaptive as PSA  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
OUTDIR = ROOT / "results" / "trackA_leadtime" / "aquassm_specificity"
CACHE0 = OUTDIR / "residual_cache.npz"
CACHE1 = OUTDIR / "residual_cache_expanded.npz"
META0 = json.load(open(OUTDIR / "cache_meta.json"))
META1 = json.load(open(OUTDIR / "cache_meta_expanded.json"))
PROC = ROOT / "data" / "processed" / "trackA"
TARGETS = [0.03, 0.05, 0.07]
DAY = 86400


def load_site_from(cache_path, site):
    z = np.load(cache_path, allow_pickle=True)
    return {
        "end_ts": z[f"{site}__end_ts"],
        "cov": z[f"{site}__cov"],
        "near": z[f"{site}__near"],
        "err": z[f"{site}__err"].astype(np.float64),
    }


def build_merged_scored(split_fn):
    scored = {}
    for meta, cache in [(META0, CACHE0), (META1, CACHE1)]:
        for site, m in meta.items():
            d = load_site_from(cache, site)
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
    M.baseline_score(scored)
    M.cusum_dev(scored)
    return scored, controls, episodes


def med(x):
    return round(float(st.median(x)), 2) if x else None


def grids(scored, controls):
    b = np.concatenate([scored[s]["b_score"][scored[s]["cal_clean"]] for s in controls])
    b = b[np.isfinite(b)]
    b_grid = np.unique(np.percentile(b, np.linspace(50, 99.97, 300)))
    c = np.concatenate([scored[s]["dev"][scored[s]["cal_clean"]] for s in controls])
    c = c[np.isfinite(c)]
    c_grid = np.unique(np.round(np.linspace(M.SLACK_K, np.percentile(c, 99.97), 300), 4))
    return b_grid, c_grid


def run_split(split_name, split_fn):
    scored, controls, episodes = build_merged_scored(split_fn)
    n_ep = len(episodes)
    overlap = sum(int((scored[s]["cal_clean"] & scored[s]["hold_clean"]).sum()) > 0
                  for s in controls)
    b_grid, c_grid = grids(scored, controls)

    rows = []
    for tgt in TARGETS:
        # baseline vs global-h CUSUM at matched held-out FPR
        tau, b_fpr = M.calibrate_to_heldout(scored, controls, tgt, b_grid, M.baseline_alarms)
        h, c_fpr = M.calibrate_to_heldout(scored, controls, tgt, c_grid, M.cusum_alarms)
        b_al = M.baseline_alarms(scored, episodes, tau)
        c_al = M.cusum_alarms(scored, episodes, h)
        b_ep, b_n, b_leads = M.episode_leads(scored, episodes, b_al)
        c_ep, c_n, c_leads = M.episode_leads(scored, episodes, c_al)
        b_det = {s for s in episodes if b_ep[s]["detected_pre_onset"]}
        c_det = {s for s in episodes if c_ep[s]["detected_pre_onset"]}
        b_med, c_med = med(b_leads), med(c_leads)
        both_within = (b_fpr <= tgt + 1e-9) and (c_fpr <= tgt + 1e-9)
        pareto = bool(both_within and c_n >= b_n and (c_med or 0) >= (b_med or 0) - 1e-9
                      and (c_n > b_n or (c_med or 0) > (b_med or 0)))

        # per-site adaptive at matched held-out FPR
        q, a_fpr, _ = PSA.adaptive_sweep_q(scored, controls, tgt)
        a_al, hmap_ep = PSA.adaptive_alarms(scored, episodes, q)
        a_ep, a_n, a_leads = M.episode_leads(scored, episodes, a_al)
        a_det = {s for s in episodes if a_ep[s]["detected_pre_onset"]}
        a_med = med(a_leads)
        a_within = a_fpr <= tgt + 1e-9
        adapt_beats_global = bool(a_within and (c_fpr <= tgt + 1e-9) and a_n >= c_n
                                  and (a_med or 0) >= (c_med or 0) - 1e-9
                                  and (a_n > c_n or (a_med or 0) > (c_med or 0)))

        rows.append({
            "target_control_fpr": tgt,
            "baseline": {
                "global_tau": round(tau, 4), "pooled_control_heldout_fpr": round(b_fpr, 4),
                "episodes_detected": b_n, "of": n_ep, "median_lead_days": b_med,
                "detected_sites": sorted(b_det),
                "leads": {s: round(b_ep[s]["lead_days"], 2) for s in b_det},
            },
            "cusum_global_h": {
                "global_h": round(h, 4), "pooled_control_heldout_fpr": round(c_fpr, 4),
                "episodes_detected": c_n, "of": n_ep, "median_lead_days": c_med,
                "detected_sites": sorted(c_det),
                "leads": {s: round(c_ep[s]["lead_days"], 2) for s in c_det},
            },
            "cusum_per_site_adaptive": {
                "per_site_target_q": round(q, 4), "pooled_control_heldout_fpr": round(a_fpr, 4),
                "episodes_detected": a_n, "of": n_ep, "median_lead_days": a_med,
                "detected_sites": sorted(a_det),
                "leads": {s: round(a_ep[s]["lead_days"], 2) for s in a_det},
            },
            "both_within_target_fpr": both_within,
            "pareto_win_cusum_over_baseline": pareto,
            "adaptive_beats_global_cusum": adapt_beats_global,
            "cusum_extra_vs_baseline": sorted(c_det - b_det),
            "baseline_extra_vs_cusum": sorted(b_det - c_det),
            "new_episodes_detected_by_cusum": sorted(c_det & {"BLWA", "PRLA", "SUGG"}),
        })

    r5 = next(r for r in rows if abs(r["target_control_fpr"] - 0.05) < 1e-9)
    precursor = verify_precursor(scored, episodes, r5["cusum_global_h"]["leads"])
    return {
        "split": split_name,
        "control_sites_with_cal_hold_overlap": overlap,
        "n_control_sites": len(controls), "n_episode_sites": n_ep,
        "control_sites": sorted(controls), "episode_sites": sorted(episodes),
        "sweep": rows,
        "precursor_check_cusum_at_5pct": precursor,
    }


def main():
    new_ep = ["BLWA", "PRLA", "SUGG"]
    new_ctrl = ["POSE", "PRPO", "REDB", "TECR", "TOMB", "WALK", "WLOU"]
    result = {
        "purpose": "EXPANDED real-event matched-FPR head-to-head: original 12 DO-hypoxia "
                   "episodes + 3 NEW NEON episodes (BLWA,PRLA,SUGG); 10 + 7 NEW controls. "
                   "Re-runs CUSUM(global-h) & per-site-adaptive vs baseline at matched "
                   "pooled clean-control HELD-OUT FPR {3%,5%,7%}. Label-free residual.",
        "new_episodes": new_ep, "new_controls": new_ctrl,
        "guardrails": {
            "precursor": "alarm ends strictly < DO<4 onset AND window NOT hypoxia-near "
                         "(re-verified from RAW DO for CUSUM at 5%)",
            "heldout_fpr": "global thresholds calibrated on pooled CONTROL cal-clean; "
                           "FPR on DISJOINT pooled CONTROL hold-clean; episodes never calibrate",
            "real_episodes": "all onsets QC-clean DO<4 crossings, each >=7 d clean runway",
            "label_free": True,
        },
        "targets": TARGETS,
        "results": {
            "primary_interleave_split": run_split("interleave", M.cal_hold_split),
            "robustness_temporal_split": run_split("temporal_single_run_fix",
                                                   cal_hold_split_temporal),
        },
    }
    prim = result["results"]["primary_interleave_split"]["sweep"]
    pareto_all = all(r["pareto_win_cusum_over_baseline"] for r in prim)
    rob = result["results"]["robustness_temporal_split"]["sweep"]
    result["verdict"] = {
        "n_episodes": result["results"]["primary_interleave_split"]["n_episode_sites"],
        "n_controls": result["results"]["primary_interleave_split"]["n_control_sites"],
        "cusum_pareto_dominates_at_all_fpr_interleave": bool(pareto_all),
        "cusum_pareto_dominates_at_all_fpr_temporal": bool(
            all(r["pareto_win_cusum_over_baseline"] for r in rob)),
        "headline_5pct": next(
            {"baseline_det": r["baseline"]["episodes_detected"],
             "baseline_lead": r["baseline"]["median_lead_days"],
             "baseline_fpr": r["baseline"]["pooled_control_heldout_fpr"],
             "cusum_det": r["cusum_global_h"]["episodes_detected"],
             "cusum_lead": r["cusum_global_h"]["median_lead_days"],
             "cusum_fpr": r["cusum_global_h"]["pooled_control_heldout_fpr"],
             "adaptive_det": r["cusum_per_site_adaptive"]["episodes_detected"],
             "adaptive_lead": r["cusum_per_site_adaptive"]["median_lead_days"],
             "adaptive_fpr": r["cusum_per_site_adaptive"]["pooled_control_heldout_fpr"],
             "new_episodes_caught_by_cusum": r["new_episodes_detected_by_cusum"]}
            for r in prim if abs(r["target_control_fpr"] - 0.05) < 1e-9),
    }

    out = OUTDIR / "expanded_matched_fpr.json"
    out.write_text(json.dumps(result, indent=2))

    p = result["results"]["primary_interleave_split"]
    print(f"=== EXPANDED matched-FPR ({p['n_episode_sites']} episodes / {p['n_control_sites']} controls, interleave) ===")
    hdr = f"{'tgt':>5} | {'BASELINE':>22} | {'CUSUM global-h':>22} | {'CUSUM per-site':>22} | pareto"
    print(hdr); print("-" * len(hdr))
    for r in prim:
        b, c, a = r["baseline"], r["cusum_global_h"], r["cusum_per_site_adaptive"]
        bs = f"{b['episodes_detected']:>2}/{b['of']} fpr={b['pooled_control_heldout_fpr']:.3f} {b['median_lead_days']}d"
        cs = f"{c['episodes_detected']:>2}/{c['of']} fpr={c['pooled_control_heldout_fpr']:.3f} {c['median_lead_days']}d"
        as_ = f"{a['episodes_detected']:>2}/{a['of']} fpr={a['pooled_control_heldout_fpr']:.3f} {a['median_lead_days']}d"
        print(f"{r['target_control_fpr']:>5.0%} | {bs:>22} | {cs:>22} | {as_:>22} | {r['pareto_win_cusum_over_baseline']}")
    print(f"CUSUM Pareto over baseline at all FPR: interleave={pareto_all} "
          f"temporal={all(r['pareto_win_cusum_over_baseline'] for r in rob)}")
    print(f"WROTE {out}")


if __name__ == "__main__":
    main()
