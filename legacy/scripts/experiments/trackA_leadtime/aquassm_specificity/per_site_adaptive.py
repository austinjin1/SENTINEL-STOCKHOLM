#!/usr/bin/env python
"""Sub-goal 1: PER-SITE ADAPTIVE CUSUM operating point vs the GLOBAL-h CUSUM.

Global-h CUSUM (established Pareto win) uses ONE threshold h for every site,
calibrated on pooled CONTROL cal-clean windows. On non-stationary sites a single
h is a compromise: a noisy site over-fires while a quiet site under-fires.

PER-SITE ADAPTIVE idea (still label-free, no hypoxia read):
  For EACH site independently, calibrate its OWN CUSUM threshold h_site so that
  the alarm rate on THAT SITE's OWN cal-clean (QC-clean, non-hypoxia-near) dev
  windows equals a common target per-site clean FPR `q`. The operating point is
  thus set from each site's own baseline residual dynamics -- exactly the local
  self-adaptation the global h cannot do -- WITHOUT ever reading current hypoxia
  (calibration windows are strictly ~hypoxia-near-excluded, and for episode sites
  the pre-onset detection window is disjoint from onset).

MATCHED-FPR HEAD-TO-HEAD (the only fair comparison):
  We sweep the per-site target q and pick the q whose POOLED CONTROL HELD-OUT FPR
  lands at the SAME budget as the global-h operating point at each target level
  {3%,5%,7%}. Then we compare episodes-detected-pre-onset and median lead at that
  matched pooled held-out FPR. The per-site scheme only "wins" if, at <= the same
  pooled held-out control FPR, it validates >= as many episodes with >= lead.

FPR caveat fix: per-site calibration makes the single-contiguous-clean-run control
(MART) calibrate & evaluate on DISJOINT halves of its OWN run (temporal split),
so no window is reused in-sample. We report BOTH the interleave and the temporal
split; the temporal split removes the ~4.8%->optimistic-FPR caveat.

Guardrails (identical to the global-h run, all verified):
  (A) PRECURSOR: alarm counts only if its window ends strictly < DO<4 onset AND
      the window is NOT hypoxia-near (re-verified from RAW DO).
  (B) HELD-OUT FPR: calibrate on cal-clean, report FPR on DISJOINT hold-clean;
      controls' hypoxia never exists, episode calibration never touches onset.
  (C) REAL EPISODES: the 12 onsets are QC-clean DO<4 crossings, >=7 d clean runway.

Runs on CPU from the shared residual cache. No anomaly labels are used.
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cusum_vs_baseline_matchedFPR as M  # noqa: E402  shared audited scoring
import rolling_cusum as RC  # noqa: E402
from matched_fpr_sweep import cal_hold_split_temporal, build_scored, verify_precursor  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
OUTDIR = ROOT / "results" / "trackA_leadtime" / "aquassm_specificity"
TARGETS = [0.03, 0.05, 0.07]
# per-site target clean-FPR grid to sweep (each maps to a pooled held-out FPR)
Q_GRID = np.round(np.linspace(0.005, 0.15, 60), 4)
DAY = 86400


def med(x):
    return round(float(st.median(x)), 2) if x else None


def calibrate_h_site(dev, cal_mask, q):
    """Smallest h so this site's OWN cal-clean CUSUM alarm rate <= q.

    Uses the site's own clean dev distribution; label-free (cal_mask excludes
    hypoxia-near windows). If the site has too few cal windows, fall back to a
    high h (effectively no alarm) so it cannot dominate the pooled FPR.
    """
    d = dev[cal_mask]
    d = d[np.isfinite(d)]
    if d.size < 20:
        return float("inf")
    hi = np.percentile(d, 99.99)
    cand = np.unique(np.round(np.linspace(M.SLACK_K, max(hi, M.SLACK_K + 1), 300), 4))
    best = float(cand[-1])
    for h in cand:
        al = RC.cusum_alarms(dev, M.SLACK_K, h)
        rate = al[cal_mask].mean() if cal_mask.any() else 0.0
        if rate <= q:
            best = float(h)
            break
    return best


_H_CACHE: dict = {}


def adaptive_alarms(scored, sites, q):
    """Per-site alarms where each site uses its own h_site calibrated to q.

    h_site depends only on (site, q) and the site's cal-clean dev, so we memoize
    it (id(scored) namespaces the cache so different splits don't collide).
    """
    out, hmap = {}, {}
    ns = id(scored)
    for s in sites:
        key = (ns, s, round(float(q), 6))
        h = _H_CACHE.get(key)
        if h is None:
            h = calibrate_h_site(scored[s]["dev"], scored[s]["cal_clean"], q)
            _H_CACHE[key] = h
        hmap[s] = h
        out[s] = RC.cusum_alarms(scored[s]["dev"], M.SLACK_K, h)
    return out, hmap


def pooled_heldout_fpr(alarms, controls, scored):
    num = den = 0
    for s in controls:
        m = scored[s]["hold_clean"]
        num += int(alarms[s][m].sum())
        den += int(m.sum())
    return (num / den) if den else 0.0


def adaptive_sweep_q(scored, controls, target):
    """Pick per-site q whose pooled CONTROL held-out FPR is the largest <= target."""
    rows = []
    for q in Q_GRID:
        al_ctrl, _ = adaptive_alarms(scored, controls, q)
        fpr = pooled_heldout_fpr(al_ctrl, controls, scored)
        rows.append((float(q), fpr))
    ok = [(q, f) for q, f in rows if f <= target + 1e-9]
    if ok:
        q_best, f_best = max(ok, key=lambda r: r[1])
    else:
        q_best, f_best = min(rows, key=lambda r: r[1])
    return q_best, f_best, rows


def run_split(split_name, split_fn):
    scored, controls, episodes = build_scored(split_fn)
    n_ep = len(episodes)
    overlap = sum(int((scored[s]["cal_clean"] & scored[s]["hold_clean"]).sum()) > 0
                  for s in controls)

    # global-h grids (identical protocol to matched_fpr_sweep)
    c = np.concatenate([scored[s]["dev"][scored[s]["cal_clean"]] for s in controls])
    c = c[np.isfinite(c)]
    c_grid = np.unique(np.round(np.linspace(M.SLACK_K, np.percentile(c, 99.97), 300), 4))

    rows = []
    for tgt in TARGETS:
        # --- GLOBAL-h CUSUM at matched held-out FPR (the established detector) ---
        h_g, g_fpr = M.calibrate_to_heldout(scored, controls, tgt, c_grid, M.cusum_alarms)
        g_al = M.cusum_alarms(scored, episodes, h_g)
        g_ep, g_n, g_leads = M.episode_leads(scored, episodes, g_al)
        g_det = {s for s in episodes if g_ep[s]["detected_pre_onset"]}

        # --- PER-SITE ADAPTIVE at matched held-out FPR ---
        q, a_fpr, q_rows = adaptive_sweep_q(scored, controls, tgt)
        a_al, hmap_ep = adaptive_alarms(scored, episodes, q)
        _, hmap_ctrl = adaptive_alarms(scored, controls, q)
        a_ep, a_n, a_leads = M.episode_leads(scored, episodes, a_al)
        a_det = {s for s in episodes if a_ep[s]["detected_pre_onset"]}

        both_within = (g_fpr <= tgt + 1e-9) and (a_fpr <= tgt + 1e-9)
        g_med, a_med = med(g_leads), med(a_leads)
        # adaptive "wins" iff it detects >= as many at <= same FPR w/ >= lead & strictly better in one
        adaptive_better = bool(
            both_within and a_n >= g_n and (a_med or 0) >= (g_med or 0) - 1e-9
            and (a_n > g_n or (a_med or 0) > (g_med or 0)))
        tie = bool(both_within and a_n == g_n and abs((a_med or 0) - (g_med or 0)) < 1e-9)
        rows.append({
            "target_control_fpr": tgt,
            "global_h": {
                "h": round(h_g, 4),
                "pooled_control_heldout_fpr": round(g_fpr, 4),
                "episodes_detected": g_n, "of": n_ep,
                "median_lead_days": g_med,
                "detected_sites": sorted(g_det),
                "leads": {s: round(g_ep[s]["lead_days"], 2) for s in g_det},
            },
            "per_site_adaptive": {
                "per_site_target_q": round(q, 4),
                "pooled_control_heldout_fpr": round(a_fpr, 4),
                "episodes_detected": a_n, "of": n_ep,
                "median_lead_days": a_med,
                "detected_sites": sorted(a_det),
                "leads": {s: round(a_ep[s]["lead_days"], 2) for s in a_det},
                "h_site_episodes": {s: (round(hmap_ep[s], 3) if np.isfinite(hmap_ep[s]) else None)
                                    for s in episodes},
                "h_site_controls": {s: (round(hmap_ctrl[s], 3) if np.isfinite(hmap_ctrl[s]) else None)
                                    for s in controls},
            },
            "both_within_target_fpr": both_within,
            "adaptive_beats_global": adaptive_better,
            "adaptive_ties_global": tie,
            "adaptive_extra_episodes": sorted(a_det - g_det),
            "global_extra_episodes": sorted(g_det - a_det),
        })

    # precursor guardrail on the adaptive detector at the 5% point
    r5 = next(r for r in rows if abs(r["target_control_fpr"] - 0.05) < 1e-9)
    precursor = verify_precursor(scored, episodes, r5["per_site_adaptive"]["leads"])
    return {
        "split": split_name,
        "control_sites_with_cal_hold_overlap": overlap,
        "n_control_sites": len(controls), "n_episode_sites": n_ep,
        "sweep": rows,
        "precursor_check_adaptive_at_5pct": precursor,
    }


def main():
    result = {
        "purpose": "PER-SITE ADAPTIVE CUSUM operating point vs GLOBAL-h CUSUM at "
                   "matched pooled clean-control HELD-OUT FPR {3%,5%,7%} on the 12 "
                   "real NEON DO-hypoxia episodes; label-free reconstruction-residual.",
        "idea": "each site's CUSUM h calibrated to a common per-site clean-FPR q on "
                "its OWN cal-clean dev distribution; q swept to match the pooled "
                "held-out control FPR of the global-h operating point.",
        "guardrails": {
            "precursor": "alarm counts only if window ends strictly < DO<4 onset AND "
                         "window NOT hypoxia-near (re-verified from RAW DO)",
            "heldout_fpr": "per-site h calibrated on that site's cal-clean; FPR on "
                           "DISJOINT hold-clean; episode calibration never touches onset",
            "real_episodes": "12 QC-clean DO<4 onsets, each >=7 d clean runway",
            "label_free": True,
        },
        "targets": TARGETS, "q_grid": Q_GRID.tolist(),
        "results": {
            "primary_interleave_split": run_split("interleave", M.cal_hold_split),
            "robustness_temporal_split": run_split("temporal_single_run_fix",
                                                   cal_hold_split_temporal),
        },
    }
    prim = result["results"]["primary_interleave_split"]["sweep"]
    result["verdict"] = {
        "adaptive_beats_global_at_any_fpr": any(r["adaptive_beats_global"] for r in prim),
        "adaptive_ties_or_beats_global_at_all_fpr": all(
            r["adaptive_beats_global"] or r["adaptive_ties_global"] for r in prim),
        "headline_5pct": next(
            {"global_det": r["global_h"]["episodes_detected"],
             "global_lead": r["global_h"]["median_lead_days"],
             "global_fpr": r["global_h"]["pooled_control_heldout_fpr"],
             "adaptive_det": r["per_site_adaptive"]["episodes_detected"],
             "adaptive_lead": r["per_site_adaptive"]["median_lead_days"],
             "adaptive_fpr": r["per_site_adaptive"]["pooled_control_heldout_fpr"]}
            for r in prim if abs(r["target_control_fpr"] - 0.05) < 1e-9),
    }

    out = OUTDIR / "per_site_adaptive.json"
    out.write_text(json.dumps(result, indent=2))

    print("=== PER-SITE ADAPTIVE vs GLOBAL-h CUSUM (matched held-out FPR, interleave) ===")
    hdr = f"{'target':>7} | {'GLOBAL-h (det, fpr, lead)':>28} | {'PER-SITE (det, fpr, lead)':>28} | adapt_wins tie"
    print(hdr); print("-" * len(hdr))
    for r in prim:
        g, a = r["global_h"], r["per_site_adaptive"]
        gs = f"{g['episodes_detected']:>2}/12 fpr={g['pooled_control_heldout_fpr']:.3f} {g['median_lead_days']}d"
        as_ = f"{a['episodes_detected']:>2}/12 fpr={a['pooled_control_heldout_fpr']:.3f} {a['median_lead_days']}d"
        print(f"{r['target_control_fpr']:>7.0%} | {gs:>28} | {as_:>28} | {r['adaptive_beats_global']} {r['adaptive_ties_global']}")
    print(f"WROTE {out}")


if __name__ == "__main__":
    main()
