#!/usr/bin/env python
"""CANDIDATE (ORACLE ONLY): LOEO matched-filter upper bound on lead time.

>>> THIS IS AN ORACLE. IT IS DISQUALIFIED FROM THE HEADLINE. <<<
It deliberately violates GUARDRAIL A (label-free): the precursor TEMPLATE is built
from other episodes' deviation signals ALIGNED TO THEIR DO<4 ONSET. That alignment
leaks onset-relative timing and precursor SHAPE into the detector. We build it only
to UPPER-BOUND what any shape-aware matched filter could achieve on this cache, and
we report it under a clearly-labeled "oracle" banner. It must never be promoted to a
real detector or the paper headline.

Construction (all hyperparameters fixed a priori; NEVER tuned on leads):
  * Per-site base signal = the audited CUSUM aggregate deviation dev[N] (RC.aggregate
    _deviation of the strictly-causal rolling z). This part is label-free and reused
    verbatim (W_BASE=240, GUARD=12, MIN_BASE=60, FLOOR=1.0, SLACK_K=1.0).
  * TEMPLATE (the oracle leak): for each episode, take the dev values in the
    LOOKBACK_DAYS-day pre-onset window (covered, ~hypoxia-near), resampled onto a
    fixed grid of TEMPLATE_LEN windows ending AT onset. Average these across all
    episodes EXCEPT the held-out one e (LEAVE-ONE-EPISODE-OUT, mandatory). Mean-
    center + unit-normalize -> a matched filter template t_e.
  * SCORE at window i (strictly causal): normalized cross-correlation of t_e against
    the trailing TEMPLATE_LEN dev values [i-TEMPLATE_LEN+1 .. i]. This is a causal
    matched-filter response; large where the recent dev trajectory matches the
    learned precursor shape.
  * ALARM: score >= thr. ONE GLOBAL thr calibrated so pooled CONTROL HELD-OUT FPR
    <= 5% (controls have NO onset, so each control is scored with a template built
    from ALL episodes -- controls never leave one out; that only inflates control
    firing, i.e. it is a CONSERVATIVE / harder specificity bar for the oracle).

Evaluation reuses the AUDITED protocol verbatim: M.calibrate_to_heldout +
M.episode_leads (precursor-only, strictly pre-onset, ~hypoxia-near). Reports BOTH
episodes-detected and median lead. Also runs the temporal-split robustness check.
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SPEC = HERE.parent
sys.path.insert(0, str(SPEC))
import cusum_vs_baseline_matchedFPR as M  # noqa: E402
import expanded_matched_fpr as E  # noqa: E402
from matched_fpr_sweep import cal_hold_split_temporal  # noqa: E402

ROOT = Path(__file__).resolve().parents[5]
OUTDIR = ROOT / "results" / "trackA_leadtime" / "aquassm_specificity" / "candidates"
OUTDIR.mkdir(parents=True, exist_ok=True)

DAY = 86400
TARGET_FPR = 0.05
LOOKBACK_DAYS = 45          # matches M.LOOKBACK_DAYS
# --- matched-filter hyperparameters, FIXED A PRIORI (never tuned on leads) ---
TEMPLATE_LEN = 60           # windows in the onset-aligned precursor template
MIN_TEMPLATE_SRC = 20       # min covered pre-onset windows for an episode to donate


def onset_aligned_template(sc, template_len):
    """Resample an episode's pre-onset dev trajectory onto a fixed onset-aligned grid.

    ORACLE STEP: uses onset_ts to align. Returns a length-`template_len` vector whose
    last entry sits at onset and first at onset-LOOKBACK, or None if too sparse.
    """
    onset = sc["onset_ts"]
    et = sc["end_ts"]
    dev = sc["dev"]
    keep = (et >= onset - LOOKBACK_DAYS * DAY) & (et < onset)
    keep &= sc["cov_ok"] & (~sc["near"]) & np.isfinite(dev)
    idx = np.where(keep)[0]
    if idx.size < MIN_TEMPLATE_SRC:
        return None
    # time-fraction of each covered pre-onset window in [0,1) (0=lookback start, ->1 onset)
    frac = (et[idx] - (onset - LOOKBACK_DAYS * DAY)) / (LOOKBACK_DAYS * DAY)
    grid = (np.arange(template_len) + 0.5) / template_len
    tmpl = np.interp(grid, frac, dev[idx])
    return tmpl


def _normalize(v):
    v = v - v.mean()
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else None


def build_templates(scored, episodes, template_len):
    """Per-episode onset-aligned dev templates (raw, un-normalized)."""
    raw = {}
    for s in episodes:
        t = onset_aligned_template(scored[s], template_len)
        if t is not None:
            raw[s] = t
    return raw


def loeo_template_for(raw, hold_site):
    """LEAVE-ONE-EPISODE-OUT averaged + normalized template for `hold_site`.

    Averages every donor template except hold_site's own. For control sites
    (hold_site not in raw) this averages ALL donors -> a harder specificity bar.
    """
    donors = [t for s, t in raw.items() if s != hold_site]
    if not donors:
        return None
    return _normalize(np.mean(donors, axis=0))


def matched_filter_score(dev, tmpl):
    """Strictly-causal normalized cross-correlation of tmpl against trailing dev.

    For window i: correlate tmpl (length L) with dev[i-L+1 .. i]. Windows with any
    non-finite / <MIN_BASE-context dev in the trailing block score -inf (no alarm).
    """
    n = len(dev)
    L = len(tmpl)
    out = np.full(n, -np.inf, np.float64)
    for i in range(L - 1, n):
        seg = dev[i - L + 1:i + 1]
        if not np.all(np.isfinite(seg)):
            continue
        seg = seg - seg.mean()
        sn = np.linalg.norm(seg)
        if sn < 1e-9:
            continue
        out[i] = float(np.dot(tmpl, seg / sn))
    return out


def make_alarm_fn(raw, template_len):
    """Return an alarm_fn(scored, sites, thr)->{site:bool[]} using LOEO templates."""
    cache = {}

    def score_for(site, sc):
        if site in cache:
            return cache[site]
        tmpl = loeo_template_for(raw, site)
        sco = (matched_filter_score(sc["dev"], tmpl) if tmpl is not None
               else np.full(len(sc["dev"]), -np.inf))
        cache[site] = sco
        return sco

    def alarm_fn(scored, sites, thr):
        out = {}
        for s in sites:
            sco = score_for(s, scored[s])
            out[s] = sco >= thr
        return out

    return alarm_fn


def score_grid(scored, controls, raw, template_len, alarm_fn):
    """Ascending threshold grid from pooled control CAL-clean matched-filter scores."""
    vals = []
    for s in controls:
        tmpl = loeo_template_for(raw, s)
        if tmpl is None:
            continue
        sco = matched_filter_score(scored[s]["dev"], tmpl)
        m = scored[s]["cal_clean"] & np.isfinite(sco)
        vals.append(sco[m])
    if not vals:
        return np.linspace(-1, 1, 200)
    v = np.concatenate(vals)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.linspace(-1, 1, 200)
    grid = np.unique(np.percentile(v, np.linspace(1, 99.9, 300)))
    # extend upward a bit so a fully-specific threshold exists
    grid = np.append(grid, [v.max() + 1e-6, 1.0 + 1e-6])
    return np.unique(grid)


def med(x):
    return round(float(st.median(x)), 2) if x else None


def run_split(split_name, split_fn):
    scored, controls, episodes = E.build_merged_scored(split_fn)
    n_ep = len(episodes)

    raw = build_templates(scored, episodes, TEMPLATE_LEN)
    donors = sorted(raw.keys())
    alarm_fn = make_alarm_fn(raw, TEMPLATE_LEN)

    grid = score_grid(scored, controls, raw, TEMPLATE_LEN, alarm_fn)
    thr, ctrl_fpr = M.calibrate_to_heldout(scored, controls, TARGET_FPR, grid, alarm_fn)

    al = alarm_fn(scored, episodes, thr)
    per_ep, n_det, leads = M.episode_leads(scored, episodes, al)
    det = sorted(s for s in episodes if per_ep[s]["detected_pre_onset"])
    leads_by_site = {s: round(per_ep[s]["lead_days"], 2) for s in det}
    m_lead = med(leads)

    row = {
        "split": split_name,
        "n_episode_sites": n_ep,
        "n_control_sites": len(controls),
        "n_template_donors": len(donors),
        "template_donors": donors,
        "global_thr": round(float(thr), 6),
        "control_heldout_fpr": round(float(ctrl_fpr), 4),
        "episodes_detected": int(n_det),
        "median_lead_days": m_lead,
        "detected_sites": det,
        "leads_by_site": leads_by_site,
    }
    return row


def main():
    prim = run_split("interleave", M.cal_hold_split)
    temp = run_split("temporal_single_run_fix", cal_hold_split_temporal)

    is_pareto = bool(prim["episodes_detected"] >= 11
                     and (prim["median_lead_days"] or 0) >= 36.1
                     and prim["control_heldout_fpr"] <= 0.05)

    result = {
        "candidate": "loeo-matched-filter-oracle",
        "ORACLE": True,
        "disqualified_from_headline": True,
        "guardrail_A_violation": "Template built from other episodes' dev signals "
                                 "ALIGNED TO DO<4 onset -> leaks onset-relative timing "
                                 "and precursor shape. Upper bound only.",
        "hyperparameters_fixed_a_priori": {
            "W_BASE": M.W_BASE, "GUARD": M.GUARD, "MIN_BASE": M.MIN_BASE,
            "FLOOR": M.FLOOR, "SLACK_K": M.SLACK_K,
            "TEMPLATE_LEN": TEMPLATE_LEN, "MIN_TEMPLATE_SRC": MIN_TEMPLATE_SRC,
            "LOOKBACK_DAYS": LOOKBACK_DAYS, "TARGET_FPR": TARGET_FPR,
        },
        "leave_one_episode_out": "mandatory (template for held-out e excludes e)",
        "headline_to_beat": {"episodes_detected": 11, "median_lead_days": 36.1,
                             "control_heldout_fpr": 0.05},
        "primary_interleave_split": prim,
        "robustness_temporal_split": temp,
        "is_pareto_win": is_pareto,
    }

    out = OUTDIR / "loeo-matched-filter-oracle.json"
    out.write_text(json.dumps(result, indent=2))

    print("=== LOEO MATCHED-FILTER **ORACLE** (DISQUALIFIED FROM HEADLINE) ===")
    for r in (prim, temp):
        print(f"[{r['split']}] donors={r['n_template_donors']}/{r['n_episode_sites']} "
              f"thr={r['global_thr']} ctrl_heldout_fpr={r['control_heldout_fpr']} "
              f"detected={r['episodes_detected']}/{r['n_episode_sites']} "
              f"median_lead={r['median_lead_days']}d")
        print(f"    detected_sites={r['detected_sites']}")
        print(f"    leads={r['leads_by_site']}")
    print(f"is_pareto_win (oracle, informational only)={is_pareto}")
    print(f"WROTE {out}")


if __name__ == "__main__":
    main()
