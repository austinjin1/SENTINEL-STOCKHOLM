#!/usr/bin/env python
"""Evaluate the causal rolling-CUSUM detector vs the baseline, honestly.

Protocol (leakage-free):
  * Calibrate a SINGLE GLOBAL CUSUM threshold h on CONTROL sites only, to the
    largest h achieving pooled control held-out-clean FPR <= target. Episode
    sites never contribute to h -> episodes are a true held-out test of both
    specificity and lead.
  * FPR is measured on CLEAN windows only: QC-covered (cov>=MIN_COV) AND not
    hypoxia-near (~near). This matches the baseline's clean-control FPR notion.
    We report per-site held-out FPR and the pooled control FPR.
  * Lead time on episode sites: first alarm strictly BEFORE onset AND in a
    non-hypoxia-near window (~near) -- so the alarm cannot be reading
    contemporaneous DO<4. lead_days = (onset - t_alarm)/day. This is STRICTER
    than the baseline (which allowed near windows up to onset).

For an apples-to-apples FPR comparison we split each site's clean windows into a
calibration half and a held-out half by INTERLEAVING contiguous clean runs (same
spirit as the baseline). h is chosen on the pooled control calibration halves;
FPR is reported on the held-out halves of every site.

No anomaly labels are used. `near` derives only from the DO<4 event definition
and is used solely to exclude hypoxia-adjacent windows from clean evaluation and
to enforce the pre-onset lead guardrail -- never fed into the score.
"""
from __future__ import annotations

import json
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

MIN_COV = 0.30
DAY = 86400
TARGET_FPR = 0.05
LOOKBACK_DAYS = 45

# --- detector hyperparameters (set once, documented; NOT tuned on episode leads) ---
W_BASE = 240        # trailing baseline window count (~20 days at 2 h stride)
GUARD = 12          # guard gap (~24 h) so the incipient shift can't seed its own baseline
MIN_BASE = 60       # need >=60 clean baseline windows to standardize
FLOOR = 1.0         # per-param z must exceed +1 robust-sigma to contribute
SLACK_K = 1.0       # CUSUM slack: ignore aggregate deviation below this


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
        "ends": z[f"{site}__ends"],
        "end_ts": z[f"{site}__end_ts"],
        "cov": z[f"{site}__cov"],
        "near": z[f"{site}__near"],
        "err": z[f"{site}__err"],
    }


def site_masks(d):
    cov_ok = d["cov"] >= MIN_COV
    clean = cov_ok & (~d["near"])           # windows usable for FPR / baseline
    return cov_ok, clean


def calibration_holdout_split(clean):
    """Interleave contiguous clean runs into calibration vs held-out clean sets.

    Returns two boolean masks over the full window index (cal_clean, hold_clean).
    """
    runs = contiguous_runs(clean)
    runs = [r for r in runs if r[1] - r[0] >= 1]
    cal = np.zeros(len(clean), bool)
    hold = np.zeros(len(clean), bool)
    for k, (lo, hi) in enumerate(runs):
        (cal if k % 2 == 0 else hold)[lo:hi] = True
    if not hold.any():          # single run -> reuse
        hold = cal.copy()
    return cal, hold


def fpr_on(alarm, mask):
    m = mask
    return float(alarm[m].mean()) if m.any() else 0.0


def evaluate(h, scored):
    """Given global h, recompute alarms per site from cached dev and report FPR."""
    per_site = {}
    ctrl_hold_num = ctrl_hold_den = 0
    for site, sc in scored.items():
        alarm = RC.cusum_alarms(sc["dev"], SLACK_K, h)
        hold = sc["hold_clean"]
        cal = sc["cal_clean"]
        fpr_h = fpr_on(alarm, hold)
        fpr_c = fpr_on(alarm, cal)
        per_site[site] = {"fpr_heldout": fpr_h, "fpr_calib": fpr_c,
                          "n_hold": int(hold.sum()), "n_cal": int(cal.sum()),
                          "role": sc["role"]}
        if sc["role"] == "control":
            ctrl_hold_num += alarm[hold].sum()
            ctrl_hold_den += hold.sum()
    pooled_ctrl_fpr = float(ctrl_hold_num / ctrl_hold_den) if ctrl_hold_den else 0.0
    return per_site, pooled_ctrl_fpr


def main():
    # 1) score every site once (dev signal is independent of h)
    scored = {}
    for site, m in META.items():
        d = load_site(site)
        cov_ok, clean = site_masks(d)
        cal_clean, hold_clean = calibration_holdout_split(clean)
        res = RC.score_site(d["err"], clean,
                            w_base=W_BASE, guard=GUARD, min_base=MIN_BASE,
                            floor=FLOOR, slack_k=SLACK_K, h=1e9)  # h unused here
        scored[site] = {"dev": res["dev"], "z": res["z"],
                        "cov_ok": cov_ok, "clean": clean,
                        "cal_clean": cal_clean, "hold_clean": hold_clean,
                        "end_ts": d["end_ts"], "near": d["near"],
                        "role": m["role"], "onset_ts": m["onset_ts"],
                        "onset_utc": m["onset_utc"]}
        print(f"[score] {site} dev_valid={np.isfinite(res['dev']).sum()}/"
              f"{len(res['dev'])} clean={int(clean.sum())}", flush=True)

    # 2) calibrate global h on pooled CONTROL calibration-clean windows
    ctrl = [s for s in scored if scored[s]["role"] == "control"]
    cal_dev = np.concatenate([scored[s]["dev"][scored[s]["cal_clean"]] for s in ctrl])
    cal_dev = cal_dev[np.isfinite(cal_dev)]
    # For calibration we need the pooled per-window alarm rate as a function of h.
    # Search h over a grid; pick the SMALLEST h whose pooled control CAL FPR <= target
    # (smallest h = most sensitive that still meets specificity).
    hi = np.percentile(cal_dev, 99.99) if cal_dev.size else 10.0
    cand = np.unique(np.round(np.linspace(SLACK_K, max(hi, SLACK_K + 1), 300), 4))
    best_h = float(cand[-1])
    for h in cand:
        num = den = 0
        for s in ctrl:
            al = RC.cusum_alarms(scored[s]["dev"], SLACK_K, h)
            m = scored[s]["cal_clean"]
            num += al[m].sum(); den += m.sum()
        if den and num / den <= TARGET_FPR:
            best_h = float(h); break

    per_site, pooled_ctrl_fpr = evaluate(best_h, scored)

    # 3) lead time on episode sites (strict: pre-onset AND ~near)
    n_val = 0
    ep_out = {}
    for site, sc in scored.items():
        if sc["role"] != "episode" or sc["onset_ts"] is None:
            continue
        alarm = RC.cusum_alarms(sc["dev"], SLACK_K, best_h)
        onset = sc["onset_ts"]
        et = sc["end_ts"]
        # detection window: within lookback before onset, covered, and NOT
        # hypoxia-near (guardrail: alarm cannot be reading contemporaneous DO<4)
        keep = (et >= onset - LOOKBACK_DAYS * DAY) & (et < onset)
        keep &= sc["cov_ok"] & (~sc["near"])
        t = RC.first_alarm_time(alarm, et, keep)
        detected = t is not None
        lead = (onset - t) / DAY if detected else None
        if detected:
            n_val += 1
        ep_out[site] = {"detected_pre_onset": detected, "lead_days": lead,
                        "onset_utc": sc["onset_utc"],
                        "fpr_heldout": per_site[site]["fpr_heldout"],
                        "n_keep": int(keep.sum())}
        print(f"[lead] {site} detected={detected} lead={lead} "
              f"fpr_hold={per_site[site]['fpr_heldout']:.3f}", flush=True)

    result = {
        "method": "causal_rolling_baseline_multivariate_cusum",
        "config": {"w_base": W_BASE, "guard": GUARD, "min_base": MIN_BASE,
                   "floor": FLOOR, "slack_k": SLACK_K, "target_fpr": TARGET_FPR,
                   "lookback_days": LOOKBACK_DAYS, "min_cov": MIN_COV,
                   "stride_h": 2.0, "global_h": best_h,
                   "calibration": "global h on pooled CONTROL cal-clean windows",
                   "lead_guardrail": "first alarm strictly pre-onset AND ~hypoxia-near"},
        "global_h": best_h,
        "pooled_control_heldout_fpr": pooled_ctrl_fpr,
        "n_validated_episodes": n_val,
        "episodes": ep_out,
        "per_site_fpr": per_site,
    }
    (OUTDIR / "rolling_cusum_results.json").write_text(json.dumps(result, indent=2))
    print(f"\nglobal_h={best_h:.4f}  pooled_ctrl_heldout_FPR={pooled_ctrl_fpr:.4f}  "
          f"validated_episodes={n_val}")
    print(f"WROTE {OUTDIR/'rolling_cusum_results.json'}")


if __name__ == "__main__":
    main()
