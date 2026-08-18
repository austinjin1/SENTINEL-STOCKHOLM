#!/usr/bin/env python
"""Track A, step 3 — aggregate the honest lead-time / specificity report.

Headline lead time is computed ONLY over specificity-validated episodes
(held-out sustained FPR <= target); sites that fire too often on clean water are
reported but excluded from the lead-time headline, because a trigger-happy
detector's "early" alarm is not a real early warning. Caveats are stated.
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[3] / "results" / "trackA_leadtime"
CLEAN_CONTROLS = {"HOPB", "MART", "MCRA"}     # doMin >= 7.7 (truly non-hypoxic)


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (float((c - h) / d), float((c + h) / d))


def main():
    res = json.load(open(OUT / "leadtime_results.json"))
    cfg = res["config"]
    sites = res["sites"]
    target = cfg["target_fpr"]

    episodes = {s: v for s, v in sites.items()
                if v.get("role") == "episode" and "lead_days" in v}
    controls = {s: v for s, v in sites.items() if v.get("role") == "control"}

    # Pooled held-out FPR over the truly-clean control sites + Wilson CI
    k = n = 0
    for s in CLEAN_CONTROLS:
        v = controls.get(s)
        if v and v.get("fpr_heldout") is not None:
            nn = int(v["n_fpr_windows"]); k += round(v["fpr_heldout"] * nn); n += nn
    ctrl_fpr = k / n if n else None
    ctrl_ci = wilson(k, n)
    all_fpr = [v["fpr_heldout"] for v in sites.values() if v.get("fpr_heldout") is not None]

    def validated(v):
        return bool(v.get("detected_pre_onset") and v["fpr_heldout"] <= target + 1e-9)

    val_leads = [v["lead_days"] for v in episodes.values()
                 if validated(v) and v.get("lead_days") is not None]

    L = []
    P = L.append
    P("# Track A — Calibrated label-free lead-time detector (real NEON data)\n")
    P(f"AquaSSM reconstruction-residual detector, per-site calibrated to a target "
      f"sustained false-positive rate of {target:.0%} (K={cfg['k_sustain']} consecutive "
      f"{cfg['stride_h']:.0f}-h windows = {cfg['k_hours']:.0f} h sustained). "
      f"Checkpoint `{cfg['checkpoint']}`. Label-free: no anomaly labels are used to "
      f"fit the detector; the DO<{cfg['do_thr']:.0f} mg/L onset scores lead time after the fact.\n")

    P("## Specificity (held-out clean periods)\n")
    if ctrl_fpr is not None:
        P(f"- Truly-clean control sites {sorted(CLEAN_CONTROLS)} pooled held-out sustained "
          f"FPR: **{ctrl_fpr:.1%}** (Wilson 95% CI {ctrl_ci[0]:.1%}-{ctrl_ci[1]:.1%}, n={n}).")
    P(f"- Mean held-out FPR across all {len(all_fpr)} sites: {np.mean(all_fpr):.1%} "
      f"(median {np.median(all_fpr):.1%}).\n")

    P("## Early warning on specificity-validated episodes\n")
    n_val = sum(validated(v) for v in episodes.values())
    P(f"- Episodes with a verified onset: {len(episodes)}. Detected before onset AND "
      f"meeting the {target:.0%} held-out FPR bar: **{n_val}/{len(episodes)}**.")
    if val_leads:
        P(f"- Median lead time (validated episodes): **{st.median(val_leads):.1f} days** "
          f"(IQR {np.percentile(val_leads,25):.1f}-{np.percentile(val_leads,75):.1f}; "
          f"range {min(val_leads):.1f}-{max(val_leads):.1f}).\n")

    P("| Site | Onset | Held-out FPR | Detected pre-onset | Lead (d) | Specificity-validated |")
    P("|------|-------|--------------|--------------------|----------|------------------------|")
    for s, v in sorted(episodes.items()):
        lead = v.get("lead_days")
        leads_str = f"{lead:.1f}" if lead is not None else "—"
        det = "yes" if v.get("detected_pre_onset") else "no"
        val = "YES" if validated(v) else "no"
        P(f"| {s} | {str(v.get('onset_utc'))[:10]} | {v['fpr_heldout']:.1%} | "
          f"{det} | {leads_str} | {val} |")

    P("\n## Honest caveats\n")
    P(f"- Small validated episode set; lead time is a case-set timing result on real NEON "
      f"streams, not a population estimate, and is bounded above by the clean pre-onset runway.")
    P("- Sites that exceed the FPR target (trigger-happy on clean water) are listed but "
      "excluded from the lead-time headline — their 'early' alarms are not real early warnings.")
    P("- Only 4 real channels (pH/DO/turbidity/conductivity); temperature & ORP are zero-padded.")
    P("- Per-site calibration on the site's own clean history (operational assumption); FPR "
      "is measured on held-out clean periods interleaved with (disjoint from) the calibration runs.")
    P("- AquaSSM sensor encoder only; not a multimodal or forecast model.")

    (OUT / "SUMMARY.md").write_text("\n".join(L))
    print("\n".join(L))
    print(f"\nWROTE {OUT/'SUMMARY.md'}")


if __name__ == "__main__":
    main()
