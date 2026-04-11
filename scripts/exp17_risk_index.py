#!/usr/bin/env python3
"""Exp17: Composite Water Quality Risk Index.

Combines AquaSSM anomaly scores (Exp NEON scan), temporal trends (Exp8),
threshold exceedance rates, and EPA event correlation into a single
composite risk index for all 32 NEON monitoring sites.

Risk tier classification:
  Tier 5 (Critical): composite > 0.70
  Tier 4 (High):     composite > 0.55
  Tier 3 (Elevated): composite > 0.40
  Tier 2 (Moderate): composite > 0.25
  Tier 1 (Low):      composite <= 0.25

Output:
  results/exp17_risk_index/risk_index_results.json
  paper/figures/fig_exp17_risk_ranking.jpg

MIT License — Bryan Cheng, 2026
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sentinel.utils.logging import get_logger

logger = get_logger(__name__)

OUTPUT_DIR = PROJECT_ROOT / "results" / "exp17_risk_index"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR    = PROJECT_ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Risk tier definitions
TIERS = [
    (0.70, "Critical",  5, "#8B0000"),
    (0.55, "High",      4, "#CC2200"),
    (0.40, "Elevated",  3, "#E87722"),
    (0.25, "Moderate",  2, "#F5C518"),
    (0.00, "Low",       1, "#2E8B57"),
]

# NEON site metadata (ecoregion/land-use type for context)
SITE_META = {
    "POSE": "Tallgrass Prairie/Agricultural runoff",
    "BLDE": "Western montane, snowmelt-driven",
    "MART": "Pacific coastal, intermittent",
    "COMO": "Rocky Mountain alpine",
    "GUIL": "Mid-Atlantic Piedmont",
    "MCDI": "Upper Midwest agricultural",
    "LECO": "Great Lakes tributary",
    "CUPE": "Southeastern Coastal Plain",
    "LIRO": "Northern Lakes wetland",
    "MCRA": "Cascade Range volcanic",
    "CRAM": "North-Central hardwood forest",
    "ARIK": "Semi-arid shortgrass steppe",
    "KING": "Southeastern Coastal Plain forest",
    "PRPO": "Prairie pothole agricultural (high baseline SpCond)",
    "BLUE": "Ozark Highlands",
    "WALK": "Eastern deciduous forest",
    "LEWI": "Atlantic coastal plain",
    "MAYF": "Southeastern longleaf pine",
    "OKSR": "Arctic/boreal tundra",
    "REDB": "Boreal peatland",
}


def load_data():
    """Load all required experiment results."""
    # NEON anomaly scan
    scan = json.load(open(PROJECT_ROOT / "results" / "neon_anomaly_scan" / "neon_scan_results.json"))
    per_site = scan["per_site"]

    # Exp8 trends
    trends = json.load(open(PROJECT_ROOT / "results" / "exp8_neon_trends" / "trend_results.json"))
    trends_per_site = trends.get("per_site", {})

    return per_site, trends_per_site


def compute_component_scores(per_site, trends_per_site):
    """Compute 4 normalized component scores per site."""
    sites = {s: r for s, r in per_site.items() if r.get("status") == "success"}

    # --- Component 1: AquaSSM anomaly level (mean_score, p95_score) ---
    mean_scores = np.array([sites[s]["mean_score"] for s in sites])
    p95_scores  = np.array([sites[s]["p95_score"]  for s in sites])
    # Weighted combination: 40% mean, 60% p95 (tail risk matters more)
    aquassm_raw = 0.4 * mean_scores + 0.6 * p95_scores
    aquassm_raw_max = aquassm_raw.max() + 1e-8
    aquassm_norm = aquassm_raw / aquassm_raw_max

    # --- Component 2: Threshold exceedance rate ---
    exceedance_raw = np.array([sites[s]["label_anomaly_rate"] for s in sites])
    exceedance_norm = np.clip(exceedance_raw, 0, 1)

    # --- Component 3: Trend severity (degrading DO or rising turbidity) ---
    trend_scores = {}
    for s in sites:
        t = trends_per_site.get(s, {})
        if not t or t.get("status") != "ok":
            trend_scores[s] = 0.0
            continue
        pt = t.get("param_trends", {})
        score = 0.0
        # Degrading DO: negative trend
        do_trend = pt.get("dissolvedOxygen", {})
        if do_trend.get("trend") in ("decreasing", "significantly decreasing"):
            tau = abs(do_trend.get("tau", 0))
            score += min(tau * 1.5, 0.5)
        # Rising turbidity
        tb_trend = pt.get("turbidity", {})
        if tb_trend.get("trend") in ("increasing", "significantly increasing"):
            tau = abs(tb_trend.get("tau", 0))
            score += min(tau * 1.5, 0.5)
        # Declining SpCond (saline dilution can indicate hydrological stress)
        sc_trend = pt.get("specificConductance", {})
        if abs(sc_trend.get("slope_per_year", 0)) > 100:  # large change
            score += 0.2
        trend_scores[s] = min(score, 1.0)

    trend_arr = np.array([trend_scores[s] for s in sites])

    # --- Component 4: Peak event severity (max_score) ---
    max_scores = np.array([sites[s]["max_score"] for s in sites])
    max_norm   = np.clip(max_scores / 0.85, 0, 1)  # 0.85 is near-max observed

    # --- Composite: weighted sum ---
    # AquaSSM level 35%, exceedance 25%, trend 20%, peak severity 20%
    weights = [0.35, 0.25, 0.20, 0.20]
    composites = (weights[0] * aquassm_norm +
                  weights[1] * exceedance_norm +
                  weights[2] * trend_arr +
                  weights[3] * max_norm)

    site_list = list(sites.keys())
    results = {}
    for i, s in enumerate(site_list):
        comp = float(composites[i])
        tier_name, tier_num, tier_color = "Low", 1, "#2E8B57"
        for thresh, name, num, color in TIERS:
            if comp >= thresh:
                tier_name, tier_num, tier_color = name, num, color
                break

        results[s] = {
            "composite_score":     round(comp, 4),
            "tier":                tier_num,
            "tier_name":           tier_name,
            "components": {
                "aquassm_level":    round(float(aquassm_norm[i]), 4),
                "exceedance_rate":  round(float(exceedance_norm[i]), 4),
                "trend_severity":   round(float(trend_arr[i]), 4),
                "peak_severity":    round(float(max_norm[i]), 4),
            },
            "raw": {
                "mean_score":        round(float(mean_scores[i]), 4),
                "p95_score":         round(float(p95_scores[i]), 4),
                "max_score":         round(float(max_scores[i]), 4),
                "label_anomaly_rate":round(float(exceedance_raw[i]), 4),
            },
            "context": SITE_META.get(s, ""),
            "_tier_color": tier_color,
        }

    return results


def tier_distribution(results):
    dist = {1: [], 2: [], 3: [], 4: [], 5: []}
    for s, d in results.items():
        dist[d["tier"]].append(s)
    return dist


def main():
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("EXP17: Composite Water Quality Risk Index")
    logger.info("=" * 60)

    per_site, trends_per_site = load_data()
    site_scores = compute_component_scores(per_site, trends_per_site)
    ranked = sorted(site_scores.items(), key=lambda x: -x[1]["composite_score"])
    dist   = tier_distribution(site_scores)

    output = {
        "n_sites": len(site_scores),
        "weights": {"aquassm_level": 0.35, "exceedance_rate": 0.25,
                    "trend_severity": 0.20, "peak_severity": 0.20},
        "tier_definitions": {
            "5_critical": ">0.70",
            "4_high":     "0.55–0.70",
            "3_elevated": "0.40–0.55",
            "2_moderate": "0.25–0.40",
            "1_low":      "<=0.25",
        },
        "tier_distribution": {f"tier_{k}": v for k, v in dist.items()},
        "ranked_sites": [{"site": s, **d} for s, d in ranked],
        "top10": [s for s, _ in ranked[:10]],
        "elapsed_s": round(time.time() - t0, 1),
    }

    out_path = OUTPUT_DIR / "risk_index_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Saved: {out_path}")

    # --- Figure: horizontal bar chart ranked by composite score ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n = len(ranked)
        fig, ax = plt.subplots(figsize=(10, max(8, n * 0.32)))
        sites_plot  = [s for s, _ in ranked[::-1]]
        scores_plot = [d["composite_score"] for _, d in ranked[::-1]]
        colors_plot = [d["_tier_color"] for _, d in ranked[::-1]]
        tier_labels = [f"T{d['tier']}" for _, d in ranked[::-1]]

        bars = ax.barh(range(n), scores_plot, color=colors_plot, edgecolor="white", linewidth=0.4)
        ax.set_yticks(range(n))
        ax.set_yticklabels([f"{s} [{tl}]" for s, tl in zip(sites_plot, tier_labels)], fontsize=8)
        ax.set_xlabel("Composite Risk Score")
        ax.set_title("SENTINEL Water Quality Risk Index — 32 NEON Sites\n"
                     "(AquaSSM 35% + Exceedance 25% + Trend 20% + Peak Severity 20%)")
        ax.axvline(0.70, color="#8B0000", ls="--", lw=0.8, label="Critical (0.70)")
        ax.axvline(0.55, color="#CC2200", ls="--", lw=0.8, label="High (0.55)")
        ax.axvline(0.40, color="#E87722", ls="--", lw=0.8, label="Elevated (0.40)")
        ax.axvline(0.25, color="#F5C518", ls="--", lw=0.8, label="Moderate (0.25)")
        ax.legend(loc="lower right", fontsize=7)

        # Annotate top 5
        for i, (s, d) in enumerate(reversed(ranked[:5])):
            ax.text(d["composite_score"] + 0.005, n - 1 - i,
                    f"{d['composite_score']:.3f}", va="center", fontsize=7)

        plt.tight_layout()
        fig_path = FIG_DIR / "fig_exp17_risk_ranking.jpg"
        plt.savefig(str(fig_path), dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved: {fig_path}")
    except Exception as e:
        logger.warning(f"Figure failed: {e}")

    # Console summary
    logger.info("\n=== COMPOSITE RISK RANKING (TOP 15) ===")
    logger.info(f"{'Rank':<5} {'Site':<6} {'Score':<8} {'Tier':<12} {'AQ':<6} {'Exc':<6} {'Trend':<6} {'Peak'}")
    logger.info("-" * 70)
    for rank, (s, d) in enumerate(ranked[:15], 1):
        c = d["components"]
        logger.info(f"  {rank:<3} {s:<6} {d['composite_score']:<8.4f} "
                    f"{d['tier_name']:<12} {c['aquassm_level']:<6.3f} "
                    f"{c['exceedance_rate']:<6.3f} {c['trend_severity']:<6.3f} {c['peak_severity']:.3f}")

    logger.info("\n=== TIER DISTRIBUTION ===")
    tier_names = {5: "Critical", 4: "High", 3: "Elevated", 2: "Moderate", 1: "Low"}
    for t in [5, 4, 3, 2, 1]:
        sites_t = dist[t]
        logger.info(f"  Tier {t} ({tier_names[t]}): {len(sites_t)} sites — {', '.join(sorted(sites_t))}")

    logger.info(f"\nElapsed: {output['elapsed_s']:.1f}s")


if __name__ == "__main__":
    main()
