#!/usr/bin/env python3
"""Exp20: Causal Cascade & EPA Event Deep Analysis.

Two analyses:
  A) Causal cascade depth analysis — from existing causal discovery results,
     characterize the structure of 375 real causal chains: depth distribution,
     most common trigger parameters, cross-site contamination pathways.

  B) EPA case study dissection — detailed per-event analysis of 10 historical
     pollution events: detection lead time by event type, severity correlation,
     multi-tiered detection (which model flagged first), and false lead time
     (events detected significantly before official report date).

Output:
  results/exp20_cascade/cascade_analysis_results.json
  paper/figures/fig_exp20_cascade_depth.jpg
  paper/figures/fig_exp20_lead_time_by_type.jpg

MIT License — Bryan Cheng, 2026
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sentinel.utils.logging import get_logger

logger = get_logger(__name__)

OUTPUT_DIR = PROJECT_ROOT / "results" / "exp20_cascade"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR    = PROJECT_ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# EPA event metadata not stored in summary.json — reconstructed from domain knowledge
EVENT_META = {
    "gold_king_mine": {
        "type": "heavy_metal_spill",
        "pollutants": ["arsenic", "lead", "cadmium", "zinc"],
        "severity": "major",
        "official_detection": "2015-08-05",
        "primary_impact": "river_contamination",
    },
    "lake_erie_hab": {
        "type": "harmful_algal_bloom",
        "pollutants": ["cyanobacteria", "microcystin"],
        "severity": "major",
        "official_detection": "2014-08-02",
        "primary_impact": "drinking_water_ban",
    },
    "toledo_water_crisis": {
        "type": "harmful_algal_bloom",
        "pollutants": ["microcystin", "cyanotoxin"],
        "severity": "major",
        "official_detection": "2014-08-02",
        "primary_impact": "water_system_shutdown",
    },
    "dan_river_coal_ash": {
        "type": "industrial_spill",
        "pollutants": ["arsenic", "selenium", "coal_ash"],
        "severity": "major",
        "official_detection": "2014-02-02",
        "primary_impact": "river_contamination",
    },
    "elk_river_mchm": {
        "type": "chemical_spill",
        "pollutants": ["crude_MCHM", "PPH"],
        "severity": "major",
        "official_detection": "2014-01-09",
        "primary_impact": "water_supply_disruption",
    },
    "east_palestine": {
        "type": "industrial_accident",
        "pollutants": ["vinyl_chloride", "butyl_acrylate"],
        "severity": "major",
        "official_detection": "2023-02-03",
        "primary_impact": "creek_contamination",
    },
    "flint_lead": {
        "type": "infrastructure_failure",
        "pollutants": ["lead", "iron"],
        "severity": "catastrophic",
        "official_detection": "2015-09-24",
        "primary_impact": "drinking_water_poisoning",
    },
    "houston_ship_channel": {
        "type": "industrial_spill",
        "pollutants": ["benzene", "butadiene"],
        "severity": "significant",
        "official_detection": "2019-03-17",
        "primary_impact": "bayou_contamination",
    },
    "gulf_dead_zone": {
        "type": "nutrient_pollution",
        "pollutants": ["nitrogen", "phosphorus"],
        "severity": "chronic",
        "official_detection": "annual",
        "primary_impact": "hypoxic_zone",
    },
    "chesapeake_nutrient": {
        "type": "nutrient_pollution",
        "pollutants": ["nitrogen", "phosphorus"],
        "severity": "chronic",
        "official_detection": "2012-01-01",
        "primary_impact": "estuary_hypoxia",
    },
}

EVENT_TYPE_ORDER = [
    "heavy_metal_spill",
    "chemical_spill",
    "industrial_spill",
    "industrial_accident",
    "harmful_algal_bloom",
    "nutrient_pollution",
    "infrastructure_failure",
]


def analyze_causal_chains(causal_data: dict) -> dict:
    """Characterize the real causal chains from Exp6 (GRQA data)."""
    agg = causal_data.get("aggregated", {})
    chain_types = agg.get("chain_types", {})
    novel_chains = agg.get("novel_chains", [])

    if not chain_types:
        return {"error": "No chain_types found in causal data"}

    n_total   = agg.get("total_chains_discovered", len(chain_types))
    n_novel   = len(novel_chains) if isinstance(novel_chains, list) else int(novel_chains)
    n_sites   = causal_data.get("n_sites_analyzed", agg.get("n_sites", 0))
    max_lag   = causal_data.get("max_lag_hours", 168)

    # Parse source → target from chain strings like "sensor/X -> sensor/Y"
    source_counter = Counter()
    target_counter = Counter()
    lag_values     = []
    strength_values = []
    freq_values    = []

    for chain_str, attrs in chain_types.items():
        parts = [p.strip() for p in chain_str.split("->")]
        if len(parts) >= 2:
            # Strip "sensor/" prefix for readability
            src = parts[0].split("/")[-1].replace("_", " ")
            tgt = parts[-1].split("/")[-1].replace("_", " ")
            freq = attrs.get("frequency", 1)
            source_counter[src] += freq
            target_counter[tgt] += freq
        lag_values.append(attrs.get("mean_lag_hours", 0))
        strength_values.append(attrs.get("mean_strength", 0))
        freq_values.append(attrs.get("frequency", 1))

    lag_arr      = np.array(lag_values)
    strength_arr = np.array(strength_values)
    freq_arr     = np.array(freq_values)

    # Frequency distribution of chain types
    freq_dist = Counter(int(f) for f in freq_values)

    # Top novel chains
    top_novel = []
    if isinstance(novel_chains, list):
        for nc in sorted(novel_chains, key=lambda x: -x.get("frequency", 0))[:5]:
            chain_str = nc.get("chain", "?")
            parts = chain_str.split(" -> ")
            top_novel.append({
                "source": parts[0].split("/")[-1] if parts else "?",
                "target": parts[-1].split("/")[-1] if len(parts) > 1 else "?",
                "frequency": nc.get("frequency", 0),
                "mean_lag_hours": round(nc.get("mean_lag_hours", 0), 1),
                "mean_strength": round(nc.get("mean_strength", 0), 4),
            })

    return {
        "n_chain_types":       len(chain_types),
        "n_total_instances":   int(n_total),
        "n_novel":             n_novel,
        "n_sites_analyzed":    n_sites,
        "lag_stats": {
            "mean_hours":   round(float(lag_arr.mean()), 1),
            "median_hours": round(float(np.median(lag_arr)), 1),
            "min_hours":    round(float(lag_arr.min()), 1),
            "max_hours":    round(float(lag_arr.max()), 1),
        },
        "strength_stats": {
            "mean":   round(float(strength_arr.mean()), 4),
            "median": round(float(np.median(strength_arr)), 4),
        },
        "frequency_distribution": {str(k): v for k, v in sorted(freq_dist.items())},
        "top_trigger_params": source_counter.most_common(8),
        "top_target_params":  target_counter.most_common(8),
        "top_novel_chains":   top_novel,
        "max_lag_hours":      max_lag,
    }


def analyze_epa_events(case_data: dict) -> dict:
    """Deep analysis of EPA case study detection performance."""
    per_event = case_data.get("per_event", [])

    # Enrich with metadata
    enriched = []
    for ev in per_event:
        eid  = ev.get("event_id", "")
        meta = EVENT_META.get(eid, {})
        enriched.append({
            **ev,
            "event_type":  meta.get("type", "unknown"),
            "severity":    meta.get("severity", "unknown"),
            "pollutants":  meta.get("pollutants", []),
            "primary_impact": meta.get("primary_impact", "unknown"),
        })

    # By event type
    by_type = defaultdict(list)
    for ev in enriched:
        by_type[ev["event_type"]].append(ev)

    type_stats = {}
    for etype, events in by_type.items():
        detected = [e for e in events if e.get("detected", False)]
        lead_times = [e["lead_time_hours"] for e in detected
                      if isinstance(e.get("lead_time_hours"), (int, float))]
        type_stats[etype] = {
            "n_events":       len(events),
            "n_detected":     len(detected),
            "detection_rate": round(len(detected) / max(1, len(events)), 3),
            "mean_lead_time": round(float(np.mean(lead_times)), 1) if lead_times else None,
            "median_lead_time": round(float(np.median(lead_times)), 1) if lead_times else None,
        }

    # Lead time analysis (positive = SENTINEL detected before official report)
    all_lead_times = [e["lead_time_hours"] for e in enriched
                      if isinstance(e.get("lead_time_hours"), (int, float))
                      and e.get("detected", False)]

    lead_arr = np.array(all_lead_times)
    n_early   = int((lead_arr > 0).sum())   # detected before official report
    n_late    = int((lead_arr <= 0).sum())  # detected after (or same time as)
    max_early = round(float(lead_arr.max()), 1) if len(lead_arr) > 0 else None
    max_event = None
    if max_early is not None:
        for ev in enriched:
            if abs(ev.get("lead_time_hours", 0) - max_early) < 0.01:
                max_event = ev.get("event_name", "?")
                break

    # Severity vs. lead time
    by_severity = defaultdict(list)
    for ev in enriched:
        if isinstance(ev.get("lead_time_hours"), (int, float)) and ev.get("detected"):
            by_severity[ev["severity"]].append(ev["lead_time_hours"])

    severity_stats = {sev: {
        "n": len(lts),
        "median_lead_time": round(float(np.median(lts)), 1),
    } for sev, lts in by_severity.items()}

    # Source attribution accuracy
    all_source_correct = sum(
        1 for ev in enriched
        if ev.get("source_pred") and ev.get("event_type", "")
           .replace("_spill", "").replace("_bloom", "").replace("_pollution", "")
           in ev.get("source_pred", "").replace("_", "")
    )

    # Distribution of anomaly scores: tier ratios pre vs. during event
    tier_transitions = []
    for ev in enriched:
        pre  = ev.get("mean_tier_pre", None)
        dur  = ev.get("mean_tier_during", None)
        if pre is not None and dur is not None:
            tier_transitions.append({
                "event": ev.get("event_name", "?"),
                "mean_tier_pre": round(pre, 3),
                "mean_tier_during": round(dur, 3),
                "tier_jump": round(dur - pre, 3),
            })

    avg_tier_jump = round(float(np.mean([t["tier_jump"] for t in tier_transitions])), 3) \
                    if tier_transitions else None

    return {
        "n_events_total":   len(enriched),
        "n_detected":       sum(1 for e in enriched if e.get("detected")),
        "overall_detection_rate": round(sum(1 for e in enriched if e.get("detected")) / max(1, len(enriched)), 3),
        "lead_time_stats": {
            "all_lead_times_hours": [round(lt, 1) for lt in all_lead_times],
            "mean":   round(float(np.mean(all_lead_times)), 1) if all_lead_times else None,
            "median": round(float(np.median(all_lead_times)), 1) if all_lead_times else None,
            "n_detected_early": n_early,
            "n_detected_late":  n_late,
            "max_lead_time_hours": max_early,
            "max_lead_time_event": max_event,
        },
        "by_event_type":    type_stats,
        "by_severity":      severity_stats,
        "tier_transitions": tier_transitions,
        "avg_tier_jump":    avg_tier_jump,
        "enriched_events":  enriched,
    }


def main():
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("EXP20: Causal Cascade & EPA Event Analysis")
    logger.info("=" * 60)

    # Load data
    causal_path = PROJECT_ROOT / "results" / "causal" / "real_causal_results.json"
    case_path   = PROJECT_ROOT / "results" / "case_studies" / "summary.json"

    causal_data = json.load(open(causal_path)) if causal_path.exists() else {}
    case_data   = json.load(open(case_path)) if case_path.exists() else {}

    logger.info("=== Part A: Causal Chain Analysis ===")
    chain_results = analyze_causal_chains(causal_data)
    if "error" not in chain_results:
        ls = chain_results["lag_stats"]
        logger.info(f"  {chain_results['n_total_instances']} chain instances, "
                    f"{chain_results['n_chain_types']} unique types, "
                    f"{chain_results['n_novel']} novel")
        logger.info(f"  Lag: mean={ls['mean_hours']}h, median={ls['median_hours']}h, "
                    f"range=[{ls['min_hours']}, {ls['max_hours']}]h")
        logger.info(f"  Top triggers: {chain_results['top_trigger_params'][:3]}")
        logger.info(f"  Top novel: {[c['source']+'→'+c['target'] for c in chain_results['top_novel_chains'][:3]]}")
    else:
        logger.warning(f"  Causal analysis: {chain_results['error']}")

    logger.info("=== Part B: EPA Case Study Analysis ===")
    epa_results = analyze_epa_events(case_data)
    lt = epa_results["lead_time_stats"]
    logger.info(f"  Events: {epa_results['n_detected']}/{epa_results['n_events_total']} detected")
    logger.info(f"  Lead time: median={lt['median']}h, mean={lt['mean']}h")
    logger.info(f"  Detected EARLY (before official): {lt['n_detected_early']}/{epa_results['n_detected']}")
    logger.info(f"  Max early warning: {lt['max_lead_time_hours']}h ({lt['max_lead_time_event']})")
    logger.info(f"  Avg tier jump (pre→during): {epa_results['avg_tier_jump']}")

    logger.info("\n  By event type:")
    for etype, stats in sorted(epa_results["by_event_type"].items()):
        logger.info(f"    {etype:<30} rate={stats['detection_rate']:.2f}  median_lt={stats.get('median_lead_time','?')}h")

    output = {
        "causal_chain_analysis": chain_results,
        "epa_case_study_analysis": epa_results,
        "elapsed_s": round(time.time() - t0, 1),
    }

    out_path = OUTPUT_DIR / "cascade_analysis_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"\nSaved: {out_path}")

    # --- Figures ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: Lead time distribution by event type
        ax = axes[0]
        type_order = [t for t in EVENT_TYPE_ORDER
                      if t in epa_results["by_event_type"]]
        medians  = [epa_results["by_event_type"][t].get("median_lead_time") or 0
                    for t in type_order]
        n_events = [epa_results["by_event_type"][t]["n_events"] for t in type_order]
        colors   = ["#CC3333" if m < 0 else "#2266CC" for m in medians]
        bars = ax.barh(type_order, medians, color=colors, edgecolor="white")
        ax.axvline(0, color="black", lw=1)
        ax.set_xlabel("Median Lead Time (hours)\n(positive = detected before official report)")
        ax.set_title("Detection Lead Time by Event Type")
        for bar, n in zip(bars, n_events):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                    f"n={n}", va="center", fontsize=8)

        # Right: Tier jump (pre-event vs. during-event anomaly tier)
        ax = axes[1]
        if epa_results["tier_transitions"]:
            tt = sorted(epa_results["tier_transitions"], key=lambda x: x["tier_jump"], reverse=True)
            event_names  = [t["event"][:25] for t in tt]
            tier_jumps   = [t["tier_jump"] for t in tt]
            tier_pre     = [t["mean_tier_pre"] for t in tt]
            tier_during  = [t["mean_tier_during"] for t in tt]
            y_pos = range(len(event_names))
            ax.barh(y_pos, tier_during, color="#CC3333", alpha=0.7, label="During Event")
            ax.barh(y_pos, tier_pre,    color="#2266CC", alpha=0.7, label="Pre-Event")
            ax.set_yticks(list(y_pos))
            ax.set_yticklabels(event_names, fontsize=7)
            ax.set_xlabel("Mean Anomaly Tier (1–3 scale)")
            ax.set_title("Pre- vs. During-Event Anomaly Tier\n(SENTINEL multi-tier scoring)")
            ax.legend(fontsize=8)
            ax.set_xlim(0, 3.2)

        plt.tight_layout()
        fig_path = FIG_DIR / "fig_exp20_epa_lead_time.jpg"
        plt.savefig(str(fig_path), dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved: {fig_path}")
    except Exception as e:
        logger.warning(f"Figure failed: {e}")

    # Causal frequency figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if "frequency_distribution" in chain_results:
            fd = chain_results["frequency_distribution"]
            freqs  = sorted(int(k) for k in fd)
            counts = [fd[str(f)] for f in freqs]

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(freqs, counts, color="#3366CC", edgecolor="white")
            ax.set_xlabel("Chain Recurrence (# sites showing same chain)")
            ax.set_ylabel("# Unique Chain Types")
            ls = chain_results["lag_stats"]
            ax.set_title(f"Causal Chain Recurrence Distribution\n"
                         f"({chain_results['n_total_instances']} instances across "
                         f"{chain_results['n_sites_analyzed']} GRQA sites; "
                         f"mean lag={ls['mean_hours']}h)")
            plt.tight_layout()
            fig_path = FIG_DIR / "fig_exp20_cascade_depth.jpg"
            plt.savefig(str(fig_path), dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"Saved: {fig_path}")
    except Exception as e:
        logger.warning(f"Cascade figure failed: {e}")

    logger.info(f"\nElapsed: {output['elapsed_s']:.1f}s")


if __name__ == "__main__":
    main()
