#!/usr/bin/env python3
"""Experiment A: Early Warning ROC at Multiple Lead Times.

For each of 6 case study events, compute: at what earliest lead time does the
score first cross threshold t?  Sweep t from 0.50 to 0.99 in 0.05 steps.

Output: results/exp_earlywarning/early_warning_results.json

MIT License — Bryan Cheng, 2026
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CASE_DIR   = PROJECT_ROOT / "results" / "case_studies_real"
OUTPUT_DIR = PROJECT_ROOT / "results" / "exp_earlywarning"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EVENT_FILES = {
    "lake_erie_hab_2023":       "HAB",
    "jordan_lake_hab_nc":       "HAB",
    "klamath_river_hab_2021":   "HAB",
    "gulf_dead_zone_2023":      "hypoxia",
    "chesapeake_hypoxia_2018":  "hypoxia",
    "mississippi_salinity_2023":"salinity",
}

THRESHOLDS = [round(t, 2) for t in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80,
                                      0.85, 0.90, 0.95, 0.99]]


def parse_dt(s: str) -> datetime:
    """Parse ISO-8601 datetime string to UTC datetime."""
    s = s.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(s).astimezone(timezone.utc)
    except Exception:
        # fallback
        for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M%z", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(s[:len(fmt)-2], fmt[:-2])
                return dt.replace(tzinfo=timezone.utc)
            except Exception:
                pass
        raise ValueError(f"Cannot parse datetime: {s!r}")


def load_event(event_id: str):
    path = CASE_DIR / f"{event_id}_scores.json"
    with open(path) as f:
        d = json.load(f)
    advisory_dt = parse_dt(d["advisory_date"] + "T00:00:00+00:00")
    scores = d["scores"]
    # Keep only windows before advisory date
    pre_event = [
        s for s in scores
        if parse_dt(s["center_time"]) < advisory_dt
    ]
    return advisory_dt, pre_event, d.get("event_id", event_id)


def first_crossing(scores_pre: list, advisory_dt: datetime, threshold: float):
    """Return (lead_hours, crossing_time) for earliest crossing, or None."""
    for s in scores_pre:
        if s["anomaly_probability"] >= threshold:
            ct = parse_dt(s["center_time"])
            lead_h = (advisory_dt - ct).total_seconds() / 3600.0
            return lead_h, s["center_time"]
    return None, None


def main():
    print("=" * 60)
    print("Experiment A: Early Warning ROC at Multiple Lead Times")
    print("=" * 60)

    # Load all events
    events = {}
    for eid, etype in EVENT_FILES.items():
        advisory_dt, pre_event, _ = load_event(eid)
        events[eid] = {
            "event_type": etype,
            "advisory_date": advisory_dt.isoformat(),
            "n_pre_event_windows": len(pre_event),
            "scores_pre": pre_event,
            "advisory_dt": advisory_dt,
        }
        print(f"  {eid}: {len(pre_event)} pre-event windows, advisory {advisory_dt.date()}")

    # Threshold sweep
    threshold_table = {}
    per_event_results = {}

    for thresh in THRESHOLDS:
        detections = []
        lead_hours_list = []
        thresh_key = f"thresh_{thresh:.2f}"

        for eid, ev in events.items():
            lead_h, crossing_time = first_crossing(
                ev["scores_pre"], ev["advisory_dt"], thresh
            )
            if lead_h is not None and lead_h >= 0:
                detections.append(eid)
                lead_hours_list.append(lead_h)
                if eid not in per_event_results:
                    per_event_results[eid] = {}
                per_event_results[eid][thresh_key] = {
                    "detected": True,
                    "lead_hours": round(lead_h, 2),
                    "crossing_time": crossing_time,
                }
            else:
                if eid not in per_event_results:
                    per_event_results[eid] = {}
                per_event_results[eid][thresh_key] = {"detected": False}

        n_detected = len(detections)
        tpr = n_detected / len(events)
        mean_lead = sum(lead_hours_list) / len(lead_hours_list) if lead_hours_list else 0.0
        min_lead  = min(lead_hours_list) if lead_hours_list else 0.0
        max_lead  = max(lead_hours_list) if lead_hours_list else 0.0

        threshold_table[thresh_key] = {
            "threshold":         thresh,
            "events_detected":   n_detected,
            "total_events":      len(events),
            "tpr":               round(tpr, 4),
            "mean_lead_hours":   round(mean_lead, 2),
            "min_lead_hours":    round(min_lead, 2),
            "max_lead_hours":    round(max_lead, 2),
            "detected_events":   detections,
        }
        print(f"  t={thresh:.2f}: {n_detected}/6 detected"
              f"  TPR={tpr:.3f}  mean_lead={mean_lead:.1f}h"
              f"  min={min_lead:.1f}h  max={max_lead:.1f}h")

    # Compute per-event summary (earliest crossing at t=0.5)
    per_event_summary = {}
    for eid, ev in events.items():
        lead_h, ct = first_crossing(ev["scores_pre"], ev["advisory_dt"], 0.50)
        max_score = max(s["anomaly_probability"] for s in ev["scores_pre"]) if ev["scores_pre"] else 0.0
        per_event_summary[eid] = {
            "event_type":           ev["event_type"],
            "advisory_date":        ev["advisory_date"],
            "n_pre_event_windows":  ev["n_pre_event_windows"],
            "max_pre_event_score":  round(max_score, 4),
            "first_crossing_t0.5":  {
                "lead_hours": round(lead_h, 2) if lead_h is not None else None,
                "crossing_time": ct,
            },
            "threshold_sweep": per_event_results.get(eid, {}),
        }

    output = {
        "experiment": "A: Early Warning ROC at Multiple Lead Times",
        "n_events":   len(events),
        "thresholds": THRESHOLDS,
        "threshold_table": threshold_table,
        "per_event":  per_event_summary,
    }

    out_path = OUTPUT_DIR / "early_warning_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")
    print("\n--- Summary Table ---")
    print(f"{'Threshold':>10} {'Detected':>10} {'TPR':>7} {'Mean Lead (h)':>15} {'Min Lead (h)':>13} {'Max Lead (h)':>13}")
    for tk, row in threshold_table.items():
        print(f"  {row['threshold']:.2f}    {row['events_detected']}/6    {row['tpr']:.3f}    {row['mean_lead_hours']:>13.1f}    {row['min_lead_hours']:>11.1f}    {row['max_lead_hours']:>11.1f}")


if __name__ == "__main__":
    main()
