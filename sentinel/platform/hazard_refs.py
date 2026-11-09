# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""EW-v2 hazard thresholds as chart reference lines, taken from ``sentinel.eval.hazards``.

Thresholds and their regulatory anchors are never typed into the platform; they are read
from the pre-registered definitions in :data:`sentinel.eval.hazards.HAZARDS`.
"""

from __future__ import annotations

from dataclasses import asdict

from ..eval.hazards import HAZARDS
from . import variables as V

NOTE = "criterion-anchored excursion, not a regulatory determination"


def hazard_table() -> list[dict]:
    out = []
    for hz in HAZARDS.values():
        base, stat = V.split_column(hz.variable)
        d = asdict(hz)
        d.update(
            {
                "base_variable": base,
                "stat": stat,
                "unit": V.UNITS.get(base),
                "label": V.LABELS.get(base, base),
                "note": NOTE,
            }
        )
        out.append(d)
    return out


def lines_for(column: str, tier: str = "daily") -> list[dict]:
    """Hazards that apply to a plotted column (daily: exact variable; hourly: same base + stat)."""
    stats = V.DAILY_STATS if tier == "daily" else V.HOURLY_STATS
    base, stat = V.split_column(column, stats)
    out = []
    for h in hazard_table():
        if tier == "daily" and h["variable"] != column:
            continue
        if tier != "daily" and (h["base_variable"] != base or h["stat"] != stat):
            continue
        rule = (
            f"daily-tier rule; onset needs {h['pre_days']} clean days before"
            if tier == "daily"
            else f"sub-daily tier applies the same threshold with a minimum duration of "
            f"{h['subdaily_min_duration']}"
        )
        out.append({**h, "applies_to": column, "tier": tier, "rule": rule})
    return out
