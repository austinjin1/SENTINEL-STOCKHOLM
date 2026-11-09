# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""EW-v2 hazard definitions and daily-tier onset labelling (RESEARCH_PLAN.md §12).

Every hazard is a *criterion-anchored excursion* (not a regulatory violation). The daily tier
uses USGS daily statistics; the sub-daily (15-min) tier applies the same thresholds with the
duration rules in ``Hazard.subdaily_min_duration`` via ``early_warning.extract_events``.
Definitions here are the ones pre-registered in docs/preregistration/ew_v2_protocol.md; any
change after registration must be logged there as a deviation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Hazard:
    key: str
    variable: str            # daily-tier column (sonde_daily naming)
    direction: str           # 'below' or 'above'
    threshold: float
    pre_margin: float        # onset requires pre-window values beyond threshold by this margin
    pre_days: int            # length of the clean pre-window (days, all observed)
    refractory_days: int     # no new onset within this many days after an event day
    subdaily_min_duration: str
    anchor: str
    pre_gap: int = 0          # days between the clean pre-window and the onset day (averaged series)


HAZARDS = {
    # H1: EPA 1986 warmwater 1-day minimum 3.0 mg/L; tiers 4.0 and 2.0 reported as sensitivity.
    "H1_hypoxia": Hazard("H1_hypoxia", "do_mgl_min", "below", 3.0, 1.0, 3, 7, "2h",
                         "EPA 440/5-86-001 DO 1-day minimum 3.0 mg/L (warmwater)"),
    "H1_hypoxia_t4": Hazard("H1_hypoxia_t4", "do_mgl_min", "below", 4.0, 1.0, 3, 7, "2h",
                            "tier: coldwater 1-day minimum 4.0 mg/L"),
    "H1_hypoxia_t2": Hazard("H1_hypoxia_t2", "do_mgl_min", "below", 2.0, 1.0, 3, 7, "2h",
                            "tier: conventional severe hypoxia 2.0 mg/L"),
    # H4: drinking-water MCL 10 mg/L nitrate as N (daily mean), onset after 14 days < 9 mg/L.
    "H4_nitrate": Hazard("H4_nitrate", "no3no2_mgl_mean", "above", 10.0, 1.0, 14, 14, "6h",
                         "40 CFR 141.62 nitrate MCL 10 mg/L as N"),
    # H5b: in-situ chlorophyll bloom > 20 µg/L (VERA Bloom_binary convention), daily mean.
    "H5b_chl_bloom": Hazard("H5b_chl_bloom", "chl_ugl_mean", "above", 20.0, 5.0, 7, 7, "24h",
                            "VERA Bloom_binary (chl-a > 20 µg/L); WHO alert levels 12/24 µg/L"),
    # H2: chloride derived from SC by a site regression (sentinel/eval/chloride.py; eligible
    # sites only). Chronic: 4-day mean > 230 mg/L after 7 observed days <= 184 mg/L (80%).
    "H2_chloride": Hazard("H2_chloride", "cl_mgl_4dmean", "above", 230.0, 46.0, 7, 7, "1h",
                          "EPA 1988 chloride chronic criterion 230 mg/L (4-day average)", pre_gap=3),
    # H2 acute: derived from daily-max SC, > 860 mg/L (1-h criterion approximated by daily max).
    "H2_chloride_acute": Hazard("H2_chloride_acute", "cl_mgl_max", "above", 860.0, 172.0, 7, 7, "1h",
                                "EPA 1988 chloride acute criterion 860 mg/L"),
}
# H3 (turbidity vs trailing background) needs a per-site derived series (not yet implemented).


def daily_onsets(x: np.ndarray, hz: Hazard) -> tuple[np.ndarray, np.ndarray]:
    """Event-day mask and onset mask for one site's daily series (NaN = missing).

    Event day: value beyond threshold. Onset: an event day whose previous ``pre_days`` days
    are all observed and beyond (threshold ± pre_margin) on the safe side, and with no event
    day in the previous ``refractory_days`` days. With ``pre_gap`` g > 0 the clean window is
    [d-g-pre_days, d-g) instead, so a trailing-mean series (H2: 4-day mean, g = 3) is not
    required to be clean on days that already average in the onset's own values.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if hz.direction == "below":
        event = x < hz.threshold
        safe = x >= hz.threshold + hz.pre_margin
    else:
        event = x > hz.threshold
        safe = x <= hz.threshold - hz.pre_margin
    event &= np.isfinite(x)
    safe &= np.isfinite(x)
    onset = np.zeros(n, dtype=bool)
    for d in np.flatnonzero(event):
        if d < max(hz.pre_days + hz.pre_gap, 1):
            continue
        if not safe[d - hz.pre_gap - hz.pre_days:d - hz.pre_gap].all():
            continue
        lo = max(0, d - hz.refractory_days)
        if event[lo:d].any():
            continue
        onset[d] = True
    return event, onset


def issue_labels(x: np.ndarray, event: np.ndarray, onset: np.ndarray, horizon: int,
                 min_coverage: float = 0.8, refractory_days: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """For issue day d: y=1 if an onset occurs in (d, d+horizon].

    valid=False if d is an event day (event already under way), or if y=0 and fewer than
    ``min_coverage`` of the days in (d, d+horizon] are observed (outcome unknown).
    With ``refractory_days`` (pass ``Hazard.refractory_days``), d is also invalid when an onset
    in (d, d+horizon] is impossible by the refractory rule, i.e. the latest event day e <= d has
    e + refractory_days >= d + horizon (every day of the window is within the refractory period;
    "continuation" days, not valid negatives). Default None keeps the original behaviour.
    """
    n = x.size
    y = np.zeros(n, dtype=int)
    valid = np.zeros(n, dtype=bool)
    obs = np.isfinite(x).astype(float)
    on = onset.astype(int)
    c_obs = np.concatenate([[0.0], np.cumsum(obs)])
    c_on = np.concatenate([[0], np.cumsum(on)])
    last_ev = np.full(n, -10**9)
    cur = -10**9
    for d in range(n):
        if event[d]:
            cur = d
        last_ev[d] = cur
    for d in range(n - horizon):
        n_on = c_on[d + 1 + horizon] - c_on[d + 1]
        cov = (c_obs[d + 1 + horizon] - c_obs[d + 1]) / horizon
        y[d] = int(n_on > 0)
        valid[d] = (not event[d]) and np.isfinite(x[d]) and (y[d] == 1 or cov >= min_coverage)
        if refractory_days is not None and last_ev[d] + refractory_days >= d + horizon:
            valid[d] = False
    return y, valid
