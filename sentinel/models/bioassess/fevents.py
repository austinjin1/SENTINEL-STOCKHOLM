# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""M5-F event track F-E (protocol ``bioassess_f_protocol.md`` v1 §6.4).

The French events are ``EventDef`` objects of the shared ``sentinel.eval.events`` v1.0 (imported, never copied or
edited; sha256 pinned) and go through its API: ``event_state`` for labels, ``exceedance_probability`` for the
derived route, ``recalibrate`` for the development-only recalibration and ``power_flag`` for the power rule.
Thresholds are in label units (mg N/L, mg P/L, pH units, % saturation, mg O2/L). Every event is evaluated on the
±30-day median label of the F-A test rows.

These events are exceedances of the French good/moderate limits by the ±30-day median. They are NOT the regulatory
status class (which uses the P90, or P10, of a series of analyses).

This module holds definitions and adapters only. It reads no data and fits no model.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from sentinel.models.bioassess.fdata import FR_FACTORS

EVENTS_V1_SHA256 = "4279d2427d777405ce1cb140cab995567a8086fee5fd4776950e4de1065954ae"   # released v1.0 (0444)
WINDOW = "pm30d"                                   # every event: the ±30-day median label
MIN_POS, MIN_CLUSTERS = 30, 10                     # power rule on the development pseudo-test
ANCHOR = ("Guide technique relatif a l'evaluation de l'etat des eaux de surface continentales, Ministere de la "
          "Transition ecologique, December 2023, Annex 6, general table, column Bon / Moyen")
REUSED = ("E2a_lab", "E5b", "E6_snap")             # v1.0 events kept for comparability with M5-X (not French limits)

# French limits in label units. The molar factors are those of the label construction (fdata.FR_FACTORS, §6.3).
LIMITS = {
    "F_NO3": 50.0 * FR_FACTORS["no3_to_n"],        # 50 mg NO3/L  = 11.295 mg N/L
    "F_PO4": 0.5 * FR_FACTORS["po4_to_p"],         # 0.5 mg PO4/L = 0.16305 mg P/L
    "F_NH4": 0.5 * FR_FACTORS["nh4_to_n"],         # 0.5 mg NH4/L = 0.38825 mg N/L
    "F_TP": 0.2,                                   # mg P/L
    "F_PH": (6.0, 9.0),                            # pH minimum 6, pH maximum 9
    "F_DOSAT": 70.0,                               # % saturation
    "F_BOD": 6.0,                                  # mg O2/L
}
# code -> (name, variable, unit, direction)
_FRENCH = {
    "F_NO3": ("nitrate-N above the French good/moderate limit (50 mg NO3/L)", "no3n_mg_l", "mg N/L", "above"),
    "F_PO4": ("orthophosphate-P above the French good/moderate limit (0.5 mg PO4/L)", "po4p_mg_l", "mg P/L", "above"),
    "F_NH4": ("ammonium-N above the French good/moderate limit (0.5 mg NH4/L)", "nh4n_mg_l", "mg N/L", "above"),
    "F_TP": ("total P above the French good/moderate limit (0.2 mg P/L)", "tp_mg_l", "mg P/L", "above"),
    "F_PH": ("pH outside the French good/moderate limits (6.0-9.0)", "ph", "pH units", "outside"),
    "F_DOSAT": ("DO saturation below the French good/moderate limit (70 %)", "do_sat_pct", "% saturation", "below"),
    "F_BOD": ("BOD5 above the French good/moderate limit (6 mg O2/L)", "bod5_mg_l", "mg O2/L", "above"),
}
# code -> (M5-F target, scale of the target). "log10": value in label units = 10 ** y; "identity": value = y
TRACK = {
    "E2a_lab": ("log10_chloride_mg_l", "log10"),
    "E5b": ("log10_tp_mg_l", "log10"),
    "E6_snap": ("ph", "identity"),
    "F_NO3": ("log10_no3n_mg_l", "log10"),
    "F_PO4": ("log10_po4p_mg_l", "log10"),
    "F_NH4": ("log10_nh4n_mg_l", "log10"),
    "F_TP": ("log10_tp_mg_l", "log10"),
    "F_PH": ("ph", "identity"),
    "F_DOSAT": ("do_sat_pct", "identity"),
    "F_BOD": ("log10_bod5_mg_l", "log10"),
}
CODES = tuple(TRACK)
# quantile levels that carry a Student-t predictive distribution through the v1.0 API (399 levels, 0.0025 apart)
T_LEVELS = np.round(np.arange(1, 400) / 400.0, 6)
_CACHE: dict = {}


def events_module():
    """The shared module, released v1.0 only (sha256 pinned); ImportError otherwise."""
    from sentinel.eval import events as EV  # noqa: PLC0415

    with open(EV.__file__, "rb") as fh:
        if hashlib.sha256(fh.read()).hexdigest() != EVENTS_V1_SHA256:
            raise ImportError("sentinel/eval/events.py is not the released v1.0 (sha256 mismatch)")
    return EV


def event_def(code: str):
    """The ``EventDef`` of ``code``: a v1.0 event (reused) or a French event built with the v1.0 class."""
    if code not in TRACK:
        raise KeyError(f"unknown M5-F event {code!r}; known: {', '.join(CODES)}")
    if code not in _CACHE:
        EV = events_module()
        if code in REUSED:
            _CACHE[code] = EV.get_event(code)
        else:
            name, var, unit, direction = _FRENCH[code]
            _CACHE[code] = EV.EventDef(code, name, "chronic", (var,), unit, direction, LIMITS[code], "snapshot",
                                       anchor=ANCHOR)
    return _CACHE[code]


def column(code: str) -> str:
    """The label / prediction column of the event (±30-day median window)."""
    return f"{TRACK[code][0]}__{WINDOW}"


def to_event_units(y, code: str) -> np.ndarray:
    y = np.asarray(y, float)
    # back-transform rounded to 6 decimals, as in the M5-X adapter (10 ** log10(t) can exceed t in floating point)
    return np.round(10.0 ** y, 6) if TRACK[code][1] == "log10" else y


def labels(y, code: str) -> np.ndarray:
    """Event state {0, 1, NaN} of the measured ±30-day median label ``y`` (target scale), by ``events.event_state``."""
    return np.asarray(events_module().event_state(to_event_units(y, code), event_def(code)), float)


def derived_probability(loc, scale, code: str, nu: float | None = None, chunk: int = 20000) -> np.ndarray:
    """P(event) from a predictive distribution on the target scale, through ``events.exceedance_probability``.

    ``nu`` None: Gaussian(loc, scale). ``nu`` given: Student-t with ``nu`` degrees of freedom (BioSet-MA), passed
    as its quantiles at ``T_LEVELS`` (the v1.0 API interpolates between quantiles; the difference from the exact
    t CDF is below 0.002, unit-tested). Rows with a non-finite location or scale give NaN."""
    EV = events_module()
    ev = event_def(code)
    space = "log10" if TRACK[code][1] == "log10" else "linear"
    loc = np.asarray(loc, float)
    scale = np.broadcast_to(np.asarray(scale, float), loc.shape).astype(float)
    if nu is None:
        return np.asarray(EV.exceedance_probability(EV.Gaussian(mu=loc, sigma=scale, space=space), ev), float)
    from scipy.stats import t as student_t

    z = student_t.ppf(T_LEVELS, float(nu))
    out = np.full(loc.shape, np.nan)
    flat_loc, flat_scale, flat_out = loc.ravel(), scale.ravel(), out.ravel()
    ok = np.flatnonzero(np.isfinite(flat_loc) & np.isfinite(flat_scale) & (flat_scale > 0))
    for s in range(0, len(ok), int(chunk)):
        r = ok[s:s + int(chunk)]
        q = EV.Quantiles(levels=T_LEVELS, values=flat_loc[r, None] + flat_scale[r, None] * z[None, :], space=space)
        flat_out[r] = EV.exceedance_probability(q, ev)
    return flat_out.reshape(loc.shape)


def power(y_evt, clusters) -> dict:
    """Power rule of §6.4 on development pseudo-test rows: >= 30 positives in >= 10 clusters."""
    y = np.asarray(y_evt, float)
    pos = y == 1
    n_pos, n_cl = int(pos.sum()), int(pd.Series(np.asarray(clusters, object)[pos]).nunique())
    word = events_module().power_flag(n_pos, n_cl, MIN_POS, MIN_CLUSTERS)
    return {"n": int(np.isfinite(y).sum()), "n_pos": n_pos, "n_clusters_pos": n_cl, "flag": word,
            "evaluated": word == "feasible"}


def crossfit_calibrated(p_raw, y_evt, fold) -> np.ndarray:
    """Development numbers: Platt map fitted on the other development folds, applied to fold g."""
    EV = events_module()
    p_raw, y_evt, fold = np.asarray(p_raw, float), np.asarray(y_evt, float), np.asarray(fold, object)
    out = np.full(len(p_raw), np.nan)
    for g in pd.unique(fold):
        tr = (fold != g) & np.isfinite(y_evt) & np.isfinite(p_raw)
        te = (fold == g) & np.isfinite(p_raw)
        if tr.sum() < 10 or len(np.unique(y_evt[tr])) < 2:
            out[te] = p_raw[te]
            continue
        out[te] = EV.recalibrate(p_raw[tr], y_evt[tr]).transform(p_raw[te])
    return out


def frozen_calibrated(p_dev_raw, y_dev, p_new_raw) -> np.ndarray:
    """Test look: Platt map fitted on development rows only (all out-of-fold probabilities), applied to new rows."""
    EV = events_module()
    p_dev_raw, y_dev = np.asarray(p_dev_raw, float), np.asarray(y_dev, float)
    ok = np.isfinite(p_dev_raw) & np.isfinite(y_dev)
    return np.asarray(EV.recalibrate(p_dev_raw[ok], y_dev[ok]).transform(p_new_raw), float)


def table() -> pd.DataFrame:
    """The event table of protocol §6.4 as built by this module (for the freeze record and the tests)."""
    rows = []
    for code in CODES:
        ev = event_def(code)
        rows.append({"code": code, "column": column(code), "scale": TRACK[code][1], "unit": ev.unit,
                     "direction": ev.direction, "threshold": ev.threshold, "reused_v1": code in REUSED})
    return pd.DataFrame(rows)


__all__ = ["CODES", "REUSED", "LIMITS", "TRACK", "WINDOW", "MIN_POS", "MIN_CLUSTERS", "T_LEVELS", "EVENTS_V1_SHA256",
           "events_module", "event_def", "column", "to_event_units", "labels", "derived_probability", "power",
           "crossfit_calibrated", "frozen_calibrated", "table"]
