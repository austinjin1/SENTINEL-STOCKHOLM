# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Shared event layer v1 (docs/plans/event_layer_plan.md §1): E-code taxonomy, state/onset labels,
the derived route P(variable beyond threshold) from a predictive distribution, validation-only
recalibration, and event-level metrics.

**Immutable after release.** Freezes of several workstreams hash this file; fixes or additions go
into a new module (``events_v2.py``), never into this one (plan §"Released versions").

Design rules
* E-codes that equal an EW-v2 hazard (E1a ≙ H1 and tiers, E2a/E2b ≙ H2, E4a ≙ H5b, E5a ≙ H4)
  take threshold, direction, variable and onset rule from :data:`sentinel.eval.hazards.HAZARDS` at
  import time and label through :func:`hazards.daily_onsets` / :func:`hazards.issue_labels`: no copies.
* Pure functions on arrays; this module reads no data and knows nothing about splits. Callers remove
  protected (test) rows before calling anything here.
* Not exported from ``sentinel/eval/__init__.py`` (hashed by two freezes); import it directly.

Conventions
* State arrays are float with values {0, 1, NaN} (NaN = not observed). Onset arrays are bool.
* Daily inputs to :func:`derive_series` must sit on a complete daily index (missing days = NaN).
* Distributions may be expressed in a transformed space (``space="log10"`` or ``"ln"``); the event
  threshold is transformed, never the prediction.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import bootstrap as B
from . import ew_scoring as EWS
from . import hazards as HZ
from . import metrics as M

__version__ = "1.0"   # released 2026-11-09; immutable (plan §"Released versions")

Direction = Literal["above", "at_or_above", "below", "outside"]


# ============================================================================ taxonomy

@dataclass(frozen=True)
class OnsetRule:
    """How onsets are found. ``method``:
    * ``"hazard"`` — :func:`hazards.daily_onsets` with these parameters (clean pre-window beyond
      threshold ± ``pre_margin``, optional ``pre_gap``, refractory period);
    * ``"clean_window"`` — event day preceded by ``pre_days`` observed non-event days and no event day in
      the previous ``refractory_days`` days (for derived series without a safe-side margin);
    * ``"run_start"`` — first day of every run of ≥ ``EventDef.duration_days`` consecutive event days."""
    method: Literal["hazard", "clean_window", "run_start"]
    pre_days: int = 0
    pre_margin: float = 0.0
    pre_gap: int = 0
    refractory_days: int = 0


@dataclass(frozen=True)
class EventDef:
    code: str
    name: str
    kind: Literal["acute", "chronic"]
    variables: tuple[str, ...]
    unit: str
    direction: Direction | None
    threshold: float | tuple[float, float] | None
    aggregation: str               # identity | snapshot | 30d_mean | 7dadm | excess_over_bg30 | ph_range_excess | report
    duration_days: int = 1
    tiers: tuple[str, ...] = ()
    hazard_key: str | None = None
    onset: OnsetRule | None = None
    anchor: str = ""
    label_source: Literal["measurement", "report", "condition_class"] = "measurement"
    parent: str | None = None


def _from_hazard(code: str, name: str, kind, key: str, unit: str, tiers=(), parent=None) -> EventDef:
    h = HZ.HAZARDS[key]
    return EventDef(code, name, kind, (h.variable,), unit, h.direction, h.threshold, "identity", tiers=tiers,
                    hazard_key=key, onset=OnsetRule("hazard", h.pre_days, h.pre_margin, h.pre_gap, h.refractory_days),
                    anchor=h.anchor, parent=parent)


def _snap(code, name, kind, var, unit, direction, thr, anchor, parent=None) -> EventDef:
    return EventDef(code, name, kind, (var,), unit, direction, thr, "snapshot", anchor=anchor, parent=parent)


def _report(code, name, kind, anchor) -> EventDef:
    return EventDef(code, name, kind, (), "", None, None, "report", anchor=anchor, label_source="report")


_EPA86_DO = "EPA 440/5-86-001 (1986) dissolved oxygen criteria"
_EVENTS_LIST = [
    _from_hazard("E1a", "hypoxia (acute)", "acute", "H1_hypoxia", "mg/L", tiers=("E1a_t4", "E1a_t2")),
    _from_hazard("E1a_t4", "hypoxia tier < 4 mg/L", "acute", "H1_hypoxia_t4", "mg/L", parent="E1a"),
    _from_hazard("E1a_t2", "hypoxia tier < 2 mg/L", "acute", "H1_hypoxia_t2", "mg/L", parent="E1a"),
    _snap("E1a_grab", "hypoxia, grab sample < 3 mg/L", "acute", "do_mgl", "mg/L", "below", 3.0, _EPA86_DO, "E1a"),
    EventDef("E1b", "hypoxia (chronic, 30-d mean)", "chronic", ("do_mgl_mean",), "mg/L", "below", 5.5, "30d_mean",
             onset=OnsetRule("hazard", pre_days=7, pre_margin=0.5, pre_gap=29, refractory_days=30),
             anchor=_EPA86_DO + ": warmwater 30-day mean 5.5 mg/L (other life stages)"),
    _from_hazard("E2a", "salinisation (chronic chloride)", "chronic", "H2_chloride", "mg/L"),
    _from_hazard("E2b", "salinisation (acute chloride)", "acute", "H2_chloride_acute", "mg/L"),
    _snap("E2a_lab", "chloride sample > 230 mg/L", "chronic", "chloride_mg_l", "mg/L", "above", 230.0,
          "EPA 440/5-88-001 (1988) chloride CCC 230 mg/L", "E2a"),
    _snap("E2c", "ionic stress (conductivity > 300 uS/cm; regional benchmark)", "chronic", "spcond_uscm", "uS/cm",
          "above", 300.0, "EPA/600/R-10/023F (2011) field-based conductivity benchmark (Central Appalachia)"),
    EventDef("E3a", "sediment plume (turbidity > 30-d background + 50)", "acute", ("turb_fnu_mean",), "FNU", "above",
             50.0, "excess_over_bg30", onset=OnsetRule("clean_window", pre_days=3, refractory_days=7),
             anchor="IDAPA 58.01.02.250.02.e (50 NTU instantaneous over background)"),
    _snap("E3a_tss", "sediment plume, TSS > 80 mg/L (snapshot)", "acute", "tss_mgl", "mg/L", "above", 80.0,
          "EIFAC (1965) / Alabaster & Lloyd (1982) suspended-solids class boundary 80 mg/L", "E3a"),
    _snap("E3a_turb", "sediment plume, turbidity > 50 NTU (snapshot, absolute)", "acute", "turbidity_ntu", "NTU",
          "above", 50.0, "IDAPA 58.01.02.250.02.e (absolute variant, no background available)", "E3a"),
    EventDef("E3b", "chronic sediment (background + 25 for >= 10 days)", "chronic", ("turb_fnu_mean",), "FNU", "above",
             25.0, "excess_over_bg30", duration_days=10, onset=OnsetRule("run_start"),
             anchor="IDAPA 58.01.02.250.02.e (25 NTU over background for more than 10 consecutive days)"),
    _from_hazard("E4a", "algal bloom (chl-a > 20 ug/L)", "acute", "H5b_chl_bloom", "ug/L", tiers=("E4a_t12", "E4a_t50")),
    _snap("E4a_t12", "algal bloom tier chl-a > 12 ug/L", "acute", "chl_ugl_mean", "ug/L", "above", 12.0,
          "WHO (2021) recreational cyanobacteria alert level 1", "E4a"),
    _snap("E4a_t50", "algal bloom tier chl-a >= 50 ug/L", "acute", "chl_ugl_mean", "ug/L", "at_or_above", 50.0,
          "orbit_protocol.md §11.1 Track E stratum", "E4a"),
    _snap("E4b", "toxic cyanoHAB (microcystins > 8 ug/L)", "acute", "microcystin_ugl", "ug/L", "above", 8.0,
          "EPA 822-R-19-001 (2019) recreational criterion 8 ug/L"),
    _report("E4c", "reported HAB (agency confirmed / advisory >= Caution)", "acute",
            "NY DEC HABs programme; CA FHAB advisory tiers (CCHAB 2016)"),
    _from_hazard("E5a", "nitrate pollution (NO3+NO2-N > 10 mg/L)", "acute", "H4_nitrate", "mg/L N"),
    _snap("E5b", "nutrient enrichment (TP > 0.1 mg/L)", "chronic", "tp_mgl", "mg/L", "above", 0.1,
          "EPA (1986) Quality Criteria for Water ('Gold Book') total phosphorus 0.1 mg/L"),
    EventDef("E5b_nrsa", "NRSA ecoregional 'Poor' nutrient condition", "chronic", ("tp_cond", "tn_cond"), "class",
             None, None, "report", anchor="EPA NARS condition thresholds", label_source="condition_class",
             parent="E5b"),
    EventDef("E6", "pH excursion (daily min < 6.5 or daily max > 9.0)", "acute", ("ph_min", "ph_max"), "pH units",
             "above", 0.0, "ph_range_excess", onset=OnsetRule("clean_window", pre_days=3, refractory_days=7),
             anchor="EPA (1986) Gold Book freshwater pH range 6.5-9.0"),
    _snap("E6_snap", "pH outside 6.5-9.0 (snapshot)", "chronic", "ph", "pH units", "outside", (6.5, 9.0),
          "EPA (1986) Gold Book freshwater pH range 6.5-9.0", "E6"),
    EventDef("E7", "thermal stress (7DADMax > 20 C)", "chronic", ("temp_c_max",), "degC", "above", 20.0, "7dadm",
             tiers=("E7_t16", "E7_t18"), onset=OnsetRule("hazard", pre_days=7, pre_margin=0.0, pre_gap=6,
                                                          refractory_days=30),
             anchor="EPA 910-B-03-002 (2003) Region 10 temperature guidance, 20 C 7DADM"),
    EventDef("E7_t16", "thermal tier 7DADMax > 16 C", "chronic", ("temp_c_max",), "degC", "above", 16.0, "7dadm",
             onset=OnsetRule("hazard", pre_days=7, pre_margin=0.0, pre_gap=6, refractory_days=30),
             anchor="EPA 910-B-03-002 (2003) core juvenile rearing 16 C", parent="E7"),
    EventDef("E7_t18", "thermal tier 7DADMax > 18 C", "chronic", ("temp_c_max",), "degC", "above", 18.0, "7dadm",
             onset=OnsetRule("hazard", pre_days=7, pre_margin=0.0, pre_gap=6, refractory_days=30),
             anchor="EPA 910-B-03-002 (2003) migration / non-core rearing 18 C", parent="E7"),
    _report("E8", "toxic chemical release reaching water (severity: RQ = PEC / HC5 > 1)", "acute",
            "NRC reporting (33 CFR 153); SSD-HC5 convention (Aldenberg & Jaworska 2000; EU TGD 2003)"),
    _report("E9", "fish kill (agency report)", "acute", "Iowa DNR fish-kill database; DE DNREC fish-kill reports"),
    _snap("E10", "faecal contamination (E. coli > 235 cfu/100 mL)", "acute", "ecoli_cfu_100ml", "cfu/100 mL", "above",
          235.0, "EPA-820-F-12-058 (2012 RWQC) beach action value"),
    _snap("E10_qpcr", "faecal contamination (enterococci qPCR > 1000 CCE/100 mL)", "acute", "entero_cce_100ml",
          "CCE/100 mL", "above", 1000.0, "EPA-820-F-12-058 (2012 RWQC) qPCR beach action value", "E10"),
    _report("E11", "invasive dreissenid establishment (first record in a HUC12)", "chronic", "USGS NAS (D6 MusselGuard)"),
]
EVENTS: dict[str, EventDef] = {e.code: e for e in _EVENTS_LIST}
EVENT_CODES: tuple[str, ...] = tuple(EVENTS)


def get_event(ev: EventDef | str) -> EventDef:
    if isinstance(ev, EventDef):
        return ev
    if ev not in EVENTS:
        raise KeyError(f"unknown event code {ev!r}; known: {', '.join(EVENT_CODES)}")
    return EVENTS[ev]


def _measured(ev: EventDef) -> EventDef:
    if ev.label_source != "measurement" or ev.threshold is None or ev.direction is None:
        raise ValueError(f"{ev.code} is defined by {ev.label_source} labels, not by a measured variable")
    return ev


# ============================================================================ labelling

def derive_series(frame: pd.DataFrame, ev: EventDef | str) -> pd.Series:
    """The series the event threshold applies to, from a daily frame (complete daily index, NaN = missing).

    identity/snapshot → the variable itself; ``30d_mean`` → 30-day trailing mean (≥ 27 observed);
    ``7dadm`` → 7-day trailing mean of the daily maximum (all 7 observed); ``excess_over_bg30`` → value minus
    the median of days d−30…d−1 (≥ 20 observed); ``ph_range_excess`` → max(6.5 − pH_min, pH_max − 9.0)."""
    ev = _measured(get_event(ev))
    if isinstance(frame.index, pd.DatetimeIndex) and len(frame.index) > 1:
        step = np.diff(frame.index.values).astype("timedelta64[D]").astype(int)
        if (step != 1).any():
            raise ValueError("derive_series needs a complete, sorted daily index (reindex missing days to NaN)")
    missing = [v for v in ev.variables if v not in frame.columns]
    if missing:
        raise KeyError(f"{ev.code} needs columns {missing}")
    x = frame[ev.variables[0]].astype(float)
    agg = ev.aggregation
    if agg in ("identity", "snapshot"):
        out = x
    elif agg == "30d_mean":
        out = x.rolling(30, min_periods=27).mean()
    elif agg == "7dadm":
        out = x.rolling(7, min_periods=7).mean()
    elif agg == "excess_over_bg30":
        out = x - x.shift(1).rolling(30, min_periods=20).median()
    elif agg == "ph_range_excess":
        lo, hi = frame[ev.variables[0]].astype(float), frame[ev.variables[1]].astype(float)
        lo_lim, hi_lim = 6.5, 9.0
        out = pd.concat([lo_lim - lo, hi - hi_lim], axis=1).max(axis=1, skipna=False)
    else:
        raise ValueError(f"unknown aggregation {agg!r}")
    return out.rename(ev.code)


def _beyond(x: np.ndarray, direction: str, threshold) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        if direction == "above":
            return x > threshold
        if direction == "at_or_above":
            return x >= threshold
        if direction == "below":
            return x < threshold
        if direction == "outside":
            lo, hi = threshold
            return (x < lo) | (x > hi)
    raise ValueError(f"unknown direction {direction!r}")


def _runs_at_least(mask: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """(days inside runs of ≥ k consecutive True, first day of each such run)."""
    inside = np.zeros(mask.size, bool)
    start = np.zeros(mask.size, bool)
    for s, e in EWS.alarm_episodes(mask):
        if e - s + 1 >= k:
            inside[s:e + 1] = True
            start[s] = True
    return inside, start


def event_state(x, ev: EventDef | str) -> np.ndarray:
    """Event state per row: 1.0 beyond the threshold (for ``duration_days`` > 1: inside a qualifying run),
    0.0 otherwise, NaN where ``x`` is not observed."""
    ev = _measured(get_event(ev))
    x = np.asarray(x, dtype=float)
    obs = np.isfinite(x)
    b = _beyond(x, ev.direction, ev.threshold) & obs
    if ev.duration_days > 1:
        b = _runs_at_least(b, ev.duration_days)[0]
    return np.where(obs, b.astype(float), np.nan)


def _hazard_for(ev: EventDef) -> HZ.Hazard:
    if ev.hazard_key is not None:
        return HZ.HAZARDS[ev.hazard_key]
    r = ev.onset
    if ev.direction not in ("above", "below"):
        raise ValueError(f"{ev.code}: the hazard onset rule needs direction above/below")
    return HZ.Hazard(ev.code, ev.variables[0], ev.direction, float(ev.threshold), r.pre_margin, r.pre_days,
                     r.refractory_days, "", ev.anchor, pre_gap=r.pre_gap)


def event_onsets(x, ev: EventDef | str) -> tuple[np.ndarray, np.ndarray]:
    """(event-day mask, onset mask) for one site's daily series of the derived variable.
    Hazard-backed codes return exactly :func:`hazards.daily_onsets` output."""
    ev = _measured(get_event(ev))
    if ev.onset is None:
        raise ValueError(f"{ev.code} is a snapshot event without an onset definition")
    x = np.asarray(x, dtype=float)
    r = ev.onset
    if r.method == "hazard":
        return HZ.daily_onsets(x, _hazard_for(ev))
    obs = np.isfinite(x)
    beyond = _beyond(x, ev.direction, ev.threshold) & obs
    if r.method == "run_start":
        inside, start = _runs_at_least(beyond, max(ev.duration_days, 1))
        return inside, start
    if r.method == "clean_window":
        safe = obs & ~beyond
        onset = np.zeros(x.size, bool)
        for d in np.flatnonzero(beyond):
            if d < r.pre_days or not safe[d - r.pre_days:d].all():
                continue
            if beyond[max(0, d - r.refractory_days):d].any():
                continue
            onset[d] = True
        return beyond, onset
    raise ValueError(f"unknown onset method {r.method!r}")


def issue_labels(x, ev: EventDef | str, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """(y, valid) for issue day d: y = 1 if an onset occurs in (d, d + horizon]; validity rules of
    :func:`hazards.issue_labels` (with the event's refractory period)."""
    ev = get_event(ev)
    x = np.asarray(x, dtype=float)
    event, onset = event_onsets(x, ev)
    return HZ.issue_labels(x, event, onset, int(horizon), refractory_days=ev.onset.refractory_days)


def coldwater_eligible(dates, temp_c_max, train_block: tuple[str, str] = ("2007-10-01", "2021-09-30"), *,
                       threshold: float = 20.0, max_frac: float = 0.5, min_days: int = 60) -> bool:
    """E7 population rule (plan §1.1): ≤ ``max_frac`` of the site's TRAIN-block Jul–Aug days have
    7DADMax > ``threshold`` (≥ ``min_days`` such days observed). Uses training-block data only."""
    s = pd.Series(np.asarray(temp_c_max, float), index=pd.DatetimeIndex(dates)).asfreq("D")
    d7 = s.rolling(7, min_periods=7).mean()
    lo, hi = (pd.Timestamp(t) for t in train_block)
    sel = d7[(d7.index >= lo) & (d7.index <= hi) & d7.index.month.isin([7, 8])].dropna()
    return bool(sel.size >= min_days and (sel > threshold).mean() <= max_frac)


# ============================================================================ derived route

_SPACES = {"linear": lambda t: t, "log10": np.log10, "ln": np.log}


@dataclass(frozen=True)
class Gaussian:
    mu: np.ndarray
    sigma: np.ndarray
    space: str = "linear"


@dataclass(frozen=True)
class Quantiles:
    levels: np.ndarray          # (K,), strictly increasing in (0, 1), K >= 2
    values: np.ndarray          # (N, K) or (K,)
    space: str = "linear"


@dataclass(frozen=True)
class GaussianMixture:
    weights: np.ndarray         # (N, K) non-negative (normalised per row)
    means: np.ndarray           # (N, K)
    sds: np.ndarray             # (N, K) >= 0
    space: str = "linear"


@dataclass(frozen=True)
class Samples:
    values: np.ndarray          # (N, S) ensemble members / posterior draws
    space: str = "linear"


def _space_threshold(t: float, space: str) -> float:
    if space not in _SPACES:
        raise ValueError(f"unknown space {space!r}")
    if space != "linear" and not t > 0:
        return -np.inf          # every positive quantity exceeds a non-positive threshold
    return float(_SPACES[space](t))


def _gauss_cdf(t: float, mu, sd) -> np.ndarray:
    mu, sd = np.asarray(mu, float), np.asarray(sd, float)
    if np.any(sd[np.isfinite(sd)] < 0):
        raise ValueError("negative standard deviation")
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (t - mu) / sd
        cdf = norm.cdf(z)
    step = (mu <= t).astype(float)                 # sigma = 0: point mass at mu (right-continuous CDF)
    out = np.where(sd == 0, step, cdf)
    return np.where(np.isfinite(mu) & np.isfinite(sd), out, np.nan)


def _quantile_cdf(t: float, levels: np.ndarray, q: np.ndarray) -> np.ndarray:
    """CDF at t from quantile sets: linear interpolation between quantiles, Gaussian tails matched to
    the two outermost quantiles on each side; crossing quantiles are made monotone (cumulative max)."""
    levels = np.asarray(levels, float)
    if levels.ndim != 1 or levels.size < 2 or np.any(np.diff(levels) <= 0) or levels[0] <= 0 or levels[-1] >= 1:
        raise ValueError("quantile levels must be strictly increasing in (0, 1), at least 2")
    q = np.atleast_2d(np.asarray(q, float))
    if q.shape[1] != levels.size:
        raise ValueError(f"values have {q.shape[1]} columns for {levels.size} levels")
    bad = ~np.isfinite(q).all(axis=1)
    q = np.where(bad[:, None], 0.0, q)
    q = np.maximum.accumulate(q, axis=1)
    n, k_ = q.shape
    z = norm.ppf(levels)
    k = (q <= t).sum(axis=1)                       # number of quantiles <= t
    out = np.empty(n)
    lo = k == 0
    hi = k == k_
    mid = ~lo & ~hi
    if mid.any():
        j = k[mid] - 1
        r = np.flatnonzero(mid)
        q0, q1 = q[r, j], q[r, j + 1]
        out[mid] = levels[j] + (levels[j + 1] - levels[j]) * (t - q0) / (q1 - q0)
    if lo.any():
        s = (q[lo, 1] - q[lo, 0]) / (z[1] - z[0])
        with np.errstate(divide="ignore", invalid="ignore"):
            zz = z[0] + (t - q[lo, 0]) / s
        out[lo] = np.where(s > 0, norm.cdf(zz), 0.0)
    if hi.any():
        s = (q[hi, -1] - q[hi, -2]) / (z[-1] - z[-2])
        with np.errstate(divide="ignore", invalid="ignore"):
            zz = z[-1] + (t - q[hi, -1]) / s
        # s > 0: Gaussian tail (equals levels[-1] at t = q_K); s = 0: no spread above q_K → CDF jumps to 1
        degenerate = np.where(t > q[hi, -1], 1.0, levels[-1])
        out[hi] = np.where(s > 0, norm.cdf(zz), degenerate)
    out[bad] = np.nan
    return out


def _cdf(pred, t_raw: float) -> np.ndarray:
    space = getattr(pred, "space", "linear")
    t = _space_threshold(t_raw, space)
    if isinstance(pred, Gaussian):
        if t == -np.inf:
            return np.where(np.isfinite(np.asarray(pred.mu, float)), 0.0, np.nan)
        return _gauss_cdf(t, pred.mu, pred.sigma)
    if isinstance(pred, Quantiles):
        if t == -np.inf:
            v = np.atleast_2d(np.asarray(pred.values, float))
            return np.where(np.isfinite(v).all(axis=1), 0.0, np.nan)
        return _quantile_cdf(t, pred.levels, pred.values)
    if isinstance(pred, GaussianMixture):
        w = np.atleast_2d(np.asarray(pred.weights, float))
        mu = np.atleast_2d(np.asarray(pred.means, float))
        sd = np.atleast_2d(np.asarray(pred.sds, float))
        if not (w.shape == mu.shape == sd.shape):
            raise ValueError("mixture weights, means and sds must have the same shape")
        if np.any(w[np.isfinite(w)] < 0):
            raise ValueError("negative mixture weight")
        tot = w.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            wn = w / tot
        if t == -np.inf:
            comp = np.where(np.isfinite(mu) & np.isfinite(sd), 0.0, np.nan)
        else:
            comp = _gauss_cdf(t, mu, sd)
        out = (wn * comp).sum(axis=1)
        okrow = np.isfinite(wn).all(axis=1) & np.isfinite(comp).all(axis=1) & (tot[:, 0] > 0)
        return np.where(okrow, out, np.nan)
    if isinstance(pred, Samples):
        v = np.atleast_2d(np.asarray(pred.values, float))
        fin = np.isfinite(v)
        n = fin.sum(axis=1)
        with np.errstate(invalid="ignore"):
            le = ((v <= t) & fin).sum(axis=1)
            return np.where(n > 0, le / np.maximum(n, 1), np.nan)
    raise TypeError(f"unsupported predictive distribution {type(pred).__name__}")


def _cdf_left(pred, t_raw: float) -> np.ndarray:
    """P(X < t): equals the CDF for continuous parts; for point masses (sigma = 0, ties in samples) the
    mass at t is excluded."""
    if isinstance(pred, Samples):
        space = pred.space
        t = _space_threshold(t_raw, space)
        v = np.atleast_2d(np.asarray(pred.values, float))
        fin = np.isfinite(v)
        n = fin.sum(axis=1)
        return np.where(n > 0, ((v < t) & fin).sum(axis=1) / np.maximum(n, 1), np.nan)
    if isinstance(pred, Gaussian):
        t = _space_threshold(t_raw, pred.space)
        mu, sd = np.asarray(pred.mu, float), np.asarray(pred.sigma, float)
        base = _gauss_cdf(t, mu, sd) if t != -np.inf else _cdf(pred, t_raw)
        return np.where((sd == 0) & (mu == t), 0.0, base)
    if isinstance(pred, GaussianMixture):
        t = _space_threshold(t_raw, pred.space)
        base = _cdf(pred, t_raw)
        w = np.atleast_2d(np.asarray(pred.weights, float))
        mu = np.atleast_2d(np.asarray(pred.means, float))
        sd = np.atleast_2d(np.asarray(pred.sds, float))
        with np.errstate(invalid="ignore", divide="ignore"):
            mass = (w / w.sum(axis=1, keepdims=True) * ((sd == 0) & (mu == t))).sum(axis=1)
        return base - mass
    return _cdf(pred, t_raw)


def exceedance_probability(pred, ev: EventDef | str) -> np.ndarray:
    """Route (i): P(variable beyond the event threshold) from a predictive distribution.

    ``pred`` is :class:`Gaussian`, :class:`Quantiles`, :class:`GaussianMixture` or :class:`Samples`, in the
    space given by its ``space`` field (``"linear"``, ``"log10"``, ``"ln"``). ``above`` → P(X > t),
    ``at_or_above`` → P(X ≥ t), ``below`` → P(X < t), ``outside`` → P(X < lo) + P(X > hi).
    NaN wherever the distribution is undefined. The distribution must describe the event's derived
    variable (e.g. the 7DADMax for E7, the 30-day mean for E1b)."""
    ev = _measured(get_event(ev))
    d, t = ev.direction, ev.threshold
    if d == "above":
        p = 1.0 - _cdf(pred, float(t))
    elif d == "at_or_above":
        p = 1.0 - _cdf_left(pred, float(t))
    elif d == "below":
        p = _cdf_left(pred, float(t))
    elif d == "outside":
        lo, hi = t
        p = _cdf_left(pred, float(lo)) + (1.0 - _cdf(pred, float(hi)))
    else:
        raise ValueError(f"unknown direction {d!r}")
    return np.clip(np.asarray(p, float), 0.0, 1.0)


@dataclass
class Calibrator:
    """Recalibration map fitted on validation rows only (callers pass validation rows; nothing here can
    see a split). ``method`` = "platt" (logistic on logit p) or "isotonic" (stored step function)."""
    method: str
    params: dict = field(default_factory=dict)
    n_fit: int = 0
    n_pos_fit: int = 0
    fit_sha256: str = ""

    def __call__(self, p) -> np.ndarray:
        return self.transform(p)

    def transform(self, p) -> np.ndarray:
        p = np.asarray(p, float)
        out = np.full(p.shape, np.nan)
        ok = np.isfinite(p)
        if self.method == "platt":
            lp = _logit(p[ok])
            out[ok] = 1.0 / (1.0 + np.exp(-(self.params["a"] * lp + self.params["b"])))
        elif self.method == "isotonic":
            out[ok] = np.interp(p[ok], self.params["x"], self.params["y"])
        else:
            raise ValueError(self.method)
        return out


_EPS = 1e-6


def _logit(p):
    p = np.clip(np.asarray(p, float), _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def _platt_fit(x: np.ndarray, y: np.ndarray, ridge: float = 1e-3, max_iter: int = 200) -> tuple[float, float]:
    """Newton–Raphson for P(y=1) = σ(a·x + b) with a tiny ridge (deterministic; finite under separation)."""
    a, b = 1.0, 0.0
    X = np.column_stack([x, np.ones_like(x)])
    w = np.array([a, b])
    for _ in range(max_iter):
        eta = np.clip(X @ w, -35, 35)
        mu = 1.0 / (1.0 + np.exp(-eta))
        g = X.T @ (y - mu) - ridge * w
        W = mu * (1 - mu)
        H = -(X.T * W) @ X - ridge * np.eye(2)
        step = np.linalg.solve(H, g)
        w = w - step
        if np.max(np.abs(step)) < 1e-10:
            break
    return float(w[0]), float(w[1])


def recalibrate(p_val, y_val, *, method: str = "platt") -> Calibrator:
    """Fit a recalibration map on VALIDATION predictions/labels (rows with finite p and y only).
    Refuses inputs without both classes."""
    p = np.asarray(p_val, float).ravel()
    y = np.asarray(y_val, float).ravel()
    if p.shape != y.shape:
        raise ValueError("p_val and y_val must have the same length")
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("y_val must be 0/1")
    if p.size == 0 or y.min() == y.max():
        raise ValueError("recalibration needs validation rows of both classes")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("p_val must lie in [0, 1]")
    sha = hashlib.sha256(np.concatenate([p, y]).astype("<f8").tobytes()).hexdigest()
    if method == "platt":
        a, b = _platt_fit(_logit(p), y)
        params = {"a": a, "b": b}
    elif method == "isotonic":
        from sklearn.isotonic import IsotonicRegression

        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p, y)
        params = {"x": np.asarray(iso.X_thresholds_, float), "y": np.asarray(iso.y_thresholds_, float)}
    else:
        raise ValueError(f"unknown method {method!r}")
    return Calibrator(method, params, int(p.size), int(y.sum()), sha)


# ============================================================================ metrics

def power_flag(n_pos_val: int, n_clusters_val: int, min_pos: int = 30, min_clusters: int = 10) -> str:
    """Plan §1.2: 'underpowered' if the validation cell has < ``min_pos`` positives or < ``min_clusters``
    clusters carrying positives; otherwise 'feasible'."""
    return "underpowered" if (int(n_pos_val) < min_pos or int(n_clusters_val) < min_clusters) else "feasible"


def _binary_rows(y, p, *extra):
    y = np.asarray(y, float).ravel()
    p = np.asarray(p, float).ravel()
    arrs = [np.asarray(e, float).ravel() if e is not None else None for e in extra]
    for a in [p] + [a for a in arrs if a is not None]:
        if a.shape != y.shape:
            raise ValueError("all arrays must have the same length")
    ok = np.isfinite(y) & np.isfinite(p)
    for a in arrs:
        if a is not None:
            ok &= np.isfinite(a)
    if not np.isin(y[ok], (0.0, 1.0)).all():
        raise ValueError("y must be 0/1 (NaN = unlabelled)")
    if np.any((p[ok] < 0) | (p[ok] > 1)):
        raise ValueError("probabilities must lie in [0, 1]")
    return ok, y, p, arrs


def _ci(metric, *arrays, groups, n_boot, seed) -> dict:
    point = float(metric(*arrays))
    n_groups = len(np.unique(groups)) if groups is not None else len(arrays[0])
    if n_groups < 2 or not np.isfinite(point):
        return {"point": point, "ci_low": np.nan, "ci_high": np.nan, "n_boot_valid": 0,
                "note": "CI undefined (fewer than 2 resampling units or undefined point estimate)"}
    r = B.bootstrap_ci(metric, *arrays, groups=groups, n_boot=n_boot, seed=seed)
    return {k: r[k] for k in ("point", "ci_low", "ci_high", "n_boot_valid")}


def state_metrics(y, p, *, groups, p_ref=None, n_boot: int = 2000, seed: int = 0) -> dict:
    """State-classification metrics (plan §1.2): AUPRC (chance = prevalence) and Brier score with
    cluster-bootstrap 95% CIs over ``groups`` (``None`` → row bootstrap), BSS vs ``p_ref`` when given,
    ECE, counts and the power flag. Rows with NaN label/probability (or NaN ``p_ref``) are dropped."""
    if p_ref is not None:          # scalar climatology or per-row reference
        p_ref = np.broadcast_to(np.asarray(p_ref, float), np.asarray(y).ravel().shape)
    ok, y, p, (ref,) = _binary_rows(y, p, p_ref)
    g = None if groups is None else np.asarray(groups).ravel()
    if g is not None and g.shape != y.shape:
        raise ValueError("groups must have the same length as y")
    y, p = y[ok], p[ok]
    g = None if g is None else g[ok]
    ref = None if ref is None else ref[ok]
    n, n_pos = int(y.size), int(y.sum()) if y.size else 0
    n_groups = len(np.unique(g)) if g is not None else n
    n_groups_pos = len(np.unique(g[y == 1])) if g is not None else n_pos
    out = {"n": n, "n_pos": n_pos, "prevalence": float(y.mean()) if n else np.nan, "n_groups": n_groups,
           "n_groups_with_pos": n_groups_pos, "unit": "group" if g is not None else "row",
           "power_flag": power_flag(n_pos, n_groups_pos)}
    nan_ci = {"point": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n_boot_valid": 0}
    if n == 0:
        out.update(auprc=dict(nan_ci), brier=dict(nan_ci), ece=np.nan)
        if ref is not None:
            out["bss"] = dict(nan_ci)
        return out
    out["auprc"] = _ci(M.auprc, y, p, groups=g, n_boot=n_boot, seed=seed)
    out["brier"] = _ci(M.brier, y, p, groups=g, n_boot=n_boot, seed=seed)
    out["ece"] = M.ece(y, p)
    if ref is not None:
        out["brier_ref"] = M.brier(y, ref)
        out["bss"] = _ci(M.brier_skill_score, y, p, ref, groups=g, n_boot=n_boot, seed=seed)
    return out


def compare_routes(y, p_derived, p_direct, *, groups, n_boot: int = 2000, seed: int = 0) -> dict:
    """Paired comparison derived − direct on identical rows (plan §2.2): ΔBrier (negative favours the
    derived route) and ΔAUPRC (positive favours derived), paired cluster bootstrap over ``groups``."""
    ok, y, pa, (pb,) = _binary_rows(y, p_derived, p_direct)
    if np.any((pb[ok] < 0) | (pb[ok] > 1)):
        raise ValueError("probabilities must lie in [0, 1]")
    g = None if groups is None else np.asarray(groups).ravel()[ok]
    y, pa, pb = y[ok], pa[ok], pb[ok]
    out = {"n": int(y.size), "n_pos": int(y.sum()) if y.size else 0,
           "brier_derived": M.brier(y, pa), "brier_direct": M.brier(y, pb),
           "auprc_derived": M.auprc(y, pa), "auprc_direct": M.auprc(y, pb)}
    n_groups = len(np.unique(g)) if g is not None else y.size
    for name, metric, hib in (("delta_brier", M.brier, False), ("delta_auprc", M.auprc, True)):
        if n_groups < 2 or y.size == 0:
            out[name] = {"diff": float(metric(y, pa) - metric(y, pb)) if y.size else np.nan,
                         "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan,
                         "note": "CI undefined (fewer than 2 resampling units)"}
        else:
            out[name] = B.paired_bootstrap(metric, y, pa, pb, groups=g, n_boot=n_boot, seed=seed,
                                           higher_is_better=hib)
    return out


def onset_metrics(issue, score, usable, onsets, *, threshold: float, h_days: int, site,
                  clusters: Mapping | None = None, n_boot: int = 2000, seed: int = 0) -> dict:
    """Onset-detection metrics (plan §1.2) on long-format site-day rows.

    ``issue`` dates, ``score`` (alarm when usable, finite and ≥ ``threshold``), ``usable`` issue rows,
    ``onsets`` (bool, onset day of the event on that row) and ``site`` are aligned arrays; each site is
    reindexed to a complete daily range (added days are unusable). Definitions (EW-v2 machinery):
    * evaluable onset: ≥ 1 usable issue row in [onset − h, onset) (``ew_scoring.evaluable_onsets``);
    * warned (hit): an alarm episode starts in [onset − h, onset); hit rate = warned / evaluable;
    * lead time: onset − earliest such episode start (days);
    * false alarm: an episode with no onset in [start, start + h] (``ew_scoring.episode_outcomes``);
      false alarms per site-year = false episodes / (usable issue days / 365.25).
    CIs: cluster bootstrap over sites grouped by ``clusters`` (site → cluster; default: site)."""
    df = pd.DataFrame({"t": pd.to_datetime(np.asarray(issue)).normalize(), "score": np.asarray(score, float),
                       "usable": np.asarray(usable, bool), "onset": np.asarray(onsets, bool),
                       "site": np.asarray(site).astype(str)})
    if df.duplicated(["site", "t"]).any():
        raise ValueError("duplicate (site, date) rows")
    h = int(h_days)
    per, leads = [], []
    for s, f in df.groupby("site", sort=True):
        f = f.set_index("t").sort_index()
        dates = pd.date_range(f.index[0], f.index[-1], freq="D")
        use = f["usable"].reindex(dates, fill_value=False).to_numpy(bool)
        sc = f["score"].reindex(dates).to_numpy(float)
        use_s = use & np.isfinite(sc)
        with np.errstate(invalid="ignore"):
            alarm = use_s & (sc >= threshold)
        on_mask = f["onset"].reindex(dates, fill_value=False).to_numpy(bool)
        frames = {s: pd.DataFrame({"usable": use, "score": sc}, index=dates)}
        evs = {s: EWS.daily_events(s, dates, on_mask, on_mask)}
        ev_ok = EWS.evaluable_onsets(frames, evs, h).get(s, [])
        eps = EWS.alarm_episodes(alarm)
        starts = dates.to_numpy()[[a for a, _ in eps]] if eps else np.array([], "datetime64[ns]")
        hd = np.timedelta64(h, "D")
        warned = 0
        for e in ev_ok:
            o = e.onset.to_datetime64()
            w = starts[(starts >= o - hd) & (starts < o)]
            if w.size:
                warned += 1
                leads.append(int((o - w.min()) / np.timedelta64(1, "D")))
        fa = EWS.episode_outcomes(dates, alarm, dates[on_mask], h)
        per.append({"site": s, "n_onsets": len(ev_ok), "n_warned": warned, "n_episodes": fa["n_episodes"],
                    "n_false": fa["n_false_episodes"], "site_years": float(use_s.sum()) / 365.25})
    P = pd.DataFrame(per, columns=["site", "n_onsets", "n_warned", "n_episodes", "n_false", "site_years"])
    cmap = {str(k): v for k, v in (clusters or {}).items()}
    grp = P["site"].map(lambda s: cmap.get(s, s)).to_numpy() if len(P) else np.array([])

    def ratio(a, b):
        sb = float(np.sum(b))
        return float(np.sum(a) / sb) if sb > 0 else np.nan

    lt = np.asarray(leads, float)
    out = {"n_sites": len(P), "n_clusters": len(np.unique(grp)) if len(P) else 0,
           "n_onsets_evaluable": int(P["n_onsets"].sum()), "n_onsets_warned": int(P["n_warned"].sum()),
           "n_alarm_episodes": int(P["n_episodes"].sum()), "n_false_episodes": int(P["n_false"].sum()),
           "site_years": float(P["site_years"].sum()), "threshold": float(threshold), "h_days": h,
           "lead_time_days_median": float(np.median(lt)) if lt.size else np.nan,
           "lead_time_days_q25": float(np.quantile(lt, 0.25)) if lt.size else np.nan,
           "lead_time_days_q75": float(np.quantile(lt, 0.75)) if lt.size else np.nan,
           "lead_times_days": lt.tolist()}
    if len(P):
        out["hit_rate"] = _ci(ratio, P["n_warned"].to_numpy(float), P["n_onsets"].to_numpy(float),
                              groups=grp, n_boot=n_boot, seed=seed)
        out["false_alarms_per_site_year"] = _ci(ratio, P["n_false"].to_numpy(float), P["site_years"].to_numpy(float),
                                                groups=grp, n_boot=n_boot, seed=seed)
    else:
        nan_ci = {"point": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n_boot_valid": 0}
        out["hit_rate"], out["false_alarms_per_site_year"] = dict(nan_ci), dict(nan_ci)
    return out


# ============================================================================ external labels

@dataclass(frozen=True)
class MatchRule:
    """Spatial/temporal rule for matching external event reports to model sites (plan §3.3).
    ``window_days`` = (lo, hi) bounds of event date − site ``date`` (used when ``sites`` has a ``date``
    column, e.g. sample or image dates); ``require_record_span`` keeps a site only if its ``first``..``last``
    record spans the event date; ``same_lake`` requires equal ``lake_id`` columns."""
    radius_km: float
    window_days: tuple[int, int] = (0, 0)
    require_record_span: bool = False
    same_lake: bool = False


def match_external(events: pd.DataFrame, sites: pd.DataFrame, rule: MatchRule) -> pd.DataFrame:
    """All (event, site) pairs satisfying ``rule``. ``events``: event_id, lat, lon, date [, lake_id];
    ``sites``: site_id, lat, lon [, first, last][, date][, lake_id][, cluster]. Straight-line
    (haversine) distance. Returns event_id, site_id, distance_km [, lag_days][, cluster]."""
    from sklearn.neighbors import BallTree

    cols = ["event_id", "site_id", "distance_km", "lag_days", "cluster"]
    ev = events.reset_index(drop=True)
    st = sites.reset_index(drop=True)
    eok = np.isfinite(ev["lat"].astype(float)) & np.isfinite(ev["lon"].astype(float))
    sok = np.isfinite(st["lat"].astype(float)) & np.isfinite(st["lon"].astype(float))
    if not eok.any() or not sok.any():
        return pd.DataFrame(columns=cols)
    st = st[sok.to_numpy()].reset_index(drop=True)
    tree = BallTree(np.radians(st[["lat", "lon"]].to_numpy(float)), metric="haversine")
    ev = ev[eok.to_numpy()].reset_index(drop=True)
    ind, dist = tree.query_radius(np.radians(ev[["lat", "lon"]].to_numpy(float)), r=rule.radius_km / 6371.0088,
                                  return_distance=True)
    rows = []
    edate = pd.to_datetime(ev["date"]) if "date" in ev else None
    for i, (ii, dd) in enumerate(zip(ind, dist)):
        for j, d in zip(ii.tolist(), dd.tolist()):
            r = {"event_id": ev.at[i, "event_id"], "site_id": st.at[j, "site_id"], "distance_km": d * 6371.0088,
                 "lag_days": np.nan, "cluster": st.at[j, "cluster"] if "cluster" in st else None}
            if rule.same_lake and (ev.at[i, "lake_id"] != st.at[j, "lake_id"] or pd.isna(ev.at[i, "lake_id"])):
                continue
            if rule.require_record_span:
                t = edate[i]
                if not (pd.Timestamp(st.at[j, "first"]) <= t <= pd.Timestamp(st.at[j, "last"])):
                    continue
            if "date" in st:
                lag = (edate[i] - pd.Timestamp(st.at[j, "date"])) / pd.Timedelta(days=1)
                if not (rule.window_days[0] <= lag <= rule.window_days[1]):
                    continue
                r["lag_days"] = float(lag)
            rows.append(r)
    return pd.DataFrame(rows, columns=cols)
