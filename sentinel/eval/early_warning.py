# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Early-warning evaluation protocol v2 ("EW-v2").

Why this exists: the v1 detector reported "25/32 events detected at <=5% FPR", but it
alarmed before 95–99% of randomly chosen *clean* dates as well (placebo failure), and its
4.6% FPR was calibrated and measured on the same windows. EW-v2 makes those failures
impossible to miss:

1. Events are defined from observations only (threshold, minimum duration, merge gap),
   with onsets flagged as censored when the onset falls in/after a data gap.
2. The target is *onset within the next h* at regular issue times; issue times inside an
   ongoing event are excluded (detecting an event that has already started is not warning).
3. Alarm thresholds are calibrated on a calibration period / site set that is disjoint
   from the test data, at a stated false-alarm rate (POFD per issue time).
4. Event-level detection uses only alarms issued strictly before onset, within h.
5. Every detection rate is reported next to a placebo detection rate computed with the
   identical rule at pseudo-onsets sampled from clean periods (season-matched), with a
   one-sided Fisher exact test. A system "works" only if real >> placebo.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from . import metrics as M


@dataclass
class Event:
    site: str
    onset: pd.Timestamp
    end: pd.Timestamp
    peak_time: pd.Timestamp
    peak_value: float
    duration_h: float
    onset_censored: bool

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("onset", "end", "peak_time"):
            d[k] = str(d[k])
        return d


def extract_events(series: pd.Series, threshold: float, *, direction: str = "below",
                   min_duration: str = "2h", merge_gap: str = "12h", max_gap: str = "3h",
                   site: str = "") -> list[Event]:
    """Threshold-exceedance events in an observed series (NaN = missing).

    A run of exceeding samples breaks when consecutive samples are more than ``max_gap``
    apart; runs separated by <= ``merge_gap`` are merged; runs shorter than
    ``min_duration`` are dropped. ``onset_censored`` is True when the previous valid
    observation before onset is missing or more than ``max_gap`` earlier.
    """
    if direction not in ("below", "above"):
        raise ValueError("direction must be 'below' or 'above'")
    s = series.dropna().sort_index()
    if s.empty:
        return []
    min_duration, merge_gap, max_gap = map(pd.Timedelta, (min_duration, merge_gap, max_gap))
    vals = s.to_numpy(float)
    times = s.index
    cond = vals < threshold if direction == "below" else vals > threshold
    dt_med = pd.Series(times).diff().median()
    dt_med = dt_med if pd.notna(dt_med) else pd.Timedelta(0)

    runs: list[list[int]] = []
    i, n = 0, len(cond)
    while i < n:
        if cond[i]:
            j = i
            while j + 1 < n and cond[j + 1] and (times[j + 1] - times[j]) <= max_gap:
                j += 1
            runs.append([i, j])
            i = j + 1
        else:
            i += 1
    merged: list[list[int]] = []
    for r in runs:
        if merged and (times[r[0]] - times[merged[-1][1]]) <= merge_gap:
            merged[-1][1] = r[1]
        else:
            merged.append(list(r))

    events = []
    for i0, j0 in merged:
        dur = times[j0] - times[i0] + dt_med
        if dur < min_duration:
            continue
        seg = s.iloc[i0:j0 + 1]
        peak_t = seg.idxmin() if direction == "below" else seg.idxmax()
        censored = i0 == 0 or (times[i0] - times[i0 - 1]) > max_gap
        events.append(Event(site, times[i0], times[j0], peak_t, float(seg.loc[peak_t]),
                            dur.total_seconds() / 3600.0, bool(censored)))
    return events


def make_issue_times(start, end, freq: str = "1D") -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(start).ceil(freq), pd.Timestamp(end), freq=freq)


def onset_labels(issue: pd.DatetimeIndex, events: list[Event], horizon: str, *,
                 series: pd.Series | None = None, obs_freq: str | None = None,
                 min_coverage: float = 0.5, refractory: str | None = "24h") -> pd.DataFrame:
    """Label each issue time t: y=1 if an (uncensored) event onset lies in (t, t+h].

    valid=False when t is inside an event, inside the refractory period after an event,
    in the pre-onset window of a censored event, or (for y=0) when fewer than
    ``min_coverage`` of the expected observations in (t, t+h] exist (unknown outcome).
    """
    h = pd.Timedelta(horizon)
    t = issue.to_numpy()
    y = np.zeros(len(issue), dtype=int)
    in_event = np.zeros(len(issue), dtype=bool)
    unknown = np.zeros(len(issue), dtype=bool)
    refr = pd.Timedelta(refractory) if refractory else None
    for ev in events:
        on, en = ev.onset.to_datetime64(), ev.end.to_datetime64()
        pre = (t >= (ev.onset - h).to_datetime64()) & (t < on)
        if ev.onset_censored:
            unknown |= pre
        else:
            y[pre] = 1
        in_event |= (t >= on) & (t <= en)
        if refr is not None:
            unknown |= (t > en) & (t <= (ev.end + refr).to_datetime64())
    coverage = np.full(len(issue), np.nan)
    if series is not None:
        obs = series.dropna().index.sort_values().to_numpy()
        lo = np.searchsorted(obs, t, side="right")
        hi = np.searchsorted(obs, (issue + h).to_numpy(), side="right")
        freq = pd.Timedelta(obs_freq) if obs_freq else pd.Series(obs).diff().median()
        expected = max(h / freq, 1.0) if pd.notna(freq) and freq > pd.Timedelta(0) else 1.0
        coverage = np.minimum((hi - lo) / expected, 1.0)
    low_cov = np.isfinite(coverage) & (coverage < min_coverage) & (y == 0)
    valid = ~in_event & ~unknown & ~low_cov
    return pd.DataFrame({"y": y, "valid": valid, "in_event": in_event, "coverage": coverage},
                        index=issue)


def calibrate_threshold(scores, y, valid, target_pofd: float) -> float:
    """Smallest threshold whose false-alarm rate on valid negatives is <= target_pofd.

    Alarm rule everywhere in EW-v2: ``score >= threshold``.
    """
    scores, y, valid = np.asarray(scores, float), np.asarray(y), np.asarray(valid, bool)
    neg = np.sort(scores[valid & (y == 0) & np.isfinite(scores)])
    if neg.size == 0:
        raise ValueError("no valid negatives to calibrate on")
    u = np.unique(neg)
    frac = (neg.size - np.searchsorted(neg, u, side="left")) / neg.size
    ok = u[frac <= target_pofd]
    return float(ok.min()) if ok.size else float(np.nextafter(neg.max(), np.inf))


def _alarm_windows(issue_t: np.ndarray, alarm: np.ndarray, onsets: np.ndarray, h: np.timedelta64):
    """For each onset, the issue indices in [onset - h, onset) and which of them alarm."""
    lo = np.searchsorted(issue_t, onsets - h, side="left")
    hi = np.searchsorted(issue_t, onsets, side="left")  # strictly before onset
    return lo, hi


def event_detection(issue: pd.DatetimeIndex, scores, usable, events: list[Event], threshold: float,
                    horizon: str) -> pd.DataFrame:
    """Per-event detection using only alarms issued strictly before onset within the horizon.

    ``usable`` marks issue times where the model had its inputs (e.g. enough history).
    Censored events and events with no usable issue time in their window are reported
    with ``evaluable=False`` and excluded from rates.
    """
    t = issue.to_numpy()
    s = np.asarray(scores, float)
    alarm = np.asarray(usable, bool) & np.isfinite(s) & (s >= threshold)
    h = pd.Timedelta(horizon).to_timedelta64()
    rows = []
    for ev in events:
        on = ev.onset.to_datetime64()
        lo, hi = _alarm_windows(t, alarm, np.array([on]), h)
        lo, hi = int(lo[0]), int(hi[0])
        n_usable = int(np.asarray(usable, bool)[lo:hi].sum())
        hits = np.flatnonzero(alarm[lo:hi]) + lo
        rows.append({
            "site": ev.site, "onset": ev.onset, "censored": ev.onset_censored,
            "evaluable": (not ev.onset_censored) and n_usable > 0,
            "n_usable_issues": n_usable, "detected": bool(hits.size),
            "lead_first_h": (on - t[hits[0]]) / np.timedelta64(1, "h") if hits.size else np.nan,
            "lead_last_h": (on - t[hits[-1]]) / np.timedelta64(1, "h") if hits.size else np.nan,
        })
    return pd.DataFrame(rows)


def sample_placebo_onsets(issue: pd.DatetimeIndex, usable, events: list[Event], *, n: int,
                          horizon: str, exclusion: str = "14D", seed: int = 0,
                          match_months_of: list[pd.Timestamp] | None = None) -> pd.DatetimeIndex:
    """Pseudo-onsets at clean times: no real event (onset−h−exclusion … end+exclusion) nearby,
    and at least one usable issue time in the preceding horizon window. If
    ``match_months_of`` is given, sampling weights match that month-of-year distribution."""
    h = pd.Timedelta(horizon)
    ex = pd.Timedelta(exclusion)
    t = issue
    ok = np.ones(len(t), dtype=bool)
    for ev in events:
        ok &= ~((t >= ev.onset - h - ex) & (t <= ev.end + ex))
    use = np.asarray(usable, bool)
    csum = np.concatenate([[0], np.cumsum(use)])
    lo = np.searchsorted(t.to_numpy(), (t - h).to_numpy(), side="left")
    hi = np.arange(len(t))
    ok &= (csum[hi] - csum[lo]) > 0
    cand = np.flatnonzero(ok)
    if cand.size == 0:
        return pd.DatetimeIndex([])
    rs = np.random.default_rng(seed)
    w = np.ones(cand.size)
    if match_months_of:
        months = pd.Series([pd.Timestamp(x).month for x in match_months_of])
        target = months.value_counts(normalize=True)
        cm = t[cand].month
        avail = pd.Series(cm).value_counts(normalize=True)
        w = np.array([target.get(m, 0.0) / avail.get(m, np.inf) for m in cm])
        if w.sum() == 0:
            w = np.ones(cand.size)
    w = w / w.sum()
    pick = rs.choice(cand, size=min(n, int(np.count_nonzero(w))), replace=False, p=w)
    return pd.DatetimeIndex(np.sort(t[pick]))


def placebo_detection(issue: pd.DatetimeIndex, scores, usable, placebo_onsets: pd.DatetimeIndex,
                      threshold: float, horizon: str, site: str = "") -> pd.DataFrame:
    fake = [Event(site, pd.Timestamp(p), pd.Timestamp(p), pd.Timestamp(p), np.nan, 0.0, False)
            for p in placebo_onsets]
    return event_detection(issue, scores, usable, fake, threshold, horizon)


def fisher_greater(real_hits: int, real_n: int, placebo_hits: int, placebo_n: int) -> float:
    """One-sided Fisher exact p-value that the real detection rate exceeds the placebo rate."""
    from scipy.stats import fisher_exact

    if real_n == 0 or placebo_n == 0:
        return np.nan
    table = [[real_hits, real_n - real_hits], [placebo_hits, placebo_n - placebo_hits]]
    return float(fisher_exact(table, alternative="greater").pvalue)


def evaluate_site_set(frames: dict[str, pd.DataFrame], events: dict[str, list[Event]], *,
                      horizon: str, threshold: float, prob_ref: float | None = None,
                      n_placebo_per_site: int = 50, exclusion: str = "14D",
                      seed: int = 0) -> dict:
    """Evaluate one test set (already disjoint from calibration) across sites.

    ``frames[site]`` is indexed by issue time with columns: score, y, valid, usable.
    Returns time-level skill, contingency scores at the given threshold, event-level
    detection with lead times, and the placebo comparison.
    """
    all_scores, all_y = [], []
    det_rows, plc_rows = [], []
    for i, (site, f) in enumerate(sorted(frames.items())):
        f = f.sort_index()
        v = f["valid"].to_numpy(bool) & f["usable"].to_numpy(bool) & np.isfinite(f["score"].to_numpy(float))
        all_scores.append(f["score"].to_numpy(float)[v])
        all_y.append(f["y"].to_numpy(int)[v])
        evs = [e for e in events.get(site, []) if f.index.min() <= e.onset <= f.index.max()]
        d = event_detection(f.index, f["score"], f["usable"], evs, threshold, horizon)
        if not d.empty:
            det_rows.append(d)
        po = sample_placebo_onsets(f.index, f["usable"], events.get(site, []), n=n_placebo_per_site,
                                   horizon=horizon, exclusion=exclusion, seed=seed + i,
                                   match_months_of=[e.onset for e in evs] or None)
        p = placebo_detection(f.index, f["score"], f["usable"], po, threshold, horizon, site)
        if not p.empty:
            plc_rows.append(p)
    s = np.concatenate(all_scores) if all_scores else np.array([])
    y = np.concatenate(all_y) if all_y else np.array([])
    out: dict = {"horizon": horizon, "threshold": threshold, "n_issue": int(y.size),
                 "n_pos_issue": int(y.sum()) if y.size else 0}
    out["auroc"] = M.auroc(y, s)
    out["auprc"] = M.auprc(y, s)
    out["prevalence"] = float(y.mean()) if y.size else np.nan
    if prob_ref is not None and s.size and np.nanmin(s) >= 0 and np.nanmax(s) <= 1:
        out["brier"] = M.brier(y, s)
        out["bss_vs_climatology"] = M.brier_skill_score(y, s, prob_ref)
    out.update({f"issue_{k}": v for k, v in M.contingency_scores(y, s >= threshold).items()})
    det = pd.concat(det_rows, ignore_index=True) if det_rows else pd.DataFrame()
    plc = pd.concat(plc_rows, ignore_index=True) if plc_rows else pd.DataFrame()
    ev_ok = det[det["evaluable"]] if not det.empty else det
    pl_ok = plc[plc["evaluable"]] if not plc.empty else plc
    rh, rn = (int(ev_ok["detected"].sum()), len(ev_ok)) if not ev_ok.empty else (0, 0)
    ph, pn = (int(pl_ok["detected"].sum()), len(pl_ok)) if not pl_ok.empty else (0, 0)
    out.update({
        "events_evaluable": rn, "events_detected": rh,
        "event_detection_rate": rh / rn if rn else np.nan,
        "placebo_n": pn, "placebo_detected": ph,
        "placebo_detection_rate": ph / pn if pn else np.nan,
        "placebo_fisher_p": fisher_greater(rh, rn, ph, pn),
        "median_lead_first_h": float(ev_ok.loc[ev_ok["detected"], "lead_first_h"].median())
        if rh else np.nan,
        "median_lead_last_h": float(ev_ok.loc[ev_ok["detected"], "lead_last_h"].median())
        if rh else np.nan,
    })
    out["detection_lift_over_placebo"] = (out["event_detection_rate"] / out["placebo_detection_rate"]
                                          if pn and ph else np.nan)
    out["_event_table"] = det
    out["_placebo_table"] = plc
    return out
