# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""EW-v2 scoring helpers on top of :mod:`sentinel.eval.early_warning` (daily issue tier).

* **Primary alarm budget (protocol §6, 2026-10-22 deviation): ALARM DAYS.** Thresholds are chosen on
  the calibration (val) block with ``early_warning.calibrate_threshold(score, y, valid,
  target_pofd=budget/30)`` over valid issue rows, i.e. alarm-days on valid NON-ONSET issue days
  ≤ ``budget`` per site per 30 days pooled (POFD ≤ 1/30; sensitivity 2/30). Every alarmed day
  counts; :func:`alarm_day_stats` reports the realised pooled and per-site rates and time-in-alarm.
* Brier score / BSS against a reference forecast (site × DOY climatology) with a site-month
  block-bootstrap CI (resampling unit: site-month, RESEARCH_PLAN §12.6).
* Episode accounting (maximal runs, optionally split into lead-length chunks) is kept ONLY as a
  labelled secondary descriptive: an episode budget can be met by a standing alarm.
* Placebo comparison: K season-matched pseudo-onsets per real onset, drawn only from clean days on
  which a real onset could have started (candidate mask) and away from ALL event periods; identical
  detection rule (alarm strictly before onset within h). H-c test = basin-group cluster bootstrap of
  POD_real − POD_placebo (``ew_controls.placebo_cluster``); the Fisher test (independent placebos,
  anti-conservative) is kept as a descriptive only.
* Lead-time convention: issue date d uses day d's complete daily statistics, so a forecast exists
  at the end of day d at the earliest; an alarm at d warns ≥ 24 h before an onset on day o only if
  d ≤ o − 2 (``pod_ge24h`` counts detections with first alarm ≥ 48 h before the onset date).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import early_warning as EW
from . import metrics as M
from .bootstrap import bootstrap_ci


def site_month_groups(site, dates) -> np.ndarray:
    d = pd.to_datetime(np.asarray(dates))
    return np.char.add(np.asarray(site).astype(str), d.strftime("|%Y-%m").to_numpy().astype(str))


def two_stage_bootstrap(metric, *arrays, outer, inner, n_boot: int = 2000, alpha: float = 0.05,
                        seed: int = 0) -> dict:
    """Hierarchical cluster bootstrap: resample OUTER groups (e.g. basin groups) with replacement,
    then, inside each drawn outer group, resample its INNER blocks (e.g. site-months) with
    replacement. Percentile CI of ``metric(*arrays)``."""
    arrays = [np.asarray(a) for a in arrays]
    outer, inner = np.asarray(outer), np.asarray(inner)
    og = pd.Series(np.arange(outer.size)).groupby([outer, inner]).apply(lambda x: x.to_numpy())
    blocks = {}
    for (o, _), idx in og.items():
        blocks.setdefault(o, []).append(idx)
    keys = list(blocks)
    rs = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        parts = []
        for o in rs.integers(0, len(keys), len(keys)):
            bl = blocks[keys[o]]
            parts += [bl[i] for i in rs.integers(0, len(bl), len(bl))]
        idx = np.concatenate(parts)
        vals[b] = metric(*(a[idx] for a in arrays))
    ok = vals[np.isfinite(vals)]
    return {"point": float(metric(*arrays)), "ci_low": float(np.quantile(ok, alpha / 2)) if ok.size else np.nan,
            "ci_high": float(np.quantile(ok, 1 - alpha / 2)) if ok.size else np.nan,
            "n_outer": len(keys), "n_inner": int(len(og)), "n_boot_valid": int(ok.size)}


def two_stage_block_sums(sums: np.ndarray, outer: np.ndarray, *, n_boot: int = 2000, seed: int = 0,
                         two_stage: bool = True) -> np.ndarray:
    """Fast hierarchical bootstrap for additive statistics. ``sums`` (n_blocks, k) holds per-inner-
    block column sums (e.g. n, Σ(p−y)², Σ(p_ref−y)²); ``outer`` the outer group of each block.
    Returns (n_boot, k) resampled column totals: outer groups drawn with replacement, then blocks
    within each drawn group (``two_stage``); with ``two_stage=False`` blocks are resampled directly.
    Exactly equivalent to resampling the rows of whole blocks."""
    sums = np.asarray(sums, float)
    rs = np.random.default_rng(seed)
    out = np.zeros((n_boot, sums.shape[1]))
    if not two_stage:
        nb = sums.shape[0]
        for b in range(n_boot):
            out[b] = sums[rs.integers(0, nb, nb)].sum(axis=0)
        return out
    outer = np.asarray(outer)
    keys, inv = np.unique(outer, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(keys.size)]
    for b in range(n_boot):
        tot = np.zeros(sums.shape[1])
        for g in rs.integers(0, keys.size, keys.size):
            m = members[g]
            tot += sums[m[rs.integers(0, m.size, m.size)]].sum(axis=0)
        out[b] = tot
    return out


def _block_sums(cols: list[np.ndarray], site, dates, basin: dict | None):
    g = site_month_groups(site, dates)
    keys, inv = np.unique(g, return_inverse=True)
    sums = np.stack([np.bincount(inv, weights=c, minlength=keys.size) for c in cols], axis=1)
    site_of = np.array([k.split("|")[0] for k in keys])
    outer = np.array([basin.get(x, x) for x in site_of]) if basin is not None else site_of
    return sums, outer


def delta_brier(y, p, p_base, *, site, dates, basin: dict | None = None, n_boot: int = 2000, seed: int = 0,
                alpha: float = 0.05) -> dict:
    """Paired ΔBS = BS(p) − BS(p_base) on the same rows (negative = p better). CI: basin groups >
    site-month blocks (primary, if ``basin``) and site-month blocks (sensitivity)."""
    y, p, q = (np.asarray(a, float) for a in (y, p, p_base))
    ok = np.isfinite(y) & np.isfinite(p) & np.isfinite(q)
    y, p, q, st, dt = y[ok], p[ok], q[ok], np.asarray(site)[ok], np.asarray(dates)[ok]
    out = {"dbs": float(np.mean((p - y) ** 2) - np.mean((q - y) ** 2)) if y.size else np.nan}
    if not y.size or not n_boot:
        return out
    sums, outer = _block_sums([np.ones_like(y), (p - y) ** 2, (q - y) ** 2], st, dt, basin)
    for tag, two in (("", basin is not None), ("_sm", False)):
        if tag == "" and not two:
            continue
        S = two_stage_block_sums(sums, outer, n_boot=n_boot, seed=seed, two_stage=two)
        d = (S[:, 1] - S[:, 2]) / np.maximum(S[:, 0], 1)
        out.update({f"dbs{tag}_ci_low": float(np.quantile(d, alpha / 2)),
                    f"dbs{tag}_ci_high": float(np.quantile(d, 1 - alpha / 2))})
    return out


def brier_skill(y, p, p_ref, *, site=None, dates=None, n_boot: int = 2000, seed: int = 0,
                basin: dict | None = None) -> dict:
    """BS, reference BS and BSS (1 − BS/BS_ref). CI: with ``basin`` (site -> basin group) the
    PRIMARY CI resamples basin groups and site-month blocks within them (two-stage); the
    site-month block bootstrap is always reported as ``bss_sm_ci_*`` (sensitivity)."""
    y, p, p_ref = (np.asarray(a, float) for a in (y, p, p_ref))
    ok = np.isfinite(y) & np.isfinite(p) & np.isfinite(p_ref)
    y, p, p_ref = y[ok], p[ok], p_ref[ok]
    out = {"n": int(y.size), "n_pos": int(y.sum()), "brier": M.brier(y, p), "brier_ref": M.brier(y, p_ref),
           "bss": M.brier_skill_score(y, p, p_ref)}
    if site is not None and dates is not None and n_boot and y.size and y.sum() > 0:
        st, dt = np.asarray(site)[ok], np.asarray(dates)[ok]
        # block sums (exactly equivalent to resampling the rows of whole blocks; fast)
        sums, outer = _block_sums([(p - y) ** 2, (p_ref - y) ** 2], st, dt, basin)

        def bss_of(S):
            with np.errstate(invalid="ignore", divide="ignore"):
                v = 1 - S[:, 0] / S[:, 1]
            return v[np.isfinite(v)]

        sm = bss_of(two_stage_block_sums(sums, outer, n_boot=n_boot, seed=seed, two_stage=False))
        out.update({"bss_sm_ci_low": float(np.quantile(sm, 0.025)), "bss_sm_ci_high": float(np.quantile(sm, 0.975)),
                    "n_site_months": int(sums.shape[0])})
        if basin is not None:
            tw = bss_of(two_stage_block_sums(sums, outer, n_boot=n_boot, seed=seed, two_stage=True))
            bg = np.array([basin.get(x, x) for x in st])
            out.update({"bss_ci_low": float(np.quantile(tw, 0.025)), "bss_ci_high": float(np.quantile(tw, 0.975)),
                        "ci_unit": "basin_group > site-month", "n_basin_groups": int(np.unique(outer).size),
                        "n_basin_groups_with_onsets": int(np.unique(bg[y == 1]).size)})
        else:
            out.update({"bss_ci_low": out["bss_sm_ci_low"], "bss_ci_high": out["bss_sm_ci_high"],
                        "ci_unit": "site-month"})
    return out


def alarm_day_threshold(score, y, valid, budget_per_30d: float = 1.0) -> float:
    """Primary EW-v2 threshold: pooled alarm-day POFD on valid non-onset rows ≤ budget/30."""
    return EW.calibrate_threshold(score, y, valid, target_pofd=float(budget_per_30d) / 30.0)


def alarm_day_stats(score, y, valid, site, threshold: float, budget_per_30d: float = 1.0) -> dict:
    """Realised alarm-day rates at ``threshold`` (alarm = score >= threshold) on valid rows."""
    score, y, valid = np.asarray(score, float), np.asarray(y, int), np.asarray(valid, bool)
    site = np.asarray(site)
    ok = valid & np.isfinite(score)
    alarm = ok & (score >= threshold)
    neg = ok & (y == 0)
    pos = ok & (y == 1)
    pooled = float(alarm[neg].mean()) if neg.any() else np.nan
    per = pd.DataFrame({"site": site[neg], "a": alarm[neg]}).groupby("site")["a"].mean() * 30.0
    return {"threshold": float(threshold), "budget_per_30d": float(budget_per_30d),
            "pooled_alarm_days_per_30d": pooled * 30.0 if np.isfinite(pooled) else np.nan,
            "meets_budget": bool(np.isfinite(pooled) and pooled <= budget_per_30d / 30.0 + 1e-12),
            "site_alarm_days_per_30d_median": float(per.median()) if len(per) else np.nan,
            "site_alarm_days_per_30d_p90": float(per.quantile(0.9)) if len(per) else np.nan,
            "site_alarm_days_per_30d_max": float(per.max()) if len(per) else np.nan,
            "frac_sites_over_budget": float((per > budget_per_30d + 1e-9).mean()) if len(per) else np.nan,
            "time_in_alarm": float(alarm[ok].mean()) if ok.any() else np.nan,
            "pod_issue": float(alarm[pos].mean()) if pos.any() else np.nan,
            "n_neg_days": int(neg.sum()), "n_pos_days": int(pos.sum())}


def alarm_episodes(alarm: np.ndarray, max_len: int | None = None) -> list[tuple[int, int]]:
    """Maximal runs of consecutive True values -> [(start, end_inclusive), ...]. With ``max_len``,
    a run longer than ``max_len`` issue days is split into consecutive episodes of at most
    ``max_len`` days (a standing alarm then counts as repeated alarms)."""
    a = np.asarray(alarm, bool).astype(np.int8)
    if not a.size:
        return []
    d = np.diff(np.concatenate([[0], a, [0]]))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1
    out = []
    for s0, e0 in zip(starts.tolist(), ends.tolist()):
        if max_len is None:
            out.append((s0, e0))
            continue
        for c in range(s0, e0 + 1, max_len):
            out.append((c, min(c + max_len - 1, e0)))
    return out


def episode_outcomes(issue: pd.DatetimeIndex, alarm: np.ndarray, onsets: pd.DatetimeIndex, h_days: int,
                     max_len: int | None = None) -> dict:
    """Hits / false alarms by the [episode start, start + h] rule, and per-onset detection
    (an onset is warned if an episode starts in [onset − h, onset)). Reference implementation
    (per series); :func:`false_alarm_rate` is the vectorised pooled version."""
    eps = alarm_episodes(alarm, max_len)
    t = pd.DatetimeIndex(issue).to_numpy()
    h = np.timedelta64(int(h_days), "D")
    on = np.sort(pd.DatetimeIndex(onsets).to_numpy())
    if t.size:  # evaluable onsets: those an issue date of this series could warn ([t0, t_last + h])
        on = on[(on >= t[0]) & (on <= t[-1] + h)]
    n_hit = n_fa = 0
    for s, _ in eps:
        lo = np.searchsorted(on, t[s], side="left")
        hi = np.searchsorted(on, t[s] + h, side="right")
        if hi > lo:
            n_hit += 1
        else:
            n_fa += 1
    starts = t[[s for s, _ in eps]] if eps else np.array([], dtype="datetime64[ns]")
    warned = 0
    for o in on:
        if ((starts >= o - h) & (starts < o)).any():
            warned += 1
    return {"n_episodes": len(eps), "n_hit_episodes": n_hit, "n_false_episodes": n_fa,
            "n_onsets": int(on.size), "n_onsets_warned": warned}


class StackedFrames:
    """All sites' daily issue series stacked into one array with h-day unusable padding before and
    after each site (onsets up to h days after the last issue date get a position; no window ever
    crosses sites). Built once per (method, lead); thresholds are then evaluated in O(N)."""

    def __init__(self, frames: dict, onsets: dict, h_days: int):
        h = int(h_days)
        sc, us, on = [], [], []
        for site in sorted(frames):
            f = frames[site].sort_index()
            n = len(f)
            pad = np.full(h, np.nan)
            sc.append(np.concatenate([pad, f["score"].to_numpy(float), pad]))
            us.append(np.concatenate([np.zeros(h, bool), f["usable"].to_numpy(bool), np.zeros(h, bool)]))
            o = np.zeros(n + 2 * h, bool)
            if len(f):
                t0 = f.index[0]
                k = ((pd.DatetimeIndex(onsets.get(site, [])) - t0).days.to_numpy() + h).astype(int)
                k = k[(k >= 0) & (k < n + 2 * h)]
                o[k] = True
            on.append(o)
        self.h = h
        self.score = np.concatenate(sc) if sc else np.zeros(0)
        self.usable = np.concatenate(us) if us else np.zeros(0, bool)
        self.onset = np.concatenate(on) if on else np.zeros(0, bool)
        c = np.concatenate([[0], np.cumsum(self.onset)])
        N = self.onset.size
        idx = np.arange(N)
        self.onset_win = (c[np.minimum(idx + h + 1, N)] - c[idx]) > 0   # onset in [t, t + h]
        self.onset_pos = np.flatnonzero(self.onset)
        self.days = int(self.usable.sum())
        self.valid_scores = self.score[self.usable & np.isfinite(self.score)]

    def stats(self, threshold: float, max_len: int | None = None) -> dict:
        alarm = self.usable & np.isfinite(self.score) & (self.score >= threshold)
        d = np.diff(np.concatenate([[0], alarm.astype(np.int8), [0]]))
        starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1
        if max_len is not None and starts.size:
            n_chunks = (ends - starts) // max_len + 1
            rep = np.repeat(starts, n_chunks)
            off = np.arange(n_chunks.sum()) - np.repeat(np.cumsum(n_chunks) - n_chunks, n_chunks)
            starts = rep + off * max_len
        hits = self.onset_win[starts] if starts.size else np.zeros(0, bool)
        is_start = np.zeros(self.onset.size + 1, np.int64)
        is_start[starts + 1] = 1
        cs = np.cumsum(is_start)                                            # cs[i] = starts in [0, i)
        o = self.onset_pos
        warned = int(((cs[o] - cs[np.maximum(o - self.h, 0)]) > 0).sum()) if o.size else 0   # [o-h, o)
        n_fa = int((~hits).sum())
        return {"n_episodes": int(starts.size), "n_hit_episodes": int(hits.sum()), "n_false_episodes": n_fa,
                "n_onsets": int(o.size), "n_onsets_warned": warned, "usable_site_days": self.days,
                "fa_per_site_30d": n_fa / (self.days / 30.0) if self.days else np.nan,
                "time_in_alarm": float(alarm.sum() / self.days) if self.days else np.nan,
                "pod_episode": warned / o.size if o.size else np.nan}


def false_alarm_rate(frames: dict, onsets: dict, threshold: float, h_days: int,
                     max_len: int | None = None) -> tuple[float, dict]:
    """Pooled false-alarm episodes per site per 30 usable issue days at ``threshold``."""
    st = StackedFrames(frames, onsets, h_days).stats(threshold, max_len)
    return st["fa_per_site_30d"], st


def calibrate_episode_threshold(frames: dict, onsets: dict, h_days: int, *, budget_per_30d: float = 1.0,
                                n_candidates: int = 400, max_len: int | None = None,
                                stacked: StackedFrames | None = None) -> dict:
    """Smallest threshold such that it and every larger candidate meet the false-alarm budget
    (conservative under the non-monotonicity of episode counts). Use calibration data only.
    ``max_len`` = split standing alarms into episodes of at most that many days (see
    :func:`alarm_episodes`); None = maximal runs as written in the draft protocol."""
    st = stacked or StackedFrames(frames, onsets, h_days)
    allv = st.valid_scores
    if not allv.size:
        raise ValueError("no usable scores to calibrate on")
    qs = np.unique(np.quantile(allv, np.linspace(0.0, 1.0, n_candidates)))
    cands = np.concatenate([qs, [np.nextafter(allv.max(), np.inf)]])
    ok_from = float(cands[-1])
    for thr in cands[::-1]:  # descending: stop at the first violation
        if not (st.stats(float(thr), max_len)["fa_per_site_30d"] <= budget_per_30d):
            break
        ok_from = float(thr)
    return {"threshold": ok_from, "budget_per_30d": budget_per_30d, "max_episode_days": max_len,
            **st.stats(ok_from, max_len)}


def daily_events(site: str, dates: pd.DatetimeIndex, event: np.ndarray, onset: np.ndarray,
                 values: np.ndarray | None = None) -> list[EW.Event]:
    """Event objects (for early_warning) from daily event/onset masks; end = last day of the
    consecutive event-day run that starts at the onset."""
    out = []
    ev = np.asarray(event, bool)
    for o in np.flatnonzero(onset):
        e = o
        while e + 1 < ev.size and ev[e + 1]:
            e += 1
        seg = values[o:e + 1] if values is not None else np.array([np.nan])
        k = int(np.nanargmin(seg)) if values is not None and np.isfinite(seg).any() else 0
        out.append(EW.Event(site, pd.Timestamp(dates[o]), pd.Timestamp(dates[e]), pd.Timestamp(dates[o + k]),
                            float(seg[k]) if values is not None else np.nan, 24.0 * (e - o + 1), False))
    return out


def evaluable_onsets(frames: dict, events: dict, h_days: int) -> dict:
    """THE evaluable-onset definition of EW-v2 (protocol §7; review 2026-11-08 C2), shared by the
    >= 50-onset claim gate, POD, the placebo comparison and every alarm-series control.

    An onset of ``events[site]`` is evaluable iff (i) it is not censored, (ii) it lies in
    (first issue date, last issue date + h] of the site's frame, and (iii) at least one USABLE issue
    row (``usable`` and finite ``score``) lies in [onset − h, onset). (iii) implies the lower bound
    of (ii); both are checked. Returns {site: [Event, ...]} (sites without evaluable onsets omitted)."""
    h = np.timedelta64(int(h_days), "D")
    out = {}
    for site, f in frames.items():
        evs = [e for e in events.get(site, []) if not e.onset_censored]
        if not evs or len(f) == 0:
            continue
        f = f.sort_index()
        t = f.index.to_numpy()
        use = f["usable"].to_numpy(bool) & np.isfinite(f["score"].to_numpy(float))
        keep = []
        for e in evs:
            on = e.onset.to_datetime64()
            if not (t[0] < on <= t[-1] + h):
                continue
            lo, hi = np.searchsorted(t, on - h, "left"), np.searchsorted(t, on, "left")
            if use[lo:hi].any():
                keep.append(e)
        if keep:
            out[site] = keep
    return out


def count_evaluable_onsets(frames: dict, events: dict, h_days: int, basin: dict | None = None) -> dict:
    """Gate counts from :func:`evaluable_onsets`: onsets, sites and basin groups with evaluable onsets."""
    ev = evaluable_onsets(frames, events, h_days)
    groups = {(basin or {}).get(str(s), str(s)) for s in ev}
    return {"n_evaluable_onsets": int(sum(len(v) for v in ev.values())), "n_sites_with_onsets": len(ev),
            "n_basin_groups_with_onsets": len(groups)}


def sample_clean_placebos(issue: pd.DatetimeIndex, usable, exclude_events: list, candidate, *, n: int,
                          horizon: str, exclusion: str = "14D", seed: int = 0,
                          match_months_of: list | None = None) -> pd.DatetimeIndex:
    """``early_warning.sample_placebo_onsets`` restricted to ``candidate`` days (e.g. days on which a
    real onset could have started). With an all-True candidate mask the draw is identical to
    ``sample_placebo_onsets`` (same seed) — tested."""
    h, ex = pd.Timedelta(horizon), pd.Timedelta(exclusion)
    t = issue
    ok = np.asarray(candidate, bool).copy()
    for ev in exclude_events:
        ok &= ~((t >= ev.onset - h - ex) & (t <= ev.end + ex))
    use = np.asarray(usable, bool)
    csum = np.concatenate([[0], np.cumsum(use)])
    lo = np.searchsorted(t.to_numpy(), (t - h).to_numpy(), side="left")
    ok &= (csum[np.arange(len(t))] - csum[lo]) > 0
    cand = np.flatnonzero(ok)
    if cand.size == 0:
        return pd.DatetimeIndex([])
    rs = np.random.default_rng(seed)
    w = np.ones(cand.size)
    if match_months_of:
        target = pd.Series([pd.Timestamp(x).month for x in match_months_of]).value_counts(normalize=True)
        cm = t[cand].month
        avail = pd.Series(cm).value_counts(normalize=True)
        w = np.array([target.get(m, 0.0) / avail.get(m, np.inf) for m in cm])
        if w.sum() == 0:
            w = np.ones(cand.size)
    w = w / w.sum()
    pick = rs.choice(cand, size=min(n, int(np.count_nonzero(w))), replace=False, p=w)
    return pd.DatetimeIndex(np.sort(t[pick]))


def placebo_comparison(frames: dict, events: dict, threshold: float, h_days: int, *, k_per_onset: int = 20,
                       exclusion: str = "14D", seed: int = 0, exclude_events: dict | None = None,
                       candidates: dict | None = None, return_frames: bool = False) -> dict:
    """Detection before real onsets vs before season-matched placebo onsets (same sites).

    ``events`` = onset events to detect; ``exclude_events`` = ALL event periods to keep placebos
    away from (default: ``events``); ``candidates[site]`` = boolean mask over the frame index of
    days eligible as pseudo-onsets (default: all)."""
    horizon = f"{int(h_days)}D"
    det, plc = [], []
    evaluable = evaluable_onsets(frames, events, h_days)   # shared definition (gate = POD = controls)
    for i, (site, f) in enumerate(sorted(frames.items())):
        f = f.sort_index()
        evs = evaluable.get(site, [])
        usable = f["usable"].to_numpy(bool) & np.isfinite(f["score"].to_numpy(float))
        if not evs:
            continue
        d = EW.event_detection(f.index, f["score"], usable, evs, threshold, horizon)
        det.append(d)
        n = k_per_onset * len(evs)
        excl = (exclude_events or events).get(site, [])
        cand = candidates.get(site) if candidates is not None else None
        cand = np.ones(len(f), bool) if cand is None else np.asarray(cand, bool)
        po = sample_clean_placebos(f.index, usable, excl, cand, n=n, horizon=horizon, exclusion=exclusion,
                                   seed=seed + i, match_months_of=[e.onset for e in evs])
        p = EW.placebo_detection(f.index, f["score"], usable, po, threshold, horizon, site)
        if not p.empty:
            plc.append(p)
    det = pd.concat(det, ignore_index=True) if det else pd.DataFrame(columns=["evaluable", "detected"])
    plc = pd.concat(plc, ignore_index=True) if plc else pd.DataFrame(columns=["evaluable", "detected"])
    ev_ok = det[det["evaluable"].astype(bool)] if len(det) else det
    pl_ok = plc[plc["evaluable"].astype(bool)] if len(plc) else plc
    rh, rn = int(ev_ok["detected"].sum()) if len(ev_ok) else 0, len(ev_ok)
    ph, pn = int(pl_ok["detected"].sum()) if len(pl_ok) else 0, len(pl_ok)
    out = {"events_evaluable": rn, "events_detected": rh, "pod_real": rh / rn if rn else np.nan,
            "placebo_n": pn, "placebo_detected": ph, "pod_placebo": ph / pn if pn else np.nan,
            "pod_real_minus_placebo": (rh / rn - ph / pn) if rn and pn else np.nan,
            "fisher_p_greater": EW.fisher_greater(rh, rn, ph, pn),
            "pod_real_ge24h": float((ev_ok["detected"].astype(bool) & (ev_ok["lead_first_h"] >= 48)).mean())
            if rn else np.nan,
            "pod_placebo_ge24h": float((pl_ok["detected"].astype(bool) & (pl_ok["lead_first_h"] >= 48)).mean())
            if pn else np.nan,
            "median_lead_first_h": float(ev_ok.loc[ev_ok["detected"].astype(bool), "lead_first_h"].median())
            if rh else np.nan}
    if return_frames:   # per-onset / per-placebo outcomes for the cluster-aware test (ew_controls.placebo_cluster)
        out["_det"], out["_plc"] = det, plc
    return out


def discrimination(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    ok = np.isfinite(y) & np.isfinite(p)
    return {"auroc": M.auroc(y[ok], p[ok]), "auprc": M.auprc(y[ok], p[ok]),
            "prevalence": float(y[ok].mean()) if ok.any() else np.nan}
