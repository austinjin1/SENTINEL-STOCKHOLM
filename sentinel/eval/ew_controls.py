# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""EW-v2 primary-decision statistics and §8 controls (pre-registered before the Test-A freeze,
2026-11-08; pre-test review C1/C4/W8).

Decision statistics
  * :func:`bss_pvalue` — one-sided bootstrap p-value P*(BSS* <= 0) with the PRIMARY resampling
    scheme (basin groups, then site-months within the drawn groups; two-stage block sums).
  * :func:`holm` — Holm step-down over a FIXED family (EW-v2: {H1, H2, H4}); adjusted p-values.
  * :func:`primary_decision` — power rule + Holm: underpowered hazards enter with p = 1 (review C3).
  * :func:`placebo_cluster` — POD_real − POD_placebo with a basin-group cluster bootstrap CI and
    one-sided p (replaces the anti-conservative Fisher test as the H-c test; Fisher kept descriptive).

Controls (all on the frozen predictions; nothing is refitted)
  * :func:`random_alarm_pod` — expected POD of random alarms issued at each site's realised
    alarm rate on its usable days (analytic), vs the observed POD.
  * :func:`circular_shift` — 1,000 circular-shift surrogates of each site's alarm series (shift
    >= 30 days, independent per site); one-sided p for the observed POD.
  * :func:`label_block_permutation` — each site's labels circularly shifted (>= 30 issue rows) against
    the frozen probabilities; null distribution of BSS, one-sided p.
  * :func:`molchan` — Molchan diagram (miss rate vs fraction of issue rows alarmed) + area skill
    (1 − 2·area; 0 = random, 1 = perfect).
  * :func:`reliability` — quantile-binned reliability table and expected calibration error.
  * :func:`relative_economic_value` — REV for cost/loss ratios (optimal threshold p >= C/L).
  * :func:`diebold_mariano` — DM test on the daily site-mean Brier loss differential (Newey–West HAC).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .ew_scoring import _block_sums, evaluable_onsets, two_stage_block_sums


# ----------------------------------------------------------------------------- decision statistics

def bss_pvalue(y, p, p_ref, *, site, dates, basin: dict, n_boot: int = 2000, seed: int = 0) -> dict:
    """One-sided p-value for H0: BSS <= 0 from the two-stage (basin > site-month) bootstrap."""
    y, p, q = (np.asarray(a, float) for a in (y, p, p_ref))
    ok = np.isfinite(y) & np.isfinite(p) & np.isfinite(q)
    if not ok.all():
        raise ValueError("non-finite probabilities on primary rows (every method must cover identical rows)")
    if y.sum() == 0:
        return {"bss": np.nan, "p_one_sided": np.nan, "n_boot_valid": 0}
    sums, outer = _block_sums([(p - y) ** 2, (q - y) ** 2], np.asarray(site), np.asarray(dates), basin)
    S = two_stage_block_sums(sums, outer, n_boot=n_boot, seed=seed, two_stage=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        b = 1 - S[:, 0] / S[:, 1]
    b = b[np.isfinite(b)]
    bss = float(1 - np.sum((p - y) ** 2) / np.sum((q - y) ** 2))
    return {"bss": bss, "p_one_sided": float((1 + np.sum(b <= 0)) / (1 + b.size)), "n_boot_valid": int(b.size),
            "bss_ci_low": float(np.quantile(b, 0.025)), "bss_ci_high": float(np.quantile(b, 0.975))}


def holm(pvals: dict, alpha: float = 0.05) -> dict:
    """Holm step-down over the given family (NaN p = not testable: counted in m, never rejected).
    Returns {name: {p, p_holm, reject}}."""
    m = len(pvals)
    items = sorted(pvals.items(), key=lambda kv: (np.inf if not np.isfinite(kv[1]) else kv[1]))
    out, run_max, still = {}, 0.0, True
    for i, (k, pv) in enumerate(items):
        adj = min(1.0, (m - i) * pv) if np.isfinite(pv) else np.nan
        run_max = max(run_max, adj) if np.isfinite(adj) else np.nan
        still = still and np.isfinite(pv) and pv <= alpha / (m - i)
        out[k] = {"p": float(pv), "p_holm": float(run_max) if np.isfinite(run_max) else np.nan, "reject": bool(still)}
    return out


def is_underpowered(n_evaluable_onsets, n_basin_groups_with_onsets, *, min_onsets: int = 50,
                    min_basin_groups: int = 2) -> bool:
    """Protocol §7 power rule (amended 2026-11-08, review C3): < ``min_onsets`` evaluable onsets
    (:func:`sentinel.eval.ew_scoring.evaluable_onsets`) OR < ``min_basin_groups`` basin groups with
    evaluable onsets (the outer bootstrap stage is degenerate). Missing counts = underpowered."""
    try:
        n, g = int(n_evaluable_onsets), int(n_basin_groups_with_onsets)
    except (TypeError, ValueError):
        return True
    return n < min_onsets or g < min_basin_groups


def primary_decision(results: dict, family, *, min_onsets: int = 50, min_basin_groups: int = 2,
                     alpha: float = 0.05) -> dict:
    """Holm over the FIXED family with the power rule applied BEFORE the step-down (review C3):
    an underpowered hazard (or one without a result) enters Holm with p = 1 (still counted in m) and
    can never be claimed; its bootstrap p is kept as ``p_one_sided_descriptive`` only.
    ``results[h]`` holds ``p_one_sided``, ``n_evaluable_onsets``, ``n_basin_groups_with_onsets``.
    Returns a new dict {h: {..., underpowered, p_holm_input, p_holm, reject_holm, claim_allowed}}."""
    out = {}
    for h in family:
        r = dict(results.get(h) or {"p_one_sided": np.nan, "note": "no selected method / no rows"})
        up = is_underpowered(r.get("n_evaluable_onsets"), r.get("n_basin_groups_with_onsets"),
                             min_onsets=min_onsets, min_basin_groups=min_basin_groups)
        p = r.get("p_one_sided", np.nan)
        p = float(p) if p is not None else np.nan
        r.update({"underpowered": bool(up), "p_one_sided_descriptive": p,
                  "p_holm_input": 1.0 if (up or not np.isfinite(p)) else p,
                  "power_rule": f">= {min_onsets} evaluable onsets and >= {min_basin_groups} basin groups with onsets"})
        out[h] = r
    hol = holm({h: out[h]["p_holm_input"] for h in family}, alpha=alpha)
    for h in family:
        rej = bool(hol[h]["reject"]) and not out[h]["underpowered"]
        out[h].update({"p_holm": hol[h]["p_holm"], "reject_holm": rej, "claim_allowed": rej})
    return out


def placebo_cluster(det: pd.DataFrame, plc: pd.DataFrame, basin: dict, *, n_boot: int = 2000, seed: int = 0) -> dict:
    """POD_real − POD_placebo, basin-group cluster bootstrap (groups drawn with replacement; all
    evaluable real and placebo onsets of a drawn group enter together)."""
    empty = {"diff": np.nan, "diff_ci_low": np.nan, "diff_ci_high": np.nan, "p_one_sided": np.nan}
    if len(det) == 0 or len(plc) == 0 or "site" not in det.columns or "site" not in plc.columns:
        return empty
    d = det[det["evaluable"].astype(bool)].assign(g=lambda x: x["site"].astype(str).map(lambda s: basin.get(s, s)))
    q = plc[plc["evaluable"].astype(bool)].assign(g=lambda x: x["site"].astype(str).map(lambda s: basin.get(s, s)))
    if d.empty or q.empty:
        return {"diff": np.nan, "diff_ci_low": np.nan, "diff_ci_high": np.nan, "p_one_sided": np.nan}
    keys = np.union1d(d["g"].unique(), q["g"].unique())
    dr = d.groupby("g")["detected"].agg(["sum", "count"]).reindex(keys, fill_value=0).to_numpy(float)
    pr = q.groupby("g")["detected"].agg(["sum", "count"]).reindex(keys, fill_value=0).to_numpy(float)
    rs = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        k = rs.integers(0, keys.size, keys.size)
        a, b = dr[k].sum(0), pr[k].sum(0)
        if a[1] > 0 and b[1] > 0:
            vals.append(a[0] / a[1] - b[0] / b[1])
    vals = np.asarray(vals)
    diff = float(dr[:, 0].sum() / dr[:, 1].sum() - pr[:, 0].sum() / pr[:, 1].sum())
    return {"diff": diff, "diff_ci_low": float(np.quantile(vals, 0.025)), "diff_ci_high": float(np.quantile(vals, 0.975)),
            "p_one_sided": float((1 + np.sum(vals <= 0)) / (1 + vals.size)), "n_basin_groups": int(keys.size)}


# ----------------------------------------------------------------------------- alarm-series controls

def _site_arrays(frames: dict, events: dict, threshold: float, h_days: int):
    """Per site: alarm (bool), usable (bool), and per evaluable event the issue-index window
    [lo, hi) strictly before onset within h days."""
    out = []
    h = np.timedelta64(int(h_days), "D")
    evaluable = evaluable_onsets(frames, events, h_days)   # shared definition (gate = POD = controls)
    for site, f in sorted(frames.items()):
        evs = evaluable.get(site, [])
        if not evs:
            continue
        f = f.sort_index()
        t = f.index.to_numpy()
        s = f["score"].to_numpy(float)
        use = f["usable"].to_numpy(bool) & np.isfinite(s)
        alarm = use & (s >= threshold)
        win = []
        for e in evs:
            on = e.onset.to_datetime64()
            lo, hi = np.searchsorted(t, on - h, "left"), np.searchsorted(t, on, "left")
            if use[lo:hi].any():
                win.append((lo, hi))
        if win:
            out.append((site, alarm, use, win))
    return out


def random_alarm_pod(frames: dict, events: dict, threshold: float, h_days: int) -> dict:
    """Observed POD vs the expected POD of random alarms at each site's realised alarm rate (per
    usable day): P(detect) = 1 − (1 − r_s)^(usable days in the window)."""
    obs, exp = [], []
    for _, alarm, use, win in _site_arrays(frames, events, threshold, h_days):
        r = alarm[use].mean() if use.any() else 0.0
        for lo, hi in win:
            obs.append(bool(alarm[lo:hi].any()))
            exp.append(1 - (1 - r) ** int(use[lo:hi].sum()))
    if not obs:
        return {"pod_obs": np.nan, "pod_random_expected": np.nan, "n_events": 0}
    return {"pod_obs": float(np.mean(obs)), "pod_random_expected": float(np.mean(exp)), "n_events": len(obs)}


def circular_shift(frames: dict, events: dict, threshold: float, h_days: int, *, n_surr: int = 1000,
                   min_shift: int = 30, seed: int = 0) -> dict:
    arr = _site_arrays(frames, events, threshold, h_days)
    if not arr:
        return {"pod_obs": np.nan, "pod_surr_mean": np.nan, "p_one_sided": np.nan, "n_events": 0}
    obs = np.mean([alarm[lo:hi].any() for _, alarm, _, win in arr for lo, hi in win])
    rs = np.random.default_rng(seed)
    pods = np.empty(n_surr)
    for b in range(n_surr):
        hit = []
        for _, alarm, use, win in arr:
            n = alarm.size
            k = int(rs.integers(min_shift, max(min_shift + 1, n - min_shift)))
            sh = np.roll(alarm, k) & use       # alarms only on days the model could have issued
            hit += [sh[lo:hi].any() for lo, hi in win]
        pods[b] = np.mean(hit)
    return {"pod_obs": float(obs), "pod_surr_mean": float(pods.mean()), "pod_surr_p95": float(np.quantile(pods, 0.95)),
            "p_one_sided": float((1 + np.sum(pods >= obs)) / (1 + n_surr)), "n_events": int(sum(len(w) for *_, w in arr)),
            "n_surrogates": n_surr, "min_shift_days": min_shift}


# ----------------------------------------------------------------------------- probability controls

def label_block_permutation(y, p, p_ref, *, site, dates, n_perm: int = 1000, min_shift: int = 30,
                            seed: int = 0) -> dict:
    """Label permutation within site that keeps the label autocorrelation: each site's label series
    (its issue rows in date order) is circularly shifted by an independent random offset of
    >= ``min_shift`` rows (any offset if the site has < 2·min_shift rows); null distribution of BSS."""
    y, p, q = (np.asarray(a, float) for a in (y, p, p_ref))
    site = np.asarray(site).astype(str)
    order = np.lexsort((pd.to_datetime(np.asarray(dates)).to_numpy(), site))
    groups = [order[np.flatnonzero(site[order] == s_)] for s_ in np.unique(site)]
    obs = 1 - np.sum((p - y) ** 2) / np.sum((q - y) ** 2)
    rs = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for b in range(n_perm):
        yp = np.empty_like(y)
        for g in groups:
            n = g.size
            k = int(rs.integers(min_shift, n - min_shift)) if n > 2 * min_shift else int(rs.integers(0, max(n, 1)))
            yp[g] = np.roll(y[g], k)
        null[b] = 1 - np.sum((p - yp) ** 2) / np.sum((q - yp) ** 2)
    return {"bss_obs": float(obs), "bss_null_mean": float(null.mean()), "bss_null_p95": float(np.quantile(null, 0.95)),
            "p_one_sided": float((1 + np.sum(null >= obs)) / (1 + n_perm)), "n_perm": n_perm}


def molchan(y, p, n_thr: int = 200) -> dict:
    y, p = np.asarray(y, int), np.asarray(p, float)
    if y.sum() == 0:
        return {"molchan_area": np.nan, "molchan_skill": np.nan}
    thr = np.unique(np.quantile(p, np.linspace(0, 1, n_thr)))
    tau = np.array([(p >= t).mean() for t in thr] + [0.0])
    nu = np.array([1 - (p[y == 1] >= t).mean() for t in thr] + [1.0])
    o = np.argsort(tau)
    tau, nu = np.concatenate([[0.0], tau[o], [1.0]]), np.concatenate([[1.0], nu[o], [0.0]])
    area = float(np.trapezoid(nu, tau))
    return {"molchan_area": area, "molchan_skill": 1 - 2 * area}


def reliability(y, p, n_bins: int = 10) -> tuple[pd.DataFrame, float]:
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, max(edges.size - 2, 0))
    t = pd.DataFrame({"b": b, "p": p, "y": y}).groupby("b").agg(n=("y", "size"), mean_p=("p", "mean"), obs=("y", "mean"))
    ece = float(np.sum(t["n"] * np.abs(t["mean_p"] - t["obs"])) / t["n"].sum())
    return t.reset_index(), ece


def relative_economic_value(y, p, alphas=(0.01, 0.02, 0.05, 0.1, 0.2, 0.5)) -> dict:
    y, p = np.asarray(y, int), np.asarray(p, float)
    s = y.mean()
    out = {}
    for a in alphas:
        act = p >= a
        e_f = a * act.mean() + ((~act) & (y == 1)).mean()
        e_c, e_p = min(a, s), a * s
        out[f"rev_{a:g}"] = float((e_c - e_f) / (e_c - e_p)) if e_c > e_p else np.nan
    return out


def diebold_mariano(loss_a, loss_b, dates, *, lag: int = 7) -> dict:
    """DM on the daily mean loss differential d_t = mean_sites(loss_a − loss_b); negative = a better.
    One-sided p for H1: a better than b. Newey–West HAC with Bartlett weights."""
    d = pd.Series(np.asarray(loss_a, float) - np.asarray(loss_b, float)).groupby(pd.to_datetime(np.asarray(dates))).mean()
    x = d.sort_index().to_numpy()
    T = x.size
    if T < 10:
        return {"dm_stat": np.nan, "dm_p_one_sided": np.nan, "n_days": int(T)}
    xc = x - x.mean()
    v = np.dot(xc, xc) / T
    for k in range(1, min(lag, T - 1) + 1):
        v += 2 * (1 - k / (lag + 1)) * np.dot(xc[k:], xc[:-k]) / T
    stat = x.mean() / math.sqrt(v / T) if v > 0 else np.nan
    p = 0.5 * (1 + math.erf(stat / math.sqrt(2))) if np.isfinite(stat) else np.nan
    return {"dm_stat": float(stat), "dm_p_one_sided": float(p), "mean_loss_diff": float(x.mean()), "n_days": int(T)}
