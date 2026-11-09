# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Link detection, recovery scoring, positive/negative controls, replication (protocol §7).

Detection of X_i → X_j in one graph replicate: some lag τ in the window with code '-->' (τ ≥ 1 or
contemporaneous oriented) or, at τ = 0, an unoriented adjacency ('o-o', 'x-x', LPCMCI marks). With an
expected sign, the partial correlation at that lag must have that sign. Site-level frequency = share of
bootstrap replicates detecting; detected ⇔ freq ≥ 0.5; robust ⇔ freq ≥ 0.8.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DETECT, ROBUST = 0.5, 0.8
ADJ0 = (1, 3, 4, 5, 6)                  # contemporaneous codes counted as adjacency i–j

# (cause, effect, expected sign, (lag_lo, lag_hi), site condition)
POSITIVE = [
    ("tair", "tw", +1, (0, 2), None),
    ("precip", "logq", +1, (0, 3), None),
    ("logq", "logturb", +1, (0, 1), None),
    ("logq", "logsc", -1, (0, 1), "dilution"),
    ("sw", "dmax", +1, (0, 0), None),
    ("tw", "dmin", -1, (0, 1), None),
]
WEATHER = ("tair", "sw", "precip")


def detect_rep(graphs: np.ndarray, vals: np.ndarray, i: int, j: int, lags=(0, 7), sign: int | None = None,
               lagged_only: bool = False, oriented_only: bool = False) -> np.ndarray:
    """(B,) bool: replicate detects i → j inside ``lags`` (with ``sign`` if given)."""
    lo, hi = lags
    B = graphs.shape[0]
    out = np.zeros(B, bool)
    for tau in range(lo, hi + 1):
        if tau >= graphs.shape[-1]:
            break
        if tau == 0:
            if lagged_only:
                continue
            g = graphs[:, i, j, 0]
            hit = (g == 1) if oriented_only else np.isin(g, ADJ0)
        else:
            hit = np.isin(graphs[:, i, j, tau], (1, 6))    # '-->' or LPCMCI 'o->' (never emitted by PCMCI+)
        if sign is not None:
            hit &= np.sign(vals[:, i, j, tau]) == sign
        out |= hit
    return out


def freq(graphs, vals, i, j, **kw) -> float:
    return float(detect_rep(graphs, vals, i, j, **kw).mean())


def recovery_scores(graphs: np.ndarray, vals: np.ndarray, names, truth: set, scored: list, *, threshold=DETECT,
                    tau_max=7) -> dict:
    """Step 1: ordered-pair recall / FPR. ``scored`` = list of (p, c) pairs entering the score; a
    contemporaneous unoriented adjacency counts as a hit for a true pair in either direction and as one
    false positive only if neither direction is true."""
    ix = {n: k for k, n in enumerate(names)}
    tp = fn = fp = tn = 0
    orient_ok = orient_n = 0
    for p, c in scored:
        if p not in ix or c not in ix:
            continue
        i, j = ix[p], ix[c]
        lag_hit = detect_rep(graphs, vals, i, j, (1, tau_max)).mean() >= threshold
        con = np.isin(graphs[:, i, j, 0], ADJ0).mean() >= threshold
        oriented = (graphs[:, i, j, 0] == 1).mean() >= threshold
        hit = bool(lag_hit or con)
        oriented = bool(oriented)
        con = bool(con)
        lag_hit = bool(lag_hit)
        is_true = (p, c) in truth
        rev_true = (c, p) in truth
        if is_true:
            tp += int(hit)
            fn += int(not hit)
            if con:
                orient_n += 1
                orient_ok += int(oriented)
        else:
            # unoriented contemporaneous adjacency of a pair that is true in the reverse direction is not an FP
            false_hit = bool(lag_hit or (oriented and not rev_true) or (con and not oriented and not rev_true))
            fp += int(false_hit)
            tn += int(not false_hit)
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn, "recall": tp / max(tp + fn, 1), "fpr": fp / max(fp + tn, 1),
            "precision": tp / max(tp + fp, 1), "orientation_acc": orient_ok / orient_n if orient_n else np.nan}


def positive_controls(graphs, vals, names, *, dilution: bool | None) -> list[dict]:
    ix = {n: k for k, n in enumerate(names)}
    rows = []
    for cause, eff, sgn, lags, cond in POSITIVE:
        if cause not in ix or eff not in ix:
            continue
        if cond == "dilution" and not dilution:
            continue
        f = freq(graphs, vals, ix[cause], ix[eff], lags=lags, sign=sgn)
        f_any = freq(graphs, vals, ix[cause], ix[eff], lags=lags)
        rows.append({"control": f"{cause}->{eff}", "freq": f, "freq_any_sign": f_any, "detected": f >= DETECT,
                     "robust": f >= ROBUST})
    return rows


def negative_n1(graphs, vals, names, tau_max=7) -> list[dict]:
    """River → weather: lagged '-->' (τ ≥ 1) or contemporaneous edge oriented into the weather node."""
    ix = {n: k for k, n in enumerate(names)}
    rows = []
    for r in names:
        if r in WEATHER:
            continue
        for w in WEATHER:
            if w not in ix:
                continue
            i, j = ix[r], ix[w]
            lag = detect_rep(graphs, vals, i, j, (1, tau_max))
            con = graphs[:, i, j, 0] == 1
            rows.append({"control": f"N1:{r}->{w}", "freq": float((lag | con).mean()),
                         "detected": float((lag | con).mean()) >= DETECT})
    return rows


def negative_lagged_into(graphs, vals, names, sources, targets, label, tau_max=7, lags=(1, None)) -> list[dict]:
    ix = {n: k for k, n in enumerate(names)}
    lo, hi = lags[0], (lags[1] if lags[1] is not None else tau_max)
    rows = []
    for s in sources:
        for t in targets:
            if s not in ix or t not in ix:
                continue
            f = freq(graphs, vals, ix[s], ix[t], lags=(lo, hi))
            rows.append({"control": f"{label}:{s}->{t}", "freq": f, "detected": f >= DETECT})
    return rows


def reversal_asymmetry(g_fwd, v_fwd, g_rev, v_rev, names, tau_max=7) -> list[dict]:
    """N2 redefined (review A2, 2026-11-08). On time-reversed data, an autocorrelated response produces
    lagged weather → river links by construction, so "any detection in reversed data" is not a false
    positive. Instead, for every weather → river pair, compare the lagged (τ ≥ 1) detection frequency in
    reversed data with that in forward data (both unconstrained). A genuine forward-time driver should be
    detected more often forward than reversed (``rev_minus_fwd`` < 0). Diagnostic only, no gate."""
    ix = {n: k for k, n in enumerate(names)}
    rows = []
    for w in WEATHER:
        if w not in ix:
            continue
        for r in names:
            if r in WEATHER:
                continue
            f = freq(g_fwd, v_fwd, ix[w], ix[r], lags=(1, tau_max))
            b = freq(g_rev, v_rev, ix[w], ix[r], lags=(1, tau_max))
            rows.append({"control": f"N2asym:{w}->{r}", "freq_fwd": f, "freq_rev": b, "rev_minus_fwd": b - f,
                         "fwd_detected": f >= DETECT})
    return rows


# ----------------------------------------------------------------------------- step 3

def site_links(graphs, vals, names, tau_max=7) -> pd.DataFrame:
    """Every ordered pair with a river child: bootstrap frequency (any sign), modal sign and modal lag."""
    rows = []
    for i, p in enumerate(names):
        for j, c in enumerate(names):
            if i == j or c in WEATHER:
                continue
            det = np.zeros(graphs.shape[0], bool)
            best_lag = np.full(graphs.shape[0], -1)
            best_val = np.zeros(graphs.shape[0])
            for tau in range(tau_max + 1):
                hit = np.isin(graphs[:, i, j, 0], ADJ0) if tau == 0 else graphs[:, i, j, tau] == 1
                v = vals[:, i, j, tau]
                better = hit & (np.abs(v) > np.abs(best_val))
                best_val = np.where(better, v, best_val)
                best_lag = np.where(better, tau, best_lag)
                det |= hit
            f = float(det.mean())
            sg = np.sign(best_val[det]) if det.any() else np.array([0.0])
            modal_sign = float(np.sign(sg.sum())) if det.any() else 0.0
            lags = best_lag[det]
            rows.append({"cause": p, "effect": c, "freq": f,
                         "sign": modal_sign, "sign_agree": float(np.mean(sg == modal_sign)) if det.any() else np.nan,
                         "lag": int(np.bincount(lags).argmax()) if det.any() else -1,
                         "mean_val": float(best_val[det].mean()) if det.any() else 0.0})
    return pd.DataFrame(rows)


def replication(links: pd.DataFrame, *, min_sites_region=5, min_regions=3, share=0.6, sign_share=0.8,
                val_links: pd.DataFrame | None = None, val_share=0.5) -> pd.DataFrame:
    """``links``: rows site × (cause, effect) with columns freq, sign, lag, huc2 (train sites).
    A link replicates iff robust (freq ≥ 0.8) at ≥ ``share`` of eligible sites in each of ≥ ``min_regions``
    HUC2 regions with ≥ ``min_sites_region`` eligible sites, ≥ ``sign_share`` of robust sites share the modal
    sign, modal lags of robust sites within ±1 of the overall mode, and detected (freq ≥ 0.5) at ≥
    ``val_share`` of eligible val-role sites."""
    out = []
    for (c, e), g in links.groupby(["cause", "effect"]):
        reg = g.groupby("huc2")
        elig = [r for r, gg in reg if len(gg) >= min_sites_region]
        ok_regions = [r for r in elig if (reg.get_group(r)["freq"] >= ROBUST).mean() >= share]
        rob = g[g["freq"] >= ROBUST]
        modal_sign = float(np.sign(rob["sign"].sum())) if len(rob) else 0.0
        sign_ok = bool(len(rob) and (rob["sign"] == modal_sign).mean() >= sign_share)
        if len(rob):
            mode_lag = int(np.bincount(rob["lag"].clip(lower=0)).argmax())
            lag_ok = bool((np.abs(rob["lag"] - mode_lag) <= 1).mean() >= sign_share)
        else:
            mode_lag, lag_ok = -1, False
        rec = {"cause": c, "effect": e, "n_sites": len(g), "share_robust": float((g["freq"] >= ROBUST).mean()),
               "n_regions_eligible": len(elig), "n_regions_ok": len(ok_regions), "modal_sign": modal_sign,
               "sign_ok": sign_ok, "modal_lag": mode_lag, "lag_ok": lag_ok}
        if val_links is not None:
            v = val_links[(val_links["cause"] == c) & (val_links["effect"] == e)]
            rec["val_n_sites"] = len(v)
            rec["val_share_detected"] = float((v["freq"] >= DETECT).mean()) if len(v) else np.nan
            val_ok = bool(len(v) >= 5 and rec["val_share_detected"] >= val_share)
        else:
            val_ok = False
        rec["step3_pass"] = bool(len(ok_regions) >= min_regions and sign_ok and lag_ok and val_ok)
        out.append(rec)
    return pd.DataFrame(out)
