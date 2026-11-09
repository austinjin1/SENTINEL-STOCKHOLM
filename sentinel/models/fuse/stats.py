# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Statistics for SENTINEL-Fuse comparisons (docs/preregistration/fuse_protocol.md §10).

* :func:`cluster_ci` — cluster (site / block) percentile bootstrap of a paired statistic with the
  pre-registered small-cluster correction (review W9): the percentile interval is widened about the
  point estimate by c = sqrt(G / (G − 1)) (the CR1 finite-cluster factor), which matters with the
  8 Tier D test sites (c = 1.069) and is negligible for Tier C (hundreds of blocks).
* :func:`leave_one_cluster_out` — the statistic recomputed with each higher-level cluster (NEON
  domain) removed; a P1 pass additionally requires the sign to survive every deletion (§8).
* :func:`hard_strata` — the a-priori difficulty strata of protocol §8.1 (defined from training-fold
  quantiles and issue-time inputs only, never from model results).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...eval.bootstrap import bootstrap_ci


def skill(a, b) -> float:
    """Relative CRPS skill 1 − mean(a) / mean(b) (a = candidate, b = reference)."""
    return float(1.0 - np.mean(a) / np.mean(b))


def cluster_ci(fn, *arrays, groups, n_boot: int = 1000, seed: int = 0, small_cluster: bool = True) -> dict:
    r = bootstrap_ci(fn, *arrays, groups=groups, n_boot=n_boot, seed=seed)
    G = int(r["n_groups"])
    c = float(np.sqrt(G / (G - 1))) if (small_cluster and G > 1) else 1.0
    pt = r["point"]
    lo = pt - c * (pt - r["ci_low"])
    hi = pt + c * (r["ci_high"] - pt)
    return {"point": pt, "ci_low": float(lo), "ci_high": float(hi), "ci_low_raw": r["ci_low"],
            "ci_high_raw": r["ci_high"], "n_groups": G, "factor": c}


def leave_one_cluster_out(fn, *arrays, clusters) -> dict:
    clusters = np.asarray(clusters)
    vals = {}
    for k in np.unique(clusters):
        keep = clusters != k
        if keep.sum() == 0:
            continue
        vals[str(k)] = float(fn(*(np.asarray(a)[keep] for a in arrays)))
    v = np.array(list(vals.values())) if vals else np.array([np.nan])
    return {"min": float(np.nanmin(v)), "max": float(np.nanmax(v)), "by_cluster": vals}


LOWER_TAIL_FAMILIES = ("do_mgl", "do_mgl_min")


def hard_strata(y: np.ndarray, anchor: np.ndarray, y_train: np.ndarray, anchor_train: np.ndarray,
                family: str, own_present: np.ndarray, n_nonhub: np.ndarray, storm: np.ndarray,
                storm_train: np.ndarray) -> dict[str, np.ndarray]:
    """Boolean masks of the a-priori strata (protocol §8.1) for evaluation rows.

    Thresholds come from the *training* rows of the same fold (dev CV) or from all development rows
    (test). ``y``/``anchor`` are model-space; ``storm`` = 3-day antecedent precipitation at issue time.
    """
    lower = family in LOWER_TAIL_FAMILIES
    yt = y_train[np.isfinite(y_train)]
    out = {}
    if yt.size:
        if lower:
            out["ext_p10"] = y < np.quantile(yt, 0.10)
            out["ext_p05"] = y < np.quantile(yt, 0.05)
        else:
            out["ext_p90"] = y > np.quantile(yt, 0.90)
            out["ext_p95"] = y > np.quantile(yt, 0.95)
    dtr = np.abs(y_train - anchor_train)
    dtr = dtr[np.isfinite(dtr)]
    if dtr.size:
        out["change_p90"] = np.abs(y - anchor) > np.quantile(dtr, 0.90)
    out["own_absent"] = ~np.asarray(own_present, bool)
    out["sparse_le2"] = np.asarray(n_nonhub) <= 2
    st = storm_train[np.isfinite(storm_train)]
    if st.size:
        out["storm_p95"] = storm > np.quantile(st, 0.95)
    return {k: np.asarray(v, bool) & np.isfinite(y) for k, v in out.items()}


def n_nonhub_present(df: pd.DataFrame, mods: list[str]) -> np.ndarray:
    return sum(df[f"{m}__present"].fillna(False).astype(int).to_numpy() for m in mods)


def p3_statistic(parts: list[tuple], n_boot: int = 1000, seed: int = 0, small_cluster: bool = True) -> dict:
    """P3 (protocol §8): mean over targets of (CRPS without m − CRPS with m) / CRPS_hub on rows where m
    is present. ``parts`` = [(sites, crps_with, crps_without, crps_hub), ...] one per target; the site
    cluster bootstrap resamples sites jointly across targets (a site's rows of every target move together)."""
    parts = [tuple(np.asarray(x) for x in p) for p in parts if len(p[0])]
    if not parts:
        return {"point": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n_groups": 0}

    allsites = np.unique(np.concatenate([p[0] for p in parts]))
    idx = [np.searchsorted(allsites, p[0]) for p in parts]

    def stat(cnt: np.ndarray | None):
        vals = []
        for (s, a, b, h), ix in zip(parts, idx):
            w = np.ones(len(s)) if cnt is None else cnt[ix]
            if w.sum() == 0:
                continue
            vals.append((np.sum(w * b) - np.sum(w * a)) / np.sum(w * h))
        return float(np.mean(vals)) if vals else np.nan

    rs = np.random.default_rng(seed)
    pt = stat(None)
    G0 = len(allsites)
    bs = [stat(np.bincount(rs.integers(0, G0, G0), minlength=G0).astype(float)) for _ in range(n_boot)]
    bs = np.array([x for x in bs if np.isfinite(x)])
    G = len(allsites)
    f = float(np.sqrt(G / (G - 1))) if (small_cluster and G > 1) else 1.0
    lo, hi = np.quantile(bs, 0.025), np.quantile(bs, 0.975)
    return {"point": pt, "ci_low": float(pt - f * (pt - lo)), "ci_high": float(pt + f * (hi - pt)),
            "n_groups": int(G), "n_targets": len(parts)}
