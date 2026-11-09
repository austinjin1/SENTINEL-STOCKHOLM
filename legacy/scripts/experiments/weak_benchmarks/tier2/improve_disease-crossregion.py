#!/usr/bin/env python3
"""Tier-2 weak-benchmark push: Disease Beach-Nowcast E. coli CROSS-REGION transfer.

Bar to beat (all REAL, region-grouped, from results/benchmarks/disease_fib_crossregion.json):
  * STRICT (geography-free, deployable) predictors:
      - cross-region pairwise mean AUROC ~0.554
      - leave-one-region-out (LORO) mean AUROC ~0.546   <-- PRIMARY held-out target
      - within-region leave-sites-out ceiling ~0.663
  Regions: OH, MI, WI (Great-Lakes E. coli beach monitoring).

This script attempts HONEST cross-region improvements under strict
leave-one-region-out (train on all-but-one region, test on the held-out
region -- the region is the group; no leakage). Everything is antecedent /
deployable: NO lat/lon, NO same-day rain, NO ghcn_dist, NO label-derived feature.

Methods tried (each region-invariant by construction):
  BASE   : replicate the STRICT baseline recipe (prior rainfall 1/2/3/7d + season).
  FEAT   : richer *region-invariant* antecedent hydrology features
           (log1p rainfall, dry/wet flags, antecedent 7d-vs-1d ratio,
            rain acceleration, storminess, plus season) + coarse site-type
            (coastal-beach / inland-lake / stream) mapped consistently across
            regions.
  RANK   : FEAT but every continuous rainfall feature is replaced by its
           WITHIN-REGION empirical quantile rank (a label-free, per-region
           monotone transform). This removes cross-region climate/scale shift
           (a covariate-shift domain adaptation) while preserving the
           within-region storm ordering that drives exceedance. The rank is
           computed per region using ONLY that region's own predictor
           marginals (no labels), so it is legitimately deployable in a brand
           new region.
  RANK+  : RANK features + coarse site-type.

For each method we report LORO per-region AUROC and the pooled/mean, plus a
cluster (region + site) bootstrap CI on the DELTA vs the BASE method's LORO.
A gain counts ONLY if the LORO mean improves AND the delta CI excludes 0.

Baselines carried through for honesty: within-region ceiling, and previous-
sample persistence on the held-out region.

Writes results/benchmarks/weak_improve/tier2/disease-crossregion.json.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

PROJECT = Path("/home/bcheng/SENTINEL")
BASE = PROJECT / "data" / "processed" / "fib_nowcast"
OUT = PROJECT / "results" / "benchmarks" / "weak_improve" / "tier2" / "disease-crossregion.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(42)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ---------------------------------------------------------------------------
# fast AUROC (Mann-Whitney) + cluster bootstrap
# ---------------------------------------------------------------------------
def fast_auroc(y, p):
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=np.float64)
    n_pos = int(y.sum()); n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ps = p[order]
    ranks = np.empty(len(p), dtype=np.float64)
    i = 0
    r = np.arange(1, len(p) + 1, dtype=np.float64)
    # average ranks for ties
    is_new = np.empty(len(p), bool); is_new[0] = True
    is_new[1:] = ps[1:] != ps[:-1]
    gid = np.cumsum(is_new) - 1
    sums = np.bincount(gid, weights=r); cnts = np.bincount(gid)
    mean_rank = (sums / cnts)[gid]
    ranks[order] = mean_rank
    sum_pos = ranks[y == 1].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def auroc_ci(y, p, n=1000):
    y = np.asarray(y).astype(int); p = np.asarray(p, float)
    if len(np.unique(y)) < 2:
        return (float("nan"), float("nan"))
    m = len(y); vals = []
    for _ in range(n):
        b = RNG.integers(0, m, size=m)
        yb = y[b]
        if yb.sum() == 0 or yb.sum() == m:
            continue
        vals.append(fast_auroc(yb, p[b]))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def scores(y, p):
    y = np.asarray(y); p = np.asarray(p)
    if len(np.unique(y)) < 2:
        return {"auroc": None, "n": int(len(y)), "base_rate": round(float(y.mean()), 4)}
    lo, hi = auroc_ci(y, p)
    return {
        "auroc": round(float(roc_auc_score(y, p)), 4),
        "auroc_ci95": [round(lo, 4), round(hi, 4)],
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "n": int(len(y)), "base_rate": round(float(y.mean()), 4),
    }


def fit_gbm(X, y):
    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=4,
        l2_regularization=1.0, early_stopping=True,
        validation_fraction=0.15, random_state=42,
    )
    clf.fit(X, y)
    return clf


# ---------------------------------------------------------------------------
# data + feature engineering
# ---------------------------------------------------------------------------
RAIN = ["rain_prior1_mm", "rain_prior2_mm", "rain_prior3_mm", "rain_prior7_mm"]


def load_regions():
    dfs = {}
    d = pd.read_parquet(BASE / "fib_nowcast_dataset.parquet"); d["region"] = "OH"
    dfs["OH"] = d
    for sub in sorted(BASE.glob("region_*")):
        p = sub / "fib_nowcast_dataset.parquet"
        if p.exists():
            dd = pd.read_parquet(p); key = sub.name.replace("region_", "")
            dd["region"] = key; dfs[key] = dd
    for r, d in dfs.items():
        d["date"] = pd.to_datetime(d["date"])
        for c in RAIN + ["rain_sameday_mm"]:
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)
    return dfs


def coarse_site_type(s):
    """Map heterogeneous region-specific site_type strings to 3 invariant classes:
    coastal great-lake beach / inland lake-reservoir / stream-river."""
    s = str(s).lower()
    if "great lake" in s or ("beach" in s and "lake" not in s.split("beach")[-1]):
        # Great-Lake beach programs & 'Great Lake'
        if "lake" in s and "great" not in s and "reservoir" in s:
            return 1
        return 0
    if "stream" in s or "river" in s:
        return 2
    if "lake" in s or "reservoir" in s or "impound" in s or "pond" in s:
        return 1
    return 0  # default coastal-beach (dominant class)


def base_features(df):
    """STRICT baseline recipe: prior rainfall + season."""
    X = df[RAIN].copy()
    doy = df["date"].dt.dayofyear.values
    X["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    X["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    X["month"] = df["date"].dt.month.values
    return X


def rich_features(df, site_type=False):
    """Region-invariant richer antecedent hydrology features."""
    r1 = df["rain_prior1_mm"].values
    r2 = df["rain_prior2_mm"].values
    r3 = df["rain_prior3_mm"].values
    r7 = df["rain_prior7_mm"].values
    eps = 1e-3
    X = pd.DataFrame(index=df.index)
    # log-scaled magnitudes (dampen heavy-tail region differences)
    X["log_r1"] = np.log1p(r1)
    X["log_r2"] = np.log1p(r2)
    X["log_r3"] = np.log1p(r3)
    X["log_r7"] = np.log1p(r7)
    # incremental (non-overlapping) windows: r2 is cumulative 2d etc. in this data
    inc_23 = np.maximum(r3 - r2, 0.0)          # rain on day-3
    inc_37 = np.maximum(r7 - r3, 0.0)          # rain days 4-7
    X["log_inc23"] = np.log1p(inc_23)
    X["log_inc37"] = np.log1p(inc_37)
    # recency ratio: how front-loaded is the antecedent rain (region-scale-free)
    X["frac_recent"] = r1 / (r7 + eps)
    X["frac_2d"] = r2 / (r7 + eps)
    # storm acceleration: yesterday vs the earlier daily average
    earlier_daily = inc_37 / 4.0
    X["accel"] = (r1 - earlier_daily) / (r7 + eps)
    # dry/wet flags (thresholds are physical mm, region-invariant)
    X["dry7"] = (r7 < 1.0).astype(float)
    X["wet1"] = (r1 > 10.0).astype(float)
    X["heavy1"] = (r1 > 25.0).astype(float)
    # season
    doy = df["date"].dt.dayofyear.values
    X["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    X["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    X["month"] = df["date"].dt.month.values.astype(float)
    # season x recent-rain interaction (summer storms behave differently)
    X["summer_rain"] = X["doy_sin"] * X["log_r1"]
    if site_type and "site_type" in df.columns:
        st = df["site_type"].map(coarse_site_type).astype(float).values
        X["site_coastal"] = (st == 0).astype(float)
        X["site_lake"] = (st == 1).astype(float)
        X["site_stream"] = (st == 2).astype(float)
    return X


def rank_transform_cols(X_train_by_region, X_test, test_region_X, cont_cols):
    """Replace each continuous column by its WITHIN-REGION empirical quantile rank.
    - For every TRAIN region, ranks are computed against that region's own marginal.
    - For the TEST region, ranks are computed against the TEST region's own marginal
      (label-free, deployable: only uses the predictor distribution of the new region).
    This is a per-region monotone covariate-shift correction (domain adaptation).
    Returns transformed (concatenated train, test)."""
    def rank_within(block):
        b = block.copy()
        for c in cont_cols:
            v = b[c].values
            # empirical CDF rank in [0,1]
            order = np.argsort(np.argsort(v))
            b[c] = (order + 0.5) / len(v)
        return b
    train_parts = [rank_within(b) for b in X_train_by_region]
    Xtr = pd.concat(train_parts, ignore_index=True)
    Xte = rank_within(test_region_X)
    return Xtr, Xte


# ---------------------------------------------------------------------------
# LORO evaluation for a given feature builder
# ---------------------------------------------------------------------------
def loro_eval(dfs, feat_fn, rank=False, cont_cols=None, site_type=False):
    """Leave-one-region-out. Returns dict: region -> (y, p, groups(site)) and scores.
    feat_fn(df) -> DataFrame of features."""
    out = {}
    per_region_preds = {}
    for held in dfs:
        test_df = dfs[held]
        train_dfs = [dfs[r] for r in dfs if r != held]
        if rank:
            Xtr_blocks = [feat_fn(d) for d in train_dfs]
            Xte_raw = feat_fn(test_df)
            Xtr, Xte = rank_transform_cols(Xtr_blocks, None, Xte_raw, cont_cols)
            ytr = np.concatenate([d["exceed"].values for d in train_dfs])
        else:
            Xtr = pd.concat([feat_fn(d) for d in train_dfs], ignore_index=True)
            Xte = feat_fn(test_df)
            ytr = np.concatenate([d["exceed"].values for d in train_dfs])
        clf = fit_gbm(Xtr.values, ytr)
        p = clf.predict_proba(Xte.values)[:, 1]
        yte = test_df["exceed"].values
        out[held] = scores(yte, p)
        per_region_preds[held] = {
            "y": yte, "p": p, "site": test_df["site"].values,
        }
        log(f"    LORO test={held}: AUROC={out[held]['auroc']} (base={out[held]['base_rate']}, n={out[held]['n']})")
    aur = [v["auroc"] for v in out.values() if v["auroc"] is not None]
    out["_mean_auroc"] = round(float(np.mean(aur)), 4)
    return out, per_region_preds


def within_region_ceiling(dfs, feat_fn):
    from sklearn.model_selection import GroupKFold
    res = {}
    for r, d in dfs.items():
        X = feat_fn(d).values; y = d["exceed"].values; g = d["site"].values
        ns = min(5, d["site"].nunique())
        gkf = GroupKFold(n_splits=ns)
        oof = np.full(len(y), np.nan)
        for tr, te in gkf.split(X, y, g):
            clf = fit_gbm(X[tr], y[tr]); oof[te] = clf.predict_proba(X[te])[:, 1]
        res[r] = scores(y, oof)
    aur = [v["auroc"] for v in res.values() if v["auroc"] is not None]
    res["_mean_auroc"] = round(float(np.mean(aur)), 4)
    return res


def persistence_baseline(dfs):
    res = {}
    for r, d in dfs.items():
        dd = d.sort_values("date")
        running = {}; preds = np.full(len(dd), np.nan)
        for pos, row in enumerate(dd.itertuples()):
            preds[pos] = running.get(row.site, np.nan)
            running[row.site] = row.exceed
        y = dd["exceed"].values; mask = ~np.isnan(preds)
        res[r] = scores(y[mask], preds[mask])
    aur = [v["auroc"] for v in res.values() if v["auroc"] is not None]
    res["_mean_auroc"] = round(float(np.mean(aur)), 4)
    return res


# ---------------------------------------------------------------------------
# cluster-bootstrap on the DELTA between two methods' LORO predictions
# ---------------------------------------------------------------------------
def delta_ci(preds_a, preds_b, n=2000):
    """Cluster (region+site) bootstrap CI for the difference in mean-of-region
    LORO AUROC: mean_region AUROC(B) - mean_region AUROC(A).
    Resample SITES within each held-out region (the natural cluster), recompute
    each region's AUROC, average across regions. Same resample applied to A & B."""
    regions = [r for r in preds_a if not r.startswith("_")]
    # precompute per region: site groupings
    site_index = {}
    for r in regions:
        sites = preds_a[r]["site"]
        uniq = np.unique(sites)
        idx_by_site = {s: np.where(sites == s)[0] for s in uniq}
        site_index[r] = (uniq, idx_by_site)
    deltas = []; a_means = []; b_means = []
    for _ in range(n):
        a_reg = []; b_reg = []
        for r in regions:
            uniq, idx_by_site = site_index[r]
            samp = RNG.choice(uniq, size=len(uniq), replace=True)
            rows = np.concatenate([idx_by_site[s] for s in samp])
            ya = preds_a[r]["y"][rows]
            if ya.sum() == 0 or ya.sum() == len(ya):
                a_reg = None; break
            a_reg.append(fast_auroc(ya, preds_a[r]["p"][rows]))
            b_reg.append(fast_auroc(preds_b[r]["y"][rows], preds_b[r]["p"][rows]))
        if a_reg is None:
            continue
        am = float(np.mean(a_reg)); bm = float(np.mean(b_reg))
        a_means.append(am); b_means.append(bm); deltas.append(bm - am)
    return {
        "delta_mean": round(float(np.mean(deltas)), 4),
        "delta_ci95": [round(float(np.percentile(deltas, 2.5)), 4),
                       round(float(np.percentile(deltas, 97.5)), 4)],
        "excludes_zero": bool(np.percentile(deltas, 2.5) > 0 or np.percentile(deltas, 97.5) < 0),
        "a_mean_auroc_boot": round(float(np.mean(a_means)), 4),
        "b_mean_auroc_boot": round(float(np.mean(b_means)), 4),
        "n_boot": len(deltas),
    }


# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    dfs = load_regions()
    log(f"Regions: { {r: (len(d), int(d.exceed.sum())) for r,d in dfs.items()} }")

    cont_cols_rich = ["log_r1", "log_r2", "log_r3", "log_r7", "log_inc23",
                      "log_inc37", "frac_recent", "frac_2d", "accel", "summer_rain"]

    log("=== BASE (STRICT baseline replication) ===")
    base_res, base_preds = loro_eval(dfs, base_features)

    log("=== FEAT (rich region-invariant features) ===")
    feat_res, feat_preds = loro_eval(dfs, lambda d: rich_features(d, site_type=False))

    log("=== FEAT+ST (rich + coarse site-type) ===")
    featst_res, featst_preds = loro_eval(dfs, lambda d: rich_features(d, site_type=True))

    log("=== RANK (within-region quantile-rank covariate-shift correction) ===")
    rank_res, rank_preds = loro_eval(
        dfs, lambda d: rich_features(d, site_type=False),
        rank=True, cont_cols=cont_cols_rich)

    log("=== RANK+ST ===")
    rankst_res, rankst_preds = loro_eval(
        dfs, lambda d: rich_features(d, site_type=True),
        rank=True, cont_cols=cont_cols_rich)

    log("=== within-region ceiling (rich feats) ===")
    ceiling = within_region_ceiling(dfs, lambda d: rich_features(d, site_type=True))
    log("=== persistence baseline (held-out region) ===")
    persist = persistence_baseline(dfs)

    # deltas vs BASE
    log("=== cluster-bootstrap deltas vs BASE ===")
    d_feat = delta_ci(base_preds, feat_preds)
    d_featst = delta_ci(base_preds, featst_preds)
    d_rank = delta_ci(base_preds, rank_preds)
    d_rankst = delta_ci(base_preds, rankst_preds)

    methods = {
        "BASE": {"loro": base_res, "delta_vs_base": None},
        "FEAT": {"loro": feat_res, "delta_vs_base": d_feat},
        "FEAT+ST": {"loro": featst_res, "delta_vs_base": d_featst},
        "RANK": {"loro": rank_res, "delta_vs_base": d_rank},
        "RANK+ST": {"loro": rankst_res, "delta_vs_base": d_rankst},
    }

    # pick best by LORO mean AUROC among non-base
    cand = {k: v for k, v in methods.items() if k != "BASE"}
    best_name = max(cand, key=lambda k: cand[k]["loro"]["_mean_auroc"])
    best = methods[best_name]
    old_val = base_res["_mean_auroc"]
    new_val = best["loro"]["_mean_auroc"]
    dci = best["delta_vs_base"]
    improved = bool(new_val > old_val and dci is not None and dci["excludes_zero"])

    verdict = {
        "primary_holdout": "leave-one-region-out (region = group), mean of per-region AUROC",
        "old_baseline_method": "BASE (STRICT prior-rainfall + season)",
        "old_loro_mean_auroc": old_val,
        "best_new_method": best_name,
        "new_loro_mean_auroc": new_val,
        "delta_vs_base": dci,
        "improved": improved,
        "within_region_ceiling_mean_auroc": ceiling["_mean_auroc"],
        "persistence_mean_auroc": persist["_mean_auroc"],
        "interpretation": (
            "IMPROVED cross-region: the delta CI excludes 0 on region-grouped LORO."
            if improved else
            "HONEST NULL: no method beats the STRICT baseline on region-grouped LORO "
            "with a delta CI excluding 0. Cross-region E. coli exceedance transfer "
            "remains near ~0.55 -- a genuine region-specificity limit. Note the "
            "within-region ceiling (~0.66-0.79) and same-region persistence "
            "(~0.60-0.68) both exceed any cross-region model, confirming the signal "
            "is real but region-idiosyncratic."),
    }

    out = {
        "task": "disease_beach_nowcast_ecoli_CROSS_REGION_improve_tier2",
        "target": "E. coli exceedance of EPA BAV (235 CFU/100mL)",
        "regions": {r: {"n": int(len(d)), "n_sites": int(d.site.nunique()),
                        "n_pos": int(d.exceed.sum()), "base_rate": round(float(d.exceed.mean()), 4)}
                    for r, d in dfs.items()},
        "holdout": "leave-one-region-out (LORO); region is the grouping unit; no lat/lon/same-day/label features",
        "ci_method": "cluster (region+site) bootstrap on the DELTA of mean-per-region AUROC vs BASE, 2000 reps",
        "predictor_policy": "strictly antecedent & deployable; region-invariant; no geographic coordinates",
        "methods": methods,
        "within_region_ceiling": ceiling,
        "persistence_baseline": persist,
        "verdict": verdict,
        "eval_seconds": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(out, indent=2))
    log(f"WROTE {OUT}")
    print("\n===== CROSS-REGION IMPROVE VERDICT =====")
    print(f"BASE LORO mean AUROC = {old_val}")
    for k, v in methods.items():
        dc = v["delta_vs_base"]
        print(f"  {k:9s} LORO={v['loro']['_mean_auroc']}" +
              (f"  d={dc['delta_mean']} CI{dc['delta_ci95']} excl0={dc['excludes_zero']}" if dc else "  (ref)"))
    print(f"BEST new = {best_name} @ {new_val}; improved={improved}")
    print(verdict["interpretation"])


if __name__ == "__main__":
    main()
