#!/usr/bin/env python3
"""Cross-region transfer test for the FIB (E. coli) beach-nowcast model.

Applies the SENTINEL transfer-gap thesis to the disease/beach-nowcast model:
does the hydrometeorological E. coli-exceedance relationship learned in one
region generalize to a geographically distinct region, or is it region-specific
(the way microcystin collapsed cross-region)?

Regions: each is a REAL WQP + GHCN pull built by fetch_fib_nowcast_region.py
(Ohio is the original fetch_fib_nowcast_data.py output). Label channel = WQP
E. coli exceedance of the EPA BAV (235 CFU/100mL). Predictors are strictly
antecedent / independent of the label.

Two honest tests:
  (A) CROSS-REGION transfer: train on region X, test on region Y (both
      directions). AUROC vs that test region's base-rate (=0.5 AUROC).
  (B) POOLED leave-one-region-out (LORO): train on all-but-one region, test on
      the held-out region. Also pooled leave-SITES-out (GroupKFold on site).

Two predictor sets, reported side by side so the transfer claim is not propped
up by memorized site coordinates:
  * STRICT (geography-free, deployable): prior-day rainfall (1/2/3/7-day) +
    season (doy_sin/cos, month). NO lat/lon, NO same-day rain, NO ghcn_dist.
  * FULL: strict + rain_sameday_mm + lat + lon + ghcn_dist_km.

Model: HistGradientBoostingClassifier (CPU). Baselines: test-region base-rate
(AUROC=0.5 by construction) and, where a prior sample exists, previous-sample
PERSISTENCE.

Writes results/benchmarks/disease_fib_crossregion.json.
"""
from __future__ import annotations

import json
import time
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupKFold

PROJECT = Path(__file__).resolve().parents[2]
BASE = PROJECT / "data" / "processed" / "fib_nowcast"
OUT = PROJECT / "results" / "benchmarks" / "disease_fib_crossregion.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(42)

STRICT_PREDICTORS = [
    "rain_prior1_mm", "rain_prior2_mm", "rain_prior3_mm", "rain_prior7_mm",
    "doy_sin", "doy_cos", "month",
]
FULL_PREDICTORS = STRICT_PREDICTORS + [
    "rain_sameday_mm", "lat", "lon", "ghcn_dist_km",
]
ALL_COLS = sorted(set(STRICT_PREDICTORS + FULL_PREDICTORS))


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _avg_rank(p):
    """Vectorized average ranks (ties -> mean rank), pure numpy. 1-based."""
    n = len(p)
    order = np.argsort(p, kind="quicksort")
    ps = p[order]
    # ordinal ranks 1..n in sorted position
    ord_rank = np.arange(1, n + 1, dtype=np.float64)
    # group equal values: for each run of ties, assign the mean ordinal rank
    # boundaries where value changes
    is_new = np.empty(n, dtype=bool)
    is_new[0] = True
    is_new[1:] = ps[1:] != ps[:-1]
    group_id = np.cumsum(is_new) - 1          # 0..G-1 per sorted element
    # sum of ordinal ranks per group and count per group
    sums = np.bincount(group_id, weights=ord_rank)
    cnts = np.bincount(group_id)
    mean_rank_per_group = sums / cnts
    ranks_sorted = mean_rank_per_group[group_id]
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = ranks_sorted
    return ranks


def _fast_auroc(y, p):
    """AUROC via the Mann-Whitney U statistic (rank-based, vectorized numpy)."""
    y = np.asarray(y); p = np.asarray(p, dtype=np.float64)
    n_pos = int(y.sum()); n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _avg_rank(p)
    sum_pos = ranks[y == 1].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def bootstrap_auroc_ci(y, p, n=500):
    """Vectorized bootstrap 95% CI for AUROC using the fast rank-based estimator."""
    y = np.asarray(y).astype(int); p = np.asarray(p, dtype=float)
    if len(np.unique(y)) < 2:
        return (float("nan"), float("nan"))
    m = len(y)
    aucs = []
    for _ in range(n):
        b = RNG.integers(0, m, size=m)
        yb = y[b]
        if yb.sum() == 0 or yb.sum() == m:
            continue
        aucs.append(_fast_auroc(yb, p[b]))
    if not aucs:
        return (float("nan"), float("nan"))
    return (float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5)))


def scores(y, p):
    y = np.asarray(y); p = np.asarray(p)
    if len(np.unique(y)) < 2:
        return {"auroc": None, "pr_auc": None, "n": int(len(y)),
                "base_rate": round(float(y.mean()), 4)}
    lo, hi = bootstrap_auroc_ci(y, p)
    return {
        "auroc": round(float(roc_auc_score(y, p)), 4),
        "auroc_ci95": [round(lo, 4), round(hi, 4)],
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "n": int(len(y)),
        "base_rate": round(float(y.mean()), 4),
    }


def fit_gbm(X, y):
    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=4,
        l2_regularization=1.0, early_stopping=True,
        validation_fraction=0.15, random_state=42,
    )
    clf.fit(X, y)
    return clf


def prep(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    for c in ALL_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    rain_cols = [c for c in ALL_COLS if c.startswith("rain")]
    df[rain_cols] = df[rain_cols].fillna(0.0)
    if "ghcn_dist_km" in df.columns:
        df["ghcn_dist_km"] = df["ghcn_dist_km"].fillna(df["ghcn_dist_km"].median())
    return df


def persistence_baseline(df_test_region_full):
    """Previous-sample persistence within a region's own chronological history.
    For each row, predict the last observed exceedance at the SAME site strictly
    before that row's date. Returns (pred, has_prior mask) aligned to df rows."""
    d = df_test_region_full.sort_values("date")
    running = {}
    preds = np.full(len(d), np.nan)
    for pos, row in enumerate(d.itertuples()):
        preds[pos] = running.get(row.site, np.nan)
        running[row.site] = row.exceed
    # realign to the original index order of df_test_region_full
    pred_series = pd.Series(preds, index=d.index)
    return pred_series.reindex(df_test_region_full.index).values


# ---------------------------------------------------------------------------
def cross_region(dfs, predictors, tag):
    """Train on region X, test on region Y for every ordered pair."""
    res = {}
    for x, y in permutations(dfs.keys(), 2):
        dx, dy = dfs[x], dfs[y]
        clf = fit_gbm(dx[predictors].values, dx["exceed"].values)
        p = clf.predict_proba(dy[predictors].values)[:, 1]
        s = scores(dy["exceed"].values, p)
        res[f"{x}->{y}"] = s
        log(f"  [{tag}] {x}->{y}: AUROC={s['auroc']} (test base={s['base_rate']}, n={s['n']})")
    return res


def within_region_baseline(dfs, predictors, tag):
    """Train and test within the SAME region via site-grouped OOF (upper bound
    on transfer). Uses GroupKFold on site so it stays leave-sites-out."""
    res = {}
    for r, d in dfs.items():
        groups = d["site"].values
        X = d[predictors].values
        yv = d["exceed"].values
        n_sites = d["site"].nunique()
        n_splits = min(5, n_sites)
        if n_splits < 2:
            res[r] = {"note": "too few sites"}
            continue
        gkf = GroupKFold(n_splits=n_splits)
        oof = np.full(len(yv), np.nan)
        for tr_i, te_i in gkf.split(X, yv, groups):
            clf = fit_gbm(X[tr_i], yv[tr_i])
            oof[te_i] = clf.predict_proba(X[te_i])[:, 1]
        s = scores(yv, oof)
        res[r] = s
        log(f"  [{tag}] within-{r} leave-sites-out OOF: AUROC={s['auroc']} (base={s['base_rate']})")
    return res


def pooled_loro(dfs, predictors, tag):
    """Leave-one-region-out: train on all other regions, test on held-out one."""
    res = {}
    for held in dfs.keys():
        train = pd.concat([d for r, d in dfs.items() if r != held], ignore_index=True)
        test = dfs[held]
        clf = fit_gbm(train[predictors].values, train["exceed"].values)
        p = clf.predict_proba(test[predictors].values)[:, 1]
        s = scores(test["exceed"].values, p)
        # persistence baseline on the held-out region
        pers = persistence_baseline(test)
        mask = ~np.isnan(pers)
        pers_s = None
        model_on_pers = None
        if mask.sum() > 20 and len(np.unique(test["exceed"].values[mask])) > 1:
            pers_s = scores(test["exceed"].values[mask], pers[mask])
            model_on_pers = scores(test["exceed"].values[mask], p[mask])
        res[held] = {
            "model": s,
            "n_train": int(len(train)),
            "train_regions": [r for r in dfs if r != held],
            "baseline_persistence": pers_s,
            "model_on_persistence_subset": model_on_pers,
        }
        log(f"  [{tag}] LORO test={held}: AUROC={s['auroc']} "
            f"(base={s['base_rate']}, n={s['n']}, "
            f"persist={pers_s['auroc'] if pers_s else None})")
    return res


def pooled_leave_sites_out(dfs, predictors, tag, n_splits=5):
    """Pool all regions, GroupKFold on site (every test site unseen anywhere)."""
    pooled = pd.concat(dfs.values(), ignore_index=True)
    groups = pooled["site"].values
    X = pooled[predictors].values
    yv = pooled["exceed"].values
    gkf = GroupKFold(n_splits=n_splits)
    oof = np.full(len(yv), np.nan)
    fold_auroc = []
    for tr_i, te_i in gkf.split(X, yv, groups):
        clf = fit_gbm(X[tr_i], yv[tr_i])
        oof[te_i] = clf.predict_proba(X[te_i])[:, 1]
        if len(np.unique(yv[te_i])) > 1:
            fold_auroc.append(round(float(roc_auc_score(yv[te_i], oof[te_i])), 4))
    s = scores(yv, oof)
    log(f"  [{tag}] pooled leave-sites-out OOF: AUROC={s['auroc']} (base={s['base_rate']})")
    return {"pooled_oof": s, "per_fold_auroc": fold_auroc,
            "n_sites": int(pooled["site"].nunique()), "n": int(len(pooled))}


def build_verdict(cross_strict, loro_strict, within_strict):
    """Mechanical, honest verdict based on the STRICT (deployable) predictors."""
    # cross-region: mean off-diagonal transfer AUROC
    cr = [v["auroc"] for v in cross_strict.values() if v.get("auroc") is not None]
    cr_mean = round(float(np.mean(cr)), 4) if cr else None
    # within-region ceiling
    wr = [v["auroc"] for v in within_strict.values() if isinstance(v, dict) and v.get("auroc")]
    wr_mean = round(float(np.mean(wr)), 4) if wr else None
    # LORO
    lo = [v["model"]["auroc"] for v in loro_strict.values()
          if v["model"].get("auroc") is not None]
    lo_mean = round(float(np.mean(lo)), 4) if lo else None

    transfers = cr_mean is not None and cr_mean > 0.60
    strong = cr_mean is not None and cr_mean > 0.68
    gap = None
    if wr_mean and cr_mean:
        gap = round(wr_mean - cr_mean, 4)

    if strong:
        verdict = ("TRANSFERS: the antecedent-rainfall + season relationship "
                   "generalizes across regions; cross-region AUROC is close to "
                   "within-region skill.")
    elif transfers:
        verdict = ("PARTIAL transfer: cross-region AUROC beats base-rate clearly "
                   "but sits below within-region skill (a real transfer gap, not "
                   "a collapse).")
    else:
        verdict = ("REGION-SPECIFIC: cross-region AUROC collapses toward base-rate; "
                   "the hydromet relationship does not port to a new region.")
    return {
        "predictor_set_for_verdict": "STRICT (geography-free, deployable)",
        "cross_region_mean_auroc": cr_mean,
        "within_region_mean_auroc_ceiling": wr_mean,
        "leave_one_region_out_mean_auroc": lo_mean,
        "transfer_gap_within_minus_cross": gap,
        "summary": verdict,
    }


def main():
    t0 = time.time()
    # discover available region datasets
    dfs = {}
    metas = {}
    # Ohio is the original build (no region_ prefix)
    oh = BASE / "fib_nowcast_dataset.parquet"
    if oh.exists():
        d = pd.read_parquet(oh)
        d["region"] = "OH"
        dfs["OH"] = prep(d)
        m = BASE / "dataset_meta.json"
        metas["OH"] = json.loads(m.read_text()) if m.exists() else {}
    for sub in sorted(BASE.glob("region_*")):
        p = sub / "fib_nowcast_dataset.parquet"
        if not p.exists():
            continue
        key = sub.name.replace("region_", "")
        d = pd.read_parquet(p)
        d["region"] = key
        dfs[key] = prep(d)
        mp = sub / "dataset_meta.json"
        metas[key] = json.loads(mp.read_text()) if mp.exists() else {}

    log(f"Regions loaded: {list(dfs.keys())}")
    for r, d in dfs.items():
        log(f"  {r}: n={len(d)} sites={d.site.nunique()} "
            f"pos={int(d.exceed.sum())} base={d.exceed.mean():.4f}")

    if len(dfs) < 2:
        raise SystemExit("Need >=2 regions for a cross-region test.")

    OH_STATECODE = "US:39"  # Ohio original meta stores 'region' not 'statecode'
    region_summary = {
        r: {
            "region_name": metas.get(r, {}).get("region_name",
                                                metas.get(r, {}).get("region", r)),
            "statecode": metas.get(r, {}).get("statecode",
                                              OH_STATECODE if r == "OH" else None),
            "n_samples": int(len(d)),
            "n_sites": int(d.site.nunique()),
            "n_positives": int(d.exceed.sum()),
            "base_rate": round(float(d.exceed.mean()), 4),
            "date_range": metas.get(r, {}).get("date_range"),
        }
        for r, d in dfs.items()
    }

    log("=== CROSS-REGION transfer (STRICT predictors) ===")
    cross_strict = cross_region(dfs, STRICT_PREDICTORS, "strict")
    log("=== CROSS-REGION transfer (FULL predictors) ===")
    cross_full = cross_region(dfs, FULL_PREDICTORS, "full")

    log("=== WITHIN-REGION leave-sites-out ceiling (STRICT) ===")
    within_strict = within_region_baseline(dfs, STRICT_PREDICTORS, "strict")
    log("=== WITHIN-REGION leave-sites-out ceiling (FULL) ===")
    within_full = within_region_baseline(dfs, FULL_PREDICTORS, "full")

    log("=== POOLED leave-one-region-out (STRICT) ===")
    loro_strict = pooled_loro(dfs, STRICT_PREDICTORS, "strict")
    log("=== POOLED leave-one-region-out (FULL) ===")
    loro_full = pooled_loro(dfs, FULL_PREDICTORS, "full")

    log("=== POOLED leave-SITES-out (STRICT) ===")
    plso_strict = pooled_leave_sites_out(dfs, STRICT_PREDICTORS, "strict")
    log("=== POOLED leave-SITES-out (FULL) ===")
    plso_full = pooled_leave_sites_out(dfs, FULL_PREDICTORS, "full")

    verdict = build_verdict(cross_strict, loro_strict, within_strict)

    out = {
        "task": "beach_nowcast_fib_exceedance_CROSS_REGION_TRANSFER",
        "target": "E. coli exceedance of EPA BAV (235 CFU/100mL, 2012 RWQC single-sample)",
        "thesis": ("Transfer-gap thesis applied to the beach-nowcast disease model: "
                   "does the antecedent-hydromet -> E. coli exceedance relationship "
                   "port across geographically distinct regions?"),
        "non_circularity": (
            "Predictors are strictly antecedent/independent (NOAA GHCN prior-day "
            "rainfall + season [+ site lat/lon in FULL only]). Label is a separate "
            "WQP E. coli channel. No FIB value or function of it is a predictor."),
        "predictor_sets": {
            "STRICT_geography_free_deployable": STRICT_PREDICTORS,
            "FULL": FULL_PREDICTORS,
        },
        "data_sources": {
            "label": "WQP Result service, characteristic 'Escherichia coli', exceed>235 CFU/100mL",
            "rainfall": "NOAA GHCN-Daily (NCEI daily-summaries), nearest-with-data station",
        },
        "model": "HistGradientBoostingClassifier (CPU)",
        "regions": region_summary,
        "cross_region_transfer": {
            "note": ("train on region X, test on region Y. AUROC=0.5 == test-region "
                     "base-rate (chance). Compare to within-region ceiling below."),
            "strict_predictors": cross_strict,
            "full_predictors": cross_full,
        },
        "within_region_leave_sites_out_ceiling": {
            "note": "same-region GroupKFold-on-site OOF; upper bound for transfer.",
            "strict_predictors": within_strict,
            "full_predictors": within_full,
        },
        "pooled_leave_one_region_out": {
            "note": ("train on all-but-one region, test on held-out region. "
                     "Includes previous-sample persistence baseline on the held-out region."),
            "strict_predictors": loro_strict,
            "full_predictors": loro_full,
        },
        "pooled_leave_sites_out": {
            "note": "pool all regions, GroupKFold on site (every test site unseen anywhere).",
            "strict_predictors": plso_strict,
            "full_predictors": plso_full,
        },
        "verdict": verdict,
        "eval_seconds": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(out, indent=2))
    log(f"WROTE {OUT}")
    print("\n===== CROSS-REGION VERDICT =====")
    print(verdict["summary"])
    print(f"cross-region mean AUROC (strict) = {verdict['cross_region_mean_auroc']}")
    print(f"within-region ceiling (strict)   = {verdict['within_region_mean_auroc_ceiling']}")
    print(f"LORO mean AUROC (strict)         = {verdict['leave_one_region_out_mean_auroc']}")
    print(f"transfer gap                     = {verdict['transfer_gap_within_minus_cross']}")


if __name__ == "__main__":
    main()
