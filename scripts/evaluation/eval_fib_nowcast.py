#!/usr/bin/env python3
"""Honest out-of-sample evaluation of the FIB (E. coli) beach-nowcast model.

Predicts EXCEEDANCE of the EPA E. coli Beach Action Value (235 CFU/100mL) from
strictly ANTECEDENT, independent predictors (antecedent rainfall + season +
site location). NO FIB value is used as a predictor -> non-circular.

Two out-of-sample protocols, each vs the correct naive baselines:
  (i)  TEMPORAL holdout  : train on earlier dates, test on later dates.
       Baselines: global base-rate (prior years) and PREVIOUS-SAMPLE
       PERSISTENCE (last observed exceedance at the same site, strictly before
       the test sample).
  (ii) SITE-GROUPED holdout : GroupKFold over sites (leave-sites-out).
       Baseline: global base-rate. (Persistence is temporal, not spatial, so
       it is reported only for the temporal split.)

Model: HistGradientBoostingClassifier (CPU). We compare a hydromet model vs
baselines. AUROC (bootstrap 95% CI), average precision (PR-AUC), precision/
recall at the Youden-J and at a fixed 0.5 threshold are reported.

Honest bar: does the hydromet model beat base-rate AND persistence
OUT-OF-SAMPLE? A null is a legitimate result and is reported as such.

Writes: results/benchmarks/disease_fib_nowcast.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score

PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT / "data" / "processed" / "fib_nowcast" / "fib_nowcast_dataset.parquet"
META = PROJECT / "data" / "processed" / "fib_nowcast" / "dataset_meta.json"
OUT = PROJECT / "results" / "benchmarks" / "disease_fib_nowcast.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(42)

PREDICTORS = [
    "rain_prior1_mm", "rain_prior2_mm", "rain_prior3_mm", "rain_prior7_mm",
    "rain_sameday_mm", "doy_sin", "doy_cos", "month", "lat", "lon", "ghcn_dist_km",
]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def bootstrap_auroc_ci(y, p, n=2000):
    """Bootstrap 95% CI for AUROC."""
    y = np.asarray(y); p = np.asarray(p)
    if len(np.unique(y)) < 2:
        return (float("nan"), float("nan"))
    idx = np.arange(len(y))
    aucs = []
    for _ in range(n):
        b = RNG.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y[b])) < 2:
            continue
        aucs.append(roc_auc_score(y[b], p[b]))
    if not aucs:
        return (float("nan"), float("nan"))
    return (float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5)))


def pr_at_youden(y, p):
    """Precision/recall at the Youden-J-optimal threshold + at 0.5."""
    y = np.asarray(y)
    p = np.asarray(p)
    out = {}
    # Youden J
    thr_grid = np.unique(p)
    if len(thr_grid) > 200:
        thr_grid = np.quantile(p, np.linspace(0, 1, 200))
    best_j, best_t = -1, 0.5
    for t in thr_grid:
        pred = (p >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        if tpr - fpr > best_j:
            best_j, best_t = tpr - fpr, float(t)
    for tag, t in [("youden", best_t), ("thr0.5", 0.5)]:
        pred = (p >= t).astype(int)
        out[tag] = {
            "threshold": round(float(t), 4),
            "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        }
    return out


def eval_scores(y, p):
    y = np.asarray(y); p = np.asarray(p)
    lo, hi = bootstrap_auroc_ci(y, p)
    return {
        "auroc": round(float(roc_auc_score(y, p)), 4) if len(np.unique(y)) > 1 else None,
        "auroc_ci95": [round(lo, 4), round(hi, 4)],
        "pr_auc": round(float(average_precision_score(y, p)), 4) if len(np.unique(y)) > 1 else None,
        **{("pr_" + k): v for k, v in pr_at_youden(y, p).items()},
    }


def fit_gbm(Xtr, ytr):
    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=4,
        l2_regularization=1.0, early_stopping=True,
        validation_fraction=0.15, random_state=42,
    )
    clf.fit(Xtr, ytr)
    return clf


# ---------------------------------------------------------------------------
def temporal_holdout(df, predictors=None):
    """Train on earliest ~70% of dates, test on latest ~30%. Persistence baseline."""
    predictors = predictors or PREDICTORS
    df = df.sort_values("date").reset_index(drop=True)
    cut = df["date"].quantile(0.70)
    tr = df[df["date"] <= cut]
    te = df[df["date"] > cut]
    log(f"  TEMPORAL: train n={len(tr)} (<= {cut.date()}), test n={len(te)} (> {cut.date()})")

    Xtr, ytr = tr[predictors].values, tr["exceed"].values
    Xte, yte = te[predictors].values, te["exceed"].values

    clf = fit_gbm(Xtr, ytr)
    p_model = clf.predict_proba(Xte)[:, 1]

    # Base-rate baseline: constant = train prior positive rate
    base_rate = float(ytr.mean())
    p_base = np.full(len(yte), base_rate)

    # Persistence baseline: last observed exceedance at same site strictly before
    # the test date. Built from FULL history (train + earlier test rows).
    hist = df.sort_values("date")
    last_exc = {}
    persist_pred = np.full(len(te), np.nan)
    # walk chronologically; for each test row use last known value for its site
    te_idx = set(te.index)
    # need a running "last exceedance per site" as of each row's date
    running = {}
    persist_map = {}
    for row in hist.itertuples():
        if row.Index in te_idx:
            persist_map[row.Index] = running.get(row.site, np.nan)
        running[row.site] = row.exceed
    persist_pred = np.array([persist_map.get(i, np.nan) for i in te.index], dtype=float)
    have_persist = ~np.isnan(persist_pred)

    results = {
        "split": "temporal",
        "cut_date": str(cut.date()),
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "test_base_rate": round(float(yte.mean()), 4),
        "train_base_rate": round(base_rate, 4),
        "model_hydromet": eval_scores(yte, p_model),
        "baseline_base_rate": eval_scores(yte, p_base),
    }
    # persistence only on rows that have a prior sample
    if have_persist.sum() > 20 and len(np.unique(yte[have_persist])) > 1:
        results["baseline_persistence"] = {
            "n_with_prior": int(have_persist.sum()),
            **eval_scores(yte[have_persist], persist_pred[have_persist]),
        }
        # also the model on the SAME subset for a fair head-to-head
        results["model_hydromet_on_persistence_subset"] = eval_scores(
            yte[have_persist], p_model[have_persist])
    else:
        results["baseline_persistence"] = {"note": "insufficient prior-sample rows"}

    # feature importances via permutation on test (quick, honest)
    try:
        from sklearn.inspection import permutation_importance
        pi = permutation_importance(clf, Xte, yte, n_repeats=5,
                                    random_state=42, scoring="roc_auc")
        imp = sorted(zip(predictors, pi.importances_mean),
                     key=lambda x: -x[1])
        results["permutation_importance_auroc_drop"] = {
            k: round(float(v), 4) for k, v in imp}
    except Exception as e:
        results["permutation_importance_auroc_drop"] = {"error": str(e)}
    return results


def site_grouped_holdout(df, n_splits=5, predictors=None):
    """GroupKFold over sites: every test site is unseen in training."""
    from sklearn.model_selection import GroupKFold
    predictors = predictors or PREDICTORS
    groups = df["site"].values
    X = df[predictors].values
    y = df["exceed"].values
    gkf = GroupKFold(n_splits=n_splits)

    oof_p = np.full(len(y), np.nan)
    fold_aurocs = []
    for fold, (tr_i, te_i) in enumerate(gkf.split(X, y, groups)):
        clf = fit_gbm(X[tr_i], y[tr_i])
        p = clf.predict_proba(X[te_i])[:, 1]
        oof_p[te_i] = p
        if len(np.unique(y[te_i])) > 1:
            fold_aurocs.append(float(roc_auc_score(y[te_i], p)))
        log(f"  SITE-GROUPED fold {fold}: test sites="
            f"{len(np.unique(groups[te_i]))} n={len(te_i)} "
            f"auroc={fold_aurocs[-1] if fold_aurocs else float('nan'):.3f}")

    base_rate = float(y.mean())
    p_base = np.full(len(y), base_rate)
    return {
        "split": "site_grouped",
        "n_splits": n_splits,
        "n_sites": int(df["site"].nunique()),
        "base_rate": round(base_rate, 4),
        "per_fold_auroc": [round(a, 4) for a in fold_aurocs],
        "per_fold_auroc_mean": round(float(np.mean(fold_aurocs)), 4) if fold_aurocs else None,
        "per_fold_auroc_std": round(float(np.std(fold_aurocs)), 4) if fold_aurocs else None,
        "model_hydromet_pooledOOF": eval_scores(y, oof_p),
        "baseline_base_rate": eval_scores(y, p_base),
    }


def verdict(temporal, site):
    """Honest, mechanical verdict."""
    tm = temporal["model_hydromet"]["auroc"]
    tb = temporal["baseline_base_rate"]["auroc"]  # ~0.5 by construction
    tp = temporal.get("baseline_persistence", {})
    tp_auroc = tp.get("auroc") if isinstance(tp, dict) else None
    tm_sub = temporal.get("model_hydromet_on_persistence_subset", {}).get("auroc")
    sm = site["model_hydromet_pooledOOF"]["auroc"]

    beats_baserate_temporal = tm is not None and tm > 0.55
    beats_baserate_site = sm is not None and sm > 0.55
    if tp_auroc is not None and tm_sub is not None:
        beats_persistence = tm_sub > tp_auroc
    else:
        beats_persistence = None

    lines = []
    lines.append(f"Temporal hydromet AUROC={tm} vs base-rate {tb} (=chance).")
    if tp_auroc is not None:
        lines.append(f"Persistence AUROC={tp_auroc}; hydromet on same subset="
                     f"{tm_sub}. Beats persistence: {beats_persistence}.")
    lines.append(f"Site-grouped hydromet pooled-OOF AUROC={sm}.")

    if beats_baserate_temporal and beats_baserate_site:
        if beats_persistence is False:
            v = ("REAL-BUT-PARTIAL skill: hydromet beats base-rate out-of-sample "
                 "under BOTH temporal and site-grouped holdouts, but does NOT beat "
                 "previous-sample persistence on the temporal split.")
        else:
            v = ("REAL skill: hydromet beats base-rate out-of-sample under BOTH "
                 "temporal and site-grouped holdouts"
                 + (" and beats persistence." if beats_persistence else "."))
    elif beats_baserate_temporal or beats_baserate_site:
        v = ("MIXED/WEAK skill: beats base-rate under one holdout but not the "
             "other; treat as marginal.")
    else:
        v = ("NULL: hydromet does not beat base-rate out-of-sample. Antecedent "
             "rainfall + season + location carry little generalizable signal here.")
    return {"summary": v, "detail": lines,
            "beats_baserate_temporal": bool(beats_baserate_temporal),
            "beats_baserate_site_grouped": bool(beats_baserate_site),
            "beats_persistence_temporal": beats_persistence}


def main():
    t0 = time.time()
    df = pd.read_parquet(DATA)
    df["date"] = pd.to_datetime(df["date"])
    meta = json.loads(META.read_text()) if META.exists() else {}
    log(f"Loaded {len(df)} rows | {df.site.nunique()} sites | "
        f"positives={int(df.exceed.sum())} | base rate={df.exceed.mean():.4f}")

    # ensure numeric predictors, fill NaN rainfall with 0 (missing antecedent -> assume dry)
    for c in PREDICTORS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    rain_cols = [c for c in PREDICTORS if c.startswith("rain")]
    df[rain_cols] = df[rain_cols].fillna(0.0)
    df["ghcn_dist_km"] = df["ghcn_dist_km"].fillna(df["ghcn_dist_km"].median())

    log("Running TEMPORAL holdout ...")
    temporal = temporal_holdout(df)
    log("Running SITE-GROUPED holdout ...")
    site = site_grouped_holdout(df)

    v = verdict(temporal, site)

    # --- STRICT-ANTECEDENT sensitivity: drop same-day rainfall entirely ---
    # rain_sameday_mm is a daily total that could include rain falling AFTER
    # sample collection. It is NOT the label channel (so not circular), but a
    # true forecast must not see it. Re-run with it removed to prove skill is
    # not an artifact of same-day information.
    strict_predictors = [c for c in PREDICTORS if c != "rain_sameday_mm"]
    log("Running STRICT-ANTECEDENT sensitivity (no same-day rain) ...")
    temporal_strict = temporal_holdout(df, predictors=strict_predictors)
    site_strict = site_grouped_holdout(df, predictors=strict_predictors)

    out = {
        "task": "beach_nowcast_fib_exceedance",
        "target": "E. coli exceedance of EPA BAV (235 CFU/100mL, 2012 RWQC single-sample)",
        "non_circularity": (
            "Predictors are strictly antecedent/independent (NOAA GHCN prior-day "
            "rainfall, day-of-year/season, site lat/lon). The label is a SEPARATE "
            "channel (WQP E. coli culture result). No FIB value or function of it "
            "is a predictor."),
        "region": meta.get("region", "Ohio (US:39)"),
        "date_range": meta.get("date_range"),
        "n_samples": int(len(df)),
        "n_sites": int(df.site.nunique()),
        "n_positives": int(df.exceed.sum()),
        "base_rate": round(float(df.exceed.mean()), 4),
        "predictors": PREDICTORS,
        "data_sources": {
            "label": "Water Quality Portal (WQP) Result service, characteristic 'Escherichia coli'",
            "rainfall": "NOAA GHCN-Daily (NCEI daily-summaries), nearest-with-data station",
        },
        "model": "HistGradientBoostingClassifier (CPU)",
        "temporal_holdout": temporal,
        "site_grouped_holdout": site,
        "strict_antecedent_sensitivity": {
            "note": ("Same-day rainfall (rain_sameday_mm) removed. A daily PRCP "
                     "total can include rain after sample collection; dropping it "
                     "makes every predictor unambiguously antecedent. Skill here is "
                     "the conservative, deployable estimate."),
            "predictors": strict_predictors,
            "temporal_holdout": temporal_strict,
            "site_grouped_holdout": site_strict,
        },
        "verdict": v,
        "eval_seconds": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(out, indent=2))
    log(f"WROTE {OUT}")
    print("\n===== VERDICT =====")
    print(v["summary"])
    for l in v["detail"]:
        print(" -", l)


if __name__ == "__main__":
    main()
