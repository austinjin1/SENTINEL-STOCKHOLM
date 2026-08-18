#!/usr/bin/env python
"""Track B 10 — can microcystin fusion transfer ACROSS regions?

The 3-way fusion beats single within-region (station CV) but collapses to chance
(~0.49) under leave-one-HUC-region-out. This tests whether the failure is
(i) region-level OFFSETS (fixable by region-conditioning / within-site relative
features) or (ii) an INVERTED relationship across regions (not fixable by features).
Variants, all under leave-one-HUC-region-out CV:
  raw                : the 3-way features as-is (baseline ~0.49)
  +domain            : add lat/lon + month (let the model condition on region/season)
  within_site_rel    : each feature minus the site's own mean (removes site offsets;
                       deployable — a site accumulates its own baseline)
  fold_rank          : rank-transform each feature FIT ON TRAIN ROWS ONLY inside each
                       leave-one-region-out fold, then map test rows into the train
                       rank space (removes monotone scale shifts; leakage-free — the
                       held-out region never influences the transform)
If any recovers cross-region AUROC clearly above 0.5, region offsets are the cause
and it's partially fixable. If all stay ~0.5, the relationship genuinely does not
transfer — an honest ceiling.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "processed" / "biology" / "noaa_habs" / "mc_fusion_dataset.parquet"
CYANO = ROOT / "data" / "processed" / "biology" / "noaa_habs" / "wqp_hab_cyanotoxins.parquet"
RAW = ROOT / "data" / "raw" / "wqp_drivers"
OUT = ROOT / "results" / "trackB_fusion" / "mc_region_transfer.json"
SEED = 0


def parse(tag):
    f = RAW / f"{tag}.csv"
    if not f.exists() or f.stat().st_size < 100:
        return pd.DataFrame(columns=["site", "date", "val"])
    df = pd.read_csv(f, low_memory=False)
    if len(df) == 0:
        return pd.DataFrame(columns=["site", "date", "val"])
    return pd.DataFrame({"site": df["MonitoringLocationIdentifier"].astype(str),
                         "date": pd.to_datetime(df["ActivityStartDate"], errors="coerce"),
                         "val": pd.to_numeric(df["ResultMeasureValue"], errors="coerce")}
                        ).dropna(subset=["date", "val"]).sort_values("date")


def nearest(dv, site, date, tol=14):
    d = dv[dv["site"] == site]
    if len(d) == 0:
        return np.nan
    dd = (d["date"] - date).abs().dt.days
    i = dd.values.argmin()
    return float(d["val"].values[i]) if dd.values[i] <= tol else np.nan


def hgb():
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300,
                                          l2_regularization=1.0, class_weight="balanced", random_state=SEED)


def train_rank_transform(X_tr, X_te):
    """Column-wise rank->quantile transform FIT on TRAIN rows only, then map
    TEST rows into the train-derived rank space via interpolation. No held-out
    (test-region) value ever influences the train transform, so this stays
    leakage-free under leave-one-region-out CV (unlike a global rankdata call)."""
    Xtr = np.asarray(X_tr, dtype="float64")
    Xte = np.asarray(X_te, dtype="float64")
    Ztr = np.empty_like(Xtr)
    Zte = np.empty_like(Xte)
    n_tr = len(Xtr)
    for j in range(Xtr.shape[1]):
        col = Xtr[:, j]
        med = np.nanmedian(col) if np.isfinite(col).any() else 0.0
        col = np.nan_to_num(col, nan=med)
        # train quantile positions in [0,1]
        order = np.argsort(col, kind="mergesort")
        ranks = np.empty(n_tr, dtype="float64")
        ranks[order] = np.arange(1, n_tr + 1)
        Ztr[:, j] = ranks / n_tr
        # map test values onto the train-sorted support (train-derived quantiles)
        sorted_col = col[order]
        te_col = np.nan_to_num(Xte[:, j], nan=med)
        pos = np.searchsorted(sorted_col, te_col, side="right")
        Zte[:, j] = pos / n_tr
    return Ztr, Zte


def region_oof(X, y, region, fold_transform=None):
    regs = pd.Series(region).value_counts()
    big = regs[regs >= 30].index
    m = np.isin(region, big)
    p = np.full(len(y), np.nan)
    Xm, ym, rm = X[m], y[m], region[m]
    for tr, te in GroupKFold(n_splits=min(5, len(big))).split(Xm, ym, rm):
        idx = np.where(m)[0]
        Xtr, Xte = Xm[tr], Xm[te]
        if fold_transform is not None:
            Xtr, Xte = fold_transform(Xtr, Xte)
        p[idx[te]] = hgb().fit(Xtr, ym[tr]).predict_proba(Xte)[:, 1]
    return roc_auc_score(y[m], p[m]), int(m.sum()), int(len(big))


def within_site_center(X, site):
    Z = X.copy()
    for s in np.unique(site):
        i = site == s
        Z[i] = X[i] - np.nanmean(X[i], axis=0, keepdims=True)
    return Z


def main():
    df = pq.read_table(DATA).to_pandas()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).reset_index(drop=True)
    cy = pq.read_table(CYANO, columns=["Location_Identifier", "Location_HUCEightDigitCode",
                                       "Location_Latitude", "Location_Longitude"]).to_pandas()
    cy["site"] = cy["Location_Identifier"].astype(str)
    g = cy.groupby("site").first()
    df["region"] = df["site"].astype(str).map(g["Location_HUCEightDigitCode"].astype(str).str.zfill(8).str[:2]).fillna("NA")
    df["lat"] = df["site"].astype(str).map(pd.to_numeric(g["Location_Latitude"], errors="coerce"))
    df["lon"] = df["site"].astype(str).map(pd.to_numeric(g["Location_Longitude"], errors="coerce"))
    df["month"] = df["date"].dt.month

    drivers = {t: parse(t) for t in ["temp", "phosphorus", "nitrogen"]}
    for t, dv in drivers.items():
        df[f"drv_{t}"] = [nearest(dv, str(s), d) for s, d in zip(df["site"].astype(str), df["date"])]

    sf = [c for c in df.columns if c.startswith("sf")]
    base = df[sf + ["CHLA", "PC", "drv_temp", "drv_phosphorus", "drv_nitrogen"]].to_numpy("float64")
    site = df["site"].astype(str).to_numpy()
    region = df["region"].to_numpy()
    domain = df[["lat", "lon", "month"]].to_numpy("float64")
    mc = df["mc"].to_numpy("float64")

    results = {"n": int(len(df)), "labels": {}}
    for name, thr in [("mc_gt1.6", 1.6), ("mc_gt0.3", 0.3)]:
        y = (mc > thr).astype(int)
        # (matrix, per-fold transform). fold_rank ranks each column on TRAIN rows
        # only inside each CV fold, then maps TEST rows into that train space, so
        # no held-out region value enters the transform (was previously a global
        # rankdata over ALL rows -> transductive test->train leak).
        variants = {
            "raw": (base, None),
            "+domain(lat,lon,month)": (np.hstack([base, domain]), None),
            "within_site_relative": (np.hstack([within_site_center(base, site), domain]), None),
            "fold_rank": (base, train_rank_transform),
        }
        res = {"pos": int(y.sum())}
        for k, (X, ft) in variants.items():
            auc, n, nr = region_oof(X, y, region, fold_transform=ft)
            res[k] = round(float(auc), 3)
        res["n_cross_region"], res["n_regions"] = n, nr
        results["labels"][name] = res
        print(f"[{name}] cross-region AUROC (leave-one-HUC-out, {nr} regions): " +
              "  ".join(f"{k}={res[k]}" for k in variants), flush=True)

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nWROTE {OUT}")


if __name__ == "__main__":
    main()
