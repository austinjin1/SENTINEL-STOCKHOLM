#!/usr/bin/env python
"""Track B 9 — robustness of the 3-modality microcystin fusion (drivers add over 2-way).

The station-grouped CV showed sat+pigment+drivers (0.823) > sat+pigment (0.776) at
thr>1.6, driver-gain +0.047 (bootstrap CI excludes 0). Before headlining, stress-test:
  (a) TEMPORAL holdout — train earlier dates, test later (out-of-time);
  (b) LEAVE-ONE-REGION-OUT — hold out whole HUC-2 regions (cross-region);
  (c) NEGATIVE CONTROL — shuffle the driver features; the 3-way must collapse to the
      2-way score (proves the driver gain is real signal, not extra-capacity overfit).
Reuses the cached WQP driver CSVs from step 08 (no re-download).
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
OUT = ROOT / "results" / "trackB_fusion" / "mc_driver_robustness.json"
SEED = 0


def hgb():
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300,
                                          l2_regularization=1.0, class_weight="balanced", random_state=SEED)


def parse(tag):
    f = RAW / f"{tag}.csv"
    if not f.exists():
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


def auc_split(Xtr, ytr, Xte, yte):
    if ytr.min() == ytr.max() or yte.min() == yte.max():
        return None
    return float(roc_auc_score(yte, hgb().fit(Xtr, ytr).predict_proba(Xte)[:, 1]))


def oof(X, y, g, ns):
    p = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=ns).split(X, y, g):
        p[te] = hgb().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return p


def main():
    df = pq.read_table(DATA).to_pandas()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).reset_index(drop=True)
    # region via HUC-2 from the cyanotoxin table
    cy = pq.read_table(CYANO, columns=["Location_Identifier", "Location_HUCEightDigitCode"]).to_pandas()
    cy["site"] = cy["Location_Identifier"].astype(str)
    huc2 = (cy.assign(h=cy["Location_HUCEightDigitCode"].astype(str).str.zfill(8).str[:2])
              .groupby("site")["h"].first())
    df["region"] = df["site"].astype(str).map(huc2).fillna("NA")

    drivers = {t: parse(t) for t in ["temp", "phosphorus", "nitrogen"]}
    for t, dv in drivers.items():
        df[f"drv_{t}"] = [nearest(dv, str(s), d) for s, d in zip(df["site"].astype(str), df["date"])]

    sf = [c for c in df.columns if c.startswith("sf")]
    sat, pig = df[sf].to_numpy("float64"), df[["CHLA", "PC"]].to_numpy("float64")
    drv = df[["drv_temp", "drv_phosphorus", "drv_nitrogen"]].to_numpy("float64")
    two, three = np.hstack([sat, pig]), np.hstack([sat, pig, drv])
    site = df["site"].astype(str).to_numpy()
    region = df["region"].to_numpy()
    mc = df["mc"].to_numpy("float64")
    rng = np.random.default_rng(SEED)

    results = {"n": int(len(df)), "labels": {}}
    for name, thr in [("mc_gt1.6", 1.6), ("mc_gt0.3", 0.3)]:
        y = (mc > thr).astype(int)
        r = {"pos": int(y.sum())}

        # (a) temporal holdout: train first 60% by date, test last 40%
        order = np.argsort(df["date"].values)
        cut = int(0.6 * len(order)); tr, te = order[:cut], order[cut:]
        r["temporal"] = {"two_way": auc_split(two[tr], y[tr], two[te], y[te]),
                         "three_way": auc_split(three[tr], y[tr], three[te], y[te]),
                         "best_single": max(auc_split(sat[tr], y[tr], sat[te], y[te]) or 0,
                                            auc_split(pig[tr], y[tr], pig[te], y[te]) or 0)}
        # (b) leave-one-region-out (pooled OOF over region folds)
        regs = pd.Series(region).value_counts()
        big = regs[regs >= 30].index
        m = np.isin(region, big)
        if m.sum() > 100 and len(big) >= 3:
            oof2 = oof(two[m], y[m], region[m], min(5, len(big)))
            oof3 = oof(three[m], y[m], region[m], min(5, len(big)))
            r["cross_region"] = {"two_way": round(float(roc_auc_score(y[m], oof2)), 3),
                                 "three_way": round(float(roc_auc_score(y[m], oof3)), 3),
                                 "n_regions": int(len(big)), "n": int(m.sum())}
        # (c) negative control: shuffle drivers -> 3-way must collapse to 2-way
        drv_shuf = drv[rng.permutation(len(drv))]
        three_shuf = np.hstack([sat, pig, drv_shuf])
        ns = min(5, df["site"].nunique())
        r["neg_control_3way_shuffled_drivers"] = round(float(roc_auc_score(y, oof(three_shuf, y, site, ns))), 3)
        r["two_way_ref"] = round(float(roc_auc_score(y, oof(two, y, site, ns))), 3)
        r["three_way_ref"] = round(float(roc_auc_score(y, oof(three, y, site, ns))), 3)

        results["labels"][name] = r
        t = r["temporal"]; cr = r.get("cross_region", {})
        print(f"\n[{name}] pos={r['pos']}", flush=True)
        print(f"  TEMPORAL  2way={t['two_way']} 3way={t['three_way']} best_single={round(t['best_single'],3)}", flush=True)
        if cr: print(f"  CROSS-REGION ({cr['n_regions']} regions) 2way={cr['two_way']} 3way={cr['three_way']}", flush=True)
        print(f"  NEG-CONTROL 3way(shuffled drv)={r['neg_control_3way_shuffled_drivers']} vs 2way_ref={r['two_way_ref']} vs 3way_ref={r['three_way_ref']}", flush=True)

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nWROTE {OUT}")


if __name__ == "__main__":
    main()
