#!/usr/bin/env python3
"""FINAL microcystin bimodal fusion test (larger N, NaN-aware).

In-situ pigment arm uses HistGBM native NaN handling + missing-indicators,
so a row needs a satellite tile AND >=1 in-situ pigment (chl-a OR phycocyanin).
Station-grouped 5-fold CV, AUROC + Hanley-McNeil CI. Real data only.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

df = pd.read_parquet("/home/bcheng/SENTINEL/data/processed/biology/noaa_habs/mc_fusion_dataset.parquet")
sf = [c for c in df.columns if c.startswith("sf")]


def hmci(auc, npos, nneg, z=1.96):
    q1 = auc / (2 - auc); q2 = 2 * auc * auc / (1 + auc)
    se = np.sqrt((auc*(1-auc)+(npos-1)*(q1-auc**2)+(nneg-1)*(q2-auc**2))/(npos*nneg))
    return max(0, auc-z*se), min(1, auc+z*se)


def cv_auc(X, y, g, seeds=(42, 7, 123)):
    aucs = []
    for s in seeds:
        oof = np.full(len(y), np.nan)
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(X, y, g):
            if y[tr].sum() < 2:
                continue
            c = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06,
                    max_depth=3, l2_regularization=1.0, random_state=0)
            c.fit(X[tr], y[tr])
            oof[te] = c.predict_proba(X[te])[:, 1]
        m = ~np.isnan(oof)
        aucs.append(roc_auc_score(y[m], oof[m]))
    return float(np.mean(aucs)), float(np.std(aucs)), int(m.sum())


def report(X, y, g, name):
    auc, sd, n = cv_auc(X, y, g)
    lo, hi = hmci(auc, int(y.sum()), int((y == 0).sum()))
    print(f"  {name:22s} AUROC={auc:.3f} +-{sd:.3f}  HM-CI({lo:.3f}-{hi:.3f})  n={n} pos={int(y.sum())}")
    return auc


# feature blocks
Xsat = df[sf].values.astype(float)
pig = df[["CHLA", "PC"]].copy()
pig["CHLA"] = np.log1p(pig["CHLA"].clip(lower=0))
pig["PC"] = np.log1p(pig["PC"].clip(lower=0))
# missing indicators help GBM use "PC was measured" signal without imputation bias
Xins = np.column_stack([pig["CHLA"].values, pig["PC"].values,
                        df["CHLA"].isna().astype(float), df["PC"].isna().astype(float)])
Xfus = np.hstack([Xsat, Xins])
g = df["site"].values

# rows usable: satellite always present; require >=1 pigment for fair fusion set
has_pig = df["CHLA"].notna().values | df["PC"].notna().values

for thr, nm in [(8.0, "microcystin > 8 ug/L (WHO recreational)"),
                (1.6, "microcystin > 1.6 ug/L (WHO drinking)"),
                (0.3, "microcystin detected (>0.3)")]:
    y_all = (df["mc"].values > thr).astype(int)
    sub = has_pig
    ys = y_all[sub]
    if ys.sum() < 10:
        continue
    print(f"\n=== {nm} | fair set: satellite tile AND >=1 in-situ pigment ===")
    print(f"    n={sub.sum()}  sites={df['site'][sub].nunique()}  pos={ys.sum()} ({ys.mean()*100:.1f}%)")
    a_sat = report(Xsat[sub], ys, g[sub], "SATELLITE (S2)")
    a_ins = report(Xins[sub], ys, g[sub], "INSITU pigment")
    a_fus = report(Xfus[sub], ys, g[sub], "FUSION")
    best_single = max(a_sat, a_ins)
    delta = a_fus - best_single
    verdict = "FUSION WINS" if delta > 0.01 else ("tie" if delta > -0.01 else "fusion worse")
    print(f"    -> fusion - best_single = {delta:+.3f}  [{verdict}]")
