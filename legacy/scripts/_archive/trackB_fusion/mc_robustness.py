#!/usr/bin/env python3
"""Robustness + honest-fusion checks for microcystin bimodal test."""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

P = Path("/home/bcheng/SENTINEL/data/processed/biology/noaa_habs/mc_fusion_dataset.parquet")
df = pd.read_parquet(P)
sf = [c for c in df.columns if c.startswith("sf")]


def hmci(auc, npos, nneg, z=1.96):
    q1 = auc / (2 - auc); q2 = 2 * auc * auc / (1 + auc)
    se = np.sqrt((auc*(1-auc)+(npos-1)*(q1-auc**2)+(nneg-1)*(q2-auc**2))/(npos*nneg))
    return max(0, auc-z*se), min(1, auc+z*se)


def run(X, y, g, label):
    m = ~np.isnan(X).any(1)
    X, y, g = X[m], y[m], g[m]
    if y.sum() < 10 or (y == 0).sum() < 10:
        return f"{label}: too few (pos={y.sum()})"
    oof = np.full(len(y), np.nan)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=42).split(X, y, g):
        if y[tr].sum() < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        c = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06,
                                           max_depth=3, l2_regularization=1.0, random_state=0)
        c.fit(sc.transform(X[tr]), y[tr])
        oof[te] = c.predict_proba(sc.transform(X[te]))[:, 1]
    mm = ~np.isnan(oof)
    auc = roc_auc_score(y[mm], oof[mm])
    lo, hi = hmci(auc, int(y[mm].sum()), int((y[mm] == 0).sum()))
    return f"{label}: AUROC={auc:.3f} ({lo:.3f}-{hi:.3f}) n={mm.sum()} pos={int(y[mm].sum())}"


g = df["site"].values
Xsat = df[sf].values.astype(float)
ins = np.column_stack([np.log1p(df["CHLA"].clip(lower=0)), np.log1p(df["PC"].clip(lower=0))])
chla_only = np.log1p(df["CHLA"].clip(lower=0)).values.reshape(-1, 1)
Xfus = np.hstack([Xsat, ins])
both = ~np.isnan(ins).any(1)

print("### Label variants (Design B: both modalities present) ###")
for thr, nm in [(8.0, ">8 WHO-recreational"), (1.6, ">1.6 WHO-drinking"), (0.3, "detect>0.3")]:
    y = (df["mc"] > thr).astype(int).values
    print(f"\n-- microcystin {nm} (pos rate {y[both].mean():.2f}) --")
    print(" ", run(Xsat[both], y[both], g[both], "SAT"))
    print(" ", run(chla_only[both], y[both], g[both], "CHLA-only"))
    print(" ", run(ins[both], y[both], g[both], "INSITU(chla+pc)"))
    print(" ", run(Xfus[both], y[both], g[both], "FUSION"))

print("\n### Does satellite add ANYTHING over pigment? residual test (WHO>8) ###")
y = (df["mc"] > 8).astype(int).values
# pigment+sat vs pigment: already above. Check sat-only among HIGH-pigment subset
hi_pig = both & (df["CHLA"].fillna(0).values > df["CHLA"][both].median())
print("  In high-chla subset, can satellite separate toxin?")
print(" ", run(Xsat[hi_pig], y[hi_pig], g[hi_pig], "SAT|highchla"))

print("\n### Clear-sky tiles only (cloud<15) ###")
clear = both & (df["cloud"].fillna(99).values < 15)
print(f"  n clear+both = {clear.sum()}")
print(" ", run(Xsat[clear], y[clear], g[clear], "SAT|clear"))
print(" ", run(ins[clear], y[clear], g[clear], "INSITU|clear"))
print(" ", run(Xfus[clear], y[clear], g[clear], "FUSION|clear"))
