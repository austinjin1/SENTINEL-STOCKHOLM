#!/usr/bin/env python3
"""Verify the microcystin fusion win: paired bootstrap CI on fusion-minus-best,
leakage checks (drop missing-indicators, drop PC entirely), and site-holdout."""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(0)
df = pd.read_parquet("/home/bcheng/SENTINEL/data/processed/biology/noaa_habs/mc_fusion_dataset.parquet")
sf = [c for c in df.columns if c.startswith("sf")]
g = df["site"].values
has_pig = df["CHLA"].notna().values | df["PC"].notna().values

Xsat = df[sf].values.astype(float)
pigC = np.log1p(df["CHLA"].clip(lower=0).values)
pigP = np.log1p(df["PC"].clip(lower=0).values)
miss = np.column_stack([df["CHLA"].isna().astype(float), df["PC"].isna().astype(float)])


def oof_pred(X, y, grp, seed=42):
    oof = np.full(len(y), np.nan)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, grp):
        if y[tr].sum() < 2:
            continue
        c = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06,
                max_depth=3, l2_regularization=1.0, random_state=0)
        c.fit(X[tr], y[tr]); oof[te] = c.predict_proba(X[te])[:, 1]
    return oof


def paired_boot(y, p_fus, p_best, grp, B=2000):
    """Cluster (site) bootstrap of AUROC difference fusion-best."""
    sites = np.unique(grp)
    diffs = []
    base = roc_auc_score(y, p_fus) - roc_auc_score(y, p_best)
    for _ in range(B):
        samp = rng.choice(sites, len(sites), replace=True)
        idx = np.concatenate([np.where(grp == s)[0] for s in samp])
        yy = y[idx]
        if yy.sum() < 2 or (yy == 0).sum() < 2:
            continue
        d = roc_auc_score(yy, p_fus[idx]) - roc_auc_score(yy, p_best[idx])
        diffs.append(d)
    diffs = np.array(diffs)
    return base, np.percentile(diffs, 2.5), np.percentile(diffs, 97.5), (diffs > 0).mean()


sub = has_pig
for thr, nm in [(8.0, "WHO>8"), (1.6, "WHO>1.6"), (0.3, "detect")]:
    y = (df["mc"].values > thr).astype(int)[sub]
    gg = g[sub]
    Xs = Xsat[sub]
    # in-situ WITH missing indicators
    Xi = np.column_stack([pigC[sub], pigP[sub], miss[sub]])
    Xf = np.hstack([Xs, Xi])
    # variant: NO missing indicators (leakage guard)
    Xi_nm = np.column_stack([pigC[sub], pigP[sub]])
    Xf_nm = np.hstack([Xs, Xi_nm])
    # variant: chl-a only (drop PC entirely)
    Xi_c = pigC[sub].reshape(-1, 1)
    Xf_c = np.hstack([Xs, Xi_c])

    ps = oof_pred(Xs, y, gg)
    pi = oof_pred(Xi, y, gg)
    pf = oof_pred(Xf, y, gg)
    m = ~np.isnan(pf) & ~np.isnan(ps) & ~np.isnan(pi)
    a_s, a_i, a_f = (roc_auc_score(y[m], ps[m]), roc_auc_score(y[m], pi[m]),
                     roc_auc_score(y[m], pf[m]))
    best = "INSITU" if a_i >= a_s else "SAT"
    pbest = pi if a_i >= a_s else ps
    base, lo, hi, pgt = paired_boot(y[m], pf[m], pbest[m], gg[m])
    print(f"\n[{nm}] n={m.sum()} pos={y[m].sum()} | SAT={a_s:.3f} INSITU={a_i:.3f} FUSION={a_f:.3f} (best={best})")
    print(f"   fusion-best paired site-bootstrap Δ={base:+.3f}  95%CI[{lo:+.3f},{hi:+.3f}]  P(Δ>0)={pgt:.3f}")

    # leakage guards
    pf_nm = oof_pred(Xf_nm, y, gg)
    pf_c = oof_pred(Xf_c, y, gg)
    mm = ~np.isnan(pf_nm)
    print(f"   FUSION no-missing-indicators AUROC={roc_auc_score(y[mm],pf_nm[mm]):.3f}")
    mc = ~np.isnan(pf_c)
    a_i_c = roc_auc_score(y[mc], oof_pred(Xi_c, y, gg)[mc])
    print(f"   FUSION chl-a-only (no PC)   AUROC={roc_auc_score(y[mc],pf_c[mc]):.3f}  vs chl-a-only-insitu={a_i_c:.3f} vs sat={a_s:.3f}")
