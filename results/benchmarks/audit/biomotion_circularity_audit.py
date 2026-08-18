#!/usr/bin/env python3
"""Focused audit of BioMotion GBM claim: circularity, straw-man, split integrity.
Reuses the EXACT build_table logic from scripts/experiments/biomotion_gbm.py.
"""
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.dummy import DummyClassifier

ROOT = Path("/home/bcheng/SENTINEL")
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))
import biomotion_gbm as B  # reuse exact loaders/label/feature logic

t0 = time.time()
results, tests = B.load_results_and_tests()
df = B.build_table(results, tests)
X, y, cols = B.feature_matrix(df)
groups = df["test_cas"].to_numpy()
print(f"n={len(df)} anomaly_rate={y.mean():.4f} base_rate_majority={max(y.mean(),1-y.mean()):.4f}")

# ---- 1. CIRCULARITY: how well does effect-group / channel ALONE predict label? ----
# The label is derived from `endpoint`. Are `effect` group one-hots a proxy for endpoint?
def auroc_grouped(Xsub, seed=0):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    tr, te = next(gss.split(Xsub, y, groups))
    clf = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05,
                                     max_depth=3, subsample=0.8, random_state=42)
    clf.fit(Xsub[tr], y[tr]); p = clf.predict_proba(Xsub[te])[:,1]
    return roc_auc_score(y[te], p)

col_idx = {c:i for i,c in enumerate(cols)}
effgrp_cols = [c for c in cols if c.startswith("effgrp_")]
chan_cols   = [c for c in cols if c.startswith("chan_")]
design_cols = ["logconc_median","logconc_max","logconc_min","logconc_mean","logconc_std",
               "logconc_range","n_levels","n_rows","exposure_hours_log","exposure_missing",
               "obs_duration_log"]
def sub(colnames):
    idx = [col_idx[c] for c in colnames if c in col_idx]
    return X[:, idx]

# per-effect-group anomaly rate (raw circularity signal)
print("\n=== effect-group presence -> anomaly rate (circularity signal) ===")
for eg in effgrp_cols:
    if eg == "effgrp_n": continue
    present = X[:, col_idx[eg]] == 1.0
    if present.sum() < 50: continue
    print(f"  {eg:14s} present n={present.sum():6d} anomaly_rate={y[present].mean():.3f} | absent anomaly_rate={y[~present].mean():.3f}")

print("\n=== ablation AUROC (chemical-grouped, seed avg of 5) ===")
def avg5(colnames, tag):
    a = np.mean([auroc_grouped(sub(colnames), s) for s in range(5)])
    print(f"  {tag:38s} auroc={a:.4f}")
    return a
a_full   = avg5(cols, "FULL (all 40 features)")
a_effgrp = avg5(effgrp_cols, "effect-group one-hots ONLY")
a_chan   = avg5(chan_cols, "measurement-channel one-hots ONLY")
a_design = avg5(design_cols, "design-only (conc+duration, NO biology)")
a_obs    = avg5(["obs_duration_log"], "obs_duration_log ALONE")
a_bio    = avg5(effgrp_cols+chan_cols+["species_number"], "biology-only (effgrp+chan+species)")

# ---- 2. STRAW-MAN: is the diffusion 0.807 baseline a weak straw-man? ----
# Compare GBM grouped vs a regularized logistic regression on same features & split.
print("\n=== STRAW-MAN check: simple baselines on SAME grouped split ===")
gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
tr, te = next(gss.split(X, y, groups))
from sklearn.preprocessing import StandardScaler
sc = StandardScaler().fit(X[tr])
lr = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(X[tr]), y[tr])
print(f"  LogisticRegression grouped auroc = {roc_auc_score(y[te], lr.predict_proba(sc.transform(X[te]))[:,1]):.4f}")
dummy_auroc = 0.5
print(f"  Dummy/majority auroc = {dummy_auroc:.4f} (base rate = {max(y.mean(),1-y.mean()):.4f})")

# ---- 3. SPLIT INTEGRITY ----
overlap = set(groups[tr]) & set(groups[te])
print(f"\n=== grouped split integrity: chemical overlap = {len(overlap)} (must be 0) ===")

out = {
  "n_tests": int(len(df)), "anomaly_rate": float(y.mean()),
  "grouped_auroc_full": float(a_full),
  "grouped_auroc_effgrp_only": float(a_effgrp),
  "grouped_auroc_chan_only": float(a_chan),
  "grouped_auroc_design_only_no_biology": float(a_design),
  "grouped_auroc_obs_duration_alone": float(a_obs),
  "grouped_auroc_biology_only": float(a_bio),
  "logreg_grouped_auroc": float(roc_auc_score(y[te], lr.predict_proba(sc.transform(X[te]))[:,1])),
  "chemical_overlap": len(overlap),
  "elapsed_sec": round(time.time()-t0,1),
}
Path("/home/bcheng/SENTINEL/results/benchmarks/audit/biomotion_circularity_audit.json").write_text(json.dumps(out, indent=2))
print("\nSaved audit json.")
print(json.dumps(out, indent=2))
