#!/usr/bin/env python3
"""Paired bootstrap: is fusion (tax+func) significantly better than tax alone?

Refits tax_logreg and fuse_logreg on the seed=42 geographic holdout, then does a
PAIRED bootstrap over the SAME test samples (resample test indices; recompute
macro-F1 for both models on the SAME resample; look at the difference). This is
the correct test for comparing two models evaluated on one test set --- unlike the
independent per-model CIs, it accounts for the correlation between the two models'
predictions on shared samples. Also reports per-class F1 deltas.

Appends result to results/benchmarks/microbiomenet_functional.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "domaingen"))

from sentinel.data.splits import geographic_holdout_split  # noqa: E402
from run_domaingen_experiment import clr, load_samples, train_logreg, SOURCE_NAMES, SEED  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

OUT_JSON = PROJECT_ROOT / "results" / "benchmarks" / "microbiomenet_functional.json"
TAX_FEATS = PROJECT_ROOT / "data" / "processed" / "microbial" / "emp_taxonomic_features.npz"
FUNC_FEATS = PROJECT_ROOT / "data" / "processed" / "microbial" / "emp_functional_features.npz"


def macro_f1(yt, yp):
    present = sorted(set(int(x) for x in yt))
    return f1_score(yt, yp, labels=present, average="macro", zero_division=0)


def main():
    log = lambda m: print(m, flush=True)  # noqa: E731
    sids, y, lats, lons, X_otu = load_samples(log)

    tf = np.load(TAX_FEATS, allow_pickle=True)
    tfi = {str(s): i for i, s in enumerate(tf["sample_ids"])}
    o = np.array([tfi[s] for s in sids])
    X_tax = np.concatenate(
        [clr(tf["X_phylum"][o]), clr(tf["X_class"][o]), clr(tf["X_order"][o]),
         clr(tf["X_family"][o]), clr(tf["X_genus"][o])], axis=1).astype(np.float32)

    ff = np.load(FUNC_FEATS, allow_pickle=True)
    ffi = {str(s): i for i, s in enumerate(ff["sample_ids"])}
    fo = np.array([ffi[s] for s in sids])
    Xf_raw = ff["X_func"][fo].astype(np.float32)
    Xf = clr(Xf_raw[:, Xf_raw.sum(0) > 0])
    X_fuse = np.concatenate([X_tax, Xf], axis=1).astype(np.float32)

    split = geographic_holdout_split(lats, lons, y.tolist(), num_clusters=20,
                                     test_frac=0.20, val_frac=0.20, seed=SEED)
    tr, te = np.array(split["train"]), np.array(split["test"])

    yhat_tax = train_logreg(X_tax[tr], y[tr], X_tax[te], y[te])
    yhat_fuse = train_logreg(X_fuse[tr], y[tr], X_fuse[te], y[te])
    yte = y[te]

    f1_tax = macro_f1(yte, yhat_tax)
    f1_fuse = macro_f1(yte, yhat_fuse)
    log(f"tax={f1_tax:.4f} fuse={f1_fuse:.4f} delta={f1_fuse-f1_tax:+.4f}")

    # paired bootstrap over test samples
    rng = np.random.default_rng(SEED)
    n = len(yte)
    diffs = []
    present = sorted(set(int(x) for x in yte))
    for _ in range(2000):
        idx = rng.integers(0, n, n)
        yt = yte[idx]
        pres = [c for c in present if c in set(yt.tolist())]
        a = f1_score(yt, yhat_tax[idx], labels=pres, average="macro", zero_division=0)
        b = f1_score(yt, yhat_fuse[idx], labels=pres, average="macro", zero_division=0)
        diffs.append(b - a)
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p_fuse_not_better = float((diffs <= 0).mean())  # one-sided bootstrap p
    log(f"paired delta (fuse-tax): mean={diffs.mean():+.4f} 95%CI [{lo:+.4f},{hi:+.4f}] "
        f"P(fuse<=tax)={p_fuse_not_better:.4f}")

    # per-class F1
    pcf_tax = f1_score(yte, yhat_tax, labels=present, average=None, zero_division=0)
    pcf_fuse = f1_score(yte, yhat_fuse, labels=present, average=None, zero_division=0)
    per_class = {SOURCE_NAMES[c]: {"tax": float(a), "fuse": float(b), "delta": float(b - a)}
                 for c, a, b in zip(present, pcf_tax, pcf_fuse)}
    for k, v in per_class.items():
        log(f"  {k:22s} tax={v['tax']:.3f} fuse={v['fuse']:.3f} d={v['delta']:+.3f}")

    result = {
        "seed": SEED,
        "tax_logreg_macro_f1": float(f1_tax),
        "fuse_logreg_macro_f1": float(f1_fuse),
        "paired_delta_mean": float(diffs.mean()),
        "paired_delta_ci95": [float(lo), float(hi)],
        "paired_bootstrap_p_fuse_not_better": p_fuse_not_better,
        "paired_bootstrap_n": 2000,
        "significant_at_0.05": bool(p_fuse_not_better < 0.05 and lo > 0),
        "per_class_f1_tax_vs_fuse": per_class,
        "interpretation": (
            "Paired bootstrap holds the test set fixed and resamples it, so it "
            "measures whether fusion beats taxonomy on the SAME samples --- the correct "
            "test for two models on one test set (the independent per-model CIs "
            "overstate uncertainty because they ignore shared-sample correlation)."
        ),
    }

    with open(OUT_JSON) as f:
        out = json.load(f)
    out["partA_paired_significance_fuse_vs_tax"] = result
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    log(f"Appended paired-significance to {OUT_JSON}")


if __name__ == "__main__":
    main()
