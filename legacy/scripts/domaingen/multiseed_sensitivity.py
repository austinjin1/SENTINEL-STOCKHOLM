#!/usr/bin/env python3
"""Multi-seed geographic-holdout sensitivity for the domaingen headline claim.

Re-runs the SAME pipeline across several k-means split seeds to confirm the
taxonomic-feature advantage over raw OTUs is not an artifact of one region
assignment. Reports mean +/- std macro-F1 for:
  - OTU GBM  (same-classifier control on raw top-5000 OTU CLR)
  - Taxonomic GBM (multi-rank CLR)
  - Taxonomic LogReg (multi-rank CLR)
across seeds {13,42,101,202,777}. Real data, identical leakage checks.
"""
from __future__ import annotations
import json, sys, time
from collections import Counter
from pathlib import Path
import numpy as np

PR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PR))
from sentinel.data.splits import geographic_holdout_split  # noqa: E402
import importlib.util
spec = importlib.util.spec_from_file_location(
    "dg", str(PR / "scripts" / "domaingen" / "run_domaingen_experiment.py"))
dg = importlib.util.module_from_spec(spec); spec.loader.exec_module(dg)

OUT = PR / "results" / "benchmarks" / "microbiomenet_domaingen_multiseed.json"


def run_seed(seed, sids, y, lats, lons, X_otu_clr, X_tax):
    split = geographic_holdout_split(lats, lons, y.tolist(), num_clusters=20,
                                     test_frac=0.20, val_frac=0.20, seed=seed)
    tr, va, te = (np.array(split[k]) for k in ("train", "val", "test"))
    # region leakage guard
    def rs(idx): return set((round(float(lats[i]),4), round(float(lons[i]),4)) for i in idx)
    leak = len(rs(tr) & rs(te)) + len(rs(va) & rs(te)) + len(rs(tr) & rs(va))
    out = {"seed": seed, "n_test": int(len(te)), "region_leak": int(leak),
           "n_test_classes": len(set(y[te].tolist()))}
    yhat, _ = dg.train_gbm(X_otu_clr[tr], y[tr], X_otu_clr[va], y[va], X_otu_clr[te], y[te])
    out["otu_gbm_macro_f1"] = dg.macro_f1_present(y[te], yhat)[0]
    yhat, _ = dg.train_gbm(X_tax[tr], y[tr], X_tax[va], y[va], X_tax[te], y[te])
    out["tax_gbm_macro_f1"] = dg.macro_f1_present(y[te], yhat)[0]
    yhat = dg.train_logreg(X_tax[tr], y[tr], X_tax[te], y[te])
    out["tax_logreg_macro_f1"] = dg.macro_f1_present(y[te], yhat)[0]
    return out


def main():
    t0 = time.time()
    log = lambda m: print(m, flush=True)
    sids, y, lats, lons, X_otu = dg.load_samples(log)
    tf = np.load(dg.TAX_FEATS, allow_pickle=True)
    idx = {str(s): i for i, s in enumerate(tf["sample_ids"])}
    o = np.array([idx[s] for s in sids])
    X_tax = np.concatenate([dg.clr(tf["X_phylum"][o]), dg.clr(tf["X_class"][o]),
                            dg.clr(tf["X_order"][o]), dg.clr(tf["X_family"][o]),
                            dg.clr(tf["X_genus"][o])], axis=1).astype(np.float32)
    X_otu_clr = dg.clr(X_otu)
    rows = []
    for seed in [13, 42, 101, 202, 777]:
        r = run_seed(seed, sids, y, lats, lons, X_otu_clr, X_tax)
        log(f"seed={seed}: otu_gbm={r['otu_gbm_macro_f1']:.4f} "
            f"tax_gbm={r['tax_gbm_macro_f1']:.4f} tax_logreg={r['tax_logreg_macro_f1']:.4f} "
            f"(n_test={r['n_test']}, leak={r['region_leak']})")
        rows.append(r)
    def summ(k):
        v = np.array([r[k] for r in rows])
        return {"mean": float(v.mean()), "std": float(v.std()), "min": float(v.min()), "max": float(v.max())}
    out = {"baseline_nn_raw_otu_macro_f1": 0.404814,
           "per_seed": rows,
           "summary": {k: summ(k) for k in ["otu_gbm_macro_f1", "tax_gbm_macro_f1", "tax_logreg_macro_f1"]},
           "elapsed_seconds": round(time.time()-t0,1)}
    OUT.write_text(json.dumps(out, indent=2))
    log(f"Wrote {OUT}")
    for k in ["otu_gbm_macro_f1","tax_gbm_macro_f1","tax_logreg_macro_f1"]:
        s=out["summary"][k]; log(f"{k}: {s['mean']:.4f} +/- {s['std']:.4f}")


if __name__ == "__main__":
    main()
