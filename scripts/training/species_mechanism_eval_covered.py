#!/usr/bin/env python3
"""Fair same-subset re-evaluation of the species env-mechanism test.

The full eval (species_mechanism_eval.py) splits over all 5,462 sites, but real
StreamCat env is resolved for only ~993 (COMID resolution stopped at 1,000). So
S1 (StreamCat-only) is tested mostly on MISSING features -> its near-chance score
is a coverage artifact, not evidence that environment lacks signal.

This restricts train AND test to sites that actually HAVE StreamCat data, keeps
the original leakage-free split, and asks the real question: when env data is
present, does real watershed environment (S1) beat geography (S3) / the proxy
floor (S0), and does adding env to geography (S4) help over geo-only (S3)?
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from species_mechanism_eval import (
    load, build_features, eval_set, macro, bootstrap_auroc, paired_delta_ci,
    FEATURE_SETS, SPECIES_NAMES, MUSSEL_IDX,
)

tab, site_sc, metric_cols = load()
sc_feat = ["sc_" + c for c in metric_cols]
df = build_features(tab, site_sc, metric_cols)
occ = (tab[[str(g) for g in range(6)]].values > 0).astype(np.float32)

covered = site_sc[metric_cols[0]].notna().values
tr = pd.Series(tab["split"].isin(["train", "val"]).values & covered, index=tab.index)
te = pd.Series((tab["split"].values == "test") & covered, index=tab.index)
print(f"COVERED sites: {int(covered.sum())} | covered train+val={int(tr.sum())} "
      f"covered test={int(te.sum())}", flush=True)

summary, store = {}, {}
for sname, sel in FEATURE_SETS.items():
    feats = [f for f in sel(sc_feat) if f in df.columns]
    res = eval_set(df, occ, tr, te, feats)
    m = macro(res, incl_mayfly=False)
    mus = res[SPECIES_NAMES[MUSSEL_IDX]]
    summary[sname] = {"n_features": len(feats), "macro_auroc_excl_mayfly": m[0],
                      "mussel_auroc": mus.get("auroc_gbm"),
                      "mussel_n_test": mus.get("n_test"), "mussel_pos": mus.get("n_present_test")}
    store[sname] = res
    print(f"  {sname:20s} nf={len(feats):3d} macro={m[0]:.3f} "
          f"mussel={mus.get('auroc_gbm')}", flush=True)

mn = SPECIES_NAMES[MUSSEL_IDX]
y = store["S0_proxy_env"][mn]["_y"]
paired = {
    "S1_streamcat_minus_S0_proxy":  paired_delta_ci(y, store["S1_streamcat"][mn]["_p_gbm"], store["S0_proxy_env"][mn]["_p_gbm"]),
    "S1_streamcat_minus_S3_geo":    paired_delta_ci(y, store["S1_streamcat"][mn]["_p_gbm"], store["S3_geo_only"][mn]["_p_gbm"]),
    "S4_streamcat_geo_minus_S3_geo": paired_delta_ci(y, store["S4_streamcat_geo"][mn]["_p_gbm"], store["S3_geo_only"][mn]["_p_gbm"]),
}
print("\n=== paired mussel deltas (covered subset) ===")
for k, v in paired.items():
    print(f"  {k}: delta={v['delta_median']:+.3f} CI[{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}] P(>0)={v['p_gt_0_frac']:.2f}")

out = {"n_covered": int(covered.sum()), "n_covered_train": int(tr.sum()),
       "n_covered_test": int(te.sum()), "n_metric_cols": len(metric_cols),
       "note": "Same-subset fair comparison: train+test restricted to sites WITH StreamCat coverage.",
       "summary": summary, "mussel_paired_deltas": paired}
OUT = Path(__file__).resolve().parents[2] / "results/benchmarks/species_mechanism_covered.json"
OUT.write_text(json.dumps(out, indent=2))
print(f"\nWROTE {OUT}")
