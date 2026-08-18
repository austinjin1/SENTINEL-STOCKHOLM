#!/usr/bin/env python
"""Track B 9 — WHICH indices/bands drove the improved satellite arm?

Two analyses on the improved satellite feature set (from 08):
  (A) Single-family AUROC: train satellite arm on ONE index family at a time
      (all its spatial stats) to rank which cyano indices carry the signal.
  (B) Drop-one-family: remove each family from the full set, measure AUROC drop.
Label focus: mc>1.6 (WHO drinking) and mc>0.3 (detection) — the shipped thresholds.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

import importlib.util
spec = importlib.util.spec_from_file_location(
    "arm", Path(__file__).parent / "08_microcystin_satellite_arm.py")
arm = importlib.util.module_from_spec(spec); spec.loader.exec_module(arm)

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data/processed/biology/noaa_habs/mc_fusion_dataset.parquet"
OUT = ROOT / "results/trackB_fusion/mc_sat_ablation.json"
SEED = 0
FAMILIES = ["NDCI", "REratio", "PCpotes", "MCIre", "FAI", "NDVI", "NDWI", "GR", "GB", "B5B3"]


def oof(X, y, g, ns):
    p = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=ns).split(X, y, g):
        p[te] = arm.hgb().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return p


def main():
    df = pq.read_table(DATA).to_pandas()
    sites = df["site"].astype(str).to_numpy()
    mc = df["mc"].to_numpy("float64")
    ns = min(5, df["site"].nunique())
    X, names = arm.load_improved_sat(df)
    names = np.array(names)
    # band-only columns (the *_ctr) form their own group
    fam_cols = {f: np.array([i for i, n in enumerate(names) if n.startswith(f + "_")]) for f in FAMILIES}
    band_cols = np.array([i for i, n in enumerate(names) if n.endswith("_ctr")])
    frac_cols = np.array([i for i, n in enumerate(names) if n.startswith("frac_")])

    out = {"labels": {}}
    for thr, tag in [(1.6, "mc_gt1.6"), (0.3, "mc_gt0.3")]:
        y = (mc > thr).astype(int)
        full = float(roc_auc_score(y, oof(X, y, sites, ns)))
        single, drop = {}, {}
        # bands-only baseline
        single["BANDS_only"] = round(float(roc_auc_score(y, oof(X[:, band_cols], y, sites, ns))), 3)
        for f in FAMILIES:
            cols = np.concatenate([fam_cols[f], band_cols])  # family + raw bands (fair floor)
            single[f + "+bands"] = round(float(roc_auc_score(y, oof(X[:, cols], y, sites, ns))), 3)
            keep = np.array([i for i in range(X.shape[1]) if i not in set(fam_cols[f])])
            drop["drop_" + f] = round(float(roc_auc_score(y, oof(X[:, keep], y, sites, ns))), 3)
        out["labels"][tag] = {"full_auroc": round(full, 3),
                              "single_family_plus_bands": single,
                              "drop_one_family": drop}
        print(f"[{tag}] full={full:.3f}")
        print("  single family(+bands):", dict(sorted(single.items(), key=lambda kv: -kv[1])))
        print("  drop-one (lower=more important):", dict(sorted(drop.items(), key=lambda kv: kv[1])))
    OUT.write_text(json.dumps(out, indent=2))
    print("WROTE", OUT)


if __name__ == "__main__":
    main()
