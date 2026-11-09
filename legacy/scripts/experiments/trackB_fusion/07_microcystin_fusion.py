#!/usr/bin/env python
"""Track B 7 — microcystin (cyanotoxin) fusion: the genuine bimodal test.

Independent re-verification + persistence of the deep-research finding that
multimodal fusion HONESTLY beats each single modality for microcystin — because
the toxin is optically invisible, so satellite spectral (bloom optics) and
in-situ pigment (CHLA/PC field probes) carry non-redundant information.

Rigor (the ship-gate): fusion AUROC must exceed the BEST single modality with a
SITE-CLUSTERED bootstrap 95% CI on the paired delta that excludes zero. Plus a
permutation control (shuffle satellite features -> fusion must collapse toward
in-situ-alone, proving the satellite arm carries real independent signal).

Honest scope: this is satellite + in-situ-*pigment* fusion (not physicochemical
sonde fusion — temp/pH/DO/nutrients were not co-located at these sites).
Station-grouped CV, non-circular label (microcystin threshold, independent of the
satellite/pigment inputs).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "processed" / "biology" / "noaa_habs" / "mc_fusion_dataset.parquet"
OUT = ROOT / "results" / "trackB_fusion" / "mc_fusion.json"
SEED = 0
NBOOT = 2000
WHO_THRESHOLDS = {"mc_gt8_WHO_recreational": 8.0,
                  "mc_gt1.6_WHO_drinking": 1.6,
                  "mc_gt0.3_detection": 0.3}


def hgb():
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300,
                                          l2_regularization=1.0, class_weight="balanced",
                                          random_state=SEED)


def oof(X, y, g, ns):
    p = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=ns).split(X, y, g):
        p[te] = hgb().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return p


def hm_ci(a, y):
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    se = np.sqrt((a * (1 - a) + (n1 - 1) * (q1 - a**2) + (n0 - 1) * (q2 - a**2)) / (n1 * n0))
    return [round(float(max(0, a - 1.96 * se)), 3), round(float(min(1, a + 1.96 * se)), 3)]


def site_bootstrap_delta(y, p_fusion, p_best, sites, seed=SEED, B=NBOOT):
    """Paired site-clustered bootstrap on AUROC(fusion) - AUROC(best single)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(sites)
    by_site = {s: np.where(sites == s)[0] for s in uniq}
    deltas = []
    for _ in range(B):
        chosen = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([by_site[s] for s in chosen])
        yi = y[idx]
        if yi.min() == yi.max():
            continue
        deltas.append(roc_auc_score(yi, p_fusion[idx]) - roc_auc_score(yi, p_best[idx]))
    deltas = np.array(deltas)
    return {"delta_median": round(float(np.median(deltas)), 3),
            "ci95": [round(float(np.percentile(deltas, 2.5)), 3),
                     round(float(np.percentile(deltas, 97.5)), 3)],
            "P_delta_gt0": round(float((deltas > 0).mean()), 3)}


def main():
    df = pq.read_table(DATA).to_pandas()
    sf = [c for c in df.columns if c.startswith("sf")]
    sat = df[sf].to_numpy("float64")
    # Clip in-situ pigment inputs to a physical range before fusion: raw CHLA/PC
    # contain non-physical sensor error codes (CHLA up to ~1e6, PC up to ~2e6, some
    # non-positive PC). Clipping to [0,5000] is a data-hygiene step (NaNs preserved
    # by np.clip and handled natively by HGB). Small clean gain at the WHO-drinking
    # threshold: mc>1.6 AUROC 0.776->0.781 (paired site-bootstrap +0.005,
    # CI[+0.002,+0.008], P>0=1.0); null at mc>0.3 and mc>8.
    insitu = np.clip(df[["CHLA", "PC"]].to_numpy("float64"), 0, 5000)
    fusion = np.hstack([sat, insitu])
    sites = df["site"].astype(str).to_numpy()
    mc = df["mc"].to_numpy("float64")
    ns = min(5, df["site"].nunique())
    rng = np.random.default_rng(SEED)

    results = {"dataset": str(DATA.name), "n": int(len(df)), "n_sites": int(df["site"].nunique()),
               "note": "satellite spectral (sf*) + in-situ PIGMENT (CHLA,PC); NOT physicochemical sonde. "
                       "Label = microcystin threshold (non-circular: independent of both inputs).",
               "labels": {}}
    for name, thr in WHO_THRESHOLDS.items():
        y = (mc > thr).astype(int)
        if y.sum() < 15 or (len(y) - y.sum()) < 15:
            results["labels"][name] = {"skipped": "too few pos/neg", "pos": int(y.sum())}
            continue
        p_sat = oof(sat, y, sites, ns)
        p_ins = oof(insitu, y, sites, ns)
        p_fus = oof(fusion, y, sites, ns)
        a_sat, a_ins, a_fus = (float(roc_auc_score(y, p)) for p in (p_sat, p_ins, p_fus))
        best_p, best_name, best_a = ((p_ins, "insitu", a_ins) if a_ins >= a_sat
                                     else (p_sat, "satellite", a_sat))
        # permutation control: shuffle satellite features, refit fusion
        sat_perm = sat[rng.permutation(len(sat))]
        p_fus_perm = oof(np.hstack([sat_perm, insitu]), y, sites, ns)
        boot = site_bootstrap_delta(y, p_fus, best_p, sites)
        res = {"pos": int(y.sum()), "n_sites_pos": int(df.loc[y == 1, "site"].nunique()),
               "satellite_auroc": round(a_sat, 3), "insitu_auroc": round(a_ins, 3),
               "fusion_auroc": round(a_fus, 3), "fusion_ci95": hm_ci(a_fus, y),
               "best_single": best_name, "fusion_minus_best": round(a_fus - best_a, 3),
               "site_bootstrap_delta": boot,
               "fusion_helps": bool(boot["ci95"][0] > 0),
               "permutation_control_fusion_auroc": round(float(roc_auc_score(y, p_fus_perm)), 3)}
        results["labels"][name] = res
        print(f"[{name}] pos={res['pos']} sat={a_sat:.3f} insitu={a_ins:.3f} "
              f"FUSION={a_fus:.3f} (Δ vs best {res['fusion_minus_best']:+.3f}, "
              f"site-boot CI {boot['ci95']} P={boot['P_delta_gt0']}) "
              f"helps={res['fusion_helps']} | perm-control={res['permutation_control_fusion_auroc']:.3f}", flush=True)

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nWROTE {OUT}")


if __name__ == "__main__":
    main()
