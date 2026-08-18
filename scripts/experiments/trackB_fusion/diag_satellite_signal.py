#!/usr/bin/env python
"""Diagnostic — does the Sentinel-2 tile carry recoverable chl-a signal?

The WaterDroneNet cls_feat scores ~0.53 on chl-a (near chance). But chl-a is
physically encoded in band ratios (green/blue, green/red, red-edge proxies).
This tests whether explicit chlorophyll-sensitive spectral features extract
signal the generic encoder misses. If yes -> engineer them into fusion. If they
are also near chance -> the satellite genuinely lacks the signal at this
co-location tolerance, and we should not pretend otherwise.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
PAIRS = ROOT / "data" / "processed" / "satellite" / "drone_wq_pairs.npz"
TB = ROOT / "data" / "processed" / "trackB"


def spectral_feats(img: np.ndarray) -> list:
    """img [4,H,W] = B02(blue),B03(green),B04(red),B08(NIR). Chl-sensitive set."""
    b, g, r, nir = img[0], img[1], img[2], img[3]
    eps = 1e-6
    f = []
    for ch in (b, g, r, nir):
        f += [float(ch.mean()), float(ch.std()),
              float(np.median(ch)), float(np.percentile(ch, 90))]
    bm, gm, rm, nm = b.mean(), g.mean(), r.mean(), nir.mean()
    # chl-relevant band-ratio panel
    f += [gm / (bm + eps), gm / (rm + eps), rm / (gm + eps), nm / (rm + eps),
          nm / (gm + eps), bm / (gm + eps), (gm - rm) / (gm + rm + eps),
          (bm - gm) / (bm + gm + eps)]
    # pixelwise indices (mean + spatial patchiness via std)
    ndvi = (nir - r) / (nir + r + eps)
    ndwi = (g - nir) / (g + nir + eps)
    gci = g / (b + eps)                       # green chlorophyll index proxy
    grr = (g - r) / (g + r + eps)
    for idx in (ndvi, ndwi, gci, grr):
        f += [float(idx.mean()), float(idx.std())]
    return f


def cv_auroc(X, y, groups, model_fn, n_splits):
    oof = np.zeros(len(y))
    gkf = GroupKFold(n_splits=n_splits)
    for tr, te in gkf.split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        m = model_fn()
        m.fit(sc.transform(X[tr]), y[tr])
        oof[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    return roc_auc_score(y, oof)


def main():
    z = np.load(PAIRS, allow_pickle=True)
    md = json.loads(str(z["metadata"]))
    d = np.load(TB / "pairs_chl.npz", allow_pickle=True)
    idx, y, site = d["tile_idx"], d["label"].astype(int), d["site"]
    imgs = z["images"][idx]
    print(f"chl samples: {len(y)}, pos={int(y.sum())}, sites={len(set(site))}", flush=True)

    X = np.array([spectral_feats(im) for im in imgs], dtype="float64")
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    ns = min(5, len(set(site)))

    res = {}
    res["spectral_logreg"] = cv_auroc(X, y, site, lambda: LogisticRegression(max_iter=2000, C=0.5), ns)
    res["spectral_gbm"] = cv_auroc(X, y, site, lambda: GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8), ns)

    print("\n=== Satellite chl-a signal ceiling (station-grouped CV AUROC) ===")
    print(f"  WaterDroneNet cls_feat (current):  ~0.548  (from fusion_results)")
    for k, v in res.items():
        print(f"  {k:22s} {v:.3f}")
    best = max(res.values())
    print(f"\n  VERDICT: explicit spectral features {'RECOVER' if best > 0.60 else 'do NOT recover'} "
          f"meaningful chl signal (best {best:.3f} vs encoder 0.548).")
    (ROOT / "results" / "trackB_fusion" / "satellite_signal_diag.json").write_text(
        json.dumps({"n": int(len(y)), "pos": int(y.sum()),
                    "wdn_clsfeat_auroc": 0.548, **res}, indent=2))


if __name__ == "__main__":
    main()
