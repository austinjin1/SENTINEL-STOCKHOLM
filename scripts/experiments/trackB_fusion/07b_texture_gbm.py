#!/usr/bin/env python
"""C1-supplement — overfitting-resistant test of the 'bloom TEXTURE' hypothesis.

The from-scratch CNN (C1) may under/overfit 51 positives on 4x224x224 tiles.
This tests the same hypothesis with a GBM on RICH SPATIAL/TEXTURE features of the
chlorophyll-sensitive indices: multi-scale patchiness (block-std at several
scales), spatial-gradient energy, quantile spread, and GLCM-style contrast on
the green-chl index. If bloom texture carries signal beyond the 16 global
band-ratios (0.716), these should push above 0.716. Same 5-fold GroupKFold.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[3]
TB = ROOT / "data" / "processed" / "trackB"
PAIRS = ROOT / "data" / "processed" / "satellite" / "drone_wq_pairs.npz"
OUT = ROOT / "results" / "trackB_fusion"
EPS = 1e-6


def block_std(a, k):
    """mean over k x k block stds -> spatial patchiness at scale k."""
    H, W = a.shape
    h, w = H // k * k, W // k * k
    a = a[:h, :w].reshape(h // k, k, w // k, k)
    return float(a.std(axis=(1, 3)).mean())


def texture_feats(img):
    b, g, r, nir = img[0], img[1], img[2], img[3]
    gci = g / (b + EPS)                     # green chlorophyll index
    grr = (g - r) / (g + r + EPS)
    ndci = (nir - r) / (nir + r + EPS)      # NIR-red proxy for surface scum
    f = []
    for idx in (gci, grr, ndci):
        f += [float(idx.mean()), float(idx.std()),
              float(np.percentile(idx, 90) - np.percentile(idx, 10))]  # quantile spread
        # multi-scale patchiness
        for k in (4, 16, 56):
            f.append(block_std(idx, k))
        # spatial gradient energy (edges = bloom fronts / streaks)
        gy, gx = np.gradient(idx.astype("float64"))
        f.append(float(np.sqrt(gx * gx + gy * gy).mean()))
        # GLCM-style contrast proxy: variance of local differences
        f.append(float(((idx[1:, :] - idx[:-1, :]) ** 2).mean()))
    return f


def hm_ci(a, y):
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    se = np.sqrt((a * (1 - a) + (n1 - 1) * (q1 - a**2) + (n0 - 1) * (q2 - a**2)) / (n1 * n0))
    return [round(max(0, a - 1.96 * se), 3), round(min(1, a + 1.96 * se), 3)]


def cv(X, y, g, ns):
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=ns).split(X, y, g):
        m = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                       learning_rate=0.05, subsample=0.8, random_state=0)
        m.fit(X[tr], y[tr]); oof[te] = m.predict_proba(X[te])[:, 1]
    return oof


def main():
    emb = np.load(TB / "embeddings_v2.npz", allow_pickle=True)
    row = {int(t): i for i, t in enumerate(emb["tile_idx"])}
    z = np.load(PAIRS, allow_pickle=True)
    d = np.load(TB / "pairs_chl.npz", allow_pickle=True)
    keep = [(i, row[int(t)], int(t)) for i, t in enumerate(d["tile_idx"]) if int(t) in row]
    li = np.array([k[0] for k in keep]); ti = np.array([k[2] for k in keep])
    ri = np.array([k[1] for k in keep])
    y = d["label"][li].astype(int); site = d["site"][li]
    imgs = z["images"][ti].astype("float32")
    spectral = emb["spectral"][ri][:, 16:32]  # the 0.716 global-ratio feature set

    Xtex = np.array([texture_feats(im) for im in imgs], dtype="float64")
    Xtex = np.nan_to_num(Xtex)
    ns = min(5, len(set(site.tolist())))

    res = {"n": int(len(y)), "pos": int(y.sum()), "n_sites": len(set(site.tolist())),
           "baseline_spectral_ratio_gbm": 0.716}
    for name, X in [("texture_only", Xtex),
                    ("spectral_ratio_only", spectral),
                    ("spectral+texture", np.hstack([spectral, Xtex]))]:
        oof = cv(np.nan_to_num(X), y, site, ns)
        a = float(roc_auc_score(y, oof))
        res[name] = {"auroc": round(a, 3), "ci95": hm_ci(a, y),
                     "pr_auc": round(float(average_precision_score(y, oof)), 3),
                     "beats_0716_CI": bool(hm_ci(a, y)[0] > 0.716)}
        print(f"  {name:22s} AUROC {a:.3f} CI{hm_ci(a,y)} PR {res[name]['pr_auc']}", flush=True)
    (OUT / "taskC_texture_gbm.json").write_text(json.dumps(res, indent=2))
    print(f"WROTE {OUT/'taskC_texture_gbm.json'}")


if __name__ == "__main__":
    main()
