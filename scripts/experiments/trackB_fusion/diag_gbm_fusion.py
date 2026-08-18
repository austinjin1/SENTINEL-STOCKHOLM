#!/usr/bin/env python
"""Decisive 'does fusion help' test on the non-trivial chl-a task, strongest models.

GBM beat the neural adapter on the tabular spectral features, so this uses GBM per
modality and on the concatenation. Station-grouped CV. Forecast tasks are excluded
here because they are persistence-dominated (current-DO rule alone ~0.99) and so
cannot discriminate fusion benefit.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[3]
TB = ROOT / "data" / "processed" / "trackB"


def gbm():
    return GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                      learning_rate=0.05, subsample=0.8, random_state=0)


def cv(X, y, g, ns):
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=ns).split(X, y, g):
        m = gbm().fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return roc_auc_score(y, oof)


def hm_ci(a, y):
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    se = np.sqrt((a * (1 - a) + (n1 - 1) * (q1 - a**2) + (n0 - 1) * (q2 - a**2)) / (n1 * n0))
    return [round(max(0, a - 1.96 * se), 3), round(min(1, a + 1.96 * se), 3)]


def main():
    emb = np.load(TB / "embeddings_v2.npz", allow_pickle=True)
    row = {int(t): i for i, t in enumerate(emb["tile_idx"])}
    d = np.load(TB / "pairs_chl.npz", allow_pickle=True)
    keep = [(row[int(t)], s) for t, s in zip(d["tile_idx"], d["site"]) if int(t) in row]
    ridx = np.array([k[0] for k in keep]); g = np.array([k[1] for k in keep])
    y = d["label"][[i for i, t in enumerate(d["tile_idx"]) if int(t) in row]].astype(int)
    sensor = emb["sensor"][ridx]
    spec_full = emb["spectral"][ridx]
    # spectral_feats() layout (see _fusion.py): [0:16]=per-band stats (noisy),
    # [16:32]=chl-sensitive band ratios + normalized indices. Use only the
    # ratio/index half: it is the chlorophyll-discriminative signal.
    spec = spec_full[:, 16:32]
    clsf = emb["sat"][ridx]
    sat = np.hstack([clsf, spec])
    ns = min(5, len(set(g)))

    feats = {"sensor": sensor, "satellite(spectral)": spec,
             "satellite(spectral+cls)": sat,
             "FUSION sensor+spectral": np.hstack([sensor, spec]),
             "FUSION sensor+spectral+cls": np.hstack([sensor, sat])}
    res = {}
    for name, X in feats.items():
        X = np.nan_to_num(X)
        a = cv(X, y, g, ns)
        res[name] = {"auroc": round(float(a), 3), "ci95": hm_ci(a, y)}
    best_single = max(res["sensor"]["auroc"], res["satellite(spectral)"]["auroc"])
    fus_key = max(["FUSION sensor+spectral", "FUSION sensor+spectral+cls"],
                  key=lambda k: res[k]["auroc"])
    fus = res[fus_key]
    res["_summary"] = {"n": int(len(y)), "pos": int(y.sum()), "best_single": best_single,
                       "best_fusion": fus_key, "fusion": fus["auroc"],
                       "gain": round(fus["auroc"] - best_single, 3),
                       "fusion_helps_CI": bool(fus["ci95"][0] > best_single)}
    print("=== GBM 'does fusion help' on chl-a (station-grouped CV) ===")
    for k, v in res.items():
        if k != "_summary":
            print(f"  {k:26s} AUROC {v['auroc']:.3f} CI{v['ci95']}")
    print("  ->", res["_summary"])
    (ROOT / "results" / "trackB_fusion" / "gbm_fusion_chl.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
