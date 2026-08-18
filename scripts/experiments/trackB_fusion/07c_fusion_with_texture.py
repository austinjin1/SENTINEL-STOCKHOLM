#!/usr/bin/env python
"""Follow-up — with the STRONGER satellite rep (spectral+texture, 0.747), does
GBM fusion now beat the new best single? If the satellite ceiling rose, fusion
may finally have room. Same 5-fold GroupKFold, GBM, Hanley-McNeil CI.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

import importlib.util
ROOT = Path(__file__).resolve().parents[3]
TB = ROOT / "data" / "processed" / "trackB"
PAIRS = ROOT / "data" / "processed" / "satellite" / "drone_wq_pairs.npz"
OUT = ROOT / "results" / "trackB_fusion"

spec = importlib.util.spec_from_file_location(
    "tex", str(Path(__file__).parent / "07b_texture_gbm.py"))
tex = importlib.util.module_from_spec(spec); spec.loader.exec_module(tex)


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
    li = np.array([k[0] for k in keep]); ri = np.array([k[1] for k in keep])
    ti = np.array([k[2] for k in keep])
    y = d["label"][li].astype(int); site = d["site"][li]
    imgs = z["images"][ti].astype("float32")
    sensor = emb["sensor"][ri]
    spectral = emb["spectral"][ri][:, 16:32]
    texture = np.nan_to_num(np.array([tex.texture_feats(im) for im in imgs], dtype="float64"))
    sat_strong = np.hstack([spectral, texture])   # 0.747 rep
    ns = min(5, len(set(site.tolist())))

    feats = {
        "sensor": sensor,
        "satellite_strong(spectral+texture)": sat_strong,
        "FUSION sensor+sat_strong": np.hstack([sensor, sat_strong]),
    }
    res = {"n": int(len(y)), "pos": int(y.sum()), "n_sites": len(set(site.tolist()))}
    for name, X in feats.items():
        oof = cv(np.nan_to_num(X), y, site, ns)
        a = float(roc_auc_score(y, oof))
        res[name] = {"auroc": round(a, 3), "ci95": hm_ci(a, y),
                     "pr_auc": round(float(average_precision_score(y, oof)), 3)}
        print(f"  {name:36s} AUROC {a:.3f} CI{hm_ci(a,y)}", flush=True)
    best_single = max(res["sensor"]["auroc"], res["satellite_strong(spectral+texture)"]["auroc"])
    fus = res["FUSION sensor+sat_strong"]
    res["best_single"] = best_single
    res["fusion_gain"] = round(fus["auroc"] - best_single, 3)
    res["fusion_helps_CI"] = bool(fus["ci95"][0] > best_single)
    print(f"  best_single={best_single:.3f} fusion={fus['auroc']:.3f} "
          f"gain={res['fusion_gain']:+.3f} helps(CI)={res['fusion_helps_CI']}", flush=True)
    (OUT / "taskC_fusion_with_texture.json").write_text(json.dumps(res, indent=2))
    print(f"WROTE {OUT/'taskC_fusion_with_texture.json'}")


if __name__ == "__main__":
    main()
