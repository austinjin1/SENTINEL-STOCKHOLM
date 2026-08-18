#!/usr/bin/env python
"""Track B 6 — does fusion help on a COMPLEMENTARY anomaly task?

Motivation: on chl-a, fusion tied the best single encoder — but chl-a is
optically dominated (the satellite alone sees algae, so the sensor is redundant),
and the plain DO forecast was persistence-trivial (88% of positives were already
hypoxic). Neither is a task where two modalities are complementary.

The right test for the SENTINEL premise — early warning of contamination — is
**new-onset forecasting**: among sites that are CURRENTLY HEALTHY (DO >= 5 mg/L),
predict a FUTURE hypoxia onset (DO < 5 at t+H). Persistence is defeated by
construction (all inputs are non-hypoxic now), so the model must read genuine
precursors. Here the sensor's chemistry trajectory and the satellite's surface
state are plausibly COMPLEMENTARY. If fusion beats both single modalities here,
the earlier "fusion doesn't help" was a wrong-task artifact.

Non-circular: the label is a FUTURE DO value, not contained in any input. Rich
sensor features (AquaSSM embedding + current 5-param chemistry) are legitimate
precursors. Satellite = chl-sensitive spectral band-ratios. Station-grouped CV.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[3]
TB = ROOT / "data" / "processed" / "trackB"
PAIRS = ROOT / "data" / "processed" / "satellite" / "drone_wq_pairs.npz"
OUT = ROOT / "results" / "trackB_fusion"


def hm_ci(a, y):
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if not n1 or not n0:
        return [None, None]
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    se = np.sqrt((a * (1 - a) + (n1 - 1) * (q1 - a**2) + (n0 - 1) * (q2 - a**2)) / (n1 * n0))
    return [round(float(max(0, a - 1.96 * se)), 3), round(float(min(1, a + 1.96 * se)), 3)]


def hgb():
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                          max_iter=300, l2_regularization=1.0,
                                          class_weight="balanced", random_state=0)


def cv_oof(X, y, g, ns):
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=ns).split(X, y, g):
        m = hgb().fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof


def evaluate(name, feats, y, g):
    ns = min(5, len(set(g)))
    res = {"n": int(len(y)), "pos": int(y.sum()), "n_sites": len(set(g))}
    oofs = {}
    for k, X in feats.items():
        oof = cv_oof(np.nan_to_num(X, nan=np.nan) if False else X, y, g, ns)
        oofs[k] = oof
        a = float(roc_auc_score(y, oof))
        res[k] = {"auroc": round(a, 3), "ci95": hm_ci(a, y),
                  "pr_auc": round(float(average_precision_score(y, oof)), 3)}
    best_single = max(res["sensor"]["auroc"], res["satellite"]["auroc"])
    fus = res["fusion"]
    res["best_single"] = best_single
    res["fusion_gain"] = round(fus["auroc"] - best_single, 3)
    res["fusion_helps_CI"] = bool(fus["ci95"][0] > best_single)
    res["persistence_auroc"] = res.get("persistence", {}).get("auroc")
    print(f"\n=== {name} (n={res['n']}, pos={res['pos']}, sites={res['n_sites']}) ===")
    for k in ["persistence", "sensor", "satellite", "fusion"]:
        if k in res and isinstance(res[k], dict):
            print(f"  {k:11s} AUROC {res[k]['auroc']:.3f} CI{res[k]['ci95']} PR {res[k]['pr_auc']:.3f}")
    print(f"  best single={best_single:.3f}  fusion gain={res['fusion_gain']:+.3f}  "
          f"fusion_helps(CI)={res['fusion_helps_CI']}")
    return res


def main():
    emb = np.load(TB / "embeddings_v2.npz", allow_pickle=True)
    row = {int(t): i for i, t in enumerate(emb["tile_idx"])}
    z = np.load(PAIRS, allow_pickle=True)
    cur = z["targets"]                                   # [N,5] current DO,pH,Turb,Temp,SpCond
    d = np.load(TB / "pairs_forecast.npz", allow_pickle=True)

    keep = [(i, row[int(t)], int(t)) for i, t in enumerate(d["tile_idx"]) if int(t) in row]
    li = np.array([k[0] for k in keep])
    ri = np.array([k[1] for k in keep])
    ti = np.array([k[2] for k in keep])
    site = d["site"][li]
    sensor_emb = emb["sensor"][ri]                       # 256 (AquaSSM window embedding)
    spectral = emb["spectral"][ri][:, 16:32]             # 16 chl-sensitive band-ratios
    cur_wq = cur[ti]                                     # 5 current chemistry values
    cur_do = cur_wq[:, 0]

    results = {"task": "new_onset_forecast", "tasks": {}}
    for h, lab in [("h7", "label_h7"), ("h14", "label_h14")]:
        healthy = np.isfinite(cur_do) & (cur_do >= 5.0)   # currently NOT hypoxic
        y = d[lab][li][healthy].astype(int)               # future DO < 5 (new onset)
        g = site[healthy]
        sen = np.hstack([sensor_emb[healthy], cur_wq[healthy]])   # emb + current chemistry
        sat = spectral[healthy]
        feats = {
            "persistence": cur_do[healthy].reshape(-1, 1),       # current DO only
            "sensor": sen,
            "satellite": sat,
            "fusion": np.hstack([sen, sat]),
        }
        results["tasks"][f"new_onset_do_{h}"] = evaluate(f"new_onset_do_{h}", feats, y, g)

    (OUT / "complementary_fusion.json").write_text(json.dumps(results, indent=2))
    print(f"\nWROTE {OUT/'complementary_fusion.json'}")


if __name__ == "__main__":
    main()
