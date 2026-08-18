#!/usr/bin/env python
"""Sanity-check the 0.97 sensor-only forecast AUROC: persistence vs leakage.

If DO<5 @ t+7d is well-predicted by the CURRENT DO (persistence) and by a plain
model on the sensor embedding, the high AUROC is honest forecasting (dominated by
autocorrelation), and the old 0.54 was an underfit of the linear-projection fusion.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
TB = ROOT / "data" / "processed" / "trackB"
PAIRS = ROOT / "data" / "processed" / "satellite" / "drone_wq_pairs.npz"


def main():
    z = np.load(PAIRS, allow_pickle=True)
    cur_do = z["targets"][:, 0]                      # same-day DO at the station
    emb = np.load(TB / "embeddings_v2.npz", allow_pickle=True)
    rowmap = {int(t): i for i, t in enumerate(emb["tile_idx"])}
    d = np.load(TB / "pairs_forecast.npz", allow_pickle=True)

    rows = [(i, int(t)) for i, t in enumerate(d["tile_idx"]) if int(t) in rowmap]
    li = np.array([r[0] for r in rows])
    tiles = np.array([r[1] for r in rows])
    site = d["site"][li]
    sen = emb["sensor"][[rowmap[t] for t in tiles]]
    cdo = cur_do[tiles]
    valid = np.isfinite(cdo)

    out = {}
    for lab in ["label_h7", "label_h14"]:
        y = d[lab][li].astype(int)
        # 1. persistence: lower current DO -> higher risk (fixed rule)
        pers = roc_auc_score(y[valid], -cdo[valid])
        # 2. "already hypoxic now" share of positives (how trivial is persistence?)
        already = float(((cdo < 5) & (y == 1))[valid].sum() / max(1, (y[valid] == 1).sum()))
        # 3. plain logistic on the sensor embedding (GroupKFold OOF)
        oof = np.zeros(len(y))
        gkf = GroupKFold(n_splits=5)
        for tr, te in gkf.split(sen, y, site):
            sc = StandardScaler().fit(sen[tr])
            m = LogisticRegression(max_iter=2000, C=0.5).fit(sc.transform(sen[tr]), y[tr])
            oof[te] = m.predict_proba(sc.transform(sen[te]))[:, 1]
        logreg = roc_auc_score(y, oof)
        out[lab] = {"persistence_currentDO_auroc": round(float(pers), 3),
                    "pos_already_hypoxic_frac": round(already, 3),
                    "sensor_embed_logreg_auroc": round(float(logreg), 3),
                    "pos": int(y.sum())}
        print(f"{lab}: persistence(currentDO)={pers:.3f}  "
              f"pos_already_hypoxic={already:.2f}  sensor_embed_logreg={logreg:.3f}")
    (ROOT / "results" / "trackB_fusion" / "forecast_persistence_check.json").write_text(
        json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
