#!/usr/bin/env python
"""Track B 5 — STRENGTHENED fusion: spectral-augmented satellite + nonlinear adapters.

The diagnostic showed chl-a signal IS recoverable from the tile (spectral GBM
0.70 vs encoder 0.55). Here the satellite branch = [WaterDroneNet cls_feat (384)
+ chl-sensitive spectral panel (28)] through a nonlinear adapter, so fusion can
exploit it. We compare, with station-grouped CV:
  sensor-only / satellite-only / fusion (Perceiver) / fusion (simple concat).
Fusion "wins" only if its CI lower bound exceeds the best single encoder.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import _fusion as F

ROOT = Path(__file__).resolve().parents[3]
TB = ROOT / "data" / "processed" / "trackB"
OUT = ROOT / "results" / "trackB_fusion"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS, BATCH, SEEDS = 70, 512, [0, 1]
SAT_DIM = 384 + F.N_SPECTRAL


def hm_ci(auc, y):
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if not n1 or not n0:
        return [None, None]
    q1, q2 = auc / (2 - auc), 2 * auc * auc / (1 + auc)
    se = np.sqrt((auc * (1 - auc) + (n1 - 1) * (q1 - auc**2) + (n0 - 1) * (q2 - auc**2)) / (n1 * n0))
    return [float(max(0, auc - 1.96 * se)), float(min(1, auc + 1.96 * se))]


def train_predict(kind, sat_tr, sen_tr, y_tr, sat_te, sen_te, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    pw = torch.tensor([(len(y_tr) - y_tr.sum()) / max(1, y_tr.sum())], device=DEV)
    bce = nn.BCEWithLogitsLoss(pos_weight=pw)
    st, se = torch.tensor(sat_tr, device=DEV), torch.tensor(sen_tr, device=DEV)
    yt = torch.tensor(y_tr, dtype=torch.float32, device=DEV)
    ste, sete = torch.tensor(sat_te, device=DEV), torch.tensor(sen_te, device=DEV)
    if kind == "fusion_concat":
        model = F.ConcatFusion(SAT_DIM).to(DEV)
    else:
        model = F.FusionClassifierV2(SAT_DIM).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    n = len(y_tr)
    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            if kind == "sensor":
                logit = model(sensor=se[b])
            elif kind == "sat":
                logit = model(sat=st[b])
            elif kind == "fusion_perc":
                logit = model(sat=st[b], sensor=se[b])
            else:
                logit = model(st[b], se[b])
            bce(logit, yt[b]).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        if kind == "sensor":
            p = torch.sigmoid(model(sensor=sete))
        elif kind == "sat":
            p = torch.sigmoid(model(sat=ste))
        elif kind == "fusion_perc":
            p = torch.sigmoid(model(sat=ste, sensor=sete))
        else:
            p = torch.sigmoid(model(ste, sete))
    return p.cpu().numpy()


def run_task(name, sat, sen, y, groups):
    ns = min(5, len(set(groups)))
    gkf = GroupKFold(n_splits=ns)
    kinds = ["sensor", "sat", "fusion_perc", "fusion_concat"]
    oof = {k: np.zeros(len(y)) for k in kinds}
    for seed in SEEDS:
        for tr, te in gkf.split(sat, y, groups):
            # standardize cls_feat, spectral, sensor separately on train
            sc_c = StandardScaler().fit(sat[tr, :384]); sc_s = StandardScaler().fit(sat[tr, 384:])
            sc_n = StandardScaler().fit(sen[tr])
            def Z(a): return np.hstack([sc_c.transform(a[:, :384]), sc_s.transform(a[:, 384:])]).astype("float32")
            Xtr, Xte = Z(sat[tr]), Z(sat[te])
            Ntr, Nte = sc_n.transform(sen[tr]).astype("float32"), sc_n.transform(sen[te]).astype("float32")
            for k in kinds:
                oof[k][te] += train_predict(k, Xtr, Ntr, y[tr], Xte, Nte, seed) / len(SEEDS)
    res = {"n": int(len(y)), "pos": int(y.sum()), "n_sites": len(set(groups))}
    for k in kinds:
        a = float(roc_auc_score(y, oof[k]))
        res[k] = {"auroc": a, "ci95": hm_ci(a, y), "pr_auc": float(average_precision_score(y, oof[k]))}
    best_single = max(res["sensor"]["auroc"], res["sat"]["auroc"])
    best_fus = max(res["fusion_perc"]["auroc"], res["fusion_concat"]["auroc"])
    which = "fusion_perc" if res["fusion_perc"]["auroc"] >= res["fusion_concat"]["auroc"] else "fusion_concat"
    res["best_single"] = best_single
    res["best_fusion"] = best_fus
    res["fusion_gain"] = best_fus - best_single
    res["fusion_helps"] = bool(res[which]["ci95"][0] > best_single)
    print(f"\n=== {name} (n={res['n']}, pos={res['pos']}, sites={res['n_sites']}) ===")
    for k in kinds:
        print(f"  {k:14s} AUROC {res[k]['auroc']:.3f} CI[{res[k]['ci95'][0]:.3f},{res[k]['ci95'][1]:.3f}] PR {res[k]['pr_auc']:.3f}")
    print(f"  best single={best_single:.3f}  best fusion={best_fus:.3f}  gain={res['fusion_gain']:+.3f}  fusion_helps(CI)={res['fusion_helps']}")
    return res


def main():
    emb = np.load(TB / "embeddings_v2.npz", allow_pickle=True)
    row = {int(t): i for i, t in enumerate(emb["tile_idx"])}
    sat_aug = np.hstack([emb["sat"], emb["spectral"]]).astype("float32")
    sen = emb["sensor"]
    results = {"epochs": EPOCHS, "seeds": SEEDS, "sat_dim": SAT_DIM, "tasks": {}}
    for name, fn, lab in [("chl_gt8", "pairs_chl.npz", "label"),
                          ("forecast_do_lt5_h7", "pairs_forecast.npz", "label_h7"),
                          ("forecast_do_lt5_h14", "pairs_forecast.npz", "label_h14")]:
        d = np.load(TB / fn, allow_pickle=True)
        rows = [(row[int(t)], s) for t, s in zip(d["tile_idx"], d["site"]) if int(t) in row]
        ridx = np.array([r[0] for r in rows]); groups = np.array([r[1] for r in rows])
        y = d[lab][[i for i, t in enumerate(d["tile_idx"]) if int(t) in row]].astype("int64")
        results["tasks"][name] = run_task(name, sat_aug[ridx], sen[ridx], y, groups)
    (OUT / "fusion_results_v2.json").write_text(json.dumps(results, indent=2))
    print(f"\nWROTE {OUT/'fusion_results_v2.json'}")


if __name__ == "__main__":
    main()
