#!/usr/bin/env python
"""Track B, step 3 — train fusion + 3 baselines, report whether fusion helps.

Station-level GroupKFold (no station in both train and test), pooled out-of-fold
predictions over 3 seeds. For each task we train, in the SAME architecture:
  * fusion        — satellite + sensor through PerceiverIO
  * sensor-only   — sensor present, satellite = learned no-data token
  * satellite-only— satellite present, sensor = learned no-data token
  * late-ensemble — two independent unimodal heads, probabilities averaged
Fusion must beat ALL THREE (CI-separated) to claim it helps. Metric: AUROC with
Hanley-McNeil 95% CI + PR-AUC. The honest expected range is modest (0.5-0.8),
NOT the retracted circular 0.99.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import _fusion as F

ROOT = Path(__file__).resolve().parents[3]
TB = ROOT / "data" / "processed" / "trackB"
OUT = ROOT / "results" / "trackB_fusion"
OUT.mkdir(parents=True, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS, BATCH, SEEDS = 60, 512, [0, 1]


def hanley_mcneil_ci(auc, y):
    n1 = int(np.sum(y == 1)); n0 = int(np.sum(y == 0))
    if n1 == 0 or n0 == 0:
        return (None, None)
    q1 = auc / (2 - auc); q2 = 2 * auc * auc / (1 + auc)
    se = np.sqrt((auc * (1 - auc) + (n1 - 1) * (q1 - auc**2) +
                  (n0 - 1) * (q2 - auc**2)) / (n1 * n0))
    return (float(max(0, auc - 1.96 * se)), float(min(1, auc + 1.96 * se)))


def train_predict(kind, sat_tr, sen_tr, y_tr, sat_te, sen_te, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    pos_w = torch.tensor([(len(y_tr) - y_tr.sum()) / max(1, y_tr.sum())], device=DEV)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    st, se = torch.tensor(sat_tr, device=DEV), torch.tensor(sen_tr, device=DEV)
    yt = torch.tensor(y_tr, dtype=torch.float32, device=DEV)
    ste, sete = torch.tensor(sat_te, device=DEV), torch.tensor(sen_te, device=DEV)

    if kind == "late":
        model = F.LateEnsembleHead().to(DEV)
    else:
        model = F.FusionClassifier(num_latents=64).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    n = len(y_tr)
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            if kind == "fusion":
                logit = model(sat=st[b], sensor=se[b])
            elif kind == "sensor":
                logit = model(sensor=se[b])
            elif kind == "sat":
                logit = model(sat=st[b])
            elif kind == "late":
                ls, ln = model(st[b], se[b])
                loss = bce(ls, yt[b]) + bce(ln, yt[b]); loss.backward(); opt.step(); continue
            loss = bce(logit, yt[b]); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        if kind == "fusion":
            p = torch.sigmoid(model(sat=ste, sensor=sete))
        elif kind == "sensor":
            p = torch.sigmoid(model(sensor=sete))
        elif kind == "sat":
            p = torch.sigmoid(model(sat=ste))
        elif kind == "late":
            ls, ln = model(ste, sete)
            p = 0.5 * (torch.sigmoid(ls) + torch.sigmoid(ln))
    return p.cpu().numpy()


def run_task(name, sat, sen, y, groups):
    n_sites = len(set(groups))
    gkf = GroupKFold(n_splits=min(5, n_sites))
    kinds = ["fusion", "sensor", "sat", "late"]
    oof = {k: np.zeros(len(y)) for k in kinds}
    for seed in SEEDS:
        for tr, te in gkf.split(sat, y, groups):
            sc_s, sc_n = StandardScaler().fit(sat[tr]), StandardScaler().fit(sen[tr])
            Xs_tr, Xs_te = sc_s.transform(sat[tr]), sc_s.transform(sat[te])
            Xn_tr, Xn_te = sc_n.transform(sen[tr]), sc_n.transform(sen[te])
            for k in kinds:
                oof[k][te] += train_predict(k, Xs_tr.astype("float32"), Xn_tr.astype("float32"),
                                            y[tr], Xs_te.astype("float32"), Xn_te.astype("float32"),
                                            seed) / len(SEEDS)
    res = {"n": int(len(y)), "pos": int(y.sum()), "n_sites": n_sites}
    for k in kinds:
        auc = float(roc_auc_score(y, oof[k]))
        lo, hi = hanley_mcneil_ci(auc, y)
        res[k] = {"auroc": auc, "ci95": [lo, hi],
                  "pr_auc": float(average_precision_score(y, oof[k]))}
    best_uni = max(res["sensor"]["auroc"], res["sat"]["auroc"], res["late"]["auroc"])
    res["fusion_minus_best_unimodal"] = res["fusion"]["auroc"] - best_uni
    res["fusion_helps"] = bool(res["fusion"]["ci95"][0] > best_uni)
    print(f"\n=== {name} (n={res['n']}, pos={res['pos']}, sites={n_sites}) ===")
    for k in kinds:
        print(f"  {k:8s} AUROC {res[k]['auroc']:.3f} "
              f"CI[{res[k]['ci95'][0]:.3f},{res[k]['ci95'][1]:.3f}] PR {res[k]['pr_auc']:.3f}")
    print(f"  fusion - best_unimodal = {res['fusion_minus_best_unimodal']:+.3f} | "
          f"fusion_helps(CI) = {res['fusion_helps']}")
    return res


def main():
    emb = np.load(TB / "embeddings.npz", allow_pickle=True)
    row = {int(t): i for i, t in enumerate(emb["tile_idx"])}
    sat_all, sen_all = emb["sat"], emb["sensor"]
    results = {"device": DEV, "epochs": EPOCHS, "seeds": SEEDS, "tasks": {}}

    tasks = [("chl_gt8", "pairs_chl.npz", "label"),
             ("forecast_do_lt5_h7", "pairs_forecast.npz", "label_h7"),
             ("forecast_do_lt5_h14", "pairs_forecast.npz", "label_h14")]
    for name, fn, lab in tasks:
        p = TB / fn
        if not p.exists():
            continue
        d = np.load(p, allow_pickle=True)
        rows = [(row[int(t)], int(t), s) for t, s in zip(d["tile_idx"], d["site"]) if int(t) in row]
        if not rows:
            continue
        ridx = np.array([r[0] for r in rows])
        keep = np.array([i for i, r in enumerate(rows)])
        y = d[lab][keep].astype("int64")
        groups = np.array([r[2] for r in rows])
        results["tasks"][name] = run_task(name, sat_all[ridx], sen_all[ridx], y, groups)

    (OUT / "fusion_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nWROTE {OUT/'fusion_results.json'}")


if __name__ == "__main__":
    main()
