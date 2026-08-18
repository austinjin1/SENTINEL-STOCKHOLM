#!/usr/bin/env python
"""C2 validation — is E2E fusion's 0.765 real synergy or just the sensor+spectral
MLP branch (which is ~0.68-0.72 alone)? Ablate the SAME E2E architecture:
  full        = CNN(tile) + MLP(sensor+spectral)
  tile_only   = CNN(tile), sensor input zeroed
  tab_only    = MLP(sensor+spectral), tile input zeroed
Same folds, same seeds, same everything. Also report per-seed spread to check
stability. If 'full' > max(tile_only, tab_only) reliably, synergy is real.
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

ROOT = Path(__file__).resolve().parents[3]
TB = ROOT / "data" / "processed" / "trackB"
PAIRS = ROOT / "data" / "processed" / "satellite" / "drone_wq_pairs.npz"
OUT = ROOT / "results" / "trackB_fusion"
DEV = "cuda" if torch.cuda.is_available() else "cpu"

import importlib.util
spec = importlib.util.spec_from_file_location(
    "m", str(Path(__file__).parent / "07_taskC_representation.py"))
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)


def hm_ci(a, y):
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    se = np.sqrt((a * (1 - a) + (n1 - 1) * (q1 - a**2) + (n0 - 1) * (q2 - a**2)) / (n1 * n0))
    return [round(max(0, a - 1.96 * se), 3), round(min(1, a + 1.96 * se), 3)]


def run(mode, imgs, sen_all, y, site, folds, seeds):
    """mode in {full, tile_only, tab_only}. Returns per-seed OOF list + mean OOF."""
    per_seed = []
    for seed in seeds:
        oof = np.zeros(len(y))
        for tr, te in folds:
            torch.manual_seed(seed); np.random.seed(seed)
            mu = imgs[tr].mean((0, 2, 3), keepdims=True)
            sd = imgs[tr].std((0, 2, 3), keepdims=True) + 1e-6
            Xtr = (imgs[tr] - mu) / sd; Xte = (imgs[te] - mu) / sd
            scn = StandardScaler().fit(sen_all[tr])
            Ntr = scn.transform(sen_all[tr]).astype("float32")
            Nte = scn.transform(sen_all[te]).astype("float32")
            if mode == "tile_only":
                Ntr = np.zeros_like(Ntr); Nte = np.zeros_like(Nte)
            elif mode == "tab_only":
                Xtr = np.zeros_like(Xtr); Xte = np.zeros_like(Xte)
            m = M.E2EFusion(sen_dim=sen_all.shape[1], emb_dim=64, p=0.4)
            p = M.train_cnn(m, Xtr, y[tr], Xte, epochs=25, bs=32, lr=8e-4,
                            sen_tr=Ntr, sen_te=Nte, wd=1e-3)
            oof[te] = p
        per_seed.append(oof)
    mean_oof = np.mean(per_seed, 0)
    seed_aucs = [round(float(roc_auc_score(y, o)), 3) for o in per_seed]
    a = float(roc_auc_score(y, mean_oof))
    return {"auroc": round(a, 3), "ci95": hm_ci(a, y),
            "pr_auc": round(float(average_precision_score(y, mean_oof)), 3),
            "per_seed_auroc": seed_aucs}


def main():
    imgs, sensor, spectral, y, site = M.load_chl()
    folds = M.make_folds(y, site)
    sen_all = np.hstack([sensor, spectral]).astype("float32")
    seeds = [0, 1, 2, 3, 4]
    res = {"n": int(len(y)), "pos": int(y.sum()), "seeds": seeds}
    for mode in ["tab_only", "tile_only", "full"]:
        r = run(mode, imgs, sen_all, y, site, folds, seeds)
        res[mode] = r
        print(f"  {mode:10s} AUROC {r['auroc']:.3f} CI{r['ci95']} "
              f"per-seed {r['per_seed_auroc']}", flush=True)
    best_branch = max(res["tab_only"]["auroc"], res["tile_only"]["auroc"])
    res["synergy_gain"] = round(res["full"]["auroc"] - best_branch, 3)
    res["synergy_real_CI"] = bool(res["full"]["ci95"][0] > best_branch)
    print(f"  best branch={best_branch:.3f} full={res['full']['auroc']:.3f} "
          f"synergy={res['synergy_gain']:+.3f} real(CI)={res['synergy_real_CI']}", flush=True)
    (OUT / "taskC_e2e_ablation.json").write_text(json.dumps(res, indent=2))
    print(f"WROTE {OUT/'taskC_e2e_ablation.json'}")


if __name__ == "__main__":
    main()
