#!/usr/bin/env python
"""TASK C — Representation & Architecture: does a learned satellite rep or
end-to-end / gated fusion change the chl-a verdict?

Baselines to beat (from gbm_fusion_chl.json, n=362, pos=51, 33 sites, 5-fold GroupKFold):
  satellite(spectral-GBM) = 0.716   <- ceiling
  sensor(AquaSSM emb GBM) = 0.681
  fusion(GBM concat)      = 0.708   (ties, no gain)

C1  Small CNN on RAW Sentinel-2 tiles (4x224x224) for chl-a, station holdout.
    Can a learned bloom-texture representation beat the 0.716 spectral ceiling?
C2  END-TO-END fusion: jointly train CNN tile encoder + sensor MLP + fusion head
    vs frozen-embedding GBM fusion (0.708).
C3  GATED / cross-attention fusion that can DOWN-WEIGHT the weak modality per
    sample. Does it reliably MATCH the best single (never worse)?

All experiments share the SAME 5-fold GroupKFold split (33 sites) so numbers are
directly comparable. Station-grouped, non-circular label (chl-a > 8 ug/L, future
of no input), Hanley-McNeil CI. REAL data only.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
TB = ROOT / "data" / "processed" / "trackB"
PAIRS = ROOT / "data" / "processed" / "satellite" / "drone_wq_pairs.npz"
OUT = ROOT / "results" / "trackB_fusion"
DEV = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 0
N_SPLITS = 5


def hm_ci(a, y):
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if not n1 or not n0:
        return [None, None]
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    se = np.sqrt((a * (1 - a) + (n1 - 1) * (q1 - a**2) + (n0 - 1) * (q2 - a**2)) / (n1 * n0))
    return [round(float(max(0, a - 1.96 * se)), 3), round(float(min(1, a + 1.96 * se)), 3)]


def summ(name, y, oof):
    a = float(roc_auc_score(y, oof))
    return {"name": name, "auroc": round(a, 3), "ci95": hm_ci(a, y),
            "pr_auc": round(float(average_precision_score(y, oof)), 3)}


# ---------------------------------------------------------------- data
def load_chl():
    emb = np.load(TB / "embeddings_v2.npz", allow_pickle=True)
    row = {int(t): i for i, t in enumerate(emb["tile_idx"])}
    z = np.load(PAIRS, allow_pickle=True)
    d = np.load(TB / "pairs_chl.npz", allow_pickle=True)
    keep = [(i, row[int(t)], int(t)) for i, t in enumerate(d["tile_idx"]) if int(t) in row]
    li = np.array([k[0] for k in keep]); ri = np.array([k[1] for k in keep])
    ti = np.array([k[2] for k in keep])
    y = d["label"][li].astype(int)
    site = d["site"][li]
    sensor = emb["sensor"][ri].astype("float32")              # 256 AquaSSM
    spectral = emb["spectral"][ri][:, 16:32].astype("float32")  # 16 chl ratios
    imgs = z["images"][ti].astype("float32")                  # [N,4,224,224]
    return imgs, sensor, spectral, y, site


# ---------------------------------------------------------------- models
class TileCNN(nn.Module):
    """Small CNN encoder over 4x224x224 -> emb_dim. Global stats + texture."""
    def __init__(self, emb_dim=64, p=0.3):
        super().__init__()
        def blk(ci, co):
            return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co),
                                 nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.f = nn.Sequential(blk(4, 16), blk(16, 32), blk(32, 64), blk(64, 64))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(p), nn.Linear(64, emb_dim))

    def forward(self, x):
        return self.head(self.f(x))


class CNNClassifier(nn.Module):
    def __init__(self, emb_dim=64, p=0.3):
        super().__init__()
        self.enc = TileCNN(emb_dim, p)
        self.cls = nn.Linear(emb_dim, 1)

    def forward(self, x, return_emb=False):
        e = self.enc(x)
        if return_emb:
            return e
        return self.cls(e).squeeze(-1)


class E2EFusion(nn.Module):
    """End-to-end: trainable CNN(tile) + MLP(sensor+spectral) -> concat -> head."""
    def __init__(self, sen_dim, emb_dim=64, p=0.3):
        super().__init__()
        self.cnn = TileCNN(emb_dim, p)
        self.sen = nn.Sequential(nn.Linear(sen_dim, 64), nn.ReLU(inplace=True), nn.Dropout(p))
        self.cls = nn.Sequential(nn.Linear(emb_dim + 64, 64), nn.ReLU(inplace=True),
                                 nn.Dropout(p), nn.Linear(64, 1))

    def forward(self, x, s):
        return self.cls(torch.cat([self.cnn(x), self.sen(s)], -1)).squeeze(-1)


class GatedFusion(nn.Module):
    """Per-sample gated fusion on FROZEN features. A learned gate g in (0,1) per
    modality lets the model DOWN-WEIGHT the weak modality per sample. Each branch
    also has its own linear logit so a pure-single-modality solution is reachable.
    Final logit = softmax-normalized gate-weighted mix of the two branch logits
    plus a small joint MLP term. Designed so fusion can collapse to best single."""
    def __init__(self, sat_dim, sen_dim, hid=64, p=0.3):
        super().__init__()
        self.sat = nn.Sequential(nn.Linear(sat_dim, hid), nn.ReLU(inplace=True), nn.Dropout(p))
        self.sen = nn.Sequential(nn.Linear(sen_dim, hid), nn.ReLU(inplace=True), nn.Dropout(p))
        self.sat_logit = nn.Linear(hid, 1)
        self.sen_logit = nn.Linear(hid, 1)
        # gate sees both branch summaries, outputs 2 logits -> softmax weights
        self.gate = nn.Sequential(nn.Linear(2 * hid, hid), nn.ReLU(inplace=True),
                                  nn.Dropout(p), nn.Linear(hid, 2))
        self.joint = nn.Sequential(nn.Linear(2 * hid, hid), nn.ReLU(inplace=True),
                                   nn.Dropout(p), nn.Linear(hid, 1))
        self.alpha = nn.Parameter(torch.tensor(-2.0))  # gate the joint term small at init

    def forward(self, sat, sen, return_gate=False):
        hs, hn = self.sat(sat), self.sen(sen)
        h = torch.cat([hs, hn], -1)
        w = torch.softmax(self.gate(h), -1)            # [B,2] per-sample modality weights
        ls, ln = self.sat_logit(hs).squeeze(-1), self.sen_logit(hn).squeeze(-1)
        mix = w[:, 0] * ls + w[:, 1] * ln
        out = mix + torch.sigmoid(self.alpha) * self.joint(h).squeeze(-1)
        if return_gate:
            return out, w
        return out


# ---------------------------------------------------------------- train loops
def make_folds(y, site):
    gkf = GroupKFold(n_splits=N_SPLITS)
    return list(gkf.split(np.zeros(len(y)), y, site))


def train_cnn(model, Xtr, ytr, Xte, epochs, bs, lr, sen_tr=None, sen_te=None, wd=1e-4):
    model.to(DEV)
    pw = torch.tensor([(len(ytr) - ytr.sum()) / max(1, ytr.sum())], device=DEV)
    bce = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    Xtr_t = torch.tensor(Xtr, device=DEV); yt = torch.tensor(ytr, dtype=torch.float32, device=DEV)
    st = torch.tensor(sen_tr, device=DEV) if sen_tr is not None else None
    n = len(ytr)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            if st is None:
                logit = model(Xtr_t[b])
            else:
                logit = model(Xtr_t[b], st[b])
            # light augmentation for CNN: random flips on the tile
            bce(logit, yt[b]).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        Xte_t = torch.tensor(Xte, device=DEV)
        if sen_te is None:
            p = torch.sigmoid(model(Xte_t)).cpu().numpy()
        else:
            sete = torch.tensor(sen_te, device=DEV)
            p = torch.sigmoid(model(Xte_t, sete)).cpu().numpy()
    return p


# ---------------------------------------------------------------- experiments
def exp_c1_cnn(imgs, y, site, folds):
    """CNN on raw tiles, station holdout. Multi-seed OOF."""
    print("\n[C1] CNN on raw Sentinel-2 tiles (chl-a) ...", flush=True)
    seeds = [0, 1, 2]
    oof = np.zeros(len(y))
    for seed in seeds:
        for tr, te in folds:
            torch.manual_seed(seed); np.random.seed(seed)
            # per-fold channel standardization from train tiles
            mu = imgs[tr].mean((0, 2, 3), keepdims=True)
            sd = imgs[tr].std((0, 2, 3), keepdims=True) + 1e-6
            Xtr = (imgs[tr] - mu) / sd
            Xte = (imgs[te] - mu) / sd
            m = CNNClassifier(emb_dim=64, p=0.4)
            p = train_cnn(m, Xtr, y[tr], Xte, epochs=25, bs=32, lr=8e-4, wd=1e-3)
            oof[te] += p / len(seeds)
    return summ("C1_cnn_raw_tile", y, oof)


def exp_c2_e2e(imgs, sen_all, y, site, folds):
    """End-to-end fusion: trainable CNN(tile)+MLP(sensor+spectral)+head."""
    print("\n[C2] End-to-end fusion (trainable CNN + sensor MLP) ...", flush=True)
    seeds = [0, 1, 2]
    oof = np.zeros(len(y))
    for seed in seeds:
        for tr, te in folds:
            torch.manual_seed(seed); np.random.seed(seed)
            mu = imgs[tr].mean((0, 2, 3), keepdims=True)
            sd = imgs[tr].std((0, 2, 3), keepdims=True) + 1e-6
            Xtr = (imgs[tr] - mu) / sd; Xte = (imgs[te] - mu) / sd
            scn = StandardScaler().fit(sen_all[tr])
            Ntr = scn.transform(sen_all[tr]).astype("float32")
            Nte = scn.transform(sen_all[te]).astype("float32")
            m = E2EFusion(sen_dim=sen_all.shape[1], emb_dim=64, p=0.4)
            p = train_cnn(m, Xtr, y[tr], Xte, epochs=25, bs=32, lr=8e-4,
                          sen_tr=Ntr, sen_te=Nte, wd=1e-3)
            oof[te] += p / len(seeds)
    return summ("C2_e2e_fusion", y, oof)


def train_gated(sat, sen, y, folds, seeds=(0, 1, 2, 3), epochs=200, lr=1e-3):
    oof = np.zeros(len(y))
    gate_sat_pos, gate_sat_neg, n_pos, n_neg = 0.0, 0.0, 0, 0
    for seed in seeds:
        for tr, te in folds:
            torch.manual_seed(seed); np.random.seed(seed)
            scs = StandardScaler().fit(sat[tr]); scn = StandardScaler().fit(sen[tr])
            Str = scs.transform(sat[tr]).astype("float32"); Ste = scs.transform(sat[te]).astype("float32")
            Ntr = scn.transform(sen[tr]).astype("float32"); Nte = scn.transform(sen[te]).astype("float32")
            m = GatedFusion(sat.shape[1], sen.shape[1], hid=64, p=0.4).to(DEV)
            pw = torch.tensor([(len(y[tr]) - y[tr].sum()) / max(1, y[tr].sum())], device=DEV)
            bce = nn.BCEWithLogitsLoss(pos_weight=pw)
            opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-3)
            St = torch.tensor(Str, device=DEV); Nt = torch.tensor(Ntr, device=DEV)
            yt = torch.tensor(y[tr], dtype=torch.float32, device=DEV)
            for ep in range(epochs):
                m.train(); opt.zero_grad()
                logit = m(St, Nt)
                bce(logit, yt).backward(); opt.step()
            m.eval()
            with torch.no_grad():
                out, w = m(torch.tensor(Ste, device=DEV), torch.tensor(Nte, device=DEV), return_gate=True)
                oof[te] += torch.sigmoid(out).cpu().numpy() / len(seeds)
                ws = w[:, 0].cpu().numpy()  # satellite gate weight
                yte = y[te]
                gate_sat_pos += ws[yte == 1].sum(); n_pos += (yte == 1).sum()
                gate_sat_neg += ws[yte == 0].sum(); n_neg += (yte == 0).sum()
    r = summ("C3_gated_fusion", y, oof)
    r["mean_sat_gate_pos"] = round(float(gate_sat_pos / max(1, n_pos)), 3)
    r["mean_sat_gate_neg"] = round(float(gate_sat_neg / max(1, n_neg)), 3)
    return r


def main():
    t0 = time.time()
    imgs, sensor, spectral, y, site = load_chl()
    print(f"chl-a: n={len(y)} pos={int(y.sum())} sites={len(set(site.tolist()))}", flush=True)
    folds = make_folds(y, site)
    sen_all = np.hstack([sensor, spectral]).astype("float32")  # sensor+spectral MLP input

    results = {"task": "chl_a_gt8", "n": int(len(y)), "pos": int(y.sum()),
               "n_sites": len(set(site.tolist())),
               "baselines": {"satellite_spectral_gbm": 0.716, "sensor_gbm": 0.681,
                             "fusion_gbm": 0.708},
               "experiments": {}}

    # C1
    r1 = exp_c1_cnn(imgs, y, site, folds)
    results["experiments"]["C1_cnn_raw_tile"] = r1
    print(f"  C1 CNN raw tile AUROC {r1['auroc']:.3f} CI{r1['ci95']} PR {r1['pr_auc']}", flush=True)

    # C2
    r2 = exp_c2_e2e(imgs, sen_all, y, site, folds)
    results["experiments"]["C2_e2e_fusion"] = r2
    print(f"  C2 end-to-end fusion AUROC {r2['auroc']:.3f} CI{r2['ci95']} PR {r2['pr_auc']}", flush=True)

    # C3 — gated fusion on FROZEN features. Satellite branch = spectral(16),
    # sensor branch = AquaSSM emb(256). (Matches the GBM-fusion feature sets.)
    print("\n[C3] Gated fusion on frozen features (spectral vs sensor) ...", flush=True)
    r3 = train_gated(spectral, sensor, y, folds)
    results["experiments"]["C3_gated_fusion"] = r3
    print(f"  C3 gated fusion AUROC {r3['auroc']:.3f} CI{r3['ci95']} PR {r3['pr_auc']}"
          f"  sat_gate(pos/neg)={r3['mean_sat_gate_pos']}/{r3['mean_sat_gate_neg']}", flush=True)

    # verdicts
    ceil = 0.716
    for key, r in results["experiments"].items():
        r["beats_satellite_ceiling"] = bool(r["ci95"][0] > ceil)
        r["vs_ceiling_point"] = round(r["auroc"] - ceil, 3)
    results["runtime_sec"] = round(time.time() - t0, 1)
    (OUT / "taskC_representation.json").write_text(json.dumps(results, indent=2))
    print(f"\nWROTE {OUT/'taskC_representation.json'}  ({results['runtime_sec']}s)")


if __name__ == "__main__":
    main()
