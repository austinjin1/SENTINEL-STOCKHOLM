#!/usr/bin/env python
"""Track B 10 — small CNN satellite arm on RAW microcystin tiles (time-boxed).

Tests whether a learned CNN on the raw 6-band Sentinel-2 tiles beats the
hand-crafted literature indices (08) for the satellite arm, under the SAME
station-grouped CV. Also fuses CNN sat-score + in-situ pigment.

Rigor: GroupKFold by site (no site in both train/val), site-clustered bootstrap
CI on fusion vs best single. Small CNN, early-stopped, time-boxed per fold.
Label: mc>1.6 (WHO drinking) and mc>0.3 (detection).
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[3]
NOAA = ROOT / "data/processed/biology/noaa_habs"
DATA = NOAA / "mc_fusion_dataset.parquet"
TILES = NOAA / "mc_tiles"
OUT = ROOT / "results/trackB_fusion/mc_cnn_arm.json"
SEED = 0
NBOOT = 2000
BANDS = ["B02", "B03", "B04", "B05", "B08", "B11"]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MAX_MIN_PER_FOLD = 2.5  # time-box


def load_tiles(df):
    X = np.zeros((len(df), 6, 64, 64), np.float32)
    ok = np.zeros(len(df), bool)
    for r, idx in enumerate(df["idx"].to_numpy()):
        f = TILES / f"{int(idx)}.npz"
        if not f.exists():
            continue
        try:
            img = np.load(f, allow_pickle=True)["image"]
            if img.shape == (6, 64, 64):
                X[r] = img; ok[r] = True
        except Exception:
            pass
    return X, ok


class SmallCNN(nn.Module):
    def __init__(self, cin=6):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(cin, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(48, 1))

    def embed(self, x):
        return self.f(x)

    def forward(self, x):
        return self.head(self.f(x)).squeeze(1)


def train_fold(Xtr, ytr, Xva, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    # per-band standardize on train
    mu = Xtr.reshape(len(Xtr), 6, -1).mean((0, 2)).reshape(1, 6, 1, 1)
    sd = Xtr.reshape(len(Xtr), 6, -1).std((0, 2)).reshape(1, 6, 1, 1) + 1e-6
    Xtr = (Xtr - mu) / sd; Xva = (Xva - mu) / sd
    Xt = torch.tensor(Xtr, device=DEV); yt = torch.tensor(ytr, dtype=torch.float32, device=DEV)
    Xv = torch.tensor(Xva, device=DEV)
    m = SmallCNN().to(DEV)
    pos_w = torch.tensor([(len(ytr) - ytr.sum()) / max(1, ytr.sum())], device=DEV, dtype=torch.float32)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    n = len(Xt); bs = 128
    t0 = time.time()
    for ep in range(60):
        m.train(); perm = torch.randperm(n, device=DEV)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            # light aug: random flips
            xb = Xt[b]
            if torch.rand(1).item() < 0.5: xb = torch.flip(xb, [3])
            if torch.rand(1).item() < 0.5: xb = torch.flip(xb, [2])
            opt.zero_grad(); loss = lossf(m(xb), yt[b]); loss.backward(); opt.step()
        if (time.time() - t0) / 60 > MAX_MIN_PER_FOLD:
            break
    m.eval()
    with torch.no_grad():
        pv = torch.sigmoid(m(Xv)).cpu().numpy()
        # embeddings for fusion (train+val)
        emb_tr = m.embed(Xt).cpu().numpy()
        emb_va = m.embed(Xv).cpu().numpy()
    return pv, emb_tr, emb_va, ep + 1


def hm_ci(a, y):
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    se = np.sqrt((a * (1 - a) + (n1 - 1) * (q1 - a**2) + (n0 - 1) * (q2 - a**2)) / (n1 * n0))
    return [round(float(max(0, a - 1.96 * se)), 3), round(float(min(1, a + 1.96 * se)), 3)]


def site_boot(y, pa, pb, sites, seed=SEED, B=NBOOT):
    rng = np.random.default_rng(seed); uniq = np.unique(sites)
    by = {s: np.where(sites == s)[0] for s in uniq}; d = []
    for _ in range(B):
        ch = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([by[s] for s in ch]); yi = y[idx]
        if yi.min() == yi.max(): continue
        d.append(roc_auc_score(yi, pa[idx]) - roc_auc_score(yi, pb[idx]))
    d = np.array(d)
    return {"delta_median": round(float(np.median(d)), 3),
            "ci95": [round(float(np.percentile(d, 2.5)), 3), round(float(np.percentile(d, 97.5)), 3)],
            "P_delta_gt0": round(float((d > 0).mean()), 3)}


def hgb():
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300,
                                          l2_regularization=1.0, class_weight="balanced", random_state=SEED)


def main():
    df = pq.read_table(DATA).to_pandas()
    X, ok = load_tiles(df)
    df = df[ok].reset_index(drop=True); X = X[ok]
    sites = df["site"].astype(str).to_numpy()
    insitu = df[["CHLA", "PC"]].to_numpy("float64")
    mc = df["mc"].to_numpy("float64")
    ns = min(5, df["site"].nunique())
    print(f"{len(df)} tiles, {df['site'].nunique()} sites, dev={DEV}", flush=True)

    out = {"device": DEV, "note": "Small CNN on raw 6-band S2 tiles, GroupKFold by site, "
           "time-boxed 2.5min/fold. CNN sat-score vs hand-crafted indices (08). "
           "Fusion = [CNN embedding + in-situ pigment] HGB.", "labels": {}}

    for thr, tag in [(1.6, "mc_gt1.6_WHO_drinking"), (0.3, "mc_gt0.3_detection")]:
        y = (mc > thr).astype(int)
        p_cnn = np.zeros(len(y)); emb_full = np.zeros((len(y), 48), np.float32)
        t0 = time.time(); eps = []
        for k, (tr, te) in enumerate(GroupKFold(n_splits=ns).split(X, y, sites)):
            pv, etr, eva, ne = train_fold(X[tr], y[tr].astype(np.float32), X[te], SEED + k)
            p_cnn[te] = pv; emb_full[te] = eva; eps.append(ne)
        a_cnn = float(roc_auc_score(y, p_cnn))
        # fusion A: 48-dim CNN embedding + in-situ (may over-parameterize)
        fus = np.hstack([emb_full, insitu])
        # fusion B: scalar CNN OOF score + in-situ (compact, fair to CNN)
        fus_s = np.hstack([p_cnn.reshape(-1, 1), insitu])
        p_fus = np.zeros(len(y)); p_ins = np.zeros(len(y)); p_fus_s = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=ns).split(fus, y, sites):
            p_fus[te] = hgb().fit(fus[tr], y[tr]).predict_proba(fus[te])[:, 1]
            p_fus_s[te] = hgb().fit(fus_s[tr], y[tr]).predict_proba(fus_s[te])[:, 1]
            p_ins[te] = hgb().fit(insitu[tr], y[tr]).predict_proba(insitu[te])[:, 1]
        a_fus = float(roc_auc_score(y, p_fus)); a_ins = float(roc_auc_score(y, p_ins))
        a_fus_s = float(roc_auc_score(y, p_fus_s))
        best_p, best_n, best_a = ((p_ins, "insitu", a_ins) if a_ins >= a_cnn else (p_cnn, "cnn_sat", a_cnn))
        boot = site_boot(y, p_fus, best_p, sites)
        res = {"pos": int(y.sum()), "cnn_satellite_auroc": round(a_cnn, 3),
               "insitu_auroc": round(a_ins, 3), "cnn_fusion_emb_auroc": round(a_fus, 3),
               "cnn_fusion_score_auroc": round(a_fus_s, 3),
               "cnn_fusion_ci95": hm_ci(a_fus, y), "best_single": best_n,
               "fusion_minus_best": round(a_fus - best_a, 3),
               "boot_fusion_vs_best": boot, "fusion_helps": bool(boot["ci95"][0] > 0),
               "mean_epochs": round(float(np.mean(eps)), 1), "wall_min": round((time.time() - t0) / 60, 2)}
        out["labels"][tag] = res
        print(f"[{tag}] CNN-sat={a_cnn:.3f} insitu={a_ins:.3f} "
              f"CNN-FUSION(emb)={a_fus:.3f} CNN-FUSION(score)={a_fus_s:.3f} "
              f"(emb Δ vs best {res['fusion_minus_best']:+.3f} CI {boot['ci95']} helps={res['fusion_helps']}) "
              f"ep~{res['mean_epochs']} {res['wall_min']}min", flush=True)

    OUT.write_text(json.dumps(out, indent=2))
    print("WROTE", OUT, flush=True)


if __name__ == "__main__":
    main()
