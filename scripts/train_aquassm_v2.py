#!/usr/bin/env python3
"""AquaSSM training v2: MPP pretrain (no physics) + anomaly fine-tune.

Iteration 1 fix: Removed physics constraint loss which caused training
instability after epoch 25. Pure MPP pretraining was stable (0.527 → 0.077).

MIT License — Bryan Cheng, 2026
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

from sentinel.models.sensor_encoder import SensorEncoder
from sentinel.utils.logging import get_logger

logger = get_logger(__name__)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = Path("checkpoints/sensor")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


class SensorDataset(Dataset):
    def __init__(self, data_dir, max_len=512):
        self.files = sorted(Path(data_dir).glob("*.npz"))
        self.max_len = max_len
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        d = np.load(self.files[idx])
        T = min(len(d["values"]), self.max_len)
        v = torch.tensor(d["values"][:T].astype(np.float32))
        dt = torch.tensor(d["delta_ts"][:T].astype(np.float32))
        lbl = d["labels"][:T].astype(np.int64) if "labels" in d else np.zeros(T, dtype=np.int64)
        ha = int((lbl > 0).any())
        return {"values": v, "delta_ts": dt, "has_anomaly": ha}


def collate(batch):
    ml = max(b["values"].shape[0] for b in batch)
    B = len(batch)
    v = torch.zeros(B, ml, 6); dt = torch.zeros(B, ml)
    ha = torch.tensor([b["has_anomaly"] for b in batch])
    for i, b in enumerate(batch):
        T = b["values"].shape[0]; v[i,:T] = b["values"]; dt[i,:T] = b["delta_ts"]
    return {"values": v, "delta_ts": dt, "has_anomaly": ha}


class AnomalyHead(nn.Module):
    def __init__(self, d=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, 1))
    def forward(self, x): return self.net(x).squeeze(-1)


def main():
    t0 = time.time()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Data
    ds_list = []
    for d in ["data/processed/sensor/pretrain", "data/processed/sensor/synthetic"]:
        p = Path(d)
        if p.exists():
            ds = SensorDataset(p)
            if len(ds) > 0: ds_list.append(ds); logger.info(f"{d}: {len(ds)}")
    full = ConcatDataset(ds_list); n = len(full)
    n_tr, n_va = int(.7*n), int(.15*n); n_te = n - n_tr - n_va
    tr, va, te = random_split(full, [n_tr, n_va, n_te], generator=torch.Generator().manual_seed(42))
    tr_dl = DataLoader(tr, batch_size=8, shuffle=True, collate_fn=collate, num_workers=0)
    va_dl = DataLoader(va, batch_size=8, collate_fn=collate, num_workers=0)
    te_dl = DataLoader(te, batch_size=8, collate_fn=collate, num_workers=0)
    logger.info(f"Split: {n_tr}/{n_va}/{n_te}")

    model = SensorEncoder().to(DEVICE)
    logger.info(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    # PHASE 1: MPP Pretrain (NO physics constraints)
    logger.info("=" * 60)
    logger.info("PHASE 1: MPP Pretrain (pure, no physics)")
    logger.info("=" * 60)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)
    best_val = float("inf")

    for ep in range(30):
        model.train(); total, nb = 0, 0
        for batch in tr_dl:
            v = batch["values"].to(DEVICE); dt = batch["delta_ts"].to(DEVICE)
            out = model.forward_pretrain(x=v, delta_ts=dt)
            loss = out["loss"]
            if torch.isnan(loss): opt.zero_grad(); continue
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); total += loss.item(); nb += 1
        sched.step()

        model.eval(); vl, vn = 0, 0
        with torch.no_grad():
            for batch in va_dl:
                v = batch["values"].to(DEVICE); dt = batch["delta_ts"].to(DEVICE)
                out = model.forward_pretrain(x=v, delta_ts=dt)
                if not torch.isnan(out["loss"]): vl += out["loss"].item(); vn += 1

        tl = total/max(nb,1); vloss = vl/max(vn,1)
        if (ep+1) % 5 == 0 or ep == 0:
            logger.info(f"Ep {ep+1:3d}/30 | Train: {tl:.4f} | Val: {vloss:.4f} | LR: {sched.get_last_lr()[0]:.2e}")
        if vn > 0 and vloss < best_val:
            best_val = vloss
            torch.save(model.state_dict(), CHECKPOINT_DIR / "aquassm_v2_pretrained.pt")

    torch.save(model.state_dict(), CHECKPOINT_DIR / "aquassm_v2_pretrained_final.pt")

    # PHASE 2: Anomaly Fine-tune
    logger.info("=" * 60)
    logger.info("PHASE 2: Anomaly Fine-tune")
    logger.info("=" * 60)
    head = AnomalyHead().to(DEVICE)
    opt2 = torch.optim.AdamW([
        {"params": model.parameters(), "lr": 1e-5},
        {"params": head.parameters(), "lr": 1e-3},
    ], weight_decay=0.01)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=30)
    best_auroc = 0

    for ep in range(30):
        model.train(); head.train()
        total, nb = 0, 0; preds, labels = [], []
        for batch in tr_dl:
            v = batch["values"].to(DEVICE); dt = batch["delta_ts"].to(DEVICE)
            ha = batch["has_anomaly"].float().to(DEVICE)
            out = model(v, dt, compute_anomaly=False)
            logit = head(out["embedding"])
            loss = nn.functional.binary_cross_entropy_with_logits(logit, ha)
            if torch.isnan(loss): opt2.zero_grad(); continue
            opt2.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), 1.0)
            opt2.step(); total += loss.item(); nb += 1
            preds.extend(torch.sigmoid(logit).detach().cpu().numpy())
            labels.extend(ha.cpu().numpy())
        sched2.step()

        try: tr_auc = roc_auc_score(labels, preds)
        except: tr_auc = 0.5

        model.eval(); head.eval()
        vp, vl = [], []
        with torch.no_grad():
            for batch in va_dl:
                v = batch["values"].to(DEVICE); dt = batch["delta_ts"].to(DEVICE)
                ha = batch["has_anomaly"].float()
                out = model(v, dt, compute_anomaly=False)
                vp.extend(torch.sigmoid(head(out["embedding"])).cpu().numpy())
                vl.extend(ha.numpy())
        try: va_auc = roc_auc_score(vl, vp)
        except: va_auc = 0.5

        if (ep+1) % 5 == 0 or ep == 0:
            logger.info(f"Ep {ep+1:3d}/30 | Loss: {total/max(nb,1):.4f} | Train AUC: {tr_auc:.4f} | Val AUC: {va_auc:.4f}")
        if va_auc > best_auroc:
            best_auroc = va_auc
            torch.save({"model": model.state_dict(), "head": head.state_dict()},
                       CHECKPOINT_DIR / "aquassm_v2_anomaly_best.pt")

    # TEST
    logger.info("=" * 60)
    logger.info("TEST EVALUATION")
    logger.info("=" * 60)
    model.eval(); head.eval()
    tp, tl = [], []
    with torch.no_grad():
        for batch in te_dl:
            v = batch["values"].to(DEVICE); dt = batch["delta_ts"].to(DEVICE)
            ha = batch["has_anomaly"].float()
            out = model(v, dt, compute_anomaly=False)
            tp.extend(torch.sigmoid(head(out["embedding"])).cpu().numpy())
            tl.extend(ha.numpy())
    tp, tl = np.array(tp), np.array(tl)
    try: auroc = roc_auc_score(tl, tp); auprc = average_precision_score(tl, tp)
    except: auroc = auprc = 0.5
    f1 = f1_score(tl, (tp>0.5).astype(int), zero_division=0)

    logger.info(f"AUROC: {auroc:.4f} | AUPRC: {auprc:.4f} | F1: {f1:.4f}")
    logger.info(f"N={len(tl)}, Pos={int(tl.sum())}")

    if auroc > 0.85: logger.info(f"HARD THRESHOLD MET: {auroc:.4f} > 0.85")
    elif auroc > 0.70: logger.info(f"ACCEPTABLE: {auroc:.4f} > 0.70")
    else: logger.info(f"BELOW THRESHOLD: {auroc:.4f} < 0.70")

    elapsed = time.time() - t0
    with open(CHECKPOINT_DIR / f"run_v2_{ts}.json", "w") as f:
        json.dump({"auroc": auroc, "auprc": auprc, "f1": f1, "best_val_auroc": best_auroc,
                    "elapsed": elapsed, "n_test": len(tl)}, f, indent=2)
    logger.info(f"Time: {elapsed/60:.1f}m")


if __name__ == "__main__":
    main()
