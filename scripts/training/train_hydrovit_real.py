#!/usr/bin/env python3
"""HydroViT — REAL Sentinel-2 x in-situ water-quality regression, STATION holdout.

Honest replacement for the QUARANTINED R2=0.893 (water_temp) number, which was
computed on paired_wq_v5.npz with FABRICATED augmented targets (now absent).

This script uses the REAL co-located dataset already on disk:
    data/processed/satellite/drone_wq_pairs.npz
        images:       (12000, 4, 224, 224) float32  Sentinel-2 L2A reflectance
                      bands B02(blue) B03(green) B04(red) B08(NIR)
        targets:      (12000, 5) float32  [DO, pH, Turb, Temp, SpCond]
                      REAL same-day in-situ USGS/NWIS measurements
        metadata:     JSON list of {site_id, date, lat, lon} per tile (379 sites)

Protocol (no data leakage):
    - STATION (site_id) holdout: a USGS station appears in EXACTLY ONE of
      train / val / test. The model cannot memorize site identity, so it must
      extract genuine signal from the spectral imagery.
    - Per-target standardization fit on TRAIN ONLY.
    - NaN targets are masked out of the loss and of each per-parameter metric.
    - Baseline = predict the TRAIN mean for every test sample (R2 = 0 by
      construction on a perfectly representative split; we report the actual
      held-out R2 of that constant predictor too, which can be < 0).

Model: the genuine HydroViTBackbone (ViT-S/16, timm) adapted to 4 input bands,
       + a shared-trunk multi-head WQ regression head.

REAL data only. No augmentation-to-target, no label leakage, no circular
features. Whatever is measured is reported.

MIT License — Bryan Cheng, 2026
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sentinel.models.satellite_encoder.hydrovit_backbone import HydroViTBackbone

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "satellite" / "drone_wq_pairs.npz"
CKPT_DIR = PROJECT_ROOT / "checkpoints" / "satellite"
RESULTS_PATH = PROJECT_ROOT / "results" / "benchmarks" / "hydrovit_real.json"

TARGET_NAMES = ["DO", "pH", "Turb", "Temp", "SpCond"]
BANDS = ["B02", "B03", "B04", "B08"]
N_TARGETS = len(TARGET_NAMES)
N_BANDS = len(BANDS)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def station_holdout_split(site_ids: np.ndarray, seed: int = 42,
                          val_frac: float = 0.15, test_frac: float = 0.15):
    """Assign each tile to train/val/test by STATION (site_id).

    A station is placed wholly into one split, so no station is shared across
    splits. Returns boolean index masks (train, val, test).
    """
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(site_ids.tolist())))
    rng.shuffle(uniq)
    n = len(uniq)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    test_sites = set(uniq[:n_test].tolist())
    val_sites = set(uniq[n_test:n_test + n_val].tolist())
    train_sites = set(uniq[n_test + n_val:].tolist())

    tr = np.array([s in train_sites for s in site_ids])
    va = np.array([s in val_sites for s in site_ids])
    te = np.array([s in test_sites for s in site_ids])
    return tr, va, te, (len(train_sites), len(val_sites), len(test_sites))


class WQDataset(Dataset):
    """Real S2 tiles + standardized targets with a NaN-validity mask."""

    def __init__(self, images_mm, idx, y_std, mask, band_mean, band_std):
        self.images = images_mm
        self.idx = idx
        self.y = y_std            # standardized targets [N, 5]
        self.mask = mask          # 1.0 where target is valid (not NaN)
        self.band_mean = band_mean.reshape(N_BANDS, 1, 1)
        self.band_std = band_std.reshape(N_BANDS, 1, 1)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        gi = self.idx[i]
        img = np.asarray(self.images[gi]).astype(np.float32)  # [4,224,224]
        img = (img - self.band_mean) / self.band_std
        return (torch.from_numpy(img),
                torch.from_numpy(self.y[i]),
                torch.from_numpy(self.mask[i]))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class WQHead(nn.Module):
    """Shared trunk + per-parameter linear heads on the ViT CLS embedding."""

    def __init__(self, in_dim: int = 384, hidden: int = 256,
                 n_targets: int = N_TARGETS, dropout: float = 0.2):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
        )
        self.heads = nn.ModuleList(
            [nn.Linear(hidden, 1) for _ in range(n_targets)]
        )

    def forward(self, cls):
        h = self.trunk(cls)
        return torch.cat([head(h) for head in self.heads], dim=1)  # [B, n_targets]


class HydroViTWQ(nn.Module):
    def __init__(self, in_chans: int = N_BANDS, pretrained: bool = True,
                 dropout: float = 0.2):
        super().__init__()
        # Genuine HydroViT backbone (ViT-S/16) adapted to 4 bands.
        self.backbone = HydroViTBackbone(
            in_chans=in_chans, img_size=224, pretrained=pretrained,
            feature_layers=(12,),
        )
        self.head = WQHead(in_dim=self.backbone.embed_dim, dropout=dropout)

    def forward(self, x):
        cls, _ = self.backbone(x)
        return self.head(cls)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def masked_r2(y_true, y_pred, valid):
    """R2 on valid entries; baseline mean is y_true mean over valid set."""
    yt = y_true[valid]
    yp = y_pred[valid]
    if len(yt) < 2:
        return None, 0
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    if ss_tot <= 0:
        return None, len(yt)
    return 1.0 - ss_res / ss_tot, len(yt)


def baseline_r2_const(y_true_test, valid_test, train_mean):
    """R2 of predicting the TRAIN mean (constant) on the test set."""
    yt = y_true_test[valid_test]
    if len(yt) < 2:
        return None, 0
    ss_res = float(np.sum((yt - train_mean) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    if ss_tot <= 0:
        return None, len(yt)
    return 1.0 - ss_res / ss_tot, len(yt)


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------
def run(epochs=40, batch_size=64, lr_head=3e-4, lr_backbone=3e-5,
        freeze_epochs=8, seed=42, device="cuda:0", num_workers=8,
        max_train=None, quick=False):
    t0 = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    # ---- Load real data ----
    print(f"[load] {DATA_PATH}", flush=True)
    d = np.load(DATA_PATH, allow_pickle=True, mmap_mode="r")
    images = d["images"]  # memmap (12000,4,224,224)
    targets = np.asarray(d["targets"]).astype(np.float64)  # (12000,5)
    meta = json.loads(d["metadata"].item())
    site_ids = np.array([m["site_id"] for m in meta])
    n_total = images.shape[0]
    assert targets.shape == (n_total, N_TARGETS)

    tr_mask, va_mask, te_mask, (ns_tr, ns_va, ns_te) = station_holdout_split(
        site_ids, seed=seed
    )
    tr_idx = np.where(tr_mask)[0]
    va_idx = np.where(va_mask)[0]
    te_idx = np.where(te_mask)[0]

    # No station shared across splits (assert no leakage)
    s_tr = set(site_ids[tr_idx]); s_va = set(site_ids[va_idx]); s_te = set(site_ids[te_idx])
    assert not (s_tr & s_va) and not (s_tr & s_te) and not (s_va & s_te), "STATION LEAK"

    if quick:
        rng = np.random.default_rng(seed)
        tr_idx = rng.choice(tr_idx, min(400, len(tr_idx)), replace=False)
        va_idx = rng.choice(va_idx, min(200, len(va_idx)), replace=False)
        te_idx = rng.choice(te_idx, min(200, len(te_idx)), replace=False)
    elif max_train is not None and len(tr_idx) > max_train:
        rng = np.random.default_rng(seed)
        tr_idx = rng.choice(tr_idx, max_train, replace=False)

    print(f"[split] stations train/val/test = {ns_tr}/{ns_va}/{ns_te}", flush=True)
    print(f"[split] tiles    train/val/test = {len(tr_idx)}/{len(va_idx)}/{len(te_idx)}", flush=True)

    # ---- Standardization on TRAIN only (targets + bands) ----
    y_tr_raw = targets[tr_idx]
    t_mean = np.nanmean(y_tr_raw, axis=0)
    t_std = np.nanstd(y_tr_raw, axis=0)
    t_std[t_std < 1e-8] = 1.0

    def std_targets(idx):
        y = targets[idx]
        mask = (~np.isnan(y)).astype(np.float32)
        ystd = np.where(np.isnan(y), 0.0, (y - t_mean) / t_std).astype(np.float32)
        return ystd, mask

    y_tr, m_tr = std_targets(tr_idx)
    y_va, m_va = std_targets(va_idx)
    y_te, m_te = std_targets(te_idx)

    # Band normalization stats from a TRAIN sample (memmap-friendly)
    rng = np.random.default_rng(seed)
    samp = tr_idx[rng.permutation(len(tr_idx))[:min(800, len(tr_idx))]]
    samp_imgs = np.asarray(images[np.sort(samp)]).astype(np.float32)
    band_mean = samp_imgs.mean(axis=(0, 2, 3))
    band_std = samp_imgs.std(axis=(0, 2, 3))
    band_std[band_std < 1e-8] = 1.0
    del samp_imgs

    ds_tr = WQDataset(images, tr_idx, y_tr, m_tr, band_mean, band_std)
    ds_va = WQDataset(images, va_idx, y_va, m_va, band_mean, band_std)
    ds_te = WQDataset(images, te_idx, y_te, m_te, band_mean, band_std)

    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,
                       num_workers=num_workers, pin_memory=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=True)
    dl_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=True)

    # ---- Model ----
    model = HydroViTWQ(in_chans=N_BANDS, pretrained=True).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] HydroViTWQ params = {n_params:,}", flush=True)

    opt = torch.optim.AdamW([
        {"params": model.head.parameters(), "lr": lr_head},
        {"params": model.backbone.parameters(), "lr": lr_backbone},
    ], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler()

    def masked_mse(pred, y, mask):
        diff = (pred - y) ** 2 * mask
        denom = mask.sum().clamp(min=1.0)
        return diff.sum() / denom

    def set_backbone_grad(flag):
        for p in model.backbone.parameters():
            p.requires_grad = flag

    @torch.no_grad()
    def evaluate(dl):
        model.eval()
        preds, ys, masks = [], [], []
        for img, y, mk in dl:
            img = img.to(dev, non_blocking=True)
            with torch.cuda.amp.autocast():
                p = model(img)
            preds.append(p.float().cpu().numpy())
            ys.append(y.numpy()); masks.append(mk.numpy())
        return np.concatenate(preds), np.concatenate(ys), np.concatenate(masks)

    best_val = -1e9
    best_state = None
    history = []
    for ep in range(epochs):
        if ep < freeze_epochs:
            set_backbone_grad(False)
        elif ep == freeze_epochs:
            set_backbone_grad(True)
            print(f"[phase2] unfreezing backbone at epoch {ep}", flush=True)

        model.train()
        run_loss = 0.0; nb = 0
        for img, y, mk in dl_tr:
            img = img.to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True); mk = mk.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                pred = model(img)
                loss = masked_mse(pred, y, mk)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            run_loss += loss.item(); nb += 1
        sched.step()

        # validation: mean R2 across targets in standardized space
        vp, vy, vm = evaluate(dl_va)
        v_r2s = []
        for j in range(N_TARGETS):
            valid = vm[:, j] > 0.5
            r2, n = masked_r2(vy[:, j], vp[:, j], valid)
            if r2 is not None:
                v_r2s.append(r2)
        v_mean = float(np.mean(v_r2s)) if v_r2s else -1e9
        history.append({"epoch": ep, "train_loss": run_loss / max(nb, 1),
                        "val_mean_r2": v_mean})
        print(f"[ep {ep:02d}] train_loss={run_loss/max(nb,1):.4f} "
              f"val_mean_r2={v_mean:.4f}", flush=True)
        if v_mean > best_val:
            best_val = v_mean
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # ---- Test with best model ----
    if best_state is not None:
        model.load_state_dict(best_state)
    tp, ty, tm = evaluate(dl_te)

    # De-standardize predictions and targets to report in physical units R2
    # (R2 is scale-invariant per parameter, but compute in real units for clarity)
    tp_real = tp * t_std + t_mean
    ty_real = ty * t_std + t_mean

    per_param = {}
    per_param_baseline = {}
    r2_list = []
    for j, name in enumerate(TARGET_NAMES):
        valid = tm[:, j] > 0.5
        r2, n = masked_r2(ty_real[:, j], tp_real[:, j], valid)
        # baseline: predict TRAIN mean (t_mean[j]) for all test samples
        b_r2, _ = baseline_r2_const(ty_real[:, j], valid, t_mean[j])
        per_param[name] = None if r2 is None else round(r2, 4)
        per_param_baseline[name] = None if b_r2 is None else round(b_r2, 4)
        if r2 is not None:
            r2_list.append(r2)
        # MAE in physical units
        yt = ty_real[valid, j]; yp = tp_real[valid, j]
        mae = float(np.mean(np.abs(yt - yp))) if len(yt) else None
        per_param[name] = {
            "r2": None if r2 is None else round(r2, 4),
            "mae": None if mae is None else round(mae, 4),
            "n_test_valid": int(n),
            "baseline_mean_r2": None if b_r2 is None else round(b_r2, 4),
        }
        print(f"[test] {name:7s} R2={per_param[name]['r2']} "
              f"MAE={per_param[name]['mae']} "
              f"baseline_R2={per_param[name]['baseline_mean_r2']} "
              f"n={n}", flush=True)

    mean_r2 = round(float(np.mean(r2_list)), 4) if r2_list else None
    elapsed = time.time() - t0

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / "hydrovit_wq_real.pt"
    torch.save({"model": model.state_dict(),
                "band_mean": band_mean, "band_std": band_std,
                "t_mean": t_mean, "t_std": t_std}, ckpt_path)

    result = {
        "_PROVENANCE": {
            "purpose": "Honest replacement for the QUARANTINED hydrovit_holdout "
                       "water_temp R2=0.893 (computed on FABRICATED augmented "
                       "targets in paired_wq_v5.npz, now absent).",
            "data": str(DATA_PATH.relative_to(PROJECT_ROOT)),
            "data_description": "REAL Sentinel-2 L2A tiles (B02,B03,B04,B08) "
                                "paired with REAL same-day in-situ USGS/NWIS "
                                "water-quality measurements.",
            "real_data_only": True,
            "augmentation_to_target": False,
            "label_leakage": False,
            "split": "STATION (site_id) holdout — no station shared across "
                     "train/val/test; model cannot memorize site identity.",
            "standardization": "targets + bands standardized on TRAIN only",
            "nan_handling": "NaN targets masked from loss and per-param metrics",
        },
        "model": "HydroViT (ViT-S/16, timm) + multi-head WQ regressor",
        "backbone": "HydroViTBackbone, in_chans=4, ImageNet-pretrained adapted",
        "n_params": int(n_params),
        "n_stations": {"train": ns_tr, "val": ns_va, "test": ns_te},
        "n_tiles": {"train": int(len(tr_idx)), "val": int(len(va_idx)),
                    "test": int(len(te_idx))},
        "target_names": TARGET_NAMES,
        "bands": BANDS,
        "epochs": epochs,
        "batch_size": batch_size,
        "best_val_mean_r2_standardized": round(float(best_val), 4),
        "per_param": per_param,
        "mean_r2_test": mean_r2,
        "water_temp_r2_test": per_param["Temp"]["r2"],
        "water_temp_baseline_mean_r2": per_param["Temp"]["baseline_mean_r2"],
        "quarantined_for_comparison": {
            "water_temp_r2": 0.8927,
            "mean_r2": 0.6524,
            "note": "FABRICATED augmented targets — NOT comparable; for context only.",
        },
        "checkpoint": str(ckpt_path.relative_to(PROJECT_ROOT)),
        "elapsed_seconds": round(elapsed, 1),
        "quick_mode": quick,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[done] wrote {RESULTS_PATH} in {elapsed:.0f}s", flush=True)
    print(json.dumps({k: result[k] for k in
                      ["mean_r2_test", "water_temp_r2_test",
                       "water_temp_baseline_mean_r2"]}, indent=2), flush=True)
    return result


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--freeze_epochs", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--max_train", type=int, default=None)
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    run(epochs=args.epochs, batch_size=args.batch_size,
        freeze_epochs=args.freeze_epochs, seed=args.seed, device=args.device,
        num_workers=args.num_workers, max_train=args.max_train, quick=args.quick)


if __name__ == "__main__":
    main()
