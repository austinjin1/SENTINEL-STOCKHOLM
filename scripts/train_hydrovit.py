#!/usr/bin/env python3
"""HydroViT training: MAE pretrain + WQ parameter regression.

MIT License — Bryan Cheng, 2026
"""

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from sentinel.models.satellite_encoder.model import SatelliteEncoder
from sentinel.models.satellite_encoder.parameter_head import WaterQualityHead
from sentinel.utils.logging import get_logger

logger = get_logger(__name__)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CKPT = Path("checkpoints/satellite")
CKPT.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path("data/processed/satellite")


# Target normalization: log-transform + z-score for stable regression
TARGET_LOG_PARAMS = {0,1,3,4,5,6,8,9,12,14}  # params that are log-normal (chla, turb, etc.)

class SatelliteDataset(Dataset):
    def __init__(self, data_dir, max_samples=None):
        self.files = sorted(Path(data_dir).glob("*.npz"))
        if max_samples:
            self.files = self.files[:max_samples]
        # Compute target stats for normalization
        all_targets = []
        for f in self.files[:200]:
            d = np.load(f, allow_pickle=True)
            t = d["targets"].astype(np.float32)
            all_targets.append(t)
        all_targets = np.stack(all_targets)
        self.target_mean = np.nanmean(all_targets, axis=0)
        self.target_std = np.nanstd(all_targets, axis=0)
        self.target_std[self.target_std < 1e-6] = 1.0

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx], allow_pickle=True)
        image = torch.tensor(data["image"].astype(np.float32))  # (13, 224, 224)
        targets = data["targets"].astype(np.float32)  # (16,) with NaN
        # Normalize targets to ~N(0,1)
        targets_norm = (targets - self.target_mean) / self.target_std
        return {"image": image, "targets": torch.tensor(targets_norm)}


def train_mae_pretrain(model, train_dl, val_dl, epochs=30, lr=1e-4):
    """Phase 1: MAE pretraining on water pixel patches."""
    logger.info("=" * 60)
    logger.info("PHASE 1: MAE Pretraining")
    logger.info("=" * 60)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    best_val = float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss, n = 0, 0
        for batch in train_dl:
            image = batch["image"].to(DEVICE)
            out = model.forward_mae(image, mask_ratio=0.75)
            loss = out["mae_loss"]  # Use only MAE loss (physics loss expects image, not patches)

            if torch.isnan(loss):
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n += 1

        # Validation
        model.eval()
        val_loss, vn = 0, 0
        with torch.no_grad():
            for batch in val_dl:
                image = batch["image"].to(DEVICE)
                out = model.forward_mae(image, mask_ratio=0.75)
                if not torch.isnan(out["mae_loss"]):
                    val_loss += out["mae_loss"].item()
                    vn += 1

        tl = total_loss / max(n, 1)
        vl = val_loss / max(vn, 1)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f"MAE Ep {epoch+1:3d}/{epochs} | Train: {tl:.4f} | Val: {vl:.4f} | nb={n}")

        if vn > 0 and vl < best_val:
            best_val = vl
            torch.save(model.state_dict(), CKPT / "hydrovit_mae_best.pt")

        # LR decay
        for pg in optimizer.param_groups:
            pg["lr"] *= 0.97


class WQRegressionHead(nn.Module):
    """Simple regression head for 16 WQ parameters."""
    def __init__(self, embed_dim=256, n_params=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, n_params),
        )
    def forward(self, x):
        return self.net(x)


def train_wq_regression(model, train_dl, val_dl, epochs=50, lr=5e-5):
    """Phase 2: WQ regression with simple head on frozen backbone embedding."""
    logger.info("=" * 60)
    logger.info("PHASE 2: WQ Regression (frozen backbone + simple head)")
    logger.info("=" * 60)

    # Freeze entire model, extract embeddings
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Precompute embeddings
    def extract_embeddings(dl):
        embs, tgts = [], []
        with torch.no_grad():
            for batch in dl:
                img = batch["image"].to(DEVICE)
                targets = batch["targets"]  # (B, 16) with NaN
                out = model(img)
                emb = out["embedding"]  # (B, 256)
                for j in range(emb.shape[0]):
                    if not torch.isnan(emb[j]).any():
                        embs.append(emb[j].cpu())
                        tgts.append(targets[j])
        if embs:
            return torch.stack(embs), torch.stack(tgts)
        return torch.zeros(0, 256), torch.zeros(0, 16)

    logger.info("Extracting embeddings...")
    tr_e, tr_t = extract_embeddings(train_dl)
    va_e, va_t = extract_embeddings(val_dl)
    logger.info(f"Train: {len(tr_e)}, Val: {len(va_e)}")

    if len(tr_e) == 0:
        logger.error("No valid embeddings!")
        return -float("inf")

    # Train simple regression head
    head = WQRegressionHead(embed_dim=tr_e.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
    best_r2 = -float("inf")

    for epoch in range(epochs):
        head.train()
        perm = torch.randperm(len(tr_e))
        e_s = tr_e[perm].to(DEVICE)
        t_s = tr_t[perm].to(DEVICE)
        total_loss, nb = 0, 0

        for i in range(0, len(e_s), 32):
            e = e_s[i:i+32]
            t = t_s[i:i+32]
            pred = head(e)
            valid = ~torch.isnan(t)
            if valid.sum() == 0:
                continue
            loss = ((pred - t) ** 2)[valid].mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            nb += 1

        # Compute R²
        head.eval()
        with torch.no_grad():
            tr_pred = head(tr_e.to(DEVICE))
            va_pred = head(va_e.to(DEVICE)) if len(va_e) > 0 else None

        # Train R²
        valid_tr = ~torch.isnan(tr_t.to(DEVICE))
        if valid_tr.sum() > 0:
            p_tr = tr_pred[valid_tr]
            t_tr = tr_t.to(DEVICE)[valid_tr]
            ss_res = ((p_tr - t_tr) ** 2).sum()
            ss_tot = ((t_tr - t_tr.mean()) ** 2).sum()
            train_r2 = (1 - ss_res / ss_tot.clamp(min=1e-8)).item()
        else:
            train_r2 = 0.0

        # Val R²
        val_r2 = 0.0
        if va_pred is not None and len(va_e) > 0:
            valid_va = ~torch.isnan(va_t.to(DEVICE))
            if valid_va.sum() > 0:
                p_va = va_pred[valid_va]
                t_va = va_t.to(DEVICE)[valid_va]
                ss_res = ((p_va - t_va) ** 2).sum()
                ss_tot = ((t_va - t_va.mean()) ** 2).sum()
                val_r2 = (1 - ss_res / ss_tot.clamp(min=1e-8)).item()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"WQ Ep {epoch+1:3d}/{epochs} | Loss: {total_loss/max(nb,1):.4f} | Train R²: {train_r2:.4f} | Val R²: {val_r2:.4f}")

        if val_r2 > best_r2:
            best_r2 = val_r2
            torch.save(head.state_dict(), CKPT / "wq_head_best.pt")

    # Unfreeze model for next steps
    for p in model.parameters():
        p.requires_grad = True

    return best_r2


def main():
    t0 = time.time()

    ds = SatelliteDataset(DATA_DIR)
    n = len(ds)
    n_tr = int(0.7 * n)
    n_va = int(0.15 * n)
    n_te = n - n_tr - n_va
    tr, va, te = random_split(ds, [n_tr, n_va, n_te], generator=torch.Generator().manual_seed(42))

    tr_dl = DataLoader(tr, batch_size=4, shuffle=True, num_workers=0)
    va_dl = DataLoader(va, batch_size=4, num_workers=0)
    te_dl = DataLoader(te, batch_size=4, num_workers=0)

    logger.info(f"Data: {n_tr}/{n_va}/{n_te} tiles ({IMG_SIZE}x{IMG_SIZE}, {N_BANDS} bands)")

    model = SatelliteEncoder().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"HydroViT: {n_params:,} parameters")

    # Phase 1: MAE pretrain
    train_mae_pretrain(model, tr_dl, va_dl, epochs=20, lr=1e-4)

    # Reload best MAE checkpoint before Phase 2
    best_mae_path = CKPT / "hydrovit_mae_best.pt"
    if best_mae_path.exists():
        logger.info(f"Reloading best MAE checkpoint from {best_mae_path}")
        state = torch.load(best_mae_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state, strict=False)

    # Phase 2: Fine-tune the model's built-in WQ head (not a separate head)
    logger.info("=" * 60)
    logger.info("PHASE 2: WQ Fine-tuning (frozen backbone + built-in WQ head)")
    logger.info("=" * 60)

    # Freeze backbone, train only water_quality_head
    for name, p in model.named_parameters():
        p.requires_grad = "water_quality_head" in name
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable params: {trainable:,} (water_quality_head only)")

    optimizer2 = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )
    best_val_loss = float("inf")
    best_r2 = -float("inf")

    for epoch in range(40):
        model.train()
        total_loss, nb = 0, 0
        for batch in tr_dl:
            image = batch["image"].to(DEVICE)
            targets = batch["targets"].to(DEVICE)
            out = model(image)
            wq = out["water_quality_params"]
            unc = out["param_uncertainty"]
            valid = ~torch.isnan(targets)
            if valid.sum() == 0:
                continue
            loss = WaterQualityHead.gaussian_nll_loss(wq, unc, targets)
            if torch.isnan(loss):
                optimizer2.zero_grad()
                continue
            optimizer2.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer2.step()
            total_loss += loss.item()
            nb += 1

        # Validation R²
        model.eval()
        val_preds = {j: [] for j in range(16)}
        val_tgts = {j: [] for j in range(16)}
        with torch.no_grad():
            for batch in va_dl:
                image = batch["image"].to(DEVICE)
                targets = batch["targets"].to(DEVICE)
                out = model(image)
                wq = out["water_quality_params"]
                valid = ~torch.isnan(targets)
                for j in range(16):
                    mask = valid[:, j]
                    if mask.sum() > 0:
                        val_preds[j].append(wq[:, j][mask].cpu())
                        val_tgts[j].append(targets[:, j][mask].cpu())

        val_r2s = []
        for j in range(16):
            if val_preds[j]:
                p = torch.cat(val_preds[j])
                t = torch.cat(val_tgts[j])
                ss_res = ((p - t) ** 2).sum()
                ss_tot = ((t - t.mean()) ** 2).sum()
                if ss_tot > 1e-8:
                    val_r2s.append((1 - ss_res / ss_tot).item())
        val_r2 = np.mean(val_r2s) if val_r2s else -1.0

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f"WQ Ep {epoch+1:3d}/40 | Loss: {total_loss/max(nb,1):.4f} | Val R²: {val_r2:.4f} | nb={nb}")

        if val_r2 > best_r2:
            best_r2 = val_r2
            torch.save(model.state_dict(), CKPT / "hydrovit_wq_best.pt")

    # Unfreeze all
    for p in model.parameters():
        p.requires_grad = True

    # Reload best WQ checkpoint for test
    wq_best_path = CKPT / "hydrovit_wq_best.pt"
    if wq_best_path.exists():
        model.load_state_dict(torch.load(wq_best_path, map_location=DEVICE, weights_only=True), strict=False)

    # Test evaluation
    logger.info("=" * 60)
    logger.info("TEST EVALUATION")
    logger.info("=" * 60)

    model.eval()
    per_param_preds = {i: [] for i in range(16)}
    per_param_tgts = {i: [] for i in range(16)}

    with torch.no_grad():
        for batch in te_dl:
            image = batch["image"].to(DEVICE)
            targets = batch["targets"].to(DEVICE)
            out = model(image)
            wq = out["water_quality_params"]
            valid = ~torch.isnan(targets)
            for j in range(16):
                mask = valid[:, j]
                if mask.sum() > 0:
                    per_param_preds[j].append(wq[:, j][mask].cpu())
                    per_param_tgts[j].append(targets[:, j][mask].cpu())

    PARAM_NAMES = [
        "chla", "turbidity", "secchi", "cdom", "tss", "tn", "tp", "do",
        "ammonia", "nitrate", "ph", "temp", "phycocyanin", "oil_prob", "acdom", "pai",
    ]

    r2_scores = {}
    for j in range(16):
        if per_param_preds[j]:
            p = torch.cat(per_param_preds[j])
            t = torch.cat(per_param_tgts[j])
            ss_res = ((p - t) ** 2).sum()
            ss_tot = ((t - t.mean()) ** 2).sum()
            r2 = (1 - ss_res / ss_tot.clamp(min=1e-8)).item()
        else:
            r2 = 0.0
        r2_scores[PARAM_NAMES[j]] = r2
        logger.info(f"  {PARAM_NAMES[j]:>15}: R² = {r2:.4f}")

    mean_r2 = np.mean(list(r2_scores.values()))
    logger.info(f"\n  Mean R²: {mean_r2:.4f}")

    if mean_r2 > 0.55:
        logger.info("*** HARD THRESHOLD MET ***")
    elif mean_r2 > 0.30:
        logger.info("ACCEPTABLE")
    else:
        logger.info(f"BELOW THRESHOLD ({mean_r2:.4f})")

    elapsed = time.time() - t0
    results = {
        "mean_r2": mean_r2,
        "best_val_r2": best_r2,
        "per_param_r2": r2_scores,
        "elapsed": elapsed,
        "n_train": n_tr,
        "n_test": n_te,
    }
    with open(CKPT / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Time: {elapsed/60:.1f}m")
    logger.info("DONE")


IMG_SIZE = 224
N_BANDS = 13

if __name__ == "__main__":
    main()
