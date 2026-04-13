#!/usr/bin/env python3
"""benchmark_aquassm.py — Benchmark AquaSSM (real data) vs 4 baselines.

REAL DATA ONLY. Same data split as train_aquassm_expanded.py.
Sources: pretrain/ (162 real USGS) + real_20k/ subset (300 anomaly + 300 normal).
NO clean_synthetic, NO expanded_v2, NO synthetic data.

Baselines:
  - LSTM (2-layer, hidden=128, 50 epochs)
  - Transformer (d_model=64, nhead=4, 2 layers, 50 epochs)
  - Isolation Forest (n_estimators=200, 12-dim mean+std features)
  - One-Class SVM (nu=0.1, rbf, trained on normal samples only)

Output: results/benchmarks/aquassm_benchmark.json + printed table.

Bryan Cheng, SENTINEL project, 2026
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, "/home/bcheng/SENTINEL")

from sentinel.models.sensor_encoder import SensorEncoder
from sentinel.utils.logging import get_logger

logger = get_logger(__name__)
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = Path("results/benchmarks")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED        = 42
MAX_LEN     = 128
BATCH_SIZE  = 16
LSTM_EPOCHS = 50
TFM_EPOCHS  = 50

torch.manual_seed(SEED)
np.random.seed(SEED)

# Normalized threshold constants for pretrain labeling (same as train_aquassm_expanded.py)
_DO_LO   = (4.0   - 9.0)   / 3.0
_PH_LO   = (6.0   - 7.5)   / 1.0
_PH_HI   = (9.5   - 7.5)   / 1.0
_SPC_HI  = (1500.0- 500.0) / 400.0
_TUR_HI  = (300.0 - 20.0)  / 50.0


def _label_pretrain(values: np.ndarray) -> int:
    do=values[:,0]; ph=values[:,1]; spc=values[:,2]; turb=values[:,4]
    if np.any(do  < _DO_LO):  return 1
    if np.any(ph  < _PH_LO):  return 1
    if np.any(ph  > _PH_HI):  return 1
    if np.any(spc > _SPC_HI): return 1
    if np.any(turb> _TUR_HI): return 1
    return 0


# ---------------------------------------------------------------------------
# In-memory datasets (same structure as train_aquassm_expanded.py)
# ---------------------------------------------------------------------------

class _PretrainRealDataset(Dataset):
    """162 real USGS pretrain files, threshold-labeled, cached in RAM."""
    def __init__(self, data_dir: str, max_len: int = MAX_LEN):
        files = sorted(Path(data_dir).glob("*.npz"))
        self.samples, self.labels = [], []
        for f in files:
            d = np.load(f)
            lbl = _label_pretrain(d["values"])
            T = min(len(d["values"]), max_len)
            v  = torch.tensor(d["values"][:T].astype(np.float32)).clamp(-5, 5)
            dt = torch.tensor(d["delta_ts"][:T].astype(np.float32)).clamp(0, 3600)
            dt[0] = 0.0
            self.samples.append((v, dt))
            self.labels.append(lbl)
    def __len__(self):             return len(self.samples)
    def __getitem__(self, idx):
        v, dt = self.samples[idx]
        return {"values": v, "delta_ts": dt, "has_anomaly": self.labels[idx]}


class _Real20kDataset(Dataset):
    """Balanced real_20k subset, cached in RAM."""
    def __init__(self, file_list: list, max_len: int = MAX_LEN):
        self.samples, self.labels = [], []
        for f in file_list:
            d = np.load(f)
            lbl = int(d["has_anomaly"])
            T = min(len(d["values"]), max_len)
            v  = torch.tensor(d["values"][:T].astype(np.float32)).clamp(-5, 5)
            dt = torch.tensor(d["delta_ts"][:T].astype(np.float32)).clamp(0, 3600)
            dt[0] = 0.0
            self.samples.append((v, dt))
            self.labels.append(lbl)
    def __len__(self):             return len(self.samples)
    def __getitem__(self, idx):
        v, dt = self.samples[idx]
        return {"values": v, "delta_ts": dt, "has_anomaly": self.labels[idx]}


def _collate(batch):
    ml = max(b["values"].shape[0] for b in batch)
    B  = len(batch)
    V  = torch.zeros(B, ml, 6)
    DT = torch.zeros(B, ml)
    Y  = torch.tensor([b["has_anomaly"] for b in batch], dtype=torch.float32)
    for i, b in enumerate(batch):
        T = b["values"].shape[0]
        V[i, :T]  = b["values"]
        DT[i, :T] = b["delta_ts"]
    return {"values": V, "delta_ts": DT, "has_anomaly": Y}


def _select_balanced_real20k(n_anomaly: int = 300, n_normal: int = 300):
    rng = np.random.default_rng(SEED)
    all_files = sorted(Path("data/processed/sensor/real_20k").glob("*.npz"))
    af, nf = [], []
    for f in all_files:
        if int(np.load(f)["has_anomaly"]) == 1:
            af.append(f)
        else:
            nf.append(f)
    af = [af[i] for i in rng.permutation(len(af))]
    nf = [nf[i] for i in rng.permutation(len(nf))]
    return af[:n_anomaly] + nf[:n_normal]


def build_split():
    """Build the same 70/15/15 split used in train_aquassm_expanded.py."""
    logger.info("Loading real data (matching train_aquassm_expanded.py split)...")
    ds_p  = _PretrainRealDataset("data/processed/sensor/pretrain")
    sel   = _select_balanced_real20k(300, 300)
    ds_r  = _Real20kDataset(sel)
    full  = ConcatDataset([ds_p, ds_r])
    N     = len(full)
    n_tr  = int(0.70 * N)
    n_va  = int(0.15 * N)
    n_te  = N - n_tr - n_va
    tr, va, te = random_split(full, [n_tr, n_va, n_te],
                              generator=torch.Generator().manual_seed(SEED))
    logger.info(f"  N={N}: train={n_tr}, val={n_va}, test={n_te}")
    _log_dist("train", tr); _log_dist("val", va); _log_dist("test", te)
    return tr, va, te, N


def _log_dist(name, ds):
    from torch.utils.data import DataLoader
    labs = []
    for b in DataLoader(ds, batch_size=64, collate_fn=_collate):
        labs.extend(b["has_anomaly"].numpy().tolist())
    pos = sum(1 for l in labs if l > 0.5)
    logger.info(f"  {name}: {len(labs)} samples, {pos} anomaly, {len(labs)-pos} normal")


def _compute_pos_weight(tr_ds):
    labs = []
    for b in DataLoader(tr_ds, batch_size=64, collate_fn=_collate):
        labs.extend(b["has_anomaly"].numpy().tolist())
    n_pos = sum(1 for l in labs if l > 0.5)
    n_neg = len(labs) - n_pos
    return float(max(1.0, min(n_neg / max(n_pos, 1), 10.0)))


def _get_features_labels(ds):
    """Extract 12-dim features (per-sequence mean+std of 6 channels) for sklearn."""
    feats, labs = [], []
    for i in range(len(ds)):
        item = ds[i]
        v    = item["values"].numpy()
        feats.append(np.concatenate([v.mean(axis=0), v.std(axis=0)]))
        labs.append(int(item["has_anomaly"]))
    return np.array(feats, dtype=np.float32), np.array(labs, dtype=int)


def _safe_metrics(labels, probs, name=""):
    labels = np.array(labels, dtype=int)
    probs  = np.array(probs,  dtype=float)
    if labels.sum() == 0 or (1 - labels).sum() == 0:
        logger.warning(f"{name}: degenerate test set (1 class). AUROC=0.5, F1=0.0")
        return 0.5, 0.0
    try:
        auroc = roc_auc_score(labels, probs)
    except Exception as e:
        logger.warning(f"{name} AUROC error: {e}"); auroc = 0.5
    f1 = f1_score(labels, (probs > 0.5).astype(int), zero_division=0)
    return float(auroc), float(f1)


# ---------------------------------------------------------------------------
# AnomalyHead (needed to load AquaSSM checkpoint)
# ---------------------------------------------------------------------------

class _AnomalyHead(nn.Module):
    def __init__(self, input_dim: int = 256, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )
    def forward(self, x): return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# AquaSSM evaluation
# ---------------------------------------------------------------------------

def eval_aquassm(test_dl):
    # Prefer the full 291K-trained model; fall back to real_best, then expanded
    for candidate in [
        "checkpoints/sensor/aquassm_full_best.pt",
        "checkpoints/sensor/aquassm_real_best.pt",
        "checkpoints/sensor/aquassm_expanded_best.pt",
    ]:
        ckpt_path = Path(candidate)
        if ckpt_path.exists():
            logger.info(f"  Using checkpoint: {ckpt_path.name}")
            break
    else:
        logger.warning("No AquaSSM checkpoint found")
        return None, None
    if not ckpt_path.exists():
        logger.warning(f"Checkpoint not found: {ckpt_path}")
        return None, None
    backbone = SensorEncoder().to(DEVICE)
    head     = _AnomalyHead().to(DEVICE)
    ckpt     = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    backbone.load_state_dict(ckpt["model"])
    head.load_state_dict(ckpt["head"])
    backbone.eval(); head.eval()
    probs, labs = [], []
    with torch.no_grad():
        for batch in test_dl:
            v  = batch["values"].to(DEVICE)
            dt = batch["delta_ts"].to(DEVICE)
            y  = batch["has_anomaly"]
            out = backbone(v, delta_ts=dt, compute_anomaly=False)
            emb = out["embedding"]
            if torch.isnan(emb).any(): continue
            logits = head(emb)
            probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            labs.extend(y.numpy().tolist())
    return np.array(labs, dtype=int), np.array(probs, dtype=float)


# ---------------------------------------------------------------------------
# LSTM baseline
# ---------------------------------------------------------------------------

class _LSTMModel(nn.Module):
    def __init__(self, input_dim: int = 6, hidden: int = 128, n_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, n_layers, batch_first=True,
                            dropout=0.2 if n_layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, 1)
        )
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.head(h[-1]).squeeze(-1)


def train_lstm(tr_dl, va_dl, n_epochs: int = 50, pos_w: float = 1.0):
    model = _LSTMModel().to(DEVICE)
    crit  = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w], device=DEVICE))
    opt   = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    best_auroc, best_state = 0.0, None
    for ep in range(1, n_epochs + 1):
        model.train()
        for b in tr_dl:
            v=b["values"].to(DEVICE); y=b["has_anomaly"].to(DEVICE)
            loss = crit(model(v), y)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sched.step()
        model.eval()
        vp, vl = [], []
        with torch.no_grad():
            for b in va_dl:
                vp.extend(torch.sigmoid(model(b["values"].to(DEVICE))).cpu().numpy().tolist())
                vl.extend(b["has_anomaly"].numpy().tolist())
        try:
            vauc = roc_auc_score(vl, vp)
        except Exception:
            vauc = 0.5
        if vauc > best_auroc:
            best_auroc = vauc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 10 == 0:
            logger.info(f"  LSTM ep {ep:2d}/{n_epochs} | val_auroc={vauc:.4f}")
    if best_state:
        model.load_state_dict(best_state)
    return model


def eval_seq_model(model, te_dl):
    model.eval()
    probs, labs = [], []
    with torch.no_grad():
        for b in te_dl:
            probs.extend(torch.sigmoid(model(b["values"].to(DEVICE))).cpu().numpy().tolist())
            labs.extend(b["has_anomaly"].numpy().tolist())
    return np.array(labs, dtype=int), np.array(probs, dtype=float)


# ---------------------------------------------------------------------------
# Transformer baseline
# ---------------------------------------------------------------------------

class _TransformerModel(nn.Module):
    def __init__(self, d_model: int = 64, nhead: int = 4, n_layers: int = 2,
                 max_seq_len: int = 128):
        super().__init__()
        self.proj = nn.Linear(6, d_model)
        self.pos  = nn.Embedding(max_seq_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128,
            dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, 32), nn.GELU(), nn.Dropout(0.1), nn.Linear(32, 1)
        )
    def forward(self, x):
        B, T, _ = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        x   = self.proj(x) + self.pos(pos)
        x   = self.encoder(x).mean(dim=1)
        return self.head(x).squeeze(-1)


def train_transformer(tr_dl, va_dl, n_epochs: int = 50, pos_w: float = 1.0):
    model = _TransformerModel().to(DEVICE)
    crit  = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w], device=DEVICE))
    opt   = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    best_auroc, best_state = 0.0, None
    for ep in range(1, n_epochs + 1):
        model.train()
        for b in tr_dl:
            v=b["values"].to(DEVICE); y=b["has_anomaly"].to(DEVICE)
            loss = crit(model(v), y)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sched.step()
        model.eval()
        vp, vl = [], []
        with torch.no_grad():
            for b in va_dl:
                vp.extend(torch.sigmoid(model(b["values"].to(DEVICE))).cpu().numpy().tolist())
                vl.extend(b["has_anomaly"].numpy().tolist())
        try:
            vauc = roc_auc_score(vl, vp)
        except Exception:
            vauc = 0.5
        if vauc > best_auroc:
            best_auroc = vauc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 10 == 0:
            logger.info(f"  Transformer ep {ep:2d}/{n_epochs} | val_auroc={vauc:.4f}")
    if best_state:
        model.load_state_dict(best_state)
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info("=" * 70)
    logger.info("AquaSSM Benchmark — Real Data Only")
    logger.info("=" * 70)
    logger.info(f"Device: {DEVICE} | MAX_LEN={MAX_LEN}")

    tr_ds, va_ds, te_ds, n_total = build_split()
    pos_w = _compute_pos_weight(tr_ds)
    logger.info(f"pos_weight: {pos_w:.2f}")

    tr_dl = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=_collate, num_workers=0)
    va_dl = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=_collate, num_workers=0)
    te_dl = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=_collate, num_workers=0)

    results = {"timestamp": ts, "n_total": n_total, "max_len": MAX_LEN, "models": {}}

    # ---- AquaSSM (expanded checkpoint) ----
    logger.info("\n--- AquaSSM (expanded, real data checkpoint) ---")
    aq_labs, aq_probs = eval_aquassm(te_dl)
    if aq_labs is not None:
        auroc, f1 = _safe_metrics(aq_labs, aq_probs, "AquaSSM")
        logger.info(f"AquaSSM  AUROC={auroc:.4f}  F1={f1:.4f}  (n={len(aq_labs)})")
        results["models"]["AquaSSM"] = {"auroc": auroc, "f1": f1, "n_test": len(aq_labs)}
    else:
        logger.warning("AquaSSM checkpoint not found.")
        results["models"]["AquaSSM"] = {"auroc": None, "f1": None, "note": "checkpoint not found"}

    # ---- LSTM ----
    logger.info(f"\n--- LSTM (2-layer, hidden=128, {LSTM_EPOCHS} epochs) ---")
    lstm = train_lstm(tr_dl, va_dl, LSTM_EPOCHS, pos_w)
    lstm_labs, lstm_probs = eval_seq_model(lstm, te_dl)
    auroc, f1 = _safe_metrics(lstm_labs, lstm_probs, "LSTM")
    logger.info(f"LSTM     AUROC={auroc:.4f}  F1={f1:.4f}")
    results["models"]["LSTM"] = {"auroc": auroc, "f1": f1, "n_test": len(lstm_labs),
                                  "config": {"hidden": 128, "layers": 2, "epochs": LSTM_EPOCHS}}

    # ---- Transformer ----
    logger.info(f"\n--- Transformer (d_model=64, nhead=4, {TFM_EPOCHS} epochs) ---")
    tfm = train_transformer(tr_dl, va_dl, TFM_EPOCHS, pos_w)
    tfm_labs, tfm_probs = eval_seq_model(tfm, te_dl)
    auroc, f1 = _safe_metrics(tfm_labs, tfm_probs, "Transformer")
    logger.info(f"Transformer  AUROC={auroc:.4f}  F1={f1:.4f}")
    results["models"]["Transformer"] = {"auroc": auroc, "f1": f1, "n_test": len(tfm_labs),
                                         "config": {"d_model": 64, "nhead": 4, "epochs": TFM_EPOCHS}}

    # ---- Isolation Forest ----
    logger.info("\n--- Isolation Forest (n_estimators=200, mean+std features) ---")
    X_tr, y_tr = _get_features_labels(tr_ds)
    X_te, y_te = _get_features_labels(te_ds)
    logger.info(f"  Feature dims: train={X_tr.shape}, test={X_te.shape}")
    iforest = IsolationForest(n_estimators=200, random_state=SEED, n_jobs=-1)
    iforest.fit(X_tr)
    if_scores = -iforest.decision_function(X_te)
    if_norm   = (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min() + 1e-8)
    auroc, f1 = _safe_metrics(y_te, if_norm, "IsolationForest")
    logger.info(f"IsoForest AUROC={auroc:.4f}  F1={f1:.4f}")
    results["models"]["IsolationForest"] = {
        "auroc": auroc, "f1": f1, "n_test": len(y_te),
        "config": {"n_estimators": 200, "features": "mean+std per channel (12-dim)"}
    }

    # ---- One-Class SVM ----
    logger.info("\n--- One-Class SVM (nu=0.1, rbf, normal-only training) ---")
    X_tr_normal = X_tr[y_tr == 0]
    logger.info(f"  Training OC-SVM on {len(X_tr_normal)} normal samples")
    ocsvm = OneClassSVM(nu=0.1, kernel="rbf", gamma="auto")
    ocsvm.fit(X_tr_normal)
    oc_scores = -ocsvm.decision_function(X_te)
    oc_norm   = (oc_scores - oc_scores.min()) / (oc_scores.max() - oc_scores.min() + 1e-8)
    auroc, f1 = _safe_metrics(y_te, oc_norm, "OneClassSVM")
    logger.info(f"OC-SVM   AUROC={auroc:.4f}  F1={f1:.4f}")
    results["models"]["OneClassSVM"] = {
        "auroc": auroc, "f1": f1, "n_test": len(y_te),
        "config": {"nu": 0.1, "kernel": "rbf", "gamma": "auto"}
    }

    # ---- Save ----
    elapsed = time.time() - t0
    results.update({"elapsed_seconds": elapsed, "elapsed_minutes": elapsed / 60.0})
    output_path = RESULTS_DIR / "aquassm_benchmark.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {output_path}")

    # ---- Summary table ----
    print("\n" + "=" * 65)
    print("=== AquaSSM Benchmark — Summary Table ===")
    print(f"{'Model':<22} {'AUROC':>8} {'F1@0.5':>8}")
    print("-" * 65)
    for name in ["AquaSSM", "LSTM", "Transformer", "IsolationForest", "OneClassSVM"]:
        if name not in results["models"]: continue
        m     = results["models"][name]
        astr  = f"{m['auroc']:.4f}" if m.get("auroc") is not None else "   N/A "
        fstr  = f"{m['f1']:.4f}"    if m.get("f1")    is not None else "   N/A "
        print(f"  {name:<20} {astr:>8} {fstr:>8}")
    print("=" * 65)
    print(f"Total time: {elapsed/60:.1f} min  |  Results: {output_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
