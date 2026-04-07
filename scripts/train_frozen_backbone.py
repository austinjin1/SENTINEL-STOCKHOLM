#!/usr/bin/env python3
"""Train anomaly head on frozen AquaSSM backbone embeddings.

The SSM is numerically fragile during fine-tuning, so we freeze it and
only train a lightweight classification head on precomputed embeddings.

MIT License — Bryan Cheng, 2026
"""
import torch, torch.nn as nn, numpy as np, json, time
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import roc_auc_score, f1_score
from sentinel.models.sensor_encoder import SensorEncoder
from sentinel.utils.logging import get_logger
logger = get_logger(__name__)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CKPT = Path("checkpoints/sensor"); CKPT.mkdir(parents=True, exist_ok=True)

with open("data/processed/sensor/good_files.json") as f:
    good_files = json.load(f)

class DS(Dataset):
    def __init__(self, files, ml=512):
        self.files = files; self.ml = ml
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        d = np.load(self.files[i])
        T = min(len(d["values"]), self.ml)
        v = torch.tensor(d["values"][:T].astype(np.float32)).clamp(-5, 5)
        dt = torch.tensor(d["delta_ts"][:T].astype(np.float32)).clamp(0, 3600)
        dt[0] = 0
        ha = int((d["labels"][:T] > 0).any()) if "labels" in d else 0
        return {"values": v, "delta_ts": dt, "has_anomaly": ha}

def collate(batch):
    ml = max(b["values"].shape[0] for b in batch); B = len(batch)
    v = torch.zeros(B, ml, 6); dt = torch.zeros(B, ml)
    ha = torch.tensor([b["has_anomaly"] for b in batch])
    for i, b in enumerate(batch):
        T = b["values"].shape[0]; v[i,:T] = b["values"]; dt[i,:T] = b["delta_ts"]
    return {"values": v, "delta_ts": dt, "has_anomaly": ha}

class Head(nn.Module):
    def __init__(self, d=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 128), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
    def forward(self, x): return self.net(x).squeeze(-1)

ds = DS(good_files)
n = len(ds); n_tr = int(.7*n); n_va = int(.15*n); n_te = n - n_tr - n_va
tr, va, te = random_split(ds, [n_tr, n_va, n_te], generator=torch.Generator().manual_seed(42))
tr_dl = DataLoader(tr, batch_size=4, shuffle=True, collate_fn=collate)
va_dl = DataLoader(va, batch_size=4, collate_fn=collate)
te_dl = DataLoader(te, batch_size=4, collate_fn=collate)
logger.info(f"Data: {n_tr}/{n_va}/{n_te}")

# Freeze SSM backbone
model = SensorEncoder().to(DEVICE)
model.eval()
for p in model.parameters(): p.requires_grad = False

# Precompute embeddings
logger.info("Precomputing embeddings...")
def extract(dl):
    embs, lbls = [], []
    with torch.no_grad():
        for batch in dl:
            v = batch["values"].to(DEVICE); dt = batch["delta_ts"].to(DEVICE)
            ha = batch["has_anomaly"].float()
            out = model(v, dt, compute_anomaly=False)
            for j in range(out["embedding"].shape[0]):
                if not torch.isnan(out["embedding"][j]).any():
                    embs.append(out["embedding"][j].cpu())
                    lbls.append(ha[j].item())
    return torch.stack(embs) if embs else torch.zeros(0,256), torch.tensor(lbls)

train_e, train_l = extract(tr_dl)
val_e, val_l = extract(va_dl)
test_e, test_l = extract(te_dl)
logger.info(f"Embeddings: tr={len(train_e)} va={len(val_e)} te={len(test_e)}")
logger.info(f"Pos rates: tr={train_l.mean():.2f} va={val_l.mean():.2f} te={test_l.mean():.2f}")

if len(train_e) == 0:
    logger.error("No valid embeddings! Exiting."); exit(1)

# Train head
head = Head().to(DEVICE)
opt = torch.optim.Adam(head.parameters(), lr=1e-3)
t0 = time.time()
best_auc = 0

for ep in range(200):
    head.train()
    perm = torch.randperm(len(train_e))
    e_s = train_e[perm].to(DEVICE); l_s = train_l[perm].to(DEVICE)
    total_loss, nb = 0, 0
    for i in range(0, len(e_s), 16):
        e = e_s[i:i+16]; l = l_s[i:i+16]
        logit = head(e); loss = nn.functional.binary_cross_entropy_with_logits(logit, l)
        opt.zero_grad(); loss.backward(); opt.step()
        total_loss += loss.item(); nb += 1

    head.eval()
    with torch.no_grad():
        tr_pred = torch.sigmoid(head(train_e.to(DEVICE))).cpu().numpy()
        va_pred = torch.sigmoid(head(val_e.to(DEVICE))).cpu().numpy()
    try: tr_auc = roc_auc_score(train_l.numpy(), tr_pred)
    except: tr_auc = 0.5
    try: va_auc = roc_auc_score(val_l.numpy(), va_pred)
    except: va_auc = 0.5

    if (ep+1) % 25 == 0 or ep == 0:
        logger.info(f"Ep {ep+1:3d}/200 | Loss: {total_loss/nb:.4f} | Tr: {tr_auc:.4f} | Va: {va_auc:.4f}")
    if va_auc > best_auc:
        best_auc = va_auc
        torch.save(head.state_dict(), CKPT / "head_best.pt")

# Test
head.eval()
with torch.no_grad():
    te_pred = torch.sigmoid(head(test_e.to(DEVICE))).cpu().numpy()
try: auroc = roc_auc_score(test_l.numpy(), te_pred)
except: auroc = 0.5
f1 = f1_score(test_l.numpy(), (te_pred > 0.5).astype(int), zero_division=0)
elapsed = time.time() - t0

logger.info(f"TEST: AUROC={auroc:.4f} F1={f1:.4f} N={len(test_l)} Pos={int(test_l.sum())}")
if auroc > 0.85: logger.info("*** HARD THRESHOLD MET! ***")
elif auroc > 0.70: logger.info("ACCEPTABLE")
else: logger.info(f"BELOW THRESHOLD ({auroc:.4f})")

json.dump({"auroc": float(auroc), "f1": float(f1), "best_val_auc": best_auc,
           "n_train": len(train_e), "n_test": len(test_l), "elapsed": elapsed},
          open(CKPT / "results_v5.json", "w"), indent=2)
logger.info("DONE")
