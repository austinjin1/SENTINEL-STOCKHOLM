# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""GLORIA band-simulation pretraining of the student's spectral branch (research 08 §c.6 R1).

GLORIA in-situ Rrs convolved with the NAIP ADS100 top-hat bands (`docs/datasheets/gloria.md`;
approximation — no measured camera response yet) → per-pixel features of :class:`SpectralBranch`.
Input scaling: x = 4·π·Rrs (≈ 4 × water-leaving reflectance) so values fall in the DN/255 range of NAIP water;
the branch's BatchNorm re-adapts its feature statistics when fine-tuned on NAIP. Unflagged samples only.
Targets are standardised with the *NAIP train-role* label mean/std so the branch output adds directly to
the student's head. Radiometric augmentation: per-band gain ±10 %, global gain ±30 %, additive noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ...core import paths
from .model import SpectralBranch

BANDS = ("red", "green", "blue", "nir")          # NAIP order R, G, B, NIR
GLORIA_TARGETS = {"secchi": "Secchi_depth", "turbidity": "Turbidity", "tss": "TSS", "chla": "Chla"}
SCALE = 4.0 * np.pi


def load_gloria() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    g = pd.read_parquet(paths.PROCESSED / "gloria" / "gloria_samples.parquet")
    cols = [f"rrs_naip_ads100_{b}" for b in BANDS]
    g = g[g[cols].notna().all(1) & ~g["Flagged"].fillna(False).astype(bool)]
    if "Turbidity_unit" in g:
        g.loc[~g["Turbidity_unit"].astype(str).str.upper().str.contains("NTU"), "Turbidity"] = np.nan
    X = (g[cols].to_numpy(float) * SCALE).clip(0, None)
    Y = np.column_stack([np.log10(g[c].where(g[c] > 0)) for c in GLORIA_TARGETS.values()])
    return X.astype(np.float32), Y.astype(np.float32), g


def pretrain(y_mean, y_std, *, steps: int = 4000, batch: int = 256, lr: float = 3e-3, seed: int = 0,
             device: str = "cpu", log=print) -> tuple[SpectralBranch, dict]:
    torch.manual_seed(seed)
    X, Y, g = load_gloria()
    ym = torch.as_tensor(np.asarray(y_mean, np.float32))
    ys = torch.as_tensor(np.asarray(y_std, np.float32))
    Xt = torch.from_numpy(X).to(device)
    Yt = ((torch.from_numpy(Y) - ym) / ys).to(device)
    M = torch.isfinite(Yt)
    Yt = torch.nan_to_num(Yt)
    T = Y.shape[1]
    br = SpectralBranch(2 * T).to(device)
    opt = torch.optim.AdamW(br.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    from .losses import gaussian_nll

    n = len(X)
    for s in range(steps):
        idx = torch.randint(0, n, (batch,), device=device)
        x = Xt[idx]
        x = x * (1 + 0.10 * (2 * torch.rand(batch, 4, device=device) - 1)) * \
            (1 + 0.30 * (2 * torch.rand(batch, 1, device=device) - 1))
        x = (x + 0.004 * torch.randn_like(x)).clamp_min(0)
        out = br(x[:, :, None, None])[:, :, 0, 0]
        mu, lv = out[:, :T], out[:, T:].clamp(-8, 6)
        loss = gaussian_nll(mu, torch.exp(lv), Yt[idx], M[idx], beta=0.5)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sch.step()
        if (s + 1) % 1000 == 0:
            log(f"gloria step {s + 1} loss {float(loss):.4f}")
    info = {"n_samples": int(n), "n_labels": {t: int(np.isfinite(Y[:, j]).sum()) for j, t in enumerate(GLORIA_TARGETS)},
            "n_datasets": int(g["Dataset_ID"].nunique()), "steps": steps}
    return br.cpu(), info
