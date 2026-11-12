# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""SimSiam (Chen & He, CVPR 2021) for the Lite MobileNetV4-Conv-S encoder on unlabelled NAIP water chips.

The views are built to target the M7 diagnosis (NAIP per-state/year vendor colour balance):
- random 128-px crops (the student's training crop) within the 256-px chip, with flips and rot90;
- strong per-band gain (±30%), global gain, gamma (0.7–1.4) and offset;
- optional ×2 block resampling and noise.

Two views share geometry only with probability 0.5, so the encoder must also match neighbouring water. Input is
DN/255, standardised with the corpus band mean/std, as the student's "raw" mode does. Only the encoder is kept
(``state["encoder"]``); the projector and predictor are discarded.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from ..lite.model import LiteStudent


class SimSiam(nn.Module):
    def __init__(self, backbone: str = "mnv4s", dim: int = 512, pred_dim: int = 128, pretrained: bool = True):
        super().__init__()
        self.encoder = LiteStudent(backbone, pretrained=pretrained).encoder
        c = self.encoder.feature_info.channels()[-1]
        self.projector = nn.Sequential(nn.Linear(c, dim, bias=False), nn.BatchNorm1d(dim), nn.ReLU(inplace=True),
                                       nn.Linear(dim, dim, bias=False), nn.BatchNorm1d(dim), nn.ReLU(inplace=True),
                                       nn.Linear(dim, dim, bias=False), nn.BatchNorm1d(dim, affine=False))
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False), nn.BatchNorm1d(pred_dim),
                                       nn.ReLU(inplace=True), nn.Linear(pred_dim, dim))

    def embed(self, x):
        return self.encoder(x)[-1].mean((2, 3))

    def forward(self, x1, x2):
        z1, z2 = self.projector(self.embed(x1)), self.projector(self.embed(x2))
        p1, p2 = self.predictor(z1), self.predictor(z2)
        return p1, p2, z1.detach(), z2.detach()


def simsiam_loss(p1, p2, z1, z2):
    return -(Fn.cosine_similarity(p1, z2, dim=-1).mean() + Fn.cosine_similarity(p2, z1, dim=-1).mean()) / 2


def view(dn: torch.Tensor, g: torch.Generator, crop: int = 128, offsets=None) -> torch.Tensor:
    """dn: (B, 4, 256, 256) float DN on device. Returns augmented DN crops (B, 4, crop, crop)."""
    B, C, H, W = dn.shape
    if offsets is None:
        offsets = torch.randint(0, H - crop + 1, (B, 2), generator=g)
    out = torch.stack([dn[i, :, int(o[0]):int(o[0]) + crop, int(o[1]):int(o[1]) + crop] for i, o in enumerate(offsets)])
    k = int(torch.randint(0, 4, (1,), generator=g))
    out = torch.rot90(out, k, (2, 3))
    if bool(torch.randint(0, 2, (1,), generator=g)):
        out = torch.flip(out, (3,))
    dev = out.device
    band = (1 + 0.3 * (2 * torch.rand(B, C, 1, 1, generator=g) - 1)).to(dev)
    glob = (1 + 0.15 * (2 * torch.rand(B, 1, 1, 1, generator=g) - 1)).to(dev)
    gamma = (0.7 + 0.7 * torch.rand(B, 1, 1, 1, generator=g)).to(dev)
    off = (6.0 * (2 * torch.rand(B, C, 1, 1, generator=g) - 1)).to(dev)
    x = ((out / 255.0).clamp_min(1e-4) ** gamma) * 255.0 * band * glob + off
    if bool(torch.rand(1, generator=g) < 0.3):
        x = Fn.interpolate(Fn.avg_pool2d(x, 2), scale_factor=2, mode="nearest")
    x = x + 1.5 * torch.randn(x.shape, generator=g).to(dev)
    return x.clamp(0, 255)


def train_simsiam(chips_u8: np.ndarray, *, epochs: int, batch: int = 128, lr: float = 0.05, wd: float = 1e-4,
                  device: str = "cuda", seed: int = 0, log=print, max_minutes: float | None = None,
                  state: dict | None = None) -> dict:
    """Train (or resume from ``state``) SimSiam; returns a state dict with the encoder, the norm stats and the
    history. Stops early (resumable) after ``max_minutes`` (shared-GPU ≤ 2 h rule)."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    n = len(chips_u8)
    x_mean = torch.tensor(chips_u8[:2000].reshape(min(n, 2000), 4, -1).mean((0, 2)) / 255.0, dtype=torch.float32)
    x_std = torch.tensor(chips_u8[:2000].reshape(min(n, 2000), 4, -1).std((0, 2)) / 255.0 + 1e-3, dtype=torch.float32)
    model = SimSiam().to(device)
    lr_b = lr * batch / 256
    opt = torch.optim.SGD([{"params": list(model.encoder.parameters()) + list(model.projector.parameters()), "fix": False},
                           {"params": model.predictor.parameters(), "fix": True}], lr=lr_b, momentum=0.9, weight_decay=wd)
    start, hist = 0, []
    if state:
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        start, hist = state["epoch"], state["history"]
    spe = n // batch
    t0 = time.time()
    xm, xs = x_mean.to(device)[None, :, None, None], x_std.to(device)[None, :, None, None]
    data = torch.from_numpy(chips_u8)          # stays in host RAM (uint8)
    for ep in range(start, epochs):
        cur = lr_b * 0.5 * (1 + math.cos(math.pi * ep / epochs))
        for pg in opt.param_groups:
            pg["lr"] = lr_b if pg.get("fix") else cur
        model.train()
        perm = torch.randperm(n, generator=g)
        tot = 0.0
        for it in range(spe):
            idx = perm[it * batch:(it + 1) * batch]
            dn = data[idx].to(device, non_blocking=True).float()
            same = bool(torch.rand(1, generator=g) < 0.5)
            off1 = torch.randint(0, 129, (len(idx), 2), generator=g)
            v1 = view(dn, g, offsets=off1)
            v2 = view(dn, g, offsets=off1 if same else None)
            x1, x2 = (v1 / 255.0 - xm) / xs, (v2 / 255.0 - xm) / xs
            p1, p2, z1, z2 = model(x1, x2)
            loss = simsiam_loss(p1, p2, z1, z2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss)
        rec = {"epoch": ep + 1, "loss": tot / max(1, spe), "minutes": (time.time() - t0) / 60}
        hist.append(rec)
        log(f"epoch {ep + 1}/{epochs} loss {rec['loss']:.4f} ({rec['minutes']:.1f} min)")
        if max_minutes and (time.time() - t0) / 60 > max_minutes and ep + 1 < epochs:
            log("time budget reached; stopping (resumable)")
            return {"model": model.state_dict(), "opt": opt.state_dict(), "epoch": ep + 1, "history": hist,
                    "encoder": model.encoder.state_dict(), "x_mean": x_mean, "x_std": x_std, "done": False}
    return {"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epochs, "history": hist,
            "encoder": model.encoder.state_dict(), "x_mean": x_mean, "x_std": x_std, "done": True}
