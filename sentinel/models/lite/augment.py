# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Radiometric/geometric training augmentation and the fixed robustness perturbations (protocol §7).

All functions operate on float tensors of DN in [0, 255], shape (N, 4, H, W), bands R, G, B, NIR.
"""

from __future__ import annotations

import torch
import torch.nn.functional as Fn


def geometric(x: torch.Tensor, w: torch.Tensor, g: torch.Generator | None = None):
    """Random flips / rot90 applied identically to chips ``x`` (N,C,H,W) and pooling maps ``w`` (N,h,w).
    The site stays at the centre, so the centred pooling window remains valid."""
    k = int(torch.randint(0, 4, (1,), generator=g))
    if k:
        x, w = torch.rot90(x, k, (2, 3)), torch.rot90(w, k, (1, 2))
    if bool(torch.randint(0, 2, (1,), generator=g)):
        x, w = torch.flip(x, (3,)), torch.flip(w, (2,))
    return x, w


def radiometric(x: torch.Tensor, *, band_gain: float = 0.10, global_gain: float = 0.15, offset_dn: float = 3.0,
                gamma: float = 0.10, noise_dn: float = 1.5, p_resample: float = 0.2) -> torch.Tensor:
    N = x.shape[0]
    dev = x.device
    gb = 1 + band_gain * (2 * torch.rand(N, 4, 1, 1, device=dev) - 1)
    gg = 1 + global_gain * (2 * torch.rand(N, 1, 1, 1, device=dev) - 1)
    off = offset_dn * torch.randn(N, 4, 1, 1, device=dev)
    gm = 1 + gamma * (2 * torch.rand(N, 1, 1, 1, device=dev) - 1)
    y = 255.0 * (x.clamp(0, 255) / 255.0) ** gm
    y = y * gb * gg + off + noise_dn * torch.randn_like(y)
    if p_resample > 0:
        sel = torch.rand(N, device=dev) < p_resample
        if sel.any():
            f = int(torch.randint(1, 3, (1,))) * 2          # x2 or x4 block averaging
            ys = y[sel]
            ys = Fn.interpolate(Fn.avg_pool2d(ys, f), scale_factor=f, mode="nearest")
            y = y.clone()
            y[sel] = ys
    return y.clamp(0, 255)


# ------------------------------------------------------------------ robustness perturbations (fixed, seeded)

def _blobs(shape, frac: float, g: torch.Generator, device) -> torch.Tensor:
    """Smooth random mask covering ~``frac`` of pixels (thresholded low-pass noise)."""
    N, _, H, W = shape
    z = torch.rand(N, 1, H // 16, W // 16, generator=g).to(device)
    z = Fn.interpolate(z, size=(H, W), mode="bilinear", align_corners=False)
    thr = torch.quantile(z.reshape(N, -1), 1 - frac, dim=1).view(N, 1, 1, 1)
    return (z >= thr).float()


def glint_aug(x: torch.Tensor, g: torch.Generator, p: float = 0.5, frac=(0.02, 0.40), amp=(40.0, 150.0)) -> torch.Tensor:
    """Training-time sun-glint augmentation (lead decision 2026-11-08; protocol §7 note): for a random fraction ``p``
    of the batch, add near-white specular blobs covering U(frac) of the pixels with amplitude U(amp) DN. Broader
    than the robustness-suite generator (coverage 10/25/40%, amplitude 60–120 DN) and drawn from the training
    generator, but the same blob family: robustness under glint is therefore optimistic for augmented models."""
    N = x.shape[0]
    sel = torch.rand(N, generator=g) < p
    if not bool(sel.any()):
        return x
    xs = x[sel.to(x.device)]
    n = xs.shape[0]
    fr = float(frac[0] + (frac[1] - frac[0]) * torch.rand(1, generator=g))
    m = _blobs(xs.shape, fr, g, x.device)
    a = (amp[0] + (amp[1] - amp[0]) * torch.rand(n, 1, 1, 1, generator=g)).to(x.device)
    out = x.clone()
    out[sel.to(x.device)] = (xs + m * a).clamp(0, 255)
    return out


def perturb(x: torch.Tensor, kind: str, level: float, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    dev = x.device
    if kind == "clean" or level == 0:
        return x
    if kind == "glint":            # specular: near-white additive, slightly stronger in NIR-poor bands
        m = _blobs(x.shape, level, g, dev)
        amp = (60 + 60 * torch.rand(x.shape[0], 1, 1, 1, generator=g)).to(dev)
        return (x + m * amp).clamp(0, 255)
    if kind == "haze":             # blue-weighted path radiance, level = blue offset in DN
        w = torch.tensor([0.6, 0.8, 1.0, 0.3], device=dev).view(1, 4, 1, 1)
        return (x + level * w).clamp(0, 255)
    if kind == "resolution":       # block-average by factor `level`, nearest up-sampling
        f = int(level)
        return Fn.interpolate(Fn.avg_pool2d(x, f), scale_factor=f, mode="nearest")
    if kind == "gain":             # per-band calibration error ±level with random sign per band
        s = (torch.randint(0, 2, (1, 4, 1, 1), generator=g) * 2 - 1).to(dev).float()
        return (x * (1 + level * s)).clamp(0, 255)
    if kind == "misreg":           # NIR band shifted by `level` px (x direction)
        y = x.clone()
        y[:, 3] = torch.roll(x[:, 3], shifts=int(level), dims=-1)
        return y
    raise ValueError(kind)


ROBUSTNESS_GRID = [("glint", 0.10), ("glint", 0.25), ("glint", 0.40), ("haze", 5), ("haze", 10), ("haze", 20),
                   ("resolution", 2), ("resolution", 4), ("resolution", 8), ("gain", 0.05), ("misreg", 1),
                   ("misreg", 3), ("misreg", 10)]
GAIN_DRAWS = 5
