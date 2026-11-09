# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Training / prediction loop for the SENTINEL-Lite student (direct in-situ supervision ± distillation).

Inputs live on the GPU as uint8; augmentation runs on the GPU. Fixed epoch budget (no early stopping on
val); the val curve is recorded for diagnosis only.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch

from . import augment as A
from .data import TARGETS
from .losses import pool_maps_median, distill_nll, gaussian_nll, pool_maps
from .model import LiteStudent

DEFAULT_CFG = dict(backbone="mnv4s", dec_ch=48, spectral=False, pretrained=True, input_mode="raw", epochs=40,
                   batch=32, lr=1e-3, lr_backbone=5e-4, wd=1e-4, beta=0.5, warmup_epochs=2, aug=True,
                   aug_band_gain=0.10, aug_global_gain=0.15, aug_offset_dn=3.0, aug_gamma=0.10, aug_noise_dn=1.5,
                   aug_p_resample=0.2, film=False, crop=0, kd_weight=0.0, kd_sigma_scale2=0.01, rgb_only=False, gloria_init=None,
                   amp=True, train_targets=None)


class Inputs:
    """Holds one role's chips + per-chip normalisation + pooling weights on a device."""

    def __init__(self, chips: np.ndarray, scene_params: np.ndarray, pool_w: np.ndarray, y: np.ndarray | None,
                 device: str):
        self.dn = torch.from_numpy(np.ascontiguousarray(chips)).to(device)          # uint8 (N,4,H,W)
        self.sp = torch.from_numpy(scene_params).to(device)                         # (N,4,2) centre, scale
        self.w = torch.from_numpy(pool_w).to(device)                                # (N,h,w)
        self.y = None if y is None else torch.from_numpy(y.astype(np.float32)).to(device)
        self.n = self.dn.shape[0]


def normalise_input(dn: torch.Tensor, sp: torch.Tensor, model: LiteStudent, mode: str, rgb_only: bool = False):
    """Return (x, x01): model input and the [0,1] DN for the spectral branch."""
    x01 = dn / 255.0
    if mode == "raw":
        x = (x01 - model.x_mean.view(1, -1, 1, 1)) / model.x_std.view(1, -1, 1, 1)
    elif mode == "scene":
        x = (dn - sp[:, :, 0, None, None]) / sp[:, :, 1, None, None]
    else:
        raise ValueError(mode)
    if rgb_only:
        x = x.clone()
        x[:, 3] = 0.0
        x01 = x01.clone()
        x01[:, 3] = x01[:, 1]      # neutral NIR for the log-ratio features
    return x, x01


def scene_cond(sp: torch.Tensor) -> torch.Tensor:
    """(N, 8) DOQQ statistics for scene-FiLM: per-band centre/255 and scale/255."""
    return torch.cat([sp[:, :, 0], sp[:, :, 1]], 1) / 255.0


def centre_crop(dn: torch.Tensor, w: torch.Tensor, crop: int, stride: int = 4):
    """Centre crop of chips (N,C,H,W) and pooling maps (N,H/stride,W/stride); weights renormalised."""
    if not crop or crop >= dn.shape[-1]:
        return dn, w
    c0 = (dn.shape[-1] - crop) // 2
    dn = dn[..., c0:c0 + crop, c0:c0 + crop]
    w = w[:, c0 // stride:(c0 + crop) // stride, c0 // stride:(c0 + crop) // stride]
    return dn, w / w.sum((1, 2), keepdim=True).clamp_min(1e-8)


def fit_stats(model: LiteStudent, tr: Inputs, y_raw: np.ndarray) -> None:
    with torch.no_grad():
        x01 = tr.dn[:: max(1, tr.n // 512)].float() / 255.0
        model.x_mean.copy_(x01.mean((0, 2, 3)))
        model.x_std.copy_(x01.std((0, 2, 3)).clamp_min(1e-3))
    model.y_mean.copy_(torch.tensor(np.nanmean(y_raw, 0), dtype=torch.float32))
    model.y_std.copy_(torch.tensor(np.nanstd(y_raw, 0), dtype=torch.float32).clamp_min(1e-3))


def _lr_lambda(cfg, steps_per_epoch):
    total = cfg["epochs"] * steps_per_epoch
    warm = cfg["warmup_epochs"] * steps_per_epoch

    def f(step):
        if step < warm:
            return (step + 1) / max(warm, 1)
        return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(total - warm, 1)))
    return f


def build_model(cfg) -> LiteStudent:
    m = LiteStudent(cfg["backbone"], pretrained=cfg["pretrained"], dec_ch=cfg["dec_ch"], spectral=cfg["spectral"],
                    film=cfg.get("film", False), film_source=cfg.get("film_source", "frame"))
    if cfg.get("gloria_init"):
        if m.spectral is None:
            raise ValueError("gloria_init requires spectral: true")
        from pathlib import Path

        from ...core import paths

        gp = Path(cfg["gloria_init"])
        gp = gp if gp.is_absolute() else paths.RUNS_ROOT / gp      # relative to $SENTINEL_RUNS
        sd = torch.load(gp, map_location="cpu", weights_only=False)
        m.spectral.load_state_dict(sd["spectral"], strict=True)
    return m


@torch.no_grad()
def predict(model: LiteStudent, inp: Inputs, cfg, batch: int = 64, perturb=None) -> tuple[np.ndarray, np.ndarray]:
    """Chip-level (μ, σ) in log10 units, shape (N, T). ``perturb``: callable on float DN batch."""
    model.eval()
    T = model.n_targets
    mus, sds = [], []
    for s in range(0, inp.n, batch):
        dn = inp.dn[s:s + batch].float()
        if perturb is not None:
            dn = perturb(dn)
        w = inp.w[s:s + batch]
        dn, w = centre_crop(dn, w, cfg.get("crop", 0))
        x, x01 = normalise_input(dn, inp.sp[s:s + batch], model, cfg["input_mode"], cfg["rgb_only"])
        with torch.autocast("cuda", enabled=cfg["amp"] and dn.is_cuda):
            out = model(x, x01 if model.needs_x01 else None, scene_cond(inp.sp[s:s + batch])).float()
        mu, var = (pool_maps_median if cfg.get("pool") == "median" else pool_maps)(out, w, T)
        mus.append((mu * model.y_std + model.y_mean).cpu().numpy())
        sds.append((torch.sqrt(var) * model.y_std).cpu().numpy())
    return np.concatenate(mus), np.concatenate(sds)


def train(cfg: dict, tr: Inputs, y_raw: np.ndarray, *, val: Inputs | None = None, val_y: np.ndarray | None = None,
          kd: dict | None = None, seed: int = 0, log=print, device: str = "cuda") -> tuple[LiteStudent, list]:
    """Train one student. ``y_raw`` (N, T) log10 labels with NaN; ``kd`` optional teacher tensors:
    {cell_map (N,h,w) long, water_w (N,h,w), t_mu (N,K,T) log10, t_var (N,K,T), t_w (N,K,T)}."""
    cfg = {**DEFAULT_CFG, **cfg}
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(cfg).to(device)
    fit_stats(model, tr, y_raw)
    ys = (torch.from_numpy(y_raw.astype(np.float32)).to(device) - model.y_mean) / model.y_std
    mask = torch.isfinite(ys)
    if cfg.get("train_targets"):           # direct-label loss only on these targets (e.g. single-target Secchi)
        from .data import TARGETS as _ALL_T

        keep = torch.tensor([t in cfg["train_targets"] for t in _ALL_T], device=mask.device)
        mask = mask & keep[None, :mask.shape[1]]
    ys = torch.nan_to_num(ys)
    if kd is not None:
        kd = {k: torch.as_tensor(v).to(device) for k, v in kd.items()}
        kd["t_mu_s"] = (kd["t_mu"] - model.y_mean) / model.y_std
        kd["t_var_s"] = kd["t_var"] / model.y_std ** 2
    bb = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    rest = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    opt = torch.optim.AdamW([{"params": bb, "lr": cfg["lr_backbone"]}, {"params": rest, "lr": cfg["lr"]}],
                            weight_decay=cfg["wd"])
    spe = max(1, tr.n // cfg["batch"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda(cfg, spe))
    use_amp = bool(cfg["amp"]) and str(device).startswith("cuda")     # CPU training runs in fp32
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    gen = torch.Generator().manual_seed(seed)
    hist = []
    T = model.n_targets
    for ep in range(cfg["epochs"]):
        model.train()
        t0 = time.time()
        perm = torch.randperm(tr.n, generator=gen).to(device)
        tot, tot_kd = 0.0, 0.0
        for it in range(spe):
            idx = perm[it * cfg["batch"]:(it + 1) * cfg["batch"]]
            dn = tr.dn[idx].float()
            w = tr.w[idx]
            if kd is not None:
                cm, ww = kd["cell_map"][idx], kd["water_w"][idx]
            if cfg["aug"]:
                if kd is None:
                    dn, w = A.geometric(dn, w, gen)
                else:   # rotate the KD maps identically
                    k = int(torch.randint(0, 4, (1,), generator=gen))
                    fl = bool(torch.randint(0, 2, (1,), generator=gen))
                    dn, w, cm, ww = (torch.rot90(a, k, (a.dim() - 2, a.dim() - 1)) for a in (dn, w, cm, ww))
                    if fl:
                        dn, w, cm, ww = (torch.flip(a, (a.dim() - 1,)) for a in (dn, w, cm, ww))
                dn = A.radiometric(dn, band_gain=cfg["aug_band_gain"], global_gain=cfg["aug_global_gain"],
                                   offset_dn=cfg["aug_offset_dn"], gamma=cfg["aug_gamma"],
                                   noise_dn=cfg["aug_noise_dn"], p_resample=cfg["aug_p_resample"])
                if cfg.get("aug_glint_p", 0) > 0:
                    dn = A.glint_aug(dn, gen, p=cfg["aug_glint_p"])
            if cfg.get("crop", 0):
                if kd is not None:     # crop the stride-4 KD maps with the same window (cells cut at the edge
                    c0 = (dn.shape[-1] - cfg["crop"]) // 2 // 4      # keep only their in-crop pixels)
                    c1 = c0 + cfg["crop"] // 4
                    cm, ww = cm[:, c0:c1, c0:c1], ww[:, c0:c1, c0:c1]
                dn, w = centre_crop(dn, w, cfg["crop"])
            x, x01 = normalise_input(dn, tr.sp[idx], model, cfg["input_mode"], cfg["rgb_only"])
            with torch.autocast("cuda", enabled=use_amp):
                out = model(x, x01 if model.needs_x01 else None, scene_cond(tr.sp[idx]))
            out = out.float()
            mu, var = pool_maps(out, w, T)
            loss = gaussian_nll(mu, var, ys[idx], mask[idx], beta=cfg["beta"])
            if kd is not None and cfg["kd_weight"] > 0:
                lkd = distill_nll(out, cm, ww, kd["t_mu_s"][idx], kd["t_var_s"][idx], kd["t_w"][idx], T,
                                  cfg["kd_sigma_scale2"])
                loss = loss + cfg["kd_weight"] * lkd
                tot_kd += float(lkd)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += float(loss)
        rec = {"epoch": ep + 1, "train_loss": tot / spe, "kd_loss": tot_kd / spe, "sec": time.time() - t0}
        if val is not None and val_y is not None and ((ep + 1) % 5 == 0 or ep + 1 == cfg["epochs"]):
            from .evaluate import mdsa_log

            mu_v, _ = predict(model, val, cfg)
            for j, t in enumerate(TARGETS):
                ok = np.isfinite(val_y[:, j])
                rec[f"val_mdsa_{t}"] = mdsa_log(val_y[ok, j], mu_v[ok, j]) if ok.any() else float("nan")
        hist.append(rec)
        log(json_line(rec))
    return model, hist


def json_line(rec: dict) -> str:
    return " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items())
