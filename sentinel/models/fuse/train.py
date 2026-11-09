# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Training loop for SENTINEL-Fuse (Stage 3 supervised fine-tuning with the modality-dropout schedule).

Modality-dropout schedule (RESEARCH_PLAN §6, docs/research/05 §e.3), per training row after a
warm-up of ``warmup_epochs`` with all available modalities:
  * with probability ``p_all`` (0.2) keep every available modality;
  * with probability ``p_single`` (0.2) keep exactly one available non-hub modality (+ hub) — this is
    the unimodal auxiliary task through the shared decoder;
  * otherwise drop each available non-hub modality independently with p ~ U(``p_drop_lo``, ``p_drop_hi``).
Hub modalities (forcings, catchment, time) are never dropped. Batches are sampled with weights
∝ frequency(availability pattern)^-``pattern_alpha`` so hub-heavy patterns do not dominate.
Early stopping uses an inner validation set made of whole training sites (never evaluation rows).
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from .model import FuseConfig, FuseModel, masked_loss, n_parameters


@dataclass
class TrainConfig:
    epochs: int = 40
    batch: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 2
    p_all: float = 0.2
    p_single: float = 0.2
    p_drop_lo: float = 0.3
    p_drop_hi: float = 0.5
    pattern_alpha: float = 0.5
    beta_nll: float = 0.5
    patience: int = 6
    amp: bool = True
    grad_clip: float = 1.0
    steps_per_epoch: int | None = None
    seed: int = 0


def to_torch(tens: dict, device) -> dict:
    out = {}
    for m, d in tens.items():
        out[m] = {"x": torch.as_tensor(d["x"], device=device), "m": torch.as_tensor(d["m"], device=device),
                  "present": torch.as_tensor(d["present"], device=device),
                  "age": torch.as_tensor(d["age"], device=device)}
    return out


def take(tt: dict, idx: torch.Tensor) -> dict:
    return {m: {k: v[idx] for k, v in d.items()} for m, d in tt.items()}


def dropout_keep(batch: dict, hub: set, rng: torch.Generator, cfg: TrainConfig, active: bool) -> dict | None:
    if not active:
        return None
    mods = [m for m in batch if m not in hub]
    if not mods:
        return None
    dev = batch[mods[0]]["present"].device
    B = batch[mods[0]]["present"].shape[0]
    P = torch.stack([batch[m]["present"].bool() for m in mods], 1)          # [B, M]
    u = torch.rand(B, generator=rng, device="cpu").to(dev)
    mode_all = u < cfg.p_all
    mode_single = (u >= cfg.p_all) & (u < cfg.p_all + cfg.p_single)
    # single: choose one present modality uniformly
    score = torch.rand(B, len(mods), generator=rng, device="cpu").to(dev) * P.float() - (~P).float()
    choice = score.argmax(1)
    single = torch.zeros_like(P)
    single[torch.arange(B, device=dev), choice] = True
    single &= P
    pdrop = (cfg.p_drop_lo + (cfg.p_drop_hi - cfg.p_drop_lo) * torch.rand(B, 1, generator=rng, device="cpu")).to(dev)
    indep = (torch.rand(B, len(mods), generator=rng, device="cpu").to(dev) >= pdrop) & P
    keep = torch.where(mode_all.view(-1, 1), P, torch.where(mode_single.view(-1, 1), single, indep))
    return {m: keep[:, i] for i, m in enumerate(mods)}


def pattern_weights(tens: dict, hub: set, alpha: float) -> np.ndarray:
    mods = [m for m in tens if m not in hub]
    if not mods:
        return np.ones(len(next(iter(tens.values()))["present"]))
    P = np.column_stack([tens[m]["present"] for m in mods]).astype(np.int64)
    key = (P * (1 << np.arange(P.shape[1]))).sum(1)
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    w = cnt[inv].astype(float) ** (-alpha)
    return w / w.sum()


def take_anchor(anc: dict | None, idx: torch.Tensor) -> dict | None:
    if anc is None:
        return None
    return {"z": anc["z"][idx], "ok": anc["ok"][idx], "owner": anc["owner"]}


def anchor_to_torch(anchor_z: np.ndarray | None, anchor_ok: np.ndarray | None, owners: list | None, device):
    if anchor_z is None:
        return None
    return {"z": torch.as_tensor(np.nan_to_num(anchor_z), dtype=torch.float32, device=device),
            "ok": torch.as_tensor(anchor_ok, device=device), "owner": owners}


def evaluate_loss(model, tt, Y, M, C, is_bin, idx, batch, beta, device, anc=None) -> float:
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for s in range(0, len(idx), batch * 4):
            b = idx[s:s + batch * 4]
            bt = torch.as_tensor(b, device=device)
            out = model(take(tt, bt), None, take_anchor(anc, bt)).float()
            loss, _ = masked_loss(out, Y[bt], M[bt], C[bt], is_bin, beta=0.0)
            tot += float(loss) * len(b)
            n += len(b)
    return tot / max(n, 1)


def predict(model, tt, idx, batch, device, keep_fn=None, anc=None) -> np.ndarray:
    model.eval()
    outs = []
    with torch.no_grad():
        for s in range(0, len(idx), batch * 4):
            b = torch.as_tensor(idx[s:s + batch * 4], device=device)
            bt = take(tt, b)
            keep = keep_fn(bt) if keep_fn is not None else None
            outs.append(model(bt, keep, take_anchor(anc, b)).float().cpu().numpy())
    return np.concatenate(outs, 0) if outs else np.zeros((0, len(model.targets), 2), np.float32)


def train_fuse(tens: dict, Yz: np.ndarray, M: np.ndarray, C: np.ndarray, is_bin: np.ndarray, tr_idx: np.ndarray,
               va_idx: np.ndarray, group_sizes: dict, targets: list, hub: set, mcfg: FuseConfig,
               tcfg: TrainConfig, device: str = "cpu", log=print, anc: dict | None = None) -> tuple[FuseModel, dict]:
    """Train on rows ``tr_idx`` (standardised targets Yz), early-stop on ``va_idx``; returns best model."""
    torch.manual_seed(tcfg.seed)
    np.random.seed(tcfg.seed)
    rng = torch.Generator().manual_seed(tcfg.seed)
    model = FuseModel(mcfg, group_sizes, targets).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=tcfg.lr,
                            weight_decay=tcfg.weight_decay)
    use_amp = tcfg.amp and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    tt = to_torch(tens, device)
    Yt = torch.as_tensor(np.nan_to_num(Yz), dtype=torch.float32, device=device)
    Mt = torch.as_tensor(M, device=device)
    Ct = torch.as_tensor(C, device=device)
    Bt = torch.as_tensor(is_bin, device=device)
    w = pattern_weights({m: {"present": d["present"][tr_idx]} for m, d in tens.items()}, hub, tcfg.pattern_alpha)
    steps = tcfg.steps_per_epoch or max(1, len(tr_idx) // tcfg.batch)
    total_steps = steps * tcfg.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=tcfg.lr, total_steps=total_steps, pct_start=0.1)
    best, best_state, bad, hist = math.inf, None, 0, []
    t0 = time.time()
    for ep in range(tcfg.epochs):
        model.train()
        draw = np.random.default_rng(tcfg.seed * 1000 + ep).choice(len(tr_idx), size=steps * tcfg.batch, p=w)
        run = 0.0
        for st in range(steps):
            b = torch.as_tensor(tr_idx[draw[st * tcfg.batch:(st + 1) * tcfg.batch]], device=device)
            bt = take(tt, b)
            keep = dropout_keep(bt, hub, rng, tcfg, active=ep >= tcfg.warmup_epochs)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                out = model(bt, keep, take_anchor(anc, b))
            loss, _ = masked_loss(out.float(), Yt[b], Mt[b], Ct[b], Bt, beta=tcfg.beta_nll)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += float(loss)
        vl = evaluate_loss(model, tt, Yt, Mt, Ct, Bt, va_idx, tcfg.batch, 0.0, device, anc)
        hist.append({"epoch": ep, "train_loss": run / steps, "val_loss": vl, "sec": round(time.time() - t0, 1)})
        log(f"    ep {ep:02d} train {run / steps:.4f} val {vl:.4f} ({time.time() - t0:.0f}s)")
        if vl < best - 1e-4:
            best, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= tcfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    info = {"best_val_loss": best, "epochs_run": len(hist), "history": hist,
            "n_params": n_parameters(model), "model_cfg": asdict(mcfg), "train_cfg": asdict(tcfg)}
    return model, info
