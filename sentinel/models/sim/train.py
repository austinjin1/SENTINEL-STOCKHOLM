# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Training loop for SENTINEL-Sim (protocol §3.3).

Gradient updates use train-role (non-cal) sites × train-block target dates only. Random windows of
``seq_len`` days preceded by ``warmup`` input-only days (which may reach back to DATA_START). Early
stopping on the cal sites' train-block loss. For the DSST runs (``train_days`` = cool years) the loss days
are cool-year days only, but warm-up/context INPUTS (weather, discharge) may come from adjacent hot years
(documented in protocol §5.4) (unseen basin groups; the validation block is never
touched by training or model selection inside a run).
"""

from __future__ import annotations

import copy
import time

import numpy as np
import torch

from . import data as D
from .model import SimModel, batch_tensors, loss_fn, target_tensors

DEFAULTS = dict(hidden=128, static_embed=32, dropout=0.2, n_layers=1, lr=1e-3, weight_decay=1e-5,
                batch=64, seq_len=365, warmup=180, batches_per_epoch=150, max_epochs=40, patience=6,
                k_prior_w=0.01, grad_clip=1.0, seed=0, cal_batch=32, target_w=None, metab_struct=False, drop_dyn=[])


def _window_sampler(d: D.SimData, idx: np.ndarray, valid_days: np.ndarray, seq_len: int, warmup: int,
                    rs: np.random.Generator):
    """Yield (site, t0) with the loss window [t0, t0+seq_len) inside ``valid_days`` and ≥ 30 targets."""
    vd = np.flatnonzero(valid_days)
    lo, hi = max(vd[0], warmup), vd[-1] - seq_len + 1
    cnt = np.isfinite(d.y[idx][:, valid_days]).any(axis=-1).sum(axis=1).astype(float)
    p = cnt / cnt.sum()
    while True:
        i = idx[rs.choice(len(idx), p=p)]
        t0 = int(rs.integers(lo, hi + 1))
        if (np.isfinite(d.y[i, t0:t0 + seq_len]).any(axis=-1) & valid_days[t0:t0 + seq_len]).sum() >= 30:
            yield i, t0


def evaluate_loss(model, d, idx, valid_days, cfg, device) -> float:
    """Mean data loss over full sequences of ``idx`` sites restricted to ``valid_days`` (chunks)."""
    model.eval()
    static_z = D.normalised_static(d)
    vd = np.flatnonzero(valid_days)
    tot, nb = 0.0, 0
    with torch.no_grad():
        for b0 in range(0, len(idx), cfg["cal_batch"]):
            ii = idx[b0:b0 + cfg["cal_batch"]]
            for c0 in range(vd[0], vd[-1] + 1, cfg["seq_len"]):
                c1 = min(c0 + cfg["seq_len"], vd[-1] + 1)
                w0 = max(0, c0 - cfg["warmup"])
                bt = batch_tensors(d, ii, w0, c1, static_z, device)
                y, m = target_tensors(d, ii, w0, c1, valid_days & (np.arange(len(valid_days)) >= c0), device)
                if m.sum() == 0:
                    continue
                out = model(**bt)
                _, parts = loss_fn(out, y, m, model.y_mean, model.y_std, k_prior_w=0.0, target_w=cfg["target_w"])
                tot += parts["data"]
                nb += 1
    return tot / max(nb, 1)


def train(d: D.SimData, cfg: dict, device, *, log=print, train_idx=None, train_days=None):
    cfg = {**DEFAULTS, **(cfg or {})}
    torch.manual_seed(cfg["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    rs = np.random.default_rng(cfg["seed"])
    tr = d.idx("train") if train_idx is None else np.asarray(train_idx)
    cal = d.idx("cal")
    tdays = D.block_mask(d.dates, "train") if train_days is None else train_days
    model = SimModel(d.norm, n_dyn=len(D.DYN_NAMES), n_static=len(D.STATIC_NAMES), hidden=cfg["hidden"],
                     static_embed=cfg["static_embed"], dropout=cfg["dropout"], n_layers=cfg["n_layers"],
                     metab_struct=cfg.get("metab_struct", False), drop_dyn=cfg.get("drop_dyn", [])).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    static_z = D.normalised_static(d)
    sampler = _window_sampler(d, tr, tdays, cfg["seq_len"], cfg["warmup"], rs)
    best, best_state, bad, hist = np.inf, None, 0, []
    L, W = cfg["seq_len"], cfg["warmup"]
    for ep in range(cfg["max_epochs"]):
        model.train()
        t_ep = time.time()
        losses = []
        for _ in range(cfg["batches_per_epoch"]):
            picks = [next(sampler) for _ in range(cfg["batch"])]
            # group by identical offsets is unnecessary: build per-window tensors and stack
            bts, ys, ms = [], [], []
            for i, t0 in picks:
                w0 = t0 - W
                bts.append(batch_tensors(d, [i], w0, t0 + L, static_z, device))
                lossdays = tdays.copy()
                lossdays[: t0] = False
                y, m = target_tensors(d, [i], w0, t0 + L, lossdays, device)
                ys.append(y)
                ms.append(m)
            bt = {k: torch.cat([b[k] for b in bts]) for k in bts[0]}
            y, m = torch.cat(ys), torch.cat(ms)
            out = model(**bt)
            loss, parts = loss_fn(out, y, m, model.y_mean, model.y_std, k_prior_w=cfg["k_prior_w"],
                                  target_w=cfg["target_w"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            losses.append(parts["data"])
        cal_loss = evaluate_loss(model, d, cal, tdays, cfg, device)
        sched.step(cal_loss)
        hist.append({"epoch": ep, "train": float(np.mean(losses)), "cal": cal_loss,
                     "lr": opt.param_groups[0]["lr"], "sec": time.time() - t_ep})
        log(f"[sim.train] ep {ep:02d} train {np.mean(losses):.4f} cal {cal_loss:.4f} "
            f"lr {opt.param_groups[0]['lr']:.1e} {time.time() - t_ep:.0f}s")
        if np.isfinite(cal_loss) and cal_loss < best - 1e-4:
            best, best_state, bad = cal_loss, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    if best_state is None:
        raise RuntimeError("cal loss was never finite — training failed (no checkpoint selected)")
    model.load_state_dict(best_state)
    return model, {"best_cal_loss": best, "history": hist, "epochs_run": len(hist), "config": cfg}
