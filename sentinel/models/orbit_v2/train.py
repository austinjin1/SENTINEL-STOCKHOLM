# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Seed-unit training for architecture round v2 (orbit_arch_v2_amendment.md §3.2).

* :func:`run_epochs_ema` — the v1 fine-tuning loop (``orbit.satwq.run_epochs``: AdamW, OneCycle, grad clip 1.0,
  last partial batch dropped, inner-cell early stopping with patience) with three additions: EMA weights (decay
  0.999, started at the first fine-tuning step from the pretrained weights) that are used for the inner criterion
  and returned for inference; a per-target loss weight and the interval likelihood for left-censored values
  (§2.4, §3.4 item 8); checkpoint / resume so that one process stays below 2 h (§3.1).
* :func:`average_mixtures` / :func:`mixture_median` — a seed unit's predictive distribution is the equal-weight
  average of its networks' mixtures; its point prediction is the median of that averaged mixture.

v1 primitives (batching, prediction, quantiles, the MdSA function) are imported, never copied.
"""

from __future__ import annotations

import copy
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..orbit.evaluate import mdsa_log10
from ..orbit.satwq import mixture_quantiles, predict_quantiles

EMA_DECAY = 0.999
UNITS = (0, 1, 2)
INITS = (0, 1, 2)


def network_seed(unit: int, init: int) -> int:
    """Network seed = 100 x unit + init (§3.2)."""
    return 100 * int(unit) + int(init)


# ----------------------------------------------------------------------------------------- loss
def mixture_loss(logw, mu, sigma, y, ymask, cens=None, weights=None):
    """Masked mixture NLL. ``y`` [C, T] standardised log10 (for a censored entry: the standardised log10 detection
    limit); ``ymask`` [C, T] float; ``cens`` [C, T] bool (left-censored -> -log CDF(y)); ``weights`` [T].
    With ``cens`` and ``weights`` None this is ``orbit.satwq.masked_nll``."""
    yy = torch.nan_to_num(y, 0.0)[..., None]
    z = (yy - mu) / sigma
    lp = logw - 0.5 * z ** 2 - torch.log(sigma) - 0.5 * math.log(2 * math.pi)
    nll = -torch.logsumexp(lp, -1)
    if cens is not None and bool(cens.any()):
        lc = -torch.logsumexp(logw + torch.special.log_ndtr(z), -1)
        nll = torch.where(cens, lc, nll)
    if weights is None:
        return (nll * ymask).sum() / ymask.sum().clamp(min=1)
    w = ymask * weights[None, :]
    return (nll * w).sum() / w.sum().clamp(min=1e-6)


# ----------------------------------------------------------------------------------------- EMA
class EMA:
    """Exponential moving average of a model's parameters (buffers are copied)."""

    def __init__(self, model: nn.Module, decay: float = EMA_DECAY):
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for pe, p in zip(self.model.parameters(), model.parameters()):
            pe.mul_(d).add_(p.detach(), alpha=1.0 - d)
        for be, b in zip(self.model.buffers(), model.buffers()):
            be.copy_(b)

    def state_dict(self):
        return self.model.state_dict()


def _clone_state(sd: dict) -> dict:
    return {k: v.detach().clone() for k, v in sd.items()}


def _atomic_save(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def inner_criterion(q_log, inner_y_std, y_mean, y_std, select_idx, select_names="") -> tuple[float, int]:
    """v1 inner-cell criterion: mean MdSA over the select targets that have >= 30 inner rows."""
    sc = []
    for j in select_idx:
        y = inner_y_std[:, j] * y_std[j] + y_mean[j]
        m = np.isfinite(y)
        if m.sum() >= 30:
            sc.append(mdsa_log10(q_log[:, j, 0][m] - y[m]))
    if not sc:
        raise RuntimeError(f"early stopping impossible: no select target {select_names} has >= 30 inner rows")
    return float(np.mean(sc)), len(sc)


def run_epochs_ema(model, train_set, *, inner_set, epochs: int, lr: float, batch: int, device, seed: int, aug: dict,
                   y_mean, y_std, select_idx, weight_decay: float = 1e-4, eval_every: int = 1, patience: int = 15,
                   log=print, ema_decay: float = EMA_DECAY, loss_weights=None, ckpt: Path | None = None,
                   ckpt_every: int = 5, amp: bool = False, max_seconds: float | None = None) -> dict:
    """Fine-tune ``model``; on return it holds the EMA weights of the best inner epoch.

    Returns {'history', 'best_epoch', 'best_inner', 'complete'}; ``complete`` is False when ``max_seconds`` ran out
    (a checkpoint was written; call again with the same ``ckpt`` to continue)."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps = epochs * max(1, len(train_set) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(steps, 4),
                                                pct_start=min(0.3, max(0.05, 3.0 / max(steps, 4))))
    ema = EMA(model, ema_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best = (np.inf, None, -1)
    hist: list[dict] = []
    start = 0
    wt = None if loss_weights is None else torch.as_tensor(np.asarray(loss_weights), dtype=torch.float32, device=device)
    cens_all = getattr(train_set, "cens", None)
    if ckpt is not None and Path(ckpt).is_file():
        st = torch.load(ckpt, map_location=device, weights_only=False)
        if st["meta"] != {"epochs": epochs, "lr": lr, "batch": batch, "seed": seed, "n_train": len(train_set)}:
            raise RuntimeError(f"checkpoint {ckpt} belongs to another run: {st['meta']}")
        model.load_state_dict(st["model"])
        ema.model.load_state_dict(st["ema"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        scaler.load_state_dict(st["scaler"])
        rng.bit_generator.state = st["rng"]
        torch.set_rng_state(st["torch_rng"])
        if st.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(st["cuda_rng"])
        best, hist, start = tuple(st["best"]), st["hist"], st["epoch"] + 1
        log(f"  resumed from {ckpt} at epoch {start} (best {best[2]}: {best[0]:.2f})")
        if st.get("stopped"):
            start = epochs

    def save(ep: int, stopped: bool = False) -> None:
        if ckpt is None:
            return
        _atomic_save({"meta": {"epochs": epochs, "lr": lr, "batch": batch, "seed": seed, "n_train": len(train_set)},
                      "model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                      "sched": sched.state_dict(), "scaler": scaler.state_dict(), "rng": rng.bit_generator.state,
                      "torch_rng": torch.get_rng_state(),
                      "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                      "best": best, "hist": hist, "epoch": ep, "stopped": stopped}, Path(ckpt))

    t0 = time.time()
    complete = True
    stop = False
    for ep in range(start, epochs):
        model.train()
        perm = rng.permutation(len(train_set))
        tot, nb = 0.0, 0
        for s in range(0, len(perm) - batch + 1, batch):
            bi = perm[s:s + batch]
            xb, yb, mb = train_set.batch(bi, device, train=True, rng=rng, aug=aug)
            cb = None if cens_all is None else torch.as_tensor(cens_all[bi], dtype=torch.bool, device=device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                logw, mu, sg = model(**xb)
            loss = mixture_loss(logw.float(), mu.float(), sg.float(), yb, mb.float(), cb, wt)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            ema.update(model)
            tot += float(loss)
            nb += 1
        rec = {"epoch": ep, "train_nll": tot / max(nb, 1)}
        stop = False
        if inner_set is not None and (ep % eval_every == 0 or ep == epochs - 1):
            q = predict_quantiles(ema.model, inner_set, device, y_mean, y_std, qs=(0.5,),
                                  ctx_off=aug.get("ctx_off", False))
            rec["inner_mdsa"], rec["inner_targets_used"] = inner_criterion(q, inner_set.y, y_mean, y_std, select_idx)
            if rec["inner_mdsa"] < best[0]:
                best = (rec["inner_mdsa"], _clone_state(ema.state_dict()), ep)
            elif ep - best[2] > patience:
                stop = True
        hist.append(rec)
        if ep % 10 == 0:
            log(f"  ep {ep} nll {rec['train_nll']:.4f} inner(EMA) {rec.get('inner_mdsa', float('nan')):.2f} "
                f"({time.time() - t0:.0f}s)")
        if stop:
            log(f"  early stop at epoch {ep} (best {best[2]}: {best[0]:.2f})")
            save(ep, stopped=True)
            break
        if (ep + 1) % ckpt_every == 0:
            save(ep)
        if max_seconds is not None and time.time() - t0 > max_seconds and ep < epochs - 1:
            save(ep)
            complete = False
            log(f"  time budget reached after epoch {ep}; checkpoint written, resume to continue")
            break
    if complete:
        if start < epochs and not stop:
            save(epochs - 1, stopped=True)            # a second call with the same checkpoint is then a no-op
        model.load_state_dict(best[1] if best[1] is not None else ema.state_dict())
    return {"history": hist, "best_epoch": best[2], "best_inner": best[0], "complete": complete}


# ----------------------------------------------------------------------------------------- seed units
def average_mixtures(mixes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Equal-weight average of mixtures: list of (w, mu, sg) each [N, T, K] -> one mixture [N, T, n * K]."""
    n = len(mixes)
    if n == 0:
        raise ValueError("no mixtures")
    w = np.concatenate([np.asarray(m[0], float) / n for m in mixes], -1)
    mu = np.concatenate([np.asarray(m[1], float) for m in mixes], -1)
    sg = np.concatenate([np.asarray(m[2], float) for m in mixes], -1)
    return w, mu, sg


def mixture_median(mix, qs=(0.5,)) -> np.ndarray:
    """Quantiles [N, T, Q] of a mixture (w, mu, sg) [N, T, K] (exact CDF bisection of v1)."""
    w, mu, sg = mix
    return np.stack([mixture_quantiles(w[:, j], mu[:, j], sg[:, j], qs) for j in range(w.shape[1])], 1)


def mixture_cdf(mix, x_log10) -> np.ndarray:
    """P(y <= x) per row and target: mix (w, mu, sg) [N, T, K], x [N, T] -> [N, T]."""
    from scipy.special import ndtr

    w, mu, sg = (np.asarray(a, float) for a in mix)
    w = w / w.sum(-1, keepdims=True)
    return (w * ndtr((np.asarray(x_log10, float)[..., None] - mu) / sg)).sum(-1)
