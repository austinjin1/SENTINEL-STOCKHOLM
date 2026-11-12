# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Resumable trainer for SONDE-HF and R-NH (≤ 2 h GPU processes; lead rule). Training data: in-fold sites × TRAIN
block only (lead ruling L3); early stopping on an inner hold-out of TRAIN-role basin groups × TRAIN block (never the
val block, amendment §2h). A checkpoint (model, optimiser, scaler, epoch, best state, RNG) is written after every
epoch; when ``max_minutes`` is reached the run stops after the current epoch and returns ``complete=False`` (the
caller exits 75 and the same command resumes). The optimiser is AdamW(lr = cfg.lr, weight_decay = cfg.weight_decay)
in EVERY stage and for both networks; after a resume both values are re-applied to every parameter group, so a
saved optimiser state can never change them. Stage 2 starts from ``init`` (stage-1 best weights) with a fresh
optimiser and trains on the ``use="real"`` product. Early-stopping rows: a seeded sample of ``es_max_samples``.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from . import nets as N


@dataclass
class TrainConfig:
    kind: str = "sonde_hf"            # "sonde_hf" | "rnh"
    targets: list = field(default_factory=list)
    rnh_target: str | None = None
    hourly: bool = True
    head: str = "quantile"            # SONDE-HF: "quantile" | "cmal" (C4); R-NH: "cmal" | "quantile"
    multitask: bool = True            # False = primary-only loss (C3)
    hidden_d: int = 256
    hidden_h: int = 128
    hidden_f: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 256
    steps_per_epoch: int = 800
    max_epochs: int = 30
    patience: int = 4
    es_max_samples: int = 30_000
    grad_clip: float = 1.0
    amp: bool = True
    seed: int = 0
    primary: list = field(default_factory=lambda: ["temp_c_mean", "do_mgl_mean", "do_mgl_min", "spcond_uscm_mean",
                                                   "turb_fnu_mean"])


ORDER = [("do_mgl_min", "do_mgl_mean", -1), ("do_mgl_max", "do_mgl_mean", 1), ("temp_c_min", "temp_c_mean", -1),
         ("temp_c_max", "temp_c_mean", 1)]


def build_model(cfg: TrainConfig, ten) -> torch.nn.Module:
    st = ten.st
    prod = ten.products["train"]
    if not isinstance(prod, np.ndarray):
        raise ValueError("exactly one declared NWP product is allowed")
    fields = [2 * prod.shape[-1]]                                    # standardised fields + availability flags
    if cfg.kind == "rnh":
        return N.RNH(st.n_dyn, st.static.shape[1], st.n_huc, fields, L=ten.L, hidden=cfg.hidden_f, head=cfg.head)
    tg = ten.variables
    order = [o for o in ORDER if o[0] in tg and o[1] in tg]
    n_h = ten.hourly.shape[-1] if (cfg.hourly and ten.hourly is not None) else 0
    return N.SondeHF(st.n_dyn, st.static.shape[1], st.n_huc, fields, tg, n_hourly=n_h, L=ten.L,
                     hidden_d=cfg.hidden_d, hidden_h=cfg.hidden_h, hidden_f=cfg.hidden_f, order=order,
                     anchor_targets=tg, head=cfg.head)


def _to(b: dict, dev, cfg: TrainConfig):
    t = lambda x, dt=torch.float32: torch.as_tensor(x, device=dev, dtype=dt)  # noqa: E731
    out = {"x_dyn": t(b["x_dyn"]), "x_static": t(b["x_static"]), "huc": t(b["huc"], torch.long),
           "x_fdet": t(b["x_fdet"]), "y": t(b["y"]), "nwp_x": [t(x) for x in b["nwp_x"]],
           "nwp_m": [t(m, torch.bool) for m in b["nwp_m"]], "a_last": t(b["a_last"]),
           "clim_mean": t(b["clim_mean"]), "clim_sd": t(b["clim_sd"])}
    if "x_hourly" in b and cfg.hourly and cfg.kind == "sonde_hf":
        out["x_hourly"], out["has_hourly"] = t(b["x_hourly"]), t(b["has_hourly"], torch.bool)
    return out


def forward(model, cfg: TrainConfig, bt: dict, ten):
    if cfg.kind == "rnh":
        return model(bt["x_dyn"], bt["x_static"], bt["huc"], bt["nwp_x"], bt["nwp_m"], bt["x_fdet"])
    return model(bt["x_dyn"], bt["x_static"], bt["huc"], bt["nwp_x"], bt["nwp_m"], bt["x_fdet"],
                 x_hourly=bt.get("x_hourly"), has_hourly=bt.get("has_hourly"), a_last=bt["a_last"],
                 clim_mean=bt["clim_mean"], clim_sd=bt["clim_sd"])


def loss_fn(out, cfg: TrainConfig, bt: dict, ten) -> torch.Tensor:
    y = bt["y"]
    if cfg.kind == "rnh":
        k = ten.variables.index(cfg.rnh_target)
        yk = y[..., k:k + 1]
        return N.CMALHead.nll(out, yk) if cfg.head == "cmal" else N.pinball(out, yk)
    if not cfg.multitask:                 # C3: loss on the primary targets only (every other target unlabelled)
        keep = torch.tensor([v in cfg.primary for v in ten.variables], device=y.device)
        y = torch.where(keep, y, torch.full_like(y, float("nan")))
    return N.CMALHead.nll(out, y) if cfg.head == "cmal" else N.pinball(out, y)


def loss_targets(cfg: TrainConfig, ten) -> list:
    """The targets whose labels enter the loss of this configuration (training rows need ≥ 1 of them)."""
    if cfg.kind == "rnh":
        return [cfg.rnh_target]
    return list(ten.variables) if cfg.multitask else [v for v in ten.variables if v in cfg.primary]


def train(cfg: TrainConfig, ten, train_si, train_ti, es_si, es_ti, ckpt: Path, *, device: str = "cuda",
          max_minutes: float | None = None, log=print, init: Path | None = None, use: str = "train") -> dict:
    """``init``: state dict loaded at a FRESH start only (stage 2 = stage-1 best weights); ``use``: the registered
    product the batches read ("train" = perfect-prog, "real" = archived forecasts)."""
    t0 = time.time()
    torch.manual_seed(cfg.seed)
    model = build_model(cfg, ten).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.startswith("cuda"))
    state = {"epoch": 0, "best": float("inf"), "bad": 0, "history": []}
    ckpt = Path(ckpt)
    if (ckpt / "last.pt").exists():
        c = torch.load(ckpt / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(c["model"])
        opt.load_state_dict(c["opt"])
        scaler.load_state_dict(c["scaler"])
        state = c["state"]
        log(f"[train] resumed at epoch {state['epoch']} (best {state['best']:.5f})")
    elif init is not None:
        model.load_state_dict(torch.load(init, map_location=device))
        log(f"[train] initialised from {init}")
    for g in opt.param_groups:            # the configuration, not a saved optimiser state, fixes these two
        g["lr"], g["weight_decay"] = cfg.lr, cfg.weight_decay
    state["optimizer"] = {"name": "AdamW", "lr": cfg.lr, "weight_decay": cfg.weight_decay, "use": use,
                          "es_max_samples": cfg.es_max_samples, "n_es_rows": int(min(cfg.es_max_samples, es_si.size)),
                          "n_train_rows": int(train_si.size)}
    ckpt.mkdir(parents=True, exist_ok=True)
    rs = np.random.default_rng(cfg.seed + 1000 * state["epoch"])
    es_pick = np.random.default_rng(cfg.seed).choice(es_si.size, min(cfg.es_max_samples, es_si.size), replace=False)
    es_si, es_ti = es_si[np.sort(es_pick)], es_ti[np.sort(es_pick)]
    while state["epoch"] < cfg.max_epochs and state["bad"] < cfg.patience:
        model.train()
        tl = []
        for _ in range(cfg.steps_per_epoch):
            j = rs.integers(0, train_si.size, cfg.batch_size)
            bt = _to(ten.batch(train_si[j], train_ti[j], use), device, cfg)
            with torch.autocast("cuda", dtype=torch.float16, enabled=cfg.amp and device.startswith("cuda")):
                out = forward(model, cfg, bt, ten)
            loss = loss_fn(out, cfg, bt, ten)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            tl.append(float(loss.detach()))
        es = evaluate_loss(model, cfg, ten, es_si, es_ti, device, use=use)
        state["epoch"] += 1
        state["history"].append({"epoch": state["epoch"], "train": float(np.mean(tl)), "es": es,
                                 "minutes": (time.time() - t0) / 60})
        if es < state["best"] - 1e-5:
            state["best"], state["bad"] = es, 0
            torch.save(model.state_dict(), ckpt / "best.pt")
        else:
            state["bad"] += 1
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                    "state": state, "cfg": asdict(cfg)}, ckpt / ".last.tmp")
        (ckpt / ".last.tmp").replace(ckpt / "last.pt")
        log(f"[train] epoch {state['epoch']} train {np.mean(tl):.5f} es {es:.5f} best {state['best']:.5f}")
        if max_minutes and (time.time() - t0) / 60 > max_minutes:
            return {"complete": False, **state}
    (ckpt / "history.json").write_text(json.dumps(state, indent=1))
    return {"complete": True, **state}


@torch.no_grad()
def evaluate_loss(model, cfg, ten, si, ti, device, bs: int = 1024, use: str = "train") -> float:
    model.eval()
    tot, n = 0.0, 0
    for a in range(0, si.size, bs):
        bt = _to(ten.batch(si[a:a + bs], ti[a:a + bs], use), device, cfg)
        with torch.autocast("cuda", dtype=torch.float16, enabled=cfg.amp and device.startswith("cuda")):
            out = forward(model, cfg, bt, ten)
        tot += float(loss_fn(out, cfg, bt, ten)) * len(si[a:a + bs])
        n += len(si[a:a + bs])
    return tot / max(n, 1)


@torch.no_grad()
def predict(model, cfg, ten, si, ti, use: str, device, bs: int = 1024, *, leads=None, var_idx=None) -> np.ndarray:
    """(n, L, V_out, 19) quantiles in MODEL space (original units, or log for log variables). ``leads`` (0-based
    lead indices) and ``var_idx`` (SONDE-HF output indices) subset the result before it is stacked."""
    model.eval()
    outs = []
    for a in range(0, si.size, bs):
        s, t = si[a:a + bs], ti[a:a + bs]
        b = ten.batch(s, t, use)
        bt = _to(b, device, cfg)
        out = forward(model, cfg, bt, ten)
        q = model.quantiles(out) if cfg.kind == "rnh" else model.quantiles(out, bt["clim_mean"], bt["clim_sd"])
        q = q.float().cpu().numpy()
        if cfg.kind == "rnh":
            k = ten.variables.index(cfg.rnh_target)
            q = b["clim_mean"][..., k:k + 1, None] + b["clim_sd"][..., k:k + 1, None] * q
        else:
            q = b["clim_mean"][..., None] + b["clim_sd"][..., None] * q
        ok = b["clim_ok"][..., k:k + 1] if cfg.kind == "rnh" else b["clim_ok"]
        q = np.where(ok[..., None], q, np.nan)                      # no climatology: not forecastable
        q = np.sort(q, axis=-1).astype(np.float32)
        if leads is not None:
            q = q[:, list(leads)]
        if var_idx is not None and cfg.kind != "rnh":
            q = q[:, :, list(var_idx)]
        outs.append(q)
    return np.concatenate(outs)
