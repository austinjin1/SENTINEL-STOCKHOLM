# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Training and prediction of the neural arms and comparators of M7 architecture round v2 (amendment §3.3, §4.3–
§4.5, §5): ANCHOR-Lite, ANCHOR-Set, the ablations A1 / A3 / A4 / A5 and the DenseNet121 comparators P3a / P3b.

Fixed epoch budgets; no early stopping; no validation row is seen by any function here except :func:`predict`.
The loss works in log10 units:

    Σ_t w_t · L_t + kd_weight · KD + 0.01 · mean(Δ²)

``L_t`` is the mean over the batch rows that carry a term for target t of: the β-NLL (β = 0.5) of an uncensored
value; the Tobit term of a censored row, multiplied by the detached σ^(2β) like the β-NLL (§3.3); and, for the NTU
head, the β-NLL of an NTRU value against μ + one learned additive offset (§3.2). w = 1, except 0.5 for SSC and
chl-a. KD is the frozen ``distill_nll`` on the leader's map head (ANCHOR-Lite family) or a squared error on the
chip-mean teacher μ (P3b); KD tensors on Test-S pairings are removed before training (ruling F4).

Readings decided here (listed for the lead in the implementation notes of this round):
* the target terms are summed (Σ_t w_t L_t as written), not averaged over targets;
* the leader's modules (encoder, decoder, map head) train at 5e-4, every new module at 1e-3; weight decay 1e-4,
  2 warm-up epochs + cosine, gradient clip 5.0 and AMP as config 3;
* scene statistics under the radiometric augmentation: a quantile q becomes f(q) with f the same gamma / gain /
  offset map as the pixels; the mean becomes f(mean); the std becomes (f(mean + std) − f(mean − std)) / 2; the
  noise and the block-resampling steps leave them unchanged;
* A1 and the DenseNet heads start at the train median with log σ² = log train variance; the DenseNet heads are not
  standardised; the Δ² penalty stays in A1 (it is part of the unchanged loss);
* ANCHOR-Set: no radiometric augmentation (§4.4 names only the pixel resampling); schedule as the other arms;
  log10(DN + 1) for the pixel logarithms; pixel features standardised with training-chip statistics;
* P3b's chip-level KD term is 0.5 · squared error on the teacher chip means standardised with train-row statistics.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as Fn

from ..lite import features as F
from ..lite.losses import distill_nll
from . import labels2b as L
from . import nets as NN
from . import tabular as T

TARGET_WEIGHT = {"ssc": 0.5, "chla": 0.5}                     # 1 for the other targets (§4.3)
BASE_CFG = dict(epochs=40, batch=32, lr=1e-3, lr_backbone=5e-4, wd=1e-4, warmup_epochs=2, beta=0.5, kd_weight=0.3,
                kd_sigma_scale2=0.01, delta_reg=0.01, amp=True, aug=True, aug_band_gain=0.10, aug_global_gain=0.15,
                aug_offset_dn=3.0, aug_gamma=0.10, aug_noise_dn=1.5, aug_p_resample=0.2, grad_clip=5.0)
ARMS = {
    "anchor_lite": dict(net="anchor"),
    "A1": dict(net="anchor", use_anchor=False),
    "A3": dict(net="anchor", use_context=False),
    "A4": dict(net="anchor", targets=tuple(L.ORIG), reduced=True),
    "A5": dict(net="anchor", attention=False),
    "anchor_set": dict(net="set", batch=64, kd_weight=0.0, aug=False),
    "P3a_densenet": dict(net="densenet", parity=False, epochs=80, kd_weight=0.0, delta_reg=0.0),
    "P3b_densenet_parity": dict(net="densenet", parity=True, epochs=80, delta_reg=0.0),
}
SET_SALT = "SENTINEL-Lite-set|"
N_STATS = len(T.SCENE_STATS)                                   # 12 statistics per band, 4 bands, then 2 fractions


def arm_config(arm: str, **over) -> dict:
    if arm not in ARMS:
        raise KeyError(f"unknown arm {arm!r} (known: {sorted(ARMS)})")
    cfg = {**BASE_CFG, "targets": tuple(L.TARGETS), "use_anchor": True, "use_context": True, "attention": True,
           "reduced": False, "parity": False, **ARMS[arm], "arm": arm}
    cfg.update({k: v for k, v in over.items() if v is not None})
    return cfg


# ------------------------------------------------------------------------------------------------ tensors
@dataclass
class Tensors:
    """One set of rows on the CPU (moved to the device batch by batch)."""
    dn: torch.Tensor                    # uint8 (N, 4, H, W)
    pool_w: torch.Tensor                # (N, h, w) frozen centre-window pooling weights
    water: torch.Tensor                 # (N, h, w) water fraction per output cell
    sc: torch.Tensor                    # (N, 56) [s, c], not standardised, NaN allowed
    gsd: torch.Tensor                   # (N,)
    anchor: torch.Tensor | None         # (N, T) log10
    y: torch.Tensor                     # (N, T) log10, NaN = no uncensored value
    y_ntru: torch.Tensor                # (N,) log10, NaN
    cens: torch.Tensor                  # (N, T) -1 / 0 / +1
    lim: torch.Tensor                   # (N, T) log10 limit
    uid: list = field(default_factory=list)
    kd: dict | None = None              # cell_map, water_w, t_mu, t_var, t_w (training rows only)
    teacher: torch.Tensor | None = None  # (N, 4) chip-mean teacher μ (P3b), NaN = none
    water_px: torch.Tensor | None = None  # uint8 (N, H, W) frozen water mask (ANCHOR-Set)
    shore_px: torch.Tensor | None = None  # float16 (N, H, W) distance to the nearest non-water pixel, in pixels

    @property
    def n(self) -> int:
        return int(self.dn.shape[0])


def build_tensors(rows: pd.DataFrame, chips: np.ndarray, sc: pd.DataFrame, anchors: pd.DataFrame | None, targets,
                  *, reduced: bool = False, with_set: bool = False) -> Tensors:
    """Tensors of ``rows``. ``reduced`` (A4): uncensored labels of ``targets`` only — no censored row, no NTRU."""
    targets = tuple(targets)
    chips = np.ascontiguousarray(chips)
    n, _, H, W = chips.shape
    gsd = rows["gsd_m"].to_numpy(np.float32)
    wm = F.water_mask(chips)
    water = wm.reshape(n, H // 4, 4, W // 4, 4).mean(axis=(2, 4)).astype(np.float32)
    y = rows[[f"y_{t}" for t in targets]].to_numpy(np.float32)
    cens = np.zeros((n, len(targets)), np.int8)
    lim = np.full((n, len(targets)), np.nan, np.float32)
    y_ntru = np.full(n, np.nan, np.float32)
    if not reduced:
        cens = rows[[f"cens_{t}" for t in targets]].to_numpy(np.int8)
        lim = rows[[f"lim_{t}" for t in targets]].to_numpy(np.float32)
        if "turbidity" in targets:
            y_ntru = rows["y_turb_ntru"].to_numpy(np.float32)
    if (np.isfinite(y) & (cens != 0)).any() or ((cens != 0) & ~np.isfinite(lim)).any():
        raise ValueError("a censored entry has an uncensored value or no limit")
    a = None
    if anchors is not None:
        a = anchors[list(targets)].to_numpy(np.float32)
        if not np.isfinite(a).all():
            raise ValueError("an anchor is missing")
    if sc.shape[1] != NN.N_SC:
        raise ValueError(f"[s, c] has {sc.shape[1]} columns")
    out = Tensors(dn=torch.from_numpy(chips), pool_w=torch.from_numpy(F.pooling_weights(chips, gsd, stride=4)),
                  water=torch.from_numpy(water), sc=torch.from_numpy(sc.to_numpy(np.float32)),
                  gsd=torch.from_numpy(gsd), anchor=None if a is None else torch.from_numpy(a),
                  y=torch.from_numpy(y), y_ntru=torch.from_numpy(y_ntru), cens=torch.from_numpy(cens),
                  lim=torch.from_numpy(lim), uid=rows["uid"].astype(str).tolist())
    if with_set:
        from scipy.ndimage import distance_transform_edt

        shore = np.zeros((n, H, W), np.float16)
        for i in range(n):
            if wm[i].any():
                shore[i] = distance_transform_edt(wm[i]).astype(np.float16)
        out.water_px = torch.from_numpy(wm.astype(np.uint8))
        out.shore_px = torch.from_numpy(shore)
    return out


def attach_kd(t: Tensors, kd: dict, kd_masked: np.ndarray) -> dict:
    """KD tensors of ``lite.distill.load_kd_tensors`` with every Test-S pairing removed (F4). Returns counts."""
    m = torch.from_numpy(np.asarray(kd_masked, bool))
    t_mu = torch.as_tensor(np.asarray(kd["t_mu"]), dtype=torch.float32).clone()
    t_var = torch.as_tensor(np.asarray(kd["t_var"]), dtype=torch.float32).clone()
    t_w = torch.as_tensor(np.asarray(kd["t_w"]), dtype=torch.float32).clone()
    before = int((t_w.flatten(1).sum(1) > 0).sum())
    t_mu[m], t_var[m], t_w[m] = float("nan"), float("nan"), 0.0
    if bool((t_w[m] > 0).any()) or bool(torch.isfinite(t_mu[m]).any()):
        raise RuntimeError("a teacher value survived the Test-S mask")
    t.kd = {"cell_map": torch.as_tensor(np.asarray(kd["cell_map"])).to(torch.int16),
            "water_w": torch.as_tensor(np.asarray(kd["water_w"]), dtype=torch.float32),
            "t_mu": t_mu, "t_var": t_var, "t_w": t_w}
    return {"n_chips_with_teacher_before_mask": before, "n_kd_masked": int(m.sum()),
            "n_chips_with_teacher": int((t_w.flatten(1).sum(1) > 0).sum())}


# ------------------------------------------------------------------------------------------------ augmentation
def radiometric(x: torch.Tensor, cfg: dict) -> tuple[torch.Tensor, dict]:
    """The frozen ``augment.radiometric`` (same draws in the same order) that also returns its per-band map."""
    N, dev = x.shape[0], x.device
    gb = 1 + cfg["aug_band_gain"] * (2 * torch.rand(N, 4, 1, 1, device=dev) - 1)
    gg = 1 + cfg["aug_global_gain"] * (2 * torch.rand(N, 1, 1, 1, device=dev) - 1)
    off = cfg["aug_offset_dn"] * torch.randn(N, 4, 1, 1, device=dev)
    gm = 1 + cfg["aug_gamma"] * (2 * torch.rand(N, 1, 1, 1, device=dev) - 1)
    y = 255.0 * (x.clamp(0, 255) / 255.0) ** gm
    y = y * gb * gg + off + cfg["aug_noise_dn"] * torch.randn_like(y)
    if cfg["aug_p_resample"] > 0:
        sel = torch.rand(N, device=dev) < cfg["aug_p_resample"]
        if sel.any():
            f = int(torch.randint(1, 3, (1,))) * 2
            ys = Fn.interpolate(Fn.avg_pool2d(y[sel], f), scale_factor=f, mode="nearest")
            y = y.clone()
            y[sel] = ys
    return y.clamp(0, 255), {"gamma": gm.view(N, 1), "gain": (gb * gg).view(N, 4), "offset": off.view(N, 4)}


def _band_map(v: torch.Tensor, p: dict) -> torch.Tensor:
    """f(v) for v (N, 4, k): 255 · (v / 255)^γ · gain + offset, clamped to the DN range."""
    g = p["gamma"].view(-1, 1, 1)
    return (255.0 * (v.clamp(0, 255) / 255.0) ** g * p["gain"].unsqueeze(-1) + p["offset"].unsqueeze(-1)).clamp(0, 255)


def scene_under_map(sc: torch.Tensor, p: dict) -> torch.Tensor:
    """[s, c] (N, 56) after the radiometric map: quantiles and the dark-NIR median through f, mean through f, std
    as the half difference of f(mean ± std); fractions, gsd and c unchanged; NaN stays NaN."""
    out = sc.clone()
    blk = sc[:, :4 * N_STATS].reshape(-1, 4, N_STATS)
    mean, std = blk[:, :, 0:1], blk[:, :, 1:2]
    new = _band_map(blk, p)
    new[:, :, 1:2] = 0.5 * (_band_map(mean + std, p) - _band_map(mean - std, p))
    out[:, :4 * N_STATS] = torch.where(torch.isfinite(blk), new, blk).reshape(-1, 4 * N_STATS)
    return out


def geometric(k: int, flip: bool, *maps):
    out = []
    for a in maps:
        if a is None:
            out.append(None)
            continue
        a = torch.rot90(a, k, (a.dim() - 2, a.dim() - 1)) if k else a
        out.append(torch.flip(a, (a.dim() - 1,)) if flip else a)
    return out


# ------------------------------------------------------------------------------------------------ loss
def target_terms(mu, logvar, y, cens, lim, *, y_ntru=None, ntru_offset=None, turb_index=None, beta: float = 0.5):
    """Per-target mean loss (T,) and the number of terms per target (T,). All inputs (N, T) in log10 units."""
    var = torch.exp(logvar).clamp_min(1e-4)
    sd = torch.sqrt(var)
    scale = var.detach() ** beta if beta > 0 else torch.ones_like(var)
    unc = torch.isfinite(y)
    nll = 0.5 * (torch.log(var) + (torch.nan_to_num(y) - mu) ** 2 / var) * scale
    left, right = cens == -1, cens == 1
    z = (torch.nan_to_num(lim) - mu) / sd
    tobit = -(torch.special.log_ndtr(z) * left + torch.special.log_ndtr(-z) * right) * scale
    num = (nll * unc).sum(0) + (tobit * (left | right)).sum(0)
    den = (unc.sum(0) + (left | right).sum(0)).float()
    if turb_index is not None and y_ntru is not None:
        m = torch.isfinite(y_ntru)
        j = turb_index
        e = torch.nan_to_num(y_ntru) - (mu[:, j] + ntru_offset)
        nn_ = 0.5 * (torch.log(var[:, j]) + e ** 2 / var[:, j]) * scale[:, j]
        add = torch.zeros_like(num)
        add[j] = (nn_ * m).sum()
        num = num + add
        cnt = torch.zeros_like(den)
        cnt[j] = m.sum()
        den = den + cnt
    return num / den.clamp_min(1.0), den


def weights_vector(targets, device) -> torch.Tensor:
    return torch.tensor([TARGET_WEIGHT.get(t, 1.0) for t in targets], dtype=torch.float32, device=device)


def total_loss(out: dict, b: dict, model, cfg: dict) -> tuple[torch.Tensor, dict]:
    targets = cfg["targets"]
    ti = targets.index("turbidity") if "turbidity" in targets else None
    off = model.res.ntru_offset if hasattr(model, "res") else model.ntru_offset
    per_t, den = target_terms(out["mu"], out["logvar"], b["y"], b["cens"], b["lim"], y_ntru=b["y_ntru"],
                              ntru_offset=off, turb_index=ti, beta=cfg["beta"])
    w = weights_vector(targets, per_t.device)
    loss = (w * per_t * (den > 0)).sum()
    parts = {"label": float(loss.detach())}
    if cfg["delta_reg"] > 0 and "delta" in out:
        reg = (out["delta"] ** 2).mean()
        loss = loss + cfg["delta_reg"] * reg
        parts["delta2"] = float(reg.detach())
    if cfg["kd_weight"] > 0 and "aux" in out and b.get("kd") is not None:
        k = b["kd"]
        lkd = distill_nll(out["aux"].float(), k["cell_map"], k["water_w"], k["t_mu_s"], k["t_var_s"], k["t_w"],
                          k["t_mu_s"].shape[-1], cfg["kd_sigma_scale2"])
        loss = loss + cfg["kd_weight"] * lkd
        parts["kd"] = float(lkd.detach())
    if cfg["kd_weight"] > 0 and "kd" in out and b.get("teacher_s") is not None:
        m = torch.isfinite(b["teacher_s"])
        lkd = (0.5 * (out["kd"] - torch.nan_to_num(b["teacher_s"])) ** 2 * m).sum() / m.sum().clamp_min(1)
        loss = loss + cfg["kd_weight"] * lkd
        parts["kd"] = float(lkd.detach())
    return loss, parts


# ------------------------------------------------------------------------------------------------ ANCHOR-Set pixels
def set_seed(uid: str) -> int:
    return int(hashlib.sha256((SET_SALT + str(uid)).encode()).hexdigest()[:8], 16)


def sample_pixels(water_px: torch.Tensor, rand: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Indices (N, 1024) into the flattened chip and their validity. Water pixels are drawn at random (``rand`` in
    (0, 1), shape (N, H·W)); a chip with fewer than 64 gets the pixels nearest the chip centre up to 64 (§4.4)."""
    n, H, W = water_px.shape
    wat = water_px.reshape(n, -1) > 0
    yy = (torch.arange(H, device=water_px.device, dtype=torch.float32) - (H - 1) / 2.0).view(H, 1)
    xx = (torch.arange(W, device=water_px.device, dtype=torch.float32) - (W - 1) / 2.0).view(1, W)
    r = torch.sqrt(yy ** 2 + xx ** 2).reshape(1, -1)
    score = torch.where(wat, rand.clamp(1e-6, 1.0), -r / (r.max() + 1.0) - 1.0)      # water first, then nearest
    idx = score.topk(NN.SET_MAX, dim=1).indices
    n_keep = wat.sum(1).clamp(NN.SET_MIN, NN.SET_MAX)
    valid = torch.arange(NN.SET_MAX, device=water_px.device).view(1, -1) < n_keep.view(-1, 1)
    return idx, valid


def pixel_features(dn: torch.Tensor, water_px: torch.Tensor, shore_px: torch.Tensor, gsd: torch.Tensor,
                   scene_p50: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """(N, P, 19): log10 DN (4); log ratios B/R, G/R, B/G, NIR/R (4); log(DN / scene p50) (4); distance to the
    chip centre in m; distance to the shore in m; local 5 × 5 std per band (4); water flag. Bands R, G, B, NIR."""
    n, _, H, W = dn.shape
    x = dn.float()
    lg = torch.log10(x + 1.0)
    r_, g_, b_, nir = lg[:, 0], lg[:, 1], lg[:, 2], lg[:, 3]
    ratios = torch.stack([b_ - r_, g_ - r_, b_ - g_, nir - r_], 1)
    p50 = torch.nan_to_num(scene_p50.float(), nan=127.0).view(n, 4, 1, 1)
    rel = lg - torch.log10(p50 + 1.0)
    m1 = Fn.avg_pool2d(x, 5, stride=1, padding=2, count_include_pad=False)
    m2 = Fn.avg_pool2d(x * x, 5, stride=1, padding=2, count_include_pad=False)
    sd = torch.sqrt((m2 - m1 * m1).clamp_min(0))
    yy = (torch.arange(H, device=dn.device, dtype=torch.float32) - (H - 1) / 2.0).view(1, H, 1)
    xx = (torch.arange(W, device=dn.device, dtype=torch.float32) - (W - 1) / 2.0).view(1, 1, W)
    g = gsd.float().view(n, 1, 1)
    rc = (torch.sqrt(yy ** 2 + xx ** 2) * g).unsqueeze(1)
    sh = (shore_px.float() * g).unsqueeze(1)
    maps = torch.cat([lg, ratios, rel, rc, sh, sd, (water_px > 0).float().unsqueeze(1)], 1)     # (N, 19, H, W)
    flat = maps.flatten(2)
    return torch.gather(flat, 2, idx.unsqueeze(1).expand(-1, flat.shape[1], -1)).permute(0, 2, 1)


def _set_inputs(t: Tensors, idx_cpu: torch.Tensor, device, rand: torch.Tensor | None):
    dn = t.dn[idx_cpu].to(device)
    wp = t.water_px[idx_cpu].to(device)
    if rand is None:                                            # evaluation: fixed by the chip id
        rs = []
        for i in idx_cpu.tolist():
            g = torch.Generator().manual_seed(set_seed(t.uid[i]))
            rs.append(torch.rand(wp.shape[1] * wp.shape[2], generator=g))
        rand = torch.stack(rs).to(device)
    pix, valid = sample_pixels(wp, rand)
    p50 = t.sc[idx_cpu][:, [6 + N_STATS * k for k in range(4)]].to(device)          # scene p50 of R, G, B, NIR
    px = pixel_features(dn, wp, t.shore_px[idx_cpu].to(device), t.gsd[idx_cpu].to(device), p50, pix)
    return px, valid


# ------------------------------------------------------------------------------------------------ models
def build_model(cfg: dict, *, densenet_weights: dict | None = None):
    if cfg["net"] == "anchor":
        return NN.AnchorLite(targets=cfg["targets"], use_anchor=cfg["use_anchor"], use_context=cfg["use_context"],
                             attention=cfg["attention"], pretrained=False)
    if cfg["net"] == "set":
        return NN.WaterSet(targets=cfg["targets"])
    if cfg["net"] == "densenet":
        return NN.DenseNetRegressor(targets=cfg["targets"], parity=cfg["parity"], weights=densenet_weights)
    raise KeyError(cfg["net"])


def _nan_stats(y: torch.Tensor, centre: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-column median of the finite values and variance around ``centre`` (around the median without one)."""
    med, var = [], []
    for j in range(y.shape[1]):
        v = y[:, j]
        ok = torch.isfinite(v)
        if centre is not None:
            ok = ok & torch.isfinite(centre[:, j])
        if not ok.any():
            med.append(0.0)
            var.append(1.0)
            continue
        m = float(v[ok].median())
        d = v[ok] - (centre[ok, j] if centre is not None else m)
        med.append(m)
        var.append(max(float((d ** 2).mean()), 1e-4))
    return torch.tensor(med), torch.tensor(var)


def init_model(model, cfg: dict, tr: Tensors, *, leader_state: dict | None = None) -> dict:
    """Train-row statistics and initial values (§4.1, §4.3). Returns what was set, for the run record."""
    info = {}
    if cfg["net"] == "anchor":
        if leader_state is not None:
            model.load_leader(leader_state)
            info["leader_init"] = True
        if cfg["use_anchor"]:
            _, var = _nan_stats(tr.y, tr.anchor)
        else:
            med, var = _nan_stats(tr.y)
            model.res.base.copy_(med)
        model.res.set_logvar_bias(var.tolist())
        model.ctx.fit(tr.sc)
        info["logvar_bias_variance"] = [float(v) for v in var]
    elif cfg["net"] == "set":
        _, var = _nan_stats(tr.y, tr.anchor)
        model.res.set_logvar_bias(var.tolist())
        model.ctx.fit(tr.sc)
        sel = torch.arange(0, tr.n, max(1, tr.n // 512))
        px, valid = _set_inputs(tr, sel, "cpu", torch.rand(len(sel), tr.dn.shape[-1] * tr.dn.shape[-2],
                                                           generator=torch.Generator().manual_seed(0)))
        model.fit_pixels(px, valid)
        info["logvar_bias_variance"] = [float(v) for v in var]
    else:
        med, var = _nan_stats(tr.y)
        model.set_output_bias(med, var)
        x01 = tr.dn[:: max(1, tr.n // 512)].float() / 255.0
        model.x_mean.copy_(x01.mean((0, 2, 3)))
        model.x_std.copy_(x01.std((0, 2, 3)).clamp_min(1e-3))
        if cfg["parity"]:
            model.ctx.fit(tr.sc)
        info["train_median"] = [float(v) for v in med]
    return info


def _param_groups(model, cfg: dict) -> list:
    if cfg["net"] == "anchor":
        return [{"params": model.leader_parameters(), "lr": cfg["lr_backbone"]},
                {"params": model.new_parameters(), "lr": cfg["lr"]}]
    if cfg["net"] == "densenet":
        return [{"params": model.backbone_parameters(), "lr": cfg["lr_backbone"]},
                {"params": model.new_parameters(), "lr": cfg["lr"]}]
    return [{"params": list(model.parameters()), "lr": cfg["lr"]}]


def _lr_lambda(cfg: dict, steps_per_epoch: int):
    total, warm = cfg["epochs"] * steps_per_epoch, cfg["warmup_epochs"] * steps_per_epoch

    def f(step):
        if step < warm:
            return (step + 1) / max(warm, 1)
        return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(total - warm, 1)))
    return f


def _x_norm(model, cfg: dict, dn: torch.Tensor) -> torch.Tensor:
    src = model.student if cfg["net"] == "anchor" else model
    return (dn / 255.0 - src.x_mean.view(1, -1, 1, 1)) / src.x_std.view(1, -1, 1, 1)


def _forward(model, cfg: dict, t: Tensors, idx: torch.Tensor, device, *, train: bool, gen=None, use_amp=False,
             kd_stats=None) -> tuple[dict, dict]:
    """One batch: inputs to the device, augmentation (training), forward pass. Returns (outputs, batch tensors)."""
    b = {"y": t.y[idx].to(device), "cens": t.cens[idx].to(device), "lim": t.lim[idx].to(device),
         "y_ntru": t.y_ntru[idx].to(device)}
    sc = t.sc[idx].to(device)
    anchor = None if t.anchor is None else t.anchor[idx].to(device)
    if cfg["net"] == "set":
        rand = torch.rand(len(idx), t.dn.shape[-1] * t.dn.shape[-2], device=device) if train else None
        px, valid = _set_inputs(t, idx, device, rand)
        with torch.autocast("cuda", enabled=use_amp):
            out = model(px, valid, sc, anchor)
        return out, b
    dn = t.dn[idx].to(device).float()
    pool_w, water = t.pool_w[idx].to(device), t.water[idx].to(device)
    cm = ww = None
    with_kd = train and t.kd is not None and cfg["kd_weight"] > 0 and cfg["net"] == "anchor"
    if with_kd:
        cm, ww = t.kd["cell_map"][idx].to(device).long(), t.kd["water_w"][idx].to(device)
    if train and cfg["aug"]:
        k = int(torch.randint(0, 4, (1,), generator=gen))
        fl = bool(torch.randint(0, 2, (1,), generator=gen))
        dn, pool_w, water, cm, ww = geometric(k, fl, dn, pool_w, water, cm, ww)
        dn, p = radiometric(dn, cfg)
        sc = scene_under_map(sc, p)
    if with_kd:
        mean, std = kd_stats
        b["kd"] = {"cell_map": cm, "water_w": ww, "t_w": t.kd["t_w"][idx].to(device),
                   "t_mu_s": (t.kd["t_mu"][idx].to(device) - mean) / std,
                   "t_var_s": t.kd["t_var"][idx].to(device) / std ** 2}
    if train and t.teacher is not None and cfg["kd_weight"] > 0 and cfg["net"] == "densenet" and cfg["parity"]:
        mean, std = kd_stats
        b["teacher_s"] = (t.teacher[idx].to(device) - mean) / std
    x = _x_norm(model, cfg, dn)
    with torch.autocast("cuda", enabled=use_amp):
        if cfg["net"] == "anchor":
            out = model(x, sc, anchor, pool_w, water, t.gsd[idx].to(device))
        else:
            out = model(x, sc if cfg["parity"] else None)
    return out, b


def train(cfg: dict, tr: Tensors, *, seed: int = 0, device: str = "cuda", leader_state: dict | None = None,
          densenet_weights: dict | None = None, log=print):
    """Train one model for the fixed number of epochs. Returns (model, history, initialisation record)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if cfg["use_anchor"] and cfg["net"] != "densenet" and tr.anchor is None:
        raise ValueError("this arm needs anchors")
    model = build_model(cfg, densenet_weights=densenet_weights)
    info = init_model(model, cfg, tr, leader_state=leader_state)
    model = model.to(device)
    kd_stats = None
    if cfg["net"] == "anchor":
        kd_stats = (model.student.y_mean.view(1, 1, -1), model.student.y_std.view(1, 1, -1))
    elif cfg["net"] == "densenet" and cfg["parity"] and tr.teacher is not None:
        tm = tr.teacher
        ok = torch.isfinite(tm)
        n = ok.sum(0).clamp_min(1)
        mean = torch.where(ok, tm, torch.zeros_like(tm)).sum(0) / n
        std = (torch.where(ok, tm - mean, torch.zeros_like(tm)) ** 2).sum(0).div(n).sqrt().clamp_min(1e-3)
        kd_stats = (mean.to(device), std.to(device))
    opt = torch.optim.AdamW(_param_groups(model, cfg), weight_decay=cfg["wd"])
    spe = max(1, tr.n // cfg["batch"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda(cfg, spe))
    use_amp = bool(cfg["amp"]) and str(device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    gen = torch.Generator().manual_seed(seed)
    hist = []
    for ep in range(cfg["epochs"]):
        model.train()
        t0 = time.time()
        perm = torch.randperm(tr.n, generator=gen)
        tot: dict = {}
        for it in range(spe):
            idx = perm[it * cfg["batch"]:(it + 1) * cfg["batch"]]
            out, b = _forward(model, cfg, tr, idx, device, train=True, gen=gen, use_amp=use_amp, kd_stats=kd_stats)
            loss, parts = total_loss(out, b, model, cfg)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {ep + 1}, step {it}")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(opt)
            scaler.update()
            sched.step()
            parts["loss"] = float(loss.detach())
            for k_, v in parts.items():
                tot[k_] = tot.get(k_, 0.0) + v
        rec = {"epoch": ep + 1, **{k_: v / spe for k_, v in tot.items()}, "sec": time.time() - t0}
        hist.append(rec)
        log(" ".join(f"{k_}={v:.4g}" if isinstance(v, float) else f"{k_}={v}" for k_, v in rec.items()))
    return model, hist, info


@torch.no_grad()
def predict(model, cfg: dict, te: Tensors, *, device: str = "cuda", batch: int = 64) -> dict:
    """``{'mu': (N, T), 'sd': (N, T), 'pooled': (N, D)}`` in log10 units, rows in ``te`` order."""
    model.eval()
    use_amp = bool(cfg["amp"]) and str(device).startswith("cuda")
    mus, sds, pooled = [], [], []
    for s in range(0, te.n, batch):
        idx = torch.arange(s, min(s + batch, te.n))
        out, _ = _forward(model, cfg, te, idx, device, train=False, use_amp=use_amp)
        mus.append(out["mu"].float().cpu().numpy())
        sds.append(torch.exp(0.5 * out["logvar"].float()).cpu().numpy())
        pooled.append(out["pooled"].float().cpu().numpy())
    return {"mu": np.concatenate(mus), "sd": np.concatenate(sds), "pooled": np.concatenate(pooled)}


def prediction_dict(cfg: dict, mu: np.ndarray) -> dict:
    """{target: array}; an A4 model scores FNU through its NTU head (§4.5)."""
    out = {t: mu[:, j] for j, t in enumerate(cfg["targets"])}
    if cfg.get("reduced") and "turb_fnu" not in out and "turbidity" in out:
        out["turb_fnu"] = out["turbidity"]
    return out


def ensemble(mus: list, sds: list) -> tuple[np.ndarray, np.ndarray]:
    """Mean μ over seeds; σ² = mean σ² + variance of μ over seeds (as the frozen training script)."""
    return np.mean(mus, 0), np.sqrt(np.mean(np.square(sds), 0) + np.var(mus, 0))


__all__ = ["ARMS", "BASE_CFG", "TARGET_WEIGHT", "Tensors", "arm_config", "build_tensors", "attach_kd", "radiometric",
           "scene_under_map", "geometric", "target_terms", "total_loss", "set_seed", "sample_pixels",
           "pixel_features", "build_model", "init_model", "train", "predict", "prediction_dict", "ensemble"]
