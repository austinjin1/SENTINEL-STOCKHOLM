# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""SatWQ-Net (orbit_protocol.md §4.1): spectral-token encoder with wavelength tags -> attention pooling over
valid water pixels -> context fusion with whole-vector dropout -> per-target Gaussian-mixture head (log10).

* A pixel is a set of band tokens: token = MLP(asinh(rho / 0.01)) + Linear(sinusoidal(lambda_nm)). Missing
  bands are masked, so any band subset (e.g. the drone's B02/B03/B04/B08) runs through the same weights.
* A chip is a set of up to 64 water pixels with offset embeddings; masked attention pooling (+ mean pool).
* Geometry (sun/view angles, AOT, shore distance, water-pixel count) is always available (image metadata);
  context (DOY harmonics + location unit vector) is dropped as a whole with probability ``p_ctx_drop``.
* Loss: masked mixture NLL over the targets present in each row (standardised log10 per target).
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import BANDS, RGBN, TARGETS

REFL_SCALE = 0.01


def sinusoidal(lam_nm: torch.Tensor, n_freq: int = 16) -> torch.Tensor:
    x = (lam_nm - 350.0) / (2300.0 - 350.0)
    f = torch.arange(n_freq, device=lam_nm.device, dtype=torch.float32)
    ang = x[..., None] * math.pi * (2.0 ** (f / 2.0))
    return torch.cat([torch.sin(ang), torch.cos(ang)], -1)


class SatWQNet(nn.Module):
    def __init__(self, n_geo: int, n_ctx: int, d: int = 64, n_layers: int = 3, n_heads: int = 4, k: int = 5,
                 n_targets: int = len(TARGETS), dropout: float = 0.1):
        super().__init__()
        self.d, self.k, self.nt = d, k, n_targets
        self.val_embed = nn.Sequential(nn.Linear(1, d), nn.GELU(), nn.Linear(d, d))
        self.wl_embed = nn.Linear(32, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        enc = nn.TransformerEncoderLayer(d, n_heads, 2 * d, dropout, batch_first=True, norm_first=True,
                                         activation="gelu")
        self.spec = nn.TransformerEncoder(enc, n_layers, enable_nested_tensor=False)
        self.pix_pos = nn.Sequential(nn.Linear(3, d), nn.GELU(), nn.Linear(d, d))
        self.pool_q = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.pool = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.pool_proj = nn.Linear(2 * d, d)
        self.geo = nn.Sequential(nn.Linear(n_geo, d), nn.GELU(), nn.Linear(d, d))
        self.ctx = nn.Sequential(nn.Linear(n_ctx, d), nn.GELU(), nn.Linear(d, d))
        self.ctx_missing = nn.Parameter(torch.zeros(d))
        self.fuse = nn.Sequential(nn.LayerNorm(3 * d), nn.Linear(3 * d, 2 * d), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(2 * d, d), nn.GELU())
        self.head = nn.Linear(d, n_targets * 3 * k)

    def encode_pixels(self, refl, lam, band_mask):
        """refl/lam/band_mask: [N, B] -> pixel embeddings [N, d]."""
        v = torch.asinh(torch.nan_to_num(refl, 0.0) / REFL_SCALE)[..., None]
        tok = self.val_embed(v) + self.wl_embed(sinusoidal(lam))
        tok = torch.cat([self.cls.expand(tok.shape[0], -1, -1), tok], 1)
        pad = torch.cat([torch.zeros_like(band_mask[:, :1]), ~band_mask], 1)
        h = self.spec(tok, src_key_padding_mask=pad)
        return h[:, 0]

    def forward(self, refl, lam, band_mask, pix_mask, offsets, geo, ctx, ctx_on):
        """refl [C, P, B]; lam [C, B]; band_mask [C, P, B] bool; pix_mask [C, P] bool; offsets [C, P, 2];
        geo [C, G]; ctx [C, X]; ctx_on [C] bool. Returns (logw, mu, sigma) each [C, T, K] (standardised)."""
        C, P, B = refl.shape
        flat = pix_mask.reshape(-1)
        lam_p = lam[:, None, :].expand(C, P, B).reshape(-1, B)
        emb = torch.zeros(C * P, self.d, device=refl.device, dtype=refl.dtype)
        emb[flat] = self.encode_pixels(refl.reshape(-1, B)[flat], lam_p[flat], band_mask.reshape(-1, B)[flat])
        r = torch.sqrt((offsets ** 2).sum(-1, keepdim=True))
        emb = emb.reshape(C, P, self.d) + self.pix_pos(torch.cat([offsets, r], -1) / 7.0)
        q = self.pool_q.expand(C, -1, -1)
        att, _ = self.pool(q, emb, emb, key_padding_mask=~pix_mask)
        m = pix_mask.float()[..., None]
        mean = (emb * m).sum(1) / m.sum(1).clamp(min=1)
        chip = self.pool_proj(torch.cat([att[:, 0], mean], -1))
        g = self.geo(geo)
        c = torch.where(ctx_on[:, None], self.ctx(ctx), self.ctx_missing.expand(C, -1))
        h = self.fuse(torch.cat([chip, g, c], -1))
        o = self.head(h).reshape(C, self.nt, 3, self.k)
        return torch.log_softmax(o[:, :, 0], -1), o[:, :, 1], F.softplus(o[:, :, 2]) + 1e-3


def masked_nll(logw, mu, sigma, y, ymask):
    """y, ymask [C, T]; returns mean NLL over present (row, target) entries."""
    yy = torch.nan_to_num(y, 0.0)[..., None]
    lp = logw - 0.5 * ((yy - mu) / sigma) ** 2 - torch.log(sigma) - 0.5 * math.log(2 * math.pi)
    nll = -torch.logsumexp(lp, -1)
    return (nll * ymask).sum() / ymask.sum().clamp(min=1)


# ------------------------------------------------------------------------------------- mixtures

def mixture_quantiles(w, mu, sg, qs=(0.5,), n_iter: int = 60) -> np.ndarray:
    """Quantiles of 1-D Gaussian mixtures by bisection on the exact CDF (review W7: a fixed grid erred by
    up to 0.07 log10 with wide low-weight components). w/mu/sg [N, K] -> [N, len(qs)] (NaN where mu NaN).
    Bracket: [min(mu - 10 sigma), max(mu + 10 sigma)] over components with weight > 1e-9."""
    from scipy.special import ndtr

    w, mu, sg = (np.asarray(a, float) for a in (w, mu, sg))
    out = np.full((w.shape[0], len(qs)), np.nan)
    ok = np.all(np.isfinite(mu), 1) & np.all(np.isfinite(sg), 1)
    if not ok.any():
        return out
    w, mu, sg = w[ok], mu[ok], sg[ok]
    w = w / w.sum(1, keepdims=True)
    live = w > 1e-9
    lo0 = np.where(live, mu - 10 * sg, np.inf).min(1)
    hi0 = np.where(live, mu + 10 * sg, -np.inf).max(1)
    res = np.empty((w.shape[0], len(qs)))
    for j, q in enumerate(qs):
        lo, hi = lo0.copy(), hi0.copy()
        for _ in range(n_iter):
            mid = 0.5 * (lo + hi)
            c = (w * ndtr((mid[:, None] - mu) / sg)).sum(1)
            below = c < q
            lo = np.where(below, mid, lo)
            hi = np.where(below, hi, mid)
        res[:, j] = 0.5 * (lo + hi)
    out[ok] = res
    return out


def mixture_exceedance(w, mu, sg, thr_log10: float) -> np.ndarray:
    from scipy.special import ndtr

    w = np.asarray(w) / np.asarray(w).sum(1, keepdims=True)
    return (w * (1 - ndtr((thr_log10 - mu) / sg))).sum(1)


# ------------------------------------------------------------------------------------- tensors

class Batchable:
    """Holds numpy arrays for chips (pixels, geometry, context, targets) and builds torch batches."""

    def __init__(self, refl, n_pix, offsets, lam, geo, ctx, y):
        self.refl, self.n_pix, self.offsets, self.lam = refl, n_pix, offsets, lam
        self.geo, self.ctx, self.y = geo, ctx, y

    def __len__(self):
        return len(self.n_pix)

    def batch(self, idx, device, *, train=False, rng=None, aug=None):
        aug = aug or {}
        refl = self.refl[idx].astype(np.float32)          # [C, P, B]
        off = self.offsets[idx].astype(np.float32)
        n = self.n_pix[idx]
        C, P, B = refl.shape
        pm = np.arange(P)[None] < n[:, None]
        if train:
            # random pixel subsets (keeps per-pixel inference in distribution)
            sub = rng.random(C) < aug.get("p_pixel_subset", 0.3)
            for i in np.flatnonzero(sub & (n > 1)):
                k = rng.integers(1, n[i] + 1)
                keep = rng.permutation(n[i])[:k]
                m = np.zeros(P, bool)
                m[keep] = True
                pm[i] = m
            mult = 1 + aug.get("mult_noise", 0.0) * rng.standard_normal((C, 1, B))
            add = aug.get("add_noise", 0.0) * rng.standard_normal((C, 1, 1)) + \
                aug.get("band_add_noise", 0.0) * rng.standard_normal((C, 1, B))
            refl = refl * mult + add
        bm = np.isfinite(refl) & pm[..., None]
        if train:
            mode = rng.random(C)
            rgbn = np.array([b in RGBN for b in BANDS])
            r_m = mode < aug.get("p_rgbn", 0.0)
            bm[r_m] &= rgbn[None, None, :]
            d_m = (mode >= aug.get("p_rgbn", 0.0)) & (mode < aug.get("p_rgbn", 0.0) + aug.get("p_band_drop", 0.0))
            for i in np.flatnonzero(d_m):
                keep = rng.random(B) > 0.25
                keep[rng.integers(B)] = True
                bm[i] &= keep[None, :]
        pm &= bm.any(-1)
        none = ~pm.any(1)
        if none.any():          # never an empty chip: keep the first pixel (all its present bands)
            pm[none, 0] = True
            bm[none, 0] = np.isfinite(refl[none, 0])
        ctx_on = np.ones(C, bool)
        if train:
            ctx_on = rng.random(C) >= aug.get("p_ctx_drop", 0.5)
        if aug.get("ctx_off", False):
            ctx_on[:] = False
        t = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=device)  # noqa: E731
        y = self.y[idx]
        return dict(refl=t(np.nan_to_num(refl)), lam=t(self.lam[idx]), band_mask=t(bm, torch.bool),
                    pix_mask=t(pm, torch.bool), offsets=t(off), geo=t(self.geo[idx]), ctx=t(self.ctx[idx]),
                    ctx_on=t(ctx_on, torch.bool)), t(np.nan_to_num(y)), t(np.isfinite(y))


# ------------------------------------------------------------------------------------- training

def run_epochs(model, train_set: Batchable, *, inner_set: Batchable | None, epochs: int, lr: float, batch: int,
               device, seed: int, aug: dict, y_mean, y_std, weight_decay: float = 1e-4, eval_every: int = 1,
               patience: int = 15, log=print, select_targets=("chla", "secchi")):
    """AdamW + cosine schedule; early stopping on mean inner MdSA over ``select_targets`` (rows present)."""
    from .evaluate import mdsa_log10

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps = epochs * max(1, len(train_set) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(steps, 4),
                                                pct_start=min(0.3, max(0.05, 3.0 / max(steps, 4))))
    best = (np.inf, None, -1)
    hist = []
    t0 = time.time()
    tidx = [TARGETS.index(t) for t in select_targets]
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(len(train_set))
        tot = 0.0
        nb = 0
        for s in range(0, len(perm) - batch + 1, batch):
            xb, yb, mb = train_set.batch(perm[s:s + batch], device, train=True, rng=rng, aug=aug)
            logw, mu, sg = model(**xb)
            loss = masked_nll(logw, mu, sg, yb, mb.float())
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss)
            nb += 1
        rec = {"epoch": ep, "train_nll": tot / max(nb, 1)}
        if inner_set is not None and (ep % eval_every == 0 or ep == epochs - 1):
            q = predict_quantiles(model, inner_set, device, y_mean, y_std, qs=(0.5,), ctx_off=aug.get("ctx_off", False))
            sc = []
            for j in tidx:
                y = inner_set.y[:, j] * y_std[j] + y_mean[j]
                m = np.isfinite(y)
                if m.sum() >= 30:
                    sc.append(mdsa_log10(q[:, j, 0][m] - y[m]))
            if not sc:
                raise RuntimeError(f"early stopping impossible: no select target in {select_targets} has >= 30 inner rows")
            rec["inner_mdsa"] = float(np.mean(sc))
            rec["inner_targets_used"] = len(sc)      # < len(select_targets) if a target had < 30 inner rows
            if rec["inner_mdsa"] < best[0]:
                best = (rec["inner_mdsa"], {k: v.detach().clone() for k, v in model.state_dict().items()}, ep)
            elif ep - best[2] > patience:
                hist.append(rec)
                log(f"  early stop at epoch {ep} (best {best[2]}: {best[0]:.2f})")
                break
        hist.append(rec)
        if ep % 10 == 0:
            log(f"  ep {ep} nll {rec['train_nll']:.4f} inner {rec.get('inner_mdsa', float('nan')):.2f} "
                f"({time.time() - t0:.0f}s)")
    if best[1] is not None:
        model.load_state_dict(best[1])
    return {"history": hist, "best_epoch": best[2], "best_inner": best[0]}


@torch.no_grad()
def predict_mixture(model, ds: Batchable, device, y_mean, y_std, batch: int = 256, ctx_off: bool = False,
                    band_subset=None):
    """Mixtures in log10 units: (w, mu, sigma) each [N, T, K]."""
    model.eval()
    W, MU, SG = [], [], []
    keep = None
    if band_subset is not None:
        keep = np.array([b in band_subset for b in BANDS])
    for s in range(0, len(ds), batch):
        idx = np.arange(s, min(s + batch, len(ds)))
        xb, _, _ = ds.batch(idx, device, train=False, aug={"ctx_off": ctx_off})
        if keep is not None:
            xb["band_mask"] &= torch.as_tensor(keep, device=device)[None, None, :]
            xb["pix_mask"] &= xb["band_mask"].any(-1)
        logw, mu, sg = model(**xb)
        W.append(torch.exp(logw).cpu().numpy())
        MU.append(mu.cpu().numpy())
        SG.append(sg.cpu().numpy())
    w, mu, sg = np.concatenate(W), np.concatenate(MU), np.concatenate(SG)
    ym, ys = np.asarray(y_mean)[None, :, None], np.asarray(y_std)[None, :, None]
    return w, mu * ys + ym, sg * ys


def predict_quantiles(model, ds, device, y_mean, y_std, qs=(0.5,), **kw):
    w, mu, sg = predict_mixture(model, ds, device, y_mean, y_std, **kw)
    out = np.stack([mixture_quantiles(w[:, j], mu[:, j], sg[:, j], qs) for j in range(w.shape[1])], 1)
    return out   # [N, T, Q]


def n_params(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))
