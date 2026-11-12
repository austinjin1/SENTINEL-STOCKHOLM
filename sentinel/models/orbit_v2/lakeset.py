# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""SatWQ-LakeSet-MT (arm A1; orbit_arch_v2_amendment.md §3.4).

1. Pixel encoder: v1 (``SatWQNet.encode_pixels`` is imported and run on this module's identically named layers).
   The pixel-offset MLP gains one input, log1p(d_p / 10 m).
2. Date embedding: v1 attention pooling + mean pooling over a date's water pixels (<= 64 for the target date, 8 for
   a context date); one set of weights for target and context dates.
3. Lake-context set: token = e_d + MLP(q_d) + MLP(dt_d); 2 SAB blocks (4 heads); PMA with one seed -> lake
   embedding l. An empty set uses a learned embedding. No count feature.
4. Land ring MLP; 5. geometry MLP (v1 geometry + glint features); 6. context MLP (DOY; lat/lon only in c1).
7. Fusion: LayerNorm -> MLP([chip_t, l, chip_t - W l, geometry, ring, context]) -> h (d = 64).
8. One K = 5 Gaussian mixture per target (standardised log10).

``use_set=False`` is configuration c3 (no context set).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..orbit.satwq import Batchable, SatWQNet
from .chipfeat import PIXEL_M, RING_DIM

N_Q = 8                 # chipfeat.date_quality
N_DT = 3                # dt / 183, |dt| / 183, log1p(|dt|) / log1p(183)
DT_MAX = 183.0
K_SET = 6
P_CONTEXT = 8
# layers whose weights are transferred from a GLORIA-pretrained v1 network (pixel encoder + date pooling)
TRANSFER = ("val_embed", "wl_embed", "cls", "spec", "pix_pos", "pool_q", "pool", "pool_proj")


def _mlp(n_in: int, d: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(n_in, d), nn.GELU(), nn.Linear(d, d))


def dt_features(dt_days) -> np.ndarray:
    dt = np.asarray(dt_days, np.float32)
    return np.stack([dt / DT_MAX, np.abs(dt) / DT_MAX, np.log1p(np.abs(dt)) / np.log1p(DT_MAX)], -1).astype(np.float32)


class SatWQLakeSetMT(nn.Module):
    def __init__(self, n_geo: int, n_ctx: int, n_targets: int, *, n_ring: int = RING_DIM, n_q: int = N_Q,
                 d: int = 64, n_layers: int = 3, n_heads: int = 4, k: int = 5, dropout: float = 0.1,
                 use_set: bool = True, n_sab: int = 2):
        super().__init__()
        self.d, self.k, self.nt, self.use_set = d, k, n_targets, use_set
        # --- v1 pixel encoder and date pooling (same names as SatWQNet so that weights transfer by name)
        self.val_embed = nn.Sequential(nn.Linear(1, d), nn.GELU(), nn.Linear(d, d))
        self.wl_embed = nn.Linear(32, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        enc = nn.TransformerEncoderLayer(d, n_heads, 2 * d, dropout, batch_first=True, norm_first=True,
                                         activation="gelu")
        self.spec = nn.TransformerEncoder(enc, n_layers, enable_nested_tensor=False)
        self.pix_pos = nn.Sequential(nn.Linear(4, d), nn.GELU(), nn.Linear(d, d))      # (dy, dx, r) / 7, log1p(d_p / 10 m)
        self.pool_q = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.pool = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.pool_proj = nn.Linear(2 * d, d)
        # --- side inputs
        self.geo = _mlp(n_geo, d)
        self.ctx = _mlp(n_ctx, d)
        self.ctx_missing = nn.Parameter(torch.zeros(d))
        self.ring = _mlp(n_ring, d)
        # --- lake-context set
        if use_set:
            self.q_mlp = _mlp(n_q, d)
            self.dt_mlp = _mlp(N_DT, d)
            self.sab = nn.ModuleList([nn.TransformerEncoderLayer(d, n_heads, 2 * d, dropout, batch_first=True,
                                                                 norm_first=True, activation="gelu")
                                      for _ in range(n_sab)])
            self.pma_seed = nn.Parameter(torch.randn(1, 1, d) * 0.02)
            self.pma = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
            self.pma_norm = nn.LayerNorm(d)
            self.pma_ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
            self.set_empty = nn.Parameter(torch.zeros(d))
            self.lake_proj = nn.Linear(d, d, bias=False)
        n_in = (6 if use_set else 4) * d
        self.fuse = nn.Sequential(nn.LayerNorm(n_in), nn.Linear(n_in, 2 * d), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(2 * d, d), nn.GELU())
        self.head = nn.Linear(d, n_targets * 3 * k)

    encode_pixels = SatWQNet.encode_pixels          # v1 code on this module's layers (imported, not copied)

    def embed_chips(self, refl, lam, band_mask, pix_mask, offsets, dp):
        """refl [N, P, B]; lam [N, B]; band_mask [N, P, B]; pix_mask [N, P]; offsets [N, P, 2]; dp [N, P]
        (already log1p(d_p / 10 m)) -> date embeddings [N, d]. Mirrors the first half of SatWQNet.forward."""
        N, P, B = refl.shape
        flat = pix_mask.reshape(-1)
        lam_p = lam[:, None, :].expand(N, P, B).reshape(-1, B)
        emb = torch.zeros(N * P, self.d, device=refl.device, dtype=refl.dtype)
        if bool(flat.any()):
            emb[flat] = self.encode_pixels(refl.reshape(-1, B)[flat], lam_p[flat],
                                           band_mask.reshape(-1, B)[flat]).to(emb.dtype)
        r = torch.sqrt((offsets ** 2).sum(-1, keepdim=True))
        pos = torch.cat([torch.cat([offsets, r], -1) / 7.0, dp[..., None]], -1)
        emb = emb.reshape(N, P, self.d) + self.pix_pos(pos)
        q = self.pool_q.expand(N, -1, -1)
        att, _ = self.pool(q, emb, emb, key_padding_mask=~pix_mask)
        m = pix_mask.to(emb.dtype)[..., None]
        mean = (emb * m).sum(1) / m.sum(1).clamp(min=1)
        return self.pool_proj(torch.cat([att[:, 0], mean], -1))

    def lake_embedding(self, C: int, c_refl, c_lam, c_band_mask, c_pix_mask, c_offsets, c_dp, c_q, c_dt, c_owner,
                       c_slot, k_set: int = K_SET):
        """Flattened context members (M of them; ``c_owner`` = batch row, ``c_slot`` = position 0..K-1) -> [C, d]."""
        dev = c_owner.device
        if c_owner.numel() == 0:
            return self.set_empty.expand(C, -1)
        tok = self.embed_chips(c_refl, c_lam, c_band_mask, c_pix_mask, c_offsets, c_dp) + self.q_mlp(c_q) \
            + self.dt_mlp(c_dt)
        x = torch.zeros(C, k_set, self.d, device=dev, dtype=tok.dtype)
        mask = torch.zeros(C, k_set, dtype=torch.bool, device=dev)
        x[c_owner, c_slot] = tok
        mask[c_owner, c_slot] = True
        empty = ~mask.any(1)
        pad = ~mask
        pad[empty, 0] = False                         # never an all-padded row (NaN attention); replaced below
        for blk in self.sab:
            x = blk(x, src_key_padding_mask=pad)
        s = self.pma_seed.expand(C, -1, -1)
        att, _ = self.pma(s, x, x, key_padding_mask=pad)
        z = self.pma_norm(att[:, 0] + s[:, 0])
        z = z + self.pma_ff(z)
        return torch.where(empty[:, None], self.set_empty.expand(C, -1).to(z.dtype), z)

    def forward(self, refl, lam, band_mask, pix_mask, offsets, geo, ctx, ctx_on, dp, ring, c_refl=None, c_lam=None,
                c_band_mask=None, c_pix_mask=None, c_offsets=None, c_dp=None, c_q=None, c_dt=None, c_owner=None,
                c_slot=None):
        """Target-date tensors as SatWQNet.forward plus dp [C, P] and ring [C, R]; context members flattened.
        Returns (logw, mu, sigma) each [C, T, K] (standardised)."""
        C = refl.shape[0]
        chip = self.embed_chips(refl, lam, band_mask, pix_mask, offsets, dp)
        parts = [chip]
        if self.use_set:
            if c_owner is None:
                lake = self.set_empty.expand(C, -1)
            else:
                lake = self.lake_embedding(C, c_refl, c_lam, c_band_mask, c_pix_mask, c_offsets, c_dp, c_q, c_dt,
                                           c_owner, c_slot)
            parts += [lake, chip - self.lake_proj(lake)]
        c = torch.where(ctx_on[:, None], self.ctx(ctx), self.ctx_missing.expand(C, -1))
        h = self.fuse(torch.cat(parts + [self.geo(geo), self.ring(ring), c], -1))
        o = self.head(h).reshape(C, self.nt, 3, self.k)
        return torch.log_softmax(o[:, :, 0], -1), o[:, :, 1], F.softplus(o[:, :, 2]) + 1e-3

    @torch.no_grad()
    def load_v1_encoder(self, v1_state: dict) -> list[str]:
        """Copy the pixel encoder and date pooling of a (GLORIA-pretrained) v1 network. ``pix_pos.0.weight`` gains a
        zero column for the d_p input, so the transferred network ignores d_p at initialisation."""
        own = self.state_dict()
        done = []
        for name, v in v1_state.items():
            if name.split(".")[0] not in TRANSFER or name not in own:
                continue
            if name == "pix_pos.0.weight":
                w = torch.zeros_like(own[name])
                w[:, :v.shape[1]] = v
                own[name] = w
            else:
                if own[name].shape != v.shape:
                    raise ValueError(f"shape mismatch for {name}: {tuple(own[name].shape)} vs {tuple(v.shape)}")
                own[name] = v.clone()
            done.append(name)
        self.load_state_dict(own)
        return done


class SetBatchable:
    """Batches for SatWQ-LakeSet-MT: a v1 :class:`Batchable` of the target chips plus d_p, the ring vector and the
    lake-context set. Context chips are held once (``cstore``, a v1 Batchable with 8 pixels per chip) and indexed
    per row by ``cidx`` [N, K] (-1 = none). v1's augmentation code runs unchanged on both.

    Training (§3.4 item 3): the whole set is dropped with ``p_set_drop`` (0.3); otherwise a uniformly drawn number
    1..n of the row's n members is kept (random subset).
    """

    def __init__(self, base: Batchable, dp_m: np.ndarray, ring: np.ndarray, *, cstore: Batchable | None = None,
                 c_dp_m: np.ndarray | None = None, c_q: np.ndarray | None = None, cidx: np.ndarray | None = None,
                 cdt: np.ndarray | None = None, cens: np.ndarray | None = None):
        self.base, self.ring = base, np.asarray(ring, np.float32)
        self.dp = np.log1p(np.nan_to_num(np.asarray(dp_m, np.float32)) / PIXEL_M)
        self.y = base.y
        self.cens = cens
        self.cstore = cstore
        if cstore is not None:
            self.c_dp = np.log1p(np.nan_to_num(np.asarray(c_dp_m, np.float32)) / PIXEL_M)
            self.c_q = np.asarray(c_q, np.float32)
            self.cidx = np.asarray(cidx, np.int64)
            self.cdt = np.asarray(cdt, np.float32)

    def __len__(self):
        return len(self.base)

    def batch(self, idx, device, *, train=False, rng=None, aug=None):
        aug = aug or {}
        idx = np.asarray(idx)
        xb, y, ym = self.base.batch(idx, device, train=train, rng=rng, aug=aug)
        t = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=device)  # noqa: E731
        xb["dp"] = t(self.dp[idx])
        xb["ring"] = t(self.ring[idx])
        if self.cstore is None:
            return xb, y, ym
        ci = self.cidx[idx].copy()
        if aug.get("set_off", False):
            ci[:] = -1
        elif train:
            ci[rng.random(len(idx)) < aug.get("p_set_drop", 0.3)] = -1
            for i in np.flatnonzero((ci >= 0).sum(1) > 1):
                have = np.flatnonzero(ci[i] >= 0)
                keep = rng.permutation(have)[: rng.integers(1, have.size + 1)]
                drop = np.setdiff1d(have, keep)
                ci[i, drop] = -1
        owner, slot = np.nonzero(ci >= 0)
        if owner.size:
            mem = ci[owner, slot]
            c_aug = {k: v for k, v in aug.items() if k not in ("p_ctx_drop", "ctx_off")}
            cb, _, _ = self.cstore.batch(mem, device, train=train, rng=rng, aug=c_aug)
            xb.update(c_refl=cb["refl"], c_lam=cb["lam"], c_band_mask=cb["band_mask"], c_pix_mask=cb["pix_mask"],
                      c_offsets=cb["offsets"], c_dp=t(self.c_dp[mem]), c_q=t(self.c_q[mem]),
                      c_dt=t(dt_features(self.cdt[idx][owner, slot])), c_owner=t(owner, torch.long),
                      c_slot=t(slot, torch.long))
        return xb, y, ym
