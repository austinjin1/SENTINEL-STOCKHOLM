# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""SENTINEL-Fuse deep model (RESEARCH_PLAN.md §6; docs/research/05 §e.3).

* **Tokenizers** — one :class:`GroupTokenizer` per modality turns engineered feature groups into
  tokens (per-group MLP on [values, observed-mask] + group embedding + modality embedding +
  log-scaled sinusoidal age embedding). A group with no observed feature is an absent token
  (key-padding mask); an absent modality produces no update at all.
* **Core** — HEALNet/Perceiver hybrid: a learned latent array is updated, layer by layer, by one
  cross-attention per modality with *modality-specific* Q/K/V weights; the update of a modality
  that is absent for a row is skipped with ``torch.where`` (bit-identical, not multiplied by 0),
  then latent self-attention.
* **Decoder** — one query per (target family × horizon), cross-attending to the latents, and a
  per-target linear head producing (μ, log σ) for Gaussian targets or a logit for events.
* **Plug-and-play** — :meth:`FuseModel.add_modality` adds a tokenizer + modality embedding + new
  cross-attention weights; the core can be frozen. Rows without the new modality get exactly the
  old predictions (tested in tests/test_fuse.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FuseConfig:
    d: int = 128
    n_latents: int = 32
    n_layers: int = 2
    n_heads: int = 4
    tok_hidden: int = 64
    dropout: float = 0.1
    dec_layers: int = 1
    families: list = field(default_factory=list)      # target families (query embeddings)
    horizons: list = field(default_factory=lambda: [0, 1, 3, 7, 14])


def age_embedding(age: torch.Tensor, d: int) -> torch.Tensor:
    """Sinusoidal embedding of log1p(age in days); age [B] -> [B, d]."""
    x = torch.log1p(age.clamp(min=0.0)).unsqueeze(-1)
    half = d // 2
    freqs = torch.exp(torch.linspace(0.0, math.log(64.0), half, device=age.device))
    ang = x * freqs
    return torch.cat([torch.sin(ang), torch.cos(ang)], -1)


class GroupTokenizer(nn.Module):
    """G feature groups (each ≤ K features) → G tokens of width d (batched per-group MLPs)."""

    def __init__(self, group_sizes: list[int], d: int, hidden: int = 64):
        super().__init__()
        G, K = len(group_sizes), max(group_sizes)
        self.G, self.K = G, K
        self.w1 = nn.Parameter(torch.randn(G, 2 * K, hidden) / math.sqrt(2 * K))
        self.b1 = nn.Parameter(torch.zeros(G, hidden))
        self.w2 = nn.Parameter(torch.randn(G, hidden, d) / math.sqrt(hidden))
        self.b2 = nn.Parameter(torch.zeros(G, d))
        self.group_emb = nn.Parameter(torch.randn(G, d) * 0.02)
        self.mod_emb = nn.Parameter(torch.randn(d) * 0.02)
        self.age_proj = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor, m: torch.Tensor, age: torch.Tensor):
        inp = torch.cat([x * m, m], -1)                                    # [B, G, 2K]
        h = F.gelu(torch.einsum("bgk,gkh->bgh", inp, self.w1) + self.b1)
        t = torch.einsum("bgh,ghd->bgd", h, self.w2) + self.b2 + self.group_emb + self.mod_emb
        t = t + self.age_proj(age_embedding(age, t.shape[-1])).unsqueeze(1)
        tok_mask = m.sum(-1) > 0                                          # [B, G] observed group
        return self.norm(t), tok_mask


class CrossBlock(nn.Module):
    """Latents ← cross-attention over one modality's tokens (pre-norm, residual returned separately)."""

    def __init__(self, d: int, n_heads: int, dropout: float):
        super().__init__()
        self.nq = nn.LayerNorm(d)
        self.nk = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(2 * d, d))

    def forward(self, lat: torch.Tensor, tok: torch.Tensor, tok_mask: torch.Tensor) -> torch.Tensor:
        # rows with no observed token: give attention one dummy key so softmax is finite; the update is
        # discarded by the caller for those rows (present=False).
        safe = tok_mask.clone()
        safe[:, 0] |= ~tok_mask.any(1)
        a, _ = self.attn(self.nq(lat), self.nk(tok), self.nk(tok), key_padding_mask=~safe, need_weights=False)
        u = a
        u = u + self.ff(lat + u)
        return u


class SelfBlock(nn.Module):
    def __init__(self, d: int, n_heads: int, dropout: float):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(d, n_heads, 2 * d, dropout, activation="gelu", batch_first=True,
                                                norm_first=True)

    def forward(self, x):
        return self.layer(x)


class FuseModel(nn.Module):
    def __init__(self, cfg: FuseConfig, modalities: dict[str, list[int]], targets: list[tuple[str, int, str]]):
        """``modalities``: name → group sizes; ``targets``: list of (family, horizon, kind)."""
        super().__init__()
        self.cfg = cfg
        d = cfg.d
        self.mod_names: list[str] = []
        self.tokenizers = nn.ModuleDict()
        self.cross = nn.ModuleDict()       # modality → ModuleList over layers
        self.latents = nn.Parameter(torch.randn(cfg.n_latents, d) * 0.02)
        self.selfs = nn.ModuleList([SelfBlock(d, cfg.n_heads, cfg.dropout) for _ in range(cfg.n_layers)])
        for name, gs in modalities.items():
            self._add(name, gs)
        fams = sorted({f for f, _, _ in targets})
        self.fam_index = {f: i for i, f in enumerate(fams)}
        hz = sorted({h for _, h, _ in targets})
        self.hz_index = {h: i for i, h in enumerate(hz)}
        self.fam_emb = nn.Embedding(len(fams), d)
        self.hz_emb = nn.Embedding(len(hz), d)
        self.targets = targets
        self.register_buffer("q_fam", torch.tensor([self.fam_index[f] for f, _, _ in targets]), persistent=False)
        self.register_buffer("q_hz", torch.tensor([self.hz_index[h] for _, h, _ in targets]), persistent=False)
        self.dec = nn.ModuleList([CrossBlock(d, cfg.n_heads, cfg.dropout) for _ in range(cfg.dec_layers)])
        self.dec_norm = nn.LayerNorm(d)
        T = len(targets)
        self.head_w = nn.Parameter(torch.randn(T, d, 2) / math.sqrt(d) * 0.1)
        self.head_b = nn.Parameter(torch.zeros(T, 2))

    # ------------------------------------------------------------------ modalities
    def _add(self, name: str, group_sizes: list[int]) -> None:
        cfg = self.cfg
        self.mod_names.append(name)
        self.tokenizers[name] = GroupTokenizer(group_sizes, cfg.d, cfg.tok_hidden)
        self.cross[name] = nn.ModuleList([CrossBlock(cfg.d, cfg.n_heads, cfg.dropout) for _ in range(cfg.n_layers)])

    def add_modality(self, name: str, group_sizes: list[int], freeze_core: bool = True) -> list[nn.Parameter]:
        """Plug-and-play: add a new modality; returns the new trainable parameters.

        With ``freeze_core`` every pre-existing parameter gets ``requires_grad=False``.
        """
        if name in self.tokenizers:
            raise ValueError(f"modality {name} already present")
        if freeze_core:
            for p in self.parameters():
                p.requires_grad_(False)
        self._add(name, group_sizes)
        dev = self.latents.device
        self.tokenizers[name].to(dev)
        self.cross[name].to(dev)
        return list(self.tokenizers[name].parameters()) + list(self.cross[name].parameters())

    # ------------------------------------------------------------------ forward
    def encode(self, batch: dict[str, dict[str, torch.Tensor]], keep: dict[str, torch.Tensor] | None = None):
        """``batch[m]`` = {x, m, present, age}; ``keep[m]`` optional [B] bool (modality dropout)."""
        B = next(iter(batch.values()))["x"].shape[0]
        lat = self.latents.unsqueeze(0).expand(B, -1, -1)
        toks = {}
        for name in self.mod_names:
            if name not in batch:
                continue
            b = batch[name]
            t, tm = self.tokenizers[name](b["x"], b["m"], b["age"])
            pres = b["present"].bool() & tm.any(1)
            if keep is not None and name in keep:
                pres = pres & keep[name]
            toks[name] = (t, tm, pres)
        for li in range(self.cfg.n_layers):
            for name in self.mod_names:
                if name not in toks:
                    continue
                t, tm, pres = toks[name]
                if not bool(pres.any()):
                    continue
                u = self.cross[name][li](lat, t, tm)
                lat = torch.where(pres.view(B, 1, 1), lat + u, lat)
            lat = self.selfs[li](lat)
        return lat

    def decode(self, lat: torch.Tensor) -> torch.Tensor:
        B = lat.shape[0]
        q = (self.fam_emb(self.q_fam) + self.hz_emb(self.q_hz)).unsqueeze(0).expand(B, -1, -1)
        full = torch.ones(B, lat.shape[1], dtype=torch.bool, device=lat.device)
        for blk in self.dec:
            q = q + blk(q, lat, full)
        q = self.dec_norm(q)
        out = torch.einsum("btd,tdk->btk", q, self.head_w) + self.head_b   # [B, T, 2]
        return out

    def forward(self, batch, keep=None, anchor: dict | None = None) -> torch.Tensor:
        """``anchor`` (optional, persistence-anchored parametrisation): {"z": [B, T] standardised last
        valid value of each target's own variable (0 where none), "ok": [B, T] bool, "owner": list of
        modality names (or None) per target}. The anchor is added to μ only where its owner modality is
        present *and kept* (so modality dropout / LOMO also remove the anchor)."""
        out = self.decode(self.encode(batch, keep))
        if anchor is None:
            return out
        B, T = out.shape[0], out.shape[1]
        cols = []
        for o in anchor["owner"]:
            if o is None or o not in batch:
                cols.append(torch.zeros(B, dtype=torch.bool, device=out.device))
                continue
            pm = batch[o]["present"].bool()
            if keep is not None and o in keep:
                pm = pm & keep[o]
            cols.append(pm)
        use = torch.stack(cols, 1) & anchor["ok"].bool()
        add = torch.where(use, anchor["z"].to(out.dtype), torch.zeros_like(anchor["z"], dtype=out.dtype))
        return torch.stack([out[..., 0] + add, out[..., 1]], -1)


# --------------------------------------------------------------------------- losses
LOG_SIGMA_MIN, LOG_SIGMA_MAX = -4.0, 3.0


def gaussian_terms(out: torch.Tensor):
    mu = out[..., 0]
    log_s = out[..., 1].clamp(LOG_SIGMA_MIN, LOG_SIGMA_MAX)
    return mu, log_s


def masked_loss(out: torch.Tensor, y: torch.Tensor, obs: torch.Tensor, cens: torch.Tensor, is_bin: torch.Tensor,
                beta: float = 0.5) -> tuple[torch.Tensor, dict]:
    """Equal-weight mean over targets of the per-target masked mean loss.

    Gaussian: β-NLL (Seitzer et al. 2022) for observed values; left-censored values (value = bound)
    contribute −log Φ((bound − μ)/σ) (Tobit). Binary: BCE on the first output (logit).
    """
    mu, log_s = gaussian_terms(out)
    s = log_s.exp()
    yz = torch.nan_to_num(y)
    z = (yz - mu) / s
    nll = 0.5 * z ** 2 + log_s
    if beta > 0:
        nll = nll * (s.detach() ** (2 * beta))
    cnll = -torch.special.log_ndtr(z)     # stable log Φ(z): gradient does not vanish for z << 0
    reg = torch.where(cens, cnll, nll)
    bce = F.binary_cross_entropy_with_logits(mu, yz.clamp(0, 1), reduction="none")
    per = torch.where(is_bin.view(1, -1), bce, reg)
    w = obs.float()
    cnt = w.sum(0)
    per_t = (per * w).sum(0) / cnt.clamp(min=1.0)
    active = cnt > 0
    loss = per_t[active].mean() if bool(active.any()) else per_t.sum() * 0.0
    return loss, {"n_active_targets": int(active.sum())}


def crps_gaussian_t(y, mu, sigma):
    z = (y - mu) / sigma
    pdf = torch.exp(-0.5 * z ** 2) / math.sqrt(2 * math.pi)
    cdf = 0.5 * (1 + torch.erf(z / math.sqrt(2.0)))
    return sigma * (z * (2 * cdf - 1) + 2 * pdf - 1 / math.sqrt(math.pi))


def n_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only))
