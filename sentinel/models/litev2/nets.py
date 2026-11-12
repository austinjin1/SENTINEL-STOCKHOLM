# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Networks of M7 architecture round v2 (amendment §4.3–§4.5, §5 P3a / P3b).

* :class:`AnchorLite` — the leader's image branch (frozen class ``LiteStudent``, used through its sub-modules and
  never edited) + FiLM from [s, c] + centre-window and attention pooling + residual heads on the GBM-parity anchor:
  ``μ_t = a_t + κ_t · tanh(Δ_t / κ_t)``. Switches give the ablations A1 (no anchor), A3 (no context), A4 (four
  targets) and A5 (centre pooling only).
* :class:`WaterSet` — ANCHOR-Set (§4.4): a set transformer over sampled water pixels with the same residual heads.
* :class:`DenseNetRegressor` — P3a (image only) and P3b (+ [s, c] and a chip-level KD head).

All heads work in log10 units directly: every model returns ``mu`` (N, T) and ``logvar`` (N, T), log σ² clamped to
(−8, 6) as in the frozen model. ``sc`` is [s (51), c (5)] standardised with train-row statistics kept in buffers;
a missing scene value is the train mean (0 after standardisation). The anchor ``a`` enters unstandardised.

Readings decided here (listed for the lead in the implementation notes of this round):
* the auxiliary KD head is the leader's map head on the FiLM-modulated decoder output — the same feature map the
  pooling reads, so at initialisation (FiLM = identity) it returns the leader's map exactly;
* an output cell is a "dark-NIR water cell" for the attention pooling when at least half of its 4 × 4 pixels are in
  the frozen ``features.water_mask``;
* the radial terms are a·r + b·r² with r in metres; a and b are stored per 100 m and per (100 m)² so that one
  optimiser step moves the logits by a comparable amount (the function family is the same);
* only the Δ row of each head's last layer is zero-initialised; the log σ² row keeps its default weights and gets
  the bias of §4.3.
"""

from __future__ import annotations

import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as Fn

from ..lite.model import LiteStudent
from . import labels2b as L

N_S, N_C = 51, 5
N_SC = N_S + N_C
KAPPA = {"secchi": 0.3}                       # log10 units; 0.5 for every other target (§4.3)
KAPPA_DEFAULT = 0.5
LOGVAR_RANGE = (-8.0, 6.0)
R_UNIT_M = 100.0
STRIDE = 4                                    # the decoder's output stride
WATER_CELL_MIN = 0.5
N_TEACHER = len(L.ORIG)


def _mlp(sizes, last_relu: bool = False) -> nn.Sequential:
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or last_relu:
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def kappa_vector(targets) -> torch.Tensor:
    return torch.tensor([KAPPA.get(t, KAPPA_DEFAULT) for t in targets], dtype=torch.float32)


class ResidualHeads(nn.Module):
    """e = MLP(context) → per target MLP([pooled, e]) → (Δ_t, log σ²_t); μ from the anchor or from a base value."""

    def __init__(self, n_pooled: int, n_context: int, targets, *, use_anchor: bool = True, width: int = 64):
        super().__init__()
        self.targets = tuple(targets)
        self.use_anchor = use_anchor
        self.embed = _mlp([n_context, width, width], last_relu=True)
        self.heads = nn.ModuleList([_mlp([n_pooled + width, width, 2]) for _ in self.targets])
        for h in self.heads:
            nn.init.zeros_(h[-1].weight[0])
            nn.init.zeros_(h[-1].bias)
        self.register_buffer("kappa", kappa_vector(self.targets))
        self.register_buffer("base", torch.zeros(len(self.targets)))     # train median, used when there is no anchor
        self.ntru_offset = nn.Parameter(torch.zeros(()))                # NTRU → NTU head, additive log10 (§3.2)

    def set_logvar_bias(self, variance) -> None:
        """log σ² bias = log of the train out-of-fold residual variance of the anchor (of the base without one)."""
        with torch.no_grad():
            for h, v in zip(self.heads, variance):
                h[-1].bias[1] = float(min(max(math.log(max(float(v), 1e-8)), LOGVAR_RANGE[0]), LOGVAR_RANGE[1]))

    def forward(self, pooled: torch.Tensor, context: torch.Tensor, anchor: torch.Tensor | None) -> dict:
        e = self.embed(context)
        z = torch.cat([pooled, e], 1)
        out = torch.stack([h(z) for h in self.heads], 1).float()        # (N, T, 2)
        delta, lv = out[..., 0], out[..., 1].clamp(*LOGVAR_RANGE)
        if self.use_anchor:
            mu = anchor.float() + self.kappa * torch.tanh(delta / self.kappa)
        else:
            mu = self.base + delta
        return {"mu": mu, "logvar": lv, "delta": delta}


class _Context(nn.Module):
    """Standardisation of [s, c] with train-row statistics (buffers; stored in the checkpoint)."""

    def __init__(self):
        super().__init__()
        self.register_buffer("sc_mean", torch.zeros(N_SC))
        self.register_buffer("sc_std", torch.ones(N_SC))

    def fit(self, sc_train: torch.Tensor) -> None:
        x = sc_train.float()
        ok = torch.isfinite(x)
        n = ok.sum(0).clamp_min(1)
        mean = torch.where(ok, x, torch.zeros_like(x)).sum(0) / n
        var = (torch.where(ok, x - mean, torch.zeros_like(x)) ** 2).sum(0) / n
        self.sc_mean.copy_(mean)
        self.sc_std.copy_(var.sqrt().clamp_min(1e-6))

    def forward(self, sc: torch.Tensor) -> torch.Tensor:
        z = (sc.float() - self.sc_mean) / self.sc_std
        return torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)


class AnchorLite(nn.Module):
    def __init__(self, *, targets=L.TARGETS, use_anchor: bool = True, use_context: bool = True,
                 attention: bool = True, backbone: str = "mnv4s", dec_ch: int = 48, pretrained: bool = False):
        super().__init__()
        self.targets = tuple(targets)
        self.use_anchor, self.use_context, self.attention = use_anchor, use_context, attention
        self.dec_ch = dec_ch
        self.student = LiteStudent(backbone, pretrained=pretrained, dec_ch=dec_ch)      # image branch + KD map head
        self.ctx = _Context()
        self.film = _mlp([N_SC, 64, 2 * dec_ch]) if use_context else None
        if self.film is not None:
            nn.init.zeros_(self.film[-1].weight)
            nn.init.zeros_(self.film[-1].bias)
        if attention:
            self.att_conv = nn.Conv2d(dec_ch, 1, 1)
            self.att_a = nn.Parameter(torch.zeros(()))
            self.att_b = nn.Parameter(torch.zeros(()))
        n_ctx = (N_SC if use_context else 0) + (len(self.targets) if use_anchor else 0)
        self.res = ResidualHeads(dec_ch * (2 if attention else 1), n_ctx, self.targets, use_anchor=use_anchor)

    # the leader's modules (initialised from config 3) and the new ones get different learning rates (§4.3)
    def leader_parameters(self):
        return list(self.student.parameters())

    def new_parameters(self):
        ids = {id(p) for p in self.student.parameters()}
        return [p for p in self.parameters() if id(p) not in ids]

    def load_leader(self, state_dict: dict) -> None:
        """Weights and buffers (x / y statistics) of a ``LiteStudent`` checkpoint of the leader's configuration."""
        self.student.load_state_dict(state_dict, strict=True)

    def decoder(self, x: torch.Tensor) -> torch.Tensor:
        """The leader's stride-4 decoder output (the code path of ``LiteStudent.forward`` up to ``smooth``)."""
        st = self.student
        feats = st.encoder(x)
        top = st.lateral[-1](feats[-1])
        for f, lat in zip(reversed(feats[:-1]), reversed(list(st.lateral[:-1]))):
            top = Fn.interpolate(top, size=f.shape[-2:], mode="nearest") + lat(f)
        return st.smooth(top)

    def attention_weights(self, top: torch.Tensor, water: torch.Tensor, gsd: torch.Tensor) -> torch.Tensor:
        """(N, h, w) weights summing to 1 over the water cells of each chip (all cells when it has none)."""
        n, _, h, w = top.shape
        cell_m = gsd.float().view(n, 1, 1) * STRIDE                      # size of an output cell in metres
        yy = (torch.arange(h, device=top.device, dtype=torch.float32) - (h - 1) / 2.0).view(1, h, 1)
        xx = (torch.arange(w, device=top.device, dtype=torch.float32) - (w - 1) / 2.0).view(1, 1, w)
        r = torch.sqrt(yy ** 2 + xx ** 2) * cell_m / R_UNIT_M           # distance to the chip centre, in 100 m
        logit = self.att_conv(top.float())[:, 0] + self.att_a * r + self.att_b * r ** 2
        m = water >= WATER_CELL_MIN
        m = m | ~m.flatten(1).any(1).view(n, 1, 1)
        logit = logit.masked_fill(~m, float("-inf"))
        return torch.softmax(logit.flatten(1), 1).view(n, h, w)

    def forward(self, x: torch.Tensor, sc: torch.Tensor, anchor: torch.Tensor | None, pool_w: torch.Tensor,
                water: torch.Tensor, gsd: torch.Tensor) -> dict:
        """x (N, 4, H, W) normalised as the leader's raw mode; sc (N, 56) raw [s, c]; anchor (N, T) log10;
        pool_w (N, h, w) frozen centre-window weights; water (N, h, w) water fraction per cell; gsd (N,) metres."""
        top = self.decoder(x)
        z = self.ctx(sc)
        if self.film is not None:
            g, b = self.film(z).chunk(2, 1)
            top = top * (1 + g[:, :, None, None]) + b[:, :, None, None]
        aux = self.student.head(top)
        nt = self.student.n_targets
        aux = torch.cat([aux[:, :nt], aux[:, nt:].clamp(*LOGVAR_RANGE)], 1)
        topf = top.float()
        pooled = (topf * pool_w.unsqueeze(1)).sum((2, 3))
        if self.attention:
            aw = self.attention_weights(topf, water, gsd)
            pooled = torch.cat([pooled, (topf * aw.unsqueeze(1)).sum((2, 3))], 1)
        parts = ([z] if self.use_context else []) + ([anchor.float()] if self.use_anchor else [])
        out = self.res(pooled, torch.cat(parts, 1), anchor)
        out["aux"] = aux
        out["pooled"] = pooled
        return out


# ------------------------------------------------------------------------------------------------ ANCHOR-Set
N_PIXEL_FEATURES = 19
SET_MAX, SET_MIN = 1024, 64


class _MAB(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.att = nn.MultiheadAttention(width, heads, batch_first=True)
        self.n1, self.n2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.ff = _mlp([width, 2 * width, width])

    def forward(self, q: torch.Tensor, k: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.n1(q + self.att(q, k, k, key_padding_mask=key_padding_mask, need_weights=False)[0])
        return self.n2(h + self.ff(h))


class _ISAB(nn.Module):
    def __init__(self, width: int, heads: int, n_inducing: int):
        super().__init__()
        self.inducing = nn.Parameter(torch.randn(1, n_inducing, width) * 0.1)
        self.m1, self.m2 = _MAB(width, heads), _MAB(width, heads)

    def forward(self, x: torch.Tensor, pad: torch.Tensor | None = None) -> torch.Tensor:
        h = self.m1(self.inducing.expand(x.shape[0], -1, -1), x, pad)
        return self.m2(x, h)


class WaterSet(nn.Module):
    """ANCHOR-Set: Linear 19 → 64; three global tokens from s, c and a; two induced set-attention blocks (16
    inducing points, width 64, 4 heads); pooling by multi-head attention with 4 seeds; the residual heads of
    ANCHOR-Lite on the 4 × 64 pooled values. No dense map and no KD term."""

    def __init__(self, *, targets=L.TARGETS, width: int = 64, heads: int = 4, n_inducing: int = 16, n_seeds: int = 4):
        super().__init__()
        self.targets = tuple(targets)
        self.ctx = _Context()
        self.register_buffer("px_mean", torch.zeros(N_PIXEL_FEATURES))
        self.register_buffer("px_std", torch.ones(N_PIXEL_FEATURES))
        self.embed_px = nn.Linear(N_PIXEL_FEATURES, width)
        self.tok_s, self.tok_c = nn.Linear(N_S, width), nn.Linear(N_C, width)
        self.tok_a = nn.Linear(len(self.targets), width)
        self.blocks = nn.ModuleList([_ISAB(width, heads, n_inducing) for _ in range(2)])
        self.seeds = nn.Parameter(torch.randn(1, n_seeds, width) * 0.1)
        self.pma = _MAB(width, heads)
        self.res = ResidualHeads(n_seeds * width, N_SC + len(self.targets), self.targets, use_anchor=True)

    def fit_pixels(self, px: torch.Tensor, valid: torch.Tensor) -> None:
        v = valid.unsqueeze(-1).float()
        n = v.sum((0, 1)).clamp_min(1)
        mean = (px.float() * v).sum((0, 1)) / n
        var = (((px.float() - mean) ** 2) * v).sum((0, 1)) / n
        self.px_mean.copy_(mean)
        self.px_std.copy_(var.sqrt().clamp_min(1e-6))

    def forward(self, px: torch.Tensor, valid: torch.Tensor, sc: torch.Tensor, anchor: torch.Tensor) -> dict:
        """px (N, P, 19) pixel features; valid (N, P) bool (False = padding); sc (N, 56); anchor (N, T)."""
        z = self.ctx(sc)
        a = anchor.float()
        tok = torch.stack([self.tok_s(z[:, :N_S]), self.tok_c(z[:, N_S:]), self.tok_a(a)], 1)
        x = torch.cat([tok, self.embed_px((px.float() - self.px_mean) / self.px_std)], 1)
        pad = torch.cat([torch.zeros(len(x), 3, dtype=torch.bool, device=x.device), ~valid], 1)
        for blk in self.blocks:
            x = blk(x, pad)
        pooled = self.pma(self.seeds.expand(len(x), -1, -1), x, pad).flatten(1)
        out = self.res(pooled, torch.cat([z, a], 1), a)
        out["pooled"] = pooled
        return out


# ------------------------------------------------------------------------------------------------ P3a / P3b
_TV_KEY = re.compile(r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$")


def densenet_state(path) -> dict:
    """The torchvision DenseNet121 ImageNet state dict from the local file (opened with ``weights_only=True``),
    keys renamed from the old 'norm.1' form to the current 'norm1' form as torchvision does on load."""
    sd = torch.load(path, map_location="cpu", weights_only=True)
    out = {}
    for k, v in sd.items():
        m = _TV_KEY.match(k)
        out[m.group(1) + m.group(2) if m else k] = v
    return out


class DenseNetRegressor(nn.Module):
    """HydroVision-style DenseNet121 regressor. P3a: global pool → Linear(1024 → 2 per target). P3b: pooled 1024 ⊕
    MLP([s, c]) 64 → 256 → heads, plus a chip-level head on the chip-mean teacher μ of the four teacher targets."""

    def __init__(self, *, targets=L.TARGETS, parity: bool = False, weights: dict | None = None):
        super().__init__()
        import torchvision

        self.targets = tuple(targets)
        self.parity = parity
        net = torchvision.models.densenet121(weights=None)
        if weights is not None:
            net.load_state_dict(weights, strict=True)
        w = net.features.conv0.weight.data                                  # (64, 3, 7, 7)
        conv = nn.Conv2d(4, w.shape[0], 7, stride=2, padding=3, bias=False)
        conv.weight.data.copy_(torch.cat([w, w.mean(1, keepdim=True)], 1))  # NIR filter = mean of the RGB filters
        net.features.conv0 = conv
        self.features = net.features
        self.register_buffer("x_mean", torch.zeros(4))
        self.register_buffer("x_std", torch.ones(4))
        T = len(self.targets)
        if parity:
            self.ctx = _Context()
            self.sc_mlp = _mlp([N_SC, 64], last_relu=True)
            self.trunk = _mlp([1024 + 64, 256], last_relu=True)
            self.out = nn.Linear(256, 2 * T)
            self.kd = nn.Linear(256, N_TEACHER)
        else:
            self.out = nn.Linear(1024, 2 * T)
        self.ntru_offset = nn.Parameter(torch.zeros(()))

    def backbone_parameters(self):
        return list(self.features.parameters())

    def new_parameters(self):
        ids = {id(p) for p in self.features.parameters()}
        return [p for p in self.parameters() if id(p) not in ids]

    def set_output_bias(self, median, variance) -> None:
        """μ bias = train median, log σ² bias = log train variance (log10 units; the heads are not standardised)."""
        T = len(self.targets)
        with torch.no_grad():
            self.out.bias[:T] = torch.as_tensor(median, dtype=torch.float32)
            lv = torch.log(torch.as_tensor(variance, dtype=torch.float32).clamp_min(1e-8)).clamp(*LOGVAR_RANGE)
            self.out.bias[T:] = lv

    def forward(self, x: torch.Tensor, sc: torch.Tensor | None = None) -> dict:
        f = Fn.adaptive_avg_pool2d(Fn.relu(self.features(x)), 1).flatten(1)
        res = {}
        if self.parity:
            f = self.trunk(torch.cat([f, self.sc_mlp(self.ctx(sc))], 1))
            res["kd"] = self.kd(f).float()
        o = self.out(f).float()
        T = len(self.targets)
        res.update({"mu": o[:, :T], "logvar": o[:, T:].clamp(*LOGVAR_RANGE), "pooled": f.float()})
        return res


def count_parameters(m: nn.Module) -> int:
    return int(sum(p.numel() for p in m.parameters()))


__all__ = ["N_S", "N_C", "N_SC", "KAPPA", "LOGVAR_RANGE", "N_PIXEL_FEATURES", "SET_MAX", "SET_MIN", "ResidualHeads",
           "AnchorLite", "WaterSet", "DenseNetRegressor", "densenet_state", "kappa_vector", "count_parameters"]
