# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Networks for SENTINEL-Sonde v2 (``sonde_arch_v2_amendment.md`` §2a–§2b; plan §4.0–§4.1).

* :class:`MaskedMeanNWP`: per-product, per-lead linear embeddings averaged over the AVAILABLE products only (Gauch
  et al. 2025); no zero-fill. A learned "no product" vector is used where no product is available.
* :class:`MonotoneQuantileHead`: 19 monotone quantiles (location + cumulative softplus increments).
* :class:`CMALHead`: countable mixture of asymmetric Laplacians (Klotz et al. 2022); NLL loss and quantiles by
  bisection on the mixture CDF.
* :class:`RNH`: the R-NH reference, a Nearing et al. 2024-style hindcast → forecast LSTM: a daily hindcast LSTM over
  365 d hands its (h, c) state through a linear map to a forecast LSTM that runs over the leads with NWP inputs; one
  target per model; CMAL head (default) or quantile head.
* :class:`SondeHF`: the rank-1 candidate. Daily hindcast LSTM (+ optional hourly hindcast LSTM over the last 14 d,
  initialised from the daily state at d − 14) → hand-off → forecast LSTM over the leads (masked-mean NWP embedding,
  deterministic clear-sky radiation and day length, lead one-hot) → multi-task monotone quantile heads with the
  learnable damped-anomaly anchor, and ordering constraints in ORIGINAL units (min ≤ mean ≤ max per quantile level;
  bounded hours-of-hypoxia).

Quantiles are produced in each target's standardised-anomaly space (z = (y − clim)/clim_sd), like the frozen LSTM.
Ordering constraints are applied after mapping to original units and the result is mapped back, so the loss stays
per-target normalised.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

N_Q = 19
TAUS = torch.linspace(0.05, 0.95, N_Q)


# ----------------------------------------------------------------------------- building blocks

class MaskedMeanNWP(nn.Module):
    def __init__(self, fields_per_product: list[int], d: int = 32):
        super().__init__()
        self.proj = nn.ModuleList([nn.Linear(f, d) for f in fields_per_product])
        self.none = nn.Parameter(torch.zeros(d))

    def forward(self, xs: list[torch.Tensor], masks: list[torch.Tensor]) -> torch.Tensor:
        """xs[p] (B, L, F_p) with NaN allowed where masked; masks[p] (B, L) bool → (B, L, d)."""
        num, den = 0.0, 0.0
        for proj, x, m in zip(self.proj, xs, masks):
            e = proj(torch.nan_to_num(x, nan=0.0))
            mf = m.to(e.dtype)[..., None]
            num = num + e * mf
            den = den + mf
        den_t = den if isinstance(den, torch.Tensor) else torch.zeros_like(num)
        out = torch.where(den_t > 0, num / den_t.clamp(min=1.0), self.none.to(num.dtype).expand_as(num))
        return out


class MonotoneQuantileHead(nn.Module):
    """Linear(d → n_out·Q) → monotone quantiles (n_out, Q)."""

    def __init__(self, d: int, n_out: int, q: int = N_Q):
        super().__init__()
        self.n_out, self.q = n_out, q
        self.lin = nn.Linear(d, n_out * q)
        with torch.no_grad():
            self.lin.weight.mul_(0.1)
            b = torch.full((n_out, q), float(math.log(math.expm1(3.29 / (q - 1)))))
            b[:, 0] = -1.645
            self.lin.bias.copy_(b.reshape(-1))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        raw = self.lin(h).float().view(*h.shape[:-1], self.n_out, self.q)
        inc = F.softplus(raw[..., 1:])
        return torch.cat([raw[..., :1], raw[..., :1] + torch.cumsum(inc, dim=-1)], dim=-1)


class CMALHead(nn.Module):
    """K-component asymmetric-Laplace mixture: params (π, μ, b, τ) per output."""

    def __init__(self, d: int, n_out: int, k: int = 3):
        super().__init__()
        self.n_out, self.k = n_out, k
        self.lin = nn.Linear(d, n_out * k * 4)

    def forward(self, h: torch.Tensor) -> dict:
        p = self.lin(h).float().view(*h.shape[:-1], self.n_out, self.k, 4)
        return {"logit": p[..., 0], "mu": p[..., 1], "b": F.softplus(p[..., 2]) + 1e-3,
                "tau": torch.sigmoid(p[..., 3]).clamp(0.02, 0.98)}

    @staticmethod
    def nll(par: dict, y: torch.Tensor) -> torch.Tensor:
        """y (..., n_out) with NaN = missing → mean NLL over present entries."""
        m = torch.isfinite(y)
        yy = torch.where(m, y, torch.zeros_like(y))[..., None]
        mu, b, tau = par["mu"], par["b"], par["tau"]
        u = (yy - mu) / b
        rho = torch.where(u >= 0, tau * u, (tau - 1) * u)
        logp = torch.log(tau * (1 - tau) / b) - rho
        lse = torch.logsumexp(torch.log_softmax(par["logit"], -1) + logp, dim=-1)
        return -torch.where(m, lse, torch.zeros_like(lse)).sum() / m.sum().clamp(min=1)

    @staticmethod
    def cdf(par: dict, y: torch.Tensor) -> torch.Tensor:
        mu, b, tau = par["mu"], par["b"], par["tau"]
        u = (y[..., None] - mu) / b
        c = torch.where(u < 0, tau * torch.exp((1 - tau) * u), 1 - (1 - tau) * torch.exp(-tau * u))
        return (torch.softmax(par["logit"], -1) * c).sum(-1)

    @classmethod
    def quantiles(cls, par: dict, taus: torch.Tensor = TAUS, iters: int = 60) -> torch.Tensor:
        """(..., n_out, Q) by bisection on the mixture CDF."""
        mu, b = par["mu"], par["b"]
        lo = (mu - 60 * b).min(-1).values[..., None].expand(*mu.shape[:-1], len(taus)).clone()
        hi = (mu + 60 * b).max(-1).values[..., None].expand(*mu.shape[:-1], len(taus)).clone()
        t = taus.to(mu.device).view(*([1] * (mu.dim() - 1)), -1)
        p = {k: v[..., None, :] for k, v in par.items()}
        for _ in range(iters):
            mid = (lo + hi) / 2
            below = cls.cdf(p, mid) < t
            lo = torch.where(below, mid, lo)
            hi = torch.where(below, hi, mid)
        return (lo + hi) / 2


def pinball(q: torch.Tensor, y: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    """q (..., n, Q), y (..., n) NaN = missing; masked mean pinball, optionally weighted per target (last dim n)."""
    taus = TAUS.to(q.device)
    m = torch.isfinite(y)
    yy = torch.where(m, y, torch.zeros_like(y))
    d = yy[..., None] - q
    loss = torch.where(m, torch.maximum(taus * d, (taus - 1) * d).mean(-1), torch.zeros_like(yy))
    if weights is not None:
        loss = loss * weights
        return loss.sum() / (m * weights).sum().clamp(min=1e-6)
    return loss.sum() / m.sum().clamp(min=1)


class StaticNet(nn.Module):
    def __init__(self, n_static: int, n_huc: int, huc_emb: int = 8, d: int = 32):
        super().__init__()
        self.huc = nn.Embedding(n_huc, huc_emb)
        self.mlp = nn.Sequential(nn.Linear(n_static + huc_emb, d), nn.ReLU())

    def forward(self, x_static, huc):
        return self.mlp(torch.cat([x_static, self.huc(huc)], -1))


# ----------------------------------------------------------------------------- R-NH reference

class RNH(nn.Module):
    """Hindcast LSTM (daily) → state hand-off → forecast LSTM over NWP; single target; CMAL or quantile head."""

    def __init__(self, n_dyn: int, n_static: int, n_huc: int, nwp_fields: list[int], *, L: int = 14,
                 hidden: int = 128, nwp_d: int = 32, head: str = "cmal", cmal_k: int = 3, dropout: float = 0.2):
        super().__init__()
        self.L, self.head_kind = L, head
        self.static = StaticNet(n_static, n_huc)
        self.hind = nn.LSTM(n_dyn + 32, hidden, batch_first=True)
        self.handoff = nn.Linear(2 * hidden, 2 * hidden)
        self.nwp = MaskedMeanNWP(nwp_fields, nwp_d)
        self.fore = nn.LSTM(nwp_d + 32 + L + 4, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = CMALHead(hidden + 32, 1, cmal_k) if head == "cmal" else MonotoneQuantileHead(hidden + 32, 1)

    def forward(self, x_dyn, x_static, huc, nwp_x: list, nwp_m: list, x_fdet):
        """x_dyn (B, C, n_dyn); nwp_x[p] (B, L, F_p); x_fdet (B, L, 4) deterministic future inputs (clear-sky SW,
        day length, DOY sin/cos — the same as SONDE-HF's forecast branch).
        Returns CMAL params with shape (B, L, 1, ·) or quantiles (B, L, 1, Q)."""
        st = self.static(x_static, huc)
        B, C, _ = x_dyn.shape
        _, (h, c) = self.hind(torch.cat([x_dyn, st[:, None].expand(-1, C, -1)], -1))
        h0, c0 = self.handoff(torch.cat([h[-1], c[-1]], -1)).chunk(2, -1)
        e = self.nwp(nwp_x, nwp_m)
        onehot = torch.eye(self.L, device=x_dyn.device, dtype=e.dtype)[None].expand(B, -1, -1)
        z = torch.cat([e, st[:, None].expand(-1, self.L, -1), onehot, x_fdet], -1)
        out, _ = self.fore(z, (torch.tanh(h0)[None].contiguous(), c0[None].contiguous()))
        return self.head(torch.cat([self.drop(out), st[:, None].expand(-1, self.L, -1)], -1))

    def quantiles(self, out) -> torch.Tensor:
        return CMALHead.quantiles(out) if self.head_kind == "cmal" else out


# ----------------------------------------------------------------------------- SONDE-HF candidate

class SondeHF(nn.Module):
    """Multi-resolution hindcast → forecast, multi-task, ordering-constrained.

    ``targets``: list of target names; ``order``: list of (child, parent, sign) with sign −1 (child ≤ parent, e.g.
    DO min ≤ DO mean) or +1 (child ≥ parent); ``bounded``: {target: (lo, hi)} mapped via lo + (hi − lo)·sigmoid
    in original units (hours of hypoxia ∈ [0, 24])."""

    def __init__(self, n_dyn: int, n_static: int, n_huc: int, nwp_fields: list[int], targets: list[str], *,
                 n_hourly: int = 0, L: int = 14, hidden_d: int = 256, hidden_h: int = 128, hidden_f: int = 128,
                 nwp_d: int = 32, order: list | None = None, bounded: dict | None = None,
                 anchor_targets: list | None = None, dropout: float = 0.2, head: str = "quantile",
                 cmal_k: int = 3):
        super().__init__()
        if head not in ("quantile", "cmal"):
            raise ValueError(head)
        self.L, self.targets, self.head_kind = L, list(targets), head
        self.T = len(targets)
        self.static = StaticNet(n_static, n_huc)
        self.daily = nn.LSTM(n_dyn + 32, hidden_d, batch_first=True)
        self.use_hourly = n_hourly > 0
        if self.use_hourly:
            self.h_init = nn.Linear(2 * hidden_d, 2 * hidden_h)
            self.hourly = nn.LSTM(n_hourly, hidden_h, batch_first=True)
            self.no_hourly = nn.Parameter(torch.zeros(hidden_h))
        d_cat = hidden_d + (hidden_h if self.use_hourly else 0)
        self.handoff = nn.Linear(d_cat, 2 * hidden_f)
        self.nwp = MaskedMeanNWP(nwp_fields, nwp_d)
        self.fore = nn.LSTM(nwp_d + 32 + L + 4, hidden_f, batch_first=True)
        self.drop = nn.Dropout(dropout)
        # C1/C2/C3/C5: monotone 19-quantile head trained with the pinball loss; C4: CMAL head trained with its NLL
        self.head = (CMALHead(hidden_f + 32, self.T, cmal_k) if head == "cmal"
                     else MonotoneQuantileHead(hidden_f + 32, self.T))
        self.order = [(self.targets.index(c), self.targets.index(p), s) for c, p, s in (order or [])]
        self.bounded = {self.targets.index(t): lim for t, lim in (bounded or {}).items()}
        self.anchor_idx = [self.targets.index(t) for t in (anchor_targets or [])]
        if self.anchor_idx:
            self.anchor_w = nn.Parameter(0.9 ** torch.arange(1, L + 1, dtype=torch.float32)[None].repeat(
                len(self.anchor_idx), 1))

    def forward(self, x_dyn, x_static, huc, nwp_x: list, nwp_m: list, x_fdet, *, x_hourly=None, has_hourly=None,
                a_last=None, clim_mean=None, clim_sd=None):
        """x_dyn (B, C, n_dyn) daily (C ≥ 15); x_hourly (B, 336, n_hourly) or None; has_hourly (B,) bool;
        x_fdet (B, L, 4) deterministic future inputs (clear-sky SW, day length, DOY sin/cos);
        a_last (B, n_anchor) last standardised anomaly of the anchored targets (0 if none);
        clim_mean/clim_sd (B, L, T) target-day climatology in original units (needed for ordering/bounds).
        Returns quantiles (B, L, T, Q) in standardised-anomaly space of each target (quantile head), or the CMAL
        parameter dict with shapes (B, L, T, K) (CMAL head; the anchor shift is added to every component location;
        ordering/bounds are applied to its quantiles in :meth:`quantiles`, not inside the likelihood)."""
        st = self.static(x_static, huc)
        B, C, _ = x_dyn.shape
        xin = torch.cat([x_dyn, st[:, None].expand(-1, C, -1)], -1)
        if self.use_hourly:
            out_a, (h_a, c_a) = self.daily(xin[:, : C - 14])
            _, (h_d, c_d) = self.daily(xin[:, C - 14:], (h_a, c_a))
            if x_hourly is not None:
                h0, c0 = self.h_init(torch.cat([h_a[-1], c_a[-1]], -1)).chunk(2, -1)
                _, (hh, _) = self.hourly(torch.nan_to_num(x_hourly, nan=0.0),
                                         (torch.tanh(h0)[None].contiguous(), c0[None].contiguous()))
                hh = hh[-1]
                if has_hourly is not None:
                    hh = torch.where(has_hourly[:, None], hh, self.no_hourly.to(hh.dtype).expand_as(hh))
            else:
                hh = self.no_hourly.to(h_d.dtype).expand(B, -1)
            hcat = torch.cat([h_d[-1], hh], -1)
        else:
            _, (h_d, _) = self.daily(xin)
            hcat = h_d[-1]
        h0, c0 = self.handoff(hcat).chunk(2, -1)
        e = self.nwp(nwp_x, nwp_m)
        onehot = torch.eye(self.L, device=x_dyn.device, dtype=e.dtype)[None].expand(B, -1, -1)
        z = torch.cat([e, st[:, None].expand(-1, self.L, -1), onehot, x_fdet], -1)
        out, _ = self.fore(z, (torch.tanh(h0)[None].contiguous(), c0[None].contiguous()))
        q = self.head(torch.cat([self.drop(out), st[:, None].expand(-1, self.L, -1)], -1))   # (B, L, T, Q)
        shift = None
        if self.anchor_idx and a_last is not None:
            shift = torch.zeros(B, self.L, self.T, device=x_dyn.device)
            shift[:, :, self.anchor_idx] = (self.anchor_w.T[None] * a_last[:, None, :]).float()
        if self.head_kind == "cmal":
            if shift is not None:
                q = {**q, "mu": q["mu"] + shift[..., None]}
            return q
        if shift is not None:
            q = q + shift[..., None]
        if (self.order or self.bounded) and clim_mean is not None:
            q = self.constrain(q, clim_mean.float(), clim_sd.float().clamp(min=1e-6))
        return q

    def quantiles(self, out, clim_mean=None, clim_sd=None) -> torch.Tensor:
        """(B, L, T, Q) standardised-anomaly quantiles from the forward output (identity for the quantile head;
        mixture quantiles by bisection, then the same ordering/bounds step, for the CMAL head)."""
        if self.head_kind != "cmal":
            return out
        q = CMALHead.quantiles(out)
        if (self.order or self.bounded) and clim_mean is not None:
            q = self.constrain(q, clim_mean.float(), clim_sd.float().clamp(min=1e-6))
        return q

    def constrain(self, q, cm, cs):
        """Ordering and bounds in ORIGINAL units, per quantile level; then back to standardised space."""
        cm = torch.nan_to_num(cm, nan=0.0)                         # targets without climatology are masked anyway
        cs = torch.nan_to_num(cs, nan=1.0).clamp(min=1e-6)
        y = cm[..., None] + cs[..., None] * q                     # (B, L, T, Q)
        cols = list(y.unbind(2))
        for c, p, s in self.order:                                # child = sort(parent ± softplus(child-parent gap))
            gap = F.softplus((cols[c] - cols[p]) * s)
            cols[c] = torch.sort(cols[p] + s * gap, dim=-1).values
        for t, (lo, hi) in self.bounded.items():
            cols[t] = lo + (hi - lo) * torch.sigmoid(cols[t])
        y = torch.stack(cols, 2)
        return (y - cm[..., None]) / cs[..., None]
