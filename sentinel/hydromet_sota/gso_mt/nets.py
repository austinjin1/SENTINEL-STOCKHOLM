# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Networks and the diel observation operator of the M6 architecture-v2 round (amendment §5 R-TD, §6.1 D1 GSO-MT).

Every network returns the same output dict over the decoded days (the last ``L + 6`` days of a window), so one
trainer, one loss and one criterion serve D1, its ablations, R-TD and D2:

``mu``      (B, Ld, 2)      daily mean of T and DO (model space: standardised with the T_mean / DO_mean scale)
``curve``   (B, Ld, 24, 2)  zero-mean diel curve at local solar hours 0..23, or None (no diel information)
``direct``  (B, Ld, 5)      independent heads for T max, T min, T 7-DADMax, DO min, DO max, or None
``sig_d``   (B, Ld, 7)      σ of the daily statistics (order of ``data.DAILY[:7]``)
``sig_i``   (B, Ld, 2)      σ of an instantaneous value
``aux_mu`` / ``aux_sig`` (B, Ld, 4)  log discharge, specific conductance, turbidity, nitrate

Diel curve of D1: g(h) = A[cos(x − θ) + κ cos(2(x − θ))], x = 2πh/24, κ ∈ [0, 0.5]; both harmonics have zero daily
mean. Daily max/min = μ + max/min of the 24-point curve; 7-DADMax = trailing 7-day mean of the daily max (in-model).
An instantaneous value at hour h is μ + the curve interpolated linearly (periodic) at h."""

from __future__ import annotations

import math

from .solar import MIX_HOURS

N_DAILY, N_AUX, PAD = 7, 4, 6
HEADS = ("diel", "indep_op", "indep")


def _torch():
    import torch

    return torch


def diel_curve(amp, c, s, kappa):
    """(…, 24) zero-mean curve from amplitude ≥ 0, unit phase vector (c, s) = (cos θ, sin θ) and skew κ."""
    torch = _torch()
    x = torch.arange(24, device=amp.device, dtype=amp.dtype) * (2.0 * math.pi / 24.0)
    cx, sx, c2x, s2x = torch.cos(x), torch.sin(x), torch.cos(2 * x), torch.sin(2 * x)
    c, s, amp, kappa = c[..., None], s[..., None], amp[..., None], kappa[..., None]
    return amp * (cx * c + sx * s + kappa * (c2x * (c * c - s * s) + s2x * (2.0 * s * c)))


def interp_curve(curve, h):
    """Curve value at local solar hour ``h`` (same leading shape as ``curve`` without its hour axis, or with one extra
    trailing axis of several hours): periodic linear interpolation between the 24 grid points."""
    torch = _torch()
    extra = h.dim() == curve.dim()
    hh = h if extra else h[..., None]
    hh = torch.remainder(hh, 24.0)
    i0 = torch.clamp(torch.floor(hh).long(), 0, 23)
    w = hh - i0.to(hh.dtype)
    i1 = torch.remainder(i0 + 1, 24)
    v = torch.gather(curve, -1, i0) * (1.0 - w) + torch.gather(curve, -1, i1) * w
    return v if extra else v[..., 0]


def daily_stats(out: dict, L: int):
    """(B, L, 7) point predictions of T mean, T max, T min, T 7-DADMax, DO mean, DO min, DO max on the last L days."""
    torch = _torch()
    mu = out["mu"]
    if out.get("direct") is not None:
        d = out["direct"]
        full = torch.stack([mu[..., 0], d[..., 0], d[..., 1], d[..., 2], mu[..., 1], d[..., 3], d[..., 4]], dim=-1)
        return full[:, -L:]
    cv = out["curve"]
    tmax = mu[..., 0] + cv[..., 0].amax(dim=-1)
    tmin = mu[..., 0] + cv[..., 0].amin(dim=-1)
    cs = torch.cumsum(torch.nn.functional.pad(tmax, (1, 0)), dim=1)
    dadm = (cs[:, 7:] - cs[:, :-7]) / 7.0                      # trailing 7-day mean; (B, Ld − 6)
    return torch.stack([mu[:, -L:, 0], tmax[:, -L:], tmin[:, -L:], dadm[:, -L:], mu[:, -L:, 1],
                        mu[:, -L:, 1] + cv[:, -L:, :, 1].amin(dim=-1), mu[:, -L:, 1] + cv[:, -L:, :, 1].amax(dim=-1)],
                       dim=-1)


def inst_point(out: dict, L: int, hour):
    """(B, L, 2) point prediction of an instantaneous value at ``hour`` (B, L, 2); NaN hour = no time → mean of the
    predictions over 08–17 h."""
    torch = _torch()
    mu = out["mu"][:, -L:]
    if out.get("curve") is None:
        return mu
    cv = out["curve"][:, -L:].permute(0, 1, 3, 2)               # (B, L, 2, 24)
    timed = torch.isfinite(hour)
    g = interp_curve(cv, torch.where(timed, hour, torch.zeros_like(hour)))
    gm = cv[..., torch.as_tensor(MIX_HOURS, device=cv.device)].mean(dim=-1)
    return mu + torch.where(timed, g, gm)


def make_net(n_dyn: int, n_static: int, cfg: dict):
    """``cfg``: static (``film`` | ``concat``), heads (one of :data:`HEADS`), hidden, dropout, static_emb,
    static_hidden, static_noise, attr_dropout, layers.

    ``layers`` (default 1; only R-TD-W has more, lead ruling L11): number of stacked LSTM layers. With one layer the
    dropout acts on the LSTM output before the heads; with more than one it acts **between the LSTM layers** (the
    ``dropout`` argument of ``torch.nn.LSTM``) and not on the output of the last layer."""
    torch = _torch()
    nn = torch.nn
    heads, static_mode = cfg.get("heads", "diel"), cfg.get("static", "film")
    if heads not in HEADS or static_mode not in ("film", "concat"):
        raise ValueError(f"unknown heads / static mode: {heads} / {static_mode}")
    hidden, emb = int(cfg["hidden"]), int(cfg.get("static_emb", 32))
    layers, p_drop = int(cfg.get("layers", 1)), float(cfg.get("dropout", 0.0))
    if layers < 1:
        raise ValueError("layers must be >= 1")
    lstm_kw = {"batch_first": True} if layers == 1 else {"batch_first": True, "num_layers": layers, "dropout": p_drop}

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.heads, self.static_mode = heads, static_mode
            self.static_noise, self.attr_dropout = float(cfg.get("static_noise", 0.0)), float(cfg.get("attr_dropout", 0.0))
            if static_mode == "film":
                sh = int(cfg.get("static_hidden", 128))
                self.enc = nn.Sequential(nn.Linear(n_static, sh), nn.GELU(), nn.LayerNorm(sh), nn.Linear(sh, emb),
                                         nn.GELU(), nn.LayerNorm(emb))
                self.proj = nn.Linear(n_dyn, hidden)
                self.film = nn.Linear(emb, 2 * hidden)
                nn.init.zeros_(self.film.weight)
                nn.init.zeros_(self.film.bias)
                self.lstm = nn.LSTM(hidden, hidden, **lstm_kw)
                feat = hidden + emb
            else:
                self.lstm = nn.LSTM(n_dyn + n_static, hidden, **lstm_kw)
                feat = hidden
            self.drop = nn.Dropout(p_drop if layers == 1 else 0.0)
            self.n_diel = 0 if heads == "indep" else 8
            self.n_direct = 0 if heads == "diel" else 5
            self.out = nn.Linear(feat, 2 + self.n_diel + self.n_direct + N_DAILY + 2 + 2 * N_AUX)
            with torch.no_grad():
                self.out.bias.zero_()
                if self.n_diel:
                    th = 2.0 * math.pi * 15.0 / 24.0               # initial peak at 15 h local solar
                    b = self.out.bias[2:2 + 8].view(2, 4)
                    b[:, 0], b[:, 1], b[:, 2], b[:, 3] = -2.0, math.cos(th), math.sin(th), -1.0
            self.src_raw = nn.Parameter(torch.full((6, 2), -2.0))   # σ_src per provider class and variable

        def encode_static(self, s):
            if self.training:
                if self.static_noise > 0:
                    s = s + self.static_noise * torch.randn_like(s)
                if self.attr_dropout > 0:
                    s = s * (torch.rand_like(s) >= self.attr_dropout).to(s.dtype)
            return s

        def forward(self, x, s, L: int):
            Ld = L + PAD
            s = self.encode_static(s)
            if self.static_mode == "film":
                z = self.enc(s)
                gb = self.film(z)
                xin = self.proj(x) * (1.0 + gb[:, None, :hidden]) + gb[:, None, hidden:]
                h, _ = self.lstm(xin)
                f = torch.cat([self.drop(h[:, -Ld:]), z[:, None, :].expand(-1, Ld, -1)], dim=-1)
            else:
                h, _ = self.lstm(torch.cat([x, s[:, None, :].expand(-1, x.shape[1], -1)], dim=-1))
                f = self.drop(h[:, -Ld:])
            return self.decode(self.out(f))

        def decode(self, o):
            sp = torch.nn.functional.softplus
            k = 2
            res = {"mu": o[..., :2], "curve": None, "direct": None}
            if self.n_diel:
                d = o[..., k:k + 8].reshape(o.shape[:-1] + (2, 4))
                k += 8
                nrm = torch.sqrt(d[..., 1] ** 2 + d[..., 2] ** 2 + 1e-6)
                cv = diel_curve(sp(d[..., 0]), d[..., 1] / nrm, d[..., 2] / nrm, 0.5 * torch.sigmoid(d[..., 3]))
                res["curve"] = cv.permute(0, 1, 3, 2)               # (B, Ld, 24, 2)
            if self.n_direct:
                res["direct"] = o[..., k:k + 5]
                k += 5
            res["sig_d"] = sp(o[..., k:k + N_DAILY]) + 1e-3
            k += N_DAILY
            res["sig_i"] = sp(o[..., k:k + 2]) + 1e-3
            k += 2
            res["aux_mu"] = o[..., k:k + N_AUX]
            res["aux_sig"] = sp(o[..., k + N_AUX:k + 2 * N_AUX]) + 1e-3
            return res

        def sigma_src(self):
            return torch.nn.functional.softplus(self.src_raw)

    return Net()


def gaussian_nll(y, mu, sigma):
    torch = _torch()
    return torch.log(sigma) + 0.5 * ((y - mu) / sigma) ** 2 + 0.5 * math.log(2.0 * math.pi)


def loss_terms(net, out: dict, b: dict, L: int) -> dict:
    """Mean NLL of every observation group present in the batch (keys: the 7 daily statistics, ``hour_T``,
    ``hour_DO``, ``grab_T``, ``grab_DO``, the 4 auxiliary targets). ``b``: yd/md (B, L, 11), yh/mh (B, L, 24, 2),
    hfrac (B,), yi/mi/hi (B, L, 2), pc (B,). Masks are 0/1 floats; masked labels are 0."""
    torch = _torch()
    from .data import DAILY

    terms = {}
    pred = daily_stats(out, L)
    nll = gaussian_nll(b["yd"][..., :N_DAILY], pred, out["sig_d"][:, -L:])
    m = b["md"][..., :N_DAILY]
    n = m.sum(dim=(0, 1))
    tot = (nll * m).sum(dim=(0, 1))
    for j in range(N_DAILY):
        if n[j] >= 1:
            terms[DAILY[j]] = tot[j] / n[j]
    nla = gaussian_nll(b["yd"][..., N_DAILY:], out["aux_mu"][:, -L:], out["aux_sig"][:, -L:])
    ma = b["md"][..., N_DAILY:]
    na = ma.sum(dim=(0, 1))
    ta = (nla * ma).sum(dim=(0, 1))
    for j in range(N_AUX):
        if na[j] >= 1:
            terms[DAILY[N_DAILY + j]] = ta[j] / na[j]
    mu, sig_i = out["mu"][:, -L:], out["sig_i"][:, -L:]
    cv = None if out.get("curve") is None else out["curve"][:, -L:]
    if b["mh"].sum() >= 1:                                      # sonde hourly rows: μ + g(h), h = bin + fraction
        if cv is None:
            ph = mu[:, :, None, :].expand(-1, -1, 24, -1)
        else:
            hh = torch.arange(24, device=mu.device, dtype=mu.dtype)[None, None, :] + b["hfrac"][:, None, None]
            g = interp_curve(cv.permute(0, 1, 3, 2), hh[:, :, None, :].expand(-1, L, 2, -1))   # (B, L, 2, 24)
            ph = mu[:, :, None, :] + g.permute(0, 1, 3, 2)
        nh = gaussian_nll(b["yh"], ph, sig_i[:, :, None, :])
        for j, v in enumerate(("hour_T", "hour_DO")):
            cnt = b["mh"][..., j].sum()
            if cnt >= 1:
                terms[v] = (nh[..., j] * b["mh"][..., j]).sum() / cnt
    if b["mi"].sum() >= 1:                                      # grab rows: variance σ_inst² + σ_src²[provider class]
        src = net.sigma_src()[b["pc"].clamp(min=0)]             # (B, 2)
        sg = torch.sqrt(sig_i ** 2 + src[:, None, :] ** 2)
        timed = torch.isfinite(b["hi"])
        if cv is None:
            ni = gaussian_nll(b["yi"], mu, sg)
        else:
            cvp = cv.permute(0, 1, 3, 2)                        # (B, L, 2, 24)
            g = interp_curve(cvp, torch.where(timed, b["hi"], torch.zeros_like(b["hi"])))
            n_t = gaussian_nll(b["yi"], mu + g, sg)
            gm = cvp[..., torch.as_tensor(MIX_HOURS, device=mu.device)]            # (B, L, 2, 10)
            n_m = gaussian_nll(b["yi"][..., None], mu[..., None] + gm, sg[..., None])
            n_u = -(torch.logsumexp(-n_m, dim=-1) - math.log(len(MIX_HOURS)))      # uniform 08–17 h mixture
            ni = torch.where(timed, n_t, n_u)
        for j, v in enumerate(("grab_T", "grab_DO")):
            cnt = b["mi"][..., j].sum()
            if cnt >= 1:
                terms[v] = (ni[..., j] * b["mi"][..., j]).sum() / cnt
    return terms


DENSE_W = {"T_mean": 1.0, "T_7dadm": 1.0, "DO_mean": 1.0, "DO_min": 1.0, "T_max": 0.5, "T_min": 0.5, "DO_max": 0.5,
           "hour_T": 0.5, "hour_DO": 0.5}
AUX_W = {"logq": 0.5, "sc": 0.25, "turb": 0.25, "no3": 0.25}
SRC_PRIOR_SCALE, SRC_PRIOR_W = 0.5, 0.01


def total_loss(net, terms: dict, grab_share: float, *, variables=("T", "DO"), aux_w: dict | None = None):
    """(1 − s)·dense + s·grab (renormalised over the parts present) + Σ λ_a·aux + half-normal prior on σ_src.
    Dense = weighted mean of the daily and hourly terms (:data:`DENSE_W`); grab = mean of the grab terms."""
    aux_w = AUX_W if aux_w is None else aux_w
    keep = lambda k: k.split("_")[0] in variables or (k.startswith(("hour_", "grab_")) and k.split("_")[1] in variables)  # noqa: E731
    dk = [k for k in terms if k in DENSE_W and keep(k)]
    gk = [k for k in terms if k.startswith("grab_") and keep(k)]
    parts, wts = [], []
    if dk:
        parts.append(sum(DENSE_W[k] * terms[k] for k in dk) / sum(DENSE_W[k] for k in dk))
        wts.append(1.0 - grab_share if gk else 1.0)
    if gk:
        parts.append(sum(terms[k] for k in gk) / len(gk))
        wts.append(grab_share if dk else 1.0)
    loss = None
    if parts and sum(wts) > 0:
        loss = sum(w * p for w, p in zip(wts, parts)) / sum(wts)
    for k, lam in aux_w.items():
        if k in terms and lam > 0:
            loss = lam * terms[k] if loss is None else loss + lam * terms[k]
    if loss is not None and gk:
        loss = loss + SRC_PRIOR_W * 0.5 * ((net.sigma_src() / SRC_PRIOR_SCALE) ** 2).sum()
    return loss


def n_params(net) -> int:
    return int(sum(p.numel() for p in net.parameters()))


__all__ = ["make_net", "loss_terms", "total_loss", "daily_stats", "inst_point", "diel_curve", "interp_curve",
           "gaussian_nll", "n_params", "N_DAILY", "N_AUX", "PAD", "HEADS"]
