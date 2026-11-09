# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""SENTINEL-Sim network + physics coupling (protocol §3).

The LSTM sees daily forcings, observed discharge, calendar and static traits; per day it emits
seven raw channels. Their mapping to physical quantities is fixed:

    T_mean  = μ_T + σ_T · r0                      (°C)        water temperature, daily mean
    T_amp   = softplus(r1)                        (°C)        half diel range; T_h = T_mean + T_amp·cos(…)
    GPP     = 3 · softplus(r2)                    (g O2 m⁻² d⁻¹, ≥ 0)
    ER      = −3 · softplus(r3)                   (g O2 m⁻² d⁻¹, ≤ 0)
    K600    = max(K600_prior(Q, slope) · exp(2.5·tanh(r4)), 0.3)   (d⁻¹) — Raymond 2012 prior, ±e^2.5 deviation
    SC      = μ_SC + σ_SC · r5                    (log(µS/cm + 1))
    turb    = μ_tu + σ_tu · r6                    (log(FNU + 0.1))

Optional v2 metabolism structure (``metab_struct=True``; protocol §3.3 lever, §12 v1.5), fixed functional
forms, no learnable parameters:

    GPP     = 3 · softplus(r2) · tanh(SW_d / I_k),   I_k = 150 W m⁻²   (Jassby–Platt light saturation)
    ER      = −3 · softplus(r3) · θ^(T_mean − 20),   θ = 1.07           (Arrhenius-type, Q10 ≈ 2)

``drop_dyn`` (list of DYN_NAMES) zeroes those normalised inputs (e.g. the DOY harmonics).

DO is never emitted by the network: it is integrated hourly by :func:`physics.integrate_do` from
GPP, ER, K600, the simulated diel temperature, light weights, hydraulic-geometry depth and site
pressure, and its daily mean/min/max are compared with the USGS daily statistics.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from . import data as D
from . import physics

N_RAW = 7
IK_WM2, THETA_ER = 150.0, 1.07
K600_FLOOR = 0.3      # d⁻¹; lowest K600 in the Powell Center estimates is of this order (review W1)
OUT_DAILY = ["temp_c_mean", "temp_c_min", "temp_c_max", "do_mgl_mean", "do_mgl_min", "do_mgl_max",
             "spcond_uscm_mean", "turb_fnu_mean"]          # == data.TARGETS (model space for SC/turb)
LATENTS = ["gpp", "er", "k600", "k600_prior", "depth_m", "do_sat_mean", "t_amp", "hyp2_h", "hyp5_h"]


class SimNet(nn.Module):
    def __init__(self, n_dyn: int, n_static: int, hidden: int = 128, static_embed: int = 32,
                 dropout: float = 0.2, n_layers: int = 1):
        super().__init__()
        self.static = nn.Sequential(nn.Linear(n_static, static_embed), nn.Tanh())
        self.lstm = nn.LSTM(n_dyn + static_embed, hidden, num_layers=n_layers, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, N_RAW)
        with torch.no_grad():   # start near physically sensible values
            self.head.bias.zero_()
            self.head.bias[1] = 0.5      # softplus(0.5) ≈ 0.97 °C half-range
            self.head.bias[2] = -0.5     # GPP ≈ 1.4
            self.head.bias[3] = -0.5     # ER ≈ −1.4
        # LSTM forget-gate bias = 3 (long memory, standard hydrology LSTM init)
        for name, p in self.lstm.named_parameters():
            if "bias_hh" in name:
                with torch.no_grad():
                    p.zero_()
                    p[hidden: 2 * hidden] = 3.0

    def forward(self, dyn: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        e = self.static(static)[:, None, :].expand(-1, dyn.shape[1], -1)
        h, _ = self.lstm(torch.cat([dyn, e], dim=-1))
        return self.head(self.drop(h))


class SimModel(nn.Module):
    """Network + fixed physics. ``norm`` = SimData.norm (training statistics only)."""

    def __init__(self, norm: dict, n_dyn: int, n_static: int, *, metab_struct: bool = False, drop_dyn=(), **kw):
        super().__init__()
        self.net = SimNet(n_dyn, n_static, **kw)
        self.metab_struct = bool(metab_struct)
        keep = torch.ones(n_dyn)
        for nme in drop_dyn or ():
            keep[D.DYN_NAMES.index(nme)] = 0.0
        self.register_buffer("dyn_keep", keep)
        isw = D.DYN_NAMES.index("srad_wm2")
        self.register_buffer("sw_mean", torch.tensor(float(norm["dyn_mean"][isw]) if "dyn_mean" in norm else 0.0))
        self.register_buffer("sw_std", torch.tensor(float(norm["dyn_std"][isw]) if "dyn_std" in norm else 1.0))
        self._isw = isw
        ym = torch.as_tensor(norm["y_mean"], dtype=torch.float32)
        ys = torch.as_tensor(norm["y_std"], dtype=torch.float32)
        self.register_buffer("y_mean", torch.nan_to_num(ym, nan=0.0))
        self.register_buffer("y_std", torch.nan_to_num(ys, nan=1.0))

    def forward(self, dyn, static, *, q_m3s, slope, light_w, pressure, return_hourly: bool = False):
        sw_raw = (dyn[..., self._isw] * self.sw_std + self.sw_mean).clamp(min=0.0)
        r = self.net(dyn * self.dyn_keep, static)
        jT, jSC, jTu = 0, 6, 7
        t_mean = self.y_mean[jT] + self.y_std[jT] * r[..., 0]
        t_amp = nn.functional.softplus(r[..., 1])
        gpp = 3.0 * nn.functional.softplus(r[..., 2])
        er = -3.0 * nn.functional.softplus(r[..., 3])
        if self.metab_struct:
            gpp = gpp * torch.tanh(sw_raw / IK_WM2)
            er = er * THETA_ER ** (t_mean.clamp(min=0.0) - 20.0)
        kp = physics.k600_prior(q_m3s, slope[:, None])
        k600 = (kp * torch.exp(2.5 * torch.tanh(r[..., 4]))).clamp(min=K600_FLOOR)
        z = physics.depth_m(q_m3s)
        t_h = physics.diel_temperature(t_mean.clamp(min=0.0), t_amp).clamp(min=0.0)
        do_raw, csat = physics.integrate_do(gpp, er, k600, t_h, light_w, z, pressure)
        do_h = do_raw.clamp(min=0.0)      # reported DO is non-negative; the LOSS uses the unclamped state
        out = {
            "_do_raw_mean": do_raw.mean(-1), "_do_raw_min": do_raw.amin(-1), "_do_raw_max": do_raw.amax(-1),
            "temp_c_mean": t_mean, "temp_c_min": t_mean - t_amp, "temp_c_max": t_mean + t_amp,
            "do_mgl_mean": do_h.mean(-1), "do_mgl_min": do_h.amin(-1), "do_mgl_max": do_h.amax(-1),
            "spcond_uscm_mean": self.y_mean[jSC] + self.y_std[jSC] * r[..., 5],
            "turb_fnu_mean": self.y_mean[jTu] + self.y_std[jTu] * r[..., 6],
            "gpp": gpp, "er": er, "k600": k600, "k600_prior": kp, "depth_m": z,
            "do_sat_mean": csat.mean(-1), "t_amp": t_amp, "k_dev_raw": r[..., 4],
            "hyp2_h": (do_h < 2.0).sum(-1).to(do_h.dtype), "hyp5_h": (do_h < 5.0).sum(-1).to(do_h.dtype),
        }
        if return_hourly:
            out["do_hourly"] = do_h
            out["t_hourly"] = t_h
        return out


_LOSS_KEY = {"do_mgl_mean": "_do_raw_mean", "do_mgl_min": "_do_raw_min", "do_mgl_max": "_do_raw_max"}


def stack_targets(out: dict, *, for_loss: bool = False) -> torch.Tensor:
    """(B, T, V) predictions; with ``for_loss`` DO statistics come from the unclamped state so the
    non-negativity clamp never blocks gradients (review W1)."""
    return torch.stack([out[_LOSS_KEY.get(v, v)] if for_loss and _LOSS_KEY.get(v) in out else out[v]
                        for v in OUT_DAILY], dim=-1)


def loss_fn(out: dict, y_model: torch.Tensor, mask: torch.Tensor, y_mean, y_std, *, k_prior_w: float = 0.01,
            target_w=None) -> tuple[torch.Tensor, dict]:
    """Masked MSE in standardised model space, averaged per target over its valid entries, plus a
    weak penalty pulling K600 toward the Raymond prior. ``y_model``/``mask`` (B, T, V)."""
    pred = stack_targets(out, for_loss=True)
    z_p = (pred - y_mean) / y_std
    z_o = (torch.nan_to_num(y_model) - y_mean) / y_std
    se = (z_p - z_o) ** 2 * mask
    n = mask.sum(dim=(0, 1))
    per = se.sum(dim=(0, 1)) / n.clamp(min=1)
    w = torch.ones_like(per) if target_w is None else torch.as_tensor(target_w, dtype=per.dtype, device=per.device)
    has = (n > 0).to(per.dtype)
    data_loss = (per * w * has).sum() / (w * has).sum().clamp(min=1e-6)
    kpen = (out["k_dev_raw"] ** 2).mean()
    total = data_loss + k_prior_w * kpen
    return total, {"data": float(data_loss.detach()), "kpen": float(kpen.detach()),
                   **{f"mse_{v}": float(per[j].detach()) for j, v in enumerate(OUT_DAILY) if n[j] > 0}}


# ----------------------------------------------------------------------------- batching helpers

def batch_tensors(d: D.SimData, idx, t0: int, t1: int, static_z: np.ndarray, device, *, forcing=None,
                  q_m3s=None):
    """Model inputs for sites ``idx`` over days [t0, t1). Optional perturbed raw forcing / discharge."""
    idx = np.asarray(idx)
    dyn = D.normalised_dynamic(d, idx, t0, t1, forcing=forcing, q_m3s=q_m3s)
    q = d.q_m3s[idx, t0:t1] if q_m3s is None else q_m3s
    doy = d.dates[t0:t1].dayofyear.to_numpy()
    lw = physics.hourly_light_weights(d.lat[idx], doy)
    tt = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=device)  # noqa: E731
    return dict(dyn=tt(dyn), static=tt(static_z[idx]), q_m3s=tt(q), slope=tt(d.slope[idx]), light_w=tt(lw),
                pressure=tt(d.pressure[idx]))


def target_tensors(d: D.SimData, idx, t0: int, t1: int, valid_days: np.ndarray, device):
    """(B, T, V) model-space targets and mask; ``valid_days`` (T_total,) bool restricts the loss dates."""
    idx = np.asarray(idx)
    y = d.y[idx, t0:t1].astype("float64")
    ym = np.empty_like(y)
    for j, v in enumerate(D.TARGETS):
        ym[..., j] = D.to_model(v, y[..., j])
    m = np.isfinite(ym) & valid_days[t0:t1][None, :, None]
    return (torch.as_tensor(np.nan_to_num(ym), dtype=torch.float32, device=device),
            torch.as_tensor(m, dtype=torch.float32, device=device))


@torch.no_grad()
def simulate(model: SimModel, d: D.SimData, idx, t_start: int, t_end: int, *, warmup: int = 180,
             chunk: int = 365, device="cpu", batch: int = 64, forcing=None, q_m3s=None,
             keys=tuple(OUT_DAILY) + tuple(LATENTS)) -> dict:
    """Simulate sites ``idx`` for days [t_start, t_end) in warm-started chunks; returns (S, T) arrays.

    ``forcing`` (S, T_all, 8) / ``q_m3s`` (S, T_all): optional perturbed full-length raw inputs
    aligned with ``idx`` (scenarios). Each chunk re-starts the LSTM and the DO state ``warmup`` days
    earlier (clipped at day 0).
    """
    model.eval()
    idx = np.asarray(idx)
    static_z = D.normalised_static(d)
    T = t_end - t_start
    res = {k: np.full((len(idx), T), np.nan, dtype="float32") for k in keys}
    for b0 in range(0, len(idx), batch):
        sl = slice(b0, min(b0 + batch, len(idx)))
        ii = idx[sl]
        for c0 in range(t_start, t_end, chunk):
            c1 = min(c0 + chunk, t_end)
            w0 = max(0, c0 - warmup)
            f = None if forcing is None else forcing[sl, w0:c1]
            q = None if q_m3s is None else q_m3s[sl, w0:c1]
            bt = batch_tensors(d, ii, w0, c1, static_z, device, forcing=f, q_m3s=q)
            out = model(**bt)
            for k in keys:
                res[k][sl, c0 - t_start: c1 - t_start] = out[k][:, c0 - w0:].float().cpu().numpy()
    return res
