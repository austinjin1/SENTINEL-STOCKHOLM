# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""D2 ThermOx-T, temperature stage (amendment §6.2): a daily heat budget of a stream reach with learned, bounded
static parameters, written in torch. NOT RELEASED by deviations entry 1: only the train-only smoke of the kill rule
(amendment §8) may run before the D2 entry (lead rulings L4, L6), which names the sha256 of this file.

The network returns the output dict of :mod:`sentinel.hydromet_sota.gso_mt.nets` (``mu``, ``curve``, ``direct``,
``sig_d``, ``sig_i``, ``aux_mu``, ``aux_sig``), so the trainer, the loss, the criterion and the stack are shared.

Structure
---------
1. **Static MLP** [n_static → 128 → 64 → 32] (input noise σ = 0.5 in training) → 13 bounded static parameters
   (:data:`PARAMS`). The last layer starts near zero with its bias at the prior median of each parameter.
2. **GRU** (hidden 64) on the dynamic inputs and the static embedding → daily latents: log discharge (standardised;
   supervised through the auxiliary discharge label with weight λ_Q), snowmelt share m ∈ [0, 1), groundwater modifier
   ∈ [0.5, 2], residual r ∈ [−c, c] °C. The same features give the statistical heads (DO mean and diel curve, all σ,
   the other auxiliary targets): the DO heads are plain statistical heads until the DO stage exists.
3. **Physics (fp32), daily, over the whole input window:**
   * equilibrium temperature Te and exchange coefficient K from the surface energy balance
     (:func:`..thermox.physics.equilibrium_temperature`, 4 Newton steps, Te clamped to [−5, 45] °C), with seasonal
     shade and wind sheltering; the same for the upstream-basin forcings (``ws_`` channels);
   * source mixing: T* = Σ w_i T_i over atmosphere (Te), groundwater (exponential kernel of air temperature +
     offset), reservoir (exponential kernel of Te) and upstream (upstream Te lagged by a travel time), then melt
     water at 0 °C: T* ← (1 − m) T*; T* is not allowed below 0 °C;
   * exact exponential integrator: with k = s_K K · 86 400 / (ρ c_p D) per day and depth D = D_ref (Q/Q̄)^b,
     Tw(t+1) = T* + (Tw(t) − T*) e^(−k) and the daily mean is T* + (Tw(t) − T*)(1 − e^(−k))/k.
4. **Diel sub-model** on the decoded days: hourly anomaly of the equilibrium temperature from the short-wave
   anomaly (solar geometry from the photoperiod) and the air-temperature anomaly (daily range), linearised around
   the daily state; integrated with the exact exponential step over 2 periodic cycles × 24 steps; the second cycle,
   with its mean removed, is the 24-point curve. Daily max, min, 7-DADMax and the grab operator come from the curve
   through the shared :func:`..gso_mt.nets.daily_stats` / ``inst_point``.

Every physical formula is called from the verified shared routines (rule X7): the energy balance from
:mod:`.physics`, the pressure from the frozen routine re-exported there. The sensitivity ∂H/∂Ta of the diel
sub-model is a central difference of :func:`.physics.heat_flux`, not a second copy of the formula.

**Implementation constants (no literature source; fixed before any fit; recorded in the D2 deviations entry):** the
bounds and prior medians of :data:`PARAMS` (shade and wind sheltering equal :data:`.physics.PRIOR`),
:data:`LEAF_OFF_FRACTION`, :data:`LEAF_WIDTH_D`, :data:`AIR_PEAK_HOUR`, :data:`TRAVEL_SPEED_KM_D`,
:data:`MAX_LAG_D`, the latent ranges. No Powell-derived prior is used (rulings §2.3)."""

from __future__ import annotations

import math

from ..gso_mt import constants as K
from ..gso_mt.nets import N_AUX, N_DAILY, PAD, diel_curve
from . import physics as TP

# name, lower bound, upper bound, prior median, log scale
PARAMS = (
    ("depth_ref_m", 0.05, 5.0, 0.5, True),        # depth at the window's geometric-mean discharge
    ("depth_exp", 0.0, 0.6, 0.3, False),          # D = D_ref (Q / Q̄)^b
    ("shade_max", 0.0, 0.95, TP.PRIOR["shade"], False),
    ("leaf_on_doy", 60.0, 180.0, 120.0, False),
    ("leaf_off_doy", 240.0, 330.0, 290.0, False),
    ("wind_shelter", 0.05, 1.0, TP.PRIOR["wind_shelter"], False),
    ("gw_odds", 0.0, 2.0, 0.10, False),           # source weights are odds against the atmospheric term (weight 1)
    ("gw_tau_d", 10.0, 365.0, 90.0, True),
    ("gw_offset_c", -3.0, 5.0, 1.0, False),
    ("res_odds", 0.0, 2.0, 0.02, False),
    ("res_tau_d", 5.0, 120.0, 30.0, True),
    ("up_odds", 0.0, 2.0, 0.20, False),
    ("exchange_scale", 0.3, 3.0, 1.0, True),
)
PARAM_NAMES = tuple(p[0] for p in PARAMS)
LEAF_OFF_FRACTION = 0.5       # shade without leaves, as a fraction of the maximum shade (banks, stems, topography)
LEAF_WIDTH_D = 10.0           # width (days) of the logistic leaf-on and leaf-off transitions
AIR_PEAK_HOUR = 15.0          # local solar hour of the daily air-temperature maximum
TRAVEL_SPEED_KM_D = 43.2      # 0.5 m s-1; travel time of the upstream term = sqrt(watershed area) / speed
MAX_LAG_D = 5.0
DEFAULT_LAG_D = 1.0           # when the watershed-area input is absent
GW_MOD_LOG_RANGE = math.log(2.0)
Q_EXP_CLAMP = 3.0             # |ln(Q / Q̄)| used by the depth relation is clamped
DTA = 0.5                     # K, half-width of the central difference for dH/dTa
MONTH_START_DOY = (1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335)   # 365-day calendar
SITE_VARS = ("tmean_c", "tmax_c", "tmin_c", "srad_wm2", "wind_ms", "sph_kgkg")
UP_VARS = ("tmean_c", "srad_wm2", "wind_ms", "sph_kgkg")


def _torch():
    import torch

    return torch


def physics_context(scales, dyn_names, static_names) -> dict:
    """Plain numbers the network needs to turn standardised inputs back into physical units (train-only constants of
    :class:`..gso_mt.data.Scales`) and labels into model space. JSON-serialisable."""
    from ..gso_mt.data import DAILY, SOLAR_CHANNELS, VAA_COLS, VAA_LOG

    dn, sn, norm = list(dyn_names), list(static_names), scales.norm
    ctx = {"dyn": {}, "static": {}, "cal": [dn.index("doy_sin1"), dn.index("doy_cos1")]}

    def add_forc(key, name):
        j = dn.index(name)
        if j >= len(norm.forc_mean):
            raise ValueError(f"{name} is not a forcing channel")
        ctx["dyn"][key] = [j, float(norm.forc_mean[j]), float(norm.forc_std[j])]

    for v in SITE_VARS:
        add_forc(v, f"f_{v}_d")
    ctx["has_upstream"] = all(f"f_ws_{v}_d" in dn for v in UP_VARS)
    if ctx["has_upstream"]:
        for v in UP_VARS:
            add_forc("ws_" + v, f"f_ws_{v}_d")
    for c in ("cloud_ratio", "photoperiod_h"):
        j, k = dn.index(f"solar_{c}"), SOLAR_CHANNELS.index(c)
        ctx["dyn"][c] = [j, float(scales.solar_mean[k]), float(scales.solar_std[k])]

    def add_static(key, name):
        if name in norm.static_cols:
            k = list(norm.static_cols).index(name)
            ctx["static"][key] = [sn.index(name), float(norm.static_mean[k]), float(norm.static_std[k]),
                                  bool(name in norm.static_log)]
        elif name in VAA_COLS and name in sn:
            k = VAA_COLS.index(name)
            ctx["static"][key] = [sn.index(name), float(scales.vaa_mean[k]), float(scales.vaa_std[k]),
                                  bool(name in VAA_LOG)]

    add_static("elev_m", "elev_m")
    add_static("ws_area_km2", "wsareasqkm")
    if "elev_m" not in ctx["static"]:
        raise ValueError("ThermOx-T needs the static input elev_m (barometric pressure)")
    jt, jq = DAILY.index("T_mean"), DAILY.index("logq")
    ctx["label"] = {"t_mean": float(scales.label_mean[jt]), "t_std": float(scales.label_std[jt]),
                    "logq_std": float(scales.label_std[jq])}
    return ctx


def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1.0 - 1e-4)
    return math.log(p / (1.0 - p))


def prior_bias() -> list:
    """Bias of the parameter layer that puts every bounded parameter at its prior median."""
    out = []
    for _, lo, hi, med, log in PARAMS:
        f = (math.log(med) - math.log(lo)) / (math.log(hi) - math.log(lo)) if log else (med - lo) / (hi - lo)
        out.append(_logit(f))
    return out


def bounded(raw):
    """(…, 13) raw values → dict name → bounded parameter (sigmoid between the bounds; log scale where flagged)."""
    torch = _torch()
    u = torch.sigmoid(raw)
    out = {}
    for j, (name, lo, hi, _, log) in enumerate(PARAMS):
        out[name] = torch.exp(math.log(lo) + u[..., j] * (math.log(hi) - math.log(lo))) if log else lo + u[..., j] * (hi - lo)
    return out


def ema(x, tau_d):
    """Exponential kernel along time: y_t = y_{t−1} + (1 − e^(−1/τ)) (x_t − y_{t−1}), started at the window mean.
    ``x`` (B, T), ``tau_d`` (B,). Closed form of the recurrence (one matrix product instead of T steps):
    y_t = d^(t+1) x̄ + (1 − d) Σ_{s ≤ t} d^(t−s) x_s with d = e^(−1/τ)."""
    torch = _torch()
    T = x.shape[1]
    r = (-1.0 / tau_d)[:, None, None]
    n = torch.arange(T, device=x.device, dtype=x.dtype)
    lag = n[:, None] - n[None, :]                                        # t − s
    w = torch.exp(r * torch.clamp(lag, min=0.0)[None]) * (lag >= 0)[None]
    a = 1.0 - torch.exp(r[:, :, 0])
    return torch.exp(r[:, :, 0] * (n[None] + 1.0)) * x.mean(dim=1, keepdim=True) + a * torch.bmm(w, x[..., None])[..., 0]


def ema_loop(x, tau_d):
    """Step-by-step form of :func:`ema` (reference for the unit test)."""
    torch = _torch()
    a = 1.0 - torch.exp(-1.0 / tau_d)
    y = x.mean(dim=1)
    ys = []
    for t in range(x.shape[1]):
        y = y + a * (x[:, t] - y)
        ys.append(y)
    return torch.stack(ys, dim=1)


def integrate_daily(t_star, k_day):
    """Exact exponential integrator of dTw/dt = k (T* − Tw) with T* and k constant within a day; started at T*_0.
    Returns the daily mean (B, T). Closed form of the recurrence Tw(t+1) = T*_t + (Tw(t) − T*_t) e^(−k_t): with
    C_t = Σ_{u ≤ t} k_u, Tw(t) = e^(−C_{t−1}) Tw(0) + Σ_{s < t} e^(−(C_{t−1} − C_s)) (1 − e^(−k_s)) T*_s; the
    cumulative sums are taken in float64 (their differences are the exponents)."""
    torch = _torch()
    T = t_star.shape[1]
    c = torch.cumsum(k_day.double(), dim=1)
    cprev = torch.cat([torch.zeros_like(c[:, :1]), c[:, :-1]], dim=1)
    n = torch.arange(T, device=t_star.device)
    before = (n[None, :] < n[:, None])[None]                             # s < t
    m = (torch.exp(-torch.clamp(cprev[:, :, None] - c[:, None, :], min=0.0)) * before).to(t_star.dtype)
    e = torch.exp(-k_day)
    tw = torch.exp(-cprev).to(t_star.dtype) * t_star[:, :1] + torch.bmm(m, ((1.0 - e) * t_star)[..., None])[..., 0]
    return t_star + (tw - t_star) * (1.0 - e) / k_day


def integrate_daily_loop(t_star, k_day):
    """Step-by-step form of :func:`integrate_daily` (reference for the unit test)."""
    torch = _torch()
    e = torch.exp(-k_day)
    phi = (1.0 - e) / k_day
    tw = t_star[:, 0]
    means = []
    for t in range(t_star.shape[1]):
        d = tw - t_star[:, t]
        means.append(t_star[:, t] + d * phi[:, t])
        tw = t_star[:, t] + d * e[:, t]
    return torch.stack(means, dim=1)


def lagged(x, lag_d):
    """``x`` (B, T) delayed by a fractional number of days ``lag_d`` (B,) (linear interpolation; the first value is
    repeated before the window)."""
    torch = _torch()
    n = torch.floor(lag_d)
    f = (lag_d - n)[:, None]
    t = torch.arange(x.shape[1], device=x.device)[None, :]
    i0 = torch.clamp(t - n.long()[:, None], min=0)
    i1 = torch.clamp(i0 - 1, min=0)
    return torch.gather(x, 1, i0) * (1.0 - f) + torch.gather(x, 1, i1) * f


def sw_shape(photoperiod_h, steps: int):
    """(…, steps) hourly short-wave factor with mean 1 over the day: ∝ max(0, cos ω − cos ω_s) at the middle of each
    step, ω the hour angle and ω_s = π N / 24 the sunset hour angle (N = photoperiod; FAO-56 eq. 25, 34). A day
    without sun gives the factor 1 (no anomaly)."""
    torch = _torch()
    h = (torch.arange(steps, device=photoperiod_h.device, dtype=photoperiod_h.dtype) + 0.5) * (24.0 / steps)
    w = (h - 12.0) * (math.pi / 12.0)
    ws = torch.clamp(photoperiod_h, 0.0, 24.0)[..., None] * (math.pi / 24.0)
    c = torch.clamp(torch.cos(w) - torch.cos(ws), min=0.0)
    m = c.mean(dim=-1, keepdim=True)
    return torch.where(m > 1e-6, c / torch.clamp(m, min=1e-6), torch.ones_like(c))


def diel_anomaly(f_eq, kappa, cycles: int):
    """Periodic response θ of dθ/dt = k (F − θ) to the equilibrium anomaly ``f_eq`` (…, steps), constant within a
    step, with ``kappa`` (…) = k · step length. ``cycles`` passes from θ = 0; the last pass, recorded at the start of
    each step and with its mean removed, is returned (…, steps)."""
    torch = _torch()
    e = torch.exp(-kappa)
    th = torch.zeros_like(kappa)
    rec = []
    steps = f_eq.shape[-1]
    for c in range(cycles):
        for j in range(steps):
            if c == cycles - 1:
                rec.append(th)
            th = f_eq[..., j] + (th - f_eq[..., j]) * e
    cv = torch.stack(rec, dim=-1)
    return cv - cv.mean(dim=-1, keepdim=True)


def make_net(n_dyn: int, n_static: int, cfg: dict, ctx: dict):
    """``cfg`` = the ``net`` block of ``configs/hydromet_sota/v2/d2.yaml`` (static_hidden, static_emb, static_noise,
    gru_hidden, n_newton, te_range, diel {enabled, cycles, steps, months}, mixing, warmup_epochs, residual_cap,
    powell_prior, prior); ``ctx`` = :func:`physics_context`."""
    torch = _torch()
    nn = torch.nn
    sp = nn.functional.softplus
    if cfg.get("powell_prior", False):
        raise ValueError("no Powell-derived prior may be used (rulings §2.3; the shared role table does not exist)")
    if tuple(float(v) for v in cfg.get("te_range", TP.TE_RANGE)) != tuple(TP.TE_RANGE):
        raise ValueError("te_range differs from thermox.physics.TE_RANGE")
    pr = cfg.get("prior") or TP.PRIOR
    if any(abs(float(pr[k]) - TP.PRIOR[k]) > 1e-12 for k in TP.PRIOR):
        raise ValueError("the prior medians of shade and wind sheltering must equal thermox.physics.PRIOR")
    mixing = cfg.get("mixing", "full")
    if mixing not in ("full", "atmosphere_only"):
        raise ValueError(f"unknown mixing {mixing!r}")
    dcfg = dict(cfg.get("diel") or {})
    diel_on, cycles, steps = bool(dcfg.get("enabled", True)), int(dcfg.get("cycles", 2)), int(dcfg.get("steps", 24))
    months = dcfg.get("months")                     # None = every decoded day; else the list of calendar months
    if steps != 24 or cycles < 1:
        raise ValueError("the diel sub-model works on 24 steps per day and at least one cycle")
    sh = [int(v) for v in cfg.get("static_hidden", [128, 64])]
    emb, hid = int(cfg.get("static_emb", 32)), int(cfg.get("gru_hidden", 64))
    cap, n_newton = float(cfg.get("residual_cap", 0.0)), int(cfg.get("n_newton", TP.N_NEWTON))
    warm = int(cfg.get("warmup_epochs", 0))
    full = mixing == "full"
    has_up = full and bool(ctx.get("has_upstream"))
    lab = ctx["label"]

    class ThermOxT(nn.Module):
        def __init__(self):
            super().__init__()
            self.static_noise = float(cfg.get("static_noise", 0.0))
            dims, layers = [n_static] + sh + [emb], []
            for a, b in zip(dims[:-1], dims[1:]):
                layers += [nn.Linear(a, b), nn.GELU(), nn.LayerNorm(b)]
            self.enc = nn.Sequential(*layers)
            self.par = nn.Linear(emb, len(PARAMS))
            self.gru = nn.GRU(n_dyn + emb, hid, batch_first=True)
            self.lat = nn.Linear(hid, 3)            # log discharge (standardised), melt share, groundwater modifier
            self.res = nn.Linear(hid, 1) if cap > 0 else None
            # statistical heads on the decoded days: DO mean, 2 × 4 diel heads (T used only without the physical
            # diel sub-model), σ of 7 daily statistics, σ of 2 instantaneous values, 3 auxiliary means, 4 auxiliary σ
            self.out = nn.Linear(hid + emb, 1 + 8 + N_DAILY + 2 + (N_AUX - 1) + N_AUX)
            with torch.no_grad():
                nn.init.normal_(self.par.weight, std=0.01)
                self.par.bias.copy_(torch.tensor(prior_bias()))
                self.lat.bias.copy_(torch.tensor([0.0, -4.0, 0.0]))
                nn.init.normal_(self.lat.weight[1:], std=0.01)
                if self.res is not None:
                    nn.init.zeros_(self.res.weight)
                    nn.init.zeros_(self.res.bias)
                self.out.bias.zero_()
                th = 2.0 * math.pi * AIR_PEAK_HOUR / 24.0
                b = self.out.bias[1:9].view(2, 4)
                b[:, 0], b[:, 1], b[:, 2], b[:, 3] = -2.0, math.cos(th), math.sin(th), -1.0
            self.src_raw = nn.Parameter(torch.full((6, 2), -2.0))
            self.register_buffer("res_on", torch.tensor(0.0 if (cap > 0 and warm > 0) else 1.0))
            for k, (j, m, s) in ctx["dyn"].items():
                self.register_buffer(f"d_{k}", torch.tensor([m, s], dtype=torch.float32))
            self.months = None if months is None else sorted(int(m) for m in months)

        # -------------------------------------------------------------- hooks used by the shared trainer
        def set_epoch(self, epoch: int) -> None:
            """Warm-up (amendment §6.2): the residual is frozen at 0 for the first ``warmup_epochs`` epochs."""
            if cap > 0:
                self.res_on.fill_(1.0 if epoch >= warm else 0.0)

        def sigma_src(self):
            return sp(self.src_raw)

        def encode_static(self, s):
            if self.training and self.static_noise > 0:
                s = s + self.static_noise * torch.randn_like(s)
            return s

        # -------------------------------------------------------------- physical inputs
        def _dyn(self, x, key):
            j = ctx["dyn"][key][0]
            ms = getattr(self, f"d_{key}")
            return x[..., j] * ms[1] + ms[0]

        @staticmethod
        def _static(s, key):
            j, m, sd, log = ctx["static"][key]
            v = s[:, j] * sd + m
            return 10.0 ** v - 1e-3 if log else v

        def _drivers(self, x, prefix, p_kpa, cloud, shade, shelter):
            return TP.surface_drivers(tmean=self._dyn(x, prefix + "tmean_c"),
                                      srad=torch.clamp(self._dyn(x, prefix + "srad_wm2"), min=0.0),
                                      wind10=torch.clamp(self._dyn(x, prefix + "wind_ms"), min=0.0),
                                      sph=torch.clamp(self._dyn(x, prefix + "sph_kgkg"), min=1e-5),
                                      p_kpa=p_kpa, cloud_ratio=cloud, shade=shade, wind_shelter=shelter)

        # -------------------------------------------------------------- forward
        def forward(self, x, s, L: int, *, return_physics: bool = False):
            Ld = L + PAD
            B, T, _ = x.shape
            # physical statics from the clean inputs (no noise); pressure from the shared frozen routine
            elev = self._static(s, "elev_m")
            p_kpa = torch.as_tensor(TP.pressure_kpa(elev.detach().cpu().numpy()), dtype=x.dtype, device=x.device)[:, None]
            if "ws_area_km2" in ctx["static"]:
                area = torch.clamp(self._static(s, "ws_area_km2"), min=0.0)
                lag = torch.clamp(torch.sqrt(area) / TRAVEL_SPEED_KM_D, 0.0, MAX_LAG_D)
            else:
                lag = torch.full((B,), DEFAULT_LAG_D, dtype=x.dtype, device=x.device)
            z = self.enc(self.encode_static(s))
            th = bounded(self.par(z))
            h, _ = self.gru(torch.cat([x, z[:, None, :].expand(-1, T, -1)], dim=-1))
            lat = self.lat(h)
            q = lat[..., 0]                                                  # standardised log discharge
            melt = torch.sigmoid(lat[..., 1])
            gw_mod = torch.exp(GW_MOD_LOG_RANGE * torch.tanh(lat[..., 2]))
            # ---- seasonal shade
            jc = ctx["cal"]
            doy = torch.remainder(torch.atan2(x[..., jc[0]], x[..., jc[1]]), 2.0 * math.pi) * (365.25 / (2.0 * math.pi)) + 1.0
            leaf = (torch.sigmoid((doy - th["leaf_on_doy"][:, None]) / LEAF_WIDTH_D)
                    * torch.sigmoid((th["leaf_off_doy"][:, None] - doy) / LEAF_WIDTH_D))
            shade = th["shade_max"][:, None] * (LEAF_OFF_FRACTION + (1.0 - LEAF_OFF_FRACTION) * leaf)
            shelter = th["wind_shelter"][:, None]
            cloud = torch.clamp(self._dyn(x, "cloud_ratio"), 0.0, 1.2)
            # ---- equilibrium temperature and exchange coefficient (site cell; upstream basin)
            kw = self._drivers(x, "", p_kpa, cloud, shade, shelter)
            te, kx = TP.equilibrium_temperature(**kw, n_newton=n_newton)
            ta = kw["ta"]
            # ---- source mixing
            if full:
                o_gw = th["gw_odds"][:, None] * gw_mod
                o_res = th["res_odds"][:, None].expand(-1, T)
                t_gw = ema(ta, th["gw_tau_d"]) + th["gw_offset_c"][:, None]
                t_res = ema(te, th["res_tau_d"])
                num, den = te + o_gw * t_gw + o_res * t_res, 1.0 + o_gw + o_res
                if has_up:
                    te_up, _ = TP.equilibrium_temperature(**self._drivers(x, "ws_", p_kpa, cloud, shade, shelter),
                                                          n_newton=n_newton)
                    o_up = th["up_odds"][:, None]
                    num, den = num + o_up * lagged(te_up, lag), den + o_up
                w_atm = (1.0 - melt) / den
                t_star = torch.clamp((1.0 - melt) * num / den, min=0.0)
            else:
                w_atm = torch.ones_like(te)
                t_star = torch.clamp(te, min=0.0)
            # ---- exact exponential integrator of the daily mean
            lnq = torch.clamp(lab["logq_std"] * (q - q.mean(dim=1, keepdim=True)), -Q_EXP_CLAMP, Q_EXP_CLAMP)
            depth = th["depth_ref_m"][:, None] * torch.exp(th["depth_exp"][:, None] * lnq)
            k_day = th["exchange_scale"][:, None] * kx * (K.SECONDS_PER_DAY / TP.RHO_CP) / depth
            tw = integrate_daily(t_star, k_day)
            f = torch.cat([h[:, -Ld:], z[:, None, :].expand(-1, Ld, -1)], dim=-1)
            o = self.out(f)
            r = cap * torch.tanh(self.res(h[:, -Ld:])[..., 0]) * self.res_on if self.res is not None else 0.0
            mu_t = (tw[:, -Ld:] + r - lab["t_mean"]) / lab["t_std"]
            # ---- diel curves: statistical heads (DO always; T where the physical sub-model is off)
            d = o[..., 1:9].reshape(o.shape[:-1] + (2, 4))
            nrm = torch.sqrt(d[..., 1] ** 2 + d[..., 2] ** 2 + 1e-6)
            cv = diel_curve(sp(d[..., 0]), d[..., 1] / nrm, d[..., 2] / nrm, 0.5 * torch.sigmoid(d[..., 3]))  # (B, Ld, 2, 24)
            cv_t = cv[:, :, 0]
            if diel_on:
                sl = slice(T - Ld, T)
                use = None
                if self.months is not None:                 # kill-rule fallback: only the days of the listed months
                    edges = torch.tensor(MONTH_START_DOY[1:], dtype=x.dtype, device=x.device) - 0.5
                    month = torch.bucketize(doy[:, sl].contiguous(), edges) + 1
                    use = torch.zeros_like(month, dtype=torch.bool)
                    for m_ in self.months:
                        use |= month == m_

                def day(v):                                 # (B, T) or (B, 1) → decoded days, optionally the subset
                    v = v[:, sl] if v.shape[1] == T else v.expand(-1, Ld)
                    return v if use is None else v[use]

                if use is None or bool(use.any()):
                    kwd = {k_: day(v) for k_, v in kw.items()}
                    te_d, k_d = day(te), day(kx)
                    hp, _ = TP.heat_flux(te_d, **{**kwd, "ta": kwd["ta"] + DTA})
                    hm, _ = TP.heat_flux(te_d, **{**kwd, "ta": kwd["ta"] - DTA})
                    dh_dta = (hp - hm) / (2.0 * DTA)                                  # W m-2 K-1
                    hmid = (torch.arange(steps, device=x.device, dtype=x.dtype) + 0.5) * (24.0 / steps)
                    ta_an = 0.5 * day(self._dyn(x, "tmax_c") - self._dyn(x, "tmin_c"))[..., None] \
                        * torch.cos((hmid - AIR_PEAK_HOUR) * (2.0 * math.pi / 24.0))
                    sw_an = kwd["sw_abs"][..., None] * (sw_shape(day(self._dyn(x, "photoperiod_h")), steps) - 1.0)
                    f_eq = day(w_atm)[..., None] * (sw_an + dh_dta[..., None] * ta_an) / k_d[..., None]
                    phys = diel_anomaly(f_eq, day(k_day) / steps, cycles) / lab["t_std"]   # (…, 24), model space
                    if use is None:
                        cv_t = phys
                    else:
                        cv_t = cv_t.clone()
                        cv_t[use] = phys.to(cv_t.dtype)
            k = 9
            res = {"mu": torch.stack([mu_t, o[..., 0]], dim=-1), "curve": torch.stack([cv_t, cv[:, :, 1]], dim=-1),
                   "direct": None, "sig_d": sp(o[..., k:k + N_DAILY]) + 1e-3}
            k += N_DAILY
            res["sig_i"] = sp(o[..., k:k + 2]) + 1e-3
            k += 2
            res["aux_mu"] = torch.cat([q[:, -Ld:, None], o[..., k:k + N_AUX - 1]], dim=-1)
            res["aux_sig"] = sp(o[..., k + N_AUX - 1:k + 2 * N_AUX - 1]) + 1e-3
            if return_physics:
                res["physics"] = {"params": th, "te": te, "k_exchange": kx, "t_star": t_star, "tw_mean": tw,
                                  "depth": depth, "k_day": k_day, "melt": melt, "shade": shade, "lag_d": lag,
                                  "residual": r, "p_kpa": p_kpa[:, 0]}
            return res

    return ThermOxT()


make_net.context = physics_context      # the shared trainer passes this context as the fourth argument

__all__ = ["make_net", "physics_context", "PARAMS", "PARAM_NAMES", "bounded", "prior_bias", "ema", "integrate_daily",
           "lagged", "sw_shape", "diel_anomaly", "ema_loop", "integrate_daily_loop"]
