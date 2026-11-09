# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Fixed oxygen physics for SENTINEL-Sim (docs/research/06 §a.6, §f.3). No learnable parameters.

* Solubility: Benson & Krause (1984) freshwater DO saturation with barometric-pressure correction —
  re-used from :mod:`sentinel.models.hydromet.physics` (owned by M6; imported, not copied, so the
  two workstreams share one tested implementation).
* Gas exchange: K_O2 = K600 · (Sc_O2(T)/600)^(−1/2), Sc_O2 freshwater polynomial (Wanninkhof 1992).
* Light: hourly fraction of the daily light total from solar elevation (declination + hour angle,
  local solar time); daily GPP is distributed in proportion to it (Appling et al. 2018 convention
  of light-proportional GPP, ER constant within the day).
* Hydraulic geometry (Raymond et al. 2012, the streamMetabolizer defaults): depth z = 0.409·Q^0.294 m,
  velocity V = 0.194·Q^0.285 m/s (Q in m³/s); prior K600 from Raymond et al. 2012 eq. "model 1":
  k600 = (V·S)^0.89 · z^0.54 · 5037 m/d, K600 = k600 / z (1/d). Used only as a weak prior.
* Integrator: one-station open-channel model
  dC/dt = K_O2(T_h)·(C_sat(T_h, P) − C) + (GPP_d·w_h·24 + ER_d)/z_d     [mg L⁻¹ d⁻¹]
  integrated EXACTLY for piecewise-constant hourly coefficients (affine map per hour,
  C_{n+1} = a_n·C_n + b_n with a_n = exp(−K_nΔt)), composed over the whole sequence with a parallel
  (Hillis–Steele) scan — unconditionally stable for any K600, differentiable, no Euler error.

Units: GPP, ER areal g O2 m⁻² d⁻¹ (ER ≤ 0), K600 d⁻¹, C mg/L (= g m⁻³), z m, T °C.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from ..hydromet.physics import do_sat_mgl, pressure_atm_from_elevation  # noqa: F401  (re-export)

HOURS = 24
DT_DAY = 1.0 / HOURS
CFS_TO_M3S = 0.028316846592
DEPTH_C, DEPTH_F = 0.409, 0.294        # Raymond 2012 / streamMetabolizer calc_depth defaults
VEL_C, VEL_F = 0.194, 0.285            # Raymond 2012 / streamMetabolizer calc_velocity defaults
T_PEAK_H = 15.0                        # local solar hour of the diel water-temperature maximum
ER_EA_EV = 0.65                        # Yvon-Durocher 2012 activation energy (checks only)
K_BOLTZ_EV = 8.617e-5


# ----------------------------------------------------------------------------- gas exchange

def schmidt_o2(temp_c):
    """Schmidt number of O2 in fresh water (Wanninkhof 1992): 1800.6 − 120.10T + 3.7818T² − 0.047608T³."""
    t = temp_c
    return 1800.6 - 120.10 * t + 3.7818 * t ** 2 - 0.047608 * t ** 3


def k600_to_ko2(k600, temp_c):
    """K_O2 = K600 · (Sc_O2(T)/600)^−0.5 (d⁻¹)."""
    sc = schmidt_o2(temp_c)
    if isinstance(sc, torch.Tensor):
        return k600 * (sc.clamp(min=50.0) / 600.0) ** -0.5
    return k600 * (np.maximum(sc, 50.0) / 600.0) ** -0.5


def arrhenius_ratio(t_c, dt_c, ea_ev: float = ER_EA_EV):
    """Rate ratio for warming t → t + dt under Boltzmann–Arrhenius kinetics (checks/E1 only)."""
    t1, t2 = np.asarray(t_c, float) + 273.15, np.asarray(t_c, float) + dt_c + 273.15
    return np.exp(ea_ev / K_BOLTZ_EV * (1.0 / t1 - 1.0 / t2))


# ----------------------------------------------------------------------------- hydraulics

def depth_m(q_m3s):
    """Mean depth from discharge (Raymond 2012 hydraulic geometry), clamped to [0.05, 15] m."""
    if isinstance(q_m3s, torch.Tensor):
        return (DEPTH_C * q_m3s.clamp(min=1e-4) ** DEPTH_F).clamp(0.05, 15.0)
    return np.clip(DEPTH_C * np.maximum(q_m3s, 1e-4) ** DEPTH_F, 0.05, 15.0)


def velocity_ms(q_m3s):
    if isinstance(q_m3s, torch.Tensor):
        return VEL_C * q_m3s.clamp(min=1e-4) ** VEL_F
    return VEL_C * np.maximum(q_m3s, 1e-4) ** VEL_F


def k600_prior(q_m3s, slope):
    """Raymond 2012 "model 1" K600 (d⁻¹) from discharge and channel slope (m/m, floored at 1e-5)."""
    if isinstance(q_m3s, torch.Tensor):
        s = slope.clamp(min=1e-5) if isinstance(slope, torch.Tensor) else max(float(slope), 1e-5)
        z = depth_m(q_m3s)
        k = (velocity_ms(q_m3s) * s) ** 0.89 * z ** 0.54 * 5037.0
        return (k / z).clamp(0.05, 150.0)
    s = np.maximum(np.asarray(slope, float), 1e-5)
    z = depth_m(q_m3s)
    k = (velocity_ms(q_m3s) * s) ** 0.89 * z ** 0.54 * 5037.0
    return np.clip(k / z, 0.05, 150.0)


# ----------------------------------------------------------------------------- light

def solar_declination(doy):
    return np.deg2rad(23.44) * np.sin(2 * np.pi * (284.0 + np.asarray(doy, float)) / 365.0)


def hourly_light_weights(lat_deg, doy) -> np.ndarray:
    """(S, D, 24) fraction of each day's light falling in local-solar hour h (hour midpoints).

    w = max(sin(elevation), 0) / Σ_h max(sin(elevation), 0); polar night → uniform zeros except a
    tiny floor so the weights still sum to 1 (GPP is then spread evenly; irrelevant in CONUS).
    """
    lat = np.deg2rad(np.asarray(lat_deg, float))[:, None, None]
    dec = solar_declination(doy)[None, :, None]
    h = (np.arange(HOURS) + 0.5)[None, None, :]
    omega = np.deg2rad(15.0 * (h - 12.0))
    sin_el = np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(omega)
    w = np.maximum(sin_el, 0.0) + 1e-9
    return (w / w.sum(axis=-1, keepdims=True)).astype("float32")


def day_length_h(lat_deg, doy) -> np.ndarray:
    """Astronomical day length (h), (S, D)."""
    lat = np.deg2rad(np.asarray(lat_deg, float))[:, None]
    dec = solar_declination(doy)[None, :]
    x = np.clip(-np.tan(lat) * np.tan(dec), -1.0, 1.0)
    return (24.0 / np.pi * np.arccos(x)).astype("float32")


def diel_temperature(t_mean, t_amp):
    """Hourly water temperature (…, 24) = mean + amp·cos(2π(h + 0.5 − T_PEAK_H)/24) (torch)."""
    h = torch.arange(HOURS, dtype=t_mean.dtype, device=t_mean.device) + 0.5
    shape = torch.cos(2 * math.pi * (h - T_PEAK_H) / HOURS)
    return t_mean[..., None] + t_amp[..., None] * shape


# ----------------------------------------------------------------------------- integrator

def affine_scan(a: torch.Tensor, b: torch.Tensor, c0: torch.Tensor) -> torch.Tensor:
    """States of the recurrence c_n = a_n·c_{n−1} + b_n (n = 1..N) for (B, N) a, b and (B,) c0.

    Inclusive Hillis–Steele scan over affine maps; O(N log N) work, log2 N sequential steps.
    """
    A, Bv = a, b
    n = a.shape[-1]
    off = 1
    while off < n:
        a_prev = torch.nn.functional.pad(A[..., :-off], (off, 0), value=1.0)
        b_prev = torch.nn.functional.pad(Bv[..., :-off], (off, 0), value=0.0)
        Bv = A * b_prev + Bv
        A = A * a_prev
        off *= 2
    return A * c0[..., None] + Bv


def integrate_do(gpp, er, k600, t_hourly, light_w, depth, pressure_atm, c0=None):
    """Hourly DO (mg/L) for B sequences of D days.

    gpp, er, k600, depth: (B, D); t_hourly, light_w: (B, D, 24); pressure_atm: (B,) or float.
    Returns (do_hourly (B, D, 24), do_sat_hourly (B, D, 24)). The state is carried continuously
    across days. c0 defaults to saturation at the first hour.
    """
    B, D, H = t_hourly.shape
    p = pressure_atm if isinstance(pressure_atm, torch.Tensor) else torch.full((B,), float(pressure_atm),
                                                                              dtype=t_hourly.dtype,
                                                                              device=t_hourly.device)
    csat = do_sat_mgl(t_hourly, p[:, None, None])
    kO2 = k600_to_ko2(k600[..., None], t_hourly).clamp(min=1e-3)                 # (B, D, 24) 1/d
    src = (gpp[..., None] * light_w * HOURS + er[..., None]) / depth[..., None]  # mg/L/d
    g = -torch.expm1(-kO2 * DT_DAY)                                              # 1 − a
    a = 1.0 - g
    b = g * csat + (g / kO2) * src
    if c0 is None:
        c0 = csat[:, 0, 0]
    c = affine_scan(a.reshape(B, D * H), b.reshape(B, D * H), c0)
    return c.reshape(B, D, H), csat


def integrate_do_euler(gpp, er, k600, t_hourly, light_w, depth, pressure_atm, c0=None, substeps: int = 60):
    """Reference explicit-Euler integrator (fine substeps) used only to test :func:`integrate_do`."""
    B, D, H = t_hourly.shape
    p = torch.full((B,), float(pressure_atm), dtype=t_hourly.dtype) if not isinstance(pressure_atm, torch.Tensor) \
        else pressure_atm
    csat = do_sat_mgl(t_hourly, p[:, None, None])
    kO2 = k600_to_ko2(k600[..., None], t_hourly)
    src = (gpp[..., None] * light_w * HOURS + er[..., None]) / depth[..., None]
    c = csat[:, 0, 0].clone() if c0 is None else c0.clone()
    out = torch.empty_like(t_hourly)
    dt = DT_DAY / substeps
    for d in range(D):
        for h in range(H):
            for _ in range(substeps):
                c = c + dt * (kO2[:, d, h] * (csat[:, d, h] - c) + src[:, d, h])
            out[:, d, h] = c
    return out, csat
