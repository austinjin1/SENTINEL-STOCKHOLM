# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Process components for SENTINEL-HydroMet (docs/research/06 §a.6).

* :func:`do_sat_mgl` — Benson & Krause (1984) freshwater O2 solubility at 1 atm (APHA Standard
  Methods 4500-O form) with the barometric-pressure correction (water-vapour term and the θ
  second-virial term). Works for numpy arrays and torch tensors (same code path via ``xp``).
* :func:`pressure_atm_from_elevation` — standard-atmosphere barometric pressure.
* :func:`air2stream_simulate` — air2stream-type lumped water-temperature model (Toffolon &
  Piccolroaz 2015) WITHOUT the discharge term (θ = 1; we never use gauged discharge, which is a
  sensor input): ``Tw[t] = Tw[t-1] + a1 + a2·Ta[t] − a3·Tw[t-1] + a5·cos(2π(doy/365.25 − a6))``,
  clipped at 0 °C. Ta[t] is the same-day air temperature (nowcast; forecast mode shifts it).
"""

from __future__ import annotations

import math

import numpy as np


def _xp(x):
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return torch
    except ImportError:  # pragma: no cover
        pass
    return np


def do_sat_mgl(temp_c, pressure_atm=1.0):
    """Dissolved-oxygen saturation (mg/L), fresh water, Benson & Krause (1984).

    ln C* = −139.34411 + 1.575701e5/T − 6.642308e7/T² + 1.243800e10/T³ − 8.621949e11/T⁴ (T in K),
    pressure correction C_p = C*·P·(1 − Pwv/P)(1 − θP) / ((1 − Pwv)(1 − θ)) with
    ln Pwv = 11.8571 − 3840.70/T − 216961/T² (atm) and θ = 0.000975 − 1.426e-5·t + 6.436e-8·t².
    Temperatures are clipped to [0, 40] °C (validity range of the fit).
    """
    xp = _xp(temp_c)
    t = xp.clip(temp_c, 0.0, 40.0) if xp is np else temp_c.clamp(0.0, 40.0)
    T = t + 273.15
    ln_c = (-139.34411 + 1.575701e5 / T - 6.642308e7 / T ** 2 + 1.243800e10 / T ** 3
            - 8.621949e11 / T ** 4)
    c = xp.exp(ln_c)
    pwv = xp.exp(11.8571 - 3840.70 / T - 216961.0 / T ** 2)
    theta = 0.000975 - 1.426e-5 * t + 6.436e-8 * t ** 2
    P = pressure_atm
    return c * P * (1 - pwv / P) * (1 - theta * P) / ((1 - pwv) * (1 - theta))


def pressure_atm_from_elevation(elev_m):
    """Standard-atmosphere pressure (atm): (1 − 2.25577e-5·z)^5.25588; NaN elevation -> 1 atm."""
    z = np.asarray(elev_m, dtype="float64")
    z = np.where(np.isfinite(z), np.clip(z, -100.0, 5000.0), 0.0)
    return (1.0 - 2.25577e-5 * z) ** 5.25588


# ----------------------------------------------------------------------------- air2stream

A2S_PARAMS = ("a1", "a2", "a3", "a5", "a6")
A2S_BOUNDS = {"a1": (-3.0, 3.0), "a2": (0.0, 1.0), "a3": (0.0, 1.0), "a5": (0.0, 3.0), "a6": (0.0, 1.0)}


def air2stream_simulate(ta, doy, params, tw0=None):
    """Simulate water temperature for S sites. ``ta`` (S, T) °C air temperature (NaN -> carried
    forward), ``doy`` (T,), ``params`` dict of (S,) arrays (or tensors). Returns (S, T).

    Causal: Tw[t] depends only on Ta[0..t]. Works with numpy or torch (autograd for calibration).
    """
    xp = _xp(ta)
    S, T = ta.shape
    a1, a2, a3, a5, a6 = (params[k] for k in A2S_PARAMS)
    if xp is np:
        ta = _ffill(ta)
        season = np.cos(2 * np.pi * (np.asarray(doy, float)[None, :] / 365.25 - np.asarray(a6)[:, None]))
        tw = np.empty((S, T))
        prev = np.maximum(ta[:, :30].mean(axis=1), 0.0) if tw0 is None else np.asarray(tw0, float)
        for t in range(T):
            prev = np.maximum(prev + a1 + a2 * ta[:, t] - a3 * prev + a5 * season[:, t], 0.0)
            tw[:, t] = prev
        return tw
    import torch

    doy_t = torch.as_tensor(np.asarray(doy, float), dtype=ta.dtype, device=ta.device)
    season = torch.cos(2 * math.pi * (doy_t[None, :] / 365.25 - a6[:, None]))
    prev = ta[:, :30].mean(dim=1).clamp(min=0.0) if tw0 is None else tw0
    out = []
    for t in range(T):
        prev = torch.relu(prev + a1 + a2 * ta[:, t] - a3 * prev + a5 * season[:, t])
        out.append(prev)
    return torch.stack(out, dim=1)


def _ffill(a: np.ndarray) -> np.ndarray:
    """Forward-fill NaN along time (causal); leading NaN -> first finite value of the row."""
    a = np.array(a, dtype="float64", copy=True)
    S, T = a.shape
    idx = np.where(np.isfinite(a), np.arange(T)[None, :], 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    out = a[np.arange(S)[:, None], idx]
    for i in np.flatnonzero(~np.isfinite(out[:, 0])):
        f = np.flatnonzero(np.isfinite(a[i]))
        if f.size:
            out[i, : f[0]] = a[i, f[0]]
    return out
