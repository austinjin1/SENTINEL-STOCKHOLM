# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Surface energy balance of a stream reach (amendment §6.2), written once for numpy arrays and torch tensors.

Net heat flux at the water surface (W m⁻², positive into the water), with water temperature Tw (°C):

    H(Tw) = (1 − shade)(1 − α) S  +  ε_w ε_a σ Ta_K⁴  −  ε_w σ Tw_K⁴  −  f(u)[e_s(Tw) − e_a]  −  γ f(u)(Tw − Ta)

The **equilibrium temperature** Te is the root H(Te) = 0 and the **exchange coefficient** K = −dH/dTw at Te
(Edinger, Duttweiler & Geyer 1968, Water Resour. Res. 4:1137). Both come from a fixed number of Newton steps started
at air temperature. With depth D the reach relaxes as dTw/dt = K (Te − Tw) / (ρ c_p D).

Formulas and constants: every number is defined once, with its source, in
:mod:`sentinel.hydromet_sota.gso_mt.constants` (unit tests against the printed values and the worked examples of the
sources in ``tests/test_hydromet_sota_v2_constants.py`` and ``tests/test_hydromet_sota_v2_physics.py``):

* saturation vapour pressure e°(T) = 0.6108 exp(17.27 T / (T + 237.3)) kPa and its slope Δ = 4098 e°/(T + 237.3)²
  — FAO-56 (Allen et al. 1998) eq. 11 and 13;
* psychrometric constant γ = 0.665e-3 · P kPa °C⁻¹ — FAO-56 eq. 8; P comes from the shared
  ``sentinel.models.hydromet.physics.pressure_atm_from_elevation`` (no second pressure formula);
* 2 m wind from 10 m wind: u2 = u10 · 4.87 / ln(67.8 · 10 − 5.42) — FAO-56 eq. 47;
* actual vapour pressure from specific humidity q: e_a = q P / (ε + (1 − ε) q), ε = 0.622 (FAO-56 eq. 8);
* open-water wind function f(u2) = 6.43 (1 + 0.536 u2) MJ m⁻² d⁻¹ kPa⁻¹ — Penman's open-water form in metric units
  as printed by Shuttleworth (2007, HESS 11:210, eq. 1); the sensible flux uses the same function times γ (Bowen);
* clear-sky atmospheric emissivity ε_clear = 1.24 (e_a[mbar] / Ta[K])^(1/7) — Brutsaert (1975), as reprinted by
  Crawford & Duchon (1999, J. Appl. Meteorol. 38:474, eq. 16); all-sky ε_a = clf + (1 − clf) ε_clear with
  clf = 1 − S / S_clear ≥ 0 — Crawford & Duchon (1999) eq. 3 and 11;
* water emissivity ε_w = 0.97 (Jobson 1978, USGS OFR 78-1026, eq. 9; 3 % of the atmospheric long wave is reflected),
  short-wave albedo of open water α = 0.08 (McMahon et al. 2013, HESS 17:1331, supplement Table S3),
  σ = 5.670374419e-8 W m⁻² K⁻⁴ (CODATA 2018, exact).

**Transcription status (2026-11-10):** all of the above were read in the cited sources on that day (Brutsaert's
coefficient only through Crawford & Duchon's reprint; Edinger et al. 1968 only through its abstract). The values enter
as priors that a model rescales (exchange scaling, shade, wind sheltering). Shade acts on the short wave only (canopy
long wave is not modelled).

No in-situ input: everything is a function of gridded forcings, elevation and learned or prior static parameters."""

from __future__ import annotations

import numpy as np

from ...models.hydromet import data as D
from ..gso_mt import constants as K
from ..gso_mt.constants import pressure_atm_from_elevation

SIGMA = K.SIGMA                   # W m-2 K-4
EMIS_WATER = K.EMIS_WATER
ALBEDO_WATER = K.ALBEDO_WATER
RHO_CP = K.RHO_CP_WATER           # J m-3 K-1 (nominal; D2 relaxation term only)
WIND_FN_A = K.WIND_FN_A_W         # W m-2 kPa-1
WIND_FN_B = K.WIND_FN_B           # per (m s-1) of 2 m wind
WIND_10M_TO_2M = K.WIND_10M_TO_2M
KPA_PER_ATM = K.KPA_PER_ATM
TE_RANGE = (-5.0, 45.0)           # clamp of the equilibrium temperature (amendment §6.2)
N_NEWTON = 4
# Prior medians of the static parameters that the prior-parameter Te of the G2 reference uses (fixed before any fit;
# D2 starts its bounded parameters at the same values: configs/hydromet_sota/v2/d2.yaml, checked by a test).
PRIOR = {"shade": 0.3, "wind_shelter": 0.6}


def _xp(x):
    if type(x).__module__.split(".")[0] == "torch":
        import torch

        return torch
    return np


def sat_vp_kpa(t):
    return K.SVP_A * _xp(t).exp(K.SVP_B * t / (t + K.SVP_C))


def sat_vp_slope(t):
    return K.SVP_SLOPE * sat_vp_kpa(t) / (t + K.SVP_C) ** 2


def pressure_kpa(elev_m) -> np.ndarray:
    return KPA_PER_ATM * pressure_atm_from_elevation(elev_m)


def psychrometric(p_kpa):
    return K.PSYCHROMETRIC * p_kpa


def vp_from_sph(q, p_kpa):
    return q * p_kpa / (K.EPSILON + (1.0 - K.EPSILON) * q)


def wind_function(u2):
    """W m⁻² kPa⁻¹ for 2 m wind speed (m s⁻¹)."""
    return WIND_FN_A * (1.0 + WIND_FN_B * u2)


def emissivity_clear(ea_kpa, ta_c):
    xp = _xp(ea_kpa)
    return K.BRUTSAERT_A * (xp.clip(K.HPA_PER_KPA * ea_kpa, 1e-3, None) / (ta_c + K.KELVIN_OFFSET)) ** K.BRUTSAERT_EXPONENT


def emissivity_all_sky(ea_kpa, ta_c, cloud_ratio):
    """``cloud_ratio`` = S / clear-sky S (the solar channel of the input set); a missing ratio is treated as clear."""
    xp = _xp(ea_kpa)
    clf = xp.clip(1.0 - cloud_ratio, 0.0, 1.0)
    return clf + (1.0 - clf) * emissivity_clear(ea_kpa, ta_c)


def heat_flux(tw, *, sw_abs, ta, ea, eps_a, fu, gamma):
    """``(H, dH/dTw)`` at water temperature ``tw`` (°C). ``sw_abs`` is the absorbed short wave."""
    tk = tw + K.KELVIN_OFFSET
    h = (sw_abs + EMIS_WATER * eps_a * SIGMA * (ta + K.KELVIN_OFFSET) ** 4 - EMIS_WATER * SIGMA * tk ** 4
         - fu * (sat_vp_kpa(tw) - ea) - gamma * fu * (tw - ta))
    dh = -(4.0 * EMIS_WATER * SIGMA * tk ** 3 + fu * (sat_vp_slope(tw) + gamma))
    return h, dh


def equilibrium_temperature(*, sw_abs, ta, ea, eps_a, fu, gamma, n_newton: int = N_NEWTON):
    """``(Te, K)``: ``n_newton`` vectorised Newton steps from air temperature, Te clamped to :data:`TE_RANGE`;
    K (W m⁻² K⁻¹) is the exchange coefficient at Te. H is strictly decreasing and concave in Tw over the range, so
    the iteration is monotone after the first step."""
    xp = _xp(ta)
    te = xp.clip(ta, *TE_RANGE)
    for _ in range(n_newton):
        h, dh = heat_flux(te, sw_abs=sw_abs, ta=ta, ea=ea, eps_a=eps_a, fu=fu, gamma=gamma)
        te = xp.clip(te - h / dh, *TE_RANGE)
    _, dh = heat_flux(te, sw_abs=sw_abs, ta=ta, ea=ea, eps_a=eps_a, fu=fu, gamma=gamma)
    return te, -dh


def surface_drivers(*, tmean, srad, wind10, sph, p_kpa, cloud_ratio, shade, wind_shelter):
    """Keyword arguments of :func:`equilibrium_temperature` from daily forcings (°C, W m⁻², m s⁻¹ at 10 m, kg kg⁻¹),
    pressure (kPa), the cloud ratio and the two surface parameters (shade ∈ [0, 1], wind sheltering ∈ [0, 1])."""
    ea = vp_from_sph(sph, p_kpa)
    return {"sw_abs": (1.0 - shade) * (1.0 - ALBEDO_WATER) * srad, "ta": tmean, "ea": ea,
            "eps_a": emissivity_all_sky(ea, tmean, cloud_ratio),
            "fu": wind_function(WIND_10M_TO_2M * wind10 * wind_shelter), "gamma": psychrometric(p_kpa)}


def te_prior(forc: np.ndarray, elev_m: np.ndarray, cloud_ratio: np.ndarray, *, prior: dict | None = None) -> np.ndarray:
    """(S, T) equilibrium temperature at the prior parameters from the **site-cell** forcing channels of
    ``HydroMetData.forc`` (the first ``len(FORCING_VARS)`` channels): the physics input of the G2 reference
    (amendment §5). NaN where a forcing is missing."""
    p = {**PRIOR, **(prior or {})}
    f = lambda v: forc[:, :, D.FORCING_VARS.index(v)].astype("float64")   # noqa: E731
    cr = np.asarray(cloud_ratio, "float64")
    kw = surface_drivers(tmean=f("tmean_c"), srad=f("srad_wm2"), wind10=f("wind_ms"), sph=f("sph_kgkg"),
                         p_kpa=pressure_kpa(elev_m)[:, None], cloud_ratio=np.where(np.isfinite(cr), cr, 1.0),
                         shade=p["shade"], wind_shelter=p["wind_shelter"])
    te, _ = equilibrium_temperature(**kw)
    return te.astype("float32")
