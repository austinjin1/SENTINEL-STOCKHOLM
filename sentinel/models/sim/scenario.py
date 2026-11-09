# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Scenario API for SENTINEL-Sim (RESEARCH_PLAN §7.1 counterfactual rules; protocol §5).

Only physical forcings may be perturbed: air temperature, discharge (with an optional precipitation
scaling) and light. No nutrient levers, no learned-embedding levers — :class:`Scenario` has no field
for them and rejects unknown keywords. Every lever is bounded by :data:`LIMITS` (the range the
protocol's metamorphic checks cover); values outside raise :class:`ScenarioRangeError`.

Choices that are part of the scenario definition (stated, not hidden):
* ``dT_air`` adds ΔT to gridMET Tmax and Tmin. Relative humidity is held constant, so VPD is scaled
  by e_s(T + ΔT)/e_s(T) (Tetens, daily-mean air temperature).
* ``q_scale`` multiplies observed daily discharge. Days whose scaled discharge falls outside the
  site's train-block [0.5, 99.5] percentile envelope are flagged as extrapolation.
* ``precip_scale`` multiplies precipitation only. Discharge is an observed input, so a precipitation
  change WITHOUT ``q_scale`` only probes the network's direct precipitation pathways; it is reported as
  "model association", not as a hydrologic scenario.
* ``sw_scale`` multiplies shortwave radiation (riparian shading / light loss); GPP's diel light
  weights are unchanged (they are geometric), the daily light input to the network is scaled.
* Air temperature beyond the site's train-block maximum (+0 °C) is flagged per day.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from . import data as D
from .model import simulate

LIMITS = {"dT_air": (-3.0, 4.0), "q_scale": (0.5, 1.5), "sw_scale": (0.5, 1.1), "precip_scale": (0.5, 1.5)}
FORBIDDEN_LEVERS = ("nutrient", "nitrate", "phosphorus", "tp", "tn", "embedding", "latent", "land_use")


class ScenarioRangeError(ValueError):
    pass


@dataclass(frozen=True)
class Scenario:
    dT_air: float = 0.0
    q_scale: float = 1.0
    sw_scale: float = 1.0
    precip_scale: float = 1.0
    label: str = ""

    def __post_init__(self):
        for k, (lo, hi) in LIMITS.items():
            v = getattr(self, k)
            if not (lo <= v <= hi):
                raise ScenarioRangeError(f"{k}={v} outside validated range [{lo}, {hi}] (protocol §5.3)")

    @classmethod
    def from_dict(cls, cfg: dict) -> "Scenario":
        bad = [k for k in cfg if k not in LIMITS and k != "label"]
        if bad:
            hint = [k for k in bad if any(f in k.lower() for f in FORBIDDEN_LEVERS)]
            raise ScenarioRangeError(f"unknown/forbidden scenario levers {bad}"
                                     + (f" (forbidden by RESEARCH_PLAN §7.1: {hint})" if hint else ""))
        return cls(**cfg)

    @property
    def is_baseline(self) -> bool:
        return self.dT_air == 0 and self.q_scale == 1 and self.sw_scale == 1 and self.precip_scale == 1

    def kind(self) -> str:
        if self.precip_scale != 1 and self.q_scale == 1:
            return "model_association"
        return "physical_forcing"


def _es_kpa(t_c):
    return 0.6108 * np.exp(17.27 * t_c / (t_c + 237.3))


def perturb(d: D.SimData, idx, scen: Scenario) -> tuple[np.ndarray, np.ndarray, dict]:
    """Perturbed raw forcing (S, T, 8) and discharge (S, T) for sites ``idx`` + extrapolation flags."""
    idx = np.asarray(idx)
    f = d.forcing[idx].copy()
    q = d.q_m3s[idx].copy()
    iT, it = D.FORCING_RAW.index("tmax_c"), D.FORCING_RAW.index("tmin_c")
    tmean0 = 0.5 * (f[..., iT] + f[..., it])
    if scen.dT_air:
        f[..., iT] += scen.dT_air
        f[..., it] += scen.dT_air
        iv = D.FORCING_RAW.index("vpd_kpa")
        f[..., iv] = f[..., iv] * _es_kpa(tmean0 + scen.dT_air) / _es_kpa(tmean0)
    if scen.sw_scale != 1:
        f[..., D.FORCING_RAW.index("srad_wm2")] *= scen.sw_scale
    if scen.precip_scale != 1:
        f[..., D.FORCING_RAW.index("precip_mm")] *= scen.precip_scale
    if scen.q_scale != 1:
        q = q * scen.q_scale
    tr = D.block_mask(d.dates, "train")
    q_lo = np.nanpercentile(np.where(tr[None, :], d.q_m3s[idx], np.nan), 0.5, axis=1)
    q_hi = np.nanpercentile(np.where(tr[None, :], d.q_m3s[idx], np.nan), 99.5, axis=1)
    t_hi = np.nanmax(np.where(tr[None, :], d.forcing[idx][..., iT], np.nan), axis=1)
    flags = {"q_extrap": (q < q_lo[:, None]) | (q > q_hi[:, None]),
             "tair_extrap": f[..., iT] > t_hi[:, None]}
    return f.astype("float32"), q.astype("float32"), flags


def run(model, d: D.SimData, idx, scen: Scenario, t_start: int, t_end: int, *, device="cpu",
        baseline: dict | None = None) -> dict:
    """Simulate baseline (unless given) and scenario for sites ``idx``; returns deltas + flags.

    Output keys: ``base``/``scen`` (dict var → (S, T)), ``delta`` (scen − base), ``flags`` (fraction of
    days flagged per site, over [t_start, t_end)), ``scenario`` (definition), ``kind``.
    """
    idx = np.asarray(idx)
    if baseline is None:
        baseline = simulate(model, d, idx, t_start, t_end, device=device)
    f, q, flags = perturb(d, idx, scen)
    sc = simulate(model, d, idx, t_start, t_end, device=device, forcing=f, q_m3s=q)
    delta = {k: sc[k] - baseline[k] for k in sc}
    fl = {k: v[:, t_start:t_end].mean(axis=1) for k, v in flags.items()}
    return {"base": baseline, "scen": sc, "delta": delta, "flags": fl, "scenario": asdict(scen),
            "kind": scen.kind(), "dates": d.dates[t_start:t_end], "sites": [d.sites[i] for i in idx]}


def summer_mask(dates: pd.DatetimeIndex) -> np.ndarray:
    return np.isin(dates.month, (6, 7, 8))
