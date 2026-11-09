# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Spill severity: per-taxon predictions -> species sensitivity distribution -> HC5 -> risk quotient.

RESEARCH_PLAN §5.4 and docs/research/04 §D.4. **Interface only** in phase 1: the functions here are
fully implemented and unit-tested on synthetic inputs, but nothing is run at scale and no severity
number is logged to the registry until the ChemTox model is frozen and the NRC/USGS matchup table
is built by the data workstream.

Pipeline

1. :func:`species_sensitivity_distribution` — fit a log-normal SSD to per-taxon predicted
   log10 EC50/LC50 values (one value per taxon; a taxon with several predicted conditions is
   averaged first, which is what "model-based SSD" means in the SSD literature). Returns the
   fitted mu/sigma, the taxon count and the model's own prediction uncertainty if supplied.
2. :func:`hc5` — the 5th percentile of that distribution (HC5), i.e. the concentration protecting
   95% of species, in log10 mol/L and in mg/L when the molar mass is known. Parametric CIs come
   from the fitted SD (and, with ``pred_sd``, from the model's predictive spread, propagated by
   adding variances — the honest treatment of "the SSD of an *estimated* distribution").
3. :func:`predicted_environmental_concentration` — PEC from a release: the amount that reached
   water divided by a dilution volume derived from gauged discharge over the mixing window
   (``discharge_m3_s * window_s``), i.e. a transparent, stated assumption, not a transport model.
4. :func:`risk_quotient` — RQ = PEC / PNEC with PNEC = HC5 / assessment factor (the standard
   regulatory form; the assessment factor is an explicit argument, default 1.0 so the raw ratio is
   never silently inflated). :func:`severity_score` wraps the whole chain for one incident and
   returns every intermediate quantity plus the uncertainty band, so a reported severity can be
   audited back to the prediction that produced it.

Everything is in log10 molar units internally (ADORE's target). Conversions to mg/L require the
molar mass and are explicit.

Input units, checked against the built NRC tables
(``$SENTINEL_DATA/processed/nrc_incidents/materials.parquet``, the data workstream's file): it
carries ``amount_in_water``/``amount_in_water_litres`` and ``amount_released_kg`` with a
qualifier column, so the in-water amount usually arrives in **litres**. This module takes
kilograms: the litres -> kilograms conversion needs a density, which must come from the chemicals
table and be recorded with the result (it is not assumed here, and ``amount_in_water_kg`` is never
silently filled from a volume).

Units and caveats are returned with the numbers, because the validation available for this module
is weak: NRC reports are initial, unvalidated, and the response-action fields are reporting-biased
(docs/research/04 §D.4 step 4). An honest evaluation ranks incidents; it does not claim to predict
fish kills.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

LOG10 = np.log(10.0)


# ----------------------------------------------------------------------------- SSD


@dataclass
class SSD:
    """A log-normal species sensitivity distribution fitted to per-taxon predictions."""

    mu: float                     # mean of log10 EC50 over taxa (log10 mol/L)
    sigma: float                  # SD across taxa (log10 units)
    n_taxa: int
    pred_sd: float | None = None  # model predictive SD (log10 units), propagated into the CI
    taxa: tuple[str, ...] = ()
    unit: str = "log10 mol/L"

    @property
    def total_sigma(self) -> float:
        """Across-taxa spread widened by the model's own uncertainty (variances add)."""
        if self.pred_sd is None:
            return self.sigma
        return float(np.sqrt(self.sigma ** 2 + self.pred_sd ** 2))

    def quantile(self, q: float, *, include_prediction_error: bool = False) -> float:
        from scipy.stats import norm

        s = self.total_sigma if include_prediction_error else self.sigma
        return float(self.mu + s * norm.ppf(q))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["total_sigma"] = self.total_sigma
        return d


def species_sensitivity_distribution(
    predictions: pd.Series | dict[str, float] | np.ndarray,
    *,
    taxa=None,
    pred_sd: float | None = None,
    min_taxa: int = 5,
) -> SSD:
    """Fit a log-normal SSD to per-taxon predicted log10 EC50/LC50 values.

    ``predictions`` may be a mapping taxon -> value, a Series indexed by taxon, or an array with
    ``taxa`` given separately (several rows per taxon are averaged first).
    """
    if isinstance(predictions, dict):
        s = pd.Series(predictions, dtype=float)
    elif isinstance(predictions, pd.Series):
        s = predictions.astype(float)
    else:
        if taxa is None:
            raise ValueError("array input needs `taxa`")
        s = pd.Series(np.asarray(predictions, dtype=float),
                      index=pd.Index(np.asarray(taxa, dtype=object).astype(str), name="taxon"))
    per_taxon = s.groupby(level=0).mean().dropna()
    if len(per_taxon) < min_taxa:
        raise ValueError(f"SSD needs >= {min_taxa} taxa, got {len(per_taxon)}")
    return SSD(mu=float(per_taxon.mean()), sigma=float(per_taxon.std(ddof=1)),
               n_taxa=len(per_taxon), pred_sd=pred_sd,
               taxa=tuple(map(str, per_taxon.index)))


def hc5(ssd: SSD, *, molar_mass_g_mol: float | None = None, include_prediction_error: bool = True,
        alpha: float = 0.05) -> dict:
    """HC5 (5th percentile of the SSD) with a parametric band; mg/L too when the molar mass is given."""
    from scipy.stats import norm

    q = 0.05
    point = ssd.quantile(q)
    widened = ssd.quantile(q, include_prediction_error=include_prediction_error)
    se = ssd.total_sigma * np.sqrt(1.0 / ssd.n_taxa + norm.ppf(q) ** 2 / (2 * (ssd.n_taxa - 1)))
    z = norm.ppf(1 - alpha / 2)
    out = {
        "hc5_log10_mol_per_L": float(point),
        "hc5_log10_mol_per_L_with_model_error": float(widened),
        "ci_low": float(widened - z * se),
        "ci_high": float(widened + z * se),
        "n_taxa": ssd.n_taxa,
        "sigma": ssd.sigma,
        "total_sigma": ssd.total_sigma,
        "ci_method": f"parametric log-normal SSD, alpha={alpha}",
    }
    out["hc5_mol_per_L"] = float(10.0 ** widened)
    if molar_mass_g_mol:
        out["hc5_mg_per_L"] = float(10.0 ** widened * float(molar_mass_g_mol) * 1e3)
        out["molar_mass_g_mol"] = float(molar_mass_g_mol)
    return out


# ----------------------------------------------------------------------------- exposure side


def predicted_environmental_concentration(
    amount_in_water_kg: float,
    *,
    discharge_m3_s: float,
    mixing_window_h: float = 24.0,
    molar_mass_g_mol: float | None = None,
    dilution_factor: float = 1.0,
) -> dict:
    """PEC from a release and a gauged-discharge dilution assumption.

    The dilution volume is ``discharge_m3_s * mixing_window_h * 3600 * dilution_factor`` m^3 — a
    stated assumption (complete mixing into the flow passing the site during the window), not a
    transport model. ``dilution_factor`` < 1 represents a partially mixed plume.
    """
    if amount_in_water_kg < 0:
        raise ValueError("amount_in_water_kg must be >= 0")
    if discharge_m3_s <= 0:
        raise ValueError("discharge_m3_s must be > 0 (no gauge match -> no PEC)")
    if mixing_window_h <= 0 or dilution_factor <= 0:
        raise ValueError("mixing_window_h and dilution_factor must be > 0")
    volume_m3 = float(discharge_m3_s) * float(mixing_window_h) * 3600.0 * float(dilution_factor)
    mg_per_L = float(amount_in_water_kg) * 1e6 / (volume_m3 * 1e3)
    out = {
        "pec_mg_per_L": mg_per_L,
        "dilution_volume_m3": volume_m3,
        "assumption": (f"complete mixing into {mixing_window_h:g} h of gauged discharge "
                       f"({discharge_m3_s:g} m3/s), dilution factor {dilution_factor:g}"),
    }
    if molar_mass_g_mol:
        mol_per_L = mg_per_L * 1e-3 / float(molar_mass_g_mol)
        out["pec_mol_per_L"] = mol_per_L
        out["pec_log10_mol_per_L"] = float(np.log10(mol_per_L)) if mol_per_L > 0 else float("-inf")
    return out


def risk_quotient(pec: float, pnec: float) -> float:
    """RQ = PEC / PNEC in matching units (RQ > 1 means the protective level is exceeded)."""
    if pnec <= 0:
        raise ValueError("pnec must be > 0")
    return float(pec) / float(pnec)


def pnec_from_hc5(hc5_value: float, *, assessment_factor: float = 1.0) -> float:
    """PNEC = HC5 / assessment factor (the factor is explicit; default 1 = raw HC5)."""
    if assessment_factor <= 0:
        raise ValueError("assessment_factor must be > 0")
    return float(hc5_value) / float(assessment_factor)


# ----------------------------------------------------------------------------- one incident


@dataclass
class SeverityResult:
    """Everything behind one severity score, so the number can be audited."""

    risk_quotient: float
    risk_quotient_low: float
    risk_quotient_high: float
    pec_mg_per_L: float
    pnec_mg_per_L: float
    hc5: dict = field(default_factory=dict)
    pec: dict = field(default_factory=dict)
    inputs: dict = field(default_factory=dict)
    caveats: tuple[str, ...] = (
        "NRC incident reports are initial and unvalidated (amount in water is often an estimate)",
        "dilution is an explicit assumption, not a transport model",
        "HC5 is model-based: it inherits the toxicity model's error, propagated into the band",
    )

    def to_dict(self) -> dict:
        return asdict(self)


def severity_score(
    per_taxon_predictions,
    *,
    amount_in_water_kg: float,
    discharge_m3_s: float,
    molar_mass_g_mol: float,
    mixing_window_h: float = 24.0,
    dilution_factor: float = 1.0,
    assessment_factor: float = 1.0,
    pred_sd: float | None = None,
    min_taxa: int = 5,
) -> SeverityResult:
    """Severity of one incident: SSD -> HC5 -> PNEC, PEC from release + dilution, then RQ.

    ``per_taxon_predictions`` are predicted log10 EC50/LC50 (mol/L) keyed by taxon; ``pred_sd`` is
    the toxicity model's predictive SD in log10 units (e.g. its test RMSE), which widens the band.
    """
    ssd = species_sensitivity_distribution(per_taxon_predictions, pred_sd=pred_sd, min_taxa=min_taxa)
    h = hc5(ssd, molar_mass_g_mol=molar_mass_g_mol)
    pec = predicted_environmental_concentration(
        amount_in_water_kg, discharge_m3_s=discharge_m3_s, mixing_window_h=mixing_window_h,
        molar_mass_g_mol=molar_mass_g_mol, dilution_factor=dilution_factor,
    )
    to_mg = lambda log10_mol: 10.0 ** log10_mol * molar_mass_g_mol * 1e3
    pnec = pnec_from_hc5(h["hc5_mg_per_L"], assessment_factor=assessment_factor)
    rq = risk_quotient(pec["pec_mg_per_L"], pnec)
    # a lower HC5 (more toxic) gives a higher RQ, hence the crossed bounds
    rq_high = risk_quotient(pec["pec_mg_per_L"],
                            pnec_from_hc5(to_mg(h["ci_low"]), assessment_factor=assessment_factor))
    rq_low = risk_quotient(pec["pec_mg_per_L"],
                           pnec_from_hc5(to_mg(h["ci_high"]), assessment_factor=assessment_factor))
    return SeverityResult(
        risk_quotient=rq, risk_quotient_low=rq_low, risk_quotient_high=rq_high,
        pec_mg_per_L=pec["pec_mg_per_L"], pnec_mg_per_L=pnec, hc5=h, pec=pec,
        inputs={
            "amount_in_water_kg": float(amount_in_water_kg),
            "discharge_m3_s": float(discharge_m3_s),
            "molar_mass_g_mol": float(molar_mass_g_mol),
            "mixing_window_h": float(mixing_window_h),
            "dilution_factor": float(dilution_factor),
            "assessment_factor": float(assessment_factor),
            "pred_sd_log10": None if pred_sd is None else float(pred_sd),
            "ssd": ssd.to_dict(),
        },
    )
