# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Harmonise parsed WQP (WQX 3.0) results to analysis-ready variables in one canonical unit each.

Every conversion is an explicit table entry (tested in ``tests/test_wqp.py``):

* mass concentrations -> mg/L or ug/L (``MASS_TO_MG_L``), molar -> mass via molar masses;
* speciation -> the variable's basis (nitrate as NO3 x 0.2259 -> as N, orthophosphate as PO4 x
  0.3261 -> as P, ammonium as NH4 x 0.7765, ammonia as NH3 x 0.8224, nitrite as NO2 x 0.3045,
  sulfate as S x 2.9959 -> as SO4); missing speciation -> the variable's default basis with
  ``speciation_assumed=True`` where the choice matters (nitrate, nitrite, ammonia, orthophosphate,
  sulfate);
* lengths -> m (Secchi) or cm (transparency tube); temperature deg F/K -> deg C;
  conductance mS/cm, mmho/cm, S/m -> uS/cm; bacteria per mL/L/dL -> per 100 mL;
* turbidity is never converted: NTU, FNU and NTRU are separate variables, other units unmapped;
* E. coli / enterococci: culture (CFU or MPN per 100 mL) and qPCR (CCE per 100 mL) are separate
  variables; a qPCR method never enters the culture variable;
* sensor/fluorescence units (RFU, QSE "ppb" CDOM) are never converted.

Rows that cannot be mapped keep all fields and get an ``unmapped_reason``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------------------------
# Molar masses (g/mol, IUPAC standard atomic weights) and speciation factors
# ------------------------------------------------------------------------------------------------
ATOMIC = {"H": 1.00794, "C": 12.0107, "N": 14.0067, "O": 15.9994, "P": 30.973762, "S": 32.065,
          "Cl": 35.453}
MOLAR = {
    "N": ATOMIC["N"], "NO3": ATOMIC["N"] + 3 * ATOMIC["O"], "NO2": ATOMIC["N"] + 2 * ATOMIC["O"],
    "NH3": ATOMIC["N"] + 3 * ATOMIC["H"], "NH4": ATOMIC["N"] + 4 * ATOMIC["H"],
    "P": ATOMIC["P"], "PO4": ATOMIC["P"] + 4 * ATOMIC["O"],
    "S": ATOMIC["S"], "SO4": ATOMIC["S"] + 4 * ATOMIC["O"], "C": ATOMIC["C"], "Cl": ATOMIC["Cl"],
}


def species_factor(species: str, basis: str) -> float:
    """Factor converting a mass expressed as ``species`` to a mass expressed as ``basis``
    (one basis atom per species formula unit, e.g. NO3 -> N = 14.0067 / 62.0049 = 0.2259)."""
    return MOLAR[basis] / MOLAR[species]


# ------------------------------------------------------------------------------------------------
# Units
# ------------------------------------------------------------------------------------------------
MASS_TO_MG_L = {
    "mg/l": 1.0, "ppm": 1.0, "mg/dm3": 1.0, "g/m3": 1.0, "ug/ml": 1.0,
    "ug/l": 1e-3, "ppb": 1e-3, "mg/m3": 1e-3, "ng/ml": 1e-3, "ug/dm3": 1e-3,
    "ng/l": 1e-6, "pg/ml": 1e-6, "ppt(trillion)": 1e-6,
    "g/l": 1e3, "mg/ml": 1e3,
}
# amount concentrations -> mol/L
MOLAR_TO_MOL_L = {"mol/l": 1.0, "mmol/l": 1e-3, "umol/l": 1e-6, "um": 1e-6, "mmol/m3": 1e-6,
                  "nmol/l": 1e-9, "nm": 1e-9, "mm": 1e-3}
EQUIV_TO_EQ_L = {"eq/l": 1.0, "meq/l": 1e-3, "ueq/l": 1e-6}
CHARGE = {"Cl": 1, "SO4": 2}  # equivalents per mole, by basis
VAR_CHARGE = {"no3": 1, "no2": 1, "nh3": 1, "chloride": 1, "sulfate": 2}  # by variable (NO3-, NO2-, NH4+)

LENGTH_TO_M = {"m": 1.0, "meter": 1.0, "meters": 1.0, "metre": 1.0, "cm": 0.01, "mm": 0.001,
               "ft": 0.3048, "feet": 0.3048, "foot": 0.3048, "in": 0.0254, "inch": 0.0254,
               "inches": 0.0254}
CONDUCTANCE_TO_US_CM = {"us/cm": 1.0, "umho/cm": 1.0, "umhos/cm": 1.0, "micromhos/cm": 1.0,
                        "ms/cm": 1e3, "mmho/cm": 1e3, "mmhos/cm": 1e3, "s/m": 1e4, "ms/m": 10.0,
                        "us/m": 0.01}
CULTURE_TO_PER_100ML = {
    **{f"{k}/100ml": 1.0 for k in ("cfu", "mpn", "#", "count", "counts", "colonies", "org",
                                   "organisms", "cells", "number")},
    **{f"{k}/dl": 1.0 for k in ("cfu", "mpn", "#")},
    **{f"{k}/ml": 100.0 for k in ("cfu", "mpn", "#")},
    **{f"{k}/l": 0.1 for k in ("cfu", "mpn", "#")},
}
QPCR_TO_CCE_100ML = {"cce/100ml": 1.0, "ce/100ml": 1.0, "cellequivalents/100ml": 1.0,
                     "cce/ml": 100.0, "cce/l": 0.1}
QPCR_OTHER = {"gc/100ml", "copies/100ml", "genecopies/100ml", "gc/ml", "copies/ml", "copies/l",
              "gc/l", "sq/100ml"}
TURBIDITY_UNITS = {"ntu": "turbidity_ntu", "fnu": "turbidity_fnu", "ntru": "turbidity_ntru"}
PH_UNITS = {"", "none", "stdunits", "std.units", "standardunits", "su", "s.u.", "ph", "phunits",
            "unitless", "units", "stdunit"}
PERCENT_UNITS = {"%", "%saturatn", "%sat", "%saturation", "percent", "%ofsaturation", "%sat."}
ABSORBANCE_TO_M1 = {"units/cm": 230.2585093, "au/cm": 230.2585093, "abs/cm": 230.2585093,
                    "absorbance/cm": 230.2585093, "m-1": 1.0, "1/m": 1.0, "/m": 1.0, "m^-1": 1.0,
                    "permeter": 1.0}
SENSOR_UNITS = {"rfu", "qse", "ppbqse", "ugqse/l", "rfu(raw)", "relativefluorescenceunits"}
TEMP_UNITS = {"degc": ("C", 1.0, 0.0), "c": ("C", 1.0, 0.0), "°c": ("C", 1.0, 0.0),
              "celsius": ("C", 1.0, 0.0), "deg.c": ("C", 1.0, 0.0), "degreesc": ("C", 1.0, 0.0),
              "degf": ("F", 5.0 / 9.0, -32.0 * 5.0 / 9.0), "f": ("F", 5.0 / 9.0, -32.0 * 5.0 / 9.0),
              "°f": ("F", 5.0 / 9.0, -32.0 * 5.0 / 9.0), "fahrenheit": ("F", 5.0 / 9.0, -32.0 * 5.0 / 9.0),
              "k": ("K", 1.0, -273.15), "kelvin": ("K", 1.0, -273.15)}

_SPECIES_TOKENS = {"n": "N", "no3": "NO3", "no2": "NO2", "nh3": "NH3", "nh4": "NH4", "p": "P",
                   "po4": "PO4", "so4": "SO4", "s": "S", "c": "C", "cl": "Cl", "no3-n": "N",
                   "no2-n": "N", "nh3-n": "N", "nh4-n": "N", "po4-p": "P", "p04": "PO4"}
_TEMP_COMP = re.compile(r"@?(at)?25(deg|°)?c$")


def norm_unit(u: str) -> str:
    """Lower-case, strip spaces, unify micro sign and 'per'. ``'mg/L as N'`` -> ``'mg/lasn'``."""
    u = (u or "").strip().lower().replace("µ", "u").replace("μ", "u")
    u = u.replace(" per ", "/").replace("liter", "l").replace("litre", "l")
    u = re.sub(r"\s+", "", u)
    return u


def split_unit(u: str) -> tuple[str, str, bool]:
    """Normalised unit -> (base unit, embedded species, temperature-compensated flag).

    ``'mg/lasn'`` -> ``('mg/l', 'N', False)``; ``'us/cm@25c'`` -> ``('us/cm', '', True)``;
    ``'mgn/l'`` -> ``('mg/l', 'N', False)``.
    """
    tc = False
    m = _TEMP_COMP.search(u)
    if m and len(u) > len(m.group(0)):
        u, tc = u[: m.start()], True
    m = re.fullmatch(r"(mg|ug|ng)-?(n|p|no3|no2|nh4|nh3|po4|so4|c)/l", u)
    if m:
        return f"{m.group(1)}/l", _SPECIES_TOKENS[m.group(2)], tc
    for base in sorted(set(MASS_TO_MG_L) | set(MOLAR_TO_MOL_L) | set(EQUIV_TO_EQ_L), key=len,
                       reverse=True):
        if u.startswith(base) and len(u) > len(base):
            rest = u[len(base):].lstrip("-_").removeprefix("as").strip("()")
            if rest in _SPECIES_TOKENS:
                return base, _SPECIES_TOKENS[rest], tc
    return u, "", tc


# ------------------------------------------------------------------------------------------------
# Variables
# ------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Var:
    name: str
    unit: str            # canonical unit label stored in the ``unit`` column
    dim: str             # conversion family
    lo: float            # plausible range (inclusive), see docs/datasheets/wqp_discrete.md
    hi: float
    basis: str = ""      # speciation basis for mass concentrations
    species: tuple[str, ...] = ()   # allowed speciations (convertible to basis)
    assume_flag: bool = False       # flag rows where the basis had to be assumed
    group: str = ""
    description: str = ""


def _v(*args, **kw) -> Var:
    return Var(*args, **kw)


VARIABLES: dict[str, Var] = {v.name: v for v in [
    # optical
    _v("chla", "ug/L", "mass_ug", 0, 10_000, group="optical",
       description="chlorophyll a, pheophytin correction not stated (lab extract)"),
    _v("chla_corrected", "ug/L", "mass_ug", 0, 10_000, group="optical",
       description="chlorophyll a corrected for / free of pheophytin"),
    _v("chla_uncorrected", "ug/L", "mass_ug", 0, 10_000, group="optical",
       description="chlorophyll a uncorrected for pheophytin"),
    _v("pheophytin", "ug/L", "mass_ug", 0, 5_000, group="optical", description="pheophytin a / pheopigments"),
    _v("tss", "mg/L", "mass_mg", 0, 100_000, group="optical", description="total suspended solids"),
    _v("ssc", "mg/L", "mass_mg", 0, 500_000, group="optical", description="suspended sediment concentration"),
    _v("turbidity_ntu", "NTU", "turb", 0, 10_000, group="optical", description="turbidity, NTU"),
    _v("turbidity_fnu", "FNU", "turb", 0, 10_000, group="optical", description="turbidity, FNU (ISO 7027)"),
    _v("turbidity_ntru", "NTRU", "turb", 0, 10_000, group="optical", description="turbidity, ratio NTU"),
    _v("secchi", "m", "length_m", 0.01, 50, group="optical", description="Secchi disk depth"),
    _v("transparency_tube", "cm", "length_cm", 0.5, 125, group="optical",
       description="transparency tube (Secchi tube) reading"),
    _v("acdom440", "1/m", "absorb", 0, 100, group="optical",
       description="CDOM absorption coefficient at 440 nm (from decadic absorbance per cm x 230.26)"),
    _v("cdom_abs", "1/m", "absorb", 0, 200, group="optical",
       description="CDOM absorption coefficient, wavelength not stated"),
    _v("phycocyanin", "ug/L", "mass_ug", 0, 10_000, group="optical", description="phycocyanin (lab extract)"),
    # nutrients
    _v("tp", "mg/L as P", "mass_mg", 0, 50, basis="P", species=("P", "PO4"), group="phosphorus",
       description="total phosphorus (unfiltered; fraction 'Total'/'Unfiltered'/unstated)"),
    _v("tdp", "mg/L as P", "mass_mg", 0, 50, basis="P", species=("P", "PO4"), group="phosphorus",
       description="total dissolved phosphorus (filtered)"),
    _v("orthop", "mg/L as P", "mass_mg", 0, 50, basis="P", species=("P", "PO4"), assume_flag=True,
       group="phosphorus", description="orthophosphate / soluble reactive phosphorus"),
    _v("tn", "mg/L as N", "mass_mg", 0, 100, basis="N", species=("N", "NO3"), group="nitrogen",
       description="total nitrogen (unfiltered; fraction 'Total'/'Unfiltered'/unstated)"),
    _v("tdn", "mg/L as N", "mass_mg", 0, 100, basis="N", species=("N", "NO3"), group="nitrogen",
       description="total dissolved nitrogen (filtered)"),
    _v("no3", "mg/L as N", "mass_mg", 0, 100, basis="N", species=("N", "NO3"), assume_flag=True,
       group="nitrogen", description="nitrate"),
    _v("no2", "mg/L as N", "mass_mg", 0, 20, basis="N", species=("N", "NO2"), assume_flag=True,
       group="nitrogen", description="nitrite"),
    _v("no3no2", "mg/L as N", "mass_mg", 0, 100, basis="N", species=("N", "NO3"), assume_flag=True,
       group="nitrogen", description="nitrate + nitrite"),
    _v("nh3", "mg/L as N", "mass_mg", 0, 100, basis="N", species=("N", "NH3", "NH4"),
       assume_flag=True, group="nitrogen", description="ammonia + ammonium"),
    _v("tkn", "mg/L as N", "mass_mg", 0, 200, basis="N", species=("N",), group="nitrogen",
       description="total Kjeldahl nitrogen (unfiltered)"),
    # carbon
    _v("doc", "mg/L as C", "mass_mg", 0, 500, basis="C", species=("C",), group="carbon",
       description="dissolved organic carbon (incl. dissolved NPOC)"),
    _v("toc", "mg/L as C", "mass_mg", 0, 500, basis="C", species=("C",), group="carbon",
       description="total organic carbon (incl. total NPOC)"),
    # ions
    _v("chloride", "mg/L", "mass_mg", 0, 250_000, basis="Cl", species=("Cl",), group="ions",
       description="chloride"),
    _v("sulfate", "mg/L as SO4", "mass_mg", 0, 100_000, basis="SO4", species=("SO4", "S"),
       assume_flag=True, group="ions", description="sulfate"),
    _v("spcond", "uS/cm", "cond", 0, 250_000, group="ions",
       description="specific conductance at 25 deg C (discrete/field measurement)"),
    # bacteria
    _v("ecoli", "CFU or MPN/100mL", "culture", 0, 1e8, group="bacteria",
       description="E. coli, culture methods (CFU or MPN; see unit_original)"),
    _v("ecoli_qpcr", "CCE/100mL", "qpcr", 0, 1e9, group="bacteria", description="E. coli by qPCR"),
    _v("enterococcus", "CFU or MPN/100mL", "culture", 0, 1e8, group="bacteria",
       description="enterococci, culture methods"),
    _v("enterococcus_qpcr", "CCE/100mL", "qpcr", 0, 1e9, group="bacteria",
       description="enterococci by qPCR (EPA 1609/1611)"),
    _v("fecal_coliform", "CFU or MPN/100mL", "culture", 0, 1e8, group="bacteria",
       description="fecal coliform, culture methods"),
    # cyanotoxins
    _v("microcystins", "ug/L", "mass_ug", 0, 100_000, group="cyanotoxins",
       description="total microcystins (+nodularins), mostly ADDA-ELISA; whole water / unstated"),
    _v("microcystins_dissolved", "ug/L", "mass_ug", 0, 100_000, group="cyanotoxins",
       description="dissolved (filtered) microcystins"),
    _v("microcystins_particulate", "ug/L", "mass_ug", 0, 100_000, group="cyanotoxins",
       description="particulate microcystins"),
    _v("microcystin_lr", "ug/L", "mass_ug", 0, 100_000, group="cyanotoxins", description="microcystin-LR"),
    _v("microcystin_rr", "ug/L", "mass_ug", 0, 100_000, group="cyanotoxins", description="microcystin-RR"),
    _v("microcystin_yr", "ug/L", "mass_ug", 0, 100_000, group="cyanotoxins", description="microcystin-YR"),
    _v("microcystin_la", "ug/L", "mass_ug", 0, 100_000, group="cyanotoxins", description="microcystin-LA"),
    _v("microcystin_other_congener", "ug/L", "mass_ug", 0, 100_000, group="cyanotoxins",
       description="other microcystin congeners (LF, LW, LY, WR, HilR, HtyR, desmethyl...; see characteristic)"),
    _v("cylindrospermopsin", "ug/L", "mass_ug", 0, 10_000, group="cyanotoxins", description="cylindrospermopsin"),
    _v("anatoxin_a", "ug/L", "mass_ug", 0, 10_000, group="cyanotoxins", description="anatoxin-a"),
    _v("saxitoxin", "ug/L", "mass_ug", 0, 10_000, group="cyanotoxins", description="saxitoxin(s)"),
    # field physchem (co-sampled)
    _v("do", "mg/L", "mass_mg", 0, 30, group="physchem", description="dissolved oxygen"),
    _v("do_pctsat", "%", "percent", 0, 300, group="physchem",
       description="dissolved oxygen, % saturation (reported under the DO characteristic)"),
    _v("ph", "pH", "ph", 1, 13, group="physchem", description="pH (field or lab)"),
    _v("temp", "deg C", "temp", -2, 50, group="physchem", description="water temperature"),
]}

# Characteristic -> variable routing. A dict maps fraction class -> variable ("*" = any).
# ``None`` marks a requested-but-ambiguous characteristic (always unmapped, counted).
_ANY = "*"
CHAR_ROUTES: dict[str, dict[str, str] | None] = {
    "Chlorophyll a": {_ANY: "chla"}, "Chlorophyll A": {_ANY: "chla"},
    "Chlorophyll a - Phytoplankton (suspended)": {_ANY: "chla"},
    "Chlorophyll a, corrected for pheophytin": {_ANY: "chla_corrected"},
    "Chlorophyll a, free of pheophytin": {_ANY: "chla_corrected"},
    "Chlorophyll a, uncorrected for pheophytin": {_ANY: "chla_uncorrected"},
    "Chlorophyll": None,  # total chlorophyll (a+b+c?) — not chlorophyll a
    "Pheophytin a": {_ANY: "pheophytin"}, "Pheophytin": {_ANY: "pheophytin"},
    "Phaeophytin - Phytoplankton (suspended)": {_ANY: "pheophytin"},
    "Pheopigments of Chlorophyll": {_ANY: "pheophytin"}, "Phaeopigment": {_ANY: "pheophytin"},
    "Total suspended solids": {_ANY: "tss"}, "Total Suspended Particulate Matter": {_ANY: "tss"},
    "Suspended Sediment Concentration (SSC)": {_ANY: "ssc"},
    "Suspended sediment concentration (SSC)": {_ANY: "ssc"},
    "Turbidity": {_ANY: "@turbidity"}, "Turbidity Field": {_ANY: "@turbidity"},
    "Depth, Secchi disk depth": {_ANY: "secchi"}, "Secchi depth": {_ANY: "secchi"},
    "Water transparency, Secchi disc": {_ANY: "secchi"},
    "Depth, Secchi disk visible at bottom (Y/N) (choice list)": {_ANY: "@aux_secchi_bottom"},
    "Transparency, tube with disk": {_ANY: "transparency_tube"},
    "Transparency, Secchi tube with disk": {_ANY: "transparency_tube"},
    "Colored dissolved organic matter (CDOM)": {_ANY: "cdom_abs"},
    "Absorbance at 440 nm": {_ANY: "acdom440"},
    "Phycocyanin": {_ANY: "phycocyanin"}, "Phycocyanins": {_ANY: "phycocyanin"},
    "Algae-Phycocyanin": {_ANY: "phycocyanin"},
    # phosphorus
    "Phosphorus": {"total": "tp", "unspecified": "tp", "dissolved": "tdp"},
    "Total Phosphorus, mixed forms": {"total": "tp", "unspecified": "tp", "dissolved": "tdp"},
    "Phosphorus as P": {"total": "tp", "unspecified": "tp", "dissolved": "tdp"},
    "Orthophosphate": {_ANY: "orthop"}, "Orthophosphate as P": {_ANY: "orthop"},
    "Orthophosphate as PO4": {_ANY: "orthop"}, "Ortho-Phosphate-Phosphorus": {_ANY: "orthop"},
    "Soluble Reactive Phosphorus (SRP)": {_ANY: "orthop"},
    # 'Phosphate-phosphorus' + fraction Total is legacy STORET "PHOSPHATE, TOTAL" (total P measured
    # as phosphate after digestion): its values match TP (median 0.08-0.13 mg/L vs 0.02-0.03 for
    # orthophosphate; 1.8x orthophosphate in 3,464 same-sample pairs, year-2000 pull). Filtered or
    # unstated fractions are ambiguous (TDP vs orthophosphate) and stay unmapped.
    "Phosphate-phosphorus": {"total": "tp"},
    "Phosphate-phosphorus as P": {"total": "tp"},
    "Phosphate-phosphorus as PO4": {"total": "tp"},
    "Phosphate": {"dissolved": "orthop"},
    # nitrogen
    **{n: {"total": "tn", "unspecified": "tn", "dissolved": "tdn"} for n in (
        "Nitrogen", "Total Nitrogen, mixed forms",
        "Nitrogen, mixed forms (NH3), (NH4), organic, (NO2) and (NO3)",
        "Total Nitrogen, mixed forms (NH3), (NH4), organic, (NO2) and (NO3)")},
    **{n: {_ANY: "no3"} for n in ("Nitrate", "Nitrate as N", "Nitrate-N", "Nitrate-nitrogen",
                                   "Nitrate-Nitrogen")},
    **{n: {_ANY: "no2"} for n in ("Nitrite", "Nitrite as N", "Nitrite-N")},
    **{n: {_ANY: "no3no2"} for n in (
        "Inorganic nitrogen (nitrate and nitrite)", "Inorganic nitrogen (nitrate and nitrite) as N",
        "Nitrate + Nitrite", "Nitrate-Nitrite",
        "Inorganic nitrogen (nitrate and nitrite) ***retired***use Nitrate + Nitrite")},
    **{n: {_ANY: "nh3"} for n in (
        "Ammonia", "Ammonia and ammonium", "Ammonia as NH3", "Ammonia-nitrogen",
        "Ammonia-nitrogen as N", "Ammonium", "Ammonium as N", "Ammonium as NH4",
        "Ammonium-nitrogen", "Ammonium-Nitrogen", "Nitrogen, ammonia (NH3) as NH4")},
    **{n: {"total": "tkn", "unspecified": "tkn"} for n in (
        "Kjeldahl nitrogen", "Total Kjeldahl nitrogen", "Total Kjeldahl nitrogen (Organic N & NH3)",
        "Nitrogen Kjeldahl")},
    # carbon: an unstated fraction is ambiguous between DOC and TOC -> unmapped
    "Organic carbon": {"dissolved": "doc", "total": "toc"},
    "Non-purgeable Organic Carbon (NPOC)": {"dissolved": "doc", "total": "toc"},
    # ions
    "Chloride": {_ANY: "chloride"},
    **{n: {_ANY: "sulfate"} for n in ("Sulfate", "Sulfate as SO4", "Sulfate as S", "Total Sulfate")},
    **{n: {_ANY: "spcond"} for n in ("Specific conductance", "Specific conductivity",
                                      "Specific conductivity***retired***use Specific conductance")},
    "Conductivity": {_ANY: "@conductivity"},
    # bacteria
    "Escherichia coli": {_ANY: "@ecoli"}, "Escherichia": None,
    "Enterococcus": {_ANY: "@enterococcus"}, "Enterococci": {_ANY: "@enterococcus"},
    "Fecal Coliform": {_ANY: "fecal_coliform"},
    # cyanotoxins
    **{n: {"dissolved": "microcystins_dissolved", "particulate": "microcystins_particulate",
           _ANY: "microcystins"} for n in (
        "Microcystin", "Microcystins and nodularins", "Total microcystins plus nodularins",
        "Microcystins/nodularin")},
    "Dissolved microcystins plus nodularins": {_ANY: "microcystins_dissolved"},
    "Particulate microcystins/nodularin": {_ANY: "microcystins_particulate"},
    "Microcystin LR": {_ANY: "microcystin_lr"}, "Microcystin RR": {_ANY: "microcystin_rr"},
    "Microcystin YR": {_ANY: "microcystin_yr"}, "Microcystin LA": {_ANY: "microcystin_la"},
    **{n: {_ANY: "microcystin_other_congener"} for n in (
        "Microcystin LF", "Microcystin LW", "Microcystin LY", "Microcystin WR", "Microcystin HilR",
        "Microcystin HiLR", "Microcystin HtyR", "Microcystin HtYR", "Microcystin dLR",
        "Microcystin dRR", "Microcystin LR [DAsp3]", "Microcystin LR, Desmethyl",
        "Microcystin LR [Dha7]", "Microcystin LR [D-Leu1]", "Microcystin RR [DAsp3]")},
    "Cylindrospermopsin": {_ANY: "cylindrospermopsin"}, "Cylindrospermopsins": {_ANY: "cylindrospermopsin"},
    "Anatoxin-A": {_ANY: "anatoxin_a"}, "Anatoxin a": {_ANY: "anatoxin_a"}, "Anatoxin": {_ANY: "anatoxin_a"},
    "Saxitoxin": {_ANY: "saxitoxin"}, "Saxitoxins": {_ANY: "saxitoxin"},
    # physchem
    **{n: {_ANY: "@do"} for n in ("Dissolved oxygen (DO)", "Dissolved oxygen",
                                   "Dissolved Oxygen***retired***use Dissolved Oxygen (DO)")},
    "pH": {_ANY: "ph"}, "PH": {_ANY: "ph"},
    "Temperature, water": {_ANY: "temp"},
}

# Speciation implied by the characteristic name itself.
NAME_SPECIES = {
    "Nitrate as N": "N", "Nitrate-N": "N", "Nitrate-nitrogen": "N", "Nitrate-Nitrogen": "N",
    "Nitrite as N": "N", "Nitrite-N": "N", "Inorganic nitrogen (nitrate and nitrite) as N": "N",
    "Ammonia as NH3": "NH3", "Ammonia-nitrogen": "N", "Ammonia-nitrogen as N": "N",
    "Ammonium as N": "N", "Ammonium as NH4": "NH4", "Ammonium-nitrogen": "N",
    "Ammonium-Nitrogen": "N", "Nitrogen, ammonia (NH3) as NH4": "NH4",
    "Orthophosphate as P": "P", "Orthophosphate as PO4": "PO4", "Ortho-Phosphate-Phosphorus": "P",
    "Phosphate-phosphorus": "P", "Phosphate-phosphorus as P": "P", "Phosphate-phosphorus as PO4": "PO4",
    "Phosphorus as P": "P", "Sulfate as SO4": "SO4", "Sulfate as S": "S",
}

# ------------------------------------------------------------------------------------------------
# Fraction classes
# ------------------------------------------------------------------------------------------------
_FRACTION_CLASS = {
    "total": "total", "unfiltered": "total", "total recoverable": "total", "recoverable": "total",
    "unfiltered, field": "total", "unfiltered, lab": "total", "whole": "total",
    "total residual": "total",
    "dissolved": "dissolved", "filtered, lab": "dissolved", "filtered, field": "dissolved",
    "filtered field and/or lab": "dissolved", "filterable": "dissolved", "field filtered": "dissolved",
    "filtered": "dissolved",
    "suspended": "particulate", "non-filterable (particle)": "particulate", "particle": "particulate",
    "particulate": "particulate", "non-filterable": "particulate", "bed sediment": "particulate",
    "": "unspecified", "none": "unspecified", "not applicable": "unspecified", "n/a": "unspecified",
}


def fraction_class(f: pd.Series) -> pd.Series:
    s = f.fillna("").astype(str).str.strip().str.lower()
    return s.map(_FRACTION_CLASS).fillna("other")


_SPEC_COL = {"as n": "N", "as no3": "NO3", "as no2": "NO2", "as nh3": "NH3", "as nh4": "NH4",
             "as p": "P", "as po4": "PO4", "as so4": "SO4", "as s": "S", "as c": "C", "as cl": "Cl",
             "": "", "none": "", "unknown": "", "not applicable": ""}


def speciation_token(s: pd.Series) -> pd.Series:
    """Result_MethodSpeciation -> species token ('' = not stated, '?' = unrecognised)."""
    x = s.fillna("").astype(str).str.strip().str.lower()
    return x.map(_SPEC_COL).fillna("?")


_QPCR_METHOD = re.compile(r"qpcr|q-pcr|\bpcr\b|1609|1611|1696|ddpcr|draft 1609", re.I)


def is_qpcr_method(method_id: pd.Series, method_name: pd.Series) -> pd.Series:
    m = method_id.fillna("").astype(str) + " " + method_name.fillna("").astype(str)
    return m.str.contains(_QPCR_METHOD)


# ------------------------------------------------------------------------------------------------
# Per-combination mapping
# ------------------------------------------------------------------------------------------------
def _convert(var: Var, ubase: str, species: str) -> tuple[float, float, str]:
    """Return (factor, offset, reason) to convert a value in ``ubase`` (normalised, species
    already split off) to ``var``'s canonical unit. ``reason`` is '' on success."""
    d = var.dim
    if ubase in SENSOR_UNITS:
        return np.nan, 0.0, "sensor_fluorescence_unit"
    if d in ("mass_mg", "mass_ug"):
        scale = 1.0 if d == "mass_mg" else 1e3
        if ubase in MASS_TO_MG_L:
            f = MASS_TO_MG_L[ubase] * scale
            if var.basis:
                f *= species_factor(species, var.basis)
            return f, 0.0, ""
        if ubase in MOLAR_TO_MOL_L and var.basis:
            # one basis atom per formula unit for every supported species
            return MOLAR_TO_MOL_L[ubase] * MOLAR[var.basis] * 1e3 * scale, 0.0, ""
        if ubase in EQUIV_TO_EQ_L and var.name in VAR_CHARGE:
            return EQUIV_TO_EQ_L[ubase] / VAR_CHARGE[var.name] * MOLAR[var.basis] * 1e3 * scale, 0.0, ""
        return np.nan, 0.0, "unit"
    if d == "length_m":
        return (LENGTH_TO_M[ubase], 0.0, "") if ubase in LENGTH_TO_M else (np.nan, 0.0, "unit")
    if d == "length_cm":
        return (LENGTH_TO_M[ubase] * 100, 0.0, "") if ubase in LENGTH_TO_M else (np.nan, 0.0, "unit")
    if d == "turb":
        return (1.0, 0.0, "") if TURBIDITY_UNITS.get(ubase) == var.name else (np.nan, 0.0, "unit")
    if d == "cond":
        return (CONDUCTANCE_TO_US_CM[ubase], 0.0, "") if ubase in CONDUCTANCE_TO_US_CM else (np.nan, 0.0, "unit")
    if d == "culture":
        return (CULTURE_TO_PER_100ML[ubase], 0.0, "") if ubase in CULTURE_TO_PER_100ML else (np.nan, 0.0, "unit")
    if d == "qpcr":
        if ubase in QPCR_TO_CCE_100ML:
            return QPCR_TO_CCE_100ML[ubase], 0.0, ""
        return np.nan, 0.0, "qpcr_gene_copies_unit" if ubase in QPCR_OTHER else "unit"
    if d == "temp":
        if ubase in TEMP_UNITS:
            _, f, off = TEMP_UNITS[ubase]
            return f, off, ""
        return np.nan, 0.0, "unit"
    if d == "ph":
        return (1.0, 0.0, "") if ubase in PH_UNITS else (np.nan, 0.0, "unit")
    if d == "percent":
        return (1.0, 0.0, "") if ubase in PERCENT_UNITS else (np.nan, 0.0, "unit")
    if d == "absorb":
        return (ABSORBANCE_TO_M1[ubase], 0.0, "") if ubase in ABSORBANCE_TO_M1 else (np.nan, 0.0, "unit")
    return np.nan, 0.0, "unit"


def _route_special(tag: str, ubase: str, qpcr: bool, temp_basis: str, tc_unit: bool) -> tuple[str, str]:
    """Resolve '@' routes that depend on the unit/method. Returns (variable, reason)."""
    if tag == "@turbidity":
        v = TURBIDITY_UNITS.get(ubase)
        return (v, "") if v else ("", "turbidity_unit_not_ntu_fnu_ntru")
    if tag in ("@ecoli", "@enterococcus"):
        base = "ecoli" if tag == "@ecoli" else "enterococcus"
        if ubase in QPCR_TO_CCE_100ML or ubase in QPCR_OTHER:
            return base + "_qpcr", ""
        if qpcr:  # qPCR method with a culture-looking unit: never let it into the culture variable
            return "", "qpcr_method_with_culture_unit"
        return base, ""
    if tag == "@do":
        return ("do_pctsat", "") if ubase in PERCENT_UNITS else ("do", "")
    if tag == "@conductivity":
        if tc_unit or "25" in temp_basis:
            return "spcond", ""
        return "", "conductivity_not_temperature_compensated"
    if tag == "@aux_secchi_bottom":
        return "", "aux_secchi_visible_at_bottom"
    return "", "route"


def map_combos(combos: pd.DataFrame) -> pd.DataFrame:
    """Map unique (characteristic, fraction, speciation, unit, qpcr, temperature_basis) rows.

    Adds: variable, unmapped_reason, factor, offset, unit_canonical, species_used,
    speciation_assumed, fraction_class.
    """
    out = combos.copy()
    out["fraction_class"] = fraction_class(out["sample_fraction"])
    out["spec_col"] = speciation_token(out["speciation"])
    res = []
    for r in out.itertuples(index=False):
        route = CHAR_ROUTES.get(r.characteristic, "missing")
        un = norm_unit(r.unit)
        ubase, uspec, tc = split_unit(un)
        rec = {"variable": "", "unmapped_reason": "", "factor": np.nan, "offset": 0.0,
               "unit_canonical": "", "species_used": "", "speciation_assumed": False}
        if route == "missing":
            rec["unmapped_reason"] = "characteristic_not_in_routes"
            res.append(rec)
            continue
        if route is None:
            rec["unmapped_reason"] = "ambiguous_characteristic"
            res.append(rec)
            continue
        target = route.get(r.fraction_class, route.get(_ANY))
        if target is None:
            rec["unmapped_reason"] = f"fraction_{r.fraction_class}"
            res.append(rec)
            continue
        if target.startswith("@"):
            target, reason = _route_special(target, ubase, bool(r.qpcr), str(r.temperature_basis), tc)
            if not target:
                rec["unmapped_reason"] = reason
                res.append(rec)
                continue
        var = VARIABLES[target]
        if un == "" and var.dim not in ("ph",):
            rec.update(variable=target, unmapped_reason="unit_missing")
            res.append(rec)
            continue
        species, assumed = "", False
        if var.basis:
            cands = {s for s in (uspec, r.spec_col if r.spec_col not in ("", "?") else "",
                                 NAME_SPECIES.get(r.characteristic, "")) if s}
            if len(cands) > 1:
                rec.update(variable=target, unmapped_reason="speciation_conflict")
                res.append(rec)
                continue
            if r.spec_col == "?" and not cands and var.assume_flag:
                rec.update(variable=target, unmapped_reason="speciation_unrecognised")
                res.append(rec)
                continue
            species = cands.pop() if cands else var.basis
            assumed = not (uspec or (r.spec_col not in ("", "?")) or r.characteristic in NAME_SPECIES)
            if species not in var.species:
                rec.update(variable=target, unmapped_reason=f"speciation_{species}_invalid")
                res.append(rec)
                continue
        f, off, reason = _convert(var, ubase, species or var.basis)
        if reason:
            rec.update(variable=target, unmapped_reason=reason)
            res.append(rec)
            continue
        rec.update(variable=target, factor=f, offset=off, unit_canonical=var.unit,
                   species_used=species, speciation_assumed=bool(assumed and var.assume_flag))
        res.append(rec)
    res = pd.DataFrame(res, index=out.index)
    return pd.concat([out, res], axis=1)


COMBO_COLS = ["characteristic", "sample_fraction", "speciation", "unit", "qpcr", "temperature_basis"]


def harmonize(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map raw parsed rows to variables and canonical units.

    Returns ``(mapped, unmapped)``. ``mapped`` has ``variable``, canonical ``value`` /
    ``detection_limit`` / ``unit``, plus ``unit_original``, ``value_original``,
    ``fraction_class``, ``species_used`` and ``speciation_assumed``. ``unmapped`` keeps the raw
    columns plus ``variable`` (if known) and ``unmapped_reason``.
    """
    if df.empty:
        return df.assign(variable=pd.Series(dtype=object)), df.assign(unmapped_reason=pd.Series(dtype=object))
    d = df.copy()
    # Non-detects often leave Result_MeasureUnit blank and state the unit only with the limit.
    blank_unit = d["unit"].fillna("").astype(str).str.strip() == ""
    d["unit"] = d["unit"].where(~(blank_unit & d["censored"].astype(bool)), d["detection_limit_unit"])
    d["qpcr"] = is_qpcr_method(d["method_id"], d["method_name"])
    d["temperature_basis"] = d["temperature_basis"].fillna("").astype(str)
    combos = d[COMBO_COLS].drop_duplicates().reset_index(drop=True)
    mc = map_combos(combos)
    d = d.merge(mc, on=COMBO_COLS, how="left", validate="many_to_one")
    empty = (d["unmapped_reason"] == "unit_missing") & d["value"].isna() & ~d["censored"].astype(bool)
    d.loc[empty, "unmapped_reason"] = "no_value_and_no_unit"
    ok = (d["unmapped_reason"] == "") & d["variable"].ne("")
    unmapped = d[~ok].drop(columns=["factor", "offset", "unit_canonical", "species_used",
                                    "speciation_assumed", "spec_col"])
    m = d[ok].copy()
    m["unit_original"] = m["unit"]
    m["value_original"] = m["value"]
    m["value"] = m["value"] * m["factor"] + m["offset"]
    # detection limits: convert with their own unit when stated (same variable/species)
    m["detection_limit"] = _convert_dl(m)
    m["unit"] = m["unit_canonical"]
    m = m.drop(columns=["factor", "offset", "unit_canonical", "spec_col"])
    return m.reset_index(drop=True), unmapped.reset_index(drop=True)


def _convert_dl(m: pd.DataFrame) -> pd.Series:
    dl = m["detection_limit"].astype("float64")
    du = m["detection_limit_unit"].fillna("").astype(str)
    same = (du == "") | (du == m["unit_original"])
    out = pd.Series(np.where(same, dl * m["factor"] + m["offset"], np.nan), index=m.index)
    other = ~same & dl.notna()
    if other.any():
        keys = pd.DataFrame({"variable": m.loc[other, "variable"], "du": du[other],
                             "species": m.loc[other, "species_used"]}).drop_duplicates()
        fac = {}
        for r in keys.itertuples(index=False):
            ubase, uspec, _ = split_unit(norm_unit(r.du))
            var = VARIABLES[r.variable]
            sp = uspec or r.species or var.basis
            if var.basis and sp not in var.species:
                fac[(r.variable, r.du, r.species)] = (np.nan, 0.0)
                continue
            f, off, reason = _convert(var, ubase, sp)
            fac[(r.variable, r.du, r.species)] = (np.nan, 0.0) if reason else (f, off)
        idx = m.index[other]
        ff = np.array([fac[(v, u, s)][0] for v, u, s in zip(m.loc[idx, "variable"], du[idx],
                                                            m.loc[idx, "species_used"])])
        oo = np.array([fac[(v, u, s)][1] for v, u, s in zip(m.loc[idx, "variable"], du[idx],
                                                            m.loc[idx, "species_used"])])
        out.loc[idx] = dl[idx].to_numpy() * ff + oo
    return out


# Depth units -> metres for sample depth
def depth_m(value: pd.Series, unit: pd.Series) -> pd.Series:
    v = pd.to_numeric(value.fillna("").astype(str).str.strip(), errors="coerce")
    f = unit.fillna("").astype(str).map(lambda u: LENGTH_TO_M.get(norm_unit(u), np.nan))
    return v * f
