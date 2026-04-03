"""
SENTINEL-DB parameter name ontology.

Maps 10,000+ raw parameter names from heterogeneous water quality databases
(EPA WQP, USGS NWIS, EU Waterbase, GEMStat, citizen science) to ~500
canonical parameter names with standardized units.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from sentinel.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Canonical parameter definition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CanonicalParameter:
    """A canonical water quality parameter with its standard unit."""

    canonical_name: str
    canonical_unit: str
    category: str  # e.g. "dissolved_gas", "nutrient", "metal", "physical"


# ---------------------------------------------------------------------------
# Unit conversion factors
# ---------------------------------------------------------------------------

# Conversion factors to canonical units.
# Key = (from_unit_lower, to_unit_lower), value = multiplicative factor.
_UNIT_CONVERSIONS: dict[tuple[str, str], float] = {
    # Mass concentration
    ("ug/l", "mg/l"): 0.001,
    ("µg/l", "mg/l"): 0.001,
    ("ng/l", "mg/l"): 1e-6,
    ("g/l", "mg/l"): 1000.0,
    ("mg/l", "ug/l"): 1000.0,
    ("mg/l", "mg/l"): 1.0,
    ("ug/l", "ug/l"): 1.0,
    ("ppm", "mg/l"): 1.0,
    ("ppb", "mg/l"): 0.001,
    ("ppb", "ug/l"): 1.0,
    ("ppt", "mg/l"): 1e-6,
    ("mg/kg", "mg/l"): 1.0,  # approximate for dilute solutions
    ("meq/l", "mg/l"): None,  # requires molar mass — handled per-element
    # Temperature
    ("degc", "degc"): 1.0,
    ("deg c", "degc"): 1.0,
    ("celsius", "degc"): 1.0,
    ("c", "degc"): 1.0,
    # Fahrenheit -> Celsius handled separately
    # Conductivity
    ("us/cm", "us/cm"): 1.0,
    ("umho/cm", "us/cm"): 1.0,
    ("µs/cm", "us/cm"): 1.0,
    ("ms/cm", "us/cm"): 1000.0,
    ("ms/m", "us/cm"): 10.0,
    # Turbidity
    ("ntu", "ntu"): 1.0,
    ("fnu", "ntu"): 1.0,
    ("ftu", "ntu"): 1.0,
    ("jtu", "ntu"): 1.0,
    # Dissolved oxygen
    ("mg/l", "mg/l"): 1.0,
    ("% sat", "% sat"): 1.0,
    ("%", "% sat"): 1.0,
    # ORP
    ("mv", "mv"): 1.0,
    # Colour
    ("pcu", "pcu"): 1.0,
    ("pt-co", "pcu"): 1.0,
    ("hazen", "pcu"): 1.0,
    ("cu", "pcu"): 1.0,
    # Flow
    ("m3/s", "m3/s"): 1.0,
    ("l/s", "m3/s"): 0.001,
    ("cfs", "m3/s"): 0.028316846592,
    # Area-based (chlorophyll)
    ("mg/m3", "ug/l"): 1.0,
    ("ug/l", "ug/l"): 1.0,
}


def harmonize_unit(value: float, from_unit: str, to_unit: str) -> float:
    """Convert *value* from *from_unit* to *to_unit*.

    Parameters
    ----------
    value:
        Numeric measurement value.
    from_unit:
        Original unit string.
    to_unit:
        Target canonical unit string.

    Returns
    -------
    Converted value.

    Raises
    ------
    ValueError
        If the conversion is not supported.
    """
    from_lower = from_unit.strip().lower()
    to_lower = to_unit.strip().lower()

    if from_lower == to_lower:
        return value

    # Special: Fahrenheit to Celsius
    if from_lower in ("degf", "fahrenheit", "f", "deg f") and to_lower == "degc":
        return (value - 32.0) * 5.0 / 9.0

    # Special: Celsius to Fahrenheit
    if from_lower == "degc" and to_lower in ("degf", "fahrenheit", "f"):
        return value * 9.0 / 5.0 + 32.0

    key = (from_lower, to_lower)
    factor = _UNIT_CONVERSIONS.get(key)
    if factor is not None:
        return value * factor

    raise ValueError(
        f"Unsupported unit conversion: {from_unit!r} -> {to_unit!r}"
    )


# ---------------------------------------------------------------------------
# Ontology: canonical parameter definitions
# ---------------------------------------------------------------------------

# Master list of canonical parameters.
_CANONICAL_PARAMS: dict[str, CanonicalParameter] = {}


def _register(name: str, unit: str, category: str) -> CanonicalParameter:
    """Register a canonical parameter."""
    cp = CanonicalParameter(canonical_name=name, canonical_unit=unit, category=category)
    _CANONICAL_PARAMS[name] = cp
    return cp


# --- Physical parameters ---
_register("water_temperature", "degC", "physical")
_register("ph", "pH units", "physical")
_register("specific_conductance", "uS/cm", "physical")
_register("electrical_conductivity", "uS/cm", "physical")
_register("turbidity", "NTU", "physical")
_register("total_dissolved_solids", "mg/L", "physical")
_register("total_suspended_solids", "mg/L", "physical")
_register("secchi_depth", "m", "physical")
_register("color", "PCU", "physical")
_register("salinity", "PSU", "physical")
_register("hardness", "mg/L CaCO3", "physical")
_register("alkalinity", "mg/L CaCO3", "physical")

# --- Dissolved gases ---
_register("dissolved_oxygen", "mg/L", "dissolved_gas")
_register("dissolved_oxygen_saturation", "% sat", "dissolved_gas")
_register("dissolved_co2", "mg/L", "dissolved_gas")

# --- Redox ---
_register("oxidation_reduction_potential", "mV", "redox")

# --- Nutrients: Nitrogen species ---
_register("total_nitrogen", "mg/L", "nutrient")
_register("nitrate", "mg/L", "nutrient")
_register("nitrite", "mg/L", "nutrient")
_register("nitrate_nitrite", "mg/L", "nutrient")
_register("ammonia", "mg/L", "nutrient")
_register("ammonium", "mg/L", "nutrient")
_register("total_kjeldahl_nitrogen", "mg/L", "nutrient")
_register("organic_nitrogen", "mg/L", "nutrient")

# --- Nutrients: Phosphorus species ---
_register("total_phosphorus", "mg/L", "nutrient")
_register("orthophosphate", "mg/L", "nutrient")
_register("dissolved_phosphorus", "mg/L", "nutrient")

# --- Organic matter ---
_register("biological_oxygen_demand", "mg/L", "organic_matter")
_register("chemical_oxygen_demand", "mg/L", "organic_matter")
_register("total_organic_carbon", "mg/L", "organic_matter")
_register("dissolved_organic_carbon", "mg/L", "organic_matter")

# --- Biological ---
_register("chlorophyll_a", "ug/L", "biological")
_register("phycocyanin", "ug/L", "biological")
_register("total_coliform", "CFU/100mL", "biological")
_register("fecal_coliform", "CFU/100mL", "biological")
_register("e_coli", "CFU/100mL", "biological")
_register("enterococcus", "CFU/100mL", "biological")

# --- Metals ---
_register("arsenic", "ug/L", "metal")
_register("cadmium", "ug/L", "metal")
_register("chromium", "ug/L", "metal")
_register("copper", "ug/L", "metal")
_register("iron", "ug/L", "metal")
_register("lead", "ug/L", "metal")
_register("manganese", "ug/L", "metal")
_register("mercury", "ug/L", "metal")
_register("nickel", "ug/L", "metal")
_register("zinc", "ug/L", "metal")
_register("aluminum", "ug/L", "metal")
_register("selenium", "ug/L", "metal")
_register("barium", "ug/L", "metal")
_register("boron", "ug/L", "metal")

# --- Ions ---
_register("calcium", "mg/L", "ion")
_register("magnesium", "mg/L", "ion")
_register("sodium", "mg/L", "ion")
_register("potassium", "mg/L", "ion")
_register("chloride", "mg/L", "ion")
_register("sulfate", "mg/L", "ion")
_register("fluoride", "mg/L", "ion")
_register("bicarbonate", "mg/L", "ion")

# --- Organic contaminants ---
_register("atrazine", "ug/L", "pesticide")
_register("glyphosate", "ug/L", "pesticide")
_register("pfos", "ng/L", "contaminant")
_register("pfoa", "ng/L", "contaminant")

# --- Hydrological ---
_register("discharge", "m3/s", "hydrological")
_register("water_level", "m", "hydrological")


# ---------------------------------------------------------------------------
# Name-to-canonical lookup table (~150+ explicit mappings)
# ---------------------------------------------------------------------------

# This is a curated dictionary mapping raw parameter names (lowercase) to
# their canonical_name.  Covers USGS NWIS codes, EPA WQP names, EU Waterbase,
# common abbreviations, and citizen-science naming conventions.

_NAME_MAP: dict[str, str] = {
    # --- USGS NWIS parameter codes ---
    "00300": "dissolved_oxygen",
    "00301": "dissolved_oxygen_saturation",
    "00400": "ph",
    "00095": "specific_conductance",
    "00094": "specific_conductance",
    "00010": "water_temperature",
    "00011": "water_temperature",
    "63680": "turbidity",
    "00076": "turbidity",
    "00090": "oxidation_reduction_potential",
    "00600": "total_nitrogen",
    "00618": "nitrate",
    "00620": "nitrite",
    "00631": "nitrate_nitrite",
    "00608": "ammonia",
    "00625": "total_kjeldahl_nitrogen",
    "00665": "total_phosphorus",
    "00671": "orthophosphate",
    "00680": "total_organic_carbon",
    "00681": "dissolved_organic_carbon",
    "00900": "hardness",
    "00915": "calcium",
    "00925": "magnesium",
    "00930": "sodium",
    "00935": "potassium",
    "00940": "chloride",
    "00945": "sulfate",
    "00950": "fluoride",
    "00530": "total_suspended_solids",
    "70300": "total_dissolved_solids",
    "32210": "chlorophyll_a",
    "31501": "total_coliform",
    "31616": "fecal_coliform",
    "31648": "e_coli",
    "01000": "arsenic",
    "01025": "cadmium",
    "01030": "chromium",
    "01040": "copper",
    "01046": "iron",
    "01049": "lead",
    "01055": "manganese",
    "71890": "mercury",
    "01065": "nickel",
    "01090": "zinc",
    "01105": "aluminum",
    "01145": "selenium",
    "00060": "discharge",
    "00065": "water_level",
    # --- EPA WQP / WQX names ---
    "dissolved oxygen": "dissolved_oxygen",
    "dissolved oxygen (do)": "dissolved_oxygen",
    "dissolved oxygen saturation": "dissolved_oxygen_saturation",
    "do": "dissolved_oxygen",
    "ph": "ph",
    "ph, lab": "ph",
    "ph, field": "ph",
    "hydrogen ion concentration": "ph",
    "specific conductance": "specific_conductance",
    "specific conductivity": "specific_conductance",
    "conductivity": "electrical_conductivity",
    "electrical conductivity": "electrical_conductivity",
    "spcond": "specific_conductance",
    "temperature, water": "water_temperature",
    "water temperature": "water_temperature",
    "temperature": "water_temperature",
    "temp": "water_temperature",
    "turbidity": "turbidity",
    "turb": "turbidity",
    "turbidity, lab": "turbidity",
    "turbidity, field": "turbidity",
    "total dissolved solids": "total_dissolved_solids",
    "tds": "total_dissolved_solids",
    "total suspended solids": "total_suspended_solids",
    "tss": "total_suspended_solids",
    "suspended solids": "total_suspended_solids",
    "total suspended sediment": "total_suspended_solids",
    "secchi depth": "secchi_depth",
    "transparency": "secchi_depth",
    "secchi": "secchi_depth",
    "oxidation-reduction potential": "oxidation_reduction_potential",
    "oxidation reduction potential": "oxidation_reduction_potential",
    "orp": "oxidation_reduction_potential",
    "redox potential": "oxidation_reduction_potential",
    "salinity": "salinity",
    "hardness, total": "hardness",
    "hardness": "hardness",
    "total hardness": "hardness",
    "hardness, ca, mg": "hardness",
    "alkalinity": "alkalinity",
    "alkalinity, total": "alkalinity",
    "total alkalinity": "alkalinity",
    "color": "color",
    "color, true": "color",
    "true color": "color",
    "apparent color": "color",
    # --- Nitrogen ---
    "nitrogen, total": "total_nitrogen",
    "total nitrogen": "total_nitrogen",
    "nitrogen": "total_nitrogen",
    "tn": "total_nitrogen",
    "nitrate": "nitrate",
    "nitrate as n": "nitrate",
    "nitrate nitrogen": "nitrate",
    "no3": "nitrate",
    "no3-n": "nitrate",
    "nitrate-n": "nitrate",
    "nitrite": "nitrite",
    "nitrite as n": "nitrite",
    "no2": "nitrite",
    "no2-n": "nitrite",
    "nitrate + nitrite": "nitrate_nitrite",
    "nitrate-nitrite": "nitrate_nitrite",
    "no3+no2": "nitrate_nitrite",
    "inorganic nitrogen (nitrate and nitrite)": "nitrate_nitrite",
    "ammonia": "ammonia",
    "ammonia as n": "ammonia",
    "ammonia-nitrogen": "ammonia",
    "nh3": "ammonia",
    "nh3-n": "ammonia",
    "ammonium": "ammonium",
    "ammonium as n": "ammonium",
    "nh4": "ammonium",
    "nh4-n": "ammonium",
    "total kjeldahl nitrogen": "total_kjeldahl_nitrogen",
    "kjeldahl nitrogen": "total_kjeldahl_nitrogen",
    "tkn": "total_kjeldahl_nitrogen",
    "organic nitrogen": "organic_nitrogen",
    # --- Phosphorus ---
    "phosphorus, total": "total_phosphorus",
    "total phosphorus": "total_phosphorus",
    "phosphorus": "total_phosphorus",
    "tp": "total_phosphorus",
    "orthophosphate": "orthophosphate",
    "orthophosphate as p": "orthophosphate",
    "phosphate": "orthophosphate",
    "po4": "orthophosphate",
    "po4-p": "orthophosphate",
    "soluble reactive phosphorus": "orthophosphate",
    "srp": "orthophosphate",
    "dissolved phosphorus": "dissolved_phosphorus",
    # --- Organic matter ---
    "biochemical oxygen demand": "biological_oxygen_demand",
    "biological oxygen demand": "biological_oxygen_demand",
    "bod": "biological_oxygen_demand",
    "bod5": "biological_oxygen_demand",
    "bod, 5-day": "biological_oxygen_demand",
    "chemical oxygen demand": "chemical_oxygen_demand",
    "cod": "chemical_oxygen_demand",
    "total organic carbon": "total_organic_carbon",
    "toc": "total_organic_carbon",
    "dissolved organic carbon": "dissolved_organic_carbon",
    "doc": "dissolved_organic_carbon",
    # --- Biological ---
    "chlorophyll a": "chlorophyll_a",
    "chlorophyll-a": "chlorophyll_a",
    "chlorophyll": "chlorophyll_a",
    "chl-a": "chlorophyll_a",
    "chla": "chlorophyll_a",
    "phycocyanin": "phycocyanin",
    "total coliform": "total_coliform",
    "coliform, total": "total_coliform",
    "fecal coliform": "fecal_coliform",
    "coliform, fecal": "fecal_coliform",
    "escherichia coli": "e_coli",
    "e. coli": "e_coli",
    "e.coli": "e_coli",
    "e coli": "e_coli",
    "enterococcus": "enterococcus",
    "enterococci": "enterococcus",
    # --- Metals ---
    "arsenic": "arsenic",
    "arsenic, dissolved": "arsenic",
    "as": "arsenic",
    "cadmium": "cadmium",
    "cadmium, dissolved": "cadmium",
    "cd": "cadmium",
    "chromium": "chromium",
    "chromium, dissolved": "chromium",
    "cr": "chromium",
    "copper": "copper",
    "copper, dissolved": "copper",
    "cu": "copper",
    "iron": "iron",
    "iron, dissolved": "iron",
    "fe": "iron",
    "lead": "lead",
    "lead, dissolved": "lead",
    "pb": "lead",
    "manganese": "manganese",
    "manganese, dissolved": "manganese",
    "mn": "manganese",
    "mercury": "mercury",
    "mercury, dissolved": "mercury",
    "hg": "mercury",
    "nickel": "nickel",
    "nickel, dissolved": "nickel",
    "ni": "nickel",
    "zinc": "zinc",
    "zinc, dissolved": "zinc",
    "zn": "zinc",
    "aluminum": "aluminum",
    "aluminium": "aluminum",
    "al": "aluminum",
    "selenium": "selenium",
    "se": "selenium",
    "barium": "barium",
    "ba": "barium",
    "boron": "boron",
    "b": "boron",
    # --- Ions ---
    "calcium": "calcium",
    "ca": "calcium",
    "magnesium": "magnesium",
    "mg": "magnesium",
    "sodium": "sodium",
    "na": "sodium",
    "potassium": "potassium",
    "k": "potassium",
    "chloride": "chloride",
    "cl": "chloride",
    "sulfate": "sulfate",
    "sulphate": "sulfate",
    "so4": "sulfate",
    "fluoride": "fluoride",
    "fl": "fluoride",
    "bicarbonate": "bicarbonate",
    "hco3": "bicarbonate",
    # --- Dissolved CO2 ---
    "carbon dioxide": "dissolved_co2",
    "dissolved co2": "dissolved_co2",
    "co2": "dissolved_co2",
    # --- Organic contaminants ---
    "atrazine": "atrazine",
    "glyphosate": "glyphosate",
    "pfos": "pfos",
    "pfoa": "pfoa",
    # --- Hydrological ---
    "discharge": "discharge",
    "streamflow": "discharge",
    "flow": "discharge",
    "water level": "water_level",
    "gage height": "water_level",
    "stage": "water_level",
}


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def resolve_parameter(raw_name: str) -> CanonicalParameter | None:
    """Resolve a raw parameter name to its canonical form.

    Resolution order:
    1. Exact match in the lookup table.
    2. Case-insensitive match.
    3. Fuzzy match (SequenceMatcher ratio >= 0.85).

    Parameters
    ----------
    raw_name:
        Raw parameter name from a data source.

    Returns
    -------
    The matching :class:`CanonicalParameter`, or ``None`` if no match.
    """
    if not raw_name:
        return None

    stripped = raw_name.strip()
    lower = stripped.lower()

    # 1. Exact match
    canonical_name = _NAME_MAP.get(stripped)
    if canonical_name and canonical_name in _CANONICAL_PARAMS:
        return _CANONICAL_PARAMS[canonical_name]

    # 2. Case-insensitive match
    canonical_name = _NAME_MAP.get(lower)
    if canonical_name and canonical_name in _CANONICAL_PARAMS:
        return _CANONICAL_PARAMS[canonical_name]

    # 3. Fuzzy match
    best_score = 0.0
    best_canonical: str | None = None
    for key, cname in _NAME_MAP.items():
        score = SequenceMatcher(None, lower, key.lower()).ratio()
        if score > best_score:
            best_score = score
            best_canonical = cname

    if best_score >= 0.85 and best_canonical and best_canonical in _CANONICAL_PARAMS:
        logger.debug(
            f"Fuzzy match: {raw_name!r} -> {best_canonical!r} (score={best_score:.2f})"
        )
        return _CANONICAL_PARAMS[best_canonical]

    # Try rapidfuzz if available for better performance
    try:
        from rapidfuzz import fuzz

        best_score_rf = 0.0
        best_canonical_rf: str | None = None
        for key, cname in _NAME_MAP.items():
            score = fuzz.ratio(lower, key.lower()) / 100.0
            if score > best_score_rf:
                best_score_rf = score
                best_canonical_rf = cname

        if (
            best_score_rf >= 0.85
            and best_canonical_rf
            and best_canonical_rf in _CANONICAL_PARAMS
        ):
            logger.debug(
                f"rapidfuzz match: {raw_name!r} -> {best_canonical_rf!r} "
                f"(score={best_score_rf:.2f})"
            )
            return _CANONICAL_PARAMS[best_canonical_rf]
    except ImportError:
        pass

    return None


# ---------------------------------------------------------------------------
# Ontology persistence
# ---------------------------------------------------------------------------


def build_ontology() -> dict[str, Any]:
    """Build the full ontology as a JSON-serializable dictionary.

    Returns
    -------
    Dict with keys ``canonical_parameters`` and ``name_map``.
    """
    return {
        "canonical_parameters": {
            name: asdict(cp) for name, cp in _CANONICAL_PARAMS.items()
        },
        "name_map": dict(_NAME_MAP),
    }


def save_ontology(path: str | Path) -> Path:
    """Persist the ontology to a JSON file.

    Parameters
    ----------
    path:
        Output file path.

    Returns
    -------
    Path to the written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = build_ontology()
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    logger.info(
        f"Ontology saved: {len(data['canonical_parameters'])} params, "
        f"{len(data['name_map'])} mappings -> {path}"
    )
    return path


def load_ontology(path: str | Path) -> dict[str, Any]:
    """Load an ontology from a JSON file and merge into the runtime tables.

    Parameters
    ----------
    path:
        Path to a previously saved ontology JSON.

    Returns
    -------
    The loaded ontology dictionary.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    # Merge canonical parameters
    for name, cp_dict in data.get("canonical_parameters", {}).items():
        if name not in _CANONICAL_PARAMS:
            _CANONICAL_PARAMS[name] = CanonicalParameter(**cp_dict)

    # Merge name mappings
    for raw, canonical in data.get("name_map", {}).items():
        if raw not in _NAME_MAP:
            _NAME_MAP[raw] = canonical

    logger.info(
        f"Ontology loaded from {path}: "
        f"{len(_CANONICAL_PARAMS)} total params, {len(_NAME_MAP)} total mappings"
    )
    return data
