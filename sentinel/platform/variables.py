# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Variable naming, units and labels for the Console.

Variable names follow the SENTINEL-Sonde convention defined by USGS parameter code in
``sentinel.data.sources.usgs_legacy`` (e.g. 00300 -> ``do_mgl``). Units are part of a
variable's identity: the unit below is the one implied by the USGS parameter code, and two
codes that differ only by unit (chlorophyll RFU vs µg/L, turbidity FNU vs NTU) are different
variables that are never merged.
"""

from __future__ import annotations

from ..data.sources.usgs_legacy import ALL_PARAMS, HYDRO_PARAMS, STATS, WQ_PARAMS

CODE_BY_NAME = {name: code for code, name in ALL_PARAMS.items()}
DAILY_STATS = tuple(STATS.values())  # max, min, mean (USGS stat codes 00001-00003)
HOURLY_STATS = ("mean", "min", "max")  # hourly aggregates built by sonde_hourly.py

UNITS = {
    "temp_c": "°C",
    "do_mgl": "mg/L",
    "do_pctsat": "percent saturation",
    "spcond_uscm": "µS/cm at 25 °C",
    "ph": "standard units",
    "turb_fnu": "FNU",
    "turb_ntu": "NTU",
    "no3no2_mgl": "mg/L as N",
    "no3_mgl": "mg/L as N",
    "fdom_qse": "µg/L as QSE",
    "chl_rfu": "RFU",
    "chl_ugl": "µg/L",
    "pc_ugl": "µg/L",
    "pc_rfu": "RFU",
    "salinity_ppt": "ppt",
    "discharge_cfs": "ft³/s",
    "gage_ft": "ft",
    "discharge_tf_cfs": "ft³/s",
}

LABELS = {
    "temp_c": "Water temperature",
    "do_mgl": "Dissolved oxygen",
    "do_pctsat": "Dissolved oxygen, saturation",
    "spcond_uscm": "Specific conductance",
    "ph": "pH",
    "turb_fnu": "Turbidity (FNU)",
    "turb_ntu": "Turbidity (NTU)",
    "no3no2_mgl": "Nitrate plus nitrite",
    "no3_mgl": "Nitrate",
    "fdom_qse": "fDOM",
    "chl_rfu": "Chlorophyll fluorescence (RFU)",
    "chl_ugl": "Chlorophyll fluorescence (µg/L)",
    "pc_ugl": "Phycocyanin fluorescence (µg/L)",
    "pc_rfu": "Phycocyanin fluorescence (RFU)",
    "salinity_ppt": "Salinity",
    "discharge_cfs": "Discharge",
    "gage_ft": "Gage height",
    "discharge_tf_cfs": "Discharge, tidally filtered",
}

# USGS site type codes (NWIS site_tp_cd) -> label.
SITE_TYPES = {
    "ST": "Stream",
    "ST-CA": "Canal",
    "ST-DCH": "Ditch",
    "ST-TS": "Tidal stream",
    "LK": "Lake, reservoir, impoundment",
    "ES": "Estuary",
    "OC": "Ocean",
    "OC-CO": "Coastal",
    "GW": "Well",
    "SP": "Spring",
    "WE": "Wetland",
    "AT": "Atmosphere",
}

WQ_NAMES = tuple(WQ_PARAMS.values())
HYDRO_NAMES = tuple(HYDRO_PARAMS.values())

# Column names a user might upload that do not identify a unit; each maps to the
# unit-specific names that do.
AMBIGUOUS = {
    "temperature": ["temp_c"],
    "temp": ["temp_c"],
    "water_temp": ["temp_c"],
    "do": ["do_mgl", "do_pctsat"],
    "oxygen": ["do_mgl", "do_pctsat"],
    "dissolved_oxygen": ["do_mgl", "do_pctsat"],
    "conductivity": ["spcond_uscm"],
    "conductance": ["spcond_uscm"],
    "sc": ["spcond_uscm"],
    "turbidity": ["turb_fnu", "turb_ntu"],
    "turb": ["turb_fnu", "turb_ntu"],
    "nitrate": ["no3no2_mgl", "no3_mgl"],
    "no3": ["no3no2_mgl", "no3_mgl"],
    "chlorophyll": ["chl_ugl", "chl_rfu"],
    "chl": ["chl_ugl", "chl_rfu"],
    "chla": ["chl_ugl", "chl_rfu"],
    "phycocyanin": ["pc_ugl", "pc_rfu"],
    "fdom": ["fdom_qse"],
    "discharge": ["discharge_cfs"],
    "flow": ["discharge_cfs"],
    "stage": ["gage_ft"],
    "salinity": ["salinity_ppt"],
}


def split_column(col: str, stats: tuple[str, ...] = DAILY_STATS) -> tuple[str, str | None]:
    """``do_mgl_min`` -> (``do_mgl``, ``min``); ``do_mgl`` -> (``do_mgl``, None)."""
    for st in stats:
        suffix = f"_{st}"
        if col.endswith(suffix) and col[: -len(suffix)] in CODE_BY_NAME:
            return col[: -len(suffix)], st
    return col, None


def describe(col: str) -> dict:
    """Metadata for a value column (base variable + optional statistic)."""
    base, stat = split_column(col)
    label = LABELS.get(base, base)
    return {
        "name": col,
        "base": base,
        "stat": stat,
        "label": f"{label}, daily {stat}" if stat else label,
        "unit": UNITS.get(base),
        "parameter_code": CODE_BY_NAME.get(base),
        "kind": "water_quality" if base in WQ_NAMES else ("hydrologic" if base in HYDRO_NAMES else "unknown"),
    }


def describe_hourly(col: str) -> dict:
    base, stat = split_column(col, HOURLY_STATS)
    d = describe(base)
    d.update(
        {
            "name": col,
            "stat": stat,
            "label": f"{LABELS.get(base, base)}, hourly {stat}" if stat else LABELS.get(base, base),
        }
    )
    return d


def base_label(base: str) -> str:
    unit = UNITS.get(base)
    return f"{LABELS.get(base, base)} ({unit})" if unit else LABELS.get(base, base)


def site_type_label(code: str | None) -> str | None:
    if not code:
        return None
    return SITE_TYPES.get(code) or SITE_TYPES.get(code.split("-")[0]) or code
