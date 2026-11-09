# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""USCG National Response Center (NRC) FOIA incident files -> tidy report and material tables.

Source: https://nrc.uscg.mil/ (FOIA files ``FOIAFiles/CY{YY}.xlsx``, one per calendar year from
1990; ``DataDictionary.xlsx``). The homepage states that these are INITIAL, unvalidated reports.
Each workbook has ten sheets keyed by ``SEQNOS`` (the NRC report number): CALLS,
INCIDENT_COMMONS, INCIDENT_DETAILS, INCIDENTS, MATERIAL_INVOLVED, MATERIAL_INV0LVED_CR (sic),
TRAINS_DETAIL, DERAILED_UNITS, VESSELS_DETAIL, MOBILE_DETAILS.

Missing-value rules applied at parse time (original kept in a ``*_qualifier``/``*_raw`` column):
  * CAS ``000000-00-0`` (and other all-zero/invalid CAS) -> missing; check digit verified.
  * amount ``0`` with unit ``UNKNOWN AMOUNT`` -> missing amount, qualifier ``UNKNOWN AMOUNT``.
  * Y/N/U flags -> nullable boolean (U = unknown -> <NA>), raw code kept.
  * coordinates: degrees/minutes/seconds + quadrant -> signed decimal degrees when degrees are
    present and every part is in range (deg <= 90/180, min/sec < 60; blank min/sec count as 0);
    0/0/0 -> missing; a blank quadrant takes the state's hemisphere (flag ``quadrant_inferred``);
    a wrong quadrant letter -> missing (``bad_quadrant``); points outside the US/territory extent
    keep ``lat_decoded``/``lon_decoded`` but get ``lat``/``lon`` = NaN (``coord_flag``). The raw
    D|M|S|Q strings are kept in ``lat_dms_raw``/``lon_dms_raw``.
Times: NRC gives local wall-clock times without a zone, so timestamps are stored as naive
``*_local`` values (not UTC) and the raw strings are kept. ``call_type == 'DRL'`` marks drills.
Caller/company names and street addresses are not carried into the tidy tables.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from sentinel.data.sources.chem_ids import normalize_cas

BASE_URL = "https://nrc.uscg.mil/FOIAFiles/"
DICTIONARY_URL = BASE_URL + "DataDictionary.xlsx"
LICENSE = "US Government work (no license stated)"
SHEETS = (
    "CALLS",
    "INCIDENT_COMMONS",
    "INCIDENT_DETAILS",
    "INCIDENTS",
    "MATERIAL_INVOLVED",
    "MATERIAL_INV0LVED_CR",
    "TRAINS_DETAIL",
    "DERAILED_UNITS",
    "VESSELS_DETAIL",
    "MOBILE_DETAILS",
)


def year_filename(year: int) -> str:
    return f"CY{year % 100:02d}.xlsx"


def year_url(year: int) -> str:
    return BASE_URL + year_filename(year)


def year_from_filename(name: str) -> int:
    yy = int(re.search(r"CY(\d{2})", name).group(1))
    return 1900 + yy if yy >= 90 else 2000 + yy


# --------------------------------------------------------------------------- workbook reading


def read_workbook(path: str | Path, sheets: Iterable[str] | None = None) -> dict[str, pd.DataFrame]:
    """All (or the named) sheets as string frames; column names upper-cased and stripped."""
    xl = pd.ExcelFile(path, engine="calamine")
    names = list(xl.sheet_names)
    want = (
        names if sheets is None else [n for n in names if _norm_sheet(n) in {_norm_sheet(s) for s in sheets}]
    )
    out = {}
    for n in want:
        df = xl.parse(n, dtype=object)
        df.columns = [str(c).strip().upper() for c in df.columns]
        out[_norm_sheet(n)] = df
    return out


def _norm_sheet(name: str) -> str:
    n = str(name).strip().upper().replace(" ", "_")
    return "MATERIAL_INV0LVED_CR" if n.replace("0", "O") == "MATERIAL_INVOLVED_CR" else n


# --------------------------------------------------------------------------- field cleaners


def _s(v) -> pd.Series:
    """Object series -> stripped strings with '' / 'nan' / None -> <NA>."""
    s = pd.Series(v, dtype="object")
    out = s.map(lambda x: None if x is None or (isinstance(x, float) and np.isnan(x)) else str(x).strip())
    out = out.where(~out.isin(["", "nan", "NaN", "NaT", "None"]), None)
    return out.astype("string")


def report_id(v) -> pd.Series:
    """SEQNOS as string without a trailing '.0' (Excel may store it as a float)."""
    s = _s(v)
    return s.str.replace(r"\.0+$", "", regex=True)


def zip_code(v) -> pd.Series:
    """ZIP codes stored as Excel numbers lose leading zeros (2134.0); restore 5-digit form."""
    s = _s(v).str.replace(r"\.0+$", "", regex=True)
    short = s.str.fullmatch(r"\d{3,4}").fillna(False)
    return s.where(~short, s.str.zfill(5))


def sheet_as_strings(df: pd.DataFrame, *, source_file: str) -> pd.DataFrame:
    """Archive form of any sheet: every value as a stripped string (datetimes ISO), SEQNOS cleaned."""
    out = pd.DataFrame({c: _s(df[c]) for c in df.columns if not str(c).upper().startswith("COLUMN")})
    if "SEQNOS" in out.columns:
        out["SEQNOS"] = report_id(out["SEQNOS"])
    out.insert(0, "source_file", source_file)
    return out


def yn_flag(v) -> pd.Series:
    """Y/YES/N/NO/U/UNKNOWN -> nullable boolean (unknown -> <NA>)."""
    s = _s(v).str.upper()
    m = {"Y": True, "YES": True, "N": False, "NO": False}
    return s.map(lambda x: m.get(x, pd.NA) if x is not pd.NA else pd.NA).astype("boolean")


def to_number(v) -> pd.Series:
    s = _s(v).str.replace(",", "", regex=False)
    return pd.to_numeric(s, errors="coerce").astype("float64")


def parse_local_datetime(v) -> pd.Series:
    """Excel datetimes or 'M/D/YYYY H:MM[:SS]' strings -> naive local timestamps (no zone)."""
    s = pd.Series(v, dtype="object")
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    is_ts = s.map(lambda x: isinstance(x, (pd.Timestamp, np.datetime64)) or hasattr(x, "year"))
    if is_ts.any():
        out[is_ts] = pd.to_datetime(s[is_ts], errors="coerce")
    rest = ~is_ts & s.notna()
    if rest.any():
        txt = _s(s[rest])
        parsed = pd.to_datetime(txt, format="%m/%d/%Y %H:%M", errors="coerce")
        for fmt in (
            "%m/%d/%Y %H:%M:%S",
            "%m/%d/%Y",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
            "%d-%b-%y %H:%M",
            "%d-%b-%y",
            "%m/%d/%y %H:%M",
        ):
            miss = parsed.isna() & txt.notna()
            if not miss.any():
                break
            parsed[miss] = pd.to_datetime(txt[miss], format=fmt, errors="coerce")
        out[rest] = parsed
    return out


def dms_to_decimal(deg, minutes, sec, quad, *, kind: str, default_quad=None) -> tuple[pd.Series, pd.Series]:
    """Degrees/minutes/seconds + quadrant -> (signed decimal degrees, flag).

    Flags: ``ok``; ``quadrant_inferred`` (quadrant blank, hemisphere taken from ``default_quad``,
    e.g. W/N for US states, E for Guam/CNMI, S for American Samoa); ``missing`` (no degrees);
    ``zero`` (0/0/0 placeholder); ``out_of_range`` (deg > 90/180, min/sec >= 60);
    ``bad_quadrant`` (quadrant is not a hemisphere letter of this axis, e.g. 'N' on a longitude).
    Only ``ok``/``quadrant_inferred`` rows get a value. Missing minutes/seconds count as 0 when
    degrees are present (reports often give whole degrees + minutes only).
    """
    d, m, s = to_number(deg), to_number(minutes), to_number(sec)
    q = _s(quad).str.upper().str.strip().str[:1]
    lim = 90.0 if kind == "lat" else 180.0
    neg_q, pos_q = ("S", "N") if kind == "lat" else ("W", "E")
    m0, s0 = m.fillna(0.0), s.fillna(0.0)
    val = d.abs() + m0 / 60.0 + s0 / 3600.0
    flag = pd.Series("ok", index=d.index, dtype="object")
    flag[d.isna().to_numpy()] = "missing"
    zero = (d.notna() & (d == 0) & (m0 == 0) & (s0 == 0)).to_numpy()
    flag[zero] = "zero"
    bad = (
        d.notna()
        & ~pd.Series(zero, index=d.index)
        & ((d.abs() > lim) | (m0 < 0) | (m0 >= 60) | (s0 < 0) | (s0 >= 60) | (val > lim))
    ).to_numpy()
    flag[bad] = "out_of_range"
    has_q = q.notna().to_numpy()
    badq = (flag == "ok").to_numpy() & has_q & ~q.isin([neg_q, pos_q]).fillna(False).to_numpy()
    flag[badq] = "bad_quadrant"
    q_eff = q.copy()
    noq = (flag == "ok").to_numpy() & ~has_q
    if default_quad is not None:
        dq = _s(default_quad).str.upper().str[:1]
        q_eff[noq] = dq[noq]
        flag[noq & dq.isin([neg_q, pos_q]).fillna(False).to_numpy()] = "quadrant_inferred"
        flag[noq & ~dq.isin([neg_q, pos_q]).fillna(False).to_numpy()] = "missing_quadrant"
    else:
        flag[noq] = "missing_quadrant"
    sign = np.where((q_eff == neg_q).fillna(False).to_numpy() | (d < 0).to_numpy(), -1.0, 1.0)
    keep = flag.isin(["ok", "quadrant_inferred"]).to_numpy()
    out = pd.Series(np.where(keep, sign * val.to_numpy(), np.nan), index=d.index, dtype="float64")
    return out, flag.astype("string")


#: (lat_min, lat_max, lon_min, lon_max) boxes covering the US, its territories and EEZ margins
US_EXTENT_BOXES = (
    (15.0, 72.0, -180.0, -64.0),  # CONUS, Alaska, Hawaii, PR/USVI, Gulf/Atlantic offshore
    (50.0, 72.0, 172.0, 180.0),  # western Aleutians
    (12.0, 22.0, 143.0, 147.0),  # Guam, Northern Mariana Islands
    (-15.5, -10.0, -172.0, -168.0),  # American Samoa
    (15.0, 30.0, 160.0, 180.0),
)  # Wake, Midway region


def in_us_extent(lat: pd.Series, lon: pd.Series) -> pd.Series:
    la, lo = lat.to_numpy(dtype=float), lon.to_numpy(dtype=float)
    ok = np.zeros(len(la), dtype=bool)
    for a, b, c, d in US_EXTENT_BOXES:
        ok |= (la >= a) & (la <= b) & (lo >= c) & (lo <= d)
    return pd.Series(ok, index=lat.index)


def default_quadrants(state) -> tuple[pd.Series, pd.Series]:
    """Hemisphere to assume when a quadrant is blank: N/W, except AS (S) and GU/MP (E)."""
    st = _s(state).str.upper()
    lat_q = pd.Series(np.where(st.eq("AS").fillna(False), "S", "N"), index=st.index)
    lon_q = pd.Series(np.where(st.isin(["GU", "MP"]).fillna(False), "E", "W"), index=st.index)
    return lat_q, lon_q


# Volume -> litres and mass -> kilograms (exact definitions; US customary units).
VOLUME_L = {
    "GALLON(S)": 3.785411784,
    "GALLONS": 3.785411784,
    "GALLON": 3.785411784,
    "BARREL(S)": 158.987294928,
    "QUART(S)": 0.946352946,
    "PINT(S)": 0.473176473,
    "CUP(S)": 0.2365882365,
    "OUNCE(S)": 0.0295735295625,
    "TABLESPOON(S)": 0.01478676478125,
    "TEASPOON(S)": 0.00492892159375,
    "LITER(S)": 1.0,
    "LITERS": 1.0,
    "MILLILITER(S)": 1e-3,
    "CUBIC METER(S)": 1000.0,
    "CUBIC FEET": 28.316846592,
    "CUBIC FOOT": 28.316846592,
    "CUBIC YARD(S)": 764.554857984,
}
MASS_KG = {
    "POUND(S)": 0.45359237,
    "POUNDS": 0.45359237,
    "TON(S)": 907.18474,
    "METRIC TON(S)": 1000.0,
    "KILOGRAM(S)": 1.0,
    "GRAM(S)": 1e-3,
    "MILLIGRAM(S)": 1e-6,
}
#: "units" that are really qualifiers: the amount next to them (usually 0) is not a measurement
UNKNOWN_AMOUNT_UNITS = {"UNKNOWN AMOUNT", "UNKNOWN", "NONE"}


def amount_with_unit(amount, unit) -> pd.DataFrame:
    """Amount + unit -> value (NaN when unknown), unit, qualifier, litres, kilograms.

    Units ``UNKNOWN AMOUNT``/``UNKNOWN``/``NONE`` (1990s files pair ``NONE`` with 0 even when the
    material reached water) make the amount missing, with the unit kept as ``qualifier``.
    ``OUNCE(S)`` is read as the US fluid ounce (NRC uses it for liquids); gas volumes (MCF,
    MMCF), counts (EACH), concentrations (PPM/PPB) and radioactivity (MILCURIE) are kept but
    not converted.
    """
    a = to_number(amount)
    u = _s(unit).str.upper()
    unknown = u.isin(list(UNKNOWN_AMOUNT_UNITS)).fillna(False)
    qual = pd.Series(pd.NA, index=a.index, dtype="string")
    qual[unknown] = u[unknown]
    val = a.mask(unknown)
    vol = u.map(VOLUME_L).astype("float64")
    mass = u.map(MASS_KG).astype("float64")
    return pd.DataFrame({"value": val, "unit": u, "qualifier": qual, "litres": val * vol, "kg": val * mass})


# --------------------------------------------------------------------------- tidy tables


def tidy_reports(book: dict[str, pd.DataFrame], *, source_file: str) -> pd.DataFrame:
    """One row per NRC report (SEQNOS) with call, location, impact and water-body fields."""
    calls = book.get("CALLS", pd.DataFrame(columns=["SEQNOS"]))
    com = book.get("INCIDENT_COMMONS", pd.DataFrame(columns=["SEQNOS"]))
    det = book.get("INCIDENT_DETAILS", pd.DataFrame(columns=["SEQNOS"]))

    def g(df, col):
        return df[col] if col in df.columns else pd.Series([None] * len(df), index=df.index, dtype="object")

    c = pd.DataFrame(
        {
            "report_id": report_id(g(calls, "SEQNOS")),
            "date_time_received_local": parse_local_datetime(g(calls, "DATE_TIME_RECEIVED")),
            "call_type": _s(g(calls, "CALLTYPE")),
            "is_drill": _s(g(calls, "CALLTYPE")).str.upper().eq("DRL").fillna(False).astype(bool),
            "responsible_org_type": _s(g(calls, "RESPONSIBLE_ORG_TYPE")),
            "responsible_state": _s(g(calls, "RESPONSIBLE_STATE")),
            "report_source": _s(g(calls, "SOURCE")),
        }
    )
    dq_lat, dq_lon = default_quadrants(g(com, "LOCATION_STATE"))
    lat, lat_flag = dms_to_decimal(
        g(com, "LAT_DEG"),
        g(com, "LAT_MIN"),
        g(com, "LAT_SEC"),
        g(com, "LAT_QUAD"),
        kind="lat",
        default_quad=dq_lat,
    )
    lon, lon_flag = dms_to_decimal(
        g(com, "LONG_DEG"),
        g(com, "LONG_MIN"),
        g(com, "LONG_SEC"),
        g(com, "LONG_QUAD"),
        kind="lon",
        default_quad=dq_lon,
    )
    i = pd.DataFrame(
        {
            "report_id": report_id(g(com, "SEQNOS")),
            "incident_datetime_local": parse_local_datetime(g(com, "INCIDENT_DATE_TIME")),
            "incident_datetime_raw": _s(g(com, "INCIDENT_DATE_TIME")),
            "incident_dtg": _s(g(com, "INCIDENT_DTG")),
            "type_of_incident": _s(g(com, "TYPE_OF_INCIDENT")),
            "incident_cause": _s(g(com, "INCIDENT_CAUSE")),
            "description_of_incident": _s(g(com, "DESCRIPTION_OF_INCIDENT")),
            "incident_location": _s(g(com, "INCIDENT_LOCATION")),
            "location_nearest_city": _s(g(com, "LOCATION_NEAREST_CITY")),
            "location_county": _s(g(com, "LOCATION_COUNTY")),
            "location_state": _s(g(com, "LOCATION_STATE")),
            "location_zip": zip_code(g(com, "LOCATION_ZIP")),
            "lat": lat,
            "lon": lon,
            "lat_flag": lat_flag,
            "lon_flag": lon_flag,
            "lat_dms_raw": (
                _s(g(com, "LAT_DEG")).fillna("")
                + "|"
                + _s(g(com, "LAT_MIN")).fillna("")
                + "|"
                + _s(g(com, "LAT_SEC")).fillna("")
                + "|"
                + _s(g(com, "LAT_QUAD")).fillna("")
            ),
            "lon_dms_raw": (
                _s(g(com, "LONG_DEG")).fillna("")
                + "|"
                + _s(g(com, "LONG_MIN")).fillna("")
                + "|"
                + _s(g(com, "LONG_SEC")).fillna("")
                + "|"
                + _s(g(com, "LONG_QUAD")).fillna("")
            ),
            "potential_flag": _s(g(com, "POTENTIAL_FLAG")),
        }
    )
    both = (i["lat"].notna() & i["lon"].notna()).to_numpy()
    inside = in_us_extent(i["lat"], i["lon"]).to_numpy()
    inferred = (
        (i["lat_flag"].eq("quadrant_inferred") | i["lon_flag"].eq("quadrant_inferred"))
        .fillna(False)
        .to_numpy()
    )
    given = (i["lat_flag"].ne("missing") & i["lon_flag"].ne("missing")).fillna(False).to_numpy()
    cflag = np.where(
        ~given,
        "missing",
        np.where(
            ~both,
            "invalid",
            np.where(~inside, "outside_us_extent", np.where(inferred, "quadrant_inferred", "ok")),
        ),
    )
    i["coord_flag"] = pd.Series(cflag, index=i.index, dtype="string")
    i["coordinates_given"] = pd.Series(given, index=i.index, dtype=bool)
    i["has_coordinates"] = pd.Series(both & inside, index=i.index, dtype=bool)
    i["lat_decoded"], i["lon_decoded"] = i["lat"].copy(), i["lon"].copy()  # before the extent check
    i.loc[~(both & inside), ["lat", "lon"]] = np.nan
    flags = {}
    for col, name in [
        ("WATER_SUPPLY_CONTAMINATED", "water_supply_contaminated"),
        ("WATERWAY_CLOSED", "waterway_closed"),
        ("ANY_EVACUATIONS", "any_evacuations"),
        ("ANY_INJURIES", "any_injuries"),
        ("ANY_FATALITIES", "any_fatalities"),
        ("ANY_DAMAGES", "any_damages"),
        ("FIRE_INVOLVED", "fire_involved"),
        ("ROAD_CLOSED", "road_closed"),
        ("TRACK_CLOSED", "track_closed"),
        ("AIR_CORRIDOR_CLOSED", "air_corridor_closed"),
        ("RELEASE_SECURED", "release_secured"),
        ("OFFSHORE", "offshore"),
        ("COMMUNITY_IMPACT", "community_impact"),
    ]:
        flags[name] = yn_flag(g(det, col))
        flags[name + "_raw"] = _s(g(det, col))
    d = pd.DataFrame(
        {
            "report_id": report_id(g(det, "SEQNOS")),
            "medium_desc": _s(g(det, "MEDIUM_DESC")),
            "additional_medium_info": _s(g(det, "ADDITIONAL_MEDIUM_INFO")),
            "body_of_water": _s(g(det, "BODY_OF_WATER")),
            "tributary_of": _s(g(det, "TRIBUTARY_OF")),
            "nearest_river_mile_marker": _s(g(det, "NEAREST_RIVER_MILE_MARKER")),
            **flags,
            "waterway_desc": _s(g(det, "WATERWAY_DESC")),
            "waterway_closure_time": _s(g(det, "WATERWAY_CLOSURE_TIME")),
            "number_evacuated": to_number(g(det, "NUMBER_EVACUATED")),
            "number_injured": to_number(g(det, "NUMBER_INJURED")),
            "number_hospitalized": to_number(g(det, "NUMBER_HOSPITALIZED")),
            "number_fatalities": to_number(g(det, "NUMBER_FATALITIES")),
            "damage_amount": to_number(g(det, "DAMAGE_AMOUNT")),
            "estimated_duration_of_release": _s(g(det, "ESTIMATED_DURATION_OF_RELEASE")),
            "duration_unit": _s(g(det, "DURATION_UNIT")),
            "release_rate": _s(g(det, "RELEASE_RATE")),
            "release_rate_unit": _s(g(det, "RELEASE_RATE_UNIT")),
            "release_rate_rate": _s(g(det, "RELEASE_RATE_RATE")),
            "sheen_size": _s(g(det, "SHEEN_SIZE")),
            "sheen_size_units": _s(g(det, "SHEEN_SIZE_UNITS")),
            "sheen_color": _s(g(det, "SHEEN_COLOR")),
            "water_temperature": _s(g(det, "WATER_TEMPERATURE")),
            "desc_remedial_action": _s(g(det, "DESC_REMEDIAL_ACTION")),
        }
    )
    dups = {}
    for nm, frame in (("CALLS", c), ("INCIDENT_COMMONS", i), ("INCIDENT_DETAILS", d)):
        dups[nm] = int(frame["report_id"].duplicated().sum())
    # a few workbooks repeat a SEQNOS inside one sheet; keep its last row (counts in attrs)
    c, i, d = (f[f["report_id"].notna()].drop_duplicates("report_id", keep="last") for f in (c, i, d))
    out = c.merge(i, on="report_id", how="outer").merge(d, on="report_id", how="outer")
    out.attrs["within_sheet_duplicates"] = dups
    out.insert(1, "source_file", source_file)
    out.insert(2, "file_year", year_from_filename(source_file))
    return out


def tidy_materials(book: dict[str, pd.DataFrame], *, source_file: str) -> pd.DataFrame:
    """One row per (report, material) from MATERIAL_INVOLVED with cleaned CAS and amounts."""
    m = book.get("MATERIAL_INVOLVED", pd.DataFrame(columns=["SEQNOS"]))

    def g(col):
        return m[col] if col in m.columns else pd.Series([None] * len(m), index=m.index, dtype="object")

    cas_raw = _s(g("CAS_NUMBER"))
    cas_norm = cas_raw.map(lambda x: normalize_cas(x) if x is not pd.NA else None).astype("string")
    rel = amount_with_unit(g("AMOUNT_OF_MATERIAL"), g("UNIT_OF_MEASURE"))
    wat = amount_with_unit(g("AMOUNT_IN_WATER"), g("UNIT_OF_MEASURE_REACH_WATER"))
    out = pd.DataFrame(
        {
            "report_id": report_id(g("SEQNOS")),
            "source_file": source_file,
            "chris_code": _s(g("CHRIS_CODE")),
            "cas_raw": cas_raw,
            "cas": cas_norm,
            "un_number": _s(g("UN_NUMBER")).str.replace(r"\.0$", "", regex=True),
            "name_of_material": _s(g("NAME_OF_MATERIAL")),
            "amount_of_material": rel["value"],
            "unit_of_measure": rel["unit"],
            "amount_qualifier": rel["qualifier"],
            "amount_released_litres": rel["litres"],
            "amount_released_kg": rel["kg"],
            "if_reached_water": yn_flag(g("IF_REACHED_WATER")),
            "if_reached_water_raw": _s(g("IF_REACHED_WATER")),
            "amount_in_water": wat["value"],
            "unit_of_measure_reach_water": wat["unit"],
            "amount_in_water_qualifier": wat["qualifier"],
            "amount_in_water_litres": wat["litres"],
            "amount_in_water_kg": wat["kg"],
        }
    )
    out.insert(1, "material_seq", out.groupby("report_id").cumcount().astype("int32"))
    return out
