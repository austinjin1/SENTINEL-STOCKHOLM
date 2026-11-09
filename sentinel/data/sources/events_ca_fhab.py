# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""California Water Boards FHAB (Freshwater Harmful Algal Bloom) portal → tidy secondary HAB labels.

Source: data.ca.gov CKAN package ``surface-water-freshwater-harmful-algal-blooms`` (State Water
Resources Control Board; licence "Other (Public Domain)"). Four CSV resources (also loaded in the
CKAN datastore): Bloom Reports, Cases, Responses (advisory actions) and Results, plus a PDF data
dictionary and disclaimer. See docs/datasheets/labels_ca_fhab.md.

What the labels mean (official disclaimer, verified 2026-10-22):
  * A *bloom report* is a voluntary submission (public / agency / partner) "assumed to represent a
    suspected bloom until it can be confirmed by technical staff". A report is treated as a
    confirmed bloom when staff recommend an advisory other than "None".
  * Advisory tiers follow the CCHAB voluntary guidance (2016): Caution (Tier 1) / Warning
    (Tier 2) / Danger (Tier 3). Caution is met by total microcystins ≥ 0.8 µg/L, anatoxin-a
    detected, cylindrospermopsin ≥ 1 µg/L, ≥ 4,000 cells/mL of potential toxin producers, OR a
    site-specific visual indicator (scum, discolouration, mats) or suspected illness → the label
    is partly visual and therefore partly circular with predicted chl-a.
  * Advisory end dates are mostly blank ("may never be an end date" for unmonitored waters).
  * Blank fields mean "unknown", not "none".

Parsing rules: every value is read as text; dates are validated against a calendar window
(impossible years such as 1976/2555/2626 → NaT with a ``*_flag``); coordinates outside a buffered
California box, (0, 0) or malformed values (e.g. -1215574) → NaN with ``coord_flag``. Staff
e-mail addresses (``Response_Update_By``) and staff names (``Case_Lead``) are dropped from the
processed tables. Results rows that are empty placeholders (value "0" with no analyte or
measurement type) get ``value`` = NaN; veterinary results are excluded per the data dictionary.
"""

from __future__ import annotations

import hashlib
import io
import re
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from . import events_common as C

CKAN_BASE = "https://data.ca.gov"
PACKAGE_ID = "surface-water-freshwater-harmful-algal-blooms"
PACKAGE_URL = f"{CKAN_BASE}/dataset/{PACKAGE_ID}"
DATASTORE_URL = f"{CKAN_BASE}/api/3/action/datastore_search"
LICENSE = "Other (Public Domain) — data.ca.gov CKAN licence_id 'other-pd'"

# Resource short names → regex on the CKAN resource name (ids resolved at run time; the ids
# seen on 2026-10-22 are kept for reference only).
DATA_RESOURCES = {
    "bloom_reports": r"bloom\s*reports?",
    "cases": r"\bcases?\b",
    "responses": r"responses?",
    "results": r"results?",
}
DOC_RESOURCES = {
    "data_dictionary": r"data\s*dictionary",
    "disclaimer": r"disclaimer",
}
RESOURCE_IDS_2026_10_22 = {
    "bloom_reports": "c6a36b91-ad38-4611-8750-87ee99e497dd",
    "cases": "67648948-034f-4882-bbc0-c07c7d38daf9",
    "responses": "4283c060-c22f-48f5-a75c-8bccf0c54a99",
    "results": "9d4e1df4-0cd6-4165-9e63-effcafd9dccc",
    "data_dictionary": "516bd039-e09c-4e2f-aa65-eda947af729b",
    "disclaimer": "9a54fa52-e16f-4943-b27d-2b4eca7c1bd6",
}

# California plus a ~0.25° buffer: FHAB legitimately covers border waters (Klamath reservoirs in
# Oregon, the Nevada side of Lake Tahoe, the Colorado River / Lake Havasu).
CA_BBOX = (-124.7, 32.3, -113.85, 42.25)
MIN_EVENT_DATE = "2014-01-01"   # FHAB system data start 2016; one 2015 observation is plausible
CSV_DT_FMT = "%m/%d/%Y %I:%M:%S %p"

TIER_NAMES = ["None", "Caution", "Warning", "Danger"]
TIER_DTYPE = pd.CategoricalDtype(TIER_NAMES, ordered=True)
TIER_LEVEL = {"none": 0, "caution": 1, "warning": 2, "danger": 3}
ADVISORY_CLASS = {
    "none": "none",
    "caution": "caution",
    "warning": "warning",
    "danger": "danger",
    "algal mat alert sign": "algal_mat_alert",
    "algal mat general awareness sign": "algal_mat_awareness",
    "general awareness": "general_awareness",
    "visual observation": "visual_observation",
    "na - refer to report details": "na_see_report",
}
# Classes that mean "no advisory recommended / not determinable" (everything else = some advisory)
NON_ADVISORY_CLASSES = {"none", "na_see_report", "unknown"}

# CCHAB trigger levels for posting planktonic advisory signs (mywaterquality.ca.gov
# /habs/resources/response-guidance.html, Table 3; verified 2026-10-22). Documentation only.
CCHAB_TRIGGER_LEVELS = {
    "total_microcystins_ug_l": {"Caution": 0.8, "Warning": 6.0, "Danger": 20.0},
    "anatoxin_a_ug_l": {"Caution": "detected", "Warning": 20.0, "Danger": 90.0},
    "cylindrospermopsin_ug_l": {"Caution": 1.0, "Warning": 4.0, "Danger": 17.0},
    "potential_toxin_producer_cells_per_ml": {"Caution": 4000},
    "site_specific_indicators": {"Caution": "discoloration, scum, algal mats, soupy or paint-like "
                                            "appearance; suspected illness"},
    "de_posting": "all no-advisory criteria met for >= 2 weeks",
}

DROP_PERSONAL = ["Response_Update_By", "Case_Lead"]


# ----------------------------------------------------------------------------- raw readers

def read_raw_csv(src: str | Path | bytes) -> pd.DataFrame:
    """Published FHAB CSV (UTF-8 with BOM) → all-string DataFrame (blank = '')."""
    if isinstance(src, (bytes, bytearray)):
        src = io.BytesIO(src)
    return pd.read_csv(src, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def datastore_records_to_frame(records: list[dict], fields: list[dict]) -> pd.DataFrame:
    """CKAN datastore records (typed JSON) → all-string DataFrame in published column order.

    Used only as a fallback when the CSV download fails. Dates come back ISO-formatted
    (``2016-04-30T00:00:00.000``) instead of the CSV's ``4/30/2016 12:00:00 AM``; the date
    parser below accepts both.
    """
    cols = [f["id"] for f in fields if f.get("id") != "_id"]

    def _txt(v):
        if v is None:
            return ""
        if isinstance(v, float):
            return str(int(v)) if v.is_integer() else repr(v)
        return str(v)
    rows = [[_txt(r.get(c)) for c in cols] for r in records]
    return pd.DataFrame(rows, columns=cols, dtype=str)


def resolve_resources(package: dict) -> dict[str, dict]:
    """Map short names (bloom_reports, cases, responses, results, data_dictionary, disclaimer)
    to CKAN resource dicts by matching resource names."""
    out: dict[str, dict] = {}
    for res in package.get("resources", []):
        name = (res.get("name") or "").lower()
        for short, pat in {**DOC_RESOURCES, **DATA_RESOURCES}.items():
            if short not in out and re.search(pat, name):
                out[short] = res
                break
    missing = [k for k in DATA_RESOURCES if k not in out]
    if missing:
        raise ValueError(f"FHAB package is missing resources: {missing}")
    return out


# ----------------------------------------------------------------------------- field parsers

def _txt(values: pd.Series | Iterable) -> pd.Series:
    s = pd.Series(values, dtype="object")
    return s.map(lambda v: "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v).strip())


def _na(values: pd.Series | Iterable) -> pd.Series:
    """Blank → <NA> (string dtype); FHAB blank means 'unknown'."""
    s = _txt(values)
    return s.where(s != "").astype("string")


def parse_datetime(values: pd.Series | Iterable, *, min_date: str = MIN_EVENT_DATE,
                   max_date: str | None = None) -> tuple[pd.Series, pd.Series]:
    """FHAB date/time text → naive local datetime + flag ('', missing, unparseable, out_of_range).

    Tries the CSV format (``%m/%d/%Y %I:%M:%S %p``) first, then element-wise parsing for anything
    else (datastore ISO strings).
    """
    txt = _txt(values)
    dt, flag = C.clean_dates(txt, min_date=min_date, max_date=max_date, fmt=CSV_DT_FMT)
    retry = (flag == "unparseable") & ~txt.str.match(r"^\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2}:\d{2} [AP]M$")
    if retry.any():
        dt2, flag2 = C.clean_dates(txt[retry], min_date=min_date, max_date=max_date)
        dt = dt.astype("datetime64[ns]")
        dt[retry] = dt2
        flag[retry] = flag2
    return dt, flag


def parse_date(values, **kw) -> tuple[pd.Series, pd.Series]:
    """As :func:`parse_datetime` but normalised to midnight (local calendar date)."""
    dt, flag = parse_datetime(values, **kw)
    return dt.dt.normalize(), flag


def advisory_class(values: pd.Series | Iterable) -> pd.Series:
    """Raw ``Advisory_Recommended`` → normalised class (unknown for blank, other_<text> if new)."""
    t = _txt(values).str.lower().str.replace(r"\s+", " ", regex=True)

    def _one(v: str) -> str:
        if v == "":
            return "unknown"
        if v in ADVISORY_CLASS:
            return ADVISORY_CLASS[v]
        return "other_" + re.sub(r"[^a-z0-9]+", "_", v).strip("_")
    return t.map(_one).astype("string")


def tier_level(classes: pd.Series) -> pd.Series:
    """Advisory class → ordinal tier (0 None, 1 Caution, 2 Warning, 3 Danger); <NA> otherwise."""
    return pd.Series(classes, dtype="object").map(TIER_LEVEL).astype("Int8")


def tier_category(levels: pd.Series) -> pd.Series:
    lv = pd.Series(levels).astype("Int8")
    names = lv.map(lambda v: TIER_NAMES[int(v)] if pd.notna(v) else None)
    return pd.Series(pd.Categorical(names, dtype=TIER_DTYPE), index=lv.index)


def yes_no(values: pd.Series | Iterable) -> pd.Series:
    """YES/Yes/yes/no/NO → boolean; 'unknown'/blank → <NA>."""
    t = _txt(values).str.lower()
    return t.map({"yes": True, "y": True, "true": True, "no": False, "n": False, "false": False}
                 ).astype("boolean")


def yes_no_unknown(values: pd.Series | Iterable) -> pd.Series:
    t = _txt(values).str.lower()
    out = t.map({"yes": "yes", "no": "no", "unknown": "unknown"})
    return out.astype("string")


def observation_channel(case_class: pd.Series | Iterable) -> pd.Series:
    """Case_Class → event_response | monitoring | other | unknown (sampling-channel stratifier)."""
    t = _txt(case_class).str.lower()
    out = np.select([t == "event response", t.str.startswith("monitoring"), t == "other", t == ""],
                    ["event_response", "monitoring", "other", "unknown"], default="other")
    return pd.Series(out, index=t.index, dtype="string")


def monitoring_type(case_class: pd.Series | Iterable) -> pd.Series:
    """'Monitoring_Local Agency' → 'local_agency'; non-monitoring classes → <NA>."""
    t = _txt(case_class)
    m = t.str.extract(r"(?i)^monitoring_(.+)$", expand=False)
    return m.str.strip().str.lower().str.replace(r"[^a-z0-9]+", "_", regex=True).astype("string")


def report_source(report_type: pd.Series | Iterable) -> pd.Series:
    t = _txt(report_type).str.lower()
    out = np.select([t.str.contains("public"), t.str.contains("agency|partner"), t == ""],
                    ["public", "agency_partner", "unknown"], default="other")
    return pd.Series(out, index=t.index, dtype="string")


def advisory_basis_flags(detail: pd.Series | Iterable) -> pd.DataFrame:
    """Flags parsed from ``Advisory_Detail`` (semicolon-joined phrases) describing label basis."""
    t = _txt(detail).str.lower()
    blank = t == ""

    def _flag(pattern: str) -> pd.Series:
        return t.str.contains(pattern, regex=True).astype("boolean").mask(blank)
    return pd.DataFrame({
        "basis_toxins": _flag(r"visual and toxins|toxin"),
        "basis_visual_only": _flag(r"visual monitoring based|visual monitoring only"),
        "below_triggers": _flag(r"below posting triggers|no cyano bloom w/testing"),
        "lake_wide": _flag(r"lake-wide|lakewide"),
        "routine_monitoring": _flag(r"routine monitoring"),
        "under_investigation": _flag(r"under investigation"),
    }, index=t.index)


def _coords(lat, lon) -> tuple[pd.Series, pd.Series, pd.Series]:
    la, lo, flag = C.clean_latlon(_txt(lat).replace("", np.nan), _txt(lon).replace("", np.nan), bbox=CA_BBOX)
    return la, lo, flag.astype("string")


def _num(values) -> pd.Series:
    return pd.to_numeric(_txt(values), errors="coerce")


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    return df[name] if name in df.columns else pd.Series("", index=df.index, dtype="object")


def _advisory_block(df: pd.DataFrame, rec_col: str = "Advisory_Recommended") -> pd.DataFrame:
    cls = advisory_class(_col(df, rec_col))
    lvl = tier_level(cls)
    return pd.DataFrame({
        "advisory_recommended": _na(_col(df, rec_col)),
        "advisory_class": cls,
        "tier_level": lvl,
        "advisory_tier": tier_category(lvl),
        "label_tier_ge_caution": (lvl >= 1).astype("boolean"),
        "any_advisory": (~cls.isin(list(NON_ADVISORY_CLASSES))).astype("boolean").mask(
            cls.isin(["unknown", "na_see_report"])),
    }, index=df.index)


def _case_block(df: pd.DataFrame, max_date: str | None) -> pd.DataFrame:
    start, _ = parse_date(_col(df, "Case_Start_Date"), max_date=max_date)
    end, _ = parse_date(_col(df, "Case_End_Date"), max_date=max_date)
    return pd.DataFrame({
        "case_class": _na(_col(df, "Case_Class")),
        "observation_channel": observation_channel(_col(df, "Case_Class")),
        "monitoring_type": monitoring_type(_col(df, "Case_Class")),
        "case_status": _na(_col(df, "Case_Status")),
        "case_year": _num(_col(df, "Case_Year")).astype("Int16"),
        "case_start_date": start,
        "case_end_date": end,
    }, index=df.index)


# ----------------------------------------------------------------------------- tidy tables

def tidy_bloom_reports(raw: pd.DataFrame, *, max_date: str | None = None) -> pd.DataFrame:
    """One row per Bloom_Report_ID (the published grain is report × linked advisory).

    When a report links several advisories, the row with the highest tier (then latest
    Advisory_Date) is kept and ``advisory_ids`` / ``n_linked_advisories`` record the others.
    """
    df = raw.copy()
    obs, obs_f = parse_date(_col(df, "Observation_Date"), max_date=max_date)
    created, _ = parse_datetime(_col(df, "Bloom_Date_Created"), max_date=max_date)
    after = obs.notna() & created.notna() & (obs > created.dt.normalize() + pd.Timedelta(days=1))
    obs_f = obs_f.where(~after, "after_created")
    lat, lon, cflag = _coords(_col(df, "Bloom_Latitude"), _col(df, "Bloom_Longitude"))
    adv = _advisory_block(df)
    a_date, a_date_f = parse_date(_col(df, "Advisory_Date"), max_date=max_date)
    rec_date, _ = parse_date(_col(df, "Advisory_Date_of_Recommendation"), max_date=max_date)
    a_start, a_start_f = parse_date(_col(df, "AdvisoryStartDate"), max_date=max_date)
    a_end, a_end_f = parse_date(_col(df, "AdvisoryEndDate"), max_date=max_date)
    last_res, _ = parse_date(_col(df, "Last_Field_Result_Sample_Date"), max_date=max_date)
    out = pd.DataFrame({
        "source": "ca_fhab",
        "report_id": C.str_id(_col(df, "Bloom_Report_ID")),
        "case_id": C.str_id(_col(df, "Case_ID")),
        "advisory_id": C.str_id(_col(df, "Advisory_ID")),
        "report_type": _na(_col(df, "Report_Type")),
        "report_source": report_source(_col(df, "Report_Type")),
        "case_assignment": _na(_col(df, "Case_Assignment")),
        "date": obs,
        "date_flag": obs_f.astype("string"),
        "created_local": created,
        "waterbody_name": _na(_col(df, "Water_Body_Name")),
        "official_waterbody_name": _na(_col(df, "Official_Water_Body_Name")),
        "landmark": _na(_col(df, "Landmark")),
        "county": _na(_col(df, "County")),
        "regional_board": _na(_col(df, "Regional_Water_Board")),
        "waterbody_type": _na(_col(df, "Water_Body_Type")),
        "waterbody_use": _na(_col(df, "Water_Body_Use")),
        "waterbody_manager": _na(_col(df, "Water_Body_Manager")),
        "drinking_water_source": yes_no_unknown(_col(df, "Drinking_Water_Source")),
        "latitude": lat,
        "longitude": lon,
        "coord_flag": cflag,
        "datum": _na(_col(df, "Bloom_Datum")),
        "reported_advisory_types": _na(_col(df, "Reported_Advisory_Types")),
    }, index=df.index)
    out = pd.concat([out, adv], axis=1)
    out = out.assign(
        advisory_date=a_date, advisory_date_flag=a_date_f.astype("string"),
        recommendation_date=rec_date,
        advisory_start_date=a_start, advisory_start_flag=a_start_f.astype("string"),
        advisory_end_date=a_end, advisory_end_flag=a_end_f.astype("string"),
        has_end_date=a_end.notna(),
        advisory_detail=_na(_col(df, "AdvisoryDetail")),
        advisory_detail_description=_na(_col(df, "Advisory_Detail_Description")),
        spatial_extent=_na(_col(df, "Spatial_Extent_of_Advisory")),
        extent_unit=_na(_col(df, "Extent_Unit_of_Measure")),
        last_field_result_sample_date=last_res,
        lab_data_linked=yes_no(_col(df, "Lab_Data_Linked_to_Bloom")),
        field_visual_linked=yes_no(_col(df, "Field_Visual_Records_Linked_to_Bloom")),
        field_measurement_linked=yes_no(_col(df, "Field_Measurement_Data_Linked_to_Bloom")),
        has_pictures=yes_no(_col(df, "Has_Pictures")),
        bloom_size=_na(_col(df, "Bloom_Size")),
        bloom_location=_na(_col(df, "Bloom_Location")),
        bloom_texture=_na(_col(df, "Bloom_Texture")),
        weather_condition=_na(_col(df, "Weather_Condition")),
        surface_water_condition=_na(_col(df, "Surface_Water_Condition")),
    )
    out = pd.concat([out, _case_block(df, max_date)], axis=1)
    out = out[out["report_id"].notna()].copy()
    # collapse report × advisory → report
    ids = (out.groupby("report_id")["advisory_id"]
           .agg(lambda s: ";".join(sorted(set(s.dropna()), key=lambda x: (len(x), x)))))
    nlink = out.groupby("report_id")["advisory_id"].nunique()
    out["_tier_sort"] = out["tier_level"].astype("float").fillna(-1)
    out = (out.sort_values(["report_id", "_tier_sort", "advisory_date"], ascending=[True, False, False],
                           na_position="last", kind="mergesort")
           .drop_duplicates("report_id", keep="first").drop(columns="_tier_sort"))
    out["advisory_ids"] = out["report_id"].map(ids).replace("", pd.NA).astype("string")
    out["n_linked_advisories"] = out["report_id"].map(nlink).astype("Int16")
    out = out.sort_values(["date", "report_id"], na_position="last", kind="mergesort")
    return out.reset_index(drop=True)


def tidy_advisories(raw: pd.DataFrame, *, max_date: str | None = None) -> pd.DataFrame:
    """One row per advisory response action (``Response_Action_ID``) — the advisory history.

    ``date`` = date of the (changed) recommendation (``Advisory_Date_of_Recommendation``, falling
    back to ``Advisory_Date``, the date the report was last verified).
    """
    df = raw.copy()
    rec, rec_f = parse_date(_col(df, "Advisory_Date_of_Recommendation"), max_date=max_date)
    adate, adate_f = parse_date(_col(df, "Advisory_Date"), max_date=max_date)
    date = rec.where(rec.notna(), adate)
    date_f = rec_f.where(rec.notna(), adate_f)
    start, start_f = parse_date(_col(df, "Advisory_Start_Date"), max_date=max_date)
    end, end_f = parse_date(_col(df, "Advisory_End_Date"), max_date=max_date)
    stamp, _ = parse_datetime(_col(df, "Response_DateTimeStamp"), max_date=max_date)
    lat, lon, cflag = _coords(_col(df, "Bloom_Latitude"), _col(df, "Bloom_Longitude"))
    out = pd.DataFrame({
        "source": "ca_fhab",
        "response_action_id": C.str_id(_col(df, "Response_Action_ID")),
        "advisory_id": C.str_id(_col(df, "Advisory_ID")),
        "report_id": C.str_id(_col(df, "Bloom_Report_ID")),
        "case_id": C.str_id(_col(df, "Case_ID")),
        "report_type": _na(_col(df, "Report_Type")),
        "report_source": report_source(_col(df, "Report_Type")),
        "date": date,
        "date_flag": date_f.astype("string"),
        "advisory_date": adate,
        "advisory_start_date": start,
        "advisory_start_flag": start_f.astype("string"),
        "advisory_end_date": end,
        "advisory_end_flag": end_f.astype("string"),
        "has_end_date": end.notna(),
        "response_timestamp_local": stamp,
        "waterbody_name": _na(_col(df, "Water_Body_Name")),
        "official_waterbody_name": _na(_col(df, "Official_Water_Body_Name")),
        "landmark": _na(_col(df, "Landmark")),
        "county": _na(_col(df, "County")),
        "regional_board": _na(_col(df, "Regional_Water_Board")),
        "waterbody_type": _na(_col(df, "Water_Body_Type")),
        "waterbody_use": _na(_col(df, "Water_Body_Use")),
        "drinking_water_source": yes_no_unknown(_col(df, "Drinking_Water_Source")),
        "latitude": lat,
        "longitude": lon,
        "coord_flag": cflag,
        "datum": _na(_col(df, "Bloom_Datum")),
    }, index=df.index)
    out = pd.concat([out, _advisory_block(df), advisory_basis_flags(_col(df, "Advisory_Detail"))], axis=1)
    out = out.assign(
        advisory_detail=_na(_col(df, "Advisory_Detail")),
        spatial_extent=_na(_col(df, "Spatial_Extent_of_Advisory")),
        extent_unit=_na(_col(df, "Extent_Unit_of_Measure")),
        display_on_map=yes_no(_col(df, "DisplayAdvisoryToMap")),
    )
    out = pd.concat([out, _case_block(df, max_date)], axis=1)
    out = out[out["response_action_id"].notna()].drop_duplicates("response_action_id")
    return out.sort_values(["date", "advisory_id", "response_action_id"], na_position="last",
                           kind="mergesort").reset_index(drop=True)


def tidy_cases(raw: pd.DataFrame, *, max_date: str | None = None) -> pd.DataFrame:
    """One row per case (a staff folder of reports; not a severity measure)."""
    df = raw.copy()
    lat, lon, cflag = _coords(_col(df, "Bloom_Latitude"), _col(df, "Bloom_Longitude"))
    stamp, _ = parse_datetime(_col(df, "Case_DateTimeStamp"), max_date=max_date)
    out = pd.DataFrame({
        "source": "ca_fhab",
        "case_id": C.str_id(_col(df, "Case_ID")),
        "case_waterbody_name": _na(_col(df, "Case_Water_Body_Name")),
        "case_assignment": _na(_col(df, "Case_Assignment")),
        "report_id": C.str_id(_col(df, "Bloom_Report_ID")),
        "report_type": _na(_col(df, "Report_Type")),
        "report_source": report_source(_col(df, "Report_Type")),
        "n_blooms_linked": _num(_col(df, "Number_of_Blooms_Linked_to_Case")).astype("Int32"),
        "n_advisories_linked": _num(_col(df, "Number_of_Advisories_Linked_to_Case")).astype("Int32"),
        "advisory_id": C.str_id(_col(df, "Advisory_ID")),
        "county": _na(_col(df, "County")),
        "regional_board": _na(_col(df, "Regional_Water_Board")),
        "waterbody_type": _na(_col(df, "Water_Body_Type")),
        "latitude": lat,
        "longitude": lon,
        "coord_flag": cflag,
        "lab_data_linked": yes_no(_col(df, "Lab_Data_Linked_to_Case")),
        "field_measurement_linked": yes_no(_col(df, "Field_Measurement_Data_Linked_to_Case")),
        "field_visual_linked": yes_no(_col(df, "Field_Visual_Records_Linked_to_Case")),
        "case_timestamp_local": stamp,
    }, index=df.index)
    out = pd.concat([out, _case_block(df, max_date), _advisory_block(df)], axis=1)
    out = out[out["case_id"].notna()].drop_duplicates("case_id")
    out["_cid"] = pd.to_numeric(out["case_id"], errors="coerce")
    out = out.sort_values(["case_year", "_cid"], na_position="last", kind="mergesort")
    return out.drop(columns="_cid").reset_index(drop=True)


_ND = re.compile(r"(?i)^(nd|n\.d\.|non[- ]?detect(ed)?|not detected|bdl|<\s*mdl|<\s*rl)$")
_LT = re.compile(r"^\s*<\s*=?\s*([0-9]*\.?[0-9]+)\s*$")


def parse_measurement(values: pd.Series | Iterable) -> pd.DataFrame:
    """Measurement text → value, censored, detection_limit, qualifier, value_category.

    '<0.15' → censored, detection_limit 0.15, value NaN (no DL/2 substitution); 'ND' → censored,
    DL unknown; 'present'/'absent' → value_category; sentinel codes (−9999 …) → NaN + qualifier.
    """
    t = _txt(values)
    lt = t.str.extract(_LT, expand=False)
    nd = t.str.match(_ND)
    cat = t.str.lower().map({"present": "present", "absent": "absent", "detected": "present",
                             "not present": "absent"})
    num, qual = C.to_number(t.where(lt.isna() & ~nd & cat.isna(), ""))
    censored = (lt.notna() | nd)
    return pd.DataFrame({
        "value": num,
        "censored": censored.astype("boolean").mask(t == ""),
        "detection_limit": pd.to_numeric(lt, errors="coerce"),
        "qualifier": qual.where(qual != "", t.where(lt.notna() | nd | cat.notna(), "")).replace("", pd.NA)
        .astype("string"),
        "value_category": cat.astype("string"),
    }, index=t.index)


def tidy_results(raw: pd.DataFrame, *, max_date: str | None = None) -> pd.DataFrame:
    """One row per published result (field visual, point measurement or lab record).

    The FHAB results table is sparse (2026-10-22: lab rows carry analyte/method but no values;
    toxin concentrations live in CEDEN). Placeholder rows (value '0' with no analyte and no
    measurement type) get value NaN and ``is_placeholder`` True. Veterinary rows are dropped.
    """
    df = raw.copy()
    dtype = _txt(_col(df, "Data_Type"))
    df = df[~dtype.str.lower().str.contains("veterin")].copy()
    meas = parse_measurement(_col(df, "Measurement_Value"))
    analyte = _txt(_col(df, "Analyte"))
    mtype = _txt(_col(df, "Measurement_Type"))
    vtxt = _txt(_col(df, "Measurement_Value"))
    placeholder = (analyte == "") & (mtype == "") & vtxt.isin(["0", ""])
    for c in ["value", "detection_limit"]:
        meas[c] = meas[c].mask(placeholder)
    meas["censored"] = meas["censored"].mask(placeholder)
    s_lat, s_lon, s_flag = _coords(_col(df, "Latitude"), _col(df, "Longitude"))
    b_lat, b_lon, b_flag = _coords(_col(df, "Bloom_Latitude"), _col(df, "Bloom_Longitude"))
    use_sample = s_lat.notna()
    sdate, sdate_f = parse_date(_col(df, "Sample_Date"), max_date=max_date)
    rdate, _ = parse_date(_col(df, "Results_Date"), max_date=max_date)
    unit = _na(_col(df, "Measurement_Unit"))
    out = pd.DataFrame({
        "source": "ca_fhab",
        "result_uid": _na(_col(df, "RESULT ID UNIQUE")),
        "result_id": C.str_id(_col(df, "Result_ID")),
        "sample_kind": _na(_col(df, "Sample_Type")),
        "report_id": C.str_id(_col(df, "Bloom_Report_ID")),
        "case_id": C.str_id(_col(df, "Case_ID")),
        "waterbody_name": _na(_col(df, "Water_Body_Name")),
        "county": _na(_col(df, "County")),
        "waterbody_type": _na(_col(df, "Water_Body_Type")),
        "date": sdate,
        "date_flag": sdate_f.astype("string"),
        "results_date": rdate,
        "latitude": s_lat.where(use_sample, b_lat),
        "longitude": s_lon.where(use_sample, b_lon),
        "coord_source": pd.Series(np.where(use_sample, "sample", np.where(b_lat.notna(), "bloom_report", "")),
                                  index=df.index).replace("", pd.NA).astype("string"),
        "coord_flag": s_flag.where(use_sample, b_flag),
        "data_type": _na(_col(df, "Data_Type")),
        "analysis_type": _na(_col(df, "Analysis_Type")),
        "analyte_class": _na(_col(df, "Analyte_Class")),
        "analyte": _na(_col(df, "Analyte")),
        "method": _na(_col(df, "Method")),
        "measurement_type": _na(_col(df, "Measurement_Type")),
        "unit": unit,
        "value_text": _na(_col(df, "Measurement_Value")),
        "is_placeholder": placeholder.to_numpy(),
    }, index=df.index)
    out = pd.concat([out, meas], axis=1)
    out = out.assign(
        sample_id=_na(_col(df, "Sample_ID")),
        sample_location=_na(_col(df, "Sample_Location")),
        site=_na(_col(df, "Site")),
        bloom_type=_na(_col(df, "Bloom_Type")),
        bloom_size=_na(_col(df, "Bloom_Size")),
        bloom_texture=_na(_col(df, "Bloom_Texture")),
        taxa=_na(_col(df, "Taxa")),
        weather_condition=_na(_col(df, "Weather_Condition")),
        water_surface_condition=_na(_col(df, "Water_Surface_Condition")),
    )
    out = out[out["result_uid"].notna()].drop_duplicates("result_uid")
    return out.reset_index(drop=True)


def _finish(df: pd.DataFrame) -> pd.DataFrame:
    """Object columns holding only text/None → pandas string dtype (stable Parquet schema)."""
    for c in df.columns:
        if df[c].dtype == object and df[c].map(lambda v: v is None or isinstance(v, str)
                                                or (isinstance(v, float) and np.isnan(v))).all():
            df[c] = df[c].astype("string")
    return df


def build_tables(raw: dict[str, pd.DataFrame], *, max_date: str | None = None) -> dict[str, pd.DataFrame]:
    """Raw resource frames {bloom_reports, cases, responses, results} → processed tables."""
    return {
        "bloom_reports": _finish(tidy_bloom_reports(raw["bloom_reports"], max_date=max_date)),
        "advisories": _finish(tidy_advisories(raw["responses"], max_date=max_date)),
        "cases": _finish(tidy_cases(raw["cases"], max_date=max_date)),
        "results": _finish(tidy_results(raw["results"], max_date=max_date)),
    }


def end_date_stats(raw_bloom_reports: pd.DataFrame, raw_responses: pd.DataFrame) -> dict:
    """How often an advisory end date is published (research claim: 238 of 3,297 starts)."""
    def _cnt(df, s, e):
        st = _txt(_col(df, s)) != ""
        en = _txt(_col(df, e)) != ""
        return {"rows": len(df), "with_start": int(st.sum()), "with_end": int(en.sum()),
                "with_start_and_end": int((st & en).sum())}
    return {"bloom_reports": _cnt(raw_bloom_reports, "AdvisoryStartDate", "AdvisoryEndDate"),
            "responses": _cnt(raw_responses, "Advisory_Start_Date", "Advisory_End_Date")}


def frame_sha1(df: pd.DataFrame) -> str:
    """Order-sensitive content hash of a raw frame (vintage fingerprint)."""
    h = hashlib.sha1()
    h.update("\x1f".join(df.columns).encode())
    for row in df.itertuples(index=False):
        h.update("\x1f".join(map(str, row)).encode())
    return h.hexdigest()
