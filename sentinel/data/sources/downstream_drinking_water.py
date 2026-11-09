# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Drinking-water labels for TapRisk (D4): SDWIS violations, UCMR 4 source-water DBP
precursors, and EPA public-water-system service-area boundaries.

Sources (verified 2026-10-22; see docs/datasheets/labels_{sdwis,ucmr,cws_boundaries}.md):

* **SDWIS** via the ECHO SDWA bulk download (quarterly snapshot; one zip of CSV tables).
  ``SDWA_VIOLATIONS_ENFORCEMENT.csv`` is a violation x enforcement-action join (one row per
  enforcement action, or one row with blank enforcement fields), so violations are
  de-duplicated on ``(PWSID, VIOLATION_ID)`` (VIOLATION_ID alone is not unique) and the
  enforcement history is summarised per violation. Code meanings come from
  ``SDWA_REF_CODE_VALUES.csv`` in the same zip and the ECHO "SDWA Download Summary" data
  dictionary (https://echo.epa.gov/tools/data-downloads/sdwa-download-summary).
* **UCMR 4** occurrence data (2018-2021 sample dates; EPA). Raw/source-water (sample point type
  ``SR``) total organic carbon (TOC) and bromide — the DBP precursors — are in
  ``UCMR4_HAA_Indicators.txt``; HAA group results (distribution system, ``DS``) and the
  cyanotoxins (entry point, ``EP``) are in ``UCMR4_All.txt``; additional data elements
  (disinfectant/treatment, bloom questionnaire) in the ``*_AddtlDataElem.txt`` files.
  Code meanings: UCMR 4 Data Summary (Tables 5, 7, 8) inside the zip.
* **EPA PWS service-area boundaries v3** (ORD SAB model; GitHub ``USEPA/ORD_SAB_Model``,
  ``PWS_Boundaries_Latest.zip`` -> ``3_0/Service_Areas_V_3_0.gpkg``, layer ``CWS``).

Privacy: ``SDWA_PUB_WATER_SYSTEMS.csv`` contains administrative-contact fields (person names,
e-mail, phone, street address). They are dropped at parse time and never written.

Join limitation (documented in the datasheets): drinking-water intake / source-water locations
are not public (security), so a PWS links to rivers only through its service-area polygon
(where water is delivered, not where it is drawn) and, for purchased water, through the
seller PWSID of its consecutive connections. Any upstream-catchment aggregation of SENTINEL
predictions for a PWS is a proxy.
"""

from __future__ import annotations

import io
import logging
import zipfile
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from . import events_common as C

log = logging.getLogger(__name__)

# ============================================================================= URLs
SDWA_ZIP_URL = "https://echo.epa.gov/files/echodownloads/SDWA_latest_downloads.zip"
SDWA_DICTIONARY_URL = "https://echo.epa.gov/tools/data-downloads/sdwa-download-summary"
ENVIROFACTS_VIOLATION_URL = "https://data.epa.gov/dmapservice/sdwis.violation"
UCMR4_ZIP_URL = "https://www.epa.gov/system/files/other-files/2024-04/ucmr4-occurrence-data.zip"
UCMR5_ZIP_URL = "https://www.epa.gov/system/files/other-files/2023-08/ucmr5-occurrence-data.zip"
UCMR_PAGE_URL = "https://www.epa.gov/dwucmr/occurrence-data-unregulated-contaminant-monitoring-rule"
SAB_PAGE_URL = ("https://www.epa.gov/ground-water-and-drinking-water/"
                "community-water-system-service-area-boundaries")
SAB_ZIP_URL = ("https://github.com/USEPA/ORD_SAB_Model/raw/refs/heads/main/Version_History/"
               "PWS_Boundaries_Latest.zip")
SAB_REPO_URL = "https://github.com/USEPA/ORD_SAB_Model"
SAB_FEATURESERVER_URL = ("https://services.arcgis.com/cJ9YHowT8TU7DUyn/arcgis/rest/services/"
                         "Water_System_Boundaries/FeatureServer/0")

# ============================================================================= SDWIS codes
# PRIMARY_SOURCE_CODE (SDWA_REF_CODE_VALUES): GW ground water; GWP ground water purchased;
# SW surface water; SWP surface water purchased; GU groundwater under the (direct) influence of
# surface water (GWUDI — regulated under the surface-water treatment rules); GUP purchased GWUDI.
SURFACE_SOURCE_CODES = ("SW", "SWP", "GU", "GUP")
SOURCE_CODE_MEANING = {
    "GW": "ground water", "GWP": "ground water purchased", "SW": "surface water",
    "SWP": "surface water purchased", "GU": "groundwater under influence of surface water",
    "GUP": "purchased groundwater under influence of surface water source",
}
NITRATE_CODES = {"1040": "Nitrate", "1041": "Nitrite", "1038": "Nitrate-Nitrite"}
DBP_THM_HAA_CODES = {"2950": "TTHM", "2456": "Total Haloacetic Acids (HAA5)"}
DBP_OTHER_CODES = {"1011": "Bromate", "1009": "Chlorite"}
TOC_CODE = "2920"  # "CARBON, TOTAL" — Stage 1 DBPR precursor (TOC) removal
SWTR_RULE_CODES = {"121": "Surface Water Treatment Rule",
                   "122": "Interim and Long Term 1 Enhanced Surface Water Treatment Rule",
                   "123": "Long Term 2 Enhanced Surface Water Treatment Rule"}
DBP_RULE_CODES = {"210": "Stage 1 Disinfectants and Disinfection Byproducts Rule",
                  "220": "Stage 2 Disinfectants and Disinfection Byproducts Rule",
                  "230": "Former Total Trihalomethane Rule"}
NITRATE_RULE_CODE = "331"
TURBIDITY_EXCEEDANCE_CODES = ("43", "44")  # Single / Monthly Turbidity Exceed (Enhanced SWTR)
SWTR_TT_OTHER_CODES = ("41", "42")         # Treatment Technique (SWTR); Failure to Filter (SWTR)
DBP_PRECURSOR_TT_CODE = "46"               # Treatment Technique Precursor Removal
MR_CATEGORIES = ("MR", "MON", "RPT")       # monitoring / reporting violation categories

LABELS = ("nitrate_mcl", "dbp_mcl", "dbp_other_mcl", "dbp_precursor_tt", "turbidity_tt", "swtr_tt_other")
LABEL_DEFINITIONS = {
    "nitrate_mcl": "health-based MCL violation (VIOLATION_CODE 01 single / 02 average), Nitrate Rule "
                   "(RULE_CODE 331), CONTAMINANT_CODE 1040 nitrate, 1041 nitrite or 1038 nitrate-nitrite",
    "dbp_mcl": "health-based MCL violation, RULE_CODE 210/220/230 (Stage 1 RAA / Stage 2 LRAA / former "
               "TTHM rule), CONTAMINANT_CODE 2950 TTHM or 2456 HAA5",
    "dbp_other_mcl": "health-based MCL violation, RULE_CODE 210, CONTAMINANT_CODE 1011 bromate or 1009 chlorite",
    "dbp_precursor_tt": "health-based TT violation 46 'Treatment Technique Precursor Removal' "
                        "(Stage 1 DBPR TOC removal), CONTAMINANT_CODE 2920",
    "turbidity_tt": "health-based TT violation 43 'Single Turbidity Exceed (Enhanced SWTR)' or 44 'Monthly "
                    "Turbidity Exceed (Enhanced SWTR)', RULE_CODE 122/123",
    "swtr_tt_other": "health-based TT violation 41 'Treatment Technique (SWTR and GWR)' or 42 'Failure to "
                     "Filter (SWTR)' under RULE_CODE 121/122/123 (SWTR family; includes pre-2002 turbidity "
                     "performance and disinfection TT failures, which SDWIS does not separate)",
}

# Superset kept by the streaming scan of the 4 GB violations table.
SCAN_RULE_CODES = frozenset(set(SWTR_RULE_CODES) | set(DBP_RULE_CODES) | {NITRATE_RULE_CODE, "130"})
SCAN_CONTAMINANT_CODES = frozenset(set(NITRATE_CODES) | set(DBP_THM_HAA_CODES) | set(DBP_OTHER_CODES)
                                   | {TOC_CODE, "0100", "0200", "0300", "0400", "0500", "0600", "0800"})

# Columns of SDWA_PUB_WATER_SYSTEMS.csv holding personal/contact information — never kept.
PWS_CONTACT_COLUMNS = ("ORG_NAME", "ADMIN_NAME", "EMAIL_ADDR", "PHONE_NUMBER", "PHONE_EXT_NUMBER",
                       "FAX_NUMBER", "ALT_PHONE_NUMBER", "ADDRESS_LINE1", "ADDRESS_LINE2",
                       "CITY_NAME", "ZIP_CODE", "COUNTRY_CODE")
PWS_KEEP_COLUMNS = ("SUBMISSIONYEARQUARTER", "PWSID", "PWS_NAME", "PRIMACY_AGENCY_CODE", "EPA_REGION",
                    "PWS_ACTIVITY_CODE", "PWS_DEACTIVATION_DATE", "PWS_TYPE_CODE", "GW_SW_CODE",
                    "PRIMARY_SOURCE_CODE", "OWNER_TYPE_CODE", "POPULATION_SERVED_COUNT",
                    "POP_CAT_5_CODE", "SERVICE_CONNECTIONS_COUNT", "IS_WHOLESALER_IND",
                    "IS_SCHOOL_OR_DAYCARE_IND", "DBPR_SCHEDULE_CAT_CODE", "LT2_SCHEDULE_CAT_CODE",
                    "CDS_ID", "SOURCE_WATER_PROTECTION_CODE", "FIRST_REPORTED_DATE",
                    "LAST_REPORTED_DATE", "STATE_CODE", "SEASONAL_STARTUP_SYSTEM")
SDWA_DATE_FMT = "%m/%d/%Y"
SDWA_MIN_DATE = "1975-01-01"   # SDWA 1974; National Interim Primary DW Regulations 1975-77
SDWA_OPEN_ENDED = "--->"       # NON_COMPL_PER_END_DATE sentinel for open-ended non-compliance
BOIL_WATER_ENF_CODES = ("SFH", "EFH")  # State / Federal Boil Water Order
# Unit conversion to mg/L for VIOL_MEASURE (only explicit units are converted).
MGL_FACTORS = {"MG/L": 1.0, "UG/L": 1e-3, "NG/L": 1e-6}
# Federal MCLs (mg/L; nitrate/nitrite as N; DBPs as running/locational annual averages) used
# only for the plausibility flag ``viol_measure_qc`` (states may set stricter MCLs).
FEDERAL_MCL_MGL = {"1040": 10.0, "1041": 1.0, "1038": 10.0, "2950": 0.080, "2456": 0.060,
                   "1011": 0.010, "1009": 1.0}
MCL_LABELS = ("nitrate_mcl", "dbp_mcl", "dbp_other_mcl")


def _open_member(src, member: str):
    """(file handle, zipfile-or-None) for a zip member, a plain file path, or a buffer."""
    if isinstance(src, (str, Path)) and str(src).endswith(".zip"):
        zf = zipfile.ZipFile(src)
        return zf.open(member), zf
    if isinstance(src, (str, Path)):
        return open(src, "rb"), None
    return src, None


def read_sdwa_csv(src, member: str | None = None, *, usecols=None, encoding: str = "utf-8") -> pd.DataFrame:
    """One SDWA CSV (zip member or path) as an all-string DataFrame."""
    fh, zf = _open_member(src, member or "")
    try:
        return pd.read_csv(fh, dtype=str, keep_default_na=False, usecols=usecols, encoding=encoding,
                           encoding_errors="replace")
    finally:
        if fh is not src:
            fh.close()
        if zf is not None:
            zf.close()


def scan_violations(src, *, member: str = "SDWA_VIOLATIONS_ENFORCEMENT.csv",
                    chunksize: int = 500_000, rule_codes: Iterable[str] = SCAN_RULE_CODES,
                    contaminant_codes: Iterable[str] = SCAN_CONTAMINANT_CODES) -> pd.DataFrame:
    """Stream the violation x enforcement CSV (zip member, CSV path or buffer); keep rows whose
    RULE_CODE or CONTAMINANT_CODE is in the requested sets. All fields stay strings.
    ``attrs['n_scanned']`` holds the number of rows read."""
    rule_codes, contaminant_codes = set(rule_codes), set(contaminant_codes)
    fh, zf = _open_member(src, member)
    kept, n = [], 0
    try:
        for chunk in pd.read_csv(fh, dtype=str, keep_default_na=False, chunksize=chunksize,
                                 encoding="utf-8", encoding_errors="replace"):
            n += len(chunk)
            m = chunk["RULE_CODE"].isin(rule_codes) | chunk["CONTAMINANT_CODE"].isin(contaminant_codes)
            if m.any():
                kept.append(chunk[m])
            log.info("scanned %d rows, kept %d", n, sum(len(k) for k in kept))
    finally:
        if fh is not src:
            fh.close()
        if zf is not None:
            zf.close()
    out = pd.concat(kept, ignore_index=True) if kept else pd.DataFrame()
    out.attrs["n_scanned"] = n
    return out


def parse_sdwa_dates(values: pd.Series, *, max_date: str | None = None,
                     min_date: str = SDWA_MIN_DATE) -> tuple[pd.Series, pd.Series]:
    """MM/DD/YYYY strings -> (datetime64, flag). Flags as :func:`events_common.clean_dates`
    plus 'open_ended' for the ``--->`` sentinel. Dates before 1975 are calendar errors."""
    s = pd.Series(values, dtype="object")
    d, flag = C.clean_dates(s, fmt=SDWA_DATE_FMT, min_date=min_date, max_date=max_date)
    flag = flag.where(s.astype(str).str.strip() != SDWA_OPEN_ENDED, "open_ended")
    return d, flag


def to_mgl(value: pd.Series, unit: pd.Series) -> pd.Series:
    """VIOL_MEASURE -> mg/L using MGL_FACTORS; blank/other units -> NaN (never guessed)."""
    f = pd.Series(unit, dtype="object").astype(str).str.strip().str.upper().map(MGL_FACTORS)
    return pd.to_numeric(value, errors="coerce") * f


def measure_qc(viol_measure_mgl: pd.Series, contaminant_code: pd.Series, label: pd.Series) -> pd.Series:
    """Plausibility flag for MCL-violation measures (values are never altered):
    'gt_100x_federal_mcl' (likely unit-entry error, e.g. µg/L entered as MG/L),
    'lt_federal_mcl' (below the federal MCL: state MCL, unit error, or averaging), <NA> otherwise."""
    ratio = pd.to_numeric(viol_measure_mgl, errors="coerce").astype(float) / \
        pd.Series(contaminant_code, dtype="object").astype(str).map(FEDERAL_MCL_MGL).astype(float)
    is_mcl = pd.Series(label, dtype="object").astype(str).isin(MCL_LABELS)
    out = np.select([is_mcl & (ratio > 100), is_mcl & (ratio < 1)], ["gt_100x_federal_mcl", "lt_federal_mcl"],
                    default="")
    return pd.Series(out, index=ratio.index, dtype="string").replace("", pd.NA)


def ref_code_map(ref: pd.DataFrame, value_type: str) -> dict[str, str]:
    r = ref[ref["VALUE_TYPE"] == value_type]
    return dict(zip(r["VALUE_CODE"].str.strip(), r["VALUE_DESCRIPTION"].str.strip()))


def tidy_pws_systems(pws: pd.DataFrame) -> pd.DataFrame:
    """SDWA_PUB_WATER_SYSTEMS rows -> tidy systems table (contact fields dropped)."""
    keep = [c for c in PWS_KEEP_COLUMNS if c in pws.columns]
    out = pws[keep].copy()
    out.columns = [c.lower() for c in out.columns]
    out = out.rename(columns={"submissionyearquarter": "submission_year_quarter",
                              "population_served_count": "population_served",
                              "service_connections_count": "service_connections"})
    for c in out.columns:
        out[c] = out[c].astype(str).str.strip()
    out["pwsid"] = C.str_id(out["pwsid"])
    for c in ("population_served", "service_connections"):
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")
    for c in ("pws_deactivation_date", "first_reported_date", "last_reported_date"):
        if c in out:
            out[c], fl = parse_sdwa_dates(out[c])
            out[c + "_flag"] = fl.where(fl != "missing", "")
    for c in ("is_wholesaler_ind", "is_school_or_daycare_ind", "seasonal_startup_system"):
        if c in out:
            out[c] = out[c].map({"Y": True, "N": False}).astype("boolean")
    out["is_surface_source"] = out["primary_source_code"].isin(SURFACE_SOURCE_CODES)
    out["primary_source_desc"] = out["primary_source_code"].map(SOURCE_CODE_MEANING).fillna("")
    out = out.replace({"": pd.NA}).convert_dtypes()
    out["is_surface_source"] = out["is_surface_source"].astype(bool)
    for c in [c for c in out.columns if c.endswith("_flag")]:
        out[c] = out[c].astype("string")
    return out.reset_index(drop=True)


def classify_violations(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``label_group`` (nitrate | dbp | dbp_precursor | swtr | other) and ``label``
    (one of LABELS, or '' for non-label violations such as monitoring/reporting)."""
    rule = df["rule_code"].astype(str)
    vc = df["violation_code"].astype(str)
    cc = df["contaminant_code"].astype(str)
    cat = df["violation_category_code"].astype(str)
    hb = df["is_health_based"].astype(bool)
    group = np.select(
        [(rule == NITRATE_RULE_CODE) & cc.isin(list(NITRATE_CODES)),
         rule.isin(list(DBP_RULE_CODES)) & cc.isin(list(DBP_THM_HAA_CODES) + list(DBP_OTHER_CODES)),
         rule.isin(list(DBP_RULE_CODES)) & (cc == TOC_CODE),
         rule.isin(list(SWTR_RULE_CODES))],
        ["nitrate", "dbp", "dbp_precursor", "swtr"], default="other")
    group = pd.Series(group, index=df.index)
    label = np.select(
        [hb & (group == "nitrate") & (cat == "MCL"),
         hb & (group == "dbp") & cc.isin(list(DBP_THM_HAA_CODES)) & (cat == "MCL"),
         hb & (group == "dbp") & cc.isin(list(DBP_OTHER_CODES)) & (cat == "MCL"),
         hb & (group == "dbp_precursor") & (vc == DBP_PRECURSOR_TT_CODE),
         hb & (group == "swtr") & vc.isin(TURBIDITY_EXCEEDANCE_CODES),
         hb & (group == "swtr") & vc.isin(SWTR_TT_OTHER_CODES)],
        list(LABELS), default="")
    out = df.copy()
    out["label_group"] = group
    out["label"] = label
    out["is_monitoring_reporting"] = cat.isin(MR_CATEGORIES)
    return out


def summarise_enforcement(scan: pd.DataFrame) -> pd.DataFrame:
    """Per (PWSID, VIOLATION_ID): enforcement counts by category, first/last dates, action types."""
    e = scan[["PWSID", "VIOLATION_ID", "ENFORCEMENT_ID", "ENFORCEMENT_DATE",
              "ENFORCEMENT_ACTION_TYPE_CODE", "ENF_ACTION_CATEGORY"]].copy()
    e = e[e["ENFORCEMENT_ID"].astype(str).str.strip() != ""]
    e = e.drop_duplicates(["PWSID", "VIOLATION_ID", "ENFORCEMENT_ID"])
    e["d"], _ = parse_sdwa_dates(e["ENFORCEMENT_DATE"])
    cat = e["ENF_ACTION_CATEGORY"].str.strip()
    e["formal"], e["informal"], e["resolving"] = (cat == "Formal"), (cat == "Informal"), (cat == "Resolving")
    e["boil"] = e["ENFORCEMENT_ACTION_TYPE_CODE"].isin(BOIL_WATER_ENF_CODES)
    g = e.groupby(["PWSID", "VIOLATION_ID"], sort=False)
    out = pd.DataFrame({
        "n_enf_actions": g.size(),
        "n_enf_formal": g["formal"].sum(), "n_enf_informal": g["informal"].sum(),
        "n_enf_resolving": g["resolving"].sum(),
        "first_enf_date": g["d"].min(), "last_enf_date": g["d"].max(),
        "has_boil_water_order": g["boil"].any(),
        "enf_action_types": g["ENFORCEMENT_ACTION_TYPE_CODE"].agg(lambda s: ";".join(sorted(set(s) - {""}))),
    }).reset_index()
    for c in ("n_enf_actions", "n_enf_formal", "n_enf_informal", "n_enf_resolving"):
        out[c] = out[c].astype("int32")
    return out


def tidy_violations(scan: pd.DataFrame, ref: pd.DataFrame | None = None, *,
                    snapshot_date: str | None = None) -> pd.DataFrame:
    """Scanned violation x enforcement rows -> one row per (PWSID, VIOLATION_ID), classified,
    typed and with an enforcement summary. ``snapshot_date`` bounds report dates (default today)."""
    vcols = ["PWSID", "VIOLATION_ID", "FACILITY_ID", "COMPL_PER_BEGIN_DATE", "COMPL_PER_END_DATE",
             "NON_COMPL_PER_BEGIN_DATE", "NON_COMPL_PER_END_DATE", "VIOLATION_CODE",
             "VIOLATION_CATEGORY_CODE", "IS_HEALTH_BASED_IND", "CONTAMINANT_CODE", "VIOL_MEASURE",
             "UNIT_OF_MEASURE", "FEDERAL_MCL", "STATE_MCL", "IS_MAJOR_VIOL_IND", "SEVERITY_IND_CNT",
             "CALCULATED_RTC_DATE", "VIOLATION_STATUS", "PUBLIC_NOTIFICATION_TIER",
             "CALCULATED_PUB_NOTIF_TIER", "VIOL_ORIGINATOR_CODE", "RULE_CODE", "RULE_GROUP_CODE",
             "RULE_FAMILY_CODE", "VIOL_FIRST_REPORTED_DATE", "VIOL_LAST_REPORTED_DATE",
             "SUBMISSIONYEARQUARTER"]
    v = scan[vcols].drop_duplicates(["PWSID", "VIOLATION_ID"]).copy()
    v.columns = [c.lower() for c in v.columns]
    v = v.rename(columns={"submissionyearquarter": "submission_year_quarter"})
    for c in v.columns:
        v[c] = v[c].astype(str).str.strip()
    v["pwsid"] = C.str_id(v["pwsid"])
    v["violation_id"] = C.str_id(v["violation_id"])
    v["facility_id"] = C.str_id(v["facility_id"])
    v["is_health_based"] = v.pop("is_health_based_ind").eq("Y")
    v["is_major_viol"] = v.pop("is_major_viol_ind").map({"Y": True, "N": False}).astype("boolean")
    hi = snapshot_date or C.retrieval_date()
    hi_future = str((pd.Timestamp(hi) + pd.DateOffset(years=5)).date())
    for c, mx in (("compl_per_begin_date", hi), ("compl_per_end_date", hi_future),
                  ("non_compl_per_begin_date", hi), ("non_compl_per_end_date", hi_future),
                  ("calculated_rtc_date", hi_future), ("viol_first_reported_date", hi),
                  ("viol_last_reported_date", hi)):
        v[c], fl = parse_sdwa_dates(v[c], max_date=mx)
        v[c + "_flag"] = fl.where(fl != "missing", "")
    bad_order = v["compl_per_end_date"].notna() & (v["compl_per_end_date"] < v["compl_per_begin_date"])
    v.loc[bad_order, "compl_per_end_date_flag"] = "before_begin"
    v.loc[bad_order, "compl_per_end_date"] = pd.NaT
    v["compl_per_days"] = ((v["compl_per_end_date"] - v["compl_per_begin_date"]).dt.days + 1).astype("Int64")
    v["viol_measure"], v["viol_measure_qualifier"] = C.to_number(v["viol_measure"], missing_codes=())
    v["state_mcl"], _ = C.to_number(v["state_mcl"], missing_codes=())
    v["viol_measure_mgl"] = to_mgl(v["viol_measure"], v["unit_of_measure"])
    for c in ("severity_ind_cnt", "public_notification_tier", "calculated_pub_notif_tier"):
        v[c] = pd.to_numeric(v[c], errors="coerce").astype("Int16")
    v = classify_violations(v)
    v["viol_measure_qc"] = measure_qc(v["viol_measure_mgl"], v["contaminant_code"], v["label"])
    if ref is not None:
        v["rule_name"] = v["rule_code"].map(ref_code_map(ref, "RULE_CODE"))
        v["violation_name"] = v["violation_code"].map(ref_code_map(ref, "VIOLATION_CODE"))
        v["contaminant_name"] = v["contaminant_code"].map(ref_code_map(ref, "CONTAMINANT_CODE"))
    enf = summarise_enforcement(scan).rename(columns={"PWSID": "pwsid", "VIOLATION_ID": "violation_id"})
    enf["pwsid"], enf["violation_id"] = C.str_id(enf["pwsid"]), C.str_id(enf["violation_id"])
    v = v.merge(enf, on=["pwsid", "violation_id"], how="left")
    for c in ("n_enf_actions", "n_enf_formal", "n_enf_informal", "n_enf_resolving"):
        v[c] = v[c].fillna(0).astype("int32")
    v["has_boil_water_order"] = v["has_boil_water_order"].astype("boolean").fillna(False).astype(bool)
    v["enf_action_types"] = v["enf_action_types"].fillna("")
    v = v.replace({"": pd.NA}).convert_dtypes()
    for c in ("is_health_based", "is_monitoring_reporting", "has_boil_water_order"):
        v[c] = v[c].astype(bool)
    for c in [c for c in v.columns if c.endswith(("_flag", "_qualifier"))]:
        v[c] = v[c].astype("string")
    for c in ("n_enf_actions", "n_enf_formal", "n_enf_informal", "n_enf_resolving"):
        v[c] = v[c].astype("int32")
    v["label"] = v["label"].fillna("")
    return v.sort_values(["pwsid", "compl_per_begin_date", "violation_id"]).reset_index(drop=True)


def attach_system_attributes(viol: pd.DataFrame, systems: pd.DataFrame) -> pd.DataFrame:
    cols = ["pwsid", "pws_type_code", "primary_source_code", "is_surface_source", "primacy_agency_code",
            "state_code", "population_served", "pop_cat_5_code", "pws_activity_code"]
    s = systems[[c for c in cols if c in systems.columns]]
    out = viol.merge(s, on="pwsid", how="left")
    out["is_surface_source"] = out["is_surface_source"].fillna(False).astype(bool)
    return out


def _qord(d: pd.Series) -> pd.Series:
    """Datetime -> integer quarter ordinal (year*4 + quarter-1); NaT -> NA."""
    return (d.dt.year * 4 + (d.dt.quarter - 1)).astype("Int64")


def _qstr(o: np.ndarray) -> np.ndarray:
    o = np.asarray(o, dtype=np.int64)
    return np.char.add(np.char.add((o // 4).astype(str), "Q"), (o % 4 + 1).astype(str))


def pws_quarter_labels(viol: pd.DataFrame, systems: pd.DataFrame, *, start: str = "2005Q1",
                       end: str = "2026Q2", snapshot_date: str | None = None,
                       lag_days: int = 365, surface_cws_only: bool = True) -> pd.DataFrame:
    """PWS x calendar-quarter label panel.

    Panel rows: every quarter from max(start, first-reported quarter) to min(end, deactivation
    quarter) for each system (default: community water systems with a surface-water primary
    source SW/SWP/GU/GUP). Flags (int8): ``hb_<label>`` = a health-based violation of that label
    whose compliance period overlaps the quarter; ``hb_<label>_onset`` = its compliance period
    begins in the quarter; ``hb_swtr_tt`` = turbidity_tt OR swtr_tt_other; ``mr_<group>`` =
    a monitoring/reporting violation of that rule group overlaps the quarter (missing-monitoring
    indicator: a 0 label in such a quarter is less certain). ``recent_reporting_lag`` marks
    quarters ending within ``lag_days`` of the snapshot (violations still being reported).
    """
    s = systems.copy()
    if surface_cws_only:
        s = s[(s["pws_type_code"] == "CWS") & s["is_surface_source"]]
    q0, q1 = pd.Period(start, "Q"), pd.Period(end, "Q")
    o0, o1 = q0.year * 4 + q0.quarter - 1, q1.year * 4 + q1.quarter - 1
    first = _qord(pd.to_datetime(s["first_reported_date"])).fillna(o0).astype(int).clip(lower=o0)
    deact = pd.to_datetime(s["pws_deactivation_date"])
    inactive = s["pws_activity_code"].astype(str) != "A"
    last = _qord(deact).where(inactive & deact.notna(), o1).fillna(o1).astype(int).clip(upper=o1)
    n = (last - first + 1).clip(lower=0).to_numpy()
    pw = np.repeat(s["pwsid"].astype(str).to_numpy(), n)
    starts = np.repeat(first.to_numpy(), n)
    offs = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    panel = pd.DataFrame({"pwsid": pw, "q": starts + offs})

    v = viol[viol["pwsid"].astype(str).isin(set(s["pwsid"].astype(str)))].copy()
    v["b"] = _qord(v["compl_per_begin_date"])
    v = v[v["b"].notna()]
    v["e"] = _qord(v["compl_per_end_date"]).fillna(v["b"]).clip(upper=o1)
    v["e"] = v[["b", "e"]].max(axis=1)
    v["key"] = np.where(v["label"].astype(str) != "", "hb_" + v["label"].astype(str),
                        np.where(v["is_monitoring_reporting"] & v["label_group"].isin(["nitrate", "dbp", "swtr"]),
                                 "mr_" + v["label_group"].astype(str), ""))
    v = v[v["key"] != ""]
    reps = (v["e"] - v["b"] + 1).astype(int).to_numpy()
    ex = pd.DataFrame({"pwsid": np.repeat(v["pwsid"].astype(str).to_numpy(), reps),
                       "key": np.repeat(v["key"].to_numpy(), reps),
                       "q": np.repeat(v["b"].astype(int).to_numpy(), reps)
                       + (np.arange(reps.sum()) - np.repeat(np.cumsum(reps) - reps, reps))})
    flags = ex.drop_duplicates().assign(x=1).pivot_table(index=["pwsid", "q"], columns="key", values="x",
                                                          aggfunc="max", fill_value=0)
    onset = v[v["key"].str.startswith("hb_")][["pwsid", "key", "b"]].rename(columns={"b": "q"})
    onset = onset.assign(key=onset["key"] + "_onset", x=1, q=onset["q"].astype(int))
    onset = onset.drop_duplicates().pivot_table(index=["pwsid", "q"], columns="key", values="x",
                                                aggfunc="max", fill_value=0)
    out = panel.merge(flags.reset_index(), on=["pwsid", "q"], how="left") \
               .merge(onset.reset_index(), on=["pwsid", "q"], how="left")
    want = ([f"hb_{lab}" for lab in LABELS] + [f"hb_{lab}_onset" for lab in LABELS]
            + ["mr_nitrate", "mr_dbp", "mr_swtr"])
    for c in want:
        out[c] = out[c].fillna(0).astype("int8") if c in out else np.int8(0)
    out["hb_swtr_tt"] = (out["hb_turbidity_tt"] | out["hb_swtr_tt_other"]).astype("int8")
    out["hb_any"] = out[[f"hb_{lab}" for lab in LABELS]].max(axis=1).astype("int8")
    out["quarter"] = _qstr(out["q"].to_numpy())
    out["quarter_start"] = pd.PeriodIndex(out["quarter"], freq="Q").start_time
    snap = pd.Timestamp(snapshot_date or C.retrieval_date())
    q_end = pd.PeriodIndex(out["quarter"], freq="Q").end_time.normalize()
    out["recent_reporting_lag"] = (snap - q_end).days < lag_days
    attrs = ["pwsid", "state_code", "primacy_agency_code", "primary_source_code", "population_served",
             "pop_cat_5_code", "owner_type_code", "pws_activity_code"]
    out = out.merge(s[[c for c in attrs if c in s.columns]].astype({"pwsid": str}), on="pwsid", how="left")
    order = (["pwsid", "quarter", "quarter_start"] + [c for c in attrs[1:] if c in out.columns]
             + [f"hb_{lab}" for lab in LABELS] + ["hb_swtr_tt", "hb_any"]
             + [f"hb_{lab}_onset" for lab in LABELS] + ["mr_nitrate", "mr_dbp", "mr_swtr",
                                                        "recent_reporting_lag"])
    out = out[order].sort_values(["pwsid", "quarter"]).reset_index(drop=True)
    out["pwsid"] = out["pwsid"].astype("string")
    out["quarter"] = out["quarter"].astype("string")
    return out


def tidy_source_facilities(fac: pd.DataFrame) -> pd.DataFrame:
    """SDWA_FACILITIES -> source facilities (IS_SOURCE_IND = Y): intakes, wells, reservoirs and
    consecutive connections (SELLER_PWSID = upstream wholesaler). No coordinates exist here."""
    f = fac[fac["IS_SOURCE_IND"].astype(str).str.strip() == "Y"].copy()
    keep = ["PWSID", "FACILITY_ID", "FACILITY_NAME", "STATE_FACILITY_ID", "FACILITY_ACTIVITY_CODE",
            "FACILITY_DEACTIVATION_DATE", "FACILITY_TYPE_CODE", "WATER_TYPE_CODE", "AVAILABILITY_CODE",
            "SELLER_TREATMENT_CODE", "SELLER_PWSID", "FILTRATION_STATUS_CODE", "IS_SOURCE_TREATED_IND",
            "FIRST_REPORTED_DATE", "LAST_REPORTED_DATE"]
    f = f[[c for c in keep if c in f.columns]]
    f.columns = [c.lower() for c in f.columns]
    for c in f.columns:
        f[c] = f[c].astype(str).str.strip()
    for c in ("pwsid", "facility_id", "seller_pwsid"):
        f[c] = C.str_id(f[c])
    for c in ("facility_deactivation_date", "first_reported_date", "last_reported_date"):
        f[c], _ = parse_sdwa_dates(f[c])
    f["is_source_treated_ind"] = f["is_source_treated_ind"].map({"Y": True, "N": False}).astype("boolean")
    return f.replace({"": pd.NA}).convert_dtypes().reset_index(drop=True)


def tidy_geographic_areas(geo: pd.DataFrame) -> pd.DataFrame:
    """SDWA_GEOGRAPHIC_AREAS -> areas served (county/city/zip/tribal) per PWS."""
    keep = ["PWSID", "GEO_ID", "AREA_TYPE_CODE", "TRIBAL_CODE", "STATE_SERVED", "ANSI_ENTITY_CODE",
            "ZIP_CODE_SERVED", "CITY_SERVED", "COUNTY_SERVED", "LAST_REPORTED_DATE"]
    g = geo[[c for c in keep if c in geo.columns]].copy()
    g.columns = [c.lower() for c in g.columns]
    for c in g.columns:
        g[c] = g[c].astype(str).str.strip()
    g["pwsid"] = C.str_id(g["pwsid"])
    g["last_reported_date"], _ = parse_sdwa_dates(g["last_reported_date"])
    return g.replace({"": pd.NA}).convert_dtypes().reset_index(drop=True)


# ============================================================================= UCMR 4
UCMR4_CYANOTOXINS = ("total microcystins", "anatoxin-a", "cylindrospermopsin", "microcystin-LA",
                     "microcystin-LF", "microcystin-LR", "microcystin-LY", "microcystin-RR",
                     "microcystin-YR", "nodularin-R")
UCMR4_HAA_GROUPS = ("HAA5", "HAA6Br", "HAA9")
UCMR4_DATE_RANGE = ("2017-01-01", "2021-12-31")
UCMR_SAMPLE_POINT_TYPES = {"EP": "entry point to the distribution system", "DS": "distribution system",
                           "SR": "source water (untreated)"}
UCMR_FACILITY_WATER_TYPES = {"SW": "surface water", "GW": "ground water",
                             "GU": "ground water under the direct influence of surface water",
                             "MX": "any combination of SW, GW and GU"}
# UCMR 4 Data Summary Tables 7-8 (additional data elements)
UCMR4_ADE_CODES = {
    "DisinfectantType": {
        "PEMB": "permanganate", "HPXB": "hydrogen peroxide", "CLGA": "gaseous chlorine",
        "CLOF": "offsite generated hypochlorite", "CLON": "onsite generated hypochlorite",
        "CAGC": "chloramine (gaseous chlorine)", "CAOF": "chloramine (offsite hypochlorite)",
        "CAON": "chloramine (onsite hypochlorite)", "CLDB": "chlorine dioxide", "OZON": "ozone",
        "ULVL": "ultraviolet light", "OTHD": "other disinfectant/oxidant", "NODU": "no disinfectant/oxidant"},
    "DisinfectantResidual": {
        "CL2": "chlorine", "CLO2": "chlorine dioxide", "CLM": "chloramines",
        "CAC": "chlorine and chloramines", "NOD": "no disinfectant residual"},
    "TreatmentInformation": {
        "CON": "conventional", "INF": "in-line filtration", "DFL": "direct filtration", "SFN": "softening",
        "SSF": "slow sand filtration", "GAC": "granular activated carbon", "POB": "pre-oxidation with chlorine",
        "RBF": "river bank filtration", "PSD": "pre-sedimentation", "BIO": "biological filtration",
        "UTR": "unfiltered treatment for surface water", "GWD": "ground water with disinfection only",
        "PAC": "powder activated carbon", "AIR": "air stripping", "MFL": "membrane filtration",
        "IEX": "ionic exchange", "DAF": "dissolved air floatation",
        "CWL": "clear well without aeration", "CWA": "clear well with aeration",
        "ADS": "aeration in distribution system", "OTH": "other treatment", "NTU": "no treatment used",
        "DKN": "do not know"},
    "CyanotoxinBloomOccurrence": {"Yes": "bloom observed near intake", "No": "never seen a bloom",
                                  "DK": "do not know", "NA": "purchased consecutive connection"},
    "CyanotoxinCyanotoxinOccurrence": {"Yes": "cyanotoxins detected in source water",
                                       "No": "never detected", "NS": "unaware of any source-water sampling"},
    "CyanotoxinPossibleBloomTreatment": {"Yes": "treatment changes indicating a bloom", "No": "no changes",
                                         "DK": "do not know"},
    "CyanotoxinPossibleBloomSourceWater": {"Yes": "source-water quality changes", "No": "no changes",
                                           "DK": "do not know"},
}
_TIMING = {"YD": "on sample day", "YW": "within past week", "YM": "past week to past month",
           "YY": "past month to past 12 months", "YP": "more than a year ago"}
UCMR4_ADE_CODES["CyanotoxinBloomOccurrenceDetail"] = dict(_TIMING)
UCMR4_ADE_CODES["CyanotoxinCyanotoxinOccurrenceDetail"] = {
    **_TIMING, "MIC": "microcystins", "CYL": "cylindrospermopsin", "ANA": "anatoxin-a", "SAX": "saxitoxins",
    "OTH": "other", "DK": "do not know"}
UCMR4_ADE_CODES["CyanotoxinPossibleBloomTreatmentDetail"] = {
    "DFR": "decrease in filter runtimes", "ITF": "increase in filtered-water turbidity",
    "ICD": "increased coagulant dose", "TOI": "increase in taste and odor",
    "IOD": "increased oxidant/disinfectant dose", "IDB": "increase in TTHM/HAA5", "OTH": "other"}
UCMR4_ADE_CODES["CyanotoxinPossibleBloomSourceWaterDetail"] = {
    "ITP": "increase in water temperature", "ITU": "increase in turbidity", "IAL": "increase in alkalinity",
    "ITO": "increase in total organic carbon", "ICD": "increase in chlorine demand",
    "IPH": "increase in pH", "DPH": "decrease in pH", "ICA": "increase in chlorophyll a",
    "IPY": "increase in phycocyanin", "INU": "increase in nutrients", "OTH": "other"}


def read_ucmr_txt(src, member: str | None = None) -> pd.DataFrame:
    """A UCMR tab-separated text file (zip member or path) as an all-string DataFrame (latin-1)."""
    if isinstance(src, (str, Path)) and str(src).endswith(".zip"):
        with zipfile.ZipFile(src) as z, z.open(member) as fh:
            return pd.read_csv(fh, sep="\t", dtype=str, keep_default_na=False, encoding="latin-1")
    return pd.read_csv(src, sep="\t", dtype=str, keep_default_na=False, encoding="latin-1")


def tidy_ucmr_results(df: pd.DataFrame, *, date_range: tuple[str, str] = UCMR4_DATE_RANGE) -> pd.DataFrame:
    """UCMR occurrence rows -> tidy results with censoring kept (no DL/2 substitution).

    ``censored`` = AnalyticalResultsSign '<' (result below the UCMR MRL; value NaN,
    ``detection_limit`` = MRL). Units are µg/L in UCMR 4 (stored as 'ug/L').
    """
    d = df.copy()
    for c in d.columns:
        d[c] = d[c].astype(str).str.strip()
    out = pd.DataFrame({
        "pwsid": C.str_id(d["PWSID"]), "pws_name": d["PWSName"], "pws_size": d["Size"],
        "facility_id": C.str_id(d["FacilityID"]), "facility_name": d["FacilityName"],
        "facility_water_type": d["FacilityWaterType"], "sample_point_id": C.str_id(d["SamplePointID"]),
        "sample_point_name": d["SamplePointName"], "sample_point_type": d["SamplePointType"],
        "sample_id": C.str_id(d["SampleID"]), "sample_event_code": d["SampleEventCode"],
        "analyte": d["Contaminant"], "method_id": d["MethodID"],
        "result_sign": d["AnalyticalResultsSign"],
        "monitoring_requirement": d["MonitoringRequirement"], "epa_region": d["Region"], "state": d["State"],
    })
    out["collection_date"], out["collection_date_flag"] = C.clean_dates(
        d["CollectionDate"], fmt="%m/%d/%Y", min_date=date_range[0], max_date=date_range[1])
    out["mrl"], _ = C.to_number(d["MRL"], missing_codes=())
    val, qual = C.to_number(d["AnalyticalResultValue"], missing_codes=())
    out["censored"] = d["AnalyticalResultsSign"].eq("<").to_numpy()
    out["value"] = val.where(~out["censored"])
    out["detection_limit"] = out["mrl"].where(out["censored"])
    out["value_qualifier"] = qual
    unit = d["Units"].str.replace("µ", "u", regex=False).str.replace("μ", "u", regex=False)
    out["unit"] = unit.where(unit != "", pd.NA)
    bad_sign = ~d["AnalyticalResultsSign"].isin(["<", "="])
    out.loc[bad_sign.to_numpy(), "value"] = np.nan
    out["sign_flag"] = np.where(bad_sign, "unknown_sign", "")
    cols = ["pwsid", "pws_name", "pws_size", "state", "epa_region", "facility_id", "facility_name",
            "facility_water_type", "sample_point_id", "sample_point_name", "sample_point_type", "sample_id",
            "sample_event_code", "collection_date", "collection_date_flag", "analyte", "method_id",
            "result_sign", "censored", "value", "detection_limit", "mrl", "unit", "value_qualifier",
            "sign_flag", "monitoring_requirement"]
    out = out[cols].replace({"": pd.NA}).convert_dtypes()
    out["censored"] = out["censored"].astype(bool)
    for c in ("value", "detection_limit", "mrl"):
        out[c] = out[c].astype("float64")
    for c in ("collection_date_flag", "sign_flag", "value_qualifier"):  # <NA> = no problem
        out[c] = out[c].astype("string")
    return out.sort_values(["pwsid", "facility_id", "sample_point_id", "collection_date", "analyte"]) \
              .reset_index(drop=True)


def ucmr4_source_water(zip_path) -> pd.DataFrame:
    """Raw/source-water (SR) TOC and bromide (UCMR4_HAA_Indicators.txt)."""
    return tidy_ucmr_results(read_ucmr_txt(zip_path, "UCMR4_HAA_Indicators.txt"))


def ucmr4_split_all(zip_path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(HAA group results, cyanotoxin results) from UCMR4_All.txt."""
    al = read_ucmr_txt(zip_path, "UCMR4_All.txt")
    haa = tidy_ucmr_results(al[al["Contaminant"].isin(UCMR4_HAA_GROUPS)])
    cy = tidy_ucmr_results(al[al["Contaminant"].isin(UCMR4_CYANOTOXINS)])
    return haa, cy


def tidy_ucmr_ade(df: pd.DataFrame) -> pd.DataFrame:
    """Additional data elements -> decoded long table (one row per element response)."""
    d = df.copy()
    for c in d.columns:
        d[c] = d[c].astype(str).str.strip()
    out = pd.DataFrame({"pwsid": C.str_id(d["PWSID"]), "facility_id": C.str_id(d["FacilityID"]),
                        "sample_point_id": C.str_id(d["SamplePointID"]),
                        "sample_event_code": d["SampleEventCode"], "element": d["AdditionalDataElement"],
                        "response": d["Response"], "if_other_text": d["IfOtherText"]})
    desc = [UCMR4_ADE_CODES.get(e, {}).get(r, "") for e, r in zip(out["element"], out["response"])]
    out["response_desc"] = desc
    return out.replace({"": pd.NA}).convert_dtypes().reset_index(drop=True)


# ============================================================================= service areas
CWS_LAYER = "CWS"
CWS_RENAME = {
    "PWSID": "pwsid", "PWS_Name": "pws_name", "Primacy_Agency": "primacy_agency", "Pop_Cat_5": "pop_cat_5",
    "Population_Served_Count": "population_served", "Service_Connections_Count": "service_connections",
    "Symbology_Field": "boundary_source", "Model_Method": "model_method",
    "Service_Area_Type": "service_area_type", "Original_Data_Provider": "original_data_provider",
    "Data_Provider_Type": "data_provider_type", "Data_Source": "data_source",
    "Modification_Method": "modification_method", "Feature_Type": "feature_type",
    "Method_Details": "method_details", "Verification_Status": "verification_status",
    "Confirmed": "confirmed", "Date_Create": "date_created",
}


def tidy_cws_boundaries(gdf, *, equal_area_crs: str = "EPSG:6933"):
    """CWS layer GeoDataFrame -> tidy GeoDataFrame in EPSG:4326.

    ``boundary_source`` keeps the SAB 'Symbology_Field' ('System Sourced' = state/utility
    supplied; 'Modeled' = EPA ORD model); ``is_modeled`` is derived from it; ``model_method``
    keeps the modelling method (Random Forest, Decision Tree, Parcel, OSM, Census Place).
    ``area_km2`` is computed on an equal-area projection (EPSG:6933). Duplicate PWSIDs are kept
    and flagged. Geometry is not simplified.
    """
    import geopandas as gpd  # local import: geo extra

    g = gdf.rename(columns={k: v for k, v in CWS_RENAME.items() if k in gdf.columns})
    keep = list(CWS_RENAME.values()) + ["geometry"]
    g = g[[c for c in keep if c in g.columns]].copy()
    for c in g.columns:
        if c != "geometry" and g[c].dtype == object:
            g[c] = g[c].astype("string").str.replace(" ", " ", regex=False).str.strip()
    g["pwsid"] = C.str_id(g["pwsid"])
    for c in ("population_served", "service_connections"):
        if c in g:
            g[c] = pd.to_numeric(g[c], errors="coerce").astype("Int64")
    g["is_modeled"] = g["boundary_source"].str.lower().eq("modeled").fillna(False).astype(bool)
    if "date_created" in g:
        g["date_created"], _ = C.clean_dates(g["date_created"], min_date="1990-01-01")
    g["state"] = g["pwsid"].str[:2]
    g["pwsid_duplicated"] = g["pwsid"].duplicated(keep=False).astype(bool)
    if g.crs is not None and g.crs.to_epsg() != 4326:
        g = g.to_crs(4326)
    g["geometry_valid"] = g.geometry.is_valid.astype(bool)
    # Area of an invalid (e.g. self-intersecting) polygon is meaningless and can be negative;
    # measure a make_valid() copy instead. The stored geometry is never repaired.
    area_geom = g.geometry.copy()
    bad = ~g["geometry_valid"]
    if bad.any():
        area_geom.loc[bad] = g.geometry[bad].make_valid()
    g["area_km2"] = area_geom.to_crs(equal_area_crs).area / 1e6
    for c in g.columns:
        if str(g[c].dtype) == "string":
            g[c] = g[c].replace("", pd.NA)
    return gpd.GeoDataFrame(g, geometry="geometry", crs=4326).reset_index(drop=True)


def read_cws_boundaries(gpkg_path, layer: str = CWS_LAYER):
    import pyogrio

    return tidy_cws_boundaries(pyogrio.read_dataframe(gpkg_path, layer=layer))


def tidy_missing_cws(df: pd.DataFrame) -> pd.DataFrame:
    """Missing_CWS_V_3_0.csv (CWS not mapped in SAB v3) -> tidy."""
    out = pd.DataFrame({"pwsid": C.str_id(df["PWS ID"]), "pws_name": df["PWS Name"].str.strip(),
                        "primacy_agency": df["Primacy Agency"].str.strip()})
    out["population_served"] = pd.to_numeric(df["Population Served Count"], errors="coerce").astype("Int64")
    out["service_connections"] = pd.to_numeric(df["Service Connections Count"], errors="coerce").astype("Int64")
    return out.convert_dtypes()


def tidy_census_crosswalk(df: pd.DataFrame, level: str) -> pd.DataFrame:
    """SAB census crosswalk (Tracts / Block_Groups) -> typed table; GEOID20 zero-padded string."""
    width = {"tract": 11, "block_group": 12}[level]
    out = df.copy()
    out["GEOID20"] = C.str_id(out["GEOID20"], width=width)
    out["PWSID"] = C.str_id(out["PWSID"])
    for c in out.columns:
        if c not in ("GEOID20", "PWSID"):
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.rename(columns={"GEOID20": "geoid20", "PWSID": "pwsid"})
    out.columns = [c.lower() for c in out.columns]
    return out


def read_text_member(zip_path, member: str, encoding: str = "utf-8") -> str:
    with zipfile.ZipFile(zip_path) as z:
        return z.read(member).decode(encoding, errors="replace")


def zip_members(zip_path) -> list[dict]:
    with zipfile.ZipFile(zip_path) as z:
        return [{"name": i.filename, "bytes": i.file_size,
                 "date_time": "{:04d}-{:02d}-{:02d}T{:02d}:{:02d}:{:02d}".format(*i.date_time)} for i in z.infolist()]


def csv_from_zip(zip_path, member: str, **kw) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as z:
        return pd.read_csv(io.BytesIO(z.read(member)), dtype=str, keep_default_na=False, **kw)
