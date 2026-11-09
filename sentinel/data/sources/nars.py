# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""EPA National Aquatic Resource Surveys (NARS) client: catalog + file download + tidy parsers.

Source: https://www.epa.gov/national-aquatic-resource-surveys/data-national-aquatic-resource-surveys
(one HTML table, one row per data file: survey | indicator | data link | metadata link).

Surveys handled: NRSA 2008-09, 2013-14, 2018-19, 2023-24 (rivers & streams) and NLA 2007, 2012,
2017, 2022 (lakes). File layouts differ between cycles (wide vs long, column names), so every
parser below maps a cycle's columns onto one tidy schema and keeps the original names in
``source_column`` / ``source_file`` fields where useful.

Pure functions only (request building, HTML/CSV parsing -> DataFrame); downloading is done by
``scripts/data/download_nars.py`` and table building by ``sentinel/data/bio_nars.py``.
"""

from __future__ import annotations

import html as _html
import re
from pathlib import PurePosixPath
from urllib.parse import urljoin, urlparse

import numpy as np
import pandas as pd

from ..bio_common import (
    bool_from_yn,
    first_present,
    id_str,
    nan_if_empty,
    parse_censored,
    read_csv_str,
    to_float,
)

DATA_PAGE_URL = ("https://www.epa.gov/national-aquatic-resource-surveys/"
                 "data-national-aquatic-resource-surveys")
EPA_BASE = "https://www.epa.gov"

SURVEYS = {
    "Rivers and Streams 2008-2009": "nrsa0809",
    "Rivers and Streams 2013-2014": "nrsa1314",
    "Rivers and Streams 2018-2019": "nrsa1819",
    "Rivers and Streams 2023-2024": "nrsa2324",
    "Lakes 2007": "nla07",
    "Lakes 2012": "nla12",
    "Lakes 2017": "nla17",
    "Lakes 2022": "nla22",
}
STREAM_SURVEYS = ["nrsa0809", "nrsa1314", "nrsa1819", "nrsa2324"]
LAKE_SURVEYS = ["nla07", "nla12", "nla17", "nla22"]
CYCLE_LABEL = {"nrsa0809": "2008-09", "nrsa1314": "2013-14", "nrsa1819": "2018-19",
               "nrsa2324": "2023-24", "nla07": "2007", "nla12": "2012", "nla17": "2017",
               "nla22": "2022"}

# Shapefile / bundle archives are not needed for tabular datasets (and the 0809 watershed
# shapefile alone is 128 MB) — except bundles that contain tables published nowhere else:
_SKIP_SUFFIXES = (".zip",)
INCLUDE_ARCHIVES = {
    # NLA 2007 "All Data" bundle: the full water-quality table (ions, nutrient species, flags) is
    # only published inside this zip.
    "nla2007_alldata.zip": ["NLA2007_WaterQuality_20091123.csv", "NLA2007_WaterQuality_Info_20091123.txt"],
}


# ------------------------------------------------------------------------ catalog

def _cell_text(cell: str) -> str:
    t = re.sub(r"<[^>]+>", " ", cell)
    return _html.unescape(re.sub(r"\s+", " ", t)).strip()


def _cell_links(cell: str) -> list[str]:
    links = re.findall(r'href="([^"]+)"', cell)
    return [_html.unescape(h) for h in links if not h.startswith("/themes") and not h.startswith("#")]


def parse_data_page(page_html: str, base: str = EPA_BASE) -> pd.DataFrame:
    """Parse the NARS data hub table into one row per (survey, indicator, data file).

    Columns: survey, survey_code, indicator, label, data_url, meta_url, data_name, meta_name.
    Rows without a downloadable data link (e.g. "Notes" pages) are dropped.
    """
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", page_html, flags=re.DOTALL | re.IGNORECASE)
    out = []
    for r in rows:
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", r, flags=re.DOTALL | re.IGNORECASE)
        if len(cells) < 3:
            continue
        survey, indicator = _cell_text(cells[0]), _cell_text(cells[1])
        if survey.lower() == "survey":
            continue
        dl = _cell_links(cells[2])
        ml = _cell_links(cells[3]) if len(cells) > 3 else []
        if not dl:
            continue
        data_url = urljoin(base, dl[0])
        if not re.search(r"\.(csv|txt|zip|xlsx)$", urlparse(data_url).path, re.IGNORECASE):
            continue
        meta_url = urljoin(base, ml[0]) if ml else None
        out.append({
            "survey": survey,
            "survey_code": SURVEYS.get(survey),
            "indicator": indicator,
            "label": _cell_text(cells[2]),
            "data_url": data_url,
            "meta_url": meta_url,
            "data_name": PurePosixPath(urlparse(data_url).path).name,
            "meta_name": PurePosixPath(urlparse(meta_url).path).name if meta_url else None,
        })
    return pd.DataFrame(out)


def select_files(catalog: pd.DataFrame, survey_codes=None, include_zip: bool = False) -> pd.DataFrame:
    """Rows for the requested surveys, excluding archives unless ``include_zip``."""
    codes = survey_codes or list(SURVEYS.values())
    sel = catalog[catalog["survey_code"].isin(codes)].copy()
    if not include_zip:
        keep_zip = sel["data_name"].isin(list(INCLUDE_ARCHIVES))
        sel = sel[~sel["data_name"].str.lower().str.endswith(_SKIP_SUFFIXES) | keep_zip]
    return sel.reset_index(drop=True)


# ------------------------------------------------------------- per-cycle file roles
# Which raw file plays which role in each survey (file names as published on the hub,
# verified 2026-10-22). ``None`` = not published for that cycle.

FILE_ROLES: dict[str, dict[str, str | list[str] | None]] = {
    "nrsa0809": {
        "siteinfo": "siteinfo_0.csv",
        "benthic_count": "nrsa0809bentcts.csv",
        "benthic_taxa": "nrsa_0809_benttaxa.csv",
        "benthic_mmi": "bentcond.csv",
        "fish_count": "fishcts.csv",
        "fish_taxa": None,
        "fish_mmi": "fishcond.csv",
        "chem": "chem.csv",
        "chem_cond": "chemcond.csv",
        "field_chem": "fieldchemmeasure.csv",
        "phab": "phabmed.csv",
        "phab_cond": "phabcond_0.csv",
        "landscape": "land.csv",
        "algal_toxin": None,
        "periphyton": None,
        "enterococci": "enterocond.csv",
        "popest": None,
    },
    "nrsa1314": {
        "siteinfo": "nrsa1314_siteinformation_wide_04292019.csv",
        "benthic_count": "nrsa1314_bentcnts_04232019.csv",
        "benthic_taxa": "nrsa1314_benttaxa_02012019.csv",
        "benthic_mmi": "nrsa1314_bentmmi_04232019.csv",
        "fish_count": "nrsa1314_fishcts_04232019.csv",
        "fish_taxa": "nrsa1314_fishtaxa_02042019.csv",
        "fish_mmi": "nrsa1314_fishmmi_04232019.csv",
        "chem": "nrsa1314_widechem_04232019.csv",
        "chem_cond": "nrsa1314_chem_04232019.csv",
        "field_chem": "nrsa1314_wide_field_meas_04232019.csv",
        "phab": "nrsa1314_phabmed_04232019.csv",
        "phab_cond": "nrsa1314_phab_04232019.csv",
        "landscape": "nrsa1314_landmet_02132019.csv",
        "algal_toxin": "nrsa1314_micx_04232019.csv",
        "periphyton": ["nrsa1314_widepbio_04232019.csv", "nrsa1314_widepchl_04232019.csv"],
        "water_chla": "nrsa1314_widewchl_04232019.csv",
        "enterococci": "nrsa1314_ente_04232019.csv",
        "popest": "nrsa1314_allcond_05312019_0.csv",
    },
    "nrsa1819": {
        "siteinfo": "NRSA_1819_SiteInfo.csv",
        "benthic_count": "nrsa_1819_benthic_macroinvertebrate_count_-_data.csv",
        "benthic_taxa": None,  # taxonomy and traits are columns of the count file
        "benthic_mmi": "nrsa-1819-benthic-macroinvertebrate-mmi-data.csv",
        "fish_count": "nrsa-1819-fish-count-data.csv",
        "fish_taxa": None,
        "fish_mmi": "nrsa-1819-fish-mmi-data.csv",
        "fish_sampling": "nrsa-1819-fish-sampling-information-data.csv",
        "chem": "nrsa_1819_water_chemistry_chla_-_data.csv",
        "chem_cond": None,
        "field_chem": "nrsa_1819_field_wide.csv",
        "phab": "nrsa_1819_physical_habitat_larger_set_of_metrics_-_data.csv",
        "phab_cond": None,
        "landscape": "NRSA_1819_landscape.csv",
        "algal_toxin": "nrsa_1819_algal_toxin_-_data.csv",
        "periphyton": ["NRSA_1819_pbio_0.csv", "NRSA_1819_PeriChla.csv"],
        "enterococci": "NRSA_1819_enterococci_0.csv",
        "popest": "nrsa1819_data_for_populationestimates.csv",
    },
    "nrsa2324": {
        "siteinfo": "nrsa2324_siteinfo.csv",
        "benthic_count": "nrsa2324_benthiccount.csv",
        "benthic_taxa": "nrsa2324_benthictaxa.csv",
        "benthic_mmi": "nrsa2324_benthicmmi.csv",
        "fish_count": "nrsa2324_fishcount.csv",
        "fish_taxa": "nrsa2324_fishtaxa.csv",
        "fish_mmi": "nrsa2324_fishmmi.csv",
        "chem": "nrsa2324_waterchemistry.csv",
        "chem_cond": None,
        "field_chem": None,
        "phab": "nrsa2324_physicalhabitat_metrics.csv",
        "phab_cond": None,
        "landscape": None,
        "algal_toxin": "nrsa2324_algaltoxin.csv",
        "periphyton": ["nrsa2324_periphytonbiomass.csv", "nrsa2324_periphytonchlorophyll-a.csv"],
        "enterococci": "nrsa2324_enterococci.csv",
        "popest": None,
    },
    "nla07": {
        "siteinfo": "nla2007_sampledlakeinformation_20091113.csv",
        "benthic_count": "nla2007_wide_benthic_08092016.csv",
        "benthic_taxa": None,
        "benthic_mmi": "nla2007_bentcond_08232016.csv",
        "zoop_count": "nla2007_zooplankton_count_20091022.csv",
        "zoop_taxa": None,
        "phyto_count": ["nla2007_phytoplankton_softalgaecount_20091023.csv",
                        "nla2007_phytoplankton_diatomcount_20091125.csv"],
        "phyto_taxa": None,
        "chem": "NLA2007_WaterQuality_20091123.csv",            # extracted from nla2007_alldata.zip
        "algal_toxin": "nla2007_recreational_conditionestimates_20091123.csv",
        "profile": "nla2007_profile_20091008.csv",
        "secchi": "nla2007_secchi_20091008.csv",
        "condition": ["nla2007_trophic_conditionestimate_20091123.csv",
                      "nla2007_plankton_oemodel_conditionestimates_20091125.csv",
                      "nla2007_meando_conditionestimate_20091123.csv",
                      "nla2007_phab_condtionestimates_20091130.csv"],
        "phab": "nla2007_phab_indexvalues.csv",
    },
    "nla12": {
        "siteinfo": "nla2012_wide_siteinfo_08232016.csv",
        "benthic_count": "nla2012_wide_benthic_08232016.csv",
        "benthic_taxa": "nla2012_benttaxa_wide_10272015.csv",
        "benthic_mmi": "nla2012_bentcond_08232016.csv",
        "zoop_count": "nla-2012-zooplankton-count-data-updated-12092021.csv",
        "zoop_taxa": "nla-2012-zooplankton-taxa-list-data-updated-12092021.csv",
        "phyto_count": "nla2012_wide_phytoplankton_count_02122014.csv",
        "phyto_taxa": "nla2012_phytotaxa_wide_10272015.csv",
        "chem": "nla2012_waterchem_wide.csv",
        "chla": "nla2012_chla_wide.csv",
        "algal_toxin": "nla2012_algaltoxins_08192016.csv",
        "atrazine": "nla2012_atrazine_08192016.csv",
        "profile": "nla2012_profile_wide.csv",
        "secchi": "nla2012_secchi_08232016.csv",
        "condition": ["nla_2012_condition_categories.csv", "nla12_keyvariables_data.csv"],
        "zoop_mmi": "nla2012_zoopcond_08192016.csv",
        "phab": "nla2012_wide_phabmet_10202016.csv",
    },
    "nla17": {
        "siteinfo": "nla_2017_site_information-data.csv",
        "benthic_count": "nla_2017_benthic_count-data.csv",
        "benthic_taxa": "nla_2017_benthic_taxa_list-data.csv",
        "benthic_mmi": "nla2017_BentMMI.csv",
        "zoop_count": "nla-2017-zooplankton-count-data.csv",
        "zoop_taxa": "nla-2017-zooplankton-taxa-list-data-updated-12092021.csv",
        "zoop_mmi": "nla2017_zoopMMI.csv",
        "phyto_count": "nla_2017_phytoplankton_count-data.csv",
        "phyto_taxa": "nla_2017_phytoplankton_taxa-data.csv",
        "chem": "nla_2017_water_chemistry_chla-data.csv",
        "algal_toxin": "nla_2017_algal_toxin-data.csv",
        "atrazine": "nla_2017_atrazine-data_0.csv",
        "profile": "nla_2017_profile-data.csv",
        "secchi": "nla_2017_secchi-data.csv",
        "condition": ["nla2017_condition_estimates_20220217_ForWebsite.csv",
                      "nla2017_all_sites-visits_data_for_population_estimates_220705_0.csv"],
        "phab": "nla_2017_phab-data.csv",
    },
    "nla22": {
        "siteinfo": "nla22_siteinfo.csv",
        "benthic_count": "nla22_benthic_counts.csv",
        "benthic_taxa": "nla22_benthic_taxa.csv",
        "benthic_mmi": "nla22_benthic_mmi_0.csv",
        "zoop_count": "nla22_zooplanktoncount_wide.csv",
        "zoop_taxa": "nla22_zooplanktontaxa.csv",
        "zoop_mmi": "nla22_zooplankton_mmi_1.csv",
        "phyto_count": "nla2022_phytoplanktoncount_wide.csv",
        "phyto_taxa": "nla2022_phytoplanktontaxa.csv",
        "chem": "nla22_waterchem_wide.csv",
        "algal_toxin": "nla22_algaltoxins.csv",
        "atrazine": "nla22_atrazine.csv",
        "profile": "nla2022_profile_wide.csv",
        "secchi": "nla22_secchi.csv",
        "condition": ["nla22_condition_combined_2024-08-13_0.csv",
                      "nla2007-2022_data_forpopestimates_indexvisits_probsites_0.csv"],
        "phab": "nla2022_phabmets_wide_0.csv",
    },
}


# ================================================================== tidy parsers

def read_raw(path) -> pd.DataFrame:
    """All-string read with upper-cased, BOM-stripped column names."""
    return read_csv_str(path)


_DATE_FORMATS = ("%m/%d/%Y", "%d-%b-%y", "%Y-%m-%d", "%m/%d/%y", "%d-%b-%Y", "%Y%m%d")


def parse_date(values) -> pd.Series:
    """Parse the mix of date formats used across NARS cycles ('6/20/2018', '5-Aug-08', ISO)."""
    s = pd.Series(values, copy=False).astype("string").str.strip()
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    for fmt in _DATE_FORMATS:
        todo = out.isna() & s.notna() & (s != "")
        if not todo.any():
            break
        out[todo] = pd.to_datetime(s[todo], format=fmt, errors="coerce")
    return out


def _uid_column(df: pd.DataFrame) -> str | None:
    if "UID" in df.columns:
        return "UID"
    # nla2017_zoopMMI.csv ships with a corrupted first header ("7") holding the UID values.
    first = df.columns[0]
    if re.fullmatch(r"\d+", str(first)) and df[first].astype(str).str.fullmatch(r"\d+").mean() > 0.9:
        return first
    return None


def visit_ids(df: pd.DataFrame, survey: str) -> pd.Series:
    """Survey-scoped visit key. UID-based except NLA 2007 (SITE_ID + VISIT_NO; its UIDs are
    not consistent across files)."""
    if survey == "nla07":
        vn = df["VISIT_NO"].astype("string").str.strip().replace("", "1")
        return ("nla07-" + df["SITE_ID"].astype("string").str.strip() + "-" + vn).astype(object)
    col = _uid_column(df)
    if col is None:
        raise KeyError(f"no UID column in {survey} table with columns {list(df.columns)[:10]}")
    uid = id_str(df[col])
    return (survey + "-" + uid.astype("string")).astype(object)


def huc8_str(series: pd.Series) -> pd.Series:
    """NARS HUC8 -> 8-digit string ('H11130301' -> '11130301'; '1010001' -> '01010001')."""
    s = series.astype("string").str.strip().str.upper().str.replace(r"^H", "", regex=True)
    s = s.str.replace(r"\.0+$", "", regex=True)
    s = s.where(s.str.fullmatch(r"\d{7,8}").fillna(False)).str.zfill(8)
    return s.astype(object).where(s.notna(), None)


def huc_fixed(series: pd.Series, width: int) -> pd.Series:
    """Fixed-width HUC codes whose leading zeros were dropped upstream (HUC10/HUC12)."""
    s = series.astype("string").str.strip().str.replace(r"\.0+$", "", regex=True)
    ok = s.str.fullmatch(rf"\d{{{width - 1},{width}}}").fillna(False)
    s = s.where(ok).str.zfill(width)
    return s.astype(object).where(s.notna(), None)


def _col(df: pd.DataFrame, *names):
    c = first_present(df, *names)
    return df[c] if c else pd.Series(None, index=df.index, dtype=object)


REFERENCE_MAP = {"R": "R", "REF": "R", "S": "S", "SO-SO": "S", "T": "T", "TRASH": "T",
                 "LEAST DISTURBED": "R", "MOST DISTURBED": "T", "INTERMEDIATE DISTURBANCE": "S"}


def parse_visits(raw: pd.DataFrame, survey: str) -> pd.DataFrame:
    """One row per site visit with location, ecoregions, hydrologic units and design flags."""
    df = raw
    resource = "streams" if survey.startswith("nrsa") else "lakes"
    date = parse_date(_col(df, "DATE_COL", "DATEPHAB"))
    ref_col = first_present(df, "RT_NRSA", "RT_NLA17", "RT_NLA12", "RT_NLA")
    ref_raw = df[ref_col].astype("string").str.strip() if ref_col else pd.Series(pd.NA, index=df.index)
    out = pd.DataFrame({
        "visit_id": visit_ids(df, survey),
        "survey": survey,
        "cycle": CYCLE_LABEL[survey],
        "resource": resource,
        "uid": id_str(df[_uid_column(df)]) if _uid_column(df) else None,
        "site_id": nan_if_empty(df["SITE_ID"]),
        "unique_id": nan_if_empty(_col(df, "UNIQUE_ID")),
        "master_site_id": nan_if_empty(_col(df, "MASTER_SITEID", "SITEID_07", "NESLAKE_ID")),
        "visit_no": nan_if_empty(_col(df, "VISIT_NO")),
        "date": date.dt.date,
        "year": date.dt.year.astype("Int64"),
        "lat": to_float(_col(df, "LAT_DD83", "LAT_DD")),
        "lon": to_float(_col(df, "LON_DD83", "LON_DD")),
        "state": nan_if_empty(_col(df, "PSTL_CODE", "STATE", "ST")),
        "ag_eco9": nan_if_empty(_col(df, "AG_ECO9", "AGGR_ECO9_2015", "WSA_ECO9")),
        "ag_eco9_name": nan_if_empty(_col(df, "AG_ECO9_NM")),
        "ag_eco3": nan_if_empty(_col(df, "AG_ECO3", "AGGR_ECO3_2015", "WSA_ECO3")),
        "us_l3code": nan_if_empty(_col(df, "US_L3CODE", "US_L3CODE_2015", "ECO_LEV_3")),
        "us_l4code": nan_if_empty(_col(df, "US_L4CODE", "US_L4CODE_2015")),
        "huc8": huc8_str(_col(df, "HUC8", "HUC_8")),
        "huc2_nars": nan_if_empty(_col(df, "HUC2", "HUC_2")),
        "huc10": huc_fixed(_col(df, "HUC10"), 10),
        "huc12": huc_fixed(_col(df, "HUC12"), 12),
        "comid": id_str(_col(df, "COMID", "COMID2012", "COM_ID")),
        "reachcode_raw": nan_if_empty(_col(df, "REACHCODE", "RCHCODE")),
        "site_type": nan_if_empty(_col(df, "SITETYPE", "SITE_CLASS", "SITE_TYPE")),
        "reference_class_raw": ref_raw.astype(object),
        "reference_class": ref_raw.str.upper().map(REFERENCE_MAP).astype(object),
        "reference_source_column": ref_col,
        "index_visit_raw": nan_if_empty(_col(df, "INDEX_VISIT", "INDEX_NLA", "INDEX_SAMP")),
        "elevation_m": to_float(_col(df, "ELEVATION", "ELEV_PT")),
        "eval_cat": nan_if_empty(_col(df, "EVAL_CAT", "EVALSTAT", "LAKE_SAMP", "NRS13_EVAL", "STATUS")),
        "design_weight": to_float(_col(df, "WGT_TP_CORE", "WGT_TP", "WGTNRSA09", "NRS13_WGTD",
                                       "WGT_ALL", "WGT_NLA", "WGT_DSGN")),
        "strahler_order": nan_if_empty(_col(df, "STRAH_ORD", "STRAHLERORDER", "SO")),
        "protocol": nan_if_empty(_col(df, "BOAT_WADE", "VALXSITE", "FLOW_SIZE", "STRAH_CAT")),
        "area_ha": to_float(_col(df, "AREA_HA")),
    })
    # HUC2 derived from HUC8 (NARS HUC2 fields carry sub-region suffixes such as 'H03W').
    out["huc2"] = out["huc8"].astype("string").str[:2].astype(object)
    # REACHCODE is 14 digits; values that went through spreadsheet scientific notation are flagged.
    rc = out["reachcode_raw"].astype("string")
    out["reachcode"] = rc.str.replace(r"^R", "", regex=True).where(
        rc.str.replace(r"^R", "", regex=True).str.fullmatch(r"\d{13,14}").fillna(False)).str.zfill(14).astype(object)
    out["reachcode_mangled"] = rc.str.contains("E+", regex=False).fillna(False)
    samp_col = first_present(df, "SITESAMP", "SAMPLED")
    if samp_col:
        out["sampled"] = df[samp_col].astype("string").str.strip().str.upper().isin(["Y", "YES"]).to_numpy()
    else:
        out["sampled"] = date.notna().to_numpy()
    out["comid_source"] = np.where(out["comid"].notna(), "site_info", None)
    vn = out["visit_no"].astype("string")
    out["within_cycle_revisit"] = (vn.notna() & (vn != "1")).astype(bool)
    idx = out["index_visit_raw"].astype("string").str.upper()
    out["index_visit"] = idx.map({"Y": True, "YES": True, "N": False, "NO": False}).astype("boolean")
    return out


# ----------------------------------------------------------------- community

TAXONOMY_COLUMNS = ["PHYLUM", "SUBPHYLUM", "CLASS", "SUBCLASS", "ORDER", "SUBORDER", "FAMILY", "SUBFAMILY",
                    "TRIBE", "GENUS", "SUBGENUS", "SPECIES", "SUBSPECIES", "DIVISION"]
TRAIT_COLUMNS = ["FFG", "HABIT", "PTV", "NON_TARGET", "ITISTSN", "ALGAL_GROUP", "CLADOCERA_SIZE",
                 "NET_SIZECLS_NEW", "NON_NATIVE", "VOLTINISM", "FAM_OR_CLS", "HABITAT_NRSA", "MIGR_NRSA",
                 "REPROD_NRSA", "TEMP_NRSA", "TOLERANCE_NRSA", "TROPHIC_NRSA", "VEL_NRSA", "TOL_VAL_EMAPW",
                 "OTU_NRSA", "SPECIES_NAME", "SPECIES_EPITHET", "ASSEMBLAGE_NAME"]


def parse_counts(raw: pd.DataFrame, survey: str, assemblage: str, *, source_file: str = "") -> pd.DataFrame:
    """Community long table: one row per (visit, sample type, taxon) with the published counts.

    ``count`` is the full laboratory count (TOTAL / FINAL_CT / COUNT / ABUND / ABUNDANCE);
    ``count_300`` the fixed-count (300-organism) subsample where published. Taxonomy and trait
    columns present in the file are carried through (prefix ``src_``) for the taxonomy table.
    """
    df = raw
    name = _col(df, "TARGET_TAXON", "FINAL_NAME", "TAXANAME")
    out = pd.DataFrame({
        "visit_id": visit_ids(df, survey),
        "survey": survey,
        "assemblage": assemblage,
        "sample_type": nan_if_empty(_col(df, "SAMPLE_TYPE")),
        "taxa_id": id_str(_col(df, "TAXA_ID", "OTU", "LAB_TAXON")),
        "taxon_name": name.astype("string").str.strip().str.upper().astype(object),
        "count": to_float(_col(df, "TOTAL", "FINAL_CT", "COUNT", "ABUNDANCE", "ABUND")),
        "count_300": to_float(_col(df, "TOTAL300", "TOTAL_300", "COUNT_300")),
        "is_distinct": to_float(_col(df, "IS_DISTINCT", "DISTINCT")),
        "is_distinct_300": to_float(_col(df, "IS_DISTINCT300", "IS_DISTINCT_300")),
        # P primary; D/BD/X field duplicates or extra QC samples (kept out of the main tables)
        "sample_category": (_col(df, "SAMPLE_CAT", "SAMPLE_CATEGORY").astype("string").str.strip()
                            .str.upper().replace("", "P").fillna("P")).astype(object),
        "source_file": source_file,
    })
    for c, new in [("BIOMASS", "biomass"), ("BIOMASS_300", "biomass_300"), ("DENSITY", "density"),
                   ("BIOVOLUME", "biovolume"), ("ANOM_CT", "anomaly_count"), ("PROPORTION", "proportion"),
                   ("TOW_VOLUME", "tow_volume")]:
        if c in df.columns:
            out[new] = to_float(df[c])
    if "NON_NATIVE" in df.columns:
        out["non_native"] = bool_from_yn(df["NON_NATIVE"])
    for c in TAXONOMY_COLUMNS + TRAIT_COLUMNS:
        if c in df.columns and c != "NON_NATIVE":
            out["src_" + c.lower()] = nan_if_empty(df[c].astype("string").str.strip().str.upper()
                                                    if c not in ("PTV", "ITISTSN") else df[c])
    # NLA 2007 phytoplankton: soft algae vs diatom count files are separate analyses.
    if survey == "nla07" and assemblage == "phytoplankton":
        out["sample_type"] = "PHYT_DIATOM" if "diatom" in source_file.lower() else "PHYT_SOFTALGAE"
    out = out[out["taxon_name"].notna() & (out["taxon_name"] != "")]
    return out.reset_index(drop=True)


def parse_taxa_table(raw: pd.DataFrame, survey: str, assemblage: str) -> pd.DataFrame:
    """Published taxa list (hierarchy + traits) as a tidy table keyed by (survey, taxa_id)."""
    df = raw
    out = pd.DataFrame({
        "survey": survey,
        "assemblage": assemblage,
        "taxa_id": id_str(_col(df, "TAXA_ID")),
        "taxon_name": _col(df, "TARGET_TAXON", "FINAL_NAME").astype("string").str.strip().str.upper().astype(object),
    })
    for c in TAXONOMY_COLUMNS + TRAIT_COLUMNS:
        if c in df.columns:
            out["src_" + c.lower()] = nan_if_empty(df[c].astype("string").str.strip().str.upper()
                                                    if c not in ("PTV", "ITISTSN") else df[c])
    return out.drop_duplicates().reset_index(drop=True)


# ------------------------------------------------------------------- chemistry

# Source analyte code -> (canonical analyte, canonical unit). Basis is implied by the code
# (TP as P, TN/NH4/NO3/NO2 as N) and stated in docs/datasheets/nars_*.md.
ANALYTES: dict[str, tuple[str, str]] = {
    "PTL": ("tp", "ug/L"), "NTL": ("tn", "mg/L"), "PTL_DISS": ("tdp", "ug/L"), "NTL_DISS": ("tdn", "mg/L"),
    "AMMONIA_N": ("nh4_n", "mg/L"), "NH4": ("nh4_n", "mg/L"), "NITRATE_N": ("no3_n", "mg/L"),
    "NO3": ("no3_n", "mg/L"), "NITRITE_N": ("no2_n", "mg/L"), "NO2": ("no2_n", "mg/L"),
    "NITRATE_NITRITE_N": ("no3no2_n", "mg/L"), "NO3NO2": ("no3no2_n", "mg/L"), "TKN": ("tkn", "mg/L"),
    "CHLORIDE": ("chloride", "mg/L"), "CL": ("chloride", "mg/L"), "SULFATE": ("sulfate", "mg/L"),
    "SO4": ("sulfate", "mg/L"), "COND": ("cond", "uS/cm"), "PH": ("ph_lab", "SU"), "PHLAB": ("ph_lab", "SU"),
    "TURB": ("turbidity", "NTU"), "DOC": ("doc", "mg/L"), "TOC": ("toc", "mg/L"), "TSS": ("tss", "mg/L"),
    "ANC": ("anc", "ueq/L"), "CALCIUM": ("calcium", "mg/L"), "CA": ("calcium", "mg/L"),
    "MAGNESIUM": ("magnesium", "mg/L"), "MG": ("magnesium", "mg/L"), "SODIUM": ("sodium", "mg/L"),
    "POTASSIUM": ("potassium", "mg/L"), "K": ("potassium", "mg/L"), "SILICA": ("silica", "mg/L"),
    "SIO2": ("silica", "mg/L"), "ALUMINUM": ("aluminum", "mg/L"), "AL": ("aluminum", "mg/L"),
    "COLOR": ("color", "PCU"), "CHLA": ("chla", "ug/L"), "CHLX": ("chla", "ug/L"),
    "CHLL": ("chla_littoral", "ug/L"), "PHEO": ("pheophytin", "ug/L"), "MICX": ("microcystin", "ug/L"),
    "MICL": ("microcystin_littoral", "ug/L"), "CYLSPER": ("cylindrospermopsin", "ug/L"),
    # MICZ = "Microcystin in legacy bottle type" (bottle-comparison sample; NRSA 2018-19, NLA 2017)
    "MICZ": ("microcystin_legacy_bottle", "ug/L"), "PERI_CHLA": ("periphyton_chla", "ug/cm2"),
    "AFDM": ("periphyton_afdm", "mg/cm2"),
    "ATRAZINE": ("atrazine", "ug/L"), "MCYST_TL_UGL": ("microcystin", "ug/L"),
    "SECCHI": ("secchi", "m"), "PH_FIELD": ("ph_field", "SU"),
}

# Unit spellings seen in NARS files -> canonical spelling. Conductivity: NARS lab conductivity is
# specific conductance (25 C); files label it 'uS/cm at 25 C', 'uS/cm' or 'uS/cm @ 25 C'. The
# original label is kept in ``unit_source``.
_UNIT_SPELLINGS = {
    "MG/L": "mg/L", "MGN/L": "mg/L", "MGP/L": "mg/L", "MG/LASN": "mg/L", "MGNL": "mg/L",
    "UG/L": "ug/L", "UGP/L": "ug/L", "UGN/L": "ug/L", "UGL": "ug/L",
    # ELISA toxin / atrazine results reported in "ppb": aqueous ppb (w/v) == ug/L (dilute water).
    "PPB": "ug/L",
    "US/CM": "uS/cm", "US/CMAT25C": "uS/cm", "US/CM@25C": "uS/cm", "USCM": "uS/cm", "US/CM25C": "uS/cm",
    "NTU": "NTU", "UEQ/L": "ueq/L", "STD.UNITS": "SU", "SU": "SU", "STDUNITS": "SU", "PH": "SU",
    "UG/CM2": "ug/cm2", "MG/CM2": "mg/cm2",
    "M": "m", "METERS": "m",
    "APHAPT-COUNITS": "PCU", "APHAPT-CO": "PCU", "PCU": "PCU", "PT-CO": "PCU", "PTCO": "PCU",
}
_UNIT_FACTORS = {("mg/L", "ug/L"): 1000.0, ("ug/L", "mg/L"): 0.001}


def canonical_unit(unit) -> str | None:
    if unit is None or (isinstance(unit, float) and np.isnan(unit)):
        return None
    s = re.sub(r"\s+", "", str(unit)).upper().replace("µ", "U").replace("μ", "U")
    if s == "":
        return None
    return _UNIT_SPELLINGS.get(s)


def to_canonical(values: pd.Series, units: pd.Series, analyte: str, target_unit: str) -> tuple[pd.Series, pd.Series]:
    """Convert values to the analyte's canonical unit with a documented factor.

    Returns (converted values, status) where status is 'ok', 'converted', 'assumed_unit'
    (blank unit for pH only) or 'unknown_unit' (value set NaN in the canonical column; the
    original stays in the long table)."""
    cu = units.map(canonical_unit)
    status = pd.Series("ok", index=values.index, dtype=object)
    out = values.astype(float).copy()
    blank = cu.isna() & units.fillna("").astype(str).str.strip().eq("")
    if target_unit == "SU":
        cu = cu.mask(blank, "SU")
        status[blank] = "assumed_unit"
    elif blank.any():
        # Blank unit (typically non-detect rows): use the file's unit for this analyte only when
        # every labelled row maps to one canonical unit.
        labelled = set(cu[~blank].dropna())
        if len(labelled) == 1 and cu[~blank].notna().all():
            cu = cu.mask(blank, labelled.pop())
            status[blank] = "assumed_unit"
    same = cu == target_unit
    for (a, b), f in _UNIT_FACTORS.items():
        if b == target_unit:
            m = cu == a
            out[m] = out[m] * f
            status[m] = "converted"
            same |= m
    keep_status = status.where(same, "unknown_unit")
    status = keep_status
    out[~same] = np.nan
    return out, status


def _nars_flag_censored(flag: pd.Series) -> pd.Series:
    toks = flag.fillna("").astype(str).str.upper().str.replace(" ", "")
    return toks.str.split(r"[;,|]").map(lambda t: "ND" in t).astype(bool)


def _chem_frame(survey, visit_id, analyte_code, result, unit, mdl=None, rl=None, flags=None,
                source_file="", source_column="") -> pd.DataFrame:
    res = parse_censored(result)
    flag = flags if flags is not None else pd.Series("", index=result.index)
    nd = _nars_flag_censored(flag)
    mdl_v = to_float(mdl) if mdl is not None else pd.Series(np.nan, index=result.index)
    rl_v = to_float(rl) if rl is not None else pd.Series(np.nan, index=result.index)
    censored = res["censored"] | nd
    dl = res["detection_limit"].where(res["censored"], mdl_v.where(mdl_v.notna(), rl_v)).where(censored)
    val = res["value"]
    # Non-detects: NARS leaves RESULT blank -> value stays NaN (no substitution at ingest).
    analyte, cunit = ANALYTES.get(analyte_code, (analyte_code.lower(), None))
    out = pd.DataFrame({
        "visit_id": visit_id.to_numpy(), "survey": survey, "analyte": analyte, "source_code": analyte_code,
        "value": val.to_numpy(), "unit_source": unit.astype(object).to_numpy() if unit is not None else None,
        "censored": censored.to_numpy(), "detection_limit": dl.to_numpy(), "mdl": mdl_v.to_numpy(),
        "rl": rl_v.to_numpy(), "flag": flag.fillna("").astype(str).to_numpy(),
        "qualifier": res["qualifier"].to_numpy(), "source_file": source_file, "source_column": source_column,
    })
    if cunit is not None:
        u = out["unit_source"] if unit is not None else pd.Series(cunit, index=out.index)
        conv, status = to_canonical(out["value"], pd.Series(u), analyte, cunit)
        out["value_canonical"] = conv.to_numpy()
        out["unit"] = cunit
        out["unit_status"] = status.to_numpy()
        dlc, _ = to_canonical(out["detection_limit"], pd.Series(u), analyte, cunit)
        out["detection_limit_canonical"] = dlc.to_numpy()
    else:
        out["value_canonical"] = np.nan
        out["unit"] = out["unit_source"]
        out["unit_status"] = "not_harmonised"
        out["detection_limit_canonical"] = np.nan
    # Reported (estimated) values below the method detection limit, e.g. NLA 2012 microcystin,
    # are kept as published and flagged; they are not converted to censored values.
    out["below_mdl"] = (out["value"] < out["mdl"]).fillna(False).to_numpy(bool)
    keep = out["value"].notna() | out["censored"] | (out["qualifier"] != "")
    return out[keep]


def _primary_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep primary/regular water samples (drop lab duplicates, splits, blanks, non-water)."""
    m = pd.Series(True, index=df.index)
    if "SAM_CODE" in df.columns:
        sc = df["SAM_CODE"].astype("string").str.upper().str.strip()
        m &= sc.isin(["REGULAR", ""]) | sc.isna()
    for cc in ("SAMPLE_CAT", "SAMPLE_CATEGORY"):
        if cc in df.columns:
            cat = df[cc].astype("string").str.upper().str.strip()
            m &= cat.isin(["P", ""]) | cat.isna()
    if "MATRIX" in df.columns:
        mx = df["MATRIX"].astype("string").str.upper().str.strip()
        m &= mx.isin(["WATER", ""]) | mx.isna()
    return df[m]


def parse_chem_suffix_wide(raw: pd.DataFrame, survey: str, *, source_file: str = "",
                           analytes: list[str] | None = None) -> pd.DataFrame:
    """Wide files with <ANALYTE>_RESULT, _UNITS/_RESULT_UNITS, _MDL, _RL, _NARS_FLAG/_QA_FLAG/_FLAG."""
    df = _primary_rows(raw)
    vid = visit_ids(df, survey)
    codes = sorted({c[: -len("_RESULT")] for c in df.columns if c.endswith("_RESULT")})
    if analytes is not None:
        codes = [c for c in codes if c in analytes]
    frames = []
    for a in codes:
        unit = first_present(df, f"{a}_UNITS", f"{a}_RESULT_UNITS")
        flags = [c for c in (f"{a}_NARS_FLAG", f"{a}_LAB_FLAG", f"{a}_QA_FLAG", f"{a}_FLAG") if c in df.columns]
        fl = df[flags].astype(str).agg(";".join, axis=1).str.strip(";") if flags else None
        frames.append(_chem_frame(survey, vid, a, df[f"{a}_RESULT"], df[unit] if unit else None,
                                  df.get(f"{a}_MDL"), df.get(f"{a}_RL"), fl, source_file, f"{a}_RESULT"))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def parse_chem_plain_wide(raw: pd.DataFrame, survey: str, columns: dict[str, tuple[str, str]], *,
                          flag_suffix: str | None = None, source_file: str = "") -> pd.DataFrame:
    """Wide files with one column per analyte and units known from the metadata file.

    ``columns`` maps source column -> (ANALYTES code, unit as documented in the metadata)."""
    df = _primary_rows(raw)
    vid = visit_ids(df, survey)
    frames = []
    for col, spec in columns.items():
        if col not in df.columns:
            continue
        code, unit = spec[0], spec[1]
        flag_col = spec[2] if len(spec) > 2 else (f"{col}{flag_suffix}" if flag_suffix else None)
        fl = df[flag_col] if flag_col and flag_col in df.columns else None
        frames.append(_chem_frame(survey, vid, code, df[col], pd.Series(unit, index=df.index), None, None, fl,
                                  source_file, col))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def parse_chem_long(raw: pd.DataFrame, survey: str, *, source_file: str = "", analyte_col: str = "ANALYTE",
                    fixed_analyte: str | None = None) -> pd.DataFrame:
    """Long files with ANALYTE, RESULT, RESULT_UNITS, MDL, RL, NARS_FLAG/QA_FLAG."""
    df = _primary_rows(raw)
    if "NOT_COLLECTED" in df.columns:
        df = df[~df["NOT_COLLECTED"].astype("string").str.upper().isin(["Y", "YES"])]
    vid = visit_ids(df, survey)
    flags = [c for c in ("NARS_FLAG", "QA_FLAG", "LAB_FLAG") if c in df.columns]
    fl = df[flags].astype(str).agg(";".join, axis=1).str.strip(";") if flags else None
    frames = []
    codes = pd.Series(fixed_analyte, index=df.index) if fixed_analyte else \
        df[analyte_col].astype("string").str.strip().str.upper()
    for code, idx in codes.groupby(codes).groups.items():
        if str(code) in ("", "BATCH_ID", "<NA>"):
            continue
        sub = df.loc[idx]
        frames.append(_chem_frame(survey, vid.loc[idx], str(code), sub["RESULT"],
                                  sub[first_present(sub, "RESULT_UNITS", "UNITS")] if first_present(
                                      sub, "RESULT_UNITS", "UNITS") else None,
                                  sub.get("MDL"), sub.get("RL"), fl.loc[idx] if fl is not None else None,
                                  source_file, "RESULT"))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# NRSA 2008-09 chem.csv: units from chem.txt (verified 2026-10-22).
NRSA0809_CHEM_COLUMNS = {
    "AL": ("AL", "mg/L"), "ANC": ("ANC", "ueq/L"), "CA": ("CA", "mg/L"), "CL": ("CL", "mg/L"),
    "COLOR": ("COLOR", "PCU"), "COND": ("COND", "uS/cm"), "DOC": ("DOC", "mg/L"), "K": ("K", "mg/L"),
    "MG": ("MG", "mg/L"), "SODIUM": ("SODIUM", "mg/L"), "NH4": ("NH4", "mg/L"), "NO2": ("NO2", "mg/L"),
    "NO3": ("NO3", "mg/L"), "NO3NO2": ("NO3NO2", "mg/L"), "NTL": ("NTL", "ug/L"), "PHLAB": ("PHLAB", "SU"),
    "PTL": ("PTL", "ug/L"), "SIO2": ("SIO2", "mg/L"), "SO4": ("SO4", "mg/L"), "TSS": ("TSS", "mg/L"),
    "TURB": ("TURB", "NTU"),
}
# NLA 2007 chemical / recreational condition files: units from their *_info_*.txt (verified).
NLA07_CHEM_COLUMNS = {"PTL": ("PTL", "ug/L"), "NTL": ("NTL", "ug/L"), "TURB": ("TURB", "NTU"),
                      "ANC": ("ANC", "ueq/L"), "DOC": ("DOC", "mg/L"), "COND": ("COND", "uS/cm"),
                      "CHLA": ("CHLA", "ug/L")}
NLA07_TOXIN_COLUMNS = {"MCYST_TL_UGL": ("MCYST_TL_UGL", "ug/L")}
# NLA 2007 full water-quality table (NLA2007_WaterQuality_Info_20091123.txt): mg/L columns (*_PPM)
# are used for ions; NH4N_PPM and NO3N_PPM are mg/L as N (checked: ueq/L x 14.007/1000 reproduces
# them). Flags such as "<RL (0.02)" mark values below the reporting limit (value reported).
NLA07_WQ_COLUMNS = {
    "PH_LAB": ("PHLAB", "SU", "PHLAB_FLAG"), "PH_FIELD": ("PH_FIELD", "SU", None),
    "COND": ("COND", "uS/cm", "COND_FLAG"), "ANC": ("ANC", "ueq/L", "ANC_FLAG"),
    "TURB": ("TURB", "NTU", "TURB_FLAG"), "TOC": ("TOC", "mg/L", "TOC_FLAG"), "DOC": ("DOC", "mg/L", "DOC_FLAG"),
    "NH4N_PPM": ("NH4", "mg/L", "NH4_FLAG"), "NO3_NO2": ("NO3NO2", "mg/L", "NO3NO2_FLAG"),
    "NTL": ("NTL", "ug/L", "NTL_FLAG"), "PTL": ("PTL", "ug/L", "PTL_FLAG"), "CL_PPM": ("CL", "mg/L", "CL_FLAG"),
    "NO3N_PPM": ("NO3", "mg/L", "NO3_FLAG"), "SO4_PPM": ("SO4", "mg/L", "SO4_FLAG"),
    "CA_PPM": ("CA", "mg/L", "CA_FLAG"), "MG_PPM": ("MG", "mg/L", "MG_FLAG"),
    "NA_PPM": ("SODIUM", "mg/L", "NA_FLAG"), "K_PPM": ("K", "mg/L", "K_FLAG"),
    "COLOR": ("COLOR", "PCU", "COLOR_FLAG"), "SIO2": ("SIO2", "mg/L", "SIO2_FLAG"),
    "CHLA": ("CHLA", "ug/L", "FLAG_LAB_CHLA"),
}


def parse_secchi(raw: pd.DataFrame, survey: str, *, source_file: str = "") -> pd.DataFrame:
    """Secchi depth (m) = mean of disappearance and reappearance depths (NARS definition; given as
    SECMEAN in 2007 and SECCHI in 2012, computed from DISAPPEARS/REAPPEARS in 2017 and 2022).
    CLEAR_TO_BOTTOM is kept as the flag (a lower bound on transparency)."""
    df = raw
    given = first_present(df, "SECMEAN", "SECCHI")
    if given:
        v = to_float(df[given])
    else:
        d, r = to_float(df["DISAPPEARS"]), to_float(df["REAPPEARS"])
        v = (d + r) / 2
        v = v.where(v.notna(), d.where(d.notna(), r))
    ctb = df["CLEAR_TO_BOTTOM"].astype(str).str.strip().str.upper() if "CLEAR_TO_BOTTOM" in df else ""
    flag = pd.Series(np.where(pd.Series(ctb).isin(["Y", "YES"]), "CLEAR_TO_BOTTOM", ""), index=df.index)
    return _chem_frame(survey, visit_ids(df, survey), "SECCHI", v.astype(str).replace("nan", ""),
                       pd.Series("m", index=df.index), None, None, flag, source_file,
                       given or "mean(DISAPPEARS, REAPPEARS)")
# NRSA 2013-14 microcystin file: MICX_RESULT etc. (handled by the suffix parser).


def parse_nla07_microcystin(raw: pd.DataFrame, survey: str = "nla07", *, source_file: str = "") -> pd.DataFrame:
    """NLA 2007 total microcystin (recreational condition file). Lakes below the 0.1 ug/L reporting
    limit were published as 0.05 ug/L (= RL/2, a substitution; MCYST_PRES '1:LESS THAN RL (0.1
    ug/L)'). Those rows become censored: value NaN, detection_limit 0.1, original value kept in
    ``qualifier``."""
    c = parse_chem_plain_wide(raw, survey, NLA07_TOXIN_COLUMNS, source_file=source_file)
    if c.empty or "MCYST_PRES" not in raw.columns:
        return c
    pres = pd.Series(raw["MCYST_PRES"].astype(str).str.upper().to_numpy(), index=visit_ids(raw, survey).to_numpy())
    less = c["visit_id"].map(pres).fillna("").str.contains("LESS THAN RL").to_numpy()
    c.loc[less, "qualifier"] = "published " + c.loc[less, "value"].astype(str) + " = RL/2 substitution"
    c.loc[less, "value"] = np.nan
    c.loc[less, "value_canonical"] = np.nan
    c.loc[less, "censored"] = True
    c.loc[less, "detection_limit"] = 0.1
    c.loc[less, "detection_limit_canonical"] = 0.1
    c.loc[less, "rl"] = 0.1
    c.loc[less, "flag"] = "LESS_THAN_RL"
    return c


def parse_field_measures(raw: pd.DataFrame, survey: str, *, source_file: str = "") -> pd.DataFrame:
    """In-situ field measurements (DO, temperature, pH, conductivity): median over the rows of a
    visit (several transects/locations in some cycles); calibration rows are excluded."""
    df = raw
    if "SAMPLE_TYPE" in df.columns:
        df = df[~df["SAMPLE_TYPE"].astype("string").str.upper().str.contains("CALIB", na=False)]
    vid = visit_ids(df, survey)
    spec = {"DO": ("do_field", "mg/L"), "TEMPERATURE": ("temp_field", "degC"), "PH": ("ph_field", "SU"),
            "CONDUCTIVITY": ("cond_field", "uS/cm")}
    frames = []
    for col, (analyte, unit) in spec.items():
        c = first_present(df, col, col.lower(), col.capitalize())
        if c is None:
            continue
        v = to_float(df[c])
        g = pd.DataFrame({"visit_id": vid, "v": v}).dropna().groupby("visit_id")["v"].agg(["median", "size"])
        frames.append(pd.DataFrame({
            "visit_id": g.index, "survey": survey, "analyte": analyte, "source_code": col,
            "value": g["median"].to_numpy(), "unit_source": unit, "censored": False, "detection_limit": np.nan,
            "mdl": np.nan, "rl": np.nan, "flag": "", "qualifier": "", "source_file": source_file,
            "source_column": c, "value_canonical": g["median"].to_numpy(), "unit": unit, "unit_status": "ok",
            "detection_limit_canonical": np.nan, "n_field_rows": g["size"].to_numpy()}))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ------------------------------------------------------------------- condition

CONDITION_SCORE_COLUMNS = {
    "MMI_BENT": "bent_mmi", "MMI_BENT_NLA12": "bent_mmi", "OE_SCORE": "bent_oe", "MMI_FISH": "fish_mmi",
    "MMI_RF_FISH_EHIGH": "fish_mmi", "MMI_RF_FISH_PLNLOW": "fish_mmi", "MMI_RF_FISH_WMTNS": "fish_mmi",
    "MMI_ZOOP_2017": "zoop_mmi", "MMI_ZOOP_NLA6": "zoop_mmi", "OE5": "plankton_oe",
}
CONDITION_CLASS_COLUMNS = {
    "BENT_MMI_COND": "bent_mmi_cond", "BENT_COND": "bent_mmi_cond", "BENT_MMI_COND_2017": "bent_mmi_cond",
    "OE_COND": "bent_oe_cond", "FISH_MMI_COND": "fish_mmi_cond", "FISH_MMI_RF_COND": "fish_mmi_cond",
    "ZOOP_MMI_COND_2017": "zoop_mmi_cond", "ZOOP_COND": "zoop_mmi_cond", "OE5_COND": "plankton_oe_cond",
    "PTL_COND": "tp_cond", "NTL_COND": "tn_cond", "SAL_COND": "salinity_cond", "ACID_COND": "acid_cond",
    "ANC_COND": "anc_cond", "ACID_CLS": "acid_cond", "CHLA_COND": "chla_cond", "TURB_COND": "turbidity_cond",
    "SALINITY_COND": "salinity_cond", "TROPHIC_STATE": "trophic_state", "DIS_O2_CLS": "do_cond",
    "DO2_COND": "do_cond", "BEDSED_COND": "bedsed_cond", "INSTRMCVR_COND": "instream_cover_cond",
    "RIPVEG_COND": "riparian_veg_cond", "RIPDIST_COND": "riparian_disturbance_cond",
    "MICX_EPA_COND": "microcystin_cond", "MICX_COND": "microcystin_cond", "MCYST_COND": "microcystin_cond",
    "MICX_DETECT": "microcystin_detect", "CYLSPER_EPA_COND": "cylindrospermopsin_cond",
    "CYLSPER_DETECT": "cylindrospermopsin_detect", "HG_COND": "fish_tissue_hg_cond",
    "ENT_1X_STV_COND": "enterococci_cond", "DRAWDOWN_COND": "drawdown_cond", "RDIS_COND": "lake_riparian_disturbance_cond",
    "LITRIPCVR_COND": "littoral_riparian_cover_cond", "LITCVR_COND": "littoral_cover_cond",
    "RVEG_COND": "lake_riparian_veg_cond", "SALINITY_CLASS": "salinity_class", "ATZ_DETECT": "atrazine_detect",
    "CYANO_COND": "cyanobacteria_cond",
}


def parse_condition(raw: pd.DataFrame, survey: str, *, source_file: str = "",
                    survey_from_design_cycle: bool = False) -> pd.DataFrame:
    """Condition labels (class + score) in long form: one row per (visit, indicator)."""
    df = raw
    if survey_from_design_cycle:
        cyc = df["DSGN_CYCLE"].astype("string").str.strip()
        rev = {"2007": "nla07", "2012": "nla12", "2017": "nla17", "2022": "nla22"}
        parts = [parse_condition(df[cyc == k], v, source_file=source_file) for k, v in rev.items()
                 if (cyc == k).any()]
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    vid = visit_ids(df, survey)
    rows = []
    for col, ind in CONDITION_CLASS_COLUMNS.items():
        if col in df.columns:
            v = df[col].astype("string").str.strip()
            v = v.mask(v.isin(["", "NA", "Not Assessed", "None"]))
            rows.append(pd.DataFrame({"visit_id": vid, "indicator": ind, "class": v.astype(object),
                                      "score": np.nan, "source_column": col}))
    for col, ind in CONDITION_SCORE_COLUMNS.items():
        if col in df.columns:
            rows.append(pd.DataFrame({"visit_id": vid, "indicator": ind, "class": None,
                                      "score": to_float(df[col]), "source_column": col}))
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out = out[out["class"].notna() | out["score"].notna()]
    out["survey"] = survey
    out["source_file"] = source_file
    return out.reset_index(drop=True)


# -------------------------------------------------------------- habitat / profiles

_ID_LIKE = {"PUBLICATION_DATE", "UID", "SITE_ID", "DATE_COL", "VISIT_NO", "AG_ECO9", "AG_ECO3", "EPA_REG",
            "STATE", "YEAR", "UNIQUE_ID", "COMID", "PROTOCOL", "PSTL_CODE", "LAT_DD83", "LON_DD83", "SITETYPE",
            "SITE_CLASS", "VALXSITE", "USE_NRSA", "XLAT_DD", "XLON_DD", "STRAHLERORDER", "GREAT_RIVER", "FW_ECO3",
            "FW_ECO9", "ECO10", "STRAH_ORD", "AGGR_ECO3_2015", "AGGR_ECO9_2015", "ECO_BIO", "STUDY", "STATION",
            "SAMPLE_TYPE", "ST", "SITE_TYPE", "LAKE_SAMP", "SAMPLED_PHAB", "DATEPHAB", "WSA_ECO3", "WSA_ECO9",
            "CLASSP", "ECOP6", "AREA_HA", "BANGMODE", "REALM", "SURVEY_YRS"}


def parse_phab_metrics(raw: pd.DataFrame, survey: str, *, source_file: str = "") -> pd.DataFrame:
    """Physical-habitat metric file -> long numeric table (visit, metric, value)."""
    df = raw
    if "STATION" in df.columns:  # NLA 2017 file is per station, not per visit: skip (no visit metrics)
        return pd.DataFrame()
    vid = visit_ids(df, survey)
    cols = [c for c in df.columns if c not in _ID_LIKE and not c.endswith("_FLAG") and not c.endswith("_COMMENT")]
    frames = []
    for c in cols:
        v = to_float(df[c])
        if v.notna().mean() < 0.05:
            continue
        frames.append(pd.DataFrame({"visit_id": vid, "metric": c, "value": v}).dropna(subset=["value"]))
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not out.empty:
        out["survey"] = survey
        out["source_file"] = source_file
        out = out.drop_duplicates(["visit_id", "metric"])
    return out


def parse_profile(raw: pd.DataFrame, survey: str, *, source_file: str = "") -> pd.DataFrame:
    """Lake depth profiles (temperature, DO, pH, conductivity by depth)."""
    df = raw
    if "SAMPLE_TYPE" in df.columns:
        st = df["SAMPLE_TYPE"].astype("string").str.upper()
        df = df[st.isna() | st.isin(["PROF", ""])]
    vid = visit_ids(df, survey)
    out = pd.DataFrame({
        "visit_id": vid, "survey": survey,
        "depth_m": to_float(_col(df, "DEPTH")),
        "temperature_c": to_float(_col(df, "TEMPERATURE", "TEMP_FIELD")),
        "do_mgl": to_float(_col(df, "OXYGEN", "DO_FIELD")),
        "ph": to_float(_col(df, "PH", "PH_FIELD")),
        "conductivity_uscm": to_float(_col(df, "CONDUCTIVITY", "COND_FIELD")),
        "metalimnion": nan_if_empty(_col(df, "METALIMNION")),
        "flag": nan_if_empty(_col(df, "OXYGEN_NARS_FLAG", "FLAG_PROFILE", "FLAG")),
        "source_file": source_file,
    })
    keep = out[["temperature_c", "do_mgl", "ph", "conductivity_uscm"]].notna().any(axis=1)
    return out[keep].reset_index(drop=True)
