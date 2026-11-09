# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""UK Environment Agency (England) ecology archive + Water Quality Archive (WQA) client.

Ecology & Fish Data Explorer bulk files (Open Government Licence v3.0), verified 2026-10-22:
    https://environment.data.gov.uk/ecology/explorer/downloads/
    INV_OPEN_DATA.zip        freshwater macroinvertebrates (SITE / METRICS / TAXA csv)
    DIAT_OPEN_DATA.zip       diatoms, light microscopy (LABP) and DNA metabarcoding (DNAP)
    OPEN_DATA_TAXON_INFO.zip taxon dictionary keyed by TAXON_LIST_ITEM_KEY (NBN keys), with
                             parent links (PARENT_TLIK) -> full hierarchy

Water Quality Archive, new API (OpenAPI 3.1 "Water Quality Archive" v1.3.1), verified 2026-10-22:
    https://environment.data.gov.uk/water-quality/api/swagger        (spec)
    GET  /water-quality/sampling-point            JSON-LD, <=250 per page, BNG by default
    POST /water-quality/data/observation          CSV, <=2500 per page, <=100 points per call,
                                                  <=1 calendar-year range for multi-point calls,
                                                  query-string filters only (a JSON body {} is
                                                  rejected with "Invalid query parameters")
    Empty results are HTTP 204. Archive coverage starts in 2000.
    Censored results are reported as strings such as "<0.03"; coded results as text.

Pure functions: build request -> parse -> tidy DataFrame. Network calls live in
``scripts/data/download_uk_ea.py``.
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from ..bio_common import id_str, parse_censored, read_csv_str

ECOLOGY_DOWNLOADS_URL = "https://environment.data.gov.uk/ecology/explorer/downloads/"
ECOLOGY_FILES = ("INV_OPEN_DATA.zip", "DIAT_OPEN_DATA.zip", "OPEN_DATA_TAXON_INFO.zip")
WQ_BASE = "https://environment.data.gov.uk/water-quality"
WQ_SPEC_URL = WQ_BASE + "/api/swagger"
WQ_SAMPLING_POINT_URL = WQ_BASE + "/sampling-point"
WQ_OBSERVATION_URL = WQ_BASE + "/data/observation"
WQ_POINT_TYPES_URL = WQ_BASE + "/api/proxy/codelist/sampling-point-type"
WQ_DETERMINAND_URL = WQ_BASE + "/codelist/determinand"
WQ_FIRST_YEAR = 2000
LICENSE = "Open Government Licence v3.0 (contains Environment Agency information © Environment Agency)"

# Freshwater sampling-point types used for pairing with river/canal invertebrate sites.
# (F6 rivers, FJ canals, FK land drains, FZ unspecified freshwater.) Lakes (FA), bathing
# waters (FL), augmentation/transfer/comparative-inlet points (FD/FB/FC) are excluded.
WQ_PAIR_POINT_TYPES = ("F6", "FJ", "FK", "FZ")

# Core determinands (EA determinand notation -> short name). Units come from the data
# (``unit`` column) and are part of a variable's identity; 0062/0077 (conductivity at 20 C vs
# 25 C) and 0068/6396 (turbidity) are kept as separate variables, never merged.
DETERMINANDS = {
    "0061": "ph",
    "0077": "cond25",
    "0062": "cond20",
    "0076": "temp_water",
    "0111": "ammoniacal_n",
    "0116": "ton_n",
    "0117": "nitrate_n",
    "0118": "nitrite_n",
    "0180": "orthophosphate_p",
    "0348": "total_p",
    "9924": "do_mgl",
    "9901": "do_pctsat",
    "0085": "bod5_atu",
    "0135": "ss_105c",
    "0172": "chloride",
    "0162": "alkalinity_ph45",
    "0183": "sulphate",
    "0068": "turbidity_0068",
    "6396": "turbidity_6396",
    "0158": "hardness_total",
}

_DATE_DMY = "%d/%m/%Y"


# --------------------------------------------------------------------- ecology files

def parse_ecology_downloads_page(page_html: str) -> list[str]:
    """File names linked from the Ecology & Fish Data Explorer downloads page."""
    return sorted(set(re.findall(r'href="([A-Za-z0-9_]+\.zip)"', page_html)))


def _zip_member(zf: zipfile.ZipFile, suffix: str) -> str:
    names = [n for n in zf.namelist() if n.upper().endswith(suffix.upper())]
    if len(names) != 1:
        raise KeyError(f"expected one member ending {suffix} in {zf.filename}, got {names}")
    return names[0]


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype("string").str.strip().replace("", pd.NA), errors="coerce")


def _date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s.astype("string").str.strip(), format=_DATE_DMY, errors="coerce").dt.date


def tidy_sites(raw: pd.DataFrame, assemblage: str) -> pd.DataFrame:
    """Site table (one row per SITE_ID) with BNG coordinates and site-level descriptors."""
    df = raw.copy()
    out = pd.DataFrame({
        "site_id": id_str(df["SITE_ID"]),
        "site_version": _num(df["SITE_VERSION"]).astype("Int64") if "SITE_VERSION" in df else pd.NA,
        "assemblage": assemblage,
        "agency_area": df.get("AGENCY_AREA"),
        "reporting_area": df.get("REPORTING_AREA"),
        "catchment": df.get("CATCHMENT"),
        "waterbody_type": df.get("WATERBODY_TYPE"),
        "waterbody_type_description": df.get("WATERBODY_TYPE_DESCRIPTION"),
        "water_body": df.get("WATER_BODY"),
        "wfd_waterbody_id": id_str(df["WFD_WATERBODY_ID"]) if "WFD_WATERBODY_ID" in df else None,
        "ngr_10_fig": df.get("NGR_10_FIG"),
        "easting_bng": _num(df["FULL_EASTING"]),
        "northing_bng": _num(df["FULL_NORTHING"]),
    })
    for c, unit in [("ALTITUDE", "m"), ("SLOPE", "m_per_km"), ("DIST_FROM_SOURCE", "km"),
                    ("DISCHARGE", "category"), ("WIDTH", "m"), ("DEPTH", "cm"),
                    ("BOULDERS_COBBLES", "pct"), ("PEBBLES_GRAVEL", "pct"), ("SAND", "pct"),
                    ("SILT_CLAY", "pct"), ("ALKALINITY", "mg_per_l_caco3"),
                    ("CONDUCTIVITY", "us_per_cm"), ("TOTAL_HARDNESS", "mg_per_l_caco3"),
                    ("CALCIUM", "mg_per_l")]:
        if c in df:
            out[f"site_{c.lower()}_{unit}"] = _num(df[c])
    for c in ("BASE_DATA_DATE", "MIN_SAMPLE_DATE", "MAX_SAMPLE_DATE"):
        if c in df:
            out[c.lower()] = _date(df[c])
    if "COUNT_OF_SAMPLES" in df:
        out["count_of_samples"] = _num(df["COUNT_OF_SAMPLES"]).astype("Int64")
    ecn = [c for c in df.columns if c.startswith("ECN_SITE")]
    if ecn:
        out["ecn_site"] = df[ecn[0]].astype("string").str.upper().eq("TRUE")
    return out


def bng_to_wgs84(easting, northing):
    """OSGB36 / British National Grid (EPSG:27700) -> WGS84 lon/lat (EPSG:4326)."""
    from pyproj import Transformer

    tr = Transformer.from_crs(27700, 4326, always_xy=True)
    e = np.asarray(easting, float)
    n = np.asarray(northing, float)
    lon, lat = tr.transform(e, n)
    lon = np.where(np.isfinite(e) & np.isfinite(n), lon, np.nan)
    lat = np.where(np.isfinite(e) & np.isfinite(n), lat, np.nan)
    return lon, lat


INDEX_COLUMNS_INV = ["BMWP_N_TAXA", "BMWP_TOTAL", "BMWP_ASPT", "CCI_N_TAXA", "CCI_CS_TOTAL", "CCI_ASPT",
                     "CSMAX_COS", "BMWP_COS", "CCI_COS", "CCI", "DEHLI_N_TAXA", "DIS_TOTAL", "DEHLI",
                     "EPSI_ML_S_GRP", "EPSI_ML_ALL_GRP", "EPSI_MIXED_LEVEL_SCORE", "EPSI_S_GRP",
                     "EPSI_ALL_GRP", "EPSI_FAMILY_SCORE", "LIFE_N_TAXA", "LIFE_SCORES_TOTAL",
                     "LIFE_FAMILY_INDEX", "LIFE_SPECIES_N_TAXA", "LIFE_SPECIES_SCORES_TOTAL",
                     "LIFE_SPECIES_INDEX", "PSI_ML_AB", "PSI_ML_ABCD", "PSI_MIXED_LEVEL_SCORE", "PSI_AB",
                     "PSI_ABCD", "PSI_FAMILY_SCORE", "WHPT_N_TAXA", "WHPT_TOTAL", "WHPT_ASPT",
                     "WHPT_NW_N_TAXA", "WHPT_NW_TOTAL", "WHPT_NW_ASPT"]
INDEX_COLUMNS_DIAT = ["TOTAL_UNITS_FOUND_LM", "TDI4", "TOTAL_TDI4_CELLS", "TDI4_PERCENT_PLANKTONIC",
                      "TDI4_PERCENT_MOTILE", "TDI4_PERCENT_PTV", "TDI4_PERCENT_SALINITY", "TDI5LM",
                      "TOTAL_TDI5LM_CELLS", "TDI5_LM_PERCENT_PLANKTONIC", "TDI5_LM_PERCENT_MOTILE",
                      "TDI5_LM_PERCENT_PTV", "TDI5_LM_PERCENT_SALINITY", "TOTAL_NGS_SEQUENCE_READS",
                      "TOTAL_TDI5_NGS_SEQUENCE_READS", "TDI5_NGS_PERCENT_PLANKTONIC",
                      "TDI5_NGS_PERCENT_MOTILE", "TDI5_NGS_PERCENT_PTV", "TDI5_NGS_PERCENT_SALINITY"]


def tidy_metrics(raw: pd.DataFrame, assemblage: str) -> pd.DataFrame:
    """One row per analysis (SAMPLE_ID x ANALYSIS_ID) with sample metadata and published indices.

    The indices (BMWP, WHPT, LIFE, PSI, DEHLI, CCI, TDI ...) are deterministic functions of the
    taxa list of the same analysis — never use them as targets for models fed those taxa.
    """
    df = raw.copy()
    out = pd.DataFrame({
        "site_id": id_str(df["SITE_ID"]),
        "sample_id": id_str(df["SAMPLE_ID"]),
        "sample_version": _num(df["SAMPLE_VERSION"]).astype("Int64"),
        "analysis_id": id_str(df["ANALYSIS_ID"]),
        "assemblage": assemblage,
        "replicate_code": df["REPLICATE_CODE"].replace("", None),
        "sample_date": _date(df["SAMPLE_DATE"]),
        "date_of_analysis": _date(df["DATE_OF_ANALYSIS"]),
        "sample_type": df["SAMPLE_TYPE"],
        "sample_method": df["SAMPLE_METHOD"],
        "sample_method_description": df["SAMPLE_METHOD_DESCRIPTION"],
        "sample_reason": df["SAMPLE_REASON"].str.strip(),
        "analysis_type": df["ANALYSIS_TYPE"],
        "analysis_method": df["ANALYSIS_METHOD"],
        "analysis_method_description": df["ANALYSIS_METHOD_DESCRIPTION"],
        "is_third_party_data": df["IS_THIRD_PARTY_DATA"].str.strip().str.upper().eq("YES"),
        "waterbody_type": df["WATERBODY_TYPE"],
    })
    cols = INDEX_COLUMNS_INV if assemblage == "invertebrate" else INDEX_COLUMNS_DIAT
    for c in cols:
        if c in df.columns:
            out["idx_" + c.lower()] = _num(df[c])
    return out


def tidy_taxa_chunk(raw: pd.DataFrame, assemblage: str) -> pd.DataFrame:
    """Long community table chunk: one row per (analysis, taxon) with the reported abundance."""
    df = raw
    if assemblage == "invertebrate":
        abund = _num(df["TOTAL_ABUNDANCE"])
        out = pd.DataFrame({
            "site_id": id_str(df["SITE_ID"]),
            "sample_id": id_str(df["SAMPLE_ID"]),
            "analysis_id": id_str(df["ANALYSIS_ID"]),
            "analysis_method": df["ANALYSIS_METHOD"],
            "taxon_list_item_key": df["TAXON_LIST_ITEM_KEY"].str.strip(),
            # TOTAL_ABUNDANCE as published: actual counts for ANAA analyses, but largely class
            # representative values (e.g. 3, 33, 333) for log/local-abundance analyses (ANLA, ANLE).
            "abundance": abund.astype("float32"),
            "abundance_category": df["ABUNDANCE_CATEGORY"].replace("", None),
        })
        # BT001 log-abundance class (A 1-9, B 10-99, C 100-999, D 1000-9999, E >=10000): the only
        # abundance measure comparable across EA analysis methods.
        out["log_abundance_class"] = out["abundance_category"].map(
            {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}).astype("Int8")
    else:
        out = pd.DataFrame({
            "site_id": id_str(df["SITE_ID"]),
            "sample_id": id_str(df["SAMPLE_ID"]),
            "analysis_id": id_str(df["ANALYSIS_ID"]),
            "analysis_type": df["ANALYSIS_TYPE"],
            "analysis_method": df["ANALYSIS_METHOD"],
            "taxon_list_item_key": df["TAXON_LIST_ITEM_KEY"].str.strip(),
            # light microscopy: valves/units counted; DNA (DNAP): sequence reads
            "abundance": _num(df["UNITS_FOUND_OR_SEQUENCE_READS"]).astype("float32"),
        })
    return out


def read_zip_table(zip_path: Path, suffix: str, **kw) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf, zf.open(_zip_member(zf, suffix)) as fh:
        return read_csv_str(fh, **kw)


def iter_zip_table(zip_path: Path, suffix: str, chunksize: int = 1_000_000, usecols=None):
    with zipfile.ZipFile(zip_path) as zf, zf.open(_zip_member(zf, suffix)) as fh:
        reader = pd.read_csv(fh, encoding="utf-8-sig", dtype=str, keep_default_na=False,
                             chunksize=chunksize, usecols=usecols)
        for ch in reader:
            ch.columns = [c.strip().upper() for c in ch.columns]
            yield ch


def tidy_taxon_info(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    out = pd.DataFrame({
        "taxon_list_item_key": df["TAXON_LIST_ITEM_KEY"].str.strip(),
        "taxon_name": df["TAXON_NAME"].str.strip(),
        "taxon_version_key": df["TAXON_VERSION_KEY"].str.strip(),
        "authority": df["AUTHORITY"].replace("", None),
        "taxon_name_current": df["TAXON_NAME_CURRENT"].str.upper().eq("YES"),
        "preferred_taxon_name": df["PREFERRED_TAXON_NAME"].str.strip(),
        "preferred_name_tvk": df["PREFERRED_NAME_TVK"].str.strip(),
        "nbn_recommended_tvk": df["NBN_RECOMMENDED_TVK"].replace("", None),
        "parent_tlik": df["PARENT_TLIK"].replace("", None),
        "parent_taxon_name": df["PARENT_TAXON_NAME"].replace("", None),
        "taxon_rank": df["TAXON_RANK"].str.strip(),
        "taxon_type": df["TAXON_TYPE"].str.strip(),
        "taxon_group_name": df["TAXON_GROUP_NAME"].str.strip(),
        "non_native": df["NON_NATIVE_SP"].str.upper().eq("T"),
        "protected": df["PROTECTED_TAXA"].str.upper().eq("T"),
    })
    return out


# ------------------------------------------------------------------ WQ archive API

def sampling_point_params(skip: int, limit: int = 250, point_types=None) -> dict:
    p = {"skip": int(skip), "limit": int(limit)}
    if point_types:
        p["samplingPointType"] = ",".join(point_types)
    return p


SAMPLING_POINT_HEADERS = {"Accept": "application/ld+json",
                          "Accept-Crs": "http://www.opengis.net/def/crs/EPSG/0/27700"}

_WKT_RE = re.compile(r"POINT\s*\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)")


def parse_sampling_points(doc: dict) -> pd.DataFrame:
    """JSON-LD hydra collection -> one row per sampling point (BNG easting/northing)."""
    rows = []
    for m in doc.get("member", []):
        geom = (m.get("geometry") or {}).get("asWKT") or ""
        mm = _WKT_RE.search(geom)
        e, n = (float(mm.group(1)), float(mm.group(2))) if mm else (np.nan, np.nan)
        if mm and "27700" not in geom:
            raise ValueError(f"unexpected CRS in {geom!r}; request BNG (EPSG:27700)")

        def lab(key, field, m=m):
            v = m.get(key) or {}
            return v.get(field) if isinstance(v, dict) else None

        rows.append({
            "wq_point": m.get("notation"),
            "wq_point_label": m.get("prefLabel"),
            "wq_point_alt_label": m.get("altLabel"),
            "wq_easting_bng": e,
            "wq_northing_bng": n,
            "wq_status": lab("samplingPointStatus", "notation"),
            "wq_status_label": lab("samplingPointStatus", "prefLabel"),
            "wq_type": lab("samplingPointType", "notation"),
            "wq_type_label": lab("samplingPointType", "prefLabel"),
            "wq_region": lab("region", "prefLabel"),
            "wq_area": lab("area", "prefLabel"),
            "wq_subarea": lab("subArea", "prefLabel"),
        })
    return pd.DataFrame(rows)


def observation_params(points, determinands, date_from: str, date_to: str, *, skip: int = 0,
                       limit: int = 2500) -> dict:
    """Query-string filters for POST /data/observation (no body; <=100 points; <=1 year)."""
    points = list(points)
    if not 1 <= len(points) <= 100:
        raise ValueError("1..100 sampling points per multi-point request")
    if pd.Timestamp(date_to) - pd.Timestamp(date_from) > pd.Timedelta(days=366):
        raise ValueError("multi-point observation requests are limited to <= 1 year")
    return {"pointNotation": ",".join(points), "determinand": ",".join(determinands),
            "dateFrom": date_from, "dateTo": date_to, "skip": int(skip), "limit": int(limit)}


OBSERVATION_HEADERS = {"Accept": "text/csv",
                       "Accept-Crs": "http://www.opengis.net/def/crs/EPSG/0/27700"}

_OBS_ID_RE = re.compile(r"/sampling-point/(?P<pt>[^/]+)/sample/(?P<sample>\d+)/observation/(?P<obs>\d+)")


def parse_observations_csv(text: str | bytes) -> pd.DataFrame:
    """WQA observation CSV -> tidy long table with censoring parsed and units kept.

    ``phenomenonTime`` is published without a UTC offset (local clock time, assumed
    Europe/London); it is stored as ``sample_time_local`` (naive) + ``date`` and converted to
    ``time_utc`` under that documented assumption (``tz_source`` column).
    """
    if isinstance(text, bytes):
        text = text.decode("utf-8-sig", errors="replace")
    if not text.strip():
        return pd.DataFrame()
    raw = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    if raw.empty:
        return pd.DataFrame()
    ids = raw["id"].str.extract(_OBS_ID_RE)
    res = parse_censored(raw["result"])
    local = pd.to_datetime(raw["phenomenonTime"], errors="coerce")
    utc = local.dt.tz_localize("Europe/London", ambiguous="NaT", nonexistent="shift_forward") \
        .dt.tz_convert("UTC")
    e_col = "samplingPoint.easting" if "samplingPoint.easting" in raw else None
    out = pd.DataFrame({
        "wq_point": raw["samplingPoint.notation"],
        "sample_notation": ids["sample"],
        "observation_notation": ids["obs"],
        "sample_time_local": local,
        "time_utc": utc,
        "date": local.dt.date,
        "tz_source": "unspecified in source; assumed Europe/London",
        "sampling_purpose": raw.get("samplingPurpose"),
        "sample_material": raw.get("sampleMaterialType"),
        "determinand": raw["determinand.notation"].str.zfill(4),
        "determinand_label": raw["determinand.prefLabel"],
        "result_raw": raw["result"],
        "value": res["value"],
        "censored": res["censored"],
        "censor_op": res["censor_op"],
        "detection_limit": res["detection_limit"],
        "qualifier": res["qualifier"],
        "unit": raw["unit"],
    })
    if e_col:
        out["wq_easting_bng"] = pd.to_numeric(raw["samplingPoint.easting"], errors="coerce")
        out["wq_northing_bng"] = pd.to_numeric(raw["samplingPoint.northing"], errors="coerce")
    return out
