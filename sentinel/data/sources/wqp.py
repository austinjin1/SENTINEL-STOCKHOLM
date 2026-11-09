# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Water Quality Portal (WQP) client for the WQX 3.0 profiles: build request -> fetch -> parse.

Endpoints (verified 2026-10-22 against https://www.waterqualitydata.us/beta/ config):

* ``POST /wqx3/Result/search?mimeType=csv&zip=yes`` with a JSON body; array filters are JSON
  lists (WQX3 rejects ``;``-joined values). ``dataProfile`` = ``fullPhysChem`` (181 columns,
  includes ``LastChangeDate``, hydrologic event, temperature basis), ``basicPhysChem`` (98) or
  ``narrow``.
* ``POST /wqx3/Station/search?mimeType=csv&zip=yes`` (no dataProfile).
* Planning only: the legacy summary service
  ``GET /data/summary/monitoringLocation/search?dataProfile=periodOfRecord&summaryYears=all``
  (per site x year x characteristic counts). It is WQX 2.2 and lacks USGS data added after
  2024-03-11, so it is never used as data, only to size request windows and as a
  completeness reference.

Server behaviour that shapes the client (measured 2026-10-22):

* WQX3 sends no count headers and has no ``/count`` endpoint.
* If the database needs more than about 60 s before the first row, the server streams what it has
  and appends ``ERROR: INCOMPLETE DATA - THE RESULTS FOR THIS REQUEST ARE NOT COMPLETE ...``.
  State-filtered queries over long date ranges hit this even for small results (e.g. one state,
  one year, 32 nitrogen names). Short date windows over a list of states start streaming in
  ~4-10 s. The downloader therefore chunks by (state list x characteristic group x date window)
  and splits any window that comes back incomplete. :class:`IncompleteResponse` signals it.
* Throughput is row-bound (~600-1,900 rows/s per request), not bandwidth-bound.

Missing-value hygiene: non-numeric result strings and sentinel codes become NaN at parse time and
the original string is kept in ``value_text`` with ``value_flag`` set.
"""

from __future__ import annotations

import io
import logging
import random
import re
import time
import zipfile
from typing import Iterable

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)

BASE_URL = "https://www.waterqualitydata.us"
RESULT_URL = f"{BASE_URL}/wqx3/Result/search"
STATION_URL = f"{BASE_URL}/wqx3/Station/search"
SUMMARY_URL = f"{BASE_URL}/data/summary/monitoringLocation/search"  # legacy, planning only
USER_AGENT = "SENTINEL-v2 research (public-data download)"
INCOMPLETE_MARKER = "INCOMPLETE DATA"

# 48 contiguous states + DC + AK + HI + PR (postal -> FIPS); WQP statecode is "US:<fips>".
STATES: dict[str, str] = {
    "AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06", "CO": "08", "CT": "09",
    "DE": "10", "DC": "11", "FL": "12", "GA": "13", "HI": "15", "ID": "16", "IL": "17",
    "IN": "18", "IA": "19", "KS": "20", "KY": "21", "LA": "22", "ME": "23", "MD": "24",
    "MA": "25", "MI": "26", "MN": "27", "MS": "28", "MO": "29", "MT": "30", "NE": "31",
    "NV": "32", "NH": "33", "NJ": "34", "NM": "35", "NY": "36", "NC": "37", "ND": "38",
    "OH": "39", "OK": "40", "OR": "41", "PA": "42", "RI": "44", "SC": "45", "SD": "46",
    "TN": "47", "TX": "48", "UT": "49", "VT": "50", "VA": "51", "WA": "53", "WV": "54",
    "WI": "55", "WY": "56", "PR": "72",
}
FIPS_TO_POSTAL = {v: k for k, v in STATES.items()}


def statecode(postal: str) -> str:
    """'MN' -> 'US:27'."""
    return f"US:{STATES[postal]}"


# ---------------------------------------------------------------------------------------------
# Characteristic groups (exact WQX CharacteristicName values, from /Codes/characteristicname and
# observed in WQX3 responses). Probe/sensor fluorescence names are deliberately absent; see
# EXCLUDED_CHARACTERISTICS. "Depth, Secchi disk visible at bottom" is requested only to flag
# right-censored Secchi readings.
# ---------------------------------------------------------------------------------------------
CHARACTERISTIC_GROUPS: dict[str, tuple[str, ...]] = {
    "optical": (
        "Chlorophyll a", "Chlorophyll a, corrected for pheophytin",
        "Chlorophyll a, uncorrected for pheophytin", "Chlorophyll a, free of pheophytin",
        "Chlorophyll a - Phytoplankton (suspended)", "Chlorophyll A", "Chlorophyll",
        "Pheophytin a", "Pheophytin", "Phaeophytin - Phytoplankton (suspended)",
        "Pheopigments of Chlorophyll", "Phaeopigment",
        "Total suspended solids", "Suspended Sediment Concentration (SSC)",
        "Suspended sediment concentration (SSC)", "Total Suspended Particulate Matter",
        "Turbidity", "Turbidity Field",
        "Depth, Secchi disk depth", "Secchi depth", "Water transparency, Secchi disc",
        "Depth, Secchi disk visible at bottom (Y/N) (choice list)",
        "Transparency, tube with disk", "Transparency, Secchi tube with disk",
        "Colored dissolved organic matter (CDOM)", "Absorbance at 440 nm",
        "Phycocyanin", "Phycocyanins", "Algae-Phycocyanin",
    ),
    "phosphorus": (
        "Phosphorus", "Total Phosphorus, mixed forms", "Phosphorus as P",
        "Orthophosphate", "Orthophosphate as P", "Orthophosphate as PO4",
        "Ortho-Phosphate-Phosphorus", "Soluble Reactive Phosphorus (SRP)",
        "Phosphate-phosphorus", "Phosphate-phosphorus as P", "Phosphate-phosphorus as PO4",
        "Phosphate",
    ),
    "nitrogen": (
        "Nitrogen", "Total Nitrogen, mixed forms",
        "Nitrogen, mixed forms (NH3), (NH4), organic, (NO2) and (NO3)",
        "Total Nitrogen, mixed forms (NH3), (NH4), organic, (NO2) and (NO3)",
        "Nitrate", "Nitrate as N", "Nitrate-N", "Nitrate-nitrogen", "Nitrate-Nitrogen",
        "Nitrite", "Nitrite as N", "Nitrite-N",
        "Inorganic nitrogen (nitrate and nitrite)", "Inorganic nitrogen (nitrate and nitrite) as N",
        "Nitrate + Nitrite", "Nitrate-Nitrite",
        "Inorganic nitrogen (nitrate and nitrite) ***retired***use Nitrate + Nitrite",
        "Ammonia", "Ammonia and ammonium", "Ammonia as NH3", "Ammonia-nitrogen",
        "Ammonia-nitrogen as N", "Ammonium", "Ammonium as N", "Ammonium as NH4",
        "Ammonium-nitrogen", "Ammonium-Nitrogen", "Nitrogen, ammonia (NH3) as NH4",
        "Kjeldahl nitrogen", "Total Kjeldahl nitrogen", "Total Kjeldahl nitrogen (Organic N & NH3)",
        "Nitrogen Kjeldahl",
    ),
    "carbon": ("Organic carbon", "Non-purgeable Organic Carbon (NPOC)"),
    "ions": (
        "Chloride", "Sulfate", "Sulfate as SO4", "Sulfate as S", "Total Sulfate",
        "Specific conductance", "Specific conductivity",
        "Specific conductivity***retired***use Specific conductance", "Conductivity",
    ),
    "bacteria": ("Escherichia coli", "Escherichia", "Enterococcus", "Enterococci", "Fecal Coliform"),
    "cyanotoxins": (
        "Microcystin", "Microcystins and nodularins", "Total microcystins plus nodularins",
        "Microcystins/nodularin", "Dissolved microcystins plus nodularins",
        "Particulate microcystins/nodularin",
        "Microcystin LR", "Microcystin RR", "Microcystin YR", "Microcystin LA", "Microcystin LF",
        "Microcystin LW", "Microcystin LY", "Microcystin WR", "Microcystin HilR",
        "Microcystin HiLR", "Microcystin HtyR", "Microcystin HtYR", "Microcystin dLR",
        "Microcystin dRR", "Microcystin LR [DAsp3]", "Microcystin LR, Desmethyl",
        "Microcystin LR [Dha7]", "Microcystin LR [D-Leu1]", "Microcystin RR [DAsp3]",
        "Cylindrospermopsin", "Cylindrospermopsins", "Anatoxin-A", "Anatoxin a", "Anatoxin",
        "Saxitoxin", "Saxitoxins",
    ),
    # Field physico-chemistry: kept only where co-sampled (same site + local date) with a result
    # from the groups above (see the downloader).
    "physchem": (
        "Dissolved oxygen (DO)", "Dissolved oxygen",
        "Dissolved Oxygen***retired***use Dissolved Oxygen (DO)", "pH", "PH", "Temperature, water",
    ),
}
TARGET_GROUPS = tuple(g for g in CHARACTERISTIC_GROUPS if g != "physchem")
CO_SAMPLED_GROUPS = ("physchem",)

# Not requested, with the reason (documentation; tests assert none is requested).
EXCLUDED_CHARACTERISTICS: dict[str, str] = {
    "Chlorophyll a (probe)": "in-situ fluorometer (sensor)",
    "Chlorophyll a (probe relative fluorescence)": "sensor RFU",
    "Chlorophyll a (probe) concentration, Cyanobacteria (bluegreen)": "sensor",
    "Chlorophyll a (probe) concentration, Cryptophyta (cryptophytes)": "sensor",
    "Chlorophyll a (probe) concentration, Xanthophyte (yellow algae)": "sensor",
    "Chlorophyll a - Periphyton (attached)": "benthic areal biomass, not water column",
    "Phaeophytin - Periphyton (attached)": "benthic areal biomass, not water column",
    "Phycocyanin (probe)": "sensor",
    "Phycocyanin (probe relative fluorescence)": "sensor RFU",
    "Fluorescent dissolved organic matter (FDOM)": "sensor fluorescence",
    "Escherichia coli DNA": "molecular marker, not culture/qPCR concentration of E. coli",
    "Microcystin/nodularin genes mcyE/ndaF": "gene copies, not toxin concentration",
    "Dissolved oxygen saturation": "not requested (DO mg/L only)",
}

# ---------------------------------------------------------------------------------------------
# Columns kept from the fullPhysChem / basicPhysChem result profiles -> tidy names.
# ---------------------------------------------------------------------------------------------
RESULT_COLUMNS: dict[str, str] = {
    "ProviderName": "provider",
    "Org_Identifier": "organization",
    "Org_Type": "org_type",
    "Project_Identifier": "project_id",
    "Location_Identifier": "site_id",
    "Location_StatePostalCode": "state",
    "Activity_ActivityIdentifier": "activity_id",
    "Activity_TypeCode": "activity_type",
    "Activity_Media": "media",
    "Activity_MediaSubdivision": "media_subdivision",
    "Activity_StartDate": "activity_start_date",
    "Activity_StartTime": "activity_start_time",
    "Activity_StartTimeZone": "activity_tz",
    "Activity_DepthHeightMeasure": "activity_depth",
    "Activity_DepthHeightMeasureUnit": "activity_depth_unit",
    "Activity_TopDepthMeasure": "activity_top_depth",
    "Activity_TopDepthMeasureUnit": "activity_top_depth_unit",
    "Activity_BottomDepthMeasure": "activity_bottom_depth",
    "Activity_BottomDepthMeasureUnit": "activity_bottom_depth_unit",
    "Activity_ActivityRelativeDepth": "activity_relative_depth",
    "Activity_HydrologicCondition": "hydrologic_condition",
    "Activity_HydrologicEvent": "hydrologic_event",
    "SampleCollectionMethod_Name": "sample_collection_method",
    "Result_ResultDetectionCondition": "detection_condition",
    "Result_Characteristic": "characteristic",
    "Result_CharacteristicUserSupplied": "characteristic_user_supplied",
    "Result_MethodSpeciation": "speciation",
    "Result_SampleFraction": "sample_fraction",
    "ResultDepthHeight_Measure": "result_depth",
    "ResultDepthHeight_MeasureUnit": "result_depth_unit",
    "Result_MeasureIdentifier": "result_id",
    "Result_Measure": "value_raw",
    "Result_MeasureUnit": "unit",
    "Result_MeasureQualifierCode": "qualifier",
    "Result_MeasureStatusIdentifier": "status",
    "Result_StatisticalBase": "statistical_base",
    "Result_MeasureType": "value_type",
    "Result_TimeBasis": "time_basis",
    "Result_MeasureTemperatureBasis": "temperature_basis",
    "DataQuality_ResultComment": "result_comment",
    "DetectionLimit_TypeA": "dl_type_a",
    "DetectionLimit_MeasureA": "dl_value_a",
    "DetectionLimit_MeasureUnitA": "dl_unit_a",
    "DetectionLimit_TypeB": "dl_type_b",
    "DetectionLimit_MeasureB": "dl_value_b",
    "DetectionLimit_MeasureUnitB": "dl_unit_b",
    "ResultAnalyticalMethod_Identifier": "method_id",
    "ResultAnalyticalMethod_IdentifierContext": "method_context",
    "ResultAnalyticalMethod_Name": "method_name",
    "LabInfo_Name": "lab_name",
    "LabInfo_LaboratoryComment": "lab_comment",
    "USGSpcode": "usgs_pcode",
    "LastChangeDate": "last_change_date",
}
# Location columns embedded in result rows (used to build/complete the station table).
RESULT_LOCATION_COLUMNS: dict[str, str] = {
    "Location_Identifier": "site_id",
    "Location_Name": "name",
    "Location_Type": "site_type",
    "Location_StatePostalCode": "state",
    "Location_CountyName": "county",
    "Location_CountyCode": "county_code",
    "Location_HUCEightDigitCode": "huc8",
    "Location_HUCTwelveDigitCode": "huc12",
    "Location_Latitude": "lat",
    "Location_Longitude": "lon",
    "Location_HorzCoordReferenceSystemDatum": "datum",
    "Location_LatitudeStandardized": "lat_std",
    "Location_LongitudeStandardized": "lon_std",
    "Location_HorzCoordStandardizedDatum": "datum_std",
    "Org_Identifier": "organization",
    "Org_FormalName": "organization_name",
    "ProviderName": "provider",
}
STATION_COLUMNS: dict[str, str] = {
    **{k: v for k, v in RESULT_LOCATION_COLUMNS.items()},
    "Location_Description": "description",
    "Location_TribalLand": "tribal_land",
    "Location_HorzCollectionMethod": "horz_collection_method",
    "Location_HorzAccuracyMeasure": "horz_accuracy",
    "Location_HorzAccuracyMeasureUnit": "horz_accuracy_unit",
    "Location_DrainageAreaMeasure": "drainage_area",
    "Location_DrainageAreaMeasureUnit": "drainage_area_unit",
    "Location_ContributingDrainageAreaMeasure": "contrib_drainage_area",
    "Location_ContributingDrainageAreaMeasureUnit": "contrib_drainage_area_unit",
    "AlternateLocation_IdentifierA": "alt_id_a",
    "AlternateLocation_IdentifierContextA": "alt_id_context_a",
}
STRING_ID_COLUMNS = ("site_id", "huc8", "huc12", "usgs_pcode", "county_code", "result_id",
                     "activity_id")

# ---------------------------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------------------------


class WQPError(RuntimeError):
    """Persistent request failure (after retries) or a non-retryable HTTP status."""


class IncompleteResponse(RuntimeError):
    """The server truncated the result set (``ERROR: INCOMPLETE DATA`` trailer)."""

    def __init__(self, msg: str, n_lines: int = 0):
        super().__init__(msg)
        self.n_lines = n_lines


_session: requests.Session | None = None


def session() -> requests.Session:
    global _session
    if _session is None:
        s = requests.Session()
        s.headers.update({"User-Agent": USER_AGENT})
        _session = s
    return _session


def _sleep_backoff(attempt: int, base: float, cap: float, retry_after: str | None = None) -> float:
    if retry_after:
        try:
            wait = float(retry_after)
        except ValueError:
            wait = base * 2 ** attempt
    else:
        wait = min(cap, base * 2 ** attempt) * (0.75 + 0.5 * random.random())
    time.sleep(wait)
    return wait


def _unzip_single(content: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        names = z.namelist()
        if len(names) != 1:
            raise zipfile.BadZipFile(f"expected one member, got {names}")
        return z.read(names[0]).decode("utf-8", errors="replace")


def _transient_400(r) -> bool:
    """During a WQP code-list outage (observed 2026-10-22 07:03-07:06 EDT) the server rejected
    valid requests with HTTP 400 "The value of providers=STORET is not in the list of enumerated
    values" (a parameter we never send). Such 400s are retried with backoff."""
    return r.status_code == 400 and "list of enumerated values" in (r.text or "")


def check_complete(text: str) -> None:
    """Raise :class:`IncompleteResponse` if the CSV ends with the server's truncation trailer."""
    tail = text[-4000:]
    if INCOMPLETE_MARKER in tail or re.search(r"(^|\n)ERROR:", tail):
        raise IncompleteResponse("server reported incomplete data", n_lines=text.count("\n"))


def post_csv(url: str, body: dict, *, retries: int = 6, backoff: float = 30.0,
             backoff_cap: float = 900.0, connect_timeout: float = 60.0,
             read_timeout: float = 900.0) -> tuple[str, dict]:
    """POST a JSON query, return (csv_text, info). Retries transient failures with backoff.

    ``IncompleteResponse`` is raised immediately (not retried): the caller must shrink the
    request. HTTP 400/404 are not retried (bad characteristic name, no such resource).
    """
    params = {"mimeType": "csv", "zip": "yes", "sorted": "no"}
    last = ""
    for attempt in range(retries):
        t0 = time.time()
        try:
            with session().post(url, params=params, json=body, stream=True,
                                timeout=(connect_timeout, read_timeout)) as r:
                if r.status_code == 200:
                    content = r.content
                    text = _unzip_single(content)
                    info = {"bytes": len(content), "seconds": round(time.time() - t0, 1),
                            "request_id": r.headers.get("wqp-request-id", "")}
                    check_complete(text)
                    return text, info
                if r.status_code in (400, 404, 405, 413, 414) and not _transient_400(r):
                    raise WQPError(f"HTTP {r.status_code} (not retried): {r.text[:300]}")
                last = f"HTTP {r.status_code}: {r.text[:200]}"
                retry_after = r.headers.get("Retry-After") if r.status_code in (429, 503) else None
        except (requests.RequestException, zipfile.BadZipFile, EOFError, OSError) as e:
            last = repr(e)[:300]
            retry_after = None
        wait = _sleep_backoff(attempt, backoff, backoff_cap, retry_after)
        log.warning("retry %d/%d after %.0fs: %s", attempt + 1, retries, wait, last)
    raise WQPError(f"failed after {retries} attempts: {last}")


def get_zip_csv(url: str, params: list[tuple[str, str]], *, retries: int = 6,
                backoff: float = 30.0, timeout: float = 900.0) -> str:
    """GET helper for the legacy summary service (repeated keys allowed)."""
    last = ""
    for attempt in range(retries):
        try:
            r = session().get(url, params=params, timeout=(60, timeout))
            if r.status_code == 200:
                text = _unzip_single(r.content)
                check_complete(text)
                return text
            if r.status_code in (400, 404, 414) and not _transient_400(r):
                raise WQPError(f"HTTP {r.status_code} (not retried): {r.text[:300]}")
            last = f"HTTP {r.status_code}: {r.text[:200]}"
            retry_after = r.headers.get("Retry-After") if r.status_code in (429, 503) else None
        except (requests.RequestException, zipfile.BadZipFile, EOFError, OSError) as e:
            last, retry_after = repr(e)[:300], None
        wait = _sleep_backoff(attempt, backoff, 900.0, retry_after)
        log.warning("retry %d/%d after %.0fs: %s", attempt + 1, retries, wait, last)
    raise WQPError(f"failed after {retries} attempts: {last}")


# ---------------------------------------------------------------------------------------------
# Request builders
# ---------------------------------------------------------------------------------------------


def _mdy(date: str | pd.Timestamp) -> str:
    """WQP wants MM-DD-YYYY."""
    return pd.Timestamp(date).strftime("%m-%d-%Y")


def result_query(states: Iterable[str], names: Iterable[str], start, end, *,
                 profile: str = "fullPhysChem", sample_media: str | None = "Water") -> dict:
    """JSON body for /wqx3/Result/search. ``states`` are postal codes; dates inclusive."""
    body = {
        "statecode": [statecode(s) for s in states],
        "characteristicName": list(names),
        "startDateLo": _mdy(start),
        "startDateHi": _mdy(end),
        "dataProfile": profile,
    }
    if sample_media:
        body["sampleMedia"] = [sample_media]
    return body


def station_query(states: Iterable[str], names: Iterable[str] | None = None,
                  start=None, end=None) -> dict:
    body: dict = {"statecode": [statecode(s) for s in states]}
    if names:
        body["characteristicName"] = list(names)
    if start is not None:
        body["startDateLo"] = _mdy(start)
    if end is not None:
        body["startDateHi"] = _mdy(end)
    return body


def fetch_results(states, names, start, end, **kw) -> tuple[str, dict]:
    return post_csv(RESULT_URL, result_query(states, names, start, end,
                                             profile=kw.pop("profile", "fullPhysChem"),
                                             sample_media=kw.pop("sample_media", "Water")), **kw)


def fetch_stations(states, names=None, start=None, end=None, **kw) -> tuple[str, dict]:
    return post_csv(STATION_URL, station_query(states, names, start, end), **kw)


def fetch_summary(state: str, names: Iterable[str]) -> str:
    """Legacy per-site/year/characteristic counts for one state (planning/reference only)."""
    params = [("statecode", statecode(state)), ("summaryYears", "all"),
              ("dataProfile", "periodOfRecord"), ("mimeType", "csv"), ("zip", "yes")]
    params += [("characteristicName", n) for n in names]
    return get_zip_csv(SUMMARY_URL, params)


# ---------------------------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------------------------

# WQX time-zone codes -> UTC offset (hours). Unknown/blank -> no time_utc.
TZ_OFFSETS_H: dict[str, float] = {
    "UTC": 0, "GMT": 0, "Z": 0,
    "EST": -5, "EDT": -4, "CST": -6, "CDT": -5, "MST": -7, "MDT": -6, "PST": -8, "PDT": -7,
    "AKST": -9, "AKDT": -8, "YST": -9, "YDT": -8, "HST": -10, "HDT": -9, "HAST": -10,
    "HADT": -9, "AST": -4, "ADT": -3, "NST": -3.5, "NDT": -2.5, "SST": -11,
}

# Sentinel codes masked at parse time (value -> NaN, original kept in value_text).
SENTINEL_CODES = frozenset({-9999999.0, -999999.0, -99999.0, -9999.0, -999.0, -99.0,
                            999999.0, 9999999.0, 99999999.0})

_NUM = re.compile(r"^\s*(?P<op><=|>=|<|>|~|=)?\s*(?P<num>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$")

_LEFT_COND = re.compile(r"not detected|non-?detect|below|less than|not present|"
                        r"detected not quantified|between inst|trace|present below", re.I)
_RIGHT_COND = re.compile(r"above|greater than|too numerous|exceeds|present above", re.I)
_NO_VALUE_COND = re.compile(r"not reported|systematic contamination|not determined|"
                            r"no result|not analyzed|sample lost|invalid", re.I)


def _blank(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip()


def parse_values(raw: pd.Series) -> pd.DataFrame:
    """Parse ``Result_Measure`` strings.

    Returns columns ``value`` (float, NaN when not numeric or a sentinel code), ``value_op``
    (``'<'``/``'>'``/``''`` from a leading operator such as ``"<0.01"``) and ``value_flag``
    (``''``, ``'non_numeric'`` or ``'sentinel_code'``).
    """
    s = _blank(raw).str.replace(",", "", regex=False)
    m = s.str.extract(_NUM)
    val = pd.to_numeric(m["num"], errors="coerce")
    op = m["op"].fillna("").replace({"<=": "<", ">=": ">", "~": "", "=": ""})
    flag = pd.Series("", index=s.index, dtype=object)
    flag[(s != "") & val.isna()] = "non_numeric"
    sentinel = val.isin(SENTINEL_CODES)
    flag[sentinel] = "sentinel_code"
    val = val.mask(sentinel)
    return pd.DataFrame({"value": val.astype("float64"), "value_op": op, "value_flag": flag})


def classify_detection(cond: pd.Series) -> pd.Series:
    """Detection condition -> ``'<'`` (left-censored), ``'>'`` (right-censored), ``'none'``
    (no usable value) or ``''`` (normal)."""
    c = _blank(cond)
    out = pd.Series("", index=c.index, dtype=object)
    out[c.str.contains(_LEFT_COND)] = "<"
    out[c.str.contains(_RIGHT_COND)] = ">"
    out[c.str.contains(_NO_VALUE_COND)] = "none"
    return out


_LEFT_DL_PREF = ("censoring level", "lower reporting limit", "reporting limit",
                 "lower quantitation limit", "practical quantitation limit",
                 "minimum reporting level", "method detection level", "detection limit",
                 "lower limit of detection", "limit of detection", "instrument detection level",
                 "estimated detection level", "sample detection limit",
                 "laboratory reporting level")
_RIGHT_DL_PREF = ("upper quantitation limit", "upper reporting limit", "censoring level",
                  "upper limit")


def _pick_dl(df: pd.DataFrame, censor: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Choose the detection/quantitation limit (value, unit, type) relevant for each row."""
    va = pd.to_numeric(_blank(df["dl_value_a"]).str.replace(",", "", regex=False), errors="coerce")
    vb = pd.to_numeric(_blank(df["dl_value_b"]).str.replace(",", "", regex=False), errors="coerce")
    ta, tb = _blank(df["dl_type_a"]).str.lower(), _blank(df["dl_type_b"]).str.lower()

    def rank(t: pd.Series, prefs: tuple[str, ...]) -> pd.Series:
        r = pd.Series(len(prefs) + 1, index=t.index, dtype="int64")
        for i, p in enumerate(reversed(prefs)):
            r[t.str.startswith(p)] = len(prefs) - 1 - i
        return r

    left_a, left_b = rank(ta, _LEFT_DL_PREF).to_numpy(), rank(tb, _LEFT_DL_PREF).to_numpy()
    right_a, right_b = rank(ta, _RIGHT_DL_PREF).to_numpy(), rank(tb, _RIGHT_DL_PREF).to_numpy()
    has_b, no_a = vb.notna().to_numpy(), va.isna().to_numpy()
    is_right = (censor == ">").to_numpy()
    use_b = np.where(is_right, has_b & (no_a | (right_b < right_a)),
                     has_b & (no_a | (left_b < left_a)))
    use_b = pd.Series(use_b.astype(bool), index=df.index)
    val = va.where(~use_b, vb)
    unit = _blank(df["dl_unit_a"]).where(~use_b, _blank(df["dl_unit_b"]))
    typ = _blank(df["dl_type_a"]).where(~use_b, _blank(df["dl_type_b"]))
    return val, unit, typ


_QC_PATTERNS = (
    ("blank", re.compile(r"blank", re.I)),
    ("spike", re.compile(r"spike|recovery|surrogate", re.I)),
    ("replicate", re.compile(r"replicate|duplicate|split", re.I)),
    ("reference", re.compile(r"reference|standard|calibration|check|blind(?! duplicate)", re.I)),
)


def classify_activity(activity_type: pd.Series) -> pd.Series:
    """Activity type -> ``''`` (field sample/measurement), ``'logger'`` (data-logger record)
    or a QC class (``'blank'``, ``'spike'``, ``'replicate'``, ``'reference'``, ``'other_qc'``)."""
    a = _blank(activity_type)
    out = pd.Series("", index=a.index, dtype=object)
    qc = a.str.startswith("Quality Control") | a.str.contains(r"\bQC\b", regex=True)
    for name, pat in reversed(_QC_PATTERNS):
        out[qc & a.str.contains(pat)] = name
    out[qc & (out == "")] = "other_qc"
    out[~qc & a.str.contains("Data Logger", case=False, regex=False)] = "logger"
    return out


_STATUS = {"accepted": "accepted", "final": "accepted", "validated": "accepted",
           "historical": "accepted", "approved": "accepted",
           "preliminary": "preliminary", "provisional": "preliminary",
           "unreviewed": "preliminary", "raw": "preliminary", "pending": "preliminary",
           "rejected": "rejected"}


def classify_status(status: pd.Series) -> pd.Series:
    s = _blank(status).str.lower()
    return s.map(_STATUS).fillna("unknown").where(s != "", "unknown")


_TIME = re.compile(r"^\s*(\d{1,2}):(\d{2})(?::(\d{2}))?")


def to_utc(date: pd.Series, tm: pd.Series, tz: pd.Series) -> pd.Series:
    """Local activity start date + time + WQX zone code -> tz-aware UTC timestamp (NaT if the
    time or the zone is missing/unknown)."""
    d = pd.to_datetime(_blank(date), format="%Y-%m-%d", errors="coerce")
    m = _blank(tm).str.extract(_TIME)
    hh = pd.to_numeric(m[0], errors="coerce")
    mm = pd.to_numeric(m[1], errors="coerce")
    ss = pd.to_numeric(m[2], errors="coerce").fillna(0)
    off = _blank(tz).str.upper().map(TZ_OFFSETS_H)
    ok = d.notna() & hh.notna() & mm.notna() & off.notna() & (hh < 24) & (mm < 60)
    local = d + pd.to_timedelta(hh * 3600 + mm * 60 + ss, unit="s")
    utc = (local - pd.to_timedelta(off * 3600, unit="s")).where(ok)
    return utc.dt.tz_localize("UTC")


def read_csv_text(text: str, usecols: Iterable[str] | None = None) -> pd.DataFrame:
    """Read a WQP CSV string as all-string columns (empty string for missing)."""
    if not text.strip():
        return pd.DataFrame()
    head = text[: text.find("\n")] if "\n" in text else text
    cols = None
    if usecols is not None:
        present = set(pd.read_csv(io.StringIO(head), nrows=0).columns)
        cols = [c for c in usecols if c in present]
    return pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False, na_filter=False,
                       usecols=cols, low_memory=False)


def parse_results(text: str, *, group: str | None = None) -> pd.DataFrame:
    """Parse a WQX3 Result CSV (full/basic profile) into the tidy raw table.

    Keeps every row (QC, rejected, censored) and adds: ``value`` (float), ``value_text``
    (original string when not a plain number), ``value_flag``, ``censored`` (bool),
    ``censor_type`` (``'<'``/``'>'``/``''``), ``detection_limit`` (+ ``_unit``, ``_type``),
    ``status_class``, ``qc_class``, ``time_utc``, ``group``.
    """
    df = read_csv_text(text, usecols=list(RESULT_COLUMNS))
    if df.empty:
        return empty_result_frame()
    df = df.rename(columns=RESULT_COLUMNS)
    for c in RESULT_COLUMNS.values():
        if c not in df.columns:
            df[c] = ""
    df = df[list(RESULT_COLUMNS.values())]
    df = df[df["site_id"] != ""].reset_index(drop=True)

    pv = parse_values(df["value_raw"])
    det = classify_detection(df["detection_condition"])
    qual = _blank(df["qualifier"]).str.upper()
    stat = _blank(df["statistical_base"]).str.lower()
    censor = pd.Series("", index=df.index, dtype=object)
    censor[(pv["value_op"] == "<") | (det == "<")] = "<"
    censor[(pv["value_op"] == ">") | (det == ">")] = ">"
    # Qualifier codes that mean "not detected" when no explicit condition is given.
    nd_qual = qual.str.contains(r"(?:^|[;,\s])(?:U|ND|<)(?:[;,\s]|$)", regex=True)
    censor[(censor == "") & nd_qual & (det != "none")] = "<"
    # Secchi/transparency-tube "Minimum" = disk visible on bottom / tube full -> lower bound.
    clarity = df["characteristic"].str.contains("Secchi|Transparency|transparency", regex=True)
    censor[(censor == "") & clarity & (stat == "minimum")] = ">"

    dl_val, dl_unit, dl_type = _pick_dl(df, censor)
    # A censored row without a DL field uses its own reported number (e.g. "<0.01", ">2419.6").
    fill = (censor != "") & dl_val.isna() & pv["value"].notna()
    dl_val = dl_val.where(~fill, pv["value"])
    dl_unit = dl_unit.where(~fill | (dl_unit != ""), df["unit"])
    dl_unit = dl_unit.where(dl_unit != "", df["unit"].where(dl_val.notna(), ""))

    value = pv["value"].where(censor == "")  # censored -> value NaN, limit in detection_limit
    value = value.where(det != "none")
    text_keep = _blank(df["value_raw"]).where(
        (pv["value_flag"] != "") | (pv["value_op"] != ""), "")

    out = df.drop(columns=["value_raw"])
    out["value"] = value
    out["value_text"] = text_keep
    out["value_flag"] = pv["value_flag"]
    out["censored"] = censor != ""
    out["censor_type"] = censor
    out["detection_limit"] = dl_val.astype("float64")
    out["detection_limit_unit"] = dl_unit
    out["detection_limit_type"] = dl_type
    out["status_class"] = classify_status(df["status"])
    out["qc_class"] = classify_activity(df["activity_type"])
    out["time_utc"] = to_utc(df["activity_start_date"], df["activity_start_time"], df["activity_tz"])
    out["group"] = group or ""
    for c in ("result_comment", "lab_comment"):
        out[c] = out[c].str.slice(0, 300)
    return out


def empty_result_frame() -> pd.DataFrame:
    cols = [c for c in RESULT_COLUMNS.values() if c != "value_raw"]
    df = pd.DataFrame({c: pd.Series(dtype=object) for c in cols})
    for c, dt in (("value", "float64"), ("detection_limit", "float64")):
        df[c] = pd.Series(dtype=dt)
    for c in ("value_text", "value_flag", "censor_type", "detection_limit_unit",
              "detection_limit_type", "status_class", "qc_class", "group"):
        df[c] = pd.Series(dtype=object)
    df["censored"] = pd.Series(dtype=bool)
    df["time_utc"] = pd.Series(dtype="datetime64[ns, UTC]")
    return df


def parse_result_locations(text: str) -> pd.DataFrame:
    """Unique site records embedded in a Result CSV (fallback/complement to Station/search)."""
    df = read_csv_text(text, usecols=list(RESULT_LOCATION_COLUMNS))
    if df.empty:
        return pd.DataFrame(columns=list(RESULT_LOCATION_COLUMNS.values()))
    df = df.rename(columns=RESULT_LOCATION_COLUMNS)
    df = df[df["site_id"] != ""].drop_duplicates("site_id")
    return _finish_stations(df)


def parse_stations(text: str) -> pd.DataFrame:
    """Parse a WQX3 Station CSV into the station table (IDs and HUCs as strings)."""
    df = read_csv_text(text, usecols=list(STATION_COLUMNS))
    if df.empty:
        return pd.DataFrame(columns=list(dict.fromkeys(STATION_COLUMNS.values())))
    df = df.rename(columns=STATION_COLUMNS)
    df = df[df["site_id"] != ""].drop_duplicates("site_id")
    return _finish_stations(df)


def _finish_stations(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ("lat", "lon", "lat_std", "lon_std", "drainage_area", "contrib_drainage_area",
              "horz_accuracy"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("huc8", "huc12"):
        if c in df.columns:
            # keep as strings; restore leading zeros lost by upstream numeric handling
            width = 8 if c == "huc8" else 12
            s = _blank(df[c])
            df[c] = s.where(~s.str.fullmatch(r"\d+") | (s.str.len() >= width), s.str.zfill(width))
    return df.reset_index(drop=True)


def parse_summary(text: str) -> pd.DataFrame:
    """Legacy periodOfRecord summary -> (site_id, year, characteristic, n_results, ...)."""
    cols = {"Provider": "provider", "MonitoringLocationIdentifier": "site_id",
            "YearSummarized": "year", "CharacteristicName": "characteristic",
            "ActivityCount": "n_activities", "ResultCount": "n_results",
            "OrganizationIdentifier": "organization",
            "ResolvedMonitoringLocationTypeName": "site_type", "StateName": "state_name",
            "HUCEightDigitCode": "huc8"}
    df = read_csv_text(text, usecols=list(cols))
    if df.empty:
        return pd.DataFrame(columns=list(cols.values()))
    df = df.rename(columns=cols)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    for c in ("n_activities", "n_results"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int64")
    return df


def name_to_group() -> dict[str, str]:
    return {n: g for g, names in CHARACTERISTIC_GROUPS.items() for n in names}


def all_requested_names() -> list[str]:
    return [n for names in CHARACTERISTIC_GROUPS.values() for n in names]


def as_numpy_str(s: pd.Series) -> np.ndarray:  # small helper used by builders
    return _blank(s).to_numpy(dtype=object)
