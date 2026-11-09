# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Washington State Toxic Algae database (https://www.nwtoxicalgae.org/).

Publisher: WA Department of Ecology Freshwater Algae Control Program; the website and
database are hosted by King County. Content: laboratory cyanotoxin results (microcystin,
anatoxin-a, cylindrospermopsin, saxitoxin) for samples submitted by local jurisdictions, lake
managers and residents, 2007-present, plus the program's site list with coordinates.

Access routes (verified 2026-10-22; no login, no robots.txt (404)):

* ``Data.aspx`` "Detailed search" ASP.NET form. Posting the form with every search field blank
  and the **"Export Toxin Data"** button returns the whole toxin table as a CSV attachment
  (``toxinYYYY_MM_DD.csv``, Windows-1252). The writer does **not** quote fields: the free-text
  ``Location`` column contains commas and literal double quotes, so rows are parsed with an
  anchored regular expression (:data:`_ROW_RE`), never with the ``csv`` module.
* ``Data.aspx`` also embeds the map layer in a hidden field (``c_SiteInfoHiddenField``):
  site name, lat/lon, current map status (Within / Exceeded / Unknown), latest value per toxin
  and latest sample date, keyed by SiteID.
* ``LocationExtractor.aspx`` "Get EIM Locations" postback → ``EIMLocation.csv`` (quoted CSV):
  SiteID, EIM location code, county, lat/lon and EIM coordinate-metadata codes for all sites.
* ``YearSummary.aspx`` / ``StateSummary.aspx``: published sample and exceedance counts, used to
  cross-check the export (:func:`crosscheck_year_summary`).
* ``Data.aspx?SiteID=<id>`` per-site HTML tables (same rows as the export; used only to verify).

Terms: every page footer binds visitors to the King County website terms of use, which state
that site content "including all text, graphics, compilations, data" is King County property
and may not be published, displayed or distributed without prior written permission. Use the
data for internal research; do not redistribute raw or row-level derived tables.

Label definitions (see docs/datasheets/labels_wa_toxic_algae.md):

* ``exceed_wa_guideline``: result >= WA DOH provisional recreational guidance value
  (:data:`WA_GUIDELINES_UGL`, "Microcystin: >= 8 µg/L" etc. on About.aspx). The site's own
  ``ExceedInd`` flag (kept as ``exceed_published``) follows the same >= rule.
* ``exceed_epa_8ugL`` (microcystin only): result > 8 µg/L (EPA 2019 recreational swimming
  advisory). Censored results are resolved only when the bound decides the label; otherwise NA.
"""

from __future__ import annotations

import gzip
import html as _html
import io
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import events_common as C

SOURCE = "wa_toxic_algae"
BASE_URL = "https://www.nwtoxicalgae.org/"
DATA_URL = BASE_URL + "Data.aspx"
LOCATION_URL = BASE_URL + "LocationExtractor.aspx"
YEAR_SUMMARY_URL = BASE_URL + "YearSummary.aspx"
STATE_SUMMARY_URL = BASE_URL + "StateSummary.aspx"
ABOUT_URL = BASE_URL + "About.aspx"
SITE_PAGE_URL = BASE_URL + "Data.aspx?SiteID={site_id}"
SCUM_PAGE_URL = BASE_URL + "Scum.aspx?LabSampleNum={lab}"
TERMS_URL = "https://www.kingcounty.gov/About/termsOfUse.aspx"
LICENSE = ("King County website terms of use: site content incl. data/compilations is King County "
           "property; no publishing/display/distribution without prior written permission. "
           "Internal research use only; do not redistribute row-level data.")

# WA DOH provisional recreational guidance values, "toxin: >= value" (About.aspx, 2026-10-22).
WA_GUIDELINES_UGL = {"microcystin": 8.0, "anatoxin-a": 1.0, "cylindrospermopsin": 15.0, "saxitoxin": 75.0}
# EPA (2019) recreational criteria / swimming advisory values; exceedance = value > criterion.
EPA_MICROCYSTIN_UGL = 8.0
WA_BBOX = (-124.9, 45.4, -116.8, 49.1)  # W, S, E, N (generous; Columbia River border sites)

EXPORT_HEADER = ["County", "Site", "LabSampleNum", "WRIANumber", "CollectDate", "Parameter",
                 "Toxin Concentration (µg/L)", "MDL (µg/L)", "ExceedInd", "ScumInd", "Location",
                 "ClientLocator", "SiteID"]
RAW_COLUMNS = ["county", "site", "lab_sample_num", "wria", "collect_date", "parameter", "conc",
               "mdl", "exceed_ind", "scum_ind", "location", "client_locator", "site_id"]

# County and Site contain no commas in the 2026-10-22 export; Location (free text) does. The
# strict middle fields (WRIA digits, mm/dd/yyyy date, alphabetic toxin name, Yes/No flags) and
# the comma-free tail (ClientLocator, numeric SiteID) anchor every split unambiguously.
_ROW_RE = re.compile(
    r"^(?P<county>[^,]*),(?P<site>.*?),(?P<lab_sample_num>[^,]*),(?P<wria>\d*),"
    r"(?P<collect_date>\d{1,2}/\d{1,2}/\d{4}),(?P<parameter>[A-Za-z][A-Za-z \-]*),"
    r"(?P<conc>[^,]*),(?P<mdl>[^,]*),(?P<exceed_ind>Yes|No|),(?P<scum_ind>Yes|No|),"
    r"(?P<location>.*),(?P<client_locator>[^,]*),(?P<site_id>\d+)\s*$")

_TOXIN_ALIASES = {
    "microcystin": "microcystin", "microcystins": "microcystin", "total microcystins": "microcystin",
    "anatoxin-a": "anatoxin-a", "anatoxin a": "anatoxin-a", "anatoxin": "anatoxin-a",
    "cylindrospermopsin": "cylindrospermopsin", "saxitoxin": "saxitoxin", "saxitoxins": "saxitoxin",
}

MAP_COLUMNS = ["map_name", "map_lat", "map_lon", "map_status", "latest_anatoxin_a",
               "latest_cylindrospermopsin", "latest_microcystin", "latest_saxitoxin",
               "map_last_sample_date", "site_id"]


# ----------------------------------------------------------------------------- form handling

def hidden_fields(page_html: str) -> dict[str, str]:
    """ASP.NET hidden state fields (__VIEWSTATE, __EVENTVALIDATION, ...) of a page."""
    out = {}
    for name in ("__VIEWSTATE", "__VIEWSTATEGENERATOR", "__VIEWSTATEENCRYPTED", "__EVENTVALIDATION"):
        m = re.search(rf'name="{re.escape(name)}" id="[^"]*" value="([^"]*)"', page_html)
        out[name] = _html.unescape(m.group(1)) if m else ""
    if not out["__VIEWSTATE"]:
        raise ValueError("page has no __VIEWSTATE: not the expected ASP.NET form")
    return out


def export_form(page_html: str, *, start: str = "", end: str = "", toxin: str = "", county: str = "",
                wria: str = "0", site_id: str = "0") -> dict[str, str]:
    """POST body for Data.aspx 'Export Toxin Data'. Blank fields = whole database.

    Dates are ``mm/dd/yyyy`` strings (the form's format); ``toxin`` is a dropdown value
    ('microcystin', 'anatoxin-a', ...).
    """
    p = "ctl00$ContentPlaceHolder1$"
    form = hidden_fields(page_html)
    form.update({
        "__EVENTTARGET": "", "__EVENTARGUMENT": "",
        p + "c_CountyDropDownList": county, p + "c_WRIADropDownList": wria,
        p + "c_SiteDropDownList": site_id, p + "c_ToxinDropDownList": toxin,
        p + "c_LowConcTextBox": "", p + "c_HighConcTextBox": "",
        p + "c_StartDateTextBox": start, p + "c_EndDateTextBox": end,
        p + "c_SampleNumberTextBox": "", p + "c_SiteIDHiddenField": "",
        p + "c_DrawMapHiddenField": "False",
        p + "c_ExportToxinButton": "Export Toxin Data",
    })
    return form


def location_form(page_html: str) -> dict[str, str]:
    form = hidden_fields(page_html)
    form.update({"__EVENTTARGET": "", "__EVENTARGUMENT": "",
                 "ctl00$ContentPlaceHolder1$c_LocationButton": "Get EIM Locations"})
    return form


def _post_attachment(url: str, form: dict, expect: str, *, timeout: float = 900.0,
                     retries: int = 4) -> tuple[bytes, dict]:
    """POST a form and return the attachment bytes; raise unless it is the expected CSV."""
    last = ""
    for attempt in range(retries):
        try:
            r = C.session().post(url, data=form, timeout=timeout)
        except Exception as e:  # noqa: BLE001 — network hiccup → retry
            last = repr(e)
            time.sleep(10.0 * (attempt + 1))
            continue
        disp = r.headers.get("Content-Disposition", "")
        if r.status_code == 200 and "attachment" in disp and expect.lower() in disp.lower():
            return r.content, {"content_type": r.headers.get("Content-Type"), "content_disposition": disp,
                               "bytes": len(r.content)}
        last = f"HTTP {r.status_code} disposition={disp!r} body={r.text[:200]!r}"
        if r.status_code in C.RETRYABLE:
            time.sleep(C._retry_after(r, 10.0 * (attempt + 1)))
            continue
        break
    raise C.HTTPFailure(f"export POST to {url} did not return a {expect!r} attachment: {last}")


def fetch_data_page() -> str:
    return C.get_text(DATA_URL)


def fetch_export(page_html: str | None = None, *, start: str = "", end: str = "",
                 toxin: str = "", sleep: float = 1.5) -> tuple[bytes, dict]:
    """Whole-database toxin export (or a filtered subset) as raw CSV bytes + response headers."""
    page_html = page_html or fetch_data_page()
    time.sleep(sleep)
    return _post_attachment(DATA_URL, export_form(page_html, start=start, end=end, toxin=toxin), "toxin")


def fetch_eim_locations(sleep: float = 1.5) -> tuple[bytes, dict]:
    page = C.get_text(LOCATION_URL)
    time.sleep(sleep)
    return _post_attachment(LOCATION_URL, location_form(page), "EIMLocation")


# ----------------------------------------------------------------------------- parsing: export

def decode_export(data: bytes) -> str:
    """The export is Windows-1252 (µ = 0xB5); strip a UTF-8 BOM defensively."""
    if data.startswith(b"\xef\xbb\xbf"):
        return data[3:].decode("utf-8")
    return data.decode("cp1252")


def parse_export(text: str) -> tuple[pd.DataFrame, list[tuple[int, str]]]:
    """Export CSV text → (all-string DataFrame with :data:`RAW_COLUMNS`, unparsed (line_no, line)).

    Raises if the header differs from :data:`EXPORT_HEADER` (schema change → re-verify).
    """
    lines = text.splitlines()
    if not lines:
        return pd.DataFrame(columns=RAW_COLUMNS), []
    header = [h.strip() for h in lines[0].split(",")]
    if header != EXPORT_HEADER:
        raise ValueError(f"unexpected export header: {header}")
    rows, bad = [], []
    for k, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        m = _ROW_RE.match(line)
        if m is None:
            bad.append((k, line))
            continue
        rows.append([m.group(c) for c in RAW_COLUMNS])
    return pd.DataFrame(rows, columns=RAW_COLUMNS, dtype=str), bad


def normalize_toxin(values: pd.Series) -> pd.Series:
    """Toxin names → canonical lower-case names; unknown names → <NA> (caller decides)."""
    s = pd.Series(values, dtype="object").map(lambda v: re.sub(r"\s+", " ", str(v)).strip().lower())
    return s.map(_TOXIN_ALIASES).astype("string")


_INEQ_RE = re.compile(r"^\s*(?P<op>[<>])\s*(?P<lim>[0-9]*\.?[0-9]+)\s*$")


def parse_concentration(conc: pd.Series, mdl: pd.Series) -> pd.DataFrame:
    """Result strings → value / censoring columns (no substitution of detection limits).

    * numeric ``'2.680'`` → value 2.68, censored False
    * ``'<MDL'`` → censored, censor_type '<', censor_limit = MDL (NaN if MDL blank)
    * ``'<0.05'`` → censored '<' at 0.05;  ``'>100'`` → censored '>' (right-censored) at 100
    * ``'<RDL'`` → censored '<' with unknown reporting limit (censor_limit NaN)
    * ``'NA'``, ``'N/A'``, ``''`` → result_missing True
    """
    conc = pd.Series(conc, dtype="object").fillna("").astype(str).str.strip()
    det_lim, _ = C.to_number(mdl)
    value, qual = C.to_number(conc)
    out = pd.DataFrame({"value_raw": conc, "value": value, "detection_limit": det_lim.to_numpy()},
                       index=conc.index)
    out["censored"] = False
    out["censor_type"] = ""
    out["censor_limit"] = np.nan
    up = conc.str.upper()
    is_mdl = up.isin(["<MDL", "< MDL", "ND", "<DL"])
    is_rdl = up.isin(["<RDL", "<RL", "<MRL", "<PQL"])
    ineq = conc.str.extract(_INEQ_RE)
    has_ineq = ineq["op"].notna()
    out.loc[is_mdl, ["censored", "censor_type"]] = [True, "<"]
    out.loc[is_mdl, "censor_limit"] = out.loc[is_mdl, "detection_limit"]
    out.loc[is_rdl, ["censored", "censor_type"]] = [True, "<"]
    out.loc[has_ineq, "censored"] = True
    out.loc[has_ineq, "censor_type"] = ineq.loc[has_ineq, "op"]
    out.loc[has_ineq, "censor_limit"] = pd.to_numeric(ineq.loc[has_ineq, "lim"])
    out["value"] = out["value"].mask(out["censored"])
    out["result_missing"] = out["value"].isna() & ~out["censored"]
    out["qualifier"] = qual.where(qual != "", "")
    out["censored"] = out["censored"].astype(bool)
    return out


def _exceed(value: pd.Series, censored: pd.Series, ctype: pd.Series, limit: pd.Series,
            threshold: pd.Series, *, inclusive: bool) -> pd.Series:
    """Exceedance with censoring resolved only when the bound decides it (else <NA>)."""
    thr = pd.to_numeric(threshold, errors="coerce")
    res = pd.Series(pd.NA, index=value.index, dtype="boolean")
    num = value.notna() & thr.notna()
    res[num] = (value[num] >= thr[num]) if inclusive else (value[num] > thr[num])
    left = censored & (ctype == "<") & limit.notna() & thr.notna()
    # true value < limit: below the threshold for sure iff limit <= threshold
    res[left & (limit <= thr)] = False
    right = censored & (ctype == ">") & limit.notna() & thr.notna()
    # true value > limit: above for sure iff limit >= threshold (strict: limit >= thr also works)
    res[right & (limit >= thr)] = True
    return res


def tidy_results(raw: pd.DataFrame, *, wria_names: dict[str, str] | None = None) -> pd.DataFrame:
    """Raw export rows → tidy label table (one row per lab sample × toxin)."""
    df = raw.copy()
    out = pd.DataFrame(index=df.index)
    out["source"] = SOURCE
    out["site_id"] = C.str_id(df["site_id"])
    out["site_name"] = df["site"].str.strip()
    out["county"] = df["county"].str.strip().replace("", pd.NA).astype("string")
    out["wria"] = C.str_id(df["wria"])
    if wria_names:
        out["wria_name"] = out["wria"].map(wria_names).astype("string")
    dates, dflag = C.clean_dates(df["collect_date"], min_date="2000-01-01", fmt="%m/%d/%Y")
    out["date"] = dates.dt.date
    out["year"] = dates.dt.year.astype("Int64")
    out["date_flag"] = dflag
    out["toxin_raw"] = df["parameter"]
    out["toxin"] = normalize_toxin(df["parameter"])
    conc = parse_concentration(df["conc"], df["mdl"])
    out["value"] = conc["value"]
    out["unit"] = "ug/L"
    out["value_raw"] = conc["value_raw"]
    out["censored"] = conc["censored"]
    out["censor_type"] = conc["censor_type"]
    out["censor_limit"] = conc["censor_limit"]
    out["detection_limit"] = conc["detection_limit"]
    out["result_missing"] = conc["result_missing"]
    out["qualifier"] = conc["qualifier"]
    out["exceed_published"] = df["exceed_ind"].map({"Yes": True, "No": False}).astype("boolean")
    out["guideline_wa_ugL"] = out["toxin"].map(WA_GUIDELINES_UGL).astype(float)
    out["exceed_wa_guideline"] = _exceed(out["value"], out["censored"], out["censor_type"],
                                         out["censor_limit"], out["guideline_wa_ugL"], inclusive=True)
    mc_thr = pd.Series(np.where(out["toxin"] == "microcystin", EPA_MICROCYSTIN_UGL, np.nan), index=out.index)
    out["exceed_epa_8ugL"] = _exceed(out["value"], out["censored"], out["censor_type"],
                                     out["censor_limit"], mc_thr, inclusive=False)
    out["scum_info"] = df["scum_ind"].map({"Yes": True, "No": False}).astype("boolean")
    out["location_text"] = df["location"].str.strip()
    out["ecology_tracking_number"] = df["client_locator"].str.strip().replace("", pd.NA).astype("string")
    out["lab_sample_num"] = df["lab_sample_num"].str.strip().replace("", pd.NA).astype("string")
    out["lab_sample_num_norm"] = (out["lab_sample_num"].str.upper().str.replace(r"\s+", "", regex=True)
                                  .astype("string"))
    key = (out["site_id"].astype(str) + ":" + pd.Series(out["date"].astype(str), index=out.index) + ":"
           + out["lab_sample_num_norm"].fillna("NOLAB").astype(str) + ":" + out["toxin"].fillna("unknown").astype(str))
    dup_rank = key.groupby(key).cumcount()
    out.insert(1, "record_id", key.where(dup_rank == 0, key + "#" + (dup_rank + 1).astype(str)))
    return out.reset_index(drop=True)


def _any3(s: pd.Series):
    """Three-valued OR: True if any True, else <NA> if any NA, else False."""
    s = s.astype("boolean")
    if s.fillna(False).any():
        return True
    if s.isna().any():
        return pd.NA
    return False


def siteday_labels(results: pd.DataFrame) -> pd.DataFrame:
    """Site × date × toxin label table (replicates / multiple sampling points pooled)."""
    r = results[results["toxin"].notna() & results["date"].notna()].copy()
    g = r.groupby(["site_id", "date", "toxin"], sort=True)
    out = g.agg(site_name=("site_name", "first"), county=("county", "first"), wria=("wria", "first"),
                n_results=("record_id", "size"),
                n_quantified=("value", lambda v: int(v.notna().sum())),
                max_value=("value", "max"),
                n_censored=("censored", "sum"),
                max_censor_limit=("censor_limit", "max")).reset_index()
    for col in ("exceed_wa_guideline", "exceed_epa_8ugL", "exceed_published", "scum_info"):
        out[col + "_any"] = g[col].agg(_any3).astype("boolean").to_numpy()
    out["year"] = pd.to_datetime(out["date"]).dt.year.astype("Int64")
    out["unit"] = "ug/L"
    out["guideline_wa_ugL"] = out["toxin"].map(WA_GUIDELINES_UGL).astype(float)
    return out


# ----------------------------------------------------------------------------- parsing: sites

def parse_map_sites(page_html: str) -> pd.DataFrame:
    """Data.aspx hidden map field → one row per site (coordinates + current map status)."""
    m = re.search(r'id="ctl00_ContentPlaceHolder1_c_SiteInfoHiddenField" value="([^"]*)"', page_html)
    if m is None:
        raise ValueError("SiteInfoHiddenField not found in Data.aspx page")
    value = _html.unescape(_html.unescape(m.group(1)))  # the field is double-escaped
    recs = [r.split("\t") for r in value.split("||") if r.strip()]
    bad = [r for r in recs if len(r) != len(MAP_COLUMNS)]
    if bad:
        raise ValueError(f"{len(bad)} map records with unexpected field count, e.g. {bad[0]}")
    df = pd.DataFrame(recs, columns=MAP_COLUMNS, dtype=str)
    df["site_id"] = C.str_id(df["site_id"])
    return df


def parse_dropdown(page_html: str, name_suffix: str) -> pd.DataFrame:
    """Options of a Data.aspx <select> whose name ends with ``name_suffix`` → (value, label)."""
    m = re.search(rf'(?is)<select[^>]*name="[^"]*{re.escape(name_suffix)}"[^>]*>(.*?)</select>', page_html)
    if m is None:
        raise ValueError(f"select {name_suffix} not found")
    opts = re.findall(r'(?is)<option[^>]*value="([^"]*)"[^>]*>(.*?)</option>', m.group(1))
    return pd.DataFrame([(_html.unescape(v), _html.unescape(t).strip()) for v, t in opts],
                        columns=["value", "label"])


def wria_names(page_html: str) -> dict[str, str]:
    dd = parse_dropdown(page_html, "c_WRIADropDownList")
    out = {}
    for v, lab in zip(dd["value"], dd["label"]):
        mm = re.match(r"^\s*(\d+)\s*-\s*(.+)$", lab)
        if mm and v != "0":
            out[str(int(mm.group(1)))] = mm.group(2).strip()
    return out


def parse_eim_locations(data: bytes) -> pd.DataFrame:
    """EIMLocation.csv (quoted CSV, Windows-1252) → all-string DataFrame with string SiteID."""
    df = pd.read_csv(io.StringIO(decode_export(data)), dtype=str, keep_default_na=False)
    need = {"Location_ID", "Location_Name", "County", "Latitude_Decimal_Degrees",
            "Longitude_Decimal_Degrees", "SiteID"}
    if not need.issubset(df.columns):
        raise ValueError(f"unexpected EIM location columns: {list(df.columns)}")
    df["SiteID"] = C.str_id(df["SiteID"])
    return df


def build_sites(eim: pd.DataFrame, map_sites: pd.DataFrame | None, results: pd.DataFrame | None
                ) -> pd.DataFrame:
    """Site table keyed by ``site_id``: coordinates (EIM export; identical to the map layer),
    county, EIM code, map status snapshot and result coverage."""
    s = pd.DataFrame({"site_id": eim["SiteID"]})
    s["eim_location_id"] = eim["Location_ID"].replace("", pd.NA).astype("string")
    county = eim["County"].str.strip()
    # EIM Location_Name is "<site name>, <County>"; drop the county suffix
    name = pd.Series([n[: -len(", " + c)] if c and n.endswith(", " + c) else n
                      for n, c in zip(eim["Location_Name"].str.strip(), county)], index=eim.index)
    s["county"] = county.replace("", pd.NA).astype("string")
    lat, lon, flag = C.clean_latlon(eim["Latitude_Decimal_Degrees"], eim["Longitude_Decimal_Degrees"],
                                    bbox=WA_BBOX)
    s["latitude"], s["longitude"], s["coord_flag"] = lat.to_numpy(), lon.to_numpy(), flag.to_numpy()
    s["coord_source"] = np.where(s["latitude"].notna(), "EIMLocation.csv (= Data.aspx map layer)", "")
    s["horizontal_datum_code"] = eim.get("Horizontal_Datum", pd.Series("", index=eim.index)).to_numpy()
    s["coord_accuracy_code"] = eim.get("Horizontal_Coordinate_Accuracy", pd.Series("", index=eim.index)).to_numpy()
    s["site_name"] = list(name)
    if map_sites is not None and len(map_sites):
        mp = map_sites.set_index("site_id")
        nm = s["site_id"].map(mp["map_name"])
        s["site_name"] = nm.where(nm.notna() & (nm.str.strip() != ""), s["site_name"])
        s["map_status_snapshot"] = s["site_id"].map(mp["map_status"]).astype("string")
        ld, _ = C.clean_dates(s["site_id"].map(mp["map_last_sample_date"]), fmt="%m/%d/%Y")
        s["map_last_sample_date"] = ld.dt.date
        s["in_map_layer"] = s["site_id"].isin(mp.index)
    s["name_has_river_or_creek"] = s["site_name"].astype(str).str.contains(
        r"\b(?:River|Creek|Slough)\b", case=False, regex=True, na=False)
    if results is not None and len(results):
        extra = sorted(set(results["site_id"].dropna()) - set(s["site_id"]))
        if extra:  # result sites missing from the EIM list (keep them, without coordinates)
            add = results[results["site_id"].isin(extra)].groupby("site_id").agg(
                site_name=("site_name", "first"), county=("county", "first")).reset_index()
            add["coord_flag"] = "missing"
            s = pd.concat([s, add], ignore_index=True)
        ok = results[results["date"].notna()]
        r = ok.groupby("site_id")
        cov = pd.DataFrame({
            "n_results": r.size(),
            "n_lab_samples": r["lab_sample_num_norm"].nunique(),
            "first_date": r["date"].min(), "last_date": r["date"].max(),
            "toxins_tested": r["toxin"].agg(lambda t: ",".join(sorted(set(t.dropna())))),
            "wria": r["wria"].agg(lambda w: w.mode().iloc[0] if w.notna().any() else pd.NA),
            "n_exceed_wa": r["exceed_wa_guideline"].agg(lambda x: int(x.fillna(False).sum())),
        })
        s = s.merge(cov, left_on="site_id", right_index=True, how="left")
        for c in ("n_results", "n_lab_samples", "n_exceed_wa"):
            s[c] = s[c].fillna(0).astype(int)
        s["in_results"] = s["n_results"] > 0
        s["wria"] = s["wria"].astype("string")
    return s.sort_values("site_id", key=lambda x: x.astype(int)).reset_index(drop=True)


# ----------------------------------------------------------------------------- summaries / checks

def parse_summary_table(page_html: str) -> pd.DataFrame:
    """YearSummary / StateSummary page → DataFrame with integer counts and canonical toxin."""
    tables = [t for t in C.html_tables(page_html) if t and "Parameter" in t[0]]
    if not tables:
        raise ValueError("no summary table with a 'Parameter' column")
    df = C.table_to_frame(tables[0])
    df = df.rename(columns={"Number of Samples": "n_samples", "Number of Exceedings": "n_exceedances",
                            "Parameter": "parameter", "Year": "year"})
    df["toxin"] = normalize_toxin(df["parameter"])
    for c in ("n_samples", "n_exceedances", "year"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="raise").astype(int)
    return df


def parse_guidelines(about_html: str) -> dict[str, float]:
    """'Microcystin: >= 8 µg/L' lines on About.aspx → {toxin: value}."""
    text = _html.unescape(re.sub(r"<[^>]+>", " ", about_html))
    out = {}
    for name, val in re.findall(r"([A-Za-z][A-Za-z\-]+)\s*:\s*>=\s*([0-9.]+)\s*(?:µg/L|ug/L)", text):
        tox = _TOXIN_ALIASES.get(name.strip().lower())
        if tox:
            out[tox] = float(val)
    return out


def crosscheck_year_summary(results: pd.DataFrame, year_summary: pd.DataFrame) -> pd.DataFrame:
    """Compare export counts with the published YearSummary table per year × toxin.

    Observed 2026-10-22: the site's "Number of Exceedings" equals the number of results with
    ``ExceedInd = Yes`` AND a quantified value strictly ABOVE the guideline
    (``n_flag_and_gt``); the ``ExceedInd`` flag itself uses >= (``n_exceed_published``), and the
    computed label ``exceed_wa_guideline`` (>=, censoring-aware) is ``n_exceed_wa_ge``.
    """
    r = results[results["toxin"].notna()].copy()
    r["_gt"] = (r["value"] > r["guideline_wa_ugL"]).fillna(False)
    r["_flag_gt"] = r["_gt"] & r["exceed_published"].fillna(False).astype(bool)
    g = r.groupby(["year", "toxin"]).agg(
        export_n=("record_id", "size"),
        n_exceed_published=("exceed_published", lambda x: int(x.fillna(False).sum())),
        n_exceed_wa_ge=("exceed_wa_guideline", lambda x: int(x.fillna(False).sum())),
        n_value_gt_guideline=("_gt", "sum"),
        n_flag_and_gt=("_flag_gt", "sum"),
    ).reset_index()
    ys = year_summary.rename(columns={"n_samples": "summary_n", "n_exceedances": "summary_exceed"})
    m = ys[["year", "toxin", "summary_n", "summary_exceed"]].merge(g, on=["year", "toxin"], how="outer")
    for c in ("summary_n", "summary_exceed", "export_n", "n_exceed_published", "n_exceed_wa_ge",
              "n_value_gt_guideline", "n_flag_and_gt"):
        m[c] = m[c].fillna(0).astype(int)
    m["d_samples"] = m["export_n"] - m["summary_n"]
    m["d_exceed"] = m["n_flag_and_gt"] - m["summary_exceed"]
    return m.sort_values(["year", "toxin"]).reset_index(drop=True)


def parse_site_page(page_html: str) -> pd.DataFrame:
    """Per-site Data.aspx?SiteID= table → DataFrame (columns as displayed). For verification."""
    tables = [t for t in C.html_tables(page_html) if t and "Collect Date" in t[0]]
    if not tables:
        return pd.DataFrame()
    return C.table_to_frame(tables[0])


def compare_site_page(page: pd.DataFrame, results: pd.DataFrame, site_id: str) -> dict:
    """Row-multiset agreement between a per-site page and the export rows for that site."""
    if page.empty:
        return {"site_id": site_id, "page_rows": 0, "export_rows": int((results["site_id"] == site_id).sum())}
    pk = (pd.to_datetime(page["Collect Date"], format="%m/%d/%Y").dt.date.astype(str) + "|"
          + normalize_toxin(page["Parameter"]).astype(str) + "|" + page["Toxin Conc. (µg/L)"].str.strip()
          + "|" + page["MDL (µg/L)"].str.strip())
    e = results[results["site_id"] == site_id]
    ek = (e["date"].astype(str) + "|" + e["toxin"].astype(str) + "|" + e["value_raw"].astype(str) + "|"
          + e["detection_limit"].map(lambda v: "" if pd.isna(v) else f"{v:.3f}"))
    pc, ec = pk.value_counts(), ek.value_counts()
    both = pc.index.union(ec.index)
    diff = int((pc.reindex(both, fill_value=0) - ec.reindex(both, fill_value=0)).abs().sum())
    return {"site_id": site_id, "page_rows": len(page), "export_rows": len(e), "row_multiset_diff": diff}


# ----------------------------------------------------------------------------- raw I/O helpers

def write_gz(text_or_bytes: str | bytes, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = text_or_bytes.encode("utf-8") if isinstance(text_or_bytes, str) else text_or_bytes
    with gzip.open(path, "wb") as fh:
        fh.write(data)
    return path


def read_gz_text(path: str | Path, encoding: str = "utf-8") -> str:
    with gzip.open(path, "rb") as fh:
        return fh.read().decode(encoding, errors="replace")
