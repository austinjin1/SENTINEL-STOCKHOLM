# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Stream-metabolism labels (daily GPP / ER / K600) for RiverPulse (RESEARCH_PLAN §8, D8).

Sources
-------
* **Powell Center synthesis** — Appling et al. 2018, *Sci. Data* 5:180292
  (doi:10.1038/sdata.2018.292); USGS data release doi:10.5066/F70864KX, ScienceBase parent
  item ``59bff507e4b091459a5e0982``. Used children: 1 site data (``site_data.tsv``),
  5 model configurations (``config.zip``), 7 model diagnostics (``diagnostics.zip``) and
  8 daily metabolism estimates and predictors (``daily_predictions.zip``). The multi-GB
  children (3 timeseries, 4 model inputs, 6 model outputs) are only catalogued.
* **StreamPULSE** bulk downloads (https://data.streampulse.org/download_bulk): plain HTML
  POST forms without login, under the StreamPULSE data policy (Open Data Commons
  Attribution License v1.0): all public daily model results, model summaries, basic site
  data. Contact names/e-mails in the site table are dropped at parse time.
* **Bernhardt et al. 2022** (*PNAS* 119:e2121976119) figshare CSV ``lotic_site_info_full``
  (doi:10.6084/m9.figshare.19074299; MIT licence): used only as a site table that gives
  NHDPlusV2 COMIDs for StreamPULSE sites. Its CSV has unquoted commas inside site names;
  :func:`read_bernhardt_site_info` repairs them.

Labels are **model-derived** (streamMetabolizer inverse modelling of diel dissolved-oxygen
curves), not direct measurements. Circularity control: the DO series of these sites must
not train an upstream DO model whose outputs are then used as RiverPulse inputs.

Parsing functions take bytes/text/paths and never touch the network; the download helpers
are thin wrappers around :mod:`sentinel.data.sources.events_common`.
"""

from __future__ import annotations

import io
import time
import zipfile
from pathlib import Path
from collections.abc import Iterable

import numpy as np
import pandas as pd
import requests

from . import events_common as C

# ----------------------------------------------------------------------------- constants

SB_ITEM_URL = "https://www.sciencebase.gov/catalog/item/{id}"
SB_CHILDREN_URL = "https://www.sciencebase.gov/catalog/items"
POWELL_DOI = "doi:10.5066/F70864KX"
POWELL_PAPER_DOI = "doi:10.1038/sdata.2018.292"
POWELL_PARENT = "59bff507e4b091459a5e0982"
# Child items of the release (ScienceBase IDs as listed in the parent item body, 2026-10-22).
POWELL_ITEMS = {
    "1_site_data": "59bff64be4b091459a5e098b",
    "2_spatial_data": "5a5e3791e4b06e28e9be48c1",  # returns "not found/secured" since the 2026-03 restructure
    "2a_site_coordinates": "5a5e37bde4b06e28e9be48ca",
    "2b_site_catchments": "5a5e37bbe4b06e28e9be48c6",
    "3_timeseries": "59c03b72e4b091459a5e0b64",
    "4_model_inputs": "59eb9b9de4b0026a55ffe37c",
    "5_model_configurations": "59eb9b86e4b0026a55ffe379",
    "6_model_outputs": "59eb9ba6e4b0026a55ffe37f",
    "7_model_diagnostics": "59eb9bafe4b0026a55ffe382",
    "8_daily_predictions": "59eb9c0ae4b0026a55ffe389",
}
# Files needed for the label tables: item key -> file names.
POWELL_NEEDED = {
    "1_site_data": ["site_data.tsv", "site_data.xml"],
    "5_model_configurations": ["config.zip", "config.xml"],
    "7_model_diagnostics": ["diagnostics.zip", "diagnostics.xml"],
    "8_daily_predictions": ["daily_predictions.zip", "daily_predictions.xml"],
}

SP_BASE = "https://data.streampulse.org"
SP_ENDPOINTS = {  # POST form actions on https://data.streampulse.org/download_bulk
    "daily_results": "/_alldailyres_download",      # all_daily_model_results.csv.zip
    "model_summary": "/_allmodelsumm_download",     # all_model_summary_data.csv.zip
    "basic_site": "/_allbasicsite_download",        # all_basic_site_data.csv.zip
}
SP_POLICY_URL = "https://data.streampulse.org/policy"
SP_LICENSE = "Open Data Commons Attribution License v1.0 (StreamPULSE data policy)"
BERNHARDT_SITE_INFO_URL = "https://ndownloader.figshare.com/files/33900755"  # lotic_site_info_full.csv.zip
BERNHARDT_SITE_INFO_DOI = "doi:10.6084/m9.figshare.19074299"

GPP_ER_UNIT = "g O2 m-2 d-1"
K600_UNIT = "d-1"
MISSING_CODES = (-999999.0, -99999.0, -9999.0)
R_MISSING = {"NA", "NaN", "nan", "NULL", "null"}
# Plausibility thresholds used by Appling et al. 2018 (Technical Validation: Model performance):
# "values between -0.5 and 0.5 are difficult to distinguish from 0".
GPP_NEG_THRESHOLD = -0.5
ER_POS_THRESHOLD = 0.5
RHAT_THRESHOLD = 1.05  # FGDC metadata: "Values near or below 1.05 indicate convergence"

US_REGIONS = {"AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "HI", "ID", "IL",
              "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE",
              "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD",
              "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "PR", "VI", "GU", "AS", "MP"}
US_BBOX = (-180.0, 13.0, -64.0, 72.0)  # W, S, E, N (CONUS + AK + HI + PR/VI + GU)

# daily_predictions.tsv column -> tidy name (units in the name where not GPP/ER/K600).
POWELL_DAILY_COLS = {
    "GPP": "gpp", "GPP.lower": "gpp_lower", "GPP.upper": "gpp_upper",
    "GPP.n_eff": "gpp_n_eff", "GPP.Rhat": "gpp_rhat",
    "ER": "er", "ER.lower": "er_lower", "ER.upper": "er_upper",
    "ER.n_eff": "er_n_eff", "ER.Rhat": "er_rhat",
    "K600": "k600", "K600.lower": "k600_lower", "K600.upper": "k600_upper",
    "K600.n_eff": "k600_n_eff", "K600.Rhat": "k600_rhat",
    "DO.obs": "do_obs_mgl", "DO.sat": "do_sat_mgl", "DO.amp": "do_amp_mgl", "DO.psat": "do_psat_pct",
    "depth": "depth_m", "temp.water": "temp_water_c", "day.length": "day_length_h",
    "discharge": "discharge_m3s", "shortwave": "shortwave_wm2", "velocity": "velocity_ms",
    "DO.tdist80": "do_tdist80_m",
}
POWELL_DIAG_COLS = {
    "K600_daily_sigma_Rhat": "k600_daily_sigma_rhat", "err_obs_iid_sigma_Rhat": "err_obs_iid_sigma_rhat",
    "err_proc_iid_sigma_Rhat": "err_proc_iid_sigma_rhat", "K_median": "k_median_d",
    "K_range": "k_range_d", "neg_GPP": "neg_gpp_pct", "pos_ER": "pos_er_pct", "run_hrs": "run_hrs",
}
SP_DAILY_NUM = {
    "GPP": "gpp", "GPP_lower": "gpp_lower", "GPP_upper": "gpp_upper",
    "ER": "er", "ER_lower": "er_lower", "ER_upper": "er_upper",
    "K600": "k600", "K600_lower": "k600_lower", "K600_upper": "k600_upper",
}
SP_MODEL_NUM = ["O2_GOF", "GPP_95CI", "ER_95CI", "prop_pos_ER", "prop_neg_GPP", "ER_K600_cor",
                "coverage", "kmax"]
SP_DROP_PERSONAL = ("contact", "contactEmail")


# ----------------------------------------------------------------------------- generic parsing

def _read_delim(src: str | bytes | Path | io.IOBase, sep: str) -> pd.DataFrame:
    """Read a delimited file as all strings ('NA' kept as text so it can be classified)."""
    if isinstance(src, (bytes, bytearray)):
        src = io.BytesIO(src)
    elif isinstance(src, str) and ("\n" in src or "\t" in src):
        src = io.StringIO(src)
    return pd.read_csv(src, sep=sep, dtype=str, keep_default_na=False, encoding="utf-8")


def read_tsv(src) -> pd.DataFrame:
    return _read_delim(src, "\t")


def read_csv(src) -> pd.DataFrame:
    return _read_delim(src, ",")


def zip_member(zip_src: str | Path | bytes, suffix: str = ".tsv") -> tuple[str, bytes]:
    """(name, bytes) of the single member ending in ``suffix`` inside a zip archive."""
    zf = zipfile.ZipFile(io.BytesIO(zip_src) if isinstance(zip_src, (bytes, bytearray)) else zip_src)
    with zf:
        names = [n for n in zf.namelist() if n.lower().endswith(suffix) and not n.startswith("__MACOSX")]
        if len(names) != 1:
            raise ValueError(f"expected one *{suffix} member, found {names}")
        return names[0], zf.read(names[0])


def numeric_with_qualifier(values: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Float values; sentinel codes and non-numeric strings → NaN with their text kept.

    R missing markers ('NA', 'NaN') and blanks are plain missing values (empty qualifier).
    """
    v, q = C.to_number(values, missing_codes=MISSING_CODES)
    q = q.where(~q.isin(R_MISSING), "")
    return v, q


def _collect_qualifiers(parts: dict[str, pd.Series], index: pd.Index) -> pd.Series:
    """Row-wise 'col=code;col=code' summary of masked values ('' when nothing was masked)."""
    out = pd.Series("", index=index, dtype="object")
    for col, q in parts.items():
        hit = q != ""
        if hit.any():
            add = col + "=" + q[hit].astype(str)
            out[hit] = np.where(out[hit] == "", add, out[hit] + ";" + add)
    return out


def nwis_id_from_site(site_id: pd.Series) -> pd.Series:
    """'nwis_01021050' → '01021050' (string; leading zeros kept)."""
    s = site_id.astype("string").str.strip()
    out = s.str.extract(r"^nwis_(\d+)$", expand=False)
    return out.astype("string")


def pad_nwis(values: pd.Series | Iterable) -> pd.Series:
    """NWIS site numbers as strings; all-digit IDs shorter than 8 are zero-padded to 8.

    USGS surface-water station numbers have 8–15 digits; spreadsheets often strip the
    leading zero (e.g. '1021050' for '01021050').
    """
    s = C.str_id(values)
    short = s.notna() & s.str.fullmatch(r"\d{1,7}").fillna(False)
    return s.where(~short, s.str.zfill(8)).astype("string")


def resolution_minutes(values: pd.Series) -> pd.Series:
    """'15min' / '15' / 15.0 → 15 (nullable Int64)."""
    s = pd.Series(values, dtype="object").astype(str).str.strip().str.replace(r"\s*min$", "", regex=True)
    return pd.to_numeric(s, errors="coerce").round().astype("Int64")


# ----------------------------------------------------------------------------- Powell Center

def sciencebase_item(item_id: str, *, timeout: float = 300.0) -> dict:
    """ScienceBase item JSON (the catalog can take > 30 s to answer)."""
    return C.get_json(SB_ITEM_URL.format(id=item_id), {"format": "json"}, timeout=timeout)


def sciencebase_children(parent_id: str = POWELL_PARENT, *, timeout: float = 300.0) -> list[dict]:
    d = C.get_json(SB_CHILDREN_URL, {"parentId": parent_id, "format": "json", "max": 100,
                                     "fields": "title,files,facets,provenance,hasChildren"},
                   timeout=timeout)
    return d.get("items", []) if isinstance(d, dict) else []


def item_files(item: dict) -> list[dict]:
    """Flat list of an item's files (direct files and shapefile 'facets') with name/size/url."""
    out = []
    for f in item.get("files", []) or []:
        out.append({"name": f.get("name"), "bytes": f.get("size"), "url": f.get("url"),
                    "content_type": f.get("contentType"), "checksum": f.get("checksum") or None})
    for facet in item.get("facets", []) or []:
        for f in facet.get("files", []) or []:
            out.append({"name": f.get("name"), "bytes": f.get("size"), "url": f.get("url"),
                        "content_type": f.get("contentType"), "checksum": f.get("checksum") or None,
                        "facet": facet.get("name")})
    return out


def item_summary(item: dict) -> dict:
    """Provenance fields for the manifest / datasheet."""
    if "title" not in item:
        return {"error": (item.get("errors") or {}).get("message", "item not available")}
    prov = item.get("provenance") or {}
    files = item_files(item)
    return {"id": item.get("id"), "title": item.get("title"),
            "date_created": prov.get("dateCreated"), "last_updated": prov.get("lastUpdated"),
            "annotation": prov.get("annotation"),
            "n_files": len(files), "total_bytes": int(sum(f["bytes"] or 0 for f in files)),
            "files": [{"name": f["name"], "bytes": f["bytes"]} for f in files]}


def parse_powell_sites(raw: pd.DataFrame) -> pd.DataFrame:
    """site_data.tsv (602 rows, all strings) → tidy site table."""
    df = raw.copy()
    out = pd.DataFrame(index=df.index)
    out["site_id"] = df["site_name"].astype("string").str.strip()
    out["nwis_id"] = pad_nwis(df["nwis_id"].where(~df["nwis_id"].isin(R_MISSING)))
    from_name = nwis_id_from_site(out["site_id"])
    mism = from_name.notna() & out["nwis_id"].notna() & (from_name != out["nwis_id"])
    if mism.any():
        raise ValueError(f"nwis_id disagrees with site_name for {int(mism.sum())} rows")
    out["nwis_id"] = out["nwis_id"].fillna(from_name)
    out["comid"] = C.str_id(df["nhdplus_id"].where(~df["nhdplus_id"].isin(R_MISSING)))
    out["comid_method"] = np.where(out["comid"].notna(),
                                   "NHDPlusV2 reach with centre point nearest the site (Appling 2018)", "")
    out["site_name"] = df["long_name"].astype("string")
    lat, _ = numeric_with_qualifier(df["lat"])
    lon, _ = numeric_with_qualifier(df["lon"])
    out["latitude"], out["longitude"], out["coord_flag"] = C.clean_latlon(lat, lon, bbox=US_BBOX)
    out["coord_datum"] = df["coord_datum"].astype("string")
    quals: dict[str, pd.Series] = {}
    out["altitude_ft"], quals["alt"] = numeric_with_qualifier(df["alt"])
    out["altitude_datum"] = df["alt_datum"].replace("", pd.NA).astype("string")
    out["nwis_site_type"] = df["site_type"].astype("string")
    for c in ("c", "f", "a", "b", "k", "m"):
        out[f"hydgeom_{c}"], quals[f"dvqcoefs.{c}"] = numeric_with_qualifier(df[f"dvqcoefs.{c}"])
    for c in ("canal", "dam", "npdes"):
        v, quals[f"struct.{c}_flag"] = numeric_with_qualifier(df[f"struct.{c}_flag"])
        out[f"struct_{c}_flag"] = v.round().astype("Int64")
    out["qualifier"] = _collect_qualifiers(quals, out.index)
    return out.reset_index(drop=True)


def parse_powell_models(diag_raw: pd.DataFrame, config_raw: pd.DataFrame | None = None) -> pd.DataFrame:
    """diagnostics.tsv (+ config.tsv) → one row per model (site × input resolution)."""
    d = diag_raw.copy()
    out = pd.DataFrame(index=d.index)
    out["site_id"] = d["site"].astype("string").str.strip()
    out["nwis_id"] = nwis_id_from_site(out["site_id"])
    out["resolution_min"] = resolution_minutes(d["resolution"])
    quals = {}
    for src, dst in POWELL_DIAG_COLS.items():
        out[dst], quals[src] = numeric_with_qualifier(d[src])
    for c in ("model_confidence", "site_min_confidence", "site_confidence"):
        out[c] = d[c].astype("string").str.strip().replace("", pd.NA)
    out["qualifier"] = _collect_qualifiers(quals, out.index)
    if config_raw is not None and len(config_raw):
        cfg = pd.DataFrame({
            "site_id": config_raw["site"].astype("string").str.strip(),
            "resolution_min": resolution_minutes(config_raw["resolution"]),
            "required_timestep_d": pd.to_numeric(config_raw["required_timestep"], errors="coerce"),
            "model_name": config_raw["model_name"].astype("string"),
            "k600_lnq_nodes_centers": config_raw["K600_lnQ_nodes_centers"].astype("string"),
            "k600_daily_sigma_sigma": pd.to_numeric(config_raw["K600_daily_sigma_sigma"], errors="coerce"),
            "burnin_steps": pd.to_numeric(config_raw["burnin_steps"], errors="coerce").astype("Int64"),
            "saved_steps": pd.to_numeric(config_raw["saved_steps"], errors="coerce").astype("Int64"),
            "depth_src": config_raw["depth.src"].astype("string"),
        })
        if cfg.duplicated(["site_id", "resolution_min"]).any():
            raise ValueError("duplicate (site, resolution) rows in config.tsv")
        out = out.merge(cfg, on=["site_id", "resolution_min"], how="left", validate="one_to_one")
    if out.duplicated(["site_id", "resolution_min"]).any():
        raise ValueError("duplicate (site, resolution) rows in diagnostics.tsv")
    return out.reset_index(drop=True)


def parse_powell_daily(raw: pd.DataFrame, models: pd.DataFrame | None = None) -> pd.DataFrame:
    """daily_predictions.tsv (all strings) → tidy daily label table.

    ``date`` is the local solar day (04:00 to 03:59 next day) of the estimate. ``models``
    (from :func:`parse_powell_models`) adds the authors' model confidence rating per row.
    """
    df = raw
    out = pd.DataFrame(index=df.index)
    out["site_id"] = df["site_name"].astype("string").str.strip()
    out["nwis_id"] = nwis_id_from_site(out["site_id"])
    dates, dflag = C.clean_dates(df["date"], min_date="2007-01-01", max_date="2017-12-31", fmt="%Y-%m-%d")
    if (dflag != "").any():
        bad = df.loc[dflag != "", "date"].unique()[:5]
        raise ValueError(f"{int((dflag != '').sum())} invalid dates in daily_predictions, e.g. {bad}")
    out["date"] = dates.dt.normalize()
    out["resolution_min"] = resolution_minutes(df["resolution"])
    quals = {}
    for src, dst in POWELL_DAILY_COLS.items():
        out[dst], quals[src] = numeric_with_qualifier(df[src])
    out["gpp_er_unit"] = GPP_ER_UNIT
    out["k600_unit"] = K600_UNIT
    out["qualifier"] = _collect_qualifiers(quals, out.index)
    out = add_plausibility_flags(out)
    rh = out[["gpp_rhat", "er_rhat", "k600_rhat"]]
    out["flag_rhat"] = (rh > RHAT_THRESHOLD).any(axis=1)
    if models is not None:
        m = models[["site_id", "resolution_min", "model_confidence", "site_min_confidence"]]
        out = out.merge(m, on=["site_id", "resolution_min"], how="left", validate="many_to_one")
    if out.duplicated(["site_id", "date"]).any():
        raise ValueError("duplicate (site_id, date) rows in daily_predictions")
    for c in ("gpp_er_unit", "k600_unit", "model_confidence", "site_min_confidence"):
        if c in out:
            out[c] = out[c].astype("category")
    return out.sort_values(["site_id", "date"]).reset_index(drop=True)


def add_plausibility_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Day-level flags using the thresholds of Appling et al. 2018 (GPP < −0.5, ER > 0.5)."""
    df = df.copy()
    df["flag_gpp_negative"] = df["gpp"] < GPP_NEG_THRESHOLD
    df["flag_er_positive"] = df["er"] > ER_POS_THRESHOLD
    return df


def powell_quality_counts(daily: pd.DataFrame) -> dict:
    """Site-day counts under the authors' model-confidence ratings and day-level thresholds."""
    est = daily["gpp"].notna() & daily["er"].notna()
    conf = daily["model_confidence"].astype("string")
    plaus = ~daily["flag_gpp_negative"] & ~daily["flag_er_positive"]
    res = {
        "site_days_total_rows": len(daily),
        "site_days_with_gpp_and_er": int(est.sum()),
        "site_days_model_confidence": {k: int(((conf == k) & est).sum()) for k in ("H", "M", "L")},
        "site_days_model_conf_H_or_M": int((conf.isin(["H", "M"]) & est).sum()),
        "site_days_model_conf_H_or_M_and_plausible": int((conf.isin(["H", "M"]) & est & plaus).sum()),
        "site_days_model_conf_H_and_plausible": int(((conf == "H") & est & plaus).sum()),
        "site_days_plausible_any_conf": int((est & plaus).sum()),
        "site_days_rhat_le_1.05": int((est & ~daily["flag_rhat"]).sum()),
        "sites_by_site_min_confidence": {
            k: int(daily.loc[daily["site_min_confidence"].astype("string") == k, "site_id"].nunique())
            for k in ("H", "M", "L")},
    }
    return res


# ----------------------------------------------------------------------------- StreamPULSE

def post_download(url: str, dest: str | Path, *, data: dict | None = None, retries: int = 5,
                  timeout: float = 600.0, overwrite: bool = False) -> tuple[Path, dict]:
    """POST a StreamPULSE bulk-download form (no credentials) and stream the zip to ``dest``.

    events_common only has GET helpers; this uses its session (project User-Agent) with
    the same retry/backoff and Retry-After rules. Returns (path, response-header summary).
    """
    dest = Path(dest)
    if dest.exists() and not overwrite:
        return dest, {"skipped_existing": True}
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    last = ""
    for attempt in range(retries):
        wait = min(5.0 * (2 ** attempt), 300.0)
        try:
            with C.session().post(url, data=data or {}, stream=True, timeout=timeout) as r:
                if r.status_code in C.RETRYABLE:
                    last = f"HTTP {r.status_code}"
                    ra = r.headers.get("Retry-After", "")
                    time.sleep(min(float(ra), 600.0) if ra.strip().isdigit() else wait)
                    continue
                if r.status_code != 200:
                    raise C.HTTPFailure(f"HTTP {r.status_code} for POST {url}: {r.text[:300]!r}")
                hdr = {"content_type": r.headers.get("Content-Type"),
                       "content_disposition": r.headers.get("Content-Disposition"),
                       "content_length": r.headers.get("Content-Length"),
                       "last_modified": r.headers.get("Last-Modified")}
                with open(part, "wb") as fh:
                    for block in r.iter_content(chunk_size=1 << 20):
                        if block:
                            fh.write(block)
            part.replace(dest)
            return dest, hdr
        except requests.RequestException as e:
            last = repr(e)
            time.sleep(wait)
    raise C.HTTPFailure(f"POST download failed: {url} :: {last}")


def _in_us(region: pd.Series, lat: pd.Series, lon: pd.Series) -> pd.Series:
    """US site: region code is a US state/territory AND coordinates inside the US bbox
    (region 'CA' is California, but this guards against non-US codes that collide)."""
    w, s, e, n = US_BBOX
    inside = (lon >= w) & (lon <= e) & (lat >= s) & (lat <= n)
    return region.astype("string").str.upper().isin(US_REGIONS).fillna(False) & inside.fillna(False)


def parse_sp_sites(raw: pd.DataFrame) -> pd.DataFrame:
    """all_basic_site_data.csv → tidy site table (contact names / e-mails are dropped)."""
    df = raw.drop(columns=[c for c in SP_DROP_PERSONAL if c in raw.columns])
    out = pd.DataFrame(index=df.index)
    out["region"] = df["regionID"].astype("string").str.strip()
    out["site"] = df["siteID"].astype("string").str.strip()
    out["site_id"] = (out["region"] + "_" + out["site"]).astype("string")
    out["site_name"] = df["siteName"].astype("string")
    out["data_source"] = df["dataSource"].astype("string")
    lat, _ = numeric_with_qualifier(df["latitude"])
    lon, _ = numeric_with_qualifier(df["longitude"])
    out["latitude"], out["longitude"], out["coord_flag"] = C.clean_latlon(lat, lon)
    out["usgs_site_no"] = pad_nwis(df["USGSgageID"].where(~df["USGSgageID"].isin(R_MISSING | {"", "0"})))
    for src, dst in (("addDate", "add_date"), ("firstRecord", "first_record"), ("lastRecord", "last_record")):
        out[dst], _ = C.clean_dates(df[src], min_date="1950-01-01")
    out["embargo_years"] = pd.to_numeric(df["embargoYears"], errors="coerce").astype("Int64")
    out["embargo_days_remaining"] = pd.to_numeric(df["embargoDaysRemaining"], errors="coerce").astype("Int64")
    out["variable_list"] = df["variableList"].astype("string")
    out["grab_var_list"] = df["grabVarList"].astype("string")
    out["in_us"] = _in_us(out["region"], out["latitude"], out["longitude"])
    # The portal lists a few sites twice (re-registrations seconds apart, e.g. CO_Wfresdown1,
    # OH_EastChagrin, IA_WalnutCreek12 on 2026-10-19). Keep the earliest registration when the
    # coordinates agree; refuse to guess when they do not.
    g = out.groupby("site_id")
    conflict = (g["latitude"].nunique() > 1) | (g["longitude"].nunique() > 1)
    if conflict.any():
        raise ValueError(f"duplicate StreamPULSE site ids with conflicting coordinates: "
                         f"{conflict[conflict].index.tolist()[:5]}")
    out["n_registrations"] = g["site_id"].transform("size").astype("int64")
    names = g["site_name"].agg(lambda x: "|".join(sorted(set(x.dropna().astype(str)))))
    out["all_site_names"] = out["site_id"].map(names).astype("string")
    out = (out.sort_values(["site_id", "add_date"], na_position="last")
              .drop_duplicates("site_id", keep="first"))
    return out.sort_index().reset_index(drop=True)


def parse_sp_models(raw: pd.DataFrame) -> pd.DataFrame:
    """all_model_summary_data.csv → one row per fitted model (site × year × run)."""
    df = raw
    out = pd.DataFrame(index=df.index)
    out["region"] = df["region"].astype("string").str.strip()
    out["site"] = df["site"].astype("string").str.strip()
    out["site_id"] = (out["region"] + "_" + out["site"]).astype("string")
    out["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    for col in ("start_date", "end_date", "run_finished"):
        out[col], _ = C.clean_dates(df[col], min_date="1990-01-01")
    for c in ("model", "method", "engine", "rm_flagged", "pool", "ode_method", "deficit_src", "interv",
              "fillgaps", "requested_variables"):
        out[c] = df[c].astype("string")
    for c in ("used_rating_curve", "proc_err", "obs_err", "proc_acor", "estimate_areal_depth"):
        out[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    quals = {}
    for c in SP_MODEL_NUM:
        out[c.lower()], quals[c] = numeric_with_qualifier(df[c])
    best = df["current_best"].astype(str).str.strip().map({"1": True, "0": False})
    out["current_best"] = best.astype("boolean")
    out["qualifier"] = _collect_qualifiers(quals, out.index)
    return out.reset_index(drop=True)


def parse_sp_daily(raw: pd.DataFrame, sites: pd.DataFrame | None = None) -> pd.DataFrame:
    """all_daily_model_results.csv → tidy daily label table.

    ``valid_day`` is StreamPULSE's own flag (1 = estimate produced from complete inputs);
    the ``errors``/``warnings`` text explains the missing estimates.
    """
    df = raw
    out = pd.DataFrame(index=df.index)
    out["region"] = df["region"].astype("string").str.strip()
    out["site"] = df["site"].astype("string").str.strip()
    out["site_id"] = (out["region"] + "_" + out["site"]).astype("string")
    dates, dflag = C.clean_dates(df["date"], min_date="1990-01-01")
    out["date"] = dates.dt.normalize()
    out["date_flag"] = dflag.replace("", pd.NA).astype("string")
    out["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    quals = {}
    for src, dst in SP_DAILY_NUM.items():
        out[dst], quals[src] = numeric_with_qualifier(df[src])
    out["gpp_er_unit"] = GPP_ER_UNIT
    out["k600_unit"] = K600_UNIT
    out["region_is_us_code"] = out["region"].str.upper().isin(US_REGIONS).fillna(False).astype(bool)
    out["valid_day"] = df["valid_day"].astype(str).str.strip().map({"1": True, "0": False}).astype("boolean")
    out["warnings"] = df["warnings"].replace("", pd.NA).astype("string")
    out["errors"] = df["errors"].replace("", pd.NA).astype("string")
    out["qualifier"] = _collect_qualifiers(quals, out.index)
    out = add_plausibility_flags(out)
    # Region codes are not case-consistent on the portal ('co_Cameo2' vs site row 'CO_Cameo2'),
    # and one site has two published IDs with different model runs ('MI_Tobacco', 'MI_TOBACCO').
    # IDs are kept as published; this flags site-days that collide case-insensitively.
    ci_key = out["site_id"].str.upper()
    out["dup_site_date_ci"] = pd.DataFrame({"k": ci_key, "d": out["date"]}).duplicated(keep=False).to_numpy()
    if sites is not None:
        exact = set(sites["site_id"])
        by_upper: dict[str, list[str]] = {}
        for sid in sites["site_id"]:
            by_upper.setdefault(str(sid).upper(), []).append(sid)
        lut = {}
        for sid in out["site_id"].dropna().unique():
            if sid in exact:
                lut[sid] = (sid, "exact")
            elif len(by_upper.get(sid.upper(), [])) == 1:
                lut[sid] = (by_upper[sid.upper()][0], "case_insensitive")
            else:
                lut[sid] = (pd.NA, "none")
        out["site_table_id"] = out["site_id"].map(lambda s: lut.get(s, (pd.NA, "none"))[0]).astype("string")
        out["site_match"] = out["site_id"].map(lambda s: lut.get(s, (pd.NA, "none"))[1]).astype("category")
        keep = ["site_id", "data_source", "usgs_site_no", "latitude", "longitude", "in_us"]
        if "comid" in sites:
            keep.append("comid")
        out = out.merge(sites[keep].rename(columns={"site_id": "site_table_id"}), on="site_table_id",
                        how="left", validate="many_to_one")
        out["in_site_table"] = out["site_match"].astype(str) != "none"
        out["in_us"] = out["in_us"].astype("boolean")  # <NA> for sites without a public site row
    for c in ("gpp_er_unit", "k600_unit", "region", "data_source"):
        if c in out:
            out[c] = out[c].astype("category")
    if out.duplicated(["site_id", "date"]).any():
        raise ValueError("duplicate (site_id, date) rows in StreamPULSE daily results")
    return out.sort_values(["site_id", "date"]).reset_index(drop=True)


def sp_quality_counts(daily: pd.DataFrame) -> dict:
    est = daily["gpp"].notna() & daily["er"].notna()
    plaus = ~daily["flag_gpp_negative"] & ~daily["flag_er_positive"]
    valid = daily["valid_day"].fillna(False).astype(bool)
    us = (daily["in_us"].astype("boolean").fillna(False).astype(bool) if "in_us" in daily
          else pd.Series(True, index=daily.index))
    return {
        "rows": len(daily), "sites": int(daily["site_id"].nunique()),
        "valid_day": int(valid.sum()), "valid_day_with_gpp_and_er": int((valid & est).sum()),
        "valid_and_plausible": int((valid & est & plaus).sum()),
        "valid_and_plausible_us": int((valid & est & plaus & us).sum()),
        "sites_us": int(daily.loc[us, "site_id"].nunique()),
    }


# ----------------------------------------------------------------------------- Bernhardt 2022 site info

BERNHARDT_SITE_COLS = ["Site_ID", "Name", "Source", "Lat", "Lon", "epsg_crs", "COMID", "VPU", "StreamOrde",
                       "Azimuth", "TH", "Width", "Width_src", "WS_area_km2", "WS_area_src"]


def read_bernhardt_site_info(text: str) -> pd.DataFrame:
    """Parse ``lotic_site_info_full.csv`` whose ``Name`` field contains unquoted commas.

    Rows with more than 15 fields have the surplus merged back into ``Name`` (field 2);
    the remaining 13 fields are numeric/codes without commas, so the split is unambiguous.
    """
    import csv
    rows = list(csv.reader(io.StringIO(text)))
    header = rows[0]
    if header != BERNHARDT_SITE_COLS:
        raise ValueError(f"unexpected header {header}")
    n = len(header)
    fixed = []
    for r in rows[1:]:
        if not r:
            continue
        if len(r) > n:
            extra = len(r) - n
            r = [r[0], ",".join(r[1:2 + extra])] + r[2 + extra:]
        if len(r) != n:
            raise ValueError(f"cannot repair row {r[:3]}")
        fixed.append(r)
    df = pd.DataFrame(fixed, columns=header, dtype=str)
    out = pd.DataFrame({
        "site_id": df["Site_ID"].astype("string").str.strip(),
        "site_name": df["Name"].astype("string").str.strip(),
        "source": df["Source"].astype("string"),
        "latitude": pd.to_numeric(df["Lat"], errors="coerce"),
        "longitude": pd.to_numeric(df["Lon"], errors="coerce"),
        "epsg_crs": df["epsg_crs"].astype("string"),
        "comid": C.str_id(df["COMID"].where(~df["COMID"].isin(R_MISSING))),
        "vpu": df["VPU"].astype("string"),
        "stream_order": pd.to_numeric(df["StreamOrde"], errors="coerce").astype("Int64"),
        "ws_area_km2": pd.to_numeric(df["WS_area_km2"], errors="coerce"),
        "ws_area_src": df["WS_area_src"].astype("string"),
        "width_m": pd.to_numeric(df["Width"], errors="coerce"),
        "width_src": df["Width_src"].astype("string"),
    })
    return out
