# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""USGS NWIS legacy WaterServices client (site catalog + daily values + short IV windows).

Legacy services are decommissioned 2027-02-22 (degradation from Nov 2026); see
docs/research/01_sensor_timeseries.md. Daily values (DV) still serve multi-site,
multi-decade requests; instantaneous values (IV) only ~2-month windows. The Water Data
OGC API client (``usgs_waterdata.py``) is the long-term path.

Missing-value hygiene (v1 bug: -999999 leaked into normalisation statistics):
non-numeric codes (Ice, Eqp, ***, Dis, Ssn, ...) and any value <= -99999 become NaN; the
original code is kept in the ``qualifier`` column.
"""

from __future__ import annotations

import io
import re
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests

SITE_URL = "https://waterservices.usgs.gov/nwis/site/"
DV_URL = "https://waterservices.usgs.gov/nwis/dv/"
IV_URL = "https://nwis.waterservices.usgs.gov/nwis/iv/"

# Continuous water-quality parameters of interest (see research report b.1).
WQ_PARAMS = {
    "00010": "temp_c",
    "00300": "do_mgl",
    "00301": "do_pctsat",
    "00095": "spcond_uscm",
    "00400": "ph",
    "63680": "turb_fnu",
    "63675": "turb_ntu",
    "99133": "no3no2_mgl",
    "99137": "no3_mgl",
    "32295": "fdom_qse",
    "32315": "chl_rfu",
    "32316": "chl_ugl",
    "32319": "pc_ugl",
    "32321": "pc_rfu",
    "00480": "salinity_ppt",
}
HYDRO_PARAMS = {"00060": "discharge_cfs", "00065": "gage_ft", "72137": "discharge_tf_cfs"}
ALL_PARAMS = {**WQ_PARAMS, **HYDRO_PARAMS}
STATS = {"00001": "max", "00002": "min", "00003": "mean", "00008": "median"}

STATES = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "HI", "ID", "IL",
          "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE",
          "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD",
          "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "PR", "VI", "GU", "AS", "MP"]

_session = requests.Session()
_session.headers.update({"User-Agent": "SENTINEL-v2 research (public-data download)",
                         "Accept-Encoding": "gzip"})


class ServiceUnavailable(RuntimeError):
    """HTTP 503 from WaterServices (often deterministic, e.g. windows spanning a DST change)."""


def _get(url: str, params: dict, *, retries: int = 6, backoff: float = 10.0, timeout: float = 300,
         retries_503: int = 2):
    last = None
    n503 = 0
    for k in range(retries):
        try:
            r = _session.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r.text
            if r.status_code == 404:  # no data for this request
                return ""
            last = f"HTTP {r.status_code}: {r.text[:200]}"
            if r.status_code == 503:
                n503 += 1
                if n503 >= retries_503:
                    raise ServiceUnavailable(f"{url} {params} :: {last}")
                time.sleep(3.0)
                continue
        except requests.RequestException as e:  # network hiccup
            last = repr(e)
        time.sleep(backoff * (k + 1))
    raise RuntimeError(f"GET failed after {retries} tries: {url} {params} :: {last}")


def read_rdb(text: str) -> pd.DataFrame:
    """Parse USGS RDB (tab-separated, '#' comments, one format row after the header)."""
    lines = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
    if len(lines) < 2:
        return pd.DataFrame()
    body = "\n".join([lines[0]] + lines[2:])
    return pd.read_csv(io.StringIO(body), sep="\t", dtype=str, keep_default_na=False)


def site_series_catalog(state: str, param_codes: list[str], data_type: str = "dv") -> pd.DataFrame:
    """Series catalog rows (site, parameter, stat, begin/end, count) for one state."""
    params = {"format": "rdb", "stateCd": state, "parameterCd": ",".join(param_codes),
              "hasDataTypeCd": data_type, "outputDataTypeCd": data_type,
              "seriesCatalogOutput": "true", "siteStatus": "all"}
    df = read_rdb(_get(SITE_URL, params))
    if df.empty:
        return df
    keep = ["agency_cd", "site_no", "station_nm", "site_tp_cd", "dec_lat_va", "dec_long_va",
            "coord_acy_cd", "dec_coord_datum_cd", "alt_va", "huc_cd", "data_type_cd", "parm_cd",
            "stat_cd", "ts_id", "loc_web_ds", "begin_date", "end_date", "count_nu"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df["state"] = state
    return df


_COL = re.compile(r"^(?P<ts>\d+)_(?P<pc>\d{5})(?:_(?P<st>\d{5}))?$")


def _wide_to_long(df: pd.DataFrame, time_col: str, has_stat: bool) -> pd.DataFrame:
    out = []
    for col in df.columns:
        m = _COL.match(col)
        if not m:
            continue
        cd = col + "_cd"
        raw = df[col].astype(str)
        val = pd.to_numeric(raw.replace("", np.nan), errors="coerce")
        nonnum = raw.where(val.isna() & (raw != ""), "")
        val = val.mask(val <= -99999)
        qual = df[cd].astype(str) if cd in df.columns else pd.Series("", index=df.index)
        qual = (qual + np.where(nonnum != "", " " + nonnum, "")).str.strip()
        part = pd.DataFrame({
            "site_no": df["site_no"].to_numpy(),
            time_col: df[time_col].to_numpy(),
            "parm_cd": m["pc"],
            "stat_cd": m["st"] if has_stat else "",
            "ts_id": m["ts"],
            "value": val.astype("float32").to_numpy(),
            "qualifier": qual.to_numpy(),
        })
        if "tz_cd" in df.columns:
            part["tz_cd"] = df["tz_cd"].to_numpy()
        out.append(part[(part["value"].notna()) | (part["qualifier"] != "")])
    if not out:
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True)


def get_dv(sites: list[str], param_codes: list[str], start: str, end: str,
           stat_codes: list[str] = tuple(STATS)) -> pd.DataFrame:
    """Daily values (long format) for up to ~10 sites per call."""
    params = {"format": "rdb", "sites": ",".join(sites), "parameterCd": ",".join(param_codes),
              "statCd": ",".join(stat_codes), "startDT": start, "endDT": end, "siteStatus": "all"}
    text = _get(DV_URL, params)
    if not text:
        return pd.DataFrame()
    # A multi-site RDB response contains one table per site; split on header lines.
    frames = []
    chunks = re.split(r"\n(?=agency_cd\t)", "\n".join(ln for ln in text.splitlines()
                                                       if not ln.startswith("#")))
    for ch in chunks:
        df = read_rdb(ch)
        if df.empty or "site_no" not in df.columns:
            continue
        frames.append(_wide_to_long(df, "datetime", has_stat=True))
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out.pop("datetime"), errors="coerce").dt.date
    return out


TZ_OFFSETS_H = {"UTC": 0, "GMT": 0, "EST": -5, "EDT": -4, "CST": -6, "CDT": -5, "MST": -7,
                "MDT": -6, "PST": -8, "PDT": -7, "AKST": -9, "AKDT": -8, "HST": -10,
                "AST": -4, "ADT": -3, "SST": -11, "GST": 10, "ChST": 10}


def get_iv(site: str, param_codes: list[str], start: str, end: str) -> pd.DataFrame:
    """Instantaneous values for ONE site over a short window (<= ~2 months on legacy)."""
    params = {"format": "rdb", "sites": site, "parameterCd": ",".join(param_codes),
              "startDT": start, "endDT": end, "siteStatus": "all"}
    df = read_rdb(_get(IV_URL, params))
    if df.empty:
        return df
    out = _wide_to_long(df, "datetime", has_stat=False)
    if out.empty:
        return out
    local = pd.to_datetime(out.pop("datetime"), errors="coerce")
    off = out["tz_cd"].map(TZ_OFFSETS_H)
    if off.isna().any():
        bad = sorted(out.loc[off.isna(), "tz_cd"].unique())
        raise ValueError(f"unknown time zone codes {bad}")
    out["time_utc"] = (local - pd.to_timedelta(off, unit="h")).dt.tz_localize("UTC")
    return out


@dataclass
class DVBatchResult:
    sites: list[str]
    n_rows: int
    path: str
