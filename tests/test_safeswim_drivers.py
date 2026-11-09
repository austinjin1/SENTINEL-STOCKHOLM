# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Parser and decision-time tests for the SafeSwim driver sources (synthetic payloads only)."""

from __future__ import annotations

import gzip

import numpy as np
import pandas as pd

from sentinel.data.sources import safeswim_drivers as SD
from sentinel.models.downstream import safeswim as S


def test_parse_usgs_dv_nodata_to_nan():
    payload = {"value": {"timeSeries": [{"variable": {"noDataValue": -999999.0}, "values": [{"value": [
        {"value": "120", "qualifiers": ["A"], "dateTime": "2010-06-01T00:00:00.000"},
        {"value": "-999999", "qualifiers": ["P", "Ice"], "dateTime": "2010-06-02T00:00:00.000"}]}]}]}}
    df = SD.parse_usgs_dv(payload, "05536000")
    assert df["discharge_cfs"].iloc[0] == 120 and np.isnan(df["discharge_cfs"].iloc[1])
    assert df["qualifiers"].iloc[1] == "P,Ice" and df["site_no"].iloc[0] == "05536000"


def test_parse_ndbc_missing_codes_and_legacy_header():
    txt = ("#YY  MM DD hh mm WDIR WSPD GST  WVHT   DPD   APD MWD   PRES  ATMP  WTMP  DEWP  VIS  TIDE\n"
           "#yr  mo dy hr mn degT m/s  m/s     m   sec   sec deg    hPa  degC  degC  degC  nmi    ft\n"
           "2010 06 01 13 50  90  5.0  6.0  99.00 6.25  4.77 999 1004.6  20.0 999.0 999.0 99.0 99.00\n")
    h = SD.parse_ndbc_stdmet(gzip.compress(txt.encode()), "45007")
    assert np.isnan(h["WVHT"].iloc[0]) and np.isnan(h["WTMP"].iloc[0]) and h["WSPD"].iloc[0] == 5.0
    d = SD.ndbc_daily(h)
    assert np.isclose(d["onshore_wind_ms"].iloc[0], 5.0)          # wind from the east (90°) = onshore
    legacy = "YYYY MM DD hh WD  WSPD GST  WVHT  DPD   APD  MWD  BAR    ATMP  WTMP  DEWP  VIS\n" \
             "2005 06 01 13 270  3.0  4.0  0.50  5.00  4.00 999 1010.0  20.0  15.0 999.0 99.0\n"
    h2 = SD.parse_ndbc_stdmet(legacy.encode(), "45007")
    assert h2["WDIR"].iloc[0] == 270 and h2["WTMP"].iloc[0] == 15.0
    assert np.isclose(SD.ndbc_daily(h2)["onshore_wind_ms"].iloc[0], -3.0)   # westerly = offshore


def test_lag_join_uses_previous_day_only():
    daily = pd.DataFrame({"v": [1.0, 2.0, 3.0]}, index=pd.to_datetime(["2022-06-01", "2022-06-02", "2022-06-03"]))
    dates = pd.Series(pd.to_datetime(["2022-06-02", "2022-06-01"]))
    j = S._lag_join(dates, daily)
    assert j["v"].iloc[0] == 1.0 and np.isnan(j["v"].iloc[1])


def test_parse_coops_daily():
    df = SD.parse_coops_daily({"data": [{"t": "2010-01-01 00:00", "v": "176.272", "f": "0,0"},
                                        {"t": "2010-01-02 00:00", "v": "", "f": "0,0"}]})
    assert df["water_level_m_igld"].iloc[0] == 176.272 and np.isnan(df["water_level_m_igld"].iloc[1])
