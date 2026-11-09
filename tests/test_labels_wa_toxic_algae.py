# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Fixture-based tests for the Washington State Toxic Algae client (no network access).

Fixtures in tests/fixtures/labels_wa_toxic_algae/ are real recorded nwtoxicalgae.org responses
retrieved 2026-10-22: a byte-exact subset of the "Export Toxin Data" CSV (all 260 rows of
SiteID 15 plus rows exercising every value format), the matching per-site page for SiteID 15,
the Data.aspx page (form state + map layer), Year/State summary pages, About.aspx and a subset
of EIMLocation.csv. Small synthetic arrays appear only in the exceedance-logic unit test.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sentinel.data.sources import events_wa_toxic_algae as W

FIX = Path(__file__).parent / "fixtures" / "labels_wa_toxic_algae"


def _read(name: str) -> str:
    return (FIX / name).read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def export_bytes() -> bytes:
    return (FIX / "toxin_export_subset.csv").read_bytes()


@pytest.fixture(scope="module")
def raw(export_bytes):
    df, bad = W.parse_export(W.decode_export(export_bytes))
    assert bad == []
    return df


@pytest.fixture(scope="module")
def res(raw):
    return W.tidy_results(raw, wria_names=W.wria_names(_read("data_page.html")))


def _true(x) -> bool:
    return x is not pd.NA and bool(x) is True


def _false(x) -> bool:
    return x is not pd.NA and bool(x) is False


def _row(res, site_id, date, toxin, value_raw=None):
    m = (res["site_id"] == site_id) & (res["date"] == dt.date.fromisoformat(date)) & (res["toxin"] == toxin)
    if value_raw is not None:
        m &= res["value_raw"] == value_raw
    sub = res[m]
    assert len(sub) >= 1, (site_id, date, toxin, value_raw)
    return sub.iloc[0]


# ----------------------------------------------------------------------------- export parsing

def test_export_is_cp1252_with_expected_header(export_bytes):
    text = W.decode_export(export_bytes)
    assert text.splitlines()[0].split(",") == W.EXPORT_HEADER
    assert "µg/L" in text.splitlines()[0]  # 0xB5 decoded as micro sign → unit is µg/L


def test_every_line_parsed(export_bytes, raw):
    n_lines = sum(1 for ln in W.decode_export(export_bytes).splitlines()[1:] if ln.strip())
    assert len(raw) == n_lines == 321
    assert list(raw.columns) == W.RAW_COLUMNS


def test_unquoted_commas_and_quotes_in_location_survive(res):
    r = _row(res, "240", "2026-07-21", "anatoxin-a")
    assert r["location_text"] == "22612 ALPINE DR, Lake Wenatchee, WA 98826"
    assert r["ecology_tracking_number"] == "FISHCHEL240_26_095"
    assert r["site_name"] == "Fish Lake" and r["county"] == "Chelan"
    assert _row(res, "98", "2025-11-04", "microcystin")["location_text"] == 'SW"Beach"'
    assert _row(res, "64", "2022-08-08", "microcystin")["location_text"] == "46°59'45.4\"N 122°46'15.5\"W"
    # a Location with an empty comma-separated part
    assert (res["location_text"] == "NE corner, , 47.6975, -121.91666").sum() >= 1


def test_bad_header_raises_and_malformed_line_reported(export_bytes):
    text = W.decode_export(export_bytes)
    lines = text.splitlines()
    with pytest.raises(ValueError):
        W.parse_export("\n".join([lines[0].replace("ExceedInd", "Exceed")] + lines[1:3]))
    broken = lines[1].rsplit(",", 1)[0] + ",not-a-site-id"   # mutate a real line inside the test
    df, bad = W.parse_export("\n".join([lines[0], lines[1], broken]))
    assert len(df) == 1 and len(bad) == 1 and bad[0][0] == 3


# ----------------------------------------------------------------------------- values, censoring

def test_left_censored_mdl_keeps_detection_limit(res):
    r = _row(res, "600", "2025-06-04", "microcystin")
    assert r["value_raw"] == "<MDL" and bool(r["censored"]) and r["censor_type"] == "<"
    assert np.isnan(r["value"])                       # never substituted
    assert r["censor_limit"] == pytest.approx(0.300) and r["detection_limit"] == pytest.approx(0.300)
    assert _false(r["exceed_wa_guideline"]) and _false(r["exceed_epa_8ugL"])


def test_explicit_inequality_and_reporting_limit(res):
    r = _row(res, "182", "2017-06-12", "anatoxin-a", "<0.05")
    assert bool(r["censored"]) and r["censor_limit"] == pytest.approx(0.05)
    assert r["detection_limit"] == pytest.approx(0.010)  # MDL column kept separately
    rdl = _row(res, "124", "2024-08-20", "saxitoxin")
    assert rdl["value_raw"] == "<RDL" and bool(rdl["censored"]) and np.isnan(rdl["censor_limit"])
    assert pd.isna(rdl["exceed_wa_guideline"])          # unknown limit → undetermined
    assert pd.isna(rdl["lab_sample_num"])                # blank lab number → <NA>


def test_right_censored_above_range(res):
    r = _row(res, "120", "2007-08-29", "microcystin")
    assert r["value_raw"] == ">100" and r["censor_type"] == ">" and r["censor_limit"] == 100
    assert np.isnan(r["value"])
    assert _true(r["exceed_wa_guideline"]) and _true(r["exceed_epa_8ugL"])
    assert _false(r["exceed_published"])  # published flag error kept as published


def test_missing_results_are_not_censored(res):
    for site, date, raw_val in [("41", "2016-07-12", "N/A"), ("35", "2016-08-16", "NA"), ("96", "2020-08-31", "")]:
        r = _row(res, site, date, "anatoxin-a", raw_val)
        assert bool(r["result_missing"]) and not bool(r["censored"]) and np.isnan(r["value"])
        assert pd.isna(r["exceed_wa_guideline"])


def test_blank_mdl_makes_censored_label_undetermined(res):
    r = _row(res, "1", "2009-10-26", "microcystin")
    assert bool(r["censored"]) and np.isnan(r["detection_limit"]) and pd.isna(r["exceed_wa_guideline"])
    assert pd.isna(r["county"]) and pd.isna(r["wria"])


# ----------------------------------------------------------------------------- labels

def test_ties_at_guideline_follow_ge_rule(res):
    r = _row(res, "94", "2024-10-28", "microcystin", "8.000")
    assert _true(r["exceed_published"]) and _true(r["exceed_wa_guideline"])
    assert _false(r["exceed_epa_8ugL"])  # EPA: > 8 µg/L
    a = _row(res, "75", "2009-08-24", "anatoxin-a", "1.000")
    assert _true(a["exceed_wa_guideline"]) and a["guideline_wa_ugL"] == 1.0
    s = _row(res, "59", "2009-08-10", "saxitoxin")
    assert s["value"] == 193.0 and _true(s["exceed_wa_guideline"])


def test_published_flag_error_is_kept_but_label_recomputed(res):
    r = _row(res, "39", "2016-10-03", "microcystin")
    assert r["value"] == 73.0
    assert _false(r["exceed_published"]) and _true(r["exceed_wa_guideline"])
    assert pd.isna(_row(res, "75", "2009-08-24", "anatoxin-a")["exceed_epa_8ugL"])  # MC-only label


def test_exceed_logic_unit():
    value = pd.Series([9.0, 8.0, np.nan, np.nan, np.nan, np.nan])
    cens = pd.Series([False, False, True, True, True, True])
    ctype = pd.Series(["", "", "<", "<", ">", ">"])
    lim = pd.Series([np.nan, np.nan, 0.3, 20.0, 100.0, 2.0])
    thr = pd.Series([8.0] * 6)
    ge = W._exceed(value, cens, ctype, lim, thr, inclusive=True)
    gt = W._exceed(value, cens, ctype, lim, thr, inclusive=False)
    assert ge.tolist()[:3] == [True, True, False] and pd.isna(ge[3]) and _true(ge[4]) and pd.isna(ge[5])
    assert gt.tolist()[:3] == [True, False, False]


# ----------------------------------------------------------------------------- dates, names, IDs

def test_dates_are_local_calendar_dates(res):
    assert all(isinstance(d, dt.date) for d in res["date"])
    assert (res["date_flag"] == "").all()
    assert res["year"].dtype == "Int64"
    assert _row(res, "120", "2007-08-29", "microcystin")["year"] == 2007


def test_toxin_names_normalised(res):
    assert set(res["toxin"]) <= set(W.WA_GUIDELINES_UGL)
    lower = res[res["toxin_raw"].str.islower()]
    assert len(lower) >= 4 and set(lower["toxin"]) <= {"microcystin", "anatoxin-a"}
    norm = W.normalize_toxin(pd.Series(["Microcystins", " anatoxin-a ", "Saxitoxin", "Nodularin"]))
    assert norm.tolist()[:3] == ["microcystin", "anatoxin-a", "saxitoxin"] and pd.isna(norm[3])


def test_ids_are_strings_and_record_ids_unique(res):
    for c in ("site_id", "wria", "ecology_tracking_number", "lab_sample_num"):
        assert str(res[c].dtype) == "string"
    assert _row(res, "75", "2009-08-24", "anatoxin-a")["wria"] == "17"
    assert _row(res, "240", "2026-07-21", "anatoxin-a")["wria_name"] == "Wenatchee"
    assert res["record_id"].is_unique
    lower_lab = res[res["lab_sample_num"] == "l64787-1"]
    assert (lower_lab["lab_sample_num_norm"] == "L64787-1").all()
    assert not res["lab_sample_num"].dropna().str.contains(r"\s$", regex=True).any()


def test_site_page_matches_export_rows(res):
    page = W.parse_site_page(_read("site_15.html"))
    chk = W.compare_site_page(page, res, "15")
    assert chk == {"site_id": "15", "page_rows": 260, "export_rows": 260, "row_multiset_diff": 0}


def test_siteday_three_valued_any_and_counts(res):
    assert W._any3(pd.Series([True, pd.NA], dtype="boolean")) is True
    assert W._any3(pd.Series([False, pd.NA], dtype="boolean")) is pd.NA
    assert W._any3(pd.Series([False, False], dtype="boolean")) is False
    sd = W.siteday_labels(res)
    assert not sd.duplicated(["site_id", "date", "toxin"]).any()
    assert sd["n_results"].sum() == len(res)
    r = sd[(sd["site_id"] == "182") & (sd["date"] == dt.date(2017, 6, 12)) & (sd["toxin"] == "anatoxin-a")]
    assert int(r["n_results"].iloc[0]) == 3 and _false(r["exceed_wa_guideline_any"].iloc[0])


# ----------------------------------------------------------------------------- sites, form, summaries

def test_map_layer_and_dropdowns():
    page = _read("data_page.html")
    mp = W.parse_map_sites(page)
    assert len(mp) == 507 and mp["site_id"].is_unique
    r = mp[mp["site_id"] == "75"].iloc[0]
    assert r["map_name"] == "Anderson Lake" and r["map_lat"] == "48.018852" and r["map_lon"] == "-122.801244"
    assert (mp["map_lat"] == "").sum() == 23
    wn = W.wria_names(page)
    assert len(wn) == 62 and wn["8"] == "Cedar-Sammamish"


def test_hidden_fields_and_export_form():
    page = _read("data_page.html")
    hf = W.hidden_fields(page)
    assert len(hf["__VIEWSTATE"]) > 100 and len(hf["__EVENTVALIDATION"]) > 100
    form = W.export_form(page)
    assert form["ctl00$ContentPlaceHolder1$c_ExportToxinButton"] == "Export Toxin Data"
    assert form["ctl00$ContentPlaceHolder1$c_StartDateTextBox"] == ""
    assert form["ctl00$ContentPlaceHolder1$c_SiteDropDownList"] == "0"
    with pytest.raises(ValueError):
        W.hidden_fields("<html>no form</html>")


def test_eim_locations_and_site_table(res):
    eim = W.parse_eim_locations((FIX / "eim_locations_subset.csv").read_bytes())
    assert len(eim) == 44 and str(eim["SiteID"].dtype) == "string"
    assert (eim["Location_ID"] == "080Ki").sum() == 2          # EIM code is not a unique key
    sites = W.build_sites(eim, W.parse_map_sites(_read("data_page.html")), res)
    assert sites["site_id"].is_unique
    s = sites.set_index("site_id")
    assert s.loc["240", "site_name"] == "Fish Lake" and s.loc["240", "county"] == "Chelan"
    assert s.loc["75", "latitude"] == pytest.approx(48.018852)
    assert s.loc["15", "n_results"] == 260 and bool(s.loc["15", "in_results"])
    blank = sites[sites["coord_flag"] == "missing"]
    assert len(blank) >= 5 and blank["latitude"].isna().all()
    # county suffix stripped from EIM names: "Hidden Lake Beach, King" → "Hidden Lake Beach"
    assert not sites["site_name"].str.endswith(", King").any()


def test_summary_pages_and_guidelines():
    ys = W.parse_summary_table(_read("year_summary.html"))
    assert len(ys) == 74
    r = ys[(ys["year"] == 2015) & (ys["toxin"] == "microcystin")].iloc[0]
    assert (r["n_samples"], r["n_exceedances"]) == (995, 147)
    ss = W.parse_summary_table(_read("state_summary.html")).set_index("toxin")
    assert ss.loc["microcystin", "n_samples"] == 16575 and ss.loc["microcystin", "n_exceedances"] == 1595
    assert ss.loc["anatoxin-a", "n_samples"] == 12702 and ss.loc["anatoxin-a", "n_exceedances"] == 418
    assert W.parse_guidelines(_read("about.html")) == W.WA_GUIDELINES_UGL


def test_crosscheck_arithmetic(res):
    ys = W.parse_summary_table(_read("year_summary.html"))
    cc = W.crosscheck_year_summary(res, ys)
    assert len(cc) == 74
    r = cc[(cc["year"] == 2024) & (cc["toxin"] == "microcystin")].iloc[0]
    sub = res[(res["year"] == 2024) & (res["toxin"] == "microcystin")]
    assert r["export_n"] == len(sub) and r["summary_n"] == 904
    assert r["d_samples"] == r["export_n"] - r["summary_n"]
    assert r["d_exceed"] == r["n_flag_and_gt"] - r["summary_exceed"]
