# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Tests for the dreissenid label clients (USGS NAS + Minnesota DNR infested waters).

Fixtures in tests/fixtures/labels_dreissena/ are real recorded responses (no network here):
* nas_records_sample.json — 17 unmodified NAS API v2 records chosen by key from the
  2026-10-22 pull, with their yrAcc classes;
* nas_pages_species3118_MT_limit1.json — three real paged responses (limit=1);
* mn_infested-waters_2026-10-21_excerpt.xlsx — 33 data rows (plus title, header, blank and
  'End of Worksheet' rows, and the column-description lines) cut from the DNR workbook of
  2026-10-21: original cell XML, styles and strings, re-packed without the other rows/logo
  (the DNR copyright notice does not allow republishing the full workbook);
* mn_infested_page_2026-10-22_excerpt.html — the count paragraph of the DNR web page.
Full-list counts (1,524 rows, 737 zebra-mussel rows, ...) are checked by the downloader and
recorded in the manifest/datasheet, not here. Small synthetic inputs are used only to exercise
edge cases of pure functions.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from sentinel.data.sources import downstream_dreissena as D
from sentinel.data.sources import events_common as C

FX = Path(__file__).parent / "fixtures" / "labels_dreissena"
MN_XLSX = FX / "mn_infested-waters_2026-10-21_excerpt.xlsx"


@pytest.fixture(scope="module")
def nas_sample():
    return json.loads((FX / "nas_records_sample.json").read_text())


@pytest.fixture(scope="module")
def occ(nas_sample):
    ya = {int(k): v for k, v in nas_sample["year_accuracy"].items()}
    return D.parse_nas_records(nas_sample["records"], year_accuracy=ya).set_index("nas_key")


@pytest.fixture(scope="module")
def mn():
    return D.parse_mn_infested(MN_XLSX)


# ----------------------------------------------------------------------------- NAS paging


def test_nas_pager_on_recorded_pages(monkeypatch):
    pages = json.loads((FX / "nas_pages_species3118_MT_limit1.json").read_text())["pages"]
    by_offset = {int(p["offset"]): p for p in pages}
    calls = []

    def fake_get_json(url, params=None, **kw):
        assert url == D.NAS_SEARCH_URL
        calls.append(dict(params))
        return by_offset[int(params["offset"])]

    monkeypatch.setattr(C, "get_json", fake_get_json)
    got = D.nas_fetch_pages(3118, page_size=1, sleep=0, params={"state": "MT"})
    # the page holding the last record already says endOfRecords == "true" → 2 requests, not 3
    assert [c["offset"] for c in calls] == [0, 1]
    assert all(c["species_ID"] == 3118 and c["state"] == "MT" for c in calls)
    recs, rep = D.records_from_pages(got)
    assert [r["key"] for r in recs] == [1322795, 1613854]
    assert rep["n_duplicate_keys"] == 0 and rep["last_end_of_records"] == "true"


def test_records_from_pages_drops_duplicate_keys():
    pages = json.loads((FX / "nas_pages_species3118_MT_limit1.json").read_text())["pages"]
    recs, rep = D.records_from_pages([pages[0], pages[0], pages[1]])  # page repeated
    assert len(recs) == 2 and rep["n_results_raw"] == 3 and rep["n_duplicate_keys"] == 1


def test_year_accuracy_map_and_conflicts():
    m, conflicts = D.year_accuracy_map({"Actual": [1, 2], "Estimated": [3], "Publication year": [2]})
    assert m[1] == "Actual" and m[3] == "Estimated" and m[2] == "conflict" and conflicts == [2]


# ----------------------------------------------------------------------------- NAS parsing


def test_nas_ids_species_and_key_strings(occ):
    assert occ.index.is_unique and len(occ) == 17
    assert set(occ["species_id"]) == {"5", "95"}
    assert occ.loc["116837", "scientific_name"] == "Dreissena bugensis"
    assert occ.loc["116837", "record_id"] == "116837"
    assert occ.loc["116837", "reference_keys"] == "26189" and occ.loc["116837", "reference_year_min"] == 1993


def test_nas_state_normalisation(occ):
    assert (occ.loc["116837", "state"], occ.loc["116837", "country"]) == ("ON", "CA")      # 'ON'
    assert (occ.loc["1927095", "state"], occ.loc["1927095", "country"]) == ("QC", "CA")    # 'QUE'
    assert (occ.loc["615085", "state"], occ.loc["615085", "country"]) == ("MB", "CA")      # 'MB'
    assert (occ.loc["118238", "state"], occ.loc["118238", "country"]) == ("VT", "US")      # 'Vermont'
    assert occ.loc["753375", "country"] == "MX" and pd.isna(occ.loc["753375", "state"])  # 'Mexico'
    assert pd.isna(occ.loc["1702337", "state_raw"]) and pd.isna(occ.loc["1702337", "country"])
    assert D.normalize_state("New York") == ("NY", "US") and D.normalize_state("") == (None, None)


def test_nas_huc_padding_and_placeholders(occ):
    assert occ.loc["116837", "huc12"] == "042600000200" and occ.loc["116837", "huc8"] == "04260000"
    assert occ["huc12"].dropna().str.fullmatch(r"\d{12}").all()
    assert occ["huc8"].dropna().str.fullmatch(r"\d{8}").all()
    # blank HUC12 in the source → missing, not ''
    assert pd.isna(occ.loc["1702337", "huc12"]) and occ.loc["1702337", "huc8"] == "04290008"
    # '00000000' is NAS's placeholder for 'outside the WBD' (Canadian records) → NA + flag
    assert pd.isna(occ.loc["615085", "huc8"])
    assert "huc8_zero_placeholder" in occ.loc["615085", "huc_flag"]
    assert occ.loc["116837", "huc_flag"] == ""


def test_nas_dates_never_invent_days(occ):
    r = occ.loc["116837"]                    # '1992-12', no day
    assert r["date_precision"] == "month" and pd.isna(r["date"])
    assert r["date_start"] == pd.Timestamp("1992-12-01") and r["date_end"] == pd.Timestamp("1992-12-31")
    r = occ.loc["118238"]                    # full date
    assert r["date_precision"] == "day" and r["date"] == pd.Timestamp("1999-06-06")
    assert bool(occ.loc["1425614", "jan1_date_suspect"]) and not bool(occ.loc["118238", "jan1_date_suspect"])


@pytest.mark.parametrize("ymd, precision, start, end", [
    ((2019, 2, 30), "month", "2019-02-01", "2019-02-28"),   # impossible day → month precision
    ((2020, 2, 29), "day", "2020-02-29", "2020-02-29"),
    ((1990, None, None), "year", "1990-01-01", "1990-12-31"),
    ((1990, 13, 1), "year", "1990-01-01", "1990-12-31"),    # impossible month → year precision
    ((None, None, None), "none", None, None),
    ((3026, 1, 1), "none", None, None),                     # future year rejected
])
def test_nas_date_fields_edge_cases(ymd, precision, start, end):
    d = D.nas_date_fields(*ymd)
    assert d["date_precision"] == precision
    if start is None:
        assert pd.isna(d["date_start"]) and pd.isna(d["date_end"])
    else:
        assert d["date_start"] == pd.Timestamp(start) and d["date_end"] == pd.Timestamp(end)


def test_nas_status_and_established(occ):
    assert occ.loc["1932940", "status"] == "locally established" and occ.loc["1932940", "is_established"]
    assert occ.loc["118238", "status"] == "collected" and not occ.loc["118238", "is_established"]
    assert pd.isna(occ.loc["1711736", "status"]) and occ.loc["1711736", "status_raw"] == ""
    assert not occ.loc["1711736", "is_established"]


def test_nas_year_accuracy_tags(occ):
    assert occ.loc["118240", "year_accuracy"] == "Estimated"
    assert occ.loc["116075", "year_accuracy"] == "Publication year"
    untagged = D.parse_nas_records(json.loads((FX / "nas_records_sample.json").read_text())["records"][:1],
                                   year_accuracy={})
    assert untagged.loc[0, "year_accuracy"] == "untagged"


def test_nas_missing_coordinates_flagged(nas_sample):
    rec = dict(nas_sample["records"][0], decimalLatitude=None)   # synthetic edge case
    rec2 = dict(nas_sample["records"][1], decimalLatitude=0.0, decimalLongitude=0.0)
    df = D.parse_nas_records([rec, rec2], year_accuracy={})
    assert df["coord_flag"].tolist() == ["missing", "zero"] and df["latitude"].isna().all()


def test_nas_rejects_duplicate_keys(nas_sample):
    with pytest.raises(ValueError):
        D.parse_nas_records([nas_sample["records"][0]] * 2)


def test_first_detection_by_huc12(nas_sample, occ):
    fd = D.first_detection_by_huc12(occ.reset_index())
    h = nas_sample["huc12_first_detection_case"]          # 040301020403: 4 zebra records
    row = fd[(fd["species_group"] == "zebra") & (fd["huc12"] == h)].iloc[0]
    assert row["first_year_any"] == 1993 and row["n_records"] == 4
    assert row["first_key_any"] == "253700"                 # tie on 1993 → smallest key
    assert row["first_status_any"] == "collected"
    assert row["first_year_established"] == 1993 and row["first_key_established"] == "1425614"
    assert row["n_established"] == 3 and row["huc8"] == "04030102" and row["states"] == "WI"
    # the dreissena_any group includes quagga: HUC12 042600000200 has both species
    anyrow = fd[(fd["species_group"] == "dreissena_any") & (fd["huc12"] == "042600000200")].iloc[0]
    assert anyrow["n_records"] == 3 and anyrow["first_year_any"] == 1990
    assert set(fd["species_group"]) == {"zebra", "quagga", "dreissena_any"}
    assert fd["huc12"].str.len().eq(12).all()


# ----------------------------------------------------------------------------- Minnesota DNR


def test_mn_parse_rows_version_and_placeholders(mn):
    df, meta = mn
    assert meta["list_version"] == "2026-10-21" and len(df) == 33 and meta["header_row"] == 1
    assert not df["waterbody_name"].str.lower().str.startswith("end of worksheet").any()
    text_cols = ["waterbody_name", "county", "species_raw", "year_listed_raw", "confirmed_or_connected_raw",
                 "dow_raw"]
    for c in text_cols:  # 'no data' placeholders are masked, never kept as values
        assert not df[c].fillna("").str.strip().str.lower().eq("no data").any()
        assert not df[c].fillna("").str.contains(r"^\s|\s$|\xa0", regex=True).any()  # whitespace normalised
    assert df["row_number"].tolist() == list(range(3, 36)) and df["record_id"].is_unique
    assert df["record_id"].iloc[0] == "mn_iw_2026-10-21_00003"
    assert df["year_listed"].dtype == "Int64" and df["year_listed"].min() == 1995
    chip = df[df["waterbody_name"].str.startswith("Chippewa River, East Branch")].iloc[0]
    assert chip["extra_cell"] == "44277" and chip["listing_basis"] == "connected"  # stray cell kept, unparsed
    assert df["extra_cell"].notna().sum() == 1


def test_mn_basis_counts_and_names(mn):
    df, _ = mn
    zm = df[df["species_common"] == "zebra mussel"]
    assert len(zm) == 17
    assert zm["listing_basis"].value_counts().to_dict() == {"confirmed": 9, "connected": 7, "unknown": 1}
    assert (zm["species_scientific"] == "Dreissena polymorpha").all()
    assert (df["waterbody_name"] == "Andrew").sum() == 3       # 'Andrew ' (trailing space) normalised
    pine = df[df["waterbody_name"].str.startswith("Pine River from the base of Cross Lake Dam")].iloc[0]
    assert pine["listing_basis"] == "unknown" and pine["confirmed_flag"] == "missing"
    multi = df[df["county_multiple"]]
    assert set(multi["county"]) == {"Multiple (Todd and Stearns)", "Multiple (Goodhue and Rice)", "Multiple",
                                    "Multiple (Carlton and St. Louis)"}


def test_mn_species_normalisation(mn):
    df, _ = mn
    assert D.normalize_mn_species("Eurasian watermilfoil ") == D.normalize_mn_species("Eurasian Watermilfoil")
    assert D.normalize_mn_species("Eurasian watermilfoil")[1] == "Myriophyllum spicatum"
    assert D.normalize_mn_species("VHS")[0] == "viral hemorrhagic septicemia (vhs)"
    assert D.normalize_mn_species("no data") == (None, None)
    assert df["species_common"].notna().all() and df["species_scientific"].notna().all()
    raw_variants = set(df.loc[df["species_common"] == "eurasian watermilfoil", "species_raw"])
    assert raw_variants == {"Eurasian watermilfoil", "Eurasian Watermilfoil"}  # trailing space stripped
    assert set(df["species_common"]) <= set(D.MN_SPECIES)


@pytest.mark.parametrize("text, basis, year, veliger, flag", [
    ("2020", "confirmed", 2020, False, ""),
    ("connected to Big Birch (77-0084) and Sylvia (73-0249)", "connected", None, False, ""),
    ("Connected to Roosevelt (11-0043)", "connected", None, False, ""),
    ("conected to South Long (18-0136)", "connected", None, False, ""),
    ("part of Jefferson chain", "connected", None, False, ""),
    ("connected to X (18-2015)", "connected", None, False, ""),     # DOW digits are not a year
    ("year unknown", "confirmed", None, False, "year_unknown"),
    ("1977; 2008; 2013; 2025", "confirmed", 1977, False, "annotated"),
    ("2017 (dead); 2018 (live)", "confirmed", 2017, False, "annotated"),
    ("2025 (veligers)", "confirmed", 2025, True, "annotated"),
    ("2017 - veligers", "confirmed", 2017, True, "annotated"),
    ("confirmed in Mississippi in 1990", "confirmed", 1990, False, "annotated"),
    ("NA", "unknown", None, False, "missing"),
    ("", "unknown", None, False, "missing"),
])
def test_mn_confirmed_rules(text, basis, year, veliger, flag):
    r = D.parse_mn_confirmed(text)
    assert r["listing_basis"] == basis and r["year_confirmed"] == year
    assert r["veliger_note"] == veliger and r["confirmed_flag"] == flag


@pytest.mark.parametrize("text, dow, dow8, cls", [
    ("21-0242", "21-0242", "21024200", "dow"),
    ("1-0179", "01-0179", "01017900", "dow"),          # single-digit county code in the list
    ("69-0378-01", "69-0378-01", "69037801", "dow"),
    ("99-9999\xa0", "99-9999", "99999900", "dow"),      # trailing NBSP
    ("61006700", "61-0067", "61006700", "dow"),
    ("NA", None, None, "river_na"),
    ("na", None, None, "river_na"),
    ("none, part of Winnibigoshish", None, None, "none"),
    ("NONE", None, None, "none"),
    ("no data", None, None, "missing"),
    ("", None, None, "missing"),
])
def test_mn_dow(text, dow, dow8, cls):
    assert D.parse_mn_dow(text) == (dow, dow8, cls)


def test_mn_dow_columns_in_table(mn):
    df, _ = mn
    d = df[df["dow_class"] == "dow"]
    assert d["dow8"].str.fullmatch(r"\d{8}").all() and d["dow"].str.fullmatch(r"\d{2}-\d{4}(-\d{2})?").all()
    assert df.loc[df["dow_raw"] == "1-0179", "dow8"].tolist() == ["01017900"]          # Hickory
    assert df.loc[df["waterbody_name"] == "Alstead Mine", "dow8"].tolist() == ["18044006"]
    assert df.loc[df["dow_class"] != "dow", "dow8"].isna().all()
    assert df["dow_class"].value_counts().to_dict() == {"dow": 20, "river_na": 9, "none": 4}


def test_mn_listing_lag(mn):
    df, _ = mn
    conf = df[(df["listing_basis"] == "confirmed") & df["year_confirmed"].notna()]
    assert (conf["listing_lag_years"] == conf["year_listed"] - conf["year_confirmed"]).all()
    # waters listed (as connected) before the species was confirmed keep the earlier listing year
    a = df[(df["dow"] == "34-0206") & (df["species_common"] == "zebra mussel")].iloc[0]  # Andrew, Kandiyohi
    assert (a["year_listed"], a["year_confirmed"], a["listing_lag_years"]) == (2018, 2021, -3)
    # waters known before the 1995 'List of Contaminated Waters' are left-censored at 1995
    b = df[df["dow"] == "62-0002"].iloc[0]                                              # Bald Eagle
    assert (b["year_listed"], b["year_confirmed"], b["listing_lag_years"]) == (1995, 1989, 6)
    assert df.loc[df["listing_basis"] != "confirmed", "listing_lag_years"].isna().all()


def test_mn_column_descriptions_and_page():
    lines = D.mn_column_descriptions(MN_XLSX)
    assert any(ln.startswith("Year species was first confirmed, or connected water body") for ln in lines)
    assert any(ln.startswith("DOW number:") and '"none"' in ln for ln in lines)
    counts = D.mn_page_counts((FX / "mn_infested_page_2026-10-22_excerpt.html").read_text())
    assert counts == {"as_of": "September 2026", "zm_confirmed_lakes_wetlands": 398,
                      "zm_connected_waters": 302, "pct_lakes_zm": 5, "mn_lakes_total": 11842}
    assert D.mn_list_version("List of Infested Waters -  October 21, 2026") == "2026-10-21"
    assert D.mn_list_version("no date here") is None
