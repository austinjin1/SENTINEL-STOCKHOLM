# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Tests for the NOAA Lake Erie bloom-severity client on recorded spreadsheets/bulletin text."""

from pathlib import Path

import pandas as pd
import pytest

from sentinel.data.sources import events_noaa_lake_erie as N

FIX = Path(__file__).parent / "fixtures" / "labels_noaa_lake_erie_hab"


@pytest.fixture(scope="module")
def vintages():
    cur, meta_c = N.parse_severity_xlsx(FIX / "curated_2000to2025.xlsx", "curated_2000to2025")
    rev, meta_r = N.parse_severity_xlsx(FIX / "revised_2000to2024.xlsx", "revised_2000to2024")
    return cur, rev, meta_c, meta_r


def test_curated_vintage(vintages):
    cur, _, meta, _ = vintages
    assert len(cur) == 26 and cur["year"].min() == 2000 and cur["year"].max() == 2025
    si = cur.set_index("year")["severity_index"]
    assert si[2011] == 10 and si[2015] == 8.4 and si[2023] == 2.9 and si[2025] == 2.4
    # SI is floored at 0.3 by NOAA → left-censored years flagged
    assert set(cur.loc[cur.si_floor_censored, "year"]) == {2000, 2001, 2002, 2005}
    assert cur.loc[cur.year == 2015, "date_ci_max"].item() == pd.Timestamp("2015-08-14")
    assert "set to 0.3" in meta["Data"] and "NCCOS" in meta["Attribution"]


def test_revised_vintage_has_no_area_or_date_columns(vintages):
    _, rev, _, meta = vintages
    assert len(rev) == 25 and rev["year"].max() == 2024
    assert rev["total_area_km2_max"].isna().all() and (rev["date_ci_max_flag"] == "absent_in_vintage").all()
    assert rev.set_index("year")["severity_index"][2023] == 2.5
    assert "reprocessed" in meta["Description"]


def test_vintages_are_kept_separate_and_differ(vintages):
    cur, rev, _, _ = vintages
    long = N.severity_long([cur, rev])
    assert long.groupby("label_vintage").size().to_dict() == {"curated_2000to2025": 26,
                                                              "revised_2000to2024": 25}
    comp = N.vintage_comparison(long).set_index("year")
    assert comp.loc[2017, "si_curated_2000to2025"] == 7.4 and comp.loc[2017, "si_revised_2000to2024"] == 6.8
    assert pd.isna(comp.loc[2025, "si_revised_2000to2024"])


def test_final_assessment_as_issued_values():
    t18 = N.parse_final_assessment((FIX / "final_2018.txt").read_text(), 2018)
    prim = t18[t18.statement_type == "primary"]
    assert prim[["year", "si"]].values.tolist() == [[2018, 3.6]]
    ment = t18[t18.statement_type == "mentioned"].set_index("year")["si"].to_dict()
    assert ment[2015] == 10.5 and ment[2011] == 10.0 and ment[2016] == 3.2 and ment[2017] == 8.0
    for name, season, si in (("final_2021.txt", 2021, 6.0), ("final_2023.txt", 2023, 5.3),
                             ("final_2025.txt", 2025, 2.4)):
        t = N.parse_final_assessment((FIX / name).read_text(), season)
        p = t[t.statement_type == "primary"]
        assert p["year"].tolist() == [season] and p["si"].tolist() == [si], name


def test_seasonal_forecast_parsing_incl_kerning_garbled_text():
    f18 = N.parse_seasonal_forecast((FIX / "forecast_2018.txt").read_text(), 2018)
    assert (f18["forecast_si"], f18["forecast_si_lo"], f18["forecast_si_hi"]) == (6.0, 5.0, 7.5)
    assert f18["issue_date"] == pd.Timestamp("2018-07-12")
    f21 = N.parse_seasonal_forecast((FIX / "forecast_2021.txt").read_text(), 2021)
    assert (f21["forecast_si"], f21["forecast_si_lo"], f21["forecast_si_hi"]) == (3.0, 2.0, 4.5)
    f26 = N.parse_seasonal_forecast((FIX / "forecast_2026.txt").read_text(), 2026)
    assert (f26["forecast_si"], f26["forecast_si_lo"], f26["forecast_si_hi"]) == (3.5, 3.0, 4.5)
    assert f26["issue_date"] == pd.Timestamp("2026-06-25") and f26["parse_ok"]


def test_squash_normalises_kerning_and_quotes():
    assert N.squash("sev erity ind ex of 4. 5’") == "severityindexof4.5'"
