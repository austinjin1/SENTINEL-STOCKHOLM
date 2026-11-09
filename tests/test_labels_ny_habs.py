# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Fixture-based tests for the NY DEC HABs label client (no network).

Fixtures are real rows recorded on 2026-10-22 from data.ny.gov (qtg2-hjth, 95my-wijm) and the
NYHABS public ArcGIS views. Tiny synthetic frames are used only for edge cases.
"""

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sentinel.data.sources import events_ny_habs as N

FIX = Path(__file__).parent / "fixtures" / "labels_ny_habs"


@pytest.fixture(scope="module")
def raw12():
    return N.read_socrata_csv(FIX / "qtg2-hjth_sample.csv")


@pytest.fixture(scope="module")
def raw19():
    return N.read_socrata_csv(FIX / "95my-wijm_sample.csv")


@pytest.fixture(scope="module")
def wy(raw12, raw19):
    return N.waterbody_years(raw12, raw19, max_date="2026-10-23")


def _feats(name):
    return json.loads((FIX / f"nyhabs_{name}_sample.json").read_text())


def test_bloom_status_mapping_and_order():
    st = N.bloom_status(pd.Series(["S", "C", "HT", "confirmed", "Confirmed_with_High_Toxins", "Confirmed", ""]))
    assert st.tolist()[:6] == ["suspicious", "confirmed", "confirmed_high_toxin", "confirmed",
                               "confirmed_high_toxin", "confirmed"]
    assert pd.isna(st.iloc[6])
    assert st.cat.ordered and st.cat.codes.tolist()[:3] == [0, 1, 2]
    assert (pd.Series(st.iloc[:1].array) < pd.Series(st.iloc[2:3].array)).all()
    assert N.status_rank(st).tolist()[:3] == [1, 2, 3]
    with pytest.raises(ValueError):
        N.bloom_status(pd.Series(["X"]))


def test_eras_harmonised(wy, raw12, raw19):
    assert len(wy) == len(raw12) + len(raw19)
    old, new = wy[wy["era"] == "2012-2018"], wy[wy["era"] == "2019+"]
    assert set(old["bloom_status"].astype(str)) == {"suspicious", "confirmed", "confirmed_high_toxin"}
    assert set(new["bloom_status"].astype(str)) == {"confirmed"}
    assert new["label_high_toxin"].isna().all() and old["label_high_toxin"].notna().all()
    assert (old.loc[old["bloom_status"] == "confirmed_high_toxin", "label_high_toxin"]).all()
    assert (old.loc[old["bloom_status"] == "suspicious", "label_confirmed"] == False).all()
    assert set(old["status_basis"]) == {"visual_or_photo", "water_sampling"}
    assert set(new["status_basis"]) == {"visual_photo_or_sampling"}
    assert new["n_reports"].notna().all() and old["n_reports"].isna().all()
    assert old["n_weeks_listed"].notna().all()
    assert (wy["first_report_date"] <= wy["last_report_date"]).all()
    assert (wy["first_report_flag"] == "").all() and (wy["coord_flag"] == "").all()


def test_ids_are_strings_and_unique(wy):
    assert str(wy["record_id"].dtype) == "string" and wy["record_id"].is_unique
    # Waneta Lake (Steuben 2017, Schuyler 2015) is published twice verbatim
    dups = wy[wy["is_exact_duplicate"]]
    assert len(dups) == 2 and dups["record_id"].str.endswith("#2").all()
    assert set(dups["waterbody_key"]) == {"waneta lake"}
    assert wy["socrata_row_id"].str.startswith("row-").all()
    assert str(wy["dec_region"].dtype) == "string"
    assert set(wy.loc[wy["era"] == "2012-2018", "dec_region"]) <= {str(i) for i in range(1, 10)}


def test_multi_county_rows_and_lake_year_collapse(wy):
    ly = N.lake_years(wy)
    waneta = wy[(wy["waterbody_key"] == "waneta lake") & (wy["year"] == 2013)]
    assert len(waneta) == 2 and set(waneta["county"]) == {"Schuyler", "Steuben"}
    lw = ly[(ly["waterbody_key"] == "waneta lake") & (ly["year"] == 2013)]
    assert len(lw) == 1 and lw.iloc[0]["counties"] == "Schuyler, Steuben" and lw.iloc[0]["n_source_records"] == 2
    # Brooks Lake 2018: Orange (S) and Richmond (C) rows share coordinates → one lake-year, max status
    b18 = ly[(ly["waterbody_key"] == "brooks lake") & (ly["year"] == 2018)]
    assert len(b18) == 1 and b18.iloc[0]["bloom_status"] == "confirmed" and b18.iloc[0]["label_confirmed"]
    # Brooks Lake 2016: two different lakes (Orange vs Staten Island) → two lake-years
    assert len(ly[(ly["waterbody_key"] == "brooks lake") & (ly["year"] == 2016)]) == 2
    # Canandaigua 2017: per-county listings with different dates → earliest first / latest last
    c17 = wy[(wy["waterbody_key"] == "canandaigua lake") & (wy["year"] == 2017)]
    lc = ly[(ly["waterbody_key"] == "canandaigua lake") & (ly["year"] == 2017)].iloc[0]
    assert lc["first_report_date"] == c17["first_report_date"].min()
    assert lc["last_report_date"] == c17["last_report_date"].max()
    assert lc["label_high_toxin"] and lc["status_rank"] == 3
    # 2019+ rows are already lake-years; their high-toxin label stays unknown
    assert ly.loc[ly["era"] == "2019+", "label_high_toxin"].isna().all()
    assert ly["lake_year_id"].is_unique


def test_2019_multi_county_and_name_key(raw19):
    t = N.tidy_summary_2019(raw19, max_date="2026-10-23")
    cay = t[t["waterbody_key"] == "cayuga lake"]
    assert len(cay) and (cay["n_counties"] >= 2).all()
    assert N.name_key(pd.Series(["Brown's Pond", "St. Lawrence  River"])).tolist() == ["browns pond",
                                                                                     "st lawrence river"]


def test_date_and_coordinate_validation_synthetic():
    df = pd.DataFrame({
        "waterbody_name": ["A Lake", "B Lake", "C Lake"], "county": ["X", "Y", "Z"], "year": ["2020", "2021", "2022"],
        "bloom_type": ["confirmed"] * 3,
        "date_first_report": ["2019-08-01T00:00:00.000", "2555-07-01T00:00:00.000", "2022-07-01T00:00:00.000"],
        "date_last_report": ["2020-08-02T00:00:00.000", "2021-08-01T00:00:00.000", "2022-07-02T00:00:00.000"],
        "number_of_reports": ["1", "2", "3"], "latitude": ["42.5", "0", "34.0"],
        "longitude": ["-75.0", "0", "-118.2"],
    })
    t = N.tidy_summary_2019(df, max_date="2026-10-23")
    assert t.loc[0, "first_report_flag"] == "year_mismatch" and t.loc[0, "first_report_date"] == pd.Timestamp("2019-08-01")
    assert pd.isna(t.loc[1, "first_report_date"]) and t.loc[1, "first_report_flag"] == "out_of_range"
    assert t.loc[1, "coord_flag"] == "zero" and np.isnan(t.loc[1, "latitude"])
    assert t.loc[2, "coord_flag"] == "outside_bbox" and np.isnan(t.loc[2, "longitude"])  # Los Angeles


def test_nyhabs_views_merge():
    frames = [N.nyhabs_frame(_feats(v), v) for v in ["archived", "archived_dil", "current", "current_dil"]]
    m = N.merge_nyhabs_views(frames)
    n_arch = len({f["attributes"]["globalid"] for f in _feats("archived")})
    n_cur = len({f["attributes"]["globalid"] for f in _feats("current")})
    assert len(m) == n_arch + n_cur and m["globalid"].is_unique
    assert set(m["views"]) == {"archived;archived_dil", "current;current_dil"}
    assert set(m["window"]) == {"archived", "current_14d"}
    assert m["status_time_utc"].notna().all()            # only present in the DIL views
    assert m["lake_sections"].notna().any()              # only present in the non-DIL archived view
    assert str(m["report_time_utc"].dt.tz) == "UTC"
    r = m[m["globalid"] == "338cada9-c061-416d-8385-460ed7772bb5"].iloc[0]
    assert r["report_time_utc"] == pd.Timestamp("2026-04-23 12:40:00", tz="UTC")
    assert r["date"] == pd.Timestamp("2026-04-23") and r["extent_rank"] == 3
    assert (m["bloom_status"] == "confirmed").all() and (m["coord_flag"] == "").all()
    nr = m[m["extent_bloom"] == "Not Reported"]
    assert len(nr) and nr["extent_rank"].isna().all() and (nr["extent_class"] == "not_reported").all()
    ow = m[m["extent_class"] == "open_water"]
    assert len(ow) and ow["extent_rank"].isna().all()


def test_local_date_uses_new_york_time():
    f = copy.deepcopy(_feats("archived")[:1])
    f[0]["attributes"]["date_time"] = int(pd.Timestamp("2026-07-02 02:30", tz="UTC").timestamp() * 1000)
    r = N.nyhabs_frame(f, "archived").iloc[0]
    assert r["date"] == pd.Timestamp("2026-07-01")      # 22:30 EDT on July 1


def test_accumulate_snapshots_latest_wins():
    m1 = N.merge_nyhabs_views([N.nyhabs_frame(_feats("archived"), "archived")])
    m2 = m1.copy()
    gid = m2["globalid"].iloc[0]
    m2.loc[m2["globalid"] == gid, "extent_bloom"] = "Large Localized"
    m2 = m2.iloc[:3]
    acc = N.accumulate_snapshots({"2026-10-22": m1, "2026-10-29": m2})
    assert len(acc) == len(m1) and acc["globalid"].is_unique
    r = acc[acc["globalid"] == gid].iloc[0]
    assert r["extent_bloom"] == "Large Localized" and r["first_seen"] == "2026-10-22" and r["last_seen"] == "2026-10-29"
    assert r["n_snapshots"] == 2
    only_first = acc[~acc["globalid"].isin(m2["globalid"])]
    assert (only_first["last_seen"] == "2026-10-22").all()
