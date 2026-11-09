# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Parsing tests for the stream-metabolism label clients (Powell Center, StreamPULSE, Bernhardt 2022).

Fixtures in tests/fixtures/labels_metabolism/ are small real subsets recorded on 2026-10-22
(StreamPULSE contact names/e-mails were redacted in the fixture). Values injected below
(sentinel codes) are synthetic and exist only to exercise the masking code paths.
No network access.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sentinel.data.sources import downstream_metabolism as M

FX = Path(__file__).parent / "fixtures" / "labels_metabolism"


# ----------------------------------------------------------------------------- generic helpers

def test_numeric_with_qualifier_masks_codes_and_keeps_text():
    v, q = M.numeric_with_qualifier(pd.Series(["3.5", "NA", "", "-9999", "-999999", "Ice", "1,200"]))
    assert v.iloc[0] == 3.5 and v.iloc[6] == 1200.0
    assert v.iloc[1:6].isna().all()
    # R missing markers and blanks are plain missing values; codes/text are kept as qualifiers
    assert q.tolist() == ["", "", "", "-9999", "-999999", "Ice", ""]


def test_pad_nwis_keeps_leading_zeros_and_long_ids():
    s = M.pad_nwis(pd.Series(["01021050", "1021050", 1021050.0, "410333095530101", None, "nan"]))
    assert s.tolist()[:4] == ["01021050", "01021050", "01021050", "410333095530101"]
    assert s.iloc[4] is pd.NA and s.iloc[5] is pd.NA
    assert M.nwis_id_from_site(pd.Series(["nwis_01021050", "AZ_LV"])).tolist()[0] == "01021050"


def test_resolution_minutes():
    assert M.resolution_minutes(pd.Series(["15min", "60min", "5", 30.0, "x"])).tolist()[:4] == [15, 60, 5, 30]


def test_zip_member_requires_single_match():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("a.tsv", "x\n1\n")
        z.writestr("b.tsv", "x\n2\n")
    with pytest.raises(ValueError):
        M.zip_member(buf.getvalue(), ".tsv")


# ----------------------------------------------------------------------------- Powell Center

def test_sciencebase_item_files_and_summary():
    item = json.loads((FX / "sb_item_8_daily_predictions.json").read_text())
    files = {f["name"]: f for f in M.item_files(item)}
    assert files["daily_predictions.zip"]["bytes"] == 100995905
    assert "daily_predictions.xml" in files
    s = M.item_summary(item)
    assert s["id"] == M.POWELL_ITEMS["8_daily_predictions"] and s["last_updated"]
    shp = json.loads((FX / "sb_item_2a_site_coordinates.json").read_text())
    names = {f["name"] for f in M.item_files(shp)}
    assert "points_shapefile.shp" in names  # shapefile parts live in 'facets'
    assert M.item_summary({"errors": {"message": "Item not found"}})["error"] == "Item not found"


def test_parse_powell_sites():
    raw = M.read_tsv(FX / "powell_site_data_sample.tsv")
    s = M.parse_powell_sites(raw)
    assert s["site_id"].str.startswith("nwis_").all()
    assert (s["site_id"].str[5:] == s["nwis_id"]).all()
    assert s.loc[s.site_id == "nwis_01021050", "nwis_id"].item() == "01021050"
    assert "410333095530101" in set(s["nwis_id"])          # 15-digit IDs untouched
    assert s["comid"].dtype == "string" and s["comid"].notna().all()
    r = s.set_index("site_id")
    assert np.isnan(r.loc["nwis_01021050", "altitude_ft"])   # 'NA' → NaN, no qualifier
    assert r.loc["nwis_01021050", "qualifier"] == ""
    assert r.loc["nwis_01104430", "struct_canal_flag"] == 95
    assert pd.isna(r.loc["nwis_01021050", "struct_dam_flag"])
    assert (s["coord_flag"] == "").all() and s["latitude"].between(17, 72).all()
    assert r.loc["nwis_01104430", "hydgeom_c"] == pytest.approx(0.509052171)


def test_parse_powell_sites_rejects_inconsistent_ids():
    raw = M.read_tsv(FX / "powell_site_data_sample.tsv")
    raw.loc[0, "nwis_id"] = "09999999"
    with pytest.raises(ValueError):
        M.parse_powell_sites(raw)


def _models():
    return M.parse_powell_models(M.read_tsv(FX / "powell_diagnostics_sample.tsv"),
                                 M.read_tsv(FX / "powell_config_sample.tsv"))


def test_parse_powell_models():
    m = _models()
    assert set(m["resolution_min"].dropna().unique()) <= {5, 10, 12, 15, 30, 60}
    multi = m[m.site_id == "nwis_01400500"]
    assert sorted(multi["resolution_min"].tolist()) == [15, 30, 60]
    assert m.loc[m.site_id == "nwis_01548303", "model_confidence"].item() == "L"
    assert m["model_name"].eq("b_Kb_oipi_tr_plrckm.stan").all()
    assert (m["neg_gpp_pct"].between(0, 100)).all()
    dup = pd.concat([M.read_tsv(FX / "powell_diagnostics_sample.tsv")] * 2)
    with pytest.raises(ValueError):
        M.parse_powell_models(dup)


def _daily_raw() -> pd.DataFrame:
    name, b = M.zip_member(FX / "powell_daily_predictions_sample.zip", ".tsv")
    assert name == "daily_predictions.tsv"
    return M.read_tsv(b)


def test_parse_powell_daily_schema_units_and_confidence():
    d = M.parse_powell_daily(_daily_raw(), _models())
    assert d["date"].dtype == "datetime64[ns]" and (d["date"] == d["date"].dt.normalize()).all()
    assert d["date"].min() >= pd.Timestamp("2007-10-01") and d["date"].max() <= pd.Timestamp("2017-01-31")
    assert set(d["gpp_er_unit"].astype(str)) == {"g O2 m-2 d-1"}
    assert set(d["k600_unit"].astype(str)) == {"d-1"}
    assert not d.duplicated(["site_id", "date"]).any()
    assert d["nwis_id"].str.fullmatch(r"\d{8,15}").all()
    # every row's (site, resolution) has a published model rating
    assert d["model_confidence"].notna().all()
    assert set(d.loc[d.site_id == "nwis_01548303", "model_confidence"].astype(str)) == {"L"}
    # credible intervals bracket the median where all three exist
    ok = d[["gpp_lower", "gpp", "gpp_upper"]].notna().all(axis=1)
    assert (d.loc[ok, "gpp_lower"] <= d.loc[ok, "gpp"] + 1e-9).all()
    assert (d.loc[ok, "gpp"] <= d.loc[ok, "gpp_upper"] + 1e-9).all()
    # 'NA' (R missing, e.g. DO.tdist80 at nwis_01648010) → NaN without a qualifier
    assert d.loc[d.site_id == "nwis_01648010", "do_tdist80_m"].isna().any()
    assert (d["qualifier"] == "").all()
    assert sorted(d.loc[d.site_id == "nwis_01400500", "resolution_min"].unique()) == [15, 30, 60]
    # day-level plausibility flags follow the Appling 2018 thresholds
    assert (d["flag_gpp_negative"] == (d["gpp"] < -0.5)).all()
    assert (d["flag_er_positive"] == (d["er"] > 0.5)).all()


def test_parse_powell_daily_masks_sentinels_with_qualifier():
    raw = _daily_raw()
    raw.loc[0, "GPP"] = "-9999"        # synthetic injection
    raw.loc[1, "discharge"] = "Ice"    # synthetic injection
    d = M.parse_powell_daily(raw)
    first = d[(d.site_id == raw.loc[0, "site_name"]) & (d.date == pd.Timestamp(raw.loc[0, "date"]))]
    assert first["gpp"].isna().item() and "GPP=-9999" in first["qualifier"].item()
    second = d[(d.site_id == raw.loc[1, "site_name"]) & (d.date == pd.Timestamp(raw.loc[1, "date"]))]
    assert second["discharge_m3s"].isna().item() and "discharge=Ice" in second["qualifier"].item()


def test_parse_powell_daily_rejects_bad_dates_and_duplicates():
    raw = _daily_raw()
    bad = raw.copy()
    bad.loc[0, "date"] = "2555-07-01"
    with pytest.raises(ValueError):
        M.parse_powell_daily(bad)
    with pytest.raises(ValueError):
        M.parse_powell_daily(pd.concat([raw, raw.iloc[[0]]], ignore_index=True))


def test_powell_quality_counts_consistent():
    d = M.parse_powell_daily(_daily_raw(), _models())
    q = M.powell_quality_counts(d)
    assert q["site_days_model_conf_H_or_M"] == (q["site_days_model_confidence"]["H"]
                                                + q["site_days_model_confidence"]["M"])
    assert q["site_days_model_conf_H_and_plausible"] <= q["site_days_model_conf_H_or_M_and_plausible"]
    assert q["site_days_total_rows"] == len(d)


# ----------------------------------------------------------------------------- StreamPULSE

def _sp_sites():
    return M.parse_sp_sites(M.read_csv(FX / "streampulse_sites_sample.csv"))


def test_parse_sp_sites_drops_personal_columns():
    s = _sp_sites()
    assert not {"contact", "contactEmail"} & set(s.columns)
    assert not s.apply(lambda c: c.astype(str).str.contains("@").any()).any()
    r = s.set_index("site_id")
    assert r.loc["AZ_LV", "usgs_site_no"] == "09511300"      # leading zero kept
    assert pd.isna(r.loc["AL_MAYF-up", "usgs_site_no"])
    assert bool(r.loc["AZ_LV", "in_us"]) and not bool(r.loc["AT_YRN116", "in_us"])
    assert r["add_date"].notna().all()


def test_parse_sp_sites_deduplicates_reregistrations():
    raw = M.read_csv(FX / "streampulse_sites_sample.csv")
    again = raw[raw.siteID == "LV"].copy()
    again["addDate"] = "2016-12-14 21:25:09"   # re-registration seconds later (as seen on the portal)
    again["siteName"] = "Lower Verde River (dup)"
    s = M.parse_sp_sites(pd.concat([raw, again], ignore_index=True))
    r = s.set_index("site_id")
    assert len(s) == len(raw) and r.loc["AZ_LV", "n_registrations"] == 2
    assert r.loc["AZ_LV", "site_name"] == "Lower Verde River"          # earliest registration kept
    assert "Lower Verde River (dup)" in r.loc["AZ_LV", "all_site_names"]
    again["latitude"] = "35.0"
    with pytest.raises(ValueError):
        M.parse_sp_sites(pd.concat([raw, again], ignore_index=True))


def test_parse_sp_daily():
    raw = M.read_csv(FX / "streampulse_daily_sample.csv")
    d = M.parse_sp_daily(raw, _sp_sites())
    assert len(d) == len(raw) and not d.duplicated(["site_id", "date"]).any()
    assert d["valid_day"].dtype == "boolean"
    inval = d[~d["valid_day"]]
    assert inval["gpp"].isna().all() and inval["errors"].notna().all()
    val = d[d["valid_day"]]
    assert val["gpp"].notna().all() and val["er"].notna().all()
    assert set(d["gpp_er_unit"].astype(str)) == {"g O2 m-2 d-1"}
    # a site with daily results but no public site-table row is kept and flagged
    assert (~d["in_site_table"]).any()
    assert d.loc[d.site_id == "AZ_LV", "usgs_site_no"].eq("09511300").all()
    assert d["date"].dt.year.between(2006, 2026).all()
    assert set(d["site_match"].astype(str)) <= {"exact", "case_insensitive", "none"}
    q = M.sp_quality_counts(d)
    assert q["valid_and_plausible"] <= q["valid_day_with_gpp_and_er"] <= q["valid_day"] <= q["rows"]


def test_parse_sp_daily_case_insensitive_site_match_and_collisions():
    raw = M.read_csv(FX / "streampulse_daily_sample.csv")
    lv = raw[(raw.region == "AZ") & (raw.site == "LV")].copy()
    lower = lv.copy()
    lower["region"] = "az"                      # the portal publishes e.g. 'co_Cameo2' for 'CO_Cameo2'
    both = pd.concat([raw[~((raw.region == "AZ") & (raw.site == "LV"))], lower, lv], ignore_index=True)
    d = M.parse_sp_daily(both, _sp_sites())
    low = d[d.site_id == "az_LV"]
    assert (low["site_match"].astype(str) == "case_insensitive").all()
    assert low["site_table_id"].eq("AZ_LV").all() and low["usgs_site_no"].eq("09511300").all()
    # 'az_LV' and 'AZ_LV' on the same dates collide case-insensitively and are flagged, not dropped
    assert d.loc[d.site_id.str.upper() == "AZ_LV", "dup_site_date_ci"].all()
    assert not d.loc[d.site_id.str.upper() != "AZ_LV", "dup_site_date_ci"].any()


def test_parse_sp_models():
    m = M.parse_sp_models(M.read_csv(FX / "streampulse_models_sample.csv"))
    assert m["current_best"].dtype == "boolean"
    assert m["site_id"].str.contains("_").all()
    assert m["coverage"].notna().any() and (m["prop_neg_gpp"].dropna().between(0, 1)).all()


# ----------------------------------------------------------------------------- Bernhardt 2022 site table

def test_bernhardt_site_info_repairs_unquoted_commas():
    b = M.read_bernhardt_site_info((FX / "bernhardt_lotic_site_info_sample.csv").read_text())
    r = b.set_index("site_id")
    assert r.loc["KS_KANSASR", "site_name"] == "Kansas River at Desoto, KS"
    assert r.loc["KS_KANSASR", "comid"] == "3730399"
    assert r.loc["nwis_01104430", "site_name"] == "HOBBS BK BELOW CAMBRIDGE RES NR KENDALL GREEN, MA"
    assert b["latitude"].notna().all() and b["longitude"].lt(0).all()
    assert b["comid"].dtype == "string"


def test_bernhardt_site_info_rejects_unexpected_header():
    with pytest.raises(ValueError):
        M.read_bernhardt_site_info("a,b,c\n1,2,3\n")
