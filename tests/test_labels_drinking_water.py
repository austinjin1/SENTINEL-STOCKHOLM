# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Fixture-based tests for sentinel.data.sources.downstream_drinking_water (no network).

Fixtures in tests/fixtures/labels_drinking_water/ are real rows copied from the official files
retrieved 2026-10-22: ECHO ``SDWA_latest_downloads.zip`` (snapshot 2026Q3; contact columns of
SDWA_PUB_WATER_SYSTEMS replaced by 'REDACTED' so no personal data is stored), UCMR 4
``ucmr4-occurrence-data.zip``, and SAB v3 ``PWS_Boundaries_Latest.zip`` (7 CWS polygons,
Missing_CWS and tract-crosswalk rows).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sentinel.data.sources import downstream_drinking_water as D

FX = Path(__file__).parent / "fixtures" / "labels_drinking_water"
SNAP = "2026-10-22"


@pytest.fixture(scope="module")
def ref():
    return D.read_sdwa_csv(FX / "sdwa_ref_code_values_sample.csv")


@pytest.fixture(scope="module")
def scan():
    return D.scan_violations(FX / "sdwa_violations_enforcement_sample.csv", chunksize=50)


@pytest.fixture(scope="module")
def systems():
    return D.tidy_pws_systems(D.read_sdwa_csv(FX / "sdwa_pub_water_systems_sample.csv"))


@pytest.fixture(scope="module")
def viol(scan, ref, systems):
    return D.attach_system_attributes(D.tidy_violations(scan, ref, snapshot_date=SNAP), systems)


def _row(viol, pwsid, vid):
    r = viol[(viol["pwsid"] == pwsid) & (viol["violation_id"] == vid)]
    assert len(r) == 1
    return r.iloc[0]


# ----------------------------------------------------------------------------- SDWIS

def test_scan_keeps_only_rule_superset(scan):
    assert scan.attrs["n_scanned"] == 207
    assert len(scan) == 204                       # 3 Total Coliform Rule (110) rows dropped
    assert not scan["RULE_CODE"].isin(["110", "350"]).any()


def test_one_row_per_violation_and_enforcement_summary(scan, viol):
    assert not viol.duplicated(["pwsid", "violation_id"]).any()
    assert len(viol) == scan[["PWSID", "VIOLATION_ID"]].drop_duplicates().shape[0] == 18
    r = _row(viol, "MD0010030", "14")
    raw = scan[(scan["PWSID"] == "MD0010030") & (scan["VIOLATION_ID"] == "14")]
    assert r["n_enf_actions"] == raw["ENFORCEMENT_ID"].replace("", np.nan).dropna().nunique()
    assert r["n_enf_actions"] == r["n_enf_formal"] + r["n_enf_informal"] + r["n_enf_resolving"]
    assert bool(_row(viol, "PA6240005", "0920641")["has_boil_water_order"])  # SFH State Boil Water Order
    assert not bool(_row(viol, "LA1081012", "2248")["has_boil_water_order"])


@pytest.mark.parametrize("pwsid,vid,label,group", [
    ("CO0101030", "4000503", "nitrate_mcl", "nitrate"),        # 331/02/1038 nitrate-nitrite
    ("IA3971026", "253601", "nitrate_mcl", "nitrate"),         # 331/01/1040
    ("IA3971026", "51897", "", "nitrate"),                     # monitoring (MR), not a label
    ("LA1081012", "2248", "dbp_mcl", "dbp"),                   # 220/02/2950 TTHM LRAA
    ("LA1081012", "2361", "dbp_mcl", "dbp"),                   # 220/02/2456 HAA5 LRAA
    ("IL1730100", "7074907", "dbp_other_mcl", "dbp"),          # 210/02/1011 bromate
    ("OK1020512", "134527", "dbp_precursor_tt", "dbp_precursor"),  # 210/46/2920 TOC removal TT
    ("AR0000474", "200905", "turbidity_tt", "swtr"),           # 122/44 monthly turbidity exceed
    ("PA6240005", "0920641", "turbidity_tt", "swtr"),          # 122/43 single turbidity exceed
    ("MD0010030", "14", "swtr_tt_other", "swtr"),              # 121/41 SWTR TT
    ("OR4100845", "2750892", "swtr_tt_other", "swtr"),         # 121/42 failure to filter
    ("070000002", "1305", "", "swtr"),                         # 122/37 TT no prior approval: not a label
    ("NM3501024", "251", "", "swtr"),                          # 123/33 report failure
    ("PA4110023", "0525149", "", "other"),                     # MRDL (disinfectant residual)
])
def test_labels(viol, pwsid, vid, label, group):
    r = _row(viol, pwsid, vid)
    assert r["label"] == label and r["label_group"] == group


def test_ids_are_strings_with_leading_zeros(viol, systems):
    assert _row(viol, "070000002", "1305")["pwsid"] == "070000002"        # tribal, digits only
    assert _row(viol, "PA6240005", "0920641")["violation_id"] == "0920641"
    assert "WA53AA220" in set(viol["pwsid"])                             # alphanumeric WA PWSID
    assert str(systems["pwsid"].dtype) == "string"


def test_dates_validated_and_sentinels(viol):
    old = _row(viol, "WY5600005", "4000806")                             # compliance period in 1902
    assert pd.isna(old["compl_per_begin_date"]) and old["compl_per_begin_date_flag"] == "out_of_range"
    oe = _row(viol, "NM3501024", "251")                                   # NON_COMPL_PER_END '--->'
    assert pd.isna(oe["non_compl_per_end_date"]) and oe["non_compl_per_end_date_flag"] == "open_ended"
    assert pd.isna(oe["compl_per_end_date"]) and pd.isna(oe["compl_per_end_date_flag"])  # blank = missing
    ok = _row(viol, "LA1081012", "2248")
    assert ok["compl_per_begin_date"] == pd.Timestamp("2013-10-01") and ok["compl_per_days"] == 92


def test_units_kept_and_mgl_conversion(viol):
    tthm = _row(viol, "LA1081012", "2248")
    assert tthm["viol_measure"] == 92.0 and tthm["unit_of_measure"] == "UG/L"
    assert tthm["viol_measure_mgl"] == pytest.approx(0.092)
    haa5 = _row(viol, "LA1081012", "2361")
    assert haa5["unit_of_measure"] == "MG/L" and haa5["viol_measure_mgl"] == pytest.approx(0.258)
    no_unit = _row(viol, "IA3971026", "253601")                           # 12 with blank unit
    assert no_unit["viol_measure"] == 12.0 and pd.isna(no_unit["viol_measure_mgl"])
    conv = D.to_mgl(pd.Series(["1", "1000", "5", "7"]), pd.Series(["MG/L", "UG/L", "", "MREM"]))
    assert conv.iloc[0] == 1.0 and conv.iloc[1] == pytest.approx(1.0) and conv.iloc[2:].isna().all()
    assert pd.isna(tthm["viol_measure_qc"]) and pd.isna(haa5["viol_measure_qc"])   # plausible LRAAs


def test_measure_qc_flags():
    # synthetic values (unit test only): TTHM 92 entered as MG/L, nitrate 5 mg/L, HAA5 0.07 mg/L, MR row
    qc = D.measure_qc(pd.Series([92.0, 5.0, 0.07, 99.0]), pd.Series(["2950", "1040", "2456", "1040"]),
                      pd.Series(["dbp_mcl", "nitrate_mcl", "dbp_mcl", ""]))
    assert qc.iloc[0] == "gt_100x_federal_mcl" and qc.iloc[1] == "lt_federal_mcl"
    assert pd.isna(qc.iloc[2]) and pd.isna(qc.iloc[3])


def test_systems_drop_contact_fields(systems):
    lower = {c.lower() for c in D.PWS_CONTACT_COLUMNS}
    assert not lower & set(systems.columns)
    assert not systems.astype(str).apply(lambda s: s.str.contains("REDACTED")).any().any()
    by = systems.set_index("pwsid")
    assert by.loc["CO0101030", "is_surface_source"] and by.loc["NM3501024", "is_surface_source"]   # SWP, GU
    assert not by.loc["FL6290530", "is_surface_source"]                                             # GW
    assert by.loc["VA1185002", "pws_deactivation_date"] == pd.Timestamp("2002-09-01")
    assert str(systems["population_served"].dtype) == "Int64"


def test_pws_quarter_panel(viol, systems):
    p = D.pws_quarter_labels(viol, systems, start="2005Q1", end="2026Q2", snapshot_date=SNAP)
    ids = set(p["pwsid"])
    assert "FL6290530" not in ids and "WA53AA220" not in ids          # GW CWS / TNCWS excluded
    assert "VA1185002" not in ids                                      # deactivated 2002, before panel
    assert not p.duplicated(["pwsid", "quarter"]).any()
    md = p[p["pwsid"] == "MD0010030"]
    assert md["quarter"].min() == "2005Q1" and md["quarter"].max() == "2016Q2"   # deactivated 2016-06-02
    on = md[md["hb_swtr_tt_other"] == 1]["quarter"]
    assert on.min() == "2005Q1" and on.max() == "2015Q4" and len(on) == 44   # 2003-06-01 .. 2015-12-31
    assert md["hb_swtr_tt_other_onset"].sum() == 0                            # onset 2003Q2, outside panel
    g = p.set_index(["pwsid", "quarter"])
    assert g.loc[("AR0000474", "2005Q1"), "hb_turbidity_tt"] == 1
    assert g.loc[("AR0000474", "2005Q1"), "hb_turbidity_tt_onset"] == 1
    assert g.loc[("AR0000474", "2005Q1"), "hb_swtr_tt"] == 1
    assert g.loc[("PA6240005", "2009Q3"), "hb_turbidity_tt"] == 1
    assert g.loc[("LA1081012", "2013Q4"), "hb_dbp_mcl"] == 1 and g.loc[("LA1081012", "2018Q4"), "hb_dbp_mcl"] == 1
    assert g.loc[("LA1081012", "2014Q1"), "hb_dbp_mcl"] == 0
    assert g.loc[("OK1020512", "2007Q4"), "hb_dbp_precursor_tt"] == 1
    assert g.loc[("IL1730100", "2006Q4"), "hb_dbp_other_mcl"] == 1
    assert p["hb_nitrate_mcl"].sum() == 0               # all nitrate MCLs in the fixture predate 2005
    assert int(p["hb_any"].sum()) == int((p[[f"hb_{x}" for x in D.LABELS]].max(axis=1)).sum())
    lag = g.xs("LA1081012")["recent_reporting_lag"]
    assert bool(lag["2026Q2"]) and bool(lag["2025Q4"]) and not bool(lag["2025Q3"])
    assert str(p["hb_dbp_mcl"].dtype) == "int8"


def test_source_facilities_and_areas():
    fac = D.tidy_source_facilities(D.read_sdwa_csv(FX / "sdwa_facilities_sample.csv"))
    raw = D.read_sdwa_csv(FX / "sdwa_facilities_sample.csv")
    assert len(fac) == (raw["IS_SOURCE_IND"] == "Y").sum()
    assert fac["seller_pwsid"].dropna().str.match(r"^[A-Z0-9]{2}").all()
    geo = D.tidy_geographic_areas(D.read_sdwa_csv(FX / "sdwa_geographic_areas_sample.csv"))
    assert len(geo) > 0 and str(geo["pwsid"].dtype) == "string"


# ----------------------------------------------------------------------------- UCMR 4

def test_ucmr_source_water_censoring():
    sw = D.tidy_ucmr_results(D.read_ucmr_txt(FX / "ucmr4_haa_indicators_sample.txt"))
    assert set(sw["analyte"]) == {"bromide", "total organic carbon"}
    assert set(sw["sample_point_type"]) == {"SR"} and set(sw["unit"]) == {"ug/L"}
    cens = sw[sw["censored"]]
    assert len(cens) == 4 and cens["value"].isna().all()
    assert set(cens.loc[cens["analyte"] == "bromide", "detection_limit"]) == {20.0}
    assert set(cens.loc[cens["analyte"] == "total organic carbon", "detection_limit"]) == {1000.0}
    det = sw[~sw["censored"]]
    assert det["value"].notna().all() and det["detection_limit"].isna().all()
    r = sw[(sw["pwsid"] == "010106001") & (sw["collection_date"] == pd.Timestamp("2018-06-12"))
           & (sw["analyte"] == "total organic carbon") & ~sw["censored"]]
    assert r["value"].tolist() == [3110.0] and r["facility_id"].tolist() == ["00007"]
    assert sw["collection_date_flag"].isna().all()            # <NA> flag = date OK


def test_ucmr_all_split_haa_and_cyanotoxins():
    al = D.read_ucmr_txt(FX / "ucmr4_all_sample.txt")
    haa = D.tidy_ucmr_results(al[al["Contaminant"].isin(D.UCMR4_HAA_GROUPS)])
    cy = D.tidy_ucmr_results(al[al["Contaminant"].isin(D.UCMR4_CYANOTOXINS)])
    assert "manganese" not in set(haa["analyte"]) | set(cy["analyte"])
    assert not haa["censored"].any() and haa["mrl"].isna().all()          # HAA groups have no MRL
    zero = haa[haa["pwsid"] == "055294214"]
    assert zero["value"].tolist() == [0.0]                                 # reported zero kept as 0
    assert cy["censored"].sum() == 4
    mc = cy[(cy["analyte"] == "total microcystins") & ~cy["censored"]]
    assert mc["value"].tolist() == [0.363]
    assert set(cy.loc[cy["censored"] & (cy["analyte"] == "anatoxin-a"), "detection_limit"]) == {0.03}


def test_ucmr_ade_decoding():
    ade = D.tidy_ucmr_ade(D.read_ucmr_txt(FX / "ucmr4_cyanotoxin_ade_sample.txt"))
    m = dict(zip(ade["element"] + ":" + ade["response"], ade["response_desc"]))
    assert m["CyanotoxinCyanotoxinOccurrence:Yes"] == "cyanotoxins detected in source water"
    assert m["CyanotoxinCyanotoxinOccurrenceDetail:MIC"] == "microcystins"
    assert m["TreatmentInformation:SFN"] == "softening"
    haa = D.tidy_ucmr_ade(D.read_ucmr_txt(FX / "ucmr4_haa_ade_sample.txt"))
    assert haa["response_desc"].notna().all()


# ----------------------------------------------------------------------------- service areas

def test_cws_boundaries_parse():
    g = D.read_cws_boundaries(FX / "cws_boundaries_sample.gpkg")
    assert g.crs.to_epsg() == 4326 and len(g) == 7
    assert g["is_modeled"].tolist() == (g["boundary_source"] == "Modeled").tolist()
    assert int(g["is_modeled"].sum()) == 3
    assert g.loc[g["pwsid"] == "FL6580531", "pwsid_duplicated"].all()
    assert int(g["pwsid_duplicated"].sum()) == 2
    assert (g["area_km2"] > 0).all() and g["geometry_valid"].dtype == bool
    assert set(g["state"]) == {"NY", "VA", "NJ", "WA", "FL", "MT"}
    assert str(g["population_served"].dtype) == "Int64"
    b = g.total_bounds
    assert -180 <= b[0] < b[2] <= -60 and 20 <= b[1] < b[3] <= 72        # lon/lat degrees


def test_cws_area_of_invalid_geometry_not_negative():
    # synthetic bow-tie polygon (unit test only): self-intersecting, GEOS planar area 0 or negative
    import geopandas as gpd
    from shapely.geometry import Polygon

    bow = Polygon([(-94.80, 39.60), (-94.70, 39.70), (-94.70, 39.60), (-94.80, 39.70), (-94.80, 39.60)])
    gdf = gpd.GeoDataFrame({"PWSID": ["MO0000000"], "Symbology_Field": ["System Sourced"],
                            "Population_Served_Count": [10], "Date_Create": ["2024-06-06"]},
                           geometry=[bow], crs=4269)
    g = D.tidy_cws_boundaries(gdf)
    assert not bool(g["geometry_valid"].iloc[0])
    assert g["area_km2"].iloc[0] > 0                       # measured on a make_valid() copy
    assert not g.geometry.iloc[0].is_valid                 # stored geometry is not repaired


def test_missing_cws_and_crosswalk_encodings():
    m = D.tidy_missing_cws(pd.read_csv(FX / "missing_cws_sample.csv", dtype=str, keep_default_na=False,
                                       encoding="cp1252"))
    assert "ACUED. RURAL DOÑA ELENA" in set(m["pws_name"])
    assert m["population_served"].tolist()[0] == 175
    t = D.tidy_census_crosswalk(pd.read_csv(FX / "tracts_sample.csv", dtype=str, keep_default_na=False), "tract")
    assert t["geoid20"].tolist()[0] == "01001020100" and t["geoid20"].str.len().eq(11).all()
    assert t["area_weight"].between(0, 1).all()
