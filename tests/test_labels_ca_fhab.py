# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Fixture-based tests for the California FHAB label client (no network).

Fixtures are real rows recorded from data.ca.gov on 2026-10-22 (staff e-mail / name columns
blanked). Small synthetic strings are used only to exercise the measurement parser.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sentinel.data.sources import events_ca_fhab as F

FIX = Path(__file__).parent / "fixtures" / "labels_ca_fhab"


@pytest.fixture(scope="module")
def raw():
    return {
        "bloom_reports": F.read_raw_csv(FIX / "bloom_reports_sample.csv"),
        "responses": F.read_raw_csv(FIX / "responses_sample.csv"),
        "cases": F.read_raw_csv(FIX / "cases_sample.csv"),
        "results": F.read_raw_csv(FIX / "results_sample.csv"),
    }


@pytest.fixture(scope="module")
def tables(raw):
    return F.build_tables(raw, max_date="2026-10-23")


def _row(df, col, value):
    sel = df[df[col] == value]
    assert len(sel) == 1, f"expected one row with {col}={value}, got {len(sel)}"
    return sel.iloc[0]


def test_resolve_resources_from_package_metadata():
    pkg = json.loads((FIX / "package_show_sample.json").read_text())
    res = F.resolve_resources(pkg)
    for short, rid in F.RESOURCE_IDS_2026_10_22.items():
        assert res[short]["id"] == rid
    with pytest.raises(ValueError):
        F.resolve_resources({"resources": [r for r in pkg["resources"] if "Results" not in r["name"]]})


def test_parse_datetime_formats_and_impossible_years():
    vals = pd.Series(["4/30/2016 12:00:00 AM", "2016-04-30T00:00:00.000", "7/9/2555 12:00:00 AM",
                      "10/25/1976 12:00:00 AM", "", "5/15/2026 12:22:58 PM"])
    d, f = F.parse_datetime(vals, max_date="2026-10-23")
    assert d[0] == pd.Timestamp("2016-04-30") and d[1] == pd.Timestamp("2016-04-30")
    assert pd.isna(d[2]) and f[2] == "out_of_range"
    assert pd.isna(d[3]) and f[3] == "out_of_range"
    assert pd.isna(d[4]) and f[4] == "missing"
    assert d[5] == pd.Timestamp("2026-05-15 12:22:58") and f[5] == ""


def test_bloom_report_dates_validated(tables):
    br = tables["bloom_reports"]
    for rid in ["4900", "4902", "5094"]:  # observation years 2555, 1976, 2626
        r = _row(br, "report_id", rid)
        assert pd.isna(r["date"]) and r["date_flag"] == "out_of_range"
    r = _row(br, "report_id", "1442")  # 2015 observation reported in 2016: plausible, kept
    assert r["date"] == pd.Timestamp("2015-09-15") and r["date_flag"] == ""
    r = _row(br, "report_id", "1595")  # observed (2018) after the report was created (2017)
    assert r["date"] == pd.Timestamp("2018-08-20") and r["date_flag"] == "after_created"


def test_bloom_report_coordinates(tables):
    br = tables["bloom_reports"]
    assert _row(br, "report_id", "2235")["coord_flag"] == "invalid"     # lon -1215574
    assert _row(br, "report_id", "2658")["coord_flag"] == "invalid"     # lat 4010.319
    assert _row(br, "report_id", "4872")["coord_flag"] == "zero"        # 0, 0
    for rid in ["2235", "2658", "4872"]:
        r = _row(br, "report_id", rid)
        assert np.isnan(r["latitude"]) and np.isnan(r["longitude"])
    r = _row(br, "report_id", "2623")  # Klamath River near the OR border: inside the buffered box
    assert r["coord_flag"] == "" and r["latitude"] == pytest.approx(42.121222)
    ok = br["latitude"].notna()
    w, s, e, n = F.CA_BBOX
    assert br.loc[ok, "longitude"].between(w, e).all() and br.loc[ok, "latitude"].between(s, n).all()


def test_one_row_per_report_with_linked_advisories(raw, tables):
    br = tables["bloom_reports"]
    assert br["report_id"].is_unique
    assert raw["bloom_reports"]["Bloom_Report_ID"].duplicated().any()  # published grain repeats reports
    r = _row(br, "report_id", "3665")
    assert r["n_linked_advisories"] == 2 and r["advisory_ids"] == "2115;2116"
    assert _row(br, "report_id", "3713")["n_linked_advisories"] == 3
    assert _row(br, "report_id", "4321")["n_linked_advisories"] == 1
    assert pd.isna(_row(br, "report_id", "4322")["advisory_id"])


def test_tier_ordering_and_label_definitions(tables):
    br = tables["bloom_reports"]
    assert br["advisory_tier"].cat.ordered
    assert list(br["advisory_tier"].cat.categories) == ["None", "Caution", "Warning", "Danger"]
    assert pd.Categorical(["Caution"], dtype=F.TIER_DTYPE) < pd.Categorical(["Danger"], dtype=F.TIER_DTYPE)
    lv = F.tier_level(F.advisory_class(pd.Series(["None", "caution", "WARNING", "Danger", "Algal mat alert sign",
                                                  "NA - refer to Report Details", "", "Brand new sign"])))
    assert lv.tolist()[:4] == [0, 1, 2, 3] and lv[4:].isna().all()
    cls = F.advisory_class(pd.Series(["", "Brand new sign"]))
    assert cls.tolist() == ["unknown", "other_brand_new_sign"]
    # label consistency
    tiered = br["tier_level"].notna()
    assert (br.loc[tiered, "label_tier_ge_caution"] == (br.loc[tiered, "tier_level"] >= 1)).all()
    assert br.loc[~tiered, "label_tier_ge_caution"].isna().all()
    mats = br[br["advisory_class"] == "algal_mat_alert"]
    assert len(mats) and mats["tier_level"].isna().all() and mats["any_advisory"].all()
    assert br.loc[br["advisory_class"] == "none", "any_advisory"].eq(False).all()
    assert br.loc[br["advisory_class"].isin(["unknown", "na_see_report"]), "any_advisory"].isna().all()
    danger = _row(br, "report_id", "1414")
    assert danger["advisory_tier"] == "Danger" and danger["tier_level"] == 3


def test_ids_are_strings_and_personal_columns_dropped(tables):
    for name, df in tables.items():
        cols = " ".join(df.columns).lower()
        assert "update_by" not in cols and "case_lead" not in cols, name
        for c in [c for c in df.columns if c.endswith("_id") or c in ("report_id", "case_id")]:
            assert str(df[c].dtype) == "string", (name, c)
            assert not df[c].dropna().str.contains(r"\.0$").any(), (name, c)
        txt = df.select_dtypes(include=["string", "object"]).astype(str)
        assert not txt.apply(lambda s: s.str.contains(r"@[A-Za-z0-9-]+\.[a-z]{2,}", regex=True)).any().any(), name


def test_advisory_history_table(tables):
    adv = tables["advisories"]
    assert adv["response_action_id"].is_unique
    assert set(adv["observation_channel"].dropna()) <= {"event_response", "monitoring", "other", "unknown"}
    loc = adv[adv["case_class"] == "Monitoring_Local Agency"]
    assert len(loc) and (loc["observation_channel"] == "monitoring").all()
    assert (loc["monitoring_type"] == "local_agency").all()
    below = adv[adv["advisory_detail"].fillna("").str.contains("Testing results below")]
    assert len(below) and below["below_triggers"].all()
    vis = adv[adv["advisory_detail"].fillna("").str.contains("visual monitoring based")]
    assert len(vis) and vis["basis_visual_only"].all()
    blank_detail = adv["advisory_detail"].isna()
    assert adv.loc[blank_detail, "basis_toxins"].isna().all()
    has_end = adv["advisory_end_date"].notna()
    assert has_end.any() and (adv.loc[has_end, "advisory_end_date"] >= adv.loc[has_end, "advisory_start_date"]).all()
    assert adv["has_end_date"].sum() == has_end.sum()
    # missing coordinates stay missing (flagged), malformed ones are NaN-ed
    assert adv.loc[adv["coord_flag"] != "", "latitude"].isna().all()


def test_cases_channel(tables):
    cs = tables["cases"]
    assert cs["case_id"].is_unique
    ev = cs[cs["case_class"] == "Event Response"]
    assert len(ev) and (ev["observation_channel"] == "event_response").all() and ev["monitoring_type"].isna().all()
    sat = cs[cs["case_class"] == "Monitoring_Satellite"]
    assert len(sat) and (sat["monitoring_type"] == "satellite").all()


def test_results_placeholders_values_and_censoring(tables):
    rs = tables["results"]
    f1 = _row(rs, "result_uid", "F1")  # field placeholder: value '0', no analyte / measurement type
    assert f1["is_placeholder"] and np.isnan(f1["value"]) and pd.isna(f1["censored"])
    assert f1["coord_source"] == "bloom_report" and f1["latitude"] == pytest.approx(41.199566)  # sample 0,0
    cond = _row(rs, "result_uid", "F1770")
    assert cond["value"] == 954 and cond["unit"] == "uS" and cond["measurement_type"] == "Conductivity"
    assert not cond["is_placeholder"] and cond["censored"] == False
    ph = _row(rs, "result_uid", "F1807")  # sample coords 0,0 → bloom-report coordinates used
    assert ph["coord_source"] == "bloom_report" and ph["value"] == pytest.approx(9.1) and pd.isna(ph["unit"])
    lab = _row(rs, "result_uid", "L1")  # lab toxin record without a published value
    assert lab["analyte_class"] == "Microcystins" and np.isnan(lab["value"]) and not lab["is_placeholder"]
    # measurement parser (synthetic strings)
    m = F.parse_measurement(pd.Series(["<0.15", "ND", "12.5", "-9999", "present", "", "<= 2"]))
    assert m.loc[0, "censored"] and m.loc[0, "detection_limit"] == pytest.approx(0.15) and np.isnan(m.loc[0, "value"])
    assert m.loc[1, "censored"] and np.isnan(m.loc[1, "detection_limit"])
    assert m.loc[2, "value"] == 12.5 and m.loc[2, "censored"] == False
    assert np.isnan(m.loc[3, "value"]) and m.loc[3, "qualifier"] == "-9999"
    assert m.loc[4, "value_category"] == "present" and np.isnan(m.loc[4, "value"])
    assert pd.isna(m.loc[5, "censored"])
    assert m.loc[6, "censored"] and m.loc[6, "detection_limit"] == 2


def test_veterinary_results_excluded(raw):
    rr = raw["results"].copy()
    rr.loc[rr.index[0], "Data_Type"] = "Veterinary"
    out = F.tidy_results(rr, max_date="2026-10-23")
    assert len(out) == len(rr) - 1


def test_datastore_fallback_matches_csv_semantics(tables):
    page = json.loads((FIX / "datastore_bloom_reports_page.json").read_text())["result"]
    df = F.datastore_records_to_frame(page["records"], page["fields"])
    assert "_id" not in df.columns and (df.dtypes == object).all()
    br = F.tidy_bloom_reports(df, max_date="2026-10-23")
    r = _row(br, "report_id", "1414")
    csv_r = _row(tables["bloom_reports"], "report_id", "1414")
    for c in ["date", "advisory_start_date", "case_id", "advisory_tier", "latitude", "longitude", "county"]:
        assert r[c] == csv_r[c], c


def test_end_date_stats(raw):
    st = F.end_date_stats(raw["bloom_reports"], raw["responses"])
    br = raw["bloom_reports"]
    assert st["bloom_reports"]["with_end"] == int((br["AdvisoryEndDate"] != "").sum())
    assert st["bloom_reports"]["with_start_and_end"] <= st["bloom_reports"]["with_end"]
