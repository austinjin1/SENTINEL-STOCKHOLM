# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Parsing tests for the Iowa DNR fish-kill client on recorded API/HTML samples (no network)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sentinel.data.sources import events_iowa_fishkill as I

FIX = Path(__file__).parent / "fixtures" / "labels_iowa_fishkill"


@pytest.fixture(scope="module")
def payload():
    return json.loads((FIX / "events_api_sample.json").read_text())


@pytest.fixture(scope="module")
def events(payload):
    return I.parse_events(payload)


def test_events_basic_schema_and_ids(payload, events):
    assert len(events) == len(payload["rows"])
    assert str(events["event_id"].dtype) == "string"
    assert set(events["event_id"]) >= {"218", "900", "1084"}
    assert (events["source"] == I.SOURCE).all() and (events["state"] == "IA").all()
    # the API's T12:00:00 is a placeholder → calendar date only
    assert (events["date"] == events["date"].dt.normalize()).all()
    assert events.loc[events.event_id == "1084", "date"].item() == pd.Timestamp("2026-10-07")
    assert events["date_flag"].eq("").all()


def test_missing_coordinates_masked_and_flagged(payload, events):
    raw_missing = {str(r["id"]) for r in payload["rows"] if r["lat"] is None}
    assert raw_missing  # fixture covers the case
    m = events[events.event_id.isin(raw_missing)]
    assert m["latitude"].isna().all() and (m["coord_flag"] == "missing").all()
    ok = events[~events.event_id.isin(raw_missing)]
    assert ok["latitude"].between(40.17, 43.71).all() and ok["longitude"].between(-96.85, -89.94).all()


def test_estimated_location_and_counts_not_zero_filled(payload, events):
    est = {str(r["id"]) for r in payload["rows"] if r["estLoc"]}
    assert set(events.loc[events["est_location"], "event_id"]) == est
    unk = {str(r["id"]) for r in payload["rows"] if r["magnitude"]["estFishKilled"] is None}
    assert events.loc[events.event_id.isin(unk), "est_fish_killed"].isna().all()  # NaN, never 0
    assert events.loc[events.event_id == "900", "est_fish_killed"].item() == 41665


def test_magnitude_bounds():
    lo, hi = I.magnitude_bounds(pd.Series(["1 - 100", "10,001 - 25,000", "1,000,000 or more",
                                           "Unknown/Unspecified", None]))
    assert lo.tolist()[:3] == [1.0, 10001.0, 1e6] and hi.tolist()[:2] == [100.0, 25000.0]
    assert np.isinf(hi[2]) and lo[3:].isna().all() and hi[3:].isna().all()


def test_derived_labels_follow_published_cause_fields(events):
    assert (events["is_natural"] == events["cause_origin"].eq("Natural")).all()
    assert (events["is_anthropogenic"] == events["cause_origin"].eq("Anthropogenic")).all()
    assert (events["is_env_wq"] == events["cause"].isin(["Environmental", "Cyanobacteria Bloom"])).all()
    assert events.loc[events.cause == "Cyanobacteria Bloom", "is_env_wq"].all()
    assert not (events["is_natural"] & events["is_anthropogenic"]).any()


def test_event_segments_links(payload):
    seg = I.parse_event_segments(payload)
    n_links = sum(len(r["impactedSegs"]) for r in payload["rows"])
    assert len(seg) == n_links
    assert seg.loc[seg.event_id == "1084"].iloc[0].to_dict() == {
        "event_id": "1084", "segment_id": "447", "is_origin_segment": True}


def test_segment_metadata_huc8_is_padded_string():
    s = I.parse_segments([json.loads((FIX / "segment_447.json").read_text())])
    r = s.iloc[0]
    assert r["segment_id"] == "447" and r["adb_code"] == "01-YEL-447" and r["huc8"] == "07060001"
    assert r["assigned_classes"] == "A1;BWW1"


def test_detail_page_fields_and_species_table():
    page = I.parse_event_page((FIX / "event_900.html").read_text(encoding="utf-8", errors="replace"))
    assert page["event_id"] == "900"
    assert page["kill_length_mi"] == pytest.approx(8.22)
    assert page["valuation_usd"] == pytest.approx(5199.97)
    assert page["investigation_expenses_usd"] == pytest.approx(810.76)
    assert page["fisheries_office"] == "Manchester Trout Hatchery"
    assert "Buck Creek" in page["comments_text"]
    sp = I.parse_species_tables(page["comments_html"])
    body = sp[~sp.is_total_row]
    assert len(body) == 5 and body["number_killed"].sum() == 41665
    assert sp.loc[sp.is_total_row, "number_killed"].item() == 41665
    assert sp.loc[sp.species_text == "Rock Bass", "value_usd"].item() == pytest.approx(690.0)


def test_detail_page_without_species_table_and_animal_waste_fields():
    page = I.parse_event_page((FIX / "event_982.html").read_text(encoding="utf-8", errors="replace"))
    assert page["event_id"] == "982" and page["kill_length_mi"] == pytest.approx(1.5)
    assert "Animal Waste Source" in page["page_fields"]
    assert I.parse_species_tables(page["comments_html"]).empty


def test_parse_event_pages_and_assemble(events):
    pages = {eid: (FIX / f"event_{eid}.html").read_text(encoding="utf-8", errors="replace")
             for eid in ("900", "982", "1084")}
    pg = I.parse_event_pages(pages)
    assert not pg["page_id_mismatch"].any()
    assert pg.loc[pg.event_id == "1084", "page_created"].item() == pd.Timestamp("2026-10-07 14:17:14")
    comids = pd.DataFrame({"event_id": pd.array(["1084"], dtype="string"),
                           "comid": pd.array([I.parse_nldi_position(json.loads(
                               (FIX / "nldi_position_1084.json").read_text()))], dtype="string"),
                           "comid_method": ["nldi"]})
    a = I.assemble_events(events, pg, comids)
    assert len(a) == len(events)
    r = a[a.event_id == "1084"].iloc[0]
    assert r["comid"] == "13211170" and r["kill_length_mi"] == pytest.approx(4.5)
    assert r["comid_context"] == "stream_reach"
    assert not a["page_date_mismatch"].any()
    assert a.loc[a.event_id == "218", "comid"].isna().all()


def test_nldi_position_empty_response():
    assert I.parse_nldi_position(None) is None
    assert I.parse_nldi_position({"features": []}) is None


def test_attains_au_id_mapping_and_merge():
    assert I.attains_au_id("01-YEL-447") == "IA 01-YEL-447"
    assert I.attains_au_id(None) is None and I.attains_au_id("  ") is None
    feats = json.loads((FIX / "attains_lines_sample.json").read_text())["features"]
    recs = [dict(f["attributes"], attains_layer="line") for f in feats]
    att = I.parse_attains(recs)
    assert set(att["attains_au_id"]) == {"IA 01-YEL-447", "IA 02-CED-3107"}
    segs = I.parse_segments([json.loads((FIX / "segment_447.json").read_text())])
    s = I.add_attains(segs, att).iloc[0]
    assert s["in_attains"] and s["attains_au_name"] == "Hecker Creek"
    assert s["attains_reporting_cycle"] == 2024 and s["attains_ir_category"] == "5"


def test_parse_attains_prefers_latest_cycle_and_line_layer():
    recs = [{"assessmentunitidentifier": "IA X", "reportingcycle": 2022, "attains_layer": "line"},
            {"assessmentunitidentifier": "IA X", "reportingcycle": 2024, "attains_layer": "point"},
            {"assessmentunitidentifier": "IA X", "reportingcycle": 2024, "attains_layer": "line"}]
    att = I.parse_attains(recs)
    assert len(att) == 1 and att.iloc[0]["attains_layer"] == "line"
    assert att.iloc[0]["attains_reporting_cycle"] == 2024


def _species(eid):
    page = I.parse_event_page((FIX / f"event_{eid}.html").read_text(encoding="utf-8", errors="replace"))
    return I.parse_species_tables(page["comments_html"])


def test_species_unlabelled_total_row_is_not_a_species():
    sp = _species("738")  # last row has an empty species cell and the total 1,256
    assert sp.loc[sp.is_total_row, "number_killed"].tolist() == [1256.0]
    assert sp.loc[~sp.is_total_row, "number_killed"].sum() == 1256.0


def test_species_number_column_beats_fish_killed_caption():
    sp = _species("1008")  # header: FISH KILLED | Species | Size of Fish | Number Killed
    assert sp["number_killed"].tolist() == [34560.0, 22.0, 398.0, 24.0]
    assert sp["species_text"].tolist()[0] == "Minnows Shiners Dace"


def test_species_colspan_aligned_size_class_table():
    sp = _species("978")  # species cells span 2–3 columns; "Total #" is column 4
    body = sp[~sp.is_total_row].set_index("species_text")["number_killed"]
    assert body["Minnows, shiners, chubs and dace"] == 4395 and body["Bluegill"] == 751
    assert body["Common Carp"] == 13


def test_species_check_against_api_count(events):
    assert _species("1084").empty  # no table on that page
    sp = _species("900").assign(event_id="900")
    chk = I.species_check(events, sp)
    assert chk.set_index("event_id").loc["900", "species_table_check"] == "match"
    sp_obj = sp.assign(is_total_row=sp["is_total_row"].astype(object))  # robust to object dtype
    assert I.species_check(events, sp_obj)["species_table_check"].tolist() == ["match"]
