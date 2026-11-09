# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Parsing tests for the City of Chicago beach label client (no network).

Fixtures in tests/fixtures/labels_chicago_beaches/ are real rows cut from the recorded Socrata
downloads (2026-10-22), chosen to exercise each documented quirk. A few tests build tiny
synthetic inputs (DST edge cases, threshold boundary) — test inputs only, never data.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sentinel.data.sources import downstream_chicago_beaches as CB

FIX = Path(__file__).parent / "fixtures" / "labels_chicago_beaches"


def _csv(name: str) -> pd.DataFrame:
    return pd.read_csv(FIX / name, dtype=str, keep_default_na=False)


@pytest.fixture(scope="module")
def lab():
    return CB.parse_lab(_csv("lab_sample.csv"))


# ----------------------------------------------------------------------------- units & splitting

def test_culture_and_qpcr_are_separate_tables_with_their_own_units(lab):
    culture, dna = lab
    assert set(culture["unit"]) == {"CFU/100 mL"} and set(dna["unit"]) == {"CCE/100 mL"}
    assert set(culture["indicator"]) == {"Escherichia coli (culture)"}
    assert set(dna["indicator"]) == {"Enterococci (qPCR, DNA)"}
    assert "exceeds_235_cfu" in culture and "exceeds_1000_cce" not in culture
    assert "exceeds_1000_cce" in dna and "exceeds_235_cfu" not in dna
    # the parent row carrying both tests (Montrose 2015-07-08) lands once in each table
    assert (culture["culture_test_id"] == "93270").sum() == 1
    assert (dna["dna_test_id"] == "35").sum() == 1


def test_units_match_publisher_column_descriptions():
    meta = json.loads((FIX / "lab_metadata_slim.json").read_text())
    desc = {c["fieldName"]: c["description"] for c in meta["columns"]}
    assert "Colony Forming Units (CFU) per 100 ml" in desc["culture_sample_1_reading"]
    assert "calibrator cell equivalents (CCE) per 100 ml" in desc["dna_sample_1_reading"]


def test_ids_are_strings_and_duplicate_dna_rows_collapse(lab):
    culture, dna = lab
    assert str(dna["dna_test_id"].dtype) == "string"
    assert str(culture["culture_test_id"].dtype) == "string"
    assert not dna["dna_test_id"].str.contains(r"\.").any()
    # DNA test 541 is repeated on two parent rows (two culture tests that day): one DNA test
    assert (dna["dna_test_id"] == "541").sum() == 1
    assert culture["culture_test_id"].isin(["112689", "112669"]).sum() == 2
    assert len(dna) == 9 and len(culture) == 12


# ----------------------------------------------------------------------------- time handling

def test_two_digit_year_is_repaired_and_flagged(lab):
    culture, dna = lab
    r = dna.set_index("dna_test_id").loc["20148"]
    assert r["date_flag"] == "year_repaired" and str(r["date"]) == "2023-06-08"
    assert r["sample_timestamp_raw"].startswith("0023-")
    c = culture.set_index("culture_test_id").loc["114144"]
    assert c["date_flag"] == "year_repaired" and str(c["date"]) == "2016-09-26"


def test_impossible_timestamps_become_nat_with_flag():
    s = pd.Series(["0099-01-01T00:00:00.000", "1776-07-04T00:00:00.000", "", "not a date"])
    local, utc, date, flag = CB.local_times(s)
    assert list(flag) == ["out_of_range", "out_of_range", "missing", "unparseable"]
    assert local.isna().all() and utc.isna().all() and all(d is None for d in date)


def test_local_time_to_utc_and_unknown_times(lab):
    culture, dna = lab
    d = dna.set_index("dna_test_id")
    assert d.loc["13893", "time_utc"] == pd.Timestamp("2021-05-28 16:58", tz="UTC")  # 11:58 CDT
    assert d.loc["13893", "time_known"]
    # published at 00:00 → time unknown: no UTC instant, local date kept
    assert not d.loc["2406", "time_known"] and pd.isna(d.loc["2406", "time_utc"])
    assert str(d.loc["2406", "date"]) == "2017-05-30"
    c = culture.set_index("culture_test_id")
    assert not c.loc["111362", "time_known"] and pd.isna(c.loc["111362", "time_utc"])  # pre-2016
    assert c.loc["113666", "time_utc"] == pd.Timestamp("2016-08-04 13:30", tz="UTC")  # 08:30 CDT


def test_dst_gap_and_overlap_are_flagged_not_guessed():
    s = pd.Series(["2021-03-14T02:30:00.000", "2021-11-07T01:30:00.000", "2021-07-01T12:00:00.000"])
    _, utc, date, flag = CB.local_times(s)
    assert list(flag) == ["dst_nonexistent", "dst_ambiguous", ""]
    assert utc.isna().tolist() == [True, True, False]
    assert [str(x) for x in date] == ["2021-03-14", "2021-11-07", "2021-07-01"]


# ----------------------------------------------------------------------------- results & labels

def test_published_mean_artifacts_are_flagged_and_result_recomputed(lab):
    culture, dna = lab
    c = culture.set_index("culture_test_id")
    # single replicate 25 published as 5 = sqrt(25 * 1): the missing replicate was treated as 1
    assert c.loc["107011", "mean_check"] == "published_sqrt_of_single_replicate"
    assert c.loc["107011", "result"] == 25 and c.loc["107011", "reading_mean_published"] == 5
    # published mean 1 without any reading → no result
    assert c.loc["106048", "mean_check"] == "published_mean_without_replicates"
    assert np.isnan(c.loc["106048", "result"])
    # two replicates: recomputed geometric mean agrees with the published value
    assert c.loc["111362", "mean_check"] == "ok"
    assert c.loc["111362", "result"] == pytest.approx(np.sqrt(21.1 * 16.9))
    # (118, 1) published as 118: kept as published, flagged
    assert dna.set_index("dna_test_id").loc["593", "mean_check"] == "published_mean_mismatch"


def test_censoring_is_recorded_not_substituted(lab):
    culture, dna = lab
    c = culture.set_index("culture_test_id")
    assert c.loc["97906", "censored"] and c.loc["97906", "censor_side"] == "right"
    assert c.loc["97906", "detection_limit"] == CB.CULTURE_UPPER_LIMIT
    assert c.loc["97906", "result"] == pytest.approx(2419.6)  # value as published, not inflated
    assert c.loc["93270", "censor_side"] == "right"  # one replicate at 2420
    assert not c.loc["107342", "censored"]  # 1986.3 is below the cap
    # zero replicate: published mean missing; geometric mean is 0, flagged left-censored
    assert c.loc["103924", "mean_check"] == "published_mean_missing"
    assert c.loc["103924", "result"] == 0 and c.loc["103924", "censor_side"] == "left"
    assert dna.set_index("dna_test_id").loc["19837", "censor_side"] == "left"


def test_exceedance_flags(lab):
    culture, dna = lab
    c, d = culture.set_index("culture_test_id"), dna.set_index("dna_test_id")
    assert bool(c.loc["107342", "exceeds_235_cfu"]) and not bool(c.loc["111362", "exceeds_235_cfu"])
    assert pd.isna(c.loc["106048", "exceeds_235_cfu"])  # no result → NA, never False
    assert bool(d.loc["13893", "exceeds_1000_cce"]) and not bool(d.loc["2289", "exceeds_1000_cce"])
    assert str(culture["exceeds_235_cfu"].dtype) == "boolean"


def test_exceedance_threshold_is_strict():
    raw = _csv("lab_sample.csv")
    row = raw[raw["culture_test_id"] == "111362"]
    probe = pd.concat([row] * 2, ignore_index=True)
    probe.loc[0, ["culture_sample_1_reading", "culture_sample_2_reading", "culture_reading_mean"]] = "235"
    probe.loc[1, ["culture_sample_1_reading", "culture_sample_2_reading", "culture_reading_mean"]] = "235.1"
    probe["culture_test_id"] = ["a", "b"]
    culture, _ = CB.parse_lab(probe)
    assert culture.set_index("culture_test_id")["exceeds_235_cfu"].astype(bool).to_dict() == {"a": False,
                                                                                              "b": True}


# ----------------------------------------------------------------------------- beaches & coordinates

def test_beach_keys_unify_renamed_beaches():
    names = ["Hartigan", "Hartigan (Albion)", "Calumet Beach", "Calumet", "63rd Street Weather Station",
             "63rdStreet", "Ohio", "Old - Ohio Street Beach", "Montrose Dog", "Montrose", "NA", ""]
    k = CB.beach_key(pd.Series(names))
    assert list(k[:10]) == ["hartigan", "hartigan", "calumet", "calumet", "63rdstreet", "63rdstreet",
                            "ohiostreet", "ohiostreet", "montrosedog", "montrose"]
    assert k[10:].isna().all()


def test_missing_coordinates_filled_from_lookup_and_flagged(lab):
    culture, _ = lab
    lookup = pd.DataFrame({"beach_key": ["hartigan"], "latitude": [42.0027], "longitude": [-87.6564]})
    out = CB.fill_coords_from_lookup(culture, lookup).set_index("culture_test_id")
    assert out.loc["111362", "coord_flag"] == "from_beach_lookup"
    assert out.loc["111362", "latitude"] == pytest.approx(42.0027)
    assert out.loc["108593", "coord_flag"] == "missing" and np.isnan(out.loc["108593", "latitude"])  # 'NA'
    assert pd.isna(out.loc["108593", "beach_key"])


def test_sensor_locations_and_lookup():
    locs = CB.parse_sensor_locations(_csv("sensor_locations.csv"))
    assert len(locs) == 9 and (locs["coord_flag"] == "").all()
    assert set(locs.loc[locs["sensor_type"] == "Water", "beach_key"]) == {
        "calumet", "63rdstreet", "montrose", "osterman", "ohiostreet", "rainbow"}


# ----------------------------------------------------------------------------- sensors

def test_water_sensor_codes_ranges_and_placeholders_are_masked():
    w = CB.parse_water_sensors(_csv("water_sensors_sample.csv"))
    g = w.set_index(["measurement_id", "variable"])
    r = g.loc[("OhioStreetBeach202310220100", "water_temperature")]
    assert np.isnan(r["value"]) and r["qualifier"] == "missing_code:-100000"
    r = g.loc[("OhioStreetBeach202005151500", "wave_height")]
    assert np.isnan(r["value"]) and r["qualifier"] == "missing_code:-99999.992"
    r = g.loc[("OhioStreetBeach202409101000", "water_temperature")]
    assert np.isnan(r["value"]) and r["qualifier"] == "out_of_range:37.3"
    r = g.loc[("OhioStreetBeach202409101000", "turbidity")]
    assert r["value"] == 0 and r["qualifier"] == "zero_reading"
    z = w[w["measurement_id"] == "63rdStreetBeach202503171930"]
    assert len(z) and z["value"].isna().all() and (z["qualifier"] == "all_zero_record").all()
    ok = g.loc[("OstermanBeach201406121600", "turbidity")]
    assert ok["value"] == pytest.approx(7.55) and ok["unit"] == "NTU" and ok["qualifier"] == ""
    assert ok["time_utc"] == pd.Timestamp("2014-06-12 21:00", tz="UTC")
    # blank cells produce no row; the old Ohio Street sensor position maps to the same beach
    assert ("OhioStreetBeach202310220100", "transducer_depth") not in g.index
    assert set(w.loc[w["sensor_name"] == "Old - Ohio Street Beach", "beach_key"]) == {"ohiostreet"}


def test_weather_missing_codes():
    w = CB.parse_weather_sensors(_csv("weather_sensors_sample.csv")).set_index(["measurement_id", "variable"])
    assert w.loc[("63rdStreetWeatherStation201602241200", "wind_speed"), "qualifier"] == "missing_code:999.9"
    assert np.isnan(w.loc[("63rdStreetWeatherStation201802241200", "solar_radiation"), "value"])
    r = w.loc[("63rdStreetWeatherStation201504250900", "rain_intensity")]
    assert r["value"] == pytest.approx(7.2) and r["unit"] == "mm/h"
    assert (w.xs("63rdStreetWeatherStation202503171930", level=0)["qualifier"] == "all_zero_record").all()


def test_sensors_wide_round_trip():
    w = CB.parse_water_sensors(_csv("water_sensors_sample.csv"))
    wide = CB.sensors_wide(w)
    row = wide[wide["time_utc"] == pd.Timestamp("2014-06-12 21:00", tz="UTC")].iloc[0]
    assert row["turbidity_NTU"] == pytest.approx(7.55) and row["water_temperature_degC"] == pytest.approx(16.1)


# ----------------------------------------------------------------------------- predictions & matching

def test_predictions_and_advisories():
    p = CB.parse_predictions(_csv("predictions_sample.csv"))
    assert list(p["predicted_level"]) == [160.0, 54.7, 109.5, 53.2]
    assert set(p["prediction_source"]) == {"DNA Model"} and set(p["beach_key"]) == {"12thstreet"}
    a = CB.parse_advisories(_csv("advisories_sample.csv")).set_index("record_id")
    assert bool(a.loc["Montrose20160610", "swim_advisory"]) is True
    assert bool(a.loc["Calumet20160604", "swim_advisory"]) is False
    r = a.loc["63rdStreet20160618"]  # -1000 / -1: no prediction issued
    assert np.isnan(r["prob_ge_235_cfu"]) and np.isnan(r["predicted_level_cfu"])
    assert "probability=-1" in r["qualifier"] and "predicted_level=-1000" in r["qualifier"]


def test_prediction_qpcr_match_same_beach_same_day(lab):
    _, dna = lab
    p = CB.parse_predictions(_csv("predictions_sample.csv"))
    m = CB.match_predictions_to_qpcr(p, dna)
    assert [str(x) for x in m["date"]] == ["2017-05-26", "2017-05-29", "2017-05-30"]  # no qPCR 05-27
    assert m["prev_gap_days"].tolist()[1:] == [3.0, 1.0]
    assert m.loc[1, "qpcr_gmean_cce"] == pytest.approx(np.sqrt(52 * 78))
    assert len(CB.match_predictions_to_qpcr(p, dna, max_prev_gap_days=3)) == 2
