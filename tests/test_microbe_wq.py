# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Join tests for sentinel.data.microbe_wq (M3). Tiny SYNTHETIC frames shaped like the NEON processed
schemas (neon_microbe_16s samples/counts/otus, neon_chem chem_event_wide/units, neon_sonde_daily)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sentinel.data import microbe_wq as M

V4V5 = "TACG" * 93          # 372 bp
V3V4 = "TGGG" * 100         # 400 bp


def _community():
    samples = pd.DataFrame({
        "dnaSampleID": ["A.SS.20200101.DNA-DNA1", "A.SS.20200101.DNA-DNA2", "B.C0.20200105.DNA-DNA1", "A.SS.20200201.DNA-DNA1"],
        "site": ["AAAA", "AAAA", "BBBB", "AAAA"], "domain": ["D01", "D01", "D02", "D01"],
        "named_location": ["AAAA.AOS.S2", "AAAA.AOS.S2", "BBBB.AOS.buoy.c0", "AAAA.AOS.S2"],
        # 2020-01-02 03:00 UTC is still 2020-01-01 in local standard time (UTC-5)
        "time_utc": pd.to_datetime(["2020-01-02 03:00", "2020-01-02 03:10", "2020-01-05 15:00", "2020-02-01 15:00"], utc=True),
        "parentSampleID": ["A.SS.20200101", "A.SS.20200101", "B.C0.20200105", "A.SS.20200201"],
        "primer_set": ["unknown", "unknown", "16S_V3V4_Pro341F-Pro805R", "unknown"],
        "release": "RELEASE-2026", "provisional": [False, False, False, False]})
    otus = pd.DataFrame({"sequenceName": ["o1", "o2", "o3", "chl"], "sequence": [V4V5, V4V5[:-4] + "GGGG", V3V4, "A" * 430],
                         "genus": ["g1", "g2", "g3", None]})
    counts = pd.DataFrame({"dnaSampleID": ["A.SS.20200101.DNA-DNA1", "A.SS.20200101.DNA-DNA1", "A.SS.20200101.DNA-DNA2",
                                           "A.SS.20200101.DNA-DNA2", "B.C0.20200105.DNA-DNA1", "A.SS.20200201.DNA-DNA1"],
                           "sequenceName": ["o1", "chl", "o1", "o2", "o3", "o1"],
                           "count": [800, 400, 500, 300, 2000, 50]})
    sites = pd.DataFrame({"site": ["AAAA", "BBBB"], "aquatic_class": ["stream", "lake"], "site_lat": [40.0, 30.0],
                          "site_lon": [-100.0, -90.0], "station_lat": [40.1, np.nan], "station_lon": [-100.1, np.nan],
                          "elev_m": [500.0, 20.0]})
    return samples, counts, otus, sites


def test_events_sum_replicates_local_date_and_primer_rule():
    s, c, o, sites = _community()
    ev, tok, seqs, audit = M.build_events(s, c, o, sites, {"AAAA": -5.0, "BBBB": -6.0}, min_reads=1000)
    assert len(ev) == 2                       # the 50-read event is dropped
    a = ev[ev["site_id"] == "AAAA"].iloc[0]
    assert str(a["collect_date"].date()) == "2020-01-01"
    assert a["n_reads"] == 2000 and a["n_dna_samples"] == 2
    assert a["primer_set"] == "V4V5" and a["primer_source"] == "amplicon_length"   # median not fooled by 'chl'
    b = ev[ev["site_id"] == "BBBB"].iloc[0]
    assert b["primer_set"] == "V3V4" and b["primer_source"] == "pcr_record"
    assert np.isclose(b["latitude"], 30.0)    # station coordinates fall back to site coordinates
    t = tok[tok["sample_id"] == a["sample_id"]]
    assert t["count"].sum() == 2000 and t["seq_id"].nunique() == 3
    assert set(seqs["seq_id"]) == set(tok["seq_id"])
    assert M.primer_from_length(253) == "V4" and M.primer_from_length(np.nan) == "unknown"


def _chem():
    wide = pd.DataFrame({
        "parentSampleID": ["A.SS.20200101", "BBBB.other", "BBBB.c0"],
        "site": ["AAAA", "BBBB", "BBBB"], "named_location": ["AAAA.AOS.S2", "BBBB.AOS.inlet", "BBBB.AOS.buoy.c0"],
        "time_utc": pd.to_datetime(["2020-01-02 03:00", "2020-01-05 16:00", "2020-01-05 20:00"], utc=True),
        "provisional": False,
        "tn_mgl": [1.0, 9.0, 0.5], "tp_mgl": [np.nan, 9.0, 0.02], "doc_mgl": [3.0, 9.0, 4.0],
        "no3_no2_n_mgl": [0.2, 9.0, 0.01], "tp_mgl__censored": [True, False, False],
        "tn_mgl__censored": [False, False, False]})
    units = pd.DataFrame({"var": ["tn_mgl", "tp_mgl", "doc_mgl", "no3_no2_n_mgl", "tp_ugl"],
                          "analyte": ["TN", "TP", "DOC", "NO3+NO2 - N", "TP"],
                          "unit": ["milligramsPerLiter"] * 4 + ["microgramsPerLiter"]})
    long = pd.DataFrame({"parentSampleID": ["A.SS.20200101"], "var": ["tp_mgl"], "detection_limit": [0.003]})
    return wide, units, long


def test_chemistry_parent_match_censoring_and_station_fallback():
    s, c, o, sites = _community()
    ev, *_ = M.build_events(s, c, o, sites, {"AAAA": -5.0, "BBBB": -6.0}, min_reads=1000)
    wide, units, long = _chem()
    ch = M.match_chemistry(ev, wide, units, long).set_index("sample_id")
    a = ev.loc[ev["site_id"] == "AAAA", "sample_id"].iloc[0]
    b = ev.loc[ev["site_id"] == "BBBB", "sample_id"].iloc[0]
    assert ch.loc[a, "chem_match"] == "parentSampleID" and ch.loc[a, "raw_tn_mgl"] == 1.0
    assert ch.loc[a, "raw_tp_mgl"] == pytest.approx(0.003) and bool(ch.loc[a, "tp_mgl_censored"])
    # BBBB has no shared parent: the inlet sample (closer in time) must NOT be used; the buoy one is
    assert ch.loc[b, "chem_match"] == "time" and ch.loc[b, "raw_tn_mgl"] == 0.5
    assert ch.loc[b, "chem_dt_h"] == pytest.approx(5.0)


def test_chem_var_requires_mg_per_l():
    _, units, _ = _chem()
    assert M.chem_var_for(units, "TP") == "tp_mgl"
    with pytest.raises(ValueError):
        M.chem_var_for(units.iloc[:0], "TN")


def test_sonde_station_and_local_date(tmp_path):
    s, c, o, sites = _community()
    ev, *_ = M.build_events(s, c, o, sites, {"AAAA": -5.0, "BBBB": -6.0}, min_reads=1000)
    d = pd.DataFrame({"site": "AAAA", "date": pd.to_datetime(["2019-12-31", "2020-01-01", "2020-01-02", "2020-01-01"]),
                      "hor": ["102", "102", "102", "101"], "ver": "100",
                      "spcond_uscm_mean": [100.0, 200.0, 300.0, 999.0], "do_mgl_mean": [8.0, np.nan, 10.0, 1.0],
                      "turb_fnu_mean": [1.0, 2.0, 3.0, 9.0], "chl_ugl_mean": [0.5, 0.6, 0.7, 9.0], "provisional": False})
    d.to_parquet(tmp_path / "AAAA.parquet")
    so = M.match_sonde(ev, tmp_path).set_index("sample_id")
    a = ev.loc[ev["site_id"] == "AAAA", "sample_id"].iloc[0]
    assert so.loc[a, "raw_sc_uscm"] == 200.0          # S2 (102), local collection date
    assert np.isnan(so.loc[a, "raw_do_mgl"])          # missing day stays missing
    assert so.loc[a, "raw_do_mgl_pm1d"] == pytest.approx(9.0)
    b = ev.loc[ev["site_id"] == "BBBB", "sample_id"].iloc[0]
    assert np.isnan(so.loc[b, "raw_sc_uscm"])          # no sonde file for BBBB


def test_visit_key_collapses_bottle_and_replicate_suffixes():
    ids = pd.Series(["ARIK.SS.20200916.FIL.NUT", "ARIK.SS.20200916.RAW.NUT.2", "ARIK.SS.20200916", "BLWA.C0.20210921.RAW.3"])
    assert M.visit_key(ids).tolist() == ["ARIK.SS.20200916"] * 3 + ["BLWA.C0.20210921"]
    w = pd.DataFrame({"parentSampleID": ["A.SS.20200101", "A.SS.20200101.RAW.NUT", "A.SS.20200101.RAW.NUT.2"],
                      "site": "AAAA", "named_location": "AAAA.AOS.S2",
                      "time_utc": pd.to_datetime(["2020-01-02 03:00"] * 3, utc=True),
                      "doc_mgl": [3.0, np.nan, np.nan], "tp_mgl": [np.nan, 0.02, 0.04],
                      "tp_mgl__censored": [None, False, True]})
    c = M.collapse_visits(w, ["doc_mgl", "tp_mgl"]).set_index("parentSampleID")
    assert c.loc["A.SS.20200101", "doc_mgl"] == 3.0
    assert c.loc["A.SS.20200101", "tp_mgl"] == pytest.approx(0.03)
    assert bool(c.loc["A.SS.20200101", "tp_mgl__censored"]) and c.loc["A.SS.20200101", "n_bottles"] == 3
