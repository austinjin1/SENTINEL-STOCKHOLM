# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Tests for the NEON client and builders (no network). Fixtures are truncated real NEON
files (RELEASE-2026, CC BY 4.0) with staff-identifier columns blanked; synthetic arrays are
used only inside tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sentinel.data import neon_aquatic as na
from sentinel.data.sources import neon

FIX = Path(__file__).parent / "fixtures" / "neon"
FAKE = "tok_TEST_not_a_real_token_123"


# --------------------------------------------------------------------------- credentials
def test_read_token_env_then_file(tmp_path):
    assert neon.read_token({"NEON_TOKEN": FAKE}, tmp_path / "missing") == FAKE
    kf = tmp_path / "keys.env"
    kf.write_text("API_USGS_PAT=x\nexport NEON_TOKEN=\"%s\"\n" % FAKE)
    assert neon.read_token({}, kf) == FAKE
    assert neon.read_token({}, tmp_path / "nope") is None


def test_token_only_in_header_never_in_url():
    url, params = neon.data_request("DP1.20093.001", "ARIK", "2018-12", package="expanded")
    assert FAKE not in url and FAKE not in str(params)
    assert "apiToken" not in params and "token" not in " ".join(params).lower()
    h = neon.request_headers(FAKE)
    assert h["X-API-Token"] == FAKE and "SENTINEL-v2" in h["User-Agent"]
    assert "X-API-Token" not in neon.request_headers(None)


def test_strip_url_and_redact():
    signed = ("https://storage.googleapis.com/neon-publication/a/b.csv?X-Goog-Algorithm=x"
              "&X-Goog-Signature=abc&apiToken=" + FAKE)
    assert neon.strip_url(signed) == "https://storage.googleapis.com/neon-publication/a/b.csv"
    msg = neon.redact(f"failed {signed} with {FAKE}", FAKE)
    assert FAKE not in msg and "X-Goog" not in msg


def test_data_request_release_param():
    _, p = neon.data_request("DP1.20288.001", "ARIK", "2023-07")
    assert p == {"package": "basic", "release": "RELEASE-2026"}
    _, p = neon.data_request("DP1.20288.001", "ARIK", "2026-07", release="PROVISIONAL")
    assert p == {"package": "basic"}


def test_data_request_validates_month():
    with pytest.raises(ValueError):
        neon.data_request("DP1.20288.001", "ARIK", "2023-7")


# --------------------------------------------------------------------------- file selection
NAMES = {
    "NEON.D10.ARIK.DP1.20288.001.102.100.100.waq_instantaneous.2023-07.basic.20251209T024159Z.csv":
        ("data", "waq_instantaneous", "102", "100"),
    "NEON.D10.ARIK.DP1.20288.001.sensor_positions.20251209T024159Z.csv": ("site_meta", "sensor_positions", None, None),
    "NEON.D10.ARIK.DP1.20288.001.variables.20251209T024159Z.csv": ("shared", "variables", None, None),
    "NEON.D10.ARIK.DP0.20288.001.categoricalCodes.20251209T024159Z.csv": ("shared", "categoricalCodes", None, None),
    "NEON.EcoCore_CSU.swc_externalLabSummaryData.20260106T143709Z.csv": ("lab", "swc_externalLabSummaryData", None, None),
    "NEON.D10.ARIK.DP1.20288.001.readme.20260123T000749Z.txt": ("doc", "readme", None, None),
    "NEON.D10.ARIK.DP1.20288.001.EML.20230701-20230731.20260123T000749Z.xml": ("doc", "EML", None, None),
    "RUSH_21A_29_2628_H25JWDRX5_16S_map.csv": ("external", None, None, None),
    "NEON.D10.ARIK.DP1.20053.001.101.100.030.TSW_30min.2023-07.basic.20251209T034712Z.csv":
        ("data", "TSW_30min", "101", "100"),
}


@pytest.mark.parametrize("name,expect", list(NAMES.items()))
def test_parse_file_name(name, expect):
    info = neon.parse_file_name(name)
    assert info["role"] == expect[0]
    assert info.get("table") == expect[1]
    if expect[2]:
        assert info["hor"] == expect[2] and info["ver"] == expect[3]


def _f(name):
    return {"name": name, "url": "https://x/" + name + "?sig=1", "size": 1, "md5": None}


def test_select_files_package_tmi_and_dedupe():
    files = [_f(n) for n in [
        "NEON.D10.ARIK.DP1.20053.001.101.100.030.TSW_30min.2023-07.basic.20251209T034712Z.csv",
        "NEON.D10.ARIK.DP1.20053.001.101.100.005.TSW_5min.2023-07.basic.20251209T034712Z.csv",
        "NEON.D10.ARIK.DP1.20053.001.101.100.030.TSW_30min.2023-07.basic.20251209T034712Z.csv",
        "NEON.D10.ARIK.DP1.20053.001.readme.20260123T000749Z.txt",
        "NEON.D10.ARIK.DP1.20053.001.variables.20251209T034712Z.csv",
    ]]
    sel = neon.select_files(files, neon.PRODUCTS["tsw"])
    names = [s["name"] for s in sel]
    assert len(names) == 2 and any("TSW_30min" in n for n in names)
    assert not any("TSW_5min" in n for n in names) and not any(n.endswith(".txt") for n in names)
    chem = [_f("NEON.D10.ARIK.DP1.20093.001.swc_fieldData.2018-12.basic.20260106T143709Z.csv"),
            _f("NEON.D10.ARIK.DP1.20093.001.swc_fieldData.2018-12.expanded.20260106T143709Z.csv")]
    sel = neon.select_files(chem, neon.PRODUCTS["chem"])  # expanded package
    assert [s["name"].split(".")[-3] for s in sel] == ["expanded"]


def test_site_months_prefers_pinned_release():
    payload = {"data": {"siteCodes": [
        {"siteCode": "ARIK", "availableReleases": [
            {"release": "PROVISIONAL", "availableMonths": ["2025-06", "2025-07"]},
            {"release": "RELEASE-2026", "availableMonths": ["2025-05", "2025-06"]}]},
        {"siteCode": "HARV", "availableReleases": [
            {"release": "RELEASE-2026", "availableMonths": ["2020-01"]}]}]}}
    sm = neon.site_months(payload, ["ARIK"])
    assert list(sm["month"]) == ["2025-05", "2025-06", "2025-07"]
    assert list(sm["release"]) == ["RELEASE-2026", "RELEASE-2026", "PROVISIONAL"]


# --------------------------------------------------------------------------- sensors
def test_sensor_csv_to_frame_types():
    df = neon.sensor_csv_to_frame(FIX / "waq_instantaneous_sample.csv")
    assert str(df["startDateTime"].dtype) == "datetime64[ns, UTC]"
    assert str(df["dissolvedOxygenFinalQF"].dtype) == "Int8"
    assert df["specificConductance"].dtype == np.float32
    assert len(df) == 5


def _synthetic_minutes():
    t = pd.date_range("2023-07-01 00:00", periods=180, freq="min", tz="UTC")
    v = np.arange(180, dtype=float)
    qf = np.zeros(180, dtype=int)
    qf[10:20] = 1        # failed QF -> masked
    v[30:40] = np.nan    # missing
    return pd.DataFrame({"startDateTime": t, "dissolvedOxygen": v, "dissolvedOxygenFinalQF": qf})


def test_mask_qf_and_hourly():
    df = _synthetic_minutes()
    var = [neon.SensorVar("do_mgl", "dissolvedOxygen", "dissolvedOxygenFinalQF", "mg/L")]
    m, counts = neon.mask_qf(df, var)
    assert counts["do_mgl"]["masked_qf"] == 10
    assert m["dissolvedOxygen"].isna().sum() == 20
    h = neon.aggregate_hourly(m, var)
    assert list(h["do_mgl_n"]) == [40, 60, 60]
    first = np.r_[np.arange(0, 10), np.arange(20, 30), np.arange(40, 60)]
    assert h["do_mgl_mean"].iloc[0] == pytest.approx(first.mean())
    assert h["do_mgl_min"].iloc[1] == 60 and h["do_mgl_max"].iloc[2] == 179


def test_sciRvw_redacted_masked():
    df = _synthetic_minutes()
    df["dissolvedOxygenFinalQFSciRvw"] = 0
    df.loc[100:104, "dissolvedOxygenFinalQFSciRvw"] = 2
    var = [neon.SensorVar("do_mgl", "dissolvedOxygen", "dissolvedOxygenFinalQF", "mg/L")]
    m, counts = neon.mask_qf(df, var)
    assert counts["do_mgl"]["masked_qf"] == 15


def test_hourly_to_daily_local_standard_time_and_min_hours():
    idx = pd.date_range("2023-07-01 07:00", periods=48, freq="h", tz="UTC")  # MST midnight
    h = pd.DataFrame({"x_mean": np.r_[np.ones(24), np.full(24, 3.0)],
                      "x_min": np.r_[np.zeros(24), np.full(24, 2.0)],
                      "x_max": np.r_[np.full(24, 2.0), np.full(24, 4.0)],
                      "x_n": np.full(48, 60)}, index=idx)
    h.iloc[30:40, :3] = np.nan          # day 2 keeps 14 hours -> below 18 -> NaN
    h.iloc[30:40, 3] = 0
    d = neon.hourly_to_daily(h, utc_offset_h=-7)
    assert list(d.index.strftime("%Y-%m-%d")) == ["2023-07-01", "2023-07-02"]
    assert d["x_mean"].iloc[0] == pytest.approx(1.0) and d["x_hours"].iloc[0] == 24
    assert np.isnan(d["x_mean"].iloc[1]) and d["x_hours"].iloc[1] == 14


def test_nitrate_unit_factor():
    assert na.UMOL_NO3_TO_MG_N == pytest.approx(0.0140067)
    assert 100 * na.UMOL_NO3_TO_MG_N == pytest.approx(1.40067)  # 100 µmol/L NO3 = 1.4 mg N/L


# --------------------------------------------------------------------------- chemistry
def test_tidy_chem_censoring_and_detection_limits():
    ext = pd.read_csv(FIX / "swc_externalLabDataByAnalyte_sample.csv", dtype=str,
                      keep_default_na=False, na_values=[""])
    lab = pd.read_csv(FIX / "swc_externalLabSummaryData_sample.csv", dtype=str,
                      keep_default_na=False, na_values=[""])
    out = neon.tidy_chem(ext, lab)
    assert len(out) == len(ext)
    tdn = out[out["analyte"] == "TDN"].iloc[0]
    assert tdn["censored"] and tdn["qualifier"] == "BDL"
    nd = out[out["qualifier"] == "ND"]
    assert len(nd) and nd["censored"].all() and nd["value"].isna().all()
    assert not out.loc[out["qualifier"].isna(), "censored"].any()
    # EcoCore TOC MDL 0.100 for 2016-06-15..2019-05-28 covers the 2018-12-04 sample
    toc = out[(out["analyte"] == "TOC") & (out["lab"] == "EcoCore_CSU")]
    assert len(toc) and np.allclose(toc["detection_limit"].to_numpy(), 0.1)
    assert out["unit"].notna().any() and out["sample_fraction"].notna().all()
    assert (out["below_mdl"] == (out["value"] < out["detection_limit"])).all()


def test_chem_var_names():
    assert na.chem_var("NO3+NO2 - N", "milligramsPerLiter") == "no3_no2_n_mgl"
    assert na.chem_var("TPC", "microgramsPerLiter") == "tpc_ugl"
    assert na.chem_var("UV Absorbance (254 nm)", None) == "uv_absorbance_254_nm"


def test_drop_pii():
    df = pd.DataFrame({"aCollectedBy": ["a@b.org"], "processedBy": ["x"], "siteID": ["ARIK"]})
    assert list(na.drop_pii(df).columns) == ["siteID"]


# --------------------------------------------------------------------------- microbes
def test_primer_sets():
    assert neon.primer_set("GTGYCAGCMGCCGCGGTAA", "CCGYCAATTYMTTTRAGTTT") == "16S_V4V5_515FY-926R"
    assert neon.primer_set("CCTACGGGNBGCASCAG", "GACTACNVGGGTATCTAATCC") == "16S_V3V4_Pro341F-Pro805R"
    assert neon.primer_set("GTGCCAGCMGCCGCGGTAA", "GGACTACNVGGGTWTCTAAT") == "16S_V4_515F-806R"
    assert neon.primer_set("", "") == "unknown"
    its = ("CTTGGTCATTTAGAGGAAGTAA", "GCTGCGTTCTTCATCGATGC")
    assert neon.primer_set(*its, "v4v5") == "16S_V4V5_515FY-926R"
    assert neon.primer_record_issue(*its, "v4v5") and neon.primer_record_issue(*its, "its") is None
    assert neon.primer_set(*its, "its").startswith("16S_other")
    assert neon.primer_set("ACGT", "TTTT", "v9") == "16S_other:v9"
    pcr = pd.read_csv(FIX / "mmg_swPcrAmplification_16S_sample.csv", dtype=str, keep_default_na=False)
    r = pcr.iloc[0]
    assert neon.primer_set(r.forwardPrimer, r.reversePrimer, r.targetSubfragment) == "16S_V4V5_515FY-926R"


def test_tidy_taxon_table():
    d = pd.read_csv(FIX / "taxon_map_16S_sample.csv", dtype=str, keep_default_na=False, na_values=[""])
    counts, otus = neon.tidy_taxon_table(d, "RUSH_test_16S_map")
    assert len(counts) == 3 and counts["count"].dtype == "Int64"
    assert set(counts["dnaSampleID"]) == {"ARIK.SS.20210602.DNA-DNA1"}
    assert otus["sequenceName"].is_unique and otus["sequence"].str.match(r"^[ACGTN]+$").all()
    assert "genusConfidence" in otus.columns and otus["genusConfidence"].dtype == np.float32


def test_fill_missing_units_same_lab_only():
    long = pd.DataFrame({"lab": ["A", "A", "A", "B", "A", "A"],
                         "analyte": ["SO4", "SO4", "Cl", "Cl", "UV Absorbance (254 nm)", "Mg"],
                         "unit": ["milligramsPerLiter", None, None, "milligramsPerLiter", None, None]})
    out = neon.fill_missing_units(long)
    assert out.loc[1, "unit"] == "milligramsPerLiter" and out.loc[1, "unit_imputed"]
    assert pd.isna(out.loc[2, "unit"]) and not out.loc[2, "unit_imputed"]   # other lab only
    assert out.loc[4, "unit"] == "absorbance units"
    assert pd.isna(out.loc[5, "unit"]) and not out.loc[0, "unit_imputed"]


def test_resolve_vars_alternatives():
    got = neon.resolve_vars(["startDateTime", "tsdWaterTempMean", "tsdWaterTempFinalQF"],
                            neon.TSD_VARS)
    assert got[0].value == "tsdWaterTempMean" and got[0].qf == "tsdWaterTempFinalQF"
    got = neon.resolve_vars(["startDateTime", "tsdWaterTempMean", "finalQF"], neon.TSD_VARS)
    assert got[0].qf == "finalQF"
    assert neon.resolve_vars(["startDateTime", "other"], neon.TSD_VARS) == []


def test_plausibility_screen_masks_and_clips():
    df = pd.DataFrame({"pH": [0.08, 7.2, 13.7], "turbidity": [-0.5, 2.0, 20000.0]})
    found = [neon.SensorVar("ph", "pH", "pHFinalQF", "pH"),
             neon.SensorVar("turb_fnu", "turbidity", "turbidityFinalQF", "FNU")]
    out, c = na.plausibility_screen(df, found)
    assert out["pH"].isna().tolist() == [True, False, True]
    assert out["turbidity"].tolist()[:2] == [0.0, 2.0] and np.isnan(out["turbidity"].iloc[2])
    assert c["ph"]["implausible"] == 2 and c["turb_fnu"]["clipped_to_zero"] == 1
