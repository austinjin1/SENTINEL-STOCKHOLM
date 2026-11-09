# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Tests for the WQP (WQX 3.0) client, unit harmonisation and QC. No network access.

Fixtures in tests/fixtures/wqp/ are row subsets of real WQX3 responses recorded 2026-10-22
(fullPhysChem Result CSV, a truncated response tail, Station and legacy summary CSVs). Edge cases
that do not occur in the recorded rows (sentinel codes, rejected status) are built inline.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sentinel.data import wqp_harmonize as H
from sentinel.data import wqp_qc as Q
from sentinel.data.sources import wqp as W

FX = Path(__file__).parent / "fixtures" / "wqp"


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return W.parse_results((FX / "result_fullphyschem_sample.csv").read_text(), group="test")


@pytest.fixture(scope="module")
def harmonized(raw):
    return H.harmonize(raw)


def _row(raw: pd.DataFrame, result_id: str) -> pd.Series:
    r = raw[raw["result_id"] == result_id]
    assert len(r) == 1, result_id
    return r.iloc[0]


# ------------------------------------------------------------------------------------------------
# request building / characteristic lists
# ------------------------------------------------------------------------------------------------
def test_states_cover_conus_ak_hi_pr():
    assert len(W.STATES) == 52
    assert {"AK", "HI", "PR", "DC"} <= set(W.STATES)
    assert W.statecode("MN") == "US:27" and W.statecode("AL") == "US:01"


def test_no_sensor_or_excluded_characteristic_requested():
    requested = set(W.all_requested_names())
    assert not requested & set(W.EXCLUDED_CHARACTERISTICS)
    assert not [n for n in requested if "probe" in n.lower() or "fluorescence" in n.lower()]
    # every requested name has an explicit harmonisation route (or is marked ambiguous)
    assert requested <= set(H.CHAR_ROUTES)


def test_result_query_body():
    b = W.result_query(["MN", "WI"], ["Chloride"], "2021-01-01", "2021-01-31")
    assert b["statecode"] == ["US:27", "US:55"]
    assert b["startDateLo"] == "01-01-2021" and b["startDateHi"] == "01-31-2021"
    assert b["dataProfile"] == "fullPhysChem" and b["sampleMedia"] == ["Water"]
    assert isinstance(b["characteristicName"], list)


# ------------------------------------------------------------------------------------------------
# HTTP layer (mocked)
# ------------------------------------------------------------------------------------------------
class _Resp:
    def __init__(self, status, content=b"", headers=None, text=""):
        self.status_code, self.content, self.headers, self.text = status, content, headers or {}, text

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _Sess:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), 0

    def post(self, *a, **k):
        self.calls += 1
        return self.responses.pop(0)


def _zip(text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("Result.csv", text)
    return buf.getvalue()


def test_post_csv_retries_then_succeeds(monkeypatch):
    csv = (FX / "result_fullphyschem_sample.csv").read_text()
    sess = _Sess([_Resp(503, headers={"Retry-After": "0"}), _Resp(200, _zip(csv))])
    monkeypatch.setattr(W, "session", lambda: sess)
    monkeypatch.setattr(W.time, "sleep", lambda s: None)
    text, info = W.post_csv(W.RESULT_URL, {"x": 1}, backoff=0.0)
    assert sess.calls == 2 and text == csv and info["bytes"] > 0


def test_post_csv_incomplete_is_raised_not_retried(monkeypatch):
    tail = (FX / "result_incomplete_tail.csv").read_text()
    sess = _Sess([_Resp(200, _zip(tail))])
    monkeypatch.setattr(W, "session", lambda: sess)
    with pytest.raises(W.IncompleteResponse):
        W.post_csv(W.RESULT_URL, {}, backoff=0.0)
    assert sess.calls == 1


def test_post_csv_codelist_outage_400_is_retried(monkeypatch):
    csv = (FX / "result_fullphyschem_sample.csv").read_text()
    bad = _Resp(400, text="Invalid request: The value of providers=STORET is not in the list of "
                          "enumerated values.")
    sess = _Sess([bad, _Resp(200, _zip(csv))])
    monkeypatch.setattr(W, "session", lambda: sess)
    monkeypatch.setattr(W.time, "sleep", lambda s: None)
    text, _ = W.post_csv(W.RESULT_URL, {}, backoff=0.0)
    assert sess.calls == 2 and text == csv


def test_post_csv_bad_request_not_retried(monkeypatch):
    sess = _Sess([_Resp(400, text="Request Entity Too Large")])
    monkeypatch.setattr(W, "session", lambda: sess)
    with pytest.raises(W.WQPError):
        W.post_csv(W.RESULT_URL, {}, backoff=0.0)
    assert sess.calls == 1


def test_check_complete_on_recorded_tail():
    with pytest.raises(W.IncompleteResponse):
        W.check_complete((FX / "result_incomplete_tail.csv").read_text())
    W.check_complete((FX / "result_fullphyschem_sample.csv").read_text())  # no raise


# ------------------------------------------------------------------------------------------------
# parsing
# ------------------------------------------------------------------------------------------------
def test_parse_values_sentinels_and_text():
    s = pd.Series(["1.5", "<0.01", ">2419.6", "-999999", "ND", "1,200", "", "Ice", "-9999",
                   "999999", "*Non-detect", "0"])
    p = W.parse_values(s)
    assert p["value"].iloc[0] == 1.5 and p["value"].iloc[5] == 1200.0 and p["value"].iloc[11] == 0
    assert p["value_op"].tolist()[:3] == ["", "<", ">"]
    assert p["value"].iloc[1] == 0.01 and p["value"].iloc[2] == 2419.6
    for i in (3, 8, 9):
        assert np.isnan(p["value"].iloc[i]) and p["value_flag"].iloc[i] == "sentinel_code"
    for i in (4, 7, 10):
        assert np.isnan(p["value"].iloc[i]) and p["value_flag"].iloc[i] == "non_numeric"
    assert p["value_flag"].iloc[6] == ""


def test_parse_results_shapes_and_ids(raw):
    assert len(raw) == 71
    assert raw["site_id"].map(type).eq(str).all()
    usgs = raw[raw["provider"] == "USGS"]
    assert usgs["site_id"].str.match(r"^USGS-\d{8,15}$").all()  # leading zeros kept as text
    assert raw["value"].dtype == "float64" and raw["censored"].dtype == bool


def test_time_utc_from_local_zone(raw):
    r = _row(raw, "STORET-1114609201")  # MPCA chl-a, CDT
    assert r["activity_tz"] == "CDT"
    local = pd.Timestamp(f"{r['activity_start_date']} {r['activity_start_time']}")
    assert r["time_utc"] == (local + pd.Timedelta(hours=5)).tz_localize("UTC")
    t = W.to_utc(pd.Series(["2022-06-01", "2022-06-01", "2022-06-01", "2022-01-15"]),
                 pd.Series(["12:00:00", "", "08:30", "23:59:59"]),
                 pd.Series(["EST", "EST", "XYZ", "AKST"]))
    assert t.iloc[0] == pd.Timestamp("2022-06-01 17:00", tz="UTC")
    assert pd.isna(t.iloc[1]) and pd.isna(t.iloc[2])  # no time / unknown zone
    assert t.iloc[3] == pd.Timestamp("2022-01-16 08:59:59", tz="UTC")


def test_censoring_from_condition_text_and_statistic(raw):
    tss = _row(raw, "STORET-1104580205")  # TSS "Not Detected", Censoring Level 4 mg/L
    assert tss["censored"] and tss["censor_type"] == "<" and np.isnan(tss["value"])
    assert tss["detection_limit"] == 4.0 and tss["detection_limit_unit"] == "mg/L"
    tube = _row(raw, "STORET-1114560640")  # tube 100 cm, StatisticalBase Minimum (tube full)
    assert tube["censor_type"] == ">" and tube["detection_limit"] == 100.0
    secchi = _row(raw, "STORET-850235060")  # value ">60" (cm)
    assert secchi["censor_type"] == ">" and secchi["detection_limit"] == 60 and secchi["value_text"] == ">60"
    nr = raw[raw["detection_condition"] == "Not Reported"].iloc[0]
    assert not nr["censored"] and np.isnan(nr["value"])


def test_classifiers():
    a = W.classify_activity(pd.Series(["Sample-Routine", "Field Msr/Obs",
                                       "Quality Control Sample-Field Blank",
                                       "Quality Control Sample-Field Replicate",
                                       "Quality Control Sample-Lab Matrix Spike",
                                       "Field Msr/Obs-Portable Data Logger",
                                       "Quality Control Field Replicate Portable Data Logger",
                                       "Quality Control Sample-Reference Sample"]))
    assert a.tolist() == ["", "", "blank", "replicate", "spike", "logger", "replicate", "reference"]
    s = W.classify_status(pd.Series(["Final", "Accepted", "Provisional", "Preliminary",
                                     "Rejected", "", "Unreviewed", "Historical"]))
    assert s.tolist() == ["accepted", "accepted", "preliminary", "preliminary", "rejected",
                          "unknown", "preliminary", "accepted"]
    d = W.classify_detection(pd.Series(["Not Detected", "Present Below Quantification Limit",
                                        "Present Above Quantification Limit", "Not Reported",
                                        "", "Detected Not Quantified", "Below Reporting Limit"]))
    assert d.tolist() == ["<", "<", ">", "none", "", "<", "<"]


def test_parse_stations_keeps_string_codes():
    st = W.parse_stations((FX / "station_sample.csv").read_text())
    assert len(st) == 15 and st["site_id"].is_unique
    assert st["huc8"].str.fullmatch(r"\d{8}").all()
    assert (st["huc8"] == "07070005").any()  # leading zero preserved
    assert st["lat"].between(42, 47.5).all() and st["lon"].between(-93, -86).all()
    for c in ("name", "site_type", "datum", "county", "state", "provider"):
        assert c in st.columns


def test_parse_summary():
    sm = W.parse_summary((FX / "summary_sample.csv").read_text())
    assert len(sm) == 40 and sm["n_results"].dtype == "int64" and sm["year"].notna().all()


# ------------------------------------------------------------------------------------------------
# units and speciation (explicit table checks)
# ------------------------------------------------------------------------------------------------
def test_species_factors():
    assert H.species_factor("NO3", "N") == pytest.approx(0.2259, abs=1e-4)
    assert H.species_factor("NO2", "N") == pytest.approx(0.3045, abs=1e-4)
    assert H.species_factor("NH4", "N") == pytest.approx(0.7765, abs=1e-4)
    assert H.species_factor("NH3", "N") == pytest.approx(0.8224, abs=1e-4)
    assert H.species_factor("PO4", "P") == pytest.approx(0.3261, abs=1e-4)
    assert H.species_factor("S", "SO4") == pytest.approx(2.9959, abs=1e-3)
    assert H.species_factor("N", "N") == 1.0


@pytest.mark.parametrize("unit,expect", [
    ("mg/L as N", ("mg/l", "N", False)), ("mg/l NO3", ("mg/l", "NO3", False)),
    ("mg N/L", ("mg/l", "N", False)), ("ug/L as P", ("ug/l", "P", False)),
    ("uS/cm @25C", ("us/cm", "", True)), ("MPN/100 mL", ("mpn/100ml", "", False)),
    ("NTU", ("ntu", "", False)), ("µg/L", ("ug/l", "", False)),
])
def test_split_unit(unit, expect):
    assert H.split_unit(H.norm_unit(unit)) == expect


def _one(characteristic, unit, value=1.0, fraction="", speciation="", method_id="", method_name="",
         temperature_basis="", dl=np.nan, dl_unit="", censored=False):
    return pd.DataFrame([{
        "characteristic": characteristic, "unit": unit, "value": value, "sample_fraction": fraction,
        "speciation": speciation, "method_id": method_id, "method_name": method_name,
        "temperature_basis": temperature_basis, "detection_limit": dl,
        "detection_limit_unit": dl_unit, "censored": censored,
        "censor_type": "<" if censored else "",
    }])


@pytest.mark.parametrize("char,unit,kw,var,value", [
    ("Chlorophyll a, corrected for pheophytin", "mg/m3", {}, "chla_corrected", 1.0),
    ("Chlorophyll a, corrected for pheophytin", "mg/L", {}, "chla_corrected", 1000.0),
    ("Chlorophyll a", "ppb", {}, "chla", 1.0),
    ("Microcystin", "ng/mL", {}, "microcystins", 1.0),
    ("Microcystin", "ng/L", {}, "microcystins", 0.001),
    ("Nitrate", "mg/L as NO3", {}, "no3", 0.225897),
    ("Nitrate", "mg/L", {"speciation": "as NO3"}, "no3", 0.225897),
    ("Nitrate", "mg/L", {"speciation": "as N"}, "no3", 1.0),
    ("Nitrate", "umol/L", {}, "no3", 0.0140067),
    ("Orthophosphate", "mg/L", {"speciation": "as PO4"}, "orthop", 0.326138),
    ("Ammonia", "mg/L", {"speciation": "as NH4"}, "nh3", 0.776488),
    ("Ammonia as NH3", "mg/L", {}, "nh3", 0.822448),
    ("Sulfate as S", "mg/L", {}, "sulfate", 2.995871),
    ("Phosphorus", "ug/L", {"fraction": "Total"}, "tp", 0.001),
    ("Phosphorus", "mg/L", {"fraction": "Filtered field and/or lab"}, "tdp", 1.0),
    ("Organic carbon", "mg/L", {"fraction": "Dissolved"}, "doc", 1.0),
    ("Organic carbon", "mg/L", {"fraction": "Unfiltered"}, "toc", 1.0),
    ("Depth, Secchi disk depth", "ft", {}, "secchi", 0.3048),
    ("Depth, Secchi disk depth", "cm", {}, "secchi", 0.01),
    ("Transparency, tube with disk", "in", {}, "transparency_tube", 2.54),
    ("Turbidity", "NTU", {}, "turbidity_ntu", 1.0),
    ("Turbidity", "FNU", {}, "turbidity_fnu", 1.0),
    ("Turbidity", "NTRU", {}, "turbidity_ntru", 1.0),
    ("Specific conductance", "mS/cm", {}, "spcond", 1000.0),
    ("Specific conductance", "umho/cm", {}, "spcond", 1.0),
    ("Conductivity", "uS/cm", {"temperature_basis": "25 Deg C"}, "spcond", 1.0),
    ("Escherichia coli", "cfu/100mL", {}, "ecoli", 1.0),
    ("Escherichia coli", "MPN/100 mL", {}, "ecoli", 1.0),
    ("Escherichia coli", "cfu/mL", {}, "ecoli", 100.0),
    ("Enterococcus", "cce/100mL", {"method_id": "1611"}, "enterococcus_qpcr", 1.0),
    ("Dissolved oxygen (DO)", "mg/L", {}, "do", 1.0),
    ("Dissolved oxygen (DO)", "%", {}, "do_pctsat", 1.0),
    ("pH", "std units", {}, "ph", 1.0),
    ("Absorbance at 440 nm", "units/cm", {}, "acdom440", 230.2585093),
    ("Nitrogen, mixed forms (NH3), (NH4), organic, (NO2) and (NO3)", "mg/L",
     {"fraction": "Unfiltered", "speciation": "as NO3"}, "tn", 0.225897),
    ("Nitrate", "ueq/L", {"speciation": "as NO3"}, "no3", 0.0140067),
    ("Chloride", "meq/L", {}, "chloride", 35.453),
    ("Sulfate", "meq/L", {}, "sulfate", 48.0313),
    ("Phosphate-phosphorus as P", "mg/l", {"fraction": "Total"}, "tp", 1.0),
])
def test_harmonize_conversions(char, unit, kw, var, value):
    m, u = H.harmonize(_one(char, unit, **kw))
    assert len(u) == 0, u["unmapped_reason"].tolist()
    assert m["variable"].iloc[0] == var
    assert m["value"].iloc[0] == pytest.approx(value, rel=1e-5)
    assert m["unit"].iloc[0] == H.VARIABLES[var].unit


def test_temperature_f_to_c():
    m, _ = H.harmonize(pd.concat([_one("Temperature, water", "deg F", 212.0),
                                  _one("Temperature, water", "deg F", 32.0),
                                  _one("Temperature, water", "deg C", 4.0)], ignore_index=True))
    assert m["value"].tolist() == pytest.approx([100.0, 0.0, 4.0])


@pytest.mark.parametrize("char,unit,kw,reason", [
    ("Turbidity", "JTU", {}, "turbidity_unit_not_ntu_fnu_ntru"),
    ("Chlorophyll a", "RFU", {}, "sensor_fluorescence_unit"),
    ("Colored dissolved organic matter (CDOM)", "RFU", {}, "sensor_fluorescence_unit"),
    ("Phycocyanin", "RFU", {}, "sensor_fluorescence_unit"),
    ("Escherichia coli", "cfu/100mL", {"method_name": "EPA 1609.1 qPCR"}, "qpcr_method_with_culture_unit"),
    ("Escherichia coli", "MPN", {}, "unit"),
    ("Escherichia coli", "", {}, "unit_missing"),
    ("Conductivity", "uS/cm", {}, "conductivity_not_temperature_compensated"),
    ("Organic carbon", "mg/L", {}, "fraction_unspecified"),
    ("Kjeldahl nitrogen", "mg/L", {"fraction": "Dissolved"}, "fraction_dissolved"),
    ("Nitrate", "mg/L as NO3", {"speciation": "as N"}, "speciation_conflict"),
    ("Total Kjeldahl nitrogen", "mg/L", {"speciation": "as NO3"}, "speciation_NO3_invalid"),
    ("Chlorophyll", "ug/L", {}, "ambiguous_characteristic"),
    ("Microcystin", "ug/g", {}, "unit"),
    ("Total suspended solids", "tons/day", {}, "unit"),
    ("Phosphate-phosphorus", "mg/L", {"fraction": "Dissolved"}, "fraction_dissolved"),
    ("Specific conductance", "mho/cm", {}, "unit"),
])
def test_harmonize_unmapped(char, unit, kw, reason):
    m, u = H.harmonize(_one(char, unit, **kw))
    assert len(m) == 0 and u["unmapped_reason"].iloc[0] == reason


def test_empty_rows_are_not_unit_problems():
    m, u = H.harmonize(_one("Chloride", "", value=np.nan))
    assert u["unmapped_reason"].tolist() == ["no_value_and_no_unit"]


def test_ecoli_qpcr_never_in_culture_variable():
    rows = pd.concat([_one("Escherichia coli", "cfu/100mL"),
                      _one("Escherichia coli", "CCE/100mL", method_id="EPA 1609.1"),
                      _one("Escherichia coli", "#/100mL", method_name="qPCR")], ignore_index=True)
    m, u = H.harmonize(rows)
    assert m["variable"].tolist() == ["ecoli", "ecoli_qpcr"]
    assert u["unmapped_reason"].tolist() == ["qpcr_method_with_culture_unit"]


def test_speciation_assumed_flag():
    m, _ = H.harmonize(pd.concat([_one("Nitrate", "mg/L"), _one("Nitrate as N", "mg/L"),
                                  _one("Phosphorus", "mg/L", fraction="Total")], ignore_index=True))
    assert m["speciation_assumed"].tolist() == [True, False, False]


def test_detection_limit_converted_with_own_unit():
    d = _one("Phosphorus", "mg/L", value=np.nan, fraction="Total", dl=5.0, dl_unit="ug/L",
             censored=True)
    m, _ = H.harmonize(d)
    assert m["detection_limit"].iloc[0] == pytest.approx(0.005)
    # blank result unit on a non-detect falls back to the limit's unit
    d = _one("Nitrate", "", value=np.nan, speciation="as NO3", dl=1.0, dl_unit="mg/L", censored=True)
    m, _ = H.harmonize(d)
    assert m["detection_limit"].iloc[0] == pytest.approx(0.225897, rel=1e-5)


def test_fixture_harmonized_units_are_canonical(harmonized):
    m, u = harmonized
    for var, g in m.groupby("variable"):
        assert (g["unit"] == H.VARIABLES[var].unit).all(), var
    assert not m["unit_original"].str.upper().str.contains("RFU").any()
    pair = m[(m["variable"] == "no3")].sort_values("species_used")
    assert pair["value"].iloc[0] == pytest.approx(pair["value"].iloc[1], rel=0.01)  # USGS as N vs as NO3


# ------------------------------------------------------------------------------------------------
# QC
# ------------------------------------------------------------------------------------------------
def _base_rows(n=1, **over) -> pd.DataFrame:
    row = {"variable": "tp", "characteristic": "Phosphorus", "site_id": "X-1",
           "activity_id": "A1", "activity_start_date": "2020-06-01", "activity_start_time": "10:00:00",
           "activity_depth": "", "activity_depth_unit": "", "result_depth": "", "result_depth_unit": "",
           "value": 0.1, "censored": False, "censor_type": "", "detection_limit": np.nan,
           "media": "Water", "qc_class": "", "statistical_base": "", "time_basis": "",
           "status_class": "accepted", "value_flag": "", "species_used": "P", "fraction_class": "total",
           "last_change_date": "2025-01-01"}
    row.update(over)
    return pd.DataFrame([row] * n)


def test_qc_steps_and_counts():
    rows = pd.concat([
        _base_rows(),                                                   # kept
        _base_rows(activity_id="A2", value=0.2, qc_class="blank", activity_start_time="11:00"),
        _base_rows(activity_id="A3", value=0.3, qc_class="logger", activity_start_time="12:00"),
        _base_rows(activity_id="A4", value=0.4, statistical_base="Daily Maximum",
                   activity_start_time="13:00"),                        # logger_or_aggregate
        _base_rows(activity_id="A5", value=0.5, status_class="rejected", activity_start_time="14:00"),
        _base_rows(activity_id="A6", value=np.nan, value_flag="sentinel_code",
                   activity_start_time="15:00"),
        _base_rows(activity_id="A7", value=np.nan, value_flag="non_numeric",
                   activity_start_time="16:00"),
        _base_rows(activity_id="A8", value=-0.2, activity_start_time="17:00"),   # out_of_range
        _base_rows(activity_id="A9", value=80.0, activity_start_time="18:00"),   # out_of_range
        _base_rows(activity_id="A10", value=0.1, last_change_date="2024-01-01"),  # duplicate
        _base_rows(activity_id="A11", media="Sediment", value=0.7),      # non_water_media
    ], ignore_index=True)
    rep = Q.QCReport()
    kept, rej = Q.qc_chunk(rows, rep)
    assert len(kept) == 1 and kept["activity_id"].iloc[0] == "A1"  # newer LastChangeDate kept
    t = rep.table().loc["tp"]
    assert t["input"] == 11 and t["output"] == 1
    assert t["non_water_media"] == 1 and t["qc_activity"] == 1 and t["logger_or_aggregate"] == 2
    assert t["rejected_status"] == 1 and t["sentinel_code"] == 1 and t["no_value"] == 1
    assert t["out_of_range"] == 2 and t["duplicate"] == 1
    assert set(rej["qc_reason"]) >= {"duplicate", "out_of_range", "sentinel_code"}


def test_exact_copies_are_deduplicated_not_dropped():
    rows = _base_rows(6)
    rows["activity_id"] = [f"C{i}" for i in range(6)]
    kept, _ = Q.qc_chunk(rows, Q.QCReport())
    assert len(kept) == 1


def test_qc_keeps_distinct_congeners_and_censored_rows():
    rows = pd.concat([
        _base_rows(variable="microcystin_other_congener", characteristic="Microcystin LF",
                   value=np.nan, censored=True, censor_type="<", detection_limit=0.25),
        _base_rows(variable="microcystin_other_congener", characteristic="Microcystin LW",
                   value=np.nan, censored=True, censor_type="<", detection_limit=0.25),
        _base_rows(variable="secchi", characteristic="Depth, Secchi disk depth", value=np.nan,
                   censored=True, censor_type=">", detection_limit=3.0, statistical_base="Minimum"),
    ], ignore_index=True)
    kept, _ = Q.qc_chunk(rows, Q.QCReport())
    assert len(kept) == 3


def test_speciation_duplicate_dropped():
    rows = pd.concat([_base_rows(variable="orthop", characteristic="Orthophosphate", value=0.032,
                                 species_used="P", fraction_class="dissolved"),
                      _base_rows(variable="orthop", characteristic="Orthophosphate", value=0.0323,
                                 species_used="PO4", fraction_class="dissolved")], ignore_index=True)
    rep = Q.QCReport()
    kept, _ = Q.qc_chunk(rows, rep)
    assert kept["species_used"].tolist() == ["P"]
    assert rep.table().loc["orthop", "speciation_duplicate"] == 1


def test_range_check_uses_limit_for_censored_rows():
    rows = pd.concat([_base_rows(value=np.nan, censored=True, censor_type="<", detection_limit=500.0),
                      _base_rows(activity_id="B", value=np.nan, censored=True, censor_type="<",
                                 detection_limit=0.0),
                      _base_rows(activity_id="C", variable="secchi", value=0.0)], ignore_index=True)
    assert Q.range_mask(rows).tolist() == [True, False, True]


def test_logger_like_series_flagged():
    many = _base_rows(6, variable="temp", characteristic="Temperature, water",
                      activity_start_time="05:00:00")
    many["value"] = [10.0, 11, 12, 13, 14, 15]
    many["activity_id"] = [f"L{i}" for i in range(6)]
    profile = _base_rows(3, variable="temp", characteristic="Temperature, water", site_id="LAKE")
    profile["activity_depth"] = ["0", "1", "2"]
    profile["activity_depth_unit"] = "m"
    rep = Q.QCReport()
    kept, rej = Q.qc_chunk(pd.concat([many, profile], ignore_index=True), rep)
    assert kept["site_id"].tolist() == ["LAKE"] * 3
    assert rep.table().loc["temp", "logger_or_aggregate"] == 6


def test_co_sampling_filter():
    rows = pd.concat([_base_rows(variable="do", characteristic="Dissolved oxygen (DO)", value=8.0),
                      _base_rows(variable="do", characteristic="Dissolved oxygen (DO)", value=8.0,
                                 site_id="X-2", activity_id="Z")], ignore_index=True)
    kept, _ = Q.qc_chunk(rows, Q.QCReport(), co_sampled={("X-1", "2020-06-01")})
    assert kept["site_id"].tolist() == ["X-1"]


def test_site_class():
    s = Q.site_class(pd.Series(["River/Stream", "Lake, Reservoir, Impoundment", "Well", "Estuary",
                                "BEACH Program Site-Great Lake", "Wetland Riverine-Emergent",
                                "Facility Public Water Supply (PWS)", ""]))
    assert s.tolist() == ["stream", "lake", "groundwater", "estuary_ocean", "lake", "wetland",
                          "facility", "unknown"]


def test_fixture_end_to_end(harmonized):
    m, _ = harmonized
    kept, rej = Q.qc_chunk(m, Q.QCReport())
    assert not kept["qc_class"].isin(["blank", "replicate", "logger"]).any()
    assert kept["statistical_base"].isin(["", "Minimum", "MPN", "Mean"]).all()
    for var, g in kept.groupby("variable"):
        v = g["value"].where(~g["censored"], g["detection_limit"])
        assert v.between(H.VARIABLES[var].lo, H.VARIABLES[var].hi).all(), var
    assert len(kept) + len(rej) == len(m)


# ------------------------------------------------------------------------------------------------
# downloader: truncated windows are split recursively (mocked server)
# ------------------------------------------------------------------------------------------------
def _load_downloader():
    import importlib.util
    p = Path(__file__).resolve().parents[1] / "scripts" / "data" / "download_wqp.py"
    spec = importlib.util.spec_from_file_location("download_wqp_under_test", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_fetch_window_splits_incomplete(monkeypatch):
    dl = _load_downloader()
    csv = (FX / "result_fullphyschem_sample.csv").read_text()
    calls = []

    def fake_post(url, body, **kw):
        a = pd.Timestamp(pd.to_datetime(body["startDateLo"], format="%m-%d-%Y"))
        b = pd.Timestamp(pd.to_datetime(body["startDateHi"], format="%m-%d-%Y"))
        calls.append((a, b))
        if (b - a).days > 7:
            raise W.IncompleteResponse("truncated", n_lines=10)
        return csv, {"bytes": 1}

    monkeypatch.setattr(dl.W, "post_csv", fake_post)
    frames, locs, info = dl.fetch_window("optical", ("Chlorophyll a",), ["MN"],
                                         pd.Timestamp("2021-01-01"), pd.Timestamp("2021-01-31"))
    leaves = [(a, b) for a, b in calls if (b - a).days <= 7]
    # leaves tile the window exactly, without gaps or overlaps
    days = sorted(d for a, b in leaves for d in pd.date_range(a, b))
    assert days == list(pd.date_range("2021-01-01", "2021-01-31"))
    assert info["splits"] >= 1 and len(frames) == len(leaves)


def test_windows_tile_period():
    dl = _load_downloader()
    for days in (31, 92, 184):
        w = dl._windows("2000-01-01", "2026-10-22", days)
        assert w[0][0] == pd.Timestamp("2000-01-01") and w[-1][1] == pd.Timestamp("2026-10-22")
        assert all(w[i + 1][0] - w[i][1] == pd.Timedelta(days=1) for i in range(len(w) - 1))
    s = dl.semimonthly_windows("2000-01-01", "2026-10-22")
    assert all(s[i + 1][0] - s[i][1] == pd.Timedelta(days=1) for i in range(len(s) - 1))
    assert len(s) == 644
