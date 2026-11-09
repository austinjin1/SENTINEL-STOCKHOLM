# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""City of Chicago / Chicago Park District beach data (Socrata, data.cityofchicago.org, no key).

Label sources for SafeSwim (RESEARCH_PLAN §8 D1, docs/research/07 §d.1):

=====================  ===========  ==========================================================
key                    dataset id   content
=====================  ===========  ==========================================================
lab                    2ivx-z93u    Beach Lab Data: culture *E. coli* (CFU/100 mL, 2006–2016) and
                                    DNA/qPCR *Enterococci* (CCE/100 mL, 2015–). Parent table of
                                    the filtered views hmqm-anjq (DNA), hh4t-tnq8 (culture) and
                                    9wd3-tqr2 (both); 17,594 + 18,002 − 629 = 34,967 parent rows.
predictions            xvsz-3xcj    Official "Beach E. coli Predictions" (2017–; "DNA Model") —
                                    the incumbent to re-score.
advisories             t62e-8nvc    Beach Swim Advisories (2016 only; culture-era model:
                                    predicted level, P(E. coli ≥ 235 CFU), advisory Y/N).
water_sensors          qmqz-2xku    Beach Water Quality – Automated Sensors (hourly, in season).
weather_sensors        k7hf-8y75    Beach Weather Stations – Automated Sensors (context only).
sensor_locations       g3ip-u8rb    Lat/lon of the water and weather sensors.
=====================  ===========  ==========================================================

Rules applied here (CONTRIBUTING_V2):

* Culture (CFU/100 mL, *E. coli*) and qPCR (CCE/100 mL, *Enterococci*) are different indicators
  and units → two separate tables; never pooled.
* Sample timestamps are local wall clock (America/Chicago) → ``time_utc`` plus local ``date``.
  Times are not recorded for culture samples before 2016 and for some DNA samples (published as
  00:00) → ``time_known=False`` and ``time_utc`` NaT (the date is kept). Two-digit-year errors
  ('0023-06-08', '0016-09-26'; neighbouring test IDs confirm 2023/2016) are repaired to 20YY and
  flagged ``year_repaired``; anything else impossible → NaT + flag.
* The test-level ``result`` is the geometric mean of the replicate readings that are present,
  recomputed here. The published ``reading_mean_published`` is kept, and ``mean_check`` records
  disagreements: for single-replicate culture tests the published "geometric mean" is √x (the
  missing replicate was treated as 1), and 61 culture rows carry a published mean of 1 with no
  readings at all.
* Censoring: culture readings cluster at 2419.6–2420 (the IDEXX Quanti-Tray/2000 MPN upper
  limit — an inference from the values, the City documents CFU), so they are right-censored;
  zero readings are below an unpublished reporting limit (left-censored). Nothing is substituted.
* Sensor missing codes (−100000, −99999.992, 999.9 wind), physically impossible values and
  all-zero placeholder records → NaN; the original text is kept in ``qualifier``.
* Decision time: culture results arrive 18–24 h after sampling (never same-day inputs); qPCR
  results ~4–5 h. The official "DNA Model" is a random forest on the *same-day* qPCR geometric
  means at 5 input beaches (Chicago/clear-water-app ``app.R``) and is issued only after those
  results are posted.
"""

from __future__ import annotations

import datetime as _dt
import re

import numpy as np
import pandas as pd

from . import events_common as C

DOMAIN = "data.cityofchicago.org"
TZ = "America/Chicago"
DATASETS = {
    "lab": "2ivx-z93u",
    "predictions": "xvsz-3xcj",
    "advisories": "t62e-8nvc",
    "water_sensors": "qmqz-2xku",
    "weather_sensors": "k7hf-8y75",
    "sensor_locations": "g3ip-u8rb",
}
# Filtered views of the lab parent table (not downloaded; row counts checked for completeness).
LAB_VIEWS = {"dna_view": "hmqm-anjq", "culture_view": "hh4t-tnq8", "both_view": "9wd3-tqr2"}

CULTURE_BAV_CFU = 235.0     # EPA 2012 RWQC E. coli BAV; CPD swim-advisory trigger (culture era)
QPCR_ADVISORY_CCE = 1000.0  # CPD qPCR Enterococci advisory level (dataset description)
CULTURE_UNIT = "CFU/100 mL"
QPCR_UNIT = "CCE/100 mL"
CULTURE_UPPER_LIMIT = 2419.6          # Quanti-Tray/2000 upper limit (inferred, see module doc)
CULTURE_CAP_WINDOW = (2419.0, 2421.0)  # published values at the cap: 2419.1 … 2420.6

# Same-day qPCR input beaches of the official random-forest "DNA Model" (clear-water-app app.R).
MODEL_INPUT_BEACHES = ("calumet", "63rdstreet", "rainbow", "montrose", "southshore")
MODEL_APP_URL = "https://github.com/Chicago/clear-water-app/blob/master/app.R"

# Lake Michigan shoreline of Chicago (W, S, E, N), generous margin.
CHICAGO_BBOX = (-87.80, 41.60, -87.45, 42.10)

TERMS_URL = "https://www.chicago.gov/city/en/narr/foia/data_disclaimer.html"


def resource_url(dataset_id: str, fmt: str = "csv") -> str:
    return f"https://{DOMAIN}/resource/{dataset_id}.{fmt}"


def page_url(dataset_id: str) -> str:
    return f"https://{DOMAIN}/d/{dataset_id}"


def fetch_dataset(key: str, **kw) -> pd.DataFrame:
    """All rows of one dataset (all-string DataFrame, stable :id order)."""
    return C.socrata_csv(DOMAIN, DATASETS.get(key, key), **kw)


def fetch_metadata(key: str) -> dict:
    return C.socrata_metadata(DOMAIN, DATASETS.get(key, key))


# ----------------------------------------------------------------------------- keys & times

# Name variants → one key. Keys are lower-case alphanumerics with parenthetical aliases and the
# words "beach"/"weather station" removed: 'Hartigan' (2006–16) and 'Hartigan (Albion)' (2017–)
# are the same beach, as are 'Margaret T Burroughs'/'… (31st)' and 'Marion Mahony Griffin'/
# '… (Jarvis)'. 'Montrose Dog' is a separate beach. The 2016 advisories write 'Ohio', and the
# sensor table has an 'Old - Ohio Street Beach' sensor position.
BEACH_ALIASES = {"ohio": "ohiostreet", "oldohiostreet": "ohiostreet"}


def beach_key(names: pd.Series | list) -> pd.Series:
    s = pd.Series(names, dtype="object").fillna("").astype(str).str.lower()
    s = s.str.replace(r"\([^)]*\)", "", regex=True)
    s = s.str.replace(r"weather\s*station", "", regex=True)
    s = s.str.replace(r"[^a-z0-9]", "", regex=True)
    s = s.str.replace(r"beach$", "", regex=True)
    s = s.replace(BEACH_ALIASES)
    return s.where(~s.isin(["", "na", "nan", "none"])).astype("string")


def repair_two_digit_year(values: pd.Series) -> tuple[pd.Series, pd.Series]:
    """'0023-06-08T…' → '2023-06-08T…' when 2000 ≤ 20YY ≤ current year; returns (text, repaired)."""
    s = pd.Series(values, dtype="object").fillna("").astype(str).str.strip()
    m = s.str.match(r"^00\d\d-")
    yy = pd.to_numeric(s.str[2:4].where(m), errors="coerce")
    ok = m & (yy + 2000 <= _dt.date.today().year)
    return s.where(~ok, "20" + s.str[2:]), ok


def local_times(values: pd.Series, *, min_date: str = "2000-01-01", repair_year: bool = True
                ) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Socrata floating timestamps (local wall clock) → (time_local, time_utc, date, flag).

    flag: '' ok, 'year_repaired', 'missing', 'unparseable', 'out_of_range' (calendar-impossible),
    'dst_nonexistent' / 'dst_ambiguous' (time_utc NaT, local date kept).
    """
    raw = pd.Series(values, dtype="object")
    text, repaired = (repair_two_digit_year(raw) if repair_year
                      else (raw, pd.Series(False, index=raw.index)))
    local, flag = C.clean_dates(text, min_date=min_date)
    local = local.astype("datetime64[ns]")
    flag = flag.where(~(repaired & (flag == "")), "year_repaired")
    utc = local.dt.tz_localize(TZ, ambiguous="NaT", nonexistent="NaT")
    dst_bad = local.notna() & utc.isna()
    if dst_bad.any():
        nonexist = local[dst_bad].dt.tz_localize(TZ, ambiguous=True, nonexistent="NaT").isna()
        flag = flag.copy()
        flag[dst_bad] = np.where(nonexist, "dst_nonexistent", "dst_ambiguous")
    date = local.dt.date.astype("object").where(local.notna(), None)
    return local, utc.dt.tz_convert("UTC"), date, flag


def _coords(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    lat = df["latitude"] if "latitude" in df else pd.Series(np.nan, index=df.index)
    lon = df["longitude"] if "longitude" in df else pd.Series(np.nan, index=df.index)
    return C.clean_latlon(lat, lon, bbox=CHICAGO_BBOX)


def _num(df: pd.DataFrame, col: str) -> tuple[pd.Series, pd.Series]:
    """Numeric reading → (value, qualifier); negative readings are not physical → NaN + qualifier."""
    raw = df[col] if col in df else pd.Series("", index=df.index)
    v, q = C.to_number(raw)
    neg = v < 0
    q = q.where(~neg, raw.astype(str))
    return v.mask(neg), q


def _gmean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Geometric mean of the replicate readings that are present (a zero reading gives 0)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        one = np.where(np.isnan(b), a, b)
        return np.where(np.isnan(a) | np.isnan(b), one, np.sqrt(a * b))


def lab_test_result(r1: pd.Series, r2: pd.Series, published: pd.Series,
                tol_abs: float = 0.5, tol_rel: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """(result, mean_check) for one lab test.

    ``result`` = geometric mean of the present replicates (NaN without readings). ``mean_check``:
    'ok' (published within ±(0.5 + 1 %)), 'published_mean_missing',
    'published_sqrt_of_single_replicate', 'published_mean_without_replicates',
    'published_mean_mismatch', 'no_result'.
    """
    a, b, p = (np.asarray(x, dtype=float) for x in (r1, r2, published))
    n = (~np.isnan(a)).astype(int) + (~np.isnan(b)).astype(int)
    rec = _gmean(a, b)
    result = np.where(n > 0, rec, np.nan)
    check = np.full(len(a), "ok", dtype=object)
    has_p = ~np.isnan(p)
    with np.errstate(invalid="ignore"):
        close = np.abs(p - rec) <= tol_abs + tol_rel * np.abs(p)
        single_val = np.where(np.isnan(a), b, a)
        sqrt_art = (n == 1) & has_p & ~close & (np.abs(p - np.sqrt(single_val)) <= 0.01 + 0.001 * np.abs(p))
    check[(n > 0) & ~has_p] = "published_mean_missing"
    check[(n > 0) & has_p & ~close] = "published_mean_mismatch"
    check[sqrt_art] = "published_sqrt_of_single_replicate"
    check[(n == 0) & has_p] = "published_mean_without_replicates"
    check[(n == 0) & ~has_p] = "no_result"
    return result, check


# ----------------------------------------------------------------------------- lab data

def parse_lab(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Beach Lab Data → (culture E. coli table, DNA/qPCR Enterococci table).

    A parent row can carry a DNA test, a culture test, or both (629 rows); each test goes to its
    own table, one output row per test (one or two replicate samples).
    """
    df = raw.copy()
    df["beach"] = df["beach"].fillna("").str.strip()
    df["beach_key"] = beach_key(df["beach"])
    df["latitude"], df["longitude"], df["coord_flag"] = _coords(df)
    return _parse_culture(df), _parse_dna(df)


def _has_any(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    m = pd.Series(False, index=df.index)
    for c in cols:
        if c in df:
            m |= df[c].fillna("").astype(str).str.strip() != ""
    return m


def _censoring(r1: pd.Series, r2: pd.Series, *, cap: tuple[float, float] | None
               ) -> tuple[pd.Series, pd.Series, pd.Series]:
    left = (r1 == 0) | (r2 == 0)
    if cap is None:
        right = pd.Series(False, index=r1.index)
    else:
        right = r1.between(*cap) | r2.between(*cap)
    side = np.where(left & right, "both", np.where(left, "left", np.where(right, "right", "")))
    dl = pd.Series(np.where(right, CULTURE_UPPER_LIMIT, np.nan), index=r1.index)
    return (left | right), pd.Series(side, index=r1.index), dl


def _midnight(t: pd.Series) -> pd.Series:
    return (t.dt.hour == 0) & (t.dt.minute == 0) & (t.dt.second == 0)


def _parse_dna(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["dna_test_id", "dna_sample_1_reading", "dna_sample_2_reading", "dna_reading_mean"]
    d = df[_has_any(df, cols)].copy()
    # A DNA test appears twice when the same day had two culture tests (IDs 69, 541): same test.
    d = d.drop_duplicates(subset=["dna_test_id", "dna_sample_timestamp", "beach"] + cols[1:])
    t_local, t_utc, date, flag = local_times(d["dna_sample_timestamp"])
    time_known = t_local.notna() & ~_midnight(t_local)
    r1, q1 = _num(d, "dna_sample_1_reading")
    r2, q2 = _num(d, "dna_sample_2_reading")
    pub, qm = _num(d, "dna_reading_mean")
    result, check = lab_test_result(r1, r2, pub)
    cens, side, dl = _censoring(r1, r2, cap=None)
    out = pd.DataFrame({
        "dna_test_id": C.str_id(d["dna_test_id"]),
        "beach": d["beach"], "beach_key": d["beach_key"],
        "sample_timestamp_raw": d["dna_sample_timestamp"],
        "sample_time_local": t_local, "time_utc": t_utc.where(time_known), "date": date,
        "date_flag": flag, "time_known": time_known,
        "indicator": "Enterococci (qPCR, DNA)", "unit": QPCR_UNIT,
        "reading_1": r1, "reading_2": r2, "reading_mean_published": pub,
        "result": result, "mean_check": check,
        "n_replicates": r1.notna().astype(int) + r2.notna().astype(int),
        "censored": cens, "censor_side": side, "detection_limit": dl,
        "qualifier": (q1 + " " + q2 + " " + qm).str.strip(),
        "latitude": d["latitude"], "longitude": d["longitude"], "coord_flag": d["coord_flag"],
    })
    out[f"exceeds_{int(QPCR_ADVISORY_CCE)}_cce"] = (out["result"] > QPCR_ADVISORY_CCE).where(
        out["result"].notna()).astype("boolean")
    return out.sort_values(["beach_key", "sample_time_local"], na_position="last").reset_index(drop=True)


def _parse_culture(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["culture_test_id", "culture_sample_1_reading", "culture_sample_2_reading", "culture_reading_mean"]
    d = df[_has_any(df, cols)].copy()
    t_local, t_utc, date, flag = local_times(d["culture_sample_1_timestamp"])
    _, t2_utc, _, flag2 = local_times(d["culture_sample_2_timestamp"])
    time_known = t_local.notna() & ~_midnight(t_local)   # no sample times before 2016
    r1, q1 = _num(d, "culture_sample_1_reading")
    r2, q2 = _num(d, "culture_sample_2_reading")
    pub, qm = _num(d, "culture_reading_mean")
    interval, qi = C.to_number(d["culture_sample_interval"])
    result, check = lab_test_result(r1, r2, pub)
    cens, side, dl = _censoring(r1, r2, cap=CULTURE_CAP_WINDOW)
    out = pd.DataFrame({
        "culture_test_id": C.str_id(d["culture_test_id"]),
        "beach": d["beach"], "beach_key": d["beach_key"],
        "sample_timestamp_raw": d["culture_sample_1_timestamp"],
        "sample_time_local": t_local, "time_utc": t_utc.where(time_known), "date": date,
        "date_flag": flag, "time_known": time_known,
        "sample_2_time_utc": t2_utc, "sample_2_date_flag": flag2, "sample_interval_min": interval,
        "indicator": "Escherichia coli (culture)", "unit": CULTURE_UNIT,
        "reading_1": r1, "reading_2": r2, "reading_mean_published": pub,
        "result": result, "mean_check": check,
        "n_replicates": r1.notna().astype(int) + r2.notna().astype(int),
        "censored": cens, "censor_side": side, "detection_limit": dl,
        "qualifier": (q1 + " " + q2 + " " + qm + " " + qi).str.strip(),
        "culture_note": d["culture_note"].fillna("").str.strip(),
        "latitude": d["latitude"], "longitude": d["longitude"], "coord_flag": d["coord_flag"],
    })
    out[f"exceeds_{int(CULTURE_BAV_CFU)}_cfu"] = (out["result"] > CULTURE_BAV_CFU).where(
        out["result"].notna()).astype("boolean")
    return out.sort_values(["beach_key", "sample_time_local"], na_position="last").reset_index(drop=True)


def fill_coords_from_lookup(df: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    """Fill missing lat/lon from the beach lookup (culture-era rows carry no coordinates)."""
    out = df.copy()
    ref = lookup.set_index("beach_key")[["latitude", "longitude"]]
    miss = out["latitude"].isna() & out["beach_key"].notna()
    lat = out.loc[miss, "beach_key"].map(ref["latitude"])
    lon = out.loc[miss, "beach_key"].map(ref["longitude"])
    ok = lat.notna() & lon.notna()
    idx = lat.index[ok]
    out.loc[idx, "latitude"] = lat[ok].astype(float)
    out.loc[idx, "longitude"] = lon[ok].astype(float)
    out.loc[idx, "coord_flag"] = "from_beach_lookup"
    return out


# ----------------------------------------------------------------------------- predictions / advisories

def parse_predictions(raw: pd.DataFrame) -> pd.DataFrame:
    """Official Chicago Park District predictions (2017–), one row per beach-day."""
    df = raw.copy()
    local, flag = C.clean_dates(df["date"], min_date="2015-01-01")
    lev, q = C.to_number(df["predicted_level"])
    neg = lev < 0
    q = q.where(~neg, "predicted_level=" + df["predicted_level"].astype(str))
    lat, lon, cflag = _coords(df)
    out = pd.DataFrame({
        "record_id": df["recordid"].astype("string"),
        "beach_name": df["beach_name"].str.strip(), "beach_key": beach_key(df["beach_name"]),
        "date": local.dt.date.astype("object").where(local.notna(), None), "date_flag": flag,
        "prediction_source": df["prediction_source"].str.strip(),
        "predicted_level": lev.mask(neg), "qualifier": q,
        # The column description says "E. coli CFU/100 mL", but the production model is a random
        # forest on same-day qPCR (CCE) inputs; its output scale is recorded as published.
        "predicted_level_unit_as_published": "CFU/100 mL (column description; scale unverified)",
        "latitude": lat, "longitude": lon, "coord_flag": cflag,
    })
    return out.sort_values(["beach_key", "date"], na_position="last").reset_index(drop=True)


def parse_advisories(raw: pd.DataFrame) -> pd.DataFrame:
    """Historical (2016) swim advisories from the culture-era statistical model."""
    df = raw.copy()
    local, flag = C.clean_dates(df["date"], min_date="2015-01-01")
    lev, q = C.to_number(df["predicted_level"])
    neg = lev < 0  # e.g. -1000 with probability -1: no prediction issued
    prob, _ = C.to_number(df["probability"])
    bad_p = (prob < 0) | (prob > 1)
    adv = df["swim_advisory"].str.strip().str.upper().map({"Y": True, "N": False}).astype("boolean")
    ql = pd.Series(np.where(neg, "predicted_level=" + df["predicted_level"].astype(str), ""), index=df.index)
    qp = pd.Series(np.where(bad_p, "probability=" + df["probability"].astype(str), ""), index=df.index)
    out = pd.DataFrame({
        "record_id": df["recordid"].astype("string"),
        "beach_name": df["beach_name"].str.strip(), "beach_key": beach_key(df["beach_name"]),
        "date": local.dt.date.astype("object").where(local.notna(), None), "date_flag": flag,
        "predicted_level_cfu": lev.mask(neg),
        "prob_ge_235_cfu": prob.mask(bad_p),
        "swim_advisory": adv,
        "swim_advisory_raw": df["swim_advisory"].str.strip(),
        "qualifier": (q + " " + ql + " " + qp).str.strip(),
    })
    return out.sort_values(["beach_key", "date"], na_position="last").reset_index(drop=True)


# ----------------------------------------------------------------------------- automated sensors

# variable → (unit, plausible range [lo, hi]); outside → NaN + "out_of_range:<raw>". Ranges are
# physical-plausibility screens for Lake Michigan beach sensors, not instrument QC.
WATER_VARS = {
    "water_temperature": ("degC", (-1.0, 35.0)),
    "turbidity": ("NTU", (0.0, 4000.0)),
    "transducer_depth": ("m", (-5.0, 20.0)),
    "wave_height": ("m", (0.0, 10.0)),
    "wave_period": ("s", (0.0, 30.0)),
    "battery_life": ("V", (0.0, 20.0)),
}
WEATHER_VARS = {
    "air_temperature": ("degC", (-40.0, 50.0)),
    "wet_bulb_temperature": ("degC", (-40.0, 50.0)),
    "humidity": ("%", (0.0, 100.0)),
    "rain_intensity": ("mm/h", (0.0, 300.0)),
    "interval_rain": ("mm", (0.0, 300.0)),
    "total_rain": ("mm", (0.0, 2000.0)),
    "precipitation_type": ("code", (0.0, 99.0)),
    "wind_direction": ("deg", (0.0, 360.0)),
    "wind_speed": ("m/s", (0.0, 75.0)),
    "maximum_wind_speed": ("m/s", (0.0, 75.0)),
    "barometric_pressure": ("hPa", (870.0, 1085.0)),
    "solar_radiation": ("W/m2", (-20.0, 1500.0)),
    "heading": ("deg", (0.0, 360.0)),
    "battery_life": ("V", (0.0, 20.0)),
}
# Documented / observed missing-value codes. Any value ≤ −9999 is also treated as a code.
MISSING_CODES = {"wind_speed": (999.9,), "maximum_wind_speed": (999.9,)}
PRECIP_TYPE_CODES = {0: "none", 60: "liquid", 70: "solid", 40: "unspecified"}


def _sensors_long(raw: pd.DataFrame, name_col: str, variables: dict) -> pd.DataFrame:
    df = raw.copy()
    t_local, t_utc, date, flag = local_times(df["measurement_timestamp"], min_date="2010-01-01",
                                             repair_year=False)
    base = pd.DataFrame({
        "sensor_name": df[name_col].str.strip(), "beach_key": beach_key(df[name_col]),
        "measurement_id": df["measurement_id"].astype("string"),
        "time_local": t_local, "time_utc": t_utc, "date": date, "date_flag": flag,
    })
    present = [v for v in variables if v in df]
    texts = {v: df[v].fillna("").astype(str).str.strip() for v in present}
    values = {v: C.to_number(texts[v], missing_codes=()) for v in present}
    # Placeholder transmissions (every reported variable exactly 0, battery included; 48 records,
    # all March 2025) carry no measurement → masked as 'all_zero_record'.
    n_rep = sum((texts[v] != "").astype(int) for v in present)
    n_zero = sum((values[v][0] == 0).astype(int) for v in present)
    all_zero = (n_rep >= 3) & (n_zero == n_rep)
    parts = []
    for var in present:
        unit, (lo, hi) = variables[var]
        txt = texts[var]
        v, q = values[var]
        code = (v <= -9999) | v.isin(MISSING_CODES.get(var, ()))
        oor = ~code & ((v < lo) | (v > hi))
        q = q.where(~code, "missing_code:" + txt).where(~oor, "out_of_range:" + txt)
        if var == "turbidity":
            q = q.where(~(v == 0), "zero_reading")  # value kept; a working sensor rarely reads 0.00
        q = q.where(~all_zero, "all_zero_record")
        part = base.copy()
        part["variable"] = var
        part["value"] = v.mask(code | oor | all_zero).astype("float64")
        part["unit"] = unit
        part["qualifier"] = q
        parts.append(part[(txt != "").to_numpy()])
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["sensor_name", "time_local", "variable"], na_position="last").reset_index(drop=True)


def parse_water_sensors(raw: pd.DataFrame) -> pd.DataFrame:
    """Hourly in-lake beach sensors → long table (one row per measurement × variable)."""
    return _sensors_long(raw, "beach_name", WATER_VARS)


def parse_weather_sensors(raw: pd.DataFrame) -> pd.DataFrame:
    """Hourly beach weather stations → long table (context covariates only)."""
    return _sensors_long(raw, "station_name", WEATHER_VARS)


def sensors_wide(long: pd.DataFrame) -> pd.DataFrame:
    """Long sensor table → wide (``<variable>_<unit>`` columns) for modelling convenience."""
    col = long["variable"] + "_" + long["unit"].str.replace(r"[^A-Za-z0-9]", "", regex=True)
    w = long.assign(col=col).pivot_table(index=["sensor_name", "beach_key", "time_utc", "date"],
                                         columns="col", values="value", aggfunc="first")
    return w.reset_index()


def parse_sensor_locations(raw: pd.DataFrame) -> pd.DataFrame:
    lat, lon, flag = _coords(raw)
    return pd.DataFrame({
        "sensor_name": raw["sensor_name"].str.strip(), "sensor_type": raw["sensor_type"].str.strip(),
        "beach_key": beach_key(raw["sensor_name"]),
        "latitude": lat, "longitude": lon, "coord_flag": flag,
    })


# ----------------------------------------------------------------------------- lookup & matching

def beach_lookup(culture: pd.DataFrame, dna: pd.DataFrame, predictions: pd.DataFrame,
                 advisories: pd.DataFrame, water_sensors: pd.DataFrame,
                 sensor_locations: pd.DataFrame) -> pd.DataFrame:
    """One row per beach key: name variants, representative lat/lon, per-dataset row counts."""
    frames = {
        "culture": culture.rename(columns={"beach": "name"}),
        "dna": dna.rename(columns={"beach": "name"}),
        "predictions": predictions.rename(columns={"beach_name": "name"}),
        "advisories": advisories.rename(columns={"beach_name": "name"}),
        "water_sensor_hours": water_sensors.drop_duplicates("measurement_id").rename(
            columns={"sensor_name": "name"}),
    }
    keys = sorted(set().union(*[set(f["beach_key"].dropna()) for f in frames.values()]))
    rows = []
    for k in keys:
        r = {"beach_key": k}
        names, lats, lons = set(), [], []
        for nm, f in frames.items():
            g = f[f["beach_key"] == k]
            r[f"n_{nm}"] = int(len(g))
            names |= set(g["name"].dropna().astype(str))
            if "latitude" in g:
                ok = g["latitude"].notna() & g["longitude"].notna() & (g.get("coord_flag", "") != "from_beach_lookup")
                lats += list(g.loc[ok, "latitude"])
                lons += list(g.loc[ok, "longitude"])
        r["names"] = " | ".join(sorted(names))
        r["latitude"] = float(np.median(lats)) if lats else np.nan
        r["longitude"] = float(np.median(lons)) if lons else np.nan
        r["n_distinct_coords"] = len(set(zip(np.round(lats, 4), np.round(lons, 4))))
        r["is_predicted_beach"] = r["n_predictions"] > 0
        r["is_model_input_beach"] = k in MODEL_INPUT_BEACHES
        loc = sensor_locations[(sensor_locations["beach_key"] == k)
                               & (sensor_locations["sensor_type"].str.lower() == "water")]
        r["water_sensor_latitude"] = float(loc["latitude"].iloc[0]) if len(loc) else np.nan
        r["water_sensor_longitude"] = float(loc["longitude"].iloc[0]) if len(loc) else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def daily_qpcr(dna: pd.DataFrame) -> pd.DataFrame:
    """Beach-day qPCR: geometric mean of the day's test results (valid dates/results only)."""
    d = dna[dna["date"].notna() & dna["result"].notna() & dna["beach_key"].notna()].copy()
    d["logv"] = np.log(d["result"].clip(lower=1.0))
    g = d.groupby(["beach_key", "date"], as_index=False).agg(
        qpcr_gmean_cce=("logv", lambda x: float(np.exp(x.mean()))), n_tests=("logv", "size"))
    return g.sort_values(["beach_key", "date"]).reset_index(drop=True)


def match_predictions_to_qpcr(predictions: pd.DataFrame, dna: pd.DataFrame,
                              max_prev_gap_days: int | None = None) -> pd.DataFrame:
    """Same-beach, same-local-date pairs of official prediction and observed qPCR.

    ``prev_gap_days``/``prev_qpcr_gmean_cce`` describe the previous qPCR day at the same beach
    (persistence baseline). With ``max_prev_gap_days`` the pairs are restricted to beach-days whose
    previous qPCR is at most that many days earlier (the rule of docs/research/07 §b.2).
    """
    q = daily_qpcr(dna)
    q["date_ts"] = pd.to_datetime(q["date"])
    q["prev_gap_days"] = q.groupby("beach_key")["date_ts"].diff().dt.days
    q["prev_qpcr_gmean_cce"] = q.groupby("beach_key")["qpcr_gmean_cce"].shift(1)
    p = predictions[predictions["date"].notna() & predictions["predicted_level"].notna()]
    p = p.groupby(["beach_key", "date"], as_index=False).agg(
        predicted_level=("predicted_level", "mean"), prediction_source=("prediction_source", "first"))
    m = p.merge(q.drop(columns="date_ts"), on=["beach_key", "date"], how="inner")
    m[f"exceeds_{int(QPCR_ADVISORY_CCE)}_cce"] = m["qpcr_gmean_cce"] > QPCR_ADVISORY_CCE
    if max_prev_gap_days is not None:
        m = m[m["prev_gap_days"] <= max_prev_gap_days]
    return m.reset_index(drop=True)


def name_variants(*series: pd.Series) -> dict[str, list[str]]:
    """Diagnostic: beach key → sorted raw name variants (to audit the key normalisation)."""
    out: dict[str, set] = {}
    for s in series:
        for nm, k in zip(s.astype(str), beach_key(s)):
            if pd.isna(k):
                continue
            out.setdefault(k, set()).add(re.sub(r"\s+", " ", nm.strip()))
    return {k: sorted(v) for k, v in sorted(out.items())}
