#!/usr/bin/env python3
"""Fetch a REAL, non-circular beach-nowcasting dataset (FIB exceedance).

Task
----
Predict EXCEEDANCE of the EPA recreational-water Beach Action Value (BAV) for
a fecal-indicator bacterium (FIB) at a monitoring site/date, from strictly
ANTECEDENT / independent predictors. This is the USGS/EPA "Virtual Beach"
nowcast problem.

  * LABEL channel  : E. coli (culture) result from the Water Quality Portal
                     (WQP) Result service. Exceedance = value > 235 CFU/100mL
                     (EPA 2012 RWQC single-sample BAV for E. coli).
  * PREDICTOR chans: (all independent of the FIB measurement)
      - Antecedent rainfall from NOAA GHCN-Daily (nearest station):
        1-, 2-, 3-, and 7-day prior totals + a same-day/prior-day flag.
      - Day-of-year (sin/cos) and month/season.
      - Site latitude/longitude (spatial context; NOT the label).
    NO FIB value or any function of it is used as a predictor.

Region: Ohio (statecode US:39) — Great Lakes / Ohio, one of the densest FIB
monitoring regions and the home of the USGS Great Lakes NowCast program.

All data are REAL public API pulls. Downloads are cached under
data/processed/fib_nowcast/ so re-runs are cheap. Time-boxed with timeouts.

Usage:
    PYTHONNOUSERSITE=1 ~/.conda/envs/physiformer/bin/python \
        scripts/data/fetch_fib_nowcast_data.py
"""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import urllib.request
import urllib.parse

PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT / "data" / "processed" / "fib_nowcast"
OUT.mkdir(parents=True, exist_ok=True)

# --- EPA thresholds (2012 Recreational Water Quality Criteria, single-sample BAV)
BAV_ECOLI = 235.0        # CFU/100mL   (freshwater / Great Lakes)
BAV_ENTERO = 104.0       # CFU/100mL   (marine)  -- not used in OH freshwater run

STATECODE = "US:39"      # Ohio
CHAR = "Escherichia coli"
DATE_LO = "01-01-2015"
DATE_HI = "12-31-2023"

WQP_RESULT = "https://www.waterqualitydata.us/data/Result/search"
WQP_STATION = "https://www.waterqualitydata.us/data/Station/search"
GHCN_STATIONS = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt"
NCEI_DATA = "https://www.ncei.noaa.gov/access/services/data/v1"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def http_get(url, timeout=180):
    req = urllib.request.Request(url, headers={"User-Agent": "sentinel-fib-nowcast/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


# ---------------------------------------------------------------------------
# 1. WQP FIB results (LABEL channel)
# ---------------------------------------------------------------------------
def fetch_fib_results() -> pd.DataFrame:
    cache = OUT / "wqp_ecoli_ohio.csv"
    if cache.exists():
        log(f"FIB results cached -> {cache.name}")
        return pd.read_csv(cache, low_memory=False)
    params = {
        "countrycode": "US", "statecode": STATECODE,
        "characteristicName": CHAR,
        "startDateLo": DATE_LO, "startDateHi": DATE_HI,
        "mimeType": "csv", "zip": "no", "dataProfile": "resultPhysChem",
    }
    url = WQP_RESULT + "?" + urllib.parse.urlencode(params)
    log(f"Fetching WQP E. coli results (this can take a minute) ...")
    raw = http_get(url, timeout=300)
    df = pd.read_csv(io.BytesIO(raw), low_memory=False)
    df.to_csv(cache, index=False)
    log(f"  wrote {len(df)} rows -> {cache.name}")
    return df


def fetch_stations() -> pd.DataFrame:
    cache = OUT / "wqp_stations_ohio.csv"
    if cache.exists():
        return pd.read_csv(cache, low_memory=False)
    params = {"countrycode": "US", "statecode": STATECODE,
              "characteristicName": CHAR, "mimeType": "csv", "zip": "no"}
    url = WQP_STATION + "?" + urllib.parse.urlencode(params)
    log("Fetching WQP station coordinates ...")
    raw = http_get(url, timeout=180)
    df = pd.read_csv(io.BytesIO(raw), low_memory=False)
    df.to_csv(cache, index=False)
    return df


# ---------------------------------------------------------------------------
# 2. GHCN-Daily rainfall (PREDICTOR channel, independent of FIB)
# ---------------------------------------------------------------------------
def load_ghcn_stations() -> pd.DataFrame:
    cache = OUT / "ghcnd_stations_ohbbox.csv"
    if cache.exists():
        return pd.read_csv(cache)
    log("Fetching GHCN station inventory ...")
    raw = http_get(GHCN_STATIONS, timeout=180).decode("utf-8", "ignore")
    rows = []
    for ln in raw.splitlines():
        try:
            sid = ln[0:11].strip()
            lat = float(ln[12:20]); lon = float(ln[21:30])
        except Exception:
            continue
        # Ohio + adjacent bbox
        if 38.0 <= lat <= 42.5 and -85.5 <= lon <= -80.0:
            rows.append((sid, lat, lon))
    d = pd.DataFrame(rows, columns=["id", "lat", "lon"])
    d.to_csv(cache, index=False)
    log(f"  {len(d)} GHCN stations in bbox")
    return d


def haversine(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1); dl = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def fetch_ghcn_prcp(station_id, y0, y1) -> pd.DataFrame | None:
    """Daily PRCP (mm) for one GHCN station over [y0, y1]."""
    cache = OUT / "ghcn_prcp" / f"{station_id}.csv"
    cache.parent.mkdir(exist_ok=True)
    if cache.exists():
        try:
            return pd.read_csv(cache, parse_dates=["date"])
        except Exception:
            return None
    params = {
        "dataset": "daily-summaries", "stations": station_id,
        "startDate": f"{y0}-01-01", "endDate": f"{y1}-12-31",
        "dataTypes": "PRCP", "format": "csv", "units": "metric",
    }
    url = NCEI_DATA + "?" + urllib.parse.urlencode(params)
    try:
        raw = http_get(url, timeout=90)
        d = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        log(f"    GHCN {station_id} failed: {e}")
        (cache.parent / f"{station_id}.MISS").write_text("miss")
        return None
    if "PRCP" not in d.columns or "DATE" not in d.columns:
        (cache.parent / f"{station_id}.MISS").write_text("nocols")
        return None
    d = d.rename(columns={"DATE": "date", "PRCP": "prcp_mm"})[["date", "prcp_mm"]]
    d["date"] = pd.to_datetime(d["date"])
    d["prcp_mm"] = pd.to_numeric(d["prcp_mm"], errors="coerce")
    d.to_csv(cache, index=False)
    return d


# ---------------------------------------------------------------------------
# 3. Build the analysis table
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    fib = fetch_fib_results()
    stations = fetch_stations()

    # --- clean FIB label channel ---
    fib = fib.copy()
    fib["date"] = pd.to_datetime(fib["ActivityStartDate"], errors="coerce")
    fib["val"] = pd.to_numeric(fib["ResultMeasureValue"], errors="coerce")
    unit = fib["ResultMeasure/MeasureUnitCode"].astype(str)
    per100 = unit.str.contains("100", na=False)  # cfu/100mL, MPN/100 ml
    fib = fib[per100 & fib["val"].notna() & fib["date"].notna()].copy()
    # Media: water only
    if "ActivityMediaName" in fib.columns:
        fib = fib[fib["ActivityMediaName"].astype(str).str.contains("Water", case=False, na=False)]
    fib["site"] = fib["MonitoringLocationIdentifier"].astype(str)
    # One value per site/day (median if multiple grabs same day) -- label only
    fib_daily = (fib.groupby(["site", "date"])["val"].median().reset_index())
    fib_daily["exceed"] = (fib_daily["val"] > BAV_ECOLI).astype(int)
    log(f"FIB label rows (site-day): {len(fib_daily)} | "
        f"sites={fib_daily.site.nunique()} | base rate={fib_daily.exceed.mean():.4f}")

    # --- site coordinates ---
    st = stations.rename(columns={
        "MonitoringLocationIdentifier": "site",
        "LatitudeMeasure": "lat", "LongitudeMeasure": "lon",
        "MonitoringLocationTypeName": "site_type"})
    st = st[["site", "lat", "lon", "site_type"]].dropna(subset=["lat", "lon"])
    st = st.drop_duplicates("site")
    fib_daily = fib_daily.merge(st, on="site", how="inner")
    log(f"After coord join: {len(fib_daily)} rows | {fib_daily.site.nunique()} sites")

    # Keep sites with enough samples (>=20) for meaningful site-grouped CV
    counts = fib_daily.groupby("site")["exceed"].agg(["size", "sum"])
    keep_sites = counts[counts["size"] >= 20].index
    fib_daily = fib_daily[fib_daily.site.isin(keep_sites)].copy()
    log(f"After >=20/site filter: {len(fib_daily)} rows | {fib_daily.site.nunique()} sites | "
        f"positives={int(fib_daily.exceed.sum())} | base rate={fib_daily.exceed.mean():.4f}")

    # --- assign K nearest GHCN stations to each FIB site (robust to gaps) ---
    ghcn = load_ghcn_stations()
    K = 6  # consider up to 6 nearest stations; use nearest-with-data per date
    site_coords = fib_daily.drop_duplicates("site")[["site", "lat", "lon"]].reset_index(drop=True)
    site2knn = {}          # site -> [(ghcn_id, dist_km), ...] sorted by distance
    for _, row in site_coords.iterrows():
        d = haversine(row.lat, row.lon, ghcn.lat.values, ghcn.lon.values)
        order = np.argsort(d)[:K]
        site2knn[row.site] = [(ghcn.id.values[i], float(d[i])) for i in order]
    unique_ghcn = sorted({gid for lst in site2knn.values() for gid, _ in lst})
    nearest_dist = {s: lst[0][1] for s, lst in site2knn.items()}
    log(f"Nearest GHCN median dist: {np.median(list(nearest_dist.values())):.1f} km | "
        f"{len(unique_ghcn)} unique GHCN stations across K={K} neighbors")

    # --- fetch PRCP for the needed GHCN stations (time-boxed) ---
    y0 = int(fib_daily.date.dt.year.min())
    y1 = int(fib_daily.date.dt.year.max())
    log(f"Fetching PRCP for {len(unique_ghcn)} GHCN stations ({y0}-{y1}) ...")
    prcp = {}
    deadline = time.time() + 600  # 10-min budget for rainfall pulls
    for k, sid in enumerate(unique_ghcn):
        if time.time() > deadline:
            log(f"  GHCN time budget hit at {k}/{len(unique_ghcn)}; using what we have")
            break
        d = fetch_ghcn_prcp(sid, y0, y1)
        if d is not None and len(d):
            s = d.dropna(subset=["prcp_mm"]).set_index("date")["prcp_mm"].sort_index()
            if len(s):
                prcp[sid] = s
        if k % 10 == 0:
            log(f"  ...{k}/{len(unique_ghcn)}")
    log(f"  {len(prcp)} GHCN stations returned usable PRCP")

    # --- build antecedent rainfall features per FIB row ---
    # For each row: iterate site's K nearest GHCN stations in order; use the
    # first station that has PRCP coverage for the [date-7, date-1] window.
    def antecedent_for_row(site, date):
        for gid, dist in site2knn.get(site, []):
            s = prcp.get(gid)
            if s is None:
                continue
            lo7 = date - pd.Timedelta(days=7)
            hi1 = date - pd.Timedelta(days=1)
            w7 = s.loc[(s.index >= lo7) & (s.index <= hi1)]
            if len(w7) == 0:
                continue  # this station has no data in the antecedent window
            def wsum(days):
                lo = date - pd.Timedelta(days=days)
                ww = s.loc[(s.index >= lo) & (s.index <= hi1)]
                return float(ww.sum()) if len(ww) else 0.0
            p1 = wsum(1); p2 = wsum(2); p3 = wsum(3); p7 = float(w7.sum())
            try:
                p0 = float(s.loc[date]) if date in s.index else 0.0
            except Exception:
                p0 = 0.0
            return (p1, p2, p3, p7, p0, dist)
        return (np.nan,) * 5 + (nearest_dist.get(site, np.nan),)

    feats = [antecedent_for_row(r.site, r.date) for r in fib_daily.itertuples()]
    fcols = ["rain_prior1_mm", "rain_prior2_mm", "rain_prior3_mm",
             "rain_prior7_mm", "rain_sameday_mm", "ghcn_dist_km"]
    fib_daily[fcols] = pd.DataFrame(feats, index=fib_daily.index)

    # --- temporal features ---
    doy = fib_daily.date.dt.dayofyear
    fib_daily["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    fib_daily["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    fib_daily["month"] = fib_daily.date.dt.month
    fib_daily["year"] = fib_daily.date.dt.year

    # Drop rows with no rainfall predictor at all (GHCN gap)
    n_before = len(fib_daily)
    have_rain = fib_daily[["rain_prior1_mm", "rain_prior3_mm", "rain_prior7_mm"]].notna().any(axis=1)
    fib_daily = fib_daily[have_rain].copy()
    log(f"Rows with >=1 rainfall predictor: {len(fib_daily)} (dropped {n_before - len(fib_daily)})")

    out = OUT / "fib_nowcast_dataset.parquet"
    fib_daily.to_parquet(out, index=False)

    meta = {
        "task": "Beach nowcasting: predict E. coli exceedance of EPA BAV",
        "label_channel": "WQP E. coli culture result (CFU or MPN /100mL)",
        "epa_threshold_bav_ecoli_cfu_100ml": BAV_ECOLI,
        "epa_criteria_source": "EPA 2012 Recreational Water Quality Criteria, single-sample BAV",
        "region": "Ohio (US:39), Great Lakes / Ohio",
        "date_range": [DATE_LO, DATE_HI],
        "predictors": fcols + ["doy_sin", "doy_cos", "month", "lat", "lon"],
        "predictor_channels_independent_of_label": True,
        "n_rows": int(len(fib_daily)),
        "n_sites": int(fib_daily.site.nunique()),
        "n_positives": int(fib_daily.exceed.sum()),
        "base_rate": float(fib_daily.exceed.mean()),
        "n_ghcn_stations_used": len(prcp),
        "ghcn_dist_km_median": float(fib_daily["ghcn_dist_km"].median()),
        "build_seconds": round(time.time() - t0, 1),
    }
    (OUT / "dataset_meta.json").write_text(json.dumps(meta, indent=2))
    log(f"WROTE {out}  ({len(fib_daily)} rows)")
    log(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
