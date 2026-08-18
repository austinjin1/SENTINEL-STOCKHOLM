#!/usr/bin/env python3
"""Fetch a REAL, non-circular FIB beach-nowcast dataset for an ARBITRARY region.

Region-parameterized generalization of scripts/data/fetch_fib_nowcast_data.py
(which is hard-coded to Ohio). Same task, same label channel, same strictly
antecedent / independent predictors:

  * LABEL   : WQP E. coli culture result; exceedance = value > 235 CFU/100mL
              (EPA 2012 RWQC single-sample BAV for E. coli, freshwater).
  * PREDICT : NOAA GHCN-Daily antecedent rainfall (1/2/3/7-day prior totals +
              same-day), day-of-year sin/cos, month, site lat/lon, ghcn_dist_km.
    NO FIB value or any function of it is used as a predictor.

Every field is a REAL public API pull, cached per-region so re-runs are cheap.
GHCN rainfall pull is time-boxed.

Usage:
    PYTHONNOUSERSITE=1 ~/.conda/envs/physiformer/bin/python \
        scripts/data/fetch_fib_nowcast_region.py --region MI

Regions are defined in REGIONS below (statecode + a GHCN station bbox).
"""
from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import urllib.request
import urllib.parse

PROJECT = Path(__file__).resolve().parents[2]
OUT_BASE = PROJECT / "data" / "processed" / "fib_nowcast"

BAV_ECOLI = 235.0  # CFU/100mL (freshwater / Great Lakes single-sample BAV)
CHAR = "Escherichia coli"
DATE_LO = "01-01-2015"
DATE_HI = "12-31-2023"

WQP_RESULT = "https://www.waterqualitydata.us/data/Result/search"
WQP_STATION = "https://www.waterqualitydata.us/data/Station/search"
GHCN_STATIONS = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt"
NCEI_DATA = "https://www.ncei.noaa.gov/access/services/data/v1"

# region key -> (statecode, ghcn bbox lat_lo, lat_hi, lon_lo, lon_hi, human name)
REGIONS = {
    "OH": ("US:39", 38.0, 42.5, -85.5, -80.0, "Ohio (Great Lakes / Ohio R.)"),
    "MI": ("US:26", 41.0, 48.5, -90.5, -82.0, "Michigan (Great Lakes)"),
    "WI": ("US:55", 42.0, 47.5, -93.5, -86.5, "Wisconsin (Great Lakes / inland)"),
    "CA": ("US:06", 32.3, 42.2, -124.6, -114.0, "California (Pacific coast + inland)"),
    "IL": ("US:17", 36.9, 42.6, -91.6, -87.0, "Illinois (Lake Michigan / inland)"),
    "MN": ("US:27", 43.4, 49.5, -97.4, -89.4, "Minnesota (inland lakes)"),
}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def http_get(url, timeout=180):
    req = urllib.request.Request(url, headers={"User-Agent": "sentinel-fib-nowcast/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def haversine(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


class RegionFetcher:
    def __init__(self, key):
        if key not in REGIONS:
            raise SystemExit(f"unknown region {key}; choices={list(REGIONS)}")
        self.key = key
        (self.statecode, self.lat_lo, self.lat_hi,
         self.lon_lo, self.lon_hi, self.name) = REGIONS[key]
        self.out = OUT_BASE / f"region_{key}"
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "ghcn_prcp").mkdir(exist_ok=True)

    # -- WQP FIB results (LABEL) --
    def fetch_fib_results(self):
        cache = self.out / "wqp_ecoli.csv"
        if cache.exists():
            log(f"[{self.key}] FIB results cached")
            return pd.read_csv(cache, low_memory=False)
        params = {
            "countrycode": "US", "statecode": self.statecode,
            "characteristicName": CHAR, "startDateLo": DATE_LO, "startDateHi": DATE_HI,
            "mimeType": "csv", "zip": "no", "dataProfile": "resultPhysChem",
        }
        url = WQP_RESULT + "?" + urllib.parse.urlencode(params)
        log(f"[{self.key}] Fetching WQP E. coli results ...")
        raw = http_get(url, timeout=600)
        df = pd.read_csv(io.BytesIO(raw), low_memory=False)
        df.to_csv(cache, index=False)
        log(f"[{self.key}]   wrote {len(df)} result rows")
        return df

    def fetch_stations(self):
        cache = self.out / "wqp_stations.csv"
        if cache.exists():
            return pd.read_csv(cache, low_memory=False)
        params = {"countrycode": "US", "statecode": self.statecode,
                  "characteristicName": CHAR, "mimeType": "csv", "zip": "no"}
        url = WQP_STATION + "?" + urllib.parse.urlencode(params)
        log(f"[{self.key}] Fetching WQP station coords ...")
        raw = http_get(url, timeout=300)
        df = pd.read_csv(io.BytesIO(raw), low_memory=False)
        df.to_csv(cache, index=False)
        return df

    def load_ghcn_stations(self):
        cache = self.out / "ghcnd_stations_bbox.csv"
        if cache.exists():
            return pd.read_csv(cache)
        # global inventory cached at base level (shared across regions)
        inv_cache = OUT_BASE / "ghcnd_stations_all.txt"
        if inv_cache.exists():
            raw = inv_cache.read_text()
        else:
            log("Fetching GHCN global station inventory (shared) ...")
            raw = http_get(GHCN_STATIONS, timeout=300).decode("utf-8", "ignore")
            inv_cache.write_text(raw)
        rows = []
        for ln in raw.splitlines():
            try:
                sid = ln[0:11].strip()
                lat = float(ln[12:20]); lon = float(ln[21:30])
            except Exception:
                continue
            if self.lat_lo <= lat <= self.lat_hi and self.lon_lo <= lon <= self.lon_hi:
                rows.append((sid, lat, lon))
        d = pd.DataFrame(rows, columns=["id", "lat", "lon"])
        d.to_csv(cache, index=False)
        log(f"[{self.key}]   {len(d)} GHCN stations in bbox")
        return d

    def fetch_ghcn_prcp(self, station_id, y0, y1):
        cache = self.out / "ghcn_prcp" / f"{station_id}.csv"
        miss = self.out / "ghcn_prcp" / f"{station_id}.MISS"
        if miss.exists():
            return None
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
            miss.write_text(f"err {e}")
            return None
        if "PRCP" not in d.columns or "DATE" not in d.columns:
            miss.write_text("nocols")
            return None
        d = d.rename(columns={"DATE": "date", "PRCP": "prcp_mm"})[["date", "prcp_mm"]]
        d["date"] = pd.to_datetime(d["date"])
        d["prcp_mm"] = pd.to_numeric(d["prcp_mm"], errors="coerce")
        d.to_csv(cache, index=False)
        return d

    def build(self, ghcn_budget_s=600):
        t0 = time.time()
        fib = self.fetch_fib_results()
        stations = self.fetch_stations()

        fib = fib.copy()
        fib["date"] = pd.to_datetime(fib["ActivityStartDate"], errors="coerce")
        fib["val"] = pd.to_numeric(fib["ResultMeasureValue"], errors="coerce")
        unit = fib["ResultMeasure/MeasureUnitCode"].astype(str)
        per100 = unit.str.contains("100", na=False)
        fib = fib[per100 & fib["val"].notna() & fib["date"].notna()].copy()
        if "ActivityMediaName" in fib.columns:
            fib = fib[fib["ActivityMediaName"].astype(str).str.contains("Water", case=False, na=False)]
        fib["site"] = fib["MonitoringLocationIdentifier"].astype(str)
        fib_daily = fib.groupby(["site", "date"])["val"].median().reset_index()
        fib_daily["exceed"] = (fib_daily["val"] > BAV_ECOLI).astype(int)
        log(f"[{self.key}] FIB site-day rows: {len(fib_daily)} | "
            f"sites={fib_daily.site.nunique()} | base={fib_daily.exceed.mean():.4f}")

        st = stations.rename(columns={
            "MonitoringLocationIdentifier": "site",
            "LatitudeMeasure": "lat", "LongitudeMeasure": "lon",
            "MonitoringLocationTypeName": "site_type"})
        st = st[["site", "lat", "lon", "site_type"]].dropna(subset=["lat", "lon"])
        st = st.drop_duplicates("site")
        fib_daily = fib_daily.merge(st, on="site", how="inner")

        counts = fib_daily.groupby("site")["exceed"].agg(["size", "sum"])
        keep = counts[counts["size"] >= 20].index
        fib_daily = fib_daily[fib_daily.site.isin(keep)].copy()
        log(f"[{self.key}] after >=20/site: {len(fib_daily)} rows | "
            f"{fib_daily.site.nunique()} sites | base={fib_daily.exceed.mean():.4f}")
        if fib_daily.empty:
            raise SystemExit(f"[{self.key}] no data after filtering")

        ghcn = self.load_ghcn_stations()
        K = 6
        site_coords = fib_daily.drop_duplicates("site")[["site", "lat", "lon"]].reset_index(drop=True)
        site2knn = {}
        for _, row in site_coords.iterrows():
            d = haversine(row.lat, row.lon, ghcn.lat.values, ghcn.lon.values)
            order = np.argsort(d)[:K]
            site2knn[row.site] = [(ghcn.id.values[i], float(d[i])) for i in order]
        unique_ghcn = sorted({gid for lst in site2knn.values() for gid, _ in lst})
        nearest_dist = {s: lst[0][1] for s, lst in site2knn.items()}
        log(f"[{self.key}] nearest GHCN median {np.median(list(nearest_dist.values())):.1f} km | "
            f"{len(unique_ghcn)} unique GHCN stations")

        y0 = int(fib_daily.date.dt.year.min())
        y1 = int(fib_daily.date.dt.year.max())
        log(f"[{self.key}] fetching PRCP for {len(unique_ghcn)} stations ({y0}-{y1}) ...")
        prcp = {}
        deadline = time.time() + ghcn_budget_s
        for k, sid in enumerate(unique_ghcn):
            if time.time() > deadline:
                log(f"[{self.key}]   GHCN budget hit at {k}/{len(unique_ghcn)}")
                break
            d = self.fetch_ghcn_prcp(sid, y0, y1)
            if d is not None and len(d):
                s = d.dropna(subset=["prcp_mm"]).set_index("date")["prcp_mm"].sort_index()
                if len(s):
                    prcp[sid] = s
            if k % 25 == 0:
                log(f"[{self.key}]   ...{k}/{len(unique_ghcn)}")
        log(f"[{self.key}]   {len(prcp)} GHCN stations with usable PRCP")

        def antecedent_for_row(site, date):
            for gid, dist in site2knn.get(site, []):
                s = prcp.get(gid)
                if s is None:
                    continue
                lo7 = date - pd.Timedelta(days=7)
                hi1 = date - pd.Timedelta(days=1)
                w7 = s.loc[(s.index >= lo7) & (s.index <= hi1)]
                if len(w7) == 0:
                    continue

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

        doy = fib_daily.date.dt.dayofyear
        fib_daily["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
        fib_daily["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
        fib_daily["month"] = fib_daily.date.dt.month
        fib_daily["year"] = fib_daily.date.dt.year
        fib_daily["region"] = self.key

        n_before = len(fib_daily)
        have_rain = fib_daily[["rain_prior1_mm", "rain_prior3_mm", "rain_prior7_mm"]].notna().any(axis=1)
        fib_daily = fib_daily[have_rain].copy()
        log(f"[{self.key}] rows with >=1 rainfall predictor: {len(fib_daily)} "
            f"(dropped {n_before - len(fib_daily)})")

        out = self.out / "fib_nowcast_dataset.parquet"
        fib_daily.to_parquet(out, index=False)
        meta = {
            "region_key": self.key,
            "region_name": self.name,
            "statecode": self.statecode,
            "date_range": [DATE_LO, DATE_HI],
            "epa_threshold_bav_ecoli_cfu_100ml": BAV_ECOLI,
            "n_rows": int(len(fib_daily)),
            "n_sites": int(fib_daily.site.nunique()),
            "n_positives": int(fib_daily.exceed.sum()),
            "base_rate": float(fib_daily.exceed.mean()),
            "n_ghcn_stations_used": len(prcp),
            "ghcn_dist_km_median": float(fib_daily["ghcn_dist_km"].median()),
            "build_seconds": round(time.time() - t0, 1),
        }
        (self.out / "dataset_meta.json").write_text(json.dumps(meta, indent=2))
        log(f"[{self.key}] WROTE {out}")
        log(json.dumps(meta, indent=2))
        return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True, help=f"one of {list(REGIONS)}")
    ap.add_argument("--ghcn-budget", type=int, default=600)
    args = ap.parse_args()
    RegionFetcher(args.region).build(ghcn_budget_s=args.ghcn_budget)


if __name__ == "__main__":
    main()
