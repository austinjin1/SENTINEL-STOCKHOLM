#!/usr/bin/env python3
"""EXPAND step 1 -- pull MORE real microcystin (WQP) across MANY states/HUC2.

Reuses the exact dataretrieval.wqp.get_results pattern from
download_habs_wqp_cyanotoxins.py, but sweeps a much broader set of states to
maximize distinct HUC2 region coverage for domain-generalization.

Writes: data/processed/biology/noaa_habs/mc_expand_raw.parquet
  columns: Location_Identifier, Location_Latitude, Location_Longitude,
           date, val (ug/L microcystin), huc2, state
Time-boxed via --minutes. Resumable per-state cache under mc_expand_cache/.
Real public data only.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dataretrieval import wqp

ROOT = Path("/home/bcheng/SENTINEL")
OUT_DIR = ROOT / "data/processed/biology/noaa_habs"
CACHE = OUT_DIR / "mc_expand_cache"
CACHE.mkdir(parents=True, exist_ok=True)

# Broad state sweep chosen to span many HUC2 first-level regions.
# (statecode FIPS -> name). Includes states already covered so the pull is a
# clean superset; dedup against existing dataset happens downstream.
STATES = {
    "US:09": "Connecticut",   "US:23": "Maine",        "US:25": "Massachusetts",  # 01 New England
    "US:36": "New York",      "US:42": "Pennsylvania", "US:24": "Maryland",       # 02 Mid-Atlantic
    "US:51": "Virginia",      "US:37": "North Carolina",                          # 02/03
    "US:12": "Florida",       "US:13": "Georgia",      "US:01": "Alabama",        # 03 South Atlantic-Gulf
    "US:39": "Ohio",          "US:26": "Michigan",     "US:18": "Indiana",        # 04/05
    "US:21": "Kentucky",      "US:47": "Tennessee",                               # 05/06
    "US:55": "Wisconsin",     "US:27": "Minnesota",    "US:19": "Iowa",           # 07
    "US:28": "Mississippi",   "US:22": "Louisiana",    "US:05": "Arkansas",       # 08/11
    "US:29": "Missouri",      "US:20": "Kansas",       "US:31": "Nebraska",       # 10
    "US:40": "Oklahoma",      "US:48": "Texas",                                   # 11/12
    "US:35": "New Mexico",    "US:08": "Colorado",     "US:49": "Utah",           # 13/14/16
    "US:04": "Arizona",       "US:32": "Nevada",                                  # 15/16
    "US:16": "Idaho",         "US:53": "Washington",   "US:41": "Oregon",         # 17
    "US:06": "California",                                                        # 18
    "US:30": "Montana",       "US:38": "North Dakota", "US:46": "South Dakota",   # 09/10/17
}

# Microcystin characteristic names that resolve in WQX3.0.
MC_NAMES = ["Microcystin", "Microcystin LR", "Microcystin-LR",
            "Total microcystins", "Microcystins and Nodularins"]

DATE_LO, DATE_HI = "01-01-2015", "12-31-2025"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def pull_state(statecode, name):
    frames = []
    for cn in MC_NAMES:
        try:
            df, _ = wqp.get_results(characteristicName=cn, statecode=statecode,
                                    startDateLo=DATE_LO, startDateHi=DATE_HI,
                                    legacy=False, dataProfile="narrow")
            if len(df):
                frames.append(df)
        except Exception as e:
            if "Bad Request" in str(e) or "No data" in str(e) or "204" in str(e):
                continue
    if not frames:
        return None
    d = pd.concat(frames, ignore_index=True)
    for c in d.select_dtypes(include="object").columns:
        d[c] = d[c].astype(str)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=20)
    args = ap.parse_args()
    deadline = time.time() + args.minutes * 60

    for sc, name in STATES.items():
        f = CACHE / f"{sc.replace(':', '_')}.parquet"
        if f.exists():
            continue
        if time.time() >= deadline:
            log("TIME BUDGET REACHED -- stopping pulls")
            break
        t0 = time.time()
        d = pull_state(sc, name)
        if d is None or len(d) == 0:
            # write an empty marker so we don't re-hit it
            pd.DataFrame().to_parquet(f)
            log(f"{name:16s} 0 rows ({time.time()-t0:.1f}s)")
            continue
        d.to_parquet(f, index=False)
        log(f"{name:16s} {len(d):6d} raw rows ({time.time()-t0:.1f}s)")

    # ---- assemble all cached states ----
    parts = []
    for f in CACHE.glob("*.parquet"):
        try:
            d = pd.read_parquet(f)
        except Exception:
            continue
        if len(d) == 0 or "Result_Characteristic" not in d.columns:
            continue
        parts.append(d)
    if not parts:
        log("NO DATA pulled.")
        return
    raw = pd.concat(parts, ignore_index=True)
    log(f"combined raw microcystin rows: {len(raw)}")

    # normalize -> microcystin ug/L only, valid lat/lon/date
    unit = raw["Result_MeasureUnit"].astype(str).str.lower()
    raw = raw[unit.isin(["ug/l", "ppb", "µg/l"])]
    raw["val"] = pd.to_numeric(raw["Result_Measure"], errors="coerce")
    raw["lat"] = pd.to_numeric(raw["Location_Latitude"], errors="coerce")
    raw["lon"] = pd.to_numeric(raw["Location_Longitude"], errors="coerce")
    raw["date"] = pd.to_datetime(raw["Activity_StartDate"], errors="coerce")
    raw = raw.dropna(subset=["val", "lat", "lon", "date"])
    raw = raw[(raw.lat.between(24, 50)) & (raw.lon.between(-125, -66))]
    # HUC2
    h8 = raw["Location_HUCEightDigitCode"].astype(str).str.replace(".0", "", regex=False)
    raw["huc2"] = np.where(h8.str.match(r"^\d+$"), h8.str.zfill(8).str[:2], None)
    raw["state"] = raw["Location_State"].astype(str)

    out = raw.rename(columns={"Location_Identifier": "Location_Identifier"})[
        ["Location_Identifier", "lat", "lon", "date", "val", "huc2", "state"]].copy()
    out = out.rename(columns={"lat": "Location_Latitude", "lon": "Location_Longitude"})
    # dedup at (site,date,val)
    out = out.drop_duplicates(subset=["Location_Identifier", "date", "val"]).reset_index(drop=True)
    out.to_parquet(OUT_DIR / "mc_expand_raw.parquet", index=False)
    log(f"WROTE mc_expand_raw.parquet: {len(out)} rows, "
        f"{out.Location_Identifier.nunique()} sites, "
        f"{out.huc2.dropna().nunique()} HUC2 regions")
    print(out.groupby("huc2").agg(rows=("val", "size"),
          sites=("Location_Identifier", "nunique"),
          pos16=("val", lambda v: int((v > 1.6).sum()))).sort_values("rows", ascending=False))


if __name__ == "__main__":
    main()
