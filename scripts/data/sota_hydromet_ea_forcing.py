#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""ERA5-Land daily forcings for EA England stations via Open-Meteo (``hydromet_ea_protocol.md`` §3, §7).

    python -B scripts/data/sota_hydromet_ea_forcing.py plan      # cells, windows, weighted-call cost (no request)
    python -B scripts/data/sota_hydromet_ea_forcing.py pull      # budget-capped pull (resume-safe cache)
    python -B scripts/data/sota_hydromet_ea_forcing.py combine   # harmonised table + manifest

* One request location per ERA5-Land 0.1° grid point (``openmeteo.cell_id``; the box centre is the grid point),
  ``models=era5_seamless``, ``cell_selection=land``, GMT days, the 9 daily variables of
  ``openmeteo.ERA5_DAILY_VARS`` harmonised to the M6 forcing channels by ``openmeteo.era5_to_harmonised``.
* Windows per cell: union over its eligible stations of [first day − 365, last day] for the EA-train/val data and, for
  stations open after 2023-11-12 by metadata, [2022-10-01, 2026-09-30] (weather is not reserved; EA sonde values are).
* **Budget:** the shared ledger (``raw/openmeteo/budget_ledger.jsonl``, day cap 8,000) plus this puller's own rolling
  24-h cap of 40 % of it (3,200 weighted calls, tag prefix ``ea:``); the puller sleeps when either is reached.
  Order: EA-test windows first, then the others.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time

import numpy as np
import pandas as pd

from sentinel.core import paths
from sentinel.data.sources import openmeteo as OM

OUT = paths.processed("ea_wq_sonde")
CACHE = paths.raw("openmeteo_ea") / "cache"
LEDGER = paths.RAW / "openmeteo" / "budget_ledger.jsonl"
SHARE = 0.40
DAY_CAP = 8000.0
MY_CAP = SHARE * DAY_CAP
TAG = "ea:"
TEST_WIN = ("2022-10-01", "2026-09-30")
log = logging.getLogger("ea_forcing")


def _my_usage(now=None) -> list[tuple[float, float]]:
    now = now or time.time()
    out = []
    if LEDGER.exists():
        for ln in LEDGER.read_text().splitlines():
            try:
                r = json.loads(ln)
            except ValueError:
                continue
            if str(r.get("tag", "")).startswith(TAG) and now - float(r["t"]) < 86400:
                out.append((float(r["t"]), float(r["w"])))
    return out


def _wait_my_cap(w: float) -> None:
    while True:
        recs = sorted(_my_usage())
        used = sum(x for _, x in recs)
        if used + w <= MY_CAP:
            return
        need, t_free = used + w - MY_CAP, None
        acc = 0.0
        for t, x in recs:
            acc += x
            if acc >= need:
                t_free = t + 86400
                break
        sl = max(60.0, (t_free or time.time() + 600) - time.time() + 5)
        log.info("own 40%% cap: %.0f used + %.1f > %.0f; sleeping %.0f min", used, w, MY_CAP, sl / 60)
        time.sleep(min(sl, 3600))


def plan() -> pd.DataFrame:
    """Rows: cell, lat, lon, start, end, priority (0 = EA-test window). Metadata-eligible stations only (snapped ≤ 1 km,
    catalogue open span ≥ 60 days in 2012-01-01 … 2026-09-30); windows from catalogue dates only (no EA value read)."""
    geo = pd.read_parquet(OUT / "stations_geo.parquet").set_index("station")
    st = pd.read_parquet(OUT / "stations.parquet").set_index("station").reindex(geo.index)
    opened = pd.to_datetime(st["date_opened"], errors="coerce").clip(lower=pd.Timestamp("2012-01-01"))
    closed = pd.to_datetime(st["date_closed"], errors="coerce").fillna(pd.Timestamp(TEST_WIN[1])).clip(
        upper=pd.Timestamp(TEST_WIN[1]))
    span = (closed - opened).dt.days
    test_days = (closed - opened.clip(lower=pd.Timestamp("2023-11-12"))).dt.days
    elig = (geo["snap_m"].fillna(np.inf) <= 1000) & (span >= 60)
    rows = []
    for s in geo.index[elig]:
        lat, lon = float(geo.at[s, "lat"]), float(geo.at[s, "lon"])
        cid = OM.cell_id(lat, lon)
        a, b = (int(x) for x in cid.split("_"))
        clat, clon = a / 10.0, b / 10.0
        pre_end = min(closed[s], pd.Timestamp("2023-09-30"))
        if pre_end - opened[s] >= pd.Timedelta(days=60):
            rows.append({"cell": cid, "lat": clat, "lon": clon, "start": opened[s] - pd.Timedelta(days=365),
                         "end": pre_end, "priority": 1, "station": s})
        if test_days[s] >= 60:
            rows.append({"cell": cid, "lat": clat, "lon": clon, "start": pd.Timestamp(TEST_WIN[0]),
                         "end": pd.Timestamp(TEST_WIN[1]), "priority": 0, "station": s})
    return pd.DataFrame(rows)


def requests_for(p: pd.DataFrame) -> pd.DataFrame:
    """Calendar-year request chunks per cell over the union of its windows; priority = min over covering windows."""
    out = []
    for cid, g in p.groupby("cell"):
        days = {}
        for r in g.itertuples(index=False):
            for d in pd.date_range(r.start, r.end, freq="D"):
                days[d] = min(days.get(d, 9), r.priority)
        s = pd.Series(days).sort_index()
        for (yr, pr), dd in s.groupby([s.index.year, s.values]):
            idx = dd.index
            # contiguous runs inside the year/priority
            br = np.r_[0, np.flatnonzero(np.diff(idx.values).astype("timedelta64[D]").astype(int) > 1) + 1, len(idx)]
            for i, j in zip(br[:-1], br[1:]):
                a, b = idx[i], idx[j - 1]
                out.append({"cell": cid, "lat": g["lat"].iloc[0], "lon": g["lon"].iloc[0], "start": a.strftime("%Y-%m-%d"),
                            "end": b.strftime("%Y-%m-%d"), "priority": int(pr),
                            "weight": OM.call_weight(len(OM.ERA5_DAILY_VARS), (b - a).days + 1)})
    return pd.DataFrame(out).sort_values(["priority", "cell", "start"]).reset_index(drop=True)


def cmd_plan(_a) -> int:
    rq = requests_for(plan())
    rq.to_parquet(OUT / "forcing_requests.parquet", index=False)
    s = rq.groupby("priority")["weight"].agg(["sum", "size"])
    print(json.dumps({"cells": int(rq["cell"].nunique()), "requests": int(len(rq)), "weight_total": round(float(rq["weight"].sum())),
                      "by_priority": {int(k): {"weight": round(float(v["sum"])), "requests": int(v["size"])} for k, v in s.iterrows()},
                      "days_at_40pct": round(float(rq["weight"].sum()) / MY_CAP, 1)}, indent=1))
    return 0


def cmd_pull(_a) -> int:
    rq = pd.read_parquet(OUT / "forcing_requests.parquet")
    led = OM.BudgetLedger(LEDGER, day_cap=DAY_CAP)
    cli = OM.OpenMeteoClient(CACHE, led)
    for k, r in enumerate(rq.itertuples(index=False)):
        url, params = OM.era5_daily_request(r.lat, r.lon, r.start, r.end, 0)
        if cli.cached(url, params) is not None:
            continue
        _wait_my_cap(r.weight)
        while True:
            try:
                cli.get(url, params, tag=f"{TAG}{r.cell}:{r.start}")
                break
            except OM.DailyLimitReached as e:
                sl = led.seconds_until_day_room(r.weight) + 30
                log.info("shared ledger full (%s); sleeping %.0f min", e, sl / 60)
                time.sleep(min(max(sl, 300), 3600))
        if k % 50 == 0:
            log.info("request %d/%d (priority %d) %s; stats %s; own 24h use %.0f", k, len(rq), r.priority, r.cell, cli.stats,
                     sum(x for _, x in _my_usage()))
    log.info("pull complete: %s", cli.stats)
    return 0


def cmd_combine(_a) -> int:
    from sentinel.core.manifest import write_manifest

    rq = pd.read_parquet(OUT / "forcing_requests.parquet")
    cli = OM.OpenMeteoClient(CACHE, OM.BudgetLedger(LEDGER, day_cap=DAY_CAP))
    parts, missing = [], 0
    for r in rq.itertuples(index=False):
        url, params = OM.era5_daily_request(r.lat, r.lon, r.start, r.end, 0)
        rec = cli.cached(url, params)
        if rec is None:
            missing += 1
            continue
        df, meta = OM.parse_daily(rec["body"])
        h = OM.era5_to_harmonised(df, meta.get("elevation"))
        h.insert(0, "cell", r.cell)
        h["era5_lat"], h["era5_lon"], h["era5_elev_m"] = meta.get("latitude"), meta.get("longitude"), meta.get("elevation")
        parts.append(h)
    f = pd.concat(parts, ignore_index=True).drop_duplicates(["cell", "date"]).sort_values(["cell", "date"])
    f.to_parquet(OUT / "forcings_era5land.parquet", index=False)
    write_manifest("ea_wq_sonde_forcing", files=[OUT / "forcings_era5land.parquet", OUT / "forcing_requests.parquet"],
                   sources=[OM.ARCHIVE_URL + " (models=era5_seamless, cell_selection=land, GMT)"],
                   description="ERA5-Land daily forcings at the ERA5-Land grid points of eligible EA stations, harmonised "
                               "to the M6 forcing channels.", license="ERA5-Land (Copernicus licence) via Open-Meteo (CC BY 4.0)",
                   synthetic=False, stats={"cells": int(f["cell"].nunique()), "rows": int(len(f)), "missing_requests": missing})
    log.info("combined %d rows, %d cells, %d requests missing", len(f), f["cell"].nunique(), missing)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["plan", "pull", "combine"])
    a = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(CACHE.parent / "pull.log")])
    return {"plan": cmd_plan, "pull": cmd_pull, "combine": cmd_combine}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
