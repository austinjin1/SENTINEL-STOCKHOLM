#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Daily meteorological forcings (2000-01-01 -> present) for every SENTINEL site.

Subcommands
-----------
gridmet   CONUS sites: bulk yearly gridMET netCDF (10 variables), nearest valid 4-km cell,
          one file at a time (download -> extract -> delete). Per-site Parquet in raw('gridmet')/sites.
era5      Sites outside the gridMET land mask (AK, HI, PR, GU, coastal gaps): Open-Meteo archive API
          (era5_seamless), deduplicated by 0.1-degree box, one request per box-year, under a
          rolling budget ledger (default 8,000 weighted calls / 24 h) -- may run for days; resume-safe.
combine   Harmonised table processed('forcings_daily')/forcings_daily.parquet (+ coverage, dictionary,
          manifest) from whatever per-site files exist.

    python -B scripts/data/download_forcings.py gridmet --start-year 2000
    python -B scripts/data/download_forcings.py era5 --day-cap 8000
    python -B scripts/data/download_forcings.py combine
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths  # noqa: E402
from sentinel.core.manifest import write_manifest  # noqa: E402
from sentinel.data.sources import gridmet as G  # noqa: E402
from sentinel.data.sources import openmeteo as OM  # noqa: E402

log = logging.getLogger("forcings")
HARM = [
    "precip_mm",
    "tmax_c",
    "tmin_c",
    "srad_wm2",
    "wind_ms",
    "rhmax_pct",
    "rhmin_pct",
    "sph_kgkg",
    "vpd_kpa",
    "pet_mm",
]
START = "2000-01-01"


def _setup_logging(d: Path, name: str) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(d / f"{name}.log"), logging.StreamHandler()],
    )


def load_registry() -> pd.DataFrame:
    return pd.read_parquet(paths.PROCESSED / "sites" / "sites.parquet")


# ============================================================================ gridMET
def gridmet(a) -> int:
    raw = paths.raw("gridmet")
    _setup_logging(raw, "gridmet")
    tmp_dir = paths.ensure(raw / "_download")  # untrusted downloads: own directory, deleted after use
    ext_dir = paths.ensure(paths.interim("gridmet") / "extract")
    reg = load_registry()
    years = list(range(a.start_year, (a.end_year or _dt.date.today().year) + 1))
    variables = a.vars.split(",") if a.vars else list(G.VARIABLES)

    # 1) site -> nearest valid cell, from a reference file (tmmx, first year)
    cells_path = raw / "site_cells.parquet"
    if cells_path.exists():
        cells = pd.read_parquet(cells_path)
    else:
        ref = tmp_dir / f"tmmx_{years[0]}.nc"
        G.download(G.file_url("tmmx", years[0]), ref)
        g = G.read_grid(ref)
        valid = ~np.isnan(g["data"][0])
        cells = G.nearest_valid_cells(
            reg["lat"].to_numpy(),
            reg["lon"].to_numpy(),
            g["lat"],
            g["lon"],
            valid,
            max_radius_cells=a.max_radius,
        )
        cells.insert(0, "site_key", reg["site_key"].to_numpy())
        cells["cell_key"] = np.where(
            cells["i"] >= 0, cells["i"].astype(str) + "_" + cells["j"].astype(str), None
        )
        cells.to_parquet(cells_path, index=False)
        del g
        if (ext_dir / f"tmmx_{years[0]}.parquet").exists():
            ref.unlink()
    ok = cells[cells["i"] >= 0]
    ucells = ok.drop_duplicates("cell_key")[["cell_key", "i", "j"]].reset_index(drop=True)
    log.info(
        "gridMET: %d/%d sites mapped to %d unique cells (max offset %.2f km)",
        len(ok),
        len(cells),
        len(ucells),
        ok["cell_dist_km"].max(),
    )

    # 2) variable-years: download -> extract -> delete (2 concurrent downloads)
    needed = set(ucells["cell_key"])
    for f in sorted(ext_dir.glob("*.parquet")):  # an extract made for another site set is redone
        if not needed <= set(pq.read_schema(f).names):
            log.warning(
                "extract %s lacks %d needed cells -> refetch",
                f.name,
                len(needed - set(pq.read_schema(f).names)),
            )
            f.unlink()
    todo = [(v, y) for y in years for v in variables if not (ext_dir / f"{v}_{y}.parquet").exists()]
    log.info("%d variable-years to fetch", len(todo))
    units_seen = {}

    def fetch(job):
        v, y = job
        dest = tmp_dir / f"{v}_{y}.nc"
        if not dest.exists():
            G.download(G.file_url(v, y), dest)
        return job, dest

    with ThreadPoolExecutor(max_workers=2) as ex:
        window = 3  # bounded prefetch: at most 3 netCDF files (~180 MB) on disk ahead of processing
        futs = [ex.submit(fetch, j) for j in todo[:window]]
        for k in range(len(todo)):
            (v, y), dest = futs[k].result()
            if k + window < len(todo):
                futs.append(ex.submit(fetch, todo[k + window]))
            g = G.read_grid(dest)
            exp_units = G.VARIABLES[v][1]
            if g["units"] != exp_units:
                raise ValueError(f"{v}_{y}: units {g['units']!r} != expected {exp_units!r}")
            units_seen[v] = g["units"]
            vals = G.extract_cells(g, ucells)
            df = pd.DataFrame(vals, columns=ucells["cell_key"].tolist())
            df.insert(0, "date", g["dates"].date)
            df.insert(1, "gridmet_status", g["status"])
            df.to_parquet(ext_dir / f"{v}_{y}.parquet", index=False, compression="zstd")
            dest.unlink()
            log.info(
                "[%d/%d] %s_%d: %d days, statuses=%s",
                k + 1,
                len(todo),
                v,
                y,
                len(df),
                pd.Series(g["status"]).value_counts().to_dict(),
            )
            del g, vals, df

    # 3) assemble per-site Parquet (file units, original short names)
    site_dir = paths.ensure(raw / "sites")
    per_var, per_status = {}, {}
    for v in variables:
        parts = [
            pd.read_parquet(ext_dir / f"{v}_{y}.parquet")
            for y in years
            if (ext_dir / f"{v}_{y}.parquet").exists()
        ]
        dfv = pd.concat(parts, ignore_index=True).drop_duplicates("date", keep="last").set_index("date")
        per_status[v] = dfv["gridmet_status"]
        per_var[v] = dfv.drop(columns="gridmet_status")
    dates = sorted(set().union(*[set(d.index) for d in per_var.values()]))
    for v in per_var:
        per_var[v] = per_var[v].reindex(dates)
    # most provisional status across variables (early > provisional > permanent; missing -> 'missing')
    sev = {"permanent": 0, "provisional": 1, "early": 2}
    codes = pd.concat([per_status[v].reindex(dates).map(sev) for v in variables], axis=1)
    status = codes.max(axis=1).map({0: "permanent", 1: "provisional", 2: "early"}).fillna("missing")
    n_written = 0
    for r in ok.itertuples():
        out = pd.DataFrame({"date": dates})
        for v in variables:
            out[v] = per_var[v][r.cell_key].to_numpy(dtype="float32")
        out["gridmet_status"] = status.to_numpy()
        out.to_parquet(site_dir / f"{r.site_key}.parquet", index=False, compression="zstd")
        n_written += 1
    stats = {
        "years": [years[0], years[-1]],
        "variables": variables,
        "n_sites": n_written,
        "n_unique_cells": int(len(ucells)),
        "n_sites_unmapped": int((cells["i"] < 0).sum()),
        "max_cell_offset_km": float(ok["cell_dist_km"].max()),
        "n_sites_offset_gt_3km": int((ok["cell_dist_km"] > 3.0).sum()),
        "first_date": str(dates[0]),
        "last_date": str(dates[-1]),
        "status_counts_last_year": status[status.index >= _dt.date(years[-1], 1, 1)].value_counts().to_dict(),
        "units_in_files": units_seen or {v: G.VARIABLES[v][1] for v in variables},
        "method": "bulk yearly netCDF from northwestknowledge.net/metdata/data; nearest valid cell "
        f"(search radius {a.max_radius} cells); packing applied by netCDF4",
    }
    (raw / "gridmet_stats.json").write_text(json.dumps(stats, indent=2, default=str))
    write_manifest(
        "gridmet_sites",
        files=sorted(site_dir.glob("*.parquet")) + [cells_path],
        sources=[G.BULK_URL],
        description="gridMET daily forcings at the nearest valid 4-km cell "
        "of each CONUS SENTINEL site (file units: K, W m-2, m/s, %, kg/kg, kPa, mm).",
        license="gridMET: public domain / CC0 (Climatology Lab, Abatzoglou 2013)",
        stats=stats,
        hash_files=False,
    )
    log.info("gridMET done: %s", stats)
    return 0


# ============================================================================ ERA5 via Open-Meteo
def era5_plan(reg: pd.DataFrame, cells: pd.DataFrame | None, end: str) -> pd.DataFrame:
    """Sites not covered by gridMET -> (box, tz) groups with a priority-ordered year list."""
    if cells is not None:
        unmapped = set(cells.loc[cells["i"] < 0, "site_key"])
    else:
        unmapped = set(reg.loc[~reg["in_conus"], "site_key"])
    s = reg[reg["site_key"].isin(unmapped)].copy()
    s["box"] = [OM.cell_id(a, b) for a, b in zip(s["lat"], s["lon"])]
    s["group"] = s["box"] + "_tz" + s["utc_offset_h"].astype(int).astype(str)
    return s


def era5(a) -> int:
    raw = paths.raw("openmeteo_era5")
    _setup_logging(raw, "era5")
    reg = load_registry()
    cells_path = paths.RAW / "gridmet" / "site_cells.parquet"
    cells = pd.read_parquet(cells_path) if cells_path.exists() else None
    if cells is None and not a.allow_bbox:
        raise SystemExit(
            "run the gridmet step first (needs raw/gridmet/site_cells.parquet) or pass --allow-bbox"
        )
    end = a.end or (_dt.date.today() - _dt.timedelta(days=7)).isoformat()
    s = era5_plan(reg, cells, end)
    s.to_parquet(raw / "era5_sites.parquet", index=False)
    ledger = OM.BudgetLedger(
        paths.RAW / "openmeteo" / "budget_ledger.jsonl",
        day_cap=a.day_cap,
        hour_cap=min(a.day_cap, 4000),
        minute_cap=400,
    )
    client = OM.OpenMeteoClient(raw / "cache", ledger)
    # one representative point per group (first site), chunks = calendar years
    groups = s.sort_values("site_key").groupby("group")
    jobs = []
    for gname, gs in groups:
        rep = gs.iloc[0]
        wq_b = pd.to_datetime(gs["wq_begin"]).min()
        wq_e = pd.to_datetime(gs["wq_end"]).max()
        for c0, c1 in OM.year_chunks(START, end):
            y = int(c0[:4])
            active = (pd.notna(wq_b) and pd.notna(wq_e) and (wq_b.year - 1) <= y <= wq_e.year) or pd.isna(
                wq_b
            )
            in_bench = y >= 2007
            prio = (0 if (active and in_bench) else 1 if in_bench else 2 if active else 3, -y)
            jobs.append(
                {
                    "group": gname,
                    "lat": float(rep["lat"]),
                    "lon": float(rep["lon"]),
                    "tz": int(rep["utc_offset_h"]),
                    "start": c0,
                    "end": c1,
                    "prio": prio,
                }
            )
    jobs = [j for j in jobs if j["prio"][0] <= a.max_prio]
    jobs.sort(key=lambda j: (j["prio"], j["group"]))
    tot_w = sum(OM.call_weight(len(OM.ERA5_DAILY_VARS), OM.n_days(j["start"], j["end"])) for j in jobs)
    log.info(
        "ERA5 plan: %d sites, %d groups, %d box-year requests, total weight %.0f calls "
        "(~%.1f days at %d/day)",
        len(s),
        s["group"].nunique(),
        len(jobs),
        tot_w,
        tot_w / a.day_cap,
        a.day_cap,
    )
    cdir = paths.ensure(raw / "groups")
    site_dir = paths.ensure(raw / "sites")

    def write_sites() -> int:
        """(Re)write per-site Parquet from whatever box-years exist (called before budget sleeps)."""
        n = 0
        for r in s.itertuples():
            parts = sorted((cdir / r.group).glob("*.parquet"))
            if not parts:
                continue
            df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True).sort_values("date")
            tmp = site_dir / f".{r.site_key}.parquet.tmp"
            df.to_parquet(tmp, index=False, compression="zstd")
            tmp.replace(site_dir / f"{r.site_key}.parquet")  # atomic: combine may read concurrently
            n += 1
        return n

    done = 0
    for k, j in enumerate(jobs):
        out = cdir / j["group"] / f"{j['start'][:4]}.parquet"
        if out.exists():
            done += 1
            continue
        url, params = OM.era5_daily_request(j["lat"], j["lon"], j["start"], j["end"], j["tz"])
        while True:
            try:
                rec = client.get(url, params, tag=f"era5:{j['group']}:{j['start'][:4]}")
                break
            except OM.DailyLimitReached as e:
                wait = max(600.0, ledger.seconds_until_day_room(OM.request_weight(url, params)))
                log.warning(
                    "budget: %s -> sleeping %.0f min (done %d/%d); per-site files refreshed: %d",
                    e,
                    wait / 60,
                    done,
                    len(jobs),
                    write_sites(),
                )
                if a.no_wait:
                    return 0
                time.sleep(min(wait, 3 * 3600))
        df, meta = OM.parse_daily(rec["body"])
        h = OM.era5_to_harmonised(df, meta.get("elevation"))
        h["om_lat"], h["om_lon"], h["om_elev_m"] = (
            meta.get("latitude"),
            meta.get("longitude"),
            meta.get("elevation"),
        )
        h["om_utc_offset_s"] = meta.get("utc_offset_seconds")
        for c in OM.ERA5_DAILY_VARS:  # keep the source columns (Open-Meteo units) too
            h["om_" + c] = df[c].astype("float32")
        out.parent.mkdir(parents=True, exist_ok=True)
        h.to_parquet(out, index=False, compression="zstd")
        done += 1
        if done % 25 == 0:
            log.info(
                "ERA5 %d/%d box-years; client=%s; ledger=%s", done, len(jobs), client.stats, ledger.used()
            )
    write_sites()
    stats = {
        "n_sites": int(len(s)),
        "n_groups": int(s["group"].nunique()),
        "n_jobs": len(jobs),
        "n_done": done,
        "total_weight_calls": round(tot_w, 1),
        "client": client.stats,
        "model": "era5_seamless",
        "end": end,
        "variables": OM.ERA5_DAILY_VARS,
    }
    (raw / "era5_stats.json").write_text(json.dumps(stats, indent=2, default=str))
    log.info("ERA5 finished: %s", stats)
    return 0


# ============================================================================ combine
DICTIONARY = [
    ("site_key", "", "SENTINEL site key"),
    (
        "date",
        "",
        "day: gridMET = MST calendar day (ends 07 UTC next day; file note5); ERA5 = local standard-time day",
    ),
    ("precip_mm", "mm/day", "gridMET pr | ERA5 precipitation_sum"),
    ("tmax_c", "degC", "gridMET tmmx-273.15 | ERA5 temperature_2m_max"),
    ("tmin_c", "degC", "gridMET tmmn-273.15 | ERA5 temperature_2m_min"),
    ("srad_wm2", "W/m2 (daily mean)", "gridMET srad | ERA5 shortwave_radiation_sum (MJ/m2) x 1e6/86400"),
    ("wind_ms", "m/s (10 m daily mean)", "gridMET vs | ERA5 wind_speed_10m_mean"),
    ("rhmax_pct", "%", "gridMET rmax | ERA5 relative_humidity_2m_max"),
    ("rhmin_pct", "%", "gridMET rmin | ERA5 relative_humidity_2m_min"),
    (
        "sph_kgkg",
        "kg/kg",
        "gridMET sph | ERA5 derived: 0.622e/(p-0.378e), e=e0(dew point), p=FAO-56 eq.7(elev)",
    ),
    ("vpd_kpa", "kPa (daily mean)", "gridMET vpd | ERA5 derived FAO-56: (e0(Tmax)+e0(Tmin))/2 - e0(Tdew)"),
    ("pet_mm", "mm/day", "gridMET pet (ASCE grass reference ET) | ERA5 et0_fao_evapotranspiration (FAO-56)"),
    ("source", "", "gridmet | era5_seamless (Open-Meteo); never mixed within a site"),
    ("status", "", "gridMET permanent/provisional/early; ERA5 'era5' (ERA5T near real time may be revised)"),
]


def combine(a) -> int:
    out_dir = paths.processed("forcings_daily")
    _setup_logging(out_dir, "combine")
    reg = load_registry()
    gdir, edir = paths.RAW / "gridmet" / "sites", paths.RAW / "openmeteo_era5" / "sites"
    out = out_dir / "forcings_daily.parquet"
    tmp = out.with_suffix(".tmp.parquet")
    schema = pa.schema(
        [("site_key", pa.string()), ("date", pa.date32())]
        + [(c, pa.float32()) for c in HARM]
        + [("source", pa.string()), ("status", pa.string())]
    )
    writer = pq.ParquetWriter(tmp, schema, compression="zstd")
    cov = []
    qc = {
        "precip_lt0": 0,
        "tmax_lt_tmin": 0,
        "rh_gt_100.5": 0,
        "rhmin_gt_rhmax": 0,
        "srad_lt0": 0,
        "wind_lt0": 0,
        "pet_lt0": 0,
        "tmax_gt_60c": 0,
        "tmin_lt_-70c": 0,
    }
    for key in reg["site_key"]:
        gp, ep = gdir / f"{key}.parquet", edir / f"{key}.parquet"
        if gp.exists():
            g = pd.read_parquet(gp)
            df = pd.DataFrame({"site_key": key, "date": pd.to_datetime(g["date"]).dt.date})
            for v, (_, _, col, _) in G.VARIABLES.items():
                df[col] = G.to_harmonised(v, g[v].to_numpy(dtype="float32"))
            df["source"], df["status"] = "gridmet", g["gridmet_status"].astype(str)
        elif ep.exists():
            e = pd.read_parquet(ep)
            df = pd.DataFrame({"site_key": key, "date": pd.to_datetime(e["date"]).dt.date})
            for c in HARM:
                df[c] = e[c].astype("float32")
            df["source"], df["status"] = "era5_seamless", "era5"
        else:
            cov.append(
                {
                    "site_key": key,
                    "source": None,
                    "first_date": None,
                    "last_date": None,
                    "n_days": 0,
                    "frac_complete_precip": 0.0,
                    "frac_complete_tmax": 0.0,
                }
            )
            continue
        df = df[df["date"] >= pd.Timestamp(START).date()].sort_values("date")
        # plausibility screen: counted (not altered); see dictionary / datasheet
        qc["precip_lt0"] += int((df["precip_mm"] < 0).sum())
        qc["tmax_lt_tmin"] += int((df["tmax_c"] < df["tmin_c"]).sum())
        qc["rh_gt_100.5"] += int(((df["rhmax_pct"] > 100.5) | (df["rhmin_pct"] > 100.5)).sum())
        qc["rhmin_gt_rhmax"] += int((df["rhmin_pct"] > df["rhmax_pct"]).sum())
        qc["srad_lt0"] += int((df["srad_wm2"] < 0).sum())
        qc["wind_lt0"] += int((df["wind_ms"] < 0).sum())
        qc["pet_lt0"] += int((df["pet_mm"] < 0).sum())
        qc["tmax_gt_60c"] += int((df["tmax_c"] > 60).sum())
        qc["tmin_lt_-70c"] += int((df["tmin_c"] < -70).sum())
        writer.write_table(pa.Table.from_pandas(df[schema.names], schema=schema, preserve_index=False))
        span = (pd.Timestamp(df["date"].max()) - pd.Timestamp(START)).days + 1
        cov.append(
            {
                "site_key": key,
                "source": df["source"].iloc[0],
                "first_date": df["date"].min(),
                "last_date": df["date"].max(),
                "n_days": int(len(df)),
                "frac_complete_precip": float(df["precip_mm"].notna().sum() / max(span, 1)),
                "frac_complete_tmax": float(df["tmax_c"].notna().sum() / max(span, 1)),
            }
        )
    writer.close()
    tmp.replace(out)
    cov = pd.DataFrame(cov)
    cov.to_parquet(out_dir / "coverage.parquet", index=False)
    pd.DataFrame(DICTIONARY, columns=["column", "unit", "definition"]).to_csv(
        out_dir / "dictionary.csv", index=False
    )
    stats = {
        "n_sites_total": int(len(reg)),
        "n_sites_with_forcings": int((cov["n_days"] > 0).sum()),
        "by_source": cov["source"].value_counts(dropna=False).astype(int).to_dict(),
        "rows": int(pq.ParquetFile(out).metadata.num_rows),
        "start": START,
        "last_date_max": str(cov["last_date"].dropna().max()) if cov["last_date"].notna().any() else None,
        "median_frac_complete_precip": float(cov.loc[cov["n_days"] > 0, "frac_complete_precip"].median())
        if (cov["n_days"] > 0).any()
        else None,
        "plausibility_counts": qc,
    }
    write_manifest(
        "forcings_daily",
        files=[out, out_dir / "coverage.parquet", out_dir / "dictionary.csv"],
        sources=[G.BULK_URL, OM.ARCHIVE_URL + "?models=era5_seamless"],
        description="Daily meteorological forcings per SENTINEL site, harmonised units "
        "(gridMET for CONUS; Open-Meteo ERA5/ERA5-Land elsewhere). OBSERVED weather: "
        "oracle if used for future days.",
        license="gridMET public domain/CC0; Open-Meteo/ERA5 CC BY 4.0 (Copernicus)",
        stats=stats,
    )
    log.info("combine done: %s", stats)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gridmet")
    g.add_argument("--start-year", type=int, default=2000)
    g.add_argument("--end-year", type=int, default=None)
    g.add_argument("--vars", default="")
    g.add_argument("--max-radius", type=int, default=3)
    e = sub.add_parser("era5")
    e.add_argument("--day-cap", type=float, default=8000)
    e.add_argument("--end", default=None)
    e.add_argument("--no-wait", action="store_true", help="exit instead of sleeping when the budget is used")
    e.add_argument("--allow-bbox", action="store_true")
    e.add_argument(
        "--max-prio",
        type=int,
        default=3,
        help="0: benchmark years (>=2007) overlapping site WQ data; 1: other years >=2007; "
        "2: pre-2007 overlapping WQ data; 3: everything",
    )
    sub.add_parser("combine")
    a = ap.parse_args()
    return {"gridmet": gridmet, "era5": era5, "combine": combine}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
