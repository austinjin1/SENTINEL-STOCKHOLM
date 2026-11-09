#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Archived operational NWP forecasts (Open-Meteo Previous Runs API) per site, lead days 1-7.

For 'Operational' forecast mode only -- never mixed with observed (oracle) weather. Hourly
``temperature_2m``, ``precipitation`` and ``shortwave_radiation`` at ``_previous_day1..7`` are
fetched per site and model in calendar-year chunks (cached, budget-ledger paced), then aggregated
to local-standard-time days with an upper bound ``max_init_utc`` on the initialisation time of
every contributing model run.

Sites: ``--sites-file`` (CSV/Parquet with ``site_key``; the later core list) or the default pilot of
50 CONUS USGS stream sites with DO + SC + temperature DV series (>= 5 y overlap, active in 2026),
stratified round-robin over HUC2 regions (deterministic).

Steps: ``probe`` (find archive start per model/variable, ~1 call per probe), ``semantics`` (check
that previous_day1 equals the run initialised <= valid-24 h using the Single Runs API on recent
runs), ``download``, ``aggregate``.

    python -B scripts/data/download_nwp_archive.py all --pilot 50 --models gfs_global,ecmwf_ifs025
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths  # noqa: E402
from sentinel.core.manifest import write_manifest  # noqa: E402
from sentinel.data.sources import openmeteo as OM  # noqa: E402

log = logging.getLogger("nwp")
SINGLE_RUNS_URL = "https://single-runs-api.open-meteo.com/v1/forecast"
PROBE_LAT, PROBE_LON = 39.63, -90.61  # Illinois River at Florence, IL (USGS-05586300)


def select_pilot(n: int, catalog: Path, reg: pd.DataFrame) -> pd.DataFrame:
    cat = pd.read_parquet(catalog)
    c = cat[(cat["data_type_cd"] == "dv") & cat["parm_cd"].isin(["00010", "00300", "00095"])].copy()
    c["b"] = pd.to_datetime(c["begin_date"], errors="coerce")
    c["e"] = pd.to_datetime(c["end_date"], errors="coerce")
    per = c.groupby(["site_no", "parm_cd"]).agg(b=("b", "min"), e=("e", "max")).reset_index()
    w = per.pivot(index="site_no", columns="parm_cd", values=["b", "e"])
    w = w.dropna()
    start = w["b"].max(axis=1)
    end = w["e"].min(axis=1)
    ov = (end - start).dt.days / 365.25
    cand = pd.DataFrame(
        {"site_key": "USGS-" + w.index.to_series(), "overlap_years": ov, "end": end}
    ).reset_index(drop=True)
    r = reg.set_index("site_key")
    cand = cand[cand["site_key"].isin(r.index)]
    cand["site_type"] = cand["site_key"].map(r["site_type"])
    cand["in_conus"] = cand["site_key"].map(r["in_conus"])
    cand["huc2"] = cand["site_key"].map(r["huc2"])
    cand = cand[
        (cand["site_type"] == "ST")
        & cand["in_conus"]
        & (cand["overlap_years"] >= 5)
        & (cand["end"] >= pd.Timestamp("2026-01-01"))
    ]
    cand = cand.sort_values(["huc2", "overlap_years", "site_key"], ascending=[True, False, True])
    groups = {h: list(g["site_key"]) for h, g in cand.groupby("huc2")}
    picked = []
    while len(picked) < n and any(groups.values()):
        for h in sorted(groups):
            if groups[h] and len(picked) < n:
                picked.append(groups[h].pop(0))
    out = cand.set_index("site_key").loc[picked].reset_index()
    out["n_candidates_total"] = len(cand)
    return out


def _probe_day(client: OM.OpenMeteoClient, model: str, day: str) -> dict:
    url, params = OM.previous_runs_request(PROBE_LAT, PROBE_LON, day, day, model, leads=(1, 7))
    body = client.get(url, params, tag=f"probe:{model}")["body"]
    h = body.get("hourly") or {}
    return {k: sum(v is not None for v in vals) for k, vals in h.items() if k != "time"}


def probe(client: OM.OpenMeteoClient, models: list[str], out: Path) -> dict:
    """First day with >= 20 h of data per model x variable (monthly scan, then daily bisection)."""
    res = {}
    if out.exists():
        res = json.loads(out.read_text())
    for m in models:
        if m in res:
            continue
        months = pd.date_range("2021-01-15", "2025-06-15", freq="MS") + pd.Timedelta(days=14)
        first_month = {}
        for d in months:
            got = _probe_day(client, m, d.strftime("%Y-%m-%d"))
            for k, n in got.items():
                if n >= 20 and k not in first_month:
                    first_month[k] = d
            if len(first_month) == 6:
                break
        starts = {}
        for k, d in first_month.items():
            lo, hi = d - pd.Timedelta(days=31), d  # data absent at lo (previous probe), present at hi
            while (hi - lo).days > 1:
                mid = lo + (hi - lo) / 2
                mid = mid.normalize()
                n = _probe_day(client, m, mid.strftime("%Y-%m-%d")).get(k, 0)
                if n >= 20:
                    hi = mid
                else:
                    lo = mid
            starts[k] = hi.strftime("%Y-%m-%d")
        res[m] = starts
        log.info("availability %s: %s", m, starts)
        out.write_text(json.dumps(res, indent=2))
    return res


def check_semantics(client: OM.OpenMeteoClient, model: str, out: Path) -> dict:
    """Compare previous_day1 with single runs initialised 0/6/12/18 h before (valid - 24 h)."""
    if out.exists():
        res = json.loads(out.read_text())
        if model in res:
            return res[model]
    today = pd.Timestamp(_dt.date.today())
    vday = (today - pd.Timedelta(days=4)).strftime("%Y-%m-%d")
    url, params = OM.previous_runs_request(
        PROBE_LAT, PROBE_LON, vday, vday, model, variables=["temperature_2m"], leads=(1,)
    )
    pr, _ = OM.parse_hourly(client.get(url, params, tag="semantics")["body"])
    pr = pr.set_index("time_utc")["temperature_2m_previous_day1"]
    found = {}
    for hh in (0, 6, 12, 18):
        v = pd.Timestamp(f"{vday}T{hh:02d}:00", tz="UTC")
        cands = {}
        for back in (24, 30, 36, 18):
            run = (v - pd.Timedelta(hours=back)).strftime("%Y-%m-%dT%H:%M")
            p = {
                "latitude": f"{PROBE_LAT:.4f}",
                "longitude": f"{PROBE_LON:.4f}",
                "run": run,
                "hourly": "temperature_2m",
                "models": model,
                "forecast_days": 3,
                "timezone": "GMT",
            }
            # weight of a single-run request: 1 variable, 3 days -> 1 call
            client.ledger.reserve(1.0, "semantics")
            try:
                r = client.s.get(SINGLE_RUNS_URL, params=p, timeout=120)
                body = r.json()
            except Exception as e:  # noqa: BLE001
                cands[back] = f"error {e!r}"
                continue
            if "hourly" not in body:
                cands[back] = body.get("reason", "n/a")
                continue
            sr, _ = OM.parse_hourly(body)
            sr = sr.set_index("time_utc")["temperature_2m"]
            cands[back] = None if v not in sr.index else float(sr.loc[v])
            time.sleep(1.0)
        found[f"{hh:02d}Z"] = {
            "previous_day1": None if v not in pr.index else float(pr.loc[v]),
            "single_runs_by_hours_before": cands,
        }
    match = []
    for hh, d in found.items():
        pv = d["previous_day1"]
        hit = [
            b
            for b, val in d["single_runs_by_hours_before"].items()
            if isinstance(val, float) and pv is not None and abs(val - pv) < 0.051
        ]
        match.append({"valid": hh, "matching_hours_before": hit})
    res = {
        "valid_day": vday,
        "detail": found,
        "matches": match,
        "interpretation": "previous_day1 at valid hour v equals the run initialised at v-24h (lead 24h) "
        "if matching_hours_before == [24]",
    }
    allres = json.loads(out.read_text()) if out.exists() else {}
    allres[model] = res
    out.write_text(json.dumps(allres, indent=2))
    log.info("semantics %s: %s", model, match)
    return res


def plan_requests(
    sites: pd.DataFrame, reg: pd.DataFrame, models: list[str], avail: dict, end: str
) -> list[dict]:
    r = reg.set_index("site_key")
    jobs = []
    for m in models:
        a = avail.get(m, {})
        # variable groups sharing a start date
        groups: dict[str, list[str]] = {}
        for v in OM.NWP_HOURLY_VARS:
            starts = [a.get(f"{v}_previous_day{n}") for n in (1, 7) if a.get(f"{v}_previous_day{n}")]
            if not starts:
                continue
            groups.setdefault(min(starts), []).append(v)
        for key in sites["site_key"]:
            lat, lon = float(r.at[key, "lat"]), float(r.at[key, "lon"])
            for s0, vars_ in sorted(groups.items()):
                for c0, c1 in OM.year_chunks(s0, end):
                    jobs.append(
                        {
                            "site_key": key,
                            "model": m,
                            "vars": vars_,
                            "start": c0,
                            "end": c1,
                            "lat": lat,
                            "lon": lon,
                        }
                    )
    return jobs


def download(client: OM.OpenMeteoClient, jobs: list[dict], raw: Path, no_wait: bool) -> dict:
    hdir = paths.ensure(raw / "hourly")
    t0, n_new = time.time(), 0
    for k, j in enumerate(jobs):
        out = hdir / j["model"] / j["site_key"] / f"{'-'.join(j['vars'])}_{j['start'][:4]}.parquet"
        if out.exists():
            continue
        url, params = OM.previous_runs_request(
            j["lat"], j["lon"], j["start"], j["end"], j["model"], variables=j["vars"]
        )
        while True:
            try:
                rec = client.get(url, params, tag=f"nwp:{j['model']}:{j['site_key']}")
                break
            except OM.DailyLimitReached as e:
                wait = max(600.0, client.ledger.seconds_until_day_room(OM.request_weight(url, params)))
                log.warning("budget: %s -> sleep %.0f min (job %d/%d)", e, wait / 60, k, len(jobs))
                if no_wait:
                    return {"stopped_at": k, "reason": "daily budget"}
                time.sleep(min(wait, 3 * 3600))
        df, meta = OM.parse_hourly(rec["body"])
        for c in df.columns:
            if c != "time_utc":
                df[c] = df[c].astype("float32")
        df.attrs = {}
        df["om_lat"], df["om_lon"], df["om_elev_m"] = (
            meta.get("latitude"),
            meta.get("longitude"),
            meta.get("elevation"),
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name("." + out.name + ".tmp")
        df.to_parquet(tmp, index=False, compression="zstd")
        tmp.replace(out)  # atomic: aggregate may run while downloading
        n_new += 1
        if n_new % 20 == 0:
            el = time.time() - t0
            log.info(
                "NWP %d/%d chunks (%d new in %.1f min); client=%s; ledger=%s",
                k + 1,
                len(jobs),
                n_new,
                el / 60,
                client.stats,
                client.ledger.used(),
            )
    return {"stopped_at": None, "new_chunks": n_new, "seconds": round(time.time() - t0, 1)}


def aggregate(
    sites: pd.DataFrame, reg: pd.DataFrame, models: list[str], raw: Path, out_dir: Path, tag: str = "pilot"
) -> pd.DataFrame:
    r = reg.set_index("site_key")
    frames, cov = [], []
    for m in models:
        for key in sites["site_key"]:
            d = raw / "hourly" / m / key
            parts = sorted(d.glob("*.parquet")) if d.exists() else []
            if not parts:
                continue
            by_group: dict[str, list[pd.DataFrame]] = {}
            meta = {}
            for p in parts:  # file name: <vars>_<year>.parquet -> stack years, then join groups
                df = pd.read_parquet(p)
                meta = df[["om_lat", "om_lon", "om_elev_m"]].iloc[0].to_dict()
                by_group.setdefault(p.stem.rsplit("_", 1)[0], []).append(
                    df.drop(columns=["om_lat", "om_lon", "om_elev_m"])
                )
            hourly = None
            for gname, dfs in sorted(by_group.items()):
                g = pd.concat(dfs, ignore_index=True).drop_duplicates("time_utc").set_index("time_utc")
                hourly = g if hourly is None else hourly.join(g, how="outer")
            hourly = hourly.sort_index().reset_index()
            for v in OM.NWP_HOURLY_VARS:
                for n in range(1, 8):
                    col = f"{v}_previous_day{n}"
                    if col not in hourly.columns:
                        hourly[col] = np.nan
            off = int(r.at[key, "utc_offset_h"])
            agg = OM.aggregate_previous_runs(hourly, off)
            agg.insert(0, "site_key", key)
            agg.insert(1, "model", m)
            agg["om_lat"], agg["om_lon"], agg["om_elev_m"] = meta["om_lat"], meta["om_lon"], meta["om_elev_m"]
            frames.append(agg)
            for n, g in agg.groupby("lead_day"):
                for v in ("tmax_c", "precip_mm", "srad_wm2"):
                    ok = g[v].notna()
                    cov.append(
                        {
                            "site_key": key,
                            "model": m,
                            "lead_day": int(n),
                            "variable": v,
                            "first_date": g.loc[ok, "date_local"].min() if ok.any() else None,
                            "last_date": g.loc[ok, "date_local"].max() if ok.any() else None,
                            "n_days": int(ok.sum()),
                            "frac_complete_since_first": float(
                                ok[g["date_local"] >= g.loc[ok, "date_local"].min()].mean()
                            )
                            if ok.any()
                            else 0.0,
                        }
                    )
    daily = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    covdf = pd.DataFrame(cov)
    if len(daily):
        daily.to_parquet(out_dir / f"nwp_daily_{tag}.parquet", index=False, compression="zstd")
        covdf.to_parquet(out_dir / f"coverage_{tag}.parquet", index=False)
    return covdf


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("step", choices=["probe", "semantics", "download", "aggregate", "all"])
    ap.add_argument("--pilot", type=int, default=50)
    ap.add_argument("--sites-file", default=None)
    ap.add_argument("--models", default="gfs_global,ecmwf_ifs025")
    ap.add_argument("--end", default=None, help="last valid day (default: today - 2 days)")
    ap.add_argument("--day-cap", type=float, default=8000)
    ap.add_argument("--no-wait", action="store_true")
    a = ap.parse_args()
    raw = paths.raw("openmeteo_nwp")
    out_dir = paths.processed("nwp_archive")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(raw / "download_nwp_archive.log"), logging.StreamHandler()],
    )
    reg = pd.read_parquet(paths.PROCESSED / "sites" / "sites.parquet")
    models = a.models.split(",")
    end = a.end or (_dt.date.today() - _dt.timedelta(days=2)).isoformat()
    if a.sites_file:
        sf = Path(a.sites_file)
        sites = pd.read_parquet(sf) if sf.suffix == ".parquet" else pd.read_csv(sf, dtype=str)
        sites = sites[["site_key"]].drop_duplicates()
        tag = "core"
    else:
        sites = select_pilot(a.pilot, paths.RAW / "usgs_dv" / "dv_series_catalog.parquet", reg)
        tag = "pilot"
    sites.to_csv(raw / f"sites_{tag}.csv", index=False)
    log.info("%d %s sites (%s)", len(sites), tag, ", ".join(sites["site_key"].head(5)))
    ledger = OM.BudgetLedger(
        paths.RAW / "openmeteo" / "budget_ledger.jsonl",
        day_cap=a.day_cap,
        hour_cap=min(4000, a.day_cap),
        minute_cap=400,
    )
    client = OM.OpenMeteoClient(raw / "cache", ledger)
    t_start = time.time()
    if a.step in ("probe", "all", "download"):
        avail = probe(client, models, raw / "availability.json")
    else:
        avail = json.loads((raw / "availability.json").read_text())
    if a.step in ("semantics", "all"):
        for m in models:
            check_semantics(client, m, raw / "lead_semantics.json")
    dl = {}
    jobs = plan_requests(sites, reg, models, avail, end)
    tot_w = sum(OM.call_weight(len(j["vars"]) * 7, OM.n_days(j["start"], j["end"])) for j in jobs)
    log.info(
        "plan: %d chunk requests, total weight %.0f calls (%.0f per site), ~%.1f days at %d/day",
        len(jobs),
        tot_w,
        tot_w / max(len(sites), 1),
        tot_w / a.day_cap,
        a.day_cap,
    )
    if a.step in ("download", "all"):
        dl = download(client, jobs, raw, a.no_wait)
    cov = pd.DataFrame()
    if a.step in ("aggregate", "all"):
        cov = aggregate(sites, reg, models, raw, out_dir, tag)
    stats = {
        "sites_tag": tag,
        "n_sites": int(len(sites)),
        "models": models,
        "end": end,
        "availability_start": avail,
        "n_chunk_requests": len(jobs),
        "planned_weight_calls": round(tot_w, 1),
        "weight_per_site_calls": round(tot_w / max(len(sites), 1), 1),
        "client": client.stats,
        "download": dl,
        "elapsed_s": round(time.time() - t_start, 1),
        "ledger_used": ledger.used(),
        "free_tier_limits": OM.FREE_LIMITS,
    }
    if len(cov):
        cs = cov.groupby(["model", "variable", "lead_day"]).agg(
            n_sites=("site_key", "nunique"),
            median_days=("n_days", "median"),
            first=("first_date", "min"),
            last=("last_date", "max"),
        )
        stats["coverage_summary"] = {
            f"{m}:{v}:lead{ld}": {
                "n_sites": int(r.n_sites),
                "median_days": float(r.median_days),
                "first": str(r.first),
                "last": str(r.last),
            }
            for (m, v, ld), r in cs.iterrows()
        }
    (raw / f"stats_{tag}.json").write_text(json.dumps(stats, indent=2, default=str))
    if a.step in ("aggregate", "all") and (out_dir / f"nwp_daily_{tag}.parquet").exists():
        write_manifest(
            f"nwp_archive_{tag}",
            files=[
                out_dir / f"nwp_daily_{tag}.parquet",
                out_dir / f"coverage_{tag}.parquet",
                raw / f"sites_{tag}.csv",
                raw / "availability.json",
                raw / "lead_semantics.json",
            ],
            sources=[OM.PREVIOUS_RUNS_URL, SINGLE_RUNS_URL],
            description="Archived NWP forecasts (Open-Meteo Previous Runs) aggregated to local standard "
            "days per lead 1-7 with max_init_utc; OPERATIONAL covariates only.",
            license="Open-Meteo CC BY 4.0 (non-commercial free tier); NOAA GFS public domain; ECMWF open "
            "data CC BY 4.0",
            stats=stats,
        )
    log.info("done: %s", {k: v for k, v in stats.items() if k not in ("coverage_summary",)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
