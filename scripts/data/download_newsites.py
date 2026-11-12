#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Pull + build the PROTECTED USGS new-site block (EW-v3-N / M1-N unseen test population; lead approval 2026-11-09).

Sites: ``docs/preregistration/sonde_newsites_list.json`` (metadata-only frozen list, sha checked). Window: WY2008–2025.
Legacy NWIS services (before the 2026-11-16 slow-down), same request code as the core pulls.

Phases (resume-safe):
  dv   legacy DV, 10 sites/request, all SENTINEL parameters, stats max/min/mean/median ->
       raw/usgs_dv_newsites/batch_*.parquet -> frozen QC (``sonde_daily.clean_long`` + ``splice_site``) ->
       processed/sonde_newsites/dv/sites/<site>.parquet
  iv   legacy IV per site-year (``download_usgs_iv.fetch_site_year``) into raw/usgs_iv_newsites/<site>/ ; when a site is
       complete: frozen QC (``sonde_hourly.clean_iv``) -> hourly tier (``sonde_hourly.hourly_site``) and the
       IV-derived daily tier (rule below) under processed/sonde_newsites/{hourly,daily_iv}/sites/ ; the raw site
       directory is then DELETED (disk cap). A per-site DONE marker records file/row counts only.
  manifest   content pin (sha256 over '<relpath>\\t<sha256>' of every built file) + integrity table (per site and
       tier: number of rows, first/last date, observed days per variable) -> processed/sonde_newsites/manifest.json.

IV-derived daily rule (pre-specified in ew_v3_protocol.md §3): per parameter, local STANDARD time calendar day (offset
of the site's standard time zone), from QC'd 15-min values; min / mean / max / median; a day is kept only if its
number of values ≥ 80 % of the series' median daily count in that water year; provisional if any value is.
Columns follow the daily tier: ``<var>_<stat>``, ``<var>_<stat>_prov``.

INTEGRITY ONLY: this script prints counts and coverage, never value statistics, onsets or exceedances.

    python -B scripts/data/download_newsites.py dv
    python -B scripts/data/download_newsites.py iv --workers 3
    python -B scripts/data/download_newsites.py manifest
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import importlib.util
import json
import logging
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from sentinel.core import paths
from sentinel.data.sonde_daily import clean_long, splice_site
from sentinel.data.sonde_hourly import clean_iv, hourly_site
from sentinel.data.sources import usgs_legacy as U

LIST = paths.REPO_ROOT / "docs" / "preregistration" / "sonde_newsites_list.json"
LIST_SHA = "6200a77cbdb0bb7963e6373a962374a764a6577bd7b945a42530a2b2e4e8b720"
RAW_DV = paths.RAW / "usgs_dv_newsites"
RAW_IV = paths.RAW / "usgs_iv_newsites"
OUT = paths.PROCESSED / "sonde_newsites"
START, END = "2007-10-01", "2025-09-30"
STD_OFFSET_H = {"EST": -5, "EDT": -5, "CST": -6, "CDT": -6, "MST": -7, "MDT": -7, "PST": -8, "PDT": -8,
                "AKST": -9, "AKDT": -9, "HST": -10, "AST": -4, "ADT": -4, "UTC": 0, "GMT": 0, "ChST": 10, "SST": -11}
STATS_IV = {"min": "min", "mean": "mean", "max": "max", "median": "median"}
log = logging.getLogger("newsites")


def _iv_module():
    spec = importlib.util.spec_from_file_location("_dl_iv", paths.REPO_ROOT / "scripts/data/download_usgs_iv.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_list() -> dict:
    if hashlib.sha256(LIST.read_bytes()).hexdigest() != LIST_SHA:
        raise SystemExit("new-site list differs from the frozen sha256")
    return json.loads(LIST.read_text())


def _sha(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


# --------------------------------------------------------------------------- DV
def phase_dv(lst) -> None:
    RAW_DV.mkdir(parents=True, exist_ok=True)
    sites = [s["site_no"] for s in lst["sites"]]
    params, stats = list(U.ALL_PARAMS), ["00001", "00002", "00003", "00008"]
    for b in range(0, len(sites), 10):
        dest = RAW_DV / f"batch_{b // 10:05d}.parquet"
        if dest.exists() or dest.with_suffix(".empty").exists():
            continue
        try:
            df = U.get_dv(sites[b:b + 10], params, START, END, stat_codes=stats)
        except Exception as e:  # noqa: BLE001 — rerun resumes
            log.error("dv batch %d failed: %s", b // 10, str(e)[:200])
            continue
        if df.empty:
            dest.with_suffix(".empty").touch()
        else:
            df.to_parquet(dest, index=False, compression="zstd")
        log.info("dv batch %d/%d: %d rows", b // 10 + 1, (len(sites) + 9) // 10, len(df))
        time.sleep(1.0)
    files = sorted(RAW_DV.glob("batch_*.parquet"))
    if not files:
        return
    dv = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    dv["date"] = pd.to_datetime(dv["date"])
    (OUT / "dv" / "sites").mkdir(parents=True, exist_ok=True)
    clean, counts = clean_long(dv)
    n = 0
    for s, g in clean.groupby("site_no"):
        wide, _ = splice_site(g)
        if not wide.empty:
            wide.to_parquet(OUT / "dv" / "sites" / f"{s}.parquet", compression="zstd")
            n += 1
    log.info("dv tier: %d sites written; QC counts %s", n, counts)


# --------------------------------------------------------------------------- IV
def daily_from_iv(clean: pd.DataFrame, tz: str | None) -> pd.DataFrame:
    off = STD_OFFSET_H.get(str(tz or ""), None)
    if off is None:
        off = int(clean["tz_cd"].astype(str).map(STD_OFFSET_H).dropna().mode().iloc[0]) if "tz_cd" in clean else 0
    t = pd.to_datetime(clean["time_utc"], utc=True).dt.tz_localize(None) + pd.Timedelta(hours=off)
    df = pd.DataFrame({"parm": clean["parm_cd"].astype(str).values, "day": t.dt.floor("D").values,
                       "value": clean["value"].astype(float).values, "prov": clean["provisional"].values})
    cols = {}
    for pc, g in df.groupby("parm"):
        if pc not in U.ALL_PARAMS:
            continue
        agg = g.groupby("day").agg(n=("value", "size"), mn=("value", "min"), me=("value", "mean"),
                                   mx=("value", "max"), md=("value", "median"), pv=("prov", "any"))
        wy = agg.index.year + (agg.index.month >= 10)
        need = 0.8 * agg.groupby(wy)["n"].transform("median")
        agg = agg[agg["n"] >= need]
        name = U.ALL_PARAMS[pc]
        for st, c in (("min", "mn"), ("mean", "me"), ("max", "mx"), ("median", "md")):
            cols[f"{name}_{st}"] = agg[c].astype("float32")
            cols[f"{name}_{st}_prov"] = agg["pv"].astype(bool)
    if not cols:
        return pd.DataFrame()
    out = pd.DataFrame(cols).sort_index()
    out.index.name = "date"
    for c in out.columns:
        if c.endswith("_prov"):
            out[c] = out[c].astype("boolean").fillna(False).astype(bool)
    return out


def site_job(dl, site: dict) -> dict:
    s = site["site_no"]
    done = OUT / "iv_done" / f"{s}.json"
    if done.exists():
        return json.loads(done.read_text())
    b = max(pd.Timestamp(min(v["begin"] for v in site["variables"].values())), pd.Timestamp(START))
    e = min(pd.Timestamp(max(v["end"] for v in site["variables"].values())), pd.Timestamp(END))
    n_rows, n_files = 0, 0
    if b <= e:
        for year, wins in dl.site_windows(b, e):
            for k in range(3):
                try:
                    n = dl.fetch_site_year(s, year, wins, list(U.ALL_PARAMS), RAW_IV)
                    break
                except Exception as ex:  # noqa: BLE001
                    if k == 2:
                        raise RuntimeError(f"{s} {year}: {str(ex)[:200]}") from ex
                    time.sleep(20 * (k + 1))
            n_rows += max(n, 0)
    files = sorted((RAW_IV / s).glob("*.parquet"))
    rec = {"site_no": s, "n_year_files": len(files), "n_raw_rows": 0, "hourly_rows": 0, "daily_iv_rows": 0}
    if files:
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        df = df[(pd.to_datetime(df["time_utc"], utc=True) >= pd.Timestamp(START, tz="UTC") - pd.Timedelta(hours=12))
                & (pd.to_datetime(df["time_utc"], utc=True) <= pd.Timestamp(END, tz="UTC") + pd.Timedelta(hours=36))]
        rec["n_raw_rows"] = int(len(df))
        cl = clean_iv(df)
        wide, _ = hourly_site(cl)
        if not wide.empty:
            (OUT / "hourly" / "sites").mkdir(parents=True, exist_ok=True)
            wide.to_parquet(OUT / "hourly" / "sites" / f"{s}.parquet", compression="zstd")
            rec["hourly_rows"] = int(len(wide))
        d = daily_from_iv(cl, site.get("tz"))
        d = d[(d.index >= START) & (d.index <= END)] if not d.empty else d
        if not d.empty:
            (OUT / "daily_iv" / "sites").mkdir(parents=True, exist_ok=True)
            d.to_parquet(OUT / "daily_iv" / "sites" / f"{s}.parquet", compression="zstd")
            rec["daily_iv_rows"] = int(len(d))
    shutil.rmtree(RAW_IV / s, ignore_errors=True)
    done.parent.mkdir(parents=True, exist_ok=True)
    done.write_text(json.dumps(rec))
    return rec


def phase_iv(lst, workers: int) -> None:
    RAW_IV.mkdir(parents=True, exist_ok=True)
    dl = _iv_module()
    dl.log = log
    sites = lst["sites"]
    ok = fail = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(site_job, dl, s): s["site_no"] for s in sites}
        for f in as_completed(futs):
            try:
                r = f.result()
                ok += 1
                if ok % 25 == 0:
                    log.info("iv progress: %d/%d sites done, %d failed", ok, len(sites), fail)
            except Exception as e:  # noqa: BLE001 — rerun resumes the failed sites
                fail += 1
                log.error("site %s failed: %s", futs[f], str(e)[:300])
    log.info("iv done: %d ok, %d failed", ok, fail)


# --------------------------------------------------------------------------- manifest (integrity only)
def phase_manifest(lst) -> None:
    files = sorted(p for t in ("dv", "hourly", "daily_iv") for p in (OUT / t / "sites").glob("*.parquet"))
    h = hashlib.sha256()
    for f in files:
        h.update(f"{f.relative_to(OUT).as_posix()}\t{_sha(f)}\n".encode())
    integ = []
    for f in files:
        d = pd.read_parquet(f)
        idx = pd.to_datetime(d.index)
        vc = [c for c in d.columns if not c.endswith(("_prov", "_ts")) and not c.startswith(("n_", "flag"))]
        integ.append({"tier": f.parent.parent.name, "site_no": f.stem, "n_rows": int(len(d)),
                      "first": str(idx.min().date()) if len(d) else None, "last": str(idx.max().date()) if len(d) else None,
                      **{f"nobs_{c}": int(d[c].notna().sum()) for c in vc}})
    pd.DataFrame(integ).to_parquet(OUT / "integrity.parquet", index=False)
    done = [json.loads(p.read_text()) for p in sorted((OUT / "iv_done").glob("*.json"))]
    man = {"name": "sonde_newsites", "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
           "protected": "EW-v3-N / M1-N unseen test block — no training, tuning, selection or target statistics; "
                        "loadable only through sentinel.data.newsite_guard with a lead-written freeze",
           "site_list": str(LIST.relative_to(paths.REPO_ROOT)), "site_list_sha256": LIST_SHA, "window": [START, END],
           "sources": [U.DV_URL, U.IV_URL], "license": "US public domain",
           "n_files": {t: len(list((OUT / t / "sites").glob("*.parquet"))) for t in ("dv", "hourly", "daily_iv")},
           "n_sites_iv_done": len(done), "content_sha256": h.hexdigest(), "n_content_files": len(files),
           "integrity_table": "integrity.parquet (counts and coverage only)"}
    (OUT / "manifest.json").write_text(json.dumps(man, indent=1))
    print(json.dumps(man, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=("dv", "iv", "manifest"))
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    RAW_IV.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(OUT / f"build_{a.phase}.log"), logging.StreamHandler()])
    lst = load_list()
    {"dv": lambda: phase_dv(lst), "iv": lambda: phase_iv(lst, a.workers), "manifest": lambda: phase_manifest(lst)}[a.phase]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
