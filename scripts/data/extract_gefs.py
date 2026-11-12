#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Stream-extract-delete of GEFSv12 c00 forecasts at sonde points (``sonde_arch_v2_amendment.md`` §2c; lead
dispatch 2026-11-10: allowed while kept extractions stay < 1 GB and ``/`` keeps ≥ 60 GB free).

Products: ``reforecast`` (AWS ``noaa-gefs-retrospective``, 00 UTC, one GRIB2 per variable × init, 0.25°) and
``operational`` (AWS ``noaa-gefs-pds``, GEFSv12 from 2020-09-23, ``pgrb2sp25`` one file per step). Variables: TMAX,
TMIN (2 m), APCP, DSWRF. Only the GRIB messages needed are fetched (HTTP byte ranges from the ``.idx``), decoded in
memory with eccodes (separate conda env ``sentinel-grib``, lead ruling R3.4; ``sentinel2`` is not modified), sampled at the nearest 0.25° grid point of each
site, and discarded — nothing global is written to disk.

Issue convention (conservative availability): issue date d (local) uses the 00 UTC run of calendar day d. Lead day
k ∈ 1..7 = local-standard-time day d + k. GEFS windows alternate (a, a+3) and (a, a+6), a ≡ 0 mod 6; 3-hourly
values are de-accumulated (APCP: difference; DSWRF: 2·avg6 − avg3; TMAX/TMIN: the 6-h extreme when it exceeds the
first half, else the first-half value as a bound) and assigned to the local day containing the window midpoint.
Daily Tmax/Tmin = extreme over the day's windows, P = sum, SW = mean; a day needs all 8 windows.

Output (compact, wide): one parquet per product × month with columns site, issue (local date), and int16 fields
``{tmax,tmin}_l{k}`` (0.1 °C), ``p_l{k}`` (0.1 mm), ``sw_l{k}`` (W/m²). Split train/val sites go to
``processed/gefs_points/<product>/``, new-site points to ``processed/gefs_points_newsites/<product>/`` (forcing
only, never targets; files 0440). Split sites are extracted only for issues ≤ 2023-09-30.

    python -B scripts/data/extract_gefs.py --product reforecast --start 2007-10-01 --end 2019-12-31 --threads 6
    python -B scripts/data/extract_gefs.py --product operational --dry-run 2
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as _dt
import json
import logging
import os
import shutil
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sentinel.core import paths  # noqa: E402

REFC = "https://noaa-gefs-retrospective.s3.amazonaws.com/GEFSv12/reforecast"
OPER = "https://noaa-gefs-pds.s3.amazonaws.com"
VARS = {"TMAX": "tmax_2m", "TMIN": "tmin_2m", "APCP": "apcp_sfc", "DSWRF": "dswrf_sfc"}
LEADS = 7
SPLIT_END = pd.Timestamp("2023-09-30")
V12_START = pd.Timestamp("2020-09-23")
FLOOR_GB = 60
log = logging.getLogger("gefs")


def free_gb() -> float:
    return shutil.disk_usage("/").free / 1e9


def http_get(url: str, rng: tuple[int, int | None] | None = None, retries: int = 5) -> bytes:
    for a in range(retries):
        try:
            req = urllib.request.Request(url)
            if rng is not None:
                req.add_header("Range", f"bytes={rng[0]}-{'' if rng[1] is None else rng[1]}")
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            if a == retries - 1:
                raise
            time.sleep(2 ** a)
            log.debug("retry %s: %s", url, e)


def parse_idx(txt: str) -> list[dict]:
    out = []
    lines = [ln for ln in txt.strip().splitlines() if ln]
    for i, ln in enumerate(lines):
        p = ln.split(":")
        start = int(p[1])
        end = int(lines[i + 1].split(":")[1]) - 1 if i + 1 < len(lines) else None
        step = p[5].split()[0]                         # e.g. "6-9"
        if "-" not in step:
            continue
        a, b = (int(x) for x in step.split("-"))
        out.append({"var": p[3], "a": a, "b": b, "start": start, "end": end})
    return out


# ----------------------------------------------------------------------------- points

def build_points() -> pd.DataFrame:
    reg = pd.read_parquet(paths.PROCESSED / "sites" / "sites.parquet")
    split = json.loads((paths.REPO_ROOT / "docs/preregistration/sonde_split.json").read_text())["site_role"]
    r = reg[reg["site_key"].str.startswith("USGS-")].copy()
    r["site"] = r["site_key"].str[5:]
    r = r[r["site"].map(split).isin(["train", "val"])]
    tv = pd.DataFrame({"site": r["site"], "lat": r["lat"], "lon": r["lon"], "off": r["utc_offset_h"],
                       "group": "trainval"})
    off = {"EST": -5, "CST": -6, "MST": -7, "PST": -8, "AKST": -9, "AST": -4, "HST": -10}
    ns = json.loads((paths.REPO_ROOT / "docs/preregistration/sonde_newsites_list.json").read_text())["sites"]
    nw = pd.DataFrame({"site": [s["site_no"] for s in ns], "lat": [s["lat"] for s in ns],
                       "lon": [s["lon"] for s in ns], "off": [off.get(s["tz"], -6) for s in ns], "group": "newsites"})
    pts = pd.concat([tv, nw], ignore_index=True).dropna(subset=["lat", "lon"]).reset_index(drop=True)
    pts["off"] = pts["off"].fillna(-6).astype(int)
    return pts


def grid_index(lat: np.ndarray, lon: np.ndarray, ni: int = 1440, nj: int = 721, res: float = 0.25) -> np.ndarray:
    j = np.clip(np.rint((90.0 - lat) / res).astype(int), 0, nj - 1)
    i = np.rint((np.mod(lon, 360.0)) / res).astype(int) % ni
    return j * ni + i


# ----------------------------------------------------------------------------- decode + aggregate

CHECK: dict = {}     # pilot correctness check: {"lat": (n,), "lon": (n,), "pos": (n,) indices into the points}


def decode(msgs: bytes, idx: np.ndarray) -> list[tuple[str, int, int, np.ndarray]]:
    import eccodes

    out, pos = [], 0
    while pos < len(msgs):
        gid = eccodes.codes_new_from_message(msgs[pos:])
        if gid is None:
            break
        try:
            n = eccodes.codes_get(gid, "totalLength")
            if eccodes.codes_get(gid, "Ni") != 1440 or eccodes.codes_get(gid, "Nj") != 721:
                raise ValueError("unexpected grid")
            short = eccodes.codes_get(gid, "shortName")
            rng = eccodes.codes_get(gid, "stepRange")
            vals = eccodes.codes_get_values(gid)
            a, b = (int(x) for x in str(rng).split("-"))
            out.append((short, a, b, vals[idx].astype(np.float64)))
            if CHECK.get("active") and CHECK["n_msgs"].get(short, 0) < 3:   # 3 messages per variable
                CHECK["n_msgs"][short] = CHECK["n_msgs"].get(short, 0) + 1
                # independent read: eccodes' own nearest-grid-point lookup
                for la, lo, p in zip(CHECK["lat"], CHECK["lon"], CHECK["pos"]):
                    ref = eccodes.codes_grib_find_nearest(gid, float(la), float(lo % 360.0))[0]["value"]
                    CHECK["diffs"].append(abs(float(ref) - float(vals[idx[p]])))
        finally:
            eccodes.codes_release(gid)
        pos += n
    return out


SHORT = {"tmax": "TMAX", "mx2t6": "TMAX", "tmin": "TMIN", "mn2t6": "TMIN", "tp": "APCP", "dswrf": "DSWRF",
         "sdswrf": "DSWRF"}


def three_hourly(recs: list) -> dict:
    """→ {var: {(a3, b3): values}} with 3-hourly windows (de-accumulated)."""
    by = {}
    for short, a, b, v in recs:
        var = SHORT.get(short, short.upper())
        by.setdefault(var, {})[(a, b)] = v
    out = {}
    for var, d in by.items():
        w = {}
        for (a, b), v in d.items():
            if b - a == 3:
                w[(a, b)] = v
        for (a, b), v in d.items():
            if b - a == 6 and (a, a + 3) in d:
                first = d[(a, a + 3)]
                if var == "APCP":
                    sec = np.maximum(v - first, 0.0)
                elif var == "DSWRF":
                    sec = np.maximum(2 * v - first, 0.0)
                elif var == "TMAX":
                    sec = np.where(v > first, v, first)
                else:
                    sec = np.where(v < first, v, first)
                w[(a + 3, b)] = sec
        out[var] = w
    return out


def daily(w3: dict, offs: np.ndarray, n_pts: int) -> dict:
    """3-hourly windows (hours after init 00Z of issue day d) → per lead day k: (n_pts,) fields."""
    res = {}
    for k in range(1, LEADS + 1):
        lo = 24 * k - offs                           # local midnight of day d+k in hours after init (offs < 0)
        for var, w in w3.items():
            vals, cnt = [], np.zeros(n_pts)
            acc = None
            for (a, b), v in w.items():
                mid = (a + b) / 2
                inday = (mid > lo) & (mid < lo + 24)
                if not inday.any():
                    continue
                cnt += inday
                if var == "TMAX":
                    acc = np.where(inday, np.fmax(acc if acc is not None else -np.inf, v), acc
                                   if acc is not None else -np.inf)
                elif var == "TMIN":
                    acc = np.where(inday, np.fmin(acc if acc is not None else np.inf, v), acc
                                   if acc is not None else np.inf)
                else:
                    acc = (acc if acc is not None else 0.0) + np.where(inday, v, 0.0)
            if acc is None:
                acc = np.full(n_pts, np.nan)
            full = cnt == 8
            if var == "DSWRF":
                acc = acc / np.maximum(cnt, 1)
            res[(var, k)] = np.where(full, acc, np.nan)
    return res


# ----------------------------------------------------------------------------- per init

def fetch_init(product: str, d: pd.Timestamp, pts_idx: np.ndarray, offs: np.ndarray, max_h: int) -> tuple[dict, int]:
    ymd = d.strftime("%Y%m%d")
    nbytes, recs = 0, []
    if product == "reforecast":
        for var, fname in VARS.items():
            base = f"{REFC}/{d.year}/{ymd}00/c00/Days:1-10/{fname}_{ymd}00_c00.grib2"
            idx = [m for m in parse_idx(http_get(base + ".idx").decode()) if m["b"] <= max_h]
            if not idx:
                continue
            blob = http_get(base, (idx[0]["start"], idx[-1]["end"]))
            nbytes += len(blob)
            recs += decode(blob, pts_idx)
    else:
        for fh in range(3, max_h + 1, 3):
            base = f"{OPER}/gefs.{ymd}/00/atmos/pgrb2sp25/gec00.t00z.pgrb2s.0p25.f{fh:03d}"
            idx = [m for m in parse_idx(http_get(base + ".idx").decode()) if m["var"] in VARS]
            for m in idx:
                blob = http_get(base, (m["start"], m["end"]))
                nbytes += len(blob)
                recs += decode(blob, pts_idx)
    return daily(three_hourly(recs), offs, pts_idx.size), nbytes


def to_frame(res: dict, sites: np.ndarray, d: pd.Timestamp) -> pd.DataFrame:
    cols = {"site": sites, "issue": np.full(sites.size, d.to_datetime64())}
    for k in range(1, LEADS + 1):
        for var, name, scale, off in (("TMAX", "tmax", 10, -273.15), ("TMIN", "tmin", 10, -273.15),
                                      ("APCP", "p", 10, 0.0), ("DSWRF", "sw", 1, 0.0)):
            v = res.get((var, k), np.full(sites.size, np.nan))
            q = np.rint((v + off) * scale)
            cols[f"{name}_l{k}"] = np.where(np.isfinite(q), np.clip(q, -32000, 32000), -32768).astype(np.int16)
    return pd.DataFrame(cols)


def run_months(product: str, days: pd.DatetimeIndex, pts: pd.DataFrame, pidx, offs, max_h: int, threads: int,
               check_first: bool = False) -> dict:
    out_tv = paths.ensure(paths.PROCESSED / "gefs_points" / product)
    out_ns = paths.ensure(paths.PROCESSED / "gefs_points_newsites" / product)
    months = sorted({(d.year, d.month) for d in days})
    stats = {"inits": 0, "failed": 0, "bytes": 0, "seconds": 0.0, "kept_bytes_written": 0, "months": 0}
    for (y, mo) in months:
        f_tv, f_ns = out_tv / f"{y}{mo:02d}.parquet", out_ns / f"{y}{mo:02d}.parquet"
        if f_tv.exists() and f_ns.exists():
            continue
        if free_gb() < FLOOR_GB + 5:
            raise SystemExit(f"free disk {free_gb():.0f} GB below floor + margin: stop")
        md = [d for d in days if d.year == y and d.month == mo]
        t0 = time.time()
        frames = []
        if check_first and stats["inits"] == 0 and md:        # correctness check on the first init (serial)
            rs = np.random.default_rng(0)
            pos = np.sort(rs.choice(len(pts), 25, replace=False))
            CHECK.update({"active": True, "lat": pts["lat"].to_numpy()[pos], "lon": pts["lon"].to_numpy()[pos],
                          "pos": pos, "diffs": [], "n_msgs": {}})
            res, nb = fetch_init(product, md[0], pidx, offs, max_h)
            CHECK["active"] = False
            stats["check"] = {"n_sites": 25, "n_values": len(CHECK["diffs"]), "messages": dict(CHECK["n_msgs"]),
                              "max_abs_diff": float(max(CHECK["diffs"])) if CHECK["diffs"] else None}
            log.info("correctness check %s: %s", product, stats["check"])
            frames.append(to_frame(res, pts["site"].to_numpy(), md[0]))
            stats["bytes"] += nb
            stats["inits"] += 1
            md = md[1:]
        with cf.ThreadPoolExecutor(max_workers=threads) as ex:
            futs = {ex.submit(fetch_init, product, d, pidx, offs, max_h): d for d in md}
            for fu in cf.as_completed(futs):
                d = futs[fu]
                try:
                    res, nb = fu.result()
                except Exception as e:  # noqa: BLE001
                    log.warning("init %s failed: %s", d.date(), e)
                    stats["failed"] += 1
                    continue
                stats["bytes"] += nb
                stats["inits"] += 1
                frames.append(to_frame(res, pts["site"].to_numpy(), d))
        stats["seconds"] += time.time() - t0
        if not frames:
            continue
        df = pd.concat(frames, ignore_index=True)
        df["group"] = np.tile(pts["group"].to_numpy(), len(frames))
        tv = df[(df["group"] == "trainval") & (df["issue"] <= SPLIT_END)].drop(columns="group")
        ns = df[df["group"] == "newsites"].drop(columns="group")
        for frame, f in ((tv, f_tv), (ns, f_ns)):
            tmp = f.with_name("." + f.name + ".tmp")
            frame.sort_values(["site", "issue"]).to_parquet(tmp, index=False, compression="zstd")
            tmp.replace(f)
            stats["kept_bytes_written"] += f.stat().st_size
        os.chmod(f_ns, 0o440)
        stats["months"] += 1
        log.info("%s %d-%02d: %d inits (%d failed so far), %.1f GB transferred, kept written %.3f GB, free %.0f GB",
                 product, y, mo, len(frames), stats["failed"], stats["bytes"] / 1e9,
                 stats["kept_bytes_written"] / 1e9, free_gb())
        if kept_total() > 1e9:
            raise SystemExit("kept extraction > 1 GB: stop (lead limit)")
    return stats


def kept_total() -> int:
    return sum(p.stat().st_size for d in ("gefs_points", "gefs_points_newsites")
               for p in (paths.PROCESSED / d).rglob("*.parquet"))


def write_manifest() -> None:
    import subprocess as sp
    env = sp.run(["/home/bcheng/miniconda3/bin/conda", "list", "-n", "sentinel-grib", "--explicit"],
                 capture_output=True, text=True).stdout
    m = paths.ensure(paths.PROCESSED / "gefs_points") / "MANIFEST_extract.json"
    m.write_text(json.dumps({"created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                             "script": "scripts/data/extract_gefs.py", "decoder_env": "conda env sentinel-grib",
                             "decoder_env_explicit": env.splitlines(), "member": "c00",
                             "variables": list(VARS), "leads": LEADS, "issue_rule": "00 UTC run of issue day d",
                             "ruling": "sonde_arch_v2_lead_rulings.md R3"}, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--product", choices=("reforecast", "operational", "both"), default="both")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--dry-run", type=int, default=0, help="process N inits, report bytes and time, write nothing")
    ap.add_argument("--pilot", action="store_true", help="R3 pilot (WY2015 reforecast + 2021-06 operational), "
                    "registry rows, then continue automatically unless a stop condition holds")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    pts = build_points()
    pidx = grid_index(pts["lat"].to_numpy(), pts["lon"].to_numpy())
    offs = pts["off"].to_numpy(float)
    max_h = int(24 * (LEADS + 1) - offs.min() + 3)              # cover local day d+7 at the westernmost site
    full = {"reforecast": pd.date_range("2007-10-01", "2019-12-31", freq="D"),
            "operational": pd.date_range(V12_START, "2025-09-30", freq="D")}
    if a.dry_run:
        product = "reforecast" if a.product == "both" else a.product
        days = pd.date_range(a.start or full[product][0], a.end or full[product][-1], freq="D")
        t0, tot = time.time(), 0
        for d in days[: a.dry_run]:
            res, nb = fetch_init(product, d, pidx, offs, max_h)
            tot += nb
            df = to_frame(res, pts["site"].to_numpy(), d)
            print(d.date(), f"{nb / 1e6:.1f} MB", f"rows {len(df)}", "non-missing tmax_l1:",
                  int((df["tmax_l1"] != -32768).sum()), "example:", df.iloc[0, 2:6].to_dict())
        el = time.time() - t0
        print(json.dumps({"inits": a.dry_run, "mb_per_init": tot / 1e6 / a.dry_run, "s_per_init": el / a.dry_run,
                          "max_h": max_h}))
        return 0
    write_manifest()
    if a.pilot:
        from sentinel.core.registry import log_result

        ref_days = pd.date_range("2014-10-01", "2015-09-30", freq="D")
        op_days = pd.date_range("2021-06-01", "2021-06-30", freq="D")
        st_r = run_months("reforecast", ref_days, pts, pidx, offs, max_h, a.threads, check_first=True)
        st_o = run_months("operational", op_days, pts, pidx, offs, max_h, a.threads, check_first=True)
        n_total = {"reforecast": len(full["reforecast"]), "operational": len(full["operational"])}
        proj_kept = (st_r["kept_bytes_written"] / max(st_r["inits"], 1) * n_total["reforecast"]
                     + st_o["kept_bytes_written"] / max(st_o["inits"], 1) * n_total["operational"]) / 1e9
        proj_h = (st_r["seconds"] / max(st_r["inits"], 1) * (n_total["reforecast"] - st_r["inits"])
                  + st_o["seconds"] / max(st_o["inits"], 1) * (n_total["operational"] - st_o["inits"])) / 3600
        checks = [st.get("check", {}).get("max_abs_diff") for st in (st_r, st_o)]
        ok_check = all(c is not None and c < 1e-6 for c in checks)
        rec = {"reforecast": st_r, "operational": st_o, "projected_kept_gb": proj_kept,
               "projected_wallclock_h": proj_h, "check_ok": ok_check}
        (paths.PROCESSED / "gefs_points" / "PILOT_REPORT.json").write_text(json.dumps(rec, indent=1, default=str))
        for metric, value in (("check_max_abs_diff", max(c for c in checks if c is not None) if any(
                c is not None for c in checks) else -1.0), ("projected_kept_gb", proj_kept),
                ("projected_wallclock_h", proj_h),
                ("throughput_inits_per_h_reforecast", st_r["inits"] / max(st_r["seconds"], 1) * 3600),
                ("throughput_inits_per_h_operational", st_o["inits"] / max(st_o["seconds"], 1) * 3600),
                ("mb_per_init_reforecast", st_r["bytes"] / max(st_r["inits"], 1) / 1e6),
                ("mb_per_init_operational", st_o["bytes"] / max(st_o["inits"], 1) / 1e6)):
            log_result(workstream="M1-sonde", experiment="sonde_arch_v2/gefs_pilot", model="extract_gefs",
                       metric=metric, value=float(value), split="data",
                       notes="R3 pilot: WY2015 reforecast + 2021-06 operational, c00; correctness check = "
                             "eccodes find_nearest vs own grid index at 25 sites", extra={k: v for k, v in rec.items()
                                                                                         if k.startswith("proj")})
        log.info("pilot: %s", {k: rec[k] for k in ("projected_kept_gb", "projected_wallclock_h", "check_ok")})
        if not ok_check or proj_kept > 1.0 or proj_h > 72:
            log.error("pilot stop condition hit: report to the lead")
            return 4
    products = ("reforecast", "operational") if a.product == "both" else (a.product,)
    for product in products:
        days = pd.date_range(a.start or full[product][0], a.end or full[product][-1], freq="D")
        run_months(product, days, pts, pidx, offs, max_h, a.threads)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
