#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Download UK Environment Agency ecology bulk files and paired Water Quality Archive chemistry.

Stages (run in order; each is resume-safe):

  ecology       Ecology & Fish Data Explorer bulk zips (INV, DIAT, TAXON_INFO). A local copy from
                ``--reuse-dir`` is used only if its name and byte size equal the official
                Content-Length (ETag/Last-Modified recorded). Zip CRCs are verified.
  points        WQA sampling points of all freshwater types (BNG coordinates), + determinand units.
  observations  WQA observations for the WQ points paired with invertebrate sites
                (rule in sentinel/data/bio_uk_ea.py: nearest freshwater point within 1 km), for
                the core determinands, only for the calendar years needed by the invertebrate
                samples' chemistry windows. <= 3 concurrent requests, 25 points x 1 year per call.

Raw output: $SENTINEL_DATA/raw/uk_ea/{ecology,wq}/...

    python -B scripts/data/download_uk_ea.py ecology --reuse-dir DIR
    python -B scripts/data/download_uk_ea.py points
    python -B scripts/data/download_uk_ea.py observations [--workers 3]
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import json
import logging
import shutil
import sys
import threading
import time
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths  # noqa: E402
from sentinel.core.manifest import write_manifest  # noqa: E402
from sentinel.data import bio_common as BC  # noqa: E402
from sentinel.data import bio_uk_ea as BU  # noqa: E402
from sentinel.data.sources import uk_ea as U  # noqa: E402

log = logging.getLogger("download_uk_ea")
FRESHWATER_TYPES = ("F6", "FJ", "FK", "FZ", "FA", "FL", "FD", "FB", "FC")


def stage_ecology(out: Path, reuse_dir: Path | None) -> int:
    eco = paths.ensure(out / "ecology")
    page = BC.http_request("GET", U.ECOLOGY_DOWNLOADS_URL).text
    today = dt.date.today().isoformat()
    (eco / f"ecology_downloads_page_{today}.html").write_text(page, encoding="utf-8")
    listed = U.parse_ecology_downloads_page(page)
    recs = []
    for name in U.ECOLOGY_FILES:
        if name not in listed:
            raise RuntimeError(f"{name} no longer listed on {U.ECOLOGY_DOWNLOADS_URL}: {listed}")
        url = U.ECOLOGY_DOWNLOADS_URL + name
        info = BC.head_info(url)
        dest = eco / name
        action = "present"
        if not (dest.exists() and dest.stat().st_size == info["content_length"]):
            src = None
            if reuse_dir and reuse_dir.exists():
                src = next((p for p in reuse_dir.rglob(name)
                            if p.is_file() and p.stat().st_size == info["content_length"]), None)
            if src is not None:
                shutil.copy2(src, dest)
                action = f"reused:{src}"
            else:
                BC.download_file(url, dest, expected_bytes=info["content_length"])
                action = "downloaded"
        with zipfile.ZipFile(dest) as zf:
            bad = zf.testzip()
            members = [(i.filename, i.file_size, "{:04d}-{:02d}-{:02d}".format(*i.date_time[:3])) for i in zf.infolist()]
        if bad:
            raise RuntimeError(f"CRC failure in {dest}: {bad}")
        recs.append({**info, "name": name, "path": str(dest), "bytes": dest.stat().st_size,
                     "action": action, "zip_members": members})
        log.info("%s %s (%d bytes, last-modified %s)", name, action, dest.stat().st_size,
                 info["last_modified"])
    (eco / "ecology_download_log.json").write_text(json.dumps(recs, indent=2))
    write_manifest("uk_ea_ecology_raw", files=[r["path"] for r in recs] + [str(eco / "ecology_download_log.json")],
                   sources=[U.ECOLOGY_DOWNLOADS_URL + n for n in U.ECOLOGY_FILES],
                   description="Environment Agency Ecology & Fish Data Explorer bulk files: freshwater "
                               "invertebrates, diatoms (LM + DNA metabarcoding), taxon dictionary.",
                   license=U.LICENSE,
                   stats={"retrieved": today, "files": {r["name"]: {k: r[k] for k in (
                       "content_length", "last_modified", "etag", "action", "zip_members")} for r in recs},
                       "reuse_rule": "same file name and byte size as official Content-Length"})
    return 0


def stage_points(out: Path) -> int:
    wq = paths.ensure(out / "wq")
    today = dt.date.today().isoformat()
    spec = BC.http_request("GET", U.WQ_SPEC_URL).text
    (wq / f"wqa_openapi_{today}.json").write_text(spec)
    frames, skip, total = [], 0, None
    pts_path = wq / "sampling_points_freshwater.parquet"
    while not pts_path.exists():
        r = BC.http_request("GET", U.WQ_SAMPLING_POINT_URL, headers=U.SAMPLING_POINT_HEADERS,
                            params=U.sampling_point_params(skip, 250, FRESHWATER_TYPES))
        doc = r.json()
        total = doc.get("totalItems", total)
        df = U.parse_sampling_points(doc)
        if df.empty:
            break
        frames.append(df)
        skip += len(df)
        log.info("sampling points %d / %s", skip, total)
        if total is not None and skip >= total:
            break
        time.sleep(0.3)
    if frames:
        pts = pd.concat(frames, ignore_index=True).drop_duplicates("wq_point")
        BC.write_parquet(pts, pts_path)
    pts = pd.read_parquet(pts_path)
    types = BC.http_request("GET", U.WQ_POINT_TYPES_URL, headers={"Accept": "application/json"}).json()
    (wq / "sampling_point_types.json").write_text(json.dumps(types, indent=1))
    dets = []
    for code in U.DETERMINANDS:
        d = BC.http_request("GET", U.WQ_DETERMINAND_URL, params={"notation": code, "limit": 5},
                            headers={"Accept": "application/json"}).json()
        for m in d.get("member", []):
            dets.append({"determinand": m.get("notation"), "label": m.get("prefLabel"),
                         "alt_label": m.get("altLabel"), "short_name": U.DETERMINANDS[code]})
        time.sleep(0.2)
    pd.DataFrame(dets).to_csv(wq / "determinands.csv", index=False)
    write_manifest("uk_ea_wq_points_raw",
                   files=[wq / "sampling_points_freshwater.parquet", wq / f"wqa_openapi_{today}.json",
                          wq / "sampling_point_types.json", wq / "determinands.csv"],
                   sources=[U.WQ_SAMPLING_POINT_URL, U.WQ_SPEC_URL, U.WQ_POINT_TYPES_URL, U.WQ_DETERMINAND_URL],
                   description="EA Water Quality Archive sampling points (freshwater types, BNG) and "
                               "determinand codelist entries for the SENTINEL core determinands.",
                   license=U.LICENSE,
                   stats={"retrieved": today, "n_points": len(pts), "total_items_reported": total,
                          "point_types_requested": FRESHWATER_TYPES,
                          "by_type": pts["wq_type"].value_counts().to_dict()})
    return 0


_lock = threading.Lock()


def _fetch_points(year: int, points: list[str], sess) -> tuple[int, list[pd.DataFrame]]:
    """All pages for one (points, year) query. Raises RuntimeError if a page keeps failing."""
    frames, skip, n_req = [], 0, 0
    while True:
        params = U.observation_params(points, list(U.DETERMINANDS), f"{year}-01-01", f"{year}-12-31",
                                      skip=skip, limit=2500)
        r = BC.http_request("POST", U.WQ_OBSERVATION_URL, params=params, headers=U.OBSERVATION_HEADERS,
                            ok_status=(200, 204), sess=sess, timeout=600, retries=4)
        n_req += 1
        if r.status_code == 204 or not r.content.strip():
            break
        df = U.parse_observations_csv(r.content)
        if df.empty:
            break
        frames.append(df)
        if len(df) < 2500:
            break
        skip += 2500
    return n_req, frames


def _fetch_chunk(year: int, chunk_id: int, points: list[str], dest: Path, sess) -> tuple[int, int, int]:
    """Fetch one chunk; on a persistent failure split the point list in half (recursively).
    Paging (skip > 0) of multi-point queries fails with HTTP 502 above ~40 points (observed
    2026-10-22), hence the default chunk of 25 points and this fallback."""
    if dest.exists():
        return 0, len(pd.read_parquet(dest, columns=["wq_point"])), 0
    n_req, frames, n_split = 0, [], 0
    stack = [points]
    while stack:
        pts = stack.pop()
        try:
            k, f = _fetch_points(year, pts, sess)
            n_req += k
            frames.extend(f)
        except RuntimeError:
            if len(pts) == 1:
                raise
            n_split += 1
            mid = len(pts) // 2
            stack.extend([pts[:mid], pts[mid:]])
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["wq_point", "date", "determinand", "value"])
    tmp = dest.with_suffix(".tmp.parquet")
    BC.write_parquet(out, tmp)
    tmp.replace(dest)
    return n_req, len(out), n_split


def stage_observations(out: Path, workers: int, max_dist_m: float, chunk_size: int = 25,
                       assemblage: str = "invertebrate") -> int:
    wq = paths.ensure(out / "wq")
    obs_dir = paths.ensure(wq / "observations")
    eco = out / "ecology"
    pts = pd.read_parquet(wq / "sampling_points_freshwater.parquet")
    zname = "INV_OPEN_DATA.zip" if assemblage == "invertebrate" else "DIAT_OPEN_DATA.zip"
    sites = BU.load_sites(eco / zname, assemblage)
    samples = BU.load_metrics(eco / zname, assemblage)
    pairs = BU.pair_sites_to_points(sites, pts, max_dist_m=max_dist_m)
    need = BU.years_needed(pairs, samples)
    prefix = f"c{chunk_size}"
    plan_name = "observation_request_plan.parquet"
    if assemblage != "invertebrate":
        # only point-years not already requested for the invertebrate plan; separate chunk names
        inv_plan = pd.read_parquet(wq / "observation_request_plan.parquet")
        need = need.merge(inv_plan, on=["wq_point", "year"], how="left", indicator=True)
        need = need[need["_merge"] == "left_only"].drop(columns="_merge")
        prefix = f"{assemblage[:4]}_c{chunk_size}"
        plan_name = f"observation_request_plan_{assemblage}.parquet"
    need.to_parquet(wq / plan_name, index=False)
    log.info("paired sites: %d; WQ points needed: %d; point-years: %d", pairs["site_id"].nunique(),
             need["wq_point"].nunique(), len(need))
    jobs = []
    for year, g in need.groupby("year"):
        pts_y = sorted(g["wq_point"].unique())
        for i in range(0, len(pts_y), chunk_size):
            chunk = pts_y[i:i + chunk_size]
            jobs.append((int(year), i // chunk_size, chunk,
                         obs_dir / f"year={int(year)}" / f"{prefix}_{i // chunk_size:04d}.parquet"))
    for _, _, _, d in jobs:
        d.parent.mkdir(parents=True, exist_ok=True)
    log.info("%d chunk-year jobs", len(jobs))
    sessions = threading.local()
    done = {"req": 0, "rows": 0, "jobs": 0, "splits": 0, "failed": []}

    def run(job):
        if not hasattr(sessions, "s"):
            sessions.s = BC.make_session()
        year, cid, chunk, dest = job
        try:
            n_req, n_rows, n_split = _fetch_chunk(year, cid, chunk, dest, sessions.s)
        except Exception as e:  # noqa: BLE001 — record and continue; rerun resumes
            with _lock:
                done["failed"].append({"year": year, "chunk": cid, "error": repr(e)[:300]})
            log.error("year %d chunk %d failed: %s", year, cid, e)
            return
        with _lock:
            done["req"] += n_req
            done["rows"] += n_rows
            done["jobs"] += 1
            done["splits"] += n_split
            if done["jobs"] % 50 == 0:
                log.info("jobs %d/%d requests %d rows %d", done["jobs"], len(jobs), done["req"], done["rows"])

    with cf.ThreadPoolExecutor(max_workers=max(1, min(workers, 4))) as ex:
        list(ex.map(run, jobs))
    files = sorted(obs_dir.rglob("*.parquet"))
    (wq / f"observation_download_log_{assemblage}.json").write_text(json.dumps(done, indent=1, default=str))
    plans = sorted(wq.glob("observation_request_plan*.parquet"))
    logs = sorted(wq.glob("observation_download_log*.json"))
    write_manifest("uk_ea_wq_observations_raw", files=files + plans + logs,
                   sources=[U.WQ_OBSERVATION_URL],
                   description="EA Water Quality Archive observations (core determinands) for sampling "
                               "points paired with EA invertebrate sites; parsed CSV -> Parquet.",
                   license=U.LICENSE,
                   stats={"retrieved": dt.date.today().isoformat(), "pairing_max_dist_m": max_dist_m,
                          "determinands": U.DETERMINANDS, "assemblage_plan": assemblage,
                          "n_jobs": len(jobs), "n_jobs_done": done["jobs"],
                          "n_requests": done["req"], "n_rows": done["rows"], "failed": done["failed"],
                          "n_splits": done["splits"],
                          "request": f"POST /data/observation, {chunk_size} points x 1 calendar year, limit "
                                     "2500 with skip paging, Accept text/csv, BNG; recursive halving of the "
                                     "point list on persistent failure"})
    log.info("observations done: %s", {k: v for k, v in done.items() if k != "failed"})
    return 0 if not done["failed"] else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["ecology", "points", "observations"])
    ap.add_argument("--reuse-dir", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--max-dist-m", type=float, default=BU.PAIR_MAX_DIST_M)
    ap.add_argument("--chunk-size", type=int, default=25)
    ap.add_argument("--assemblage", choices=["invertebrate", "diatom"], default="invertebrate",
                    help="whose sample windows drive the request plan (diatom = extra point-years only)")
    a = ap.parse_args()
    out = paths.raw("uk_ea")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(out / f"download_{a.stage}.log"), logging.StreamHandler()])
    if a.stage == "ecology":
        return stage_ecology(out, a.reuse_dir)
    if a.stage == "points":
        return stage_points(out)
    return stage_observations(out, a.workers, a.max_dist_m, a.chunk_size, a.assemblage)


if __name__ == "__main__":
    raise SystemExit(main())
