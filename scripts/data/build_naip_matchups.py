#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""NAIP 4-band chips + nearest Sentinel-2 chips for WQP surface samples within +-1 day of a NAIP
acquisition (M7 SENTINEL-Lite; docs/research/08 §b.2, §e.3).

Targets: Secchi, turbidity, TSS (and SSC), chl-a. States: Minnesota + the 3 states with the most
lake/reservoir samples of these targets in NAIP years (2010-2023), or ``--states``; ``--national``
uses every state present in the WQP table (disk-budget permitting).

Per (site, NAIP DOQQ) with a sample within +-1 d: a 256x256 NAIP chip at native GSD (uint8 DN,
R/G/B/NIR) and per-DOQQ radiometric statistics; for the same site and NAIP date, every S2 L2A
acquisition within +-1 d is SCL-screened and the nearest usable one gets a full S2 chip (teacher).

Outputs: processed('naip_matchups')/{matchups.parquet, naip_scene_stats.parquet, coverage.json};
chips: processed('naip_chips')/wqp, processed('s2_chips')/naip_teacher.

    python -B scripts/data/build_naip_matchups.py --source processed
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths  # noqa: E402
from sentinel.core.manifest import write_manifest  # noqa: E402
from sentinel.data import chips as C  # noqa: E402
from sentinel.data import matchups as M  # noqa: E402
from sentinel.data import naip as N  # noqa: E402
from sentinel.data.sources import stac as S  # noqa: E402

log = logging.getLogger("naip_matchups")
TARGETS = ("secchi", "turbidity", "tss", "ssc", "chla")
WINDOW = 1


def naip_items_for_sites(sites: pd.DataFrame, cache: Path, chunk: int = 40) -> dict[str, dict]:
    """NAIP items whose footprint intersects any site (MultiPoint queries; cached item JSON)."""
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    f = cache / "items.jsonl"
    done_f = cache / "queried_points.json"
    items: dict[str, dict] = {}
    if f.exists():
        for line in f.read_text().splitlines():
            it = json.loads(line)
            items[it["id"]] = it
    done = set(json.loads(done_f.read_text())) if done_f.exists() else set()
    todo = sites[~sites.point_key.isin(done)]
    with open(f, "a") as fh:
        for k in range(0, len(todo), chunk):
            g = todo.iloc[k:k + chunk]
            geom = {"type": "MultiPoint", "coordinates": [[float(a), float(b)] for a, b in zip(g.lon, g.lat)]}
            for it in S.search_naip(intersects=geom, start="2010-01-01", end="2023-12-31"):
                if it["id"] not in items:
                    items[it["id"]] = it
                    fh.write(json.dumps(it, separators=(",", ":")) + "\n")
            done |= set(g.point_key)
            if (k // chunk) % 25 == 0:
                done_f.write_text(json.dumps(sorted(done)))
                log.info("NAIP item search %d/%d sites, %d items", k + len(g), len(todo), len(items))
    done_f.write_text(json.dumps(sorted(done)))
    return items


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["processed", "raw"], default="processed")
    ap.add_argument("--states", nargs="*", default=None)
    ap.add_argument("--n-extra-states", type=int, default=3)
    ap.add_argument("--national", action="store_true")
    ap.add_argument("--read-workers", type=int, default=4)
    a = ap.parse_args()
    out = paths.processed("naip_matchups")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(out / "build.log"), logging.StreamHandler()])
    logging.getLogger("rasterio").setLevel(logging.WARNING)
    res, stations, used = M.load_wqp(a.source)
    smp = M.wqp_surface_samples(res, stations)
    del res
    smp = smp[smp.target.isin(TARGETS) & smp.date.between("2010-01-01", "2024-01-31")]
    smp = smp[smp.lat.between(17, 72) & smp.lon.between(-170, -64)].copy()
    if "state" not in smp.columns:
        raise KeyError("WQP table lacks a state column")
    if a.national:
        states = sorted(smp.state.dropna().unique())
    elif a.states:
        states = a.states
    else:
        lake = smp[smp.site_class == "lake_reservoir"]
        counts = lake.groupby("state").size().sort_values(ascending=False)
        states = ["MN"] + [s for s in counts.index if s != "MN"][: a.n_extra_states]
    log.info("states: %s", states)
    smp = smp[smp.state.isin(states)].copy()
    smp["point_key"] = [M.point_key(lo, la) for lo, la in zip(smp.lon, smp.lat)]
    smp["sample_key"] = (smp.site_id + "|" + smp.date.dt.strftime("%Y-%m-%d") + "|" + smp.target + "|" +
                         smp.characteristic + "|" + smp.unit)
    sites = smp.drop_duplicates("point_key")[["point_key", "site_id", "lat", "lon", "site_class", "state"]]
    jrc = M.site_jrc(sites, paths.interim("naip_matchups") / "site_jrc.parquet", logger=log.info)
    sites = sites.merge(jrc, on="point_key", how="left")
    sites["jrc_water_ok"] = sites.jrc_water_ok.astype("boolean").fillna(False).astype(bool)
    wsites = sites[sites.jrc_water_ok]
    log.info("samples %d at %d sites (%d JRC-water sites)", len(smp), len(sites), len(wsites))
    items = naip_items_for_sites(wsites, paths.interim("naip_catalog") / "wqp_sites")
    cat = pd.DataFrame([{"item_id": i["id"], "date": S.naip_acquisition_date(i),
                         "year": int(i["properties"].get("naip:year", 0)), "gsd": i["properties"].get("gsd"),
                         "geometry": json.dumps(i["geometry"])} for i in items.values()])
    si = N.sites_in_items(wsites, cat)
    mt = smp[smp.point_key.isin(wsites.point_key)].merge(si, on="point_key")
    mt["dt_days_naip"] = (mt.date - mt.naip_date).dt.days
    mt = mt[mt.dt_days_naip.abs() <= WINDOW].copy()
    log.info("NAIP matchups (+-%d d): %d samples, %d (site, DOQQ) pairs; by target %s", WINDOW, len(mt),
             mt[["point_key", "item_id"]].drop_duplicates().shape[0], mt.target.value_counts().to_dict())
    # ---- NAIP chips
    nreq = mt.drop_duplicates(["point_key", "item_id"])[["point_key", "item_id"]].merge(
        wsites[["point_key", "lon", "lat"]], on="point_key")
    nreq["request_key"] = nreq.item_id + "|" + nreq.point_key
    nw = C.ChipWriter(paths.processed("naip_chips") / "wqp", layers=N.NAIP_LAYERS, dtype="uint8",
                      semantics=N.NAIP_SEMANTICS)
    nx = N.NaipExtractor(nw, out / "naip_scene_stats.parquet", workers=min(4, a.read_workers), logger=log.info)
    ncounts = nx.run(nreq, items)
    nidx = C.consolidate(nw.root)
    # ---- nearest S2 within +-1 d of the NAIP date (teacher)
    pairs = mt.drop_duplicates(["point_key", "item_id"])[["point_key", "item_id", "naip_date"]].merge(
        wsites[["point_key", "lon", "lat"]], on="point_key")
    pairs = pairs[pairs.naip_date >= pd.Timestamp("2015-06-23")]
    need: dict[str, list] = {}
    for r in pairs.itertuples():
        for d in M.date_range_strings(r.naip_date, WINDOW, WINDOW):
            need.setdefault(d, []).append((r.point_key, r.lon, r.lat))
    pbd = {d: pd.DataFrame(v, columns=["point_key", "lon", "lat"]).drop_duplicates("point_key") for d, v in need.items()}
    s2_teacher = pd.DataFrame()
    if pbd:
        cand, scat = M.plan_s2_candidates(pbd, paths.interim("s2_scene_catalog") / "naip_teacher", logger=log.info)
        cand = cand.drop_duplicates(["point_key", "item_id"])
        cand["request_key"] = cand.item_id + "|" + cand.point_key
        cand = cand.merge(wsites[["point_key", "lon", "lat"]], on="point_key")
        scenes = M.scenes_from_catalog(scat)
        sw = C.ChipWriter(paths.processed("s2_chips") / "naip_teacher", layers=C.S2_STORE_LAYERS, dtype="int16",
                          semantics=C.S2_SEMANTICS)
        sx = M.S2Extractor(sw, read_workers=a.read_workers, scene_workers=3, logger=log.info)
        scr = sx.screen_scl(cand[["request_key", "item_id", "lon", "lat"]], scenes,
                            paths.interim("s2_screen") / "naip_teacher").drop_duplicates("request_key", keep="last")
        pc = pairs.merge(cand[["point_key", "item_id", "request_key"]].rename(columns={"item_id": "s2_item_id"}),
                         on="point_key").merge(scr.drop(columns=["item_id"]), on="request_key", how="left")
        st = pd.to_datetime(pc.sensing_time_utc, utc=True, errors="coerce")
        pc["dt_days_s2_naip"] = (st.dt.tz_localize(None).dt.normalize() - pc.naip_date).dt.days
        pc = pc[pc.dt_days_s2_naip.abs() <= WINDOW]
        pc["usable"] = (pc.screen_status == "ok") & ~pc.nodata_centre.astype("boolean").fillna(True).astype(bool) & \
            (pc.scl_c7_n_valid >= 5)
        pc["abs_dt"] = pc.dt_days_s2_naip.abs()
        best = pc[pc.usable].sort_values(["point_key", "item_id", "abs_dt", "scene_cloud_cover"]).drop_duplicates(
            ["point_key", "item_id"])
        req = best.drop_duplicates("request_key")  # pairs already carry the site lon/lat
        sx.run(req[["request_key", "s2_item_id", "lon", "lat", "point_key"]].rename(columns={"s2_item_id": "item_id"}),
               scenes)
        sx.close()
        sidx = C.consolidate(sw.root)
        s2_teacher = best[["point_key", "item_id", "s2_item_id", "request_key", "dt_days_s2_naip"]].merge(
            sidx[["request_key", "chip_id", "status", "scl_c3_n_water", "sensing_time_utc", "processing_baseline",
                  "offset_suspect"]], on="request_key", how="left").rename(
            columns={"request_key": "s2_request_key", "chip_id": "s2_chip_id", "status": "s2_status",
                     "scl_c3_n_water": "s2_scl_c3_n_water", "sensing_time_utc": "s2_sensing_time_utc"})
    # ---- assemble
    nidx2 = nidx.rename(columns={"request_key": "naip_request_key", "chip_id": "naip_chip_id", "status": "naip_status",
                                 "shard": "naip_shard"})
    mt["naip_request_key"] = mt.item_id + "|" + mt.point_key
    keep = [c for c in nidx2.columns if c not in ("lon", "lat", "point_key", "item_id", "naip_item_id", "naip_date")]
    mt = mt.merge(nidx2[keep], on="naip_request_key", how="left")
    if len(s2_teacher):
        mt = mt.merge(s2_teacher, on=["point_key", "item_id"], how="left")
    mt = mt.rename(columns={"item_id": "naip_item_id"})
    dest = out / "matchups.parquet"
    mt.to_parquet(dest, index=False, compression="zstd")
    cov = {"states": states, "n_samples_matched": int(len(mt)),
           "by_target": mt.target.value_counts().to_dict(),
           "by_state_target": {f"{k[0]}|{k[1]}": int(v) for k, v in mt.groupby(["state", "target"]).size().items()},
           "lake_reservoir_by_target": mt[mt.site_class == "lake_reservoir"].target.value_counts().to_dict(),
           "naip_chips_ok": int((nidx.status == "ok").sum()), "naip_counts": ncounts,
           "with_s2_teacher_chip": int((mt.get("s2_status") == "ok").sum()) if "s2_status" in mt else 0,
           "naip_years": mt.naip_year.value_counts().sort_index().to_dict()}
    (out / "coverage.json").write_text(json.dumps(cov, indent=1, default=str))
    log.info("coverage: %s", cov)
    files = [dest, out / "coverage.json", out / "naip_scene_stats.parquet", nw.root / "index.parquet"]
    write_manifest("naip_matchups", files=[f for f in files if f.exists()],
                   sources=[str(f) for f in used] + [S.PLANETARY_COMPUTER_URL, S.EARTH_SEARCH_URL],
                   description="WQP surface Secchi/turbidity/TSS/chl-a samples within +-1 d of NAIP acquisitions: NAIP "
                               "4-band 256x256 chips (raw DN) with per-DOQQ statistics, and the nearest usable S2 L2A "
                               "chip within +-1 d of the NAIP date (teacher).",
                   license="NAIP public domain (USDA FSA); WQP public; Copernicus Sentinel data",
                   synthetic=False, stats=json.loads(json.dumps(cov, default=str)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
