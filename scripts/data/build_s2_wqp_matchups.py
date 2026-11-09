#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Sentinel-2 L2A matchups for Water Quality Portal surface samples (chl-a, TSS, SSC, turbidity,
Secchi, microcystin, phycocyanin).

Input: processed('wqp_discrete') from the WQP workstream (``--source raw`` reads its raw parse
output instead, raw('wqp')/results + stations, when the processed table is not ready).

Rules (docs/datasheets/s2_wqp_matchups.md):
  * surface samples only: depth <= 1 m or unspecified (Secchi exempt); QC activities, loggers and
    rejected results dropped; replicates -> median per site, local date, characteristic, unit;
  * site water test: JRC GSW occurrence >= 50 % at the site or a pixel with occurrence >= 50 %
    within 60 m (``jrc_water_ok``); lakes/reservoirs/rivers/estuaries flagged (``site_class``);
  * every S2 acquisition within +-``window`` days covering the site is found (ES C1 first, MPC
    fallback); SCL screened; usable ones (>= 5 SCL-valid pixels in the centre 7x7) get a chip;
  * one row per (sample, acquisition): ``dt_days``, ``win_same_day``, ``win_pm1``, ``win_pm3``.

Outputs: processed('s2_wqp_matchups')/{samples,sites,matchups}.parquet, coverage.json (``--out-name`` selects
another processed/<name>, e.g. ``s2_wqp_matchups_hist`` for 2015-2023); chips: processed('s2_chips')/wqp.

    python -B scripts/data/build_s2_wqp_matchups.py --start 2015-06-23 --end 2025-12-31
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
from sentinel.data.sources import stac as S  # noqa: E402

log = logging.getLogger("s2_wqp")
TARGET_PRIORITY = {"microcystin": 0, "phycocyanin": 1, "chla": 2, "secchi": 3, "tss": 4, "ssc": 5,
                   "turbidity": 6}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["processed", "raw"], default="processed")
    ap.add_argument("--start", default="2015-06-23")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--states", nargs="*", default=None, help="postal codes; default all")
    ap.add_argument("--targets", nargs="*", default=list(M.WQP_TARGETS))
    ap.add_argument("--max-chips", type=int, default=0, help="cap on full chips (priority order)")
    ap.add_argument("--read-workers", type=int, default=8)
    ap.add_argument("--scene-workers", type=int, default=4)
    ap.add_argument("--campaign", default="wqp")
    ap.add_argument("--out-name", default="s2_wqp_matchups",
                    help="processed/<name> for the tables and manifest name (e.g. s2_wqp_matchups_hist for a "
                         "2015-2023 build next to the 2024-2025 national table); chips always go to s2_chips/<campaign>")
    a = ap.parse_args()
    out = paths.processed(a.out_name)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(out / "build.log"), logging.StreamHandler()])
    logging.getLogger("rasterio").setLevel(logging.WARNING)
    res, stations, used = M.load_wqp(a.source)
    log.info("WQP %s: %d result rows, %s stations, files=%d", a.source, len(res),
             None if stations is None else len(stations), len(used))
    smp = M.wqp_surface_samples(res, stations)
    del res
    smp = smp[smp.target.isin(a.targets) & smp.date.between(pd.Timestamp(a.start), pd.Timestamp(a.end))]
    if a.states and "state" in smp.columns:
        smp = smp[smp.state.isin(a.states)]
    smp = smp[smp.lat.between(-90, 90) & smp.lon.between(-180, 180)].copy()
    smp["point_key"] = [M.point_key(lo, la) for lo, la in zip(smp.lon, smp.lat)]
    smp["sample_key"] = (smp.site_id + "|" + smp.date.dt.strftime("%Y-%m-%d") + "|" + smp.target + "|" +
                         smp.characteristic + "|" + smp.unit)
    log.info("surface samples: %d (%s), sites %d", len(smp), smp.target.value_counts().to_dict(),
             smp.site_id.nunique())
    sites = smp.drop_duplicates("point_key")[["point_key", "site_id", "lat", "lon", "site_class"]]
    jrc = M.site_jrc(sites, paths.interim("s2_wqp") / "site_jrc.parquet", logger=log.info)
    sites = sites.merge(jrc, on="point_key", how="left")
    sites["jrc_water_ok"] = sites.jrc_water_ok.astype("boolean").fillna(False).astype(bool)
    sites.to_parquet(out / "sites.parquet", index=False)
    smp = smp.merge(sites[["point_key", "jrc_water_ok", "jrc_occ_point", "jrc_dist_water_m"]], on="point_key")
    smp.to_parquet(out / "samples.parquet", index=False, compression="zstd")
    water = smp[smp.jrc_water_ok]
    log.info("samples at JRC-water sites: %d of %d (sites %d of %d)", len(water), len(smp),
             water.point_key.nunique(), smp.point_key.nunique())
    # planning: dates within +-window of each sample
    sd = water.drop_duplicates(["point_key", "date"])[["point_key", "lon", "lat", "date"]]
    need: dict[str, list] = {}
    for r in sd.itertuples():
        for d in M.date_range_strings(r.date, a.window, a.window):
            need.setdefault(d, []).append((r.point_key, r.lon, r.lat))
    pbd = {d: pd.DataFrame(v, columns=["point_key", "lon", "lat"]).drop_duplicates("point_key")
           for d, v in need.items()}
    cand, cat = M.plan_s2_candidates(pbd, paths.interim("s2_scene_catalog") / a.campaign, logger=log.info)
    cand = cand.drop_duplicates(["point_key", "item_id"])
    cand["request_key"] = cand.item_id + "|" + cand.point_key
    cand = cand.merge(sites[["point_key", "lon", "lat"]], on="point_key")
    scenes = M.scenes_from_catalog(cat)
    log.info("candidates: %d (point, scene) pairs, %d scenes", len(cand), len(scenes))
    writer = C.ChipWriter(paths.processed("s2_chips") / a.campaign, layers=C.S2_STORE_LAYERS, dtype="int16",
                          semantics=C.S2_SEMANTICS)
    ex = M.S2Extractor(writer, read_workers=a.read_workers, scene_workers=a.scene_workers, logger=log.info)
    scr = ex.screen_scl(cand[["request_key", "item_id", "lon", "lat"]], scenes,
                        paths.interim("s2_screen") / a.campaign).drop_duplicates("request_key", keep="last")
    # sample x acquisition table
    mt = water[["sample_key", "point_key", "date", "target"]].merge(
        cand[["point_key", "item_id", "request_key", "source"]], on="point_key")
    mt = mt.merge(scr.drop(columns=["item_id"]), on="request_key", how="left")
    st = pd.to_datetime(mt.sensing_time_utc, utc=True, errors="coerce")
    mt = pd.concat([mt, M.window_columns(mt.date, st)], axis=1)
    mt = mt[mt.abs_dt_days <= a.window]
    mt["usable"] = (mt.screen_status == "ok") & ~mt.nodata_centre.astype("boolean").fillna(True).astype(bool) & \
        (mt.scl_c7_n_valid >= 5)
    req = mt[mt.usable].merge(water[["sample_key", "site_class"]].drop_duplicates("sample_key"), on="sample_key")
    # Priority tiers (documented in s2_wqp_matchups.md): 0 = rare toxin/pigment targets or the 2024+
    # test years (beyond AquaCorr), 1 = lakes/reservoirs, 2 = everything else; then |dt|, then target.
    rare = req.target.isin(["microcystin", "phycocyanin"]) | (req.date >= pd.Timestamp("2024-01-01"))
    req["tier"] = np.where(rare, 0, np.where(req.site_class == "lake_reservoir", 1, 2))
    req["prio"] = req.target.map(TARGET_PRIORITY).fillna(9)
    req = req.sort_values(["tier", "abs_dt_days", "prio"]).drop_duplicates("request_key")
    if a.max_chips:
        req = req.head(a.max_chips)
    log.info("chip requests by tier: %s", req.tier.value_counts().to_dict())
    req = req.merge(cand[["request_key", "lon", "lat"]], on="request_key")
    log.info("full chips requested: %d", len(req))
    ex.run(req[["request_key", "item_id", "lon", "lat", "point_key"]], scenes)
    ex.close()
    idx = C.consolidate(writer.root)
    idx["offset_suspect"] = C.recompute_offset_suspect(idx)
    keep = [c for c in idx.columns if c not in ("lon", "lat", "point_key", "item_id", "sensing_time_utc",
                                                 "platform", "scene_cloud_cover")]
    mt = mt.merge(idx[keep], on="request_key", how="left", suffixes=("", "_chip"))
    mt["chip_ok"] = mt.status == "ok"
    mt["qc_water_c3"] = mt.chip_ok & (mt.scl_c3_n_water_chip >= 5)
    mt["qc_pass"] = mt.qc_water_c3 & (mt.offset_suspect.fillna("") == "")
    dest = out / "matchups.parquet"
    mt.to_parquet(dest, index=False, compression="zstd")
    cov = {"source": a.source, "input_files": [str(f) for f in used], "n_samples": int(len(smp)),
           "n_samples_jrc_water": int(len(water)), "n_sites": int(smp.point_key.nunique()),
           "n_sites_jrc_water": int(water.point_key.nunique()),
           "samples_by_target": smp.target.value_counts().to_dict(),
           "site_class": sites.site_class.value_counts().to_dict(),
           "date_range": [str(smp.date.min()), str(smp.date.max())],
           "n_candidates": int(len(cand)), "n_chips_ok": int((idx.status == "ok").sum()),
           "extraction_stats": ex.stats,
           "offset_group_check": C.offset_group_check(idx).to_dict("records") if (idx.status == "ok").any() else []}
    for w, col in ((0, "win_same_day"), (1, "win_pm1"), (3, "win_pm3")):
        sub = mt[mt[col] & mt.qc_pass]
        cov[f"samples_qc_pass_{col}"] = sub.groupby("target").sample_key.nunique().to_dict()
    (out / "coverage.json").write_text(json.dumps(cov, indent=1, default=str))
    log.info("coverage: %s", cov)
    write_manifest(a.out_name, files=[dest, out / "samples.parquet", out / "sites.parquet",
                                            out / "coverage.json", writer.root / "index.parquet"],
                   sources=[str(f) for f in used] + [S.EARTH_SEARCH_URL, S.PLANETARY_COMPUTER_URL],
                   description="WQP surface samples (chl-a, TSS, SSC, turbidity, Secchi, microcystin, phycocyanin) "
                               f"matched to every S2 L2A acquisition within +-{a.window} d; SCL-screened chips.",
                   license="WQP (public); Copernicus Sentinel data (free, full and open); JRC GSW",
                   synthetic=False, stats=json.loads(json.dumps(cov, default=str)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
