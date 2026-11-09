#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Extract SENTINEL S2 L2A chips for the selected AquaCorr-Global S2 matchups.

For every selected AquaCorr row (processed('aquacorr_global')/s2_chip_requests.parquet) the *same*
acquisition and MGRS tile that ACOLITE used is located in Earth Search ``sentinel-2-c1-l2a``
(tile + platform + granule sensing time within 120 s); acquisitions missing there (2015-2018,
most of 2022) are taken from Planetary Computer ``sentinel-2-l2a`` (tile + platform + date; the
datatake start must precede the granule time by < 40 min; exact time verified afterwards from the
granule XML). Chips: 64x64 at 10 m, 12 bands + AOT + SCL + JRC (see docs/datasheets/s2_chips.md).

Outputs:
  processed('s2_chips')/aquacorr/           chip store (shards + index)
  processed('aquacorr_global')/aquacorr_s2_l2a_matchups.parquet   AquaCorr rows + chip metadata

    python -B scripts/data/build_aquacorr_s2_chips.py --stage plan
    python -B scripts/data/build_aquacorr_s2_chips.py --stage extract   # resume-safe
    python -B scripts/data/build_aquacorr_s2_chips.py --stage join
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths  # noqa: E402
from sentinel.core.manifest import write_manifest  # noqa: E402
from sentinel.data import chips as C  # noqa: E402
from sentinel.data import matchups as M  # noqa: E402
from sentinel.data.sources import aquacorr as A  # noqa: E402
from sentinel.data.sources import stac as S  # noqa: E402

log = logging.getLogger("aquacorr_chips")
CAMPAIGN = "aquacorr"


def _match_designated(r, scenes: list[S.S2Scene], source: str):
    """Item for the AquaCorr tile + platform + acquisition time (ES: granule time within 120 s;
    MPC: datatake start precedes the granule time by -2..40 min)."""
    best = None
    for sc in scenes:
        if sc.tile != r.mgrs_tile or sc.platform != r.sensor:
            continue
        lead = (r.scene_sensing_time - pd.Timestamp(sc.datetime_item)).total_seconds()
        ok = abs(lead) <= 120 if source == S.ES else -120 <= lead <= 2400
        if ok and (best is None or abs(lead) < abs(best[0])):
            best = (lead, sc)
    return best


def _match_alternate(r, scenes: list[S.S2Scene], orbit: int | None, source: str):
    """Another tile of the same acquisition (platform, date, relative orbit) containing the point."""
    best = None
    for sc in scenes:
        if sc.platform != r.sensor or (orbit is not None and sc.relative_orbit != orbit):
            continue
        if orbit is None:  # fall back to time proximity when the designated item is unknown
            lead = (r.scene_sensing_time - pd.Timestamp(sc.datetime_item)).total_seconds()
            if not (abs(lead) <= 300 if source == S.ES else -300 <= lead <= 2700):
                continue
        d = S.point_interior_distance(sc.geometry, r.lon, r.lat)
        if d > 0 and (best is None or d > best[0]):
            best = (d, sc)
    return best


def plan(req: pd.DataFrame, cat_dir: Path, workers: int = 4) -> pd.DataFrame:
    """Resolve each chip request to one STAC item (same acquisition as ACOLITE); cached per date.

    match_kind: 'designated' (AquaCorr tile contains the point), 'alternate_tile' (the AquaCorr tile's
    raster does not contain the point - an AquaCorr defect - so another MGRS tile of the same
    acquisition is used), 'none'."""
    cat_dir.mkdir(parents=True, exist_ok=True)
    req = req.copy()
    req["date"] = pd.to_datetime(req.date_insitu).dt.strftime("%Y-%m-%d")
    cols = ["chip_request_key", "item_id", "source", "match_kind", "point_in_designated_tile",
            "plan_time_diff_s", "item_json"]

    def one(date: str, g_all: pd.DataFrame) -> pd.DataFrame:
        f = cat_dir / f"{date}.parquet"
        cached = pd.read_parquet(f) if f.exists() else None
        g = g_all if cached is None else g_all[~g_all.chip_request_key.isin(cached.chip_request_key)]
        if g.empty:
            return cached[cached.chip_request_key.isin(g_all.chip_request_key)]
        dtr = f"{date}T00:00:00Z/{date}T23:59:59Z"
        rows, js = [], {}

        def search(source, tiles=None, pts=None):
            coll = S.S2_ES_C1 if source == S.ES else S.S2_MPC
            items = []
            if tiles:
                key = "grid:code" if source == S.ES else "s2:mgrs_tile"
                vals = [f"MGRS-{t}" for t in tiles] if source == S.ES else list(tiles)
                for k in range(0, len(vals), 50):
                    items += S.stac_search(source, collections=[coll], datetime=dtr,
                                           query={key: {"in": vals[k:k + 50]}})
            if pts:
                for k in range(0, len(pts), 40):
                    items += S.stac_search(source, collections=[coll], datetime=dtr,
                                           intersects={"type": "MultiPoint", "coordinates": pts[k:k + 40]})
            for i in items:
                js[i["id"]] = i
            sc = [S.parse_s2_item(i, source) for i in {i["id"]: i for i in items}.values()]
            return S.dedupe_mpc_reprocessing(sc) if source == S.MPC else sc

        def add(r, sc, source, kind, inside, dt_s):
            rows.append({"chip_request_key": r.chip_request_key, "item_id": sc.id if sc else None,
                         "source": source if sc else None, "match_kind": kind,
                         "point_in_designated_tile": inside, "plan_time_diff_s": dt_s,
                         "item_json": json.dumps(js[sc.id], separators=(",", ":")) if sc else None})

        pending = list(g.itertuples())
        for source in (S.ES, S.MPC):
            if not pending:
                break
            scenes = search(source, tiles=sorted({r.mgrs_tile for r in pending}))
            need_alt, still = [], []
            for r in pending:
                m = _match_designated(r, scenes, source)
                if m is None:
                    still.append(r)
                    continue
                inside = S.point_interior_distance(m[1].geometry, r.lon, r.lat) > 0
                if inside:
                    add(r, m[1], source, "designated", True, m[0])
                else:
                    need_alt.append((r, m[1].relative_orbit, m[0]))
            if need_alt:
                alt_scenes = search(source, pts=[[float(r.lon), float(r.lat)] for r, _, _ in need_alt])
                for r, orbit, dt_s in need_alt:
                    m = _match_alternate(r, alt_scenes, orbit, source)
                    if m is not None:
                        add(r, m[1], source, "alternate_tile", False, dt_s)
                    else:
                        add(r, None, source, "none", False, dt_s)
            pending = still
        for r in pending:
            add(r, None, None, "none", None, np.nan)
        out = pd.DataFrame(rows, columns=cols)
        if cached is not None:
            out = pd.concat([cached, out], ignore_index=True)
        tmp = f.with_suffix(".tmp")
        out.to_parquet(tmp, index=False, compression="zstd")
        tmp.replace(f)
        return out[out.chip_request_key.isin(g_all.chip_request_key)]

    groups = list(req.groupby("date"))
    res = []
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(one, d, g): d for d, g in groups}
        for n, fu in enumerate(as_completed(futs), 1):
            try:
                res.append(fu.result())
            except Exception as e:  # noqa: BLE001 — rerun resumes
                log.error("plan %s failed: %r", futs[fu], e)
            if n % 100 == 0:
                log.info("planned %d/%d dates", n, len(groups))
    return pd.concat(res, ignore_index=True) if res else pd.DataFrame(columns=cols)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["plan", "extract", "join", "all"], default="all")
    ap.add_argument("--limit", type=int, default=0, help="debug: first N chip requests")
    ap.add_argument("--read-workers", type=int, default=8)
    ap.add_argument("--scene-workers", type=int, default=3)
    ap.add_argument("--retry-failed", action="store_true")
    a = ap.parse_args()
    proc = paths.processed("aquacorr_global")
    store_dir = paths.processed("s2_chips") / CAMPAIGN
    cat_dir = paths.interim("s2_scene_catalog") / CAMPAIGN
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(proc / "chips.log"), logging.StreamHandler()])
    logging.getLogger("rasterio").setLevel(logging.WARNING)
    sel = pd.read_parquet(proc / "s2_chip_requests.parquet")
    req = sel.drop_duplicates("chip_request_key").copy()
    if a.limit:
        req = req.head(a.limit)
    log.info("%d chip requests (%d AquaCorr rows)", len(req), len(sel))

    if a.stage in ("plan", "all", "extract"):
        planned = plan(req, cat_dir)
        planned.drop(columns="item_json").to_parquet(proc / "s2_chip_plan.parquet", index=False)
        log.info("plan: %d resolved (%s), kinds %s, %d unresolved", planned.item_id.notna().sum(),
                 planned.source.value_counts().to_dict(), planned.match_kind.value_counts().to_dict(),
                 planned.item_id.isna().sum())
    if a.stage in ("extract", "all"):
        planned = planned[planned.item_id.notna()]
        scenes = {}
        for r in planned.drop_duplicates("item_id").itertuples():
            scenes[r.item_id] = S.parse_s2_item(json.loads(r.item_json), r.source)
        rq = req.merge(planned[["chip_request_key", "item_id", "plan_time_diff_s", "match_kind",
                                "point_in_designated_tile"]], on="chip_request_key")
        rq = rq.rename(columns={"chip_request_key": "request_key"})
        rq = rq[["request_key", "item_id", "lon", "lat", "chip_point_key", "scene_id", "mgrs_tile",
                 "scene_sensing_time", "plan_time_diff_s", "match_kind", "point_in_designated_tile"]].rename(
            columns={"scene_id": "aquacorr_scene_id", "mgrs_tile": "aquacorr_tile",
                     "scene_sensing_time": "aquacorr_sensing_time"})
        rq["aquacorr_sensing_time"] = rq.aquacorr_sensing_time.astype(str)
        writer = C.ChipWriter(store_dir, layers=C.S2_STORE_LAYERS, dtype="int16",
                              semantics=C.S2_SEMANTICS, retry_failed=a.retry_failed)
        ex = M.S2Extractor(writer, read_workers=a.read_workers, scene_workers=a.scene_workers,
                           screen=False, logger=log.info)
        ex.run(rq, scenes)
        ex.close()
        C.consolidate(store_dir)
    if a.stage in ("join", "all"):
        idx = C.consolidate(store_dir)
        idx["offset_suspect"] = C.recompute_offset_suspect(idx)  # current flag definition
        group_check = C.offset_group_check(idx)
        group_check.to_csv(proc / "offset_group_check.csv", index=False)
        idx = idx.rename(columns={"request_key": "chip_request_key"})
        out = sel.merge(idx, on="chip_request_key", how="left", suffixes=("", "_chip"))
        st = pd.to_datetime(out.sensing_time_utc, utc=True, errors="coerce")
        # ES C1 granule times agree with ACOLITE's to < 1 s; MPC items of baseline 02.xx carry an
        # MTD_TL SENSING_TIME up to ~10 min earlier (metadata convention), so the acquisition is
        # identified by platform + UTC date + relative orbit + tile/footprint, not by the time.
        out["l2a_minus_aquacorr_time_s"] = (st - out.scene_sensing_time).dt.total_seconds()
        out["l2a_same_acquisition"] = (out.platform == out.sensor) & \
            (st.dt.strftime("%Y-%m-%d") == pd.to_datetime(out.date_insitu).dt.strftime("%Y-%m-%d")) & \
            out.match_kind.isin(["designated", "alternate_tile"])
        out["status"] = out.status.fillna("not_extracted")
        ok = out.status == "ok"
        out["qc_water_c3"] = ok & (out.scl_c3_n_water >= 5)
        out["qc_jrc_water"] = ok & out.jrc_water_ok.astype("boolean").fillna(False).astype(bool)
        out["qc_pass"] = out.qc_water_c3 & out.qc_jrc_water & out.l2a_same_acquisition & \
            (out.offset_suspect.fillna("") == "")
        dest = proc / "aquacorr_s2_l2a_matchups.parquet"
        out.to_parquet(dest, index=False, compression="zstd")
        stats = {"rows": int(len(out)), "status": out.status.value_counts().to_dict(),
                 "ok": int(ok.sum()), "qc_water_c3": int(out.qc_water_c3.sum()),
                 "qc_jrc_water": int(out.qc_jrc_water.sum()), "qc_pass": int(out.qc_pass.sum()),
                 "stac_source": out.stac_source.value_counts().to_dict(),
                 "baselines": out.processing_baseline.value_counts().to_dict(),
                 "same_acquisition": int(out.l2a_same_acquisition.sum()),
                 "scl_c3_cloud_any": int((ok & ((out.scl_c3_cloud_medium + out.scl_c3_cloud_high
                                                 + out.scl_c3_thin_cirrus) > 0)).sum()),
                 "qc_pass_values": {v: int(out.loc[out.qc_pass, f"{v}_value"].notna().sum())
                                    for v in ("chla", "tss", "sdd", "turbidity")},
                 "offset_suspect": out.loc[ok, "offset_suspect"].value_counts().to_dict(),
                 "offset_group_check": group_check.to_dict("records")}
        log.info("join: %s", stats)
        files = sorted((store_dir / "shards").glob("*.npz")) + [store_dir / "index.parquet",
                                                               store_dir / "layers.json", dest]
        write_manifest("s2_chips_aquacorr", files=files, sources=[S.EARTH_SEARCH_URL, S.PLANETARY_COMPUTER_URL],
                       description="Sentinel-2 L2A 64x64 10 m chips (12 bands + AOT + SCL + JRC) for AquaCorr-Global "
                                   "S2 matchups (US + Europe, passed_qc, priority-selected), same acquisition/tile "
                                   "as ACOLITE; BOA offset harmonised; joined matchup table.",
                       license="Copernicus Sentinel data (free, full and open); JRC GSW (no restriction); "
                               "AquaCorr CC-BY-4.0",
                       synthetic=False, stats=json.loads(json.dumps(stats, default=str)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
