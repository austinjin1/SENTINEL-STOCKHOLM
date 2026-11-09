# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Sentinel-2 matchup engine shared by AquaCorr, WQP, the microcystin benchmark and NAIP teachers.

1. *Planning*: ``plan_s2_candidates`` finds every S2 L2A acquisition covering each point on the dates
   it needs (one STAC query per date and region; Earth Search C1 first, Planetary Computer for points
   C1 does not cover that day), and caches the scene catalogue (item JSON) so extraction can resume
   without re-querying.
2. *Extraction*: ``S2Extractor`` reads chips scene by scene: each COG is opened once per scene and
   all windows for that scene are read from it. All network I/O for chips (COG windows, JRC windows,
   metadata XML) goes through one thread pool of ``read_workers`` (<= 8) threads. An optional SCL
   pre-screen reads only the SCL layer first and skips acquisitions that are cloudy/no-data at the
   point (centre 7x7 at 10 m with < ``screen_min_valid`` usable pixels).
3. *Matchup bookkeeping*: ``window_columns`` adds dt_days / same-day / +-1 / +-3 day flags.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import math
import threading
import time
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from . import chips as C
from .sources import stac as S

log = logging.getLogger("sentinel.matchups")

# Regions used for date-wise bbox STAC queries (points outside all boxes get their own bbox).
QUERY_REGIONS = {
    "conus_w": (-125.0, 24.0, -100.0, 49.5),
    "conus_e": (-100.0, 24.0, -66.5, 49.5),
    "alaska": (-170.0, 51.0, -129.0, 71.5),
    "hawaii": (-161.0, 18.5, -154.0, 22.5),
    "caribbean": (-68.0, 17.5, -64.0, 18.8),
    "europe_w": (-25.0, 34.0, 10.0, 72.0),
    "europe_e": (10.0, 34.0, 45.0, 72.0),
}


# --------------------------------------------------------------------------- scene catalogue


def scene_row(sc: S.S2Scene) -> dict[str, Any]:
    """Scene metadata columns written into every chip index row."""
    o = sc.offset
    return {
        "item_id": sc.id, "stac_source": sc.source, "collection": sc.collection, "platform": sc.platform,
        "mgrs_tile": sc.tile, "sensing_time_utc": sc.sensing_time, "item_datetime": sc.datetime_item,
        "processing_baseline": sc.baseline, "offset_dn": o.offset_dn if o else None,
        "offset_source": o.source if o else None, "offset_already_applied": o.already_applied if o else None,
        "offset_consistent": o.consistent if o else None, "offset_xml_verified": sc.offset_xml_verified,
        "scene_cloud_cover": sc.cloud_cover, "sun_zenith": sc.sun_zenith, "sun_azimuth": sc.sun_azimuth,
        "view_zenith": sc.view_zenith, "view_azimuth": sc.view_azimuth, "relative_orbit": sc.relative_orbit,
        "product_uri": sc.product_uri, "epsg": sc.epsg,
    }


_KEEP_ASSETS = {S.ES: {l.es_key for l in S.S2_LAYERS} | {"product_metadata", "granule_metadata"},
                S.MPC: {l.mpc_key for l in S.S2_LAYERS} | {"product-metadata", "granule-metadata"}}


def trim_item(it: dict, source: str) -> dict:
    """STAC item reduced to what chip extraction needs (properties, geometry, chip-layer and
    metadata assets); hrefs stay unsigned."""
    keep = _KEEP_ASSETS[source]
    return {"type": it.get("type"), "id": it["id"], "collection": it.get("collection"), "bbox": it.get("bbox"),
            "geometry": it["geometry"], "properties": it["properties"],
            "assets": {k: {kk: vv for kk, vv in a.items() if kk in ("href", "raster:bands", "type")}
                       for k, a in it["assets"].items() if k in keep}}


def items_to_catalog(items: Iterable[tuple[str, dict]]) -> pd.DataFrame:
    """[(source, item dict)] -> catalogue rows (trimmed item JSON, hrefs unsigned)."""
    rows = []
    for source, it in items:
        it = trim_item(it, source)
        sc = S.parse_s2_item(it, source)
        rows.append({"item_id": sc.id, "source": source, "tile": sc.tile, "platform": sc.platform,
                     "date": sc.datetime_item[:10], "baseline": sc.baseline,
                     "relative_orbit": sc.relative_orbit, "cloud_cover": sc.cloud_cover,
                     "item_json": json.dumps(it, separators=(",", ":"))})
    return pd.DataFrame(rows)


def scenes_from_catalog(cat: pd.DataFrame) -> dict[str, S.S2Scene]:
    return {r.item_id: S.parse_s2_item(json.loads(r.item_json), r.source) for r in cat.itertuples()}


def _bbox_of(lon: np.ndarray, lat: np.ndarray, pad: float = 0.02) -> list[float]:
    return [float(lon.min() - pad), float(lat.min() - pad), float(lon.max() + pad), float(lat.max() + pad)]


def _region_groups(pts: pd.DataFrame) -> list[pd.DataFrame]:
    out, rest = [], np.ones(len(pts), bool)
    for x0, y0, x1, y1 in QUERY_REGIONS.values():
        m = rest & (pts.lon.values >= x0) & (pts.lon.values < x1) & (pts.lat.values >= y0) & (pts.lat.values < y1)
        if m.any():
            out.append(pts[m])
            rest &= ~m
    if rest.any():  # elsewhere: 5-degree cells
        other = pts[rest]
        cell = (np.floor(other.lon / 5) * 1000 + np.floor(other.lat / 5)).astype(int)
        out.extend(g for _, g in other.groupby(cell))
    return out


def _query_points(source: str, date: str, pts: pd.DataFrame, max_points_multipoint: int = 40) -> list[dict]:
    """Items on ``date`` covering any of ``pts``; MultiPoint intersects for few points, else bbox."""
    items: dict[str, dict] = {}
    for grp in _region_groups(pts):
        if len(grp) <= max_points_multipoint:
            geom = {"type": "MultiPoint", "coordinates": [[float(a), float(b)] for a, b in
                                                            zip(grp.lon.values, grp.lat.values)]}
            kw = {"intersects": geom}
        else:
            kw = {"bbox": _bbox_of(grp.lon.values, grp.lat.values)}
        coll = S.S2_ES_C1 if source == S.ES else S.S2_MPC
        for it in S.stac_search(source, collections=[coll],
                                datetime=f"{date}T00:00:00Z/{date}T23:59:59Z", **kw):
            items[it["id"]] = it
    return list(items.values())


def _covers(items: list[dict], pts: pd.DataFrame) -> dict[str, list[str]]:
    """point key -> ids of items whose data footprint contains the point."""
    from shapely import STRtree, points
    from shapely.geometry import shape
    if not items:
        return {}
    geoms = [shape(it["geometry"]) for it in items]
    tree = STRtree(geoms)
    pg = points(pts.lon.values, pts.lat.values)
    idx_pt, idx_geom = tree.query(pg, predicate="within")
    out: dict[str, list[str]] = defaultdict(list)
    keys = pts.point_key.values
    for i, j in zip(idx_pt, idx_geom):
        out[keys[i]].append(items[j]["id"])
    return out


def plan_s2_candidates(points_by_date: dict[str, pd.DataFrame], catalog_dir: Path, *,
                       workers: int = 4, log_every: int = 50,
                       logger: Callable[[str], None] = log.info) -> tuple[pd.DataFrame, pd.DataFrame]:
    """For each UTC date, find S2 acquisitions covering the points (columns point_key, lon, lat).

    Earth Search C1 is queried first; points it does not cover that day are queried on Planetary
    Computer (reprocessing duplicates resolved to the highest baseline). Results are cached per date
    in ``catalog_dir`` (``<date>.parquet``: catalogue + candidate pairs) so planning is resume-safe.
    Returns (candidates[point_key, date, item_id, source], catalogue).
    """
    catalog_dir = Path(catalog_dir)
    catalog_dir.mkdir(parents=True, exist_ok=True)

    def one(date: str):
        f = catalog_dir / f"{date}.parquet"
        fc = catalog_dir / f"{date}.cand.parquet"
        fq = catalog_dir / f"{date}.queried.json"
        pts = points_by_date[date].drop_duplicates("point_key")
        if f.exists() and fc.exists() and fq.exists():
            queried = set(json.loads(fq.read_text()))
            if set(pts.point_key.astype(str)) <= queried:  # cached answer covers every point
                return date, pd.read_parquet(fc), pd.read_parquet(f)
        es_items = _query_points(S.ES, date, pts)
        cov = _covers(es_items, pts)
        rest = pts[~pts.point_key.isin(cov.keys())]
        mpc_items = _query_points(S.MPC, date, rest) if len(rest) else []
        if mpc_items:
            sc = S.dedupe_mpc_reprocessing([S.parse_s2_item(i, S.MPC) for i in mpc_items])
            keep = {s.id for s in sc}
            mpc_items = [i for i in mpc_items if i["id"] in keep]
        cov_m = _covers(mpc_items, rest)
        used = {i for v in cov.values() for i in v} | {i for v in cov_m.values() for i in v}
        cat = items_to_catalog([(S.ES, i) for i in es_items if i["id"] in used] +
                               [(S.MPC, i) for i in mpc_items if i["id"] in used])
        rows = [(k, date, i, S.ES) for k, v in cov.items() for i in v] + \
               [(k, date, i, S.MPC) for k, v in cov_m.items() for i in v]
        cand = pd.DataFrame(rows, columns=["point_key", "date", "item_id", "source"])
        cat.to_parquet(f, index=False, compression="zstd")
        cand.to_parquet(fc, index=False, compression="zstd")
        fq.write_text(json.dumps(sorted(pts.point_key.astype(str))))
        return date, cand, cat

    cands, cats = [], []
    dates = sorted(points_by_date)
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(one, d): d for d in dates}
        for n, fut in enumerate(as_completed(futs), 1):
            d = futs[fut]
            try:
                _, cand, cat = fut.result()
                cands.append(cand)
                cats.append(cat)
            except Exception as e:  # noqa: BLE001 — logged; rerun resumes
                logger(f"planning failed for {d}: {e!r}")
            if n % log_every == 0:
                logger(f"planned {n}/{len(dates)} dates")
    cand = pd.concat(cands, ignore_index=True) if cands else pd.DataFrame(
        columns=["point_key", "date", "item_id", "source"])
    cat = pd.concat(cats, ignore_index=True).drop_duplicates("item_id") if cats else pd.DataFrame()
    return cand, cat


# ------------------------------------------------------------------------------- extraction


class _LRU(OrderedDict):
    def __init__(self, maxsize: int):
        super().__init__()
        self.maxsize = maxsize
        self.lock = threading.Lock()

    def get_or(self, key, fn):
        with self.lock:
            if key in self:
                self.move_to_end(key)
                return self[key]
        val = fn()
        with self.lock:
            self[key] = val
            if len(self) > self.maxsize:
                self.popitem(last=False)
        return val


class S2Extractor:
    """Scene-grouped S2 chip extraction into a ``ChipWriter`` (layers ``chips.S2_STORE_LAYERS``)."""

    def __init__(self, writer: C.ChipWriter, *, read_workers: int = 8, scene_workers: int = 3,
                 screen: bool = False, screen_min_valid: int = 5, size: int = 64,
                 logger: Callable[[str], None] = log.info):
        if read_workers > 8:
            raise ValueError("read concurrency is capped at 8")
        self.w = writer
        self.read_pool = ThreadPoolExecutor(read_workers, thread_name_prefix="cogread")
        self.scene_workers = scene_workers
        self.screen = screen
        self.screen_min_valid = screen_min_valid
        self.size = size
        self.jrc = S.JrcOccurrence()
        self._jrc_cache = _LRU(30000)
        self.log = logger
        self.stats: dict[str, int] = defaultdict(int)
        self._slock = threading.Lock()
        self.layers = [l for l in S.S2_LAYERS]
        self.read_timeout = 300.0
        self._last_progress = time.time()
        self._wd_stop = True

    def _count(self, k: str, n: int = 1) -> None:
        with self._slock:
            self.stats[k] += n
            if not k.startswith("error"):  # failures are not progress (a hung pool only produces errors)
                self._last_progress = time.time()

    def _start_watchdog(self, stall_s: float) -> None:
        """Exit the process (code 3) if nothing completes for ``stall_s`` seconds: a hung HTTP read
        must not block a resume-safe job forever; the caller's shell loop restarts it."""
        import os
        self._last_progress = time.time()
        self._wd_stop = False

        def watch():
            while not self._wd_stop:
                time.sleep(30)
                if time.time() - self._last_progress > stall_s:
                    self.log(f"WATCHDOG: no progress for {stall_s:.0f}s; flushing and exiting for restart")
                    try:
                        self.w.flush()
                    finally:
                        os._exit(3)
        threading.Thread(target=watch, daemon=True, name="watchdog").start()
        if threading.current_thread() is threading.main_thread():
            import signal

            def on_term(signum, frame):  # lossless stop: write buffered chips, then exit
                self.log("SIGTERM: flushing buffered chips before exit")
                try:
                    self.w.flush()
                finally:
                    os._exit(143)
            signal.signal(signal.SIGTERM, on_term)

    def _jrc(self, lon: float, lat: float):
        key = (round(lon, 5), round(lat, 5))

        def fetch():
            try:
                w = self.read_pool.submit(self.jrc.window, lon, lat).result(timeout=self.read_timeout)
            except Exception as e:  # noqa: BLE001
                return None, {"jrc_error": repr(e)[:120]}
            if w is None:
                return None, {"jrc_error": "no_tile"}
            occ, tr = w
            return (occ, tr), S.jrc_point_stats(occ, tr, lon, lat)
        return self._jrc_cache.get_or(key, fetch)

    def _read_layers(self, scene: S.S2Scene, layers: Sequence[S.Layer], pts: list[tuple[float, float]]):
        futs = {l.name: self.read_pool.submit(S.s2_band_windows, scene.assets[l.name], l, pts, self.size)
                for l in layers}
        return {name: f.result(timeout=self.read_timeout) for name, f in futs.items()}

    def _scene_task(self, scene: S.S2Scene, reqs: pd.DataFrame) -> None:
        keys = reqs.request_key.tolist()
        meta_cols = [c for c in reqs.columns if c not in ("request_key",)]
        try:
            if scene.source == S.MPC or scene.sensing_time is None:
                self.read_pool.submit(S.complete_scene_metadata, scene).result(timeout=self.read_timeout)
            srow = scene_row(scene)
            if scene.offset is None or not scene.offset.consistent:
                for k, (_, r) in zip(keys, reqs.iterrows()):
                    self.w.add(k, None, {**{c: r[c] for c in meta_cols}, **srow}, status="offset_inconsistent")
                self._count("offset_inconsistent", len(keys))
                return
            missing = [l.name for l in self.layers if l.name not in scene.assets]
            if missing:
                for k, (_, r) in zip(keys, reqs.iterrows()):
                    self.w.add(k, None, {**{c: r[c] for c in meta_cols}, **srow},
                               status=f"missing_assets:{','.join(missing)}")
                self._count("missing_assets", len(keys))
                return
            pts = list(zip(reqs.lon.astype(float), reqs.lat.astype(float)))
            scl_layer = [l for l in self.layers if l.name == "SCL"]
            other = [l for l in self.layers if l.name != "SCL"]
            got = self._read_layers(scene, scl_layer, pts)
            scl_arrays, grids, _info = got["SCL"]
            keep = []
            for i, (k, (_, r)) in enumerate(zip(keys, reqs.iterrows())):
                scl = np.asarray(scl_arrays[i])
                c = scl[S.centre_slice(self.size, 1), S.centre_slice(self.size, 1)]
                if int(c[0, 0]) == 0:
                    self.w.add(k, None, {**{cc: r[cc] for cc in meta_cols}, **srow}, status="nodata_centre")
                    self._count("nodata_centre")
                    continue
                if self.screen:
                    c7 = scl[S.centre_slice(self.size, 7), S.centre_slice(self.size, 7)]
                    nvalid = int((~np.isin(c7, S.SCL_INVALID)).sum())
                    if nvalid < self.screen_min_valid:
                        self.w.add(k, None, {**{cc: r[cc] for cc in meta_cols}, **srow,
                                             **S.scl_summary(c7, "scl_c7")}, status="screen_invalid")
                        self._count("screen_invalid")
                        continue
                keep.append(i)
            if not keep:
                return
            kpts = [pts[i] for i in keep]
            got2 = self._read_layers(scene, other, kpts)
            for j, i in enumerate(keep):
                k = keys[i]
                r = reqs.iloc[i]
                g = grids[i]
                chip = np.empty((len(C.S2_STORE_LAYERS), self.size, self.size), np.int16)
                for li, name in enumerate(C.S2_STORE_LAYERS):
                    if name == "SCL":
                        chip[li] = np.asarray(scl_arrays[i]).astype(np.int16)
                    elif name == "JRC":
                        continue
                    else:
                        arrs, g2, _ = got2[name]
                        if (g2[j].row0, g2[j].col0, g2[j].x0, g2[j].y0) != (g.row0, g.col0, g.x0, g.y0):
                            raise ValueError(f"grid mismatch {name} {scene.id}")
                        lay = next(l for l in self.layers if l.name == name)
                        off = scene.offset.offset_dn if lay.kind == "reflectance" else 0
                        chip[li] = S.harmonise_dn(arrs[j], off)
                jw, jstats = self._jrc(*pts[i])
                ji = C.S2_STORE_LAYERS.index("JRC")
                if jw is not None:
                    chip[ji] = S.jrc_on_chip_grid(jw[0], jw[1], g).astype(np.int16)
                else:
                    chip[ji] = S.JRC_NODATA
                feats = C.s2_chip_features(chip)
                fy, fx = g.subpixel
                row = {**{c: r[c] for c in meta_cols}, **srow, **jstats, **feats,
                       "chip_crs": g.crs, "chip_left": g.bounds[0], "chip_top": g.bounds[3],
                       "chip_res_m": g.res, "point_row": g.point_row, "point_col": g.point_col,
                       "point_subpix_y": fy, "point_subpix_x": fx,
                       "offset_suspect": C.s2_offset_suspect(feats)}
                self.w.add(k, chip, row, status="ok")
                self._count("ok")
        except Exception as e:  # noqa: BLE001 — record and continue; retry later with retry_failed
            if isinstance(e, TimeoutError):
                # a read thread is hung inside GDAL: the pool is poisoned. Flush finished chips and exit
                # (code 3) so the restart loop resumes with fresh threads; this scene is redone.
                import os
                self.log(f"read timeout in {scene.id}: flushing and exiting for restart")
                try:
                    self.w.flush()
                finally:
                    os._exit(3)
            for k, (_, r) in zip(keys, reqs.iterrows()):
                if not self.w.is_done(k):
                    self.w.add(k, None, {**{c: r[c] for c in meta_cols}, "item_id": scene.id},
                               status=f"error:{type(e).__name__}:{str(e)[:150]}")
            self._count("error", len(keys))

    def screen_scl(self, requests: pd.DataFrame, scenes: dict[str, S.S2Scene], cache: Path, *,
                   log_every_scenes: int = 200, max_points_per_scene_task: int = 128) -> pd.DataFrame:
        """SCL-only pass (1 COG read per scene): centre 3x3 / 7x7 class counts per request, no chips.
        Results are appended to ``cache`` (Parquet parts) so the pass is resume-safe."""
        cache = Path(cache)
        cache.mkdir(parents=True, exist_ok=True)
        parts = sorted(cache.glob("part-*.parquet"))
        done = pd.concat([pd.read_parquet(f) for f in parts], ignore_index=True) if parts else pd.DataFrame()
        done_keys = set(done.request_key) if len(done) else set()
        todo = requests[~requests.request_key.isin(done_keys) & requests.item_id.isin(scenes.keys())]
        groups = []
        for sid, g in todo.groupby("item_id", sort=False):
            for k in range(0, len(g), max_points_per_scene_task):
                groups.append((sid, g.iloc[k:k + max_points_per_scene_task]))
        self.log(f"SCL screening: {len(todo)} requests in {len(groups)} scene tasks ({len(done_keys)} cached)")
        lock = threading.Lock()
        buf: list[dict] = []
        nparts = [len(parts)]

        def flush():
            with lock:
                if not buf:
                    return
                df = C.normalise_columns(pd.DataFrame(buf))
                df.to_parquet(cache / f"part-{nparts[0]:05d}.parquet", index=False)
                nparts[0] += 1
                buf.clear()

        def task(sid, g):
            sc = scenes[sid]
            rows = []
            try:
                if sc.source == S.MPC or sc.sensing_time is None:
                    self.read_pool.submit(S.complete_scene_metadata, sc).result()
                pts = list(zip(g.lon.astype(float), g.lat.astype(float)))
                scl_layer = next(l for l in self.layers if l.name == "SCL")
                arrays, grids, _ = self.read_pool.submit(
                    S.s2_band_windows, sc.assets["SCL"], scl_layer, pts, self.size).result(timeout=self.read_timeout)
                for key, a in zip(g.request_key, arrays):
                    a = np.asarray(a)
                    c7 = a[S.centre_slice(self.size, 7), S.centre_slice(self.size, 7)]
                    c3 = a[S.centre_slice(self.size, 3), S.centre_slice(self.size, 3)]
                    rows.append({"request_key": key, "item_id": sid, "screen_status": "ok",
                                 "sensing_time_utc": sc.sensing_time, "platform": sc.platform,
                                 "scene_cloud_cover": sc.cloud_cover,
                                 "nodata_centre": int(a[self.size // 2, self.size // 2]) == 0,
                                 **S.scl_summary(c3, "scl_c3"), **S.scl_summary(c7, "scl_c7")})
            except Exception as e:  # noqa: BLE001
                rows = [{"request_key": k, "item_id": sid, "screen_status": f"error:{type(e).__name__}"}
                        for k in g.request_key]
            with lock:
                buf.extend(rows)
                big = len(buf) >= 5000
            self._last_progress = time.time()
            if big:
                flush()

        t0 = time.time()
        self._start_watchdog(1200.0)
        with ThreadPoolExecutor(self.scene_workers, thread_name_prefix="scl") as ex:
            futs = [ex.submit(task, sid, g) for sid, g in groups]
            for n, f in enumerate(as_completed(futs), 1):
                f.result()
                if n % log_every_scenes == 0:
                    self.log(f"SCL scenes {n}/{len(groups)} ({n / (time.time() - t0) * 3600:.0f}/h)")
        flush()
        self._wd_stop = True
        parts = sorted(cache.glob("part-*.parquet"))
        return pd.concat([pd.read_parquet(f) for f in parts], ignore_index=True) if parts else pd.DataFrame()

    def run(self, requests: pd.DataFrame, scenes: dict[str, S.S2Scene], *,
            log_every_scenes: int = 100, max_points_per_scene_task: int = 64,
            stall_s: float = 1200.0) -> dict[str, int]:
        """``requests``: columns request_key, item_id, lon, lat (+ passthrough metadata)."""
        todo = requests[~requests.request_key.astype(str).map(self.w.is_done)]
        todo = todo[todo.item_id.isin(scenes.keys())]
        groups = []
        for sid, g in todo.groupby("item_id", sort=False):
            for s in range(0, len(g), max_points_per_scene_task):
                groups.append((sid, g.iloc[s:s + max_points_per_scene_task]))
        self.log(f"extraction: {len(todo)} requests in {len(groups)} scene tasks "
                 f"({len(requests) - len(todo)} already done or without scene)")
        t0 = time.time()
        self._start_watchdog(stall_s)
        with ThreadPoolExecutor(self.scene_workers, thread_name_prefix="scene") as ex:
            futs = [ex.submit(self._scene_task, scenes[sid], g) for sid, g in groups]
            for n, f in enumerate(as_completed(futs), 1):
                f.result()
                if n % log_every_scenes == 0:
                    el = time.time() - t0
                    self.log(f"scenes {n}/{len(groups)} ({n / el * 3600:.0f}/h) stats={dict(self.stats)}")
        self._wd_stop = True
        self.w.flush()
        self.log(f"extraction done in {time.time() - t0:.0f}s stats={dict(self.stats)}")
        return dict(self.stats)

    def close(self) -> None:
        self.read_pool.shutdown(wait=False, cancel_futures=True)
        self.w.close()


# ------------------------------------------------------------------------ matchup bookkeeping


def window_columns(sample_date: pd.Series, image_time_utc: pd.Series) -> pd.DataFrame:
    """dt_days = image UTC date - sample date (calendar days), |dt|, and window membership flags.
    Sample dates are local calendar dates (WQP ActivityStartDate); S2 overpasses are ~10:30 local
    solar time, so the UTC date equals the local date for the Americas and Europe."""
    sd = pd.to_datetime(sample_date).dt.normalize()
    it = pd.to_datetime(image_time_utc, utc=True).dt.tz_localize(None).dt.normalize()
    d = (it - sd).dt.days
    return pd.DataFrame({"dt_days": d, "abs_dt_days": d.abs(), "win_same_day": d.abs() == 0,
                         "win_pm1": d.abs() <= 1, "win_pm3": d.abs() <= 3}, index=sample_date.index)


def point_key(lon: float, lat: float) -> str:
    return f"{lat:.6f}_{lon:.6f}"


def date_range_strings(center: str | dt.date, before: int, after: int) -> list[str]:
    c = pd.Timestamp(center).normalize()
    return [(c + pd.Timedelta(days=k)).strftime("%Y-%m-%d") for k in range(-before, after + 1)]


# ------------------------------------------------------------------------------ WQP samples

# Exact WQX CharacteristicName values (WQP workstream list, sentinel/data/sources/wqp.py) per target.
WQP_TARGETS: dict[str, tuple[str, ...]] = {
    "chla": ("Chlorophyll a", "Chlorophyll a, corrected for pheophytin", "Chlorophyll a, uncorrected for pheophytin",
             "Chlorophyll a, free of pheophytin", "Chlorophyll a - Phytoplankton (suspended)", "Chlorophyll A",
             "Chlorophyll"),
    "tss": ("Total suspended solids", "Total Suspended Particulate Matter"),
    "ssc": ("Suspended Sediment Concentration (SSC)", "Suspended sediment concentration (SSC)"),
    "turbidity": ("Turbidity", "Turbidity Field"),
    "secchi": ("Depth, Secchi disk depth", "Secchi depth", "Water transparency, Secchi disc"),
    "microcystin": ("Microcystin", "Microcystins and nodularins", "Total microcystins plus nodularins",
                    "Microcystins/nodularin", "Dissolved microcystins plus nodularins",
                    "Particulate microcystins/nodularin", "Microcystin LR", "Microcystin RR", "Microcystin YR",
                    "Microcystin LA"),
    "phycocyanin": ("Phycocyanin", "Phycocyanins", "Algae-Phycocyanin"),
}
NAME_TO_TARGET = {n: t for t, names in WQP_TARGETS.items() for n in names}
DEPTH_EXEMPT = ("secchi",)          # Secchi is an optical depth, not a sampling depth
SURFACE_MAX_M = 1.0
_LEN_TO_M = {"m": 1.0, "meters": 1.0, "meter": 1.0, "ft": 0.3048, "feet": 0.3048, "foot": 0.3048,
             "in": 0.0254, "inches": 0.0254, "cm": 0.01, "mm": 0.001}

SITE_CLASSES = (
    ("lake_reservoir", ("lake", "reservoir", "impoundment", "pond")),
    ("river_stream", ("river", "stream", "creek")),
    ("estuary_coastal", ("estuary", "ocean", "bay", "coastal", "sea")),
    ("wetland", ("wetland", "marsh", "swamp")),
    ("canal", ("canal", "ditch", "channel")),
)


def site_class(site_type: pd.Series) -> pd.Series:
    """WQX MonitoringLocationTypeName -> lake_reservoir / river_stream / estuary_coastal / wetland /
    canal / other (first match, case-insensitive)."""
    s = site_type.fillna("").astype(str).str.lower()
    out = pd.Series("other", index=s.index, dtype=object)
    done = pd.Series(False, index=s.index)
    for name, keys in SITE_CLASSES:
        m = ~done & s.str.contains("|".join(keys), regex=True)
        out[m] = name
        done |= m
    out[s == ""] = "unknown"
    return out


def depth_m(value: pd.Series, unit: pd.Series) -> pd.Series:
    """Depth in metres from a WQX depth measure + unit (NaN when either is missing/unknown)."""
    v = pd.to_numeric(value, errors="coerce")
    f = unit.fillna("").astype(str).str.strip().str.lower().map(_LEN_TO_M)
    return v * f


def _col(df: pd.DataFrame, *names: str, default=None) -> pd.Series:
    for n in names:
        if n in df.columns:
            return df[n]
    if default is None:
        raise KeyError(f"none of {names} in columns")
    return pd.Series(default, index=df.index)


def wqp_surface_samples(res: pd.DataFrame, stations: pd.DataFrame | None = None) -> pd.DataFrame:
    """Filter + tidy WQP results to surface optical/toxin samples for satellite matchups.

    Accepts the WQP workstream's tidy result columns (``site_id``, ``activity_start_date``,
    ``time_utc``, ``characteristic``, ``value``, ``unit``, depth fields, ``qc_class``,
    ``status_class``, ``censored``, ``detection_limit``) and their processed variants (``variable``,
    ``date``). Rules: target characteristic names only (probe/RFU names never included); QC
    activities, logger records and rejected results dropped; depth <= 1 m or unspecified (Secchi
    exempt); replicates at the same site, local date, variable, characteristic and unit are
    aggregated to their median (n kept); censored values stay NaN with detection limits.
    """
    df = res.copy()
    char = _col(df, "characteristic", "CharacteristicName").astype(str)
    df["target"] = char.map(NAME_TO_TARGET)
    if "variable" in df.columns:  # processed table may already name the variable
        df["target"] = df["target"].fillna(df["variable"].where(df["variable"].isin(WQP_TARGETS)))
    df = df[df["target"].notna()].copy()
    df["characteristic"] = _col(df, "characteristic", "CharacteristicName", default="").astype(str)
    qc = _col(df, "qc_class", default="").fillna("").astype(str)
    st = _col(df, "status_class", default="unknown").fillna("unknown").astype(str)
    df = df[(qc == "") & (st != "rejected")].copy()
    d_res = depth_m(_col(df, "result_depth", default=np.nan), _col(df, "result_depth_unit", default=""))
    d_act = depth_m(_col(df, "activity_depth", default=np.nan), _col(df, "activity_depth_unit", default=""))
    d_top = depth_m(_col(df, "activity_top_depth", default=np.nan), _col(df, "activity_top_depth_unit", default=""))
    df["sample_depth_m"] = d_res.fillna(d_act).fillna(d_top)
    rel = _col(df, "activity_relative_depth", default="").fillna("").astype(str).str.lower()
    exempt = df["target"].isin(DEPTH_EXEMPT)
    surface = (df["sample_depth_m"].isna() | (df["sample_depth_m"] <= SURFACE_MAX_M)) & ~rel.str.contains("bottom")
    df["depth_rule"] = np.where(exempt, "exempt", np.where(df["sample_depth_m"].isna(), "unspecified", "<=1m"))
    df = df[surface | exempt].copy()
    df["date"] = pd.to_datetime(_col(df, "date", "activity_start_date"), errors="coerce").dt.normalize()
    df = df[df["date"].notna()]
    df["value"] = pd.to_numeric(_col(df, "value"), errors="coerce")
    df["censored"] = _col(df, "censored", default=False).fillna(False).astype(bool)
    df["detection_limit"] = pd.to_numeric(_col(df, "detection_limit", default=np.nan), errors="coerce")
    df["unit"] = _col(df, "unit", default="").fillna("").astype(str)
    df["time_utc"] = pd.to_datetime(_col(df, "time_utc", default=pd.NaT), utc=True, errors="coerce")
    df["site_id"] = df["site_id"].astype(str)
    keys = ["site_id", "date", "target", "characteristic", "unit"]
    agg = df.groupby(keys, dropna=False).agg(
        value=("value", "median"), value_min=("value", "min"), value_max=("value", "max"),
        n_results=("value", "size"), n_censored=("censored", "sum"),
        detection_limit=("detection_limit", "max"), time_utc=("time_utc", "min"),
        sample_depth_m=("sample_depth_m", "min"), depth_rule=("depth_rule", "first")).reset_index()
    agg["censored"] = agg["value"].isna() & (agg["n_censored"] > 0)
    if stations is not None and len(stations):
        st_cols = [c for c in ("site_id", "lat", "lon", "lat_std", "lon_std", "site_type", "name", "huc8",
                               "organization", "provider") if c in stations.columns]
        if "state" not in agg.columns and "state" in stations.columns:
            st_cols.append("state")
        agg = agg.merge(stations[st_cols].drop_duplicates("site_id"), on="site_id", how="left")
    if "lat_std" in agg.columns:  # standardized WGS84 coordinates when available
        agg["lat"] = agg["lat_std"].fillna(agg.get("lat"))
        agg["lon"] = agg["lon_std"].fillna(agg.get("lon"))
    agg["site_class"] = site_class(agg["site_type"]) if "site_type" in agg.columns else "unknown"
    agg["site_class_source"] = np.where(agg["site_class"] != "unknown", "site_type", "none")
    if "name" in agg.columns:  # fall back to the location name when the WQX type is empty
        by_name = site_class(agg["name"]).replace({"unknown": "other"})
        m = (agg["site_class"] == "unknown") & (by_name != "other")
        agg.loc[m, "site_class"] = by_name[m]
        agg.loc[m, "site_class_source"] = "name"
    return agg


# ------------------------------------------------------------------------- WQP loading


RESULT_KEEP = ("site_id", "activity_start_date", "date", "time_utc", "characteristic", "variable", "value",
               "unit", "censored", "detection_limit", "qc_class", "status_class", "activity_depth",
               "activity_depth_unit", "activity_top_depth", "activity_top_depth_unit", "result_depth",
               "result_depth_unit", "activity_relative_depth", "state", "sample_fraction", "method_id")


def load_wqp(source: str, targets: Sequence[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame | None, list[Path]]:
    """(results, stations, files read) from processed('wqp_discrete') or raw('wqp').

    Tables are recognised by their columns (results: site_id + value + characteristic/variable;
    stations: site_id + lat + lon without value). Result rows are filtered at read time to the
    target characteristic names (or variables), so the multi-group raw tables never load fully.
    Files that cannot be read (e.g. being written) are skipped and reported."""
    import pyarrow.parquet as pq
    from ..core import paths
    root = paths.PROCESSED / "wqp_discrete" if source == "processed" else paths.RAW / "wqp"
    files = sorted(root.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files under {root}")
    tg = list(targets) if targets else list(WQP_TARGETS)
    names = [n for t in tg for n in WQP_TARGETS[t]]
    res, sta, used, skipped = [], [], [], []
    for f in files:
        try:
            cols = set(pq.ParquetFile(f).schema_arrow.names)
            if {"site_id", "value"} <= cols and ({"characteristic", "variable"} & cols):
                keep = [c for c in RESULT_KEEP if c in cols]
                key, vals = ("characteristic", names) if "characteristic" in cols else ("variable", tg)
                try:
                    df = pq.read_table(f, columns=keep, filters=[(key, "in", vals)]).to_pandas()
                except Exception:  # noqa: BLE001 — e.g. an all-null typed column in an empty chunk
                    df = pq.read_table(f, columns=keep).to_pandas()
                    df = df[df[key].isin(vals)]
                if len(df):
                    res.append(df)
                used.append(f)
            elif {"site_id", "lat", "lon"} <= cols and "value" not in cols:
                sta.append(pd.read_parquet(f))
                used.append(f)
        except Exception as e:  # noqa: BLE001 — partially written file etc.
            skipped.append((str(f), repr(e)[:100]))
    if skipped:
        log.warning("skipped %d unreadable WQP files (first: %s)", len(skipped), skipped[:2])
    if not res:
        raise FileNotFoundError(f"no WQP result rows for {tg} under {root}")
    stations = None
    if sta:
        stations = pd.concat([d for d in sta if len(d)], ignore_index=True)
        stations["_nn"] = stations[["lat", "lon"]].notna().sum(axis=1) + \
            (stations.get("site_type", pd.Series("", index=stations.index)).fillna("") != "").astype(int)
        stations = stations.sort_values("_nn", ascending=False).drop_duplicates("site_id").drop(columns="_nn")
    return pd.concat(res, ignore_index=True), stations, used


def site_jrc(sites: pd.DataFrame, cache: Path, workers: int = 4,
             logger: Callable[[str], None] = log.info) -> pd.DataFrame:
    """JRC point statistics per site (cached, resume-safe)."""
    done = pd.read_parquet(cache) if cache.exists() else pd.DataFrame(columns=["point_key"])
    todo = sites[~sites.point_key.isin(done.point_key)]
    if len(todo):
        jrc = S.JrcOccurrence()

        def one(r):
            try:
                w = jrc.window(r.lon, r.lat)
                st = S.jrc_point_stats(w[0], w[1], r.lon, r.lat) if w else {"jrc_error": "no_tile"}
            except Exception as e:  # noqa: BLE001
                st = {"jrc_error": repr(e)[:120]}
            return {"point_key": r.point_key, **st}
        rows = []
        with ThreadPoolExecutor(workers) as ex:
            for k, row in enumerate(ex.map(one, todo.itertuples()), 1):
                rows.append(row)
                if k % 2000 == 0:
                    logger(f"JRC {k}/{len(todo)}")
                    part = pd.concat([done, C.normalise_columns(pd.DataFrame(rows))], ignore_index=True)
                    part.to_parquet(cache, index=False)
        done = pd.concat([done, C.normalise_columns(pd.DataFrame(rows))], ignore_index=True)
        done.to_parquet(cache, index=False)
    return done
