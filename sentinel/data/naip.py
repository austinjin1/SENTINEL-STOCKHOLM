# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""NAIP (4-band aerial, Planetary Computer ``naip``) matchups for the M7 RGB+NIR drone student.

NAIP is 8-bit, vendor colour-balanced per state/year and NOT reflectance; chips are stored as raw
digital numbers (uint8, band order R, G, B, NIR) together with per-scene (DOQQ) radiometric
statistics from a decimated read of the whole image, so per-scene normalisation can be fitted later.
NAIP STAC ``datetime`` is the acquisition date only (no time of day; a DOQQ can be a mosaic of
flight lines, the STAC date is the one recorded for the DOQQ).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from . import chips as C
from .sources import stac as S

log = logging.getLogger("sentinel.naip")
NAIP_LAYERS = ["R", "G", "B", "NIR"]
NAIP_SEMANTICS = {
    "values": "uint8 digital numbers as distributed (not reflectance); 0 = outside image",
    "bands": "R, G, B, NIR (NAIP band order)",
    "grid": "native NAIP GSD (0.3/0.6/1.0 m by state-year), DOQQ CRS (NAD83 UTM); point in pixel (128, 128)",
    "radiometry": "per-scene statistics in naip_scene_stats.parquet (decimated whole-DOQQ read)",
}
YEARS = tuple(range(2010, 2024))


def state_catalog(state: str, cache_dir: Path, years: Iterable[int] = YEARS) -> pd.DataFrame:
    """All NAIP DOQQ items of a state (one row per item: id, date, year, gsd, geometry JSON, href)."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    f = cache_dir / f"{state.lower()}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    rows = []
    for y in years:
        for it in S.search_naip(state=state, year=y):
            p = it["properties"]
            rows.append({"item_id": it["id"], "state": p.get("naip:state"), "year": int(p.get("naip:year", y)),
                         "date": S.naip_acquisition_date(it), "gsd": p.get("gsd"),
                         "geometry": json.dumps(it["geometry"]), "bbox": json.dumps(it.get("bbox")),
                         "href": it["assets"]["image"]["href"].split("?")[0], "epsg": p.get("proj:epsg") or
                         (int(str(p.get("proj:code")).split(":")[1]) if str(p.get("proj:code", "")).startswith("EPSG:") else None)})
        time.sleep(0.2)
    df = pd.DataFrame(rows)
    df.to_parquet(f, index=False)
    return df


def sites_in_items(sites: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    """(point_key, item_id, naip_date, year, gsd) for every DOQQ whose footprint contains a site."""
    from shapely import STRtree, points
    from shapely.geometry import shape
    if catalog.empty or sites.empty:
        return pd.DataFrame(columns=["point_key", "item_id", "naip_date", "naip_year", "gsd"])
    geoms = [shape(json.loads(g)) for g in catalog.geometry]
    tree = STRtree(geoms)
    pts = points(sites.lon.values, sites.lat.values)
    ip, ig = tree.query(pts, predicate="within")
    interior = np.array([geoms[j].boundary.distance(pts[i]) for i, j in zip(ip, ig)])
    out = pd.DataFrame({"point_key": sites.point_key.values[ip], "item_id": catalog.item_id.values[ig],
                        "naip_date": pd.to_datetime(catalog.date.values[ig]), "naip_year": catalog.year.values[ig],
                        "gsd": catalog.gsd.values[ig], "interior_deg": interior})
    # overlapping DOQQs of the same acquisition date: keep the one where the site is most interior
    out = out.sort_values(["point_key", "naip_date", "interior_deg"], ascending=[True, True, False])
    return out.drop_duplicates(["point_key", "naip_date"]).reset_index(drop=True)


def naip_chip_features(chip: np.ndarray) -> dict[str, float]:
    """Centre statistics of a NAIP chip (DN): per-band medians in the centre 9x9 / 31x31 pixels,
    the 'dark-NIR' (NIR < G, water-like) fraction, saturation and fill fractions."""
    size = chip.shape[-1]
    out: dict[str, float] = {}
    valid = (chip > 0).all(axis=0)
    out["chip_fill_frac"] = float(1 - valid.mean())
    out["chip_saturated_frac"] = float((chip == 255).any(axis=0).mean())
    for k in (9, 31):
        sl = S.centre_slice(size, k)
        v = valid[sl, sl]
        for i, b in enumerate(NAIP_LAYERS):
            x = chip[i][sl, sl][v]
            out[f"c{k}_{b}_median"] = float(np.median(x)) if x.size else float("nan")
        dn = (chip[3][sl, sl] < chip[1][sl, sl]) & v
        out[f"c{k}_darknir_frac"] = float(dn.mean())
    out["chip_darknir_frac"] = float(((chip[3] < chip[1]) & valid).mean())
    return out


class NaipExtractor:
    """Item-grouped NAIP chip reads (one COG open per DOQQ) + cached per-scene statistics."""

    def __init__(self, writer: C.ChipWriter, stats_path: Path, *, workers: int = 4, size: int = 256,
                 logger: Callable[[str], None] = log.info):
        self.w = writer
        self.workers = workers
        self.size = size
        self.stats_path = Path(stats_path)
        self.log = logger
        self.stats = pd.read_parquet(self.stats_path) if self.stats_path.exists() else pd.DataFrame()
        self._lock = threading.Lock()
        self.counts: dict[str, int] = defaultdict(int)

    def _scene_stats(self, item: dict) -> None:
        with self._lock:
            if len(self.stats) and item["id"] in set(self.stats.naip_item_id):
                return
        st = S.naip_scene_stats(item)
        with self._lock:
            self.stats = pd.concat([self.stats, pd.DataFrame([st])], ignore_index=True)

    def run(self, requests: pd.DataFrame, items: dict[str, dict]) -> dict[str, int]:
        """``requests``: request_key, item_id, lon, lat (+ passthrough columns)."""
        todo = requests[~requests.request_key.map(self.w.is_done) & requests.item_id.isin(items.keys())]
        groups = list(todo.groupby("item_id"))
        self.log(f"NAIP extraction: {len(todo)} requests in {len(groups)} DOQQs")

        def task(iid: str, g: pd.DataFrame):
            item = items[iid]
            meta_cols = [c for c in g.columns if c != "request_key"]
            try:
                self._scene_stats(item)
                pts = list(zip(g.lon.astype(float), g.lat.astype(float)))
                arrs, grids, info = S.read_naip_chips(item, pts, self.size)
                for (k, (_, r)), a, gr in zip(zip(g.request_key, g.iterrows()), arrs, grids):
                    row = {**{c: r[c] for c in meta_cols}, "naip_item_id": iid,
                           "naip_date": S.naip_acquisition_date(item), "gsd_m": info["gsd"], "chip_crs": gr.crs,
                           "chip_left": gr.bounds[0], "chip_top": gr.bounds[3], **naip_chip_features(a)}
                    if (a[:, self.size // 2, self.size // 2] == 0).all():
                        self.w.add(k, None, row, status="outside_image")
                        self.counts["outside_image"] += 1
                    else:
                        self.w.add(k, a, row, status="ok")
                        self.counts["ok"] += 1
            except Exception as e:  # noqa: BLE001
                for k in g.request_key:
                    self.w.add(k, None, {"naip_item_id": iid}, status=f"error:{type(e).__name__}:{str(e)[:120]}")
                self.counts["error"] += len(g)

        t0 = time.time()
        with ThreadPoolExecutor(self.workers) as ex:
            futs = [ex.submit(task, iid, g) for iid, g in groups]
            for n, f in enumerate(as_completed(futs), 1):
                f.result()
                if n % 100 == 0:
                    self.log(f"NAIP DOQQs {n}/{len(groups)} ({n / (time.time() - t0) * 3600:.0f}/h) {dict(self.counts)}")
                    self.flush_stats()
        self.w.flush()
        self.flush_stats()
        return dict(self.counts)

    def flush_stats(self) -> None:
        with self._lock:
            if len(self.stats):
                C.normalise_columns(self.stats).to_parquet(self.stats_path, index=False)
