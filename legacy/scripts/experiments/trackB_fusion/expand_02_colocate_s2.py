#!/usr/bin/env python3
"""EXPAND step 2 -- download Sentinel-2 tiles for the expanded microcystin rows.

IDENTICAL to download_microcystin_s2.py: same 6 bands [B02,B03,B04,B05,B08,B11],
64x64 window, +/-7d nearest low-cloud scene, Planetary Computer. Tiles cached to
mc_expand_tiles/<i>.npz keyed by row index in mc_expand_targets.parquet.

Time-boxed via --minutes. Prioritizes BREADTH: --per-region caps tiles attempted
per HUC2 so many regions get covered before any one region is exhausted.
"""
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/bcheng/SENTINEL")
OUT_DIR = ROOT / "data/processed/biology/noaa_habs"
TARGETS = OUT_DIR / "mc_expand_targets.parquet"
TILE_CACHE = OUT_DIR / "mc_expand_tiles"
TILE_CACHE.mkdir(parents=True, exist_ok=True)

# EXACT same band list / geometry as the original download.
BANDS = ["B02", "B03", "B04", "B05", "B08", "B11"]
PATCH = 64
MAX_CLOUD = 40
WINDOW_DAYS = 7
MAX_WORKERS = 10


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def download_one(idx, lat, lon, date_str):
    import planetary_computer
    import pystac_client
    import rasterio
    from pyproj import Transformer

    cache = TILE_CACHE / f"{idx}.npz"
    if cache.exists():
        return ("cached", idx)
    fail = TILE_CACHE / f"{idx}.fail"
    if fail.exists():
        return ("failskip", idx)
    try:
        catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )
        bbox = [lon - 0.01, lat - 0.01, lon + 0.01, lat + 0.01]
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        t0 = (dt - timedelta(days=WINDOW_DAYS)).strftime("%Y-%m-%d")
        t1 = (dt + timedelta(days=WINDOW_DAYS)).strftime("%Y-%m-%d")
        search = catalog.search(
            collections=["sentinel-2-l2a"], bbox=bbox,
            datetime=f"{t0}/{t1}",
            query={"eo:cloud_cover": {"lt": MAX_CLOUD}}, max_items=8,
        )
        items = list(search.items())
        if not items:
            fail.write_text("no_items")
            return ("noscene", idx)
        tgt = dt.timestamp()
        items.sort(key=lambda it: abs(
            datetime.fromisoformat(
                it.properties["datetime"].replace("Z", "+00:00")
            ).timestamp() - tgt))
        item = planetary_computer.sign(items[0])
        cloud = item.properties.get("eo:cloud_cover", -1)
        actual_date = item.properties["datetime"][:10]
        bands = []
        for b in BANDS:
            asset = item.assets.get(b)
            if asset is None:
                fail.write_text("no_asset")
                return ("noasset", idx)
            with rasterio.open(asset.href) as src:
                tr = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                x, y = tr.transform(lon, lat)
                py, px = src.index(x, y)
                half = PATCH // 2
                if not (half <= px < src.width - half and
                        half <= py < src.height - half):
                    fail.write_text("edge")
                    return ("edge", idx)
                win = rasterio.windows.Window(px - half, py - half, PATCH, PATCH)
                arr = src.read(1, window=win).astype(np.float32) / 10000.0
                if arr.shape != (PATCH, PATCH):
                    fail.write_text("shape")
                    return ("shape", idx)
                bands.append(arr)
        img = np.stack(bands, 0)
        if np.isnan(img).any() or img.max() < 0.001:
            fail.write_text("empty")
            return ("empty", idx)
        np.savez_compressed(cache, image=img, actual_date=actual_date,
                            cloud=cloud, target_date=date_str)
        return ("ok", idx)
    except Exception as e:
        return ("err:" + repr(e)[:50], idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--per-region", type=int, default=120,
                    help="cap tiles attempted per HUC2 (breadth over depth)")
    args = ap.parse_args()

    df = pd.read_parquet(TARGETS).reset_index(drop=True)
    df["date_str"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    log(f"Total expanded site-date rows: {len(df)}")

    done = {f.stem for f in TILE_CACHE.glob("*.npz")}
    failed = {f.stem for f in TILE_CACHE.glob("*.fail")}
    log(f"cached: {len(done)} | prior fails: {len(failed)}")

    # BREADTH-first: interleave rows across regions, cap per region.
    df["huc2"] = df["huc2"].fillna("XX")
    jobs = []
    for huc, g in df.groupby("huc2"):
        g = g.sort_values("date")
        # prefer positives + spread of dates; simple cap
        gi = list(g.index)[: args.per_region]
        jobs.extend(gi)
    # round-robin across regions so early time budget covers many regions
    by_reg = {}
    for i in jobs:
        by_reg.setdefault(df.at[i, "huc2"], []).append(i)
    order = []
    while any(by_reg.values()):
        for h in list(by_reg):
            if by_reg[h]:
                order.append(by_reg[h].pop(0))
    order = [i for i in order
             if str(i) not in done and str(i) not in failed]
    log(f"jobs this pass (breadth-ordered): {len(order)}")

    deadline = time.time() + args.minutes * 60
    counts, n = {}, 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {}
        it = iter(order)
        for _ in range(min(MAX_WORKERS * 3, len(order))):
            try:
                i = next(it)
                futs[pool.submit(download_one, int(i), float(df.at[i, "Location_Latitude"]),
                                 float(df.at[i, "Location_Longitude"]), df.at[i, "date_str"])] = i
            except StopIteration:
                break
        while futs:
            for fut in as_completed(list(futs)):
                futs.pop(fut)
                status, idx = fut.result()
                key = status.split(":")[0]
                counts[key] = counts.get(key, 0) + 1
                n += 1
                if n % 50 == 0:
                    log(f"  {n} processed | {counts}")
                if time.time() < deadline:
                    try:
                        i = next(it)
                        futs[pool.submit(download_one, int(i), float(df.at[i, "Location_Latitude"]),
                                         float(df.at[i, "Location_Longitude"]), df.at[i, "date_str"])] = i
                    except StopIteration:
                        pass
                break
            if time.time() >= deadline:
                log("TIME BUDGET REACHED -- stopping submission")
                break
    ok = len(list(TILE_CACHE.glob("*.npz")))
    log(f"Pass done. processed={n} counts={counts} | TOTAL cached tiles: {ok}")


if __name__ == "__main__":
    main()
