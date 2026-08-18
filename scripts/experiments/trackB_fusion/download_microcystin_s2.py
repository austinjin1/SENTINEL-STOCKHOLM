#!/usr/bin/env python3
"""Download Sentinel-2 L2A tiles for microcystin site/date pairs.

For each (site, measurement date) it grabs the nearest low-cloud S2 scene
within +/- 7 days, reads a 64x64 window of 6 bands around the site, and
stores the tile. Resumable via per-tile cache. Time-boxed via --minutes.

Output tiles: data/processed/biology/noaa_habs/mc_tiles/<idx>.npz
MIT License -- Bryan Cheng, 2026
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path("/home/bcheng/SENTINEL")
TARGETS = PROJECT / "data/processed/biology/noaa_habs/mc_download_targets.parquet"
TILE_CACHE = PROJECT / "data/processed/biology/noaa_habs/mc_tiles"
TILE_CACHE.mkdir(parents=True, exist_ok=True)

# Blue, Green, Red, RedEdge(B05), NIR(B08), SWIR(B11) -- HAB-relevant bands
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
        actual_date = item.properties["datetime"][:10]
        cloud = item.properties.get("eo:cloud_cover", -1)
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
        return ("err:" + repr(e)[:60], idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=15)
    ap.add_argument("--limit", type=int, default=0, help="max jobs (0=all)")
    args = ap.parse_args()

    df = pd.read_parquet(TARGETS)
    df = df.reset_index(drop=True)
    df["date_str"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    log(f"Total site-date pairs: {len(df)}")

    jobs = []
    for i, r in df.iterrows():
        jobs.append((int(i), float(r["Location_Latitude"]),
                     float(r["Location_Longitude"]), r["date_str"]))
    if args.limit:
        jobs = jobs[:args.limit]

    done = {f.stem for f in TILE_CACHE.glob("*.npz")}
    failed = {f.stem for f in TILE_CACHE.glob("*.fail")}
    log(f"Already cached: {len(done)} | prior fails: {len(failed)}")
    jobs = [j for j in jobs if str(j[0]) not in done and str(j[0]) not in failed]
    log(f"Jobs to run this pass: {len(jobs)}")

    deadline = time.time() + args.minutes * 60
    counts = {}
    n = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {}
        it = iter(jobs)
        # prime
        for _ in range(min(MAX_WORKERS * 3, len(jobs))):
            try:
                j = next(it)
                futs[pool.submit(download_one, *j)] = j
            except StopIteration:
                break
        while futs:
            for fut in as_completed(list(futs)):
                j = futs.pop(fut)
                status, idx = fut.result()
                key = status.split(":")[0]
                counts[key] = counts.get(key, 0) + 1
                n += 1
                if n % 50 == 0:
                    log(f"  {n} processed | {counts}")
                if time.time() < deadline:
                    try:
                        nj = next(it)
                        futs[pool.submit(download_one, *nj)] = nj
                    except StopIteration:
                        pass
                break
            if time.time() >= deadline:
                log("TIME BUDGET REACHED -- stopping submission")
                break
    log(f"Pass done. processed={n} counts={counts}")
    ok = len(list(TILE_CACHE.glob("*.npz")))
    log(f"TOTAL cached tiles now: {ok}")


if __name__ == "__main__":
    main()
