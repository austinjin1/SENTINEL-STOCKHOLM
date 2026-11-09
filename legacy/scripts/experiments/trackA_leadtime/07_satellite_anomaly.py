#!/usr/bin/env python3
"""Track A step 7 -- SELF-NORMALIZED satellite bloom-anomaly time series per NEON site.

For each of the 15 real Track-A hypoxia-episode sites, download Sentinel-2 L2A
scenes from Microsoft Planetary Computer over (a) a BASELINE history and (b) a
pre-onset window, sample the water pixel(s) at/near the NEON sensor location,
compute NDCI/NDVI/chl proxy (SAME bands + NDCI formula as trackB colocate code:
B02,B03,B04,B05,B08,B11; NDCI=(B05-B04)/(B05+B04)), then z-score each date's chl
proxy against THAT site's OWN baseline distribution (self-normalization =
cross-region invariant "how abnormally green is this site now vs its own history").

First look: does sat_anomaly_z RISE in the weeks before the DO-hypoxia onset?

Writes:
  data/processed/trackA/neon_satellite_anomaly.parquet
  data/processed/trackA/neon_satellite_coverage.json

Time-boxed via --minutes. Breadth over depth: round-robins scene downloads
across sites so every site gets sampled before any one site is exhausted.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/bcheng/SENTINEL")
OUTDIR = ROOT / "data" / "processed" / "trackA"
TILE_CACHE = OUTDIR / "s2_anomaly_tiles"
TILE_CACHE.mkdir(parents=True, exist_ok=True)

# EXACT same band list as trackB colocate.
BANDS = ["B02", "B03", "B04", "B05", "B08", "B11"]
PATCH = 16          # small window centered on sensor; NEON sites are small.
MAX_CLOUD = 40
MAX_WORKERS = 8

# 15 real Track-A hypoxia-episode sites: onset UTC (from extract_summary /
# expanded_manifest) + lat/lon + waterbody type (from exp14/exp3 coord tables,
# the coords already used in Track A extraction).
SITES = {
    # 12 original episodes
    "MCDI": {"onset": "2024-07-27", "lat": 40.700, "lon": -99.102, "type": "stream"},
    "ARIK": {"onset": "2024-05-17", "lat": 39.758, "lon": -102.447, "type": "stream"},
    "KING": {"onset": "2024-07-31", "lat": 35.962, "lon": -84.289, "type": "stream"},
    "BIGC": {"onset": "2025-08-01", "lat": 37.068, "lon": -119.255, "type": "stream"},
    "PRIN": {"onset": "2024-10-25", "lat": 29.682, "lon": -103.782, "type": "stream"},
    "COMO": {"onset": "2026-01-30", "lat": 40.035, "lon": -105.544, "type": "stream"},
    "MAYF": {"onset": "2025-04-17", "lat": 32.959, "lon": -87.123, "type": "stream"},
    "LECO": {"onset": "2025-08-03", "lat": 35.691, "lon": -84.285, "type": "stream"},
    "CRAM": {"onset": "2023-05-03", "lat": 45.795, "lon": -89.524, "type": "lake"},
    "LIRO": {"onset": "2023-06-09", "lat": 45.998, "lon": -89.705, "type": "lake"},
    "BARC": {"onset": "2024-12-08", "lat": 29.676, "lon": -82.008, "type": "lake"},
    "GUIL": {"onset": "2026-02-10", "lat": 35.689, "lon": -79.498, "type": "stream"},
    # 3 expansion episodes
    "BLWA": {"onset": "2024-09-04", "lat": 32.541, "lon": -87.801, "type": "river"},
    "PRLA": {"onset": "2024-08-04", "lat": 46.770, "lon": -99.111, "type": "lake"},
    "SUGG": {"onset": "2025-06-02", "lat": 29.688, "lon": -82.018, "type": "lake"},
}

BASELINE_MONTHS = 18       # >=18 months of history for the site's own seasonal normal
PREONSET_DAYS = 120        # >=120 days before onset


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def list_scenes(lat, lon, t0, t1):
    """All L2A scenes over [t0,t1] under MAX_CLOUD, sorted by date."""
    import planetary_computer
    import pystac_client
    cat = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    bbox = [lon - 0.01, lat - 0.01, lon + 0.01, lat + 0.01]
    search = cat.search(
        collections=["sentinel-2-l2a"], bbox=bbox,
        datetime=f"{t0}/{t1}",
        query={"eo:cloud_cover": {"lt": MAX_CLOUD}}, max_items=500,
    )
    items = list(search.items())
    # dedup by date, keep lowest cloud per calendar day
    by_day = {}
    for it in items:
        d = it.properties["datetime"][:10]
        c = it.properties.get("eo:cloud_cover", 100)
        if d not in by_day or c < by_day[d][1]:
            by_day[d] = (it, c)
    out = sorted(by_day.values(), key=lambda x: x[0].properties["datetime"])
    return [it for it, _ in out]


def sample_scene(site, item, lat, lon):
    """Download PATCH window at (lat,lon), return per-band medians over water-ish
    pixels. Cache to npz keyed by site+date. Returns dict or None."""
    import planetary_computer
    import rasterio
    from pyproj import Transformer
    date = item.properties["datetime"][:10]
    cache = TILE_CACHE / f"{site}_{date}.npz"
    if cache.exists():
        try:
            z = np.load(cache, allow_pickle=True)
            return {"date": date, "bands": z["bands"], "cloud": float(z["cloud"])}
        except Exception:
            pass
    fail = TILE_CACHE / f"{site}_{date}.fail"
    if fail.exists():
        return None
    try:
        item = planetary_computer.sign(item)
        cloud = float(item.properties.get("eo:cloud_cover", -1))
        bands = []
        half = PATCH // 2
        for b in BANDS:
            asset = item.assets.get(b)
            if asset is None:
                fail.write_text("no_asset")
                return None
            with rasterio.open(asset.href) as src:
                tr = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                x, y = tr.transform(lon, lat)
                py, px = src.index(x, y)
                if not (half <= px < src.width - half and half <= py < src.height - half):
                    fail.write_text("edge")
                    return None
                win = rasterio.windows.Window(px - half, py - half, PATCH, PATCH)
                arr = src.read(1, window=win).astype(np.float32) / 10000.0
                if arr.shape != (PATCH, PATCH):
                    fail.write_text("shape")
                    return None
                bands.append(arr)
        img = np.stack(bands, 0)  # (6, PATCH, PATCH)
        if np.isnan(img).any() or img.max() < 0.001:
            fail.write_text("empty")
            return None
        # Water masking: NDWI = (B03-B08)/(B03+B08) > 0 tends to be water; and
        # exclude bright/cloud pixels (B02 high). Keep water-ish pixels; if too
        # few, fall back to the central pixel block.
        b02, b03, b04, b05, b08, b11 = img
        ndwi = (b03 - b08) / (b03 + b08 + 1e-6)
        water = (ndwi > 0.0) & (b02 < 0.20)
        if water.sum() < 3:
            # fallback: central 4x4 block
            c0 = PATCH // 2 - 2
            water = np.zeros_like(water)
            water[c0:c0 + 4, c0:c0 + 4] = True
        # per-band median over selected pixels
        med = np.array([np.median(bnd[water]) for bnd in img], dtype=np.float32)
        np.savez_compressed(cache, bands=med, cloud=cloud,
                            n_water=int(water.sum()))
        return {"date": date, "bands": med, "cloud": cloud}
    except Exception as e:
        return {"err": repr(e)[:60]}


def build_windows(onset):
    """Return (baseline t0,t1) and (preonset t0,t1) date strings."""
    on = datetime.strptime(onset, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    # baseline: >=18 months centered around the onset year, but NOT the 120 d
    # pre-onset window (that's the test window). Use [onset-18mo .. onset-120d]
    # plus a symmetric look-after isn't needed; extend baseline back further so
    # the seasonal normal is well sampled.
    base_t1 = on - timedelta(days=PREONSET_DAYS)
    base_t0 = on - timedelta(days=30 * BASELINE_MONTHS)
    pre_t0 = on - timedelta(days=PREONSET_DAYS)
    pre_t1 = on
    fmt = lambda d: d.strftime("%Y-%m-%d")
    return (fmt(base_t0), fmt(base_t1)), (fmt(pre_t0), fmt(pre_t1)), on


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=40)
    ap.add_argument("--max-scenes-per-site", type=int, default=80)
    args = ap.parse_args()
    deadline = time.time() + args.minutes * 60

    # 1) enumerate scenes per site (baseline + preonset), tag which window.
    plan = {}  # site -> list of (item, is_baseline)
    for site, m in SITES.items():
        (b0, b1), (p0, p1), on = build_windows(m["onset"])
        try:
            base_items = list_scenes(m["lat"], m["lon"], b0, b1)
            pre_items = list_scenes(m["lat"], m["lon"], p0, p1)
        except Exception as e:
            log(f"[{site}] scene-list ERROR {repr(e)[:60]}")
            plan[site] = []
            continue
        # cap baseline scenes for breadth (keep spread across time)
        if len(base_items) > args.max_scenes_per_site:
            idx = np.linspace(0, len(base_items) - 1, args.max_scenes_per_site).astype(int)
            base_items = [base_items[i] for i in idx]
        tagged = [(it, True) for it in base_items] + [(it, False) for it in pre_items]
        plan[site] = tagged
        log(f"[{site}] {m['type']}: baseline={len(base_items)} preonset={len(pre_items)} scenes listed")
        if time.time() > deadline:
            log("DEADLINE during listing")
            break

    # 2) round-robin download across sites (breadth first).
    order = []
    pools = {s: list(v) for s, v in plan.items()}
    while any(pools.values()):
        for s in list(pools):
            if pools[s]:
                order.append((s, *pools[s].pop(0)))

    rows = []
    counts = {}
    done = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {}
        it = iter(order)
        for _ in range(min(MAX_WORKERS * 3, len(order))):
            try:
                s, item, isb = next(it)
                futs[pool.submit(sample_scene, s, item, SITES[s]["lat"], SITES[s]["lon"])] = (s, isb)
            except StopIteration:
                break
        while futs:
            for fut in as_completed(list(futs)):
                s, isb = futs.pop(fut)
                res = fut.result()
                done += 1
                if res and "bands" in res:
                    b02, b03, b04, b05, b08, b11 = [float(x) for x in res["bands"]]
                    ndci = (b05 - b04) / (b05 + b04 + 1e-6)
                    ndvi = (b08 - b04) / (b08 + b04 + 1e-6)
                    rows.append({
                        "site": s, "date": res["date"], "ndci": ndci, "ndvi": ndvi,
                        "cloud": res.get("cloud", -1), "is_baseline": bool(isb),
                        "B03": b03, "B04": b04, "B05": b05, "B08": b08,
                    })
                    counts[s] = counts.get(s, 0) + 1
                if done % 40 == 0:
                    log(f"  downloaded {done}/{len(order)} | usable rows {len(rows)}")
                if time.time() < deadline:
                    try:
                        s2, item2, isb2 = next(it)
                        futs[pool.submit(sample_scene, s2, item2, SITES[s2]["lat"], SITES[s2]["lon"])] = (s2, isb2)
                    except StopIteration:
                        pass
                break
            if time.time() >= deadline:
                log("TIME BUDGET REACHED -- stopping downloads")
                break

    if not rows:
        log("NO usable rows collected.")
        (OUTDIR / "neon_satellite_coverage.json").write_text(json.dumps(
            {"sites": {s: {"usable_scenes": 0} for s in SITES}}, indent=2))
        return

    df = pd.DataFrame(rows).drop_duplicates(["site", "date"]).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    # chl proxy: NDCI is a red-edge chlorophyll index; use NDCI directly as the
    # chl proxy (higher = more chlorophyll / greener water).
    df["chl_proxy"] = df["ndci"]

    # 3) self-normalize per site against its OWN baseline distribution.
    out_parts = []
    coverage = {}
    for site, g in df.groupby("site"):
        g = g.sort_values("date").copy()
        onset = pd.Timestamp(SITES[site]["onset"], tz=None)
        g["days_to_onset"] = (onset - g["date"]).dt.days  # >0 = before onset
        base = g[g["is_baseline"]]
        mu = base["chl_proxy"].mean()
        sd = base["chl_proxy"].std(ddof=0)
        if not np.isfinite(sd) or sd < 1e-6:
            # fall back to all-scene stats if baseline too thin
            mu = g["chl_proxy"].mean()
            sd = g["chl_proxy"].std(ddof=0)
        sd = sd if (np.isfinite(sd) and sd > 1e-6) else np.nan
        g["sat_anomaly_z"] = (g["chl_proxy"] - mu) / sd
        out_parts.append(g)
        pre = g[~g["is_baseline"]]
        coverage[site] = {
            "type": SITES[site]["type"],
            "onset": SITES[site]["onset"],
            "usable_scenes_total": int(len(g)),
            "usable_scenes_baseline": int(g["is_baseline"].sum()),
            "usable_scenes_preonset": int((~g["is_baseline"]).sum()),
            "baseline_mu_ndci": float(mu) if np.isfinite(mu) else None,
            "baseline_sd_ndci": float(sd) if np.isfinite(sd) else None,
        }
    # sites with zero usable scenes
    for site in SITES:
        if site not in coverage:
            coverage[site] = {
                "type": SITES[site]["type"], "onset": SITES[site]["onset"],
                "usable_scenes_total": 0, "usable_scenes_baseline": 0,
                "usable_scenes_preonset": 0,
                "reason": "no_usable_s2 (cloud/edge/too_small/empty)",
            }

    final = pd.concat(out_parts, ignore_index=True)
    final = final[["site", "date", "ndci", "ndvi", "chl_proxy", "sat_anomaly_z",
                   "is_baseline", "days_to_onset", "cloud"]]
    final.to_parquet(OUTDIR / "neon_satellite_anomaly.parquet", index=False)
    (OUTDIR / "neon_satellite_coverage.json").write_text(json.dumps(coverage, indent=2))
    log(f"WROTE {OUTDIR/'neon_satellite_anomaly.parquet'} rows={len(final)}")
    log(f"WROTE {OUTDIR/'neon_satellite_coverage.json'}")

    # 4) FIRST LOOK: does anomaly rise before onset at covered sites?
    print("\n=== FIRST LOOK: satellite anomaly before DO-hypoxia onset ===", flush=True)
    leads = []
    for site, g in final.groupby("site"):
        pre = g[(~g["is_baseline"]) & (g["days_to_onset"] >= 0) &
                (g["days_to_onset"] <= PREONSET_DAYS)].sort_values("date")
        if len(pre) < 2:
            print(f"[{site}] preonset scenes={len(pre)} -- too few for lead", flush=True)
            continue
        # first scene where anomaly exceeds +1 sigma AND stays elevated (last
        # sustained rise). Lead = days_to_onset at that scene.
        elevated = pre[pre["sat_anomaly_z"] >= 1.0]
        maxz = pre["sat_anomaly_z"].max()
        if len(elevated):
            lead = int(elevated.iloc[0]["days_to_onset"])
            leads.append(lead)
            print(f"[{site}] preonset={len(pre)} maxZ={maxz:.2f} "
                  f"first z>=1 at {lead} d before onset (LEAD)", flush=True)
        else:
            print(f"[{site}] preonset={len(pre)} maxZ={maxz:.2f} "
                  f"never exceeded +1 sigma (no lead signal)", flush=True)
    if leads:
        print(f"\nSites with pre-onset satellite lead (z>=1): {len(leads)}", flush=True)
        print(f"Median satellite lead: {int(np.median(leads))} days "
              f"(range {min(leads)}..{max(leads)})", flush=True)
    else:
        print("\nNo site showed a pre-onset satellite anomaly lead (z>=1).", flush=True)


if __name__ == "__main__":
    main()
