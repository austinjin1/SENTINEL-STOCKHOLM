#!/usr/bin/env python3
"""TRACK B Phase-1 confirmation: for each candidate HAB-advisory lake, verify
Sentinel-2 L2A (Microsoft Planetary Computer) coverage over the lake pixel for
(a) the ADVISORY YEAR and (b) >=2 CLEAN comparison years, during the summer
bloom season. A "usable" scene = a low-cloud scene whose water pixel(s) at the
sensor location yield a valid NDCI (i.e. sonde is on water, scene not clouded
out at that pixel). This lets us test later whether the advisory-year bloom is
anomalously severe vs the lake's own clean-year seasonal baseline.

Only COUNTS scenes (cheap: no band download for the count -- uses STAC metadata
cloud_cover). For a subset it also samples the actual sensor pixel to confirm the
pixel is on water and a real NDCI can be extracted (proves colocation works).

Prints per-lake, per-year usable-scene counts and writes a JSON.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path("/home/bcheng/SENTINEL")
OUT = ROOT / "data" / "processed" / "trackB" / "hab_s2_coverage.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

BANDS = ["B04", "B05", "B03", "B08"]  # for NDCI + NDWI water check
PATCH = 16
MAX_CLOUD = 40  # scene-level cloud filter
SEASON = ("05-01", "10-31")  # bloom season window each year


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# candidate lakes: name -> sensor lat/lon, advisory year, clean comparison years
LAKES = {
    "Detroit_Lake_OR_2024": {
        "lat": 44.7183, "lon": -122.2461, "advisory_year": 2024,
        "clean_years": [2021, 2022, 2023],
        "official": "2024-07-16", "usgs": "444306122144600"},
    "Seneca_Lake_NY_2019": {
        "lat": 42.840806, "lon": -76.945583, "advisory_year": 2019,
        "clean_years": [2020, 2021, 2022],
        "official": "2019", "usgs": "425027076564401"},
    "Milford_Lake_KS_2016": {
        "lat": 39.2164, "lon": -97.0050, "advisory_year": 2016,
        "clean_years": [2017, 2018, 2019],
        "official": "2016-06-30", "usgs": "391259097001800"},
}


def get_catalog():
    import planetary_computer, pystac_client
    return pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace)


def count_scenes(cat, lat, lon, year):
    """STAC-count low-cloud scenes over the pixel for [year-05-01 .. year-10-31]."""
    bbox = [lon - 0.01, lat - 0.01, lon + 0.01, lat + 0.01]
    dt = f"{year}-{SEASON[0]}/{year}-{SEASON[1]}"
    search = cat.search(collections=["sentinel-2-l2a"], bbox=bbox, datetime=dt,
                        query={"eo:cloud_cover": {"lt": MAX_CLOUD}}, max_items=500)
    items = list(search.items())
    by_day = {}
    for it in items:
        d = it.properties["datetime"][:10]
        c = it.properties.get("eo:cloud_cover", 100)
        if d not in by_day or c < by_day[d][1]:
            by_day[d] = (it, c)
    return by_day  # date -> (item, cloud)


def sample_pixel(item, lat, lon):
    """Download PATCH window, return NDCI over water pixels or None."""
    import planetary_computer, rasterio
    from pyproj import Transformer
    try:
        item = planetary_computer.sign(item)
        half = PATCH // 2
        bands = {}
        for b in BANDS:
            asset = item.assets.get(b)
            if asset is None:
                return None
            with rasterio.open(asset.href) as src:
                tr = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                x, y = tr.transform(lon, lat)
                py, px = src.index(x, y)
                if not (half <= px < src.width - half and half <= py < src.height - half):
                    return None
                win = rasterio.windows.Window(px - half, py - half, PATCH, PATCH)
                arr = src.read(1, window=win).astype(np.float32) / 10000.0
                if arr.shape != (PATCH, PATCH):
                    return None
                bands[b] = arr
        b03, b04, b05, b08 = bands["B03"], bands["B04"], bands["B05"], bands["B08"]
        ndwi = (b03 - b08) / (b03 + b08 + 1e-6)
        water = (ndwi > 0.0) & (b03 < 0.30)
        if water.sum() < 3:
            return None
        ndci = (b05 - b04) / (b05 + b04 + 1e-6)
        return float(np.median(ndci[water])), int(water.sum())
    except Exception as e:
        return None


def main():
    n_sample = int(sys.argv[1]) if len(sys.argv) > 1 else 3  # pixel-verify N scenes/year
    cat = get_catalog()
    result = {}
    for name, m in LAKES.items():
        log(f"=== {name}  sensor ({m['lat']},{m['lon']}) ===")
        years = [m["advisory_year"]] + m["clean_years"]
        per_year = {}
        for yr in years:
            by_day = count_scenes(cat, m["lat"], m["lon"], yr)
            role = "ADVISORY" if yr == m["advisory_year"] else "clean"
            # pixel-verify a few evenly-spaced scenes: confirm sonde pixel is on water
            dates = sorted(by_day)
            verified = 0
            ndci_samples = []
            if dates and n_sample > 0:
                idx = np.linspace(0, len(dates) - 1, min(n_sample, len(dates))).astype(int)
                for i in idx:
                    it, _ = by_day[dates[i]]
                    r = sample_pixel(it, m["lat"], m["lon"])
                    if r is not None:
                        verified += 1
                        ndci_samples.append(round(r[0], 4))
            per_year[str(yr)] = {
                "role": role, "low_cloud_scenes": len(by_day),
                "date_first": dates[0] if dates else None,
                "date_last": dates[-1] if dates else None,
                "pixel_verified_on_water": verified, "pixel_sampled": int(min(n_sample, len(dates))),
                "ndci_samples": ndci_samples,
            }
            log(f"  {yr} [{role:8s}] scenes={len(by_day):3d} "
                f"water_verified={verified}/{min(n_sample,len(dates))} ndci={ndci_samples}")
        # coverage confirmed = advisory year AND >=2 clean years each have >=6 low-cloud
        # scenes AND >=1 pixel-verified-on-water sample.
        def ok(y):
            d = per_year[str(y)]
            return d["low_cloud_scenes"] >= 6 and d["pixel_verified_on_water"] >= 1
        clean_ok = [y for y in m["clean_years"] if ok(y)]
        confirmed = ok(m["advisory_year"]) and len(clean_ok) >= 2
        result[name] = {
            "usgs_site": m["usgs"], "official": m["official"],
            "sensor_latlon": [m["lat"], m["lon"]],
            "advisory_year": m["advisory_year"], "clean_years_tested": m["clean_years"],
            "clean_years_confirmed": clean_ok,
            "data_confirmed": bool(confirmed),
            "per_year": per_year,
        }
        log(f"  -> data_confirmed={confirmed}  clean_years_confirmed={clean_ok}")
    OUT.write_text(json.dumps(result, indent=2))
    log(f"WROTE {OUT}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
