#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""HydroLAKES v1.0 polygons for North America + Europe -> processed('hydrolakes')/hydrolakes_na_eu.parquet
(GeoParquet, EPSG:4326). The raw zip (820 MB) is deleted afterwards unless --keep-zip (its sha256 and
retrieval time are kept in the manifest).

    curl -A "SENTINEL-v2 research (public-data download)" -o $SENTINEL_DATA/raw/hydrolakes/HydroLAKES_polys_v10_shp.zip \
        https://data.hydrosheds.org/file/hydrolakes/HydroLAKES_polys_v10_shp.zip
    python -B scripts/data/build_hydrolakes.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths  # noqa: E402
from sentinel.core.manifest import write_manifest  # noqa: E402
from sentinel.core.registry import file_sha256  # noqa: E402
from sentinel.data.sources import hydrolakes as H  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-zip", action="store_true")
    a = ap.parse_args()
    raw = paths.raw("hydrolakes")
    z = raw / "HydroLAKES_polys_v10_shp.zip"
    sha = file_sha256(z)
    retrieved = (raw / "retrieved_utc.txt").read_text().strip() if (raw / "retrieved_utc.txt").exists() else ""
    parts = [H.read_region(z, r) for r in H.REGIONS]
    import geopandas as gpd

    gdf = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=parts[0].crs)
    # a lake straddling the two boxes cannot occur (boxes are disjoint), but keep ids unique anyway
    gdf = gdf.drop_duplicates("Hylak_id").reset_index(drop=True)
    out = paths.processed("hydrolakes") / "hydrolakes_na_eu.parquet"
    gdf.to_parquet(out, compression="zstd")
    stats = {"n_lakes": int(len(gdf)), "by_region": gdf.region.value_counts().to_dict(),
             "area_km2_quantiles": gdf.Lake_area.quantile([0, .5, .9, .99, 1]).round(3).to_dict(),
             "raw_zip_sha256": sha, "raw_zip_bytes": z.stat().st_size, "retrieved_utc": retrieved,
             "regions_bbox": {k: v[0] for k, v in H.REGIONS.items()}}
    write_manifest("hydrolakes", files=[out], sources=[H.URL], license=H.LICENSE, synthetic=False, stats=stats,
                   description="HydroLAKES v1.0 lake polygons (>= 10 ha), North America + Europe bounding boxes; "
                               "used for water-body identity in SENTINEL-Orbit splits.")
    print(stats)
    if not a.keep_zip:
        z.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
