#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Build the SENTINEL site registry -> $SENTINEL_DATA/processed/sites/sites.parquet.

USGS sites = every site with a daily water-quality series in raw('usgs_dv')/dv_series_catalog.parquet,
enriched with NWIS expanded site info (time zone, altitude datum, drainage area); NEON = aquatic
sites from the public NEON site API (no token). Missing HUCs come from a WBD point query.
All HTTP responses are cached under raw('site_registry') so re-runs are offline and resume-safe.

    python -B scripts/data/build_sites.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths  # noqa: E402
from sentinel.core.manifest import write_manifest  # noqa: E402
from sentinel.data import sites as S  # noqa: E402
from sentinel.data.sources import usgs_legacy as U  # noqa: E402

log = logging.getLogger("build_sites")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--catalog", default=str(paths.RAW / "usgs_dv" / "dv_series_catalog.parquet"))
    ap.add_argument("--no-neon", action="store_true")
    a = ap.parse_args()
    raw = paths.raw("site_registry")
    out_dir = paths.processed("sites")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(raw / "build_sites.log"), logging.StreamHandler()],
    )
    cat = pd.read_parquet(a.catalog)
    usgs = S.usgs_sites_from_catalog(cat, U.WQ_PARAMS)
    log.info("USGS sites with DV WQ series: %d", len(usgs))
    info = S.fetch_nwis_site_info(sorted(usgs["site_id"]), cache_dir=raw / "nwis_site")
    log.info("NWIS expanded site info rows: %d", len(info))
    neon = None if a.no_neon else S.neon_aquatic_sites(cache_dir=raw / "neon")
    if neon is not None:
        log.info("NEON aquatic sites: %d", len(neon))
        neon.to_parquet(raw / "neon_aquatic_sites.parquet", index=False)
    reg = S.build_registry(usgs, info, neon, wbd_lookup=lambda lon, lat: S.wbd_huc12(lon, lat, raw / "wbd"))
    # NHDPlus V2 COMID (shared site key convention: agency id + COMID + HUC12) once NLDI has run
    links_p = paths.PROCESSED / "site_network" / "nodes.parquet"
    if not links_p.exists():
        links_p = paths.RAW / "nldi" / "comid_links.parquet"
    if links_p.exists():
        links = pd.read_parquet(links_p, columns=["site_key", "comid", "comid_method"])
        reg = reg.merge(links, on="site_key", how="left")
        reg["comid"] = reg["comid"].astype("string")
        log.info("joined COMIDs from %s: %d sites linked", links_p, reg["comid"].notna().sum())
    out = out_dir / "sites.parquet"
    reg.to_parquet(out, index=False, compression="zstd")
    stats = {
        "n_sites": int(len(reg)),
        "n_usgs": int((reg.network == "USGS").sum()),
        "n_neon": int((reg.network == "NEON").sum()),
        "n_in_conus_bbox": int(reg.in_conus.sum()),
        "huc_source_counts": reg.huc_source.value_counts(dropna=False).astype(int).to_dict(),
        "missing_huc8": int(reg.huc8.isna().sum()),
        "missing_tz": int(reg.utc_offset_h.isna().sum()),
        "n_with_comid": int(reg["comid"].notna().sum()) if "comid" in reg else 0,
        "site_type_counts": reg.site_type.value_counts().astype(int).to_dict(),
        "state_counts": reg.state.value_counts().astype(int).to_dict(),
        "catalog": a.catalog,
        "retrieved": pd.Timestamp.utcnow().isoformat(),
    }
    write_manifest(
        "sites",
        files=[out, raw / "neon_aquatic_sites.parquet"] if neon is not None else [out],
        sources=[
            a.catalog,
            S.NWIS_SITE_URL + "?siteOutput=expanded",
            f"{S.NEON_API}/sites",
            f"{S.NEON_API}/locations/<site>",
            S.WBD_HUC12_URL,
        ],
        description="SENTINEL site registry: USGS sites with NWIS daily water-quality series + "
        "NEON aquatic sites; coordinates, type, state, HUC2/4/8/12, elevation, "
        "time zone, drainage area, WQ series summary.",
        license="US public domain (USGS); NEON data CC BY 4.0",
        stats=stats,
    )
    log.info("wrote %s (%d rows); stats=%s", out, len(reg), stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
