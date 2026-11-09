#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Download GLORIA (PANGAEA 948492, CC BY 4.0) and the official SRF tables (ESA S2 MSI v3.2,
USGS Landsat 8 OLI and Landsat 9 OLI-2 band-average RSRs) into raw('gloria') and raw('srf').

    python -B scripts/data/download_gloria.py
"""

from __future__ import annotations

import datetime as dt
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths  # noqa: E402
from sentinel.core.manifest import write_manifest  # noqa: E402
from sentinel.data.sources import gloria as G  # noqa: E402

log = logging.getLogger("gloria_dl")


def main() -> int:
    raw, srf = paths.raw("gloria"), paths.raw("srf")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(raw / "download.log"), logging.StreamHandler()])
    files = G.download_all(raw, srf, log=log.info)
    when = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    write_manifest("gloria_raw", files=[f for f in files if f.parent == raw], sources=[G.GLORIA_URL],
                   description="GLORIA-2022.zip (Lehmann et al. 2023): hyperspectral Rrs 350-900 nm + chl-a, TSS, "
                               "aCDOM440, Secchi, turbidity, QC flags, 7,572 samples.",
                   license=G.LICENSE, stats={"doi": G.GLORIA_DOI, "retrieved_utc": when})
    write_manifest("srf_tables_raw", files=[f for f in files if f.parent == srf], sources=list(G.SRF_URLS.values()),
                   description="Official spectral response tables: ESA Sentinel-2 SRF v3.2 (S2A/S2B), USGS L8 OLI and "
                               "L9 OLI-2 band-average RSR workbooks.",
                   license="ESA Copernicus documentation (public); USGS (public domain)",
                   stats={"retrieved_utc": when})
    log.info("done: %s", [f.name for f in files])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
