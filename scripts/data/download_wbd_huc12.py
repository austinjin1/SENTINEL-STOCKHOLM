#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Download the CONUS WBD HUC12 universe (codes, tohuc, approximate centroids) for D6 MusselGuard.

Sequential paged queries per HUC2 (01–18), 2,000 features/page, >= 2 s apart, cached
(raw('wbd_huc12')/cache). Writes processed('wbd_huc12')/huc12.parquet + manifest `wbd_huc12`.

    python -B scripts/data/download_wbd_huc12.py
"""

from __future__ import annotations

import argparse
import datetime as _dt
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths  # noqa: E402
from sentinel.core.manifest import write_manifest  # noqa: E402
from sentinel.data.sources import wbd_huc12 as W  # noqa: E402

log = logging.getLogger("wbd_huc12")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--huc2", default=",".join(f"{i:02d}" for i in range(1, 19)))
    a = ap.parse_args()
    raw = paths.raw("wbd_huc12")
    out_dir = paths.processed("wbd_huc12")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(raw / "download.log"), logging.StreamHandler()])
    client = W.WBDClient(raw / "cache")
    frames, per = [], {}
    for h2 in a.huc2.split(","):
        off = 0
        while True:
            body = client.get(W.query_params(h2, off))
            df = W.parse_page(body)
            frames.append(df)
            off += len(df)
            if len(df) < 2000 and not body.get("exceededTransferLimit"):
                break
            if len(df) == 0:
                break
        per[h2] = off
        log.info("HUC2 %s: %d features (stats %s)", h2, off, client.stats)
    d = W.tidy(pd.concat(frames, ignore_index=True))
    out = out_dir / "huc12.parquet"
    d.to_parquet(out, index=False, compression="zstd")
    stats = {"n_huc12": int(len(d)), "per_huc2": per, "n_centroid_missing": int(d["lat"].isna().sum()),
             "tohuc_kind": d["tohuc_kind"].value_counts().to_dict(), "http": client.stats,
             "retrieved_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
             "query": {k: v for k, v in W.query_params("01", 0).items() if k not in ("where", "resultOffset")}}
    write_manifest("wbd_huc12", files=[out], sources=[W.QUERY], license="US public domain (USGS WBD)",
                   description="CONUS WBD HUC12 codes, names, areas, states, tohuc and approximate centroids "
                               "(generalised geometry, 0.02 deg) — D6 MusselGuard candidate universe",
                   synthetic=False, stats=stats)
    log.info("wrote %s (%d rows)", out, len(d))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
