#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Build processed('nars_streams') and processed('nars_lakes') from raw('nars').

Requires ``scripts/data/download_nars.py`` to have run (manifest ``nars_raw``).

    python -B scripts/data/build_nars.py [--resource streams lakes]
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths  # noqa: E402
from sentinel.core.manifest import read_manifest, write_manifest  # noqa: E402
from sentinel.data import bio_nars as B  # noqa: E402
from sentinel.data.sources import nars as N  # noqa: E402

log = logging.getLogger("build_nars")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resource", nargs="*", default=["streams", "lakes"], choices=["streams", "lakes"])
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raw_root = paths.raw("nars")
    raw_manifest = read_manifest("nars_raw")
    for res in a.resource:
        name = f"nars_{res}"
        out = paths.processed(name)
        for old in out.glob("*.parquet"):
            old.unlink()
        stats = B.build_resource(raw_root, out, res)
        surveys = N.STREAM_SURVEYS if res == "streams" else N.LAKE_SURVEYS
        write_manifest(
            name, files=sorted(out.glob("*.parquet")) + sorted(out.glob("*.json")) + sorted(out.glob("*.csv")),
            sources=[N.DATA_PAGE_URL],
            description=f"EPA NARS {'NRSA rivers & streams' if res == 'streams' else 'NLA lakes'} "
                        f"({', '.join(N.CYCLE_LABEL[s] for s in surveys)}): visits, community long tables, "
                        "chemistry (long + wide, canonical units, censoring), condition labels, physical "
                        "habitat" + (", depth profiles" if res == "lakes" else ", periphyton biomass") + ".",
            license="US Government work (EPA); no license statement on the NARS data hub",
            stats={"built": dt.date.today().isoformat(), "raw_manifest": "nars_raw",
                   "raw_manifest_created": raw_manifest["created"], **stats},
        )
        log.info("%s: %s", name, stats["rows"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
