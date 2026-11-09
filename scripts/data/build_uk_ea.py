#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Build processed('uk_ea_ecology') from raw('uk_ea'): EA invertebrate + diatom communities,
site/sample tables, taxon dictionary, WQ sampling points, site<->WQ pairs, paired chemistry.

Requires ``scripts/data/download_uk_ea.py ecology|points|observations`` (manifests
``uk_ea_ecology_raw``, ``uk_ea_wq_points_raw``, ``uk_ea_wq_observations_raw``). Without the
observations stage the chemistry tables are omitted (logged in the manifest stats).

    python -B scripts/data/build_uk_ea.py
"""

from __future__ import annotations

import datetime as dt
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths  # noqa: E402
from sentinel.core.manifest import read_manifest, write_manifest  # noqa: E402
from sentinel.data import bio_uk_ea as BU  # noqa: E402
from sentinel.data.sources import uk_ea as U  # noqa: E402

log = logging.getLogger("build_uk_ea")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raw = paths.raw("uk_ea")
    out = paths.processed("uk_ea_ecology")
    for old in out.glob("*.parquet"):
        old.unlink()
    stats = BU.build(raw, out)
    upstream = {}
    for m in ("uk_ea_ecology_raw", "uk_ea_wq_points_raw", "uk_ea_wq_observations_raw"):
        try:
            upstream[m] = read_manifest(m)["created"]
        except FileNotFoundError:
            upstream[m] = None
    write_manifest(
        "uk_ea_ecology", files=sorted(out.glob("*.parquet")),
        sources=[U.ECOLOGY_DOWNLOADS_URL + f for f in U.ECOLOGY_FILES] + [U.WQ_SAMPLING_POINT_URL,
                                                                        U.WQ_OBSERVATION_URL],
        description="Environment Agency (England) freshwater invertebrate and diatom communities "
                    "(1965-2026) with site/sample metadata and published indices, plus Water Quality "
                    "Archive chemistry paired by nearest freshwater sampling point (<=1 km) and "
                    "date windows (+-30 d, prior 365 d).",
        license=U.LICENSE,
        stats={"built": dt.date.today().isoformat(), "upstream_manifests": upstream,
               "pairing_rule": BU.__doc__.split("Pairing rule")[1].strip()[:1500], **stats},
    )
    log.info("done: %s", {k: v for k, v in stats.items() if k in ("sites", "pairing")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
