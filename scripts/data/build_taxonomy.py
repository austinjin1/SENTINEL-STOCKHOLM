#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Build processed('taxonomy_harmonised'): source taxon -> harmonised lineage + analysis resolution.

Requires processed('nars_streams'), processed('nars_lakes'), processed('uk_ea_ecology') and the
GBIF cache from ``scripts/data/download_taxonomy_gbif.py`` (manifest ``taxonomy_gbif_raw``).

    python -B scripts/data/build_taxonomy.py
"""

from __future__ import annotations

import datetime as dt
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths  # noqa: E402
from sentinel.core.manifest import read_manifest, write_manifest  # noqa: E402
from sentinel.data import bio_taxonomy as T  # noqa: E402

log = logging.getLogger("build_taxonomy")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = paths.processed("taxonomy_harmonised")
    for old in out.glob("*.parquet"):
        old.unlink()
    jsonl = paths.raw("taxonomy_gbif") / "gbif_matches.jsonl"
    summary = T.build({r: paths.processed(r) for r in ("nars_streams", "nars_lakes")},
                      paths.processed("uk_ea_ecology"), jsonl, out)
    upstream = {}
    for m in ("nars_streams", "nars_lakes", "uk_ea_ecology", "taxonomy_gbif_raw"):
        try:
            upstream[m] = read_manifest(m)["created"]
        except FileNotFoundError:
            upstream[m] = None
    write_manifest("taxonomy_harmonised", files=sorted(out.glob("*.parquet")) + [out / "summary.json"],
                   sources=["processed:nars_streams", "processed:nars_lakes", "processed:uk_ea_ecology",
                            "https://api.gbif.org/v1/species/match"],
                   description="Harmonised taxonomy for NARS (benthic, fish, zooplankton, phytoplankton) and EA "
                               "(invertebrates, diatoms) source taxa: cleaned names, identification rank, lineage "
                               "(GBIF backbone or source consensus), analysis resolution, mixed-resolution OTUs, "
                               "conflicts.",
                   license="derived: US Government (NARS) + OGL v3.0 (EA) + GBIF Backbone (CC BY 4.0)",
                   stats={"built": dt.date.today().isoformat(), "upstream_manifests": upstream, **summary})
    log.info("taxonomy: %s", {k: summary[k] for k in ("source_taxa", "lineage_source_counts", "names_with_conflict")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
