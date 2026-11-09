#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Event-layer feasibility [S6] side checks (no model labels): NRC report coordinate coverage in Iowa and
Delaware (why fish kills cannot be matched to NRC reports by distance), and the NEON qPCR target genes (why
E10 pathogen labels do not exist for M3). Plan §2.1, §3.3.

    python -B scripts/eval/events_feasibility_side_checks.py [--register]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import events_feasibility_common as C

from sentinel.core import paths

D = paths.PROCESSED


def main() -> int:
    a = C.parser(__doc__).parse_args()
    out = C.out_dir(a, "side_checks")
    nr = pq.read_table(D / "nrc_incidents" / "reports.parquet", columns=["location_state", "lat_decoded"]).to_pandas()
    table, rows = [], []
    for st in ("IA", "DE"):
        s = nr[nr["location_state"] == st]
        n_c = int(np.isfinite(s["lat_decoded"].astype(float)).sum())
        table.append({"check": f"NRC_{st}_reports", "value": len(s), "with_coordinates": n_c})
        rows.append({"model": "M4_chemtox", "target": f"NRC_{st}", "split": "all_years_metadata",
                     "metric": "n_reports_with_coordinates", "value": n_c, "n": len(s)})
    t = pq.read_table(D / "neon_microbe_qpcr" / "mga_swGroupAbundances.parquet", columns=["targetGene", "targetTaxonGroup"])
    genes = sorted(pc.unique(t["targetGene"]).to_pylist())
    groups = sorted(pc.unique(t["targetTaxonGroup"]).to_pylist())
    table.append({"check": "NEON_qPCR_targetGene", "value": len(genes), "with_coordinates": None,
                  "detail": ";".join(map(str, genes)) + " | " + ";".join(map(str, groups))})
    rows.append({"model": "M3_microbe", "target": "E10_faecal_markers", "split": "all_years_metadata",
                 "metric": "n_qpcr_target_genes", "value": len(genes), "n": t.num_rows,
                 "extra": {"genes": genes, "taxon_groups": groups}})
    res = pd.DataFrame(table)
    f = C.write(res, out, "side_checks", {"created": C.stamp()})
    print(res.to_string(index=False))
    if a.register:
        print(f"registered {C.register(rows, script=__file__, artifacts=[f], data_manifests=['nrc_incidents'])} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
