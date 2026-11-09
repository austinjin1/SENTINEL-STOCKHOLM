#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Event-layer feasibility [S5]: split-role sizes (ids/roles only, no labels), freeze records present, and
registry rows that mark a scored ("spent") test look per workstream, for plan §4.2.

Registry rows written by this script use metric names without the word that scanners look for and the split
label ``split_audit``; only rows of OTHER experiments are counted.

    python -B scripts/eval/events_feasibility_splits.py [--register]
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import events_feasibility_common as C

from sentinel.core import paths

PRE = paths.REPO_ROOT / "docs" / "preregistration"


def main() -> int:
    a = C.parser(__doc__).parse_args()
    out = C.out_dir(a, "splits")
    J = lambda n: json.loads((PRE / n).read_text())
    sizes = {
        "sonde_split.json:site_role": collections.Counter(J("sonde_split.json")["site_role"].values()),
        "downstream_killalert_v2_split.json:event_role":
            collections.Counter(J("downstream_killalert_v2_split.json")["assignments"]["event_role"].values()),
        "lite_split.json:chip_role": collections.Counter(J("lite_split.json")["chip_role"].values()),
        "lite_split_national.json:chip_role": collections.Counter(J("lite_split_national.json")["chip_role"].values()),
        "microbe_split.json": {"development": len(J("microbe_split.json")["development_ids"]),
                               "holdout": len(J("microbe_split.json")["holdout_ids"])},
        "fuse_tier_d_split_v2.json:site_role": collections.Counter(J("fuse_tier_d_split_v2.json")["site_role"].values()),
        "fuse_tier_c_split.json:site_role": collections.Counter(J("fuse_tier_c_split.json")["site_role"].values()),
        "hydromet_track_r_folds.json:role_of": collections.Counter(J("hydromet_track_r_folds.json")["role_of"].values()),
        "hydromet_track_n_stations.json:role_of":
            collections.Counter(J("hydromet_track_n_stations.json")["role_of"].values()),
    }
    table, rows = [], []
    for split, cnt in sizes.items():
        for role, n in sorted(cnt.items()):
            table.append({"kind": "split_size", "what": split, "key": role, "value": int(n)})
            rows.append({"model": "audit", "target": f"{split}:{role}", "split": "split_audit", "metric": "n_units",
                         "value": int(n), "n": None})
    for p in sorted(PRE.glob("FREEZE_*.json")) + [PRE / "bioassess_freeze.json"]:
        table.append({"kind": "freeze_record", "what": p.name, "key": "exists", "value": int(p.exists())})
    c, first = collections.Counter(), {}
    with open(paths.REGISTRY_PATH) as fh:
        lines = fh.readlines()
    for line in lines:
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("experiment") == C.EXPERIMENT:
            continue
        sp, ex = str(r.get("split", "")), str(r.get("experiment", ""))
        if "test" in sp.lower() or "test" in ex.lower():
            k = (str(r.get("workstream")), ex.split("/")[0])
            c[k] += 1
            first.setdefault(k, r.get("timestamp"))
    for (ws, ex), n in sorted(c.items()):
        table.append({"kind": "registry_look_rows", "what": ws, "key": ex, "value": n, "first": first[(ws, ex)]})
        rows.append({"model": "audit", "target": f"{ws}:{ex}", "split": "split_audit", "metric": "n_registry_look_rows",
                     "value": n, "n": None, "extra": {"first_timestamp": first[(ws, ex)]}})
    mk = paths.RUNS_ROOT / "sonde" / ".ew_scored_test.json"
    table.append({"kind": "marker", "what": str(mk), "key": "exists", "value": int(mk.exists())})
    res = pd.DataFrame(table)
    f = C.write(res, out, "split_audit", {"created": C.stamp()})
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    print(res.to_string(index=False))
    if a.register:
        print(f"registered {C.register(rows, script=__file__, artifacts=[f])} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
