#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Single machine-readable source of the new-site populations (ew_v3_protocol.md / m1n_protocol.md §2a–§2b).
Inputs: the frozen list, ``sonde_newsites_overlap.json``, ``sonde_newsites_v1_audit.json`` and the chloride regression
table (site IDs + eligible flag only). Writes ``docs/preregistration/sonde_newsites_tiers.json`` (refuses to overwrite).

  hard   = v1 OUTCOME-level use (C-F1) | loaded by a recorded run | M6 track-N station within 1 km | Powell DO-derived labels | eligible SC->Cl
           regression already built in the shared chloride tier (derived H2 labels)
  legacy = v1 value-level use, not hard (lead ruling 2026-11-10; kept in the primary, excluded in S1b)
  full   = §2a full exposure set (hard tier of §2a + nominal DEV-split members)
  primary = all - hard ; S1b = all - hard - legacy ; S1 = all - full - hard (subset of the primary) ; S2 = all
"""
import hashlib
import json

import pandas as pd

from sentinel.core import paths

P = paths.REPO_ROOT / "docs" / "preregistration"
OUT = P / "sonde_newsites_tiers_v2.json"   # v2: C-F1 (v1 outcome-level use -> hard)
if OUT.exists():
    raise SystemExit(f"{OUT} exists")
ids = {s["site_no"] for s in json.loads((P / "sonde_newsites_list.json").read_text())["sites"]}
ov = json.loads((P / "sonde_newsites_overlap.json").read_text())
v1 = json.loads((P / "sonde_newsites_v1_audit_v2.json").read_text())
reg = pd.read_parquet(paths.PROCESSED / "chloride_derived" / "regressions.parquet")
reg["site_no"] = reg["site_no"].astype(str)
cl = set(reg.loc[reg["eligible"].astype(bool) & reg["site_no"].isin(ids), "site_no"])
hard = set(ov["loaded_by_any_recorded_run"]) | set(ov["track_n_colocated"]["1000m"]["sites"]) | \
    set(ov["catalogue_exposure_not_splits"]["labels_powell_metabolism/sites.parquet"][1]) | cl | \
    set(v1["v1_outcome_level_to_hard"]["sites"])
legacy = set(v1["legacy_tier"]["sites"]) - hard
full = set(ov["exclusion_set"]["sites"]) | cl
pops = {"primary": sorted(ids - hard), "S1b": sorted(ids - hard - legacy), "S1": sorted(ids - full - hard), "S2": sorted(ids)}
rec = {"hard": sorted(hard), "hard_components": {"chloride_eligible_site": sorted(cl),
                                                  "v1_outcome_level": v1["v1_outcome_level_to_hard"]["sites"]},
       "legacy": sorted(legacy),
       "full": sorted(full), "populations": pops, "n": {k: len(v) for k, v in pops.items()},
       "inputs_sha256": {f: hashlib.sha256((P / f).read_bytes()).hexdigest() for f in
                         ("sonde_newsites_list.json", "sonde_newsites_overlap.json", "sonde_newsites_v1_audit_v2.json")}}
OUT.write_text(json.dumps(rec, indent=1))
OUT.chmod(0o444)
print(json.dumps({"n": rec["n"], "hard": len(hard), "legacy": len(legacy), "full": len(full),
                  "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest()}))
