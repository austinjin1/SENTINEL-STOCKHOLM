#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Derive the OGC-qualifier handling of the live trial from agreement with the LEGACY training tier (review W-D1).

Data: the cached OGC ``daily`` records of WY2022–23 (val block) at train/val-role live-config sites
(``dryrun/label_identity/records_wy2022_23.parquet``) and the frozen legacy snapshot. No trial outcome, no test block.
Only single-series (site, parameter, statistic) keys are used (no series-choice confound), and only values carrying
exactly ONE qualifier (a value with several qualifiers would attribute one qualifier's masking to the others; the first
run of 2026-11-09 counted multi-qualifier values and was revised for this reason, thresholds unchanged). Values with
several qualifiers are masked if any of their qualifiers is MASK.

Pre-specified rule, per qualifier name q (over all values carrying q, before any masking):
  frac_missing = share whose (site, variable, date) value is ABSENT in the legacy tier
  frac_equal   = share present in legacy with |Δ| ≤ 1e-4
  * MASK  if frac_missing ≥ 0.5                (the legacy tier dropped these values)
  * KEEP  if frac_equal ≥ 0.9                  (the legacy tier kept them unchanged)
  * MASK  otherwise (conservative), and flagged
  * qualifiers with no value / not seen: MASK unless they never carry values (handled by the value being null)
Writes ``dryrun/qualifier_map/derived.json`` (table + decisions); ``live_data.MASK_OGC`` / ``KEEP_OGC`` must equal it
(test ``test_qualifier_map_matches_derivation``).

    python -B scripts/live/derive_qualifier_map.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import live_common as LC
from sentinel.data.sources.usgs_legacy import ALL_PARAMS, STATS

CACHE = LC.DRYRUN_ROOT / "label_identity" / "records_wy2022_23.parquet"
OUT = LC.DRYRUN_ROOT / "qualifier_map" / "derived.json"


def _names(q) -> list[str]:
    if q is None or (isinstance(q, float) and np.isnan(q)):
        return []
    q = str(q)
    if q in ("None", "<NA>", "nan", ""):
        return []
    return [t.strip(" '\"").upper() for t in q.strip("[]").split(",") if t.strip(" '\"")]


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--first-run-method", action="store_true",
                    help="reproduce the superseded first run (values with several qualifiers counted for each)")
    a = ap.parse_args()
    out = OUT.with_name("derived_firstrun.json") if a.first_run_method else OUT
    cfg, _ = LC.load_config()
    r = pd.read_parquet(CACHE)
    r = r[r["parameter_code"].isin(ALL_PARAMS) & r["statistic_id"].isin(STATS)].copy()
    r["site"] = r["monitoring_location_id"].str.replace("USGS-", "", regex=False)
    key = ["site", "parameter_code", "statistic_id"]
    single = r.groupby(key)["time_series_id"].transform("nunique") == 1
    r = r[single & r["value"].notna()].copy()
    r["var"] = r["parameter_code"].map(ALL_PARAMS) + "_" + r["statistic_id"].map(STATS)
    r["date"] = pd.to_datetime(r["time"].astype(str).str[:10])
    r["qn"] = r["qualifier"].map(_names)
    snapdir = __import__("pathlib").Path(cfg["data_pins"]["sonde_daily_dir"])
    legacy = {}
    for s in r["site"].unique():
        p = snapdir / "sites" / f"{s}.parquet"
        if p.exists():
            d = pd.read_parquet(p)
            d.index = pd.to_datetime(d.index)
            legacy[s] = d
    rows = []
    hz_vars = {"do_mgl_min", "do_mgl_mean", "spcond_uscm_mean", "spcond_uscm_max", "no3no2_mgl_mean"}
    for _, g in r[r["qn"].map(len) > 0 if a.first_run_method else r["qn"].map(len) == 1].iterrows():
        d = legacy.get(g["site"])
        lv = np.nan
        if d is not None and g["var"] in d.columns and g["date"] in d.index:
            lv = d.at[g["date"], g["var"]]
        for q in g["qn"]:
            rows.append({"q": q, "var": g["var"], "hazard_var": g["var"] in hz_vars, "missing": not np.isfinite(lv),
                         "equal": bool(np.isfinite(lv) and abs(float(lv) - float(g["value"])) <= 1e-4)})
    t = pd.DataFrame(rows)
    table = {}
    for q, g in t.groupby("q"):
        fm, fe = float(g["missing"].mean()), float(g["equal"].mean())
        dec = "mask" if fm >= 0.5 else ("keep" if fe >= 0.9 else "mask")
        table[q] = {"n_values": int(len(g)), "frac_missing_in_legacy": round(fm, 4), "frac_equal_legacy": round(fe, 4),
                    "decision": dec, "flag": bool(fm < 0.5 and fe < 0.9),
                    "hazard_vars": {"n": int(g["hazard_var"].sum()),
                                    "frac_missing": round(float(g.loc[g["hazard_var"], "missing"].mean()), 4)
                                    if g["hazard_var"].any() else None}}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rule": __doc__.split("Pre-specified rule")[1].split("Writes")[0].strip(),
                               "data": str(CACHE), "table": table}, indent=1))
    print(json.dumps(table, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
