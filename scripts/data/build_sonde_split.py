#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Create the SENTINEL-Sonde / EW-v2 site split (RESEARCH_PLAN.md §14).

    python scripts/data/build_sonde_split.py --dev     # temporary, smoke tests only
    python scripts/data/build_sonde_split.py           # final: requires site_network edges

The final split merges HUC4 units across flow-connected stations (NLDI edges) and is written
once to docs/preregistration/sonde_split.json with its SHA-256; it must not be regenerated
after model selection starts (the script refuses to overwrite it).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from sentinel.core import paths
from sentinel.data.sonde_splits import check_no_cross_role_edges, make_split

PREREG = paths.REPO_ROOT / "docs" / "preregistration"
SURFACE = {"ST", "ST-TS", "ST-CA", "ST-DCH"}  # M1 headline = streams/rivers; lakes & estuaries are separate tracks
TARGET_VARS = ["temp_c_mean", "do_mgl_mean", "do_mgl_min", "spcond_uscm_mean", "ph_mean", "ph_median",
               "turb_fnu_mean", "turb_fnu_median", "no3no2_mgl_mean", "chl_ugl_mean", "fdom_qse_mean"]


def benchmark_sites(min_days: int = 730) -> pd.DataFrame:
    reg = pd.read_parquet(paths.PROCESSED / "sites" / "sites.parquet")
    reg = reg[(reg["network"] == "USGS") & reg["site_type"].isin(SURFACE)].copy()
    idx = pd.read_parquet(paths.PROCESSED / "sonde_daily" / "index.parquet")
    ok = idx[idx["var"].isin(TARGET_VARS) & (idx["n_days"] >= min_days)]["site_no"].unique()
    reg = reg[reg["site_id"].isin(ok)]
    out = pd.DataFrame({"site_no": reg["site_id"].astype(str), "lat": reg["lat"], "lon": reg["lon"],
                        "huc_cd": reg["huc8"].astype(str)})
    return out.dropna(subset=["lat", "lon"]).reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true")
    a = ap.parse_args()
    sites = benchmark_sites()
    edges_path = paths.PROCESSED / "site_network" / "edges.parquet"
    edges = None
    if edges_path.exists():
        e = pd.read_parquet(edges_path)
        strip = lambda s: s.astype(str).str.replace("USGS-", "", regex=False)  # noqa: E731
        edges = pd.DataFrame({"src": strip(e["src"]), "dst": strip(e["dst"])})
    elif not a.dev:
        raise SystemExit("final split needs processed/site_network/edges.parquet (flow connectivity)")
    name = "sonde_split_dev.json" if a.dev else "sonde_split.json"
    dest = PREREG / name
    if dest.exists() and not a.dev:
        raise SystemExit(f"{dest} exists — the final split is frozen and must not be regenerated")
    note = ("DEV split for smoke tests only — NOT for model selection or reporting"
            if a.dev else "FINAL split (frozen at creation; see ew_v2_protocol.md §10)")
    sp = make_split(sites, edges, notes=note)
    dest.write_text(sp.to_json())
    rep = {"file": name, "sha256": sp.sha256(), "n_sites": len(sites), "notes": note,
           "roles": {r: len(sp.sites(r)) for r in ("train", "val", "test")},
           "n_basin_groups": len(set(sp.basin_group.values())),
           "edges_used": 0 if edges is None else int(len(edges)),
           "cross_role_edges": 0 if edges is None else check_no_cross_role_edges(sp, edges)}
    idx = pd.read_parquet(paths.PROCESSED / "sonde_daily" / "index.parquet")
    idx = idx[idx["var"].isin(TARGET_VARS) & (idx["n_days"] >= 730)]
    idx["role"] = idx["site_no"].map(sp.site_role)
    rep["sites_per_var_role"] = idx.dropna(subset=["role"]).groupby(["var", "role"]).size().unstack(
        fill_value=0).to_dict(orient="index")
    (PREREG / name.replace(".json", "_report.json")).write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps({k: v for k, v in rep.items() if k != "sites_per_var_role"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
