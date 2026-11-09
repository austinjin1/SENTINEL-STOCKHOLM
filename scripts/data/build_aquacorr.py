#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Audit AquaCorr-Global and convert it to a tidy Parquet table; select S2 rows for L2A chips.

Inputs: raw('aquacorr_global') from scripts/data/download_aquacorr.py.
Outputs (processed('aquacorr_global')):
  aquacorr_matchups.parquet   all 1.98 M matchups, decoded source flags, assumed-unit columns,
                              duplicate-structure and spectral-plausibility columns
  audit.json                  audit numbers (duplicates, windows, chl-a methods, QC, geography)
  s2_chip_requests.parquet    S2 rows chosen for our own L2A chips (US + Europe, passed_qc,
                              priority lab chl-a > Secchi > other chl-a > TSS/turbidity,
                              Europe quota, total budget)

    python -B scripts/data/build_aquacorr.py --budget 120000 --europe-quota 20000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths  # noqa: E402
from sentinel.core.manifest import write_manifest  # noqa: E402
from sentinel.data.sources import aquacorr as A  # noqa: E402

log = logging.getLogger("aquacorr_build")
SEED = 20261022


def jsonable(o):
    """Recursively convert dict keys to str and numpy scalars to Python types."""
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    return o


def select_s2(t: pd.DataFrame, budget: int, europe_quota: int) -> pd.DataFrame:
    """Priority selection of S2 matchups for L2A chip extraction (deterministic)."""
    s2 = t[t.sensor.isin(A.S2_SENSORS) & t.passed_qc & (t.coord_centroid == 0)
           & t.region.isin(["US_conus", "US_alaska", "US_hawaii", "US_puerto_rico", "Europe"])].copy()
    has = s2[["chla_value", "tss_value", "sdd_value", "turbidity_value"]].notna()
    s2 = s2[has.any(axis=1)]
    # One row per physical acquisition: AquaCorr repeats a sample for every overlapping MGRS tile
    # of the same overpass. Keep the tile in the site's own UTM zone, then most valid pixels.
    zone = (np.floor((s2.lon + 180) / 6) + 1).astype(int).astype(str).str.zfill(2)
    s2["_own_zone"] = (s2.mgrs_tile.astype(str).str[:2] == zone).astype(int)
    s2 = s2.sort_values(["_own_zone", "window_valid_pixels", "scene_id"], ascending=[False, False, True])
    s2["n_tiles_same_acquisition"] = s2.groupby(["site_id", "date_insitu", "sensor"]).scene_id.transform("nunique")
    s2 = s2.drop_duplicates(["site_id", "date_insitu", "sensor"]).drop(columns="_own_zone")
    rng = np.random.default_rng(SEED)
    rec = s2.chla_recommended.fillna(False).astype(bool) & s2.chla_value.notna()
    s2["priority"] = (8 * rec + 4 * s2.sdd_value.notna() + 2 * s2.chla_value.notna()
                      + 1 * (s2.tss_value.notna() | s2.turbidity_value.notna())).astype(int)
    s2["_tie"] = rng.random(len(s2))
    s2 = s2.sort_values(["priority", "_tie"], ascending=[False, True])
    eu = s2[s2.region == "Europe"].head(europe_quota)
    us = s2[s2.region != "Europe"].head(budget - len(eu))
    sel = pd.concat([us, eu]).drop(columns="_tie")
    # one chip per (point, scene): rows with identical coordinates and scene share a chip
    sel["chip_point_key"] = sel.lat.round(6).map("{:.6f}".format) + "_" + sel.lon.round(6).map("{:.6f}".format)
    sel["chip_request_key"] = sel.scene_id.astype(str) + "|" + sel.chip_point_key
    return sel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=120_000)
    ap.add_argument("--europe-quota", type=int, default=20_000)
    a = ap.parse_args()
    raw = paths.raw("aquacorr_global")
    out = paths.processed("aquacorr_global")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(out / "build.log"), logging.StreamHandler()])
    src = raw / "aquacorr_matchups.parquet"
    meta = json.loads((raw / "aquacorr_metadata.json").read_text())
    rec = json.loads((raw / "zenodo_record.json").read_text())
    log.info("reading %s", src)
    t = A.tidy_matchups(pd.read_parquet(src))
    log.info("tidy: %d rows x %d cols", len(t), t.shape[1])
    audit = A.audit_matchups(t, meta)
    audit["zenodo_files_published"] = sorted(f["key"] for f in rec.get("files", []))
    audit["files_listed_in_metadata_but_not_published"] = sorted(
        set(meta.get("files", {})) - set(audit["zenodo_files_published"]))
    (out / "audit.json").write_text(json.dumps(jsonable(audit), indent=1, default=str))
    log.info("audit written")
    tidy_path = out / "aquacorr_matchups.parquet"
    t.to_parquet(tidy_path, index=False, compression="zstd")
    sel = select_s2(t, a.budget, a.europe_quota)
    req_path = out / "s2_chip_requests.parquet"
    sel.to_parquet(req_path, index=False, compression="zstd")
    stats = {"n_rows": len(t), "n_s2_selected_rows": len(sel),
             "n_s2_selected_chip_requests": int(sel.chip_request_key.nunique()),
             "selected_by_region": sel.region.value_counts().to_dict(),
             "selected_by_priority": sel.priority.value_counts().sort_index().to_dict(),
             "selected_values": {v: int(sel[f"{v}_value"].notna().sum()) for v in
                                 ("chla", "tss", "sdd", "turbidity")},
             "budget": a.budget, "europe_quota": a.europe_quota, "seed": SEED}
    log.info("selection: %s", stats)
    write_manifest("aquacorr_global", files=[tidy_path, out / "audit.json", req_path],
                   sources=[A.RECORD_API, f"https://doi.org/{A.ZENODO_DOI}"],
                   description="AquaCorr-Global v1.0 matchups, tidy Parquet with decoded source flags, "
                               "assumed-unit columns, duplicate and spectral-plausibility columns; audit; "
                               "S2 rows selected for SENTINEL L2A chip extraction.",
                   license=A.LICENSE, synthetic=False, stats=jsonable(stats))
    log.info("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
