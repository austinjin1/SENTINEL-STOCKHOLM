#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""GLORIA -> tidy samples with simulated sensor-band Rrs (S2A/S2B MSI, L8 OLI, L9 OLI-2, cameras).

Inputs: raw('gloria')/GLORIA-2022.zip, raw('srf')/*.xlsx (scripts/data/download_gloria.py).
Outputs (processed('gloria')):
  gloria_samples.parquet        7,572 samples: ids (GLORIA_ID, Dataset_ID, Site_name), coordinates,
                                time, water-body type, chl-a/TSS/aCDOM440/Secchi/turbidity + unit
                                columns, methods, QC flags, simulated band Rrs ``rrs_<sensor>_<band>``
  gloria_rrs_1nm.parquet        GLORIA_ID + Rrs_350..Rrs_900 (sr^-1, float32) and Rrs_std
  srf_1nm.parquet, srf_info.parquet   SRFs used on the 1 nm grid + band coverage table
  summary.json

    python -B scripts/data/build_gloria.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths  # noqa: E402
from sentinel.core.manifest import write_manifest  # noqa: E402
from sentinel.data.sources import gloria as G  # noqa: E402

log = logging.getLogger("gloria_build")


def main() -> int:
    out = paths.processed("gloria")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(out / "build.log"), logging.StreamHandler()])
    zp = paths.raw("gloria") / "GLORIA-2022.zip"
    srf_dir = paths.raw("srf")
    tables = G.read_gloria(zp)
    R = G.rrs_matrix(tables["rrs"])
    srf, info = G.srf_table(srf_dir / "S2-SRF_COPE-GSEG-EOPG-TN-15-0007_3.2.xlsx",
                            srf_dir / "L8_OLI_RSR.xlsx", srf_dir / "L9_OLI2_RSR.xlsx")
    log.info("SRFs: %d bands simulated, %d not (outside 350-900 nm)", info.simulated.sum(), (~info.simulated).sum())
    sim = G.simulate_bands(R, srf)
    df = G.tidy_samples(tables, sim)
    files = []
    p = out / "gloria_samples.parquet"
    df.to_parquet(p, index=False, compression="zstd")
    files.append(p)
    hyp = tables["rrs"].copy()
    hyp[[c for c in hyp.columns if c != "GLORIA_ID"]] = hyp[[c for c in hyp.columns if c != "GLORIA_ID"]].astype("float32")
    if "rrs_std" in tables:
        sd = tables["rrs_std"].copy()
        sd[[c for c in sd.columns if c != "GLORIA_ID"]] = sd[[c for c in sd.columns if c != "GLORIA_ID"]].astype("float32")
        hyp = hyp.merge(sd, on="GLORIA_ID", how="left")
    p = out / "gloria_rrs_1nm.parquet"
    hyp.to_parquet(p, index=False, compression="zstd")
    files.append(p)
    for name, t in (("srf_1nm.parquet", srf), ("srf_info.parquet", info)):
        t.to_parquet(out / name, index=False)
        files.append(out / name)
    summ = G.summarize(df, info)
    (out / "summary.json").write_text(json.dumps(summ, indent=1, default=str))
    files.append(out / "summary.json")
    log.info("summary: %s", {k: summ[k] for k in ("n_samples", "n_datasets", "any_qc_flag", "targets")})
    write_manifest("gloria", files=files, sources=[G.GLORIA_URL] + list(G.SRF_URLS.values()),
                   description="GLORIA samples (Lehmann et al. 2023) with water-quality targets, QC flags and "
                               "simulated band Rrs for S2A/S2B MSI (ESA SRF v3.2), L8 OLI / L9 OLI-2 (USGS RSR) and "
                               "two camera band models (top-hat approximations).",
                   license=G.LICENSE, synthetic=False, stats=json.loads(json.dumps(summ, default=str)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
