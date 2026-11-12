#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""COUNTS ONLY for the M3 architecture-round-v2 amendment (docs/preregistration/microbe_arch_v2_amendment.md).

No label value is printed or stored. For every candidate analyte it counts community events with a same-visit value
(the builder's own key: ``sentinel.data.microbe_wq.collapse_visits`` on ``chem_parent``), censored flags, sites,
domains and time-matched rows, separately for

* ``002/dev``      DP1.20141.002 events before 2024-01-01 (development),
* ``002/holdout``  DP1.20141.002 events from 2024-01-01 (protected: counts only),
* ``001/train1718`` DP1.20141.001 events of 2017-2018 (``lodo_x001`` training rows),
* ``001/reserve``  DP1.20141.001 events before 2017-01-01 (protected cross-release reserve: counts only).

It also stores detection-limit METADATA of the lab records (not values) and the label-free coverage of the qPCR
input (DP1.20278.001 16S group abundances joined by dnaSampleID; filtered volume by geneticSampleID).

    PYTHONPATH=. python scripts/data/microbe_geo_counts.py <out.json> [--data-root DIR]
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from sentinel.data.microbe_wq import collapse_visits

GEO8 = ["lab_ph", "cl_mgl", "so4_mgl", "ca_mgl", "dic_mgl", "field_do_sat_pct", "nh4_n_mgl",
        "uv_absorbance_254_nm_au"]
OTHER = ["nh4_n_ugl", "mg_mgl", "na_mgl", "k_mgl", "si_mgl", "f_mgl", "tds_mgl", "tdn_mgl", "tdp_mgl",
         "field_spcond_uscm", "field_do_mgl", "field_temp_c", "ortho_p_mgl", "ortho_p_ugl", "tss_mgl", "br_mgl",
         "lab_anc_meql", "anc_meql", "lab_alk_meql", "fe_mgl", "mn_mgl", "no2_n_mgl", "toc_mgl",
         "uv_absorbance_280_nm_au", "specificconductance_uscm"]
VARS = GEO8 + OTHER
FLOORED = ["cl_mgl", "so4_mgl", "ca_mgl", "dic_mgl", "nh4_n_mgl", "uv_absorbance_254_nm_au"]


def block(release: str, dates: pd.Series) -> np.ndarray:
    d = pd.to_datetime(dates)
    if release == "002":
        return np.where(d < "2024-01-01", "dev", "holdout")
    return np.where(d < "2017-01-01", "reserve", "train1718")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--data-root", default="/home/bcheng/STOCKHOLM/sentinel_data/processed/")
    a = ap.parse_args(argv)
    D = a.data_root.rstrip("/") + "/"
    w = pd.read_parquet(D + "neon_chem/chem_event_wide.parquet")
    w = w[~w["provisional"].fillna(False).astype(bool)]
    w["time_utc"] = pd.to_datetime(w["time_utc"], utc=True)
    c = collapse_visits(w, VARS).set_index("parentSampleID")

    q = pd.read_parquet(D + "neon_microbe_qpcr/mga_swGroupAbundances.parquet",
                        columns=["dnaSampleID", "targetGene", "meanCopyNumber", "provisional", "laboratoryName",
                                 "batchID"])
    q = q[(q["targetGene"] == "16S") & ~q["provisional"].fillna(False).astype(bool)]
    q_ok = set(q.loc[q["meanCopyNumber"].notna(), "dnaSampleID"].astype(str))
    fg = pd.read_parquet(D + "neon_microbe_seqmeta/amc_fieldGenetic.parquet",
                         columns=["geneticSampleID", "geneticFilteredSampleVolume"])
    vol_ok = set(fg.loc[fg["geneticFilteredSampleVolume"].notna(), "geneticSampleID"].astype(str))

    out = {"blocks": {}, "qpcr_meta": {"n_16s_rows_nonprovisional": int(len(q)),
                                       "n_labs": int(q["laboratoryName"].nunique()),
                                       "n_batches": int(q["batchID"].nunique())}}
    for rel, path in [("002", "microbe_wq/samples.parquet"), ("001", "microbe_wq_v1/samples.parquet")]:
        s = pd.read_parquet(D + path, columns=["sample_id", "site_id", "domain_id", "collect_date", "chem_parent",
                                               "chem_match", "dna_sample_ids"])
        s["block"] = block(rel, s["collect_date"])
        dna = s["dna_sample_ids"].fillna("").str.split("|")
        s["qpcr"] = [any(x in q_ok for x in ids) for ids in dna]
        # geneticSampleID = dnaSampleID without the trailing "-DNA<n>" extraction suffix
        s["vol"] = [any(x.rsplit("-", 1)[0] in vol_ok for x in ids) for ids in dna]
        j = s.join(c, on="chem_parent", how="left")
        for blk, g in j.groupby("block"):
            rows = {}
            for v in VARS:
                has = g[v].notna()
                cen = g.get(f"{v}__censored")
                cen = cen.fillna(False).astype(bool) if cen is not None else pd.Series(False, index=g.index)
                rows[v] = {"n": int(has.sum()), "n_cens": int(cen.sum()), "n_cens_noval": int((cen & ~has).sum()),
                           "n_cens_val": int((cen & has).sum()), "n_usable": int((has | cen).sum()),
                           "sites": int(g.loc[has, "site_id"].nunique()),
                           "domains": int(g.loc[has, "domain_id"].nunique()),
                           "n_time_match": int((has & (g["chem_match"] == "time")).sum())}
            non_nh4 = [v for v in GEO8 if v != "nh4_n_mgl"]
            out["blocks"][f"{rel}/{blk}"] = {
                "n_events": int(len(g)), "n_with_chem_parent": int(g["chem_parent"].notna().sum()),
                "sites": int(g["site_id"].nunique()), "domains": int(g["domain_id"].nunique()),
                "n_all7_non_nh4": int(g[non_nh4].notna().all(axis=1).sum()),
                "n_qpcr_16s": int(g["qpcr"].sum()), "n_qpcr_16s_and_volume": int((g["qpcr"] & g["vol"]).sum()),
                "n_filtered_volume": int(g["vol"].sum()), "vars": rows}

    # detection-limit metadata of the lab records (attributes of the method, not sample values)
    cl = pd.read_parquet(D + "neon_chem/chem_long.parquet", columns=["var", "detection_limit", "provisional"])
    cl = cl[~cl["provisional"].fillna(False).astype(bool) & cl["var"].isin(FLOORED)]
    dl = {}
    for v, g in cl.groupby("var"):
        x = pd.to_numeric(g["detection_limit"], errors="coerce").dropna()
        pos = x[x > 0]
        dl[v] = {"n_records": int(len(g)), "n_with_dl": int(len(x)), "n_dl_zero": int((x == 0).sum()),
                 "min_nonzero": float(pos.min()) if len(pos) else None,
                 "median": float(x.median()) if len(x) else None, "max": float(x.max()) if len(x) else None}
    out["detection_limit_meta"] = dl

    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    for k, v in out["blocks"].items():
        print("\n==", k, {x: y for x, y in v.items() if x != "vars"})
        print(pd.DataFrame(v["vars"]).T.loc[GEO8 + ["field_temp_c", "ortho_p_mgl", "field_do_mgl",
                                                    "field_spcond_uscm"]].to_string())
    print(json.dumps(out["detection_limit_meta"], indent=1))
    print(out["qpcr_meta"])


if __name__ == "__main__":
    main()
