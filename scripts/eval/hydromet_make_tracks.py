#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Freeze the M6 track-R fold file (protocol §10) and write its power check, BEFORE any model is fitted or
scored on track R. Reads only site metadata, the FINAL split, the sonde_daily index (record spans; no values)
and — for the power check — the registered original-track val CIs (`M6/val_v2` score directory).

    python -B scripts/eval/hydromet_make_tracks.py --score-dir $SENTINEL_RUNS/hydromet/score_val_v2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from sentinel.core import paths
from sentinel.models.hydromet import data as D
from sentinel.models.hydromet import tracks as TR

OUT = paths.REPO_ROOT / "docs" / "preregistration" / "hydromet_track_r_folds.json"
REPORT = paths.REPO_ROOT / "docs" / "preregistration" / "hydromet_track_r_power.json"
PRIMARY = ["temp_c_mean", "temp_c_7dadm", "do_mgl_mean", "do_mgl_min"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--score-dir", required=True)
    a = ap.parse_args()
    sp, info = D.load_split()
    sites, elig = D.eligible_sites(sp, ("train", "val", "test"))
    reg = pd.read_parquet(D.SITES_FILE)
    reg = reg[reg["network"] == "USGS"].drop_duplicates("site_id").set_index("site_id")
    huc8 = reg.loc[sites, "huc8"].astype(str).to_dict()
    fold_of = TR.assign_folds(sites, huc8, sp.basin_group)
    roles = {s: sp.site_role[s] for s in sites}
    body = TR.folds_payload(fold_of, sp.sha256(), roles)
    if OUT.exists():
        old = json.loads(OUT.read_text())
        if old["sha256"] != body["sha256"]:
            raise SystemExit(f"{OUT} exists with a different sha256 — the track-R folds are frozen")
    else:
        OUT.write_text(json.dumps(body, indent=0))
    # ---- power check (record spans only; ~ sites with >= 1 year of record overlapping the val block)
    idx = pd.read_parquet(D.SONDE_INDEX)
    idx = idx[(pd.to_datetime(idx["first"]) <= "2023-06-30") & (pd.to_datetime(idx["last"]) >= "2022-01-01")]
    cube_var = {"temp_c_mean": "temp_c_mean", "temp_c_7dadm": "temp_c_max", "do_mgl_mean": "do_mgl_mean",
                "do_mgl_min": "do_mgl_min"}
    prim = pd.read_csv(Path(a.score_dir) / "primary_val_val.csv")
    m_holm = 12  # track-R Holm family: 3 models (ealstm, ealstm_full, ealstm_gbmblend) x 4 endpoints
    za, zb = norm.ppf(1 - 0.05 / (2 * m_holm)), norm.ppf(0.8)
    rep = {"fold_file": str(OUT.name), "fold_sha256": body["sha256"], "split_sha256": sp.sha256(),
           "eligibility": elig, "per_fold": {}, "power": {},
           "method": ("SE of the median-KGE difference under basin-group resampling scales ~ 1/sqrt(G); SE_orig from "
                      "the M6/val_v2 group-bootstrap CI width of ealstm vs the selected GBM ((hi-lo)/3.92); "
                      "MDE = (z_{1-alpha/(2m)} + z_0.8) * SE_orig * sqrt(G_orig / G_R), worst-case Holm alpha/m, m=12. "
                      "G counts are upper bounds from record spans (the >= 60 approved val-block days rule applies at scoring).")}
    df = pd.DataFrame({"site": sites, "fold": [fold_of[s] for s in sites], "role": [roles[s] for s in sites],
                       "bg": [sp.basin_group[s] for s in sites]})
    rep["per_fold"] = {f: {r: int(n) for r, n in g["role"].value_counts().items()} for f, g in df.groupby("fold")}
    cv = df[df["role"].isin(["train", "val"])]
    for v in PRIMARY:
        have = set(idx.loc[idx["var"] == cube_var[v], "site_no"])
        d = cv[cv["site"].isin(have)]
        r = prim[(prim["model"] == "ealstm") & (prim["var"] == v)].iloc[0]
        se = (r["gbm_ci_high"] - r["gbm_ci_low"]) / 3.92
        g0, g = int(r["gbm_n_groups"]), int(d["bg"].nunique())
        rep["power"][v] = {"cv_sites": int(len(d)), "cv_groups": g, "orig_val_groups": g0,
                           "orig_se": round(float(se), 4),
                           "mde_orig": round(float((za + zb) * se), 4),
                           "mde_track_r": round(float((za + zb) * se * np.sqrt(g0 / max(g, 1))), 4),
                           "groups_per_fold": d.groupby("fold")["bg"].nunique().to_dict()}
    REPORT.write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps({"sha256": body["sha256"], "per_fold": rep["per_fold"],
                      "power": {k: {kk: vv for kk, vv in v.items() if kk != "groups_per_fold"}
                                for k, v in rep["power"].items()}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
