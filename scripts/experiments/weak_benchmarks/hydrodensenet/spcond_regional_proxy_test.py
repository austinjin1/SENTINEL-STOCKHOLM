#!/usr/bin/env python
"""Is imagery-only SpCond R2=0.51 a REAL optical retrieval or a REGIONAL PROXY?

SpCond (dissolved-ion conductance) is NOT an optical property, yet the cycle-14
optical GBM gets a station-grouped-holdout R2 ~0.51 on it. Two explanations:

  (H1 real optical) reflectance carries genuine within-site conductance signal.
  (H2 regional proxy) reflectance encodes WHERE the tile is (land-cover / basin /
      water color of that region). Each USGS station sits in a narrow conductance
      band, and conductance is spatially clustered (coastal/arid = high, montane =
      low). A model that reads "this looks like region X" and outputs region X's
      typical conductance scores high R2 on a station-grouped split WITHOUT any true
      optical conductance retrieval. The station-grouped split does NOT protect
      against this because nearby unseen stations share the regional look.

DECISIVE DECOMPOSITION
----------------------
Split each OOF prediction into BETWEEN-station and WITHIN-station components:
  * BETWEEN: can optics rank STATIONS by their mean SpCond? (regional proxy skill)
  * WITHIN : after removing each station's own mean (from BOTH y and pred), does
             optics explain the residual day-to-day conductance variation at a site?
If BETWEEN R2 is high and WITHIN R2 ~ 0, the "SpCond from optics" is a regional
proxy, NOT a genuine optical retrieval — exactly what you'd expect for a
non-optical analyte. We also run the same decomposition for Temp (should have
real within-site optical/seasonal signal -> positive within R2) and DO as controls.

This uses the SAME OOF predictions machinery as honest_hydrodensenet.py.
REAL data only. No paper edits, no commits.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import r2_score

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "hh", str(Path(__file__).with_name("honest_hydrodensenet.py")))
hh = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(hh)

ROOT = Path("/home/bcheng/SENTINEL")
OUT = ROOT / "results/benchmarks/weak_improve/hydrodensenet/spcond_regional_proxy_test.json"


def within_between_r2(y, p, sites, log=False):
    if log:
        pos = y > 0
        y, p, sites = np.log(y[pos]), p[pos], sites[pos]
    uniq = np.unique(sites)
    # station means from TRUTH (these define the between component)
    ybar = {s: y[sites == s].mean() for s in uniq}
    pbar = {s: p[sites == s].mean() for s in uniq}
    y_between = np.array([ybar[s] for s in sites])
    p_between = np.array([pbar[s] for s in sites])
    y_within = y - y_between
    p_within = p - p_between
    # station-level between R2 (one point per station, weighted by count)
    st_y = np.array([ybar[s] for s in uniq])
    st_p = np.array([pbar[s] for s in uniq])
    r2_between = r2_score(st_y, st_p) if len(uniq) > 2 else float("nan")
    # within R2: does pred residual track truth residual? only sites with >=3 obs
    mask_multi = np.array([np.sum(sites == s) >= 3 for s in sites])
    if mask_multi.sum() > 5 and np.var(y_within[mask_multi]) > 0:
        r2_within = r2_score(y_within[mask_multi], p_within[mask_multi])
    else:
        r2_within = float("nan")
    return {
        "overall_r2": round(float(r2_score(y, p)), 4),
        "between_station_r2": round(float(r2_between), 4),
        "within_station_r2": round(float(r2_within), 4),
        "n_stations": int(len(uniq)),
        "frac_variance_between": round(float(np.var(y_between) / (np.var(y) + 1e-12)), 4),
    }


def main():
    d = np.load(hh.NPZ, allow_pickle=True)
    images = d["images"]; targets = d["targets"].astype("float64")
    meta = json.loads(d["metadata"].item())
    sites_all = np.array([m["site_id"] for m in meta])
    Xopt, _ = hh.build_optical_matrix(images)
    finite = np.isfinite(Xopt).all(axis=1)

    out = {"_PROVENANCE": {
        "purpose": "Decompose imagery-only station-grouped R2 into BETWEEN-station "
                   "(regional proxy) vs WITHIN-station (true optical retrieval) for "
                   "SpCond, with Temp/DO/Turb as controls. Tests whether SpCond R2~0.5 "
                   "is a spurious regional proxy for a non-optical analyte.",
        "data": "drone_wq_pairs.npz (REAL Sentinel-2 x USGS). Imagery-only, 4 bands.",
        "method": "GroupKFold OOF preds; remove per-station mean from truth AND pred; "
                  "between = station-mean truth vs station-mean pred; within = residuals "
                  "(stations with >=3 obs).",
    }, "targets": {}}

    for tgt, log in [("SpCond", True), ("Temp", False), ("DO", False), ("Turb", True)]:
        ti = hh.TGT.index(tgt)
        y = targets[:, ti]
        m = finite & np.isfinite(y)
        if log:
            m = m & (y > 0)
        g = sites_all[m]; ym = y[m]; Xm = Xopt[m]
        ns = min(5, np.unique(g).size)
        p = hh.oof_reg(Xm, np.log(ym) if log else ym, g, ns)
        if log:
            # predictions are in log space; decompose in log space directly
            dec = within_between_r2(ym, p, g, log=True)
        else:
            dec = within_between_r2(ym, p, g, log=False)
        dec["scale"] = "log" if log else "raw"
        out["targets"][tgt] = dec
        print(tgt, dec)

    # verdicts
    sp = out["targets"]["SpCond"]; tp = out["targets"]["Temp"]
    out["verdict"] = {
        "SpCond": (
            f"SpCond overall R2={sp['overall_r2']} decomposes into between-station "
            f"R2={sp['between_station_r2']} (regional proxy) and within-station "
            f"R2={sp['within_station_r2']} (true optical). "
            f"{sp['frac_variance_between']*100:.0f}% of SpCond variance is between-station. "
            + ("Within-station R2 <= 0 => the SpCond 'retrieval' is a SPURIOUS REGIONAL "
               "PROXY (optics reads location, not conductance), consistent with SpCond "
               "being non-optical."
               if sp['within_station_r2'] <= 0.05 else
               "Some within-station optical signal exists.")),
        "Temp_control": (
            f"Temp within-station R2={tp['within_station_r2']} (should be positive if "
            "genuine optical/seasonal signal) — control that the decomposition detects "
            "real within-site skill."),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print("WROTE", OUT)
    print(json.dumps(out["verdict"], indent=2))


if __name__ == "__main__":
    main()
