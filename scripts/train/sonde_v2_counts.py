#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Counts for the consolidated deviations file of architecture round v2
(docs/preregistration/sonde_arch_v2_dev_lead_rulings.md F, I; verifier item 7). COUNTS AND WEATHER ONLY: no
water-quality label value, no label statistic and no forecast of any model is computed, printed or written. What
is written: numbers of sites, site-variable pairs and rows by reference-window category; the bias-map source
counts and error-model settings of the product build; GEFS availability; and a weather-only diagnostic
(bias-corrected GEFS minus observed gridMET, train block only).

    python -B scripts/train/sonde_v2_counts.py step1|cv     -> <runs>/sonde/v2_smoke/counts_<use>.json
    python -B scripts/train/sonde_v2_counts.py log          -> registry row of the weather-only diagnostic from
                                                               counts_cv.json (refuses without the lead stamp)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sentinel.core import paths  # noqa: E402
from sentinel.models.sonde_v2 import declared as D  # noqa: E402
from sentinel.models.sonde_v2 import nwp as NWP  # noqa: E402
from sentinel.models.sonde_v2 import parity  # noqa: E402
from sentinel.models.sonde_v2 import prep as PR  # noqa: E402
from sentinel.models.sonde_v2 import products as P  # noqa: E402

OUT = paths.RUNS_ROOT / "sonde" / "v2_smoke"
OP_DIAG = "operational_train_2020-09-23_2021-09-23"

use = sys.argv[1]
if use == "log":
    from sentinel.core import registry

    parity.require_stamp("registry row of the weather-only bias diagnostic")
    f = OUT / "counts_cv.json"
    res = json.loads(f.read_text())
    d = res["bias_corrected_minus_observed"][OP_DIAG]
    row = registry.log_result(
        workstream="M1-sonde", experiment="sonde_arch_v2/p3_bias_diagnostic", model="gefs_c00_bias_corrected",
        metric="max_abs_mean_residual_tmean_c_leads_1_7", value=float(np.max(np.abs(d["tmean_c"]["mean"]))),
        split="train", n=int(d["tmean_c"]["n"][0]),
        notes="weather-only, train block only, report-only: bias-corrected operational GEFS c00 minus gridMET, "
              "issue days 2020-09-23..2021-09-23, map fitted on reforecast pairs only; no water-quality value read",
        extra={"per_field_per_lead": d, "bias_fit": res["products"]["bias_fit"], "sites": res["n_sites"],
               "counts_file": str(f)})
    print(f"registry row {row['id']}")
    raise SystemExit(0)
if use not in ("step1", "cv"):
    raise SystemExit(__doc__)
from sentinel.models.sonde import runner  # noqa: E402

runner.limit_cpu_threads(D.GBM_SETTINGS["num_threads"])
OUT.mkdir(parents=True, exist_ok=True)
out = OUT / f"counts_{use}.json"
t0 = time.time()
blocks, roles = (("train",), ("train",)) if use == "step1" else (("train", "val"), ("train", "val"))
data = PR.prepare_v2(blocks=blocks, roles=roles, log=print)
dates, cube = data.cube.dates, data.cube
T = len(dates)
te = int(np.searchsorted(dates, pd.Timestamp("2021-09-30"), side="right"))     # first index after the train block
res = {"use": use, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "code_files": parity.code_files(),
       "n_sites": len(data.sites), "roles": {str(k): int(v) for k, v in zip(*np.unique(data.roles, return_counts=True))},
       "dates": [str(dates[0].date()), str(dates[-1].date())], "ref": {"days": PR.REF_DAYS, "min_obs": PR.REF_MIN_OBS,
                                                                        "embargo": PR.REF_EMBARGO}, "variables": {}}
# INWIN[s, t]: issue day t lies inside the reference window (first .. first+728) of at least one modelled input
# variable of site s that has a reference climatology (its climatology then contains days after t)
INWIN = np.zeros((len(data.sites), T), bool)
tt = np.arange(T)[None, :]
for j, v in enumerate(cube.variables):
    if v in PR.NOT_MODELLED:
        continue
    fin = np.isfinite(cube.y[:, :, j])
    has = fin.any(1) & (data.ref_issue_start[:, j] < PR.NEVER)
    f = fin.argmax(1)[:, None]
    INWIN |= has[:, None] & (tt >= f) & (tt < f + PR.REF_DAYS - 1)
for v in list(D.SCORED) + [x for x in D.AUX_REPORTED if x in cube.variables]:
    if v not in cube.variables:
        continue
    j = cube.var_index(v)
    has = np.isfinite(cube.y[:, :, j]).any(1)
    st = data.ref_issue_start[:, j]
    cat = {}
    for role in np.unique(data.roles):
        r = data.roles == role
        cat[str(role)] = {
            "site_variable_pairs_with_data": int((has & r).sum()),
            "no_reference_climatology_(<365_observed_days_in_window)": int((has & r & (st >= PR.NEVER)).sum()),
            "first_usable_issue_in_train_block": int((has & r & (st < te)).sum()),
            "first_usable_issue_after_train_block_inside_loaded_dates": int((has & r & (st >= te) & (st < T)).sum()),
            "first_usable_issue_after_loaded_dates": int((has & r & (st >= T) & (st < PR.NEVER)).sum())}
    rows = {}
    for b in blocks:
        rr = data.rows(v, b)
        dr = data.n_dropped_ref_window[(v, b)]
        rows[b] = {"common_rows_kept": int(len(rr)), "common_rows_dropped_by_issue_rule": dr["rows"],
                   "scored_kept": {int(h): int(rr.score_mask[h].sum()) for h in data.horizons},
                   "scored_dropped_by_issue_rule": dr["scored"],
                   "sites_with_kept_rows": int(np.unique(rr.si).size)}
    # reference-window position (index level): first finite index f, window = indices f .. f+729
    first = np.where(has, np.isfinite(cube.y[:, :, j]).argmax(1), -1)
    we = first + PR.REF_DAYS                                   # first index after the window
    ok = has & (st < PR.NEVER)
    pos = {}
    for role in np.unique(data.roles):
        r = data.roles == role
        pos[str(role)] = {
            "first_finite_on_cube_start_2007-10-01": int((has & r & (first == 0)).sum()),
            "window_ends_in_train_block_with_clim": int((ok & r & (we <= te)).sum()),
            "window_ends_in_val_block_with_clim": int((ok & r & (we > te) & (we <= T)).sum()),
            "window_cut_by_end_of_loaded_dates_with_clim": int((ok & r & (we > T)).sum()),
            "window_cut_by_end_of_loaded_dates_no_clim": int((has & ~ok & r & (we > T)).sum()),
            "window_complete_no_clim": int((has & ~ok & r & (we <= T)).sum())}
    for b in blocks:                                           # rows issued inside another input variable's window
        rr = data.rows(v, b)
        rows[b]["rows_issued_inside_reference_window_of_another_input_variable"] = int(INWIN[rr.si, rr.ti].sum())
    res["variables"][v] = {"pairs": cat, "window_position": pos, "rows": rows}
st_ = data.static
res["statics"] = {}
for role in np.unique(data.roles):
    r = data.roles == role
    res["statics"][str(role)] = {
        "sites": int(r.sum()),
        "sites_with_nhd_slope": int((np.isfinite(st_["slope"].to_numpy(float)) & r).sum()) if "slope" in st_ else None,
        "sites_without_elevation": int((~np.isfinite(st_["elev_m"].to_numpy(float)) & r).sum()),
        "sites_without_latitude": int((~np.isfinite(st_["lat"].to_numpy(float)) & r).sum()),
        "site_types": {str(k): int(v) for k, v in zip(*np.unique(st_["site_type"].astype(str).to_numpy()[r], return_counts=True))}
        if "site_type" in st_ else None}
res["static_columns"] = [c for c in st_.columns]
out.write_text(json.dumps(res, indent=1))
print(f"reference-window counts written ({time.time() - t0:.0f} s)", flush=True)
prod = P.build(data, use=use, log=print)
res["products"] = prod.info
out.write_text(json.dumps(res, indent=1, default=str))
g, obs_f = prod.gefs, prod.obs_f
ti_, pi_, si_ = NWP.F_IDX["tmean_c"], NWP.F_IDX["precip_mm"], NWP.F_IDX["srad_wm2"]
ft = np.isfinite(g[..., ti_])                                   # (S, T, 7)
blk = {"train": np.arange(T) < te}
if use == "cv":
    blk["val"] = np.arange(T) >= te
avail = {}
for b, m in blk.items():
    c = ft[:, m]
    n = int(c.sum())
    avail[b] = {"site_issue_lead_cells": int(c.size), "cells_with_tmean": n,
                "share_with_tmean": n / max(c.size, 1),
                "cells_with_tmean_missing_precip": int((c & ~np.isfinite(g[:, m][..., pi_])).sum()),
                "cells_with_tmean_missing_shortwave": int((c & ~np.isfinite(g[:, m][..., si_])).sum()),
                "issue_days": int(m.sum()), "issue_days_without_any_gefs": int((~ft[:, m, 0].any(0)).sum())}
none = ~ft[:, :, 0].any(0)
runs, a = [], None
for i in range(T):
    if none[i] and a is None:
        a = i
    if a is not None and (not none[i] or i == T - 1):
        e = i if none[i] else i - 1
        runs.append([str(dates[a].date()), str(dates[e].date()), int(e - a + 1)])
        a = None
res["gefs_availability"] = {"by_block": avail, "issue_day_runs_without_any_gefs": runs,
                            "sites_without_any_gefs": int((~ft[:, :, 0].any(1)).sum())}


def resid(a_, b_):
    sel = np.asarray((dates >= pd.Timestamp(a_)) & (dates <= pd.Timestamp(b_)))
    d = g[:, sel] - obs_f[:, sel]                                # (S, t, 7, F)
    return {f: {"mean": [float(x) for x in np.nanmean(d[..., k], axis=(0, 1))],
                "mean_abs": [float(x) for x in np.nanmean(np.abs(d[..., k]), axis=(0, 1))],
                "n": [int(x) for x in np.isfinite(d[..., k]).sum(axis=(0, 1))]} for f, k in NWP.F_IDX.items()}


# weather-only train-block diagnostic: bias-corrected GEFS minus observed gridMET, per lead (1..7) and field
diag = {"reforecast_fit_2007-10-01_2019-09-23": resid("2007-10-01", "2019-09-23")}
if use == "cv":
    diag[OP_DIAG] = resid("2020-09-23", "2021-09-23")
res["bias_corrected_minus_observed"] = diag
out.write_text(json.dumps(res, indent=1, default=str))
print(json.dumps({k: prod.info[k] for k in ("coverage", "bias_fit", "gefs_issue_days")}, indent=1, default=str))
print(json.dumps(res["gefs_availability"], indent=1))
print(f"done ({time.time() - t0:.0f} s) -> {out}")
