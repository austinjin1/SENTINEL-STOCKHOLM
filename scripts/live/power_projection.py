#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""A-priori onset-count projection for the EW-v2 live trial (amendment §8), from TRAIN/VAL data only.

Population: the live-config sites with train or val role (test-role sites are never read here). Blocks: train
(WY2008–21) and val (WY2022–23), Approved values, frozen ``hazards.daily_onsets``. Output: onsets per observed
site-year by block, the share of onsets by calendar month, and the projected number of onsets in the trial
windows (H1/H4: onset dates 2026-12-02 .. 2027-10-03 window; H2 primary: 2026-12-02 .. 2027-04-03) for the full
configured site list, assuming the train/val-role rate applies to every role. Writes
``$SENTINEL_RUNS/live/ew_v2/power_projection.json`` and registry rows (experiment ``ew_v2_live/power``).

    python -B scripts/live/power_projection.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import live_common as LC
import live_data as LD
from sentinel.core.registry import log_result
from sentinel.eval.hazards import HAZARDS, daily_onsets
from sentinel.models.sonde import common

VAL_END = "2023-09-30"


def main() -> int:
    cfg, cfg_sha = LC.load_config()
    LD.SNAPSHOT_DIR["dir"] = __import__("pathlib").Path(cfg["data_pins"]["sonde_daily_dir"])
    LD.SNAPSHOT_DIR["sha"] = cfg["data_pins"]["sonde_daily_site_sha256"]
    out = {"config_sha256": cfg_sha, "population": "live-config sites with train/val role; train+val blocks; Approved",
           "hazards": {}}
    for hz in cfg["models"]:
        tv = [r["site_no"] for r in cfg["sites"][hz] if r["role"] in ("train", "val")]
        cube, sc_parts = LD.build_cube(tv, cfg["models"][hz]["cube_variables"], VAL_END, window_start="2021-10-01",
                                       window_frames=None, window_from_snapshot=True)
        var = HAZARDS[hz].variable
        if hz.startswith("H2_"):
            y_cl, p_cl = LD.chloride_cube(cube, sc_parts, common.chloride_regressions())
            x_all, prov = y_cl[:, :, 1].astype(float), p_cl[:, :, 1]
        else:
            j = cube.var_index(var)
            x_all, prov = cube.y[:, :, j].astype(float), cube.prov[:, :, j]
        x_all = np.where(prov, np.nan, x_all)          # Approved only
        dates = cube.dates
        onset_dates, site_years = [], {"train": 0.0, "val": 0.0}
        n_on = {"train": 0, "val": 0}
        for s in range(len(cube.sites)):
            x = x_all[s]
            if not np.isfinite(x).any():
                continue
            _, on = daily_onsets(x, HAZARDS[hz])
            obs = np.isfinite(x)
            for b, (a0, a1) in {"train": ("2007-10-01", "2021-09-30"), "val": ("2021-10-01", VAL_END)}.items():
                m = (dates >= a0) & (dates <= a1)
                site_years[b] += obs[m].sum() / 365.25
                n_on[b] += int(on[m].sum())
            onset_dates += list(dates[on])
        months = pd.Series(pd.DatetimeIndex(onset_dates).month).value_counts().sort_index()
        share = (months / months.sum()).reindex(range(1, 13), fill_value=0.0)
        rate = {b: n_on[b] / site_years[b] if site_years[b] else float("nan") for b in n_on}
        full = share[[12, 1, 2, 3, 4, 5, 6, 7, 8, 9]].sum()        # Dec..Sep onsets (approx. trial window)
        winter = share[[12, 1, 2, 3]].sum()
        n_cfg = len(cfg["sites"][hz])
        # fraction of the configured train/val sites that actually reported the variable in the val block
        act = float(np.mean([np.isfinite(x_all[s, dates >= "2021-10-01"]).any() for s in range(len(cube.sites))]))
        proj = {}
        for b in ("train", "val"):
            per_site = rate[b]
            proj[b] = {"full_window": per_site * n_cfg * full, "h2_winter_window": per_site * n_cfg * winter}
        rec = {"n_sites_trainval": len(tv), "n_sites_config_all_roles": n_cfg, "onsets": n_on,
               "observed_site_years": {k: round(v, 1) for k, v in site_years.items()},
               "rate_per_observed_site_year": rate, "month_share": share.round(4).to_dict(),
               "share_dec_sep": float(full), "share_dec_mar": float(winter),
               "projected_onsets_assuming_full_year_coverage": proj,
               "note": "projection = rate per observed site-year x configured sites (all roles) x month share; "
                       "assumes every configured site reports all year (upper bound); test-role values not used"}
        out["hazards"][hz] = rec
        print(hz, json.dumps(rec, default=float)[:900])
        for b in ("train", "val"):
            log_result(workstream="M1-sonde", experiment="ew_v2_live/power", model=f"labels:{hz}", metric=f"onsets_per_site_year_{b}",
                       value=float(rate[b]), split=f"{b}/sonde_split_{cfg['split_sha256'][:12]}", n=n_on[b],
                       notes="live-trial power projection; train/val-role live sites only", target=hz)
            log_result(workstream="M1-sonde", experiment="ew_v2_live/power", model=f"labels:{hz}",
                       metric=f"projected_trial_onsets_from_{b}_rate", value=float(proj[b]["full_window"]),
                       split=f"{b}/sonde_split_{cfg['split_sha256'][:12]}", n=n_cfg, target=hz,
                       notes="upper bound: full coverage of all configured sites, Dec-Sep onsets")
    p = LC.LIVE_ROOT / "power_projection.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1, default=float))
    print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
