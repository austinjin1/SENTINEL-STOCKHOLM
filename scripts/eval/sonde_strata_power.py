#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""M1-X harder track (RESEARCH_PLAN §15.5 amendment items 5-6): power check of the a-priori
difficulty strata on the VAL block, BEFORE any model is scored or selected on them.

Strata (issue-time information only; thresholds from each site's TRAINING block WY2008-21):
  X1 storm  : trailing 3-day observed precipitation (gridMET, lagged as in common.load_weather)
              >= the site's training-block P95 of the same quantity;
  X2 heat   : trailing 3-day mean tmax >= the site's training-block P95 (any season);
  X3 extreme: target-day observation beyond the site x DOY training climatology by more than
              1.645 clim SD (upper tail; DO lower tail) -- an OUTCOME stratum, scored only with the
              threshold-weighted CRPS (proper), never by conditioning ordinary CRPS on it.
Counts rows, sites and basin groups per variable x stratum x site role at h = 3 d.

    python -B scripts/eval/sonde_strata_power.py --config configs/sonde/common.yaml --set use_weather=true
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sentinel.data.sonde_splits import TIME_BLOCKS
from sentinel.models.sonde import runner
from sentinel.models.sonde.data import to_model_space


def roll(x: np.ndarray, k: int, fn) -> np.ndarray:
    out = np.full_like(x, np.nan)
    for t in range(k - 1, x.shape[1]):
        out[:, t] = fn(x[:, t - k + 1:t + 1], axis=1)
    return out


def main() -> int:
    ap = runner.base_parser(__doc__)
    a = ap.parse_args()
    cfg = runner.apply_overrides(runner.load_config(a.config, smoke=a.smoke), a.set)
    cfg["use_weather"] = True
    run_dir = runner.run_directory("strata_power", cfg, a.run_name)
    log = runner.make_logger(run_dir / "log.txt")
    data = runner.prepare(a, cfg, run_dir, training=True, log=log)
    dates = pd.DatetimeIndex(data.cube.dates)
    tr = np.asarray(dates <= pd.Timestamp(TIME_BLOCKS["train"][1]))
    w = data.weather
    with np.errstate(all="ignore"):
        p3 = roll(w[:, :, 0].astype(float), 3, np.sum)
        t3 = roll(w[:, :, 1].astype(float), 3, np.mean)
        thr_p = np.nanpercentile(np.where(tr[None], p3, np.nan), 95, axis=1)
        thr_t = np.nanpercentile(np.where(tr[None], t3, np.nan), 95, axis=1)
    storm = p3 >= thr_p[:, None]
    heat = t3 >= thr_t[:, None]
    basin = {str(k): str(v) for k, v in data.split.basin_group.items()}
    sites = np.asarray(data.sites)
    roles = np.asarray(data.roles)
    out, h = [], 3
    for var in runner.model_variables(cfg, data):
        r = data.rows(var, "val")
        si, ti = r.si, r.ti
        j = data.cube.var_index(var)
        tt = np.minimum(ti + h, len(dates) - 1)
        y = to_model_space(var, data.cube.y[si, tt, j].astype(float))
        doy = dates.dayofyear.to_numpy()[tt] - 1
        cm, cs = data.clim_mean[si, doy, j], data.clim_std[si, doy, j]
        z = (y - cm) / cs
        ext = (z <= -1.645) if var.startswith("do_") else (z >= 1.645)
        for name, m in (("all", np.ones(len(si), bool)), ("X1_storm", storm[si, ti]), ("X2_heat", heat[si, ti]),
                        ("X3_extreme_target", ext & np.isfinite(y))):
            for role in ("all", "val"):
                k = m & (np.ones(len(si), bool) if role == "all" else roles[si] == role)
                s_ = np.unique(sites[si[k]])
                out.append({"var": var, "h": h, "stratum": name, "site_role": role, "n_rows": int(k.sum()),
                            "n_sites": int(s_.size), "n_basin_groups": len({basin.get(str(s)) for s in s_})})
    res = pd.DataFrame(out)
    res.to_csv(run_dir / "strata_power_val.csv", index=False)
    log(f"[strata] VAL power check (no model scored):\n{res.to_string(index=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
