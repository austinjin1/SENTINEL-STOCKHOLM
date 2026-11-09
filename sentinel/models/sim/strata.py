# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Sim hard track (protocol §4.6, added 2026-11-08 under RESEARCH_PLAN §15.5 amendment items 5–7).

A-priori difficulty strata for the hindcast benchmark. Each is defined from forcings or discharge only, with
site-specific thresholds fitted on the site's TRAIN-block days, and is fixed before any model result on it
is seen:

* ``heatwave``  — target day inside a run of ≥ 3 consecutive days whose gridMET daily-mean air-temperature
                  anomaly (vs the site's train-block DOY climatology, ±15 d) is ≥ the site's train-block P90
                  of that anomaly (the §7 step-4 definition);
* ``low_flow``  — observed (unfilled) discharge at the target day ≤ the site's train-block P10;
* ``high_flow`` — observed discharge at the target day ≥ the site's train-block P95 (storm/high-flow);
* hot water years — the train-cool/test-hot experiment (§5.4), scored separately;
* ``region_holdout`` — leave-HUC2-region-out training (:data:`REGION_FOLDS`): every train/cal-role site in
                  the fold's HUC2 regions is removed before training, and the fold's val-role sites are scored.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from ...core import paths
from . import data as D
from .dsst import fit_climatology

STRATA = ("heatwave", "low_flow", "high_flow")
HW_Q, HW_RUN, LOW_Q, HIGH_Q = 0.90, 3, 0.10, 0.95
# Folds of HUC2 regions containing val-role sites (declared 2026-11-08 from the role × HUC2 table only,
# before any region-held-out model existed): neighbouring regions grouped, ~20–60 val sites per fold.
REGION_FOLDS = {"F1_ohio": ["05"], "F2_northeast_greatlakes": ["01", "04"],
                "F3_pacific": ["17", "18"], "F4_interior": ["07", "08", "11", "12", "13"]}
# power rule per stratum × variable × horizon cell (val cell during development)
MIN_SITES, MIN_GROUPS, MIN_ROWS = 10, 5, 500


def site_huc2(d: D.SimData) -> np.ndarray:
    reg = pd.read_parquet(paths.PROCESSED / "sites" / "sites.parquet", columns=["site_id", "huc2"])
    h = reg.assign(s=reg["site_id"].astype(str)).set_index("s")["huc2"].astype(str)
    return np.array([str(h.get(s, "")).zfill(2) for s in d.sites])


def day_flags(d: D.SimData) -> dict[str, np.ndarray]:
    """{stratum: (S, T) bool} for every site × day (target-day membership)."""
    tr = D.block_mask(d.dates, "train")
    ta = 0.5 * (d.forcing[:, :, 0] + d.forcing[:, :, 1]).astype("float64")
    cm, _ = fit_climatology(ta, d.dates, tr)
    anom = ta - cm[:, d.dates.dayofyear.to_numpy() - 1]
    thr = np.nanquantile(np.where(tr[None, :], anom, np.nan), HW_Q, axis=1)
    hot = np.nan_to_num(anom >= thr[:, None]).astype(bool)
    hw = np.zeros_like(hot)
    S, T = hot.shape
    for i in range(S):
        run = 0
        for t in range(T):
            run = run + 1 if hot[i, t] else 0
            if run >= HW_RUN:
                hw[i, t - run + 1: t + 1] = True
    q = np.where(d.q_missing, np.nan, d.q_m3s.astype("float64"))
    qtr = np.where(tr[None, :], q, np.nan)
    lo = np.nanquantile(qtr, LOW_Q, axis=1)
    hi = np.nanquantile(qtr, HIGH_Q, axis=1)
    with np.errstate(invalid="ignore"):
        low = np.nan_to_num(q <= lo[:, None]).astype(bool)
        high = np.nan_to_num(q >= hi[:, None]).astype(bool)
    return {"heatwave": hw, "low_flow": low, "high_flow": high}


def row_mask(rows: pd.DataFrame, h: int, flags: np.ndarray, d: D.SimData) -> np.ndarray:
    """Hindcast rows (site, issue_date) whose TARGET day d + h is in the stratum."""
    pos = {s: i for i, s in enumerate(d.sites)}
    tpos = pd.Series(np.arange(len(d.dates)), index=d.dates)
    si = rows["site"].map(pos).to_numpy()
    ti = tpos.reindex(pd.DatetimeIndex(rows["issue_date"])).to_numpy().astype(int) + h
    ok = ti < flags.shape[1]
    out = np.zeros(len(rows), bool)
    out[ok] = flags[si[ok], ti[ok]]
    return out


def subset(hc: dict, masks: dict) -> dict:
    out = {}
    for k, r in hc.items():
        m = masks[k]
        if not m.any():
            continue
        out[k] = {"rows": r["rows"][m].reset_index(drop=True), "y": r["y"][m],
                  "q": {mm: q[m] for mm, q in r["q"].items()}, "phi_pool": r.get("phi_pool")}
    return out


def concat(parts: list[dict]) -> dict:
    out = {}
    for k in sorted({k for p in parts for k in p}):
        ps = [p[k] for p in parts if k in p]
        out[k] = {"rows": pd.concat([p["rows"] for p in ps], ignore_index=True),
                  "y": np.concatenate([p["y"] for p in ps]),
                  "q": {m: np.concatenate([p["q"][m] for p in ps]) for m in ps[0]["q"]}}
    return out


def power_table(hc: dict, d: D.SimData, flags: dict, variables, horizons) -> pd.DataFrame:
    """Rows / sites / basin groups per stratum × variable × horizon (no model output used)."""
    recs = []
    for (v, h), r in hc.items():
        if v not in variables or h not in horizons:
            continue
        for s, f in flags.items():
            m = row_mask(r["rows"], h, f, d)
            rr = r["rows"][m]
            n_s, n_g = rr["site"].nunique(), rr["basin_group"].nunique()
            recs.append({"stratum": s, "var": v, "h": h, "n_rows": int(m.sum()), "n_sites": int(n_s),
                         "n_groups": int(n_g), "powered": bool(n_s >= MIN_SITES and n_g >= MIN_GROUPS
                                                                 and m.sum() >= MIN_ROWS)})
    return pd.DataFrame(recs)


def sha256_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()
