# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Hourly SENTINEL-Sonde tier from raw USGS instantaneous (≈15-min) values.

Per site: mask estimated/censored/bracket-coded values, aggregate each time series to hourly
mean/min/max/count (UTC hours), splice multiple series exactly like the daily tier (primary =
longest; secondary fills gaps only when consistent on >= 72 overlapping hours), and keep the
provisional flag. The 15-min values themselves stay in raw/usgs_iv for event detection with
sub-hourly duration rules (EW-v2 H1 requires >= 2 h below threshold).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .sonde_daily import CLIP_ZERO, PLAUSIBLE, _tolerance
from .sources.usgs_legacy import ALL_PARAMS

MASK_QUALIFIER_IV = r"(?:\be\b|:e|\[\d+\]|<|>)"


def clean_iv(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["parm_cd"].astype(str).isin(ALL_PARAMS)].copy()
    df["parm_cd"] = df["parm_cd"].astype(str)
    q = df["qualifier"].astype(str)
    df.loc[q.str.contains(MASK_QUALIFIER_IV, regex=True), "value"] = np.nan
    lo = df["parm_cd"].map(lambda p: PLAUSIBLE[p][0])
    hi = df["parm_cd"].map(lambda p: PLAUSIBLE[p][1])
    df.loc[(df["value"] < lo) | (df["value"] > hi), "value"] = np.nan
    clip = df["parm_cd"].isin(CLIP_ZERO) & (df["value"] < 0)
    df.loc[clip, "value"] = 0.0
    df["provisional"] = q.str.startswith("P")
    return df[df["value"].notna()]


def hourly_site(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Clean 15-min long frame for ONE site -> wide hourly frame (index = UTC hour)."""
    log, cols = [], {}
    df = df.copy()
    df["hour"] = df["time_utc"].dt.floor("h")
    for parm, g in df.groupby("parm_cd"):
        name = ALL_PARAMS[parm]
        per_ts = {}
        for ts, s in g.groupby("ts_id"):
            a = s.groupby("hour").agg(mean=("value", "mean"), min=("value", "min"),
                                      max=("value", "max"), n=("value", "size"),
                                      prov=("provisional", "max"))
            per_ts[ts] = a
        order = sorted(per_ts, key=lambda t: -len(per_ts[t]))
        prim = per_ts[order[0]].copy()
        for ts in order[1:]:
            sec = per_ts[ts]
            ov = prim.index.intersection(sec.index)
            ok = True
            if len(ov) >= 72:
                mad = float(np.median(np.abs(prim.loc[ov, "mean"] - sec.loc[ov, "mean"])))
                tol = _tolerance(parm, prim["mean"])
                ok = mad <= tol
                log.append({"var": name, "primary": order[0], "secondary": ts, "overlap_h": len(ov),
                            "median_abs_diff": mad, "tolerance": tol, "used": ok})
            else:
                log.append({"var": name, "primary": order[0], "secondary": ts, "overlap_h": len(ov),
                            "median_abs_diff": None, "tolerance": None, "used": True})
            if ok:
                prim = pd.concat([prim, sec.loc[sec.index.difference(prim.index)]])
        prim = prim.sort_index()
        for k in ("mean", "min", "max"):
            cols[f"{name}_{k}"] = prim[k].astype("float32")
        cols[f"{name}_n"] = prim["n"].astype("int16")
        cols[f"{name}_prov"] = prim["prov"].astype(bool)
    if not cols:
        return pd.DataFrame(), log
    wide = pd.DataFrame(cols).sort_index()
    for c in wide.columns:
        if c.endswith("_prov"):
            wide[c] = wide[c].astype("boolean").fillna(False).astype(bool)
        elif c.endswith("_n"):
            wide[c] = wide[c].fillna(0).astype("int16")
    wide.index.name = "time_utc"
    return wide, log
