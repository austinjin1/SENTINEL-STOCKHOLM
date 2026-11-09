# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Build the SENTINEL-Sonde daily tier from raw USGS daily values.

Rules (each one is logged into the dataset summary):
  * Variables are identified by USGS parameter code; units never mixed (e.g. chl RFU 32315
    and µg/L 32316 are separate variables).
  * Values whose qualifier marks them as estimated (``e``), computed from an incomplete day
    (``[4]``), or censored (``<``/``>``) are masked. Non-numeric codes (Ice, Eqp, Dis, Bkw,
    ***, ...) were already NaN at ingest.
  * Physically implausible values are masked with variable-specific ranges (``PLAUSIBLE``).
  * Several time series (``ts_id``) can exist per site/parameter/statistic. The longest is
    primary; others only fill dates where the primary is missing, and only if, where they
    overlap the primary by >= 30 days, their median absolute difference is within the
    variable's tolerance. Each value keeps its source ``ts_id`` (column ``<var>_<stat>_ts``)
    so splices are traceable.
  * ``<var>_<stat>_prov`` marks provisional (not yet approved) values; EW-v2 scores on
    approved values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .sources.usgs_legacy import ALL_PARAMS, STATS

# Plausible ranges (inclusive) for masking. Discharge/gage may be negative at tidal sites.
PLAUSIBLE = {
    "00010": (-1.0, 40.0), "00300": (0.0, 25.0), "00301": (0.0, 300.0), "00095": (0.0, 100000.0),
    "00400": (3.0, 11.5), "63680": (-1.0, 10000.0), "63675": (-1.0, 10000.0), "99133": (0.0, 100.0),
    "99137": (-0.1, 100.0), "32295": (0.0, 1000.0), "32315": (-1.0, 1000.0), "32316": (-1.0, 2000.0),
    "32319": (-1.0, 2000.0), "32321": (-1.0, 1000.0), "00480": (0.0, 45.0),
    "00060": (-1e7, 1e7), "00065": (-1e3, 1e4), "72137": (-1e7, 1e7),
}
# Variables where small negative sensor readings are clipped to 0 (inside the plausible range).
CLIP_ZERO = {"63680", "63675", "99137", "32315", "32316", "32319", "32321"}


def _tolerance(parm: str, ref: pd.Series) -> float:
    med = float(np.nanmedian(np.abs(ref))) if len(ref) else 0.0
    fixed = {"00010": 0.5, "00300": 0.5, "00301": 5.0, "00400": 0.2, "99133": 0.2, "99137": 0.2}
    if parm in fixed:
        return fixed[parm]
    return max(0.2 * med, 1.0) if parm in {"63680", "63675", "32295", "32315", "32316", "32319", "32321"} \
        else max(0.05 * med, 1.0)


MASK_QUALIFIER = r"(?:\be\b|:e|\[4\]|<|>)"


def clean_long(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Mask qualified/implausible values in a long DV table; return (clean, counts)."""
    counts = {"rows_in": int(len(df))}
    df = df[df["parm_cd"].isin(ALL_PARAMS) & df["stat_cd"].isin(STATS)].copy()
    q = df["qualifier"].astype(str)
    bad_q = q.str.contains(MASK_QUALIFIER, regex=True)
    counts["masked_qualifier"] = int((bad_q & df["value"].notna()).sum())
    df.loc[bad_q, "value"] = np.nan
    lo = df["parm_cd"].map(lambda p: PLAUSIBLE[p][0])
    hi = df["parm_cd"].map(lambda p: PLAUSIBLE[p][1])
    out = (df["value"] < lo) | (df["value"] > hi)
    counts["masked_implausible"] = int(out.sum())
    df.loc[out, "value"] = np.nan
    clip = df["parm_cd"].isin(CLIP_ZERO) & (df["value"] < 0)
    counts["clipped_to_zero"] = int(clip.sum())
    df.loc[clip, "value"] = 0.0
    df["provisional"] = q.str.startswith("P")
    df = df[df["value"].notna()]
    counts["rows_valid"] = int(len(df))
    return df, counts


def splice_site(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """One site's clean long table -> wide daily frame + splice log."""
    log = []
    cols = {}
    for (parm, stat), g in df.groupby(["parm_cd", "stat_cd"], observed=True):
        name = f"{ALL_PARAMS[parm]}_{STATS[stat]}"
        series = {ts: s.set_index("date")[["value", "provisional"]] for ts, s in g.groupby("ts_id")}
        order = sorted(series, key=lambda t: -len(series[t]))
        prim = series[order[0]].copy()
        prim["ts"] = order[0]
        for ts in order[1:]:
            sec = series[ts]
            ov = prim.index.intersection(sec.index)
            ok = True
            if len(ov) >= 30:
                mad = float(np.nanmedian(np.abs(prim.loc[ov, "value"] - sec.loc[ov, "value"])))
                tol = _tolerance(parm, prim["value"])
                ok = mad <= tol
                log.append({"var": name, "primary": order[0], "secondary": ts, "overlap_days": len(ov),
                            "median_abs_diff": mad, "tolerance": tol, "used": ok})
            else:
                log.append({"var": name, "primary": order[0], "secondary": ts, "overlap_days": len(ov),
                            "median_abs_diff": None, "tolerance": None, "used": True})
            if ok:
                add = sec.loc[sec.index.difference(prim.index)].copy()
                add["ts"] = ts
                prim = pd.concat([prim, add])
        prim = prim.sort_index()
        cols[name] = prim["value"].astype("float32")
        cols[f"{name}_prov"] = prim["provisional"].astype(bool)
        cols[f"{name}_ts"] = prim["ts"].astype(str)
    if not cols:
        return pd.DataFrame(), log
    wide = pd.DataFrame(cols)
    wide.index = pd.to_datetime(wide.index)
    wide.index.name = "date"
    for c in wide.columns:
        if c.endswith("_prov"):
            wide[c] = wide[c].astype("boolean").fillna(False).astype(bool)
    return wide.sort_index(), log
