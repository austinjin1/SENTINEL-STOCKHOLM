# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Sub-daily (hourly-tier) DO/temperature features for the EW-v2 event heads (daily issue).

For each site and LOCAL STANDARD-TIME day d (site UTC offset from the site registry, no DST), from
hourly means of the hourly tier (processed/sonde_hourly; QC'd USGS IV) whose hour ENDS within day d:
  do_amp        diel DO amplitude (max − min of hourly means)
  do_night_min  minimum hourly DO in local hours 00–06
  do_eve_mean   mean hourly DO in local hours 18–23 (closest to issue time)
  do_hrs_lt5    number of hours with DO < 5 mg/L
  temp_amp      diel water-temperature amplitude
  do_cov        fraction of the 24 hours observed
plus trailing summaries (3-day night-minimum slope, 3-day mean amplitude). Provisional hourly
values are dropped (approved only). Day d's features use only hours ending by the end of day d,
which is when the daily issue is made — the same information cut-off as the daily tier.
Sites without hourly data get NaN (tree models / imputation flags handle it).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...core import paths
from .features import lag, rolling_mean, rolling_slope

HOURLY_DIR = paths.PROCESSED / "sonde_hourly" / "sites"
FEATS = ("do_amp", "do_night_min", "do_eve_mean", "do_hrs_lt5", "temp_amp", "do_cov")


def site_daily_from_hourly(df: pd.DataFrame, utc_offset_h: float, dates: pd.DatetimeIndex) -> np.ndarray:
    """(T, len(FEATS)) local-day features for one site's hourly table (index time_utc)."""
    out = np.full((len(dates), len(FEATS)), np.nan, dtype=np.float32)
    if "do_mgl_mean" not in df.columns:
        return out
    t = pd.DatetimeIndex(df.index).tz_convert("UTC").tz_localize(None)
    # hour-ending local time: an hourly mean stamped at the hour start covers [t, t+1h)
    end_local = t + pd.Timedelta(hours=1) + pd.Timedelta(hours=float(utc_offset_h))
    day = (end_local - pd.Timedelta(microseconds=1)).normalize()
    hour = (end_local - pd.Timedelta(hours=1)).hour
    do = df["do_mgl_mean"].to_numpy(float)
    if "do_mgl_prov" in df.columns:
        do = np.where(df["do_mgl_prov"].to_numpy(bool), np.nan, do)
    tp = df["temp_c_mean"].to_numpy(float) if "temp_c_mean" in df.columns else np.full(len(df), np.nan)
    if "temp_c_prov" in df.columns:
        tp = np.where(df["temp_c_prov"].to_numpy(bool), np.nan, tp)
    g = pd.DataFrame({"day": day, "hour": hour, "do": do, "tp": tp})
    g = g[g["day"].between(dates[0], dates[-1])]
    if g.empty:
        return out
    gb = g.groupby("day")
    feats = pd.DataFrame({
        "do_amp": gb["do"].max() - gb["do"].min(),
        "do_night_min": g[g["hour"] <= 6].groupby("day")["do"].min(),
        "do_eve_mean": g[g["hour"] >= 18].groupby("day")["do"].mean(),
        "do_hrs_lt5": g.assign(b=(g["do"] < 5).astype(float).where(np.isfinite(g["do"]))).groupby("day")["b"].sum(min_count=1),
        "temp_amp": gb["tp"].max() - gb["tp"].min(),
        "do_cov": gb["do"].count() / 24.0,
    })
    feats.loc[feats["do_cov"] < 0.5, ["do_amp", "do_night_min", "do_eve_mean", "do_hrs_lt5"]] = np.nan
    feats = feats.reindex(dates)
    return feats[list(FEATS)].to_numpy(np.float32)


def load_subdaily(sites: list[str], dates: pd.DatetimeIndex, utc_offset: dict) -> np.ndarray:
    """(S, T, len(FEATS)); NaN for sites without an hourly table."""
    out = np.full((len(sites), len(dates), len(FEATS)), np.nan, dtype=np.float32)
    cols = ["do_mgl_mean", "do_mgl_prov", "temp_c_mean", "temp_c_prov"]
    for i, s in enumerate(sites):
        p = HOURLY_DIR / f"{s}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        df = df[[c for c in cols if c in df.columns]]
        out[i] = site_daily_from_hourly(df, utc_offset.get(s, -6.0), dates)
    return out


def subdaily_blocks(arr: np.ndarray) -> dict[str, np.ndarray]:
    """Dense causal feature blocks (S, T) for FeatureBuilder.table(..., blocks=...)."""
    blk = {f"sd_{n}": arr[:, :, k].astype(float) for k, n in enumerate(FEATS)}
    nm = blk["sd_do_night_min"]
    blk["sd_night_min_slope3"] = rolling_slope(nm, 3, 2)
    blk["sd_night_min_chg1"] = nm - lag(nm, 1)
    blk["sd_amp_mean3"] = rolling_mean(blk["sd_do_amp"], 3, 1)
    blk["sd_hrs_lt5_sum3"] = rolling_mean(blk["sd_do_hrs_lt5"], 3, 1) * 3
    return blk
