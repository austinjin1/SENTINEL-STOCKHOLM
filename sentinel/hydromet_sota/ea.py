# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Environment Agency (England) Hydrology API — water-quality sondes for the M6 new-geography track
(``docs/plans/ea_england_allocation.md``; build ``scripts/data/sota_hydromet_ea_build.py``).

API (checked 2026-11-10): ``https://environment.data.gov.uk/hydrology``
* ``/id/stations.json?observedProperty=<p>&_limit=10000`` — stations (notation, label, lat, long, easting, northing,
  dateOpened, dateClosed, status, riverName/catchmentArea/nrfaStationID for hydrometric stations);
* ``/id/measures.json?observedProperty=<p>&_limit=10000`` — measures (``<station>-temp-i-subdaily-C``,
  ``<station>-do-i-subdaily-mgL``, ``<station>-do-i-subdaily-pct``, daily qualified temperature at 16 flow stations);
* ``/data/readings.csv?observedProperty=<p>&mineq-date=<d0>&max-date=<d1>&_limit=2000000`` — all readings of a
  property in [d0, d1) (``max-date`` exclusive), columns ``measure, dateTime, date, value, completeness, quality, qcode``.
Properties used: ``temperature``, ``dissolved-oxygen``. Licence: Open Government Licence v3.0 ("Contains Environment
Agency information © Environment Agency and/or database right").

**Reservation (allocation note, rule 1):** every EA value dated on or after 2023-10-01 is M6's unseen test block. This
module never requests a date ≥ :data:`RESERVED_FROM` and :func:`parse_readings` refuses any such row.

Daily statistics (pure functions, :func:`daily_stats`): readings of quality Suspect/Missing and non-finite values are
dropped; physical range checks; spikes (> threshold from the centred 5-reading median) and flat lines (≥ 12 h of an
identical value) are removed; a day is valid when the valid readings cover ≥ 80 % of the expected count (from the
measure's modal interval) and span ≥ 20 h.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd

BASE = "https://environment.data.gov.uk/hydrology"
RESERVED_FROM = pd.Timestamp("2023-10-01")
START = pd.Timestamp("2012-01-01")
PROPS = ("temperature", "dissolved-oxygen")
LICENCE = ("Open Government Licence v3.0 — Contains Environment Agency information © Environment Agency and/or "
           "database right")
RANGES = {"temp_c": (-2.0, 35.0), "do_mgl": (0.0, 25.0), "do_pct": (0.0, 300.0)}
SPIKE = {"temp_c": 5.0, "do_mgl": 5.0, "do_pct": 60.0}
FLAT_HOURS = 12.0
MIN_COVER = 0.8
MIN_SPAN_H = 20.0
DROP_QUALITY = {"Suspect", "Missing"}


def month_starts(start=START, end=RESERVED_FROM) -> list[pd.Timestamp]:
    return list(pd.date_range(start, end - pd.Timedelta(days=1), freq="MS"))


def readings_url(prop: str, m0: pd.Timestamp) -> tuple[str, dict]:
    m1 = min(m0 + pd.offsets.MonthBegin(1), RESERVED_FROM)
    if m1 > RESERVED_FROM or m0 >= RESERVED_FROM:
        raise ValueError("request reaches into the reserved EA test period (>= 2023-10-01)")
    return (f"{BASE}/data/readings.csv", {"observedProperty": prop, "mineq-date": m0.strftime("%Y-%m-%d"),
                                          "max-date": m1.strftime("%Y-%m-%d"), "_limit": 2000000})


def measure_var(notation: str) -> str | None:
    n = notation.rsplit("/", 1)[-1]
    if "-temp-" in n:
        return "temp_c" if n.endswith("-C") or "-C-" in n else None
    if "-do-" in n:
        return "do_mgl" if n.endswith("mgL") else ("do_pct" if n.endswith("pct") else None)
    return None


def parse_readings(raw: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw), dtype={"measure": str, "quality": str, "completeness": str, "qcode": str},
                     usecols=["measure", "dateTime", "date", "value", "completeness", "quality", "qcode"])
    df["measure"] = df["measure"].str.rsplit("/", n=1).str[-1]
    df["dateTime"] = pd.to_datetime(df["dateTime"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if (df["date"] >= RESERVED_FROM).any() or (df["dateTime"] >= RESERVED_FROM).any():
        raise ValueError("EA readings dated >= 2023-10-01 (reserved test block) in a pre-freeze download")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def _flat_mask(v: np.ndarray, t: np.ndarray, hours: float) -> np.ndarray:
    """True for readings inside a run of identical consecutive values lasting ≥ ``hours``."""
    out = np.zeros(len(v), bool)
    if len(v) < 2:
        return out
    brk = np.r_[True, v[1:] != v[:-1]]
    rid = np.cumsum(brk) - 1
    first = pd.Series(t).groupby(rid).transform("min").to_numpy()
    last = pd.Series(t).groupby(rid).transform("max").to_numpy()
    dur = (last - first) / np.timedelta64(1, "h")
    return dur >= hours


def clean_measure(g: pd.DataFrame, var: str) -> tuple[pd.DataFrame, dict]:
    """QC one measure's readings (sorted by time). Returns kept rows and drop counts."""
    g = g.sort_values("dateTime")
    info = {"n_raw": int(len(g))}
    bad_q = g["quality"].isin(DROP_QUALITY) | ~np.isfinite(g["value"].to_numpy(dtype="float64")) | g["dateTime"].isna()
    info["drop_quality_or_missing"] = int(bad_q.sum())
    g = g[~bad_q]
    lo, hi = RANGES[var]
    rng = (g["value"] < lo) | (g["value"] > hi)
    info["drop_range"] = int(rng.sum())
    g = g[~rng]
    if len(g) >= 5:
        med = g["value"].rolling(5, center=True, min_periods=3).median()
        sp = (g["value"] - med).abs() > SPIKE[var]
        info["drop_spike"] = int(sp.sum())
        g = g[~sp.to_numpy()]
    else:
        info["drop_spike"] = 0
    fl = _flat_mask(g["value"].to_numpy(), g["dateTime"].to_numpy(), FLAT_HOURS)
    info["drop_flatline"] = int(fl.sum())
    g = g[~fl]
    info["n_kept"] = int(len(g))
    return g, info


def daily_stats(g: pd.DataFrame, var: str) -> pd.DataFrame:
    """Per-day mean/min/max of QC'd readings of one measure, with coverage and validity."""
    if g.empty:
        return pd.DataFrame()
    dt = g["dateTime"].diff().dt.total_seconds().dropna()
    dt = dt[dt > 0]
    step = float(dt.round(-1).mode().iloc[0]) if len(dt) else 900.0
    step = min(max(step, 60.0), 3600.0)
    expected = 86400.0 / step
    d = g.groupby("date").agg(n=("value", "size"), mean=("value", "mean"), min=("value", "min"), max=("value", "max"),
                              t0=("dateTime", "min"), t1=("dateTime", "max"),
                              n_unchecked=("quality", lambda s: int((s == "Unchecked").sum())),
                              n_good=("quality", lambda s: int((s == "Good").sum())))
    d["span_h"] = (d["t1"] - d["t0"]).dt.total_seconds() / 3600 + step / 3600
    d["step_s"] = step
    d["coverage"] = d["n"] / expected
    d["valid"] = (d["coverage"] >= MIN_COVER) & (d["span_h"] >= MIN_SPAN_H)
    return d.drop(columns=["t0", "t1"]).reset_index()
