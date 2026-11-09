# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""QC for harmonised WQP discrete results, with per-step row counts.

Steps (applied in this order by :func:`qc_chunk`; each removal is counted per variable):

1. ``non_water_media``      Activity_Media != "Water" (requests already ask for sampleMedia=Water)
2. ``qc_activity``          QC activity types (blanks, spikes, replicates, duplicates, reference)
3. ``logger_or_aggregate``  data-logger activity types, time-aggregated results (TimeBasis set),
                            summary statistics (daily/weekly min/max/median/geometric mean...) and
                            logger-like series posted as field measurements: > 4 distinct values
                            of one characteristic at the same site/date/time/depth (time may be
                            blank), or (DO, pH, temperature) > 12 distinct times at one
                            site/date/depth.
                            "Minimum"
                            Secchi/tube readings were turned into right-censored values at parse
                            time and are kept
4. ``rejected_status``      Result_MeasureStatusIdentifier = Rejected
5. ``sentinel_code``        values equal to a sentinel code (-999999, -9999, 999999, ...) masked at
                            parse time and not censored
6. ``no_value``             no numeric value and not censored (non-numeric text, "Not Reported",
                            censored without any limit)
7. ``out_of_range``         value (or the detection limit of a censored row) outside the variable's
                            plausible range (``VARIABLES[v].lo/hi``)
8. ``speciation_duplicate`` the same activity reported in two speciations (e.g. orthophosphate as P
                            and as PO4, USGS ALGOR-derived): the converted copy is dropped
9. ``duplicate``            same variable, characteristic, site, local date+time, depth, value (6
                            significant digits), censoring and limit (rows without a time must
                            also share the activity): keep one (accepted > preliminary, then most
                            recent LastChangeDate)
10. ``not_co_sampled``      (physchem only) no QC-passed target result at the same site on the
                            same local date
"""

from __future__ import annotations

import re
from collections import defaultdict

import numpy as np
import pandas as pd

from .wqp_harmonize import VARIABLES, depth_m

QC_STEPS = ("non_water_media", "qc_activity", "logger_or_aggregate", "rejected_status",
            "sentinel_code", "no_value", "out_of_range", "speciation_duplicate", "duplicate",
            "not_co_sampled")

_OK_STAT = {"", "mpn", "mean", "actual", "none"}


class QCReport:
    """Accumulates per-variable counts: input rows, removals per step, output rows."""

    def __init__(self):
        self.counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def add(self, variable: pd.Series, step: str) -> None:
        for v, n in variable.value_counts().items():
            self.counts[str(v)][step] += int(n)

    def merge(self, other: "QCReport") -> None:
        for v, d in other.counts.items():
            for k, n in d.items():
                self.counts[v][k] += n

    def table(self) -> pd.DataFrame:
        cols = ["input", *QC_STEPS, "output"]
        df = pd.DataFrame({v: {c: d.get(c, 0) for c in cols} for v, d in self.counts.items()}).T
        df = df.reindex(columns=cols).fillna(0).astype("int64").sort_index()
        df.index.name = "variable"
        return df

    def to_dict(self) -> dict:
        return {v: dict(d) for v, d in self.counts.items()}


FIELD_METER_VARS = ("temp", "do", "do_pctsat", "ph")
MAX_SAME_INSTANT = 4
MAX_TIMES_PER_DAY = 12


def high_frequency_mask(df: pd.DataFrame) -> pd.Series:
    """Logger-like series posted as discrete field measurements (see step 3)."""
    t = df["activity_start_time"].fillna("").astype(str).str.strip()
    inst = [df["variable"], df["characteristic"], df["site_id"], df["activity_start_date"], t,
            df["depth_m_key"]]
    vals = _sig(df["value"].where(~df["censored"], df["detection_limit"]), 4).fillna(-1e300)
    a = vals.groupby(inst).transform("nunique") > MAX_SAME_INSTANT
    day = [df["variable"], df["site_id"], df["activity_start_date"], df["depth_m_key"]]
    meter = df["variable"].isin(FIELD_METER_VARS)
    n_times = t.where(t != "").groupby(day).transform("nunique")
    b = meter & (n_times > MAX_TIMES_PER_DAY)
    return (a | b).fillna(False).astype(bool)


def aggregate_mask(df: pd.DataFrame) -> pd.Series:
    """True for data-logger records, time-aggregated results, summary statistics and
    logger-like high-frequency series (see :func:`high_frequency_mask`)."""
    stat = df["statistical_base"].fillna("").astype(str).str.strip().str.lower()
    tb = df["time_basis"].fillna("").astype(str).str.strip()
    clarity_min = df["variable"].isin(["secchi", "transparency_tube"]) & (stat == "minimum") & (
        df["censor_type"] == ">")
    bad_stat = ~stat.isin(_OK_STAT) & ~clarity_min
    hf = high_frequency_mask(df) if "depth_m_key" in df.columns else False
    return (df["qc_class"] == "logger") | (tb != "") | bad_stat | hf


def _sig(x: pd.Series, digits: int = 6) -> pd.Series:
    """Round to ``digits`` significant figures (NaN-safe)."""
    v = x.to_numpy(dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        mag = np.where((v == 0) | ~np.isfinite(v), 0, np.floor(np.log10(np.abs(v))))
        scale = 10.0 ** (digits - 1 - mag)
        out = np.round(v * scale) / scale
    return pd.Series(np.where(np.isfinite(v), out, np.nan), index=x.index)


def range_mask(df: pd.DataFrame) -> pd.Series:
    """True where the value (or, for censored rows, the limit) is outside the plausible range."""
    lo = df["variable"].map({k: v.lo for k, v in VARIABLES.items()}).astype("float64")
    hi = df["variable"].map({k: v.hi for k, v in VARIABLES.items()}).astype("float64")
    val = df["value"].where(~df["censored"], df["detection_limit"])
    bad = val.notna() & ((val < lo) | (val > hi))
    # a left-censored limit of 0 is uninformative but not implausible
    bad &= ~(df["censored"] & (df["censor_type"] == "<") & (val == 0))
    return bad


def speciation_duplicate_mask(df: pd.DataFrame) -> pd.Series:
    """Converted-speciation rows whose activity also reports the variable in its native basis."""
    basis = df["variable"].map({k: v.basis for k, v in VARIABLES.items()}).fillna("")
    converted = (basis != "") & (df["species_used"] != "") & (df["species_used"] != basis)
    if not converted.any():
        return pd.Series(False, index=df.index)
    key = ["variable", "activity_id", "fraction_class", "depth_m_key"]
    native = df.loc[~converted & (basis != ""), key].drop_duplicates()
    native["_has_native"] = True
    probe = df.loc[converted, key].merge(native, on=key, how="left")["_has_native"]
    probe = probe.astype("boolean").fillna(False)
    out = pd.Series(False, index=df.index)
    out[converted[converted].index] = probe.to_numpy(dtype=bool)
    return out


def duplicate_mask(df: pd.DataFrame) -> pd.Series:
    """Exact duplicates (same variable/site/date/time/depth/value/censoring/limit); keeps the
    accepted, most recently changed record."""
    if df.empty:
        return pd.Series(False, index=df.index)
    t = df["activity_start_time"].fillna("").astype(str).str.strip()
    k = pd.DataFrame({
        "variable": df["variable"], "characteristic": df["characteristic"],
        "site_id": df["site_id"], "date": df["activity_start_date"], "time": t,
        "activity": df["activity_id"].where(t == "", ""), "depth": df["depth_m_key"],
        "value": _sig(df["value"]).fillna(-1e300), "censor": df["censor_type"],
        "dl": _sig(df["detection_limit"]).fillna(-1e300),
    })
    order = pd.DataFrame({
        "acc": (df["status_class"] != "accepted").astype(int),
        "chg": df["last_change_date"].fillna(""),
    })
    ranked = k.assign(_acc=order["acc"], _chg=order["chg"]).sort_values(
        ["_acc", "_chg"], ascending=[True, False], kind="stable")
    dup = ranked.duplicated(subset=list(k.columns), keep="first")
    return dup.reindex(df.index)


def qc_chunk(m: pd.DataFrame, report: QCReport, co_sampled: set | None = None
             ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the QC steps to harmonised rows. Returns ``(kept, rejected)``; ``rejected`` has a
    ``qc_reason`` column (one reason per row: the first failing step). ``co_sampled`` (a set of
    ``(site_id, activity_start_date)``) enables step 10 for physchem rows."""
    if m.empty:
        return m, m.assign(qc_reason=pd.Series(dtype=object))
    m = m.copy()
    m["depth_m"] = depth_m(m["activity_depth"], m["activity_depth_unit"])
    rd = depth_m(m["result_depth"], m["result_depth_unit"])
    m["depth_m"] = m["depth_m"].where(m["depth_m"].notna(), rd)
    m["depth_m_key"] = m["depth_m"].round(2).fillna(-1.0)
    report.add(m["variable"], "input")
    rejected = []

    def drop(mask: pd.Series, step: str):
        nonlocal m
        mask = mask.fillna(False).astype(bool)
        if mask.any():
            r = m[mask]
            report.add(r["variable"], step)
            rejected.append(r.assign(qc_reason=step))
            m = m[~mask]

    drop(m["media"].fillna("").str.strip().str.lower().ne("water"), "non_water_media")
    drop(m["qc_class"].isin(["blank", "spike", "replicate", "reference", "other_qc"]), "qc_activity")
    drop(aggregate_mask(m), "logger_or_aggregate")
    drop(m["status_class"] == "rejected", "rejected_status")
    drop((m["value_flag"] == "sentinel_code") & ~m["censored"], "sentinel_code")
    no_value = (m["value"].isna() & ~m["censored"]) | (m["censored"] & m["detection_limit"].isna())
    drop(no_value, "no_value")
    drop(range_mask(m), "out_of_range")
    drop(speciation_duplicate_mask(m), "speciation_duplicate")
    drop(duplicate_mask(m), "duplicate")
    if co_sampled is not None and len(m):
        keys = pd.Series(list(zip(m["site_id"], m["activity_start_date"])), index=m.index)
        drop(~keys.isin(co_sampled), "not_co_sampled")
    report.add(m["variable"], "output")
    rej = pd.concat(rejected, ignore_index=True) if rejected else m.iloc[:0].assign(qc_reason="")
    return m.drop(columns=["depth_m_key"]), rej.drop(columns=["depth_m_key"], errors="ignore")


_SITE_CLASSES = (  # earlier entries win
    ("lake", re.compile(r"lake|reservoir|impoundment|pond", re.I)),
    ("estuary_ocean", re.compile(r"estuar|ocean|tidal|bay|harbor|harbour|sea\b|coast", re.I)),
    ("wetland", re.compile(r"wetland|marsh|swamp", re.I)),
    ("stream", re.compile(r"stream|river|creek|canal|ditch|channel|spring|seep", re.I)),
    ("groundwater", re.compile(r"well|ground ?water|aquifer|subsurface|borehole", re.I)),
    ("facility", re.compile(r"facility|outfall|sewer|sewage|effluent|treatment|pipe|storm|mine", re.I)),
    ("beach", re.compile(r"beach", re.I)),
)


def site_class(site_type: pd.Series) -> pd.Series:
    """Coarse class from WQX MonitoringLocationTypeName (BEACH sites are classed by water body)."""
    s = site_type.fillna("").astype(str)
    out = pd.Series("other", index=s.index, dtype=object)
    for name, pat in reversed(_SITE_CLASSES):
        out[s.str.contains(pat)] = name
    beach = s.str.contains("BEACH", case=False)
    out[beach & s.str.contains("Lake", case=False)] = "lake"
    out[beach & s.str.contains("Ocean|Estuary|Bay", case=False)] = "estuary_ocean"
    out[beach & s.str.contains("River|Stream", case=False)] = "stream"
    out[s == ""] = "unknown"
    return out
