#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""IV-derived daily rule: capped revision on the DEV half, single confirmation on the CONFIRM half (lead ruling
2026-11-10, option ii; split + rules frozen first in ``docs/preregistration/sonde_ivrule_split.json``).

Split v2 (C-F2): confirmation half without the 54 previously evaluated sites; dev selection needs BOTH criteria.
Variants (≤ 3 revisions; v0 = the rule that failed):
  v0  all IV series of a parameter pooled per local-STANDARD-time day (current ``download_newsites.daily_from_iv``)
  R1  per-series daily stats; the series with the most kept days is primary, other series fill only missing days
      (as ``sonde_daily.splice_site``); local standard time; frozen IV QC (``sonde_hourly.clean_iv``)
  R2  R1 with DST-aware local time (``tz_cd`` of each record)
  R3  R1 with plausibility-only QC (no qualifier masking)
All: day kept if its count ≥ 80 % of the series' median daily count in that water year.
Population: train-role split sites with raw legacy IV, TRAIN BLOCK only. No new-site data.

  --phase dev       evaluate v0, R1, R2, R3 on the dev half; select per the frozen rule; log each to the registry
  --phase confirm   evaluate the SELECTED variant ONCE on the confirm half (O_EXCL marker; refuses a second run)

    python -B scripts/data/ivrule_revision.py --phase dev
    python -B scripts/data/ivrule_revision.py --phase confirm
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd

from sentinel.core import paths
from sentinel.core.registry import log_result
from sentinel.data.sonde_daily import PLAUSIBLE
from sentinel.data.sonde_hourly import clean_iv
from sentinel.data.sources.usgs_legacy import ALL_PARAMS, TZ_OFFSETS_H
from sentinel.eval.hazards import HAZARDS, daily_onsets

SPLIT = paths.REPO_ROOT / "docs" / "preregistration" / "sonde_ivrule_split_v2.json"   # v2 (C-F2)
SPLIT_SHA = "f2ebe52f53f4"
OUT = paths.RUNS_ROOT / "sonde" / "ivrule_revision"
TRAIN_END = pd.Timestamp("2021-09-30")
STD = {"EST": -5, "EDT": -5, "CST": -6, "CDT": -6, "MST": -7, "MDT": -7, "PST": -8, "PDT": -8, "AKST": -9, "AKDT": -9,
       "HST": -10, "AST": -4, "ADT": -4, "UTC": 0, "GMT": 0}
VARIANTS = ("v0", "R1", "R2", "R3")
DO_PC = "00300"


def light_clean(df):
    df = df[df["parm_cd"].astype(str).isin(ALL_PARAMS)].copy()
    df["parm_cd"] = df["parm_cd"].astype(str)
    lo = df["parm_cd"].map(lambda p: PLAUSIBLE[p][0])
    hi = df["parm_cd"].map(lambda p: PLAUSIBLE[p][1])
    df.loc[(df["value"] < lo) | (df["value"] > hi), "value"] = np.nan
    df["provisional"] = df["qualifier"].astype(str).str.startswith("P")
    return df[df["value"].notna()]


def daily_min(cl: pd.DataFrame, *, per_series: bool, dst: bool) -> pd.Series:
    d = cl[cl["parm_cd"] == DO_PC]
    if d.empty:
        return pd.Series(dtype=float)
    t = pd.to_datetime(d["time_utc"], utc=True).dt.tz_localize(None)
    off = d["tz_cd"].astype(str).map(TZ_OFFSETS_H if dst else STD).fillna(-6).astype(float)
    day = (t + pd.to_timedelta(off, unit="h")).dt.floor("D")
    x = pd.DataFrame({"day": day.values, "v": d["value"].astype(float).values,
                      "ts": d["ts_id"].astype(str).values if per_series else "all"})
    out = None
    for _, g in sorted(x.groupby("ts"), key=lambda kv: -kv[1]["day"].nunique()):
        a = g.groupby("day").agg(n=("v", "size"), mn=("v", "min"))
        wy = a.index.year + (a.index.month >= 10)
        a = a[a["n"] >= 0.8 * a.groupby(wy)["n"].transform("median")]["mn"]
        out = a if out is None else pd.concat([out, a[~a.index.isin(out.index)]]).sort_index()
    return out if out is not None else pd.Series(dtype=float)


def variant_series(df, v):
    if v == "v0":
        return daily_min(clean_iv(df), per_series=False, dst=False)
    if v == "R1":
        return daily_min(clean_iv(df), per_series=True, dst=False)
    if v == "R2":
        return daily_min(clean_iv(df), per_series=True, dst=True)
    if v == "R3":
        return daily_min(light_clean(df), per_series=True, dst=False)
    raise ValueError(v)


def evaluate(sites, variants):
    snap = (paths.PROCESSED / "sonde_daily").resolve() / "sites"
    acc = {v: {"n": 0, "ok": 0, "on_iv": 0, "on_dv": 0, "on_both": 0} for v in variants}
    for s in sites:
        files = [f for f in sorted((paths.RAW / "usgs_iv" / s).glob("*.parquet")) if int(f.stem) <= 2021]
        p = snap / f"{s}.parquet"
        if not files or not p.exists():
            continue
        dv = pd.read_parquet(p)
        if "do_mgl_min" not in dv:
            continue
        dv.index = pd.to_datetime(dv.index)
        b = dv["do_mgl_min"][dv.index <= TRAIN_END].dropna()
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        if df.empty or not (df["parm_cd"].astype(str) == DO_PC).any():
            continue
        for v in variants:
            a = variant_series(df, v)
            a = a[a.index <= TRAIN_END]
            j = pd.concat([a.rename("iv"), b.rename("dv")], axis=1).dropna()
            acc[v]["n"] += len(j)
            acc[v]["ok"] += int((np.abs(j["iv"] - j["dv"]) <= 0.1).sum())
            if len(a) and len(b):
                idx = pd.date_range(min(a.index.min(), b.index.min()), TRAIN_END)
                _, oa = daily_onsets(a.reindex(idx).to_numpy(float), HAZARDS["H1_hypoxia"])
                _, ob = daily_onsets(b.reindex(idx).to_numpy(float), HAZARDS["H1_hypoxia"])
                acc[v]["on_iv"] += int(oa.sum())
                acc[v]["on_dv"] += int(ob.sum())
                acc[v]["on_both"] += int((oa & ob).sum())
    res = {}
    for v, r in acc.items():
        jac = r["on_both"] / max(r["on_iv"] + r["on_dv"] - r["on_both"], 1)
        res[v] = {**r, "frac_within_0.1": r["ok"] / max(r["n"], 1), "onset_jaccard": jac}
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("dev", "confirm"), required=True)
    a = ap.parse_args()
    sp = json.loads(SPLIT.read_text())
    if not hashlib.sha256(SPLIT.read_bytes()).hexdigest().startswith(SPLIT_SHA):
        raise SystemExit("split file changed")
    OUT.mkdir(parents=True, exist_ok=True)
    split_label = f"train/ivrule_{SPLIT_SHA}"
    if a.phase == "dev":
        if (OUT / "dev_selection.json").exists():
            raise SystemExit("dev phase already done (selection is final)")
        res = evaluate(sp["dev"], VARIANTS)
        for v, r in res.items():
            log_result(workstream="M1-sonde", experiment="ew_v3_m1n/ivrule_dev", model=f"daily_from_iv:{v}",
                       metric="frac_within_0.1mgl_do_min", value=r["frac_within_0.1"], n=r["n"], split=split_label,
                       notes="dev half, train block", extra={"onset_jaccard": r["onset_jaccard"]})
        order = ["R1", "R2", "R3"]                       # selection among REVISIONS (v0 is the failed reference)
        ok = [v for v in order if res[v]["frac_within_0.1"] >= 0.99 and res[v]["onset_jaccard"] >= 0.90]   # C-F2
        sel = max(ok, key=lambda v: (round(res[v]["frac_within_0.1"], 6), -order.index(v))) if ok else None
        rec = {"results": res, "selected": sel, "rule": sp["selection"],
               "consequence": None if sel else "no variant met both criteria on the dev half -> no confirmation; "
                                                "option (i): EW-v3-N H1 is NON-CONFIRMATORY"}
        fd = os.open(OUT / "dev_selection.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        with os.fdopen(fd, "w") as fh:
            fh.write(json.dumps(rec, indent=1))
        print(json.dumps(rec, indent=1))
    else:
        sel = json.loads((OUT / "dev_selection.json").read_text())["selected"]
        if sel is None:
            raise SystemExit("no eligible variant on the dev half: option (i) applies; no confirmation is run")
        marker = OUT / "CONFIRMED_ONCE.json"
        fd = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)   # single confirmation
        res = evaluate(sp["confirm"], (sel,))[sel]
        ok = bool(res["frac_within_0.1"] >= 0.99 and res["onset_jaccard"] >= 0.90)
        rec = {"selected": sel, "confirm": res, "passed": ok,
               "consequence": "IV-derived daily tier admitted to the primary" if ok else
               "FAILED -> option (i): EW-v3-N H1 is NON-CONFIRMATORY; IV-only sites stay a secondary cell"}
        with os.fdopen(fd, "w") as fh:
            fh.write(json.dumps(rec, indent=1))
        log_result(workstream="M1-sonde", experiment="ew_v3_m1n/ivrule_confirm", model=f"daily_from_iv:{sel}",
                   metric="frac_within_0.1mgl_do_min", value=res["frac_within_0.1"], n=res["n"], split=split_label,
                   notes=f"confirm half, single look; passed={ok}", extra={"onset_jaccard": res["onset_jaccard"]})
        print(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
