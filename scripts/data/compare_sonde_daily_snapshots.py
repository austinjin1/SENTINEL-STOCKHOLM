#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Verify that a refreshed sonde_daily snapshot is identical to the calibration snapshot on every
date <= --until (train + val blocks end 2023-09-30), site by site and column by column (NaN-aware).
Only rows <= --until are compared; later rows are cut immediately after reading and never examined
(the only post-cut statistic is the pooled share of provisional flags by water year, an approval-
status descriptor that is not stratified by site role).

    python -B scripts/data/compare_sonde_daily_snapshots.py --old sonde_daily --new sonde_daily_refresh_20261108
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from sentinel.core import paths


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--old", default="sonde_daily")
    ap.add_argument("--new", required=True)
    ap.add_argument("--until", default="2023-09-30")
    a = ap.parse_args()
    cut = pd.Timestamp(a.until)
    od, nd = paths.processed(a.old) / "sites", paths.processed(a.new) / "sites"
    old = {p.stem for p in od.glob("*.parquet")}
    new = {p.stem for p in nd.glob("*.parquet")}
    diff_sites, n_cmp, prov = [], 0, {}
    for s in sorted(old & new):
        x = pd.read_parquet(od / f"{s}.parquet")
        y = pd.read_parquet(nd / f"{s}.parquet")
        if getattr(y.index, "tz", None) is not None and cut.tz is None:   # hourly tier (UTC index)
            cut = cut.tz_localize("UTC") + pd.Timedelta(hours=23, minutes=59)
        yp = y[y.index > cut]
        for c in [c for c in yp.columns if c.endswith("_prov") and c.replace("_prov", "") in yp.columns]:
            v = yp[c.replace("_prov", "")].notna()   # pooled approval-status descriptor
            for wy, g in yp.loc[v, c].groupby(np.where(yp.index[v].month >= 10, yp.index[v].year + 1, yp.index[v].year)):
                t = prov.setdefault(int(wy), [0, 0])
                t[0] += int(g.sum())
                t[1] += int(g.size)
        x, y = x[x.index <= cut], y[y.index <= cut]
        n_cmp += 1
        cols = sorted(set(x.columns) | set(y.columns))
        x, y = x.reindex(columns=cols), y.reindex(columns=cols)
        vals = [c for c in cols if not c.endswith(("_prov", "_ts"))]
        x = x.dropna(how="all", subset=vals)
        y = y.dropna(how="all", subset=vals)
        # a flag/series-id column whose value column has no observation in the compared range carries
        # no information (it exists only because the series starts after --until); drop it on both sides
        def has_obs(flag):
            base = flag[: flag.rindex("_")]
            vs = [v for v in vals if (v == base or v.startswith(base + "_")) and not v.endswith("_n")]
            return any(df[v].notna().any() for df in (x, y) for v in vs if v in df.columns)

        empty = [c for c in cols if c.endswith(("_prov", "_ts")) and not has_obs(c)]
        cols = [c for c in cols if c not in empty]
        x, y = x[cols], y[cols]
        if not x.index.equals(y.index):
            diff_sites.append((s, "index"))
            continue
        for c in cols:
            a_, b_ = x[c], y[c]
            if c.endswith(("_prov", "_ts")):
                same = (a_.astype(str) == b_.astype(str)) | (a_.isna() & b_.isna())
            else:
                same = (a_ == b_) | (a_.isna() & b_.isna())
            if not bool(same.all()):
                diff_sites.append((s, c))
                break
    out = {"old": a.old, "new": a.new, "until": a.until, "n_sites_compared": n_cmp,
           "only_in_old": sorted(old - new), "only_in_new": sorted(new - old)[:50], "n_only_in_new": len(new - old),
           "n_sites_differing_until": len(diff_sites), "differing_examples": diff_sites[:20],
           "identical_until": not diff_sites and not (old - new),
           "pooled_frac_provisional_by_wy_after_cut": {k: v[0] / v[1] for k, v in sorted(prov.items()) if v[1]}}
    (paths.processed(a.new) / f"compare_vs_{a.old}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("only_in_new",)}, indent=1)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
