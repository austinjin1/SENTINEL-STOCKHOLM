#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Build the EW-v2 H2 derived chloride series (sentinel/eval/chloride.py).

Pairs WQP lab chloride at USGS stations with the same station's USGS daily-mean specific
conductance (sonde_daily), fits the pre-specified per-site regression on training-block pairs
and writes derived daily series for eligible sites:

    $SENTINEL_DATA/processed/chloride_derived/regressions.parquet   one row per candidate site
    $SENTINEL_DATA/processed/chloride_derived/pairs.parquet         all pairs (role = fit / later)
    $SENTINEL_DATA/processed/chloride_derived/sites/<site>.parquet  cl_mgl_mean / _4dmean / _max

    python scripts/data/build_chloride_derived.py
    # rebuild from the promoted sonde_daily into a new directory + identity check vs the old build
    python -B scripts/data/build_chloride_derived.py --out-name chloride_derived_refresh_20261108 \
        --compare-with chloride_derived --until 2023-09-30

The identity check (``compare_vs_<old>.json`` in the new directory) requires identical regressions
(a, b, r2, n_pairs, eligible, reason) and identical derived series on every date <= --until (NaN-aware);
rows after --until are cut before comparison and never examined.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from sentinel.core import paths
from sentinel.core.manifest import write_manifest
from sentinel.eval import chloride as C

def compare(old_dir, new_dir, until: str) -> dict:
    """Identity of regressions and of derived series <= ``until`` (old build vs new build)."""
    cut = pd.Timestamp(until)
    cols = ["site_no", "a", "b", "r2", "n_pairs", "eligible", "reason"]
    ro = pd.read_parquet(old_dir / "regressions.parquet")
    rn = pd.read_parquet(new_dir / "regressions.parquet")
    ro, rn = (r[[c for c in cols if c in r.columns]].sort_values("site_no").reset_index(drop=True) for r in (ro, rn))
    regs_same = ro.equals(rn)
    old = {p.stem for p in (old_dir / "sites").glob("*.parquet")}
    new = {p.stem for p in (new_dir / "sites").glob("*.parquet")}
    diff, maxabs = [], 0.0
    for s in sorted(old & new):
        x = pd.read_parquet(old_dir / "sites" / f"{s}.parquet")
        y = pd.read_parquet(new_dir / "sites" / f"{s}.parquet")
        x, y = x[x.index <= cut], y[y.index <= cut]
        if not (x.index.equals(y.index) and list(x.columns) == list(y.columns)):
            diff.append((s, "index/columns"))
            continue
        for c in x.columns:
            a_, b_ = x[c].to_numpy(float), y[c].to_numpy(float)
            if not np.array_equal(np.isnan(a_), np.isnan(b_)):
                diff.append((s, c + ":nan"))
                break
            ok = ~np.isnan(a_)
            if ok.any():
                maxabs = max(maxabs, float(np.max(np.abs(a_[ok] - b_[ok]))))
            if not np.array_equal(a_[ok], b_[ok]):
                diff.append((s, c))
                break
    return {"old": str(old_dir), "new": str(new_dir), "until": until, "regressions_identical": bool(regs_same),
            "n_regressions": int(len(rn)), "n_eligible": int(rn["eligible"].sum()),
            "n_sites_compared": len(old & new), "only_in_old": sorted(old - new), "only_in_new": sorted(new - old),
            "n_sites_differing_until": len(diff), "differing_examples": diff[:20], "max_abs_diff_until": maxabs,
            "identical_until": bool(regs_same and not diff and old == new)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-name", default="chloride_derived")
    ap.add_argument("--sonde-daily", default="sonde_daily", help="processed/<name> daily tier to read SC from")
    ap.add_argument("--compare-with", default=None, help="processed/<name> of the previous build (identity check)")
    ap.add_argument("--until", default="2023-09-30")
    a = ap.parse_args()
    OUT = paths.processed(a.out_name)
    if a.compare_with and OUT.resolve() == paths.processed(a.compare_with).resolve():
        raise SystemExit("--out-name must differ from --compare-with")
    (OUT / "sites").mkdir(parents=True, exist_ok=True)
    SD = paths.PROCESSED / a.sonde_daily
    cl = pd.read_parquet(paths.processed("wqp_discrete"), filters=[("variable", "=", "chloride")],
                         columns=["site_id", "activity_start_date", "value", "unit", "censored",
                                  "activity_type", "provider"])
    cl = cl[cl["site_id"].str.startswith("USGS-") & (cl["unit"] == "mg/L")]
    cl["site_no"] = cl["site_id"].str.slice(5)
    idx = pd.read_parquet(SD / "index.parquet")
    sc_sites = set(idx.loc[idx["var"] == "spcond_uscm_mean", "site_no"])
    cl = cl[cl["site_no"].isin(sc_sites)]
    print(f"{len(cl):,} chloride rows at {cl['site_no'].nunique():,} USGS sites with daily SC")
    regs, pairs_all, n_written = [], [], 0
    for site, g in cl.groupby("site_no"):
        d = pd.read_parquet(SD / "sites" / f"{site}.parquet")
        if "spcond_uscm_mean" not in d:
            continue
        pairs = C.pair_samples(g.rename(columns={"activity_start_date": "date"}), d["spcond_uscm_mean"])
        fit = C.fit_site(pairs)
        regs.append({"site_no": site, **fit, "n_pairs_all": len(pairs)})
        pairs["site_no"] = site
        pairs["role"] = (pairs["date"] <= C.FIT_END).map({True: "fit", False: "later"})
        pairs_all.append(pairs)
        if fit["eligible"]:
            ser = C.derived_series(d["spcond_uscm_mean"], d.get("spcond_uscm_max"), fit)
            ser.to_parquet(OUT / "sites" / f"{site}.parquet", compression="zstd")
            n_written += 1
    regs = pd.DataFrame(regs)
    regs.to_parquet(OUT / "regressions.parquet", index=False)
    pairs_all = pd.concat(pairs_all, ignore_index=True)
    pairs_all.to_parquet(OUT / "pairs.parquet", index=False)
    stats = {"candidate_sites": int(len(regs)), "eligible_sites": int(regs["eligible"].sum()),
             "reasons": regs.loc[~regs["eligible"], "reason"].value_counts().to_dict(),
             "pairs": int(len(pairs_all)), "fit_end": str(C.FIT_END.date()),
             "rules": {"min_pairs": C.MIN_PAIRS, "min_r2": C.MIN_R2, "max_extrap": C.MAX_EXTRAP}}
    stats["sonde_daily_dir"] = str(SD.resolve())
    print(json.dumps(stats, indent=1))
    if a.compare_with:
        cmp_ = compare(paths.processed(a.compare_with), OUT, a.until)
        (OUT / f"compare_vs_{a.compare_with}.json").write_text(json.dumps(cmp_, indent=1))
        print(json.dumps(cmp_, indent=1)[:2000])
        stats["identity_check"] = {k: cmp_[k] for k in ("identical_until", "until", "n_sites_compared",
                                                       "regressions_identical")}
    write_manifest(a.out_name, files=[OUT / "regressions.parquet", OUT / "pairs.parquet",
                                              *sorted((OUT / "sites").glob("*.parquet"))],
                   sources=["manifest:wqp_discrete", f"manifest:{a.sonde_daily}"], hash_files=True,
                   license="US public domain (derived)",
                   description="EW-v2 H2 chloride derived from USGS daily SC via per-site OLS on "
                               "training-block WQP chloride pairs.", stats=stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
