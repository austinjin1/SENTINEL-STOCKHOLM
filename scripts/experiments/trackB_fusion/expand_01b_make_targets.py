#!/usr/bin/env python3
"""EXPAND step 1b -- turn mc_expand_raw into S2 download targets (breadth-capped).

One row per (site,date) with the MAX microcystin that day (so a positive day
stays positive). Caps rows per HUC2 to keep S2 downloads tractable while
prioritizing region BREADTH. Keeps all NEW regions; down-samples the
already-huge original regions (03 Florida) so budget spreads across domains.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/bcheng/SENTINEL")
OUT_DIR = ROOT / "data/processed/biology/noaa_habs"
RAW = OUT_DIR / "mc_expand_raw.parquet"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap-per-region", type=int, default=400)
    args = ap.parse_args()
    df = pd.read_parquet(RAW)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    # one row per site-date: keep the max toxin (preserve positives)
    df = (df.sort_values("val")
            .drop_duplicates(subset=["Location_Identifier", "date"], keep="last")
            .reset_index(drop=True))
    df["huc2"] = df["huc2"].fillna("XX")

    # per region: keep ALL positives (mc>0.3) + a sample of negatives up to cap
    rng = np.random.default_rng(0)
    keep = []
    for huc, g in df.groupby("huc2"):
        pos = g[g.val > 0.3]
        neg = g[g.val <= 0.3]
        n_neg = max(0, args.cap_per_region - len(pos))
        if len(neg) > n_neg:
            neg = neg.sample(n=n_neg, random_state=0)
        keep.append(pd.concat([pos, neg]))
    out = pd.concat(keep).reset_index(drop=True)
    out.to_parquet(OUT_DIR / "mc_expand_targets.parquet", index=False)
    print(f"targets: {len(out)} rows, {out.Location_Identifier.nunique()} sites, "
          f"{out.huc2.replace('XX', np.nan).dropna().nunique()} HUC2 regions")
    g = out.groupby("huc2").agg(rows=("val", "size"),
        sites=("Location_Identifier", "nunique"),
        pos16=("val", lambda v: int((v > 1.6).sum())),
        pos03=("val", lambda v: int((v > 0.3).sum()))).sort_values("rows", ascending=False)
    pd.set_option("display.width", 200)
    print(g)


if __name__ == "__main__":
    main()
