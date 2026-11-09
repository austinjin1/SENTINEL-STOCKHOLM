# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Regenerate the frozen SENTINEL-Fuse splits in memory and verify them against the frozen files.

    python -B scripts/data/make_fuse_splits.py check            # D and C: regenerate, compare sha256
    python -B scripts/data/make_fuse_splits.py check --tier d
    python -B scripts/data/make_fuse_splits.py write --tier d   # only if the file does not exist

Review item W11 (REVIEW_REPORT_PRE_TRAINING.md, Fuse): the generator calls used for the frozen
files. The frozen files are never overwritten (``splits.save_split`` refuses). Tier D depends only on
``processed('neon_sites')``. Tier C was generated from the 2026-10-25 ``fuse_tier_c/rows.parquet``
(pre-dedupe, review W4), rows with ``y_gt8`` labelled; after the rebuild that removed duplicate NLA/WQP rows the regenerated split
can differ in row-count-dependent choices — ``check`` then reports the difference and verifies the
property that matters: every current Tier C site has a frozen role (site roles are what the
frozen file fixes; the rebuild only drops rows).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys

import pandas as pd

from sentinel.core import paths
from sentinel.models.fuse import splits as sp

FROZEN = {"d": (sp.TIER_D_SPLIT, "20adbe595ed42a46c490da31e16743353ce819e9d84cb8eedffa15303a26610a"),
          "c": (sp.TIER_C_SPLIT, "e2fd5de9da4079091445c0635f8ccf1febe7c878a7dc7a3beb679ad0b68e3f37"),
          "wp1": (sp.TIER_WP1_SPLIT, "b9e0d5c44d8a83c3bc4abc8517d66fb520186fcf6e64db59aa3e4d40e050288a")}


def generate(tier: str) -> dict:
    if tier == "wp1":
        m1 = json.loads(sp.M1_SPLIT.read_text())
        return sp.make_tier_wp1_split(m1, sp.file_sha256(sp.M1_SPLIT))
    if tier == "d":
        sites = pd.read_parquet(paths.processed("neon_sites") / "sites.parquet")
        return sp.make_tier_d_split(sites)
    rows = pd.read_parquet(paths.processed("fuse_tier_c") / "rows.parquet",
                           columns=["site_id", "lat", "lon", "huc2", "y_gt8"])
    # the frozen file was generated from the rows with an 8 µg/L label (y_gt8 not NaN); censored rows
    # whose detection limit exceeds 8 µg/L have no label, so 5 such sites have no role (never used).
    return sp.make_tier_c_split(rows[rows["y_gt8"].notna()])


def serialise(obj: dict) -> bytes:
    return json.dumps(obj, indent=1, sort_keys=True, default=str).encode()   # identical to save_split


def check(tier: str) -> bool:
    path, sha = FROZEN[tier]
    frozen_ok = sp.file_sha256(path) == sha
    regen = generate(tier)
    rsha = hashlib.sha256(serialise(regen)).hexdigest()
    same = rsha == sha
    print(f"tier {tier}: frozen file sha ok={frozen_ok}; regenerated sha {rsha[:12]}… identical={same}")
    if not same:
        fz = sp.load_split(path, sha)
        diff = [s for s, r in regen["site_role"].items() if fz["site_role"].get(s) != r]
        print(f"  roles differing between regenerated and frozen: {len(diff)} of {len(regen['site_role'])}")
        if tier == "c":
            rows = pd.read_parquet(paths.processed("fuse_tier_c") / "rows.parquet", columns=["site_id"])
            rows = pd.read_parquet(paths.processed("fuse_tier_c") / "rows.parquet", columns=["site_id", "y_gt8"])
            missing = set(rows.loc[rows["y_gt8"].notna(), "site_id"]) - set(fz["site_role"])
            print(f"  labelled Tier C sites without a frozen role: {len(missing)} (must be 0; unroled sites are "
                  f"dropped by drop_protected)")
            return frozen_ok and not missing
    return frozen_ok and same


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["check", "write"])
    ap.add_argument("--tier", choices=["d", "c", "wp1", "all"], default="all")
    a = ap.parse_args(argv)
    tiers = ["d", "c", "wp1"] if a.tier == "all" else [a.tier]
    ok = True
    for t in tiers:
        if a.mode == "write":
            print(t, sp.save_split(generate(t), FROZEN[t][0]))   # raises if the file exists
        else:
            ok &= check(t)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
