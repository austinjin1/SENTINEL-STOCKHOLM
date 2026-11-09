# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Frozen SENTINEL-Fuse splits (docs/preregistration/fuse_protocol.md §4).

Tier D (NEON): whole NEON *domains* are held out, which implies leave-site-out and keeps the one
flow-connected chain (MAYF → BLWA → TOMB, all in D08) together. Test domains are chosen by a
salted-sha1 order with composition caps (row counts / labels never inspected); the remaining
domains form 5 development folds (leave-domain-group-out CV, balanced by site count).

Tier C (cyanotoxins): (1) Test-R = whole HUC2 regions (hash order, row-count rule only);
(2) Test-S = 20-km spatial blocks of the remaining sites (hash order, ~15 % of rows), with every
non-test site within 5 km of a Test-S site moved to role ``buffer`` (never used); (3) dev CV folds
over 20-km blocks (primary), 5-km blocks and leave-HUC2-out (secondary). HUC codes are strings.

Every split file is written once, its sha256 recorded in the protocol, and verified on load.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ...core import paths
from ...eval.splits import assert_disjoint, nearest_distance_km, spatial_blocks

PREREG = paths.REPO_ROOT / "docs" / "preregistration"
TIER_D_SPLIT_V1 = PREREG / "fuse_tier_d_split.json"      # superseded (no lake/river in test), kept unchanged
TIER_D_SPLIT = PREREG / "fuse_tier_d_split_v2.json"
TIER_C_SPLIT = PREREG / "fuse_tier_c_split.json"
TIER_WP1_SPLIT = PREREG / "fuse_tier_wp1_split.json"   # harder track W-P1 (protocol §12)
M1_SPLIT = PREREG / "sonde_split.json"
M1_SPLIT_SHA = "fd4bb14f"                                # prefix check; full sha verified by fuse_tier_w
SALT_WP1 = "sentinel-fuse-tierWP1-v1"
SALT_D = "sentinel-fuse-tierD-v1"   # same salt in v2 (only the selection rule changed)
SALT_C = "sentinel-fuse-tierC-v1"


def _h(salt: str, key: str) -> str:
    return hashlib.sha1(f"{salt}|{key}".encode()).hexdigest()


def file_sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# --------------------------------------------------------------------------- Tier D
def make_tier_d_split(sites: pd.DataFrame, *, min_test_sites: int = 7, max_test_sites: int = 8,
                      max_test_lakes: int = 2, max_test_rivers: int = 1, min_test_lakes: int = 1,
                      min_test_rivers: int = 1, n_folds: int = 5, version: str = "fuse_tier_d_v2") -> dict:
    """``sites``: columns site, domain, aquatic_type (from processed('neon_sites')).

    v2 rule (composition only, no outcome data): walk domains in sha1(salt|domain) order and
    (1) take the first domain containing a non-wadeable river that fits the caps, (2) if no lake yet,
    the first domain containing a lake that fits, (3) fill with further fitting domains in the same
    order until >= ``min_test_sites``. Caps: <= max sites / lakes / rivers.
    """
    s = sites[["site", "domain", "aquatic_type"]].copy()
    s["kind"] = s["aquatic_type"].map({"Aquatic Wadeable Stream": "stream", "Aquatic Lake": "lake",
                                       "Aquatic Non-wadeable Stream": "river"})
    doms = sorted(s["domain"].unique(), key=lambda d: _h(SALT_D, d))
    comp = {d: (len(g), int((g.kind == "lake").sum()), int((g.kind == "river").sum()))
            for d, g in s.groupby("domain")}
    test, n_t, n_l, n_r = [], 0, 0, 0

    def fits(d):
        n, nl, nr = comp[d]
        return n_t + n <= max_test_sites and n_l + nl <= max_test_lakes and n_r + nr <= max_test_rivers

    def take(d):
        nonlocal n_t, n_l, n_r
        test.append(d)
        n, nl, nr = comp[d]
        n_t, n_l, n_r = n_t + n, n_l + nl, n_r + nr

    for need, idx in (("river", 2), ("lake", 1)):
        have = n_r if need == "river" else n_l
        if have >= (min_test_rivers if need == "river" else min_test_lakes):
            continue
        for d in doms:
            if d not in test and comp[d][idx] > 0 and fits(d):
                take(d)
                break
    for d in doms:
        if n_t >= min_test_sites:
            break
        if d not in test and fits(d):
            take(d)
    if n_l < min_test_lakes or n_r < min_test_rivers or n_t < min_test_sites:
        raise ValueError("no test set satisfies the composition rule")
    dev = [d for d in doms if d not in test]
    # balanced folds: domains largest-first (hash tie-break) to the fold with fewest sites
    size = s.groupby("domain").size()
    order = sorted(dev, key=lambda d: (-size[d], _h(SALT_D + "|fold", d)))
    load = np.zeros(n_folds, int)
    fold_of = {}
    for d in order:
        f = int(np.argmin(load))
        fold_of[d] = f
        load[f] += size[d]
    site_role = {r.site: ("test" if r.domain in test else "dev") for r in s.itertuples()}
    site_fold = {r.site: (fold_of[r.domain] if r.domain in fold_of else None) for r in s.itertuples()}
    assert_disjoint(test=[x for x, r in site_role.items() if r == "test"],
                    dev=[x for x, r in site_role.items() if r == "dev"])
    return {"version": version, "salt": SALT_D,
            "rule": ("domains in sha1(salt|domain) order; first fitting domain with a river, then (if needed) "
                     f"first fitting domain with a lake, then fill in order to >= {min_test_sites} test sites; caps "
                     f"sites <= {max_test_sites}, lakes <= {max_test_lakes}, rivers <= {max_test_rivers}; dev "
                     f"domains -> {n_folds} folds, largest first to the fold with fewest sites"),
            "supersedes": "fuse_tier_d_split.json (v1: hash-order rule without lake/river minimums gave an "
                          "all-stream test set; replaced before any model was fit)",
            "test_domains": test, "dev_domains": dev, "fold_of_domain": fold_of,
            "site_role": site_role, "site_fold": site_fold,
            "counts": {"test_sites": n_t, "test_lakes": n_l, "test_rivers": n_r,
                       "dev_sites": int(sum(r == "dev" for r in site_role.values())),
                       "fold_sites": load.tolist()}}


# --------------------------------------------------------------------------- Tier C
def make_tier_c_split(rows: pd.DataFrame, *, region_frac=(0.08, 0.15), max_region_frac: float = 0.2,
                      spatial_frac: float = 0.15, block_km: float = 20.0, buffer_km: float = 5.0,
                      n_folds: int = 5) -> dict:
    """``rows``: one row per labelled site-day with site_id, lat, lon, huc2 (string)."""
    r = rows[["site_id", "lat", "lon", "huc2"]].copy()
    r["huc2"] = r["huc2"].astype("string")
    sites = r.groupby("site_id").agg(lat=("lat", "first"), lon=("lon", "first"), huc2=("huc2", "first"),
                                     n=("lat", "size")).reset_index()
    total = int(sites["n"].sum())
    # ---- Test-R: whole HUC2 regions (row counts only)
    reg = sites.groupby("huc2", dropna=False)["n"].sum()
    reg = reg[reg.index.notna()]
    regs = sorted(reg.index, key=lambda h: _h(SALT_C, f"huc2:{h}"))
    test_r, cum = [], 0
    for h in regs:
        if reg[h] / total > max_region_frac:
            continue
        if (cum + reg[h]) / total <= region_frac[1]:
            test_r.append(str(h))
            cum += int(reg[h])
        if cum / total >= region_frac[0]:
            break
    role = pd.Series("dev", index=sites["site_id"])
    role[sites.set_index("site_id")["huc2"].isin(test_r).to_numpy()] = "test_region"
    role[sites.set_index("site_id")["huc2"].isna().to_numpy()] = "dev"
    # ---- Test-S: 20-km blocks among non-region sites
    rest = sites[role.to_numpy() == "dev"].copy()
    rest["block20"] = spatial_blocks(rest["lat"], rest["lon"], block_km)
    bw = rest.groupby("block20")["n"].sum()
    blocks = sorted(bw.index, key=lambda b: _h(SALT_C, f"b20:{b}"))
    tgt = spatial_frac * rest["n"].sum()
    test_b, cum = [], 0
    for b in blocks:
        if cum >= tgt:
            break
        test_b.append(b)
        cum += int(bw[b])
    ts = rest["block20"].isin(test_b)
    role[rest.loc[ts, "site_id"].to_numpy()] = "test_spatial"
    # ---- buffer: non-test sites within buffer_km of any Test-S site
    tsx = sites.set_index("site_id").loc[role[role == "test_spatial"].index]
    cand = sites.set_index("site_id").loc[role[role == "dev"].index]
    d = nearest_distance_km(cand["lat"], cand["lon"], tsx["lat"], tsx["lon"])
    role[cand.index[d <= buffer_km]] = "buffer"
    # Test-R sites near dev sites are fine (region holdout); dev sites near Test-R within buffer -> buffer too
    trx = sites.set_index("site_id").loc[role[role == "test_region"].index]
    cand = sites.set_index("site_id").loc[role[role == "dev"].index]
    if len(trx):
        d2 = nearest_distance_km(cand["lat"], cand["lon"], trx["lat"], trx["lon"])
        role[cand.index[d2 <= buffer_km]] = "buffer"
    # ---- dev CV folds
    dev = sites.set_index("site_id").loc[role[role == "dev"].index].reset_index()
    folds = {}
    for name, km in (("block20", 20.0), ("block5", 5.0)):
        b = spatial_blocks(dev["lat"], dev["lon"], km)
        bw = pd.Series(dev["n"].to_numpy(), index=b).groupby(level=0).sum()
        order = sorted(bw.index, key=lambda x: (-bw[x], _h(SALT_C, f"{name}:{x}")))
        load = np.zeros(n_folds)
        fob = {}
        for x in order:
            f = int(np.argmin(load))
            fob[x] = f
            load[f] += bw[x]
        folds[name] = {s: int(fob[bb]) for s, bb in zip(dev["site_id"], b)}
    hs = sorted(dev["huc2"].dropna().unique(), key=lambda h: _h(SALT_C, f"lro:{h}"))
    folds["huc2"] = {s: str(h) for s, h in zip(dev["site_id"], dev["huc2"])}
    assert_disjoint(dev=role[role == "dev"].index, test_s=role[role == "test_spatial"].index,
                    test_r=role[role == "test_region"].index, buffer=role[role == "buffer"].index)
    cnt = {k: {"sites": int((role == k).sum()),
               "rows": int(sites.set_index("site_id").loc[role[role == k].index, "n"].sum())}
           for k in ("dev", "test_spatial", "test_region", "buffer")}
    return {"version": "fuse_tier_c_v1", "salt": SALT_C,
            "rule": (f"Test-R: HUC2 regions in sha1 order (skip regions > {max_region_frac:.0%} of rows) until "
                     f"{region_frac[0]:.0%}-{region_frac[1]:.0%} of rows; Test-S: {block_km:.0f}-km blocks of the "
                     f"rest in sha1 order until {spatial_frac:.0%} of their rows; non-test sites within "
                     f"{buffer_km:.0f} km of any test site -> buffer (unused); dev folds: {n_folds} folds over "
                     "20-km and 5-km blocks (largest first to the lightest fold) + leave-HUC2-out"),
            "test_regions_huc2": test_r, "n_test_blocks": len(test_b),
            "site_role": role.to_dict(), "dev_folds": folds, "huc2_order": hs, "counts": cnt}


# --------------------------------------------------------------------------- Tier W-P1 (harder track)
def make_tier_wp1_split(m1: dict, m1_sha: str, n_folds: int = 5) -> dict:
    """W-P1 track (protocol §12): site roles copied from the frozen M1 split — M1 ``train``/``val``
    sites → ``dev``, M1 ``test`` sites → ``test`` (never trained on). Dev CV folds = whole M1 basin
    groups (HUC4-based, flow-connected stations together) in salted-sha1 order, largest first to the
    fold with the fewest sites. Time windows: dev issue dates WY2009–WY2023 with target dates
    ≤ 2023-09-30; test issue dates 2023-11-12 … 2025-09-30 − h (42-d embargo after the boundary),
    i.e. unseen sites × unseen years (Test-W-A); test-role sites × dev years = Test-W-S (secondary)."""
    role = {s: ("test" if r == "test" else "dev") for s, r in m1["site_role"].items()}
    bg = m1["basin_group"]
    dev_bg = sorted({bg[s] for s, r in role.items() if r == "dev"})
    test_bg = {bg[s] for s, r in role.items() if r == "test"}
    assert not (set(dev_bg) & test_bg), "M1 basin groups span roles"
    size = pd.Series([bg[s] for s, r in role.items() if r == "dev"]).value_counts()
    order = sorted(dev_bg, key=lambda g: (-size[g], _h(SALT_WP1, g)))
    load = np.zeros(n_folds, int)
    fold_of_bg = {}
    for g in order:
        f = int(np.argmin(load))
        fold_of_bg[g] = f
        load[f] += size[g]
    site_fold = {s: (fold_of_bg[bg[s]] if r == "dev" else None) for s, r in role.items()}
    assert_disjoint(test=[s for s, r in role.items() if r == "test"], dev=[s for s, r in role.items() if r == "dev"])
    return {"version": "fuse_tier_wp1_v1", "salt": SALT_WP1, "source_split": "sonde_split.json", "source_sha256": m1_sha,
            "rule": ("roles copied from the M1 split (train/val -> dev, test -> test); dev folds = whole M1 basin groups, "
                     "salted-sha1 order, largest first to the fold with fewest sites"),
            "windows": {"dev_issue": ["2008-10-01", "2023-09-30"], "dev_target_max": "2023-09-30",
                        "test_a_issue": ["2023-11-12", "2025-09-30"], "test_a_target_max": "2025-09-30",
                        "embargo_days": 42},
            "site_role": role, "site_fold": site_fold, "fold_of_basin_group": fold_of_bg,
            "counts": {"dev_sites": int(sum(r == "dev" for r in role.values())),
                       "test_sites": int(sum(r == "test" for r in role.values())),
                       "dev_basin_groups": len(dev_bg), "test_basin_groups": len(test_bg),
                       "fold_sites": load.tolist()}}


# --------------------------------------------------------------------------- IO
def save_split(obj: dict, path: Path) -> str:
    if path.exists():
        raise FileExistsError(f"{path} exists — frozen splits are never overwritten")
    path.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str))
    return file_sha256(path)


def load_split(path: Path, expected_sha: str | None = None) -> dict:
    if expected_sha is not None:
        got = file_sha256(path)
        if got != expected_sha:
            raise ValueError(f"split {path.name} sha256 {got} != frozen {expected_sha}")
    return json.loads(Path(path).read_text())
