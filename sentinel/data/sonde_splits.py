# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Frozen spatial/temporal split for SENTINEL-Sonde and EW-v2 (RESEARCH_PLAN.md §14).

Spatial roles are assigned to *basin groups*: HUC4 units merged (union-find) whenever a
station in one is flow-connected to a station in another (edges from NLDI navigation), so
upstream/downstream neighbours can never sit on opposite sides of the split. Groups are then
split ~70/10/20 (train/val/test) by number of sites with a fixed seed. Time blocks are fixed
water years with a 42-day embargo.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..eval.splits import assert_disjoint

TIME_BLOCKS = {
    "train": ("2007-10-01", "2021-09-30"),
    "val": ("2021-10-01", "2023-09-30"),
    "test": ("2023-10-01", "2025-09-30"),
    "test_b": ("2025-10-01", "2026-09-30"),
}
EMBARGO_DAYS = 42


class _UnionFind:
    def __init__(self, items):
        self.p = {i: i for i in items}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def basin_groups(sites: pd.DataFrame, edges: pd.DataFrame | None = None) -> pd.Series:
    """Map site_no -> basin group id (HUC4 merged across flow-connected stations)."""
    huc4 = sites.set_index("site_no")["huc_cd"].astype(str).str.zfill(8).str[:4]
    huc4 = huc4.where(huc4.str.fullmatch(r"\d{4}"), "unk_" + huc4.index.to_series())
    uf = _UnionFind(sorted(set(huc4)))
    if edges is not None and len(edges):
        known = set(huc4.index)
        for s, d in edges[["src", "dst"]].astype(str).itertuples(index=False):
            if s in known and d in known:
                uf.union(huc4[s], huc4[d])
    return huc4.map(uf.find).rename("basin_group")


@dataclass
class SondeSplit:
    site_role: dict
    basin_group: dict
    time_blocks: dict = field(default_factory=lambda: dict(TIME_BLOCKS))
    embargo_days: int = EMBARGO_DAYS
    seed: int = 0
    notes: str = ""

    def sites(self, role: str) -> list[str]:
        return sorted(s for s, r in self.site_role.items() if r == role)

    def time_mask(self, dates, block: str, *, horizon_days: int = 0, context_days: int = 0) -> np.ndarray:
        """Issue dates whose context AND forecast window stay inside ``block`` minus embargo.

        A sample issued at d uses inputs (d - context, d] and targets (d, d + horizon]; it is
        kept only if both lie in [start, end] and, for non-train blocks, the context starts
        after ``start`` (so no input overlaps the previous block) — the 42-day embargo
        additionally drops issue dates within 42 days after a block start.
        """
        d = pd.to_datetime(pd.Series(dates)).to_numpy()
        start, end = (np.datetime64(x) for x in self.time_blocks[block])
        emb = np.timedelta64(self.embargo_days, "D")
        lo = start + (emb if block != "train" else np.timedelta64(0, "D"))
        ok = (d - np.timedelta64(context_days, "D") >= start) & (d >= lo) \
            & (d + np.timedelta64(horizon_days, "D") <= end)
        return ok

    def to_json(self) -> str:
        return json.dumps({"site_role": self.site_role, "basin_group": self.basin_group,
                           "time_blocks": self.time_blocks, "embargo_days": self.embargo_days,
                           "seed": self.seed, "notes": self.notes}, sort_keys=True, indent=1)

    def sha256(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()

    @classmethod
    def load(cls, path: str | Path) -> "SondeSplit":
        d = json.loads(Path(path).read_text())
        return cls(d["site_role"], d["basin_group"], d["time_blocks"], d["embargo_days"], d["seed"],
                   d.get("notes", ""))


def make_split(sites: pd.DataFrame, edges: pd.DataFrame | None = None, *, seed: int = 20261022,
               fracs=(0.7, 0.1, 0.2), notes: str = "") -> SondeSplit:
    groups = basin_groups(sites, edges)
    size = groups.value_counts()
    rs = np.random.default_rng(seed)
    order = size.index.to_numpy()[rs.permutation(len(size))]
    cum = np.cumsum(size.loc[order].to_numpy()) / size.sum()
    role_of_group = {}
    for g, c in zip(order, cum):
        role_of_group[g] = "train" if c <= fracs[0] else ("val" if c <= fracs[0] + fracs[1] else "test")
    site_role = {s: role_of_group[g] for s, g in groups.items()}
    sp = SondeSplit(site_role, groups.to_dict(), seed=seed, notes=notes)
    assert_disjoint(train=sp.sites("train"), val=sp.sites("val"), test=sp.sites("test"))
    # No basin group may span roles (the point of grouping).
    roles = pd.DataFrame({"g": groups, "r": pd.Series(site_role)})
    assert (roles.groupby("g")["r"].nunique() == 1).all()
    return sp


def check_no_cross_role_edges(split: SondeSplit, edges: pd.DataFrame) -> int:
    """Number of flow-connected site pairs that straddle roles (must be 0)."""
    r = split.site_role
    bad = 0
    for s, d in edges[["src", "dst"]].astype(str).itertuples(index=False):
        if s in r and d in r and r[s] != r[d]:
            bad += 1
    return bad
