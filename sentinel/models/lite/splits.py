# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Frozen SENTINEL-Lite split (protocol §3): whole water bodies in ≥ 25-km spatial blocks + a year cut.

Roles per chip:
  * ``train``      — train-block water bodies, NAIP year ≤ ``last_train_year``
  * ``val``        — val-block water bodies, NAIP year ≤ ``last_train_year`` (model selection only)
  * ``test``       — test-block water bodies, all years (Test-A, headline: unseen water bodies)
  * ``test_time``  — train/val-block water bodies, NAIP year > ``last_train_year`` (Test-B: future cycle)
  * ``excluded``   — train/val water bodies with a site ≤ ``buffer_km`` from a test site (buffer), and
                     train water bodies ≤ ``buffer_km`` from a val site (keeps validation honest)

Blocks are assigned to roles by :func:`sentinel.eval.splits.group_split` weighted by chip counts;
a water body belongs to the block of its site-centroid. The split JSON is hashed (sha256 over the
canonical JSON); the hash is written into the protocol before any model selection.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from ...eval.splits import assert_disjoint, group_split, nearest_distance_km, spatial_blocks

ROLES = ("train", "val", "test", "test_time", "excluded")
PROTECTED_ROLES = ("test", "test_time")
DEFAULTS = {"block_km": 25.0, "test_frac": 0.20, "val_frac": 0.25, "seed": 20261024, "buffer_km": 2.0,
            "last_train_year": 2021, "water_body_radius_km": 2.0, "water_body_name_km": 40.0,
            "buffer_val_km": 2.0}
LABEL_HASH_COLS = ("naip_chip_id", "water_body", "site_id", "naip_year", "y_secchi", "y_turbidity", "y_tss", "y_chla")


def labels_sha256(lab: pd.DataFrame) -> str:
    """Content hash of the label table (review W6): chip ids, groups and log10 labels (rounded to 1e-9)."""
    d = lab[list(LABEL_HASH_COLS)].sort_values("naip_chip_id").copy()
    for c in ("y_secchi", "y_turbidity", "y_tss", "y_chla"):
        d[c] = d[c].round(9)
    return hashlib.sha256(d.to_csv(index=False, na_rep="NA").encode()).hexdigest()


def build_split(lab: pd.DataFrame, **kw) -> dict:
    p = {**DEFAULTS, **kw}
    wb = lab.groupby("water_body").agg(lat=("lat", "mean"), lon=("lon", "mean"), n=("naip_chip_id", "size"))
    wb["block"] = spatial_blocks(wb["lat"], wb["lon"], p["block_km"])
    bw = wb.groupby("block")["n"].sum().to_dict()
    bs = group_split(list(bw), test_frac=p["test_frac"], val_frac=p["val_frac"], seed=p["seed"], weights=bw)
    brole = {b: r for r, bl in (("train", bs.train), ("val", bs.val), ("test", bs.test)) for b in bl}
    wb["space_role"] = wb["block"].map(brole)
    # buffers (site level, whole water bodies excluded)
    site_xy = lab.drop_duplicates("site_id")[["site_id", "lat", "lon", "water_body"]]
    site_xy = site_xy.assign(space_role=site_xy["water_body"].map(wb["space_role"]))

    def near(src_role: str, ref_role: str, km: float | None = None) -> set:
        a = site_xy[site_xy["space_role"] == src_role]
        r = site_xy[site_xy["space_role"] == ref_role]
        if a.empty or r.empty:
            return set()
        d = nearest_distance_km(a["lat"], a["lon"], r["lat"], r["lon"])
        return set(a.loc[d <= (p["buffer_km"] if km is None else km), "water_body"])

    excl = near("train", "test") | near("val", "test") | near("train", "val", p.get("buffer_val_km"))
    wb["excluded"] = wb.index.isin(sorted(excl))
    role_of_wb = wb["space_role"].where(~wb["excluded"], "excluded")
    yr = lab["naip_year"].astype(int)
    sr = lab["water_body"].map(role_of_wb)
    role = np.where(sr == "test", "test",
                    np.where(sr == "excluded", "excluded",
                             np.where(yr > p["last_train_year"], "test_time", sr)))
    chips = pd.DataFrame({"naip_chip_id": lab["naip_chip_id"].astype(str), "role": role,
                          "water_body": lab["water_body"].astype(str)}).sort_values("naip_chip_id")
    split = {"version": 2, "params": p, "labels_sha256": labels_sha256(lab),
             "water_body_role": {k: str(v) for k, v in sorted(role_of_wb.items())},
             "block_role": {k: v for k, v in sorted(brole.items())},
             "chip_role": dict(zip(chips["naip_chip_id"], chips["role"]))}
    check_split(split, lab)
    return split


def check_split(split: dict, lab: pd.DataFrame) -> None:
    """Leakage checks (R4): labels unchanged; water bodies disjoint across train/val/test; years respected."""
    if split.get("labels_sha256") and split["labels_sha256"] != labels_sha256(lab):
        raise AssertionError("label table differs from the one the split was frozen with (labels_sha256)")
    cr = pd.Series(split["chip_role"])
    l2 = lab.set_index(lab["naip_chip_id"].astype(str))
    r = cr.reindex(l2.index)
    if r.isna().any():
        raise AssertionError(f"{int(r.isna().sum())} chips have no role")
    wbs = {k: set(l2.loc[r == k, "water_body"]) for k in ("train", "val", "test")}
    assert_disjoint(**wbs)
    assert_disjoint(train=set(l2.loc[r == "train", "site_id"]), test=set(l2.loc[r == "test", "site_id"]),
                    val=set(l2.loc[r == "val", "site_id"]))
    ly = split["params"]["last_train_year"]
    if (l2.loc[r.isin(["train", "val"]), "naip_year"] > ly).any():
        raise AssertionError("train/val chips after last_train_year")
    # train-to-test distance must exceed the buffer
    tr = l2[r == "train"].drop_duplicates("site_id")
    te = l2[r == "test"].drop_duplicates("site_id")
    va = l2[r == "val"].drop_duplicates("site_id")
    buf = split["params"]["buffer_km"]
    bv = split["params"].get("buffer_val_km")
    bv = buf if bv is None else bv
    for a, b, name, lim in ((tr, te, "train->test", buf), (va, te, "val->test", buf), (tr, va, "train->val", bv)):
        if len(a) and len(b):
            d = nearest_distance_km(a["lat"], a["lon"], b["lat"], b["lon"])
            if d.min() <= lim:
                raise AssertionError(f"{name} min distance {d.min():.2f} km <= buffer {lim} km")


def split_sha256(split: dict) -> str:
    blob = json.dumps({k: split.get(k) for k in ("version", "params", "labels_sha256", "water_body_role", "chip_role")},
                      sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


def split_report(split: dict, lab: pd.DataFrame) -> dict:
    from ...eval.splits import nearest_distance_km as nd

    l2 = lab.assign(role=lab["naip_chip_id"].astype(str).map(split["chip_role"]))
    rep = {"sha256": split_sha256(split), "params": split["params"], "n_chips": {}, "n_labels": {},
           "n_water_bodies": {}, "n_lake_labels": {}}
    for r in ROLES:
        s = l2[l2["role"] == r]
        rep["n_chips"][r] = int(len(s))
        rep["n_water_bodies"][r] = int(s["water_body"].nunique())
        rep["n_labels"][r] = {t: int(s[f"y_{t}"].notna().sum()) for t in ("secchi", "turbidity", "tss", "chla")}
        lk = s[s["site_class"] == "lake_reservoir"]
        rep["n_lake_labels"][r] = {t: int(lk[f"y_{t}"].notna().sum()) for t in ("secchi", "turbidity", "tss", "chla")}
    tr = l2[l2["role"] == "train"].drop_duplicates("site_id")
    for r in ("val", "test"):
        s = l2[l2["role"] == r].drop_duplicates("site_id")
        if len(s) and len(tr):
            d = nd(s["lat"], s["lon"], tr["lat"], tr["lon"])
            rep[f"{r}_to_train_km"] = {"min": float(d.min()), "median": float(np.median(d))}
    return rep


def roles_for(split: dict, chip_ids) -> np.ndarray:
    return np.array([split["chip_role"].get(str(c), "missing") for c in chip_ids])
