# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""SENTINEL-Lite national-2 split (lite_national_protocol.md N4/N5; SOTA lever 1(a), ±7-day Secchi window).

National-2 adds TRAINING rows only. Every national-1 chip keeps its role and its label row, so every evaluation
block (val, val_n, test, test_n, test_time, test_time_n) is byte-identical to national-1, with ±1-day labels.

A new chip (in the ±7 d build, ``processed/naip_matchups/national_pm7``, but not in national-1) is ``train`` only if
all of the following hold; otherwise it is ``excluded``:
1. its water body (national-2 grouping over the union of sites, same proxy rules) holds no national-1 chip whose
   role is not ``train``;
2. if the water body holds no national-1 chip at all, its 25-km block is a train block. Train blocks are the frozen
   4-state block roles and the national-1 new-block roles. Blocks seen in neither get the salted random rule with
   the national-1 parameters, and only a "train" outcome is kept;
3. its NAIP year is ≤ 2021;
4. each of its sites is > 2 km from every evaluation-role site.

Labels of new chips follow the same rules (``data.build_label_table``), with the Secchi window of the build. The
file name contains "national", so it is outside the 4-state scoring-code hash (``guard.NATIONAL_EXCLUDE``).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ...core import paths
from ...eval.splits import group_split, nearest_distance_km, spatial_blocks
from . import national as N

SPLIT_FILE = paths.REPO_ROOT / "docs" / "preregistration" / "lite_split_national2.json"
FROZEN_SHA256 = "976b5be7ed7c76e9417d526f41e7b6899f88e49e3dc00b7dba077bf78f40ebe2"   # pinned 2026-11-10, before any model saw new rows
EVAL_ROLES = ("val", "val_n", "test", "test_n", "test_time", "test_time_n")
PM7_DIR = "national_pm7"


def matchups_path() -> Path:
    return paths.processed("naip_matchups") / PM7_DIR / "matchups.parquet"


def labels_path() -> Path:
    return paths.processed("lite_labels") / "labels_national2.parquet"


def scene_stats_path() -> Path:
    """National-1 DOQQ stats ∪ the stats of DOQQs first read by the ±7 d build (one row per DOQQ item)."""
    return paths.processed("naip_matchups") / PM7_DIR / "naip_scene_stats_merged.parquet"


def merge_scene_stats() -> Path:
    a = pd.read_parquet(N.national_scene_stats_path())
    b = pd.read_parquet(paths.processed("naip_matchups") / PM7_DIR / "naip_scene_stats.parquet")
    key = [c for c in ("item_id", "naip_item_id") if c in a.columns][0]
    m = pd.concat([a, b[~b[key].isin(set(a[key]))]], ignore_index=True)
    m.to_parquet(scene_stats_path(), index=False)
    return scene_stats_path()


def build_labels() -> tuple[pd.DataFrame, dict]:
    from . import data as D

    n1 = pd.read_parquet(N.labels_path())
    m = pd.read_parquet(matchups_path())
    new_lab, rep = D.build_label_table(m, radius_km=N.PARAMS["water_body_radius_km"],
                                       name_km=N.PARAMS["water_body_name_km"], use_hydrolakes=True)
    new_lab["naip_chip_id"] = new_lab["naip_chip_id"].astype(str)
    wb2 = new_lab.drop_duplicates("site_id").set_index("site_id")["water_body"]
    n1 = n1.assign(source2="national1", national2_wb=n1["site_id"].map(wb2))
    add = new_lab[~new_lab["naip_chip_id"].isin(set(n1["naip_chip_id"].astype(str)))].copy()
    add["source2"] = "pm7_new"
    add["source"] = "new"
    add["national2_wb"] = add["water_body"]
    add["national_wb"] = add["water_body"]
    add["water_body"] = "N2:" + add["water_body"].astype(str)
    cols = [c for c in n1.columns if c in add.columns]
    lab = pd.concat([n1[cols], add[cols]], ignore_index=True)
    return lab, {"pm7_label_report": rep, "n_national1_chips": int(len(n1)), "n_new_chips": int(len(add))}


def build_split(lab: pd.DataFrame) -> dict:
    n1 = N.load_split()
    from .guard import load_split as load_original

    orig = load_original()
    p = dict(N.PARAMS)
    lab = lab.copy()
    lab["naip_chip_id"] = lab["naip_chip_id"].astype(str)
    role = lab["naip_chip_id"].map(n1["chip_role"])
    is_new = role.isna().to_numpy()
    old = lab[~is_new].assign(role=role[~is_new])
    new = lab[is_new]
    # rule 1: water bodies holding national-1 chips
    roles_in_wb = old.groupby("national2_wb")["role"].agg(lambda s: set(s))
    touched = new["national2_wb"].isin(roles_in_wb.index).to_numpy()
    wb_all_train = new["national2_wb"].map(lambda w: roles_in_wb.get(w, set()) <= {"train"}).to_numpy(bool)
    ok = np.where(touched, wb_all_train, True)
    # rule 2: untouched water bodies by block role
    fresh = new[~touched]
    wb = fresh.groupby("national2_wb").agg(lat=("lat", "mean"), lon=("lon", "mean"), n=("naip_chip_id", "size"))
    wb["block"] = spatial_blocks(wb["lat"], wb["lon"], p["block_km"])
    known = {**orig["block_role"], **n1.get("block_role_new", {})}
    unseen = sorted(set(wb["block"]) - set(known))
    bw = wb[wb["block"].isin(unseen)].groupby("block")["n"].sum().to_dict()
    bs = group_split(unseen, test_frac=p["test_frac"], val_frac=p["val_frac"], seed=p["seed"], weights=bw)
    brole = {**{b: r for r, bl in (("train", bs.train), ("val", bs.val), ("test", bs.test)) for b in bl}, **known}
    wb_train = (wb["block"].map(brole) == "train")
    ok_fresh = fresh["national2_wb"].map(wb_train).fillna(False).to_numpy(bool)
    ok[~touched] = ok_fresh
    # rule 3: years
    ok &= new["naip_year"].astype(int).to_numpy() <= p["last_train_year"]
    # rule 4: 2-km from every evaluation-role site
    ev = old[old["role"].isin(EVAL_ROLES)].drop_duplicates("site_id")
    d = nearest_distance_km(new["lat"], new["lon"], ev["lat"], ev["lon"])
    ok &= d > p["buffer_km"]
    new_role = np.where(ok, "train", "excluded")
    chip_role = dict(n1["chip_role"])
    chip_role.update(dict(zip(new["naip_chip_id"], new_role)))
    split = {"version": "national-2", "params": {**p, "secchi_window_days": 7, "new_rows": "train-only"},
             "national1_split_sha256": n1["_sha256"], "labels_sha256": N.labels_sha256(lab),
             "block_role_unseen": {b: brole[b] for b in unseen}, "chip_role": chip_role}
    check_split(split, lab, n1)
    return split


def check_split(split: dict, lab: pd.DataFrame, n1: dict) -> None:
    r = lab["naip_chip_id"].astype(str).map(split["chip_role"])
    if r.isna().any():
        raise AssertionError("chips without a national-2 role")
    n1r = pd.Series(n1["chip_role"])
    same = r[lab["naip_chip_id"].astype(str).isin(n1r.index)]
    if not (same.to_numpy() == n1r.reindex(lab.loc[same.index, "naip_chip_id"].astype(str)).to_numpy()).all():
        raise AssertionError("a national-1 chip changed role")
    new = lab[~lab["naip_chip_id"].astype(str).isin(n1r.index)].assign(role=r)
    if set(new["role"]) - {"train", "excluded"}:
        raise AssertionError("new chips may only be train or excluded")
    l2 = lab.assign(role=r.to_numpy())
    ev_wb = set(l2.loc[l2["role"].isin(EVAL_ROLES), "national2_wb"].dropna())
    bad = set(new.loc[new["role"] == "train", "national2_wb"]) & ev_wb
    if bad:
        raise AssertionError(f"LEAK: {len(bad)} new train water bodies hold evaluation chips")
    nt = new[new["role"] == "train"].drop_duplicates("site_id")
    ev = l2[l2["role"].isin(EVAL_ROLES)].drop_duplicates("site_id")
    if len(nt) and len(ev):
        dmin = nearest_distance_km(nt["lat"], nt["lon"], ev["lat"], ev["lon"]).min()
        if dmin <= split["params"]["buffer_km"]:
            raise AssertionError(f"new train site {dmin:.2f} km from an evaluation site")
    if (nt["naip_year"].astype(int) > split["params"]["last_train_year"]).any():
        raise AssertionError("new train chips after last_train_year")


def split_sha256(split: dict) -> str:
    blob = json.dumps({k: split.get(k) for k in ("version", "params", "national1_split_sha256", "labels_sha256",
                                                 "chip_role")}, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


def load_split(path: Path = SPLIT_FILE) -> dict:
    split = json.loads(Path(path).read_text())
    sha = split_sha256(split)
    if FROZEN_SHA256 != "PENDING" and sha != FROZEN_SHA256:
        from .guard import ProtectedDataError

        raise ProtectedDataError(f"national-2 split sha256 {sha[:12]} != frozen {FROZEN_SHA256[:12]}")
    split.update(_sha256=sha, _path=str(path), _track="national2")
    return split


def role_frame(split: dict, roles, *, training: bool = True) -> pd.DataFrame:
    from .guard import ProtectedDataError

    bad = sorted({str(x) for x in roles} & set(N.NATIONAL_PROTECTED))
    if bad:
        raise ProtectedDataError(f"national-2: protected roles {bad} are not loaded in this phase")
    lab = pd.read_parquet(labels_path())
    if N.labels_sha256(lab) != split["labels_sha256"]:
        raise ProtectedDataError("labels_national2.parquet differs from the national-2 split's labels_sha256")
    lab["role"] = lab["naip_chip_id"].astype(str).map(split["chip_role"])
    return lab[lab["role"].isin(list(roles))].reset_index(drop=True)


def activate() -> None:
    """Point the national-track entry points (``national.load_split``, ``role_frame``, ``labels_path``,
    ``national_scene_stats_path``) at national-2 for this process only. The hashed ``lite_*`` scripts then run
    unchanged on national-2: train/val frames, baselines, compare and teacher targets."""
    orig_tvf = N.train_val_frames
    N.load_split = lambda *a, **k: load_split()
    N.role_frame = role_frame
    N.labels_path = labels_path
    N.national_scene_stats_path = scene_stats_path

    def tvf(track: str):
        out = orig_tvf(track)
        if track == "national":     # separate registry experiment for national-2 runs
            return (*out[:4], "lite_val_national2")
        return out

    N.train_val_frames = tvf
