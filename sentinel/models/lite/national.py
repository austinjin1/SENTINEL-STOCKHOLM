# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""SENTINEL-Lite **national track** (RESEARCH_PLAN §15.5 amendment 2026-11-08, items 5–7; protocol §12).

An additional, larger benchmark built from the national NAIP × WQP matchup build
(``build_naip_matchups.py --national``). It never replaces the original 4-state track, whose frozen split
(``lite_split.json``) and labels (``lite_labels/labels.parquet``) are untouched.

Construction (all decided before any model output on the new data is seen):
1. **Frozen chips keep everything.** Every chip of the frozen 4-state split keeps its role and its frozen label
   row (water body ids prefixed ``F:``). Frozen ``test``/``test_time`` stay protected.
2. **New chips** (not in the frozen split) get labels by the same rules (``data.build_label_table``) and water
   bodies by the same proxy over the union of all national sites.
   - A new chip whose national water body contains a frozen chip inherits: ``train`` if every frozen chip of that
     water body is ``train`` (and its year is ≤ 2021), else ``excluded`` (never new test rows in old water bodies).
   - Other new water bodies: the 25-km block of their centroid; blocks that exist in the frozen split keep their
     frozen block role; new blocks are randomly assigned (test 20%, val 25%, chip-weighted, seed 20261108).
     Roles: ``test_n`` (Test-N, all years; headline), ``val_n``, ``train``; year cut as the original track
     (train/val years ≤ 2021; ``test_time_n`` = 2022–2023 at train/val_n water bodies).
   - Buffers (site level, whole water bodies excluded): new train/val_n water bodies with a site ≤ 2 km from any
     test or test_n site; new train water bodies ≤ 2 km from a val or val_n site.
3. Protected roles: ``test``, ``test_time``, ``test_n``, ``test_time_n``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ...core import paths
from ...eval.splits import assert_disjoint, group_split, nearest_distance_km, spatial_blocks

NATIONAL_SPLIT_FILE = paths.REPO_ROOT / "docs" / "preregistration" / "lite_split_national.json"
NATIONAL_PROTECTED = ("test", "test_time", "test_n", "test_time_n")
PARAMS = {"block_km": 25.0, "test_frac": 0.20, "val_frac": 0.25, "seed": 20261108, "buffer_km": 2.0,
          "last_train_year": 2021, "water_body_radius_km": 2.0, "water_body_name_km": 40.0}


NATIONAL_DIR = "national"      # processed/naip_matchups/national/ (the canonical 4-state files stay in place)


def national_matchups_path() -> Path:
    return paths.processed("naip_matchups") / NATIONAL_DIR / "matchups.parquet"


def national_scene_stats_path() -> Path:
    return paths.processed("naip_matchups") / NATIONAL_DIR / "naip_scene_stats.parquet"


def labels_path() -> Path:
    return paths.processed("lite_labels") / "labels_national.parquet"


def build_labels(frozen_split: dict) -> tuple[pd.DataFrame, dict]:
    """Union label table: frozen rows (``source = frozen``) + new chips from the national matchups."""
    from . import data as D

    m = pd.read_parquet(national_matchups_path())
    new_lab, rep = D.build_label_table(m, radius_km=PARAMS["water_body_radius_km"],
                                       name_km=PARAMS["water_body_name_km"], use_hydrolakes=True)
    frozen = D.load_labels()
    fz_ids = set(frozen["naip_chip_id"].astype(str))
    new_lab["naip_chip_id"] = new_lab["naip_chip_id"].astype(str)
    # national water body of every frozen chip (via its site) -> inheritance rule
    nat_wb = new_lab.drop_duplicates("site_id").set_index("site_id")["water_body"]
    frozen = frozen.assign(source="frozen", national_wb=frozen["site_id"].map(nat_wb),
                           water_body="F:" + frozen["water_body"].astype(str))
    add = new_lab[~new_lab["naip_chip_id"].isin(fz_ids)].copy()
    add["source"] = "new"
    add["national_wb"] = add["water_body"]
    add["water_body"] = "N:" + add["water_body"].astype(str)
    cols = [c for c in frozen.columns if c in add.columns]
    lab = pd.concat([frozen[cols], add[cols]], ignore_index=True)
    rep = {"national_label_report": rep, "n_frozen_chips": int(len(frozen)), "n_new_chips": int(len(add)),
           "n_new_chips_in_frozen_set_dropped": int(new_lab["naip_chip_id"].isin(fz_ids).sum()),
           "frozen_split_sha256": frozen_split["_sha256"]}
    return lab, rep


def labels_sha256(lab: pd.DataFrame) -> str:
    from .splits import LABEL_HASH_COLS

    d = lab[list(LABEL_HASH_COLS)].sort_values("naip_chip_id").copy()
    for c in ("y_secchi", "y_turbidity", "y_tss", "y_chla"):
        d[c] = d[c].round(9)
    return hashlib.sha256(d.to_csv(index=False, na_rep="NA").encode()).hexdigest()


def build_split(lab: pd.DataFrame, frozen_split: dict, **kw) -> dict:
    p = {**PARAMS, **kw}
    lab = lab.copy()
    lab["naip_chip_id"] = lab["naip_chip_id"].astype(str)
    role = lab["naip_chip_id"].map(frozen_split["chip_role"])
    is_new = role.isna().to_numpy()
    fr = lab[~is_new].assign(role=role[~is_new])
    # rule 2a: new chips in national water bodies that contain frozen chips
    fz_roles = fr.groupby("national_wb")["role"].agg(lambda s: set(s))
    yr = lab["naip_year"].astype(int).to_numpy()
    new = lab[is_new]
    touched = new["national_wb"].isin(fz_roles.index).to_numpy()
    all_train = new["national_wb"].map(lambda w: fz_roles.get(w, set()) == {"train"}).to_numpy(bool)
    out_role = role.copy().astype(object)
    idx_new = np.flatnonzero(is_new)
    out_role.iloc[idx_new[touched]] = np.where(all_train[touched] & (yr[idx_new[touched]] <= p["last_train_year"]),
                                              "train", "excluded")
    # rule 2b: untouched new water bodies -> blocks
    fresh = new[~touched]
    wb = fresh.groupby("water_body").agg(lat=("lat", "mean"), lon=("lon", "mean"), n=("naip_chip_id", "size"))
    wb["block"] = spatial_blocks(wb["lat"], wb["lon"], p["block_km"])
    fb = frozen_split["block_role"]
    new_blocks = sorted(set(wb["block"]) - set(fb))
    bw = wb[wb["block"].isin(new_blocks)].groupby("block")["n"].sum().to_dict()
    bs = group_split(new_blocks, test_frac=p["test_frac"], val_frac=p["val_frac"], seed=p["seed"], weights=bw)
    brole = {**{b: r for r, bl in (("train", bs.train), ("val", bs.val), ("test", bs.test)) for b in bl}, **fb}
    wb["space_role"] = wb["block"].map(brole).map({"train": "train", "val": "val_n", "test": "test_n"})
    # buffers against every test-like / val-like site (frozen and new)
    sites = lab.drop_duplicates("site_id")[["site_id", "lat", "lon", "water_body"]]
    site_role = pd.concat([fr.drop_duplicates("site_id").set_index("site_id")["role"],
                           fresh.drop_duplicates("site_id").set_index("site_id")["water_body"].map(wb["space_role"])])
    sites = sites.assign(r=sites["site_id"].map(site_role))

    def near(src_wbs: set, ref_roles: tuple) -> set:
        a = sites[sites["water_body"].isin(src_wbs)]
        r = sites[sites["r"].isin(ref_roles)]
        if a.empty or r.empty:
            return set()
        d = nearest_distance_km(a["lat"], a["lon"], r["lat"], r["lon"])
        return set(a.loc[d <= p["buffer_km"], "water_body"])

    tr_w = set(wb.index[wb["space_role"] == "train"])
    va_w = set(wb.index[wb["space_role"] == "val_n"])
    excl = near(tr_w | va_w, ("test", "test_n")) | near(tr_w, ("val", "val_n"))
    # also the inherited new train chips: their sites must keep the buffer to test / test_n sites
    inh = lab.iloc[idx_new[touched]]
    inh_train = inh[out_role.iloc[idx_new[touched]].to_numpy() == "train"]
    bad_inh = near(set(inh_train["water_body"]), ("test", "test_n", "val", "val_n")) if len(inh_train) else set()
    wb["role"] = np.where(wb.index.isin(sorted(excl)), "excluded", wb["space_role"])
    fr_role = fresh["water_body"].map(wb["role"]).to_numpy()
    fyr = fresh["naip_year"].astype(int).to_numpy()
    fr_role = np.where(np.isin(fr_role, ["train", "val_n"]) & (fyr > p["last_train_year"]), "test_time_n", fr_role)
    out_role.loc[fresh.index] = fr_role
    if bad_inh:
        m = lab["water_body"].isin(bad_inh).to_numpy() & is_new & (out_role.to_numpy() == "train")
        out_role[m] = "excluded"
    split = {"version": "national-1", "params": p, "frozen_split_sha256": frozen_split["_sha256"],
             "labels_sha256": labels_sha256(lab), "block_role_new": {b: brole[b] for b in new_blocks},
             "chip_role": dict(zip(lab["naip_chip_id"], out_role.astype(str)))}
    check_split(split, lab, frozen_split)
    return split


def split_sha256(split: dict) -> str:
    blob = json.dumps({k: split.get(k) for k in ("version", "params", "frozen_split_sha256", "labels_sha256",
                                                 "chip_role")}, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


def check_split(split: dict, lab: pd.DataFrame, frozen_split: dict) -> None:
    """Frozen roles unchanged; water bodies and sites disjoint between training and every evaluation role; years;
    2-km buffers from train to every test-like and val-like site."""
    r = lab["naip_chip_id"].astype(str).map(split["chip_role"])
    if r.isna().any():
        raise AssertionError("chips without a national role")
    fz = pd.Series(frozen_split["chip_role"])
    if not (r[lab["naip_chip_id"].astype(str).isin(fz.index)].to_numpy()
            == fz.reindex(lab["naip_chip_id"].astype(str)).dropna().to_numpy()).all():
        raise AssertionError("a frozen chip changed role")
    l2 = lab.assign(role=r.to_numpy())
    groups = {"train": ("train",), "val": ("val", "val_n"), "test": ("test",), "test_n": ("test_n",)}
    for key in ("water_body", "site_id"):
        assert_disjoint(**{g: set(l2.loc[l2.role.isin(rs), key].dropna()) for g, rs in groups.items()})
    # national water bodies: a frozen-frozen merge cannot be undone (frozen roles are fixed), but no NEW chip may
    # sit in a national water body shared by training and an evaluation role
    nw = {g: set(l2.loc[l2.role.isin(rs), "national_wb"].dropna()) for g, rs in groups.items()}
    new_nw = set(l2.loc[(l2["source"] == "new") & l2.role.isin(sum(groups.values(), ())), "national_wb"])
    for g in ("val", "test", "test_n"):
        bad = nw["train"] & nw[g] & new_nw
        if bad:
            raise AssertionError(f"LEAK: {len(bad)} national water bodies with new chips in both train and {g}")
    if (l2.loc[l2.role.isin(["train", "val", "val_n"]), "naip_year"].astype(int) > split["params"]["last_train_year"]).any():
        raise AssertionError("train/val chips after last_train_year")
    tr = l2[l2.role == "train"].drop_duplicates("site_id")
    for ref in (("test", "test_n"), ("val", "val_n")):
        b = l2[l2.role.isin(ref)].drop_duplicates("site_id")
        new_tr = tr[tr["source"] == "new"]
        if len(new_tr) and len(b):
            d = nearest_distance_km(new_tr["lat"], new_tr["lon"], b["lat"], b["lon"])
            if d.min() <= split["params"]["buffer_km"]:
                raise AssertionError(f"new train site {d.min():.2f} km from a {ref} site")


def split_report(split: dict, lab: pd.DataFrame) -> dict:
    l2 = lab.assign(role=lab["naip_chip_id"].astype(str).map(split["chip_role"]))
    rep = {"sha256": split_sha256(split), "params": split["params"], "by_role": {}}
    for role, s in l2.groupby("role"):
        rep["by_role"][role] = {"chips": int(len(s)), "water_bodies": int(s["water_body"].nunique()),
                                "states": int(s["state"].nunique()),
                                "labels": {t: int(s[f"y_{t}"].notna().sum()) for t in ("secchi", "turbidity", "tss", "chla")},
                                "lake_labels": {t: int(s.loc[s.site_class == "lake_reservoir", f"y_{t}"].notna().sum())
                                                for t in ("secchi", "turbidity", "tss", "chla")},
                                "label_water_bodies": {t: int(s.loc[s[f"y_{t}"].notna(), "water_body"].nunique())
                                                       for t in ("secchi", "turbidity", "tss", "chla")},
                                "new_chips": int((s["source"] == "new").sum())}
    return rep


def load_split(path: Path = NATIONAL_SPLIT_FILE, check_frozen: bool = True) -> dict:
    split = json.loads(Path(path).read_text())
    sha = split_sha256(split)
    if check_frozen:
        from .guard import FROZEN_NATIONAL_SPLIT_SHA256, ProtectedDataError

        if FROZEN_NATIONAL_SPLIT_SHA256 != "PENDING" and sha != FROZEN_NATIONAL_SPLIT_SHA256:
            raise ProtectedDataError(f"national split sha256 {sha[:12]} != frozen {FROZEN_NATIONAL_SPLIT_SHA256[:12]}")
    split["_sha256"] = sha
    split["_path"] = str(path)
    split["_track"] = "national"
    return split


def role_frame(split: dict, roles, *, training: bool = True) -> pd.DataFrame:
    """National label rows for ``roles``; protected national roles never reach training/selection code."""
    from .guard import ProtectedDataError

    bad = sorted({str(r) for r in roles} & set(NATIONAL_PROTECTED))
    if bad:
        raise ProtectedDataError(f"national track: protected roles {bad} are not loaded in this phase")
    lab = pd.read_parquet(labels_path())
    if labels_sha256(lab) != split["labels_sha256"]:
        raise ProtectedDataError("labels_national.parquet differs from the frozen national split's labels_sha256")
    lab["role"] = lab["naip_chip_id"].astype(str).map(split["chip_role"])
    return lab[lab["role"].isin(list(roles))].reset_index(drop=True)


def train_val_frames(track: str):
    """(split, train frame, val frame, scene-stats path, experiment) for ``track`` ∈ {original, national}.
    National val = frozen ``val`` ∪ ``val_n`` (roles kept in the ``role`` column for by-subset reporting)."""
    from . import data as D
    from . import guard as G

    if track == "original":
        split = G.load_split()
        lab = D.role_frame(split, ("train", "val"), training=True)
        stats, exp = None, "lite_val"
    elif track == "national":
        split = load_split()
        lab = role_frame(split, ("train", "val", "val_n"))
        stats, exp = national_scene_stats_path(), "lite_val_national"
    else:
        raise ValueError(f"unknown track {track!r}")
    G.assert_no_protected(lab["role"])
    bad = sorted(set(lab["role"]) & set(NATIONAL_PROTECTED))
    if bad:
        raise G.ProtectedDataError(f"protected roles {bad}")
    tr = lab[lab.role == "train"].reset_index(drop=True)
    va = lab[lab.role.isin(["val", "val_n"])].reset_index(drop=True)
    return split, tr, va, stats, exp
