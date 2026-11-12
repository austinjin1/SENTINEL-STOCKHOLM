# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Checks of M7 architecture round v2 (M7_verify ruling Q5; lead order of 2026-11-10 20:20 EST, items A3 and A6).

* :func:`refit_overlap` / :func:`assert_refit_clean` — the table the anchor model is refitted on (the v2 training
  rows) holds no chip and no water body of a validation or test role (Q5 (i)).
* :func:`cross_track_conflicts` / :func:`assert_one_grouping` — no ``train_p2`` water body is a national-2b
  training water body under another id (Q5 (ii)); as built the two id sets come from two groupings, so the test is
  geometric: no ``train_p2`` row within the grouping radius of a national-2b training row, and no shared
  HydroLAKES id.
* :func:`e3a_turb_count` — the one-shot count step of event E3a_turb on ``val_n`` (A6).

Keys only. :func:`protected_keys` reads the columns chip id, water body, site, position and HydroLAKES id of the
validation and test roles, never a label column; nothing of it is printed beyond counts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import labels2b as L
from . import p2_index as PI
from . import rows as RW
from . import stamp

WORKSTREAM = "M7-lite"
TIME_ROLES = ("test_time", "test_time_n")             # later dates of training water bodies, by design of the split
WB_ROLES = ("val", "val_n", "test", "test_n", "val_p2", "test_p2")   # water bodies held out as a whole
CHIP_ROLES = (*WB_ROLES, *TIME_ROLES)
RADIUS_KM = 2.0                                       # the water-body grouping radius of the P2 build (p2.WB_PARAMS)
KEY_COLS = ("naip_chip_id", "site_id", "water_body", "lat", "lon", "hylak_id")


class CheckError(RuntimeError):
    """A row rule of the round is broken."""


# ------------------------------------------------------------------------------------------------ keys
def national_keys() -> pd.DataFrame:
    """uid, track, role and :data:`KEY_COLS` of every national-2b chip (no label column is read)."""
    split = L.load_split()
    k = pd.read_parquet(L.labels_path(), columns=list(KEY_COLS))
    k["naip_chip_id"] = k["naip_chip_id"].astype(str)
    k["role"] = k["naip_chip_id"].map(split["chip_role"])
    k["uid"] = k["naip_chip_id"]
    k["track"] = "national"
    return k


def p2_keys() -> pd.DataFrame:
    """uid, track, role and :data:`KEY_COLS` of every P2 index row (no label column is read)."""
    k = pd.read_parquet(PI.index_path(), columns=[*KEY_COLS, "role"])
    k["uid"] = RW.P2_PREFIX + k["naip_chip_id"].astype(str)
    k["track"] = "p2"
    return k


def protected_keys() -> pd.DataFrame:
    """Keys of every chip of a validation or test role of either track."""
    k = pd.concat([national_keys(), p2_keys()], ignore_index=True)
    return k[k["role"].isin(CHIP_ROLES)].reset_index(drop=True)


def _hylak(s: pd.Series) -> pd.Series:
    h = pd.to_numeric(s, errors="coerce")
    return h.where(h > 0)


# ------------------------------------------------------------------------------------------------ Q5 (i)
def refit_overlap(tr: pd.DataFrame, keys: pd.DataFrame, radius_km: float = 2.0) -> dict:
    """Counts of the overlap between the refit table ``tr`` (uid, track, role, water_body) and the protected keys.

    ``n_chip``: refit rows whose chip is a protected chip. ``n_water_body``: refit rows whose water body (same
    track, same id) holds a chip of a wholly held-out role (:data:`WB_ROLES`). ``n_water_body_time``: refit rows
    whose water body also has rows of a time-held-out test role — the design of ``test_time`` / ``test_time_n``,
    counted and reported, not a failure (M7_verify Q5: "anchors on time-held-out rows of training water bodies are
    reported as a separate stratum"). ``n_cross_track_near`` / ``n_cross_track_hylak``: refit rows within
    ``radius_km`` of, or sharing a HydroLAKES id with, a protected chip of the other track (whose water-body ids
    come from another grouping)."""
    bad_role = sorted(set(tr["role"].astype(str)) - {"train", "train_p2"})
    tw = tr["track"].astype(str) + "|" + tr["water_body"].astype(str)
    kw = keys["track"].astype(str) + "|" + keys["water_body"].astype(str)
    held = set(kw[keys["role"].isin(WB_ROLES)])
    timed = set(kw[keys["role"].isin(TIME_ROLES)])
    out = {"n_rows": int(len(tr)), "roles_not_train": bad_role,
           "n_chip": int(tr["uid"].astype(str).isin(set(keys["uid"].astype(str))).sum()),
           "n_water_body": int(tw.isin(held).sum()),
           "n_water_body_time": int(tw.isin(timed).sum()),
           "n_water_bodies_time": int(tw[tw.isin(timed)].nunique()),
           "n_cross_track_near": 0, "n_cross_track_hylak": 0, "radius_km": float(radius_km)}
    if {"lat", "lon"} <= set(tr.columns) and {"lat", "lon"} <= set(keys.columns):
        from ...eval.splits import nearest_distance_km

        for a, b in (("national", "p2"), ("p2", "national")):       # the two tracks have separate id sets
            t, k = tr[tr["track"] == a], keys[keys["track"] == b]
            if len(t) and len(k):
                out["n_cross_track_near"] += int((nearest_distance_km(t["lat"], t["lon"], k["lat"], k["lon"])
                                                  <= radius_km).sum())
                if "hylak_id" in tr and "hylak_id" in keys:
                    out["n_cross_track_hylak"] += int(_hylak(t["hylak_id"]).isin(
                        set(_hylak(k["hylak_id"]).dropna())).sum())
    return out


def assert_refit_clean(tr: pd.DataFrame, keys: pd.DataFrame | None = None) -> dict:
    """Refuse a refit table that holds a validation- or test-role chip or water body. Returns the counts."""
    o = refit_overlap(tr, protected_keys() if keys is None else keys)
    if o["roles_not_train"] or o["n_chip"] or o["n_water_body"] or o["n_cross_track_near"] \
            or o["n_cross_track_hylak"]:
        raise CheckError(f"the refit table holds evaluation-role rows: {o}")
    return o


# ------------------------------------------------------------------------------------------------ Q5 (ii)
def cross_track_conflicts(tr: pd.DataFrame, radius_km: float = RADIUS_KM) -> dict:
    """Counts of ``train_p2`` rows that could be a national-2b training water body under another id: rows within
    ``radius_km`` of a national training row with a different water-body id, and rows that share a HydroLAKES id
    with a national training row of a different water-body id. ``tr``: track, water_body, lat, lon[, hylak_id]."""
    from ...eval.splits import nearest_distance_km

    nat, p2 = tr[tr["track"] == "national"], tr[tr["track"] == "p2"]
    out = {"n_national": int(len(nat)), "n_p2": int(len(p2)), "radius_km": float(radius_km),
           "n_shared_id": int(len(set(nat["water_body"].astype(str)) & set(p2["water_body"].astype(str)))),
           "n_near": 0, "min_km": None, "n_hylak": 0}
    if not len(nat) or not len(p2):
        return out
    d = nearest_distance_km(p2["lat"], p2["lon"], nat["lat"], nat["lon"])
    out["min_km"] = float(np.min(d))
    out["n_near"] = int((d <= radius_km).sum())          # a row this near has, by construction, another id
    if "hylak_id" in tr:
        hn = pd.DataFrame({"h": _hylak(nat["hylak_id"]), "wb": nat["water_body"].astype(str)}).dropna()
        hp = pd.DataFrame({"h": _hylak(p2["hylak_id"]), "wb": p2["water_body"].astype(str)}).dropna()
        m = hp.merge(hn.drop_duplicates(), on="h", suffixes=("_p2", "_nat"))
        out["n_hylak"] = int((m["wb_p2"] != m["wb_nat"]).sum())
    return out


def assert_one_grouping(tr: pd.DataFrame, radius_km: float = RADIUS_KM) -> dict:
    c = cross_track_conflicts(tr, radius_km)
    if c["n_shared_id"] or c["n_near"] or c["n_hylak"]:
        raise CheckError(f"a train_p2 water body may be a national-2b training water body under another id: {c}")
    return c


def train_keys() -> pd.DataFrame:
    """The v2 training rows with their HydroLAKES id (training roles only)."""
    tr = RW.train_rows()
    h = pd.concat([national_keys(), p2_keys()], ignore_index=True)
    h = h[h["role"].isin(("train", "train_p2"))].drop_duplicates("uid").set_index("uid")["hylak_id"]
    return tr.assign(hylak_id=h.reindex(tr["uid"].astype(str)).to_numpy())


# ------------------------------------------------------------------------------------------------ A6: E3a_turb
E3A_EXPERIMENT = "lite/e3a_turb_count_national2b"
E3A_MODEL = "lite_E3a_turb_count"
E3A_MIN_POS, E3A_MIN_WB = 30, 10                     # lite_national_protocol.md N3 (events plan §1.2)


def e3a_counts(rows: pd.DataFrame) -> dict:
    """n, positives and water bodies with a positive of E3a_turb (turbidity > 50 NTU) by the frozen event scorer."""
    from ...events import lite as EL

    y = EL.labels(rows["y_turbidity"].to_numpy(float), "E3a_turb")
    ok = np.isfinite(y)
    n_pos = int(np.nansum(y[ok]))
    n_wb = int(rows.loc[ok & (y == 1), "water_body"].nunique())
    return {"n": int(ok.sum()), "n_pos": n_pos, "n_wb_pos": n_wb, "n_wb": int(rows.loc[ok, "water_body"].nunique()),
            "feasible": bool(n_pos >= E3A_MIN_POS and n_wb >= E3A_MIN_WB)}


def e3a_turb_count(registry: list | None = None, log_row=None, frames: dict | None = None) -> dict:
    """One-shot count of E3a_turb on ``val_n`` of the national-2b labels: after the lead's stamp, once, registered.
    Fewer than 30 positives or fewer than 10 water bodies with a positive → E3a_turb is descriptive only on the
    national track. Both label views are counted (``rerun``: configurations 1–6; ``v2``: the new arms); the event
    is confirmatory only if both pass. A second call is refused."""
    stamp.require_stamp("the E3a_turb count on val_n")
    if registry is None:
        from ...core.registry import read_registry

        registry = read_registry()
    if [r for r in registry if r.get("workstream") == WORKSTREAM and r.get("experiment") == E3A_EXPERIMENT]:
        raise CheckError("the E3a_turb count on val_n is already registered; it is counted once")
    if frames is None:
        split = L.load_split()
        frames = {v: L.role_frame(split, ["val_n"], view=v) for v in ("rerun", "v2")}
    bad = {v: sorted(set(f["role"].astype(str)) - {"val_n"}) for v, f in frames.items() if "role" in f}
    if any(bad.values()):
        raise CheckError(f"the E3a_turb count takes val_n rows only: {bad}")
    counts = {v: e3a_counts(f) for v, f in frames.items()}
    feasible = all(c["feasible"] for c in counts.values())
    out = {"counts": counts, "feasible": feasible, "status": "confirmatory" if feasible else "descriptive_only",
           "rule": f"descriptive only if n_pos < {E3A_MIN_POS} or n_wb_pos < {E3A_MIN_WB} (N3)"}
    if log_row is None:
        from ...core.registry import log_result as log_row
    c = counts.get("rerun", next(iter(counts.values())))
    log_row(workstream=WORKSTREAM, experiment=E3A_EXPERIMENT, model=E3A_MODEL, metric="n_pos", value=float(c["n_pos"]),
            split="val", n=int(c["n"]), target="E3a_turb", is_baseline=False,
            notes="one-shot count of E3a_turb positives on val_n of the national-2b labels (lead ruling A6); "
                  "no model output is used",
            extra={"written_by": "M7", "label_version": L.VERSION, **out})
    return out


__all__ = ["CheckError", "WB_ROLES", "TIME_ROLES", "CHIP_ROLES", "RADIUS_KM", "national_keys", "p2_keys",
           "protected_keys", "refit_overlap", "assert_refit_clean", "cross_track_conflicts", "assert_one_grouping",
           "train_keys", "E3A_EXPERIMENT", "E3A_MIN_POS", "E3A_MIN_WB", "e3a_counts", "e3a_turb_count"]
