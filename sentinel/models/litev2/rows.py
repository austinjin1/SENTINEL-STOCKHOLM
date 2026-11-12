# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Rows of architecture round v2 in one frame format (amendment §3.4, §4.1, §8.7).

* **v2 training rows** = national-2b ``train`` (view ``v2``) ∪ ``train_p2`` after the F7 removals. Every new arm
  and every comparator is fitted on exactly these rows (:func:`train_rows`).
* **evaluation blocks**: ``national_val`` (= ``val`` ∪ ``val_n`` of national-2b) and ``val_p2``. Both are returned
  only after the lead's stamp (``stamp.guard_roles``; the loaders underneath check again). No test role is loaded
  here.
* The two chip stores number their chips independently, so a P2 chip id can equal a national chip id of another
  site. ``uid`` is the unique row key: the chip id for national rows, ``"p2:" + chip id`` for P2 rows.

Columns of every frame: :data:`KEY_COLS`, the label columns ``y_<target>`` for the six targets and ``y_turb_ntru``
(log10, uncensored values only), ``cens_<target>`` (-1 left, +1 right, 0 none) and ``lim_<target>`` (log10 limit).
``dt_label_days`` is the number of days between the image and the nearest sample: a label-side field, used only by
the CyFi-style comparators P2a / P2b (M7_verify missed[5]) and never by a candidate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..lite import data as D
from ..lite import national2 as N2
from . import labels2b as L
from . import p2_index as PI
from . import stamp

KEY_COLS = ("uid", "naip_chip_id", "track", "role", "site_id", "water_body", "site_class", "lat", "lon", "doy",
            "gsd_m", "naip_item_id", "naip_year", "naip_date", "chip_crs", "chip_left", "chip_top", "s2_chip_id",
            "dt_label_days")
Y_COLS = tuple(L.Y_COLS)                                            # six targets + y_turb_ntru
CENS_TARGETS = ("secchi", "turbidity", "turb_fnu", "tss", "ssc", "chla")
CENS_COLS = tuple(f"cens_{t}" for t in CENS_TARGETS)
LIM_COLS = tuple(f"lim_{t}" for t in CENS_TARGETS)
EVAL_BLOCKS = {"national_val": ("national", ("val", "val_n")), "val_p2": ("p2", ("val_p2",))}
P2_PREFIX = "p2:"


class RowError(RuntimeError):
    """A v2 row frame breaks one of its rules."""


def _finish(df: pd.DataFrame) -> pd.DataFrame:
    for c in (*Y_COLS, *LIM_COLS):
        if c not in df:
            df[c] = np.nan
        df[c] = df[c].astype(float)
    for c in CENS_COLS:
        if c not in df:
            df[c] = 0
        df[c] = df[c].fillna(0).astype(int)
    out = df[[*KEY_COLS, *Y_COLS, *CENS_COLS, *LIM_COLS]].reset_index(drop=True)
    check(out)
    return out


def check(rows: pd.DataFrame) -> None:
    """Structural rules of a v2 frame (values are not inspected beyond finite / missing)."""
    if rows["uid"].duplicated().any():
        raise RowError("duplicate uid in a v2 row frame")
    for t in CENS_TARGETS:
        c, y, lim = rows[f"cens_{t}"].to_numpy(), rows[f"y_{t}"].to_numpy(float), rows[f"lim_{t}"].to_numpy(float)
        if not np.isin(c, (-1, 0, 1)).all():
            raise RowError(f"cens_{t} holds a value outside -1, 0, 1")
        if (np.isfinite(y) & (c != 0)).any():
            raise RowError(f"{t}: a censored row also has an uncensored value (§3.3)")
        if ((c != 0) & ~np.isfinite(lim)).any():
            raise RowError(f"{t}: a censored row has no limit")
    protected = set(L.TEST_ROLES) | {"test_p2"}
    if set(rows["role"].astype(str)) & protected:
        raise RowError("a protected role reached a v2 row frame")


def national_rows(roles) -> pd.DataFrame:
    """National-2b rows of ``roles`` (view ``v2``) in the v2 frame format."""
    stamp.guard_roles(roles, "national-2b rows for a v2 arm or comparator")
    split = L.load_split()
    lab = L.role_frame(split, roles, view="v2").copy()
    lab["uid"] = lab["naip_chip_id"].astype(str)
    lab["track"] = "national"
    lab["dt_label_days"] = (pd.to_datetime(lab["sample_date"]) - pd.to_datetime(lab["naip_date"])).dt.days.astype(float)
    return _finish(lab)


def p2_rows(roles) -> pd.DataFrame:
    """P2 rows of ``roles`` (``train_p2`` after F7, ``val_p2`` after the stamp) in the v2 frame format."""
    bad = sorted({str(r) for r in roles} - {"train_p2", "val_p2"})
    if bad:
        raise RowError(f"P2 roles {bad} are not loaded by the v2 row loader")
    ix = PI.load_index(list(roles), what="lite P2 rows for a v2 arm or comparator").copy()
    ix["uid"] = P2_PREFIX + ix["naip_chip_id"].astype(str)
    ix["track"] = "p2"
    ix["s2_chip_id"] = None
    ix["dt_label_days"] = ix["dt_days"].astype(float)
    ix["cens_turb_fnu"] = ix["cens"].astype(int)
    ix["lim_turb_fnu"] = ix["cens_log10_limit"].astype(float)
    return _finish(ix)


def train_rows() -> pd.DataFrame:
    """The v2 training rows: national-2b ``train`` ∪ ``train_p2`` (F7 applied). No validation row is touched."""
    out = pd.concat([national_rows(["train"]), p2_rows(["train_p2"])], ignore_index=True)
    check(out)
    if set(out["role"]) - {"train", "train_p2"}:
        raise RowError("a non-training role reached the v2 training rows")
    return out


def eval_rows(block: str) -> pd.DataFrame:
    """Rows of an evaluation block of :data:`EVAL_BLOCKS`. Needs the lead's stamp."""
    if block not in EVAL_BLOCKS:
        raise RowError(f"unknown evaluation block {block!r} (known: {sorted(EVAL_BLOCKS)})")
    track, roles = EVAL_BLOCKS[block]
    stamp.guard_roles(roles, f"v2 evaluation block {block}")
    return national_rows(roles) if track == "national" else p2_rows(roles)


def scene_table(rows: pd.DataFrame) -> pd.DataFrame:
    """DOQQ scene statistics (all columns of the scene files) aligned with ``rows``; a missing scene is an error."""
    out = []
    for track, path in (("national", N2.scene_stats_path()), ("p2", PI.out_dir() / "naip_scene_stats.parquet")):
        m = (rows["track"] == track).to_numpy()
        if not m.any():
            continue
        st = D.scene_stats(path).reindex(rows.loc[m, "naip_item_id"])
        st.index = rows.index[m]
        out.append(st)
    st = pd.concat(out).reindex(rows.index)
    if st.isna().all(axis=1).any():
        raise RowError(f"{int(st.isna().all(axis=1).sum())} rows lack DOQQ statistics")
    return st


def chips(rows: pd.DataFrame, cache_name: str | None = None) -> np.ndarray:
    """uint8 (N, 4, 256, 256) chips in ``rows`` order. National chips come from the national store (cached as one
    .npy when ``cache_name`` is given), P2 chips from the P2 store."""
    from ...data.chips import ChipStore

    out = np.zeros((len(rows), 4, D.CHIP_SIZE, D.CHIP_SIZE), np.uint8)
    nat = (rows["track"] == "national").to_numpy()
    if nat.any():
        sub = rows.loc[nat]
        out[nat] = (np.asarray(D.cached_chips(sub, cache_name)) if cache_name
                    else D.load_chips(sub["naip_chip_id"].astype(str).tolist()))
    if (~nat).any():
        out[~nat] = D.load_chips(rows.loc[~nat, "naip_chip_id"].astype(str).tolist(), store=ChipStore(PI.chip_store()))
    return out


__all__ = ["KEY_COLS", "Y_COLS", "CENS_TARGETS", "CENS_COLS", "LIM_COLS", "EVAL_BLOCKS", "RowError", "check",
           "national_rows", "p2_rows", "train_rows", "eval_rows", "scene_table", "chips"]
