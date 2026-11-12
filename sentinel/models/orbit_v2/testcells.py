# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Test-S cells from keys and geometry only (amendment §1.6, §6.4): scene keys, the reference scene set of the
development rows, the scene-disjoint flag, the primary-cell rule and the distance to training points.

Nothing here reads a label, a prediction or a pixel. The functions take key columns (STAC item id, sensing time,
role block, coordinates), so the same code serves the keys-only split report (which fixes the primary cell before
the freeze, from counts) and the look script (which writes the flag into the sealed tables).

Conservative reading (reported to the lead): a row whose scene key cannot be formed (no MGRS tile in the item id,
or no sensing date) is NOT scene-disjoint, and is counted."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..orbit import features as F1
from ..orbit.split import nearest_distance_km

KEY_COLS = ("item_id", "sensing_time_utc")
MISSING = "NA"
TILE_RE = r"(?:_T|_)(\d{2}[C-X][A-Z]{2})(?:_|$)"       # the rule of scripts/data/orbitv2_testS_keycounts.py (§1.4)
MIN_ROW_SHARE, MIN_BLOCK_SHARE = 0.60, 0.80             # §1.6
PRIMARY_TARGET = "chla"                                 # §1.6: the rule is evaluated on the chl-a cell


class CellError(RuntimeError):
    pass


def scene_key(item_id, sensing) -> np.ndarray:
    """(MGRS tile, sensing date) as ``"<tile>|<YYYYMMDD>"``; ``NA`` for a part that cannot be read."""
    tile = pd.Series(np.asarray(item_id, dtype=object)).astype(str).str.extract(TILE_RE)[0]
    day = pd.to_datetime(pd.Series(np.asarray(sensing, dtype=object)), utc=True, errors="coerce").dt.strftime("%Y%m%d")
    return (tile.fillna(MISSING) + "|" + day.fillna(MISSING)).to_numpy()


def keyed(keys) -> np.ndarray:
    return np.array([MISSING not in str(k).split("|") for k in np.asarray(keys)], bool)


def row_keys(df: pd.DataFrame) -> np.ndarray:
    missing = [c for c in KEY_COLS if c not in df.columns]
    if missing:
        raise CellError(f"the table has no {missing} column: scene keys cannot be formed (amendment §1.6)")
    return scene_key(df["item_id"].to_numpy(), df["sensing_time_utc"].to_numpy())


def reference_scenes(dev: pd.DataFrame, fit_scheme: str) -> tuple[set, dict]:
    """Scene keys of every fit-cell and inner-cell row of the development table (any target, any water type,
    added rows included) under the arms' fit scheme. Returns (keys, counts)."""
    fit, inner = F1.fit_and_inner(dev, fit_scheme)
    m = np.asarray(fit, bool) | np.asarray(inner, bool)
    keys = row_keys(dev[m])
    ok = keyed(keys)
    return set(keys[ok]), {"n_reference_rows": int(m.sum()), "n_reference_rows_without_key": int((~ok).sum()),
                           "n_reference_scenes": int(len(set(keys[ok]))), "fit_scheme": fit_scheme}


def scene_disjoint(keys, reference: set) -> tuple[np.ndarray, dict]:
    """True where the row's scene is in no fit or inner row's scene. A row without a key is not disjoint."""
    keys = np.asarray(keys)
    ok = keyed(keys)
    flag = ok & ~np.fromiter((k in reference for k in keys), bool, len(keys))
    return flag, {"n": int(len(keys)), "n_without_key": int((~ok).sum()), "n_scene_disjoint": int(flag.sum())}


def primary_cell(disjoint, blocks) -> tuple[str, dict]:
    """§1.6 on the chl-a rows of ``testxarchive_lake``: the scene-disjoint subset is the primary cell if it keeps
    at least 60 % of the rows and at least 80 % of the role blocks; otherwise all rows. Counts only."""
    disjoint, blocks = np.asarray(disjoint, bool), np.asarray(blocks)
    if not len(disjoint) or len(disjoint) != len(blocks):
        raise CellError("the primary-cell rule needs the chl-a rows of the cell and their role blocks")
    n, nb = len(disjoint), len(np.unique(blocks))
    nd, nbd = int(disjoint.sum()), len(np.unique(blocks[disjoint]))
    rec = {"n_rows": n, "n_blocks": nb, "n_rows_scene_disjoint": nd, "n_blocks_scene_disjoint": nbd,
           "row_share": nd / n, "block_share": nbd / nb, "min_row_share": MIN_ROW_SHARE,
           "min_block_share": MIN_BLOCK_SHARE}
    ok = nd >= MIN_ROW_SHARE * n and nbd >= MIN_BLOCK_SHARE * nb
    return ("scene_disjoint" if ok else "all_rows"), rec


def train_points(dev: pd.DataFrame) -> pd.DataFrame:
    """Coordinates of the train-role development rows (§6.4: "> 10 km from any train-role point")."""
    return dev.loc[dev["role"].astype(str) == "train", ["lat", "lon"]].drop_duplicates().reset_index(drop=True)


def dist_train_km(df: pd.DataFrame, train: pd.DataFrame) -> np.ndarray:
    if not len(train):
        raise CellError("no train-role point")
    return np.asarray(nearest_distance_km(df["lat"], df["lon"], train["lat"], train["lon"]), float)
