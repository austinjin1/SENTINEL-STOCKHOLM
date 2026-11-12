# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Lake-context set C(w, t) of arm A1 (docs/preregistration/orbit_arch_v2_amendment.md §3.4 item 3) and the
development context cache with a refusing loader (§4 item 2).

Rule (fixed in the amendment and lead ruling R-7 of orbit_arch_v2_lead_rulings.md; nothing here is tuned and
nothing here is a setting):

* members: other chips of the same water body AND the same split role as the row (:data:`RULE_R7`) — in-lake rows:
  the same HydroLAKES id (a chip of that lake with another role is an error); non-lake rows: chips centred within
  100 m of the row's point that have the row's split role and 200-km block and are not in a lake — that pass the
  image QC and whose sensing date t' has 3 d <= |t' - t| <= 183 d;
* buffer-excluded chips are never used;
* date limit: rows are dated <= 2023-12-31 and so are their context chips (:func:`date_limit`);
* one chip per date: the chip nearest to the row's point (tie: lowest chip id);
* choice: up to K = 6 dates, lowest AOT first (unknown AOT last), ties by smaller |dt|, then the earlier date.

Only image metadata is read. :func:`assert_no_labels` refuses any table that carries a label column, so a candidate
table cannot bring a label into the selection. No context count is produced as a model feature.

The development cache holds chips of train-role or val-role water bodies dated <= 2023-12-31 only;
:func:`check_dev_chips` is run when the cache is written and again when it is loaded. Context chips of test-role
water bodies are read only inside the open Test-S claim, by the look's own module.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..orbit.data import TARGETS
from . import targets as T

K = 6
DT_MIN_D = 3
DT_MAX_D = 183
NONLAKE_RADIUS_M = 100.0
CAND_COLS = ("chip_id", "waterbody", "in_lake", "block", "role", "buffer_excluded", "qc_ok", "lon", "lat", "date",
             "aot")
ROW_COLS = ("waterbody", "in_lake", "block", "role", "lon", "lat", "date")
META_FILE = "context_meta.parquet"
ARRAY_FILE = "context_arrays.npz"
INFO_FILE = "context_info.json"
_EARTH_M = 6_371_008.8


class ContextError(RuntimeError):
    """The context rule or the development-cache rule would be broken."""


def label_columns() -> set:
    """Column names that hold a label or a label-derived quantity."""
    out = {"value", "value_min", "value_max", "detection_limit", "censored", "y", "label"}
    for t in (*TARGETS, *T.SPECS):
        out |= {t, f"{t}_cens", f"{t}_dl", f"{t}_value", f"{t}_outlier"}
    return out


def assert_no_labels(df: pd.DataFrame, what: str = "table") -> None:
    bad = sorted(set(df.columns) & label_columns())
    if bad:
        raise ContextError(f"{what} carries label columns {bad}; the context set is built from imagery metadata only")


def date_limit() -> pd.Timestamp:
    return pd.Timestamp(T.dev_date_max())


def _days(d) -> np.ndarray:
    return pd.to_datetime(pd.Series(d)).dt.normalize().to_numpy("datetime64[D]").astype(np.int64)


def _dist_m(lon0, lat0, lon, lat) -> np.ndarray:
    a = T._unit_xyz(lon, lat) - T._unit_xyz([lon0], [lat0])
    return 2.0 * np.arcsin(np.clip(np.linalg.norm(a, axis=1) / 2.0, 0, 1)) * _EARTH_M


def _pick(day: int, lon: float, lat: float, c_day, c_lon, c_lat, c_id, c_aot, k: int):
    """Positions (into the candidate arrays handed in) and dt [days] of the chosen context dates."""
    dt = c_day - day
    m = np.flatnonzero((np.abs(dt) >= DT_MIN_D) & (np.abs(dt) <= DT_MAX_D))
    if not m.size:
        return m, dt[m]
    d = _dist_m(lon, lat, c_lon[m], c_lat[m])
    o = m[np.lexsort((c_id[m], d, c_day[m]))]            # per date: nearest chip, tie lowest chip id
    o = o[np.r_[True, c_day[o][1:] != c_day[o][:-1]]]
    aot = np.where(np.isfinite(c_aot[o]), c_aot[o], np.inf)
    o = o[np.lexsort((c_day[o], np.abs(dt[o]), aot))][:k]  # lowest AOT, then smaller |dt|, then the earlier date
    return o, dt[o]


RULE_R7 = ("R-7: a context chip has the same frozen-rule water body and the same split role as its row; non-lake "
           "rows: chip centred within 100 m, same split role, same 200-km block, not in-lake; never buffer-excluded")


def select(rows: pd.DataFrame, cand: pd.DataFrame, *, k: int = K) -> tuple[np.ndarray, np.ndarray, dict]:
    """(cidx [N, k], cdt [N, k], counts). ``cidx`` holds positions into ``cand`` (-1 = none); ``cdt`` holds t' - t
    in days (NaN = none).

    ``rows``: :data:`ROW_COLS` (``date`` = sensing date of the row's own chip). ``cand``: :data:`CAND_COLS`, imagery
    metadata only. Membership is :data:`RULE_R7` (lead ruling R-7; clarification C-2: a constant, no switch):
    in-lake members share the lake, hence the role (checked, an error otherwise); non-lake members have the row's
    split role and 200-km block and are not in a lake."""
    for name, df, cols in (("rows", rows, ROW_COLS), ("candidates", cand, CAND_COLS)):
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ContextError(f"{name}: missing columns {missing}")
        assert_no_labels(df, name)
    if cand["chip_id"].duplicated().any():
        raise ContextError("candidate chip ids are not unique")
    lim = int(_days([date_limit()])[0])
    r_day = _days(rows["date"])
    if (r_day > lim).any():
        raise ContextError(f"{int((r_day > lim).sum())} rows are dated after {date_limit().date()}")
    c_day_all = _days(cand["date"])
    use = (cand["qc_ok"].to_numpy(bool) & ~cand["buffer_excluded"].to_numpy(bool) & (c_day_all <= lim)
           & np.isfinite(cand["lon"].to_numpy(float)) & np.isfinite(cand["lat"].to_numpy(float)))
    pos = np.flatnonzero(use)
    c_day, c_lon, c_lat = c_day_all[pos], cand["lon"].to_numpy(float)[pos], cand["lat"].to_numpy(float)[pos]
    c_id = cand["chip_id"].astype(str).to_numpy()[pos]
    c_aot = pd.to_numeric(cand["aot"], errors="coerce").to_numpy(float)[pos]
    c_wb = cand["waterbody"].astype(str).to_numpy()[pos]
    c_role = cand["role"].astype(str).to_numpy()[pos]
    c_block = cand["block"].astype(str).to_numpy()[pos]
    c_lake = cand["in_lake"].to_numpy(bool)[pos]
    n = len(rows)
    cidx = np.full((n, k), -1, np.int64)
    cdt = np.full((n, k), np.nan, np.float32)
    r_lon, r_lat = rows["lon"].to_numpy(float), rows["lat"].to_numpy(float)
    r_wb, r_role = rows["waterbody"].astype(str).to_numpy(), rows["role"].astype(str).to_numpy()
    r_lake = rows["in_lake"].to_numpy(bool)
    r_block = rows["block"].astype(str).to_numpy()
    n_role_dropped = 0

    def put(i, members):
        nonlocal n_role_dropped
        if not r_lake[i]:
            keep = (c_role[members] == r_role[i]) & (c_block[members] == r_block[i]) & ~c_lake[members]
            n_role_dropped += int((~keep).sum())
            members = members[keep]
        if not members.size:
            return
        o, dt = _pick(r_day[i], r_lon[i], r_lat[i], c_day[members], c_lon[members], c_lat[members], c_id[members],
                      c_aot[members], k)
        cidx[i, :o.size] = pos[members[o]]
        cdt[i, :o.size] = dt

    # in-lake rows: chips of the same HydroLAKES water body
    lake_members = pd.Series(np.flatnonzero(c_lake)).groupby(c_wb[c_lake]).apply(lambda s: s.to_numpy()).to_dict() \
        if c_lake.any() else {}
    for i in np.flatnonzero(r_lake):
        mem = lake_members.get(r_wb[i])
        if mem is None:
            continue
        if (c_role[mem] != r_role[i]).any():
            raise ContextError(f"water body {r_wb[i]}: a chip of the same lake has another split role")
        put(i, mem)
    # non-lake rows: chips centred within 100 m of the row's point
    nl = np.flatnonzero(~r_lake & np.isfinite(r_lon) & np.isfinite(r_lat))
    if nl.size and pos.size:
        from scipy.spatial import cKDTree

        tree = cKDTree(T._unit_xyz(c_lon, c_lat))
        chord = 2.0 * np.sin(NONLAKE_RADIUS_M / _EARTH_M / 2.0)
        for i, mem in zip(nl, tree.query_ball_point(T._unit_xyz(r_lon[nl], r_lat[nl]), chord)):
            if mem:
                put(i, np.asarray(sorted(mem), np.int64))
    n_ctx = (cidx >= 0).sum(1)
    counts = {"n_rows": int(n), "n_candidates": int(len(cand)), "n_candidates_usable": int(pos.size),
              "rows_with_0_context_dates": int((n_ctx == 0).sum()),
              "rows_with_1_to_5_context_dates": int(((n_ctx >= 1) & (n_ctx < k)).sum()),
              f"rows_with_{k}_context_dates": int((n_ctx == k).sum()),
              "n_distinct_context_chips": int(np.unique(cidx[cidx >= 0]).size),
              "rule": RULE_R7, "n_nonlake_members_dropped_other_role_block_or_lake": int(n_role_dropped)}
    return cidx, cdt, counts


# ------------------------------------------------------------------------------------ development cache

def check_dev_chips(meta: pd.DataFrame, what: str = "context chips") -> None:
    """Raise unless every chip belongs to a train-role or val-role water body, is dated <= the development limit,
    is not buffer-excluded and carries no label column."""
    need = {"chip_id", "role", "buffer_excluded", "date", "waterbody"}
    if not need <= set(meta.columns):
        raise ContextError(f"{what}: missing columns {sorted(need - set(meta.columns))}")
    assert_no_labels(meta, what)
    role = meta["role"].astype(str)
    bad_role = ~role.isin(T.DEV_ROLES)
    bad_buf = meta["buffer_excluded"].astype(bool)
    bad_date = _days(meta["date"]) > int(_days([date_limit()])[0])
    if bad_role.any() or bad_buf.any() or bad_date.any():
        raise ContextError(f"{what}: {int(bad_role.sum())} chips of a non-development role, {int(bad_buf.sum())} "
                           f"buffer-excluded, {int(bad_date.sum())} dated after {date_limit().date()}; refused")
    if meta["chip_id"].duplicated().any():
        raise ContextError(f"{what}: duplicate chip ids")
    wb = meta.drop_duplicates(["waterbody", "role"])["waterbody"]
    if wb.duplicated().any():
        raise ContextError(f"{what}: a water body appears with two roles")


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def dev_allow_list(index: pd.DataFrame) -> np.ndarray:
    """Chip ids that may enter the development context cache, decided from keys: train-role or val-role, not
    buffer-excluded, not protected, dated <= the development limit. ``index``: ``chip_id``, ``role``,
    ``buffer_excluded``, ``protected``, ``date``."""
    need = {"chip_id", "role", "buffer_excluded", "protected", "date"}
    if not need <= set(index.columns):
        raise ContextError(f"allow-list: missing columns {sorted(need - set(index.columns))}")
    assert_no_labels(index, "allow-list index")
    ok = (index["role"].astype(str).isin(T.DEV_ROLES).to_numpy() & ~index["buffer_excluded"].to_numpy(bool)
          & ~index["protected"].to_numpy(bool) & (_days(index["date"]) <= int(_days([date_limit()])[0])))
    return np.unique(index["chip_id"].astype(str).to_numpy()[ok])


def write_cache(d: Path, meta: pd.DataFrame, arrays: dict, *, allow, info: dict | None = None) -> dict:
    """Write the development context cache: ``meta`` (one row per chip) and ``arrays`` (each with ``len(meta)`` as
    first dimension). ``allow``: the chip-id allow-list of :func:`dev_allow_list`; a chip outside it is refused.
    The chips are checked first; nothing is written when a check fails."""
    check_dev_chips(meta)
    outside = ~meta["chip_id"].astype(str).isin(set(map(str, allow))).to_numpy()
    if outside.any():
        raise ContextError(f"{int(outside.sum())} chips are not in the development allow-list; refused")
    for name, a in arrays.items():
        if len(a) != len(meta):
            raise ContextError(f"array {name!r} has {len(a)} entries for {len(meta)} chips")
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    meta.reset_index(drop=True).to_parquet(d / META_FILE, index=False)
    np.savez(d / ARRAY_FILE, **{k: np.asarray(v) for k, v in arrays.items()})
    rec = {**(info or {}), "n_chips": int(len(meta)), "date_limit": str(date_limit().date()),
           "roles": sorted(set(meta["role"].astype(str))), "arrays": sorted(arrays),
           "sha256": {META_FILE: _sha(d / META_FILE), ARRAY_FILE: _sha(d / ARRAY_FILE)}}
    (d / INFO_FILE).write_text(json.dumps(rec, indent=1) + "\n")
    return rec


def load_cache(d: Path) -> tuple[pd.DataFrame, dict, dict]:
    """(meta, arrays, info) of a development context cache. Refuses a cache whose files do not match the recorded
    hashes or that holds any chip :func:`check_dev_chips` refuses."""
    d = Path(d)
    info = json.loads((d / INFO_FILE).read_text())
    for name, sha in info["sha256"].items():
        if _sha(d / name) != sha:
            raise ContextError(f"{d / name}: content differs from the hash recorded when the cache was written")
    meta = pd.read_parquet(d / META_FILE)
    check_dev_chips(meta, what=f"context cache {d.name}")
    with np.load(d / ARRAY_FILE, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    if any(len(a) != len(meta) for a in arrays.values()):
        raise ContextError("context cache: array length differs from the chip table")
    return meta, arrays, info


# ------------------------------------------------------------------------------------ store pin (R-7)

STORES = ("aquacorr", "wqp")     # R-7: the context stores are s2_chips/aquacorr and s2_chips/wqp only
PIN_CHIPS, PIN_INFO = "chips.parquet", "pin.json"
PIN_COLS = ("store", "store_chip_id", "shard")


def pin_dir() -> Path:
    from ...core import paths

    return paths.RUNS_ROOT / "orbit" / "v2" / "context_store_pin"


def store_dir(store: str) -> Path:
    from ...core import paths

    if store not in STORES:
        raise ContextError(f"chip store {store!r} is not a context store {STORES}")
    return paths.processed("s2_chips") / store


def pin_table(index_by_store: dict) -> pd.DataFrame:
    """One row per chip (:data:`PIN_COLS`) of the chip indexes handed in ({store: index with ``chip_id`` and
    ``shard``}). Only the two stores of :data:`STORES` are accepted and both must be present."""
    if tuple(sorted(index_by_store)) != tuple(sorted(STORES)):
        raise ContextError(f"the pin covers exactly the stores {STORES}, got {sorted(index_by_store)}")
    parts = [pd.DataFrame({"store": st, "store_chip_id": index_by_store[st]["chip_id"].astype(str).to_numpy(),
                           "shard": index_by_store[st]["shard"].astype(str).to_numpy()}) for st in STORES]
    return pd.concat(parts, ignore_index=True).drop_duplicates().sort_values(list(PIN_COLS)).reset_index(drop=True)


def write_pin(d: Path, chips: pd.DataFrame, *, shard_file=None, created: str = "", log=print) -> dict:
    """Write the content pin of the context stores: the chip table and the sha256 of every shard file it names.
    Made once, at the first dev_v5m build; an existing pin is never replaced (refused)."""
    d = Path(d)
    if (d / PIN_INFO).exists() or (d / PIN_CHIPS).exists():
        raise ContextError(f"{d}: a context store pin exists; it is made once and never replaced")
    if list(chips.columns) != list(PIN_COLS) or chips.duplicated().any():
        raise ContextError(f"pin table needs exactly the columns {PIN_COLS}, without duplicate rows")
    shard_file = shard_file or (lambda store, shard: store_dir(store) / "shards" / shard)
    sh = chips.drop_duplicates(["store", "shard"])
    shas = {}
    for i, (st, s) in enumerate(zip(sh["store"], sh["shard"])):
        shas[f"{st}/{s}"] = _sha(shard_file(st, s))
        if (i + 1) % 50 == 0:
            log(f"[pin] hashed {i + 1}/{len(sh)} shard files")
    d.mkdir(parents=True, exist_ok=True)
    chips.to_parquet(d / PIN_CHIPS, index=False)
    rec = {"created": created, "rule": "R-7: context stores s2_chips/aquacorr and s2_chips/wqp only, content-pinned "
           "at the first dev_v5m build; any store or chip added after the pin is excluded",
           "stores": list(STORES), "n_chips": {st: int((chips["store"] == st).sum()) for st in STORES},
           "n_shards": int(len(sh)), "chips_sha256": _sha(d / PIN_CHIPS), "shard_sha256": shas}
    (d / PIN_INFO).write_text(json.dumps(rec, indent=1) + "\n")
    return rec


def load_pin(d: Path) -> tuple[pd.DataFrame, dict]:
    """(chip table, record) of a pin; refuses a chip table that differs from the recorded hash."""
    d = Path(d)
    rec = json.loads((d / PIN_INFO).read_text())
    if _sha(d / PIN_CHIPS) != rec["chips_sha256"]:
        raise ContextError(f"{d / PIN_CHIPS}: content differs from the hash recorded in the pin")
    chips = pd.read_parquet(d / PIN_CHIPS)
    if list(chips.columns) != list(PIN_COLS) or set(chips["store"]) - set(STORES):
        raise ContextError("the pin's chip table has other columns or another store")
    return chips, rec


def apply_pin(cand: pd.DataFrame, chips: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Candidates (``store``, ``store_chip_id``, ``shard``) that the pin names; a store outside :data:`STORES` and
    every chip added after the pin are dropped (counted)."""
    key = lambda df: (df["store"].astype(str) + "\x1f" + df["store_chip_id"].astype(str) + "\x1f"  # noqa: E731
                      + df["shard"].astype(str))
    keep = key(cand).isin(set(key(chips))).to_numpy() & cand["store"].astype(str).isin(STORES).to_numpy()
    return cand[keep].reset_index(drop=True), {"n_candidates": int(len(cand)), "n_in_pin": int(keep.sum()),
                                               "n_not_in_pin_dropped": int((~keep).sum())}


def check_pinned_shards(rec: dict, used, *, shard_file=None) -> int:
    """Raise unless every (store, shard) of ``used`` is in the pin and its file has the pinned sha256. Returns the
    number of shard files checked."""
    shard_file = shard_file or (lambda store, shard: store_dir(store) / "shards" / shard)
    n = 0
    for st, s in sorted(set((str(a), str(b)) for a, b in used)):
        want = rec["shard_sha256"].get(f"{st}/{s}")
        if want is None:
            raise ContextError(f"shard {st}/{s} is not in the context store pin")
        if _sha(shard_file(st, s)) != want:
            raise ContextError(f"shard {st}/{s}: content differs from the context store pin")
        n += 1
    return n
