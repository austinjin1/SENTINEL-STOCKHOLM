# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Data version dev_v5m (docs/preregistration/orbit_arch_v2_amendment.md §4): dev_v5 plus the co-located
multi-target join, per-pixel shore distance, land ring, glint features and the lake-context index.

This module holds the rules and the cache writer/loader; the build script is
``scripts/data/orbitv2_build_dev_v5m.py``. Rules enforced here, at write time and again at load time:

* every dev_v5 row is kept, with equal row id, chip id and base-target values (:func:`check_base_rows`);
* no protected row; development roles and years only; no new-target value on a buffer-excluded row; censoring
  columns are consistent (:func:`check_dev_rows`);
* the context index points only into the development context cache (:mod:`.context`), and every context chip of
  a row obeys the membership rule of lead ruling R-7 again (:func:`check_context`; a constant, no switch: C-2);
* context chips come from the stores ``s2_chips/aquacorr`` and ``s2_chips/wqp`` only, and only chips named by the
  content pin made at the first build (:func:`context.write_pin`);
* the files on disk equal the hashes recorded when the cache was written;
* no protected row id or chip id is in any file of the cache (:func:`audit_artefacts`, R-21).

Context candidates are built from the chip-store index (image metadata) only: :func:`context_candidates`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ...core import paths
from ..orbit import data as D
from ..orbit import guard
from ..orbit import split as SP
from . import chipfeat as F
from . import context as CX
from . import targets as T

VERSION = "dev_v5m"
BASE_VERSION = "dev_v5"
CONTEXT_STORES = CX.STORES                    # R-7: s2_chips/aquacorr and s2_chips/wqp only; no other setting
P_CONTEXT = 8
TABLE, PIXELS, FEATURES, INFO, CONTEXT_DIR = "table.parquet", "pixels.npz", "features.npz", "info.json", "context"
ROW_ARRAYS = ("dp_m", "ring", "glint", "cidx", "cdt")
RW3_COLS = tuple(f"rw3_{b}" for b in D.BANDS)      # chip-index columns: exact 3 x 3 water median per band
CONTEXT_ARRAYS = ("refl", "offsets", "n_pix", "dp_m", "q", "rw3")   # per context chip; rw3 [m, 12] = RW3_COLS
CONTEXT_META = ("chip_id", "waterbody", "in_lake", "block", "role", "buffer_excluded", "date", "platform", "lon",
                "lat")
ARTEFACT_SUFFIXES = (".parquet", ".npz", ".json")
BASE_EQUAL_COLS = ("chip_id", *D.TARGETS)


class DevmError(RuntimeError):
    """A dev_v5m rule would be broken."""


def new_cols() -> list:
    return [c for t in T.SPECS for c in (t, f"{t}_cens", f"{t}_dl")]


def cache_dir(version: str = VERSION) -> Path:
    return paths.RUNS_ROOT / "orbit" / "cache" / version


# ------------------------------------------------------------------------------------ row rules

def check_base_rows(base: pd.DataFrame, df: pd.DataFrame) -> dict:
    """Every dev_v5 row is in dev_v5m once, with equal chip id and base-target values; an added row carries no
    base-target value and at least one new-target value or censored record."""
    if df["row_id"].duplicated().any() or base["row_id"].duplicated().any():
        raise DevmError("duplicate row ids")
    added = df["added"].to_numpy(bool)
    kept = df.loc[~added].set_index("row_id")
    b = base.set_index("row_id")
    if set(kept.index) != set(b.index):
        raise DevmError(f"dev_v5 rows are not kept one to one: {len(set(b.index) - set(kept.index))} missing, "
                        f"{len(set(kept.index) - set(b.index))} unknown rows not marked as added")
    kept = kept.loc[b.index]
    for c in BASE_EQUAL_COLS:
        x, y = kept[c].to_numpy(), b[c].to_numpy()
        same = (x == y) | (pd.isna(x) & pd.isna(y))
        if not same.all():
            raise DevmError(f"column {c!r} differs from {BASE_VERSION} on {int((~same).sum())} rows")
    add = df.loc[added]
    if len(add):
        if add[list(D.TARGETS)].notna().any().any():
            raise DevmError("an added row carries a base-target value")
        has_new = np.zeros(len(add), bool)
        for t in T.SPECS:
            has_new |= add[t].notna().to_numpy() | add[f"{t}_cens"].to_numpy(bool)
        if not has_new.all():
            raise DevmError(f"{int((~has_new).sum())} added rows carry no new-target value")
    return {"n_base_rows": int(len(b)), "n_added_rows": int(added.sum())}


def check_dev_rows(df: pd.DataFrame) -> None:
    """Protected rows, non-development roles or years, new-target values on buffer-excluded rows and inconsistent
    censoring columns are refused."""
    guard.check_rows(df["protected"].to_numpy(bool), training=True, what=f"{VERSION} rows")
    bad_role = ~df["role"].astype(str).isin(T.DEV_ROLES).to_numpy()
    bad_tb = ~df["tblock"].astype(str).isin(T.DEV_TBLOCKS).to_numpy()
    bad_date = CX._days(df["date"]) > int(CX._days([CX.date_limit()])[0])
    if bad_role.any() or bad_tb.any() or bad_date.any():
        raise DevmError(f"{VERSION}: {int(bad_role.sum())} rows of a non-development role, {int(bad_tb.sum())} of a "
                        f"non-development time block, {int(bad_date.sum())} dated after {CX.date_limit().date()}")
    buf = df["buffer_excluded"].to_numpy(bool)
    for t, spec in T.SPECS.items():
        v, c, dl = df[t].to_numpy(float), df[f"{t}_cens"].to_numpy(bool), df[f"{t}_dl"].to_numpy(float)
        if (buf & (np.isfinite(v) | c)).any():
            raise DevmError(f"{t}: a buffer-excluded row carries a new-target value")
        if (c & np.isfinite(v)).any():
            raise DevmError(f"{t}: a row is both quantified and censored")
        if (c & ~(np.isfinite(dl) & (dl >= spec.lo) & (dl <= spec.hi))).any():
            raise DevmError(f"{t}: a censored row has no detection limit inside the valid range")
        if (np.isfinite(v) & ((v < spec.lo) | (v > spec.hi))).any():
            raise DevmError(f"{t}: a value lies outside the valid range")


# ------------------------------------------------------------------------------------ context candidates

def sensing_date(sensing_time_utc) -> pd.Series:
    return pd.to_datetime(pd.Series(np.asarray(sensing_time_utc)), utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()


def context_candidates(index: pd.DataFrame, store: str, split: pd.DataFrame) -> pd.DataFrame:
    """Candidate table of one chip store for :func:`context.select`, from image metadata only.

    ``index``: rows of the store's chip index (status ok). ``split``: ``split.assign`` output for the chip points
    and sensing dates, aligned with ``index``. The image QC is :func:`data.image_qc` with ``same_acquisition`` set
    true: that flag describes a sample-to-image match, which a context chip does not have. AOT is ``aot_c3``, the
    Sen2Cor AOT of the centre window that v1 uses. Chip ids are prefixed with the store name."""
    if store not in CONTEXT_STORES:
        raise DevmError(f"chip store {store!r} is not a context store {CONTEXT_STORES}")
    if len(index) != len(split):
        raise DevmError("index and split differ in length")
    CX.assert_no_labels(index, f"chip index {store}")
    idx = index.reset_index(drop=True)
    sp = split.reset_index(drop=True)
    qc = D.image_qc(idx.assign(same_acquisition=True)).to_numpy(bool)
    out = pd.DataFrame({
        "chip_id": store + ":" + idx["chip_id"].astype(str), "store": store, "store_chip_id": idx["chip_id"].astype(str),
        "shard": idx["shard"].astype(str), "waterbody": sp["waterbody"].astype(str), "in_lake": sp["in_lake"].astype(bool),
        "block": sp["block"].astype(str), "role": sp["role"].astype(str), "tblock": sp["tblock"].astype(str),
        "buffer_excluded": sp["buffer_excluded"].astype(bool), "protected": sp["protected"].astype(bool),
        "qc_ok": qc, "lon": idx["lon"].astype(float), "lat": idx["lat"].astype(float),
        "date": sensing_date(idx["sensing_time_utc"]), "aot": pd.to_numeric(idx["aot_c3"], errors="coerce"),
    })
    for c in ("sun_zenith", "view_zenith", "sun_azimuth", "view_azimuth", "processing_baseline", "item_id",
              "platform"):                    # platform: band wavelengths of the chip (lakeset input)
        out[c] = idx[c].to_numpy()
    missing = [c for c in RW3_COLS if c not in idx.columns]
    if missing:
        raise DevmError(f"chip index {store}: missing 3 x 3 water-median columns {missing}")
    for c in RW3_COLS:                        # image statistics of the chip (comparator features, amendment §3.5)
        out[c] = pd.to_numeric(idx[c], errors="coerce").to_numpy(np.float32)
    out = out[out["date"].notna()].reset_index(drop=True)
    if out["chip_id"].duplicated().any():
        out = out.sort_values(["chip_id", "shard"]).drop_duplicates("chip_id").reset_index(drop=True)
    return out


def store_candidates(log=print, index=D._index) -> pd.DataFrame:
    """Candidates of the context stores (:data:`CONTEXT_STORES`, nothing else), with split columns from the frozen
    rule. The lake assignment is read-only (:func:`targets.lakes_readonly`): the shared lake cache is never
    written. The caller applies the content pin (:func:`context.apply_pin`)."""
    parts = []
    for st in CONTEXT_STORES:
        idx = index(st)
        dates = sensing_date(idx["sensing_time_utc"])
        lon, lat = idx["lon"].to_numpy(float), idx["lat"].to_numpy(float)
        sp = SP.assign(lon, lat, dates.to_numpy(), lakes=T.lakes_readonly(lon, lat, log=log), log=log)
        parts.append(context_candidates(idx, st, sp))
        log(f"[devm] context candidates {st}: {len(parts[-1])} chips")
    return pd.concat(parts, ignore_index=True)


def context_chip_arrays(chip: np.ndarray, layers=None) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """(refl [8, 12], offsets [8, 2], n_valid, d_p [8]) of one context chip: v1's pixel rule with 8 pixels."""
    kw = {} if layers is None else {"layers": layers}
    refl, off, n, _ = D.chip_pixels(chip, p_max=P_CONTEXT, **kw)
    return refl, off, n, F.dp_at(F.dp_map(chip, **kw), off, n)


def check_context(df: pd.DataFrame, cidx: np.ndarray, cdt: np.ndarray, cmeta: pd.DataFrame) -> dict:
    """Re-check the membership rule (:data:`context.RULE_R7`) for every (row, context chip) pair of a cache; counts
    by context availability. In-lake rows: same water body and role. Non-lake rows: same role and 200-km block, the
    chip not in a lake and centred within 100 m. No chip is buffer-excluded."""
    cidx = np.asarray(cidx)
    if cidx.shape != (len(df), CX.K) or np.asarray(cdt).shape != cidx.shape:
        raise DevmError(f"context index shape {cidx.shape} for {len(df)} rows and K = {CX.K}")
    if cidx.max(initial=-1) >= len(cmeta) or cidx.min(initial=-1) < -1:
        raise DevmError("context index points outside the context cache")
    r, s = np.nonzero(cidx >= 0)
    c = cidx[r, s]
    if r.size:
        dt = CX._days(cmeta["date"])[c] - CX._days(df["chip_date"])[r]
        if not np.array_equal(dt.astype(np.float32), np.asarray(cdt, np.float32)[r, s]):
            raise DevmError("stored context time differences do not match the chip dates")
        if ((np.abs(dt) < CX.DT_MIN_D) | (np.abs(dt) > CX.DT_MAX_D)).any():
            raise DevmError("a context chip lies outside the 3-183 day window")
        lake = df["in_lake"].to_numpy(bool)[r]
        wb_same = cmeta["waterbody"].astype(str).to_numpy()[c] == df["waterbody"].astype(str).to_numpy()[r]
        role_same = cmeta["role"].astype(str).to_numpy()[c] == df["role"].astype(str).to_numpy()[r]
        if (lake & ~(wb_same & role_same)).any():
            raise DevmError("an in-lake row has a context chip of another water body or role")
        if (~lake & ~role_same).any():
            raise DevmError("a non-lake row has a context chip of another split role")
        block_same = cmeta["block"].astype(str).to_numpy()[c] == df["block"].astype(str).to_numpy()[r]
        if (~lake & (~block_same | cmeta["in_lake"].to_numpy(bool)[c])).any():
            raise DevmError("a non-lake row has a context chip of another 200-km block or an in-lake chip")
        if cmeta["buffer_excluded"].to_numpy(bool)[c].any():
            raise DevmError("a context chip is buffer-excluded")
        nl = ~lake
        if nl.any():
            dist = CX._dist_m(df["lon"].to_numpy(float)[r][nl], df["lat"].to_numpy(float)[r][nl],
                              cmeta["lon"].to_numpy(float)[c][nl], cmeta["lat"].to_numpy(float)[c][nl])
            if not (np.asarray(dist) <= CX.NONLAKE_RADIUS_M * (1.0 + 1e-6)).all():
                raise DevmError("a non-lake row has a context chip centred more than 100 m away")
        key = pd.DataFrame({"r": r, "d": CX._days(cmeta["date"])[c]})
        if key.duplicated().any():
            raise DevmError("a row has two context chips of the same date")
    n = (cidx >= 0).sum(1)
    return {"rows_with_0_context_dates": int((n == 0).sum()),
            "rows_with_1_to_5_context_dates": int(((n >= 1) & (n < CX.K)).sum()),
            f"rows_with_{CX.K}_context_dates": int((n == CX.K).sum())}


# ------------------------------------------------------------------------------------ cache

def write(d: Path, df: pd.DataFrame, pix: dict, feats: dict, info: dict, *, base: pd.DataFrame, cmeta: pd.DataFrame,
          carrays: dict, allow) -> dict:
    """Write a dev_v5m cache into ``d``. ``feats``: :data:`ROW_ARRAYS` (first dimension = rows). ``cmeta`` /
    ``carrays`` / ``allow``: the development context cache (:func:`context.write_cache`). Every check runs before
    the first file is written."""
    missing = [c for c in (*new_cols(), "added", "chip_date", "row_id", "protected", "role", "tblock",
                           "buffer_excluded", "waterbody", "in_lake", "date") if c not in df.columns]
    if missing:
        raise DevmError(f"table: missing columns {missing}")
    check_dev_rows(df)
    counts = check_base_rows(base, df)
    for name in ROW_ARRAYS:
        if name not in feats or len(feats[name]) != len(df):
            raise DevmError(f"row array {name!r} is missing or has the wrong length")
    for name, a in pix.items():
        if len(a) != len(df):
            raise DevmError(f"pixel array {name!r} has the wrong length")
    missing = [c for c in CONTEXT_META if c not in cmeta.columns] + [n for n in CONTEXT_ARRAYS if n not in carrays]
    if missing:
        raise DevmError(f"context cache: missing chip columns or arrays {missing}")
    CX.check_dev_chips(cmeta)
    counts["context"] = check_context(df, feats["cidx"], feats["cdt"], cmeta)
    d = Path(d)
    CX.write_cache(d / CONTEXT_DIR, cmeta, carrays, allow=allow, info={"k": CX.K, "p_context": P_CONTEXT})
    df.reset_index(drop=True).to_parquet(d / TABLE, index=False)
    np.savez(d / PIXELS, **{k: np.asarray(v) for k, v in pix.items()})
    np.savez(d / FEATURES, **{k: np.asarray(feats[k]) for k in ROW_ARRAYS})
    rec = {**info, "version": VERSION, "base_version": BASE_VERSION, "context_rule": CX.RULE_R7,
           "join_rule": T.RULE_R8, "context_stores": list(CONTEXT_STORES), "counts": counts,
           "sha256": {f: CX._sha(d / f) for f in (TABLE, PIXELS, FEATURES)}}
    (d / INFO).write_text(json.dumps(rec, indent=1, default=str) + "\n")
    return rec


def _reassign(df: pd.DataFrame) -> pd.DataFrame:
    lon, lat = df["lon"].to_numpy(float), df["lat"].to_numpy(float)
    quiet = lambda *a: None  # noqa: E731
    return SP.assign(lon, lat, df["date"].to_numpy(), lakes=T.lakes_readonly(lon, lat, log=quiet), log=quiet)


def load(d: Path | None = None, *, training: bool = True, base: pd.DataFrame | None = None, assign=_reassign):
    """(table, pixels, features, (context meta, context arrays), info) of a dev_v5m cache. Refuses a cache whose
    files differ from the recorded hashes, that holds a protected or non-development row or chip, whose split
    columns differ from the frozen rule (``assign``), or that breaks the context rule. With ``base`` the dev_v5
    equality is checked again."""
    d = cache_dir() if d is None else Path(d)
    info = json.loads((d / INFO).read_text())
    for name, sha in info["sha256"].items():
        if CX._sha(d / name) != sha:
            raise DevmError(f"{d / name}: content differs from the hash recorded when the cache was written")
    df = pd.read_parquet(d / TABLE)
    guard.check_rows(df["protected"].to_numpy(bool), training=training, what=f"{VERSION} cached rows")
    check_dev_rows(df)
    re_ = assign(df)
    for c in ("role", "block", "tblock", "waterbody", "buffer_excluded", "protected"):
        if not np.array_equal(re_[c].to_numpy(), df[c].to_numpy()):
            raise DevmError(f"cached split column {c!r} differs from the frozen rule: rebuild the cache")
    if base is not None:
        check_base_rows(base, df)
    with np.load(d / PIXELS, allow_pickle=False) as z:
        pix = {k: z[k] for k in z.files}
    with np.load(d / FEATURES, allow_pickle=False) as z:
        feats = {k: z[k] for k in z.files}
    cmeta, carrays, _ = CX.load_cache(d / CONTEXT_DIR)
    if len(cmeta):                    # the chips' split columns are re-derived from the frozen rule as well
        rc = assign(cmeta[["lon", "lat", "date"]])
        if rc["protected"].to_numpy(bool).any():
            raise DevmError("the context cache holds a chip the frozen rule protects")
        for c in ("role", "block", "waterbody", "in_lake", "buffer_excluded"):
            if not np.array_equal(rc[c].to_numpy(), cmeta[c].to_numpy()):
                raise DevmError(f"context chip column {c!r} differs from the frozen rule: rebuild the cache")
    check_context(df, feats["cidx"], feats["cdt"], cmeta)
    return df, pix, feats, (cmeta, carrays), info


# ------------------------------------------------------------------------------------ artefact audit (R-21)

def full_chip_ids(store, chip_id) -> np.ndarray:
    """"<store>:<chip id>", the chip id form of the context cache."""
    return (pd.Series(np.asarray(store)).astype(str) + ":" + pd.Series(np.asarray(chip_id)).astype(str)).to_numpy()


def audit_artefacts(d: Path, protected_row_ids, protected_chip_ids) -> dict:
    """Raise if any file under the cache folder ``d`` names a protected row id or a protected chip id.

    ``protected_row_ids``: row ids (as ``row_id``). ``protected_chip_ids``: "<store>:<chip id>". Every file is
    read: each string column of a parquet table, each string array of an npz file and each token of a json file is
    compared with both sets; the row table's (``chip_store``, ``chip_id``) pair is compared as "<store>:<chip id>".
    A file of another type is refused (it could not be audited). Returns counts of what was read."""
    import re

    d = Path(d)
    rows, chips = set(map(str, protected_row_ids)), set(map(str, protected_chip_ids))
    both = rows | chips
    n = {"files": 0, "string_columns": 0, "string_arrays": 0, "json_tokens": 0,
         "n_protected_row_ids": len(rows), "n_protected_chip_ids": len(chips)}

    def hit(values, what):
        k = int(pd.Series(np.asarray(values, dtype=object).ravel()).astype(str).isin(both).sum())
        if k:
            raise DevmError(f"{what}: {k} entries name a protected row or chip id")

    for f in sorted(p for p in d.rglob("*") if p.is_file()):
        rel = f.relative_to(d)
        if f.suffix not in ARTEFACT_SUFFIXES:
            raise DevmError(f"{rel}: not an auditable artefact type {ARTEFACT_SUFFIXES}")
        n["files"] += 1
        if f.suffix == ".parquet":
            t = pd.read_parquet(f)
            for c in t.columns:
                if t[c].dtype == object or pd.api.types.is_string_dtype(t[c]):
                    hit(t[c].dropna().to_numpy(), f"{rel}:{c}")
                    n["string_columns"] += 1
            if {"chip_store", "chip_id"} <= set(t.columns):
                hit(full_chip_ids(t["chip_store"], t["chip_id"]), f"{rel}:chip_store+chip_id")
        elif f.suffix == ".npz":
            with np.load(f, allow_pickle=False) as z:
                for k in z.files:
                    if z[k].dtype.kind in "USO":
                        hit(z[k], f"{rel}:{k}")
                        n["string_arrays"] += 1
        else:
            tok = set(re.findall(r"[A-Za-z0-9_:|.\-]+", f.read_text()))
            n["json_tokens"] += len(tok)
            if tok & both:
                raise DevmError(f"{rel}: {len(tok & both)} tokens name a protected row or chip id")
    return n
