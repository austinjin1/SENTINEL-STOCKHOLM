# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""M2 Test-S exclusion flag for SENTINEL-Lite round v2 (conflict C2, lead decisions 2026-11-10).

M2 Test-S (``testxarchive_lake``): in-lake, non-buffer rows of test-role water bodies of orbit split v3 dated
2015-06-23..2023-12-31. It is never used to train or tune a model that takes Sentinel-2 input or that is fused
with M2.

Ruling for M7: a Lite chip on a Test-S site-date keeps its in-situ label and stays in M7 training (the student sees
NAIP only), but its Sentinel-2 teacher pairing carries **no KD term and no teacher-derived feature** in any v2 arm
or comparator; and no M7 output is ever an input, teacher or fused component of anything scored on Test-S.

This module holds the flag only (keys: chip id, coordinates, dates, teacher chip id). It reads no label value.

* ``on_orbit_test_lake`` — in-lake, non-buffer point of an orbit-v3 test-role water body (any date).
* ``on_test_s``          — the same, with the chip's site-date inside the archive window. The site-date of a Lite
  chip is its NAIP acquisition date (the teacher scene is paired to it); a chip is also flagged when its in-situ
  sample date or its teacher scene date falls in the window (the three agree except at the window edges).
* ``kd_masked``          — ``on_test_s`` and the chip has a teacher pairing (``s2_chip_id``).

Scoring rule (lead ruling 2026-11-10 18:26 EST, in force until the lead says M2S/test is sealed): no Sentinel-2-input
model is scored on a row of an M2 test-role water body dated 2023-12-31 or earlier. :func:`teacher_score_blocked`
flags such rows (buffer rows of the water body included, no lower date bound, a row without any date is blocked);
:func:`teacher_chip_ids` gives the teacher chips that may be predicted and :func:`assert_teacher_scorable` is the
hard failure in front of the teacher call.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ARCHIVE_FROM, ARCHIVE_TO = pd.Timestamp("2015-06-23"), pd.Timestamp("2023-12-31")
DATE_COLS = ("naip_date", "sample_date", "s2_date")
FLAG_COLS = ("on_orbit_test_lake", "on_test_s", "on_test_s_naip_date", "kd_masked")


class MaskError(RuntimeError):
    """A teacher-derived quantity would reach a row on an M2 Test-S site-date."""


def _orbit_assign(lon, lat) -> pd.DataFrame:
    """Orbit split v3 columns per point. Geometry only; M2's lake cache is read-free and write-free here."""
    from ..orbit import split as OS

    quiet = lambda *_a, **_k: None  # noqa: E731
    lakes = OS.assign_lakes(lon, lat, use_cache=False, log=quiet)
    dummy = np.full(len(lon), np.datetime64("2019-07-01"))          # role and buffer do not depend on the date
    return OS.assign(lon, lat, dummy, lakes=lakes, log=quiet)


def _in_window(dates) -> np.ndarray:
    d = pd.to_datetime(pd.Series(np.asarray(dates)), errors="coerce")
    if getattr(d.dt, "tz", None) is not None:
        d = d.dt.tz_localize(None)
    return ((d >= ARCHIVE_FROM) & (d <= ARCHIVE_TO)).fillna(False).to_numpy(bool)


def flags(chips: pd.DataFrame, *, assign=None) -> pd.DataFrame:
    """Flag table (``naip_chip_id`` + :data:`FLAG_COLS`) for Lite chips.

    ``chips``: naip_chip_id, lat, lon, naip_date, s2_chip_id and, when present, sample_date / s2_date.
    ``assign``: (lon, lat) -> frame with ``in_lake``, ``role``, ``buffer_excluded`` (default: orbit split v3)."""
    need = ["naip_chip_id", "lat", "lon", "naip_date", "s2_chip_id"]
    miss = [c for c in need if c not in chips.columns]
    if miss:
        raise MaskError(f"column(s) {miss} missing")
    if chips["naip_chip_id"].duplicated().any():
        raise MaskError("duplicate chip ids")
    lon, lat = chips["lon"].to_numpy(float), chips["lat"].to_numpy(float)
    a = (assign or _orbit_assign)(lon, lat)
    if len(a) != len(chips):
        raise MaskError("the split assignment returned a different number of rows")
    lake = (a["in_lake"].to_numpy(bool) & (a["role"].astype(str).to_numpy() == "test")
            & ~a["buffer_excluded"].to_numpy(bool))
    win_naip = _in_window(chips["naip_date"])
    win_any = win_naip.copy()
    for c in DATE_COLS[1:]:
        if c in chips.columns:
            win_any |= _in_window(chips[c])
    out = pd.DataFrame({"naip_chip_id": chips["naip_chip_id"].astype(str).to_numpy()})
    out["on_orbit_test_lake"] = lake
    out["on_test_s"] = lake & win_any
    out["on_test_s_naip_date"] = lake & win_naip
    out["kd_masked"] = out["on_test_s"].to_numpy() & chips["s2_chip_id"].notna().to_numpy()
    return out


def counts(flag_table: pd.DataFrame, roles: pd.Series | None = None) -> dict:
    """Counts for run records (per role when ``roles`` maps chip id -> role)."""
    def one(g):
        return {"chips": int(len(g)), **{c: int(g[c].sum()) for c in FLAG_COLS}}

    if roles is None:
        return one(flag_table)
    r = flag_table["naip_chip_id"].map(roles).fillna("no role")
    return {str(k): one(g) for k, g in flag_table.groupby(r.to_numpy())}


def mask_teacher(table: pd.DataFrame, flag_table: pd.DataFrame, teacher_cols, *, kd_weight_col: str | None = None
                 ) -> tuple[pd.DataFrame, dict]:
    """Copy of ``table`` with every teacher-derived column set to NaN (and the KD weight to 0) on ``kd_masked`` chips.

    Every chip of ``table`` must be in the flag table: an unflagged chip is refused, not treated as unmasked."""
    f = flag_table.drop_duplicates("naip_chip_id").set_index("naip_chip_id")["kd_masked"]
    ids = table["naip_chip_id"].astype(str)
    if not ids.isin(f.index).all():
        raise MaskError(f"{int((~ids.isin(f.index)).sum())} chips have no Test-S flag")
    m = f.reindex(ids).to_numpy(bool)
    out = table.copy()
    n_set = {}
    for c in teacher_cols:
        if c not in out.columns:
            raise MaskError(f"teacher column {c!r} missing")
        n_set[c] = int((m & out[c].notna().to_numpy()).sum())
        out[c] = out[c].astype(float).where(~m, np.nan)
    if kd_weight_col is not None:
        out[kd_weight_col] = np.where(m, 0.0, out[kd_weight_col].astype(float))
    return out, {"n_rows": int(len(out)), "n_kd_masked": int(m.sum()), "n_teacher_values_removed": n_set}


def assert_masked(table: pd.DataFrame, flag_table: pd.DataFrame, teacher_cols, *, kd_weight_col: str | None = None
                  ) -> None:
    """Raise unless no teacher-derived value and no positive KD weight sits on a ``kd_masked`` chip."""
    f = flag_table.drop_duplicates("naip_chip_id").set_index("naip_chip_id")["kd_masked"]
    ids = table["naip_chip_id"].astype(str)
    if not ids.isin(f.index).all():
        raise MaskError(f"{int((~ids.isin(f.index)).sum())} chips have no Test-S flag")
    m = f.reindex(ids).to_numpy(bool)
    for c in teacher_cols:
        n = int((m & table[c].notna().to_numpy()).sum())
        if n:
            raise MaskError(f"{n} teacher values in {c!r} on M2 Test-S site-dates")
    if kd_weight_col is not None:
        n = int((m & (table[kd_weight_col].to_numpy(float) > 0)).sum())
        if n:
            raise MaskError(f"{n} rows on M2 Test-S site-dates carry a KD weight")


# --------------------------------------------------------------------------- scoring an S2-input model (teacher)
def teacher_score_blocked(rows: pd.DataFrame, *, assign=None) -> np.ndarray:
    """True where no Sentinel-2-input model may be scored on the row (see the module text).

    ``rows``: lat, lon and at least one of ``naip_date`` / ``sample_date`` / ``s2_date``; ``dt_days_s2_naip`` (when
    present) moves the NAIP date back by its absolute value, so the teacher scene date is covered without reading
    the teacher store. Keys only."""
    for c in ("lat", "lon"):
        if c not in rows.columns:
            raise MaskError(f"column {c!r} missing")
    have = [c for c in DATE_COLS if c in rows.columns]
    if not have:
        raise MaskError(f"none of the date columns {DATE_COLS} is present")
    a = (assign or _orbit_assign)(rows["lon"].to_numpy(float), rows["lat"].to_numpy(float))
    if len(a) != len(rows):
        raise MaskError("the split assignment returned a different number of rows")
    lake = a["in_lake"].to_numpy(bool) & (a["role"].astype(str).to_numpy() == "test")
    dates = []
    for c in have:
        d = pd.to_datetime(pd.Series(rows[c].to_numpy()), errors="coerce", utc=True).dt.tz_localize(None)
        dates.append(d)
        if c == "naip_date" and "dt_days_s2_naip" in rows.columns:
            lag = pd.to_timedelta(np.ceil(np.abs(pd.to_numeric(pd.Series(rows["dt_days_s2_naip"].to_numpy()),
                                                               errors="coerce")).fillna(0.0)), unit="D")
            dates.append(d - lag)
    first = pd.concat(dates, axis=1).min(axis=1)
    early = (first.isna() | (first <= ARCHIVE_TO)).to_numpy(bool)
    return lake & early


def teacher_chip_ids(rows: pd.DataFrame, blocked: np.ndarray) -> list[str]:
    """Teacher (Sentinel-2) chip ids that may be predicted: paired to at least one row and to no blocked row."""
    ids = rows["s2_chip_id"]
    bad = set(ids[np.asarray(blocked, bool) & ids.notna().to_numpy()].astype(str))
    return sorted(set(ids.dropna().astype(str)) - bad)


def assert_teacher_scorable(rows: pd.DataFrame, chip_ids, blocked: np.ndarray) -> None:
    """Raise :class:`MaskError` if a teacher chip about to be predicted is paired to a blocked row."""
    blocked = np.asarray(blocked, bool)
    if len(blocked) != len(rows):
        raise MaskError("the blocked flag does not match the rows")
    hit = rows["s2_chip_id"].astype("string").isin([str(c) for c in chip_ids]).fillna(False).to_numpy() & blocked
    if hit.any():
        raise MaskError(f"{int(hit.sum())} rows of M2 test-role water bodies dated <= {ARCHIVE_TO.date()} would be "
                        "scored with a Sentinel-2-input model")
