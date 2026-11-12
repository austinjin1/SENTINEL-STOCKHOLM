# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""The M7 keys-only list of lead ruling R-1 / clarification C-1 (docs/preregistration/orbit_arch_v2_lead_rulings.md).

On 2026-11-09 M7's sealed lite/test look scored a Sentinel-2-input SatWQ-Net on site-dates of M2 test-role lakes
dated <= 2023. Every Test-S row that matches a row of M7's list leaves every Test-S cell.

* The list is :data:`LIST_PATH` with sha256 :data:`LIST_SHA256` (110 data rows, key columns only). M2 reads this
  file and nothing else of M7. :func:`load_list` refuses a file whose sha256 differs, whose columns are not exactly
  :data:`COLUMNS` (so no value column can arrive) or whose row count is not :data:`N_ROWS`.
* Match rule (:func:`match`), for any list row: (i) the row's M2 water body equals ``m2_waterbody`` and its
  Sentinel-2 sensing date is within 1 day of ``s2_sensing_date``; or (ii) its site id and sample date equal
  ``site_id`` and ``sample_date``.

Keys only: no label value, prediction or score is read or produced here. The dev_v5m builder refuses to run when
:func:`load_list` fails and writes the number of matched Test-S key rows (a count) to its build counts; the Test-S
look applies :func:`match` to its rows before any cell is formed.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from ...core import paths

LIST_PATH = paths.RUNS_ROOT / "lite" / "counts_arch_v2" / "c2_teacher_scored_m2_test_lake_keys.csv"
LIST_SHA256 = "4786aa121eb673ced502d487fb4af714fff37027677677048d2024c960cfc699"
N_ROWS = 110
COLUMNS = ("m2_waterbody", "s2_sensing_date", "source", "site_id", "sample_date", "s2_chip_id", "lite_role",
           "m2_buffer_excluded", "in_m2_test_s_window")
SENSING_TOL_D = 1
ROW_COLS = ("waterbody", "sensing_date", "site_id", "date")
RULE = ("C-1: a Test-S row leaves every Test-S cell if, for any list row, (i) its M2 water body equals m2_waterbody "
        "and its S2 sensing date is within 1 day of s2_sensing_date, or (ii) its site id and sample date equal "
        "site_id and sample_date")


class M7ListError(RuntimeError):
    """The M7 keys list is missing, changed or not a keys-only table: nothing may be built or scored."""


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_list(path=None, sha256: str = LIST_SHA256, n_rows: int = N_ROWS) -> pd.DataFrame:
    """The list as strings plus parsed day columns ``_s2_day`` and ``_sample_day``. Refuses (``M7ListError``) a
    missing file, another sha256, other columns or another row count. ``path`` / ``sha256`` / ``n_rows`` exist for
    the unit tests; every caller in the repository uses the defaults."""
    p = Path(path) if path is not None else LIST_PATH
    if not p.is_file():
        raise M7ListError(f"M7 keys list {p} does not exist")
    got = file_sha256(p)
    if got != sha256:
        raise M7ListError(f"M7 keys list {p}: sha256 {got} differs from the ruled {sha256}; refused")
    df = pd.read_csv(p, dtype=str, keep_default_na=False)
    if tuple(df.columns) != COLUMNS:
        raise M7ListError(f"M7 keys list: columns {list(df.columns)} are not the ruled key columns {list(COLUMNS)}")
    if len(df) != n_rows:
        raise M7ListError(f"M7 keys list: {len(df)} data rows, the ruling says {n_rows}")
    df["_s2_day"] = _days(df["s2_sensing_date"])
    df["_sample_day"] = _days(df["sample_date"])
    return df


def _days(x) -> np.ndarray:
    """Whole UTC days since 1970-01-01 as float (NaN = no date). Accepts dates, timestamps (with or without a time
    zone) and ISO 8601 strings; a non-empty entry that cannot be read as a date raises (it must not turn silently
    into "no date", which would let a listed row stay)."""
    s = pd.Series(np.asarray(x, dtype=object))
    empty = s.isna().to_numpy() | np.isin(s.astype(str).str.strip().to_numpy(), ("", "nan", "NaT", "None", "<NA>"))
    t = pd.to_datetime(s.where(~empty, None), errors="coerce", utc=True, format="ISO8601")
    bad = t.isna().to_numpy() & ~empty
    if bad.any():
        raise M7ListError(f"{int(bad.sum())} date entries cannot be read as ISO 8601 dates")
    d = t.dt.tz_localize(None).dt.normalize().to_numpy("datetime64[D]")
    return np.where(np.isnat(d), np.nan, d.astype("int64").astype(float))


def match(rows: pd.DataFrame, lst: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """(mask, counts). ``mask[i]`` is True when row i matches any list row under :data:`RULE`.

    ``rows``: :data:`ROW_COLS` — ``waterbody`` (frozen-rule M2 water body), ``sensing_date`` (Sentinel-2 sensing date
    of the row's chip), ``site_id`` and ``date`` (local sample date). A missing key never matches: an empty or NaN
    water body or site id, or a missing date, cannot satisfy its clause. ``counts`` holds numbers of rows only."""
    missing = [c for c in ROW_COLS if c not in rows.columns]
    if missing:
        raise M7ListError(f"match: rows lack the key columns {missing}")
    if "_s2_day" not in lst.columns or "_sample_day" not in lst.columns:
        raise M7ListError("match: the list did not come from load_list")
    n = len(rows)

    def keys(s):
        s = pd.Series(np.asarray(s, dtype=object))
        ok = s.notna().to_numpy()
        out = s.astype(str).str.strip().to_numpy()
        return np.where(ok & ~np.isin(out, ("", "nan", "None", "<NA>")), out, None)

    wb, site = keys(rows["waterbody"]), keys(rows["site_id"])
    s2d, smp = _days(rows["sensing_date"]), _days(rows["date"])
    by_wb: dict = {}
    for w, d in zip(keys(lst["m2_waterbody"]), lst["_s2_day"].to_numpy(float)):
        if w is not None and np.isfinite(d):
            by_wb.setdefault(w, []).append(d)
    pairs = {(s, d) for s, d in zip(keys(lst["site_id"]), lst["_sample_day"].to_numpy(float))
             if s is not None and np.isfinite(d)}
    m1, m2 = np.zeros(n, bool), np.zeros(n, bool)
    for i in range(n):
        if wb[i] is not None and np.isfinite(s2d[i]) and wb[i] in by_wb:
            m1[i] = bool((np.abs(np.asarray(by_wb[wb[i]]) - s2d[i]) <= SENSING_TOL_D).any())
        if site[i] is not None and np.isfinite(smp[i]):
            m2[i] = (site[i], smp[i]) in pairs
    counts = {"n_rows": int(n), "n_matched": int((m1 | m2).sum()), "n_rule_i_waterbody_sensing_date": int(m1.sum()),
              "n_rule_ii_site_sample_date": int(m2.sum()), "n_both": int((m1 & m2).sum())}
    return m1 | m2, counts


def record() -> dict:
    """What a report states about the list (no row content)."""
    return {"path": str(LIST_PATH), "sha256": LIST_SHA256, "n_list_rows": N_ROWS, "rule": RULE}
