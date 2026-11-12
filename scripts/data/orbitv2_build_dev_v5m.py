# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Build data version dev_v5m (docs/preregistration/orbit_arch_v2_amendment.md §4, as ruled in
docs/preregistration/orbit_arch_v2_lead_rulings.md; where the two differ, the rulings win).

dev_v5m = every dev_v5 row unchanged + new-target columns (TP, TN, DOC, turbidity-FNU, SSC, aCDOM440: value,
censoring flag, detection limit) + per-pixel shore distance, land ring and glint features + the lake-context index,
plus chip-bearing, QC-passing site-dates that carry only new-target values.

Two modes:

  --mode keys    counts only. Reads site, date, depth, fraction and censoring keys; no value column and no detection
                 limit is read from any new-target table, and no chip array is read. Prints how many development
                 rows could carry each new target (an upper bound) and the context availability.
  --mode build   reads new-target values for development sites only (targets.read_results refuses anything else),
                 extracts the chip features and writes the cache through devm.write, which re-checks every rule.

Rules that are constants here, with no switch (lead clarification C-2):

  R-7   a context chip has the row's frozen-rule water body and split role (context.RULE_R7); the context stores
        are s2_chips/aquacorr and s2_chips/wqp only, content-pinned at the first build (context.write_pin); a chip
        or store added after the pin is excluded.
  R-8   the 100 m join of an AquaCorr row goes to one WQP station, the nearest whose frozen-rule water body and
        split role equal the row's, ties by the lower station id, chosen from keys (targets.RULE_R8, C-9).
  R-1   the M7 keys list (m7keys.py; C-1) must have the ruled sha256, otherwise nothing runs. The number of Test-S
        key rows it removes is written to the build counts (a count; no value).
  R-9   AquaCorr's own doc_value / acdom440_value are used only if AquaCorr's metadata state the unit explicitly
        (aquacorr_unit_check: metadata only). They do not, so no AquaCorr value column of a new target is read.
  R-21  split roles are assigned from keys first; a new-target value column is read only for development sites and
        development dates (targets.read_results refuses anything else); the context cache is built from the
        development allow-list; after writing, every file of the cache is searched for protected row and chip ids
        (devm.audit_artefacts) and the cache is loaded once through the refusing loader.
  R-25  a build needs >= 60 GB free on / after its projected size and writes < 10 GB.

Protected rows never enter: dev_v5 holds none, added rows are dropped by keys right after split assignment, and the
context cache is built from the development allow-list. Keys of protected rows (site, date, chip id, split columns)
are held in memory for the Test-S key counts and the audit; no value of theirs is read and prints give counts only.
The shared lake-assignment cache is not written.

    python -B scripts/data/orbitv2_build_dev_v5m.py --mode keys
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths  # noqa: E402
from sentinel.models.orbit import data as D  # noqa: E402
from sentinel.models.orbit import split as SP  # noqa: E402
from sentinel.models.orbit_v2 import chipfeat as F  # noqa: E402
from sentinel.models.orbit_v2 import context as CX  # noqa: E402
from sentinel.models.orbit_v2 import devm as DM  # noqa: E402
from sentinel.models.orbit_v2 import m7keys as M7  # noqa: E402
from sentinel.models.orbit_v2 import targets as T  # noqa: E402

DEDUPE_RADIUS_M = 100.0       # protocol §2.1: one row per date within 100 m (dev_v5 rows are kept, added rows give way)
SAMPLE_KEYS = ["point_key", "date", "site_id", "lat", "lon", "provider", "jrc_water_ok"]
PROT_KEY_COLS = ["source", "row_id", "chip_store", "chip_id", "site_id", "date", "sensing_time_utc", "item_id",
                 "waterbody", "in_lake", "block", "role", "tblock", "buffer_excluded", "protected"]
MIN_FREE_GB_AFTER = 60.0      # R-25
MAX_ARTEFACT_GB = 10.0        # R-25
AQUACORR_UNIT_TARGETS = {"doc": "doc_value", "acdom440": "acdom440_value"}      # R-9


MATCHUP_KEYS = ("point_key", "date", "request_key", "win_same_day", "chip_ok")
VALUE_LIKE = ("value", "value_min", "value_max", "detection_limit", "censored", "n_censored", "target",
              "characteristic", "unit")


def matchup_columns(names, want) -> list:
    """Columns of a matchup table that ``added_rows`` reads: the wanted key and image columns that exist, in file
    order. Never a sample-value column (:data:`VALUE_LIKE`), whatever ``want`` says."""
    bad = sorted(set(want) & set(VALUE_LIKE))
    if bad:
        raise DM.DevmError(f"added_rows was asked for value columns {bad}")
    missing = [c for c in MATCHUP_KEYS if c not in names]
    if missing:
        raise DM.DevmError(f"matchup table lacks the key columns {missing}")
    return [c for c in names if c in set(want)]


def _row_ids(source, row_src_id, chip_id) -> list:
    return [hashlib.sha1(f"{s}|{i}|{c}".encode()).hexdigest()[:16] for s, i, c in zip(source, row_src_id, chip_id)]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------------------------ added rows (keys only)

def added_rows(base: pd.DataFrame, sets=None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(candidates, protected keys, counts). Candidates: chip-bearing, image-QC-passing, non-protected WQP
    point-dates that are not dev_v5 rows. Built from keys and image metadata: no sample value is read. The chip of
    a point-date is chosen as data.wqp_rows does. Protected keys (:data:`PROT_KEY_COLS`): the point-dates the frozen
    rule protects, dropped here right after the split assignment; key columns only."""
    import pyarrow.parquet as pq

    ren = {f"{c}_chip": c for c in ("scl_c3_n_water", "scl_c7_n_water", "scl_c3_cloud_medium", "scl_c3_cloud_high",
                                     "scl_c3_thin_cirrus", "scl_c3_cloud_shadow")}
    want = {*MATCHUP_KEYS, *D.IMG_COLS, *ren, *ren.values()}
    smp, mt = [], []
    for name in (sets or D.WQP_MATCHUP_SETS):
        d = paths.PROCESSED / name
        if not ((d / "samples.parquet").exists() and (d / "matchups.parquet").exists()):
            continue
        smp.append(pd.read_parquet(d / "samples.parquet", columns=SAMPLE_KEYS))
        cols = matchup_columns(pq.read_schema(d / "matchups.parquet").names, want)
        m = pd.read_parquet(d / "matchups.parquet", columns=cols)           # key and image columns by name
        mt.append(m[m.win_same_day & m.chip_ok])
    smp = pd.concat(smp, ignore_index=True)
    smp = smp[smp.jrc_water_ok.astype("boolean").fillna(False)]
    smp["date"] = pd.to_datetime(smp.date).dt.normalize()
    meta = smp.sort_values(["point_key", "date", "site_id"]).drop_duplicates(["point_key", "date"])
    mt = pd.concat(mt, ignore_index=True)
    mt["date"] = pd.to_datetime(mt.date).dt.normalize()
    mt = mt.drop(columns=[v for v in ren.values() if v in mt.columns]).rename(columns=ren)
    mt = mt.sort_values(["point_key", "date", "scl_c3_n_water", "request_key"], ascending=[True, True, False, True])
    mt = mt.drop_duplicates(["point_key", "date"])
    idx = D._index("wqp")[["request_key", "chip_id", "shard", "platform", "sensing_time_utc", "scene_cloud_cover",
                           "item_id"]]
    keep = [c for c in D.IMG_COLS if c in mt.columns and c not in idx.columns]
    mt = mt[["point_key", "date", "request_key"] + keep].merge(idx, on="request_key", how="inner")
    df = meta.drop(columns="jrc_water_ok").merge(mt, on=["point_key", "date"], how="inner")
    out = pd.DataFrame({"source": "wqp", "row_src_id": (df.point_key + "|" + df.date.dt.strftime("%Y-%m-%d")).to_numpy(),
                        "site_id": df.site_id.astype(str).to_numpy(), "provider": df.provider.astype(str).to_numpy(),
                        "lat": df.lat.astype(float).to_numpy(), "lon": df.lon.astype(float).to_numpy(),
                        "date": df.date.to_numpy(), "chip_store": "wqp", "same_acquisition": True})
    for c in D.IMG_COLS:
        if c in df:
            out[c] = df[c].to_numpy()
    counts = {"wqp_point_dates_with_same_day_chip": int(len(out))}
    out = out[~out.row_src_id.isin(set(base.loc[base.source == "wqp", "row_src_id"]))]
    counts["not_in_base"] = int(len(out))
    out = out[D.image_qc(out).to_numpy()].reset_index(drop=True)
    counts["image_qc_pass"] = int(len(out))
    lon, lat = out.lon.to_numpy(), out.lat.to_numpy()
    sp = SP.assign(lon, lat, out.date.to_numpy(), lakes=T.lakes_readonly(lon, lat, log=log), log=lambda *a: None)
    out = pd.concat([out, sp], axis=1)
    counts["protected_dropped"] = int(out.protected.sum())                  # count only
    prot = out[out.protected.to_numpy(bool)].copy()
    prot["row_id"] = _row_ids(prot.source, prot.row_src_id, prot.chip_id)
    prot = prot[PROT_KEY_COLS].reset_index(drop=True)
    out = out[~out.protected.to_numpy(bool) & ~out.buffer_excluded.to_numpy(bool)].reset_index(drop=True)
    counts["non_protected_non_buffer"] = int(len(out))
    # a dev_v5 row on the same date within 100 m already represents the site-date
    both = pd.concat([base[["lat", "lon", "date"]], out[["lat", "lon", "date"]]], ignore_index=True)
    comp = D._components_within(both.lat.to_numpy(), both.lon.to_numpy(), both.date.astype(str).to_numpy(),
                                DEDUPE_RADIUS_M)
    dup = np.isin(comp[len(base):], comp[:len(base)])
    out = out[~dup].reset_index(drop=True)
    counts["after_dedupe_against_base"] = int(len(out))
    out["n_merged"] = 1
    out["doy"] = pd.to_datetime(out.date).dt.dayofyear
    out["year"] = pd.to_datetime(out.date).dt.year
    out["row_id"] = _row_ids(out.source, out.row_src_id, out.chip_id)
    return out, prot, counts


def aquacorr_protected_keys() -> pd.DataFrame:
    """Key columns (:data:`PROT_KEY_COLS`) of the chip-bearing, image-QC-passing AquaCorr rows the frozen rule
    protects. Rows are assembled as data.aquacorr_rows does, without any target column: no value is read."""
    proc = paths.processed("aquacorr_global")
    sel = pd.read_parquet(proc / "s2_chip_requests.parquet",
                          columns=["matchup_id", "site_id", "lat", "lon", "date_insitu", "sensor", "chip_request_key"])
    plan = pd.read_parquet(proc / "s2_chip_plan.parquet", columns=["chip_request_key", "match_kind"])
    idx = D._index("aquacorr").rename(columns={"request_key": "chip_request_key"})
    df = sel.merge(plan, on="chip_request_key", how="left").merge(
        idx[["chip_request_key"] + [c for c in D.IMG_COLS if c in idx.columns]], on="chip_request_key", how="inner")
    st = pd.to_datetime(df.sensing_time_utc, utc=True, errors="coerce")
    same_acq = (df.platform == df.sensor) & (st.dt.strftime("%Y-%m-%d") == pd.to_datetime(df.date_insitu).dt.strftime(
        "%Y-%m-%d")) & df.match_kind.isin(["designated", "alternate_tile"])
    a = pd.DataFrame({"source": "aquacorr", "row_src_id": df.matchup_id.astype(str), "site_id": df.site_id.astype(str),
                      "lat": df.lat.astype(float), "lon": df.lon.astype(float),
                      "date": pd.to_datetime(df.date_insitu).dt.normalize(), "chip_store": "aquacorr",
                      "same_acquisition": same_acq.to_numpy()})
    for c in D.IMG_COLS:
        if c in df:
            a[c] = df[c].to_numpy()
    a = a[D.image_qc(a).to_numpy()].reset_index(drop=True)
    lon, lat = a.lon.to_numpy(), a.lat.to_numpy()
    sp = SP.assign(lon, lat, a.date.to_numpy(), lakes=T.lakes_readonly(lon, lat, log=log), log=lambda *x: None)
    a = pd.concat([a, sp], axis=1)
    a = a[a.protected.to_numpy(bool)].copy()
    a["row_id"] = _row_ids(a.source, a.row_src_id, a.chip_id)
    return a[PROT_KEY_COLS].reset_index(drop=True)


def _cell_count(g: pd.DataFrame) -> dict:
    sd = g.waterbody.astype(str) + "|" + g.site_id.astype(str) + "|" + pd.to_datetime(g.date).dt.strftime("%Y-%m-%d")
    return {"n_key_rows": int(len(g)), "n_site_dates": int(sd.nunique()),
            "n_waterbodies": int(g.waterbody.nunique()), "n_role_blocks": int(g.block.nunique())}


def c1_key_counts(prot: pd.DataFrame, lst: pd.DataFrame) -> dict:
    """C-1 / R-1, keys only: Test-S key rows (test role x in-lake x not buffer-excluded x development years) before
    and after the rows matching M7's keys list leave. Counts only; these are chip-bearing, image-QC-passing key rows
    of any target before the label rules, so each is an upper bound of the scored rows."""
    ts = prot[(prot.role.astype(str) == "test") & prot.in_lake.to_numpy(bool) & ~prot.buffer_excluded.to_numpy(bool)
              & prot.tblock.astype(str).isin(T.DEV_TBLOCKS).to_numpy()].reset_index(drop=True)
    m, mc = M7.match(pd.DataFrame({"waterbody": ts.waterbody.astype(str).to_numpy(),
                                   "sensing_date": DM.sensing_date(ts.sensing_time_utc).to_numpy(),
                                   "site_id": ts.site_id.astype(str).to_numpy(), "date": ts.date.to_numpy()}), lst)
    return {**M7.record(), "definition": "Test-S key rows: test role x in-lake x not buffer-excluded x sample date in a "
            "development year block; chip-bearing and image-QC-passing; any target; keys only",
            "before": {"all": _cell_count(ts), **{s: _cell_count(g) for s, g in ts.groupby("source")}},
            "matched_and_removed": mc,
            "removed_by_source": {s: int(m[(ts.source == s).to_numpy()].sum()) for s in sorted(set(ts.source))},
            "after": {"all": _cell_count(ts[~m]), **{s: _cell_count(g) for s, g in ts[~m].groupby("source")}}}


def aquacorr_unit_check() -> dict:
    """R-9: are the units of AquaCorr's doc_value and acdom440_value stated explicitly in AquaCorr's metadata?

    Reads metadata only: the column names of the published matchup table (a ``<target>_unit`` column, as the four
    base targets have) and the ``parameters`` entry of ``aquacorr_metadata.json`` (a key naming a unit). No row is
    read. Returns, per target, what was found and ``unit_stated``."""
    import pyarrow.parquet as pq

    raw = paths.RAW / "aquacorr_global"
    names = set(pq.read_schema(raw / "aquacorr_matchups.parquet").names)
    meta = json.loads((raw / "aquacorr_metadata.json").read_text())
    params = meta.get("parameters", {}) if isinstance(meta, dict) else {}
    out = {}
    for t, col in AQUACORR_UNIT_TARGETS.items():
        p = params.get(t, {}) if isinstance(params, dict) else {}
        unit_keys = sorted(k for k in (p if isinstance(p, dict) else {}) if "unit" in str(k).lower())
        has_col = f"{t}_unit" in names
        out[t] = {"value_column_present": col in names, "unit_column_present": has_col,
                  "metadata_parameter_keys": sorted(map(str, p)) if isinstance(p, dict) else [],
                  "metadata_unit_keys": unit_keys, "unit_stated": bool(has_col or unit_keys)}
    out["outcome"] = ("AquaCorr's metadata state no unit for doc_value or acdom440_value: under R-9 these values are "
                      "used neither for fitting nor for scoring" if not any(out[t]["unit_stated"] for t in
                                                                           AQUACORR_UNIT_TARGETS)
                      else "a unit is stated for at least one target")
    return out


def drop_protected_ids(rows: pd.DataFrame, cand: pd.DataFrame, prot_row_ids, prot_chip_ids):
    """(rows, candidates, counts) after R-21's id rule, from keys only. Context candidates whose chip id is
    protected (by the frozen rule or because a protected key row uses the chip) are dropped; added rows with a
    protected row id or on such a chip are dropped; dev_v5 rows are kept (one to one) and only counted."""
    prot_row_ids, prot_chip_ids = set(map(str, prot_row_ids)), set(map(str, prot_chip_ids))
    bad = rows.row_id.astype(str).isin(prot_row_ids).to_numpy() | pd.Series(
        DM.full_chip_ids(rows.chip_store, rows.chip_id)).isin(prot_chip_ids).to_numpy()
    added = rows.added.to_numpy(bool)
    cbad = cand.chip_id.astype(str).isin(prot_chip_ids).to_numpy()
    counts = {"n_base_rows_on_a_protected_id": int((bad & ~added).sum()),
              "n_added_rows_on_a_protected_id_dropped": int((bad & added).sum()),
              "n_context_candidates_protected_dropped": int(cbad.sum()),
              "of_which_not_protected_by_their_own_keys": int((cbad & ~cand.protected.to_numpy(bool)).sum())}
    return rows[~(bad & added)].reset_index(drop=True), cand[~cbad].reset_index(drop=True), counts


def check_no_protected_ids(rows: pd.DataFrame, cmeta: pd.DataFrame, prot_row_ids, prot_chip_ids) -> None:
    """R-21, before anything is written: no row id, row chip or context chip of the build is a protected id.
    Counts only in the message."""
    prot_row_ids, prot_chip_ids = set(map(str, prot_row_ids)), set(map(str, prot_chip_ids))
    n_row = int(rows.row_id.astype(str).isin(prot_row_ids).sum())
    n_chip = int(pd.Series(DM.full_chip_ids(rows.chip_store, rows.chip_id)).isin(prot_chip_ids).sum()) \
        if "chip_store" in rows.columns else 0
    n_ctx = int(cmeta.chip_id.astype(str).isin(prot_chip_ids).sum())
    if n_row or n_chip or n_ctx:
        raise DM.DevmError(f"protected ids in the build: {n_row} row ids, {n_chip} row chips, {n_ctx} context chips; "
                           "nothing was written")


# ------------------------------------------------------------------------------------ new-target join

def join_sites(rows: pd.DataFrame, stations: pd.DataFrame | None = None) -> tuple[np.ndarray, pd.DataFrame, dict]:
    """(join site per row (None = none), development site table, counts), from keys only.

    WQP rows: the row's own site, kept only when the station's own coordinates give, under the frozen rule, the
    row's water body, a development role equal to the row's and no buffer exclusion (R-21: the role of every site whose values will be
    read is decided from keys first). AquaCorr rows: targets.nearest_station (R-8 / C-9). Buffer-excluded rows get
    no site. ``stations``: the WQP station table (default: wqp_discrete/_stations.parquet; key columns only)."""
    site = np.full(len(rows), None, dtype=object)
    buf = rows.buffer_excluded.to_numpy(bool)
    r_role = rows.role.astype(str).to_numpy()
    dev_row = np.isin(r_role, T.DEV_ROLES) & ~buf
    if stations is None:
        stations = pd.read_parquet(paths.PROCESSED / "wqp_discrete" / "_stations.parquet",
                                   columns=["site_id", "lat", "lon", "lat_std", "lon_std"])
    stations = stations.drop_duplicates("site_id")
    stations = stations.assign(site_id=stations.site_id.astype(str))
    wq = (rows.source == "wqp").to_numpy() & dev_row
    r_site = rows.site_id.astype(str).to_numpy()
    counts = {"wqp_rows_development_non_buffer": int(wq.sum())}
    st_w = pd.DataFrame(columns=["site_id", "role", "buffer_excluded"])
    if wq.any():
        wroles = T.site_roles(stations[stations.site_id.isin(set(r_site[wq]))], log=log).set_index("site_id")
        s_role = wroles.role.reindex(r_site).fillna("").astype(str).to_numpy()
        s_wb = wroles.waterbody.reindex(r_site).fillna("").astype(str).to_numpy()
        s_buf = wroles.buffer_excluded.reindex(r_site).astype("boolean").fillna(True).to_numpy(bool)
        okw = wq & (s_role == r_role) & (s_wb == rows.waterbody.astype(str).to_numpy()) & ~s_buf
        site[okw] = r_site[okw]
        counts["wqp_rows_with_site"] = int(okw.sum())
        counts["wqp_rows_station_role_or_waterbody_differs_unknown_or_buffer_no_site"] = int((wq & ~okw).sum())
        st_w = pd.DataFrame({"site_id": r_site[okw], "role": r_role[okw], "buffer_excluded": False}).drop_duplicates()
    else:
        counts["wqp_rows_with_site"] = 0
    aq = np.flatnonzero((rows.source == "aquacorr").to_numpy() & dev_row)
    st_dev = pd.DataFrame(columns=["site_id", "role", "buffer_excluded"])
    if aq.size:
        from scipy.spatial import cKDTree

        st = stations
        slat, slon = st.lat_std.fillna(st.lat).to_numpy(float), st.lon_std.fillna(st.lon).to_numpy(float)
        ok = np.isfinite(slat) & np.isfinite(slon)
        st, slat, slon = st[ok].reset_index(drop=True), slat[ok], slon[ok]
        a = rows.iloc[aq]
        tree = cKDTree(T._unit_xyz(a.lon.to_numpy(float), a.lat.to_numpy(float)))
        chord = 2.0 * np.sin(T.JOIN_RADIUS_M * 1.05 / 6_371_008.8 / 2.0)
        near = np.array([len(x) > 0 for x in tree.query_ball_point(T._unit_xyz(slon, slat), chord)])
        counts["stations_within_100m_of_an_aquacorr_row"] = int(near.sum())
        if near.any():
            roles = T.site_roles(st[near], log=log)
            dev = roles[roles.role.isin(T.DEV_ROLES) & ~roles.buffer_excluded.astype(bool)].reset_index(drop=True)
            counts["of_which_development_stations"] = int(len(dev))
            j = T.nearest_station(a[["lon", "lat", "waterbody", "role"]], dev)
            got = j.join_site_id.notna().to_numpy()
            if got.any():                           # R-8 is re-checked on the result, not assumed
                d = dev.set_index("site_id")
                same = (d.role.reindex(j.join_site_id[got]).to_numpy() == a.role.to_numpy()[got]) & (
                    d.waterbody.reindex(j.join_site_id[got]).to_numpy() == a.waterbody.to_numpy()[got])
                if not same.all():
                    raise DM.DevmError("a joined station has another water body or split role than its row")
            site[aq[got]] = j.join_site_id.to_numpy()[got]
            counts["aquacorr_rows_with_station"] = int(got.sum())
            st_dev = dev.loc[dev.site_id.isin(set(j.join_site_id[got])), ["site_id", "role", "buffer_excluded"]]
    sites = pd.concat([st_w, st_dev], ignore_index=True).drop_duplicates(["site_id", "role"])
    if sites.site_id.duplicated().any():            # cannot happen: every role above is the station's own role
        raise DM.DevmError("a site id appears with two split roles")
    counts["rule"] = T.RULE_R8
    return site, sites.reset_index(drop=True), counts


def join_targets(rows: pd.DataFrame, site: np.ndarray, sites: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, dict]:
    """New-target columns for ``rows``. ``keys`` mode: presence only (values NaN), from key columns."""
    r = pd.DataFrame({"site_id": pd.Series(site, index=rows.index).fillna("").astype(str), "date": rows.date})
    out, counts = {}, {}
    for t in T.SPECS:
        if not len(sites):
            out[t], out[f"{t}_cens"], out[f"{t}_dl"] = np.full(len(rows), np.nan), False, np.full(len(rows), np.nan)
            counts[t] = {"rows_with_a_surface_result_key": 0} if mode == "keys" else {"rows_quantified": 0,
                                                                                    "rows_left_censored": 0}
        elif mode == "keys":
            keys = T.read_keys(t, sites.site_id.tolist(), date_max=T.dev_date_max())
            kc = T.key_counts(keys, t)
            have = pd.MultiIndex.from_frame(kc[["site_id", "date"]].assign(date=pd.to_datetime(kc.date).dt.normalize()))
            hit = pd.MultiIndex.from_arrays([r.site_id.to_numpy(), pd.to_datetime(r.date).dt.normalize().to_numpy()]
                                            ).isin(have)
            out[t], out[f"{t}_cens"], out[f"{t}_dl"] = np.full(len(rows), np.nan), hit, np.full(len(rows), np.nan)
            counts[t] = {"rows_with_a_surface_result_key": int(hit.sum())}
        else:
            lab, c = T.dev_labels(t, sites)
            j = T.join_site_date(r, lab, t)
            for col in j.columns:
                out[col] = j[col].to_numpy()
            counts[t] = {**c, "rows_quantified": int(j[t].notna().sum()), "rows_left_censored": int(j[f"{t}_cens"].sum())}
    return pd.DataFrame(out, index=rows.index), counts


# ------------------------------------------------------------------------------------ chip arrays

def _row_shard(args):
    store, shard, chip_ids, offsets, n_pix = args
    z = np.load(paths.processed("s2_chips") / store / "shards" / shard)
    try:
        return [(cid,) + F.chip_features(z[cid], off, int(n)) for cid, off, n in zip(chip_ids, offsets, n_pix)]
    finally:
        z.close()


def _ctx_shard(args):
    store, shard, chip_ids = args
    z = np.load(paths.processed("s2_chips") / store / "shards" / shard)
    try:
        return [(cid,) + DM.context_chip_arrays(z[cid]) for cid in chip_ids]
    finally:
        z.close()


def row_features(df: pd.DataFrame, pix: dict, workers: int) -> dict:
    jobs = [(st, sh, g.chip_id.tolist(), pix["offsets"][g.index.to_numpy()], pix["n_pix"][g.index.to_numpy()])
            for (st, sh), g in df.reset_index(drop=True).groupby(["chip_store", "shard"])]
    res = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, part in enumerate(ex.map(_row_shard, jobs)):
            res.update({cid: (dp, ring) for cid, dp, ring in part})
            if (i + 1) % 50 == 0:
                log(f"[devm] row features: {i + 1}/{len(jobs)} shards")
    return {"dp_m": np.stack([res[c][0] for c in df.chip_id]).astype(np.float32),
            "ring": np.stack([res[c][1] for c in df.chip_id]).astype(np.float32),
            "glint": F.glint_features(df.sun_zenith, df.view_zenith, df.sun_azimuth, df.view_azimuth).astype(np.float32)}


def context_arrays(cmeta: pd.DataFrame, workers: int) -> dict:
    jobs = [(st, sh, g.store_chip_id.tolist()) for (st, sh), g in cmeta.groupby(["store", "shard"])]
    res = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for (st, _, _), part in zip(jobs, ex.map(_ctx_shard, jobs)):
            res.update({f"{st}:{cid}": rest for cid, *rest in part})
    get = lambda i: [res[c][i] for c in cmeta.chip_id]  # noqa: E731
    return {"refl": np.stack(get(0)).astype(np.float16), "offsets": np.stack(get(1)).astype(np.int8),
            "n_pix": np.asarray(get(2), np.int32), "dp_m": np.stack(get(3)).astype(np.float32),
            "rw3": cmeta[list(DM.RW3_COLS)].to_numpy(np.float32),     # exact 3 x 3 water medians (chip index)
            "q": F.date_quality(cmeta.sun_zenith, cmeta.view_zenith, cmeta.sun_azimuth, cmeta.view_azimuth, cmeta.aot,
                                cmeta.processing_baseline)}


# ------------------------------------------------------------------------------------ main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("keys", "build"), required=True)
    ap.add_argument("--base-version", default=DM.BASE_VERSION)
    ap.add_argument("--out", default=None, help="cache folder (default: the dev_v5m cache)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: the first N base rows (needs --out)")
    ap.add_argument("--no-added", action="store_true", help="smoke test: no added rows (needs --out)")
    ap.add_argument("--sets", default="", help="smoke test: comma-separated WQP matchup sets for added rows")
    ap.add_argument("--counts-json", default=None)
    a = ap.parse_args(argv)
    lst = M7.load_list()            # C-1: refuses to run when the M7 keys list is missing or its sha256 differs
    smoke = bool(a.limit or a.no_added or a.sets or a.base_version != DM.BASE_VERSION)
    out = Path(a.out) if a.out else DM.cache_dir()
    if smoke and a.mode == "build" and out.resolve() == DM.cache_dir().resolve():
        raise SystemExit("a smoke build must not write the dev_v5m cache: give --out")
    if a.mode == "build" and out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty: a dev_v5m cache is written once and never over an existing one")
    # the content pin of the context stores (R-7): the real one for a real build, a private one for a smoke build
    pin_d = CX.pin_dir() if out.resolve() == DM.cache_dir().resolve() else Path(str(out) + "_context_store_pin")
    base_d = paths.RUNS_ROOT / "orbit" / "cache" / a.base_version
    if not (base_d / "table.parquet").exists():
        raise SystemExit(f"base cache {a.base_version} does not exist yet")
    if a.mode == "build":           # R-25: >= 60 GB free on / after the projected size; all artefacts < 10 GB
        base_gb = sum(f.stat().st_size for f in base_d.rglob("*") if f.is_file()) / 1e9
        projected_gb = 2.0 * base_gb + 1.0
        free_gb = shutil.disk_usage("/").free / 1e9
        log(f"[devm] disk: {free_gb:.1f} GB free on /, projected size {projected_gb:.1f} GB")
        if projected_gb >= MAX_ARTEFACT_GB or free_gb - projected_gb < MIN_FREE_GB_AFTER:
            raise SystemExit(f"disk rule R-25: projected {projected_gb:.1f} GB (limit {MAX_ARTEFACT_GB} GB), "
                             f"{free_gb:.1f} GB free (needs {MIN_FREE_GB_AFTER} GB after the build); stop and report")
    base, bpix, binfo = D.load_dev(a.base_version)
    if a.limit:
        base = base.iloc[: a.limit].reset_index(drop=True)
        bpix = {k: v[: a.limit] for k, v in bpix.items()}
    log(f"[devm] base {a.base_version}: {len(base)} rows")
    counts = {"mode": a.mode, "base_version": a.base_version, "n_base_rows": int(len(base)),
              "context_rule": CX.RULE_R7, "join_rule": T.RULE_R8, "context_stores": list(DM.CONTEXT_STORES),
              "smoke": smoke, "aquacorr_unit_check": aquacorr_unit_check()}
    if any(counts["aquacorr_unit_check"][t]["unit_stated"] for t in AQUACORR_UNIT_TARGETS):
        raise SystemExit("R-9: AquaCorr's metadata now state a unit for doc_value / acdom440_value; no reviewed "
                         "reader for these values exists in this builder: stop and report to the lead")

    add = pd.DataFrame(columns=base.columns)
    prot = aquacorr_protected_keys()
    if not a.no_added:
        add, wprot, counts["added_rows"] = added_rows(base, [x for x in a.sets.split(",") if x] or None)
        prot = pd.concat([prot, wprot], ignore_index=True)
        log(f"[devm] added-row candidates: {counts['added_rows']}")
    counts["protected_key_rows"] = int(len(prot))                           # count only
    counts["m7_c1"] = c1_key_counts(prot, lst)
    log(f"[devm] C-1: {counts['m7_c1']['matched_and_removed']} of {counts['m7_c1']['before']['all']}")
    # v1's column turbidity_fnu (FNU-class turbidity of the matchup samples, never fitted by v1) keeps its values
    # under the name turbidity_fnu_v1; turbidity_fnu is the arch-v2 §2.4 target read from wqp_discrete
    base = base.rename(columns={"turbidity_fnu": "turbidity_fnu_v1"})
    rows = pd.concat([base.assign(added=False), add.assign(added=True)], ignore_index=True)
    rows["added"] = rows["added"].astype(bool)
    rows["chip_date"] = DM.sensing_date(rows.sensing_time_utc).to_numpy()

    site, sites, counts["join_sites"] = join_sites(rows)
    log(f"[devm] join sites: {counts['join_sites']}; {len(sites)} development sites")
    new, counts["targets"] = join_targets(rows, site, sites, a.mode)
    rows = pd.concat([rows, new], axis=1)
    has_new = np.zeros(len(rows), bool)
    for t in T.SPECS:
        has_new |= rows[t].notna().to_numpy() | rows[f"{t}_cens"].to_numpy(bool)
    keep = ~rows.added.to_numpy() | has_new
    counts["added_rows_kept"] = int((rows.added.to_numpy() & has_new).sum())
    rows = rows[keep].reset_index(drop=True)

    cand = DM.store_candidates(log=log)
    prot_chip_ids = set(cand.chip_id[cand.protected.to_numpy(bool)]) | set(
        DM.full_chip_ids(prot.chip_store, prot.chip_id))
    prot_row_ids = set(prot.row_id)
    # R-21: a chip that a protected key row uses is a protected chip id as well. Such a chip is never a context
    # candidate; an added row on such a chip is dropped; a dev_v5 row on such a chip stops a build (count reported)
    rows, cand, counts["protected_id_overlap"] = drop_protected_ids(rows, cand, prot_row_ids, prot_chip_ids)
    if a.mode == "build" and counts["protected_id_overlap"]["n_base_rows_on_a_protected_id"]:
        raise SystemExit(f"R-21: {counts['protected_id_overlap']['n_base_rows_on_a_protected_id']} dev_v5 rows carry "
                         "a protected row id or the chip of a protected row; stop and report to the lead")
    counts["added_rows_kept"] = int(rows.added.sum())
    by = rows.assign(_c=rows.role + "x" + rows.tblock + np.where(rows.in_lake, "_lake", "_nonlake"))
    counts["rows_by_cell"] = {
        t: by[by[t].notna() | by[f"{t}_cens"]].groupby("_c").size().astype(int).to_dict() for t in T.SPECS}
    log(f"[devm] rows: {len(rows)} ({counts['added_rows_kept']} added)")
    pin_rec = None
    if a.mode == "build" and not (pin_d / CX.PIN_INFO).exists():           # first build: make the pin (R-7)
        pin_rec = CX.write_pin(pin_d, CX.pin_table({st: D._index(st) for st in CX.STORES}),
                               created=time.strftime("%Y-%m-%d %H:%M:%S %Z"), log=log)
        log(f"[devm] context store pin written: {pin_d} ({pin_rec['n_chips']}, {pin_rec['n_shards']} shard files)")
    if (pin_d / CX.PIN_INFO).exists():
        pin_chips, pin_rec = CX.load_pin(pin_d)
        cand, counts["context_store_pin"] = CX.apply_pin(cand, pin_chips)
        counts["context_store_pin"].update(pin_dir=str(pin_d), chips_sha256=pin_rec["chips_sha256"])
    else:                                                                    # keys mode before the first build
        counts["context_store_pin"] = {"note": "no pin yet (it is made at the first build): these context counts "
                                       "use every chip now in the two stores and are provisional"}
    allow = CX.dev_allow_list(cand)
    counts["context_candidates"] = {"n_chips_pinned_stores": int(len(cand)), "n_allow_list": int(len(allow)),
                                    "n_protected_chip_ids": len(prot_chip_ids)}
    cand = cand[cand.chip_id.isin(set(allow))].reset_index(drop=True)
    rctx = rows[["waterbody", "in_lake", "block", "role", "lon", "lat"]].assign(date=rows.chip_date)
    cidx, cdt, counts["context"] = CX.select(rctx, cand[list(CX.CAND_COLS)])
    used = np.unique(cidx[cidx >= 0])
    remap = np.full(len(cand), -1, np.int64)
    remap[used] = np.arange(used.size)
    cidx = np.where(cidx >= 0, remap[np.clip(cidx, 0, None)], -1)
    cmeta = cand.iloc[used].reset_index(drop=True)
    log(f"[devm] context: {counts['context']}")

    if a.counts_json:
        Path(a.counts_json).write_text(json.dumps(counts, indent=1, default=str) + "\n")
    if a.mode == "keys":
        print(json.dumps(counts, indent=1, default=str))
        log("DONE (keys only: no value column and no chip array was read; nothing written to the cache)")
        return counts

    pix = bpix
    n_add = int(rows.added.sum())
    if n_add:
        apix = D.extract_pixels(rows[rows.added].reset_index(drop=True), workers=a.workers, log=log)
        pix = {k: np.concatenate([bpix[k], apix[k]]) for k in bpix}
    feats = row_features(rows, pix, a.workers)
    feats.update(cidx=cidx, cdt=cdt)
    n_sh = CX.check_pinned_shards(pin_rec, zip(cmeta.store, cmeta.shard))    # content equals the pin (R-7)
    log(f"[devm] {n_sh} context shard files equal the pin")
    carrays = context_arrays(cmeta, a.workers)
    info = {"created": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "base_info_n_rows_final": binfo.get("n_rows_final"),
            "build_counts": counts, "amendment": "docs/preregistration/orbit_arch_v2_amendment.md §4",
            "rulings": "docs/preregistration/orbit_arch_v2_lead_rulings.md",
            "definitions_c8": {
                "context_image_qc": "data.image_qc with same_acquisition set true",
                "sen2cor_aot": "aot_c3", "context_row_date": "UTC sensing date of the row's chip",
                "added_row_dedupe": "an added row within 100 m of, and on the same date as, a dev_v5 row is dropped",
                "turbidity_fnu_v1": "v1 column turbidity_fnu carried under this name; not the FNU auxiliary target"}}
    check_no_protected_ids(rows, cmeta, prot_row_ids, prot_chip_ids)        # R-21, before the first file is written
    rec = DM.write(out, rows, pix, feats, info, base=base, cmeta=cmeta[["chip_id", "store", "store_chip_id", "shard",
                   "waterbody", "in_lake", "block", "role", "tblock", "buffer_excluded", "date", "aot",
                   "platform", "lon", "lat"]],
                   carrays=carrays, allow=allow)
    try:
        audit = DM.audit_artefacts(out, prot_row_ids, prot_chip_ids)         # R-21: no protected row or chip id
        DM.load(out, base=base)                                              # the refusing loader accepts it
    except Exception:
        shutil.rmtree(out)           # the folder was empty before this build: a cache that fails the audit is not kept
        log(f"[devm] {out} removed: the written cache failed the artefact audit or the loader")
        raise
    size_gb = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e9
    log(f"[devm] wrote {out}: {rec['counts']}; sha256 {rec['sha256']}; {size_gb:.2f} GB; audit {audit}")
    if size_gb >= MAX_ARTEFACT_GB:
        raise SystemExit(f"disk rule R-25: the cache is {size_gb:.2f} GB (limit {MAX_ARTEFACT_GB} GB); report")
    log("DONE")
    return counts


if __name__ == "__main__":
    main()
