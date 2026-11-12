#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Keys-only count of M2 Test-S (`testxarchive_lake`) site-dates. 2026-11-10.

Reads ONLY key / imagery-metadata columns (site, lat, lon, date, chip QC, item id, target NAME). No label value,
no label statistic and no prediction is read. Output: counts (rows, water bodies, role blocks). Does not write the
shared lake-assignment cache.

Pre-registration support for docs/preregistration/orbit_arch_v2_amendment.md (arch-v2 §1.4, §1.6, §2.6). The WQP
2015-2023 part is an UPPER BOUND (sample keys before the same-day scene match, chips and image QC).

    python -B scripts/data/orbitv2_testS_keycounts.py docs/preregistration/orbit_testS_keycounts_20261110.json
"""
import os, sys, json, re, hashlib
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[k] = "4"
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from sentinel.core import paths
from sentinel.models.orbit import split as SP, data as D, features as F
from sentinel.data.sources import hydrolakes as H

T0, T1 = pd.Timestamp("2015-06-23"), pd.Timestamp("2023-12-31")
OUT = sys.argv[1]
log = lambda *a: print(*a, flush=True)


def lakes_readonly(lon, lat):
    """SP.assign_lakes without writing the shared cache (chain6b owns it)."""
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    keys = SP.point_key(lon, lat)
    cp = SP._lake_cache_path()
    cache = pd.read_parquet(cp) if cp.exists() else pd.DataFrame(columns=["point_key"])
    uk, first = np.unique(keys, return_index=True)
    todo = np.setdiff1d(uk, cache["point_key"].to_numpy())
    if todo.size:
        sel = first[np.searchsorted(uk, todo)]
        lakes = SP._load_lakes()
        log(f"[keys] HydroLAKES assignment for {todo.size} points (in memory only)")
        a = H.assign_points(lon[sel], lat[sel], lakes, max_dist_m=SP.LAKE_MAX_DIST_M)
        b = H.assign_points(lon[sel], lat[sel], lakes, max_dist_m=SP.BUFFER_M)
        a["near_hylak_id"] = b["hylak_id"].to_numpy()
        a["near_dist_m"] = b["lake_dist_m"].to_numpy()
        a["near_pour_lon"] = b["lake_pour_lon"].to_numpy()
        a["near_pour_lat"] = b["lake_pour_lat"].to_numpy()
        a.insert(0, "point_key", todo)
        cache = pd.concat([cache, a], ignore_index=True) if len(cache) else a
    return cache.set_index("point_key").loc[keys].reset_index()


def assign(lon, lat, dates):
    return SP.assign(lon, lat, dates, lakes=lakes_readonly(lon, lat), log=log)


def cnt(g, key="site_date"):
    return {"n_site_dates": int(g[key].nunique()), "n_rows": int(len(g)), "n_waterbodies": int(g.waterbody.nunique()),
            "n_blocks": int(g.block.nunique())}


def scene_key(item_id, sensing):
    tile = pd.Series(item_id).astype(str).str.extract(r"(?:_T|_)(\d{2}[C-X][A-Z]{2})(?:_|$)")[0]
    day = pd.to_datetime(pd.Series(sensing), utc=True, errors="coerce").dt.strftime("%Y%m%d")
    return (tile.fillna("NA") + "|" + day.fillna("NA")).to_numpy()


res = {"created": "2026-11-10", "definition": "testxarchive_lake = test role (split v3) x in-lake x not buffer x "
       "sample date 2015-06-23..2023-12-31; keys only (no label value read)", "split_sha256": SP.split_sha256()}

# ---------------------------------------------------------------- AquaCorr (chips extracted; image QC; de-dup 20 m)
proc = paths.processed("aquacorr_global")
sel = pd.read_parquet(proc / "s2_chip_requests.parquet",
                      columns=["matchup_id", "site_id", "source", "lat", "lon", "date_insitu", "sensor",
                               "chip_request_key"])
plan = pd.read_parquet(proc / "s2_chip_plan.parquet", columns=["chip_request_key", "match_kind"])
idx = D._index("aquacorr").rename(columns={"request_key": "chip_request_key"})
df = sel.merge(plan, on="chip_request_key", how="left").merge(
    idx[["chip_request_key"] + [c for c in D.IMG_COLS if c in idx.columns]], on="chip_request_key", how="inner")
st = pd.to_datetime(df.sensing_time_utc, utc=True, errors="coerce")
same_acq = (df.platform == df.sensor) & (st.dt.strftime("%Y-%m-%d") == pd.to_datetime(df.date_insitu).dt.strftime(
    "%Y-%m-%d")) & df.match_kind.isin(["designated", "alternate_tile"])
a = pd.DataFrame({"source": "aquacorr", "row_src_id": df.matchup_id.astype(str), "site_id": df.site_id.astype(str),
                  "lat": df.lat.astype(float), "lon": df.lon.astype(float),
                  "date": pd.to_datetime(df.date_insitu).dt.normalize(), "same_acquisition": same_acq.to_numpy()})
for c in D.IMG_COLS:
    if c in df:
        a[c] = df[c].to_numpy()
for t in D.TARGETS:
    a[t] = np.nan                       # no label value is read; dedupe only needs the columns to exist
a = a[D.image_qc(a).to_numpy()].reset_index(drop=True)
a = D.dedupe(a, log)
sp = assign(a.lon.to_numpy(), a.lat.to_numpy(), a.date.to_numpy())
a = pd.concat([a.reset_index(drop=True), sp], axis=1)
a["site_date"] = SP.point_key(a.lon, a.lat) + "|" + a.date.dt.strftime("%Y-%m-%d")
a["scene"] = scene_key(a.item_id, a.sensing_time_utc)
arch = (a.date >= T0) & (a.date <= T1)
ts = a[(a.role == "test") & a.in_lake & ~a.buffer_excluded.astype(bool) & arch]
res["aquacorr_qc_chip_rows"] = {
    "note": "chip-bearing, image-QC-passing, de-duplicated AquaCorr rows (any target; values not read). Upper bound "
            "for each target's labelled rows; per-target counts are in orbit_split_report_dev_v4.json",
    "all": cnt(ts), "train_years_le2021": cnt(ts[ts.tblock == "train"]), "val_years_2022_23": cnt(ts[ts.tblock == "val"]),
    "small_lake_lt1km2": cnt(ts[ts.lake_area_km2 < 1.0]),
    "by_year": {int(y): int(n) for y, n in ts.date.dt.year.value_counts().sort_index().items()}}
# scene-disjointness preview (keys only): scenes of train-role x train-year rows (fit + inner), any water type
fit = a[(a.role == "train") & (a.tblock == "train") & ~a.buffer_excluded.astype(bool)]
val = a[(a.role == "val") & ~a.protected.astype(bool) & ~a.buffer_excluded.astype(bool)]
fs, vs = set(fit.scene), set(val.scene)
dj = ts[~ts.scene.isin(fs)]
dj2 = ts[~ts.scene.isin(fs | vs)]
res["aquacorr_scene_disjoint_preview"] = {
    "note": "scene = (MGRS tile, sensing date). Reference = AquaCorr fit+inner rows only (WQP 2015-2023 rows not built "
            "yet); the binding count is made on dev_v5m labelled rows per target",
    "n_fit_inner_scenes": len(fs), "all_rows": cnt(ts), "disjoint_from_fit_inner": cnt(dj),
    "frac_rows": round(len(dj) / max(len(ts), 1), 4), "frac_blocks": round(dj.block.nunique() / max(ts.block.nunique(), 1), 4),
    "disjoint_from_fit_inner_and_val": cnt(dj2),
    "frac_rows_incl_val": round(len(dj2) / max(len(ts), 1), 4),
    "frac_blocks_incl_val": round(dj2.block.nunique() / max(ts.block.nunique(), 1), 4)}
json.dump(res, open(OUT, "w"), indent=1)
log(json.dumps(res, indent=1))

# ---------------------------------------------------------------- WQP 2015-2023 (sample keys; chips still being built)
d = paths.PROCESSED / "s2_wqp_matchups_hist"
smp = pd.read_parquet(d / "samples.parquet", columns=["site_id", "point_key", "date", "target", "lat", "lon",
                                                      "jrc_water_ok"])
smp = smp[smp.jrc_water_ok.astype("boolean").fillna(False)]
smp["date"] = pd.to_datetime(smp.date).dt.normalize()
base = smp[smp.target.isin(D.TARGETS)]
pts = smp.drop_duplicates("point_key")[["point_key", "lat", "lon"]].reset_index(drop=True)
lk = lakes_readonly(pts.lon.to_numpy(float), pts.lat.to_numpy(float))
sp = SP.assign(pts.lon.to_numpy(float), pts.lat.to_numpy(float), np.full(len(pts), np.datetime64("2020-01-01")), lakes=lk, log=log)
pts = pd.concat([pts, sp[["waterbody", "in_lake", "block", "role", "buffer_excluded", "lake_area_km2"]]], axis=1)
tsp = pts[(pts.role == "test") & pts.in_lake & ~pts.buffer_excluded.astype(bool)]
w = base.merge(tsp, on="point_key", how="inner")
w = w[(w.date >= T0) & (w.date <= T1)]
w["site_date"] = w.point_key + "|" + w.date.dt.strftime("%Y-%m-%d")
res["wqp_hist_sample_keys"] = {
    "note": "UPPER BOUND before chip extraction, same-day window and image QC: WQP sample site-dates (JRC water ok) of "
            "the four base targets in Test-S lakes. Chl-a QC-pass was ~58% in the 2024-25 build",
    "all": cnt(w), "by_target": {t: cnt(g) for t, g in w.groupby("target")},
    "small_lake_lt1km2": cnt(w[w.lake_area_km2 < 1.0]),
    "train_years_le2021": cnt(w[w.date.dt.year <= 2021]), "val_years_2022_23": cnt(w[w.date.dt.year >= 2022])}
json.dump(res, open(OUT, "w"), indent=1)
# Track N co-location preview (keys only): same WQP site + same local date
allw = smp.merge(tsp, on="point_key", how="inner")
allw = allw[(allw.date >= T0) & (allw.date <= T1)]
allw["site_date"] = allw.point_key + "|" + allw.date.dt.strftime("%Y-%m-%d")
sd_any = allw.drop_duplicates("site_date")[["site_id", "date", "site_date", "waterbody", "block"]]
sd_chl = w[w.target == "chla"].drop_duplicates("site_date")[["site_date"]].assign(has_chla=True)
out = {}
import pyarrow.parquet as pq
for v in ("tp", "tn", "doc", "turbidity_fnu", "ssc"):
    f = paths.PROCESSED / "wqp_discrete" / f"variable={v}" / "data.parquet"
    cols = ["site_id", "activity_start_date"] + (["fraction_class"] if v in ("tp", "tn") else [])
    k = pq.read_table(f, columns=cols, use_threads=False).to_pandas()
    if v in ("tp", "tn"):
        k = k[k.fraction_class == "total"]
    k["date"] = pd.to_datetime(k.activity_start_date, errors="coerce").dt.normalize()
    k = k.dropna(subset=["date"]).drop_duplicates(["site_id", "date"])[["site_id", "date"]]
    j = sd_any.merge(k, on=["site_id", "date"], how="inner").merge(sd_chl, on="site_date", how="left")
    out[v] = {"at_any_matchup_site_date": cnt(j), "at_chla_site_dates": cnt(j[j.has_chla.fillna(False).astype(bool)])}
    del k
res["trackN_colocation_wqp_hist_keys"] = {
    "note": "UPPER BOUND (pre-chip, pre-QC): Test-S WQP matchup-sample site-dates that also have a same-site same-date "
            "WQP sample of the new target (TP/TN: fraction_class total). Keys only", **out}
json.dump(res, open(OUT, "w"), indent=1)
log(json.dumps(res, indent=1))
log("DONE")
