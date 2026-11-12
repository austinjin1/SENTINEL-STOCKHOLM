#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""WQP label rule (orbit_protocol.md change log, 2026-11-10): recount of the protected cells. COUNTS ONLY.

A. dev_v4 composition (AquaCorr chips + the 2024-2025 WQP matchup table): ``data.protected_cell_counts`` (rows
   with a value inside the valid range, water bodies and role blocks per target)
   - under the labels as stored in the matchup table (reference; its WQP counts must reproduce
     ``orbit_split_report_dev_v4.json``), and
   - under the rule (``sentinel/models/orbit/wqp_surface.py``).
   The headline cell is ``testxtest_lake``.
B. Test-S (``testxarchive_lake``), WQP part: 2015-2023 sample keys of in-lake, non-buffer, test-role sites. This is
   an UPPER BOUND (before the same-day scene match, chips and image QC), before and after the rule.

No label value, label statistic or prediction is printed or written: every output is a count of rows, site-dates,
water bodies or role blocks. The shared lake-assignment cache is read, never written. The binding counts are made
by the dev_v5 build (``orbit_split_report_dev_v5.json``) and, for Test-S, by the dev_v5m build.

    python -B scripts/data/orbit_wqp_rule_cellcounts.py --register
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "4")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths  # noqa: E402
from sentinel.data.sources import hydrolakes as H  # noqa: E402
from sentinel.models.orbit import data as D  # noqa: E402
from sentinel.models.orbit import split as SP  # noqa: E402
from sentinel.models.orbit import wqp_surface as WS  # noqa: E402

T0, T1 = pd.Timestamp("2015-06-23"), pd.Timestamp("2023-12-31")
HEADLINE = "testxtest_lake"


def log(*a):
    print(*a, flush=True)


def lakes_readonly(lon, lat) -> pd.DataFrame:
    """``SP.assign_lakes`` without writing the shared cache (new points are assigned in memory only)."""
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    keys = SP.point_key(lon, lat)
    cp = SP._lake_cache_path()
    cache = pd.read_parquet(cp) if cp.exists() else pd.DataFrame(columns=["point_key"])
    uk, first = np.unique(keys, return_index=True)
    todo = np.setdiff1d(uk, cache["point_key"].to_numpy())
    if todo.size:
        sel = first[np.searchsorted(uk, todo)]
        lakes = SP._load_lakes()
        log(f"[cells] HydroLAKES assignment for {todo.size} points (in memory only)")
        a = H.assign_points(lon[sel], lat[sel], lakes, max_dist_m=SP.LAKE_MAX_DIST_M)
        b = H.assign_points(lon[sel], lat[sel], lakes, max_dist_m=SP.BUFFER_M)
        a["near_hylak_id"] = b["hylak_id"].to_numpy()
        a["near_dist_m"] = b["lake_dist_m"].to_numpy()
        a["near_pour_lon"] = b["lake_pour_lon"].to_numpy()
        a["near_pour_lat"] = b["lake_pour_lat"].to_numpy()
        a.insert(0, "point_key", todo)
        cache = pd.concat([cache, a], ignore_index=True) if len(cache) else a
    return cache.set_index("point_key").loc[keys].reset_index()


def cells_dev_v4_composition(aq: pd.DataFrame, rule: bool) -> dict:
    """``build_table`` up to ``protected_cell_counts`` for AquaCorr + the 2024-2025 WQP table."""
    sets, apply_ = D.WQP_MATCHUP_SETS, WS.apply
    D.WQP_MATCHUP_SETS = ("s2_wqp_matchups",)
    if not rule:
        WS.apply = lambda smp, log=print, side=None: smp        # labels as stored (reference pass)
    try:
        w = D.wqp_rows(log)
    finally:
        D.WQP_MATCHUP_SETS, WS.apply = sets, apply_
    raw = pd.concat([aq, w], ignore_index=True)
    qc = D.image_qc(raw)
    df = D.dedupe(raw[qc.to_numpy()].reset_index(drop=True), log)
    lon, lat = df.lon.to_numpy(float), df.lat.to_numpy(float)
    sp = SP.assign(lon, lat, df.date.to_numpy(), lakes=lakes_readonly(lon, lat), log=log)
    df = pd.concat([df.reset_index(drop=True), sp], axis=1)
    return {"n_rows_with_chip": int(len(raw)), "by_source": {k: int(v) for k, v in raw.source.value_counts().items()},
            "image_qc_pass": int(qc.sum()), "n_rows_after_dedupe": int(len(df)),
            "protected_cell_counts": D.protected_cell_counts(df)}


def cnt(g: pd.DataFrame) -> dict:
    return {"n_site_dates": int(g.site_date.nunique()), "n_waterbodies": int(g.waterbody.nunique()),
            "n_blocks": int(g.block.nunique())}


def test_s_wqp_keys() -> dict:
    name = "s2_wqp_matchups_hist"
    s = pd.read_parquet(paths.PROCESSED / name / "samples.parquet",
                        columns=["sample_key", "point_key", "date", "target", "unit", "lat", "lon", "value",
                                 "jrc_water_ok"])
    s = s[s.target.isin(D.TARGETS) & s.jrc_water_ok.astype("boolean").fillna(False).astype(bool)]
    s = s.reset_index(drop=True)
    s["date"] = pd.to_datetime(s.date).dt.normalize()
    pts = s.drop_duplicates("point_key")[["point_key", "lat", "lon"]].reset_index(drop=True)
    lon, lat = pts.lon.to_numpy(float), pts.lat.to_numpy(float)
    sp = SP.assign(lon, lat, np.full(len(pts), np.datetime64("2020-01-01")), lakes=lakes_readonly(lon, lat), log=log)
    pts = pd.concat([pts, sp[["waterbody", "in_lake", "block", "role", "buffer_excluded", "lake_area_km2"]]], axis=1)
    tsp = pts[(pts.role == "test") & pts.in_lake & ~pts.buffer_excluded.astype(bool)]
    w = s.merge(tsp.drop(columns=["lat", "lon"]), on="point_key", how="inner")
    w = w[(w.date >= T0) & (w.date <= T1)].reset_index(drop=True)
    w["site_date"] = w.point_key + "|" + w.date.dt.strftime("%Y-%m-%d")
    side = WS.load_side().set_index("sample_key")
    if not w.sample_key.isin(side.index).all():
        raise WS.SurfaceRuleError("Test-S sample keys missing from the side table")
    r = side.reindex(w.sample_key)
    unit_ok = np.array([u in D.WQP_UNIT_FACTORS.get(t, {}) for t, u in zip(w.target, w.unit)])
    # booleans only: "has a usable label" before (stored value) and after (rule value); no value is reported
    w["before"] = np.isfinite(w.value.to_numpy(float)) & unit_ok
    w["after"] = r["surface"].to_numpy(bool) & np.isfinite(r["value"].to_numpy(float)) & unit_ok
    w["no_qc"] = (r["drop_reason"] == WS.DROP_REASONS[0]).to_numpy()
    w["no_surf"] = (r["drop_reason"] == WS.DROP_REASONS[1]).to_numpy()
    w["yr"] = np.where(w.date.dt.year <= 2021, "train_years_le2021", "val_years_2022_23")
    out = {"note": "UPPER BOUND before the same-day scene match, chips and image QC: WQP sample site-dates (JRC water "
                   "ok, M2-accepted unit, value present) of the four base targets at in-lake, non-buffer, test-role "
                   "sites, 2015-06-23..2023-12-31. Keys and booleans only",
           "all_keys_any_unit": cnt(w), "by_target": {}}
    for t, g in w.groupby("target"):
        rec = {"before": cnt(g[g.before]), "after": cnt(g[g.after]),
               "n_samples_dropped_no_qc_passed_result": int((g.before & g.no_qc).sum()),
               "n_samples_dropped_no_surface_result": int((g.before & g.no_surf).sum()),
               "small_lake_lt1km2": {"before": cnt(g[g.before & (g.lake_area_km2 < 1.0)]),
                                     "after": cnt(g[g.after & (g.lake_area_km2 < 1.0)])}}
        for y, gg in g.groupby("yr"):
            rec[y] = {"before": cnt(gg[gg.before]), "after": cnt(gg[gg.after])}
        out["by_target"][t] = rec
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--register", action="store_true")
    ap.add_argument("--date", default=time.strftime("%Y%m%d"))
    a = ap.parse_args()
    WS.require_rulings()
    meta = json.loads((WS.side_dir() / WS.META_FILE).read_text())
    aq = D.aquacorr_rows(log)
    before = cells_dev_v4_composition(aq, rule=False)
    after = cells_dev_v4_composition(aq, rule=True)
    del aq
    ref = json.loads((paths.REPO_ROOT / "docs" / "preregistration" / "orbit_split_report_dev_v4.json").read_text())
    ref = ref["protected_cell_counts"]
    # the reference pass must reproduce the WQP counts of the dev_v4 split report (AquaCorr chips may have grown)
    mism = {}
    for cell, tg in ref.items():
        for t, c in tg.items():
            got = before["protected_cell_counts"].get(cell, {}).get(t, {}).get("by_source", {}).get("wqp", 0)
            if got != c.get("by_source", {}).get("wqp", 0):
                mism[f"{cell}/{t}"] = [c.get("by_source", {}).get("wqp", 0), got]
    rep = {"created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "rule": WS.RULE,
           "side_table_sha256": meta["side_table_sha256"], "split_sha256": SP.split_sha256(),
           "scope": "COUNTS ONLY (rows, site-dates, water bodies, role blocks). No label value was printed or "
                    "written. Binding counts: orbit_split_report_dev_v5.json (dev_v5 build) and the dev_v5m build",
           "dev_v4_composition": {"note": "AquaCorr chips + the 2024-2025 WQP table (s2_wqp_matchups); "
                                          "protected_cell_counts as in data.build_table",
                                  "labels_as_stored": before, "under_the_rule": after,
                                  "reference_reproduces_dev_v4_report_wqp_counts": not mism,
                                  "reference_mismatches": mism},
           "headline_cell": {t: {"before": {k: before["protected_cell_counts"][HEADLINE][t][k]
                                            for k in ("n", "n_waterbodies", "n_blocks")},
                                 "after": {k: after["protected_cell_counts"][HEADLINE][t][k]
                                           for k in ("n", "n_waterbodies", "n_blocks")}} for t in D.TARGETS},
           "test_s_wqp_hist_sample_keys": test_s_wqp_keys()}
    rp = paths.REPO_ROOT / "docs" / "preregistration" / f"orbit_wqp_rule_cellcounts_{a.date}.json"
    rp.write_text(json.dumps(rep, indent=1) + "\n")
    log(json.dumps({"headline_cell": rep["headline_cell"], "reference_mismatches": mism,
                    "test_s": rep["test_s_wqp_hist_sample_keys"]["by_target"]}, indent=1))
    if a.register:
        from sentinel.core.registry import log_result

        for t in D.TARGETS:
            h = rep["headline_cell"][t]
            e = log_result(workstream="M2-orbit", experiment="orbit_wqp_depth_rule", model="counts_protected_cells",
                           metric="n_rows_headline_cell_under_rule", value=float(h["after"]["n"]),
                           n=int(h["before"]["n"]), target=t,
                           split=f"{HEADLINE} (dev_v4 composition: AquaCorr + WQP 2024-2025; counts only)",
                           config={"rule": WS.RULE, "side_table_sha256": meta["side_table_sha256"],
                                   "split_sha256": SP.split_sha256()},
                           artifacts=[str(rp.relative_to(paths.REPO_ROOT)), "scripts/data/orbit_wqp_rule_cellcounts.py"],
                           notes="Protected cell recount under the WQP label rule (lead rulings 2026-11-10). Counts "
                                 "only; n = rows before the rule. Design quantity, superseded by the dev_v5 split "
                                 "report",
                           extra={**h, "test_s_wqp_hist_sample_keys": rep["test_s_wqp_hist_sample_keys"]["by_target"][t]
                                  if t in rep["test_s_wqp_hist_sample_keys"]["by_target"] else None})
            log(f"[registry] {t}: row id {e['id']}")
    log(json.dumps({"report": str(rp)}))
    log("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
