#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Build the WQP surface-label side table (depth rule; lead ruling 2026-11-10) and count what the rule changes.

The rule and its causes are in ``sentinel/models/orbit/wqp_surface.py``. This script
1. builds ``processed/orbit_wqp_surface/surface_samples.parquet`` from the processed ``wqp_discrete`` tables with
   the unchanged ``sentinel.data.matchups.wqp_surface_samples`` (depth-blind pass over every file the matchup
   builds read = what they computed; label pass = the documented rule on the harmonised, QC-passed tables), and
   checks that the depth-blind pass reproduces every built ``samples.parquet`` (keys, n_results, values);
2. counts, per base target x matchup table x split role, the samples the rule drops or changes. COUNTS ONLY: no
   label value, label statistic or prediction is printed or written, for any role;
3. writes ``docs/preregistration/orbit_wqp_depth_rule_counts_<date>.json``, the manifest ``orbit_wqp_surface`` and
   (``--register``) the registry rows.

The shared lake-assignment cache is read, never written.

    python -B scripts/train/orbit_build_wqp_surface.py --register
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
from sentinel.core.manifest import write_manifest  # noqa: E402
from sentinel.data.sources import hydrolakes as H  # noqa: E402
from sentinel.models.orbit import data as D  # noqa: E402
from sentinel.models.orbit import split as SP  # noqa: E402
from sentinel.models.orbit import wqp_surface as WS  # noqa: E402


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
        log(f"[counts] HydroLAKES assignment for {todo.size} points (in memory only)")
        a = H.assign_points(lon[sel], lat[sel], lakes, max_dist_m=SP.LAKE_MAX_DIST_M)
        b = H.assign_points(lon[sel], lat[sel], lakes, max_dist_m=SP.BUFFER_M)
        a["near_hylak_id"] = b["hylak_id"].to_numpy()
        a["near_dist_m"] = b["lake_dist_m"].to_numpy()
        a["near_pour_lon"] = b["lake_pour_lon"].to_numpy()
        a["near_pour_lat"] = b["lake_pour_lat"].to_numpy()
        a.insert(0, "point_key", todo)
        cache = pd.concat([cache, a], ignore_index=True) if len(cache) else a
    return cache.set_index("point_key").loc[keys].reset_index()


def role_counts(side: pd.DataFrame) -> dict:
    """Per matchup table: counts by target x role (and x year block) of the base-target samples at JRC-water
    sites, i.e. the rows ``data.wqp_rows`` reads. Sample level, before the same-day scene match and image QC."""
    ref = side.set_index("sample_key")[["surface", "drop_reason", "n_results", "n_results_all", "sample_depth_m",
                                        "value"]]
    out = {}
    for name in D.WQP_MATCHUP_SETS:
        f = paths.PROCESSED / name / "samples.parquet"
        if not f.exists():
            out[name] = "not built yet"
            continue
        s = pd.read_parquet(f, columns=["sample_key", "target", "point_key", "lon", "lat", "date", "value",
                                        "jrc_water_ok"])
        s = s[s.target.isin(D.TARGETS) & s.jrc_water_ok.astype("boolean").fillna(False).astype(bool)]
        s = s.reset_index(drop=True)
        found = s.sample_key.isin(ref.index)
        if not found.all():
            raise WS.SurfaceRuleError(f"{name}: {int((~found).sum())} sample keys missing from the side table")
        r = ref.reindex(s.sample_key)
        pts = s.drop_duplicates("point_key")[["point_key", "lon", "lat"]].reset_index(drop=True)
        lon, lat = pts.lon.to_numpy(float), pts.lat.to_numpy(float)
        sp = SP.assign(lon, lat, np.full(len(pts), np.datetime64("2020-01-01")), lakes=lakes_readonly(lon, lat),
                       log=log)
        s["role"] = s.point_key.map(dict(zip(pts.point_key, sp["role"].to_numpy()))).to_numpy()
        s["tblock"] = SP.time_block(pd.to_datetime(s.date).to_numpy())
        s["surface"] = r["surface"].to_numpy(bool)
        s["no_qc"] = (r["drop_reason"] == WS.DROP_REASONS[0]).to_numpy()
        s["no_surf"] = (r["drop_reason"] == WS.DROP_REASONS[1]).to_numpy()
        s["fewer"] = s.surface & (r["n_results"].to_numpy() < r["n_results_all"].to_numpy())
        s["depth_known"] = r["sample_depth_m"].notna().to_numpy()
        s["had_label"] = np.isfinite(s["value"].to_numpy(float))            # booleans only; no value is reported
        s["has_label"] = s.surface & np.isfinite(r["value"].to_numpy(float))

        def cnt(g):
            return {"n_samples": int(len(g)), "n_dropped": int((~g.surface).sum()),
                    "n_dropped_no_qc_passed_result": int(g.no_qc.sum()),
                    "n_dropped_no_surface_result": int(g.no_surf.sum()),
                    "n_kept_with_fewer_results": int(g.fewer.sum()),
                    "n_kept_depth_known": int((g.surface & g.depth_known).sum()),
                    "n_with_label_before": int(g.had_label.sum()), "n_with_label_after": int(g.has_label.sum())}

        tab = {"n_key_missing": 0, "by_target_role": {}, "by_target_role_tblock": {}, "by_target": {}}
        for t, g in s.groupby("target"):
            tab["by_target"][t] = cnt(g)
            for role, gg in g.groupby("role"):
                tab["by_target_role"][f"{t}/{role}"] = cnt(gg)
                for tb, ggg in gg.groupby("tblock"):
                    tab["by_target_role_tblock"][f"{t}/{role}x{tb}"] = cnt(ggg)
        out[name] = tab
        log(f"[counts] {name}: {json.dumps(tab['by_target'])}")
    return out


def register(counts: dict, meta: dict, rp: Path) -> None:
    """Registry rows (counts only): one summary row with the side-table sha256, the rule text and the whole count
    table, then one row per target x table x role x drop reason."""
    from sentinel.core.registry import log_result

    WS.require_rulings()                # registry rows describe a ruled rule only
    config = {"rule": WS.RULE, "rulings": WS.RULINGS, "split_sha256": SP.split_sha256(),
              "side_table_sha256": meta["side_table_sha256"], "sidecars_left_out": list(WS.SIDECARS)}
    artifacts = [str(rp.relative_to(paths.REPO_ROOT)), "sentinel/models/orbit/wqp_surface.py",
                 "scripts/train/orbit_build_wqp_surface.py"]
    note = ("WQP label rule (lead rulings 2026-11-10): labels from the harmonised, QC-passed wqp_discrete tables "
            "only, surface results only; samples without such a result are dropped. Counts only, no label value "
            "read out. Recorded before the dev_v5 build")
    tabs = {name: tab for name, tab in counts.items() if isinstance(tab, dict)}
    n_all = sum(c["n_samples"] for tab in tabs.values() for c in tab["by_target"].values())
    n_drop = sum(c["n_dropped"] for tab in tabs.values() for c in tab["by_target"].values())
    e = log_result(workstream="M2-orbit", experiment="orbit_wqp_depth_rule", model="counts_depth_rule",
                   metric="n_samples_dropped", value=float(n_drop), n=int(n_all),
                   split="all built WQP matchup tables x base targets x all roles (sample level; counts only)",
                   config=config, artifacts=artifacts, notes=note + ". Summary row: full count table in extra",
                   extra={"side_table_sha256": meta["side_table_sha256"], "rule": WS.RULE,
                          "side_table": {k: meta[k] for k in (
                              "n_samples", "n_no_qc_passed_result", "n_no_surface_result", "n_kept",
                              "n_result_rows_sidecar", "results_flagged_bottom", "reproduces_built_tables")},
                          "counts_by_table_target_role": {name: tab["by_target_role"] for name, tab in tabs.items()},
                          "counts_by_table_target_role_tblock": {name: tab["by_target_role_tblock"]
                                                                 for name, tab in tabs.items()},
                          "tables_not_built_yet": [n for n, t in counts.items() if not isinstance(t, dict)]})
    log(f"[registry] summary row id {e['id']}")
    n_rows = 0
    for name, tab in tabs.items():
        for key, c in tab["by_target_role"].items():
            t, role = key.split("/")
            for metric in ("n_dropped_no_surface_result", "n_dropped_no_qc_passed_result"):
                log_result(workstream="M2-orbit", experiment="orbit_wqp_depth_rule", model="counts_depth_rule",
                           metric=metric.replace("n_dropped", "n_samples_dropped"), value=float(c[metric]),
                           n=c["n_samples"], target=t, split=f"{name} x role {role} (sample level; counts only)",
                           config=config, artifacts=artifacts, notes=note,
                           extra={k: v for k, v in c.items() if k != "n_samples"})
                n_rows += 1
    log(f"[registry] {n_rows} rows by table x target x role x drop reason")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--register", action="store_true")
    ap.add_argument("--register-only", action="store_true",
                    help="write the registry rows for the side table and counts report already on disk (no rebuild)")
    ap.add_argument("--date", default=time.strftime("%Y%m%d"))
    a = ap.parse_args()
    rp = paths.REPO_ROOT / "docs" / "preregistration" / f"orbit_wqp_depth_rule_counts_{a.date}.json"
    if a.register_only:
        WS.require_rulings()
        WS.load_side()                  # sha, rule and input checks
        meta = json.loads((WS.side_dir() / WS.META_FILE).read_text())
        rep = json.loads(rp.read_text())
        if rep["side_table"]["side_table_sha256"] != meta["side_table_sha256"] or rep["rule"] != WS.RULE:
            raise WS.SurfaceRuleError(f"{rp.name} does not describe the side table on disk: rebuild")
        register(rep["tables"], meta, rp)
        log("DONE")
        return 0
    meta = WS.build(log)
    side = pd.read_parquet(WS.side_dir() / WS.SIDE_FILE)
    WS.check_depth_read(side[side.surface], "side table")
    counts = role_counts(side)
    rep = {"created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "rule": WS.RULE,
           "scope": "base-target samples at JRC-water sites of each built matchup table (sample level; before the "
                    "same-day scene match, chips and image QC). COUNTS ONLY: no label value was printed or written",
           "rulings": WS.RULINGS,
           "split_sha256": SP.split_sha256(), "side_table": {k: meta[k] for k in (
               "n_samples", "n_no_qc_passed_result", "n_no_surface_result", "n_kept", "n_kept_with_fewer_results",
               "n_kept_non_secchi", "n_kept_non_secchi_depth_known", "by_target", "sidecars_left_out",
               "result_rows_by_file", "n_result_rows_sidecar", "results_flagged_bottom", "reproduces_built_tables",
               "side_table_sha256", "matchups_py_sha256")},
           "input_sha256": meta["input_sha256"], "tables": counts}
    rp.write_text(json.dumps(rep, indent=1) + "\n")
    d = WS.side_dir()
    write_manifest(WS.SIDE_NAME, files=[d / WS.SIDE_FILE, d / WS.META_FILE], sources=["manifest:wqp_discrete"],
                   description="WQP surface-sample labels with the depth rule applied (depth_m <= 1 m or unknown, "
                               "not flagged bottom; Secchi exempt; harmonised QC-passed tables only), keyed by the "
                               "Sentinel-2 matchup sample_key; side table for "
                               "sentinel/models/orbit/data.py::wqp_rows",
                   license="derived: WQP public", synthetic=False,
                   stats=json.loads(json.dumps(rep["side_table"])))
    if a.register:
        register(counts, meta, rp)
    log(json.dumps({"report": str(rp), "side_table_sha256": meta["side_table_sha256"]}, indent=1))
    log("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
