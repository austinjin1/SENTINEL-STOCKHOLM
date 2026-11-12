#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Write the frozen EW-v2 live-trial configuration ``configs/live/ew_v2_live.json`` (once; refuses to overwrite).

Site lists use METADATA only (no data value is read): the frozen split (sites, roles, basin groups), the daily-tier
index of the promoted 2026-11-08 snapshot (``first``/``last`` date per site × variable) and, for H2, the frozen
training-block chloride regressions (eligible flag). Rule (amendment §3):
  * site in ``sonde_split.json`` (any role),
  * the hazard variable has data starting on/before 2020-09-30 (training-block climatology possible) and a last
    value on/after 2026-10-01 (operating in WY2027 at the 2026-11-08 pull),
  * H2 additionally: an eligible frozen SC→Cl regression.
Models are the pre-existing calibration files (no fitting here): H1/H4 = the provisional-input live variant
``ew_live_variant_cal_nosubdaily`` (calibrated 2026-11-08 19:25–19:42, before Test-A), H2 = the frozen EW-v2
calibration (``FREEZE_ew.json`` thresholds_sha256). Methods, Platt maps and alarm thresholds are read from those
files at issue time and verified by SHA-256; nothing is re-tuned.

    python -B scripts/live/build_live_config.py
"""

from __future__ import annotations

import datetime as _dt
import json

import pandas as pd

from live_common import CONFIG, LIVE_BLOCK, REPO, SCHEMA_VERSION, sha256_file, sonde_events_module
from sentinel.core import paths
from sentinel.models.sonde.data import TARGETS

RUNS = paths.RUNS_ROOT / "sonde"
LIVE_VARIANT = RUNS / "ew_live_variant_cal_nosubdaily" / "event_calibration.json"
FROZEN_EW = RUNS / "ew_v2_cand_bag5_wxrunoff" / "event_calibration.json"
WEATHER_VARS_CFG = ["cl_mgl_4dmean", "cl_mgl_max", "no3no2_mgl_mean"]
HAZARD_VAR = {"H1_hypoxia": "do_mgl_min", "H4_nitrate": "no3no2_mgl_mean", "H2_chloride": "spcond_uscm_mean"}


def main() -> int:
    if CONFIG.exists():
        raise SystemExit(f"{CONFIG} exists (frozen). Changes need a dated deviation in the amendment.")
    freeze_ew = json.loads((REPO / "docs/preregistration/FREEZE_ew.json").read_text())
    if sha256_file(FROZEN_EW) != freeze_ew["thresholds_sha256"]:
        raise SystemExit("frozen EW calibration file does not match FREEZE_ew.json")
    split = json.loads((REPO / "docs/preregistration/sonde_split.json").read_text())
    role, basin = split["site_role"], split["basin_group"]
    idx = pd.read_parquet(paths.PROCESSED / "sonde_daily" / "index.parquet")
    idx["site_no"] = idx["site_no"].astype(str)
    regs = pd.read_parquet(paths.PROCESSED / "chloride_derived" / "regressions.parquet")
    regs["site_no"] = regs["site_no"].astype(str)
    eligible = set(regs.loc[regs["eligible"].astype(bool), "site_no"])
    sites = {}
    for hz, var in HAZARD_VAR.items():
        m = idx[(idx["var"] == var) & (pd.to_datetime(idx["first"]) <= "2020-09-30")
                & (pd.to_datetime(idx["last"]) >= LIVE_BLOCK[0]) & idx["site_no"].isin(role)]
        s = sorted(set(m["site_no"]))
        if hz == "H2_chloride":
            s = [x for x in s if x in eligible]
        sites[hz] = [{"site_no": x, "role": role[x], "basin_group": str(basin[x])} for x in s]
    se = sonde_events_module()
    pins = {r: se.snapshot_content_sha256(r) for r in ("sonde_daily",)}
    snap_dir = (paths.PROCESSED / "sonde_daily").resolve()       # physical dir, not the loader symlink
    all_sites = sorted({r["site_no"] for v in sites.values() for r in v})
    site_sha = {s: sha256_file(snap_dir / "sites" / f"{s}.parquet") for s in all_sites
                if (snap_dir / "sites" / f"{s}.parquet").is_file()}
    cfg = {
        "version": "ew_v2_live_v1",
        "schema": SCHEMA_VERSION,
        "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "split_sha256": freeze_ew["split_sha256"],
        "trial": {
            "issue_start": "2026-12-01", "issue_end": "2028-09-30",
            "live_block": list(LIVE_BLOCK),
            "issue_time_utc": "12:00 (cron with CRON_TZ=UTC)", "on_time_deadline_utc": "23:59",
            "data_through": "issue calendar day minus 1 (d = D - 1); forecast windows (d, d + L]",
            "leads": [1, 3, 7], "primary_lead": 3,
            "labels": "Approved values only (provisional = missing), frozen hazards.py rules",
            "looks": {
                "prov_interim": {"earliest": "2027-10-15", "issues": ["2026-12-01", "2027-09-30"],
                                 "content": "operational metrics, realised alarm rates, provisional onset counts; "
                                            "no probabilistic skill metric"},
                "count_only": {"earliest": "2028-06-30", "issues": ["2026-12-01", "2027-09-30"],
                               "approval_cutoff": "2028-06-30", "content": "evaluable H1 onsets + basin groups only"},
                "final_wy2027": {"earliest": "2028-06-30", "issues": ["2026-12-01", "2027-09-30"],
                                 "targets_end": "2027-09-30", "approval_cutoff": "2028-06-30",
                                 "runs_if": "count_only: H1 meets the power rule"},
                "final_pooled": {"earliest": "2029-06-30", "issues": ["2026-12-01", "2028-09-30"],
                                 "targets_end": "2028-09-30", "approval_cutoff": "2029-06-30",
                                 "runs_if": "count_only: H1 underpowered on WY2027"}},
        },
        "models": {
            "H1_hypoxia": {"calibration": str(LIVE_VARIANT), "calibration_sha256": sha256_file(LIVE_VARIANT),
                           "cube_variables": list(TARGETS), "chloride": False,
                           "feature_cfg": {"use_weather": True, "weather_vars": WEATHER_VARS_CFG, "use_subdaily": False},
                           "selected_method": "stack"},
            "H4_nitrate": {"calibration": str(LIVE_VARIANT), "calibration_sha256": sha256_file(LIVE_VARIANT),
                           "cube_variables": list(TARGETS), "chloride": False,
                           "feature_cfg": {"use_weather": True, "weather_vars": WEATHER_VARS_CFG, "use_subdaily": False},
                           "selected_method": "stack"},
            "H2_chloride": {"calibration": str(FROZEN_EW), "calibration_sha256": sha256_file(FROZEN_EW),
                            "cube_variables": list(TARGETS), "chloride": True,
                            "feature_cfg": {"use_weather": True, "weather_vars": WEATHER_VARS_CFG, "use_subdaily": True},
                            "selected_method": freeze_ew["selected_method"]["H2_chloride"]},
        },
        "sites": sites,
        "n_sites": {hz: len(v) for hz, v in sites.items()},
        "n_sites_by_role": {hz: pd.Series([r["role"] for r in v]).value_counts().to_dict() for hz, v in sites.items()},
        "fetch": {"source": "USGS Water Data OGC API, collection 'daily'", "lookback_days": 45,
                  "batch_sites": 20, "weather": "gridMET bulk current-year netCDF, frozen nearest cells "
                                                "(raw/gridmet/site_cells.parquet)"},
        "data_pins": {
            "sonde_daily_content_sha256": pins["sonde_daily"],
            "sonde_daily_dir": str(snap_dir),
            "sonde_daily_site_sha256": site_sha,
            "chloride_regressions_sha256": sha256_file(paths.PROCESSED / "chloride_derived" / "regressions.parquet"),
            "sites_parquet_sha256": sha256_file(paths.PROCESSED / "sites" / "sites.parquet"),
            "gridmet_site_cells_sha256": sha256_file(paths.RAW / "gridmet" / "site_cells.parquet"),
            "split_file_sha256": sha256_file(REPO / "docs/preregistration/sonde_split.json"),
        },
    }
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(json.dumps(cfg, indent=1, sort_keys=False))
    CONFIG.chmod(0o444)
    print(json.dumps({"config": str(CONFIG), "sha256": sha256_file(CONFIG), "n_sites": cfg["n_sites"],
                      "by_role": cfg["n_sites_by_role"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
